#include <math.h>
#include <stdio.h>

#include "roipatch_dfvs_pool3d_gpu.h"
#include "cuda_utils.h"

__global__ void repeat_pooled_pts_idx(int boxes_num, int sampled_pts_num, int *pooled_pts_idx, const int *pooled_pts_num){
    // params pooled_pts_num: (B, M)
    // params pooled_pts_idx: (B, M, 512)

    int sample_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (sample_pt_idx >= sampled_pts_num) return;

    int cnt = pooled_pts_num[bs_idx * boxes_num + box_idx];
    if (cnt == 0) {
        return;
    }
    else if (cnt <= sample_pt_idx) {
        // duplicate same points for sampling
        int duplicate_idx = sample_pt_idx % cnt;
        int base_offset = bs_idx * boxes_num * sampled_pts_num + box_idx * sampled_pts_num;
        pooled_pts_idx[base_offset + sample_pt_idx] = pooled_pts_idx[base_offset + duplicate_idx];
    }
}

__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}

__device__ inline int simple_hash(int k, int hash_size) {
    return k % hash_size;
}

__global__ void assign_pts_to_box3d_part_local_stack_v2(int pts_num, int dvs_sampled_pts_num, int num_boxes_per_patch, 
    int hash_size, float lambda, float delta, const float *xyz, const float *boxes3d, const int *point2patch_indices, const int *patch2box_indices, 
    int *hash_table, int *pooled_pts_num, int *pooled_pts_idx) {
    // params xyz: (N1 + N2 + ..., 3)
    // params boxes3d: (B, M, 7)
    // params point2patch_indices: (N1 + N2 + ..., )
    // params patch2box_indices: (K1 + K2 + ..., num_boxes_per_patch + 1) [cnt, stacked_idx1, ...]
    // params hash_table: (B, M, hash_size)
    // params pooled_pts_idx: (B, M, 512)
    // params pooled_pts_num: (B, M)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_per_pat_idx = blockIdx.y;

    if (pt_idx >= pts_num) {
        return;
    }
    int patch2box_idx = point2patch_indices[pt_idx] * (num_boxes_per_patch + 1);
    int num_valid_box_per_patch = patch2box_indices[patch2box_idx];
    if (box_per_pat_idx >= num_valid_box_per_patch) {
        return;
    }

    int box_idx = patch2box_indices[patch2box_idx + 1 + box_per_pat_idx];

    float z = xyz[pt_idx * 3 + 2], cz = boxes3d[box_idx * 7 + 2], dz = boxes3d[box_idx * 7 + 5];
    if (fabsf(z - cz) > dz / 2.0)
        return;
    const float MARGIN = 1e-5;
    float local_x = 0, local_y = 0;
    float x = xyz[pt_idx * 3 + 0], y = xyz[pt_idx * 3 + 1];
    float cx = boxes3d[box_idx * 7 + 0], cy = boxes3d[box_idx * 7 + 1];
    float dx = boxes3d[box_idx * 7 + 3], dy = boxes3d[box_idx * 7 + 4], rz = boxes3d[box_idx * 7 + 6];
    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    int cur_in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    if (cur_in_flag) {
        float local_z = z - cz;
        float res = max(0.02, lambda * exp(-sqrt(cx * cx + cy * cy + cz * cz) / delta));
        int out_x = int(dx / res);
        int out_y = int(dy / res);
        int out_z = int(dz / res);
        int x_idx = int((local_x + dx / 2) / res);
        int y_idx = int((local_y + dy / 2) / res);
        int z_idx = int((local_z + dz / 2) / res);
        x_idx = min(max(x_idx, 0), out_x - 1);
        y_idx = min(max(y_idx, 0), out_y - 1);
        z_idx = min(max(z_idx, 0), out_z - 1);
        int key = x_idx * out_y * out_z + y_idx * out_z + z_idx;
        int hash_idx = simple_hash(key, hash_size);
        int prob_cnt = 1;
        while(true) {
            int cur_hash_idx = (hash_idx + ((prob_cnt & 1) ? -1 : 1) * (prob_cnt / 2) * (prob_cnt / 2)) % hash_size;
            cur_hash_idx = (cur_hash_idx + hash_size) % hash_size;
            int prev_key = atomicCAS(hash_table + box_idx * hash_size + cur_hash_idx, EMPTY_KEY, key); // insert key when empty
            if (prev_key == EMPTY_KEY) {
                int assign_pts_num = atomicAdd(pooled_pts_num + box_idx, 1);
                if (assign_pts_num < dvs_sampled_pts_num) {
                    pooled_pts_idx[box_idx * dvs_sampled_pts_num + assign_pts_num] = pt_idx;
                }
                break;
            }
            else if (prev_key == key)
                break;
            // hash_idx = (hash_idx + 1) % hash_size;
            prob_cnt += 1;
            if (prob_cnt > hash_size || prob_cnt > dvs_sampled_pts_num) {
                break;
            }
        }
    }
}

__device__ void __update(float *dists, int *dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void roiwise_fps_stack(int boxes_num, int sampled_pts_num, int fps_pts_num,
                                   const float *xyz, const int *pts_assign, const int *pooled_pts_num, float *temp, int *pooled_pts_idx){
    // params xyz: (N1 + N2 + ..., 3)
    // params temp: (B, M, 8192)
    // params pts_assign: (B, M, 8192)
    // params pooled_pts_num: (B, M)
    // params pooled_pts_idx: (B, M, 512)
    int box_idx = blockIdx.x;
    int bs_idx = blockIdx.y;
    int stack_box_idx = bs_idx * boxes_num + box_idx;
    int valid_sampled_pts_num = pooled_pts_num[stack_box_idx];
    if (valid_sampled_pts_num == 0) return;

    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = pts_assign[stack_box_idx * sampled_pts_num + 0];
    if (tid == 0)
        pooled_pts_idx[stack_box_idx * fps_pts_num + 0] = old;

    for (int j = 1; j < fps_pts_num && j < valid_sampled_pts_num; j++) {
        int besti = 0;
        float best = -1;
        float x1 = xyz[old * 3 + 0];
        float y1 = xyz[old * 3 + 1];
        float z1 = xyz[old * 3 + 2];
        for (int k = tid; k < sampled_pts_num && k < valid_sampled_pts_num; k += stride) {
            int assign_idx = stack_box_idx * sampled_pts_num + k;
            int k_ = pts_assign[assign_idx];
            float x2 = xyz[k_ * 3 + 0];
            float y2 = xyz[k_ * 3 + 1];
            float z2 = xyz[k_ * 3 + 2];
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = temp[assign_idx];  // NO READ-WRITE CONFLICT
            if (j == 1 || d < d2) {
                temp[assign_idx] = d;
                d2 = d;
            }
            besti = d2 > best ? k_ : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
                __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }
        if (block_size >= 512) {
            if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            pooled_pts_idx[stack_box_idx * fps_pts_num + j] = old;
    }
}

void roilocal_dfvs_pool3d_kernel_launcher_stack_v2(int batch_size, int pts_num, int boxes_num, int dvs_pts_num, int fps_pts_num, 
    int num_boxes_per_patch, int hash_size, float lambda, float delta, const float *xyz, const float *boxes3d,
    const int *point2patch_indices, const int *patch2box_indices, int *pooled_pts_num, int *pooled_pts_idx) {
    
    int *hash_table = NULL;
    cudaMalloc(&hash_table, batch_size * boxes_num * hash_size * sizeof(int));
    cudaMemset(hash_table, EMPTY_KEY, batch_size * boxes_num * hash_size * sizeof(int));
    
    int *pts_assign = NULL;
    cudaMalloc(&pts_assign, batch_size * boxes_num * dvs_pts_num * sizeof(int));

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), num_boxes_per_patch);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_pts_to_box3d_part_local_stack_v2<<<blocks, threads>>>(pts_num, dvs_pts_num,
        num_boxes_per_patch, hash_size, lambda, delta, xyz, boxes3d, point2patch_indices, patch2box_indices, 
        hash_table, pooled_pts_num, pts_assign);
    
    float *temp = NULL;
    cudaMalloc(&temp, batch_size * boxes_num * dvs_pts_num * sizeof(float));
    
    dim3 blocks2(boxes_num, batch_size);
    roiwise_fps_stack<THREADS_PER_BLOCK><<<blocks2, threads>>>(boxes_num, dvs_pts_num, fps_pts_num, xyz, pts_assign, pooled_pts_num, temp, pooled_pts_idx);

    dim3 blocks4(DIVUP(fps_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
    repeat_pooled_pts_idx<<<blocks4, threads>>>(boxes_num, fps_pts_num, pooled_pts_idx, pooled_pts_num);
    
    cudaFree(temp);
    cudaFree(pts_assign);
    cudaFree(hash_table);
}
