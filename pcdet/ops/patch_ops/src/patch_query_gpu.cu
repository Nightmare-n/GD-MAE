#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

#include "patch_query_gpu.h"
#include "cuda_utils.h"

struct Point {
    float x, y;
    __device__ Point() {}
    __device__ Point(float _x, float _y){
        x = _x, y = _y;
    }

    __device__ void set(float _x, float _y){
        x = _x; y = _y;
    }

    __device__ Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline void get_AA_box(const float *box, float *AA_box) {
    float angle = box[6], dx_half = box[3] / 2, dy_half = box[4] / 2;
    float x1 = box[0] - dx_half, y1 = box[1] - dy_half;
    float x2 = box[0] + dx_half, y2 = box[1] + dy_half;

    Point center(box[0], box[1]);

    Point box_corners[4];
    box_corners[0].set(x1, y1);
    box_corners[1].set(x2, y1);
    box_corners[2].set(x2, y2);
    box_corners[3].set(x1, y2);

    // get oriented corners
    float angle_cos = cos(angle), angle_sin = sin(angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center, angle_cos, angle_sin, box_corners[k]);
        if (k == 0) {
            AA_box[0] = box_corners[k].x, AA_box[1] = box_corners[k].y;
            AA_box[2] = box_corners[k].x, AA_box[3] = box_corners[k].y;
        }
        else {
            AA_box[0] = min(AA_box[0], box_corners[k].x);
            AA_box[1] = min(AA_box[1], box_corners[k].y);
            AA_box[2] = max(AA_box[2], box_corners[k].x);
            AA_box[3] = max(AA_box[3], box_corners[k].y);
        }
    }
}

__global__ void patch_query_kernel_stack(int batch_size, int boxes_num, int boxes_per_patch_num, 
        int range_x, int range_y, float offset_x, float offset_y, float patch_size_x, float patch_size_y, 
        const float *boxes3d, const int *patch_indices, int *patch2box_indices) {
    // :param boxes3d: (B, M, 7)
    // :param patch_indices: (B, Y, X) 
    // output:
    //      patch2box_indices: (M1 + M2, boxes_per_patch_num + 1) [cnt, idx1, ...]
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= boxes_num) {
        return;
    }
    int bs_idx = blockIdx.y;

    const float *cur_box = boxes3d + bs_idx * boxes_num * 7 + box_idx * 7;

    float AA_box[4];
    get_AA_box(cur_box, AA_box);
    int patch_AA_box[4];
    for (int i = 0; i < 2; i++) {
        patch_AA_box[i * 2] = (AA_box[i * 2] - offset_x) / patch_size_x;
        patch_AA_box[i * 2 + 1] = (AA_box[i * 2 + 1] - offset_y) / patch_size_y;
    }

    for (int cur_x = patch_AA_box[0]; cur_x <= patch_AA_box[2]; cur_x++) {
        if (cur_x < 0 || cur_x >= range_x) continue;

        for (int cur_y = patch_AA_box[1]; cur_y <= patch_AA_box[3]; cur_y++) {
            if (cur_y < 0 || cur_y >= range_y) continue;

            int index = bs_idx * range_y * range_x + \
                        cur_y * range_x + \
                        cur_x;
            int patch_idx = patch_indices[index];
            if (patch_idx < 0) continue;

            int patch2box_idx = patch_idx * (boxes_per_patch_num + 1);
            int cnt = atomicAdd(patch2box_indices + patch2box_idx, 1);
            if (cnt < boxes_per_patch_num) {
                patch2box_indices[patch2box_idx + cnt + 1] = bs_idx * boxes_num + box_idx;
            }
        }
    }
}

void patch_query_kernel_launcher_stack(int batch_size, int boxes_num, int boxes_per_patch_num, 
        int range_x, int range_y, float offset_x, float offset_y, float patch_size_x, float patch_size_y,
        const float *boxes3d, const int *patch_indices, int *patch2box_indices) {
    // :param boxes3d: (B, M, 7)
    // :param patch_indices: (B, Y, X) 
    // output:
    //      patch2box_indices: (M1 + M2, boxes_per_patch_num + 1) [cnt, stacked_idx1, ...]
    cudaError_t err;

    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);

    patch_query_kernel_stack<<<blocks, threads>>>(batch_size, boxes_num, boxes_per_patch_num, range_x, range_y, offset_x, offset_y, 
        patch_size_x, patch_size_y, boxes3d, patch_indices, patch2box_indices);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
