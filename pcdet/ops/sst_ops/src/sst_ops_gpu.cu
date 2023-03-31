#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define EMPTY_KEY -1

__device__ inline int simple_hash(int k, int hash_size) {
    return k % hash_size;
}

__global__ void ingroup_inds_kernel(int N, const long *group_inds, long *out_inds, int *ingroup_counter) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N) return;
    long this_group_id = group_inds[pt_idx];
    int cnt = atomicAdd(ingroup_counter + this_group_id, 1);
    out_inds[pt_idx] = cnt;
}

__global__ void group_inner_inds_kernel(int N, int K, const long *inverse_inds, long *group_inds, int *ingroup_counter) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N) return;
    long this_group_id = inverse_inds[pt_idx];
    int cnt = atomicAdd(ingroup_counter + this_group_id, 1);
    if (cnt < K) group_inds[this_group_id * K + cnt] = pt_idx;
}

__global__ void repeat_group_idx_kernel(int M, int K, const int *ingroup_counter, long *group_inds){
    // params ingroup_counter: (M,)
    // params group_inds: (M, K)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;
    int cnt = ingroup_counter[pt_idx];
    if (cnt == 0) return;
    for (int i = cnt; i < K; i++) 
        group_inds[pt_idx * K + i] = group_inds[pt_idx * K + i % cnt];
}

void ingroup_inds_launcher(int N, int max_group_id, const long *group_inds, long *out_inds) {
    int *ingroup_counter = NULL;
    cudaMalloc(&ingroup_counter, (max_group_id + 1) * sizeof(int));
    cudaMemset(ingroup_counter, 0, (max_group_id + 1) * sizeof(int));
    
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    ingroup_inds_kernel<<<blocks, threads>>>(N, group_inds, out_inds, ingroup_counter);
    cudaFree(ingroup_counter);
}

void group_inner_inds_launcher(int N, int M, int K, const long *inverse_inds, long *group_inds) {
    int *ingroup_counter = NULL;
    cudaMalloc(&ingroup_counter, M * sizeof(int));
    cudaMemset(ingroup_counter, 0, M * sizeof(int));
    
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    group_inner_inds_kernel<<<blocks, threads>>>(N, K, inverse_inds, group_inds, ingroup_counter);

    dim3 blocks1(DIVUP(M, THREADS_PER_BLOCK));
    repeat_group_idx_kernel<<<blocks1, threads>>>(M, K, ingroup_counter, group_inds);
    cudaFree(ingroup_counter);
}
