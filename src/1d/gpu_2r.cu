#include <mma.h>
#include <iostream>
#include "1d_utils.h"
#include <chrono>

using namespace nvcuda;

#define HALO 4
#define WARP_PER_BLOCK 8
#define BLOCK_SIZE_COL 1024
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2 * 8)
#define IDX(x, y, ldm) ((x) * (ldm) + (y))

__constant__ double param_matrix_d[16 * 8];

__global__ void kernel_1d2r(const double *__restrict__ in, double *__restrict__ out) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL];
    int begin = blockIdx.x * BLOCK_SIZE_COL;

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_COL; i += totalThreads) {
        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(dst), "l"(&in[begin + i]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);

    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int warp_begin = warp_id * 8 * 4;

    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag1;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag1;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag[4];

#pragma unroll
    for (int row = 0; row < 4; row++) {
        wmma::load_matrix_sync(param_frag1, param_matrix_d + row * 32, 8);
#pragma unroll
        for (int data = 0; data < 2; data++) {
            wmma::load_matrix_sync(in_frag1, sharedmem + warp_begin + 8 * data + 4 * row,
                                   D_BLOCK_SIZE_COL / 8);
            wmma::mma_sync(acc_frag[data], in_frag1, param_frag1, acc_frag[data]);
        }
    }

#pragma unroll
    for (int data = 0; data < 2; data++) {
        wmma::store_matrix_sync(
                out + begin + warp_begin + IDX(data, 4, 8),
                acc_frag[data], D_BLOCK_SIZE_COL / 8, wmma::mem_row_major);
    }
}


void
gpu_1d2r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times,
         const int input_n) {
    double param_matrix_h[8 * 16] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 9; row++) {
            param_matrix_h[(row + col) * 8 + col] = params[row];
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 16 * 8 * sizeof(double)));

    const int cols = input_n + 2 * HALO;
    const size_t array_size = cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++) {
        CUDAKERNELCHECK((kernel_1d2r<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2])));
//        CUDAKERNELCHECK((kernel_star1d2r_step2_m2<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2])));
//        CUDAKERNELCHECK(
//                (kernel_star1d2r_step2_m3<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2])));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "LoRAStencil(1D 1d2r): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
              << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double) input_n * times * 2) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size - sizeof(double), cudaMemcpyDeviceToHost));


}