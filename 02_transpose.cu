// ============================================================
// 02 · Transpose —— 用图亲眼看“访存合并”值多少钱
//
// 编译运行：
//   nvcc -O3 -arch=sm_75 -I common 02_transpose.cu -o transpose && ./transpose
//
// 本实验回答的问题：
//   Q1 朴素的矩阵转置为什么慢？（warp 里 32 个线程的写入地址跳着走）
//   Q2 shared memory 怎么把“读写两边”都变成合并访存？
//   Q3 什么是 bank conflict？padding 为什么能消除它？
//
// 预期结论：v1 → v2 有 2~3 倍提升；v2 → v3（加 padding）再提升约 10%。
//   这是 Week 1 里“收益最直观”的实验，数字一定要自己跑出来。
// ============================================================
#include <cstdio>
#include <cstdlib>
#include "bench.cuh"

#define TILE 32   // 一个 block 负责 32x32 的 tile；32 也正好是 warp 大小

// ------------------------------------------------------------
// v1：朴素转置
//   读 in[y*n + x]：warp 内 x 连续 → 地址连续 → 合并 ✓
//   写 out[x*n + y]：warp 内 x 连续但行号在变 → 地址相差 n*4 字节 → 完全不合并 ✗
// 一行数据 32 个 float 被拆成 32 次事务去写，带宽利用率极差。
// ------------------------------------------------------------
__global__ void transpose_v1(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;   // 列
    int y = blockIdx.y * TILE + threadIdx.y;   // 行
    if (x < n && y < n) out[x * n + y] = in[y * n + x];
}

// ------------------------------------------------------------
// v2：shared memory tile
// 分两步走，每一步都是合并的：
//   第 1 步：全局读 → smem。按“输入 tile”的行序读，合并 ✓
//   第 2 步：smem → 全局写。注意输出块坐标做了交换
//           （输入块的 (by,bx) 是输出块的 (bx,by)），写合并 ✓
// 代价：smem 读 tile[threadIdx.x][threadIdx.y] 是“跨行取”，
//       同 warp 的 32 个线程命中同一个 bank → 32 路 bank conflict（下一步修）
// ------------------------------------------------------------
__global__ void transpose_v2(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];     // 全局读：合并
    __syncthreads();

    // 输出坐标：块号交换，块内 (tx,ty) 也要换成转置后的角色
    int xo = blockIdx.y * TILE + threadIdx.x;           // warp 内连续 → 写合并
    int yo = blockIdx.x * TILE + threadIdx.y;
    out[yo * n + xo] = tile[threadIdx.x][threadIdx.y];  // smem 读：bank conflict!
}

// ------------------------------------------------------------
// v3：+1 padding 消除 bank conflict
//   smem 是 32 个 bank，按字地址交错分布。
//   tile[32][32]：同一列的地址全部落在同一个 bank（32*i+j ≡ j mod 32）
//   tile[32][33]：多一列 padding，列内地址错开 1 个 bank → conflict 消失
// 代价：每行多 4 字节，几乎免费。
// ------------------------------------------------------------
__global__ void transpose_v3(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];              // 唯一改动：+1

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();

    int xo = blockIdx.y * TILE + threadIdx.x;
    int yo = blockIdx.x * TILE + threadIdx.y;
    out[yo * n + xo] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int n = 4096;   // 必须是 TILE 的倍数（本周先不考虑边界）
    const size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < n * n; ++i) h_in[i] = rand() / (float)RAND_MAX;
    for (int y = 0; y < n; ++y)                 // CPU 参考答案
        for (int x = 0; x < n; ++x)
            h_ref[(size_t)x * n + y] = h_in[(size_t)y * n + x];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(n / TILE, n / TILE);
    printf("== 02 transpose  (n = %d, T4 带宽峰值约 300 GB/s) ==\n", n);

    float ms1 = bench_ms([&] { transpose_v1<<<grid, block>>>(d_in, d_out, n); });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    verify("v1 naive     ", h_out, h_ref, n * n);
    printf("v1 naive        %8.3f ms  %7.1f GB/s\n", ms1, 2.0 * bytes / ms1 / 1e6);

    float ms2 = bench_ms([&] { transpose_v2<<<grid, block>>>(d_in, d_out, n); });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    verify("v2 smem      ", h_out, h_ref, n * n);
    printf("v2 smem         %8.3f ms  %7.1f GB/s  (vs v1: %.2fx)\n", ms2, 2.0 * bytes / ms2 / 1e6, ms1 / ms2);

    float ms3 = bench_ms([&] { transpose_v3<<<grid, block>>>(d_in, d_out, n); });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    verify("v3 smem+pad  ", h_out, h_ref, n * n);
    printf("v3 smem+padding %8.3f ms  %7.1f GB/s  (vs v2: %.2fx)\n", ms3, 2.0 * bytes / ms3 / 1e6, ms2 / ms3);

    // 字节数说明：读一遍 + 写一遍 = 2 份数据

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
