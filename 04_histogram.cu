// ============================================================
// 04 · Histogram —— 原子操作与“减少竞争”的思路
//
// 编译运行：
//   nvcc -O3 -arch=sm_75 -I common 04_histogram.cu -o histogram && ./histogram
//
// 本实验回答的问题：
//   Q1 原子操作为什么必要、为什么慢？（多个线程抢写同一个地址只能排队）
//   Q2 shared memory 私有直方图为什么能大幅减少竞争？
//
// 预期结论：v2 比 v1 快数倍到数十倍（竞争被 smem 私有副本吸收）。
// ============================================================
#include <cstdio>
#include <cstdlib>
#include "bench.cuh"

#define BINS 256

// ------------------------------------------------------------
// v1：直接原子加到全局显存
// 最容易写的版本。问题：256 个 bin 里热点 bin（数据均匀分布时好一些，
// 偏斜数据时灾难）会成为串行点——所有 block 的原子加都挤同 256 个地址。
// ------------------------------------------------------------
__global__ void histo_v1(const unsigned char* data, unsigned int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&out[data[i]], 1u);
}

// ------------------------------------------------------------
// v2：shared memory 私有直方图 + 最后合并
// 每个 block 先在自己的 smem 里统计（竞争只发生在 256 个线程之间，
// 且 smem 原子操作快得多），block 结束时一次性合并回全局。
// 网格跨步遍历让每个线程处理多个元素，摊薄初始化和合并的开销。
// ------------------------------------------------------------
__global__ void histo_v2(const unsigned char* data, unsigned int* out, int n) {
    __shared__ unsigned int priv[BINS];
    int tid = threadIdx.x;
    for (int k = tid; k < BINS; k += blockDim.x) priv[k] = 0;   // 协作清零
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x)
        atomicAdd(&priv[data[i]], 1u);
    __syncthreads();

    for (int k = tid; k < BINS; k += blockDim.x)
        atomicAdd(&out[k], priv[k]);                            // 合并回全局
}

int main() {
    const int n = 1 << 24;
    unsigned char* h = (unsigned char*)malloc(n);
    srand(42);
    for (int i = 0; i < n; ++i) h[i] = rand() & 0xFF;

    unsigned int ref[BINS] = {0};                     // CPU 参考答案
    for (int i = 0; i < n; ++i) ref[h[i]]++;

    unsigned char* d_data;
    unsigned int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h, n, cudaMemcpyHostToDevice));

    const int block = 256;
    printf("== 04 histogram  (n = %d) ==\n", n);

    auto check = [&](const char* name) {
        unsigned int got[BINS];
        CUDA_CHECK(cudaMemcpy(got, d_hist, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        bool ok = true;
        unsigned long long total = 0;
        for (int k = 0; k < BINS; ++k) { total += got[k]; if (got[k] != ref[k]) ok = false; }
        printf("[%s] total = %llu  %s\n", name, total, ok ? "OK" : "FAIL");
    };

    {   // v1
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        int grid = (n + block - 1) / block;
        float ms = bench_ms([&] { histo_v1<<<grid, block>>>(d_data, d_hist, n); });
        printf("v1 global atomic   %8.3f ms\n", ms);
        check("v1");
    }
    {   // v2
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        int grid = 2048;   // 少量 block + grid-stride
        float ms = bench_ms([&] { histo_v2<<<grid, block>>>(d_data, d_hist, n); });
        printf("v2 smem private    %8.3f ms\n", ms);
        check("v2");
    }

    cudaFree(d_data); cudaFree(d_hist); free(h);
    return 0;
}
