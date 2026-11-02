// ============================================================
// 03 · Reduction —— 四级优化阶梯（本周最重要的实验）
//
// 编译运行：
//   nvcc -O3 -arch=sm_75 -I common 03_reduction.cu -o reduction && ./reduction
//
// 任务：把 16.7M 个数加成一个数。这个实验教会你 Week 1 的核心方法论：
//   每一版只改一个点 → 预测它会快多少 → 跑数字验证 → 想明白为什么。
//
// 阶梯（对应 PMPP 第 9 章）：
//   v1  树形归约（朴素版）        —— 正确，但 warp 分支严重
//   v2  交错寻址（地址连续）      —— 消除 warp divergence
//   v3  线程粗化（每线程多算几个）—— 减少块数、复用合并读
//   v4  warp shuffle（蝶形归约）  —— 最后 32 个元素不再用 smem + 同步
//
// 预期结论：v1 → v4 能有接近 2 倍提升，最终 200+ GB/s（T4 峰值约 300）。
// ============================================================
#include <cstdio>
#include <cstdlib>
#include "bench.cuh"

#define BLOCK 256

// ------------------------------------------------------------
// v1：树形归约（朴素版）
// 每个 block 用 smem 归约 2*BLOCK 个元素；stride 加倍，活跃线程减半。
// 问题：if (tid % (2*stride) == 0) → 同一 warp 内线程“有的干活有的闲着”，
//       且活跃线程地址跳着走 → divergence + 不合并，双重浪费。
// ------------------------------------------------------------
__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * (BLOCK * 2) + tid;
    float a = (i < n) ? in[i] : 0.f;
    float b = (i + BLOCK < n) ? in[i + BLOCK] : 0.f;
    s[tid] = a + b;
    __syncthreads();

    for (int stride = 1; stride < BLOCK; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];   // 活跃线程间隔 2*stride
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ------------------------------------------------------------
// v2：交错寻址（interleaved addressing）
// 只改了一处：改成 stride 减半、活跃的是“前面连续 tid < stride”的线程。
// 同一 warp 里 32 个线程要么全部干活、要么全部休息 → 无 divergence，
// 且活跃线程访问的 smem 地址连续。
// ------------------------------------------------------------
__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * (BLOCK * 2) + tid;
    float a = (i < n) ? in[i] : 0.f;
    float b = (i + BLOCK < n) ? in[i + BLOCK] : 0.f;
    s[tid] = a + b;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];              // 活跃线程连续
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ------------------------------------------------------------
// v3：+ 线程粗化（thread coarsening）
// 每个线程先用 grid-stride loop 把多个元素累加进寄存器（acc 就在寄存器里，
// 零成本），再进 smem 树。全局读从“2 元素/线程”变成“16 元素/线程”，
// block 数从 32768 降到 4096，调度开销更小、合并读更充分。
// ------------------------------------------------------------
__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    float acc = 0.f;
    for (; i < n; i += (size_t)BLOCK * gridDim.x) acc += in[i];   // grid-stride
    s[tid] = acc;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ------------------------------------------------------------
// v4：+ warp shuffle
// 当 stride 降到 32 以内时，归约只涉及一个 warp 内的 32 个线程：
//   __shfl_down_sync 直接在线程“寄存器之间”传数据，
//   不用写 smem、不用 __syncthreads()（warp 天然同步）。
// 这是 warp 级原语最重要的应用，务必亲手写一遍。
// ------------------------------------------------------------
__global__ void reduce_v4(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    float acc = 0.f;
    for (; i < n; i += (size_t)BLOCK * gridDim.x) acc += in[i];
    s[tid] = acc;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 16; stride >>= 1) {     // smem 树到 32 为止
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {                        // 最后一个 warp 收尾
        float v = s[tid];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (tid == 0) out[blockIdx.x] = v;
    }
}

int main() {
    const int n = 1 << 24;
    const size_t bytes = (size_t)n * sizeof(float);
    const int max_blocks = (n + 2 * BLOCK - 1) / (2 * BLOCK);   // 32768

    // ---- 数据 + CPU 参考答案（double 累加，误差最小）----
    float* h = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < n; ++i) h[i] = rand() / (float)RAND_MAX - 0.5f;
    double ref = 0.0;
    for (int i = 0; i < n; ++i) ref += h[i];

    float *d_in, *d_part, *h_part;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_part, (size_t)max_blocks * sizeof(float)));
    h_part = (float*)malloc((size_t)max_blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    printf("== 03 reduction  (n = %d, ref = %.4f) ==\n", n, ref);
    printf("说明：kernel 算出每个 block 的部分和，host 端再求总sum；"
           "带宽 = n*4字节/时间\n\n");

    // 通用收尾：把部分和拷回 host，用 double 求总和
    auto finalize = [&](int nblocks) {
        CUDA_CHECK(cudaMemcpy(h_part, d_part, (size_t)nblocks * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0.0;
        for (int i = 0; i < nblocks; ++i) sum += h_part[i];
        double rel = fabs(sum - ref) / fabs(ref);
        printf("    sum = %.4f  rel_err = %.2e  %s\n", sum, rel, rel < 1e-3 ? "OK" : "FAIL");
    };

    float ms1 = 0, ms2 = 0, ms3 = 0, ms4 = 0;
    {   // v1：每 block 归约 2*BLOCK 个元素
        int grid = max_blocks;
        ms1 = bench_ms([&] { reduce_v1<<<grid, BLOCK>>>(d_in, d_part, n); });
        printf("v1 naive tree      %8.3f ms  %7.1f GB/s\n", ms1, bytes / ms1 / 1e6);
        finalize(grid);
    }
    {   // v2
        int grid = max_blocks;
        ms2 = bench_ms([&] { reduce_v2<<<grid, BLOCK>>>(d_in, d_part, n); });
        printf("v2 interleaved     %8.3f ms  %7.1f GB/s  (vs v1: %.2fx)\n", ms2, bytes / ms2 / 1e6, ms1 / ms2);
        finalize(grid);
    }
    {   // v3：block 数降到 4096
        int grid = (n + BLOCK - 1) / BLOCK < 4096 ? (n + BLOCK - 1) / BLOCK : 4096;
        ms3 = bench_ms([&] { reduce_v3<<<grid, BLOCK>>>(d_in, d_part, n); });
        printf("v3 +coarsening     %8.3f ms  %7.1f GB/s  (vs v2: %.2fx)\n", ms3, bytes / ms3 / 1e6, ms2 / ms3);
        finalize(grid);
    }
    {   // v4
        int grid = (n + BLOCK - 1) / BLOCK < 4096 ? (n + BLOCK - 1) / BLOCK : 4096;
        ms4 = bench_ms([&] { reduce_v4<<<grid, BLOCK>>>(d_in, d_part, n); });
        printf("v4 +warp shuffle   %8.3f ms  %7.1f GB/s  (vs v3: %.2fx)\n", ms4, bytes / ms4 / 1e6, ms3 / ms4);
        finalize(grid);
    }

    cudaFree(d_in); cudaFree(d_part); free(h_part); free(h);
    return 0;
}
