// ============================================================
// 05 · Stencil（本周毕业练习）—— 一维 5 点 stencil，自己动手写
//
// 编译运行：
//   nvcc -O3 -arch=sm_75 -I common 05_stencil_TODO.cu -o stencil && ./stencil
//
// 任务：y[i] = C0*x[i] + C1*(x[i-1] + x[i+1]) + C2*(x[i-2] + x[i+2])
//       越界位置（i-2<0 或 i+2>=n）的 x 一律当作 0
//
// 你要写两个 kernel（搜 TODO）：
//   v1  朴素版：直接从全局内存读 5 个点
//   v2  halo 版：shared memory 协作加载 + 读取
//
// 这个练习检验你是否真正掌握了本实验包的所有套路：
//   索引计算 / grid-stride / __syncthreads / smem / 合并访存。
// 写完 v2 能看到明显加速。卡住了就把代码贴给我（或要提示）。
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "bench.cuh"

#define RADIUS 2
#define BLOCK 256

// 权重通过 kernel 参数传入（不用纠结 constant memory，那是下周的话题）
// C0 = 1.0, C1 = 0.8, C2 = 0.6

// ------------------------------------------------------------
// TODO v1：朴素版
//   1. 用 grid-stride loop 覆盖所有 i（复习 01 的 v2）
//   2. 读 x[i-2..i+2] 时先判断越界：越界当 0
//      提示：写一个小 helper 或直接三目运算符
// ------------------------------------------------------------
__global__ void stencil_v1(const float* x, float* y, int n,
                           float c0, float c1, float c2) {
    // TODO: 在这里实现
    (void)x; (void)y; (void)n; (void)c0; (void)c1; (void)c2;   // 写完可删
}

// ------------------------------------------------------------
// TODO v2：shared memory halo 版
//   1. __shared__ float s[BLOCK + 2*RADIUS];
//   2. 每个 block 负责一段 [block_start, block_start + BLOCK)，
//      但要额外把左右各 RADIUS 个“邻居”也搬进 smem：
//        s[RADIUS + tid] = x[block_start + tid]        （主体）
//      左 halo（s[0..1]）和右 halo（s[BLOCK+RADIUS ..]）分别由
//      tid < 2*RADIUS 的线程协作加载（注意越界补 0）
//   3. __syncthreads() 后，每个线程从 smem 读邻居算 y
//   4. 全局读写地址都要用 grid-stride 之外再想清楚：
//      建议 grid = ceil(n / BLOCK)，主体不用 grid-stride，只处理本块区间
//      （边界判断照旧）
// ------------------------------------------------------------
__global__ void stencil_v2(const float* x, float* y, int n,
                           float c0, float c1, float c2) {
    // TODO: 在这里实现
    (void)x; (void)y; (void)n; (void)c0; (void)c1; (void)c2;
}

int main() {
    const int n = 1 << 22;
    const size_t bytes = (size_t)n * sizeof(float);
    const float C0 = 1.0f, C1 = 0.8f, C2 = 0.6f;

    float *h = (float*)malloc(bytes), *hy = (float*)malloc(bytes), *href = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < n; ++i) h[i] = rand() / (float)RAND_MAX - 0.5f;

    auto at = [&](int i) { return (i < 0 || i >= n) ? 0.f : h[i]; };   // 越界当 0
    for (int i = 0; i < n; ++i)                                        // CPU 参考
        href[i] = C0 * at(i) + C1 * (at(i - 1) + at(i + 1)) + C2 * (at(i - 2) + at(i + 2));

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice));

    printf("== 05 stencil  (n = %d) ==\n", n);

    {   // v1
        int grid = (n + BLOCK - 1) / BLOCK;
        float ms = bench_ms([&] { stencil_v1<<<grid, BLOCK>>>(d_x, d_y, n, C0, C1, C2); });
        CUDA_CHECK(cudaMemcpy(hy, d_y, bytes, cudaMemcpyDeviceToHost));
        verify("v1 naive", hy, href, n);
        printf("v1 naive   %8.3f ms\n", ms);
    }
    {   // v2
        int grid = (n + BLOCK - 1) / BLOCK;
        float ms = bench_ms([&] { stencil_v2<<<grid, BLOCK>>>(d_x, d_y, n, C0, C1, C2); });
        CUDA_CHECK(cudaMemcpy(hy, d_y, bytes, cudaMemcpyDeviceToHost));
        verify("v2 halo  ", hy, href, n);
        printf("v2 halo    %8.3f ms\n", ms);
    }

    // 说明：填完 TODO 之前 FAIL 是正常的；两个都 OK 后对比时间。
    // 思考题：v1 里同一个 x[i] 被相邻 5 个线程各读一次，为什么 L1 cache
    // 没能让 v1 和 v2 一样快？（提示： smem 是软件控制的 cache，
//     想想你能不能保证它“一定命中”。）

    cudaFree(d_x); cudaFree(d_y);
    free(h); free(hy); free(href);
    return 0;
}
