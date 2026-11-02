// ============================================================
// 01 · Vector Add —— 第一个 kernel：建立“线程 ↔ 数据”的映射直觉
//
// 编译运行（Colab T4）：
//   nvcc -O3 -arch=sm_75 -I common 01_vector_add.cu -o vec_add && ./vec_add
//
// 本实验回答三个问题：
//   Q1 朴素版本能跑多快？离显存带宽极限有多远？
//   Q2 grid-stride loop 解决什么问题？
//   Q3 float4 向量化对“带宽已打满”的 kernel 还有用吗？
//
// 预期结论：三个版本差距不大，都能到峰值带宽的 70%+（T4 峰值约 300 GB/s）。
//   第一个教训：vector add 是纯访存型 kernel，本来就没什么可优化的——
//   动手前先判断瓶颈在哪，别为打满带宽的 kernel 浪费时间。
// ============================================================
#include <cstdio>
#include <cstdlib>
#include "bench.cuh"

// ------------------------------------------------------------
// v1：一个线程处理一个元素
// 这就是索引公式最原始的样子：
//   i = blockIdx.x * blockDim.x + threadIdx.x   （块起点 + 块内偏移）
// ------------------------------------------------------------
__global__ void vec_add_v1(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];      // if 是防御最后一组线程越界
}

// ------------------------------------------------------------
// v2：grid-stride loop —— 写“通用 kernel”的标准姿势
// 线程总数不够覆盖 n 时，每个线程跨过一个 grid 总长，回头继续处理
// 好处：1) 任何 n 都能覆盖，grid 大小随便定
//       2) 后面所有 kernel（reduction/stencil）都复用这个模式
// ------------------------------------------------------------
__global__ void vec_add_v2(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;    // 全局线程总数 = “步长”
    for (; i < n; i += stride) c[i] = a[i] + b[i];
}

// ------------------------------------------------------------
// v3：float4 向量化 —— 一条访存指令搬 16 字节而不是 4 字节
// 注意：要求 n 能被 4 整除，且指针 16 字节对齐（cudaMalloc 返回的指针天然对齐）
// ------------------------------------------------------------
__global__ void vec_add_v3(const float4* a, const float4* b, float4* c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i];
        float4 y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

int main() {
    const int n = 1 << 24;                        // 16.7M 元素
    const size_t bytes = (size_t)n * sizeof(float);

    // ---- 准备数据 + CPU 参考答案 ----
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes);
    float *hc = (float*)malloc(bytes), *href = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < n; ++i) {
        ha[i] = rand() / (float)RAND_MAX - 0.5f;
        hb[i] = rand() / (float)RAND_MAX - 0.5f;
        href[i] = ha[i] + hb[i];
    }

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    printf("== 01 vector add  (n = %d, T4 带宽峰值约 300 GB/s) ==\n", n);

    {   // ---- v1 ----
        int grid = (n + block - 1) / block;
        float ms = bench_ms([&] { vec_add_v1<<<grid, block>>>(da, db, dc, n); });
        CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
        verify("v1", hc, href, n);
        printf("v1 one-elem-per-thread  %8.3f ms  %7.1f GB/s\n", ms, 3.0 * bytes / ms / 1e6);
    }
    {   // ---- v2 ----
        int grid = 2048;   // 故意少开点 block，让 grid-stride 生效
        float ms = bench_ms([&] { vec_add_v2<<<grid, block>>>(da, db, dc, n); });
        CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
        verify("v2", hc, href, n);
        printf("v2 grid-stride          %8.3f ms  %7.1f GB/s\n", ms, 3.0 * bytes / ms / 1e6);
    }
    {   // ---- v3 ----
        int n4 = n / 4;
        int grid = (n4 + block - 1) / block;
        float ms = bench_ms([&] { vec_add_v3<<<grid, block>>>((float4*)da, (float4*)db, (float4*)dc, n4); });
        CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
        verify("v3", hc, href, n);
        printf("v3 float4               %8.3f ms  %7.1f GB/s\n", ms, 3.0 * bytes / ms / 1e6);
    }

    // 字节数说明：读 a + 读 b + 写 c = 3 份数据，所以 GB/s = 3*bytes / 时间

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc); free(href);
    return 0;
}
