# Week 1 CUDA 实验包

配合《算子开发转岗实操指南》Week 0-1 使用。核心理念：**不背优化技巧，用"预测 → 验证"的循环把每个技巧变成你自己跑出来的数字。**

## 文件清单与学习顺序

| 文件 | 内容 | 性质 |
|---|---|---|
| `common/bench.cuh` | 计时 / 校验工具（warmup + 中位数计时 + CUDA_CHECK） | 公共设施，先读一遍 |
| `01_vector_add.cu` | 3 个版本：朴素 / grid-stride / float4 | 完整示例 |
| `02_transpose.cu` | 3 个版本：朴素 / smem tile / +padding | 完整示例 |
| `03_reduction.cu` | 4 级阶梯：naive tree → interleaved → coarsening → warp shuffle | **本周核心**，完整示例 |
| `04_histogram.cu` | 2 个版本：全局原子 / smem 私有直方图 | 完整示例 |
| `05_stencil_TODO.cu` | 一维 5 点 stencil，两个 kernel 留空 | **毕业练习，自己写** |

## 快速开始（Colab）

把本目录上传到你的 GitHub 仓库（比如 `cuda-triton-kernels`），然后在 Colab 里：

```python
!git clone https://github.com/<你的用户名>/cuda-triton-kernels.git
%cd cuda-triton-kernels/week1-cuda-labs
```

```python
# 逐个编译运行（T4 用 sm_75；如果分到 P100 换成 sm_60）
!nvcc -O3 -arch=sm_75 -I common 01_vector_add.cu -o vec_add && ./vec_add
!nvcc -O3 -arch=sm_75 -I common 02_transpose.cu -o transpose && ./transpose
!nvcc -O3 -arch=sm_75 -I common 03_reduction.cu -o reduction && ./reduction
!nvcc -O3 -arch=sm_75 -I common 04_histogram.cu -o histogram && ./histogram
!nvcc -O3 -arch=sm_75 -I common 05_stencil_TODO.cu -o stencil && ./stencil
```

## 学习方法：四步循环（比看十遍代码有用）

以 `03_reduction.cu` 为例：

1. **先跑**：运行 v1，看懂输出的每个数字（GB/s 怎么算的？离峰值多远？）
2. **读 diff**：v1 → v2 只改了几行。先别看注释里的解释，自己猜"为什么这样会快"
3. **预测再验证**：猜一个倍率（比如"应该快 30%"），然后跑 v2 对答案。**猜错的地方就是你认知的缺口**，回头翻 PMPP 对应章节
4. **合上代码重写**：能不看源码写出 v2，才算掌握，再进入 v3

> 铁律：每个实验都要求"跑出数字"，没有数字的阅读等于没学。把每次的实测数字记在下面的表里。

## 预期现象速查（T4 量级参考，以你的实测为准）

| 实验 | 预期现象 | 你的实测 |
|---|---|---|
| 01 vector add | 三个版本都接近带宽极限（峰值约 300 GB/s，能到 70%+），差距小 | |
| 02 transpose | v1→v2 提升约 2~3 倍；v2→v3 再提升约 10% | |
| 03 reduction | 逐级提升，v1→v4 合计接近 2 倍 | |
| 04 histogram | v2 明显快（全局原子竞争被 smem 私有副本吸收） | |
| 05 stencil | v2 比 v1 快（幅度自己验证） | |

## 完成标准（Week 1 验收）

- [ ] 能不看源码重写 reduction 的 v2 和 v4（包括 `__shfl_down_sync`）
- [ ] 能用自己的话解释：transpose v2 为什么快、padding 加在哪、为什么有效
- [ ] 能解释 grid-stride loop 的两个好处
- [ ] 05 stencil 的两个 kernel 全部完成且 OK
- [ ] 每个实验的实测数字已填进上表，并各写一句结论

## 遇到问题

- 编译错误：先看 `CUDA_CHECK` 打印的文件行号；索引类错误多半是 `blockIdx/threadIdx` 公式抄错
- 结果 FAIL：先检查边界条件（`i < n`、越界补 0），再检查 `__syncthreads()` 位置
- 数字和预期差太远：正常，把代码 + 实测数字贴出来一起分析
- stencil 卡住：可以先要提示，不丢人
