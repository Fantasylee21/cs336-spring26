Problem (mixed_precision_accumulation):  Mixed-Precision Accumulation 
  Case 1 (fp32 + fp32):  10.0001335144   (error ~0.0013%)
  Case 2 (fp16 + fp16):   9.9531250000   (error ~0.47%)
  Case 3 (fp32 + fp16):  10.0021362305   (error ~0.02%)
  Case 4 (fp32 + fp16→fp32): 10.0021362305  (same as Case 3)

  Case 1 误差很小，0.01 在 float32 中无法精确表示（实际是 0.009999999776...），累积 1000 次后舍入误差仅为 ~0.0013%。

  Case 2 误差最大（~4.7%），原因是 float16 只有 ~3 位有效数字。当累加器 s 增长到 8-10 附近时，float16 的 ULP（最小可分辨单位）约为
  0.01，此后再加 0.01 会被舍入到 0，后续几百次加法完全丢失。这就是 catastrophic cancellation 
  的反面——吸收误差：大数加小数时，小数的精度被大数的量级吞没。

  Case 3/4 结果相同，都是 10.002 而非 10.0，因为问题出在源头——0.01 在 float16 中存为 0.010002136...，已经有 +0.02% 的系统偏差。用 float32
  累加只能避免累加过程的精度丢失，但无法纠正增量本身已是"错的"。Case 2 恰好因为后期加法被吞没，"歪打正着"抵消了部分偏差（9.95），但这纯属巧合。


LayerNorm 为何被特殊对待 / BF16 是否不同
  
  LayerNorm 需要计算 var = E[x^2] - E[x]^2。FP16 的动态范围仅 ±65504，x^2 极易溢出（例如输入 300 就产生 90000 > 65504）。此外 FP16 只有 3
  位有效数字，在大序列长度上做 reduction 时累积误差严重。因此 PyTorch 将 LayerNorm 硬编码为 FP32 计算。
  
  BF16 的指数位与 FP32 相同（8 bit），动态范围一致，不会出现 x^2 溢出。因此在 BF16 autocast 下，LayerNorm 不需要像 FP16 那样被特殊对待——PyTorch
  的 BF16 autocast policy 也更宽松，允许更多操作直接在 BF16 下运行。不过 BF16 的尾数位少（7 bit vs FP32 的 23
  bit），在极长序列上做均值/方差归约时仍可能有微小精度损失，但不会产生 FP16 那种灾难性溢出。