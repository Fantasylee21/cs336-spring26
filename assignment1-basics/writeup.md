Problem (unicode1):  Understanding Unicode

(a) '\x00'

(b) Printed representation such as 'print' is human-readable, while __repr__() is meant to be unambiguous and can be used to recreate the object if needed.

(c) 
```
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

Problem (unicode2):  Unicode Encodings

(a) UTF-8 byte tokenizers yield far smaller, more compact representations for common Latin, numeric, and ASCII text than UTF-16/UTF-32, which waste space with fixed two/four-byte codepoints for basic characters

(b) This test with ASCII "hello" coincidentally works, but passing multi-byte text like "中文".encode("utf-8") will throw UnicodeDecodeError. The encoding rule of UTF-8: A single Unicode character is encoded as 1 to 4 consecutive bytes. Not every byte corresponds to a single character independently.

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes) -> str:
    return bytestring.decode("utf-8")

# Test call
res = decode_utf8_bytes_to_str_wrong("hello呵呵".encode("utf-8"))
print(res)

```

(c) The first byte 0xFF starts with eight 1 bits, which UTF-8 defines as an invalid leading byte with no valid character length encoding, making this two-byte sequence entirely undecodable to any Unicode character under the UTF-8 standard.

Problem (train_bpe_tinystories)
(a)

train on Apple M5 MacbookPro, 10 cores, 24GB RAM

Training time: 124.1 seconds (2.1 minutes)

Peak memory: 14871.9 MB

Longest token ID: 7160

Longest token length: 15 bytes

Longest token (UTF-8): 'Ġaccomplishment'

make sense

(b) 每次合并都要全量扫描 pair_counts 找最大值

Problem (train_bpe_expts_owt):  BPE Training on OpenWebText 
trained on Intel(R) Xeon(R) Processor @ 2.90GHz 16 cores 400G
Training time: 43417.6 seconds (723.6 minutes)
Peak memory: 101518.2 MB
Vocabulary size: 32000
Number of merges: 31743
Longest token ID: 25822
Longest token length: 64 bytes
Longest token (UTF-8): 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'

Problem (tokenizer_experiments):  Experiments with tokenizers

TinyStories Tokenizer :
macbook:
TinyStories: 10751 bytes / 2663 tokens = 4.04 bytes/token
throughput: 3230.36 total_bytes/sec
OpenWebText: 50460 bytes / 14821 tokens = 3.40 bytes/token
throughput: 2947.30 total_bytes/sec
alibaba server:
[TinyStories]
  10,751 bytes -> 2,663 tokens (4.04 bytes/token)
  4.05s, throughput: 2,654.32 bytes/s

[OpenWebText]
  50,460 bytes -> 14,821 tokens (3.40 bytes/token)
  20.36s, throughput: 2,478.30 bytes/s

OpenWebText Tokenizer :
[TinyStories]
  10,751 bytes -> 2,760 tokens (3.90 bytes/token)
  14.42s, throughput: 745.32 bytes/s

[OpenWebText]
  50,460 bytes -> 11,201 tokens (4.50 bytes/token)
  60.11s, throughput: 839.51 bytes/s

825GB = 825 * 1024 * 1024 * 1024 = 865075200 bytes

TinyStories Tokenizer :
time1 = 865075200 / 2947.30 = 293514.47 seconds = 3.40 days
time2 = 865075200 / 3230.36 = 267795.29 seconds = 3.10 days

OpenWebText Tokenizer :
time1 = 865075200 / 839.51 = 1030452.53 seconds = 11.93 days
time1 = 865075200 / 745.32 = 1160676.22 seconds = 13.43 days

(d) uint16 的取值范围是 0–65535，而 10K vocab 的 token ID 最大只到 9999

优化版本
Training time: 109.8 seconds (1.8 minutes)
Peak memory: 52.8 MB
Vocabulary size: 10000
Number of merges: 9743
Longest token ID: 7160
Longest token length: 15 bytes
Longest token (UTF-8): ' accomplishment'

TinyStories Tokenizer :

[TinyStories]
  10,751 bytes -> 2,663 tokens (4.04 bytes/token)
  0.00s, throughput: 3,202,625.16 bytes/s

[OpenWebText]
  50,460 bytes -> 14,821 tokens (3.40 bytes/token)
  0.02s, throughput: 3,031,505.83 bytes/s

time1 = 865075200 / 3202625.16 = 270.11 seconds = 4.50 minutes
time2 = 865075200 / 3031505.83 = 285.36 seconds = 4.76 minutes

(a) Parameter Count for GPT-2 XL

  Architecture components (no bias, no weight tying):

  ┌─────────────────────────┬─────────────┬────────────────┐

  │        Component        │    Shape    │     Count      │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ Token embedding         │ V × d       │ 50,257 × 1,600 │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ Per block: Q/K/V/O proj │ 4 × (d × d) │ 4 × 1,600²     │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ Per block: RMSNorm (×2) │ 2 × d       │ 2 × 1,600      │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ Per block: FFN W1       │ f × d       │ 4,288 × 1,600  │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ Per block: FFN W2       │ d × f       │ 1,600 × 4,288  │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ Per block: FFN W3       │ f × d       │ 4,288 × 1,600  │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ Final RMSNorm           │ d           │ 1,600          │

  ├─────────────────────────┼─────────────┼────────────────┤

  │ LM head                 │ V × d       │ 50,257 × 1,600 │

  └─────────────────────────┴─────────────┴────────────────┘

  Per block: 4d² + 2d + 3df = 4·1,600² + 3,200 + 3·1,600·4,288 = 10,240,000 + 3,200 + 20,582,400 = 30,825,600

  Total: 2·Vd + 48·30,825,600 + d = 2·80,411,200 + 1,479,628,800 + 1,600 = ≈1.64B parameters

  Memory: 1.64B × 4 bytes = ≈6.56 GB (6.11 GiB)

  ---
  (b) Matrix Multiplies & Total FLOPs
  
  Denote L = context_length = 1024. For each matmul A(m×n) × B(n×p), FLOPs = 2mnp.

  Per block:

  ┌─────┬───────────────────────┬─────────────────────────┬───────┐

  │  #  │       Operation       │         Shapes          │ FLOPs │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 1   │ Q = x @ W_q^T         │ (L,d) × (d,d)           │ 2Ld²  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 2   │ K = x @ W_k^T         │ (L,d) × (d,d)           │ 2Ld²  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 3   │ V = x @ W_v^T         │ (L,d) × (d,d)           │ 2Ld²  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 4   │ Q_h @ K_h^T (h heads) │ h × (L, d/h) × (d/h, L) │ 2L²d  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 5   │ A_h @ V_h (h heads)   │ h × (L, L) × (L, d/h)   │ 2L²d  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 6   │ Out = concat @ W_o^T  │ (L,d) × (d,d)           │ 2Ld²  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 7   │ W1(x)                 │ (L,d) × (d,f)           │ 2Ldf  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 8   │ W3(x)                 │ (L,d) × (d,f)           │ 2Ldf  │

  ├─────┼───────────────────────┼─────────────────────────┼───────┤

  │ 9   │ W2(gate ⊙ W3)         │ (L,f) × (f,d)           │ 2Ldf  │

  └─────┴───────────────────────┴─────────────────────────┴───────┘

  Per block: 8Ld² + 4L²d + 6Ldf

  LM head: (L,d) × (d,V) → 2LdV

  Plugging in GPT-2 XL numbers (L=1024, d=1600, f=4288, N=48, V=50257):

  ┌───────────────────┬───────────┬────────────┬───────┐

  │     Component     │ Per Block │ ×48 Blocks │   %   │

  ├───────────────────┼───────────┼────────────┼───────┤

  │ QKV+O proj (8Ld²) │ 20.97B    │ 1,006.6B   │ 28.6% │

  ├───────────────────┼───────────┼────────────┼───────┤

  │ SDPA (4L²d)       │ 6.71B     │ 322.1B     │ 9.2%  │

  ├───────────────────┼───────────┼────────────┼───────┤

  │ FFN (6Ldf)        │ 42.15B    │ 2,023.4B   │ 57.5% │

  ├───────────────────┼───────────┼────────────┼───────┤

  │ LM head (2LdV)    │ —         │ 164.7B     │ 4.7%  │

  ├───────────────────┼───────────┼────────────┼───────┤

  │ Total             │ 69.84B    │ 3,516.8B   │ 100%  │

  └───────────────────┴───────────┴────────────┴───────┘

  Total: ≈3.52 TFLOPs

  ---
  (c) Most FLOP-Intensive Component
  
  The FFN (SwiGLU) dominates at 57.5% of total FLOPs, followed by the attention QKV+O projections at 28.6%. Together they account for ~86% of
  all compute. The FFN is the clear bottleneck because its hidden dimension (f = 4288) is 2.68× wider than d_model.

  ---
  (d) Scaling Across GPT-2 Sizes
  
  f values: Small=2048, Medium=2688, Large=3392 (all nearest multiples of 64 to 8/3·d)

  ┌─────────────┬───────────────────┬─────────────────────┬────────────────────┬─────────────────┐

  
  │  Component  │ Small (12L, 768d) │ Medium (24L, 1024d) │ Large (36L, 1280d) │ XL (48L, 1600d) │
  
  ├─────────────┼───────────────────┼─────────────────────┼────────────────────┼─────────────────┤
  
  │ QKV+O proj  │       19.9%       │        25.1%        │       27.3%        │      28.6%      │
  
  ├─────────────┼───────────────────┼─────────────────────┼────────────────────┼─────────────────┤
  
  │ SDPA        │       13.3%       │        12.6%        │       10.9%        │      9.2%       │
  
  ├─────────────┼───────────────────┼─────────────────────┼────────────────────┼─────────────────┤
  
  │ FFN         │       39.8%       │        49.5%        │       54.3%        │      57.5%      │
  
  ├─────────────┼───────────────────┼─────────────────────┼────────────────────┼─────────────────┤
  
  │ LM head     │       27.1%       │        12.8%        │        7.4%        │      4.7%       │
  
  ├─────────────┼───────────────────┼─────────────────────┼────────────────────┼─────────────────┤
  
  │ Total FLOPs │       0.29T       │        0.82T        │       1.77T        │      3.52T      │
  └─────────────┴───────────────────┴─────────────────────┴────────────────────┴─────────────────┘

  Trend: As model size increases, the FFN and QKV+O projections consume proportionally more FLOPs (both scale with d_model² or d_model·d_ff),
  while the LM head and SDPA become proportionally less significant. The LM head is a major contributor only in the smallest model (27%) but
  shrinks rapidly because V is fixed.

  ---
  (e) GPT-2 XL with Context Length 16,384
  
  L increases 16× (1024 → 16384). Since QKV+O and FFN are O(L) but SDPA is O(L²):

  ┌─────────────┬────────┬─────────┐

  
  │  Component  │ L=1024 │ L=16384 │
  
  ├─────────────┼────────┼─────────┤
  
  │ QKV+O proj  │ 28.6%  │ 12.1%   │
  
  ├─────────────┼────────┼─────────┤
  
  │ SDPA        │ 9.2%   │ 61.7%   │
  
  ├─────────────┼────────┼─────────┤
  
  │ FFN         │ 57.5%  │ 24.2%   │
  
  ├─────────────┼────────┼─────────┤
  
  │ LM head     │ 4.7%   │ 2.0%    │
  
  ├─────────────┼────────┼─────────┤
  
  │ Total FLOPs │ 3.52T  │ 133.6T  │
  
  └─────────────┴────────┴─────────┘

  Total FLOPs increases ~38×. The SDPA (Q@K^T) becomes the dominant bottleneck at 61.7% of all FLOPs due to its O(L²) scaling, completely
  overtaking the FFN which dominated at shorter context lengths.



  (a) Peak Memory for AdamW
     
  Parameters (elements, no bias, no weight tying, d_ff = 8d/3):

  P = 2Vd + N(12d² + 2d) + d

  ┌─────────────────────────┬─────────────────────────┐

  │        Component        │ Memory (bytes, float32) │

  ├─────────────────────────┼─────────────────────────┤

  │ Parameters              │ 4P                      │

  ├─────────────────────────┼─────────────────────────┤

  │ Gradients               │ 4P                      │

  ├─────────────────────────┼─────────────────────────┤

  │ Optimizer state (m, v)  │ 8P                      │

  ├─────────────────────────┼─────────────────────────┤

  │ Activations (see below) │ 4·A                     │

  └─────────────────────────┴─────────────────────────┘

  Activations (per block, stored for backward):


  ┌──────────────────────────────────────┬──────────┐

  │            Component                 │ Elements │

  ├──────────────────────────────────────┼──────────┤

  │ RMSNorm1 input                       │ BLd      │

  ├──────────────────────────────────────┼──────────┤

  │ RMSNorm1 output (= QKV input)        │ BLd      │

  ├──────────────────────────────────────┼──────────┤

  │ Q, K, V (for SDPA backward)          │ 3BLd     │

  ├──────────────────────────────────────┼──────────┤

  │ Attention scores (QK^T)              │ BhL²     │

  ├──────────────────────────────────────┼──────────┤

  │ Attention weights (post-softmax)     │ BhL²     │

  ├──────────────────────────────────────┼──────────┤

  │ Context (= input to O_proj)          │ BLd      │

  ├──────────────────────────────────────┼──────────┤

  │ RMSNorm2 input                       │ BLd      │

  ├──────────────────────────────────────┼──────────┤

  │ RMSNorm2 output (= W1/W3 input)      │ BLd      │

  ├──────────────────────────────────────┼──────────┤

  │ W1 output (= SiLU input, not output) │ BL·d_ff  │

  ├──────────────────────────────────────┼──────────┤

  │ W3 output (for multiply backward)    │ BL·d_ff  │

  ├──────────────────────────────────────┼──────────┤

  │ Gated output (= input to W2)         │ BL·d_ff  │

  └──────────────────────────────────────┴──────────┘

  Per block: 8BLd + 3BL·d_ff + 2BhL² = 2BL(8d + Lh) (using d_ff = 8d/3)

  Beyond blocks: RMSNorm_final input (BLd) + LM head input (BLd) + logits (BLV) = BL(2d + V)

  Total activations: A = BL[N·16d + 2NLh + 2d + V]

  Total memory (bytes): M = 16P + 4·BL[16Nd + 2NLh + 2d + V]

  ---
  (b) GPT-2 XL → Batch Size for 80GB
  
  Plug in V=50257, L=1024, N=48, d=1600, h=25, d_ff=4288:

  P = 2·50257·1600 + 48(12·1600² + 2·1600) + 1600 = 1.636B

  ┌─────────────────────┬──────────┐

  │      Component      │  Memory  │

  ├─────────────────────┼──────────┤

  │ Parameters (4P)     │ 6.54 GB  │

  ├─────────────────────┼──────────┤

  │ Gradients (4P)      │ 6.54 GB  │

  ├─────────────────────┼──────────┤

  │ Optimizer m, v (8P) │ 13.08 GB │

  ├─────────────────────┼──────────┤

  │ Subtotal (16P)      │ 26.17 GB │

  └─────────────────────┴──────────┘

  Activations per batch element:
  A/B = 1024[16·48·1600 + 2·48·1024·25 + 2·1600 + 50257] = 1024[3,739,857] = 3.83B elements

  Per batch: 3.83B × 4 bytes = 15.32 GB

  Total: M = 26.17 + 15.32·B GB

  For 80GB: B_max = ⌊(80 − 26.17) / 15.32⌋ = 3

  ---
  (c) AdamW FLOPs per Step
  
  Per parameter, AdamW step operations:

  ┌───────────────────────────────┬─────────────────────────────────────────────┐
  
  │           Operation           │                    FLOPs                    │
  
  ├───────────────────────────────┼─────────────────────────────────────────────┤
  
  │ Weight decay: p ← p − lr·wd·p │ 2 (1 mult + 1 sub)                          │
  
  ├───────────────────────────────┼─────────────────────────────────────────────┤
  
  │ m ← β₁m + (1−β₁)g             │ 3 (2 mult + 1 add)                          │
  
  ├───────────────────────────────┼─────────────────────────────────────────────┤
  
  │ v ← β₂v + (1−β₂)g²            │ 4 (1 square + 2 mult + 1 add)               │
  
  ├───────────────────────────────┼─────────────────────────────────────────────┤
  
  │ Bias correction scalar        │ 0 (negligible)                              │
  
  ├───────────────────────────────┼─────────────────────────────────────────────┤
  
  │ p ← p − α·m̂/(√v̂ + ε)          │ 5 (1 sqrt + 1 add + 1 div + 1 mult + 1 sub) │
  
  ├───────────────────────────────┼─────────────────────────────────────────────┤
  
  │ Total per parameter           │ 14                                          │
  
  └───────────────────────────────┴─────────────────────────────────────────────┘

  AdamW FLOPs = 14P — independent of batch size since optimizer operates on parameters directly.

  For GPT-2 XL: 14 × 1.636B ≈ 22.9 GFLOPs (negligible vs forward+backward).

  ---
  (d) Training Time on Single H100
  
  From problem (b) earlier: forward FLOPs per sample = N(8Ld² + 4L²d + 6Ldf) + 2LdV = 3.52 TFLOPs

  For B=1024: Forward = 3.52 × 1024 = 3,605 TFLOPs

  Backward = 2 × Forward = 7,210 TFLOPs (per problem statement)

  AdamW = 22.9 GFLOPs ≈ 0.023 TFLOPs (negligible)

  Total per step = 10,815 TFLOPs

  H100 at 50% MFU: 495 × 0.50 = 247.5 TFLOPs/s

  Time per step = 10,815 / 247.5 = 43.7 seconds

  400K steps × 43.7s = 17,480,000s ≈ 4,856 hours (≈202 days)

  This is why training GPT-2 XL-scale models requires many GPUs in parallel — 202 days on a single H100 is impractical for iteration.