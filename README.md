# teenygrad+
**Teenygrad with comments and explanations.**


If you want to understand tinygrad, you should read the most minal version teenygrad.
If you want to understand teenygrad, you should read the annotated version teenygrad+.


The original teenygrad has *_zero_* comments! Let's fix that and take teenygrad apart so we understand every bit of it.
```bash
Original Metrics
╭─────────────────────┬───────────╮
│  Total Files        │  9        │
│  Total Directories  │  3        │
│  Total Lines        │  1,259    │
│  Code Lines         │  1,045    │
│  Comment Lines      │  0        │
│  Empty Lines        │  214      │
│  Functions          │  282      │
│  Classes            │  42       │
╰─────────────────────┴───────────╯
```
(_Metrics and terminal overviews created with [PyCodar](https://github.com/QuentinWach/PyCodar)._)

### Get Started
After cloning this repository, you can install the only two dependencies numpy and tqdm, then run the MNIST example to check that this code does indeed work.
```bash
pip install numpy tqdm
python mnist.py
```

### Code
Looking at the core files, we can immediately see that most of it is within the `tensor.py` module with the second largest module being the `mlops.py` module. All others are inredibly small with `realize.py` and `symbolic.py` basically empty.

```bash
File Distribution
╭─────────┬───────────────┬────────┬────────────┬─────────┬─────────┬───────────╮
│  Path   │  File         │  Code  │  Comments  │  Empty  │  Total  │     Size  │
├─────────┼───────────────┼────────┼────────────┼─────────┼─────────┼───────────┤
│  Root   │  mlops.py     │   161  │         0  │     50  │    211  │   8.41KB  │
│  Root   │  realize.py   │     1  │         0  │      0  │      1  │   0.05KB  │
│  Root   │  __init__.py  │     1  │         0  │      0  │      1  │   0.06KB  │
│  Root   │  tensor.py    │   691  │         0  │    133  │    824  │  48.78KB  │
│  Root   │  ops.py       │    13  │         0  │      2  │     15  │   0.86KB  │
│  Root   │  lazy.py      │    63  │         0  │     12  │     75  │   3.37KB  │
│  Root   │  helpers.py   │    81  │         3  │     19  │    103  │   4.57KB  │
│  shape  │  symbolic.py  │     1  │         0  │      0  │      1  │   0.01KB  │
│  nn     │  optim.py     │    61  │         0  │      9  │     70  │   3.49KB  │
╰─────────┴───────────────┴────────┴────────────┴─────────┴─────────┴───────────╯
```
Why is that?



### Files, Function, and Methods
If we go 

```bash
📁 Root
├── 📄 __init__.py
├── 📄 helpers.py
│   ├── 🔷 DType
│   │   └── 🔹 __repr__
│   ├── 🔷 dtypes
│   │   ├── 🔹 is_int
│   │   ├── 🔹 is_float
│   │   ├── 🔹 is_unsigned
│   │   └── 🔹 from_np
│   ├── 🔸 dedup
│   ├── 🔸 argfix
│   ├── 🔸 make_pair
│   ├── 🔸 flatten
│   ├── 🔸 argsort
│   ├── 🔸 all_int
│   ├── 🔸 round_up
│   └── 🔸 getenv
├── 📄 lazy.py
│   ├── 🔷 RawCPUBuffer
│   │   ├── 🔹 __init__
│   │   └── 🔹 toCPU
│   └── 🔷 LazyBuffer
│       ├── 🔹 __init__
│       ├── 🔹 base
│       ├── 🔹 dtype
│       ├── 🔹 realized
│       ├── 🔹 shape
│       ├── 🔹 __repr__
│       ├── 🔹 schedule
│       ├── 🔹 is_unrealized_contiguous_const
│       ├── 🔹 copy_to_device
│       ├── 🔹 fromCPU
│       ├── 🔹 loadop
│       ├── 🔹 contiguous
│       ├── 🔹 const
│       ├── 🔹 cast
│       ├── 🔹 e
│       ├── 🔹 r
│       ├── 🔹 reshape
│       ├── 🔹 expand
│       ├── 🔹 shrink
│       ├── 🔹 permute
│       ├── 🔹 pad
│       └── 🔹 stride
├── 📄 mlops.py
│   ├── 🔷 Contiguous
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 ContiguousBackward
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Cast
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Zero
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Neg
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Sin
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Relu
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Log
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Exp
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Sqrt
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Sigmoid
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Less
│   │   └── 🔹 forward
│   ├── 🔷 Add
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Sub
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Mul
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Div
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Where
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Sum
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Max
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Expand
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Reshape
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Permute
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Pad
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   ├── 🔷 Shrink
│   │   ├── 🔹 forward
│   │   └── 🔹 backward
│   └── 🔷 Flip
│       ├── 🔹 forward
│       └── 🔹 backward
├── 📄 ops.py
│   ├── 🔷 UnaryOps
│   ├── 🔷 BinaryOps
│   ├── 🔷 ReduceOps
│   ├── 🔷 TernaryOps
│   ├── 🔷 MovementOps
│   ├── 🔷 LoadOps
│   └── 🔷 Device
│       └── 🔹 canonicalize
├── 📄 realize.py
│   └── 🔸 run_schedule
├── 📄 tensor.py
│   ├── 🔷 Function
│   │   ├── 🔹 __init__
│   │   ├── 🔹 forward
│   │   ├── 🔹 backward
│   │   └── 🔹 apply
│   ├── 🔷 Tensor
│   │   ├── 🔹 __init__
│   │   ├── 🔹 __repr__
│   │   ├── 🔹 __hash__
│   │   ├── 🔹 device
│   │   ├── 🔹 shape
│   │   ├── 🔹 dtype
│   │   ├── 🔹 corealize
│   │   ├── 🔹 realize
│   │   ├── 🔹 assign
│   │   ├── 🔹 detach
│   │   ├── 🔹 numpy
│   │   ├── 🔹 item
│   │   ├── 🔹 to
│   │   ├── 🔹 to_
│   │   ├── 🔹 _loadop
│   │   ├── 🔹 empty
│   │   ├── 🔹 manual_seed
│   │   ├── 🔹 rand
│   │   ├── 🔹 full
│   │   ├── 🔹 zeros
│   │   ├── 🔹 ones
│   │   ├── 🔹 arange
│   │   ├── 🔹 eye
│   │   ├── 🔹 full_like
│   │   ├── 🔹 zeros_like
│   │   ├── 🔹 ones_like
│   │   ├── 🔹 randn
│   │   ├── 🔹 randint
│   │   ├── 🔹 normal
│   │   ├── 🔹 uniform
│   │   ├── 🔹 scaled_uniform
│   │   ├── 🔹 glorot_uniform
│   │   ├── 🔹 kaiming_uniform
│   │   ├── 🔹 kaiming_normal
│   │   ├── 🔹 multinomial
│   │   ├── 🔹 deepwalk
│   │   ├── 🔹 backward
│   │   ├── 🔹 reshape
│   │   ├── 🔹 expand
│   │   ├── 🔹 permute
│   │   ├── 🔹 flip
│   │   ├── 🔹 shrink
│   │   ├── 🔹 pad
│   │   ├── 🔹 __getitem__
│   │   ├── 🔹 __setitem__
│   │   ├── 🔹 slice
│   │   ├── 🔹 gather
│   │   ├── 🔹 cat
│   │   ├── 🔹 stack
│   │   ├── 🔹 repeat
│   │   ├── 🔹 chunk
│   │   ├── 🔹 squeeze
│   │   ├── 🔹 unsqueeze
│   │   ├── 🔹 pad2d
│   │   ├── 🔹 T
│   │   ├── 🔹 transpose
│   │   ├── 🔹 flatten
│   │   ├── 🔹 _reduce
│   │   ├── 🔹 sum
│   │   ├── 🔹 max
│   │   ├── 🔹 min
│   │   ├── 🔹 mean
│   │   ├── 🔹 std
│   │   ├── 🔹 _softmax
│   │   ├── 🔹 softmax
│   │   ├── 🔹 log_softmax
│   │   ├── 🔹 argmax
│   │   ├── 🔹 argmin
│   │   ├── 🔹 _pool
│   │   ├── 🔹 avg_pool2d
│   │   ├── 🔹 max_pool2d
│   │   ├── 🔹 conv_transpose2d
│   │   ├── 🔹 conv2d
│   │   ├── 🔹 dot
│   │   ├── 🔹 _cumsum
│   │   ├── 🔹 cumsum
│   │   ├── 🔹 _tri
│   │   ├── 🔹 triu
│   │   ├── 🔹 tril
│   │   ├── 🔹 neg
│   │   ├── 🔹 contiguous
│   │   ├── 🔹 contiguous_backward
│   │   ├── 🔹 log
│   │   ├── 🔹 log2
│   │   ├── 🔹 exp
│   │   ├── 🔹 exp2
│   │   ├── 🔹 relu
│   │   ├── 🔹 sigmoid
│   │   ├── 🔹 sin
│   │   ├── 🔹 sqrt
│   │   ├── 🔹 rsqrt
│   │   ├── 🔹 cos
│   │   ├── 🔹 tan
│   │   ├── 🔹 trunc
│   │   ├── 🔹 ceil
│   │   ├── 🔹 floor
│   │   ├── 🔹 square
│   │   ├── 🔹 clip
│   │   ├── 🔹 abs
│   │   ├── 🔹 sign
│   │   ├── 🔹 reciprocal
│   │   ├── 🔹 elu
│   │   ├── 🔹 celu
│   │   ├── 🔹 swish
│   │   ├── 🔹 silu
│   │   ├── 🔹 relu6
│   │   ├── 🔹 hardswish
│   │   ├── 🔹 tanh
│   │   ├── 🔹 sinh
│   │   ├── 🔹 cosh
│   │   ├── 🔹 atanh
│   │   ├── 🔹 asinh
│   │   ├── 🔹 acosh
│   │   ├── 🔹 hardtanh
│   │   ├── 🔹 gelu
│   │   ├── 🔹 quick_gelu
│   │   ├── 🔹 leakyrelu
│   │   ├── 🔹 mish
│   │   ├── 🔹 softplus
│   │   ├── 🔹 softsign
│   │   ├── 🔹 _broadcasted
│   │   ├── 🔹 _to_float
│   │   ├── 🔹 add
│   │   ├── 🔹 sub
│   │   ├── 🔹 mul
│   │   ├── 🔹 div
│   │   ├── 🔹 pow
│   │   ├── 🔹 matmul
│   │   ├── 🔹 maximum
│   │   ├── 🔹 minimum
│   │   ├── 🔹 where
│   │   ├── 🔹 __neg__
│   │   ├── 🔹 __add__
│   │   ├── 🔹 __sub__
│   │   ├── 🔹 __mul__
│   │   ├── 🔹 __pow__
│   │   ├── 🔹 __truediv__
│   │   ├── 🔹 __matmul__
│   │   ├── 🔹 __radd__
│   │   ├── 🔹 __rsub__
│   │   ├── 🔹 __rmul__
│   │   ├── 🔹 __rpow__
│   │   ├── 🔹 __rtruediv__
│   │   ├── 🔹 __rmatmul__
│   │   ├── 🔹 __iadd__
│   │   ├── 🔹 __isub__
│   │   ├── 🔹 __imul__
│   │   ├── 🔹 __ipow__
│   │   ├── 🔹 __itruediv__
│   │   ├── 🔹 __imatmul__
│   │   ├── 🔹 __lt__
│   │   ├── 🔹 __gt__
│   │   ├── 🔹 __ge__
│   │   ├── 🔹 __le__
│   │   ├── 🔹 __ne__
│   │   ├── 🔹 __eq__
│   │   ├── 🔹 linear
│   │   ├── 🔹 sequential
│   │   ├── 🔹 layernorm
│   │   ├── 🔹 batchnorm
│   │   ├── 🔹 dropout
│   │   ├── 🔹 scaled_dot_product_attention
│   │   ├── 🔹 binary_crossentropy
│   │   ├── 🔹 binary_crossentropy_logits
│   │   ├── 🔹 sparse_categorical_crossentropy
│   │   ├── 🔹 cast
│   │   ├── 🔹 bitcast
│   │   ├── 🔹 float
│   │   ├── 🔹 half
│   │   ├── 🔹 ndim
│   │   ├── 🔹 numel
│   │   ├── 🔹 element_size
│   │   ├── 🔹 nbytes
│   │   └── 🔹 is_floating_point
│   ├── 🔷 train
│   │   ├── 🔹 __init__
│   │   ├── 🔹 __enter__
│   │   └── 🔹 __exit__
│   ├── 🔸 _deepwalk
│   ├── 🔸 normalize_int
│   ├── 🔸 apply_matrix
│   └── 🔸 fix
├── 📁 nn
│   └── 📄 optim.py
│       ├── 🔷 Optimizer
│       │   ├── 🔹 __init__
│       │   ├── 🔹 zero_grad
│       │   └── 🔹 realize
│       ├── 🔷 SGD
│       │   ├── 🔹 __init__
│       │   └── 🔹 step
│       ├── 🔷 LAMB
│       │   ├── 🔹 __init__
│       │   └── 🔹 step
│       ├── 🔸 AdamW
│       └── 🔸 Adam
└── 📁 shape
    └── 📄 symbolic.py
```
