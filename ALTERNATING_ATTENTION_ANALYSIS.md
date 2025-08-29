# ModernBERT 交替注意力机制实现分析

## 概述

ModernBERT 博客中提到的交替注意力（Alternating Attention）机制是该模型的核心创新之一。该机制通过在不同层之间交替使用全局注意力（Global Attention）和局部注意力（Local Attention）来实现长序列处理的效率提升。

## 核心实现原理

### 1. 配置参数

交替注意力机制通过以下两个关键配置参数控制：

- **`sliding_window`**: 局部注意力的窗口大小
  - 默认值：-1（禁用）
  - 示例：128（表示每个token关注其周围128个token，左右各64个）
  
- **`global_attn_every_n_layers`**: 全局注意力的频率
  - 默认值：-1（禁用）
  - 示例：3（表示每3层使用一次全局注意力）

### 2. 实现逻辑

在所有注意力类的 `__init__` 方法中，都包含以下核心逻辑：

```python
if config.global_attn_every_n_layers > 0:
    if config.sliding_window == -1:
        raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
    if layer_id % config.global_attn_every_n_layers != 0:
        # 使用局部注意力
        self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
    else:
        # 使用全局注意力
        self.sliding_window = (-1, -1)
else:
    # 所有层都使用局部注意力
    self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
```

### 3. 决策规则

- **全局注意力层**: `layer_id % global_attn_every_n_layers == 0`
  - 设置 `sliding_window = (-1, -1)`，表示可以关注所有token
  
- **局部注意力层**: `layer_id % global_attn_every_n_layers != 0`
  - 设置 `sliding_window = (窗口大小/2, 窗口大小/2)`，表示只关注邻近token

## 具体实现文件

### 1. 配置文件 (`src/bert_layers/configuration_bert.py`)

```python
class FlexBertConfig(TransformersBertConfig):
    def __init__(
        self,
        # ... 其他参数
        sliding_window: int = -1,                    # 滑动窗口大小
        global_attn_every_n_layers: int = -1,       # 全局注意力频率
        local_attn_rotary_emb_base: float = -1,     # 局部注意力的旋转嵌入基数
        local_attn_rotary_emb_dim: int | None = None, # 局部注意力的旋转嵌入维度
        # ... 其他参数
    ):
```

配置验证逻辑：
```python
# 确保全局注意力频率是合理的
if global_attn_every_n_layers > 0 and (self.num_hidden_layers - 1) % global_attn_every_n_layers != 0:
    raise ValueError(f"{global_attn_every_n_layers=} must be a divisor of one less than {self.num_hidden_layers=}")

# 滑动窗口验证
if self.sliding_window != -1:
    if not self.use_fa2:
        raise ValueError("Sliding window attention is only supported with FlashAttention2")
    if self.sliding_window % 2 != 0 and self.sliding_window % 64 != 0:
        raise ValueError(f"Sliding window must be an even number and divisible by 64")
```

### 2. 注意力实现 (`src/bert_layers/attention.py`)

该文件包含多个注意力类，都实现了相同的交替逻辑：

- `FlexBertPaddedAttention`
- `FlexBertPaddedParallelAttention`
- `FlexBertUnpadAttention`
- `FlexBertUnpadParallelAttention`
- `FlexBertPaddedRopeAttention`
- `FlexBertPaddedRopeParallelAttention`
- `FlexBertUnpadRopeAttention`
- `FlexBertUnpadRopeParallelAttention`

每个类在前向传播时都会使用 Flash Attention：

```python
# 使用 Flash Attention 2 实现滑动窗口
attn = flash_attn_qkvpacked_func(
    qkv,
    dropout_p=self.p_dropout,
    deterministic=self.deterministic_fa2,
    window_size=self.sliding_window,  # 关键参数
)
```

### 3. Flash Attention 集成

实现依赖于 Flash Attention 2 库：

```python
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func
```

窗口大小的含义：
- `window_size=(-1, -1)`: 全局注意力，关注所有token
- `window_size=(64, 64)`: 局部注意力，关注左右各64个token（总共128个）

## 配置示例

### 实际配置文件示例

从 `yamls/main/flex-bert-base.yaml`:

```yaml
model_config:
  num_hidden_layers: 12        # 总共12层
  sliding_window: 128          # 滑动窗口大小为128
  global_attn_every_n_layers: 3  # 每3层使用全局注意力
```

这个配置的注意力模式：
- 第0层：全局注意力 (0 % 3 == 0)
- 第1层：局部注意力 (1 % 3 != 0)，窗口大小 (64, 64)
- 第2层：局部注意力 (2 % 3 != 0)，窗口大小 (64, 64)
- 第3层：全局注意力 (3 % 3 == 0)
- 第4层：局部注意力 (4 % 3 != 0)，窗口大小 (64, 64)
- 第5层：局部注意力 (5 % 3 != 0)，窗口大小 (64, 64)
- 第6层：全局注意力 (6 % 3 == 0)
- 第7层：局部注意力 (7 % 3 != 0)，窗口大小 (64, 64)
- 第8层：局部注意力 (8 % 3 != 0)，窗口大小 (64, 64)
- 第9层：全局注意力 (9 % 3 == 0)
- 第10层：局部注意力 (10 % 3 != 0)，窗口大小 (64, 64)
- 第11层：局部注意力 (11 % 3 != 0)，窗口大小 (64, 64)

## 性能优势

### 1. 计算复杂度降低

- **全局注意力**: O(n²) 复杂度，其中 n 是序列长度
- **局部注意力**: O(n × w) 复杂度，其中 w 是窗口大小（128）

对于长序列，局部注意力显著降低了计算成本。

### 2. 信息传播机制

- **全局注意力层**: 允许远距离token之间的信息交换
- **局部注意力层**: 专注于局部上下文建模
- **交替模式**: 既保证了全局信息传播，又降低了整体计算成本

### 3. 博客中提到的效果

正如博客所述："every token only attends to the 128 tokens nearest to itself (local attention)"，这正是通过 `sliding_window: 128` 配置实现的。

## 技术细节

### 1. Flash Attention 2 依赖

滑动窗口注意力只在 Flash Attention 2 后端支持：

```python
if self.sliding_window[0] > 0:
    raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")
```

### 2. 旋转位置编码优化

对于使用 RoPE 的注意力层，局部注意力可以使用不同的旋转嵌入参数：

```python
rotary_base = config.rotary_emb_base
rotary_dim = config.rotary_emb_dim
if self.sliding_window != (-1, -1):
    if config.local_attn_rotary_emb_base != -1:
        rotary_base = config.local_attn_rotary_emb_base
    if config.local_attn_rotary_emb_dim is not None:
        rotary_dim = config.local_attn_rotary_emb_dim
```

### 3. 测试覆盖

在 `tests/test_sdpa_fa2.py` 中有全面的测试：

```python
if sliding_window:
    config.model.model_config.sliding_window = 64
    config.model.model_config.num_hidden_layers = 3
    config.model.model_config.global_attn_every_n_layers = random.choice([-1, 2])
```

## 总结

ModernBERT 的交替注意力机制是一个精巧的设计，通过简单的模运算决定每层使用哪种注意力类型，结合 Flash Attention 2 的滑动窗口功能，实现了在保持模型表现力的同时显著提升长序列处理效率的目标。这个实现充分体现了"simple but effective"的设计理念。