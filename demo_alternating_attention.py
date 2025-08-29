#!/usr/bin/env python3
"""
ModernBERT 交替注意力机制演示脚本

这个脚本演示了 ModernBERT 如何决定每一层使用全局注意力还是局部注意力。
"""

def demonstrate_alternating_attention(num_layers: int, global_attn_every_n_layers: int, sliding_window: int):
    """
    演示交替注意力的决策逻辑
    
    Args:
        num_layers: 总层数
        global_attn_every_n_layers: 全局注意力频率
        sliding_window: 滑动窗口大小
    """
    print(f"ModernBERT 交替注意力配置演示")
    print(f"=" * 50)
    print(f"总层数: {num_layers}")
    print(f"全局注意力频率: 每 {global_attn_every_n_layers} 层")
    print(f"滑动窗口大小: {sliding_window}")
    print(f"=" * 50)
    
    global_layers = []
    local_layers = []
    
    for layer_id in range(num_layers):
        # 这是 ModernBERT 源码中的核心逻辑
        if global_attn_every_n_layers > 0:
            if layer_id % global_attn_every_n_layers == 0:
                # 全局注意力
                window_size = (-1, -1)
                attention_type = "全局注意力"
                global_layers.append(layer_id)
            else:
                # 局部注意力
                window_size = (sliding_window // 2, sliding_window // 2)
                attention_type = "局部注意力"
                local_layers.append(layer_id)
        else:
            # 所有层都使用局部注意力
            window_size = (sliding_window // 2, sliding_window // 2)
            attention_type = "局部注意力"
            local_layers.append(layer_id)
        
        print(f"第 {layer_id:2d} 层: {attention_type:6s} | window_size={window_size}")
    
    print(f"\n" + "=" * 50)
    print(f"统计信息:")
    print(f"全局注意力层 ({len(global_layers)} 层): {global_layers}")
    print(f"局部注意力层 ({len(local_layers)} 层): {local_layers}")
    
    if global_layers:
        global_ratio = len(global_layers) / num_layers * 100
        print(f"全局注意力比例: {global_ratio:.1f}%")
        print(f"计算复杂度降低估算: 局部注意力层节省 ~{100 - sliding_window/1024*100:.1f}% 计算量 (假设序列长度1024)")


def main():
    """主函数：演示不同配置下的注意力模式"""
    
    print("=" * 80)
    print("ModernBERT 交替注意力机制实现演示")
    print("=" * 80)
    
    # 示例1：ModernBERT Base 配置（来自 flex-bert-base.yaml）
    print("\n【示例 1】ModernBERT Base 配置")
    demonstrate_alternating_attention(
        num_layers=12,
        global_attn_every_n_layers=3,
        sliding_window=128
    )
    
    # 示例2：更密集的全局注意力
    print("\n【示例 2】更密集的全局注意力 (每2层)")
    demonstrate_alternating_attention(
        num_layers=12,
        global_attn_every_n_layers=2,
        sliding_window=128
    )
    
    # 示例3：较少的全局注意力
    print("\n【示例 3】较少的全局注意力 (每4层)")
    demonstrate_alternating_attention(
        num_layers=12,
        global_attn_every_n_layers=4,
        sliding_window=128
    )
    
    # 示例4：纯局部注意力
    print("\n【示例 4】纯局部注意力 (禁用全局注意力)")
    demonstrate_alternating_attention(
        num_layers=12,
        global_attn_every_n_layers=-1,  # 禁用
        sliding_window=128
    )
    
    print("\n" + "=" * 80)
    print("核心实现代码片段 (来自 src/bert_layers/attention.py):")
    print("=" * 80)
    print("""
# ModernBERT 源码中的核心决策逻辑
if config.global_attn_every_n_layers > 0:
    if config.sliding_window == -1:
        raise ValueError("global_attn_every_n_layers requires sliding_window to be set")
    if layer_id % config.global_attn_every_n_layers != 0:
        # 局部注意力：窗口大小分为左右两部分
        self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
    else:
        # 全局注意力：(-1, -1) 表示关注所有 token
        self.sliding_window = (-1, -1)
else:
    # 所有层都使用局部注意力
    self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

# 在前向传播中使用 Flash Attention
attn = flash_attn_qkvpacked_func(
    qkv,
    dropout_p=self.p_dropout,
    deterministic=self.deterministic_fa2,
    window_size=self.sliding_window,  # 这里应用窗口大小
)
    """)
    
    print("\n" + "=" * 80)
    print("说明:")
    print("- window_size=(-1, -1): 全局注意力，每个 token 可以关注序列中的所有 token")
    print("- window_size=(64, 64): 局部注意力，每个 token 只关注左右各 64 个 token (总共 128 个)")
    print("- 这种交替模式在保持全局信息传播的同时，显著降低了计算复杂度")
    print("- 博客中提到的 '128 tokens nearest to itself' 就是通过 sliding_window=128 实现的")
    print("=" * 80)


if __name__ == "__main__":
    main()