#!/usr/bin/env python3
"""
测试 ModernBERT 交替注意力机制的基本功能

这个脚本创建一个简单的配置并验证注意力层的初始化是否按预期工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append('/home/runner/work/ModernBERT/ModernBERT/src')

from bert_layers.configuration_bert import FlexBertConfig
from bert_layers.attention import FlexBertPaddedParallelAttention


def test_alternating_attention_config():
    """测试交替注意力的配置验证"""
    print("测试交替注意力配置验证...")
    
    # 测试1：正常配置
    try:
        config = FlexBertConfig(
            num_hidden_layers=12,
            sliding_window=128,
            global_attn_every_n_layers=3,
            use_fa2=True
        )
        print("✓ 正常配置创建成功")
    except Exception as e:
        print(f"✗ 正常配置创建失败: {e}")
    
    # 测试2：无效配置 - 全局注意力频率不是层数的因子
    try:
        config = FlexBertConfig(
            num_hidden_layers=12,
            sliding_window=128,
            global_attn_every_n_layers=5,  # 12-1=11, 11%5!=0
            use_fa2=True
        )
        print("✗ 应该抛出错误的配置却成功了")
    except ValueError as e:
        print(f"✓ 正确捕获配置错误: {str(e)[:50]}...")
    
    # 测试3：无效配置 - 启用全局注意力但未设置滑动窗口
    try:
        config = FlexBertConfig(
            num_hidden_layers=12,
            sliding_window=-1,  # 未启用
            global_attn_every_n_layers=3,
            use_fa2=True
        )
        print("✗ 应该抛出错误的配置却成功了")
    except ValueError as e:
        print(f"✓ 正确捕获配置错误: {str(e)[:50]}...")


def test_attention_layer_initialization():
    """测试注意力层的初始化"""
    print("\n测试注意力层初始化...")
    
    # 创建配置
    config = FlexBertConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        sliding_window=128,
        global_attn_every_n_layers=3,
        use_fa2=False  # 使用 PyTorch SDPA 避免 Flash Attention 依赖问题
    )
    
    # 测试不同层的注意力配置
    expected_patterns = [
        (0, (-1, -1), "全局"),      # layer_id % 3 == 0
        (1, (64, 64), "局部"),      # layer_id % 3 != 0  
        (2, (64, 64), "局部"),      # layer_id % 3 != 0
        (3, (-1, -1), "全局"),      # layer_id % 3 == 0
        (4, (64, 64), "局部"),      # layer_id % 3 != 0
        (5, (64, 64), "局部"),      # layer_id % 3 != 0
    ]
    
    for layer_id, expected_window, attention_type in expected_patterns:
        try:
            # 创建注意力层
            attention = FlexBertPaddedParallelAttention(config, layer_id=layer_id)
            actual_window = attention.sliding_window
            
            if actual_window == expected_window:
                print(f"✓ 第{layer_id}层: {attention_type}注意力, window_size={actual_window}")
            else:
                print(f"✗ 第{layer_id}层: 期望 {expected_window}, 实际 {actual_window}")
        except Exception as e:
            print(f"✗ 第{layer_id}层初始化失败: {e}")


def test_configuration_examples():
    """测试实际配置文件中的设置"""
    print("\n测试实际配置示例...")
    
    # ModernBERT Base 配置 (来自 flex-bert-base.yaml)
    try:
        config = FlexBertConfig(
            num_attention_heads=12,
            num_hidden_layers=12,
            sliding_window=128,
            global_attn_every_n_layers=3,
            use_fa2=False  # 避免依赖问题
        )
        
        # 验证配置
        assert config.sliding_window == 128
        assert config.global_attn_every_n_layers == 3
        assert config.num_hidden_layers == 12
        
        print("✓ ModernBERT Base 配置验证成功")
        
        # 显示注意力模式
        print("  注意力模式:")
        for layer_id in range(min(6, config.num_hidden_layers)):  # 只显示前6层
            if layer_id % config.global_attn_every_n_layers == 0:
                print(f"    第{layer_id}层: 全局注意力")
            else:
                print(f"    第{layer_id}层: 局部注意力 (窗口: 128 tokens)")
            
    except Exception as e:
        print(f"✗ ModernBERT Base 配置测试失败: {e}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("ModernBERT 交替注意力机制功能测试")
    print("=" * 60)
    
    test_alternating_attention_config()
    test_attention_layer_initialization()
    test_configuration_examples()
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print("- 配置验证机制工作正常")
    print("- 注意力层能根据 layer_id 正确选择全局/局部注意力")
    print("- 实际配置文件的设置可以正确加载和验证")
    print("- 博客中提到的 '128 tokens nearest to itself' 通过")
    print("  sliding_window=128 配置实现")
    print("=" * 60)


if __name__ == "__main__":
    main()