"""
Q3 演示：为什么同一问题答案不同？—— LLM 生成机制

核心：LLM 不是"事实查询机"，而是"概率预测机"。
每次生成 = 下一 token 的概率分布 → 按策略采样 → 结果可能不同。

三个演示：
  Part 1: 模拟生成过程 —— 逐 token 预测 + 采样
  Part 2: Temperature 效果对比 —— 如何控制随机性
  Part 3: Top-p / Top-k 采样 —— 更精细的随机控制
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import math


def sep(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def demo_1_sampling_process():
    """
    模拟 LLM 逐 token 生成的过程，展示为什么每次结果不同
    """
    sep("Part 1: LLM 生成过程 —— 为什么每次答案不一样")

    # 模拟词表（简化版）
    vocab = ["的", "一", "是", "在", "不", "了", "有", "和", "人", "这",
             "2", "两", "二", "贰", "俩", "亲", "。", "，", "！", "\n"]

    # 假设输入是 "1+1="，模型输出了下一个 token 的概率分布
    torch.manual_seed(42)

    # 模拟 logits（模型的原始输出）
    print("\n--- 场景: 用户问 '1+1=?' ---")
    print(f"\n输入: '1+1='")

    logits = torch.tensor([0.1, 0.2, 0.3, 0.05, 0.02, 0.1, 0.1, 0.05,
                           0.02, 0.05, 8.5, 1.5, 0.8, 0.3, 0.1, 0.4, 1.2,
                           0.3, 0.1, 0.6])  # "2" 的 logit 最高

    probs = F.softmax(logits, dim=-1)

    # 展示概率分布
    print(f"\n下一个 Token 的概率分布 (Top 5):")
    sorted_idx = torch.argsort(probs, descending=True)
    for rank, idx in enumerate(sorted_idx[:5]):
        bar = "█" * int(probs[idx] * 50)
        print(f"  {rank + 1}. '{vocab[idx]}'  {probs[idx]:.4f}  {bar}")

    # 模拟多次采样
    print(f"\n运行 5 次生成过程 (Temperature=1.0):")
    print(f"{'次数':>4s}  {'第1个Token':>8s}  {'第2个Token':>8s}  {'第3个Token':>8s}  {'完整输出'}")
    print(f"{'─' * 4}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 20}")

    for run in range(5):
        torch.manual_seed(100 + run)

        # 模拟 LLM 的重复预测过程
        tokens = []
        logits_seq = logits.clone()  # 初始 prompt
        for _ in range(3):
            probs = F.softmax(logits_seq, dim=-1)
            next_id = torch.multinomial(probs, 1).item()  # 按概率采样
            tokens.append(vocab[next_id])
            # 简化：下一次的 logits 随机变化一下（真实中是模型重新计算的）
            logits_seq = logits_seq + torch.randn_like(logits_seq) * 0.3

        print(f"  {run + 1:>4d}  {tokens[0]:>8s}  {tokens[1]:>8s}  {tokens[2]:>8s}  "
              f"{''.join(tokens)}")

    print(f"\n关键洞察:")
    print(f"  1. LLM 不是查数据库——它每次都重新计算概率分布")
    print('  2. 即使最高概率的 token 是 "2"，仍有 ~15% 概率选别的')
    print(f"  3. 每一步都基于上一步的结果（自回归），误差会累积")


def demo_2_temperature():
    """
    演示 Temperature 如何影响生成多样性
    """
    sep('Part 2: Temperature -- 随机性的"旋钮"')

    # 模拟一个概率分布
    torch.manual_seed(123)
    logits = torch.tensor([2.0, 0.8, 0.5, -0.5, -2.0])
    tokens = ["2", "两", "二", "贰", "俩"]

    print(f"\n原始 Logits:")
    for t, l in zip(tokens, logits):
        print(f"  '{t}': {l:.2f}")

    print(f"\n{'Temp':>6s}  ", end="")
    for t in tokens:
        print(f"{t:>8s}", end="")
    print(f"  {'最高概率':>10s}  {'说明'}")
    print(f"{'─' * 6}  {'─' * 56}  {'─' * 20}")

    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)

        print(f"  {temp:<6.1f}", end="")
        for p in probs:
            print(f"  {p:.4f}", end="")
        max_p = probs.max().item()
        desc = ""
        if temp <= 0.5:
            desc = "几乎总是选 '2'"
        elif temp <= 1.5:
            desc = "正常多样性"
        else:
            desc = "随机性很高"
        print(f"  {max_p:.4f}      {desc}")

    # 模拟多次实验
    print(f"\n模拟 100 次生成，不同 Temperature 下第一个 token 的选择分布:")
    print(f"{'Temp':>6s}  {'2':>6s}  {'两':>6s}  {'二':>6s}  {'贰':>6s}  {'俩':>6s}")
    print(f"{'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 6}")

    for temp in [0.1, 0.5, 1.0, 2.0]:
        counts = {t: 0 for t in tokens}
        for i in range(100):
            torch.manual_seed(i)
            scaled = logits / temp
            probs = F.softmax(scaled, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            counts[tokens[idx]] += 1
        print(f"  {temp:<6.1f}", end="")
        for t in tokens:
            print(f"  {counts[t]:>4d}", end="")
        print()

    print(f"\n公式: P(token_i) = softmax(logits_i / T)")
    print(f"  T → 0:   概率分布变得极端 → 总选最高概率 → 确定性输出")
    print(f"  T = 1:   原始概率分布 → 标准采样")
    print(f"  T → ∞:   概率分布趋于均匀 → 完全随机")


def demo_3_top_p_top_k():
    """
    演示 Top-p / Top-k 采样策略
    """
    sep("Part 3: Top-p & Top-k —— 更精细的随机控制")

    # 模拟大词表中的概率分布（模拟 20 个候选）
    torch.manual_seed(456)
    logits = torch.randn(20) * 1.5
    logits[3] = 3.0  # 让一个 token 明显突出
    logits[7] = 2.0

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    print(f"\n候选 token 的概率分布 (top 8):")
    print(f"  {'Rank':>4s}  {'概率':>8s}  {'累计概率':>8s}  条形图")
    print(f"  {'─' * 4}  {'─' * 8}  {'─' * 8}  {'─' * 25}")
    cumsum = 0
    for rank in range(min(8, len(sorted_probs))):
        p = sorted_probs[rank].item()
        cumsum += p
        bar = "█" * int(p * 40)
        print(f"  {rank + 1:>4d}  {p:>8.4f}  {cumsum:>8.4f}  {bar}")

    # Top-p 演示
    print(f"\n--- Top-p (Nucleus Sampling) ---")
    print(f"  Top-p=0.5: 只从累计概率达 50% 的候选中采样")
    print(f"  Top-p=0.9: 只从累计概率达 90% 的候选中采样 → 截断尾部低概率噪声")

    for top_p in [0.5, 0.9, 1.0]:
        cumsum = torch.cumsum(sorted_probs, dim=0)
        cutoff = (cumsum >= top_p).nonzero(as_tuple=True)[0][0] + 1
        print(f"  Top-p={top_p}: 保留前 {cutoff} 个候选")

    # Top-k 演示
    print(f"\n--- Top-k ---")
    print(f"  Top-k=3: 只从概率最高的 3 个候选中采样")

    for top_k in [1, 3, 10, 20]:
        print(f"  Top-k={top_k:>2d}: 候选数={min(top_k, 20)}")

    # 组合使用
    print(f"\n--- 生产环境推荐 ---")
    print(f"  Temperature=0.7 + Top-p=0.9 + Top-k=50")
    print(f"  这个组合既保证质量，又有适度多样性")
    print(f"\n为什么不用 Temperature=0 (完全确定)？")
    print(f"  1. 容易陷入重复循环 (A→B→A→B...)")
    print(f"  2. 创造性任务 (写作/头脑风暴) 需要随机性")
    print(f"  3. 代码生成 / 数学计算 → 可以用低 Temperature")


def run_q3():
    print("=" * 55)
    print("  Q3: 为什么同一问题答案不同？—— LLM 生成机制")
    print("=" * 55)
    demo_1_sampling_process()
    demo_2_temperature()
    demo_3_top_p_top_k()
    print(f"\n{'=' * 55}")
    print("  Q3 Demo 完成")
    print("=" * 55)


if __name__ == "__main__":
    run_q3()
