"""
Q2 演示：DeepSeek/Kimi/Qwen 的核心区别 —— Transformer + RoPE + MoE

四个演示模块：
  Part 1: Self-Attention —— Q/K/V 矩阵 + 注意力权重可视化
  Part 2: 为什么除以 √d —— 数值稳定性推导
  Part 3: RoPE 旋转位置编码 —— 位置信息的几何直觉
  Part 4: MoE 混合专家 —— DeepSeek 为什么便宜
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


# ============================================================
# Part 1: Self-Attention —— Q/K/V 矩阵
# ============================================================
def demo_1_self_attention():
    sep('Part 1: Self-Attention -- 每个 Token 都在"开会"')

    # 模拟 4 个 token，每个用 8 维向量表示
    # 句子: "北京 今天 天气 好"
    seq_len = 4
    d_model = 8
    torch.manual_seed(42)

    # 随机初始化 token 向量（实际中是 Embedding 输出）
    X = torch.randn(seq_len, d_model)
    tokens = ["北京", "今天", "天气", "好"]

    # ---- 演示：直接点积 vs 线性变换后点积 ----
    # Q/K/V 的本质：通过可学习的线性变换 W_Q/W_K/W_V 把输入 X 映射到查询/键/值空间
    W_Q = torch.randn(d_model, d_model) * 0.1
    W_K = torch.randn(d_model, d_model) * 0.1
    W_V = torch.randn(d_model, d_model) * 0.1

    Q = X @ W_Q  # (4, 8) —— 每个 token 的"查询"向量："我在找什么？"
    K = X @ W_K  # (4, 8) —— 每个 token 的"键"向量："我能提供什么？"
    V = X @ W_V  # (4, 8) —— 每个 token 的"值"向量："我实际给你的信息"

    # ---- Attention Score = Q × K^T ----
    scores = (Q @ K.T) / math.sqrt(d_model)  # 除以 √d 防止 softmax 饱和
    attn_weights = F.softmax(scores, dim=-1)  # 每行归一化：一个 token 对所有 token 的关注度

    # ---- 输出 = attention_weights × V ----
    output = attn_weights @ V

    # ---- 可视化 ----
    print("\nQ @ K^T / √d = 注意力分数矩阵 (4×4):")
    print(f"         {'|'.join(f'{t:^8s}' for t in tokens)}")
    for i, t in enumerate(tokens):
        row = "  ".join(f"{scores[i][j]:>8.4f}" for j in range(seq_len))
        print(f"  {t:<4s} [{row}]")

    print(f"\nSoftmax 后 → 注意力权重 (每行之和 ≈ 1):")
    print(f"         {'|'.join(f'{t:^8s}' for t in tokens)}")
    for i, t in enumerate(tokens):
        row = "  ".join(f"{attn_weights[i][j]:>8.4f}" for j in range(seq_len))
        print(f"  {t:<4s} [{row}]")

    # ---- 解读 ----
    print(f"\n解读: ")
    print('  - "今天" 最关注谁？看它那行最大的权重值')
    print(f"  - 对角线通常较大（每个词最关注自己）")
    print('  - 注意力的本质：让每个词根据语义，决定"听谁的话"')

    # ---- Multi-Head 概念 ----
    print(f"\nMulti-Head Attention: 上面的过程 × N 个头并行执行")
    print(f"  头 1: 可能关注语法结构  (主语-谓语)")
    print(f"  头 2: 可能关注语义关系  (天气-好)")
    print(f"  头 3: 可能关注位置信息  (前文-后文)")
    print(f"  ...")
    print(f"  最后把所有头的输出拼接起来 → 完整理解")


# ============================================================
# Part 2: 为什么除以 √d
# ============================================================
def demo_2_scale_dot():
    sep("Part 2: 为什么 Q·K^T 要除以 √d？")

    print("\n如果不除以 √d，会发生什么？")

    d = 64  # 常见注意力头维度
    torch.manual_seed(123)

    # 随机初始化 Q 和 K（每个元素 ~ N(0,1)）
    Q = torch.randn(1, d)
    K = torch.randn(1, d)

    raw_score = (Q @ K.T).item()
    scaled_score = raw_score / math.sqrt(d)

    print(f"\n  d = {d}")
    print(f"  Q·K^T (原始)        = {raw_score:.2f}")
    print(f"  Q·K^T / √d         = {scaled_score:.2f}")
    print(f"  softmax(Q·K^T)     ≈ {torch.softmax(torch.tensor([raw_score, 0, 0, 0]), dim=0)}")
    print(f"  softmax(Q·K^T/√d)  ≈ {torch.softmax(torch.tensor([scaled_score, 0, 0, 0]), dim=0)}")

    # 统计实验：多个随机向量的点积方差
    print(f"\n统计实验 (10000 次):")
    scores_no_scale = []
    scores_scale = []

    for _ in range(10000):
        q = torch.randn(d)
        k = torch.randn(d)
        scores_no_scale.append((q @ k).item())
        scores_scale.append((q @ k / math.sqrt(d)).item())

    scores_no_scale = torch.tensor(scores_no_scale)
    scores_scale = torch.tensor(scores_scale)

    print(f"  不除 √d: 均值={scores_no_scale.mean():.2f}, 方差={scores_no_scale.var():.2f}")
    print(f"  除以 √d: 均值={scores_scale.mean():.2f}, 方差={scores_scale.var():.2f}")

    print(f"\n结论:")
    print(f"  不除 √d → 方差 ≈ {d} = {d} → softmax 输出极端(接近 one-hot)")
    print(f"  除以 √d → 方差 ≈ 1 → softmax 输出平缓，梯度能正常传播")
    print(f"  除以 √d 本质是【归一化】，让注意力分布保持在合理范围")


# ============================================================
# Part 3: RoPE —— 旋转位置编码
# ============================================================
def demo_3_rope():
    sep("Part 3: RoPE —— 旋转位置编码（Kimi 长上下文的秘密）")

    print("\n核心思想: 用旋转角度编码位置 → 相对位置 = 角度差 → 天然可外推")

    d = 2  # 用 2 维方便可视化（实际中高维也按对分组旋转）

    # ---- 3.1 单个位置的旋转演示 ----
    print(f"\n--- Sub 1: 旋转是如何编码位置的 ---")
    pos_0 = 0
    pos_1 = 1
    pos_2 = 2

    # 基础频率 θ_i = 1 / 10000^(2i/d)，这里 d=2 简化为一个固定角度
    theta = 1.0  # 简化的旋转角度步长

    # 原始向量（位置无关的语义向量）
    x = torch.tensor([1.0, 0.0])  # 指向正右

    # RoPE 对每个位置施加旋转
    def rotate(vec, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)
        return torch.tensor([
            vec[0] * cos - vec[1] * sin,
            vec[0] * sin + vec[1] * cos
        ])

    x0 = rotate(x, pos_0 * theta)
    x1 = rotate(x, pos_1 * theta)
    x2 = rotate(x, pos_2 * theta)

    print(f"\n  原始向量 x = [1.0, 0.0]")
    print(f"  位置 0 (角度=0.00 rad): [{x0[0]:.3f}, {x0[1]:.3f}]")
    print(f"  位置 1 (角度=1.00 rad): [{x1[0]:.3f}, {x1[1]:.3f}]")
    print(f"  位置 2 (角度=2.00 rad): [{x2[0]:.3f}, {x2[1]:.3f}]")

    # ---- 3.2 相对位置是关键 ----
    print(f"\n--- Sub 2: 相对位置 = 角度差 ---")

    # 两个 token 之间的注意力分数 = Q_pos1 · K_pos2
    # 在 RoPE 下：Q·K = (R(pos1)·x_q) · (R(pos2)·x_k)
    #               只依赖 |pos1 - pos2|（旋转矩阵的正交性质）

    q_base = torch.tensor([1.0, 0.0])
    k_base = torch.tensor([0.8, 0.6])

    pairs = [
        ("猫(位置0) → 老鼠(位置2)", 0, 2),
        ("老鼠(位置0) → 猫(位置2)", 0, 2),
        ("猫(位置0) → 老鼠(位置1)", 0, 1),
    ]

    print(f"\n  Q_基 = {q_base.tolist()},  K_基 = {k_base.tolist()}")
    print(f"\n  {'场景':<25s}  {'Q·K (点积)':>10s}  说明")
    print(f"  {'─' * 25}  {'─' * 10}  {'─' * 20}")

    for label, pos_q, pos_k in pairs:
        q_rot = rotate(q_base, pos_q * theta)
        k_rot = rotate(k_base, pos_k * theta)
        score = torch.dot(q_rot, k_rot).item()
        note = "相对距离=2" if abs(pos_q - pos_k) == 2 else "相对距离=1"
        print(f"  {label:<23s}  {score:>10.4f}  {note}")

    print(f"\n观察:")
    print("  '猫->老鼠'和'老鼠->猫'的 attention score 相同 -> 只依赖相对位置 |pos1-pos2|")
    print(f"  这就是 RoPE 能外推的关键：不记绝对位置，只记相对关系")

    # ---- 3.3 RoPE 为什么支持外推 ----
    print(f"\n--- Sub 3: RoPE vs 绝对位置编码 → 长上下文外推 ---")

    print(f"\n  绝对位置编码:")
    print(f"    训练时 max_position = 512")
    print(f"    位置 513 没有对应的编码向量 → 直接崩溃")

    print(f"\n  RoPE:")
    print(f"    训练时只见过长度 512")
    print(f"    但相对位置 513 的角度差 = (pos_513 - pos_0) × θ")
    print('    模型虽然没见过这个具体角度，但已经学过了"旋转"的模式')
    print(f"    所以可以外推 — 就像学会了加法就能算更大的数")
    print(f"\n  Kimi 正是在 RoPE 外推上做了优化，才能支持 200 万上下文")


# ============================================================
# Part 4: MoE —— DeepSeek 为什么便宜
# ============================================================
def demo_4_moe():
    sep("Part 4: MoE —— DeepSeek 的省钱秘诀")

    print("\n传统 Dense 模型: 每个 token 经过所有参数")
    print("  Qwen-7B: 7B 参数 × 1 次 = 7B 计算量 / token")
    print("")
    print("MoE 模型: 每个 token 只经过少数专家")
    print("  DeepSeek-V3: 671B 总参数，每次只激活 37B")
    print("  671B ÷ 37B ≈ 18 → 推理只用 1/18 的计算量")

    # ---- 模拟 MoE 路由器 ----
    print(f"\n--- 模拟: MoE 路由器如何选择专家 ---")
    torch.manual_seed(42)

    num_experts = 4
    top_k = 2         # 每次激活 2 个专家
    d_model = 8

    # 模拟一个 token 向量 (经过 Self-Attention 后)
    token_hidden = torch.randn(d_model)

    # 路由器：一个线性层，输出到 num_experts 维
    router = torch.randn(num_experts, d_model)
    expert_logits = router @ token_hidden  # (4,) 每个专家的得分
    expert_probs = F.softmax(expert_logits, dim=-1)

    print(f"\n  输入 token 向量: [{', '.join(f'{token_hidden[i]:.2f}' for i in range(4))}...]")
    print(f"\n  路由器输出 (每个专家的得分):")
    for i in range(num_experts):
        print(f"    专家 {i}: {expert_logits[i]:.4f}  →  softmax 后: {expert_probs[i]:.4f}")

    # Top-K 选择
    topk_values, topk_indices = torch.topk(expert_probs, top_k)
    print(f"\n  Top-{top_k} 激活: 专家 {topk_indices.tolist()} (权重: {[f'{v:.3f}' for v in topk_values.tolist()]})")
    print(f"  只计算 {top_k}/{num_experts} 个专家的 FFN → 省了 {100 - top_k * 100 / num_experts:.0f}% 的计算")

    # ---- 对比表格 ----
    print(f"\n--- 模型架构对比 ---")
    print(f"\n  {'模型':<20s} {'架构':<12s} {'总参数':<10s} {'每次激活':<10s} {'特点'}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 20}")
    print(f"  {'Qwen-7B':<20s} {'Dense':<12s} {'7B':<10s} {'7B':<10s} {'全参数，中小规模'}")
    print(f"  {'DeepSeek-V3':<20s} {'MoE':<12s} {'671B':<10s} {'37B':<10s} {'稀疏激活，推理成本低'}")
    print(f"  {'Kimi (Moonshot)':<20s} {'Dense+RoPE':<12s} {'~8B':<10s} {'~8B':<10s} {'RoPE外推，长上下文'}")
    print(f"  {'Llama-3-70B':<20s} {'Dense':<12s} {'70B':<10s} {'70B':<10s} {'全参数，强基座能力'}")

    print(f"\n总结:")
    print(f"  DeepSeek 便宜 = MoE 稀疏激活 (算得少)")
    print(f"  Kimi 上下文长 = RoPE 外推优化 (看得远)")
    print(f"  Qwen 全覆盖 = Dense 多尺寸 (不同场景不同尺寸)")


def run_q2():
    print("=" * 55)
    print("  Q2: DeepSeek/Kimi/Qwen 核心区别 —— Transformer + RoPE")
    print("=" * 55)
    demo_1_self_attention()
    demo_2_scale_dot()
    demo_3_rope()
    demo_4_moe()
    print(f"\n{'=' * 55}")
    print("  Q2 Demo 完成")
    print("=" * 55)


if __name__ == "__main__":
    run_q2()
