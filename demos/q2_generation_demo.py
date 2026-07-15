"""
Q3 演示：为什么同一问题答案不同？—— LLM 生成机制

核心：LLM 不是"事实查询机"，而是"概率预测机"。
每次生成 = 下一 token 的概率分布 → 按策略采样 → 结果可能不同。

五个演示：
  Part 0: 在真实语料上训练微型字符级语言模型（真正学会预测下一个字符）
  Part 1: 逐 token 预测演示 —— 模型如何"接龙"
  Part 2: 用训练好的模型生成文本 —— 感受自回归生成
  Part 3: Temperature 效果对比 —— 如何控制随机性
  Part 4: Top-p / Top-k 采样 —— 更精细的随机控制
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.tokenizer import CharacterTokenizer


def sep(title):
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")


# ================================================================
#  微型字符级语言模型：Embedding → Linear → Softmax
#  这是你能写出最简单的"LLM"，只有 2 层，但能学会预测下一个字符
# ================================================================
class TinyCharLM(nn.Module):
    """微型字符级语言模型 —— 最简单的 LLM"""

    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  # 字符 → 向量
        self.output = nn.Linear(embed_dim, vocab_size)    # 向量 → 预测下一个字符

    def forward(self, x):
        # x: [batch, seq_len] 的 token IDs
        emb = self.embed(x)           # [batch, seq_len, embed_dim]
        logits = self.output(emb)     # [batch, seq_len, vocab_size]
        return logits

    @torch.no_grad()
    def predict_next(self, token_id):
        """给定一个 token ID，预测下一个 token 的概率分布"""
        x = torch.tensor([[token_id]], dtype=torch.long)
        logits = self.forward(x)      # [1, 1, vocab_size]
        probs = F.softmax(logits[0, 0], dim=-1)
        return probs

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens=20, temperature=1.0, top_p=1.0, top_k=0, trace=False):
        """自回归生成文本 —— 逐 token 预测、采样、拼接"""
        # 构建特殊 Token 的 ID 集合（生成时屏蔽它们）
        special_ids = set()
        for t in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            if t in tokenizer.vocab:
                special_ids.add(tokenizer.vocab[t])

        ids = tokenizer.encode(prompt)
        trace_info = []

        for step in range(max_new_tokens):
            # 用最后一个 token 预测下一个
            last_char = tokenizer.decode([ids[-1]])
            x = torch.tensor([[ids[-1]]], dtype=torch.long)
            logits = self.forward(x)[0, 0] / temperature

            # 屏蔽特殊 Token（训练时没见过，权重随机，低 T 时容易胜出）
            for sid in special_ids:
                logits[sid] = float('-inf')

            # Top-k 过滤
            if top_k > 0:
                filtered_k = min(top_k, (logits > float('-inf')).sum().item())
                if filtered_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, filtered_k)
                    mask = torch.full_like(logits, float('-inf'))
                    mask[topk_idx] = topk_vals
                    logits = mask

            # Top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = (cumsum > top_p).nonzero(as_tuple=True)[0]
                if len(cutoff) > 0:
                    # 在原 logits 中将 cutoff 之后的 token 置为 -inf（保留原始位置映射）
                    for idx in sorted_idx[cutoff[0] + 1:]:
                        logits[idx.item()] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            next_char = tokenizer.decode([next_id])
            ids.append(next_id)

            max_p = probs.max().item()
            top3 = torch.topk(probs, min(3, (probs > 0).sum().item()))
            top3_str = " ".join(
                f"'{tokenizer.decode([i.item()])}'({p.item():.3f})"
                for i, p in zip(top3.indices, top3.values)
            )
            trace_info.append(f"    Step {step+1}: '{last_char}' → '{next_char}' (p={max_p:.3f}) | Top: {top3_str}")

        result = tokenizer.decode(ids)
        if trace:
            return result, trace_info
        return result


# ================================================================
#  Part 0: 训练微型语言模型
# ================================================================
def demo_0_train_model():
    sep("Part 0: 在真实语料上训练微型字符级语言模型")

    # 1. 加载语料并分词
    corpus_path = Path(__file__).resolve().parent.parent / "dataset" / "corpus.txt"
    tokenizer = CharacterTokenizer()
    tokenizer.train(str(corpus_path))

    text = Path(corpus_path).read_text(encoding="utf-8").replace("\n", "")
    all_ids = tokenizer.encode(text)
    vocab_size = len(tokenizer.vocab)
    print(f"  词表大小: {vocab_size}  语料长度: {len(text)} 字符  Token数: {len(all_ids)}")

    # 2. 构造训练数据：每个字符 → 下一个字符
    X = torch.tensor(all_ids[:-1], dtype=torch.long)
    Y = torch.tensor(all_ids[1:], dtype=torch.long)

    print(f"\n  训练数据示例 (前 10 组):")
    for i in range(min(10, len(X))):
        ch_in = tokenizer.decode([X[i].item()])
        ch_out = tokenizer.decode([Y[i].item()])
        print(f"    输入: '{ch_in}' ({X[i].item():>3d})  →  标签: '{ch_out}' ({Y[i].item():>3d})")

    # 3. 创建模型
    torch.manual_seed(42)
    model = TinyCharLM(vocab_size, embed_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型参数量: {total_params} （Embedding {vocab_size}×16 + Linear 16×{vocab_size}）")

    # 4. 训练
    print(f"\n  开始训练 (batch 梯度下降, 150 轮)...")
    for epoch in range(150):
        model.train()
        logits = model(X.unsqueeze(0))   # [1, seq_len, vocab_size]
        loss = loss_fn(logits[0], Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 30 == 0:
            # 计算准确率
            preds = logits[0].argmax(dim=-1)
            acc = (preds == Y).float().mean().item()
            print(f"    Epoch {epoch + 1:>3d}/150  Loss: {loss.item():.4f}  Acc: {acc:.3f}")

    print(f"\n  ✅ 训练完成！模型已经学会了字符间的统计规律")

    return model, tokenizer, all_ids, X, Y, loss_fn


# ================================================================
#  Part 1: 逐 token 预测演示
# ================================================================
def demo_1_predict_next(model, tokenizer):
    sep("Part 1: 用训练好的模型预测下一个 Token")

    print("\n  LLM 的核心操作：给定上文，预测下一个 token 的概率分布")
    print("  「下一个」不是随机猜的 —— 是训练数据中统计出的规律\n")

    test_prefixes = [
        "北京",
        "上海",
        "人工",
        "Token",
        "天安",
        "机器",
    ]

    for prefix in test_prefixes:
        ids = tokenizer.encode(prefix)
        if len(ids) == 0:
            continue
        last_id = ids[-1]
        probs = model.predict_next(last_id)

        top_k = 5
        top_probs, top_ids = torch.topk(probs, min(top_k, len(probs)))

        print(f"  输入: '{prefix}'  →  最后一个字符是 '{prefix[-1]}' (ID={last_id})")
        print(f"  模型预测的下一个字符 Top {top_k}:")

        # 检查语料中验证
        full_text = Path(__file__).resolve().parent.parent / "dataset" / "corpus.txt"
        full_text = full_text.read_text(encoding="utf-8")
        for rank, (tid, tp) in enumerate(zip(top_ids, top_probs)):
            ch = tokenizer.decode([tid.item()])
            bar = "█" * int(tp.item() * 40)
            # 在语料中找 "prefix + ch" 的出现次数
            count = full_text.count(prefix + ch)
            in_corpus = f" 语料中 '\\n{prefix}{ch}' 出现 {count} 次" if count > 0 else ""
            print(f"    {rank + 1}. '{ch}'  {tp.item():.4f}  {bar}{in_corpus}")

        print()

    print("  关键洞察:")
    print("    1. 模型不是查字典 —— 它从训练数据中学到了统计规律")
    print('    2. "北京" 后面大概率是 "天"(语料中"北京天"出现最多)')
    print('    3. "天安" 后面大概率是 "门"(语料中"天安门"出现最多)')
    print("    4. 这就是 LLM 「预测下一个 Token」的本质！")


# ================================================================
#  Part 2: 自回归生成
# ================================================================
def demo_2_generate(model, tokenizer):
    sep("Part 2: 自回归生成 —— 把「预测下一个」连起来就是「生成」")

    print("\n  自回归 = 每步预测下一个 token → 拼接到输入 → 继续预测 → ...")
    print("  就像玩「词语接龙」，每一步的输出成为下一步的输入\n")

    prompts = [
        "北京",
        "人工",
        "机器",
        "大语言",
        "自",
    ]

    for prompt in prompts:
        torch.manual_seed(42)
        result = model.generate(tokenizer, prompt, max_new_tokens=15, temperature=1.0)
        print(f"  Prompt: '{prompt}'")
        print(f"  生成:   '{result}'")
        print()

    print("  注意到：有些生成合理（如「北京天安门」），有些不太通顺")
    print("  原因：模型只看 1 个字符的上下文（太短），且训练数据有限")
    print("  真正的 LLM 用 Transformer 看几千个字符的上下文 → 效果更好")


# ================================================================
#  Part 3: Temperature 效果
# ================================================================
def demo_3_temperature(model, tokenizer):
    sep("Part 3: Temperature —— 随机性的「旋钮」")

    last_id = tokenizer.encode("北京")[-1]
    probs_raw = model.predict_next(last_id)  # 原始概率（相当于 T=1）

    # 展示不同 Temperature 对概率分布的影响
    logits = torch.log(probs_raw + 1e-10)

    top_ids = torch.topk(probs_raw, 5).indices
    top_chars = [tokenizer.decode([tid.item()]) for tid in top_ids]

    print(f"\n  Temperature 对 '北京' 后面预测的影响:")
    print(f"  {'T':>6s}  ", end="")
    for ch in top_chars:
        print(f"{ch:>8s}", end="")
    print(f"  {'最高概率':>10s}  {'说明'}")
    print(f"  {'─' * 6}  {'─' * (9 * len(top_chars))}  {'─' * 20}")

    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)
        print(f"  {temp:<6.1f}", end="")
        for tid in top_ids:
            print(f"  {probs[tid]:.4f}", end="")
        max_p = probs.max().item()
        desc = "几乎确定" if temp <= 0.2 else ("正常多样" if temp <= 1.5 else "接近随机")
        print(f"  {max_p:.4f}      {desc}")

    # 多次生成对比
    print(f"\n  用同一 Prompt '北京'，不同 T 各生成 5 次对比:")
    for temp in [0.1, 0.7, 2.0]:
        print(f"\n    T={temp}:")
        results = []
        for i in range(5):
            torch.manual_seed(i * 100)
            r = model.generate(tokenizer, "北京", max_new_tokens=12, temperature=temp)
            results.append(r)
            unique = len(set(results))
        for r in results:
            print(f"      → '{r}'")
        print(f"    5 次中了 {unique} 种不同结果")

    print(f"\n  公式: P(token_i) = softmax(logits_i / T)")
    print(f"    T → 0:  分布极端 → 总选最高概率 → 确定性（适合数学/代码）")
    print(f"    T = 1:  原始分布 → 标准采样")
    print(f"    T → ∞: 趋于均匀 → 完全随机（适合头脑风暴/创意写作）")


# ================================================================
#  Part 4: Top-p / Top-k
# ================================================================
def demo_4_top_p_top_k(model, tokenizer):
    sep("Part 4: Top-p & Top-k —— 更精细的随机控制")

    last_id = tokenizer.encode("人工")[-1]
    probs = model.predict_next(last_id)

    sorted_p, sorted_i = torch.sort(probs, descending=True)

    print(f"\n  '人工' 后面预测的 Top 10 候选:")
    print(f"  {'Rank':>4s}  {'字符':>4s}  {'概率':>8s}  {'累计':>8s}  条形图")
    print(f"  {'─' * 4}  {'─' * 4}  {'─' * 8}  {'─' * 8}  {'─' * 25}")
    cumsum = 0.0
    cutoff_p50 = cutoff_p90 = None
    for rank in range(min(10, len(sorted_p))):
        p = sorted_p[rank].item()
        cumsum += p
        ch = tokenizer.decode([sorted_i[rank].item()])
        bar = "█" * int(p * 40)
        marker = ""
        if cutoff_p50 is None and cumsum >= 0.5:
            cutoff_p50 = rank + 1
            marker = " ← Top-p=0.5 截断线"
        if cutoff_p90 is None and cumsum >= 0.9:
            cutoff_p90 = rank + 1
            marker = " ← Top-p=0.9 截断线"
        print(f"  {rank + 1:>4d}  {ch:>4s}  {p:>8.4f}  {cumsum:>8.4f}  {bar}{marker}")

    print(f"\n  Top-p 效果:")
    print(f"    p=0.5: 只保留前 {cutoff_p50} 个候选（丢弃尾部低概率噪声）")
    print(f"    p=0.9: 只保留前 {cutoff_p90} 个候选（更宽松，保留更多多样性）")
    print(f"    p=1.0: 所有候选都参与（可能选中极低概率的奇怪字符）")

    # Top-k
    print(f"\n  Top-k 效果:")
    for top_k in [1, 3, 10]:
        ids = tokenizer.encode("人工")
        r = model.generate(tokenizer, "人工", max_new_tokens=8, temperature=1.0, top_k=top_k)
        print(f"    k={top_k:>2d}: '{r}'")

    # 生产推荐
    print(f"\n  --- 生产环境推荐组合 ---")
    print(f"    Temperature=0.7 + Top-p=0.9 + Top-k=50")
    print(f"    这个组合: 保证输出质量 + 适度多样性 + 截断尾部噪声")
    print(f"\n  不同场景建议:")
    print(f"    代码/数学:  T=0.1~0.3  (要准确)")
    print(f"    通用对话:  T=0.6~0.8  (平衡)")
    print(f"    创意写作:  T=0.9~1.2  (要多样性)")


# ================================================================
#  Part 5: 交互式对话模拟 —— 同一输入，多次不同输出
# ================================================================
def demo_5_interactive_chat(model, tokenizer):
    sep("Part 5: 交互式对话模拟 —— 同一问题，答案不同")

    # 默认生成参数（可运行时调节）
    current_temp = 0.8
    current_top_p = 0.9
    current_top_k = 20

    print("\n  💬 这是一个模拟 LLM 对话的交互环节")
    print("  ")
    print("  你可以输入任何话，模型会生成回复。")
    print("  输入同一句话多次，你会看到每次回复都不一样！")
    print("  这就是概率采样导致的效果。")
    print()
    print("  命令:")
    print("    输入文字 → 生成回复（用当前参数）")
    print("    :repeat <N> <text> → 同一输入重复 N 次（对比不同输出）")
    print("    :probe <text> → 探测：展示下一个字符对不同 T 的概率变化")
    print("    :trace <text> → 追踪：逐步展示每一步生成过程和概率")
    print("    :temp <值>  → 修改 Temperature（当前: 0.8）")
    print("    :topk <值>  → 修改 Top-k（当前: 20, 0=关闭）")
    print("    :topp <值>  → 修改 Top-p（当前: 0.9, 1=关闭）")
    print("    :params     → 查看当前参数")
    print("    :history    → 查看对话历史")
    print("    :quit       → 退出")
    print()

    history = []

    while True:
        try:
            raw = input("  👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not raw:
            continue

        # 退出
        if raw == ":quit":
            print("  再见！")
            break

        # 查看当前参数
        if raw == ":params":
            print(f"  ⚙️  当前参数: Temperature={current_temp}, Top-p={current_top_p}, Top-k={current_top_k}")
            print(f"      说明: T→0 更确定, T→大 更随机 | Top-k=0 无限制 | Top-p=1 无限制")
            continue

        # 修改 Temperature
        if raw.startswith(":temp "):
            try:
                v = float(raw.split()[1])
                if v <= 0:
                    print("  ❌ Temperature 必须 > 0")
                else:
                    current_temp = v
                    print(f"  ✅ Temperature 已设为 {v}")
                    if v <= 0.3:
                        print(f"     (低 T: 输出几乎确定，适合代码/数学)")
                    elif v <= 0.8:
                        print(f"     (中 T: 平衡确定性与多样性，适合通用对话)")
                    else:
                        print(f"     (高 T: 输出多样随机，适合创意写作)")
            except (ValueError, IndexError):
                print("  用法: :temp 0.7")
            continue

        # 修改 Top-k
        if raw.startswith(":topk "):
            try:
                v = int(raw.split()[1])
                if v < 0:
                    print("  ❌ Top-k 必须 >= 0")
                else:
                    current_top_k = v
                    print(f"  ✅ Top-k 已设为 {v}" + (" (无限制)" if v == 0 else f" (每步只考虑前 {v} 个候选)"))
            except (ValueError, IndexError):
                print("  用法: :topk 10")
            continue

        # 修改 Top-p
        if raw.startswith(":topp "):
            try:
                v = float(raw.split()[1])
                if not 0 < v <= 1:
                    print("  ❌ Top-p 必须在 (0, 1] 范围内")
                else:
                    current_top_p = v
                    print(f"  ✅ Top-p 已设为 {v}" + (" (无限制)" if v == 1 else f" (累积概率达 {v} 时截断)"))
            except (ValueError, IndexError):
                print("  用法: :topp 0.95")
            continue

        # 探测命令：展示不同 T 对下一个字符概率的影响
        if raw.startswith(":probe "):
            text = raw[len(":probe "):].strip()
            ids = tokenizer.encode(text)
            if not ids:
                print("  无法编码该文本")
                continue

            last_ch = tokenizer.decode([ids[-1]])
            probs_raw = model.predict_next(ids[-1])
            logits = torch.log(probs_raw + 1e-10)

            # 找出 Top 8 候选（用 T=1 排序）
            top8_ids = torch.topk(probs_raw, 8).indices
            top8_chars = [tokenizer.decode([i.item()]) for i in top8_ids]

            print(f"\n  ╔{'═' * 68}╗")
            print(f"  ║  🔬 探测: 输入「{text}」→ 最后一个字符「{last_ch}」后，下一个字符的概率分布{' ' * max(0, 7 - len(text))}║")
            print(f"  ╚{'═' * 68}╝")
            print(f"  {'T':>6s} │", end="")
            for ch in top8_chars:
                print(f" {ch:>6s}", end="")
            print(f"  │  {'最高概率':>8s}  状态")
            print(f"  {'─' * 6}─┼{'─' * (7 * len(top8_chars))}─┼─{'─' * 20}")

            for t in [0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                scaled = logits / t
                probs = F.softmax(scaled, dim=-1)
                top_p = probs[top8_ids]
                print(f"  {t:>5.2f}  │", end="")
                for p in top_p:
                    print(f" {p.item():.4f}", end="")
                max_p = probs.max().item()
                if t <= 0.1:
                    status = "← T→0 几乎确定（贪心）"
                elif t <= 0.5:
                    status = "← 较确定，少量多样性"
                elif t <= 1.0:
                    status = "← 原始分布，适度随机"
                elif t <= 2.0:
                    status = "← 分布变平，更多样化"
                else:
                    status = "← T→∞ 趋于均匀分布"
                print(f"  │  {max_p:.4f}     {status}")

            print(f"\n  💡 核心发现:")
            print(f"     T=0.01:  最高概率接近 1.0 → 每次都选同一个字 → 无随机")
            print(f"     T=1.0:   保留原始统计规律 → 有随机但合理")
            print(f"     T=3.0:   分布接近均匀 → 完全随机 → 容易出乱码")
            print(f"     公式: P(token_i) = softmax(logits_i / T)")
            print()

            # 实际生成对比（用当前 top-k/top-p）
            print(f"  用当前参数 (top-p={current_top_p}, top-k={current_top_k}) 实际生成对比:")
            for t in [0.1, 0.8, 2.0]:
                results = []
                for i in range(3):
                    r = model.generate(tokenizer, text, max_new_tokens=12,
                                       temperature=t, top_p=current_top_p, top_k=current_top_k)
                    results.append(r)
                unique = len(set(results))
                print(f"    T={t:<4.1f}: ", end="")
                for r in results:
                    print(f"'{r}'  ", end="")
                print(f"  ({unique}/3 种不同)")
            continue

        # 追踪命令：逐步展示生成过程
        if raw.startswith(":trace "):
            text = raw[len(":trace "):].strip()
            result, trace_info = model.generate(
                tokenizer, text,
                max_new_tokens=10, temperature=current_temp,
                top_p=current_top_p, top_k=current_top_k, trace=True
            )
            print(f"\n  🔍 逐步追踪生成过程 (T={current_temp}, top-p={current_top_p}, top-k={current_top_k})")
            print(f"     Prompt: '{text}'")
            for line in trace_info:
                print(line)
            print(f"     → 最终输出: '{result}'")
            continue

        # 历史
        if raw == ":history":
            if not history:
                print("  (暂无对话)")
            else:
                for i, (usr, bot, t) in enumerate(history, 1):
                    print(f"\n  ── 第 {i} 轮 (T={t})──")
                    print(f"  👤 你: {usr}")
                    print(f"  🤖 模型: {bot}")
            continue

        # 重复命令
        if raw.startswith(":repeat "):
            parts = raw.split(" ", 2)
            if len(parts) < 3:
                print("  用法: :repeat <次数> <要输入的文本>")
                continue
            try:
                n = int(parts[1])
            except ValueError:
                print("  次数必须是数字")
                continue
            prompt = parts[2]
            print(f"\n  ╔{'═' * 56}╗")
            print(f"  ║  同一输入 「{prompt}」 重复 {n} 次 (T={current_temp}){' ' * (24 - len(prompt))}║")
            print(f"  ╚{'═' * 56}╝")

            results = []
            for i in range(n):
                result = model.generate(
                    tokenizer, prompt,
                    max_new_tokens=20, temperature=current_temp,
                    top_p=current_top_p, top_k=current_top_k
                )
                results.append(result)
                marker = " ✅ 不同" if i > 0 and result != results[0] else (" 🔄 相同" if i > 0 else "")
                print(f"    第{i + 1}次: 🤖 {result}{marker}")

            unique = len(set(results))
            print(f"\n  📊 {n} 次生成，得到 {unique} 种不同输出 (T={current_temp})")
            if unique == 1:
                print(f"  💡 T={current_temp} 太低了，模型几乎确定 → 试试 :temp 1.5 提高随机性")
            elif unique == n:
                print(f"  🎯 每次都不一样！完美展示了 LLM 的随机性")
            else:
                print(f"  🎯 {n - unique} 次出现了重复，{unique} 种不同输出")
            continue

        # 正常对话
        prompt = raw
        result = model.generate(
            tokenizer, prompt,
            max_new_tokens=25, temperature=current_temp,
            top_p=current_top_p, top_k=current_top_k
        )
        print(f"  🤖 模型 (T={current_temp}): {result}")

        # 展示采样细节
        ids = tokenizer.encode(prompt)
        if ids:
            probs = model.predict_next(ids[-1])
            logits = torch.log(probs + 1e-10)
            # 用当前 Temperature 缩放
            scaled = logits / current_temp
            scaled_probs = F.softmax(scaled, dim=-1)

            top3_ids = torch.topk(scaled_probs, 3).indices
            top3_chars = [tokenizer.decode([i.item()]) for i in top3_ids]
            top3_probs = [scaled_probs[i].item() for i in top3_ids]
            raw_top3 = [probs[i].item() for i in top3_ids]
            top1_id = scaled_probs.argmax().item()
            chosen = result[len(prompt)] if len(result) > len(prompt) else "?"

            print(f"  🔍 第一个字的选择细节 (T={current_temp}):")
            for i, (c, sp, rp) in enumerate(zip(top3_chars, top3_probs, raw_top3)):
                arrow = " ← 选中" if c == chosen else ""
                delta = "不变" if abs(sp - rp) < 0.001 else (f"↑ 从 {rp:.3f}" if sp > rp else f"↓ 从 {rp:.3f}")
                print(f"     {i + 1}. '{c}'  T缩放后={sp:.3f}  原始={rp:.3f}  ({delta}){arrow}")
            if chosen != tokenizer.decode([top1_id]):
                print(f"     ⚡ '{chosen}' 不是最高概率 '{tokenizer.decode([top1_id])}' → 随机采样生效！")
            else:
                print(f"     📌 选中了最高概率的 '{chosen}'")

        history.append((raw, result, current_temp))


# ================================================================
#  Main
# ================================================================
def run_q3():
    """非交互模式：只执行训练 → 给测试调用"""
    # Part 0: 训练模型
    model, tokenizer, all_ids, X, Y, loss_fn = demo_0_train_model()
    return model, tokenizer


def run_q3_chat():
    """交互模式：训练模型后进入对话模拟"""
    model, tokenizer = run_q3()
    demo_5_interactive_chat(model, tokenizer)


if __name__ == "__main__":
    import sys
    if "--chat" in sys.argv:
        run_q3_chat()
    else:
        run_q3()
