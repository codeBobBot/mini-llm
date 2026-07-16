"""
Q1 演示：为什么 Token 计费不一样？—— BPE 分词器

对比 CharacterTokenizer 和 BPETokenizer 对同一文本的切分结果，
直观展示不同分词算法导致 token 数不同的根本原因。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import CharacterTokenizer
from tokenizer.bpe import BPETokenizer


def sep(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def _base_char_count():
    """计算语料库中去除特殊 token 后的基础字符数"""
    char_tok = CharacterTokenizer()
    char_tok.train("dataset/corpus.txt")
    return len(char_tok.vocab) - len(char_tok.SPECIAL_TOKENS)


def demo_1_same_text_different_tokens():
    """
    同一个文本，不同 tokenizer 切出的 token 数不同
    """
    sep("Part 1: 同一文本 → 不同 Token 数")

    base_chars = _base_char_count()
    # BPE 词表大小必须在基础字符之上留有合并空间
    bpe_vocab_size = base_chars + 200

    char_tok = CharacterTokenizer()
    char_tok.train("dataset/corpus.txt")

    bpe_tok = BPETokenizer(vocab_size=bpe_vocab_size)
    bpe_tok.train("dataset/corpus.txt")

    print(f"\n[语料库基础字符数]: {base_chars}")
    print(f"[BPE 词表大小]: {bpe_vocab_size} = {base_chars}(基础字符) + {bpe_vocab_size - base_chars}(合并)")
    print(f"[BPE 实际合并次数]: {len(bpe_tok.merge_history)}\n")

    # 选语料库中常见的多字词组合，确保 BPE 能学到合并规则
    test_texts = [
        "人工智能正在改变世界",
        "机器学习是人工智能的核心分支",
        "注意力机制是Transformer的核心创新",
        "大语言模型是自然语言处理的前沿技术",
        "位置编码让模型知道词语的顺序",
    ]

    header = f"{'文本':<34s} {'字符级':>6s} {'BPE':>6s} {'减少':>6s}"
    print(header)
    print("-" * len(header))

    for text in test_texts:
        char_ids = char_tok.encode(text)
        bpe_ids = bpe_tok.encode(text)
        reduction = len(char_ids) - len(bpe_ids)

        # 解码出实际 token
        char_tokens = [char_tok.id_to_token[i] for i in char_ids
                       if char_tok.id_to_token[i] not in char_tok.SPECIAL_TOKENS]
        bpe_tokens = [bpe_tok.id_to_token[i] for i in bpe_ids
                      if bpe_tok.id_to_token[i] not in bpe_tok.SPECIAL_TOKENS]

        print(f"{text:<32s} {len(char_ids):>4d}个 {len(bpe_ids):>4d}个 {reduction:>4d}个")
        print(f"  字符级: {' | '.join(char_tokens)}")
        print(f"  BPE  : {' | '.join(bpe_tokens)}\n")

    print(f"\n结论: BPE 将高频组合合并为一个 token，有效减少了 token 数量。")
    print(f"      如果 vocab_size 太小（≤ 基础字符数），BPE 退化为字符分词器，没有合并效果。")


def demo_2_bpe_merge_process():
    """
    展示 BPE 逐步合并过程 + 最终 token 对比
    """
    sep("Part 2: BPE 学到了哪些组合？—— 现场推演")

    base_chars = _base_char_count()
    bpe_vocab_size = base_chars + 200

    bpe = BPETokenizer(vocab_size=bpe_vocab_size)
    bpe.train("dataset/corpus.txt")

    # 选一段能充分展示 BPE 合并效果的文本
    text = "人工智能正在改变我们的生活方式，大语言模型是自然语言处理的前沿技术"

    char_tok = CharacterTokenizer()
    char_tok.train("dataset/corpus.txt")

    char_ids = char_tok.encode(text)
    char_tokens = [
        char_tok.id_to_token[i]
        for i in char_ids
        if char_tok.id_to_token[i] not in char_tok.SPECIAL_TOKENS
    ]

    bpe_ids = bpe.encode(text)
    bpe_tokens = [
        bpe.id_to_token[i]
        for i in bpe_ids
        if bpe.id_to_token[i] not in bpe.SPECIAL_TOKENS
    ]

    print(f"\n原文: {text}\n")
    print(f"字符级 ({len(char_tokens)} tokens):")
    print(f"  {' | '.join(char_tokens)}")
    print(f"\nBPE     ({len(bpe_tokens)} tokens):")
    print(f"  {' | '.join(bpe_tokens)}")
    print(f"\n>> BPE 减少了 {len(char_tokens) - len(bpe_tokens)} 个 token（合并率 {len(bpe_tokens) / len(char_tokens):.0%}）")

    # 展示合并历史
    print(f"\nBPE 部分合并规则 (共 {len(bpe.merge_history)} 条，展示前 12 条):")
    print(f"{'步骤':>4s}  {'Pair':>16s}  →  {'新Token':<16s}  {'频率':>4s}")
    print(f"{'─' * 4}  {'─' * 16}  {'─' * 2}  {'─' * 16}  {'─' * 4}")
    for i, (a, b, merged, freq) in enumerate(bpe.merge_history[:12]):
        print(f"{i + 1:>4d}  ({a}, {b}){'':>10s} →  {merged:<16s}  {freq:>4d}")
    if len(bpe.merge_history) > 12:
        print(f"  ... 共 {len(bpe.merge_history)} 条合并规则")

    # 额外演示：逐步骤展示一段短文本的 BPE 编码过程
    sep("Part 2.1: 短文本逐步合并过程")
    short_text = "北京天安门广场很壮观"
    print(f"\n原文: {short_text} (共 {len(short_text)} 字符)")
    print(f"\n逐步合并:")
    tokens = list(short_text)
    print(f"  初始: {' | '.join(tokens)}  ({len(tokens)} tokens)")
    for step, (a, b, merged, freq) in enumerate(bpe.merge_history):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if new_tokens != tokens:
            tokens = new_tokens
            step_label = f"步骤{step+1}"
            print(f"  {step_label:>5s}: {' | '.join(tokens)}  ({len(tokens)} tokens)")
    print(f"\n最终: {len(tokens)} 个 token，比字符级减少了 {len(short_text) - len(tokens)} 个")



def run_q1():
    print("=" * 55)
    print("  Q1: 为什么 Token 计费不一样？—— BPE 分词器")
    print("=" * 55)
    demo_1_same_text_different_tokens()
    demo_2_bpe_merge_process()
    print(f"\n{'=' * 55}")
    print("  Q1 Demo 完成")
    print("=" * 55)


if __name__ == "__main__":
    run_q1()
