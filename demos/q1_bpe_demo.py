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


def demo_1_same_text_different_tokens():
    """
    同一个文本，不同 tokenizer 切出的 token 数不同
    """
    sep("Part 1: 同一文本 → 不同 Token 数")

    test_texts = [
        "北京今天天气真好，适合出去玩",
        "机器学习是人工智能的核心分支",
        "DeepSeek 的 MoE 架构让推理成本大幅降低",
    ]

    char_tok = CharacterTokenizer()
    char_tok.train("dataset/corpus.txt")

    bpe_tok = BPETokenizer(vocab_size=100)
    bpe_tok.train("dataset/corpus.txt")

    print(f"\n{'文本':<30s} {'字符分词器':>10s} {'BPE分词器':>10s}")
    print(f"{'─' * 30} {'─' * 10} {'─' * 10}")

    for text in test_texts:
        char_ids = char_tok.encode(text)
        bpe_ids = bpe_tok.encode(text)
        print(f"{text:<28s} {len(char_ids):>8d} 个 {len(bpe_ids):>8d} 个")

    print(f"\n结论: 同一段文本，BPE 通过合并高频组合减少了 token 数量。")
    print(f"      不同厂商用不同的 BPE 词表 → Token 计费不同是必然的。")


def demo_2_bpe_merge_process():
    """
    逐步展示 BPE 的合并过程 —— 现场推演
    """
    sep("Part 2: BPE 合并过程 —— 现场推演")

    text = "北京北京天安门天安门广场"
    print(f"\n原始文本: {text}")
    print(f"字符数: {len(text)}")

    # 用一个小 vocabsize 强制只合并几次，方便展示
    bpe = BPETokenizer(vocab_size=50)
    bpe._train_bpe(text)

    print(f"\nBPE 合并历史 ({len(bpe.merge_history)} 步):")
    print(f"{'步骤':>4s}  {'Pair':>12s}  {'→':>2s}  {'新Token':<12s}  {'频率':>4s}")
    print(f"{'─' * 4}  {'─' * 12}  {'─' * 2}  {'─' * 12}  {'─' * 4}")

    for i, (a, b, merged, freq) in enumerate(bpe.merge_history):
        print(f"{i + 1:>4d}  ({a},{b}){'':>8s} →  {merged:<12s}  {freq:>4d}")

    # 展示最终编码结果
    ids = bpe.encode(text)
    tokens = [bpe.id_to_token[i] for i in ids]
    print(f"\n最终切分结果: {' | '.join(tokens)}")
    print(f"Token 数量: {len(ids)}")

    # 与字符级对比
    print(f"\n对比字符级分词器:")
    char_tok = CharacterTokenizer()
    char_tok.train("dataset/corpus.txt")
    char_ids = char_tok.encode(text)
    char_tokens = [char_tok.id_to_token[i] for i in char_ids]
    print(f"  字符级: {' | '.join(char_tokens)} → {len(char_ids)} tokens")
    print(f"  BPE:    {' | '.join(tokens)} → {len(ids)} tokens")


def demo_3_vocab_size_effect():
    """
    BPE 词表大小如何影响 token 数 —— 厂商的 trade-off
    """
    sep("Part 3: 词表大小决定 Token 数（厂商的 trade-off）")

    text = "北京今天天气真好适合出去玩机器学习深度学习很火"

    char_tok = CharacterTokenizer()
    char_tok.train("dataset/corpus.txt")
    char_count = len(char_tok.encode(text))
    print(f"\n文本: {text}")
    print(f"字符级分词器 token 数: {char_count} (每个字 = 1 token)")
    print(f"\n{'Vocab Size':>14s}  {'Token 数':>10s}  {'说明':>20s}")
    print(f"{'─' * 14}  {'─' * 10}  {'─' * 20}")

    for vs in [30, 50, 80, 120, 180, 300]:
        bpe = BPETokenizer(vocab_size=vs)
        bpe.train("dataset/corpus.txt")
        tok_count = len(bpe.encode(text))
        desc = "小词表" if vs < 60 else ("中词表" if vs < 120 else "大词表")
        print(f"{vs:>14d}  {tok_count:>10d}  {desc:<20s}")

    print(f"\n结论: 词表越大 → token 越少 → 推理越快 → 但显存占用更多")
    print(f"      不同厂商在【速度】和【显存】之间做不同的 trade-off")
    print(f"      这就是为什么每个厂商的 token 计费都不一样")


def run_q1():
    print("=" * 55)
    print("  Q1: 为什么 Token 计费不一样？—— BPE 分词器")
    print("=" * 55)
    demo_1_same_text_different_tokens()
    demo_2_bpe_merge_process()
    demo_3_vocab_size_effect()
    print(f"\n{'=' * 55}")
    print("  Q1 Demo 完成")
    print("=" * 55)


if __name__ == "__main__":
    run_q1()
