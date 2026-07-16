"""
BPE (Byte Pair Encoding) 分词器

实现了最简单的 BPE 算法：
1. 将所有文本拆成字符
2. 反复统计并合并最频繁的相邻字符对
3. 直到词表达到指定大小

这用于现场演示"为什么不同厂商的 Token 计费不同"。
"""
from collections import Counter

from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE 分词器 —— 通过统计字符对频率来合并 token。

    与 CharacterTokenizer 的区别：
        CharacterTokenizer：每个字符 = 1 个 token（"北京" → 2 token）
        BPETokenizer：     "北京" 如果是高频组合 → 1 个 token（"北京" → 1 token）

    这解释了为什么同一段文本，在不同 tokenizer 下 token 数不同。
    """

    def __init__(self, vocab_size=300):
        """
        参数：
            vocab_size: 目标词表大小。越大，合并的 token 越多，单 token 越长。
                       不同厂商的 vocab_size 不同，导致 token 化结果不同。
        """
        super().__init__()
        self.target_vocab_size = vocab_size
        # 记录合并历史，用于演示 BPE 的逐步合并过程
        self.merge_history = []

    def train(self, corpus_path):
        """训练 BPE 分词器"""
        from pathlib import Path
        text = Path(corpus_path).read_text(encoding="utf-8")
        text = text.replace("\n", "").replace(" ", "")
        self._train_bpe(text)

    def _train_bpe(self, text):
        """
        BPE 核心训练算法 —— 每次合并最频繁的相邻字符对。

        这是最简单的实现，适合演示原理。
        """
        if not text:
            return

        # Step 1: 把文本拆成单个字符（用空格分隔，方便之后合并）
        tokens = list(text)
        # print(f"tokens: {tokens}")

        # Step 2: 收集初始字符作为基础词表
        vocab = {ch: i for i, ch in enumerate(sorted(set(tokens)))}

        self.merge_history = []

        # Step 3: 反复合并，直到词表达到目标大小
        while len(vocab) < self.target_vocab_size:
            pair_counts = Counter()
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += 1

            # pair_counts 为空字典，没有相邻 pair 可以统计
            if not pair_counts:
                break

            # 找最频繁的 pair
            best_pair, freq = pair_counts.most_common(1)[0]

            if freq <= 1:
                break  # 所有 pair 都只出现一次，再合并没意义

            # 合并：用新 token 替换所有出现的旧 pair
            new_token = best_pair[0] + best_pair[1]
            self.merge_history.append((best_pair[0], best_pair[1], new_token, freq))

            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            # 更新词表（为新合并 token 分配 ID）
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

        self.vocab = vocab
        self._add_special_tokens()
        self._rebuild_id_to_token()

    def encode(self, text):
        """
        BPE 编码：用训练好的合并规则递归拆分文本。

        如果字符/子词不在词表中，回退到 <UNK>。
        """
        tokens = list(text)
        # 按合并历史的顺序（最早合并的优先），尝试递归合并
        for a, b, merged, _ in self.merge_history:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == a
                        and tokens[i + 1] == b):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(t, self.vocab.get("<UNK>", 0)) for t in tokens]

    def decode(self, ids):
        """BPE 解码：直接拼接 token 字符串"""
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return "".join(t for t in tokens if t not in self.SPECIAL_TOKENS)
