from pathlib import Path

from .base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """
    字符级分词器（Character-level Tokenizer）

    将文本中的每个**字符**当作一个独立的 token，是最简单的分词方式。
    中文天然适合字符级分词，因为每个汉字本身就是一个有意义的单元。

    工作流程：
        1. train()  - 扫描语料库，建立字符 -> ID 的映射表（词表）
        2. encode() - 将文本中的每个字符映射为对应的 ID
        3. decode() - 将 ID 序列还原为文本

    示例：
        tokenizer = CharacterTokenizer()
        tokenizer.train("corpus.txt")
        ids = tokenizer.encode("你好")   # -> [10, 25]
        text = tokenizer.decode([10, 25]) # -> "你好"
    """

    # ----------------------------------------------------------------
    # Template Method: train() 负责 I/O + 预处理，build_vocab() 负责建词表
    # 以后 BPE 等子类只需重写 build_vocab()，train() 流程不用改
    # ----------------------------------------------------------------

    def train(self, corpus_path):
        """
        在语料库上训练分词器，建立词表。

        流程（Template Method）：
            1. 读取语料库文件
            2. 文本预处理（子类可按需扩展）
            3. 调用 build_vocab() 建词表（子类可重写）
            4. 追加特殊 Token
            5. 构建反向映射表

        参数：
            corpus_path: 语料库文件的路径（纯文本，UTF-8 编码）
        """
        text = Path(corpus_path).read_text(encoding="utf-8")

        # --- 预处理管道：以后可逐步扩展 ---
        text = text.replace("\n", "")
        # text = text.lower()
        # text = normalize(text)
        # ----------------------------------

        self.build_vocab(text)
        self._add_special_tokens()
        self._rebuild_id_to_token()

    def build_vocab(self, text):
        """
        基于预处理后的文本建立字符级词表。

        以后 BPE 等分词器只需重写此方法即可。

        参数：
            text: 预处理后的纯文本
        """
        # 收集所有不重复的字符并排序（保证每次训练结果一致）
        chars = sorted(set(text))
        self.vocab = {
            token: idx
            for idx, token in enumerate(chars)
        }

    def encode(self, text):
        """
        将文本编码为 ID 序列。

        对输入文本中的每一个字符，查找它在词表中的 ID。
        如果字符不在词表中（训练时没见过），则用 <UNK> 的 ID 代替。

        参数：
            text: 要编码的字符串

        返回：
            list[int]: 每个字符对应的 ID 列表
        """
        ids = [
            self.vocab.get(ch, self.vocab["<UNK>"])
            for ch in text
        ]

        # --- 扩展点：以后可在此处增加 ---
        # if add_bos:
        #     ids.insert(0, self.vocab["<BOS>"])
        # if add_eos:
        #     ids.append(self.vocab["<EOS>"])
        # if max_length:
        #     ids = ids[:max_length]
        # --------------------------------

        return ids

    def decode(self, ids):
        """
        将 ID 序列解码回文本。

        对 ID 列表中的每一个 ID，查找它对应的字符并拼接成字符串。
        如果 ID 不在反查表中，则用 "<UNK>" 占位。

        参数：
            ids: ID 列表

        返回：
            str: 解码后的文本
        """
        tokens = [
            self.id_to_token.get(i, "<UNK>")
            for i in ids
        ]

        # --- 扩展点：以后可在此处增加 ---
        # tokens = [t for t in tokens if t != "<PAD>"]
        # --------------------------------

        text = "".join(tokens)
        return text