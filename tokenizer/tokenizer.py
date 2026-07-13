from pathlib import Path


class CharacterTokenizer:
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

    def __init__(self):
        """初始化空的分词器，尚未建立词表"""
        # vocab: 字符 -> ID 的映射，例如 {"天": 0, "安": 1, ...}
        self.vocab = {}
        # id_to_token: ID -> 字符的反向映射，例如 {0: "天", 1: "安", ...}
        self.id_to_token = {}

    def train(self, corpus_path):
        """
        在语料库上训练分词器，建立词表。

        处理步骤：
            1. 读取语料库文件的全部文本
            2. 提取所有出现过的唯一字符（去重、排序）
            3. 为每个字符分配一个唯一的数字 ID
            4. 追加特殊 token：<PAD>（填充）和 <UNK>（未知字符）

        参数：
            corpus_path: 语料库文件的路径（纯文本，UTF-8 编码）
        """
        # 读取语料库全部文本
        text = Path(corpus_path).read_text(encoding="utf-8")

        # 去掉换行符后，收集所有不重复的字符并排序（保证每次训练结果一致）
        chars = sorted(set(text.replace("\n", "")))

        # 为每个字符分配 ID：第 0 个字符 ID=0，第 1 个字符 ID=1，以此类推
        self.vocab = {
            token: idx
            for idx, token in enumerate(chars)
        }

        # 添加两个特殊 token：
        #   <PAD> - 填充符，用于将不等长的序列填充到相同长度（batch 训练时需要）
        #   <UNK> - 未知符，用于处理训练时未出现过的新字符
        self.vocab["<PAD>"] = len(self.vocab)
        self.vocab["<UNK>"] = len(self.vocab)

        # 构建反向映射表（ID -> 字符），用于 decode 时快速查找
        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
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
        return [
            self.vocab.get(ch, self.vocab["<UNK>"])
            for ch in text
        ]

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
        return "".join(
            self.id_to_token.get(i, "<UNK>")
            for i in ids
        )