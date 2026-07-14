import json
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):

    # 特殊 Token 列表 —— 子类可覆盖以扩展（开放封闭原则）
    # 以后只需在这里追加 <BOS> / <EOS> / <MASK> 等，无需改动子类逻辑
    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
    ]

    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}

    def _add_special_tokens(self):
        """将 SPECIAL_TOKENS 中的所有特殊 Token 追加到词表中"""
        for token in self.SPECIAL_TOKENS:
            self.vocab[token] = len(self.vocab)

    def _rebuild_id_to_token(self):
        """根据 vocab 重建 id_to_token 反向映射"""
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def save_vocab(self, path):
        with open(path, "w", encoding="utf8") as f:
            json.dump(
                self.vocab,
                f,
                ensure_ascii=False,
                indent=2
            )

    def load_vocab(self, path):
        with open(path, encoding="utf8") as f:
            self.vocab = json.load(f)
        self._rebuild_id_to_token()

    @property
    def vocab_size(self):
        return len(self.vocab)

    @abstractmethod
    def train(self, corpus_path):
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, ids):
        pass
    