"""
CharacterTokenizer 单元测试
"""
import sys
import unittest
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import CharacterTokenizer


class TestCharacterTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """所有测试共享同一个训练好的 tokenizer 实例"""
        cls.tokenizer = CharacterTokenizer()
        cls.tokenizer.train("dataset/corpus.txt")

    def test_train_vocab_not_empty(self):
        """训练后词表不应为空"""
        self.assertGreater(len(self.tokenizer.vocab), 0)

    def test_train_special_tokens_exist(self):
        """词表中应包含所有特殊 Token"""
        for token in CharacterTokenizer.SPECIAL_TOKENS:
            self.assertIn(token, self.tokenizer.vocab)

    def test_encode_returns_list(self):
        """encode 应返回列表"""
        ids = self.tokenizer.encode("你好")
        self.assertIsInstance(ids, list)

    def test_encode_length_matches_text(self):
        """encode 返回的 ID 数量应等于输入字符数"""
        text = "北京今天下雨"
        ids = self.tokenizer.encode(text)
        self.assertEqual(len(ids), len(text))

    def test_decode_returns_string(self):
        """decode 应返回字符串"""
        text = self.tokenizer.decode([0, 1])
        self.assertIsInstance(text, str)

    def test_encode_decode_roundtrip(self):
        """encode → decode 应能还原原文（训练集内字符）"""
        text = "北京今天下雨"
        ids = self.tokenizer.encode(text)
        restored = self.tokenizer.decode(ids)
        self.assertEqual(text, restored)

    def test_unk_handling(self):
        """未在词表中的字符应返回 <UNK> 的 ID"""
        ids = self.tokenizer.encode("😀")
        self.assertEqual(ids, [self.tokenizer.vocab["<UNK>"]])

    def test_id_to_token_sync(self):
        """id_to_token 应与 vocab 保持同步"""
        self.assertEqual(
            len(self.tokenizer.vocab),
            len(self.tokenizer.id_to_token)
        )

    def test_vocab_size_property(self):
        """vocab_size 属性应正确反映词表大小"""
        self.assertEqual(
            self.tokenizer.vocab_size,
            len(self.tokenizer.vocab)
        )

    def test_save_and_load_vocab(self):
        """保存再加载词表，encode/decode 结果应一致"""
        save_path = "tests/_test_vocab.json"
        text = "上海今天晴天"

        ids_before = self.tokenizer.encode(text)

        self.tokenizer.save_vocab(save_path)

        # 新建一个 tokenizer 并加载词表
        new_tokenizer = CharacterTokenizer()
        new_tokenizer.load_vocab(save_path)

        ids_after = new_tokenizer.encode(text)
        self.assertEqual(ids_before, ids_after)

        # 清理临时文件
        Path(save_path).unlink()

    def test_vocab_size(self):
        """vocab_size 应该正确"""
        expected_min = len(CharacterTokenizer.SPECIAL_TOKENS)
        self.assertGreaterEqual(
            self.tokenizer.vocab_size,
            expected_min
        )


if __name__ == "__main__":
    unittest.main()
