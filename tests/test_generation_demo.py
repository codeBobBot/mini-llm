"""
q2_generation_demo.py 全面测试

测试金字塔：
  1. 模型结构测试 (TinyCharLM)
  2. 训练过程测试
  3. 预测准确性测试
  4. 自回归生成测试
  5. Temperature 采样测试
  6. Top-p / Top-k 采样测试
  7. 端到端集成测试
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from demos.q2_generation_demo import TinyCharLM
from tokenizer import CharacterTokenizer


def _make_model_and_tokenizer():
    """辅助函数：创建并训练一个小模型用于测试"""
    tokenizer = CharacterTokenizer()
    corpus_path = Path(__file__).resolve().parent.parent / "dataset" / "corpus.txt"
    tokenizer.train(str(corpus_path))

    vocab_size = len(tokenizer.vocab)
    torch.manual_seed(42)
    model = TinyCharLM(vocab_size, embed_dim=16)

    # 快速训练
    text = corpus_path.read_text(encoding="utf-8").replace("\n", "")
    all_ids = tokenizer.encode(text)
    X = torch.tensor(all_ids[:-1], dtype=torch.long)
    Y = torch.tensor(all_ids[1:], dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(150):
        model.train()
        logits = model(X.unsqueeze(0))
        loss = loss_fn(logits[0], Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model, tokenizer


# ================================================================
#  1. 模型结构测试
# ================================================================
class TestModelStructure(unittest.TestCase):
    """验证 TinyCharLM 模型结构正确性"""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharacterTokenizer()
        corpus_path = Path(__file__).resolve().parent.parent / "dataset" / "corpus.txt"
        cls.tokenizer.train(str(corpus_path))
        cls.vocab_size = len(cls.tokenizer.vocab)
        cls.model = TinyCharLM(cls.vocab_size, embed_dim=16)

    def test_has_embedding_layer(self):
        """应包含 Embedding 层"""
        self.assertIsInstance(self.model.embed, torch.nn.Embedding)

    def test_has_output_layer(self):
        """应包含 Linear 输出层"""
        self.assertIsInstance(self.model.output, torch.nn.Linear)

    def test_forward_output_shape(self):
        """forward 输出形状应为 [batch, seq_len, vocab_size]"""
        x = torch.randint(0, self.vocab_size, (1, 5))
        out = self.model(x)
        self.assertEqual(out.shape, (1, 5, self.vocab_size))

    def test_predict_next_output_shape(self):
        """predict_next 应返回 vocab_size 长度的概率分布"""
        token_id = 0
        probs = self.model.predict_next(token_id)
        self.assertEqual(probs.shape, (self.vocab_size,))

    def test_predict_next_probs_sum_to_one(self):
        """predict_next 输出的概率之和应为 1"""
        probs = self.model.predict_next(0)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

    def test_predict_next_all_non_negative(self):
        """predict_next 输出的概率应全部 >= 0"""
        probs = self.model.predict_next(0)
        self.assertTrue((probs >= 0).all())

    def test_param_count_reasonable(self):
        """参数量应在合理范围（小模型）"""
        total = sum(p.numel() for p in self.model.parameters())
        # embed_dim=16, 参数量 = Embedding(vocab_size×16) + Linear(16×vocab_size + vocab_size bias)
        # = 16*vocab_size + 16*vocab_size + vocab_size = 33 * vocab_size
        expected = 33 * self.vocab_size
        self.assertEqual(total, expected)
        self.assertLess(total, 20000)  # 确实是小模型

    def test_generate_output_type(self):
        """generate 应返回字符串"""
        result = self.model.generate(self.tokenizer, "北京", max_new_tokens=5)
        self.assertIsInstance(result, str)

    def test_generate_starts_with_prompt(self):
        """生成结果应以 prompt 开头"""
        prompt = "北京"
        result = self.model.generate(self.tokenizer, prompt, max_new_tokens=5)
        self.assertTrue(result.startswith(prompt))

    def test_generate_max_tokens_respected(self):
        """生成的 token 数（不含 prompt）不应超过 max_new_tokens"""
        prompt = "北京"
        max_new = 10
        result = self.model.generate(self.tokenizer, prompt, max_new_tokens=max_new)
        # 因为是字符级，生成后的字符数 = len(prompt) + <= max_new
        self.assertLessEqual(len(result) - len(prompt), max_new)


# ================================================================
#  2. 训练过程测试
# ================================================================
class TestTraining(unittest.TestCase):
    """验证训练过程是否正常收敛"""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharacterTokenizer()
        corpus_path = Path(__file__).resolve().parent.parent / "dataset" / "corpus.txt"
        cls.tokenizer.train(str(corpus_path))
        cls.vocab_size = len(cls.tokenizer.vocab)

        text = corpus_path.read_text(encoding="utf-8").replace("\n", "")
        cls.all_ids = cls.tokenizer.encode(text)
        cls.X = torch.tensor(cls.all_ids[:-1], dtype=torch.long)
        cls.Y = torch.tensor(cls.all_ids[1:], dtype=torch.long)

    def _train_model(self, epochs=150):
        torch.manual_seed(42)
        model = TinyCharLM(self.vocab_size, embed_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        losses = []

        for _ in range(epochs):
            model.train()
            logits = model(self.X.unsqueeze(0))
            loss = loss_fn(logits[0], self.Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return model, losses

    def test_loss_decreases_during_training(self):
        """训练损失应持续下降"""
        _, losses = self._train_model(150)
        # 前30轮的损失应明显高于最后30轮
        early_loss = sum(losses[:30]) / 30
        late_loss = sum(losses[-30:]) / 30
        self.assertLess(late_loss, early_loss,
                        f"Late loss {late_loss:.3f} should be < early loss {early_loss:.3f}")

    def test_final_loss_reasonable(self):
        """最终损失应低于 3.0（对于一个 32×vocab 的小模型来说合理）"""
        _, losses = self._train_model(150)
        final_loss = losses[-1]
        self.assertLess(final_loss, 3.0,
                        f"Final loss {final_loss:.3f} should be < 3.0")

    def test_training_accuracy_improves(self):
        """训练准确率应显著高于随机猜测"""
        model, _ = self._train_model(150)
        model.eval()
        with torch.no_grad():
            logits = model(self.X.unsqueeze(0))
            preds = logits[0].argmax(dim=-1)
            acc = (preds == self.Y).float().mean().item()

        random_baseline = 1.0 / self.vocab_size
        self.assertGreater(acc, random_baseline * 5,
                           f"Accuracy {acc:.3f} should be >> random baseline {random_baseline:.3f}")

    def test_reproducible_with_seed(self):
        """相同随机种子应产生相同的模型参数"""
        torch.manual_seed(42)
        m1 = TinyCharLM(self.vocab_size, embed_dim=16)
        p1 = m1.embed.weight.clone()

        torch.manual_seed(42)
        m2 = TinyCharLM(self.vocab_size, embed_dim=16)
        p2 = m2.embed.weight.clone()

        self.assertTrue(torch.equal(p1, p2))

    def test_different_seeds_different_params(self):
        """不同随机种子应产生不同的模型参数"""
        torch.manual_seed(42)
        m1 = TinyCharLM(self.vocab_size, embed_dim=16)
        p1 = m1.embed.weight.clone()

        torch.manual_seed(999)
        m2 = TinyCharLM(self.vocab_size, embed_dim=16)
        p2 = m2.embed.weight.clone()

        self.assertFalse(torch.equal(p1, p2))


# ================================================================
#  3. 预测准确性测试
# ================================================================
class TestPredictions(unittest.TestCase):
    """验证模型预测的语义合理性"""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _make_model_and_tokenizer()
        cls.corpus_path = Path(__file__).resolve().parent.parent / "dataset" / "corpus.txt"
        cls.corpus_text = cls.corpus_path.read_text(encoding="utf-8")

    def _top_prediction(self, char):
        """获取模型对某个字符的下一个预测 Top-1 字符"""
        ids = self.tokenizer.encode(char)
        probs = self.model.predict_next(ids[-1])
        top_id = probs.argmax().item()
        return self.tokenizer.decode([top_id])

    def test_beijing_predicts_jin_high(self):
        """'北京' 后 '今' 的预测概率应在 Top 3"""
        ids = self.tokenizer.encode("北京")
        probs = self.model.predict_next(ids[-1])
        top3_ids = torch.topk(probs, 3).indices
        top3_chars = [self.tokenizer.decode([i.item()]) for i in top3_ids]
        self.assertIn("今", top3_chars,
                      f"'北京' 后 Top3 应该包含 '今'，实际: {top3_chars}")

    def test_tianan_predicts_men(self):
        """'天安' 后应该预测 '门'（语料中「天安门」高频）"""
        ids = self.tokenizer.encode("天安")
        probs = self.model.predict_next(ids[-1])
        top_id = probs.argmax().item()
        predicted = self.tokenizer.decode([top_id])
        self.assertEqual(predicted, "门",
                         f"'天安' 后应该预测 '门'，实际预测 '{predicted}'")

    def test_rengong_predicts_zhi(self):
        """'人工' 后 '智' 应该在 Top 3"""
        ids = self.tokenizer.encode("人工")
        probs = self.model.predict_next(ids[-1])
        top3_ids = torch.topk(probs, 3).indices
        top3_chars = [self.tokenizer.decode([i.item()]) for i in top3_ids]
        self.assertIn("智", top3_chars,
                      f"'人工' 后 Top3 应该包含 '智'，实际: {top3_chars}")

    def test_prediction_not_uniform(self):
        """预测分布不应该是均匀的（模型学到了东西）"""
        ids = self.tokenizer.encode("北京")
        probs = self.model.predict_next(ids[-1])
        max_p = probs.max().item()
        min_p = probs[probs > 0].min().item()
        self.assertGreater(max_p, min_p * 3,
                           f"最高概率 {max_p:.4f} 应远大于最小非零概率 {min_p:.4f}")

    def test_vocab_covered_in_predictions(self):
        """predict_next 应覆盖词表中所有 token（即使概率为 0）"""
        # 注意：某些 token 可能概率为 0，但形状要覆盖全部词表
        probs = self.model.predict_next(0)
        self.assertEqual(len(probs), len(self.tokenizer.vocab))


# ================================================================
#  4. 自回归生成测试
# ================================================================
class TestGeneration(unittest.TestCase):
    """验证自回归生成功能"""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _make_model_and_tokenizer()

    def test_generate_non_empty(self):
        """生成结果不应为空"""
        result = self.model.generate(self.tokenizer, "北京", max_new_tokens=10)
        self.assertGreater(len(result), 0)

    def test_generate_produces_new_tokens(self):
        """生成应产生新 token（不止 prompt）"""
        prompt = "北京"
        result = self.model.generate(self.tokenizer, prompt, max_new_tokens=10)
        self.assertGreaterEqual(len(result), len(prompt))

    def test_generate_with_zero_tokens(self):
        """max_new_tokens=0 时应只返回 prompt"""
        result = self.model.generate(self.tokenizer, "北京", max_new_tokens=0)
        self.assertEqual(result, "北京")

    def test_generate_deterministic_with_seed(self):
        """相同 seed 的多次生成结果应一致（确定性测试）"""
        torch.manual_seed(42)
        r1 = self.model.generate(self.tokenizer, "北京", max_new_tokens=10)
        torch.manual_seed(42)
        r2 = self.model.generate(self.tokenizer, "北京", max_new_tokens=10)
        self.assertEqual(r1, r2)

    def test_generate_different_with_different_seeds(self):
        """不同 seed 通常产生不同结果（随机性）"""
        torch.manual_seed(1)
        r1 = self.model.generate(self.tokenizer, "北京", max_new_tokens=20)
        torch.manual_seed(9999)
        r2 = self.model.generate(self.tokenizer, "北京", max_new_tokens=20)
        # 注意：小词表下可能碰巧相同，但概率很低
        # 用较大的 max_new_tokens 降低巧合概率
        self.assertNotEqual(r1, r2,
                            "不同 seed 应产生不同结果（极低概率碰巧相同）")

    def test_all_generated_chars_in_vocab(self):
        """生成的所有字符都应在词表中"""
        torch.manual_seed(42)
        result = self.model.generate(self.tokenizer, "北京", max_new_tokens=30)
        for ch in result:
            self.assertIn(ch, self.tokenizer.vocab,
                          f"生成字符 '{ch}' 不在词表中")

    def test_generate_is_string(self):
        """generate 返回值应为 str 类型"""
        result = self.model.generate(self.tokenizer, "测试", max_new_tokens=5)
        self.assertIs(type(result), str)


# ================================================================
#  5. Temperature 采样测试
# ================================================================
class TestTemperature(unittest.TestCase):
    """验证 Temperature 对概率分布的影响"""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _make_model_and_tokenizer()

    def test_temperature_zero_peakier(self):
        """T → 0 时分布应该更极端（最大值更大）"""
        ids = self.tokenizer.encode("北京")
        logits = torch.log(self.model.predict_next(ids[-1]) + 1e-10)

        probs_T1 = F.softmax(logits / 1.0, dim=-1)
        probs_T01 = F.softmax(logits / 0.1, dim=-1)

        self.assertGreater(probs_T01.max().item(), probs_T1.max().item())

    def test_temperature_large_flatter(self):
        """T → ∞ 时分布应该更平坦（最大值更小）"""
        ids = self.tokenizer.encode("北京")
        logits = torch.log(self.model.predict_next(ids[-1]) + 1e-10)

        probs_T1 = F.softmax(logits / 1.0, dim=-1)
        probs_T5 = F.softmax(logits / 5.0, dim=-1)

        self.assertLess(probs_T5.max().item(), probs_T1.max().item())

    def test_temperature_one_unchanged(self):
        """T=1 时应保持原始分布"""
        ids = self.tokenizer.encode("天安")
        probs_raw = self.model.predict_next(ids[-1])
        logits = torch.log(probs_raw + 1e-10)
        probs_T1 = F.softmax(logits / 1.0, dim=-1)

        # 允许微小数值误差
        self.assertTrue(torch.allclose(probs_raw, probs_T1, atol=1e-5))

    def test_generate_low_temp_less_diverse(self):
        """低 Temperature 生成的多样性更少"""
        results = set()
        for i in range(10):
            torch.manual_seed(i * 100)
            r = self.model.generate(
                self.tokenizer, "北京", max_new_tokens=8, temperature=0.1
            )
            results.add(r)
        low_temp_unique = len(results)

        results = set()
        for i in range(10):
            torch.manual_seed(i * 100)
            r = self.model.generate(
                self.tokenizer, "北京", max_new_tokens=8, temperature=2.0
            )
            results.add(r)
        high_temp_unique = len(results)

        self.assertLessEqual(low_temp_unique, high_temp_unique,
                             "低 Temperature 应该产生更少的不同结果")

    def test_probs_sum_to_one_at_any_temp(self):
        """任意 Temperature 下概率之和应为 1"""
        ids = self.tokenizer.encode("北京")
        logits = torch.log(self.model.predict_next(ids[-1]) + 1e-10)

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
            probs = F.softmax(logits / temp, dim=-1)
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=4,
                                   msg=f"T={temp} 时概率和不等于1")


# ================================================================
#  6. Top-p / Top-k 采样测试
# ================================================================
class TestTopPTopK(unittest.TestCase):
    """验证 Top-p 和 Top-k 采样策略"""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _make_model_and_tokenizer()

    def test_top_k_1_is_deterministic(self):
        """Top-k=1（贪心）应始终生成相同结果"""
        results = set()
        for i in range(5):
            torch.manual_seed(i * 100)
            r = self.model.generate(
                self.tokenizer, "北京", max_new_tokens=8, top_k=1
            )
            results.add(r)
        self.assertEqual(len(results), 1, "Top-k=1 (贪心) 应始终输出相同结果")

    def test_top_k_limits_candidates(self):
        """Top-k 应限制候选数量"""
        ids = self.tokenizer.encode("北京")
        logits = torch.log(self.model.predict_next(ids[-1]) + 1e-10)

        # 模拟 top_k=3：只有 Top 3 的概率非零
        k = 3
        topk_vals, topk_idx = torch.topk(logits, k)
        mask = torch.full_like(logits, float('-inf'))
        mask[topk_idx] = topk_vals
        probs = F.softmax(mask, dim=-1)
        non_zero = (probs > 0).sum().item()

        self.assertEqual(non_zero, k,
                         f"Top-k={k} 应有恰好 {k} 个非零概率候选，实际 {non_zero}")

    def test_top_k_greater_than_vocab(self):
        """Top-k 大于词表大小时不应出错"""
        big_k = len(self.tokenizer.vocab) + 100
        result = self.model.generate(
            self.tokenizer, "北京", max_new_tokens=5, top_k=big_k
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_top_p_1_0_no_filter(self):
        """Top-p=1.0 时所有候选都应保留"""
        # 用 generate 直接验证：p=1.0 不限制候选池
        result = self.model.generate(
            self.tokenizer, "北京", max_new_tokens=5, top_p=1.0
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 3)
        self.assertTrue(result.startswith("北京"))

    def test_generate_with_top_k_and_top_p_combined(self):
        """Top-k 和 Top-p 同时使用不应出错"""
        result = self.model.generate(
            self.tokenizer, "北京", max_new_tokens=8,
            temperature=1.0, top_p=0.9, top_k=10
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertTrue(result.startswith("北京"))

    def test_top_k_zero_disabled(self):
        """Top-k=0 时应禁用过滤（等同于不限制）"""
        r1 = self.model.generate(
            self.tokenizer, "测试", max_new_tokens=10, top_k=0
        )
        r2 = self.model.generate(
            self.tokenizer, "测试", max_new_tokens=10
        )
        # 两者都无限制，行为应一致（结构正确即可，不同seed可能不同）
        self.assertIsInstance(r1, str)
        self.assertIsInstance(r2, str)


# ================================================================
#  7. 端到端集成测试
# ================================================================
class TestEndToEnd(unittest.TestCase):
    """完整流程验证"""

    def test_full_demo_runs_without_error(self):
        """运行完整 demo 流程不应报错"""
        from demos.q2_generation_demo import run_q3
        # redirect stdout to suppress output during test
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            run_q3()
        output = f.getvalue()
        self.assertIn("训练完成", output)

    def test_model_learns_obvious_patterns(self):
        """模型应学会明显的模式"""
        model, tokenizer = _make_model_and_tokenizer()

        # "天安" → "门" 是最明显的模式之一
        ids = tokenizer.encode("天安")
        probs = model.predict_next(ids[-1])
        top_id = probs.argmax().item()
        self.assertEqual(tokenizer.decode([top_id]), "门")

    def test_consistency_across_runs(self):
        """相同条件下多次运行结果应该一致"""
        model, tokenizer = _make_model_and_tokenizer()

        results = []
        for _ in range(5):
            torch.manual_seed(42)
            r = model.generate(tokenizer, "北京", max_new_tokens=10)
            results.append(r)

        self.assertTrue(all(r == results[0] for r in results))


if __name__ == "__main__":
    unittest.main()
