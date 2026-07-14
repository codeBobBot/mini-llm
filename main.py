from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from tokenizer import CharacterTokenizer
from dataset.dataset import TextDataset


def test_tokenizer():
    """验证 tokenizer 序列化 / 反序列化一致性"""
    VOCAB_PATH = "tokenizer/vocab.json"
    TEST_CASES = ["深圳", "北京今天下雨", "ChatGPT", "Python3"]

    original = CharacterTokenizer()
    original.train("dataset/corpus.txt")
    original.save_vocab(VOCAB_PATH)
    print(f"vocab saved : {VOCAB_PATH}  (size: {original.vocab_size})")

    loaded = CharacterTokenizer()
    loaded.load_vocab(VOCAB_PATH)
    print(f"vocab loaded: {VOCAB_PATH}  (size: {loaded.vocab_size})")

    all_ok = True
    for text in TEST_CASES:
        ids_ori = original.encode(text)
        ids_ld  = loaded.encode(text)
        dec_ori = original.decode(ids_ori)
        dec_ld  = loaded.decode(ids_ld)

        ok_ids = ids_ori == ids_ld
        ok_dec = dec_ori == dec_ld
        if not (ok_ids and ok_dec):
            all_ok = False

        status = "PASS" if (ok_ids and ok_dec) else "FAIL"
        print(f"  [{status}] '{text}' -> {ids_ori} -> '{dec_ori}'")

    print(f"Result: {'All PASSED' if all_ok else 'SOME FAILED'}")


def test_text_dataset():
    """
    验证 TextDataset 滑动窗口切分 + Tensor 输出。

    现在 __getitem__ 返回 torch.Tensor (dtype=torch.long)，
    测试验证切片逻辑和数据类型均正确。
    """
    token_ids = [1, 2, 3, 4, 5, 6]
    block_size = 3

    dataset = TextDataset(token_ids, block_size)

    # 期望输出（值）
    expected = [
        ([1, 2, 3], [2, 3, 4]),
        ([2, 3, 4], [3, 4, 5]),
        ([3, 4, 5], [4, 5, 6]),
    ]

    print(f"\ntoken_ids = {token_ids}")
    print(f"block_size = {block_size}")
    print(f"dataset length = {len(dataset)}  (expected: {len(expected)})")
    print()

    all_ok = True
    for i in range(len(dataset)):
        x, y = dataset[i]
        exp_x, exp_y = expected[i]

        # __getitem__ 现在返回 Tensor，tolist() 转回列表对比
        x_list = x.tolist()
        y_list = y.tolist()

        ok = (x_list == exp_x and y_list == exp_y)
        if not ok:
            all_ok = False

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  sample[{i}]  Input: {x_list}  Label: {y_list}  "
              f"dtype: {x.dtype}")

    print(f"\nResult: {'All PASSED' if all_ok else 'SOME FAILED'}")


def test_dataloader():
    """
    演示 DataLoader + 训练循环模板。

    DataLoader 三个核心参数：
        dataset   - TextDataset 实例，负责按索引取样本
        batch_size - 每次从 dataset 取多少个样本拼成一个 batch
        shuffle    - 每个 Epoch 是否随机打乱样本顺序

    为什么 shuffle=True：
        如果语料是 "AAAA... BBBB... CCCC..." 这种分块的，
        不打乱会导致模型先学 A、再学 B、最后学 C，训练容易偏。
        打乱后每个 batch 混合不同内容的样本，训练更稳定。

    训练循环是 PyTorch 通用模板：
        for input_ids, labels in loader:
            # input_ids.shape  -> (batch_size, block_size)
            # labels.shape     -> (batch_size, block_size)
            # 以后在这里：前向传播 → 计算 loss → 反向传播 → 更新参数
    """
    # ---- 1. 准备数据：加载语料并编码 ----
    tokenizer = CharacterTokenizer()
    tokenizer.train("dataset/corpus.txt")
    print(f"vocab_size = {tokenizer.vocab_size}")

    # 读取语料库并编码为 token ID 序列
    text = Path("dataset/corpus.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    print(f"total tokens = {len(token_ids)}")

    # ---- 2. 创建 Dataset ----
    block_size = 8
    dataset = TextDataset(token_ids, block_size)
    print(f"dataset samples = {len(dataset)}  (block_size={block_size})")

    # ---- 3. 创建 DataLoader ----
    batch_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True    # 每个 Epoch 随机打乱，防止模型学偏
    )
    print(f"loader batches = {len(loader)}  (batch_size={batch_size})")
    print()

    # ---- 4. 训练循环模板 ----
    print("--- Training Loop ---")
    for step, (input_ids, labels) in enumerate(loader):
        print(f"Step {step}:")
        print(f"  input_ids.shape = {input_ids.shape}")
        print(f"  labels.shape    = {labels.shape}")
        print(f"  input_ids.dtype = {input_ids.dtype}")

        # shape 解读：
        #   (batch_size, sequence_length)
        #   4 = 一次拿 4 个样本
        #   8 = 每个样本 8 个 token（block_size）

        # --- 扩展点：以后在这里训练模型 ---
        # logits = model(input_ids)       # 前向传播
        # loss = criterion(logits, labels) # 计算 loss
        # optimizer.zero_grad()            # 清除梯度
        # loss.backward()                 # 反向传播
        # optimizer.step()                # 更新参数
        # --------------------------------

        # 只演示前 3 个 batch
        if step >= 2:
            print(f"  ... (showing first 3 batches only)")
            break

    print("\nDone! shape = (batch_size, sequence_length) = "
          f"({batch_size}, {block_size})")
    print("This is exactly the input format Transformer expects.")


def main():
    print("=" * 50)
    print("  Test 1: Tokenizer vocab save/load")
    print("=" * 50)
    test_tokenizer()

    print("\n" + "=" * 50)
    print("  Test 2: TextDataset tensor output")
    print("=" * 50)
    test_text_dataset()

    print("\n" + "=" * 50)
    print("  Test 3: DataLoader + Training Loop")
    print("=" * 50)
    test_dataloader()


if __name__ == "__main__":
    main()
