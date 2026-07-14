import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    基于滑动窗口的语言模型训练数据集。

    它将一个长 token 序列切成等长的 (输入, 标签) 对，标签是输入向后偏移一位，
    实现标准的 **下一 token 预测 (next-token prediction)** 训练目标。

    设计意图：
        语言模型的任务是"给定前文，预测下一个 token"。
        把长文本切成固定长度的窗口，可以让模型在一段连续的上下文上学习。

    示例：
        token_ids = [1, 2, 3, 4, 5, 6]
        block_size = 3

        dataset[0] -> Input: [1, 2, 3]  Label: [2, 3, 4]
        dataset[1] -> Input: [2, 3, 4]  Label: [3, 4, 5]
        dataset[2] -> Input: [3, 4, 5]  Label: [4, 5, 6]

    参数：
        token_ids: 编码后的 token ID 列表
        block_size: 每个样本的上下文窗口大小（即模型一次接收的 token 数）
    """

    def __init__(self, token_ids, block_size):
        """
        初始化数据集。

        参数：
            token_ids: list[int]，完整语料编码后的 ID 序列
            block_size: int，滑动窗口大小
        """
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        """
        返回数据集样本总数。

        为什么是 len(token_ids) - block_size：
            最后一个窗口的起始位置是 len(token_ids) - block_size - 1，
            因为每个样本需要 block_size 个 input token + 1 个 label token，
            总共 block_size + 1 个连续位置。
            所以有效样本数 = 总长度 - block_size。

        返回：
            int: 样本数量
        """
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        """
        返回第 idx 个训练样本，转为 torch.Tensor。

        切片逻辑：
            x = tokens[idx : idx + block_size]         -- 当前窗口的输入
            y = tokens[idx + 1 : idx + block_size + 1] -- 输入向后偏移一位，作为标签

        为什么用 dtype=torch.long：
            Embedding 层的输入必须是整数 ID，不能用 float32。
            因为 Embedding 本质上是一个查表操作，索引必须为整数。

        参数：
            idx: int，样本索引（0 到 len(self)-1）

        返回：
            tuple[Tensor, Tensor]: (x, y)，shape 均为 (block_size,)，dtype 为 torch.long
        """
        # 输入：从 idx 开始取 block_size 个 token
        x = self.token_ids[
            idx: idx + self.block_size
        ]

        # 标签：输入整体向后偏移 1 位，长度同样是 block_size
        y = self.token_ids[
            idx + 1: idx + self.block_size + 1
        ]

        # 模型只认 Tensor，且 Embedding 要求整数类型 (torch.long)
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )