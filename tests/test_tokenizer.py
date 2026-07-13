import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保能导入 tokenizer 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer()

tokenizer.train("dataset/corpus.txt")

ids = tokenizer.encode("深圳")

print(ids)

text = tokenizer.decode(ids)

print(text)