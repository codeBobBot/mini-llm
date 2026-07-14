from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import CharacterTokenizer

VOCAB_PATH = "tokenizer/vocab.json"
TEST_CASES = ["深圳", "北京今天下雨", "ChatGPT", "Python3"]  # 涵盖已知 + UNK


def main():
    # ---- 1. 训练原始分词器并保存 ----
    original = CharacterTokenizer()
    original.train("dataset/corpus.txt")
    original.save_vocab(VOCAB_PATH)
    print(f"vocab saved : {VOCAB_PATH}  (size: {original.vocab_size})")

    # ---- 2. 新分词器加载词表 ----
    loaded = CharacterTokenizer()
    loaded.load_vocab(VOCAB_PATH)
    print(f"vocab loaded: {VOCAB_PATH}  (size: {loaded.vocab_size})")

    # ---- 3. 验证 encode / decode 完全一致 ----
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

    print(f"\n{'All tests PASSED' if all_ok else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()