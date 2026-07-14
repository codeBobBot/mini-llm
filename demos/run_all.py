#!/usr/bin/env python3
"""
LLM 知识分享 —— 现场代码演示总入口

用法:
    python demos/run_all.py          # 依次运行三个演示
    python demos/run_all.py q1       # 只运行 Q1
    python demos/run_all.py q2       # 只运行 Q2
    python demos/run_all.py q3       # 只运行 Q3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════╗
║    三问 LLM 底牌 —— BPE · Transformer · RoPE              ║
║    现场代码演示                                           ║
╚═══════════════════════════════════════════════════════════╝
""")


def main():
    print_banner()

    args = sys.argv[1:]

    # 允许只运行单个演示
    run_all = len(args) == 0
    run_q1 = run_all or "q1" in args
    run_q2 = run_all or "q2" in args
    run_q3 = run_all or "q3" in args

    if run_q1:
        from demos.q1_bpe_demo import run_q1
        run_q1()

    if run_q2:
        from demos.q2_transformer_demo import run_q2
        run_q2()

    if run_q3:
        from demos.q3_generation_demo import run_q3
        run_q3()

    if run_all:
        print(f"\n{'=' * 55}")
        print(f"  所有演示完成！总结:")
        print(f"  Q1 → BPE 分词器：Token 计费为何不同")
        print(f"  Q2 → Transformer + RoPE：厂商核心差异在哪")
        print(f"  Q3 → 生成机制：答案为何次次不同")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
