import argparse
import json
import os
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.utils.walk_forward import run_walk_forward


def main():
    parser = argparse.ArgumentParser(description="运行单标的 walk-forward 验证")
    parser.add_argument("--train-size", type=int, default=84, help="训练窗口 bars 数")
    parser.add_argument("--test-size", type=int, default=21, help="测试窗口 bars 数")
    parser.add_argument("--step-size", type=int, default=None, help="窗口滚动步长，默认等于 test-size")
    parser.add_argument("--output-dir", default=None, help="结果目录，默认写入 results/walk_forward/<timestamp>")
    parser.add_argument("--no-paper", action="store_true", help="只跑研究回测，不跑 paper")
    args = parser.parse_args()

    payload = run_walk_forward(
        config=load_config(),
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        include_paper=not args.no_paper,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"output_dir={payload['output_dir']}")


if __name__ == "__main__":
    main()
