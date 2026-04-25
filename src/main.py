import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.app_service import run_main_pipeline
from src.utils.config_loader import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _log_key_metrics(summary: Dict[str, Any]):
    if not summary:
        return

    if "final_equity" in summary:
        logger.info("最终权益: %.2f", float(summary["final_equity"]))
    if "total_return" in summary:
        logger.info("总收益率: %.2f%%", float(summary["total_return"]) * 100)
    if "sharpe_ratio" in summary:
        logger.info("夏普比率: %.2f", float(summary["sharpe_ratio"]))
    if "max_drawdown" in summary:
        logger.info("最大回撤: %.2f%%", float(summary["max_drawdown"]) * 100)
    if "fees_paid" in summary:
        logger.info("累计费用: %.2f", float(summary["fees_paid"]))
    if "rebalances" in summary:
        logger.info("调仓次数: %s", summary["rebalances"])


def main():
    try:
        logger.info("开始加载配置")
        config = load_config()
        logger.info("配置加载完成")

        output_root = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        result = run_main_pipeline(config=config, output_dir=output_root)

        logger.info("=" * 60)
        logger.info("主流程完成: run_type=%s", result.get("run_type", "unknown"))
        _log_key_metrics(result.get("summary", {}) or {})
        logger.info("结果目录: %s", result.get("relative_output_dir", output_root))
        logger.info("=" * 60)
    except Exception as exc:
        logger.exception("程序执行失败: %s", exc)
        raise


if __name__ == "__main__":
    main()
