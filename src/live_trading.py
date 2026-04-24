import logging
import os
import sys

if __package__ in {None, ''}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.execution.live_trading_service import LiveTradingService
from src.utils.config_loader import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    service = LiveTradingService(config)
    payload = service.run_forever()
    if payload is None:
        logger.warning("live trading service 未返回有效结果")
        return

    summary = payload["summary"]
    logger.info(
        "live trading service 完成: latest=%s, processed_rows=%s, final_equity=%.2f, total_return=%.2f%%, rejected=%s, canceled=%s, output=%s",
        payload["latest_timestamp"],
        payload["processed_rows"],
        summary["final_equity"],
        summary["total_return"] * 100,
        summary["rejected_orders"],
        summary["canceled_orders"],
        payload["results_dir"],
    )


if __name__ == "__main__":
    main()
