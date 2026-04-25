import unittest
from unittest import mock

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from src.utils.config_loader import load_config


class TestConfigBehavior(unittest.TestCase):
    def test_llm_is_disabled_without_api_key(self):
        with mock.patch("src.utils.config_loader._api_keys_path", return_value="missing-api-keys.yaml"), mock.patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": ""},
            clear=False,
        ):
            config = load_config()
        self.assertIn("llm", config)
        self.assertIn("enabled", config["llm"])
        self.assertIsInstance(config["llm"]["enabled"], bool)
        self.assertFalse(config["llm"]["enabled"])

    def test_symbol_and_interval_exist(self):
        config = load_config()
        self.assertTrue(config["data"]["symbol"])
        self.assertTrue(config["data"]["interval"])


@unittest.skipUnless(pd is not None, "需要安装 pandas 才能运行策略参数测试")
class TestStrategyParameters(unittest.TestCase):
    def test_ma_conditions_use_configured_fast_and_slow_windows(self):
        from src.agents.ml_strategy_agent import MLStrategyAgent

        agent = MLStrategyAgent(
            {
                "strategy": {
                    "fast_ma": 10,
                    "slow_ma": 20,
                    "rsi_long_threshold": 55,
                    "min_volume_ratio": 1.0,
                    "allow_short": True,
                },
                "risk": {},
                "llm": {"enabled": False},
            },
            data_loader=None,
        )
        current_data = pd.Series(
            {
                "ma5": 100.0,
                "ma10": 90.0,
                "ma20": 95.0,
                "rsi": 60.0,
                "macd": 1.0,
                "signal": 0.0,
                "volume_ratio": 1.1,
            }
        )

        self.assertFalse(agent._check_long_entry_conditions(current_data, {"trend_strength": 0.01}))

    def test_event_factor_can_suppress_signal_when_risk_is_high(self):
        from src.agents.ml_strategy_agent import MLStrategyAgent

        agent = MLStrategyAgent(
            {
                "data": {"symbol": "sample.us"},
                "strategy": {
                    "fast_ma": 10,
                    "slow_ma": 20,
                    "rsi_long_threshold": 55,
                    "min_volume_ratio": 1.0,
                    "allow_short": False,
                },
                "risk": {},
                "llm": {"enabled": False},
            },
            data_loader=None,
        )
        factor = {
            "event_risk_score": 0.9,
            "confidence_score": 0.9,
            "direction_hint": "bullish",
        }

        self.assertEqual(agent._apply_event_factor_filter(1, factor), 0)


if __name__ == "__main__":
    unittest.main()
