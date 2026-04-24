import unittest

from src.utils.config_loader import DEFAULT_CONFIG, load_config


class TestSmoke(unittest.TestCase):
    def test_default_config_has_core_sections(self):
        self.assertIn("data", DEFAULT_CONFIG)
        self.assertIn("strategy", DEFAULT_CONFIG)
        self.assertIn("risk", DEFAULT_CONFIG)
        self.assertIn("backtest", DEFAULT_CONFIG)
        self.assertIn("paper_trading", DEFAULT_CONFIG)
        self.assertIn("portfolio", DEFAULT_CONFIG)
        self.assertIn("live_trading", DEFAULT_CONFIG)
        self.assertIn("live_risk", DEFAULT_CONFIG)
        self.assertIn("broker", DEFAULT_CONFIG)
        self.assertIn("live_service", DEFAULT_CONFIG)
        self.assertIn("llm", DEFAULT_CONFIG)

    def test_load_config_returns_dict(self):
        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn("data", config)
        self.assertIn("strategy", config)
        self.assertIn("risk", config)
        self.assertIn("paper_trading", config)
        self.assertIn("portfolio", config)
        self.assertIn("live_trading", config)
        self.assertIn("live_risk", config)
        self.assertIn("broker", config)
        self.assertIn("live_service", config)


if __name__ == "__main__":
    unittest.main()
