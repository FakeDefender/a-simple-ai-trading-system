import unittest

from src.utils.config_loader import load_config


class TestConfigBehavior(unittest.TestCase):
    def test_llm_is_disabled_without_api_key(self):
        config = load_config()
        self.assertIn("llm", config)
        self.assertIn("enabled", config["llm"])
        self.assertIsInstance(config["llm"]["enabled"], bool)

    def test_symbol_and_interval_exist(self):
        config = load_config()
        self.assertTrue(config["data"]["symbol"])
        self.assertTrue(config["data"]["interval"])


if __name__ == "__main__":
    unittest.main()
