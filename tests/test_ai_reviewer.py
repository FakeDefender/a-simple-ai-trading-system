import unittest
from unittest import mock

from src.utils.ai_reviewer import generate_ai_review


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def is_available(self):
        return True

    def chat(self, system_prompt, user_prompt, **kwargs):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, **kwargs})
        return self.response


class TestAIReviewer(unittest.TestCase):
    def _sample_inputs(self):
        performance = {
            "total_return": -0.18,
            "annual_return": -0.22,
            "sharpe_ratio": -1.8,
            "win_rate": 0.25,
            "profit_factor": 0.38,
            "total_trades": 16,
            "benchmark_return": 0.27,
        }
        risk = {"max_drawdown": -0.24, "volatility": 0.13}
        backtest = {"trade_statistics": {"total_trades": 16, "win_rate": 0.25}}
        params = {
            "position_size": 1.0,
            "allow_short": True,
            "rsi_long_threshold": 55,
            "stop_loss_pct": 0.03,
        }
        return performance, risk, ["基线建议"], backtest, params

    def test_generate_ai_review_uses_local_rules_without_client(self):
        performance, risk, recommendations, backtest, params = self._sample_inputs()

        review = generate_ai_review(
            config={"llm": {"enabled": False}},
            symbol="aapl.us",
            performance=performance,
            risk_metrics=risk,
            recommendations=recommendations,
            backtest_results=backtest,
            strategy_params=params,
        )

        self.assertFalse(review["llm_used"])
        self.assertEqual(review["source"], "local_rules")
        self.assertIn("parameter_suggestions", review)
        self.assertGreater(len(review["risk_notes"]), 0)

    def test_generate_ai_review_uses_available_llm_json(self):
        performance, risk, recommendations, backtest, params = self._sample_inputs()
        client = FakeLLMClient(
            '{"headline":"模型复盘","key_findings":["发现"],"risk_notes":["风险"],'
            '"parameter_suggestions":[{"name":"position_size","current":1,"suggested":0.5,"reason":"降风险"}],'
            '"next_steps":["复测"]}'
        )

        review = generate_ai_review(
            config={"llm": {"enabled": True}},
            symbol="aapl.us",
            performance=performance,
            risk_metrics=risk,
            recommendations=recommendations,
            backtest_results=backtest,
            strategy_params=params,
            llm_client=client,
        )

        self.assertTrue(review["llm_used"])
        self.assertEqual(review["source"], "llm")
        self.assertEqual(review["headline"], "模型复盘")
        self.assertEqual(len(client.calls), 1)

    def test_generate_ai_review_uses_fast_review_limits(self):
        performance, risk, recommendations, backtest, params = self._sample_inputs()
        client = FakeLLMClient(
            '{"headline":"快速复盘","key_findings":["发现"],"risk_notes":["风险"],'
            '"parameter_suggestions":[],"next_steps":["复测"]}'
        )

        review = generate_ai_review(
            config={
                "llm": {
                    "enabled": True,
                    "review_max_tokens": 321,
                    "review_timeout": 4.5,
                    "review_model": "fast-review-model",
                }
            },
            symbol="aapl.us",
            performance=performance,
            risk_metrics=risk,
            recommendations=recommendations,
            backtest_results=backtest,
            strategy_params=params,
            llm_client=client,
        )

        self.assertTrue(review["llm_used"])
        self.assertEqual(client.calls[0]["max_tokens"], 321)
        self.assertEqual(client.calls[0]["timeout"], 4.5)
        self.assertEqual(client.calls[0]["model"], "fast-review-model")
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(client.calls[0]["thinking"], "disabled")
        self.assertLess(len(client.calls[0]["user_prompt"]), 2000)

    def test_generate_ai_review_falls_back_when_llm_returns_non_json(self):
        performance, risk, recommendations, backtest, params = self._sample_inputs()
        client = FakeLLMClient("这不是 JSON")

        review = generate_ai_review(
            config={"llm": {"enabled": True}},
            symbol="aapl.us",
            performance=performance,
            risk_metrics=risk,
            recommendations=recommendations,
            backtest_results=backtest,
            strategy_params=params,
            llm_client=client,
        )

        self.assertFalse(review["llm_used"])
        self.assertEqual(review["source"], "local_rules_after_llm_parse_error")
        self.assertIn("llm_error", review)

    def test_openai_client_disables_sdk_retries_by_default(self):
        from src.utils.openai_client import OpenAIClient

        with mock.patch("src.utils.openai_client.OpenAI") as openai_class:
            OpenAIClient(
                {
                    "llm": {
                        "enabled": True,
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "base_url": "https://api.deepseek.com",
                        "timeout": 7,
                    },
                    "api_keys": {"deepseek": {"api_key": "unit-test-key"}},
                }
            )

        openai_class.assert_called_once_with(
            api_key="unit-test-key",
            base_url="https://api.deepseek.com",
            timeout=7.0,
            max_retries=0,
        )

    def test_openai_client_passes_deepseek_json_and_thinking_options(self):
        from src.utils.openai_client import OpenAIClient

        fake_message = mock.Mock(content='{"ok":true}')
        fake_choice = mock.Mock(message=fake_message)
        fake_response = mock.Mock(choices=[fake_choice])

        with mock.patch("src.utils.openai_client.OpenAI") as openai_class:
            client_instance = openai_class.return_value
            client_instance.chat.completions.create.return_value = fake_response
            client = OpenAIClient(
                {
                    "llm": {
                        "enabled": True,
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "base_url": "https://api.deepseek.com",
                        "thinking": "enabled",
                        "reasoning_effort": "low",
                    },
                    "api_keys": {"deepseek": {"api_key": "unit-test-key"}},
                }
            )

            result = client.chat(
                "system",
                "user",
                response_format={"type": "json_object"},
                max_tokens=12,
            )

        self.assertEqual(result, '{"ok":true}')
        kwargs = client_instance.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertNotIn("temperature", kwargs)


if __name__ == "__main__":
    unittest.main()
