import unittest


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def is_available(self):
        return True

    def chat(self, system_prompt, user_prompt, **kwargs):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, **kwargs})
        return self.response


class TestLlmEventFactor(unittest.TestCase):
    def test_generate_llm_event_factor_returns_fallback_when_disabled(self):
        from src.utils.llm_event_factor import generate_llm_event_factor

        factor = generate_llm_event_factor(
            config={"llm": {"enabled": False}},
            symbol="aapl.us",
            latest_snapshot={"close": 100.0},
            llm_client=None,
        )

        self.assertEqual(factor["source"], "disabled")
        self.assertFalse(factor["llm_used"])

    def test_generate_llm_event_factor_parses_structured_json(self):
        from src.utils.llm_event_factor import generate_llm_event_factor

        client = FakeLLMClient(
            '{"headline":"财报前观望","sentiment_score":0.25,"event_risk_score":0.6,'
            '"confidence_score":0.8,"direction_hint":"bullish"}'
        )
        factor = generate_llm_event_factor(
            config={"llm": {"enabled": True, "event_factor_enabled": True}},
            symbol="aapl.us",
            latest_snapshot={"close": 100.0},
            llm_client=client,
        )

        self.assertTrue(factor["llm_used"])
        self.assertEqual(factor["source"], "llm_event_factor")
        self.assertEqual(factor["direction_hint"], "bullish")
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
