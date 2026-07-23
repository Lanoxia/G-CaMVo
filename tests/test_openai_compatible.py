import io
import json
import tempfile
import unittest
from pathlib import Path

from camvo.llms.openai_compatible import OpenAICompatibleLLMClient
from camvo.security.real_experiment import run_real_provider_experiment
from camvo.security.tasks import OPTC_BINARY_DETECTION_TASK
from camvo.types import AnnotationItem, ModelPricing


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._stream = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._stream.read()


class OpenAICompatibleTests(unittest.TestCase):
    def test_builds_strict_request_and_parses_usage_without_leaking_key(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "label": "malicious",
                                        "confidence": 0.91,
                                        "rationale": "suspicious process chain",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 17},
                }
            )

        client = OpenAICompatibleLLMClient(
            "test-model",
            ModelPricing(1.0, 4.0),
            provider_model="provider/model-v1",
            base_url="https://provider.invalid/v1/",
            api_key="secret-that-must-not-enter-cache-version",
            task=OPTC_BINARY_DETECTION_TASK,
            max_retries=0,
            temperature=0.35,
            extra_body={"reasoning_effort": "low", "model": "cannot-override"},
            urlopen=fake_urlopen,
        )
        item = AnnotationItem(
            "event-1",
            "PROCESS CREATE from unknown actor",
            ("benign", "malicious"),
        )
        response = client.predict(item)
        request_body = json.loads(captured["request"].data)

        self.assertEqual(response.label, "malicious")
        self.assertEqual((response.input_tokens, response.output_tokens), (123, 17))
        self.assertEqual(request_body["model"], "provider/model-v1")
        self.assertEqual(request_body["temperature"], 0.35)
        self.assertEqual(request_body["reasoning_effort"], "low")
        self.assertEqual(captured["timeout"], 60.0)
        self.assertNotIn("secret-that-must-not-enter-cache-version", client.prompt_version)
        self.assertIn(OPTC_BINARY_DETECTION_TASK.prompt_version, client.prompt_version)

    def test_can_omit_temperature_for_reasoning_models(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data)
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "label": "benign",
                                        "confidence": 0.8,
                                        "rationale": "no malicious evidence",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )

        client = OpenAICompatibleLLMClient(
            "reasoning",
            ModelPricing(1.1, 4.4),
            provider_model="o3-mini-2025-01-31",
            base_url="https://provider.invalid/v1",
            api_key="secret",
            task=OPTC_BINARY_DETECTION_TASK,
            temperature=None,
            max_tokens_field="max_completion_tokens",
            max_retries=0,
            urlopen=fake_urlopen,
        )
        client.predict(AnnotationItem("e", "benign process", ("benign", "malicious")))
        self.assertNotIn("temperature", captured["body"])
        self.assertIn("max_completion_tokens", captured["body"])

    def test_real_runner_refuses_unverified_example_prices_before_any_call(self) -> None:
        config = {
            "schema_version": 1,
            "pricing_verified": False,
            "dataset": {"kind": "casie", "path": "unused", "max_items": 1},
            "router": {},
            "graph": {},
            "budget": {},
            "models": [
                {"model_id": "a"},
                {"model_id": "b"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pricing_verified"):
                run_real_provider_experiment(path)


if __name__ == "__main__":
    unittest.main()
