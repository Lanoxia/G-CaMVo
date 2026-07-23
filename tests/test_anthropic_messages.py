import io
import json
import unittest

from camvo.llms.anthropic_messages import AnthropicMessagesLLMClient
from camvo.security.tasks import OPTC_BINARY_DETECTION_TASK
from camvo.types import AnnotationItem, ModelPricing


class _FakeResponse:
    def __init__(self, payload):
        self._stream = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._stream.read()


class AnthropicMessagesTests(unittest.TestCase):
    def test_native_request_and_usage_parsing(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "label": "malicious",
                                    "confidence": 0.93,
                                    "rationale": "suspicious provenance",
                                }
                            ),
                        }
                    ],
                    "usage": {"input_tokens": 101, "output_tokens": 23},
                }
            )

        client = AnthropicMessagesLLMClient(
            "modern/claude-sonnet-4.6",
            ModelPricing(3.0, 15.0),
            provider_model="claude-sonnet-4-6",
            api_key="secret-that-must-not-enter-cache-version",
            task=OPTC_BINARY_DETECTION_TASK,
            temperature=0.35,
            max_retries=0,
            urlopen=fake_urlopen,
        )
        response = client.predict(
            AnnotationItem("event-1", "PROCESS CREATE from unknown actor", ("benign", "malicious"))
        )
        body = json.loads(captured["request"].data)

        self.assertEqual(response.label, "malicious")
        self.assertEqual((response.input_tokens, response.output_tokens), (101, 23))
        self.assertEqual(body["model"], "claude-sonnet-4-6")
        self.assertEqual(body["temperature"], 0.35)
        self.assertEqual(
            captured["request"].get_header("X-api-key"),
            "secret-that-must-not-enter-cache-version",
        )
        self.assertEqual(captured["request"].get_header("Anthropic-version"), "2023-06-01")
        self.assertNotIn("secret-that-must-not-enter-cache-version", client.prompt_version)


if __name__ == "__main__":
    unittest.main()
