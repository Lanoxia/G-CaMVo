import io
import json
import unittest

from camvo.llms.dify_workflow import DifyWorkflowLLMClient
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


def _item() -> AnnotationItem:
    return AnnotationItem(
        "event-1",
        "PROCESS CREATE from an unknown actor",
        ("benign", "malicious"),
    )


class DifyWorkflowTests(unittest.TestCase):
    def test_calls_selected_branch_and_parses_aggregate_usage(self) -> None:
        captured = {}
        result = json.dumps(
            {
                "label": "malicious",
                "confidence": 0.93,
                "rationale": "unexpected process chain",
            }
        )

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(
                {
                    "task_id": "task-redacted",
                    "workflow_run_id": "run-redacted",
                    "data": {
                        "status": "succeeded",
                        "outputs": {"result": result, "model_tier": "strong"},
                        "total_tokens": 160,
                        "elapsed_time": 1.25,
                    },
                }
            )

        client = DifyWorkflowLLMClient(
            "dify/strong",
            ModelPricing(3.0, 15.0),
            workflow_model="strong",
            base_url="https://dify.invalid/v1/",
            api_key="secret-that-must-not-enter-the-fingerprint",
            task=OPTC_BINARY_DETECTION_TASK,
            timeout_seconds=90,
            platform_user="minkali",
            user_token="adams-user-token-that-must-not-enter-the-fingerprint",
            urlopen=fake_urlopen,
        )
        response = client.predict(_item())
        request = captured["request"]
        body = json.loads(request.data)

        self.assertEqual(request.full_url, "https://dify.invalid/v1/workflows/run")
        self.assertEqual(request.get_header("Adams-platform-user"), "minkali")
        self.assertEqual(
            request.get_header("Adams-user-token"),
            "adams-user-token-that-must-not-enter-the-fingerprint",
        )
        self.assertEqual(body["inputs"]["model_tier"], "strong")
        self.assertIn("Allowed labels", body["inputs"]["user_prompt"])
        self.assertEqual(body["response_mode"], "blocking")
        self.assertEqual(captured["timeout"], 90)
        self.assertEqual(response.label, "malicious")
        self.assertEqual(response.input_tokens + response.output_tokens, 160)
        self.assertEqual(response.raw["dify_total_tokens"], 160)
        self.assertEqual(
            response.raw["token_split"], "estimated_from_dify_aggregate_total"
        )
        self.assertNotIn("secret-that-must-not-enter-the-fingerprint", client.prompt_version)
        self.assertNotIn("adams-user-token", client.prompt_version)

    def test_rejects_partial_adams_authentication(self) -> None:
        with self.assertRaisesRegex(ValueError, "both be configured"):
            DifyWorkflowLLMClient(
                "dify/cheap-a",
                ModelPricing(0.1, 0.5),
                workflow_model="cheap_a",
                base_url="https://dify.invalid/v1",
                api_key="secret",
                task=OPTC_BINARY_DETECTION_TASK,
                platform_user="minkali",
            )

    def test_rejects_failed_workflow_without_exposing_response_body(self) -> None:
        def fake_urlopen(_request, timeout):
            self.assertGreater(timeout, 0)
            return _FakeResponse(
                {
                    "data": {
                        "status": "failed",
                        "outputs": {},
                        "error": "internal sensitive details",
                    }
                }
            )

        client = DifyWorkflowLLMClient(
            "dify/cheap-a",
            ModelPricing(0.1, 0.5),
            workflow_model="cheap_a",
            base_url="https://dify.invalid/v1",
            api_key="secret",
            task=OPTC_BINARY_DETECTION_TASK,
            urlopen=fake_urlopen,
        )
        with self.assertRaisesRegex(RuntimeError, "invalid structured response") as caught:
            client.predict(_item())
        self.assertNotIn("internal sensitive details", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
