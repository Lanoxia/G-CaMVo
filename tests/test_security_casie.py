import json
import tempfile
import unittest
from pathlib import Path

from camvo.security.casie import (
    CASIE_LABELS,
    build_casie_event_adjacency,
    load_casie_event_items,
)
from camvo.security.splits import grouped_split, stratified_grouped_split
from camvo.security.sampling import graph_preserving_group_sample
from camvo.types import AnnotationItem


class CasieLoaderTests(unittest.TestCase):
    def test_loads_marked_event_mentions_and_tolerates_bad_events(self) -> None:
        content = "A phishing email reached staff. A patch fixed the vulnerability."
        phishing = content.index("phishing email")
        patch = content.index("patch")
        payload = {
            "content": content,
            "info": {"title": "Test incident"},
            "cyberevent": {
                "hopper": [
                    {
                        "events": [
                            {
                                "index": "E1",
                                "type": "Attack",
                                "subtype": "Phishing",
                                "realis": "Actual",
                                "nugget": {
                                    "startOffset": phishing,
                                    "endOffset": phishing + len("phishing email"),
                                    "text": "phishing email",
                                },
                                "argument": [{"role": {"type": "Victim"}}],
                            },
                            {
                                "index": "E2",
                                "type": "Vulnerability-related",
                                "subtype": "PatchVulnerability",
                                "realis": "Actual",
                                "nugget": {
                                    "startOffset": patch,
                                    "endOffset": patch + len("patch"),
                                    "text": "patch",
                                },
                            },
                            {
                                "index": "bad",
                                "type": "Attack",
                                "subtype": "Unknown",
                                "nugget": {"startOffset": 0, "endOffset": 1, "text": "A"},
                            },
                        ]
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "annotation"
            annotation.mkdir()
            (annotation / "1.json").write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_casie_event_items(directory, context_chars=64)

        self.assertEqual(dataset.stats.files_scanned, 1)
        self.assertEqual(dataset.stats.events_loaded, 2)
        self.assertEqual(dataset.stats.events_skipped, 1)
        self.assertEqual(len(dataset.items), 2)
        self.assertEqual(dataset.items[0].labels, CASIE_LABELS)
        self.assertIn("[EVENT]", dataset.items[0].text)
        self.assertEqual(dataset.items[0].metadata["gold_label"], "Phishing")

        adjacency = build_casie_event_adjacency(dataset.items)
        self.assertIn(dataset.items[1].item_id, adjacency[dataset.items[0].item_id])

    def test_grouped_split_never_separates_one_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "annotation"
            annotation.mkdir()
            for document_id in range(20):
                content = "A phishing message was reported. A patch was released."
                first = content.index("phishing")
                second = content.index("patch")
                events = []
                for index, (start, text, subtype) in enumerate(
                    (
                        (first, "phishing", "Phishing"),
                        (second, "patch", "PatchVulnerability"),
                    )
                ):
                    events.append(
                        {
                            "index": f"E{index}",
                            "type": "Security",
                            "subtype": subtype,
                            "realis": "Actual",
                            "nugget": {
                                "startOffset": start,
                                "endOffset": start + len(text),
                                "text": text,
                            },
                            "argument": [],
                        }
                    )
                payload = {
                    "content": content,
                    "info": {"title": f"Document {document_id}"},
                    "cyberevent": {"hopper": [{"events": events}]},
                }
                (annotation / f"{document_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            dataset = load_casie_event_items(directory, context_chars=64, strict=True)
            split = grouped_split(
                dataset.items,
                group_key=lambda item: str(item.metadata["document_id"]),
                calibration_fraction=0.25,
                validation_fraction=0.25,
                seed=9,
            )

        locations = {}
        for name, partition in (
            ("calibration", split.calibration),
            ("validation", split.validation),
            ("test", split.test),
        ):
            for item in partition:
                document_id = str(item.metadata["document_id"])
                locations.setdefault(document_id, set()).add(name)
        self.assertTrue(all(len(location) == 1 for location in locations.values()))
        self.assertEqual(sum(map(len, locations.values())), 20)

    def test_missing_dataset_is_explicit(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_casie_event_items("does-not-exist")

    def test_stratified_grouped_split_preserves_groups_and_label_coverage(self) -> None:
        labels = ("a", "b", "c")
        items = [
            AnnotationItem(
                item_id=f"doc-{document}:event-{event}",
                text=f"document {document} event {event}",
                labels=labels,
                metadata={
                    "document_id": f"doc-{document}",
                    "gold_label": labels[document % len(labels)],
                },
            )
            for document in range(30)
            for event in range(2)
        ]
        split = stratified_grouped_split(
            items,
            group_key=lambda item: str(item.metadata["document_id"]),
            calibration_fraction=0.2,
            validation_fraction=0.2,
            seed=5,
        )

        seen: dict[str, str] = {}
        for name, partition in (
            ("calibration", split.calibration),
            ("validation", split.validation),
            ("test", split.test),
        ):
            self.assertEqual({str(item.metadata["gold_label"]) for item in partition}, set(labels))
            for item in partition:
                document = str(item.metadata["document_id"])
                if document in seen:
                    self.assertEqual(seen[document], name)
                else:
                    seen[document] = name

    def test_stratified_grouped_split_scales_beyond_pair_swap_threshold(self) -> None:
        labels = ("a", "b", "c")
        items = [
            AnnotationItem(
                item_id=f"doc-{document}:event",
                text=f"document {document}",
                labels=labels,
                metadata={
                    "document_id": f"doc-{document}",
                    "gold_label": labels[document % len(labels)],
                },
            )
            for document in range(300)
        ]

        split = stratified_grouped_split(
            items,
            group_key=lambda item: str(item.metadata["document_id"]),
            calibration_fraction=0.2,
            validation_fraction=0.1,
            seed=17,
        )

        self.assertEqual(
            (len(split.calibration), len(split.validation), len(split.test)),
            (60, 30, 210),
        )
        for partition in (split.calibration, split.validation, split.test):
            self.assertEqual(
                {str(item.metadata["gold_label"]) for item in partition}, set(labels)
            )

    def test_graph_preserving_sampler_keeps_document_blocks_and_exact_size(self) -> None:
        items = []
        for document in range(8):
            for offset in range(3):
                label = CASIE_LABELS[(document + offset) % len(CASIE_LABELS)]
                items.append(
                    AnnotationItem(
                        item_id=f"d{document}:e{offset}",
                        text=f"document {document} event {offset}",
                        labels=CASIE_LABELS,
                        metadata={
                            "document_id": str(document),
                            "start_offset": offset,
                            "gold_label": label,
                        },
                    )
                )
        sampled = graph_preserving_group_sample(
            items,
            12,
            7,
            group_key=lambda item: str(item.metadata["document_id"]),
            order_key=lambda item: int(item.metadata["start_offset"]),
        )
        self.assertEqual(len(sampled), 12)
        positions = {}
        for index, item in enumerate(sampled):
            positions.setdefault(item.metadata["document_id"], []).append(index)
        self.assertTrue(
            all(indices == list(range(min(indices), max(indices) + 1)) for indices in positions.values())
        )


if __name__ == "__main__":
    unittest.main()
