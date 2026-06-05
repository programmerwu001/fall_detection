import importlib
import importlib.util
import unittest


class EventStateTest(unittest.TestCase):
    def test_event_state_module_defines_shared_status_strings(self):
        spec = importlib.util.find_spec("services.event_state")
        self.assertIsNotNone(spec, "services.event_state module should exist")

        event_state = importlib.import_module("services.event_state")

        expected_statuses = {
            "yolo_candidate",
            "vlm_pending",
            "vlm_processing",
            "confirmed_fall",
            "rejected",
            "need_human_review",
            "vlm_failed",
            "privacy_pending",
            "integrity_pending",
            "retention_pending",
            "archived",
        }
        self.assertEqual(set(event_state.ALL_EVENT_STATUSES), expected_statuses)
        self.assertEqual(len(event_state.ALL_EVENT_STATUSES), len(expected_statuses))
        self.assertEqual(event_state.FINAL_DETECTION_STATUSES, (
            event_state.CONFIRMED_FALL,
            event_state.REJECTED,
            event_state.NEED_HUMAN_REVIEW,
        ))


if __name__ == "__main__":
    unittest.main()
