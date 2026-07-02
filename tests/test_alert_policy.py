import unittest

from services.alert_policy import (
    HIGH_RISK,
    LOW_RISK,
    NO_ALARM,
    map_vlm_decision,
)


class AlertPolicyTest(unittest.TestCase):
    def test_maps_vlm_results_to_risk_levels(self):
        self.assertEqual(map_vlm_decision("confirmed_fall").risk_level, HIGH_RISK)
        self.assertEqual(map_vlm_decision("need_human_review").risk_level, LOW_RISK)
        self.assertEqual(map_vlm_decision("uncertain").risk_level, LOW_RISK)
        self.assertEqual(map_vlm_decision("rejected").risk_level, NO_ALARM)

    def test_maps_vlm_failure_to_low_risk_degraded_yolo_fallback(self):
        decision = map_vlm_decision("failed")

        self.assertEqual(decision.risk_level, LOW_RISK)
        self.assertEqual(decision.decision_source, "yolo_fallback")
        self.assertTrue(decision.system_degraded)
        self.assertEqual(decision.vlm_status, "failed")

    def test_maps_vlm_timeout_to_low_risk_degraded_yolo_fallback(self):
        decision = map_vlm_decision("timeout")

        self.assertEqual(decision.risk_level, LOW_RISK)
        self.assertEqual(decision.decision_source, "yolo_fallback")
        self.assertTrue(decision.system_degraded)
        self.assertEqual(decision.vlm_status, "timeout")


if __name__ == "__main__":
    unittest.main()
