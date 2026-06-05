import unittest

from services.yolo_candidate_detector import YoloCandidateDetector


class YoloCandidateDetectorTest(unittest.TestCase):
    def test_scoring_emits_high_confidence_lying_candidate(self):
        detector = YoloCandidateDetector(
            candidate_threshold=0.5,
            min_candidate_gap_ms=1000,
        )
        keypoints = [[0.0, 0.0, 0.0] for _ in range(13)]
        keypoints[5] = [90.0, 100.0, 0.9]
        keypoints[6] = [110.0, 100.0, 0.9]
        keypoints[11] = [190.0, 100.0, 0.9]
        keypoints[12] = [210.0, 100.0, 0.9]
        detection = {
            "bbox": [40.0, 300.0, 220.0, 380.0],
            "confidence": 1.0,
            "keypoints": keypoints,
            "track_id": 1,
        }

        candidate = detector._score_detection(
            detection=detection,
            camera_id="cam1",
            frame_id=7,
            timestamp_ms=2000,
            source_uri="video.mp4",
            width=400,
            height=400,
        )

        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate.score, 0.5)
        self.assertEqual(candidate.candidate_id, "cam1_1_7_2000")
        self.assertTrue(candidate.reason["lying_by_pose"])

    def test_candidate_gap_suppresses_repeated_track(self):
        detector = YoloCandidateDetector(
            candidate_threshold=0.5,
            min_candidate_gap_ms=1000,
        )
        keypoints = [[0.0, 0.0, 0.0] for _ in range(13)]
        keypoints[5] = [90.0, 100.0, 0.9]
        keypoints[6] = [110.0, 100.0, 0.9]
        keypoints[11] = [190.0, 100.0, 0.9]
        keypoints[12] = [210.0, 100.0, 0.9]
        detection = {
            "bbox": [40.0, 300.0, 220.0, 380.0],
            "confidence": 1.0,
            "keypoints": keypoints,
            "track_id": 1,
        }

        first = detector._score_detection(
            detection=detection,
            camera_id="cam1",
            frame_id=1,
            timestamp_ms=0,
            source_uri="",
            width=400,
            height=400,
        )
        second = detector._score_detection(
            detection=detection,
            camera_id="cam1",
            frame_id=2,
            timestamp_ms=500,
            source_uri="",
            width=400,
            height=400,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
