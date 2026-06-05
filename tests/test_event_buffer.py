import unittest

from services.event_buffer import EventBuffer, EventBufferError


class EventBufferTest(unittest.TestCase):
    def test_prunes_by_time_window_and_returns_copies(self):
        buffer = EventBuffer(max_seconds=1.0)
        buffer.extend(
            [
                {"camera_id": "cam1", "timestamp_ms": 0, "frame_id": 0},
                {"camera_id": "cam1", "timestamp_ms": 500, "frame_id": 1},
                {"camera_id": "cam1", "timestamp_ms": 1500, "frame_id": 2},
            ]
        )

        self.assertEqual(buffer.size("cam1"), 2)
        window = buffer.get_window("cam1", 0, 1500)
        self.assertEqual([packet["frame_id"] for packet in window], [1, 2])

        window[0]["frame_id"] = 99
        self.assertEqual(buffer.get_all("cam1")[0]["frame_id"], 1)

    def test_rejects_invalid_packets(self):
        buffer = EventBuffer()

        with self.assertRaises(EventBufferError):
            buffer.append({"camera_id": "", "timestamp_ms": 1})

        with self.assertRaises(EventBufferError):
            buffer.append({"camera_id": "cam1"})

        with self.assertRaises(EventBufferError):
            buffer.append({"camera_id": "cam1", "timestamp_ms": "bad"})

    def test_get_recent_uses_latest_timestamp(self):
        buffer = EventBuffer(max_seconds=5.0)
        buffer.extend(
            [
                {"camera_id": "cam1", "timestamp_ms": 0, "frame_id": 0},
                {"camera_id": "cam1", "timestamp_ms": 1000, "frame_id": 1},
                {"camera_id": "cam1", "timestamp_ms": 2500, "frame_id": 2},
            ]
        )

        recent = buffer.get_recent("cam1", seconds=1.5)

        self.assertEqual([packet["frame_id"] for packet in recent], [1, 2])


if __name__ == "__main__":
    unittest.main()
