import unittest

from services.video_vlm_verifier import (
    DEFAULT_FALL_VERIFICATION_PROMPT,
    VideoVLMVerifier,
    _even_indices,
    _get_input_ids,
)


class AttrInputs:
    input_ids = "attr_ids"


class VideoVLMVerifierTest(unittest.TestCase):
    def test_parse_valid_json_response(self):
        verifier = VideoVLMVerifier(
            backend="callable",
            model_id="test-model",
            verifier_callable=lambda images, prompt: "{}",
        )

        parsed = verifier._parse_response(
            """
            ```json
            {"result": "confirmed_fall", "confidence": 0.82, "reason": "down", "visible_evidence": "lying on floor"}
            ```
            """
        )

        self.assertEqual(parsed["result"], "confirmed_fall")
        self.assertEqual(parsed["confidence"], 0.82)
        self.assertEqual(parsed["visible_evidence"], ["lying on floor"])

    def test_parse_invalid_response_falls_back_to_review(self):
        verifier = VideoVLMVerifier(
            backend="callable",
            model_id="test-model",
            verifier_callable=lambda images, prompt: "{}",
        )

        with self.assertLogs("services.video_vlm_verifier", level="WARNING"):
            parsed = verifier._parse_response("body is heavily occluded and unclear")

        self.assertEqual(parsed["result"], "need_human_review")
        self.assertEqual(parsed["confidence"], 0.0)

    def test_even_indices_cover_edges(self):
        self.assertEqual(_even_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(_even_indices(3, 5), [0, 1, 2])
        self.assertEqual(_even_indices(0, 5), [])

    def test_get_input_ids_supports_mapping_and_attribute_inputs(self):
        self.assertEqual(_get_input_ids({"input_ids": "dict_ids"}), "dict_ids")
        self.assertEqual(_get_input_ids(AttrInputs()), "attr_ids")
        self.assertIsNone(_get_input_ids(object()))

    def test_default_prompt_accepts_simulated_falls_on_floor_mats(self):
        self.assertIn("simulated", DEFAULT_FALL_VERIFICATION_PROMPT)
        self.assertIn("controlled", DEFAULT_FALL_VERIFICATION_PROMPT)
        self.assertIn("floor mat", DEFAULT_FALL_VERIFICATION_PROMPT)
        self.assertIn("later recovers", DEFAULT_FALL_VERIFICATION_PROMPT)


if __name__ == "__main__":
    unittest.main()
