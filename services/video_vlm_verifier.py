"""
Small Video VLM verifier for fuzzy fall candidates.

The YOLO stage should call this verifier only for short candidate clips. The
verifier samples a small number of frames, asks a local multimodal model for a
structured judgement, and returns a normalized result.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import cv2
except ImportError:  # pragma: no cover - handled when clip sampling is used
    cv2 = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError:  # pragma: no cover - handled when images are converted
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


DEFAULT_FALL_VERIFICATION_PROMPT = """
You are verifying a possible fall event in an elder-care monitoring system.

Definition of a fall:
- A person transitions from standing, walking, or sitting into lying or collapsing
  on the floor or ground.
- The person remains down or appears unable to recover normally for a short time.
- For dataset or test-scene videos, a controlled or simulated fall should still
  be marked as confirmed_fall when the visible body rapidly transitions from
  standing, walking, or sitting into lying or collapsing on the floor, ground,
  or a floor mat.

Do not mark these as falls:
- lying on a bed, sofa, or chair
- normal sitting, bending, kneeling, stretching, or caregiver-assisted movement
- unclear cases where the body is heavily occluded
- Do not reject a fall only because the scene looks experimental, there is a
  protective floor mat, or the person later recovers or stands up.

Use only visible evidence in the frames. If evidence is insufficient, return
need_human_review.

Return ONLY valid JSON with this schema（don't omit any symbols）:
{
  "result": "confirmed_fall" | "rejected" | "need_human_review",
  "confidence": 0.0,
  "reason": "short explanation",
  "visible_evidence": ["evidence item"]
}
""".strip()


class VideoVLMVerifierError(RuntimeError):
    """Raised when the Video VLM verifier cannot load or run."""


@dataclass
class VLMVerification:
    """Normalized VLM verification result."""

    camera_id: str
    candidate_id: str
    result: str
    confidence: float
    reason: str
    visible_evidence: List[str]
    raw_response: str
    model_id: str
    timestamp_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        return self.result == "confirmed_fall"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "candidate_id": self.candidate_id,
            "result": self.result,
            "confidence": self.confidence,
            "reason": self.reason,
            "visible_evidence": self.visible_evidence,
            "raw_response": self.raw_response,
            "model_id": self.model_id,
            "timestamp_ms": self.timestamp_ms,
            "metadata": self.metadata,
            "is_confirmed": self.is_confirmed,
        }


class VideoVLMVerifier:
    """
    Verify fuzzy fall candidates with a local multimodal model.

    Supported backends:
        transformers:
            Uses Hugging Face AutoProcessor + AutoModelForImageTextToText when
            available. This is the preferred path for MiniCPM-V 4.6.
        minicpm_chat:
            Uses the cloned third_party/MiniCPM-V chat.py wrapper with a contact
            sheet image. This is useful for legacy MiniCPM-V chat models.
        callable:
            Uses a user-provided callable(images, prompt) -> text.
    """

    VALID_RESULTS = {"confirmed_fall", "rejected", "need_human_review"}

    def __init__(
        self,
        model_id: str = "openbmb/MiniCPM-V-4.6",
        backend: str = "transformers",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_frames: int = 12,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
        prompt_template: str = DEFAULT_FALL_VERIFICATION_PROMPT,
        verifier_callable: Optional[Callable[[List[Any], str], str]] = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be provided")
        if backend not in {"transformers", "minicpm_chat", "callable"}:
            raise ValueError("backend must be transformers, minicpm_chat, or callable")
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than 0")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")
        if backend == "callable" and verifier_callable is None:
            raise ValueError("verifier_callable is required when backend='callable'")

        self.model_id = model_id
        self.backend = backend
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self.prompt_template = prompt_template
        self.verifier_callable = verifier_callable

        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._chat_model: Any = None

    def load(self) -> "VideoVLMVerifier":
        """Load the configured VLM backend lazily."""
        if self.backend == "callable":
            return self
        if self.backend == "transformers":
            self._load_transformers_backend()
            return self
        if self.backend == "minicpm_chat":
            self._load_minicpm_chat_backend()
            return self
        raise VideoVLMVerifierError(f"Unsupported backend: {self.backend}")

    def verify(
        self,
        candidate: Dict[str, Any],
        frames: Optional[Sequence[Any]] = None,
        clip_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a candidate and return a normalized dictionary result."""
        return self.verify_candidate(candidate, frames=frames, clip_path=clip_path).to_dict()

    def verify_candidate(
        self,
        candidate: Dict[str, Any],
        frames: Optional[Sequence[Any]] = None,
        clip_path: Optional[str] = None,
    ) -> VLMVerification:
        """
        Verify a fall candidate using sampled frames or a short video clip path.

        Args:
            candidate: Candidate dictionary from YoloCandidateDetector.
            frames: Optional sequence of OpenCV BGR frames or PIL images.
            clip_path: Optional local video path to sample when frames is omitted.
        """
        images = self._collect_images(frames=frames, clip_path=clip_path)
        if not images:
            raise ValueError("frames or clip_path must provide at least one image")

        prompt = self._build_prompt(candidate)
        raw_response = self._generate(images, prompt)
        parsed = self._parse_response(raw_response)

        return VLMVerification(
            camera_id=str(candidate.get("camera_id", "unknown")),
            candidate_id=str(candidate.get("candidate_id", "")),
            result=parsed["result"],
            confidence=parsed["confidence"],
            reason=parsed["reason"],
            visible_evidence=parsed["visible_evidence"],
            raw_response=raw_response,
            model_id=self.model_id,
            timestamp_ms=int(candidate.get("timestamp_ms", 0)),
            metadata={
                "backend": self.backend,
                "sampled_frame_count": len(images),
                "clip_path": clip_path,
                "candidate_score": candidate.get("score"),
            },
        )

    def _generate(self, images: List[Any], prompt: str) -> str:
        if self.backend == "callable":
            assert self.verifier_callable is not None
            return str(self.verifier_callable(images, prompt))

        self.load()
        if self.backend == "transformers":
            return self._generate_with_transformers(images, prompt)
        if self.backend == "minicpm_chat":
            return self._generate_with_minicpm_chat(images, prompt)
        raise VideoVLMVerifierError(f"Unsupported backend: {self.backend}")

    def _load_transformers_backend(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText

                model_cls = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModel

                model_cls = AutoModel
        except Exception as exc:  # pragma: no cover - dependency specific
            raise VideoVLMVerifierError(
                "Transformers backend requires torch and transformers. Install "
                "the MiniCPM-V requirements before running the verifier."
            ) from exc

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
            )
            self._model.eval()
            self._torch = torch
        except Exception as exc:  # pragma: no cover - model/runtime specific
            raise VideoVLMVerifierError(
                "Failed to load VLM model with transformers. Check model_id, "
                "network/model cache, GPU memory, and dependency versions."
            ) from exc

        logger.info(
            "Loaded Video VLM verifier: backend=transformers model_id=%s",
            self.model_id,
        )

    def _load_minicpm_chat_backend(self) -> None:
        if self._chat_model is not None:
            return

        project_root = Path(__file__).resolve().parents[1]
        minicpm_repo = project_root / "third_party" / "MiniCPM-V"
        if not minicpm_repo.exists():
            raise VideoVLMVerifierError(
                "third_party/MiniCPM-V was not found. Clone OpenBMB/MiniCPM-V first."
            )

        repo_path = str(minicpm_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from chat import MiniCPMVChat

            self._chat_model = MiniCPMVChat(self.model_id)
        except Exception as exc:  # pragma: no cover - model/runtime specific
            raise VideoVLMVerifierError(
                "Failed to load MiniCPM-V chat backend. Check dependencies, "
                "model_id, CUDA availability, and downloaded weights."
            ) from exc

        logger.info(
            "Loaded Video VLM verifier: backend=minicpm_chat model_id=%s",
            self.model_id,
        )

    def _generate_with_transformers(self, images: List[Any], prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
            )

        inputs = self._move_inputs_to_model(inputs)
        input_ids = _get_input_ids(inputs)
        input_length = (
            input_ids.shape[-1]
            if input_ids is not None and hasattr(input_ids, "shape")
            else 0
        )

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, **generate_kwargs)

        if input_length:
            generated = generated[:, input_length:]

        if hasattr(self._processor, "batch_decode"):
            text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        else:
            text = self._processor.decode(generated[0], skip_special_tokens=True)
        return str(text).strip()

    def _generate_with_minicpm_chat(self, images: List[Any], prompt: str) -> str:
        contact_sheet = _make_contact_sheet(images)
        image_b64 = _pil_to_base64(contact_sheet)
        question = json.dumps(
            [{"role": "user", "content": prompt}],
            ensure_ascii=True,
        )
        return str(self._chat_model.chat({"image": image_b64, "question": question}))

    def _move_inputs_to_model(self, inputs: Any) -> Any:
        if not hasattr(inputs, "to"):
            return inputs
        device = getattr(self._model, "device", None)
        if device is None:
            return inputs
        return inputs.to(device)

    def _collect_images(
        self,
        frames: Optional[Sequence[Any]],
        clip_path: Optional[str],
    ) -> List[Any]:
        if frames is not None:
            return _sample_sequence([_to_pil_image(frame) for frame in frames], self.max_frames)
        if clip_path is not None:
            return _sample_clip_images(clip_path, self.max_frames)
        return []

    def _build_prompt(self, candidate: Dict[str, Any]) -> str:
        compact_candidate = {
            "camera_id": candidate.get("camera_id"),
            "candidate_id": candidate.get("candidate_id"),
            "timestamp_ms": candidate.get("timestamp_ms"),
            "track_id": candidate.get("track_id"),
            "yolo_candidate_score": candidate.get("score"),
            "bbox": candidate.get("bbox"),
            "reason": candidate.get("reason"),
        }
        evidence = json.dumps(compact_candidate, ensure_ascii=True, default=str)
        return (
            self.prompt_template
            + "\n\nYOLO candidate evidence:\n"
            + evidence
            + "\n\nThe frames are ordered from earlier to later. Judge the whole sequence."
        )

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        parsed = _extract_json_object(raw_response)
        if parsed is None:
            logger.warning("VLM response did not contain valid JSON: %s", raw_response)
            return _fallback_parse_response(raw_response)

        result = str(parsed.get("result", "")).strip()
        if result not in self.VALID_RESULTS:
            result = _normalize_result_from_text(json.dumps(parsed, ensure_ascii=True))

        confidence = _safe_float(parsed.get("confidence"), default=0.0)
        confidence = max(0.0, min(1.0, confidence))
        reason = str(parsed.get("reason", "")).strip()
        evidence = parsed.get("visible_evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            evidence = []

        return {
            "result": result,
            "confidence": confidence,
            "reason": reason,
            "visible_evidence": [str(item) for item in evidence],
        }


def _sample_clip_images(clip_path: str, max_frames: int) -> List[Any]:
    if cv2 is None:
        raise VideoVLMVerifierError(
            "OpenCV is required to sample frames from clip_path."
        )
    path = Path(clip_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate clip does not exist: {clip_path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoVLMVerifierError(f"Failed to open candidate clip: {clip_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            indices = _even_indices(frame_count, max_frames)
            images = []
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if ok:
                    images.append(_to_pil_image(frame))
            return images

        images = []
        while len(images) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            images.append(_to_pil_image(frame))
        return images
    finally:
        capture.release()


def _sample_sequence(items: Sequence[Any], max_items: int) -> List[Any]:
    if len(items) <= max_items:
        return list(items)
    return [items[index] for index in _even_indices(len(items), max_items)]


def _even_indices(length: int, count: int) -> List[int]:
    if length <= 0 or count <= 0:
        return []
    if count == 1:
        return [0]
    if length <= count:
        return list(range(length))
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def _to_pil_image(frame: Any) -> Any:
    if Image is None:
        raise VideoVLMVerifierError("Pillow is required for VLM image conversion.")
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if not hasattr(frame, "shape"):
        raise TypeError("frames must be OpenCV/numpy arrays or PIL images")

    if len(frame.shape) == 2:
        return Image.fromarray(frame).convert("RGB")
    if len(frame.shape) >= 3 and frame.shape[2] >= 3:
        if cv2 is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame[:, :, ::-1]
        return Image.fromarray(rgb).convert("RGB")
    raise TypeError("unsupported frame shape for image conversion")


def _make_contact_sheet(images: List[Any], columns: int = 4, cell_width: int = 320) -> Any:
    if Image is None or ImageDraw is None or ImageOps is None:
        raise VideoVLMVerifierError("Pillow is required to create contact sheets.")
    if not images:
        raise ValueError("images must not be empty")

    rows = math.ceil(len(images) / float(columns))
    label_height = 24
    cells = []
    for index, image in enumerate(images):
        image = ImageOps.contain(image.convert("RGB"), (cell_width, cell_width))
        cell = Image.new("RGB", (cell_width, cell_width + label_height), "white")
        cell.paste(image, ((cell_width - image.width) // 2, label_height))
        draw = ImageDraw.Draw(cell)
        draw.text((6, 4), f"frame {index + 1}", fill=(0, 0, 0))
        cells.append(cell)

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_width + label_height)),
        "white",
    )
    for index, cell in enumerate(cells):
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_width + label_height)
        sheet.paste(cell, (x, y))
    return sheet


def _pil_to_base64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _fallback_parse_response(raw_response: str) -> Dict[str, Any]:
    result = _normalize_result_from_text(raw_response)
    confidence = 0.5 if result == "confirmed_fall" else 0.0
    if result == "need_human_review":
        confidence = 0.0
    return {
        "result": result,
        "confidence": confidence,
        "reason": "The VLM response was not valid JSON; parsed conservatively.",
        "visible_evidence": [],
    }


def _normalize_result_from_text(text: str) -> str:
    lowered = text.lower()
    if "need_human_review" in lowered or "unclear" in lowered or "insufficient" in lowered:
        return "need_human_review"
    if "confirmed_fall" in lowered or '"fall": true' in lowered:
        return "confirmed_fall"
    if "rejected" in lowered or '"fall": false' in lowered or "not a fall" in lowered:
        return "rejected"
    return "need_human_review"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_input_ids(inputs: Any) -> Any:
    try:
        return inputs["input_ids"]
    except (KeyError, TypeError, AttributeError):
        return getattr(inputs, "input_ids", None)
