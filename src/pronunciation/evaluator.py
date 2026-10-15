"""Speech-to-text backend used to compare a spoken phrase with its reference."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .scoring import PronunciationResult, assess_pronunciation


class PronunciationEvaluator:
    """Lazy-loaded Faster-Whisper backend with configurable CTranslate2 inference."""

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def evaluate(
        self,
        reference: str,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> PronunciationResult:
        """Transcribe audio with a pretrained model and score it against the reference."""
        if not reference.strip():
            raise ValueError("Select a non-empty English reference phrase first.")
        if sample_rate != 16000:
            raise ValueError("The pretrained speech model requires audio sampled at 16000 Hz.")
        if len(audio) < sample_rate // 2:
            raise ValueError("The recording is too short. Please speak for at least half a second.")

        segments, _ = self._get_model().transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
        )
        transcription = " ".join(segment.text.strip() for segment in segments).strip()
        if not transcription:
            raise ValueError("No speech was recognized. Check the microphone and try again.")
        return assess_pronunciation(reference, transcription)
