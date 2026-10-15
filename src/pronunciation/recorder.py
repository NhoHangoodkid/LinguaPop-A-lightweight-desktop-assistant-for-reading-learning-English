"""Microphone recording helpers."""

import numpy as np
from numpy.typing import NDArray


def record_audio(seconds: float, sample_rate: int = 16000) -> NDArray[np.float32]:
    """Record mono float audio from the default microphone."""
    if seconds <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    import sounddevice as sd

    recording = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return recording.reshape(-1)
