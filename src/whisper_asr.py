from __future__ import annotations

import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Windows + CUDA: help faster-whisper / CTranslate2 locate torch CUDA DLLs.
if os.name == "nt" and torch.cuda.is_available():
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib_dir.exists():
        os.add_dll_directory(str(torch_lib_dir))

from faster_whisper import WhisperModel


@dataclass
class ASRResult:
    text: str
    elapsed: float
    segments: list


class WhisperASR:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
        vad_filter: bool = True,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.6,
        initial_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.initial_prompt = initial_prompt

        print("=== Load faster-whisper ASR ===")
        print(f"model        : {self.model_name}")
        print(f"device       : {self.device}")
        print(f"compute_type : {self.compute_type}")
        print(f"initial_prompt: {'ON' if self.initial_prompt else 'OFF'}")
        print()

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe_bytes(self, raw: bytes, channels: int, rate: int, language: str) -> ASRResult:
        wav_path = self._write_temp_wav(raw, channels, rate)

        try:
            t0 = time.perf_counter()

            segments_iter, _info = self.model.transcribe(
                wav_path,
                language=language,
                task="transcribe",
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                condition_on_previous_text=self.condition_on_previous_text,
                no_speech_threshold=self.no_speech_threshold,
                initial_prompt=self.initial_prompt,
            )

            segments = list(segments_iter)
            text = "".join(seg.text for seg in segments).strip()
            elapsed = time.perf_counter() - t0

            return ASRResult(text=text, elapsed=elapsed, segments=segments)

        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    @staticmethod
    def _write_temp_wav(raw: bytes, channels: int, rate: int) -> str:
        """
        Write temporary WAV.

        Improvement:
          If input is stereo/multi-channel, downmix to mono first.
          This reduces data size and often improves ASR stability for speech.
        """
        audio = np.frombuffer(raw, dtype=np.int16)

        out_channels = channels

        if channels > 1 and audio.size >= channels:
            audio = audio.reshape(-1, channels)
            audio = audio.astype(np.float32).mean(axis=1)
            audio = np.clip(audio, -32768, 32767).astype(np.int16)
            out_channels = 1

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(out_channels)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(audio.tobytes())

        return path
