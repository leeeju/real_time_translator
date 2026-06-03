from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio


@dataclass
class AudioChunk:
    raw: bytes
    channels: int
    rate: int
    rms: float
    chunk_start: float
    overlap_seconds: float = 0.0


def list_audio_devices() -> None:
    pa = pyaudio.PyAudio()
    try:
        print("=== Audio Devices ===")
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            print(
                f"[{i}] "
                f"in={d.get('maxInputChannels')}, "
                f"out={d.get('maxOutputChannels')}, "
                f"loopback={d.get('isLoopbackDevice', False)}, "
                f"rate={int(d.get('defaultSampleRate', 0))}, "
                f"name={d.get('name')}"
            )
    finally:
        pa.terminate()


def audio_rms(raw_bytes: bytes) -> float:
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


class AudioCapture:
    """
    Windows microphone / WASAPI loopback audio capture.

    overlap_seconds:
      Adds the previous tail to the current ASR chunk.
      This gives Whisper context near chunk boundaries without increasing wait time.
    """

    def __init__(
        self,
        source: str = "loopback",
        chunk_seconds: float = 2.0,
        overlap_seconds: float = 0.3,
        rms_threshold: float = 300.0,
        device_index: Optional[int] = None,
    ):
        self.source = source
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = max(0.0, overlap_seconds)
        self.rms_threshold = rms_threshold
        self.device_index = device_index

    def run(self, audio_queue: queue.Queue, stop_event: threading.Event) -> None:
        pa = pyaudio.PyAudio()

        try:
            if self.device_index is not None:
                device = pa.get_device_info_by_index(self.device_index)
            elif self.source == "mic":
                device = pa.get_default_input_device_info()
            elif self.source == "loopback":
                device = self._get_default_loopback_device(pa)
            else:
                raise ValueError(f"Unsupported audio source: {self.source}")

            channels = int(device["maxInputChannels"])
            rate = int(device["defaultSampleRate"])

            if channels <= 0:
                raise RuntimeError(f"Selected device has no input channels: {device['name']}")

            frames_per_buffer = 1024
            frames_per_chunk = int(rate * self.chunk_seconds)
            overlap_frames = int(rate * self.overlap_seconds)
            bytes_per_frame = channels * 2
            overlap_bytes = overlap_frames * bytes_per_frame
            previous_tail = b""

            print("=== Audio Capture ===")
            print(f"source   : {self.source}")
            print(f"index    : {device['index']}")
            print(f"name     : {device['name']}")
            print(f"channels : {channels}")
            print(f"rate     : {rate}")
            print(f"chunk    : {self.chunk_seconds:.2f} sec")
            print(f"overlap  : {self.overlap_seconds:.2f} sec")
            print(f"rms th   : {self.rms_threshold}")
            print()

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=frames_per_buffer,
            )

            try:
                while not stop_event.is_set():
                    frames = []
                    captured = 0
                    chunk_start = time.perf_counter()

                    while captured < frames_per_chunk and not stop_event.is_set():
                        data = stream.read(frames_per_buffer, exception_on_overflow=False)
                        frames.append(data)
                        captured += frames_per_buffer

                    current_raw = b"".join(frames)
                    rms = audio_rms(current_raw)

                    if rms < self.rms_threshold:
                        if overlap_bytes > 0:
                            previous_tail = current_raw[-overlap_bytes:]
                        continue

                    asr_raw = previous_tail + current_raw if previous_tail else current_raw

                    if overlap_bytes > 0:
                        previous_tail = current_raw[-overlap_bytes:]

                    item = AudioChunk(
                        raw=asr_raw,
                        channels=channels,
                        rate=rate,
                        rms=rms,
                        chunk_start=chunk_start,
                        overlap_seconds=self.overlap_seconds,
                    )

                    if audio_queue.full():
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            pass

                    audio_queue.put(item)

            finally:
                stream.stop_stream()
                stream.close()

        finally:
            pa.terminate()

    @staticmethod
    def _get_default_loopback_device(pa: pyaudio.PyAudio):
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as exc:
            raise RuntimeError("WASAPI is unavailable. Check Windows audio devices.") from exc

        default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if default_speakers.get("isLoopbackDevice"):
            return default_speakers

        for loopback in pa.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                return loopback

        raise RuntimeError("Could not find loopback device for default output device.")
