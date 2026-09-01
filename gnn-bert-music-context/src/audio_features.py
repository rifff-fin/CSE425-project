from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np


@dataclass
class AudioSegment:
    """Represents one temporal chunk of a track and its extracted features."""

    start_sec: float
    end_sec: float
    signal: np.ndarray
    log_mel: np.ndarray
    chroma: np.ndarray
    mfcc: np.ndarray


class AudioProcessor:
    """Audio preprocessing pipeline for Librosa-based music feature extraction."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_chroma: int = 12,
        n_mfcc: int = 20,
        segment_window_sec: float = 5.0,
        segment_hop_sec: float = 2.5,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc
        self.segment_window_sec = segment_window_sec
        self.segment_hop_sec = segment_hop_sec
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else float(sample_rate / 2)

    def load_audio(self, file_path: str | Path) -> np.ndarray:
        """Load and resample an audio file to the configured sample rate."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        signal, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        if signal.size == 0:
            raise ValueError(f"Loaded audio is empty for file: {path}")

        signal = signal.astype(np.float32)
        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        return signal.astype(np.float32)

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude using z-score-like scaling."""
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D waveform, got shape {signal.shape}")
        if signal.size == 0:
            raise ValueError("Cannot normalize an empty signal.")

        mean = float(np.mean(signal))
        std = float(np.std(signal))
        if std < 1e-8:
            return signal.astype(np.float32)
        return ((signal - mean) / std).astype(np.float32)

    def compute_log_mel(self, signal: np.ndarray) -> np.ndarray:
        """Compute a track-normalized log-mel spectrogram with shape (128, T)."""
        if signal.ndim != 1:
            raise ValueError(f"Expected signal shape (n_samples,), got {signal.shape}")

        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
        log_mel = log_mel.astype(np.float32)

        mel_mean = np.mean(log_mel)
        mel_std = np.std(log_mel)
        if mel_std > 1e-8:
            log_mel = (log_mel - mel_mean) / mel_std
        return log_mel.astype(np.float32)

    def compute_chroma(self, signal: np.ndarray) -> np.ndarray:
        """Compute 12-bin chroma features."""
        if signal.ndim != 1:
            raise ValueError(f"Expected signal shape (n_samples,), got {signal.shape}")

        chroma = librosa.feature.chroma_cqt(
            y=signal,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )
        return chroma.astype(np.float32)

    def compute_mfcc(self, signal: np.ndarray) -> np.ndarray:
        """Compute MFCC features for segment-level node descriptors."""
        if signal.ndim != 1:
            raise ValueError(f"Expected signal shape (n_samples,), got {signal.shape}")

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc.astype(np.float32)

    def segment_signal(self, signal: np.ndarray) -> List[AudioSegment]:
        """Segment a full signal into fixed temporal windows with overlap."""
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D audio signal, got {signal.shape}")
        if signal.size == 0:
            raise ValueError("Cannot segment an empty audio signal.")

        window_samples = int(self.segment_window_sec * self.sample_rate)
        hop_samples = int(self.segment_hop_sec * self.sample_rate)

        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("Window size and hop size must be positive.")

        segments: List[AudioSegment] = []
        for start_idx in range(0, len(signal), hop_samples):
            end_idx = min(start_idx + window_samples, len(signal))
            if end_idx - start_idx < window_samples // 2:
                break

            chunk = signal[start_idx:end_idx]
            if chunk.size < 1:
                continue

            log_mel = self.compute_log_mel(chunk)
            chroma = self.compute_chroma(chunk)
            mfcc = self.compute_mfcc(chunk)

            segments.append(
                AudioSegment(
                    start_sec=start_idx / self.sample_rate,
                    end_sec=end_idx / self.sample_rate,
                    signal=chunk,
                    log_mel=log_mel,
                    chroma=chroma,
                    mfcc=mfcc,
                )
            )

        if not segments:
            raise ValueError("No valid segments were produced from the given audio signal.")
        return segments

    def segment_file(self, file_path: str | Path) -> List[AudioSegment]:
        """Load an audio file and return fixed-window temporal segments."""
        signal = self.load_audio(file_path)
        signal = self.normalize_signal(signal)
        return self.segment_signal(signal)

    def summarize_segment(self, segment: AudioSegment) -> np.ndarray:
        """Convert a segment into a compact node feature vector using chroma + MFCC stats."""
        chroma_mean = np.mean(segment.chroma, axis=1)
        mfcc_mean = np.mean(segment.mfcc, axis=1)
        vector = np.concatenate([chroma_mean.astype(np.float32), mfcc_mean.astype(np.float32)])
        if vector.size == 0:
            raise ValueError("Segment feature vector is empty.")
        return vector.astype(np.float32)


if __name__ == "__main__":
    processor = AudioProcessor()

    synthetic_signal = np.sin(2 * np.pi * 220 * np.linspace(0, 5, 22050 * 5, endpoint=False)).astype(np.float32)
    segments = processor.segment_signal(synthetic_signal)

    print(f"Segment count: {len(segments)}")
    print(f"Log-mel shape: {segments[0].log_mel.shape}")
    print(f"Chroma shape: {segments[0].chroma.shape}")
    print(f"MFCC shape: {segments[0].mfcc.shape}")
    print(f"Node feature shape: {processor.summarize_segment(segments[0]).shape}")
