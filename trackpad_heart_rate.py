#!/usr/bin/env python3
"""
MacBook trackpad heart-rate estimator with the clean white UI.

This is not a medical device. It estimates a pulse only when the trackpad
contact/pressure stream has a stable repeating component in the human
heart-rate band.

The BPM is computed from completed 15-second measurement batches. The displayed
number is not a continuously-smoothed tracker, so a stale low-frequency guess
cannot drag the result toward 50 BPM while a new measurement is still collecting.

Run:
    python3 trackpad_heart_rate.py

Useful:
    python3 trackpad_heart_rate.py --touch-probe
    python3 trackpad_heart_rate.py --pressure-only
    python3 trackpad_heart_rate.py --simulate
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal


# ============================================================
# DETECTOR / INPUT CONSTANTS
# ============================================================

TARGET_SAMPLE_RATE_HZ = 60.0
MEASUREMENT_SECONDS = 15.0
WINDOW_SECONDS = MEASUREMENT_SECONDS
MIN_SECONDS = 14.2
MIN_ACTIVE_FRACTION = 0.90
MIN_BPM = 45.0
MAX_BPM = 130.0
RESULT_HOLD_SECONDS = 3.0
CONTACT_LATCH_SECONDS = 0.45
CONTACT_DROPOUT_RESET_SECONDS = 0.85
CONTACT_PRESSURE_MIN = 0.0005
TOUCH_CONTACT_MAX_AGE_SECONDS = 0.35
TOUCH_CONTACT_MIN = 1e-7
MAX_TOUCH_CONTACTS = 16
DEFAULT_BPM = 72.0


@dataclass(frozen=True)
class HeartRateResult:
    detected: bool
    bpm: float | None
    confidence: float
    reason: str
    active_seconds: float
    peak_ratio: float = 0.0
    autocorr_peak: float = 0.0
    remaining_seconds: int | None = None


@dataclass(frozen=True)
class BpmCandidate:
    bpm: float
    peak_ratio: float
    dominance: float
    autocorr_peak: float
    stability: float
    score: float


@dataclass(frozen=True)
class TrackpadSignal:
    value: float
    active: bool
    source: str
    fingers: int
    events: int
    age: float
    error: str | None = None


class MTPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
    ]


class MTVector(ctypes.Structure):
    _fields_ = [
        ("position", MTPoint),
        ("velocity", MTPoint),
    ]


class MTContact(ctypes.Structure):
    """Best-known MultitouchSupport contact layout used by Mac trackpads."""

    _fields_ = [
        ("frame", ctypes.c_int),
        ("timestamp", ctypes.c_double),
        ("identifier", ctypes.c_int),
        ("state", ctypes.c_int),
        ("finger_id", ctypes.c_int),
        ("hand_id", ctypes.c_int),
        ("normalized", MTVector),
        ("size", ctypes.c_float),
        ("unknown3", ctypes.c_int),
        ("angle", ctypes.c_float),
        ("major_axis", ctypes.c_float),
        ("minor_axis", ctypes.c_float),
        ("unknown4", MTVector),
        ("unknown5", ctypes.c_int),
        ("unknown6", ctypes.c_int),
        ("unknown7", ctypes.c_float),
    ]


class TouchContactSampler:
    """Reads light finger contacts from macOS's private MultitouchSupport API."""

    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._value = 0.0
        self._finger_count = 0
        self._events = 0
        self._last_update = 0.0
        self._devices: list[int] = []
        self._mt = None
        self._cf = None
        self._callback = None
        self._callback_type = None

    def start(self) -> bool:
        try:
            self._mt = ctypes.CDLL(
                "/System/Library/PrivateFrameworks/"
                "MultitouchSupport.framework/MultitouchSupport"
            )
            self._cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )

            self._cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
            self._cf.CFArrayGetCount.restype = ctypes.c_long
            self._cf.CFArrayGetValueAtIndex.argtypes = [
                ctypes.c_void_p,
                ctypes.c_long,
            ]
            self._cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p

            self._callback_type = ctypes.CFUNCTYPE(
                None,
                ctypes.c_void_p,
                ctypes.POINTER(MTContact),
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
            )
            self._callback = self._callback_type(self._handle_frame)

            self._mt.MTDeviceCreateList.argtypes = []
            self._mt.MTDeviceCreateList.restype = ctypes.c_void_p
            self._mt.MTRegisterContactFrameCallback.argtypes = [
                ctypes.c_void_p,
                self._callback_type,
            ]
            self._mt.MTRegisterContactFrameCallback.restype = None
            self._mt.MTDeviceStart.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self._mt.MTDeviceStart.restype = ctypes.c_int
            if hasattr(self._mt, "MTDeviceStop"):
                self._mt.MTDeviceStop.argtypes = [ctypes.c_void_p]
                self._mt.MTDeviceStop.restype = ctypes.c_int

            device_array = self._mt.MTDeviceCreateList()
            if not device_array:
                self.error = "no multitouch device list"
                return False

            device_count = int(self._cf.CFArrayGetCount(device_array))
            if device_count <= 0:
                self.error = "no multitouch devices"
                return False

            for index in range(device_count):
                device = self._cf.CFArrayGetValueAtIndex(device_array, index)
                if not device:
                    continue
                self._mt.MTRegisterContactFrameCallback(device, self._callback)
                self._mt.MTDeviceStart(device, 0)
                self._devices.append(int(device))

            self.available = bool(self._devices)
            if not self.available:
                self.error = "could not start multitouch devices"
            return self.available
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            return False

    def stop(self) -> None:
        if self._mt is None or not hasattr(self._mt, "MTDeviceStop"):
            return
        for device in self._devices:
            try:
                self._mt.MTDeviceStop(ctypes.c_void_p(device))
            except Exception:
                pass

    def read(self) -> TrackpadSignal:
        with self._lock:
            value = self._value
            fingers = self._finger_count
            events = self._events
            last_update = self._last_update
            error = self.error
        now = time.monotonic()
        age = now - last_update if last_update else float("inf")
        active = (
            self.available
            and fingers > 0
            and value > TOUCH_CONTACT_MIN
            and age <= TOUCH_CONTACT_MAX_AGE_SECONDS
        )
        return TrackpadSignal(value, active, "touch", fingers, events, age, error)

    def _handle_frame(
        self,
        _device,
        contacts,
        contact_count,
        _timestamp,
        _frame,
    ) -> None:
        try:
            count = max(0, min(int(contact_count), MAX_TOUCH_CONTACTS))
            total = 0.0
            active_count = 0

            for index in range(count):
                contact = contacts[index]
                size = self._sane_positive(contact.size, limit=4.0)
                major = self._sane_positive(contact.major_axis, limit=4.0)
                minor = self._sane_positive(contact.minor_axis, limit=4.0)

                area = major * minor if major > 0.0 and minor > 0.0 else 0.0
                if size <= 0.0 and area <= 0.0:
                    continue

                if size > 0.0 and area > 0.0:
                    contact_value = (0.75 * size) + (0.25 * area)
                else:
                    contact_value = size if size > 0.0 else area

                total += contact_value
                active_count += 1

            with self._lock:
                self._value = float(total)
                self._finger_count = active_count
                self._events += 1
                self._last_update = time.monotonic()
        except Exception as exc:
            with self._lock:
                self.error = str(exc)

    @staticmethod
    def _sane_positive(value: float, limit: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0 or value > limit:
            return 0.0
        return value


class HeartRateEstimator:
    """Turns finite trackpad signal batches into heart-rate estimates."""

    def __init__(self, sample_rate_hz: float = TARGET_SAMPLE_RATE_HZ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.samples: deque[tuple[float, float, bool]] = deque(
            maxlen=int((MEASUREMENT_SECONDS + 3.0) * self.sample_rate_hz)
        )
        self._capture_started_at: float | None = None
        self._last_active_at: float | None = None
        self._result_hold_until: float | None = None
        self._last_completed_result = HeartRateResult(
            False, None, 0.0, "waiting for trackpad signal", 0.0
        )
        self._last_result = self._last_completed_result

    def add_sample(self, timestamp: float, value: float, active: bool) -> None:
        timestamp = float(timestamp)
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        active = bool(active)

        if (
            self._result_hold_until is not None
            and timestamp < self._result_hold_until
        ):
            return
        if (
            self._result_hold_until is not None
            and timestamp >= self._result_hold_until
        ):
            self._result_hold_until = None

        if active:
            if self._capture_started_at is None:
                self.samples.clear()
                self._capture_started_at = timestamp
            self._last_active_at = timestamp
            self.samples.append((timestamp, max(0.0, value), True))
            return

        if self._capture_started_at is None:
            self.samples.clear()
            return

        if (
            self._last_active_at is not None
            and timestamp - self._last_active_at > CONTACT_DROPOUT_RESET_SECONDS
        ):
            self.samples.clear()
            self._capture_started_at = None
            self._last_active_at = None
            return

        self.samples.append((timestamp, 0.0, False))

    def estimate(self, now: float | None = None) -> HeartRateResult:
        now = time.monotonic() if now is None else float(now)
        if self._capture_started_at is None:
            if (
                self._result_hold_until is not None
                and now < self._result_hold_until
                and self._last_completed_result.bpm is not None
            ):
                return self._remember(
                    HeartRateResult(
                        True,
                        self._last_completed_result.bpm,
                        self._last_completed_result.confidence,
                        "measurement complete",
                        self._last_completed_result.active_seconds,
                        self._last_completed_result.peak_ratio,
                        self._last_completed_result.autocorr_peak,
                    )
                )
            if self._last_completed_result.bpm is not None:
                return self._remember(
                    HeartRateResult(
                        True,
                        self._last_completed_result.bpm,
                        self._last_completed_result.confidence,
                        "waiting for a new steady 15s touch",
                        0.0,
                        self._last_completed_result.peak_ratio,
                        self._last_completed_result.autocorr_peak,
                    )
                )
            return self._remember(
                HeartRateResult(
                    False, None, 0.0, "waiting for trackpad signal", 0.0
                )
            )

        elapsed = max(0.0, now - self._capture_started_at)
        if elapsed < MEASUREMENT_SECONDS:
            return self._remember(self._progress_result(elapsed))

        result = self._finish_measurement()
        return self._remember(result)

    def _progress_result(self, elapsed: float) -> HeartRateResult:
        active_seconds = self._active_seconds_in_current_capture()
        remaining = max(0, int(math.ceil(MEASUREMENT_SECONDS - elapsed)))
        reason = f"hold still for {remaining}"
        return HeartRateResult(
            False,
            None,
            0.0,
            reason,
            min(MEASUREMENT_SECONDS, active_seconds),
            remaining_seconds=remaining,
        )

    def _finish_measurement(self) -> HeartRateResult:
        assert self._capture_started_at is not None
        start = self._capture_started_at
        end = start + MEASUREMENT_SECONDS
        rows = [row for row in self.samples if start <= row[0] <= end]
        prior_bpm = self._last_completed_result.bpm
        prior_confidence = self._last_completed_result.confidence
        result = self._estimate_rows(rows, prior_bpm, prior_confidence)
        if result.bpm is not None:
            self._last_completed_result = result
        self._result_hold_until = (
            end + RESULT_HOLD_SECONDS if result.bpm is not None else None
        )

        self.samples.clear()
        self._capture_started_at = None
        self._last_active_at = None
        return result

    def _estimate_rows(
        self,
        rows: list[tuple[float, float, bool]],
        prior_bpm: float | None = None,
        prior_confidence: float = 0.0,
    ) -> HeartRateResult:
        fs = self.sample_rate_hz
        min_samples = int(MIN_SECONDS * fs * 0.72)
        if len(rows) < min_samples:
            return HeartRateResult(
                False, None, 0.0, "measurement rejected; not enough samples", 0.0
            )

        times = np.array([r[0] for r in rows], dtype=np.float64)
        values = np.array([r[1] for r in rows], dtype=np.float64)
        active = np.array([r[2] for r in rows], dtype=bool)
        duration = float(times[-1] - times[0])
        if duration < MIN_SECONDS:
            return HeartRateResult(
                False,
                None,
                0.0,
                "measurement rejected; hold contact for the full 15s",
                max(0.0, duration),
            )

        active_fraction = float(np.mean(active))
        active_seconds = duration * active_fraction
        if active_fraction < MIN_ACTIVE_FRACTION:
            return HeartRateResult(
                False,
                None,
                0.0,
                "measurement rejected; keep fingers steady for the full 15s",
                active_seconds,
            )

        active_times = times[active]
        active_values = values[active]
        if active_values.size < min_samples or float(np.ptp(active_values)) < 1e-7:
            return HeartRateResult(
                False,
                None,
                0.0,
                "measurement rejected; signal is flat",
                active_seconds,
            )

        grid = np.arange(times[0], times[-1], 1.0 / fs, dtype=np.float64)
        if grid.size < int(MIN_SECONDS * fs):
            return HeartRateResult(
                False,
                None,
                0.0,
                "measurement rejected; not enough even samples",
                active_seconds,
            )
        x = np.interp(grid, active_times, active_values)

        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        if mad > 1e-10:
            x = np.clip(x, median - 6.0 * mad, median + 6.0 * mad)
        x = signal.detrend(x, type="linear")

        try:
            band_sos = signal.butter(
                3,
                (MIN_BPM / 60.0, MAX_BPM / 60.0),
                btype="bandpass",
                fs=fs,
                output="sos",
            )
            y = signal.sosfiltfilt(band_sos, x)
        except ValueError:
            return HeartRateResult(
                False,
                None,
                0.0,
                "measurement rejected; samples are noisy",
                active_seconds,
            )

        y = np.asarray(y, dtype=np.float64)
        y -= float(np.mean(y))
        pulse_std = float(np.std(y))
        if pulse_std < 1e-8:
            return HeartRateResult(
                False,
                None,
                0.0,
                "measurement rejected; pulse pattern is tiny",
                active_seconds,
            )
        y /= pulse_std

        candidate = self._select_bpm_candidate(y, fs, prior_bpm, prior_confidence)
        if candidate is None:
            return HeartRateResult(
                False,
                None,
                0.0,
                "measurement rejected; no reliable pulse pattern",
                active_seconds,
            )

        confidence = max(0.0, min(1.0, candidate.score))
        if candidate.peak_ratio < 2.4 and candidate.autocorr_peak < 0.08:
            return HeartRateResult(
                False,
                None,
                confidence * 0.5,
                "measurement rejected; repeating pattern is weak",
                active_seconds,
                candidate.peak_ratio,
                candidate.autocorr_peak,
            )
        if confidence < 0.28:
            return HeartRateResult(
                False,
                None,
                confidence,
                "measurement rejected; pulse candidate is unstable",
                active_seconds,
                candidate.peak_ratio,
                candidate.autocorr_peak,
            )

        if candidate.bpm < 62.0 and confidence < 0.55:
            reason = "low-BPM estimate from fixed 15s window"
        elif confidence < 0.45:
            reason = "weak estimate from fixed 15s window"
        else:
            reason = "heart-rate estimate from fixed 15s window"
        return HeartRateResult(
            True,
            candidate.bpm,
            confidence,
            reason,
            active_seconds,
            candidate.peak_ratio,
            candidate.autocorr_peak,
        )

    def waveform(self, seconds: float = 5.0, points: int = 240) -> np.ndarray:
        if not self.samples:
            return np.zeros(points, dtype=np.float64)
        now = self.samples[-1][0]
        rows = [r for r in self.samples if r[0] >= now - seconds and r[2]]
        if len(rows) < 4:
            return np.zeros(points, dtype=np.float64)
        times = np.array([r[0] for r in rows], dtype=np.float64)
        x = np.array([r[1] for r in rows], dtype=np.float64)
        grid = np.linspace(times[0], times[-1], points)
        y = np.interp(grid, times, x)
        y = signal.detrend(y, type="linear")
        scale = float(np.percentile(np.abs(y), 95))
        if scale <= 1e-10:
            return np.zeros(points, dtype=np.float64)
        return np.clip(y / scale, -1.0, 1.0)

    def _remember(self, result: HeartRateResult) -> HeartRateResult:
        self._last_result = result
        return result

    def _active_seconds_in_current_capture(self) -> float:
        if self._capture_started_at is None or len(self.samples) < 2:
            return 0.0
        rows = list(self.samples)
        duration = max(0.0, float(rows[-1][0] - rows[0][0]))
        if duration <= 0.0:
            return 0.0
        active_fraction = sum(1 for row in rows if row[2]) / float(len(rows))
        return duration * active_fraction

    @staticmethod
    def _periodogram(y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        nfft = 1 << int(math.ceil(math.log2(max(16, y.size * 8))))
        window = np.hanning(y.size)
        spectrum = np.abs(np.fft.rfft(y * window, n=nfft)) ** 2
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        return freqs, spectrum

    @staticmethod
    def _refined_frequency(
        power: np.ndarray, freqs: np.ndarray, peak_index: int
    ) -> float:
        peak_freq = float(freqs[peak_index])
        if 0 < peak_index < power.size - 1:
            left = float(power[peak_index - 1])
            center = float(power[peak_index])
            right = float(power[peak_index + 1])
            denom = left - (2.0 * center) + right
            if abs(denom) > 1e-18:
                step = float(freqs[1] - freqs[0])
                peak_freq += 0.5 * (left - right) / denom * step
        return peak_freq

    def _select_bpm_candidate(
        self,
        y: np.ndarray,
        fs: float,
        prior_bpm: float | None = None,
        prior_confidence: float = 0.0,
    ) -> BpmCandidate | None:
        freqs, spectrum = self._periodogram(y, fs)
        mask = (freqs >= MIN_BPM / 60.0) & (freqs <= MAX_BPM / 60.0)
        if not np.any(mask):
            return None

        band_power = spectrum[mask]
        band_freqs = freqs[mask]
        if band_power.size < 3 or float(np.max(band_power)) <= 0.0:
            return None

        floor = float(np.median(band_power)) + 1e-18
        max_power = float(np.max(band_power))
        distance = max(
            1,
            int(round((5.0 / 60.0) / float(band_freqs[1] - band_freqs[0]))),
        )
        prominence = max(max_power * 0.025, floor * 1.5)
        peak_indices, _props = signal.find_peaks(
            band_power,
            distance=distance,
            prominence=prominence,
        )
        if peak_indices.size == 0:
            peak_indices = np.array([int(np.argmax(band_power))], dtype=int)

        top_by_power = np.argsort(band_power)[-8:]
        candidate_indices = sorted(
            {int(i) for i in peak_indices} | {int(i) for i in top_by_power}
        )
        candidates: list[BpmCandidate] = []
        band_sum = float(np.sum(band_power)) + 1e-18
        for index in candidate_indices:
            if index <= 0 or index >= band_power.size - 1:
                continue
            peak_freq = self._refined_frequency(band_power, band_freqs, index)
            bpm = max(MIN_BPM, min(MAX_BPM, peak_freq * 60.0))
            peak = float(band_power[index])
            peak_ratio = peak / floor
            dominance = peak / band_sum
            candidate_waveform = self._narrowband_for_bpm(y, fs, bpm)
            ac_bpm, ac_peak = self._autocorr_near_bpm(candidate_waveform, fs, bpm)
            if ac_bpm is not None and abs(ac_bpm - bpm) <= max(8.0, bpm * 0.08):
                bpm = (0.82 * bpm) + (0.18 * ac_bpm)

            power_norm = min(1.0, math.sqrt(max(0.0, peak / max_power)))
            ratio_score = min(
                1.0,
                max(0.0, math.log(max(peak_ratio, 1.0)) / math.log(18.0)),
            )
            dominance_score = min(1.0, max(0.0, dominance / 0.22))
            ac_score = min(1.0, max(0.0, ac_peak / 0.34))
            stability = self._segment_stability(candidate_waveform, fs, bpm)
            score = (
                0.30 * power_norm
                + 0.22 * ratio_score
                + 0.16 * dominance_score
                + 0.22 * ac_score
                + 0.10 * stability
            )
            if bpm < 58.0:
                score *= 0.70
            elif bpm < 64.0:
                score *= 0.82
            candidates.append(
                BpmCandidate(
                    bpm=bpm,
                    peak_ratio=peak_ratio,
                    dominance=dominance,
                    autocorr_peak=ac_peak,
                    stability=stability,
                    score=score,
                )
            )

        if not candidates:
            return None

        best = max(candidates, key=lambda item: item.score)
        if best.bpm < 78.0:
            higher_floor = 95.0 if best.bpm >= 65.0 else 70.0
            higher = [
                item
                for item in candidates
                if higher_floor <= item.bpm <= MAX_BPM
                and item.score >= best.score * 0.50
                and item.peak_ratio >= best.peak_ratio * 0.03
                and item.stability >= 0.45
            ]
            if higher:
                best = max(higher, key=lambda item: (item.score, item.peak_ratio))

        if (
            prior_bpm is not None
            and prior_confidence >= 0.45
            and math.isfinite(float(prior_bpm))
        ):
            prior_bpm = float(prior_bpm)
            prior_near = [
                item
                for item in candidates
                if abs(item.bpm - prior_bpm) <= max(12.0, prior_bpm * 0.12)
            ]
            if abs(best.bpm - prior_bpm) >= 22.0 and prior_near:
                prior_best = max(prior_near, key=lambda item: item.score)
                if (
                    prior_best.score >= best.score * 0.42
                    and prior_best.peak_ratio >= best.peak_ratio * 0.05
                    and prior_best.stability >= 0.35
                ):
                    best = prior_best

            if prior_bpm >= 85.0 and best.bpm <= 78.0:
                return None
        return best

    @staticmethod
    def _narrowband_for_bpm(y: np.ndarray, fs: float, bpm: float) -> np.ndarray:
        center_hz = max(MIN_BPM / 60.0, min(MAX_BPM / 60.0, float(bpm) / 60.0))
        width_hz = 18.0 / 60.0
        low_hz = max(MIN_BPM / 60.0, center_hz - width_hz)
        high_hz = min(MAX_BPM / 60.0, center_hz + width_hz)
        if high_hz <= low_hz:
            return y
        try:
            sos = signal.butter(
                2,
                (low_hz, high_hz),
                btype="bandpass",
                fs=fs,
                output="sos",
            )
            narrowed = signal.sosfiltfilt(sos, y)
        except ValueError:
            return y
        narrowed = np.asarray(narrowed, dtype=np.float64)
        narrowed -= float(np.mean(narrowed))
        scale = float(np.std(narrowed))
        if scale <= 1e-10:
            return y
        return narrowed / scale

    @staticmethod
    def _normalized_autocorr(y: np.ndarray) -> np.ndarray | None:
        corr = signal.correlate(y, y, mode="full", method="auto")
        corr = corr[corr.size // 2 :]
        if corr.size < 3 or abs(float(corr[0])) <= 1e-18:
            return None
        corr = corr / float(corr[0])
        return corr

    @classmethod
    def _autocorr_near_bpm(
        cls, y: np.ndarray, fs: float, bpm: float
    ) -> tuple[float | None, float]:
        corr = cls._normalized_autocorr(y)
        if corr is None:
            return None, 0.0
        min_lag = max(1, int(round(fs * 60.0 / MAX_BPM)))
        max_lag = min(corr.size - 1, int(round(fs * 60.0 / MIN_BPM)))
        if max_lag <= min_lag:
            return None, 0.0
        target_lag = fs * 60.0 / max(MIN_BPM, min(MAX_BPM, float(bpm)))
        center = int(round(target_lag))
        span = max(2, int(round(target_lag * 0.08)))
        low = max(min_lag, center - span)
        high = min(max_lag, center + span)
        if high <= low:
            return None, 0.0
        segment = corr[low : high + 1]
        lag = low + int(np.argmax(segment))
        peak = float(corr[lag])
        return 60.0 * fs / float(lag), peak

    @staticmethod
    def _segment_stability(y: np.ndarray, fs: float, bpm: float) -> float:
        midpoint = y.size // 2
        first = y[:midpoint]
        second = y[midpoint:]
        if first.size < int(3.0 * fs) or second.size < int(3.0 * fs):
            return 0.5
        freq = bpm / 60.0
        p1 = HeartRateEstimator._single_frequency_power(first, fs, freq)
        p2 = HeartRateEstimator._single_frequency_power(second, fs, freq)
        larger = max(p1, p2)
        if larger <= 1e-18:
            return 0.0
        return max(0.0, min(1.0, math.sqrt(min(p1, p2) / larger)))

    @staticmethod
    def _single_frequency_power(segment: np.ndarray, fs: float, freq: float) -> float:
        centered = np.asarray(segment, dtype=np.float64)
        centered = centered - float(np.mean(centered))
        if centered.size < 3:
            return 0.0
        window = np.hanning(centered.size)
        times = np.arange(centered.size, dtype=np.float64) / fs
        basis = np.exp(-2j * math.pi * freq * times)
        return float(abs(np.sum(centered * window * basis)) ** 2)


def run_simulation() -> int:
    """Quick detector check without opening the Mac trackpad GUI."""

    fs = TARGET_SAMPLE_RATE_HZ
    bpm_true = 72.0
    estimator = HeartRateEstimator(fs)
    start = time.monotonic()
    rng = np.random.default_rng(42)
    result = None
    last_detected = None
    for index in range(int(28.0 * fs)):
        t = start + index / fs
        seconds = index / fs
        baseline = 0.35 + 0.05 * math.sin(2 * math.pi * 0.08 * seconds)
        pulse = 0.0025 * math.sin(2 * math.pi * (bpm_true / 60.0) * seconds)
        artifact = 0.0
        if 7.0 < seconds < 7.5:
            artifact = 0.08 * math.sin(2 * math.pi * 2.0 * seconds)
        noise = float(rng.normal(0.0, 0.0012))
        estimator.add_sample(t, baseline + pulse + artifact + noise, True)
        if index % int(fs) == 0:
            result = estimator.estimate(t)
            if result.detected and result.bpm is not None:
                last_detected = result

    result = last_detected if last_detected is not None else result
    if result is None:
        print("no result")
        return 1
    if result.detected:
        print(
            f"simulation detected {result.bpm:.0f} bpm "
            f"(true {bpm_true:.0f}, confidence {result.confidence:.2f})"
        )
        return 0
    print(f"simulation not detected: {result.reason}")
    return 1


def run_touch_probe(seconds: float) -> int:
    sampler = TouchContactSampler()
    if not sampler.start():
        print(f"touch contact reader failed: {sampler.error}")
        return 2

    print("Touch probe running. Rest fingers on the trackpad without clicking.")
    started = time.monotonic()
    next_print = started
    try:
        while time.monotonic() - started < seconds:
            now = time.monotonic()
            if now >= next_print:
                sample = sampler.read()
                value = sample.value if math.isfinite(sample.value) else 0.0
                age = sample.age if math.isfinite(sample.age) else -1.0
                print(
                    f"touch active={sample.active} fingers={sample.fingers} "
                    f"value={value:.6f} events={sample.events} age={age:.2f}s"
                )
                next_print = now + 0.5
            time.sleep(0.02)
    finally:
        sampler.stop()
    return 0


def run_gui(use_touch: bool = True) -> int:
    try:
        import objc
        from AppKit import (
            NSApp,
            NSApplication,
            NSApplicationActivationPolicyRegular,
            NSBackingStoreBuffered,
            NSBezierPath,
            NSColor,
            NSEvent,
            NSEventMaskLeftMouseDown,
            NSEventMaskLeftMouseDragged,
            NSEventMaskLeftMouseUp,
            NSEventMaskPressure,
            NSEventTypeLeftMouseUp,
            NSFont,
            NSFontAttributeName,
            NSForegroundColorAttributeName,
            NSMakeRect,
            NSMakeSize,
            NSRunLoopCommonModes,
            NSWindow,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskMiniaturizable,
            NSWindowStyleMaskResizable,
            NSWindowStyleMaskTitled,
        )
        from Foundation import NSString, NSTimer, NSRunLoop
        from PyObjCTools import AppHelper
    except Exception as exc:
        print(f"Could not import macOS AppKit/PyObjC: {exc}", file=sys.stderr)
        print("Install with: python3 -m pip install pyobjc", file=sys.stderr)
        return 2

    class TrackpadPulseView(objc.lookUpClass("NSView")):
        def initWithFrame_(self, frame):
            self = objc.super(TrackpadPulseView, self).initWithFrame_(frame)
            if self is None:
                return None

            self.estimator = HeartRateEstimator(TARGET_SAMPLE_RATE_HZ)
            self.touch_sampler = TouchContactSampler() if use_touch else None
            self.touch_enabled = (
                self.touch_sampler.start() if self.touch_sampler is not None else False
            )

            # Raw GUI sampling state.
            self.last_pressure = 0.0
            self.last_signal = 0.0
            self.last_source = "touch" if self.touch_enabled else "pressure"
            self.last_finger_count = 0
            self.last_touch_error = (
                None if self.touch_sampler is None else self.touch_sampler.error
            )
            self.contact_until = 0.0
            self.event_count = 0
            self.pressure_event_count = 0
            self.result = HeartRateResult(
                False, None, 0.0, "waiting for trackpad signal", 0.0
            )
            self.last_terminal_print = 0.0

            # UI-only animation state. Does not affect estimator input, BPM, or graph.
            self.current_phase = 0.0

            self._monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskPressure
                | NSEventMaskLeftMouseDown
                | NSEventMaskLeftMouseDragged
                | NSEventMaskLeftMouseUp,
                self._handleLocalEvent,
            )
            self._timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0 / TARGET_SAMPLE_RATE_HZ, self, "tick:", None, True
            )
            NSRunLoop.currentRunLoop().addTimer_forMode_(
                self._timer, NSRunLoopCommonModes
            )
            try:
                self.setAcceptsTouchEvents_(True)
            except Exception:
                pass
            return self

        def acceptsFirstResponder(self):
            return True

        def mouseDown_(self, event):
            self._recordEvent(event)

        def mouseDragged_(self, event):
            self._recordEvent(event)

        def pressureChangeWithEvent_(self, event):
            self._recordEvent(event)

        def mouseUp_(self, event):
            self._recordEvent(event)
            self.contact_until = time.monotonic() + 0.10

        @objc.python_method
        def _handleLocalEvent(self, event):
            self._recordEvent(event)
            return event

        @objc.python_method
        def _recordEvent(self, event):
            try:
                pressure = float(event.pressure())
            except Exception:
                pressure = 0.0
            if not math.isfinite(pressure):
                pressure = 0.0

            now = time.monotonic()
            try:
                event_type = int(event.type())
            except Exception:
                event_type = 0

            if event_type == int(NSEventTypeLeftMouseUp):
                self.last_pressure = max(0.0, pressure)
                self.contact_until = now + 0.10
                return

            if pressure > CONTACT_PRESSURE_MIN:
                self.last_pressure = pressure
                self.contact_until = now + CONTACT_LATCH_SECONDS
                self.pressure_event_count += 1

        def tick_(self, _timer):
            now = time.monotonic()

            touch_signal = (
                self.touch_sampler.read() if self.touch_sampler is not None else None
            )
            if touch_signal is not None:
                self.last_touch_error = touch_signal.error

            pressure_active = (
                now <= self.contact_until and self.last_pressure > CONTACT_PRESSURE_MIN
            )
            if touch_signal is not None and touch_signal.active:
                sample_value = touch_signal.value
                active = True
                self.last_source = "touch"
                self.last_finger_count = touch_signal.fingers
                self.event_count = touch_signal.events
            elif pressure_active:
                sample_value = self.last_pressure
                active = True
                self.last_source = "pressure"
                self.last_finger_count = 0
                self.event_count = self.pressure_event_count
            else:
                sample_value = 0.0
                active = False
                self.last_finger_count = 0
                if self.touch_enabled:
                    self.last_source = "touch"
                    self.event_count = touch_signal.events if touch_signal else 0
                else:
                    self.last_source = "pressure"
                    self.event_count = self.pressure_event_count

            self.last_signal = sample_value
            self.estimator.add_sample(now, sample_value, active)
            self.result = self.estimator.estimate(now)

            current_bpm = self.result.bpm if self.result.bpm is not None else DEFAULT_BPM
            self.current_phase = (
                self.current_phase + (float(current_bpm) / 60.0) / TARGET_SAMPLE_RATE_HZ
            ) % 1.0

            if now - self.last_terminal_print >= 1.0:
                self.last_terminal_print = now
                if self.result.remaining_seconds is not None:
                    print(f"hold still for {self.result.remaining_seconds}")
                elif self.result.bpm is not None:
                    print(f"estimated heart rate {self.result.bpm:5.1f} bpm")
                else:
                    print(f"heart rate not detected: {self.result.reason}")
            self.setNeedsDisplay_(True)

        def drawRect_(self, rect):
            bounds = self.bounds()
            width = max(1.0, float(bounds.size.width))
            height = max(1.0, float(bounds.size.height))

            bg = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.965, 0.967, 0.968, 1.0
            )
            bg.setFill()
            NSBezierPath.fillRect_(bounds)

            text_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.030, 0.034, 0.038, 1.0
            )
            muted_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.34, 0.37, 0.39, 1.0
            )
            soft_muted_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.48, 0.51, 0.53, 1.0
            )
            panel_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                1.0, 1.0, 1.0, 1.0
            )
            trace_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.86, 0.025, 0.075, 1.0
            )
            trace_soft_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.80, 0.12, 0.16, 0.42
            )
            grid_minor_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.94, 0.945, 0.948, 1.0
            )
            grid_major_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.875, 0.89, 0.895, 1.0
            )
            red_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.93, 0.02, 0.09, 1.0
            )
            yellow_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.98, 0.71, 0.02, 1.0
            )
            green_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.05, 0.67, 0.28, 1.0
            )
            border_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.82, 0.835, 0.845, 1.0
            )

            bpm = self.result.bpm
            bpm_ready = bpm is not None
            status_text = self._statusText(bpm_ready)
            content_width = min(width - 72.0, 860.0)
            content_x = (width - content_width) / 2.0
            top_y = height - 58.0

            self._drawText(
                "Heart Beat Detector",
                content_x,
                top_y,
                content_width,
                34.0,
                25,
                text_color,
                weight=0.32,
            )
            self._drawText(
                "Proof of Concept",
                content_x,
                top_y - 30.0,
                content_width,
                22.0,
                12,
                muted_color,
            )
            self._drawText(
                "Light touch the trackpad. Hold still for 15 seconds.",
                content_x,
                top_y - 52.0,
                content_width,
                22.0,
                12,
                muted_color,
            )
            self._drawText(
                status_text,
                content_x,
                top_y - 78.0,
                content_width,
                22.0,
                12,
                self._statusColor(bpm_ready, yellow_color, green_color, muted_color),
                weight=0.28,
            )

            remaining = self.result.remaining_seconds
            if remaining is not None:
                bpm_text = f"{remaining}"
                metric_label = "SEC"
            else:
                bpm_text = f"{bpm:.0f}" if bpm is not None else "--"
                metric_label = "BPM"
            metric_y = max(292.0, top_y - 168.0)

            self._drawText(
                bpm_text,
                content_x,
                metric_y - 4.0,
                150.0,
                64.0,
                48,
                text_color,
                monospace=True,
                weight=0.42,
            )
            self._drawText(
                metric_label,
                content_x + 150.0,
                metric_y + 10.0,
                70.0,
                26.0,
                16,
                muted_color,
                weight=0.20,
            )

            pulse = self._pulseEnvelope(self.current_phase) if bpm is not None else 0.0
            heart_size = 48.0 + (7.0 * pulse)
            self._drawHeartGlyph(
                content_x + content_width - 70.0,
                metric_y + 31.0,
                heart_size,
                red_color,
            )

            panel_width = content_width
            panel_y = 140.0
            panel_height = max(
                150.0,
                min(220.0, metric_y - panel_y - 48.0),
            )
            panel_x = content_x
            graph_rect = NSMakeRect(panel_x, panel_y, panel_width, panel_height)
            self._fillRoundedRect(graph_rect, 8.0, panel_color)
            self._drawMonitorGrid(graph_rect, grid_minor_color, grid_major_color)
            self._strokeRoundedRect(graph_rect, 8.0, border_color, 1.0)

            waveform = self.estimator.waveform(points=220)
            if bool(np.max(np.abs(waveform)) > 1e-4):
                self._drawPulseWaveform(graph_rect, waveform, trace_color, 2.0)
            else:
                self._drawWaitingTrace(graph_rect, trace_soft_color)
                self._drawText(
                    self._waitText(),
                    panel_x + 24.0,
                    panel_y + (panel_height / 2.0) - 13.0,
                    panel_width - 48.0,
                    26.0,
                    14,
                    soft_muted_color,
                    align="center",
                )

            scale_y = max(78.0, min(112.0, panel_y - 86.0))
            self._drawRangeScale(
                width,
                scale_y,
                bpm,
                red_color,
                yellow_color,
                green_color,
                soft_muted_color,
                text_color,
            )

            self._drawText(
                "Experimental only, not for medical decisions.",
                24,
                12,
                width - 48,
                20,
                12,
                muted_color,
                align="center",
            )

        @objc.python_method
        def _statusColor(self, bpm_ready, hold_color, green_color, muted_color):
            if bpm_ready and self.result.confidence > 0.0:
                return green_color
            if self.result.active_seconds > 0.1 or self.last_signal > TOUCH_CONTACT_MIN:
                return hold_color
            return muted_color

        @objc.python_method
        def _statusText(self, bpm_ready):
            active = max(0.0, float(self.result.active_seconds))
            if self.last_signal <= TOUCH_CONTACT_MIN and active < 0.1:
                return "Touch the trackpad"
            if self.result.remaining_seconds is not None:
                return f"Hold still for {self.result.remaining_seconds}"
            if not bpm_ready and active < MEASUREMENT_SECONDS:
                remaining = int(math.ceil(MEASUREMENT_SECONDS - active))
                return f"Hold still for {remaining}"
            if not bpm_ready:
                return "Try again and keep your finger still"
            return "Done"

        @objc.python_method
        def _waitText(self):
            active = max(0.0, float(self.result.active_seconds))
            if self.last_signal <= TOUCH_CONTACT_MIN and active < 0.1:
                return "Touch the trackpad"
            if self.result.remaining_seconds is not None:
                return f"Hold still for {self.result.remaining_seconds}"
            if active < MEASUREMENT_SECONDS:
                return f"Hold still {int(math.ceil(MEASUREMENT_SECONDS - active))}s"
            return "Keep holding still"

        @objc.python_method
        def _fillRoundedRect(self, rect, radius, color):
            color.setFill()
            try:
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, radius, radius
                )
            except Exception:
                path = NSBezierPath.bezierPathWithRect_(rect)
            path.fill()

        @objc.python_method
        def _strokeRoundedRect(self, rect, radius, color, line_width):
            color.setStroke()
            try:
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, radius, radius
                )
            except Exception:
                path = NSBezierPath.bezierPathWithRect_(rect)
            path.setLineWidth_(float(line_width))
            path.stroke()

        @objc.python_method
        def _pulseEnvelope(self, phase):
            phase = float(phase)
            if phase < 0.10:
                return 1.0 - (phase / 0.10)
            if 0.30 <= phase < 0.42:
                return 0.35 * (1.0 - ((phase - 0.30) / 0.12))
            return 0.0

        @objc.python_method
        def _drawMonitorGrid(self, rect, minor_color, major_color):
            x0 = float(rect.origin.x)
            y0 = float(rect.origin.y)
            w = float(rect.size.width)
            h = float(rect.size.height)
            step = 18.0

            minor = NSBezierPath.bezierPath()
            index = 0
            x = x0 + step
            while x < x0 + w:
                if index % 5 != 4:
                    minor.moveToPoint_((x, y0))
                    minor.lineToPoint_((x, y0 + h))
                x += step
                index += 1
            index = 0
            y = y0 + step
            while y < y0 + h:
                if index % 5 != 4:
                    minor.moveToPoint_((x0, y))
                    minor.lineToPoint_((x0 + w, y))
                y += step
                index += 1
            minor_color.setStroke()
            minor.setLineWidth_(0.6)
            minor.stroke()

            major = NSBezierPath.bezierPath()
            x = x0 + step * 5.0
            while x < x0 + w:
                major.moveToPoint_((x, y0))
                major.lineToPoint_((x, y0 + h))
                x += step * 5.0
            y = y0 + step * 5.0
            while y < y0 + h:
                major.moveToPoint_((x0, y))
                major.lineToPoint_((x0 + w, y))
                y += step * 5.0
            major_color.setStroke()
            major.setLineWidth_(0.9)
            major.stroke()

        @objc.python_method
        def _drawPulseWaveform(self, graph_rect, waveform, color, line_width):
            baseline_y = float(graph_rect.origin.y + graph_rect.size.height * 0.50)
            amp = float(graph_rect.size.height * 0.42)
            left = float(graph_rect.origin.x + 18.0)
            graph_width = float(graph_rect.size.width - 36.0)
            path = NSBezierPath.bezierPath()
            for i, value in enumerate(waveform):
                x_pos = left + graph_width * (i / max(1, waveform.size - 1))
                y_pos = baseline_y + amp * max(-1.0, min(1.0, float(value)))
                if i == 0:
                    path.moveToPoint_((x_pos, y_pos))
                else:
                    path.lineToPoint_((x_pos, y_pos))
            color.setStroke()
            path.setLineWidth_(float(line_width))
            path.stroke()

        @objc.python_method
        def _drawWaitingTrace(self, graph_rect, color):
            mid_y = float(graph_rect.origin.y + graph_rect.size.height / 2.0)
            left = float(graph_rect.origin.x + 24.0)
            right = float(graph_rect.origin.x + graph_rect.size.width - 24.0)
            path = NSBezierPath.bezierPath()
            path.moveToPoint_((left, mid_y))
            path.lineToPoint_((right, mid_y))
            color.setStroke()
            path.setLineWidth_(1.7)
            path.stroke()

        @objc.python_method
        def _drawHeartGlyph(self, center_x, center_y, size, color):
            size = float(size)
            self._drawText(
                "\u2665",
                float(center_x) - size / 2.0,
                float(center_y) - size * 0.54,
                size,
                size,
                size,
                color,
                weight=0.34,
                align="center",
            )

        @objc.python_method
        def _drawRangeScale(
            self,
            width,
            y,
            bpm,
            red_color,
            yellow_color,
            green_color,
            pointer_color,
            text_color,
        ):
            bar_width = min(width - 90.0, 760.0)
            bar_x = (width - bar_width) / 2.0
            gap = 7.0
            segment_width = (bar_width - (gap * 4.0)) / 5.0
            labels = ["40-", "40-60", "60-80", "80-100", "100+"]
            colors = [red_color, yellow_color, green_color, yellow_color, red_color]

            for index, label in enumerate(labels):
                x = bar_x + index * (segment_width + gap)
                rect = NSMakeRect(x, y, segment_width, 8.5)
                self._fillRoundedRect(rect, 4.25, colors[index])
                self._drawText(
                    label,
                    x,
                    y - 43.0,
                    segment_width,
                    30.0,
                    18,
                    text_color,
                    weight=0.18,
                    align="center",
                )

            if bpm is None:
                return

            bpm = max(30.0, min(130.0, float(bpm)))
            spans = [
                (30.0, 40.0),
                (40.0, 60.0),
                (60.0, 80.0),
                (80.0, 100.0),
                (100.0, 130.0),
            ]
            segment_index = 4
            for index, (low, high) in enumerate(spans):
                if bpm <= high:
                    segment_index = index
                    break
            low, high = spans[segment_index]
            frac = 0.0 if high <= low else (bpm - low) / (high - low)
            pointer_x = (
                bar_x
                + segment_index * (segment_width + gap)
                + max(0.0, min(1.0, frac)) * segment_width
            )
            pointer = NSBezierPath.bezierPath()
            pointer.moveToPoint_((pointer_x, y + 18.0))
            pointer.lineToPoint_((pointer_x - 12.0, y + 43.0))
            pointer.lineToPoint_((pointer_x + 12.0, y + 43.0))
            pointer.closePath()
            pointer_color.setFill()
            pointer.fill()

        @objc.python_method
        def _drawText(
            self,
            text,
            x,
            y,
            width,
            height,
            size,
            color,
            monospace=False,
            weight=0.0,
            align="left",
        ):
            if monospace and hasattr(NSFont, "monospacedDigitSystemFontOfSize_weight_"):
                font = NSFont.monospacedDigitSystemFontOfSize_weight_(size, weight)
            elif hasattr(NSFont, "systemFontOfSize_weight_"):
                font = NSFont.systemFontOfSize_weight_(size, weight)
            else:
                font = NSFont.systemFontOfSize_(size)
            attrs = {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: color,
            }
            text_value = NSString.stringWithString_(str(text))
            draw_x = float(x)
            draw_width = float(width)
            if align != "left":
                try:
                    text_size = text_value.sizeWithAttributes_(attrs)
                    measured_width = min(float(text_size.width) + 2.0, float(width))
                    if align == "center":
                        draw_x = float(x) + max(0.0, (float(width) - measured_width) / 2.0)
                    elif align == "right":
                        draw_x = float(x) + max(0.0, float(width) - measured_width)
                    draw_width = measured_width
                except Exception:
                    pass
            text_value.drawInRect_withAttributes_(
                NSMakeRect(draw_x, float(y), draw_width, float(height)), attrs
            )

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

    style = (
        NSWindowStyleMaskTitled
        | NSWindowStyleMaskClosable
        | NSWindowStyleMaskMiniaturizable
        | NSWindowStyleMaskResizable
    )
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(160.0, 130.0, 920.0, 620.0),
        style,
        NSBackingStoreBuffered,
        False,
    )
    window.setMinSize_(NSMakeSize(760.0, 560.0))
    window.setTitle_("Heart Beat Detector - Proof of Concept")
    view = TrackpadPulseView.alloc().initWithFrame_(window.contentView().bounds())
    view.setAutoresizingMask_(18)
    window.setContentView_(view)
    window.makeFirstResponder_(view)
    window.makeKeyAndOrderFront_(None)
    window.setAcceptsMouseMovedEvents_(True)
    NSApp.activateIgnoringOtherApps_(True)

    if use_touch:
        print("Rest one or two fingertips on the trackpad without clicking.")
    else:
        print("Pressure-only mode: hold a steady fingertip press inside the window.")
    print("Hold still for 15 seconds.")
    AppHelper.runEventLoop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate heart rate from MacBook trackpad touch or pressure."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run a synthetic pressure-signal test instead of opening the GUI",
    )
    parser.add_argument(
        "--pressure-only",
        action="store_true",
        help="disable light-touch private API and use Force Touch pressure events only",
    )
    parser.add_argument(
        "--touch-probe",
        action="store_true",
        help="print raw light-touch contact values instead of opening the GUI",
    )
    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=8.0,
        help="how long --touch-probe should run",
    )
    args = parser.parse_args(argv)
    if args.simulate:
        return run_simulation()
    if args.touch_probe:
        return run_touch_probe(max(0.5, args.probe_seconds))
    return run_gui(use_touch=not args.pressure_only)


if __name__ == "__main__":
    raise SystemExit(main())
