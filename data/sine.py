# Plot a sine wave
from typing import Tuple

import numpy as np


def generate_sine_wave(
    frequency: float = 10,
    amplitude: float = 1.0,
    phase: float = 0,
    sampling_rate: int = 1000,
    duration: float = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a sine wave with the given frequency, amplitude, phase, sampling rate, and duration.

    Parameters
    ----------
    frequency : float
        The frequency of the sine wave in Hz.
    amplitude : float
        The amplitude of the sine wave.
    phase : float
        The phase of the sine wave in radians.
    sampling_rate : int
        The sampling rate of the sine wave in Hz.
    duration : float
        The duration of the sine wave in seconds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the sine wave and the time array.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration))
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return sine_wave, t


sine_wave, t = generate_sine_wave()
