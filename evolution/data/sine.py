import matplotlib.pyplot as plt
import numpy as np
from data import Dataset


def generate_sine_wave(
    frequency: float = 10,
    amplitude: float = 1.0,
    phase: float = 0,
    sampling_rate: int = 1000,
    duration: float = 1,
) -> "Dataset":
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
    Dataset
        A named tuple containing the sine wave, the indices array, and the parameters.
    """
    metadata = {
        "frequency": frequency,
        "amplitude": amplitude,
        "phase": phase,
        "sampling_rate": sampling_rate,
        "duration": duration,
    }
    indices = np.linspace(0, duration, int(sampling_rate * duration))
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * indices + phase)
    return Dataset(sine_wave, indices, metadata, 0)


def plot_sine_wave(sine_wave: Dataset):
    """Plot the sine wave using matplotlib."""
    plt.plot(sine_wave.indices, sine_wave.data)
    plt.title("Sine Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


sine_dataset = generate_sine_wave(
    frequency=10, amplitude=1.0, phase=0, sampling_rate=1000, duration=1
)
