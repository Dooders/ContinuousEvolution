# Plot a sine wave
import argparse
import collections

import matplotlib.pyplot as plt
import numpy as np

SineWave = collections.namedtuple("SineWave", ["data", "indices", "metadata"])


def generate_sine_wave(
    frequency: float = 10,
    amplitude: float = 1.0,
    phase: float = 0,
    sampling_rate: int = 1000,
    duration: float = 1,
) -> SineWave:
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
    SineWave
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
    return SineWave(sine_wave, indices, metadata)


def plot_sine_wave(sine_wave: SineWave):
    """Plot the sine wave using matplotlib."""
    plt.plot(sine_wave.indices, sine_wave.data)
    plt.title("Sine Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot a sine wave.")
    parser.add_argument(
        "--frequency", type=float, default=10, help="Frequency of the sine wave in Hz"
    )
    parser.add_argument(
        "--amplitude", type=float, default=1.0, help="Amplitude of the sine wave"
    )
    parser.add_argument(
        "--phase", type=float, default=0, help="Phase of the sine wave in radians"
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=1000, help="Sampling rate in Hz"
    )
    parser.add_argument(
        "--duration", type=float, default=1, help="Duration of the sine wave in seconds"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Don't plot the sine wave"
    )

    args = parser.parse_args()

    sine_wave = generate_sine_wave(
        frequency=args.frequency,
        amplitude=args.amplitude,
        phase=args.phase,
        sampling_rate=args.sampling_rate,
        duration=args.duration,
    )

    print(f"Generated sine wave with parameters: {sine_wave.metadata}")

    if not args.no_plot:
        plot_sine_wave(sine_wave)
