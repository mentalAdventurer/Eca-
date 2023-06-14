import numpy as np  
import filter
from filter import rel_euclidean_distance, signal_noise_ratio
import pytest
from scipy.io import wavfile

EUCLIDEAN_TOL = 1e-1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sinus():
    duration = 1.0
    sampling_rate = 44100
    frequency = 100

    t = np.linspace(0, duration, int(duration * sampling_rate))
    signal = 100 * np.sin(2 * np.pi * frequency * t)
    noise = np.zeros_like(signal)

    reduced_noise = filter.spectral_gate(signal, sampling_rate, noise=noise)

    distance = rel_euclidean_distance(reduced_noise, signal)
    assert distance < EUCLIDEAN_TOL


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sinus_random_noise():
    duration = 1.0
    sampling_rate = 44100
    frequency = 100

    t = np.linspace(0, duration, int(duration * sampling_rate))
    signal = 100 * np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, 1, len(signal))
    noise_signal = signal + noise

    reduced_noise = filter.spectral_gate(noise_signal, sampling_rate, noise=noise)

    distance = rel_euclidean_distance(reduced_noise, signal)
    assert distance < EUCLIDEAN_TOL
