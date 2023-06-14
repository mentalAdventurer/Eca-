import numpy as np
import filter

EUCLIDEAN_TOL = 1

def rel_euclidean_distance(vector1, vector2):
    distance = np.linalg.norm(vector1 - vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    rel_distance = distance / (norm1+norm2)
    return rel_distance


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
