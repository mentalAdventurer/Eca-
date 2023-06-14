from noisereduce.noisereduce import SpectralGateStationary
import numpy as np

CHUNK = 1024
RATE = 44100
CHANNELS = 2


def rel_euclidean_distance(vector1, vector2):
    """
    Calculates the relative Euclidean distance between two vectors.

    :param vector1: First vector.
    :type vector1: numpy.ndarray
    :param vector2: Second vector.
    :type vector2: numpy.ndarray
    :raises ValueError: If both vectors do not have the same shape.
    :return: Relative Euclidean distance between the two vectors.
    :rtype: float
    """
    if vector1.shape != vector2.shape:
        raise ValueError("Both vector must have the same shape")

    distance = np.linalg.norm(vector1 - vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 + norm1 == 0:
        rel_distance = 0
    else:
        rel_distance = distance / (norm1 + norm2)

    return rel_distance


def signal_noise_ratio(signal, noise):
    """
    Calculates the signal-to-noise ratio (snr) between a signal and noise.

    :param signal: Signal array.
    :type signal: numpy.ndarray
    :param noise: Noise array.
    :type noise: numpy.ndarray
    :return: snr between the signal and noise.
    :rtype: float
    """
    # use int64 to avoid overflow
    signal_int64 = signal.astype(np.int64)
    noise_int64 = noise.astype(np.int64)

    signal_power = np.mean(signal_int64**2)
    noise_power = np.mean(noise_int64**2)

    if noise_power == 0:
        snr = np.inf
    else:
        snr = 10 * np.log10(signal_power / noise_power)

    return snr


def spectral_gate(data, rate, noise=None):
    sg = SpectralGateStationary(
        y=data,
        sr=rate,
        y_noise=noise,
        prop_decrease=1.0,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        n_std_thresh_stationary=1.5,
        tmp_folder=None,
        chunk_size=600000,
        padding=30000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        clip_noise_stationary=True,
        use_tqdm=False,
        n_jobs=-1,
    )
    reduced_noise = sg.get_traces()
    return reduced_noise
