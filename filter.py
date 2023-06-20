"""
Noise Reduction using Spectral Gating: Filter

This module contains the filter functions for the noise reduction using spectral gating.

:Author: Fabian Tschohl
:Student id: 51843947
:Date: 20.06.2023
"""

from noisereduce.noisereduce import SpectralGateStationary
import numpy as np
import wave

CHUNK = 8192
RATE = 44100
CHANNELS = 2
SAMPLEWIDTH = 2
SAMPWIDTH = 2


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

    if norm1 + norm2 == 0:
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

    if signal_power == 0 and noise_power == 0:
        snr = 0
    elif signal_power == 0:
        snr = -np.inf
    elif noise_power == 0:
        snr = np.inf
    else:
        snr = 10 * np.log10(signal_power / noise_power)

    return snr


def mse(signal1, signal2):
    """
    Calculates the mean squared error (mse) between two signals.

    :param signal1: First signal array.
    :type signal1: numpy.ndarray
    :param signal2: Second signal array.
    :type signal2: numpy.ndarray
    :return: mse between the two signals.
    :rtype: float
    """
    # use int64 to avoid overflow
    signal1_int64 = signal1.astype(np.int64)
    signal2_int64 = signal2.astype(np.int64)

    mse = np.mean((signal1_int64 - signal2_int64) ** 2)

    return mse


def rmse(signal1, signal2):
    """
    Calculates the root mean squared error (rmse) between two signals.

    :param signal1: First signal array.
    :type signal1: numpy.ndarray
    :param signal2: Second signal array.
    :type signal2: numpy.ndarray
    :return: rmse between the two signals.
    :rtype: float
    """

    return np.sqrt(mse(signal1, signal2))


def psnr(signal1, signal2):
    """
    Calculates the peak signal-to-noise ratio (psnr) between two signals.

    :param signal1: First signal array.
    :type signal1: numpy.ndarray
    :param signal2: Second signal array.
    :type signal2: numpy.ndarray
    :return: psnr between the two signals.
    :rtype: float
    """
    # use int64 to avoid overflow
    signal1_int64 = signal1.astype(np.int64)
    signal2_int64 = signal2.astype(np.int64)

    mse = np.mean((signal1_int64 - signal2_int64) ** 2)

    if mse == 0:
        psnr = np.inf
    elif np.max(signal1_int64) - np.min(signal1_int64) == 0:
        psnr = -np.inf
    else:
        psnr = 10 * np.log10((np.max(signal1_int64) - np.min(signal1_int64)) ** 2 / mse)

    return psnr


def ncc(signal1, signal2):
    """
    Calculates the normalized cross-correlation (ncc) between two signals.

    :param signal1: First signal array.
    :type signal1: numpy.ndarray
    :param signal2: Second signal array.
    :type signal2: numpy.ndarray
    :return: ncc between the two signals.
    :rtype: float
    """
    # use int64 to avoid overflow
    signal1_int64 = signal1.astype(np.int64)
    signal2_int64 = signal2.astype(np.int64)

    signal1_mean = np.mean(signal1_int64)
    signal2_mean = np.mean(signal2_int64)

    signal1_std = np.std(signal1_int64)
    signal2_std = np.std(signal2_int64)

    if signal1_std == 0 or signal2_std == 0:
        ncc = 0
    else:
        ncc = np.mean(
            (signal1_int64 - signal1_mean) * (signal2_int64 - signal2_mean)
        ) / (signal1_std * signal2_std)

    return ncc


def power_loss(signal1, signal2):
    """
    Calculates the power loss between two signals

    :param signal1: First signal array.
    :type signal1: numpy.ndarray
    :param signal2: Second signal array.
    :type signal2: numpy.ndarray
    :return: power loss between the two signals.
    :rtype: float
    """
    # use int64 to avoid overflow
    signal1_int64 = signal1.astype(np.int64)
    signal2_int64 = signal2.astype(np.int64)

    signal1_power = np.mean(signal1_int64**2)
    signal2_power = np.mean(signal2_int64**2)

    return signal1_power - signal2_power


def spectral_gate(data, rate, noise=None):
    """
    Filter out noise from signal using spectral gating method.

    :param data: data to be filtered
    :type data: numpy.ndarray
    :param rate: sample rate of data
    :type rate: int
    :param noise: noise for masking
    :type noise: numpy.ndarray
    :return: filtered data
    :rtype: numpy.ndarray
    """

    # when last section is too narrow discard the part
    min_data_size = 1024
    if data.size <= min_data_size:
        return np.zeros_like(data)

    # filter out noise from signal using spectral gating method
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
        n_fft=min_data_size,
        win_length=None,
        hop_length=None,
        clip_noise_stationary=True,
        use_tqdm=False,
        n_jobs=-1,
    )
    reduced_noise = sg.get_traces()
    return reduced_noise


def filter_evaluation(read_target, write_target, signal1, signal2):
    """
    Evaluate the filtering result.

    :param read_target: read target
    :type read_target: str
    :param write_target: write target
    :type write_target: str
    :param signal1: First signal array.
    :type signal1: numpy.ndarray
    :param signal2: Second signal array.
    :type signal2: numpy.ndarray
    :return: evaluation result
    :rtype: dict
    """

    evaluation = {"Sig1": {}, "Sig2": {}, "Diff": {}}

    # Set the evaluation metrics for the first signal.
    if type(read_target) == wave.Wave_read:
        evaluation["Sig1"]["Source"] = "File"
        evaluation["Sig1"]["SampleRate"] = read_target.getframerate()
        evaluation["Sig1"]["Channels"] = read_target.getnchannels()
        evaluation["Sig1"]["BitsPerSample"] = read_target.getsampwidth() * 8
    else:
        raise TypeError(
            "read_target must be either wave.Wave_read or pyaudio.PyAudio.Stream"
        )

    # Set the evaluation metrics for the second signal.
    if type(write_target) == wave.Wave_write:
        evaluation["Sig2"]["Sink"] = "File"
        evaluation["Sig2"]["SampleRate"] = write_target.getframerate()
        evaluation["Sig2"]["Channels"] = write_target.getnchannels()
        evaluation["Sig2"]["BitsPerSample"] = write_target.getsampwidth() * 8
    else:
        raise TypeError(
            "write_target must be either wave.Wave_write or pyaudio.PyAudio.Stream"
        )

    # Calculate the evaluation metrics for differences between the two signals.
    evaluation["Diff"]["mse"] = mse(signal1, signal2)
    evaluation["Diff"]["rmse"] = rmse(signal1, signal2)
    evaluation["Diff"]["psnr"] = psnr(signal1, signal2)
    evaluation["Diff"]["ncc"] = ncc(signal1, signal2)
    evaluation["Diff"]["power_loss"] = power_loss(signal1, signal2)

    return evaluation


def print_result(evaluation):
    """
    Print the evaluation results.

    :param evaluation: evaluation result
    :type evaluation: dict
    """
    for key in evaluation:
        print(key)
        for k, v in evaluation[key].items():
            print(f"\t{k}: {v}")

    print("\nEvaluation Metrics Legend:")
    print("(mse):".ljust(15), "mean square error")
    print("(rmse):".ljust(15), "root mean square error")
    print("(psnr):".ljust(15), "peak signal-to-noise ratio")
    print("(ncc):".ljust(15), "normalized cross correlation")
    print("(power_loss):".ljust(15), "power loss")
