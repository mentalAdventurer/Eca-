import numpy as np
import filter
from filter import rel_euclidean_distance, signal_noise_ratio
import pytest
from scipy.io import wavfile

EUCLIDEAN_TOL = 1e-1
SNR_TOL = 20


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sinus_euc():
    """
    Test filter for relative euclidean distance with sinusoidal signal and zero noise
    """
    duration = 30
    sampling_rate = 44100
    frequency = 100

    # Generate signal and noise
    t = np.linspace(0, duration, int(duration * sampling_rate))
    signal = 100 * np.sin(2 * np.pi * frequency * t)
    noise = np.zeros_like(signal)

    reduced_noise = filter.spectral_gate(signal, sampling_rate, noise=noise)

    distance = rel_euclidean_distance(reduced_noise, signal)
    assert distance < EUCLIDEAN_TOL


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sinus_random_noise_euc():
    """
    Test filter for relative euclidean distance with sinusodal signal and random generated noise.
    """
    duration = 1.0
    sampling_rate = 44100
    frequency = 100

    # Generate signal and noise
    t = np.linspace(0, duration, int(duration * sampling_rate))
    signal = 100 * np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, 1, len(signal))
    noise_signal = signal + noise

    reduced_noise = filter.spectral_gate(noise_signal, sampling_rate, noise=noise)

    distance = rel_euclidean_distance(reduced_noise, signal)
    assert distance < EUCLIDEAN_TOL


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sinus_snr():
    """
    Test filter for signal to noise ratio with sinusoidal signal and zero noise
    """
    duration = 1.0
    sampling_rate = 44100
    frequency = 100

    # Generate signal and noise
    t = np.linspace(0, duration, int(duration * sampling_rate))
    signal = 100 * np.sin(2 * np.pi * frequency * t)
    noise = np.zeros_like(signal)

    reduced_noise = filter.spectral_gate(signal, sampling_rate, noise=noise)

    snr = signal_noise_ratio(reduced_noise, noise)
    assert snr > SNR_TOL


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sinus_random_noise_snr():
    """
    Test filter for signal to noise ratio with sinusoidal signal and random generated noise.
    """
    duration = 1.0
    sampling_rate = 44100
    frequency = 100

    # generate signal
    t = np.linspace(0, duration, int(duration * sampling_rate))
    signal = 100 * np.sin(2 * np.pi * frequency * t)

    # add silence to beginning of signal
    noise_len = signal.shape[0]
    signal = np.hstack((np.zeros_like(signal), signal))

    # add noise to signal
    noise = np.random.normal(0, 1, len(signal))
    noise_signal = signal + noise

    # Filter Signal
    signal_reduced_noise = filter.spectral_gate(
        noise_signal, sampling_rate, noise=noise
    )
    reduced_noise = signal_reduced_noise[0:noise_len]

    snr = signal_noise_ratio(signal_reduced_noise, reduced_noise)
    assert snr > SNR_TOL


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_audio_snr():
    """
    Test filter for signal to noise ratio with example audio file.
    Position of the noise in the signal is static and the signal is known.
    """
    rate_signal, signal = wavfile.read("./records/record_30.wav")
    noise_range = int(signal.shape[0] / 6)  # extract part of the signal just noise

    # Filter
    signal_reduced_noise = filter.spectral_gate(signal.T, rate_signal, noise=None)
    reduced_noise = signal_reduced_noise[:, 1000:noise_range]

    snr = signal_noise_ratio(signal_reduced_noise[0, :], reduced_noise[0, :])
    assert snr > SNR_TOL
