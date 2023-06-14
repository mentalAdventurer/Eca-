import numpy as np
import filter
from filter import rel_euclidean_distance, signal_noise_ratio
import pytest
from scipy.io import wavfile

EUCLIDEAN_TOL = 1e-1
SNR_TOL = 20


def test_rel_euclidean_distance_zero():
    """
    Test rel_euclidean_distance for zero vector as input value.
    """
    zero_vector = np.zeros((2, 10000))
    one_vector = np.ones((2, 10000))
    distance1 = rel_euclidean_distance(zero_vector, one_vector)
    distance2 = rel_euclidean_distance(one_vector, zero_vector)
    distance3 = rel_euclidean_distance(zero_vector, zero_vector)

    assert distance1 == 1.0
    assert distance2 == 1.0
    assert distance3 == 0


def test_rel_euclidean_distance_high_num_neg():
    """
    Test rel_euclidean_distance for long array with high and negative numbers.
    """
    vector1 = np.ones((2, 20000)) * 50000
    vector2 = np.ones((2, 20000)) * -100000
    distance1 = rel_euclidean_distance(vector1, vector2)
    distance2 = rel_euclidean_distance(vector2, vector2)

    assert distance1 == 1.0
    assert distance2 == 0


def test_signal_noise_ratio_zero():
    """
    Test signal_noise_ratio for zero vector as input value.
    """
    zero_vector = np.zeros((2, 10000))
    one_vector = np.ones((2, 10000))
    snr1 = signal_noise_ratio(zero_vector, one_vector)
    snr2 = signal_noise_ratio(one_vector, zero_vector)
    snr3 = signal_noise_ratio(zero_vector, zero_vector)
    snr4 = signal_noise_ratio(one_vector, one_vector)

    assert snr1 == -np.inf
    assert snr2 == np.inf
    assert snr3 == 0
    assert snr4 == 0


def test_signal_noise_ratio_high_num_neg():
    """
    Test signal_noise_ratio for long array with high and negative numbers.
    """
    vector1 = np.ones((2, 20000)) * 50000
    vector2 = np.ones((2, 20000)) * -100000
    snr1 = signal_noise_ratio(vector1, vector2)
    snr2 = signal_noise_ratio(vector2, vector2)
    snr3 = signal_noise_ratio(vector2, vector1)

    assert snr1 < 0.0
    assert snr2 == 0
    assert snr3 > 0.0


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
