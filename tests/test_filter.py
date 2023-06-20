import numpy as np
import filter
from filter import rel_euclidean_distance, signal_noise_ratio
import pytest

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


def test_mse_zero():
    """
    Test mse for zero vector as input value.
    """
    zero_vector = np.zeros((2, 10000))
    one_vector = np.ones((2, 10000))
    mse1 = filter.mse(zero_vector, one_vector)
    mse2 = filter.mse(one_vector, zero_vector)
    mse3 = filter.mse(zero_vector, zero_vector)
    mse4 = filter.mse(one_vector, one_vector)

    assert mse1 == 1.0
    assert mse2 == 1.0
    assert mse3 == 0
    assert mse4 == 0


def test_mse_high_num_neg():
    """
    Test mse for long array with high and negative numbers.
    Check for overflow.
    """
    vector1 = np.ones((2, 20000)) * 50000
    vector2 = np.ones((2, 20000)) * -100000
    mse1 = filter.mse(vector1, vector2)
    mse2 = filter.mse(vector2, vector2)
    mse3 = filter.mse(vector2, vector1)

    assert mse1 == 22500000000.0
    assert mse2 == 0
    assert mse3 == 22500000000.0


def test_mse():
    """
    Test mse for two vectors with same values.
    """
    vector1 = np.ones((2, 20))
    vector2 = np.ones((2, 20))
    mse1 = filter.mse(vector1, vector2)

    assert mse1 == 0


def test_rmse_zero():
    """
    Test rmse for zero vector as input value.
    """
    zero_vector = np.zeros((2, 10000))
    one_vector = np.ones((2, 10000))
    rmse1 = filter.rmse(zero_vector, one_vector)
    rmse2 = filter.rmse(one_vector, zero_vector)
    rmse3 = filter.rmse(zero_vector, zero_vector)
    rmse4 = filter.rmse(one_vector, one_vector)

    assert rmse1 == 1.0
    assert rmse2 == 1.0
    assert rmse3 == 0
    assert rmse4 == 0


def test_rmse_high_num_neg():
    """
    Test rmse for long array with high and negative numbers.
    Check for overflow.
    """
    vector1 = np.ones((2, 20000)) * 50000
    vector2 = np.ones((2, 20000)) * -100000
    rmse1 = filter.rmse(vector1, vector2)
    rmse2 = filter.rmse(vector2, vector2)
    rmse3 = filter.rmse(vector2, vector1)

    assert rmse1 == 150000.0
    assert rmse2 == 0
    assert rmse3 == 150000.0


def test_rmse():
    """
    Test rmse for two vectors with same values.
    """
    vector1 = np.ones((2, 20))
    vector2 = np.ones((2, 20))
    rmse1 = filter.rmse(vector1, vector2)

    assert rmse1 == 0


def test_psnr_zero():
    """
    Test psnr for zero vector as input value.
    """
    zero_vector = np.zeros((2, 10000))
    one_vector = np.ones((2, 10000))
    psnr1 = filter.psnr(zero_vector, one_vector)
    psnr2 = filter.psnr(one_vector, zero_vector)
    psnr3 = filter.psnr(zero_vector, zero_vector)
    psnr4 = filter.psnr(one_vector, one_vector)

    assert psnr1 == -np.inf
    assert psnr2 == -np.inf
    assert psnr3 == np.inf
    assert psnr4 == np.inf


def test_psnr():
    """
    Test psnr for two vectors with same values.
    """
    vector1 = np.ones((2, 20))
    vector2 = np.ones((2, 20))
    psnr1 = filter.psnr(vector1, vector2)

    assert psnr1 == np.inf


def test_ncc_zero():
    """
    Test ncc for zero vector as input value.
    """
    zero_vector = np.zeros((2, 10000))
    one_vector = np.ones((2, 10000))
    ncc1 = filter.ncc(zero_vector, one_vector)
    ncc2 = filter.ncc(one_vector, zero_vector)
    ncc3 = filter.ncc(zero_vector, zero_vector)
    ncc4 = filter.ncc(one_vector, one_vector)

    assert ncc1 == 0
    assert ncc2 == 0
    assert ncc3 == 0
    assert ncc4 == 0


def test_power_loss_zero():
    """
    Test power_loss for zero vector as input value.
    """
    zero_vector = np.zeros((2, 10000))
    one_vector = np.ones((2, 10000))
    power_loss1 = filter.power_loss(zero_vector, one_vector)
    power_loss2 = filter.power_loss(one_vector, zero_vector)
    power_loss3 = filter.power_loss(zero_vector, zero_vector)
    power_loss4 = filter.power_loss(one_vector, one_vector)

    assert power_loss1 == -1
    assert power_loss2 == 1
    assert power_loss3 == 0
    assert power_loss4 == 0


def test_power_loss():
    """
    Test power_loss for two vectors with same values.
    """
    vector1 = np.ones((2, 20))
    vector2 = np.ones((2, 20))
    power_loss1 = filter.power_loss(vector1, vector2)

    assert power_loss1 == 0


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
