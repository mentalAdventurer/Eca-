from noisereduce.noisereduce import SpectralGateStationary

CHUNK = 1024
RATE = 44100
CHANNELS = 2


def spectral_gate(data,rate,noise=None):
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
