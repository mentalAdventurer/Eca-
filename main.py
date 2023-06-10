import numpy as np
import pyaudio
import input_output_handler as io
from noisereduce.noisereduce import SpectralGateStationary
import wave
import pytest

CHUNK = 1024
RATE = 44100
CHANNELS = 2

# Paser
input_filename, output_filename, noise_filename = io.get_args()

## Init
sg = SpectralGateStationary(
    y=[],
    sr=RATE,
    y_noise=None,
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
clean_up_array = []
audio = pyaudio.PyAudio()
clean_up_array.append(audio)
rate = RATE
data = []

if input_filename:
    read_target = wave.open(input_filename, "rb")
    clean_up_array.append(read_target)
    rate = read_target.getframerate()
    samplewidth = read_target.getsampwidth()
    data = [0]
else:
    read_target = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    samplewidth = audio.get_sample_size(pyaudio.paInt16)
    clean_up_array.append(read_target)

if output_filename:
    write_target = wave.open(output_filename, "wb")
    write_target.setframerate(RATE)
    write_target.setnchannels(CHANNELS)
    write_target.setsampwidth(samplewidth)
    clean_up_array.append(write_target)
else:
    write_target = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=rate,
        output=True,
        frames_per_buffer=CHUNK,
    )
    clean_up_array.append(write_target)

try:
    while len(data) > 0 or type(read_target) == pyaudio.PyAudio.Stream:
        data = io.read_input(read_target,CHUNK)
        reduced_noise = sg.get_traces()
        io.write_output(bool(output_filename), reduced_noise)
    io.clean_up(clean_up_array)

except KeyboardInterrupt:
    io.clean_up(clean_up_array)
