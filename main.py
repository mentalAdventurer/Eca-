import numpy as np
import pyaudio
import input_output_handler as io
from filter import RATE, CHUNK, CHANNELS, SAMPLEWIDTH
from filter import spectral_gate as filter

CHUNK = 1024


def main(input_filename, output_filename, noise_filename=None):
    read_target, write_target = io.get_targets(input_filename, output_filename)
    clean_up_array = [read_target, write_target]

    audio = pyaudio.PyAudio()
    clean_up_array.append(audio)

    data = io.read_input(read_target, CHUNK)
    try:
        while len(data) > 0 or type(read_target) == pyaudio.PyAudio.Stream:
            reduced_noise = filter(data.T, RATE)
            io.write_output(write_target, reduced_noise.T)
            data = io.read_input(read_target, CHUNK)
        io.clean_up(clean_up_array)

    except KeyboardInterrupt:
        io.clean_up(clean_up_array)


## Main Loop
if __name__ == "__main__":
    input_filename, output_filename, noise_filename = io.get_args()
    main(input_filename, output_filename, noise_filename)
