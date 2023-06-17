import numpy as np
import pyaudio
import input_output_handler as io
from filter import RATE, CHUNK
from filter import spectral_gate as filter


def main(input_filename, output_filename, noise_filename=None):
    """
    Assembly of filter and streams

    :param input_filename: filename of input file
    :type input_filename: str or None
    :param output_filename: filename of output file
    :type output_filename: str or None
    :param noise_filename: filename of noise file
    :type noise_filename: str or None
    :raises KeyboardInterrupt: if user interrupts the program
    """

    # TODO: implement noise_filename
    # TODO: implement argument parser
    # TODO: implement nested structure

    # initialize the input and output targets
    read_target, write_target = io.get_targets(input_filename, output_filename)
    clean_up_array = [read_target, write_target]

    # loop: read -> filter -> write until empty or KeyboardInterrupt
    data = io.read_input(read_target, CHUNK)
    try:
        while data.size > 0 or type(read_target) == pyaudio.PyAudio.Stream:
            reduced_noise = filter(data, RATE)
            io.write_output(write_target, reduced_noise)
            data = io.read_input(read_target, CHUNK)
        io.clean_up(clean_up_array)

    except KeyboardInterrupt:
        io.clean_up(clean_up_array)


## Main Loop
if __name__ == "__main__":
    input_filename, output_filename, noise_filename = io.get_args()
    main(input_filename, output_filename, noise_filename)
