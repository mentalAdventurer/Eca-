import numpy as np
import pyaudio
import input_output_handler as io
from filter import RATE, CHUNK, filter_evaluation, print_result
from filter import spectral_gate as filter


def main(input_filename, output_filename, noise_filename):
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
    # initialize the input and output targets
    read_target, write_target = io.get_targets(input_filename, output_filename)
    clean_up_array = [read_target, write_target]

    # read the noise file
    noise = io.read_noise(noise_filename)

    # Collector for the evaluation results
    input_collector = []
    output_collector = []

    # loop: read -> filter -> write until empty or KeyboardInterrupt
    data = io.read_input(read_target, CHUNK)
    try:
        while data.size > 0 or type(read_target) == pyaudio.PyAudio.Stream:
            input_collector.append(data)
            reduced_noise = filter(data, RATE, noise)
            output_collector.append(reduced_noise)
            io.write_output(write_target, reduced_noise)
            data = io.read_input(read_target, CHUNK)
        io.clean_up(clean_up_array)

        # Evaluate the filtering result.
        input_signal = np.concatenate(input_collector, axis=1)
        output_signal = np.concatenate(output_collector, axis=1)
        evaluation = filter_evaluation(
            read_target, write_target, input_signal, output_signal
        )
        print_result(evaluation)

    except KeyboardInterrupt:
        io.clean_up(clean_up_array)

        # Evaluate the filtering result.
        input_signal = np.concatenate(input_collector)
        output_signal = np.concatenate(output_collector)
        evaluation = filter_evaluation(
            read_target, write_target, input_signal, output_signal
        )
        print_result(evaluation)


if __name__ == "__main__":
    input_filename, output_filename, noise_filename = io.get_args()
    main(input_filename, output_filename, noise_filename)
