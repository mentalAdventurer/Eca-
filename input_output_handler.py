import numpy as np
import pyaudio
import wave
from filter import CHUNK, RATE, CHANNELS, SAMPWIDTH
import argparse


def get_args():
    """
    Returns the arguments for the main loop

    :return: input_filename, output_filename, noise_filename
    :rtype: str, str, str
    """
    parser = argparse.ArgumentParser(description="Filtering of audio files")
    parser.add_argument("-i", "--input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("-n", "--noise", type=str, help="Noise file")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    noise_file = args.noise

    # Not Implemented
    if input_file is None:
        raise NotImplementedError("Input from microphone not implemented yet")
    if output_file is None:
        raise NotImplementedError("Output to speaker not implemented yet")

    return input_file, output_file, noise_file


def get_targets(input_filename, output_filename):
    """
    Returns the input and output target for the main loop.
    If input_filename is None, the input target is a pyaudio stream.
    If output_filename is None, the output target is a pyaudio stream.

    :param input_filename: filename of the input file
    :type input_filename: str or None
    :param output_filename: filename of the output file
    :type output_filename: str or None
    :return: input_target, output_target
    :rtype: tuple
    """
    read_target = None
    write_target = None

    # determine the input target
    if input_filename:
        read_target = wave.open(input_filename, "rb")
        params = read_target.getparams()
    else:
        params = None
        read_target = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

    # determine the output target
    if output_filename:
        write_target = wave.open(output_filename, "wb")

        # if set use the params from the input file
        if params:
            write_target.setparams(params)
        else:
            write_target.setframerate(RATE)
            write_target.setnchannels(CHANNELS)
            write_target.setsampwidth(SAMPWIDTH)

    else:
        write_target = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK,
        )

    return read_target, write_target


def read_noise(noise_filename):
    """
    Reads the noise file and returns it as a numpy array.

    :param noise_filename: filename of the noise file
    :type noise_filename: str
    :return: noise
    :rtype: numpy.ndarray
    """
    if noise_filename is None:
        return None

    noise_file = wave.open(noise_filename, "rb")
    num_frames = noise_file.getnframes()
    num_channels = noise_file.getnchannels()
    buffer = noise_file.readframes(num_frames)
    noise = np.frombuffer(buffer, dtype=np.int16)
    noise = noise.reshape((num_channels, -1))

    return noise


def read_input(target, chunk):
    """
    Reads the input from the target and returns it as a numpy array.

    :param target: target for reading
    :type target: wave.Wave_read or pyaudio.PyAudio.Stream
    :param chunk: number of frames to read from the target
    :type chunk: int
    :return: data
    :rtype: numpy.ndarray
    """

    data = np.array([])
    if type(target) == wave.Wave_read:
        channels = target.getnchannels()
        buffer = target.readframes(chunk)
        data = np.frombuffer(buffer, dtype=np.int16)
        data = data.reshape((channels, -1))
    elif type(target) == pyaudio.PyAudio.Stream:
        # TODO: Add support for pyaudio stream
        raise NotImplementedError
    else:
        print(f"Target for reading of unkown type: {type(target)}")
    return data


def write_output(target, data):
    """
    Writes the data to the target.

    :param target: target for writing
    :type target: wave.Wave_write or pyaudio.PyAudio.Stream
    :param data: data to write
    :type data: numpy.ndarray
    """
    if type(target) == wave.Wave_write:
        data = data.reshape((-1,))
        data = data.astype(np.int16).tobytes()
        target.writeframes(data)
    elif type(target) == pyaudio.PyAudio.Stream:
        raise NotImplementedError
    else:
        print(f"Target for reading of unkown type: {type(target)}")


def clean_up(clean_up_array):
    """
    Terminates and closes open streams, audios connections and files

    :param clean_up_array: Objekts to close or terminate
    :type clean_up_array: list
    """
    for item in clean_up_array:
        if type(item) == pyaudio.PyAudio.Stream:
            item.stop_stream()
            item.close()
        elif type(item) == pyaudio.PyAudio:
            item.terminate()
        elif type(item) == wave.Wave_read or type(item) == wave.Wave_write:
            item.close()
