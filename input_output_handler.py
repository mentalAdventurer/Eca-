import numpy as np
import pyaudio
import wave
from filter import CHUNK, RATE, CHANNELS, SAMPLEWIDTH
from scipy.io import wavfile


def get_args():
    return "./records/record_30.wav", "test.wav", None


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
    if input_filename:
        read_target = wave.open(input_filename, "rb")
    else:
        read_target = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
    if output_filename:
        write_target = wave.open(output_filename, "wb")
        write_target.setframerate(RATE)
        write_target.setnchannels(CHANNELS)
        write_target.setsampwidth(SAMPLEWIDTH)
    else:
        write_target = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK,
        )
    return read_target, write_target


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
        data = data.reshape((-1, channels))
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
        target.writeframes(data)
    elif type(target) == pyaudio.PyAudio.Stream:
        raise NotImplementedError


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
