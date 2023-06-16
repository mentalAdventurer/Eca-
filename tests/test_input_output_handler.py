import wave
import numpy as np
import pyaudio
import input_output_handler as io


def test_read_input_type_not_supported(capfd):
    """
    Test if the function read_input raises an error if the type of the target is not supported.
    """
    target = 4.5
    data = io.read_input(target, 1024)
    captured = capfd.readouterr()
    assert captured.out.strip() == f"Target for reading of unkown type: {type(target)}"


def test_write_output_sanity():
    """
    Test if the function write_output writes the correct data.
    """
    output_filename = "test.wav"
    input_filename = "./records/record_30.wav"

    read_target, write_target = io.get_targets(input_filename, output_filename)

    # Read Data from test file
    data = io.read_input(read_target, 1024)
    read_target.close()

    # Write Data to test file
    io.write_output(write_target, data)
    write_target.close()

    # Read Data from test file agian
    read_target, _ = io.get_targets(output_filename, None)
    new_data = io.read_input(read_target, 1024)
    read_target.close()

    assert np.array_equal(data, new_data)


def test_get_targets_type():
    """
    Test if the function get_targets returns the correct types.
    """

    input_filename = "./records/record_30.wav"
    output_filename = "test.wav"
    read_target, write_target = io.get_targets(input_filename, output_filename)
    assert type(read_target) == wave.Wave_read
    assert type(write_target) == wave.Wave_write

    input_filename = None
    output_filename = None
    read_target, write_target = io.get_targets(input_filename, output_filename)
    assert type(read_target) == pyaudio.PyAudio.Stream
    assert type(write_target) == pyaudio.PyAudio.Stream
