import wave
import numpy as np
import input_output_handler as io
import pytest


def test_read_input_type_not_supported(capfd):
    """
    Test if the function read_input raises an error if the type of the target is not supported.
    """
    target = 4.5
    data = io.read_input(target, 1024)
    captured = capfd.readouterr()
    assert captured.out.strip() == f"Target for reading of unkown type: {type(target)}"


def test_read_input_type_supported():
    """
    Test if the function get_targets returns the correct types and greate than 0.
    """
    input_filename = "./records/record_30.wav"
    target, _ = io.get_targets(input_filename, None)
    data = io.read_input(target, 1024)

    assert type(data) == np.ndarray
    assert data.size > 0


def test_write_output_sanity():
    """
    Test if the function write_output writes the correct data.
    """
    output_filename = "./tests/test.wav"
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
    output_filename = "./tests/test.wav"
    read_target, write_target = io.get_targets(input_filename, output_filename)
    assert type(read_target) == wave.Wave_read
    assert type(write_target) == wave.Wave_write

    input_filename = None
    output_filename = None
    read_target, write_target = io.get_targets(input_filename, output_filename)


def test_get_targets_wrong_type():
    """
    Test if the function get_targets raises an error if the type of the target is not supported.
    """
    input_filename = 5.7
    output_filename = "./tests/test.wav"
    with pytest.raises(TypeError, match="Input Filename must be a string or None"):
        read_target, write_target = io.get_targets(input_filename, output_filename)

    input_filename = "./records/record_30.wav"
    output_filename = 5.7
    with pytest.raises(TypeError, match="Output Filename must be a string or None"):
        read_target, write_target = io.get_targets(input_filename, output_filename)


def test_read_noise_type_not_supported():
    """
    Test if the function read_noise raises an error if the type of the target is not supported.
    """
    noise_filename = 4.5
    with pytest.raises(TypeError):
        noise = io.read_noise(noise_filename)

    noise_filename = []
    with pytest.raises(TypeError):
        noise = io.read_noise(noise_filename)

    noise_filename = 5
    with pytest.raises(TypeError):
        noise = io.read_noise(noise_filename)


def test_read_noise_type_supported():
    """
    Test if the function read_noise returns the correct types and greate than 0.
    """
    noise_filename = "./records/noise_30.wav"
    noise = io.read_noise(noise_filename)

    assert type(noise) == np.ndarray
    assert noise.size > 0


def test_clean_up_closing():
    """
    Test if the function clean_up closes the targets.
    """
    input_filename = "./records/record_30.wav"
    output_filename = "./tests/test.wav"
    read_target, write_target = io.get_targets(input_filename, output_filename)
    data = io.read_input(read_target, 1024)

    io.clean_up([read_target, write_target])

    with pytest.raises(ValueError, match="read of closed file"):
        _ = io.read_input(read_target, 1024)
    with pytest.raises(
        AttributeError, match="'NoneType' object has no attribute 'write'"
    ):
        _ = io.write_output(write_target, data)


def test_clean_up_wrong_type(capfd):
    """
    Test if the function clean_up raises an error if the type of the target is not supported.
    """
    input_filename = "./records/record_30.wav"
    output_filename = "./tests/test.wav"
    read_target, write_target = io.get_targets(input_filename, output_filename)
    data = io.read_input(read_target, 1024)

    # Check if the function raises an error if the target is not a list
    with pytest.raises(TypeError, match="Target must be a list"):
        io.clean_up(read_target)
    with pytest.raises(TypeError, match="Target must be a list"):
        io.clean_up(write_target)
    with pytest.raises(TypeError, match="Target must be a list"):
        io.clean_up(data)

    # Check if the function reacts if the target in the list is not supported
    io.clean_up([read_target, write_target, data])
    captured = capfd.readouterr()
    assert captured.out.strip() == f"Target for closing of unkown type: {type(data)}"
