from main import main


def test_main():
    input_filename = "./records/record_30.wav"
    output_filename = ".test.wav"
    main(input_filename, output_filename)
    assert True
