# Projekt-Exam 2

---
author: Fabian Tschohl
date: 20.06.2022
---

## Overview and Motivation
When the fan starts on my notebook, a background noise is generated. The aim would be to filter this background noise. The programme should be able to read audio streams from microphones and audio files, filter the background noise and then send the filtered signal to a virtual microphone or an audio file. Ideally, the programme should filter the noise in real time. **However, only the filtering of wav-files should be relevant for the grading in this project.**

**Project progress:**
- [x] Filter implementation¹
- [x] Read from wav-file¹
- [x] write to wav-file¹
- [ ] read from microphone
- [ ] write to speaker

¹Exam relevant

## Dependencies
- numpy (v1.24.3)
- scipy (v1.10.1)
- noisereduce (v2.0.1)
- pyaudio >= 0.2.13 (**required**)
- wave

Although reading from the microphone and writing to the speaker have not yet been implemented, preparations have already been made. Therefore pyaudio is needed as a dependency. 

> **Warning**
> The program uses a type comparison with `pyaudio.PyAudio.Stream`.
> This is implemented in the latest version of pyaudio. Make sure you use the latest version.

## Getting started

For the first use of the programme, a sample audio file is attached to the project. Simply execute the programm by providing an output filename and input filename.

```sh
python ./main.py -i ./records/record_30.wav -o filter_output.wav
```
If the user has an isolated recording of the background noise, this can be passed to the programme to create a customised filter mask.

```sh
python ./main.py -i ./records/record_30.wav -o filter_output.wav -n ./records/noise_30.wav
```

> **Warning**
> If the input or ouput file argument is missing, the programme will use the microphone or the speaker as default.
> As this is not yet implemented, this will lead to an error.

## Testing
This project uses the pytest framwork for testing the functions. The corresponding files can be found in the folder `./tests/`.
To run the tests, execute the following command.
```sh
pytest
```
and 
```sh
pytest -v
```
to get more Information.

> **Note**
> Make sure that you are also in the root folder of the project.
 
> **Note**
> During the tests, a file called `test.wav` is created.
> There is no guarantee or requirement that this file is a valid audio file.
> It is only used for testing purposes.  

## Grading Information
### Nested Structur
You may or may not have trouble finding the nested structure. 
In order to fully meet this requirement, an evaluation of the audio has been created which uses a nested structure to collect and calculate the audio signal's characteristic metrics. 
This is used to print user information at the end of the program flow. Take a look at the `filter_evaluation` function in the `filter.py` file.

### Pytest
You may also be wondering why pytest has been used instead of doctest. Of course, this is not an expression of youthful recklessness but was discussed and approved with Joanna.

## Troubleshooting
#### NotImplementedError: Output to speaker not implemented yet
You have to define input and output filename parameter in oder to work

#### AttributeError: type object 'PyAudio' has no attribute 'Stream'
You have not the latest pyaudio == 0.2.13 installed. Try to upgrade it.
If you are unable to upgrade due to lack of support for your distribution.
As a **last resort**, there is a version without pyaudio available on github, but keep in mind that this is not the main branch. Some tests have been excluded and are not as thoroughly tested.
```sh
git clone -b pyaudio-disable https://github.com/mentalAdventurer/Python-Exam2.git
```
