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

## Grading Information
### Nested Structur
You may or may not have trouble finding the nested structure. 
In order to fully meet this requirement, an evaluation of the audio has been created which uses a nested structure to collect and calculate the audio signal's characteristic metrics. 
This is used to print user information at the end of the program flow. Take a look at the `filter_evaluation` function in the `filter.py` file.

### Pytest
You may also be wondering why pytest has been used instead of doctest. Of course, this is not an expression of youthful recklessness but was discussed and approved with Joanna.

