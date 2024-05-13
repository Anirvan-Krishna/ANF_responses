# EC60004: Neuronal Coding of Sensory Information 

 This repository consists the MATLAB code files for the course project if the course Neuronal Coding of Sensory Information at IIT Kharagpur. The project analysies the neural perception in the auditory system using cat auditory nerve fiber model propose by Zilany and Carney.

## Part A

### Question 1: Response to Tones (Rate Representation)

This part of the code analyzes the response of auditory nerve fibers to tones at different frequencies and intensities.

#### Initializations

- `bf`: Best frequencies for analysis.
- `intensities`: Sound Pressure Level (SPL) in dB.
- `tones`: Tone frequencies.
- `ramp_time`: Ramp time for stimuli.
- `duration`: Stimulus duration.
- `Fs`: Sampling frequency.

#### PSTH Parameters

- `repititions`: Number of repetitions of stimulus.
- `rate_matrix`: Matrix to store rates.

#### Creating Stimuli and Generating Output

This section generates stimuli and calculates the corresponding response rates of auditory nerve fibers.

#### Plotting

The code plots rate vs frequency and rate vs intensity graphs.

### Question 2

This section analyzes the response of auditory nerve fibers to stimuli with different intensities and frequencies.

#### Initializations

- `bf`: Best frequency.
- `intensities`: Sound Pressure Level (SPL) in dB.
- `tones`: Tone frequencies.

#### PSTH Parameters

- `repititions`: Number of repetitions of stimulus.
- `rate_matrix`: Matrix to store rates.

#### Creating Stimuli and Generating Output

This section generates stimuli and calculates the corresponding response rates of auditory nerve fibers.

#### Plotting

The code plots rate vs intensity graphs.

### Question 3

This section analyzes the response of auditory nerve fibers to speech signals.

#### Initializations

- `nF`: Frequencies for analysis.
- `cmap1`: Color map.
- `win`: Window size.
- `wshift`: Window shift.
- `t3`: Time vector.

#### Generating Output

This section generates stimuli from speech signals and calculates the corresponding response rates of auditory nerve fibers.

#### Plotting

The code plots spectrograms and response rates.

## Part B (Extra Credit)

This part of the code processes audio signals using bandpass filters and generates output audio files.

#### Initializations

- `N`: Number of bandpass filters.
- `z`: Accumulated results.

#### Audio Processing

The code applies bandpass filters to the audio signal and generates output files.

## Instructions

- Clone the repository.
- Run the MATLAB scripts for analysis.

# References
Zilany, Muhammad S.A., and Laurence R. Carney. "Power-law dynamics in an auditory-nerve model." Journal of Neuroscience 30.25 (2010): 8439-8452.

