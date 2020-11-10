# Automatic-Calibration

This is a LabVIEW implementation of the automatic calibration method described in "Automation of binaural headphone audio calibration on an artificial head" (Kenneth Ooi, Yonggang Xie, Bhan Lam, Woon-Seng Gan).

Getting started
---------------
Firstly, clone this repository by manually downloading it from https://github.com/xxxx/xxxx.git, or enter the following line from a terminal:

    git clone https://github.com/xxxx/xxxx.git

Then, run the LabVIEW Virtual Instrument (VI) at `Code\Main.vi`. There is a user guide in the VI, which is also replicated here in the <a href='#User Guide'>User Guide</a> section.

The <a href='#system_setup'>System setup</a> section details the exact software versions that we used to create and test the VI.

Lastly, if you encounter a situation where LabVIEW asks you to choose dependencies when opening `Main.vi`, choose `MeasureLeq.vi` for "Leqfunction(SubVI)" and `GlobalVariable.vi` for "Global 5".

Directory structure
-------------------
The structure of the repository is as follows:

    .
    ├── Code                       
    │   ├── GlobalVariable.vi      # A global variable that records the maximum amplitude at any point of a given audio track. Used to check if signal has been clipped.
    │   ├── Main.vi                # Top-level VI with detailed user guide. Run this.
    │   └── MeasureLeq.vi          # Sub-VI to measure Leq of audio tracks.
    │   
    ├── README.md                  # This file
    │
    ├── SampleCSV                  
    │   └── Input.csv              # CSV file with input parameters for our working example.
    │
    └── SampleData                 # Sample 10-second long tracks for our working example.
        ├── pink_noise.wav
        ├── sine_1000hz.wav
        ├── sine_250hz.wav
        └── white_noise.wav

Example
-------
We have also provided an example to show how the automatic calibration VI works.

1. Select `Input.csv` in the `SampleCSV` folder as the input CSV file in `Main.vi` and type in the path to the CSV file to write the outputs to.
2. Enter your settings and parameters according to the user guide in `Main.vi`.
3. Click the "Run" button or press "Ctrl+R" to run the VI.
    
System setup <a name='system_setup'>
------------
The system we used to test the code in this repository is as follows: 
- OS: Windows 10
- Processor: Intel Core i7-7700HQ CPU @ 2.80GHz
- RAM: 16.0GB

The main driver and software versions were:
- LabVIEW 2017, 32-bit (downloadable <a href='https://www.ni.com/en-sg/support/downloads/software-products/download.labview.html'> here </a> with an NI user account)
- LabVIEW Sound and Vibration Toolkit 2019 (downloadable <a href='https://www.ni.com/en-sg/shop/software/products/labview-sound-and-vibration-toolkit.html'> here </a> with an NI user account)

User Guide <a name='User Guide'>
-----------

This LabVIEW Virtual Instrument (VI) helps you to automatically search for suitable gains to apply to audio tracks (mono or stereo) stored in a folder of your system to an Leq level (within a certain tolerance) that you can specify. The VI will output the values of these gains into a CSV file, which you can then apply to those audio tracks in order to calibrate them to the Leq level that you specified. Note that the original audio tracks are not modified in any way by the VI.

The VI searches for the suitable gains by firstly adjusting an initial guess based on an initial measurement. If the adjustment is not within the desired tolerance level, then a further boundary search, followed by a binary search, is used to refine the gain to the desired tolerance level. For more information on the calibration algorithm, please refer to our paper "Automation of binaural headphone audio calibration on an artificial head".

The steps to use this VI are as follows:
                                                              
**1. Set overall parameters**
1) Under "Path to input CSV file", type in the path to the CSV file containing the filepaths for the tracks to calibrate, together with the input parameters for each track<a href="#*">*</a>. 
2) Under "Path to output CSV file", type in the path to the CSV file that you want to save the VI's output to. The VI will append to the rows of this CSV file if it already exists, and will the create the CSV file if it does not already exist <a href="#**"> **</a>.

**2. Set Leq Measurement Parameters**
1) Under both "Recording channel (left)" and "Recording channel (right)", enter the corresponding sensitivity, units of measurement, value of the reference for dB calculation, and the pregain of the microphones corresponding to the left and right channels of the head and torso simulator, respectively.
2) Under "Playback Device ID", enter the integer corresponding to the device ID of your playback device (which should normally be a pair of circumaural headphones or a soundcard). To check the device IDs of all playback/recording devices on your system, please refer to the tab "X. Sound Device List".
3) Under "Weighting setting", select the weighting filter to be applied to the values recorded by the micrphones on the head and torso simulator.
4) If necessary, the "XControl Settings" panel allows you to specify additional device settings according to your recording devices <a href="#***">***</a>. These include:
     - "NI-DAQmx Channel" -- The names of the input channels corresponding to the left and right channels of the head and torso 
                             simulator. The left channel name should be above the right channel name in this column. E.g. if
                             "cDAQ1Mod1/ai1" corresponds to the left channel and "cDAQ1Mod1/ai0" corresponds to the right
                             channel, then the order in which the channels should appear in this column is "cDAQ1Mod1/ai1" on 
                             top, and "cDAQ1Mod1/ai0" at the bottom.
     - "Coupling" -- The coupling mode for the microphones of the head and torso simulator.
     - "Excitation" -- If your microphones (sensors) require excitation.

**3. Run VI (Ctrl+R) and Check Current Status**

The "3. Current Status" tab will show the following items:
- "Progress (%)" -- The proportion of tracks for which an appropriate gain has been found to calibrate it to the desired Leq,                                                       expressed as a percentage. 
- "Number of tracks to calibrate" -- Number of files to calibrate, as read by the VI from your input CSV file.
    
_Input parameters for current track:_
- All input parameters for the current track as read from the input CSV file headers<a href="#*">*</a>.</p>

*Results for current track:*
- "Adjusted initial guess" -- The gain after the first adjustment to the initial guess (denoted as G' in our paper).
- "Boundary search iteration (n)" -- The current iteration of the boundary search, or 0 if boundary search has not started.
- "Lower limit (L)" -- The lower limit found by the boundary search, displayed after it has completed.
- "Upper limit (U)" -- The upper limit found by the boundary search, displayed after it has completed.
- "Binary search iteration (n)" -- The current iteration of the binary search, or 0 if the binary search has not started.
- "Current gain (G)" -- If the binary search has started, this will be the gain applied to the current audio track to obtain 
                        the Leq value in "Current Leq (C)". Otherwise, this will be 0.
- "Current Leq (C)" -- If the binary search has started, this will be the Leq value measured by the VI when the current 
                       audio track is played back at the gain value in "Current gain (G)". Otherise, this will be 0.

*Final results for previous track:*
- "Final gain" -- The gain that needs to be applied to the previous audio track to calibrate it to the value in "Final Leq".
- "Final Leq" -- The Leq value measured by the VI when the audio track is played back at the gain value in "Final gain".
- "Total # of calls to MeasureLeq" -- The number of times that MeasureLeq was called for the previous track.

Except for "Final results for previous track", all the variables will be initialized to 0 at the end of each file and updated when called.

Footnotes
---------
<a name="*">
<p>*The input CSV file should contain a header row, followed by one row for every file that you want to calibrate. The header should contain the following columns in order:</p>

- "Filepath" -- The path to the audio track to be calibrated (absolute path or relative path to the CSV file's directory). 
- "Desired Leq (D)" -- The Leq that you wish to calibrate the current track to, in decibels.
- "Tolernce (T)" -- The tolerance of current track for the calibration, in decibels. T must be a positive real number.
                    The VI will output a value of the gain that calibrates the track to an Leq in the range [D-T, D+T].
- "Initial guess (G)" -- Your initial guess for the gain. G must be a positive real number.
- "Multiplier (M)" -- The multiplier for the boundary search, which controls the coarseness of the boundary search.
                      M must be a real number strictly greater than 1.
                      In general, a larger value will allow the boundary search to converge more quickly,
                      at the expense of a larger search space for the binary search.
- "Max. iterations for boundary search (N)" -- The maximum number of iterations allowed for the boundary search. The search
                                               is considered unsuccessful if the number of iterations exceeds this value.
- "Max. iterations for binary search (N)" -- The maximum number of iterations allowed for the binary search. The search
                                             is considered unsuccessful if the number of iterations exceeds this value.
<a name='**'>
<p>**The output CSV file will contain a header row, followed by one row for every non-header row in the input CSV file.
The header will contain the following columns in order:</p>

- "Filepath" -- The path to the audio track to be calibrated.
- "Gain" -- A positive real number specifying the gain that needs to be applied to the audio track to calibrate it to the
            value in "Leq" column. If the search is unsuccessful, the value will be -1.
- "Leq" -- The Leq value measured by the VI when the audio track is played back at the gain value in the "Gain" column.
- "Calls to MeasureLeq" -- The total number of times that the VI played back the audio track to obtain the value in the 
                           "Gain" column.
- "Boundary search iterations" -- The number of iterations taken for the boundary search for this audio track.
- "Binary search iterations" -- The number of iterations taken for the binary search for this audio track.
- "Within range" -- "Y" if the value in "gain" does not result in clipping of the audio track,
                     and "N" if it results in clipping.
- "Date" -- System date when the VI accomplished searching for the audio track located at the path in "Filepath".
- "Time" -- System time when the VI accomplished searching for the audio track located at the path in "Filepath".
If many values in the "Gain" column are -1 (indicating many unsuccessful searches), try increasing the value of N, M, or T, in that order. If many values in the "Within range" column are "N", try increasing the physical gain of your system (e.g. turn up your volumn knob).
<a name="***">
<p>***Note that values in the "Master Settings" panel will override those in the "XControl Settings" panel. If you encounter a "memory is full" error, try stopping the VI and running it again without changing any further settings. For further information, please refer to https://zone.ni.com/reference/en-XX/help/372416L-01/sndvibtk/daqmx_configuration/.</p>

Contact
-------
If you encounter any problems/bugs in the source code, please drop an email at <a href='mailto:bhanlam@ntu.edu.sg'> bhanlam@ntu.edu.sg</a> or <a href='mailto:wooi002@e.ntu.edu.sg'> wooi002@e.ntu.edu.sg</a>.
