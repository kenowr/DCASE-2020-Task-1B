import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa, librosa.display

__version__ = '20200615_dcase2020'

def audio_to_mel_magnitude_spectrogram(input_data = np.sin(np.linspace(0,440,44100)*np.pi), sr = None, n_fft = 2048, hop_length = 1024, center = False,
                                       n_mels = 48, fmin = 0, fmax = 24000, ref = 1.0,
                                       plot_spectrogram = False, titles = [], figsize = (20,4), fontsize = 20, vmin = None, vmax = None,
                                       **kwargs):
    '''
    ========
     Inputs
    ========
    
    input_data : str or np.ndarray
        If a string, it is the filepath of the input data (that
        will be read by sf.read).
        If it is an np.ndarray, it should be either a (n,)- or
        (c,n)-shaped array if it is single-channel or multi-
        channel respectively, where c is the number of channels in
        the signal and n is the number of samples in each channel.
        It will represent the signal in floating point numbers
        between -1 and 1. This function will convert a (n,)-
        shaped array to a (1,n)-shaped array while running.
        Default is a one-second 440Hz sine tone sampled at 44100Hz.
    
    sr : int
        The sampling rate of the signal specified by input_data.
        If it is not specified, then the native sampling rate of
        the file in input_data will be used (if input_data is a
        string) or a default sampling rate of 44100Hz will be used
        (if input_data is an np.ndarray).
    
    n_fft : int
        The number of samples in each time window for the
        calculation of the STFT of the input signal using
        librosa.core.stft.
        
    hop_length : int
        The number of overlapping samples in each time window for
        the calculation of the STFT of the input signal using
        librosa.core.stft.
    
    center : bool
        If True, centers the window at the current time index of
        the signal before performing FFT on it. If False, does not
        center the window, i.e. the current time index IS the 
        first index of the window. This is as per the parameter in
        librosa.core.stft.
        
    n_mels : int
        The number of mel bands used for calculation of the log-
        frequency power spectrom by librosa.feature.melspectrogram
    
    fmin : float
        The minimum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    
    fmax : float
        The maximum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    
    ref : float or callable
        The reference value or a function which returns the
        reference value for which dB computation is done in
        librosa.power_to_db.
    
    plot_spectrogram : bool
        If True, plots one spectrogram per channel in the signal
        specified in input_data. Otherwise, plots nothing.
    
    titles : list of str
        A list of c strings, where c is the number of channels in
        input_data, corresponding to the title of each mel
        spectrogram that is plotted. A default string will be
        assigned to each spectrogram if titles is an empty list.
        
    figsize : tuple of 2 int values
        If plot == True, this is the size of the spectrogram plot.
        If plot == False, this argument is ignored.
        
    fontsize : int
        The font size for the title of the plots, if plot == True.
        The font size of other elements in the plot will be 
        resized relative to this number. If plot == False, this
        argument is ignored.
    
    vmin : float or None
        The minimum value of the colour bar legend in the
        plt.clim() call. If None, the minimum of the input
        spectrogram values is used.
    
    vmax : float or None
        The maximum value of the colour bar legend in the
        plt.clim() call. If None, the maximum of the input
        spectrogram values is used.

    **kwargs : dict
        Extra keyword arguments to control the plot formatting
        with librosa.display.specshow (e.g. x_axis = 's',
        y_axis = 'mel', etc.)

    =========
     Outputs
    =========
    
    mel_spectrograms : np.ndarray
        An (n_mels, t, c)-shaped array containing the mel-
        spectrograms of each channel, where n_mels is the number
        of mel bands specified in the input argument to this
        function, t is the number of time bins in the STFT, and
        c is the number of channels in input_data.
        t depends on the input argument center as follows:
        If center is True, then t = np.ceil(n/hop_length).
        If center is False, then t = np.floor(n/hop_length)-1

    ==============
     Dependencies
    ==============
    
    librosa, librosa.display, matplotlib.pyplot (as plt),
    numpy (as np), soundfile (as sf)
    
    Written by Kenneth Ooi.
    '''
    ## EXCEPTION HANDLING

    if type(input_data) == str: # If the input data entered is a string,...
        input_data, native_sr = sf.read(input_data) # ...then we read the filename specified in the string.
        input_data = np.transpose(input_data) # Transpose the input data to fit the (c,n)-shape desired.
        sr = native_sr if sr == None else sr
    elif type(input_data) == np.ndarray: # else we assume it is an np.ndarray
        sr = 44100 if sr == None else sr
    else:
        print('Invalid input data type! Input data must either be a string or a numpy.ndarray. Program terminating.')
        return


    # At this point, input_data should be either a (n,)- or (c,n)-shaped array.
    if len(input_data.shape) == 1: # If it's a (n,)-shaped array,...
        input_data = np.expand_dims(input_data,0) # ...then convert it into a (1,n)-shaped array.
    
    if len(titles) == 0:
        titles = ['Log-frequency power spectrogram (channel {:d})'.format(i) for i in range(input_data.shape[0])] # The default title for each spectrogram just iterates on the channel number.
    elif input_data.shape[0] != len(titles):
        print('Number of titles does not match number of channels in input data! Program terminating.')
        return
    
    if input_data.shape[0] > input_data.shape[1]:
        print('Warning: The input data appears to have more channels than samples. Perhaps you intended to input its transpose?')

    ## CALCULATE MEL SPECTROGRAM OF FIRST CHANNEL

    # Firstly, calculate the short-time Fourier transform (STFT) of the signal with librosa.core.stft.
    # We typecast input_data[0] as a Fortran-contiguous array because librosa.core.stft does vectorised operations on it,
    # and numpy array slices are typically not Fortran-contiguous.
    input_stft = librosa.core.stft(y = np.asfortranarray(input_data[0]), n_fft = n_fft, hop_length = hop_length, center = center)

    # Then, calculate the mel magnitude spectrogram of the STFT'd signal with librosa.feature.melspectrogram.
    power_spectrogram = librosa.feature.melspectrogram(S = np.abs(input_stft)**2, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft, fmin = fmin, fmax = fmax)

    # Convert the power spectrogram into into units of decibels with librosa.power_to_db.
    mel_spectrogram = librosa.power_to_db(power_spectrogram, ref = ref)

    # mel_spectrogram is an np.array. We typecast all elements to np.float32 to ensure all output data types match.
    mel_spectrogram = mel_spectrogram.astype(np.float32)


    ## PLOT MEL SPECTROGRAM OF FIRST CHANNEL

    if plot_spectrogram:
        plt.figure(figsize = figsize) # Sets size of plot.
        librosa.display.specshow(mel_spectrogram, sr = sr, hop_length = hop_length, fmin = fmin, fmax = fmax, **kwargs)
        plt.title(titles[0], fontsize = fontsize)         # Set title of figure.
        plt.xlabel('Time/s', fontsize = fontsize)         # Add in x-axis label to graph.
        plt.xticks(fontsize = 0.7*fontsize)               # Set font size for x-axis ticks (i.e. the numbers at each grid line).
        plt.ylabel('Frequency/Hz', fontsize = fontsize)   # Add in y-axis label to graph.
        plt.yticks(fontsize = 0.7*fontsize)               # Set font size for y-axis ticks (i.e. the numbers at each grid line).
        plt.colorbar(format='%+3.1f dB')                  # Adds in colour bar (legend) for values in spectrogram.
        plt.clim(vmin = vmin,vmax = vmax)                 # Defines the colour bar limits.
        plt.show()                                        # Display the actual plot on the IPython console.


    ## INITIALISE OUTPUT ARRAY (3-DIMENSIONAL) BASED ON FIRST CHANNEL MEL SPECTROGRAM SHAPE

    mel_spectrograms = np.zeros((mel_spectrogram.shape[0],mel_spectrogram.shape[1],input_data.shape[0]))
    mel_spectrograms[:,:,0] = mel_spectrogram # Put the first channel mel spectrogram in first.


    ## CALCULATE AND PLOT MEL SPECTROGRAM OF OTHER CHANNELS, IF ANY

    for i in range(1,input_data.shape[0]): # for i in 1:(number of channels),...
        input_stft = librosa.core.stft(y = np.asfortranarray(input_data[i]), n_fft = n_fft, hop_length = hop_length, center = center)
        power_spectrogram = librosa.feature.melspectrogram(S = np.abs(input_stft)**2, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft, fmin = fmin, fmax = fmax)
        mel_spectrogram = librosa.power_to_db(power_spectrogram, ref = ref)
        mel_spectrogram = mel_spectrogram.astype(np.float32)
        mel_spectrograms[:,:,i] = mel_spectrogram # Put the calculated mel spectrogram as that of the ith channel.

        if plot_spectrogram:
            plt.figure(figsize = figsize) # Sets size of plot.
            librosa.display.specshow(mel_spectrogram, sr = sr, hop_length = hop_length, fmin = fmin, fmax = fmax, **kwargs)
            plt.title(titles[i], fontsize = fontsize)  # Set title of figure.
            plt.xlabel('Time/s', fontsize = fontsize)         # Add in x-axis label to graph.
            plt.xticks(fontsize = 0.7*fontsize)               # Set font size for x-axis ticks (i.e. the numbers at each grid line).
            plt.ylabel('Frequency/Hz', fontsize = fontsize)   # Add in y-axis label to graph.
            plt.yticks(fontsize = 0.7*fontsize)               # Set font size for y-axis ticks (i.e. the numbers at each grid line).
            plt.colorbar(format='%+3.1f dB')                  # Adds in colour bar (legend) for values in spectrogram.
            plt.clim(vmin = vmin,vmax = vmax)                 # Defines the colour bar limits.
            plt.show()                                        # Display the actual plot on the IPython console.

    return mel_spectrograms