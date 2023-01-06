import numpy as np

def create_concatenated_sines(frequency_list, total_time, number_of_samples):
    '''
    This function returns a series of sine functions concatenated
    
    Paramaters:
    frequency_list: list of frequencies (in Hz)
    total_time: total duration of the signal (in sec)
    number_of_samples: number of time points in the signal
    '''
    period = total_time / number_of_samples # sampling period
    sampling_frequency = 1/period # sampling frequency

    # time arrays
    time_array = np.linspace(0, total_time/len(frequency_list), num=int(number_of_samples/len(frequency_list)))
    # xb = np.linspace(0, t_n/4, num=int(N/4))

    amplitude_list = []
    for i, frequency in enumerate(frequency_list):
        amplitude_list += [np.sin(2*np.pi*frequency*time_array)]
    # signal composed as concatenation of sines
    composite_signal = np.concatenate(amplitude_list)
    return time_array, composite_signal

def create_mixed_sines(frequency_list, total_time, number_of_samples):
    '''
    This function returns a linear sum of sine functions
    
    Paramaters:
    frequency_list: list of frequencies (in Hz)
    total_time: total duration of the signal (in sec)
    number_of_samples: number of time points in the signal
    '''
    period = total_time / number_of_samples # sampling period
    sampling_frequency = 1/period # sampling frequency

    # time arrays
    time_array = np.linspace(0, total_time, num=number_of_samples)
    # xb = np.linspace(0, t_n/4, num=int(N/4))

    # frequencies used to build signals (in Hz)
    mixed_signal = np.zeros(time_array.shape)
    for i, frequency in enumerate(frequency_list):
        mixed_signal += np.sin(2*np.pi*frequency*time_array)
        
    return time_array, mixed_signal

def create_CAP_signal(total_time, sampling_frequency, time_shift):
    '''
    This function returns a cardiac action potential signal (CAP).
    Obtained from the output of an online model of ventricular action potential Williams et al., 2015.
    Parameters chosen to generate this signal were: “Mahajan et al. (2008) rabbit ventricular cell model”, 0.5Hz pacing frequency, 1min maximum pacing time and “0μM compound concentration”.
    
    Paramaters:
    total_time: total duration of the signal (in sec)
    sampling_frequency: in Hz (max 10kHz)
    time_shift: time point when signal should start (in sec)
    '''
    file_path = 'datasets/AP.txt'
    
    n_samples = int((sampling_frequency*total_time) + 1) #number of samples
    time_array = np.linspace(0, total_time, n_samples) #time vector

    AP_original = np.loadtxt(file_path,delimiter='\t')

    AP_original[:,0] = AP_original[:,0]/1000                       #ms to sec
    xvals = np.arange(AP_original[0,0], AP_original[-1,0], 0.0001) #generate a time vetor with a constant sampling frequency
    yinterp = np.interp(xvals, AP_original[:,0], AP_original[:,1]) #interpolate values where original AP time data is missing
    idx_shift = np.argmin(abs(time_array-time_shift))
    yinterp2 = np.interp(time_array, xvals, yinterp)   #interpolate (subsample) data to experimental video sampling frequency   
    signal = np.roll(yinterp2,idx_shift)                  #shift AP onset from 0 to 0.5sec
    signal = signal+abs(signal[-1])                          #shift amp from ~-80 to 0
    signal = signal/np.amax(signal)                          #normalize amplitude
    max_idx = np.argmax(abs(signal))              #index of maximal absolute amplitude
    return time_array, signal