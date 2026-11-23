from scipy.signal import butter, filtfilt


def bp_filter (Fs, hf, bf, ordre, xb) :
    filtre = butter(ordre, [bf, hf], btype='bandpass', fs=Fs, output='ba')
    signal_filtre = filtfilt(filtre[0], filtre[1], xb)
    
    return(signal_filtre)
    

    