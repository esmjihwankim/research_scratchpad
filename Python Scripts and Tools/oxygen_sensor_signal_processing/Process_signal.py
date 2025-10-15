'''
Savitsky-Golay filter
'''

from scipy import fftpack
from scipy import signal
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import diff
import matplotlib.pyplot as plt

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff


def optimize_savgol(dev):
    error = 100
    opt_window_size = 10
    opt_order = 1
    # i is the window size used for filtering
    # j is the order of fitted polynomial
    for i in range(11,100):
        for j in range(1,10):
            devs = signal.savgol_filter(dev, i, j)
            devs = np.squeeze(np.asarray(devs))
            error_try = mean_squared_error(dev, devs)
            if error > error_try:
                if np.mean(np.absolute(diff(devs))) < 0.0001:
                    error = error_try
                    opt_window_size = i
                    opt_order = j
    return [error, opt_window_size, opt_order]


def optimize_original_savgol(dev):
    error = 100
    opt_window_size = 10
    opt_order = 1
    for i in range (11,100):
        for j in range(1,10):
            devs = signal.savgol_filter(dev, i, j)
            devs = np.squeeze(np.asarray(devs))
            error_try = mean_squared_error(dev, devs)
            if error > error_try:
                if np.mean(np.absolute(diff(devs))) < 0.0001:
                    error = error_try
                    opt_window_size = i
                    opt_order = j
    return [error, opt_window_size, opt_order]


# Fourier transform
def fft_lowpass(data):
    signal = data
    sample_rate = 0.2
    N = len(signal)
    freq_axis = fftfreq(N, d=0.2)
    normalize = N/2
    fft_signal = fft(signal)
    norm_amplitude = np.abs(fft_signal)/normalize

    plt.plot(freq_axis, norm_amplitude)
    plt.xlabel('Frequency[Hz]')
    plt.title('Spectrum')
    plt.show()



"""
## EXAMPLE
Fs = 1000
T = 1/Fs
end_time = 1
time = np.linspace(0, end_time, Fs)
amp = [2, 1, 0.5, 0.2]
freq = [10, 20, 30, 40]

signal_1 = amp[0] * np.sin(freq[0]*2*np.pi*time)
signal_2 = amp[1] * np.sin(freq[1]*2*np.pi*time)
signal_3 = amp[2] * np.sin(freq[2]*2*np.pi*time)
signal_4 = amp[3] * np.sin(freq[3]*2*np.pi*time)

signal = signal_1 + signal_2 + signal_3 + signal_4

s_fft = np.fft.fft(signal)
amplitude = abs(s_fft) * (2/len(s_fft))
frequency = np.fft.fftfreq(len(s_fft), T)

plt.xlim(0, 50)
plt.stem(frequency, amplitude)
plt.grid(True)
plt.show()
"""