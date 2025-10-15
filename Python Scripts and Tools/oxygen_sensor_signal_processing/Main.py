import glob
import pandas as pd
import Process_signal
import Conditions
import Show_result
import numpy as np

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from scipy import signal

path = glob.glob('../../Dataset/B2-2-10/*.DTA')
for file in path:
    # READ FILES
    datafile = open(file)
    lines = datafile.readlines()

    # EXTRACT AXIS LABEL FROM FILE NAMES
    label_text = file.split("VISHAL-")[1]
    label_text = label_text.split((".DTA"))[0]
    print(label_text)
    # EXTRACT VOLTAGE AND CURRENT DATA FROM THE FILE
    count = 84
    for i in lines[84:len(lines)]:
        lines[count] = i.split()
        count += 1
    df = pd.DataFrame(lines[85:len(lines)], columns=[0, 1, 'voltage', 'current', 4, 5, 6, 7, 8, 9, 10])
    df = df[['voltage', 'current']]
    df['voltage'] = df['voltage'].astype(float)
    df['current'] = df['current'].astype(float)
    df['current'] = df['current'].apply(lambda x: x * 1000000)
    x = df['voltage']
    y = df['current']
    list_x = list(x)
    list_y = list(y)

    # COMPUTE DERIVATIVE FOR RAW SIGNAL
    raw_d1 = []
    for i in range(len(list_x)-1):
        raw_d1.append((list_y[i + 1] - list_y[i]) / (list_x[i + 1] - list_x[i]))
    raw_d2 = []
    for i in range(len(raw_d1) - 1):
        raw_d2.append((raw_d1[i + 1] - raw_d1[i]) / (list_x[i + 1] - list_x[i]))

    # PAD THE ARRAYS WITH FINAL VALUES TO EQUALIZE LENGTH
    raw_d1.append(raw_d1[-1])
    raw_d2.append(raw_d2[-1])
    raw_d2.append(raw_d2[-1])

    # APPLY SAVITSKY-GOLAY TO RAW SIGNAL
    savgol_list = Process_signal.optimize_savgol(list_y)
    opt_window_size1 = savgol_list[1]
    opt_order1 = savgol_list[2]
    list_y_savgol_filtered = signal.savgol_filter(list_y, 53, 3), # savgol_filter(data, window size, order of polynomial)

    # TODO: APPLY FFT-LPF TO RAW SIGNAL
    list_y_lowpass_filtered = Process_signal.fft_lowpass(list_y)


    # COMPUTE FILTERED DERIVATIVE
    filtered_d1 = []
    for i in range(len(x) - 1):
        filtered_d1.append((list_y_savgol_filtered[0][i + 1] - list_y_savgol_filtered[0][i]) / (list_x[i + 1] - list_x[i]))
    filtered_d1.append(filtered_d1[len(filtered_d1) - 1])  # concatenate last value for uniform data length

    filtered_d2 = []
    for i in range(len(x)-1):
        filtered_d2.append((filtered_d1[i+1] - filtered_d1[i]) / (list_x[i+1] - list_x[i]))
    filtered_d2.append(filtered_d2[len(filtered_d2) - 1])

    # TODO: (OPTIONAL) SAVE THE PLOTS ON DISK

    # TODO: Apply Conditions
    position_vector = [0] * len(list_x)

    # applying voltage range condition
    # position_vector = Conditions.voltage_range(position_vector=position_vector, list_x=list_x)
    # applying derivative range condition
    position_vector = Conditions.derivative_range(position_vector=position_vector, d1_data=filtered_d1)
    # applying filtered derivative
    #position_vector = Conditions.maximum_yvalue(position_vector=position_vector, y_data=list_y)


    #print(list_x)
    #print(position_vector)

    # TODO: PINPOINT RESULT AFTER CONDITIONS
    # At this stage, the index of the value reflecting the oxygen level is specified
    list_possible_predictions = []
    ideal_score = 0
    ideal_position = 0

    # find the highest score
    for index, element in enumerate(position_vector):
        if element > ideal_score:
            ideal_score = element

    # find the position of the highest score
    for index, element in enumerate(position_vector):
        if element == ideal_score:
            list_possible_predictions.append([list_x[index], list_y[index]])

    Show_result.plot_result_with_level_prediction(label_text=label_text, list_x=list_x, list_y=list_y, list_possible_predictions=list_possible_predictions)















