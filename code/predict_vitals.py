import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import find_peaks

def hear_rate(peaklist,fs):
    RR_list = []
    cnt = 0
    
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
        RR_list.append(ms_dist) #Append to list
        cnt += 1
    
    bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal
    print ("Average Heart Beat is: %.01f" %bpm) #Round off to 1 decimal and print



def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = args.batch_size
    sample_data_path = args.video_path
    distance=args.distance
    dXsub,fs = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))
    
    ########## calculate peaks ####################
    peaks, _ = find_peaks(pulse_pred, distance=distance)
    hear_rate(peaks,fs)
    ########## Plot ##################

    plt.subplot(211)
    plt.plot(pulse_pred) 
    plt.plot(peaks, pulse_pred[peaks], "x")
    plt.title('Pulse Prediction')
    plt.subplot(212)
    plt.plot(resp_pred)
    plt.title('Resp Prediction')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--distance', type=int, default = 15, help='distance range from 10-30')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)

