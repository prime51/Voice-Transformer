# -*- coding: utf-8 -*-
import sys
import wave
import numpy as np
import pylab as plt
import seaborn
import os
def printWAV(wave_path):
    """
    function:
        print the spectral graph of a wave file

    parameter:
        wave_path: the path of wave file
    """
    f = wave.open(wave_path, "rb")
    params = f.getparams()  
    nchannels, sampwidth, framerate, nframes = params[:4]
    # print(nchannels, sampwidth, framerate, nframes)
    str_data  = f.readframes(nframes)  
    f.close()
    wave_data = np.fromstring(str_data, dtype = np.short)
    time=np.arange(0, nframes) * (1.0 / framerate)
    # print(wave_data.shape, time.shape)
    if nchannels == 2:
        if(wave_data.shape[0] % 2 != 0):
            wave_data = np.delete(wave_data, wave_data.shape[0] - 1, axis=0)
        wave_data.shape = -1, 2
        wave_data = wave_data.T
        # print(wave_data.shape, time.shape)
        plt.figure(1,figsize=(13,8)) 
        plt.subplot(2, 1, 1)
        plt.plot(time, wave_data[0])
        plt.title('left')
        plt.subplot(2, 1, 2)  
        plt.plot(time, wave_data[1], c="r")  
        plt.title('right')
        plt.xlabel("time")

    elif nchannels == 1:
        # print(wave_data.shape, time.shape)
        plt.figure(1,figsize=(13,8))
        plt.plot(time, wave_data)
        plt.xlabel("time")
    save_path = wave_path.replace('audio','images').replace('wav','jpg')
    plt.savefig(save_path)

filename = sys.argv[1]
printWAV(filename)