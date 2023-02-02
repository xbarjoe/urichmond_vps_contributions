import speech_recognition as sr
from os import path
import os.path
import operator
from os import system
from os import listdir
from os.path import isfile, join
import wave
import scipy as sc
import librosa
import IPython.display as ipd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import math
#import librosa as lb
import scipy
from sklearn.decomposition import PCA
import pandas as pd
from os import listdir
from os.path import isfile, join
import time
from itertools import product
import datetime
import sys
import pandas as pd
from pydub import AudioSegment
import soundfile as sf
import scipy.fftpack as trans

def output(file):
    fs, data = scipy.io.wavfile.read(file)
    plt.figure(figsize=(20,10))
    plt.plot(data)
    plt.show()
    dct = trans.dct(data)
    plt.figure(figsize=(20,10))
    plt.plot(dct)
    plt.show()
    
def ccrowd(file):
    fs, data = scipy.io.wavfile.read(file)
    plt.plot(data)
    dct = trans.dct(data,norm='ortho')
    
    plt.plot(dct)
    
    index=len(file)-4
    tempfile = file[0:index]
    scipy.io.wavfile.write(tempfile+"_dct.wav", fs, dct.astype(np.int16))
    
    plt.show()
    
    fs2, data2 = scipy.io.wavfile.read(tempfile+"_dct.wav")
    
    data3 = sorted(data2)
    high=data3[len(data2)-1000:]
    low=data3[:1000]
    data1000 = list()
    for i in range(0,len(data2)):
        if data2[i] in high or data2[i] in low:
            data1000.append(0)
        else:
            data1000.append(data2[i])
    dataOneMillion = np.asarray(data1000)
    idct = trans.idct(dataOneMillion,n=len(data),norm='ortho')
    #plt.plot(data2)
    scipy.io.wavfile.write(tempfile+"_dct2.wav", fs, idct.astype(np.int16))
    
def scrowd(file):
    fs, data = scipy.io.wavfile.read(file)
    plt.plot(data)
    dst = trans.dst(data,norm='ortho',type=2)
    
    plt.plot(dst)
    
    index=len(file)-4
    tempfile = file[0:index]
    scipy.io.wavfile.write(tempfile+"_dst.wav", fs, dst.astype(np.int16))
    
    plt.show()
    
    fs2, data2 = scipy.io.wavfile.read(tempfile+"_dst.wav")
    
    data3 = sorted(data2)
    high=data3[len(data2)-1000:]
    low=data3[:1000]
    data1000 = list()
    for i in range(1,len(data2)):
        if data2[i] in high or data2[i] in low:
            data1000.append(0)
        else:
            data1000.append(data2[i])
    dataOneMillion = np.asarray(data1000)
    idst = trans.dst(dataOneMillion,n=len(data),norm='ortho',type=3)
    #plt.plot(data2)
    scipy.io.wavfile.write(tempfile+"_dst2.wav", fs, idst.astype(np.int16))
def run2(file):
    fs, data = scipy.io.wavfile.read(file)
    
    data2 = data.copy()
    fftdata = trans.dct(data2)
    freqs = fftdata.copy()
    for i in range(1,len(fftdata)):
        temp = fftdata[i]
        fftdata[i]=fftdata[i]**2
        #print("Replaced "+str(temp)+" with "+str(fftdata[i]))
    data3 = sorted(fftdata)
    
    low = data3[0:100]
    newfile = list()
    newfile.append(fftdata[0])
    plt.plot(fftdata)
    plt.show()
    for i in range(1,len(freqs)):
        if freqs[i]**2 in low:
            #print("Appended 0")
            newfile.append(0)
        else:
            newfile.append(freqs[i])
            #print("Appended "+str(freqs[i]))
    newdata=np.asarray(newfile,dtype=np.int16)
    plt.plot(newdata)
    tempdata=trans.idct(newdata,n=len(data))
    
    index=len(file)-4
    tempfile = file[0:index]
    scipy.io.wavfile.write(tempfile+"_fft.wav", fs, tempdata.astype(np.int16))
    
    
def run(file):
    #Reads file and appends 0 until sample is an even divisor of 825 (25ms)
    quicktemp = list()
    fs, data = scipy.io.wavfile.read(file)
    size = len(data)
    if len(data)//825!=0:
        for i in range(0,len(data)):
            #print("made it to index "+str(i)+" in data")
            quicktemp.append(data[i])
        while size > 825:
            print(size)
            size = size - 825
        for i in range(0,size):
            quicktemp.append(0)
        data=np.asarray(quicktemp)
    print(data)
    
    numBins = np.ceil(len(data)/825)
    
    startBin = 0
    
    endBin = 825
    
    #applies the attack on each window
    while(startBin<len(data)):
        #creates a 25ms long window
        tempBin=data[startBin:endBin]
        
        #computes the rfft
        fft = np.fft.rfft(tempBin)
        print(fft)
        #square the fft
        for i in range(0,len(fft)):
            temp = fft[i]
            fft[i]=fft[i]**2
            
        #sort the fft coefficents 
        sorbet = fft.copy()
        for i in range(0,len(sorbet)-1):
            for j in range(0,len(sorbet)-i):
                if sorbet[i]>sorbet[i+1]:
                    temp = sorbet[i]
                    sorbet[i] = sorbet[i+1]
                    sorbet[i+1]=temp
                    
        #Zero a value if it is one of the nth smallest fft coefficents 
        blankarr=list()
        for i in range(0,len(fft)):
            for j in range(0,100):
                if fft[i] == sorbet[j]:
                    #print("Found fft coefficent "+str(sorbet[j])+" at index "+str(i))
                    blankarr.append(0)
                    
                else:
                    blankarr.append(fft[i])
                 
        
        modified=np.asarray(blankarr)        
        ifft = np.fft.ifft(modified,n=825)
        
        new = list()
        #build the array back up
        for i in range(0,startBin):
            new.append(data[i])
        for j in range(0,len(ifft)):
            new.append(ifft[j])
        for k in range(endBin,len(data)):
            new.append(data[k])
        #update values
        newdata = np.asarray(new,dtype=np.int16)
        startBin+=725
        endBin+=825
        print("Completed "+str((startBin/len(data))*100)+"%")
    #output
    index=len(file)-4
    tempfile = file[0:index]
    scipy.io.wavfile.write(tempfile+"_modified.wav", fs, newdata.astype(np.int16))

    return newdata