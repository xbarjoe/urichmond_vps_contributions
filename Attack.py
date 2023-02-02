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
from playsound import playsound


#replication of hadi's code
location="LibriSpeech/files/wavfiles/"
old=r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/"+location
new=r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/"+location+"Test/test_"

def bsHadi(my_file):
    fs, data = scipy.io.wavfile.read(old+my_file)
    if transcribe(old+my_file,'google') == "nothing":
        print("could not transcribe "+my_file)
        return None
    else:
        print("Sample: Good")
        fftdata = np.fft.rfft(data)
        sortfft = []
        for i in range(0,len(fftdata)):
            sortfft.append(abs(fftdata[i]))
        sort=sorted(sortfft)
        newn = .5
        oldn = 0
        return bsHadiHelper(data,fftdata,sort,oldn,newn)
def bsHadiHelper(my_file,fft,sort,oldn,newn):
    #if the difference bettween the new proportion of coefs and old proportion is sufficently small
    if abs(oldn-newn)<=.0001:
        print("Done")
        print("orignal")
        print(transcribe(old+my_file,'google'))
        print("modified")
        print(transcribe(new+my_file,'google'))
        return None
    else:
        cutoff = newn*len(sort)
        tempfft=fft
        for i in range(0,cutoff):
            for j in range(0,len(fft)):
                coef = abs(tempfft[j])
                if coef == sort[i]:
                    tempfft[j] = 0
                print("Percent Done: "+str(i*100/cutoff)+"%")
        newdata = np.fft.ifft(tempfft,len(my_data))
        hello=np.asarray(newdata)
        oldfile=old+my_file
        newfile=new+my_file
        scipy.io.wavfile.write(newfile, fs, hello.astype(data.dtype))
        if transcribe(oldfile,'google') == transcribe(newfile,'google'):
            print("MATCHED TRANSCRIPTION, TRYING LARGER")
            return bsHadiHelper(my_file,fft,sort,newn,(newn+abs(.5-newn)))
        else:
            print("TRANSCRIPTION DIFFERENT, TRYING SMALLER")
            return bsHadiHelper(my_file,fft,sort,newn,(newn-abs(.5-newn)))
    
                

#Batch for samples that have already been modified
def batch_transcribe(path):
    df = pd.DataFrame(columns=["Sample","Original","Modified","Result"])
    directory = os.fsencode(path).decode('UTF-8')
    print(directory)
    curindex=0
    numSamples=6221
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            curindex=curindex+1
            original=transcribe(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/BulkWords/"+filename, "google")
            perturbed=transcribe(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/BulkWords/(#FOLDER HERE#)/"+"test_"+filename, "google")
            print("File: "+filename+"               "+"Results: "+original+" / "+perturbed+"          "+"%: "+str((curindex/numSamples)*100))
            if original!= "nothing":
                if perturbed == "nothing":
                    #Original WAS transcribed properly, Modified was NOT
                    result="Perturbation"
                elif perturbed != "nothing" and perturbed != original:
                    result="Mistranscription"
                    #Original WAS transcribed properly, Modified transcribed to something DIFFERENT
                else:
                    result = "No Effect"
                    #Original transcription = Modified transcription
            elif original=="nothing":
                if perturbed != "nothing":
                    result="Fixed"
                    #Original WAS NOT transcribed properly, Modified WAS
                else:
                    result="Fail"
                    #Original and Modified both failed to transcribe
            df.loc[len(df)] = [filename,original,perturbed,result]
            #Creating and saving the results as a dataTable/csv
    df.to_csv(r'/Users/stephenowen/Desktop/2019RESEARCH/SR2019/Results.csv')
    return df
        
        
def batch(path):
    #Calls rolling_average() on a directory of samples and returns a data frame similar to batch_transcribe()
    df = pd.DataFrame(columns=["Sample","Original","Modified","Result"])
    directory = os.fsencode(path).decode('UTF-8')
    results = list()
    print(directory)
    curIndex=0
    numSamples=len(os.listdir(directory))
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            temp=rolling_average(filename,curIndex,numSamples/files/wavfiles)
            curIndex=curIndex+1
            if temp is not None:
                 df.loc[len(df)] = temp
    df.to_csv(r'/Users/stephenowen/Desktop/2019RESEARCH/SR2019/55Smooth_Results.csv')
    return results
            

def transcribe(my_path,model):
    wit_key = ''

    AUDIO_FILE =  path.join(my_path)

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(my_path) as source:
        audio = r.record(source)  # read the entire audio file
    
    #attempt transcription with google
    if(model == 'google'):
        # Using Google's Algorithm 
        try:
            #Google CAN transcribe the Audio
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            #Google CAN'T transcribe the audio
            return "nothing"

        except sr.RequestError as e:
            print("Google error; {0}".format(e))
    
#Rolling Average Modification (~23% Success)
def rolling_average(my_file,curindex,mysize):
    #Reading in the .wav file
    fs, data = scipy.io.wavfile.read(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/LibriSpeech/files/wavfile"+my_file)

    elementsInBucket = 23
    n = int(len(data)/elementsInBucket)

    #Breaks array into buckets of elements
    def createBuckets(arr, n):
        length = len(arr)
        return [ arr[i*length // n: (i+1)*length // n] 
                for i in range(n) ]


    arr = np.copy(data)
    size=len(arr)
    splitArray = createBuckets(arr,n)
    l = list()
    #Creating an array representing the fourier transform of the audio sample
    fftarr = np.fft.rfft(arr)

    index = 0
    #Creates another array B of the same length as the original array A, where B[i]=(A[i]+A[i-1])/2 
    while index < len(fftarr):
        if index>5 and index<len(fftarr)-5:
            l.append((fftarr[index-4]+fftarr[index-3]+fftarr[index-2]+fftarr[index-1]+fftarr[index]+fftarr[index+1]+fftarr[index+2]+fftarr[index+3]+fftarr[index+4])/10)
            index=index+1
        else:
            l.append(fftarr[index])
            index=index+1
    
    modarr = np.asarray(l)
    #Creates a modified audio file by taking the inverse fourier transform
    backarr = np.fft.ifft(modarr,size)

    
    hello=np.asarray(backarr)
    #Writes Modified file to disk
    scipy.io.wavfile.write(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/LibriSpeech/files/wavfiles/Test/"+"test_"+my_file, fs, hello.astype(data.dtype))
    #Transcribes both the original and the modified sample using google
    sample = AudioSegment.from_wav(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/LibriSpeech/files/wavfiles/Test/"+"test_"+my_file)
    sample = sample+13
    sample.export(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/LibriSpeech/files/wavfiles/Test/"+"test_"+my_file, format="wav")
    
    original=transcribe(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/LibriSpeech/files/wavfiles/"+my_file, "google")
    perturbed=transcribe(r"/Users/stephenowen/Desktop/2019RESEARCH/SR2019/LibriSpeech/files/wavfiles/Test/"+"test_"+my_file, "google")

    original=transcribe(r"/Users/mw7zd/Desktop/SummerResearch/StephenCode/SR2019/BulkWords/"+my_file, "google")
    perturbed=transcribe(r"/Users/mw7zd/Desktop/SummerResearch/StephenCode/SR2019/BulkWords/22smooth/"+"test_"+my_file, "google")

    sample = AudioSegment.from_wav(r"/Users/mw7zd/Desktop/SummerResearch/StephenCode/SR2019/BulkWords/22smooth/"+"test_"+my_file)
    sample = sample + 9
    sample.export(r"/Users/mw7zd/Desktop/SummerResearch/StephenCode/SR2019/BulkWords/22smooth/"+"test_"+my_file, format = "wav")

    #progress meter for batch file processing
    print("File: "+my_file+"               "+"Results: "+original+" / "+perturbed+"          "+"%: "+str((curindex/mysize)*100))
        
    if original!= "nothing":
            if perturbed == "nothing":
                #Original WAS transcribed properly, Modified was NOT
                return [my_file,original,perturbed,"Perturbation"]
            elif perturbed != "nothing" and perturbed != original:
                return [my_file,original,perturbed,"Mistranscription"]
                #Original WAS transcribed properly, Modified transcribed to something DIFFERENT
            else:
                return [my_file,original,perturbed,"No Effect"]
                #Original transcription = Modified transcription
    elif original=="nothing":
            if perturbed != "nothing":
                return [my_file,original,perturbed,"Fixed"]
                #Original WAS NOT transcribed properly, Modified WAS
            else:
                return [my_file,original,perturbed,"Fail"]
                #Original and Modified both failed to transcribe
                