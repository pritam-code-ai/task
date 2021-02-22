# Libraries
# !pip install audio2numpy
# !pip install xgboost
# !pip install pydub 
# !pip install ffmpeg 

import xgboost as xgb
from pydub import AudioSegment
from pydub.utils import which
import pandas as pd
import os
import numpy as np
# from datetime import datetime, timedelta
import datetime as dt
import random
from audio2numpy import open_audio
import pickle 
from math import floor
from sklearn.neighbors import NearestNeighbors
pickle_file_name = "signal_sampling_rate_list.pkl"
NearestNeighbors_model_name = "NearestNeighbors.pkl"
path_total_audio_file = "./sounds/samples/vi95kMQ65UeU7K1wae12D1GUeXd2/"

root_path = "./"

def build_folder(path1):
  path_to_folder = path1
  if not os.path.exists(path_to_folder):
      os.makedirs(path_to_folder)
  return path_to_folder

build_folder(root_path)

# Read in relevant data files
samples_df = pd.read_csv(root_path+'samples_short.csv')
ground_truth = pd.read_csv(root_path+'ground_truth_short.csv')
perfect = pd.read_csv(root_path+'perfect.csv')

directory_of_sounds = root_path+'sounds/samples/'
os.path.isdir(directory_of_sounds)
len(os.listdir(directory_of_sounds + '/vi95kMQ65UeU7K1wae12D1GUeXd2')) # should be 22

print(samples_df)

print(perfect)

def create_new_audio_file(file_path, file_name, event_occur_time_start):
  AudioSegment.converter = which("ffmpeg")
  t1 = (event_occur_time_start+1.1) * 1000 #Works in milliseconds
  t2 = (event_occur_time_start +2.4) * 1000
  newAudio = AudioSegment.from_file(file_path, format="mp4")
  newAudio = newAudio[t1:t2]
  print(path_to_event_audio+file_name.split(".")[0]+".wav")
  newAudio.export(path_to_event_audio+file_name.split(".")[0]+".wav", format="wav")
  print("file_built . ")

path_to_event_audio = root_path +"event_audio_file_collection/"

build_folder(path_to_event_audio)

for i in range(len(perfect)):
  file_path = perfect["file"][i]
  file_path = root_path + file_path
  print(file_path)
  file_name = file_path.split("/")[-1]
  print(file_name)
  event_occur_time_start = perfect["peak_start"][i]
  print(event_occur_time_start)
  create_new_audio_file(file_path, file_name, event_occur_time_start)

def build_model():
  signal_sampling_rate_list = []
  for file1 in os.listdir(path_to_event_audio):
    fp = path_to_event_audio + file1  # change to the correct path to your file accordingly
    signal, sampling_rate = open_audio(fp)
    signal_sampling_rate_list.append([signal, sampling_rate])
    del signal
    del sampling_rate
  with open(pickle_file_name, 'wb') as f:
    pickle.dump(signal_sampling_rate_list, f)
    feature1 = [feature2[0] for feature2 in signal_sampling_rate_list]
    sampling_rate1 = [feature2[1] for feature2 in signal_sampling_rate_list]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(feature1)
    # Its important to use binary mode 
    knnPickle = open(NearestNeighbors_model_name, 'wb') 
    # source, destination 
    pickle.dump(neigh, knnPickle)                      

# load the model from disk
# loaded_model = pickle.load(open(NearestNeighbors_model_name, 'rb'))

build_model()
m4a_to_wav_path2  = "m4a_to_wav/"
build_folder("m4a_to_wav/")

def get_peak_start_time_in_second(path_total_audio_file, file1):
  # for file1 in os.listdir(path_total_audio_file):
    newAudio = AudioSegment.from_file(path_total_audio_file, format="mp4")
    newAudio.export(m4a_to_wav_path2+file1.split(".")[0]+".wav", format="wav")
    signal, sampling_rate = open_audio(m4a_to_wav_path2 + file1.split(".")[0]+".wav")
    print(file1.split(".")[0]+".wav")
    signal_data1 = signal[:-57330]
    test_data = []
    count1 = floor(len(signal_data1)/57330)
    signal_data1 = signal_data1[0:(count1 * 57330)]
    test_data = np.split(signal_data1, count1)
    test_data = np.array(test_data, dtype=object)
    loaded_model = pickle.load(open(NearestNeighbors_model_name, 'rb'))
    data6 = [loaded_model.kneighbors([feature2]) for feature2 in test_data]
    distance1 = [float(data1[0]) for data1 in data6]
    position1 = [float(data1[1]) for data1 in data6]
    data_dictionary = {"distance" : distance1, "position" : position1}
    print("before sort = ", distance1)
    distance2 = distance1.copy()
    distance1.sort()
    print("distance2 = ", distance2)
    print("after sort = ", distance1)
    distance1 = distance1[0:5]
    print("sorted first 5 element = ", distance1)
    #print("distance2 = ", distance2)
    print("sampling_rate = ", sampling_rate, "len(signal) = ", len(signal), "len(distance1) = ", len(distance1))
    array_per_sample = len(signal) / sampling_rate
    list_of_second_of_peak_start = []
    for i in range(len(distance1)):
      distance_less_index = distance2.index(distance1[i])
      peak_start_count = (distance_less_index * array_per_sample) + distance_less_index
      second_of_peak_start_count = peak_start_count / array_per_sample
      print("second_of_peak_start_count = ", second_of_peak_start_count)  
      list_of_second_of_peak_start.append(second_of_peak_start_count)
    print("list_of_second_of_peak_start = ", list_of_second_of_peak_start)
    return list_of_second_of_peak_start


#########################
# TASK DESCRIPTION
#########################

### The data ############
# You've been provided with 22 thirty-second sound files, spanning 11 minutes.
# The user of the device coughed 10 times during this 11 minute period.
# The exact timestamps of these coughs are in `ground_truth`
# `samples_df` has the mapping of the 30 second files, along with the time ("timestamp") at which time the 30 second
# recording started

### The task ############
# Write a function in python which takes a sound file as an argument
# and returns a dataframe of times (number of second since file start) at which a cough occurred.
# This is an event detection task in which it is far better to OVER-detect than
# to UNDER-detect. In regards to tuning prec/recall, you should consider
# a correctly detected cough to be worth 10 "points" and a "false positive"
# (ie, a "peak" which is not a cough) to be worth -1 points.
# The simpler, the better. No need to use Tensorflow, pre-trained models, or anything like that.
# Feel free to use libraries, but know that this is not a test of your modeling skills.

def detect_coughs(this_file):
    # Replace the below random code with something meaningful which
    # generates a one-column dataframe with a column named "peak_start"
    file_name = this_file.split("/")[-1]
    get_peak_start_time_in_second(this_file, file_name)
    peaks = np.random.sample(5) * 30
    # peaks.sort()
    out = pd.DataFrame({'peak_start': peaks})
    return(out)

# Run function on all sounds
sounds_dir = directory_of_sounds + 'vi95kMQ65UeU7K1wae12D1GUeXd2/'
all_sounds = os.listdir(sounds_dir)
out_list = []
for i in range(len(all_sounds)):
    this_file = sounds_dir + all_sounds[i]
    this_result = detect_coughs(this_file)
    this_result['file'] = this_file
    print("this_result = ", this_result)
    out_list.append(this_result)
print("out_list = ", out_list)    
final = pd.concat(out_list)
print(final)
final.to_csv(root_path+'final.csv')

# Grade the approach
true_positives = 0
# Detect if coughs were correctly corrected
for i in range(len(perfect)):
    this_cough = perfect.iloc[i]
    same_file = final[final['file'] == this_cough['file']]
    # Get time differences
    same_file['time_diff'] = this_cough['peak_start'] - same_file['peak_start']
    keep = same_file[same_file['time_diff'] <= 0.4]
    keep = keep[keep['time_diff'] >= -0.4]
    if len(keep) > 0:
        print('Correctly found the cough at ', str(round(this_cough['peak_start'], 2)) + ' in ' + this_cough['file'])
        true_positives = true_positives + 1
    else:
        print('Missed the cough at ', str(round(this_cough['peak_start'], 2)) + ' in ' + this_cough['file'])
        pass
# Now measure false positives
false_positives = len(final) - true_positives
print('Detected ' + str(false_positives) + ' false positives')
# Calculate final score
final_score = (true_positives * 10) - false_positives
print('FINAL SCORE OF: ' + str(final_score))