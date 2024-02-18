import numpy as np
import pandas as pd
import os 
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa as lb
import tensorflow_io as tfio
from tqdm import tqdm 
from IPython.display import Audio
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import pickle 
from tensorflow.keras.models import load_model
import streamlit as st

st.title(":guitar: :musical_score: Chord Craft: Crafting Harmonic from Audio  :musical_score: :guitar:")
st.write("This is a simple web app that uses a Convolutional Neural Network to predict the chord from an audio file. The model was trained on the our own dataset and has an accuracy of 95% on the test set. The model was trained using the [Librosa](https://librosa.org/doc/main/index.html) library for audio processing and the [Tensorflow](https://www.tensorflow.org/) library for the model building and training. The model was trained on a 2D representation of the audio data, which was obtained using the Short Time Fourier Transform (STFT).")

st.write("The model was trained on the following chords: A, C, F, The model was trained on 2 second audio clips of the Ukele instrudment chords. The model was trained on 80% of the data and tested on the remaining 20% of the data. The model was trained for 50 epochs with a batch size of 16.")




DATAPATH = "dataset_chords"
def load_wav_file(filepath, waveform = True, name = "Audio file"): 
    """
    This function will load the audio file into 16Khz sampling rate
    
    returns wav file in tensorflow format
    """
    wav, rate = lb.load(filepath, sr = 16000)
    wav = tf.constant(wav)
    if waveform: 
        plt.plot(wav, label= name)
        plt.legend()
        plt.show()
    return wav

def convert_wav_to_spectrogram(wav,name = "Spectrogram",show = True, frame_length = 320, frame_step = 32, average_length = 35000):
    """
    convert tensorflow formated wav file to spectrogram of desired framelenght and fram steps
    
    also convert them to a average length and if not them add padding to them
    
    and returns spectrogram with mono channel
    """
    wav = wav[:average_length]
    #add zero padding at end
    zero_padding = tf.zeros([average_length] - tf.shape(wav), dtype = tf.float32)
    # concate that to audio file at end
    wav = tf.concat([wav, zero_padding], 0)
    # Create spectrogram of wavefrom file
    spectrogram = tf.signal.stft(wav, frame_length = 320, frame_step = 32)
    #convert to absolute form
    spectrogram = abs(spectrogram)
    #Expand dims for adding channel (mono audio) 1
    spectrogram = tf.expand_dims(spectrogram, axis = 2)
    if show:
        plt.figure(figsize = (30, 20))
        plt.imshow(tf.transpose(spectrogram)[0])
        plt.title(name)
        plt.show()
    return spectrogram

def preprocessing(filepath, labels): 
    wav = load_wav_file(filepath, waveform=False)
    spectrogram = convert_wav_to_spectrogram(wav,show=False)
    return spectrogram, labels


def convert_pred_label(y_pred): 
    y_pred = np.argmax(y_pred, axis = 1)
    labels = os.listdir(DATAPATH)
    results = [[labels[x]] for x in y_pred]
    return np.array(results)
def show_label_with_pred_prob(y_pred): 
    chords = convert_pred_label(y_pred)
    label_index = np.argmax(y_pred, axis = 1)
    pred_prob = np.array([[y_pred[i][label_index[i]]] for i in range(len(y_pred))])
    result = np.hstack((chords, pred_prob))
    return result
def convert_prob_label_index(y_pred): 
    return np.argmax(y_pred, axis = 1)

def make_pred_custom_chord(filepath, model): 
    wav = load_wav_file(filepath)
    spectrogram = convert_wav_to_spectrogram(wav)
    pred = model.predict(tf.expand_dims(spectrogram, axis = 0))
    return show_label_with_pred_prob(pred)

def load_mp3_slited_files(path): 
    wav = load_wav_file(path, waveform=False)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, 
                                                               sequence_length=35000, 
                                                               sequence_stride=35000,
                                                               batch_size=1)
    slices = []
    for sample in audio_slices.as_numpy_iterator():
        slices.append(convert_wav_to_spectrogram(sample[0][0], show = False))
    slices = tf.data.Dataset.from_tensor_slices(slices)
    slices = slices.batch(len(audio_slices))
    return slices

st.title(":notes: Upload A single [Clip] file")
audio_file = st.file_uploader("Upload a Chord", type=['wav', 'mp3'])

def predict_chord():
    if audio_file is None:
        st.error("Please upload an audio file/ Select one Category")
        return
    st.sidebar.audio(audio_file, format='audio/wav')
    print(audio_file.name)

    # st.save_audio(audio_file, 'audio.wav')
    with open("audio_files/"+audio_file.name, 'wb') as f: 
        f.write(audio_file.getbuffer())

    # Load the model
    path = "C:/Mitsgwl/sem 5/Minor_project/models/Classifier_ACF_25_epochs.keras"
    Classifier = load_model(path)
    results = make_pred_custom_chord("audio_files/"+audio_file.name, model = Classifier)

    st.sidebar.write("The model has predicted the following chords from the audio file")
    i = 0
    for chord, precentage in results:
        write  = "🎼 Chord: {} ||  Accuracy: {:.2f}% 🎼".format(chord, float(precentage)*100)
        st.sidebar.success(write)
        i+=1

st.button("Predict Chord", on_click=predict_chord)
    
st.title(":musical_keyboard: Upload an Music [Audio] file")
song_file = st.file_uploader("Upload a Sound", type=['wav', 'mp3'])

def predict_Song():
    if song_file is None:
        st.error("Please upload an audio file/ Select one Category")
        return
    st.sidebar.audio(song_file, format='audio/wav')
    print(song_file.name)

    # st.save_audio(audio_file, 'audio.wav')
    with open("audio_files/"+song_file.name, 'wb') as f: 
        f.write(song_file.getbuffer())

    # Load the model
    path = "C:/Mitsgwl/sem 5/Minor_project/models/Classifier_ACF_25_epochs.keras"
    Classifier = load_model(path)
    Slices = load_mp3_slited_files("audio_files/"+song_file.name)
    results = convert_pred_label(Classifier.predict(Slices))[:, -1]

    st.sidebar.write("The model has predicted the following chords from the audio file")
    i = 0
    for chord in results:
        write  = "Chords: {} ".format(chord)
        st.sidebar.success(write)
        i+=1

    

st.button("Predict Song", on_click=predict_Song)
    

