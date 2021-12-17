#!/usr/bin/python3

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from music21 import *
from scipy.signal import savgol_filter

import cv2
import subprocess
import sys
import shutil
import os
import argparse
import configparser

import midi

def peaks(seq):
    data = []
    for i, x in enumerate(seq):
        if i <= 1 or i >= len(seq) - 2:
            continue
    #    if seq[i - 2] < seq[i - 1] < x and seq[i + 2] < seq[i + 1] < x:
        data.append((i, x))
    return sorted(data, key=lambda x: -x[1])

def db_to_vol(db,freq):
    transformed = np.fix(np.float((127 - 3 * abs(db))) * np.exp(-freq / 2000))
    return transformed if transformed > 32 else 0


def keydiff(freq1, freq2):
    return abs(12 * np.log2(freq1 / freq2))


def make_note(freq, dur, vol):
    n = note.Note()
    p = pitch.Pitch()
    p.frequency = freq
    n.pitch = p
    n.duration = duration.Duration(dur)
    n.volume.velocity = vol
    return n

def make_stream(top_freqs, keydiff_threshold=1):
    s = stream.Stream()

    freqs = np.array([f for (f, i) in top_freqs])
    intensities = np.array([i for (f, i) in top_freqs])
    default_duration = 0.25  # 16th note

    for voice, intensity in zip(freqs.T, intensities.T):
        part = stream.Part()
        last_frequency = voice[0]
        dur = default_duration
        vol = db_to_vol(intensity[0],last_frequency)

        for note_idx in range(1, len(voice)):
            if keydiff(voice[note_idx], last_frequency) >= keydiff_threshold:
                n = make_note(last_frequency, dur, vol)
                part.append(n)

                # reset
                last_frequency = voice[note_idx]
                dur = default_duration
                vol = db_to_vol(intensity[note_idx], last_frequency)
            else:
                dur += default_duration

        n = make_note(last_frequency, dur, vol)
        part.append(n)
        s.insert(0, part)
    return s

def prominent_frequencies(spectrogram, num_peaks):
    frequencies_copy = dict(enumerate(librosa.fft_frequencies(sr=48000, n_fft=4096)))
    prom_freqs = []
    for time_slice in spectrogram.T:
        pitches = []
        intensities = []

        # remove high frequencies by only taking values at indices 0-200
        time_slice = time_slice[:200]

        # remove low volumes that are insignificant
        time_slice = [x if x > -60 else -80 for x in time_slice] # -80 is no volume

        # filter out low frequencies
        for i in range(6):
            time_slice[i] = -80

        # smooth frequencies
        time_slice = savgol_filter(time_slice, 9, 3)

        # store with intensity
        peaks = []
        for i, x in enumerate(time_slice):
            if i <= 1 or i >= len(time_slice) - 2:
                continue
            peaks.append((i, x))
        peaks = sorted(peaks, key=lambda x: -x[1])
        
        for (idx, value) in peaks[:num_peaks]:
            hz = frequencies_copy[idx]
            if hz not in pitches:
                pitches.append(hz)
                intensities.append(value)

        # account for not enough peaks (silence)
        while len(pitches) < num_peaks:
            pitches.append(1)
            intensities.append(-80)

        prom_freqs.append((pitches, intensities))
    return prom_freqs

def data_to_midi(data, sample_rate, output_file):
    spectrogram = librosa.stft(data, n_fft=4096, hop_length=512)
    db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    prom_freqs = prominent_frequencies(db, 12) # 12 is an arbitrary number of peaks
    
    # writing the data to the midi
    s = make_stream(prom_freqs, 2)
    s.insert(0, tempo.MetronomeMark(number=500))
    s.write("midi", output_file)
    
def convert_wav_to_midi(input_filename_wav, output_filename_midi):
    data, sample_rate = sf.read(input_filename_wav, dtype='float32')
    data_to_midi(data, sample_rate, output_filename_midi)
    
# convert_wav_to_midi(output_filename_wav, output_filename_midi)