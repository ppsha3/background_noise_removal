#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
from pydub import AudioSegment
import librosa
import moviepy.editor as mp
import soundfile as sf


def getAudioFromVideo(video_path, audio_path):
    '''
        This will fetch the audio from video described by video_path 
        and store it in mp3 file in audio_path.
    '''
    
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

    return True


def addAudioToVideo(src, dest, audio):
    '''
        Adds audio from audio_path to video at video_path and outputs to dest_path
    '''

    video = mp.VideoFileClip(src)
    video.write_videofile(dest, audio=audio)

    return True


def readAudio(audio_path, sr=None, chunk_size=None):
    '''
        This will read audio from the audio file. If chunk_size 
        is specified, it will slice the audio into chunks of length chunk_size.
    '''

    # We will get the amplitude of the specified audio file into variable audio
    raw_audio, samplerate = librosa.core.load(audio_path, sr=sr)

    # slicing the audio
    chunk = chunk_size * samplerate      # 1 second of audio has 44.1 KHz of "samples" 

    audio = np.array([raw_audio[i:i+chunk] for i in range(0, len(raw_audio), chunk)])

    return raw_audio, audio, chunk


def computeEnergy(audio):
    '''
        Compute the energy of the audio. Audio could be sliced audio of previously specified chunks.
    '''

    audio_amp =  np.abs(audio) ** 2    
    return [np.sum(x) for x in audio_amp]


def createSilence(secs):
    duration = secs * 1000 * 4
    silence = AudioSegment.silent(duration=duration)  #duration in milliseconds

    silence = silence.get_array_of_samples()

    return np.array(silence)


def main():

    # video_path = 'data/video/test.mp4'
    audio_path = 'data/audio/test.mp3'

    # getAudioFromVideo(video_path, audio_path)

    chunk_size = 4

    # raw_audio, audio, chunk = readAudio(audio_path, chunk_size=chunk_size, sr=None)  # 'sr=None' means it will default to the native sample rate of the audio file
    
    # np.save('audio.npy', audio)
    # np.save('raw_audio.npy', raw_audio)
    audio = np.load('audio.npy', allow_pickle=True)
    raw_audio = np.load('raw_audio.npy', allow_pickle=True)
    
    energy = computeEnergy(audio)

    # print(energy)

    # plt.hist(energy) 
    # plt.show()

    threshold = 1600
    chunk = 4 * 44100
    silence = createSilence(secs=chunk_size)

    for i in range(len(energy)):

        if energy[i] > threshold:
            start = i+1 * chunk
            end = start + chunk
            raw_audio[start:end] = silence

    audio_path = 'data/audio/edited.wav'
    sf.write(audio_path, raw_audio, 44100)

    video_path = 'data/video/test.mp4'
    dest_path = 'data/video/edited.wav'

    # addAudioToVideo(src=video_path, dest=dest_path, audio=audio_path)

main()