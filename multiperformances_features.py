# pip install pytube; pip install moviepy librosa piano_transcription_inference; sudo apt-get instsudo apt-get install ffmpeg
import math
import os
import os.path
from multiprocessing import Pool

from pytube import YouTube
from moviepy.editor import VideoFileClip

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import pretty_midi
import numpy as np
from tqdm import tqdm
import logging

from get_cqt import extract_mel, extract_mel_v2
from utils import load_binary
from scipy.signal import resample

def remove_strange_characters(string):
    string = string.replace("#", "")
    string = string.replace("'", "")
    string = string.replace('"', "")
    string = string.replace('?', "")
    string = string.replace("!", "")
    string = string.replace("_", " ")
    string = string.replace("\u2013", "")
    string = string.replace("/", "")
    string = string.replace(":", "")
    string = string.replace("&", "")
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace("\u00ef", "")
    string = string.replace("\u00ea", "")
    string = string.replace("\u00e9", "")
    string = string.replace("\u00c9", "")
    string = string.replace("\u2014", "")
    string = string.replace("\u201c", "")
    string = string.replace("\u201d", "")
    string = string.replace("\u00b4", "")
    string = string.replace("\u00ed", "")
    string = string.replace("\u00e8", "")
    string = string.replace("\u00ed", "")
    string = string.replace("\u2018", "")
    string = string.replace("\u00c5", "")
    string = string.replace("\u0002", "")
    string = string.replace("\u00e1", "")
    string = string.replace("\u00f3", "")
    string = string.replace("\u2019", "")
    return string

def save_binary(data, path):
    import pickle
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def extract_midi(path_mp3, path_midi):
    # Load audio
    (audio, _) = load_audio(path_mp3, sr=sample_rate, mono=True)
    # Transcriptor
    transcriptor = PianoTranscription(device='cuda:0', checkpoint_path=None)  # device: 'cuda' | 'cpu'
    # Transcribe and write out to MIDI file
    transcriptor.transcribe(audio, path_midi)


def create_onset_matrix(midi_data, fs=100):
    """
    Create an onset matrix from a PrettyMIDI object.

    Args:
    midi_data (pretty_midi.PrettyMIDI): The MIDI data.
    fs (int): The sampling frequency (frames per second) for the onset matrix.

    Returns:
    numpy.ndarray: A matrix representing the onsets of notes.
    """
    import numpy as np

    # Determine the total length of the midi file in seconds
    total_length = midi_data.get_end_time()

    # Create a matrix to store onsets
    onsets_matrix = np.zeros((128, int(total_length * fs)))

    # Populate the onsets matrix
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_frame = math.trunc(note.start * fs)
            if start_frame < onsets_matrix.shape[1]:
                onsets_matrix[note.pitch, start_frame] = 1
    return onsets_matrix[21:109].transpose()


def convert2pianoroll(path_midi, path_pianoroll, metadata, fs):
    midi_data = pretty_midi.PrettyMIDI(path_midi)
    # Retrieve piano roll of the MIDI file
    piano_roll = midi_data.get_piano_roll(fs=fs)
    #reduce to 88 keys
    piano_roll = piano_roll[21:109].transpose()
    print(path_pianoroll, piano_roll.shape)
    # normalize pianoroll
    piano_roll = piano_roll / 127
    # get onset matrix
    onset_matrix = create_onset_matrix(midi_data, fs=fs)
    assert onset_matrix.shape == piano_roll.shape, "onset matrix and piano roll should have the same shape"
    if "start" in metadata.keys() and "end" in metadata.keys():
        start, end = metadata["start"], metadata["end"]
        start = int(start * fs)
        end = int(end * fs)
        piano_roll = piano_roll[start:end]
        onset_matrix = onset_matrix[start:end]
        print("start and end")
    elif "only_start" in metadata.keys():
        start = metadata["only_start"]
        start = int(start * fs)
        piano_roll = piano_roll[start:]
        onset_matrix = onset_matrix[start:]
        print("only start")
    elif "only_end" in metadata.keys():
        end = metadata["only_end"]
        end = int(end * fs)
        piano_roll = piano_roll[:end]
        onset_matrix = onset_matrix[:end]
        print("only end")
    # save pianoroll
    save_binary(piano_roll, path_pianoroll)
    # save onset matrix
    save_binary(onset_matrix, path_pianoroll.replace(".bin", "_onset.bin"))


def process_video(arguments):
    url_youtube, path_video, path_mp3, path_mel, path_midi = arguments
    print(path_mp3)
    # extract_mel(path_mp3, path_mel)
    extract_midi(path_mp3, path_midi)
    path_bin_process = path_midi.replace(".mid", ".bin").replace("midi/", "pianoroll/")
    convert2pianoroll((path_midi, path_bin_process))

from mutagen.mp3 import MP3

def is_more_than_15(mp3_path):
    """
    Checks if the duration of an MP3 file is more than 15 minutes.

    Args:
    mp3_path (str): The file path of the MP3 file.

    Returns:
    bool: True if the MP3 duration is more than 15 minutes, False otherwise.
    """
    audio = MP3(mp3_path)
    duration = audio.info.length  # duration in seconds
    return duration > 15 * 60


def load_json(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return data


from pytube import YouTube
from moviepy.editor import *


def download_youtube_video_as_mp3(url, path, start_time, end_time):
    """
    Download a video from a YouTube URL and save it as an MP3 file in the specified path.

    Parameters:
    - url: The YouTube URL of the video to download.
    - path: The directory path where the MP3 file will be saved.
    """
    # Create a YouTube object with the URL
    yt = YouTube(url)

    # Get the highest quality audio stream available
    try:
        video = yt.streams.filter(only_audio=True).first()
    except Exception as e:
        logging.exception(f"Error downloading {url}: {e}")
        return

    # Download the audio stream
    out_file = video.download(output_path=f"tmp/{path}")

    # Load the downloaded audio file
    mp3_file = AudioFileClip(out_file)
    
    # Set the start and end time if provided
    mp3_file = mp3_file.subclip(float(start_time), float(end_time))

    # Save the audio as an MP3 file
    mp3_filename = out_file.split("\\")[-1].replace(".mp4", ".mp3")
    mp3_file.write_audiofile(path)

    # Optionally, remove the original download if it's not in MP3 format
    os.remove(out_file)


def extract_cqt_full(path_mp3, path_mel, metadata):
    # Load the audio at a sampling rate of 44100 Hz
    sample_rate = 44100
    audio, _ = librosa.load(path_mp3, sr=sample_rate, mono=True)



    hop_length = 160
    cqt = librosa.cqt(audio, sr=sample_rate, hop_length=hop_length, n_bins=88, bins_per_octave=12)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    log_mel_spectrogram = log_cqt.T


    # Handle metadata for slicing the log Mel spectrogram
    if "start" in metadata.keys() and "end" in metadata.keys():
        start, end = metadata["start"], metadata["end"]
        start_frame = int(start * sample_rate / hop_length)
        end_frame = int(end * sample_rate / hop_length)
        log_mel_spectrogram = log_mel_spectrogram[start_frame:end_frame]
    elif "only_start" in metadata.keys():
        start = metadata["only_start"]
        start_frame = int(start * sample_rate / hop_length)
        log_mel_spectrogram = log_mel_spectrogram[start_frame:]
    elif "only_end" in metadata.keys():
        end = metadata["only_end"]
        end_frame = int(end * sample_rate / hop_length)
        log_mel_spectrogram = log_mel_spectrogram[:end_frame]

    # Save the log Mel spectrogram
    print(log_mel_spectrogram.shape)
    save_binary(log_mel_spectrogram, path_mel)

def downsample_log_mel_spectrogram(log_mel_spectrogram, target_fs):
    """
    Downsample a log Mel spectrogram to a target frame rate (frames per second).

    Parameters:
    log_mel_spectrogram (numpy.ndarray): The original log Mel spectrogram.
    original_fs (int): The original frame rate (frames per second).
    target_fs (int): The target frame rate (frames per second).

    Returns:
    numpy.ndarray: The downsampled log Mel spectrogram.
    """
    # Calculate the number of original time frames per target time frame
    original_fs = 44100 / 160
    ratio = original_fs / target_fs
    # Downsample along the time axis
    downsampled_spectrogram = resample(log_mel_spectrogram, int(log_mel_spectrogram.shape[0] / ratio), axis=0)

    return downsampled_spectrogram


def downsample_cqt(path_mel, to_save, fs):
    print(path_mel)
    mel = load_binary(path_mel)
    new_mel = downsample_log_mel_spectrogram(mel, fs)
    print(new_mel.shape)
    save_binary(new_mel.T, to_save)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR, filename="multiperformances.log")
    fs = 5

    dir_names = ["multi/mp3", "multi/midi", f"multi/pr{fs}", f"multi/cqt{fs}", "multi/cqt_full"]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    for file_name in tqdm(os.listdir("benchmark_multiperformances")):
        if not file_name.endswith(".json"):
            continue
        
        data = load_json(f"benchmark_multiperformances/{file_name}")
        piece_index = file_name.replace(".json", "")

        for idx, dd in data.items():
            idx = piece_index + ":" + idx
            # download the video from youtube and save into mp3
            if not os.path.exists(f"multi/mp3/{idx}.mp3"):
                download_youtube_video_as_mp3(dd["youtube_url"], f"multi/mp3/{idx}.mp3", dd["start_time"], dd["end_time"])
                if not os.path.exists(f"multi/mp3/{idx}.mp3"):
                    continue

            # transcribe midi from audio with Kong et al (tiktok)
            if not os.path.exists(f"multi/midi/{idx}.mid"):
                extract_midi(f"multi/mp3/{idx}.mp3", f"multi/midi/{idx}.mid")
            # pianoroll from midi
            if not os.path.exists(f"multi/mel/{idx}.bin"):
                convert2pianoroll(f"multi/midi/{idx}.mid", f"multi/pr{fs}/{idx}.bin", dd, fs)
            # cqt from mp3
            if not os.path.exists(f"multi/cqt_full/{idx}.bin"):
                extract_cqt_full(f"multi/mp3/{idx}.mp3", f"multi/cqt_full/{idx}.bin", dd)
            # downsample cqt
            if not os.path.exists(f"multi/cqt{fs}/{idx}.bin"):
                downsample_cqt(f"multi/cqt_full/{idx}.bin", f"multi/cqt{fs}/{idx}.bin", fs)