# pip install pytube; pip install moviepy librosa piano_transcription_inference; sudo apt-get instsudo apt-get install ffmpeg
import math
import os
import os.path

from pytube import YouTube
from moviepy.editor import *
import yt_dlp

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import pretty_midi
import numpy as np
from tqdm import tqdm
import logging

from utils import load_binary
from scipy.signal import resample


def remove_strange_characters(string):
    strange_characters = ["#", "'", '"', '?', '!', "_", "\u2013", "/", ":", "&", "[", "]", "\u00ef", "\u00ea", "\u00e9",
                         "\u00c9", "\u2014", "\u201c", "\u201d", "\u00b4", "\u00ed", "\u00e8", "\u00ed", "\u2018",
                         "\u00c5", "\u0002", "\u00e1", "\u00f3", "\u2019"]
    for character in strange_characters:
        string = string.replace(character, "")
        
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
    # normalize pianoroll
    piano_roll = piano_roll / 127
    # get onset matrix
    onset_matrix = create_onset_matrix(midi_data, fs=fs)
    assert onset_matrix.shape == piano_roll.shape, "onset matrix and piano roll should have the same shape"
    # save pianoroll
    save_binary(piano_roll, path_pianoroll)
    # save onset matrix
    save_binary(onset_matrix, path_pianoroll.replace(".bin", "_onset.bin"))


def process_video(arguments):
    url_youtube, path_video, path_mp3, path_mel, path_midi = arguments
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


def download_youtube_video_as_mp3(url, file_name, start_time, end_time):
    """
    Download a video from a YouTube URL and save it as an MP3 file in the specified path. It first tries using pytube,
    and if it fails, it falls back to yt-dlp.

    Parameters:
    - url: The YouTube URL of the video to download.
    - file_name: The directory path where the MP3 file will be saved.
    - start_time: The start time in seconds from where the audio should be trimmed.
    - end_time: The end time in seconds to where the audio should be trimmed.
    """
    try:
        # Try downloading with pytube
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=f"tmp/{file_name}")
        mp3_file_path = out_file.replace(".mp4", ".mp3")
        
        # Rename the file to have a .mp3 extension
        os.rename(out_file, mp3_file_path)
    except Exception as e:
        logging.exception(f"pytube failed. Error: {e}")
        try:
            # Setup yt-dlp options for downloading audio
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': f'tmp/{file_name[:-4]}',
            }

            # Use yt-dlp to download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find the downloaded MP3 file
            for file in os.listdir(f'tmp/{file_name}'):
                if file.endswith('.mp3'):
                    mp3_file_path = os.path.join(f'tmp/{file_name}', file)
                    break
        except Exception as e:
            logging.exception(f"yt-dlp failed. Error: {e}")
            return

    # Load the downloaded MP3 file
    audio_clip = AudioFileClip(mp3_file_path)
    
    # Set the start and end time if provided, ensure they are within the duration
    end_time = min(end_time, audio_clip.duration)
    assert start_time < end_time <= audio_clip.duration, f"Invalid start and end times: {start_time}, {end_time}, {audio_clip.duration}"
    
    # Trim the audio file
    trimmed_audio = audio_clip.subclip(start_time, end_time)

    # Save the trimmed audio as an MP3 file
    trimmed_audio.write_audiofile(mp3_file_path)

    # Close the audio file to free resources
    audio_clip.close()
    trimmed_audio.close()


def extract_cqt_full(path_mp3, path_mel, metadata):
    # Load the audio at a sampling rate of 44100 Hz
    sample_rate = 44100
    audio, _ = librosa.load(path_mp3, sr=sample_rate, mono=True)

    hop_length = 160
    cqt = librosa.cqt(audio, sr=sample_rate, hop_length=hop_length, n_bins=88, bins_per_octave=12)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    log_mel_spectrogram = log_cqt.T

    # Save the log Mel spectrogram
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
    mel = load_binary(path_mel)
    new_mel = downsample_log_mel_spectrogram(mel, fs)
    save_binary(new_mel.T, to_save)


if __name__ == '__main__':
    # Clear the log file
    log_file = "multiperformances.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Configure logging
    logging.basicConfig(level=logging.ERROR, filename=log_file)
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