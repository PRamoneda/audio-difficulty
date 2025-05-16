import json
import os.path
import pdb
from subprocess import call

import librosa
import numpy as np
from librosa import load
from librosa.feature import melspectrogram
from scipy.signal import resample
from tqdm import tqdm

from utils import load_binary


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


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


# def extract_mel(path_mp3, path_mel, fs, metadata):
#     sample_rate = 44100
#
#     # Calculate the duration of each frame in seconds (1 frame per fs frames per second)
#     frame_duration = 1.0 / fs
#
#     # Convert frame duration to samples
#     hop_length = int(frame_duration * sample_rate)
#
#     audio, _ = load(path_mp3, sr=sample_rate, mono=True)
#     mel = melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=hop_length, n_mels=229, fmin=30, fmax=8000)
#     mel = mel.transpose()
#     mel = (mel - np.amin(mel)) / (np.amax(mel) - np.amin(mel))
#
#     if "start" in metadata.keys() and "end" in metadata.keys():
#         start, end = metadata["start"], metadata["end"]
#         start = int(start * fs)
#         end = int(end * fs)
#         mel = mel[start:end]
#         print("start and end")
#     elif "only_start" in metadata.keys():
#         start = metadata["only_start"]
#         start = int(start * fs)
#         mel = mel[start:]
#         print("only start")
#     elif "only_end" in metadata.keys():
#         end = metadata["only_end"]
#         end = int(end * fs)
#         mel = mel[:end]
#         print("only end")
#     save_binary(mel, path_mel)



def extract_mel(path_mp3, path_mel, fs, rep, metadata):
    # Load the audio at a sampling rate of 44100 Hz
    sample_rate = 44100
    audio, _ = librosa.load(path_mp3, sr=sample_rate, mono=True)


    if rep == "mel":
        # Compute the STFT with a Hann window
        n_fft = 1024
        hop_length = 160
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
        # Compute the magnitude of the STFT
        magnitude = np.abs(stft)

        # Generate the Mel filter banks
        n_mels = 64
        mel_filter_banks = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

        # Apply the Mel filter banks
        mel_spectrogram = np.dot(mel_filter_banks, magnitude ** 2)

        # Convert to log scale (add a small value to avoid log(0))
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)
        # pdb.set_trace()
        # Transpose so that time is the first dimension
        log_mel_spectrogram = log_mel_spectrogram.T
    elif rep == "mel_grande":
        # Compute the STFT with a Hann window
        n_fft = 1024
        hop_length = 160
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
        # Compute the magnitude of the STFT
        magnitude = np.abs(stft)

        # Generate the Mel filter banks
        n_mels = 128
        mel_filter_banks = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        # Apply the Mel filter banks
        mel_spectrogram = np.dot(mel_filter_banks, magnitude ** 2)

        # Convert to log scale (add a small value to avoid log(0))
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)
        # pdb.set_trace()
        # Transpose so that time is the first dimension
        log_mel_spectrogram = log_mel_spectrogram.T
    elif rep == "mel_enorme":
        # Compute the STFT with a Hann window
        n_fft = 1024
        hop_length = 160
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
        # Compute the magnitude of the STFT
        magnitude = np.abs(stft)

        # Generate the Mel filter banks
        n_mels = 700
        mel_filter_banks = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        # Apply the Mel filter banks
        mel_spectrogram = np.dot(mel_filter_banks, magnitude ** 2)

        # Convert to log scale (add a small value to avoid log(0))
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)
        # pdb.set_trace()
        # Transpose so that time is the first dimension
        log_mel_spectrogram = log_mel_spectrogram.T
    elif rep == "cqt":
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



import numpy as np
import scipy.signal

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



def compute_videos():
    # Define the directories for the MP4 and MP3 files
    mp4_directory = '/mnt/disk/pedro/videos_download_big/videos'
    mp3_directory = '/mnt/disk/pedro/videos_download_big/mp3'

    # Create the mp3 directory if it doesn't exist
    if not os.path.exists(mp3_directory):
        os.makedirs(mp3_directory)

    # List all the MP4 files in the mp4_directory
    mp4_files = [file for file in os.listdir(mp4_directory) if file.endswith('.mp4')]

    # Convert each MP4 to MP3 if the MP3 does not already exist
    for mp4_file in mp4_files:
        mp3_file = os.path.splitext(mp4_file)[0] + '.mp3'
        mp3_file_path = os.path.join(mp3_directory, mp3_file)
        mp4_file_path = os.path.join(mp4_directory, mp4_file)

        # Check if the corresponding MP3 file already exists
        if not os.path.exists(mp3_file_path):
            # Command to convert MP4 to MP3 using ffmpeg
            command = ['ffmpeg', '-i', mp4_file_path, '-q:a', '0', '-map', 'a', mp3_file_path]
            try:
                call(command)
            except:
                print(f"Error with {mp4_file_path}")

    print("Conversion completed.")


def extract_mel_v2(path_mel, fs, rep, metadata):
    print(path_mel)
    mel = load_binary(path_mel)
    new_mel = downsample_log_mel_spectrogram(mel, fs)
    print(new_mel.shape)
    save_binary(new_mel.T, f"../videos_download/{rep}{fs}/" + path_mel.split("/")[-1])


def compute_mels(fs, rep="mel"):
    if not os.path.exists(f"../videos_download/{rep}{fs}"):
        os.mkdir(f"../videos_download/{rep}{fs}")

    if not os.path.exists(f"/mnt/disk/pedro/videos_download_big/{rep}"):
        os.mkdir(f"/mnt/disk/pedro/videos_download_big/{rep}")

    index = (load_json("../videos_download/final_index/new_clean_data.json").keys())
    metadata = load_json("../videos_download/final_index/new_clean_data.json")

    malas = []
    sucias = {}
    for x in list(os.listdir(f"/mnt/disk/pedro/videos_download_big/mp3")):
        if x.endswith(".mp3"):
            if remove_strange_characters(x[:-4]) in sucias and "#" in x[:-4]:
                continue
            else:
                sucias[remove_strange_characters(x[:-4])] = x

    for idx in tqdm(index):
        if idx not in malas and not os.path.exists(f"/mnt/disk/pedro/videos_download_big/{rep}/{idx}.bin"):
            extract_mel(f"/mnt/disk/pedro/videos_download_big/mp3/{sucias[idx]}",
                       f"/mnt/disk/pedro/videos_download_big/{rep}/{idx}.bin", fs, rep, metadata[idx])
        if idx not in malas and not os.path.exists(f"../videos_download/{rep}{fs}/{idx}.bin"):
            extract_mel_v2(f"/mnt/disk/pedro/videos_download_big/{rep}/{idx}.bin", fs, rep, metadata[idx])


def mel_dasaem():
    sample_rate = 44100
    fs = 5

    # Calculate the duration of each frame in seconds (1 frame per fs frames per second)
    frame_duration = 1.0 / fs

    # Convert frame duration to samples
    hop_length = int(frame_duration * sample_rate)

    audio, _ = load("1.mp3", sr=sample_rate, mono=True)
    mel = melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=hop_length, n_mels=229, fmin=30, fmax=8000, center=False)
    mel = mel.transpose()
    mel = (mel - np.amin(mel)) / (np.amax(mel) - np.amin(mel))
    return mel.shape


def checkear_dasaem(path):
    pr5 = load_binary(path)
    print(pr5.shape)

    mel = load_binary(path.replace("pianoroll5", "mel"))
    # mel10 = load_binary(path.replace("pianoroll5", "mel10"))
    print(mel.shape)
    mel5 = downsample_log_mel_spectrogram(mel, 5)
    print(mel5.shape)

    # create figure with both and save it
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(pr5.T)
    axs[1].imshow(mel5.T)

    plt.savefig("pr5_mel5.png")


def printar():
    for path in os.listdir("../videos_download/mel5"):
        if path.endswith(".bin"):
            print(path, end=" ")
            data = load_binary(f"../videos_download/mel5/{path}")
            if data.shape[1] != 64:
                print("mal", data.shape)

                mel = load_binary("../videos_download/mel/" + path)
                # mel5 = downsample_log_mel_spectrogram(mel, 5)
                extract_mel_v2("../videos_download/mel/" + path, 5, None)
                print(f"../videos_download/mel5/{path}")
                mel5 = load_binary(f"../videos_download/mel5/{path}")
                print("fixed", mel5.shape)


def harris():
    metadata = load_json("../videos_download/final_index/new_clean_data.json")
    extract_mel(path_mp3="Harris_R.Study_in_Blue_and_Green_(Paekakariki).mp3",
                path_mel="Harris_R.Study_in_Blue_and_Green_(Paekakariki).bin",
                fs=5,
                metadata=metadata["Harris R.Study in Blue and Green (Paekakariki)"])
    mel = load_binary("Harris_R.Study_in_Blue_and_Green_(Paekakariki).bin")
    print(mel.shape)


if __name__ == '__main__':
    # compute_videos()
    # harris()
    # printar()
    # compute_mels(20, "mel_grande")
    # compute_mels(20, "mel_enorme")
    compute_mels(20, "cqt")
    # compute_mels(10)
    # compute_mels(20)
    # print(load_binary("../videos_download/pianoroll5/Bartok B.Variations No 5 from For Children 2 BB 53 Sz 42.bin").shape)
    # checkear_dasaem("../videos_download/pianoroll5/Albeniz I.Zortzico mvt 6 from Espana Op 165.bin")

