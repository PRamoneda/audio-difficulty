# pip install pytube; pip install moviepy librosa piano_transcription_inference; sudo apt-get instsudo apt-get install ffmpeg
import math
import os.path
from multiprocessing import Pool

from pytube import YouTube
from moviepy.editor import VideoFileClip

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import pretty_midi
import numpy as np
from tqdm import tqdm


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

def extract_mel(path_mp3, path_mel):
    (audio, _) = load_audio(path_mp3, sr=sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=640, n_mels=229, fmin=30, fmax=8000)
    mel = mel.transpose()
    mel = (mel - np.amin(mel)) / (np.amax(mel) - np.amin(mel))
    save_binary(mel, path_mel)

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


def convert2pianoroll(args):
    path_midi, path_pianoroll, metadata, fs = args
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



if __name__ == '__main__':
    # data = load_json("metadata_women_extended.json")
    fs = 20
    arguments, arguments_video = [], []
    pool = Pool(processes=1)
    more_long = []
    total_data = load_json("final_index/new_clean_data.json")

    for idx in total_data.keys():
            print(idx)
            idx = str(idx)
            path_video = "videos/" + idx + ".mp4"
            path_mp3 = "mp3/" + idx + ".mp3"
            path_midi = "midi/" + idx + ".mid"
            pianoroll_path = path_midi.replace(".mid", ".bin").replace("midi/", "")
            if not os.path.exists(f"pianoroll{fs}/" + remove_strange_characters(pianoroll_path).replace(".bin", "_onset.bin")):
                arguments.append((
                    path_midi,
                    f"pianoroll{fs}/" + remove_strange_characters(pianoroll_path),
                    total_data[idx],
                    fs
                ))
            if not os.path.exists(path_midi):
                arguments_video.append(("", path_video, path_mp3, None, path_midi))
                more_long.append(idx)
    # pool.map(convert2pianoroll, arguments)
    # pool.map(process_video, arguments)
    print(more_long)
    for arg in tqdm(arguments_video):
        process_video(arg)
    for arg in tqdm(arguments):
        convert2pianoroll(arg)
    pool.close()
    pool.join()


