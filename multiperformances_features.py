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

from utils import *
from scipy.signal import resample


def extract_midi(path_mp3, path_midi):
    assert os.path.exists(path_mp3), f"File {path_mp3} does not exist"
    # Load audio
    (audio, _) = librosa.load(path_mp3, sr=sample_rate, mono=True)
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


def download_youtube_video_as_mp3(url, file_name, start_time, end_time):
    """
    Download a video from a YouTube URL and save it as an MP3 file in the specified path. It first tries using pytube,
    and if it fails, it falls back to yt-dlp.

    Parameters:
    - url: The YouTube URL of the video to download.
    - file_name: The file name with .mp3 extension to save the audio.
    - start_time: The start time in seconds from where the audio should be trimmed.
    - end_time: The end time in seconds to where the audio should be trimmed.
    """
    tmp_mp3_file_name = f'tmp/{file_name}' # Ends with .mp3
    try:
        # Try downloading with pytube
        yt = YouTube(url)

        # Get the highest quality audio stream available
        video = yt.streams.filter(only_audio=True).first()

        # Download the audio stream
        out_file = video.download(filename=tmp_mp3_file_name[:-4]+'.mp4')
        # Load the downloaded audio file
        audio_clip = AudioFileClip(out_file)
        # Remove the created mp4 file
        os.remove(out_file)
        
        
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
                'outtmpl': f'{tmp_mp3_file_name[:-4]}.%(ext)s',
            }

            # Use yt-dlp to download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            # Load the downloaded MP3 file
            audio_clip = AudioFileClip(tmp_mp3_file_name)

        except Exception as e:
            logging.exception(f"yt-dlp failed. Error: {e}")
            return
    
    # Set the start and end time if provided, ensure they are within the duration
    end_time = min(end_time, audio_clip.duration)
    assert start_time < end_time <= audio_clip.duration, f"Invalid start and end times: {start_time}, {end_time}, {audio_clip.duration}"
    
    # Trim the audio file
    trimmed_audio = audio_clip.subclip(start_time, end_time)

    # Save the trimmed audio as an MP3 file
    trimmed_audio.write_audiofile(file_name)

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


def init(inference_type, fs):
    # init logging
    log_file = f"{inference_type}_features.log"
    with open(log_file, "w") as file:
        logging.basicConfig(level=logging.ERROR, filename=log_file)

    global DIR_DICT
    DIR_DICT = {
        'mp3': f'{inference_type}/mp3',
        'midi': f'{inference_type}/midi',
        'pr': f'{inference_type}/pr{fs}',
        'cqt': f'{inference_type}/cqt{fs}',
        'cqt_full': f'{inference_type}/cqt_full',
        'tmp': 'tmp'
    }

    for dir_path in DIR_DICT.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)



def get_filenames(idx, piece_index):
    idx = piece_index + ":" + idx
    mp3_fn = f"{DIR_DICT['mp3']}/{idx}.mp3"
    midi_fn = f"{DIR_DICT['midi']}/{idx}.mid"
    pr_fn = f"{DIR_DICT['pr']}/{idx}.bin"
    cqt_fn = f"{DIR_DICT['cqt']}/{idx}.bin"
    cqt_full_fn = f"{DIR_DICT['cqt_full']}/{idx}.bin"


    return idx, mp3_fn, midi_fn, pr_fn, cqt_fn, cqt_full_fn


if __name__ == '__main__':
    fs = 5
    init('multi', 5)

    for file_name in tqdm(os.listdir("benchmark_multiperformances")):
        if not file_name.endswith(".json"):
            continue
        
        data = load_json(f"benchmark_multiperformances/{file_name}")
        piece_index = file_name.replace(".json", "")

        for idx, dd in data.items():
            idx, mp3_fn, midi_fn, pr_fn, cqt_fn, cqt_full_fn = get_filenames(idx, piece_index)

            # download the video from youtube and save into mp3
            if not os.path.exists(mp3_fn):
                download_youtube_video_as_mp3(dd["youtube_url"], mp3_fn, dd["start_time"], dd["end_time"])
                if not os.path.exists(mp3_fn):
                    continue
            # transcribe midi from audio with Kong et al (tiktok)
            if not os.path.exists(midi_fn):
                extract_midi(mp3_fn, midi_fn)
            # pianoroll from midi
            if not os.path.exists(pr_fn):
                convert2pianoroll(midi_fn, pr_fn, dd, fs)
            # cqt from mp3
            if not os.path.exists(cqt_full_fn):
                extract_cqt_full(mp3_fn, cqt_full_fn, dd)
            # downsample cqt
            if not os.path.exists(cqt_fn):
                downsample_cqt(cqt_full_fn, cqt_fn, fs)