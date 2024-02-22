# pip install pytube; pip install moviepy librosa piano_transcription_inference; sudo apt-get instsudo apt-get install ffmpeg
import math
import os.path
from multiprocessing import Pool

import torch
from pytube import YouTube
from moviepy.editor import VideoFileClip

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import pretty_midi
import numpy as np
from tqdm import tqdm
import logging
import unicodedata

import utils
from get_cqt import extract_mel, extract_mel_v2, load_json, save_binary
from make_table_basic_model import AudioModel as AudioModelBasic
from make_table_era_model import AudioModel as AudioModelEra
from make_table_multirank_model import AudioModel as AudioModelMultirank
from utils import load_binary, prediction2label
from scipy.signal import resample


def get_cqt(rep, k):
    inp_data = utils.load_binary(f"multi/{rep}/{k}.bin")
    inp_data = torch.tensor(inp_data, dtype=torch.float32).cuda()
    inp_data = inp_data.unsqueeze(0).unsqueeze(0).transpose(2, 3)
    return inp_data


def get_pianoroll(rep, k):
    inp_pr = utils.load_binary(f"multi/{rep}/{k}.bin")
    inp_on = utils.load_binary(f"multi/{rep}/{k}_onset.bin")
    inp_pr = torch.from_numpy(inp_pr).float().cuda()
    inp_on = torch.from_numpy(inp_on).float().cuda()
    inp_data = torch.stack([inp_pr, inp_on], dim=1)
    inp_data = inp_data.unsqueeze(0).permute(0, 2, 1, 3)
    return inp_data


def compute_multi(rep, mode="basic"):
    print(f"Computing {rep} {mode}")
    benchmark_data = load_json("benchmark_data.json")

    for split in range(5):
        if mode == "basic":
            model = AudioModelBasic(11, rep, False, False, False)
        elif mode == "era":
            model = AudioModelEra(11, rep, False)
        elif mode == "multirank":
            model = AudioModelMultirank(11, rep, False)
        checkpoint = torch.load(f"models/{rep}/checkpoint_{split}.pth", map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        model.eval()
        save = {}
        with torch.inference_mode():
            for file_name in tqdm(os.listdir("benchmark_multiperformances")):
                if not file_name.endswith(".json"):
                    continue

                data = load_json(f"benchmark_multiperformances/{file_name}")
                piece_name = file_name.replace(".json", "")
                for k in data.keys():
                    k = piece_name + ":" + k
                    if "cqt" in rep:
                        inp_data = get_cqt("cqt5", k)
                    elif "pianoroll" in rep or "pr" in rep:
                        inp_data = get_pianoroll("pr5", k)
                    elif "multi" in rep and "multirank" not in rep:
                        x1 = get_pianoroll("pr5", k)
                        x2 = get_cqt("cqt5", k)[:, :, :x1.shape[2]]
                        inp_data = [x1, x2]
                    log_prob = model(inp_data, None)
                    if mode == "multirank" or mode == "era":
                        log_prob = log_prob[0]
                    pred = prediction2label(log_prob).cpu().tolist()[0]
                    save[k] = {
                        "log_prob": log_prob.cpu().tolist(),
                        "pred": pred,
                        "true": int(benchmark_data[unicodedata.normalize('NFC', piece_name)]["ps_rating"])
                    }
                save_binary(save, f"multi/logits/{rep}_split_{split}.bin")


def init(inference_type):
    # init logging
    log_file = f"{inference_type}_logits.log"
    with open(log_file, "w") as file:
        logging.basicConfig(level=logging.ERROR, filename=log_file)

    # TODO(minigb): Make this dir_dict to be sharable
    dir_dict = {
        'logits': f'{inference_type}/logits'
    }

    for dir_path in dir_dict.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


if __name__ == '__main__':
    init('multi')

    # cqt5
    compute_multi("audio_midi_cqt5_ps_v5", mode="basic")
    # pr5
    compute_multi("audio_midi_pianoroll_ps_5_v4", mode="basic")
    # mm5
    compute_multi("audio_midi_multi_ps_v5", mode="basic")
    # multirank5
    compute_multi("audio_midi_cqt5multiranking_v10", mode="multirank")
    compute_multi("audio_midi_pr5multiranking_v10", mode="multirank")
    # era
    compute_multi("audio_midi_cqt5era_v1", mode="era")
    compute_multi("audio_midi_pr5era_v1", mode="era")