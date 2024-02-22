import os
from collections import defaultdict

from scipy.stats import kendalltau
from statistics import mean, stdev

from utils import *


def compute_tauc(bin_file_name):
    data = load_binary(bin_file_name)
    pred_list, true_list = [], []
    for _, data_pr in data.items():
        pred_list.append(int(data_pr['pred']))
        true_list.append(int(data_pr['true']))
    tau_c_value = kendalltau(x=true_list, y=pred_list).statistic
    return tau_c_value


def mean_std(data):
    mean_val = mean(data)
    std_val = stdev(data)

    return {"data": data, "mean": mean_val, "std": std_val}


def print_for_paper(result):
    model_type_dict = {
        # single task
        'audio_midi_cqt5_ps_v5' : 'CQT5',
        'audio_midi_pianoroll_ps_5_v4' : 'PR5',
        'audio_midi_multi_ps_v5' : "MM",

        # multi task with era
        'audio_midi_cqt5era_v1' : 'CQT5',
        'audio_midi_pr5era_v1' : 'PR5',

        # multi task with multi ranking
        'audio_midi_cqt5multiranking_v10' : 'CQT5',
        'audio_midi_pr5multiranking_v10' : 'PR5'
    }

    model_types = list(model_type_dict.keys())
    assert len(model_types) == len(result), f"{len(model_types)} != {len(result)}"
    assert set(model_types) == set(result.keys()), f"{model_types} != {set(result.keys())}"

    for model_type in model_types:
        data = result[model_type]
        formatted_mean = f"{data['mean']:.3f}".lstrip('0')
        formatted_std = f"{data['std']:.3f}".lstrip('0')

        print(f"& {model_type_dict[model_type]}$_{{5}}$ & {formatted_mean} ({formatted_std}) \\\\")


if __name__ == "__main__":
    result = defaultdict(list)
    for filename in os.listdir("multi/logits"):
        if not filename.endswith(".bin"):
            continue

        model_type = filename.split("_split")[0]
        result[model_type].append(compute_tauc(f"multi/logits/{filename}"))

    for model_type, data in result.items():
        result[model_type] = mean_std(data)

    save_json(result, "multi_tauc.json")
    print_for_paper(result)
