import os
from collections import defaultdict

from scipy.stats import kendalltau
from statistics import mean, stdev
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from argparse import ArgumentParser

from utils import *


MODEL_TYPE_DICT = {
    # single task
    'audio_midi_cqt5_ps_v5' : 'CQT',
    'audio_midi_pianoroll_ps_5_v4' : 'PR',
    'audio_midi_multi_ps_v5' : "MM",

    # multi task with era
    'audio_midi_cqt5era_v1' : 'CQT',
    'audio_midi_pr5era_v1' : 'PR',

    # multi task with multi ranking
    'audio_midi_cqt5multiranking_v10' : 'CQT',
    'audio_midi_pr5multiranking_v10' : 'PR'
}


def get_mse_macro(y_true, y_pred):
    mse_each_class = []
    for true_class in set(y_true):
        tt, pp = zip(*[[tt, pp] for tt, pp in zip(y_true, y_pred) if tt == true_class])
        mse_each_class.append(mean_squared_error(y_true=tt, y_pred=pp))
    return mean(mse_each_class)


def compute_metrics(dir_path, model_type, ps_only):
    mse, acc, tau_c = [], [], []
    for split in range(5):
        bin_file_name = f'{dir_path}/{model_type}_split_{split}.bin'
        data = load_binary(bin_file_name)

        pred_list, true_list = [], []
        for data_key, data_pr in data.items():
            if ps_only and '_from_ps' not in data_key:
                continue
            true_list.append(int(data_pr['true']))
            pred_list.append(int(data_pr['pred']))
        if ps_only:
            assert len(true_list) == 55, f"{len(true_list)} != 55"
        
        mse.append(get_mse_macro(true_list, pred_list))
        acc.append(balanced_accuracy_score(true_list, pred_list))
        tau_c.append(kendalltau(x=true_list, y=pred_list).statistic)

    metrics = {
        "mse": mse,
        "acc": acc,
        "tau_c": tau_c
    }

    return metrics


def print_for_paper(result):
    model_types = list(MODEL_TYPE_DICT.keys())
    assert len(model_types) == len(result), f"{len(model_types)} != {len(result)}"
    assert set(model_types) == set(result.keys()), f"{model_types} != {set(result.keys())}"
    decimal_points = {
        'mse': (2, 2),
        'acc': (1, 2),
        'tau_c': (3, 3)
    }

    for model_type in model_types:
        table_line = f"& {MODEL_TYPE_DICT[model_type]}$_{{5}}$"
        split_results = result[model_type]
        for metric_name, data in split_results.items(): # mse, acc, tau_c
            mean_val, std_val = mean(data), stdev(data)
            if metric_name == "acc":
                mean_val *= 100
                std_val *= 100
            formatted_mean = f"{mean_val:.{decimal_points[metric_name][0]}f}".lstrip('0')
            formatted_std = f"{std_val:.{decimal_points[metric_name][1]}f}".lstrip('0')
            table_line += f" & {formatted_mean}({formatted_std})"

        table_line += " \\\\"
        print(table_line)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inference_type", type=str, required=True)
    parser.add_argument("--ps_only", action='store_true')
    args = parser.parse_args()

    inference_type = args.inference_type
    ps_only = args.ps_only
    
    dir_path = f"{inference_type}/logits"
    result = defaultdict()

    for model_type in MODEL_TYPE_DICT.keys():
        result[model_type] = compute_metrics(dir_path, model_type, ps_only)

    print(inference_type, ps_only)
    save_json(result, f"cache/{inference_type}_{ps_only}_metrics.json")
    print_for_paper(result)