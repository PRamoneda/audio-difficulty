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
    for model_type, data in result.items():
        print(f"{model_type} & {data['mean']:.4f} & {data['std']:.4f} \\\\")


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
