from statistics import mean, stdev
from scipy.stats import kendalltau

from utils import *


def compute_tauc(bin_file_name):
    data = load_binary(bin_file_name)
    tau_c = []
    pred_list, true_list = [], []
    for key, data_pr in data.items():
        pred_list.append(int(data_pr['pred']))
        true_list.append(int(data_pr['true']))
    tau_c_value = kendalltau(x=true_list, y=pred_list).statistic
    print(f"{tau_c_value:.3f}")


if __name__ == "__main__":
    for split in range(5):
        compute_tauc(f"multi/logits/audio_midi_cqt5_ps_v5_split_{split}.bin")