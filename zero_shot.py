from statistics import mean, stdev

from scipy.stats import kendalltau

from utils import load_binary


def zero_shot(model):
    tau_c = []
    for split in range(5):
        data = load_binary(f"hv/logits/{model}_split_{split}.bin")
        preds, trues = [], []
        for k, v in data.items():
            # print(k, v["pred"], v["true"])
            preds.append(v["pred"])
            trues.append(v["true"])
        tau_c.append(kendalltau(preds, trues).statistic)
    print(f"Model {model} & {mean(tau_c):.3f}({stdev(tau_c):.3f})")


if __name__ == '__main__':
    # cqt5
    zero_shot("audio_midi_cqt5_ps_v5")
    # pr5
    zero_shot("audio_midi_pianoroll_ps_5_v4")
    # mm5
    zero_shot("audio_midi_multi_ps_v5")
    # multirank5
    zero_shot("audio_midi_cqt5multiranking_v10")
    zero_shot("audio_midi_pr5multiranking_v10")
    # era
    zero_shot("audio_midi_cqt5era_v1")
    zero_shot("audio_midi_pr5era_v1")