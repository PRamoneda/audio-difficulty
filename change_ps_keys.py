import os
from pathlib import Path
from utils import *
from collections import OrderedDict


def init(inference_type, fs):
    global DIR_DICT
    DIR_DICT = {
        'mp3': f'{inference_type}/mp3',
        'midi': f'{inference_type}/midi',
        'pr': f'{inference_type}/pr{fs}',
        'cqt': f'{inference_type}/cqt{fs}',
        'cqt_full': f'{inference_type}/cqt_full',
        'tmp': 'tmp'
    }

if __name__ == "__main__":
    init('multi', 5)
    dir_name = Path('benchmark_multiperformances')

    for file_name in os.listdir(dir_name):
        if not file_name.endswith('.json'):
            continue
        
        json_filename = dir_name / file_name
        data = load_json(f'{dir_name}/{file_name}')
        first_key = list(data.keys())[0]
        assert '_from_ps' in first_key, f'Error: {first_key} does not contain "from_ps"'
        original_key = first_key.replace('_from_ps', '')

        for dirs in DIR_DICT.values():
            for file_name in os.listdir(dirs):
                # move the file to have the new name
                if original_key in file_name:
                    new_name = file_name.replace(original_key, first_key)
                    os.rename(f'{dirs}/{file_name}', f'{dirs}/{new_name}')