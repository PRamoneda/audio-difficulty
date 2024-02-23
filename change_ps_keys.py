import os
from pathlib import Path
from utils import *
from collections import OrderedDict

if __name__ == "__main__":
    dir_name = Path('benchmark_multiperformances')
    for file_name in os.listdir(dir_name):
        if not file_name.endswith('.json'):
            continue
        
        json_filename = dir_name / file_name
        data = load_json(f'{dir_name}/{file_name}')
        keys, values = list(data.keys()), list(data.values())
        keys[0] = keys[0] + '_from_ps'
        data_new = {key:value for key, value in zip(keys, values)}
        save_json(data_new, json_filename, sort_keys = False)