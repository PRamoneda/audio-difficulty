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
        # Assuming the dictionary has at least one key-value pair
        first_key = list(data.keys())[0]
        new_key = first_key + '_from_ps'
        data_new = OrderedDict([(new_key, data.pop(first_key))] + list(data.items()))
        save_json(data_new, json_filename)