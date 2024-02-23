import os
from utils import *

if __name__ == "__main__":
    for file_name in os.listdir('benchmark_multiperformances'):
        if not file_name.endswith('.json'):
            continue
        data = load_json(f'benchmark_multiperformances/{file_name}')
        # Assuming the dictionary has at least one key-value pair
        first_key = list(data.keys())[0]
        new_key = first_key + '_from_ps'
        data[new_key] = data.pop(first_key)
        save_json(data, f'benchmark_multiperformances/{file_name}')