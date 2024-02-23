import json
import pickle


def load_json(name_file):
    data = None
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data


def save_json(dictionary, name_file, sort_keys=True):
    with open(name_file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=sort_keys, indent=4)


def prediction2label(pred):
    """Convert ordinal predictions to class labels, e.g.

    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data


def save_binary(dictionary, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


def remove_strange_characters(string):
    STRANGE_CHARACTERS = ["#", "'", '"', '?', '!', "_", "\u2013", "/", ":", "&", "[", "]", "\u00ef", "\u00ea", "\u00e9",
                        "\u00c9", "\u2014", "\u201c", "\u201d", "\u00b4", "\u00ed", "\u00e8", "\u00ed", "\u2018", "\u00c5",
                        "\u0002", "\u00e1", "\u00f3", "\u2019"]
    
    for character in STRANGE_CHARACTERS:
        string = string.replace(character, "")
        
    return string