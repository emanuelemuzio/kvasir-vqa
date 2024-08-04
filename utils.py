import json

def get_model(PreTrainedConfig, model_name):
    return PreTrainedConfig.from_pretrained(model_name)

def list_of_strings(arg):
    return arg.split(',')

""""

Simple function for loading the json files, which contains data about the annotated answers and questions

"""

def load_json(path: str):
    f = open(path)
    data = json.load(f)
    return data

"""

Since there are abstract scenese with binary Q/A, normal abstract scenes and real scenes, 
We differentiate which data we are going to use by passing the relative keys as arguments to the testing script

"""

def map_data(keys: list):
    data_mapping = load_json('data_mapping.json')

    assert(len(keys) > 0)

    while len(keys) > 0:
        key = keys.pop(0)
        if key in data_mapping.keys():
            data_mapping = data_mapping[key]
            if len(keys) == 0:
                return data_mapping
        else:
            keys.append(key)

test = map_data(['real'])