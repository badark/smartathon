import os, sys
import torch
from PIL import Image
import cv2
from IPython.display import display
from collections import defaultdict
import random
import pandas as pd
import json
from utils import csv_to_dict

if os.name == 'nt':
    cwd = os.getcwd()
    root_dir, src_dir = os.path.split(cwd)
    train_csv_path = root_dir + '\\data\\train.csv'
else:
    train_csv_path = 'data/train.csv'

csv_data = pd.read_csv(train_csv_path)
# remove the BAD_STREETLIGHT example
csv_data = csv_data[csv_data['name'] != 'BAD_STREETLIGHT']
dict_data = csv_to_dict(csv_data=csv_data)

dict_data_keys = list(dict_data.keys())
# we shuffle the list
random.seed(14)
random.shuffle(dict_data_keys)
num_records = len(dict_data_keys)
num_val_test = num_records//10

test_data = {k: dict_data[k] for k in dict_data_keys[:num_val_test]}
val_data = {k: dict_data[k] for k in dict_data_keys[num_val_test:2*num_val_test]}
train_data = {k: dict_data[k] for k in dict_data_keys[2*num_val_test:]}

if os.name == 'nt':
    json.dump(train_data, open(root_dir + '\\data\\train_split.json', 'w'))
    json.dump(val_data, open(root_dir + '\\data\\val_split.json', 'w'))
    json.dump(test_data, open(root_dir + '\\data\\test_split.json', 'w'))
else:
    json.dump(train_data, open('data/train_split.json', 'w'))
    json.dump(val_data, open('data/val_split.json', 'w'))
    json.dump(test_data, open('data/test_split.json', 'w'))