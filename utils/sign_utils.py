import os
from random import shuffle
import math
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

BASE_PATH = os.path.abspath('.') + '/data'
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
data_folder = os.listdir(BASE_PATH)[1:][0]

def read_files(data_folder):
    data = []
    sub_folders = os.listdir(BASE_PATH + '/' + str(data_folder))[1:]
    for folder in sub_folders:
        signs = os.listdir(BASE_PATH + '/' + str(data_folder) + '/' + folder)
        for sign in signs:
            image_path = (folder, BASE_PATH + '/' + str(data_folder) + '/' + folder + '/' + sign)
            data.append(image_path)
    return data

def split_files(data):
    ratio = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO
    assert(ratio == 1.0)
    shuffle(data)
    train_split = math.floor(len(data) * TRAIN_RATIO)
    valid_split = math.floor(len(data) * VALIDATION_RATIO)
    test_split = math.floor(len(data) * TEST_RATIO)
    valid_index = train_split + valid_split
    test_index = valid_index + test_split
    training = data[:train_split]
    validation = data[train_split:valid_index]
    testing = data[valid_index:test_index]
    return training, validation, testing

def create_data_set(data, path):
    if not os.path.isdir(path):
        os.makedirs(path)
        for file in tqdm(data):
            sign_path = os.path.join(path, file[0])
            if os.path.isdir(sign_path):
                image = cv2.imread(file[1])
                image_path = sign_path + '/' + file[1].split('/')[-1]
                cv2.imwrite(image_path, image)
            else:
                os.makedirs(sign_path)
                image = cv2.imread(file[1])
                image_path = sign_path + '/' + file[1].split('/')[-1]
                cv2.imwrite(image_path, image)
    else:
        for file in tqdm(data):
            sign_path = os.path.join(path, file[0])
            if os.path.isdir(sign_path):
                image = cv2.imread(file[1])
                image_path = sign_path + '/' + file[1].split('/')[-1]
                cv2.imwrite(image_path, image)
            else:
                os.makedirs(sign_path)
                image = cv2.imread(file[1])
                image_path = sign_path + '/' + file[1].split('/')[-1]
                cv2.imwrite(image_path, image)


data = read_files(data_folder)
training, validation, testing = split_files(data)

print ('trainning size', len(training))
print ('validation size', len(validation))
print ('testing size', len(testing))

train_path = BASE_PATH + '/' + 'training_set'
valid_path = BASE_PATH + '/' + 'validation_set'
test_path = BASE_PATH + '/' + 'testing_set'
print ('intializing creating datasets...............')
create_data_set(training, train_path)
create_data_set(validation, valid_path)
create_data_set(testing, test_path)