import os
from random import shuffle
import math
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
import h5py
import json



class DataPreperation(object):

    def __init__(self, base_path, data):
        self.base_path = base_path
        self.train_ratio = 0.8
        self.validation_ratio = 0.1
        self.test_ratio = 0.1
        self.data_folder = [x for x in os.listdir(self.base_path)[1:] if x == data][0]
        self.n_classes = len(os.listdir(self.base_path+self.data_folder)[1:]) # due to .DS_Store

    def read_files(self):
        data = []
        sub_folders = os.listdir(self.base_path + str(self.data_folder))
        if '.DS_Store' in sub_folders:
            sub_folders = sub_folders[1:]
        for folder in sub_folders:
            signs = os.listdir(self.base_path + str(self.data_folder) + '/' + folder)
            for sign in signs:
                image_path = (folder, self.base_path + str(self.data_folder) + '/' + folder + '/' + sign)
                data.append(image_path)
        return data

    def split_files(self, data):
        ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        assert(ratio == 1.0)
        shuffle(data)
        train_split = math.floor(len(data) * self.train_ratio)
        valid_split = math.floor(len(data) * self.validation_ratio)
        test_split = math.floor(len(data) * self.test_ratio)
        valid_index = train_split + valid_split
        test_index = valid_index + test_split
        training = data[:train_split]
        validation = data[train_split:valid_index]
        testing = data[valid_index:test_index]
        self.data_sanity_check(train=training, valid=validation, test=testing)
        return training, validation, testing

    def create_data_set(self, data, path):
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

    def create_h5_dataset(self, dir, fname):
        data = []
        labels = []
        sub_folders = os.listdir(dir)
        filename = fname
        for folder in tqdm(sub_folders):
            images = os.listdir(dir + folder)
            for image in images:
                image_path = dir + folder + '/' + image
                img = Image.open(image_path)
                image_data = np.asarray(img)
                data.append(image_data)
                labels.append(image.split('_')[0])
        dataset = np.asarray(data)
        out = h5py.File(filename, "w")
        out.create_dataset('data', data=dataset)
        out.close()

    def create_labels(self, dir, fname):
        folders = os.listdir(dir)
        labels = []
        for f in tqdm(folders):
            subfolder = os.listdir(dir + f)
            for images in subfolder:
                labels.append(images.split('_')[0])
        label = np.asarray(labels)
        label = np.unique(label, return_inverse=True)[1]
        out = h5py.File(fname, "w")
        out.create_dataset('labels', data=label)
        out.close()

    def save_id(self, path, fname):
        fname_path = self.base_path + fname
        id = {}
        labels = os.listdir(path)
        uniques = np.unique(labels)
        for index in range(self.n_classes):
            value = uniques[index]
            id[index] = value
        with open(fname_path, 'w') as f:
            json.dump(id, f)
        f.close()

    def get_train_path(self):
        return self.base_path + 'training_set/'

    def get_valid_path(self):
        return self.base_path + 'validation_set/'

    def get_test_path(self):
        return self.base_path + 'testing_set/'

    def get_base_path(self):
        return self.base_path

    def data_sanity_check(self, train, valid, test):
        """

        :param train:
        :param valid:
        :param test:
        :return:
        """

        assert len(list(set(train).intersection(valid))) == 0
        assert len(list(set(train).intersection(test))) == 0
        assert len(list(set(test).intersection(valid))) == 0
