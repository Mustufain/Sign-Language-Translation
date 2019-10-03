from __future__ import absolute_import, division, print_function, unicode_literals
import keras
from keras.applications import InceptionV3
from keras.layers import Dense,  GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os


# ! pip install -U --force-reinstall --no-dependencies git+https://github.com/datumbox/keras@bugfix/trainable_bn
#  tensorboard --logdir="./logs/inception_logs"

class CustomInceptionV3(object):

    def __init__(self, base_dir, version):
        self.base_dir = base_dir
        self.seed = 1234
        self.checkpoint_path = 'checkpoint_model/inceptionv3.h5'
        self.export_path = 'export_model/'
        self.batch_size = 64
        self.epochs = 1
        self.n_classes = 10
        self.lr = 0.0001
        self.visualization_folder = 'visualization/'
        self.version = version

        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    def load_dataset(self):
        train_dataset = h5py.File(self.base_dir + 'train.h5', 'r')
        train_dataset = np.array(train_dataset["data"][:], dtype=np.float32)
        valid_dataset = h5py.File(self.base_dir + 'validation.h5', 'r')
        valid_dataset = np.array(valid_dataset["data"][:], dtype=np.float32)
        test_dataset = h5py.File(self.base_dir + 'test.h5', 'r')
        test_dataset = np.array(test_dataset["data"][:], dtype=np.float32)
        train_dataset /= 255.0
        valid_dataset /= 255.0
        test_dataset /= 255.0
        return train_dataset, valid_dataset, test_dataset


    def load_labels(self):
        train_labels = h5py.File(self.base_dir + 'train_labels.h5', 'r')
        valid_labels = h5py.File(self.base_dir + 'valid_labels.h5', 'r')
        test_labels = h5py.File(self.base_dir + 'test_labels.h5', 'r')
        train_labels = np.array(train_labels['labels'][:], dtype=np.int32)
        valid_labels = np.array(valid_labels['labels'][:], dtype=np.int32)
        test_labels = np.array(test_labels['labels'][:], dtype=np.int32)
        train_labels = keras.utils.to_categorical(train_labels)
        valid_labels = keras.utils.to_categorical(valid_labels)
        test_labels = keras.utils.to_categorical(test_labels)
        return train_labels, valid_labels, test_labels

    def create_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.n_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        adam = Adam(lr=self.lr)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def run(self):
        train_dataset, valid_dataset, test_dataset = self.load_dataset()
        train_labels, valid_labels, test_labels = self.load_labels()
        train_datagen = ImageDataGenerator()
        valid_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        nb_train_samples = train_dataset.shape[0]
        nb_validation_samples = valid_dataset.shape[0]
        nb_test_samples = test_dataset.shape[0]

        train_generator = train_datagen.flow(x=train_dataset,
                                             y=train_labels,
                                             batch_size=self.batch_size,
                                             seed=self.seed)

        valid_generator = valid_datagen.flow(x=valid_dataset,
                                             y=valid_labels,
                                             batch_size=self.batch_size,
                                             seed=self.seed)

        test_generator = test_datagen.flow(x=test_dataset,
                                           y=test_labels,
                                           batch_size=self.batch_size,
                                           seed=1234)

        # Save the model after every epoch.
        mc_top = ModelCheckpoint(self.checkpoint_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
        # Save the TensorBoard logs.
        tb = TensorBoard(log_dir='./logs/inception_logs', histogram_freq=0, write_graph=False, write_images=False)
        model = self.create_model()
        classifier = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // self.batch_size,
            epochs=self.epochs,
            validation_data=valid_generator,
            validation_steps=nb_validation_samples // self.batch_size,
            callbacks=[mc_top, tb],
            verbose=1)
        score = model.evaluate_generator(test_generator, steps=nb_test_samples, verbose=1)
        print ('Accuracy on test set', score[1])
        model_history = classifier.history
        self.plot_loss(model_history)
        self.plot_accuracy(model_history)


    def plot_loss(self, model):
        path = self.visualization_folder + 'inception_loss.png'
        plt.figure()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.plot(model['loss'])
        plt.plot(model['val_loss'])
        plt.legend(['training_loss', 'validation_loss'])
        plt.savefig(path)

    def plot_accuracy(self, model):
        path = self.visualization_folder + 'inception_accuracy.png'
        plt.figure()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.plot(model['acc'])
        plt.plot(model['val_acc'])
        plt.legend(['training_accuracy', 'validation_accuracy'])
        plt.savefig(path)

    def export_model(self):
        version_path = self.export_path + 'inception_v3/' + str(self.version)

        if not os.path.exists(version_path):
            os.makedirs(version_path)

        model_path = self.checkpoint_path
        tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
        model = tf.keras.models.load_model(model_path)

        # Fetch the Keras session and save the model
        # The signature definition is defined by the input and output tensors
        # And stored with the default serving key
        with tf.keras.backend.get_session() as sess:
            tf.saved_model.simple_save(
                sess,
                version_path,
                inputs={'input_image': model.input},
                outputs={t.name: t for t in model.outputs})

