from pathlib import Path
from datetime import datetime
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras import Model
from keras.utils.multi_gpu_utils import multi_gpu_model
from classification_models import Classifiers
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import pandas as pd

from utils.callbacks.omp_plotter_callback import PlotterCallback
from utils.generators.regression_param_generator import EmgImageGenerator
from utils.losses.numpy_losses import *

feature_len = 30
num_channels = 8

def _get_cpu_model(input_size=None, activation=None, initial_weights=None, is_corruption=False):
    model = Sequential()
    #model.add(Conv2D(16, kernel_size=(6, 6), activation='relu', input_shape=(8, 8, num_imfs), padding='same'))
    #model.add(Conv2D(4, (4, 4), activation='relu', padding='same'))
    model.add(Flatten(input_shape=input_size))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(9, activation=None))

    loss = keras.losses.mean_squared_error
    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(1e-2)
                  )
    if initial_weights:
        model.load_weights(initial_weights)
    return model

def get_cpu_model(input_size=None, activation=None, initial_weights=None, is_corruption=False):
    Resnet50v2, preprocessor = Classifiers.get('resnet50v2')
    model = Resnet50v2(input_shape=input_size, include_top=False, pooling='max')
    model_inputs = model.inputs
    model_outputs = model.output
    model_outputs = Dense(16, activation='relu')(model_outputs)
    model_outputs = Dense(9, activation=None)(model_outputs)
    model = Model(model_inputs, model_outputs)
    loss = keras.losses.mean_squared_error
    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(1e-5)
                  )
    if initial_weights:
        model.load_weights(initial_weights)
    return model

num_gpus = len(K.tensorflow_backend._get_available_gpus())
get_model = get_cpu_model

def get_model_checkpoint(experiment_dir):
    weights_path = experiment_dir.joinpath('weights', '{epoch:02d}.hdf5')
    if not weights_path.parents[0].exists():
        weights_path.parents[0].mkdir(parents=True)
    weights_path = str(weights_path)
    model_checkpoint = ModelCheckpoint(weights_path)
    return model_checkpoint

if __name__ == '__main__':
    # todo ideal would be to define the train metric as tf metrics in order to avoid reiterating over the train data
    train_features_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'train_features.csv').resolve()
    train_labels_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'train_labels.csv').resolve()
    val_features_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'val_features.csv').resolve()
    val_labels_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'val_labels.csv').resolve()


    date_id = datetime.now().strftime('%Y%m%d%H%M')
    experiment_dir = Path(__file__, '..').joinpath('files', 'deep_learning', date_id).resolve()
    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    initial_epoch = 0
    batch_size = 32
    #initial_weights = experiment_dir.joinpath('weights', f'{initial_epoch}.hdf5').resolve()
    #initial_weights = initial_weights if num_gpus else None
    initial_weights = None
    activation = None
    resize_to = 80
    train_emg_gen = EmgImageGenerator(train_features_path, train_labels_path, batch_size=batch_size, scaler=None, resize=resize_to, mirror=True)
    train_scaler = None # train_emg_gen.scaler
    callback_train_emg_gen = EmgImageGenerator(train_features_path, train_labels_path, batch_size=batch_size, scaler=train_scaler,  resize=resize_to, mirror=True)
    callback_val_emg_gen = EmgImageGenerator(val_features_path, val_labels_path, batch_size=batch_size, scaler=train_scaler, resize=resize_to, mirror=True)

    #input_shape = (feature_len, num_channels)
    input_shape = (*train_emg_gen.mirror_shape, 3)
    model = get_model(input_size=input_shape, activation=activation, initial_weights=initial_weights)

    train_gen = train_emg_gen.train_generator()
    loss = numpy_mse
    p = PlotterCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    #model_checkpoint = get_model_checkpoint(experiment_dir)

    callbacks = [p]#, model_checkpoint]
    model.fit_generator(train_gen,
                        steps_per_epoch=train_emg_gen.num_samples // train_emg_gen.batch_size,
                        epochs=100000,
                        callbacks=callbacks,
                        verbose=1,
                        initial_epoch=initial_epoch)
