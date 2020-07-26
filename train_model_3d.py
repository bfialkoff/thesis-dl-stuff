from pathlib import Path
from datetime import datetime
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

from utils.resnet3d import Resnet3DBuilder
from utils.callbacks.plotter_callback import PlotterCallback
from utils.callbacks.classification_callback import ClassificationCallback
from utils.generators.sequential_emg_generator import EmgImageGenerator
from utils.losses.numpy_losses import *
from utils.preprocessing.data_preprocessor import DataPreprocessor

num_imfs = 4
window_sample_length = 256



def _get_cpu_model(input_size=None, activation=None, initial_weights=None, is_corruption=False):
    model = Sequential()

    model.add(
        Conv3D(32, kernel_size=(32, 3, 3), activation='relu', input_shape=(window_sample_length, 8, 8, num_imfs)))
    model.add(Conv3D(16, (16,3,3), padding='Same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling3D())
    # model.add(Dropout(0.25))

    model.add(Conv3D(32, (16,3,3), padding='Same', activation='relu'))
    model.add(Conv3D(64, (8,3,3), padding='Same', activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D())
    #model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(1, activation=None))
    if is_corruption:
        loss = keras.losses.binary_crossentropy
        print('bce')
    else:
        loss = keras.losses.mean_squared_error
        print('mse')
    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(1e-3)
                  )
    if initial_weights:
        model.load_weights(initial_weights)
    return model

def get_cpu_model(input_size=None, activation=None, initial_weights=None, is_corruption=False):
    input_shape = (window_sample_length, 8, 8, num_imfs)
    model = Resnet3DBuilder.build_resnet_50((window_sample_length, 8, 8, num_imfs), 1)
    if is_corruption:
        loss = keras.losses.binary_crossentropy
        print('bce')
    else:
        loss = keras.losses.mean_squared_error
        print('mse')

    if num_gpus > 1:
        model = keras.utils.multi_gpu_utils.multi_gpu_model(model, gpus=num_gpus)

    if initial_weights:
        model.load_weights(initial_weights)

    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(1e-4)
                  )
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
    train_path = Path(__file__, '..', 'files', 'emd_dl_train_annotations.csv').resolve()
    val_path = Path(__file__, '..', 'files', 'emd_dl_val_annotations.csv').resolve()
    train_preprocessor = DataPreprocessor()
    #val_preprocessor = DataPreprocessor()
    train_annotations = pd.read_csv(train_path)
    val_annotations = pd.read_csv(val_path)
    train_annotations = train_annotations.fillna(0)
    val_annotations = val_annotations.fillna(0)


    date_id = datetime.now().strftime('%Y%m%d%H%M')
    #date_id ='202007221536'
    experiment_dir = Path('/media/adam/e46d6141-876f-4b0c-90da-9e9e217986f2/betzalel_personal/').joinpath('files', 'deep_learning', date_id).resolve()

    initial_epoch = 0
    initial_weights = experiment_dir.joinpath('weights', f'{initial_epoch}.hdf5').resolve()
    #initial_weights = initial_weights if num_gpus else None
    activation = None

    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    batch_size = 32
    input_size = (224, 224, 3) if (num_gpus and False) else [None]
    is_corruption = False

    train_emg_gen = EmgImageGenerator(train_annotations.copy(), batch_size, scaler=None, input_size=input_size[0], num_imfs=num_imfs)
    train_scaler = train_emg_gen.scaler
    callback_train_emg_gen = EmgImageGenerator(train_annotations.copy(), batch_size, scaler=train_scaler, input_size=input_size[0], num_imfs=num_imfs)
    callback_val_emg_gen = EmgImageGenerator(val_annotations, batch_size, scaler=train_scaler, input_size=input_size[0], num_imfs=num_imfs)


    model = get_model(activation=activation, initial_weights=initial_weights, input_size=input_size)
    if is_corruption:
        train_gen = train_emg_gen.corruption_train_generator()
        loss = numpy_bce
        p = ClassificationCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    else:
        train_gen = train_emg_gen.train_generator()
        loss = numpy_mse
        p = PlotterCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    model_checkpoint = get_model_checkpoint(experiment_dir)

    callbacks = [p, model_checkpoint]
    model.fit_generator(train_gen,
                        steps_per_epoch=train_emg_gen.num_samples // train_emg_gen.batch_size,
                        epochs=100000,
                        callbacks=callbacks,
                        verbose=1,
                        initial_epoch=initial_epoch)
