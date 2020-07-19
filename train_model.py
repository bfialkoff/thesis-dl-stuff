from pathlib import Path
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import Model
from keras.utils.multi_gpu_utils import multi_gpu_model
from classification_models import Classifiers
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import pandas as pd

from utils.callbacks.plotter_callback import PlotterCallback
from utils.callbacks.classification_callback import ClassificationCallback
from utils.generators.new_image_builder import EmgImageGenerator, permute_axes_subtract
from utils.losses.numpy_losses import *
from utils.preprocessing.data_preprocessor import DataPreprocessor

num_imfs = 4


def get_gpu_model(input_size=None, activation=None, initial_weights=None, is_corruption=False):
    ResNet50v2, preprocess_input = Classifiers.get('resnet50v2')
    model = ResNet50v2(input_shape=input_size, weights='imagenet', classes=1, include_top=False, pooling='avg')
    model_inputs = model.inputs
    model_outsputs = model.output
    model_outsputs = Dense(128, activation='relu')(model_outsputs)
    model_outsputs = Dense(32, activation='relu')(model_outsputs)
    model_outsputs = Dense(1, activation=activation)(model_outsputs)
    model = Model(model_inputs, model_outsputs)
    model = multi_gpu_model(model, gpus=2)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam()
                  )
    return model

def get_cpu_model(input_size=None, activation=None, initial_weights=None, is_corruption=False):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, num_imfs)))
    model.add(Conv2D(4, (3, 3), activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation=keras.activations.selu))
    if is_corruption:
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.mean_squared_error
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(1e-3)
                  )
    if initial_weights:
        model.load_weights(initial_weights)
    return model

def _get_cpu_model(input_size=None, activation=None, initial_weights=None, is_corruption=False):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, num_imfs), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation=None))
    if is_corruption:
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.mean_squared_error
    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(1e-2)
                  )
    if initial_weights:
        model.load_weights(initial_weights)
    return model


num_gpus = len(K.tensorflow_backend._get_available_gpus())
get_model = get_gpu_model if num_gpus else get_cpu_model

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
    #train_preprocessor = DataPreprocessor()
    #val_preprocessor = DataPreprocessor()
    train_annotations = pd.read_csv(train_path)
    val_annotations = pd.read_csv(val_path)
    train_annotations = train_annotations.fillna(0)
    val_annotations = val_annotations.fillna(0)
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaler.fit(train_annotations['force'].values.reshape(-1, 1))


    date_id = datetime.now().strftime('%Y%m%d%H%M')
    experiment_dir = Path(__file__, '..', 'files', 'deep_learning', date_id).resolve()
    #initial_weights = experiment_dir.joinpath('weights', '999.hdf5').resolve()
    initial_weights = None
    #initial_weights = initial_weights if num_gpus else None
    activation = None

    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    batch_size = 64 if num_gpus else 256
    input_size = (224, 224, 3) if num_gpus else [None]
    is_corruption = False

    train_emg_gen = EmgImageGenerator(train_annotations.copy(), batch_size, scaler=train_scaler, input_size=input_size[0], num_imfs=num_imfs)
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
                        epochs=2000,
                        callbacks=callbacks,
                        verbose=1,
                        initial_epoch=0)
