from pathlib import Path
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import Model
from keras.utils.multi_gpu_utils import multi_gpu_model
from classification_models import Classifiers
from keras import backend as K

from plotter_callback import PlotterCallback
from classification_callback import ClassificationCallback
from image_builder import EmgImageGenerator
from numpy_losses import *




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

def get_cpu_model(input_size=None,activation=None, initial_weights=None, is_corruption=False):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu', input_shape=(8, 8, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=activation))

    if is_corruption:
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.mean_squared_error
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam()
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
    train_path = Path(__file__, '..', 'files', 'dl_train_annotations.csv').resolve()
    val_path = Path(__file__, '..', 'files', 'dl_val_annotations.csv').resolve()

    date_id = datetime.now().strftime('%Y%m%d%H%M')
    experiment_dir = Path(__file__, '..', 'files', 'deep_learning', date_id).resolve()
    initial_weights = Path(__file__, '..', 'files', 'deep_learning', '202006021715_corruption', 'weights', '07.hdf5').resolve()
    initial_weights = None if num_gpus else initial_weights
    activation = None

    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    batch_size = 64 if num_gpus else 256
    is_corruption = False
    train_emg_gen = EmgImageGenerator(train_path, batch_size)
    callback_train_emg_gen = EmgImageGenerator(train_path, batch_size)
    callback_val_emg_gen = EmgImageGenerator(val_path, batch_size)


    model = get_model(activation=activation, initial_weights=initial_weights)
    if is_corruption:
        loss = numpy_bce
        p = ClassificationCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    else:
        # todo think about non-linear activations here, like 2*sigmoid - 1 or tanh
        #  basically think about how the force is scaled and choose try an activation function with a corresponding image
        loss = numpy_mse
        p = PlotterCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    model_checkpoint = get_model_checkpoint(experiment_dir)

    callbacks = [p, model_checkpoint]
    model.fit_generator(train_emg_gen.corruption_train_generator(),
                        steps_per_epoch=train_emg_gen.num_samples // train_emg_gen.batch_size,
                        epochs=50,
                        callbacks=callbacks,
                        verbose=1)