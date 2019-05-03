import datetime
import os
import keras.backend as K
import numpy as np
from keras.layers import Conv2D, Lambda, Input, Flatten, Dense, Multiply
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from agents.constants import ATARI_INPUT_SHAPE


def build_model(obs_shape, action_size):

    frames_input = Input(obs_shape, name='frames')
    actions_input = Input((action_size,), name='actions')

    normalized = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
    conv_1 = Conv2D(16, 8, strides=4, activation='relu')(normalized)
    conv_2 = Conv2D(32, 4, strides=2, activation='relu')(conv_1)
    conv_flatten = Flatten()(conv_2)
    hidden = Dense(256, activation='relu')(conv_flatten)
    output = Dense(action_size)(hidden)

    filtered = Multiply(name='q_val')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered)
    model.summary()

    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)

    return model


def build_larger_model(obs_shape, action_size):

    frames_input = Input(obs_shape, name='frames')
    actions_input = Input((action_size,), name='actions')

    x = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
    x = Conv2D(32, 8, strides=4, activation='relu')(x)
    x = Conv2D(64, 4, strides=2, activation='relu')(x)
    x = Conv2D(64, 3, strides=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)

    output = Dense(action_size)(x)
    filtered = Multiply(name='q_val')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered)
    model.summary()

    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)

    return model


def build_dense_model(obs_size, action_size):
    obs = Input((obs_size,), name='input')
    actions = Input((action_size,), name='actions')

    x = Dense(obs_size, activation='relu')(obs)
    #x = Dense(32, activation='relu')(x)
    #x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(action_size)(x)

    filtered = Multiply(name='q_val')([output, actions])

    model = Model(inputs=[obs, actions], outputs=filtered)
    model.summary()

    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)

    return model


def get_curr_datetime():
    return str(datetime.datetime.now().strftime("%Y-%m-%d"))


def get_model_path(name):
    return './weights/{}_{}.h5'.format(name, get_curr_datetime())


def model_exists():
    return os.path.isfile(get_model_path())


def save_model(model, name):
    model.save(get_model_path(name))


def load_latest_model(name):
    candidates = list(
        filter(lambda x: x.startswith(name) and x.endswith('.h5'),
               os.listdir('./')))

    candidates.sort(reverse=True)

    print("Found candidate", candidates[0])

    model = load_model(candidates[0], custom_objects={'huber_loss': huber_loss})
    return model


def load_model_by_filename(name):
    return load_model(name, custom_objects={'huber_loss': huber_loss})


def huber_loss(y_true, y_pred):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

    error = y_true - y_pred

    quad_term = error*error / 2
    lin_term = abs(error) - 0.5
    use_linear = (abs(error) > 1.0)
    use_linear = K.cast(use_linear, 'float32')

    return use_linear * lin_term + (1 - use_linear) * quad_term

