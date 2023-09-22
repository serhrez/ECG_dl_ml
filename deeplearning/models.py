from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from keras.models import Model
import numpy as np

def get_extr_feats_model_multiple_feats_extr(n_classes, length, last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(length, 1), dtype=np.float32, name='signal')
    x = signal
    num_filters = 128
    for i in range(8):
        x = Conv1D(num_filters, kernel_size, padding='same', use_bias=False,
                kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if i % 2 == 1:
            num_filters /= 2
        if i < 4:
            x = MaxPooling1D(4)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(32, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(16, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(8, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(4, activation='relu', kernel_initializer=kernel_initializer)(x)
    diagn = Dense(n_classes, activation=last_layer)(x)
    model = Model(signal, diagn)
    return model

def optimal_feats_model(n_input):
    input = Input(shape=(n_input, 1), dtype=np.float32, name='signal')
    kernel_initializer = 'he_normal'
    x = Flatten()(input)
    if n_input > 64:
        x = Dense(64, activation='relu', kernel_initializer=kernel_initializer)(x)
    if n_input > 32:
        x = Dense(32, activation='relu', kernel_initializer=kernel_initializer)(x)
    if n_input > 16:
        x = Dense(16, activation='relu', kernel_initializer=kernel_initializer)(x)
    if n_input > 8:
        x = Dense(8, activation='relu', kernel_initializer=kernel_initializer)(x)
    if n_input > 4:
        x = Dense(4, activation='relu', kernel_initializer=kernel_initializer)(x)
    diagn = Dense(3, activation='sigmoid', kernel_initializer=kernel_initializer)(x)
    model = Model(input, diagn)
    return model

def model_with_needed_layers(model):
    return Model(model.input, model.layers[-2].output)

def get_feats_model(feat_num, lead_num, num_of_dense_layers=3, add_pooling_layer=False, dropout_rate=0):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(lead_num, feat_num), dtype=np.float32, name='signal')
    x = signal
    if add_pooling_layer:
        x = MaxPooling1D(2, data_format="channels_first")(x)

    x = Flatten()(x)

    num_of_filters = 2 ** (num_of_dense_layers + 2)
    for i in range(num_of_dense_layers):
        x = Dense(num_of_filters, activation='relu', kernel_initializer=kernel_initializer)(x)
        if dropout_rate != 0:
            x = Dropout(dropout_rate)(x)
        num_of_filters /= 2
    
    diagn = Dense(3, activation='sigmoid', kernel_initializer=kernel_initializer)(x)
    model = Model(signal, diagn)
    return model