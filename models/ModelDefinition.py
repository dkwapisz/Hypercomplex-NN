from HypercomplexKeras import Algebra
from HypercomplexKeras.Convolutional import HyperConv2D
from keras import Sequential, Input
from keras.src.layers import GlobalMaxPooling2D, Dense, Conv2D, MaxPooling2D, Flatten


def get_basic_cnn_model() -> Sequential:

    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(Conv2D(100, (3, 3)))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def get_hypercomplex_cnn_model(algebra: Algebra) -> Sequential:

    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(HyperConv2D(100, (3, 3), algebra=algebra))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model