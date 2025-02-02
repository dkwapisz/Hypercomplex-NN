from HypercomplexKeras import Algebra
from HypercomplexKeras.Convolutional import HyperConv2D
from keras import Sequential, Input
from keras.src.layers import GlobalMaxPooling2D, Dense, Conv2D, Flatten


def get_basic_cnn_model(input_shape, num_classes) -> Sequential:

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def get_hypercomplex_cnn_model(input_shape, num_classes, algebra: Algebra) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(HyperConv2D(32, (3, 3), activation='relu', algebra=algebra))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model