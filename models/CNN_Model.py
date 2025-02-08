import keras_tuner as kt
from keras import Sequential, Input
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam

from models.ModelBase import ModelBase


def build_model_tuner(hp, input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    units = hp.Int('units', min_value=64, max_value=256, step=64)
    model.add(Dense(units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.003, 0.001, 0.0003, 0.0001])),
        metrics=metrics
    )

    return model

def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model

def create_tuner(input_shape, num_classes, metrics):
    return kt.Hyperband(
        lambda hp: build_model_tuner(hp, input_shape, num_classes, metrics),
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='tuner_results',
        project_name='CNN_Tuning'
    )

class CNN_Model(ModelBase):
    def __init__(self, tune_model, input_shape, num_classes, color_space, metrics):
        if tune_model:
            super().__init__(color_space, create_tuner(input_shape, num_classes, metrics))
        else:
            super().__init__(color_space, None)
            self.model = build_model(input_shape, num_classes, metrics)
