import keras_tuner as kt
from HypercomplexKeras.Convolutional import HyperConv2D
from keras import Sequential, Input
from keras.src.layers import MaxPooling2D, Flatten, Dense
from keras.src.optimizers import Adam

from models.ModelBase import ModelBase
from models.ModelUtils import algebras


def build_model(hp, input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    filters = hp.Int('filters', min_value=16, max_value=32, step=16)
    model.add(HyperConv2D(filters, (3, 3), activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    filters = hp.Int('filters', min_value=32, max_value=64, step=32)
    model.add(HyperConv2D(filters, (3, 3), activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    filters = hp.Int('filters', min_value=64, max_value=128, step=64)
    model.add(HyperConv2D(filters, (3, 3), activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(Flatten())

    units = hp.Int('units', min_value=32, max_value=256, step=64)
    model.add(Dense(units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.003, 0.001, 0.0003, 0.0001])),
        metrics=metrics
    )

    return model

def create_tuner(input_shape, num_classes, metrics, algebra):
    return kt.Hyperband(
        lambda hp: build_model(hp, input_shape, num_classes, metrics, algebra),
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='tuner_results',
        project_name='HyperComplex_CNN_Tuning'
    )

class HyperComplexCNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space, metrics, algebra: str):
        super().__init__(color_space, create_tuner(input_shape, num_classes, metrics, algebras[algebra]))