from HypercomplexKeras import Algebra
from HypercomplexKeras.Convolutional import HyperConv2D
from keras import Sequential, Input
from keras.src.layers import GlobalMaxPooling2D, Flatten, Dense

from models.ModelBase import ModelBase
from models.ModelUtils import algebras


def create_hypercomplex_cnn_model(input_shape, num_classes, metrics, algebra: Algebra) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(HyperConv2D(32, (3, 3), activation='relu', algebra=algebra))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model


class HyperComplexCNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space, metrics, algebra: str):
        super().__init__(color_space)
        self.model = create_hypercomplex_cnn_model(input_shape, num_classes, metrics, algebras[algebra])
        self.algebra_name = algebra
        self.color_space = color_space

    def __str__(self):
        return f"HyperComplexCNN - {self.color_space} - {self.algebra_name}"