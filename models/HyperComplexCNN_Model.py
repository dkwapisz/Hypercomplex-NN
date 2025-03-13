from HypercomplexKeras.Convolutional import HyperConv2D
from keras import Sequential, Input
from keras.src.layers import MaxPooling2D, Flatten, Dense
from keras.src.optimizers import Adam

from models.ModelBase import ModelBase
from models.ModelUtils import algebras


def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(16, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(32, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model


class HyperComplexCNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space, metrics, algebra: str):
        super().__init__(color_space)
        self.model = build_model(input_shape, num_classes, metrics, algebras[algebra])


if __name__ == "__main__":
    build_model((100, 100, 4), 1, ["accuracy"], algebras["Quaternions"]).summary()
