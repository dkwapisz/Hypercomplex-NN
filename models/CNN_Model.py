from keras import Sequential, Input
from keras.src.layers import Conv2D, GlobalMaxPooling2D, Flatten, Dense

from models.ModelBase import ModelBase


def create_cnn_model(input_shape, num_classes) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

class CNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space):
        super().__init__(color_space)
        self.model = create_cnn_model(input_shape, num_classes)

    def __str__(self):
        return f"CNN - {self.color_space}"