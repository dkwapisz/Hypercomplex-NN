import optuna
from keras import Sequential, Input
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.optimizers import Adam

from models.ModelBase import ModelBase, early_stopping_tuning


# def build_model(input_shape, num_classes, metrics):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#
#     model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
#     model.add(MaxPooling2D(strides=2))
#
#     model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#     model.add(MaxPooling2D(strides=2))
#
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#     model.add(MaxPooling2D(strides=2))
#
#     model.add(Flatten())
#     model.add(Dense(num_classes, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)
#
#     return model


def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model


def objective_cnn(trial, train_dataset, val_dataset, input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = trial.suggest_int("num_layers", 2, 6)
    for i in range(num_layers):
        filters = trial.suggest_categorical(f"filters_{i}", [8, 16, 32, 64, 128, 256])
        kernel_size_num = trial.suggest_categorical(f"kernels_size_{i}", [3, 5, 7])
        model.add(Conv2D(filters, (kernel_size_num, kernel_size_num), padding='same', activation='relu'))
        if trial.suggest_categorical(f"pool_{i}", [True, False]):
            strides = trial.suggest_categorical(f"strides_{i}", [None, 1, 2])
            model.add(MaxPooling2D(strides=strides))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, verbose=0, callbacks=[early_stopping_tuning])

    return history.history['val_accuracy'][-1]

def perform_model_tuning_cnn(train_dataset, val_dataset, input_shape, num_classes):
    study = optuna.create_study(study_name="CNN-model-tuning", direction="maximize")
    study.optimize(lambda trial: objective_cnn(trial, train_dataset, val_dataset, input_shape, num_classes),
                   n_trials=100)
    return study.best_params

class CNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space, metrics):
        super().__init__(color_space)
        self.model = build_model(input_shape, num_classes, metrics)
