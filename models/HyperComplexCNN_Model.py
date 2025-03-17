import optuna
from HypercomplexKeras.Convolutional import HyperConv2D
from keras import Sequential, Input
from keras.src.layers import MaxPooling2D, Flatten, Dense, GlobalMaxPooling2D, Dropout
from keras.src.optimizers import Adam

from models.ModelBase import ModelBase, early_stopping_tuning
from models.ModelUtils import algebras


# def build_model(input_shape, num_classes, metrics, algebra):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#
#     model.add(HyperConv2D(16, (3, 3), padding='SAME', activation='relu', algebra=algebra))
#     model.add(MaxPooling2D(strides=2))
#
#     model.add(HyperConv2D(32, (3, 3), padding='SAME', activation='relu', algebra=algebra))
#     model.add(MaxPooling2D(strides=2))
#
#     model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
#     model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
#     model.add(MaxPooling2D(strides=2))
#
#     model.add(Flatten())
#     model.add(Dense(num_classes, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)
#
#     return model


def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(32, (3, 3), activation='relu', algebra=algebra))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(64, (3, 3), activation='relu', algebra=algebra))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(128, (3, 3), activation='relu', algebra=algebra))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(256, (5, 5), activation='relu', algebra=algebra))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model

def objective_hcnn(trial, train_dataset, val_dataset, input_shape, num_classes, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = trial.suggest_int("num_layers", 2, 5)
    filters_base = trial.suggest_categorical(f"filters_base_num", [2, 4, 8])
    for i in range(num_layers):
        kernel_size_num = trial.suggest_categorical(f"kernels_size_{i}", [3, 5])
        model.add(HyperConv2D(filters_base * (2 ** i), (kernel_size_num, kernel_size_num), padding='SAME',
                              activation='relu', algebra=algebra))
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, verbose=0, callbacks=[early_stopping_tuning])

    return history.history['val_accuracy'][-1]

def perform_model_tuning_hcnn(train_dataset, val_dataset, input_shape, num_classes, algebra):
    study = optuna.create_study(study_name="HCNN-model-tuning", direction="maximize")
    study.optimize(lambda trial: objective_hcnn(trial, train_dataset, val_dataset, input_shape, num_classes,
                                                algebras[algebra]), n_trials=300)
    return study.best_params

class HyperComplexCNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space, metrics, algebra: str):
        super().__init__(color_space)
        self.model = build_model(input_shape, num_classes, metrics, algebras[algebra])


if __name__ == "__main__":
    build_model((100, 100, 4), 8, ["accuracy"], algebras["Quaternions"]).summary()
