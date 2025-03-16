import optuna
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.src.optimizers import Adam

from models.ModelBase import ModelBase


def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model


def objective_cnn(trial, train_dataset, val_dataset, input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = trial.suggest_int("num_layers", 1, 8)
    activation = trial.suggest_categorical("activation", ["relu", "elu", "gelu", "silu"])

    for i in range(num_layers):
        filters = trial.suggest_categorical(f"filters_{i}", [4, 8, 16, 32, 64, 128, 256, 512])
        kernel_size_num = trial.suggest_categorical(f"kernels_size_{i}", [1, 3, 5, 7])

        model.add(Conv2D(filters, (kernel_size_num, kernel_size_num), padding='same',
                         activation=activation))

        if trial.suggest_categorical(f"double_conv_{i}", [True, False]):
            model.add(Conv2D(filters, (kernel_size_num, kernel_size_num), padding='same',
                             activation=activation))

        if trial.suggest_categorical(f"pool_{i}", [True, False]):
            strides = trial.suggest_categorical(f"strides_{i}", [None, 1, 2, 3])
            model.add(MaxPooling2D(strides=strides))

        if trial.suggest_categorical(f"batch_norm_{i}", [True, False]):
            model.add(BatchNormalization())

        if trial.suggest_categorical(f"dropout_{i}", [True, False]):
            dropout_rate = trial.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(Dropout(dropout_rate))

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['val_accuracy', 'val_loss'])

    early_stopping_tuning = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, verbose=0, callbacks=[early_stopping_tuning, lr_scheduler])

    return history.history['val_accuracy'][-1]

def perform_model_tuning_cnn(train_dataset, val_dataset, input_shape, num_classes):
    study = optuna.create_study(study_name="CNN-optimizer", direction="maximize", sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_cnn(trial, train_dataset, val_dataset, input_shape, num_classes),
                   n_trials=300)
    return study.best_params

class CNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space, metrics):
        super().__init__(color_space)
        self.model = build_model(input_shape, num_classes, metrics)
