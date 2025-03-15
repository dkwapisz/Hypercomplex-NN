import optuna
from HypercomplexKeras.Convolutional import HyperConv2D
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
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


def objective_hcnn(trial, train_dataset, val_dataset, input_shape, num_classes, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = trial.suggest_int("num_layers", 1, 8)

    for i in range(num_layers):
        filters = trial.suggest_categorical(f"filters_{i}", [1, 2, 4, 8, 16, 32, 64, 128])
        kernel_size_num = trial.suggest_categorical(f"kernels_size_{i}", [1, 3, 5, 7])

        model.add(HyperConv2D(filters, (kernel_size_num, kernel_size_num), padding='SAME',
                              activation='relu', algebra=algebra))

        if trial.suggest_categorical(f"double_conv_{i}", [True, False]):
            model.add(HyperConv2D(filters, (kernel_size_num, kernel_size_num), padding='SAME',
                                  activation='relu', algebra=algebra))

        if trial.suggest_categorical(f"max_pool_{i}", [True, False]):
            strides = trial.suggest_categorical(f"strides_{i}", [None, 1, 2, 3])
            pool_size_num = trial.suggest_categorical(f"pool_size_{i}", [2, 3])
            model.add(MaxPooling2D(strides=strides, pool_size=(pool_size_num, pool_size_num)))

        if trial.suggest_categorical(f"batch_norm_{i}", [True, False]):
            model.add(BatchNormalization())

        if trial.suggest_categorical(f"dropout_{i}", [True, False]):
            dropout_rate = trial.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(Dropout(dropout_rate))

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

    early_stopping_tuning = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, verbose=0,
                        callbacks=[early_stopping_tuning, lr_scheduler])

    return history.history['val_accuracy'][-1]


def perform_model_tuning_hcnn(train_dataset, val_dataset, input_shape, num_classes, algebra):
    study = optuna.create_study(study_name="HCNN-optimizer", direction="maximize", sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_hcnn(trial, train_dataset, val_dataset, input_shape, num_classes,
                                                algebras[algebra]), n_trials=200)
    return study.best_params


class HyperComplexCNN_Model(ModelBase):
    def __init__(self, input_shape, num_classes, color_space, metrics, algebra: str):
        super().__init__(color_space)
        self.model = build_model(input_shape, num_classes, metrics, algebras[algebra])


if __name__ == "__main__":
    build_model((100, 100, 4), 1, ["accuracy"], algebras["Quaternions"]).summary()
