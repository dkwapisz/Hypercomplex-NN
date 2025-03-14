from keras.src.callbacks import EarlyStopping

early_stopping_tuning = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True,
                               verbose=1)

class ModelBase:
    def __init__(self, color_space):
        self.model = None
        self.color_space = color_space

    def get_model(self):
        return self.model

