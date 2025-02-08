class ModelBase:
    def __init__(self, color_space, tuner):
        self.model = None
        self.color_space = color_space
        self.tuner = tuner

    def tune_model(self, train_dataset, val_dataset, epochs):
        self.tuner.search(train_dataset, epochs=epochs, validation_data=val_dataset)
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = self.tuner.hypermodel.build(best_hps)

    def get_model(self):
        return self.model