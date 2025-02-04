class ModelBase:
    def __init__(self, color_space):
        self.model = None
        self.color_space = color_space

    def fit(self, train_dataset, val_dataset, epochs, verbose=1):
        return self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=verbose)

    def evaluate(self, test_dataset, batch_size=32, verbose=1):
        return self.model.evaluate(test_dataset, batch_size=batch_size, verbose=verbose)