class ModelBase:
    def __init__(self, color_space):
        self.model = None
        self.color_space = color_space

    def get_model(self):
        return self.model