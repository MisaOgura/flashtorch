class Backprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
