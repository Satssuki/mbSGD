class FoldWrapper:
    fold_x = []
    fold_y = []

    def __init__(self, x, y):
        self.fold_x = x
        self.fold_y = y

    def get_folds_x(self):
        return self.fold_x

    def get_folds_y(self):
        return self.fold_y
