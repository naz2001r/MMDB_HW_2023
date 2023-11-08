import numpy as np


class RandomClassificator:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        return np.random.choice([0, 1], size=1)[0]


if __name__ == '__main__':
    # TODO FIT MODEL AND SAVE RESULTS
    for _ in range(10):
        print(RandomClassificator().predict())
