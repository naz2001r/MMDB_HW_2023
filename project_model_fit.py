import numpy as np
import json

class RandomClassificator:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        return np.random.choice([0, 1], size=1)[0]


if __name__ == '__main__':
    # TODO FIT MODEL AND SAVE RESULTS
    # for _ in range(10):
    #     print(RandomClassificator().predict())
    with open('data1.json', 'r') as f:
        data = json.load(f)
    print(len(data))
