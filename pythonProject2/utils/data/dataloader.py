import numpy as np
import random
from tensor import Tensor


class dataloader:
    def __init__(self, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, num_examples, self.batch_size):
            batch_indices = np.array(indices[i: min(i + self.batch_size, num_examples)])
            yield Tensor(features.data[batch_indices], float, True), Tensor(labels.data[batch_indices], float, True)