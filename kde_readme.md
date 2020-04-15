The `KDEHelper` class takes in a flattened NumPy array and fits a Gaussian KDE to it. This KDE is unbounded so we transform the original data (which is between 0 and 1) to the unbounded space using the inverse sigmoid function before feeding it to the class.

Its `sample_and_transform()` method can then be used to sample from that fitted KDE and transform the samples back into [0, 1]. It takes in a positional `size` argument which needs to be a 2-element tuple representing the size of the matrix the samples should be returned as.
