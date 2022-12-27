import sklearn
import numpy as np
import time

if __name__ == '__main__':
    start = time.time()
    print(np.sum(np.arange(19)))
    end = time.time()
    print(end - start)