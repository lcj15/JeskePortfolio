import sklearn
import numpy as np
import time

if __name__ == '__main__':
    start = time.time()
    print(np.sum(np.arange(19)))
    end = time.time()
    print(end - start)

# TODO create main methods for each classifier
# TODO import other metrics for classification (like ROC)
# TODO create abstract unsupervised learning, transformer (PCA, LDA)
# TODO create transfer learning library
