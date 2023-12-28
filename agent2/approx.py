import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

x = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5]])
y = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,4], [1,3,6]])
# y = np.array([[2,2,2], [3,3,3], [4,4,4]])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)