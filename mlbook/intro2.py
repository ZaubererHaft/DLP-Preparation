import numpy as np
from scipy import sparse

identity = np.eye(4)
print("array:\n", identity)

sparse_matrix = sparse.csr_matrix(identity)
print("sparce matrix in CSR format", sparse_matrix)