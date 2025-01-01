import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
	"""
	data: np.ndarray – symmetric diagonalizable real-valued matrix
	num_steps: int – number of power method steps

	Returns:
	eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
	eigenvector: np.ndarray – corresponding eigenvector estimation
	"""
	X = np.random.rand(data.shape[0])

	for step in range(num_steps):
		X1 = np.dot(data, X)
		norma = np.linalg.norm(X1, ord=2)

		X = X1 / norma

	eigenvalue = np.dot(X.T, np.dot(data, X))
	return eigenvalue, X