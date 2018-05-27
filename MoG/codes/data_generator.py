import numpy as np
import matplotlib.pyplot as plt

#
#data
#classification: USA & Chinese & Korean
#features: GRE_verbal, GRE_reasoning 
#
#data sample:
#167 105
#
def generateData(sigma, mu, N):
	X = np.zeros((N, 2))
	X = np.matrix(X)
	Y = np.zeros((N, 1))
	Y = np.matrix(Y)
	mu0, mu1, mu2 = mu
	sigma0, sigma1, sigma2 = sigma
	for i in range(N):
		if np.random.random(1) < 0.333:
			X[i, :] = np.random.multivariate_normal(mu0, sigma0, 1)
		elif np.random.random(1) < 0.667:
			X[i, :] = np.random.multivariate_normal(mu1, sigma1, 1)
		else:
			X[i, :] = np.random.multivariate_normal(mu2, sigma2, 1)
	return X

def originDataPlt(X, title):
	plt.subplot(1, 1, 1)
	plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c='r', marker='o')
	plt.title(title)
	plt.show()

if __name__ == '__main__':
	N = 900
	k = 3
	# the mean of verbal and quantity
	mu0 = [90, 80]
	mu1 = [60, 100]
	mu2 = [80, 90]
	mu = mu0, mu1, mu2
	sigma0 = np.matrix([[10, -8], [-8, 10]])
	sigma1 = np.matrix([[15, -10], [-10, 15]])
	sigma2 = np.matrix([[20, 15], [15, 20]])
	sigma = sigma0, sigma1, sigma2
	X = generateData(sigma, mu, N)

	openfile = open('data.txt', 'w')
	for i in range(N):
		openfile.write(str(X[i, 0]) + ' ' + str(X[i, 1]) + '\n')


	originDataPlt(X, title="Original Data Distribution")





