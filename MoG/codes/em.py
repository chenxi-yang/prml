import math
import matplotlib.pyplot as plt
import numpy as np



def load_data(fname):
	with open(fname, 'r') as f:
		data = []
		for line in f:
			line = line.strip().split()
			x1 = float(line[0])
			x2 = float(line[1])
			data.append([x1, x2])
		return np.array(data)

def normal(x, mu, sigma):
	n = np.shape(x)[0]
	exp = float(-0.5 * (x - mu) * (sigma.I) * ((x - mu).T))
	div = pow(2 * np.pi, 0.5) * pow(np.linalg.det(sigma), 0.5)
	prob = math.exp(exp) / div
	return prob

def em(x, k, iterNum=500):
	m, n = np.shape(x)
	# initialization
	alpha = [1/3, 1/3, 1/3]
	#mu = np.random.random((k, 2))
	#mu = np.matrix(mu)
	mu = [x[24, :], x[33, :], x[45, :]]
	mu = np.matrix(mu)
	sigma = [np.matrix([[20, -10], [-10, 20]]) for x in range(k)]
	gamma = np.matrix(np.zeros((m, k)))

	for i in range(iterNum):
		# eStep
		for j in range(m):
			sumPm = 0
			for p in range(k):
				gamma[j, p] = alpha[p] * normal(x[j, :], mu[p], np.matrix(sigma[p]))
				sumPm += gamma[j, p]
			for p in range(k):
				gamma[j, p] /= sumPm
		sumGamma = np.sum(gamma, axis=0)
		# mStep
		for p in range(k):
			# update parameter
			mu[p] = np.zeros((1, n))
			sigma[p] = np.zeros((n, n))
			# update mu
			for j in range(m):
				mu[p] += gamma[j, p] * x[j, :]
			mu[p] /= sumGamma[0, p]
			# update sigma
			for j in range(m):
				diff = x[j, :] - mu[p]
				sigma[p] += gamma[j, p] * diff.T * diff		
			sigma[p] /= sumGamma[0, p]
			# update alpha
			alpha[p] = sumGamma[0, p] / m
	print('mu: ',mu)
	print('alpha: ', alpha)
	print('sigma: ', sigma)
	return gamma

def classify(x, k, iterNum=500):
	m, n = np.shape(x)
	gamma = em(x, k, iterNum)
	y = np.zeros(m)
	for j in range(m):
		y[j] = np.argmax(gamma[j, :])
		#for p in range(k):
			#print(str(gamma[j, p])+' and '+str(np.argmax(gamma[j, :])))
			#if gamma[j, p] == np.amax(gamma[j, :]):
				#print('in')
	return y

def mogPlt(x, y, title):
	m = np.shape(x)[0]
	color = ['b', 'r', 'g']
	plt.subplot(1, 1, 1)
	for j in range(m):
		plt.scatter(x[j, 0], x[j, 1], c=color[int(y[j])], marker='o')
	plt.title(title)
	plt.show()

if __name__ == '__main__':
	file = 'data.txt'

	data = load_data(file) #load data

	x = data[:, :]

	y = classify(x, k=3, iterNum=5)
	mogPlt(x, y, title='Data Classified, Iteration = 5')






	


