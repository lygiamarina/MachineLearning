import sys
import numpy
import pandas
import scipy.optimize as scipyOpt

filesDir = "../"
filesName = ["53.txt", "78.txt", "80.txt", "100.txt", "105.txt"]

dataList = []
for fileName in filesName:
	frame = pandas.read_csv(filesDir+fileName, header=None)
	dataList.append(frame)
data = pandas.concat(dataList)

for i in range(len(data.columns)):
	if (i not in [4,5]):
		del data[i]

data.rename(columns={4:'Vazao', 5:'Latencia'}, inplace=True)

X = numpy.array(data.Latencia.get_values())
varX = numpy.var(X)
meanX = numpy.mean(X)

Y = numpy.array(data.Vazao.get_values())
varY = numpy.var(Y)
meanY = numpy.mean(Y)

covXY = numpy.cov(X,Y)
covXY = covXY[0][1]

def f(ab):
	return varY + (meanY**2) - 2*ab[0]*(covXY + meanX*meanY) - 2*ab[1]*meanY + \
	(ab[0]**2)*(varX + (meanX**2)) + 2*ab[0]*ab[1]*meanX + (ab[1]**2)

ab0 = numpy.array([1,1])
results = scipyOpt.minimize(f, ab0, method='Nelder-Mead')
sys.stdout = open('results.txt', 'w')
print "[a,b] = [{0[0]},{0[1]}]".format(results.x)
print "Cov(X,Y)=%d"%covXY
