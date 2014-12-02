import sys
import numpy
import pandas
import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn import preprocessing

filesDir = "../"
filesName = ["53.txt", "78.txt", "80.txt", "100.txt", "105.txt"]

data = pandas.DataFrame()
dataList = []
for fileName in filesName:
	frame = pandas.read_csv(filesDir+fileName, header=None)
	dataList.append(frame)
data = pandas.concat(dataList)
featuresData = data.copy()

vazao = 'Col5'
hora = 'Col3'
latencia = 'Col6'

for i in range(len(data.columns)):
	if (i not in [2,4,5]):
		del data[i]
		del featuresData[i]
	elif (i in [4]):
		del featuresData[i]

data.rename(columns={2:'Col3', 4:'Col5', 5:'Col6'}, inplace=True)
#featuresData.rename(columns={2:'Col3', 5:'Col6'}, inplace=True)

dList = [2,3,5]

for d in dList:
	gmm = mixture.GMM(n_components=d, covariance_type='diag')
	gmm.fit(data)
	likelihoods = gmm.score(data)
	
	#sys.stdout = open("scoreD"+str(d)+".txt", 'w')
	#print sum(likelihoods)
	
	#X = numpy.arange(data[['Col3']].min(), data[['Col3']].max(), (data[['Col3']].max()-data[['Col3']].min())/80)
	#Y = numpy.arange(data[['Col6']].min(), data[['Col6']].max(), (data[['Col6']].max()-data[['Col6']].min())/80)
	#Z = numpy.arange(data[['Col5']].min(), data[['Col5']].max(), (data[['Col5']].max()-data[['Col5']].min())/80)
	
	#continuousData = pandas.DataFrame(X)
	#continuousData.insert(len(continuousData.columns), 'Col5', Z)
	#continuousData.insert(len(continuousData.columns), 'Col6', Y)
	#continuousData.rename(columns={0:'Col3', 1:'Col5', 2:'Col6'}, inplace=True)
	
	#labels = gmm.predict(continuousData)
	
	#clusterData = pandas.DataFrame(continuousData)
	#clusterData.insert(len(clusterData.columns), 'Label', labels)
	
	sample = gmm.sample(3425)
	
	figure = pylab.figure(figsize=pylab.figaspect(0.5))
	axes = figure.add_subplot(1, 2, 1, projection='3d')
	
	axes.set_title('Question 3 - Data')
	axes.set_xlabel('Hora')
	axes.set_ylabel('Latencia')
	axes.set_zlabel('Vazao')
	
	axes.view_init(elev=30., azim=22)
	axes.scatter(data.Col3, data.Col5, data.Col6, c='green')
	
	axes = figure.add_subplot(1, 2, 2, projection='3d')
	
	axes.set_title('Question 3 - Gaussian Mixture - d = '+str(d))
	axes.set_xlabel('Hora')
	axes.set_ylabel('Latencia')
	axes.set_zlabel('Vazao')
	
	axes.view_init(elev=30., azim=22)
	axes.scatter(sample[:,0], sample[:,1], sample[:,2], c='red')
	
	pylab.savefig("d"+str(d)+".png")
