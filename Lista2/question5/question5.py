import numpy
import pandas
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans

filesDir = "../"
filesName = ["53.txt", "78.txt", "80.txt", "100.txt", "105.txt"]
clientes = [53, 78, 80, 100, 105]

data = pandas.DataFrame()
dataList = []
for fileName in filesName:
	frame = pandas.read_csv(filesDir+fileName, header=None)
	dataList.append(frame)
totalData = pandas.concat(dataList)
data = totalData.copy()

for i in range(len(totalData.columns)):
	if (i not in [0,2,4,5]):
		del totalData[i]
	if (i not in [4,5]):
		del data[i]

totalData.rename(columns={0: 'Cliente', 2:'Hora', 4:'Vazao', 5:'Latencia'}, inplace=True)
data.rename(columns={4:'Vazao', 5:'Latencia'}, inplace=True)

X = numpy.array(data.Latencia.get_values())
Y = numpy.array(data.Vazao.get_values())

kList = [3,4]


for k in kList:
	clustering = KMeans(n_clusters=k)
	clustering.fit(data)
	labels = numpy.array(clustering.labels_)
	
	figure = pyplot.figure(1, figsize=(5, 5))
	pyplot.scatter(X, Y, c=labels.astype(numpy.float))
	
	pyplot.xlabel('Latencia')
	pyplot.ylabel('Vazao')
	pyplot.title('Question 5 - K-means: k = '+str(k))
	pyplot.savefig('k'+str(k)+'.png')
	
	clusterData = totalData.copy()
	clusterData.insert(len(clusterData.columns), 'Cluster', labels)
	
	maxHora = clusterData.Hora.max()
	horaMatrix = numpy.array([[1.0]*k]*(1+maxHora))
	for kHora in range(1+maxHora):
		for kLabel in range(k):
			result = clusterData.loc[(clusterData.Hora == kHora) & (clusterData.Cluster == kLabel)]
			cluster = clusterData.loc[clusterData.Cluster == kLabel]
			horaMatrix[kHora][kLabel] = (1.0*len(result))/len(cluster)
	horaMatrix = pandas.DataFrame(numpy.array(horaMatrix))
	horaMatrix.to_csv('k%d- RazaoHora.txt'%k, sep='\t', encoding='utf-8')
	
	clienteMatrix = numpy.array([[1.0]*len(clientes)]*k)
	for i in range(len(clientes)):
		for kLabel in range(k):
			result = clusterData.loc[(clusterData.Cliente == clientes[i]) & (clusterData.Cluster == kLabel)]
			cluster = clusterData.loc[clusterData.Cluster == kLabel]
			clienteMatrix[kLabel][i] = (1.0*len(result))/len(cluster)
	clienteMatrix = pandas.DataFrame(numpy.array(clienteMatrix))
	clienteMatrix.rename(columns={0:'53', 1:'78', 2:'80', 3:'100', 4:'105'}, inplace=True)
	clienteMatrix.to_csv('k%d- RazaoCliente.txt'%k, sep='\t', encoding='utf-8')
