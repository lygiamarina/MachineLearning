import sys
import numpy
import pandas
import pylab
import statsmodels.formula.api as smFormula
from mpl_toolkits.mplot3d import Axes3D

filesDir = "../"
filesName = ["53.txt", "78.txt", "80.txt", "100.txt", "105.txt"]

data = pandas.DataFrame()
dataList = []
for fileName in filesName:
	frame = pandas.read_csv(filesDir+fileName, header=None)
	dataList.append(frame)
data = pandas.concat(dataList)
	
hora = 'Col3'
latencia = 'Col6'

for i in range(len(data.columns)):
	if (i not in [2,5]):
		del data[i]

data.rename(columns={2:'Col3', 5:'Col6'}, inplace=True)

dList = [2,3,5,10]

for d in dList:
	formulaString = latencia + ' ~ 1'
	
	for expo in range(1,d+1):
		formulaString += ' + I('+hora+' ** '+str(expo)+')'

	poly_2 = smFormula.ols(formula=formulaString, data=data).fit()
	
	sys.stdout = open("summaryD"+str(d)+".txt", 'w')
	print poly_2.summary()
	
	figure = pylab.figure()
	X = numpy.arange(data.Col3.min(), data.Col3.max()+2, (data.Col3.max()-data.Col3.min())/10)
	
	Z = numpy.array([poly_2.params[0]]*len(X))
	
	for expo in range(1,d+1):
		Z += poly_2.params[expo]*(X**expo)
		
	pylab.scatter(data.Col3.get_values(), data.Col6.get_values())
	pylab.plot(X, Z, c='red')

	pylab.title('Question 1.1 - Polynomial Regression - d = '+str(d))
	pylab.xlabel('Hora')
	pylab.ylabel('Latencia')
	pylab.savefig("d"+str(d)+".png")
	
