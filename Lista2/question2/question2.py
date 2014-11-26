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
	
vazao = 'Col5'
hora = 'Col3'
latencia = 'Col6'

for i in range(len(data.columns)):
	if (i not in [2,4,5]):
		del data[i]

data.rename(columns={2:'Col3', 4:'Col5', 5:'Col6'}, inplace=True)

dList = [2,3,5,10]

for d in dList:
	formulaString = vazao + ' ~ 1'
	
	for expo in range(1,d+1):
		formulaString += ' + I('+hora+' ** '+str(expo)+')'
		formulaString += ' + I('+latencia+' ** '+str(expo)+')'

	poly_2 = smFormula.gls(formula=formulaString, data=data).fit()
	
	
	sys.stdout = open("summaryD"+str(d)+".txt", 'w')
	print poly_2.summary()
	
	figure = pylab.figure()
	axes = Axes3D(figure)
	X = numpy.arange(data[['Col3']].min(), data[['Col3']].max(), (data[['Col3']].max()-data[['Col3']].min())/80)
	Y = numpy.arange(data[['Col6']].min(), data[['Col6']].max(), (data[['Col6']].max()-data[['Col6']].min())/80)
	X, Y = numpy.meshgrid(X,Y)
	Z = numpy.array([[poly_2.params[0]]*len(X)]*len(Y))
	
	for expo in range(1,d+1):
		Z += poly_2.params[(2*expo)-1]*(X**expo) + poly_2.params[2*expo]*(Y**expo)

	axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
	axes.set_title('Question 2 - Bivariate Polynomial Regression - d = '+str(d))
	axes.set_xlabel('Hora')
	axes.set_ylabel('Latencia')
	axes.set_zlabel('Predicted Vazao')
	
	axes.view_init(elev=30., azim=150)
	pylab.savefig("d"+str(d)+".png")
	
