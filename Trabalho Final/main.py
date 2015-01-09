import sys
import numpy
import pandas
import ghmm
import datetime

filesDir = "./"
filesName = ["53.txt", "78.txt", "80.txt", "100.txt", "105.txt"]

for fileName in filesName:
	data = pandas.DataFrame()
	data = pandas.read_csv(filesDir+fileName, header=None)
	
	data.rename(columns={0:'Cliente', 1:'Data', 2:'Hora', 3:'Since1970', 4:'Vazao', 5:'Latencia'}, inplace=True)
	
	del data['Cliente']
	del data['Since1970']
	
	strFormat = '%d/%m/%Y'
	data['Data'] = pandas.to_datetime(data['Data'], format=strFormat)	
	delta = datetime.timedelta(days=10)
	startDate = data['Data'][0]
	training = data.loc[data.Data < startDate+delta]
	real = data.loc[data.Data >= startDate+delta]
	
	numberDays = len(training.loc[data.Hora == 00])
	
	maxLatencia = training.Latencia.max()
	
	statesLabel = [20,40,60,80,100]
	rangeLatencia = {}
	pi = []
	transition = []
	
	for i in statesLabel:
		resultLatencia = training.loc[(training.Latencia > ((i-20)*maxLatencia/100.0)) & (training.Latencia <= (i*maxLatencia/100.0))]
		numberDaysState = len(resultLatencia.loc[resultLatencia.Hora == 00])
		rangeLatencia[i] = resultLatencia
		pi.append((1.0*numberDaysState)/(1.0*numberDays))
		transition.append([])
	
	possibilities = []
	for i in statesLabel:
		poss = 0.0
		for j in statesLabel:
			freq = 0.0
			for h in range(24):
				if h < 23:
					resultHi = rangeLatencia[i].loc[rangeLatencia[i].Hora == h]
					resultHj = rangeLatencia[j].loc[rangeLatencia[j].Hora == h+1]
					result = pandas.merge(resultHi, resultHj, on='Data', how='inner')
				else:
					resultHi = rangeLatencia[i].loc[rangeLatencia[i].Hora == h]
					resultHj = rangeLatencia[j].loc[rangeLatencia[j].Hora == 00]
					result = pandas.merge(resultHi, resultHj, on='Data', how='inner')
				freq += len(result)
			transition[(i/20)-1].append(freq)
			poss += freq
		possibilities.append(poss)

	emission = []
	lastStd = 0
	for i in range(len(statesLabel)):
		for j in range(len(statesLabel)):
			if (possibilities[i] > 0):
				transition[i][j] = transition[i][j]/possibilities[i]
		vazao = numpy.array(rangeLatencia[(i+1)*20].Vazao.get_values())
		mean = numpy.mean(vazao)
		std = numpy.std(vazao)
		if std == 0.0:
			std = lastStd
		lastStd = std
		emission.append([mean, std])
		
	
	#pi - Vetor de probabilidade de se comecar em estado
	#A - Matriz de probabilidade de transicao
	#B - Distribuicao das observacoes
	#__call__(self, emissionDomain, distribution, A, B, pi, hmmName=None, labelDomain=None, labelList=None, densities=None)
	
	ghmmFloat = ghmm.Float()
	ghmmGaussianDistribution = ghmm.GaussianDistribution(ghmmFloat)
	
	hmm = ghmm.HMMFromMatrices(ghmmFloat, ghmmGaussianDistribution, transition, emission, pi)
	emissionTrainingNP = training.Vazao.get_values()
	emissionTraining = []
	for i in range(len(emissionTrainingNP)):
		emissionTraining.append(emissionTrainingNP[i])
		
	sys.stdout = open('model'+fileName+'.txt', 'w')
	print hmm.verboseStr()

	hmm.baumWelch(ghmm.EmissionSequence(ghmmFloat, emissionTraining))

	sys.stdout = open('model'+fileName+'BAUM.txt', 'w')
	print hmm.verboseStr()
	break
