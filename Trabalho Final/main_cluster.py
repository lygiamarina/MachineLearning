import sys
import numpy
import pandas
import ghmm
import datetime
import scipy.stats as spyStats
from sklearn.cluster import KMeans


filesDir = "./"
filesName = ["53.txt", "78.txt", "80.txt", "100.txt", "105.txt"]
nStates = 5

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
	
	trainingCopy = training.copy()
	del trainingCopy['Data']
	del trainingCopy['Hora']
	del trainingCopy['Vazao']
	
	clustering = KMeans(n_clusters=nStates)
	clustering.fit(trainingCopy)
	labels = numpy.array(clustering.labels_)
	
	training['Cluster'] = labels
	
	
	maxLatencia = training.Latencia.max()
	
	statesLabel = range(nStates)
	states = {}
	pi = []
	transition = []
	
	for i in statesLabel:
		resultState = training.loc[training.Cluster == i]
		numberDaysState = len(resultState.loc[resultState.Hora == 00])
		states[i] = resultState
		pi.append((1.0*numberDaysState)/(1.0*numberDays))
		transition.append([])
	
	possibilities = []
	for i in statesLabel:
		poss = 0.0
		for j in statesLabel:
			freq = 0.0
			for h in range(24):
				if h < 23:
					resultHi = states[i].loc[states[i].Hora == h]
					resultHj = states[j].loc[states[j].Hora == h+1]
					result = pandas.merge(resultHi, resultHj, on='Data', how='inner')
				else:
					resultHi = states[i].loc[states[i].Hora == h]
					resultHj = states[j].loc[states[j].Hora == 00]
					result = pandas.merge(resultHi, resultHj, on='Data', how='inner')
				freq += len(result)
			transition[i].append(freq)
			poss += freq
		possibilities.append(poss)

	emission = []
	meanStd = []
	lastVar = 0
	for i in range(len(statesLabel)):
		for j in range(len(statesLabel)):
			if (possibilities[i] > 0):
				transition[i][j] = transition[i][j]/possibilities[i]
		vazao = numpy.array(states[i].Vazao.get_values())
		mean = numpy.mean(vazao)		
		var = numpy.var(vazao)
		std = numpy.std(vazao)
		if var == 0.0:
			var = lastVar
		lastVar = var
		emission.append([mean, var])
		meanStd.append([mean, std])
		
	
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
		ok = False
		for j in range(len(meanStd)):
			mean = meanStd[j][0]
			std = meanStd[j][1]
			obs = emissionTrainingNP[i]
			if obs >= mean-(2*std) and obs <= mean+(2*std):
				ok = True
		if ok:
			emissionTraining.append(emissionTrainingNP[i])
			
	emissionTraining = ghmm.EmissionSequence(ghmmFloat, emissionTraining)

	sys.stdout = open('model'+fileName, 'w')
	print hmm.verboseStr()

	hmm.baumWelch(emissionTraining)

	sys.stdout = open('modelBAUM'+fileName, 'w')
	print hmm.verboseStr()
	
	nHours = len(real)
	predicted = hmm.sampleSingle(nHours)
	realVazao = real.Vazao.get_values()
	
	sys.stdout = open('predict'+fileName, 'w')
	print predicted.verboseStr()
	
	RMSLE = 0.0
	for i in range(nHours):
		sub = numpy.log(predicted[i]+1) - numpy.log(realVazao[i] + 1)
		RMSLE += numpy.square(sub)
	RMSLE = numpy.sqrt(RMSLE/nHours)
	
	print 'RMSLE = ',RMSLE
