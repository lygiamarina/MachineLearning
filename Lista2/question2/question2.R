library(lattice)
library(mnormt)

if (getwd() != "C:/Users/Lygia/Dropbox/UFRJ/Aprendizado de Máquina/Lista2")
{
	setwd("C:/Users/Lygia/Dropbox/UFRJ/Aprendizado de Máquina/Lista2");
}

data_53 = read.csv("53.txt", header=FALSE)
data_78 = read.csv("78.txt", header=FALSE)
data_80 = read.csv("80.txt", header=FALSE)
data_100 = read.csv("100.txt", header=FALSE)
data_105 = read.csv("105.txt", header=FALSE)

completeData = rbind(data_53, data_78, data_80, data_100, data_105)

hora = completeData[,3]
vazao = completeData[,5]
latencia = completeData[,6]

mHora = matrix(hora)
mLatencia = matrix(latencia)
mVazao = matrix(vazao)

dList = list(a="2", b="3", c="5", d="10")
mGaussian = NULL

for (d in dList)
{
	dInt = strtoi(d)

	regressionLatencia = lm(vazao ~ poly(latencia, dInt))
	wLatencia = matrix(coef(regressionLatencia))

	regressionHora = lm(vazao ~ poly(hora, dInt))
	wHora = matrix(coef(regressionHora))

	sigma = cov(cbind(mLatencia, mHora))

	mGaussian = Vectorize(function(x,y)
			{
				expo = 1
				polyX = matrix(1)
				polyY = matrix(1)

				while (expo <= dInt)
				{
					polyX = rbind(polyX, matrix(x^expo))
					polyY = rbind(polyY, matrix(y^expo))
					expo = expo + 1
				}
				
				meanX = t(wLatencia)%*%polyX
				meanY = t(wHora)%*%polyY
				meanXY = matrix(meanX)
				meanXY = rbind(meanXY, meanY)

				dmnorm(cbind(x,y), t(meanXY), sigma, log=TRUE)
				
			})

	minLatencia = latencia[which.min(latencia)]
	maxLatencia = latencia[which.max(latencia)]

	minHora = hora[which.min(hora)]
	maxHora = hora[which.max(hora)]

	z <- outer(minLatencia:maxLatencia, minHora:maxHora, mGaussian)	

	print(persp(minLatencia:maxLatencia, minHora:maxHora, z, main = "Bivariate Normal Distribution",,
		col="orchid2", theta = 55, phi = 30, r = 40, d = 0.1, expand = 0.5,
		ltheta = 90, lphi = 180, shade = 0.4, ticktype = "detailed", nticks=5))
}

