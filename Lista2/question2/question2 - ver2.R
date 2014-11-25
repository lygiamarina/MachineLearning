library(lattice)

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

dList = list(a="2")
mGaussian = NULL

for (d in dList)
{
	dInt = strtoi(d)

	regression = lm(vazao ~ poly(latencia, dInt)+poly(hora, dInt))
	w = matrix(coef(regression))

	regressionSurface = Vectorize(function(x,y)
			{
				expo = 1
				polyXY = matrix(1)

				while (expo <= dInt)
				{
					polyXY = rbind(polyXY, matrix(x^expo))
					polyXY = rbind(polyXY, matrix(y^expo))
					expo = expo + 1
				}
				
				return(t(w)%*%polyXY)			
			})

	minLatencia = latencia[which.min(latencia)]
	maxLatencia = latencia[which.max(latencia)]

	minHora = hora[which.min(hora)]
	maxHora = hora[which.max(hora)]

	z <- outer(mLatencia, mHora, regressionSurface)	

	print(wireframe(mLatencia, mHora, z))
}