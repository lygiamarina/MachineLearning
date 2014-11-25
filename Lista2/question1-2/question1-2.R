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

dList = list(a="2", b="3", c="5", d="10", e="20")

for (d in dList)
{
	dInt = strtoi(d)
	regression = lm(vazao ~ poly(hora, dInt))

	fileNameTxt = paste("question1-2/d_", d, ".txt", sep="")
	sink(file=fileNameTxt) 
	print(summary(regression))
	print(logLik(regression)) 
	sink(NULL)

	fileNamePng = paste("question1-2/d_", d, ".png", sep="")
	title = paste("Polynomial Regression, d = ", d, sep="")
	png(filename=fileNamePng)
	plot(vazao ~ hora, main=title)
	lines(sort(hora), fitted(regression)[order(hora)], col='red')
	dev.off()
}