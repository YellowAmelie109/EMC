library(readxl)
#library(hash)
library(ggplot2)
library(ggrepel)

#Remove responses which did not correctly answer the first question with "Bird"
setwd("~/EMC/Results")
ResultsRaw <- read_excel("Classifying images into unsuitable categories (Responses).xlsx")
Pigeoncats <- ResultsRaw$`Pigeon: What category would you select for this image?`
Results <- ResultsRaw[-c(which(Pigeoncats != "Bird")), ]
remove(Pigeoncats)
remove(ResultsRaw)

#Get neural network responses
ResultsNN <- read_excel("NNResults.xlsx")

#Get LLM responses
ResultsLLM <- read_excel("LLMResults.xlsx")

#Plot scatter graph of number of people who chose a category by average confidence score for that category
CatvConf <- function(catCol,confCol,i,ver){
  cats=unique(catCol)
  avg=c()
  count=c()
  for (cat in cats){
    avg <- append(avg,mean(confCol[catCol==cat]))
    count <- append(count,length(confCol[catCol==cat]))
  }
  
  data=data.frame(count,avg)
  ggplot(data, aes(x=count,y=avg,fill=cats,colour=cats)) +
    geom_point(shape=4,size=0.5,stroke=1.25) + 
    geom_text_repel(aes(label=cats)) + 
    labs(x = "Count", y="Average Confidence", title = "Number of people who chose each category vs. Average \nconfidence score for each category") + 
    theme(legend.position="none") +
    ylim(0,10)
  name=sprintf("%d%d.png",ver,i)
  ggsave(name)
}

#CatvConf(Results$`Chappell: What category would you select for this image? 4`,
 #           Results$`Chappell: How well does the category you chose fit the image, in your opinion? 4`)
#for(i in seq(2,ncol(Results),by=3)){
 # CatvConf(unlist(Results[,i]),unlist(Results[,i+2]),i,"humanPlot")
#}

humanCats=c()
NNCats=c()
LLMCats=c()
for(i in seq(2,ncol(Results),by=3)){
  humanCats<-append(humanCats,names(sort(table(Results[,i]), decreasing = TRUE)[1]))
  NNCats<-append(NNCats,names(sort(table(ResultsNN[,i]), decreasing = TRUE)[1]))
  LLMCats<-append(LLMCats,names(sort(table(ResultsLLM[,i]), decreasing = TRUE)[1]))
}
cats=c("Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck")
humanVal=c()
NNVal=c()
LLMVal=c()
for(cat in cats){
  humanVal<-append(humanVal,length(which(humanCats==cat)))
  NNVal<-append(NNVal,length(which(NNCats==cat)))
  LLMVal<-append(LLMVal,length(which(LLMCats==cat)))
}
data=data.frame(Categories=rep(cats,3),
                Source=c(rep("Humans",10),rep("NN",10),rep("LLM",10)),
                Values=c(humanVal,NNVal,LLMVal))
ggplot(data,aes(x=Categories,y=Values,fill=Source)) +
  geom_bar(position="dodge",stat="identity")
