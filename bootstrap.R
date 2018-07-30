library("AER")
library(glmnet)
library(MASS)
library(ggplot2)
library(e1071)
library(randomForest)
library(class)
library(mlr)
library(Amelia)
library(ROCR)

x <- 28101990

set.seed(x)
yourdata<- sample(1:150000,5000,replace=TRUE)
datatowork <- data[yourdata,]

#we have NAs, do not use mean value, because it is biased
#Remove rows with missing values
naIdxDep <- which(is.na(datatowork$NumberOfDependents))
naIdxInc <- which(is.na(datatowork$MonthlyIncome))
naIdx <- union(naIdxDep, naIdxInc)
removedData <- datatowork[naIdx,]
remainedData <- datatowork[-naIdx,]

#Bootstrap
num <- 100

test<-numeric()
#logisticAcc<-numeric()
logisticPred<-numeric()
#ldaAcc<-numeric()
ldaPred<-numeric()
ldaPost<-numeric()
#nbAcc<-numeric()
nbPred<-numeric()
#rfAcc<-numeric()
rfPred<-numeric()
#svmAcc<-numeric()
svmPred<-numeric()
#knnAcc<-numeric()
knnPred<-numeric()

for (i in 1:num) {
  print(i)
  #create train and test datasets
  trainIdx <- sample(1:3982,3000, replace=TRUE)
  trainData <- remainedData[trainIdx,]
  testData <- remainedData[-trainIdx,]
  test <- append(test, testData$SeriousDlqin2yrs)
  
  #logistic Regression
  logb <- glm(SeriousDlqin2yrs ~ NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                MonthlyIncome, data = trainData, family = binomial)
  predTrain <- predict(logb, trainData, interval = "confidence")
  # need to find sensitivity and specificity to find limit
  logPredict <- predict(logb, testData, type = "response")
  #  t <- table(factor(logPredict>0.5),testData$SeriousDlqin2yrs)
  #  logisticAcc <- append(logisticAcc, sum(diag(t))/dim(testData)[1])
  logisticPred <- append(logisticPred, logPredict)
  
  #LDA
  ldaVars <- lda(as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                   NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                   MonthlyIncome, data = trainData)
  ldaPredict <- predict(ldaVars, testData, type = "response")
  ldaClass <- ldaPredict$class
  posterior <- ldaPredict$posterior
  #  t <- table(ldaClass,testData$SeriousDlqin2yrs)
  #  ldaAcc <- append(ldaAcc, sum(diag(t))/dim(testData)[1])
  ldaPred <- append(ldaPred, ldaClass)
  ldaPost <- append(ldaPost, posterior[,2])
  
  #Naive Bayes  
  nb<- naiveBayes(as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                    NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                    MonthlyIncome, data = trainData)
  nbPredict <- predict(nb, testData)
  #  t <- table(nbPredict,testData$SeriousDlqin2yrs)
  #  nbAcc <- append(nbAcc, sum(diag(t))/dim(testData)[1])
  nbPred <- append(nbPred, nbPredict)
  
  #Random Forests
  rf <- randomForest(as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                       NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                       MonthlyIncome, data = trainData, ntree=200,
                     mtry=5, importance=TRUE)
  pred<-predict(rf, testData, type = "prob") 
  #  t <- table(factor(pred[,2]>0.5),testData$SeriousDlqin2yrs)
  #  rfAcc <- sum(diag(t))/dim(testData)[1]
  rfPred <- append(rfPred, pred[,2])
  
  #KNN
  knnPred <- append(knnPred, knn(trainData[c(1,3,4,6,8,9,10)],
                                 testData[c(1,3,4,6,8,9,10)], cl=trainData$SeriousDlqin2yrs,k=1))
  
  #SVM
  tune.out <- tune(svm ,as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                     NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                     MonthlyIncome, data = trainData, 
                   kernel ="linear",ranges=list(cost=1, gamma=0.5, probability = TRUE))
  best <- tune.out$best.model
  svmPred <- append(svmPred, predict(best,testData,probability = TRUE))
}

#Logistic Regression
t <- table(factor(logisticPred>0.5),test)
sum(diag(t))/length(test)
pr <- prediction(logisticPred, test)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf,colorize=TRUE)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#LDA
t <- table(ldaPred,test)
sum(diag(t))/length(test)
pred <- prediction(ldaPost, test) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#Naive Bayes
t <- table(nbPred,test)
sum(diag(t))/length(test)
predvec <- ifelse(nbPred=="1", 1, 0)
realvec <- ifelse(test=="1", 1, 0)
pred <- prediction(predvec,realvec)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#Random Forests
t <- table(factor(rfPred>0.5),test)
sum(diag(t))/length(test)
pred <- prediction(rfPred,test)
prf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(prf,colorize=TRUE)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#KNN
t<- table(test,knnPred)
sum(diag(t))/length(test)
pred <- prediction(as.integer(knnPred)-1, test) 
perf <- performance(pred,"tpr","fpr")
plot(perf, colorize = TRUE)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#SVM
t <- table(svmPred,test)
sum(diag(t))/length(test)
pred <- prediction(svmPred, test) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))
