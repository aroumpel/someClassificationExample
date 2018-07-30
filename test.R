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

summary(datatowork)

#we have NAs, do not use mean value, because it is biased
#Remove rows with missing values
naIdxDep <- which(is.na(datatowork$NumberOfDependents))
naIdxInc <- which(is.na(datatowork$MonthlyIncome))
naIdx <- union(naIdxDep, naIdxInc)
removedData <- datatowork[naIdx,]
remainedData <- datatowork[-naIdx,]

#Because we removed 1018 rows, all the models have to be trained using cross-validation
trainIdx <- sample(1:3982,3000, replace=FALSE)
trainData <- remainedData[trainIdx,]
testData <- remainedData[-trainIdx,]

#only good for linear correlation, and continuous variables.
scatter1 <- scatterplotMatrix(trainData)
scatter2 <- scatterplotMatrix(trainData[c(1,3,4,6,8,9,10)])

#From now on we forget about the testData :)
#First, we have to choose which variables we need
#Models: Logistic Regression, Random Forests, SVMs, Naive Bayes

#Correlation
cor1 <- cor(trainData) # not useful
cor2 <- cor(trainData, method = "spearman") #notice the difference


#Logistic Regression
null <- glm(SeriousDlqin2yrs~1, data=trainData, family=binomial)
full <- glm(SeriousDlqin2yrs~., data=trainData, family=binomial)
#Variable Selection
#step
forward <- step(null, scope=list(lower=null, upper=full), direction="forward")
backward <- step(null, scope=list(lower=null, upper=full), direction="backward")
both <- step(null, scope=list(lower=null, upper=full), direction="both")
#AIC
#AIC <- stepAIC(full)

hist(trainData$age, breaks = 50, xlab = "Age", main ="")
hist(trainData$NumberOfTime30.59DaysPastDueNotWorse[which(trainData$NumberOfTime30.59DaysPastDueNotWorse<10)], 
     xlim = c(0,10), breaks = c(0,1,2,3,4,5,6,7,8,9,10), 
     xlab = "30-59 days past due", main ="")
hist(trainData$NumberOfTime60.89DaysPastDueNotWorse[which(trainData$NumberOfTime60.89DaysPastDueNotWorse<5)], 
     xlim = c(0,5), breaks = c(0,1,2,3,4),
     xlab = "60-89 days past due", main ="")
hist(trainData$NumberOfTimes90DaysLate[which(trainData$NumberOfTimes90DaysLate<=10)], 
     xlim = c(0,10), breaks = c(0,1,2,3,4,5,6,7,8,9,10),
     xlab = "90 days past due", main ="")
hist(trainData$MonthlyIncome[which(trainData$MonthlyIncome<30000)], 
     xlim = c(0,30000), xlab = "Monthly income", main ="")
hist(trainData$NumberRealEstateLoansOrLines[which(trainData$NumberRealEstateLoansOrLines<=11)], 
     xlim = c(0,11), breaks = c(0,1,2,3,4,5,6,7,8,9,10,11),
     xlab = "# of loans/lines", main ="")

#forward and both ways give the same predictors
logb <- glm(SeriousDlqin2yrs ~ NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
           NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
           MonthlyIncome, data = trainData, family = binomial)
predTrain <- predict(logb, trainData, interval = "confidence")
# need to find sensitivity and specificity to find limit
logPredict <- predict(logb, testData, type = "response")
t <- table(factor(logPredict>0.5),testData$SeriousDlqin2yrs)
sum(diag(t))/dim(testData)[1]

pr <- prediction(logPredict, testData$SeriousDlqin2yrs)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf,colorize=TRUE)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#goodness of fit
ci1 <- confint(logb)

p <- ggplot(trainData, aes(x=as.factor(SeriousDlqin2yrs), y = NumberOfTimes90DaysLate)) + 
  geom_boxplot(outlier.colour=NA) + 
  scale_y_continuous(limits=c(0,5), expand = c(0, 0))
p

p1 <- ggplot(trainData, aes(x=as.factor(SeriousDlqin2yrs), y = MonthlyIncome)) + 
  geom_boxplot(outlier.colour=2) + 
  scale_y_continuous(limits=c(1500,15000), expand = c(0, 0))
p1

p2 <- ggplot(trainData, aes(x=as.factor(SeriousDlqin2yrs), y = age)) + 
  geom_boxplot(outlier.colour=2) 
p2

p3 <- ggplot(trainData, aes(x=as.factor(SeriousDlqin2yrs), y = NumberRealEstateLoansOrLines)) + 
  geom_boxplot(outlier.colour=2) 
p3

#LDA - is not feature selection
ldaVars <- lda(as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                 NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                 MonthlyIncome, data = trainData)
ldaPredict <- predict(ldaVars, testData, type = "response")
ldaClass <- ldaPredict$class
posterior <- ldaPredict$posterior
t <- table(ldaClass,testData$SeriousDlqin2yrs)
sum(diag(t))/dim(testData)[1]

pred <- prediction(posterior[,2], testData$SeriousDlqin2yrs) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#Lasso?
train_mat <- model.matrix(SeriousDlqin2yrs~., data = trainData)[,-1]
models_lasso <- glmnet(train_mat,trainData$SeriousDlqin2yrs,alpha=1, lambda=0.5, family="binomial")
lasso.cv <- cv.glmnet(train_mat,trainData$SeriousDlqin2yrs, alpha=1, lambda=c(0.01, 0.1, 0.3, 0.5), family="binomial")
plot(lasso.cv)

#Naive Bayes  
nb<- naiveBayes(as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                  NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                  MonthlyIncome, data = trainData)
nbPredict <- predict(nb, testData)
t <- table(nbPredict,testData$SeriousDlqin2yrs)
sum(diag(t))/dim(testData)[1]

predvec <- ifelse(nbPredict=="1", 1, 0)
realvec <- ifelse(testData$SeriousDlqin2yrs=="1", 1, 0)
pr <- prediction(predvec,realvec)

#pr <- prediction(as.integer(nbPredict)-1, testData$SeriousDlqin2yrs)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf,colorize=TRUE)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#Random Forests
rf <- randomForest(as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                     NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                     MonthlyIncome, data = trainData, ntree=200,
                   mtry=5, importance=TRUE)
pred<-predict(rf, testData, type = "prob") 
t <- table(factor(pred[,2]>0.5),testData$SeriousDlqin2yrs)
sum(diag(t))/dim(testData)[1]

pred <- prediction(pred[,2],testData$SeriousDlqin2yrs)
prf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(prf,colorize=TRUE)

auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#KNN
km1<-knn(trainData[c(1,3,4,6,8,9,10)],
         testData[c(1,3,4,6,8,9,10)], cl=trainData$SeriousDlqin2yrs,k=1)
t<- table(testData[,1],km1)
sum(diag(t))/dim(testData)[1]

pred <- prediction(as.integer(km1)-1, testData$SeriousDlqin2yrs) 
perf <- performance(pred,"tpr","fpr")
plot(perf, colorize = TRUE)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))

#SVMs -- needs a considerable amount of time
tune.out <- tune(svm ,as.factor(SeriousDlqin2yrs)~NumberOfTimes90DaysLate + NumberOfTime30.59DaysPastDueNotWorse + 
                   NumberOfTime60.89DaysPastDueNotWorse + age + NumberRealEstateLoansOrLines + 
                   MonthlyIncome, data = trainData, 
                 kernel ="radial",ranges=list(cost=c(0.1,1,10),
                                              gamma=c(0.5,1,2), probability = TRUE))
#summary(tune.out)
best <- tune.out$best.model
yhat.opt = predict(best,testData,probability = TRUE)
t <- table(yhat.opt,testData$SeriousDlqin2yrs)
sum(diag(t))/dim(testData)[1]

pred <- prediction(attributes(yhat.opt)$probabilities[,2], testData$SeriousDlqin2yrs) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE)))
