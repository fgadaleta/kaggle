library(randomForest)

# load data
train <- read.csv(file="data/train.csv", header=T, sep=",")
test  <- read.csv(file="data/test.csv", header=T, sep=",")

rows_train <- train[,1]
rows_test <- test[,1]
labels <- as.factor(train[,95])
train <- train[,-95]
train <- train[,-1]
test <- test[,-1]

sel <- sample(x = 1:61878, size = 61878, replace = F)
rf <- randomForest(train[sel,], as.integer(labels[sel]), xtest=test, ntree=2500)
predictions <- levels(labels)[rf$test$predicted]
plot(rf)

############## Gradient Boosting  #######################
library(gbm)
library(dplyr)

gbm_model = gbm.fit(x = train[sel,], y =  labels[sel], 
                n.trees = 5000, distribution="multinomial",
                shrinkage = 0.001, 
                verbose = T)

summary(gbm_model)
best.iter = gbm.perf(gbm_model)
gbm_predictions = predict(gbm_model, newdata = test, n.trees = 5000, type = )



############## kernel methods  #######################
library(kernlab)

traindata = cbind(train, as.integer(labels))

kern_model = ksvm(traindata[,94]~., data= traindata, kernel="rbfdot" )
kern_predictions = predict(kern_model, test)

#######################################################
## write results to file                             ##
#######################################################
line = "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9"
write(line, file="../results/rf_submit.csv")

for(i in 1:length(predictions)) {
  line = ""
  label = strsplit(predictions[i], '_')[[1]][2]
  line = sprintf("%d",i)
  for(j in 1:9) {
    if(j==as.integer(label))
      line = sprintf("%s,1", line)
    else
      line = sprintf("%s,0", line)
  }
  write(line, file="../results/rf_submit.csv", append = TRUE)
}


