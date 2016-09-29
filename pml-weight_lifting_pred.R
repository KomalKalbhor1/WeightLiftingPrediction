#
# Practical Machine Learning Weight Lifting predition 
# @author: Andrei Ionut Damian
#
# In this script I have used a specific approach for data exploration and clening and
# also a specific one for model selection, training, cross-validation and finally testing
#
# For data exploration the script analyses and cleans:
#   - prediction related useless data columns such as index, person names, etc
#   - NA data predictor variables
#   - empty data predictor variables
#   - near zero variance predictor variables
#   - predictor variables correlation analysis (pair feature plot too computing expensive)
#   - train a CART model and summarize variables importance
#
# For model preparation, selection, cross-validationL
#   - from initial training dataset three subsets are generated: 
#       training, cross-validation and testing
#   - PCA is used to preprocess training data then PCA model is used to fit cross/testing
#   - a list of models is used
#   - each model is evaluated including running time
#
library(caret)

# setup the parallel processing so we can use all cores
# in order to speed up all the heavy computing tasks
library(doParallel)
library(foreach)
### Register parallel backend
avail_cores <- detectCores() # available cores
p_cluster <- makeCluster(avail_cores-1)
registerDoParallel(p_cluster)
sprintf("Cores registered: %d",getDoParWorkers())


##
##  LOADING AND CLEANING
##
# identify current script working directory
mypath <- dirname(sys.frame(1)$ofile)
# load all data in data exploration dataframe "exploreData"
# make a copy in finalData that will be used later for final cleaning process
exploreData <- read.csv(paste0(mypath,"/pml-training.csv"))
finalData <- data.frame(exploreData)



# first lets to a quick analysis of zero variance predictors
# we will display the total ammount of near-zero variance predictors
# and the head of the table containing them
# we are going to maintain a list of ALL DROPPED COLUMNS
# in order to use it for the final test dataset pre-processing
# together with the PCA model if any
varinfo <- nearZeroVar(exploreData, saveMetrics = TRUE)
sortedVarinfo <- varinfo[order(varinfo["percentUnique"]),]
sprintf("Total number of near-zero var preds: %d", sum(sortedVarinfo$nzv))
head(sortedVarinfo[sortedVarinfo$nzv==TRUE,])

#--------------

# obviously we have a lot of "unclean" data so we need to start 
# the data inspection and cleanning process
# first drop totally useless columns such as observation number, name
# then start working on the NA columns
# find if there are NA-only columns or columns with more than 95% NA
# get na column indexes then get the actual column names
# and display them.
dropped_columns <- c("X",
                     "user_name", 
                     "raw_timestamp_part_1",
                     "raw_timestamp_part_2",
                     "cvtd_timestamp")
exploreData <- exploreData[setdiff(colnames(exploreData),dropped_columns)]


##
##  NA predictors analysis
##
# perform first data cleaning removing all na columns 
na_cols <- colSums(is.na(exploreData)) >= nrow(exploreData)*0.95
na_col_names <- colnames(exploreData)[na_cols]
sprintf("NA columns: %d\n",length(na_col_names))
print(na_col_names)
nona_columns <- setdiff(colnames(exploreData),na_col_names)
exploreData <- exploreData[nona_columns]
dropped_columns <- c(dropped_columns, na_col_names)

#--------------

##
##  empty predictors analysis
##
# get columns that are actually empty (actually similar to na - more than 95% empty)
# display all of them and perform cleaning on dataset
empty_cols <- colSums(exploreData=="") >= nrow(exploreData)*0.95
empty_col_names <- colnames(exploreData)[empty_cols]
sprintf("Empty columns: %d\n",length(empty_col_names))
print(empty_col_names)
nonempty_columns <- setdiff(colnames(exploreData),empty_col_names)
exploreData <- exploreData[nonempty_columns] 
dropped_columns <- c(dropped_columns,empty_col_names)

#----------------

##
##  Predictors variance analysis
##
# now analyze again the predictors variance using nearZeroVar
# we will sort and display the predictors with least variance
# and also display all factor variables and their summary 
# omiting the label
# drop the factor variables with near-zero variance
varinfo <- nearZeroVar(exploreData, saveMetrics = TRUE)
sortedVarinfo <- varinfo[order(varinfo["percentUnique"]),]
factor_col_names <- setdiff(names(Filter(is.factor, exploreData)),c("classe"))
print("Near zero variance:")
head(sortedVarinfo)
print("Factors variables:")
print(factor_col_names)
summary(exploreData[factor_col_names])
exploreData <- exploreData[setdiff(colnames(exploreData),factor_col_names)]
dropped_columns <- c(dropped_columns, factor_col_names)

#--------------
##
## Variable correlation analysis and feature pairs plot
##
#  analyze the predictor variables correlation in order determine if we 
# have very high correlation by calculating correlation matrix
correlationMatrix <- cor(exploreData[,setdiff(colnames(exploreData),c("classe"))])
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(colnames(exploreData)[highlyCorrelated])
x <- exploreData[,setdiff(colnames(exploreData),c("classe"))]
y <- exploreData$classe
# conclusion: due to high correlation between variable we might need to apply PCA later
#
# finally train a simple model and analyze/estimate variable importance
simple.model <- train(classe ~.,
                      data = exploreData,
                      method = "rpart")
importance <- varImp(simple.model, scale=FALSE)
# summarize importance
print(importance)

#---------------



##
## TRAINING AND TESTING MODELS 
##
# now prepare training, cross-validation and test datasets 
# training dataset 60%
# crossval dataset 20%
# testing dataset 20%

# first we have the list of all dropped columns - apply it to finaData

good_columns <- setdiff(colnames(finalData), dropped_columns)
pred_columns <- setdiff(good_columns, c("classe"))
finalData <- finalData[good_columns]
inTraining <- createDataPartition(finalData$classe, p=0.6, list=FALSE)
trainingStd <- finalData[inTraining,]
testdataStd <- finalData[-inTraining,]
inVal <- createDataPartition(testdataStd$classe, p=0.5, list=FALSE)
crossvalStd <- testdataStd[inVal,]
testingStd <- testdataStd[-inVal,]
##
## Now the PCA pre-processing stage (if needed)
##
usePCA = FALSE
if (usePCA)
{
  PCA.model <- preProcess(trainingStd[pred_columns],method="pca", thresh=0.95)
  training <- predict(PCA.model, trainingStd)
  crossvalidation <- predict(PCA.model,crossvalStd )
  testing <- predict(PCA.model, testingStd)  
} else
{
  training <- trainingStd
  crossvalidation <- crossvalStd
  testing <- testingStd
}


# I will train several different models, analyze them and then and then choose 
# the best model based on best cross validation score 
# So first stage is timed training for each proposed model and cross-validations
# Keep all accuracy values in vectors then combine in dataframe to finally display

All.Methods <- c("lda","rpart","knn","lvq","xgbTree","lssvmRadial")
nr_models <- length(All.Models)
Cross.Accuracy <- c()
Training.Time <- c()
bestAccuracy <- 0 
set.seed(1234)

for (c_model in 1:nr_models){
  

  methodName <-  All.Methods[c_model]
  print(paste0("Training ",methodName,"..."))
  tmr_start <- proc.time()
  curr.model <- train(classe ~ .,
                      data = training,
                      method = methodName)
  tmr_end <- proc.time()
  print(paste0("Done training ",methodName,"."))  
  Training.Time[c_model] = (tmr_end-tmr_start)[3]
  
  preds<- predict(curr.model,crossvalidation)

  cfm <- confusionMatrix(preds,crossvalStd$classe)
  Cross.Accuracy[c_model] <- cfm$overall['Accuracy']
  
  if(bestAccuracy < Cross.Accuracy[c_model]){
    best.model <- curr.model
    bestAccuracy <- Cross.Accuracy[c_model]
  }
  
}

summary_info <- data.frame(All.Models,Cross.Accuracy,Training.Time)
summary_info <- summary_info[order(summary_info$Cross.Accuracy),]
print(summary_info)

##
## now we have our model lets apply it on testing dataset and display confusion matrix
## we can visually compare test result with cross validation one
##
print(paste("Predictic with:",best.model$method))
testpred <- predict(best.model,testing)
confusionMatrix(testpred,testingStd$classe)

##
## now finally apply best model on unseen observation
##



stopCluster(p_cluster)