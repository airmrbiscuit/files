# Using machine learning for R

library('C50')
library('gmodels')

credit <- read.csv("credit.csv")

table(credit$default)

credit$default <- as.factor(credit$default)

# Random order the data
# runif generates 100 random numbers

set.seed(12345)
credit_rand <- credit[order(runif(1000)),]

# Test & train

credit_train <- credit_rand[1:900,]
credit_test <- credit_rand[901:1000,]

# Pull out the independent / target variable, which is default and column 17
# Create model

credit_model <- C5.0(credit_train[-17], credit_train$default)

# View model 

credit_model

# If checking balance is unknown then not likely to default
# Otherwise if checking account balance is less than 0, between 1-200, 200+ and
# Credit history is good or perfect and
# There is more than one dependent then likely to default
# 358 / 44 shows that of the 358 examples, 44 were incorrectly classified

summary(credit_model)

# One of the final tables is a confusion matrix and an error rate
# But be warned if its just predicting on training set this might be over-fitted

#### Use to predict ####

credit_pred <- predict(credit_model, credit_test)

CrossTable(credit_test$default, credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# Not a great result having missed a number of defaulters
# Hence we can try and boost by using more separate decision trees

credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10)

summary(credit_boost10)

# error rate is much improved

credit_boost_pred10 <- predict(credit_boost10, credit_test)

CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# an improvement but still not picking out a lot of actual defaults (16/32)

# One option is to make some mistakes more costly than others
# We can penalise false negatives more than false positives - so to avoid the expensive decision of a loan to a default, even if do miss some good business

error_cost <- matrix(c(0,1,4,0), nrow = 2)

credit_cost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)

credit_cost_pred <- predict(credit_cost, credit_test)

CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# This does prevent a lot of the false positives, at a cost of course overall

#### Attempting new methodology - random forest ####

set.seed(415)

# With a big set you might reduce number of trees, or restrict tree complexity and rows sampled

fit_RF <- randomForest(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
                    data=credit_train, 
                    importance=TRUE, 
                    ntree=1000)

credit_RF_pred <- predict(fit_RF, credit_test)

CrossTable(credit_test$default, credit_RF_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

varImpPlot(fit_RF)

#### Alternative methodology with rpart

fit_RP <- rpart(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
             data=credit_train,
             method="class")


fancyRpartPlot(fit_RP)

credit_RP_pred <- predict(fit_RP, credit_test)

credit_RP_pred <- as.data.frame(credit_RP_pred)

credit_RP_pred$Factor <- ifelse(credit_RP_pred$`1` > 0.5,2,1)

credit_RP_pred2 <- as.factor(credit_RP_pred$Factor)

CrossTable(credit_test$default, credit_RF_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# test_RF <- as.data.frame(credit_RF_pred)
# test_RP <- as.data.frame(credit_RP_pred2)

# Values are .7 and .3 which is the split of independent variable (i.e. at highest level 30% default)
# We then break by checking_balance into two categories (from four)
# If not those things then only .14 default (which is 46% of node population)
# If you have a poor credit history then you will be .69 likelihood at the bottom, but either way not good

# Can control your own tree, rpart itself will use complexity
# check the help file but this one says a minimum of 10 items before can split, seeking to control over-fitting

fit_RP2 <- rpart(as.factor(default) ~ checking_balance + credit_history,
                data=credit_train,
                method="class",
                control=rpart.control( minsplit = 10 ))

fancyRpartPlot(fit_RP2)

# this method allows you to snip parts of the tree to tidy up

new.fit <- prp(fit_RP,snip=TRUE)$obj

fancyRpartPlot(new.fit)