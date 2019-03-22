# Using machine learning for R

library('C50')
library('gmodels')

library('tree')
library('ISLR')
library('MASS')
library('randomForest')
library('GBM')

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



#### Datacamp tutorial on D Trees ####

carseats <- Carseats

# Nice histogram

hist(carseats$Sales)

# For a tree we convert into binary
# Interesting method to do it with

High = ifelse(carseats$Sales<=8, "No", "Yes")
carseats = data.frame(carseats, High)

# Create a tree, without Sales

tree.carseats <- tree(High~. -Sales, data = carseats)

# View structure and visualise

summary(tree.carseats)



# Might want to extract details from this for other purposes

tree.carseats

plot(tree.carseats)
text(tree.carseats, pretty = 0)

# Pruning / training

set.seed(101)
train=sample(1:nrow(carseats), 250)

tree.carseats = tree(High~.-Sales, carseats, subset=train)
plot(tree.carseats)
text(tree.carseats, pretty=0)

# Predict & confusion matrix
# (72 + 43) / 150 = 0.76 error rate

tree.pred = predict(tree.carseats, carseats[-train,], type="class")

with(carseats[-train,], table(tree.pred, High))

# Big tree can have too much variance so we cross-validate to prune, using the misclassification error as the basis

cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
cv.carseats

plot(cv.carseats)

# Prune the tree to pick a value in the downward steps, say 12
# Its shallower and more intelligable, but the error rate is broadly the same

prune.carseats = prune.misclass(tree.carseats, best = 12)
plot(prune.carseats)
text(prune.carseats, pretty=0)

tree.pred = predict(prune.carseats, carseats[-train,], type="class")
with(carseats[-train,], table(tree.pred, High))

# Training set

carseats_train <- carseats[1:250,]
carseats_test <- carseats[251:400,]

# Alternative Method

rf.carseats= randomForest(High~. -Sales  , data = carseats_train)
rf.carseats

test_pred_rf <- predict(rf.carseats, carseats_test, type = "class")
test_pred_rf

table(test_pred_rf, carseats_test$High)


# Comparison with tree model (maybe expect it to fit well on training)

tree.carseats2 = tree(High~.-Sales, carseats, subset = train)

tree.pred2 = predict(tree.carseats2, carseats[-train,], type="class")

with(carseats[-train,], table(tree.pred, High))

# Test applied to tree model

tree.pred4 = predict(tree.carseats2, carseats_test, type="class")

table(tree.pred4, carseats_test$High)

# Comparison 2 (as can't yet subset)

tree.carseats3 = tree(High~.-Sales, carseats_train)

tree.pred3 = predict(tree.carseats3, carseats_train, type="class")

with(carseats_train, table(tree.pred3, High))

# tree.pred2 = predict(tree.carseats2, carseats, type="class")
# carseats2 <- cbind(carseats, tree.pred2)
# with(carseats, table(tree.pred2, High))

#### Random Forest ####

# Often a single tree isn't so great for prediction errors

# Looking at RF and Boosting

boston <- Boston

# Create a training set

set.seed(101)
train = sample(1:nrow(boston), 300)

# medv = is the median value of owner-occupied homes, and other stats are about the suburbs of poston
# Runs a random forest (regression with 500 trees)
# MSR and % variance are based on out of bag estimates
# mtry us the tuning argument, which is the number of variables selected when you make a split

rf.boston = randomForest(medv~., data = boston, subset = train)
rf.boston

# for a series of random forests, here we are using 13 variables so mtry from 1 to 13
# to record errors set up two variables oob.err and test.err
# loop mtry from 1 to 13 and restrict number of trees to 350
# extract the MSE and predict, and compute test error

oob.err = double(13)
test.err = double(13)
for(mtry in 1:13){
  fit = randomForest(medv~., data = boston, subset=train, mtry=mtry, ntree = 350)
  oob.err[mtry] = fit$mse[350]
  pred = predict(fit, boston[-train,])
  test.err[mtry] = with(boston[-train,], mean( (medv-pred)^2 ))
}

# 4550 trees (13 x 350)
# matplot plot, bind the oob and test into a matrix, pch = diamon, colours = red and blue, type equals both (lines and points)

matplot(1:mtry, cbind(test.err, oob.err), pch = 23, col = c("red", "blue"), type = "b", ylab="Mean Squared Error")
legend("topright", legend = c("OOB", "Test"), pch = 23, col = c("red", "blue"))

# Ideally the two curves would line up but test error is a bit lower
# there is a lot of variability in the test error estimate
# OOB was on one dataset and test error on other, the differences are within the standard errors
# Red and blue are correlated, mtry of 4 looks optimal for test error
# Since tree has 26 MSE, and you can get to 15 with four tiers

# Using previous methodology  - don't think can tree with all that numerical


#### Boosting ####

# Boosting grows smaller and stubbier trees
# Gaussian distribution with squared error loss, 10000 shallow trees with 4 split in each, shrinkage is 0.01

boost.boston = gbm(medv~. -h_medv, data = boston[train,], distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)
summary(boost.boston)

# the resultant plot gives you variable importance- here is rm (number of rooms) and lstat (% of lower status indivs)
# plot those

plot(boost.boston,i="lstat")
plot(boost.boston,i="rm")

# Shows the higher the p. of lower status, the lower the housing price & reversed for rooms

# To apply to the test dataset
# Grid of trees in steps of 100 to 10000
# Predict using the boosted model with n.trees as argument
# dimensions of matrix are 206 test observations and 100 different predict vectors on 100 different values of tree

n.trees = seq(from = 100, to = 10000, by = 100)
predmat = predict(boost.boston, newdata = boston[-train,], n.trees = n.trees)
dim(predmat)

# Compute the test error
# predmat is a matrix, medv is a vector (so predmat - medv) is a matrix of differences
# Use apply function to the colums of those square differences (the mean) to compute the column-wise squared error for the predict vectors
# Plot using similar parameters to one used for random forest, and a boosting error plot show

boost.err = with(boston[-train,], apply( (predmat - medv)^2, 2, mean) )
plot(n.trees, boost.err, pch = 23, ylab = "Mean Squared Error", xlab = "# Trees", main = "Boosting Test Error")
abline(h = min(test.err), col = "red")

# Boosting error really drops with the volume f trees
# Shows boosting is reluctant to over-fit
# Can see that test error for boosting does get below the test error for random forest (red line)

# Boosting usually outperforms RF but RF easier to implement (less tuning parameters)

