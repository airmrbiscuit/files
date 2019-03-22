#### Decision Trees ####

# Effectively a flow chart for classifcation, first node is the best true / false based on entropy / information gain
# Consider PCA if you have lots of features
# Using minimum value of records prevents over-fitting


library('C50')
library('gmodels')
library('tree')
library('ISLR')
library('MASS')
library('randomForest')
library('gbm')
library('rpart')
library('rattle') # rPart visuals
library('rpart.plot') # rPart visuals
library('RColorBrewer')# rPart visuals

#### Credit based analysis ####

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


#### Create Model ####
# Ultimately all of these generate the same result on this dataset...
# Need to add random forest & log reg (assume all the same outcome...)
# Then apply to an alternative dataset (FPL?)



# Remove the independent / target variable (column 17) & create model

credit_model <- C5.0(credit_train[-17], credit_train$default)

# Limit the over-fitting by using minimum volume in each category

credit_model <- C5.0(credit_train[-17], credit_train$default, control = C5.0Control(minCases = 20))

# Use boosting - Create additional trees and use as vote, tend to be shallower and then weighted
# In this instance slightly better error ratio but many more non-predicted defaults (true negative)

credit_model <- C5.0(credit_train[-17], credit_train$default, trials = 10)

# Further option is error cost, make the true negative more penalised. It works to avoid those if they are more expensive, but do miss others

error_cost <- matrix(c(0,1,4,0), nrow = 2)

credit_model <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)

# Use tree (its another CART) and uses all variables
# Use cross_validation (multiple trees) and decide which has the lowest misclassifcation level

credit_model <- tree(default~. , data = credit_train)

credit_cv = cv.tree(credit_model, FUN = prune.misclass)
plot(credit_cv)

credit_model = prune.misclass(credit_model, best = 8)


# Alternative decision tree is rPart (recursive partition) - can choose method = class for classification or anova for regression (i.e. predict a number)

credit_model <- rpart(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
                data=credit_train,
                method="class")

# rPart with a control for minimum entries on the tree

credit_model <- rpart(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
                      data=credit_train,
                      method="class",
                      control=rpart.control( minsplit = 10 ))
                      




#### Viewing the model ####


# View model (basic contents)

credit_model

# Detailed summary of tree, the confusion matrix and key attributes

summary(credit_model)

# Visual tree - not much detail on there really

plot(credit_model)
text(credit_model, pretty = 0)

# Much More attractive tree (but only for rpart)

fancyRpartPlot(credit_model)

# Actually snip parts of the tree to tidy up

credit_model <- prp(credit_model,snip=TRUE)$obj

fancyRpartPlot(credit_model)


#### Using to Predict ####



# Create a prediction on the test dataset and view

credit_pred <- predict(credit_model, credit_test)

CrossTable(credit_test$default, credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))



# Where algo generates a probability need to determine a classifcation rule

credit_pred <- predict(credit_model, credit_test)

credit_test2 <- cbind(credit_test, credit_pred)

credit_test2$pred <- ifelse(credit_test2$`1` > 0.7,1,2)

CrossTable(credit_test$default, credit_test2$pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))