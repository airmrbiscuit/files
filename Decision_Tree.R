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
library('dplyr') # data manipulation
library('ggplot2')
library('ggthemes') # visualization


#### Credit based analysis ####

credit <- read.csv("credit.csv")

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



#1 Remove the independent / target variable (column 17) & create model

credit_model <- C5.0(credit_train[-17], credit_train$default)

#2 Limit the over-fitting by using minimum volume in each category

credit_model <- C5.0(credit_train[-17], credit_train$default, control = C5.0Control(minCases = 20))

#3 Use boosting - Create additional trees and use as vote, tend to be shallower and then weighted
# In this instance slightly better error ratio but many more non-predicted defaults (true negative)

credit_model <- C5.0(credit_train[-17], credit_train$default, trials = 10)

#4 Further option is error cost, make the true negative more penalised. It works to avoid those if they are more expensive, but do miss others

error_cost <- matrix(c(0,1,4,0), nrow = 2)

credit_model <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)

#5 Use tree (its another CART) and uses all variables
#6 Use cross_validation (multiple trees) and decide which has the lowest misclassifcation level

credit_model <- tree(default~. , data = credit_train)

credit_cv = cv.tree(credit_model, FUN = prune.misclass)
plot(credit_cv)

credit_model = prune.misclass(credit_model, best = 8)


#7 Alternative decision tree is rPart (recursive partition) - can choose method = class for classification or anova for regression (i.e. predict a number)

credit_model <- rpart(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
                data=credit_train,
                method="class")

#8 rPart with a control for minimum entries on the tree

credit_model <- rpart(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
                      data=credit_train,
                      method="class",
                      control=rpart.control( minsplit = 10 ))
                      

#9 Random Forest

credit_model <- randomForest(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
                      data=credit_train)




#10 Series of random forests (mtry) which does generate the most correct

credit_model <- randomForest(as.factor(default) ~ checking_balance + credit_history + savings_balance + foreign_worker + months_loan_duration + other_debtors + job,
                             data=credit_train, mtry = 7)




#11 Boosting grows smaller and stubbier trees - Gaussian distribution with squared error loss, 10000 shallow trees with 4 split in each, shrinkage is 0.01
# equally good result as the multiple RF

credit_model = gbm(default~. , data = credit_train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)



# Ideally would run a couple at once (which is what the pros are doing)
# Explanation of the maths within the downloaded book



#### Viewing the model ####


# View model (basic contents)

credit_model

# Detailed summary

summary(credit_model)

# Visual tree - not much detail on there really
# Green lne for 2, red for 1 and black line is overall

plot(credit_model)
text(credit_model, pretty = 0)

#plot(credit_model, ylim=c(0,0.36)) # alternative view

# Much More attractive tree (but only for rpart)

fancyRpartPlot(credit_model)

# Actually snip parts of the tree to tidy up

credit_model <- prp(credit_model,snip=TRUE)$obj

fancyRpartPlot(credit_model)

# View factor significance (random forest)

importance    <- randomForest::importance(credit_model)

varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))


# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

legend('topright', colnames(credit_model$err.rate), col=1:3, fill=1:3)







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