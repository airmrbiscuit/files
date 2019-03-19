# Titanic Practice

# Necessary packages

library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('titanic') # titanic data
library('rpart') # CART tree
library('rattle') # rPart visuals
library('rpart.plot') # rPart visuals
library('RColorBrewer')# rPart visuals



# Check the data

train <- titanic_train
test <- titanic_test

full <- bind_rows(train, test)

str(full)

# Pclass - class
# SibSp - siblings / spouse aboard
# Parch - parents / children

prop.table(table(train$Survived))

# Proportion by row

prop.table(table(train$Sex, train$Survived),1)

#### Feature engineering ####

# Extract passenger title from the passenger name

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
table(full$Sex, full$Title)

# Tidy up the range

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by sex again
table(full$Sex, full$Title)

# Finally, grab surname from passenger name
full$Surname <- sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])

# Family

# Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

# Create a family variable 
full$Family <- paste(full$Surname, full$Fsize, sep='_')

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()


# Discretize family size - too many big groups
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'


# Show family size by survival using a mosaic plot
# blue means there are more observations in the cell than would be expected under a null hypothesis (and red is the reverse)

mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)

# This variable appears to have a lot of missing values
# The passenger cabin and their deck
full$Cabin[1:28]


# The first character is the deck. For example:
strsplit(full$Cabin[2], NULL)[[1]]

# Create a Deck variable. Get passenger deck A - F:
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

#### Missing Data ####

# Small set so wouldn't delete, so either replace with average or use a prediction


# Since their fare was $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] <- 'C'

# Replace missing fare value with median fare for class/embarkment
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

# Further value imputation using MICE
# This will generate age values for NA's based on other variables

# Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 




# Save the complete output 
mice_output <- complete(mice_mod)

#Let's compare the results we get with the original distribution of passenger ages to ensure that nothing has gone completely awry.

# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Replace Age variable from the mice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

#### Additional Feature engineering ####

# Create child and mother dependent variables

# First we'll look at the relationship between age & survival
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()



# Create the column child, and indicate whether child or adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# Show counts
table(full$Child, full$Survived)


# Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

# Show counts
table(full$Mother, full$Survived)


# Finish by factorizing our two new factor variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

# Checks whether anything missing

md.pattern(full)

#### Prediction ####

# Split the data back into a train set and a test set
train <- full[1:891,]
test <- full[892:1309,]


# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)

# Show model error
# Black line shows the overall error rate
# red is the line for died, and green for survived

plot(rf_model, ylim=c(0,0.36))


# to understand significance
# plot the mean decrease in Gini calculated across all trees

# Get importance
importance    <- importance(rf_model)
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

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

#### Alternative using RPART ####

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=train,
             method="class")


fancyRpartPlot(fit)

# Shows the survival rate .62 vs .38
# If males only .19 survive (65% of sample in this group)

# Can control your own tree, rpart itself will use complexity
# check the help file but this one says a minimum of 10 items before can split, seeking to control over-fitting

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
               data=train,
               method="class",
               control=rpart.control( minsplit = 10 ))

fancyRpartPlot(fit)

# this method allows you to snip parts of the tree to tidy up

new.fit <- prp(fit,snip=TRUE)$obj

fancyRpartPlot(new.fit)


#### Further feature engineering ####

