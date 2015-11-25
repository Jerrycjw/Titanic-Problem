import numpy as np
import pandas as pd
import csv as csv

from sklearn.naive_bayes import GaussianNB

# Extract the training and test data
trainingSet = pd.read_csv('train.csv', header=0)
testSet = pd.read_csv('test.csv', header=0)

# make a new column 'gender' with boolean values for each sex(will definitely use): 
trainingSet['Gender'] = trainingSet['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Creating an array of median ages, for each pclass and gender: 
median_ages = np.zeros((2,3))
# Now populate this array:
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = trainingSet[(trainingSet['Gender'] == i) & \
                              (trainingSet['Pclass'] == j+1)]['Age'].dropna().median()

# Making a copy of the Age column, so can fill in missing values without deleting original.
trainingSet['AgeFill'] = trainingSet['Age']

# Fill in the values with the median age for the corresponding gender and pclass:
for i in range(0, 2):
    for j in range(0, 3):
        trainingSet.loc[ (trainingSet.Age.isnull()) & (trainingSet.Gender == i) & (trainingSet.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

# This will round the ages to the nearest multiple of 10. Quick and dirty way of binning. Will definitely use.
trainingSet['AgeRounded'] = np.round(trainingSet['AgeFill'], -1)

# Adding a column to indicate which Age values were filled. May or may not use. 
trainingSet['AgeIsNull'] = pd.isnull(trainingSet.Age).astype(int)

# Making a new column with a numeric value for Embarked. Try using and not using this. 
# There is one missing value for Embarked, so filled them with 1 (assuming S):
trainingSet['EmbarkedInt'] = trainingSet['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).fillna(1)

# Add some feature engineering here
# Using SibSp and Parch to calculate family size (Thinking of using ticket number and family name to further engineer this):
trainingSet['FamilySize'] = trainingSet['SibSp'] + trainingSet['Parch'] + 1

# Combining age and class as an additional feature: 
trainingSet['Age*Class'] = trainingSet.AgeRounded * trainingSet.Pclass

# Stripping out titles to use for feature engineering
trainingSet['Title'] = trainingSet['Name'].str.split('.').str.get(0)
trainingSet['Title'] = trainingSet['Title'].str.split(', ').str.get(1)
trainingSet['TitleVal'] = trainingSet['Title'].map( {'Mr': 5, 'Miss': 1, 'Mlle': 1, 'Master': 1, 'Mrs': 5, 'Ms': 5, 'Mme': 5 ,'Dr': 3, 'Rev': 3, 'Col': 2, 'Major': 2, 'Don': 2, 'Capt': 2, 'Sir': 2, 'Lady': 1, 'Jonkheer': 1, 'the Countess': 1} ).astype(int)
trainingSet['Gender*Title'] = trainingSet.TitleVal * trainingSet.Gender 

# Drop remaining columns that are not numeric:
trainingSet = trainingSet.drop(['PassengerId', 'Fare', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title', 'TitleVal', 'Age', 'AgeFill', 'EmbarkedInt', 'SibSp', 'Parch'], axis=1) 

# Convert back to a numpy array:
training_set = trainingSet.values

# Okay, now to do everything I did to the training set to the test set as well:
# make a new column 'gender' with boolean values for each sex(will definitely use):  
testSet['Gender'] = testSet['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Creating an array of median ages, for each pclass and gender: 
median_ages = np.zeros((2,3))
# Now populate this array:
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = testSet[(trainingSet['Gender'] == i) & \
                              (testSet['Pclass'] == j+1)]['Age'].dropna().median()

# Making a copy of the Age column, so can fill in missing values without deleting original.
testSet['AgeFill'] = testSet['Age']

# Fill in the values with the median age for the corresponding gender and pclass:
for i in range(0, 2):
    for j in range(0, 3):
        testSet.loc[ (testSet.Age.isnull()) & (testSet.Gender == i) & (testSet.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

# This will round the ages to the nearest multiple of 10. Quick and dirty way of binning. Will definitely use.
testSet['AgeRounded'] = np.round(testSet['AgeFill'], -1) 

# Adding a column to indicate which Age values were filled. May or may not use. 
testSet['AgeIsNull'] = pd.isnull(testSet.Age).astype(int)

# Making a new column with a numeric value for Embarked. Try using and not using this. 
# There is one missing value for Embarked, so filled them with 1 (assuming S):
testSet['EmbarkedInt'] = testSet['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).fillna(1)

# Add some feature engineering here
# Using SibSp and Parch to calculate family size (Thinking of using ticket number and family name to further engineer this):
testSet['FamilySize'] = testSet['SibSp'] + testSet['Parch'] + 1 

# Combining age and class as an additional feature: 
testSet['Age*Class'] = testSet.AgeRounded * testSet.Pclass

# Stripping out titles to use for feature engineering:
testSet['Title'] = testSet['Name'].str.split('.').str.get(0)
testSet['Title'] = testSet['Title'].str.split(', ').str.get(1)
testSet['TitleVal'] = testSet['Title'].map( {'Mr': 5, 'Miss': 1, 'Master': 1, 'Mrs': 5, 'Ms': 5, 'Dr': 3, 'Rev': 3, 'Col': 2, 'Dona': 1} ).astype(int)
testSet['Gender*Title'] = testSet.TitleVal * testSet.Gender 

# Save the test IDs, before dropping:
idsTest = testSet['PassengerId'].values

# Drop remaining columns that are not numeric:
testSet = testSet.drop(['PassengerId', 'Fare', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title', 'TitleVal', 'Age', 'AgeFill', 'EmbarkedInt', 'SibSp', 'Parch'], axis=1) 

# Convert back to a numpy array:
test_set = testSet.values

# Try Naive Bayes on the numeric data:
# Declare the Naive Bayes model
nbModel = GaussianNB()

# Fit the training set to the Survived labels and create the Naive Bayes model
nbModel = nbModel.fit(training_set[0::,1::],training_set[0::,0])

GaussianNB()

# Take the same Naive Bayes model and predict on the test set
result = nbModel.predict(test_set).astype(int)

predictions_file = open("mynbGB.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(idsTest, result))
predictions_file.close()

# print nbModel.score(test_set[0::,1::], test_set[0::,0])

# print test_set[0::,1::]

# print test_set[0::,0]

# For debugging purposes:
# print result
# print training_set
# print test_set
# print testSet['Title'].value_counts()
# print trainingSet['Gender*Title']
# print idsTest
