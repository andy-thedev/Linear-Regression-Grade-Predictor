# Imports for Linear Regression model creation
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Imports for visualizations, and model saving
# Save models so you do not have to train every time you want to use it
# Also save models as accuracy fluctuates, and we wish to maintain the one with highest accuracy
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


# ORGANIZING THE SAMPLE/DATA-----------------------------------------------------------------------------------------


# Reads the csv file into an array
data = pd.read_csv("student-mat.csv", sep=";")

# Prints the first few lines of the array
print(data.head())

# Remove unnecessary factors
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Assign column name we wish to predict ("Answer sheet")
predict = "G3"

# Drops the prediction from the dataset, and saves it into an array.
x = np.array(data.drop([predict], 1))

# Only retrieves the prediction from the dataset, and saves it into an array
y = np.array(data[predict])


# CREATING AND TRAINING THE REGRESSION MODEL------------------------------------------------------------------------



"""
# Randomly assigns 10% of data for testing, and 90% of the data for training
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Initialize a linear regression model
linear = linear_model.LinearRegression()

# Train the model, and create a line of best fit
linear.fit(x_train, y_train)

# Determine the accuracy of the training model, using a different data set
acc = linear.score(x_test, y_test)
print(acc)

# Saves the trained model
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)
"""


# REPETITIVELY RUNNING THE TRAINING MODEL UNTIL HIGH ACCURACY ACHIEVED
# WHEN ACCURACY REQUIREMENT MET, SAVE MODEL-------------------------------------------------------------------------


# Only here due to outputs starting line 105. Remove if such outputs are not needed
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

"""
print("\nModel Accuracies:")
best = 0
# Fit regression model 30 times, and end up saving the one with the highest accuracy
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# Loads the saved model into variable linear to use
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# How to observe the constants in the line of best fit
print("\nCoefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_, "\n")


# USING THE MODEL TO PREDICT----------------------------------------------------------------------------------------


# Attempt to predict the final score
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    # Prevent unreasonable predictions
    if predictions[x] < 0:
        predictions[x] = 0
    # Observe each of:
        # The predictions of our model,
        # The input data utilized to create the predictions,
        # And the actual final grade correspondent to the input data, respectively
    print(predictions[x], x_test[x], y_test[x])


# PLOT FOR VISUALIZATION--------------------------------------------------------------------------------------------


# Change p to see correlations for other factors
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
