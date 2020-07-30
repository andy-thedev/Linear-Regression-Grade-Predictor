A repository containing a grade predictor utilizing linear regression.

Introduction:

The model takes two grades, the study time, number of failures, and absences of a high school student, and attempts to predict his/her final grade

Training dataset of students' information was collected from the UCI machine learning repository, an archive containing 557 datasets

Regression1.py -> The main algorithm

student-mat.csv -> A csv table containing student information, with rows being each student, and columns being features such as: school, sex, age, address, famsize, etc.

studentmodel.pickle -> A saved model from a previous, successful run

Design description:

1) Retrieves dataset of students, and removes presumably unnecessary features, leaving only the columns: grade 1, grade 2, grade 3, study time, number of failures, and absences

2) Separates the grade 3 column, as we wish to predict the student's final score (Grade 3) using only the first two grades, the study time, number of failures, and absences.

3) The model is fit utilizing linear regression, based on the selected features repetitively (30 times for my saved model), and the one with the highest accuracy is saved

(Trained model is saved with pickle, and the model can be loaded by remaining the code unchanged (remove comments on lines 42, and 59 to train **once** and save a model
To repetitively fit, and save model with highest accuracy, remove comments on line 69 and 84)

4) A plot algorithm for visualization is included. We may see the relationship between the first grade and the final grade, and we may compare different correlations and

visualizations by manipulating the section's parameters (variable p, vector data["G3"], and labels). This was done mainly for personal practise.

Libraries utilized: pandas, numpy, sklearn, matplotlib, pickle

Dataset utilized: student-mat.csv from the UCI (University of California, Irvine) machine learning repository

Accuracy achieved: 0.92
