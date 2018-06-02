# Kernel Support Vector Machine Implementation

This is an implementation of the Kernel Support Vector Machine Classifier using Fast Gradient Descent Algorithm. The objective function used is the huberized hinge loss. It supports multiclass classification using the OneVsRest strategy where a binary classifier is trained for each class. Thus, n binary classifiers are trained in a One-Vs-Rest fashion for n-ary classification. The predicted class is chosen using the highest positive score among all classes.

The code is segregated into files as below:  
**CLASSIFIER**
svm_classifier.py : Trains a multiclass kernel based SVM model on the training data. Kernel can be either linear, polynomial or RBF (Radial Basis Function). Parameters for each type of kernel are passed through the params parameter as a dictionary. It provides a flag to enable cross validation. It takes as input data split into train and test sets. Output classes are expected to have values 0,1,2…,n.  
  
**PREPROCESSING:**  
split_standardize.py :  Used to split input data matrix X and output vector Y into X_train, y_train, X_test and y_test. It further standardizes X_train and X_test matrices.  
  
**DATASETS:**  
load_digits_data.py: Loads the digits dataset from the sklearn library for multiclass classification.
load_simulated_data.py: Generates a simulated dataset to fit the SVM Classifier.  
  
**DEMO:**  
svm_demo_real_data.py: Demo of fitting the svm_classifier to a real world dataset which is the digits dataset from sklearn.
svm_demo_simulated_data.py: Demo of fitting the svm_classifier to a simulated dataset.  
  
**PERFORMANCE:**  
svm_compare_perf.py: Compare the perform of svm_classifier with sklearn’s SVC.  
