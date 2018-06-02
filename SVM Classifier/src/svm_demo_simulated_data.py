import split_standardize
import numpy as np
import matplotlib.pyplot as plt
import svm_classifier
import load_simulated_data

if __name__ == '__main__':

    print("Testing svm_classifer using RBF kernel with sigma=0.5 on simulated data with cross validation..")
    # Set the parameters
    kernel='rbf'
    params={'sigma':0.5}
    lambs = [0.01, 0.1, 1]
    folds=3
    
    # Load the data
    X_train, y_train, X_test, y_test = load_simulated_data.load_simulated_data(n=1000)
    
    # Fit the model
    acc_rbf = svm_classifier.multiclass_svm(X_train, y_train, X_test, y_test, lambs=lambs, folds=3, kernel=kernel, params=params, 
                                            max_iter=1000, eps=1e-5, cross_validate=True, verbose=False)
    
    # Print the accuracy
    print('Accuracy of kernel {} with params {} : {}'.format(kernel, params, acc_rbf))
    
    
    ##### Polynomial Kernel of Order 7
    
    print("Testing svm_classifer using polynomial kernel of order 7 on simulated data with lambda 1e-3..")
    
    # Set the parameters
    kernel='poly'
    params={'order':7}
    lambs = [1e-3]
    folds=3
    
    # Load the data
    X_train, y_train, X_test, y_test = load_simulated_data(n=1000)
    
    # Fit the model
    acc_poly = svm_classifier.multiclass_svm(X_train, y_train, X_test, y_test, lambs=lambs, folds=3, kernel=kernel, params=params, 
                                     max_iter=1000, eps=1e-5, cross_validate=False, verbose=False)
    
    # Print the accuracy
    print('Accuracy of kernel {} with params {} : {}'.format(kernel, params, acc_poly))
    
    
