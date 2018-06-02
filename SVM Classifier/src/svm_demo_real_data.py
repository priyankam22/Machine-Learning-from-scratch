if __name__ == '__main__':

    import load_digits_data
    import svm_classifier

    # ### Demo using real world dataset
    
    # #### RBF kernel demo with cross validation
    
    # Set the parameters
    kernel='rbf'
    params={'sigma':0.5}
    lambs = [0.01, 0.1, 1]
    folds=3
    
    # Load the data
    X_train, y_train, X_test, y_test = load_digits_data.load_digits_data()
    
    # Fit the model
    acc_rbf = svm_classifier.multiclass_svm(X_train, y_train, X_test, y_test, lambs=lambs, folds=3, kernel=kernel, params=params, 
                                                   max_iter=1000, eps=1e-5, cross_validate=True, verbose=False)
    
    # Print the accuracy
    print('Accuracy of kernel {} with params {} : {}'.format(kernel, params, acc_rbf))
    
    
    # #### Polynomial kernel demo with fixed lambda 1e-5
    
    # Set the parameters
    kernel='poly'
    params={'order':7}
    lambs = [1e-5]
    folds=3
    
    # Load the data
    X_train, y_train, X_test, y_test = load_digits_data.load_digits_data()
    
    # Fit the model
    acc_poly = svm_classifier.multiclass_svm(X_train, y_train, X_test, y_test, lambs=lambs, folds=3, kernel=kernel, params=params, 
                                               max_iter=1000, eps=1e-5, cross_validate=False, verbose=True)
    
    # Print the accuracy
    print('Accuracy of kernel {} with params {} : {}'.format(kernel, params, acc_poly))
