# ### Compare performance of svm_classifier to Sklearn

import svm_classifier
import load_simulated_data
from sklearn.svm import SVC

if __name__ == '__main__':

    # Set the parameters
    kernel = 'poly'
    degree = 7
    params = {'order':degree}
    lambs  = [1e-3]
    n      = 1000   
    C      = 1/(n*lambs[0])
    folds  = 3
    
    # Load the data
    X_train, y_train, X_test, y_test = load_simulated_data.load_simulated_data(n)
    
    # Fit the model
    acc_poly = svm_classifier.multiclass_svm(X_train, y_train, X_test, y_test, lambs=lambs, folds=3, kernel=kernel, params=params, 
                                             max_iter=1000, eps=1e-5, cross_validate=False, verbose=False)
    
    # Print the accuracy
    print('Algorithm accuracy with kernel {} of degree{} and lambda {} : {}'.format(kernel, degree, lambs[0], acc_poly))
    
    svm = SVC(kernel=kernel, degree=degree, C=C).fit(X_train, y_train)
    acc_sk = svm.score(X_test, y_test)
    print('Skearn SVC accuracy of kernel {} of degree {} and params {} : {}'.format(kernel, degree, lambs[0], acc_sk))