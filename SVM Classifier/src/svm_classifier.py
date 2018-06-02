import numpy as np
import scipy
import time

from sklearn.model_selection import KFold
    
# ### SVM Algorithm using Fast Gradient Descent

# #### Compute K-Gram Matrix


def computegram(X, Z=None, kernel='poly', params={}):
    '''
    Computes the gram matrix for two input matrices using the input kernel function.
    
    Inputs: 
    - X      : Matrix with observations as rows
    - Z      : Another matrix with observations as rows
    - kernel : type of kernel - 'linear', 'poly' or 'rbf'
    - params : dictionary of parameters for the correponding kernel. 
               kernel       params keys     descr
               ---------   ------------    -------
               linear       None           No parameters expected
               rbf          sigma          Kernel Coefficient
               poly         order          Order of the polynomial
               poly         coef0          independent cofficient in kernel function
    
    Output: 
    - gram   : Gram matrix    
    
    '''
    if Z is None:
        Z = X
    # Polynomial kernel 
    if kernel == 'poly':
        gram = poly(X, Z, params)
    # Radial Basis Function Kernel
    elif kernel == 'rbf':
        gram = rbf(X, Z, params)
    # Linear Kernel
    else:
        gram = linear(X, Z, params)        

    return gram



def linear(X, Z, params):
    '''
    Computes the gram matrix for two input matrices for a linear kernel.
    
    Inputs: 
    - X      : Matrix with observations as rows
    - Z      : Another matrix with observations as rows
    - params : None expected
    
    Output: 
    - gram   : Gram matrix    
    
    '''    
    
    return np.dot(X, Z.T)



def poly(X, Z, params):
    '''
    Computes the gram matrix for two input matrices for a polynomial kernel.
    
    Inputs: 
    - X      : Matrix with observations as rows
    - Z      : Another matrix with observations as rows
    - params : dictionary of parameters for the correponding kernel. 
               params keys     descr
               ---------      ------------    
               sigma          Kernel Coefficient
    
    Output: 
    - gram   : Gram matrix    
    
    '''    
    order = params.get('order', 3)
    coef0 = params.get('coef0', 1)
    return (np.dot(X, Z.T) + coef0)**order



def rbf(X, Z, params):
    '''
    Computes the gram matrix for two input matrices for a polynomial kernel.
    
    Inputs: 
    - X      : Matrix with observations as rows
    - Z      : Another matrix with observations as rows
    - params : dictionary of parameters for the correponding kernel. 
               params keys     descr
               ---------      ------------    
               order          Order of the polynomial
               coef0          independent cofficient in kernel function
    
    Output: 
    - gram   : Gram matrix    
    
    '''   
    # Set the default sigma value to 0.5
    sigma = params.get('sigma', 0.5)
    
    # Z is a matrix with 2 dimensions
    if Z.ndim == 2:
        gram = np.exp( -1/(2*sigma**2) * np.linalg.norm(np.subtract(X[:,:,np.newaxis], Z[:,:,np.newaxis].T), axis=1)**2)
    # Z is a matrix with 1 dimension
    else: 
        gram = np.exp( -1/(2*sigma**2) * np.linalg.norm(np.subtract(X, Z.T), axis=1)**2)
                
    return gram



# #### Objective function for huberized hinge loss with L2 regularization

def obj(beta, K, y, lamb, h=0.5):
    '''
    Computes the value of the L2 regularized huberized hinge loss objective function of SVM.

    Inputs:
    - beta : Vector of coefficients to be optimized
    - K    : Gram matrix consisting of evaluations of the kernel k(x_i, x_j) for i,j=1,...,n
    - y    : Labels y_1,...,y_n corresponding to x_1,...,x_n
    - lamb : Penalty parameter lambda
    - h    : huberized hinge loss parameter
    
    Output:
    - grad : Value of the objective function at beta
    '''
    
    K_beta = np.dot(K, beta) 
    yt  = y*K_beta
    
    # Compute the huberized hinge loss   
    lhh = ((1+h-yt)**2)/(4*h)*(np.abs(1-yt) <= h) + (1-yt)*(yt<(1-h))
    
    # Gradient of huberized hinge loss function and L2 penalty term   
    obj = np.mean(lhh) + lamb*beta.dot(K_beta)
    
    return obj


# #### Compute gradient of the objective function


def computegrad(beta, K, y, lamb, h=0.5):
    '''
    Computes the gradient of the L2 regularized huberized hinge loss objective function of SVM.
    
    Inputs:
    - beta : Vector of coefficients to be optimized
    - K    : Gram matrix of dimensions consisting of evaluations of the kernel k(x_i, x_j) for i,j=1,...,n
    - y    : Labels y_1,...,y_n corresponding to x_1,...,x_n
    - lamb : Regularization parameter lambda
    - h    : huberized hinge loss parameter    
   
    Output:
    - grad : Value of the gradient at beta    
    
    '''
    
    K_beta = np.dot(K,beta) 
    yt = y*K_beta
    
    # Compute the huberized hinge loss
    lhh = -(1+h-yt)/(2*h)*y*(np.abs(1-yt) <= h) -y*(yt<(1-h))
    
    # Gradient of huberized hinge loss function and L2 penalty term
    grad = np.mean(lhh[:,np.newaxis]*K, axis=0) + 2*lamb*K_beta

    return grad



# #### Backtracking line search for step size


def backtracking_eta(beta, K, y, lamb, eta_init=1, a=0.5, b=0.1, maxiter=1000):
    """
    Performs backtracking line search to find the optimal step size for a given beta
    
    Inputs:
      - beta     : vector of coefficients with current values
      - K        : Gram matrix
      - y        : Vector of output labels
      - lamb     : l2 regularization parameter
      - eta_init : Starting (maximum) step size            
      - a        : Constant used to define sufficient decrease condition
      - b        : Fraction by which we decrease eta if the previous eta doesn't work
      - maxiter  : Maximum number of iterations to run the algorithm
      
    Output:
      - eta      : Optimal step size to use
          
    """    
    # Compute gradient and its norm
    grad_beta = computegrad(beta, K, y, lamb)
    norm_grad_beta = np.linalg.norm(grad_beta)
    
    # Compute current objective value
    obj_curr = obj(beta, K, y, lamb)

    # Initialize variables
    eta = eta_init 
    found_eta = 0
    iter = 0
        
    # Iterate till optimal step size is found or maximum iterations are reached.
    while found_eta == 0 and iter < maxiter:
        
        # Compute new objective
        obj_next = obj(beta - eta*grad_beta, K, y, lamb)
        
        if obj_next < obj_curr - a*eta*norm_grad_beta**2:
            # Found the optimum step size
            found_eta = 1
        elif iter == maxiter-1:
            # Max number of iterations reached
            eta = eta_init
            break
        else:
            # Reduce eta by a factor of b
            eta *= b
            iter += 1
            
    return eta


# #### Fast gradient algorithm


def fastgradalgo(beta_init, theta_init, K, y, lamb=0, eta_init=1, max_iter=1000, eps=1e-5):
    '''
    Fast gradient descent algorithm for finding the optimal beta coefficients.
    
    Inputs:
      - beta_init  : Starting values of beta coefficients
      - theta_init : Starting values of theta coefficients
      - K          : Gram matrix
      - y          : Vector of output labels
      - lamb       : l2 regularization parameter    
      - max_iter   : Maximum number of iterations to run the algorithm      
      - eps        : Value for convergence criterion for the norm of the gradient
    
    Output:
      - beta_vals  : Matrix of estimated beta's at each iteration,
                     with the most recent values in the last row.
    
    '''       
    # Initialize variables
    beta  = beta_init
    theta = theta_init
    eta   = eta_init
    beta_vals = beta
    iter = 0
    
    # Compute gradients
    grad_theta = computegrad(theta, K, y, lamb)
    grad_beta = computegrad(beta, K, y, lamb)
    
    # Update beta till gradient is lesser than epsilon or maximum iterations are reached   
    while iter < max_iter and np.linalg.norm(grad_beta) > eps:
        
        # Get optimal step size
        eta = backtracking_eta(theta, K, y, lamb, eta_init=eta)
        
        # Compute new beta and theta
        beta_new = theta - eta*grad_theta
        theta = beta_new + iter/(iter + 3) * (beta_new - beta)
        beta = beta_new
        
        # Append the new beta values to the output beta matrix
        beta_vals = np.vstack((beta_vals, beta_new))
        
        # Recompute the gradient using new beta and theta
        grad_beta  = computegrad(beta, K, y, lamb)
        grad_theta = computegrad(theta, K, y, lamb)
        
        iter += 1
                      
    return beta_vals


# #### Train SVM 


def train_svm(X, y, lamb=0, kernel='poly', params={}, max_iter=1000, eps=1e-3, normalize=True):
    '''
    Trains the kernel based SVM model on the training data.
    
    Inputs:
      - X          : Input feature matrix
      - y          : Vector of output labels    
      - lamb       : l2 regularization parameter    
      - kernel     : type of kernel function - 'linear', 'poly' or 'rbf'
      - params     : dictionary of parameters for the correponding kernel. 
                     kernel       params keys     descr
                     ---------   ------------    -------
                     linear       None           No parameters expected
                     rbf          sigma          Kernel Coefficient
                     poly         order          Order of the polynomial
                     poly         coef0          independent cofficient in kernel function   
      - max_iter   : Maximum number of iterations to run the algorithm      
      - eps        : Value for convergence criterion for the norm of the gradient
      - normalize  : Normalize the K gram matrix after evaluation
    
    Output:
      - beta_vals  : Matrix of estimated beta's at each iteration,
                     with the most recent values in the last row.
    '''
    # Initialize the vectors
    n = len(y)   
    beta_init = np.zeros(n)
    theta_init = np.zeros(n)

    # Compute the K-gram matrix
    K = computegram(X, X, kernel=kernel, params=params)
    
    # Normalize the gram matrix
    if normalize:
        K_diag = np.diag(K).copy()
        K = K/np.sqrt(np.outer(K_diag, K_diag))
    
    # Set eta_init based on an upper bound on the Lipschitz constant
    try:
        eta_init = 1 / scipy.linalg.eigh(2 / n * np.dot(K, K) + 2 * lamb * K, eigvals=(n - 1, n - 1),eigvals_only=True)[0]
    except np.linalg.LinAlgError:
        eta_init = 1
    
    # Run the fast gradient algorithm to optimize the beta coefficents
    beta_vals = fastgradalgo(beta_init, theta_init, K, y, lamb, eta_init, max_iter, eps)
    
    return beta_vals


# #### Predict labels


def predict(X_trn, X_tst, beta, output='class', kernel='poly', params={}):
    """
    Predict the output class or score for given X and beta
    
    Input:
      - X_trn   : Input features of training data
      - X_tst   : Input features of test data
      - beta    : Regression coefficients to be used
      - output  : 'class' for class labels
                  'score' for score of prediction
    Output:
      - y_pred  : array of predicted class labels or scores
      
    """    
    # Initialize variables
    n = len(X_tst)
    y_vals = np.zeros(n)
    
    # Get the score of each test point
    for i in range(n):
        y_vals[i] = np.dot(computegram(X_trn, X_tst[i], kernel=kernel, params=params).T, beta)
        
    # Output the class
    if output == 'class':
        return np.sign(y_vals) 
    # Output the score
    else:
        return y_vals


# #### Compute misclassification errors


def misclassification_error(X_trn, X_tst, y_tst, beta, kernel='poly', params={}):
    ''' 
    Calculates the misclassification error for binary classes.
    
    Inputs:
      - X_trn   : Input feature matrix for training
      - X_tst   : Input feature matrix for testing
      - y_tst   : Vector of output labels for testing. Assumes class labels start from 0,1,2...  
      - beta    : Estimated coefficients to evaluate
      - kernel  : type of kernel function - 'linear', 'poly' or 'rbf'
      - params  : dictionary of parameters for the correponding kernel. 
                     kernel       params keys     descr
                     ---------   ------------    -------
                     linear       None           No parameters expected
                     rbf          sigma          Kernel Coefficient
                     poly         order          Order of the polynomial
                     poly         coef0          independent cofficient in kernel function   
   
    Output:
      - error   : Misclassification error on test set   
    '''
    # Make the predictions
    y_preds = predict(X_trn, X_tst, beta, output='class', kernel=kernel, params=params)
    
    # Compute the misclassification error
    error = np.mean(y_tst != y_preds) 
    return error



# #### Predict for multiclass using OneVsRest classifier outputs


def predict_class(X_trn, X_tst, beta_vals, kernel='poly', params={}):
    ''' 
    Returns the predicted class using the majority vote across all one-vs-rest classifiers.
    
    Inputs:
      - X_trn       : Input feature matrix for training
      - X_tst       : Input feature matrix for testing
      - beta_vals   : Matrix of estimated coefficients for all one-vs-rest classifiers
      - kernel      : type of kernel function - 'linear', 'poly' or 'rbf'
      - params      : dictionary of parameters for the correponding kernel. 
                       kernel       params keys     descr
                       ---------   ------------    -------
                       linear       None           No parameters expected
                       rbf          sigma          Kernel Coefficient
                       poly         order          Order of the polynomial
                       poly         coef0          independent cofficient in kernel function   
   
    Output:
      - final_preds : Misclassification error on test set      
    '''
    # Initialize variables
    all_preds = []
    final_preds = []    
    n = X_tst.shape[0]

    # Get predictions for each classifier
    for beta in beta_vals:
        y_preds = predict(X_trn, X_tst, beta, output='score', kernel=kernel, params=params)
        
        # Store the magnitude of the score if the specific class has a positive vote else set to zero
        all_preds.append([score if np.sign(score) == 1 else 0 for score in y_preds])

    all_pred_arr = np.array(all_preds)
   
    # Get the majority vote from the predictions using the magnitude of score
    final_preds = np.argmax(all_pred_arr, axis=0)

    return final_preds


# ### Cross Validation

# #### K Fold cross validation


def run_kfold(X, y, lamb=0, folds=3, kernel='poly', params={}, max_iter=1000, eps=1e-5, verbose=False):
    '''
    Runs kfold validation for specified lambda
  
    Inputs:
      - X         : Input feature matrix
      - y         : Vector of output labels. Assumes class labels start from 0,1,2...          
      - lamb      : l2 regularization parameter
      - folds     : Folds for K Fold cross validation
      - kernel    : type of kernel function - 'linear', 'poly' or 'rbf'
      - params    : dictionary of parameters for the correponding kernel. 
                         kernel       params keys     descr
                         ---------   ------------    -------
                         linear       None           No parameters expected
                         rbf          sigma          Kernel Coefficient
                         poly         order          Order of the polynomial
                         poly         coef0          independent cofficient in kernel function   
      - max_iter  : Maximum number of iterations to run the algorithm      
      - eps       : Value for convergence criterion for the norm of the gradient
      - verbose   : Print debugging information if true
    
    Output:
      - acc       : List of accuracies for all K Folds
    '''
    # Initialize list for storing accuracies for all K Folds
    acc = []
    
    # Run KFold to get K folds of feature matrix
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    kf.get_n_splits(X)

    # Fit SVM on every fold to get the accuracy
    for train_index, test_index in kf.split(X):
        
        # Extract the fold of data
        X_trn, X_val = X[train_index], X[test_index]
        y_trn, y_val = y[train_index], y[test_index]

        # Fit the SVM model
        beta_vals = train_svm(X_trn, y_trn, lamb=lamb, kernel=kernel, params=params, max_iter=max_iter, eps=eps)

        # Get optimal beta
        beta_opt = beta_vals[-1]

        # Compute the misclassification error and accuracy
        error = misclassification_error(X_trn, X_val, y_val, beta=beta_opt, kernel=kernel, params=params)
        acc.append(1-error)

    return acc


# #### Train SVM using cross validation for binary classes


def train_svm_cv(X, y, lambs=[0], folds=3, kernel='poly', params={}, max_iter=1000, eps=1e-5, verbose=False):
    '''
    Perform k-fold cross validation using SVM fast gradient algorithm with huberized hinge loss and l2 regularization
    
    Inputs:
      - X         : Input feature matrix
      - y         : Vector of output labels. Assumes class labels start from 0,1,2...          
      - lambs     : a list of l2 regularization parameters to cross-validate. 
                    Only first element is used if cross_validate=False   
      - folds     : Folds for K Fold cross validation
      - kernel    : type of kernel function - 'linear', 'poly' or 'rbf'
      - params    : dictionary of parameters for the correponding kernel. 
                         kernel       params keys     descr
                         ---------   ------------    -------
                         linear       None           No parameters expected
                         rbf          sigma          Kernel Coefficient
                         poly         order          Order of the polynomial
                         poly         coef0          independent cofficient in kernel function   
      - max_iter  : Maximum number of iterations to run the algorithm      
      - eps       : Value for convergence criterion for the norm of the gradient
      - verbose   : Print debugging information if true
    
    Output:
      - beta_lamb : Optimal lambda to be used
      
    '''    
    # Initialize variables
    start_time = time.time()
    max_acc = 0
    best_lamb = 0
    
    # Loop over all lambda values and choose the one with best accuracy
    for l in lambs:
        
        if verbose:
            print("Checking lambda: ", l)
        
        # Run K-Fold cross validation to get the accuracies for all K folds
        accuracy = run_kfold(X, y, lamb=l, folds=folds, kernel=kernel, params=params, max_iter=max_iter, eps=eps, verbose=verbose)
        
        # Compute the mean accuracy
        mean_acc = np.mean(accuracy)
        if verbose:
            print("Accuracy for lambda {} : {}".format(l, mean_acc))
            
        # Update the best lambda and accuracy if a better accuracy was achieved
        if  mean_acc > max_acc:
            max_acc = mean_acc
            best_lamb = l

    if verbose:
        time_elapsed = time.time() - start_time
        print("Total time elapsed (mins):", time_elapsed/60)
    
    return best_lamb


# #### Train multiclass SVM

def multiclass_svm(X_trn, y_trn, X_tst, y_tst, lambs=[0], folds=3, kernel='poly', params={}, 
                   max_iter=1000, eps=1e-5, cross_validate=True, verbose=False):
    ''' 
    Trains a kernel SVM classifier with L2 regularized huberized hinge loss using cross validation.
    
    Inputs:
      - X_trn          : Input feature matrix for training
      - y_trn          : Vector of output labels for training. Assumes class labels start from 0,1,2...          
      - X_tst          : Input feature matrix for testing
      - y_tst          : Vector of output labels for testing. Assumes class labels start from 0,1,2...    
      - lambs          : a list of l2 regularization parameters to cross-validate. 
                         Only first element is used if cross_validate=False   
      - folds          : Folds for K Fold cross validation
      - kernel         : type of kernel function - 'linear', 'poly' or 'rbf'
      - params         : dictionary of parameters for the correponding kernel. 
                         kernel       params keys     descr
                         ---------   ------------    -------
                         linear       None           No parameters expected
                         rbf          sigma          Kernel Coefficient
                         poly         order          Order of the polynomial
                         poly         coef0          independent cofficient in kernel function   
      - max_iter       : Maximum number of iterations to run the algorithm      
      - eps            : Value for convergence criterion for the norm of the gradient
      - normalize      : Normalize the K gram matrix after evaluation
      - cross_validate : Use cross validation if True, else fit the first lambda value passed
      - verbose   : Print debugging information if true      
    
    Output:
      - beta_vals      : Matrix of estimated beta's at each iteration,
                         with the most recent values in the last row.    
    '''
    # Initialize list to store optimal betas for all classes
    opt_betas = []
    
    # Number of classes to categorize assuming classes have values in [0,1,2,..n]
    num_classes = np.max(y_trn) + 1
    if verbose:
        print('Number of classifiers to build: ', num_classes)

    start_time = time.time()
    
    # For binary classification, only one classifier is required
    if num_classes == 2:
        num_classes = 1
    
    # Build classifier for each class
    for i in range(num_classes):
        if verbose:
            print("Building classifier for class {}..".format(i))
        X_train = X_trn
        y_train = np.array([1 if y_i == i else -1 for y_i in y_trn])
        
        if cross_validate:
            # Perform K Fold cross validation to get optimal lambda
            if verbose:
                print("Running {}-fold cross validation..".format(folds))
                
            lamb = train_svm_cv(X_train, y_train, lambs=lambs, folds=folds, kernel=kernel, 
                                params=params, max_iter=max_iter, eps=eps, verbose=verbose)
            if verbose:
                print("Optimal lambda for class {} is :{}".format(i, lamb))
                
        else:
            # Use the first element from input list as the lambda value
            lamb = lambs[0]

        # Fit the SVM on the full training data using the lambda derived above
        opt_beta_vals = train_svm(X_train, y_train, lamb=lamb, kernel=kernel, params=params, max_iter=max_iter, eps=eps)
        
        # Append the optimal betas of last iteration for each classifier
        opt_betas.append(opt_beta_vals[-1])
        
        if verbose:
            time_elapsed = time.time() - start_time
            print("Time elapsed (mins):", time_elapsed/60)
        
    # Make final predictions
    final_preds = predict_class(X_trn, X_tst, opt_betas, kernel=kernel, params=params) 
        
    # Compute accuracy
    accuracy = np.mean(final_preds == y_tst)
        
    if verbose:
        print("Accuracy on test data using {} kernel with params {}: {}".format(kernel, params, accuracy))
            
    return accuracy
    
