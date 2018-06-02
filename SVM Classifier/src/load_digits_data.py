from sklearn import datasets
import split_standardize
    
def load_digits_data():
    '''
    Loads the in-built digits dataset of sklearn, performs train-test split and standardizes the data.
    
     Output:
         X_train : Input feature matrix for training
         y_train : Vector of output labels for training
         X_test  : Input feature matrix for testing
         y_test  : Vector of output labels for testing
    '''    
    
    digits = datasets.load_digits()

    X = digits.data
    y = digits.target

    return split_standardize.split_standardize(X, y, test_size=0.25, random_state=0)

    



