from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
def split_standardize(X, y, test_size=0.25, random_state=0):
    ''' 
    Split the input data into train and test sets and standardize the input feature matrix
    
    Inputs:
      - X            : Input feature matrix
      - y            : Vector of output labels. Assumes class labels start from 0,1,2...          
      - test_size    : proportion of data in test set
      - random_state : Random seed for splitting data
      
     Output:
         X_train : Input feature matrix for training
         y_train : Vector of output labels for training
         X_test  : Input feature matrix for testing
         y_test  : Vector of output labels for testing
    '''
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
    print('Data splitted into X_train: {}, y_train: {}, X_test: {}, y_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    # Standardize the data
    scaler = StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    X_train = scaler.transform(X_train)
    
    return X_train, y_train, X_test, y_test

    