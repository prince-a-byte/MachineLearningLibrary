# Making the Linear Regression Model.......

import numpy as np

class LinearRegression:

    def __init__(self,learning_rate=0.1,solver='ols',epochs = 1000):
        self.lr = learning_rate
        self.solver = solver
        self.coef_ = None
        self.intercept_ = None
        self.epochs = epochs

    def fit(self,X_train,y_train):

        # Checking for the solver
        if(self.solver == 'ols'):

            # inserting 1 in the X matrix in the zeroth column....
            X_train_Augmented = np.insert(X_train,0,1,axis=1)

            weights = np.linalg.inv(np.dot(X_train_Augmented.T,X_train_Augmented)).dot(X_train_Augmented.T).dot(y_train)
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]

            return self.coef_, self.intercept_



        elif(self.solver == 'gd'):
            # Initialising the weights
            weights = np.ones(X_train.shape[1] + 1)
            weights[0] = 0
            
            # Calculating the gradient
            gradient = None

            # Just need to update the weieghts matrix
            weights = weights - self.lr * gradient


        else:
            print("Invalid Solver ---> 'ols -> Ordinary Least Square', 'gd -> Gradient Descent'")

    def predict(self,X_test):

        return np.dot(X_test,self.coef_) + self.intercept_