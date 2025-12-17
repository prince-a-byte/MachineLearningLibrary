# Making the Linear Regression Model.......

import numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures


# Some functions that are used in this code
def step(z):
    return 1 if z>0 else 0

class LinearRegression:

    def __init__(self,learning_rate=0.01,solver='ols',epochs = 1000):
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

            # Using the Ordinary Least Square method Formula to fin coefficent and intercept.
            weights = np.linalg.inv(np.dot(X_train_Augmented.T,X_train_Augmented)).dot(X_train_Augmented.T).dot(y_train)
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]

            return self.coef_, self.intercept_



        elif(self.solver == 'gd'):
            # Initialising the weights
            self.coef_ = np.ones(X_train.shape[1])
            self.intercept_ = 0

            

            # Just need to update the weieghts matrix
            for i in range(1,self.epochs):
               
                #Calculating the predicted value for each changed value of the weights.
                y_hat = np.dot(X_train,self.coef_) + self.intercept_

                # Calculating the gradient
                intercept_der = -2 * np.mean(y_train - y_hat)
                coef_der = -2 * np.dot((y_train - y_hat),X_train) / X_train.shape[0]

                # Updating the coefficent and the intercept
                self.intercept_ = self.intercept_ - self.lr * intercept_der
                self.coef_ = self.coef_ - self.lr * coef_der

                if(i % 50 == 0):
                    print(f'Epochs: {i}, Weights: {self.coef_} , Intercept: {self.intercept_}')


            return self.coef_,self.intercept_
        
        
        # Just in case User used some random solver :)
        else:
            print("Invalid Solver ---> 'ols -> Ordinary Least Square', 'gd -> Gradient Descent'")

    def predict(self,X_test):

        return np.dot(X_test,self.coef_) + self.intercept_


class PolynomialRegression:
    pass

class Perceptron:
    
    def __init__(self,learning_rate=0.1,epochs=1000):
        self.epochs = epochs
        self.lr = learning_rate
        self.weights = None


    def fit(self,X_train,y_train):

        X = np.insert(X_train,0,1,axis=1)

        # Initializing weights.
        self.weights = np.ones(X.shape[1])


        for i in range(self.epochs):

            # Selecting a random data point
            idx = np.random.randint(1,X.shape[0])

            # Calculating the Predicted value.
            y_hat = 1 if np.dot(X[idx], self.weights) > 0 else 0

            # Updating the weights
            self.weights = self.weights + self.lr * (y_train[idx] - y_hat) * X[idx]

        self.coef_ = self.weights[1:]
        self.intercept_ = self.weights[0]

  
        return self.coef_,self.intercept_

    def predict(self,X_test):
        X = np.insert(X_test,0,1,axis=1)
        y_pred = np.array([])
        for i in range(X.shape[0]):
          if np.dot(X[i],self.weights)>0 :
            y_pred = np.append(y_pred,1)
          else:
            y_pred = np.append(y_pred,0)

        return y_pred

class LogisticRegression:
    pass