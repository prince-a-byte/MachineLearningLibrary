# Making the Linear Regression Model.......

import numpy as np

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