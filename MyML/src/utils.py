# ---------------------------------------------------------------------------------------------------------------------------- #

# Importing libraries.....
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------------- #

# Making first the train test split function to split test and train data.....

def train_test_split(X, y, test_size=0.2,random_state=42):

        # Using the random state to use as seed for numpy random seed.
        np.random.seed(random_state)

        # This is where we getting the total Number of indices let say = 8 and making an array of [0 to 7].....
        indicies = np.arange(X.shape[0])

        # Now we are randomly shuffling the indices here.... 
        np.random.shuffle(indicies)

        # splitting the randomly shuffled indices at the basis of test_size
        split_idx = int(X.shape[0] * (1 - test_size))

        # Now appointing the indiecs as test and train and sending the data to the respective array.
        train_idx, test_idx = indicies[:split_idx], indicies[split_idx:]
        return X[train_idx],X[test_idx], y[train_idx],y[test_idx]
    
# ---------------------------------------------------------------------------------------------------------------------------- #

# Making the mean squared error .......

def mean_squared(y_test,y_pred):
        return np.mean((y_test - y_pred)**2)

# ---------------------------------------------------------------------------------------------------------------------------- #

# Making the mean absolute error.....

def mean_absolute_error(y_test,y_pred):
        MAE = 0
        for i in range(y_test.shape[0]):
            if((y_test[i] - y_pred[i]) >= 0):
                MAE += ((y_test[i] - y_pred[i])) / y_test.shape[0]

            else:
                MAE += ((y_pred[i] - y_test[i])) / y_test.shape[0]
        return MAE
    
# ---------------------------------------------------------------------------------------------------------------------------- #

# Making Root mean Squared Error......

def root_mean_squared_error(y_test,y_pred):
        MAE = np.mean((y_test - y_pred)**2)
        RMSE = np.sqrt(MAE)  
        return RMSE
    
# ---------------------------------------------------------------------------------------------------------------------------- #

# Making the r2_score calculator........

def r2_score(y_test,y_pred):

        # By using the r2 score formula getting the denominator and numerator sperately and returning the result.
        ss_r = np.sum((y_test - y_pred) ** 2)
        ss_m = np.sum((y_test - np.mean(y_test)) ** 2)

        return (1 - (ss_r/ss_m))
    
# ---------------------------------------------------------------------------------------------------------------------------- #

# Calculating the adjusted r2_score......

def adjusted_r2_score(y_test,y_pred,no_of_independent_feature):
        # Calculating the r2 score
        r2 = r2_score(y_test,y_pred)

        r2_adjusted = 1 - (((1-r2)*(y_test.shape[0] - 1)) / (y_test.shape[0] - 1 - no_of_independent_feature))

        return r2_adjusted