from src.utils import Utils
import numpy as np

# generating random data..
X = np.random.randn(1000,10)
y = np.random.randint(0,3,size=1000)

X_train, X_test, y_train, y_test = Utils.train_test_split(X, y)
print(X_train)
print(X_test)
print(y_train)
print(y_test)