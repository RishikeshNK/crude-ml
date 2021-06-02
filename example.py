# Relative imports
from helpers.linalg import *
from helpers.linear_models import NormalEquation

# Creating synthetic data
X = [[1, 1], 
     [1, 2], 
     [2, 2], 
     [2, 3]]

y = add(dot(X, [[1], [2]]), nmat(4, 1, 3))

# Creating an instance of the NormalEquation()
norm_eqn = NormalEquation()

# Training the model with the normal equation
norm_eqn.fit(X, y)

# Viewing the parameters
print("Coefficients : ")
display(norm_eqn.coef_)

# Viewing the intercept
print("\nIntercept : ", end="")
display(norm_eqn.intercept_)
# We get y = 1 * x_0 + 2 * x_1 + 3!

# Predicting on new data
print("\nPrediction for [3, 5] : ", end="")
display(norm_eqn.predict([3, 5]))

# The answer is correct since (3 * 1) + (5 * 2) + 3 = 16.0

