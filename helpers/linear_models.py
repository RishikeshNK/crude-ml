from .linalg import *


class NormalEquation:

    def __init__(self, fit_intercept:bool = True):
        """
        A simple Linear Regression model fit using the Normal Equation.

            :param fit_intercept: Whether to fit an intercept term along with the model coefficients
        """
        self.fit_intercept = fit_intercept

        self.theta_ = None
        self.coef_ = None
        self.intercept_ = None

        self._is_fit = False

    def __structure_input(self, matrix):
        """
        Private function to ensure that the the data structure is a list of lists.
            :param matrix: the matrix that must be formatted

            :return: correctly formatted input
        """
        if not isinstance(matrix[0], list):
            return [matrix]
        else:
            return matrix

    def __add_ones_column(self, X):
        """
        Private function to concatenate a column of ones to the input
            :param X: input matrix

            :return: input matrix with a column of ones on the left
        """
        ones_ = ones(len(X), 1)

        return concatenate(ones_, X, axis=0)

    def __check_dims(self, X, y):
        """
        Private function to check the dimensions of the input matrices.
            :param X: data predictors
            :param y: data labels

            :return: boolean value
        """
        if len(X) != len(y):
            return False
        else:
            return True

    def __set_intercept(self, theta_):
        """
        Private function to set the intercept value from the theta_ value trained
            :param theta_: values of weights and bias found using the fit method

            :return: value of the intercept (bias)
        """
        if self.fit_intercept:
            return theta_[0]
        else:
            return [[0]]

    def __set_coefs(self, theta_):
        """
        Private function to set the coefficient values from the theta_ value trained
            :param theta_: values of weights and bias found using the fit method

            :return: values of coefficients (weights)
        """
        if self.fit_intercept:
            return theta_[1:]
        else:
            return theta_

    def fit(self, X, y):
        """
        Callable method of an instance to determine linear coefficients for the data
            :param X: data predictors
            :param y: data labels
        """
        self.X = self.__structure_input(X)
        self.y = self.__structure_input(y)

        if not self.__check_dims(X, y):
            raise ValueError(
                f"The number of rows in predictors (Size : {len(self.X), len(self.X[0])}) is not equal to the number of rows in the labels (Size : {len(self.y), len(self.y[0])})")

        if self.fit_intercept:
            self.X = self.__add_ones_column(self.X)

        self.theta_ = dot(
            dot(inverse(dot(transpose(self.X), self.X)), transpose(self.X)), self.y)

        self._is_fit = True

        self.intercept_ = self.__set_intercept(self.theta_)
        self.coef_ = self.__set_coefs(self.theta_)

    def predict(self, X_pred):
        """
        Callable method of an instance to make predictions based on the values of theta from fit method
            :param X_pred: input data to be predicted from.

            :return: predictions from model parameters
        """
        if self._is_fit == None:
            raise ValueError(
                "The values of the thetas are not set. Please call the fit method before calling the predict function.")

        self.X_pred = self.__structure_input(X_pred)

        if self.fit_intercept:
            self.X_pred = self.__add_ones_column(self.X_pred)

        return dot(self.X_pred, self.theta_)
