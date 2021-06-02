def display(matrix, dp=2):
    """
    Displays the matrix in a readable format.
        :param matrix: a list of lists corresponding to the matrix
        :param dp: number of decimal places to which the elements are to be rounded
    """
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    [print([round(element, dp) for element in row]) for row in matrix]


def nmat(rows, columns, value):
    """
    Creates a matrix with all elements equal to the value entered
        :param rows: the number of rows in the matrix
        :param columns: the number of columns in the matrix
        :param value: value for each element

        :return: list of lists that correspond to the matrix
    """
    matrix = []
    while len(matrix) < rows:
        matrix.append([])
        while len(matrix[-1]) < columns:
            matrix[-1].append(value)

    return matrix


def zeros(rows, columns):
    """
    Creates a matrix with all elements as zeros.
        :param rows: the number of rows in the matrix
        :param columns: the number of columns in the matrix

        :return: list of lists that correspond to the matrix
    """
    return nmat(rows, columns, 0.0)


def ones(rows, columns):
    """
    Creates a matrix with all elements as ones.
        :param rows: the number of rows in the matrix
        :param columns: the number of columns in the matrix

        :return: list of lists that correspond to the matrix
    """
    return nmat(rows, columns, 1.0)


def identity(n):
    """
    Creates a square matrix with ones on the main diagonal.
        :param n: the number of rows and columns in square matrix
    """
    matrix = zeros(n, n)
    for i in range(n):
        matrix[i][i] = 1.0

    return matrix


def copy(matrix):
    """
    Creates a copy of the matrix passed.
        :param matrix: the matrix to be copied

        :return: a copy of the matrix passed
    """
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    rows = len(matrix)
    columns = len(matrix[0])
    copy = zeros(rows, columns)

    for i in range(rows):
        for j in range(columns):
            copy[i][j] = matrix[i][j]

    return copy


def transpose(matrix):
    """
    Computes the transpose of the matrix passed.
        :param matrix: the matrix to be transposed

        :return: transposed matrix
    """

    if not isinstance(matrix[0], list):
        matrix = [matrix]

    rows = len(matrix)
    columns = len(matrix[0])

    transpose = zeros(columns, rows)

    for i in range(rows):
        for j in range(columns):
            transpose[j][i] = matrix[i][j]

    return transpose


def add(A, B):
    """
    Adds the two matrices passed.
        :param A: the first matrix
        :param B: the second matrix

        :return: the sum of the matrices
    """
    rowsA = len(A)
    columnsA = len(A[0])
    rowsB = len(B)
    columnsB = len(B[0])

    if rowsA != rowsB or columnsA != columnsB:
        raise ArithmeticError("The two matrices are not conformable")

    matSum = zeros(rowsA, columnsB)

    for i in range(rowsA):
        for j in range(columnsB):
            matSum[i][j] = A[i][j] + B[i][j]

    return matSum


def subtract(A, B):
    """
    Subtracts matrix B from matrix A
        :param A: the first matrix
        :param B: the second matrix

        :return: the difference of the matrices
    """
    rowsA = len(A)
    columnsA = len(A[0])
    rowsB = len(B)
    columnsB = len(B[0])

    if rowsA != rowsB or columnsA != columnsB:
        raise ArithmeticError("The two matrices are not conformable")

    matDiff = zeros(rowsA, columnsB)

    for i in range(rowsA):
        for j in range(columnsB):
            matDiff[i][j] = A[i][j] - B[i][j]

    return matDiff


def dot(A, B):
    """
    Multiplies matrix A and B i.e. A * B (order matters)
        :param A: the first matrix or scalar
        :param B: the second matrix

        :return: the multiplacation of the matrices
    """
    if not isinstance(A[0], list):
        A = [A]

    if not isinstance(B[0], list):
        B = [B]

    rowsA = len(A)
    columnsA = len(A[0])
    rowsB = len(B)
    columnsB = len(B[0])

    if columnsA != rowsB:
        raise ArithmeticError(
            f"The matrices with size ({rowsA}, {columnsA}) and ({rowsB}, {columnsB}) are not conformable.")

    dotProd = zeros(rowsA, columnsB)

    for i in range(rowsA):
        for j in range(columnsB):
            sum = 0
            for k in range(columnsA):
                sum += A[i][k] * B[k][j]
            dotProd[i][j] = sum

    return dotProd


def sdot(scalar, matrix):
    """
    Multiplies the matrix with the given scalar.
        :param scalar: the scalar to multiply with
        :param matrix: the matrix to multiply with

        :return: the scalar multiplacation of the matrix with the scalar passed
    """
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    rows = len(matrix)
    columns = len(matrix[0])

    mat = zeros(rows, columns)

    for i in range(rows):
        for j in range(columns):
            mat[i][j] = scalar * matrix[i][j]

    return mat


def minor(matrix, i, j):
    """
    Computes the minor of the matrix passed at the given location
        :param matrix: matrix whose minor must be found
        :param i: the ith row to be removed to get the minor
        :param j: the jth column to be removed to get the minor

        :return: minor matrix at the given index
    """
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    rows = len(matrix)
    columns = len(matrix[0])

    if i < 1 or j < 1 or i > rows or j > rows:
        raise ArithmeticError(
            f"Minor only exists for values of i and j from 1 to {rows} and 1 to {columns} for this matrix respectively.")

    return [row[:j - 1] + row[j:] for row in (matrix[:i - 1] + matrix[i:])]


def determinant(matrix):
    """
    Computes the determinant of any dimentional square matrix.
        :param matrix: matrix whose determinant must be found

        :return: determinant of the matrix passed
    """
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    if len(matrix) != len(matrix[0]):
        raise ArithmeticError(
            f"Determinant can only be found for square matrices (NxN). Matrix passed with size ({len(matrix)}, {len(matrix[0])}) is not a square matrix")

    # For 2x2s
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # General case
    det = 0
    for column in range(len(matrix)):
        det += ((-1)**column) * matrix[0][column] * \
            determinant(minor(matrix, 1, column + 1))

    return det


def inverse(matrix):
    """
    Computes the inverse of any dimentional square matrix.
        :param matrix: matrix whose inverse must be found

        :return: inverse of the matrix passed
    """
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    if len(matrix) != len(matrix[0]):
        raise ArithmeticError(
            f"Inverse can only be found for square matrices (NxN). Matrix passed with size ({len(matrix)}, {len(matrix[0])}) is not a square matrix")

    det = determinant(matrix)

    # For 2x2
    if len(matrix) == 2:
        return [[matrix[1][1] / det, -1 * matrix[0][1] / det],
                [-1 * matrix[1][0] / det, matrix[0][0] / det]]

    # Calculating the Matrix of Cofactors
    cofactors = []

    for row in range(len(matrix)):
        cofactorRow = []
        for column in range(len(matrix)):
            matMinor = minor(matrix, row + 1, column + 1)
            cofactorRow.append(((-1)**(row + column)) * determinant(matMinor))
        cofactors.append(cofactorRow)

    # Calculating the Adjugate of the Matrix of Cofactors
    cofactors = transpose(cofactors)

    # Multiplying the Adjugate by 1 / determinant (scalar)
    for row in range(len(cofactors)):
        for column in range(len(cofactors)):
            cofactors[row][column] = cofactors[row][column] / det

    return cofactors


def norm(vector):
    """
    Computes the magnitude of the vector passed
        :param vector: vector whose magnitude must be found

        :return: the norm of the vector
    """
    if not isinstance(vector[0], list):
        vector = [vector]

    if len(vector) > 1 and len(vector[0]) > 1:
        raise ArithmeticError(
            f"The matrix passed with size ({len(vector)}, {len(vector[0])}) was not a row or a column vector.")

    norm = float(0)
    for row in vector:
        for value in row:
            norm += value ** 2

    norm = norm ** 0.5
    return norm


def unit(vector):
    """
    Computes the unit vector in the direction of the vector passed
        :param vector: vector whose unit vector must be found

        :return: unit vector of the vector passed
    """
    if not isinstance(vector[0], list):
        vector = [vector]

    rows = len(vector)
    columns = len(vector[0])

    if rows > 1 and columns > 1:
        raise ArithmeticError(
            f"The matrix passed with size ({rows}, {columns}) was not a row or a column vector.")

    magnitude = norm(vector)
    unit = copy(vector)

    for i in range(rows):
        for j in range(columns):
            unit[i][j] = float(unit[i][j] / magnitude)

    return unit


def concatenate(A, B, axis):
    """
    Concatenates the two matrices along the given axis.
        :param A: first matrix
        :param B: second matrix
        :param axis: axis along which the concatenation should occur. 0 for horizontally, 1 for vertically, and None for flattening.

        :return: concatenated matrix of the two matrices passed
    """

    if not isinstance(A[0], list):
        A = [A]

    if not isinstance(B[0], list):
        B = [B]

    if axis not in [0, 1, None]:
        raise ValueError(
            "The axis value entered is incorrect. Available values are 0, 1, or None.")

    rowsA = len(A)
    rowsB = len(B)
    columnsA = len(A[0])
    columnsB = len(B[0])

    if axis is None:
        concat = [[]]

        for row in A:
            for element in row:
                concat[0].append(element)

        for row in B:
            for element in row:
                concat[0].append(element)

        return concat

    if axis == 0:

        if rowsA != rowsB:
            raise ArithmeticError(
                f"Can not concatenate matrices with size ({rowsA}, {columnsA}) and ({rowsB}, {columnsB}) with axis=0. The rows in the two matrices must be the same.")

        concat = []

        for index in range(rowsA):
            rowTemp = A[index] + B[index]
            concat.append(rowTemp)

        return concat

    if axis == 1:

        if columnsA != columnsB:
            raise ArithmeticError(
                f"Can not concatenate matrices with size ({rowsA}, {columnsA}) and ({rowsB}, {columnsB}) with axis=1. The columns in the two matrices must be the same.")

        concat = []

        for row in A:
            concat.append(row)

        for row in B:
            concat.append(row)

        return concat
