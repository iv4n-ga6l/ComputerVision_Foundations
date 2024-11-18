"""
Create a Python script that performs matrix operations (addition, multiplication, ..etc) using NumPy.
"""

import numpy as np

class MatrixOperations:
    """A class for performing various matrix operations using NumPy."""
    
    @staticmethod
    def add_matrices(matrix1, matrix2):
        """
        Add two matrices.
        
        Args:
            matrix1 (ndarray): First input matrix
            matrix2 (ndarray): Second input matrix
            
        Returns:
            ndarray: Result of matrix addition
        """
        try:
            return np.add(matrix1, matrix2)
        except ValueError as e:
            return f"Error: Matrices must have the same dimensions. {str(e)}"

    @staticmethod
    def subtract_matrices(matrix1, matrix2):
        """
        Subtract second matrix from first matrix.
        
        Args:
            matrix1 (ndarray): First input matrix
            matrix2 (ndarray): Second input matrix
            
        Returns:
            ndarray: Result of matrix subtraction
        """
        try:
            return np.subtract(matrix1, matrix2)
        except ValueError as e:
            return f"Error: Matrices must have the same dimensions. {str(e)}"

    @staticmethod
    def multiply_matrices(matrix1, matrix2):
        """
        Multiply two matrices.
        
        Args:
            matrix1 (ndarray): First input matrix
            matrix2 (ndarray): Second input matrix
            
        Returns:
            ndarray: Result of matrix multiplication
        """
        try:
            return np.matmul(matrix1, matrix2)
        except ValueError as e:
            return f"Error: Number of columns in first matrix must equal number of rows in second matrix. {str(e)}"

    @staticmethod
    def transpose_matrix(matrix):
        """
        Transpose a matrix.
        
        Args:
            matrix (ndarray): Input matrix
            
        Returns:
            ndarray: Transposed matrix
        """
        return np.transpose(matrix)

    @staticmethod
    def inverse_matrix(matrix):
        """
        Calculate the inverse of a matrix.
        
        Args:
            matrix (ndarray): Input matrix
            
        Returns:
            ndarray: Inverse of the matrix if it exists
        """
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return "Error: Matrix is not invertible"

    @staticmethod
    def determinant(matrix):
        """
        Calculate the determinant of a matrix.
        
        Args:
            matrix (ndarray): Input matrix
            
        Returns:
            float: Determinant of the matrix
        """
        try:
            return np.linalg.det(matrix)
        except np.linalg.LinAlgError as e:
            return f"Error calculating determinant: {str(e)}"

    @staticmethod
    def eigenvalues(matrix):
        """
        Calculate the eigenvalues of a matrix.
        
        Args:
            matrix (ndarray): Input matrix
            
        Returns:
            ndarray: Array of eigenvalues
        """
        try:
            return np.linalg.eigvals(matrix)
        except np.linalg.LinAlgError as e:
            return f"Error calculating eigenvalues: {str(e)}"

if __name__ == "__main__":
    # Sample matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    matrix_ops = MatrixOperations()
    
    # Demonstrate operations
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    
    print("\nMatrix Addition (A + B):")
    print(matrix_ops.add_matrices(A, B))
    
    print("\nMatrix Multiplication (A × B):")
    print(matrix_ops.multiply_matrices(A, B))
    
    print("\nTranspose of A:")
    print(matrix_ops.transpose_matrix(A))
    
    print("\nDeterminant of A:")
    print(matrix_ops.determinant(A))
    
    print("\nInverse of A:")
    print(matrix_ops.inverse_matrix(A))
    
    print("\nEigenvalues of A:")
    print(matrix_ops.eigenvalues(A))