package utils;

public class Matrix {

    public static double[][] multiplyByConstant(double[][] matrix, double constant) {
        int nrows = matrix.length;
        int ncols = matrix[0].length;

        double[][] result = new double[nrows][ncols];
        for (int i = 0; i < nrows; ++i) {
            result[i] = multiplyByConstant(matrix[i], constant);
        }
        return result;
    }

    public static double[] multiplyByConstant(double[] matrix, double constant) {
        int ncols = matrix.length;
        double[] result = new double[ncols];
        for (int i = 0; i < ncols; ++i)
            result[i] = matrix[i] * constant;
        return result;
    }

    public static double[][] sumMatrix(double[][] matrix1, double[][] matrix2) {
        int nrows = matrix1.length;
        int ncols = matrix1[0].length;

        assert nrows == matrix2.length && ncols == matrix2[0].length;

        double[][] result = new double[nrows][ncols];
        for (int i = 0; i < nrows; ++i)
            result[i] = sumVector(matrix1[i], matrix2[i]);

        return result;
    }

    public static double[] sumVector(double[] vector1, double[] vector2) {
        int ncols = vector1.length;

        assert ncols == vector2.length;

        double[] result = new double[ncols];
        for (int i = 0; i < ncols; ++i)
            result[i] = vector1[i] + vector2[i];

        return result;
    }

    public static double[][] transpose(double[][] matrix) {
        int nrows = matrix.length;
        int ncols = matrix[0].length;

        double[][] result = new double[nrows][ncols];
        for (int i = 0; i < nrows; ++i)
            for (int j = 0; j < ncols; ++j)
                result[j][i] = matrix[i][j];

        return result;
    }

    public static double determinant(double[][] matrix) {
        int nrows = matrix.length;
        int ncols = matrix[0].length;

        assert nrows == ncols;

        if (nrows == 2) {
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]);
        }

        double result = 0D;
        for (int i = 0; i < nrows; ++i) {
            result += (i / 2 == 0 ? 1 : -1) * matrix[0][i] * determinant(submatrix(matrix, 0, i));
        }
        return result;
    }

    public static double[][] submatrix(double[][] matrix, int excludingRow, int excludingCol) {
        int nrows = matrix.length;
        int ncols = matrix[0].length;

        int r = -1;
        double[][] result = new double[nrows-1][ncols-1];
        for (int i = 0; i < nrows; ++i) {
            if (i == excludingRow)
                continue;
            r++;
            int c = -1;
            for (int j = 0; j < ncols; ++j) {
                if (j == excludingCol)
                    continue;
                c++;
                result[r][c] = matrix[i][j];
            }
        }
        return result;
    }

    public static double[][] cofactor(double[][] matrix) {
        int nrows = matrix.length;
        int ncols = matrix[0].length;

        assert nrows == ncols;

        double[][] result = new double[nrows][ncols];
        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < ncols; ++j) {
                result[i][j] = (i / 2 == 0 ? 1 : -1) * (j / 2 == 0 ? 1 : -1) * determinant(submatrix(matrix, i , j));
            }
        }
        return result;
    }

    public static double[][] inverse(double[][] matrix) {
        double[][] result = transpose(cofactor(matrix));
        result = multiplyByConstant(result, determinant(matrix));
        return result;
    }
}
