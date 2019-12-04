package miclustering.distances;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;

import java.util.Arrays;

public class CauchySchwarzDistance extends MIDistance {

    @Override
    protected double computeDistance(Instances i1, Instances i2) {
        double[][] data1 = ProcessDataset.bagToMatrix(i1);
        double[][] data2 = ProcessDataset.bagToMatrix(i2);
        int n1 = data1.length;
        int n2 = data2.length;

        RealMatrix matrix1 = new Array2DRowRealMatrix(data1);
        RealMatrix matrix2 = new Array2DRowRealMatrix(data2);

        double[] sample = new double[n1 + n2];
        for (int i = 0; i < n1; ++i) sample[i] = new ArrayRealVector(data1[i]).getNorm();
        for (int i = 0; i < n2; ++i) sample[n1 + i] = new ArrayRealVector(data2[i]).getNorm();
        double h = bandwithEstimator(sample);

        double kij = 0D;
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                kij += kernelDensity(2*h, matrix1.getRowVector(i), matrix2.getRowVector(j));
            }
        }
        double kii = 0D;
        for (int i = 0; i < n1; ++i) {
            kii += kernelDensity(2*h, matrix1.getRowVector(i), matrix1.getRowVector(i));
        }
        double kjj = 0D;
        for (int j = 0; j < n2; ++j) {
            kjj += kernelDensity(2*h, matrix2.getRowVector(j), matrix2.getRowVector(j));
        }

        double result = kij / FastMath.sqrt(kii * kjj);
        result = FastMath.abs(-FastMath.log(result));

        return result;
    }

    private double kernelDensity(double sigma, RealVector instance1, RealVector instance2) {
        int n = instance1.getDimension();
        RealVector subtraction = instance1.subtract(instance2);
        double result = subtraction.dotProduct(subtraction) / (-2 * sigma * sigma);
        result = FastMath.exp(result) / FastMath.pow(2 * FastMath.PI * sigma * sigma, n/2);
        return result;
    }

    private double bandwithEstimator (double[] data) {
        int n = data.length;
        double std = new StandardDeviation().evaluate(data);
        double[] dataSort = data.clone();
        Arrays.sort(dataSort);
        double iqr = dataSort[n*3/4] - dataSort[n/4];
        return 0.99 * FastMath.min(std, iqr/1.34) / FastMath.pow(n, 0.2);
    }

    @Override
    public String toString() {
        return "Cauchy-Schwarz Distance";
    }
}
