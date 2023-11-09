package miclustering.distances;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.moment.VectorialMean;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;


public class MahalanobisDistance extends MIDistance {

    protected double computeDistance(Instances i1, Instances i2) {
        double[][] data1 = ProcessDataset.bagToMatrix(i1);
        double[][] data2 = ProcessDataset.bagToMatrix(i2);

        VectorialMean vm1 = new VectorialMean(i1.numAttributes());
        for (double[] instance : data1) vm1.increment(instance);

        VectorialMean vm2 = new VectorialMean(i2.numAttributes());
        for (double[] instance: data2) vm2.increment(instance);

        RealMatrix mean1 = new Array2DRowRealMatrix(vm1.getResult());
        RealMatrix mean2 = new Array2DRowRealMatrix(vm2.getResult());
        RealMatrix meanSub = mean1.subtract(mean2);

        RealMatrix invCov = getInvCov(data1, data2);

        RealMatrix result = meanSub.transpose().multiply(invCov).multiply(meanSub);
        return result.getData()[0][0];
    }

    private static RealMatrix getInvCov(double[][] data1, double[][] data2) {
        RealMatrix cov1;
        if (data1.length > 1) {
            Covariance getCov1 = new Covariance(data1);
            cov1 = getCov1.getCovarianceMatrix();
        } else {
            cov1 = new Array2DRowRealMatrix(new double[data1[0].length][data1[0].length]);
        }

        RealMatrix cov2;
        if (data2.length > 1) {
            Covariance getCov2 = new Covariance(data2);
            cov2 = getCov2.getCovarianceMatrix();
        } else {
            cov2 = new Array2DRowRealMatrix(new double[data2[0].length][data2[0].length]);
        }

        cov1 = cov1.scalarMultiply(0.5);
        cov2 = cov2.scalarMultiply(0.5);
        RealMatrix invCov = cov1.add(cov2);

        DecompositionSolver solver = new LUDecomposition(invCov).getSolver();
        if (solver.isNonSingular())
            invCov = solver.getInverse();
        else
            invCov = new SingularValueDecomposition(invCov).getSolver().getInverse();
        return invCov;
    }

    @Override
    public String toString() {
        return "Mahalanobis Distance";
    }
}
