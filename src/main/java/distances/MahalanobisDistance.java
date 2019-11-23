package distances;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.moment.VectorialMean;
import utils.ProcessDataset;
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

        Covariance getCov1 = new Covariance(data1);
        RealMatrix cov1 = getCov1.getCovarianceMatrix();

        Covariance getCov2 = new Covariance(data2);
        RealMatrix cov2 = getCov2.getCovarianceMatrix();

        cov1 = cov1.scalarMultiply(0.5);
        cov2 = cov2.scalarMultiply(0.5);
        RealMatrix invCov = cov1.add(cov2);

        DecompositionSolver solver = new LUDecomposition(invCov).getSolver();
        if (solver.isNonSingular())
            invCov = solver.getInverse();
        else
            invCov = new SingularValueDecomposition(invCov).getSolver().getInverse();

        RealMatrix result = meanSub.transpose().multiply(invCov).multiply(meanSub);
        return result.getData()[0][0];
    }

    @Override
    public String toString() {
        return "Mahalanobis Distance";
    }
}
