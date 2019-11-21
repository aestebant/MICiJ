package distances;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.moment.VectorialMean;
import utils.ProcessDataset;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.neighboursearch.PerformanceStats;

import java.util.*;


public class MahalanobisDistance extends MIDistance {

    protected double computeDistance(Instances i1, Instances i2) {
        double[][] data1 = new double[i1.numInstances()][i1.numAttributes()];
        double[][] data2 = new double[i2.numInstances()][i2.numAttributes()];

        VectorialMean vm1 = new VectorialMean(i1.numAttributes());
        VectorialMean vm2 = new VectorialMean(i2.numAttributes());

        for (int i = 0; i < data1.length; ++i) {
            for (int j = 0; j < data1[0].length; ++j) {
                data1[i][j] = i1.get(i).value(j);
            }
            vm1.increment(data1[i]);
        }
        for (int i = 0; i < data2.length; ++i) {
            for (int j = 0; j < data2[0].length; ++j) {
                data2[i][j] = i2.get(i).value(j);
            }
            vm2.increment(data2[i]);
        }

        Covariance getCov1 = new Covariance(data1);
        RealMatrix cov1 = getCov1.getCovarianceMatrix();
        Covariance getCov2 = new Covariance(data2);
        RealMatrix cov2 = getCov2.getCovarianceMatrix();

        RealMatrix mean1 = new Array2DRowRealMatrix(vm1.getResult());
        RealMatrix mean2 = new Array2DRowRealMatrix(vm2.getResult());
        RealMatrix meanSub = mean1.subtract(mean2);

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
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] strings) throws Exception {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
