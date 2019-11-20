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


public class MahalanobisDistance implements DistanceFunction {

    private static Map<Set<Integer>, Double> cachedDistances = new HashMap<>();

    @Override
    public double distance(Instance bag1, Instance bag2, PerformanceStats performanceStats) throws Exception {
        Set<Integer> key = new HashSet<>(2);
        key.add(bag1.hashCode());
        key.add(bag2.hashCode());
        double distance;
        if (cachedDistances.containsKey(key)) {
            distance = cachedDistances.get(key);
        } else {
            distance = computeDistance(bag1, bag2);
            cachedDistances.put(key, distance);
        }

        if (performanceStats != null)
            performanceStats.incrCoordCount();

        return distance;
    }

    private double computeDistance(Instance bag1, Instance bag2) {
        Instances i1 = ProcessDataset.extractInstances(bag1);
        Instances i2 = ProcessDataset.extractInstances(bag2);

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
    public double distance(Instance bag1, Instance bag2) {
        try {
            return distance(bag1, bag2, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public double distance(Instance bag1, Instance bag2, double v) {
        try {
            return distance(bag1, bag2, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public double distance(Instance bag1, Instance bag2, double v, PerformanceStats performanceStats) {
        try {
            return distance(bag1, bag2, performanceStats);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public void setInstances(Instances instances) {

    }

    @Override
    public Instances getInstances() {
        return null;
    }

    @Override
    public void setAttributeIndices(String s) {

    }

    @Override
    public String getAttributeIndices() {
        return null;
    }

    @Override
    public void setInvertSelection(boolean b) {

    }

    @Override
    public boolean getInvertSelection() {
        return false;
    }

    @Override
    public void postProcessDistances(double[] doubles) {

    }

    @Override
    public void update(Instance instance) {

    }

    @Override
    public void clean() {

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
