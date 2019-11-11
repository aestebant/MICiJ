package distances;

import utils.Matrix;
import utils.ProcessDataset;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.neighboursearch.PerformanceStats;

import java.util.Enumeration;

public class MahalanobisDistance implements DistanceFunction {

    @Override
    public double distance(Instance bag1, Instance bag2, PerformanceStats performanceStats) throws Exception {
        Instances i1 = ProcessDataset.extractInstances(bag1);
        Instances i2 = ProcessDataset.extractInstances(bag2);

        assert i1.numAttributes() == i2.numAttributes();
        int numAtt = i1.numAttributes();

        double[] mean1 = getMeans(i1);
        double[] mean2 = getMeans(i2);
        double[][] cov1 = getCovarianceMatrix(i1);
        double[][] cov2 = getCovarianceMatrix(i2);

        double[] subMeans = Matrix.sumVector(mean1, Matrix.multiplyByConstant(mean2, -1));
        double[][] sumCovs = Matrix.sumMatrix(Matrix.multiplyByConstant(cov1, 1D/2), Matrix.multiplyByConstant(cov2, 1D/2));
        double[][] invertCovs = Matrix.inverse(sumCovs);

        double[] aux = new double[numAtt];
        for (int i = 0; i < numAtt; ++i) {
            for (int j = 0; j < numAtt; ++j) {
                aux[i] += subMeans[j] * invertCovs[i][j];
            }
        }
        double distance = 0D;
        for (int i = 0; i < numAtt; ++i) {
            distance += aux[i] * subMeans[i];
        }

        if (performanceStats != null)
            performanceStats.incrCoordCount();

        return distance;
    }

    private double[] getMeans(Instances instances) {
        int nAtt = instances.numAttributes();
        double[] means = new double[nAtt];
        for (int i = 0; i < nAtt; ++i)
            means[i] = instances.meanOrMode(i);
        return means;
    }

    private double[][] getCovarianceMatrix(Instances instances) {
        int nAtt = instances.numAttributes();
        int nInst = instances.numInstances();

        double[] means = getMeans(instances);

        double[][] covariances = new double[nAtt][nAtt];
        for (int i = 0; i < nAtt; ++i) {
            covariances[i][i] = instances.variance(i);
            for (int j = i+1; j < nAtt; ++j) {
                double aux3 = 0D;
                for (int k = 0; k < nInst; ++k) {
                    double aux1 = instances.get(k).value(i) - means[i];
                    double aux2 = instances.get(k).value(j) - means[j];
                    aux3 += aux1 * aux2;
                }
                covariances[i][j] = aux3 / instances.numInstances();
                covariances[j][i] = covariances[i][j];
            }

        }
        return covariances;
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
