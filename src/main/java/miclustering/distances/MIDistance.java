package miclustering.distances;

import miclustering.utils.ProcessDataset;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import weka.core.*;
import weka.core.neighboursearch.PerformanceStats;

import java.util.*;

public abstract class MIDistance implements DistanceFunction {

    protected abstract double computeDistance(Instances i1, Instances i2);

    double[][] matrixDistance(Instances i1, Instances i2) {
        int n1 = i1.numInstances();
        int n2 = i2.numInstances();
        DistanceMeasure baseDistance = new EuclideanDistance();

        double[][] result = new double[n1][n2];

        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                double[] i1Array = ProcessDataset.instanceToArray(i1.get(i));
                double[] i2Array = ProcessDataset.instanceToArray(i2.get(j));
                result[i][j] = baseDistance.compute(i1Array, i2Array);
            }
        }

        return result;
    }

    @Override
    public double distance(Instance bag1, Instance bag2, PerformanceStats performanceStats) throws Exception {
        ProcessDataset pd = new ProcessDataset();
        Instances i1 = pd.extractInstances(bag1);
        Instances i2 = pd.extractInstances(bag2);

        if (i1.numAttributes() != i2.numAttributes()) {
            throw (new Exception("Number of attributes is not equals"));
        }

        double distance = computeDistance(i1, i2);

        if (performanceStats != null)
            performanceStats.incrCoordCount();

        //System.out.println("MI distance between " + bag1.value(0) + " and " + bag2.value(0) + "-> " + distance);

        return distance;
    }

    @Override
    public double distance(Instance bag1, Instance bag2) {
        double result = 0D;
        try {
            result = this.distance(bag1, bag2, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public double distance(Instance bag1, Instance bag2, double cutOffValue) {
        return this.distance(bag1, bag2);
    }

    @Override
    public double distance(Instance bag1, Instance bag2, double cutOffValue, PerformanceStats performanceStats) {
        double result = 0D;
        try {
            result = this.distance(bag1, bag2, performanceStats);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
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
