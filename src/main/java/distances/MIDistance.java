package distances;

import utils.ProcessDataset;
import weka.core.*;
import weka.core.neighboursearch.PerformanceStats;

import java.util.*;

public abstract class MIDistance implements DistanceFunction {
    private DistanceFunction baseDistFun;
    private static Map<Set<Integer>, Double> cachedDistances = new HashMap<>();

    public MIDistance() {
        baseDistFun = new EuclideanDistance();
    }

    protected abstract double computeDistance(Instances i1, Instances i2);

    protected double[][] matrixDistance(Instances i1, Instances i2) {
        baseDistFun.setInstances(i1);
        int n1 = i1.numInstances();
        int n2 = i2.numInstances();

        double[][] result = new double[n1][n2];

        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                result[i][j] = baseDistFun.distance(i1.get(i), i2.get(j));
            }
        }

        return result;
    }

    @Override
    public double distance(Instance bag1, Instance bag2, PerformanceStats performanceStats) throws Exception {
        Instances i1 = ProcessDataset.extractInstances(bag1);
        Instances i2 = ProcessDataset.extractInstances(bag2);

        if (i1.numAttributes() != i2.numAttributes()) {
            throw (new Exception("Number of attributes is not equals"));
        }

        Set<Integer> key = new HashSet<>(2);
        key.add(bag1.hashCode());
        key.add(bag2.hashCode());
        double distance;
        if (cachedDistances.containsKey(key)) {
            distance = cachedDistances.get(key);
        } else {
            distance = computeDistance(i1, i2);
            cachedDistances.put(key, distance);
        }

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
        baseDistFun.setInstances(instances);
    }

    @Override
    public Instances getInstances() {
        return baseDistFun.getInstances();
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
