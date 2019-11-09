package distances;

import weka.core.*;
import weka.core.neighboursearch.PerformanceStats;

import java.util.*;

public class HausdorffDistance implements DistanceFunction {

    private static final int MIN = 1;
    private static final int MAX = 2;
    private static final int AVE = 3;

    private int type = AVE;
    private final DistanceFunction df;

    public HausdorffDistance() {
        df = new EuclideanDistance();
    }

    public HausdorffDistance(Instances instances) {
        df = new EuclideanDistance(instances);
    }

    @Override
    public double distance(Instance bag1, Instance bag2, PerformanceStats performanceStats) throws Exception {
        Instances i1 = preprocess(bag1);
        Instances i2 = preprocess(bag2);

        assert i1.numAttributes() == i2.numAttributes();

        double distance = 0D;
        switch (type) {
            case MIN:
                distance = minHausdorff(i1, i2);
                break;
            case MAX:
                distance = maxHausdorff(i1, i2);
                break;
            case AVE:
                distance = aveHausdorff(i1, i2);
        }

        if (performanceStats != null)
            performanceStats.incrCoordCount();

        //System.out.println("Hausdorff distance between " + bag1.value(0) + " and " + bag2.value(0) + "-> " + distance);

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

    private Instances preprocess(Instance bag) {
        boolean isBag = false;
        try {
            isBag = bag.attribute(1).isRelationValued();
        } catch (Exception ignored) {
            //Capturar excepción Instance doesn't have access to a dataset e ignorarla.
        }

        if (isBag)
            return new Instances(bag.relationalValue(1));
        else {
            ArrayList<Attribute> attributes = new ArrayList<>(bag.numAttributes());
            for (int i = 0; i < bag.numAttributes(); ++i) {
                attributes.add(new Attribute("att" + i));
            }
            Instances instances = new Instances("aux", attributes, 1);
            instances.add(bag);
            return instances;
        }
    }

    private double maxHausdorff(Instances i1, Instances i2) {
        double[][] distances = computeDistancesMatrix(i1, i2);
        List<Double> minByRows = getMinByRows(distances);
        double[][] distancesTrans = transpose(distances);
        List<Double> minByCols = getMinByRows(distancesTrans);
        return Math.max(Collections.max(minByRows), Collections.max(minByCols));
    }

    private double minHausdorff(Instances i1, Instances i2) {
        double[][] distances = computeDistancesMatrix(i1, i2);
        List<Double> minByRows = getMinByRows(distances);
        double[][] distancesTrans = transpose(distances);
        List<Double> minByCols = getMinByRows(distancesTrans);
        return Math.min(Collections.min(minByRows), Collections.min(minByCols));
    }

    private double aveHausdorff(Instances i1, Instances i2) {
        double[][] distances = computeDistancesMatrix(i1, i2);
        List<Double> minByRows = getMinByRows(distances);
        double[][] distancesTrans = transpose(distances);
        List<Double> minByCols = getMinByRows(distancesTrans);
        double sum1 = minByRows.stream().mapToDouble(Double::doubleValue).sum();
        double sum2 = minByCols.stream().mapToDouble(Double::doubleValue).sum();
        return (sum1 + sum2) / (i1.size() + i2.size());
    }

    private double[][] computeDistancesMatrix(Instances i1, Instances i2) {
        int size1 = i1.size();
        int size2 = i2.size();
        double[][] distances = new double[size1][size2];
        for (int i = 0; i < size1; ++i) {
            for (int j = 0; j < size2; ++j) {
                distances[i][j] = df.distance(i1.get(i), i2.get(j));
            }
        }
        return distances;
    }

    private double[][] transpose(double[][] matrix) {
        int nrows = matrix.length;
        int ncols = matrix[0].length;

        double[][] result = new double[ncols][nrows];

        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < ncols; ++j) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    private List<Double> getMinByRows(double[][] matrix) {
        List<Double> minByRows = new ArrayList<>(matrix.length);
        for (double[] row : matrix) {
            DoubleSummaryStatistics stat = Arrays.stream(row).summaryStatistics();
            minByRows.add(stat.getMin());
        }
        return minByRows;
    }

    @Override
    public String toString() {
        String result = "ERROR";
        switch (type) {
            case MIN:
                result = "Minimal Hausdorff Distance";
                break;
            case MAX:
                result = "Maximal Hausdorff Distance";
                break;
            case AVE:
                result = "Average Hausdorff Distance";
                break;
        }
        return result;
    }

    private void printDistances(double[][] distances) {
        for (double[] distance : distances) {
            for (double v : distance) System.out.printf("%.2f ", v);
            System.out.println();
        }
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
    public void setInstances(Instances instances) {
        df.setInstances(instances.get(0).relationalValue(1));
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
    public Enumeration<Option> listOptions() {
        Vector<Option> vector = new Vector<>();
        vector.add(new Option("Type of distance (default average).", "hausdorff-type", 1, "-hausdorff-type <average|minimal|maximal>"));
        return vector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String type = Utils.getOption("hausdorff-type", options);
        if (type.equals("minimal"))
            this.type = MIN;
        else if (type.equals("maximal"))
            this.type = MAX;
        else
            this.type = AVE;
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
