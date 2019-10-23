package distances;

import weka.core.*;
import weka.core.neighboursearch.PerformanceStats;

import java.util.*;

public class HausdorffDistance implements DistanceFunction {

    private static final int MIN = 1;
    private static final int MAX = 2;
    private static final int AVE = 3;

    private int option = MAX;

    private Instances instances;
    private String attributeIndices;
    private boolean invertSelection = false;

    @Override
    public double distance(Instance bag1, Instance bag2, PerformanceStats performanceStats) throws Exception {
        Instances i1 = preprocess(bag1);

        Instances i2;
        // Comprobar si se está calculando la distancia con un centroide directamente
        if (!bag2.attribute(1).isRelationValued() && bag2.numAttributes() == i1.numAttributes()) {
            i2 = new Instances(instances, 1);
            i2.add(bag2);
        }
        else {
            i2 = preprocess(bag2);
        }

        assert i1.numAttributes() == i2.numAttributes();

        double distance = 0D;
        switch (option) {
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
        Instances instances;
        try {
            if (bag.attribute(1).isRelationValued())
                instances = bag.relationalValue(1);
            else {
                ArrayList<Attribute> attInfo = new ArrayList<>(bag.numAttributes());
                for (int i = 0; i < bag.numAttributes(); ++i)
                    attInfo.add(new Attribute("att" + i));
                instances = new Instances("aux", attInfo, 1);
                instances.add(bag);
            }
        } catch (Exception ignored) {
            ArrayList<Attribute> attInfo = new ArrayList<>(bag.numAttributes());
            for (int i = 0; i < bag.numAttributes(); ++i)
                attInfo.add(new Attribute("att" + i));
            instances = new Instances("aux", attInfo, 1);
            instances.add(bag);
        }
        return instances;
    }

    private double maxHausdorff(Instances i1, Instances i2) {

        double[][] distances = euclideanDistance(i1, i2);

        List<Double> minByRows = new ArrayList<>(i1.size());
        for (int i = 0; i < i1.size(); ++i) {
            DoubleSummaryStatistics stat = Arrays.stream(distances[i]).summaryStatistics();
            minByRows.add(stat.getMin());
        }

        double[][] distancesTrans = transpose(distances);
        List<Double> minByCols = new ArrayList<>(i2.size());
        for (int i = 0; i < i2.size(); ++i) {
            DoubleSummaryStatistics stat = Arrays.stream(distancesTrans[i]).summaryStatistics();
            minByCols.add(stat.getMin());
        }

        return Math.max(Collections.max(minByRows), Collections.max(minByCols));
    }

    private double minHausdorff(Instances i1, Instances i2) {

        double[][] distances = euclideanDistance(i1, i2);

        List<Double> minByRows = new ArrayList<>(i1.size());
        for (int i = 0; i < i1.size(); ++i) {
            DoubleSummaryStatistics stat = Arrays.stream(distances[i]).summaryStatistics();
            minByRows.add(stat.getMin());
        }

        double[][] distancesTrans = transpose(distances);
        List<Double> minByCols = new ArrayList<>(i2.size());
        for (int i = 0; i < i2.size(); ++i) {
            DoubleSummaryStatistics stat = Arrays.stream(distancesTrans[i]).summaryStatistics();
            minByCols.add(stat.getMin());
        }

        return Math.min(Collections.min(minByRows), Collections.min(minByCols));
    }

    private double aveHausdorff(Instances i1, Instances i2) {

        double[][] distances = euclideanDistance(i1, i2);

        List<Double> minByRows = new ArrayList<>(i1.size());
        for (int i = 0; i < i1.size(); ++i) {
            DoubleSummaryStatistics stat = Arrays.stream(distances[i]).summaryStatistics();
            minByRows.add(stat.getMin());
        }

        double[][] distancesTrans = transpose(distances);

        List<Double> minByCols = new ArrayList<>(i2.size());
        for (int i = 0; i < i2.size(); ++i) {
            DoubleSummaryStatistics stat = Arrays.stream(distancesTrans[i]).summaryStatistics();
            minByCols.add(stat.getMin());
        }

        double sum1 = minByRows.stream().mapToDouble(Double::doubleValue).sum();
        double sum2 = minByCols.stream().mapToDouble(Double::doubleValue).sum();

        return (sum1 + sum2) / (i1.size() + i2.size());
    }

    private double[][] euclideanDistance(Instances i1, Instances i2) {
        int size1 = i1.size();
        int size2 = i2.size();

        double[][] distances = new double[size1][size2];
        for (int i = 0; i < size1; ++i) {
            for (int j = 0; j < size2; ++j) {
                double sum = 0D;
                for (int k = 0; k < i1.numAttributes(); ++k) {
                    double diff = i1.get(i).value(k) - i2.get(j).value(k);
                    sum += diff * diff;
                }
                distances[i][j] = Math.sqrt(sum);
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

    @Override
    public String toString() {
        String result = "ERROR";
        switch (option) {
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
        this.instances = instances;
    }

    @Override
    public Instances getInstances() {
        return instances;
    }

    @Override
    public void setAttributeIndices(String s) {
        attributeIndices = s;
    }

    @Override
    public String getAttributeIndices() {
        return attributeIndices;
    }

    @Override
    public void setInvertSelection(boolean b) {
        invertSelection = b;
    }

    @Override
    public boolean getInvertSelection() {
        return invertSelection;
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
