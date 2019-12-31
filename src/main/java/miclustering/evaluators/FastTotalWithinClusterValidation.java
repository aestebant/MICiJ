package miclustering.evaluators;

import org.apache.commons.math3.util.FastMath;
import weka.core.Instances;

import java.util.List;
import java.util.stream.DoubleStream;

public class FastTotalWithinClusterValidation {
    private Instances instances;
    private int maxNumClusters;
    private double max = 0D;
    private double min = Double.POSITIVE_INFINITY;

    public FastTotalWithinClusterValidation(Instances instances, int maxNumClusters) {
        this.instances = instances;
        this.maxNumClusters = maxNumClusters;
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster) {
        double[] sumsByCluster = new double[maxNumClusters];
        for (int i = 0; i < clusterAssignments.size(); ++i) {
            for (int j = 0; j < instances.get(i).relationalValue(1).numAttributes(); ++j)
                sumsByCluster[clusterAssignments.get(i)] += FastMath.pow(instances.get(i).relationalValue(1).meanOrMode(j), 2);
        }
        double result = DoubleStream.of(sumsByCluster).sum();
        for (int i = 0; i < maxNumClusters; ++i) {
            if (bagsPerCluster[i] > 0)
                result -= sumsByCluster[i] / bagsPerCluster[i];
        }
        return result;
    }

    public double selectorModification(List<Integer> clusterAssignments, int[] bagsPerCluster) {
        double ftwcv = computeIndex(clusterAssignments, bagsPerCluster);

        double e = 0D;
        for (int count : bagsPerCluster) {
            if (count > 0)
                e++;
        }
        e /= maxNumClusters;
        if (e == 1D) {
            if (ftwcv > max)
                max = ftwcv;
            if (ftwcv < min)
                min = ftwcv;
            return 1.5 * max - ftwcv;
        } else {
            if (min == Double.POSITIVE_INFINITY)
                return e;
            return min * e;
        }
    }

    public void restartMin() {
        min = Double.POSITIVE_INFINITY;
    }
}
