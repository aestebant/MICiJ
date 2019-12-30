package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import org.apache.commons.math3.util.FastMath;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

public class RMSStdDev {
    private Instances instances;
    private int maxNumClusters;
    private DistanceFunction distanceFunction;
    private int numThreads;

    public RMSStdDev(Instances instances, int maxNumClusters, DistanceFunction distanceFunction, int numThreads) {
        this.instances = instances;
        this.maxNumClusters = maxNumClusters;
        this.distanceFunction = distanceFunction;
        this.numThreads = numThreads;
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster) {
        Map<Integer, Instance> centroids = DatasetCentroids.compute(instances, maxNumClusters, clusterAssignments, numThreads);
        double rmssd = 0D;
        for (int i = 0; i < instances.numInstances(); ++i) {
            rmssd += FastMath.pow(distanceFunction.distance(instances.get(i), centroids.get(clusterAssignments.get(i))), 2);
        }
        double divisor = 0D;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (bagsPerCluster[i] > 0)
                divisor += (bagsPerCluster[i] - 1);
        }
        divisor *= instances.get(0).relationalValue(1).numAttributes();
        return FastMath.sqrt(rmssd / divisor);
    }
}
