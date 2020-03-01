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
    private DatasetCentroids datasetCentroids;

    public RMSStdDev(Instances instances, int maxNumClusters, DistanceFunction distanceFunction) {
        this.instances = instances;
        this.maxNumClusters = maxNumClusters;
        this.distanceFunction = distanceFunction;
        this.datasetCentroids = new DatasetCentroids(instances, maxNumClusters, distanceFunction);
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster, boolean parallelize) {
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);
        return computeIndex(clusterAssignments, bagsPerCluster, centroids);
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster, Map<Integer, Instance> centroids) {
        double rmssd = 0D;
        for (int i = 0; i < instances.numInstances(); ++i) {
            if (clusterAssignments.get(i) > -1)
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
