package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import org.apache.commons.math3.util.FastMath;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

public class TotalWithinClusterVariation {
    private Instances instances;
    private DistanceFunction distanceFunction;
    private DatasetCentroids datasetCentroids;

    public TotalWithinClusterVariation(Instances instances, int maxNumClusters, DistanceFunction distanceFunction) {
        this.instances = instances;
        this.distanceFunction = distanceFunction;
        this.datasetCentroids = new DatasetCentroids(instances, maxNumClusters, distanceFunction);
    }

    public double computeIndex(List<Integer> clusterAssignments, boolean parallelize) {
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);
        return computeIndex(clusterAssignments, centroids);
    }

    public double computeIndex(List<Integer> clusterAssignments, Map<Integer, Instance> centroids) {
        double twcv = 0D;
        for (int i = 0; i < instances.numInstances(); ++i) {
            twcv += FastMath.pow(distanceFunction.distance(instances.get(i), centroids.get(clusterAssignments.get(i))), 2);
        }
        return twcv;
    }
}
