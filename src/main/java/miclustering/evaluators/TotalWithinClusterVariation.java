package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import org.apache.commons.math3.util.FastMath;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

public class TotalWithinClusterVariation {
    private final Instances dataset;
    private final DistanceFunction distanceFunction;
    private final DatasetCentroids datasetCentroids;

    public TotalWithinClusterVariation(Instances dataset, int maxNumClusters, DistanceFunction distanceFunction) {
        this.dataset = dataset;
        this.distanceFunction = distanceFunction;
        this.datasetCentroids = new DatasetCentroids(dataset, maxNumClusters, distanceFunction);
    }

    public double computeIndex(List<Integer> clusterAssignments, boolean parallelize) {
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);
        return computeIndex(clusterAssignments, centroids);
    }

    public double computeIndex(List<Integer> clusterAssignments, Map<Integer, Instance> centroids) {
        double twcv = 0D;
        for (int i = 0; i < dataset.numInstances(); ++i) {
            twcv += FastMath.pow(distanceFunction.distance(dataset.get(i), centroids.get(clusterAssignments.get(i))), 2);
        }
        return twcv;
    }
}
