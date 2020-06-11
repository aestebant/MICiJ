package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.util.FastMath;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

public class XieBeniIndex {
    private final Instances dataset;
    private final int maxNumClusters;
    private final DistanceFunction distanceFunction;
    private final DatasetCentroids datasetCentroids;

    public XieBeniIndex(Instances dataset, int maxNumClusters, DistanceFunction distanceFunction) {
        this.dataset = dataset;
        this.maxNumClusters = maxNumClusters;
        this.distanceFunction = distanceFunction;
        this.datasetCentroids = new DatasetCentroids(dataset, maxNumClusters, distanceFunction);
    }
    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster, boolean parallelize) {
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);
        return computeIndex(clusterAssignments, bagsPerCluster, centroids);
    }
    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster, Map<Integer, Instance> centroids) {
        double[] sumDist = new double[maxNumClusters];
        for (int i = 0; i < dataset.numInstances(); ++i) {
            if (clusterAssignments.get(i) > -1)
                sumDist[clusterAssignments.get(i)] += FastMath.pow(distanceFunction.distance(dataset.get(i), centroids.get(clusterAssignments.get(i))), 2);
        }

        for (int i = 0; i < maxNumClusters; ++i)
            sumDist[i] /= bagsPerCluster[i];

        double minClusterDist = Double.POSITIVE_INFINITY;
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = i + 1; j < maxNumClusters; ++j) {
                if (bagsPerCluster[i] > 0 && bagsPerCluster[j] > 0) {
                    double distance = FastMath.pow(distanceFunction.distance(centroids.get(i), centroids.get(j)), 2);
                    if (distance < minClusterDist)
                        minClusterDist = distance;
                }
            }
        }

        Max max = new Max();
        return max.evaluate(sumDist) / minClusterDist;
    }
}
