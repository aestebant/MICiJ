package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;

public class DaviesBouldinIndex {
    private final Instances dataset;
    private final int maxNumClusters;
    private final DistanceFunction distanceFunction;
    private final DatasetCentroids datasetCentroids;

    public DaviesBouldinIndex(Instances dataset, int maxNumClusters, DistanceFunction distanceFunction) {
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
                sumDist[clusterAssignments.get(i)] += distanceFunction.distance(dataset.get(i), centroids.get(clusterAssignments.get(i)));
        }
        for (int i = 0; i < maxNumClusters; ++i)
            sumDist[i] /= bagsPerCluster[i];

        double[][] dividend = new double[maxNumClusters][maxNumClusters];
        double[][] divisor = new double[maxNumClusters][maxNumClusters];
        for (int i = 0; i < maxNumClusters; ++i) {
            dividend[i][i] = Double.NEGATIVE_INFINITY;
            divisor[i][i] = Double.POSITIVE_INFINITY;
            for (int j = i + 1; j < maxNumClusters; ++j) {
                if (bagsPerCluster[i] > 0 && bagsPerCluster[j] > 0) {
                    dividend[i][j] = sumDist[i] + sumDist[j];
                    dividend[j][i] = dividend[i][j];

                    divisor[i][j] = distanceFunction.distance(centroids.get(i), centroids.get(j));
                    divisor[j][i] = divisor[i][j];
                }
            }
        }

        double[] db = new double[maxNumClusters];
        for (int i = 0; i < maxNumClusters; ++i) {
            Max max = new Max();
            Min min = new Min();
            db[i] = max.evaluate(dividend[i]) / min.evaluate(divisor[i]);
        }

        return DoubleStream.of(db).sum() / maxNumClusters;
    }
}
