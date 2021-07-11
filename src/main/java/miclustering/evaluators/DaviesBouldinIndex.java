package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.stream.DoubleStream;

public class DaviesBouldinIndex {
    private final Instances dataset;
    private final int maxNumClusters;
    private final DistanceFunction distanceFunction;
    private final DatasetCentroids datasetCentroids;
    private final boolean parallelize;

    public DaviesBouldinIndex(Instances dataset, int maxNumClusters, DistanceFunction distanceFunction, boolean parallelize) {
        this.dataset = dataset;
        this.maxNumClusters = maxNumClusters;
        this.distanceFunction = distanceFunction;
        this.datasetCentroids = new DatasetCentroids(dataset, maxNumClusters, distanceFunction);
        this.parallelize = parallelize;
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster, boolean parallelize) {
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);
        return computeIndex(clusterAssignments, bagsPerCluster, centroids);
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster, Map<Integer, Instance> centroids) {
        double[] sumDist = new double[maxNumClusters];

        if (parallelize) {
            ExecutorService executor = Executors.newFixedThreadPool((int) (Runtime.getRuntime().availableProcessors() * 0.25));
            Collection<Callable<Double[]>> collection = new ArrayList<>(dataset.numInstances());
            for (int i = 0; i < dataset.numInstances(); ++i) {
                Integer assignment = clusterAssignments.get(i);
                if (assignment > -1)
                    collection.add(new ParallelizeComputeIndex(dataset.get(i), centroids.get(assignment), assignment));
            }
            try {
                List<Future<Double[]>> futures = executor.invokeAll(collection);
                for (Future<Double[]> future : futures) {
                    Double[] result = future.get();
                    double sum = result[0];
                    int assignment = result[1].intValue();
                    sumDist[assignment] += sum;
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            executor.shutdown();
        } else {
            for (int i = 0; i < dataset.numInstances(); ++i) {
                int assignment = clusterAssignments.get(i);
                if (assignment > -1)
                    sumDist[assignment] += distanceFunction.distance(dataset.get(i), centroids.get(assignment));
            }
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

    private class ParallelizeComputeIndex implements Callable<Double[]> {
        Instance instance;
        Instance centroid;
        Integer assignment;
        public ParallelizeComputeIndex(Instance instance, Instance centroid, Integer assignment) {
            this.instance = instance;
            this.centroid = centroid;
            this.assignment = assignment;
        }
        @Override
        public Double[] call() throws Exception {
            double sum = distanceFunction.distance(instance, centroid);
            return new Double[]{sum, assignment.doubleValue()};
        }
    }
}
