package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import org.apache.commons.math3.util.FastMath;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

public class RMSStdDev {
    private final Instances dataset;
    private final int maxNumClusters;
    private final DistanceFunction distanceFunction;
    private final DatasetCentroids datasetCentroids;
    private final boolean parallelize;

    public RMSStdDev(Instances dataset, int maxNumClusters, DistanceFunction distanceFunction, boolean parallelize) {
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
        double rmssd = 0D;
        ExecutorService executor;
        if (parallelize) {
            executor = Executors.newFixedThreadPool((int) (Runtime.getRuntime().availableProcessors() * 0.25));
        } else {
            executor = Executors.newFixedThreadPool(1);
        }
        Collection<Callable<Double>> collection = new ArrayList<>(dataset.numInstances());
        for (int i = 0; i < dataset.numInstances(); ++i) {
            int assignment = clusterAssignments.get(i);
            if (assignment > -1)
                collection.add(new ParallelizeComputeIndex(dataset.get(i), centroids.get(assignment)));
        }
        try {
            List<Future<Double>> futures = executor.invokeAll(collection);
            for (Future<Double> future : futures)
                rmssd += future.get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        double divisor = 0D;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (bagsPerCluster[i] > 0)
                divisor += (bagsPerCluster[i] - 1);
        }
        divisor *= dataset.get(0).relationalValue(1).numAttributes();
        return FastMath.sqrt(rmssd / divisor);
    }

    private class ParallelizeComputeIndex implements Callable<Double> {
        Instance instance;
        Instance centroid;

        public ParallelizeComputeIndex(Instance instance, Instance centroid) {
            this.instance = instance;
            this.centroid = centroid;
        }

        @Override
        public Double call() throws Exception {
            return FastMath.pow(distanceFunction.distance(instance, centroid), 2);
        }
    }
}
