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

public class TotalWithinClusterVariation {
    private final Instances dataset;
    private final DistanceFunction distanceFunction;
    private final DatasetCentroids datasetCentroids;
    private final boolean parallelize;

    public TotalWithinClusterVariation(Instances dataset, int maxNumClusters, DistanceFunction distanceFunction, boolean parallelize) {
        this.dataset = dataset;
        this.distanceFunction = distanceFunction;
        this.datasetCentroids = new DatasetCentroids(dataset, maxNumClusters, distanceFunction);
        this.parallelize = parallelize;
    }

    public double computeIndex(List<Integer> clusterAssignments, boolean parallelize) {
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);
        return computeIndex(clusterAssignments, centroids);
    }

    public double computeIndex(List<Integer> clusterAssignments, Map<Integer, Instance> centroids) {
        double twcv = 0D;

        if(parallelize) {
            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            Collection<Callable<Double>> collection = new ArrayList<>(dataset.numInstances());
            for (int i = 0; i < dataset.numInstances(); ++i)
                collection.add(new ParallelizeComputeIndex(dataset.get(i), centroids.get(clusterAssignments.get(i))));
            try {
                List<Future<Double>> futures = executor.invokeAll(collection);
                for (Future<Double> future : futures) {
                    twcv += future.get();
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        } else {
            for (int i = 0; i < dataset.numInstances(); ++i) {
                twcv += FastMath.pow(distanceFunction.distance(dataset.get(i), centroids.get(clusterAssignments.get(i))), 2);
            }
        }
        return twcv;
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
