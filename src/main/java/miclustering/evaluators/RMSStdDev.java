package miclustering.evaluators;

import org.apache.commons.math3.util.FastMath;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

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
        Map<Integer, Instances> clusters = createClusters(clusterAssignments);
        Map<Integer, Instance> centroids = getCentroids(clusters);
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

    private Map<Integer, Instances> createClusters(List<Integer> clusterAssignments) {
        Map<Integer, Instances> bagsPerCluster = new HashMap<>(maxNumClusters);
        for (int cluster = 0; cluster < this.maxNumClusters; ++cluster) {
            bagsPerCluster.put(cluster, new Instances(instances, 0));
        }
        for (int i = 0; i < instances.numInstances(); ++i) {
            bagsPerCluster.get(clusterAssignments.get(i)).add(instances.get(i));
        }
        return bagsPerCluster;
    }

    private Map<Integer, Instance> getCentroids(Map<Integer, Instances> bagsPerCluster) {
        Map<Integer, Instance> centroids = new HashMap<>(maxNumClusters);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        Collection<Callable<Map<Integer, Instance>>> collection = new ArrayList<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            collection.add(new ParallelizeComputeCentroid(i, bagsPerCluster.get(i)));
        }
        try {
            List<Future<Map<Integer, Instance>>> futures = executor.invokeAll(collection);
            for (Future<Map<Integer, Instance>> future : futures) {
                Map<Integer, Instance> result = future.get();
                for (Map.Entry<Integer, Instance> r : result.entrySet()) {
                    centroids.put(r.getKey(), r.getValue());
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        return centroids;
    }

    private class ParallelizeComputeCentroid implements Callable<Map<Integer, Instance>> {
        Instances cluster;
        Integer idx;
        ParallelizeComputeCentroid(Integer idx, Instances cluster) {
            this.idx = idx;
            this.cluster = cluster;
        }
        @Override
        public Map<Integer, Instance> call() throws Exception {
            Instance centroid = computeCentroid(cluster);
            Map<Integer, Instance> result = new HashMap<>();
            result.put(idx, centroid);
            return result;
        }
    }

    private Instance computeCentroid(Instances members) {
        int numInstAttributes = members.get(0).relationalValue(1).numAttributes();

        Instances aux = new Instances(members.get(0).relationalValue(1));
        for (Instance member : members) {
            aux.addAll(member.relationalValue(1));
        }

        double[] means = new double[numInstAttributes];
        for (int i = 0; i < numInstAttributes; ++i) {
            means[i] = aux.meanOrMode(i);
        }

        return new DenseInstance(1.0D, means);
    }
}
