package miclustering.utils;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

public class DatasetCentroids {
    public static Map<Integer, Instance> compute(Instances instances, int maxNumClusters, List<Integer> clusterAssignments, int nThreads) {
        Map<Integer, Instances> clusters = createClusters(instances, maxNumClusters, clusterAssignments);
        return getCentroids(maxNumClusters, clusters, nThreads);
    }

    private static Map<Integer, Instances> createClusters(Instances instances, int maxNumClusters, List<Integer> clusterAssignments) {
        Map<Integer, Instances> bagsPerCluster = new HashMap<>(maxNumClusters);
        for (int cluster = 0; cluster < maxNumClusters; ++cluster) {
            bagsPerCluster.put(cluster, new Instances(instances, 0));
        }
        for (int i = 0; i < instances.numInstances(); ++i) {
            bagsPerCluster.get(clusterAssignments.get(i)).add(instances.get(i));
        }
        return bagsPerCluster;
    }

    private static Map<Integer, Instance> getCentroids(int maxNumClusters, Map<Integer, Instances> bagsPerCluster, int nThreads) {
        Map<Integer, Instance> centroids = new HashMap<>(maxNumClusters);

        ExecutorService executor = Executors.newFixedThreadPool(nThreads);
        Collection<Callable<Map<Integer, Instance>>> collection = new ArrayList<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            if (bagsPerCluster.get(i).numInstances() > 0)
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

    private static class ParallelizeComputeCentroid implements Callable<Map<Integer, Instance>> {
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

    public static Instance computeCentroid(Instances members) {
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
