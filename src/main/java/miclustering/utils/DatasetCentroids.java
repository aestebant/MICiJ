package miclustering.utils;

import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

public class DatasetCentroids {
    private final Instances dataset;
    private final int maxNumClusters;
    private final DistanceFunction distanceFunction;

    public DatasetCentroids(Instances dataset, int maxNumClusters, DistanceFunction distanceFunction) {
        this.dataset = dataset;
        this.maxNumClusters = maxNumClusters;
        this.distanceFunction = distanceFunction;
    }

    public Map<Integer, Instance> compute(List<Integer> clusterAssignments, boolean parallelize) {
        Map<Integer, Instances> clusters = createClusters(clusterAssignments);
        return getCentroids(clusters, parallelize);
    }

    private Map<Integer, Instances> createClusters(List<Integer> clusterAssignments) {
        Map<Integer, Instances> bagsPerCluster = new HashMap<>(maxNumClusters);
        for (int cluster = 0; cluster < maxNumClusters; ++cluster) {
            bagsPerCluster.put(cluster, new Instances(dataset, 0));
        }
        for (int i = 0; i < dataset.numInstances(); ++i) {
            if (clusterAssignments.get(i) > -1)
                bagsPerCluster.get(clusterAssignments.get(i)).add(dataset.get(i));
        }
        return bagsPerCluster;
    }

    private Map<Integer, Instance> getCentroids(Map<Integer, Instances> bagsPerCluster, boolean parallelize) {
        Map<Integer, Instance> centroids = new HashMap<>(maxNumClusters);

        if (parallelize) {
            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
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
        } else {
            for (int i = 0; i < maxNumClusters; ++i) {
                if (bagsPerCluster.get(i).numInstances() > 0)
                    centroids.put(i, computeCentroid(bagsPerCluster.get(i)));
            }
        }

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

    public Instance computeCentroid(Instances members) {
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

    public double[] distanceToCentroids(Map<Integer, Instance> centroids, int InstanceIdx) {
        double[] distances = new double[maxNumClusters];
        for (int i = 0; i < maxNumClusters; ++i) {
            if (centroids.containsKey(i))
                distances[i] = distanceFunction.distance(dataset.get(i), centroids.get(i));
        }
        return distances;
    }
}
