package miclustering.algorithms;

import miclustering.utils.ProcessDataset;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.core.*;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class OneStepKMeans {
    private Instances dataset;
    private DistanceFunction distanceFunction;
    private int numClusters;

    public OneStepKMeans(String datasetPath, String distanceClass, String distanceConfig, int numClusters) {
        Instances dataset = ProcessDataset.readArff(datasetPath);
        dataset.setClassIndex(2);
        try {
            distanceFunction = (DistanceFunction) Utils.forName(DistanceFunction.class, distanceClass, Utils.splitOptions(distanceConfig));
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.dataset = dataset;
        this.numClusters = numClusters;
    }

    public List<Integer> evaluate (List<Integer> clusterAssignments) {
        Map<Integer, Instances> bagsPerCluster = createClusters(clusterAssignments);
        Map<Integer, Instance> centroids = getCentroids(bagsPerCluster);
        return assignBagsToClusters(centroids);
    }

    private Map<Integer, Instances> createClusters(List<Integer> clusterAssignments) {
        Map<Integer, Instances> bagsPerCluster = new HashMap<>(numClusters);
        for (int cluster = 0; cluster < this.numClusters; ++cluster) {
            bagsPerCluster.put(cluster, new Instances(dataset, 0));
        }
        for (int i = 0; i < dataset.numInstances(); ++i) {
            bagsPerCluster.get(clusterAssignments.get(i)).add(dataset.get(i));
        }
        return bagsPerCluster;
    }

    private Map<Integer, Instance> getCentroids(Map<Integer, Instances> bagsPerCluster) {
        Map<Integer, Instance> centroids = new HashMap<>(numClusters);

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Collection<Callable<Map<Integer, Instance>>> collection = new ArrayList<>(numClusters);
        for (int i = 0; i < numClusters; ++i) {
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

    private List<Integer> assignBagsToClusters(Map<Integer, Instance> centroids) {
        double[][] distances = new double[dataset.numInstances()][numClusters];

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Collection<Callable<ResultAssignation>> collection = new ArrayList<>(dataset.numInstances());
        for (int i = 0; i < dataset.numInstances(); ++i) {
            collection.add(new ParallelizeAssignation(centroids, dataset.get(i), i));
        }
        try {
            List<Future<ResultAssignation>> futures = executor.invokeAll(collection);
            for (Future<ResultAssignation> future : futures) {
                ResultAssignation result = future.get();
                distances[result.getBagId()] = result.getDistances();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        RealMatrix distMatrix = new Array2DRowRealMatrix(distances);
        List<Integer> clusterAssignments = new ArrayList<>(dataset.numInstances());
        int[] clusterCounts = new int[numClusters];
        Min getMin = new Min();
        for (int i = 0; i < dataset.numInstances(); ++i) {
            double min = getMin.evaluate(distMatrix.getRow(i));
            int clusterIdx = Arrays.stream(distances[i]).boxed().collect(Collectors.toList()).indexOf(min);
            clusterAssignments.add(clusterIdx);
            clusterCounts[clusterIdx]++;
        }

        //TODO esto puede fallar si hay más de un cluster vacío y coinciden en su bolsa más cercana.
        for (int i = 0; i < numClusters; ++i) {
            if (clusterCounts[i] == 0) {
                double min = getMin.evaluate(distMatrix.getColumn(i));
                int closer = Arrays.stream(distMatrix.getColumn(i)).boxed().collect(Collectors.toList()).indexOf(min);
                clusterAssignments.set(closer, i);
            }
        }

        return clusterAssignments;
    }

    private class ParallelizeAssignation implements Callable<ResultAssignation> {
        Map<Integer, Instance> centroids;
        Instance bag;
        int idx;
        ParallelizeAssignation(Map<Integer, Instance> centroids, Instance bag, int idx) {
            this.centroids = centroids;
            this.bag = bag;
            this.idx = idx;
        }
        @Override
        public ResultAssignation call() throws Exception {
            double[] distances = computeAssignation(centroids, bag);
            return new ResultAssignation(idx, distances);
        }
    }

    private double[] computeAssignation(Map<Integer, Instance> centroids, Instance bag) {
        double[] distances = new double[numClusters];
        for (int i = 0; i < this.numClusters; ++i) {
            distances[i] = distanceFunction.distance(bag, centroids.get(i));
        }
        return distances;
    }

    private class ResultAssignation {
        int bagId;
        double[] distances;

        public ResultAssignation(int bagId, double[] distances) {
            this.bagId = bagId;
            this.distances = distances;
        }

        public int getBagId() {
            return bagId;
        }

        public double[] getDistances() {
            return distances;
        }
    }
}
