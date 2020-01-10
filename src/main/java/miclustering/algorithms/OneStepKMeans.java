package miclustering.algorithms;

import miclustering.utils.DatasetCentroids;
import miclustering.utils.ProcessDataset;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class OneStepKMeans {
    private Instances dataset;
    private DistanceFunction distanceFunction;
    private DatasetCentroids datasetCentroids;
    private int numClusters;
    private boolean checkValidSolution;

    public OneStepKMeans(String datasetPath, String distanceClass, String distanceConfig, int numClusters, boolean checkValidSolution) {
        Instances dataset = ProcessDataset.readArff(datasetPath);
        dataset.setClassIndex(2);
        try {
            distanceFunction = (DistanceFunction) Utils.forName(DistanceFunction.class, distanceClass, Utils.splitOptions(distanceConfig));
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.dataset = dataset;
        this.numClusters = numClusters;
        this.checkValidSolution = checkValidSolution;
        datasetCentroids = new DatasetCentroids(dataset, numClusters, distanceFunction);
    }

    public OneStepKMeans(Instances dataset, DistanceFunction distanceFunction, int numClusters, boolean checkValidSolution) {
        this.dataset = dataset;
        this.distanceFunction = distanceFunction;
        this.numClusters = numClusters;
        this.checkValidSolution = checkValidSolution;
        datasetCentroids = new DatasetCentroids(dataset, numClusters, distanceFunction);
    }

    public List<Integer> evaluate (List<Integer> clusterAssignments, boolean parallelize) {
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);
        return assignBagsToClusters(centroids, parallelize);
    }

    public List<Integer> assignBagsToClusters(int[] centroidsIdx, boolean parallelize) {
        Map<Integer, Instance> centroids = new HashMap<>(centroidsIdx.length);
        for (int i = 0; i < centroidsIdx.length; ++i) {
            Instance centroid = dataset.get(centroidsIdx[i]);
            centroids.put(i, centroid);
        }
        return assignBagsToClusters(centroids, parallelize);
    }

    public List<Integer> assignBagsToClusters(Map<Integer, Instance> centroids, boolean parallelize) {
        double[][] distances = new double[dataset.numInstances()][numClusters];

        if (parallelize) {
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
        } else {
            for (int i = 0; i < dataset.numInstances(); ++i) {
                distances[i] = computeAssignation(centroids, dataset.get(i));
            }
        }

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

        if (checkValidSolution) {
            //TODO esto puede fallar si hay más de un cluster vacío y coinciden en su bolsa más cercana.
            for (int i = 0; i < numClusters; ++i) {
                if (clusterCounts[i] == 0) {
                    double min = getMin.evaluate(distMatrix.getColumn(i));
                    int closer = Arrays.stream(distMatrix.getColumn(i)).boxed().collect(Collectors.toList()).indexOf(min);
                    clusterAssignments.set(closer, i);
                }
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

    public double[] computeAssignation(Map<Integer, Instance> centroids, Instance bag) {
        double[] distances = new double[numClusters];
        for (int i = 0; i < this.numClusters; ++i) {
            distances[i] = distanceFunction.distance(bag, centroids.get(i));
        }
        return distances;
    }

    private static class ResultAssignation {
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

    public DatasetCentroids getDatasetCentroids() {
        return datasetCentroids;
    }
}
