package miclustering.evaluators;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

public class SilhouetteIndex {
    private int maxNumClusters;
    private DistanceFunction distanceFunction;
    private int numThreads;
    private double[][] distances;

    public SilhouetteIndex(Instances instances,  int maxNumClusters, DistanceFunction distanceFunction, int numThreads) {
        this.maxNumClusters = maxNumClusters;
        this.distanceFunction = distanceFunction;
        this.numThreads = numThreads;
        distances = computeDistanceMatrix(instances);
    }

    public double computeIndex(Vector<Integer> clusterAssignments, int actualNumClusters, int[] instancesPerCluster) {
        if (actualNumClusters == 0)
            return -1;

        int numInstances = clusterAssignments.size();
        if (instancesPerCluster == null) {
            instancesPerCluster = new int[maxNumClusters];
            for (Integer classIdx : clusterAssignments) {
                instancesPerCluster[classIdx]++;
            }
        }

        List<Double> silhouette = new ArrayList<>(numInstances);

        for (int point = 0; point < numInstances; ++point) {
            double[] meanDistToCluster = new double[maxNumClusters];
            for (int other = 0; other < numInstances; ++other) {
                if (other != point && clusterAssignments.get(other) != -1)
                    meanDistToCluster[clusterAssignments.get(other)] += distances[point][other];
            }
            for (int c = 0; c < maxNumClusters; ++c) {
                if (c == clusterAssignments.get(point))
                    meanDistToCluster[c] /= (instancesPerCluster[c] - 1);
                else
                    meanDistToCluster[c] /= instancesPerCluster[c];
            }
            double aPoint = 0;
            if (clusterAssignments.get(point) != -1)
                aPoint = meanDistToCluster[clusterAssignments.get(point)];

            List<Double> possibleB = new ArrayList<>(maxNumClusters -1);
            for (int j = 0; j < maxNumClusters; ++j) {
                if (j != clusterAssignments.get(point))
                    possibleB.add(meanDistToCluster[j]);
            }
            double bPoint = 0;
            if (possibleB.size() > 0)
                bPoint = Collections.min(possibleB);

            if (aPoint < bPoint)
                silhouette.add(1 - aPoint / bPoint);
            else if (aPoint == bPoint)
                silhouette.add(0D);
            else if (aPoint > bPoint)
                silhouette.add(bPoint / aPoint - 1);
        }
        OptionalDouble average = silhouette.stream().mapToDouble(a -> a).average();
        return average.isPresent()? average.getAsDouble() : -1;
    }

    private double[][] computeDistanceMatrix(Instances instances) {
        int numBags = instances.numInstances();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        Collection<Callable<Double[]>> collection = new ArrayList<>(numBags);
        for (int i = 0; i < numBags; ++i) {
            for (int j = i+1; j < numBags; ++j) {
                collection.add(new Wrapper(instances.get(i), instances.get(j), i, j));
            }
        }
        double[][] distances = new double[numBags][numBags];
        try {
            List<Future<Double[]>> futures = executor.invokeAll(collection);
            for (Future<Double[]> future : futures) {
                Double[] result = future.get();
                distances[result[0].intValue()][result[1].intValue()] = result[2];
                distances[result[1].intValue()][result[0].intValue()] = result[2];
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();
        return distances;
    }

    private class Wrapper implements Callable<Double[]> {
        private int aIdx, bIdx;
        private Instance a, b;
        Wrapper(Instance a, Instance b, int aIdx, int bIdx) {
            this.aIdx = aIdx;
            this.bIdx = bIdx;
            this.a = a;
            this.b = b;
        }
        @Override
        public Double[] call() throws Exception {
            double distance = distanceFunction.distance(a, b);
            return new Double[]{(double) aIdx, (double) bIdx, distance};
        }
    }
}
