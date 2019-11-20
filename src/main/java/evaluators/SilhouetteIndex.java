package evaluators;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

public class SilhouetteIndex {
    private Instances instances;
    private Vector<Integer> clusterAssignments;
    private int maxNumClusters;
    private int actualNumClusters;
    private DistanceFunction distanceFunction;
    private int[] instancesPerCluster;
    private int numThreads;

    public SilhouetteIndex(Instances instances, Vector<Integer> clusterAssignments, int maxNumClusters, int actualNumClusters, DistanceFunction distanceFunction, int[] instancesPerCluster, int numThreads) {
        this.instances = instances;
        this.clusterAssignments = clusterAssignments;
        this.maxNumClusters = maxNumClusters;
        this.actualNumClusters = actualNumClusters;
        this.distanceFunction = distanceFunction;
        this.instancesPerCluster = instancesPerCluster;
        this.numThreads = numThreads;
    }

    public double computeIndex() {
        int numInstances = clusterAssignments.size();

        if (actualNumClusters == 0)
            return -1;

        double[][] distances = computeDistanceMatrix(instances);

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
        int numInstances = instances.numInstances();

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        Collection<Callable<Double[]>> collection = new ArrayList<>(numInstances);
        for (int i = 0; i < numInstances; ++i) {
            for (int j = i+1; j < numInstances; ++j) {
                collection.add(new Wrapper(instances.get(i), i, instances.get(j), j));
            }
        }
        double[][] distances = new double[numInstances][numInstances];
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
        private Instance a;
        private Instance b;
        private int aIdx;
        private int bIdx;
        Wrapper(Instance a, int aIdx, Instance b, int bIdx) {
            this.a = a;
            this.b = b;
            this.aIdx = aIdx;
            this.bIdx = bIdx;
        }
        @Override
        public Double[] call() throws Exception {
            double distance = distanceFunction.distance(a,b);
            return new Double[]{(double) aIdx, (double) bIdx, distance};
        }
    }
}
