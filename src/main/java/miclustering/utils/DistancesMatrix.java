package miclustering.utils;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;

public class DistancesMatrix {

    private DistanceFunction distanceFunction;

     public double[][] compute(Instances instances, int numThreads, DistanceFunction distanceFunction) {
        int numBags = instances.numInstances();
        this.distanceFunction = distanceFunction;

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