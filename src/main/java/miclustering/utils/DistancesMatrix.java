package miclustering.utils;

import miclustering.distances.HausdorffDistance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;

public class DistancesMatrix {

    private DistanceFunction distanceFunction;
    private boolean isDistance = false;

     public double[][] compute(Instances instances, DistanceFunction distanceFunction, boolean parallelize) {
        int numBags = instances.numInstances();
        this.distanceFunction = distanceFunction;
        if (distanceFunction instanceof HausdorffDistance) {
            // De momento sólo se ha implementado esa métrica que sea distancia, el resto son disimilaridades
            if (((HausdorffDistance) distanceFunction).getType() == HausdorffDistance.MAXMIN)
                isDistance = true;
        }

        double[][] distances = new double[numBags][numBags];

        if (parallelize) {
            ExecutorService executor = Executors.newFixedThreadPool((int) (Runtime.getRuntime().availableProcessors() * 0.25));
            Collection<Callable<Double[]>> collection = new ArrayList<>(numBags);
            for (int i = 0; i < numBags; ++i) {
                int init = isDistance ? i + 1 : 0;
                for (int j = init; j < numBags; ++j) {
                    if (i != j)
                        collection.add(new Wrapper(instances.get(i), instances.get(j), i, j));
                }
            }
            try {
                List<Future<Double[]>> futures = executor.invokeAll(collection);
                for (Future<Double[]> future : futures) {
                    Double[] result = future.get();
                    distances[result[0].intValue()][result[1].intValue()] = result[2];
                    if (isDistance)
                        distances[result[1].intValue()][result[0].intValue()] = result[2];
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            executor.shutdown();
        }else {
            for (int i = 0; i < numBags; ++i) {
                int init = isDistance ? i + 1 : 0;
                for (int j = init; j < numBags; ++j) {
                    if (i != j) {
                        distances[i][j] = distanceFunction.distance(instances.get(i), instances.get(j));
                        if (isDistance)
                            distances[j][i] = distances[i][j];
                    }
                }
            }
        }

        return distances;
    }

    private class Wrapper implements Callable<Double[]> {
        private final int aIdx;
        private final int bIdx;
        private final Instance a;
        private final Instance b;
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
