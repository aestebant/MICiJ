package miclustering.algorithms;

import miclustering.utils.ProcessDataset;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

public class BAMIC extends MIKMeans {

    @Override
    protected void randomInit(Instances data) throws Exception {
        centroids = new HashMap<>(numClusters);

        Random random = new Random(this.getSeed());
        Map<DecisionTableHashKey, Integer> initialClusters = new HashMap<>();

        int clusterIdx = 0;
        for (int i = data.numInstances() - 1; i >= 0; --i) {
            int bagIdx = random.nextInt(i + 1);
            DecisionTableHashKey hk = new DecisionTableHashKey(data.get(bagIdx), data.numAttributes(), true);
            if (!initialClusters.containsKey(hk)) {
                centroids.put(clusterIdx, data.get(bagIdx));
                clusterIdx++;
                initialClusters.put(hk, null);
            }
            data.swap(i, bagIdx);
            if (centroids.size() == numClusters) {
                break;
            }
        }
        startingPoints = new Instances(data.get(0).relationalValue(1), numClusters);
        for (int i = 0; i < numClusters; ++i)
            startingPoints.add(centroids.get(i));
    }

    @Override
    protected int computeCentroids(Instances[] clusters) {
        int emptyClusterCount = 0;
        centroids = new HashMap<>(numClusters);
        for (int i = 0; i < currentNClusters; ++i){
            if (clusters[i].numInstances() == 0)
                emptyClusterCount++;
            else
                centroids.put(i, computeCentroid(clusters[i]));
        }
        return emptyClusterCount;
    }

    private Instance computeCentroid(Instances cluster) {
        Instances groupInstances = new Instances (cluster.get(0).relationalValue(1), cluster.numInstances());
        for (Instance i : cluster)
            groupInstances.addAll(i.relationalValue(1));
        Instance groupBag = ProcessDataset.copyBag(cluster.get(0));
        groupBag.relationalValue(1).delete();
        groupBag.relationalValue(1).addAll(groupInstances);

        ExecutorService executor;
        if (parallelize)
            executor = Executors.newFixedThreadPool((int) (Runtime.getRuntime().availableProcessors() * 0.25));
        else
            executor = Executors.newFixedThreadPool(1);
        Collection<Callable<Map<Integer, Double>>> collection = new ArrayList<>(cluster.numInstances());
        for (int i = 0; i < cluster.numInstances(); ++i)
            collection.add(new ParallelizeComputeCentroid(i, cluster.get(i), groupBag));

        int minimal = -1;
        try {
            Map<Integer, Double> distances = new HashMap<>(cluster.numInstances());
            List<Future<Map<Integer, Double>>> futures = executor.invokeAll(collection);
            for (Future<Map<Integer, Double>> future : futures) {
                Map<Integer, Double> result = future.get();
                for (Map.Entry<Integer, Double> r : result.entrySet())
                    distances.put(r.getKey(), r.getValue());
            }
            minimal = Collections.min(distances.entrySet(), Map.Entry.comparingByValue()).getKey();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        return cluster.get(minimal);
    }

    private class ParallelizeComputeCentroid implements Callable<Map<Integer, Double>> {
        Integer idx;
        Instance instance;
        Instance groupBag;

        public ParallelizeComputeCentroid(Integer idx, Instance instance, Instance groupBag) {
            this.idx = idx;
            this.instance = instance;
            this.groupBag = groupBag;
        }

        @Override
        public Map<Integer, Double> call() throws Exception {
            double distance = distFunction.distance(instance, groupBag);
            Map<Integer, Double> result = new HashMap<>(1);
            result.put(idx, distance);
            return result;
        }
    }

    /*@Override
    protected Instance computeCentroid(Instances members) {
        //Instance centroid = super.computeCentroid(members);
        Instances groupInstances = new Instances (members.get(0).relationalValue(1), members.numInstances());
        for (Instance i : members)
            groupInstances.addAll(i.relationalValue(1));
        Instance groupBag = ProcessDataset.copyBag(members.get(0));
        groupBag.relationalValue(1).delete();
        groupBag.relationalValue(1).addAll(groupInstances);

        int idxMin = 0;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < members.numInstances(); ++i) {
            double distance = distFunction.distance(members.get(i), groupBag);
            if (distance < minDistance) {
                minDistance = distance;
                idxMin = i;
            }
        }

        return members.get(idxMin);
    }*/
}
