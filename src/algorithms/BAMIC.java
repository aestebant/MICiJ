package algorithms;

import weka.classifiers.rules.DecisionTableHashKey;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

public class BAMIC extends MISimpleKMeans {

    @Override
    protected void randomInit(Instances data) throws Exception {
        this.centroids = new Instances(data, this.numClusters);

        Random random = new Random(this.getSeed());
        Map<DecisionTableHashKey, Integer> initialClusters = new HashMap<>();

        for (int i = data.numInstances() - 1; i >= 0; --i) {
            int bagIdx = random.nextInt(i + 1);
            DecisionTableHashKey hk = new DecisionTableHashKey(data.get(bagIdx), data.numAttributes(), true);
            if (!initialClusters.containsKey(hk)) {
                this.centroids.add(data.get(bagIdx));
                initialClusters.put(hk, null);
            }
            data.swap(i, bagIdx);
            if (this.centroids.numInstances() == this.numClusters) {
                break;
            }
        }
        this.startingPoints = new Instances(this.centroids);
    }

    @Override
    protected Instance computeCentroid(Instances members) {
        Instance centroid = super.computeCentroid(members);

        int idxMin = 0;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < members.numInstances(); ++i) {
            double distance = distFunction.distance(members.get(i), centroid);
            if (distance < minDistance) {
                minDistance = distance;
                idxMin = i;
            }
        }

        return members.get(idxMin);
    }
}
