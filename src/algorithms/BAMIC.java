package algorithms;

import weka.classifiers.rules.DecisionTableHashKey;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class BAMIC extends MISimpleKMeans {

    @Override
    protected void randomInit(Instances data) throws Exception {
        this.m_ClusterCentroids = new Instances(data, this.m_NumClusters);

        Random random = new Random(this.getSeed());
        Map<DecisionTableHashKey, Integer> initialClusters = new HashMap<>();

        for (int i = data.numInstances() - 1; i >= 0; --i) {
            int bagIdx = random.nextInt(i + 1);
            DecisionTableHashKey hk = new DecisionTableHashKey(data.get(bagIdx), data.numAttributes(), true);
            if (!initialClusters.containsKey(hk)) {
                this.m_ClusterCentroids.add(data.get(bagIdx));
                initialClusters.put(hk, null);
            }
            data.swap(i, bagIdx);
            if (this.m_ClusterCentroids.numInstances() == this.m_NumClusters) {
                break;
            }
        }
        this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
    }

    @Override
    protected Instance moveCentroid(int clusterIdx, Instances members, Boolean addToCentroidInstances) {
        int numInstAttributes = members.get(0).relationalValue(1).numAttributes();

        Instances aux = new Instances(members.get(0).relationalValue(1));
        for (Instance member : members) {
            aux.addAll(member.relationalValue(1));
        }

        double[] means = new double[numInstAttributes];
        for (int i = 0; i < numInstAttributes; ++i) {
            means[i] = aux.meanOrMode(i);
        }

        Instance centroid = new DenseInstance(1D, means);
        aux.add(centroid);

        int idxMin = 0;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < members.numInstances(); ++i) {
            double distance = m_DistanceFunction.distance(members.get(i), aux.lastInstance());
            if (distance < minDistance) {
                minDistance = distance;
                idxMin = i;
            }
        }

        if (addToCentroidInstances) {
            this.m_ClusterCentroids.add(members.get(idxMin));
        }

        return members.get(idxMin);
    }
}
