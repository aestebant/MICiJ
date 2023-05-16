package miclustering.algorithms.evolutionary;

import miclustering.algorithms.OneStepKMeans;

public class FastKMeansOperator extends KMeansOperator {
    public FastKMeansOperator(String dataset, String distFunc, String confDist, int numClusters) {
        super(dataset, distFunc, confDist, numClusters);
        oskm = new OneStepKMeans(dataset, distFunc, confDist, numClusters, false);
    }
}
