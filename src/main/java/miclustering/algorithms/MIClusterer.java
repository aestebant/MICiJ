package miclustering.algorithms;

import weka.core.DistanceFunction;

public interface MIClusterer extends weka.clusterers.Clusterer {
    void setOptions(String[] options) throws Exception;

    DistanceFunction getDistanceFunction();

    double getElapsedTime();
}
