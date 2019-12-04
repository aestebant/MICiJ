package miclustering.algorithms;

import weka.core.DistanceFunction;

public interface MyClusterer extends weka.clusterers.Clusterer {
    void setOptions(String[] options) throws Exception;
    DistanceFunction getDistanceFunction();
    double getElapsedTime();
}
