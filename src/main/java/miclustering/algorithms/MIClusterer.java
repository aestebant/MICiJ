package miclustering.algorithms;

import weka.core.DistanceFunction;

import java.util.List;

public interface MIClusterer extends weka.clusterers.Clusterer {
    void setOptions(String[] options) throws Exception;

    DistanceFunction getDistanceFunction();

    double getElapsedTime();

    List<Integer> getClusterAssignments();
}
