package algorithms;

public interface MyClusterer extends weka.clusterers.Clusterer {
    void setOptions(String[] options) throws Exception;
}
