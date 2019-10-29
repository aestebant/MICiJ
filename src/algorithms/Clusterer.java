package algorithms;

public interface Clusterer extends weka.clusterers.Clusterer {
    void setOptions(String[] options) throws Exception;
}
