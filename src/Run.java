import algorithms.MISimpleKMeans;
import evaluators.ClusterEvaluation;
import utils.ProcessDataset;
import weka.clusterers.Clusterer;

public class Run {
    public static void main(String[] args) {

        Clusterer clusterer = new MISimpleKMeans();
        try {
            ((MISimpleKMeans) clusterer).setNumClusters(2);
        } catch (Exception e) {
            e.printStackTrace();
        }
        //MIDBSCAN clusterer = new MIDBSCAN();
        //MIOPTICS clusterer = new MIOPTICS();
        // E: epsion, M: min points
        /*String clusterOptions = " -E 1.4 -M 8";
        try {
            clusterer.setOptions(weka.core.Utils.splitOptions(clusterOptions));
        } catch (Exception e) {
            e.printStackTrace();
        }*/

        try {
            clusterer.buildClusterer(ProcessDataset.readArff("datasets/eastwest_relational-z4.arff"));
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(clusterer.toString());

        String evalOptions = "-t datasets/component_relational.arff"
                + " -c last";
        try {
            System.out.println(ClusterEvaluation.evaluateClusterer(clusterer, weka.core.Utils.splitOptions(evalOptions)));
        } catch (Exception e) {
            e.printStackTrace();
        }
        /*ClusterEvaluation evaluation = new ClusterEvaluation();

        evaluation.setClusterer(clusterer);
        try {
            evaluation.evaluateClusterer(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(evaluation.clusterResultsToString());*/
    }
}
