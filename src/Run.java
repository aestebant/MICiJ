import algorithms.MISimpleKMeans;
import evaluators.ClusterEvaluation;

public class Run {
    public static void main(String[] args) {

        MISimpleKMeans clusterer = new MISimpleKMeans();
        //MIDBSCAN clusterer = new MIDBSCAN();
        //MIOPTICS clusterer = new MIOPTICS();
        // E: epsion, M: min points
        /*String clusterOptions = " -E 1.4 -M 8";
        try {
            clusterer.setOptions(weka.core.Utils.splitOptions(clusterOptions));
        } catch (Exception e) {
            e.printStackTrace();
        }*/

        String evalOptions = "-t /home/aurora/Escritorio/prueba.arff"
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
