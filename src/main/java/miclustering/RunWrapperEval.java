package miclustering;

import miclustering.evaluators.WrapperEvaluation;

import java.util.Map;
import java.util.Vector;

public class RunWrapperEval {
    public static void main(String[] args) {
        String dataset = "datasets/animals_relational.arff";
        int k = 3;
        String distance = "HausdorffDistance";
        String distanceConfig = "-hausdorff-type 3";
        WrapperEvaluation evaluation = new WrapperEvaluation(dataset, distance, distanceConfig, k);

        int[] resFromJCLEC = {0, 2, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 0, 1, 0, 2, 2, 0, 1, 2, 1, 2, 1, 1, 2, 0, 1, 2, 0, 2, 1, 2, 2, 2, 0, 2, 2, 1, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 0, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 2, 2, 0, 2, 0, 1, 1, 0, 0, 1, 1, 0, 2, 1, 0, 0, 1, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 0, 1, 0, 2, 0, 1, 2, 0, 1, 1, 2, 0, 2, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 1, 0, 0, 1, 2, 0, 2, 2, 0, 0, 0, 2, 1, 2, 1, 2, 0, 1, 2, 2, 2, 1, 0, 2, 1, 2, 1, 2, 2, 2, 1, 0, 1, 1, 2, 1, 0, 2, 2, 0, 1, 0, 2, 1, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, 2, 0, 1, 2, 0, 1, 2, 1, 1, 1, 1, 0, 0, 2, 0, 1, 0, 2, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1, 2, 0, 1, 1, 0, 1, 2, 1, 2, 0, 2, 1, 1, 1, 1, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1};
        Vector<Integer> clusterAssignmet = new Vector<>(resFromJCLEC.length);
        for (int value : resFromJCLEC) clusterAssignmet.addElement(value);

        double silhouette = evaluation.getSilhouette(clusterAssignmet);
        double sdbw = evaluation.getSdbw(clusterAssignmet);
        Map<String, String> ev = evaluation.getExternalEvaluation(clusterAssignmet);

        String result = String.join(" , ", String.valueOf(silhouette), String.valueOf(sdbw), ev.get("purity"),
                ev.get("rand"), ev.get("macro-precision"), ev.get("macro-recall"), ev.get("macro-f1"), ev.get("confmat"));
        System.out.println(result);
    }
}
