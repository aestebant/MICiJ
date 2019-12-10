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
        Map<String, String> externalEval = evaluation.getExternalEvaluation(clusterAssignmet);

        String sb = "1000,1000,0.9,0.5,animals_relational,," + evaluation.getDistanceFunction() + "," + k + "," +
                silhouette + "," + sdbw + "," + externalEval.get("purity") + "," + externalEval.get("rand") + "," +
                externalEval.get("confmat") + "\n";
        System.out.println(sb);
    }
}
