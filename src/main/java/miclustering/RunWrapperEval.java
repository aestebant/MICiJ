package miclustering;

import miclustering.evaluators.WrapperEvaluation;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;

import java.util.Map;
import java.util.Vector;

public class RunWrapperEval {
    public static void main(String[] args) {
        String dataset = "datasets/animals_relational-z1.arff";
        String distance = "HausdorffDistance";
        String distanceConfig = "-hausdorff-type 0";

        Instances data = ProcessDataset.readArff(dataset);
        int[] resFromJCLEC = new int[data.numInstances()];
        for (int i = 0; i < data.numInstances(); ++i) {
            resFromJCLEC[i] = (int) data.get(i).value(2);
        }
        int k = data.numDistinctValues(2);

        Vector<Integer> clusterAssignmet = new Vector<>(resFromJCLEC.length);
        for (int value : resFromJCLEC) clusterAssignmet.addElement(value);

        WrapperEvaluation evaluation = new WrapperEvaluation(dataset, distance, distanceConfig, k);

        double dbcv = evaluation.getDBCV(clusterAssignmet);
        double silhouette = evaluation.getSilhouette(clusterAssignmet);
        double sdbw = evaluation.getSdbw(clusterAssignmet);
        Map<String, String> ev = evaluation.getExternalEvaluation(clusterAssignmet);

        String result = String.join(" , ", String.valueOf(dbcv), String.valueOf(silhouette), String.valueOf(sdbw), ev.get("purity"),
                ev.get("rand"), ev.get("macro-precision"), ev.get("macro-recall"), ev.get("macro-f1"), ev.get("confmat"));
        System.out.println(result);
    }
}
