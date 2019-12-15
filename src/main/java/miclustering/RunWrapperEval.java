package miclustering;

import miclustering.evaluators.WrapperEvaluation;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;

import java.util.Map;
import java.util.Vector;

public class RunWrapperEval {
    public static void main(String[] args) {
        String[] datasets = {
                "component_relational",
                "eastwest_relational",
                "elephant_relational",
                "fox_relational",
                "function_relational",
                "musk1_relational",
                "musk2_relational",
                "mutagenesis3_atoms_relational",
                "mutagenesis3_bonds_relational",
                "mutagenesis3_chains_relational",
                "process_relational",
                "suramin_relational",
                "tiger_relational",
                "trx_relational",
                "westeast_relational",
                "animals_relational"
        };
        //String dataset = "animals_relational-z1.arff";
        String distance = "HausdorffDistance";
        String distanceConfig = "-hausdorff-type 0";

        for (String dataset : datasets) {
            Instances data = ProcessDataset.readArff("datasets/" + dataset + ".arff");
            int[] resFromJCLEC = new int[data.numInstances()];
            for (int i = 0; i < data.numInstances(); ++i) {
                resFromJCLEC[i] = (int) data.get(i).value(2);
            }
            int k = data.numDistinctValues(2);

            Vector<Integer> clusterAssignmet = new Vector<>(resFromJCLEC.length);
            for (int value : resFromJCLEC) clusterAssignmet.addElement(value);

            WrapperEvaluation evaluation = new WrapperEvaluation("datasets/" + dataset + ".arff", distance, distanceConfig, k);

            double dbcv = evaluation.getDBCV(clusterAssignmet);
            double silhouette = evaluation.getSilhouette(clusterAssignmet);
            double sdbw = evaluation.getSdbw(clusterAssignmet);
            Map<String, String> ev = evaluation.getExternalEvaluation(clusterAssignmet);

            String result = String.join(" , ", String.valueOf(dbcv), String.valueOf(silhouette), String.valueOf(sdbw), ev.get("purity"),
                    ev.get("rand"), ev.get("macro-precision"), ev.get("macro-recall"), ev.get("macro-f1"), ev.get("confmat"));
            System.out.println(result);
        }
    }
}
