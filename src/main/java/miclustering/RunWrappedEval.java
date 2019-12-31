package miclustering;

import miclustering.evaluators.WrappedEvaluation;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RunWrappedEval {
    public static void main(String[] args) {
        String[] datasets = {
//                "component_relational",
//                "eastwest_relational",
                "elephant_relational",
//                "fox_relational",
//                "function_relational",
//                "musk1_relational",
//                "musk2_relational",
//                "mutagenesis3_atoms_relational",
//                "mutagenesis3_bonds_relational",
//                "mutagenesis3_chains_relational",
//                "process_relational",
//                "suramin_relational",
//                "tiger_relational",
//                "trx_relational",
//                "westeast_relational",
////                "animals_relational"
        };
        //String dataset = "animals_relational-z1.arff";
        String distance = "HausdorffDistance";
        String distanceConfig = "-hausdorff-type 0";

        for (String dataset : datasets) {
            Instances data = ProcessDataset.readArff("datasets/" + dataset + ".arff");
            int[] resFromJCLEC = new int[data.numInstances()];
            for (int i = 0; i < data.numInstances(); ++i) {
                resFromJCLEC[i] = (int) data.get(i).value(2);
                //resFromJCLEC[i] = 0;
            }
            int k = data.numDistinctValues(2);

            List<Integer> clusterAssignmet = new ArrayList<>(resFromJCLEC.length);
            for (int value : resFromJCLEC) clusterAssignmet.add(value);

            WrappedEvaluation evaluation = new WrappedEvaluation("datasets/" + dataset + ".arff", distance, distanceConfig, k);

            double rmssd = evaluation.getRMSSD(clusterAssignmet);
            double twcv = evaluation.getTWCV(clusterAssignmet);
            double ftwcv = evaluation.getFastTWCV(clusterAssignmet);
            double dbcv = evaluation.getDBCV(clusterAssignmet);
            double silhouette = evaluation.getSilhouette(clusterAssignmet);
            double xb = evaluation.getXB(clusterAssignmet);
            double db = evaluation.getDB(clusterAssignmet);
            double sdbw = evaluation.getSdbw(clusterAssignmet);
            Map<String, String> ev = evaluation.getExternalEvaluation(clusterAssignmet);

            String result = String.join(" , ", String.valueOf(rmssd), String.valueOf(twcv), String.valueOf(ftwcv),
                    String.valueOf(silhouette), String.valueOf(xb), String.valueOf(db),
                    String.valueOf(sdbw), String.valueOf(dbcv), ev.get("entropy"), ev.get("purity"),
                    ev.get("rand"), ev.get("macro-precision"), ev.get("macro-recall"), ev.get("macro-f1"), ev.get("macro-specificity"), ev.get("confmat"));

            System.out.println("RMSSD , TWCV , FTWCV , Silhouette , XB , DB , SDBW , DBCV , Entropy , Purity , Rand index , Precision , Recall , F1 , Specificity , Confusion Matrix");
            System.out.println(result);
        }
    }
}
