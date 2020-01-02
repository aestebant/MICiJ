package miclustering;

import miclustering.algorithms.MIClusterer;
import miclustering.evaluators.ClusterEvaluation;
import miclustering.utils.LoadByName;
import weka.clusterers.Clusterer;
import weka.core.Utils;

import java.util.HashMap;
import java.util.Map;

public class Run {
    public static void main(String[] args) {

        String[] datasets = {
//                "animals_relational"
//                "component_relational",
                "eastwest_relational",
//                "elephant_relational",
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
//                "westeast_relational"
        };

        String[] standardization = {
                "",
//                "-z4",
//                "-z5"
        };

        String[] clustering = {
//                "MIDBSCAN",
//                "MISimpleKMeans",
                "BAMIC",
        };

        Map<String, String> options = new HashMap<>();
        options.put("MIDBSCAN", " -E 2.5 -M 3 -A HausdorffDistance -hausdorff-type 0");
        options.put("MISimpleKMeans", "-N 2 -A HausdorffDistance -hausdorff-type 0");
        options.put("BAMIC", "-A HausdorffDistance -hausdorff-type 0");
        options.put("MIOPTICS", "");

        for (String d : datasets) {
            for (String z : standardization) {
                for (String c : clustering) {

                    System.out.println("=========================================");
                    System.out.println("Algorithm: " + c);
                    System.out.println("Dataset: " + d);
                    System.out.println("Standarization: " + z);
                    System.out.println("=========================================");

                    Clusterer clusterer = LoadByName.clusterer("miclustering.algorithms." + c);

                    try {
                        ((MIClusterer) clusterer).setOptions(Utils.splitOptions(options.get(c)));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    ClusterEvaluation evaluation = new ClusterEvaluation();
                    String pathDataset = "datasets/" + d + z + ".arff";
                    String evalOptions = "-d " + pathDataset + " -c last -k 2 -parallelize -p -A HausdorffDistance -hausdorff-type 0";
                    try {
                        evaluation.setOptions(Utils.splitOptions(evalOptions));
                        evaluation.evaluateClusterer(clusterer, true);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    System.out.println(clusterer.toString());
                    System.out.println(evaluation.printFullEvaluation());
                }
            }

        }
    }
}
