import algorithms.MyClusterer;
import evaluators.ClusterEvaluation;
import utils.LoadByName;
import utils.ProcessDataset;
import weka.clusterers.Clusterer;
import weka.core.Utils;

import java.util.HashMap;
import java.util.Map;

public class Run {
    public static void main(String[] args) {

        String[] datasets = {
//                "component_relational",
//                "eastwest_relational",
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
                "westeast_relational"
        };

        String[] standardization = {
//                "",
//                "-z4",
                "-z5"
        };

        String[] clustering = {
                "MIDBSCAN",
                "MISimpleKMeans",
                "BAMIC",
        };

        Map<String, String> options = new HashMap<>();
        options.put("MIDBSCAN", " -E 0.5 -M 2 -output-clusters -hausdorff-type average");
        options.put("MISimpleKMeans", "-S 155 -num-slots 3 -V -hausdorff-type average");
        options.put("BAMIC", "-S 155 -num-slots 3 -V");
        options.put("MIOPTICS", "");

        for (String d : datasets) {
            for (String z : standardization) {
                for (String c : clustering) {

                    System.out.println("=========================================");
                    System.out.println("Algorithm: " + c);
                    System.out.println("Dataset: " + d);
                    System.out.println("Standarization: " + z);
                    System.out.println("=========================================");

                    Clusterer clusterer = LoadByName.clusterer("algorithms." + c);

                    try {
                        ((MyClusterer) clusterer).setOptions(Utils.splitOptions(options.get(c)));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    /*try {
                        clusterer.buildClusterer(ProcessDataset.readArff("datasets/" + d + z + ".arff"));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }*/


                    ClusterEvaluation evaluation = new ClusterEvaluation();
                    String evalOptions = " -c last";
                    try {
                        evaluation.setOptions(Utils.splitOptions(evalOptions));
                        evaluation.setClusterer(clusterer, Utils.splitOptions(options.get(c)));
                        evaluation.evaluateClusterer(ProcessDataset.readArff("datasets/" + d + z + ".arff"));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    System.out.println(clusterer.toString());
                    System.out.println(evaluation);
                }
            }

        }
    }
}
