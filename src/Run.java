import algorithms.MyClusterer;
import evaluators.ClusterEvaluation;
import utils.ProcessDataset;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;

import java.lang.reflect.InvocationTargetException;
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
//                "MIDBSCAN",
//                "MISimpleKMeans",
                "BAMIC"
        };

        Map<String, String> options = new HashMap<>();
        options.put("MIDBSCAN", " -E 1.4 -M 8");
        options.put("MISimpleKMeans", "-num-slots 2 -V");
        options.put("BAMIC", "-num-slots 2 -V");

        for (String d : datasets) {
            for (String z : standardization) {
                for (String c : clustering) {

                    System.out.println("=========================================");
                    System.out.println("Algorithm: " + c);
                    System.out.println("Dataset: " + d);
                    System.out.println("Standarization: " + z);
                    System.out.println("=========================================");

                    Class<? extends AbstractClusterer> absCluster = null;
                    try {
                        absCluster = Class.forName("algorithms." + c).asSubclass(AbstractClusterer.class);
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    }
                    assert absCluster != null;
                    Clusterer clusterer = null;
                    try {
                        clusterer = absCluster.getDeclaredConstructor().newInstance();
                    } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
                        e.printStackTrace();
                    }
                    assert clusterer != null;

                    try {
                        ((MyClusterer) clusterer).setOptions(weka.core.Utils.splitOptions(options.get(c)));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    try {
                        clusterer.buildClusterer(ProcessDataset.readArff("datasets/" + d + z + ".arff"));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    System.out.println(clusterer.toString());

                    String evalOptions = "-t datasets/" + d + z + ".arff -c last";
                    try {
                        System.out.println(ClusterEvaluation.evaluateClusterer(clusterer, weka.core.Utils.splitOptions(evalOptions)));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }

        }
    }
}
