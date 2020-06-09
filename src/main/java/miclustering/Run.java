package miclustering;

import miclustering.algorithms.MIClusterer;
import miclustering.evaluators.ClusterEvaluation;
import miclustering.utils.LoadByName;
import miclustering.utils.ProcessDataset;
import weka.core.Utils;

import java.util.HashMap;
import java.util.Map;

public class Run {
    public static void main(String[] args) {

        String[] datasets = {
                "musk1",
                "musk2",
//                "BirdsBrownCreeper",
//                "BirdsChestnut-backedChickadee",
//                "BirdsHammondsFlycatcher",
//                "CorelAfrican",
//                "CorelAntique",
//                "CorelBattleships",
//                "Harddrive1",
//                "ImageElephant",
//                "ImageFox",
//                "ImageTiger",
//                "Messidor",
//                "mutagenesis3_atoms",
//                "mutagenesis3_bonds",
//                "mutagenesis3_chains",
//                "Newsgroups1",
//                "Newsgroups2",
//                "Newsgroups3",
//                "suramin",
//                "DirectionEastwest",
//                "Thioredoxin",
//                "UCSBBreastCancer",
//                "Web1",
//                "Web2",
//                "Web3",
//                "Graz02bikes",
//                "Graz02car",
//                "Graz02people",
//                "standardMI_Maron",
//                "BiocreativeComponent",
//                "BiocreativeFunction",
//                "BiocreativeProcess",
        };

        String[] standardization = {
//                "",
//                "-z4",
                "-z5"
        };

        String[] clustering = {
//                "MIDBSCAN",
//                "MIOPTICS"
//                "MIKMeans",
                "BAMIC",
        };

        Map<String, String> options = new HashMap<>();
        options.put("MIDBSCAN", "-E 0.8 -A HausdorffDistance -hausdorff-type 0");
        options.put("MIOPTICS", "-E 10 -A HausdorffDistance -hausdorff-type 0");
        options.put("MIKMeans", "-N 2 -A HausdorffDistance -hausdorff-type 0");
        options.put("BAMIC", "-N 2 -A HausdorffDistance -hausdorff-type 0");

        for (String d : datasets) {
            for (String z : standardization) {
                for (String c : clustering) {
                    MIClusterer clusterer = (MIClusterer) LoadByName.clusterer("miclustering.algorithms." + c);
                    String pathDataset = "datasets/" + d + z + ".arff";
                    try {
                        clusterer.setOptions(Utils.splitOptions(options.get(c)));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    System.out.println("=========================================");
                    System.out.println("Algorithm: " + c);
                    System.out.println("Distance: " + clusterer.getDistanceFunction());
                    System.out.println("Dataset: " + d);
                    System.out.println("Standarization: " + z);
                    System.out.println("=========================================");

                    try {
                        clusterer.buildClusterer(ProcessDataset.readArff(pathDataset));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    ClusterEvaluation evaluation = new ClusterEvaluation();
                    String evalOptions = "-d " + pathDataset + " -c last -k 2 -parallelize";
                    try {
                        evaluation.setOptions(Utils.splitOptions(evalOptions));
                        evaluation.evaluateClusterer(clusterer, true);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    //System.out.println(clusterer.toString());
                    System.out.println(evaluation.printFullEvaluation());
                }
            }

        }
    }
}
