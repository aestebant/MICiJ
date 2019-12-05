package miclustering;

import miclustering.algorithms.MyClusterer;
import miclustering.evaluators.ClusterEvaluation;
import miclustering.utils.LoadByName;
import miclustering.utils.ProcessDataset;
import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.core.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class RunExperiment {

    private static String[] datasets;
    private static String[] standardization;
    private static String[] clustering;
    private static Map<String, List<String>> clusterConfig;

    private static int nThreads = 1;

    private static String reportTitle;
    private static File reportDirectory;
    private static FileWriter reportFileWriter;

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Wrong use. Format <reportTitle>");
            return;
        }
        reportTitle = args[0];
        nThreads = Runtime.getRuntime().availableProcessors();

        setExperiments();
        setSaveResults();

        String evaluationConfig = "-c last  -num-threads " + nThreads;

        int nConfigs = 0;
        for (Map.Entry<String, List<String>> e: clusterConfig.entrySet()) {
            nConfigs += e.getValue().size();
        }
        int totalIterations = datasets.length * standardization.length * nConfigs;
        int currentIteration = 0;

        for (String d : datasets) {
            for (String z : standardization) {
                for (String c : clustering) {
                    List<String> configs = clusterConfig.get(c);
                    for (String config : configs) {
                        currentIteration++;

                        String control = "Iteration " + currentIteration + " of " + totalIterations + ": " +
                                " clustering: " + c +
                                " | configuration: " + config +
                                " | dataset: " + d +
                                " | standardization: " + z;
                        System.out.println(control);

                        Clusterer clusterer = LoadByName.clusterer("miclustering.algorithms." + c);
                        Instances dataset = ProcessDataset.readArff("datasets/" + d + z + ".arff");
                        ClusterEvaluation eval = new ClusterEvaluation();
                        try {
                            eval.setOptions(Utils.splitOptions(evaluationConfig));
                            eval.setClusterer(clusterer, Utils.splitOptions(config));
                            eval.evaluateClusterer(dataset);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        try {
                            reportFileWriter.flush();
                            reportFileWriter.write(c + "," + config + "," + ((MyClusterer)clusterer).getDistanceFunction() + "," + d + "," + z + "," + eval.getActualNumClusters()
                                    + "," + dataset.numInstances() + "," + (dataset.numInstances()-eval.getUnclusteredInstances()) + "," + eval.getUnclusteredInstances() + "," + eval.getSilhouette() + ","
                                    + eval.getSdbw() + "," + eval.getPurity() + "," + eval.getRand() + "," + eval.getConfussion() + "," + ((MyClusterer)clusterer).getElapsedTime() + "\n"
                            );
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        saveFullReport(clusterer, eval, currentIteration);
                    }
                }
            }
        }
        System.out.println("Finish");
        try {
            reportFileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void setExperiments() {
        datasets = new String[]{
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
//                "westeast_relational",
                "animals_relational"
        };

        standardization = new String[]{
                "",
                "-z1",
                "-z5"
        };

        clustering = new String[]{
//                "MIDBSCAN",
//                "MISimpleKMeans",
                "BAMIC",
        };

        List<String> kMeansConfig = new ArrayList<>();
        for (int k = 2; k <= 4; ++k) {
            for (String hausdorff : new ArrayList<>(Arrays.asList("0", "1", "2", "3"))) {
                kMeansConfig.add("-N " + k + " -num-slots " + nThreads + " -V -hausdorff-type " + hausdorff);
            }
        }
        List<String> dbscanConfig = new ArrayList<>();
        for (double eps : new double[]{0.6, 0.9, 1.2, 1.5}) {
            for (int minPts = 2; minPts <= 4; ++minPts) {
                for (String hausdorff : new ArrayList<>(Arrays.asList("minimal", "maximal", "average"))) {
                    dbscanConfig.add("-E " + eps + " -M " + minPts + " -output-clusters -num-threads " + nThreads + " -hausdorff-type " + hausdorff);
                }
            }
        }

        clusterConfig = new HashMap<>();
        clusterConfig.put("MISimpleKMeans", kMeansConfig);
        clusterConfig.put("BAMIC", kMeansConfig);
        clusterConfig.put("MIDBSCAN", dbscanConfig);
    }

    private static void setSaveResults() {
        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");
        Date date = new Date(System.currentTimeMillis());
        String dateString = dateFormat.format(date);
        String actualReportTitle = reportTitle + "_" + dateString;
        reportDirectory = new File(actualReportTitle);
        if (!reportDirectory.mkdir()) {
            throw new RuntimeException("Error creating report directory");
        }
        File reportFile = new File(actualReportTitle + ".report.csv");
        System.out.println(reportFile.getAbsoluteFile().getParent());
        File parentDirectory = new File(reportFile.getAbsoluteFile().getParent());
        if (!parentDirectory.exists() && !parentDirectory.mkdirs()) {
            throw new RuntimeException("Error creating report directory");
        }
        try {
            reportFileWriter = new FileWriter(reportFile);
            reportFileWriter.flush();
            reportFileWriter.write("Algorithm,Configuration,Distance Function,Dataset,Standardization,#Clusters,#Bags,#Clusterd bags,#Unclustered bags,Silhouette index,S_Dbw index,Purity,Rand index,#Confusion Matrix (a cluster by each |),#Time of Clustering\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void saveFullReport(Clusterer clusterer, ClusterEvaluation evaluation, int currentIteration) {
        String filename = "experiment_" + currentIteration + ".txt";
        File file = new File(reportDirectory, filename);
        FileWriter filewriter;
        try {
            filewriter = new FileWriter(file);
            filewriter.flush();
            filewriter.write(clusterer.toString());
            filewriter.write(evaluation.toString());
            filewriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
