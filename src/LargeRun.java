import evaluators.ClusterEvaluation;
import utils.LoadByName;
import utils.ProcessDataset;
import weka.clusterers.Clusterer;
import weka.core.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class LargeRun {

    private static String[] datasets;
    private static String[] standardization;
    private static String[] clustering;
    private static Map<String, List<String>> clusterConfig;
    private static String evaluationConfig = "-c last";

    private static int nThreads = 1;

    private static String reportTitle;
    private static String actualReportTitle;
    private static File reportFile;
    private static File reportDirectory;
    private static FileWriter reportFileWriter;

    private static void setExperiments() {
        datasets = new String[]{
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
                "westeast_relational"
        };

        standardization = new String[]{
                "",
                "-z4",
                "-z5"
        };

        clustering = new String[]{
                "MIDBSCAN",
                "MISimpleKMeans",
                "BAMIC",
        };

        List<String> kMeansConfig = new ArrayList<>();
        for (int k = 2; k <= 6; ++k) {
            for (String hausodorff : new ArrayList<>(Arrays.asList("minimal", "maximal", "average"))) {
                kMeansConfig.add("-N " + k + " -num-slots " + nThreads + " -V -hausdorff-type " + hausodorff);
            }
        }
        List<String> dbscanConfig = new ArrayList<>();
        for (double eps = 0.6; eps <= 1.6; eps += 0.2) {
            for (int minPts = 2; minPts <= 6; ++minPts) {
                for (String hausodorff : new ArrayList<>(Arrays.asList("minimal", "maximal", "average"))) {
                    dbscanConfig.add("-E " + eps + " -M " + minPts + " -output-clusters -hausdorff-type " + hausodorff);
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
        actualReportTitle = reportTitle + "_" + dateString;
        reportDirectory = new File(actualReportTitle);
        reportFile = new File(actualReportTitle + ".report.csv");
        File reportDirectory = new File(reportFile.getParent());
        if (!reportDirectory.exists() && !reportDirectory.mkdirs()) {
            throw new RuntimeException("Error creating report directory");
        }
        try {
            reportFileWriter = new FileWriter(reportFile);
            reportFileWriter.flush();
            reportFileWriter.write("Algorithm,Configuration,Dataset,Standardization,#Clusters,#Unclustered,Silhouette index, S_Dbw index, Purity, Rand index\n");
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

    public static void main(String[] args) {
        setExperiments();
        setSaveResults();

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
                                " configuration: " + config +
                                " dataset: " + d +
                                " standardization: " + z;
                        System.out.println(control);

                        Clusterer clusterer = LoadByName.clusterer("algorithms." + c);

                        ClusterEvaluation evaluation = new ClusterEvaluation();
                        try {
                            evaluation.setOptions(Utils.splitOptions(evaluationConfig));
                            evaluation.setClusterer(clusterer, Utils.splitOptions(config));
                            evaluation.evaluateClusterer(ProcessDataset.readArff("datasets/" + d + z + ".arff"));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        try {
                            reportFileWriter.flush();
                            reportFileWriter.write(c + "," + config + "," + d + "," + z + "," + evaluation.getNumClusters()
                                    + "," + evaluation.getUnclusteredInstances() + "," + evaluation.getSilhouette() + ","
                                    + evaluation.getSdbw() + "," + evaluation.getPurity() + "," + evaluation.getRand() + "\n"
                            );
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        saveFullReport(clusterer, evaluation, currentIteration);
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
}
