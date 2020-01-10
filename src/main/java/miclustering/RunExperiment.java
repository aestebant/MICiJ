package miclustering;

import miclustering.algorithms.MIClusterer;
import miclustering.distances.HausdorffDistance;
import miclustering.evaluators.ClusterEvaluation;
import miclustering.evaluators.ExtEvalResult;
import miclustering.utils.LoadByName;
import miclustering.utils.PrintConfusionMatrix;
import weka.clusterers.Clusterer;
import weka.core.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class RunExperiment {

    private static String[] dataset;
    private static String[] standardization;
    private static String[] clustering;
    private static Map<String, List<String>> clusterConfig;

    private static String reportTitle;
    private static FileWriter reportFileWriter;

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Wrong use. Format <reportTitle>");
            return;
        }
        reportTitle = args[0];

        setExperiments();
        setSaveResults();

        int nConfigs = 0;
        for (Map.Entry<String, List<String>> e: clusterConfig.entrySet()) {
            nConfigs += e.getValue().size();
        }
        int totalIterations = dataset.length * standardization.length * nConfigs;
        int currentIteration = 0;

        for (String c : clustering) {
            for (String d : dataset) {
                for (String z : standardization) {
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
                        try {
                            ((MIClusterer) clusterer).setOptions(Utils.splitOptions(config));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        //Instances dataset = ProcessDataset.readArff("datasets/" + d + z + ".arff");
                        String pathDataset = "datasets/" + d + z + ".arff";
                        int type = ((HausdorffDistance) ((MIClusterer) clusterer).getDistanceFunction()).getType();
                        String evalOptions = "-d " + pathDataset + " -c last -k 2 -parallelize -A HausdorffDistance -hausdorff-type " + type;
                        System.out.println(evalOptions);
                        ClusterEvaluation eval = new ClusterEvaluation();
                        try {
                            eval.setOptions(Utils.splitOptions(evalOptions));
                            eval.evaluateClusterer(clusterer, true);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        String distance = ((MIClusterer)clusterer).getDistanceFunction().toString();
                        int actualNClusters = eval.getActualNumClusters();
                        int clusteredBags = eval.getInstances().numInstances()-eval.getUnclusteredInstances();
                        int unclusteredBags = eval.getUnclusteredInstances();
                        double rmsstd = eval.getRmssd();
                        double silhouette = eval.getSilhouette();
                        double xb = eval.getXb();
                        double db = eval.getDb();
                        double sdbw = eval.getSdbw();
                        double dbcv = eval.getDbcv();
                        ExtEvalResult cer = eval.getExtEvalResult();
                        double entropy = eval.getEntropy();
                        double purity = eval.getPurity();
                        double rand = eval.getRand();
                        double precision = eval.getMacroPrecision();
                        double recall = eval.getMacroRecall();
                        double f1 = eval.getMacroF1();
                        double specificity = eval.getMacroSpecificity();
                        double time = ((MIClusterer)clusterer).getElapsedTime();
                        String reportTitle = saveFullReport(clusterer, eval);
                        String report = String.join(",", c, config, d + z, distance, String.valueOf(actualNClusters),
                                String.valueOf(clusteredBags), String.valueOf(unclusteredBags), String.valueOf(rmsstd), String.valueOf(silhouette),
                                String.valueOf(xb), String.valueOf(db), String.valueOf(sdbw), String.valueOf(dbcv),
                                String.valueOf(entropy), String.valueOf(purity), String.valueOf(rand), String.valueOf(precision),
                                String.valueOf(recall), String.valueOf(f1), String.valueOf(specificity), PrintConfusionMatrix.singleLine(cer.getConfMatrix()),
                                String.valueOf(time), reportTitle);
                        try {
                            reportFileWriter.flush();
                            reportFileWriter.write(report + "\n");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

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
        dataset = new String[]{
//                "component_relational",
                "elephant_relational",
                "fox_relational",
                "tiger_relational",
                "mutagenesis3_atoms_relational",
                "mutagenesis3_bonds_relational",
                "mutagenesis3_chains_relational",
//                "function_relational",
                "musk1_relational",
                "musk2_relational",
//                "process_relational",
//                "suramin_relational",
//                "trx_relational",
                "eastwest_relational",
                "westeast_relational",
//                "animals_relational"
        };

        standardization = new String[]{
                "",
                "-z1",
                "-z5"
        };

        clustering = new String[]{
//                "MIDBSCAN",
                "MISimpleKMeans",
                "BAMIC",
        };

        List<String> kMeansConfig = new ArrayList<>();
        for (int k = 2; k <= 2; ++k) {
            for (String hausdorff : new ArrayList<>(Arrays.asList("0", "1", "2", "3"))) {
                kMeansConfig.add("-N " + k + " -V -A HausdorffDistance -hausdorff-type " + hausdorff);
            }
        }
        List<String> dbscanConfig = new ArrayList<>();
        for (double eps : new double[]{0.6, 0.9, 1.2, 1.5}) {
            for (int minPts = 2; minPts <= 4; ++minPts) {
                for (String hausdorff : new ArrayList<>(Arrays.asList("minimal", "maximal", "average"))) {
                    dbscanConfig.add("-E " + eps + " -M " + minPts + " -output-clusters -hausdorff-type " + hausdorff);
                }
            }
        }

        clusterConfig = new HashMap<>();
        clusterConfig.put("MISimpleKMeans", kMeansConfig);
        clusterConfig.put("BAMIC", kMeansConfig);
//        clusterConfig.put("MIDBSCAN", dbscanConfig);
    }

    private static void setSaveResults() {
        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");
        Date date = new Date(System.currentTimeMillis());
        String dateString = dateFormat.format(date);
        String actualReportTitle = reportTitle + "_" + dateString;

        File reportFile = new File(actualReportTitle + ".final.csv");
        File reportDirectory = new File(reportFile.getParent());
        if (!reportDirectory.exists() && !reportDirectory.mkdirs()) {
            throw new RuntimeException("Error creating report directory");
        }
        try {
            reportFileWriter = new FileWriter(reportFile);
            reportFileWriter.flush();
            reportFileWriter.write(dateString + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            reportFileWriter = new FileWriter(reportFile);
            reportFileWriter.flush();
            reportFileWriter.write("Algorithm,Configuration,Dataset,Distance Function,Clusters,Clusterd bags,Unclustered bags,RMSSD,Silhouete,XB,DB,S_Dbw,DBCV,Entropy,Purity,Rand index,Precision,Recall,F1,Specificity,Conf Matrix,Time,Report\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String saveFullReport(Clusterer clusterer, ClusterEvaluation evaluation) {
        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");
        Date date = new Date(System.currentTimeMillis());
        String dateString = dateFormat.format(date);
        String actualReportTitle = reportTitle + "_" + dateString + ".report.txt";
        File file = new File(actualReportTitle);
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
        return actualReportTitle;
    }
}
