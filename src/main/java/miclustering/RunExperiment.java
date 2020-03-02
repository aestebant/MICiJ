package miclustering;

import miclustering.algorithms.MIClusterer;
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
        for (Map.Entry<String, List<String>> e : clusterConfig.entrySet()) {
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

                        MIClusterer clusterer = (MIClusterer) LoadByName.clusterer("miclustering.algorithms." + c);
                        try {
                            clusterer.setOptions(Utils.splitOptions(config));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        String pathDataset = "datasets/" + d + z + ".arff";
                        String evalOptions = "-d " + pathDataset + " -c last -k 2 -parallelize";
                        ClusterEvaluation eval = new ClusterEvaluation();
                        try {
                            eval.setOptions(Utils.splitOptions(evalOptions));
                            eval.evaluateClusterer(clusterer, true);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        String distance = clusterer.getDistanceFunction().toString();
                        int actualNClusters = eval.getActualNumClusters();
                        int clusteredBags = eval.getInstances().numInstances() - eval.getUnclusteredInstances();
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
                        double time = clusterer.getElapsedTime();
                        String reportTitle = saveFullReport(clusterer, eval);
                        String report = String.join(",", c, config, d + z, distance, String.valueOf(actualNClusters),
                                String.valueOf(clusteredBags), String.valueOf(unclusteredBags), String.valueOf(rmsstd), String.valueOf(silhouette),
                                String.valueOf(xb), String.valueOf(db), String.valueOf(sdbw), String.valueOf(dbcv),
                                String.valueOf(entropy), String.valueOf(purity), String.valueOf(rand), String.valueOf(precision),
                                String.valueOf(recall), String.valueOf(f1), String.valueOf(specificity), PrintConfusionMatrix.singleLine(cer),
                                String.valueOf(time), reportTitle);
                        try {
                            reportFileWriter.write(report + "\n");
                            reportFileWriter.flush();
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
                "BirdsBrownCreeper",
                "BirdsChestnut-backedChickadee",
                "BirdsHammondsFlycatcher",
                "CorelAfrican",
                "CorelAntique",
                "CorelBattleships",
                "Harddrive1",
                "ImageElephant",
                "ImageFox",
                "ImageTiger",
                "Messidor",
                "mutagenesis3_atoms",
                "mutagenesis3_bonds",
                "mutagenesis3_chains",
                "Newsgroups1",
                "Newsgroups2",
                "Newsgroups3",
                "suramin",
                "DirectionEastwest",
                "Thioredoxin",
                "UCSBBreastCancer",
                "Web1",
                "Web2",
                "Web3",
                "Graz02bikes",
                "Graz02car",
                "Graz02people",
                "standardMI_Maron",
                "BiocreativeComponent",
                "BiocreativeFunction",
                "BiocreativeProcess",
        };

        standardization = new String[]{
//                "",
//                "-z1",
                "-z5"
        };

        clustering = new String[]{
                "MIDBSCAN",
//                "MIKMeans",
//                "BAMIC",
        };

        List<String> kMeansConfig = new ArrayList<>();
        for (int k = 2; k <= 2; ++k) {
            for (String hausdorff : new ArrayList<>(Arrays.asList("0", "1", "2", "3"))) {
                kMeansConfig.add("-N " + k + " -V -parallelize -A HausdorffDistance -hausdorff-type " + hausdorff);
            }
        }
        List<String> dbscanConfig = new ArrayList<>();
        for (double eps : new double[]{0.2, 0.4, 0.6, 0.8, 0.9}) {
            for (String hausdorff : new ArrayList<>(Arrays.asList("0", "1", "2", "3"))) {
                dbscanConfig.add("-E " + eps + " -output-clusters -A HausdorffDistance -hausdorff-type " + hausdorff);
            }
        }

        clusterConfig = new HashMap<>();
        clusterConfig.put("MIKMeans", kMeansConfig);
        clusterConfig.put("BAMIC", kMeansConfig);
        clusterConfig.put("MIDBSCAN", dbscanConfig);
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
