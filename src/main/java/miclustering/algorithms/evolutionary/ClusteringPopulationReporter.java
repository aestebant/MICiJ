package miclustering.algorithms.evolutionary;

import miclustering.utils.PrintConfusionMatrix;
import jclec.AlgorithmEvent;
import jclec.IFitness;
import jclec.IIndividual;
import jclec.algorithm.PopulationAlgorithm;
import jclec.algorithm.classic.CHC;
import jclec.listener.PopulationReporter;
import jclec.util.IndividualStatistics;
import jclec.util.random.RanecuFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Comparator;
import java.util.List;

public class ClusteringPopulationReporter extends PopulationReporter {
    @Override
    protected void doIterationReport(PopulationAlgorithm algorithm, boolean force) {
        // Fitness comparator
        Comparator<IFitness> comparator = algorithm.getEvaluator().getComparator();
        // Population individuals
        List<IIndividual> inhabitants = algorithm.getInhabitants();
        // Actual generation
        int generation = algorithm.getGeneration();

        // Check if this is correct generation
        if (!force && generation % reportFrequency != 0) {
            return;
        }

        // Save population individuals (if this option was chosen)
        if (saveCompletePopulation) {
            String filename = "generation" + generation + ".individuals.txt";
            File file = new File(reportDirectory, filename);
            FileWriter filewriter;
            try {
                filewriter = new FileWriter(file);
                for (IIndividual ind : inhabitants) {
                    filewriter.flush();
                    filewriter.write(ind + "\n");
                }
                filewriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Best individual
        IIndividual best = IndividualStatistics.bestIndividual(inhabitants, comparator);
        ClusteringEvaluator eval = ((ClusteringEvaluator) algorithm.getEvaluator());
        eval.fullEvaluation(best);

        // Do population report
        StringBuilder sb = new StringBuilder("Generation " + generation + " Report\n");
        sb.append("Best individual: ").append(best).append("\n");
        sb.append("RMSSD: ").append(eval.getClusterEval().getRmssd()).append("\n");
        sb.append("Silhouete: ").append(eval.getClusterEval().getSilhouette()).append("\n");
        sb.append("XB: ").append(eval.getClusterEval().getXb()).append("\n");
        sb.append("DB: ").append(eval.getClusterEval().getDb()).append("\n");
        sb.append("S_Dbw: ").append(eval.getClusterEval().getSdbw()).append("\n");
        sb.append("DBCV: ").append(eval.getClusterEval().getDbcv()).append("\n");
        sb.append("Entropy: ").append(eval.getClusterEval().getEntropy()).append("\n");
        sb.append("Purity: ").append(eval.getClusterEval().getPurity()).append("\n");
        sb.append("Rand: ").append(eval.getClusterEval().getRand()).append("\n");
        sb.append("Precision: ").append(eval.getClusterEval().getPurity()).append("\n");
        sb.append("Recall: ").append(eval.getClusterEval().getMacroRecall()).append("\n");
        sb.append("F1: ").append(eval.getClusterEval().getMacroF1()).append("\n");
        sb.append("Specificity: ").append(eval.getClusterEval().getMacroSpecificity()).append("\n");
        sb.append("Conf Mat: ").append(PrintConfusionMatrix.singleLine(eval.getClusterEval().getExtEvalResult())).append("\n");

        // Worst individual
        IIndividual worst = IndividualStatistics.worstIndividual(inhabitants, comparator);
        sb.append("Worst individual: ").append(worst).append("\n");
        // Median individual
        IIndividual median = IndividualStatistics.medianIndividual(inhabitants, comparator);
        sb.append("Median individual: ").append(median).append("\n");
        // Average fitness and fitness variance
        double[] avgvar = IndividualStatistics.averageFitnessAndFitnessVariance(inhabitants);
        sb.append("Average fitness = ").append(avgvar[0]).append("\n");
        sb.append("Fitness variance = ").append(avgvar[1]).append("\n");

        // Write report string to the standard output (if necessary)
        if (reportOnConsole) {
            System.out.println(sb.toString());
        }

        // Write string to the report file (if necessary)
        if (reportOnFile) {
            try {
                reportFileWriter.write(sb.toString());
                reportFileWriter.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void algorithmFinished(AlgorithmEvent event) {
        // Do last generation report
        doIterationReport((PopulationAlgorithm) event.getAlgorithm(), true);

        if (reportOnFile) {
            try {
                doFinalReport((PopulationAlgorithm) event.getAlgorithm());
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }

        // Close report file if necessary
        if (reportOnFile && reportFile != null) {
            try {
                reportFileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void doFinalReport(PopulationAlgorithm algorithm) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append(algorithm.getGeneration()).append(",").append(algorithm.getPopulationSize()).append(",");
        if (algorithm instanceof MIGKA) {
            MIGKA sge = (MIGKA) algorithm;
            sb.append(sge.getMutator().getMutProb()).append(",").append(sge.getKmo().getMutProb()).append(",");
        } else if (algorithm instanceof MIGCUK) {
            MIGCUK migcuk = (MIGCUK) algorithm;
            sb.append(migcuk.getRecombinator().getRecProb()).append(",").append(migcuk.getMutator().getMutProb()).append(",");
        } else if (algorithm instanceof CHCMIClustering) {
            CHCMIClustering chc = (CHCMIClustering) algorithm;
            sb.append(chc.getInitialD()).append(",").append(chc.getNumberOfSurvivors()).append(",").append(chc.getCm().getMutProb())
                    .append(",").append(chc.getKmo().getMutProb()).append(",");
        }

        ClusteringEvaluator evaluator = (ClusteringEvaluator) algorithm.getEvaluator();
        sb.append(evaluator.getDataset()).append(", ").append(evaluator.getClusterEval().getDistanceFunction()).append(", ")
                .append(evaluator.getNumClusters()).append(", ").append(evaluator.getMetric()).append(", ");

        // Best individual
        List<IIndividual> inhabitants = algorithm.getInhabitants();
        Comparator<IFitness> comparator = algorithm.getEvaluator().getComparator();
        IIndividual best = IndividualStatistics.bestIndividual(inhabitants, comparator);
        ClusteringEvaluator eval = ((ClusteringEvaluator) algorithm.getEvaluator());
        eval.fullEvaluation(best);

        sb.append(eval.getClusterEval().getRmssd()).append(", ").append(eval.getClusterEval().getSilhouette()).append(", ").append(eval.getClusterEval().getXb()).append(", ")
                .append(eval.getClusterEval().getDb()).append(", ").append(eval.getClusterEval().getSdbw()).append(", ").append(eval.getClusterEval().getDbcv()).append(", ")
                .append(eval.getClusterEval().getEntropy()).append(", ").append(eval.getClusterEval().getPurity()).append(", ").append(eval.getClusterEval().getRand()).append(", ")
                .append(eval.getClusterEval().getMacroPrecision()).append(", ").append(eval.getClusterEval().getMacroRecall()).append(", ")
                .append(eval.getClusterEval().getMacroF1()).append(", ").append(eval.getClusterEval().getMacroSpecificity()).append(", ")
                .append(PrintConfusionMatrix.singleLine(eval.getClusterEval().getExtEvalResult())).append(", ");

        String finalReport = reportTitle + ".final.csv";
        File fileReport = new File(finalReport);
        BufferedWriter bwFinal;
        if (fileReport.exists()) {
            bwFinal = new BufferedWriter(new FileWriter(finalReport, true));
            bwFinal.write(System.getProperty("line.separator"));
        } else {
            bwFinal = new BufferedWriter(new FileWriter(finalReport));
            bwFinal.append("Generations,Population,");
            if (algorithm instanceof MIGKA)
                bwFinal.append("Mut. prob,KMO prob,");
            else if (algorithm instanceof MIGCUK)
                bwFinal.append("Mut. prob,Rec. prob,");
            else if (algorithm instanceof CHC)
                bwFinal.append("Init d,Survivors,Mut. prob,KMO prob,");

            bwFinal.write("Dataset,Distance,Clusters,Fitness,RMSSD,Silhouete,XB,DB," +
                    "S_Dbw,DBCV,Entropy,Purity,Rand index,Precision,Recall,F1,Specificity,Conf Mat,Report,Seed\n");
        }

        sb.append(actualReportTitle).append(", ");
        sb.append(((RanecuFactory) algorithm.getRandGenFactory()).getSeed());
        bwFinal.write(sb.toString());
        bwFinal.close();
    }
}
