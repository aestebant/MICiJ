package miclustering.algorithms.evolutionary;

import miclustering.evaluators.ClusterEvaluation;
import jclec.IIndividual;
import jclec.fitness.SimpleValueFitness;
import jclec.intarray.IntArrayIndividual;
import org.apache.commons.configuration.Configuration;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.List;

public class MIGGAEvaluator extends ClusteringEvaluator {
    @Override
    protected void evaluate(IIndividual ind) {
        int[] genotype = ((IntArrayIndividual) ind).getGenotype();
        List<Integer> clusterAssignment = new ArrayList<>(genotype.length - 1);
        for (int i = 0; i < genotype.length - 1; ++i)
            clusterAssignment.add(genotype[i]);
        double fitness = -1;
        switch (metric) {
            case "rmssd":
                fitness = clusterEval.computeRmssd(clusterAssignment, false);
                break;
            case "silhouette":
                fitness = clusterEval.computeSilhouette(clusterAssignment);
                break;
            case "db":
                fitness = clusterEval.computeDb(clusterAssignment, false);
                break;
            case "xb":
                fitness = clusterEval.computeXb(clusterAssignment, false);
                break;
            case "sdbw":
                fitness = clusterEval.computeSdbw(clusterAssignment);
                break;
            case "dbcv":
                fitness = clusterEval.computeDbcv(clusterAssignment);
                break;
            case "twcv":
                fitness = clusterEval.computeTwcv(clusterAssignment, false);
                break;
            case "ftwcv":
                fitness = clusterEval.computeFtwcv(clusterAssignment);
                break;
            default:
                System.err.println("Not known metric");
                System.exit(-1);
        }
        ind.setFitness(new SimpleValueFitness(fitness));
    }

    @Override
    protected void fullEvaluation(IIndividual ind) {
        int[] genotype = ((IntArrayIndividual) ind).getGenotype();
        List<Integer> clusterAssignment = new ArrayList<>(genotype.length - 1);
        for (int i = 0; i < genotype.length - 1; ++i)
            clusterAssignment.add(genotype[i]);
        clusterEval.fullEvaluation(clusterAssignment, true);
    }

    @Override
    public void configure(Configuration settings) {
        dataset = settings.getString("dataset");
        k = settings.getInt("max-of-clusters");
        distanceFunction = settings.getString("distance[@type]");
        distanceConfig = settings.getString("distance.config", "");
        metric = settings.getString("metric", "silhouette");
        String evalConfig = "-d " + dataset + " -c last -k " + k + " -parallelize -r -A " + String.join(" ", distanceFunction, distanceConfig);
        clusterEval = new ClusterEvaluation();
        try {
            clusterEval.setOptions(Utils.splitOptions(evalConfig));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
