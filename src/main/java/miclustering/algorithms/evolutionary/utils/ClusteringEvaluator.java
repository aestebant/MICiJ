package miclustering.algorithms.evolutionary.utils;

import miclustering.evaluators.ClusterEvaluation;
import jclec.IConfigure;
import jclec.IFitness;
import jclec.IIndividual;
import jclec.base.AbstractParallelEvaluator;
import jclec.fitness.SimpleValueFitness;
import jclec.fitness.ValueFitnessComparator;
import jclec.intarray.IntArrayIndividual;
import org.apache.commons.configuration.Configuration;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ClusteringEvaluator extends AbstractParallelEvaluator implements IConfigure {
    protected ClusterEvaluation clusterEval;
    private Comparator<IFitness> comparator;
    protected String metric;
    protected String dataset;
    protected int k;
    protected String distanceFunction;
    protected String distanceConfig;

    public ClusterEvaluation getClusterEval() {
        return clusterEval;
    }

    @Override
    protected void evaluate(IIndividual ind) {
        int[] genotype = ((IntArrayIndividual) ind).getGenotype();
        List<Integer> clusterAssignment = new ArrayList<>(genotype.length);
        for (int value : genotype) clusterAssignment.add(value);
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

    protected void fullEvaluation(IIndividual ind) {
        int[] genotype = ((IntArrayIndividual) ind).getGenotype();
        List<Integer> clusterAssignment = new ArrayList<>(genotype.length);
        for (int value : genotype) clusterAssignment.add(value);
        clusterEval.fullEvaluation(clusterAssignment, true);
    }

    @Override
    public long getEvaluationTime() {
        return 0;
    }

    @Override
    public Comparator<IFitness> getComparator() {
        boolean minimize = true;
        switch (metric) {
            case "silhouette": case "dbcv": case "ftwcv":
                minimize = false;
                break;
            case "rmssd": case "db": case "xb": case "sdbw": case "twcv":
                minimize = true;
                break;
            default:
                System.err.println("Not known metric");
                System.exit(-1);
        }
        if (comparator == null) {
            comparator = new ValueFitnessComparator(minimize);
        }
        return comparator;
    }

    public String getDataset() {
        return dataset;
    }

    public int getNumClusters() {
        return k;
    }

    public String getMetric() {
        return metric;
    }

    @Override
    public void configure(Configuration settings) {
        dataset = settings.getString("dataset");
        k = settings.getInt("num-clusters", 2);
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
