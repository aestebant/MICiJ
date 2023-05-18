package miclustering.algorithms.evolutionary.migcuk;

import miclustering.algorithms.OneStepKMeans;
import jclec.IIndividual;
import jclec.fitness.SimpleValueFitness;
import jclec.intarray.IntArrayIndividual;
import miclustering.algorithms.evolutionary.utils.ClusteringEvaluator;
import org.apache.commons.configuration.Configuration;

import java.util.List;

public class MIGCUKEvaluator extends ClusteringEvaluator {

    private OneStepKMeans oskm;

    @Override
    protected void evaluate(IIndividual ind) {
        int[] genotype = ((IntArrayIndividual) ind).getGenotype();
        List<Integer> clusterAssignment = oskm.assignBagsToClusters(genotype, false);
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
        List<Integer> clusterAssignment = oskm.assignBagsToClusters(genotype, false);
        clusterEval.fullEvaluation(clusterAssignment, true);
    }

    @Override
    public void configure(Configuration settings) {
        super.configure(settings);
        k = settings.getInt("kmax", 2);
        oskm = new OneStepKMeans(dataset, distanceFunction, distanceConfig, k, true);
        //TODO QUEDA PENDIENTE LA IMPLEMENTACIÓN PARA K VARIABLE
    }

    public OneStepKMeans getOskm() {
        return oskm;
    }
}
