package miclustering.algorithms.evolutionary;

import jclec.IIndividual;
import jclec.fitness.ValueFitnessComparator;
import jclec.intarray.IntArrayIndividual;
import miclustering.algorithms.evolutionary.utils.ClusteringMutator;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import weka.core.Instance;

import java.util.*;
import java.util.stream.DoubleStream;

public class FastClusteringMutator extends ClusteringMutator {

    @Override
    protected IIndividual mutateInd(IIndividual individual) {
        // Genotype length
        int gl = species.getGenotypeLength();
        // Individual to be mutated
        IntArrayIndividual mutant = (IntArrayIndividual) individual;
        // Creates mutant genotype
        int[] mgenome = new int[gl];
        System.arraycopy(mutant.getGenotype(), 0, mgenome, 0, gl);

        List<Integer> clusterAssignments = new ArrayList<>(gl);
        for (int value : mgenome)
            clusterAssignments.add(value);
        Map<Integer, Instance> centroids = evaluator.getClusterEval().getDatasetCentroids().compute(clusterAssignments, false);

        // Mutate loci...
        for (int i = 0; i < gl; i++) {
            if (randgen.coin(locusMutationProb)) {
                double[] distances = evaluator.getClusterEval().getDatasetCentroids().distanceToCentroids(centroids, clusterAssignments.get(i));
                Max getMax = new Max();
                double[] fitness = new double[evaluator.getNumClusters()];
                for (int j = 0; j < evaluator.getNumClusters(); ++j)
                    fitness[j] = 1.5 * getMax.evaluate(distances) - distances[j] + 0.5;

                Integer[] clusters = new Integer[evaluator.getNumClusters()];
                Double[] probabilities = new Double[evaluator.getNumClusters()];
                for (int j = 0; j < evaluator.getNumClusters(); ++j) {
                    clusters[j] = j;
                    probabilities[j] = fitness[j] / DoubleStream.of(fitness).sum();
                }

                final List<Integer> clusterCopy = Arrays.asList(clusters);
                ArrayList<Integer> sortedClusters = new ArrayList<>(clusterCopy);
                sortedClusters.sort(Comparator.comparing(s -> probabilities[clusterCopy.indexOf(s)]));
                Arrays.sort(probabilities);

                double random = randgen.uniform(0,1);
                if (!((ValueFitnessComparator) evaluator.getComparator()).isMinimize()) {
                    Collections.reverse(sortedClusters);
                }
                double sum = 0D;
                int j = 0;
                while (sum < random) {
                    sum += probabilities[j];
                    ++j;
                }
                mgenome[i] = sortedClusters.get(j-1);
            }
        }
        // Returns mutant
        return species.createIndividual(mgenome);
    }
}
