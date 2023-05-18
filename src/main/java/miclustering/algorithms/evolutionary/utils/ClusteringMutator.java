package miclustering.algorithms.evolutionary.utils;

import jclec.IIndividual;
import jclec.fitness.SimpleValueFitness;
import jclec.fitness.ValueFitnessComparator;
import jclec.intarray.IntArrayIndividual;
import jclec.intarray.mut.UniformMutator;

import java.util.*;
import java.util.stream.DoubleStream;

public class ClusteringMutator extends UniformMutator {
    protected ClusteringEvaluator evaluator;

    @Override
    protected IIndividual mutateInd (IIndividual individual) {
        // Genotype length
        int gl = species.getGenotypeLength();
        // Individual to be mutated
        IntArrayIndividual mutant = (IntArrayIndividual) individual;
        // Creates mutant genotype
        int[] mgenome = new int[gl];
        System.arraycopy(mutant.getGenotype(), 0, mgenome, 0, gl);
        // Mutate loci...
        for (int i = 0; i < gl; i++) {
            if (randgen.coin(locusMutationProb)) {
                double[] fitness = new double[evaluator.getNumClusters()];
                for (int j = 0; j < evaluator.getNumClusters(); ++j) {
                    mgenome[i] = j;
                    IntArrayIndividual aux = new IntArrayIndividual(mgenome);
                    evaluator.evaluate(aux);
                    fitness[j] = ((SimpleValueFitness) aux.getFitness()).getValue();
                }
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

    public void setEvaluator(ClusteringEvaluator evaluator) {
        this.evaluator = evaluator;
    }
}
