package miclustering.algorithms.evolutionary;

import miclustering.algorithms.OneStepKMeans;
import jclec.IIndividual;
import jclec.ISpecies;
import jclec.base.AbstractParallelMutator;
import jclec.intarray.IntArrayIndividual;
import jclec.intarray.IntArraySpecies;
import jclec.util.intset.IIntegerSet;

import java.util.ArrayList;
import java.util.List;

public class KMeansOperator extends AbstractParallelMutator {
    /**
     * Individuals species
     */
    protected transient IntArraySpecies species;

    /**
     * Individuals schema
     */
    protected transient IIntegerSet[] schema;

    protected OneStepKMeans oskm;

    public KMeansOperator(String dataset, String distFunc, String confDist, int numClusters) {
        super();
        oskm = new OneStepKMeans(dataset, distFunc, confDist, numClusters, true);
    }

    @Override
    protected void prepareMutation() {
        ISpecies spc = context.getSpecies();
        if (spc instanceof IntArraySpecies) {
            // Sets individual species
            this.species = (IntArraySpecies) spc;
            // Sets genotype schema
            this.schema = this.species.getGenotypeSchema();
        } else {
            throw new IllegalStateException("Invalid species in context");
        }
    }

    @Override
    protected void mutateNext() {
        IntArrayIndividual mutant = (IntArrayIndividual) parentsBuffer.get(parentsCounter);
        // Returns mutant
        sonsBuffer.add(mutateInd(mutant));
    }

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
        for (int gen : mgenome)
            clusterAssignments.add(gen);

        List<Integer> mutation = oskm.evaluate(clusterAssignments, false);

        for (int i = 0; i < mutation.size(); ++i)
            mgenome[i] = mutation.get(i);
        // Returns mutant
        return species.createIndividual(mgenome);
    }
}
