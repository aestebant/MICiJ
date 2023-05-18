package miclustering.algorithms.evolutionary.migcuk;

import jclec.IIndividual;
import jclec.intarray.IntArrayIndividual;
import jclec.intarray.mut.UniformMutator;

public class MIGCUKMutator extends UniformMutator {
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
                // New locus value
                int newval;
                // Choose mutated value
                do {
                    newval = schema[i].getRandom(randgen);
                } while (mgenome[i] == newval);
                // Assigns new value
                mgenome[i] = newval;
            }
        }

        //TODO HACER GENERAL PARA CUALQUIER LONGITUD DE GENOTIPO
        while (mgenome[0] == mgenome[1]) {
            int maxVar = (int) (0.25*schema[1].size());
            // New locus value
            int newval;
            // Choose mutated value
            do {
                newval = schema[1].getRandom(randgen);
            } while (mgenome[1] == newval || newval > mgenome[1] + maxVar || newval < mgenome[1] - maxVar);
            // Assigns new value
            mgenome[1] = newval;
        }
        // Returns mutant
        return species.createIndividual(mgenome);
    }
}
