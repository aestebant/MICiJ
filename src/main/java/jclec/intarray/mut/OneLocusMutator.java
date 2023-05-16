package jclec.intarray.mut;

import jclec.IIndividual;
import jclec.intarray.IntArrayIndividual;

/**
 * One locus mutation in IntArrayIndividuals
 *
 * @author Sebastian Ventura
 */

public class OneLocusMutator extends IntArrayMutator {

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    public OneLocusMutator() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // AbstractMutator methods

    @Override
    protected IIndividual mutateInd(IIndividual individual) {
        // Genome length
        int gl = species.getGenotypeLength();
        IntArrayIndividual mutant = (IntArrayIndividual) individual;
        // Creates mutant genotype
        int[] mgenome = new int[gl];
        System.arraycopy(mutant.getGenotype(), 0, mgenome, 0, gl);
        // Choose mutation point
        int mp = getMutableLocus();
        // Flip selected point
        flip(mgenome, mp);
        return species.createIndividual(mgenome);
    }

    // java.lang.Object methods
    public boolean equals(Object other) {
        return other instanceof OneLocusMutator;
    }
}
