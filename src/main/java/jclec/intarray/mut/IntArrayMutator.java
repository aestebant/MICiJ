package jclec.intarray.mut;

import jclec.ISpecies;
import jclec.base.AbstractParallelMutator;
import jclec.intarray.IntArrayIndividual;
import jclec.intarray.IntArraySpecies;
import jclec.util.intset.IIntegerSet;


/**
 * IntArrayIndividual (and subclasses) specific mutator.
 *
 * @author Sebastian Ventura
 */

public abstract class IntArrayMutator extends AbstractParallelMutator {

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------- Internal variables
    /////////////////////////////////////////////////////////////////

    /**
     * Individuals species
     */
    protected transient IntArraySpecies species;

    /**
     * Individuals schema
     */
    protected transient IIntegerSet[] schema;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    public IntArrayMutator() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    // AbstractMutator methods

    /**
     * {@inheritDoc}
     */
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
        // Individual to mutate
        IntArrayIndividual mutant = (IntArrayIndividual) parentsBuffer.get(parentsCounter);
        // Returns mutant
        sonsBuffer.add(mutateInd(mutant));
    }

    /**
     * Gets a mutate locus in represented individuals
     */
    protected final int getMutableLocus() {
        int genotypeLength = schema.length;
        int ml;
        do {
            ml = randgen.choose(0, genotypeLength);
        }
        while (schema[ml].size() == 1);
        // Return mutation point
        return ml;
    }

    /**
     * Flip method.
     *
     * @param chrom Chromosome affected
     * @param locus Locus affected
     */
    protected final void flip(int[] chrom, int locus) {
        // New locus value
        int newval;
        // Choose mutated value
        do {
            newval = schema[locus].getRandom(randgen);
        }
        while (chrom[locus] == newval);
        // Assigns new value
        chrom[locus] = newval;
    }
}
