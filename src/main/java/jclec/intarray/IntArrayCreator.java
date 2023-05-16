package jclec.intarray;

import jclec.ISpecies;
import jclec.base.AbstractCreator;
import jclec.util.intset.IIntegerSet;

/**
 * Creation of IntArrayIndividual (and subclasses).
 *
 * @author Sebastian Ventura
 */

public class IntArrayCreator extends AbstractCreator {

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------- Internal variables
    /////////////////////////////////////////////////////////////////

    /**
     * Associated species
     */
    protected transient IntArraySpecies species;

    /**
     * Genotype schema
     */
    protected transient IIntegerSet[] schema;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    public IntArrayCreator() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // java.lang.Object methods

    public boolean equals(Object other) {
        return other instanceof IntArrayCreator;
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    // AbstractCreator methods

    @Override
    protected void prepareCreation() {
        ISpecies spc = context.getSpecies();
        if (spc instanceof IntArraySpecies) {
            // Sets individual species
            this.species = (IntArraySpecies) spc;
            // Sets genotype schema
            this.schema = this.species.getGenotypeSchema();
        } else {
            throw new IllegalStateException("Illegal species in context");
        }
    }

    @Override
    protected void createNext() {
        createdBuffer.add(species.createIndividual(createGenotype()));
    }

    /////////////////////////////////////////////////////////////////
    // ---------------------------------------------- Private methods
    /////////////////////////////////////////////////////////////////

    /**
     * Create a int [] genotype, filling it randomly
     */
    protected int[] createGenotype() {
        int gl = schema.length;
        int[] result = new int[gl];
        for (int i = 0; i < gl; i++) {
            result[i] = schema[i].getRandom(randgen);
        }
        return result;
    }
}
