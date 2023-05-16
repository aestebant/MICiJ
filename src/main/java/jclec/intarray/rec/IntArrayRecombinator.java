package jclec.intarray.rec;

import jclec.ISpecies;
import jclec.base.AbstractParallelRecombinator;
import jclec.intarray.IntArraySpecies;

/**
 * IntArrayIndividual (and subclasses) specific recombinator.
 *
 * @author Sebastian Ventura
 */

public abstract class IntArrayRecombinator extends AbstractParallelRecombinator {
    private static final long serialVersionUID = 1527304795896439539L;

    /////////////////////////////////////////////////////////////////
    // --------------------------------------------------- Attributes
    /////////////////////////////////////////////////////////////////

    /**
     * Individual species
     */

    protected IntArraySpecies species;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor.
     */

    public IntArrayRecombinator() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    // AbstractRecombinator methods

    /**
     * Sets ppl = 2
     * <p>
     * {@inheritDoc}
     */

    @Override
    protected void setPpl() {
        this.ppl = 2;
    }

    /**
     * Sets spl = 2
     * <p>
     * {@inheritDoc}
     */

    @Override
    protected void setSpl() {
        this.spl = 2;
    }

    /**
     * {@inheritDoc}
     */

    @Override
    protected void prepareRecombination() {
        // Sets individual species
        ISpecies spc = context.getSpecies();
        if (spc instanceof IntArraySpecies) {
            this.species = (IntArraySpecies) spc;
        } else {
            throw new IllegalStateException("Invalid population species");
        }
    }
}
