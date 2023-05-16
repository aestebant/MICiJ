package jclec.intarray;

import jclec.IFitness;
import jclec.IIndividual;
import jclec.base.AbstractIndividual;
import org.apache.commons.lang.builder.EqualsBuilder;

/**
 * Individual with a byte array as genotype.
 *
 * @author Sebastian Ventura
 */

public class IntArrayIndividual extends AbstractIndividual<int[]> {

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    public IntArrayIndividual() {
        super();
    }

    /**
     * Constructor that sets individual genotype
     *
     * @param genotype Individual genotype
     */
    public IntArrayIndividual(int[] genotype) {
        super(genotype);
    }

    /**
     * Constructor that sets individual genotype and fitness
     *
     * @param genotype Individual genotype
     * @param fitness  Individual fitness
     */
    public IntArrayIndividual(int[] genotype, IFitness fitness) {
        super(genotype, fitness);
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // IIndividual interface

    /**
     * {@inheritDoc}
     */
    public IIndividual copy() {
        // Genotype length
        int gl = genotype.length;
        // Allocate a copy of genotype
        int[] gother = new int[genotype.length];
        // Copy genotype
        System.arraycopy(genotype, 0, gother, 0, gl);
        // Create new individuals, then return it
        if (fitness != null) {
            return new IntArrayIndividual(gother, fitness.copy());
        } else {
            return new IntArrayIndividual(gother);
        }
    }

    // java.lang.Object methods

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(Object other) {
        if (other instanceof IntArrayIndividual) {
            IntArrayIndividual iaother = (IntArrayIndividual) other;
            EqualsBuilder eb = new EqualsBuilder();
            eb.append(genotype, iaother.genotype);
            eb.append(fitness, iaother.fitness);
            return eb.isEquals();
        } else {
            return false;
        }
    }
}
