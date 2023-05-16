package jclec.intarray.mut;

import jclec.IConfigure;
import jclec.IIndividual;
import jclec.intarray.IntArrayIndividual;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.lang.builder.EqualsBuilder;

/**
 * Uniform mutator for IntArrayIndividual (and subclasses).
 *
 * @author Sebastian Ventura
 */

public class UniformMutator extends IntArrayMutator implements IConfigure {

    /////////////////////////////////////////////////////////////////
    // --------------------------------------------------- Properties
    /////////////////////////////////////////////////////////////////

    /**
     * Crossover probability
     */
    protected double locusMutationProb;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    public UniformMutator() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // Setting and getting properties

    /**
     * @return Returns the crossprob.
     */
    public double getLocusMutationProb() {
        return locusMutationProb;
    }

    /**
     * @param mutProb New mutation probability
     */
    public final void setLocusMutationProb(double mutProb) {
        this.locusMutationProb = mutProb;
    }

    // IConfigure interface

    /**
     * Configuration method.
     * <p>
     * Configuration parameters for UniformMutator are:
     *
     * <ul>
     * <li>
     * <code>[@evaluate]: boolean (default = true)</code></p>
     * If this parameter is set to <code>true</true> individuals will
     * be evaluated after its creation.
     * </li>
     * <li>
     * <code>[@locus-mutation-prob]: double (default = 0.5)</code></p>
     * Locus mutation probability.
     * </li>
     * <li>
     * <code>random-generator: complex</code></p>
     * Random generator used in individuals mutation.
     * <ul>
     * <li>
     * <code>random-generator[@type] String (default 'jclec.random.Ranecu')</code>
     * </li>
     * </ul>
     * </li>
     * </ul>
     */
    public void configure(Configuration configuration) {
        // Get the 'locus-mutation-prob' property
        double locusMutationProb = configuration.getDouble("[@locus-mut-prob]", 0.5);
        setLocusMutationProb(locusMutationProb);
    }

    // java.lang.Object methods
    public boolean equals(Object other) {
        if (other instanceof UniformMutator) {
            // Type conversion
            UniformMutator cother = (UniformMutator) other;
            // Equals Builder
            EqualsBuilder eb = new EqualsBuilder();
            eb.append(locusMutationProb, cother.locusMutationProb);
            // Returns
            return eb.isEquals();
        } else {
            return false;
        }
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    // AbstractMutator methods

    @Override
    protected IIndividual mutateInd(IIndividual individual) {
        // Genotype length
        int gl = species.getGenotypeLength();
        IntArrayIndividual mutant = (IntArrayIndividual) individual;
        // Creates mutant genotype
        int[] mgenome = new int[gl];
        System.arraycopy(mutant.getGenotype(), 0, mgenome, 0, gl);
        // Mutate loci...
        for (int i = 0; i < gl; i++) {
            if (randgen.coin(locusMutationProb)) flip(mgenome, i);
        }
        return species.createIndividual(mgenome);
    }
}
