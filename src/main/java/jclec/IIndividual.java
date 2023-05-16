package jclec;

/**
 * General purpose individual.
 * <p>
 * This interface  defines common operations for a  general-purpose individual.
 * Methods implementation  depends on the individual's  encoding -that is, what
 * is the type of genotype it presents, the presence of phenotype, ... </p>
 *
 * @author Sebastian Ventura
 */

public interface IIndividual extends JCLEC {
    /////////////////////////////////////////////////////////////////
    // ---------------------------------- Setting and getting fitness
    /////////////////////////////////////////////////////////////////

    /**
     * Access to individual fitness.
     *
     * @return Individual fitness
     */

    IFitness getFitness();

    /**
     * Sets the fitness of this individual.
     *
     * @param fitness New fitness value.
     */

    void setFitness(IFitness fitness);

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------------- Copy method
    /////////////////////////////////////////////////////////////////

    /**
     * Copy method.
     *
     * @return a copy of this individual.
     */

    IIndividual copy();

    /**
     * Equals method.
     *
     * @param other object to compare
     * @return true if the individual is equal to the other, false otherwise
     */

    boolean equals(Object other);
}
