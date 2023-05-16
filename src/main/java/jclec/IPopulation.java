package jclec;

import java.util.List;

/**
 * Generic population.
 *
 * @author Sebastian Ventura
 */

public interface IPopulation extends ISystem {
    // System species

    /**
     * Access to system species.
     *
     * @return System species
     */

    ISpecies getSpecies();

    // System evaluator

    /**
     * Access to system evaluator.
     */

    IEvaluator getEvaluator();

    // Generation counter

    /**
     * Access to actual generation.
     *
     * @return Actual generation
     */

    int getGeneration();

    // Population individuals

    /**
     * Access to population inhabitants.
     *
     * @return Population inhabitants
     */

    List<IIndividual> getInhabitants();
}
