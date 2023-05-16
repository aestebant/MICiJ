package jclec.fitness;

import jclec.IFitness;

/**
 * Fitness that contains a double value.
 * <p>
 * This kind of fitness is used in roulette-based selection.
 *
 * @author Sebastian Ventura
 */

public interface IValueFitness extends IFitness {
    /**
     * @return Value for this fitness.
     * @throws UnsupportedOperationException The method hasn't been implemented.
     */

    double getValue();

    /**
     * Sets fitness value.
     *
     * @param value New fitness value
     */

    void setValue(double value);
}
