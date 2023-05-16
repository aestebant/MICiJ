package jclec.fitness;

import jclec.IFitness;

import java.util.Comparator;

/**
 * Compare two IValueFitness objects according their associated fitness value.
 *
 * @author Sebastian Ventura
 */

public class ValueFitnessComparator implements Comparator<IFitness> {
    /////////////////////////////////////////////////////////////////
    // --------------------------------------------------- Properties
    /////////////////////////////////////////////////////////////////

    /**
     * Fitness is to minimize or to maximize
     */

    private boolean minimize = false;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor.
     */

    public ValueFitnessComparator() {
        super();
    }

    /**
     * Default constructor. Sets the ascendent flag.
     */

    public ValueFitnessComparator(boolean minimize) {
        super();
        setMinimize(minimize);
    }

    /////////////////////////////////////////////////////////////////
    // ------------------------------- Setting and getting properties
    /////////////////////////////////////////////////////////////////

    /**
     * Access to 'inverse' flag
     *
     * @return Inverse flag value
     */

    public boolean isMinimize() {
        return minimize;
    }

    /**
     * Sets the minimization flag
     *
     * @param minimize minimization flag
     */

    public void setMinimize(boolean minimize) {
        this.minimize = minimize;
    }

    /////////////////////////////////////////////////////////////////
    // ------------------ Implementing Comparator<IFitness> interface
    /////////////////////////////////////////////////////////////////

    /**
     * {@inheritDoc}
     */

    public int compare(IFitness fitness1, IFitness fitness2) {
        double f1value, f2value;
        try {
            f1value = ((IValueFitness) fitness1).getValue();
        } catch (ClassCastException e) {
            throw new IllegalArgumentException
                    ("IValueFitness expected in fitness1");
        }
        try {
            f2value = ((IValueFitness) fitness2).getValue();
        } catch (ClassCastException e) {
            throw new IllegalArgumentException
                    ("IValueFitness expected in fitness2");
        }
        if (f1value > f2value) {
            return minimize ? -1 : 1;
        } else if (f1value < f2value) {
            return minimize ? 1 : -1;
        } else {
            return 0;
        }
    }
}
