package jclec.problem.util.dataset;

import jclec.JCLEC;

/**
 * Interface for dataset examples
 *
 * @author Alberto Cano
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna
 * @author Juan Luis Olmo
 */

public interface IExample extends JCLEC {
    /**
     * Returns vector's attribute value in internal format.
     *
     * @param attributeIndex attribute index for value to read.
     * @return the specified value as a double
     */

    double getValue(int attributeIndex);

    /**
     * Checks if the example is equal to another
     *
     * @return true or false
     */

    boolean equals(Object other);

    /**
     * Copy method
     *
     * @return A copy of this example
     */

    IExample copy();
}