package jclec;

import java.io.Serializable;

/**
 * Generic tool.
 * <p>
 * This interface represents objects that perform a task in an evolutionary
 * algorithm: individual selection, crossover or mutation operations, etc.
 *
 * @author Sebastian Ventura
 */

public interface ITool extends Serializable {
    /**
     * Set the system where ...
     *
     * @param context Execution context
     */

    void contextualize(ISystem context);
}


