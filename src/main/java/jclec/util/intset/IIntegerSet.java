package jclec.util.intset;

import jclec.JCLEC;
import jclec.util.random.IRandGen;

/**
 * Integer values range
 *
 * @author Sebastian Ventura
 */

public interface IIntegerSet extends JCLEC {
    /**
     * Returns true if value belongs to this range.
     *
     * @param value Value that will be tested
     * @return true if this value...
     */

	boolean contains(int value);

    /**
     * Number of elements in this range.
     *
     * @return Range size
     */

	int size();

    /**
     * Returns a random value contained in this range.
     *
     * @param randgen Random generator used.
     * @return A random value ...
     */

	int getRandom(IRandGen randgen);
}
