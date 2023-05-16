package jclec.util.random;

import jclec.JCLEC;

/**
 * Random generator factory.
 * 
 * @author Sebastian Ventura
 */

public interface IRandGenFactory extends JCLEC 
{
	/**
	 * Factory method.
	 * 
	 * @return A new instance of a random generator
	 */
	
	IRandGen createRandGen();
}
