package jclec;

import java.util.List;

/**
 * Generate new individuals by recombination.
 * 
 * @author Sebastian Ventura
 */

public interface IRecombinator extends ITool 
{
	/**
	 * Informs about the number of parents in a litter.
	 * 
	 * @return Number of parents per litter
	 */

	int getPpl();

	/**
	 * Informs about the number of sons per litter.
	 * 
	 * @return Number of sons per litter
	 */

	int getSpl();
	
	/**
	 * Recombination method.
	 * 
	 * @param parents Individuals to recombine
	 * 
	 * @return Recombination result
	 */

	List<IIndividual> recombine(List<IIndividual> parents);
}
