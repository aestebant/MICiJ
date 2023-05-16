package jclec.fitness;

import jclec.IFitness;

import java.util.Comparator;

/**
 * Compare two ICompositeFitness objects according one of their components. The
 * index of selected component is a user-defined parameter. 
 *  
 * @author Sebastian Ventura
 */

public class ComponentComparator extends CompositeFitnessComparator 
{
	/////////////////////////////////////////////////////////////////
	// --------------------------------------------------- Properties
	/////////////////////////////////////////////////////////////////
	
	/** Index of component that will be compared */
	
	protected int activeComponentIndex;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////
	
	/**
	 * Empty constructor
	 */
	
	public ComponentComparator() 
	{
		super();
	}

	/**
	 * Constructor that sets component comparators and active component index.
	 */
	
	public ComponentComparator(Comparator<IFitness>[] componentComparators, int activeComponentIndex) 
	{
		super(componentComparators);
	}

	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Protected methods
	/////////////////////////////////////////////////////////////////
		
	@Override
	protected int compare(ICompositeFitness cfitness0, ICompositeFitness cfitness1) 
	{
		return componentComparators[activeComponentIndex].compare
			(cfitness0.getComponent(activeComponentIndex), cfitness1.getComponent(activeComponentIndex));
	}

}
