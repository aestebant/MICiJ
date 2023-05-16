package jclec.problem.classification;

import jclec.IIndividual;

/**
 * Individual representation interface for evolutionary algorithms for classification 
 * 
 * @author Alberto Cano 
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna 
 * @author Juan Luis Olmo
 */

public interface IClassifierIndividual extends IIndividual
{
	/**
	 * Access to individual phenotype
	 * 
	 * @return A classifier represented by this individual
	 */
	
	public IClassifier getPhenotype();
}