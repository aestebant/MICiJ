package jclec;

import java.util.Comparator;
import java.util.List;

/**
 * Individuals evaluator.
 * 
 * @author Sebastian Ventura
 * 
 * @see jclec.IFitness
 */

public interface IEvaluator extends JCLEC
{
	/**
	 * Evaluation method.
	 * 
	 * @param inds Individuals to evaluate
	 */
	
	void evaluate(List<IIndividual> inds);
	
	/**
	 * Get the number of individuals evaluated until now.
	 * 
	 * @return Number of evaluations until now
	 */

	/**
	 * Get the evaluation time executed until now.
	 *
	 * @return Evaluation time
	 */

	long getEvaluationTime();

	int getNumberOfEvaluations();
	
	/**
	 * Access to fitness comparator.
	 * 
	 * @return Actual fitness comparator
	 */

	Comparator<IFitness> getComparator();
}
