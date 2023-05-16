package jclec.problem.classification.multiinstance;

import jclec.problem.classification.IClassifier;
import jclec.problem.util.dataset.IDataset;
import jclec.problem.util.dataset.IExample;

/**
 * Interface for multi-instance classifiers
 *  
 * @author Alberto Cano 
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna 
 * @author Juan Luis Olmo
 */

public interface IMIClassifier extends IClassifier
{ 
	/**
	 * Bag classification
	 * 
	 * @param bag Bag to classify
	 * 
	 * @return A double that represents the class label for this bag
	 */
	
	public double classify(IExample bag);
	
	/**
	 * Dataset classification
	 * 
	 * Classify all the bags contained in this dataset
	 * 
	 * @param dataset Dataset which bags will be classified
	 * 
	 * @return Array of class labels
	 */
	
	public double[] classify(IDataset dataset);
	
	/**
	 * Obtains the confusion matrix for a dataset
	 * 
	 * @param dataset the dataset to classify
	 * 
	 * @return the confusion matrix
	 */
	
	public int[][] getConfusionMatrix(IDataset dataset);

	/**
	 * Copy method
	 * 
	 * @return a copy of the classifier
	 */
	
	public IMIClassifier copy();
}