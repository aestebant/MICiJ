package jclec.problem.classification.multilabel;

import jclec.problem.classification.IClassifier;
import jclec.problem.util.dataset.IDataset;

/**
 * Interface for multi-label classifiers
 *
 * @author Alberto Cano
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna
 * @author Juan Luis Olmo
 * @author Jose Luis Avila
 */

public interface IMLClassifier extends IClassifier {
    /**
     * Dataset classification
     * <p>
     * Classify all the instances contained in this dataset
     *
     * @param dataset Dataset which instances will be classified
     * @return Array of Arrays of class labels
     */

    public double[][] classify(IDataset dataset);

    /**
     * Copy method
     *
     * @return a copy of the classifier
     */

    public IMLClassifier copy();
}