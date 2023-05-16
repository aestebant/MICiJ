package jclec.problem.classification;

import jclec.JCLEC;
import jclec.problem.util.dataset.IMetadata;

/**
 * Interface for classifiers
 *
 * @author Alberto Cano
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna
 * @author Juan Luis Olmo
 */

public interface IClassifier extends JCLEC {
    /**
     * Shows the classifier
     *
     * @param metadata Metadata to show the attribute and class names
     * @return the classifier
     */
    String toString(IMetadata metadata);

    /**
     * Copy method
     *
     * @return a copy of the classifier
     */
    IClassifier copy();
}