package jclec.problem.classification.multiinstance;

import jclec.problem.classification.ClassificationMetadata;
import jclec.problem.util.dataset.IDataset;
import jclec.problem.util.dataset.IExample;
import jclec.problem.util.dataset.attribute.CategoricalAttribute;
import jclec.problem.util.dataset.attribute.IAttribute;

/**
 * Multi-instance classification metadata
 *
 * @author Alberto Cano
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna
 * @author Juan Luis Olmo
 */

public class MIClassificationMetadata extends ClassificationMetadata {
    /////////////////////////////////////////////////////////////////
    // ------------------------------------------- Internal variables
    /////////////////////////////////////////////////////////////////

    private static final long serialVersionUID = 7370514914850394015L;

    /**
     * Class attribute index
     */

    private int classIndex = -1;

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------------- Constructor
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */

    public MIClassificationMetadata() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    /**
     * Get the class attribute index
     *
     * @return the classIndex
     */
    public int getClassIndex() {
        return classIndex;
    }

    /**
     * Set the class attribute index
     *
     * @param classIndex the classIndex
     */
    public void setClassIndex(int classIndex) {
        this.classIndex = classIndex;
    }

    /**
     * {@inheritDoc}
     */

    public int numberOfAttributes() {
        return attributesList.size() - 1;
    }

    /**
     * {@inheritDoc}
     */

    public int numberOfClasses() {
        CategoricalAttribute catAttr = (CategoricalAttribute) getAttribute(classIndex);
        return catAttr.getCategories().size();
    }

    /**
     * Get the class attribute
     *
     * @return the class attribute
     */
    public IAttribute getClassAttribute() {
        return attributesList.get(classIndex);
    }

    /**
     * {@inheritDoc}
     */
    public boolean isClassAttribute(int attributeIndex) {
        return attributeIndex == classIndex;
    }

    /**
     * Returns the number of bags of the different classes
     *
     * @param dataset the dataset
     * @return array of number of instances per class
     */
    public int[] numberOfExamples(IDataset dataset) {
        int[] numBags = new int[numberOfClasses()];

        for (IExample bag : dataset.getExamples()) {
            numBags[(int) ((MIBag) bag).getClassValue()]++;
        }

        return numBags;
    }

    /**
     * Copy method
     *
     * @return A copy of this metadata
     */

    public MIClassificationMetadata copy() {
        MIClassificationMetadata metadata = new MIClassificationMetadata();

        metadata.attributesList = this.attributesList;
        metadata.attributesMap = this.attributesMap;
        metadata.setClassIndex(classIndex);

        return metadata;
    }
}