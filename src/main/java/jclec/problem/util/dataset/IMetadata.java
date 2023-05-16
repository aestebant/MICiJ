package jclec.problem.util.dataset;

import jclec.JCLEC;
import jclec.problem.util.dataset.attribute.IAttribute;

/**
 * Dataset specification
 *
 * @author Alberto Cano
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna
 * @author Juan Luis Olmo
 */

public interface IMetadata extends JCLEC {
    /**
     * Returns number of mining attributes in mining data specification.
     *
     * @return number of mining attributes
     */

    int numberOfAttributes();

    /**
     * Returns number of classes.
     *
     * @return number of classes
     */

    int numberOfClasses();

    /**
     * Get index of given attribute in this specification.
     *
     * @param attribute The attribute
     * @return index of attribute, -1 if attribute is not found
     */

    int getIndex(IAttribute attribute);

    /**
     * Get index of given attribute in this specification.
     *
     * @param attributeName Attribute name
     * @return index of attribute, -1 if attribute is not found
     */

    int getAttributeIndex(String attributeName);

    /**
     * Get mining attribute by name.
     *
     * @param attributeName name of attribute required
     * @return specified mining attribute, null if not found
     */

    IAttribute getAttribute(String attributeName);

    /**
     * Get mining attribute by index of the array of attributes of
     * mining data specification.
     *
     * @param attributeIndex index of attribute required
     * @return specified mining attribute, null if not found
     */

    IAttribute getAttribute(int attributeIndex);

    /**
     * Adds an attribute to this metadata.
     * <p>
     * If the name of the new attribute is empty or there already
     * exists an attribute with the same name, it is not added to
     * the name hashtable.
     * This means that it could not be retrieved via its name. It is
     * highly recommended only to use attributes with unique names.
     *
     * @param attribute the attribute to add
     * @return true attribute also added to name hashtable, false if attribute
     * name is null or there already exists an attribute with the same name
     */

    boolean addAttribute(IAttribute attribute);

    /**
     * Checks if the indexed attribute correspond to a class attribute
     *
     * @param attributeIndex index of attribute
     * @return true or false
     */

    boolean isClassAttribute(int attributeIndex);

    /**
     * Returns the number of examples of the different classes
     *
     * @param dataset the dataset
     * @return array of number of examples per class
     */
    int[] numberOfExamples(IDataset dataset);

    /**
     * Copy method
     *
     * @return A copy of this metadata
     */

    IMetadata copy();
}
