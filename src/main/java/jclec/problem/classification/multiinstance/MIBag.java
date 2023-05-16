package jclec.problem.classification.multiinstance;

import jclec.problem.util.dataset.IExample;
import jclec.problem.util.dataset.Instance;

import java.util.ArrayList;

/**
 * Bag implementation for multi-instance classification
 *
 * @author Alberto Cano
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna
 * @author Juan Luis Olmo
 */

public class MIBag implements IExample {
    /////////////////////////////////////////////////////////////
    // ----------------------------------------------- Properties
    /////////////////////////////////////////////////////////////

    private static final long serialVersionUID = 2141837612603678135L;

    /**
     * Bag internal identification
     */

    private double bagID;

    /**
     * List of instances
     */

    private ArrayList<Instance> instances = new ArrayList<>();

    /**
     * Class value
     */

    private double classValue = -1;

    /**
     * Weight of this instance
     */

    private double weight;

    /////////////////////////////////////////////////////////////
    // --------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////

    /**
     * Default constructor
     */

    public MIBag() {
        super();
    }

    /////////////////////////////////////////////////////////////
    // ------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////

    /**
     * Returns the weight of this instance.
     *
     * @return instance weight
     */

    public double getWeight() {
        return weight;
    }

    /**
     * Sets the weigth for this instance
     *
     * @param weight New weigth value
     */

    public final void setWeight(double weight) {
        this.weight = weight;
    }

    /**
     * Get the class of the instance
     *
     * @return the class value
     */

    public double getClassValue() {
        return classValue;
    }

    /**
     * Set the class of the instance
     *
     * @param classValue the class
     */

    public void setClassValue(double classValue) {
        this.classValue = classValue;
    }

    /**
     * Get the bag instances
     *
     * @return the instances
     */
    public ArrayList<Instance> getInstances() {
        return instances;
    }

    /**
     * Set the bag instances
     *
     * @param instances the instances to set
     */
    public void setInstances(ArrayList<Instance> instances) {
        this.instances = instances;
    }

    /**
     * Add an instance to the bag
     *
     * @param instance the instance
     */
    public void addInstance(Instance instance) {
        this.instances.add(instance);
    }

    /**
     * Get the bag identification
     *
     * @return the bagID
     */
    public double getBagID() {
        return bagID;
    }

    /**
     * Set the bag identification
     *
     * @param bagID the bag identification
     */
    public void setBagID(double bagID) {
        this.bagID = bagID;
    }

    /**
     * Checks if the example is equal to another
     *
     * @return true or false
     */
    @Override
    public boolean equals(Object other) {
        if (other instanceof MIBag) {
            MIBag cother = (MIBag) other;

            if (bagID != cother.getBagID()) return false;
            if (classValue != cother.getClassValue()) return false;
            if (instances.size() != cother.getInstances().size()) return false;

            if (cother.getInstances().containsAll(instances) && instances.containsAll(cother.getInstances()))
                return true;
            else return false;

        } else {
            return false;
        }
    }

    /**
     * Copy method
     *
     * @return A copy of this instance
     */
    @Override
    public MIBag copy() {
        MIBag bag = new MIBag();

        bag.setBagID(bagID);

        bag.setClassValue(classValue);

        ArrayList<Instance> instances = new ArrayList<Instance>();

        for (Instance instance : this.instances)
            instances.add(instance.copy());

        bag.setInstances(instances);

        return bag;
    }

    /**
     * This method should not be used when multi-instance
     *
     * @param attributeIndex the index of the attribute
     * @return -1
     */
    @Deprecated
    public double getValue(int attributeIndex) {
        return -1;
    }
}