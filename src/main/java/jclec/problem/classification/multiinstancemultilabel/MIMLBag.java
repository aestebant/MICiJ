package jclec.problem.classification.multiinstancemultilabel;

import jclec.problem.util.dataset.IExample;
import jclec.problem.util.dataset.Instance;

import java.util.ArrayList;

/**
 * Bag implementation for multi-instance multi-label classification
 * 
 * @author Alberto Cano 
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna 
 * @author Juan Luis Olmo
 */

public class MIMLBag implements IExample
{
	/////////////////////////////////////////////////////////////
	// ----------------------------------------------- Properties
	/////////////////////////////////////////////////////////////

	private static final long serialVersionUID = 2141837612603678135L;

	/** Bag internal identification */
	
	private double bagID;
	
	/** List of instances */
	
	private ArrayList<Instance> instances = new ArrayList<Instance>();
	
	/** Class values */
	
	private double[] classValues;

	/** Weight of this bag */

	private double weight;

	/////////////////////////////////////////////////////////////
	// --------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////

	/**
	 * Default constructor
	 */

	public MIMLBag()
	{
		super();
	}

	/////////////////////////////////////////////////////////////
	// ------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////
	
	/**
     * Returns the weight of this bag.
     *
     * @return bag weight
     */

	public double getWeight() 
	{
		return weight;
	}

	/**
	 * Sets the weigth for this bag
	 * 
	 * @param weight New weigth value
	 */

	public final void setWeight(double weight)
	{
		this.weight = weight;
	}

	/**
     * Get the classes of the bag
     *
     * @return the class values
     */

	public double[] getClassValues() 
	{
		return classValues;
	}

	/**
     * Set the classes of the bag
     *
     * @param classValues the class values
     */

	public void setClassValues(double[] classValues) {
		this.classValues = classValues;
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
	public void addInstance(Instance instance)
	{
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
    public boolean equals(Object other)
    {
    	if (other instanceof MIMLBag) {
    		MIMLBag cother = (MIMLBag) other;
    		
    		if(bagID != cother.getBagID()) return false;   
    		if(instances.size() != cother.getInstances().size()) return false;
    		if(classValues.length != cother.getClassValues().length) return false;
    		
    		for(int i = 0; i < classValues.length; i++)
    			if(classValues[i] != cother.getClassValues()[i])
    				return false;
    		
    		if(cother.getInstances().containsAll(instances) && instances.containsAll(cother.getInstances())) return true;
    		else return false;
    		
		} else {
			return false;
		}
    }

	/**
	 * Copy method
	 * 
	 * @return A copy of this bag
	 */

	public MIMLBag copy() 
	{
		MIMLBag bag = new MIMLBag();
		
		bag.setBagID(bagID);
		
		double[] classValues = new double[this.classValues.length];
		
		for(int i=0; i<this.classValues.length; i++)
			classValues[i] = this.classValues[i];
		
		bag.setClassValues(classValues);
		
		ArrayList<Instance> instances = new ArrayList<Instance>();

		for(Instance instance : this.instances)
			instances.add(instance.copy());
				
		bag.setInstances(instances);
		
		return bag;
	}
	
	/**
	 * This method should not be used when multi-instance multi-label
	 * 
	 * @param attributeIndex the index of the attribute
	 * @return -1
	 */
	@Deprecated
	public double getValue(int attributeIndex) {
		return -1;
	}
}