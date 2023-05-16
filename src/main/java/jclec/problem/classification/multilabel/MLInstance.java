package jclec.problem.classification.multilabel;

import jclec.problem.util.dataset.Instance;

/**
 * Dataset instance for multi-label classification
 * 
 * @author Alberto Cano 
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna 
 * @author Juan Luis Olmo
 */

public class MLInstance extends Instance
{
	/////////////////////////////////////////////////////////////
	// ----------------------------------------------- Properties
	/////////////////////////////////////////////////////////////

	private static final long serialVersionUID = 2141837612603678135L;

	/** Class values */
	
	private double[] classValues;

	/////////////////////////////////////////////////////////////
	// --------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////

	/**
	 * Default constructor
	 * 
	 * @param numberAttributes Number of attributes
	 */

	public MLInstance(int numberAttributes)
	{
		super(numberAttributes);
	}

	/////////////////////////////////////////////////////////////
	// ------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////
	
	/**
     * Get the classes of the instance
     *
     * @return the class values
     */

	public double[] getClassValues() 
	{
		return classValues;
	}

	/**
     * Set the classes of the instance
     *
     * @param classValues array of class values
     */

	public void setClassValues(double[] classValues) {
		this.classValues = classValues;
	}
	
	/**
    * Checks if the example is equal to another
    * 
    * @return true or false
    */
    @Override
    public boolean equals(Object other)
    {
    	if (other instanceof MLInstance) {
    		MLInstance cother = (MLInstance) other;
    		
    		if(super.equals(other) == false) return false;
    		if(classValues.length != cother.getClassValues().length) return false;
    		
    		for(int i = 0; i < classValues.length; i++)
    			if(classValues[i] != cother.getClassValues()[i])
    				return false;
    		
    		return true;
    		
		} else {
			return false;
		}
    }

	/**
	 * Copy method
	 * 
	 * @return A copy of this instance
	 */

	public MLInstance copy() 
	{
		MLInstance instance = new MLInstance(values.length);

		for(int i=0; i<values.length; i++)
			instance.setValue(i,values[i]);
		
		double[] classValues = new double[this.classValues.length];
		
		for(int i=0; i<this.classValues.length; i++)
			classValues[i] = this.classValues[i];
		
		instance.setClassValues(classValues);
		
		return instance;
	}
}