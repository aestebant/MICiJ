package jclec.problem.classification.multilabel;

import jclec.problem.classification.ClassificationMetadata;
import jclec.problem.util.dataset.IDataset;
import jclec.problem.util.dataset.IExample;
import jclec.problem.util.dataset.attribute.IAttribute;

/**
 * Multi-label classification metadata
 *  
 * @author Alberto Cano 
 * @author Amelia Zafra
 * @author Sebastian Ventura
 * @author Jose M. Luna 
 * @author Juan Luis Olmo
 * @author Jose Luis Avila
 */

public class MLClassificationMetadata extends ClassificationMetadata
{
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------- Internal variables
	/////////////////////////////////////////////////////////////////
	
	private static final long serialVersionUID = 7370514914850394015L;
	
	/** Number of classes */
	
	private int numberClasses = -1;

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    
	public MLClassificationMetadata(int numberClasses)
	{
		super();
		
		this.numberClasses = numberClasses;
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	/**
	 * {@inheritDoc}
	 */
	
	public int numberOfAttributes() 
	{
		return attributesList.size() - numberClasses;
	}

	/**
	 * {@inheritDoc}
	 */
	
	public int numberOfClasses() 
	{
		return numberClasses;
	}
	
	/**
	 * Set the number of classes
	 * 
	 * @param numberClasses The number of classes
	 */
	
	public void setNumberClasses(int numberClasses)
	{
		this.numberClasses = numberClasses;
	}
	
	/**
	 * {@inheritDoc}
	 */
    public boolean isClassAttribute( int attributeIndex )
    {
    	if (attributeIndex >= numberOfAttributes())
    		return true;
    	else
    		return false;
    }
    
    /**
     * Get the class attribute
     * 
     * @param Class the class attribute index
     * 
     * @return the class attribute
     */
    public IAttribute getClassAttribute(int Class)
    {
    	return attributesList.get(numberOfAttributes() + Class);
    }
    
	/**
	 * Returns the number of instances of the different classes
	 * 
	 * @param dataset the dataset
	 * @return array of number of instances per class
	 */
	@Override
	public int[] numberOfExamples(IDataset dataset)
	{
		int[] numInstances = new int[numberClasses];
		
		for(IExample instance : dataset.getExamples())
		{
			for(int i = 0; i < numberClasses; i++)
				if(((MLInstance) instance).getClassValues()[i] == 1)
					numInstances[i]++;
		}
		
		return numInstances;
	}
	
	/**
	 * Copy method
	 * 
	 * @return A copy of this metadata
	 */
	
	public MLClassificationMetadata copy()
	{
		MLClassificationMetadata metadata = new MLClassificationMetadata(numberClasses);
		
		metadata.attributesList = this.attributesList;
		metadata.attributesMap = this.attributesMap;
		
		return metadata;
	}
}