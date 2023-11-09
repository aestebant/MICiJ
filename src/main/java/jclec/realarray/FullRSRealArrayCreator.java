package jclec.realarray;

import jclec.ISpecies;
import jclec.base.AbstractCreator;
import jclec.util.range.IRange;

/**
 * Creation of RealArrayIndividual (and subclasses).
 * 
 * @author Sebastian Ventura
 */

public class FullRSRealArrayCreator extends AbstractCreator {
	/////////////////////////////////////////////////////////////////
	// --------------------------------------- Serialization constant
	/////////////////////////////////////////////////////////////////

	/** Generated by Eclipse */

	private static final long serialVersionUID = -2638928425169895614L;

	/////////////////////////////////////////////////////////////////
	// ------------------------------------------- Internal variables
	/////////////////////////////////////////////////////////////////

	/** Associated species */

	protected transient RealArraySpecies species;

	/** Genotype schema */

	protected transient IRange[] schema;
	
	private final int studentSimsIdx = 10;
	private final int subjectSimsIdx = 12;
	private final int studentSimsOpc = 5;
	private final int subjectSimsOpc = 2;

	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	public FullRSRealArrayCreator() {
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////

	// java.lang.Object methods

	/**
	 * {@inheritDoc}
	 */

	@Override
	public boolean equals(Object other) {
		if (other instanceof FullRSRealArrayCreator) {
			return true;
		} else {
			return false;
		}
	}

	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Protected methods
	/////////////////////////////////////////////////////////////////

	// AbstractCreator methods

	@Override
	protected void prepareCreation() {
		// Get context species
		ISpecies spc = context.getSpecies();
		if (spc instanceof RealArraySpecies) {
			this.species = (RealArraySpecies) spc;
		} else {
			throw new IllegalArgumentException("RealArraySpecies expected");
		}
		// Get individuals schema
		schema = species.getGenotypeSchema();
	}

	@Override
	protected void createNext() {
		createdBuffer.add(species.createIndividual(createGenotype()));
	}

	/////////////////////////////////////////////////////////////////
	// ---------------------------------------------- Private methods
	/////////////////////////////////////////////////////////////////

	/**
	 * Create a double [] genotype, filling it randomly
	 */

	private final double[] createGenotype() {
		int gl = schema.length;
		double[] result = new double[gl];
		for (int i = 0; i < studentSimsIdx; i++)
			result[i] = schema[i].getRandom(randgen);
		
		for (int i = studentSimsIdx; i < subjectSimsIdx; ++i)
			result[i] = randgen.choose(0, studentSimsOpc);
		
		for (int i = subjectSimsIdx; i < gl; ++i)
			result[i] = randgen.choose(0, subjectSimsOpc);
		
		return result;
	}	
}