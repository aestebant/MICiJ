package jclec.algorithm.classic;

import jclec.IConfigure;
import jclec.IIndividual;
import jclec.ISelector;
import jclec.algorithm.PopulationAlgorithm;
import jclec.base.FilteredMutator;
import jclec.base.FilteredRecombinator;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationRuntimeException;
import org.apache.commons.lang.builder.EqualsBuilder;

/**
 * <strong><u>S</u></strong>imple <strong><u>G</u></strong>enerational algorithm.
 *
 * @author Sebastian Ventura
 */

public class SG extends PopulationAlgorithm {
    /////////////////////////////////////////////////////////////////
    // --------------------------------------------------- Properties
    /////////////////////////////////////////////////////////////////

    /**
     * Parents selector
     */
    protected ISelector parentsSelector;

    /**
     * Individuals mutator
     */
    protected FilteredMutator mutator;

    /**
     * Individuals recombinator
     */
    protected FilteredRecombinator recombinator;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty (default) constructor
     */
    public SG() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // Getting and setting properties

    /**
     * Access to parents selector
     *
     * @return Parents selector
     */
    public ISelector getParentsSelector() {
        return parentsSelector;
    }

    /**
     * Sets the parents selector.
     *
     * @param parentsSelector New parents selector
     */
    public void setParentsSelector(ISelector parentsSelector) {
        // Sets the parents selector
        this.parentsSelector = parentsSelector;
        // Contextualize parents selector
        parentsSelector.contextualize(this);
    }

    // Generation plan

    /**
     * Access to parents recombinator
     *
     * @return Actual parents recombinator
     */
    public FilteredRecombinator getRecombinator() {
        return recombinator;
    }

    /**
     * Access to individuals mutator.
     *
     * @return Individuals mutator
     */
    public FilteredMutator getMutator() {
        return mutator;
    }

    // IConfigure interface

    /**
     * Configuration method.
     * <p>
     * Configuration parameters for BaseAlgorithm class are:
     *
     * <ul>
     * <li>
     * <code>species: ISpecies (complex)</code></p>
     * Individual species
     * </li><li>
     * <code>evaluator: IEvaluator (complex)</code></p>
     * Individuals evaluator
     * </li><li>
     * <code>population-size: int</code></p>
     * Population size
     * </li><li>
     * <code>max-of-generations: int</code></p>
     * Maximum number of generations
     * </li>
     * <li>
     * <code>provider: IProvider (complex)</code></p>
     * Individuals provider
     * </li>
     * <li>
     * <code>parents-selector: ISelector (complex)</code>
     * </li>
     * <li>
     * <code>recombinator: (complex)</code>
     * <ul>
     * <li>
     * <code>recombinator.decorated: IRecombinator (complex)</code></p>
     * Recombination operator
     * </li><li>
     * <code>recombinator.recombination-prob double</code></p>
     * Recombination probability
     * </li>
     * </ul>
     * </li>
     * <li>
     * <code>mutator: (complex)</code>
     * <ul>
     * <li>
     * <code>mutator.decorated: IMutator (complex) </code></p>
     * Mutation operator
     * </li><li>
     * <code>mutator.mutation-prob double</code></p>
     * Mutation probability
     * </li>
     * </ul>
     * </li>
     * </ul>
     */

    @SuppressWarnings("unchecked")
    public void configure(Configuration configuration) {
        // Call super.configure() method
        super.configure(configuration);
        // Parents selector
        try {
            // Selector classname
            String parentsSelectorClassname = configuration.getString("parents-selector[@type]");
            // Species class
            Class<? extends ISelector> parentsSelectorClass = (Class<? extends ISelector>) Class.forName(parentsSelectorClassname);
            // Species instance
            ISelector parentsSelector = parentsSelectorClass.newInstance();
            // Configure species if necessary
            if (parentsSelector instanceof IConfigure) {
                // Extract species configuration
                Configuration parentsSelectorConfiguration = configuration.subset("parents-selector");
                // Configure species
                ((IConfigure) parentsSelector).configure(parentsSelectorConfiguration);
            }
            // Set species
            setParentsSelector(parentsSelector);
        } catch (ClassNotFoundException e) {
            throw new ConfigurationRuntimeException("Illegal parents selector classname");
        } catch (InstantiationException | IllegalAccessException e) {
            throw new ConfigurationRuntimeException("Problems creating an instance of parents selector", e);
        }
        // Recombinator
        if (configuration.containsKey("recombinator.decorated[@type]")) {
            recombinator = new FilteredRecombinator(this);
            recombinator.configure(configuration.subset("recombinator"));
        }
        // Mutator
        if (configuration.containsKey("mutator.decorated[@type]")) {
            mutator = new FilteredMutator(this);
            mutator.configure(configuration.subset("mutator"));
        }
    }

    // java.lang.Object methods

    @Override
    public boolean equals(Object other) {
        if (other instanceof SG) {
            SG cother = (SG) other;
            EqualsBuilder eb = new EqualsBuilder();
            // Call super method
            eb.appendSuper(super.equals(other));
            // Parents selector
            eb.append(parentsSelector, cother.parentsSelector);
            // Mutator
            eb.append(mutator, cother.mutator);
            // Recombinator
            eb.append(recombinator, cother.recombinator);
            // Return test result
            return eb.isEquals();
        } else {
            return false;
        }
    }


    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    @Override
    protected void doSelection() {
        pset = parentsSelector.select(bset);
    }

    @Override
    protected void doGeneration() {
        // Recombine parents
        cset = recombinator.recombine(pset);
        // Add non-recombined inds.
        // These individuals are references to existent individuals
        // (elements of bset) so we make a copy of them
        for (IIndividual ind : recombinator.getSterile())
            cset.add(ind.copy());
        // Mutate filtered inds
        cset = mutator.mutate(cset);
        // Add non-mutated inds.
        // These individuals don't have to be copied, because there are original individuals (not references)
        cset.addAll(mutator.getSterile());
        // Evaluate all new individuals
        evaluator.evaluate(cset);
    }

    @Override
    protected void doReplacement() {
        rset = bset;
    }

    @Override
    protected void doUpdate() {
        // Sets new bset
        bset = cset;
        // Clear pset, rset & cset
        pset = null;
        rset = null;
        cset = null;
    }
}
