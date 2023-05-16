package jclec.intarray.rec;

import jclec.IIndividual;
import jclec.intarray.IntArrayIndividual;

import java.util.ArrayList;
import java.util.List;

/**
 * One point crossover operator for IntArrayIndividual and its subclasses.
 *
 * @author Sebastian Ventura
 */

public class OnePointCrossover extends IntArrayRecombinator {

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    public OnePointCrossover() {
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
        return other instanceof OnePointCrossover;
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    // AbstractRecombinator methods

    /**
     * {@inheritDoc}
     */
    @Override
    protected void recombineNext() {
        List<IIndividual> parents = new ArrayList<>(2);
        parents.add(parentsBuffer.get(parentsCounter));
        parents.add(parentsBuffer.get(parentsCounter + 1));
        sonsBuffer.addAll(recombineInd(parents));
    }

    @Override
    protected List<IIndividual> recombineInd(List<IIndividual> individuals) {
        // Genotype length
        int gl = species.getGenotypeLength();
        // Parents genotypes
        int[] p0_genome = ((IntArrayIndividual) individuals.get(0)).getGenotype();
        int[] p1_genome = ((IntArrayIndividual) individuals.get(1)).getGenotype();
        // Creating sons genotypes
        int[] s0_genome = new int[gl];
        int[] s1_genome = new int[gl];
        // Sets a crossover point
        int cp = randgen.choose(1, gl);
        // First son' genotype
        System.arraycopy(p0_genome, 0, s0_genome, 0, cp);
        System.arraycopy(p1_genome, cp, s0_genome, cp, gl - cp);
        // Second son' genotype
        System.arraycopy(p1_genome, 0, s1_genome, 0, cp);
        System.arraycopy(p0_genome, cp, s1_genome, cp, gl - cp);
        // Put sons in s
        List<IIndividual> sons = new ArrayList<>(2);
        sons.add(species.createIndividual(s0_genome));
        sons.add(species.createIndividual(s1_genome));
        return sons;
    }
}
