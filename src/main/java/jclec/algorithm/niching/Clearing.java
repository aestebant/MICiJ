package jclec.algorithm.niching;

import jclec.IIndividual;
import jclec.fitness.IValueFitness;
import jclec.fitness.SimpleValueFitness;
import org.apache.commons.configuration.Configuration;

import java.util.List;

/**
 * Fitness clearing algorithm.
 * <p>
 * NOTE: This algorithm assumes that the optimal points to search are maximum points.
 * * Esto se va a arreglar añadiendo el "clearValue", que establece a qué valor limpiar el fitness, dejando de ser 0 por defecto
 *
 * @author Sebastian Ventura
 * @author Amelia Zafra
 */

public class Clearing extends SpatialNiching {

    /////////////////////////////////////////////////////////////////
    // --------------------------------------------------- Properties
    /////////////////////////////////////////////////////////////////

    /**
     * Number of individuals in each sub-population
     */
    private int kappa;
    private double clearValue = 5.0;
    private boolean maximization = false;


    /////////////////////////////////////////////////////////////////
    // ------------------------------------------- Internal variables
    /////////////////////////////////////////////////////////////////

    /**
     * Niche radius
     */
    private transient double deltaShare = 0.0;

    /**
     * Sharing distance
     */
    private transient double[][] dValues;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */
    public Clearing() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // Setting and getting properties
    private void setKappa(int kappa) {
        this.kappa = kappa;
    }

    public int getKappa() {
        return this.kappa;
    }

    // IConfigure interface

    /**
     * {@inheritDoc}
     * <p>
     * Configuration parameters for this algorithm are:
     */

    @Override
    public void configure(Configuration configuration) {
        // Call super.configure method
        super.configure(configuration);
        // Set kappa value
        int kappa = configuration.getInt("kappa");
        setKappa(kappa);
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    /**
     *
     */

    @Override
    protected void createNiches() {
        // Get all bset individual in sorted form and put them in nset
        int bsetSize = bset.size();
        List<IIndividual> aux = bettersSelector.select(bset, bsetSize);
        for (IIndividual ind : aux)
            nset.add(ind.copy());

        ////////////////////
        // Clearing process
        ////////////////////

        // Calculate dValues
        calculateDValues();
        // Calculate delta share
        calculateDeltaShare();

        for (int i = 0; i < bsetSize; i++) {
            // Current individual
            IIndividual ind_i = nset.get(i);
            // If fitness has not been cleared
            if (((IValueFitness) ind_i.getFitness()).getValue() > clearValue) {
                // Number of individuals per niche
                int numberOfWinners = 1;
                for (int j = i + 1; j < bsetSize; j++) {
                    // Other individual
                    IIndividual ind_j = nset.get(j);
                    // If individual fitness has not been cleared and
                    // individual belongs to current niche
                    if (maximization) {
                        if ((((IValueFitness) ind_i.getFitness()).getValue() > clearValue) && (dValues[i][j] < deltaShare)) {
                            if (numberOfWinners < kappa) {
                                numberOfWinners++;
                            } else {
                                SimpleValueFitness newFitness = new SimpleValueFitness(clearValue);
                                ind_j.setFitness(newFitness);
                            }
                        }
                    }
                    else {
                        if ((((IValueFitness) ind_i.getFitness()).getValue() < clearValue) && (dValues[i][j] < deltaShare)) {
                            if (numberOfWinners < kappa) {
                                numberOfWinners++;
                            } else {
                                SimpleValueFitness newFitness = new SimpleValueFitness(clearValue);
                                ind_j.setFitness(newFitness);
                            }
                        }
                    }
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////
    // ---------------------------------------------- Private methods
    /////////////////////////////////////////////////////////////////

    /**
     * Calculate distances between individuals in bset
     */
    private void calculateDValues() {
        // Allocate space for sh_dij
        int dijSize = bset.size();
        dValues = new double[dijSize][];
        for (int i = 0; i < dijSize; i++)
            dValues[i] = new double[dijSize];
        //Calculate sh_dij values
        for (int i = 0; i < dijSize; i++) {
            // Diagonal value is zero
            dValues[i][i] = 0.0;
            // Values over and under diagonal
            for (int j = i + 1; j < dijSize; j++)
                dValues[i][j] = dValues[j][i] = distance.distance(bset.get(i), bset.get(j));
        }
    }

    /**
     * Calculate the niche radius (delta_share)
     * <p>
     * The established niche radius consists of considering within a
     * niche all individuals with a distance smaller than 20% of the
     * maximum distance between all individuals.
     */
    private void calculateDeltaShare() {
        double maximumDistance = Double.MIN_VALUE;
        for (int i = 0; i < dValues.length; i++) {
            for (int j = 0; j < dValues[i].length; j++) {
                if (dValues[i][j] > maximumDistance) {
                    maximumDistance = dValues[i][j];
                }
            }
        }
        // Set niche radius value
        deltaShare = 0.2 * maximumDistance;
    }
}
