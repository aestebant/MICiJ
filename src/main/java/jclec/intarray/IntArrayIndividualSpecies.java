package jclec.intarray;


import jclec.IConfigure;
import jclec.util.intset.IIntegerSet;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationRuntimeException;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.ToStringBuilder;

/**
 * IntArrayIndividual species
 *
 * @author Sebastian Ventura
 */

public class IntArrayIndividualSpecies extends IntArraySpecies implements IConfigure {

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */

    public IntArrayIndividualSpecies() {
        super();
    }

    /**
     * Constructor that sets genotype schema.
     *
     * @param genotypeSchema Genotype schema
     */

    public IntArrayIndividualSpecies(IIntegerSet[] genotypeSchema) {
        super();
        setGenotypeSchema(genotypeSchema);
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // Setting properties

    /**
     * Set genotype schema
     *
     * @param genotypeSchema New genotype schema
     */

    public void setGenotypeSchema(IIntegerSet[] genotypeSchema) {
        this.genotypeSchema = genotypeSchema;
    }

    // BinArrayIndividual factory

    /**
     * {@inheritDoc}
     */

    @Override
    public IntArrayIndividual createIndividual(int[] genotype) {
        return new IntArrayIndividual(genotype);
    }

    // IConfigure interface

    public void configure(Configuration settings) {
        // Genotype lenght
        int genotypeLength = settings.getList("genotype-schema.locus[@type]").size();
        // Genotype schema
        IIntegerSet[] genotypeSchema = new IIntegerSet[genotypeLength];
        // Set genotype schema components
        for (int i = 0; i < genotypeLength; i++) {
            // Get component classname
            String componentClassname = settings.getString("genotype-schema.locus(" + i + ")[@type]");
            try {
                Class<?> componentClass = Class.forName(componentClassname);
                // Set schema component
                genotypeSchema[i] = (IIntegerSet) componentClass.newInstance();
                // Configure component
                if (genotypeSchema[i] instanceof IConfigure) {
                    ((IConfigure) genotypeSchema[i]).configure(settings.subset("genotype-schema.locus(" + i + ")"));
                }
            } catch (ClassNotFoundException e) {
                throw new ConfigurationRuntimeException
                        ("Illegal schema classname");
            } catch (InstantiationException | IllegalAccessException e) {
                throw new ConfigurationRuntimeException
                        ("Problems creating an instance of schema", e);
            }
        }
        // Assign genotype schema
        setGenotypeSchema(genotypeSchema);
    }

    // java.lang.Object methods

    /**
     * {@inheritDoc}
     */

    public String toString() {
        // Performs Schema rendering
        ToStringBuilder tsb = new ToStringBuilder(this);
        // Append schema
        tsb.append("schema", genotypeSchema);
        // Returns rendered schema
        return tsb.toString();
    }

    /**
     * {@inheritDoc}
     */

    public boolean equals(Object other) {
        if (other instanceof IntArrayIndividualSpecies) {
            EqualsBuilder eb = new EqualsBuilder();
            IntArrayIndividualSpecies iaoth = (IntArrayIndividualSpecies) other;
            eb.append(this.genotypeSchema, iaoth.genotypeSchema);
            return eb.isEquals();
        } else {
            return false;
        }
    }
}
