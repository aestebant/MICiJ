package miclustering.algorithms.evolutionary;

import jclec.IConfigure;
import jclec.algorithm.classic.SGE;
import jclec.problem.classification.multiinstance.MIClassificationMetadata;
import jclec.problem.util.dataset.AbstractDataset;
import jclec.problem.util.dataset.ArffDataSet;
import jclec.problem.util.dataset.IDataset;
import org.apache.commons.configuration.Configuration;

public class MIGGA extends SGE implements IConfigure {

    protected IDataset dataset;

    @Override
    public void configure(Configuration settings) {
        String pathDataset = settings.getString("evaluator.dataset");
        setDataset(pathDataset);

        int nBags = dataset.numberOfExamples();
        int maxK = settings.getInt("evaluator.max-of-clusters");

        settings.addProperty("provider[@type]", "miclustering.algorithms.evolutionary.migga.MIGGAArrayCreator");
        settings.addProperty("species[@type]", "jclec.intarray.IntArrayIndividualSpecies");
        for (int i = 0; i < nBags; ++i) {
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@type]", "jclec.util.intset.Interval");
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@left]", "0");
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@right]", Integer.toString(maxK));
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@closure]", "closed-closed");
        }

        settings.addProperty("species.genotype-schema.locus(" + nBags + ")[@type]", "jclec.util.intset.Interval");
        settings.addProperty("species.genotype-schema.locus(" + nBags + ")[@left]", "1");
        settings.addProperty("species.genotype-schema.locus(" + nBags + ")[@right]", Integer.toString(maxK));
        settings.addProperty("species.genotype-schema.locus(" + nBags + ")[@closure]", "closed-closed");

        super.configure(settings);
    }

    @Override
    protected void doGeneration() {
        // Mutate filtered inds
        cset = pset;
        // Evaluate all new individuals
        evaluator.evaluate(cset);
    }

    protected void setDataset(String pathToDataset) {
        dataset = new ArffDataSet();
        ((AbstractDataset) dataset).setFileName(pathToDataset);
        dataset.setMetadata(new MIClassificationMetadata());
        // numOfAttributes devuelve el número de atributos de instancia -> coincide con el valor de índice de la clase
        ((MIClassificationMetadata) dataset.getMetadata()).setClassIndex(dataset.getMetadata().numberOfAttributes());
        dataset.loadExamples();
    }
}

