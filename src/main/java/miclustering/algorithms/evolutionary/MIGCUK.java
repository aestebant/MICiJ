package miclustering.algorithms.evolutionary;

import jclec.algorithm.classic.SGE;
import jclec.problem.classification.multiinstance.MIClassificationMetadata;
import jclec.problem.util.dataset.AbstractDataset;
import jclec.problem.util.dataset.ArffDataSet;
import jclec.problem.util.dataset.IDataset;
import org.apache.commons.configuration.Configuration;

public class MIGCUK extends SGE {
    protected IDataset dataset;

    @Override
    public void configure(Configuration configuration) {
        String pathDataset = configuration.getString("evaluator.dataset");
        setDataset(pathDataset);
        int kmin = configuration.getInt("evaluator.kmin", 2);
        int kmax = configuration.getInt("evaluator.kmax", 2);
        int nBags = dataset.numberOfExamples();
        configuration.addProperty("species[@type]", "jclec.intarray.IntArrayIndividualSpecies");
        for (int i = 0; i <= kmax - kmin + 1; ++i) {
            configuration.addProperty("species.genotype-schema.locus(" + i + ")[@type]", "jclec.util.intset.Interval");
            configuration.addProperty("species.genotype-schema.locus(" + i + ")[@left]", "0");
            configuration.addProperty("species.genotype-schema.locus(" + i + ")[@right]", Integer.toString(nBags - 1));
            configuration.addProperty("species.genotype-schema.locus(" + i + ")[@closure]", "closed-closed");
        }

        super.configure(configuration);
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
