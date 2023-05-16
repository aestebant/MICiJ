package miclustering.algorithms.evolutionary;

import jclec.algorithm.classic.SGE;
import jclec.problem.classification.multiinstance.MIClassificationMetadata;
import jclec.problem.util.dataset.AbstractDataset;
import jclec.problem.util.dataset.ArffDataSet;
import jclec.problem.util.dataset.IDataset;
import org.apache.commons.configuration.Configuration;

public class MIEvoCluster extends SGE {
    private IDataset dataset;

    @Override
    public void configure(Configuration settings) {
        String pathDataset = settings.getString("evaluator.dataset");
        setDataset(pathDataset);

        int nBags = dataset.numberOfExamples();
        int kmin = settings.getInt("evaluator.kmin");
        int kmax = settings.getInt("evaluator.kmax");
        settings.addProperty("species[@type]", "jclec.intarray.IntArrayIndividualSpecies");
        for (int i = 0; i < nBags; ++i) {
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@type]", "jclec.util.intset.Interval");
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@left]", Integer.toString(kmin));
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@right]", Integer.toString(kmax));
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@closure]", "closed-closed");
        }

        super.configure(settings);
    }

    private void setDataset(String pathToDataset) {
        dataset = new ArffDataSet();
        ((AbstractDataset) dataset).setFileName(pathToDataset);
        dataset.setMetadata(new MIClassificationMetadata());
        ((MIClassificationMetadata) dataset.getMetadata()).setClassIndex(dataset.getMetadata().numberOfAttributes());
        dataset.loadExamples();
    }
}
