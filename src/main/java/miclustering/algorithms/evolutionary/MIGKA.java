package miclustering.algorithms.evolutionary;

import jclec.IConfigure;
import jclec.algorithm.classic.SGE;
import jclec.base.FilteredMutator;
import jclec.problem.classification.multiinstance.MIClassificationMetadata;
import jclec.problem.util.dataset.AbstractDataset;
import jclec.problem.util.dataset.ArffDataSet;
import jclec.problem.util.dataset.IDataset;
import miclustering.algorithms.evolutionary.utils.ClusteringEvaluator;
import miclustering.algorithms.evolutionary.utils.ClusteringMutator;
import org.apache.commons.configuration.Configuration;

public class MIGKA extends SGE implements IConfigure {

    protected IDataset dataset;
    protected FilteredMutator kmo;

    @Override
    public void configure(Configuration settings) {
        String pathDataset = settings.getString("evaluator.dataset");
        setDataset(pathDataset);

        int k = settings.getInt("evaluator.num-clusters");

        int nBags = dataset.numberOfExamples();

        settings.addProperty("species[@type]", "jclec.intarray.IntArrayIndividualSpecies");
        for (int i = 0; i < nBags; ++i) {
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@type]", "jclec.util.intset.Interval");
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@left]", "0");
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@right]", Integer.toString(k -1));
            settings.addProperty("species.genotype-schema.locus(" + i + ")[@closure]", "closed-closed");
        }

        super.configure(settings);

        if (mutator.getDecorated() instanceof ClusteringMutator)
            ((ClusteringMutator) mutator.getDecorated()).setEvaluator((ClusteringEvaluator) evaluator);

        String distFunc = settings.getString("evaluator.distance[@type]");
        String distConf = settings.getString("evaluator.distance.config");
        KMeansOperator kmo = new KMeansOperator(pathDataset, distFunc, distConf, k);
        this.kmo = new FilteredMutator(this);
        this.kmo.setMutProb(settings.getDouble("kmeans-operator[@mut-prob]"));
        this.kmo.setDecorated(kmo);
    }

    @Override
    protected void doGeneration() {
        // Mutate filtered inds
        cset = mutator.mutate(pset);
        // Add non-mutated inds.
        // These individuals don't have to be copied, because there are original individuals (not references)
        cset.addAll(mutator.getSterile());
        cset = kmo.mutate(cset);
        cset.addAll(kmo.getSterile());

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

    public FilteredMutator getKmo() {
        return kmo;
    }
}
