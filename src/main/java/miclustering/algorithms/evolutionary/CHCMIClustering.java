package miclustering.algorithms.evolutionary;

import jclec.algorithm.classic.CHC;
import jclec.base.FilteredMutator;
import jclec.problem.classification.multiinstance.MIClassificationMetadata;
import jclec.problem.util.dataset.AbstractDataset;
import jclec.problem.util.dataset.ArffDataSet;
import jclec.problem.util.dataset.IDataset;
import miclustering.algorithms.evolutionary.utils.ClusteringEvaluator;
import miclustering.algorithms.evolutionary.utils.ClusteringMutator;
import org.apache.commons.configuration.Configuration;

public class CHCMIClustering extends CHC {
    private IDataset dataset;
    private FilteredMutator kmo;
    private FilteredMutator cm;

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

        ClusteringMutator cm = new ClusteringMutator();
        cm.configure(settings.subset("cluster-mutator"));
        cm.setEvaluator((ClusteringEvaluator) evaluator);
        this.cm = new FilteredMutator(this);
        this.cm.setMutProb(settings.getDouble("cluster-mutator[@mut-prob]"));
        this.cm.setDecorated(cm);

        String distFunc = settings.getString("evaluator.distance[@type]");
        String distConf = settings.getString("evaluator.distance.config");
        KMeansOperator kmo = new KMeansOperator(pathDataset, distFunc, distConf, k);
        this.kmo = new FilteredMutator(this);
        this.kmo.setMutProb(settings.getDouble("kmeans-operator[@mut-prob]"));
        this.kmo.setDecorated(kmo);
    }

    @Override
    protected void doGeneration() {
        // Create sons
        cset = recombinator.recombine(pset);

        cset = cm.mutate(cset);
        cset.addAll(cm.getSterile());
        cset = kmo.mutate(cset);
        cset.addAll(kmo.getSterile());

        // Evaluate sons
        evaluator.evaluate(cset);
    }

    private void setDataset(String pathToDataset) {
        dataset = new ArffDataSet();
        ((AbstractDataset) dataset).setFileName(pathToDataset);
        dataset.setMetadata(new MIClassificationMetadata());
        ((MIClassificationMetadata) dataset.getMetadata()).setClassIndex(dataset.getMetadata().numberOfAttributes());
        dataset.loadExamples();
    }

    public FilteredMutator getKmo() {
        return kmo;
    }

    public FilteredMutator getCm() {
        return cm;
    }
}
