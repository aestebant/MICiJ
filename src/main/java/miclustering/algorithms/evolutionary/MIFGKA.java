package miclustering.algorithms.evolutionary;

import jclec.base.FilteredMutator;
import org.apache.commons.configuration.Configuration;

public class MIFGKA extends MIGKA {

    @Override
    public void configure(Configuration settings) {
        super.configure(settings);

        String pathDataset = settings.getString("evaluator.dataset");
        setDataset(pathDataset);
        int k = settings.getInt("evaluator.num-clusters");
        String distFunc = settings.getString("evaluator.distance[@type]");
        String distConf = settings.getString("evaluator.distance.config");
        KMeansOperator kmo = new FastKMeansOperator(pathDataset, distFunc, distConf, k);
        this.kmo = new FilteredMutator(this);
        this.kmo.setMutProb(settings.getDouble("kmeans-operator[@mut-prob]"));
        this.kmo.setDecorated(kmo);
    }

    @Override
    protected void doGeneration() {
        ((ClusteringEvaluator) evaluator).getClusterEval().restartFtwcvMin();
        super.doGeneration();
    }
}
