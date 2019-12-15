package miclustering.evaluators;

import miclustering.utils.PrintConfusionMatrix;
import miclustering.utils.ProcessDataset;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class WrapperEvaluation {
    private DistanceFunction distanceFunction;
    private SilhouetteIndex silhouette;
    private S_DbwIndex sdbw;
    private DBCV dbcv;
    private ClassEvaluation classEval;
    private int maxNumClusters;

    public WrapperEvaluation(String datasetPath, String distanceClass, String distanceConfig, int maxNumClusters) {
        Instances dataset = ProcessDataset.readArff(datasetPath);
        dataset.setClassIndex(2);
        try {
            distanceFunction = (DistanceFunction) Utils.forName(DistanceFunction.class, distanceClass, Utils.splitOptions(distanceConfig));
        } catch (Exception e) {
            e.printStackTrace();
        }
        sdbw = new S_DbwIndex(dataset, maxNumClusters, distanceFunction);
        int nThreads = Runtime.getRuntime().availableProcessors();
        silhouette = new SilhouetteIndex(dataset, maxNumClusters, distanceFunction, nThreads);
        dbcv = new DBCV(dataset, distanceFunction, maxNumClusters, nThreads);
        int nClass = dataset.numDistinctValues(2);
        classEval = new ClassEvaluation(dataset, maxNumClusters, nClass);
        this.maxNumClusters = maxNumClusters;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public double getSilhouette(Vector<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return silhouette.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getSdbw(Vector<Integer> clusterAssignments) {
        return sdbw.computeIndex(clusterAssignments);
    }

    public double getDBCV(Vector<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return dbcv.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public Map<String, String> getExternalEvaluation(Vector<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        ClassEvalResult cer = null;
        try {
            cer = classEval.computeConfusionMatrix(clusterAssignments, bagsPerCluster);
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert cer != null;
        double[] precision = classEval.computePrecision(cer);
        double[] recall = classEval.computeRecall(cer);
        double[] f1 = classEval.computeF1(precision, recall);
        Mean mean = new Mean();
        double[] weights = new double[bagsPerCluster.length];
        for (int i = 0; i < weights.length; ++i)
            weights[i] = (double) bagsPerCluster[i] / clusterAssignments.size();

        Map<String, String> result = new HashMap<>(2);
        result.put("purity", String.valueOf(classEval.computePurity(cer.getConfMatrix())));
        result.put("rand", String.valueOf(classEval.computeRandIndex(cer)));
        result.put("macro-precision", String.valueOf(mean.evaluate(precision, weights)));
        result.put("macro-recall", String.valueOf(mean.evaluate(recall, weights)));
        result.put("macro-f1", String.valueOf(mean.evaluate(f1, weights)));
        result.put("micro-precision", Arrays.toString(precision));
        result.put("micro-recall", Arrays.toString(recall));
        result.put("micro-f1", Arrays.toString(f1));
        result.put("confmat", PrintConfusionMatrix.singleLine(cer.getConfMatrix()));
        return result;
    }

    private int[] countBagsPerCluster(Vector<Integer> clusterAssignments, int maxNumClusters) {
        int[] instancesPerCluster = new int[maxNumClusters];
        for (Integer classIdx : clusterAssignments) {
            instancesPerCluster[classIdx]++;
        }
        return instancesPerCluster;
    }
}
