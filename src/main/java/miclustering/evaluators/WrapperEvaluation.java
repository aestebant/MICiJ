package miclustering.evaluators;

import miclustering.utils.PrintConfusionMatrix;
import miclustering.utils.ProcessDataset;
import org.apache.commons.math3.stat.descriptive.moment.GeometricMean;
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
        silhouette = new SilhouetteIndex(dataset, maxNumClusters, distanceFunction, Runtime.getRuntime().availableProcessors());
        int nClass = dataset.numDistinctValues(2);
        classEval = new ClassEvaluation(dataset, maxNumClusters, nClass);
        this.maxNumClusters = maxNumClusters;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public double getSilhouette(Vector<Integer> clusterAssignments) {
        int[] instancesPerCluster = countInstancesPerCluster(clusterAssignments, maxNumClusters);
        return silhouette.computeIndex(clusterAssignments, instancesPerCluster);
    }

    public double getSdbw(Vector<Integer> clusterAssignments) {
        return sdbw.computeIndex(clusterAssignments);
    }

    public Map<String, String> getExternalEvaluation(Vector<Integer> clusterAssignments) {
        int[] instancesPerCluster = countInstancesPerCluster(clusterAssignments, maxNumClusters);
        ClassEvalResult cer = null;
        try {
            cer = classEval.computeConfusionMatrix(clusterAssignments, instancesPerCluster);
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert cer != null;
        double[] precision = classEval.computePrecision(cer);
        double[] recall = classEval.computeRecall(cer);
        double[] f1 = classEval.computeF1(precision, recall);
        GeometricMean gm = new GeometricMean();

        Map<String, String> result = new HashMap<>(2);
        result.put("purity", String.valueOf(classEval.computePurity(cer.getConfMatrix())));
        result.put("rand", String.valueOf(classEval.computeRandIndex(cer)));
        result.put("macro-precision", String.valueOf(gm.evaluate(precision)));
        result.put("macro-recall", String.valueOf(gm.evaluate(recall)));
        result.put("macro-f1", String.valueOf(gm.evaluate(f1)));
        result.put("micro-precision", Arrays.toString(precision));
        result.put("micro-recall", Arrays.toString(recall));
        result.put("micro-f1", Arrays.toString(f1));
        result.put("confmat", PrintConfusionMatrix.singleLine(cer.getConfMatrix()));
        return result;
    }

    private int[] countInstancesPerCluster(Vector<Integer> clusterAssignments, int maxNumClusters) {
        int[] instancesPerCluster = new int[maxNumClusters];
        for (Integer classIdx : clusterAssignments) {
            instancesPerCluster[classIdx]++;
        }
        return instancesPerCluster;
    }
}
