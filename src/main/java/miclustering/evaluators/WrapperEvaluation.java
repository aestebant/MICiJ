package miclustering.evaluators;

import miclustering.utils.PrintConfusionMatrix;
import miclustering.utils.ProcessDataset;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.Utils;

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
        Map<String, String> result = new HashMap<>(2);
        assert cer != null;
        result.put("purity", String.valueOf(classEval.computePurity(cer.getConfMatrix())));
        result.put("rand", String.valueOf(classEval.computeRandIndex(cer.getConfMatrix(), cer.getClusterToClass())));
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
