package miclustering.evaluators;

import miclustering.utils.ProcessDataset;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Vector;

public class WrapperEvaluation {
    private DistanceFunction distanceFunction;
    private SilhouetteIndex silhouette;
    private S_DbwIndex sdbw;
    private ClassEvaluation classEval;

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
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public double getSilhouette(Vector<Integer> clusterAssignment) {
        return silhouette.computeIndex(clusterAssignment, null);
    }

    public double getSdbw(Vector<Integer> clusterAssignment) {
        return sdbw.computeIndex(clusterAssignment);
    }

    public double getPurity(Vector<Integer> clusterAssignment) {
        try {
            classEval.computeEval(clusterAssignment);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classEval.computePurity();
    }

    public double getRand(Vector<Integer> clusterAssignment) {
        return classEval.computeRandIndex();
    }
}
