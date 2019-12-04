package miclustering.evaluators;

import miclustering.utils.ProcessDataset;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Collections;
import java.util.Vector;

public class WrapperEvaluation {
    private Instances dataset;
    private DistanceFunction distanceFunction;
    private SilhouetteIndex evaluator;

    public WrapperEvaluation(String datasetPath, String distanceClass, String distanceConfig, int maxNumClusters) {
        dataset = ProcessDataset.readArff(datasetPath);
        try {
            distanceFunction = (DistanceFunction) Utils.forName(DistanceFunction.class, distanceClass, Utils.splitOptions(distanceConfig));
        } catch (Exception e) {
            e.printStackTrace();
        }
        //evaluator = new S_DbwIndex(dataset, maxNumClusters, distanceFunction);
        evaluator = new SilhouetteIndex(dataset, maxNumClusters, distanceFunction, Runtime.getRuntime().availableProcessors());
    }

    public double getEvaluation(Vector<Integer> clusterAssignment) {
        //return evaluator.computeIndex(clusterAssignment, Collections.max(clusterAssignment)+1, null);
        return evaluator.computeIndex(clusterAssignment, Collections.max(clusterAssignment)+1, null);
    }
}
