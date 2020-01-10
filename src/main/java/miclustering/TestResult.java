package miclustering;

import miclustering.algorithms.OneStepKMeans;
import miclustering.evaluators.ClusterEvaluation;
import weka.core.Utils;

import java.util.List;

public class TestResult {
    public static void main(String[] args) {
        int[] result = {101,114};
        String dataset = "datasets/elephant_relational-z5.arff";
        OneStepKMeans oskm = new OneStepKMeans(dataset, "HausdorffDistance", "-hausdorff-type 0", 2, true);
        List<Integer> clusterAssignments = oskm.assignBagsToClusters(result, true);

        //List<Integer> clusterAssignments = new ArrayList<>(result.length);
        //for (int value : result) clusterAssignments.add(value);

        System.out.println(clusterAssignments);

        String options = "-d " + dataset + " -c last -k 2 -parallelize -A HausdorffDistance -hausdorff-type 0";
        ClusterEvaluation ce = new ClusterEvaluation();
        try {
            ce.setOptions(Utils.splitOptions(options));
        } catch (Exception e) {
            e.printStackTrace();
        }
        ce.fullEvaluation(clusterAssignments, true);

        System.out.println(ce.printFullEvaluation());
    }
}
