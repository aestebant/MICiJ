package miclustering;

import miclustering.evaluators.ClusterEvaluation;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.List;

public class TestResult {
    public static void main(String[] args) {
        int[] result = {};
        List<Integer> clusterAssignments = new ArrayList<>(result.length);
        for (int value : result) clusterAssignments.add(value);

        String options = "-d -c last -k 2 -parallelize -A HausdorffDistance -hausdorff-type 0";
        ClusterEvaluation ce = new ClusterEvaluation();
        try {
            ce.setOptions(Utils.splitOptions(options));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
