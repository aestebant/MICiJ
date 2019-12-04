package miclustering;

import miclustering.evaluators.WrapperEvaluation;

import java.util.Vector;

public class Pru {
    public static void main(String[] args) {
        String dataset = "datasets/eastwest_relational.arff";
        String distance = "MahalanobisDistance";
        int k = 2;
        WrapperEvaluation evaluation = new WrapperEvaluation(dataset, distance, k);

        int[] resFromJCLEC = {1,1,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,1};
        Vector<Integer> clusterAssignmet = new Vector<>(resFromJCLEC.length);
        for (int value : resFromJCLEC) clusterAssignmet.addElement(value);

        System.out.println(evaluation.getEvaluation(clusterAssignmet));
    }
}
