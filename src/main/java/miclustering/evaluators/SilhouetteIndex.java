package miclustering.evaluators;

import miclustering.utils.DistancesMatrix;
import weka.core.DistanceFunction;
import weka.core.Instances;

import java.util.*;

public class SilhouetteIndex {
    private int maxNumClusters;
    private double[][] distances;

    public SilhouetteIndex(Instances instances,  int maxNumClusters, DistanceFunction distanceFunction, int numThreads) {
        this.maxNumClusters = maxNumClusters;
        DistancesMatrix dm = new DistancesMatrix();
        distances = dm.compute(instances, numThreads, distanceFunction);
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster) {
        int actualNumClusters = Collections.max(clusterAssignments) + 1;
        if (actualNumClusters == 0)
            return -1;
        int numInstances = clusterAssignments.size();

        List<Double> silhouette = new ArrayList<>(numInstances);

        for (int point = 0; point < numInstances; ++point) {
            double[] meanDistToCluster = new double[maxNumClusters];
            for (int other = 0; other < numInstances; ++other) {
                if (other != point && clusterAssignments.get(other) != -1)
                    meanDistToCluster[clusterAssignments.get(other)] += distances[point][other];
            }
            for (int c = 0; c < maxNumClusters; ++c) {
                if (c == clusterAssignments.get(point))
                    meanDistToCluster[c] /= (bagsPerCluster[c] - 1);
                else
                    meanDistToCluster[c] /= bagsPerCluster[c];
            }
            double aPoint = 0;
            if (clusterAssignments.get(point) != -1)
                aPoint = meanDistToCluster[clusterAssignments.get(point)];

            List<Double> possibleB = new ArrayList<>(maxNumClusters -1);
            for (int j = 0; j < maxNumClusters; ++j) {
                if (j != clusterAssignments.get(point))
                    possibleB.add(meanDistToCluster[j]);
            }
            double bPoint = 0;
            if (possibleB.size() > 0)
                bPoint = Collections.min(possibleB);

            if (aPoint < bPoint)
                silhouette.add(1 - aPoint / bPoint);
            else if (aPoint == bPoint)
                silhouette.add(0D);
            else if (aPoint > bPoint)
                silhouette.add(bPoint / aPoint - 1);
        }
        OptionalDouble average = silhouette.stream().mapToDouble(a -> a).average();
        return average.isPresent()? average.getAsDouble() : -1;
    }
}
