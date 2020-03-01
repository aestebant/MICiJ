package miclustering.evaluators;

import miclustering.utils.DistancesMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import weka.core.DistanceFunction;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SilhouetteIndex {
    private int maxNumClusters;
    private double[][] distances;

    public SilhouetteIndex(Instances instances,  int maxNumClusters, DistanceFunction distanceFunction, boolean parallelize) {
        this.maxNumClusters = maxNumClusters;
        DistancesMatrix dm = new DistancesMatrix();
        distances = dm.compute(instances, distanceFunction, parallelize);
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster) {
        int actualNumClusters = Collections.max(clusterAssignments) + 1;
        if (actualNumClusters == 0)
            return -1;
        int numInstances = clusterAssignments.size();

        double[] silhouette = new double[numInstances];

        for (int point = 0; point < numInstances; ++point) {
            double[] meanDistToCluster = new double[maxNumClusters];
            for (int other = 0; other < numInstances; ++other) {
                if (other != point && clusterAssignments.get(other) > -1)
                    meanDistToCluster[clusterAssignments.get(other)] += distances[point][other];
            }
            for (int c = 0; c < maxNumClusters; ++c) {
                if (c == clusterAssignments.get(point))
                    meanDistToCluster[c] /= (bagsPerCluster[c] - 1);
                else
                    meanDistToCluster[c] /= bagsPerCluster[c];
            }
            double aPoint = 0;
            if (clusterAssignments.get(point) > -1)
                aPoint = meanDistToCluster[clusterAssignments.get(point)];

            List<Double> possibleB = new ArrayList<>(maxNumClusters - 1);
            for (int j = 0; j < maxNumClusters; ++j) {
                if (j != clusterAssignments.get(point))
                    possibleB.add(meanDistToCluster[j]);
            }
            double bPoint = 0;
            if (possibleB.size() > 0)
                bPoint = Collections.min(possibleB);

            if (aPoint < bPoint)
                silhouette[point] = 1 - aPoint / bPoint;
            else if (aPoint == bPoint)
                silhouette[point] = 0D;
            else if (aPoint > bPoint)
                silhouette[point] = bPoint / aPoint - 1;
        }
        Mean mean = new Mean();
        return mean.evaluate(silhouette);
    }
}
