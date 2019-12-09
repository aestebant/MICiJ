package miclustering.evaluators;

import miclustering.utils.ProcessDataset;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.FastMath;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.stream.IntStream;

/**
 * Esta métrica no tiene en cuenta las instancias clasificadas como ruido.
 */
public class S_DbwIndex {
    private int maxNumClusters;
    private DistanceFunction distanceFunction;
    private Instances instances;

    public S_DbwIndex(Instances instances, int maxNumClusters, DistanceFunction distanceFunction) {
        this.instances = instances;
        this.maxNumClusters = maxNumClusters;
        this.distanceFunction = distanceFunction;
    }

    public double computeIndex(Vector<Integer> clusterAssignments) {
        int actualNumClusters = Collections.max(clusterAssignments) + 1;
        if (actualNumClusters == 0)
            return Double.POSITIVE_INFINITY;

        int[] nInstPerCluster = new int[maxNumClusters];
        for (int i = 0; i < clusterAssignments.size(); ++i) {
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx >= 0)
                nInstPerCluster[clusterIdx] += instances.get(i).relationalValue(1).numInstances();
        }

        Map<Integer, Instances> instancesUnion = new HashMap<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            instancesUnion.put(i, new Instances(instances.get(0).relationalValue(1), nInstPerCluster[i]));
        }
        for (int i = 0; i < clusterAssignments.size(); ++i) {
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx >= 0)
                instancesUnion.get(clusterIdx).addAll(instances.get(i).relationalValue(1));
        }

        Map<Integer, Double> l2NormClusters = new HashMap<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            double[] variances = instancesUnion.get(i).variances();
            l2NormClusters.put(i, new ArrayRealVector(variances).getNorm());
        }

        double scat = computeScat(actualNumClusters, nInstPerCluster, instancesUnion, l2NormClusters);
        double densBw = computeDens_Bw(clusterAssignments, actualNumClusters, nInstPerCluster, instancesUnion, l2NormClusters);

        return scat + densBw;
    }

    /**
     * Intra-cluster variance
     */
    private double computeScat(int actualNumClusters, int[] nInstPerCluster, Map<Integer, Instances> instancesUnion, Map<Integer, Double> l2NormClusters) {
        Instances datasetUnion = new Instances(instances.get(0).relationalValue(1), IntStream.of(nInstPerCluster).sum());
        for (int i = 0; i < maxNumClusters; ++i)
            datasetUnion.addAll(instancesUnion.get(i));
        double l2NormDataset = new ArrayRealVector(datasetUnion.variances()).getNorm();

        double result = 0;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (nInstPerCluster[i] > 0)
                result += l2NormClusters.get(i) / l2NormDataset;
        }
        result /= actualNumClusters;

        return result;
    }

    /**
     * Inter-cluster density
     */
    private double computeDens_Bw(Vector<Integer> clusterAssignments, int actualNumClusters, int[] nInstPerCluster, Map<Integer, Instances> instancesUnion, Map<Integer, Double> l2NormClusters) {
        int numInstances = instances.numInstances();

        double stdev = l2NormClusters.values().stream().mapToDouble(Double::doubleValue).sum();
        stdev = FastMath.sqrt(stdev) / actualNumClusters;

        double result = 0D;
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = 0; j < maxNumClusters; ++j) {
                if (i != j) {
                    Instances u = new Instances(instances.get(0).relationalValue(1), nInstPerCluster[i] + nInstPerCluster[j]);
                    u.addAll(instancesUnion.get(i));
                    u.addAll(instancesUnion.get(j));

                    Instance aux = ProcessDataset.copyBag(instances.get(0));
                    aux.relationalValue(1).delete();
                    aux.relationalValue(1).addAll(u);

                    double densityU = 0D;
                    double densityI = 0D;
                    double densityJ = 0D;
                    for (int k = 0; k < numInstances; ++k) {
                        if (clusterAssignments.get(k) == i || clusterAssignments.get(k) == j) {
                            double distance = distanceFunction.distance(aux, instances.get(k));
                            if (distance <= stdev)
                                densityU++;
                            if (clusterAssignments.get(k) == i)
                                densityI++;
                            else
                                densityJ++;
                        }
                    }
                    result += densityU / FastMath.max(densityI, densityJ);
                }
            }
        }
        return result / (actualNumClusters * (actualNumClusters -1));
    }

    /**
     * Previous approach of Dens_Bw in which bags are summarized in single instances.
     * Not valid for distribution-based miclustering.distances
     */
    @SuppressWarnings("unused")
    private double computeDens_BwSingleInstance(Vector<Integer> clusterAssignments, int actualNumClusters, int[] nInstPerCluster, Map<Integer, Instances> instancesUnion, Map<Integer, Double> l2NormClusters) {
        int numAttributes = instances.get(0).relationalValue(1).numAttributes();
        int numInstances = instances.numInstances();

        double stdev = l2NormClusters.values().stream().mapToDouble(Double::doubleValue).sum();
        stdev = FastMath.sqrt(stdev) / actualNumClusters;

        Map<Integer, Instance> clustersCenters = new HashMap<>(actualNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            double[] mean = new double[numAttributes];
            for (int j = 0; j < numAttributes; ++j)
                mean[j] = instancesUnion.get(i).meanOrMode(j);
            clustersCenters.put(i, new DenseInstance(1D, mean));
        }

        double result = 0D;
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = 0; j < maxNumClusters; ++j) {
                if (i != j && nInstPerCluster[i] > 0 && nInstPerCluster[j] > 0) {
                    double[] means = new double[numAttributes];
                    for (int k = 0; k < numAttributes; ++k)
                        means[k] = (clustersCenters.get(i).value(k) + clustersCenters.get(j).value(k)) / 2;
                    Instance u = new DenseInstance(1D, means);

                    double densityU = 0D;
                    double densityI = 0D;
                    double densityJ = 0D;

                    for (int k = 0; k < numInstances; ++k) {
                        if (clusterAssignments.get(k) == i || clusterAssignments.get(k) == j) {
                            double distance = distanceFunction.distance(u, instances.get(k));
                            if (distance <= stdev)
                                densityU++;
                            if (clusterAssignments.get(k) == i)
                                densityI++;
                            else
                                densityJ++;
                        }
                    }
                    result += densityU / FastMath.max(densityI, densityJ);
                }
            }
        }
        return result / (actualNumClusters * (actualNumClusters -1));
    }
}
