package evaluators;

import org.apache.commons.math3.geometry.euclidean.twod.Vector2DFormat;
import org.apache.commons.math3.linear.ArrayRealVector;
import utils.ProcessDataset;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Esta métrica no tiene en cuenta las instancias clasificadas como ruido.
 */
public class S_DbwIndex {
    private int actualNumClusters;
    private int maxNumClusters;
    private Vector<Integer> clusterAssignments;
    private DistanceFunction distanceFunction;
    private Instances instances;
    private Map<Integer, Double> l2NormClusters;
    private Map<Integer, Instances> instancesUnion;
    private int[] nInstPerCluster;

    public S_DbwIndex(Instances instances, int actualNumClusters, int maxNumClusters, Vector<Integer> clusterAssignments, DistanceFunction distanceFunction) {
        this.instances = instances;
        this.actualNumClusters = actualNumClusters;
        this.maxNumClusters = maxNumClusters;
        this.clusterAssignments = clusterAssignments;
        this.distanceFunction = distanceFunction;
    }

    public double computeIndex() {
        if (actualNumClusters == 0)
            return 0D;

        double scat = computeScat();
        double densBw = computeDens_Bw();

        return scat + densBw;
    }

    /**
     * Intra-cluster variance
     */
    private double computeScat() {
        int numInstances = instances.numInstances();

        nInstPerCluster = new int[maxNumClusters];
        for (int i = 0; i < numInstances; ++i) {
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx >= 0)
                nInstPerCluster[clusterIdx] += instances.get(i).relationalValue(1).numInstances();
        }

        instancesUnion = new HashMap<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            instancesUnion.put(i, new Instances(instances.get(0).relationalValue(1), nInstPerCluster[i]));
        }
        for (int i = 0; i < numInstances; ++i) {
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx >= 0)
                instancesUnion.get(clusterIdx).addAll(instances.get(i).relationalValue(1));
        }

        l2NormClusters = new HashMap<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            double[] variances = instancesUnion.get(i).variances();
            l2NormClusters.put(i, new ArrayRealVector(variances).getNorm());
        }

        Instances datasetUnion = new Instances(instances.get(0).relationalValue(1), IntStream.of(nInstPerCluster).sum());
        for (int i = 1; i < maxNumClusters; ++i)
            datasetUnion.addAll(instancesUnion.get(i));
        double[] variancesDataset = datasetUnion.variances();
        double l2NormDataset = new ArrayRealVector(variancesDataset).getNorm();

        double result = 0;
        for (int i = 0; i < maxNumClusters; ++i)
            result += l2NormClusters.get(i) / l2NormDataset;
        result /= actualNumClusters;

        return result;
    }

    /**
     * Inter-cluster density
     */
    private double computeDens_Bw() {
        int numInstances = instances.numInstances();

        double stdev = l2NormClusters.values().stream().mapToDouble(Double::doubleValue).sum();
        stdev = Math.sqrt(stdev) / actualNumClusters;

        double result = 0D;
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = 0; j < maxNumClusters; ++j) {
                if (i != j) {
                    Instances u = new Instances(instances.get(0).relationalValue(1), nInstPerCluster[i] + nInstPerCluster[j]);
                    u.addAll(instancesUnion.get(i));
                    u.addAll(instancesUnion.get(j));

                    System.out.println("U debería tener: " + instancesUnion.get(i).numInstances() + " + " + instancesUnion.get(j).numInstances());
                    System.out.println("Instancia de prueba tiene " + instances.get(0).relationalValue(1).numInstances());

                    Instance aux = ProcessDataset.copyBag(instances.get(0));
                    aux.relationalValue(1).delete();
                    aux.relationalValue(1).addAll(u);

                    System.out.println("U tiene al final: " + aux.relationalValue(1).numInstances());
                    System.out.println("===============================================");

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
                    result += densityU / Math.max(densityI, densityJ);
                }
            }
        }

        return result / (actualNumClusters * (actualNumClusters -1));
    }

    /**
     * Previous approach of Dens_Bw in which bags are summarized in single instances.
     * Not valid for distribution-based distances
     */
    private double computeDens_BwSingleInstance() {
        int numAttributes = instances.get(0).relationalValue(1).numAttributes();
        int numInstances = instances.numInstances();

        double stdev = l2NormClusters.values().stream().mapToDouble(Double::doubleValue).sum();
        stdev = Math.sqrt(stdev) / actualNumClusters;

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
                if (i != j) {
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
                    result += densityU / Math.max(densityI, densityJ);
                }
            }
        }
        return result / (actualNumClusters * (actualNumClusters -1));
    }
}
