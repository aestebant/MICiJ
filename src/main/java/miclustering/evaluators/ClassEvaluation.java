package miclustering.evaluators;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Vector;

public class ClassEvaluation {
    private Instances instances;
    private int maxNumClusters;
    private int numClasses;

    public ClassEvaluation(Instances instances, int maxNumClusters, int numClasses) {
        this.instances = instances;
        this.maxNumClusters = maxNumClusters;
        this.numClasses = numClasses;
    }

    public ClassEvalResult computeConfusionMatrix(Vector<Integer> clusterAssignments, int[] instancesPerCluster) throws Exception {
        int[][] confMatrix = new int[maxNumClusters][numClasses];
        for(int i = 0; i < clusterAssignments.size(); ++i) {
            Instance instance = instances.get(i);
            if (clusterAssignments.get(i) != -1 && !instance.classIsMissing()) {
                confMatrix[clusterAssignments.get(i)][(int) instance.classValue()]++;
            }
        }
        double[] best = new double[maxNumClusters + 1];
        best[maxNumClusters] = 1.7976931348623157E308D;
        double[] current = new double[maxNumClusters + 1];
        mapClasses(maxNumClusters, 0, confMatrix, instancesPerCluster, current, best, 0);

        int[] clusterToClass = new int[maxNumClusters + 1];
        for(int i = 0; i < maxNumClusters + 1; ++i) {
            clusterToClass[i] = (int)best[i];
        }
        return new ClassEvalResult(confMatrix, clusterToClass);
    }

    private static void mapClasses(int numClusters, int lev, int[][] counts, int[] clusterTotals, double[] current, double[] best, int error) throws Exception {
        if (lev == numClusters) {
            if ((double) error < best[numClusters]) {
                best[numClusters] = error;
                System.arraycopy(current, 0, best, 0, numClusters);
            }
        } else if (clusterTotals[lev] == 0) {
            current[lev] = -1.0D;
            mapClasses(numClusters, lev + 1, counts, clusterTotals, current, best, error);
        } else {
            current[lev] = -1.0D;
            mapClasses(numClusters, lev + 1, counts, clusterTotals, current, best, error + clusterTotals[lev]);
            for(int i = 0; i < counts[0].length; ++i) {
                if (counts[lev][i] > 0) {
                    boolean ok = true;
                    for(int j = 0; j < lev; ++j) {
                        if ((int) current[j] == i) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        current[lev] = i;
                        mapClasses(numClusters, lev + 1, counts, clusterTotals, current, best, error + (clusterTotals[lev] - counts[lev][i]));
                    }
                }
            }
        }
    }

    public double computeRandIndex(int[][]confMatrix, int[] classToCluster) {
        double rand = 0;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (classToCluster[i] > -1)
                rand += confMatrix[i][classToCluster[i]];
        }
        // Al dividir por el nº total de instancias también estamos penalizando si hay algunas clasificadas como ruido
        return rand / instances.numInstances();
    }

    public double computePurity(int[][] confMatrix) {
        double purity = 0;
        for(int i = 0; i < maxNumClusters; ++i) {
            if (Arrays.stream(confMatrix[i]).max().isPresent())
                purity += Arrays.stream(confMatrix[i]).max().getAsInt();
        }
        // Al dividir por el nº total de instancias también estamos penalizando si hay algunas clasificadas como ruido
        return purity / instances.numInstances();
    }
}
