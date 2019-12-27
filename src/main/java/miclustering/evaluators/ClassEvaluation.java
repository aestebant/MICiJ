package miclustering.evaluators;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.List;

public class ClassEvaluation {
    private Instances instances;
    private int maxNumClusters;
    private int numClasses;

    public ClassEvaluation(Instances instances, int maxNumClusters, int numClasses) {
        this.instances = instances;
        this.maxNumClusters = maxNumClusters;
        this.numClasses = numClasses;
    }

    public ClassEvalResult computeConfusionMatrix(List<Integer> clusterAssignments, int[] bagsPerCluster) throws Exception {
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
        mapClasses(maxNumClusters, 0, confMatrix, bagsPerCluster, current, best, 0);

        int[] classToCluster = new int[maxNumClusters + 1];
        for(int i = 0; i < maxNumClusters + 1; ++i) {
            classToCluster[i] = (int)best[i];
        }
        return new ClassEvalResult(confMatrix, classToCluster);
    }

    private static void mapClasses(int numClusters, int lev, int[][] counts, int[] bagsPerCluster, double[] current, double[] best, int error) throws Exception {
        if (lev == numClusters) {
            if ((double) error < best[numClusters]) {
                best[numClusters] = error;
                System.arraycopy(current, 0, best, 0, numClusters);
            }
        } else if (bagsPerCluster[lev] == 0) {
            current[lev] = -1.0D;
            mapClasses(numClusters, lev + 1, counts, bagsPerCluster, current, best, error);
        } else {
            current[lev] = -1.0D;
            mapClasses(numClusters, lev + 1, counts, bagsPerCluster, current, best, error + bagsPerCluster[lev]);
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
                        mapClasses(numClusters, lev + 1, counts, bagsPerCluster, current, best, error + (bagsPerCluster[lev] - counts[lev][i]));
                    }
                }
            }
        }
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

    public double computeRandIndex(ClassEvalResult cer) {
        int[] classToCluster = cer.getClusterToClass();
        int[][] confMatrix = cer.getConfMatrix();
        double rand = 0;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (classToCluster[i] > -1)
                rand += confMatrix[i][classToCluster[i]];
        }
        // Al dividir por el nº total de instancias también estamos penalizando si hay algunas clasificadas como ruido
        return rand / instances.numInstances();
    }

    public double[] computePrecision(ClassEvalResult cer) {
        int[] clusterToClass = cer.getClusterToClass();
        int[][] confMatrix = cer.getConfMatrix();
        double[] precision = new double[clusterToClass.length - 1];
        for (int i = 0; i < confMatrix.length; ++i) {
            if (clusterToClass[i] > -1) {
                int tp = confMatrix[i][clusterToClass[i]];
                int fp = 0;
                for (int j = 0; j < confMatrix[i].length; ++j) {
                    if (j != clusterToClass[i])
                        fp += confMatrix[i][j];
                }
                precision[clusterToClass[i]] = (double) tp / (tp+fp);
            }
        }
        return precision;
    }

    public double[] computeRecall(ClassEvalResult cer) {
        int[] clusterToClass = cer.getClusterToClass();
        int[][] confMatrix = cer.getConfMatrix();
        double[] recall = new double[clusterToClass.length - 1];
        for (int i = 0; i < confMatrix.length; ++i) {
            if (clusterToClass[i] > -1) {
                int tp = confMatrix[i][clusterToClass[i]];
                int fn = 0;
                for (int j = 0; j < confMatrix.length; ++j) {
                    if (j != i)
                        fn += confMatrix[j][clusterToClass[i]];
                }
                recall[clusterToClass[i]] = (double) tp / (tp + fn);
            }
        }
        return recall;
    }

    public double[] computeF1(ClassEvalResult cer, double[] precision, double[] recall) {
        int[] clusterToClass = cer.getClusterToClass();
        double[] f1 = new double[precision.length];
        for (int i = 0; i < precision.length; ++i){
            if (clusterToClass[i] > -1)
                f1[clusterToClass[i]] = 2 * (precision[clusterToClass[i]] * recall[clusterToClass[i]]) / (precision[clusterToClass[i]] + recall[clusterToClass[i]]);
        }
        return f1;
    }

    public double[] computeSpecificity(ClassEvalResult cer) {
        int[] clusterToClass = cer.getClusterToClass();
        int[][] confMatrix = cer.getConfMatrix();
        double[] specificity = new double[clusterToClass.length - 1];
        for (int i = 0; i < confMatrix.length; ++i) {
            if (clusterToClass[i] > -1) {
                int tn = 0;
                for (int j = 0; j < confMatrix.length; ++j) {
                    if (j != i) {
                        for (int k = 0; k < confMatrix[j].length; ++k) {
                            if (k != clusterToClass[i])
                                tn += confMatrix[j][k];
                        }
                    }
                }
                int fp = 0;
                for (int j = 0; j < confMatrix[i].length; ++j) {
                    if (j != clusterToClass[i])
                        fp += confMatrix[i][j];
                }
                specificity[clusterToClass[i]] = (double) tn / (tn + fp);
            }
        }
        return specificity;
    }

    public double getMacroMeasure(ClassEvalResult cer, double[] measure, List<Integer> clusterAssignments, int[] bagsPerCluster) {
        int[] clusterToClass = cer.getClusterToClass();
        if (measure.length == 2)
            return measure[1];
        else {
            double[] weights = new double[clusterToClass.length - 1];
            for (int i = 0; i < clusterToClass.length - 1; ++i) {
                if (clusterToClass[i] > -1) {
                    weights[clusterToClass[i]] = (double) bagsPerCluster[i] / clusterAssignments.size();
                }
            }
            Mean mean = new Mean();
            return mean.evaluate(measure, weights);
        }
    }
}
