package miclustering.utils;

import miclustering.evaluators.ClassEvalResult;
import weka.core.Attribute;
import weka.core.Utils;

public class PrintConfusionMatrix {
    public static String singleLine(int[][] confMat) {
        int nClusters = confMat.length;
        int nClasses = confMat[0].length;

        StringBuilder result = new StringBuilder();
        for (int i = 0; i < nClusters; ++i) {
            for (int j = 0; j < nClasses; ++j)
                result.append(confMat[i][j]).append(" ");
            if (i < nClusters - 1)
                result.append("| ");
        }
        return result.toString();
    }

    public static String severalLines(ClassEvalResult classEvalResult, int[] bagsPerCluster, Attribute classAtt) {
        int maxNumClusters = bagsPerCluster.length;
        int actualNumClusters = maxNumClusters;
        for (int value : bagsPerCluster) {
            if (value < 1)
                actualNumClusters--;
        }
        int nClasses = classAtt.numValues();

        StringBuilder matrix = new StringBuilder();
        int maxVal = 0;
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = 0; j < classAtt.numValues(); ++j) {
                if (classEvalResult.getConfMatrix()[i][j] > maxVal) {
                    maxVal = classEvalResult.getConfMatrix()[i][j];
                }
            }
        }
        int Cwidth = 1 + Math.max((int) (Math.log(maxVal) / Math.log(10D)), (int) (Math.log(actualNumClusters) / Math.log(10D)));

        for (int i = 0; i < maxNumClusters; ++i) {
            if (bagsPerCluster[i] > 0) {
                matrix.append(" ").append(Utils.doubleToString(i, Cwidth, 0));
            }
        }
        matrix.append("  <-- assigned to cluster\n");

        for (int i = 0; i < nClasses; ++i) {
            for (int j = 0; j < maxNumClusters; ++j) {
                if (bagsPerCluster[j] > 0)
                    matrix.append(" ").append(Utils.doubleToString(classEvalResult.getConfMatrix()[j][i], Cwidth, 0));
            }
            matrix.append(" | ").append(classAtt.value(i)).append("\n");
        }
        return matrix.toString();
    }
}
