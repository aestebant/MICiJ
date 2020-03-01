package miclustering.utils;

import miclustering.evaluators.ExtEvalResult;
import weka.core.Attribute;
import weka.core.Utils;

import java.util.Arrays;
import java.util.Collections;

public class PrintConfusionMatrix {
    public static String singleLine(ExtEvalResult cer) {
        int[][] confMat = cer.getConfMatrix();
        int nClusters = confMat.length;
        int nClasses = confMat[0].length;

        StringBuilder result = new StringBuilder();
        for (int i = 0; i < nClusters; ++i) {
            for (int j = 0; j < nClasses; ++j)
                result.append(confMat[i][j]).append(" ");
            if (i < nClusters - 1)
                result.append("| ");
        }
        if (Arrays.stream(cer.getUnnasigned()).sum() > 0) {
            result.append(" unnasigned: ");
            for (int i = 0; i < nClasses; ++i) {
                result.append(cer.getUnnasigned()[i]);
                if (i < nClasses - 1)
                    result.append(" |");
            }
        }

        return result.toString();
    }

    public static String severalLines(ExtEvalResult cer, int[] bagsPerCluster, Attribute classAtt) {
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
                if (cer.getConfMatrix()[i][j] > maxVal) {
                    maxVal = cer.getConfMatrix()[i][j];
                }
            }
        }
        for (int i = 0; i < classAtt.numValues(); ++i) {
            if (cer.getUnnasigned()[i] > maxVal)
                maxVal = cer.getUnnasigned()[i];
        }

        int width = 1;
        if (actualNumClusters > 0)
            width += Math.max((int) (Math.log(maxVal) / Math.log(10D)), (int) (Math.log(actualNumClusters) / Math.log(10D)));

        for (int i = 0; i < nClasses; ++i) {
            matrix.append(" ").append(String.format("%1$" + width + "s", classAtt.value(i)));
        }
        matrix.append(" <- real classes\n");
        String bar = String.join("", Collections.nCopies(matrix.length(), "-"));
        matrix.append(bar).append("\n");

        for (int i = 0; i < maxNumClusters; ++i) {
            if (bagsPerCluster[i] > 0) {
                for (int j = 0; j < nClasses; ++j)
                    matrix.append(" ").append(Utils.doubleToString(cer.getConfMatrix()[i][j], width, 0));
                matrix.append(" | predicted cluster: ").append(i).append("\n");
            }
        }

        if (Arrays.stream(cer.getUnnasigned()).sum() > 0) {
            matrix.append(bar).append("\n");
            for (int i = 0; i < nClasses; ++i) {
                matrix.append(" ").append(Utils.doubleToString(cer.getUnnasigned()[i], width, 0));
            }
            matrix.append(" | unnasigned\n");
        }

        return matrix.toString();
    }
}
