package miclustering.evaluators;

public class ExtEvalResult {
    private int[][] confMatrix;
    private int[] clusterToClass;

    public ExtEvalResult(int[][] confMatrix, int[] mapClasses) {
        this.confMatrix = confMatrix;
        this.clusterToClass = mapClasses;
    }

    public int[][] getConfMatrix() {
        return confMatrix;
    }

    public int[] getClusterToClass() {
        return clusterToClass;
    }
}
