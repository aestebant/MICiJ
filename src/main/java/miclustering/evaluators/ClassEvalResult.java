package miclustering.evaluators;

public class ClassEvalResult {
    private int[][] confMatrix;
    private int[] clusterToClass;

    public ClassEvalResult(int[][] confMatrix, int[] mapClasses) {
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
