package miclustering.evaluators;

public class ExtEvalResult {
    private final int[][] confMatrix;
    private final int[] clusterToClass;
    private final int[] unnasigned;

    public ExtEvalResult(int[][] confMatrix, int[] mapClasses, int[] unnasigned) {
        this.confMatrix = confMatrix;
        this.clusterToClass = mapClasses;
        this.unnasigned = unnasigned;
    }

    public int[][] getConfMatrix() {
        return confMatrix;
    }

    public int[] getClusterToClass() {
        return clusterToClass;
    }

    public int[] getUnnasigned() {
        return unnasigned;
    }
}
