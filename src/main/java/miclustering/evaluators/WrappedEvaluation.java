package miclustering.evaluators;

import miclustering.utils.DatasetCentroids;
import miclustering.utils.PrintConfusionMatrix;
import miclustering.utils.ProcessDataset;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WrappedEvaluation {
    private DistanceFunction distanceFunction;
    private RMSStdDev rmssd;
    private TotalWithinClusterVariation twcv;
    private FastTotalWithinClusterValidation ftwcv;
    private SilhouetteIndex silhouette;
    private XieBeniIndex xb;
    private DaviesBouldinIndex db;
    private S_DbwIndex sdbw;
    private DBCV dbcv;
    private ClassEvaluation classEval;
    private int maxNumClusters;
    private Instances dataset;
    private int nThreads;

    public WrappedEvaluation(String datasetPath, String distanceClass, String distanceConfig, int maxNumClusters) {
        dataset = ProcessDataset.readArff(datasetPath);
        dataset.setClassIndex(2);
        try {
            distanceFunction = (DistanceFunction) Utils.forName(DistanceFunction.class, distanceClass, Utils.splitOptions(distanceConfig));
        } catch (Exception e) {
            e.printStackTrace();
        }
        nThreads = Runtime.getRuntime().availableProcessors();
        rmssd = new RMSStdDev(dataset, maxNumClusters, distanceFunction, nThreads);
        twcv = new TotalWithinClusterVariation(dataset, maxNumClusters, distanceFunction, nThreads);
        ftwcv = new FastTotalWithinClusterValidation(dataset, maxNumClusters);
        sdbw = new S_DbwIndex(dataset, maxNumClusters, distanceFunction);
        silhouette = new SilhouetteIndex(dataset, maxNumClusters, distanceFunction, nThreads);
        xb = new XieBeniIndex(dataset, maxNumClusters, distanceFunction, nThreads);
        db = new DaviesBouldinIndex(dataset, maxNumClusters, distanceFunction, nThreads);
        dbcv = new DBCV(dataset, distanceFunction, maxNumClusters, nThreads);
        int nClass = dataset.numDistinctValues(2);
        classEval = new ClassEvaluation(dataset, maxNumClusters, nClass);
        this.maxNumClusters = maxNumClusters;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public double getRMSSD(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return rmssd.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getSilhouette(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return silhouette.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getXB(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return xb.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getDB(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return db.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getSdbw(List<Integer> clusterAssignments) {
        return sdbw.computeIndex(clusterAssignments);
    }

    public double getDBCV(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return dbcv.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public Map<String, String> getExternalEvaluation(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        ClassEvalResult cer = null;
        try {
            cer = classEval.computeConfusionMatrix(clusterAssignments, bagsPerCluster);
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert cer != null;
        double[] precision = classEval.computePrecision(cer);
        double[] recall = classEval.computeRecall(cer);
        double[] f1 = classEval.computeF1(cer, precision, recall);
        double[] specificity = classEval.computeSpecificity(cer);

        Map<String, String> result = new HashMap<>(2);
        result.put("confmat", PrintConfusionMatrix.singleLine(cer.getConfMatrix()));
        result.put("entropy", String.valueOf(classEval.computeEntropy(cer.getConfMatrix(), bagsPerCluster)));
        result.put("purity", String.valueOf(classEval.computePurity(cer.getConfMatrix())));
        result.put("rand", String.valueOf(classEval.computeRandIndex(cer)));
        result.put("micro-precision", Arrays.toString(precision));
        result.put("micro-recall", Arrays.toString(recall));
        result.put("micro-f1", Arrays.toString(f1));
        result.put("micro-specificity", Arrays.toString(specificity));
        result.put("macro-precision", String.valueOf(classEval.getMacroMeasure(cer, precision, clusterAssignments, bagsPerCluster)));
        result.put("macro-recall", String.valueOf(classEval.getMacroMeasure(cer, recall, clusterAssignments, bagsPerCluster)));
        result.put("macro-f1", String.valueOf(classEval.getMacroMeasure(cer, f1, clusterAssignments, bagsPerCluster)));
        result.put("macro-specificity", String.valueOf(classEval.getMacroMeasure(cer, specificity, clusterAssignments, bagsPerCluster)));

        return result;
    }

    private int[] countBagsPerCluster(List<Integer> clusterAssignments, int maxNumClusters) {
        int[] instancesPerCluster = new int[maxNumClusters];
        for (Integer classIdx : clusterAssignments) {
            instancesPerCluster[classIdx]++;
        }
        return instancesPerCluster;
    }

    public double getTWCV(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return twcv.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getFastTWCV(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        //return ftwcv.computeIndex(clusterAssignments, bagsPerCluster);
        return ftwcv.selectorModification(clusterAssignments, bagsPerCluster);
    }

    public void restartFTWCVMin() {
        ftwcv.restartMin();
    }

    public Map<Integer, Instance> getCentroids(List<Integer> clusterAssignments) {
        return DatasetCentroids.compute(dataset, maxNumClusters, clusterAssignments, nThreads);
    }

    public double[] computeDistances(int instanceIdx, Map<Integer, Instance> centroids) {
        double[] distances = new double[maxNumClusters];
        for (Map.Entry<Integer, Instance> entry : centroids.entrySet())
            distances[entry.getKey()] = distanceFunction.distance(dataset.get(instanceIdx), entry.getValue());
        return distances;
    }
}
