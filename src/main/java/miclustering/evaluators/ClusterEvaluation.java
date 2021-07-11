package miclustering.evaluators;

import miclustering.algorithms.ClusterLikeClass;
import miclustering.algorithms.MIClusterer;
import miclustering.distances.HausdorffDistance;
import miclustering.utils.DatasetCentroids;
import miclustering.utils.LoadByName;
import miclustering.utils.PrintConfusionMatrix;
import miclustering.utils.ProcessDataset;
import weka.clusterers.Clusterer;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.MethodDescriptor;
import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;

public class ClusterEvaluation implements Serializable, OptionHandler, RevisionHandler {
    private DistanceFunction distanceFunction;
    private Instances instances;
    private DatasetCentroids datasetCentroids;
    private boolean reuseEvaluator = false;

    private List<Integer> clusterAssignments;
    private int[] bagsPerCluster;
    private int unclusteredInstances;
    private int actualNumClusters;

    private int classAtt;
    private int maxNumClusters;
    private boolean printClusterAssignments;
    private String attributeRangeString;

    private RMSStdDev rmssd;
    private double computedRmssd;
    private SilhouetteIndex silhouette;
    private double computedSilhouette;
    private XieBeniIndex xb;
    private double computedXb;
    private DaviesBouldinIndex db;
    private double computedDb;
    private S_DbwIndex sdbw;
    private double computedSdbw;
    private DBCV dbcv;
    private double computedDbcv;
    private ExternalEvaluation extEval;
    private ExtEvalResult extEvalResult;
    private double entropy;
    private double purity;
    private double rand;
    private double[] precision;
    private double[] recall;
    private double[] f1;
    private double[] specificity;

    private TotalWithinClusterVariation twcv;
    private FastTotalWithinClusterValidation ftwcv;

    public void evaluateClusterer(MIClusterer clusterer, boolean parallelize) throws Exception {
        Instances processData;
        if (classAtt > -1 && !(clusterer instanceof ClusterLikeClass)) {
            Remove removeClass = new Remove();
            removeClass.setAttributeIndices(String.valueOf(classAtt + 1)); // No sé por qué aquí cuenta los índices empezando en 1
            removeClass.setInvertSelection(false);
            try {
                removeClass.setInputFormat(instances);
            } catch (Exception e) {
                e.printStackTrace();
            }
            processData = Filter.useFilter(instances, removeClass);
        } else {
            processData = new Instances(instances);
        }

        clusterer.buildClusterer(processData);
        //TODO Si se reusa el evaluador hay que tener cuidado con el nº de clusters
        maxNumClusters = Math.max(clusterer.numberOfClusters(), maxNumClusters);

        if (!reuseEvaluator) {
            distanceFunction = clusterer.getDistanceFunction();
            rmssd = new RMSStdDev(instances, maxNumClusters, distanceFunction, parallelize);
            sdbw = new S_DbwIndex(instances, maxNumClusters, distanceFunction, parallelize);
            silhouette = new SilhouetteIndex(instances, maxNumClusters, distanceFunction, parallelize);
            xb = new XieBeniIndex(instances, maxNumClusters, distanceFunction, parallelize);
            db = new DaviesBouldinIndex(instances, maxNumClusters, distanceFunction, parallelize);
            dbcv = new DBCV(instances, distanceFunction, maxNumClusters, parallelize);
            if (classAtt > -1)
                extEval = new ExternalEvaluation(instances, maxNumClusters, instances.numDistinctValues(classAtt));
            twcv = new TotalWithinClusterVariation(instances, maxNumClusters, distanceFunction, parallelize);
            ftwcv = new FastTotalWithinClusterValidation(instances, maxNumClusters);
            datasetCentroids = new DatasetCentroids(instances, maxNumClusters, distanceFunction);
        }

        clusterAssignments = clusterer.getClusterAssignments();
        fullEvaluation(clusterAssignments, parallelize);
    }

    public void fullEvaluation(List<Integer> clusterAssignments, boolean parallelize) {
        bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        actualNumClusters = countRealClusters(maxNumClusters, bagsPerCluster);
        Map<Integer, Instance> centroids = datasetCentroids.compute(clusterAssignments, parallelize);

        computedRmssd = rmssd.computeIndex(clusterAssignments, bagsPerCluster, centroids);
        computedSilhouette = silhouette.computeIndex(clusterAssignments, bagsPerCluster);
        computedXb = xb.computeIndex(clusterAssignments, bagsPerCluster, centroids);
        computedDb = db.computeIndex(clusterAssignments, bagsPerCluster, centroids);
        computedSdbw = sdbw.computeIndex(clusterAssignments);
        computedDbcv = dbcv.computeIndex(clusterAssignments, bagsPerCluster);

        if (classAtt > -1) {
            extEval = new ExternalEvaluation(instances, maxNumClusters, instances.numClasses());
            extEvalResult = extEval.computeConfusionMatrix(clusterAssignments, bagsPerCluster);
            entropy = extEval.computeEntropy(extEvalResult.getConfMatrix(), bagsPerCluster);
            purity = extEval.computePurity(extEvalResult.getConfMatrix());
            rand = extEval.computeRandIndex(extEvalResult);
            precision = extEval.computePrecision(extEvalResult);
            recall = extEval.computeRecall(extEvalResult);
            f1 = extEval.computeF1(extEvalResult, precision, recall);
            specificity = extEval.computeSpecificity(extEvalResult);
        }
        this.clusterAssignments = clusterAssignments;
    }

    private List<Integer> getClusterAssignments(Clusterer clusterer, Instances instances) {
        List<Integer> clusterAssignments = new ArrayList<>(instances.numInstances());
        for (Instance i : instances) {
            try {
                clusterAssignments.add(clusterer.clusterInstance(i));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return clusterAssignments;
    }

    private int[] countBagsPerCluster(List<Integer> clusterAssignments, int maxNumClusters) {
        int[] instancesPerCluster = new int[maxNumClusters];
        for (Integer clusterIdx : clusterAssignments) {
            if (clusterIdx < 0)
                unclusteredInstances++;
            else
                instancesPerCluster[clusterIdx]++;
        }
        return instancesPerCluster;
    }

    private int countRealClusters(int maxNumClusters, int[] instancesPerCluster) {
        int actualNumClusters = maxNumClusters;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (instancesPerCluster[i] == 0)
                actualNumClusters--;
        }
        return actualNumClusters;
    }

    public int getUnclusteredInstances() {
        return unclusteredInstances;
    }

    public int getActualNumClusters() {
        return actualNumClusters;
    }

    public double computeRmssd(List<Integer> clusterAssignments, boolean parallelize) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return rmssd.computeIndex(clusterAssignments, bagsPerCluster, parallelize);
    }

    public double getRmssd() {
        return computedRmssd;
    }

    public double computeSilhouette(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return silhouette.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getSilhouette() {
        return computedSilhouette;
    }

    public double computeXb(List<Integer> clusterAssignments, boolean parallelize) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return xb.computeIndex(clusterAssignments, bagsPerCluster, parallelize);
    }

    public double getXb() {
        return computedXb;
    }

    public double computeDb(List<Integer> clusterAssignments, boolean parallelize) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return db.computeIndex(clusterAssignments, bagsPerCluster, parallelize);
    }

    public double getDb() {
        return computedDb;
    }

    public double computeSdbw(List<Integer> clusterAssignments) {
        return sdbw.computeIndex(clusterAssignments);
    }

    public double getSdbw() {
        return computedSdbw;
    }

    public double computeDbcv(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        return dbcv.computeIndex(clusterAssignments, bagsPerCluster);
    }

    public double getDbcv() {
        return computedDbcv;
    }

    public double computeTwcv(List<Integer> clusterAssignments, boolean parallelize) {
        return twcv.computeIndex(clusterAssignments, parallelize);
    }

    public double computeFtwcv(List<Integer> clusterAssignments) {
        int[] bagsPerCluster = countBagsPerCluster(clusterAssignments, maxNumClusters);
        //return ftwcv.computeIndex(clusterAssignments, bagsPerCluster);
        return ftwcv.selectorModification(clusterAssignments, bagsPerCluster);
    }

    public void restartFtwcvMin() {
        ftwcv.restartMin();
    }

    public ExtEvalResult getExtEvalResult() {
        return extEvalResult;
    }

    public double getEntropy() {
        return entropy;
    }

    public double getPurity() {
        return purity;
    }

    public double getRand() {
        return rand;
    }

    public double getMacroPrecision() {
        return ExternalEvaluation.getMacroMeasure(extEvalResult, precision, clusterAssignments, bagsPerCluster);
    }

    public double getMacroRecall() {
        return ExternalEvaluation.getMacroMeasure(extEvalResult, recall, clusterAssignments, bagsPerCluster);
    }

    public double getMacroF1() {
        return ExternalEvaluation.getMacroMeasure(extEvalResult, f1, clusterAssignments, bagsPerCluster);
    }

    public double getMacroSpecificity() {
        return ExternalEvaluation.getMacroMeasure(extEvalResult, specificity, clusterAssignments, bagsPerCluster);
    }

    public String printFullEvaluation() {
        StringBuilder result = new StringBuilder();
        result.append("Evaluation\n----------------\n");

        int totalClusteredInstances = Utils.sum(bagsPerCluster);
        if (totalClusteredInstances > 0) {
            result.append("Clustered Instances:\n");

            int clustFieldWidth = (int) (Math.log(maxNumClusters) / Math.log(10D) + 1D);
            int numInstFieldWidth = (int) (Math.log(clusterAssignments.size()) / Math.log(10D) + 1D);
            for (int i = 0; i < maxNumClusters; ++i) {
                if (bagsPerCluster[i] > 0) {
                    result.append(Utils.doubleToString(i, clustFieldWidth, 0))
                            .append("      ").append(Utils.doubleToString(bagsPerCluster[i], numInstFieldWidth, 0))
                            .append(" (").append(Utils.doubleToString((double) bagsPerCluster[i] / totalClusteredInstances * 100, 3, 0)).append("%)\n");
                }
            }
        }
        if (unclusteredInstances > 0) {
            result.append("Unclustered instances: ").append(unclusteredInstances).append("\n");
        }

        if (classAtt > -1) {
            result.append("Class attribute: \"").append(instances.classAttribute().name()).append("\"\n");
            result.append("Confusion Matrix:\n");
            result.append(PrintConfusionMatrix.severalLines(extEvalResult, bagsPerCluster, instances.classAttribute()));

            if (totalClusteredInstances > 0) {
                int Cwidth = 1 + (int) (Math.log(maxNumClusters) / Math.log(10D));
                result.append("Assignation:\n");
                for (int i = 0; i < maxNumClusters; ++i) {
                    if (bagsPerCluster[i] > 0) {
                        result.append("\tCluster ").append(Utils.doubleToString(i, Cwidth, 0));
                        result.append(" <-- ");
                        if (extEvalResult.getClusterToClass()[i] < 0) {
                            result.append("No class\n");
                        } else {
                            result.append(instances.classAttribute().value(extEvalResult.getClusterToClass()[i])).append("\n");
                        }
                    }
                }
            }
            result.append("Incorrectly clustered instances :\t")
                    .append(extEvalResult.getClusterToClass()[maxNumClusters])
                    .append(" + ")
                    .append(Arrays.toString(extEvalResult.getUnnasigned()))
                    .append("\t(")
                    .append((double) (extEvalResult.getClusterToClass()[maxNumClusters] + Arrays.stream(extEvalResult.getUnnasigned()).sum()) / instances.numInstances() * 100.0D)
                    .append(" %)\n");
        }

        if (printClusterAssignments)
            result.append(printClusterings());

        result.append("Internal validation metrics:\n");
        result.append("\tRMSSTD: ").append(computedRmssd).append("\n");
        result.append("\tSilhouette index: ").append(computedSilhouette).append("\n");
        result.append("\tXB* index: ").append(computedXb).append("\n");
        result.append("\tDB* index: ").append(computedDb).append("\n");
        result.append("\tS_Dbw index: ").append(computedSdbw).append("\n");
        result.append("\tDBCV: ").append(computedDbcv).append("\n");
        if (classAtt > -1) {
            result.append("External validation metrics:\n");
            result.append("\tEntropy: ").append(entropy).append("\n");
            result.append("\tPurity: ").append(purity).append("\n");
            result.append("\tRand index: ").append(rand).append("\n");
            result.append("\tPrecision: ").append(Arrays.toString(precision)).append("\tMacro: ").append(ExternalEvaluation.getMacroMeasure(extEvalResult, precision, clusterAssignments, bagsPerCluster)).append("\n");
            result.append("\tRecall: ").append(Arrays.toString(recall)).append("\tMacro: ").append(ExternalEvaluation.getMacroMeasure(extEvalResult, recall, clusterAssignments, bagsPerCluster)).append("\n");
            result.append("\tF1 measure: ").append(Arrays.toString(f1)).append("\tMacro: ").append(ExternalEvaluation.getMacroMeasure(extEvalResult, f1, clusterAssignments, bagsPerCluster)).append("\n");
            result.append("\tSpecificity: ").append(Arrays.toString(specificity)).append("\tMacro: ").append(ExternalEvaluation.getMacroMeasure(extEvalResult, specificity, clusterAssignments, bagsPerCluster));
        }

        return result.toString();
    }

    private String printClusterings() {
        Range attributesToOutput = null;
        if (!attributeRangeString.equals("0")) {
            attributesToOutput = new Range(attributeRangeString);
        }

        StringBuilder result = new StringBuilder();
        result.append("Clustering assignments:\n");
        for (int i = 0; i < clusterAssignments.size(); ++i) {
            result.append(i).append(" --> ")
                    .append(clusterAssignments.get(i)).append(" (")
                    .append(attributeValuesString(instances.get(i), attributesToOutput))
                    .append(")\n");
        }
        return result.toString();
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public Instances getInstances() {
        return instances;
    }

    public DatasetCentroids getDatasetCentroids() {
        return datasetCentroids;
    }

    //TODO No adaptado a MI
    private String attributeValuesString(Instance instance, Range attRange) {
        StringBuilder text = new StringBuilder();
        if (attRange != null) {
            boolean firstOutput = true;
            attRange.setUpper(instance.numAttributes() - 1);
            for (int i = 0; i < instance.numAttributes(); ++i) {
                if (attRange.isInRange(i)) {
                    if (firstOutput) {
                        text.append("(");
                    } else {
                        text.append(",");
                    }
                    text.append(instance.toString(i));
                    firstOutput = false;
                }
            }
            if (!firstOutput) {
                text.append(")");
            }
        }
        return text.toString();
    }

    private static String getGlobalInfo(Clusterer clusterer) throws Exception {
        BeanInfo bi = Introspector.getBeanInfo(clusterer.getClass());
        MethodDescriptor[] methods = bi.getMethodDescriptors();
        Object[] args = new Object[0];
        String result = "\nSynopsis for " + clusterer.getClass().getName() + ":\n\n";

        for (MethodDescriptor method : methods) {
            String name = method.getDisplayName();
            Method meth = method.getMethod();
            if (name.equals("globalInfo")) {
                String globalInfo = (String) meth.invoke(clusterer, args);
                result = result + globalInfo;
                break;
            }
        }
        return result;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 14165 $");
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> result = new Vector<>();
        result.addElement(new Option("\tOutput help information", "help", 0, "-h"));
        result.addElement(new Option("\tOutput synopsis for clusterer (use in conjunction with -h)", "synopsis", 0, "-synopsis"));
        result.addElement(new Option("\tSet training file", "t", 1, "-t <path-training-file>"));
        result.addElement(new Option("\tSet test file", "T", 1, "-T <path-test-file>"));
        result.addElement(new Option("\tAlways train the clusterer in batch mode, never incrementally.", "force-batch", 0, "-force-batch"));
        result.addElement(new Option("\tSet model input file", "l", 1, "-l <path-input-file>"));
        result.addElement(new Option("\tSet model output file", "d", 1, "-d <path-output-file>"));
        result.addElement(new Option("\tOutput predictions. Predictions are for training file\n\tif only training file is specified," +
                "\n\totherwise predictions are for the test file.\n\tThe range specifies attribute values to be output" +
                "\n\twith the predictions.", "p", 1, "-p <attribute-range>"));
        result.addElement(new Option("\tSet cross validation (only applied to Distribution Clusterers", "x", 0, "-x"));
        result.addElement(new Option("\tSet the seed for randomizing the data in cross-validation", "s", 1, "-s <seed>"));
        result.addElement(new Option("\tSet class attribute. If supplied, class is ignored" +
                "\n\tduring clustering but is used in a classes to\n\tclusters evaluation.", "c", 1, "-c <class-idx>"));
        result.addElement(new Option("\tSet number of threads to run in parallel", "num-threads", 1, "-num-threads <int>"));
        return result.elements();
    }

    public Enumeration<Option> listOptions(Clusterer clusterer) {
        Vector<Option> result = new Vector<>(Collections.list(listOptions()));
        if (clusterer instanceof Drawable) {
            result.addElement(new Option("\tOutputs the graph representation of the clusterer to the file.", "g", 1, "-g <path-graph-file>"));
        }
        if (clusterer instanceof OptionHandler) {
            result.addAll(Collections.list(((OptionHandler) clusterer).listOptions()));
        }
        return result.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        boolean help = Utils.getFlag("h", options);
        if (help) {
            System.out.println(listOptions());
            return;
        }
        String datasetPath = Utils.getOption("d", options);
        instances = ProcessDataset.readArff(datasetPath);
        String classString = Utils.getOption("c", options);
        setClass(classString);
        maxNumClusters = Integer.parseInt(Utils.getOption("k", options));
        reuseEvaluator = Utils.getFlag("r", options);
        String distFunctionClass = Utils.getOption("A", options);
        boolean parallelize = Utils.getFlag("parallelize", options);

        attributeRangeString = Utils.getOption("p", options);
        if (attributeRangeString.length() != 0)
            printClusterAssignments = true;

        if (reuseEvaluator) {
            distanceFunction = LoadByName.distanceFunction(distFunctionClass, options);
            rmssd = new RMSStdDev(instances, maxNumClusters, distanceFunction, parallelize);
            sdbw = new S_DbwIndex(instances, maxNumClusters, distanceFunction, parallelize);
            silhouette = new SilhouetteIndex(instances, maxNumClusters, distanceFunction, parallelize);
            xb = new XieBeniIndex(instances, maxNumClusters, distanceFunction, parallelize);
            db = new DaviesBouldinIndex(instances, maxNumClusters, distanceFunction, parallelize);
            dbcv = new DBCV(instances, distanceFunction, maxNumClusters, parallelize);
            if (classAtt > -1)
                extEval = new ExternalEvaluation(instances, maxNumClusters, instances.numDistinctValues(classAtt));
            twcv = new TotalWithinClusterVariation(instances, maxNumClusters, distanceFunction, parallelize);
            ftwcv = new FastTotalWithinClusterValidation(instances, maxNumClusters);
            datasetCentroids = new DatasetCentroids(instances, maxNumClusters, distanceFunction);
        }

        Utils.checkForRemainingOptions(options);
    }

    private void setClass(String classString) {
        if (classString.length() != 0) {
            if (classString.equals("last"))
                classAtt = instances.numAttributes() - 1;
            else if (classString.equals("first"))
                classAtt = 0;
            else {
                classAtt = Integer.parseInt(classString);
            }
            instances.setClassIndex(classAtt);
        } else {
            classAtt = instances.classIndex();
        }
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
