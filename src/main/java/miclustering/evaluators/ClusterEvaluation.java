package miclustering.evaluators;

import miclustering.algorithms.MIClusterer;
import miclustering.utils.PrintConfusionMatrix;
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
    private Clusterer clusterer;
    private DistanceFunction distFunction;
    private Instances instances;

    private List<Integer> clusterAssignments;
    private int[] bagsPerCluster;
    private int unclusteredInstances;
    private int actualNumClusters;

    private String classString;
    private int classAtt;

    private double silhouette;
    private double sdbw;
    private double dbcv;
    private ClassEvalResult classEvalResult;
    private double purity;
    private double rand;
    private double[] precision;
    private double[] recall;
    private double[] f1;
    private double[] specificity;

    private boolean printClusterAssignments;
    private String attributeRangeString;
    private int numThreads;
    private ClassEvaluation ce;

    public void evaluateClusterer(Instances data) throws Exception {
        if (clusterer == null) {
            throw new Exception("Clusterer not set");
        }

        setClass(data);
        instances = data;
        distFunction = ((MIClusterer) clusterer).getDistanceFunction();

        Instances processData;
        if (classAtt > -1) {
            Remove removeClass = new Remove();
            removeClass.setAttributeIndices(String.valueOf(classAtt + 1)); // No sé por qué aquí cuenta los índices empezando en 1
            removeClass.setInvertSelection(false);
            try {
                removeClass.setInputFormat(data);
            } catch (Exception e) {
                e.printStackTrace();
            }
            processData = Filter.useFilter(data, removeClass);
        } else {
            processData = new Instances(data);
        }

        clusterer.buildClusterer(processData);
        int maxNumClusters = clusterer.numberOfClusters();
        countInstancesPerCluster(clusterer, processData);
        actualNumClusters = countRealClusters(maxNumClusters, bagsPerCluster);

        SilhouetteIndex s = new SilhouetteIndex(processData, maxNumClusters, distFunction, numThreads);
        silhouette = s.computeIndex(clusterAssignments, bagsPerCluster);
        S_DbwIndex sDbw = new S_DbwIndex(processData, maxNumClusters, distFunction);
        sdbw = sDbw.computeIndex(clusterAssignments);
        DBCV densityBCV = new DBCV(processData, distFunction, maxNumClusters, numThreads);
        dbcv = densityBCV.computeIndex(clusterAssignments, bagsPerCluster);

        if (classAtt > -1) {
            ce = new ClassEvaluation(data, maxNumClusters, data.numClasses());
            classEvalResult = ce.computeConfusionMatrix(clusterAssignments, bagsPerCluster);
            purity = ce.computePurity(classEvalResult.getConfMatrix());
            rand = ce.computeRandIndex(classEvalResult);
            precision = ce.computePrecision(classEvalResult);
            recall = ce.computeRecall(classEvalResult);
            f1 = ce.computeF1(classEvalResult, precision, recall);
            specificity = ce.computeSpecificity(classEvalResult);
        }

    }

    private void setClass(Instances data) {
        if (classString.length() != 0) {
            if (classString.equals("last"))
                classAtt = data.numAttributes() - 1;
            else if (classString.equals("first"))
                classAtt = 0;
            else {
                classAtt = Integer.parseInt(classString);
            }
            data.setClassIndex(classAtt);
        } else {
            classAtt = data.classIndex();
        }
    }

    public void setClusterer(Clusterer clusterer, String[] options) {
        this.clusterer = clusterer;
        if (clusterer instanceof OptionHandler) {
            try {
                ((OptionHandler) clusterer).setOptions(options);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void countInstancesPerCluster(Clusterer clusterer, Instances instances) {
        int maxNumClusters = 0;
        try {
            maxNumClusters = clusterer.numberOfClusters();
        } catch (Exception e) {
            e.printStackTrace();
        }
        bagsPerCluster = new int[maxNumClusters];
        clusterAssignments = new ArrayList<>(instances.numInstances());
        for (Instance i : instances) {
            try {
                int clusterIdx = clusterer.clusterInstance(i);
                bagsPerCluster[clusterIdx]++;
                clusterAssignments.add(clusterIdx);
            } catch (Exception e) { // Unclustered instance
                unclusteredInstances++;
                clusterAssignments.add(-1);
            }
        }
    }

    private int countRealClusters(int maxNumClusters, int[] instancesPerCluster) {
        int actualNumClusters = maxNumClusters;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (instancesPerCluster[i] == 0)
                actualNumClusters--;
        }
        return actualNumClusters;
    }

    @Override
    public String toString() {
        if (clusterer == null) {
            return "No clusterer built yet";
        }

        StringBuilder result = new StringBuilder();

        result.append("Algorithm: ").append(clusterer.getClass().getName()).append("\n");
        result.append("Dataset: ").append(instances.relationName()).append("\n");
        result.append("Evaluation\n----------------\n");

        int maxNumClusters = bagsPerCluster.length;
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
            result.append(PrintConfusionMatrix.severalLines(classEvalResult, bagsPerCluster, instances.classAttribute()));

            if (totalClusteredInstances > 0) {
                int Cwidth = 1 + (int) (Math.log(maxNumClusters) / Math.log(10D));
                result.append("Assignation:\n");
                for (int i = 0; i < maxNumClusters; ++i) {
                    if (bagsPerCluster[i] > 0) {
                        result.append("\tCluster ").append(Utils.doubleToString(i, Cwidth, 0));
                        result.append(" <-- ");
                        if (classEvalResult.getClusterToClass()[i] < 0) {
                            result.append("No class\n");
                        } else {
                            result.append(instances.classAttribute().value(classEvalResult.getClusterToClass()[i])).append("\n");
                        }
                    }
                }
            }
            result.append("Incorrectly clustered instances :\t")
                    .append(classEvalResult.getClusterToClass()[maxNumClusters])
                    .append("\t(")
                    .append((double) classEvalResult.getClusterToClass()[maxNumClusters] / instances.numInstances() * 100.0D)
                    .append(" %)\n");
        }

        if (printClusterAssignments)
            result.append(printClusterings());

        result.append("Internal validation metrics:\n");
        result.append("\tSilhouette index: ").append(silhouette).append("\n");
        result.append("\tS_Dbw index: ").append(sdbw).append("\n");
        result.append("\tDBCV: ").append(dbcv).append("\n");
        if (classAtt > -1) {
            result.append("External validation metrics:\n");
            result.append("\tPurity: ").append(purity).append("\n");
            result.append("\tRand index: ").append(rand).append("\n");
            result.append("\tPrecision: ").append(Arrays.toString(precision)).append("\tMacro: ").append(ce.getMacroMeasure(classEvalResult, precision, clusterAssignments, bagsPerCluster)).append("\n");
            result.append("\tRecall: ").append(Arrays.toString(recall)).append("\tMacro: ").append(ce.getMacroMeasure(classEvalResult, recall, clusterAssignments, bagsPerCluster)).append("\n");
            result.append("\tF1 measure: ").append(Arrays.toString(f1)).append("\tMacro: ").append(ce.getMacroMeasure(classEvalResult, f1, clusterAssignments, bagsPerCluster)).append("\n");
            result.append("\tSpecificity: ").append(Arrays.toString(specificity)).append("\tMacro: ").append(ce.getMacroMeasure(classEvalResult, specificity, clusterAssignments, bagsPerCluster));
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

    public int getUnclusteredInstances() {
        return unclusteredInstances;
    }

    public int getActualNumClusters() {
        return actualNumClusters;
    }

    public double getSilhouette() {
        return silhouette;
    }

    public double getSdbw() {
        return sdbw;
    }

    public double getDbcv() {
        return dbcv;
    }

    public ClassEvalResult getClassEvalResult() {
        return classEvalResult;
    }

    public double getPurity() {
        return purity;
    }

    public double getRand() {
        return rand;
    }

    public double getMacroPrecision() {
        return ce.getMacroMeasure(classEvalResult, precision, clusterAssignments, bagsPerCluster);
    }

    public double getMacroRecall() {
        return ce.getMacroMeasure(classEvalResult, recall, clusterAssignments, bagsPerCluster);
    }

    public double getMacroF1() {
        return ce.getMacroMeasure(classEvalResult, f1, clusterAssignments, bagsPerCluster);
    }

    public double getSpecificity() {
        return ce.getMacroMeasure(classEvalResult, specificity, clusterAssignments, bagsPerCluster);
    }

    public DistanceFunction getDistanceFunction() {
        return distFunction;
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
        boolean synopsis = Utils.getFlag("synopsis", options);
        if (help) {
            if (synopsis) {
                System.out.println(listOptions(clusterer));
            } else {
                System.out.println(listOptions());
            }
            return;
        }
        classString = Utils.getOption("c", options);
        attributeRangeString = Utils.getOption("p", options);
        String sThreads = Utils.getOption("num-threads", options);
        if (sThreads.length() > 0)
            numThreads = Integer.parseInt(sThreads);
        if (attributeRangeString.length() != 0)
            printClusterAssignments = true;

        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
