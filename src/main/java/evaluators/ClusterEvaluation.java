package evaluators;

import algorithms.MyClusterer;
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
import java.util.concurrent.*;

public class ClusterEvaluation implements Serializable, OptionHandler, RevisionHandler {
    private Clusterer clusterer;
    private DistanceFunction distFunction;
    private Instances instances;

    private Vector<Integer> clusterAssignments;
    private int[] instancesPerCluster;
    private int unclusteredInstances;
    private int maxNumClusters;
    private int actualNumClusters;

    private double silhouette;
    private double sdbw;

    private String classString;
    private int classAtt;
    private int numClasses;
    private int[] classToCluster;
    private int[][] confusion;
    private int[] clusterTotals;
    private double purity;
    private double rand;

    private boolean printClusterAssignments;
    private boolean forceBatch;
    private String attributeRangeString;
    private int numThreads;

    public void evaluateClusterer(Instances data) throws Exception {
        if (clusterer == null) {
            throw new Exception("Clusterer not set");
        }

        instances = new Instances(data);
        setClass(instances);

        distFunction = ((MyClusterer) clusterer).getDistanceFunction();

        Instances aux;
        if (classAtt > -1) {
            Remove removeClass = new Remove();
            removeClass.setAttributeIndices(String.valueOf(classAtt + 1)); // No sé por qué aquí cuenta los índices empezando en 1
            removeClass.setInvertSelection(false);
            try {
                removeClass.setInputFormat(instances);
            } catch (Exception e) {
                e.printStackTrace();
            }
            aux = Filter.useFilter(instances, removeClass);
        } else {
            aux = instances;
        }

        clusterer.buildClusterer(aux);

        countInstancesPerCluster(aux);
        maxNumClusters = clusterer.numberOfClusters();
        actualNumClusters = clusterer.numberOfClusters();
        for (int i = 0; i < clusterer.numberOfClusters(); ++i) {
            if (instancesPerCluster[i] == 0)
                actualNumClusters--;
        }

        SilhouetteIndex s = new SilhouetteIndex(aux, clusterAssignments, maxNumClusters, actualNumClusters, distFunction, instancesPerCluster, numThreads);
        silhouette = s.computeIndex();
        S_DbwIndex sDbw = new S_DbwIndex(aux, actualNumClusters, maxNumClusters, clusterAssignments, distFunction);
        sdbw = sDbw.computeIndex();

        if (classAtt > -1) {
            ClassEvaluation ce = new ClassEvaluation(instances, maxNumClusters, numClasses, clusterAssignments);
            confusion = ce.computeEval();
            clusterTotals = ce.getClusterTotals();
            classToCluster = ce.getClassToCluster();
            purity = ce.computePurity();
            rand = ce.computeRandIndex();
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
        numClasses = data.numDistinctValues(classAtt);
    }

    private void countInstancesPerCluster(Instances instances) throws Exception {
        int numClusters = clusterer.numberOfClusters();
        instancesPerCluster = new int[numClusters];
        clusterAssignments = new Vector<>(instances.numInstances());

        for (Instance i : instances) {
            try {
                int clusterIdx = clusterer.clusterInstance(i);
                instancesPerCluster[clusterIdx]++;
                clusterAssignments.add(clusterIdx);
            } catch (Exception e) { // Unclustered instance
                unclusteredInstances++;
                clusterAssignments.add(-1);
            }
        }
    }

    @Override
    public String toString() {
        if (clusterer == null) {
            return "No clusterer built yet";
        }

        StringBuilder result = new StringBuilder();

        int totalClusteredInstances = Utils.sum(instancesPerCluster);
        if (totalClusteredInstances > 0) {
            result.append("Clustered Instances:\n");

            int clustFieldWidth = (int) (Math.log(maxNumClusters) / Math.log(10D) + 1D);
            int numInstFieldWidth = (int) (Math.log(clusterAssignments.size()) / Math.log(10D) + 1D);
            for (int i = 0; i < maxNumClusters; ++i) {
                if (instancesPerCluster[i] > 0) {
                    result.append(Utils.doubleToString(i, clustFieldWidth, 0))
                            .append("      ").append(Utils.doubleToString(instancesPerCluster[i], numInstFieldWidth, 0))
                            .append(" (").append(Utils.doubleToString((double) instancesPerCluster[i] / totalClusteredInstances * 100, 3, 0)).append("%)\n");
                }
            }
        }
        if (unclusteredInstances > 0) {
            result.append("\nUnclustered instances: ").append(unclusteredInstances);
        }

        result.append("\nSilhouette index: ").append(silhouette);
        result.append("\nS_Dbw index: ").append(sdbw);

        if (classAtt > -1) {
            result.append("\n\nClass attribute: \"").append(instances.classAttribute().name()).append("\"\n");
            result.append("Classes to Clusters:\n");
            result.append(printConfusionMatrix(clusterTotals, new Instances(instances, 0))).append("\n");

            int Cwidth = 1 + (int) (Math.log(maxNumClusters) / Math.log(10D));
            for (int i = 0; i < maxNumClusters; ++i) {
                if (clusterTotals[i] > 0) {
                    result.append("Cluster ").append(Utils.doubleToString(i, Cwidth, 0));
                    result.append(" <-- ");
                    if (classToCluster[i] < 0) {
                        result.append("No class\n");
                    } else {
                        result.append(instances.classAttribute().value(classToCluster[i])).append("\n");
                    }
                }
            }

            result.append("\nIncorrectly clustered instances :\t")
                    .append(classToCluster[maxNumClusters]).append("\t(")
                    .append(Utils.doubleToString((double) classToCluster[maxNumClusters] / instances.numInstances() * 100.0D, 8, 4))
                    .append(" %)\n");

            result.append("Purity: ").append(purity).append("\n");
            result.append("Rand index: ").append(rand).append("\n");
        }
        if (printClusterAssignments)
            result.append(printClusterings());

        return result.toString();
    }

    private String printConfusionMatrix(int[] clusterTotals, Instances inst) {
        StringBuilder matrix = new StringBuilder();

        int maxVal = 0;
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                if (confusion[i][j] > maxVal) {
                    maxVal = confusion[i][j];
                }
            }
        }
        int Cwidth = 1 + Math.max((int) (Math.log(maxVal) / Math.log(10D)), (int) (Math.log(actualNumClusters) / Math.log(10D)));

        for (int i = 0; i < maxNumClusters; ++i) {
            if (clusterTotals[i] > 0) {
                matrix.append(" ").append(Utils.doubleToString(i, Cwidth, 0));
            }
        }
        matrix.append("  <-- assigned to cluster\n");

        for (int i = 0; i < numClasses; ++i) {
            for (int j = 0; j < maxNumClusters; ++j) {
                if (clusterTotals[j] > 0)
                    matrix.append(" ").append(Utils.doubleToString(confusion[j][i], Cwidth, 0));
            }
            matrix.append(" | ").append(inst.classAttribute().value(i)).append("\n");
        }
        return matrix.toString();
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

    public String getConfussion() {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = 0; j < numClasses; ++j)
                result.append(confusion[i][j]).append(" ");
            result.append("|");
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

    public double getPurity() {
        return purity;
    }

    public double getRand() {
        return rand;
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
        result.addElement(new Option("\tOutput predictions. Predictions are for training file\n\tif only training file is specified,\n\totherwise predictions are for the test file.\n\tThe range specifies attribute values to be output\n\twith the predictions.", "p", 1, "-p <attribute-range>"));
        result.addElement(new Option("\tSet cross validation (only applied to Distribution Clusterers", "x", 0, "-x"));
        result.addElement(new Option("\tSet the seed for randomizing the data in cross-validation", "s", 1, "-s <seed>"));
        result.addElement(new Option("\tSet class attribute. If supplied, class is ignored\n\tduring clustering but is used in a classes to\n\tclusters evaluation.", "c", 1, "-c <class-idx>"));
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
        forceBatch = Utils.getFlag("force-batch", options);
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
