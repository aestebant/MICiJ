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
        }
        else {
            aux = instances;
        }

        clusterer.buildClusterer(aux);

        countInstancesPerCluster(instances);
        maxNumClusters = clusterer.numberOfClusters();
        actualNumClusters = clusterer.numberOfClusters();
        for (int i = 0; i < clusterer.numberOfClusters(); ++i) {
            if (instancesPerCluster[i] == 0)
                actualNumClusters--;
        }

        silhouette = computeSilhouetteIndex(instances);
        sdbw = computeSDbwIndex(instances);

        if (classAtt > -1) {
            this.classEvaluation();
            purity = computePurity();
            rand = computeRandIndex();
        }

    }

    public void setClusterer(Clusterer clusterer, String[] options) {
        this.clusterer = clusterer;
        if (clusterer instanceof OptionHandler) {
            try {
                ((OptionHandler)clusterer).setOptions(options);
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

    private void countInstancesPerCluster(Instances instances)  throws Exception {
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

    private double computeSilhouetteIndex(Instances instances) {
        int numInstances = clusterAssignments.size();

        if (actualNumClusters == 0)
            return -1;

        double[][] distances = computeDistanceMatrix(instances);

        List<Double> silhouette = new ArrayList<>(numInstances);

        for (int point = 0; point < numInstances; ++point) {
            double[] meanDistToCluster = new double[maxNumClusters];
            for (int other = 0; other < numInstances; ++other) {
                if (other != point && clusterAssignments.get(other) != -1)
                    meanDistToCluster[clusterAssignments.get(other)] += distances[point][other];
            }
            for (int c = 0; c < maxNumClusters; ++c) {
                if (c == clusterAssignments.get(point))
                    meanDistToCluster[c] /= (instancesPerCluster[c] - 1);
                else
                    meanDistToCluster[c] /= instancesPerCluster[c];
            }
            double aPoint = 0;
            if (clusterAssignments.get(point) != -1)
                aPoint = meanDistToCluster[clusterAssignments.get(point)];

            List<Double> possibleB = new ArrayList<>(maxNumClusters -1);
            for (int j = 0; j < maxNumClusters; ++j) {
                if (j != clusterAssignments.get(point))
                    possibleB.add(meanDistToCluster[j]);
            }
            double bPoint = 0;
            if (possibleB.size() > 0)
                bPoint = Collections.min(possibleB);

            if (aPoint < bPoint)
                silhouette.add(1 - aPoint / bPoint);
            else if (aPoint == bPoint)
                silhouette.add(0D);
            else if (aPoint > bPoint)
                silhouette.add(bPoint / aPoint - 1);
        }
        OptionalDouble average = silhouette.stream().mapToDouble(a -> a).average();
        return average.isPresent()? average.getAsDouble() : -1;
    }

    private double[][] computeDistanceMatrix(Instances instances) {
        int numInstances = instances.numInstances();

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        Collection<Callable<Double[]>> collection = new ArrayList<>(numInstances);
        for (int i = 0; i < numInstances; ++i) {
            for (int j = i+1; j < numInstances; ++j) {
                collection.add(new Wrapper(instances.get(i), i, instances.get(j), j));
            }
        }
        double[][] distances = new double[numInstances][numInstances];
        try {
            List<Future<Double[]>> futures = executor.invokeAll(collection);
            for (Future<Double[]> future : futures) {
                Double[] result = future.get();
                distances[result[0].intValue()][result[1].intValue()] = result[2];
                distances[result[1].intValue()][result[0].intValue()] = result[2];
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        return distances;
    }

    private class Wrapper implements Callable<Double[]> {
        private Instance a;
        private Instance b;
        private int aIdx;
        private int bIdx;
        Wrapper(Instance a, int aIdx, Instance b, int bIdx) {
            this.a = a;
            this.b = b;
            this.aIdx = aIdx;
            this.bIdx = bIdx;
        }
        @Override
        public Double[] call() throws Exception {
            double distance = distFunction.distance(a,b);
            return new Double[]{(double) aIdx, (double) bIdx, distance};
        }
    }

    private double computeSDbwIndex(Instances instances) {
        // Esta métrica no tiene en cuenta las instancias clasificadas como ruido.
        int numAttributes = instances.get(0).relationalValue(1).numAttributes();
        int numInstances = instances.numInstances();

        if (actualNumClusters == 0)
            return 0D;

        double scat = 0D;
        Map<Integer, Instances> clustersSummary = new HashMap<>(actualNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            clustersSummary.put(i, new Instances(instances.get(0).relationalValue(1), 1));
        }
        for (int i = 0; i < numInstances; ++i) {
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx > -1) {
                Instances bag = instances.get(i).relationalValue(1);
                double[] mean = new double[numAttributes];
                for (int j = 0; j < numAttributes; ++j)
                    mean[j] = bag.meanOrMode(j);
                clustersSummary.get(clusterIdx).add(new DenseInstance(1D, mean));
            }
        }
        Instances datasetSummary = new Instances(instances.get(0).relationalValue(1), numInstances);
        for (int i = 0; i < maxNumClusters; ++i)
            datasetSummary.addAll(clustersSummary.get(i));

        double[] varDataset = datasetSummary.variances();
        double euclidNormDataset = 0D;
        for (double v : varDataset)
            euclidNormDataset += v * v;
        euclidNormDataset = Math.sqrt(euclidNormDataset);

        List<Double> euclidNormCluster = new ArrayList<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            double[] varCluster = clustersSummary.get(i).variances();
            euclidNormCluster.add(0D);
            for (double v : varCluster)
                euclidNormCluster.set(i, euclidNormCluster.get(i) + v*v);
            scat += euclidNormCluster.get(i) / euclidNormDataset;
        }
        scat /= actualNumClusters;

        double stdev = euclidNormCluster.stream().mapToDouble(Double::doubleValue).sum();
        stdev = Math.sqrt(stdev) / actualNumClusters;

        Map<Integer, Instance> clustersCenters = new HashMap<>(actualNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            double[] mean = new double[numAttributes];
            for (int j = 0; j < numAttributes; ++j)
                mean[j] = clustersSummary.get(i).meanOrMode(j);
            clustersCenters.put(i, new DenseInstance(1D, mean));
        }

        double densBw = 0D;
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = 0; j < maxNumClusters; ++j) {
                if (i != j) {
                    double[] means = new double[numAttributes];
                    for (int k = 0; k < numAttributes; ++k)
                        means[k] = (clustersCenters.get(i).value(k) + clustersCenters.get(j).value(k)) / 2;
                    Instance u = new DenseInstance(1D, means);

                    double densityU = 0D;
                    double densityI = 0D;
                    double densityJ = 0D;

                    for (int k = 0; k < numInstances; ++k) {
                        if (clusterAssignments.get(k) == i || clusterAssignments.get(k) == j) {
                            double distance = distFunction.distance(u, instances.get(k));
                            if (distance <= stdev)
                                densityU++;
                            if (clusterAssignments.get(k) == i)
                                densityI++;
                            else
                                densityJ++;
                        }
                    }
                    densBw += densityU / Math.max(densityI, densityJ);
                }
            }
        }
        densBw /= (actualNumClusters * (actualNumClusters -1));

        return scat + densBw;
    }

    private void classEvaluation() throws Exception {
        instances.setClassIndex(classAtt);
        int numInstances = instances.numInstances();

        confusion = new int[maxNumClusters][numClasses];
        clusterTotals = new int[maxNumClusters];
        for(int i = 0; i < numInstances; ++i) {
            Instance instance = instances.get(i);
            if (clusterAssignments.get(i) != -1 && !instance.classIsMissing()) {
                confusion[clusterAssignments.get(i)][(int) instance.classValue()]++;
                clusterTotals[clusterAssignments.get(i)]++;
            }
        }
        double[] best = new double[maxNumClusters + 1];
        best[maxNumClusters] = 1.7976931348623157E308D;
        double[] current = new double[maxNumClusters + 1];
        mapClasses(maxNumClusters, 0, confusion, clusterTotals, current, best, 0);
        classToCluster = new int[maxNumClusters + 1];
        for(int i = 0; i < maxNumClusters + 1; ++i) {
            classToCluster[i] = (int)best[i];
        }
    }

    private double computeRandIndex() {
        double rand = 0;
        for (int i = 0; i < maxNumClusters; ++i) {
            if (classToCluster[i] > -1)
                rand += confusion[i][classToCluster[i]];
        }
        // Al dividir por el nº total de instancias también estamos penalizando si hay algunas clasificadas como ruido
        return rand / instances.numInstances();
    }

    private double computePurity() {
        double purity = 0;
        for(int i = 0; i < maxNumClusters; ++i) {
            if (Arrays.stream(confusion[i]).max().isPresent())
                purity += Arrays.stream(confusion[i]).max().getAsInt();
        }
        // Al dividir por el nº total de instancias también estamos penalizando si hay algunas clasificadas como ruido
        return purity / instances.numInstances();
    }

    private static void mapClasses(int numClusters, int lev, int[][] counts, int[] clusterTotals, double[] current, double[] best, int error) throws Exception {
        if (lev == numClusters) {
            if ((double) error < best[numClusters]) {
                best[numClusters] = error;
                System.arraycopy(current, 0, best, 0, numClusters);
            }
        } else if (clusterTotals[lev] == 0) {
            current[lev] = -1.0D;
            mapClasses(numClusters, lev + 1, counts, clusterTotals, current, best, error);
        } else {
            current[lev] = -1.0D;
            mapClasses(numClusters, lev + 1, counts, clusterTotals, current, best, error + clusterTotals[lev]);
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
                        mapClasses(numClusters, lev + 1, counts, clusterTotals, current, best, error + (clusterTotals[lev] - counts[lev][i]));
                    }
                }
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
                    .append(Utils.doubleToString((double)classToCluster[maxNumClusters] / instances.numInstances() * 100.0D, 8, 4))
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
        for(int i = 0; i < maxNumClusters; ++i) {
            for(int j = 0; j < numClasses; ++j) {
                if (confusion[i][j] > maxVal) {
                    maxVal = confusion[i][j];
                }
            }
        }
        int Cwidth = 1 + Math.max((int)(Math.log(maxVal) / Math.log(10D)), (int)(Math.log(actualNumClusters) / Math.log(10D)));

        for(int i = 0; i < maxNumClusters; ++i) {
            if (clusterTotals[i] > 0) {
                matrix.append(" ").append(Utils.doubleToString(i, Cwidth, 0));
            }
        }
        matrix.append("  <-- assigned to cluster\n");

        for(int i = 0; i < numClasses; ++i) {
            for(int j = 0; j < maxNumClusters; ++j) {
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
        for(int i = 0; i < clusterAssignments.size(); ++i) {
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
            for(int i = 0; i < instance.numAttributes(); ++i) {
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
