package miclustering.algorithms;

import miclustering.distances.HausdorffDistance;
import miclustering.utils.DatasetCentroids;
import miclustering.utils.LoadByName;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class MIKMeans extends RandomizableClusterer implements MIClusterer, NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {
    int numClusters = 2;
    protected int currentNClusters;
    private int maxIterations = 500;
    private int initializationMethod = 0;
    DistanceFunction distFunction = new HausdorffDistance();
    private DatasetCentroids datasetCentroids;
    private boolean showStdDevs = false;
    protected boolean parallelize = true;

    Instances startingPoints;
    Map<Integer, Instance> centroids;
    private List<Integer> clusterAssignments;
    private Instances clusterStdDevs;
    private double[] fullMeans;
    private double[] fullStdDevs;
    private double[] clustersSize;
    private int iterations = 0;
    private double elapsedTime;
    private Instances instances;
    private OneStepKMeans oneStepKMeans;

    public MIKMeans() {
        this.m_SeedDefault = 10;
        this.setSeed(this.m_SeedDefault);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);
        result.enable(Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        return result;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        this.getCapabilities().testWithFail(data);

        long startTime = System.currentTimeMillis();
        int numInstAttributes = data.get(0).relationalValue(1).numAttributes();
        instances = data;

        Instances aux = new Instances(instances.get(0).relationalValue(1));
        for (int i = 1; i < instances.size(); ++i) {
            aux.addAll(instances.get(i).relationalValue(1));
        }
        if (showStdDevs) {
            fullStdDevs = aux.variances();
        }

        this.fullMeans = new double[numInstAttributes];
        for (int i = 0; i < numInstAttributes; ++i)
            fullMeans[i] = aux.meanOrMode(i);

        if (showStdDevs) {
            for (int i = 0; i < numInstAttributes; ++i) {
                this.fullStdDevs[i] = Math.sqrt(fullStdDevs[i]);
            }
        }

        datasetCentroids = new DatasetCentroids(instances, numClusters, distFunction);

        clusterAssignments = new ArrayList<>(instances.numInstances());
        Instances initBags = new Instances(instances);
        if (this.initializationMethod == 0) {
            randomInit(initBags);
        }

        currentNClusters = centroids.size();

        oneStepKMeans = new OneStepKMeans(instances, distFunction, numClusters, true);

        this.iterations = 0;
        int index;
        boolean converged = false;
        Instances[] bagsPerCluster = new Instances[currentNClusters];
        while (!converged) {
            this.iterations++;
            List<Integer> newAssignation = oneStepKMeans.assignBagsToClusters(centroids, parallelize);
            converged = clusterAssignments.equals(newAssignation);
            clusterAssignments = newAssignation;

            for (int cluster = 0; cluster < currentNClusters; ++cluster) {
                bagsPerCluster[cluster] = new Instances(instances, 0);
            }
            for (int i = 0; i < instances.numInstances(); ++i) {
                bagsPerCluster[clusterAssignments.get(i)].add(instances.instance(i));
            }

            int emptyClusterCount = computeCentroids(bagsPerCluster);

            if (iterations == maxIterations) {
                converged = true;
            }

            if (emptyClusterCount > 0) {
                currentNClusters -= emptyClusterCount;
                if (!converged) {
                    bagsPerCluster = new Instances[currentNClusters];
                } else {
                    Instances[] aux2 = new Instances[currentNClusters];
                    index = 0;
                    int j = 0;
                    while (true) {
                        if (j >= bagsPerCluster.length) {
                            bagsPerCluster = aux2;
                            break;
                        }
                        if (bagsPerCluster[j].numInstances() > 0) {
                            aux2[index] = bagsPerCluster[j];
                            ++index;
                        }
                        ++j;
                    }
                }
            }
            System.gc();
        }
        clusterAssignments = oneStepKMeans.evaluate(clusterAssignments, parallelize);

        if (showStdDevs) {
            clusterStdDevs = new Instances(instances.get(0).relationalValue(1), currentNClusters);
        }
        clustersSize = new double[currentNClusters];
        for (int i = 0; i < currentNClusters; ++i) {
            if (showStdDevs) {
                Instances instancesPerCluster = new Instances(bagsPerCluster[i].get(0).relationalValue(1));
                for (int j = 1; j < bagsPerCluster[i].numInstances(); ++j)
                    instancesPerCluster.addAll(bagsPerCluster[i].get(j).relationalValue(1));

                double[] variances = instancesPerCluster.variances();
                for (index = 0; index < numInstAttributes; ++index) {
                    variances[index] = Math.sqrt(variances[index]);
                }
                clusterStdDevs.add(new DenseInstance(1D, variances));
            }
            clustersSize[i] = bagsPerCluster[i].numInstances();
        }

        long finishTime = System.currentTimeMillis();
        elapsedTime = (double) (finishTime - startTime) / 1000.0D;
    }

    protected void randomInit(Instances data) throws Exception {
        centroids = new HashMap<>(numClusters);

        Random random = new Random(this.getSeed());
        Map<DecisionTableHashKey, Integer> initialClusters = new HashMap<>();
        int numInstAttributes = data.get(0).relationalValue(1).numAttributes();

        int clusterIdx = 0;
        for (int i = data.numInstances() - 1; i >= 0; --i) {
            int bagIdx = random.nextInt(i + 1);
            DecisionTableHashKey hk = new DecisionTableHashKey(data.get(bagIdx), data.numAttributes(), true);
            if (!initialClusters.containsKey(hk)) {
                double[] mean = new double[numInstAttributes];
                for (int j = 0; j < numInstAttributes; ++j)
                    mean[j] = data.get(bagIdx).relationalValue(1).meanOrMode(j);
                Instance centroid = new DenseInstance(1D, mean);
                centroids.put(clusterIdx, centroid);
                clusterIdx++;
                initialClusters.put(hk, null);
            }
            data.swap(i, bagIdx);
            if (centroids.size() == numClusters) {
                break;
            }
        }
        startingPoints = new Instances(data.get(0).relationalValue(1), numClusters);
        for (int i = 0; i < numClusters; ++i)
            startingPoints.add(centroids.get(i));
    }

    protected int computeCentroids(Instances[] clusters) {
        int emptyClusterCount = 0;
        centroids = new HashMap<>(numClusters);

        ExecutorService executor;
        if (parallelize)
            executor = Executors.newFixedThreadPool((int) (0.5 * Runtime.getRuntime().availableProcessors()));
        else
            executor = Executors.newFixedThreadPool(1);
        Collection<Callable<Map<Integer, Instance>>> collection = new ArrayList<>(numClusters);
        for (int i = 0; i < currentNClusters; ++i) {
            if (clusters[i].numInstances() == 0)
                emptyClusterCount++;
            else
                collection.add(new ParallelizeComputeCentroids(i, clusters[i]));
        }
        try {
            List<Future<Map<Integer, Instance>>> futures = executor.invokeAll(collection);
            for (Future<Map<Integer, Instance>> future : futures) {
                Map<Integer, Instance> result = future.get();
                centroids.putAll(result);
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        return emptyClusterCount;
    }

    private class ParallelizeComputeCentroids implements Callable<Map<Integer, Instance>> {
        Instances cluster;
        Integer idx;
        ParallelizeComputeCentroids(Integer idx, Instances cluster) {
            this.idx = idx;
            this.cluster = cluster;
        }
        @Override
        public Map<Integer, Instance> call() throws Exception {
            //Instance centroid = computeCentroid(cluster);
            Instance centroid = datasetCentroids.computeCentroid(cluster);
            centroid.setDataset(cluster.get(0).relationalValue(1));
            Map<Integer, Instance> result = new HashMap<>(1);
            result.put(idx, centroid);
            return result;
        }
    }

    /*protected Instance computeCentroid(Instances members) {
        Instance centroid = datasetCentroids.computeCentroid(members);
        centroid.setDataset(members.get(0).relationalValue(1));
        return centroid;
    }*/

    @Override
    public int clusterInstance(Instance bag) {
        if (instances.contains(bag))
            return clusterAssignments.get(bag.dataset().indexOf(bag));
        else {
            double[] distances = oneStepKMeans.computeAssignation(centroids, bag);
            Min getMin = new Min();
            double min = getMin.evaluate(distances);
            return Arrays.stream(distances).boxed().collect(Collectors.toList()).indexOf(min);
        }
    }

    @Override
    public int numberOfClusters() throws Exception {
        return numClusters;
    }

    @Override
    public double getElapsedTime() {
        return elapsedTime;
    }

    @Override
    public List<Integer> getClusterAssignments() {
        return clusterAssignments;
    }

    //TODO No adaptado a MI
    public Enumeration<Option> listOptions() {
        Vector<Option> result = new Vector<>();
        result.addElement(new Option("\tNumber of clusters.\n\t(default 2).", "N", 1, "-N <num>"));
        result.addElement(new Option("\tInitialization method to use.\n\t0 = random, 1 = k-means++, 2 = canopy, 3 = farthest first.\n\t(default = 0)", "init", 1, "-init"));
        result.addElement(new Option("\tUse canopies to reduce the number of distance calculations.", "C", 0, "-C"));
        result.addElement(new Option("\tMaximum number of candidate canopies to retain in memory\n\tat any one time when using canopy clustering.\n\tT2 distance plus, data characteristics,\n\twill determine how many candidate canopies are formed before\n\tperiodic and final pruning are performed, which might result\n\tin exceess memory consumption. This setting avoids large numbers\n\tof candidate canopies consuming memory. (default = 100)", "-max-candidates", 1, "-max-candidates <num>"));
        result.addElement(new Option("\tHow often to prune low density canopies when using canopy clustering. \n\t(default = every 10,000 training instances)", "periodic-pruning", 1, "-periodic-pruning <num>"));
        result.addElement(new Option("\tMinimum canopy density, when using canopy clustering, below which\n\t a canopy will be pruned during periodic pruning. (default = 2 instances)", "min-density", 1, "-min-density"));
        result.addElement(new Option("\tThe T2 distance to use when using canopy clustering. Values < 0 indicate that\n\ta heuristic based on attribute std. deviation should be used to set this.\n\t(default = -1.0)", "t2", 1, "-t2"));
        result.addElement(new Option("\tThe T1 distance to use when using canopy clustering. A value < 0 is taken as a\n\tpositive multiplier for T2. (default = -1.5)", "t1", 1, "-t1"));
        result.addElement(new Option("\tDisplay std. deviations for centroids.\n", "V", 0, "-V"));
        result.addElement(new Option("\tDon't replace missing values with mean/mode.\n", "M", 0, "-M"));
        result.add(new Option("\tDistance function to use.\n\t(default: HausdorffDistance)", "A", 1, "-A <classname and options>"));
        result.add(new Option("\tMaximum number of iterations.\n", "I", 1, "-I <num>"));
        result.addElement(new Option("\tPreserve order of instances.\n", "O", 0, "-O"));
        result.addElement(new Option("\tEnables faster distance calculations, using cut-off values.\n\tDisables the calculation/output of squared errors/miclustering.distances.\n", "fast", 0, "-fast"));
        result.addElement(new Option("\tNumber of execution slots.\n\t(default 1 - i.e. no parallelism)", "num-slots", 1, "-num-slots <num>"));
        result.addAll(Collections.list(super.listOptions()));
        return result.elements();
    }

    public String numClustersTipText() {
        return "set number of clusters";
    }

    public void setNumClusters(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Number of clusters must be > 0");
        } else {
            numClusters = n;
        }
    }

    public String initializationMethodTipText() {
        return "The initialization method to use. Random, k-means++, Canopy or farthest first";
    }

    public void setInitializationMethod(Integer selection) {
        this.initializationMethod = selection;
    }

    public int getInitializationMethod() {
        return this.initializationMethod;
    }

    public String maxIterationsTipText() {
        return "set maximum number of iterations";
    }

    public void setMaxIterations(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Maximum number of iterations must be > 0");
        } else {
            this.maxIterations = n;
        }
    }

    public int getMaxIterations() {
        return this.maxIterations;
    }

    public String displayStdDevsTipText() {
        return "Display std deviations of numeric attributes and counts of nominal attributes.";
    }

    public void setDisplayStdDevs(boolean stdD) {
        this.showStdDevs = stdD;
    }

    public boolean getDisplayStdDevs() {
        return this.showStdDevs;
    }

    public String dontReplaceMissingValuesTipText() {
        return "Replace missing values globally with mean/mode.";
    }

    public String distanceFunctionTipText() {
        return "The distance function to use for instances comparison (default: weka.core.EuclideanDistance). ";
    }

    public DistanceFunction getDistanceFunction() {
        return this.distFunction;
    }

    public String fastDistanceCalcTipText() {
        return "Uses cut-off values for speeding up distance calculation, but suppresses also the calculation and output of the within cluster sum of squared errors/sum of miclustering.distances.";
    }

    public String numExecutionSlotsTipText() {
        return "The number of execution slots (threads) to use. Set equal to the number of available cpu/cores";
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        this.showStdDevs = Utils.getFlag("V", options);

        String initM = Utils.getOption("init", options);
        if (initM.length() > 0) {
            this.setInitializationMethod(Integer.parseInt(initM));
        }

        String n = Utils.getOption('N', options);
        if (n.length() != 0) {
            this.setNumClusters(Integer.parseInt(n));
        }

        String i = Utils.getOption("I", options);
        if (i.length() != 0) {
            this.setMaxIterations(Integer.parseInt(i));
        }

        String distFunctionClass = Utils.getOption('A', options);
        distFunction = LoadByName.distanceFunction(distFunctionClass, options);

        parallelize = Utils.getFlag("parallelize", options);

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        result.add("-init");
        result.add(String.valueOf(this.getInitializationMethod()));
        if (this.showStdDevs) {
            result.add("-V");
        }
        result.add("-N" + numClusters);
        result.add("-A");
        result.add((distFunction.getClass().getName() + " " + Utils.joinOptions(distFunction.getOptions())).trim());
        result.add("-I");
        result.add(String.valueOf(this.getMaxIterations()));
        Collections.addAll(result, super.getOptions());
        return result.toArray(new String[0]);
    }

    public String toString() {
        if (this.centroids == null) {
            return "No clusterer built yet!";
        }

        StringBuilder result = new StringBuilder();

        result.append("\nNumber of iterations: ").append(this.iterations).append("\n");
        result.append("Distance-type: ").append(distFunction).append("\n");
        result.append("Initial starting points (");
        switch (this.initializationMethod) {
            case 0:
                result.append("random");
                break;
            case 1:
                result.append("k-means++");
                break;
            case 2:
                result.append("canopy");
                break;
            case 3:
                result.append("farthest first");
                break;
        }
        result.append("):\n");

        if (this.initializationMethod != 2) {
            for (int i = 0; i < this.startingPoints.numInstances(); ++i) {
                result.append("\tCluster ").append(i).append(": ").append(this.startingPoints.instance(i)).append("\n");
            }
        }

        result.append("Final cluster centroids:\n");
        for (int i = 0; i < centroids.size(); ++i) {
            result.append("\tCluster ").append(i).append(": ").append(centroids.get(i)).append("\n");
        }
        DecimalFormat decimalFormat = new DecimalFormat(".##");
        result.append("Elapsed time: ").append(decimalFormat.format(elapsedTime)).append("\n");

        result.append(printSurvey());

        return result.toString();
    }

    private String pad(String source, String padChar, int length, boolean leftPad) {
        StringBuilder temp = new StringBuilder();
        if (leftPad) {
            for (int i = 0; i < length; ++i) {
                temp.append(padChar);
            }
            temp.append(source);
        } else {
            temp.append(source);
            for (int i = 0; i < length; ++i) {
                temp.append(padChar);
            }
        }
        return temp.toString();
    }

    private String printSurvey() {
        StringBuilder result = new StringBuilder();

        Instances clustersToPrint;
        int numAttributes;
        if (centroids.entrySet().iterator().next().getValue().attribute(1).isRelationValued()) {
            clustersToPrint = new Instances(centroids.entrySet().iterator().next().getValue().relationalValue(1), numClusters);
            numAttributes = clustersToPrint.numAttributes();
            for (Map.Entry<Integer, Instance> entry : centroids.entrySet()) {
                double[] mean = new double[numAttributes];
                for (int j = 0; j < numAttributes; ++j) {
                    mean[j] = entry.getValue().relationalValue(1).meanOrMode(j);
                }
                clustersToPrint.add(new DenseInstance(1D, mean));
            }
        }
        else {
            clustersToPrint = new Instances(centroids.entrySet().iterator().next().getValue().dataset(), numClusters);
            for (Map.Entry<Integer, Instance> entry : centroids.entrySet()) {
                clustersToPrint.add(entry.getValue());
            }
            numAttributes = clustersToPrint.numAttributes();
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        String clustNum;
        String plusMinus = "+/-";

        boolean containsNumeric = false;
        int maxV;
        for (int i = 0; i < currentNClusters; ++i) {
            for (maxV = 0; maxV < numAttributes; ++maxV) {
                if (clustersToPrint.attribute(maxV).name().length() > maxAttWidth) {
                    maxAttWidth = clustersToPrint.attribute(maxV).name().length();
                }
                if (clustersToPrint.attribute(maxV).isNumeric()) {
                    containsNumeric = true;
                    double width = Math.log(Math.abs(clustersToPrint.instance(i).value(maxV))) / Math.log(10.0D);
                    if (width < 0.0D) {
                        width = 1.0D;
                    }
                    width += 6.0D;
                    if ((int) width > maxWidth) {
                        maxWidth = (int) width;
                    }
                }
            }
        }
        for (int i = 0; i < numAttributes; ++i) {
            if (clustersToPrint.attribute(i).isNominal()) {
                Attribute a = clustersToPrint.attribute(i);
                for (i = 0; i < clustersToPrint.numInstances(); ++i) {
                    clustNum = a.value((int) clustersToPrint.instance(i).value(i));
                    if (clustNum.length() > maxWidth) {
                        maxWidth = clustNum.length();
                    }
                }
                for (i = 0; i < a.numValues(); ++i) {
                    clustNum = a.value(i) + " ";
                    if (clustNum.length() > maxAttWidth) {
                        maxAttWidth = clustNum.length();
                    }
                }
            }
        }
        double[] auxSize = clustersSize;
        maxV = auxSize.length;
        String strVal;
        for (int i = 0; i < maxV; ++i) {
            double m_ClusterSize = auxSize[i];
            strVal = "(" + m_ClusterSize + ")";
            if (strVal.length() > maxWidth) {
                maxWidth = strVal.length();
            }
        }
        if (showStdDevs && maxAttWidth < "missing".length()) {
            maxAttWidth = "missing".length();
        }
        maxAttWidth += 2;
        if (showStdDevs && containsNumeric) {
            maxWidth += plusMinus.length();
        }
        if (maxAttWidth < "Attribute".length() + 2) {
            maxAttWidth = "Attribute".length() + 2;
        }
        if (maxWidth < "Full Data".length()) {
            maxWidth = "Full Data".length() + 1;
        }

        result.append(this.pad("Cluster#", " ", maxAttWidth + maxWidth * 2 + 2 - "Cluster#".length(), true)).append("\n");
        result.append(this.pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));
        result.append(this.pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

        for (int i = 0; i < currentNClusters; ++i) {
            clustNum = String.valueOf(i);
            result.append(this.pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        result.append("\n");
        String cSize = "(" + Utils.sum(clustersSize) + ")";
        result.append(this.pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(), true));

        for (int i = 0; i < currentNClusters; ++i) {
            cSize = "(" + this.clustersSize[i] + ")";
            result.append(this.pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
        }
        result.append("\n");
        result.append(this.pad("", "=", maxAttWidth + maxWidth * (currentNClusters + 1) + currentNClusters + 1, true));
        result.append("\n");

        for (int i = 0; i < numAttributes; ++i) {
            String attName = clustersToPrint.attribute(i).name();
            result.append(attName);

            for (int j = 0; j < maxAttWidth - attName.length(); ++j) {
                result.append(" ");
            }

            String meanFull = Utils.doubleToString(this.fullMeans[i], maxWidth, 4).trim();
            result.append(this.pad(meanFull, " ", maxWidth + 1 - meanFull.length(), true));

            for (int j = 0; j < currentNClusters; ++j) {
                String meanCluster = Utils.doubleToString(clustersToPrint.instance(j).value(i), maxWidth, 4).trim();
                result.append(this.pad(meanCluster, " ", maxWidth + 1 - meanCluster.length(), true));
            }
            result.append("\n");

            if (this.showStdDevs) {
                String stdDevFull = plusMinus + Utils.doubleToString(this.fullStdDevs[i], maxWidth, 4).trim();
                result.append(this.pad(stdDevFull, " ", maxWidth + maxAttWidth + 1 - stdDevFull.length(), true));

                for (int j = 0; j < currentNClusters; ++j) {
                    String stdDevCluster = plusMinus + Utils.doubleToString(clusterStdDevs.instance(j).value(i), maxWidth, 4).trim();
                    result.append(this.pad(stdDevCluster, " ", maxWidth + 1 - stdDevCluster.length(), true));
                }
                result.append("\n\n");
            }
        }
        return  result.toString();
    }

    public Instances getClusterStandardDevs() {
        return this.clusterStdDevs;
    }

    public double[] getClusterSizes() {
        return this.clustersSize;
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "D. Arthur and S. Vassilvitskii");
        result.setValue(Field.TITLE, "k-means++: the advantages of carefull seeding");
        result.setValue(Field.BOOKTITLE, "Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete miclustering.algorithms");
        result.setValue(Field.YEAR, "2007");
        result.setValue(Field.PAGES, "1027-1035");
        return result;
    }

    public String globalInfo() {
        return "Cluster data using the k means algorithm. Can use either the Euclidean distance (default) or the Manhattan distance. If the Manhattan distance is used, then centroids are computed as the component-wise median rather than mean. For more information see:\n\n" + this.getTechnicalInformation().toString();
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 11444 $");
    }
}
