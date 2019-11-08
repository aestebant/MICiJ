package algorithms;

import distances.HausdorffDistance;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.Canopy;
import weka.clusterers.FarthestFirst;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.*;

public class MISimpleKMeans extends RandomizableClusterer implements MyClusterer, NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {
    int numClusters = 2;
    private int maxIterations = 500;
    private int initializationMethod = 0;
    DistanceFunction distFunction = new HausdorffDistance();
    int executionSlots = 1;
    private boolean preserveOrder = false;
    private boolean showStdDevs = false;
    private boolean fastDistCalc = false;
    private boolean useCanopi = false;

    Instances startingPoints;
    Instances centroids;
    private Instances clusterStdDevs;
    private double[] fullMeans;
    private double[] fullStdDevs;
    private double[] clustersSize;
    private int iterations = 0;
    private double[] squaredErrors;
    private int[] assignments = null;
    private List<long[]> centroidCanopyAssignments;
    private List<long[]> dataPointCanopyAssignments;
    private Canopy canopyClusters;
    private int maxCanopyCandidates = 100;
    private int periodicPruningRate = 10000;
    private double minClusterDensity = 2.0D;
    private double t2 = -1.0D;
    private double t1 = -1.25D;
    private double elapsedTime;

    public MISimpleKMeans() {
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
        Instances instances = new Instances(data);
        instances.setClassIndex(-1);

        distFunction.setInstances(data);

        Instances aux = new Instances(instances.get(0).relationalValue(1));
        for (int i = 1; i < instances.size(); ++i) {
            aux.addAll(instances.get(i).relationalValue(1));
        }
        if (this.showStdDevs) {
            this.fullStdDevs = aux.variances();
        }

        this.fullMeans = new double[numInstAttributes];
        for (int i = 0; i < numInstAttributes; ++i)
            this.fullMeans[i] = aux.meanOrMode(i);

        if (this.showStdDevs) {
            for (int i = 0; i < numInstAttributes; ++i) {
                this.fullStdDevs[i] = Math.sqrt(this.fullStdDevs[i]);
            }
        }

        int[] clusterAssignments = new int[instances.numInstances()];
        if (this.preserveOrder) {
            this.assignments = clusterAssignments;
        }

        Instances initInstances;
        if (this.preserveOrder) {
            initInstances = new Instances(instances);
        } else {
            initInstances = instances;
        }

        switch (this.initializationMethod) {
            case 0:
                this.randomInit(initInstances);
                break;
            case 1:
                this.kMeansPlusPlusInit(initInstances);
                break;
            case 2:
                this.canopyInit(initInstances);
                break;
            case 3:
                this.farthestFirstInit(initInstances);
                break;
        }

        this.numClusters = this.centroids.numInstances();
        this.squaredErrors = new double[this.numClusters];

        this.iterations = 0;
        int index;
        boolean converged = false;
        Instances[] bagsPerCluster = new Instances[this.numClusters];
        while (!converged) {
            this.iterations++;
            converged = this.assignInstancesToCluster(instances, clusterAssignments, false, true);

            for (int cluster = 0; cluster < this.numClusters; ++cluster) {
                bagsPerCluster[cluster] = new Instances(instances, 0);
            }
            for (int i = 0; i < instances.numInstances(); ++i) {
                bagsPerCluster[clusterAssignments[i]].add(instances.instance(i));
            }

            int emptyClusterCount = moveCentroids(bagsPerCluster);

            if (this.iterations == this.maxIterations) {
                converged = true;
            }

            if (emptyClusterCount > 0) {
                this.numClusters -= emptyClusterCount;
                if (!converged) {
                    bagsPerCluster = new Instances[this.numClusters];
                } else {
                    Instances[] aux2 = new Instances[this.numClusters];
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
        }

        if (!this.fastDistCalc) {
            assignInstancesToCluster(instances, clusterAssignments, true, false);
        }

        if (this.showStdDevs) {
            this.clusterStdDevs = new Instances(instances.get(0).relationalValue(1), this.numClusters);
        }
        this.clustersSize = new double[this.numClusters];
        for (int i = 0; i < this.numClusters; ++i) {
            if (this.showStdDevs) {
                Instances instancesPerCluster = new Instances(bagsPerCluster[i].get(0).relationalValue(1));
                for (int j = 0; j < bagsPerCluster[i].numInstances(); ++j)
                    instancesPerCluster.addAll(bagsPerCluster[i].get(j).relationalValue(1));

                double[] variances = instancesPerCluster.variances();
                for (index = 0; index < numInstAttributes; ++index) {
                    variances[index] = Math.sqrt(variances[index]);
                }
                this.clusterStdDevs.add(new DenseInstance(1D, variances));
            }
            this.clustersSize[i] = bagsPerCluster[i].numInstances();
        }

        long finishTime = System.currentTimeMillis();
        elapsedTime = (double) (finishTime - startTime) / 1000.0D;
    }

    protected void randomInit(Instances data) throws Exception {
        this.centroids = new Instances(data.get(0).relationalValue(1), this.numClusters);

        Random random = new Random(this.getSeed());
        Map<DecisionTableHashKey, Integer> initialClusters = new HashMap<>();
        int numInstAttributes = data.get(0).relationalValue(1).numAttributes();

        for (int i = data.numInstances() - 1; i >= 0; --i) {
            int bagIdx = random.nextInt(i + 1);
            DecisionTableHashKey hk = new DecisionTableHashKey(data.get(bagIdx), data.numAttributes(), true);
            if (!initialClusters.containsKey(hk)) {
                double[] mean = new double[numInstAttributes];
                for (int j = 0; j < numInstAttributes; ++j)
                    mean[j] = data.get(bagIdx).relationalValue(1).meanOrMode(j);
                Instance centroid = new DenseInstance(1D, mean);
                this.centroids.add(centroid);
                initialClusters.put(hk, null);
            }
            data.swap(i, bagIdx);
            if (this.centroids.numInstances() == this.numClusters) {
                break;
            }
        }
        this.startingPoints = new Instances(this.centroids);
    }

    private boolean assignInstancesToCluster(Instances instances, int[] clusterAssignments, boolean updateErrors, boolean useFastDistCalc) throws Exception {
        boolean converged = true;

        ExecutorService executor = Executors.newFixedThreadPool(executionSlots);
        Collection<Callable<Integer[]>> collection = new ArrayList<>(instances.numInstances());
        for (int i = 0; i < instances.numInstances(); ++i) {
            collection.add(new WrapperAsssignation(instances.get(i), i, updateErrors, useFastDistCalc));
        }
        try {
            List<Future<Integer[]>> futures = executor.invokeAll(collection);
            for (Future<Integer[]> future : futures) {
                Integer[] result = future.get();
                if (clusterAssignments[result[0]] != result[1])
                    converged = false;
                if (!updateErrors)
                    clusterAssignments[result[0]] = result[1];
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        return converged;
    }

    private class WrapperAsssignation implements Callable<Integer[]> {
        Instance instance;
        int idx;
        boolean useFastDistCalc;
        boolean updateErrors;
        WrapperAsssignation(Instance instance, int idx, boolean useFastDistCalc, boolean updateErrors) {
            this.instance = instance;
            this.idx = idx;
            this.useFastDistCalc = useFastDistCalc;
            this.updateErrors = updateErrors;
        }
        @Override
        public Integer[] call() throws Exception {
            int newCluster = clusterProcessedInstance(instance, updateErrors, useFastDistCalc, null);
            return new Integer[]{idx, newCluster};
        }
    }

    private int clusterProcessedInstance(Instance instance, boolean updateErrors, boolean useFastDistCalc, long[] instanceCanopies) {
        double minDist = Double.MAX_VALUE;
        int bestCluster = 0;
        for (int i = 0; i < this.numClusters; ++i) {
            double dist;
            if (useFastDistCalc) {
                dist = this.distFunction.distance(instance, this.centroids.get(i), minDist);
            } else {
                dist = this.distFunction.distance(instance, this.centroids.get(i));
            }
            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }

        if (updateErrors) {
            minDist *= minDist * instance.weight();
            double[] squaredErrors1 = this.squaredErrors;
            squaredErrors1[bestCluster] += minDist;
        }

        return bestCluster;
    }

    private int moveCentroids(Instances[] clusters) {
        int emptyClusterCount = 0;
        this.centroids = new Instances(centroids, numClusters);

        ExecutorService executor = Executors.newFixedThreadPool(executionSlots);
        Collection<Callable<Map<Integer, Instance>>> collection = new ArrayList<>(numClusters);
        for (int i = 0; i < numClusters; ++i) {
            if (clusters[i].numInstances() == 0)
                emptyClusterCount++;
            else
                collection.add(new WrapperCentroid(i, clusters[i]));
        }
        try {
            List<Future<Map<Integer, Instance>>> futures = executor.invokeAll(collection);
            for (Future<Map<Integer, Instance>> future : futures) {
                Map<Integer, Instance> result = future.get();
                for (Map.Entry<Integer, Instance> r : result.entrySet()) {
                    centroids.add(r.getValue());
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        return emptyClusterCount;
    }

    protected class WrapperCentroid implements Callable<Map<Integer, Instance>> {
        Instances cluster;
        Integer idx;
        WrapperCentroid(Integer idx, Instances cluster) {
            this.idx = idx;
            this.cluster = cluster;
        }
        @Override
        public Map<Integer, Instance> call() throws Exception {
            Instance centroid = computeCentroid(cluster);
            Map<Integer, Instance> result = new HashMap<>();
            result.put(idx, centroid);
            return result;
        }
    }

    Instance computeCentroid(Instances members) {
        int numInstAttributes = members.get(0).relationalValue(1).numAttributes();

        Instances aux = new Instances(members.get(0).relationalValue(1));
        for (Instance member : members) {
            aux.addAll(member.relationalValue(1));
        }

        double[] means = new double[numInstAttributes];
        for (int i = 0; i < numInstAttributes; ++i) {
            means[i] = aux.meanOrMode(i);
        }

        return new DenseInstance(1.0D, means);
    }

    @Override
    public int clusterInstance(Instance instance) {
        return this.clusterProcessedInstance(instance, false, true, null);
    }

    @Override
    public int numberOfClusters() throws Exception {
        return this.numClusters;
    }

    @Override
    public double getElapsedTime() {
        return elapsedTime;
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
        result.addElement(new Option("\tEnables faster distance calculations, using cut-off values.\n\tDisables the calculation/output of squared errors/distances.\n", "fast", 0, "-fast"));
        result.addElement(new Option("\tNumber of execution slots.\n\t(default 1 - i.e. no parallelism)", "num-slots", 1, "-num-slots <num>"));
        result.addAll(Collections.list(super.listOptions()));
        return result.elements();
    }

    //TODO No adaptado a MI
    private void canopyInit(Instances data) throws Exception {
        if (this.canopyClusters == null) {
            this.canopyClusters = new Canopy();
            this.canopyClusters.setNumClusters(this.numClusters);
            this.canopyClusters.setSeed(this.getSeed());
            this.canopyClusters.setT2(this.getCanopyT2());
            this.canopyClusters.setT1(this.getCanopyT1());
            this.canopyClusters.setMaxNumCandidateCanopiesToHoldInMemory(this.getCanopyMaxNumCanopiesToHoldInMemory());
            this.canopyClusters.setPeriodicPruningRate(this.getCanopyPeriodicPruningRate());
            this.canopyClusters.setMinimumCanopyDensity(this.getCanopyMinimumCanopyDensity());
            this.canopyClusters.setDebug(this.getDebug());
            this.canopyClusters.buildClusterer(data);
        }
        centroids = this.canopyClusters.getCanopies();
        startingPoints = new Instances(this.canopyClusters.getCanopies());
        centroidCanopyAssignments = new ArrayList<>();
        dataPointCanopyAssignments = new ArrayList<>();
    }

    //TODO No adaptado a MI
    private void farthestFirstInit(Instances data) throws Exception {
        FarthestFirst ff = new FarthestFirst();
        ff.setNumClusters(this.numClusters);
        ff.buildClusterer(data);
        this.centroids = ff.getClusterCentroids();
        this.startingPoints = new Instances(this.centroids);
    }

    //TODO No adaptado a MI
    private void kMeansPlusPlusInit(Instances data) throws Exception {
        Random random = new Random(this.getSeed());
        Map<DecisionTableHashKey, String> initC = new HashMap<>();
        int index = random.nextInt(data.numInstances());
        this.centroids.add(data.instance(index));
        DecisionTableHashKey hk = new DecisionTableHashKey(data.instance(index), data.numAttributes(), true);
        initC.put(hk, null);
        int iteration = 0;
        int remainingInstances = data.numInstances() - 1;
        if (this.numClusters > 1) {
            double[] distances = new double[data.numInstances()];
            double[] cumProbs = new double[data.numInstances()];

            for (int i = 0; i < data.numInstances(); ++i) {
                distances[i] = this.distFunction.distance(data.instance(i), this.centroids.instance(iteration));
            }

            for (int i = 1; i < this.numClusters; ++i) {
                double[] weights = new double[data.numInstances()];
                System.arraycopy(distances, 0, weights, 0, distances.length);
                Utils.normalize(weights);
                double sumOfProbs = 0.0D;

                for (int k = 0; k < data.numInstances(); ++k) {
                    sumOfProbs += weights[k];
                    cumProbs[k] = sumOfProbs;
                }

                cumProbs[data.numInstances() - 1] = 1.0D;
                double prob = random.nextDouble();

                for (int k = 0; k < cumProbs.length; ++k) {
                    if (prob < cumProbs[k]) {
                        Instance candidateCenter = data.instance(k);
                        hk = new DecisionTableHashKey(candidateCenter, data.numAttributes(), true);
                        if (!initC.containsKey(hk)) {
                            initC.put(hk, null);
                            this.centroids.add(candidateCenter);
                        } else {
                            System.err.println("We shouldn't get here....");
                        }
                        --remainingInstances;
                        break;
                    }
                }
                ++iteration;
                if (remainingInstances == 0) {
                    break;
                }

                for (int k = 0; k < data.numInstances(); ++k) {
                    if (distances[k] > 0.0D) {
                        double newDist = this.distFunction.distance(data.instance(k), this.centroids.instance(iteration));
                        if (newDist < distances[k]) {
                            distances[k] = newDist;
                        }
                    }
                }
            }
        }
    }

    public String numClustersTipText() {
        return "set number of clusters";
    }

    public void setNumClusters(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Number of clusters must be > 0");
        } else {
            this.numClusters = n;
        }
    }

    public int getNumClusters() {
        return this.numClusters;
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

    public String reduceNumberOfDistanceCalcsViaCanopiesTipText() {
        return "Use canopy clustering to reduce the number of distance calculations performed by k-means";
    }

    public void setReduceNumberOfDistanceCalcsViaCanopies(boolean c) {
        this.useCanopi = c;
    }

    public boolean getReduceNumberOfDistanceCalcsViaCanopies() {
        return this.useCanopi;
    }

    public String canopyPeriodicPruningRateTipText() {
        return "If using canopy clustering for initialization and/or speedup this is how often to prune low density canopies during training";
    }

    public void setCanopyPeriodicPruningRate(int p) {
        this.periodicPruningRate = p;
    }

    public int getCanopyPeriodicPruningRate() {
        return this.periodicPruningRate;
    }

    public String canopyMinimumCanopyDensityTipText() {
        return "If using canopy clustering for initialization and/or speedup this is the minimum T2-based density below which a canopy will be pruned during periodic pruning";
    }

    public void setCanopyMinimumCanopyDensity(double dens) {
        this.minClusterDensity = dens;
    }

    public double getCanopyMinimumCanopyDensity() {
        return this.minClusterDensity;
    }

    public String canopyMaxNumCanopiesToHoldInMemoryTipText() {
        return "If using canopy clustering for initialization and/or speedup this is the maximum number of candidate canopies to retain in main memory during training of the canopy clusterer. T2 distance and data characteristics determine how many candidate canopies are formed before periodic and final pruning are performed. There may not be enough memory available if T2 is set too low.";
    }

    public void setCanopyMaxNumCanopiesToHoldInMemory(int max) {
        this.maxCanopyCandidates = max;
    }

    public int getCanopyMaxNumCanopiesToHoldInMemory() {
        return this.maxCanopyCandidates;
    }

    public String canopyT2TipText() {
        return "The T2 distance to use when using canopy clustering. Values < 0 indicate that this should be set using a heuristic based on attribute standard deviation";
    }

    public void setCanopyT2(double t2) {
        this.t2 = t2;
    }

    public double getCanopyT2() {
        return this.t2;
    }

    public String canopyT1TipText() {
        return "The T1 distance to use when using canopy clustering. Values < 0 are taken as a positive multiplier for the T2 distance";
    }

    public void setCanopyT1(double t1) {
        this.t1 = t1;
    }

    public double getCanopyT1() {
        return this.t1;
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

    public void setDistanceFunction(DistanceFunction df, String[] options) throws Exception {
        this.distFunction = df;
        distFunction.setOptions(options);
    }

    public String preserveInstancesOrderTipText() {
        return "Preserve order of instances.";
    }

    public void setPreserveInstancesOrder(boolean r) {
        this.preserveOrder = r;
    }

    public boolean getPreserveInstancesOrder() {
        return this.preserveOrder;
    }

    public String fastDistanceCalcTipText() {
        return "Uses cut-off values for speeding up distance calculation, but suppresses also the calculation and output of the within cluster sum of squared errors/sum of distances.";
    }

    public void setFastDistanceCalc(boolean value) {
        this.fastDistCalc = value;
    }

    public boolean getFastDistanceCalc() {
        return this.fastDistCalc;
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

        this.useCanopi = Utils.getFlag('C', options);
        String mc = Utils.getOption("max-candidates", options);
        if (mc.length() > 0) {
            this.setCanopyMaxNumCanopiesToHoldInMemory(Integer.parseInt(mc));
        }

        String pp = Utils.getOption("periodic-pruning", options);
        if (pp.length() > 0) {
            this.setCanopyPeriodicPruningRate(Integer.parseInt(pp));
        }

        String md = Utils.getOption("min-density", options);
        if (md.length() > 0) {
            this.setCanopyMinimumCanopyDensity(Double.parseDouble(md));
        }

        String t2 = Utils.getOption("t2", options);
        if (t2.length() > 0) {
            this.setCanopyT2(Double.parseDouble(t2));
        }

        String t1 = Utils.getOption("t1", options);
        if (t1.length() > 0) {
            this.setCanopyT1(Double.parseDouble(t1));
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
        if (distFunctionClass.length() != 0) {
            String[] distFunctionClassSpec = Utils.splitOptions(distFunctionClass);
            if (distFunctionClassSpec.length == 0) {
                throw new Exception("Invalid DistanceFunction specification string.");
            }

            String className = distFunctionClassSpec[0];
            distFunctionClassSpec[0] = "";
            this.setDistanceFunction((DistanceFunction) Utils.forName(DistanceFunction.class, className, distFunctionClassSpec), options);
        } else {
            this.setDistanceFunction(new HausdorffDistance(), options);
        }

        this.preserveOrder = Utils.getFlag("O", options);
        this.fastDistCalc = Utils.getFlag("fast", options);
        String slotsS = Utils.getOption("num-slots", options);
        if (slotsS.length() > 0) {
            executionSlots = Integer.parseInt(slotsS);
        }

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        result.add("-init");
        result.add(String.valueOf(this.getInitializationMethod()));
        if (this.useCanopi) {
            result.add("-C");
        }
        result.add("-max-candidates");
        result.add(String.valueOf(this.getCanopyMaxNumCanopiesToHoldInMemory()));
        result.add("-periodic-pruning");
        result.add(String.valueOf(this.getCanopyPeriodicPruningRate()));
        result.add("-min-density");
        result.add(String.valueOf(this.getCanopyMinimumCanopyDensity()));
        result.add("-t1");
        result.add(String.valueOf(this.getCanopyT1()));
        result.add("-t2");
        result.add(String.valueOf(this.getCanopyT2()));
        if (this.showStdDevs) {
            result.add("-V");
        }
        result.add("-N" + numClusters);
        result.add("-A");
        result.add((this.distFunction.getClass().getName() + " " + Utils.joinOptions(this.distFunction.getOptions())).trim());
        result.add("-I");
        result.add(String.valueOf(this.getMaxIterations()));
        if (this.preserveOrder) {
            result.add("-O");
        }
        if (this.fastDistCalc) {
            result.add("-fast");
        }
        result.add("-num-slots " + executionSlots);
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
        if (!this.fastDistCalc) {
            if (this.distFunction instanceof EuclideanDistance) {
                result.append("Within cluster sum of squared errors: ").append(Utils.sum(this.squaredErrors)).append("\n");
            } else {
                result.append("Sum of within cluster distances: ").append(Utils.sum(this.squaredErrors)).append("\n");
            }
        }
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
        } else {
            result.append("\t").append(this.canopyClusters.toString(false)).append("\n");
        }

        if (this.useCanopi) {
            result.append("Reduced number of distance calculations by using canopies.\n");
            if (this.initializationMethod != 2) {
                result.append("\tCanopy T2 radius: ").append(String.format("%-10.3f", this.canopyClusters.getActualT2())).append("\n");
                result.append("\tCanopy T1 radius: ").append(String.format("%-10.3f", this.canopyClusters.getActualT1())).append("\n");
            }
        }

        result.append("Final cluster centroids:\n");
        for (int i = 0; i < centroids.numInstances(); ++i) {
            result.append("\tCluster ").append(i).append(": ").append(centroids.instance(i)).append("\n");
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
        if (centroids.attribute(1).isRelationValued()) {
            clustersToPrint = new Instances(centroids.get(0).relationalValue(1), numClusters);
            numAttributes = centroids.get(0).relationalValue(1).numAttributes();
            for (int i = 0; i < numClusters; ++i) {
                double[] mean = new double[numAttributes];
                for (int j = 0; j < numAttributes; ++j) {
                    mean[j] = centroids.get(i).relationalValue(1).meanOrMode(j);
                }
                clustersToPrint.add(new DenseInstance(1D, mean));
            }
        }
        else {
            clustersToPrint = new Instances(centroids);
            numAttributes = centroids.numAttributes();
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        String clustNum;
        String plusMinus = "+/-";

        boolean containsNumeric = false;
        int maxV;
        for (int i = 0; i < centroids.numInstances(); ++i) {
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
        double[] auxSize = this.clustersSize;
        maxV = auxSize.length;
        String strVal;
        for (int i = 0; i < maxV; ++i) {
            double m_ClusterSize = auxSize[i];
            strVal = "(" + m_ClusterSize + ")";
            if (strVal.length() > maxWidth) {
                maxWidth = strVal.length();
            }
        }
        if (this.showStdDevs && maxAttWidth < "missing".length()) {
            maxAttWidth = "missing".length();
        }
        maxAttWidth += 2;
        if (this.showStdDevs && containsNumeric) {
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

        for (int i = 0; i < this.numClusters; ++i) {
            clustNum = String.valueOf(i);
            result.append(this.pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        result.append("\n");
        String cSize = "(" + Utils.sum(this.clustersSize) + ")";
        result.append(this.pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(), true));

        for (int i = 0; i < this.numClusters; ++i) {
            cSize = "(" + this.clustersSize[i] + ")";
            result.append(this.pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
        }
        result.append("\n");
        result.append(this.pad("", "=", maxAttWidth + maxWidth * (clustersToPrint.numInstances() + 1) + clustersToPrint.numInstances() + 1, true));
        result.append("\n");

        for (int i = 0; i < numAttributes; ++i) {
            String attName = clustersToPrint.attribute(i).name();
            result.append(attName);

            for (int j = 0; j < maxAttWidth - attName.length(); ++j) {
                result.append(" ");
            }

            String meanFull = Utils.doubleToString(this.fullMeans[i], maxWidth, 4).trim();
            result.append(this.pad(meanFull, " ", maxWidth + 1 - meanFull.length(), true));

            for (int j = 0; j < this.numClusters; ++j) {
                String meanCluster = Utils.doubleToString(clustersToPrint.instance(j).value(i), maxWidth, 4).trim();
                result.append(this.pad(meanCluster, " ", maxWidth + 1 - meanCluster.length(), true));
            }
            result.append("\n");

            if (this.showStdDevs) {
                String stdDevFull = plusMinus + Utils.doubleToString(this.fullStdDevs[i], maxWidth, 4).trim();
                result.append(this.pad(stdDevFull, " ", maxWidth + maxAttWidth + 1 - stdDevFull.length(), true));

                for (int j = 0; j < this.numClusters; ++j) {
                    String stdDevCluster = plusMinus + Utils.doubleToString(this.clusterStdDevs.instance(j).value(i), maxWidth, 4).trim();
                    result.append(this.pad(stdDevCluster, " ", maxWidth + 1 - stdDevCluster.length(), true));
                }
                result.append("\n\n");
            }
        }
        return  result.toString();
    }

    public Instances getClusterCentroids() {
        return this.centroids;
    }

    public Instances getClusterStandardDevs() {
        return this.clusterStdDevs;
    }

    public double getSquaredError() {
        return this.fastDistCalc ? Double.NaN : Utils.sum(this.squaredErrors);
    }

    public double[] getClusterSizes() {
        return this.clustersSize;
    }

    public int[] getAssignments() throws Exception {
        if (!this.preserveOrder) {
            throw new Exception("The assignments are only available when order of instances is preserved (-O)");
        } else if (this.assignments == null) {
            throw new Exception("No assignments made.");
        } else {
            return this.assignments;
        }
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "D. Arthur and S. Vassilvitskii");
        result.setValue(Field.TITLE, "k-means++: the advantages of carefull seeding");
        result.setValue(Field.BOOKTITLE, "Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms");
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
