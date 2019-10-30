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

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MISimpleKMeans extends RandomizableClusterer implements MyClusterer, NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {
    int m_NumClusters = 2;
    private int m_MaxIterations = 500;
    private int m_initializationMethod = 0;
    DistanceFunction m_DistanceFunction = new HausdorffDistance();
    private boolean m_PreserveOrder = false;
    private boolean m_displayStdDevs = false;
    private boolean m_FastDistanceCalc = false;
    private boolean m_speedUpDistanceCompWithCanopies = false;

    Instances m_initialStartPoints;
    Instances m_ClusterCentroids;
    private Instances m_ClusterStdDevs;
    private double[] m_FullMeansOrMediansOrModes;
    private double[] m_FullStdDevs;
    private double[] m_ClusterSizes;
    private int m_Iterations = 0;
    private double[] m_squaredErrors;
    private int[] m_Assignments = null;
    private List<long[]> m_centroidCanopyAssignments;
    private List<long[]> m_dataPointCanopyAssignments;
    private Canopy m_canopyClusters;
    private int m_maxCanopyCandidates = 100;
    private int m_periodicPruningRate = 10000;
    private double m_minClusterDensity = 2.0D;
    private double m_t2 = -1.0D;
    private double m_t1 = -1.25D;
    private int m_executionSlots = 1;
    private transient ExecutorService m_executorPool;

    public MISimpleKMeans() {
        this.m_SeedDefault = 10;
        this.setSeed(this.m_SeedDefault);
    }

    private void startExecutorPool() {
        if (this.m_executorPool != null) {
            this.m_executorPool.shutdownNow();
        }
        this.m_executorPool = Executors.newFixedThreadPool(this.m_executionSlots);
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

    private int launchMoveCentroids(Instances[] clusters) {
        int emptyClusterCount = 0;
        List<Future<Instance>> results = new ArrayList<>();
        Future<Instance> d;
        for (int i = 0; i < this.m_NumClusters; ++i) {
            if (clusters[i].numInstances() == 0) {
                ++emptyClusterCount;
            } else {
                d = this.m_executorPool.submit(new MISimpleKMeans.KMeansComputeCentroidTask(i, clusters[i]));
                results.add(d);
            }
        }
        try {
            for (Future<Instance> result : results) {
                this.m_ClusterCentroids.add(result.get());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return emptyClusterCount;
    }

    private boolean launchAssignToClusters(Instances insts, int[] clusterAssignments) throws Exception {
        int numPerTask = insts.numInstances() / this.m_executionSlots;
        List<Future<Boolean>> results = new ArrayList<>();

        for (int i = 0; i < this.m_executionSlots; ++i) {
            int start = i * numPerTask;
            int end = start + numPerTask;
            if (i == this.m_executionSlots - 1) {
                end = insts.numInstances();
            }

            Future<Boolean> futureKM = this.m_executorPool.submit(new MISimpleKMeans.KMeansClusterTask(insts, start, end, clusterAssignments));
            results.add(futureKM);
        }

        boolean converged = true;
        for (Future<Boolean> result : results) {
            if (!result.get()) {
                converged = false;
            }
        }
        return converged;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        this.getCapabilities().testWithFail(data);

        int numInstAttributes = data.get(0).relationalValue(1).numAttributes();

        Instances instances = new Instances(data);
        instances.setClassIndex(-1);

        Instances aux = new Instances(instances.get(0).relationalValue(1));
        for (int i = 1; i < instances.size(); ++i) {
            aux.addAll(instances.get(i).relationalValue(1));
        }
        if (this.m_displayStdDevs) {
            this.m_FullStdDevs = aux.variances();
        }

        this.m_FullMeansOrMediansOrModes = new double[numInstAttributes];
        for (int i = 0; i < numInstAttributes; ++i)
            this.m_FullMeansOrMediansOrModes[i] = aux.meanOrMode(i);

        if (this.m_displayStdDevs) {
            for (int i = 0; i < numInstAttributes; ++i) {
                this.m_FullStdDevs[i] = Math.sqrt(this.m_FullStdDevs[i]);
            }
        }

        int[] clusterAssignments = new int[instances.numInstances()];
        if (this.m_PreserveOrder) {
            this.m_Assignments = clusterAssignments;
        }

        Instances initInstances;
        if (this.m_PreserveOrder) {
            initInstances = new Instances(instances);
        } else {
            initInstances = instances;
        }

        if (this.m_speedUpDistanceCompWithCanopies) {
            canopyInit(initInstances);
            this.m_centroidCanopyAssignments = new ArrayList<>();
            this.m_dataPointCanopyAssignments = new ArrayList<>();
        }

        switch (this.m_initializationMethod) {
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

        if (this.m_speedUpDistanceCompWithCanopies) {
            for (int i = 0; i < instances.numInstances(); ++i) {
                this.m_dataPointCanopyAssignments.add(this.m_canopyClusters.assignCanopies(instances.instance(i)));
            }
        }

        this.m_NumClusters = this.m_ClusterCentroids.numInstances();
        this.m_squaredErrors = new double[this.m_NumClusters];
        this.startExecutorPool();

        this.m_Iterations = 0;
        int index;
        boolean converged = false;
        Instances[] bagsPerCluster = new Instances[this.m_NumClusters];
        while (!converged) {
            if (this.m_speedUpDistanceCompWithCanopies) {
                this.m_centroidCanopyAssignments.clear();
                for (int i = 0; i < this.m_ClusterCentroids.numInstances(); ++i) {
                    this.m_centroidCanopyAssignments.add(this.m_canopyClusters.assignCanopies(this.m_ClusterCentroids.instance(i)));
                }
            }

            this.m_Iterations++;
            converged = true;
            if (this.m_executionSlots > 1 && instances.numInstances() >= 2 * this.m_executionSlots) {
                converged = this.launchAssignToClusters(instances, clusterAssignments);
            } else {
                for (int i = 0; i < instances.numInstances(); ++i) {
                    Instance toCluster = instances.instance(i);
                    index = this.clusterProcessedInstance(toCluster, false, true,
                            this.m_speedUpDistanceCompWithCanopies ? this.m_dataPointCanopyAssignments.get(i) : null);
                    if (index != clusterAssignments[i]) {
                        converged = false;
                    }
                    clusterAssignments[i] = index;
                }
            }

            this.m_ClusterCentroids = new Instances(m_ClusterCentroids, this.m_NumClusters);

            for (int cluster = 0; cluster < this.m_NumClusters; ++cluster) {
                bagsPerCluster[cluster] = new Instances(instances, 0);
            }
            for (int i = 0; i < instances.numInstances(); ++i) {
                bagsPerCluster[clusterAssignments[i]].add(instances.instance(i));
            }

            int emptyClusterCount = 0;
            if (this.m_executionSlots > 1 && instances.numInstances() >= 2 * this.m_executionSlots) {
                emptyClusterCount = this.launchMoveCentroids(bagsPerCluster);
            } else {
                for (int cluster = 0; cluster < this.m_NumClusters; ++cluster) {
                    if (bagsPerCluster[cluster].numInstances() == 0) {
                        emptyClusterCount++;
                    } else {
                        this.moveCentroid(cluster, bagsPerCluster[cluster], true);
                    }
                }
            }

            if (this.m_Iterations == this.m_MaxIterations) {
                converged = true;
            }

            if (emptyClusterCount > 0) {
                this.m_NumClusters -= emptyClusterCount;
                if (!converged) {
                    bagsPerCluster = new Instances[this.m_NumClusters];
                } else {
                    Instances[] aux2 = new Instances[this.m_NumClusters];
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

        if (!this.m_FastDistanceCalc) {
            for (int i = 0; i < instances.numInstances(); ++i) {
                this.clusterProcessedInstance(instances.instance(i), true, false, null);
            }
        }

        if (this.m_displayStdDevs) {
            this.m_ClusterStdDevs = new Instances(instances.get(0).relationalValue(1), this.m_NumClusters);
        }
        this.m_ClusterSizes = new double[this.m_NumClusters];
        for (int i = 0; i < this.m_NumClusters; ++i) {
            if (this.m_displayStdDevs) {
                Instances instancesPerCluster = new Instances(bagsPerCluster[i].get(0).relationalValue(1));
                for (int j = 0; j < bagsPerCluster[i].numInstances(); ++j)
                    instancesPerCluster.addAll(bagsPerCluster[i].get(j).relationalValue(1));

                double[] variances = instancesPerCluster.variances();
                for (index = 0; index < numInstAttributes; ++index) {
                    variances[index] = Math.sqrt(variances[index]);
                }
                this.m_ClusterStdDevs.add(new DenseInstance(1D, variances));
            }
            this.m_ClusterSizes[i] = bagsPerCluster[i].numInstances();
        }

        this.m_executorPool.shutdown();
        this.m_DistanceFunction.clean();
    }

    protected void randomInit(Instances data) throws Exception {
        this.m_ClusterCentroids = new Instances(data.get(0).relationalValue(1), this.m_NumClusters);

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
                this.m_ClusterCentroids.add(centroid);
                initialClusters.put(hk, null);
            }
            data.swap(i, bagIdx);
            if (this.m_ClusterCentroids.numInstances() == this.m_NumClusters) {
                break;
            }
        }
        this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
    }

    //TODO No adaptado a MI
    private void canopyInit(Instances data) throws Exception {
        if (this.m_canopyClusters == null) {
            this.m_canopyClusters = new Canopy();
            this.m_canopyClusters.setNumClusters(this.m_NumClusters);
            this.m_canopyClusters.setSeed(this.getSeed());
            this.m_canopyClusters.setT2(this.getCanopyT2());
            this.m_canopyClusters.setT1(this.getCanopyT1());
            this.m_canopyClusters.setMaxNumCandidateCanopiesToHoldInMemory(this.getCanopyMaxNumCanopiesToHoldInMemory());
            this.m_canopyClusters.setPeriodicPruningRate(this.getCanopyPeriodicPruningRate());
            this.m_canopyClusters.setMinimumCanopyDensity(this.getCanopyMinimumCanopyDensity());
            this.m_canopyClusters.setDebug(this.getDebug());
            this.m_canopyClusters.buildClusterer(data);
        }
        this.m_ClusterCentroids = this.m_canopyClusters.getCanopies();
        this.m_initialStartPoints = new Instances(this.m_canopyClusters.getCanopies());
    }

    //TODO No adaptado a MI
    private void farthestFirstInit(Instances data) throws Exception {
        FarthestFirst ff = new FarthestFirst();
        ff.setNumClusters(this.m_NumClusters);
        ff.buildClusterer(data);
        this.m_ClusterCentroids = ff.getClusterCentroids();
        this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
    }

    //TODO No adaptado a MI
    private void kMeansPlusPlusInit(Instances data) throws Exception {
        Random random = new Random(this.getSeed());
        Map<DecisionTableHashKey, String> initC = new HashMap<>();
        int index = random.nextInt(data.numInstances());
        this.m_ClusterCentroids.add(data.instance(index));
        DecisionTableHashKey hk = new DecisionTableHashKey(data.instance(index), data.numAttributes(), true);
        initC.put(hk, null);
        int iteration = 0;
        int remainingInstances = data.numInstances() - 1;
        if (this.m_NumClusters > 1) {
            double[] distances = new double[data.numInstances()];
            double[] cumProbs = new double[data.numInstances()];

            for (int i = 0; i < data.numInstances(); ++i) {
                distances[i] = this.m_DistanceFunction.distance(data.instance(i), this.m_ClusterCentroids.instance(iteration));
            }

            for (int i = 1; i < this.m_NumClusters; ++i) {
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
                            this.m_ClusterCentroids.add(candidateCenter);
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
                        double newDist = this.m_DistanceFunction.distance(data.instance(k), this.m_ClusterCentroids.instance(iteration));
                        if (newDist < distances[k]) {
                            distances[k] = newDist;
                        }
                    }
                }
            }
        }
    }

    protected Instance moveCentroid(int clusterIdx, Instances members, Boolean addToCentroidInstances) {
        int numInstAttributes = members.get(0).relationalValue(1).numAttributes();

        Instances aux = new Instances(members.get(0).relationalValue(1));
        for (Instance member : members) {
            aux.addAll(member.relationalValue(1));
        }

        double[] means = new double[numInstAttributes];
        for (int i = 0; i < numInstAttributes; ++i) {
            means[i] = aux.meanOrMode(i);
        }

        Instance centroid = new DenseInstance(1.0D, means);

        if (addToCentroidInstances) {
            this.m_ClusterCentroids.add(centroid);
        }
        return centroid;
    }

    private int clusterProcessedInstance(Instance instance, boolean updateErrors, boolean useFastDistCalc, long[] instanceCanopies) {
        double minDist = 2.147483647E9D;
        int bestCluster = 0;

        for (int i = 0; i < this.m_NumClusters; ++i) {
            double dist;
            if (useFastDistCalc) {
                if (this.m_speedUpDistanceCompWithCanopies && instanceCanopies != null && instanceCanopies.length > 0) {
                    try {
                        if (!Canopy.nonEmptyCanopySetIntersection(this.m_centroidCanopyAssignments.get(i), instanceCanopies)) {
                            continue;
                        }
                    } catch (Exception var12) {
                        var12.printStackTrace();
                    }

                    dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.get(i), minDist);
                } else {
                    dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.get(i), minDist);
                }
            } else {
                dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.get(i));
            }

            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }

        if (updateErrors) {
            minDist *= minDist * instance.weight();
            double[] var10000 = this.m_squaredErrors;
            var10000[bestCluster] += minDist;
        }

        return bestCluster;
    }

    @Override
    public int clusterInstance(Instance instance) {
        return this.clusterProcessedInstance(instance, false, true, null);
    }

    @Override
    public int numberOfClusters() throws Exception {
        return this.m_NumClusters;
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

    public String numClustersTipText() {
        return "set number of clusters";
    }

    public void setNumClusters(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Number of clusters must be > 0");
        } else {
            this.m_NumClusters = n;
        }
    }

    public int getNumClusters() {
        return this.m_NumClusters;
    }

    public String initializationMethodTipText() {
        return "The initialization method to use. Random, k-means++, Canopy or farthest first";
    }

    public void setInitializationMethod(Integer selection) {
        this.m_initializationMethod = selection;
    }

    public int getInitializationMethod() {
        return this.m_initializationMethod;
    }

    public String reduceNumberOfDistanceCalcsViaCanopiesTipText() {
        return "Use canopy clustering to reduce the number of distance calculations performed by k-means";
    }

    public void setReduceNumberOfDistanceCalcsViaCanopies(boolean c) {
        this.m_speedUpDistanceCompWithCanopies = c;
    }

    public boolean getReduceNumberOfDistanceCalcsViaCanopies() {
        return this.m_speedUpDistanceCompWithCanopies;
    }

    public String canopyPeriodicPruningRateTipText() {
        return "If using canopy clustering for initialization and/or speedup this is how often to prune low density canopies during training";
    }

    public void setCanopyPeriodicPruningRate(int p) {
        this.m_periodicPruningRate = p;
    }

    public int getCanopyPeriodicPruningRate() {
        return this.m_periodicPruningRate;
    }

    public String canopyMinimumCanopyDensityTipText() {
        return "If using canopy clustering for initialization and/or speedup this is the minimum T2-based density below which a canopy will be pruned during periodic pruning";
    }

    public void setCanopyMinimumCanopyDensity(double dens) {
        this.m_minClusterDensity = dens;
    }

    public double getCanopyMinimumCanopyDensity() {
        return this.m_minClusterDensity;
    }

    public String canopyMaxNumCanopiesToHoldInMemoryTipText() {
        return "If using canopy clustering for initialization and/or speedup this is the maximum number of candidate canopies to retain in main memory during training of the canopy clusterer. T2 distance and data characteristics determine how many candidate canopies are formed before periodic and final pruning are performed. There may not be enough memory available if T2 is set too low.";
    }

    public void setCanopyMaxNumCanopiesToHoldInMemory(int max) {
        this.m_maxCanopyCandidates = max;
    }

    public int getCanopyMaxNumCanopiesToHoldInMemory() {
        return this.m_maxCanopyCandidates;
    }

    public String canopyT2TipText() {
        return "The T2 distance to use when using canopy clustering. Values < 0 indicate that this should be set using a heuristic based on attribute standard deviation";
    }

    public void setCanopyT2(double t2) {
        this.m_t2 = t2;
    }

    public double getCanopyT2() {
        return this.m_t2;
    }

    public String canopyT1TipText() {
        return "The T1 distance to use when using canopy clustering. Values < 0 are taken as a positive multiplier for the T2 distance";
    }

    public void setCanopyT1(double t1) {
        this.m_t1 = t1;
    }

    public double getCanopyT1() {
        return this.m_t1;
    }

    public String maxIterationsTipText() {
        return "set maximum number of iterations";
    }

    public void setMaxIterations(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Maximum number of iterations must be > 0");
        } else {
            this.m_MaxIterations = n;
        }
    }

    public int getMaxIterations() {
        return this.m_MaxIterations;
    }

    public String displayStdDevsTipText() {
        return "Display std deviations of numeric attributes and counts of nominal attributes.";
    }

    public void setDisplayStdDevs(boolean stdD) {
        this.m_displayStdDevs = stdD;
    }

    public boolean getDisplayStdDevs() {
        return this.m_displayStdDevs;
    }

    public String dontReplaceMissingValuesTipText() {
        return "Replace missing values globally with mean/mode.";
    }

    public String distanceFunctionTipText() {
        return "The distance function to use for instances comparison (default: weka.core.EuclideanDistance). ";
    }

    public DistanceFunction getDistanceFunction() {
        return this.m_DistanceFunction;
    }

    public void setDistanceFunction(DistanceFunction df) throws Exception {
        this.m_DistanceFunction = df;
    }

    public String preserveInstancesOrderTipText() {
        return "Preserve order of instances.";
    }

    public void setPreserveInstancesOrder(boolean r) {
        this.m_PreserveOrder = r;
    }

    public boolean getPreserveInstancesOrder() {
        return this.m_PreserveOrder;
    }

    public String fastDistanceCalcTipText() {
        return "Uses cut-off values for speeding up distance calculation, but suppresses also the calculation and output of the within cluster sum of squared errors/sum of distances.";
    }

    public void setFastDistanceCalc(boolean value) {
        this.m_FastDistanceCalc = value;
    }

    public boolean getFastDistanceCalc() {
        return this.m_FastDistanceCalc;
    }

    public String numExecutionSlotsTipText() {
        return "The number of execution slots (threads) to use. Set equal to the number of available cpu/cores";
    }

    public void setNumExecutionSlots(int slots) {
        this.m_executionSlots = slots;
    }

    public int getNumExecutionSlots() {
        return this.m_executionSlots;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        this.m_displayStdDevs = Utils.getFlag("V", options);

        String initM = Utils.getOption("init", options);
        if (initM.length() > 0) {
            this.setInitializationMethod(Integer.parseInt(initM));
        }

        this.m_speedUpDistanceCompWithCanopies = Utils.getFlag('C', options);
        String temp = Utils.getOption("max-candidates", options);
        if (temp.length() > 0) {
            this.setCanopyMaxNumCanopiesToHoldInMemory(Integer.parseInt(temp));
        }

        temp = Utils.getOption("periodic-pruning", options);
        if (temp.length() > 0) {
            this.setCanopyPeriodicPruningRate(Integer.parseInt(temp));
        }

        temp = Utils.getOption("min-density", options);
        if (temp.length() > 0) {
            this.setCanopyMinimumCanopyDensity(Double.parseDouble(temp));
        }

        temp = Utils.getOption("t2", options);
        if (temp.length() > 0) {
            this.setCanopyT2(Double.parseDouble(temp));
        }

        temp = Utils.getOption("t1", options);
        if (temp.length() > 0) {
            this.setCanopyT1(Double.parseDouble(temp));
        }

        String optionString = Utils.getOption('N', options);
        if (optionString.length() != 0) {
            this.setNumClusters(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("I", options);
        if (optionString.length() != 0) {
            this.setMaxIterations(Integer.parseInt(optionString));
        }

        String distFunctionClass = Utils.getOption('A', options);
        if (distFunctionClass.length() != 0) {
            String[] distFunctionClassSpec = Utils.splitOptions(distFunctionClass);
            if (distFunctionClassSpec.length == 0) {
                throw new Exception("Invalid DistanceFunction specification string.");
            }

            String className = distFunctionClassSpec[0];
            distFunctionClassSpec[0] = "";
            this.setDistanceFunction((DistanceFunction) Utils.forName(DistanceFunction.class, className, distFunctionClassSpec));
        } else {
            this.setDistanceFunction(new HausdorffDistance());
        }

        this.m_PreserveOrder = Utils.getFlag("O", options);
        this.m_FastDistanceCalc = Utils.getFlag("fast", options);
        String slotsS = Utils.getOption("num-slots", options);
        if (slotsS.length() > 0) {
            this.setNumExecutionSlots(Integer.parseInt(slotsS));
        }

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        result.add("-init");
        result.add(String.valueOf(this.getInitializationMethod()));
        if (this.m_speedUpDistanceCompWithCanopies) {
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
        if (this.m_displayStdDevs) {
            result.add("-V");
        }
        result.add("-N");
        result.add(String.valueOf(this.getNumClusters()));
        result.add("-A");
        result.add((this.m_DistanceFunction.getClass().getName() + " " + Utils.joinOptions(this.m_DistanceFunction.getOptions())).trim());
        result.add("-I");
        result.add(String.valueOf(this.getMaxIterations()));
        if (this.m_PreserveOrder) {
            result.add("-O");
        }
        if (this.m_FastDistanceCalc) {
            result.add("-fast");
        }
        result.add("-num-slots");
        result.add(String.valueOf(this.getNumExecutionSlots()));
        Collections.addAll(result, super.getOptions());
        return result.toArray(new String[0]);
    }

    public String toString() {
        if (this.m_ClusterCentroids == null) {
            return "No clusterer built yet!";
        }

        StringBuilder result = new StringBuilder();

        result.append("\nNumber of iterations: ").append(this.m_Iterations).append("\n");
        if (!this.m_FastDistanceCalc) {
            if (this.m_DistanceFunction instanceof EuclideanDistance) {
                result.append("Within cluster sum of squared errors: ").append(Utils.sum(this.m_squaredErrors)).append("\n");
            } else {
                result.append("Sum of within cluster distances: ").append(Utils.sum(this.m_squaredErrors)).append("\n");
            }
        }
        result.append("Initial starting points (");
        switch (this.m_initializationMethod) {
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

        if (this.m_initializationMethod != 2) {
            for (int i = 0; i < this.m_initialStartPoints.numInstances(); ++i) {
                result.append("\tCluster ").append(i).append(": ").append(this.m_initialStartPoints.instance(i)).append("\n");
            }
        } else {
            result.append("\t").append(this.m_canopyClusters.toString(false)).append("\n");
        }

        if (this.m_speedUpDistanceCompWithCanopies) {
            result.append("Reduced number of distance calculations by using canopies.\n");
            if (this.m_initializationMethod != 2) {
                result.append("\tCanopy T2 radius: ").append(String.format("%-10.3f", this.m_canopyClusters.getActualT2())).append("\n");
                result.append("\tCanopy T1 radius: ").append(String.format("%-10.3f", this.m_canopyClusters.getActualT1())).append("\n");
            }
        }

        result.append("Final cluster centroids:\n");
        for (int i = 0; i < this.m_initialStartPoints.numInstances(); ++i) {
            result.append("\tCluster ").append(i).append(": ").append(this.m_ClusterCentroids.instance(i)).append("\n");
        }

        printSurvey(result);

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

    private void printSurvey(StringBuilder result) {
        Instances clustersToPrint;
        int numAttributes;
        if (m_ClusterCentroids.attribute(1).isRelationValued()) {
            clustersToPrint = new Instances(m_ClusterCentroids.get(0).relationalValue(1), m_NumClusters);
            numAttributes = m_ClusterCentroids.get(0).relationalValue(1).numAttributes();
            for (int i = 0; i < m_NumClusters; ++i) {
                double[] mean = new double[numAttributes];
                for (int j = 0; j < numAttributes; ++j) {
                    mean[j] = m_ClusterCentroids.get(i).relationalValue(1).meanOrMode(j);
                }
                clustersToPrint.add(new DenseInstance(1D, mean));
            }
        }
        else {
            clustersToPrint = new Instances(m_ClusterCentroids);
            numAttributes = m_ClusterCentroids.numAttributes();
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        String clustNum;
        String plusMinus = "+/-";

        boolean containsNumeric = false;
        int maxV;
        for (int i = 0; i < m_NumClusters; ++i) {
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
        double[] auxSize = this.m_ClusterSizes;
        maxV = auxSize.length;
        String strVal;
        for (int i = 0; i < maxV; ++i) {
            double m_ClusterSize = auxSize[i];
            strVal = "(" + m_ClusterSize + ")";
            if (strVal.length() > maxWidth) {
                maxWidth = strVal.length();
            }
        }
        if (this.m_displayStdDevs && maxAttWidth < "missing".length()) {
            maxAttWidth = "missing".length();
        }
        maxAttWidth += 2;
        if (this.m_displayStdDevs && containsNumeric) {
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

        for (int i = 0; i < this.m_NumClusters; ++i) {
            clustNum = String.valueOf(i);
            result.append(this.pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        result.append("\n");
        String cSize = "(" + Utils.sum(this.m_ClusterSizes) + ")";
        result.append(this.pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(), true));

        for (int i = 0; i < this.m_NumClusters; ++i) {
            cSize = "(" + this.m_ClusterSizes[i] + ")";
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

            String meanFull = Utils.doubleToString(this.m_FullMeansOrMediansOrModes[i], maxWidth, 4).trim();
            result.append(this.pad(meanFull, " ", maxWidth + 1 - meanFull.length(), true));

            for (int j = 0; j < this.m_NumClusters; ++j) {
                String meanCluster = Utils.doubleToString(clustersToPrint.instance(j).value(i), maxWidth, 4).trim();
                result.append(this.pad(meanCluster, " ", maxWidth + 1 - meanCluster.length(), true));
            }
            result.append("\n");

            if (this.m_displayStdDevs) {
                String stdDevFull = plusMinus + Utils.doubleToString(this.m_FullStdDevs[i], maxWidth, 4).trim();
                result.append(this.pad(stdDevFull, " ", maxWidth + maxAttWidth + 1 - stdDevFull.length(), true));

                for (int j = 0; j < this.m_NumClusters; ++j) {
                    String stdDevCluster = plusMinus + Utils.doubleToString(this.m_ClusterStdDevs.instance(j).value(i), maxWidth, 4).trim();
                    result.append(this.pad(stdDevCluster, " ", maxWidth + 1 - stdDevCluster.length(), true));
                }
                result.append("\n\n");
            }
        }
    }

    public Instances getClusterCentroids() {
        return this.m_ClusterCentroids;
    }

    public Instances getClusterStandardDevs() {
        return this.m_ClusterStdDevs;
    }

    public double getSquaredError() {
        return this.m_FastDistanceCalc ? Double.NaN : Utils.sum(this.m_squaredErrors);
    }

    public double[] getClusterSizes() {
        return this.m_ClusterSizes;
    }

    public int[] getAssignments() throws Exception {
        if (!this.m_PreserveOrder) {
            throw new Exception("The assignments are only available when order of instances is preserved (-O)");
        } else if (this.m_Assignments == null) {
            throw new Exception("No assignments made.");
        } else {
            return this.m_Assignments;
        }
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 11444 $");
    }

    private class KMeansClusterTask implements Callable<Boolean> {
        int m_start;
        int m_end;
        Instances m_inst;
        int[] m_clusterAssignments;

        KMeansClusterTask(Instances inst, int start, int end, int[] clusterAssignments) {
            this.m_start = start;
            this.m_end = end;
            this.m_inst = inst;
            this.m_clusterAssignments = clusterAssignments;
        }

        public Boolean call() {
            boolean converged = true;
            for (int i = this.m_start; i < this.m_end; ++i) {
                Instance toCluster = this.m_inst.instance(i);
                long[] instanceCanopies = MISimpleKMeans.this.m_speedUpDistanceCompWithCanopies ? MISimpleKMeans.this.m_dataPointCanopyAssignments.get(i) : null;
                int newC = this.clusterInstance(toCluster, instanceCanopies);
                if (newC != this.m_clusterAssignments[i]) {
                    converged = false;
                }
                this.m_clusterAssignments[i] = newC;
            }
            return converged;
        }

        int clusterInstance(Instance inst, long[] instanceCanopies) {
            double minDist = 2.147483647E9D;
            int bestCluster = 0;
            for (int i = 0; i < MISimpleKMeans.this.m_NumClusters; ++i) {
                if (MISimpleKMeans.this.m_speedUpDistanceCompWithCanopies && instanceCanopies != null && instanceCanopies.length > 0) {
                    try {
                        if (!Canopy.nonEmptyCanopySetIntersection(MISimpleKMeans.this.m_centroidCanopyAssignments.get(i), instanceCanopies)) {
                            continue;
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                double dist = MISimpleKMeans.this.m_DistanceFunction.distance(inst, MISimpleKMeans.this.m_ClusterCentroids.instance(i), minDist);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = i;
                }
            }
            return bestCluster;
        }
    }

    class KMeansComputeCentroidTask implements Callable<Instance> {
        Instances m_cluster;
        int m_centroidIndex;

        KMeansComputeCentroidTask(int centroidIndex, Instances cluster) {
            this.m_cluster = cluster;
            this.m_centroidIndex = centroidIndex;
        }

        public Instance call() {
            return MISimpleKMeans.this.moveCentroid(-1, this.m_cluster, false);
        }
    }
}
