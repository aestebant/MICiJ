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
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MISimpleKMeans extends RandomizableClusterer implements NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {
    protected ReplaceMissingValues m_ReplaceMissingFilter;
    protected int m_NumClusters = 2;
    protected Instances m_initialStartPoints;
    protected Instances m_ClusterCentroids;
    protected Instances m_ClusterStdDevs;
    protected double[][][] m_ClusterNominalCounts;
    protected double[][] m_ClusterMissingCounts;
    protected double[] m_FullMeansOrMediansOrModes;
    protected double[] m_FullStdDevs;
    protected double[][] m_FullNominalCounts;
    protected double[] m_FullMissingCounts;
    protected boolean m_displayStdDevs;
    protected boolean m_dontReplaceMissing = false;
    protected double[] m_ClusterSizes;
    protected int m_MaxIterations = 500;
    protected int m_Iterations = 0;
    protected double[] m_squaredErrors;
    protected DistanceFunction m_DistanceFunction = new HausdorffDistance();
    protected boolean m_PreserveOrder = false;
    protected int[] m_Assignments = null;
    protected boolean m_FastDistanceCalc = false;
    public static final int RANDOM = 0;
    public static final int KMEANS_PLUS_PLUS = 1;
    public static final int CANOPY = 2;
    public static final int FARTHEST_FIRST = 3;
    public static final Tag[] TAGS_SELECTION = new Tag[]{new Tag(0, "Random"), new Tag(1, "k-means++"), new Tag(2, "Canopy"), new Tag(3, "Farthest first")};
    protected int m_initializationMethod = 0;
    protected boolean m_speedUpDistanceCompWithCanopies = false;
    protected List<long[]> m_centroidCanopyAssignments;
    protected List<long[]> m_dataPointCanopyAssignments;
    protected Canopy m_canopyClusters;
    protected int m_maxCanopyCandidates = 100;
    protected int m_periodicPruningRate = 10000;
    protected double m_minClusterDensity = 2.0D;
    protected double m_t2 = -1.0D;
    protected double m_t1 = -1.25D;
    protected int m_executionSlots = 1;
    protected transient ExecutorService m_executorPool;
    protected int m_completed;
    protected int m_failed;

    public MISimpleKMeans() {
        this.m_SeedDefault = 10;
        this.setSeed(this.m_SeedDefault);
    }

    protected void startExecutorPool() {
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

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);
        result.enable(Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        return result;
    }

    protected int launchMoveCentroids(Instances[] clusters) {
        int emptyClusterCount = 0;
        List<Future<double[]>> results = new ArrayList<>();

        Future<double[]> d;
        for (int i = 0; i < this.m_NumClusters; ++i) {
            if (clusters[i].numInstances() == 0) {
                ++emptyClusterCount;
            } else {
                d = this.m_executorPool.submit(new MISimpleKMeans.KMeansComputeCentroidTask(i, clusters[i]));
                results.add(d);
            }
        }

        try {

            for (Future<double[]> result : results) {
                d = result;
                this.m_ClusterCentroids.add(new DenseInstance(1.0D, d.get()));
            }
        } catch (Exception var6) {
            var6.printStackTrace();
        }

        return emptyClusterCount;
    }

    protected boolean launchAssignToClusters(Instances insts, int[] clusterAssignments) throws Exception {
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

    public void buildClusterer(Instances data) throws Exception {
        this.m_canopyClusters = null;
        this.getCapabilities().testWithFail(data);
        this.m_Iterations = 0;
        this.m_ReplaceMissingFilter = new ReplaceMissingValues();
        Instances instances = new Instances(data);
        instances.setClassIndex(-1);
        if (!this.m_dontReplaceMissing) {
            this.m_ReplaceMissingFilter.setInputFormat(instances);
            instances = Filter.useFilter(instances, this.m_ReplaceMissingFilter);
        }

        int numInstAttributes = instances.get(0).relationalValue(1).numAttributes();

        this.m_ClusterNominalCounts = new double[this.m_NumClusters][numInstAttributes][];
        this.m_ClusterMissingCounts = new double[this.m_NumClusters][numInstAttributes];

        Instances aux = new Instances(instances.get(0).relationalValue(1));
        for (int i = 1; i < instances.size(); ++i) {
            aux.addAll(instances.get(i).relationalValue(1));
        }

        if (this.m_displayStdDevs) {
            this.m_FullStdDevs = aux.variances();
        }

        this.m_FullMeansOrMediansOrModes = this.moveCentroid(0, instances, true, false);
        this.m_FullMissingCounts = this.m_ClusterMissingCounts[0];
        this.m_FullNominalCounts = this.m_ClusterNominalCounts[0];
        double sumOfWeights = aux.sumOfWeights();

        for (int i = 0; i < numInstAttributes; ++i) {
            if (instances.get(0).relationalValue(1).attribute(i).isNumeric()) {
                if (this.m_displayStdDevs) {
                    this.m_FullStdDevs[i] = Math.sqrt(this.m_FullStdDevs[i]);
                }

                if (this.m_FullMissingCounts[i] == sumOfWeights) {
                    this.m_FullMeansOrMediansOrModes[i] = Double.NaN;
                }
            } else if (this.m_FullMissingCounts[i] > this.m_FullNominalCounts[i][Utils.maxIndex(this.m_FullNominalCounts[i])]) {
                this.m_FullMeansOrMediansOrModes[i] = -1.0D;
            }
        }

        this.m_ClusterCentroids = new Instances(instances, this.m_NumClusters);
        int[] clusterAssignments = new int[instances.numInstances()];
        if (this.m_PreserveOrder) {
            this.m_Assignments = clusterAssignments;
        }

        this.m_DistanceFunction.setInstances(instances);
        Random RandomO = new Random(this.getSeed());
        Map<DecisionTableHashKey, Integer> initC = new HashMap<>();
        DecisionTableHashKey hk;
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

        if (this.m_initializationMethod == 1) {
            this.kMeansPlusPlusInit(initInstances);
            this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
        } else if (this.m_initializationMethod == 2) {
            this.canopyInit(initInstances);
            this.m_initialStartPoints = new Instances(this.m_canopyClusters.getCanopies());
        } else if (this.m_initializationMethod == 3) {
            this.farthestFirstInit(initInstances);
            this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
        } else {
            for (int i = initInstances.numInstances() - 1; i >= 0; --i) {
                int instIndex = RandomO.nextInt(i + 1);
                hk = new DecisionTableHashKey(initInstances.instance(instIndex), initInstances.numAttributes(), true);
                if (!initC.containsKey(hk)) {
                    this.m_ClusterCentroids.add(initInstances.instance(instIndex));
                    initC.put(hk, null);
                }

                initInstances.swap(i, instIndex);
                if (this.m_ClusterCentroids.numInstances() == this.m_NumClusters) {
                    break;
                }
            }
            this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
        }

        if (this.m_speedUpDistanceCompWithCanopies) {
            for (int i = 0; i < instances.numInstances(); ++i) {
                this.m_dataPointCanopyAssignments.add(this.m_canopyClusters.assignCanopies(instances.instance(i)));
            }
        }

        this.m_NumClusters = this.m_ClusterCentroids.numInstances();
        boolean converged = false;
        Instances[] tempI = new Instances[this.m_NumClusters];
        this.m_squaredErrors = new double[this.m_NumClusters];
        this.m_ClusterNominalCounts = new double[this.m_NumClusters][numInstAttributes][0];
        this.m_ClusterMissingCounts = new double[this.m_NumClusters][numInstAttributes];
        this.startExecutorPool();

        int index;
        while (!converged) {
            if (this.m_speedUpDistanceCompWithCanopies) {
                this.m_centroidCanopyAssignments.clear();
                for (int i = 0; i < this.m_ClusterCentroids.numInstances(); ++i) {
                    this.m_centroidCanopyAssignments.add(this.m_canopyClusters.assignCanopies(this.m_ClusterCentroids.instance(i)));
                }
            }

            int emptyClusterCount = 0;
            ++this.m_Iterations;
            converged = true;
            if (this.m_executionSlots > 1 && instances.numInstances() >= 2 * this.m_executionSlots) {
                converged = this.launchAssignToClusters(instances, clusterAssignments);
            } else {
                for (int i = 0; i < instances.numInstances(); ++i) {
                    Instance toCluster = instances.instance(i);
                    index = this.clusterProcessedInstance(toCluster, false, true, this.m_speedUpDistanceCompWithCanopies ? (long[]) this.m_dataPointCanopyAssignments.get(i) : null);
                    if (index != clusterAssignments[i]) {
                        converged = false;
                    }

                    clusterAssignments[i] = index;
                }
            }

            this.m_ClusterCentroids = new Instances(instances, this.m_NumClusters);

            for (int i = 0; i < this.m_NumClusters; ++i) {
                tempI[i] = new Instances(instances, 0);
            }

            for (int i = 0; i < instances.numInstances(); ++i) {
                tempI[clusterAssignments[i]].add(instances.instance(i));
            }

            if (this.m_executionSlots > 1 && instances.numInstances() >= 2 * this.m_executionSlots) {
                emptyClusterCount = this.launchMoveCentroids(tempI);
            } else {
                for (int i = 0; i < this.m_NumClusters; ++i) {
                    if (tempI[i].numInstances() == 0) {
                        ++emptyClusterCount;
                    } else {
                        this.moveCentroid(i, tempI[i], true, true);
                    }
                }
            }

            if (this.m_Iterations == this.m_MaxIterations) {
                converged = true;
            }

            if (emptyClusterCount > 0) {
                this.m_NumClusters -= emptyClusterCount;
                if (!converged) {
                    tempI = new Instances[this.m_NumClusters];
                } else {
                    Instances[] t = new Instances[this.m_NumClusters];
                    index = 0;
                    int j = 0;
                    while (true) {
                        if (j >= tempI.length) {
                            tempI = t;
                            break;
                        }
                        if (tempI[j].numInstances() > 0) {
                            t[index] = tempI[j];
                            if (tempI[j].get(0).relationalValue(1).numAttributes() >= 0)
                                System.arraycopy(this.m_ClusterNominalCounts[j], 0, this.m_ClusterNominalCounts[index], 0, tempI[j].get(0).relationalValue(1).numAttributes());
                            ++index;
                        }
                        ++j;
                    }
                }
            }

            if (!converged) {
                this.m_ClusterNominalCounts = new double[this.m_NumClusters][instances.get(0).relationalValue(1).numAttributes()][0];
            }
        }

        if (!this.m_FastDistanceCalc) {
            for (int i = 0; i < instances.numInstances(); ++i) {
                this.clusterProcessedInstance(instances.instance(i), true, false, null);
            }
        }

        if (this.m_displayStdDevs) {
            this.m_ClusterStdDevs = new Instances(instances, this.m_NumClusters);
        }

        this.m_ClusterSizes = new double[this.m_NumClusters];

        for (int i = 0; i < this.m_NumClusters; ++i) {
            if (this.m_displayStdDevs) {
                double[] vals2 = tempI[i].variances();
                for (index = 0; index < instances.get(0).relationalValue(1).numAttributes(); ++index) {
                    if (instances.get(0).relationalValue(1).attribute(index).isNumeric()) {
                        vals2[index] = Math.sqrt(vals2[index]);
                    } else {
                        vals2[index] = Utils.missingValue();
                    }
                }

                this.m_ClusterStdDevs.add(new DenseInstance(1.0D, vals2));
            }

            this.m_ClusterSizes[i] = tempI[i].sumOfWeights();
        }

        this.m_executorPool.shutdown();
        this.m_DistanceFunction.clean();
    }

    protected void canopyInit(Instances data) throws Exception {
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
    }

    protected void farthestFirstInit(Instances data) throws Exception {
        FarthestFirst ff = new FarthestFirst();
        ff.setNumClusters(this.m_NumClusters);
        ff.buildClusterer(data);
        this.m_ClusterCentroids = ff.getClusterCentroids();
    }

    protected void kMeansPlusPlusInit(Instances data) throws Exception {
        Random randomO = new Random(this.getSeed());
        Map<DecisionTableHashKey, String> initC = new HashMap<>();
        int index = randomO.nextInt(data.numInstances());
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
                double prob = randomO.nextDouble();

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

    protected double[] moveCentroid(int centroidIndex, Instances members, boolean updateClusterInfo, boolean addToCentroidInstances) {
        int numInstAttributes = members.get(0).relationalValue(1).numAttributes();
        double[] vals = new double[numInstAttributes];
        double[][] nominalDists = new double[numInstAttributes][];
        double[] weightMissing = new double[numInstAttributes];
        double[] weightNonMissing = new double[numInstAttributes];

        for (int j = 0; j < numInstAttributes; ++j) {
            if (members.get(0).relationalValue(1).attribute(j).isNominal()) {
                nominalDists[j] = new double[members.get(0).relationalValue(1).attribute(j).numValues()];
            }
        }

        for (Instance bag : members) {
            for (Instance inst : bag.relationalValue(1)) {
                for (int i = 0; i < numInstAttributes; ++i) {
                    if (inst.isMissing(i)) {
                        weightMissing[i] += inst.weight();
                    } else {
                        weightNonMissing[i] += inst.weight();
                        if (inst.attribute(i).isNumeric()) {
                            vals[i] += inst.weight() * inst.value(i);
                        } else {
                            double[] var10000 = nominalDists[i];
                            int var10001 = (int) inst.value(i);
                            var10000[var10001] += inst.weight();
                        }
                    }
                }
            }
        }

        for (int i = 0; i < numInstAttributes; ++i) {
            if (members.get(0).relationalValue(1).attribute(i).isNumeric()) {
                if (weightNonMissing[i] > 0.0D) {
                    vals[i] /= weightNonMissing[i];
                } else {
                    vals[i] = Utils.missingValue();
                }
            } else {
                double max = -1.7976931348623157E308D;
                double maxIndex = -1.0D;

                for (int j = 0; j < nominalDists[i].length; ++j) {
                    if (nominalDists[i][j] > max) {
                        max = nominalDists[i][j];
                        maxIndex = j;
                    }

                    if (max < weightMissing[i]) {
                        vals[i] = Utils.missingValue();
                    } else {
                        vals[i] = maxIndex;
                    }
                }
            }
        }

        /*if (this.m_DistanceFunction instanceof ManhattanDistance) {
            int middle = (members.numInstances() - 1) / 2;
            boolean dataIsEven = members.numInstances() % 2 == 0;
            Instances sortedMembers;
            if (this.m_PreserveOrder) {
                sortedMembers = members;
            } else {
                sortedMembers = new Instances(members);
            }

            for (int j = 0; j < members.numAttributes(); ++j) {
                if (weightNonMissing[j] > 0.0D && members.attribute(j).isNumeric()) {
                    if (members.numInstances() == 1) {
                        vals[j] = members.instance(0).value(j);
                    } else {
                        vals[j] = sortedMembers.kthSmallestValue(j, middle + 1);
                        if (dataIsEven) {
                            vals[j] = (vals[j] + sortedMembers.kthSmallestValue(j, middle + 2)) / 2.0D;
                        }
                    }
                }
            }
        }*/

        if (updateClusterInfo) {
            for (int j = 0; j < numInstAttributes; ++j) {
                this.m_ClusterMissingCounts[centroidIndex][j] = weightMissing[j];
                this.m_ClusterNominalCounts[centroidIndex][j] = nominalDists[j];
            }
        }

        if (addToCentroidInstances) {
            this.m_ClusterCentroids.add(new DenseInstance(1.0D, vals));
        }

        return vals;
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

                    dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.instance(i), minDist);
                } else {
                    dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.instance(i), minDist);
                }
            } else {
                dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.instance(i));
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

    public int clusterInstance(Instance instance) throws Exception {
        Instance inst;
        if (!this.m_dontReplaceMissing) {
            this.m_ReplaceMissingFilter.input(instance);
            this.m_ReplaceMissingFilter.batchFinished();
            inst = this.m_ReplaceMissingFilter.output();
        } else {
            inst = instance;
        }

        return this.clusterProcessedInstance(inst, false, true, null);
    }

    public int numberOfClusters() throws Exception {
        return this.m_NumClusters;
    }

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
        result.add(new Option("\tDistance function to use.\n\t(default: weka.core.EuclideanDistance)", "A", 1, "-A <classname and options>"));
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

    public void setInitializationMethod(SelectedTag method) {
        if (method.getTags() == TAGS_SELECTION) {
            this.m_initializationMethod = method.getSelectedTag().getID();
        }

    }

    public SelectedTag getInitializationMethod() {
        return new SelectedTag(this.m_initializationMethod, TAGS_SELECTION);
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

    public void setDontReplaceMissingValues(boolean r) {
        this.m_dontReplaceMissing = r;
    }

    public boolean getDontReplaceMissingValues() {
        return this.m_dontReplaceMissing;
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

    public void setOptions(String[] options) throws Exception {
        this.m_displayStdDevs = Utils.getFlag("V", options);
        this.m_dontReplaceMissing = Utils.getFlag("M", options);
        String initM = Utils.getOption("init", options);
        if (initM.length() > 0) {
            this.setInitializationMethod(new SelectedTag(Integer.parseInt(initM), TAGS_SELECTION));
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
        result.add("" + this.getInitializationMethod().getSelectedTag().getID());
        if (this.m_speedUpDistanceCompWithCanopies) {
            result.add("-C");
        }

        result.add("-max-candidates");
        result.add("" + this.getCanopyMaxNumCanopiesToHoldInMemory());
        result.add("-periodic-pruning");
        result.add("" + this.getCanopyPeriodicPruningRate());
        result.add("-min-density");
        result.add("" + this.getCanopyMinimumCanopyDensity());
        result.add("-t1");
        result.add("" + this.getCanopyT1());
        result.add("-t2");
        result.add("" + this.getCanopyT2());
        if (this.m_displayStdDevs) {
            result.add("-V");
        }

        if (this.m_dontReplaceMissing) {
            result.add("-M");
        }

        result.add("-N");
        result.add("" + this.getNumClusters());
        result.add("-A");
        result.add((this.m_DistanceFunction.getClass().getName() + " " + Utils.joinOptions(this.m_DistanceFunction.getOptions())).trim());
        result.add("-I");
        result.add("" + this.getMaxIterations());
        if (this.m_PreserveOrder) {
            result.add("-O");
        }

        if (this.m_FastDistanceCalc) {
            result.add("-fast");
        }

        result.add("-num-slots");
        result.add("" + this.getNumExecutionSlots());
        Collections.addAll(result, super.getOptions());
        return result.toArray(new String[0]);
    }

    public String toString() {
        if (this.m_ClusterCentroids == null) {
            return "No clusterer built yet!";
        } else {
            int maxWidth = 0;
            int maxAttWidth = 0;
            boolean containsNumeric = false;

            int maxV;
            for (int i = 0; i < this.m_NumClusters; ++i) {
                for (maxV = 0; maxV < this.m_ClusterCentroids.numAttributes(); ++maxV) {
                    if (this.m_ClusterCentroids.attribute(maxV).name().length() > maxAttWidth) {
                        maxAttWidth = this.m_ClusterCentroids.attribute(maxV).name().length();
                    }

                    if (this.m_ClusterCentroids.attribute(maxV).isNumeric()) {
                        containsNumeric = true;
                        double width = Math.log(Math.abs(this.m_ClusterCentroids.instance(i).value(maxV))) / Math.log(10.0D);
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

            String clustNum;
            for (int i = 0; i < this.m_ClusterCentroids.numAttributes(); ++i) {
                if (this.m_ClusterCentroids.attribute(i).isNominal()) {
                    Attribute a = this.m_ClusterCentroids.attribute(i);

                    for (i = 0; i < this.m_ClusterCentroids.numInstances(); ++i) {
                        clustNum = a.value((int) this.m_ClusterCentroids.instance(i).value(i));
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

            if (this.m_displayStdDevs) {
                for (int i = 0; i < this.m_ClusterCentroids.numAttributes(); ++i) {
                    if (this.m_ClusterCentroids.attribute(i).isNominal()) {
                        maxV = Utils.maxIndex(this.m_FullNominalCounts[i]);
                        int percent = 6;
                        clustNum = "" + this.m_FullNominalCounts[i][maxV];
                        if (clustNum.length() + percent > maxWidth) {
                            maxWidth = clustNum.length() + 1;
                        }
                    }
                }
            }

            double[] var21 = this.m_ClusterSizes;
            maxV = var21.length;

            String strVal;
            for (int i = 0; i < maxV; ++i) {
                double m_ClusterSize = var21[i];
                strVal = "(" + m_ClusterSize + ")";
                if (strVal.length() > maxWidth) {
                    maxWidth = strVal.length();
                }
            }

            if (this.m_displayStdDevs && maxAttWidth < "missing".length()) {
                maxAttWidth = "missing".length();
            }

            String plusMinus = "+/-";
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

            if (maxWidth < "missing".length()) {
                maxWidth = "missing".length() + 1;
            }

            StringBuilder temp = new StringBuilder();
            temp.append("\nkMeans\n======\n");
            temp.append("\nNumber of iterations: ").append(this.m_Iterations);
            if (!this.m_FastDistanceCalc) {
                temp.append("\n");
                if (this.m_DistanceFunction instanceof EuclideanDistance) {
                    temp.append("Within cluster sum of squared errors: ").append(Utils.sum(this.m_squaredErrors));
                } else {
                    temp.append("Sum of within cluster distances: ").append(Utils.sum(this.m_squaredErrors));
                }
            }

            temp.append("\n\nInitial starting points (");
            switch (this.m_initializationMethod) {
                case 1:
                    temp.append("k-means++");
                    break;
                case 2:
                    temp.append("canopy");
                    break;
                case 3:
                    temp.append("farthest first");
                    break;
                default:
                    temp.append("random");
            }

            temp.append("):\n");
            if (this.m_initializationMethod != 2) {
                temp.append("\n");

                for (int i = 0; i < this.m_initialStartPoints.numInstances(); ++i) {
                    temp.append("Cluster ").append(i).append(": ").append(this.m_initialStartPoints.instance(i)).append("\n");
                }
            } else {
                temp.append(this.m_canopyClusters.toString(false));
            }

            if (this.m_speedUpDistanceCompWithCanopies) {
                temp.append("\nReduced number of distance calculations by using canopies.");
                if (this.m_initializationMethod != 2) {
                    temp.append("\nCanopy T2 radius: ").append(String.format("%-10.3f", this.m_canopyClusters.getActualT2()));
                    temp.append("\nCanopy T1 radius: ").append(String.format("%-10.3f", this.m_canopyClusters.getActualT1())).append("\n");
                }
            }

            if (!this.m_dontReplaceMissing) {
                temp.append("\nMissing values globally replaced with mean/mode");
            }

            temp.append("\n\nFinal cluster centroids:\n");
            temp.append(this.pad("Cluster#", " ", maxAttWidth + maxWidth * 2 + 2 - "Cluster#".length(), true));
            temp.append("\n");
            temp.append(this.pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));
            temp.append(this.pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

            for (int i = 0; i < this.m_NumClusters; ++i) {
                clustNum = "" + i;
                temp.append(this.pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
            }

            temp.append("\n");
            String cSize = "(" + Utils.sum(this.m_ClusterSizes) + ")";
            temp.append(this.pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(), true));

            int i;
            for (i = 0; i < this.m_NumClusters; ++i) {
                cSize = "(" + this.m_ClusterSizes[i] + ")";
                temp.append(this.pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
            }

            temp.append("\n");
            temp.append(this.pad("", "=", maxAttWidth + maxWidth * (this.m_ClusterCentroids.numInstances() + 1) + this.m_ClusterCentroids.numInstances() + 1, true));
            temp.append("\n");

            for (i = 0; i < this.m_ClusterCentroids.numAttributes(); ++i) {
                String attName = this.m_ClusterCentroids.attribute(i).name();
                temp.append(attName);

                for (int j = 0; j < maxAttWidth - attName.length(); ++j) {
                    temp.append(" ");
                }

                String valMeanMode;
                if (this.m_ClusterCentroids.attribute(i).isNominal()) {
                    if (this.m_FullMeansOrMediansOrModes[i] == -1.0D) {
                        valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = this.pad(strVal = this.m_ClusterCentroids.attribute(i).value((int) this.m_FullMeansOrMediansOrModes[i]), " ", maxWidth + 1 - strVal.length(), true);
                    }
                } else if (Double.isNaN(this.m_FullMeansOrMediansOrModes[i])) {
                    valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                } else {
                    valMeanMode = this.pad(strVal = Utils.doubleToString(this.m_FullMeansOrMediansOrModes[i], maxWidth, 4).trim(), " ", maxWidth + 1 - strVal.length(), true);
                }

                temp.append(valMeanMode);

                for (int j = 0; j < this.m_NumClusters; ++j) {
                    if (this.m_ClusterCentroids.attribute(i).isNominal()) {
                        if (this.m_ClusterCentroids.instance(j).isMissing(i)) {
                            valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                        } else {
                            valMeanMode = this.pad(strVal = this.m_ClusterCentroids.attribute(i).value((int) this.m_ClusterCentroids.instance(j).value(i)), " ", maxWidth + 1 - strVal.length(), true);
                        }
                    } else if (this.m_ClusterCentroids.instance(j).isMissing(i)) {
                        valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = this.pad(strVal = Utils.doubleToString(this.m_ClusterCentroids.instance(j).value(i), maxWidth, 4).trim(), " ", maxWidth + 1 - strVal.length(), true);
                    }

                    temp.append(valMeanMode);
                }

                temp.append("\n");
                if (this.m_displayStdDevs) {
                    String stdDevVal;
                    if (this.m_ClusterCentroids.attribute(i).isNominal()) {
                        Attribute a = this.m_ClusterCentroids.attribute(i);

                        for (int j = 0; j < a.numValues(); ++j) {
                            String val = "  " + a.value(j);
                            temp.append(this.pad(val, " ", maxAttWidth + 1 - val.length(), false));
                            double count = this.m_FullNominalCounts[i][j];
                            int k = (int) (this.m_FullNominalCounts[i][j] / Utils.sum(this.m_ClusterSizes) * 100.0D);
                            String percentS = "" + k + "%)";
                            percentS = this.pad(percentS, " ", 5 - percentS.length(), true);
                            stdDevVal = "" + count + " (" + percentS;
                            stdDevVal = this.pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                            temp.append(stdDevVal);

                            for (k = 0; k < this.m_NumClusters; ++k) {
                                k = (int) (this.m_ClusterNominalCounts[k][i][j] / this.m_ClusterSizes[k] * 100.0D);
                                percentS = "" + k + "%)";
                                percentS = this.pad(percentS, " ", 5 - percentS.length(), true);
                                stdDevVal = "" + this.m_ClusterNominalCounts[k][i][j] + " (" + percentS;
                                stdDevVal = this.pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                                temp.append(stdDevVal);
                            }

                            temp.append("\n");
                        }

                        if (this.m_FullMissingCounts[i] > 0.0D) {
                            temp.append(this.pad("  missing", " ", maxAttWidth + 1 - "  missing".length(), false));
                            double count = this.m_FullMissingCounts[i];
                            int percent = (int) (this.m_FullMissingCounts[i] / Utils.sum(this.m_ClusterSizes) * 100.0D);
                            String percentS = "" + percent + "%)";
                            percentS = this.pad(percentS, " ", 5 - percentS.length(), true);
                            stdDevVal = "" + count + " (" + percentS;
                            stdDevVal = this.pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                            temp.append(stdDevVal);

                            for (int k = 0; k < this.m_NumClusters; ++k) {
                                percent = (int) (this.m_ClusterMissingCounts[k][i] / this.m_ClusterSizes[k] * 100.0D);
                                percentS = "" + percent + "%)";
                                percentS = this.pad(percentS, " ", 5 - percentS.length(), true);
                                stdDevVal = "" + this.m_ClusterMissingCounts[k][i] + " (" + percentS;
                                stdDevVal = this.pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                                temp.append(stdDevVal);
                            }

                            temp.append("\n");
                        }

                        temp.append("\n");
                    } else {
                        if (Double.isNaN(this.m_FullMeansOrMediansOrModes[i])) {
                            stdDevVal = this.pad("--", " ", maxAttWidth + maxWidth + 1 - 2, true);
                        } else {
                            stdDevVal = this.pad(strVal = plusMinus + Utils.doubleToString(this.m_FullStdDevs[i], maxWidth, 4).trim(), " ", maxWidth + maxAttWidth + 1 - strVal.length(), true);
                        }

                        temp.append(stdDevVal);

                        for (int j = 0; j < this.m_NumClusters; ++j) {
                            if (this.m_ClusterCentroids.instance(j).isMissing(i)) {
                                stdDevVal = this.pad("--", " ", maxWidth + 1 - 2, true);
                            } else {
                                stdDevVal = this.pad(strVal = plusMinus + Utils.doubleToString(this.m_ClusterStdDevs.instance(j).value(i), maxWidth, 4).trim(), " ", maxWidth + 1 - strVal.length(), true);
                            }

                            temp.append(stdDevVal);
                        }

                        temp.append("\n\n");
                    }
                }
            }

            temp.append("\n\n");
            return temp.toString();
        }
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

    public Instances getClusterCentroids() {
        return this.m_ClusterCentroids;
    }

    public Instances getClusterStandardDevs() {
        return this.m_ClusterStdDevs;
    }

    public double[][][] getClusterNominalCounts() {
        return this.m_ClusterNominalCounts;
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
        protected int m_start;
        protected int m_end;
        protected Instances m_inst;
        protected int[] m_clusterAssignments;

        public KMeansClusterTask(Instances inst, int start, int end, int[] clusterAssignments) {
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

        protected int clusterInstance(Instance inst, long[] instanceCanopies) {
            double minDist = 2.147483647E9D;
            int bestCluster = 0;

            for (int i = 0; i < MISimpleKMeans.this.m_NumClusters; ++i) {
                if (MISimpleKMeans.this.m_speedUpDistanceCompWithCanopies && instanceCanopies != null && instanceCanopies.length > 0) {
                    try {
                        if (!Canopy.nonEmptyCanopySetIntersection(MISimpleKMeans.this.m_centroidCanopyAssignments.get(i), instanceCanopies)) {
                            continue;
                        }
                    } catch (Exception var10) {
                        var10.printStackTrace();
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

    private class KMeansComputeCentroidTask implements Callable<double[]> {
        protected Instances m_cluster;
        protected int m_centroidIndex;

        public KMeansComputeCentroidTask(int centroidIndex, Instances cluster) {
            this.m_cluster = cluster;
            this.m_centroidIndex = centroidIndex;
        }

        public double[] call() {
            return MISimpleKMeans.this.moveCentroid(this.m_centroidIndex, this.m_cluster, true, false);
        }
    }
}
