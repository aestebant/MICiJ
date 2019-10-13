package evaluators;

import distances.HausdorffDistance;
import weka.clusterers.Clusterer;
import weka.clusterers.DensityBasedClusterer;
import weka.clusterers.UpdateableClusterer;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.MethodDescriptor;
import java.io.*;
import java.lang.reflect.Method;
import java.util.*;

public class ClusterEvaluation implements Serializable, RevisionHandler {
    private Clusterer clusterer;

    private Vector<Integer> clusterAssignments;
    private int[] instancesPerCluster;
    private int unclusteredInstances;

    private HausdorffDistance distFunction = new HausdorffDistance();

    private final StringBuffer resultBuffer;

    private int[] classToCluster = null;

    public void setClusterer(Clusterer clusterer) {
        this.clusterer = clusterer;
    }

    public String clusterResultsToString() {
        return this.resultBuffer.toString();
    }

    public ClusterEvaluation(Clusterer clusterer) {
        this.setClusterer(clusterer);
        this.resultBuffer = new StringBuffer();
    }

    public void evaluateClusterer(String sourceFileName, int classIdx, Remove removeClass) throws Exception {

        countInstancesPerCluster(sourceFileName, classIdx, removeClass);
        double logLikelihood = getLogLikelihood(sourceFileName);
        double silhouette = indexSilhouette(sourceFileName);
        double sdbw = indexSDbw(sourceFileName);

        int totalClusteredInstances = Utils.sum(instancesPerCluster);
        if (totalClusteredInstances > 0) {
            resultBuffer.append("Clustered Instances:\n\n");

            int clustFieldWidth = (int) (Math.log(clusterer.numberOfClusters()) / Math.log(10D) + 1D);
            int numInstFieldWidth = (int) (Math.log(clusterAssignments.size()) / Math.log(10D) + 1D);
            for (int i = 0; i < clusterer.numberOfClusters(); ++i) {
                if (instancesPerCluster[i] > 0) {
                    resultBuffer.append(Utils.doubleToString((double) i, clustFieldWidth, 0))
                            .append("      ").append(Utils.doubleToString(instancesPerCluster[i], numInstFieldWidth, 0))
                            .append(" (").append(Utils.doubleToString((double) instancesPerCluster[i] / totalClusteredInstances * 100, 3, 0)).append("%)\n");
                }
            }
        }
        if (unclusteredInstances > 0) {
            resultBuffer.append("\nUnclustered instances: ").append(unclusteredInstances);
        }

        resultBuffer.append("\nSilhouette index: ").append(silhouette);
        resultBuffer.append("\nS_Dbw: ").append(sdbw);

        if (clusterer instanceof DensityBasedClusterer) {
            resultBuffer.append("\n\nLog likelihood: ")
                    .append(Utils.doubleToString(logLikelihood, 1, 5)).append("\n");
        }

        if (classIdx > -1) {
            this.evaluateClustersWithRespectToClass(sourceFileName, classIdx);
        }
    }

    private void countInstancesPerCluster(String sourceFileName, int classIdx, Remove removeClass)  throws Exception {
        DataSource source = new DataSource(sourceFileName);
        Instances instances = source.getStructure(classIdx);
        int numClusters = clusterer.numberOfClusters();

        instancesPerCluster = new int[numClusters];

        assert instances != null;
        clusterAssignments = new Vector<>(instances.numInstances());

        Instances forBatchPredictors = removeClass != null ?
                new Instances(removeClass.getOutputFormat(), 0) : new Instances(instances, 0);

        while (source.hasMoreElements(instances)) {
            Instance instance = source.nextElement(instances);
            if (classIdx > -1) {
                assert removeClass != null;
                removeClass.input(instance);
                try {
                    removeClass.batchFinished();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                instance = removeClass.output();
            }

            if (clusterer instanceof BatchPredictor && ((BatchPredictor) clusterer).implementsMoreEfficientBatchPrediction()) {
                forBatchPredictors.add(instance);
            } else {
                try {
                    int clusterIdx = clusterer.clusterInstance(instance);
                    instancesPerCluster[clusterIdx]++;
                    clusterAssignments.add(clusterIdx);
                } catch (Exception e) {
                    // Unclustered instance
                    unclusteredInstances++;
                    clusterAssignments.add(-1);
                }
            }
        }

        if (clusterer instanceof BatchPredictor && ((BatchPredictor) clusterer).implementsMoreEfficientBatchPrediction()) {
            double[][] dists = new double[0][];
            try {
                dists = ((BatchPredictor) clusterer).distributionsForInstances(forBatchPredictors);
            } catch (Exception e) {
                e.printStackTrace();
            }
            for (double[] d : dists) {
                int clusterIdx = Utils.maxIndex(d);
                unclusteredInstances++;
                clusterAssignments.add(clusterIdx);
            }
        }
    }


    private double getLogLikelihood(String sourceFileName)  throws Exception {
        DataSource source = new DataSource(sourceFileName);
        Instances instances = source.getDataSet();
        Instances forBatchPredictors = clusterer instanceof BatchPredictor && ((BatchPredictor) clusterer)
                .implementsMoreEfficientBatchPrediction() ? new Instances(instances, 0) : null;

        double logLikelihood = 0D;

        if (clusterer instanceof DensityBasedClusterer) {
            while (source.hasMoreElements(instances)) {
                Instance inst = source.nextElement(instances);
                if (forBatchPredictors != null) {
                    forBatchPredictors.add(inst);
                } else {
                    try {
                        logLikelihood += ((DensityBasedClusterer) clusterer).logDensityForInstance(inst);
                    } catch (Exception ignored) {
                    }
                }
            }
        }
        logLikelihood /= Utils.sum(instancesPerCluster);

        return logLikelihood;
    }

    private double indexSilhouette(String sourceFileName)  throws Exception {
        DataSource source = new DataSource(sourceFileName);
        Instances instances = source.getDataSet();
        int numInstances = clusterAssignments.size();
        int numClusters = clusterer.numberOfClusters();

        double[][] distances = new double[numInstances][numInstances];
        for (int i = 0; i < clusterAssignments.size(); ++i) {
            for (int j = i+1; j < clusterAssignments.size(); ++j) {
                distances[i][j] = distFunction.distance(instances.get(i), instances.get(j));
                distances[j][i] = distances[i][j];
            }
        }

        double silhouette = 0D;

        for (int clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx) {
            double sumCluster = 0D;
            for (int i = 0; i < clusterAssignments.size(); ++i) {
                if (clusterAssignments.get(i) == clusterIdx) {
                    double[] sum = new double[numClusters];
                    for (int j = 0; j < clusterAssignments.size(); ++j) {
                        sum[clusterIdx] += distances[i][j];
                    }
                    double a = sum[clusterIdx] / (double) (instancesPerCluster[clusterIdx] - 1);
                    List<Double> possibleB = new ArrayList<>(numClusters-1);
                    for (int j = 0; j < numClusters; ++j) {
                        if (j != clusterIdx)
                            possibleB.add(sum[clusterIdx]);
                    }
                    double b;
                    try {
                        b = Collections.min(possibleB);
                    }catch (Exception ignored) {
                        b = 0;
                    }

                    sumCluster += (b-a) / Math.max(b, a);
                }
            }
            silhouette += sumCluster/instancesPerCluster[clusterIdx];
        }
        return silhouette / numClusters;
    }

    private double indexSDbw(String sourceFileName) throws Exception {
        DataSource source = new DataSource(sourceFileName);
        Instances instances = source.getDataSet();
        int numClusters = clusterer.numberOfClusters();
        int numAttributes = instances.get(0).relationalValue(1).numAttributes();

        List<Instances> bagsSumaries = new ArrayList<>(numClusters);
        for (int i = 0; i < numClusters; ++i) {
            bagsSumaries.add(new Instances(instances.get(0).relationalValue(1)));
            bagsSumaries.get(i).delete();
        }

        for (int i = 0; i < instances.size(); ++i) {
            Instance aux = instances.get(i).relationalValue(1).get(0);
            for (int j = 0; j < aux.numAttributes(); ++j) {
                aux.setValue(j, instances.get(i).relationalValue(1).meanOrMode(j));
            }
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx > -1)
                bagsSumaries.get(clusterIdx).add(aux);
        }

        Instances datasetSummary = new Instances(bagsSumaries.get(0));
        for (int i = 1; i < numClusters; ++i)
            datasetSummary.addAll(bagsSumaries.get(i));

        double[] varDataset = datasetSummary.variances();
        double euclidNormDataset = 0D;
        for (double v : varDataset)
            euclidNormDataset += v * v;
        euclidNormDataset = Math.sqrt(euclidNormDataset);

        List<Double> euclidNormCluster = new ArrayList<>(numClusters);
        double scat = 0D;
        for (int i = 0; i < numClusters; ++i) {
            double[] varCluster = bagsSumaries.get(i).variances();
            euclidNormCluster.add(0D);
            for (double v : varCluster)
                euclidNormCluster.set(i, euclidNormCluster.get(i) + v*v);
            scat += euclidNormCluster.get(i) / euclidNormDataset;
        }
        scat /= numClusters;

        double stdev = euclidNormCluster.stream().mapToDouble(Double::doubleValue).sum();
        stdev = Math.sqrt(stdev) / numClusters;

        List<Instance> clustersCenters = new ArrayList<>(numClusters);
        for (int i = 0; i < numClusters; ++i) {
            Instance mean = new DenseInstance(numAttributes);
            for (int j = 0; j < numAttributes; ++j) {
                mean.setValue(j, bagsSumaries.get(i).meanOrMode(j));
            }
            clustersCenters.add(mean);
        }

        double densBw = 0D;
        for (int i = 0; i < numClusters; ++i) {
            for (int j = 0; j < numClusters; ++j) {
                if (i == j)
                    break;

                Instance u = new DenseInstance(numAttributes);

                for (int k = 0; k < numAttributes; ++k)
                    u.setValue(k, (clustersCenters.get(i).value(k) + clustersCenters.get(j).value(k) / 2));
                double densityU = 0D;
                double densityI = 0D;
                double densityJ = 0D;
                for (int k = 0; k < clusterAssignments.size(); ++k) {
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
        densBw /= (numClusters * (numClusters-1));

        return scat + densBw;
    }

    private void evaluateClustersWithRespectToClass(String sourceFileName, int classIdx) throws Exception {
        DataSource source = new DataSource(sourceFileName);

        Instances instances = source.getStructure(classIdx);
        instances.setClassIndex(classIdx);
        int numClusters = clusterer.numberOfClusters();
        int numClasses = instances.classAttribute().numValues();

        int[][] counts = new int[numClusters][numClasses];
        int[] clusterTotals = new int[numClusters];

        for(int i = 0; source.hasMoreElements(instances); ++i) {
            Instance instance = source.nextElement(instances);
            if (clusterAssignments.get(i) != -1 && !instance.classIsMissing()) {
                counts[clusterAssignments.get(i)][(int) instance.classValue()]++;
                clusterTotals[clusterAssignments.get(i)]++;
            }
        }

        double[] best = new double[numClusters + 1];
        best[numClusters] = 1.7976931348623157E308D;
        double[] current = new double[numClusters + 1];
        mapClasses(numClusters, 0, counts, clusterTotals, current, best, 0);

        resultBuffer.append("\n\nClass attribute: ").append(instances.classAttribute().name()).append("\n");

        resultBuffer.append("Classes to Clusters:\n");
        String matrixString = toMatrixString(counts, clusterTotals, new Instances(instances, 0));
        resultBuffer.append(matrixString).append("\n");

        int Cwidth = 1 + (int)(Math.log(numClusters) / Math.log(10D));
        for(int i = 0; i < numClusters; ++i) {
            if (clusterTotals[i] > 0) {
                resultBuffer.append("Cluster ").append(Utils.doubleToString((double) i, Cwidth, 0));
                resultBuffer.append(" <-- ");
                if (best[i] < 0) {
                    resultBuffer.append("No class\n");
                } else {
                    resultBuffer.append(instances.classAttribute().value((int)best[i])).append("\n");
                }
            }
        }

        resultBuffer.append("\nIncorrectly clustered instances :\t").append((int) best[numClusters]).append("\t(")
                .append(Utils.doubleToString(best[numClusters] / clusterAssignments.size() * 100.0D, 8, 4)).append(" %)\n");
        classToCluster = new int[numClusters];

        for(int i = 0; i < numClusters; ++i) {
            classToCluster[i] = (int)best[i];
        }
    }

    private static void mapClasses(int numClusters, int lev, int[][] counts, int[] clusterTotals, double[] current, double[] best, int error) {
        if (lev == numClusters) {
            if ((double) error < best[numClusters]) {
                best[numClusters] = error;
                System.arraycopy(current, 0, best, 0, numClusters);
            }
        } else if (clusterTotals[lev] == 0) {
            current[lev] = -1.0D;
            mapClasses(numClusters,lev + 1, counts, clusterTotals, current, best, error);
        } else {
            current[lev] = -1.0D;
            mapClasses(numClusters,lev + 1, counts, clusterTotals, current, best, error + clusterTotals[lev]);
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

    private String toMatrixString(int[][] counts, int[] clusterTotals, Instances inst) {
        StringBuilder matrix = new StringBuilder();
        int maxval = 0;

        int numClusters = 0;
        try {
            numClusters = clusterer.numberOfClusters();
        } catch (Exception e) {
            e.printStackTrace();
        }

        for(int Cwidth = 0; Cwidth < numClusters; ++Cwidth) {
            for(int i = 0; i < counts[Cwidth].length; ++i) {
                if (counts[Cwidth][i] > maxval) {
                    maxval = counts[Cwidth][i];
                }
            }
        }

        int Cwidth = 1 + Math.max((int)(Math.log((double) maxval) / Math.log(10D)), (int)(Math.log((double) numClusters) / Math.log(10D)));
        matrix.append("\n");
        for(int i = 0; i < numClusters; ++i) {
            if (clusterTotals[i] > 0) {
                matrix.append(" ").append(Utils.doubleToString((double)i, Cwidth, 0));
            }
        }
        matrix.append("  <-- assigned to cluster\n");
        for(int i = 0; i < counts[0].length; ++i) {
            for(int j = 0; j < numClusters; ++j) {
                if (clusterTotals[j] > 0) {
                    matrix.append(" ").append(Utils.doubleToString((double)counts[j][i], Cwidth, 0));
                }
            }
            matrix.append(" | ").append(inst.classAttribute().value(i)).append("\n");
        }
        return matrix.toString();
    }

    public static String evaluateClusterer(Clusterer clusterer, String[] options) throws Exception {

        boolean forceBatch = Utils.getFlag("force-batch-training", options);
        boolean updateable = clusterer instanceof UpdateableClusterer && !forceBatch;


        if (Utils.getFlag('h', options) || Utils.getFlag("help", options)) {
            boolean globalInfo = Utils.getFlag("synopsis", options) || Utils.getFlag("info", options);
            throw new Exception("Help requested." + makeOptionString(clusterer, globalInfo));
        }

        String trainFileName;
        String testFileName;
        String objectInputFileName;
        String objectOutputFileName;
        String graphFileName;
        boolean printClusterAssignments = false;
        Range attributesToOutput = null;
        int seed = 1;
        int folds = 10;
        boolean doXval = false;
        try {
            objectInputFileName = Utils.getOption('l', options);
            objectOutputFileName = Utils.getOption('d', options);
            trainFileName = Utils.getOption('t', options);
            testFileName = Utils.getOption('T', options);
            graphFileName = Utils.getOption('g', options);

            String attributeRangeString;
            try {
                attributeRangeString = Utils.getOption('p', options);
            } catch (Exception var29) {
                throw new Exception(var29.getMessage() + "\nNOTE: the -p option has changed. It now expects a parameter specifying a range of attributes to list with the predictions. Use '-p 0' for none.");
            }

            if (attributeRangeString.length() != 0) {
                printClusterAssignments = true;
                if (!attributeRangeString.equals("0")) {
                    attributesToOutput = new Range(attributeRangeString);
                }
            }

            if (trainFileName.length() == 0) {
                if (objectInputFileName.length() == 0) {
                    throw new Exception("No training file and no object input file given.");
                }

                if (testFileName.length() == 0) {
                    throw new Exception("No training file and no test file given.");
                }
            } else if (objectInputFileName.length() != 0 && !printClusterAssignments) {
                throw new Exception("Can't use both train and model file unless -p specified.");
            }

            String seedString = Utils.getOption('s', options);
            if (seedString.length() != 0) {
                seed = Integer.parseInt(seedString);
            }

            String foldsString = Utils.getOption('x', options);
            if (foldsString.length() != 0) {
                folds = Integer.parseInt(foldsString);
                doXval = true;
            }
        } catch (Exception e) {
            throw new Exception('\n' + e.getMessage() + makeOptionString(clusterer, false));
        }

        DataSource train = null;
        Instances trainInstances = null;
        int classIdx = -1;

        if (trainFileName.length() != 0) {
            train = new DataSource(trainFileName);
            trainInstances = train.getStructure();
            String classString = Utils.getOption('c', options);

            if (classString.length() != 0) {
                if (classString.compareTo("last") == 0) {
                    classIdx = trainInstances.numAttributes() - 1;
                } else if (classString.compareTo("first") == 0) {
                    classIdx = 0;
                } else {
                    classIdx = Integer.parseInt(classString);
                }
                /*if (classIdx != -1) {
                    if (doXval || testFileName.length() != 0) {
                        throw new Exception("Can only do class based evaluation on the training data");
                    }
                    if (objectInputFileName.length() != 0) {
                        throw new Exception("Can't load a clusterer and do class based evaluation");
                    }
                    if (objectOutputFileName.length() != 0) {
                        throw new Exception("Can't do class based evaluation and save clusterer");
                    }
                }*/
            } else if (trainInstances.classIndex() != -1) {
                classIdx = trainInstances.classIndex();
                System.err.println("Note: using class attribute from dataset, i.e., attribute #" + classIdx);
            }

            if (classIdx != -1) {
                if (classIdx < 0 || classIdx > trainInstances.numAttributes() - 1) {
                    throw new Exception("Class is out of range!");
                }
                if (!trainInstances.attribute(classIdx).isNominal()) {
                    throw new Exception("Class must be nominal!");
                }
            }
        }

        String[] savedOptions = new String[options.length];
        System.arraycopy(options, 0, savedOptions, 0, options.length);

        if (objectInputFileName.length() != 0) {
            Utils.checkForRemainingOptions(options);
        }
        if (clusterer instanceof OptionHandler) {
            ((OptionHandler)clusterer).setOptions(options);
        }

        Utils.checkForRemainingOptions(options);

        Instances trainHeader = trainInstances;

        if (objectInputFileName.length() != 0) {
            ObjectInputStream objectInputStream = new ObjectInputStream(new BufferedInputStream(new FileInputStream(objectInputFileName)));
            clusterer = (Clusterer)objectInputStream.readObject();
            try {
                trainHeader = (Instances) objectInputStream.readObject();
            } catch (Exception e) {
                e.printStackTrace();
            }
            objectInputStream.close();
        }

        Remove removeClass = null;
        if (classIdx > -1) {
            removeClass = new Remove();
            removeClass.setAttributeIndices(String.valueOf(classIdx+1)); // No sé por qué aquí cuenta los índices empezando en 1
            removeClass.setInvertSelection(false);
            try {
                removeClass.setInputFormat(trainInstances);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if (classIdx > -1) {
            if (updateable)
                trainInstances = Filter.useFilter(trainInstances, removeClass);
            else
                trainInstances = Filter.useFilter(train.getDataSet(), removeClass);
        }

        clusterer.buildClusterer(trainInstances);

        if (updateable) {
            assert train != null;
            while (train.hasMoreElements(trainInstances)) {
                Instance instance = train.nextElement(trainInstances);
                if (classIdx > -1) {
                    removeClass.input(instance);
                    removeClass.batchFinished();
                    instance = removeClass.output();
                }
                ((UpdateableClusterer) clusterer).updateClusterer(instance);
            }
            ((UpdateableClusterer) clusterer).updateFinished();
        }

        ClusterEvaluation ceTrain = new ClusterEvaluation(clusterer);
        ceTrain.evaluateClusterer(trainFileName, classIdx, removeClass);

        StringBuilder result = new StringBuilder();

        if (printClusterAssignments && testFileName.length() == 0)
            result.append(printClusterings(clusterer, train, attributesToOutput));
        result.append("\n\n=== Clustering stats for training data ===\n\n").append(ceTrain.clusterResultsToString());

        if (testFileName.length() != 0) {
            DataSource test = new DataSource(testFileName);
            Instances testInstances = test.getStructure();
            assert trainHeader != null;
            if (!trainHeader.equalHeaders(testInstances)) {
                throw new Exception("Training and testing data are not compatible\n" + trainHeader.equalHeadersMsg(testInstances));
            }

            ClusterEvaluation ceTest = new ClusterEvaluation(clusterer);
            ceTest.evaluateClusterer(testFileName, classIdx, removeClass);

            if (printClusterAssignments)
                result.append(printClusterings(clusterer, test, attributesToOutput));
            result.append("\n\n=== Clustering stats for testing data ===\n\n").append(ceTest.clusterResultsToString());
        }

        if (clusterer instanceof DensityBasedClusterer && doXval && testFileName.length() == 0 && objectInputFileName.length() == 0) {
            Random random = new Random(seed);
            random.setSeed(seed);
            trainInstances = train.getDataSet();
            trainInstances.randomize(random);

            result.append(crossValidateModel(clusterer.getClass().getName(), trainInstances, folds, savedOptions, random));
        }

        if (objectOutputFileName.length() != 0) {
            saveClusterer(objectOutputFileName, clusterer, trainHeader);
        }

        if (clusterer instanceof Drawable && graphFileName.length() != 0) {
            BufferedWriter writer = new BufferedWriter(new FileWriter(graphFileName));
            writer.write(((Drawable)clusterer).graph());
            writer.newLine();
            writer.flush();
            writer.close();
        }

        return result.toString();
    }

    private static void saveClusterer(String fileName, Clusterer clusterer, Instances header) throws Exception {
        ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(fileName)));
        oos.writeObject(clusterer);
        if (header != null) {
            oos.writeObject(header);
        }
        oos.flush();
        oos.close();
    }

    private static double crossValidateModel(DensityBasedClusterer clusterer, Instances data, int numFolds, Random random) throws Exception {
        double foldAv = 0.0D;
        data = new Instances(data);
        data.randomize(random);

        for(int i = 0; i < numFolds; ++i) {
            Instances train = data.trainCV(numFolds, i, random);
            clusterer.buildClusterer(train);
            Instances test = data.testCV(numFolds, i);
            for(int j = 0; j < test.numInstances(); ++j) {
                try {
                    foldAv += clusterer.logDensityForInstance(test.instance(j));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        return foldAv / (double)data.numInstances();
    }

    private static String crossValidateModel(String clustererString, Instances data, int numFolds, String[] options, Random random) throws Exception {
        Clusterer clusterer;
        String[] savedOptions = null;
        double CvAv;
        StringBuilder CvString = new StringBuilder();
        if (options != null) {
            savedOptions = new String[options.length];
        }

        data = new Instances(data);

        try {
            clusterer = (Clusterer)Class.forName(clustererString).newInstance();
        } catch (Exception var12) {
            throw new Exception("Can't find class with name " + clustererString + '.');
        }

        if (!(clusterer instanceof DensityBasedClusterer)) {
            throw new Exception(clustererString + " must be a distrinbution clusterer.");
        } else {
            if (options != null) {
                System.arraycopy(options, 0, savedOptions, 0, options.length);
            }

            if (clusterer instanceof OptionHandler) {
                try {
                    ((OptionHandler)clusterer).setOptions(savedOptions);
                    Utils.checkForRemainingOptions(savedOptions);
                } catch (Exception var11) {
                    throw new Exception("Can't parse given options in cross-validation!");
                }
            }
            CvAv = crossValidateModel((DensityBasedClusterer)clusterer, data, numFolds, random);
            CvString.append("\n").append(numFolds).append(" fold CV Log Likelihood: ").append(Utils.doubleToString(CvAv, 6, 4)).append("\n");
            return CvString.toString();
        }
    }

    private static String printClusterings(Clusterer clusterer, DataSource source, Range attributesToOutput) throws Exception {
        StringBuilder result = new StringBuilder();

        Instances structure = source.getStructure();
        Instances forBatchPredictors = clusterer instanceof BatchPredictor && ((BatchPredictor) clusterer).implementsMoreEfficientBatchPrediction() ?
                new Instances(source.getStructure(), 0) : null;

        int instIdx = 0;
        while(source.hasMoreElements(structure)) {
            Instance instance = source.nextElement(structure);
            if (forBatchPredictors != null) {
                forBatchPredictors.add(instance);
            } else {
                try {
                    int clusterIdx = clusterer.clusterInstance(instance);
                    result.append(instIdx).append(" ").append(clusterIdx).append(" ").append(attributeValuesString(instance, attributesToOutput)).append("\n");
                } catch (Exception var16) {
                    result.append(instIdx).append(" Unclustered ").append(attributeValuesString(instance, attributesToOutput)).append("\n");
                }
                instIdx++;
            }
        }

        if (forBatchPredictors != null) {
            double[][] dists = ((BatchPredictor)clusterer).distributionsForInstances(forBatchPredictors);

            for (double[] d : dists) {
                int clusterIdx = Utils.maxIndex(d);
                result.append(instIdx).append(" ").append(clusterIdx).append(" ")
                        .append(attributeValuesString(forBatchPredictors.instance(instIdx), attributesToOutput)).append("\n");
                instIdx++;
            }
        }
        return result.toString();
    }

    private static String attributeValuesString(Instance instance, Range attRange) {
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

    private static String makeOptionString(Clusterer clusterer, boolean globalInfo) {
        StringBuilder optionsText = new StringBuilder();
        optionsText.append("\n\nGeneral options:\n\n");
        optionsText.append("-h or -help\n");
        optionsText.append("\tOutput help information.\n");
        optionsText.append("-synopsis or -info\n");
        optionsText.append("\tOutput synopsis for clusterer (use in conjunction  with -h)\n");
        optionsText.append("-t <name of training file>\n");
        optionsText.append("\tSets training file.\n");
        optionsText.append("-T <name of test file>\n");
        optionsText.append("\tSets test file.\n");
        optionsText.append("-force-batch-training\n");
        optionsText.append("\tAlways train the clusterer in batch mode, never incrementally.\n");
        optionsText.append("-l <name of input file>\n");
        optionsText.append("\tSets model input file.\n");
        optionsText.append("-d <name of output file>\n");
        optionsText.append("\tSets model output file.\n");
        optionsText.append("-p <attribute range>\n");
        optionsText.append("\tOutput predictions. Predictions are for training file\n\tif only training file is specified,\n\totherwise predictions are for the test file.\n\tThe range specifies attribute values to be output\n\twith the predictions. Use '-p 0' for none.\n");
        optionsText.append("-x <number of folds>\n");
        optionsText.append("\tOnly Distribution Clusterers can be cross validated.\n");
        optionsText.append("-s <random number seed>\n");
        optionsText.append("\tSets the seed for randomizing the data in cross-validation\n");
        optionsText.append("-c <class index>\n");
        optionsText.append("\tSet class attribute. If supplied, class is ignored");
        optionsText.append("\n\tduring clustering but is used in a classes to");
        optionsText.append("\n\tclusters evaluation.\n");
        if (clusterer instanceof Drawable) {
            optionsText.append("-g <name of graph file>\n");
            optionsText.append("\tOutputs the graph representation of the clusterer to the file.\n");
        }

        if (clusterer instanceof OptionHandler) {
            optionsText.append("\nOptions specific to ").append(clusterer.getClass().getName()).append(":\n\n");
            Enumeration enu = ((OptionHandler)clusterer).listOptions();

            while(enu.hasMoreElements()) {
                Option option = (Option)enu.nextElement();
                optionsText.append(option.synopsis()).append('\n');
                optionsText.append(option.description()).append("\n");
            }
        }

        if (globalInfo) {
            try {
                String gi = getGlobalInfo(clusterer);
                optionsText.append(gi);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return optionsText.toString();
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
}
