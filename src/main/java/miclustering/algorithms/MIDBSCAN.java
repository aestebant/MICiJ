package miclustering.algorithms;

import miclustering.algorithms.utils.DataObject;
import miclustering.algorithms.utils.Database;
import miclustering.distances.HausdorffDistance;
import miclustering.utils.LoadByName;
import weka.clusterers.AbstractClusterer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.text.DecimalFormat;
import java.util.*;

public class MIDBSCAN extends AbstractClusterer implements MIClusterer, OptionHandler, TechnicalInformationHandler {
    private double epsilon = 0.9;
    private int minPoints;
    private int numGeneratedClusters;
    private int numNoises;
    private DistanceFunction distFunction = new HausdorffDistance();
    private Database database;
    private int clusterID;
    private double elapsedTime;
    private boolean printClusterAssignments;
    private List<Integer> clusterAssignments;

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        return result;
    }

    @Override
    public void buildClusterer(Instances instances) throws Exception {
        this.getCapabilities().testWithFail(instances);
        long startTime = System.currentTimeMillis();
        if (minPoints < 2)
            minPoints = instances.get(0).relationalValue(1).numAttributes() + 1;
        numGeneratedClusters = 0;
        numNoises = 0;
        clusterID = 0;
        database = new Database(distFunction, instances);
        clusterAssignments = new ArrayList<>(instances.numInstances());

        for (Instance instance : database.getInstances()) {
            DataObject dataObject = new DataObject(instance, instance.stringValue(0));
            database.insert(dataObject);
        }

        Iterator<DataObject> it = database.dataObjectIterator();
        while (it.hasNext()) {
            DataObject bag = it.next();
            //System.out.println("ANALIZANDO BOLSA " + bag.getKey());
            if (bag.getClusterLabel() == DataObject.UNCLASSIFIED && this.expandCluster(bag)) {
                //System.out.println("HA ENTRADO EN EL CLÚSTER " + bag.getClusterLabel());
                clusterID++;
                numGeneratedClusters++;
            }
        }

        for (Instance instance : database.getInstances()) {
            DataObject dataObject = database.getDataObject(instance.stringValue(0));
            clusterAssignments.add(dataObject.getClusterLabel());
        }

        long finishTime = System.currentTimeMillis();
        elapsedTime = (double) (finishTime - startTime) / 1000.0D;
    }

    private boolean expandCluster(DataObject dataObject) {
        List<DataObject> nEps = database.epsilonRangeQuery(epsilon, dataObject);
        //System.out.println("Elementos a distancia epsilon: " + nEps.size());
        if (nEps.size() < minPoints) {
            dataObject.setClusterLabel(DataObject.NOISE);
            numNoises++;
            return false;
        } else {
            for (int i = 0; i < nEps.size(); ++i) {
                DataObject neighbor = nEps.get(i);
                if (neighbor.getKey().equals(dataObject.getKey())) {
                    nEps.remove(i);
                    --i;
                }
                neighbor.setClusterLabel(this.clusterID);
            }
            for (int i = 0; i < nEps.size(); ++i) {
                DataObject neighbor = nEps.get(i);
                List<DataObject> nEpsOfNeighbor = database.epsilonRangeQuery(epsilon, neighbor);
                //System.out.println("A SU VEZ, " + neighbor.getKey() + " TIENE DE VECINOS " + nEpsOfNeighbor.size());
                if (nEpsOfNeighbor.size() >= minPoints) {
                    for (DataObject p : nEpsOfNeighbor) {
                        if (p.getClusterLabel() == DataObject.UNCLASSIFIED || p.getClusterLabel() == DataObject.NOISE) {
                            if (p.getClusterLabel() == DataObject.UNCLASSIFIED) {
                                nEps.add(p);
                            }
                            p.setClusterLabel(this.clusterID);
                        }
                    }
                }
                nEps.remove(i);
                --i;
            }
            return true;
        }
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        return database.getDataObject(instance.stringValue(0)).getClusterLabel();
    }

    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("Clustered DataObjects: ").append(database.size()).append("\n");
        result.append("Number of instance attributes: ").append(database.getInstances().get(0).relationalValue(1).numAttributes()).append("\n");
        result.append("Epsilon: ").append(epsilon).append("; minPoints: ").append(minPoints).append("\n");
        result.append("Distance-type: ").append(this.getDistanceFunction()).append("\n");
        result.append("Number of generated clusters: ").append(numGeneratedClusters).append("\n");
        result.append("Number of noisily instances: ").append(numNoises).append("\n");
        DecimalFormat decimalFormat = new DecimalFormat(".##");
        result.append("Elapsed time: ").append(decimalFormat.format(elapsedTime)).append("\n");

        if (printClusterAssignments) {
            result.append("Cluster assigntments:\n");
            for (int i = 0; i < clusterAssignments.size(); ++i) {
                Instance instance = database.getInstances().instance(i);
                DataObject dataObject = database.getDataObject(instance.stringValue(0));
                result.append(dataObject.getKey())
                        .append("  -->  ").append(dataObject.getClusterLabel() == DataObject.NOISE ? "NOISE\n" : dataObject.getClusterLabel())
                        .append(" (").append(dataObject).append(")\n");
            }
        }
        return result.toString();
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> vector = new Vector<>();
        vector.addElement(new Option("\tepsilon (default = 0.9)", "E", 1, "-E <double>"));
        vector.addElement(new Option("\tminPoints (default = 6)", "M", 1, "-M <int>"));
        vector.add(new Option("\tDistance function to use.\n\t(default: HausdorffDistance)", "A", 1, "-A <classname and options>"));
        vector.add(new Option("\tOutput clusters assignments", "output-clusters", 0, "-output-clusters"));
        vector.add(new Option("\tNumber of threads to run in parallel", "num-slots", 1, "-num-slots <int>"));
        return vector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String minPoints = Utils.getOption('M', options);
        if (!minPoints.isEmpty()) {
            this.setMinPoints(Integer.parseInt(minPoints));
        }

        String epsilon = Utils.getOption('E', options);
        if (!epsilon.isEmpty()) {
            this.setEpsilon(Double.parseDouble(epsilon));
        }

        String distFunctionClass = Utils.getOption('A', options);
        distFunction = LoadByName.distanceFunction(distFunctionClass, options);

        printClusterAssignments = Utils.getFlag("output-clusters", options);

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        result.add("-M");
        result.add(Integer.toString(minPoints));
        result.add("-E");
        result.add(Double.toString(epsilon));
        result.add("-A");
        result.add((distFunction.getClass().getName() + " " + Utils.joinOptions(distFunction.getOptions())).trim());
        return result.toArray(new String[0]);
    }

    public void setMinPoints(int minPoints) {
        this.minPoints = minPoints;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public int numberOfClusters() {
        return this.numGeneratedClusters;
    }

    @Override
    public double getElapsedTime() {
        return elapsedTime;
    }

    @Override
    public List<Integer> getClusterAssignments() {
        return clusterAssignments;
    }

    public DistanceFunction getDistanceFunction() {
        return this.distFunction;
    }

    public void setDistanceFunction(DistanceFunction df, String[] options) throws Exception {
        this.distFunction = df;
        distFunction.setOptions(options);
    }

    public String distanceFunctionTipText() {
        return "The distance function to use for finding neighbours (default: weka.core.EuclideanDistance). ";
    }

    public String epsilonTipText() {
        return "radius of the epsilon-range-queries";
    }

    public String minPointsTipText() {
        return "minimun number of DataObjects required in an epsilon-range-query";
    }

    public String globalInfo() {
        return "Basic implementation of DBSCAN clustering algorithm that should *not* be used as a reference for runtime benchmarks: more sophisticated implementations exist! Clustering of new instances is not supported. More info:\n\n " + this.getTechnicalInformation().toString();
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Martin Ester and Hans-Peter Kriegel and Joerg Sander and Xiaowei Xu");
        result.setValue(Field.TITLE, "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise");
        result.setValue(Field.BOOKTITLE, "Second International Conference on Knowledge Discovery and Data Mining");
        result.setValue(Field.EDITOR, "Evangelos Simoudis and Jiawei Han and Usama M. Fayyad");
        result.setValue(Field.YEAR, "1996");
        result.setValue(Field.PAGES, "226-231");
        result.setValue(Field.PUBLISHER, "AAAI Press");
        return result;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8108 $");
    }
}
