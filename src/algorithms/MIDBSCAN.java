package algorithms;

import algorithms.utils.DataObject;
import algorithms.utils.Database;
import distances.HausdorffDistance;
import weka.clusterers.AbstractClusterer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

public class MIDBSCAN extends AbstractClusterer implements MyClusterer, OptionHandler, TechnicalInformationHandler {
    private double epsilon = 0.9;
    private int minPoints = 6;
    private int numGeneratedClusters;
    private int numNoises;
    private DistanceFunction distFunction = new HausdorffDistance();
    private Database database;
    private int clusterID;
    private double elapsedTime;
    private boolean printClusterAssignments;

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
        numGeneratedClusters = 0;
        numNoises = 0;
        clusterID = 0;
        database = new Database(distFunction, instances);

        for (int i = 0; i < database.getInstances().numInstances(); ++i) {
            Instance instance = database.getInstances().instance(i);
            DataObject dataObject = new DataObject(instance, instance.stringValue(0));
            database.insert(dataObject);
        }

        for (Iterator i = database.dataObjectIterator(); i.hasNext(); ) {
            DataObject dataObject = (DataObject) i.next();
            //System.out.println("ANALIZANDO BOLSA " + dataObject.getKey());
            if (dataObject.getClusterLabel() == DataObject.UNCLASSIFIED && this.expandCluster(dataObject)) {
                //System.out.println("HA ENTRADO EN EL CLÚSTER " + dataObject.getClusterLabel());
                clusterID++;
                numGeneratedClusters++;
            }
        }

        long finishTime = System.currentTimeMillis();
        elapsedTime = (double) (finishTime - startTime) / 1000.0D;
    }

    private boolean expandCluster(DataObject dataObject) {
        List<DataObject> seedList = database.epsilonRangeQuery(epsilon, dataObject);
        //System.out.println("Elementos a distancia epsilon: " + seedList.size());
        if (seedList.size() < minPoints) {
            dataObject.setClusterLabel(DataObject.NOISE);
            numNoises++;
            return false;
        } else {
            for (int i = 0; i < seedList.size(); ++i) {
                DataObject seedListDataObject = seedList.get(i);
                if (seedListDataObject.getKey().equals(dataObject.getKey())) {
                    seedList.remove(i);
                    --i;
                }
                seedListDataObject.setClusterLabel(this.clusterID);
            }

            for (int i = 0; i < seedList.size(); ++i) {
                DataObject seedListDataObject = seedList.get(i);
                List<DataObject> seedListDataObject_Neighbourhood = database.epsilonRangeQuery(epsilon, seedListDataObject);
                //System.out.println("A SU VEZ, " + seedListDataObject.getKey() + " TIENE DE VECINOS " + seedListDataObject_Neighbourhood.size());
                if (seedListDataObject_Neighbourhood.size() >= minPoints) {
                    for (DataObject p : seedListDataObject_Neighbourhood) {
                        if (p.getClusterLabel() == DataObject.UNCLASSIFIED || p.getClusterLabel() == DataObject.NOISE) {
                            if (p.getClusterLabel() == DataObject.UNCLASSIFIED) {
                                seedList.add(p);
                            }
                            p.setClusterLabel(this.clusterID);
                        }
                    }
                }
                seedList.remove(i);
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
            for (int i = 0; i < database.getInstances().numInstances(); ++i) {
                Instance instance = database.getInstances().instance(i);
                DataObject dataObject = database.getDataObject(instance.stringValue(0));
                result.append(dataObject.getKey())
                        .append("  -->  ").append(dataObject.getClusterLabel() == DataObject.NOISE ? "NOISE\n" : dataObject.getClusterLabel())
                        .append(" (").append(dataObject.toString()).append(")\n");
            }
        }

        return result.toString() + "\n";
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> vector = new Vector<>();
        vector.addElement(new Option("\tepsilon (default = 0.9)", "E", 1, "-E <double>"));
        vector.addElement(new Option("\tminPoints (default = 6)", "M", 1, "-M <int>"));
        vector.add(new Option("\tDistance function to use.\n\t(default: weka.core.EuclideanDistance)", "A", 1, "-A <classname and options>"));
        vector.add(new Option("\tOutput clusters assignments", "output-clusters", 0, "-output-clusters"));
        return vector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String epsilon = Utils.getOption('E', options);
        if (epsilon.length() != 0) {
            this.setEpsilon(Double.parseDouble(epsilon));
        }

        String minPoints = Utils.getOption('M', options);
        if (minPoints.length() != 0) {
            this.setMinPoints(Integer.parseInt(minPoints));
        }

        String distance = Utils.getOption('A', options);
        if (distance.length() != 0) {
            String[] distSpec = Utils.splitOptions(distance);
            if (distSpec.length == 0) {
                throw new Exception("Invalid DistanceFunction specification string.");
            }
            String className = distSpec[0];
            distSpec[0] = "";
            this.setDistanceFunction((DistanceFunction) Utils.forName(DistanceFunction.class, className, distSpec));
        } else {
            this.setDistanceFunction(new HausdorffDistance());
        }

        printClusterAssignments = Utils.getFlag("output-clusters", options);
    }

    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        result.add("-E");
        result.add(Double.toString(epsilon));
        result.add("-M");
        result.add(Integer.toString(minPoints));
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

    public DistanceFunction getDistanceFunction() {
        return this.distFunction;
    }

    public void setDistanceFunction(DistanceFunction df) {
        this.distFunction = df;
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
