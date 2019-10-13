package algorithms;

import algorithms.utils.DataObject;
import algorithms.utils.Database;
import distances.HausdorffDistance;
import weka.clusterers.AbstractClusterer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

public class MIDBSCAN extends AbstractClusterer implements OptionHandler, TechnicalInformationHandler {
    static final long serialVersionUID = -1666498248451219728L;
    private double epsilon = 0.9;
    private int minPoints = 6;
    private int numberOfGeneratedClusters;
    private DistanceFunction m_DistanceFunction = new HausdorffDistance();
    private Database database;
    private int clusterID;
    private int processed_InstanceID;
    private double elapsedTime;

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

    public void buildClusterer(Instances instances) throws Exception {
        this.getCapabilities().testWithFail(instances);
        long time_1 = System.currentTimeMillis();
        this.processed_InstanceID = 0;
        this.numberOfGeneratedClusters = 0;
        this.clusterID = 0;
        ReplaceMissingValues replaceMissingValues_Filter = new ReplaceMissingValues();
        replaceMissingValues_Filter.setInputFormat(instances);
        Instances filteredInstances = Filter.useFilter(instances, replaceMissingValues_Filter);
        this.database = new Database(this.getDistanceFunction(), filteredInstances);

        DataObject dataObject;
        for (int i = 0; i < this.database.getInstances().numInstances(); ++i) {
            dataObject = new DataObject(this.database.getInstances().instance(i), Integer.toString(i), this.database);
            this.database.insert(dataObject);
        }

        Iterator iterator = this.database.dataObjectIterator();

        while (iterator.hasNext()) {
            dataObject = (DataObject) iterator.next();
            //System.out.println("ANALIZANDO BOLSA " + dataObject.getKey());
            if (dataObject.getClusterLabel() == -1 && this.expandCluster(dataObject)) {
                //System.out.println("HA ENTRADO EN EL CLÚSTER " + dataObject.getClusterLabel());
                ++this.clusterID;
                ++this.numberOfGeneratedClusters;
            }
        }

        long time_2 = System.currentTimeMillis();
        this.elapsedTime = (double) (time_2 - time_1) / 1000.0D;
    }

    private boolean expandCluster(DataObject dataObject) {
        List<DataObject> seedList = this.database.epsilonRangeQuery(this.getEpsilon(), dataObject);
        //System.out.println("Elementos a distancia epsilon: " + seedList.size());
        if (seedList.size() < this.getMinPoints()) {
            dataObject.setClusterLabel(-2147483648);
            return false;
        } else {
            for (int j = 0; j < seedList.size(); ++j) {
                DataObject seedListDataObject = seedList.get(j);
                if (seedListDataObject.getKey().equals(dataObject.getKey())) {
                    seedList.remove(j);
                    --j;
                }
                seedListDataObject.setClusterLabel(this.clusterID);
            }

            for (int j = 0; j < seedList.size(); ++j) {
                DataObject seedListDataObject = seedList.get(j);
                List<DataObject> seedListDataObject_Neighbourhood = this.database.epsilonRangeQuery(this.getEpsilon(), seedListDataObject);
                //System.out.println("A SU VEZ, " + seedListDataObject.getKey() + " TIENE DE VECINOS " + seedListDataObject_Neighbourhood.size());
                if (seedListDataObject_Neighbourhood.size() >= this.getMinPoints()) {
                    for (DataObject p : seedListDataObject_Neighbourhood) {
                        if (p.getClusterLabel() == -1 || p.getClusterLabel() == -2147483648) {
                            if (p.getClusterLabel() == -1) {
                                seedList.add(p);
                            }
                            p.setClusterLabel(this.clusterID);
                        }
                    }
                }
                seedList.remove(j);
                --j;
            }
            return true;
        }
    }

    public int clusterInstance(Instance instance) throws Exception {
        if (this.processed_InstanceID >= this.database.size()) {
            this.processed_InstanceID = 0;
        }

        int cnum = this.database.getDataObject(Integer.toString(this.processed_InstanceID++)).getClusterLabel();
        if (cnum == -2147483648) {
            throw new Exception();
        } else {
            return cnum;
        }
    }

    public int numberOfClusters() {
        return this.numberOfGeneratedClusters;
    }

    public Enumeration<Option> listOptions() {
        Vector<Option> vector = new Vector<>();
        vector.addElement(new Option("\tepsilon (default = 0.9)", "E", 1, "-E <double>"));
        vector.addElement(new Option("\tminPoints (default = 6)", "M", 1, "-M <int>"));
        vector.add(new Option("\tDistance function to use.\n\t(default: weka.core.EuclideanDistance)", "A", 1, "-A <classname and options>"));
        return vector.elements();
    }

    public void setOptions(String[] options) throws Exception {
        String optionString = Utils.getOption('E', options);
        if (optionString.length() != 0) {
            this.setEpsilon(Double.parseDouble(optionString));
        }

        optionString = Utils.getOption('M', options);
        if (optionString.length() != 0) {
            this.setMinPoints(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption('A', options);
        if (optionString.length() != 0) {
            String[] distSpec = Utils.splitOptions(optionString);
            if (distSpec.length == 0) {
                throw new Exception("Invalid DistanceFunction specification string.");
            }

            String className = distSpec[0];
            distSpec[0] = "";
            this.setDistanceFunction((DistanceFunction) Utils.forName(DistanceFunction.class, className, distSpec));
        } else {
            this.setDistanceFunction(new HausdorffDistance());
        }

    }

    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        result.add("-E");
        result.add("" + this.getEpsilon());
        result.add("-M");
        result.add("" + this.getMinPoints());
        result.add("-A");
        result.add((this.m_DistanceFunction.getClass().getName() + " " + Utils.joinOptions(this.m_DistanceFunction.getOptions())).trim());
        return result.toArray(new String[0]);
    }

    public void setMinPoints(int minPoints) {
        this.minPoints = minPoints;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public double getEpsilon() {
        return this.epsilon;
    }

    public int getMinPoints() {
        return this.minPoints;
    }

    public String distanceFunctionTipText() {
        return "The distance function to use for finding neighbours (default: weka.core.EuclideanDistance). ";
    }

    public DistanceFunction getDistanceFunction() {
        return this.m_DistanceFunction;
    }

    public void setDistanceFunction(DistanceFunction df) {
        this.m_DistanceFunction = df;
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

    public String toString() {
        StringBuilder stringBuffer = new StringBuilder();
        stringBuffer.append("DBSCAN clustering results\n========================================================================================\n\n");
        stringBuffer.append("Clustered DataObjects: ").append(this.database.size()).append("\n");
        stringBuffer.append("Number of instance attributes: ").append(this.database.getInstances().get(0).relationalValue(1).numAttributes()).append("\n");
        stringBuffer.append("Epsilon: ").append(this.getEpsilon()).append("; minPoints: ").append(this.getMinPoints()).append("\n");
        stringBuffer.append("Distance-type: ").append(this.getDistanceFunction()).append("\n");
        stringBuffer.append("Number of generated clusters: ").append(this.numberOfGeneratedClusters).append("\n");
        DecimalFormat decimalFormat = new DecimalFormat(".##");
        stringBuffer.append("Elapsed time: ").append(decimalFormat.format(this.elapsedTime)).append("\n\n");

        for (int i = 0; i < this.database.size(); ++i) {
            DataObject dataObject = this.database.getDataObject(Integer.toString(i));
            stringBuffer.append("(").append(Utils.doubleToString(Double.parseDouble(dataObject.getKey()), Integer.toString(this.database.size()).length(), 0)).append(".) ").append(Utils.padRight(dataObject.toString(), 69)).append("  -->  ").append(dataObject.getClusterLabel() == -2147483648 ? "NOISE\n" : dataObject.getClusterLabel() + "\n");
        }

        return stringBuffer.toString() + "\n";
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8108 $");
    }
}
