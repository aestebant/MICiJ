package algorithms;

import algorithms.utils.DataObject;
import algorithms.utils.Database;
import algorithms.utils.EpsilonRange_ListElement;
import algorithms.utils.opticgui.OPTICS_Visualizer;
import algorithms.utils.opticgui.SERObject;
import distances.HausdorffDistance;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.forOPTICSAndDBScan.Utils.UpdateQueue;
import weka.clusterers.forOPTICSAndDBScan.Utils.UpdateQueueElement;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

public class MIOPTICS extends AbstractClusterer implements MyClusterer, OptionHandler, TechnicalInformationHandler {
    static final long serialVersionUID = 274552680222105221L;
    private double epsilon = 0.9D;
    private int minPoints = 6;
    private int numberOfGeneratedClusters;
    private DistanceFunction m_DistanceFunction;
    private Database database;
    private double elapsedTime;
    private boolean writeOPTICSresults = false;
    private ArrayList<DataObject> resultVector;
    private boolean showGUI = true;
    private File databaseOutput = new File(".");

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
        this.resultVector = new ArrayList<>();
        long time_1 = System.currentTimeMillis();
        this.numberOfGeneratedClusters = 0;
        ReplaceMissingValues replaceMissingValues_Filter = new ReplaceMissingValues();
        replaceMissingValues_Filter.setInputFormat(instances);
        Instances filteredInstances = Filter.useFilter(instances, replaceMissingValues_Filter);
        this.database = new Database(this.getDistanceFunction(), filteredInstances);

        for (int i = 0; i < this.database.getInstances().numInstances(); ++i) {
            DataObject dataObject = new DataObject(this.database.getInstances().instance(i), Integer.toString(i));
            this.database.insert(dataObject);
        }

        UpdateQueue seeds = new UpdateQueue();
        Iterator iterator = this.database.dataObjectIterator();

        while (iterator.hasNext()) {
            DataObject dataObject = (DataObject) iterator.next();
            if (!dataObject.isProcessed()) {
                this.expandClusterOrder(dataObject, seeds);
            }
        }

        long time_2 = System.currentTimeMillis();
        this.elapsedTime = (double) (time_2 - time_1) / 1000.0D;
        if (this.writeOPTICSresults) {
            GregorianCalendar gregorianCalendar = new GregorianCalendar();
            String timeStamp = gregorianCalendar.get(Calendar.DATE) + "-" + (gregorianCalendar.get(Calendar.MONTH) + 1) + "-" + gregorianCalendar.get(Calendar.YEAR) + "--" + gregorianCalendar.get(Calendar.HOUR_OF_DAY) + "-" + gregorianCalendar.get(Calendar.MINUTE) + "-" + gregorianCalendar.get(Calendar.SECOND);
            String fileName = "OPTICS_" + timeStamp + ".TXT";
            FileWriter fileWriter = new FileWriter(fileName);
            BufferedWriter bufferedOPTICSWriter = new BufferedWriter(fileWriter);

            for (DataObject dataObject : this.resultVector) {
                bufferedOPTICSWriter.write(this.format_dataObject(dataObject));
            }

            bufferedOPTICSWriter.flush();
            bufferedOPTICSWriter.close();
        }

        if (!this.databaseOutput.isDirectory()) {
            try {
                FileOutputStream fos = new FileOutputStream(this.databaseOutput);
                ObjectOutputStream oos = new ObjectOutputStream(fos);
                oos.writeObject(this.getSERObject());
                oos.flush();
                oos.close();
                fos.close();
            } catch (Exception var15) {
                System.err.println("Error writing generated database to file '" + this.getDatabaseOutput() + "': " + var15);
                var15.printStackTrace();
            }
        }

        if (this.showGUI) {
            new OPTICS_Visualizer(this.getSERObject(), "OPTICS Visualizer - Main Window");
        }

    }

    private void expandClusterOrder(DataObject dataObject, UpdateQueue seeds) {
        List list = this.database.coreDistance(this.getMinPoints(), this.getEpsilon(), dataObject);
        List epsilonRange_List = (List) list.get(1);
        dataObject.setReachabilityDistance(2.147483647E9D);
        dataObject.setCoreDistance((Double) list.get(2));
        dataObject.setProcessed(true);
        this.resultVector.add(dataObject);
        if (dataObject.getCoreDistance() != 2.147483647E9D) {
            this.update(seeds, epsilonRange_List, dataObject);

            while (seeds.hasNext()) {
                UpdateQueueElement updateQueueElement = seeds.next();
                DataObject currentDataObject = (DataObject) updateQueueElement.getObject();
                currentDataObject.setReachabilityDistance(updateQueueElement.getPriority());
                List list_1 = this.database.coreDistance(this.getMinPoints(), this.getEpsilon(), currentDataObject);
                List epsilonRange_List_1 = (List) list_1.get(1);
                currentDataObject.setCoreDistance((Double) list_1.get(2));
                currentDataObject.setProcessed(true);
                this.resultVector.add(currentDataObject);
                if (currentDataObject.getCoreDistance() != 2.147483647E9D) {
                    this.update(seeds, epsilonRange_List_1, currentDataObject);
                }
            }
        }

    }

    private String format_dataObject(DataObject dataObject) {
        return "(" + Utils.doubleToString(Double.parseDouble(dataObject.getKey()), Integer.toString(this.database.size()).length(), 0) + ".) " + Utils.padRight(dataObject.toString(), 40) + "  -->  c_dist: " + (dataObject.getCoreDistance() == 2.147483647E9D ? Utils.padRight("UNDEFINED", 12) : Utils.padRight(Utils.doubleToString(dataObject.getCoreDistance(), 2, 3), 12)) + " r_dist: " + (dataObject.getReachabilityDistance() == 2.147483647E9D ? Utils.padRight("UNDEFINED", 12) : Utils.doubleToString(dataObject.getReachabilityDistance(), 2, 3)) + "\n";
    }

    private void update(UpdateQueue seeds, List epsilonRange_list, DataObject centralObject) {
        double coreDistance = centralObject.getCoreDistance();
        double new_r_dist;

        for (Object o : epsilonRange_list) {
            EpsilonRange_ListElement listElement = (EpsilonRange_ListElement) o;
            DataObject neighbourhood_object = listElement.getDataObject();
            if (!neighbourhood_object.isProcessed()) {
                new_r_dist = Math.max(coreDistance, listElement.getDistance());
                seeds.add(new_r_dist, neighbourhood_object, neighbourhood_object.getKey());
            }
        }

    }

    public int clusterInstance(Instance instance) throws Exception {
        throw new Exception();
    }

    @Override
    public int numberOfClusters() throws Exception {
        return this.numberOfGeneratedClusters;
    }

    public Enumeration<Option> listOptions() {
        Vector<Option> vector = new Vector<>();
        vector.addElement(new Option("\tepsilon (default = 0.9)", "E", 1, "-E <double>"));
        vector.addElement(new Option("\tminPoints (default = 6)", "M", 1, "-M <int>"));
        vector.add(new Option("\tDistance function to use.\n\t(default: weka.core.EuclideanDistance)", "A", 1, "-A <classname and options>"));
        vector.addElement(new Option("\twrite results to OPTICS_#TimeStamp#.TXT - File", "F", 0, "-F"));
        vector.addElement(new Option("\tsuppress the display of the GUI after building the clusterer", "no-gui", 0, "-no-gui"));
        vector.addElement(new Option("\tThe file to save the generated database to. If a directory\n\tis provided, the database doesn't get saved.\n\tThe generated file can be viewed with the OPTICS Visualizer:\n\t  java " + OPTICS_Visualizer.class.getName() + " [file.ser]\n" + "\t(default: .)", "db-output", 1, "-db-output <file>"));
        return vector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String optionString = Utils.getOption('E', options);
        if (optionString.length() != 0) {
            this.setEpsilon(Double.parseDouble(optionString));
        } else {
            this.setEpsilon(0.9D);
        }

        optionString = Utils.getOption('M', options);
        if (optionString.length() != 0) {
            this.setMinPoints(Integer.parseInt(optionString));
        } else {
            this.setMinPoints(6);
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

        this.setWriteOPTICSresults(Utils.getFlag('F', options));
        this.setShowGUI(!Utils.getFlag("no-gui", options));
        optionString = Utils.getOption("db-output", options);
        if (optionString.length() != 0) {
            this.setDatabaseOutput(new File(optionString));
        } else {
            this.setDatabaseOutput(new File("."));
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
        if (this.getWriteOPTICSresults()) {
            result.add("-F");
        }

        if (!this.getShowGUI()) {
            result.add("-no-gui");
        }

        result.add("-db-output");
        result.add("" + this.getDatabaseOutput());
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

    public void setDistanceFunction(DistanceFunction df) throws Exception {
        this.m_DistanceFunction = df;
    }

    public boolean getWriteOPTICSresults() {
        return this.writeOPTICSresults;
    }

    public void setWriteOPTICSresults(boolean writeOPTICSresults) {
        this.writeOPTICSresults = writeOPTICSresults;
    }

    public boolean getShowGUI() {
        return this.showGUI;
    }

    public void setShowGUI(boolean value) {
        this.showGUI = value;
    }

    public File getDatabaseOutput() {
        return this.databaseOutput;
    }

    public void setDatabaseOutput(File value) {
        this.databaseOutput = value;
    }

    public ArrayList getResultVector() {
        return this.resultVector;
    }

    public String epsilonTipText() {
        return "radius of the epsilon-range-queries";
    }

    public String minPointsTipText() {
        return "minimun number of DataObjects required in an epsilon-range-query";
    }

    public String writeOPTICSresultsTipText() {
        return "if the -F option is set, the results are written to OPTICS_#TimeStamp#.TXT";
    }

    public String showGUITipText() {
        return "Defines whether the OPTICS Visualizer is displayed after the clusterer has been built or not.";
    }

    public String databaseOutputTipText() {
        return "The optional output file for the generated database object - can be viewed with the OPTICS Visualizer.\njava " + OPTICS_Visualizer.class.getName() + " [file.ser]";
    }

    public String globalInfo() {
        return "Basic implementation of OPTICS clustering algorithm that should *not* be used as a reference for runtime benchmarks: more sophisticated implementations exist! Clustering of new instances is not supported. More info:\n\n " + this.getTechnicalInformation().toString();
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Mihael Ankerst and Markus M. Breunig and Hans-Peter Kriegel and Joerg Sander");
        result.setValue(Field.TITLE, "OPTICS: Ordering Points To Identify the Clustering Structure");
        result.setValue(Field.BOOKTITLE, "ACM SIGMOD International Conference on Management of Data");
        result.setValue(Field.YEAR, "1999");
        result.setValue(Field.PAGES, "49-60");
        result.setValue(Field.PUBLISHER, "ACM Press");
        return result;
    }

    public SERObject getSERObject() {
        return new SERObject(this.resultVector, this.database.size(), this.database.getInstances().numAttributes(), this.getEpsilon(), this.getMinPoints(), this.writeOPTICSresults, this.getDistanceFunction(), this.numberOfGeneratedClusters, Utils.doubleToString(this.elapsedTime, 3, 3));
    }

    public String toString() {
        StringBuilder stringBuffer = new StringBuilder();
        stringBuffer.append("OPTICS clustering results\n============================================================================================\n\n");
        stringBuffer.append("Clustered DataObjects: ").append(this.database.size()).append("\n");
        stringBuffer.append("Number of attributes: ").append(this.database.getInstances().get(0).relationalValue(1).numAttributes()).append("\n");
        stringBuffer.append("Epsilon: ").append(this.getEpsilon()).append("; minPoints: ").append(this.getMinPoints()).append("\n");
        stringBuffer.append("Write results to file: ").append(this.writeOPTICSresults ? "yes" : "no").append("\n");
        stringBuffer.append("Distance-type: ").append(this.getDistanceFunction()).append("\n");
        stringBuffer.append("Number of generated clusters: ").append(this.numberOfGeneratedClusters).append("\n");
        DecimalFormat decimalFormat = new DecimalFormat(".##");
        stringBuffer.append("Elapsed time: ").append(decimalFormat.format(this.elapsedTime)).append("\n\n");

        for (DataObject dataObject : this.resultVector) {
            stringBuffer.append(this.format_dataObject(dataObject));
        }

        return stringBuffer.toString() + "\n";
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10838 $");
    }
}
