package algorithms.utils;

import weka.core.Instance;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

import java.io.Serializable;

public class DataObject implements Serializable, RevisionHandler {
    public static final int UNCLASSIFIED = -1;
    public static final int NOISE = -2147483648;
    public static final double UNDEFINED = 2.147483647E9D;
    private static final long serialVersionUID = -4408119914898291075L;
    private Instance instance;
    private String key;
    private int clusterID;
    private boolean processed;
    private double c_dist;
    private double r_dist;
    private Database database;

    public DataObject(Instance originalInstance, String key, Database database) {
        this.database = database;
        this.key = key;
        this.instance = originalInstance;
        this.clusterID = -1;
        this.processed = false;
        this.c_dist = 2.147483647E9D;
        this.r_dist = 2.147483647E9D;
    }

    public Instance getInstance() {
        return this.instance;
    }

    public String getKey() {
        return this.key;
    }

    public void setKey(String key) {
        this.key = key;
    }

    public void setClusterLabel(int clusterID) {
        this.clusterID = clusterID;
    }

    public int getClusterLabel() {
        return this.clusterID;
    }

    public void setProcessed(boolean processed) {
        this.processed = processed;
    }

    public boolean isProcessed() {
        return this.processed;
    }

    public void setCoreDistance(double c_dist) {
        this.c_dist = c_dist;
    }

    public double getCoreDistance() {
        return this.c_dist;
    }

    public void setReachabilityDistance(double r_dist) {
        this.r_dist = r_dist;
    }

    public double getReachabilityDistance() {
        return this.r_dist;
    }

    public String toString() {
        return this.instance.toString();
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10838 $");
    }
}
