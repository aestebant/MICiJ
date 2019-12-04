package miclustering.algorithms.utils.opticgui;

import weka.core.DistanceFunction;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

import java.io.Serializable;
import java.util.ArrayList;

public class SERObject implements Serializable, RevisionHandler {
    private static final long serialVersionUID = -6022057864970639151L;
    private ArrayList resultVector;
    private int databaseSize;
    private int numberOfAttributes;
    private double epsilon;
    private int minPoints;
    private boolean opticsOutputs;
    private DistanceFunction distanceFunction;
    private int numberOfGeneratedClusters;
    private String elapsedTime;

    public SERObject(ArrayList resultVector, int databaseSize, int numberOfAttributes, double epsilon, int minPoints, boolean opticsOutputs, DistanceFunction distance, int numberOfGeneratedClusters, String elapsedTime) {
        this.resultVector = resultVector;
        this.databaseSize = databaseSize;
        this.numberOfAttributes = numberOfAttributes;
        this.epsilon = epsilon;
        this.minPoints = minPoints;
        this.opticsOutputs = opticsOutputs;
        this.distanceFunction = distance;
        this.numberOfGeneratedClusters = numberOfGeneratedClusters;
        this.elapsedTime = elapsedTime;
    }

    public ArrayList getResultVector() {
        return this.resultVector;
    }

    public int getDatabaseSize() {
        return this.databaseSize;
    }

    public int getNumberOfAttributes() {
        return this.numberOfAttributes;
    }

    public double getEpsilon() {
        return this.epsilon;
    }

    public int getMinPoints() {
        return this.minPoints;
    }

    public boolean isOpticsOutputs() {
        return this.opticsOutputs;
    }

    public DistanceFunction getDistanceFunction() {
        return this.distanceFunction;
    }

    public int getNumberOfGeneratedClusters() {
        return this.numberOfGeneratedClusters;
    }

    public String getElapsedTime() {
        return this.elapsedTime + " sec";
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10838 $");
    }
}

