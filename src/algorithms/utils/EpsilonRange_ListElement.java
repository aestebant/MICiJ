package algorithms.utils;

import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

public class EpsilonRange_ListElement implements RevisionHandler {
    private DataObject dataObject;
    private double distance;

    public EpsilonRange_ListElement(double distance, DataObject dataObject) {
        this.distance = distance;
        this.dataObject = dataObject;
    }

    public double getDistance() {
        return this.distance;
    }

    public DataObject getDataObject() {
        return this.dataObject;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8108 $");
    }
}
