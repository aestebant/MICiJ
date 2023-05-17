package miclustering.algorithms.utils;

import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

public class NEpsElement implements RevisionHandler {
    final private DataObject dataObject;
    final private double distance;

    public NEpsElement(double distance, DataObject dataObject) {
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
