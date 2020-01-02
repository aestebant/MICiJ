package miclustering.utils;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.core.DistanceFunction;
import weka.core.Utils;

import java.lang.reflect.InvocationTargetException;

public class LoadByName {
    public static Clusterer clusterer(String name) {
        Class<? extends AbstractClusterer> absCluster = null;
        try {
            absCluster = Class.forName(name).asSubclass(AbstractClusterer.class);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        assert absCluster != null;
        Clusterer clusterer = null;
        try {
            clusterer = absCluster.getDeclaredConstructor().newInstance();
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            e.printStackTrace();
        }
        return clusterer;
    }

    public static DistanceFunction distanceFunction(String name, String[] options) {
        String[] distFunctionClassSpec = new String[0];
        try {
            distFunctionClassSpec = Utils.splitOptions(name);
        } catch (Exception e) {
            e.printStackTrace();
        } if (distFunctionClassSpec.length == 0) {
            System.err.println("Invalid DistanceFunction specification string.");
            System.exit(-1);
        }
        String className = distFunctionClassSpec[0];
        distFunctionClassSpec[0] = "";
        DistanceFunction distanceFunction = null;
        try {
            distanceFunction = (DistanceFunction) Utils.forName(DistanceFunction.class, className, distFunctionClassSpec);
            distanceFunction.setOptions(options);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return distanceFunction;
    }
}
