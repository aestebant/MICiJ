package utils;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;

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
}
