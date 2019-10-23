package utils;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class ProcessDataset {
    public static Instances readArff(String path) {
        ArffLoader.ArffReader arff = null;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            arff = new ArffLoader.ArffReader(reader);
        } catch (IOException e) {
            e.printStackTrace();
        }
        assert arff != null;
        Instances data = arff.getData();
        return data;
    }
}
