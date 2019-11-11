package utils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

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
        return arff.getData();
    }

    public static Instances extractInstances(Instance bag) {
        boolean isBag = false;
        try {
            isBag = bag.attribute(1).isRelationValued();
        } catch (Exception ignored) {
            //Capturar excepción Instance doesn't have access to a dataset e ignorarla.
        }

        if (isBag)
            return new Instances(bag.relationalValue(1));
        else {
            ArrayList<Attribute> attributes = new ArrayList<>(bag.numAttributes());
            for (int i = 0; i < bag.numAttributes(); ++i) {
                attributes.add(new Attribute("att" + i));
            }
            Instances instances = new Instances("aux", attributes, 1);
            instances.add(bag);
            return instances;
        }
    }
}
