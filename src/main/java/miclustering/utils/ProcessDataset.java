package miclustering.utils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

import java.io.*;
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

    // Util porque el constructor de copia por defecto de Weka para las instancias no copia los atributos relacionales
    public static Instance copyBag(Instance bag) {
        ByteArrayOutputStream aux = new ByteArrayOutputStream();
        ArffSaver saver = new ArffSaver();
        saver.setRetrieval(Saver.INCREMENTAL);
        Instance copy = null;

        try {
            saver.setDestination(aux);
            saver.setStructure(bag.dataset());
            saver.writeIncremental(bag);
            saver.getWriter().flush();

            Reader reader = new InputStreamReader(new ByteArrayInputStream(aux.toByteArray()));
            ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(reader);
            copy = arffReader.getData().get(0);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return copy;
    }

    public static double[] instanceToArray(Instance instance) {
        double[] array = new double[instance.numAttributes()];
        for (int i = 0; i < instance.numAttributes(); ++i) {
            array[i] = instance.value(i);
        }
        return array;
    }

    public static double[][] bagToMatrix(Instances bag) {
        double[][] matrix = new double[bag.numInstances()][bag.numAttributes()];
        for (int i = 0; i < bag.numInstances(); ++i) {
            matrix[i] = instanceToArray(bag.get(i));
        }
        return matrix;
    }
}
