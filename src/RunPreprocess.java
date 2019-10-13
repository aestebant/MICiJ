import filters.MIStandardization;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class RunPreprocess {
    public static void main(String[] args) {
        String[] datasets = {
                "/home/aurora/Escritorio/MILClustering/datasets/mutagenesis3_bonds_relational"
        };

        for (String d : datasets) {
            ArffLoader.ArffReader arff = null;
            try {
                BufferedReader reader = new BufferedReader(new FileReader(d + ".arff"));
                arff = new ArffLoader.ArffReader(reader);
            } catch (IOException e) {
                e.printStackTrace();
            }
            assert arff != null;
            Instances data = arff.getData();

            MIStandardization filter = new MIStandardization();
            data = filter.z5(data);

            /*int size = data.numInstances();
            for (int i = size-1; i >= 0; --i) {
                if (i != 0 && i != 3 && i != 14)
                    data.delete(i);
            }*/

            try {
                ConverterUtils.DataSink.write(d + "-z5.arff", data);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
