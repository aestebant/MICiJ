import filters.MIStandardization;
import utils.ProcessDataset;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class RunPreprocess {
    public static void main(String[] args) {
        String[] datasets = {
                "/home/aurora/Escritorio/MILClustering/datasets/mutagenesis3_bonds_relational"
        };

        for (String d : datasets) {
            Instances data = ProcessDataset.readArff(d + ".arff");

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
