package miclustering;

import miclustering.filters.MIStandardization;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class RunPreprocess {
    public static void main(String[] args) {
        String[] datasets = {
                "component_relational",
                "eastwest_relational",
                "elephant_relational",
                "fox_relational",
                "function_relational",
                "musk1_relational",
                "musk2_relational",
                "mutagenesis3_atoms_relational",
                "mutagenesis3_bonds_relational",
                "mutagenesis3_chains_relational",
                "process_relational",
                "suramin_relational",
                "tiger_relational",
                "trx_relational",
                "westeast_relational"
        };

        MIStandardization filter = new MIStandardization();
        for (String d : datasets) {
            Instances data = ProcessDataset.readArff("datasets/" + d + ".arff");

            Instances dataZ4 = filter.z4(data);
            Instances dataZ5 = filter.z5(data);

            try {
                ConverterUtils.DataSink.write("datasets/" + d + "-z4.arff", dataZ4);
                ConverterUtils.DataSink.write("datasets/" + d + "-z5.arff", dataZ5);
            } catch (Exception e) {
                e.printStackTrace();
            }

            System.out.println("Finished dataset " + d);
        }
    }
}
