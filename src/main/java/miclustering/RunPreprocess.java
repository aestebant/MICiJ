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
                "westeast_relational",
                "animals_relational"
        };

        MIStandardization filter = new MIStandardization();
        for (String d : datasets) {
            for (int i = 0; i < 2; ++i) {
                Instances data = ProcessDataset.readArff("datasets/" + d + ".arff");
                String ext = "";
                switch (i) {
                    case 0:
                        filter.z1(data);
                        ext = "-z1";
                        break;
                    case 1:
                        filter.z5(data);
                        ext = "-z5";
                        break;
                }
                try {
                    ConverterUtils.DataSink.write("datasets/" + d + ext + ".arff", data);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            System.out.println("Finished dataset " + d);
        }
    }
}
