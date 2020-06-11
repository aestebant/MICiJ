package miclustering;

import miclustering.filters.MIUnsupervisedDiscretization;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class RunDiscretization {
    public static void main(String[] args) {
        String[] datasets = {
//                "BirdsBrownCreeper",
//                "BirdsChestnut-backedChickadee",
//                "BirdsHammondsFlycatcher",
//                "BiocreativeComponent",
//                "BiocreativeFunction",
//                "BiocreativeProcess",
//                "CorelAfrican",
//                "CorelAntique",
//                "CorelBattleships",
//                "Harddrive1",
                "ImageElephant",
//                "ImageFox",
//                "ImageTiger",
//                "Messidor",
//                "mutagenesis3_atoms",
//                "mutagenesis3_bonds",
//                "mutagenesis3_chains",
//                "Newsgroups1",
//                "Newsgroups2",
//                "Newsgroups3",
//                "suramin",
//                "DirectionEastwest",
//                "DirectionWesteast",
//                "Thioredoxin",
//                "UCSBBreastCancer",
//                "Web1",
//                "Web2",
//                "Web3",
//                "Graz02bikes",
//                "Graz02car",
//                "Graz02people",
//                "standardMI_Maron",
//                "musk1",
//                "musk2",
        };

        MIUnsupervisedDiscretization filter = new MIUnsupervisedDiscretization();
        for (String d : datasets) {
            Instances data = ProcessDataset.readArff("datasets/" + d + "-z5.arff");
            data = filter.discretization(data);
            try {
                ConverterUtils.DataSink.write("datasets/" + d + "-z5-discr.arff", data);
            } catch (Exception e) {
                e.printStackTrace();
            }
            System.out.println("Finished dataset " + d);
        }
    }
}
