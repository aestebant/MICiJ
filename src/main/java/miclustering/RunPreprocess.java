package miclustering;

import miclustering.filters.MIStandardization;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class RunPreprocess {
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
//                "ImageElephant",
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
                "DirectionWesteast",
//                "Thioredoxin",
//                "UCSBBreastCancer",
//                "Web1",
//                "Web2",
//                "Web3",
//                "Graz02bikes",
//                "Graz02car",
//                "Graz02people",
//                "standardMI_Maron",
                "musk1",
                "musk2",
        };

        MIStandardization filter = new MIStandardization();
        for (String d : datasets) {
            for (int i = 1; i < 2; ++i) {
                Instances data = ProcessDataset.readArff("/home/aurora/Escritorio/datasets/" + d + ".arff");
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
