package miclustering;

import miclustering.filters.DatasetUnion;
import miclustering.utils.ProcessDataset;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class RunUnion {
    public static void main(String[] args) {
        List<Instances> datasets = new ArrayList<>(3);

        datasets.add(ProcessDataset.readArff("datasets/fox_relational.arff"));
        datasets.add(ProcessDataset.readArff("datasets/elephant_relational.arff"));
        datasets.add(ProcessDataset.readArff("datasets/tiger_relational.arff"));

        String newRelationName = "animals_relational";

        Instances result = DatasetUnion.union(datasets, newRelationName);
    }
}
