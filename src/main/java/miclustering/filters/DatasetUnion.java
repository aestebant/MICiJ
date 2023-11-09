package miclustering.filters;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddValues;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class DatasetUnion {
    public static Instances union(List<Instances> datasets, String unionRelationName) {
        int nAtt = datasets.get(0).get(0).relationalValue(1).numAttributes();

        int nBags = 0;
        StringBuilder idClasses = new StringBuilder();
        for (int i = 0; i < datasets.size(); ++i) {
            Instances d = datasets.get(i);
            if (nAtt != d.get(0).relationalValue(1).numAttributes()) {
                System.err.println("The number of instance attributes is not equal for all the datasets");
                System.exit(-1);
            }
            for (Instance inst : d) {
                if (inst.value(d.numAttributes()-1) == 1)
                    nBags++;
            }
            idClasses.append(i).append(",");
        }

        StringBuilder idValues = new StringBuilder();
        for (int i = 0; i < nBags; ++i) {
            idValues.append(i).append(",");
        }

        AddValues addIds = new AddValues();
        try {
            addIds.setOptions(Utils.splitOptions("-C first -L " + idValues.toString()));
            for (int i = 0; i < datasets.size(); ++i) {
                addIds.setInputFormat(datasets.get(i));
                datasets.set(i, Filter.useFilter(datasets.get(i), addIds));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        AddValues addClasses = new AddValues();
        try {
            addClasses.setOptions(Utils.splitOptions("-C last -L " + idClasses));
            for (int i = 0; i < datasets.size(); ++i) {
                addClasses.setInputFormat(datasets.get(i));
                datasets.set(i, Filter.useFilter(datasets.get(i), addClasses));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        Instances ref = datasets.get(0);
        Instances result = new Instances(ref.stringFreeStructure(), nBags);
        result.setRelationName(unionRelationName);

        ArffSaver saver = new ArffSaver();
        saver.setRetrieval(Saver.INCREMENTAL);
        saver.setStructure(result);
        try {
            saver.setFile(new File("datasets/" + unionRelationName + ".arff"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        int currId = 0;
        for (int i = 0; i < datasets.size(); ++i) {
            Instances d = datasets.get(i);
            for (int j = 0; j < d.numInstances(); ++j) {
                Instance inst = d.get(j);
                if (inst.value(d.numAttributes()-1) == 1) {
                    inst.setValue(0, currId);
                    inst.setValue(d.numAttributes()-1, i);
                    try {
                        saver.writeIncremental(inst);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    currId++;
                }
            }
        }

        /*RemoveWithValues removeWithValues = new RemoveWithValues();
        try {
            removeWithValues.setInputFormat(result);
            removeWithValues.setOptions(Utils.splitOptions("-C first -L " + (currId + 1) + "-last -H"));
            result = Filter.useFilter(result, removeWithValues);
        } catch (Exception e) {
            e.printStackTrace();
        }
        result.setRelationName(unionRelationName);*/

        try {
            saver.getWriter().flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return result;
    }
}
