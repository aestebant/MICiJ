package miclustering.filters;

import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.MultiInstanceToPropositional;
import weka.filters.unsupervised.attribute.PropositionalToMultiInstance;

public class MIUnsupervisedDiscretization {

    public Instances discretization(Instances dataset) {
        dataset.setClassIndex(2);
        Instances expanded = null;
        MultiInstanceToPropositional expand = new MultiInstanceToPropositional();
        try {
            expand.setOptions(Utils.splitOptions("-A 1"));
            expand.setInputFormat(dataset);
            expanded = Filter.useFilter(dataset, expand);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Discretize discretize = new Discretize();
        try {
            discretize.setOptions(Utils.splitOptions("-O -B 10 -M -1.0 -R first-last -precision 6 -unset-class-temporarily"));
            discretize.setInputFormat(expanded);
            expanded = Filter.useFilter(expanded, discretize);
        } catch (Exception e) {
            e.printStackTrace();
        }
        PropositionalToMultiInstance contract = new PropositionalToMultiInstance();
        Instances result = null;
        try {
            contract.setOptions(Utils.splitOptions("-S 1 -B first -no-weights"));
            contract.setInputFormat(expanded);
            result = Filter.useFilter(expanded, contract);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }
}
