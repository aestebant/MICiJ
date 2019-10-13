package filters;

import weka.core.Instance;
import weka.core.Instances;

public class MIStandardization {

    private Instances dataset;

    public Instances z5(Instances dataset) {
        this.dataset = new Instances(dataset);
        int numAtt = dataset.get(0).relationalValue(1).numAttributes();
        for (int att = 0; att < numAtt; ++att) {
            double minValue = getMin(att);
            double maxValue = getMax(att);
            //System.out.println("Minimal: " + minValue + " Maximal: " + maxValue);
            for (Instance bag : this.dataset) {
                Instances instances = bag.relationalValue(1);
                for (Instance i : instances) {
                    if (minValue == maxValue)
                        i.setValue(att, 0D);
                    else {
                        double standardized = (i.value(att) - minValue) / (maxValue - minValue);
                        i.setValue(att, standardized);
                    }
                }
            }
        }
        return this.dataset;
    }

    private double getMin(int attIdx) {
        double min = dataset.get(0).relationalValue(1).get(0).value(attIdx);
        for (Instance bag : dataset) {
            Instances instances = bag.relationalValue(1);
            for (Instance i : instances) {
                if (i.value(attIdx) < min) {
                    min = i.value(attIdx);
                }
            }
        }
        return min;
    }

    private double getMax(int attIdx) {
        double max = dataset.get(0).relationalValue(1).get(0).value(attIdx);
        for (Instance bag : dataset) {
            Instances instances = bag.relationalValue(1);
            for (Instance i : instances) {
                if (i.value(attIdx) > max) {
                    max = i.value(attIdx);
                }
            }
        }
        return max;
    }
}
