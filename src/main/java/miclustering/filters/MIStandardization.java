package miclustering.filters;

import miclustering.utils.ProcessDataset;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.core.Instance;
import weka.core.Instances;

public class MIStandardization {

    private double[] min;
    private double[] max;
    private double[] mean;
    private double[] std;

    private void preparation(Instances dataset) {
        int nAtt = dataset.get(0).relationalValue(1).numAttributes();
        int nInst = 0;
        for (Instance i : dataset) {
            nInst += i.relationalValue(1).numInstances();
        }

        double[][] arrayData = new double[nInst][nAtt];
        int k = 0;
        for (Instance i : dataset) {
            for (Instance j : i.relationalValue(1)) {
                arrayData[k] = ProcessDataset.instanceToArray(j);
                k++;
            }
        }
        RealMatrix data = new Array2DRowRealMatrix(arrayData);

        min = new double[nAtt];
        max = new double[nAtt];
        mean = new double[nAtt];
        std = new double[nAtt];
        for (int i = 0; i < nAtt; ++i) {
            Min computeMin = new Min();
            min[i] = computeMin.evaluate(data.getColumn(i));
            Max computeMax = new Max();
            max[i] = computeMax.evaluate(data.getColumn(i));
            Mean computeMean = new Mean();
            mean[i] = computeMean.evaluate(data.getColumn(i));
            StandardDeviation computeStd = new StandardDeviation();
            std[i] = computeStd.evaluate(data.getColumn(i));
        }
    }

    public Instances z1(Instances dataset) {
        preparation(dataset);
        for (Instance bag : dataset) {
            for (Instance instance : bag.relationalValue(1)) {
                for (int i = 0; i < instance.numAttributes(); ++i) {
                    double val = (instance.value(i) - mean[i]) / std[i];
                    if (Double.isNaN(val))
                        instance.setValue(i, 0);
                    else
                        instance.setValue(i, val);
                }
            }
        }
        return dataset;
    }

    public Instances z5(Instances dataset) {
        preparation(dataset);
        for (Instance bag : dataset) {
            for (Instance instance : bag.relationalValue(1)) {
                for (int i = 0; i < instance.numAttributes(); ++i) {
                    double val = (instance.value(i) - min[i]) / (max[i] - min[i]);
                    if (Double.isNaN(val))
                        instance.setValue(i, 0);
                    else
                        instance.setValue(i, val);
                }
            }
        }
        return dataset;
    }

}
