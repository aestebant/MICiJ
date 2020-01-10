package miclustering.distances;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.VectorialMean;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.core.*;

import java.util.*;

public class HausdorffDistance extends MIDistance {

    public static final int MAXMIN = 0;
    public static final int MINMIN = 1;
    public static final int MEANMIN = 2;
    public static final int MEAN = 3;

    private int type = MAXMIN;

    protected double computeDistance(Instances i1, Instances i2) {
        double[][] distances = matrixDistance(i1, i2);

        int n1 = i1.numInstances();
        int n2 = i2.numInstances();

        double[] minByRows = null;
        if (type == MAXMIN || type == MINMIN || type == MEANMIN) {
            minByRows = new double[n1];
            for (int i = 0; i < n1; ++i) {
                minByRows[i] = new Min().evaluate(distances[i]);
            }
        }

        double result = 0D;

        if (type == MAXMIN) {
            result = new Max().evaluate(minByRows);
        } else if (type == MINMIN) {
            result = new Min().evaluate(minByRows);
        } else if (type == MEANMIN) {
            result = new Mean().evaluate(minByRows);
        } else if (type == MEAN) {
            VectorialMean mean = new VectorialMean(n2);
            for (int i = 0; i < n1; ++i) {
                mean.increment(distances[i]);
            }
            result = new Mean().evaluate(mean.getResult());
        }
        return result;
    }

    @Override
    public String toString() {
        String result = "ERROR";
        switch (type) {
            case MAXMIN:
                result = "Maximal Minimal Hausdorff Distance";
                break;
            case MINMIN:
                result = "Minimal Hausdorff Distance";
                break;
            case MEANMIN:
                result = "Average Minimal Hausdorff Distance";
                break;
            case MEAN:
                result = "Average Hausdorff Distance";
                break;
        }
        return result;
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> vector = new Vector<>();
        vector.add(new Option("Type of distance (default average).", "hausdorff-type", 1, "-hausdorff-type <average|minimal|maximal>"));
        return vector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String type = Utils.getOption("hausdorff-type", options);
        if (type.length() > 0) {
            switch (Integer.parseInt(type)) {
                case MAXMIN:
                    this.type = MAXMIN;
                    break;
                case MINMIN:
                    this.type = MINMIN;
                    break;
                case MEANMIN:
                    this.type = MEANMIN;
                    break;
                case MEAN:
                    this.type = MEAN;
                    break;
            }
        }
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }

    public int getType() {
        return type;
    }
}
