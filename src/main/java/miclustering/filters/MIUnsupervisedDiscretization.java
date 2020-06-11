package miclustering.filters;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.MultiInstanceToPropositional;
import weka.filters.unsupervised.attribute.PropositionalToMultiInstance;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class MIUnsupervisedDiscretization extends Discretize {

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
        try {
            setOptions(Utils.splitOptions("-O -B 10 -M -1.0 -R first-last -precision 6 -unset-class-temporarily -Y"));
            setInputFormat(expanded);
            expanded = Filter.useFilter(expanded, this);
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
        assert result != null;
        result.setRelationName(dataset.relationName());
        return result;
    }

    /**
     * Set the output format. Takes the currently defined cutpoints and
     * m_InputFormat and calls setOutputFormat(Instances) appropriately.
     */
    @Override
    protected void setOutputFormat() {

        if (m_CutPoints == null) {
            setOutputFormat(null);
            return;
        }
        ArrayList<Attribute> attributes = new ArrayList<>(getInputFormat()
                .numAttributes());
        int classIndex = getInputFormat().classIndex();
        for (int i = 0, m = getInputFormat().numAttributes(); i < m; ++i) {
            if ((m_DiscretizeCols.isInRange(i))
                    && (getInputFormat().attribute(i).isNumeric())
                    && (getInputFormat().classIndex() != i)) {

                Set<String> cutPointsCheck = new HashSet<>();
                double[] cutPoints = m_CutPoints[i];
                if (!m_MakeBinary) {
                    ArrayList<String> attribValues;
                    if (cutPoints == null) {
                        attribValues = new ArrayList<>(1);
                        attribValues.add("All");
                    } else {
                        attribValues = new ArrayList<>(cutPoints.length + 1);
                        if (m_UseBinNumbers) {
                            for (int j = 0, n = cutPoints.length; j <= n; ++j) {
                                attribValues.add("B" + (j + 1));
                            }
                        } else {
                            for (int j = 0, n = cutPoints.length; j <= n; ++j) {
                                String newBinRangeString = binRangeString(cutPoints, j, getBinRangePrecision());
                                if (!cutPointsCheck.add(newBinRangeString)) {
                                    throw new IllegalArgumentException(
                                            "A duplicate bin range was detected. Try increasing the bin range precision.");
                                }
                                attribValues.add(newBinRangeString);
                            }
                        }
                    }
                    Attribute newAtt = new Attribute(getInputFormat().attribute(i).name(), attribValues);
                    newAtt.setWeight(getInputFormat().attribute(i).weight());
                    attributes.add(newAtt);
                } else {
                    if (cutPoints == null) {
                        ArrayList<String> attribValues = new ArrayList<>(1);
                        attribValues.add("All");
                        Attribute newAtt = new Attribute(getInputFormat().attribute(i).name(), attribValues);
                        newAtt.setWeight(getInputFormat().attribute(i).weight());
                        attributes.add(newAtt);
                    } else {
                        if (i < getInputFormat().classIndex()) {
                            classIndex += cutPoints.length - 1;
                        }
                        for (int j = 0, n = cutPoints.length; j < n; ++j) {
                            ArrayList<String> attribValues = new ArrayList<>(2);
                            if (m_UseBinNumbers) {
                                attribValues.add("B1");
                                attribValues.add("B2");
                            } else {
                                double[] binaryCutPoint = { cutPoints[j] };
                                String newBinRangeString1 = binRangeString(binaryCutPoint, 0, m_BinRangePrecision);
                                String newBinRangeString2 = binRangeString(binaryCutPoint, 1, m_BinRangePrecision);
                                if (newBinRangeString1.equals(newBinRangeString2)) {
                                    throw new IllegalArgumentException(
                                            "A duplicate bin range was detected. Try increasing the bin range precision.");
                                }
                                attribValues.add(newBinRangeString1);
                                attribValues.add(newBinRangeString2);
                            }
                            Attribute newAtt = new Attribute(getInputFormat().attribute(i)
                                    .name() + "_" + (j + 1), attribValues);
                            if (getSpreadAttributeWeight()) {
                                newAtt.setWeight(getInputFormat().attribute(i).weight() / cutPoints.length);
                            } else {
                                newAtt.setWeight(getInputFormat().attribute(i).weight());
                            }
                            attributes.add(newAtt);
                        }
                    }
                }
            } else {
                attributes.add((Attribute) getInputFormat().attribute(i).copy());
            }
        }
        Instances outputFormat = new Instances(getInputFormat().relationName(),
                attributes, 0);
        outputFormat.setClassIndex(classIndex);
        setOutputFormat(outputFormat);
    }

    /**
     * Get a bin range string for a specified bin of some attribute's cut points.
     *
     * @param cutPoints The attribute's cut points; never null.
     * @param j The bin number (zero based); never out of range.
     * @param precision the precision for the range values
     *
     * @return The bin range string.
     */
    private static String binRangeString(double[] cutPoints, int j, int precision) {
        assert cutPoints != null;

        int n = cutPoints.length;
        assert 0 <= j && j <= n;

        if (j == 0)
            return "(-inf:" + Utils.doubleToString(cutPoints[0], precision) + "]";
        if (j == n)
            return "(" + Utils.doubleToString(cutPoints[n-1], precision) + ":inf)";
        return "(" + Utils.doubleToString(cutPoints[j-1], precision) + ":" + Utils.doubleToString(cutPoints[j], precision) + "]";
    }
}
