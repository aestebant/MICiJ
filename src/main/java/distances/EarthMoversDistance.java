package distances;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.linear.*;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class EarthMoversDistance extends MIDistance {

    protected double computeDistance(Instances i1, Instances i2) {
        int n1 = i1.numInstances();
        int n2 = i2.numInstances();

        double[][] distances = matrixDistance(i1, i2);
        double[] flow = optimizeFlow(distances,n1, n2);

        double result = 0D;
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                result += flow[i*n2 + j] * distances[i][j];
            }
        }

        return result;
    }

    private double[] optimizeFlow(double[][] distances, int n1, int n2) {
        RealVector d = new ArrayRealVector(n1*n2);
        for (int i = 0; i < n1; ++i)
            d.append(new ArrayRealVector(distances[i]));
        LinearObjectiveFunction objective = new LinearObjectiveFunction(d, 0);

        List<LinearConstraint> constraints = new ArrayList<>();
        RealVector coefs = new ArrayRealVector(n1*n2, 1);
        constraints.add(new LinearConstraint(coefs, Relationship.EQ, 1));
        for (int i = 0; i < n1*n2; i += n2) {
            RealVector ones = new ArrayRealVector(n2, 1);
            RealVector prev = new ArrayRealVector(i, 0);
            RealVector post = new ArrayRealVector(n1*n2 - i - n2, 0);
            RealVector coefsRow = prev.append(ones).append(post);
            constraints.add(new LinearConstraint(coefsRow, Relationship.LEQ, 1D/n1));
        }
        for (int i = 0; i < n2; ++i) {
            RealVector coefsCol = new ArrayRealVector(n1*n2, 0);
            for (int j = i; j < n1*n2; j += n2)
                coefsCol.setEntry(j, 1);
            constraints.add(new LinearConstraint(coefsCol, Relationship.LEQ, 1D/n2));
        }
        LinearConstraintSet constraintSet = new LinearConstraintSet(constraints);

        NonNegativeConstraint positivity = new NonNegativeConstraint(true);

        SimplexSolver solver = new SimplexSolver();
        PointValuePair result = solver.optimize(objective, constraintSet, positivity);
        return result.getPoint();
    }

    @Override
    public String toString() {
        return "Earth Movers Distance";
    }
}
