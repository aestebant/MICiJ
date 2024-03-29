package miclustering.evaluators;

import miclustering.utils.DistancesMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import org.apache.commons.math3.util.FastMath;
import org.jgrapht.Graph;
import org.jgrapht.alg.interfaces.SpanningTreeAlgorithm;
import org.jgrapht.alg.spanning.KruskalMinimumSpanningTree;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleWeightedGraph;
import weka.core.DistanceFunction;
import weka.core.Instances;

import java.util.*;

public class DBCV {
    private final Instances dataset;
    private final double[][] distancesMatrix;
    private final int maxNumClusters;

    public DBCV(Instances dataset, DistanceFunction distanceFunction, int maxNumClusters, boolean parallelize) {
        this.dataset = dataset;
        this.maxNumClusters = maxNumClusters;
        DistancesMatrix dm = new DistancesMatrix();
        distancesMatrix = dm.compute(dataset, distanceFunction, parallelize);
    }

    public double computeIndex(List<Integer> clusterAssignments, int[] bagsPerCluster) {
        double[] coreDist = allPointsCoreDist(clusterAssignments, bagsPerCluster);
        Map<Integer, SpanningTreeAlgorithm.SpanningTree<WeightedEdge>> mst = mutualReachDistMST(clusterAssignments, coreDist);
        double[][] dspc = densitySeparation(mst, coreDist);
        double[] vc = new double[maxNumClusters];
        for (int i = 0; i < maxNumClusters; ++i) {
            Min min = new Min();
            double minDspc = min.evaluate(dspc[i]);
            double dsc = densitySparseness(mst.get(i));
            vc[i] = (minDspc - dsc) / FastMath.max(minDspc, dsc);
        }
        double[] weights = new double[maxNumClusters];
        for (int i = 0; i < maxNumClusters; ++i)
            weights[i] = (double) bagsPerCluster[i] / dataset.numInstances();
        Mean mean = new Mean();
        double result;
        try {
            result = mean.evaluate(vc, weights);
        } catch (Exception e) {
            result = 0;
        }
        return result;
    }

    private double[] allPointsCoreDist(List<Integer> clusterAssignments, int[] bagsPerCluster) {
        int d = dataset.get(0).relationalValue(1).numAttributes();
        double[] result = new double[clusterAssignments.size()];
        for (int i = 0; i < clusterAssignments.size(); ++i) {
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx < 0)
                break;
            for (int j = 0; j < clusterAssignments.size(); ++j) {
                if (i != j && clusterIdx == clusterAssignments.get(j)) {
                    result[i] += FastMath.pow(1D / distancesMatrix[i][j], d);
                }
            }
            if (result[i] == 0D)
                result[i] = Double.MAX_VALUE;
            else
                result[i] = FastMath.pow(result[i] / (bagsPerCluster[clusterIdx] - 1), -1D/d);
        }
        return result;
    }

    private double mutualReachDist(double[] coreDist, int bag1Idx, int bag2Idx) {
        Max max = new Max();
        return max.evaluate(new double[]{coreDist[bag1Idx], coreDist[bag2Idx], distancesMatrix[bag1Idx][bag2Idx]});
    }

    private Map<Integer, SpanningTreeAlgorithm.SpanningTree<WeightedEdge>> mutualReachDistMST(List<Integer> clusterAssignments, double[] coreDist) {
        Map<Integer, Graph<Integer, WeightedEdge>> g = new HashMap<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            g.put(i, new SimpleWeightedGraph<>(WeightedEdge.class));
        }
        for (int i = 0; i < clusterAssignments.size(); ++i) {
            int clusterIdx = clusterAssignments.get(i);
            if (clusterIdx < 0)
                break;
            g.get(clusterIdx).addVertex(i);
            for (int j = i + 1; j < clusterAssignments.size(); ++j) {
                if (clusterIdx == clusterAssignments.get(j)) {
                    if (!g.get(clusterIdx).containsVertex(j))
                        g.get(clusterIdx).addVertex(j);
                    WeightedEdge edge = g.get(clusterIdx).addEdge(i, j);
                    g.get(clusterIdx).setEdgeWeight(edge, mutualReachDist(coreDist, i, j));
                }
            }
        }
        Map<Integer, SpanningTreeAlgorithm.SpanningTree<WeightedEdge>> mst = new HashMap<>(maxNumClusters);
        for (int i = 0; i < maxNumClusters; ++i) {
            KruskalMinimumSpanningTree<Integer, WeightedEdge> kruskal = new KruskalMinimumSpanningTree<>(g.get(i));
            mst.put(i,kruskal.getSpanningTree());
        }
        return mst;
    }

    private double densitySparseness(SpanningTreeAlgorithm.SpanningTree<WeightedEdge> mst) {
        double maxWeight = Double.NEGATIVE_INFINITY;
        for (WeightedEdge edge : mst) {
            double currWeight = edge.getWeight();
            if (currWeight > maxWeight)
                maxWeight = currWeight;
        }
        return maxWeight;
    }

    private double[][] densitySeparation(Map<Integer, SpanningTreeAlgorithm.SpanningTree<WeightedEdge>> mst, double[]coreDist) {
        double[][] result = new double[maxNumClusters][maxNumClusters];
        for (int i = 0; i < maxNumClusters; ++i) {
            for (int j = i + 1; j < maxNumClusters; ++j) {
                double minReachDist = Double.POSITIVE_INFINITY;
                Iterator<WeightedEdge> itI = mst.get(i).getEdges().iterator();
                while (itI.hasNext()) {
                    WeightedEdge edgeI = itI.next();
                    Integer bagIIdx = edgeI.getSource();
                    Iterator<WeightedEdge> itJ = mst.get(j).getEdges().iterator();
                    while (itJ.hasNext()) {
                        WeightedEdge edgeJ = itJ.next();
                        Integer bagJIdx = edgeJ.getSource();
                        double reachDist = mutualReachDist(coreDist, bagIIdx, bagJIdx);
                        if (reachDist < minReachDist)
                            minReachDist = reachDist;

                        if (!itJ.hasNext()) {
                            Integer lastBag2Idx = edgeJ.getTarget();
                            double lastReachDist = mutualReachDist(coreDist, bagIIdx, lastBag2Idx);
                            if (lastReachDist < minReachDist)
                                minReachDist = lastReachDist;
                        }
                    }

                    if (!itI.hasNext()) {
                        Integer lastBagIIdx = edgeI.getTarget();
                        Iterator<WeightedEdge> itJAux = mst.get(j).getEdges().iterator();
                        while (itJAux.hasNext()) {
                            WeightedEdge edgeJ = itJAux.next();
                            Integer bagJIdx = edgeJ.getSource();
                            double reachDist = mutualReachDist(coreDist, lastBagIIdx, bagJIdx);
                            if (reachDist < minReachDist)
                                minReachDist = reachDist;

                            if (!itJAux.hasNext()) {
                                Integer lastBagJIdx = edgeJ.getTarget();
                                double lastReachDist = mutualReachDist(coreDist, lastBagIIdx, lastBagJIdx);
                                if (lastReachDist < minReachDist)
                                    minReachDist = lastReachDist;
                            }
                        }
                    }
                }
                result[i][j] = minReachDist;
                result[j][i] = minReachDist;
            }
            result[i][i] = Double.POSITIVE_INFINITY;
        }
        return result;
    }

    public static class WeightedEdge extends DefaultWeightedEdge {
        public Integer getSource() {
            return (Integer) super.getSource();
        }

        public Integer getTarget() {
            return (Integer) super.getTarget();
        }

        public double getWeight() {
            return super.getWeight();
        }
    }
}
