package miclustering.algorithms;

import weka.clusterers.AbstractClusterer;
import weka.core.Capabilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ClusterLikeClass extends AbstractClusterer implements MIClusterer {
	int numClusterers;
	private Instances instances;
	Map<Integer, Instance> centroids;
	private List<Integer> clusterAssignments;
	private double elapsedTime;

	@Override
	public void setOptions(String[] options) throws Exception {

	}

	@Override
	public DistanceFunction getDistanceFunction() {
		return null;
	}

	@Override
	public double getElapsedTime() {
		return elapsedTime;
	}

	@Override
	public void buildClusterer(Instances data) throws Exception {
		long startTime = System.currentTimeMillis();
		int numInstAttributes = data.get(0).relationalValue(1).numAttributes();
		instances = data;

		numClusterers = instances.numClasses();
		clusterAssignments = new ArrayList<>(instances.numInstances());
		for (Instance instance : instances) {
			clusterAssignments.add((int) instance.classValue());
		}


		Instances aux = new Instances(instances.get(0).relationalValue(1));
		for (int i = 1; i < instances.size(); ++i) {
			aux.addAll(instances.get(i).relationalValue(1));
		}

		long finishTime = System.currentTimeMillis();
		elapsedTime = (double) (finishTime - startTime) / 1000.0D;
	}

	@Override
	public int clusterInstance(Instance bag) {
		if (instances.contains(bag))
			return clusterAssignments.get(bag.dataset().indexOf(bag));
		else {
			return bag.classIndex();
		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return new double[0];
	}

	@Override
	public int numberOfClusters() throws Exception {
		return numClusterers;
	}

	@Override
	public List<Integer> getClusterAssignments() {
		return clusterAssignments;
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enable(Capabilities.Capability.NOMINAL_CLASS);
		result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
		result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
		return result;
	}
}
