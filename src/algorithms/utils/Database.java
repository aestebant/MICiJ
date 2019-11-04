package algorithms.utils;

import weka.clusterers.forOPTICSAndDBScan.Utils.PriorityQueue;
import weka.clusterers.forOPTICSAndDBScan.Utils.PriorityQueueElement;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

import java.io.Serializable;
import java.util.*;

public class Database implements Serializable, RevisionHandler {
    private TreeMap<String, DataObject> treeMap;
    private Instances instances;
    private DistanceFunction m_DistanceFunction;

    public Database(DistanceFunction distFunc, Instances instances) {
        this.instances = instances;
        this.treeMap = new TreeMap<>();
        this.m_DistanceFunction = distFunc;
        this.m_DistanceFunction.setInstances(instances);
    }

    public DataObject getDataObject(String key) {
        return this.treeMap.get(key);
    }

    public List<DataObject> epsilonRangeQuery(double epsilon, DataObject queryDataObject) {
        List<DataObject> epsilonRange_List = new ArrayList<>();
        for (Iterator i = dataObjectIterator(); i.hasNext(); ) {
            DataObject dataObject = (DataObject) i.next();
            double distance = this.m_DistanceFunction.distance(queryDataObject.getInstance(), dataObject.getInstance());
            if (distance < epsilon) {
                epsilonRange_List.add(dataObject);
            }
        }
        return epsilonRange_List;
    }

    private List k_nextNeighbourQuery(int k, double epsilon, DataObject dataObject) {
        List<EpsilonRange_ListElement> epsilonRange = new ArrayList<>();
        PriorityQueue priorityQueue = new PriorityQueue();
        for(Iterator i = this.dataObjectIterator(); i.hasNext(); ) {
            DataObject next_dataObject = (DataObject)i.next();
            double dist = this.m_DistanceFunction.distance(dataObject.getInstance(), next_dataObject.getInstance());
            if (dist <= epsilon) {
                epsilonRange.add(new EpsilonRange_ListElement(dist, next_dataObject));
            }

            if (priorityQueue.size() < k) {
                priorityQueue.add(dist, next_dataObject);
            } else if (dist < priorityQueue.getPriority(0)) {
                priorityQueue.next();
                priorityQueue.add(dist, next_dataObject);
            }
        }

        List<PriorityQueueElement> nextNeighbours = new ArrayList<>();
        while(priorityQueue.hasNext()) {
            nextNeighbours.add(0, priorityQueue.next());
        }

        List result = new ArrayList<>();
        result.add(nextNeighbours);
        result.add(epsilonRange);
        return result;
    }

    public List coreDistance(int minPoints, double epsilon, DataObject dataObject) {
        List list = this.k_nextNeighbourQuery(minPoints, epsilon, dataObject);
        if (((List)list.get(1)).size() < minPoints) {
            list.add(DataObject.UNDEFINED);
            return list;
        } else {
            List nextNeighbours_List = (List)list.get(0);
            PriorityQueueElement priorityQueueElement = (PriorityQueueElement) nextNeighbours_List
                    .get(nextNeighbours_List.size() - 1);
            if (priorityQueueElement.getPriority() <= epsilon) {
                list.add(priorityQueueElement.getPriority());
                return list;
            } else {
                list.add(DataObject.UNDEFINED);
                return list;
            }
        }
    }

    public int size() {
        return this.treeMap.size();
    }

    public Iterator keyIterator() {
        return this.treeMap.keySet().iterator();
    }

    public Iterator dataObjectIterator() {
        return this.treeMap.values().iterator();
    }

    public boolean contains(DataObject dataObject_Query) {
        Iterator iterator = this.dataObjectIterator();

        DataObject dataObject;
        do {
            if (!iterator.hasNext()) {
                return false;
            }

            dataObject = (DataObject)iterator.next();
        } while(!dataObject.equals(dataObject_Query));

        return true;
    }

    public void insert(DataObject dataObject) {
        this.treeMap.put(dataObject.getKey(), dataObject);
    }

    public Instances getInstances() {
        return this.instances;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10838 $");
    }
}
