package miclustering.algorithms.utils;

import weka.clusterers.forOPTICSAndDBScan.Utils.PriorityQueue;
import weka.clusterers.forOPTICSAndDBScan.Utils.PriorityQueueElement;
import weka.core.*;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.*;

public class Database implements Serializable, RevisionHandler {
    private TreeMap<String, DataObject> treeMap;
    private Instances instances;
    private final DistanceFunction df;

    public Database(DistanceFunction distFunc, Instances instances) {
        this.instances = instances;
        this.treeMap = new TreeMap<>();
        df = distFunc;
        df.setInstances(instances);
    }

    public DataObject getDataObject(String key) {
        return this.treeMap.get(key);
    }

    public List<DataObject> epsilonRangeQuery(double epsilon, DataObject queryDataObject) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Collection<Callable<DataObject>> collection = new ArrayList<>();

        List<DataObject> nEps = new ArrayList<>();
        for (Iterator i = dataObjectIterator(); i.hasNext(); ) {
            DataObject dataObject = (DataObject) i.next();
            collection.add(new Wrapper(queryDataObject, dataObject, epsilon));
        }

        try {
            List<Future<DataObject>> futures = executor.invokeAll(collection);
            for (Future<DataObject> future : futures) {
                DataObject dataObject = future.get();
                if(dataObject != null) {
                    nEps.add(dataObject);
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();
        return nEps;
    }

     class Wrapper implements Callable<DataObject> {
        private DataObject bag1;
        private DataObject bag2;
        private double epsilon;

        private Wrapper(DataObject bag1, DataObject bag2, double epsilon) {
            this.bag1 = bag1;
            this.bag2 = bag2;
            this.epsilon = epsilon;
        }

        @Override
        public DataObject call() throws Exception {
            double distance = df.distance(bag1.getInstance(), bag2.getInstance());
            if (distance <= epsilon) {
                return bag2;
            } else {
                return null;
            }
        }
    }

    // SOLO SE USA EN OPTICS... NO MUY SEGURA DE CÓMO VA
    @SuppressWarnings("unchecked")
    private List kNextNeighbourQuery(int k, double epsilon, DataObject dataObject) {
        List<NEpsElement> epsilonRange = new ArrayList<>();
        PriorityQueue priorityQueue = new PriorityQueue();
        for(Iterator i = this.dataObjectIterator(); i.hasNext(); ) {
            DataObject next_dataObject = (DataObject)i.next();
            double distance = df.distance(dataObject.getInstance(), next_dataObject.getInstance());
            if (distance <= epsilon) {
                epsilonRange.add(new NEpsElement(distance, next_dataObject));
            }

            if (priorityQueue.size() < k) {
                priorityQueue.add(distance, next_dataObject);
            } else if (distance < priorityQueue.getPriority(0)) {
                priorityQueue.next();
                priorityQueue.add(distance, next_dataObject);
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

    // SÓLO SE USA EN OPTICS
    @SuppressWarnings("unchecked")
    public List coreDistance(int minPoints, double epsilon, DataObject dataObject) {
        List list = this.kNextNeighbourQuery(minPoints, epsilon, dataObject);
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
