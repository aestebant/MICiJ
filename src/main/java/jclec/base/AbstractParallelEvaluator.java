package jclec.base;

import jclec.IIndividual;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;

/**
 * IEvaluator parallel abstract implementation.
 *
 * @author Alberto Cano
 * @author Sebastian Ventura
 */

public abstract class AbstractParallelEvaluator extends AbstractEvaluator {
    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    private static final long serialVersionUID = 3270366292953409655L;

    /**
     * Empty constructor.
     */

    public AbstractParallelEvaluator() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    /**
     * For all individuals in "inds" array: if individual fitness  is
     * null, then evaluate this individual.
     * <p>
     * This method is final. Is anyone wants implement this method in
     * another way should create a new IEvaluator class.
     * <p>
     * {@inheritDoc}
     */

    public void evaluate(List<IIndividual> inds) {
        long time = System.currentTimeMillis();

        ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Collection<Callable<Void>> collection = new ArrayList<>(inds.size());
        for (IIndividual ind : inds) {
            if (ind.getFitness() == null) {
                collection.add(new evaluationThread(ind));
                numberOfEvaluations++;
            }
        }
        try {
            List<Future<Void>> futures = threadExecutor.invokeAll(collection);
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        threadExecutor.shutdown();

        try {
            if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
                System.out.println("Threadpool timeout occurred");
        } catch (InterruptedException ie) {
            System.out.println("Threadpool prematurely terminated due to interruption in thread that created pool");
        }

        executionTime += System.currentTimeMillis() - time;
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Evaluation Thread
    /////////////////////////////////////////////////////////////////

    private class evaluationThread implements Callable<Void> {
        private IIndividual ind;

        public evaluationThread(IIndividual ind) {
            this.ind = ind;
        }

        @Override
        public Void call() throws Exception {
            evaluate(ind);
            return null;
        }
    }
}