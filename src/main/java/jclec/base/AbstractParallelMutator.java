package jclec.base;

import jclec.IIndividual;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;

/**
 * IMutator abstract implementation.
 *
 * @author Sebastian Ventura
 */

public abstract class AbstractParallelMutator extends AbstractMutator {

    /**
     * {@inheritDoc}
     */
    @Override
    public List<IIndividual> mutate(List<IIndividual> parents) {
        // Sets p list to actual parents
        parentsBuffer = parents;
        // Prepare recombination process
        prepareMutation();
        // Create a new list to put sons in it
        sonsBuffer = new ArrayList<>();

        ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Collection<Callable<IIndividual>> collection = new ArrayList<>(parentsBuffer.size());
        for (IIndividual ind : parents) {
            collection.add(new MutationThread(ind));
        }
        try {
            List<Future<IIndividual>> futures = threadExecutor.invokeAll(collection);
            for (Future<IIndividual> future : futures) {
                sonsBuffer.add(future.get());
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

        // Returns sons list
        return sonsBuffer;
    }

    protected abstract IIndividual mutateInd(IIndividual individual);

    private class MutationThread implements Callable<IIndividual> {
        private IIndividual individual;

        MutationThread(IIndividual individual) {
            this.individual = individual;
        }

        @Override
        public IIndividual call() throws Exception {
            return mutateInd(individual);
        }
    }
}
