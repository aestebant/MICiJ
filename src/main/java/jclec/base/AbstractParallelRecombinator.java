package jclec.base;


import jclec.IIndividual;
import jclec.IRecombinator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;

/**
 * IRecombinator abstract implementation.
 *
 * @author Sebastian Ventura
 */

public abstract class AbstractParallelRecombinator  extends AbstractRecombinator implements IRecombinator {
    /**
     * {@inheritDoc}
     */

    @Override
    public List<IIndividual> recombine(List<IIndividual> parents) {
        // Sets p list to actual parents
        parentsBuffer = parents;
        // Prepare recombination process
        prepareRecombination();
        // Create a new list to put sons in it
        sonsBuffer = new ArrayList<>();

        ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Collection<Callable<List<IIndividual>>> collection = new ArrayList<>(parentsBuffer.size());
        for (parentsCounter = 0; parentsCounter <= parents.size() - ppl; parentsCounter += ppl) {
            List<IIndividual> individuals = new ArrayList<>(ppl);
            for (int i = parentsCounter; i < parentsCounter + ppl; ++i)
                individuals.add(parentsBuffer.get(i));
            collection.add(new RecombinationThread(individuals));
        }
        try {
            List<Future<List<IIndividual>>> futures = threadExecutor.invokeAll(collection);
            for (Future<List<IIndividual>> future : futures) {
                sonsBuffer.addAll(future.get());
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

    protected abstract List<IIndividual> recombineInd(List<IIndividual> individuals);

    private class RecombinationThread implements Callable<List<IIndividual>> {
        private final List<IIndividual> individuals;

        RecombinationThread(List<IIndividual> individuals) {
            this.individuals = individuals;
        }

        @Override
        public List<IIndividual> call() {
            return recombineInd(individuals);
        }
    }
}
