package jclec.algorithm;

import jclec.AlgorithmEvent;
import jclec.IAlgorithm;
import jclec.IAlgorithmListener;
import jclec.IConfigure;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationRuntimeException;

import java.util.ArrayList;

/**
 * IAlgorithm abstract implementation.
 *
 * @author Sebastian Ventura
 */

@SuppressWarnings("serial")
public abstract class AbstractAlgorithm implements IAlgorithm, IConfigure {
    /////////////////////////////////////////////////////////////////
    // --------------------------------------------- Algorithm states
    /////////////////////////////////////////////////////////////////

    private static final int NEW = 0;
    private static final int READY = 1;
    private static final int RUNNING = 2;
    protected static final int FINISHED = 3;
    private static final int TERMINATED = 4;

    /**
     * Current algorithm state
     */
    protected int state = NEW;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------- Internal variables
    /////////////////////////////////////////////////////////////////

    /**
     * Registered listeners collection
     */
    private ArrayList<IAlgorithmListener> listeners = new ArrayList<>();

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty (default) constructor
     */
    public AbstractAlgorithm() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // IAlgorithm interface

    /**
     * {@inheritDoc}
     */
    public final void addListener(IAlgorithmListener listener) {
        listeners.add(listener);
    }

    /**
     * {@inheritDoc}
     */
    public final boolean removeListener(IAlgorithmListener listener) {
        return listeners.remove(listener);
    }

    // Execution methods

    /**
     * {@inheritDoc}
     */
    @Override
    public void pause() {
        state = READY;
    }

    /**
     * {@inheritDoc}
     */

    @Override
    public void terminate() {
        state = TERMINATED;
    }

    /**
     * {@inheritDoc}
     */

    public void execute() {
        do {
            switch (state) {
                case (NEW):
                    // Change current state
                    state = RUNNING;
                    // Call doInit() method
                    doInit();
                    // Fire algorithm started event
                    fireAlgorithmStarted();
                    // Finish this switch
                    break;
                case (READY):
                    // Change current state
                    state = RUNNING;
                    // Finish this switch
                    break;
                case (RUNNING):
                    // Perform an iteration
                    doIterate();
                    // Fire Iteration completed event
                    if (state == RUNNING)
                        fireIterationCompleted();
                    // Finish this switch
                    break;
            }
        }
        while (state == RUNNING);
        // If algorithm has finished...
        if (state == FINISHED) {
            // Fire algorithm terminated event
            fireAlgorithmFinished();
            // Change current state
            state = NEW;
            // Finish this switch
            return;
        }
        // If algorithm was terminated...
        if (state == TERMINATED) {
            // Fire algorithm terminated event
            fireAlgorithmTerminated();
            // Change current state
            state = NEW;
            // Finish this switch
        }
    }

    // IConfigure interface

    /**
     * {@inheritDoc}
     * <p>
     * This method register one or several algorithm listeners to this algorithm.
     */

    @SuppressWarnings("unchecked")
    public void configure(Configuration configuration) {
        // Number of defined listeners
        int numberOfListeners = configuration.getList("listener[@type]").size();
        // For each listener in list
        for (int i = 0; i < numberOfListeners; i++) {
            String header = "listener(" + i + ")";
            try {
                // Listener classname
                String listenerClassname = configuration.getString(header + "[@type]");
                // Listener class
                Class<? extends IAlgorithmListener> listenerClass =
                        (Class<? extends IAlgorithmListener>) Class.forName(listenerClassname);
                // Listener instance
                IAlgorithmListener listener = listenerClass.newInstance();
                // Configure listener (if necessary)
                if (listener instanceof IConfigure) {
                    ((IConfigure) listener).configure(configuration.subset(header));
                }
                // Add this listener to the algorithm
                addListener(listener);
            } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
                throw new ConfigurationRuntimeException("Illegal listener classname", e);
            }
        }
    }

    /////////////////////////////////////////////////////////////////
    // -------------------------------------------- Protected methods
    /////////////////////////////////////////////////////////////////

    // Algorithm execution

    /**
     * Perform algorithm initialization.
     */
    protected abstract void doInit();

    /**
     * Perform an algorithm iteration.
     */
    protected abstract void doIterate();

    // Fire events
    private void fireAlgorithmStarted() {
        AlgorithmEvent event = new AlgorithmEvent(this);

        for (IAlgorithmListener listener : listeners) {
            listener.algorithmStarted(event);
        }
    }

    private void fireIterationCompleted() {
        AlgorithmEvent event = new AlgorithmEvent(this);

        for (IAlgorithmListener listener : listeners) {
            listener.iterationCompleted(event);
        }
    }

    private void fireAlgorithmFinished() {
        AlgorithmEvent event = new AlgorithmEvent(this);

        for (IAlgorithmListener listener : listeners) {
            listener.algorithmFinished(event);
        }
    }

    private void fireAlgorithmTerminated() {
        AlgorithmEvent event = new AlgorithmEvent(this);

        for (IAlgorithmListener listener : listeners) {
            listener.algorithmTerminated(event);
        }
    }
}

