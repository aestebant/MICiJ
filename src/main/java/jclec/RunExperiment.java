package jclec;

import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.XMLConfiguration;

import java.io.File;

/**
 * Experiments runner
 *
 * @author Sebastian Ventura
 */

public class RunExperiment {
    /**
     * Usage message error
     */

    private static final String USAGE_MES = "\tjclec.RunExperiment <experiment file>\t\t(Execute experiment)";

    /**
     * Main method
     *
     * @param args Configuration File
     */

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage:" + USAGE_MES);
            System.exit(1);
        }
        // First arg must be a filename
        else {
            long start = System.nanoTime();

            ExperimentBuilder builder = new ExperimentBuilder();

            System.out.println("Initializing job");

            // Expand the processes and execute them
            for (String experiment : builder.buildExperiment(args[0])) {
                System.out.println("Algorithm started");
                executeJob(experiment);
                System.out.println("Algorithm finished");
            }
            double time = (System.nanoTime() - start) * 1e-9;
            System.out.println("Job finished");
            System.out.println("Time (s): " + time);
        }
    }

    /**
     * Execute experiment
     *
     * @param jobFilename Configuration file of the experiment
     */

    @SuppressWarnings("unchecked")
    private static void executeJob(String jobFilename) {
        // Try open job file
        File jobFile = new File(jobFilename);
        if (jobFile.exists()) {
            try {
                // Job configuration
                XMLConfiguration jobConf = new XMLConfiguration(jobFile);
                // Process header
                String header = "process";
                // Create and configure algorithms
                String aname = jobConf.getString(header + "[@algorithm-type]");
                Class<IAlgorithm> aclass = (Class<IAlgorithm>) Class.forName(aname);
                IAlgorithm algorithm = aclass.getDeclaredConstructor().newInstance();
                // Configure runner
                if (algorithm instanceof IConfigure) {
                    ((IConfigure) algorithm).configure(jobConf.subset(header));
                }
                // Execute algorithm runner
                algorithm.execute();
            } catch (ConfigurationException e) {
                System.out.println("Configuration exception ");
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("Job file not found");
            System.exit(1);
        }
    }
}