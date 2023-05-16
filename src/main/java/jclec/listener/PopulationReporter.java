package jclec.listener;

import jclec.*;
import jclec.algorithm.PopulationAlgorithm;
import jclec.util.IndividualStatistics;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.lang.builder.EqualsBuilder;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

/**
 * This class is a listener for PopulationAlgorithms, that performs a report of
 * the actual population. This report consists on ...
 *
 * @author Sebastian Ventura
 */

public class PopulationReporter implements IAlgorithmListener, IConfigure {

    /////////////////////////////////////////////////////////////////
    // --------------------------------------------------- Properties
    /////////////////////////////////////////////////////////////////

    /**
     * Name of the report
     */
    protected String reportTitle;

    /**
     * Report frequency
     */
    protected int reportFrequency;

    /**
     * Show report on console?
     */
    protected boolean reportOnConsole;

    /**
     * Write report on file?
     */
    protected boolean reportOnFile;

    /**
     * Save all population individuals?
     */
    protected boolean saveCompletePopulation;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------- Internal variables
    /////////////////////////////////////////////////////////////////

    /**
     * Report file
     */
    protected File reportFile;

    /**
     * Report file writer
     */
    protected FileWriter reportFileWriter;
    protected String actualReportTitle;

    /**
     * Directory for saving complete populations
     */
    protected File reportDirectory;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    public PopulationReporter() {
        super();
    }

    /////////////////////////////////////////////////////////////////
    // ----------------------------------------------- Public methods
    /////////////////////////////////////////////////////////////////

    // Setting and getting properties
    private void setReportTitle(String reportTitle) {
        this.reportTitle = reportTitle;
    }
    private void setReportFrequency(int reportFrequency) {
        this.reportFrequency = reportFrequency;
    }
    private void setReportOnCconsole(boolean reportOnCconsole) {
        this.reportOnConsole = reportOnCconsole;
    }
    private void setReportOnFile(boolean reportOnFile) {
        this.reportOnFile = reportOnFile;
    }
    private void setSaveCompletePopulation(boolean saveCompletePopulation) {
        this.saveCompletePopulation = saveCompletePopulation;
    }

    // IConfigure interface

    @Override
    public void configure(Configuration settings) {
        // Set report title (default "untitled")
        String reportTitle = settings.getString("report-title", "untitled");
        setReportTitle(reportTitle);
        // Set report frequency (default 10 generations)
        int reportFrequency = settings.getInt("report-frequency", 10);
        setReportFrequency(reportFrequency);
        // Set console report (default on)
        boolean reportOnConsole = settings.getBoolean("report-on-console", true);
        setReportOnCconsole(reportOnConsole);
        // Set file report (default off)
        boolean reportOnFile = settings.getBoolean("report-on-file", false);
        setReportOnFile(reportOnFile);
        // Set save individuals (default false)
        boolean saveCompletePopulation = settings.getBoolean("save-complete-population", false);
        setSaveCompletePopulation(saveCompletePopulation);
    }

    // IAlgorithmListener interface

    @Override
    public void algorithmStarted(AlgorithmEvent event) {
        // Create report title for this instance
        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HHmmss");
        Date date = new Date(System.currentTimeMillis());
        String dateString = dateFormat.format(date);
        actualReportTitle = reportTitle + "_" + dateString;

        // If save complete population create a directory for storing
        // individual population files
        if (saveCompletePopulation) {
            reportDirectory = new File(actualReportTitle);
            if (!reportDirectory.mkdir()) {
                throw new RuntimeException("Error creating report directory");
            }
        }
        // If report is stored in a text file, create report file
        if (reportOnFile) {
            reportFile = new File(actualReportTitle + ".report.txt");
            File reportDirectory = new File(reportFile.getParent());
            if (!reportDirectory.exists() && !reportDirectory.mkdirs()) {
                throw new RuntimeException("Error creating report directory");
            }
            try {
                reportFileWriter = new FileWriter(reportFile);
                reportFileWriter.flush();
                reportFileWriter.write(dateString + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        // Do an iteration report
        doIterationReport((PopulationAlgorithm) event.getAlgorithm(), true);
    }

    @Override
    public void iterationCompleted(AlgorithmEvent event) {
        doIterationReport((PopulationAlgorithm) event.getAlgorithm(), false);
    }

    @Override
    public void algorithmFinished(AlgorithmEvent event) {
        // Do last generation report
        doIterationReport((PopulationAlgorithm) event.getAlgorithm(), true);
        // Close report file if necessary
        if (reportOnFile && reportFile != null) {
            try {
                reportFileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void algorithmTerminated(AlgorithmEvent e) {

    }

    // java.lang.Object methods

    @Override
    public boolean equals(Object other) {
        if (other instanceof PopulationReporter) {
            PopulationReporter cother = (PopulationReporter) other;
            EqualsBuilder eb = new EqualsBuilder();
            // reportTitle
            eb.append(reportTitle, cother.reportTitle);
            // reportFrequency
            eb.append(reportFrequency, cother.reportFrequency);
            // reportOnConsole
            eb.append(reportOnConsole, cother.reportOnConsole);
            // reportOnFile
            eb.append(reportOnFile, cother.reportOnFile);
            // saveCompletePopulation
            eb.append(saveCompletePopulation, cother.saveCompletePopulation);
            return eb.isEquals();
        } else {
            return false;
        }
    }

    protected void doIterationReport(PopulationAlgorithm algorithm, boolean force) {
        // Fitness comparator
        Comparator<IFitness> comparator = algorithm.getEvaluator().getComparator();
        // Population individuals
        List<IIndividual> inhabitants = algorithm.getInhabitants();
        // Actual generation
        int generation = algorithm.getGeneration();

        // Check if this is correct generation
        if (!force && generation % reportFrequency != 0) {
            return;
        }

        // Save population individuals (if this option was chosen)
        if (saveCompletePopulation) {
            String filename = "generation" + generation + ".individuals.txt";
            File file = new File(reportDirectory, filename);
            FileWriter filewriter;
            try {
                filewriter = new FileWriter(file);
                for (IIndividual ind : inhabitants) {
                    filewriter.flush();
                    filewriter.write(ind + "\n");
                }
                filewriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Do population report
        StringBuilder sb = new StringBuilder("Generation " + generation + " Report\n");
        // Best individual
        IIndividual best = IndividualStatistics.bestIndividual(inhabitants, comparator);
        sb.append("Best individual: ").append(best).append("\n");
        // Worst individual
        IIndividual worst = IndividualStatistics.worstIndividual(inhabitants, comparator);
        sb.append("Worst individual: ").append(worst).append("\n");
        // Median individual
        IIndividual median = IndividualStatistics.medianIndividual(inhabitants, comparator);
        sb.append("Median individual: ").append(median).append("\n");
        // Average fitness and fitness variance
        double[] avgvar = IndividualStatistics.averageFitnessAndFitnessVariance(inhabitants);
        sb.append("Average fitness = ").append(avgvar[0]).append("\n");
        sb.append("Fitness variance = ").append(avgvar[1]).append("\n");

        // Write report string to the standard output (if necessary)
        if (reportOnConsole) {
            System.out.println(sb.toString());
        }

        // Write string to the report file (if necessary)
        if (reportOnFile) {
            try {
                reportFileWriter.write(sb.toString());
                reportFileWriter.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
