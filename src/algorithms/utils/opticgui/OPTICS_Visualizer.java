package algorithms.utils.opticgui;

import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.gui.LookAndFeel;

import javax.swing.*;
import javax.swing.table.DefaultTableColumnModel;
import javax.swing.table.TableColumn;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.*;
import java.util.ArrayList;
import java.util.GregorianCalendar;

public class OPTICS_Visualizer implements RevisionHandler {
    private SERObject serObject;
    private JFrame frame;
    private JFrame statisticsFrame;
    private JFrame helpFrame;
    private FrameListener frameListener;
    private JToolBar toolBar;
    private JButton toolBarButton_open;
    private JButton toolBarButton_save;
    private JButton toolBarButton_parameters;
    private JButton toolBarButton_help;
    private JButton toolBarButton_about;
    private JMenuBar defaultMenuBar;
    private JMenuItem open;
    private JMenuItem save;
    private JMenuItem exit;
    private JMenuItem parameters;
    private JMenuItem help;
    private JMenuItem about;
    private JTabbedPane tabbedPane;
    private JTable resultVectorTable;
    private GraphPanel graphPanel;
    private JScrollPane graphPanelScrollPane;
    private JPanel settingsPanel;
    private JCheckBox showCoreDistances;
    private JCheckBox showReachabilityDistances;
    private int verValue = 30;
    private JSlider verticalSlider;
    private JButton coreDistanceColorButton;
    private JButton reachDistanceColorButton;
    private JButton graphBackgroundColorButton;
    private JButton resetColorButton;
    private JFileChooser jFileChooser;
    private String lastPath;

    public OPTICS_Visualizer(SERObject serObject, String title) {
        this.serObject = serObject;
        LookAndFeel.setLookAndFeel();
        this.frame = new JFrame(title);
        this.frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                OPTICS_Visualizer.this.frame.dispose();
            }
        });
        this.frame.getContentPane().setLayout(new BorderLayout());
        this.frame.setSize(new Dimension(800, 600));
        Dimension screenDimension = Toolkit.getDefaultToolkit().getScreenSize();
        Rectangle windowRectangle = this.frame.getBounds();
        this.frame.setLocation((screenDimension.width - windowRectangle.width) / 2, (screenDimension.height - windowRectangle.height) / 2);
        this.frameListener = new OPTICS_Visualizer.FrameListener();
        this.jFileChooser = new JFileChooser();
        this.jFileChooser.setFileFilter(new SERFileFilter("ser", "Java Serialized Object File (*.ser)"));
        this.createGUI();
        this.frame.setVisible(true);
        this.frame.toFront();
    }

    private void createGUI() {
        this.setMenuBar(this.constructDefaultMenuBar());
        this.frame.getContentPane().add(this.createToolBar(), "North");
        this.frame.getContentPane().add(this.createTabbedPane(), "Center");
        this.frame.getContentPane().add(this.createSettingsPanel(), "South");
        this.disableSettingsPanel();
    }

    private JComponent createSettingsPanel() {
        this.settingsPanel = new JPanel(new GridBagLayout());
        OPTICS_Visualizer.SettingsPanelListener panelListener = new OPTICS_Visualizer.SettingsPanelListener();
        JPanel setPanelLeft = new JPanel(new GridBagLayout());
        setPanelLeft.setBorder(BorderFactory.createTitledBorder(" General Settings "));
        JPanel checkBoxesPanel = new JPanel(new GridLayout(1, 2));
        this.showCoreDistances = new JCheckBox("Show Core-Distances");
        this.showCoreDistances.setSelected(true);
        this.showReachabilityDistances = new JCheckBox("Show Reachability-Distances");
        this.showReachabilityDistances.setSelected(true);
        this.showCoreDistances.addItemListener(e -> {
            if (e.getStateChange() == 1) {
                OPTICS_Visualizer.this.graphPanel.setShowCoreDistances(true);
                OPTICS_Visualizer.this.graphPanel.adjustSize(OPTICS_Visualizer.this.serObject);
                OPTICS_Visualizer.this.graphPanel.repaint();
            } else if (e.getStateChange() == 2) {
                OPTICS_Visualizer.this.graphPanel.setShowCoreDistances(false);
                OPTICS_Visualizer.this.graphPanel.adjustSize(OPTICS_Visualizer.this.serObject);
                OPTICS_Visualizer.this.graphPanel.repaint();
            }

        });
        this.showReachabilityDistances.addItemListener(e -> {
            if (e.getStateChange() == 1) {
                OPTICS_Visualizer.this.graphPanel.setShowReachabilityDistances(true);
                OPTICS_Visualizer.this.graphPanel.adjustSize(OPTICS_Visualizer.this.serObject);
                OPTICS_Visualizer.this.graphPanel.repaint();
            } else if (e.getStateChange() == 2) {
                OPTICS_Visualizer.this.graphPanel.setShowReachabilityDistances(false);
                OPTICS_Visualizer.this.graphPanel.adjustSize(OPTICS_Visualizer.this.serObject);
                OPTICS_Visualizer.this.graphPanel.repaint();
            }

        });
        checkBoxesPanel.add(this.showCoreDistances);
        checkBoxesPanel.add(this.showReachabilityDistances);
        JPanel verticalAdPanel = new JPanel(new BorderLayout());
        final JLabel verValueLabel = new JLabel("Vertical Adjustment: " + this.verValue);
        verticalAdPanel.add(verValueLabel, "North");
        this.verticalSlider = new JSlider(0, 0, this.frame.getHeight(), this.verValue);
        this.verticalSlider.setMajorTickSpacing(100);
        this.verticalSlider.setMinorTickSpacing(10);
        this.verticalSlider.setPaintTicks(true);
        this.verticalSlider.setPaintLabels(true);
        this.verticalSlider.addChangeListener(e -> {
            if (!OPTICS_Visualizer.this.verticalSlider.getValueIsAdjusting()) {
                OPTICS_Visualizer.this.verValue = OPTICS_Visualizer.this.verticalSlider.getValue();
                verValueLabel.setText("Vertical Adjustment: " + OPTICS_Visualizer.this.verValue);
                OPTICS_Visualizer.this.graphPanel.setVerticalAdjustment(OPTICS_Visualizer.this.verValue);
                OPTICS_Visualizer.this.graphPanel.repaint();
            }

        });
        verticalAdPanel.add(this.verticalSlider, "Center");
        setPanelLeft.add(checkBoxesPanel, new GridBagConstraints(0, 0, 1, 1, 1.0D, 1.0D, 10, 1, new Insets(5, 5, 5, 5), 0, 0));
        setPanelLeft.add(verticalAdPanel, new GridBagConstraints(0, 1, 1, 1, 1.0D, 1.0D, 10, 1, new Insets(5, 5, 5, 5), 0, 0));
        this.settingsPanel.add(setPanelLeft, new GridBagConstraints(0, 0, 1, 1, 3.0D, 1.0D, 10, 1, new Insets(5, 5, 5, 0), 0, 0));
        JPanel setPanelRight = new JPanel(new GridBagLayout());
        setPanelRight.setBorder(BorderFactory.createTitledBorder(" Colors "));
        JPanel colorsPanel = new JPanel(new GridLayout(4, 2, 10, 10));
        colorsPanel.add(new JLabel("Core-Distance: "));
        this.coreDistanceColorButton = new JButton();
        this.coreDistanceColorButton.setBackground(new Color(100, 100, 100));
        this.coreDistanceColorButton.addActionListener(panelListener);
        colorsPanel.add(this.coreDistanceColorButton);
        colorsPanel.add(new JLabel("Reachability-Distance: "));
        this.reachDistanceColorButton = new JButton();
        this.reachDistanceColorButton.setBackground(Color.orange);
        this.reachDistanceColorButton.addActionListener(panelListener);
        colorsPanel.add(this.reachDistanceColorButton);
        colorsPanel.add(new JLabel("Graph Background: "));
        this.graphBackgroundColorButton = new JButton();
        this.graphBackgroundColorButton.setBackground(new Color(255, 255, 179));
        this.graphBackgroundColorButton.addActionListener(panelListener);
        colorsPanel.add(this.graphBackgroundColorButton);
        colorsPanel.add(new JLabel());
        this.resetColorButton = new JButton("Reset");
        this.resetColorButton.addActionListener(panelListener);
        colorsPanel.add(this.resetColorButton);
        setPanelRight.add(colorsPanel, new GridBagConstraints(0, 0, 1, 1, 1.0D, 1.0D, 10, 1, new Insets(5, 5, 5, 5), 0, 0));
        this.settingsPanel.add(setPanelRight, new GridBagConstraints(1, 0, 1, 1, 1.0D, 1.0D, 10, 1, new Insets(5, 5, 5, 5), 0, 0));
        return this.settingsPanel;
    }

    private void disableSettingsPanel() {
        this.verticalSlider.setEnabled(false);
        this.coreDistanceColorButton.setEnabled(false);
        this.reachDistanceColorButton.setEnabled(false);
        this.graphBackgroundColorButton.setEnabled(false);
        this.resetColorButton.setEnabled(false);
        this.settingsPanel.setVisible(false);
    }

    private void enableSettingsPanel() {
        this.verticalSlider.setEnabled(true);
        this.coreDistanceColorButton.setEnabled(true);
        this.reachDistanceColorButton.setEnabled(true);
        this.graphBackgroundColorButton.setEnabled(true);
        this.resetColorButton.setEnabled(true);
        this.settingsPanel.setVisible(true);
    }

    private JComponent createTabbedPane() {
        this.tabbedPane = new JTabbedPane();
        this.tabbedPane.addTab("Table", new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Table16.gif"))), this.clusteringResultsTable(), "Show table of DataObjects, Core- and Reachability-Distances");
        if (this.serObject != null) {
            this.tabbedPane.addTab("Graph - Epsilon: " + this.serObject.getEpsilon() + ", MinPoints: " + this.serObject.getMinPoints(), new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Graph16.gif"))), this.graphPanel(), "Show Plot of Core- and Reachability-Distances");
        } else {
            this.tabbedPane.addTab("Graph - Epsilon: --, MinPoints: --", new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Graph16.gif"))), this.graphPanel(), "Show Plot of Core- and Reachability-Distances");
        }

        this.tabbedPane.addChangeListener(e -> {
            int c = OPTICS_Visualizer.this.tabbedPane.getSelectedIndex();
            if (c == 0) {
                OPTICS_Visualizer.this.disableSettingsPanel();
            } else {
                OPTICS_Visualizer.this.enableSettingsPanel();
            }

        });
        return this.tabbedPane;
    }

    private JComponent createToolBar() {
        this.toolBar = new JToolBar();
        this.toolBar.setName("OPTICS Visualizer ToolBar");
        this.toolBar.setFloatable(false);
        this.toolBarButton_open = new JButton(new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Open16.gif"))));
        this.toolBarButton_open.setToolTipText("Open OPTICS-Session");
        this.toolBarButton_open.addActionListener(this.frameListener);
        this.toolBar.add(this.toolBarButton_open);
        this.toolBarButton_save = new JButton(new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Save16.gif"))));
        this.toolBarButton_save.setToolTipText("Save OPTICS-Session");
        this.toolBarButton_save.addActionListener(this.frameListener);
        this.toolBar.add(this.toolBarButton_save);
        this.toolBar.addSeparator(new Dimension(10, 25));
        this.toolBarButton_parameters = new JButton(new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Parameters16.gif"))));
        this.toolBarButton_parameters.setToolTipText("Show epsilon, MinPoints...");
        this.toolBarButton_parameters.addActionListener(this.frameListener);
        this.toolBar.add(this.toolBarButton_parameters);
        this.toolBar.addSeparator(new Dimension(10, 25));
        this.toolBarButton_help = new JButton(new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Help16.gif"))));
        this.toolBarButton_help.setToolTipText("Help topics");
        this.toolBarButton_help.addActionListener(this.frameListener);
        this.toolBar.add(this.toolBarButton_help);
        this.toolBarButton_about = new JButton(new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Information16.gif"))));
        this.toolBarButton_about.setToolTipText("About");
        this.toolBarButton_about.addActionListener(this.frameListener);
        this.toolBar.add(this.toolBarButton_about);
        return this.toolBar;
    }

    private JComponent clusteringResultsTable() {
        this.resultVectorTable = new JTable();
        String[] resultVectorTableColumnNames = new String[]{"Key", "DataObject", "Core-Distance", "Reachability-Distance"};
        DefaultTableColumnModel resultVectorTableColumnModel = new DefaultTableColumnModel();

        for(int i = 0; i < resultVectorTableColumnNames.length; ++i) {
            TableColumn tc = new TableColumn(i);
            tc.setHeaderValue(resultVectorTableColumnNames[i]);
            resultVectorTableColumnModel.addColumn(tc);
        }

        ResultVectorTableModel resultVectorTableModel;
        if (this.serObject != null) {
            resultVectorTableModel = new ResultVectorTableModel(this.serObject.getResultVector());
        } else {
            resultVectorTableModel = new ResultVectorTableModel(null);
        }

        this.resultVectorTable = new JTable(resultVectorTableModel, resultVectorTableColumnModel);
        this.resultVectorTable.getColumnModel().getColumn(0).setPreferredWidth(70);
        this.resultVectorTable.getColumnModel().getColumn(1).setPreferredWidth(400);
        this.resultVectorTable.getColumnModel().getColumn(2).setPreferredWidth(150);
        this.resultVectorTable.getColumnModel().getColumn(3).setPreferredWidth(150);
        this.resultVectorTable.setAutoResizeMode(0);
        return new JScrollPane(this.resultVectorTable, 22, 32);
    }

    private JComponent graphPanel() {
        if (this.serObject == null) {
            this.graphPanel = new GraphPanel(new ArrayList(), this.verValue, true, true);
        } else {
            this.graphPanel = new GraphPanel(this.serObject.getResultVector(), this.verValue, true, true);
            this.graphPanel.setPreferredSize(new Dimension(10 * this.serObject.getDatabaseSize() + this.serObject.getDatabaseSize(), this.graphPanel.getHeight()));
        }

        this.graphPanel.setBackground(new Color(255, 255, 179));
        this.graphPanel.setOpaque(true);
        this.graphPanelScrollPane = new JScrollPane(this.graphPanel, 22, 32);
        return this.graphPanelScrollPane;
    }

    private JMenuBar constructDefaultMenuBar() {
        this.defaultMenuBar = new JMenuBar();
        JMenu fileMenu = new JMenu("File");
        fileMenu.setMnemonic('F');
        this.open = new JMenuItem("Open...", new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Open16.gif"))));
        this.open.setMnemonic('O');
        this.open.setAccelerator(KeyStroke.getKeyStroke(79, 2));
        this.open.addActionListener(this.frameListener);
        fileMenu.add(this.open);
        this.save = new JMenuItem("Save...", new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Save16.gif"))));
        this.save.setMnemonic('S');
        this.save.setAccelerator(KeyStroke.getKeyStroke(83, 2));
        this.save.addActionListener(this.frameListener);
        fileMenu.add(this.save);
        fileMenu.addSeparator();
        this.exit = new JMenuItem("Exit", 88);
        this.exit.addActionListener(this.frameListener);
        fileMenu.add(this.exit);
        this.defaultMenuBar.add(fileMenu);
        JMenu toolsMenu = new JMenu("View");
        toolsMenu.setMnemonic('V');
        this.parameters = new JMenuItem("Parameters...", new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Parameters16.gif"))));
        this.parameters.setMnemonic('P');
        this.parameters.setAccelerator(KeyStroke.getKeyStroke(80, 2));
        this.parameters.addActionListener(this.frameListener);
        toolsMenu.add(this.parameters);
        this.defaultMenuBar.add(toolsMenu);
        JMenu miscMenu = new JMenu("Help");
        miscMenu.setMnemonic('H');
        this.help = new JMenuItem("Help Topics", new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Help16.gif"))));
        this.help.setMnemonic('H');
        this.help.setAccelerator(KeyStroke.getKeyStroke(72, 2));
        this.help.addActionListener(this.frameListener);
        miscMenu.add(this.help);
        this.about = new JMenuItem("About...", new ImageIcon(Toolkit.getDefaultToolkit().getImage(ClassLoader.getSystemResource("weka/clusterers/forOPTICSAndDBScan/OPTICS_GUI/Graphics/Information16.gif"))));
        this.about.setMnemonic('A');
        this.about.setAccelerator(KeyStroke.getKeyStroke(65, 2));
        this.about.addActionListener(this.frameListener);
        miscMenu.add(this.about);
        this.defaultMenuBar.add(miscMenu);
        return this.defaultMenuBar;
    }

    private void setMenuBar(JMenuBar menuBar) {
        this.frame.setJMenuBar(menuBar);
    }

    private void loadStatisticsFrame() {
        this.statisticsFrame = new JFrame("Parameters");
        this.statisticsFrame.getContentPane().setLayout(new BorderLayout());
        JPanel statPanel_Labels = new JPanel(new GridBagLayout());
        JPanel statPanel_Labels_Left = new JPanel(new GridLayout(9, 1));
        JPanel statPanel_Labels_Right = new JPanel(new GridLayout(9, 1));
        statPanel_Labels_Left.add(new JLabel("Number of clustered DataObjects: "));
        statPanel_Labels_Right.add(new JLabel(Integer.toString(this.serObject.getDatabaseSize())));
        statPanel_Labels_Left.add(new JLabel("Number of attributes: "));
        statPanel_Labels_Right.add(new JLabel(Integer.toString(this.serObject.getNumberOfAttributes())));
        statPanel_Labels_Left.add(new JLabel("Epsilon: "));
        statPanel_Labels_Right.add(new JLabel(Double.toString(this.serObject.getEpsilon())));
        statPanel_Labels_Left.add(new JLabel("MinPoints: "));
        statPanel_Labels_Right.add(new JLabel(Integer.toString(this.serObject.getMinPoints())));
        statPanel_Labels_Left.add(new JLabel("Write results to file: "));
        statPanel_Labels_Right.add(new JLabel(this.serObject.isOpticsOutputs() ? "yes" : "no"));
        statPanel_Labels_Left.add(new JLabel("Distance-Type: "));
        statPanel_Labels_Right.add(new JLabel(this.serObject.getDistanceFunction().toString()));
        statPanel_Labels_Left.add(new JLabel("Number of generated clusters: "));
        statPanel_Labels_Right.add(new JLabel(Integer.toString(this.serObject.getNumberOfGeneratedClusters())));
        statPanel_Labels_Left.add(new JLabel("Elapsed-time: "));
        statPanel_Labels_Right.add(new JLabel(this.serObject.getElapsedTime()));
        statPanel_Labels.setBorder(BorderFactory.createTitledBorder(" OPTICS parameters "));
        statPanel_Labels.add(statPanel_Labels_Left, new GridBagConstraints(0, 0, 1, 1, 1.0D, 1.0D, 10, 1, new Insets(0, 5, 2, 0), 0, 0));
        statPanel_Labels.add(statPanel_Labels_Right, new GridBagConstraints(1, 0, 1, 1, 3.0D, 1.0D, 10, 1, new Insets(0, 5, 2, 5), 0, 0));
        this.statisticsFrame.getContentPane().add(statPanel_Labels, "Center");
        this.statisticsFrame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                OPTICS_Visualizer.this.statisticsFrame.dispose();
            }
        });
        JPanel okButtonPanel = new JPanel(new GridBagLayout());
        JButton okButton = new JButton("OK");
        okButton.addActionListener(e -> {
            if (e.getActionCommand().equals("OK")) {
                OPTICS_Visualizer.this.statisticsFrame.dispose();
            }

        });
        okButtonPanel.add(okButton, new GridBagConstraints(0, 0, 1, 1, 1.0D, 1.0D, 10, 0, new Insets(5, 0, 5, 0), 0, 0));
        this.statisticsFrame.getContentPane().add(okButtonPanel, "South");
        this.statisticsFrame.setSize(new Dimension(500, 300));
        Rectangle frameDimension = this.frame.getBounds();
        Point p = this.frame.getLocation();
        Rectangle statisticsFrameDimension = this.statisticsFrame.getBounds();
        this.statisticsFrame.setLocation((frameDimension.width - statisticsFrameDimension.width) / 2 + (int)p.getX(), (frameDimension.height - statisticsFrameDimension.height) / 2 + (int)p.getY());
        this.statisticsFrame.setVisible(true);
        this.statisticsFrame.toFront();
    }

    private void loadHelpFrame() {
        this.helpFrame = new JFrame("Help Topics");
        this.helpFrame.getContentPane().setLayout(new BorderLayout());
        JPanel helpPanel = new JPanel(new GridBagLayout());
        JTextArea helpTextArea = new JTextArea();
        helpTextArea.setEditable(false);
        helpTextArea.append("OPTICS Visualizer Help\n===========================================================\n\nOpen\n - Open OPTICS-Session\n   [Ctrl-O], File | Open\n\nSave\n - Save OPTICS-Session\n   [Ctrl-S], File | Save\n\nExit\n - Exit OPTICS Visualizer\n   [Alt-F4], File | Exit\n\nParameters\n - Show epsilon, MinPoints...\n   [Ctrl-P], View | Parameters\n\nHelp Topics\n - Show this frame\n   [Ctrl-H], Help | Help Topics\n\nAbout\n - Copyright-Information\n   [Ctrl-A], Help | About\n\n\nTable-Pane:\n-----------------------------------------------------------\nThe table represents the calculated clustering-order.\nTo save the table please select File | Save from the\nmenubar. Restart OPTICS with the -F option to obtain\nan ASCII-formatted file of the clustering-order.\n\nGraph-Pane:\n-----------------------------------------------------------\nThe graph draws the plot of core- and reachability-\ndistances. By (de-)activating core- and reachability-\ndistances in the 'General Settings'-Panel you can\ninfluence the visualization in detail. Simply use the\n'Vertical Adjustment'-Slider to emphasize the plot of\ndistances. The 'Colors'-Panel lets you define different\ncolors of the graph background, core- and reachability-\ndistances. Click the 'Reset'-Button to restore the\ndefaults.\n");
        final JScrollPane helpTextAreaScrollPane = new JScrollPane(helpTextArea, 22, 32);
        helpTextAreaScrollPane.setBorder(BorderFactory.createEtchedBorder());
        helpPanel.add(helpTextAreaScrollPane, new GridBagConstraints(0, 0, 1, 1, 1.0D, 1.0D, 10, 1, new Insets(5, 5, 7, 5), 0, 0));
        this.helpFrame.getContentPane().add(helpPanel, "Center");
        this.helpFrame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                OPTICS_Visualizer.this.helpFrame.dispose();
            }

            public void windowOpened(WindowEvent e) {
                helpTextAreaScrollPane.getVerticalScrollBar().setValue(0);
            }
        });
        JPanel closeButtonPanel = new JPanel(new GridBagLayout());
        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(e -> {
            if (e.getActionCommand().equals("Close")) {
                OPTICS_Visualizer.this.helpFrame.dispose();
            }

        });
        closeButtonPanel.add(closeButton, new GridBagConstraints(0, 0, 1, 1, 1.0D, 1.0D, 10, 0, new Insets(0, 0, 5, 0), 0, 0));
        this.helpFrame.getContentPane().add(closeButtonPanel, "South");
        this.helpFrame.setSize(new Dimension(480, 400));
        Rectangle frameDimension = this.frame.getBounds();
        Point p = this.frame.getLocation();
        Rectangle helpFrameDimension = this.helpFrame.getBounds();
        this.helpFrame.setLocation((frameDimension.width - helpFrameDimension.width) / 2 + (int)p.getX(), (frameDimension.height - helpFrameDimension.height) / 2 + (int)p.getY());
        this.helpFrame.setVisible(true);
        this.helpFrame.toFront();
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10838 $");
    }

    /*public static void main(String[] args) throws Exception {
        SERObject serObject = null;
        if (args.length == 1) {
            System.out.println("Attempting to load: " + args[0]);
            ObjectInputStream is = null;

            try {
                FileInputStream fs = new FileInputStream(args[0]);
                is = new ObjectInputStream(fs);
                serObject = (SERObject)is.readObject();
            } catch (Exception var12) {
                JOptionPane.showMessageDialog(null, "Error loading file:\n" + var12, "Error", 0);
            } finally {
                try {
                    assert is != null;
                    is.close();
                } catch (Exception ignored) {
                }

            }
        }

        new OPTICS_Visualizer(serObject, "OPTICS Visualizer - Main Window");
    }*/

    private class SettingsPanelListener implements ActionListener, RevisionHandler {
        private SettingsPanelListener() {
        }

        public void actionPerformed(ActionEvent e) {
            Color c;
            if (e.getSource() == OPTICS_Visualizer.this.coreDistanceColorButton) {
                c = this.getSelectedColor("Select 'Core-Distance' color");
                if (c != null) {
                    OPTICS_Visualizer.this.coreDistanceColorButton.setBackground(c);
                    OPTICS_Visualizer.this.graphPanel.setCoreDistanceColor(c);
                }
            }

            if (e.getSource() == OPTICS_Visualizer.this.reachDistanceColorButton) {
                c = this.getSelectedColor("Select 'Reachability-Distance' color");
                if (c != null) {
                    OPTICS_Visualizer.this.reachDistanceColorButton.setBackground(c);
                    OPTICS_Visualizer.this.graphPanel.setReachabilityDistanceColor(c);
                }
            }

            if (e.getSource() == OPTICS_Visualizer.this.graphBackgroundColorButton) {
                c = this.getSelectedColor("Select 'Graph Background' color");
                if (c != null) {
                    OPTICS_Visualizer.this.graphBackgroundColorButton.setBackground(c);
                    OPTICS_Visualizer.this.graphPanel.setBackground(c);
                }
            }

            if (e.getSource() == OPTICS_Visualizer.this.resetColorButton) {
                OPTICS_Visualizer.this.coreDistanceColorButton.setBackground(new Color(100, 100, 100));
                OPTICS_Visualizer.this.graphPanel.setCoreDistanceColor(new Color(100, 100, 100));
                OPTICS_Visualizer.this.reachDistanceColorButton.setBackground(Color.orange);
                OPTICS_Visualizer.this.graphPanel.setReachabilityDistanceColor(Color.orange);
                OPTICS_Visualizer.this.graphBackgroundColorButton.setBackground(new Color(255, 255, 179));
                OPTICS_Visualizer.this.graphPanel.setBackground(new Color(255, 255, 179));
                OPTICS_Visualizer.this.graphPanel.repaint();
            }

        }

        private Color getSelectedColor(String title) {
            return JColorChooser.showDialog(OPTICS_Visualizer.this.frame, title, Color.black);
        }

        public String getRevision() {
            return RevisionUtils.extract("$Revision: 10838 $");
        }
    }

    private class FrameListener implements ActionListener, RevisionHandler {
        private FrameListener() {
        }

        public void actionPerformed(ActionEvent e) {
            if (e.getSource() == OPTICS_Visualizer.this.parameters || e.getSource() == OPTICS_Visualizer.this.toolBarButton_parameters) {
                OPTICS_Visualizer.this.loadStatisticsFrame();
            }

            if (e.getSource() == OPTICS_Visualizer.this.about || e.getSource() == OPTICS_Visualizer.this.toolBarButton_about) {
                JOptionPane.showMessageDialog(OPTICS_Visualizer.this.frame, "OPTICS Visualizer\n$ Rev 1.4 $\n\nCopyright (C) 2004 Rainer Holzmann, Zhanna Melnikova-Albrecht", "About", 1);
            }

            if (e.getSource() == OPTICS_Visualizer.this.help || e.getSource() == OPTICS_Visualizer.this.toolBarButton_help) {
                OPTICS_Visualizer.this.loadHelpFrame();
            }

            if (e.getSource() == OPTICS_Visualizer.this.exit) {
                OPTICS_Visualizer.this.frame.dispose();
            }

            if (e.getSource() == OPTICS_Visualizer.this.open || e.getSource() == OPTICS_Visualizer.this.toolBarButton_open) {
                OPTICS_Visualizer.this.jFileChooser.setDialogTitle("Open OPTICS-Session");
                if (OPTICS_Visualizer.this.lastPath == null) {
                    OPTICS_Visualizer.this.lastPath = System.getProperty("user.dir");
                }

                OPTICS_Visualizer.this.jFileChooser.setCurrentDirectory(new File(OPTICS_Visualizer.this.lastPath));
                int ret = OPTICS_Visualizer.this.jFileChooser.showOpenDialog(OPTICS_Visualizer.this.frame);
                SERObject serObject_1 = null;
                if (ret == 0) {
                    File f = OPTICS_Visualizer.this.jFileChooser.getSelectedFile();

                    try {
                        FileInputStream fs = new FileInputStream(f.getAbsolutePath());
                        ObjectInputStream is = new ObjectInputStream(fs);
                        serObject_1 = (SERObject)is.readObject();
                        is.close();
                    } catch (FileNotFoundException var10) {
                        JOptionPane.showMessageDialog(OPTICS_Visualizer.this.frame, "File not found.", "Error", 0);
                    } catch (ClassNotFoundException var11) {
                        JOptionPane.showMessageDialog(OPTICS_Visualizer.this.frame, "OPTICS-Session could not be read.", "Error", 0);
                    } catch (IOException var12) {
                        JOptionPane.showMessageDialog(OPTICS_Visualizer.this.frame, "This file does not contain a valid OPTICS-Session.", "Error", 0);
                    }

                    if (serObject_1 != null) {
                        int ret_1 = JOptionPane.showConfirmDialog(OPTICS_Visualizer.this.frame, "Open OPTICS-Session in a new window?", "Open", 1);
                        switch(ret_1) {
                            case 0:
                                new OPTICS_Visualizer(serObject_1, "OPTICS Visualizer - " + f.getName());
                                break;
                            case 1:
                                OPTICS_Visualizer.this.serObject = serObject_1;
                                OPTICS_Visualizer.this.resultVectorTable.setModel(new ResultVectorTableModel(OPTICS_Visualizer.this.serObject.getResultVector()));
                                OPTICS_Visualizer.this.tabbedPane.setTitleAt(1, "Graph - Epsilon: " + OPTICS_Visualizer.this.serObject.getEpsilon() + ", MinPoints: " + OPTICS_Visualizer.this.serObject.getMinPoints());
                                OPTICS_Visualizer.this.graphPanel.setResultVector(OPTICS_Visualizer.this.serObject.getResultVector());
                                OPTICS_Visualizer.this.graphPanel.adjustSize(OPTICS_Visualizer.this.serObject);
                                OPTICS_Visualizer.this.graphPanel.repaint();
                        }
                    }
                }
            }

            if (e.getSource() == OPTICS_Visualizer.this.save || e.getSource() == OPTICS_Visualizer.this.toolBarButton_save) {
                OPTICS_Visualizer.this.jFileChooser.setDialogTitle("Save OPTICS-Session");
                GregorianCalendar gregorianCalendar = new GregorianCalendar();
                String timeStamp = gregorianCalendar.get(5) + "-" + (gregorianCalendar.get(2) + 1) + "-" + gregorianCalendar.get(1) + "--" + gregorianCalendar.get(11) + "-" + gregorianCalendar.get(12) + "-" + gregorianCalendar.get(13);
                String filename = "OPTICS_" + timeStamp + ".ser";
                File file = new File(filename);
                OPTICS_Visualizer.this.jFileChooser.setSelectedFile(file);
                if (OPTICS_Visualizer.this.lastPath == null) {
                    OPTICS_Visualizer.this.lastPath = System.getProperty("user.dir");
                }

                OPTICS_Visualizer.this.jFileChooser.setCurrentDirectory(new File(OPTICS_Visualizer.this.lastPath));
                int retx = OPTICS_Visualizer.this.jFileChooser.showSaveDialog(OPTICS_Visualizer.this.frame);
                if (retx == 0) {
                    file = OPTICS_Visualizer.this.jFileChooser.getSelectedFile();

                    try {
                        FileOutputStream fsx = new FileOutputStream(file.getAbsolutePath());
                        ObjectOutputStream os = new ObjectOutputStream(fsx);
                        os.writeObject(OPTICS_Visualizer.this.serObject);
                        os.flush();
                        os.close();
                    } catch (IOException var9) {
                        JOptionPane.showMessageDialog(OPTICS_Visualizer.this.frame, "OPTICS-Session could not be saved.", "Error", 0);
                    }
                }
            }

        }

        public String getRevision() {
            return RevisionUtils.extract("$Revision: 10838 $");
        }
    }
}
