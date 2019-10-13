package algorithms.utils.opticgui;

import algorithms.utils.DataObject;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.ArrayList;

public class GraphPanel extends JComponent implements RevisionHandler {
    private static final long serialVersionUID = 7917937528738361470L;
    private ArrayList resultVector;
    private int verticalAdjustment;
    private Color coreDistanceColor;
    private Color reachabilityDistanceColor;
    private int widthSlider;
    private boolean showCoreDistances;
    private boolean showReachabilityDistances;
    private int recentIndex = -1;

    public GraphPanel(ArrayList resultVector, int verticalAdjustment, boolean showCoreDistances, boolean showReachbilityDistances) {
        this.resultVector = resultVector;
        this.verticalAdjustment = verticalAdjustment;
        this.coreDistanceColor = new Color(100, 100, 100);
        this.reachabilityDistanceColor = Color.orange;
        this.widthSlider = 5;
        this.showCoreDistances = showCoreDistances;
        this.showReachabilityDistances = showReachbilityDistances;
        this.addMouseMotionListener(new GraphPanel.MouseHandler());
    }

    protected void paintComponent(Graphics g) {
        if (this.isOpaque()) {
            Dimension size = this.getSize();
            g.setColor(this.getBackground());
            g.fillRect(0, 0, size.width, size.height);
        }

        int stepSize = 0;

        for(int vectorIndex = 0; vectorIndex < this.resultVector.size(); ++vectorIndex) {
            double coreDistance = ((DataObject)this.resultVector.get(vectorIndex)).getCoreDistance();
            double reachDistance = ((DataObject)this.resultVector.get(vectorIndex)).getReachabilityDistance();
            int cDist;
            if (coreDistance == 2.147483647E9D) {
                cDist = this.getHeight();
            } else {
                cDist = (int)(coreDistance * (double)this.verticalAdjustment);
            }

            int rDist;
            if (reachDistance == 2.147483647E9D) {
                rDist = this.getHeight();
            } else {
                rDist = (int)(reachDistance * (double)this.verticalAdjustment);
            }

            int x = vectorIndex + stepSize;
            if (this.isShowCoreDistances()) {
                g.setColor(this.coreDistanceColor);
                g.fillRect(x, this.getHeight() - cDist, this.widthSlider, cDist);
            }

            if (this.isShowReachabilityDistances()) {
                int sizer = this.widthSlider;
                if (!this.isShowCoreDistances()) {
                    sizer = 0;
                }

                g.setColor(this.reachabilityDistanceColor);
                g.fillRect(x + sizer, this.getHeight() - rDist, this.widthSlider, rDist);
            }

            if (this.isShowCoreDistances() && this.isShowReachabilityDistances()) {
                stepSize += this.widthSlider * 2;
            } else {
                stepSize += this.widthSlider;
            }
        }

    }

    public void setResultVector(ArrayList resultVector) {
        this.resultVector = resultVector;
    }

    public void setNewToolTip(String toolTip) {
        this.setToolTipText(toolTip);
    }

    public void adjustSize(SERObject serObject) {
        int i = 0;
        if (this.isShowCoreDistances() && this.isShowReachabilityDistances()) {
            i = 10;
        } else if (this.isShowCoreDistances() && !this.isShowReachabilityDistances() || !this.isShowCoreDistances() && this.isShowReachabilityDistances()) {
            i = 5;
        }

        this.setSize(new Dimension(i * serObject.getDatabaseSize() + serObject.getDatabaseSize(), this.getHeight()));
        this.setPreferredSize(new Dimension(i * serObject.getDatabaseSize() + serObject.getDatabaseSize(), this.getHeight()));
    }

    public boolean isShowCoreDistances() {
        return this.showCoreDistances;
    }

    public void setShowCoreDistances(boolean showCoreDistances) {
        this.showCoreDistances = showCoreDistances;
    }

    public boolean isShowReachabilityDistances() {
        return this.showReachabilityDistances;
    }

    public void setShowReachabilityDistances(boolean showReachabilityDistances) {
        this.showReachabilityDistances = showReachabilityDistances;
    }

    public void setVerticalAdjustment(int verticalAdjustment) {
        this.verticalAdjustment = verticalAdjustment;
    }

    public void setCoreDistanceColor(Color coreDistanceColor) {
        this.coreDistanceColor = coreDistanceColor;
        this.repaint();
    }

    public void setReachabilityDistanceColor(Color reachabilityDistanceColor) {
        this.reachabilityDistanceColor = reachabilityDistanceColor;
        this.repaint();
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10838 $");
    }

    private class MouseHandler extends MouseMotionAdapter implements RevisionHandler {
        private MouseHandler() {
        }

        public void mouseMoved(MouseEvent e) {
            this.showToolTip(e.getX());
        }

        private void showToolTip(int x) {
            int i = 0;
            if (GraphPanel.this.isShowCoreDistances() && GraphPanel.this.isShowReachabilityDistances()) {
                i = 11;
            } else if (GraphPanel.this.isShowCoreDistances() && !GraphPanel.this.isShowReachabilityDistances() || !GraphPanel.this.isShowCoreDistances() && GraphPanel.this.isShowReachabilityDistances() || !GraphPanel.this.isShowCoreDistances() && !GraphPanel.this.isShowReachabilityDistances()) {
                i = 6;
            }

            if (x / i == GraphPanel.this.recentIndex) {

            } else {
                GraphPanel.this.recentIndex = x / i;
                DataObject dataObject = null;

                try {
                    dataObject = (DataObject) GraphPanel.this.resultVector.get(GraphPanel.this.recentIndex);
                } catch (Exception ignored) {
                }

                if (dataObject != null) {
                    if (!GraphPanel.this.isShowCoreDistances() && !GraphPanel.this.isShowReachabilityDistances()) {
                        GraphPanel.this.setNewToolTip("<html><body><b>Please select a distance</b></body></html>");
                    } else {
                        GraphPanel.this.setNewToolTip("<html><body><table><tr><td>DataObject:</td><td>" + dataObject + "</td></tr>" + "<tr><td>Key:</td><td>" + dataObject.getKey() + "</td></tr>" + "<tr><td>" + (GraphPanel.this.isShowCoreDistances() ? "<b>" : "") + "Core-Distance:" + (GraphPanel.this.isShowCoreDistances() ? "</b>" : "") + "</td><td>" + (GraphPanel.this.isShowCoreDistances() ? "<b>" : "") + (dataObject.getCoreDistance() == 2.147483647E9D ? "UNDEFINED" : Utils.doubleToString(dataObject.getCoreDistance(), 3, 5)) + (GraphPanel.this.isShowCoreDistances() ? "</b>" : "") + "</td></tr>" + "<tr><td>" + (GraphPanel.this.isShowReachabilityDistances() ? "<b>" : "") + "Reachability-Distance:" + (GraphPanel.this.isShowReachabilityDistances() ? "</b>" : "") + "</td><td>" + (GraphPanel.this.isShowReachabilityDistances() ? "<b>" : "") + (dataObject.getReachabilityDistance() == 2.147483647E9D ? "UNDEFINED" : Utils.doubleToString(dataObject.getReachabilityDistance(), 3, 5)) + (GraphPanel.this.isShowReachabilityDistances() ? "</b>" : "") + "</td></tr>" + "</table></body></html>");
                    }
                }

            }
        }

        public String getRevision() {
            return RevisionUtils.extract("$Revision: 10838 $");
        }
    }
}
