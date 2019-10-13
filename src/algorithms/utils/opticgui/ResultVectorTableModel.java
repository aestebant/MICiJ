package algorithms.utils.opticgui;

import algorithms.utils.DataObject;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

import javax.swing.table.AbstractTableModel;
import java.util.ArrayList;

public class ResultVectorTableModel extends AbstractTableModel implements RevisionHandler {
    private static final long serialVersionUID = -7732711470435549210L;
    private ArrayList resultVector;

    public ResultVectorTableModel(ArrayList resultVector) {
        this.resultVector = resultVector;
    }

    public int getRowCount() {
        return this.resultVector == null ? 0 : this.resultVector.size();
    }

    public int getColumnCount() {
        return this.resultVector == null ? 0 : 4;
    }

    public Object getValueAt(int row, int column) {
        DataObject dataObject = (DataObject)this.resultVector.get(row);
        switch(column) {
        case 0:
            return dataObject.getKey();
        case 1:
            return dataObject;
        case 2:
            return dataObject.getCoreDistance() == 2.147483647E9D ? "UNDEFINED" : Utils.doubleToString(dataObject.getCoreDistance(), 3, 5);
        case 3:
            return dataObject.getReachabilityDistance() == 2.147483647E9D ? "UNDEFINED" : Utils.doubleToString(dataObject.getReachabilityDistance(), 3, 5);
        default:
            return "";
        }
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10838 $");
    }
}
