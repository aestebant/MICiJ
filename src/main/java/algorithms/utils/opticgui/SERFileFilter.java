package algorithms.utils.opticgui;

import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

import javax.swing.filechooser.FileFilter;
import java.io.File;

public class SERFileFilter extends FileFilter implements RevisionHandler {
    private String extension;
    private String description;

    public SERFileFilter(String extension, String description) {
        this.extension = extension;
        this.description = description;
    }

    public boolean accept(File f) {
        if (f != null) {
            if (f.isDirectory()) {
                return true;
            }

            String filename = f.getName();
            int i = filename.lastIndexOf(46);
            if (i > 0 && i < filename.length() - 1) {
                this.extension = filename.substring(i + 1).toLowerCase();
            }

            if (this.extension.equals("ser")) {
                return true;
            }
        }

        return false;
    }

    public String getDescription() {
        return this.description;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8108 $");
    }
}

