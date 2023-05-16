package jclec.intarray;

import jclec.IDistance;
import jclec.IIndividual;

public class HammingDistance implements IDistance {
    @Override
    public double distance(IIndividual one, IIndividual other) {
        // Individual genotypes
        int [] gone   = ((IntArrayIndividual) one).getGenotype();
        int [] gother = ((IntArrayIndividual) other).getGenotype();

        int distance = 0;
        for (int i = 0; i < gone.length; ++i) {
            if (gone[i] != gother[i])
                distance++;
        }

        return distance;
    }
}
