package miclustering.algorithms.evolutionary.mievocluster;

import jclec.IIndividual;
import jclec.intarray.IntArrayIndividual;
import jclec.intarray.mut.IntArrayMutator;
import miclustering.algorithms.evolutionary.utils.ReorderIndividual;

import java.util.Arrays;
import java.util.IntSummaryStatistics;

public class UnguidedRemoveAndReclasifyMutator extends IntArrayMutator {

    private int minK;
    private int maxK;
    private double probG;
    private double probL;

    public UnguidedRemoveAndReclasifyMutator(double probG, double probL) {
        this.probG = probG;
        this.probL = probL;
    }

    @Override
    protected IIndividual mutateInd(IIndividual individual) {
        IntArrayIndividual mutant = (IntArrayIndividual) individual;
        int[] gnome = mutant.getGenotype();
        IntSummaryStatistics stat = Arrays.stream(gnome).summaryStatistics();
        int currentK = stat.getMax() + 1;

        // Creates mutant genotype
        int[] mgenome = Arrays.copyOf(gnome, gnome.length);

        do {
            for (int i = 0; i < currentK; ++i) {
                if (randgen.uniform(0, 1) < probG) {
                    for (int j = 0; j < gnome.length; ++j) {
                        if (gnome[j] == i && randgen.uniform(0, 1) <= probL) {
                            mgenome[j] = randgen.choose(0, maxK);
                        }
                    }
                }
            }
            ReorderIndividual.reorder(mgenome);
            IntSummaryStatistics stat2 = Arrays.stream(mgenome).summaryStatistics();
            currentK = stat2.getMax() + 1;
        } while (currentK < minK);
        return species.createIndividual(mgenome);
    }
}
