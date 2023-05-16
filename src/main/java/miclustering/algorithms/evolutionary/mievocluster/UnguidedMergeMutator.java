package miclustering.algorithms.evolutionary.mievocluster;

import jclec.IIndividual;
import jclec.intarray.IntArrayIndividual;
import jclec.intarray.mut.IntArrayMutator;
import miclustering.algorithms.evolutionary.utils.ReorderIndividual;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class UnguidedMergeMutator extends IntArrayMutator {
    private int minK;
    private int maxK;
    private double probG;

    @Override
    protected IIndividual mutateInd(IIndividual individual) {
        IntArrayIndividual mutant = (IntArrayIndividual) individual;
        int[] gnome = mutant.getGenotype();
        IntSummaryStatistics stat = Arrays.stream(gnome).summaryStatistics();
        int currentK = stat.getMax() + 1;

        if (currentK == minK && minK == maxK)
            return species.createIndividual(gnome);

        List<Integer> shuffleClusters = IntStream.range(0, currentK).boxed().collect(Collectors.toList());
        Collections.shuffle(shuffleClusters, new Random(randgen.choose(50)));

        // Creates mutant genotype
        int[] mgenome = Arrays.copyOf(gnome, gnome.length);

        for (int i = 0; i < shuffleClusters.size() - 1; ++i) {
            if (randgen.uniform(0, 1) < probG) {
                int newK = shuffleClusters.get(i + 1);
                for (int j = 0; j < gnome.length; ++j) {
                    if (gnome[j] == shuffleClusters.get(i)) {
                        mgenome[j] = newK;
                    }
                }
                i++;
                currentK--;
                if (currentK == minK)
                    break;
            }
        }
        ReorderIndividual.reorder(mgenome);
        return species.createIndividual(mgenome);
    }
}
