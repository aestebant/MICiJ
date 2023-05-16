package miclustering.algorithms.evolutionary.mievocluster;

import jclec.IIndividual;
import jclec.intarray.IntArrayIndividual;
import jclec.intarray.rec.IntArrayRecombinator;
import miclustering.algorithms.evolutionary.utils.ReorderIndividual;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class UnguidedCrossover extends IntArrayRecombinator {

    private int minK;
    private double probG;
    private double probL;

    public UnguidedCrossover(double probG, double probL) {
        this.probG = probG;
        this.probL = probL;
    }
    @Override
    protected List<IIndividual> recombineInd(List<IIndividual> individuals) {
        // Parents genotypes
        int[] p0_genome = ((IntArrayIndividual) individuals.get(0)).getGenotype();
        int[] p1_genome = ((IntArrayIndividual) individuals.get(1)).getGenotype();
        // Do crossover
        int[] s0_genome = crossover(p0_genome, p1_genome);
        int[] s1_genome = crossover(p1_genome, p0_genome);
        // Put sons in s
        List<IIndividual> sons = new ArrayList<>(2);
        sons.add(species.createIndividual(s0_genome));
        sons.add(species.createIndividual(s1_genome));
        return sons;
    }

    private int[] crossover(int[] primaryParent, int[] secondaryParent) {
        int[] offspring = new int[primaryParent.length];

        IntSummaryStatistics stat0 = Arrays.stream(primaryParent).summaryStatistics();
        IntSummaryStatistics stat1 = Arrays.stream(secondaryParent).summaryStatistics();
        int k0 = stat0.getMax();
        int k1 = stat1.getMax();

        List<Integer> shuffleClusters = IntStream.range(0, k1).boxed().collect(Collectors.toList());
        Collections.shuffle(shuffleClusters, new Random(randgen.choose(50)));
        int currentK;
        do {
            for (int i = 0; i <= Integer.min(k0, k1); ++i) {
                if (randgen.uniform(0, 1) < probG) {
                    int nReplacements = 0;
                    for (int j = 0; j < primaryParent.length; ++j) {
                        if (primaryParent[j] == i && randgen.uniform(0, 1) <= probL) {
                            offspring[j] = randgen.choose(0, Integer.max(k0, k1) + 1);
                            nReplacements++;
                        } else {
                            offspring[j] = primaryParent[j];
                        }
                    }
                    List<Integer> replacements = new ArrayList<>(nReplacements);
                    for (int j = 0; j < secondaryParent.length; ++j) {
                        if (secondaryParent[j] == shuffleClusters.get(i)) {
                            replacements.add(j);
                        }
                    }
                    Collections.shuffle(replacements, new Random(randgen.choose(50)));
                    for (int j = 0; j < Integer.min(nReplacements, replacements.size()); ++j) {
                        offspring[replacements.get(j)] = shuffleClusters.get(i);
                    }
                }
            }
            ReorderIndividual.reorder(offspring);
            IntSummaryStatistics stat2 = Arrays.stream(offspring).summaryStatistics();
            currentK = stat2.getMax() + 1;
        } while (currentK < minK);
        return offspring;
    }

    @Override
    protected void recombineNext() {
        List<IIndividual> parents = new ArrayList<>(2);
        parents.add(parentsBuffer.get(parentsCounter));
        parents.add(parentsBuffer.get(parentsCounter + 1));
        sonsBuffer.addAll(recombineInd(parents));
    }
}
