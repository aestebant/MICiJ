package miclustering.algorithms.evolutionary.utils;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ReorderIndividual {
    public static void reorder(int[] ind) {
        Map<Integer, Integer> count = new HashMap<>(10);
        for (int gen : ind) {
            if (!count.containsKey(gen)) {
                count.put(gen, 0);
            }
            count.put(gen, count.get(gen) + 1);
        }
        Stream<Map.Entry<Integer, Integer>> sorted = count.entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue()));
        List<Map.Entry<Integer, Integer>> collect = sorted.collect(Collectors.toList());
        List<Integer> sortedLabels = new ArrayList<>(collect.size());
        for (Map.Entry<Integer, Integer> entry : collect)
            sortedLabels.add(entry.getKey());
        for (int i = 0; i < ind.length; ++i) {
            ind[i] = sortedLabels.indexOf(ind[i]);
        }
    }
}
