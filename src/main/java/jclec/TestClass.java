package jclec;

import jclec.intarray.HammingDistance;
import jclec.intarray.IntArrayCreator;
import jclec.intarray.IntArrayIndividualSpecies;
import jclec.util.random.IRandGen;
import jclec.util.random.RanecuFactory;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.XMLConfiguration;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.OptionalDouble;

public class TestClass {

    private static ISpecies species;
    private static IRandGen randgen;

    public static void main(String[] args) {


        Configuration configuration = null;
        try {
            configuration = new XMLConfiguration("configuration/Pru.cfg");
        } catch (ConfigurationException e) {
            e.printStackTrace();
        }
        assert configuration != null;
        configuration = configuration.subset("process.species");

        RanecuFactory randGenFactory = new RanecuFactory();
        randGenFactory.setSeed(1856484);
        randgen = randGenFactory.createRandGen();
        species = new IntArrayIndividualSpecies();

        ((IntArrayIndividualSpecies) species).configure(configuration);
        int populationSize = 100;
        //IDistance distance = new RSDistance();
        //IDistance distance = new RSHammingDistance();
        IDistance distance = new HammingDistance();

        IPopulation context = new IPopulation() {
            @Override
            public ISpecies getSpecies() {
                return species;
            }

            @Override
            public IEvaluator getEvaluator() {
                return null;
            }

            @Override
            public int getGeneration() {
                return 0;
            }

            @Override
            public List<IIndividual> getInhabitants() {
                return null;
            }

            @Override
            public IRandGen createRandGen() {
                return randgen;
            }
        };
        IProvider provider = new IntArrayCreator();
        provider.contextualize(context);

        List<IIndividual> set = provider.provide(populationSize);
        shuffle(set);
        List<Double> distances = new ArrayList<>(populationSize /2);
        for (int i = 0; i < populationSize - 1; i += 2) {
            // Couple of individuals
            IIndividual ind1 = set.get(i);
            IIndividual ind2 = set.get(i + 1);
            double d = distance.distance(ind1, ind2);
            distances.add(d);
            System.out.println(d);
        }
        OptionalDouble average = distances.stream().mapToDouble(a -> a).average();

        System.out.println("Media de distancias en la población: " + average.getAsDouble());

        System.out.println(distance.getClass().getName());
    }

    /**
     * Based on shuffle implemented in java.util.Collections
     */
    private static void shuffle(List<IIndividual> list) {
        int size = list.size();
        IIndividual[] arr = list.toArray(new IIndividual[size]);
        // Shuffle array
        for (int i = size; i > 1; i--) {
            swap(arr, i - 1, randgen.choose(i));
        }
        // Dump array back into list
        ListIterator<IIndividual> it = list.listIterator();
        for (IIndividual anArr : arr) {
            it.next();
            it.set(anArr);
        }
    }

    /**
     * Swaps the two specified elements in the specified array.
     */
    private static void swap(IIndividual[] arr, int i, int j) {
        IIndividual tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
