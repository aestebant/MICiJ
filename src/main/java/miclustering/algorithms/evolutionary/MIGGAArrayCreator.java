package miclustering.algorithms.evolutionary;

import jclec.intarray.IntArrayCreator;

public class MIGGAArrayCreator extends IntArrayCreator {
    @Override
    protected int[] createGenotype() {
        int gl = schema.length;
        int[] result = new int[gl];
        result[gl-1] = schema[gl-1].getRandom(randgen);
        for (int i = 0; i < gl-1; ++i)
            result[i] = randgen.choose(0, result[gl-1]);
        return result;
    }
}
