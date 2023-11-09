package jclec.selector;

import jclec.IIndividual;
import jclec.ISystem;
import jclec.fitness.IValueFitness;

import java.util.Random;

/*
 * Nombre: UniversalStocasticoSelector
 * Autor: Rafael AyllA?n Iglesias
 * Tipo: Clase publica
 * Extiende: La clase StochasticSelector y la interfaz IIndividual
 * Implementa: Nada
 * Variables de la clase: serialVersionUID (generado por eclipse)
 *  					 - TotalFitness es un double que almacena el fitness total de la
 * 						 poblacion
 * 						 - i es un entero usado de contador para saber cuantos individuos
 * 						 han sido analizados
 *    					 - restantes es un entero usado para controlar en cual de las dos
 *    					 partes del metodo estamos con valor 1 para buscar individuos enteros
 * 						 - numSeleccionados es un entero que almacena el numero de individuos
 * 						 seleccionados hasta el momento
 * 						 - suma es un parametro del algoritmo que almacena la suma de las
 * 						 aptitudes respecto al resto de la poblacion de los individuos
 * 						 que son seleccionados
 * 						 - aleatorio es un parametro aleatotio e incremental del algoritmo
 * 						 que se usa para tomar la decision de seleccion del individuo
 * Metodos: Protegidos: prepareSelection
 * 						selectNext
 * 			Publicos: Ninguno
 * Objetivo de la clase: Esta clase pretende implementar el selector por el metodo del
 * 						 universal estocastico, con lo cual en esta clase en su metodo
 *                       prepareSelection se inicianilaran los datos que posteriormente
 *                       el metodo selectNext utilizara para poder seleccionar los
 *                       individuos de la poblacion
 *
 */

/**
 * Stochastic universal sampling selector.
 *
 * @author Rafael Ayllon-Iglesias
 * @author Sebastian Ventura
 */

public class StochasticUniversalSampling extends StochasticSelector {

    /////////////////////////////////////////////////////////////////
    // --------------------------------------- Serialization constant
    /////////////////////////////////////////////////////////////////

    /**
     * Generated by Eclipse
     */

    private static final long serialVersionUID = -7486679623259737868L;

    /**
     * Fitness total de la poblacion
     */

    protected transient double TotalFitness = 0.0;

    /**
     * Variable de recorrido de la poblacion
     */

    protected transient int i = 0;

    /**
     * Variable de control sobre parte del algoritmo en el que te encuentras
     */

    int restantes = 1;

    /**
     * Es la suma de los ei
     */

    double suma = 0.0;

    /**
     * Es el aleatorio del algoritmo ptr
     */

    double aleatorio;

    /**
     * Variable que mantiene el numero de elemenetos seleccionados
     */

    int numSeleccionados = 0;

    /////////////////////////////////////////////////////////////////
    // ------------------------------------------------- Constructors
    /////////////////////////////////////////////////////////////////

    /**
     * Empty constructor
     */

    public StochasticUniversalSampling() {
        super();
        Random rnd = new Random();
        aleatorio = rnd.nextDouble();
    }

    /**
     * Constructor that contextualizes selector
     *
     * @param context Execution context
     */

    public StochasticUniversalSampling(ISystem context) {
        super(context);
        Random rnd = new Random();
        aleatorio = rnd.nextDouble();
    }

    /////////////////////////////////////////////////////////////////
    // ------------------------- Overwriting AbstractSelector methods
    /////////////////////////////////////////////////////////////////

    /**
     * Nombre: prepareSelection
     * Autor: Rafael AyllA?n Iglesias.
     * Tipo funcion: Protegida
     * Valores de entrada: Ninguno
     * Valores de salida: Ninguno
     * Funciones que utiliza: Ninguna
     * Variables: Ninguna
     * Objetivo: Preparar las variables para que la funcion selectNext pueda realizar
     * la seleccion de los individuos de la poblacion
     */

    @Override
    protected void prepareSelection() {
        //Inicializo datos en cada generacion
        numSeleccionados = 0;
        //Obtengo el fitnes total de la poblacion
        for (IIndividual ind : actsrc) {
            double FitnessActual = ((IValueFitness) ind.getFitness()).getValue();
            TotalFitness += FitnessActual;
        }
    }

    /**
     * Nombre: selectNext
     * Autor: Rafael AyllA?n Iglesias.
     * Tipo funcion: Protegida
     * Valores de entrada: Ninguno
     * Valores de salida: El individuo seleccionado
     * Funciones que utiliza: Ninguna
     * Variables:- conteo es ena variable que almacena ei del algoritmo
     * - ind es el individuo actualmente analizado de la poblacion
     * - FitnessActual es un double que almacena el fitnes del individuo
     * actualmente analizado de la poblacion
     * Objetivo: Esta funcion realiza le eleccion de cual es el siguiente individuo
     * a seleccionar de la poblacion
     */

    @Override
    protected IIndividual selectNext() {
        //Selects individual
        double conteo; //El dato ei
        //repito el algoritmo mientras no tengo todos seleccionados
        while (numSeleccionados < actsrcsz) {
			do {
				for (int j = i; j < actsrcsz; j++) {
					//empiezo desde el ultimo elemento analizado en la pasada anterior
					IIndividual ind = actsrc.get(j);
					double FitnessActual = ((IValueFitness) ind.getFitness()).getValue();
					//Calculo su valor con respecto a la poblacion
					conteo = (FitnessActual * actsrcsz) / TotalFitness;
					//compruebo si suma anterior sigue siendo mejor
					if (suma > aleatorio) {
						j--; //Vuelvo al dato anterior
					} else {
						suma = suma + conteo;    //Actualizo la suma
					}
					//Compruabo si he recorrido completamente la poblacion
					if (i == actsrcsz - 1)
						i = -1;//Reinicio el analisis
					if (suma > aleatorio) {
						i++;//Prepararo para comprobar siguiente elemento

						//Actualizo el valor del aleatorio
						aleatorio++;
						//Selecciono el elemento actual
						numSeleccionados++;
						return ind;
					}
					i++;
				}
			} while (suma < (aleatorio - 1));//Compruebo no se halla acabado el ciclo
		}
        //Este caso nunca se dara
        return actsrc.get(0);
    }
}