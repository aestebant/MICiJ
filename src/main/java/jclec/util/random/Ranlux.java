package jclec.util.random;

import org.apache.commons.lang.builder.EqualsBuilder;

/**
 * RandLux is an advanced  pseudo-random  number  generator based on the RCARRY
 * algorithm proposed in 1991 by Marsaglia and Zaman.</p> 
 * 
 * RCARRY used  a  subtract-and-borrow  algorithm with a period on the order of  
 * 10<SUP>171</SUP> but  still had  detectable  correlations  between  numbers. 
 * Martin Luescher  proposed the  "RANLUX" algorithm  in  1993; this  algorithm 
 * generates  pseudo-random  numbers using  RCARRY, but throws  away numbers to 
 * destroy correlations. Thus RANLUX  trades execution speed for quality.</p>
 * 
 * Choosing  a larger luxury setting one gets better random numbers slower. By 
 * the tests available  at the time it was  proposed, RANLUX at default luxury
 * setting appears to be a significant advance quality over previous generators.
 * 
 * <BR>
 * <BR>
 * <CENTER><TABLE BORDER WIDTH=80%>
 * <TR>
 * <TD ALIGN=center COLSPAN=3><A NAME="luxury"> <FONT SIZE=+2>LUXURY LEVELS
 * </FONT> </A></TD>
 * </TR>
 * <TR>
 * <TD>level</TD>
 * <TD ALIGN=center>p</TD>
 * <TD><BR>
 * </TD>
 * </TR>
 * <TR>
 * <TD ALIGN=center>0</TD>
 * <TD ALIGN="center">24</TD>
 * <TD>equivalent to the original <TT>RCARRY</TT> of Marsaglia and Zaman,
 * very long period, but fails many tests.</TD>
 * </TR>
 * <TR>
 * <TD ALIGN=center>1</TD>
 * <TD ALIGN=center>48</TD>
 * <TD>considerable improvement in quality over level 0, now passes the gap
 * test, but still fails spectral test.</TD>
 * </TR>
 * <TR>
 * <TD ALIGN=center>2</TD>
 * <TD ALIGN=center>97</TD>
 * <TD>passes all known tests, but theoretically still defective.</TD>
 * </TR>
 * <TR BGCOLOR="#FFA0A0">
 * <TD ALIGN=center BGCOLOR="#FFA0A0">3</TD>
 * <TD ALIGN=center>223</TD>
 * <TD>DEFAULT VALUE. Any theoretically possible correlations have very small
 * chance of being observed.</TD>
 * </TR>
 * <TR>
 * <TD ALIGN=center>4
 * <TD ALIGN=center>389</TD>
 * <TD>highest possible luxury, all 24 bits chaotic.</TD>
 * </TR>
 * 
 * </TABLE> </CENTER> <BR>
 * <CENTER><FONT SIZE=+1> <B>VALIDATION </B> </FONT> </CENTER>
 * 
 * The Java version of <TT>RANLUX</TT> has been verified against published
 * values of numbers 1-5 and 101-105 produced by the reference implementation of
 * <TT>RANLUX</TT> for the following initial conditions:
 * 
 * <UL>
 * <LI>Default initialization: <CODE>Ranlux()</CODE>
 * <LI>Initialization with: <CODE>Ranlux(0,0)</CODE>
 * <LI>Initialization with: <CODE>Ranlux(389,1)</CODE>
 * <LI>Initialization with: <CODE>Ranlux(75,0)</CODE>
 * </UL>
 * 
 * References:
 * <UL>
 * <LI>M. Luscher, <CITE>Computer Physics Communications </CITE> <B>79 </B>
 * (1994) 100
 * <LI>F. James, <CITE>Computer Physics Communications </CITE> <B>79 </B>
 * (1994) 111
 * <LI><A
 * HREF="http://www.mpa-garching.mpg.de/~tomek/htmls/refs/ranlux.about.html">About
 * <TT>RANLUX</TT> random number generator: Excerpts from discussion in the
 * Usenet news groups </A>
 * <LI><A
 * HREF="http://www.mpa-garching.mpg.de/~tomek/htmls/refs/ranlux.f90_2.html">Miller's
 * FORTRAN 90 implementation of <TT>RANLUX</TT> with test code </A>
 * 
 * </UL>
 * 
 * @author ... adapted by Sebastian Ventura 
 */

public class Ranlux extends AbstractRandGen 
{
	///////////////////////////////////////////////////////////////////
	// ----------------------------------------- Serialization constant
	///////////////////////////////////////////////////////////////////

	/** Generated by Eclipse */
	
	private static final long serialVersionUID = 7707398773259479291L;

	///////////////////////////////////////////////////////////////////
	// ------------------------------------------------------ Constants
	///////////////////////////////////////////////////////////////////

	static final int GIGA = 1000000000;
	static final int TWOP12 = 4096;
	static final int ITWO24 = 1 << 24;
	static final int ICONS = 2147483563;
	static final int [] ND_SKIP = {0, 24, 73, 199, 365};
	static final int [] NEXT = 
		{ 
			0, 24,  1,  2,  3,  
			4,  5,  6,  7,  8, 
			9, 10, 11, 12, 13, 
		   14, 15, 16, 17, 18, 
		   19, 20, 21, 22, 23
		};
	
	///////////////////////////////////////////////////////////////////
	// ----------------------------------------------------- Atributtes
	///////////////////////////////////////////////////////////////////
	
	/** Fixed in initialization method */
	
	int nskip;

	/** Fixed in initialization method */

	float twom24;

	/** Fixed in initialization method */
	
	float twom12;

	int i24;
	int j24;
	int in24;
	int kount;
	float carry;
	float [] seeds;

	///////////////////////////////////////////////////////////////////
	// --------------------------------------------------- Constructors
	///////////////////////////////////////////////////////////////////

	/**
	 * Empty constructor 
	 */
	
	protected Ranlux() 
	{
		super();
	}

	/**
	 * Default constructor. Used by the RandGen factory.
	 * 
	 * @param luxlev Luxury level
	 * @param seed   First seed
	 */
	
	public Ranlux(int luxlev, int seed) 
	{
		super();
		rluxgo(luxlev, seed);
	}

	/////////////////////////////////////////////////////////////////
	// ------------------------------ Implementing IRandGen interface
	/////////////////////////////////////////////////////////////////

	/**
	 * {@inheritDoc}
	 */
	
	public final double raw() 
	{
		int i;
		float uni, out;

		uni = seeds[j24] - seeds[i24] - carry;
		if (uni < (float) 0.0) {
			uni = uni + (float) 1.0;
			carry = twom24;
		} 
		else {
			carry = (float) 0.0;
		}

		seeds[i24] = uni;

		i24 = NEXT[i24];
		j24 = NEXT[j24];

		out = uni;

		if (uni < twom12) {
			out += twom24 * seeds[j24];
		}

		/* zero is forbidden in case user wants logarithms */

		if (out == 0.0) {
			out = twom24 * twom24;
		}

		in24++;

		if (in24 == 24) {
			in24 = 0;
			kount += nskip;
			for (i = 1; i <= nskip; i++) {
				uni = seeds[j24] - seeds[i24] - carry;
				if (uni < (float) 0.0) {
					uni = uni + (float) 1.0;
					carry = twom24;
				} 
				else {
					carry = (float) 0.0;
				}

				seeds[i24] = uni;

				i24 = NEXT[i24];
				j24 = NEXT[j24];
			}
		}

		kount++;
		if (kount >= GIGA) {
			kount -= GIGA;
		}
		return out;
	}
	
	////////////////////////////////////////////////////////////////////
	// ------------------------------ Overwrite java.lang.Object methods
	////////////////////////////////////////////////////////////////////
	
	@Override
	
	public boolean equals(Object other)
	{
		if (other instanceof Ranlux) {
			Ranlux o = (Ranlux) other;
			EqualsBuilder eb = new EqualsBuilder();
			eb.append(this.nskip, o.nskip);
			eb.append(this.twom24, o.twom24);
			eb.append(this.twom12, o.twom12);
			eb.append(this.i24, o.i24);
			eb.append(this.j24, o.j24);
			eb.append(this.in24, o.in24);
			eb.append(this.kount, o.kount);
			eb.append(this.carry, o.carry);
			eb.append(this.seeds, o.seeds);
			return eb.isEquals();
		}
		else {
			return false;
		}
	}

	/////////////////////////////////////////////////////////////////
	// ---------------------------------------------- Private methods
	/////////////////////////////////////////////////////////////////

	/**
	 * Initialization method.
	 * 
	 * @param lux Luxury level
	 * @param ins Seeds generator
	 */
	
	private final void rluxgo(int lux, int ins) 
	{
		// Check luxury level
		if (lux<0 || lux>4) {
			throw new IllegalArgumentException
				("Luxury level must be an integer in the range [0,4]");
		}
		// Init luxury level
		nskip = ND_SKIP[lux];
		// Init in24
		in24 = 0;
		// Init seeds		
		if (ins <= 0) {
			throw new IllegalArgumentException("ins must be a positive integer");
		}
		// Init twom24
		twom24 = (float) 1.0;
		// Prepare seeds initialization
		int [] iseeds = new int[24 + 1];
		int aux = ins;
		for (int i = 1; i <= 24; i++) {
			twom24 = twom24 * (float) 0.5;
			int k = aux / 53668;
			aux = 40014 * (aux - k * 53668) - k * 12211;
			if (aux < 0) {
				aux = aux + ICONS;
			}
			iseeds[i] = aux % ITWO24;
		};
		// Init twom12
		twom12 = twom24 * 4096;
		// Init seeds
		seeds = new float[24 + 1];
		for (int i = 1; i <= 24; i++) {
			seeds[i] = iseeds[i] * twom24;
		};
		// Init i24
		i24 = 24;
		// Init j24
		j24 = 10;
		// Init carry
		carry = (float) 0.0;
		if (seeds[24] == 0.0) {
			carry = twom24;
		}
		// Init kount
		kount = 0;
	}
}