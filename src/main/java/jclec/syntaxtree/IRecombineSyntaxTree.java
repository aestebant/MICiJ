package jclec.syntaxtree;

import jclec.util.random.IRandGen;

import java.io.Serializable;

/**
 * Recombine an expression tree
 * 
 * @author Sebastian Ventura
 */

public interface IRecombineSyntaxTree extends Serializable
{
	/**
	 * Recombination at expression tree level.
	 * 
	 * @param ptree0 First parent expression tree 
	 * @param ptree1 Second parent expression tree
	 * @param stree0 First son expression tree
	 * @param stree1 Second son expression tree
	 * @param schema Expression trees schema 
	 * @param randgen Random generator used in operation
	 */
	
	public abstract void recombine(SyntaxTree ptree0, SyntaxTree ptree1, SyntaxTree stree0, SyntaxTree stree1, SyntaxTreeSchema schema, IRandGen randgen); 
}
