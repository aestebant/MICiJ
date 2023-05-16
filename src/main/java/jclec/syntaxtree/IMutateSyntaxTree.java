package jclec.syntaxtree;

import jclec.util.random.IRandGen;

import java.io.Serializable;

/**
 * A single mutation operator for expression tree individuals
 * 
 * @author Sebastian Ventura
 */

public interface IMutateSyntaxTree extends Serializable
{
	/**
	 * 
	 * @param tree
	 * @param treeSchema
	 * @param randgen
	 * 
	 * @return A new expression tree
	 */
	
	public abstract SyntaxTree mutateSyntaxTree(SyntaxTree tree, SyntaxTreeSchema treeSchema, IRandGen randgen);
}
