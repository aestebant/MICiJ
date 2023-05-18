# MICiJ: Multi-Instance Clustering in Java

MICiJ is a project of library for unsupervised machine learning with flexible representation of information. Specifically, it contains several adaptations of popular clustering algorithms to Multi-Instance Learning (MIL). MICiJ contains both iterative and evolutionary clustering algorithm, based in two popular machine learning libraries, [Weka](https://www.cs.waikato.ac.nz/ml/weka/) and [JCLEC](https://jclec.sourceforge.net) respectively.

Currently, the implemented methods in MICiJ are the following:

**Iterative algorithms**

* **BAMIC**: M. L. Zhang and Z. H. Zhou, “Multi-instance clustering with applications to multi-instance prediction,” *Applied Intelligence*, vol. 31, no. 1, pp. 47–68, 2009.
* **MIKMeans**: A. Esteban, A. Zafra, and S. Ventura, “A Preliminary Study on Evolutionary Clustering for Multiple Instance Learning,” in *Proceedings of the 2020 IEEE Congress on Evolutionary Computation*, 2020, pp. 1–8.
* **OneStepKMeans**: an adaptation of MIKMeans that only performs one iteration of the algorithm.
* **MIDBSCAN**: an adaptation to MIL of E. Martin, H.-P. Kriegel, J. Sander, and X. Xu, “A density-based algorithm for discovering clusters in large spatial databases with noise.,” in *Proceedings of 2nd International Conference on Knowledge Discovery and Data Mining*, 1996, pp. 226–231.
* **MIOPTICS**: an adaptation to MIL of M. Ankerst, M. M. Breunig, H. Kriegel, and J. Sander, “OPTICS: Ordering Points To Identify the Clustering Structure,” in *Proceedings of the 1999 ACM SIGMOD international conference on Management of data*, 1999, pp. 49–60.

**Evolutionary algorithms**

* **MIGKA**: A. Esteban, A. Zafra, and S. Ventura, “A Preliminary Study on Evolutionary Clustering for Multiple Instance Learning,” in *Proceedings of the 2020 IEEE Congress on Evolutionary Computation*, 2020, pp. 1–8.
* **MIFGKA**: A. Esteban, A. Zafra, and S. Ventura, “A Preliminary Study on Evolutionary Clustering for Multiple Instance Learning,” in *Proceedings of the 2020 IEEE Congress on Evolutionary Computation*, 2020, pp. 1–8.
* **MIGCUK**: A. Esteban, A. Zafra, and S. Ventura, “A Preliminary Study on Evolutionary Clustering for Multiple Instance Learning,” in *Proceedings of the 2020 IEEE Congress on Evolutionary Computation*, 2020, pp. 1–8.
* **CHCMIC**: A. Esteban, A. Zafra, and S. Ventura, “A Preliminary Study on Evolutionary Clustering for Multiple Instance Learning,” in *Proceedings of the 2020 IEEE Congress on Evolutionary Computation*, 2020, pp. 1–8.
* **MIEvoCluster** : an adaptation to MIL of P. C. H. Ma, K. C. C. Chan, X. Yao, and D. K. Y. Chiu, “An evolutionary clustering algorithm for gene expression microarray data analysis,” IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 296–314, 2006.


## Instalation

Currently, the only available way of working with MICiJ is installing the source code and compiling the JAR locally. For that, just clone the repository and open in with any Java IDE. The dependencies needed are:

| Program      | Version |
|--------------|---------|
| JDK          | \>= 1.8 |
| Apache Maven | \>= 3.8 |

The rest of the dependencies specific to the library are managed with Maven through the [pom.xml](https://github.com/aestebant/mi-clustering/blob/master/pom.xml) file.


## Quick start

The class `miclustering.Run` contains all the necessary code for running a single or multiple experiments of the iterative algorithms. First, the _name_ of the datasets included in the experiment are specified in the `datasets` structure, assuming that the file containing the dataset is in the path *datasets/\<name\>.arff* in the project folder. By default, 32 classic datasets are listed, but only the uncommented are included in the experiment.
```
String[] datasets = {
    "musk1",
    // "musk2",
    ...
    // "BiocreativeFunction",
    // "BiocreativeProcess",
};
```
Additionally, the specified datasets can be modified by a standardization method with the structure `standardization`. The standardized version must exist previously under the path *datasets/\<name\>\<standardized extension\>.arff*.
```
String[] standardization = {
    // "",
    // "-z4",
    "-z5"
};
```
Finally, the algorithm or algorithms to use and their configurations are specified in the following structures. The configurations are specified as Strings following the format of each algorithm. See each algorithm for more information about the available parameters.
```
String[] clustering = {
    // "MIDBSCAN",
    // "MIOPTICS"
    // "MIKMeans",
    "BAMIC",
};

Map<String, String> options = new HashMap<>();
options.put("MIDBSCAN", "-E 0.8 -A HausdorffDistance -hausdorff-type 0");
options.put("MIOPTICS", "-E 10 -A HausdorffDistance -hausdorff-type 0");
options.put("MIKMeans", "-N 2 -A HausdorffDistance -hausdorff-type 0");
options.put("BAMIC", "-N 2 -A HausdorffDistance -hausdorff-type 0");
```

After configuring the experiment and running the program, the output will include a header with the experiment information and the evaluation result including internal metrics (related to the properties of the formed clusters) and external metrics (related to the relationship of the formed clusters with the actual classes if they are known). For example:
```
=========================================
Algorithm: BAMIC
Distance: Maximal Minimal Hausdorff Distance
Dataset: musk1
Standarization: -z5
=========================================
Evaluation
----------------
Clustered Instances:
0      63 ( 68%)
1      29 ( 32%)
Class attribute: "class"
Confusion Matrix:
  0  1 <- real classes
-----------------------
 32 31 | predicted cluster: 0
 13 16 | predicted cluster: 1
Assignation:
	Cluster 0 <-- 0
	Cluster 1 <-- 1
Incorrectly clustered instances :	44 + [0, 0]	(47.82608695652174 %)
Internal validation metrics:
	RMSSTD: 0.22648817375261854
	Silhouette index: 0.16263170494882995
	XB* index: 0.03885383953857572
	DB* index: 2.5772094610868033
	S_Dbw index: 0.8640387632525972
	DBCV: -0.6868138899787306
External validation metrics:
	Entropy: 0.9974378487105109
	Purity: 0.5217391304347826
	Rand index: 0.5217391304347826
	Precision: [0.5079365079365079, 0.5517241379310345]	Macro: 0.5517241379310345
	Recall: [0.7111111111111111, 0.3404255319148936]	Macro: 0.3404255319148936
	F1 measure: [0.5925925925925924, 0.4210526315789473]	Macro: 0.4210526315789473
	Specificity: [0.3404255319148936, 0.7111111111111111]	Macro: 0.7111111111111111
```


## Data format

MICiJ works with multi-instance datasets in *ARFF* format in its multi-instance variant, that contains three attributes in the first level:
* **bag-id**: nominal attribute; unique identifier for each bag.
* **bag**: relational attribute; contains the instances of a sample.
  * The data of the relational attribute is surrounded by quotes and the single instances inside the bag are separated by `\n`
* **class**: the class label associated to the sample.

For example:
```
@relation musk1

 @attribute molecule_name {MUSK-jf78,MUSK-jf67,MUSK-jf59,...,NON-MUSK-199}
 @attribute bag relational
   @attribute f1 numeric
   @attribute f2 numeric
   @attribute f3 numeric
   ...
   @attribute f166 numeric
 @end bag
 @attribute class {0,1}

 @data
 MUSK-188,"42,-198,-109,-75,-117,11,23,-88,-28,-27,...,48,-37,6,30\n42,-191,-142,-65,-117,55,49,-170,-45,5,...,48,-37,5,30\n...",1
 ...
```
