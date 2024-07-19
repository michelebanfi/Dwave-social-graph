#import "@preview/slydst:0.1.1": *

#show: slides.with(
  title: "Structural Imbalance usign DWave",
  subtitle: "Worlds nations trade data",
  date: none,
  authors: ("Michele Banfi 869294"),
  layout: "medium",
  ratio: 4/3,
  title-color: none,
)

== Outline
#outline()

= Dataset

== National Dyadic Trade
This dataset tracks total national trade and bilateral trade flows between states from 1870-2014. In order to create a graph the change in percentage between 2013 and 2014 was calculated and used as edges between the nations; the nations became the nodes of the graph.
//$
//x_t =(x_t - x_(t-1)) dot 100
//$

#figure(
  image("Media/original_graph.png", width: 60%),
  caption: [Visual representation of the graph. On the left the positive edges and on the right the negative edges],
)

The graph was trimmed to reduce the dimensionality of the problem. Only 55 nodes were preserved

#figure(
  image("Media/prunedGraph.png", width: 80%),
  caption: [Trimmed graph],
)

= Problem Formulation

== Constraints

To calculate the structural imbalance of the network the constraint and objective to minimize are define.

*Constraints:*
$
sum_(c=1)^k x_(i,c) = 1 quad forall i in V
$
Each node should assigned to only one cluster. Where:
- $x_(i, j)$ is a binary variable that is 1 if node is assigned to cluster $c$, and 0 otherwise
- $k$ is the total number of clusters
- $V$ is the set of all nodes in the graph
\
\
\

== Objective function

*Objective function:*
Two objective functions are defined. One to maximize Intra-cluster similarity and the other to minimize Inter-cluster similarity.

*Intra-cluster term:*
For nodes $u$ and $v$ in the same cluster $i$ the intra-cluster similarity is maximized
$
sum_((u,v) in E) sum_(c=1)^k w_(u v) dot x_(u, i) dot x_(v, i)
$
*Inter-cluster term:*
For nodes $u$ and $v$ in the same clusters $i$ and $j$, the inter-cluster similarity is minimized
$
- sum_(u, v in E) sum_(i = 1)^k sum_(j=i+1)^k w_(u v) dot ( x_(u, i) dot x_(v, j) + x_(u, j) dot (x_v, i))
$
\
\
Putting them together:
$
max sum_((u,v) in E) sum_(c=1)^k w_(u v) dot x_(u, i) dot x_(v, i) - sum_(u, v in E) sum_(i = 1)^k sum_(j=i+1)^k w_(u v) dot ( x_(u, i) dot x_(v, j) + x_(u, j) dot (x_v, i))
$
The first term increases the objective for nodes in the same cluster (intra-cluster), and the second term decreases it for nodes in different clusters (inter-cluster).

= Results

== Cluster result
Here is presented the best clustering solution having energy equal to $-337.45454171584834$

#figure(
  image("Media/world_map.png", width: 70%),
  caption: [Best clustering solution],
)

Other 125 solutions were sampled, here is the distribution of them
#figure(
  image("Media/energy_distribution.png", width: 100%),
  caption: [Best clustering solution],
)