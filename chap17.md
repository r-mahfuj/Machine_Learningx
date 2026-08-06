# Chapter 17

# Atomistic Line Graph Neural Network (ALIGNN)

---

# Learning Objectives

After studying this chapter, the reader will be able to

* Understand why conventional graph neural networks are insufficient for many crystalline materials.
* Explain the importance of bond-angle-dependent interactions in determining material properties.
* Construct both crystal graphs and line graphs from atomic structures.
* Derive the mathematical formulation of the Atomistic Line Graph Neural Network (ALIGNN).
* Understand edge-gated graph convolution and angular message passing.
* Implement ALIGNN using modern deep learning frameworks.
* Train ALIGNN on large materials databases such as JARVIS and the Materials Project.
* Evaluate ALIGNN for regression and classification tasks.
* Compare ALIGNN with CGCNN, SchNet, MEGNet, DimeNet, GemNet, and other atomistic graph neural networks.
* Understand the advantages, limitations, and future research directions of ALIGNN.

---

# 17.1 Introduction

Graph Neural Networks (GNNs) have fundamentally changed how machine learning models represent materials. Unlike traditional machine learning algorithms that rely on handcrafted descriptors, graph neural networks learn directly from the atomic structure itself. In this representation, atoms become nodes, chemical bonds become edges, and the crystal is naturally described as a graph.

This graph-based paradigm has proven remarkably successful because it mirrors the physical nature of condensed matter. A crystal is not merely a collection of atoms; it is a network of interacting atoms whose properties emerge from both local chemical environments and long-range structural organization.

The first major breakthrough in graph-based materials informatics was the **Crystal Graph Convolutional Neural Network (CGCNN)**, introduced by Xie and Grossman in 2018. CGCNN demonstrated that a graph neural network could accurately predict a wide variety of material properties directly from crystal structures without requiring manually engineered descriptors. This work established graph neural networks as one of the dominant approaches in computational materials science.

Subsequent models, including **SchNet** and **MEGNet**, further improved representation learning by introducing continuous-filter convolutions, learned edge representations, and more expressive message-passing mechanisms. These models significantly improved prediction accuracy across numerous benchmark datasets and broadened the applicability of graph neural networks to molecules, crystals, and atomistic simulations.

Despite these advances, an important limitation remained.

Almost all early graph neural networks represented atomic interactions primarily through **pairwise relationships**.

In other words, information flowed between pairs of neighboring atoms,

$$
A \longleftrightarrow B,
$$

while the geometric relationship among three or more atoms was only learned indirectly through multiple layers of message passing.

For many materials, this approximation is insufficient.

---

## 17.1.1 Geometry Governs Materials

One of the central ideas in materials science is that **structure determines properties**.

This statement appears throughout crystallography, solid-state physics, physical metallurgy, and quantum chemistry. However, the word *structure* encompasses much more than the positions of atoms.

A complete structural description includes

* atomic identities,
* bond lengths,
* bond angles,
* coordination numbers,
* local symmetry,
* long-range periodicity,
* lattice distortions.

Early graph neural networks effectively modeled the first two items but largely ignored the explicit representation of bond angles.

At first glance, this omission may seem insignificant. After all, if every bond length is known, shouldn't the geometry also be determined?

The answer is **no**.

Two materials can possess nearly identical bond lengths while exhibiting completely different bond angles, leading to dramatically different physical properties.

Understanding this distinction is one of the primary motivations behind ALIGNN.

---

## 17.1.2 Why Pairwise Information Is Not Enough

Consider three atoms

```text
A —— B —— C
```

A conventional graph neural network stores

* distance between A and B,
* distance between B and C.

However, another equally important quantity exists:

```text
      A
       \
        \
         B
        /
       /
      C
```

the bond angle

$$
\angle ABC.
$$

This angle determines the relative orientation of the two bonds.

A graph containing only pairwise distances does not explicitly encode this information.

Although sufficiently deep neural networks may eventually infer angular relationships indirectly, this process is inefficient and places unnecessary burden on the learning algorithm.

Instead of asking the neural network to discover geometric relationships from repeated pairwise interactions, ALIGNN introduces these relationships directly into the graph representation.

This idea dramatically improves learning efficiency.

---

## 17.1.3 A Materials Science Perspective

To appreciate why bond angles matter, consider several familiar materials.

### Diamond

Each carbon atom in diamond forms four covalent bonds arranged in a tetrahedral geometry.

The bond angle is approximately

$$
109.47^\circ.
$$

This tetrahedral arrangement gives diamond

* exceptional hardness,
* high thermal conductivity,
* wide electronic band gap.

The bond angle is just as important as the bond length.

If the same carbon atoms retained identical bond lengths but different bond angles, the resulting material would no longer possess diamond's remarkable properties.

---

### Graphene

Graphene consists of carbon atoms arranged in a two-dimensional honeycomb lattice.

Every carbon atom exhibits

$$
120^\circ
$$

bond angles resulting from

$$
sp^2
$$

hybridization.

These bond angles create

* delocalized π electrons,
* extremely high electrical conductivity,
* exceptional carrier mobility,
* remarkable mechanical flexibility.

Again, geometry—not merely pairwise distances—determines the material's behavior.

---

### Silicon

Crystalline silicon shares the same diamond cubic structure as carbon.

Its electronic properties depend sensitively on the tetrahedral coordination around every silicon atom.

Small angular distortions introduced by

* strain,
* defects,
* impurities,

can significantly alter

* carrier mobility,
* band gap,
* phonon transport.

A model that explicitly understands bond angles naturally captures these effects.

---

### Perovskites

Perovskites provide an even stronger example.

In compounds such as

$$
ABX_3,
$$

the rotation and tilting of the

$$
BX_6
$$

octahedra modify

* electronic structure,
* ferroelectric behavior,
* dielectric properties,
* ionic conductivity.

Remarkably,

the bond lengths often change only slightly,

while the bond angles undergo substantial variation.

Many important phase transitions in perovskites are therefore driven primarily by angular distortions rather than changes in interatomic distance.

---

These examples reveal a common theme:

> **Many material properties are controlled not only by which atoms are bonded, but also by how those bonds are arranged in space.**

A graph neural network that ignores angular information is therefore missing an important part of the underlying physics.

---

## 17.1.4 Learning Higher-Order Interactions

Most early graph neural networks model interactions between **two atoms at a time**.

These are called **two-body interactions**.

Mathematically,

a message from atom

$j$

to atom

$i$

can be written as

$$
\mathbf{m}_{ij}
===============

f(\mathbf{h}_i,\mathbf{h}*j,\mathbf{e}*{ij}),
$$

where

* $\mathbf{h}_i$ is the feature vector of atom $i$,
* $\mathbf{h}_j$ is the feature vector of atom $j$,
* $\mathbf{e}_{ij}$ represents the bond connecting them.

Although highly successful,

this formulation neglects interactions involving three atoms simultaneously.

Many physical systems depend on

$$
(i,j,k),
$$

rather than simply

$$
(i,j).
$$

Examples include

* bond-angle potentials,
* covalent bonding,
* orbital hybridization,
* lattice distortions,
* crystal-field effects,
* angular strain.

These are collectively referred to as **three-body interactions**.

ALIGNN was specifically designed to learn these interactions efficiently.

---

## 17.1.5 The Key Idea Behind ALIGNN

The central innovation of ALIGNN is surprisingly elegant.

Instead of attempting to redesign graph convolution itself, the authors asked a different question:

> **What if bonds themselves became graph nodes?**

This simple idea completely changes the representation.

Instead of constructing only the crystal graph,

ALIGNN constructs **two coupled graphs**.

The first graph is the familiar crystal graph.

```text
Atoms

↓

Nodes

Bonds

↓

Edges
```

The second graph is called the **line graph**.

In the line graph,

the roles are reversed.

```text
Original Bond

↓

Node

Shared Atom

↓

Edge
```

Now,

two bonds become connected whenever they share a common atom.

This seemingly small modification allows the neural network to perform message passing directly between neighboring bonds.

Consequently,

bond angles become explicit graph features rather than hidden relationships that must be inferred.

This is the defining contribution of ALIGNN.

---

## 17.1.6 Why This Was a Major Breakthrough

The introduction of the line graph solved a long-standing challenge in atomistic graph neural networks.

Previous architectures attempted to learn angular information implicitly through increasingly deep message-passing networks.

ALIGNN instead incorporated this geometric information directly into the graph topology.

This change produced several important advantages.

First,

the network became more physically informed.

Second,

learning became more data efficient because angular relationships no longer had to be discovered from scratch.

Third,

prediction accuracy improved across a wide range of benchmark datasets, particularly for properties strongly influenced by local geometry.

Most importantly,

the approach remained computationally practical.

Rather than introducing expensive tensor operations or complex spherical harmonics, ALIGNN leveraged a well-established concept from graph theory—the **line graph**—to incorporate higher-order geometric information in a natural and scalable way.

This elegant combination of graph theory, materials science, and deep learning has made ALIGNN one of the most influential graph neural network architectures for crystalline materials.

---

## Looking Ahead

In the next section, we will investigate **why bond angles play such a fundamental role in determining the behavior of materials**. Rather than immediately introducing line graphs and neural networks, we will first develop the physical intuition behind angular interactions from the perspectives of quantum mechanics, chemical bonding, crystallography, and solid-state physics. This understanding will provide the scientific foundation needed to appreciate why ALIGNN represents such an important advancement over earlier graph neural network architectures.


# 17.2 Why Bond Angles Matter

The success of the Atomistic Line Graph Neural Network (ALIGNN) is founded on a simple but profound observation:

> **Most material properties are determined not only by which atoms are bonded, but also by how those bonds are arranged in three-dimensional space.**

Traditional graph neural networks primarily model **pairwise (two-body) interactions**, where information is exchanged between neighboring atoms connected by a bond. While this representation captures interatomic distances, it neglects an equally important aspect of atomic geometry—the **relative orientation of neighboring bonds**.

For many classes of materials, bond angles are not merely geometric descriptors; they directly influence electronic structure, chemical bonding, lattice stability, elasticity, thermal transport, magnetic ordering, and phase transformations.

Understanding why bond angles matter is therefore essential before studying the mathematical formulation of ALIGNN.

---

# 17.2.1 Structure Determines Properties

One of the first principles taught in materials science is

> **Processing → Structure → Properties → Performance**

Among these four stages, **structure** plays the central role because nearly every material property originates from atomic arrangement.

However, atomic structure consists of multiple levels of information.

At the simplest level, we know

* which atoms are present,
* where they are located.

At a higher level, we identify

* neighboring atoms,
* bond lengths,
* coordination numbers.

Finally, the complete local geometry includes

* bond angles,
* torsion (dihedral) angles,
* polyhedral connectivity,
* lattice symmetry.

A complete structural description therefore contains much more information than a list of pairwise distances.

---

Consider two neighboring atoms.

```text id="b1d0gr"
A -------- B
```

The bond length is simply

$$
r_{AB}.
$$

Now introduce a third atom.

```text id="a9xy2m"
A

 \

  \

   B

  /

 /

C
```

Immediately,

a new geometric quantity appears:

$$
\theta
======

\angle ABC.
$$

The local atomic environment is now determined by both

$$
r_{AB},
\qquad
r_{BC},
\qquad
\theta.
$$

The first two quantities describe **distance**,

while the third describes **orientation**.

Both are required for a complete description of local geometry.

---

# 17.2.2 Pairwise Interactions

Most conventional graph neural networks assume that the interaction between atoms depends primarily on interatomic distance.

For atoms

$i$

and

$j$,

the interaction energy may be written as

$$
E_{ij}
======

f(r_{ij}),
$$

where

* $r_{ij}$ is the distance between atoms,
* $f$ is a learned nonlinear function.

This assumption works well for many systems.

For example,

metallic bonding often depends predominantly on neighbor distances because metallic electrons are highly delocalized.

Likewise,

many ionic interactions can be reasonably approximated using pairwise Coulomb interactions,

$$
E
=

\frac{q_iq_j}
{4\pi\varepsilon_0r_{ij}}.
$$

Although these models capture an important part of atomic interactions,

they ignore an essential fact:

**many chemical bonds are directional.**

---

# 17.2.3 Directional Chemical Bonding

Unlike metallic bonding,

covalent bonding depends strongly on orbital orientation.

When two atoms form a covalent bond,

their atomic orbitals overlap.

The amount of overlap depends not only on distance,

but also on direction.

For example,

consider two carbon atoms.

The overlap between two

$p$

orbitals changes dramatically as the orbitals rotate.

```text id="z4p5tx"
Maximum Overlap

p ===== p
```

versus

```text id="e7v1ql"
Little Overlap

p

|

|

p
```

Even if the bond length remains identical,

rotating the orbitals changes

* bond strength,
* electron density,
* bonding energy.

Consequently,

the bond angle becomes a physically meaningful quantity.

This directionality is one of the fundamental reasons why bond-angle information must be incorporated into machine learning models for materials.

---

# 17.2.4 Hybridization and Bond Angles

One of the clearest demonstrations of the importance of bond angles is found in orbital hybridization.

Consider carbon.

Carbon can form several different hybridization states.

| Hybridization |     Bond Angle | Geometry        |
| ------------- | -------------: | --------------- |
| $sp$          |    $180^\circ$ | Linear          |
| $sp^2$        |    $120^\circ$ | Trigonal planar |
| $sp^3$        | $109.47^\circ$ | Tetrahedral     |

Notice that

the **same element**

produces

three completely different geometries.

The distinguishing feature is not composition.

It is not even bond length.

It is primarily the **bond angle**.

---

### Example: Diamond

Diamond is composed entirely of carbon atoms.

Each carbon atom forms four

$sp^3$

hybridized covalent bonds.

```text id="w6f2kh"
          C

         /|\

        / | \

       C--C--C

        \ | /

         \|/

          C
```

The ideal bond angle is

$$
109.47^\circ.
$$

This geometry maximizes orbital overlap,

creating one of the strongest crystal structures known.

Its properties include

* extreme hardness,
* high thermal conductivity,
* large band gap.

---

### Example: Graphene

Graphene also consists entirely of carbon atoms.

However,

its bonding is completely different.

Each atom is

$sp^2$

hybridized,

forming

$$
120^\circ
$$

bond angles.

```text id="k8l1vr"
      C

    /   \

   C-----C

    \   /

      C
```

The change from

$$
109.47^\circ
$$

to

$$
120^\circ
$$

completely transforms the material.

Graphene becomes

* electrically conductive,
* mechanically flexible,
* atomically thin,
* chemically distinct.

Thus,

bond angle alone fundamentally changes material properties.

---

# 17.2.5 Coordination Polyhedra

Many crystalline materials are best described in terms of coordination polyhedra.

Instead of focusing on individual bonds,

materials scientists study the arrangement of neighboring atoms around a central atom.

Common coordination environments include

| Coordination | Geometry        |
| ------------ | --------------- |
| 2            | Linear          |
| 3            | Trigonal planar |
| 4            | Tetrahedral     |
| 4            | Square planar   |
| 6            | Octahedral      |
| 8            | Cubic           |

These geometries are defined primarily by **bond angles**.

For example,

an octahedron contains

```text id="i5ot1s"
        X

        |

X ----- M ----- X

        |

        X
```

Neighboring bonds meet at

$$
90^\circ
$$

while opposite bonds form

$$
180^\circ.
$$

These angular relationships strongly influence

* crystal-field splitting,
* magnetic behavior,
* optical properties,
* electronic conductivity.

A neural network that ignores these angles loses valuable structural information.

---

# 17.2.6 Bond Angles and Potential Energy

Bond angles are also fundamental in atomistic potential energy models.

Many classical force fields include an explicit angular term.

For three atoms

$i$,

$j$,

and

$k$,

the angular contribution is often written as

$$
E_{\mathrm{angle}}
==================

k_\theta
(\theta-\theta_0)^2,
$$

where

* $\theta$ is the current bond angle,
* $\theta_0$ is the equilibrium angle,
* $k_\theta$ is the angular force constant.

This equation immediately reveals an important fact:

the energy depends directly on the bond angle.

If the angle changes,

the energy changes,

even when bond lengths remain constant.

Machine learning models that do not explicitly encode angular information must learn this relationship indirectly from data, making the learning process more difficult and less data-efficient.

---

# 17.2.7 Why Two Crystals Can Have Similar Bond Lengths but Different Properties

A common misconception is that similar bond lengths imply similar material properties.

This is rarely true.

Consider two hypothetical crystals.

Both possess identical bond lengths,

$$
r = 2.35\ \text{Å}.
$$

However,

their bond angles differ.

Crystal A

```text id="j3e0lh"
109°

Tetrahedral
```

Crystal B

```text id="w2cmgk"
90°

Distorted
```

Although every bond length is identical,

the electronic structure,

mechanical stiffness,

and lattice stability

may differ dramatically.

The reason is that electrons respond to the complete three-dimensional environment,

not merely pairwise distances.

ALIGNN is specifically designed to distinguish between such structures by explicitly modeling angular relationships.

---

# 17.2.8 The Limitation of Conventional Graph Neural Networks

A conventional crystal graph contains

```text id="jlwmz8"
Atoms

↓

Nodes

Bonds

↓

Edges
```

Messages are exchanged only between neighboring atoms.

To infer bond-angle information,

the network must combine information from several layers of message passing.

For example,

to understand

$$
A-B-C,
$$

the model first learns

$$
A \rightarrow B,
$$

then

$$
B \rightarrow C,
$$

and only after multiple updates may it infer the angle between the two bonds.

This indirect learning process has several disadvantages:

* It requires deeper networks.
* Training becomes more difficult.
* Larger datasets are needed.
* Important geometric relationships may be diluted through repeated message passing.

These limitations motivated researchers to search for a representation that encodes angular information directly.

---

# 17.2.9 ALIGNN's Solution

ALIGNN introduces a fundamentally different idea.

Instead of treating only atoms as graph nodes,

it also represents **bonds as nodes** in a second graph known as the **line graph**.

This transformation allows neighboring bonds to communicate directly.

As a result,

bond-angle information becomes an explicit part of message passing rather than an implicit feature that must be inferred.

This elegant graph-theoretic construction enables ALIGNN to learn three-body interactions efficiently while remaining computationally scalable.

In the next section, we will formalize this idea by introducing the **crystal graph representation** used in ALIGNN, defining its nodes, edges, feature vectors, and mathematical notation before progressing to the construction of the line graph itself.


# 17.3 Crystal Graph Representation

Before introducing the line graph and the ALIGNN architecture, we must first understand how a crystalline material is converted into a graph. This transformation is the foundation upon which every graph neural network for materials science is built.

Although the idea of representing a crystal as a graph appears simple, constructing an informative graph requires careful consideration of crystallography, periodic boundary conditions, neighborhood definitions, feature engineering, and computational efficiency. Poor graph construction can significantly degrade model performance regardless of how powerful the neural network architecture may be.

In this section, we develop the mathematical framework of crystal graphs from first principles. We begin with the physical description of a crystal and gradually transform it into a graph suitable for deep learning.

---

# 17.3.1 From Crystal Structure to Graph

A crystalline material is defined by two fundamental components:

* A **lattice**, which specifies the periodic arrangement of space.
* A **basis**, which specifies the atoms associated with each lattice point.

Mathematically, a crystal can be expressed as

$$
\boxed{
\mathcal{C} = (\mathcal{L}, \mathcal{A})
}
$$

where

* $\mathcal{L}$ denotes the lattice,
* $\mathcal{A}$ denotes the collection of atoms occupying the lattice.

The lattice itself is generated by three primitive vectors

$$
\mathbf{a}_1,;
\mathbf{a}_2,;
\mathbf{a}_3.
$$

Every lattice point is obtained through integer combinations of these vectors,

$$
\boxed{
\mathbf{R}
==========

n_1\mathbf{a}_1
+
n_2\mathbf{a}_2
+
n_3\mathbf{a}_3,
}
$$

where

$$
n_1,n_2,n_3\in\mathbb{Z}.
$$

Each lattice point contains one or more atoms that form the basis.

---

## Example: Silicon

Silicon crystallizes in the diamond cubic structure.

Its unit cell contains

* eight atoms,
* tetrahedral coordination,
* periodic repetition in all directions.

Instead of storing infinitely many atoms, we store

* one unit cell,
* lattice vectors,
* periodic boundary conditions.

This compact representation contains the complete crystal structure.

---

# 17.3.2 Why Graph Representation?

Neural networks require numerical data.

A crystal, however, is naturally represented by

* lattice vectors,
* atomic coordinates,
* atomic species.

These are not immediately compatible with graph neural networks.

Instead, we convert the crystal into a graph

$$
\boxed{
\mathcal{G}
===========

(\mathcal{V},\mathcal{E}),
}
$$

where

* $\mathcal{V}$ is the set of nodes,
* $\mathcal{E}$ is the set of edges.

In ALIGNN,

**nodes correspond to atoms**

while

**edges correspond to neighboring atomic interactions**.

Graph representation offers several advantages:

* It naturally handles crystals with different numbers of atoms.
* Local interactions are explicitly represented.
* The graph is invariant to atom indexing.
* Message passing mirrors physical interactions between neighboring atoms.
* The representation is compatible with modern graph deep learning libraries such as DGL and PyTorch Geometric.

---

# 17.3.3 Nodes: Representing Atoms

Every atom becomes one graph node.

If the crystal contains

$$
N
$$

atoms,

then

$$
|\mathcal{V}| = N.
$$

Each node possesses a feature vector

$$
\boxed{
\mathbf{h}_i
\in
\mathbb{R}^{d_h},
}
$$

where

* $i$ denotes the atom index,
* $d_h$ is the node feature dimension.

The feature vector contains information describing the atom.

---

## Typical Atomic Features

Possible node features include

| Feature                 | Example |
| ----------------------- | ------- |
| Atomic number           | 14      |
| Group                   | IV      |
| Period                  | 3       |
| Electronegativity       | 1.90    |
| Atomic radius           | 111 pm  |
| Covalent radius         | 116 pm  |
| Valence electrons       | 4       |
| Electron affinity       | 1.39 eV |
| First ionization energy | 8.15 eV |
| Atomic mass             | 28.09 u |

In practice,

ALIGNN usually begins with

the atomic number

and learns an embedding vector.

Instead of manually engineering dozens of descriptors,

the network learns a dense representation

$$
\boxed{
\mathbf{h}_i
============

\mathrm{Embedding}(Z_i),
}
$$

where

$$
Z_i
$$

is the atomic number.

This approach is analogous to word embeddings used in natural language processing.

---

# 17.3.4 Learned Atomic Embeddings

Suppose our dataset contains

94

chemical elements.

A simple one-hot representation would require

94 dimensions.

Instead,

ALIGNN learns

a dense embedding,

for example,

$$
94
\rightarrow
64.
$$

Thus,

Silicon

($Z=14$)

may become

$$
\mathbf{h}_{\mathrm{Si}}
========================

[-0.42,;
1.15,;
\cdots,;
0.73].
$$

These numbers have no direct physical interpretation.

Instead,

they are optimized during training so that chemically similar elements acquire similar representations.

For example,

Silicon and Germanium often become close in embedding space because they exhibit similar chemical behavior.

---

# 17.3.5 Edges: Representing Atomic Interactions

The next step is determining which atoms should be connected.

Unlike molecules,

crystals do not possess a unique bond list.

Instead,

neighbors must be determined algorithmically.

An edge exists whenever two atoms satisfy a neighborhood criterion.

The graph edge is written as

$$
\boxed{
(i,j)
\in
\mathcal{E}.
}
$$

This indicates that atoms

$i$

and

$j$

interact directly.

Unlike many social or citation graphs,

crystal graphs are usually **undirected**.

Therefore,

$$
(i,j)
=====

(j,i).
$$

Many graph libraries nevertheless store

both directions,

creating

two directed edges

for efficient message passing.

---

# 17.3.6 Determining Neighboring Atoms

A crucial question is:

> Which atoms should be connected?

There is no universally correct answer.

Several approaches exist.

---

## Fixed Cutoff Radius

The most common method connects atoms whose distance satisfies

$$
\boxed{
r_{ij}
<
r_c,
}
$$

where

* $r_{ij}$ is the interatomic distance,
* $r_c$ is the cutoff radius.

Typical cutoff values range from

5 Å

to

8 Å,

depending on the dataset.

---

Advantages

* Simple
* Efficient
* Widely used

Disadvantages

* Neighbor count varies
* Sensitive to cutoff choice

---

## k-Nearest Neighbors

Alternatively,

each atom may connect to exactly

$k$

nearest atoms.

Advantages

* Constant graph degree
* Stable computational cost

Disadvantages

* May connect physically unrelated atoms
* Choice of $k$ affects accuracy

---

## Voronoi Neighbors

A more physically motivated method uses

Voronoi tessellation.

Atoms sharing a Voronoi face become neighbors.

Advantages

* Geometry-based
* Adaptive

Disadvantages

* More computationally expensive
* Difficult for highly distorted structures

---

ALIGNN typically adopts a cutoff-radius strategy together with neighbor limits to ensure computational efficiency.

---

# 17.3.7 Edge Features

Edges also possess feature vectors.

For edge

$(i,j)$,

the feature vector is

$$
\boxed{
\mathbf{e}_{ij}
\in
\mathbb{R}^{d_e}.
}
$$

Unlike node features,

edge features describe relationships between atoms.

Typical edge features include

* interatomic distance,
* displacement vector,
* periodic image information,
* radial basis expansion.

---

The simplest feature is

the distance

$$
r_{ij}
======

|\mathbf{x}_j-\mathbf{x}_i|.
$$

However,

using only the raw distance often limits expressive power.

Instead,

ALIGNN expands distances into a higher-dimensional representation.

We will study this radial basis expansion later in the chapter.

---

# 17.3.8 Atomic Coordinates

Each atom possesses Cartesian coordinates

$$
\boxed{
\mathbf{x}_i
============

(x_i,y_i,z_i).
}
$$

The distance between two atoms is

$$
\boxed{
r_{ij}
======

|
\mathbf{x}_j-\mathbf{x}_i
|.
}
$$

These coordinates are essential because

edge construction,

distance computation,

and later bond-angle calculations

all originate from them.

---

# 17.3.9 Periodic Boundary Conditions

One of the defining characteristics of crystalline materials is periodicity.

Unlike isolated molecules,

crystals extend infinitely.

Therefore,

neighbors may exist across unit-cell boundaries.

Consider an atom near the edge of the unit cell.

```text
+----------------------+
|                     ●|
|                      |
|                      |
|●                     |
+----------------------+
```

Although the atoms appear far apart inside the unit cell,

they are actually nearest neighbors because the crystal repeats periodically.

Graph construction must therefore consider neighboring periodic images.

Mathematically,

the displacement becomes

$$
\boxed{
\mathbf{r}_{ij}
===============

## \mathbf{x}_j

\mathbf{x}_i
+
\mathbf{T},
}
$$

where

$$
\mathbf{T}
==========

n_1\mathbf{a}_1
+
n_2\mathbf{a}_2
+
n_3\mathbf{a}_3
$$

is a lattice translation vector.

Ignoring periodicity would produce incorrect neighbor lists and severely degrade prediction accuracy.

---

# 17.3.10 Graph Connectivity

The completed crystal graph can now be summarized as

```text
Atom

↓

Node

↓

Node Features

↓

Neighbor Search

↓

Edges

↓

Edge Features

↓

Crystal Graph
```

Every node stores atomic information.

Every edge stores interaction information.

The resulting graph provides a compact yet expressive representation of the crystal suitable for graph neural networks.

---

# 17.3.11 Mathematical Representation of the Crystal Graph

The complete crystal graph is formally defined as

$$
\boxed{
\mathcal{G}
===========

(\mathcal{V},
\mathcal{E},
\mathbf{H},
\mathbf{E}),
}
$$

where

* $\mathcal{V}$ is the node set,
* $\mathcal{E}$ is the edge set,
* $\mathbf{H}$ contains node features,
* $\mathbf{E}$ contains edge features.

Node feature matrix

$$
\boxed{
\mathbf{H}
\in
\mathbb{R}^{N\times d_h},
}
$$

Edge feature matrix

$$
\boxed{
\mathbf{E}
\in
\mathbb{R}^{M\times d_e},
}
$$

where

* $N$ is the number of atoms,
* $M$ is the number of graph edges.

These matrices serve as the primary inputs to the ALIGNN model.

---

# 17.3.12 Limitations of the Crystal Graph

Despite its effectiveness, the crystal graph still contains an important limitation.

Every edge represents only a **pairwise interaction**.

The graph knows

* which atoms are neighbors,
* how far apart they are,

but it does **not explicitly encode the relationship between neighboring bonds**.

For example, consider the local environment

```text
A

 \

  B

 /

C
```

The crystal graph contains

* edge $(A,B)$,
* edge $(B,C)$,

but it does not directly store the angle

$$
\angle ABC.
$$

Recovering this angle requires combining information from multiple edges, making the learning process indirect.

This observation motivates the central innovation of ALIGNN.

Instead of modifying the graph convolution itself, ALIGNN constructs an entirely new graph—the **line graph**—in which **bonds become nodes** and **bond-angle relationships become edges**.

In the next section, we will develop the theory of line graphs from graph theory and show how this elegant transformation allows ALIGNN to perform explicit angular message passing.


# 17.4 Line Graph Theory

In the previous section, we represented a crystal as a graph in which **atoms are nodes** and **neighboring atomic interactions are edges**. This representation is sufficient for learning pairwise interactions, but it cannot explicitly model the geometric relationship between neighboring bonds.

The key innovation of ALIGNN is that it does **not** attempt to modify the message-passing equations of the crystal graph. Instead, it introduces an entirely new graph derived from the original crystal graph. This graph is known as the **line graph**.

Although the concept of a line graph originated in graph theory more than seventy years ago, ALIGNN demonstrated that it provides an elegant and computationally efficient mechanism for incorporating angular information into graph neural networks for crystalline materials.

In this section, we study line graphs from both mathematical and physical perspectives. By the end of this section, the reader should understand not only how a line graph is constructed but also why it naturally encodes bond-angle interactions.

---

# 17.4.1 The Fundamental Idea

Consider a simple crystal graph

```text id="lg1"
A -------- B -------- C
```

The graph contains

* three atoms,
* two bonds.

Traditionally,

graph neural networks perform message passing between the **atoms**.

The messages are

```text id="lg2"
A  ←→  B

B  ←→  C
```

Notice something interesting.

The two bonds

$$
(A,B)
$$

and

$$
(B,C)
$$

share the common atom

$$
B.
$$

This shared atom defines a bond angle

$$
\angle ABC.
$$

However,

the original graph contains no direct connection between these two bonds.

Instead,

both bonds communicate only indirectly through atom

$$
B.
$$

The ALIGNN paper asks an important question:

> **What if the bonds themselves could communicate directly?**

This single idea gives rise to the line graph.

---

# 17.4.2 What Is a Line Graph?

A **line graph** is a graph constructed from another graph.

Instead of treating atoms as nodes,

the line graph treats **edges of the original graph as nodes**.

Mathematically,

if the original graph is

$$
\mathcal{G}
===========

(\mathcal{V},\mathcal{E}),
$$

then its line graph is

$$
\boxed{
\mathcal{L}(\mathcal{G}).
}
$$

The defining property is

> Every edge in the original graph becomes a node in the line graph.

Furthermore,

> Two nodes in the line graph are connected if their corresponding edges in the original graph share a common endpoint.

Although this definition appears abstract,

its geometric meaning is remarkably intuitive.

---

# 17.4.3 Constructing the Simplest Line Graph

Consider again

```text id="lg3"
A -------- B -------- C
```

The crystal graph contains two edges

```text id="lg4"
e₁ = (A,B)

e₂ = (B,C)
```

Since the two edges share atom

$$
B,
$$

the line graph becomes

```text id="lg5"
e₁ -------- e₂
```

Notice what happened.

The atoms disappeared.

The bonds became nodes.

The shared atom created an edge.

This new edge represents the bond angle

$$
\angle ABC.
$$

This is the key insight behind ALIGNN.

---

# 17.4.4 A Larger Example

Consider a tetrahedral coordination environment.

```text id="lg6"
          D

          |

          |

A ------- B ------- C

          |

          |

          E
```

The crystal graph contains four bonds

```text id="lg7"
AB

BC

BD

BE
```

Each bond becomes a node in the line graph.

Now determine which bonds share atom

$$
B.
$$

Every pair shares

$$
B,
$$

therefore the line graph becomes

```text id="lg8"
        AB

      / |  \

    BC BD BE

      \ | /

      (complete)
```

Every neighboring bond can now exchange information directly.

Observe that

the line graph naturally connects all bond pairs surrounding the central atom.

Those connections correspond exactly to bond angles.

---

# 17.4.5 Formal Definition

Let

$$
\mathcal{G}
===========

(\mathcal{V},\mathcal{E})
$$

be a graph.

Suppose

$$
e_i
\in
\mathcal{E}
$$

denotes one edge.

The line graph

$$
\mathcal{L}(\mathcal{G})
$$

contains

* one node for every edge,

that is,

$$
|\mathcal{V}_L|
===============

|\mathcal{E}|.
$$

Two nodes

$$
e_i,
e_j
$$

are connected whenever

$$
\boxed{
e_i
\cap
e_j
\neq
\varnothing.
}
$$

In words,

two edges become neighbors if they share at least one endpoint.

For crystalline materials,

that endpoint corresponds to the atom at which a bond angle is formed.

---

# 17.4.6 Why This Represents Bond Angles

Consider three atoms

```text id="lg9"
A

 \

  \

   B

  /

 /

C
```

The crystal graph contains

```text id="lg10"
Edge 1

A — B

Edge 2

B — C
```

The line graph transforms this into

```text id="lg11"
(A,B)

   |

(B,C)
```

The connecting edge implicitly represents

$$
\angle ABC.
$$

Therefore,

instead of computing angles through repeated graph convolutions,

ALIGNN directly constructs a graph whose topology already contains angular relationships.

This dramatically simplifies learning.

---

# 17.4.7 Information Flow in the Two Graphs

It is useful to compare information propagation in the crystal graph and the line graph.

---

### Crystal Graph

```text id="lg12"
Atom

↓

Neighbor Atom

↓

Neighbor Atom

↓

Neighbor Atom
```

Information flows

between atoms.

---

### Line Graph

```text id="lg13"
Bond

↓

Neighbor Bond

↓

Neighbor Bond
```

Information flows

between bonds.

---

ALIGNN performs message passing on **both graphs simultaneously**.

This is one of the major reasons for its superior predictive performance.

---

# 17.4.8 Physical Interpretation

The crystal graph answers questions such as

* Which atoms interact?
* What is the bond length?
* Which neighboring atoms exist?

The line graph answers different questions

* Which bonds share an atom?
* What is the local bond geometry?
* Which bond angles exist?
* How are neighboring bonds oriented?

Thus,

the two graphs describe complementary physical information.

| Crystal Graph         | Line Graph              |
| --------------------- | ----------------------- |
| Atoms                 | Bonds                   |
| Distances             | Angles                  |
| Pairwise interactions | Three-body interactions |
| Atomic environment    | Bond environment        |

Neither graph alone provides a complete description of the local atomic structure.

ALIGNN therefore learns from both simultaneously.

---

# 17.4.9 Graph Sizes

Suppose the crystal graph contains

* $N$ atoms,
* $M$ bonds.

Then

Crystal Graph

* Nodes

$$N$$

* Edges

$$M.$$

The line graph contains

* Nodes

$$M,$$

since every bond becomes one node.

The number of edges in the line graph depends on the local coordination.

If one atom has

$$k$$

neighboring bonds,

then it contributes

$$\boxed{\frac{k(k-1)}{2}}$$

line-graph edges,

because every pair of bonds meeting at that atom forms one angular connection.

This observation immediately explains why line graphs are often denser than crystal graphs.

---

# 17.4.10 Computational Complexity

Constructing the crystal graph primarily involves neighbor searching.

For

$$N$$

atoms,

neighbor search algorithms typically scale approximately as

$$O(N \log N)$$

using spatial partitioning methods such as KD-trees or cell lists.

Constructing the line graph requires examining the neighbors of every atom.

If atom

$i$

has

$$k_i$$

neighbors,

then the total number of line-graph edges is

$$\boxed{\sum_i \frac{k_i(k_i-1)}{2}.}$$

In most crystalline materials,

the coordination number is relatively small (typically between 4 and 12), so line graph construction remains computationally tractable.

However, for structures with very high coordination or large cutoff radii, the line graph can become substantially larger than the original crystal graph, increasing both memory consumption and computational cost.

---

# 17.4.11 Why Not Store Angles Directly?

A natural question arises:

> If bond angles are important, why not simply compute every angle and store it as an edge feature?

While this approach is possible, it has important limitations.

First, bond angles would remain **static descriptors**. The neural network could read their values but could not naturally propagate information between neighboring angular environments.

Second, learning would still occur primarily on the crystal graph, limiting the ability to model interactions among bonds.

By constructing a dedicated line graph, ALIGNN transforms bond angles into **first-class entities** within the learning process. Bond environments evolve through message passing just as atomic environments do, allowing the network to learn rich higher-order geometric representations rather than treating angles as fixed inputs.

---

# 17.4.12 Why the Line Graph Is an Elegant Solution

The introduction of the line graph is elegant for three reasons.

1. **It preserves the graph neural network framework.**
Rather than inventing an entirely new neural architecture, ALIGNN extends standard message passing to a second graph.
2. **It naturally incorporates higher-order geometry.**
Bond-angle information emerges from graph topology instead of requiring handcrafted engineering.
3. **It remains computationally scalable.**
The construction relies on established graph algorithms and integrates seamlessly with graph deep learning libraries.

This combination of mathematical simplicity, physical interpretability, and computational efficiency is one of the principal reasons why ALIGNN became a landmark architecture in materials informatics.

---

## Looking Ahead

Now that we understand the theory of line graphs, the next step is to apply these ideas to crystalline materials in practice. In the next section, we will study **bond graph and line graph construction**, showing how crystal structures are converted into the dual-graph representation used by ALIGNN. We will discuss neighbor searching, periodic boundary conditions, graph-building algorithms, and the practical implementation of these ideas using modern graph deep learning libraries.

# 17.5 Bond Graph Construction

In the previous section, we introduced the mathematical concept of a **line graph** and showed that it provides an elegant mechanism for representing bond-angle relationships. However, understanding the theory alone is not sufficient for implementing ALIGNN.

A practical implementation requires answering several important questions:

* How are neighboring atoms identified?
* How are bonds represented computationally?
* How is the line graph constructed from the crystal graph?
* How are periodic boundary conditions handled?
* How are bond angles identified efficiently?
* How are the resulting graphs stored for neural network training?

These questions are the focus of this section.

Unlike molecular graphs, where bonds are explicitly defined by chemical connectivity, crystalline materials generally do not contain an explicit bond list. Instead, the bond graph must be generated algorithmically from the atomic coordinates and lattice vectors. The quality of this graph construction directly influences the quality of the learned representations.

---

# 17.5.1 From Crystal Structure to Dual Graphs

ALIGNN does not operate on a single graph.

Instead, every crystal is transformed into **two coupled graphs**:

1. **Crystal Graph**
2. **Line Graph**

This preprocessing pipeline is illustrated below.

```text id="alignn_pipeline"
Crystal Structure (CIF)

        │
        ▼

Atomic Coordinates
Lattice Vectors

        │
        ▼

Neighbor Search

        │
        ▼

Crystal Graph

        │
        ▼

Line Graph

        │
        ▼

ALIGNN

```

Notice that the line graph is **not** constructed directly from the crystal structure.

Instead,

it is generated from the crystal graph.

Therefore,

building an accurate crystal graph is the first and most important preprocessing step.

---

# 17.5.2 Neighbor Search

The first task is determining which atoms should interact.

Suppose atom

$i$

has Cartesian position

$$\mathbf{x}_i.$$

For every other atom

$j$,

we compute the distance

$$\boxed{r_{ij} = \left\vert{} \mathbf{x}_j-\mathbf{x}_i \right\vert{}.}$$

If

$$r_{ij}<r_c,$$

the atoms are considered neighbors.

The cutoff radius

$r_c$

is a hyperparameter.

Typical values are

| Dataset | Typical Cutoff |
| --- | --- |
| Molecules | 4–5 Å |
| Crystals | 6–8 Å |
| Dense metals | 8–10 Å |

Increasing the cutoff captures more long-range interactions but also increases computational cost.

---

# 17.5.3 Why Cutoff Selection Matters

Choosing the cutoff radius is not merely a programming detail.

It changes the graph itself.

Suppose the cutoff is too small.

```text id="cutoff_small"
A ----- B

      C

```

Only one bond may be detected.

The graph becomes disconnected.

Important physical interactions are lost.

---

Now consider an excessively large cutoff.

```text id="cutoff_large"
Every atom connected
to almost every atom

```

The graph becomes overly dense.

Consequences include

* unnecessary computation,
* higher GPU memory usage,
* increased training time,
* noisier message passing.

An appropriate cutoff therefore balances

* physical realism,
* computational efficiency.

---

# 17.5.4 Maximum Neighbor Constraint

Many crystals exhibit varying coordination numbers.

Some atoms may have

4 neighbors,

while others possess

12 or more.

To maintain efficient batching,

ALIGNN usually limits the maximum number of neighbors.

Suppose

$$k_{\max}=12.$$

If an atom has

18

neighbors,

only the

12

closest neighbors are retained.

The resulting graph has approximately uniform density across the dataset.

This greatly simplifies mini-batch training on GPUs.

---

# 17.5.5 Periodic Boundary Conditions

Neighbor searching becomes more complicated for crystalline materials because the unit cell repeats infinitely.

Consider the following unit cell.

```text id="pbc_cell"
+--------------------+
|                  ● |
|                    |
|                    |
| ●                  |
+--------------------+

```

The two atoms appear far apart.

However,

after applying periodic boundary conditions,

they are actually nearest neighbors.

Ignoring periodicity would produce an incorrect graph.

---

Mathematically,

the displacement vector becomes

$$\boxed{\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + \mathbf{T},}$$

where

$$\mathbf{T} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3$$

is a lattice translation.

The neighbor search therefore considers atoms in adjacent periodic images.

---

## Minimum Image Convention

Rather than checking infinitely many periodic images,

modern implementations use the **minimum image convention**.

For every atom pair,

the algorithm finds the shortest possible periodic displacement.

Thus,

$$r_{ij} = \min \left( \vert{} \mathbf{x}_j-\mathbf{x}_i+\mathbf{T} \vert{} \right).$$

This produces the physically meaningful nearest-neighbor distance while keeping the computation tractable.

---

# 17.5.6 Building the Crystal Graph

After neighbor searching,

constructing the crystal graph is straightforward.

For every neighboring atom pair,

create an edge.

Algorithmically,

```text id="graph_algorithm"
For every atom i

    Find neighboring atoms

    For every neighbor j

        Create edge (i,j)

        Compute edge features

```

The graph now contains

* node list,
* edge list,
* node features,
* edge features.

This graph is identical to the input used by models such as CGCNN.

ALIGNN now performs one additional step.

---

# 17.5.7 Converting Bonds into Nodes

Suppose the crystal graph contains

```text id="bond_nodes1"
A ---- B ---- C

```

The graph edges are

$$(A,B)$$

and

$$(B,C).$$

During line graph construction,

every edge becomes a node.

```text id="bond_nodes2"
Node 1

↓

(A,B)

Node 2

↓

(B,C)

```

This is the defining transformation of the ALIGNN preprocessing pipeline.

---

# 17.5.8 Determining Line-Graph Edges

The next question is

Which bond nodes should be connected?

The rule is extremely simple.

Two bond nodes become neighbors whenever their corresponding crystal-graph edges share one atom.

Example

```text id="shared_atom"
A

 \

  B

 /

C

```

Crystal graph edges

```text id="shared_edges"
AB

BC

```

These two edges share atom

$$B.$$

Therefore,

the line graph contains

```text id="line_edge"
AB

 |

BC

```

This edge represents the angular relationship

$$\angle ABC.$$

---

# 17.5.9 General Construction Algorithm

Suppose

atom

$i$

has neighbors

$$j_1, j_2, \ldots, j_k.$$

The corresponding crystal graph contains

$$k$$

edges,

namely

$$(i,j_1), (i,j_2), \ldots, (i,j_k).$$

Every pair of these bonds defines one bond angle.

Therefore,

the line graph contains

$$\boxed{\frac{k(k-1)}{2}}$$

connections around atom

$i$.

For example,

if

$$k=4,$$

the number of bond-angle connections becomes

$$\frac{4\times3}{2} = 6.$$

This simple combinatorial relationship explains why line graphs are denser than crystal graphs.

---

# 17.5.10 Bond Angle Identification

Every edge in the line graph corresponds to one bond angle.

Suppose

the neighboring bonds are

$$(i,j)$$

and

$$(i,k).$$

The angle is

$$\boxed{\theta = \cos^{-1} \left( \frac{(\mathbf{x}_j-\mathbf{x}_i) \cdot (\mathbf{x}_k-\mathbf{x}_i)}{\vert{}\mathbf{x}_j-\mathbf{x}_i\vert{} ; \vert{}\mathbf{x}_k-\mathbf{x}_i\vert{}} \right).}$$

Notice something important.

Although ALIGNN constructs the line graph using graph topology, the geometric interpretation of each line-graph edge is precisely this bond angle.

Depending on the implementation, the angle itself may be used as an edge feature or represented implicitly through learned geometric embeddings.

---

# 17.5.11 Data Structures Used in Practice

Modern implementations do not store graphs as adjacency matrices because most crystal graphs are sparse.

Instead,

they use sparse graph data structures.

For example,

the crystal graph is represented as

```text id="graph_storage"
Nodes

Edges

Node Features

Edge Features

```

The line graph has the same structure,

except that

its nodes correspond to bonds.

Libraries such as **DGL** and **PyTorch Geometric** efficiently manage these sparse structures, allowing message passing to scale to millions of edges.

---

# 17.5.12 Graph Construction Workflow

The complete preprocessing pipeline can now be summarized.

```text id="workflow"
Read CIF

      │

      ▼

Read lattice

Read coordinates

      │

      ▼

Neighbor Search

      │

      ▼

Crystal Graph

      │

      ▼

Convert edges → nodes

      │

      ▼

Construct Line Graph

      │

      ▼

Generate Features

      │

      ▼

ALIGNN Input

```

This workflow is executed once during dataset preparation and provides the dual-graph representation used throughout training.

---

# 17.5.13 Practical Considerations

Although graph construction appears conceptually straightforward, several practical issues deserve attention.

### Duplicate Edges

In periodic systems, the same physical interaction may be encountered through different lattice images. Care must be taken to avoid introducing duplicate edges that would artificially increase the connectivity of the graph.

### Numerical Precision

Distances and angles are computed using floating-point arithmetic. Small numerical errors can accumulate, particularly when evaluating inverse trigonometric functions. Robust implementations typically clamp cosine values to the interval

$$[-1,1]$$

before computing the inverse cosine.

### Neighbor Ordering

The order in which neighbors are processed should not affect the model's predictions. Consequently, graph neural networks are designed to be **permutation invariant**, ensuring that different neighbor orderings produce identical outputs.

### Memory Consumption

For highly coordinated structures or large cutoff radii, the number of line-graph edges can grow rapidly. Efficient sparse storage and mini-batching are therefore essential for scaling ALIGNN to large datasets.

---

# 17.5.14 Transition to Message Passing

At this point, we have completed the graph construction stage.

Every crystal has been transformed into two interconnected graphs:

* the **crystal graph**, which represents atomic connectivity, and
* the **line graph**, which represents relationships between neighboring bonds.

The next question is how these two graphs exchange information during learning.

To answer this, we must first study the **Edge-Gated Graph Convolution**, the fundamental computational building block of ALIGNN. Unlike conventional graph convolutions that update only atomic features, the edge-gated formulation simultaneously learns from node and edge representations, providing the mechanism through which information flows between the crystal graph and the line graph. This operation forms the mathematical core of the ALIGNN architecture and will be derived in the next section.

# 17.6 Edge-Gated Graph Convolution

The previous sections focused on **how ALIGNN represents a crystal**. We began with the crystal structure, converted it into a crystal graph, and then transformed that graph into a line graph that explicitly captures bond-angle relationships.

However, constructing these graphs alone does not constitute a learning algorithm.

The next question is:

> **How does information flow through these graphs?**

The answer lies in the **Edge-Gated Graph Convolution (EGGC)**, the fundamental computational building block of ALIGNN.

Unlike conventional graph convolution layers that primarily update **node features**, the edge-gated convolution simultaneously updates

* atomic features,
* bond features,

allowing both to evolve during learning.

This is a major departure from earlier graph neural networks.

Instead of treating edge features merely as fixed descriptors, ALIGNN treats them as **learnable entities** that participate actively in message passing.

This section develops the theory of edge-gated graph convolution from first principles.

---

# 17.6.1 Revisiting Conventional Graph Convolution

Before studying ALIGNN, let us briefly review how a standard graph neural network performs message passing.

Suppose a graph contains

$$N$$

nodes.

Each node possesses a feature vector

$$\mathbf{h}_i \in \mathbb{R}^{d}.$$

During one graph convolution layer,

node

$i$

collects information from its neighboring nodes,

$$\mathcal{N}(i).$$

The generic message-passing framework is

$$\boxed{\mathbf{h}_i^{(l+1)} = U \left( \mathbf{h}_i^{(l)}, ; \sum_{j\in\mathcal{N}(i)} M \left( \mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij} \right) \right),}$$

where

* $M(\cdot)$ denotes the **message function**,
* $U(\cdot)$ denotes the **update function**,
* $l$ denotes the network layer.

The overall computation proceeds in three stages.

```text id="eggc_1"
Neighbor Features

        │

        ▼

Message Function

        │

        ▼

Aggregation

        │

        ▼

Node Update

```

This framework is sufficiently general to describe many popular GNN architectures, including GraphSAGE, GCN, GAT, CGCNN, and MEGNet.

---

# 17.6.2 The Limitation of Fixed Edge Features

In many graph neural networks,

the edge features remain unchanged throughout training.

Suppose an edge initially stores

* bond length,
* bond type,
* distance embedding.

Then,

during every graph convolution,

these values simply assist message computation.

They are never updated.

Mathematically,

the edge feature

$$\mathbf{e}_{ij}$$

is treated as

a constant.

The node evolves,

but the bond does not.

This assumption is appropriate for many graph problems,

such as social networks,

where relationships rarely change.

However,

it is less suitable for atomistic systems.

---

Consider a carbon-carbon bond.

Initially,

the bond feature might simply encode

$$r_{CC} = 1.54;\text{Å}.$$

As neighboring atoms exchange information,

our understanding of this bond should also evolve.

The bond should gradually encode

* local chemistry,
* surrounding geometry,
* coordination environment,
* electronic influence.

Therefore,

the bond representation should itself become learnable.

This is the central motivation behind edge-gated convolution.

---

# 17.6.3 Learning Bonds as Dynamic Objects

ALIGNN treats bonds as dynamic entities.

Instead of storing

$$\mathbf{e}_{ij}$$

once,

the network learns

$$\mathbf{e}_{ij}^{(0)}, \mathbf{e}_{ij}^{(1)}, \mathbf{e}_{ij}^{(2)}, \ldots$$

Each graph convolution layer produces a new bond representation.

Consequently,

both atoms and bonds evolve simultaneously.

The learning process becomes

```text id="eggc_2"
Atoms

↓

Update

↓

Better Atom Features

──────────────

Bonds

↓

Update

↓

Better Bond Features

```

The two representations continuously reinforce one another.

---

# 17.6.4 Why Bond Features Should Change

Consider silicon.

Initially,

the network only knows

```text id="eggc_3"
Si

Bond length

2.35 Å

```

After several message-passing iterations,

the network has learned additional information.

For example,

the same bond may now implicitly encode

* tetrahedral coordination,
* neighboring silicon atoms,
* local strain,
* electronic environment,
* defect proximity.

Although the physical bond length remains unchanged,

the **feature vector representing that bond becomes increasingly informative**.

This richer representation allows subsequent graph convolutions to make more accurate predictions.

---

# 17.6.5 Message Passing as Information Exchange

To understand edge-gated convolution,

let us first examine the information flow.

Suppose atom

$i$

communicates with atom

$j$.

Information passes through the connecting bond.

```text id="eggc_4"
Atom i

   │

 Bond

   │

Atom j

```

Traditional GNNs focus primarily on

the two atoms.

ALIGNN instead recognizes

that the bond itself carries information.

Therefore,

every interaction consists of three components.

```text id="eggc_5"
Atom

↓

Bond

↓

Atom

```

This seemingly small conceptual change dramatically increases expressive power.

---

# 17.6.6 Computing Messages

Suppose

atom

$i$

receives information from

neighbor

$j$.

The transmitted message depends on

* the receiving atom,
* the sending atom,
* the connecting bond.

A general message function is

$$\boxed{\mathbf{m}_{ij} = \phi_m ( \mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij} ),}$$

where

* $\phi_m$ denotes a learnable neural network.

Notice that

the bond participates directly in message construction.

The message is therefore

context-dependent.

Two identical neighboring atoms may transmit different messages

if their bond environments differ.

---

# 17.6.7 Why Gating?

Suppose every neighboring atom contributes equally.

Then,

the update becomes

$$\sum_j \mathbf{m}_{ij}.$$

Unfortunately,

this assumption is unrealistic.

Not every bond is equally important.

For example,

consider

```text id="eggc_6"
Weak Bond

A -------- B

Strong Bond

C ========= D

```

Clearly,

the stronger interaction should contribute more strongly during message passing.

ALIGNN therefore introduces a **gate**.

The gate acts as a learned importance weight.

Instead of simply summing messages,

the network computes

$$\boxed{\mathbf{m}_{ij}' = g_{ij} , \mathbf{m}_{ij},}$$

where

$$g_{ij}$$

is the gate value.

---

# 17.6.8 Gate Values

The gate is computed from the edge feature.

A simple formulation is

$$\boxed{g_{ij} = \sigma ( \mathbf{W}_g \mathbf{e}_{ij} + \mathbf{b}_g ),}$$

where

* $\sigma(\cdot)$ is the sigmoid function,
* $\mathbf{W}_g$ is a learnable weight matrix,
* $\mathbf{b}_g$ is the bias vector.

Since

$$0 < \sigma(x) < 1,$$

the gate behaves like an attention coefficient.

Large values

allow information to pass.

Small values

suppress unimportant interactions.

Unlike Graph Attention Networks (GAT), where attention is computed primarily from node features, ALIGNN derives the gate from the **bond representation**, making the interaction explicitly dependent on the chemistry and geometry of the bond.

---

# 17.6.9 Physical Interpretation of the Gate

The gate can be interpreted as a learned measure of **bond importance**.

During training,

the network gradually discovers which interactions are most informative for the prediction task.

For example,

in predicting elastic constants,

strong covalent bonds may receive larger gate values,

while weak long-range interactions receive smaller values.

Similarly,

for magnetic materials,

exchange pathways between magnetic ions may be emphasized,

whereas chemically irrelevant interactions are attenuated.

Importantly,

these importance weights are **not manually specified**.

They emerge automatically through optimization.

Thus,

the gate provides an adaptive mechanism for filtering information according to the underlying physics present in the training data.

---

# 17.6.10 Aggregating Gated Messages

Once every message has been weighted,

the receiving atom aggregates information from all neighbors.

The aggregated message is

$$\boxed{\mathbf{m}_i = \sum_{j\in\mathcal{N}(i)} g_{ij} , \mathbf{m}_{ij}.}$$

Several important properties follow immediately.

* The aggregation is **permutation invariant**, meaning the order of neighbors does not matter.
* The gate allows different neighbors to contribute unequally.
* Richer edge representations lead to more informative messages.

This aggregated message will then be combined with the atom's existing representation to produce an updated node feature.

---

# 17.6.11 Updating Atomic Features

The new representation of atom

$i$

is computed by combining

its previous feature

with the aggregated neighbor information.

A common formulation is

$$\boxed{\mathbf{h}_i^{(l+1)} = \phi_u \left( \mathbf{h}_i^{(l)}, \mathbf{m}_i \right),}$$

where

$\phi_u$

is typically implemented using

* linear layers,
* nonlinear activation functions,
* residual connections,
* normalization layers.

The updated feature now contains information from

* the atom itself,
* neighboring atoms,
* the learned bond representations.

---

# 17.6.12 Updating Edge Features

The defining feature of ALIGNN is that **edges are also updated**.

Rather than remaining fixed,

the bond representation evolves after every convolution layer.

A general update rule is

$$\boxed{\mathbf{e}_{ij}^{(l+1)} = \phi_e \left( \mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij}^{(l)} \right),}$$

where

$\phi_e$

is another learnable neural network.

This update allows bond features to incorporate information from the atoms they connect while retaining their previous representation.

As layers accumulate,

edge features become increasingly expressive descriptors of the local chemical environment.

---

# 17.6.13 Summary of the Edge-Gated Convolution Layer

A single edge-gated graph convolution layer therefore performs four coordinated operations:

1. **Construct messages** using node and edge features.
2. **Compute edge gates** that determine the importance of each interaction.
3. **Aggregate gated messages** to update atomic representations.
4. **Update bond representations** so that edges evolve alongside nodes.

Unlike traditional graph convolutions, the edge-gated formulation treats the graph as a dynamic system in which both atoms and bonds learn increasingly rich representations through successive layers.

---

## Looking Ahead

In the next section, we will build upon the edge-gated convolution by introducing **Angular Message Passing**, the defining innovation of ALIGNN. We will show how the line graph enables message passing **between bonds rather than between atoms**, allowing the network to learn explicit three-body interactions and bond-angle-dependent physics that cannot be captured by conventional pairwise graph neural networks.

# 17.7 Angular Message Passing

The previous section introduced the **Edge-Gated Graph Convolution (EGGC)**, which enables simultaneous learning of atomic and bond representations. Although this significantly improves upon earlier graph neural networks, it still operates primarily on the **crystal graph**, where information flows between atoms connected by bonds.

However, many physical phenomena cannot be completely described by pairwise interactions alone.

Consider the following atomic arrangement.

```text
      A

       \

        \

         B

        /

       /

      C

```

The crystal graph contains two bonds

$$(A,B)$$

and

$$(B,C).$$

The geometric relationship between these two bonds,

namely the bond angle

$$\angle ABC,$$

plays a decisive role in determining

* orbital overlap,
* bond hybridization,
* lattice stability,
* electronic structure,
* mechanical properties.

Traditional message passing treats these two bonds as independent.

ALIGNN instead allows **neighboring bonds to exchange information directly**.

This process is known as **angular message passing**.

It is the defining innovation of the ALIGNN architecture.

---

# 17.7.1 From Two-Body to Three-Body Interactions

Classical graph neural networks model interactions between

two atoms.

Mathematically,

the message is

$$\boxed{(i,j).}$$

These are called **two-body interactions**.

Many physical systems, however, are governed by

three-body interactions,

which involve

$$(i,j,k).$$

Examples include

* bond-angle potentials,
* covalent bonding,
* hybridization,
* crystal field effects,
* lattice distortions,
* phonon interactions.

Instead of depending solely on

the distance

$$r_{ij},$$

the energy may depend on

$$\boxed{(r_{ij}, r_{jk}, \theta_{ijk}).}$$

The bond angle

becomes an independent variable.

This additional geometric information cannot be represented explicitly by conventional graph convolutions.

---

# 17.7.2 Physical Motivation

Consider the Stillinger–Weber potential,

which is widely used for silicon.

The total energy consists of

a two-body term

plus

a three-body term,

$$\boxed{E = \sum_{i<j} V_2(r_{ij}) + \sum_{i<j<k} V_3(r_{ij},r_{ik},\theta_{jik}).}$$

Notice that

the second term depends directly on

the bond angle.

If a machine learning model ignores

$$\theta,$$

it cannot easily reproduce such a potential.

ALIGNN was designed specifically to learn these higher-order interactions.

---

# 17.7.3 Bond-Centered Learning

Traditional graph neural networks are

**atom-centered**.

The computational graph looks like

```text
Atom

↓

Neighbor Atom

↓

Neighbor Atom

```

ALIGNN introduces an additional perspective.

It also becomes

**bond-centered**.

The computational graph now becomes

```text
Bond

↓

Neighbor Bond

↓

Neighbor Bond

```

Each bond exchanges information

with neighboring bonds.

The neighborhood of a bond is defined through

the line graph.

---

# 17.7.4 The Line Graph as a Communication Network

Recall that

the line graph contains

* bonds as nodes,
* shared atoms as edges.

Consider

```text
A

 \

  B

 /

C

```

Crystal graph

```text
(A,B)

(B,C)

```

Line graph

```text
(A,B)

   |

(B,C)

```

Message passing now occurs

between

the two bonds.

Instead of atoms communicating,

the **bonds themselves communicate**.

This allows the network to learn

how neighboring bonds influence one another.

---

# 17.7.5 Bond Features Become Messages

Suppose

bond

$$(A,B)$$

possesses feature vector

$$\mathbf{e}_{AB},$$

while

bond

$$(B,C)$$

possesses

$$
\mathbf{e}_{BC}.
$$

Angular message passing computes

a message

$$
\boxed{
\mathbf{m}_{AB\rightarrow BC}
=============================

\phi
(
\mathbf{e}*{AB},
\mathbf{e}*{BC}
),
}
$$

where

$\phi$

is a learnable neural network.

Unlike conventional message passing,

no atomic features appear explicitly.

Instead,

the message depends on

neighboring bonds.

---

# 17.7.6 Why Neighboring Bonds Matter

Imagine two silicon bonds.

Initially,

their feature vectors contain only

distance information.

```text
Bond 1

↓

2.35 Å

──────────────

Bond 2

↓

2.35 Å
```

After angular message passing,

each bond also learns

* neighboring bond lengths,
* local geometry,
* coordination environment,
* bond orientation,
* local symmetry.

Consequently,

the updated bond representation

contains substantially richer structural information.

---

# 17.7.7 General Message Function

Suppose

bond

$i$

communicates with neighboring bond

$j$

inside the line graph.

A general message is

$$
\boxed{
\mathbf{m}_{ij}
===============

\phi_m
(
\mathbf{e}_i,
\mathbf{e}_j
),
}
$$

where

* $\mathbf{e}_i$ denotes one bond,
* $\mathbf{e}_j$ denotes a neighboring bond.

The message function

is usually implemented

as

a multilayer perceptron (MLP).

---

# 17.7.8 Angular Aggregation

Each bond receives

messages

from every neighboring bond.

Suppose

bond

$i$

has neighbors

$$
\mathcal N(i).
$$

The aggregated angular information becomes

$$
\boxed{
\mathbf{a}_i
============

\sum_{j\in\mathcal N(i)}
\mathbf{m}_{ij}.
}
$$

This aggregation resembles

ordinary graph convolution,

except

that it occurs on

the line graph

rather than

the crystal graph.

---

# 17.7.9 Updating Bond Representations

Once

the neighboring bond information

has been aggregated,

the bond feature is updated.

A generic update rule is

$$
\boxed{
\mathbf{e}_i'
=============

\phi_u
(
\mathbf{e}_i,
\mathbf{a}_i
).
}
$$

Thus,

every bond

gradually learns

about

its surrounding bond network.

After multiple ALIGNN layers,

the feature vector of a single bond

contains information from

an increasingly large angular neighborhood.

---

# 17.7.10 Alternating Message Passing

One of ALIGNN's most elegant ideas is that

message passing alternates

between

the two graphs.

Instead of performing

many updates

only on atoms,

ALIGNN alternates between

1. Line graph updates
2. Crystal graph updates

Conceptually,

the computation proceeds as

```text
Crystal Graph

↓

Update Atoms

↓

Updated Bonds

↓

Line Graph

↓

Update Bonds

↓

Updated Atomic Messages

↓

Crystal Graph

↓

Repeat
```

Thus,

information continuously flows

between

atoms

and

bonds.

This alternating strategy allows atomic and angular information to reinforce one another throughout the network.

---

# 17.7.11 Physical Interpretation

Consider a silicon atom.

Initially,

its feature vector contains

only

basic chemical information.

After one crystal-graph update,

the atom learns about

its neighboring atoms.

Next,

the line graph updates

the surrounding bonds.

These updated bond features now encode

* bond angles,
* local coordination,
* geometric distortions.

When information returns

to the crystal graph,

the atom receives

not only

neighboring atomic information,

but also

knowledge of

the surrounding angular geometry.

Consequently,

the atomic representation becomes significantly richer than in conventional graph neural networks.

---

# 17.7.12 Relation to Three-Body Physics

Angular message passing provides a natural approximation to

three-body interactions.

Instead of explicitly computing

every possible triplet,

ALIGNN lets

neighboring bonds exchange learned messages.

As a result,

the model can represent

physical effects arising from

triplets of atoms,

including

* tetrahedral coordination,
* octahedral distortions,
* angular strain,
* crystal symmetry,
* orbital orientation.

The network therefore captures

important structural physics

without explicitly hard-coding

analytical angular potential functions.

---

# 17.7.13 Advantages Over Pairwise Message Passing

Compared with conventional graph neural networks, angular message passing offers several important advantages:

1. **Explicit geometric reasoning**
   Bond-angle relationships are represented directly through the line graph rather than inferred indirectly from repeated pairwise updates.

2. **Improved data efficiency**
   Because angular information is encoded in the graph topology, the network requires fewer layers and often less training data to learn geometry-dependent properties.

3. **Better physical inductive bias**
   Many crystalline materials are governed by directional bonding. Angular message passing naturally reflects this underlying physics.

4. **Enhanced predictive performance**
   Explicit modeling of three-body interactions has been shown to improve predictions for formation energies, elastic constants, electronic properties, and other geometry-sensitive quantities.

---

# 17.7.14 Summary

Angular message passing is the defining feature that distinguishes ALIGNN from earlier atomistic graph neural networks.

Rather than relying solely on pairwise atomic interactions, ALIGNN introduces a second communication pathway in which **neighboring bonds exchange information through the line graph**. This enables the network to learn rich three-body geometric relationships while remaining within the familiar framework of message passing.

By alternating updates between the crystal graph and the line graph, ALIGNN continuously refines both atomic and bond representations, allowing geometric information to propagate efficiently throughout the structure.

---

## Looking Ahead

We have now examined the two fundamental computational components of ALIGNN:

* **Edge-Gated Graph Convolution**, which updates atomic and bond features on the crystal graph, and
* **Angular Message Passing**, which updates bond features on the line graph.

The next step is to combine these ideas into the complete **ALIGNN architecture**. We will examine how individual interaction blocks are organized, how information flows through successive layers, how residual connections and normalization stabilize training, and how the network ultimately predicts material properties from crystal structures.

# 17.8 The ALIGNN Architecture

At this point, we have developed all of the fundamental ideas required to understand ALIGNN.

We have learned

* how a crystal is converted into a graph,
* how a line graph is constructed,
* how edge-gated graph convolutions update atomic and bond features,
* how angular message passing allows neighboring bonds to communicate.

The next step is to combine these individual components into one complete neural network architecture.

This section examines the complete ALIGNN model exactly as it operates during training and inference. Rather than viewing the crystal graph and line graph as separate entities, we will see that ALIGNN treats them as **two coupled computational graphs** whose representations evolve together through repeated interaction blocks.

---

# 17.8.1 High-Level View of ALIGNN

Unlike CGCNN, which performs message passing only on atoms, ALIGNN alternates between updates on the **line graph** and the **crystal graph**.

A simplified view of the network is

```text id="alignn_arch_1"
Crystal Structure

        │

        ▼

Crystal Graph

        │

        ▼

Line Graph

        │

        ▼

ALIGNN Block 1

        │

        ▼

ALIGNN Block 2

        │

        ▼

ALIGNN Block 3

        │

       ...

        │

        ▼

Readout Layer

        │

        ▼

Prediction
```

Each ALIGNN block refines

* atomic features,
* bond features,

using information exchanged between the two graphs.

---

# 17.8.2 Inputs to the Network

Every ALIGNN model receives four primary inputs.

### Node Features

Each atom possesses an initial feature vector

$$
\mathbf{h}_i^{(0)}.
$$

These vectors are usually obtained through an embedding layer applied to the atomic numbers.

---

### Edge Features

Each bond possesses an initial feature vector

$$
\mathbf{e}_{ij}^{(0)}.
$$

These features typically include

* radial basis expansion of distance,
* displacement information,
* periodic image information.

---

### Crystal Graph

The crystal graph specifies

* atoms,
* neighboring atoms,
* connectivity.

---

### Line Graph

The line graph specifies

* bonds,
* neighboring bonds,
* angular connectivity.

Together,

these four components completely describe the local atomic geometry.

---

# 17.8.3 Initial Embedding Layers

The raw inputs are not directly suitable for deep learning.

Instead,

they are projected into a common hidden space.

For atomic features,

$$
\boxed{
\mathbf{h}_i^{(0)}
==================

\phi_h
(Z_i),
}
$$

where

* $Z_i$ is the atomic number,
* $\phi_h$ denotes the embedding layer.

Similarly,

edge features are transformed as

$$
\boxed{
\mathbf{e}_{ij}^{(0)}
=====================

\phi_e
(r_{ij}),
}
$$

where

$r_{ij}$

may first be expanded using radial basis functions before entering the embedding network.

The embedding dimension is typically between

64

and

256,

depending on the model configuration.

---

# 17.8.4 The ALIGNN Interaction Block

The **ALIGNN interaction block** is the fundamental building block of the network.

Unlike conventional graph neural networks that update only node features, an ALIGNN block performs **two sequential graph convolutions**:

1. **Line graph convolution**
2. **Crystal graph convolution**

A single interaction block can therefore be represented as

```text id="alignn_arch_2"
Bond Features

        │

        ▼

Line Graph Convolution

        │

Updated Bond Features

        │

        ▼

Crystal Graph Convolution

        │

Updated Atomic Features
```

This alternating update allows angular information to influence atomic representations during every layer.

---

# 17.8.5 Step 1: Line Graph Update

The first operation in each ALIGNN block is message passing on the line graph.

Each node in the line graph corresponds to a bond.

Neighboring bond features exchange information,

producing updated bond representations.

Mathematically,

$$
\boxed{
\mathbf{e}^{(l+\frac12)}
========================

f_{\mathrm{line}}
(
\mathbf{e}^{(l)}
),
}
$$

where

$f_{\mathrm{line}}$

denotes the edge-gated convolution operating on the line graph.

Notice that

only bond features are updated during this stage.

Atomic features remain unchanged.

---

# 17.8.6 Step 2: Crystal Graph Update

Once the bond representations have been improved,

the crystal graph convolution begins.

The updated bond features now participate in atomic message passing.

The atomic update becomes

$$
\boxed{
\mathbf{h}^{(l+1)}
==================

f_{\mathrm{atom}}
(
\mathbf{h}^{(l)},
\mathbf{e}^{(l+\frac12)}
).
}
$$

Compared with conventional graph neural networks,

this update is considerably richer,

because every bond now contains information about

its surrounding bond angles.

Consequently,

atomic representations become increasingly geometry-aware.

---

# 17.8.7 Information Flow Through One ALIGNN Block

The information flow within a single interaction block can be summarized as follows.

```text id="alignn_arch_3"
Bond Features

        │

        ▼

Angular Message Passing

(Line Graph)

        │

Updated Bond Features

        │

        ▼

Atomic Message Passing

(Crystal Graph)

        │

Updated Atom Features
```

Notice that

information always flows

from

line graph

to

crystal graph.

This ordering ensures that angular information is incorporated before atomic features are updated.

---

# 17.8.8 Stacking Multiple ALIGNN Blocks

One interaction block captures only a limited neighborhood.

To learn long-range structural information,

multiple ALIGNN blocks are stacked sequentially.

Suppose

the network contains

$L$

blocks.

The computation becomes

$$
\boxed{
(\mathbf{H}^{(0)},\mathbf{E}^{(0)})
\rightarrow
(\mathbf{H}^{(1)},\mathbf{E}^{(1)})
\rightarrow
\cdots
\rightarrow
(\mathbf{H}^{(L)},\mathbf{E}^{(L)}).
}
$$

Each successive layer enlarges the receptive field.

After

$L$

blocks,

every atom has indirectly received information from atoms several hops away,

including their angular environments.

---

# 17.8.9 Residual Connections

Deep graph neural networks often suffer from

* vanishing gradients,
* over-smoothing,
* optimization difficulties.

To alleviate these problems,

ALIGNN employs **residual connections**.

Instead of learning an entirely new representation,

the layer learns only a correction.

The update becomes

$$
\boxed{
\mathbf{h}^{(l+1)}
==================

\mathbf{h}^{(l)}
+
F
(
\mathbf{h}^{(l)}
),
}
$$

where

$F$

represents the operations performed by the interaction block.

Residual learning offers several advantages:

* easier optimization,
* improved gradient flow,
* deeper architectures,
* faster convergence.

The same principle is applied to edge representations.

---

# 17.8.10 Normalization Layers

During training,

feature magnitudes may grow or shrink unpredictably.

Normalization stabilizes the optimization process.

ALIGNN typically employs **Layer Normalization** after major computational blocks.

Given an input feature vector

$\mathbf{x}$,

layer normalization computes

$$
\boxed{
\mathrm{LayerNorm}(\mathbf{x})
==============================

\gamma
\frac{
\mathbf{x}
----------

\mu
}
{
\sqrt{\sigma^2+\varepsilon}
}
+
\beta,
}
$$

where

* $\mu$ is the feature mean,
* $\sigma^2$ is the feature variance,
* $\gamma$ and $\beta$ are learnable parameters.

Normalization improves numerical stability and accelerates convergence.

---

# 17.8.11 Nonlinear Activation Functions

After each linear transformation,

a nonlinear activation function is applied.

Common choices include

* ReLU,
* SiLU (Swish),
* GELU.

The original ALIGNN implementation primarily uses the **SiLU** activation,

defined as

$$
\boxed{
\mathrm{SiLU}(x)
================

x
,\sigma(x),
}
$$

where

$\sigma(x)$

is the sigmoid function.

Compared with ReLU,

SiLU provides smoother gradients,

which often improve optimization in atomistic neural networks.

---

# 17.8.12 Readout Layer

After the final ALIGNN block,

every atom possesses a learned feature vector

$$
\mathbf{h}_i^{(L)}.
$$

To predict a material property,

the atomic representations must be combined into a single graph-level representation.

This process is called **readout**.

A common readout operation is

global sum pooling,

$$
\boxed{
\mathbf{h}_{\mathrm{graph}}
===========================

\sum_{i=1}^{N}
\mathbf{h}_i^{(L)}.
}
$$

Alternative pooling strategies include

* mean pooling,
* max pooling,
* attention pooling,
* set2set pooling.

The resulting graph representation is then passed through one or more fully connected layers to produce the final prediction.

---

# 17.8.13 Prediction Head

The final multilayer perceptron maps the graph representation to the desired property.

For regression tasks,

such as formation energy prediction,

the output is

$$
\boxed{
\hat{y}
=======

f_{\mathrm{MLP}}
(
\mathbf{h}_{\mathrm{graph}}
).
}
$$

For classification tasks,

such as metal versus semiconductor prediction,

the output becomes

$$
\boxed{
\hat{\mathbf{p}}
================

\mathrm{Softmax}
(
f_{\mathrm{MLP}}
(
\mathbf{h}_{\mathrm{graph}}
)
).
}
$$

Thus,

the same ALIGNN encoder can be used for both regression and classification by simply changing the prediction head and loss function.

---

# 17.8.14 Complete Forward Pass

The complete forward propagation through ALIGNN can now be summarized as

```text id="alignn_arch_4"
Crystal Structure

        │

        ▼

Node Embedding

Edge Embedding

        │

        ▼

Crystal Graph

+

Line Graph

        │

        ▼

ALIGNN Block × L

        │

        ▼

Graph Readout

        │

        ▼

MLP

        │

        ▼

Material Property
```

This modular architecture makes ALIGNN both expressive and flexible, enabling it to model complex geometry-dependent interactions while remaining compatible with standard graph deep learning frameworks.

---

# 17.8.15 Why the Architecture Works

The strength of ALIGNN lies in the interplay between its two graphs.

* The **crystal graph** captures pairwise atomic connectivity.
* The **line graph** captures angular relationships between neighboring bonds.

Within each interaction block, information alternates between these two representations. As a result, every updated atomic feature reflects not only the identities and distances of neighboring atoms but also the geometry of the surrounding bond network.

This dual-graph design provides a strong physical inductive bias for crystalline materials, allowing ALIGNN to achieve high accuracy across a wide range of property prediction tasks.

---

## Looking Ahead

Now that we have established the complete ALIGNN architecture, we are ready to examine its mathematical foundations in greater detail. In the next section, **Mathematical Formulation of ALIGNN**, we will derive the equations governing edge-gated graph convolution, line graph message passing, residual updates, and the complete forward propagation through the network. This section will connect the intuitive architectural concepts developed so far with the rigorous mathematical framework used in the original ALIGNN model.


# 17.9 Mathematical Formulation of ALIGNN

The previous sections introduced the intuition behind ALIGNN and described its architecture at a conceptual level. While these explanations provide an understanding of how the model operates, they do not fully reveal the mathematical principles governing the network.

In this section, we derive the mathematical formulation of ALIGNN from first principles. We will begin with the graph representation of a crystal, formulate edge-gated graph convolution, derive the line graph message-passing equations, and finally combine these operations into the complete ALIGNN forward propagation.

Unlike many research papers that present equations with minimal explanation, our goal is to explain **what every symbol represents, why each equation is needed, and how each mathematical operation relates to the underlying physics of crystalline materials.**

---

# 17.9.1 Mathematical Representation of the Crystal Graph

Consider a crystal graph

$$
\boxed{
G=(V,E),
}
$$

where

* $V$ is the set of atoms (nodes),
* $E$ is the set of bonds (edges).

Suppose

$$
N=|V|
$$

atoms are present.

Each atom possesses a feature vector

$$
\boxed{
\mathbf{h}_i\in\mathbb{R}^{d_h},
}
$$

where

* $i$ denotes the atom index,
* $d_h$ is the node feature dimension.

Collecting all node features gives the node feature matrix

$$
\boxed{
\mathbf{H}
==========

[\mathbf{h}_1,\mathbf{h}_2,\ldots,\mathbf{h}_N]^T
\in
\mathbb{R}^{N\times d_h}.
}
$$

Similarly,

suppose the graph contains

$$
M=|E|
$$

edges.

Each edge possesses a feature vector

$$
\boxed{
\mathbf{e}_{ij}
\in
\mathbb{R}^{d_e},
}
$$

where

$d_e$

is the edge feature dimension.

The edge feature matrix is therefore

$$
\boxed{
\mathbf{E}
\in
\mathbb{R}^{M\times d_e}.
}
$$

Thus,

the input to ALIGNN consists of

$$
(\mathbf{H},\mathbf{E}).
$$

---

# 17.9.2 Neighbor Set

For every atom

$i$,

define its neighboring atoms as

$$
\boxed{
\mathcal N(i)
=============

{
j
\mid
(i,j)\in E
}.
}
$$

The neighborhood determines

which atoms participate in message passing.

Only neighboring atoms contribute to the update of

atom

$i$.

This locality reflects the physical principle that atomic interactions are primarily determined by nearby atoms.

---

# 17.9.3 Initial Node Embedding

Each atom is identified by its atomic number

$$
Z_i.
$$

Rather than using

one-hot vectors,

ALIGNN learns a continuous embedding

$$
\boxed{
\mathbf{h}_i^{(0)}
==================

\mathrm{Embedding}(Z_i).
}
$$

The embedding layer is simply a learnable lookup table,

transforming

the discrete atomic number

into

a dense feature vector.

For example,

$$
14
\longrightarrow
[-0.31,;0.58,;\ldots,;1.12].
$$

The network learns these embeddings automatically during training.

---

# 17.9.4 Initial Edge Embedding

Each bond is characterized by its interatomic distance

$$
r_{ij}.
$$

Rather than using

the raw scalar distance,

ALIGNN first applies a nonlinear embedding,

typically using radial basis functions,

followed by a linear transformation,

$$
\boxed{
\mathbf{e}_{ij}^{(0)}
=====================

\phi_r(r_{ij}).
}
$$

Here,

$\phi_r$

represents the distance embedding function.

This transformation maps a single scalar

into a high-dimensional vector,

making it easier for the neural network to learn nonlinear distance-dependent interactions.

---

# 17.9.5 Message Construction

Suppose

atom

$j$

sends information

to

atom

$i$.

The message depends on

* sender atom,
* receiver atom,
* connecting bond.

The general message function is

$$
\boxed{
\mathbf{m}_{ij}
===============

\phi_m
(
\mathbf{h}_i,
\mathbf{h}*j,
\mathbf{e}*{ij}
),
}
$$

where

$\phi_m$

is implemented

as a multilayer perceptron (MLP).

Unlike classical GNNs,

the bond feature explicitly participates in constructing the message.

Consequently,

the transmitted information depends on both chemistry and geometry.

---

# 17.9.6 Edge Gate

Not every bond contributes equally.

ALIGNN therefore computes

a gate

for every edge.

The gate is

$$
\boxed{
g_{ij}
======

\sigma
(
W_g
\mathbf{e}_{ij}
+
b_g
),
}
$$

where

* $W_g$ is a learnable weight matrix,
* $b_g$ is the bias,
* $\sigma(\cdot)$ denotes the sigmoid activation.

Since

$$
0<g_{ij}<1,
$$

the gate acts as an adaptive importance coefficient.

Large values indicate

important interactions,

while small values suppress less informative messages.

---

# 17.9.7 Gated Message

The gated message becomes

$$
\boxed{
\tilde{\mathbf m}_{ij}
======================

g_{ij}
\mathbf m_{ij}.
}
$$

This equation is one of the defining characteristics of edge-gated graph convolution.

Instead of treating every neighboring interaction equally,

ALIGNN learns how strongly each bond should influence its neighboring atoms.

---

# 17.9.8 Message Aggregation

The receiving atom

collects messages

from all neighbors.

Aggregation is performed using

summation,

$$
\boxed{
\mathbf m_i
===========

\sum_{j\in\mathcal N(i)}
\tilde{\mathbf m}_{ij}.
}
$$

The sum operator possesses an important property:

it is permutation invariant.

Consequently,

changing the order of neighboring atoms

does not affect the prediction.

This is an essential requirement for physical systems,

because atomic indexing has no physical meaning.

---

# 17.9.9 Node Update Equation

After aggregation,

the node representation is updated.

A general update equation is

$$
\boxed{
\mathbf h_i'
============

\phi_h
(
\mathbf h_i,
\mathbf m_i
).
}
$$

Most implementations realize

$\phi_h$

using

linear layers,

activation functions,

and residual connections.

This updated feature now contains

information from

both the atom itself

and

its neighboring atomic environment.

---

# 17.9.10 Edge Update Equation

Unlike conventional graph neural networks,

ALIGNN also updates edge features.

The new bond representation is

$$
\boxed{
\mathbf e_{ij}'
===============

\phi_e
(
\mathbf h_i,
\mathbf h_j,
\mathbf e_{ij}
).
}
$$

Notice that

the edge update depends on

both endpoint atoms.

Thus,

bond representations evolve together with atomic representations throughout training.

---

# 17.9.11 Line Graph Message Passing

The line graph contains

bonds as nodes.

Suppose

bond

$a$

and

bond

$b$

share a common atom.

They become neighbors

inside the line graph.

The message between bonds is

$$
\boxed{
\mathbf m_{ab}^{L}
==================

\phi_L
(
\mathbf e_a,
\mathbf e_b
).
}
$$

Here,

$\phi_L$

is another neural network.

Unlike the crystal graph,

only bond representations participate.

---

# 17.9.12 Angular Aggregation

Each bond receives

messages from neighboring bonds,

$$
\boxed{
\mathbf a_i
===========

\sum_{j\in\mathcal N_L(i)}
\mathbf m_{ij}^{L},
}
$$

where

$\mathcal N_L(i)$

denotes

the neighboring bonds

inside the line graph.

The resulting vector

contains information about

the local angular environment.

---

# 17.9.13 Bond Update on the Line Graph

The updated bond representation becomes

$$
\boxed{
\mathbf e_i^{L}
===============

\phi_u
(
\mathbf e_i,
\mathbf a_i
).
}
$$

These updated bond features

are subsequently returned

to

the crystal graph,

where they participate

in the next atomic update.

This alternating computation is

the defining feature

of ALIGNN.

---

# 17.9.14 Residual Learning

To facilitate optimization,

ALIGNN employs residual learning.

Instead of replacing

the previous representation,

each layer learns only a correction,

$$
\boxed{
\mathbf h^{(l+1)}
=================

\mathbf h^{(l)}
+
F_h
(
\mathbf h^{(l)}
).
}
$$

Similarly,

for edge features,

$$
\boxed{
\mathbf e^{(l+1)}
=================

\mathbf e^{(l)}
+
F_e
(
\mathbf e^{(l)}
).
}
$$

Residual learning improves

gradient flow,

accelerates convergence,

and enables much deeper architectures.

---

# 17.9.15 Layer Normalization

After every major update,

layer normalization is applied,

$$
\boxed{
\mathrm{LayerNorm}
(
\mathbf x
)
=

\gamma
\frac{
\mathbf x-\mu
}
{
\sqrt{\sigma^2+\varepsilon}
}
+
\beta.
}
$$

Normalization stabilizes feature distributions,

reduces internal covariate shift,

and improves training stability.

---

# 17.9.16 Readout Function

After

$L$

ALIGNN blocks,

each atom possesses

its final representation,

$$
\mathbf h_i^{(L)}.
$$

To obtain

a graph-level representation,

ALIGNN performs

global pooling,

$$
\boxed{
\mathbf h_G
===========

\sum_{i=1}^{N}
\mathbf h_i^{(L)}.
}
$$

Alternative pooling strategies include

* mean pooling,
* max pooling,
* attention pooling.

For most property prediction tasks,

sum pooling performs remarkably well because many extensive properties (such as total energy) scale naturally with system size.

---

# 17.9.17 Final Prediction

The graph representation is passed through a multilayer perceptron,

$$
\boxed{
\hat y
======

f_{\mathrm{MLP}}
(
\mathbf h_G
).
}
$$

For regression,

$\hat y$

is a continuous value,

such as

* formation energy,
* band gap,
* elastic modulus.

For classification,

the prediction becomes

$$
\boxed{
\hat{\mathbf p}
===============

\mathrm{Softmax}
(
f_{\mathrm{MLP}}
(
\mathbf h_G
)
).
}
$$

---

# 17.9.18 Complete Mathematical Pipeline

The complete forward propagation through ALIGNN can now be summarized mathematically as

$$
\boxed{
\begin{aligned}
Z_i
&\longrightarrow
\mathbf h_i^{(0)}
[4pt]
r_{ij}
&\longrightarrow
\mathbf e_{ij}^{(0)}
[4pt]
\mathbf e^{(l)}
&\longrightarrow
\text{Line Graph Update}
[4pt]
\mathbf h^{(l)}
&\longrightarrow
\text{Crystal Graph Update}
[4pt]
\mathbf h^{(L)}
&\longrightarrow
\text{Readout}
[4pt]
\mathbf h_G
&\longrightarrow
\text{MLP}
[4pt]
&\longrightarrow
\hat y.
\end{aligned}
}
$$

This sequence captures the entire ALIGNN computation, from raw crystal structure to the predicted material property.

---

# 17.9.19 Interpretation of the Mathematical Framework

The equations presented above reveal why ALIGNN is more expressive than earlier graph neural networks.

At every layer, the model simultaneously learns

* **atomic representations**, which describe the local chemical environment,
* **bond representations**, which encode pairwise interactions, and
* **angular representations**, which capture three-body geometric relationships through the line graph.

By alternating message passing between the crystal graph and the line graph, ALIGNN continuously couples chemistry and geometry into a unified latent representation. This dual-graph mathematical framework allows the network to model complex structure–property relationships that are difficult or impossible to capture using purely pairwise graph convolutions.

---

## Looking Ahead

With the mathematical foundations now established, the next section will transition from theory to implementation. We will build **ALIGNN step by step in PyTorch**, beginning with graph preprocessing, constructing crystal and line graphs, implementing the edge-gated convolution layer, assembling complete ALIGNN blocks, and finally training the network on real materials datasets such as the JARVIS-DFT database. This implementation-oriented section will bridge the gap between the mathematical derivations developed in this chapter and practical research applications.

# 17.10 PyTorch Implementation of ALIGNN

So far, we have studied ALIGNN from a theoretical perspective. We have examined the crystal graph, line graph, edge-gated graph convolution, angular message passing, and the mathematical formulation of the architecture.

Understanding the mathematics, however, is only the first step.

A materials informatics researcher must also know **how ALIGNN is implemented in practice**.

Fortunately, ALIGNN follows the same philosophy as most modern deep learning models:

> **Complex architectures are built from a small number of simple PyTorch modules.**

In this section, we will implement ALIGNN step by step.

Rather than immediately presenting the complete implementation, we will build the network incrementally.

By the end of this section, the reader will understand

* how crystal graphs are represented,
* how line graphs are constructed,
* how edge-gated convolutions are implemented,
* how ALIGNN blocks are assembled,
* how the complete model is built.

The implementation presented here closely follows the original ALIGNN architecture while remaining easy to understand and modify for research purposes.

---

# 17.10.1 Required Libraries

ALIGNN relies on several Python libraries.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

import numpy as np
```

Here,

* **PyTorch** performs tensor computations.
* **DGL (Deep Graph Library)** manages crystal graphs and line graphs.
* **NumPy** assists with preprocessing.

Although PyTorch Geometric can also be used, the original ALIGNN implementation was developed using DGL.

---

# 17.10.2 Representing the Crystal Graph

Suppose a crystal contains

four atoms.

The graph can be represented as

```text
Atom 0 ----- Atom 1

   |            |

   |            |

Atom 2 ----- Atom 3
```

In DGL,

the graph is created from two tensors.

```python
src = torch.tensor([0, 0, 1, 2])
dst = torch.tensor([1, 2, 3, 3])

graph = dgl.graph((src, dst))
```

Every edge corresponds to one neighboring atomic interaction.

---

# 17.10.3 Node Features

Suppose

each atom possesses

64-dimensional features.

```python
graph.ndata["h"] = torch.randn(
    graph.num_nodes(),
    64
)
```

The resulting tensor has shape

```text
(number_of_atoms, 64)
```

For example,

```text
(32, 64)
```

for a crystal containing

32 atoms.

---

# 17.10.4 Edge Features

Similarly,

each bond possesses

64-dimensional edge features.

```python
graph.edata["e"] = torch.randn(
    graph.num_edges(),
    64
)
```

Now both

atoms

and

bonds

contain learnable representations.

---

# 17.10.5 Constructing the Line Graph

One of the most elegant aspects of DGL is that

the line graph can be generated automatically.

```python
line_graph = dgl.line_graph(
    graph,
    backtracking=False
)
```

This single command performs all of the graph-theoretic operations discussed earlier.

Specifically,

it

* converts edges into nodes,
* connects neighboring bonds,
* preserves graph connectivity.

Thus,

the implementation directly mirrors the mathematical definition of a line graph.

---

# 17.10.6 Atomic Embedding Layer

The first learnable component of ALIGNN is the atomic embedding.

```python
class AtomEmbedding(nn.Module):

    def __init__(self,
                 num_elements,
                 hidden_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            num_elements,
            hidden_dim
        )

    def forward(self, atomic_number):

        return self.embedding(
            atomic_number
        )
```

Suppose

the dataset contains

100 elements.

```python
atom_embed = AtomEmbedding(
    100,
    64
)
```

Each atomic number is transformed into

a 64-dimensional feature vector.

---

# 17.10.7 Edge Embedding Layer

Edge features are projected into the hidden space using a linear layer.

```python
class EdgeEmbedding(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim):

        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            hidden_dim
        )

    def forward(self, edge_feature):

        return self.linear(edge_feature)
```

For example,

radial basis function features of dimension

80

may be transformed into

64

hidden features.

---

# 17.10.8 Why Separate Embeddings?

Notice that

ALIGNN uses

two independent embedding networks.

```text
Atomic Number

↓

Atom Embedding

↓

Atom Features

──────────────

Distance Features

↓

Edge Embedding

↓

Bond Features
```

Atoms

and

bonds

contain fundamentally different information.

Therefore,

they require different neural networks.

---

# 17.10.9 Edge-Gated Convolution Module

We now implement

the edge-gated convolution.

First,

define the neural network.

```python
class EdgeGatedConv(nn.Module):

    def __init__(self,
                 hidden_dim):

        super().__init__()

        self.edge_gate = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.node_update = nn.Linear(
            hidden_dim,
            hidden_dim
        )
```

This module will later compute

* gate values,
* messages,
* node updates.

---

# 17.10.10 Computing the Gate

Inside

the forward function,

the gate becomes

```python
gate = torch.sigmoid(
    self.edge_gate(edge_feature)
)
```

Recall

our mathematical formulation,

$$
g_{ij}
======

\sigma
(
W_g e_{ij}
+b
).
$$

The PyTorch implementation is therefore

almost identical

to

the mathematical equation.

---

# 17.10.11 Message Passing

Messages are multiplied

by

their gates.

```python
message = gate * edge_feature
```

Each neighboring bond now contributes

according to

its learned importance.

This simple multiplication is

one of the defining operations

of ALIGNN.

---

# 17.10.12 Aggregating Messages

DGL performs message aggregation automatically.

```python
graph.edata["m"] = message

graph.update_all(

    fn.copy_e("m", "m"),

    fn.sum("m", "agg")

)
```

After this operation,

each node contains

```python
graph.ndata["agg"]
```

which stores

the aggregated message from all neighboring edges.

Notice that the implementation requires only two DGL primitives:

* `copy_e`, which sends edge messages to connected nodes.
* `sum`, which aggregates incoming messages.

DGL handles the underlying graph traversal efficiently, allowing the code to remain concise while scaling to very large crystal graphs.

---

# 17.10.13 Updating Node Features

The aggregated message is combined with the existing node representation.

```python
new_node = self.node_update(

    graph.ndata["agg"]

)
```

A residual connection is then applied.

```python
new_node = new_node + graph.ndata["h"]
```

Residual learning improves optimization and enables deeper ALIGNN models.

---

# 17.10.14 Complete Edge-Gated Convolution Layer

Putting the individual steps together,

the forward function becomes

```python
def forward(

    self,

    graph,

    node_feature,

    edge_feature

):

    gate = torch.sigmoid(

        self.edge_gate(edge_feature)

    )

    message = gate * edge_feature

    graph.ndata["h"] = node_feature

    graph.edata["m"] = message

    graph.update_all(

        fn.copy_e("m", "m"),

        fn.sum("m", "agg")

    )

    new_node = self.node_update(

        graph.ndata["agg"]

    )

    new_node = new_node + node_feature

    return new_node
```

Although simplified for clarity, this implementation captures the essential idea of edge-gated message passing.

---

# 17.10.15 Implementing the Line Graph Update

The same edge-gated convolution can be applied to the line graph.

Instead of updating atoms,

the nodes of the line graph correspond to **bonds**.

```python
updated_edge = self.line_conv(

    line_graph,

    edge_feature,

    angle_feature

)
```

Conceptually,

nothing changes.

Only the meaning of

the graph nodes

is different.

This elegant symmetry

is one reason why ALIGNN remains computationally efficient.

---

# 17.10.16 Building an ALIGNN Block

An ALIGNN block combines

1. Line graph convolution.
2. Crystal graph convolution.

```python
class ALIGNNBlock(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.line_conv = EdgeGatedConv(

            hidden_dim

        )

        self.atom_conv = EdgeGatedConv(

            hidden_dim

        )
```

The forward pass proceeds in two stages:

1. update bond features using the line graph,
2. update atom features using the crystal graph.

This alternating update is the defining characteristic of the ALIGNN architecture.

---

# 17.10.17 Building the Complete Model

The full ALIGNN network is created by stacking multiple interaction blocks.

```python
self.blocks = nn.ModuleList(

    [

        ALIGNNBlock(hidden_dim)

        for _ in range(num_layers)

    ]

)
```

During the forward pass,

each block is applied sequentially,

allowing information to propagate through progressively larger atomic neighborhoods.

---

# 17.10.18 Practical Implementation Considerations

While the simplified implementation presented above captures the essential ideas, production implementations include several additional components:

* **Residual connections** after both atom and bond updates.
* **Layer normalization** to stabilize training.
* **Dropout** for regularization.
* **Radial basis function (RBF) expansions** for distance embeddings.
* **Efficient batching** of multiple crystal graphs.
* **Graph readout layers** for property prediction.

These enhancements improve numerical stability, training speed, and predictive performance without changing the underlying computational principles.

---

# 17.10.19 Summary

The implementation of ALIGNN demonstrates an important lesson:

Although the model appears mathematically sophisticated, its code is composed of a relatively small number of reusable building blocks:

* embedding layers,
* edge-gated graph convolutions,
* line graph operations,
* residual connections,
* graph pooling,
* fully connected prediction heads.

By combining these components, ALIGNN constructs a powerful dual-graph neural network capable of learning both pairwise and angular interactions in crystalline materials.

---

## Looking Ahead

In the next section, we will move beyond the network architecture and study the **training procedure** for ALIGNN. We will discuss dataset preparation, mini-batch graph loading, loss functions, optimization algorithms, learning-rate scheduling, checkpointing, validation strategies, and practical techniques for training large-scale atomistic graph neural networks on datasets such as JARVIS-DFT and the Materials Project. This will complete the transition from architectural design to a fully trainable research-grade ALIGNN model.


# 17.11 Training ALIGNN

Designing a powerful neural network architecture is only half of the problem. The other half is learning the optimal parameters that allow the network to accurately predict material properties.

Training ALIGNN differs from training ordinary neural networks in several important ways.

Unlike image classification, where every sample has the same dimensions, crystal structures

* contain different numbers of atoms,
* have different numbers of bonds,
* produce graphs of different sizes,
* possess different line graph structures.

Therefore, training an atomistic graph neural network requires specialized data preparation, batching algorithms, and optimization strategies.

In this section, we study the complete ALIGNN training pipeline from raw crystal structures to a fully trained model.

---

# 17.11.1 Overview of the Training Pipeline

Training ALIGNN consists of several stages.

```text id="alignn_train_1"
Crystal Dataset

        │

        ▼

Read CIF Files

        │

        ▼

Graph Construction

        │

        ▼

Line Graph Construction

        │

        ▼

Mini-batching

        │

        ▼

Forward Pass

        │

        ▼

Loss Computation

        │

        ▼

Backpropagation

        │

        ▼

Parameter Update

        │

        ▼

Repeat
```

Each stage plays a critical role in the final model performance.

---

# 17.11.2 Preparing the Dataset

ALIGNN requires three categories of information for every material.

1. Crystal structure
2. Material property
3. Atomic species

For example,

| Material | Structure | Target           |
| -------- | --------- | ---------------- |
| Silicon  | CIF       | Band gap         |
| Copper   | CIF       | Formation energy |
| Diamond  | CIF       | Bulk modulus     |

The crystal structure is usually stored in

* CIF
* POSCAR
* XYZ
* JARVIS format

while the target may be

* formation energy,
* band gap,
* elastic modulus,
* magnetic moment,
* dielectric constant.

---

# 17.11.3 Loading Crystal Structures

The first step is reading the crystal structure.

Using pymatgen,

this is straightforward.

```python
from pymatgen.core import Structure

structure = Structure.from_file(
    "Si.cif"
)
```

The resulting object contains

* lattice vectors,
* atomic coordinates,
* chemical species,
* periodic boundary conditions.

This information is sufficient for graph construction.

---

# 17.11.4 Constructing Graphs

Each crystal is transformed into

two graphs.

```text id="alignn_train_2"
Crystal

↓

Crystal Graph

↓

Line Graph
```

This preprocessing usually occurs

before training begins,

allowing graph generation to be reused across epochs.

Some implementations store the graphs directly on disk to reduce preprocessing overhead during training.

---

# 17.11.5 Creating a PyTorch Dataset

A custom dataset class is typically used.

```python
class CrystalDataset(

    torch.utils.data.Dataset

):

    def __init__(

        self,

        graphs,

        labels

    ):

        self.graphs = graphs

        self.labels = labels

    def __len__(self):

        return len(self.graphs)

    def __getitem__(self, idx):

        return (

            self.graphs[idx],

            self.labels[idx]

        )
```

Each dataset sample returns

* crystal graph,
* line graph,
* target property.

---

# 17.11.6 Mini-Batch Training

Unlike images,

graphs possess different sizes.

Suppose

sample one contains

20 atoms,

while

sample two contains

80 atoms.

Stacking them as ordinary tensors is impossible.

Instead,

DGL batches graphs together.

```python
from dgl.dataloading import GraphDataLoader

loader = GraphDataLoader(

    dataset,

    batch_size=32,

    shuffle=True

)
```

Internally,

DGL merges multiple graphs into

one large disconnected graph.

Message passing remains independent

inside each individual crystal.

---

# 17.11.7 Forward Propagation

The forward pass consists of

four stages.

```text id="alignn_train_3"
Batch Graphs

↓

Embedding

↓

ALIGNN Blocks

↓

Pooling

↓

Prediction
```

PyTorch code is concise.

```python
prediction = model(

    graph,

    line_graph

)
```

Internally,

this performs

hundreds of tensor operations,

but

the user interacts

with a single function call.

---

# 17.11.8 Choosing the Loss Function

The choice of loss depends on

the prediction task.

---

## Regression

Formation energy,

band gap,

bulk modulus,

and elastic constants

are regression problems.

The most common loss is

Mean Squared Error (MSE).

$$
\boxed{
L
=

\frac{1}{N}
\sum_{i=1}^{N}
(
y_i-\hat y_i
)^2.
}
$$

PyTorch implementation

```python
criterion = nn.MSELoss()
```

---

## Mean Absolute Error

Some researchers prefer

Mean Absolute Error.

$$
\boxed{
L
=

\frac{1}{N}
\sum_i
|y_i-\hat y_i|.
}
$$

MAE is

less sensitive

to large outliers.

---

## Classification

For classification,

Cross Entropy Loss is used.

```python
criterion = nn.CrossEntropyLoss()
```

---

# 17.11.9 Computing the Loss

Inside the training loop,

the loss becomes

```python
loss = criterion(

    prediction,

    target

)
```

This scalar quantity measures

how well

the model predicts

the desired property.

The objective of training

is to minimize

this value.

---

# 17.11.10 Backpropagation

Once

the loss has been computed,

PyTorch automatically calculates

all gradients.

```python
loss.backward()
```

This single command performs

automatic differentiation

through

every ALIGNN block,

every graph convolution,

every embedding,

and

every linear layer.

Thousands of partial derivatives

are computed automatically.

---

# 17.11.11 Optimizer

After gradients are available,

the parameters are updated.

The most common optimizer is

Adam.

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3

)
```

The parameter update is

performed using

```python
optimizer.step()
```

After updating,

the gradients are reset.

```python
optimizer.zero_grad()
```

The standard training sequence is therefore

```python
loss.backward()

optimizer.step()

optimizer.zero_grad()
```

---

# 17.11.12 Complete Training Loop

The complete training loop looks as follows.

```python
for epoch in range(num_epochs):

    for graph, target in loader:

        prediction = model(graph)

        loss = criterion(

            prediction,

            target

        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

This loop is repeated

until convergence.

---

# 17.11.13 Learning Rate Scheduling

A fixed learning rate is rarely optimal throughout training.

Initially,

large updates help the optimizer explore the parameter space.

Later,

smaller updates allow the model to converge smoothly.

A learning-rate scheduler automatically adjusts the step size.

For example,

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    factor=0.5,

    patience=10

)
```

After each validation epoch,

the scheduler is updated.

```python
scheduler.step(

    validation_loss

)
```

Adaptive learning-rate scheduling often leads to faster convergence and better final performance.

---

# 17.11.14 Validation

Training loss alone is not sufficient.

A model may achieve extremely low training error while performing poorly on unseen data.

To monitor generalization,

the dataset is divided into

* training set,
* validation set,
* test set.

A common split is

| Split      | Percentage |
| ---------- | ---------: |
| Training   |        80% |
| Validation |        10% |
| Test       |        10% |

The validation set is evaluated after each epoch without updating the model parameters.

---

# 17.11.15 Early Stopping

If the validation loss stops improving,

continuing training may lead to overfitting.

Early stopping prevents unnecessary training.

A simple strategy is

```text
If validation loss

does not improve

for 20 epochs,

stop training.
```

Early stopping

reduces computation,

prevents overfitting,

and usually produces better-performing models.

---

# 17.11.16 Saving the Best Model

During training,

the model with the lowest validation loss should be saved.

```python
torch.save(

    model.state_dict(),

    "best_alignn.pt"

)
```

Later,

the model can be restored.

```python
model.load_state_dict(

    torch.load(

        "best_alignn.pt"

    )

)
```

This ensures that the final deployed model corresponds to the best-performing checkpoint rather than the final training epoch.

---

# 17.11.17 Monitoring Training

Researchers typically record

* training loss,
* validation loss,
* learning rate,
* MAE,
* RMSE,
* epoch time.

These quantities are commonly visualized using TensorBoard, Weights & Biases (W&B), or custom logging scripts.

Tracking these metrics helps diagnose issues such as overfitting, underfitting, or unstable optimization.

---

# 17.11.18 Hyperparameters

Several hyperparameters strongly influence ALIGNN performance.

| Hyperparameter    |         Typical Values |
| ----------------- | ---------------------: |
| Hidden dimension  |                 64–256 |
| ALIGNN blocks     |                    4–8 |
| Batch size        |                  16–64 |
| Learning rate     | $10^{-3}$ to $10^{-4}$ |
| Weight decay      | $10^{-6}$ to $10^{-5}$ |
| Dropout           |                0.0–0.2 |
| Cutoff radius     |                  5–8 Å |
| Maximum neighbors |                  12–20 |

Selecting appropriate values often requires systematic experimentation or automated hyperparameter optimization.

---

# 17.11.19 Practical Challenges

Training ALIGNN on large materials datasets introduces several practical challenges.

### GPU Memory

Line graphs can contain significantly more edges than crystal graphs, increasing memory consumption.

### Data Imbalance

Some material properties exhibit highly skewed distributions, making balanced sampling or target normalization beneficial.

### Long Training Times

Large datasets such as JARVIS-DFT or Materials Project may require many hours or even days of GPU training.

### Reproducibility

Setting random seeds, recording software versions, and saving preprocessing parameters are essential for obtaining reproducible scientific results.

---

# 17.11.20 Summary

Training ALIGNN follows the standard deep learning workflow while incorporating graph-specific preprocessing and dual-graph message passing.

The complete training process consists of

1. Reading crystal structures.
2. Constructing crystal and line graphs.
3. Creating mini-batches.
4. Performing forward propagation through ALIGNN.
5. Computing the loss.
6. Backpropagating gradients.
7. Updating parameters with an optimizer.
8. Monitoring validation performance and saving the best model.

With these components in place, ALIGNN becomes a fully trainable atomistic graph neural network capable of learning accurate structure–property relationships from large materials databases.

---

## Looking Ahead

Having learned how to train ALIGNN, we are now prepared to explore its **applications in materials science**. In the next section, we will examine how ALIGNN has been successfully applied to problems such as formation energy prediction, band gap estimation, elastic property prediction, defect modeling, catalyst discovery, battery materials screening, and high-throughput materials design. We will also compare its performance with earlier models such as CGCNN, SchNet, and MEGNet across a variety of benchmark datasets.

# 17.12 Applications of ALIGNN in Materials Science

The ultimate purpose of developing machine learning models such as ALIGNN is not merely to achieve lower prediction errors, but to accelerate scientific discovery.

Traditional computational materials science relies heavily on **Density Functional Theory (DFT)** and experimental characterization. Although these approaches are highly accurate, they are computationally expensive and time-consuming. A single DFT calculation for a complex crystal may require several hours or even days on a high-performance computing cluster.

ALIGNN provides an attractive alternative. Once trained, it can predict material properties within milliseconds while maintaining accuracy that often approaches first-principles calculations.

This dramatic reduction in computational cost makes ALIGNN a valuable tool for high-throughput materials screening, inverse design, and accelerated materials discovery.

In this section, we examine the major research applications of ALIGNN and discuss why its dual-graph architecture makes it particularly well suited for crystalline materials.

---

# 17.12.1 Formation Energy Prediction

One of the most important quantities in computational materials science is the **formation energy**.

Formation energy measures the thermodynamic stability of a material.

Mathematically,

$$
\boxed{
E_f
===

## E_{\text{compound}}

\sum_i
n_i\mu_i,
}
$$

where

* $E_{\text{compound}}$ is the total energy of the crystal,
* $n_i$ is the number of atoms of element $i$,
* $\mu_i$ is the reference chemical potential.

A lower formation energy generally indicates greater thermodynamic stability.

---

## Why Formation Energy Matters

Formation energy is used to determine

* whether a material can exist,
* phase stability,
* competing crystal structures,
* synthesis feasibility.

Traditionally,

obtaining

$$
E_f
$$

requires a complete DFT calculation.

ALIGNN predicts

formation energies

directly from crystal structures,

reducing computation from hours

to milliseconds.

---

## Typical Workflow

```text id="alignn_app1"
Crystal Structure

        │

        ▼

ALIGNN

        │

        ▼

Formation Energy

        │

        ▼

Stable?

Yes / No
```

Large material databases can therefore be screened rapidly.

---

# 17.12.2 Band Gap Prediction

The electronic band gap

is one of the most important properties

of semiconductors.

The band gap is defined as

$$
\boxed{
E_g
= E_{\text{CBM}}

E_{\text{VBM}},
}
$$

where

* CBM denotes the conduction band minimum,
* VBM denotes the valence band maximum.

---

## Importance

Band gap determines

* electrical conductivity,
* optical absorption,
* photovoltaic efficiency,
* LED performance,
* transistor behavior.

Small geometric changes

can significantly alter

the electronic band structure.

Because ALIGNN explicitly models

bond angles,

it often predicts band gaps

more accurately

than pairwise graph neural networks.

---

## High-Throughput Semiconductor Screening

A typical screening workflow is

```text id="alignn_app2"
100,000 Crystal Structures

        │

        ▼

ALIGNN

        │

        ▼

Predicted Band Gaps

        │

        ▼

Select

1–2 eV Materials
```

Only the most promising candidates

are subsequently evaluated

using expensive DFT calculations.

---

# 17.12.3 Elastic Property Prediction

Mechanical properties

depend strongly on

atomic arrangement.

Examples include

* bulk modulus,
* shear modulus,
* Young's modulus,
* Poisson's ratio.

These quantities are derived

from the elastic tensor

$$
C_{ij}.
$$

For example,

the bulk modulus

can be expressed as

$$
\boxed{
K
=

-,
V
\frac{dP}{dV}.
}
$$

---

## Why ALIGNN Performs Well

Elastic properties depend heavily on

* bond stiffness,
* bond angles,
* crystal symmetry.

Since ALIGNN explicitly learns

angular relationships,

it is particularly effective

for predicting

mechanical properties.

---

# 17.12.4 Phonon and Vibrational Properties

Atomic vibrations

depend not only on

interatomic distances,

but also

on local geometry.

Important quantities include

* phonon frequencies,
* Debye temperature,
* thermal conductivity,
* lattice heat capacity.

These properties influence

* thermoelectric materials,
* heat management,
* superconductors.

Traditional phonon calculations

require expensive

density functional perturbation theory.

ALIGNN provides

fast surrogate models

for many vibrational properties.

---

# 17.12.5 Battery Materials Discovery

One of the largest application areas

of machine learning

is battery research.

Researchers search for

* cathodes,
* anodes,
* solid electrolytes.

Important predicted properties include

* formation energy,
* voltage,
* ionic conductivity,
* diffusion barrier,
* structural stability.

Typical workflow

```text id="alignn_app3"
Candidate Crystal

        │

        ▼

ALIGNN

        │

        ▼

Voltage

Formation Energy

Band Gap

        │

        ▼

Best Battery Candidates
```

This approach dramatically reduces

the number of

DFT calculations required.

---

# 17.12.6 Catalysis

Catalytic activity

depends strongly on

surface geometry.

Small changes

in bond angles

may significantly alter

adsorption energies.

Important catalytic properties include

* adsorption energy,
* reaction barrier,
* surface stability,
* activation energy.

Because ALIGNN captures

three-body geometric interactions,

it often produces

better adsorption predictions

than models relying solely on pairwise distances.

Applications include

* hydrogen evolution,
* oxygen reduction,
* carbon dioxide reduction,
* ammonia synthesis.

---

# 17.12.7 Defect Property Prediction

Perfect crystals

rarely exist.

Real materials contain

* vacancies,
* interstitials,
* substitutions,
* grain boundaries.

Defects

modify

the local atomic geometry.

Since ALIGNN models

both

bond lengths

and

bond angles,

it can effectively learn

how defects influence

material properties.

Applications include

* defect formation energy,
* defect migration,
* irradiation damage,
* semiconductor doping.

---

# 17.12.8 High-Entropy Alloys

High-entropy alloys (HEAs)

contain multiple principal elements,

creating highly complex local environments.

Examples include

$$
\mathrm{CoCrFeMnNi}
$$

and related compositions.

Predicting

their properties

is extremely challenging,

because neighboring atoms

vary considerably.

ALIGNN's local message passing

naturally adapts

to

these varying chemical environments,

making it suitable

for

HEA research.

---

# 17.12.9 Crystal Structure Screening

One of the greatest advantages

of ALIGNN

is its speed.

Suppose

one million

hypothetical crystal structures

have been generated.

Traditional workflow

```text id="alignn_app4"
1,000,000 Structures

↓

DFT

↓

Impossible
```

ALIGNN workflow

```text id="alignn_app5"
1,000,000 Structures

↓

ALIGNN

↓

Top 5,000 Candidates

↓

DFT
```

The computational savings

can exceed

several orders of magnitude.

---

# 17.12.10 Integration with Generative Models

Modern materials discovery increasingly combines

**generative models** with **property predictors**.

A generative model proposes new crystal structures,

while ALIGNN rapidly evaluates their properties.

The workflow is

```text id="alignn_app6"
Generator

↓

New Crystal

↓

ALIGNN

↓

Property Prediction

↓

Optimization

↓

Improved Crystal
```

This closed-loop framework forms the basis of inverse materials design, where the objective is to generate structures that satisfy desired target properties.

---

# 17.12.11 Integration with Active Learning

ALIGNN is also widely used within **active learning** workflows.

Instead of labeling every material using DFT,

the model identifies the most informative candidates.

A typical cycle is

```text id="alignn_app7"
Initial Dataset

        │

        ▼

Train ALIGNN

        │

        ▼

Predict Unlabeled Materials

        │

        ▼

Select Most Uncertain Samples

        │

        ▼

Run DFT

        │

        ▼

Add New Data

        │

        ▼

Retrain ALIGNN
```

This iterative strategy minimizes the number of expensive first-principles calculations while maximizing model improvement.

---

# 17.12.12 Performance Compared with Earlier Models

ALIGNN has been benchmarked against several widely used graph neural networks for materials science.

| Model             | Pairwise Interactions | Angular Information | Line Graph | Typical Performance                     |
| ----------------- | --------------------- | ------------------- | ---------- | --------------------------------------- |
| Linear Regression | No                    | No                  | No         | Low                                     |
| Random Forest     | Indirect              | No                  | No         | Moderate                                |
| CGCNN             | Yes                   | No                  | No         | Good                                    |
| SchNet            | Yes                   | Implicit            | No         | Very Good                               |
| MEGNet            | Yes                   | Partial             | No         | Excellent                               |
| **ALIGNN**        | Yes                   | **Explicit**        | **Yes**    | **State-of-the-art on many benchmarks** |

The primary advantage of ALIGNN is not simply greater model complexity, but its ability to explicitly incorporate angular information through the line graph, providing a physically meaningful inductive bias for crystalline materials.

---

# 17.12.13 Current Limitations

Despite its impressive performance, ALIGNN is not without limitations.

### Computational Cost

The addition of the line graph increases memory usage and computation relative to pairwise graph neural networks.

### Dependence on Training Data

Like all supervised learning models, ALIGNN cannot reliably extrapolate far beyond the chemical and structural diversity represented in its training dataset.

### Long-Range Interactions

Although message passing captures local environments effectively, very long-range electrostatic interactions may require additional modeling strategies or larger receptive fields.

### Dynamic Processes

The standard ALIGNN architecture is designed primarily for static crystal structures. Time-dependent phenomena such as molecular dynamics trajectories require further extensions or hybrid approaches.

Understanding these limitations is essential when selecting ALIGNN for a particular research problem.

---

# 17.12.14 Summary

ALIGNN has emerged as one of the most influential graph neural network architectures for crystalline materials because it combines

* explicit angular modeling,
* physically motivated graph representations,
* scalable message passing,
* high predictive accuracy.

Its applications span a broad range of materials science, including

* formation energy prediction,
* electronic structure,
* mechanical properties,
* battery materials,
* catalysis,
* defect engineering,
* high-entropy alloys,
* high-throughput screening,
* inverse design,
* active learning.

Rather than replacing first-principles methods, ALIGNN serves as an efficient surrogate model that dramatically accelerates the exploration of chemical and structural space, enabling researchers to focus expensive quantum mechanical calculations on the most promising candidate materials.

---

## Looking Ahead

The final section of this chapter will present a **comprehensive comparison of ALIGNN with other atomistic graph neural networks**, including CGCNN, SchNet, MEGNet, M3GNet, and other modern architectures. We will compare their graph representations, message-passing strategies, geometric modeling capabilities, computational complexity, strengths, limitations, and typical application domains. This comparison will provide practical guidance for selecting the most appropriate graph neural network architecture for a given materials informatics problem.

# 17.13 Comparison of ALIGNN with Other Atomistic Graph Neural Networks

Over the past decade, graph neural networks have transformed computational materials science. Numerous architectures have been proposed, each addressing specific limitations of its predecessors.

Rather than asking

> *Which model is the best?*

a more useful question is

> **Which model is best suited for a particular materials science problem?**

Each architecture makes different assumptions about atomic interactions, geometric information, computational efficiency, and scalability.

In this section, we compare ALIGNN with the most influential atomistic graph neural networks discussed throughout this book.

These include

* CGCNN
* SchNet
* MEGNet
* PhysNet
* M3GNet
* ALIGNN

By understanding their similarities and differences, researchers can select the most appropriate model for their own investigations.

---

# 17.13.1 Evolution of Atomistic Graph Neural Networks

The development of atomistic GNNs can be viewed as a gradual incorporation of increasingly rich physical information.

```text id="alignn_compare_1"
CGCNN

↓

SchNet

↓

MEGNet

↓

PhysNet

↓

M3GNet

↓

ALIGNN
```

Each new architecture addressed limitations of earlier models.

For example,

* CGCNN introduced crystal graph convolutions.
* SchNet introduced continuous filters.
* MEGNet jointly updated atoms and bonds.
* PhysNet focused on molecular quantum interactions.
* M3GNet incorporated three-body interactions.
* ALIGNN introduced explicit line graphs for angular message passing.

Thus,

ALIGNN did not replace previous models arbitrarily.

Instead,

it represents one step in the continuing evolution of physically informed graph neural networks.

---

# 17.13.2 Comparison of Graph Representations

The first major difference between these models lies in how they represent atomic structures.

| Model      | Graph Nodes   | Graph Edges         | Line Graph |
| ---------- | ------------- | ------------------- | ---------- |
| CGCNN      | Atoms         | Bonds               | No         |
| SchNet     | Atoms         | Distances           | No         |
| MEGNet     | Atoms         | Bonds               | No         |
| PhysNet    | Atoms         | Distances           | No         |
| M3GNet     | Atoms         | Bonds               | No         |
| **ALIGNN** | Atoms + Bonds | Bonds + Bond Angles | **Yes**    |

Most earlier models operate using only the crystal graph.

ALIGNN uniquely introduces

a second graph

that represents

relationships between bonds.

This additional graph allows explicit learning of bond-angle interactions.

---

# 17.13.3 Treatment of Geometry

Geometry is one of the most important aspects of atomistic machine learning.

Different architectures incorporate geometric information in different ways.

| Model      | Distances | Bond Angles  | Three-Body Effects |
| ---------- | --------- | ------------ | ------------------ |
| CGCNN      | ✓         | ✗            | ✗                  |
| SchNet     | ✓         | Implicit     | Partial            |
| MEGNet     | ✓         | Limited      | Partial            |
| PhysNet    | ✓         | Implicit     | Partial            |
| M3GNet     | ✓         | ✓            | ✓                  |
| **ALIGNN** | ✓         | **Explicit** | **✓**              |

Notice the distinction between

**implicit**

and

**explicit**

angular modeling.

SchNet,

PhysNet,

and some other architectures

may eventually learn angular relationships

through repeated message passing.

ALIGNN,

however,

models them directly

through

the line graph.

This provides

a stronger physical inductive bias.

---

# 17.13.4 Message Passing Strategy

The core of every graph neural network

is message passing.

Different architectures

perform message passing

in different ways.

### CGCNN

Messages flow

between atoms.

```text id="alignn_compare_2"
Atom

↓

Neighbor Atom
```

---

### SchNet

Continuous filters

weight

neighbor contributions.

```text id="alignn_compare_3"
Atom

↓

Distance Filter

↓

Neighbor Atom
```

---

### MEGNet

Messages update

both

nodes

and

edges.

```text id="alignn_compare_4"
Atom

↓

Bond

↓

Atom
```

---

### ALIGNN

Two independent message-passing operations

occur.

```text id="alignn_compare_5"
Bond

↓

Neighbor Bond

(Line Graph)

↓

Updated Bond

↓

Atom Update

(Crystal Graph)
```

The dual-graph strategy

is

the defining innovation

of ALIGNN.

---

# 17.13.5 Computational Complexity

More expressive models

generally require

more computation.

Approximate computational cost

is summarized below.

| Model   | Relative Computational Cost |
| ------- | --------------------------- |
| CGCNN   | Low                         |
| SchNet  | Moderate                    |
| PhysNet | Moderate                    |
| MEGNet  | High                        |
| M3GNet  | High                        |
| ALIGNN  | High                        |

The line graph

contains additional nodes

and

additional edges.

Consequently,

ALIGNN requires

more GPU memory

than

CGCNN

or

SchNet.

The increased computational cost,

however,

often results

in improved predictive accuracy

for geometry-sensitive properties.

---

# 17.13.6 Prediction Accuracy

Across many benchmark datasets,

modern graph neural networks

significantly outperform

traditional machine learning methods.

A qualitative comparison

is shown below.

| Model             | Typical Accuracy              |
| ----------------- | ----------------------------- |
| Linear Regression | Low                           |
| Random Forest     | Moderate                      |
| CGCNN             | High                          |
| SchNet            | Very High                     |
| MEGNet            | Excellent                     |
| M3GNet            | Excellent                     |
| ALIGNN            | Excellent to State-of-the-Art |

It should be emphasized

that

performance depends on

* dataset,
* target property,
* training procedure,
* hyperparameters.

No model

is universally superior

for every application.

---

# 17.13.7 Strengths of ALIGNN

ALIGNN possesses several important advantages.

### Explicit Angular Modeling

Instead of inferring

bond-angle effects,

ALIGNN represents them directly.

---

### Strong Physical Inductive Bias

The architecture closely mirrors

physical interactions

found in

real crystals.

---

### High Accuracy

ALIGNN has demonstrated

excellent performance

for

* formation energies,
* band gaps,
* elastic properties,
* thermodynamic stability.

---

### Flexible Architecture

The dual-graph framework

can be adapted

to

many prediction tasks,

including

both

regression

and

classification.

---

### Compatibility

ALIGNN integrates naturally

with

DGL,

PyTorch,

JARVIS,

and

modern materials databases.

---

# 17.13.8 Limitations of ALIGNN

Despite its advantages,

ALIGNN is not always the ideal choice.

### Higher Memory Consumption

Constructing

the line graph

significantly increases

GPU memory usage.

Large crystal structures

may require

substantially more resources.

---

### Longer Training Time

Dual-graph message passing

is computationally more expensive

than

single-graph architectures.

---

### Increased Implementation Complexity

Compared with CGCNN,

ALIGNN requires

* crystal graph,
* line graph,
* alternating convolutions,
* more complicated batching.

Consequently,

implementation

is more challenging.

---

### Data Requirements

More expressive models

often require

larger training datasets

to realize

their full potential.

---

# 17.13.9 When Should You Use ALIGNN?

ALIGNN is particularly well suited for problems where local geometry strongly influences material behavior.

Recommended applications include

* semiconductors,
* covalent crystals,
* ceramics,
* battery materials,
* catalysts,
* elastic property prediction,
* defect engineering,
* structural stability analysis.

For simpler tasks,

such as rough property estimation on small datasets,

a lighter model like CGCNN may already provide satisfactory performance.

The choice of architecture should therefore balance predictive accuracy, computational resources, and the complexity of the target problem.

---

# 17.13.10 Practical Model Selection Guide

The following table provides a practical summary.

| Research Goal                                                 | Recommended Model |
| ------------------------------------------------------------- | ----------------- |
| Fast baseline model                                           | CGCNN             |
| Continuous geometric interactions                             | SchNet            |
| Joint node-edge learning                                      | MEGNet            |
| Molecular quantum chemistry                                   | PhysNet           |
| Universal atomistic potentials                                | M3GNet            |
| Crystal property prediction with explicit angular information | **ALIGNN**        |

Rather than replacing earlier models, ALIGNN complements them by addressing problems where directional bonding and local geometry play a dominant role.

---

# 17.13.11 Future Directions

Research in atomistic machine learning continues to evolve rapidly.

Several promising directions include

* equivariant graph neural networks that respect rotational symmetry,
* foundation models trained on millions of crystal structures,
* multimodal models combining crystal structures with spectroscopy or microscopy,
* uncertainty-aware graph neural networks,
* self-supervised pretraining for materials,
* hybrid models integrating DFT with machine learning,
* generative crystal design using diffusion and transformer architectures.

Many of these emerging methods build upon ideas pioneered by architectures such as ALIGNN, particularly the importance of incorporating physically meaningful geometric information into the learning process.

---

# 17.13.12 Chapter Summary

In this chapter, we studied **Atomistic Line Graph Neural Networks (ALIGNN)** in depth.

We began by motivating the need for explicit angular information in crystalline materials and showed why pairwise interactions alone are often insufficient for accurately describing complex structure–property relationships.

The chapter covered

* the motivation behind ALIGNN,
* crystal graph construction,
* line graph construction,
* bond graph representation,
* edge-gated graph convolution,
* angular message passing,
* the complete ALIGNN architecture,
* mathematical formulation,
* PyTorch implementation,
* model training,
* practical applications,
* comparison with other atomistic graph neural networks.

The defining innovation of ALIGNN is its **dual-graph architecture**, where the crystal graph models atomic connectivity while the line graph models relationships between neighboring bonds. By alternating message passing between these two graphs, ALIGNN explicitly incorporates three-body geometric information into the learned representations, resulting in highly accurate predictions for a broad range of materials properties.

ALIGNN represents a significant milestone in the evolution of graph neural networks for materials science. More importantly, it illustrates a broader principle that has guided the development of modern materials informatics:

> **Machine learning models become more powerful when their architecture reflects the underlying physics of the problem.**

As research progresses toward equivariant neural networks, foundation models, and AI-driven materials discovery, the concepts introduced by ALIGNN—explicit geometry, physically informed message passing, and rich graph representations—will continue to influence the next generation of atomistic machine learning methods.

---

## Next Chapter

In **Chapter 18**, we move beyond graph neural networks for individual crystal structures and explore **Large Language Models (LLMs) and Foundation Models for Materials Science**. We will study how transformer-based architectures, scientific language models, multimodal AI systems, and materials foundation models are reshaping scientific literature mining, property prediction, autonomous research assistants, inverse materials design, and AI-driven materials discovery pipelines. This chapter will bridge graph neural networks with the rapidly emerging world of generative AI for materials informatics.
