# Chapter 22 — MACE: Higher-Order Equivariant Message Passing for Universal Machine Learning Potentials

---

# 22.1 Introduction

The last three chapters introduced the evolution of equivariant machine learning interatomic potentials.

We began with **Crystal Graph Convolutional Neural Networks (CGCNN)**, which demonstrated how graph neural networks could learn materials properties directly from crystal structures.

We then studied **NequIP**, which incorporated rotational equivariance into graph neural networks, allowing the model to learn physically consistent interatomic potentials with remarkable accuracy.

Finally, in Chapter 21, we explored **Allegro**, which significantly improved computational efficiency by replacing iterative message passing with highly expressive local equivariant interactions.

These three models represent major milestones in the development of machine learning interatomic potentials.

However, researchers soon realized that another important challenge remained.

Even though NequIP and Allegro achieved excellent performance, they still faced limitations when describing **complex many-body atomic interactions**.

This challenge led to the development of **MACE (Higher-Order Equivariant Message Passing Neural Networks)**.

Rather than focusing primarily on computational efficiency,

MACE focuses on **increasing the expressive power of local atomic representations**.

Today,

MACE is widely regarded as one of the most accurate and versatile equivariant machine learning interatomic potentials.

It has become a standard choice for

* molecular simulations,
* crystalline materials,
* liquids,
* amorphous systems,
* biomolecular modeling,
* foundation models for atomistic machine learning.

In this chapter, we will study MACE from first principles.

Unlike many introductory resources,

our objective is not simply to explain how to use the model.

Instead,

we will understand

* why MACE was developed,
* the mathematical ideas behind higher-order message passing,
* its architecture,
* its training methodology,
* its computational characteristics,
* and its role in modern materials informatics.

---

# 22.1.1 Why Was MACE Developed?

The development of MACE was motivated by an important observation.

Most existing machine learning interatomic potentials describe interactions by combining information from neighboring atoms through pairwise message passing.

Even though multiple message-passing layers allow information to propagate over larger neighborhoods,

the interactions are still fundamentally constructed from repeated combinations of pairwise information.

For many materials,

this approach is sufficient.

However,

real materials exhibit much richer interactions.

Examples include

* angular bonding,
* orbital hybridization,
* cooperative atomic motion,
* many-body polarization,
* complex coordination environments.

Capturing these effects accurately often requires higher-order correlations between neighboring atoms.

Researchers therefore asked an important question.

> **Can a neural network learn many-body interactions directly instead of building them gradually through repeated pairwise message passing?**

MACE provides an affirmative answer to this question.

---

# 22.1.2 The Fundamental Idea of MACE

The central philosophy of MACE can be summarized in a single sentence.

> **Instead of repeatedly exchanging pairwise messages, MACE explicitly constructs higher-order equivariant many-body features.**

This idea may appear subtle,

but it fundamentally changes how atomic environments are represented.

Traditional message-passing networks gradually build complex information through many sequential updates.

MACE instead constructs highly expressive local representations directly by combining multiple neighboring interactions through higher-order tensor operations.

As a result,

the network can describe complex local chemistry using fewer interaction layers while maintaining excellent accuracy.

---

# 22.1.3 Pairwise vs Many-Body Thinking

To understand the motivation,

consider a simple example.

Suppose a central atom has four neighboring atoms arranged in a tetrahedral geometry.

A pairwise interaction model considers relationships such as

* center–neighbor 1,
* center–neighbor 2,
* center–neighbor 3,
* center–neighbor 4.

These interactions describe each bond independently.

However,

the geometry of the tetrahedron also depends on

* the angles between neighboring atoms,
* the collective arrangement of all neighbors,
* correlations involving three or more atoms simultaneously.

These collective effects are examples of **many-body interactions**.

MACE is specifically designed to represent such interactions efficiently.

---

# 22.1.4 From Pairwise Features to Correlation Features

Suppose

$$
\mathbf h_i
$$

represents the embedding of atom

$$
i.
$$

Traditional graph neural networks construct interactions primarily through neighboring pairs,

symbolically represented as

$$
(\mathbf h_i,\mathbf h_j).
$$

MACE instead constructs higher-order correlations such as

$$
(\mathbf h_i,\mathbf h_j,\mathbf h_k),
$$

or even larger combinations involving several neighboring atoms.

These higher-order features provide a much richer description of the local atomic environment.

---

# 22.1.5 Why Higher-Order Correlations Matter

Many physical phenomena cannot be described accurately using only pairwise interactions.

Examples include

* bond-angle preferences,
* directional covalent bonding,
* crystal field effects,
* cooperative distortions,
* lattice instabilities,
* collective atomic rearrangements.

These phenomena depend on relationships among multiple neighboring atoms rather than isolated bonds.

Because MACE directly models higher-order correlations,

it often achieves greater accuracy than architectures relying solely on repeated pairwise message passing.

---

# 22.1.6 Equivariance Remains Essential

Although MACE introduces higher-order interactions,

it does **not** abandon rotational symmetry.

Like NequIP and Allegro,

MACE remains fully equivariant with respect to three-dimensional rotations.

If the atomic structure is rotated by

$$
R
\in
\mathrm{SO}(3),
$$

the network satisfies

$$
f(RX)
=====

R
f(X),
$$

for vector quantities,

while scalar quantities remain invariant.

Thus,

MACE preserves exactly the same physical symmetry principles introduced in earlier chapters.

---

# 22.1.7 Relationship to Previous Models

It is useful to position MACE within the historical development of machine learning interatomic potentials.

Conceptually,

the progression can be viewed as

```text
Classical Force Fields

↓

Graph Neural Networks

↓

CGCNN

↓

Equivariant GNNs

↓

NequIP

↓

Local Edge Learning

↓

Allegro

↓

Higher-Order Equivariant Learning

↓

MACE
```

Each model addresses limitations of its predecessors.

MACE builds directly upon the mathematical foundations developed for NequIP while extending the expressive power of local representations.

---

# 22.1.8 MACE Is Not Simply "Another GNN"

At first glance,

MACE may appear to be another equivariant graph neural network.

This impression is misleading.

The defining innovation of MACE is not merely another message-passing strategy.

Instead,

it introduces a fundamentally different representation of local atomic environments based on higher-order equivariant correlations.

Consequently,

its internal feature representations differ substantially from those of NequIP and Allegro.

---

# 22.1.9 Design Goals of MACE

The developers of MACE sought to satisfy several objectives simultaneously.

The model should

* preserve rotational equivariance,
* capture many-body interactions efficiently,
* remain computationally practical,
* scale to large atomistic datasets,
* generalize across diverse chemical systems,
* achieve state-of-the-art accuracy.

Meeting all of these goals required several architectural innovations that we will study throughout this chapter.

---

# 22.1.10 Applications of MACE

Because of its expressive higher-order representations,

MACE has been applied successfully to numerous atomistic problems.

These include

* molecular dynamics,
* crystalline materials,
* amorphous solids,
* liquids,
* biomolecules,
* catalytic surfaces,
* defects,
* interfaces,
* reactive systems.

More recently,

MACE has also become the foundation of several large-scale pretrained machine learning interatomic potentials.

This makes it one of the most influential architectures in modern atomistic machine learning.

---

# 22.1.11 Roadmap of This Chapter

This chapter will proceed through the following progression.

First,

we will understand why pairwise message passing has fundamental limitations.

Next,

we will introduce higher-order correlations and explain how MACE represents them mathematically.

We will then examine

* the complete MACE architecture,
* interaction blocks,
* higher-order tensor products,
* training methodology,
* computational complexity,
* implementation,
* applications,
* comparisons with NequIP and Allegro.

By the end of the chapter,

you will understand not only how MACE operates,

but also why it has become one of the leading architectures for machine learning interatomic potentials.

---

# 22.1.12 Key Takeaways

MACE represents the next major step in the evolution of equivariant machine learning interatomic potentials. While preserving the rotational equivariance introduced by NequIP, it significantly increases the expressive power of local atomic representations by explicitly constructing higher-order many-body correlations. This allows the model to capture complex chemical environments more efficiently and accurately than architectures based solely on repeated pairwise message passing. As a result, MACE has become one of the most widely used and influential neural network potentials in modern computational materials science.

---

## Transition to Section 22.2 — Limitations of Pairwise Message Passing

Before studying the architecture of MACE, it is essential to understand the problem it was designed to solve. In the next section, we will analyze the theoretical limitations of pairwise message-passing neural networks and show why higher-order many-body representations are necessary for accurately modeling complex atomic environments.

# 22.2 Limitations of Pairwise Message Passing

The development of MACE was not motivated by dissatisfaction with NequIP or Allegro.

On the contrary,

both models represented major breakthroughs in equivariant machine learning.

NequIP demonstrated that rotational equivariance dramatically improves the accuracy of neural interatomic potentials.

Allegro showed that these models could be made computationally efficient enough for large-scale molecular dynamics.

Despite these achievements,

researchers recognized that there remained a deeper theoretical limitation.

This limitation concerns **how information is represented inside the neural network**.

Most graph neural networks—including NequIP—construct increasingly complex atomic representations by repeatedly exchanging **pairwise messages** between neighboring atoms.

While this strategy is extremely successful, it does not directly represent higher-order correlations among multiple neighboring atoms.

Understanding this limitation is essential because it explains **why MACE introduces higher-order equivariant features**.

---

# 22.2.1 What Is Pairwise Message Passing?

Consider a graph representing an atomic structure.

Each node corresponds to an atom,

and each edge represents a neighboring interaction.

During message passing,

every neighboring atom sends information to the central atom.

Mathematically,

a typical update takes the form

$$
\mathbf h_i^{(t+1)}
===================

\phi
\left(
\mathbf h_i^{(t)},
\sum_{j\in\mathcal N(i)}
m_{ij}^{(t)}
\right),
$$

where

* (\mathbf h_i^{(t)}) is the representation of atom (i) at layer (t),
* (m_{ij}^{(t)}) is the message sent from atom (j),
* (\phi) denotes the update function.

Each message depends primarily on a **pair of atoms**,

namely,

the sender

$$
j
$$

and the receiver

$$
i.
$$

---

# 22.2.2 Local Pairwise View

Suppose a central atom has four neighbors.

The message-passing process treats each neighboring interaction independently.

Conceptually,

```text id="mace_pair1"
Neighbor A

↓

Center Atom

↑

Neighbor B

↑

Neighbor C

↑

Neighbor D
```

The central atom receives

four separate messages.

Each message contains information about only one neighboring pair.

---

# 22.2.3 What Information Is Missing?

Now consider two neighboring atoms,

A and B,

around the same central atom.

Their relative arrangement determines

* bond angle,
* orbital overlap,
* coordination geometry.

However,

the message from A

does not directly know about

B.

Likewise,

the message from B

does not directly know about

A.

The relationship

between A and B

must emerge indirectly through repeated message passing.

---

# 22.2.4 Example: Water Molecule

Consider the water molecule.

The oxygen atom interacts with

two hydrogen atoms.

The chemical properties of water are determined not only by the two O–H bond lengths,

but also by the

H–O–H bond angle.

Conceptually,

```text id="mace_pair2"
H

 \

  O

 /

H
```

The angle between the two hydrogen atoms plays a fundamental role in

* molecular polarity,
* hydrogen bonding,
* electronic structure.

A purely pairwise description cannot directly encode this three-body relationship.

---

# 22.2.5 Example: Tetrahedral Coordination

Consider silicon in the diamond crystal structure.

Each silicon atom has four nearest neighbors.

The stability of the crystal depends not merely on four independent Si–Si bonds,

but on their collective tetrahedral arrangement.

Conceptually,

```text id="mace_pair3"
      •

     /|\

    • • •

      |

      •
```

The relative orientations of all neighboring atoms determine

* bond angles,
* hybridization,
* crystal stability.

Representing these relationships solely through pairwise interactions is inefficient.

---

# 22.2.6 Building Higher-Order Information Indirectly

Pairwise message-passing networks can still learn many-body effects.

However,

they do so indirectly.

For example,

Layer 1

learns pairwise interactions.

Layer 2

combines information from neighboring pairs.

Layer 3

combines the combinations.

Conceptually,

```text id="mace_pair4"
Pairwise

↓

Pairwise + Pairwise

↓

Three-Body

↓

Higher-Order
```

Thus,

higher-order information emerges gradually.

---

# 22.2.7 The Cost of Repeated Message Passing

This indirect construction introduces several disadvantages.

First,

multiple interaction layers become necessary.

Second,

each additional layer increases computational cost.

Third,

larger receptive fields require repeated synchronization across the graph.

Consequently,

complex many-body relationships may require

many sequential operations

before they become fully represented.

---

# 22.2.8 Information Dilution

Another issue is information dilution.

Every message-passing layer compresses neighboring information into a finite-dimensional node representation.

As information propagates,

important geometric relationships may become more difficult to preserve.

Conceptually,

```text id="mace_pair5"
Local Geometry

↓

Compressed

↓

Compressed Again

↓

Compressed Again

↓

Node Representation
```

Although modern architectures minimize this problem,

it remains an inherent consequence of repeated message aggregation.

---

# 22.2.9 Over-Smoothing

One of the best-known theoretical problems of deep graph neural networks is **over-smoothing**.

As more message-passing layers are added,

neighboring node representations gradually become similar.

Eventually,

different atoms may become difficult to distinguish.

Conceptually,

```text id="mace_pair6"
Layer 1

Different Features

↓

Layer 4

More Similar

↓

Layer 8

Nearly Identical
```

Although equivariant architectures alleviate this issue,

deep message passing still faces practical limitations.

---

# 22.2.10 Higher-Order Correlations Are Fundamental

Many physical interactions naturally involve three or more atoms.

Examples include

* bond-angle potentials,
* torsional interactions,
* crystal-field splitting,
* cooperative lattice distortions,
* hydrogen-bond networks,
* many-body polarization.

These interactions cannot always be represented efficiently using independent pairwise messages.

Instead,

they require explicit many-body representations.

---

# 22.2.11 Mathematical Perspective

Suppose pairwise information is represented by

$$
f(i,j).
$$

Three-body information requires functions of the form

$$
f(i,j,k).
$$

Four-body information involves

$$
f(i,j,k,l).
$$

The complexity of atomic interactions therefore increases rapidly with correlation order.

Traditional message passing approximates these higher-order functions through repeated pairwise updates.

MACE instead constructs higher-order representations directly.

---

# 22.2.12 An Analogy

Imagine describing a conversation in a meeting.

A pairwise approach records only

* Person A speaking to Person B,
* Person B speaking to Person C.

The overall group discussion must then be reconstructed indirectly.

A higher-order approach instead observes the entire conversation simultaneously.

It naturally captures

* group dynamics,
* collective decisions,
* shared context.

MACE follows this second philosophy.

---

# 22.2.13 Why This Matters for Materials

Materials often exhibit cooperative behavior.

Examples include

* ferroelectric distortions,
* octahedral tilting,
* magnetic ordering,
* structural phase transitions,
* correlated lattice vibrations.

These phenomena arise from the collective arrangement of multiple atoms.

A representation capable of modeling higher-order interactions therefore provides a more faithful description of the underlying physics.

---

# 22.2.14 MACE's Solution

MACE addresses these limitations by introducing **higher-order equivariant correlations**.

Rather than repeatedly propagating pairwise messages,

it constructs representations that explicitly encode interactions among multiple neighboring atoms.

These representations are built using higher-order tensor products,

which preserve rotational equivariance while dramatically increasing expressive power.

This architectural innovation allows MACE to capture rich local chemistry using fewer interaction layers than traditional message-passing networks.

---

# 22.2.15 Key Takeaways

Traditional equivariant graph neural networks construct complex atomic representations through repeated pairwise message passing. Although this strategy can eventually capture many-body effects, it does so indirectly, requiring multiple interaction layers and repeated graph-wide communication. Many important physical phenomena, however, depend explicitly on higher-order correlations among several neighboring atoms. MACE addresses this limitation by representing these higher-order interactions directly, providing a richer and more physically expressive description of local atomic environments while preserving rotational equivariance.

---

## Transition to Section 22.3 — Higher-Order Equivariant Correlations

Having identified the limitations of pairwise message passing, we are now ready to study the central innovation of MACE. In the next section, we will introduce **higher-order equivariant correlations**, explain how they are constructed mathematically using tensor products, and show why they enable MACE to model complex many-body interactions more efficiently than previous equivariant neural network architectures.

# 22.3 Higher-Order Equivariant Correlations

In the previous section, we identified the fundamental limitation of pairwise message passing.

Although repeated message passing can eventually approximate many-body interactions,

the network constructs these interactions only **indirectly**.

MACE adopts a completely different strategy.

Instead of building higher-order interactions gradually,

it represents them **explicitly**.

This idea is the defining innovation of the MACE architecture.

Everything that distinguishes MACE from previous equivariant graph neural networks ultimately originates from one concept:

> **Higher-order equivariant correlations.**

Understanding this concept is essential because it explains why MACE achieves such remarkable accuracy across diverse molecular and crystalline systems.

---

# 22.3.1 What Is a Correlation?

Before discussing higher-order correlations,

let us first understand the meaning of the word **correlation** in this context.

Suppose two neighboring atoms interact.

Their relationship depends on quantities such as

* distance,
* orientation,
* chemical identity.

A pairwise interaction therefore represents a correlation between **two atoms**.

Mathematically,

we may write this as

$$
C^{(2)}
=======

f(i,j),
$$

where

$$
i
$$

and

$$
j
$$

identify two atoms.

This is called a **second-order correlation**.

---

# 22.3.2 Three-Body Correlations

Now consider three atoms,

$$
i,
;
j,
;
k.
$$

The interaction between these atoms depends not only on

the three pairwise distances,

but also on

the angle formed by the three atoms.

Conceptually,

```text id="mace_corr1"
j

 \

  i

 /

k
```

The geometry cannot be fully described using independent pairwise interactions.

Instead,

the entire triplet contributes simultaneously.

This produces a

**third-order correlation**,

which may be represented symbolically as

$$
C^{(3)}
=======

f(i,j,k).
$$

---

# 22.3.3 Four-Body Correlations

The idea naturally extends to larger groups.

For four atoms,

we obtain

$$
C^{(4)}
=======

f(i,j,k,l).
$$

These higher-order correlations describe increasingly complex local environments,

including

* tetrahedral arrangements,
* square-planar coordination,
* octahedral geometry,
* cooperative distortions.

As the correlation order increases,

the representation captures progressively richer structural information.

---

# 22.3.4 Why Correlation Order Matters

Consider carbon in diamond.

Its electronic properties depend not only on individual C–C bonds,

but also on

the tetrahedral arrangement of all four neighboring atoms.

Similarly,

transition-metal oxides depend strongly on

* octahedral coordination,
* ligand orientation,
* neighboring oxygen positions.

These environments cannot be characterized completely using only pairwise relationships.

Higher-order correlations naturally encode these collective geometries.

---

# 22.3.5 From Pairwise Messages to Correlation Features

Traditional message passing constructs

$$
C^{(3)}
$$

approximately through repeated applications of

$$
C^{(2)}.
$$

Conceptually,

```text id="mace_corr2"
Pairwise

↓

Pairwise

↓

Pairwise

↓

Approximate Three-Body
```

MACE instead constructs

higher-order correlations directly.

Conceptually,

```text id="mace_corr3"
Neighbor Features

↓

Higher-Order Tensor Product

↓

Three-Body Correlation
```

The resulting representation is considerably more expressive.

---

# 22.3.6 Maintaining Rotational Equivariance

Introducing higher-order correlations is not sufficient by itself.

The representation must also remain equivariant.

Suppose the atomic structure is rotated.

The higher-order features must rotate consistently.

Mathematically,

if

$$
R
\in
\mathrm{SO}(3),
$$

then

$$
\boxed{
\Phi(RX)
========

R
,
\Phi(X)
}
$$

for vector-valued representations,

while scalar quantities remain unchanged.

Thus,

higher-order features obey exactly the same symmetry principles introduced in earlier chapters.

---

# 22.3.7 Building Correlations with Tensor Products

The mathematical tool that makes this possible is the **tensor product**.

Suppose two equivariant features are represented by

$$
\mathbf a
$$

and

$$
\mathbf b.
$$

Their tensor product is

$$
\mathbf a
\otimes
\mathbf b.
$$

If a third feature

$$
\mathbf c
$$

is included,

we obtain

$$
(\mathbf a
\otimes
\mathbf b)
\otimes
\mathbf c.
$$

Repeating this process generates increasingly higher-order correlations.

This is precisely the mechanism employed by MACE.

---

# 22.3.8 Recursive Construction

One of the elegant aspects of MACE is that higher-order correlations are built recursively.

Conceptually,

```text id="mace_corr4"
First-Order

↓

Tensor Product

↓

Second-Order

↓

Tensor Product

↓

Third-Order

↓

Tensor Product

↓

Fourth-Order
```

Each additional tensor product increases the expressive power of the local representation.

---

# 22.3.9 Correlation Order in Practice

Although arbitrarily high orders are theoretically possible,

practical models usually employ relatively low correlation orders.

Common choices include

* second order,
* third order,
* fourth order.

Higher correlation orders increase expressive power,

but they also increase computational cost.

Therefore,

model designers must balance

accuracy

and

efficiency.

---

# 22.3.10 Physical Interpretation

Higher-order correlations may be viewed as describing

the collective behavior of neighboring atoms.

Rather than asking

> "How does atom A interact with atom B?"

the network instead asks

> "How does the entire local neighborhood interact as a single geometric object?"

This shift in perspective represents one of the major conceptual advances introduced by MACE.

---

# 22.3.11 Relation to Many-Body Potentials

Traditional empirical force fields often include explicit many-body terms.

Examples include

* Stillinger–Weber potentials,
* Tersoff potentials,
* Brenner potentials.

These analytical potentials contain hand-designed angular functions.

MACE learns analogous many-body relationships directly from data.

Unlike empirical force fields,

however,

the functional form is not predefined.

Instead,

it is discovered automatically during training.

---

# 22.3.12 Richer Local Representations

The local representation produced by MACE therefore contains substantially more information than a conventional node embedding.

Rather than encoding

only pairwise relationships,

it captures

* bond lengths,
* bond angles,
* local coordination,
* higher-order geometric correlations,
* chemical identity.

This richer representation enables highly accurate energy prediction using relatively shallow architectures.

---

# 22.3.13 Why Fewer Layers Are Needed

Because higher-order interactions are represented explicitly,

the network no longer relies on many sequential message-passing layers to discover them.

Consequently,

important local chemistry can often be captured using fewer interaction blocks.

This reduces

* information loss,
* oversmoothing,
* optimization difficulty.

It also contributes to the excellent empirical performance of MACE.

---

# 22.3.14 Conceptual Comparison

The philosophical difference between NequIP and MACE can be summarized as follows.

NequIP follows

```text id="mace_corr5"
Pairwise Messages

↓

Repeated Updates

↓

Emergent Many-Body Features
```

MACE instead follows

```text id="mace_corr6"
Higher-Order Tensor Products

↓

Explicit Many-Body Features

↓

Energy Prediction
```

Both approaches ultimately describe many-body physics,

but they construct their internal representations in fundamentally different ways.

---

# 22.3.15 Key Takeaways

Higher-order equivariant correlations are the defining innovation of MACE. Instead of constructing many-body interactions indirectly through repeated pairwise message passing, MACE explicitly represents collective geometric relationships among multiple neighboring atoms using recursive equivariant tensor products. These higher-order features preserve rotational symmetry while providing a much richer description of local atomic environments, allowing the model to capture complex chemical interactions more efficiently and accurately than conventional message-passing architectures.

---

## Transition to Section 22.4 — Mathematical Foundations of Higher-Order Tensor Products

Higher-order correlations in MACE are constructed through repeated equivariant tensor products. In the next section, we will study the mathematical foundations of these tensor products, understand how recursive tensor contractions generate increasingly expressive atomic representations, and see why this framework allows MACE to model many-body interactions while rigorously preserving rotational equivariance.

# 22.4 Mathematical Foundations of Higher-Order Tensor Products

In the previous section, we introduced the central idea behind MACE:

> **Higher-order equivariant correlations are constructed using recursive tensor products.**

While this idea is conceptually simple, its mathematical foundation is considerably deeper.

Tensor products are not unique to machine learning.

They are fundamental mathematical objects used throughout

* quantum mechanics,
* continuum mechanics,
* elasticity,
* crystallography,
* representation theory,
* theoretical physics.

In MACE, tensor products serve a very specific purpose:

they combine multiple equivariant features into richer representations while preserving rotational symmetry.

This section develops the mathematical foundation required to understand how MACE performs this operation.

---

# 22.4.1 Review: Scalars, Vectors, and Higher-Order Tensors

Before discussing tensor products, let us briefly review tensors themselves.

A **scalar** is a zero-order tensor.

Examples include

* energy,
* mass,
* temperature,
* atomic charge.

A scalar remains unchanged under rotation.

Mathematically,

$$
s' = s.
$$

---

A **vector** is a first-order tensor.

Examples include

* force,
* velocity,
* displacement,
* electric field.

If the coordinate system is rotated by a rotation matrix

$$
R,
$$

then the vector transforms as

$$
\boxed{
\mathbf v'
==========

R\mathbf v
}
$$

---

A **second-order tensor** describes relationships between vectors.

Examples include

* stress,
* strain,
* dielectric tensor,
* moment of inertia.

Under rotation,

a second-order tensor transforms according to

$$
\boxed{
T'
==

RTR^{T}
}
$$

This transformation preserves the physical meaning of the tensor regardless of the coordinate system.

---

# 22.4.2 What Is a Tensor Product?

Suppose we have two vectors,

$$
\mathbf a
=========

(a_x,a_y,a_z)
$$

and

$$
\mathbf b
=========

(b_x,b_y,b_z).
$$

Their tensor product is written as

$$
\boxed{
\mathbf a
\otimes
\mathbf b
}
$$

Unlike the dot product,

which produces a scalar,

the tensor product produces a **matrix**.

Explicitly,

$$
\mathbf a
\otimes
\mathbf b
=========

\begin{bmatrix}
a_xb_x & a_xb_y & a_xb_z\
a_yb_x & a_yb_y & a_yb_z\
a_zb_x & a_zb_y & a_zb_z
\end{bmatrix}.
$$

Notice that every component of one vector is multiplied with every component of the other.

The resulting tensor therefore contains much more information than either vector alone.

---

# 22.4.3 Difference Between Dot Product and Tensor Product

Because both operations combine vectors, they are sometimes confused.

However, they are fundamentally different.

The **dot product**

$$
\boxed{
\mathbf a\cdot\mathbf b
=======================

\sum_i
a_ib_i
}
$$

compresses two vectors into a **single scalar**.

Information is lost during this compression.

---

The **tensor product**

$$
\boxed{
\mathbf a
\otimes
\mathbf b
}
$$

retains every pairwise interaction between vector components.

Consequently,

it preserves much richer geometric information.

Conceptually,

```text id="tensor1"
Vector

×

Vector

↓

Dot Product

↓

Scalar
```

versus

```text id="tensor2"
Vector

×

Vector

↓

Tensor Product

↓

Matrix
```

This additional information is precisely why tensor products are so powerful in equivariant neural networks.

---

# 22.4.4 Higher-Order Tensor Products

Tensor products can be applied repeatedly.

Suppose we have three vectors,

$$
\mathbf a,
\mathbf b,
\mathbf c.
$$

A third-order tensor may be constructed as

$$
\boxed{
(\mathbf a
\otimes
\mathbf b)
\otimes
\mathbf c
}
$$

Similarly,

four vectors produce

$$
((\mathbf a
\otimes
\mathbf b)
\otimes
\mathbf c)
\otimes
\mathbf d.
$$

Every additional tensor product increases the order of the representation.

---

# 22.4.5 Why Recursive Tensor Products Matter

Suppose a local atomic environment contains

five neighboring atoms.

Pairwise methods describe

each bond separately.

MACE instead repeatedly combines neighboring information.

Conceptually,

```text id="tensor3"
Neighbor Features

↓

Tensor Product

↓

Second-Order Feature

↓

Tensor Product

↓

Third-Order Feature

↓

Tensor Product

↓

Fourth-Order Feature
```

Each stage captures increasingly complex many-body relationships.

---

# 22.4.6 Tensor Products Preserve Equivariance

One of the most remarkable properties of tensor products is that they naturally preserve rotational symmetry.

Suppose

$$
\mathbf a'
==========

R\mathbf a
$$

and

$$
\mathbf b'
==========

R\mathbf b.
$$

Their tensor product transforms as

$$
\boxed{
(R\mathbf a)
\otimes
(R\mathbf b)
============

R
(\mathbf a\otimes\mathbf b)
R^{T}
}
$$

Thus,

the combined representation remains physically consistent under rotation.

This property is the mathematical reason why tensor products are used throughout MACE.

---

# 22.4.7 Tensor Products and Many-Body Physics

Consider three neighboring atoms.

Their relative arrangement depends on

* bond lengths,
* bond angles,
* orientations.

A tensor product combines these geometric quantities into a unified mathematical object.

Instead of describing

each interaction independently,

the tensor product represents

their collective relationship.

Consequently,

higher-order tensor products naturally encode many-body physics.

---

# 22.4.8 Tensor Products in Representation Theory

The tensor products used in MACE are not ordinary matrix multiplications.

Instead,

they operate on **irreducible representations of the rotation group**.

Suppose two irreducible representations are denoted by

$$
l_1
$$

and

$$
l_2.
$$

Their tensor product decomposes into

$$
\boxed{
l_1
\otimes
l_2
===

|l_1-l_2|,
;
|l_1-l_2|+1,
;
\ldots,
;
l_1+l_2
}
$$

This decomposition is identical to the angular momentum coupling rules encountered in quantum mechanics.

It is implemented mathematically using **Clebsch–Gordan coefficients**.

---

# 22.4.9 Example

Suppose two vectors each correspond to

$$
l=1.
$$

Their tensor product becomes

$$
\boxed{
1
\otimes
1
=

0
\oplus
1
\oplus
2
}
$$

This means the product contains

* one scalar,
* one vector,
* one second-order tensor.

Remarkably,

a simple tensor product automatically generates features of different geometric orders.

This greatly enriches the representation.

---

# 22.4.10 Recursive Feature Construction in MACE

MACE repeatedly applies these tensor products.

Suppose

$$
\mathbf h^{(0)}
$$

is the initial atomic embedding.

After one tensor product,

we obtain

$$
\mathbf h^{(1)}.
$$

After another,

$$
\mathbf h^{(2)}.
$$

Symbolically,

$$
\boxed{
\mathbf h^{(k+1)}
=================

\mathcal T
\left(
\mathbf h^{(k)}
\right)
}
$$

where

$$
\mathcal T
$$

represents an equivariant tensor-product operation.

Each stage increases the expressive power of the representation.

---

# 22.4.11 Correlation Order Emerges Naturally

An important consequence of recursive tensor products is that correlation order increases automatically.

Conceptually,

```text id="tensor4"
Atomic Embedding

↓

Tensor Product

↓

Pair Correlation

↓

Tensor Product

↓

Three-Body Correlation

↓

Tensor Product

↓

Four-Body Correlation
```

Thus,

higher-order many-body interactions emerge naturally from repeated tensor operations.

---

# 22.4.12 Computational Cost

Higher-order tensor products are substantially more expressive than ordinary linear layers.

However,

this expressive power comes at a computational cost.

The number of tensor components increases rapidly with correlation order.

Consequently,

practical implementations typically restrict

* maximum angular momentum,
* maximum correlation order.

These choices balance

accuracy

and

computational efficiency.

---

# 22.4.13 Why MACE Remains Practical

Despite using higher-order tensor products,

MACE remains computationally practical because

* tensor products are localized,
* correlation order is limited,
* efficient implementations are provided by **e3nn**,
* optimized GPU kernels perform the contractions efficiently.

Thus,

the expressive benefits greatly outweigh the additional computational expense.

---

# 22.4.14 Physical Interpretation

One may interpret recursive tensor products as progressively building a richer description of the local atomic environment.

Initially,

the network knows only

individual atoms.

After the first tensor product,

it understands

pairwise interactions.

After the second,

it recognizes

three-body geometry.

Further tensor products capture

increasingly complex coordination environments.

Rather than memorizing isolated bonds,

the network learns the **collective geometry of the neighborhood**.

---

# 22.4.15 Key Takeaways

Higher-order tensor products provide the mathematical foundation of MACE. Unlike ordinary linear operations or pairwise message passing, recursive equivariant tensor products combine multiple geometric features into increasingly expressive many-body representations while rigorously preserving rotational symmetry. Through repeated tensor contractions and irreducible representation decomposition, MACE constructs rich local atomic descriptors capable of capturing complex chemical environments far more directly than conventional graph neural networks.

---

## Transition to Section 22.5 — Correlation Order in MACE

Although higher-order tensor products can theoretically be applied indefinitely, practical models must choose a finite **correlation order**. This choice determines how many-body interactions are represented, how expressive the model becomes, and how much computational cost is incurred. In the next section, we will study the concept of correlation order in detail and understand why it is one of the most important hyperparameters in the MACE architecture.

# 22.5 Correlation Order in MACE

In the previous section, we learned that MACE constructs rich atomic representations through **recursive equivariant tensor products**.

However, one important question remains unanswered.

> **How many tensor products should be applied?**

In principle, there is no mathematical limit.

One could continue constructing

* second-order,
* third-order,
* fourth-order,
* fifth-order,
* even higher-order correlations.

Unfortunately, every increase in correlation order also increases computational complexity.

Therefore, every practical implementation of MACE must choose a **maximum correlation order**.

This choice is one of the most important design decisions in the entire architecture.

It determines

* the expressive power of the model,
* the computational cost,
* the memory requirement,
* the training time,
* the inference speed.

Understanding correlation order is therefore essential for understanding why different MACE models perform differently.

---

# 22.5.1 What Is Correlation Order?

Correlation order simply refers to **how many neighboring atomic features are combined simultaneously** through recursive tensor products.

For example,

a first-order representation describes individual atomic features.

A second-order representation combines two features.

A third-order representation combines three features.

A fourth-order representation combines four features.

Mathematically,

the correlation order

$$
\nu
$$

indicates the number of interacting feature tensors involved in constructing a representation.

---

# 22.5.2 First-Order Correlations

The simplest representation is the first-order representation.

Here,

each atom is represented independently.

Conceptually,

```text id="corr_order1"
Atom

↓

Embedding
```

Mathematically,

$$
\mathbf h^{(1)}
===============

\mathbf h.
$$

No neighboring information has yet been incorporated.

Such a representation contains only chemical identity and basic atomic information.

It cannot describe bonding.

---

# 22.5.3 Second-Order Correlations

The next level combines two neighboring features.

Conceptually,

```text id="corr_order2"
Feature A

×

Feature B

↓

Tensor Product

↓

Second-Order Feature
```

Mathematically,

$$
\boxed{
\mathbf h^{(2)}
===============

\mathbf h_i
\otimes
\mathbf h_j
}
$$

These features describe pairwise interactions.

Examples include

* bond distances,
* bond orientations,
* pairwise chemical interactions.

Many graph neural networks primarily rely on representations of this type.

---

# 22.5.4 Third-Order Correlations

The next level incorporates three interacting features simultaneously.

Conceptually,

```text id="corr_order3"
Feature A

×

Feature B

↓

Tensor Product

↓

Intermediate Feature

×

Feature C

↓

Third-Order Feature
```

Mathematically,

$$
\boxed{
\mathbf h^{(3)}
===============

(
\mathbf h_i
\otimes
\mathbf h_j
)
\otimes
\mathbf h_k
}
$$

These representations naturally encode

* bond angles,
* local coordination,
* three-body interactions.

Many chemically important interactions first become explicit at this level.

---

# 22.5.5 Fourth-Order Correlations

A fourth-order representation combines four neighboring features.

Symbolically,

$$
\boxed{
\mathbf h^{(4)}
===============

(
(
\mathbf h_i
\otimes
\mathbf h_j
)
\otimes
\mathbf h_k
)
\otimes
\mathbf h_l
}
$$

These representations capture much more complex local geometry,

including

* tetrahedral coordination,
* octahedral distortions,
* collective neighbor arrangements.

For many crystalline materials,

fourth-order correlations provide a highly expressive local description.

---

# 22.5.6 Physical Interpretation

Correlation order has a simple physical interpretation.

| Correlation Order | Captures                                |
| ----------------- | --------------------------------------- |
| First             | Individual atoms                        |
| Second            | Pairwise bonds                          |
| Third             | Bond angles and three-body interactions |
| Fourth            | Local coordination geometry             |
| Higher            | Increasingly complex many-body effects  |

As correlation order increases,

the model develops a progressively richer understanding of the local atomic environment.

---

# 22.5.7 Why Not Use Infinite Correlation Order?

A natural question is

> **If higher correlation orders are more expressive, why not use infinitely high order?**

The answer lies in computational complexity.

Every additional tensor product dramatically increases

* the number of feature channels,
* the number of tensor contractions,
* memory consumption,
* computation time.

Beyond a certain point,

the additional expressive power becomes much smaller than the computational cost.

Therefore,

practical models use relatively low correlation orders.

---

# 22.5.8 Growth of Computational Cost

Suppose

each feature vector has dimension

$$
d.
$$

A first-order representation contains approximately

$$
d
$$

values.

A second-order tensor contains roughly

$$
d^2
$$

values.

A third-order tensor contains approximately

$$
d^3.
$$

A fourth-order tensor contains approximately

$$
d^4.
$$

Although symmetry reduces the actual number of independent components,

the computational cost still grows rapidly.

---

# 22.5.9 Diminishing Returns

Increasing correlation order always increases expressive power,

but the improvement is not unlimited.

Conceptually,

```text id="corr_order4"
Accuracy

↑

|

|           _______

|         /

|       /

|     /

|___/

+-------------------->

Correlation Order
```

Initially,

accuracy improves rapidly.

Eventually,

the curve begins to flatten.

Beyond this point,

higher correlation orders produce only marginal improvements while substantially increasing computational cost.

---

# 22.5.10 Choosing the Correlation Order

Selecting the appropriate correlation order depends on the application.

Simple molecular systems may require only

second-order

or

third-order correlations.

Complex crystalline materials often benefit from

higher-order representations.

Large foundation models may employ even richer feature spaces,

provided sufficient computational resources are available.

Thus,

correlation order is a tunable hyperparameter rather than a fixed architectural requirement.

---

# 22.5.11 Correlation Order and Receptive Field

It is important to distinguish

**correlation order**

from

**receptive field**.

Correlation order describes

how many features are combined simultaneously.

The receptive field describes

how many neighboring atoms influence the representation.

These concepts are related,

but they are not identical.

A model may have

* a small receptive field,
* but a high correlation order,

or vice versa.

MACE increases expressive power primarily through higher-order correlations rather than dramatically increasing the receptive field.

---

# 22.5.12 Correlation Order Versus Message Passing Depth

Another common misconception is that higher correlation order is equivalent to adding more message-passing layers.

They address different aspects of representation learning.

Increasing message-passing depth allows information to travel farther through the graph.

Increasing correlation order enriches the local description of the atomic neighborhood.

Conceptually,

```text id="corr_order5"
More Layers

↓

Larger Neighborhood
```

versus

```text id="corr_order6"
Higher Correlation Order

↓

Richer Local Representation
```

MACE focuses primarily on the second strategy.

---

# 22.5.13 Effect on Expressive Power

One of the theoretical advantages of higher-order correlations is that they allow the model to distinguish local environments that appear identical under pairwise descriptions.

For example,

two atomic environments may possess

identical bond lengths,

yet differ in

* bond-angle distribution,
* coordination geometry,
* many-body arrangement.

A pairwise representation may struggle to distinguish them.

A higher-order representation naturally separates them.

This significantly improves the discriminative power of the learned features.

---

# 22.5.14 Practical Correlation Orders in MACE

Most practical MACE implementations do not use extremely high correlation orders.

Instead,

they employ moderate values that balance

accuracy

and

efficiency.

Typical production models commonly use

* second-order,
* third-order,
* or fourth-order correlations,

depending on the target application and computational budget.

These choices have been shown to provide excellent performance across a wide range of molecular and materials datasets.

---

# 22.5.15 Key Takeaways

Correlation order is one of the defining hyperparameters of the MACE architecture. It specifies how many neighboring feature tensors are combined through recursive equivariant tensor products to construct local atomic representations. Increasing the correlation order enables the network to model progressively richer many-body interactions, such as bond angles, coordination geometry, and complex local environments. However, higher correlation orders also increase computational cost, making the selection of an appropriate order a balance between expressive power and efficiency.

---

## Transition to Section 22.6 — The Complete MACE Architecture

Having established the mathematical foundations of higher-order correlations, we are now ready to assemble the complete MACE model. In the next section, we will examine the full computational pipeline—from atomic embeddings and neighbor graph construction to higher-order equivariant interaction blocks and atomic energy prediction—and understand how all of these components work together to produce one of the most accurate machine learning interatomic potentials available today.

# 22.6 The Complete MACE Architecture

By this point, we have introduced the two central ideas behind MACE:

1. **Equivariant feature representations**, inherited from NequIP.
2. **Higher-order correlations**, introduced through recursive tensor products.

These ideas, however, are only individual building blocks.

To understand how MACE functions as a machine learning interatomic potential, we must assemble these components into a complete computational pipeline.

This section presents the **entire architecture of MACE**, following the flow of data from raw atomic coordinates to the final prediction of energies and forces.

Rather than studying isolated mathematical operations, we will now see how every module interacts with the others.

---

# 22.6.1 Overview of the Architecture

At a high level, MACE transforms an atomic structure into a total energy prediction through a sequence of geometric operations.

The complete workflow can be summarized as

```text
Atomic Structure

↓

Neighbor Graph Construction

↓

Atomic Embeddings

↓

Radial Basis Expansion

↓

Spherical Harmonics

↓

Equivariant Message Passing

↓

Higher-Order Correlation Blocks

↓

Atomic Feature Update

↓

Atomic Energy Prediction

↓

Total Energy

↓

Automatic Differentiation

↓

Atomic Forces
```

Unlike conventional graph neural networks,

the distinguishing stage is the **Higher-Order Correlation Blocks**, which greatly increase the expressive power of the learned representations.

---

# 22.6.2 Input Representation

The input to MACE consists of an atomistic structure.

For every atom,

the model receives

* atomic number,
* Cartesian coordinates,
* simulation cell (if periodic),
* periodic boundary information.

Mathematically,

a structure may be represented as

$$
\mathcal G
==========

(V,E),
$$

where

* (V) represents atoms,
* (E) represents neighboring interactions.

Unlike images,

graphs contain no regular grid.

Consequently,

the first task of the network is to construct the graph.

---

# 22.6.3 Neighbor Graph Construction

Using a cutoff radius

$$
r_c,
$$

neighboring atoms are connected.

Two atoms,

$$
i
$$

and

$$
j,
$$

are connected whenever

$$
\boxed{
|
\mathbf r_i
-----------

\mathbf r_j
|
<
r_c
}
$$

This produces a sparse graph containing only physically meaningful local interactions.

Conceptually,

```text
Atom

↓

Neighbor Search

↓

Local Graph
```

This graph forms the computational backbone of the network.

---

# 22.6.4 Atomic Embedding Layer

Raw atomic numbers cannot be processed directly by the neural network.

Instead,

every element is mapped to a learnable feature vector.

If

$$
Z_i
$$

denotes the atomic number,

the embedding operation becomes

$$
\boxed{
\mathbf h_i^{(0)}
=================

\mathrm{Embed}(Z_i)
}
$$

Initially,

these embeddings contain only chemical identity.

No geometric information has yet been incorporated.

---

# 22.6.5 Relative Position Vectors

For every neighboring pair,

the relative position vector is computed.

$$
\boxed{
\mathbf r_{ij}
==============

## \mathbf r_j

\mathbf r_i
}
$$

This vector provides

* bond length,
* bond orientation.

Unlike absolute coordinates,

relative vectors preserve translational invariance.

---

# 22.6.6 Radial Basis Functions

The distance

$$
r_{ij}
======

|
\mathbf r_{ij}
|
$$

is expanded into a set of radial basis functions.

Instead of using the raw distance,

the network learns

smooth basis coefficients,

$$
\boxed{
\mathbf e_{ij}
==============

\phi(r_{ij})
}
$$

These basis functions provide a richer description of interatomic distances.

As in NequIP and Allegro,

the radial basis functions are learnable.

---

# 22.6.7 Spherical Harmonics

The direction of every bond is encoded using spherical harmonics.

Given the unit vector

$$
\hat{\mathbf r}_{ij},
$$

the angular representation becomes

$$
\boxed{
Y_l^m
(
\hat{\mathbf r}_{ij}
)
}
$$

These functions provide a rotationally equivariant description of bond orientation.

Together,

the radial basis and spherical harmonics completely characterize the local geometry of every edge.

---

# 22.6.8 Initial Edge Features

The radial and angular information are combined to construct edge features.

Conceptually,

```text
Distance

+

Direction

↓

Edge Feature
```

Mathematically,

the edge representation depends on

* radial basis expansion,
* spherical harmonic coefficients,
* atomic embeddings.

These edge features become the input to the interaction blocks.

---

# 22.6.9 Equivariant Message Passing

Like NequIP,

MACE performs equivariant message passing.

Neighboring atoms exchange geometric information,

producing updated atomic representations.

A generic update may be written as

$$
\boxed{
\mathbf h_i'
============

\phi
\left(
\mathbf h_i,
\sum_{j}
m_{ij}
\right)
}
$$

However,

this is **not** where MACE differs from NequIP.

The major innovation comes afterward.

---

# 22.6.10 Higher-Order Correlation Block

After message passing,

MACE applies recursive tensor products to construct higher-order correlations.

Conceptually,

```text
Neighbor Features

↓

Tensor Product

↓

Second-Order Feature

↓

Tensor Product

↓

Third-Order Feature

↓

Tensor Product

↓

Higher-Order Feature
```

These recursive tensor operations generate highly expressive local representations.

Instead of relying on many message-passing layers,

MACE enriches the feature space directly.

---

# 22.6.11 Recursive Feature Expansion

Suppose

$$
\mathbf h^{(k)}
$$

is the current feature representation.

The higher-order interaction block computes

$$
\boxed{
\mathbf h^{(k+1)}
=================

\mathcal T
\left(
\mathbf h^{(k)}
\right)
}
$$

where

$$
\mathcal T
$$

represents an equivariant tensor-product operator.

Each application increases the correlation order.

Consequently,

the representation gradually captures

* pair interactions,
* three-body interactions,
* four-body interactions,
* higher-order local chemistry.

---

# 22.6.12 Atomic Feature Update

After constructing higher-order features,

the atomic representation is updated.

Conceptually,

```text
Previous Feature

+

Higher-Order Feature

↓

Updated Atomic Representation
```

This updated feature now contains

* chemical identity,
* local geometry,
* many-body correlations,
* rotational information.

It is substantially richer than the initial embedding.

---

# 22.6.13 Multiple Interaction Blocks

The architecture does not stop after a single interaction block.

Instead,

multiple MACE blocks are stacked.

Conceptually,

```text
Interaction Block 1

↓

Interaction Block 2

↓

Interaction Block 3

↓

Interaction Block 4
```

Each block increases the expressive power of the atomic representations.

Unlike conventional GNNs,

these blocks focus on enriching local many-body features rather than simply propagating information farther across the graph.

---

# 22.6.14 Atomic Energy Prediction

After the final interaction block,

each atom possesses a refined local representation.

A readout network converts this representation into an atomic energy contribution.

Mathematically,

$$
\boxed{
E_i
===

f
(
\mathbf h_i
)
}
$$

where

$$
f
$$

is a small neural network.

Because

$$
E_i
$$

is a scalar,

the readout operates only on invariant components of the feature representation.

---

# 22.6.15 Total Energy

The total energy is obtained by summing all atomic contributions.

$$
\boxed{
E_{\mathrm{total}}
==================

\sum_i
E_i
}
$$

This decomposition guarantees

* size extensivity,
* permutation invariance.

Both properties are essential for physically meaningful atomistic simulations.

---

# 22.6.16 Force Prediction

The network predicts only

the total energy.

Atomic forces are computed automatically.

$$
\boxed{
\mathbf F_i
===========

*

\frac{\partial
E_{\mathrm{total}}
}
{\partial
\mathbf r_i}
}
$$

Automatic differentiation ensures that

forces remain perfectly consistent with the learned energy surface.

---

# 22.6.17 Architectural Summary

The entire MACE pipeline may now be summarized as

```text
Atomic Structure

↓

Neighbor Graph

↓

Atomic Embeddings

↓

Radial Basis Functions

↓

Spherical Harmonics

↓

Equivariant Message Passing

↓

Higher-Order Correlation Blocks

↓

Updated Atomic Features

↓

Atomic Energies

↓

Total Energy

↓

Automatic Differentiation

↓

Atomic Forces
```

Every component contributes to preserving physical symmetry while improving expressive power.

---

# 22.6.18 Comparison with NequIP

At first glance,

the architectures of NequIP and MACE appear remarkably similar.

Both employ

* neighbor graphs,
* radial basis functions,
* spherical harmonics,
* equivariant message passing,
* energy decomposition.

The principal difference lies inside the interaction block.

| Component                              | NequIP  | MACE      |
| -------------------------------------- | ------- | --------- |
| Neighbor graph                         | ✓       | ✓         |
| Radial basis                           | ✓       | ✓         |
| Spherical harmonics                    | ✓       | ✓         |
| Equivariant message passing            | ✓       | ✓         |
| Higher-order recursive tensor products | ✗       | ✓         |
| Explicit many-body correlations        | Limited | Extensive |

Thus,

MACE can be viewed as a substantial extension of the NequIP architecture rather than an entirely different framework.

---

# 22.6.19 Why This Architecture Works

The success of MACE stems from the combination of three complementary ideas.

First,

equivariance ensures that physical symmetries are preserved.

Second,

local message passing captures neighboring interactions efficiently.

Third,

recursive higher-order tensor products dramatically enrich the local representation.

Together,

these components enable the model to describe highly complex atomic environments while maintaining rigorous physical consistency.

---

# 22.6.20 Key Takeaways

The MACE architecture combines equivariant message passing with recursive higher-order tensor products to construct exceptionally expressive local atomic representations. Starting from atomic embeddings, radial basis functions, and spherical harmonics, the network progressively enriches local features through higher-order correlation blocks before predicting atomic energies whose sum yields the total potential energy. Forces are then obtained through automatic differentiation, ensuring complete physical consistency. This combination of geometric symmetry and explicit many-body representation is the defining characteristic that distinguishes MACE from previous equivariant graph neural networks.

---

## Transition to Section 22.7 — The MACE Interaction Block

Although we have seen the overall architecture, the true computational heart of MACE lies inside its **interaction block**. Unlike conventional graph neural networks, this block performs recursive equivariant tensor-product operations that generate higher-order many-body correlations. In the next section, we will dissect the MACE interaction block layer by layer, understand every mathematical operation it performs, and explain why it is responsible for the extraordinary expressive power of the entire architecture.

# 22.7 The MACE Interaction Block

In the previous section, we studied the complete MACE architecture.

We observed that the overall pipeline closely resembles other equivariant neural network potentials.

The model

* constructs a neighbor graph,
* computes radial basis functions,
* evaluates spherical harmonics,
* performs equivariant message passing,
* predicts atomic energies.

However, one component remains largely unexplored.

This component is the **MACE Interaction Block**.

It is the interaction block that transforms a relatively simple equivariant graph neural network into one capable of modeling highly complex many-body atomic interactions.

In many ways,

the interaction block is the "engine" of MACE.

Understanding this block means understanding the entire model.

---

# 22.7.1 What Is an Interaction Block?

An interaction block is a computational module that updates atomic representations.

Its purpose is to incorporate information from neighboring atoms and produce richer feature vectors.

Conceptually,

```text id="mace_block1"
Atomic Features

+

Neighbor Features

↓

Interaction Block

↓

Updated Atomic Features
```

Unlike ordinary neural network layers,

the MACE interaction block preserves rotational equivariance while simultaneously constructing higher-order correlations.

---

# 22.7.2 Position Within the Network

Recall the overall architecture.

```text id="mace_block2"
Atomic Embedding

↓

Interaction Block

↓

Interaction Block

↓

Interaction Block

↓

Atomic Energy
```

Each interaction block progressively enriches the atomic representation.

The output of one block becomes the input of the next.

Consequently,

the feature representation becomes increasingly expressive throughout the network.

---

# 22.7.3 Inputs to the Interaction Block

Each interaction block receives three types of information.

### Atomic Features

These describe the current representation of every atom,

$$
\mathbf h_i.
$$

Initially,

they originate from atomic embeddings.

In later layers,

they contain learned many-body information.

---

### Edge Features

For every neighboring pair,

the block also receives edge information,

including

* radial basis coefficients,
* spherical harmonics,
* relative position vectors.

These describe the geometry of the local environment.

---

### Neighbor Graph

Finally,

the graph connectivity determines

which atoms exchange information.

Only neighboring atoms within the cutoff radius participate in the interaction.

---

# 22.7.4 Internal Workflow

The internal computation performed by a MACE interaction block may be summarized as

```text id="mace_block3"
Atomic Features

↓

Neighbor Aggregation

↓

Equivariant Messages

↓

Higher-Order Tensor Products

↓

Feature Mixing

↓

Updated Atomic Features
```

Each stage serves a distinct purpose.

---

# 22.7.5 Step 1 — Neighbor Aggregation

The first step resembles traditional message passing.

Each neighboring atom contributes information to the central atom.

Suppose

$$
\mathcal N(i)
$$

denotes the neighbors of atom

$$
i.
$$

Messages are computed for every neighboring pair.

Symbolically,

$$
m_{ij}
======

f
(
\mathbf h_i,
\mathbf h_j,
\mathbf e_{ij}
).
$$

Here,

$$
\mathbf e_{ij}
$$

contains the geometric edge information.

---

# 22.7.6 Step 2 — Equivariant Messages

Unlike conventional graph neural networks,

the messages are equivariant.

Suppose the structure is rotated.

Every message transforms consistently.

Mathematically,

$$
\boxed{
m(RX)
=====

R
m(X)
}
$$

for vector-valued quantities.

This guarantees that the learned physics remains independent of the coordinate system.

---

# 22.7.7 Step 3 — Tensor Product Layer

After computing neighbor messages,

MACE applies its defining operation:

the equivariant tensor product.

Conceptually,

```text id="mace_block4"
Neighbor Message

×

Atomic Feature

↓

Tensor Product

↓

Higher-Order Feature
```

This operation combines multiple geometric quantities simultaneously,

producing much richer local representations than ordinary message passing.

---

# 22.7.8 Recursive Correlation Construction

The tensor-product layer is not applied only once.

Instead,

it operates recursively.

Suppose

$$
\mathbf h^{(1)}
$$

is the first-order representation.

Then,

$$
\boxed{
\mathbf h^{(2)}
===============

\mathcal T
(
\mathbf h^{(1)}
)
}
$$

A second tensor product generates

$$
\boxed{
\mathbf h^{(3)}
===============

\mathcal T
(
\mathbf h^{(2)}
)
}
$$

Further repetitions continue increasing the correlation order.

This recursive construction is responsible for MACE's expressive many-body representations.

---

# 22.7.9 Mixing Different Angular Momenta

Every tensor product generates features with different angular momentum orders.

Suppose two vector features

$$
l=1
$$

are combined.

The resulting tensor decomposes as

$$
\boxed{
1
\otimes
1
=

0
\oplus
1
\oplus
2
}
$$

Thus,

a single tensor product simultaneously produces

* scalar features,
* vector features,
* second-order tensor features.

These different representations are all retained inside the interaction block.

---

# 22.7.10 Feature Mixing

After constructing higher-order features,

the block mixes them using equivariant linear transformations.

Conceptually,

```text id="mace_block5"
Scalar Features

↓

Linear Mixing

↓

Updated Scalars
```

```text id="mace_block6"
Vector Features

↓

Linear Mixing

↓

Updated Vectors
```

Each irreducible representation is processed independently,

ensuring that rotational symmetry is preserved throughout the computation.

---

# 22.7.11 Residual Connection

Modern deep neural networks frequently employ residual connections,

and MACE is no exception.

Rather than replacing the previous representation,

the interaction block adds new information to it.

Mathematically,

$$
\boxed{
\mathbf h_i'
============

\mathbf h_i
+
\Delta
\mathbf h_i
}
$$

where

$$
\Delta
\mathbf h_i
$$

represents the newly learned higher-order correction.

Residual learning improves

* optimization,
* gradient flow,
* training stability.

---

# 22.7.12 Why Residual Connections Help

Without residual connections,

deep networks often suffer from

* vanishing gradients,
* unstable optimization,
* slower convergence.

Conceptually,

```text id="mace_block7"
Input Feature

↓

Interaction Block

↓

Correction

↓

Add Input

↓

Updated Feature
```

Instead of relearning everything,

each block only learns what must change.

---

# 22.7.13 Multiple Interaction Blocks

The MACE architecture typically contains several interaction blocks.

Conceptually,

```text id="mace_block8"
Block 1

↓

Block 2

↓

Block 3

↓

Block 4
```

Each block increases the richness of the representation.

Early blocks primarily capture

local bonding.

Later blocks refine

higher-order many-body geometry.

---

# 22.7.14 Information Flow

The evolution of atomic representations may be viewed as

```text id="mace_block9"
Chemical Identity

↓

Pairwise Geometry

↓

Three-Body Correlations

↓

Higher-Order Correlations

↓

Atomic Energy
```

Thus,

the interaction block gradually transforms simple embeddings into physically meaningful many-body descriptors.

---

# 22.7.15 Computational Perspective

Although interaction blocks are mathematically sophisticated,

their computational structure remains highly regular.

Each block performs

1. Neighbor aggregation

2. Tensor-product construction

3. Feature decomposition

4. Linear mixing

5. Residual update

These operations repeat throughout the network.

Consequently,

the architecture remains modular and scalable.

---

# 22.7.16 Comparison with NequIP

Both NequIP and MACE employ interaction blocks.

However,

their internal computations differ substantially.

| Property                               | NequIP Interaction Block | MACE Interaction Block |
| -------------------------------------- | ------------------------ | ---------------------- |
| Equivariant message passing            | ✓                        | ✓                      |
| Tensor products                        | ✓                        | ✓                      |
| Recursive higher-order tensor products | Limited                  | Extensive              |
| Explicit correlation order             | No                       | Yes                    |
| Many-body feature construction         | Indirect                 | Direct                 |
| Representation richness                | High                     | Very High              |

The primary innovation of MACE lies inside these interaction blocks.

---

# 22.7.17 Why the Interaction Block Is the Core Innovation

The success of MACE does not arise from a radically different overall architecture.

Instead,

it comes from replacing a relatively conventional equivariant interaction module with one capable of constructing explicit higher-order correlations.

In other words,

the interaction block is where the network learns

* local chemistry,
* bond-angle dependence,
* coordination environments,
* many-body physics.

Without this block,

MACE would reduce to a more conventional equivariant graph neural network.

---

# 22.7.18 Key Takeaways

The interaction block is the computational heart of MACE. It receives atomic and edge features, performs equivariant neighbor aggregation, constructs recursive higher-order tensor-product representations, mixes irreducible representations through equivariant linear transformations, and updates atomic features using residual connections. By explicitly building higher-order many-body correlations inside each interaction block, MACE achieves a far richer representation of local atomic environments than previous equivariant graph neural networks, making this module the defining innovation of the architecture.

---

## Transition to Section 22.8 — Higher-Order Message Passing in MACE

Although the interaction block constructs higher-order correlations, it still operates within a message-passing framework. In the next section, we will study **higher-order message passing** itself, understand how information propagates through the graph in MACE, and see how recursive tensor products fundamentally change the nature of message exchange compared with conventional graph neural networks such as NequIP.

# 22.8 Higher-Order Message Passing in MACE

In previous chapters, we repeatedly encountered the term **message passing**.

For most graph neural networks,

message passing is the central computational operation.

Every node receives information from its neighboring nodes,

updates its representation,

and then passes this updated representation to other neighbors.

MACE also belongs to the family of message-passing neural networks.

However,

its message-passing mechanism is fundamentally different from those found in conventional graph neural networks.

Instead of transmitting only pairwise information,

MACE propagates **higher-order equivariant representations**.

This distinction is subtle,

but it is responsible for much of the model's expressive power.

In this section,

we will study exactly how message passing operates inside MACE and understand why it differs from earlier architectures such as CGCNN and NequIP.

---

# 22.8.1 Review of Conventional Message Passing

A standard graph neural network updates node representations by aggregating information from neighboring nodes.

Mathematically,

the update may be written as

$$
\boxed{
\mathbf h_i^{(t+1)}
===================

\phi
\left(
\mathbf h_i^{(t)},
\sum_{j\in\mathcal N(i)}
m_{ij}^{(t)}
\right)
}
$$

where

* (\mathbf h_i^{(t)}) is the representation of atom (i) at layer (t),
* (m_{ij}^{(t)}) is the message received from neighbor (j),
* (\phi) is the update function.

Every interaction occurs between **pairs of atoms**.

---

# 22.8.2 Pairwise Message Passing

Suppose a central atom has four neighbors.

The communication pattern appears as

```text id="mace_msg1"
Neighbor A

↓

Center Atom

↑

Neighbor B

↑

Neighbor C

↑

Neighbor D
```

Each message is computed independently.

After aggregation,

the central atom updates its feature vector.

This process repeats across multiple layers.

---

# 22.8.3 How Complex Information Emerges

Because every message is pairwise,

complex many-body information is not represented immediately.

Instead,

it emerges gradually.

For example,

Layer 1 captures pair interactions.

Layer 2 combines those pair interactions.

Layer 3 combines the combinations.

Conceptually,

```text id="mace_msg2"
Layer 1

↓

Pairwise Features

↓

Layer 2

↓

Three-Body Information

↓

Layer 3

↓

Higher-Order Information
```

Thus,

traditional message passing builds many-body physics indirectly.

---

# 22.8.4 MACE Changes the Nature of Messages

MACE adopts a fundamentally different strategy.

Instead of transmitting only pairwise information,

the messages themselves already contain higher-order correlations.

Conceptually,

```text id="mace_msg3"
Neighbor Features

↓

Tensor Products

↓

Higher-Order Message

↓

Receiving Atom
```

Thus,

every message carries much richer geometric information.

---

# 22.8.5 Message Representation

Rather than representing messages simply as vectors,

MACE represents them using equivariant tensors.

A generic message may be written as

$$
\boxed{
m_{ij}
======

f
(
\mathbf h_i,
\mathbf h_j,
\mathbf e_{ij}
)
}
$$

where

* (\mathbf h_i) is the feature of the receiving atom,
* (\mathbf h_j) is the feature of the neighboring atom,
* (\mathbf e_{ij}) contains radial and angular information.

Unlike conventional GNNs,

the function

$$
f
$$

includes recursive tensor-product operations.

---

# 22.8.6 Messages Are Equivariant

Every message transforms correctly under rotation.

Suppose the structure is rotated by

$$
R.
$$

Then

$$
\boxed{
m(RX)
=====

R
m(X)
}
$$

for vector-valued quantities.

This guarantees that

the network's internal communication respects physical symmetry.

Consequently,

all subsequent computations remain rotationally equivariant.

---

# 22.8.7 Local Higher-Order Features

One of the defining characteristics of MACE is that higher-order correlations are constructed **locally**.

Only neighboring atoms participate.

Suppose the neighborhood of atom

$$
i
$$

contains

$$
\mathcal N(i).
$$

Tensor products are computed only using atoms inside this local neighborhood.

The resulting message therefore summarizes

the entire local geometry

rather than only individual bonds.

---

# 22.8.8 Recursive Message Construction

Messages themselves are generated recursively.

Conceptually,

```text id="mace_msg4"
Neighbor Features

↓

First Tensor Product

↓

Second-Order Message

↓

Second Tensor Product

↓

Third-Order Message

↓

Third Tensor Product

↓

Higher-Order Message
```

The receiving atom therefore obtains information that already contains complex many-body correlations.

---

# 22.8.9 Comparison with NequIP

NequIP also performs equivariant message passing.

However,

its messages primarily describe pairwise interactions.

Higher-order relationships emerge through repeated interaction layers.

MACE instead enriches the message itself before aggregation.

Conceptually,

NequIP follows

```text id="mace_msg5"
Pairwise Message

↓

Update

↓

Pairwise Message

↓

Update

↓

Higher-Order Representation
```

MACE instead follows

```text id="mace_msg6"
Higher-Order Message

↓

Update

↓

Even Richer Higher-Order Message
```

Thus,

the expressive power resides inside each message.

---

# 22.8.10 Aggregation

After higher-order messages are constructed,

they are aggregated.

The aggregation operation remains permutation invariant.

Mathematically,

$$
\boxed{
M_i
===

\sum_{j\in\mathcal N(i)}
m_{ij}
}
$$

Other aggregation operators are theoretically possible,

such as

* averaging,
* weighted summation,
* attention.

However,

summation remains the standard choice because it preserves extensivity.

---

# 22.8.11 Feature Update

The aggregated message is combined with the current atomic representation.

Symbolically,

$$
\boxed{
\mathbf h_i'
============

\phi
(
\mathbf h_i,
M_i
)
}
$$

Unlike ordinary neural networks,

the update preserves

* equivariance,
* angular momentum decomposition,
* parity information.

Consequently,

the updated representation remains physically meaningful.

---

# 22.8.12 Message Passing Depth

Although MACE uses message passing,

it generally requires fewer interaction layers than earlier architectures.

Why?

Because each message already contains higher-order information.

Instead of relying on many sequential updates,

the model constructs rich local features immediately.

This reduces the need for deep message-passing stacks.

---

# 22.8.13 Computational Efficiency

Constructing higher-order messages is more expensive than constructing pairwise messages.

However,

the increased expressiveness often reduces the number of required interaction blocks.

Consequently,

the overall computational cost remains competitive.

This balance between

message richness

and

network depth

is one of the major strengths of MACE.

---

# 22.8.14 Physical Interpretation

One may think of conventional message passing as exchanging

"neighbor reports."

Each neighbor tells the central atom only about itself.

MACE instead exchanges

"neighborhood summaries."

Each message already contains information about

* neighboring atoms,
* their orientations,
* their collective geometry,
* many-body relationships.

As a result,

the receiving atom gains a much deeper understanding of its local environment after every interaction.

---

# 22.8.15 Information Flow

The complete communication process inside a MACE interaction block may be summarized as

```text id="mace_msg7"
Atomic Features

↓

Neighbor Information

↓

Higher-Order Tensor Products

↓

Higher-Order Messages

↓

Permutation-Invariant Aggregation

↓

Equivariant Update

↓

New Atomic Features
```

This process repeats throughout the network until sufficiently rich representations have been learned.

---

# 22.8.16 Why Higher-Order Message Passing Matters

Many chemical phenomena depend on collective local environments rather than isolated bonds.

Examples include

* tetrahedral coordination,
* octahedral distortions,
* aromatic ring stability,
* hydrogen-bond networks,
* cooperative lattice motion.

Higher-order message passing enables these phenomena to be represented much more naturally than conventional pairwise communication.

This capability is one of the primary reasons for the outstanding accuracy of MACE across diverse atomistic systems.

---

# 22.8.17 Key Takeaways

MACE retains the message-passing paradigm of graph neural networks but fundamentally changes the information carried by each message. Instead of transmitting only pairwise interactions, MACE constructs higher-order equivariant messages through recursive tensor products before aggregation. These richer messages encode complex many-body correlations directly, allowing the network to capture sophisticated local chemistry with fewer interaction layers while rigorously preserving rotational symmetry.

---

## Transition to Section 22.9 — Energy Readout and Force Prediction in MACE

After several rounds of higher-order message passing, each atom possesses a highly expressive equivariant representation describing its local environment. The final task is to convert these representations into physically meaningful quantities such as total energy and atomic forces. In the next section, we will examine the readout mechanism of MACE, understand how atomic energies are computed, and see how forces are obtained automatically through differentiation of the learned energy function.

# 22.9 Energy Readout and Force Prediction in MACE

After several interaction blocks, the MACE network has transformed simple atomic embeddings into highly expressive **equivariant many-body representations**.

At this stage,

each atom possesses a feature vector that contains information about

* its chemical identity,
* neighboring atoms,
* bond lengths,
* bond angles,
* higher-order many-body correlations,
* local crystal geometry.

However,

these representations are not yet useful for atomistic simulations.

Ultimately,

a machine learning interatomic potential must predict physical quantities such as

* total energy,
* atomic forces,
* stress tensor.

This final stage is known as the **readout**.

The readout converts learned geometric representations into physically meaningful observables.

---

# 22.9.1 Purpose of the Readout Layer

The interaction blocks produce feature representations.

The readout converts those features into energy.

Conceptually,

```text id="mace_read1"
Atomic Representation

↓

Readout Network

↓

Atomic Energy
```

Unlike image classification,

where the network predicts discrete labels,

MACE predicts continuous physical quantities.

The readout therefore behaves as a regression network.

---

# 22.9.2 Local Atomic Energy

One of the central assumptions of modern machine learning interatomic potentials is

**energy decomposition**.

Instead of predicting the total energy directly,

the network predicts an energy contribution for every atom.

Mathematically,

the atomic energy is

$$
\boxed{
E_i
===

f
(
\mathbf h_i
)
}
$$

where

* (\mathbf h_i) is the final atomic representation,
* (f) is the readout neural network.

The readout is usually a small multilayer perceptron (MLP).

---

# 22.9.3 Why Predict Atomic Energies?

Predicting atomic energies offers several advantages.

First,

the total energy automatically scales with system size.

Second,

different structures containing different numbers of atoms can be handled naturally.

Third,

the model becomes **size extensive**.

Suppose two systems are completely independent.

Their total energy should satisfy

$$
\boxed{
E(A+B)
======

E(A)
+
E(B)
}
$$

Atomic energy decomposition guarantees this property automatically.

---

# 22.9.4 Total Energy Prediction

After computing every atomic contribution,

the total energy becomes

$$
\boxed{
E_{\mathrm{total}}
==================

\sum_{i=1}^{N}
E_i
}
$$

where

$$
N
$$

is the number of atoms.

Conceptually,

```text id="mace_read2"
Atom 1

↓

Energy

↓

Atom 2

↓

Energy

↓

Atom 3

↓

Energy

↓

Sum

↓

Total Energy
```

This summation is permutation invariant,

meaning that changing the order of atoms does not affect the prediction.

---

# 22.9.5 Why the Readout Uses Only Scalar Features

The internal representations inside MACE contain

* scalars,
* vectors,
* higher-order tensors.

However,

energy is a scalar quantity.

A physical energy cannot depend on the orientation of the coordinate system.

Therefore,

the readout uses only the **rotation-invariant scalar components** of the final representation.

Suppose the final feature vector contains

$$
\mathbf h_i
===========

{
\mathbf h_i^{(0)},
\mathbf h_i^{(1)},
\mathbf h_i^{(2)}
},
$$

where

* (l=0) represents scalars,
* (l=1) represents vectors,
* (l=2) represents tensors.

Only

$$
\mathbf h_i^{(0)}
$$

is passed to the energy readout.

This guarantees rotational invariance.

---

# 22.9.6 Energy Is the Primary Prediction

An important point is that

MACE predicts

**only energy**.

The network does **not**

predict forces directly.

Instead,

forces are obtained from the energy using differentiation.

This design ensures complete physical consistency.

---

# 22.9.7 Force Prediction

In classical mechanics,

force is the negative gradient of potential energy.

The same principle applies here.

For atom

$$
i,
$$

the force is

$$
\boxed{
\mathbf F_i
===========

*

\frac{\partial
E_{\mathrm{total}}
}
{\partial
\mathbf r_i}
}
$$

This equation is one of the defining properties of machine learning interatomic potentials.

Rather than learning energy and force independently,

the model learns a single energy surface.

Forces emerge automatically.

---

# 22.9.8 Why Automatic Differentiation Works

Every operation inside MACE is differentiable.

These operations include

* embeddings,
* radial basis functions,
* spherical harmonics,
* tensor products,
* linear layers,
* readout network.

Consequently,

automatic differentiation can compute

$$
\frac{\partial E}{\partial \mathbf r_i}
$$

exactly.

PyTorch performs this differentiation efficiently during both training and inference.

---

# 22.9.9 Energy–Force Consistency

Because forces originate from the energy,

they are guaranteed to satisfy physical conservation laws.

Suppose the predicted energy changes continuously.

Then,

the derived forces also change continuously.

This consistency is essential for

* molecular dynamics,
* geometry optimization,
* phonon calculations.

Predicting forces independently could violate these physical relationships.

---

# 22.9.10 Stress Tensor Prediction

For crystalline materials,

another important quantity is the stress tensor.

Stress measures how the energy changes under deformation of the simulation cell.

Mathematically,

the stress tensor is obtained as

$$
\boxed{
\sigma
======

\frac{1}{V}
\frac{\partial
E
}
{\partial
\varepsilon}
}
$$

where

* (V) is the cell volume,
* (\varepsilon) denotes strain.

Like forces,

stress is computed through automatic differentiation.

---

# 22.9.11 Advantages of Energy-Based Learning

Predicting energy first offers several important advantages.

### Physical Consistency

Forces always remain consistent with the energy surface.

---

### Energy Conservation

Molecular dynamics simulations conserve energy more accurately.

---

### Stable Optimization

Geometry optimization becomes more reliable because forces always correspond to an underlying potential.

---

### Better Generalization

The network learns one physically meaningful function instead of multiple unrelated targets.

---

# 22.9.12 Multi-Task Training

Although the forward pass predicts only energy,

training often uses multiple supervision signals.

The loss function may include

* energy loss,
* force loss,
* stress loss.

Conceptually,

```text id="mace_read3"
Predicted Energy

↓

Automatic Differentiation

↓

Predicted Forces

↓

Predicted Stress

↓

Combined Loss
```

All three quantities contribute to optimization.

---

# 22.9.13 Computational Graph

The complete computational graph for prediction is

```text id="mace_read4"
Atomic Coordinates

↓

MACE

↓

Atomic Features

↓

Readout

↓

Atomic Energies

↓

Summation

↓

Total Energy

↓

Automatic Differentiation

↓

Forces & Stress
```

Every quantity ultimately depends on the atomic coordinates.

---

# 22.9.14 Comparison with Classical Potentials

Traditional force fields compute

energy

using analytical equations,

then derive forces analytically.

For example,

a Lennard–Jones potential uses

$$
E(r)
====

4\varepsilon
\left[
\left(
\frac{\sigma}{r}
\right)^{12}
------------

\left(
\frac{\sigma}{r}
\right)^6
\right].
$$

Forces follow from

$$
F
=

*

\frac{dE}{dr}.
$$

MACE follows exactly the same physical principle.

The only difference is that

the energy function is represented by a neural network rather than a fixed analytical equation.

---

# 22.9.15 Practical Workflow

During inference,

the computational procedure is remarkably simple.

```text id="mace_read5"
Atomic Structure

↓

MACE Forward Pass

↓

Total Energy

↓

Automatic Differentiation

↓

Atomic Forces

↓

Molecular Dynamics
```

This workflow is repeated millions of times during atomistic simulations.

---

# 22.9.16 Why This Design Is So Powerful

One of the greatest strengths of MACE is that it does not attempt to learn every physical quantity separately.

Instead,

it learns the **potential energy surface** itself.

Once this surface has been learned accurately,

all derivative quantities become immediately available,

including

* forces,
* stress,
* elastic response,
* phonons,
* vibrational properties.

Thus,

a single neural network provides access to a wide range of atomistic observables.

---

# 22.9.17 Key Takeaways

The final stage of MACE converts higher-order equivariant atomic representations into atomic energy contributions using a small invariant readout network. The total energy is obtained by summing all atomic energies, ensuring size extensivity and permutation invariance. Rather than predicting forces directly, MACE computes them through automatic differentiation of the learned energy function, guaranteeing complete physical consistency between energy, forces, and stress. This energy-first design mirrors the principles of classical interatomic potentials while leveraging the expressive power of modern equivariant deep learning.

---

## Transition to Section 22.10 — Training MACE: Loss Functions, Optimization, and Hyperparameters

With the architecture now complete, the next step is understanding how MACE learns from quantum mechanical reference data. In the following section, we will examine the complete training procedure, including dataset preparation, energy and force supervision, loss functions, optimization algorithms, hyperparameter selection, and practical considerations for developing high-accuracy MACE interatomic potentials.

# 22.10 Training MACE: Loss Functions, Optimization, and Hyperparameters

Designing an expressive neural network architecture is only half of the problem.

The second half is **training**.

A poorly trained MACE model, regardless of how sophisticated its architecture may be, will produce inaccurate energies and unstable molecular dynamics trajectories.

Conversely, a well-trained MACE model can achieve **Density Functional Theory (DFT)-level accuracy** while being several orders of magnitude faster during inference.

Training MACE involves much more than simply minimizing an error function.

The model must learn

* quantum mechanical energies,
* atomic forces,
* stress tensors,
* chemical diversity,
* structural diversity,
* physical symmetries,

while simultaneously avoiding overfitting.

This section examines the complete training pipeline used in modern MACE models.

---

# 22.10.1 Overview of the Training Pipeline

The entire workflow can be summarized as

```text id="mace_train1"
DFT Calculations

↓

Atomic Structures

↓

Neighbor Graph Construction

↓

MACE Forward Pass

↓

Predicted Energy

↓

Automatic Differentiation

↓

Predicted Forces

↓

Loss Function

↓

Backpropagation

↓

Optimizer

↓

Updated Parameters
```

This cycle is repeated for many epochs until convergence.

---

# 22.10.2 Training Data

Unlike image classification,

where labels are discrete,

MACE requires quantum mechanical reference data.

Each training sample typically contains

* atomic numbers,
* Cartesian coordinates,
* periodic cell,
* total energy,
* atomic forces,
* stress tensor (optional).

Mathematically,

one training example may be represented as

$$
\boxed{
(
\mathbf Z,
\mathbf R,
E,
\mathbf F,
\sigma
)
}
$$

where

* (\mathbf Z) denotes atomic numbers,
* (\mathbf R) denotes coordinates,
* (E) is the total energy,
* (\mathbf F) contains atomic forces,
* (\sigma) is the stress tensor.

---

# 22.10.3 Sources of Training Data

Most MACE models are trained using

Density Functional Theory calculations.

Common sources include

* Quantum ESPRESSO,
* VASP,
* ABINIT,
* CASTEP,
* GPAW,
* CP2K.

Public datasets include

* Materials Project,
* OC20,
* OC22,
* MD22,
* ANI,
* SPICE,
* NOMAD.

The quality of the training data largely determines the quality of the resulting model.

---

# 22.10.4 Why Force Labels Are Essential

Forces contain much more information than energies.

Consider a system containing

$$
N
$$

atoms.

Each configuration provides

one total energy,

but

$$
3N
$$

force components.

For a structure containing

100 atoms,

we obtain

* one energy label,
* 300 force labels.

Consequently,

forces provide far richer supervision.

Modern interatomic potentials therefore rely heavily on force matching.

---

# 22.10.5 Forward Pass

During training,

the model performs a standard forward pass.

```text id="mace_train2"
Atomic Structure

↓

MACE

↓

Atomic Energies

↓

Total Energy
```

The total energy prediction is

$$
\boxed{
E_{\mathrm{pred}}
=================

\sum_i
E_i
}
$$

No forces are predicted directly.

---

# 22.10.6 Automatic Differentiation

The predicted forces are computed automatically.

$$
\boxed{
\mathbf F_{\mathrm{pred}}
=========================

*

\frac{\partial
E_{\mathrm{pred}}
}
{\partial
\mathbf R}
}
$$

Because every layer inside MACE is differentiable,

PyTorch computes these derivatives efficiently using reverse-mode automatic differentiation.

---

# 22.10.7 Energy Loss

The simplest loss compares predicted and reference energies.

The Mean Squared Error (MSE) is commonly used.

$$
\boxed{
L_E
===

\frac{1}{N_s}
\sum
(
E_{\mathrm{pred}}
-----------------

E_{\mathrm{true}}
)^2
}
$$

where

$$
N_s
$$

is the number of structures.

---

# 22.10.8 Force Loss

Force prediction is usually the dominant component of the loss.

For every atom,

the force error becomes

$$
\boxed{
L_F
===

\frac{1}{3N}
\sum_i
|
\mathbf F_i^{\mathrm{pred}}
---------------------------

\mathbf F_i^{\mathrm{true}}
|^2
}
$$

Since every atom contributes three force components,

this term often contains far more information than the energy loss.

---

# 22.10.9 Stress Loss

For periodic crystals,

stress prediction is also important.

The stress loss is

$$
\boxed{
L_{\sigma}
==========

|
\sigma_{\mathrm{pred}}
----------------------

\sigma_{\mathrm{true}}
|^2
}
$$

Stress supervision improves predictions for

* lattice constants,
* elastic constants,
* structural relaxation.

---

# 22.10.10 Combined Loss Function

The complete training objective combines multiple loss terms.

$$
\boxed{
L
=

w_E
L_E
+
w_F
L_F
+
w_{\sigma}
L_{\sigma}
}
$$

where

* (w_E) controls the importance of energy,
* (w_F) controls force supervision,
* (w_{\sigma}) controls stress supervision.

Choosing these weights appropriately is one of the most important aspects of MACE training.

---

# 22.10.11 Why Force Loss Usually Dominates

Since each structure contributes only

one energy,

but

hundreds or even thousands of force components,

most MACE models assign greater importance to force prediction.

Conceptually,

```text id="mace_train3"
Energy

↓

Small Contribution

Force

↓

Large Contribution

Stress

↓

Optional Contribution

↓

Total Loss
```

Force supervision enables the model to learn the shape of the potential energy surface much more accurately.

---

# 22.10.12 Backpropagation

Once the total loss has been computed,

gradients are obtained using backpropagation.

Mathematically,

the gradient of every trainable parameter

$$
\theta
$$

is

$$
\boxed{
\frac{\partial
L
}
{\partial
\theta}
}
$$

PyTorch automatically computes these gradients using the computational graph generated during the forward pass.

---

# 22.10.13 Optimizer

After gradients have been computed,

the optimizer updates the model parameters.

The most common optimizer is **Adam**.

The parameter update is

$$
\boxed{
\theta
\leftarrow
\theta
------

\eta
\nabla_{\theta}
L
}
$$

where

* (\eta) is the learning rate.

Although Adam internally uses adaptive moment estimation, this equation captures the general optimization principle.

---

# 22.10.14 Learning Rate Scheduling

Training usually begins with a relatively large learning rate.

As optimization progresses,

the learning rate is gradually reduced.

Conceptually,

```text id="mace_train4"
Learning Rate

↑

|

|\

| \

|  \

|   \_____

+---------------->

Training Epoch
```

Learning rate scheduling improves convergence and often leads to lower final errors.

---

# 22.10.15 Batch Training

Instead of processing the entire dataset simultaneously,

training uses mini-batches.

Conceptually,

```text id="mace_train5"
Dataset

↓

Mini-Batch 1

↓

Update

↓

Mini-Batch 2

↓

Update

↓

Mini-Batch 3

↓

Update
```

Mini-batch optimization reduces memory usage while accelerating training.

---

# 22.10.16 Epochs

One **epoch** corresponds to processing every structure in the training dataset once.

Training typically requires

* tens,
* hundreds,
* or even thousands of epochs,

depending on

* dataset size,
* model complexity,
* convergence criteria.

---

# 22.10.17 Validation

Training performance alone is insufficient.

A separate validation dataset monitors generalization.

Typical validation metrics include

* Energy Mean Absolute Error (MAE),
* Force MAE,
* Stress MAE.

If validation error begins increasing while training error continues decreasing,

the model is overfitting.

---

# 22.10.18 Early Stopping

To prevent overfitting,

training often employs early stopping.

Conceptually,

```text id="mace_train6"
Validation Error

↓

Improves

↓

Plateaus

↓

Begins Increasing

↓

Stop Training
```

The parameters from the best validation epoch are retained.

---

# 22.10.19 Important Hyperparameters

Several hyperparameters strongly influence MACE performance.

These include

| Hyperparameter                        | Role                                      |
| ------------------------------------- | ----------------------------------------- |
| Cutoff radius                         | Neighbor interactions                     |
| Number of interaction blocks          | Network depth                             |
| Correlation order                     | Many-body complexity                      |
| Maximum angular momentum ((l_{\max})) | Equivariant feature richness              |
| Number of channels                    | Feature dimensionality                    |
| Learning rate                         | Optimization speed                        |
| Batch size                            | Training stability                        |
| Weight decay                          | Regularization                            |
| Loss weights                          | Balance between energy, force, and stress |

Proper tuning of these parameters is essential for achieving high accuracy.

---

# 22.10.20 Computational Requirements

Training MACE is computationally intensive.

Typical requirements include

* modern GPUs,
* substantial GPU memory,
* efficient tensor-product implementations,
* optimized PyTorch and e3nn libraries.

Training large foundation models may require

multiple GPUs

or distributed training across computing clusters.

---

# 22.10.21 Key Takeaways

Training MACE involves optimizing an equivariant neural network using quantum mechanical reference data. The model predicts total energy, derives forces and stress through automatic differentiation, and minimizes a combined loss containing energy, force, and stress errors. Force supervision usually dominates because it provides significantly richer information about the potential energy surface. Successful training depends not only on the architecture but also on high-quality datasets, carefully chosen hyperparameters, effective optimization strategies, and rigorous validation procedures.

---

## Transition to Section 22.11 — Implementing MACE with PyTorch and e3nn

Having understood the theoretical foundations and training methodology of MACE, we are now ready to move from theory to implementation. In the next section, we will build a practical MACE workflow using **PyTorch**, **e3nn**, and the official **MACE** software package, learning how to prepare datasets, configure models, train interatomic potentials, evaluate their performance, and perform atomistic simulations.

# 22.11 Implementing MACE with PyTorch and e3nn

Up to this point, we have focused on the theoretical foundations of MACE.

We now understand

* equivariant representations,
* recursive tensor products,
* higher-order message passing,
* energy prediction,
* force computation,
* training procedures.

The next step is equally important:

**implementing MACE in practice.**

Fortunately, researchers do **not** need to implement the complete MACE architecture from scratch.

The official MACE project provides an optimized implementation built on top of

* PyTorch,
* e3nn,
* ASE (Atomic Simulation Environment).

These libraries handle

* equivariant tensor operations,
* automatic differentiation,
* neighbor graph construction,
* molecular dynamics interfaces,
* GPU acceleration.

In this section, we will study how to install, configure, train, and use MACE for practical atomistic simulations.

---

# 22.11.1 Software Stack

The MACE ecosystem consists of several interconnected software packages.

```text id="mace_impl1"
PyTorch

↓

e3nn

↓

MACE

↓

ASE

↓

Atomistic Simulations
```

Each package has a specific role.

| Library | Purpose                                |
| ------- | -------------------------------------- |
| PyTorch | Deep learning framework                |
| e3nn    | Equivariant neural network operations  |
| MACE    | Machine learning interatomic potential |
| ASE     | Structure handling and simulations     |

---

# 22.11.2 Why e3nn Is Required

Implementing equivariant tensor products manually would be extremely difficult.

For example,

the network must correctly compute

* spherical harmonics,
* irreducible representations,
* Clebsch–Gordan tensor products,
* equivariant linear layers.

The **e3nn** library provides efficient implementations of these operations.

Without e3nn,

building MACE from scratch would require thousands of lines of highly specialized mathematical code.

---

# 22.11.3 Installing MACE

The recommended approach is to create a dedicated Python environment.

```bash
conda create -n mace python=3.11
```

Activate the environment.

```bash
conda activate mace
```

Install PyTorch.

```bash
pip install torch torchvision torchaudio
```

Install e3nn.

```bash
pip install e3nn
```

Finally,

install the official MACE package.

```bash
pip install mace-torch
```

The exact installation command may change in future releases, so users should always consult the official project documentation for the latest instructions.

---

# 22.11.4 Verifying the Installation

A simple Python script can verify that MACE has been installed successfully.

```python
import torch
import e3nn
import mace

print("PyTorch:", torch.__version__)
print("e3nn:", e3nn.__version__)
print("MACE installed successfully!")
```

If the script executes without errors,

the software stack is ready.

---

# 22.11.5 Preparing the Dataset

MACE expects atomistic structures together with reference properties.

Each structure should contain

* atomic numbers,
* coordinates,
* energies,
* forces,
* optional stress tensors.

Conceptually,

```text id="mace_impl2"
Atomic Structure

+

Energy

+

Forces

+

Stress

↓

Training Sample
```

These data are typically obtained from DFT calculations.

---

# 22.11.6 Supported Data Formats

The official implementation supports several common atomistic formats.

Examples include

* `.xyz`
* `.extxyz`
* `.traj`
* ASE databases
* custom datasets converted into ASE objects

Among these,

**Extended XYZ (extxyz)** is the most commonly used format because it stores

* atomic positions,
* lattice vectors,
* energies,
* forces,

within a single file.

---

# 22.11.7 Example of an Extended XYZ File

A simplified example is shown below.

```text
2
energy=-5.421
H      0.000    0.000    0.000
H      0.740    0.000    0.000
```

In practical datasets,

additional information such as forces and lattice vectors is also included.

---

# 22.11.8 Dataset Loading

Using ASE,

loading structures is straightforward.

```python
from ase.io import read

structures = read("dataset.extxyz", index=":")
```

The variable

```python
structures
```

contains a list of atomic configurations that can be passed directly to MACE.

---

# 22.11.9 Data Splitting

The dataset is usually divided into

* training set,
* validation set,
* test set.

A common split is

```text id="mace_impl3"
Dataset

↓

Training (80%)

Validation (10%)

Test (10%)
```

The validation set is used during training,

while the test set remains untouched until final evaluation.

---

# 22.11.10 Model Configuration

Before training,

the architecture must be specified.

Typical configuration parameters include

* cutoff radius,
* maximum angular momentum,
* correlation order,
* number of interaction blocks,
* hidden channels.

Conceptually,

```text id="mace_impl4"
Configuration

↓

Model Initialization
```

These parameters determine the capacity of the model.

---

# 22.11.11 Typical Hyperparameters

Although the optimal values depend on the dataset,

a representative configuration might include

| Parameter                             | Typical Value |
| ------------------------------------- | ------------- |
| Cutoff radius                         | 5–6 Å         |
| Correlation order                     | 3 or 4        |
| Maximum angular momentum ((l_{\max})) | 2 or 3        |
| Number of interaction blocks          | 2–4           |
| Hidden channels                       | 64–256        |

Larger models generally provide higher accuracy but require more computational resources.

---

# 22.11.12 Starting Training

The official MACE package provides command-line utilities for training.

A typical workflow resembles

```bash
mace_run_train \
    --train_file train.extxyz \
    --valid_file valid.extxyz \
    --model_dir models/
```

Additional command-line options specify

* learning rate,
* cutoff radius,
* correlation order,
* batch size,
* optimizer settings.

These options vary slightly between software versions.

---

# 22.11.13 Training Workflow

During training,

the following sequence repeats.

```text id="mace_impl5"
Mini-Batch

↓

Forward Pass

↓

Energy Prediction

↓

Force Computation

↓

Loss

↓

Backpropagation

↓

Parameter Update
```

This process continues until convergence.

---

# 22.11.14 Monitoring Training

Several quantities are monitored during optimization.

These include

* energy loss,
* force loss,
* validation loss,
* learning rate.

Conceptually,

```text id="mace_impl6"
Epoch

↓

Training Loss

↓

Validation Loss

↓

Checkpoint
```

The best-performing checkpoint is usually selected according to validation performance.

---

# 22.11.15 Saving the Model

After training,

the optimized neural network parameters are stored.

```python
torch.save(model.state_dict(), "mace_model.pt")
```

Later,

the model can be reloaded without retraining.

---

# 22.11.16 Loading a Trained Model

A trained model is restored using

```python
model.load_state_dict(torch.load("mace_model.pt"))
model.eval()
```

The call to

```python
eval()
```

switches the network into inference mode.

---

# 22.11.17 Performing Energy Prediction

Once loaded,

the model predicts atomic energies for new structures.

Conceptually,

```text id="mace_impl7"
New Structure

↓

Trained MACE

↓

Predicted Energy
```

The prediction procedure is identical to the forward pass used during training,

except that no parameter updates occur.

---

# 22.11.18 Molecular Dynamics with ASE

One of the major strengths of MACE is its integration with ASE.

Once the trained model is attached as a calculator,

ASE can perform

* molecular dynamics,
* geometry optimization,
* vibrational analysis,
* structure relaxation.

Conceptually,

```text id="mace_impl8"
ASE

↓

MACE Calculator

↓

Energy

↓

Forces

↓

Simulation
```

This allows MACE models to replace expensive DFT calculations in many workflows.

---

# 22.11.19 GPU Acceleration

Training and inference benefit enormously from GPU execution.

PyTorch automatically moves tensors to the GPU.

```python
device = torch.device("cuda")
model.to(device)
```

Modern NVIDIA GPUs can accelerate tensor-product operations by more than an order of magnitude compared with CPU execution.

---

# 22.11.20 Best Practices

When implementing MACE in research projects,

several practical recommendations improve reliability.

* Normalize energies when appropriate.
* Ensure force labels are accurate.
* Shuffle training data every epoch.
* Monitor validation performance.
* Save checkpoints regularly.
* Use mixed-precision training when supported.
* Verify predictions on unseen structures before production simulations.

These practices reduce training instability and improve generalization.

---

# 22.11.21 Common Implementation Challenges

New users frequently encounter several issues.

| Problem              | Possible Cause                                     |
| -------------------- | -------------------------------------------------- |
| Slow training        | Large correlation order or insufficient GPU memory |
| Poor accuracy        | Low-quality training data                          |
| Overfitting          | Dataset too small or model too large               |
| Diverging loss       | Learning rate too high                             |
| Incorrect forces     | Missing or inconsistent force labels               |
| Out-of-memory errors | Batch size too large                               |

Understanding these issues greatly simplifies model development.

---

# 22.11.22 Complete Practical Workflow

The entire implementation process can be summarized as

```text id="mace_impl9"
DFT Dataset

↓

ASE Structures

↓

MACE Training

↓

Saved Model

↓

ASE Calculator

↓

Molecular Dynamics

↓

Materials Simulation
```

This workflow represents how MACE is commonly used in modern computational materials science.

---

# 22.11.23 Key Takeaways

The official MACE software provides a practical and highly optimized implementation built on top of PyTorch, e3nn, and ASE. Rather than implementing equivariant tensor operations manually, researchers can focus on preparing high-quality datasets, configuring appropriate model hyperparameters, training robust interatomic potentials, and integrating the resulting models into atomistic simulation workflows. This software ecosystem enables MACE to serve as a practical replacement for many expensive quantum mechanical calculations while maintaining near-DFT accuracy.

---

## Transition to Section 22.12 — MACE in Real Materials Science Applications

With the implementation complete, the final step is understanding how MACE is applied to real scientific problems. In the next section, we will explore practical applications across materials science, including molecular dynamics, crystal structure optimization, defect calculations, phase stability, diffusion, battery materials, catalysis, and large-scale simulations that are computationally prohibitive with conventional DFT methods.



# 22.12 MACE in Real Materials Science Applications

Developing an accurate interatomic potential is **not** the final objective.

The real purpose of MACE is to enable scientific discovery.

By learning a quantum mechanically accurate potential energy surface, MACE allows researchers to investigate systems that are far beyond the reach of conventional Density Functional Theory (DFT).

A DFT calculation that requires

* several hours,
* several days,
* or even several weeks

may often be replaced by a MACE prediction that requires only

* milliseconds,
* or a few seconds.

This enormous computational acceleration enables entirely new classes of atomistic simulations.

In this section, we examine how MACE is applied across modern materials science research.

---

# 22.12.1 Why Machine Learning Potentials Matter

Before discussing specific applications,

it is useful to compare three common approaches for atomistic simulations.

| Method                    | Accuracy     | Computational Cost | Transferability               |
| ------------------------- | ------------ | ------------------ | ----------------------------- |
| Classical force fields    | Low–Moderate | Very Low           | Limited                       |
| Density Functional Theory | Very High    | Very High          | High                          |
| MACE                      | Near-DFT     | Low                | High (within training domain) |

MACE occupies an attractive middle ground.

It retains much of the accuracy of DFT while approaching the computational efficiency of classical force fields.

---

# 22.12.2 Large-Scale Molecular Dynamics

One of the most important applications of MACE is **molecular dynamics (MD)**.

In MD,

Newton's equations of motion are solved repeatedly.

The force acting on every atom determines how the structure evolves with time.

The governing equation is

$$
\boxed{
m_i
\frac{d^2\mathbf r_i}{dt^2}
===========================

\mathbf F_i
}
$$

Since MACE predicts forces much faster than DFT,

simulations that were previously impossible become practical.

---

# 22.12.3 Long-Time Simulations

DFT-based molecular dynamics is typically limited to

* hundreds of atoms,
* tens of picoseconds.

MACE enables simulations involving

* thousands of atoms,
* nanoseconds,
* or even microseconds,

depending on the available computational resources.

This allows researchers to study slow physical processes that cannot be observed with conventional quantum mechanical simulations.

---

# 22.12.4 Crystal Structure Optimization

Another common application is structural relaxation.

The objective is to find the atomic coordinates corresponding to the minimum potential energy.

Conceptually,

```text id="mace_app1"
Initial Structure

↓

Energy

↓

Forces

↓

Geometry Optimization

↓

Relaxed Structure
```

During optimization,

the atomic positions are updated until the forces become sufficiently small.

---

# 22.12.5 Lattice Optimization

For periodic materials,

not only atomic positions,

but also lattice vectors can be optimized.

The optimization simultaneously minimizes

* atomic forces,
* stress tensor.

This enables prediction of

* equilibrium lattice constants,
* equilibrium crystal structures,
* elastic equilibrium.

---

# 22.12.6 Point Defects

Defects strongly influence the properties of real materials.

MACE enables efficient calculations of

* vacancies,
* interstitials,
* substitutional defects,
* antisite defects.

Large supercells that are prohibitively expensive for DFT become computationally accessible.

---

# 22.12.7 Diffusion

Atomic diffusion governs numerous materials processes,

including

* battery operation,
* alloy formation,
* corrosion,
* creep,
* ionic conductivity.

By performing long molecular dynamics simulations,

MACE enables direct observation of diffusion events.

The diffusion coefficient is often computed from the mean squared displacement.

$$
\boxed{
D
=

\lim_{t\rightarrow\infty}
\frac{
\langle
|\mathbf r(t)-\mathbf r(0)|^2
\rangle
}
{6t}
}
$$

Long trajectories are essential for obtaining reliable diffusion coefficients.

---

# 22.12.8 Phase Stability

Many materials undergo structural phase transitions.

Examples include

* solid-solid transitions,
* melting,
* crystallization.

Because MACE permits simulations over large spatial and temporal scales,

it can capture these transformations much more effectively than DFT.

---

# 22.12.9 Thermal Properties

MACE can also predict temperature-dependent properties,

including

* thermal expansion,
* heat capacity,
* thermal conductivity,
* phonon dynamics.

These properties require extensive sampling of atomic motion,

which becomes computationally practical using machine learning potentials.

---

# 22.12.10 Mechanical Properties

Mechanical deformation can be investigated by applying strain to the simulation cell.

Conceptually,

```text id="mace_app2"
Crystal

↓

Apply Strain

↓

Atomic Relaxation

↓

Stress

↓

Mechanical Properties
```

This workflow enables calculation of

* elastic constants,
* stress-strain curves,
* fracture behavior,
* plastic deformation.

---

# 22.12.11 Surface Science

Surfaces possess atomic environments that differ significantly from the bulk.

MACE has been widely applied to

* surface relaxation,
* reconstruction,
* adsorption,
* surface diffusion.

These studies are especially important for catalysis.

---

# 22.12.12 Catalysis

Catalytic reactions often involve

* adsorbates,
* transition states,
* bond breaking,
* bond formation.

These processes require highly accurate interatomic potentials.

MACE models trained on appropriate datasets can reproduce catalytic energy landscapes with near-DFT accuracy while dramatically reducing computational cost.

---

# 22.12.13 Battery Materials

Battery research is one of the fastest-growing applications of machine learning interatomic potentials.

Typical investigations include

* lithium diffusion,
* sodium diffusion,
* interface stability,
* solid electrolytes,
* electrode degradation,
* dendrite formation.

Because these phenomena require large simulation cells and long timescales,

MACE is particularly well suited for such studies.

---

# 22.12.14 Amorphous Materials

Unlike crystals,

amorphous materials lack long-range order.

Large simulation cells are therefore necessary.

Examples include

* glasses,
* amorphous silicon,
* oxide glasses,
* polymers.

MACE enables simulations involving thousands of atoms,

making realistic amorphous structure generation feasible.

---

# 22.12.15 Grain Boundaries

Polycrystalline materials contain numerous grain boundaries.

These interfaces influence

* strength,
* diffusion,
* corrosion,
* fracture.

Grain-boundary models often contain tens of thousands of atoms,

placing them beyond routine DFT calculations.

Machine learning potentials provide an effective alternative.

---

# 22.12.16 Dislocations

Plastic deformation occurs through the motion of dislocations.

Studying dislocation dynamics requires

* large simulation cells,
* long trajectories.

MACE enables atomistic investigation of

* dislocation nucleation,
* glide,
* interaction,
* pinning.

---

# 22.12.17 High-Entropy Alloys

High-entropy alloys contain many different chemical species.

The enormous configurational complexity makes exhaustive DFT calculations impractical.

MACE models trained on representative alloy configurations can efficiently explore

* phase stability,
* local ordering,
* defect formation,
* mechanical behavior.

---

# 22.12.18 Radiation Damage

Radiation creates energetic collision cascades involving thousands of atoms.

These events evolve extremely rapidly and require millions of force evaluations.

MACE has become an attractive approach for studying

* displacement cascades,
* defect clustering,
* radiation tolerance,
* nuclear materials.

---

# 22.12.19 Computational Screening

Because MACE predictions are fast,

large numbers of candidate structures can be evaluated.

Conceptually,

```text id="mace_app3"
Thousands of Structures

↓

MACE Evaluation

↓

Energy Ranking

↓

Most Promising Candidates
```

This dramatically accelerates materials discovery.

---

# 22.12.20 Coupling with Active Learning

Many research workflows combine MACE with active learning.

The cycle proceeds as follows.

```text id="mace_app4"
Initial Dataset

↓

Train MACE

↓

Run Simulation

↓

Detect Uncertain Structures

↓

DFT Calculations

↓

Expand Dataset

↓

Retrain
```

This iterative strategy improves accuracy while minimizing the number of expensive DFT calculations.

---

# 22.12.21 Foundation Models

Recent MACE models have been trained on millions of quantum mechanical structures.

Instead of specializing in a single material,

these **foundation models** generalize across

* molecules,
* crystals,
* surfaces,
* liquids,
* interfaces.

Fine-tuning such pretrained models requires much less data than training from scratch, making them highly attractive for new research problems.

---

# 22.12.22 Advantages and Limitations

Although MACE is extremely powerful,

it is not without limitations.

### Advantages

* Near-DFT accuracy
* Orders-of-magnitude faster than DFT
* Equivariant by construction
* Excellent scalability
* Suitable for molecular dynamics
* Applicable to diverse materials systems

### Limitations

* Performance depends strongly on the training dataset.
* Extrapolation beyond the training domain can produce unreliable predictions.
* Training requires substantial computational resources.
* Foundation models improve generalization but do not eliminate extrapolation errors.

Recognizing these limitations is essential for responsible scientific use.

---

# 22.12.23 Future Directions

Research on MACE continues to evolve rapidly.

Current directions include

* larger foundation models,
* uncertainty-aware potentials,
* active learning,
* multi-fidelity training,
* integration with diffusion models,
* autonomous laboratories,
* inverse materials design.

These developments are expected to make machine learning interatomic potentials an increasingly central tool in computational materials science.

---

# 22.12.24 Key Takeaways

MACE has transformed atomistic simulation by combining near-DFT accuracy with computational efficiency approaching that of classical force fields. Its applications span molecular dynamics, crystal structure optimization, defect physics, diffusion, catalysis, battery materials, mechanical behavior, radiation damage, high-entropy alloys, and large-scale computational screening. When combined with active learning and foundation-model pretraining, MACE provides a powerful framework for accelerating materials discovery and enabling simulations that were previously computationally infeasible.

---

# End of Chapter 22

By the end of this chapter, you should be able to:

* Explain the motivation behind MACE and how it extends NequIP.
* Understand the mathematical foundations of higher-order equivariant tensor products.
* Describe the architecture of MACE, including interaction blocks, recursive correlations, and higher-order message passing.
* Explain how energies, forces, and stresses are predicted from learned atomic representations.
* Train a MACE model using quantum mechanical datasets and understand the role of loss functions, optimization, and hyperparameter selection.
* Implement MACE using PyTorch, e3nn, and the official MACE software package.
* Apply MACE to real-world materials science problems, including molecular dynamics, defect simulations, diffusion studies, catalysis, and large-scale materials discovery.

With this foundation established, the next chapter will build on these ideas by introducing **Chapter 23 — Representation Learning for Materials**, where we move beyond interatomic potentials and explore how neural networks learn compact, chemically meaningful latent representations that power modern foundation models and generative AI for materials discovery.



# 22.13 Code Implementation with MACE

Understanding the mathematical foundations of MACE is only the first step toward applying it in real-world research. To build a practical machine learning interatomic potential, we must translate the theoretical concepts developed throughout this chapter into an executable workflow.

In this section, we will implement MACE from the ground up using the official **mace-torch** package together with **PyTorch**, **e3nn**, and the **Atomic Simulation Environment (ASE)**. Unlike the earlier sections, which focused on concepts and mathematical derivations, this section is entirely practical. Every major step in a typical research workflow will be demonstrated with executable code and accompanied by detailed explanations.

By the end of this section, you will be able to

- install the complete MACE software stack,
- prepare atomistic datasets,
- inspect and preprocess training structures,
- train a MACE model,
- monitor training,
- evaluate model accuracy,
- predict energies and forces,
- perform structural relaxation,
- run molecular dynamics simulations,
- and use trained MACE models in your own research projects.

The implementation presented here follows the workflow commonly used in modern computational materials science research.

---

# 22.13.1 Setting Up the Python Environment

Before installing MACE, it is highly recommended to create a dedicated Python environment. This avoids conflicts with other scientific software and ensures reproducibility.

For users of **Conda**, create a new environment with Python 3.11.

```bash
conda create -n mace python=3.11
```

Activate the environment.

```bash
conda activate mace
```

Verify that the correct Python interpreter is active.

```bash
python --version
```

Example output

```text
Python 3.11.x
```

Using an isolated environment makes it easier to

- manage package dependencies,
- reproduce experiments,
- update libraries independently,
- avoid conflicts between different machine learning frameworks.

---

# 22.13.2 Checking GPU Availability

Although MACE can run on CPUs, training modern models on a CPU is usually impractical. A CUDA-enabled GPU is strongly recommended.

Before installing PyTorch, verify that the NVIDIA driver detects your GPU.

```bash
nvidia-smi
```

Typical output

```text
+------------------------------------------------------+
| GPU Name                  Memory Usage               |
| NVIDIA RTX 4090           0 MiB / 24564 MiB          |
+------------------------------------------------------+
```

If this command fails,

- install the NVIDIA driver,
- verify CUDA installation,
- restart the system if necessary.

If no GPU is available, MACE will still function, but training may be significantly slower.

---

# 22.13.3 Installing PyTorch

Install the latest version of PyTorch compatible with your CUDA version.

For example,

```bash
pip install torch torchvision torchaudio
```

To verify the installation, open Python.

```bash
python
```

Then execute

```python
import torch

print(torch.__version__)
```

Example output

```text
2.8.0
```

Next, verify GPU support.

```python
import torch

print(torch.cuda.is_available())
```

Example output

```text
True
```

If the output is

```text
False
```

then PyTorch cannot detect CUDA.

This issue should be resolved before continuing.

---

# 22.13.4 Understanding the Available GPU

It is useful to inspect the GPU that PyTorch will use.

```python
import torch

print(torch.cuda.get_device_name(0))
```

Example output

```text
NVIDIA GeForce RTX 4090
```

The total GPU memory may also be queried.

```python
import torch

memory = torch.cuda.get_device_properties(0).total_memory

print(memory / 1024**3, "GB")
```

Example output

```text
24.0 GB
```

Knowing the available memory helps determine

- batch size,
- model size,
- correlation order.

---

# 22.13.5 Installing e3nn

MACE relies heavily on the **e3nn** library for equivariant neural network operations.

Install it using

```bash
pip install e3nn
```

Verify the installation.

```python
import e3nn

print(e3nn.__version__)
```

If no errors appear, the installation is complete.

---

# 22.13.6 Installing the Official MACE Package

The official implementation is distributed through **mace-torch**.

Install it using

```bash
pip install mace-torch
```

This package installs

- MACE
- training utilities
- inference tools
- pretrained models
- command-line interfaces

After installation, verify that Python can import the package.

```python
import mace

print("MACE installed successfully.")
```

If this command executes successfully, the core software stack is ready.

---

# 22.13.7 Installing ASE

Most practical MACE workflows use the **Atomic Simulation Environment (ASE)**.

Install ASE.

```bash
pip install ase
```

Verify installation.

```python
import ase

print(ase.__version__)
```

ASE provides

- structure manipulation,
- geometry optimization,
- molecular dynamics,
- file input/output,
- visualization.

Nearly every practical example in the remainder of this chapter will use ASE.

---

# 22.13.8 Installing Supporting Libraries

Several additional scientific libraries are useful.

```bash
pip install numpy scipy matplotlib pandas scikit-learn tqdm
```

These packages provide

- numerical computation,
- plotting,
- data analysis,
- progress bars,
- machine learning utilities.

---

# 22.13.9 Verifying the Complete Installation

A useful verification script imports every required package.

```python
import torch
import e3nn
import ase
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())
print("e3nn :", e3nn.__version__)
print("ASE :", ase.__version__)

print("Installation Successful!")
```

A successful output indicates that the environment is ready for MACE development.

---

# 22.13.10 Recommended Project Structure

Organizing files properly becomes increasingly important as projects grow.

A recommended directory structure is

```text
MACE_Project/

├── data/
│   ├── train.extxyz
│   ├── valid.extxyz
│   └── test.extxyz

├── models/

├── checkpoints/

├── logs/

├── figures/

├── notebooks/

├── scripts/

└── outputs/
```

Each directory has a clear purpose.

| Directory | Purpose |
|------------|----------|
| `data/` | Training datasets |
| `models/` | Final trained models |
| `checkpoints/` | Intermediate training checkpoints |
| `logs/` | Training logs |
| `figures/` | Plots and visualizations |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Python scripts |
| `outputs/` | Prediction results |

Maintaining an organized project structure greatly improves reproducibility and collaboration.

---

# 22.13.11 Why Environment Setup Matters

Many issues encountered during machine learning research arise not from the model itself but from inconsistent software environments.

Examples include

- incompatible CUDA versions,
- mismatched PyTorch builds,
- outdated dependencies,
- conflicting Python packages.

Taking the time to build a clean environment at the beginning of a project prevents many hours of debugging later.

---

# 22.13.12 Key Takeaways

Before training or using a MACE model, it is essential to prepare a stable software environment. This includes installing PyTorch with CUDA support, the **e3nn** library for equivariant operations, the official **mace-torch** implementation, and **ASE** for atomistic simulations. A properly configured environment, together with a well-organized project directory, provides the foundation for all subsequent MACE workflows.

---

## Transition to Section 22.13.13

With the software environment successfully configured, the next step is learning how atomistic structures are represented inside Python. Every dataset, simulation, and prediction in the MACE ecosystem revolves around the **ASE `Atoms` object**. In the next section, we will examine this fundamental data structure in detail, exploring how it stores atomic coordinates, lattice vectors, chemical species, periodic boundary conditions, energies, forces, and additional metadata that form the basis of all practical MACE implementations.

# 22.13.13 Understanding the ASE `Atoms` Object

Before we can train a MACE model, we must understand the most important data structure in the entire ASE ecosystem—the **`Atoms` object**.

Nearly every operation performed by MACE begins with an ASE `Atoms` object. Whether we

- load a dataset,
- build a neighbor graph,
- train a neural network,
- predict energies,
- calculate forces,
- optimize structures,
- or run molecular dynamics,

the input is always an `Atoms` object.

For this reason, understanding this class is essential before proceeding to model training.

---

# 22.13.13.1 What Is an ASE `Atoms` Object?

The `Atoms` class is the central object of the **Atomic Simulation Environment (ASE)**.

It represents an atomistic system in Python.

Unlike a simple coordinate matrix, an `Atoms` object stores almost everything required to describe a material or molecule.

An `Atoms` object contains

- atomic symbols,
- atomic numbers,
- Cartesian coordinates,
- lattice vectors,
- periodic boundary conditions,
- velocities,
- magnetic moments,
- charges,
- energies,
- forces,
- stress,
- additional user-defined information.

Conceptually,

```text
                ASE Atoms Object
                        │
     ┌──────────────────┼──────────────────┐
     │                  │                  │
 Atomic Data       Structural Data    Physical Data
     │                  │                  │
     ├── Symbols         ├── Positions      ├── Energy
     ├── Numbers         ├── Cell           ├── Forces
     ├── Masses          ├── PBC            ├── Stress
     ├── Charges         ├── Constraints    ├── Velocity
     └── Spins           └── Neighbors      └── Metadata
```

Everything required by MACE is ultimately extracted from this object.

---

# 22.13.13.2 Importing ASE

The required imports are

```python
from ase import Atoms
from ase.io import read, write
```

Here,

- `Atoms` creates new structures,
- `read()` loads existing files,
- `write()` exports structures.

---

# 22.13.13.3 Creating Your First `Atoms` Object

Suppose we want to construct a hydrogen molecule.

```python
from ase import Atoms

atoms = Atoms(
    "H2",
    positions=[
        [0.000, 0.000, 0.000],
        [0.740, 0.000, 0.000]
    ]
)
```

This single object now contains the complete molecular structure.

Printing it gives

```python
print(atoms)
```

Output

```text
Atoms(symbols='H2', pbc=False)
```

---

# 22.13.13.4 Understanding the Constructor

Let us examine every argument.

```python
Atoms(
    symbols="H2",
    positions=[
        [0.000,0.000,0.000],
        [0.740,0.000,0.000]
    ]
)
```

### `symbols`

Specifies the chemical elements.

Examples

```python
"H2"

"Si2"

"NaCl"

"Fe2O3"

"C6H6"
```

ASE automatically determines the atomic numbers.

---

### `positions`

Stores Cartesian coordinates.

Each row corresponds to one atom.

```python
[
 [0.000,0.000,0.000],
 [0.740,0.000,0.000]
]
```

Units are **Ångström**.

---

# 22.13.13.5 Accessing Atomic Positions

Positions are stored in

```python
atoms.positions
```

Example

```python
print(atoms.positions)
```

Output

```text
[[0.00 0.00 0.00]
 [0.74 0.00 0.00]]
```

The returned object is a NumPy array.

Its shape is

```python
print(atoms.positions.shape)
```

Output

```text
(2,3)
```

Meaning

- 2 atoms
- 3 spatial coordinates

---

# 22.13.13.6 Accessing Individual Coordinates

The first atom

```python
print(atoms.positions[0])
```

Output

```text
[0.0 0.0 0.0]
```

Second atom

```python
print(atoms.positions[1])
```

Output

```text
[0.74 0.00 0.00]
```

Individual coordinates

```python
print(atoms.positions[1][0])
```

Output

```text
0.74
```

This is the x-coordinate of the second atom.

---

# 22.13.13.7 Atomic Numbers

Internally,

ASE stores elements using atomic numbers.

```python
print(atoms.numbers)
```

Output

```text
[1 1]
```

Hydrogen has atomic number 1.

Another example

```python
Atoms("NaCl")
```

would produce

```python
print(atoms.numbers)
```

```text
[11 17]
```

---

# 22.13.13.8 Chemical Symbols

To retrieve symbols,

```python
print(atoms.get_chemical_symbols())
```

Output

```text
['H', 'H']
```

Unlike

```python
atoms.numbers
```

this returns strings.

---

# 22.13.13.9 Number of Atoms

The total number of atoms is

```python
print(len(atoms))
```

or

```python
print(atoms.get_global_number_of_atoms())
```

Output

```text
2
```

This is one of the most frequently used methods during data preprocessing.

---

# 22.13.13.10 Simulation Cell

For crystals,

the simulation cell defines the lattice vectors.

Initially,

our hydrogen molecule has no cell.

```python
print(atoms.cell)
```

Output

```text
Cell([0.0, 0.0, 0.0])
```

For a silicon crystal,

we may define

```python
atoms.cell = [
    [5.43,0,0],
    [0,5.43,0],
    [0,0,5.43]
]
```

Now

```python
print(atoms.cell)
```

Output

```text
Cell([[5.43,0,0],
      [0,5.43,0],
      [0,0,5.43]])
```

---

# 22.13.13.11 Periodic Boundary Conditions

Crystals require periodic boundaries.

Check them using

```python
print(atoms.pbc)
```

Output

```text
[False False False]
```

Enable periodicity

```python
atoms.pbc = True
```

or

```python
atoms.pbc = [True,True,True]
```

Now

```python
print(atoms.pbc)
```

Output

```text
[ True  True  True ]
```

---

# 22.13.13.12 Fractional Coordinates

ASE stores Cartesian coordinates by default.

Fractional coordinates are obtained using

```python
atoms.get_scaled_positions()
```

Suppose

```python
atoms.cell = [
    [5,0,0],
    [0,5,0],
    [0,0,5]
]
```

An atom located at

```text
[2.5,2.5,2.5]
```

has fractional coordinates

```text
[0.5,0.5,0.5]
```

Fractional coordinates are especially useful for crystalline materials.

---

# 22.13.13.13 Atomic Masses

Masses can be obtained using

```python
print(atoms.get_masses())
```

Example

```text
[1.008 1.008]
```

ASE automatically assigns standard atomic masses.

---

# 22.13.13.14 Metadata

Additional information is stored inside

```python
atoms.info
```

For example,

```python
atoms.info["energy"] = -10.543
```

Later,

```python
print(atoms.info)
```

Output

```python
{'energy': -10.543}
```

This dictionary commonly stores

- total energy,
- temperature,
- pressure,
- identifiers,
- dataset labels.

---

# 22.13.13.15 Per-Atom Arrays

Properties that belong to individual atoms are stored inside

```python
atoms.arrays
```

Example

```python
print(atoms.arrays.keys())
```

Possible output

```text
dict_keys([
'positions',
'forces'
])
```

Each entry has one value per atom.

For example,

```python
atoms.arrays["forces"]
```

may return

```text
[[ 0.12 -0.21 0.05]
 [-0.12 0.21 -0.05]]
```

---

# 22.13.13.16 Why the `Atoms` Object Is Central to MACE

Every MACE workflow begins with an `Atoms` object.

The complete pipeline is

```text
Crystal Structure

        │

        ▼

ASE Atoms Object

        │

        ▼

Neighbor Graph Construction

        │

        ▼

MACE Neural Network

        │

        ▼

Energy Prediction

        │

        ▼

Automatic Differentiation

        │

        ▼

Atomic Forces
```

Notice that the neural network never receives a CIF file, an XYZ file, or raw coordinates directly.

Everything first becomes an `Atoms` object.

This object serves as the interface between

- atomistic simulation software,
- datasets,
- machine learning models,
- and molecular dynamics engines.

---

# 22.13.13.17 Summary

The ASE `Atoms` object is the fundamental data structure used throughout the MACE ecosystem. It stores atomic identities, Cartesian coordinates, lattice vectors, periodic boundary conditions, energies, forces, stress, and additional metadata within a single unified interface. Every practical workflow—including dataset preparation, neighbor graph construction, model training, energy prediction, and molecular dynamics—begins with an `Atoms` object. A thorough understanding of this class is therefore essential before moving to the implementation of MACE training.

---

## Transition to Section 22.13.14

Now that we understand how atomic structures are represented in memory, the next step is learning how to read real datasets from disk. In the following section, we will work with **Extended XYZ (`.extxyz`)** files, the most widely used dataset format for training MACE models, and learn how to inspect structures, energies, forces, and stress tensors programmatically.

# 22.13.14 Loading an Extended XYZ (`.extxyz`) Dataset

Now that we understand the ASE `Atoms` object, we are ready to load real atomistic datasets.

Almost every modern machine learning interatomic potential—including MACE, NequIP, Allegro, CHGNet, and many others—uses datasets stored in the **Extended XYZ (`.extxyz`)** format.

Unlike an ordinary XYZ file, an Extended XYZ file stores not only atomic coordinates but also important physical quantities such as

- total energy,
- atomic forces,
- stress tensor,
- lattice vectors,
- periodic boundary conditions,
- atomic charges,
- magnetic moments,
- and any additional user-defined properties.

Because of its flexibility and simplicity, `.extxyz` has become one of the standard formats for training machine learning interatomic potentials.

In this section, we will learn how to

- load an Extended XYZ dataset,
- inspect its contents,
- understand its structure,
- access individual configurations,
- and prepare it for MACE training.

---

# 22.13.14.1 What Is an Extended XYZ File?

A normal XYZ file contains only

- number of atoms,
- comment line,
- atomic symbols,
- Cartesian coordinates.

For example,

```text
2
Hydrogen Molecule
H   0.000   0.000   0.000
H   0.740   0.000   0.000
```

This format is sufficient for visualization.

However,

it cannot be used directly to train a machine learning interatomic potential because it contains no information about

- energy,
- forces,
- stress,
- periodic cell.

The Extended XYZ format solves this limitation.

---

# 22.13.14.2 Structure of an Extended XYZ File

A simplified Extended XYZ file may look like

```text
2
Lattice="8.0 0 0 0 8.0 0 0 0 8.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-6.759 pbc="T T T"
H   0.000   0.000   0.000   0.102   -0.015   0.008
H   0.740   0.000   0.000  -0.102    0.015  -0.008
```

Compared with a normal XYZ file,

the second line now stores

- lattice vectors,
- energy,
- periodic boundary conditions,
- property definitions.

Each atom additionally stores

- Cartesian coordinates,
- force components.

---

# 22.13.14.3 Reading an Extended XYZ File

ASE makes reading datasets extremely simple.

Import the reader.

```python
from ase.io import read
```

Suppose the dataset is named

```text
train.extxyz
```

We can load the entire dataset using

```python
structures = read("train.extxyz", index=":")
```

The parameter

```python
index=":"
```

means

> Read every structure contained in the file.

---

# 22.13.14.4 Understanding the Return Value

After loading,

```python
print(type(structures))
```

returns

```text
<class 'list'>
```

Each element of this list is an ASE `Atoms` object.

Therefore,

```python
structures[0]
```

represents

the first structure.

Similarly,

```python
structures[1]
```

represents

the second structure.

Conceptually,

```text
train.extxyz

        │

        ▼

read()

        │

        ▼

List

 ├── Atoms 1

 ├── Atoms 2

 ├── Atoms 3

 ├── ...

 └── Atoms N
```

---

# 22.13.14.5 Determining Dataset Size

The number of structures is simply

```python
print(len(structures))
```

Example

```text
5423
```

This indicates that the dataset contains

5,423 atomic configurations.

---

# 22.13.14.6 Accessing Individual Structures

The first structure

```python
atoms = structures[0]
```

The tenth structure

```python
atoms = structures[9]
```

The final structure

```python
atoms = structures[-1]
```

This indexing behaves exactly like a normal Python list.

---

# 22.13.14.7 Printing a Structure

Printing an ASE object

```python
print(atoms)
```

might produce

```text
Atoms(symbols='Si64',
      pbc=True,
      cell=[10.86,10.86,10.86])
```

This immediately provides

- chemical composition,
- periodicity,
- simulation cell.

---

# 22.13.14.8 Viewing Atomic Positions

Atomic positions are obtained using

```python
print(atoms.positions)
```

Example

```text
[[0.000 0.000 0.000]

 [2.715 2.715 0.000]

 [5.430 0.000 2.715]

 ...
]
```

The returned object is a NumPy array.

Its shape is

```python
print(atoms.positions.shape)
```

Example

```text
(64,3)
```

meaning

- 64 atoms,
- 3 coordinates per atom.

---

# 22.13.14.9 Viewing Atomic Species

Chemical symbols

```python
print(atoms.get_chemical_symbols())
```

Output

```text
['Si','Si','Si',...]
```

Atomic numbers

```python
print(atoms.numbers)
```

Output

```text
[14 14 14 ...]
```

---

# 22.13.14.10 Viewing the Simulation Cell

The lattice vectors are stored in

```python
print(atoms.cell)
```

Example

```text
Cell([[10.86,0,0],

      [0,10.86,0],

      [0,0,10.86]])
```

This information is required for constructing periodic neighbor graphs.

---

# 22.13.14.11 Viewing Periodic Boundary Conditions

```python
print(atoms.pbc)
```

Output

```text
[ True True True ]
```

Periodic systems generally have

```python
True
```

along every direction.

---

# 22.13.14.12 Viewing Stored Metadata

Additional properties are stored inside

```python
print(atoms.info)
```

Example

```python
{
'energy': -327.561
}
```

Many datasets store

- energy,
- pressure,
- temperature,
- identifiers,

inside this dictionary.

---

# 22.13.14.13 Accessing the Total Energy

Most MACE datasets include the reference energy.

Depending on the dataset,

the energy may be retrieved using

```python
print(atoms.info["energy"])
```

Example

```text
-327.561
```

Some datasets instead attach a calculator, allowing

```python
atoms.get_potential_energy()
```

Both approaches are commonly encountered.

---

# 22.13.14.14 Viewing Atomic Forces

Forces are usually stored inside

```python
atoms.arrays["forces"]
```

Example

```python
print(atoms.arrays["forces"])
```

Output

```text
[[ 0.021 -0.017 0.003]

 [-0.021 0.017 -0.003]

 ...
]
```

Each row corresponds to one atom.

Each column corresponds to

- Fx,
- Fy,
- Fz.

---

# 22.13.14.15 Determining the Number of Atoms

The total number of atoms is

```python
print(len(atoms))
```

Example

```text
64
```

This value is frequently used during batching and graph construction.

---

# 22.13.14.16 Looping Through the Dataset

Processing every structure is straightforward.

```python
for atoms in structures:

    print(len(atoms))
```

This loop iterates through the entire dataset.

Replacing

```python
print(len(atoms))
```

with other operations allows us to preprocess every configuration automatically.

---

# 22.13.14.17 Computing Dataset Statistics

For example,

the average number of atoms can be computed.

```python
num_atoms = []

for atoms in structures:
    num_atoms.append(len(atoms))

average = sum(num_atoms) / len(num_atoms)

print(average)
```

Similar statistics may be computed for

- energies,
- forces,
- lattice constants,
- volumes.

These summaries help us understand the diversity of the dataset before training.

---

# 22.13.14.18 Why `.extxyz` Is Ideal for MACE

The Extended XYZ format combines nearly everything required for machine learning into a single file.

It stores

- geometry,
- periodicity,
- energy,
- forces,
- stress,
- arbitrary metadata.

Consequently,

ASE can immediately convert every configuration into an `Atoms` object,

which MACE can process directly.

No additional parsing is usually required.

---

# 22.13.14.19 Summary

The Extended XYZ format is one of the most widely used dataset formats for training machine learning interatomic potentials. Using ASE, an entire dataset can be loaded with a single command, producing a list of `Atoms` objects. Each object contains atomic coordinates, lattice vectors, periodic boundary conditions, energies, forces, and additional metadata, making `.extxyz` an ideal interface between quantum mechanical datasets and the MACE training pipeline.

---

## Transition to Section 22.13.15

After loading a dataset, the next step is to inspect its physical quantities in greater detail. Before training a neural network, it is good practice to verify that the energies, forces, stress tensors, chemical compositions, and simulation cells have been loaded correctly. In the next section, we will explore techniques for inspecting and validating these quantities before constructing a MACE model.

# 22.13.15 Exploring Energies, Forces, and Stress

Successfully loading an Extended XYZ dataset does **not** guarantee that the data are suitable for training.

Before training any machine learning interatomic potential, it is essential to inspect the dataset carefully. Many training failures originate from incorrect or incomplete datasets rather than problems with the neural network itself.

A researcher should always verify that

- every structure contains an energy,
- every atom has an associated force,
- stress tensors are available when required,
- atomic positions are reasonable,
- lattice vectors are valid,
- periodic boundary conditions are correctly defined.

This process is known as **dataset validation**.

---

# 22.13.15.1 Why Dataset Inspection Is Important

Suppose one configuration has

- missing forces,
- incorrect energy units,
- corrupted lattice vectors,
- or inconsistent periodic boundary conditions.

During training,

the optimizer will attempt to minimize a loss using incorrect labels.

As a consequence,

- training may diverge,
- validation errors may become extremely large,
- molecular dynamics may become unstable,
- the final model may produce physically meaningless predictions.

Therefore,

dataset inspection should always be performed before model training.

---

# 22.13.15.2 Selecting a Single Structure

Let us begin by selecting the first configuration.

```python
from ase.io import read

structures = read("train.extxyz", index=":")

atoms = structures[0]
```

The variable

```python
atoms
```

is now an ASE `Atoms` object containing a single atomic configuration.

---

# 22.13.15.3 Viewing Basic Information

The simplest inspection begins with

```python
print(atoms)
```

Example output

```text
Atoms(symbols='Si64',
      pbc=True,
      cell=[10.86,10.86,10.86])
```

From this summary we immediately know

- chemical composition,
- number of atoms,
- whether the system is periodic,
- simulation cell dimensions.

---

# 22.13.15.4 Viewing Chemical Composition

The chemical symbols are

```python
print(atoms.get_chemical_symbols())
```

Example

```text
['Si',
 'Si',
 'Si',
 ...
]
```

The corresponding atomic numbers are

```python
print(atoms.numbers)
```

Output

```text
[14 14 14 ...]
```

---

# 22.13.15.5 Counting Each Element

For multicomponent systems,

it is useful to count the number of atoms of each element.

```python
from collections import Counter

composition = Counter(atoms.get_chemical_symbols())

print(composition)
```

Example

```text
Counter({
'Li':16,
'P':4,
'S':16
})
```

This allows rapid verification of chemical composition.

---

# 22.13.15.6 Viewing Atomic Coordinates

```python
print(atoms.positions)
```

Example

```text
[[0.000 0.000 0.000]

 [2.715 2.715 0.000]

 ...

]
```

The coordinate array has shape

```python
print(atoms.positions.shape)
```

Output

```text
(64,3)
```

meaning

- 64 atoms
- 3 Cartesian coordinates

---

# 22.13.15.7 Viewing the Simulation Cell

The lattice vectors define the periodic simulation box.

```python
print(atoms.cell)
```

Example

```text
Cell([[10.86,0.00,0.00],

      [0.00,10.86,0.00],

      [0.00,0.00,10.86]])
```

The cell volume may be computed using

```python
print(atoms.get_volume())
```

Example

```text
1281.50
```

Units are

Å³.

---

# 22.13.15.8 Checking Periodic Boundary Conditions

```python
print(atoms.pbc)
```

Output

```text
[ True  True  True ]
```

For isolated molecules,

the output would instead be

```text
[False False False]
```

---

# 22.13.15.9 Inspecting Total Energy

Many datasets store energies inside

```python
atoms.info
```

Check the available metadata.

```python
print(atoms.info.keys())
```

Example

```text
dict_keys([
'energy'
])
```

Retrieve the energy.

```python
print(atoms.info["energy"])
```

Example

```text
-327.561
```

Some datasets instead use

```python
atoms.get_potential_energy()
```

Both approaches are common.

---

# 22.13.15.10 Inspecting Atomic Forces

Atomic forces are usually stored inside

```python
atoms.arrays["forces"]
```

Display the forces.

```python
print(atoms.arrays["forces"])
```

Example

```text
[[ 0.013 -0.025  0.007]

 [-0.013  0.025 -0.007]

 ...
]
```

The shape is

```python
print(atoms.arrays["forces"].shape)
```

Output

```text
(64,3)
```

One force vector exists for every atom.

---

# 22.13.15.11 Computing Force Magnitudes

The magnitude of every force can be calculated.

```python
import numpy as np

forces = atoms.arrays["forces"]

magnitudes = np.linalg.norm(forces, axis=1)

print(magnitudes)
```

Example

```text
[0.029

 0.029

 0.018

 ...
]
```

Large force magnitudes often indicate

- unstable structures,
- unrelaxed geometries,
- transition states.

---

# 22.13.15.12 Maximum Force

A useful quantity is the largest atomic force.

```python
print(np.max(magnitudes))
```

Example

```text
0.462
```

Many geometry optimizations stop once

```text
Maximum Force < 0.01 eV/Å
```

---

# 22.13.15.13 Inspecting the Stress Tensor

Some datasets contain stress information.

Check whether stress exists.

```python
print(atoms.info.keys())
```

Possible output

```text
dict_keys([
'energy',
'stress'
])
```

Retrieve it.

```python
print(atoms.info["stress"])
```

Example

```text
[[ 1.21  0.00  0.01]

 [ 0.00  1.18 -0.02]

 [ 0.01 -0.02  1.19]]
```

Stress supervision improves lattice optimization during training.

---

# 22.13.15.14 Checking Available Arrays

ASE stores atom-wise properties inside

```python
print(atoms.arrays.keys())
```

Possible output

```text
dict_keys([
'numbers',
'positions',
'forces'
])
```

Other datasets may additionally contain

- charges,
- magnetic moments,
- velocities,
- spins.

---

# 22.13.15.15 Inspecting Every Structure

Rather than checking only one configuration,

we should inspect the entire dataset.

```python
for atoms in structures:

    print(len(atoms),
          atoms.info["energy"])
```

Example

```text
64  -327.56

64  -327.81

64  -327.72

...
```

This confirms that every configuration has

- the correct number of atoms,
- a valid energy.

---

# 22.13.15.16 Checking for Missing Energies

A useful validation step is

```python
missing = 0

for atoms in structures:

    if "energy" not in atoms.info:
        missing += 1

print("Missing Energies:", missing)
```

The desired output is

```text
Missing Energies: 0
```

---

# 22.13.15.17 Checking for Missing Forces

Similarly,

```python
missing = 0

for atoms in structures:

    if "forces" not in atoms.arrays:
        missing += 1

print("Missing Forces:", missing)
```

Output

```text
Missing Forces: 0
```

If missing values exist,

those configurations should generally be removed before training.

---

# 22.13.15.18 Summary Statistics

It is often useful to compute dataset-wide statistics.

```python
energies = []

for atoms in structures:

    energies.append(atoms.info["energy"])
```

Average energy

```python
import numpy as np

print(np.mean(energies))
```

Minimum energy

```python
print(np.min(energies))
```

Maximum energy

```python
print(np.max(energies))
```

Such statistics help identify

- outliers,
- corrupted structures,
- inconsistent calculations.

---

# 22.13.15.19 Visualizing Energy Distribution

A histogram provides a quick overview.

```python
import matplotlib.pyplot as plt

plt.hist(energies, bins=40)

plt.xlabel("Energy (eV)")

plt.ylabel("Count")

plt.title("Energy Distribution")

plt.show()
```

A smooth distribution generally indicates a well-sampled dataset.

Unexpected spikes or isolated points may indicate problematic structures.

---

# 22.13.15.20 Best Practices Before Training

Before starting MACE training,

always verify

- ✔ every structure loads correctly,
- ✔ energies exist,
- ✔ forces exist,
- ✔ lattice vectors are valid,
- ✔ periodic boundary conditions are correct,
- ✔ chemical compositions are reasonable,
- ✔ no corrupted configurations are present.

These simple checks prevent many common training failures.

---

# 22.13.15.21 Summary

Dataset validation is a crucial step in every machine learning workflow. Before training a MACE model, researchers should carefully inspect energies, atomic forces, stress tensors, lattice vectors, periodic boundary conditions, and chemical compositions. Performing these checks ensures that the neural network receives consistent and physically meaningful supervision, greatly improving training stability and final model accuracy.

---

## Transition to Section 22.13.16

Once the dataset has been thoroughly inspected and validated, it is ready for preprocessing. The next step is to divide the data into **training**, **validation**, and **test** subsets, ensuring that the neural network is trained fairly and its predictive performance can be evaluated on previously unseen structures.

# 22.13.16 Preparing the Dataset for Training

Once the dataset has been inspected and validated, it must be prepared for machine learning.

Raw DFT calculations are **not** immediately suitable for training a MACE model. Before optimization begins, the data should be organized into appropriate subsets, checked for consistency, and preprocessed to ensure efficient learning.

Dataset preparation is one of the most important stages of any machine learning project. A well-designed neural network cannot compensate for poorly prepared data.

In this section, we will learn how to

- split the dataset into training, validation, and test sets,
- shuffle the configurations,
- inspect the resulting subsets,
- save the processed datasets,
- and understand why these steps are necessary.

---

# 22.13.16.1 Why Dataset Splitting Is Necessary

Suppose we train a neural network using every available structure.

After training, we evaluate the model using the **same dataset**.

Even if the reported error is extremely small, this does **not** indicate that the model has learned the underlying physics.

Instead, it may simply have memorized the training data.

To evaluate the true predictive capability of the model, some structures must be withheld during training.

Therefore, the dataset is divided into three independent subsets.

```text
Entire Dataset

        │

        ▼

 ┌──────────────┬──────────────┬──────────────┐
 │              │              │
 ▼              ▼              ▼

Training     Validation      Test

Learning     Hyperparameter  Final Evaluation
             Tuning
```

Each subset serves a different purpose.

---

# 22.13.16.2 Training Set

The **training set** is used to optimize the neural network parameters.

During each iteration,

the optimizer computes

- predicted energies,
- predicted forces,
- predicted stress,

compares them with the DFT values,

and updates the network weights.

The model only "learns" from the training set.

---

# 22.13.16.3 Validation Set

The validation set is **never used for parameter updates**.

Instead, it is used to

- monitor overfitting,
- tune hyperparameters,
- select the best checkpoint,
- determine when training should stop.

A good validation error indicates that the model generalizes beyond the training configurations.

---

# 22.13.16.4 Test Set

The test set should remain completely untouched until the model has finished training.

Its purpose is to provide an unbiased estimate of model performance.

Once evaluation has been completed,

the test structures should not be reused during further model development.

---

# 22.13.16.5 Typical Dataset Ratios

A common split is

| Dataset | Percentage |
|----------|-----------:|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

Other ratios are also common.

For very large datasets,

95–3–2 is often sufficient.

For smaller datasets,

70–15–15 may provide a better estimate of generalization.

---

# 22.13.16.6 Loading the Dataset

Begin by reading the dataset.

```python
from ase.io import read

structures = read("train.extxyz", index=":")
```

Determine the total number of structures.

```python
print(len(structures))
```

Example

```text
12500
```

---

# 22.13.16.7 Shuffling the Dataset

The original ordering of the structures may contain hidden correlations.

For example,

- all low-temperature structures,
- followed by high-temperature structures,
- followed by liquid structures.

Training on this ordered sequence can bias the optimization.

Therefore,

shuffle the dataset before splitting.

```python
import random

random.shuffle(structures)
```

Now the configurations are randomly ordered.

---

# 22.13.16.8 Why Random Shuffling Matters

Consider the following ordered dataset.

```text
Structure 1

Structure 2

Structure 3

...

Structure 5000
```

If all low-energy structures appear first,

the optimizer initially sees only one region of configuration space.

Random shuffling distributes

- crystal phases,
- temperatures,
- defect structures,
- chemical compositions,

throughout the dataset,

leading to more stable optimization.

---

# 22.13.16.9 Computing Dataset Sizes

Determine the number of structures assigned to each subset.

```python
n_total = len(structures)

n_train = int(0.80 * n_total)

n_valid = int(0.10 * n_total)

n_test = n_total - n_train - n_valid
```

Print the result.

```python
print(n_train)
print(n_valid)
print(n_test)
```

Example

```text
10000

1250

1250
```

---

# 22.13.16.10 Splitting the Dataset

Slice the shuffled list.

```python
train_structures = structures[:n_train]

valid_structures = structures[n_train:n_train+n_valid]

test_structures = structures[n_train+n_valid:]
```

Now we have three independent datasets.

---

# 22.13.16.11 Verifying the Split

Check the number of structures.

```python
print(len(train_structures))

print(len(valid_structures))

print(len(test_structures))
```

Example

```text
10000

1250

1250
```

Always verify that

```python
len(train_structures) +
len(valid_structures) +
len(test_structures)
```

equals the original dataset size.

---

# 22.13.16.12 Saving the New Datasets

ASE makes writing datasets straightforward.

```python
from ase.io import write

write(
    "train.extxyz",
    train_structures
)

write(
    "valid.extxyz",
    valid_structures
)

write(
    "test.extxyz",
    test_structures
)
```

Each subset is now stored independently.

---

# 22.13.16.13 Inspecting the Training Set

Read the new training dataset.

```python
train = read(
    "train.extxyz",
    index=":"
)

print(len(train))
```

Example

```text
10000
```

This confirms that the file has been written correctly.

---

# 22.13.16.14 Checking Energy Distributions

Ideally,

the energy distributions of

- training,
- validation,
- test

should be similar.

Extract training energies.

```python
train_energy = [
    atoms.info["energy"]
    for atoms in train_structures
]
```

Similarly,

```python
valid_energy = [
    atoms.info["energy"]
    for atoms in valid_structures
]

test_energy = [
    atoms.info["energy"]
    for atoms in test_structures
]
```

---

# 22.13.16.15 Visualizing the Splits

Plot the distributions.

```python
import matplotlib.pyplot as plt

plt.hist(
    train_energy,
    bins=50,
    alpha=0.5,
    label="Train"
)

plt.hist(
    valid_energy,
    bins=50,
    alpha=0.5,
    label="Validation"
)

plt.hist(
    test_energy,
    bins=50,
    alpha=0.5,
    label="Test"
)

plt.xlabel("Energy (eV)")

plt.ylabel("Count")

plt.legend()

plt.show()
```

The three histograms should overlap reasonably well.

Large discrepancies indicate an unbalanced split.

---

# 22.13.16.16 Reproducibility

Random shuffling should be reproducible.

Set the random seed.

```python
import random

random.seed(42)

random.shuffle(structures)
```

Using the same seed ensures that future runs generate identical dataset splits.

This is essential for reproducible scientific research.

---

# 22.13.16.17 When Random Splitting Is Not Enough

Simple random splitting is often sufficient.

However,

for certain datasets,

better strategies exist.

Examples include

- composition-based splitting,
- time-based splitting,
- molecule-based splitting,
- crystal prototype splitting,
- active-learning splits.

These methods reduce data leakage between subsets and provide a more realistic estimate of model performance.

---

# 22.13.16.18 Data Leakage

One of the most common mistakes is **data leakage**.

For example,

suppose a crystal appears in both

- training
- and test

with only a tiny atomic displacement.

Although technically different configurations,

they are nearly identical.

The reported test error will therefore appear unrealistically low.

Proper dataset splitting avoids this problem.

---

# 22.13.16.19 Best Practices

Before beginning training,

always verify that

- ✔ every subset loads correctly,
- ✔ no structures are duplicated,
- ✔ the distributions are similar,
- ✔ the random seed is recorded,
- ✔ dataset statistics are documented.

These simple practices greatly improve reproducibility.

---

# 22.13.16.20 Summary

Preparing the dataset is one of the most important stages of machine learning. After validating the atomic configurations, the dataset should be randomly shuffled, divided into independent training, validation, and test subsets, and written to separate files. Proper dataset preparation prevents overfitting, enables reliable model evaluation, and provides a reproducible foundation for training a MACE interatomic potential.

---

## Transition to Section 22.13.17

With the datasets prepared, we are finally ready to construct a MACE neural network. In the next section, we will examine the architecture of the official MACE implementation, understand its configuration parameters, and initialize our first trainable MACE model before beginning optimization.

# 22.13.17 Building a MACE Model

With the dataset prepared, we are now ready to construct our first **MACE neural network**.

Unlike conventional neural networks, where the architecture usually consists of fully connected layers or convolutional layers, a MACE model is composed of multiple interacting modules specifically designed for atomistic systems.

These modules include

- neighbor graph construction,
- equivariant feature embeddings,
- interaction blocks,
- higher-order correlation blocks,
- readout layers,
- energy aggregation.

Fortunately, the official **mace-torch** implementation provides a high-level interface that automatically assembles these components.

Before training, however, we must understand the purpose of the most important hyperparameters and how they influence the architecture.

---

# 22.13.17.1 Model Construction Workflow

Building a MACE model follows a systematic pipeline.

```text
Training Dataset

        │

        ▼

Atomic Species

        │

        ▼

Neighbor Graph

        │

        ▼

Embedding Layer

        │

        ▼

Interaction Blocks

        │

        ▼

Higher-Order Correlations

        │

        ▼

Readout Layer

        │

        ▼

Atomic Energies

        │

        ▼

Total Energy
```

Each component corresponds to one stage of the forward pass described earlier in this chapter.

---

# 22.13.17.2 Why We Must Configure the Model

Every materials dataset is different.

For example,

a model trained for

- silicon,
- metallic alloys,
- battery materials,
- organic molecules,

may require different network capacities.

Therefore, before training, we specify several architectural parameters that determine

- the expressive power,
- computational cost,
- memory usage,
- prediction accuracy.

These are known as **hyperparameters** because they are selected before optimization begins.

---

# 22.13.17.3 Important Hyperparameters

Some of the most important MACE hyperparameters are

| Hyperparameter | Purpose |
|----------------|---------|
| `r_max` | Neighbor cutoff radius |
| `num_interactions` | Number of interaction blocks |
| `max_L` | Maximum angular momentum |
| `hidden_irreps` | Hidden representation size |
| `correlation` | Correlation order |
| `num_radial_basis` | Number of radial basis functions |

Each parameter controls a different aspect of the network architecture.

---

# 22.13.17.4 Cutoff Radius (`r_max`)

The cutoff radius determines which neighboring atoms interact.

Mathematically,

neighbors satisfy

$$
r_{ij} < r_{\max}
$$

where

- $r_{ij}$ is the distance between atoms,
- $r_{\max}$ is the cutoff radius.

Typical values are

| Material | Typical Cutoff |
|-----------|---------------:|
| Molecules | 4–5 Å |
| Covalent Crystals | 5–6 Å |
| Metals | 5–7 Å |
| Complex Oxides | 6–8 Å |

A larger cutoff captures more physical interactions but increases computational cost.

---

# 22.13.17.5 Number of Interaction Blocks

Interaction blocks determine how many rounds of message passing occur.

```text
Interaction Block 1

↓

Interaction Block 2

↓

Interaction Block 3

↓

Interaction Block 4
```

Increasing the number of interaction blocks enables information to propagate over longer distances.

Typical values are

```text
2

3

4
```

More blocks generally improve accuracy but increase training time.

---

# 22.13.17.6 Hidden Irreducible Representations

Unlike ordinary neural networks,

MACE represents features using irreducible representations of the rotation group.

For example,

```text
128x0e + 128x1o
```

means

- 128 scalar features,
- 128 vector features.

Larger hidden representations increase model capacity.

Common choices include

```text
64x0e + 64x1o

128x0e +128x1o

256x0e +256x1o
```

---

# 22.13.17.7 Maximum Angular Momentum

The parameter

```text
max_L
```

determines the highest spherical harmonic degree used.

Typical values are

| max_L | Meaning |
|-------:|---------|
| 1 | Low angular resolution |
| 2 | Moderate resolution |
| 3 | High angular resolution |

Larger values allow the model to capture more complex angular environments.

However,

they also increase computational cost considerably.

---

# 22.13.17.8 Correlation Order

One of the defining characteristics of MACE is its higher-order body correlations.

Typical values include

```text
Correlation = 2

Correlation = 3

Correlation = 4
```

Increasing the correlation order allows the network to capture

- three-body,
- four-body,
- five-body,

interactions implicitly.

Higher correlation generally improves accuracy but increases memory consumption.

---

# 22.13.17.9 Radial Basis Functions

Distances are expanded using radial basis functions.

Instead of representing

```text
r = 2.43 Å
```

as a single number,

the network converts it into a vector of basis function values.

The number of basis functions is typically

```text
8

10

12

16
```

More basis functions provide finer distance resolution.

---

# 22.13.17.10 Typical Model Configuration

A representative medium-sized MACE model may use

| Parameter | Value |
|-----------|-------|
| Cutoff radius | 5.0 Å |
| Interaction blocks | 2 |
| Correlation order | 3 |
| Maximum angular momentum | 2 |
| Hidden irreps | 128x0e + 128x1o |
| Radial basis functions | 8 |

This configuration provides a good balance between accuracy and computational efficiency for many materials systems.

---

# 22.13.17.11 Constructing a Model

The official package provides high-level interfaces for model creation.

A simplified example is

```python
from mace import models

model = models.MACE(
    r_max=5.0,
    num_interactions=2,
    max_L=2,
    correlation=3
)
```

This initializes a new neural network with randomly initialized parameters.

> **Note:** The exact constructor arguments may vary between MACE releases. Always consult the official documentation corresponding to the installed version.

---

# 22.13.17.12 Viewing the Model

Printing the model

```python
print(model)
```

produces a summary similar to

```text
MACE(

  Embedding

  Interaction Block ×2

  Readout

)
```

For larger models,

many additional layers and equivariant operations will also be displayed.

Inspecting the model summary is useful for verifying that the intended architecture has been created.

---

# 22.13.17.13 Counting Trainable Parameters

A useful diagnostic is the number of trainable parameters.

```python
total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(total_parameters)
```

Example output

```text
1,824,537
```

Larger models generally contain more trainable parameters and require

- more memory,
- more training data,
- longer optimization.

---

# 22.13.17.14 Moving the Model to the GPU

If a CUDA-enabled GPU is available,

move the model to GPU memory.

```python
import torch

device = torch.device("cuda")

model = model.to(device)
```

To verify,

```python
print(next(model.parameters()).device)
```

Example output

```text
cuda:0
```

Training on the GPU is typically an order of magnitude faster than on the CPU.

---

# 22.13.17.15 Choosing an Appropriate Model Size

There is no universally optimal architecture.

The choice depends on

- dataset size,
- chemical complexity,
- available GPU memory,
- desired accuracy.

General recommendations are

| Dataset Size | Recommended Model |
|--------------|------------------|
| <10,000 structures | Small |
| 10,000–100,000 structures | Medium |
| >100,000 structures | Large |

Using an excessively large model on a small dataset often leads to overfitting.

---

# 22.13.17.16 Common Mistakes

When constructing a MACE model, beginners often make several mistakes.

Examples include

- choosing an excessively large cutoff radius,
- using a very high correlation order without sufficient GPU memory,
- selecting an unnecessarily large hidden representation,
- ignoring the relationship between dataset size and model complexity.

These choices can significantly increase training time without improving predictive accuracy.

---

# 22.13.17.17 Summary

Constructing a MACE model involves selecting a set of architectural hyperparameters that define the network's capacity and computational requirements. Important choices include the cutoff radius, number of interaction blocks, hidden irreducible representations, maximum angular momentum, correlation order, and radial basis functions. These hyperparameters determine how effectively the network can learn the underlying quantum mechanical potential energy surface while balancing accuracy, memory usage, and computational cost.

---

## Transition to Section 22.13.18

With the neural network architecture fully defined, the next step is to train the model. We will prepare the optimizer, define the learning objective, launch the training process, and examine how MACE learns to predict energies and forces from quantum mechanical reference data.

# 22.13.18 Training a MACE Model

With the dataset prepared and the neural network architecture constructed, we are now ready to train our first MACE model.

Training is the process through which the neural network gradually learns the relationship between atomic configurations and their corresponding quantum mechanical properties.

Unlike traditional regression models, MACE does not learn from handcrafted descriptors. Instead, it learns directly from atomic structures by repeatedly comparing its predictions with Density Functional Theory (DFT) reference data and updating its internal parameters through gradient-based optimization.

The objective of training is to determine a set of neural network parameters that minimizes the discrepancy between

- predicted energies,
- predicted atomic forces,
- predicted stress tensors,

and their corresponding DFT values.

---

# 22.13.18.1 Overview of the Training Pipeline

The complete training workflow is illustrated below.

```text
Training Dataset

        │

        ▼

Neighbor Graph Construction

        │

        ▼

Forward Pass

        │

        ▼

Predicted Energy

Predicted Forces

Predicted Stress

        │

        ▼

Loss Function

        │

        ▼

Backpropagation

        │

        ▼

Gradient Computation

        │

        ▼

Optimizer Update

        │

        ▼

Updated Neural Network
```

This cycle is repeated thousands of times until the model converges.

---

# 22.13.18.2 The Official MACE Training Script

The easiest way to train a MACE model is to use the official training script provided by the `mace-torch` package.

The basic command is

```bash
mace_run_train \
    --train_file train.extxyz \
    --valid_file valid.extxyz \
    --model MACE
```

This command launches the complete training pipeline.

However, a practical research model requires many additional parameters.

---

# 22.13.18.3 A Complete Training Command

A typical training command is shown below.

```bash
mace_run_train \
    --name silicon_model \
    --train_file train.extxyz \
    --valid_file valid.extxyz \
    --model MACE \
    --hidden_irreps "128x0e + 128x1o" \
    --num_interactions 2 \
    --correlation 3 \
    --r_max 5.0 \
    --batch_size 8 \
    --max_num_epochs 200 \
    --energy_weight 1.0 \
    --forces_weight 100.0 \
    --stress_weight 1.0 \
    --lr 0.001 \
    --device cuda
```

This single command is sufficient to train a complete MACE model.

The remainder of this section explains each parameter in detail.

---

# 22.13.18.4 Naming the Model

```bash
--name silicon_model
```

This specifies the experiment name.

All generated files

- checkpoints,
- logs,
- trained models,

will use this name.

For example,

```text
silicon_model.model

silicon_model.log

silicon_model_best.model
```

Using descriptive names helps organize multiple experiments.

---

# 22.13.18.5 Training Dataset

```bash
--train_file train.extxyz
```

This specifies the dataset used for optimization.

Every atomic configuration in this file contributes to updating the neural network parameters.

---

# 22.13.18.6 Validation Dataset

```bash
--valid_file valid.extxyz
```

The validation dataset is **never** used for gradient updates.

Instead, it is used to

- monitor generalization,
- select the best checkpoint,
- determine early stopping.

---

# 22.13.18.7 Model Type

```bash
--model MACE
```

The official implementation supports multiple architectures.

Here we explicitly request the MACE architecture.

---

# 22.13.18.8 Hidden Representations

```bash
--hidden_irreps "128x0e + 128x1o"
```

This determines the size of the hidden feature space.

Larger representations generally improve accuracy but require

- more GPU memory,
- longer training,
- larger datasets.

---

# 22.13.18.9 Number of Interaction Blocks

```bash
--num_interactions 2
```

This specifies how many interaction blocks perform message passing.

Increasing this value allows information to propagate across larger neighborhoods.

Typical values are

- 2
- 3
- 4

---

# 22.13.18.10 Correlation Order

```bash
--correlation 3
```

The correlation order determines the highest body-order interaction represented by the model.

Higher values increase expressive power but also increase computational cost.

---

# 22.13.18.11 Cutoff Radius

```bash
--r_max 5.0
```

Neighbor interactions are computed only within this radius.

Mathematically,

$$
r_{ij} < r_{\max}
$$

Typical values range between

- 4 Å
- 8 Å

depending on the material system.

---

# 22.13.18.12 Batch Size

```bash
--batch_size 8
```

Instead of processing the entire dataset simultaneously,

training proceeds in small batches.

For example,

```text
Dataset

↓

8 Structures

↓

Forward Pass

↓

Parameter Update
```

Small batch sizes

- reduce GPU memory usage,
- produce noisier gradients,

while large batches require more memory but yield smoother optimization.

---

# 22.13.18.13 Number of Epochs

```bash
--max_num_epochs 200
```

One epoch corresponds to processing the entire training dataset once.

If the dataset contains

10,000 structures,

then

```text
200 epochs
```

means the network will process

2,000,000 structures

(counting repeated passes).

---

# 22.13.18.14 Loss Weights

MACE simultaneously learns

- energies,
- forces,
- stresses.

The relative importance of each target is controlled using

```bash
--energy_weight

--forces_weight

--stress_weight
```

For example,

```bash
--energy_weight 1.0

--forces_weight 100.0

--stress_weight 1.0
```

assigns much greater importance to force prediction.

This is common because every structure contains

- one energy,

but

- three force components for every atom.

---

# 22.13.18.15 Learning Rate

```bash
--lr 0.001
```

The learning rate controls the magnitude of each optimization step.

Large values may cause

- unstable training,
- divergence.

Very small values may produce

- slow convergence.

Typical values are

```text
0.001

0.0005

0.0001
```

---

# 22.13.18.16 Selecting the Device

Training on the GPU is specified by

```bash
--device cuda
```

CPU training uses

```bash
--device cpu
```

GPU training is strongly recommended for realistic datasets.

---

# 22.13.18.17 Starting Training

After launching the command,

training begins automatically.

Typical console output appears as

```text
Epoch 1

Training Loss

Validation Loss

Learning Rate

Time
```

As training proceeds,

these values are updated after every epoch.

---

# 22.13.18.18 What Happens During Each Epoch?

Every epoch consists of four major stages.

```text
Training Data

↓

Mini-Batches

↓

Forward Pass

↓

Loss Computation

↓

Backpropagation

↓

Optimizer Update
```

This sequence repeats until every structure has been processed.

---

# 22.13.18.19 Saving Checkpoints

During training,

MACE periodically saves checkpoints.

These files contain

- neural network parameters,
- optimizer state,
- training progress.

If training is interrupted,

it can be resumed from the most recent checkpoint rather than restarting from the beginning.

---

# 22.13.18.20 Best Model Selection

The final epoch is **not always** the best model.

Instead,

MACE usually selects the checkpoint with the lowest validation error.

This checkpoint is typically saved separately.

For example,

```text
silicon_model_best.model
```

Using the best validation checkpoint generally yields better predictive performance on unseen structures.

---

# 22.13.18.21 Monitoring GPU Usage

While training,

GPU utilization can be monitored using

```bash
nvidia-smi
```

Typical output includes

- GPU utilization,
- memory usage,
- running processes.

Monitoring GPU memory helps determine whether

- larger batch sizes,
- larger models,

can be accommodated.

---

# 22.13.18.22 Common Training Problems

Several issues commonly arise during training.

Examples include

- GPU out-of-memory errors,
- exploding loss,
- validation loss increasing,
- NaN gradients,
- extremely slow convergence.

These issues often originate from

- inappropriate learning rates,
- unsuitable batch sizes,
- corrupted datasets,
- excessively large model architectures.

---

# 22.13.18.23 Summary

Training a MACE model consists of repeatedly presenting batches of atomic structures to the neural network, computing predicted energies, forces, and stresses, evaluating the loss against DFT reference values, and updating the model parameters using gradient-based optimization. The official `mace_run_train` script automates this process while allowing extensive control through hyperparameters such as cutoff radius, interaction blocks, hidden representations, learning rate, batch size, and loss weights.

---

## Transition to Section 22.13.19

After the training process begins, the neural network continuously reports information about its progress. Understanding these logs is essential for determining whether the model is learning successfully, converging properly, or beginning to overfit. In the next section, we will examine the training log in detail and learn how to interpret every reported quantity.

# 22.13.19 Understanding the Training Log

After launching the training process, MACE continuously prints information describing the current state of optimization. These messages are collectively known as the **training log**.

For many beginners, the training log appears to be a collection of random numbers. However, for an experienced researcher, it provides a detailed picture of how well the neural network is learning.

A careful examination of the training log allows us to determine

- whether the model is converging,
- whether the learning rate is appropriate,
- whether overfitting has begun,
- whether the optimizer is stable,
- whether additional training is necessary.

Learning to interpret the training log is therefore an essential skill for developing reliable machine learning interatomic potentials.

---

# 22.13.19.1 Example Training Log

A typical training log produced by MACE may resemble the following.

```text
Epoch 1/200

Train Loss:          5.273

Validation Loss:     5.441

Energy RMSE:         185.2 meV

Force RMSE:          0.421 eV/Å

Stress RMSE:         0.185 GPa

Learning Rate:       0.001000

Time:                38.4 s
```

After every epoch, these values are updated.

---

# 22.13.19.2 Epoch Number

The first quantity is

```text
Epoch 1/200
```

This indicates

- the current training iteration,
- the total number of epochs.

For example,

```text
Epoch 125/200
```

means

- 125 complete passes through the dataset have been completed,
- 75 epochs remain.

---

# 22.13.19.3 Training Loss

Example

```text
Train Loss: 5.273
```

The training loss measures how well the neural network fits the **training dataset**.

It is computed from

- energy error,
- force error,
- stress error,

according to the weighted loss function discussed previously.

Generally,

the training loss should decrease steadily during optimization.

A typical trend is

```text
Epoch 1      5.27

Epoch 20     1.81

Epoch 50     0.94

Epoch 100    0.41

Epoch 200    0.18
```

A decreasing training loss indicates that the optimizer is successfully improving the model.

---

# 22.13.19.4 Validation Loss

Example

```text
Validation Loss: 5.441
```

Unlike the training loss,

the validation loss is computed using structures that the optimizer has **never used for parameter updates**.

This quantity estimates how well the model generalizes to unseen data.

Ideally,

both losses decrease together.

For example,

```text
Epoch

Train

Validation

1

5.27

5.44

20

1.81

1.96

50

0.94

1.02

100

0.41

0.46
```

This behavior indicates healthy learning.

---

# 22.13.19.5 Energy RMSE

One of the most important reported quantities is

```text
Energy RMSE
```

Example

```text
Energy RMSE: 18.4 meV/atom
```

RMSE stands for

**Root Mean Square Error**.

For energies,

it measures

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(E_i^{\mathrm{pred}}-E_i^{\mathrm{DFT}})^2
}
$$

where

- $E_i^{\mathrm{pred}}$ is the predicted energy,
- $E_i^{\mathrm{DFT}}$ is the reference DFT energy.

Smaller values indicate better agreement with DFT.

---

# 22.13.19.6 Interpreting Energy RMSE

Typical energy errors are

| Energy RMSE | Interpretation |
|-------------|---------------|
| >100 meV/atom | Poor |
| 50–100 meV/atom | Moderate |
| 10–50 meV/atom | Good |
| <10 meV/atom | Excellent |

The acceptable value depends on the intended application.

For high-accuracy atomistic simulations,

errors below

```text
10 meV/atom
```

are often desirable.

---

# 22.13.19.7 Force RMSE

The force error is equally important.

Example

```text
Force RMSE: 0.041 eV/Å
```

It is computed as

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{3N}
\sum_{i=1}^{N}
\|
\mathbf{F}_i^{\mathrm{pred}}
-
\mathbf{F}_i^{\mathrm{DFT}}
\|^2
}
$$

Because forces determine

- molecular dynamics,
- geometry optimization,
- phonons,

this quantity often receives the greatest emphasis during training.

---

# 22.13.19.8 Typical Force Errors

Approximate guidelines are

| Force RMSE | Quality |
|-------------|---------|
| >0.20 eV/Å | Poor |
| 0.10 eV/Å | Moderate |
| 0.05 eV/Å | Good |
| <0.02 eV/Å | Excellent |

Most modern MACE models aim for force errors below

```text
0.05 eV/Å
```

---

# 22.13.19.9 Stress RMSE

If stress supervision is enabled,

the training log also reports

```text
Stress RMSE
```

Example

```text
Stress RMSE: 0.083 GPa
```

Stress prediction is especially important when

- optimizing lattice parameters,
- studying pressure effects,
- performing NPT molecular dynamics.

---

# 22.13.19.10 Learning Rate

Example

```text
Learning Rate: 0.001000
```

This indicates the optimizer's current step size.

Many training schedules gradually decrease the learning rate.

Example

```text
Epoch

Learning Rate

1

0.001

50

0.0005

100

0.0001

150

0.00005
```

Reducing the learning rate near convergence helps produce a more stable solution.

---

# 22.13.19.11 Epoch Time

Example

```text
Time: 38.4 s
```

This represents the time required to complete one epoch.

The total training time is approximately

$$
T_{\mathrm{total}}
=
N_{\mathrm{epochs}}
\times
T_{\mathrm{epoch}}
$$

For example,

```text
200 epochs

×

38 seconds

≈

2.1 hours
```

---

# 22.13.19.12 Healthy Training Behavior

A successful training run typically exhibits

```text
Training Loss

↓

↓

↓

Validation Loss

↓

↓

↓

Energy RMSE

↓

↓

↓

Force RMSE

↓

↓

↓

Stress RMSE

↓

↓

↓

Learning Rate

↓

↓

↓

Stable Convergence
```

All error metrics gradually decrease before reaching a plateau.

---

# 22.13.19.13 Signs of Overfitting

One of the easiest ways to identify overfitting is by comparing

- training loss,
- validation loss.

Example

```text
Epoch

Training

Validation

40

0.82

0.91

60

0.51

0.73

80

0.32

0.96

100

0.19

1.42
```

Notice that

- training loss continues decreasing,
- validation loss begins increasing.

This indicates that the model is memorizing the training data instead of learning general physical relationships.

---

# 22.13.19.14 Signs of Underfitting

Underfitting appears differently.

Example

```text
Epoch

Training

Validation

1

5.1

5.3

20

4.9

5.0

50

4.7

4.9

100

4.6

4.8
```

Both losses remain high.

Possible causes include

- model too small,
- insufficient interaction blocks,
- learning rate too low,
- inadequate training time.

---

# 22.13.19.15 Exploding Loss

Sometimes,

training suddenly becomes unstable.

Example

```text
Epoch

Loss

1

2.4

2

3.1

3

12.5

4

287.3

5

NaN
```

Possible causes include

- learning rate too large,
- corrupted dataset,
- numerical instability.

Training should usually be stopped immediately.

---

# 22.13.19.16 Monitoring Convergence

A practical strategy is to record the validation loss after every epoch.

When it stops improving,

additional training often provides little benefit.

Many researchers therefore employ

- early stopping,
- checkpoint selection,
- learning rate scheduling.

These techniques improve efficiency and reduce overfitting.

---

# 22.13.19.17 Saving the Best Checkpoint

The model with the **lowest validation loss** is usually selected as the final model.

Even if training continues,

later epochs may not improve predictive accuracy.

Consequently,

the best-performing checkpoint is generally used for

- inference,
- molecular dynamics,
- geometry optimization,
- publication-quality results.

---

# 22.13.19.18 Summary

The training log provides a real-time summary of the optimization process. By monitoring quantities such as the training loss, validation loss, energy RMSE, force RMSE, stress RMSE, learning rate, and epoch time, researchers can determine whether the model is converging successfully, beginning to overfit, or suffering from numerical instability. Correct interpretation of these metrics is essential for producing reliable and physically meaningful MACE models.

---

## Transition to Section 22.13.20

Although the numerical values reported during training provide valuable information, they are often easier to interpret visually. In the next section, we will learn how to visualize the training history by plotting the evolution of the loss, energy error, and force error, allowing us to identify convergence, overfitting, and optimization problems more effectively.

# 22.13.20 Visualizing the Training Process

Although numerical logs provide detailed information about the optimization process, it is often much easier to understand the behavior of a neural network through visualization.

Machine learning researchers rarely judge a model solely from printed loss values. Instead, they routinely generate plots showing how various performance metrics evolve throughout training.

Visualization enables us to

- determine whether the model is converging,
- identify overfitting,
- detect underfitting,
- observe learning rate changes,
- compare multiple experiments,
- and communicate results effectively in research papers.

In this section, we will learn how to visualize the training history of a MACE model using Python and Matplotlib.

---

# 22.13.20.1 Why Plot Training Curves?

Suppose the training log contains

```text
Epoch 1   Loss = 8.12
Epoch 2   Loss = 7.54
Epoch 3   Loss = 6.98
...
Epoch 200 Loss = 0.21
```

Although this information is useful, it is difficult to immediately recognize trends from hundreds of numerical values.

Instead, plotting the loss produces a figure such as

```text
Loss

8 │\
7 │ \
6 │  \
5 │   \
4 │    \
3 │      \
2 │        \
1 │          \_____
0 └──────────────────────

      Epoch
```

Within a few seconds, we can determine whether the model has converged successfully.

---

# 22.13.20.2 Typical Metrics to Visualize

During MACE training, the following quantities are commonly plotted.

- Training loss
- Validation loss
- Energy RMSE
- Force RMSE
- Stress RMSE
- Learning rate

These curves provide complementary information about the optimization process.

---

# 22.13.20.3 Importing Required Libraries

Begin by importing the required packages.

```python
import pandas as pd
import matplotlib.pyplot as plt
```

Here,

- **Pandas** is used to read the training log,
- **Matplotlib** is used to create publication-quality figures.

---

# 22.13.20.4 Reading the Training Log

Many MACE training runs save metrics in a CSV file.

Suppose the file is named

```text
training_log.csv
```

Load it using

```python
history = pd.read_csv("training_log.csv")
```

Display the first few rows.

```python
print(history.head())
```

Example output

```text
   epoch  train_loss  valid_loss  energy_rmse  force_rmse
0      1      8.431      8.672       352.4        0.462
1      2      7.921      8.014       331.7        0.439
2      3      7.235      7.403       304.1        0.401
```

---

# 22.13.20.5 Inspecting Available Columns

Before plotting,

inspect the available columns.

```python
print(history.columns)
```

Example

```text
Index([
'epoch',
'train_loss',
'valid_loss',
'energy_rmse',
'force_rmse',
'stress_rmse',
'learning_rate'
])
```

The exact column names depend on the training script.

---

# 22.13.20.6 Plotting the Training Loss

The simplest visualization is the training loss.

```python
plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["train_loss"]
)

plt.xlabel("Epoch")

plt.ylabel("Training Loss")

plt.title("Training Loss")

plt.show()
```

A successful run should produce a steadily decreasing curve.

---

# 22.13.20.7 Plotting the Validation Loss

Validation loss is plotted similarly.

```python
plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["valid_loss"]
)

plt.xlabel("Epoch")

plt.ylabel("Validation Loss")

plt.title("Validation Loss")

plt.show()
```

The validation curve provides the best indication of generalization.

---

# 22.13.20.8 Plotting Both Losses Together

The most common visualization compares

- training loss,
- validation loss.

```python
plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training"
)

plt.plot(
    history["epoch"],
    history["valid_loss"],
    label="Validation"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training History")

plt.legend()

plt.show()
```

This comparison immediately reveals overfitting.

---

# 22.13.20.9 Interpreting the Curves

A healthy training process often looks like

```text
Loss

8 │\
7 │ \
6 │  \
5 │   \
4 │    \
3 │      \
2 │        \
1 │          \_____
0 └──────────────────────

      Epoch
```

Both curves decrease together.

Eventually,

they flatten,

indicating convergence.

---

# 22.13.20.10 Identifying Overfitting

An overfitted model often exhibits

```text
Loss

Training

8 │\
7 │ \
6 │  \
5 │   \
4 │    \
3 │      \
2 │        \
1 │         \____

Validation

8 │\
7 │ \
6 │  \
5 │   \
4 │    \__
3 │       \____
2 │            \
1 │             \
0 └─────────────────────

      Epoch
```

Notice

- training loss decreases,
- validation loss increases.

This indicates that the network is memorizing the training data.

---

# 22.13.20.11 Plotting Energy RMSE

Energy prediction accuracy is another important metric.

```python
plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["energy_rmse"]
)

plt.xlabel("Epoch")

plt.ylabel("Energy RMSE (meV/atom)")

plt.title("Energy Prediction Error")

plt.show()
```

The curve should decrease throughout training.

---

# 22.13.20.12 Plotting Force RMSE

Similarly,

plot the force error.

```python
plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["force_rmse"]
)

plt.xlabel("Epoch")

plt.ylabel("Force RMSE (eV/Å)")

plt.title("Force Prediction Error")

plt.show()
```

Because MACE is primarily optimized for force prediction,

this is one of the most informative plots.

---

# 22.13.20.13 Plotting Stress RMSE

If stress training is enabled,

visualize the stress error.

```python
plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["stress_rmse"]
)

plt.xlabel("Epoch")

plt.ylabel("Stress RMSE (GPa)")

plt.title("Stress Prediction Error")

plt.show()
```

---

# 22.13.20.14 Visualizing the Learning Rate

Many training schedules gradually reduce the learning rate.

```python
plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["learning_rate"]
)

plt.xlabel("Epoch")

plt.ylabel("Learning Rate")

plt.title("Learning Rate Schedule")

plt.show()
```

Typical learning-rate schedules appear as

```text
Learning Rate

0.0010 │───────────
        │
0.0005 │      ───────
        │
0.0001 │           ──────

        └───────────────────

              Epoch
```

---

# 22.13.20.15 Saving Figures

Figures should be saved for future reference.

```python
plt.savefig(
    "training_loss.png",
    dpi=300,
    bbox_inches="tight"
)
```

Saving figures at

```text
300 dpi
```

is recommended for publication-quality images.

---

# 22.13.20.16 Comparing Multiple Experiments

Researchers often compare several models simultaneously.

For example,

```python
plt.plot(
    history_small["epoch"],
    history_small["valid_loss"],
    label="Small Model"
)

plt.plot(
    history_large["epoch"],
    history_large["valid_loss"],
    label="Large Model"
)

plt.legend()

plt.show()
```

Such comparisons help identify the best-performing architecture.

---

# 22.13.20.17 Using TensorBoard

In addition to Matplotlib,

many researchers monitor training using TensorBoard.

TensorBoard provides

- live loss curves,
- parameter histograms,
- gradient statistics,
- learning-rate schedules,
- model graphs.

Launch TensorBoard using

```bash
tensorboard --logdir logs/
```

Then open

```text
http://localhost:6006
```

in a web browser.

TensorBoard is especially useful for monitoring long training runs.

---

# 22.13.20.18 Best Practices

When analyzing training curves,

always check

- ✔ training loss decreases,
- ✔ validation loss decreases,
- ✔ energy RMSE decreases,
- ✔ force RMSE decreases,
- ✔ stress RMSE decreases (if applicable),
- ✔ learning rate changes as expected,
- ✔ no sudden spikes appear.

These visual inspections often reveal issues long before the final model is evaluated.

---

# 22.13.20.19 Summary

Visualizing the training process provides a clear understanding of how the MACE model learns over time. By plotting the training loss, validation loss, energy RMSE, force RMSE, stress RMSE, and learning rate, researchers can identify convergence, overfitting, underfitting, and optimization problems that may not be obvious from numerical logs alone. These plots are indispensable tools for model development and are routinely included in scientific publications.

---

## Transition to Section 22.13.21

After confirming that the model has trained successfully, the next step is to preserve the trained neural network for future use. In the following section, we will learn how to save trained MACE models, load them back into memory, and use them for inference without retraining.


# 22.13.21 Saving and Loading Trained MACE Models

Training a MACE model may require

- several hours,
- several days,
- or even several weeks

depending on the size of the dataset and the complexity of the neural network.

Once the model has been trained successfully, it becomes a valuable scientific asset. Repeating the entire training process every time predictions are needed would be computationally wasteful.

Instead, the trained neural network is **saved** to disk. The saved model can later be loaded to perform

- energy prediction,
- force prediction,
- molecular dynamics,
- geometry optimization,
- phonon calculations,
- active learning,
- or transfer learning.

Saving trained models is therefore one of the most important steps in the entire workflow.

---

# 22.13.21.1 Why Save a Model?

Suppose training required

```text
48 hours
```

on an NVIDIA A100 GPU.

Without saving,

any interruption such as

- computer shutdown,
- power failure,
- accidental deletion,

would require restarting the training from scratch.

By saving checkpoints,

training progress can always be recovered.

---

# 22.13.21.2 What Is Stored?

A trained MACE model contains far more than just its architecture.

Typically, the saved checkpoint stores

- neural network weights,
- model architecture,
- optimizer state,
- learning rate scheduler,
- epoch number,
- random seed,
- training metadata.

Together, these allow the training session to be resumed exactly where it stopped.

---

# 22.13.21.3 Checkpoints During Training

During optimization,

MACE periodically writes checkpoint files.

A typical output directory may look like

```text
results/

├── checkpoint_epoch_10.pt

├── checkpoint_epoch_20.pt

├── checkpoint_epoch_30.pt

├── best_model.pt

├── training.log

└── config.yaml
```

Each checkpoint represents the state of the neural network after a particular epoch.

---

# 22.13.21.4 The Best Model

Most training runs produce a special file

```text
best_model.pt
```

This is usually the model with the **lowest validation error**.

Notice that

the final epoch is **not necessarily** the best model.

For example,

```text
Epoch 150

Validation RMSE = 22 meV

Epoch 180

Validation RMSE = 17 meV

Epoch 200

Validation RMSE = 24 meV
```

Although training ended at epoch 200,

the checkpoint from epoch 180 is superior.

Therefore,

the best validation checkpoint should usually be used for inference.

---

# 22.13.21.5 Saving a Model with PyTorch

Since MACE is implemented in PyTorch,

models are saved using

```python
torch.save()
```

For example,

```python
import torch

torch.save(
    model.state_dict(),
    "mace_model.pt"
)
```

This stores all trainable parameters.

---

# 22.13.21.6 What Is `state_dict()`?

Every PyTorch neural network contains

```python
model.state_dict()
```

This object is a Python dictionary containing

- weights,
- biases,
- normalization parameters,
- learned tensors.

For example,

```python
state = model.state_dict()

print(state.keys())
```

Typical output

```text
embedding.weight

interaction_blocks.0.weight

interaction_blocks.1.weight

readout.weight

...
```

Each tensor represents a learned parameter.

---

# 22.13.21.7 Saving the Entire Model

Another possibility is

```python
torch.save(
    model,
    "complete_model.pt"
)
```

This saves

- the architecture,
- the parameters,

together.

However,

this approach is generally less portable across different software versions.

Most researchers therefore prefer saving only the `state_dict()`.

---

# 22.13.21.8 Loading Saved Parameters

To reload a trained model,

first recreate the same architecture.

```python
model = create_model()
```

Then load the saved parameters.

```python
model.load_state_dict(
    torch.load("mace_model.pt")
)
```

The model now contains the trained weights.

---

# 22.13.21.9 Switching to Evaluation Mode

Before making predictions,

always switch the network into evaluation mode.

```python
model.eval()
```

This disables training-specific operations such as

- dropout,
- batch normalization updates.

Although MACE does not always use these layers,

calling

```python
model.eval()
```

is standard PyTorch practice.

---

# 22.13.21.10 Loading onto the GPU

Suppose the model was trained on the GPU.

Move it to CUDA.

```python
device = "cuda"

model.to(device)
```

Similarly,

input data should also be moved to the same device.

```python
graph = graph.to(device)
```

Neural network parameters and input tensors must always reside on the same device.

---

# 22.13.21.11 Loading onto the CPU

If a GPU is unavailable,

the model can be loaded on the CPU.

```python
model.load_state_dict(
    torch.load(
        "mace_model.pt",
        map_location="cpu"
    )
)
```

This automatically converts all tensors to CPU memory.

---

# 22.13.21.12 Resuming Training

Suppose training stopped unexpectedly after

```text
Epoch 85
```

Resume from the latest checkpoint.

```python
checkpoint = torch.load(
    "checkpoint_epoch_85.pt"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

epoch = checkpoint["epoch"]
```

Training can now continue from epoch 86.

---

# 22.13.21.13 Inspecting Loaded Parameters

Verify that the model loaded correctly.

```python
for name, parameter in model.named_parameters():

    print(
        name,
        parameter.shape
    )
```

Example

```text
embedding.weight

torch.Size([128,32])

interaction.weight

torch.Size([256,128])

readout.weight

torch.Size([1,128])
```

This confirms that all tensors have been restored successfully.

---

# 22.13.21.14 Preventing Gradient Computation

During inference,

gradients are unnecessary.

Disable them using

```python
with torch.no_grad():

    prediction = model(graph)
```

Benefits include

- lower memory usage,
- faster prediction,
- reduced computational overhead.

---

# 22.13.21.15 Saving Training Metadata

In addition to the model,

researchers should record

- dataset version,
- cutoff radius,
- hidden irreps,
- correlation order,
- learning rate,
- optimizer,
- random seed,
- software versions.

A common approach is to save these settings in

```text
config.yaml
```

or

```text
config.json
```

This greatly improves reproducibility.

---

# 22.13.21.16 Version Compatibility

One common mistake is attempting to load

```text
Model trained with

MACE v0.3
```

using

```text
MACE v0.5
```

Changes in

- architecture,
- parameter names,
- software implementation,

may prevent successful loading.

For long-term reproducibility,

always record

- Python version,
- PyTorch version,
- MACE version,
- CUDA version.

---

# 22.13.21.17 Organizing Saved Models

A well-organized research directory may look like

```text
Project/

├── data/

│   ├── train.extxyz

│   ├── valid.extxyz

│   └── test.extxyz

├── models/

│   ├── best_model.pt

│   ├── checkpoint_epoch_50.pt

│   ├── checkpoint_epoch_100.pt

│   └── final_model.pt

├── logs/

│   ├── training.log

│   └── tensorboard/

├── figures/

└── config.yaml
```

Maintaining a clear directory structure simplifies collaboration and future analysis.

---

# 22.13.21.18 Best Practices

When saving trained MACE models,

always

- ✔ save the best validation checkpoint,
- ✔ retain intermediate checkpoints,
- ✔ store the training configuration,
- ✔ record software versions,
- ✔ verify that the model can be reloaded successfully,
- ✔ back up important checkpoints.

These practices ensure that valuable computational results are never lost.

---

# 22.13.21.19 Summary

Training a MACE model is often computationally expensive, making it essential to preserve the learned parameters for future use. PyTorch provides efficient mechanisms for saving and loading model weights through the `state_dict()`, while checkpoint files additionally preserve optimizer states and training progress. Proper model management—including version tracking, configuration storage, and organized directory structures—ensures reproducibility, facilitates collaboration, and allows trained interatomic potentials to be reused for prediction, simulation, and further development.

---

## Transition to Section 22.13.22

With the trained model successfully saved and reloaded, we are finally ready to use it for scientific prediction. In the next section, we will perform inference with a trained MACE model, demonstrating how to predict energies, atomic forces, and stresses for completely new crystal structures that were never seen during training.

# 22.13.22 Performing Inference with a Trained MACE Model

After successfully training and saving a MACE model, the next objective is to use it for **prediction**.

The process of applying a trained neural network to previously unseen atomic structures is called **inference**.

During inference,

the model parameters remain fixed.

Unlike training,

- no gradients are computed,
- no optimizer is used,
- no parameter updates occur.

Instead, the neural network performs only a forward pass to predict the physical properties of an atomic structure.

Inference is the stage at which a trained MACE model becomes useful for real scientific applications, including

- high-throughput materials screening,
- molecular dynamics,
- geometry optimization,
- defect calculations,
- active learning,
- autonomous materials discovery.

---

# 22.13.22.1 The Inference Pipeline

The inference workflow is considerably simpler than the training workflow.

```text
Crystal Structure

        │

        ▼

ASE Atoms Object

        │

        ▼

Neighbor Graph Construction

        │

        ▼

Trained MACE Model

        │

        ▼

Predicted Energy

Predicted Forces

Predicted Stress
```

Notice that

there is

- no loss function,
- no optimizer,
- no backpropagation.

The neural network simply performs a forward evaluation.

---

# 22.13.22.2 Loading the Trained Model

Suppose the trained model is stored as

```text
best_model.pt
```

First,

construct the model architecture.

```python
model = create_model()
```

Next,

load the trained parameters.

```python
import torch

model.load_state_dict(
    torch.load("best_model.pt")
)
```

Finally,

switch the network to evaluation mode.

```python
model.eval()
```

The model is now ready for inference.

---

# 22.13.22.3 Loading a New Structure

Suppose we have an unseen crystal stored in

```text
new_structure.extxyz
```

Read it using ASE.

```python
from ase.io import read

atoms = read("new_structure.extxyz")
```

This produces an ASE `Atoms` object.

---

# 22.13.22.4 Inspecting the Structure

Before prediction,

verify the structure.

```python
print(atoms)
```

Example

```text
Atoms(
    symbols='Si64',
    pbc=True
)
```

Check the number of atoms.

```python
print(len(atoms))
```

Example

```text
64
```

---

# 22.13.22.5 Converting to a Graph

MACE operates on graphs rather than directly on atomic coordinates.

The atomic structure must therefore be converted into a neighbor graph.

Conceptually,

```text
Atoms

↓

Neighbor Search

↓

Graph Construction

↓

Node Features

Edge Features

↓

Graph
```

The official MACE utilities perform this conversion automatically during prediction.

---

# 22.13.22.6 Disabling Gradient Computation

Inference does not require gradients.

Always use

```python
with torch.no_grad():

    prediction = model(graph)
```

Disabling gradients

- reduces memory consumption,
- accelerates inference,
- avoids unnecessary computations.

---

# 22.13.22.7 Predicting the Total Energy

The simplest prediction is the total energy.

Conceptually,

```python
with torch.no_grad():

    prediction = model(graph)

energy = prediction["energy"]
```

Example output

```text
-327.842 eV
```

This represents the total energy of the entire atomic configuration.

---

# 22.13.22.8 Predicting Atomic Forces

MACE simultaneously predicts atomic forces.

```python
forces = prediction["forces"]

print(forces)
```

Example

```text
[[ 0.012

  -0.031

   0.004]

[-0.012

  0.031

 -0.004]

...
]
```

The shape is

```python
print(forces.shape)
```

Example

```text
(64,3)
```

Each row corresponds to

- Fx,
- Fy,
- Fz.

---

# 22.13.22.9 Predicting Stress

If the model was trained with stress supervision,

it also predicts

```python
stress = prediction["stress"]

print(stress)
```

Example

```text
[[ 1.25 0.01 0.00]

 [0.01 1.27 0.00]

 [0.00 0.00 1.22]]
```

Stress prediction is useful for

- cell optimization,
- elastic properties,
- constant-pressure simulations.

---

# 22.13.22.10 Interpreting the Predictions

The prediction object typically contains

```text
Prediction

├── Energy

├── Forces

└── Stress
```

Depending on the implementation,

additional outputs may also be available,

including

- atomic energies,
- node embeddings,
- latent representations.

---

# 22.13.22.11 Predicting Multiple Structures

Entire datasets can be processed automatically.

```python
structures = read(
    "test.extxyz",
    index=":"
)

for atoms in structures:

    graph = convert_to_graph(atoms)

    with torch.no_grad():

        prediction = model(graph)

    print(prediction["energy"])
```

This loop performs inference for every configuration.

---

# 22.13.22.12 Storing Predictions

Predicted values are usually stored for later analysis.

```python
predicted_energy = []

for atoms in structures:

    graph = convert_to_graph(atoms)

    with torch.no_grad():

        output = model(graph)

    predicted_energy.append(
        output["energy"]
    )
```

The resulting list can later be saved as

- CSV,
- NumPy arrays,
- Pandas DataFrames.

---

# 22.13.22.13 Saving Results

For example,

save predicted energies.

```python
import pandas as pd

df = pd.DataFrame({

    "Predicted Energy":

    predicted_energy

})

df.to_csv(

    "predictions.csv",

    index=False

)
```

Example output

```text
Predicted Energy

-327.842

-328.105

-327.921

...
```

---

# 22.13.22.14 Measuring Inference Speed

Inference is significantly faster than DFT.

Measure prediction time.

```python
import time

start = time.time()

with torch.no_grad():

    prediction = model(graph)

end = time.time()

print(

    end-start

)
```

Example

```text
0.0042 seconds
```

The same DFT calculation might require

```text
20 minutes
```

This enormous speedup is one of the major advantages of machine learning interatomic potentials.

---

# 22.13.22.15 Comparing DFT and MACE Predictions

Suppose the DFT energy is

```text
-327.901 eV
```

The MACE prediction is

```text
-327.842 eV
```

The prediction error is

```python
error = abs(

-327.842

-

(-327.901)

)

print(error)
```

Output

```text
0.059 eV
```

Such comparisons are commonly performed on the test dataset.

---

# 22.13.22.16 Running Batch Inference Efficiently

Instead of predicting one structure at a time,

large datasets are usually processed in batches.

Benefits include

- improved GPU utilization,
- reduced runtime,
- higher throughput.

Batch inference is particularly important for

- million-structure screening,
- active learning,
- large materials databases.

---

# 22.13.22.17 Common Applications

Once trained,

MACE inference is used for

- geometry optimization,
- molecular dynamics,
- crystal structure relaxation,
- phonon calculations,
- defect energetics,
- high-throughput screening,
- active learning,
- autonomous laboratories.

These applications form the foundation of modern atomistic machine learning.

---

# 22.13.22.18 Best Practices

During inference,

always

- ✔ switch the model to evaluation mode,
- ✔ disable gradients,
- ✔ verify the input structure,
- ✔ record prediction time,
- ✔ compare predictions with DFT whenever possible,
- ✔ store results systematically.

Following these practices ensures efficient and reproducible scientific workflows.

---

# 22.13.22.19 Summary

Inference is the process of applying a trained MACE model to new atomic structures without updating the network parameters. By loading the trained model, converting atomic structures into graphs, disabling gradient computation, and performing a forward pass, MACE rapidly predicts total energies, atomic forces, and stress tensors. Compared with first-principles calculations, inference is several orders of magnitude faster, making trained MACE models invaluable tools for large-scale materials screening and atomistic simulations.

---

## Transition to Section 22.13.23

Predicting energies and forces is only the first step. One of the greatest strengths of MACE is its seamless integration with the Atomic Simulation Environment (ASE), allowing the model to act as a drop-in replacement for expensive quantum mechanical calculators. In the next section, we will learn how to use a trained MACE model as an ASE calculator for geometry optimization, molecular dynamics, and other atomistic simulations.

# 22.13.23 Using MACE as an ASE Calculator

In the previous section, we learned how to perform inference conceptually using a trained MACE model.

Although this demonstrates the underlying machine learning workflow, **it is not how MACE is typically used in research**.

In practical atomistic simulations, researchers almost never interact with the neural network directly.

Instead, MACE is used as an **ASE Calculator**.

The ASE Calculator interface allows MACE to behave exactly like a Density Functional Theory (DFT) calculator. Once attached to an `Atoms` object, ASE automatically requests

- total energy,
- atomic forces,
- stress tensor,

whenever they are needed.

This design is one of the greatest strengths of MACE because **every ASE algorithm immediately becomes compatible with the trained neural network**.

This includes

- geometry optimization,
- molecular dynamics,
- elastic constant calculations,
- equation of state calculations,
- nudged elastic band (NEB),
- phonons,
- vibrational analysis,

and many other atomistic simulation techniques.

---

# 22.13.23.1 ASE Calculator Architecture

The ASE workflow is remarkably elegant.

```text
                ASE Atoms Object
                       │
                       ▼
              Attached Calculator
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
     Total Energy    Forces       Stress
```

The calculator may be

- Quantum ESPRESSO
- VASP
- GPAW
- MACE
- CHGNet
- NequIP
- Allegro

From ASE's perspective, **they all behave identically**.

---

# 22.13.23.2 Why Use the Calculator Interface?

Suppose we wish to relax a crystal structure.

Without ASE,

we would have to

1. Predict forces.
2. Update positions.
3. Predict forces again.
4. Update positions.
5. Repeat hundreds of times.

ASE performs all of this automatically.

Once a calculator is attached,

every optimization algorithm simply calls

```python
atoms.get_forces()
```

or

```python
atoms.get_potential_energy()
```

ASE handles everything else.

---

# 22.13.23.3 Importing the Required Packages

We begin by importing ASE.

```python
from ase.io import read
```

Now import the MACE calculator.

```python
from mace.calculators import MACECalculator
```

This is the primary interface between ASE and MACE.

---

# 22.13.23.4 Loading a Trained Model

Assume training produced

```text
best_model.model
```

or

```text
MACE.model
```

Create the calculator.

```python
calculator = MACECalculator(
    model_paths="best_model.model",
    device="cuda"
)
```

This loads

- neural network weights,
- architecture,
- species information,

into memory.

If a GPU is unavailable,

simply use

```python
device="cpu"
```

---

# 22.13.23.5 Understanding `model_paths`

The argument

```python
model_paths=
```

specifies the trained model file.

Example

```python
calculator = MACECalculator(
    model_paths="results/best_model.model",
    device="cuda"
)
```

If multiple models are supplied,

MACE automatically performs ensemble predictions.

For example,

```python
calculator = MACECalculator(
    model_paths=[
        "model1.model",
        "model2.model",
        "model3.model"
    ],
    device="cuda"
)
```

Model ensembles generally improve prediction robustness.

---

# 22.13.23.6 Understanding the Device Argument

```python
device="cuda"
```

means

perform all neural network computations on the GPU.

Alternatively,

```python
device="cpu"
```

uses the CPU.

Typical performance comparison

| Device | Relative Speed |
|----------|---------------:|
| CPU | 1× |
| RTX 4090 | 20–60× |
| NVIDIA A100 | 50–150× |

The exact speedup depends on

- system size,
- batch size,
- GPU architecture.

---

# 22.13.23.7 Reading a Structure

Load an atomic structure.

For example,

```python
atoms = read("silicon.extxyz")
```

or

```python
atoms = read("structure.cif")
```

ASE supports numerous formats including

- CIF
- XYZ
- EXTXYZ
- POSCAR
- VASP
- PDB

---

# 22.13.23.8 Attaching the Calculator

Attach the trained MACE model.

```python
atoms.calc = calculator
```

This single line transforms the structure into a machine-learning-powered atomistic system.

Internally,

```text
Atoms

↓

MACE Calculator

↓

Neural Network

↓

Energy

Forces

Stress
```

---

# 22.13.23.9 Predicting the Total Energy

Once attached,

predicting the total energy becomes trivial.

```python
energy = atoms.get_potential_energy()

print(energy)
```

Example

```text
-327.845 eV
```

Notice that

there is no direct interaction with the neural network.

ASE automatically performs

- graph construction,
- forward propagation,
- energy extraction.

---

# 22.13.23.10 Predicting Atomic Forces

Similarly,

```python
forces = atoms.get_forces()

print(forces)
```

Example

```text
[[ 0.012

  -0.031

   0.004]

[-0.012

  0.031

 -0.004]

...
]
```

Shape

```python
print(forces.shape)
```

Output

```text
(64,3)
```

Each row represents

- Fx
- Fy
- Fz

for one atom.

---

# 22.13.23.11 Predicting the Stress Tensor

If the model supports stress prediction,

simply call

```python
stress = atoms.get_stress()

print(stress)
```

Example

```text
[

1.25

1.28

1.21

0.01

0.00

-0.01

]
```

ASE returns the stress in Voigt notation.

---

# 22.13.23.12 Understanding Lazy Evaluation

An important feature of ASE is

**lazy evaluation**.

When

```python
atoms.calc = calculator
```

is executed,

**nothing is computed immediately**.

Computation only occurs when one of the following is requested.

```python
atoms.get_potential_energy()

atoms.get_forces()

atoms.get_stress()
```

This avoids unnecessary calculations.

---

# 22.13.23.13 Reusing Previous Results

Suppose we call

```python
atoms.get_potential_energy()
```

twice.

ASE automatically caches the result.

If the atomic positions have not changed,

the neural network is **not evaluated again**.

However,

after changing positions,

```python
atoms.positions[0][0] += 0.01
```

ASE automatically recognizes that

the previous results are invalid,

and recomputes the energy.

---

# 22.13.23.14 Using Multiple Structures

Inference over many structures is straightforward.

```python
structures = read(
    "test.extxyz",
    index=":"
)

for atoms in structures:

    atoms.calc = calculator

    energy = atoms.get_potential_energy()

    print(energy)
```

This loop can evaluate thousands of structures automatically.

---

# 22.13.23.15 Saving Predicted Energies

Predictions may be collected.

```python
energies = []

for atoms in structures:

    atoms.calc = calculator

    energies.append(
        atoms.get_potential_energy()
    )
```

Later,

save them.

```python
import pandas as pd

pd.DataFrame({
    "Energy": energies
}).to_csv(
    "predictions.csv",
    index=False
)
```

---

# 22.13.23.16 Measuring Prediction Speed

Measure inference time.

```python
import time

start = time.time()

energy = atoms.get_potential_energy()

end = time.time()

print(
    end - start
)
```

Example

```text
0.0038 seconds
```

This demonstrates why machine learning potentials are so powerful.

A calculation requiring

minutes or hours

with DFT

can often be completed in

milliseconds

using MACE.

---

# 22.13.23.17 Common Calculator Methods

Once a calculator is attached,

the following ASE methods become available.

| Method | Description |
|----------|-------------|
| `atoms.get_potential_energy()` | Total energy |
| `atoms.get_forces()` | Atomic forces |
| `atoms.get_stress()` | Stress tensor |
| `atoms.get_total_energy()` | Total energy including kinetic energy (when applicable) |
| `atoms.get_kinetic_energy()` | Kinetic energy (for MD simulations) |

These methods are identical regardless of whether the underlying calculator is

- MACE,
- Quantum ESPRESSO,
- VASP,
- GPAW,
- CHGNet.

---

# 22.13.23.18 Why This Interface Is Powerful

Because MACE follows the ASE Calculator API,

every ASE algorithm immediately works with it.

For example,

```text
Geometry Optimization

↓

ASE Optimizer

↓

atoms.get_forces()

↓

MACE Calculator

↓

Neural Network
```

The optimizer does not know

or care

whether the forces came from

- DFT,
- MACE,
- CHGNet,
- NequIP.

This abstraction is one of ASE's greatest strengths.

---

# 22.13.23.19 Summary

The ASE Calculator interface is the standard way of using trained MACE models in practice. By attaching a `MACECalculator` to an ASE `Atoms` object, researchers can compute energies, forces, and stress tensors using simple ASE method calls such as `get_potential_energy()`, `get_forces()`, and `get_stress()`. Because MACE fully implements the ASE Calculator API, it integrates seamlessly with the extensive ecosystem of ASE simulation tools, enabling geometry optimization, molecular dynamics, phonon calculations, and many other atomistic simulations without modifying existing workflows.

---

## Transition to Section 22.13.24

Now that the trained MACE model behaves exactly like a conventional atomistic calculator, we can use it to perform one of the most common tasks in computational materials science: **geometry optimization**. In the next section, we will use ASE optimizers together with the MACE calculator to relax atomic structures until the predicted forces become negligibly small.

# 22.13.24 Geometry Optimization with MACE

One of the most common applications of an interatomic potential is **geometry optimization**, also called **structural relaxation**.

In nearly every computational materials science workflow, the initial atomic coordinates are not in their equilibrium configuration. Before calculating material properties such as

- formation energy,
- elastic constants,
- phonons,
- diffusion barriers,
- electronic properties,

the structure must first be relaxed to its minimum-energy configuration.

With Density Functional Theory (DFT), this process may require hundreds of expensive self-consistent field (SCF) calculations. Using a trained MACE potential, the same optimization can often be completed **thousands of times faster** while maintaining DFT-level accuracy.

---

# 22.13.24.1 What Is Geometry Optimization?

Geometry optimization is the process of finding the atomic arrangement that minimizes the total potential energy of the system.

Mathematically, we seek the atomic coordinates

$$
\mathbf{R}^*
=
\arg\min_{\mathbf{R}}
E(\mathbf{R})
$$

where

- $\mathbf{R}$ represents all atomic coordinates,
- $E(\mathbf{R})$ is the total potential energy,
- $\mathbf{R}^*$ is the equilibrium structure.

At equilibrium,

the net force acting on every atom approaches zero.

---

# 22.13.24.2 Relationship Between Energy and Forces

Recall that atomic forces are obtained from the energy gradient.

$$
\mathbf{F}_i
=
-
\nabla_i
E
$$

or equivalently,

$$
\mathbf{F}_i
=
-
\frac{\partial E}{\partial \mathbf{R}_i}
$$

Therefore,

minimizing the energy is equivalent to reducing the forces acting on all atoms.

---

# 22.13.24.3 Optimization Workflow

The complete optimization procedure is

```text
Initial Structure

        │

        ▼

Compute Energy

Compute Forces

        │

        ▼

Move Atoms

        │

        ▼

Compute New Forces

        │

        ▼

Move Again

        │

        ▼

Repeat Until

Maximum Force < Threshold
```

This iterative process continues until convergence.

---

# 22.13.24.4 Optimization in ASE

ASE provides several optimization algorithms.

Some of the most commonly used are

| Optimizer | Characteristics |
|-----------|-----------------|
| BFGS | General-purpose optimizer |
| LBFGS | Memory-efficient version of BFGS |
| FIRE | Fast inertial relaxation engine |
| MDMin | Molecular dynamics minimization |
| GPMin | Gaussian-process optimizer |

Among these,

**BFGS** and **FIRE** are the most frequently used with MACE.

---

# 22.13.24.5 Loading the Structure

Begin by reading the structure.

```python
from ase.io import read

atoms = read("structure.cif")
```

The structure may also be stored as

- POSCAR
- EXTXYZ
- XYZ
- CIF

or any other ASE-supported format.

---

# 22.13.24.6 Attaching the MACE Calculator

Load the trained model.

```python
from mace.calculators import MACECalculator

calculator = MACECalculator(
    model_paths="best_model.model",
    device="cuda"
)
```

Attach it to the atomic structure.

```python
atoms.calc = calculator
```

The optimizer will now obtain all energies and forces directly from the neural network.

---

# 22.13.24.7 Using the BFGS Optimizer

Import the optimizer.

```python
from ase.optimize import BFGS
```

Create the optimizer.

```python
optimizer = BFGS(atoms)
```

The optimizer now controls the movement of atoms.

---

# 22.13.24.8 Running the Optimization

The optimization begins with

```python
optimizer.run(fmax=0.01)
```

Here,

```python
fmax
```

is the convergence criterion.

Optimization stops when

$$
\max_i
\|
\mathbf{F}_i
\|
<
0.01
\;
\mathrm{eV/Å}
$$

---

# 22.13.24.9 Understanding `fmax`

The parameter

```python
fmax
```

represents the maximum force allowed on any atom.

Common values are

| fmax | Application |
|------:|-------------|
| 0.10 eV/Å | Quick relaxation |
| 0.05 eV/Å | Routine optimization |
| 0.02 eV/Å | Accurate optimization |
| 0.01 eV/Å | High-precision optimization |

Smaller values generally produce more accurate structures but require additional optimization steps.

---

# 22.13.24.10 Saving the Relaxed Structure

After optimization,

save the relaxed structure.

```python
from ase.io import write

write(
    "relaxed_structure.cif",
    atoms
)
```

The optimized coordinates can now be used for further calculations.

---

# 22.13.24.11 Comparing Initial and Final Energies

Before optimization,

```python
initial_energy = atoms.get_potential_energy()
```

After optimization,

```python
final_energy = atoms.get_potential_energy()
```

Compute the energy change.

```python
print(
    final_energy - initial_energy
)
```

Example

```text
-1.42 eV
```

The negative value indicates that the optimized structure is more stable.

---

# 22.13.24.12 Monitoring Optimization Progress

BFGS prints information similar to

```text
Step

Energy (eV)

Max Force (eV/Å)

0

-325.42

0.812

1

-326.18

0.511

2

-326.91

0.233

3

-327.14

0.091

4

-327.22

0.018

5

-327.23

0.009
```

Observe that

- the energy decreases,
- the forces gradually approach zero.

---

# 22.13.24.13 Writing an Optimization Trajectory

Optimization trajectories are valuable for visualization.

```python
optimizer = BFGS(
    atoms,
    trajectory="relaxation.traj"
)
```

Every optimization step is now stored.

The trajectory can later be replayed using ASE visualization tools.

---

# 22.13.24.14 Using the FIRE Optimizer

An alternative optimizer is FIRE.

```python
from ase.optimize import FIRE

optimizer = FIRE(atoms)

optimizer.run(
    fmax=0.01
)
```

FIRE is particularly effective for

- large systems,
- noisy force fields,
- machine learning potentials.

---

# 22.13.24.15 Relaxing the Unit Cell

Sometimes both

- atomic positions,
- lattice vectors,

must be optimized.

ASE provides

```python
UnitCellFilter
```

for this purpose.

```python
from ase.filters import UnitCellFilter

ucf = UnitCellFilter(atoms)

optimizer = BFGS(ucf)

optimizer.run(fmax=0.01)
```

Now both

- atom positions,
- cell dimensions

are optimized simultaneously.

---

# 22.13.24.16 Visualizing the Relaxed Structure

The optimized structure may be viewed using

```python
from ase.visualize import view

view(atoms)
```

Researchers often compare

- initial structure,
- optimized structure,

to observe atomic displacements.

---

# 22.13.24.17 Why MACE Makes Relaxation Fast

Consider a structure requiring

300 optimization steps.

With DFT,

each force evaluation may require

```text
5 minutes
```

Total runtime

```text
300 × 5 min

=

1500 minutes

≈ 25 hours
```

With MACE,

each force evaluation may require only

```text
0.01 seconds
```

Total runtime

```text
300 × 0.01

=

3 seconds
```

This dramatic speedup enables

- high-throughput relaxation,
- screening of thousands of materials,
- large-scale atomistic simulations.

---

# 22.13.24.18 Best Practices

When performing geometry optimization,

always

- ✔ verify that the calculator is attached correctly,
- ✔ choose an appropriate optimizer,
- ✔ use a suitable force threshold,
- ✔ save the optimization trajectory,
- ✔ write the relaxed structure to disk,
- ✔ compare the initial and final energies,
- ✔ inspect the final atomic positions.

---

# 22.13.24.19 Summary

Geometry optimization is one of the most important applications of a trained MACE model. By combining the MACE calculator with ASE optimizers such as BFGS or FIRE, researchers can efficiently relax atomic structures to their equilibrium configurations using machine-learning-predicted energies and forces. Because MACE evaluations are orders of magnitude faster than DFT calculations, structural relaxation that once required hours or days can often be completed within seconds while preserving near-DFT accuracy.

---

## Transition to Section 22.13.25

Geometry optimization is a static calculation that identifies the equilibrium structure. However, many important materials phenomena—such as diffusion, phase transitions, defect migration, and thermal expansion—depend on **atomic motion over time**. In the next section, we will use the trained MACE model to perform **molecular dynamics (MD) simulations** within the Atomic Simulation Environment.

# 22.13.25 Molecular Dynamics Simulations with MACE

Geometry optimization determines the equilibrium atomic structure by locating the minimum of the potential energy surface. However, real materials are rarely static. At finite temperatures, atoms continuously vibrate, collide, diffuse, rotate, and undergo structural transformations.

To study these dynamic processes, we perform **Molecular Dynamics (MD)** simulations.

In Molecular Dynamics, the positions and velocities of atoms evolve over time according to Newton's equations of motion, while the interatomic forces are supplied by an underlying potential. Traditionally, these forces are computed using Density Functional Theory (DFT) or classical force fields. With MACE, the forces are predicted by a machine learning interatomic potential that closely approximates DFT while being several orders of magnitude faster.

This combination enables simulations that are both **accurate** and **computationally efficient**, making MACE a powerful tool for modern atomistic materials research.

---

# 22.13.25.1 What Is Molecular Dynamics?

Molecular Dynamics (MD) is a numerical technique that predicts how atoms move over time by solving Newton's second law.

For each atom,

$$
\mathbf{F}_i = m_i\mathbf{a}_i
$$

where

- $\mathbf{F}_i$ is the force,
- $m_i$ is the atomic mass,
- $\mathbf{a}_i$ is the acceleration.

Since

$$
\mathbf{a}_i=\frac{d^2\mathbf{r}_i}{dt^2},
$$

the atomic positions evolve according to

$$
m_i\frac{d^2\mathbf{r}_i}{dt^2}
=
\mathbf{F}_i.
$$

In every MD step,

1. Compute atomic forces.
2. Compute accelerations.
3. Update velocities.
4. Update atomic positions.
5. Repeat.

---

# 22.13.25.2 The Molecular Dynamics Loop

The complete simulation follows the cycle

```text
Initial Structure

        │

        ▼

Calculate Forces

        │

        ▼

Compute Accelerations

        │

        ▼

Update Velocities

        │

        ▼

Update Positions

        │

        ▼

New Atomic Structure

        │

        └───────────────┐
                        │
                        ▼
                 Repeat Thousands
                     of Times
```

This loop is executed until the desired simulation time is reached.

---

# 22.13.25.3 Why Use MACE for Molecular Dynamics?

DFT-based Molecular Dynamics (often called **Ab Initio Molecular Dynamics** or **AIMD**) is extremely accurate but computationally expensive.

For example,

| Method | Typical Force Evaluation |
|---------|-------------------------:|
| DFT | 10–300 seconds |
| Classical Force Field | <0.001 s |
| MACE | 0.001–0.05 s |

MACE provides a balance between

- DFT-level accuracy,
- near-classical simulation speed.

This enables simulations of

- larger systems,
- longer time scales,
- more realistic materials.

---

# 22.13.25.4 ASE Molecular Dynamics Framework

ASE provides a complete Molecular Dynamics framework.

The workflow is

```text
Atoms

↓

MACE Calculator

↓

ASE MD Integrator

↓

Trajectory

↓

Analysis
```

The MD integrator automatically

- requests forces from MACE,
- integrates Newton's equations,
- updates the structure.

---

# 22.13.25.5 Loading the Structure

Load the atomic structure.

```python
from ase.io import read

atoms = read("silicon.cif")
```

Attach the trained MACE calculator.

```python
from mace.calculators import MACECalculator

calculator = MACECalculator(
    model_paths="best_model.model",
    device="cuda"
)

atoms.calc = calculator
```

---

# 22.13.25.6 Initializing Atomic Velocities

Before starting MD,

the atoms must be assigned initial velocities.

ASE provides

```python
MaxwellBoltzmannDistribution
```

which generates velocities following the Maxwell–Boltzmann distribution.

```python
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.units import kB

MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=300
)
```

Here,

```text
300 K
```

is the desired temperature.

The assigned velocities satisfy the statistical distribution expected for a system in thermal equilibrium.

---

# 22.13.25.7 Why Maxwell–Boltzmann Distribution?

At thermal equilibrium,

atomic velocities are **not identical**.

Instead,

their probability follows

$$
P(v)
\propto
v^2
\exp\left(
-\frac{mv^2}{2k_BT}
\right).
$$

This distribution ensures that

- some atoms move slowly,
- most move near the average speed,
- a few move much faster.

Using this initialization produces physically realistic simulations.

---

# 22.13.25.8 Choosing an Integrator

ASE offers several integration algorithms.

Common choices include

| Integrator | Typical Use |
|------------|-------------|
| VelocityVerlet | NVE simulations |
| Langevin | NVT simulations |
| Berendsen | Temperature control |
| NoseHoover | Canonical ensemble |

The simplest is **Velocity Verlet**.

---

# 22.13.25.9 Velocity Verlet Integration

Import the integrator.

```python
from ase.md.verlet import VelocityVerlet
```

Create the simulation.

```python
from ase import units

dyn = VelocityVerlet(
    atoms,
    timestep=1.0 * units.fs
)
```

Here,

```text
1 femtosecond
```

is the integration timestep.

---

# 22.13.25.10 Choosing the Time Step

The timestep determines how frequently Newton's equations are solved.

Typical values are

| Material | Timestep |
|----------|----------|
| Metals | 1–2 fs |
| Semiconductors | 1 fs |
| Molecules containing H | 0.25–0.5 fs |

Choosing a timestep that is too large can produce unstable trajectories.

---

# 22.13.25.11 Running Molecular Dynamics

Execute

```python
dyn.run(5000)
```

This performs

```text
5000

time steps.
```

With a timestep of

```text
1 fs
```

the total simulation time is

$$
5000
\times
1\;\mathrm{fs}
=
5000\;\mathrm{fs}
=
5\;\mathrm{ps}.
$$

---

# 22.13.25.12 Saving the Trajectory

During the simulation,

it is useful to save every configuration.

```python
from ase.io.trajectory import Trajectory

traj = Trajectory(
    "md.traj",
    "w",
    atoms
)

dyn.attach(
    traj.write,
    interval=10
)
```

Now,

every

```text
10

steps
```

the current atomic configuration is stored.

---

# 22.13.25.13 Printing Simulation Information

Monitor the simulation by printing

- energy,
- temperature,
- simulation step.

```python
def print_status():

    print(
        atoms.get_potential_energy(),
        atoms.get_temperature()
    )

dyn.attach(
    print_status,
    interval=100
)
```

Example output

```text
Step 100

Energy

-327.82 eV

Temperature

302 K
```

---

# 22.13.25.14 Visualizing the Trajectory

After completion,

the trajectory can be viewed.

```python
from ase.visualize import view

trajectory = read(
    "md.traj",
    index=":"
)

view(trajectory)
```

ASE displays the atomic motion as an animation.

---

# 22.13.25.15 Total Energy During MD

For an ideal NVE simulation,

the total energy

$$
E_{\text{total}}
=
E_{\text{kinetic}}
+
E_{\text{potential}}
$$

should remain approximately constant.

Small fluctuations occur due to numerical integration,

but large drifts indicate

- timestep too large,
- unstable potential,
- numerical errors.

---

# 22.13.25.16 Applications of MACE Molecular Dynamics

Machine learning molecular dynamics is widely used for studying

- thermal expansion,
- melting,
- diffusion,
- phase transitions,
- defect migration,
- lithium-ion transport,
- phonon dynamics,
- mechanical deformation,
- crack propagation,
- grain boundary motion.

Many of these simulations would be prohibitively expensive using DFT alone.

---

# 22.13.25.17 Comparison with Ab Initio Molecular Dynamics

| Feature | AIMD | MACE MD |
|---------|------|---------|
| Forces | DFT | Neural Network |
| Accuracy | Very High | Near DFT |
| Speed | Slow | Very Fast |
| Typical Atoms | 100–500 | Thousands to Millions |
| Typical Simulation Time | Few ps | Hundreds of ps to ns |

This dramatic increase in accessible length and time scales is one of the principal reasons machine learning interatomic potentials have become central to computational materials science.

---

# 22.13.25.18 Complete Example

A minimal Molecular Dynamics simulation is shown below.

```python
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator

atoms = read("silicon.cif")

atoms.calc = MACECalculator(
    model_paths="best_model.model",
    device="cuda"
)

MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=300
)

dyn = VelocityVerlet(
    atoms,
    timestep=1.0 * units.fs
)

dyn.run(5000)
```

This example demonstrates the complete workflow from loading a trained MACE model to performing a Molecular Dynamics simulation.

---

# 22.13.25.19 Best Practices

When performing Molecular Dynamics with MACE,

always

- ✔ relax the structure before starting MD,
- ✔ initialize velocities using the Maxwell–Boltzmann distribution,
- ✔ choose an appropriate timestep,
- ✔ monitor total energy,
- ✔ save the trajectory,
- ✔ verify that temperature remains physically reasonable,
- ✔ compare selected snapshots with DFT whenever possible.

---

# 22.13.25.20 Summary

Molecular Dynamics extends machine learning interatomic potentials beyond static structure prediction by enabling the simulation of atomic motion over time. Using the MACE calculator together with ASE's Molecular Dynamics framework, researchers can efficiently perform simulations that predict energies and forces at near-DFT accuracy while achieving computational speeds that are orders of magnitude faster than Ab Initio Molecular Dynamics. This capability makes MACE an indispensable tool for investigating temperature-dependent phenomena, diffusion processes, phase transformations, and large-scale atomistic behavior.

---

## Transition to Section 22.13.26

Thus far, we have focused on using MACE as a standalone interatomic potential. In practical research, however, MACE is often integrated into **active learning workflows**, where Molecular Dynamics identifies new atomic configurations, DFT labels these configurations, and the neural network is retrained iteratively. In the next section, we will study how MACE fits into the complete active learning cycle for continuously improving machine learning interatomic potentials.

# 22.13.26 Active Learning with MACE

One of the greatest advantages of modern machine learning interatomic potentials is that they are **not static models**. Unlike traditional empirical force fields, which remain unchanged after parameter fitting, a MACE model can continuously improve as new training data become available.

This process is known as **Active Learning**.

Instead of constructing a massive dataset before training, active learning allows the model itself to identify configurations where its predictions are uncertain. These configurations are then evaluated using high-accuracy Density Functional Theory (DFT), added to the training dataset, and used to retrain the neural network.

The result is a progressively more accurate interatomic potential that focuses computational resources only on the most informative atomic structures.

Active learning has become one of the most important methodologies in modern materials informatics because it enables the automatic construction of highly accurate machine learning potentials with significantly fewer expensive DFT calculations.

---

# 22.13.26.1 Motivation

Suppose we wish to develop a MACE potential for silicon.

Initially,

we may have

```text
500

DFT structures.
```

After training,

the model performs well near equilibrium.

However,

during Molecular Dynamics,

the system may encounter

- highly distorted structures,
- defect configurations,
- high-temperature environments,
- atomic collisions.

Since these structures were absent from the original dataset,

the model may produce unreliable predictions.

The obvious solution would be

"Generate millions of DFT structures."

Unfortunately,

this is computationally impossible.

Instead,

active learning identifies only the configurations that actually require new DFT calculations.

---

# 22.13.26.2 The Basic Idea

The central philosophy of active learning is simple.

```text
Train Model

↓

Run Simulation

↓

Detect Uncertain Structures

↓

Perform DFT

↓

Add New Data

↓

Retrain Model

↓

Repeat
```

Each iteration expands the model's knowledge.

---

# 22.13.26.3 Why Random Sampling Is Inefficient

Imagine exploring the potential energy surface.

```text
Huge Configuration Space

□□□□□□□□□□□□□□□□□□□□

□□□□□□□□□□□□□□□□□□□□

□□□□□□□□□□□□□□□□□□□□
```

Most randomly selected structures

- are physically unrealistic,
- duplicate existing information,
- contribute little to model improvement.

Running DFT on these structures wastes computational resources.

Active learning instead selects only the regions where the model lacks confidence.

---

# 22.13.26.4 Exploration Versus Exploitation

Active learning balances two competing objectives.

### Exploration

Discover entirely new regions of configuration space.

Examples

- high temperatures,
- defects,
- surfaces,
- grain boundaries.

### Exploitation

Improve accuracy in regions already visited.

Examples

- equilibrium crystals,
- phonon vibrations,
- elastic deformation.

An effective active learning strategy balances both objectives.

---

# 22.13.26.5 Active Learning Cycle

The complete workflow is

```text
Initial DFT Dataset

        │

        ▼

Train MACE

        │

        ▼

Run MD

        │

        ▼

Estimate Uncertainty

        │

        ▼

Select Important Structures

        │

        ▼

Run DFT

        │

        ▼

Expand Dataset

        │

        ▼

Retrain MACE

        │

        └───────────────┐
                        │
                        ▼
                     Repeat
```

Each cycle improves the neural network.

---

# 22.13.26.6 Step 1 — Initial Dataset

Every active learning workflow begins with

a relatively small DFT dataset.

Example

```text
500

relaxed structures

+

phonons

+

strained cells
```

These structures provide the initial knowledge required for training.

---

# 22.13.26.7 Step 2 — Train the Initial MACE Model

Train MACE normally.

```bash
mace_run_train \
    --train_file train.extxyz \
    --valid_fraction 0.1 \
    --energy_key energy \
    --forces_key forces
```

This produces

```text
best_model.model
```

---

# 22.13.26.8 Step 3 — Run Molecular Dynamics

Next,

perform Molecular Dynamics using the trained model.

```text
Initial Structure

↓

MACE MD

↓

Thousands of Configurations
```

Most of these configurations are predicted accurately.

However,

some may lie outside the training distribution.

---

# 22.13.26.9 Step 4 — Estimate Model Confidence

The crucial question is

> **Can the model trust its own prediction?**

Machine learning models generally cannot answer this directly.

Instead,

uncertainty is estimated using techniques such as

- model ensembles,
- committee models,
- Bayesian inference,
- Monte Carlo Dropout.

These methods are discussed in detail in **Chapter 26: Uncertainty Quantification**.

---

# 22.13.26.10 Committee Models

A common strategy is to train several independent MACE models.

```text
Model 1

Model 2

Model 3

Model 4
```

Each predicts

- energy,
- forces,
- stress.

If all models agree,

the prediction is likely reliable.

If they disagree,

the configuration is considered uncertain.

---

# 22.13.26.11 Example of Model Agreement

Suppose four models predict

```text
Energy

-325.812

-325.815

-325.809

-325.814
```

All predictions are nearly identical.

This indicates

high confidence.

---

# 22.13.26.12 Example of Model Disagreement

Now consider

```text
Energy

-325.2

-329.1

-326.8

-322.5
```

Large disagreement indicates

the configuration lies outside the training distribution.

This structure should be labeled using DFT.

---

# 22.13.26.13 Selecting New Structures

Only highly uncertain configurations are selected.

```text
100000

MD Frames

↓

Uncertainty Estimation

↓

150

Selected Structures

↓

DFT
```

Notice that

only

```text
0.15%
```

of the simulation requires expensive quantum calculations.

---

# 22.13.26.14 Running New DFT Calculations

The selected structures are evaluated using

- Quantum ESPRESSO,
- VASP,
- CASTEP,
- GPAW,

or another first-principles code.

Each calculation provides

- total energy,
- atomic forces,
- stress tensor.

These become new training labels.

---

# 22.13.26.15 Expanding the Dataset

The newly labeled structures are appended.

```text
Old Dataset

500 Structures

+

150 New Structures

↓

650 Structures
```

The expanded dataset now contains information about previously unexplored regions.

---

# 22.13.26.16 Retraining the Model

Retrain MACE using the expanded dataset.

```bash
mace_run_train \
    --train_file updated_train.extxyz \
    --valid_fraction 0.1
```

The new model generally exhibits

- lower prediction error,
- improved stability,
- better transferability.

---

# 22.13.26.17 Multiple Active Learning Iterations

Active learning rarely ends after one cycle.

Instead,

the workflow becomes

```text
Iteration 1

↓

500 Structures

↓

Train

↓

MD

↓

DFT

↓

650 Structures

↓

Iteration 2

↓

Train

↓

MD

↓

DFT

↓

900 Structures

↓

Iteration 3

↓

...
```

Eventually,

the model converges to a robust representation of the target material.

---

# 22.13.26.18 Advantages of Active Learning

Compared with constructing a dataset entirely by hand,

active learning

- reduces DFT calculations,
- minimizes redundant data,
- focuses on difficult configurations,
- improves model robustness,
- accelerates potential development,
- enables autonomous dataset generation.

These advantages make active learning one of the defining features of modern machine learning interatomic potentials.

---

# 22.13.26.19 Practical Applications

Active learning has been successfully applied to

- high-temperature Molecular Dynamics,
- crack propagation,
- defect migration,
- battery materials,
- catalytic surfaces,
- amorphous materials,
- grain boundaries,
- phase transitions,
- irradiation damage,
- alloy design.

Many state-of-the-art interatomic potentials are constructed almost entirely through active learning.

---

# 22.13.26.20 Best Practices

When performing active learning,

always

- ✔ begin with a diverse initial dataset,
- ✔ monitor model uncertainty,
- ✔ avoid selecting redundant structures,
- ✔ validate newly generated DFT data,
- ✔ retrain after every active learning cycle,
- ✔ evaluate performance on an independent test set,
- ✔ repeat until prediction errors stabilize.

---

# 22.13.26.21 Summary

Active learning transforms the development of machine learning interatomic potentials from a one-time training procedure into an iterative improvement process. By allowing the model to identify configurations where it is uncertain, active learning concentrates expensive DFT calculations on the most informative atomic structures. These newly labeled configurations are incorporated into the training dataset, enabling successive generations of MACE models to become increasingly accurate, robust, and transferable. As a result, active learning has become a cornerstone of modern machine learning potential development and large-scale atomistic simulation.

---

## Transition to Section 22.13.27

Active learning provides a systematic framework for improving machine learning interatomic potentials. However, before deploying a newly trained model in scientific research, it is essential to verify that it reproduces reference quantum mechanical calculations with sufficient accuracy. In the next section, we will study **validation and benchmarking of MACE models**, including comparisons with DFT, error metrics, parity plots, force correlation analysis, and rigorous performance evaluation.

# 22.13.27 Validation and Benchmarking of MACE Models

Training a MACE model is only the beginning. Before the model can be trusted for scientific simulations, it must undergo rigorous validation and benchmarking.

A machine learning interatomic potential should never be judged solely by its training loss. A model may achieve an extremely low training error while failing completely on unseen atomic configurations. Therefore, proper evaluation requires comparing the model against independent Density Functional Theory (DFT) calculations using objective statistical metrics.

Validation answers a fundamental scientific question:

> **How accurately does the trained MACE model reproduce quantum mechanical calculations on structures it has never seen before?**

This section introduces the methodologies used by researchers to evaluate the quality, robustness, and transferability of MACE models.

---

# 22.13.27.1 Why Validation Is Necessary

Suppose a MACE model is trained using

```text
10,000

DFT structures.
```

During training,

the neural network minimizes the prediction error on these structures.

However,

a successful machine learning model must generalize beyond the training data.

For example,

the model may later encounter

- higher temperatures,
- strained crystals,
- defects,
- vacancies,
- surfaces,
- grain boundaries,
- amorphous structures.

Validation determines whether the model can accurately predict these unseen configurations.

---

# 22.13.27.2 Training, Validation, and Test Sets

Machine learning datasets are typically divided into three subsets.

```text
Entire Dataset

──────────────────────────────

Training Set

↓

Model Learning

──────────────────────────────

Validation Set

↓

Hyperparameter Selection

──────────────────────────────

Test Set

↓

Final Evaluation
```

Each subset serves a distinct purpose.

### Training Set

Used to optimize the neural network parameters.

### Validation Set

Used during training to monitor generalization and select the best checkpoint.

### Test Set

Used only once after training to evaluate the final model.

The test set must never influence the training process.

---

# 22.13.27.3 Common Validation Metrics

For interatomic potentials,

the most important quantities are

- Energy
- Atomic Forces
- Stress Tensor

Each quantity is evaluated independently using statistical error metrics.

The most common metrics are

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Maximum Error
- Coefficient of Determination ($R^2$)

---

# 22.13.27.4 Mean Absolute Error (MAE)

The Mean Absolute Error measures the average magnitude of prediction errors.

$$
\mathrm{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
$$

where

- $y_i$ is the DFT value,
- $\hat{y}_i$ is the MACE prediction,
- $N$ is the number of samples.

MAE is easy to interpret because it has the same units as the predicted quantity.

Example

```text
Energy MAE

=

8 meV/atom
```

means that the average prediction error is only

```text
8 meV

per atom.
```

---

# 22.13.27.5 Root Mean Square Error (RMSE)

RMSE penalizes large prediction errors more strongly.

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
$$

Because the errors are squared,

large deviations contribute disproportionately.

RMSE is therefore more sensitive to outliers than MAE.

---

# 22.13.27.6 Mean Absolute Error Versus RMSE

Suppose the prediction errors are

```text
1

2

3

20
```

The MAE becomes

```text
6.5
```

whereas the RMSE is significantly larger because the error of

```text
20
```

is squared.

Thus,

RMSE highlights rare but severe prediction failures.

---

# 22.13.27.7 Energy Error

Energy errors are usually reported in

```text
meV/atom
```

rather than

```text
eV
```

This normalization allows fair comparison between systems containing different numbers of atoms.

Typical values

| Energy MAE | Interpretation |
|------------|----------------|
| >100 meV/atom | Poor |
| 30–100 meV/atom | Moderate |
| 10–30 meV/atom | Good |
| <10 meV/atom | Excellent |

---

# 22.13.27.8 Force Error

Forces determine atomic motion,

so force accuracy is often more important than energy accuracy.

Force errors are reported in

```text
eV/Å
```

Typical values

| Force MAE | Interpretation |
|-----------|----------------|
| >0.10 eV/Å | Poor |
| 0.05–0.10 eV/Å | Acceptable |
| 0.02–0.05 eV/Å | Good |
| <0.02 eV/Å | Excellent |

---

# 22.13.27.9 Stress Error

Stress prediction is essential for

- lattice optimization,
- elastic constants,
- constant-pressure molecular dynamics.

Stress errors are commonly reported in

```text
GPa
```

---

# 22.13.27.10 Computing MAE in Python

Suppose we have

```python
import numpy as np

true_energy = np.array([
    -10.1,
    -9.8,
    -10.4
])

pred_energy = np.array([
    -10.0,
    -9.9,
    -10.3
])
```

Compute the MAE.

```python
mae = np.mean(
    np.abs(
        true_energy -
        pred_energy
    )
)

print(mae)
```

Output

```text
0.10
```

---

# 22.13.27.11 Computing RMSE

Similarly,

```python
rmse = np.sqrt(
    np.mean(
        (
            true_energy -
            pred_energy
        )**2
    )
)

print(rmse)
```

---

# 22.13.27.12 Using scikit-learn

The same calculations can be performed using

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

mae = mean_absolute_error(
    true_energy,
    pred_energy
)

rmse = np.sqrt(
    mean_squared_error(
        true_energy,
        pred_energy
    )
)
```

This approach is commonly used in machine learning workflows.

---

# 22.13.27.13 Coefficient of Determination ($R^2$)

Another useful metric is the coefficient of determination.

$$
R^2
=
1
-
\frac{
\sum
(y-\hat y)^2
}{
\sum
(y-\bar y)^2
}
$$

Interpretation

| $R^2$ | Meaning |
|-------|----------|
| 1.0 | Perfect prediction |
| 0.99 | Excellent |
| 0.95 | Very good |
| 0.80 | Moderate |
| 0 | No predictive ability |

For high-quality MACE models,

$R^2$ is often greater than

```text
0.99
```

for both energies and forces.

---

# 22.13.27.14 Evaluating Forces

Forces contain three components per atom.

Suppose

```python
forces_true.shape
```

returns

```text
(1000,3)
```

Flatten them before computing statistics.

```python
mae_force = mean_absolute_error(

    forces_true.flatten(),

    forces_pred.flatten()

)
```

This evaluates every force component equally.

---

# 22.13.27.15 Validation Workflow

The complete evaluation pipeline is

```text
Test Dataset

↓

Predict Using MACE

↓

Compare with DFT

↓

Compute

MAE

RMSE

R²

↓

Generate Figures

↓

Report Performance
```

This workflow should be performed before deploying any model.

---

# 22.13.27.16 Reporting Results

A typical benchmark table might appear as

| Quantity | MAE | RMSE |
|-----------|----:|------:|
| Energy | 6.4 meV/atom | 8.2 meV/atom |
| Forces | 0.019 eV/Å | 0.027 eV/Å |
| Stress | 0.23 GPa | 0.35 GPa |

Such tables are routinely included in research publications.

---

# 22.13.27.17 Beyond Numerical Metrics

Although numerical metrics are important,

they do not tell the complete story.

Researchers also examine

- parity plots,
- residual distributions,
- force correlation plots,
- error histograms,
- uncertainty estimates,
- outlier analysis.

These visual diagnostics often reveal systematic prediction errors that are not obvious from MAE or RMSE alone.

---

# 22.13.27.18 Best Practices

When validating a MACE model,

always

- ✔ evaluate only on an independent test set,
- ✔ report both MAE and RMSE,
- ✔ evaluate energies, forces, and stresses separately,
- ✔ normalize energy errors per atom,
- ✔ report the units of every metric,
- ✔ compare with published benchmarks whenever possible,
- ✔ include visual diagnostic plots in addition to numerical metrics.

---

# 22.13.27.19 Summary

Validation is a critical stage in the development of machine learning interatomic potentials. By comparing MACE predictions against independent DFT calculations, researchers can quantify the accuracy and reliability of the model using metrics such as MAE, RMSE, and the coefficient of determination. Proper validation ensures that the trained potential is capable of generalizing beyond its training data and provides confidence that it can be safely used in atomistic simulations.

---

## Transition to Section 22.13.28

While numerical metrics summarize prediction accuracy, they cannot reveal the full relationship between machine learning predictions and DFT reference values. In the next section, we will learn how to construct and interpret **parity plots**, **force correlation plots**, **residual analysis**, and other visualization techniques that are standard for evaluating MACE and other machine learning interatomic potentials.

# 22.13.28 Visualization and Error Analysis

Numerical metrics such as MAE and RMSE provide concise summaries of model performance, but they do not reveal *how* the model makes mistakes. Two models may have identical MAE values while exhibiting completely different error patterns. One model may produce uniformly small errors, whereas another may perform exceptionally well for most structures but fail catastrophically for a few.

Therefore, modern machine learning research relies heavily on **visual diagnostics**. Visualization allows researchers to identify systematic biases, detect outliers, evaluate generalization, and understand whether prediction errors arise from specific regions of configuration space.

In virtually every publication introducing a new machine learning interatomic potential—including MACE, NequIP, Allegro, CHGNet, and DeepMD—visual analysis accompanies numerical metrics. In this section, we study the most important visualization techniques used to evaluate machine learning interatomic potentials.

---

# 22.13.28.1 Why Visualization Matters

Suppose two MACE models both report

```text
Energy MAE

=

8 meV/atom
```

Without additional analysis,

we cannot determine

- whether the errors are uniformly distributed,
- whether high-energy structures are predicted poorly,
- whether rare configurations dominate the error,
- whether the model systematically overestimates or underestimates energies.

Visualization answers these questions immediately.

---

# 22.13.28.2 Types of Diagnostic Plots

The most common evaluation plots include

- Energy parity plots,
- Force parity plots,
- Stress parity plots,
- Residual plots,
- Error histograms,
- Learning curves,
- Uncertainty plots.

Each provides different information about model behavior.

---

# 22.13.28.3 Energy Parity Plot

The energy parity plot is the most widely used evaluation figure.

Its purpose is to compare

- DFT energies,
- MACE-predicted energies.

The ideal parity plot is

```text
Predicted Energy

^

|

|                    •

|                 •

|              •

|           •

|        •

|     •

|  •

+---------------------------->

      DFT Energy
```

Every point lying exactly on the diagonal corresponds to a perfect prediction.

---

# 22.13.28.4 Mathematical Interpretation

Suppose

- $E_{\mathrm{DFT}}$ is the reference energy,
- $E_{\mathrm{ML}}$ is the predicted energy.

For perfect predictions,

$$
E_{\mathrm{ML}}
=
E_{\mathrm{DFT}}
$$

Therefore,

the ideal parity plot follows the line

$$
y=x.
$$

The closer the data lie to this line,

the more accurate the model.

---

# 22.13.28.5 Creating an Energy Parity Plot

Assume

```python
energy_true
```

contains DFT energies,

and

```python
energy_pred
```

contains MACE predictions.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(
    energy_true,
    energy_pred,
    s=12
)

minimum = min(
    energy_true.min(),
    energy_pred.min()
)

maximum = max(
    energy_true.max(),
    energy_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    linewidth=2
)

plt.xlabel("DFT Energy (eV)")

plt.ylabel("MACE Energy (eV)")

plt.title("Energy Parity Plot")

plt.show()
```

The dashed red line represents the ideal prediction.

---

# 22.13.28.6 Interpreting the Plot

Three common situations occur.

### Excellent Model

```text
Points tightly clustered
around the diagonal.
```

### Moderate Model

```text
Small random scatter
around the diagonal.
```

### Poor Model

```text
Large deviations

systematic offsets

clusters far from
the diagonal.
```

---

# 22.13.28.7 Force Parity Plot

Forces are usually more numerous than energies.

Every atom contributes

three force components

$$
F_x,\;
F_y,\;
F_z.
$$

The parity plot compares

- DFT forces,
- predicted forces.

---

# 22.13.28.8 Creating a Force Parity Plot

Flatten all force components.

```python
plt.figure(figsize=(6,6))

plt.scatter(

    forces_true.flatten(),

    forces_pred.flatten(),

    s=3

)

minimum = min(

    forces_true.min(),

    forces_pred.min()

)

maximum = max(

    forces_true.max(),

    forces_pred.max()

)

plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    "r--"

)

plt.xlabel("DFT Force (eV/Å)")

plt.ylabel("MACE Force (eV/Å)")

plt.show()
```

Force parity plots often contain millions of points.

---

# 22.13.28.9 Stress Parity Plot

Stress tensors are evaluated similarly.

Each component is compared against the DFT reference.

This is particularly important when the model will be used for

- lattice optimization,
- NPT Molecular Dynamics,
- elastic constant calculations.

---

# 22.13.28.10 Residual Plot

Parity plots show agreement.

Residual plots show the actual prediction errors.

The residual is

$$
r_i
=
y_i
-
\hat y_i.
$$

For energies,

$$
r_i
=
E_{\mathrm{DFT}}
-
E_{\mathrm{ML}}.
$$

---

# 22.13.28.11 Creating a Residual Plot

```python
residual = (

    energy_true -

    energy_pred

)

plt.figure(figsize=(7,5))

plt.scatter(

    energy_true,

    residual,

    s=12

)

plt.axhline(

    0,

    color="red",

    linestyle="--"

)

plt.xlabel("DFT Energy")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.show()
```

---

# 22.13.28.12 Interpreting Residuals

An ideal residual plot

```text
Residual

^

|

|  • • •

| • • •

|••••••••••

| • • •

|  • •

+-------------------->

        Energy
```

Characteristics

- centered around zero,
- no visible pattern,
- approximately constant variance.

Any systematic trend suggests model bias.

---

# 22.13.28.13 Error Histogram

Another useful visualization is the error histogram.

It shows the distribution of prediction errors.

```python
errors = (

    energy_true -

    energy_pred

)

plt.hist(

    errors,

    bins=50

)

plt.xlabel("Prediction Error")

plt.ylabel("Count")

plt.show()
```

---

# 22.13.28.14 Interpreting Histograms

A desirable histogram is

- symmetric,
- centered at zero,
- narrow.

A skewed histogram indicates

systematic overprediction

or

systematic underprediction.

Wide distributions indicate poor model accuracy.

---

# 22.13.28.15 Learning Curves

Learning curves monitor

training error

and

validation error

throughout optimization.

Typical plot

```text
Loss

^

|\
| \
|  \

|   \_____

|        \__

+-------------------->

      Epoch
```

---

# 22.13.28.16 Interpreting Learning Curves

Ideal behavior

```text
Training Loss

↓

Validation Loss

↓

Both converge
```

Signs of overfitting

```text
Training Loss

↓

Validation Loss

↓

Validation suddenly rises
```

Signs of underfitting

```text
Training Loss

High

Validation Loss

High
```

---

# 22.13.28.17 Outlier Analysis

Occasionally,

certain structures exhibit unusually large prediction errors.

These are called

**outliers**.

Outliers often correspond to

- unusual bonding,
- rare chemical environments,
- highly distorted geometries,
- extrapolation beyond the training data.

Active learning frequently begins by selecting these structures.

---

# 22.13.28.18 Visualization Workflow

The complete evaluation pipeline becomes

```text
Test Dataset

↓

MACE Predictions

↓

MAE

↓

RMSE

↓

Parity Plot

↓

Residual Plot

↓

Histogram

↓

Outlier Analysis

↓

Model Assessment
```

This workflow is standard in modern machine learning publications.

---

# 22.13.28.19 Best Practices

When evaluating a MACE model,

always

- ✔ generate energy parity plots,
- ✔ generate force parity plots,
- ✔ examine residual distributions,
- ✔ inspect error histograms,
- ✔ monitor learning curves,
- ✔ investigate outliers,
- ✔ report numerical metrics together with visual diagnostics.

Visual analysis provides insight that numerical summaries alone cannot capture.

---

# 22.13.28.20 Summary

Visualization plays a central role in assessing the quality of machine learning interatomic potentials. Parity plots reveal the agreement between MACE predictions and DFT reference values, residual plots expose systematic prediction biases, histograms characterize the distribution of errors, and learning curves diagnose optimization behavior. Together with numerical metrics such as MAE and RMSE, these visual tools provide a comprehensive understanding of model performance and help identify opportunities for further improvement.

---

## Transition to Section 22.13.29

After validating and visualizing the performance of a MACE model, the final step is to understand how these models are used in real scientific research. In the next section, we will examine **practical applications of MACE**, including high-throughput materials screening, crystal structure optimization, battery materials, catalysis, defect physics, molecular dynamics, and large-scale atomistic simulations reported in the recent literature.

# 22.13.29 Scientific Applications of MACE

By this point, we have learned

- the theoretical foundations of MACE,
- its architecture,
- message passing,
- higher-order equivariant interactions,
- training procedures,
- inference,
- geometry optimization,
- molecular dynamics,
- active learning,
- validation,
- benchmarking.

The natural question is now

> **Where is MACE actually used in scientific research?**

The answer is simple:

**almost everywhere that atomistic simulations are performed.**

Since its introduction, MACE has rapidly become one of the most widely adopted machine learning interatomic potentials in computational materials science, chemistry, and condensed matter physics. Its combination of **high accuracy**, **excellent scalability**, and **equivariance** makes it suitable for a broad range of scientific problems that were previously accessible only through computationally expensive first-principles calculations.

In this section, we explore some of the most important research applications of MACE.

---

# 22.13.29.1 High-Throughput Materials Screening

One of the primary applications of MACE is **high-throughput screening**.

Instead of evaluating a few candidate materials,

researchers often wish to investigate

```text
10,000

100,000

or even

1,000,000

candidate structures.
```

Performing DFT calculations for such a large dataset would require enormous computational resources.

A trained MACE model enables rapid prediction of

- total energies,
- forces,
- structural stability,

for every candidate.

The workflow is

```text
Candidate Structures

↓

MACE Prediction

↓

Rank by Stability

↓

Select Best Candidates

↓

DFT Verification
```

Only the most promising materials are subsequently evaluated using DFT.

---

# 22.13.29.2 Crystal Structure Relaxation

Many crystal structures obtained from

- experimental databases,
- crystal structure prediction,
- generative models,

are not fully relaxed.

Using MACE,

thousands of crystal structures can be optimized efficiently.

```text
Initial Crystal

↓

MACE Relaxation

↓

Equilibrium Structure

↓

Property Calculation
```

Because structural relaxation is significantly faster than DFT,

large crystal databases can be processed within practical time limits.

---

# 22.13.29.3 Crystal Structure Prediction

Modern crystal structure prediction algorithms generate thousands of candidate structures.

Most of these candidates are unstable.

MACE serves as a rapid filter.

```text
Generated Structures

↓

MACE Energy

↓

Ranking

↓

Lowest-Energy Candidates

↓

DFT
```

This substantially reduces the number of expensive first-principles calculations.

---

# 22.13.29.4 Battery Materials

Machine learning interatomic potentials are particularly valuable for battery research.

Typical applications include

- lithium diffusion,
- sodium diffusion,
- solid electrolytes,
- electrode materials,
- phase stability,
- ionic conductivity.

For example,

a MACE potential can simulate lithium diffusion over hundreds of picoseconds,

whereas equivalent DFT simulations are often limited to only a few picoseconds.

---

# 22.13.29.5 Diffusion Studies

Diffusion determines

- battery charging speed,
- hydrogen transport,
- alloy homogenization,
- defect migration.

MACE enables long Molecular Dynamics simulations,

allowing researchers to observe diffusion events that would be inaccessible with DFT.

Typical workflow

```text
Relaxed Structure

↓

MACE Molecular Dynamics

↓

Atomic Trajectories

↓

Mean Square Displacement

↓

Diffusion Coefficient
```

---

# 22.13.29.6 Defect Physics

Real materials always contain defects.

Examples include

- vacancies,
- interstitials,
- substitutional atoms,
- dislocations,
- grain boundaries.

These defects often involve thousands of atoms.

Traditional DFT calculations become prohibitively expensive.

MACE enables

- defect formation energy calculations,
- defect migration,
- interaction between defects,
- large supercell simulations.

---

# 22.13.29.7 Surface Science

Catalytic reactions occur on surfaces rather than within bulk crystals.

Typical investigations involve

- adsorption,
- desorption,
- diffusion,
- reconstruction.

MACE allows rapid optimization of

surface slabs

containing hundreds or thousands of atoms.

---

# 22.13.29.8 Catalysis

Catalysis research frequently requires evaluating

thousands of adsorption geometries.

For example,

```text
Catalyst Surface

+

Adsorbate

↓

Geometry Optimization

↓

Adsorption Energy

↓

Reaction Pathway
```

Replacing DFT relaxations with MACE dramatically accelerates catalyst screening.

---

# 22.13.29.9 Grain Boundaries

Grain boundaries strongly influence

- strength,
- corrosion,
- diffusion,
- conductivity.

Large grain-boundary models often contain

```text
10,000+

atoms.
```

These systems are far beyond the practical limits of conventional DFT.

MACE enables atomistic simulations of these complex interfaces while maintaining near-DFT accuracy.

---

# 22.13.29.10 Fracture Mechanics

Mechanical failure originates from

- crack initiation,
- crack propagation,
- bond breaking.

Studying these processes requires large atomistic simulations.

Using MACE,

researchers can investigate

- crack growth,
- dislocation emission,
- fracture toughness,

at scales previously accessible only with empirical force fields.

---

# 22.13.29.11 Phase Transitions

Materials frequently undergo structural phase transitions.

Examples include

- melting,
- crystallization,
- martensitic transformations,
- polymorphic transitions.

These processes require

long Molecular Dynamics trajectories,

making MACE an ideal computational tool.

---

# 22.13.29.12 Amorphous Materials

Unlike crystals,

amorphous materials possess

no long-range order.

Examples include

- glasses,
- amorphous silicon,
- amorphous oxides,
- polymer networks.

Because of their structural complexity,

large simulation cells are essential.

Machine learning potentials enable realistic simulations of these disordered systems.

---

# 22.13.29.13 Mechanical Properties

Mechanical behavior can be investigated by

applying strain.

```text
Crystal

↓

Apply Strain

↓

MACE Relaxation

↓

Stress

↓

Elastic Constants
```

Such calculations are used to determine

- Young's modulus,
- shear modulus,
- bulk modulus,
- Poisson ratio.

---

# 22.13.29.14 Thermal Properties

Thermal simulations using MACE include

- thermal expansion,
- heat capacity,
- phonon dynamics,
- lattice vibrations,
- thermal conductivity.

Long Molecular Dynamics trajectories significantly improve statistical accuracy.

---

# 22.13.29.15 Large-Scale Molecular Dynamics

One of MACE's greatest strengths is scalability.

Typical simulation sizes are

| Method | Typical Number of Atoms |
|---------|------------------------:|
| DFT | 100–500 |
| AIMD | 100–300 |
| Classical Force Field | Millions |
| MACE | Thousands to Millions* |

\*The exact system size depends on the available GPU memory and model complexity.

This enables simulations over length and time scales that bridge the gap between first-principles methods and classical force fields.

---

# 22.13.29.16 Integration with Active Learning

MACE is frequently combined with active learning.

```text
Initial Dataset

↓

Train MACE

↓

Run MD

↓

Detect Uncertain Structures

↓

DFT

↓

Retrain

↓

Improved MACE
```

This iterative workflow has become standard practice for constructing highly accurate machine learning potentials.

---

# 22.13.29.17 Scientific Software Ecosystem

MACE integrates naturally with many widely used computational materials science tools.

Examples include

- ASE,
- Quantum ESPRESSO,
- VASP,
- LAMMPS,
- OpenMM,
- OVITO,
- pymatgen,
- matminer.

This interoperability allows researchers to incorporate MACE into existing simulation workflows with minimal modification.

---

# 22.13.29.18 Advantages Over Classical Force Fields

Compared with traditional empirical force fields,

MACE provides

- DFT-quality accuracy,
- transferable chemical environments,
- improved treatment of bond breaking and formation,
- support for multiple chemical species,
- systematic improvement through active learning.

These advantages explain its rapidly growing adoption across materials science and chemistry.

---

# 22.13.29.19 Current Research Directions

Research involving MACE continues to expand rapidly.

Active areas include

- foundation interatomic models,
- multi-element universal potentials,
- reactive simulations,
- uncertainty-aware machine learning potentials,
- integration with diffusion models,
- autonomous laboratories,
- robotic materials discovery,
- exascale atomistic simulations.

These developments are expected to define the next generation of computational materials research.

---

# 22.13.29.20 Summary

MACE has emerged as one of the leading machine learning interatomic potentials for atomistic simulation. Its combination of equivariant message passing, near-DFT accuracy, and computational efficiency enables applications ranging from crystal structure optimization and molecular dynamics to battery materials, catalysis, defect physics, grain boundary simulations, fracture mechanics, and high-throughput materials discovery. By integrating seamlessly with established software such as ASE and supporting active learning workflows, MACE provides a versatile platform for modern computational materials science and is increasingly becoming a standard tool in both academic research and industrial applications.

---

# 22.13.29.21 Chapter Summary

Throughout **Chapter 22**, we explored the development of modern equivariant machine learning interatomic potentials, beginning with the theoretical foundations established by NequIP and culminating in the advanced capabilities of MACE and Allegro. We examined the mathematical principles of equivariance, higher-order message passing, local descriptors, tensor-product interactions, and linear-scaling architectures. We then studied the complete practical workflow, including dataset preparation, model training, inference, ASE integration, geometry optimization, molecular dynamics, active learning, validation, benchmarking, visualization, and real-world scientific applications.

The concepts introduced in this chapter provide the foundation for the next stage of atomistic machine learning: **large-scale foundation models and universal interatomic potentials**.

---

## Transition to Chapter 23

The previous chapters focused on predicting material properties and atomic interactions using supervised machine learning models. However, a fundamental challenge in materials discovery remains: **how can we design entirely new materials rather than merely predict the properties of existing ones?**

In the next chapter, we move beyond predictive models to **Generative Artificial Intelligence for Materials Discovery**, where machine learning is used to create novel crystal structures, molecules, and materials with targeted properties. We will study Variational Autoencoders (VAEs), Graph VAEs, Diffusion Models, Crystal Diffusion, Score-Based Generative Models, Generative Adversarial Networks (GANs), property-conditioned generation, and emerging text-guided materials design systems that are reshaping the future of computational materials science.
