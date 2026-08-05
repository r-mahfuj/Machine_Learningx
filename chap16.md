Perfect. Since this is a new chapter and our book is highly detailed, we'll begin with **Chapter 16.1** and build it systematically, just as we did for MEGNet.

---

# Chapter 16 — SchNet: Continuous-Filter Neural Networks for Atomistic Systems

# 16.1 Introduction to SchNet

Deep learning has transformed many areas of science by enabling computers to automatically learn complex patterns directly from data. In materials science and computational chemistry, however, developing effective neural network architectures has historically been far more challenging than in fields such as computer vision or natural language processing.

Unlike images, which are arranged on regular two-dimensional grids, or sentences, which are organized as ordered sequences of words, molecules and crystalline materials possess irregular three-dimensional atomic structures. Every material contains a different number of atoms, different chemical elements, varying bond lengths, and diverse local environments. These characteristics make conventional neural network architectures unsuitable for atomistic systems.

Early machine learning methods attempted to overcome this problem by designing handcrafted descriptors that summarized the structure of a material. Examples include Coulomb matrices, Bag-of-Bonds, SOAP descriptors, Atom-Centered Symmetry Functions (ACSFs), and descriptors generated using packages such as Matminer. These descriptors enabled classical machine learning algorithms such as Random Forests, Support Vector Machines, and Gaussian Process Regression to predict material properties with reasonable accuracy.

However, handcrafted descriptors introduced significant limitations. They required extensive domain expertise, often failed to generalize across different chemical systems, and could omit important structural information. More importantly, the success of the machine learning model became heavily dependent on the quality of the manually engineered features rather than on the learning capability of the neural network itself.

The emergence of Graph Neural Networks (GNNs) represented a major breakthrough. Instead of describing materials using handcrafted descriptors, atoms were represented as graph nodes and chemical bonds as graph edges. Models such as **CGCNN** and **MEGNet** demonstrated that graph-based neural networks could automatically learn meaningful atomic representations directly from crystal structures.

Although these graph-based approaches significantly advanced materials informatics, they still possessed important limitations. Most graph neural networks require the construction of an explicit graph before learning begins. Creating such graphs involves choosing

* cutoff radii,
* neighbor definitions,
* edge construction rules,
* bonding criteria.

These design choices influence the resulting graph and may affect prediction accuracy. Furthermore, representing atomic interactions as discrete graph edges does not naturally capture the continuous nature of interatomic distances. In reality, moving an atom by a small amount changes the interaction strength continuously rather than causing abrupt changes.

These limitations motivated the development of **SchNet**, one of the most influential neural network architectures for atomistic systems.

Unlike previous graph neural networks, SchNet does not rely on manually designed edge features or predefined bond types. Instead, it learns interactions directly from the continuous three-dimensional coordinates of atoms. Rather than asking whether two atoms are connected by a bond, SchNet asks a more physically meaningful question:

> **How does the interaction between two atoms change continuously as the distance between them changes?**

This seemingly simple idea fundamentally changed how neural networks model molecules and materials.

Instead of representing interactions using discrete graph edges,

SchNet introduces **continuous-filter convolutions**, allowing the neural network to learn interaction functions that vary smoothly with interatomic distance.

This innovation provides several important advantages.

First, atomic interactions become continuous rather than discrete.

Second, the model naturally respects the geometric structure of molecules and crystals.

Third, no handcrafted chemical descriptors are required.

Finally, the model learns both atomic representations and interaction functions simultaneously through end-to-end optimization.

---

# The Historical Development of SchNet

SchNet was introduced in 2017 by

**Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller**

in the landmark paper

> **SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions**

The primary objective of SchNet was to develop a neural network capable of learning quantum mechanical properties directly from atomic coordinates.

Unlike traditional graph neural networks that primarily focused on property prediction, SchNet was specifically designed for atomistic simulations.

The original model demonstrated state-of-the-art performance on several benchmark datasets, including

* QM9
* MD17

and later became the foundation for many subsequent architectures, including

* SchNetPack
* PhysNet
* PaiNN
* DimeNet
* GemNet
* SpookyNet
* NequIP
* Allegro
* MACE

Consequently, SchNet occupies a central position in the evolution of modern atomistic machine learning.

---

# Why Is SchNet Different?

To understand the importance of SchNet, consider two neighboring carbon atoms.

Traditional graph neural networks represent their interaction as

```text
Carbon —— Carbon
```

The edge simply indicates that the atoms are neighbors.

However, this representation ignores an important physical fact.

Suppose the distance between the atoms changes slightly.

```text
1.40 Å

↓

1.41 Å

↓

1.42 Å
```

The interaction between the atoms also changes continuously.

Chemical bonding is not binary.

There is no sudden transition between

"bond"

and

"no bond."

Instead,

interaction strength changes smoothly with atomic distance.

SchNet models precisely this continuous behavior.

Instead of assigning a fixed edge feature,

it learns a continuous function

```text
Distance

↓

Neural Network

↓

Interaction Strength
```

As atoms move,

their interactions evolve smoothly,

making SchNet particularly suitable for molecular dynamics and energy prediction.

---

# Core Philosophy of SchNet

The philosophy underlying SchNet can be summarized in four principles.

### 1. Learn Directly from Atomic Coordinates

Rather than relying on handcrafted structural descriptors,

SchNet uses only

* atomic numbers,
* atomic coordinates.

Everything else is learned automatically.

---

### 2. Continuous Interaction Functions

Instead of discrete graph convolutions,

SchNet learns continuous interaction filters that depend on interatomic distance.

This enables much more realistic modeling of atomic interactions.

---

### 3. End-to-End Learning

Every component of the model,

including

* atomic embeddings,
* interaction functions,
* distance representations,
* prediction layers,

is optimized simultaneously during training.

No manually engineered chemical knowledge is required.

---

### 4. Physics-Inspired Design

SchNet incorporates several ideas inspired by physics.

For example,

* interactions depend on distance,
* energy is predicted as a sum of atomic contributions,
* forces are obtained by differentiating the predicted energy,
* the model naturally satisfies permutation invariance,
* translational invariance is built into the architecture.

These properties make SchNet particularly well suited for atomistic simulations.

---

# Typical Workflow of SchNet

A high-level overview of SchNet is shown below.

```text
Atomic Numbers

+

Atomic Coordinates

↓

Neighbor Search

↓

Distance Calculation

↓

Radial Basis Expansion

↓

Continuous Filter Network

↓

Interaction Blocks

↓

Updated Atomic Representations

↓

Atom-wise Energy Prediction

↓

Total Molecular/Crystal Energy

↓

Automatic Differentiation

↓

Atomic Forces
```

Unlike conventional neural networks,

every stage of SchNet is designed around the physical structure of atoms rather than abstract feature vectors.

---

# Properties Predicted by SchNet

SchNet has been successfully applied to a wide range of atomistic prediction tasks.

For molecules,

it predicts

* total energy,
* atomization energy,
* dipole moment,
* polarizability,
* HOMO energy,
* LUMO energy,
* HOMO–LUMO gap,
* heat capacity,
* vibrational properties.

For crystalline materials,

SchNet has been used to predict

* formation energy,
* band gap,
* elastic constants,
* bulk modulus,
* adsorption energy,
* defect formation energy,
* magnetic properties,
* lattice energy.

Because forces can be obtained through automatic differentiation,

SchNet also serves as a neural-network interatomic potential for molecular dynamics simulations.

---

# Why SchNet Matters in Materials Informatics

SchNet represents a major conceptual shift in machine learning for atomistic systems.

Earlier approaches relied heavily on manually engineered descriptors or explicitly defined graph structures.

SchNet demonstrated that a neural network could instead learn directly from the continuous geometry of atoms, using only atomic identities and spatial coordinates.

This idea has profoundly influenced the development of modern atomistic machine learning.

Many state-of-the-art architectures introduced after SchNet—including PaiNN, DimeNet, GemNet, MACE, and NequIP—retain the central philosophy established by SchNet while extending it with additional physical information such as angular interactions, equivariant message passing, and higher-order geometric representations.

Consequently, understanding SchNet is essential not only for mastering one specific architecture but also for understanding the design principles underlying nearly all modern neural interatomic potentials.

---

# What You Will Learn in This Chapter

By the end of this chapter, you will be able to:

* Explain why continuous-filter convolutions are superior to traditional graph convolutions for atomistic systems.
* Understand the mathematical foundations of SchNet.
* Construct radial basis function distance embeddings.
* Implement continuous interaction blocks from scratch in PyTorch.
* Predict molecular and crystal energies using SchNet.
* Compute atomic forces through automatic differentiation.
* Train SchNet on datasets such as QM9, MD17, and Materials Project.
* Optimize SchNet for research-scale applications.
* Interpret learned atomic representations and interaction filters.
* Build a complete research-grade SchNet implementation suitable for modern materials informatics studies.

---

# Chapter Roadmap

This chapter is organized to progress from fundamental concepts to a complete research implementation.

We begin by examining **why continuous geometric representations are necessary** for atomistic learning and why conventional graph neural networks are insufficient for modeling quantum interactions.

Next, we develop the mathematical foundations of continuous-filter convolutions, radial basis function expansions, interaction blocks, and energy-based learning.

Building upon these concepts, we implement every component of SchNet from scratch using **PyTorch**, carefully explaining tensor dimensions, computational flow, and optimization strategies.

Finally, we extend the implementation to predict both **energies and forces**, train the model on real-world datasets, analyze its learned representations, compare it with modern atomistic architectures, and explore practical research applications in computational materials science and molecular machine learning.

This journey will transform SchNet from a research paper into a model that you can understand mathematically, implement independently, modify for new scientific problems, and use confidently in your own materials informatics research.

# 16.2 Motivation: Why Was SchNet Developed?

Every successful scientific model begins with a problem that existing methods cannot solve satisfactorily. SchNet was not developed simply to introduce another graph neural network. It was created to address several fundamental challenges that arose when applying deep learning to molecules and crystalline materials.

To appreciate the importance of SchNet, we must first understand why traditional machine learning methods—and even early graph neural networks—were insufficient for accurately modeling atomistic systems.

---

# The Fundamental Challenge of Atomistic Learning

Unlike images or text, atomistic systems exist in **continuous three-dimensional space**.

Consider a water molecule.

```text id="mot01"
      H
       \
        O
       /
      H
```

The identity of the molecule depends not only on the atoms present but also on

* their three-dimensional positions,
* bond lengths,
* bond angles,
* relative orientation.

If one hydrogen atom moves slightly,

```text id="mot02"
Original Position

↓

Small Displacement

↓

New Position
```

the molecular energy changes smoothly.

There is no abrupt jump in physical properties.

This continuous behavior is one of the defining characteristics of atomic systems.

---

# Why Images Are Easier Than Molecules

Convolutional Neural Networks (CNNs) revolutionized computer vision because images possess a highly regular structure.

Every pixel occupies a fixed position in a rectangular grid.

For example,

```text id="mot03"
□ □ □ □

□ □ □ □

□ □ □ □
```

Each pixel always has neighboring pixels in predictable locations.

Consequently,

the same convolution kernel can slide across the image,

detecting patterns efficiently.

Atoms, however, are fundamentally different.

```text id="mot04"
Atom A

      Atom B


          Atom C

   Atom D
```

There is no regular grid.

Atoms may appear anywhere in three-dimensional space.

The number of neighboring atoms also varies from one atom to another.

Therefore,

standard convolution operations cannot be applied directly.

---

# Why Fully Connected Networks Fail

One might attempt to represent atomic coordinates using a simple vector.

For example,

```text id="mot05"
[x₁, y₁, z₁,

 x₂, y₂, z₂,

 x₃, y₃, z₃$$
```

A fully connected neural network could then process this vector.

Unfortunately,

this approach suffers from several major problems.

---

## Problem 1 — Variable Number of Atoms

Different molecules contain different numbers of atoms.

Water contains

```text id="mot06"
3 atoms
```

Methane contains

```text id="mot07"
5 atoms
```

Benzene contains

```text id="mot08"
12 atoms
```

Proteins may contain

```text id="mot09"
Thousands of atoms
```

A fixed-length input vector cannot naturally accommodate molecules of arbitrary size.

---

## Problem 2 — Atom Ordering

Suppose we describe methane.

One possible ordering is

```text id="mot10"
C H H H H
```

Another equally valid ordering is

```text id="mot11"
H H C H H
```

The molecule remains identical.

However,

a conventional neural network would receive completely different input vectors.

Ideally,

the prediction should remain unchanged regardless of how atoms are indexed.

This requirement is known as **permutation invariance**.

---

## Problem 3 — Translation

Suppose an entire molecule moves

10 Å to the right.

```text id="mot12"
Original Molecule

↓

Translate Entire Molecule

↓

Same Molecule
```

Its energy,

band gap,

and chemical identity remain unchanged.

Therefore,

the model should ignore absolute position.

This property is called **translational invariance**.

---

## Problem 4 — Rotation

Now rotate the molecule.

```text id="mot13"
Original Orientation

↓

Rotate 90°

↓

Same Molecule
```

Again,

all physical properties remain identical.

Consequently,

the model must also satisfy **rotational invariance**.

---

# Why Handcrafted Descriptors Were Not Enough

Before deep learning,

researchers designed descriptors that attempted to summarize molecular geometry.

Popular examples include

* Coulomb Matrix
* SOAP
* ACSF
* Bag of Bonds
* MBTR

These descriptors enabled conventional machine learning algorithms to achieve impressive accuracy.

However,

they suffered from several drawbacks.

---

## Manual Feature Engineering

Every descriptor required

careful mathematical design.

Researchers had to decide

which structural information was important.

Unfortunately,

important information could be omitted unintentionally.

---

## Limited Generalization

A descriptor designed for

organic molecules

might perform poorly on

metallic alloys.

Similarly,

a descriptor optimized for crystals

might not generalize well to isolated molecules.

---

## Increasing Complexity

As datasets grew larger,

descriptors became increasingly complicated.

Some contained

hundreds

or even

thousands

of manually engineered features.

This increased computational cost

while reducing interpretability.

---

# Why Early Graph Neural Networks Were Still Limited

Graph neural networks represented an enormous improvement.

Atoms became nodes.

Chemical interactions became edges.

```text id="mot14"
Atom

↓

Node

Bond

↓

Edge
```

Models such as CGCNN and MEGNet learned directly from crystal graphs.

Nevertheless,

these models still relied on an important assumption.

The graph had to be constructed before learning began.

---

# The Problem with Explicit Graph Construction

Consider two atoms separated by

2.99 Å.

Suppose the cutoff radius is

3.00 Å.

These atoms become neighbors.

Now move one atom slightly.

The distance becomes

3.01 Å.

```text id="mot15"
2.99 Å

↓

Neighbor

3.01 Å

↓

Not Neighbor
```

A movement of only

0.02 Å

causes the graph structure to change abruptly.

Physically,

atomic interactions do not change this way.

Real interactions vary continuously.

The graph,

however,

changes discontinuously.

This mismatch motivated a new approach.

---

# Why Bond Definitions Are Ambiguous

In many materials,

there is no universally accepted definition of a bond.

Consider sodium chloride.

Should Na–Cl be considered

a bond,

or merely neighboring ions?

What about metallic bonding?

Hydrogen bonding?

Van der Waals interactions?

Different graph construction methods may produce

different graphs

for exactly the same crystal.

Consequently,

the neural network becomes dependent on arbitrary preprocessing choices.

---

# The Continuous Nature of Atomic Interactions

In quantum mechanics,

interatomic interactions depend primarily on

distance.

As two atoms move,

their interaction strength changes smoothly.

Conceptually,

```text id="mot16"
Distance

↓

Strong Interaction

↓

Medium Interaction

↓

Weak Interaction
```

There is no abrupt transition.

Instead,

the interaction function resembles

a continuous curve.

Traditional graph convolutions,

however,

treat neighboring atoms almost identically,

regardless of subtle distance variations.

SchNet was designed specifically to model these continuous interaction functions.

---

# The Core Idea Behind SchNet

Instead of asking

> Are these two atoms connected?

SchNet asks

> How strongly should these two atoms interact at this exact distance?

Rather than assigning

fixed edge weights,

SchNet learns

continuous filters

whose values depend on

interatomic distance.

Conceptually,

```text id="mot17"
Atomic Distance

↓

Radial Basis Expansion

↓

Filter Network

↓

Continuous Interaction Weight
```

As the distance changes,

the interaction weight changes smoothly.

This closely matches the behavior predicted by physics.

---

# Dynamic Neighborhoods

SchNet also treats neighborhoods differently.

Neighbor relationships are determined dynamically

using a cutoff radius,

but the interaction strength inside that cutoff is learned continuously.

Suppose three neighbors lie at

```text id="mot18"
1.8 Å

2.3 Å

3.4 Å
```

Rather than assigning identical edge importance,

SchNet learns

different interaction filters

for each distance.

Consequently,

closer atoms may contribute more strongly,

while distant neighbors contribute less.

The network learns this relationship directly from data.

---

# Why Coordinates Are More Fundamental Than Bonds

Chemical bonds are useful concepts,

but they are not fundamental quantities.

Atomic coordinates,

on the other hand,

are directly measured

or calculated

through

* X-ray diffraction,
* neutron diffraction,
* electron microscopy,
* molecular dynamics,
* density functional theory.

Given only atomic coordinates,

one can compute

* distances,
* bond angles,
* coordination numbers,
* local environments.

The reverse is generally impossible.

Therefore,

SchNet begins with the most fundamental structural information available:

```text id="mot19"
Atomic Numbers

+

Atomic Coordinates
```

Everything else is learned automatically.

---

# Physics Meets Deep Learning

SchNet represents one of the earliest successful attempts to combine

deep learning

with

physical principles.

Its design incorporates several important ideas.

* Interactions depend on interatomic distance.
* Atomic contributions combine to produce total energy.
* Forces are obtained as energy gradients.
* Predictions are invariant to translation and permutation.
* Continuous functions replace discrete edge weights.

These principles make SchNet considerably more suitable for atomistic simulations than conventional neural network architectures.

---

# Motivation in One Sentence

The motivation behind SchNet can be summarized as follows:

> **Instead of forcing atomistic systems into discrete graph representations with handcrafted interaction rules, SchNet learns continuous interaction functions directly from atomic coordinates, allowing neural networks to model quantum mechanical behavior in a physically meaningful and fully differentiable manner.**

---

# Summary

The development of SchNet was motivated by the unique challenges posed by atomistic systems. Conventional neural networks cannot naturally process molecules and crystals because these systems contain variable numbers of atoms, arbitrary ordering, continuous spatial coordinates, and geometric symmetries such as translation and rotation. Although handcrafted descriptors and early graph neural networks improved predictive performance, they still relied on manually designed features or discrete graph constructions that inadequately captured the continuous nature of interatomic interactions.

SchNet addressed these limitations by learning directly from atomic numbers and three-dimensional coordinates. Instead of representing interactions as fixed graph edges, it introduced **continuous-filter convolutions**, enabling interaction strengths to vary smoothly with interatomic distance. This design closely reflects the underlying physics of atomic systems and laid the foundation for a new generation of neural networks capable of predicting energies, forces, and other quantum mechanical properties directly from atomic geometry.

In the next section, **16.3 Mathematical Representation of Atomistic Systems**, we will develop the mathematical framework used by SchNet, introducing atomic coordinates, atomic numbers, neighbor lists, distance matrices, periodic boundary conditions, and tensor representations that form the input to the network.

# 16.3 Mathematical Representation of Atomistic Systems

Before we can understand how SchNet performs continuous-filter convolutions, we must first understand **how atoms, molecules, and crystals are represented mathematically**.

Unlike traditional machine learning algorithms that operate on feature vectors or images, SchNet works directly with the fundamental quantities that describe an atomistic system:

* atomic identities,
* three-dimensional coordinates,
* interatomic distances,
* neighboring atoms.

Every subsequent component of SchNet—including radial basis expansions, interaction blocks, continuous filters, and energy prediction—depends on these mathematical representations.

Therefore, mastering this section is essential for understanding both the theory and implementation of SchNet.

---

# From Matter to Mathematics

Consider a simple crystal.

```text id="math01"
Si

Si

Si

Si

```

A materials scientist immediately recognizes this as silicon.

However, a neural network cannot interpret chemical symbols directly.

The crystal must first be converted into mathematical objects.

SchNet represents every material using two fundamental pieces of information.

```text id="math02"
Atomic Numbers

+

Atomic Coordinates

```

Remarkably,

these two quantities alone are sufficient to reconstruct the complete atomic geometry.

Everything else,

including

* distances,
* neighbor relationships,
* local environments,

can be computed automatically.

---

# Atomic Numbers

Every atom is identified by its atomic number.

The atomic number specifies

the number of protons inside the nucleus.

For example,

| Element | Symbol | Atomic Number |
| --- | --- | --- |
| Hydrogen | H | 1 |
| Carbon | C | 6 |
| Nitrogen | N | 7 |
| Oxygen | O | 8 |
| Silicon | Si | 14 |
| Iron | Fe | 26 |

Instead of storing

```text id="math03"
Carbon

Oxygen

Hydrogen

```

SchNet stores

```text id="math04"
6

8

1

```

Mathematically,

the atomic numbers are represented as

$$\mathbf{Z} = (Z_1, Z_2, \ldots, Z_N),$$

where

* $N$ is the total number of atoms,
* $Z_i$ denotes the atomic number of atom $i$.

---

## Example

Water contains

```text id="math05"
O

H

H

```

Its atomic-number vector is

$$\mathbf{Z} = (8, 1, 1).$$

PyTorch implementation

```python
import torch

atomic_numbers = torch.tensor([8, 1, 1])

print(atomic_numbers)

```

Output

```text
tensor([8, 1, 1])

```

Notice that

the neural network never receives the chemical symbols.

It only receives integer atomic numbers.

---

# Three-Dimensional Atomic Coordinates

Atomic identities alone are insufficient.

Two materials containing identical atoms can exhibit completely different properties if the atoms occupy different positions.

Therefore,

SchNet also requires the Cartesian coordinates of every atom.

For atom $i$,

its position is

$$\mathbf{R}_i = (x_i, y_i, z_i).$$

The complete atomic configuration is

$$\mathbf{R} = (\mathbf{R}_1, \mathbf{R}_2, \ldots, \mathbf{R}_N).$$

---

## Coordinate Matrix

For $N$ atoms,

the coordinate matrix has dimensions

$$N \times 3.$$

For example,

a water molecule may be represented as

$$\mathbf{R} = \begin{bmatrix} 0.000 & 0.000 & 0.000 \\ 0.758 & 0.000 & 0.504 \\ -0.758 & 0.000 & 0.504 \end{bmatrix}.$$

Each row corresponds to one atom.

Each column represents

* x,
* y,
* z coordinates.

---

## PyTorch Representation

```python
coordinates = torch.tensor([
    [0.000, 0.000, 0.000],
    [0.758, 0.000, 0.504],
    [-0.758, 0.000, 0.504]
], dtype=torch.float32)

print(coordinates.shape)

```

Output

```text
torch.Size([3, 3])

```

The tensor contains

three atoms,

each described by

three coordinates.

---

# Why Coordinates Matter

Suppose we keep the same atoms

but slightly move one hydrogen atom.

```text id="math06"
Original Geometry

↓

Move One Atom

↓

New Geometry

```

The molecule remains water,

yet

* bond lengths,
* bond angles,
* potential energy,
* vibrational frequencies,

all change.

Therefore,

coordinates carry essential physical information.

---

# Coordinate Tensor Dimensions

Suppose a batch contains

32 molecules,

each containing

20 atoms.

The coordinate tensor has shape

```text
(32, 20, 3)

```

where

* 32 = batch size,
* 20 = number of atoms,
* 3 = Cartesian coordinates.

When molecules contain different numbers of atoms,

padding or specialized batching strategies are used.

We will study these techniques later in the chapter.

---

# Euclidean Distance Between Two Atoms

SchNet does not operate directly on coordinates.

Instead,

it computes

interatomic distances.

Consider atoms

$i$

and

$j$.

Their Euclidean distance is

$$r_{ij} = \left\vert{} \mathbf{R}_i - \mathbf{R}_j \right\vert{}.$$

Expanding the norm gives

$$r_{ij} = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2}.$$

This is one of the most important equations in SchNet.

Almost every interaction depends on

$r_{ij}$.

---

## Example

Suppose

Atom A

```text
(0,0,0)

```

Atom B

```text
(1,2,2)

```

Then

$$r = \sqrt{1^2+2^2+2^2} = 3.$$

---

## PyTorch Implementation

```python
import torch

r1 = torch.tensor([0.0, 0.0, 0.0])
r2 = torch.tensor([1.0, 2.0, 2.0])

distance = torch.norm(r1 - r2)

print(distance)

```

Output

```text
tensor(3.)

```

---

# Pairwise Distance Matrix

For

$N$

atoms,

SchNet computes

all pairwise distances.

The result is

an

$$N \times N$$

distance matrix.

For example,

```text id="math07"
Atom 1

Atom 2

Atom 3

```

produces

$$D = \begin{bmatrix} 0 & r_{12} & r_{13} \\ r_{21} & 0 & r_{23} \\ r_{31} & r_{32} & 0 \end{bmatrix}.$$

Important properties

* diagonal entries are zero,
* the matrix is symmetric,

$$r_{ij} = r_{ji}.$$

---

## Computing Distance Matrix in PyTorch

```python
coordinates = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])

distance_matrix = torch.cdist(coordinates, coordinates)

print(distance_matrix)

```

Output

```text
tensor([[0.0000, 1.0000, 1.0000],
        [1.0000, 0.0000, 1.4142],
        [1.0000, 1.4142, 0.0000]])

```

---

# Why Distances Instead of Coordinates?

One may wonder

why SchNet converts coordinates into distances.

The reason is simple.

Distances automatically satisfy several important physical symmetries.

Suppose an entire molecule moves

10 Å

to the right.

Coordinates change.

Distances do not.

Similarly,

rotating the entire molecule changes coordinates,

but leaves every interatomic distance unchanged.

Distances therefore provide

a geometry representation that is naturally invariant to translation and rotation.

This makes them an ideal input for learning atomic interactions.

---

# Neighbor Lists

Although the distance matrix contains all pairwise distances,

most atoms interact strongly only with nearby atoms.

Therefore,

SchNet constructs a **neighbor list** using a cutoff radius.

For atom $i$,

the neighbor set is

$$\mathcal{N}(i) = \{ j \mid r_{ij} < r_c \},$$

where

$r_c$

is the cutoff distance.

---

## Example

Suppose

```text id="math08"
Cutoff Radius

=

5 Å

```

Neighbor distances

| Neighbor | Distance |
| --- | --- |
| Atom B | 2.3 Å |
| Atom C | 3.7 Å |
| Atom D | 5.6 Å |

Only

Atoms B and C

become neighbors.

Atom D

is ignored.

This greatly reduces computational cost.

---

# Why Use Neighbor Lists?

Without a cutoff,

the number of pairwise interactions grows approximately as

$$N^2.$$

For

10,000 atoms,

this corresponds to

100 million

pairwise distances.

Most of these interactions are negligible.

Using neighbor lists reduces the computational complexity dramatically, making SchNet practical for large molecular and crystalline systems.

---

# Tensor Representation Used by SchNet

After preprocessing,

a material is represented by several tensors.

| Tensor | Shape | Description |
| --- | --- | --- |
| Atomic Numbers | $(N)$ | Element identities |
| Coordinates | $(N,3)$ | Cartesian positions |
| Neighbor Indices | $(E,2)$ | Neighbor pairs |
| Distances | $(E)$ | Interatomic distances |

Here,

$E$

denotes the total number of neighbor interactions after applying the cutoff.

These tensors form the input to the continuous-filter convolution layers.

---

# From Geometry to Learning

At this stage,

the neural network has not yet learned anything.

It simply possesses

* atomic identities,
* coordinates,
* neighbor relationships,
* distances.

The next challenge is to transform these raw geometric quantities into meaningful numerical features that a neural network can process efficiently.

Simply feeding raw distances into a neural network is often ineffective because small changes in distance can produce highly nonlinear changes in atomic interactions.

To address this problem, SchNet first converts distances into a richer representation using **Radial Basis Functions (RBFs)**.

Before studying RBFs in detail, however, we must understand the core mathematical operation that makes SchNet unique: the **continuous convolution**.

---

# Summary

SchNet represents atomistic systems using only two fundamental inputs: atomic numbers and three-dimensional Cartesian coordinates. Atomic numbers identify the chemical species, while coordinates describe the spatial arrangement of atoms. From these quantities, the model computes interatomic distances, constructs neighbor lists using a cutoff radius, and organizes the information into tensors suitable for deep learning.

Distances play a central role because they preserve the essential geometric relationships between atoms while remaining invariant to translations and rotations of the entire system. These geometric representations form the foundation upon which SchNet builds continuous interaction functions and ultimately predicts energies, forces, and other material properties.

In the next section, **16.4 From Crystal Graphs to Continuous Geometry**, we will examine how SchNet departs from conventional graph neural networks by replacing discrete graph convolutions with a continuous geometric representation, laying the conceptual foundation for **continuous-filter convolutions**, the defining innovation of the SchNet architecture.

# 16.4 From Crystal Graphs to Continuous Geometry

In the previous chapter, we studied **MEGNet**, where materials are represented as **graphs**. Each atom is treated as a node, neighboring atoms are connected by edges, and message passing is performed over these discrete graph connections. This representation has proven extremely successful and forms the foundation of many graph neural networks used in materials informatics.

However, despite their success, graph-based representations possess an important limitation. A graph is ultimately a **discrete approximation** of an atomistic system, whereas atoms exist in **continuous three-dimensional space**.

SchNet was developed to bridge this gap.

Instead of treating atomic interactions as fixed graph edges, SchNet models interactions as **continuous functions of interatomic distance**. This seemingly small change fundamentally alters how information is represented and propagated through the network.

Understanding this transition—from discrete crystal graphs to continuous geometry—is essential before studying continuous-filter convolutions.

---

# How CGCNN and MEGNet Represent Materials

Recall that a crystal graph consists of three fundamental components:

* Nodes (atoms)
* Edges (neighbor relationships)
* Global information (optional)

For example,

```text
      O
     / \
   Ti---O
     \
      O

```

can be represented mathematically as

$$G = (V,E),$$

where

* $V$ denotes the set of atoms (vertices),
* $E$ denotes the set of edges (neighbor connections).

Each atom possesses a feature vector

$$\mathbf{x}_i,$$

while each edge contains information describing the interaction between atoms.

During message passing,

information flows only through these predefined edges.

Conceptually,

```text
Atom

↓

Neighbor Edge

↓

Neighbor Atom

↓

Updated Features

```

This framework works remarkably well for many prediction tasks.

---

# The Hidden Assumption Behind Graph Neural Networks

Although graph neural networks appear to learn directly from crystal structures,

they actually rely on an important preprocessing step.

Before training begins,

the graph itself must be constructed.

This requires choosing

* a cutoff radius,
* neighbor-search algorithm,
* edge construction strategy,
* bond definitions.

For example,

suppose the cutoff radius is

$$r_c = 5 \, \text{\AA}.$$

Any two atoms satisfying

$$r_{ij} < 5 \, \text{\AA}$$

become neighbors.

Otherwise,

they are disconnected.

This creates a binary decision.

```text
Distance < Cutoff

↓

Connected

Distance > Cutoff

↓

Disconnected

```

---

# The Problem with Discrete Edges

Consider two atoms.

Initially,

their separation is

$$r_{ij} = 4.99 \, \text{\AA}.$$

Because

$$4.99 < 5.00,$$

the graph contains an edge.

Now imagine one atom moves only

$$0.02 \, \text{\AA}.$$

The distance becomes

$$r_{ij} = 5.01 \, \text{\AA}.$$

Suddenly,

the edge disappears.

Conceptually,

```text
4.99 Å

↓

Edge Exists

5.01 Å

↓

Edge Removed

```

From the perspective of graph construction,

the neighborhood changes abruptly.

From the perspective of physics,

almost nothing has changed.

This discontinuity is one of the major motivations behind SchNet.

---

# Real Atomic Interactions Are Continuous

In reality,

interatomic interactions vary smoothly with distance.

Suppose two atoms gradually move apart.

```text
1.5 Å

↓

2.0 Å

↓

2.5 Å

↓

3.0 Å

↓

3.5 Å

```

The interaction energy changes continuously.

There is no sudden transition where atoms instantly stop interacting.

Chemical bonding,

electrostatic interactions,

and van der Waals forces all decay smoothly as the distance increases.

A physically realistic neural network should therefore model interactions in the same manner.

---

# From Binary Connections to Continuous Interactions

Traditional graph neural networks ask a simple question:

> Are these two atoms connected?

SchNet asks a different question:

> How strong is the interaction between these two atoms at this exact distance?

Instead of representing interactions using binary graph edges,

SchNet learns a continuous function

$$f(r_{ij}),$$

where

* $r_{ij}$ is the interatomic distance,
* $f(\cdot)$ is a neural network.

As atoms move,

the interaction strength changes smoothly.

```text
Distance

↓

Continuous Function

↓

Interaction Strength

```

This idea lies at the heart of SchNet.

---

# Geometry Becomes the Primary Representation

Instead of relying primarily on graph connectivity,

SchNet begins with atomic geometry.

Every atom is described by

* atomic number,
* Cartesian coordinates.

From these quantities,

the network computes

* distances,
* neighboring atoms,
* interaction strengths.

Notice the important distinction.

Graph neural networks begin with

```text
Graph

↓

Learning

```

SchNet begins with

```text
Geometry

↓

Continuous Functions

↓

Learning

```

The graph is no longer the central object.

Instead,

geometry itself becomes the representation.

---

# Dynamic Interactions

Another advantage of SchNet is that interaction strengths are **dynamic**.

Suppose atom $i$ has three neighbors.

Their distances are

$$1.8 \, \text{\AA},$$

$$2.6 \, \text{\AA},$$

and

$$4.3 \, \text{\AA}.$$

A conventional graph convolution often treats all three neighbors similarly,

because each is connected by an edge.

SchNet,

however,

learns three different interaction filters.

Conceptually,

```text
1.8 Å

↓

Strong Filter

2.6 Å

↓

Medium Filter

4.3 Å

↓

Weak Filter

```

The interaction depends continuously on distance rather than merely on connectivity.

---

# Neighbor Lists Still Exist

At this point,

it might appear that SchNet completely eliminates graphs.

This is not entirely true.

SchNet still uses a **neighbor list** to reduce computational cost.

Only atoms within a cutoff radius participate in the interaction calculation.

Mathematically,

the neighbor set is

$$\mathcal{N}(i) = \left\{ j \mid r_{ij} < r_c \right\}.$$

However,

the crucial difference is that

the **interaction strength inside the cutoff is learned continuously**.

The cutoff merely limits computation;

it does not define the interaction itself.

---

# Comparing Graph-Based and Continuous Representations

| Graph Neural Networks | SchNet |
| --- | --- |
| Discrete graph representation | Continuous geometric representation |
| Edges represent interactions | Distances determine interactions |
| Binary neighbor relationships | Continuous interaction functions |
| Edge features are predefined or learned | Interaction filters are generated from distances |
| Graph connectivity drives learning | Geometry drives learning |

This comparison highlights the conceptual shift introduced by SchNet.

---

# Why Continuous Geometry Improves Learning

Continuous geometric representations offer several important advantages.

### Smooth Physical Behavior

Small atomic movements produce small changes in interaction strength,

which is consistent with physical laws.

---

### Better Force Prediction

Because interactions vary continuously,

the predicted energy is also continuous.

This allows atomic forces to be computed as

$$\mathbf{F} = - \nabla E,$$

using automatic differentiation.

We will study this concept in detail later in the chapter.

---

### More Accurate Quantum Modeling

Quantum mechanical interactions depend strongly on

* distance,
* local environment,
* atomic identity.

Continuous interaction filters capture these dependencies far more naturally than binary graph edges.

---

### Reduced Dependence on Handcrafted Features

Traditional graph neural networks often require manually designed edge features,

such as

* bond length,
* bond type,
* coordination number.

SchNet learns interaction functions directly from distances,

greatly reducing the need for feature engineering.

---

# An Intuitive Analogy

Imagine standing in a room with several people.

A graph-based representation asks only

> "Who is close enough to count as your neighbor?"

Everyone inside the cutoff is treated as connected.

SchNet instead asks

> "How strongly does each person's voice reach you?"

Someone standing one meter away is heard much more clearly than someone standing four meters away,

even though both are inside the room.

Interaction strength changes continuously with distance,

just as sound intensity gradually decreases.

This analogy closely mirrors how SchNet models atomic interactions.

---

# Preparing for Continuous-Filter Convolutions

We have now established the central idea behind SchNet.

Instead of propagating information through discrete graph edges,

the model propagates information using **continuous interaction functions** that depend on interatomic distances.

The next challenge is to determine **how these continuous interaction functions are learned**.

Simply feeding raw distances into a neural network is not sufficient.

Distances must first be transformed into a richer representation that allows the network to learn complex, nonlinear interaction patterns efficiently.

This transformation leads directly to one of the defining innovations of SchNet:

**Continuous-Filter Convolutions (CFConv).**

---

# Summary

Traditional graph neural networks represent atomistic systems using discrete graphs, where atoms are connected through predefined edges determined by neighbor-search algorithms and cutoff radii. Although effective, this representation introduces discontinuities because small changes in atomic positions can abruptly alter graph connectivity.

SchNet addresses this limitation by replacing discrete edge-based interactions with continuous functions of interatomic distance. Rather than asking whether two atoms are connected, SchNet learns **how strongly they interact** at every distance within the cutoff radius. This continuous geometric representation more closely reflects the underlying physics of molecular and crystalline systems, enabling smoother energy surfaces, accurate force prediction, and more realistic modeling of quantum mechanical interactions.

In the next section, **16.5 Continuous Convolution**, we will derive the mathematical foundations of continuous convolution and explain why traditional convolutional neural networks cannot be directly applied to atomistic systems, ultimately leading to the development of SchNet's continuous-filter convolution operator.

# 16.5 Continuous Convolution

One of the defining innovations of **SchNet** is the introduction of **continuous convolution**. This idea fundamentally changes how convolution is performed in neural networks operating on atomistic systems.

Traditional convolutional neural networks were designed for data that lie on **regular grids**, such as images. Atoms, however, do not occupy fixed grid locations. They are distributed irregularly in three-dimensional space, and their positions vary continuously. Consequently, the standard convolution operation used in computer vision cannot be applied directly to molecules or crystals.

SchNet overcomes this challenge by replacing discrete convolution kernels with **continuous filter functions** that depend on the spatial positions of atoms.

In this section, we will develop the mathematical intuition behind continuous convolution, understand why conventional convolutions fail for atomistic systems, and prepare the foundation for the **Continuous-Filter Convolution (CFConv)** introduced in the next section.

---

# Revisiting Classical Convolution

Before introducing continuous convolution, let us briefly review how convolution works in a conventional CNN.

Consider a one-dimensional signal

$$x=[x_1,x_2,x_3,x_4,x_5].$$

Suppose we use a convolution kernel

$$w=[w_1,w_2,w_3].$$

The convolution output at position $i$ is

$$y_i=\sum_{k=-1}^{1}w_kx_{i+k}.$$

Each output value is computed by multiplying neighboring input values with **fixed kernel weights**.

The important observation is that the kernel weights

$$w_1, \; w_2, \; w_3$$

are **constant**.

The same kernel is applied everywhere in the signal.

---

## Two-Dimensional Image Convolution

For images, the idea is identical.

Suppose the kernel is

$$K = \begin{bmatrix} k_{11} & k_{12} & k_{13} \\ k_{21} & k_{22} & k_{23} \\ k_{31} & k_{32} & k_{33} \end{bmatrix}.$$

The kernel slides across the image.

```text
Image

↓

3×3 Kernel

↓

Feature Map

```

Because every image pixel lies on a regular grid,

the same kernel can be reused at every location.

This property is known as **weight sharing**.

---

# Why Classical Convolution Works for Images

Image pixels satisfy several important assumptions.

1. Pixels lie on a regular rectangular grid.
2. Every pixel has neighbors in predictable locations.
3. Neighbor spacing is constant.

For example,

```text
□ □ □ □

□ □ □ □

□ □ □ □

```

Every horizontal neighbor is exactly one pixel away.

Every vertical neighbor is exactly one pixel away.

Therefore,

one fixed convolution kernel can successfully process the entire image.

---

# Why Atoms Are Different

Atoms do not satisfy these assumptions.

Consider four atoms.

```text
      A

 B

           C

      D

```

Notice several important differences.

* There is no regular grid.
* Neighbor spacing varies.
* Every atom has a different local environment.
* Coordinates are continuous.

The distance between

A and B

may be

$$1.42 \, \text{\AA},$$

while

A and C

may be

$$2.87 \, \text{\AA}.$$

Unlike image pixels,

neighbor positions are not fixed.

Consequently,

a fixed convolution kernel is no longer meaningful.

---

# The Failure of Fixed Convolution Kernels

Suppose we attempted to apply a CNN kernel directly to atomic coordinates.

Which kernel weight should correspond to

an atom located

$$1.37 \, \text{\AA}$$

away?

What about

$$1.92 \, \text{\AA}?$$

Or

$$2.64 \, \text{\AA}?$$

There is no predefined grid cell for these distances.

The conventional convolution operator has no natural way to process arbitrary atomic positions.

Therefore,

the kernel itself must become a **function of position**.

---

# The Core Idea of Continuous Convolution

Instead of assigning a fixed weight,

SchNet computes the weight dynamically.

Suppose two atoms are separated by distance

$$r_{ij}.$$

Instead of using

$$w,$$

SchNet uses

$$W(r_{ij}),$$

where

$$W(\cdot)$$

is a learnable continuous function.

Thus,

different interatomic distances produce different convolution filters.

---

# From Constant Weights to Continuous Functions

Traditional convolution

$$\text{Weight} = \text{constant}.$$

Continuous convolution

$$\text{Weight} = W(r).$$

Instead of storing one number,

the network learns an entire function.

Conceptually,

```text
Distance

↓

Neural Network

↓

Filter Weight

```

Every unique distance generates its own filter.

---

# Continuous Convolution as Distance-Dependent Filtering

Suppose atom $i$ has three neighbors.

Their distances are

$$1.5 \, \text{\AA},$$

$$2.1 \, \text{\AA},$$

$$3.8 \, \text{\AA}.$$

The learned filter network produces

```text
1.5 Å

↓

Filter A

2.1 Å

↓

Filter B

3.8 Å

↓

Filter C

```

Unlike CNNs,

every neighbor receives its own filter.

The filters vary smoothly as distances change.

---

# Mathematical Formulation

Suppose

$$\mathbf{x}_j$$

denotes the feature vector of neighboring atom $j$.

Traditional graph convolution often computes

$$\mathbf{m}_{ij} = W\mathbf{x}_j,$$

where

$$W$$

is a constant weight matrix.

Continuous convolution replaces this by

$$\mathbf{m}_{ij} = W(r_{ij})\mathbf{x}_j.$$

Notice the crucial difference.

The filter

$$W(r_{ij})$$

depends directly on the distance between atoms.

Therefore,

every neighbor contributes differently.

---

# Continuous Filters Are Learnable

One might wonder

how

$$W(r)$$

is determined.

It is **not** specified manually.

Instead,

another neural network learns the mapping

$$r \longrightarrow W(r).$$

During training,

the filter network automatically discovers

how interaction strength should vary with distance.

This learning process is completely data driven.

---

# Smoothness of Continuous Filters

An important consequence is smooth behavior.

Suppose the distance changes slightly.

$$2.00 \, \text{\AA} \rightarrow 2.02 \, \text{\AA}.$$

A fixed graph representation may remain unchanged,

or suddenly change after crossing a cutoff.

A continuous filter,

however,

changes gradually.

```text
Distance

↓

Filter Value

↓

Smooth Change

```

This produces much smoother energy surfaces,

which is essential for force prediction.

---

# Physical Interpretation

Continuous filters resemble interaction potentials used in physics.

For example,

consider the Lennard–Jones potential.

Its value changes continuously with distance.

Similarly,

electrostatic interactions satisfy

$$E \propto \frac{1}{r}.$$

SchNet does not explicitly use these equations.

Instead,

it learns an interaction function directly from data.

This allows the network to approximate complex quantum mechanical interactions without manually specifying physical formulas.

---

# Continuous Convolution Is Translation Invariant

Suppose every atom is translated by

$$(5,3,-2).$$

Their coordinates become

$$\mathbf{R}_i' = \mathbf{R}_i + (5,3,-2).$$

However,

the distance between any two atoms remains

$$r_{ij}.$$

Therefore,

the continuous filters remain unchanged.

SchNet is naturally **translation invariant**.

---

# Continuous Convolution Is Rotation Invariant

Now rotate the entire molecule.

Individual coordinates change,

but pairwise distances remain identical.

Consequently,

the learned filters

$$W(r_{ij})$$

also remain unchanged.

This provides **rotational invariance**.

---

# Continuous Convolution Is Permutation Invariant

Suppose atoms are reordered.

```text
C O H

↓

H C O

```

The atom indices change,

but

* atomic identities,
* coordinates,
* pairwise distances

remain the same.

Therefore,

the predicted energy does not depend on the ordering of atoms.

This satisfies the important physical principle of **permutation invariance**.

---

# Comparing Classical and Continuous Convolution

| Classical CNN | Continuous Convolution |
| --- | --- |
| Regular grid | Irregular atomic positions |
| Fixed kernel | Distance-dependent filter |
| Constant weights | Learned continuous function |
| Image pixels | Atoms |
| Kernel slides over image | Filter generated for each atomic pair |

This comparison highlights why continuous convolution is much more suitable for atomistic systems.

---

# Why Continuous Convolution Matters

Continuous convolution provides several important advantages.

**Adaptive interactions**

Each neighbor receives a unique interaction weight determined by its distance.

---

**Smooth energy surfaces**

Small atomic movements produce smooth changes in predicted energy.

---

**Physics-inspired representation**

Interactions depend on geometry rather than arbitrary graph edges.

---

**Foundation for force prediction**

Because the convolution operation is fully differentiable,

forces can later be computed as

$$\mathbf{F} = - \nabla E.$$

---

**General applicability**

The same framework works for

* molecules,
* crystals,
* surfaces,
* defects,
* amorphous materials,

without requiring manually defined bond types.

---

# From Continuous Convolution to Continuous-Filter Convolution

Although continuous convolution provides the conceptual foundation,

SchNet introduces an even more powerful idea.

Rather than directly learning a continuous weight function,

it constructs **continuous filters** using a dedicated neural network.

These filters are generated dynamically from interatomic distances and are then applied to neighboring atomic feature vectors during message passing.

This operation,

known as **Continuous-Filter Convolution (CFConv),**

is the defining computational building block of SchNet.

---

# Summary

Traditional convolutional neural networks rely on fixed kernels that operate on data arranged on regular grids, making them unsuitable for molecules and crystalline materials where atoms occupy arbitrary positions in continuous three-dimensional space. SchNet addresses this limitation by replacing fixed convolution kernels with **continuous, distance-dependent filter functions**. Instead of assigning the same weight to every neighboring atom, the network learns a unique interaction filter for each interatomic distance.

This formulation naturally respects the continuous nature of atomic interactions, preserves important physical symmetries such as translation, rotation, and permutation invariance, and produces smooth energy landscapes that are essential for accurate force prediction. Continuous convolution therefore provides the conceptual foundation for SchNet's architecture.

In the next section, **16.6 Continuous-Filter Convolution (CFConv)**, we will derive the complete mathematical formulation of SchNet's convolution operator, explain how the filter-generating network is constructed, analyze the tensor operations involved, and implement the CFConv layer from scratch using PyTorch.

# 16.6 Continuous-Filter Convolution (CFConv)

In the previous section, we introduced the idea of **continuous convolution** and explained why traditional convolutional operators cannot be directly applied to atomistic systems. We saw that instead of using a fixed convolution kernel, SchNet learns **distance-dependent filters** that change continuously with the geometry of the system.

However, an important question remains unanswered:

> **How are these continuous filters actually constructed and applied?**

The answer is the **Continuous-Filter Convolution (CFConv)** layer.

CFConv is the computational core of SchNet. Every interaction between neighboring atoms passes through this operation, making it responsible for learning how atomic environments influence one another. Understanding CFConv is therefore essential for understanding not only SchNet but also many later atomistic neural networks that build upon its principles.

In this section, we derive the CFConv operation from first principles, explain its mathematical formulation, analyze every tensor involved, and develop an intuitive understanding before implementing it in PyTorch.

---

# 16.6.1 From Fixed Kernels to Dynamic Filters

Recall the standard convolution operation from classical CNNs.

For a one-dimensional signal, convolution can be written as

$$y_i=\sum_{k}w_kx_{i+k},$$

where

* $x_{i+k}$ is the neighboring input,
* $w_k$ is a fixed kernel weight.

The crucial point is that the kernel weights are **constant**.

Regardless of where the convolution is applied,

the same weights are reused.

This property is known as **weight sharing**.

For images, this assumption is reasonable because neighboring pixels are always arranged on a regular grid.

Atoms, however, are fundamentally different.

Their positions vary continuously,

and the spacing between neighboring atoms is never constant.

Therefore,

using a fixed convolution kernel is physically unrealistic.

Instead,

SchNet replaces the constant weight

$$w_k$$

with a function that depends on the interatomic distance.

Conceptually,

$$w_k \quad\longrightarrow\quad W(r_{ij}),$$

where

* $r_{ij}$ is the distance between atoms $i$ and $j$,
* $W(\cdot)$ is a learnable filter-generating function.

This simple replacement transforms ordinary convolution into continuous-filter convolution.

---

# 16.6.2 Atomic Features Before Convolution

Assume a material contains $N$ atoms.

Each atom is represented by an embedding vector

$$\mathbf{x}_i\in\mathbb{R}^{F},$$

where

* $F$ is the embedding dimension,
* $\mathbf{x}_i$ contains the learned representation of atom $i$.

For example,

if

$$F=128,$$

then every atom is represented by a vector containing 128 learned features.

Collecting all atoms together gives

$$\mathbf{X}=\begin{bmatrix}\mathbf{x}_1\\mathbf{x}_2\\vdots\\mathbf{x}_N\end{bmatrix}\in\mathbb{R}^{N\times F}.$$

This feature matrix serves as the input to the CFConv layer.

---

# 16.6.3 Computing Pairwise Distances

The first step of CFConv is computing distances between neighboring atoms.

Suppose atom $i$ is located at

$$\mathbf{R}_i=(x_i,y_i,z_i),$$

and atom $j$ is located at

$$\mathbf{R}_j=(x_j,y_j,z_j).$$

Their Euclidean distance is

$$r_{ij}=\left|\mathbf{R}_i-\mathbf{R}_j\right|.$$

Expanding the norm,

$$r_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2}.$$

These distances become the primary geometric input to the filter network.

---

# 16.6.4 Why Raw Distances Are Not Enough

One might ask,

why not simply feed

$$r_{ij}$$

directly into a neural network?

Although this is theoretically possible,

it performs poorly in practice.

Consider three neighboring atoms.

| Pair | Distance |
| ---- | -------: |
| A–B  |   1.80 Å |
| A–C  |   1.85 Å |
| A–D  |   1.90 Å |

These values are numerically very close.

A neural network receiving only raw distances may struggle to distinguish subtle geometric differences.

Instead,

SchNet first transforms each distance into a richer representation using **Radial Basis Functions (RBFs)**.

For now,

we denote this transformation as

$$\mathbf{e}_{ij}=\phi(r_{ij}),$$

where

* $\phi(\cdot)$ is the RBF expansion,
* $\mathbf{e}_{ij}$ is the distance embedding.

The details of RBFs will be covered in Section **16.8**.

---

# 16.6.5 The Filter-Generating Network

The embedded distance vector

$$\mathbf{e}_{ij}$$

is passed through a small neural network.

Mathematically,

$$\mathbf{W}_{ij} = \text{MLP}(\mathbf{e}_{ij}).$$

Here,

* MLP denotes a multilayer perceptron,
* $\mathbf{W}_{ij}$ is the continuous filter generated specifically for atoms $i$ and $j$.

Notice something remarkable.

Unlike CNNs,

there is **no single convolution kernel**.

Instead,

every neighboring pair receives its own filter.

Conceptually,

```text
Distance

↓

Radial Basis Expansion

↓

Neural Network

↓

Continuous Filter
```

Thus,

two neighboring atoms separated by different distances will receive different convolution filters.

---

# 16.6.6 Applying the Continuous Filter

Once the filter has been generated,

it acts on the neighboring atomic features.

Suppose neighboring atom $j$ has feature vector

$$
\mathbf{x}_j.
$$

Its contribution to atom $i$ is

$$\mathbf{m}_{ij} = \mathbf{W}_{ij} \odot \mathbf{x}_j,$$

where

$\odot$

denotes element-wise multiplication.

Unlike classical graph convolutions,

the filter itself depends on geometry.

Therefore,

the same neighboring atom contributes differently depending on its distance.

---

# 16.6.7 Aggregating Neighbor Information

Every neighboring atom contributes a message.

The final message received by atom $i$ is

$$\mathbf{m}_i = \sum_{j\in\mathcal{N}(i)} \mathbf{W}_{ij} \odot \mathbf{x}_j.$$

This is the central equation of Continuous-Filter Convolution.

Each neighboring atom contributes

* its own feature vector,
* multiplied by
* a filter generated specifically for its distance.

The summation combines information from the entire local atomic environment.

---

# 16.6.8 Updating Atomic Features

The aggregated message is then combined with the current atomic representation.

A simplified update rule is

$$\boxed{\mathbf{x}_i' = \mathbf{x}_i + \sum_{j\in\mathcal{N}(i)} \mathbf{W}_{ij} \odot \mathbf{x}_j}$$
Expanding the message,

$$\boxed{\mathbf{x}_i' = \mathbf{x}_i + \sum_{j\in\mathcal{N}(i)} \mathbf{W}_{ij} \odot \mathbf{x}_j}$$

This equation defines the essence of the SchNet interaction mechanism.

Unlike ordinary graph convolutions,

the interaction weights are not fixed parameters.

They are generated dynamically from atomic geometry.

---

# 16.6.9 Why CFConv Is More Expressive

Suppose atom $i$ has two carbon neighbors.

Neighbor A

$$r = 1.42 \, \text{\AA}$$

Neighbor B

$$r = 2.75 \, \text{\AA}$$

A traditional graph convolution might treat both neighbors similarly because they share the same edge type.

CFConv,

however,

generates two completely different filters.

$$W(1.42) \neq W(2.75).$$

Thus,

the model automatically learns that

closer atoms generally influence one another differently from more distant atoms.

---

# 16.6.10 Tensor Shapes Inside CFConv

Understanding tensor dimensions is essential for implementing SchNet.

Assume

* Number of atoms = $N$
* Number of neighbor pairs = $E$
* Feature dimension = $F$

The tensors are

| Tensor              | Shape   |
| ------------------- | ------- |
| Atomic features     | $(N,F)$ |
| Coordinates         | $(N,3)$ |
| Neighbor indices    | $(E,2)$ |
| Distances           | $(E)$   |
| Distance embeddings | $(E,K)$ |
| Generated filters   | $(E,F)$ |
| Neighbor features   | $(E,F)$ |
| Messages            | $(E,F)$ |
| Updated features    | $(N,F)$ |

Here,

$K$

denotes the number of radial basis functions.

This tensor flow forms the computational backbone of SchNet.

---

# 16.6.11 Intuitive Interpretation

Imagine standing in a crowded room.

Each person speaks with a different loudness depending on how far away they are.

Nearby voices are louder.

Distant voices are quieter.

Instead of simply counting who is present,

your brain naturally weights each voice according to its distance.

CFConv behaves similarly.

Each neighboring atom contributes information,

but the importance of that contribution is determined continuously by its distance.

---

# 16.6.12 Why CFConv Is Physically Meaningful

Continuous-filter convolution possesses several desirable physical properties.

**Distance sensitivity**

Every interaction depends explicitly on interatomic distance.

---

**Smoothness**

Small atomic displacements produce small changes in interaction strength.

---

**Differentiability**

Every operation is differentiable,

allowing energy gradients to be computed.

---

**Adaptability**

Interaction filters are learned from data,

rather than manually specified.

---

**Generality**

The same operation works for

* molecules,
* crystals,
* surfaces,
* amorphous materials,
* defects,
* interfaces.

---

# Summary

Continuous-Filter Convolution replaces the fixed convolution kernels used in traditional neural networks with **dynamic filters generated from interatomic distances**. Starting from atomic embeddings and Cartesian coordinates, SchNet computes pairwise distances, transforms them into learned distance embeddings, generates continuous filters through a neural network, and uses these filters to weight the contribution of neighboring atoms during message passing.

The resulting update equation,

$\displaystyle \boxed{\mathbf{x}_i' = \mathbf{x}_i + \sum_{j\in\mathcal{N}(i)} \mathbf{W}_{ij} \odot \mathbf{x}_j}$

captures the central idea of SchNet: **atomic interactions are not fixed but vary continuously with geometry**. This makes CFConv both physically meaningful and highly expressive, enabling SchNet to learn smooth energy surfaces and accurate atomic interactions directly from data.

In the next section, **16.6.13 Mathematical Derivation of Continuous-Filter Convolution**, we will derive the CFConv equation rigorously from continuous convolution theory, examine the role of linear transformations and filter networks, and connect SchNet's formulation to the broader framework of graph neural networks and continuous integral operators.

## 16.6.13 Derivation of the Continuous-Filter Convolution Equation

In the previous sections, we introduced the Continuous-Filter Convolution (CFConv) operation and presented its final mathematical form. Although the equation appears simple, it is the result of several important ideas that combine concepts from graph neural networks, continuous mathematics, and quantum chemistry.

Rather than memorizing the CFConv equation, it is far more valuable to understand **how it is derived**. Once the derivation becomes clear, every component of SchNet—from radial basis functions to interaction blocks—will appear as a natural consequence of this formulation.

In this section, we derive the CFConv equation step by step, beginning with ordinary convolution and gradually transforming it into the operator used in SchNet.

---

## Step 1 — Classical Discrete Convolution

Consider a one-dimensional discrete signal

$$
x=[x_1,x_2,\ldots,x_n].
$$

A classical convolution computes the output at position $i$ as

$$
y_i=\sum_k w_k x_{i+k}.
$$

Here,

* $x_{i+k}$ is a neighboring input value.
* $w_k$ is a fixed convolution weight.

Notice an important property.

The weights

$$
w_k
$$

are learned **once** during training and then reused everywhere.

This weight sharing is possible because the neighboring pixels of an image always occupy fixed positions.

---

## Step 2 — Why This Cannot Be Used for Atoms

Suppose we now replace image pixels with atoms.

Each atom has a position

$$
\mathbf{R}_i=(x_i,y_i,z_i).
$$

Unlike pixels,

atoms are **not** arranged on a regular grid.

Consider two neighboring atoms.

Atom A may be located

* 1.41 Å away,

while another neighbor may be

* 2.73 Å away.

Since neighboring atoms appear at arbitrary locations,

there is no longer a fixed index $k$ describing the neighborhood.

Consequently,

the convolution weight

$$
w_k
$$

loses its physical meaning.

We need a new definition of convolution.

---

## Step 3 — Replace Position Index with Geometry

Instead of describing neighbors by their position in a grid,

we describe them by their **physical distance**.

The distance between atoms $i$ and $j$ is

$$
r_{ij}=|\mathbf{R}_i-\mathbf{R}_j|.
$$

This single quantity captures the geometric relationship between two atoms.

Rather than assigning a fixed weight,

we now allow the weight to depend on distance.

Mathematically,

$$
w_k
\quad\rightarrow\quad
W(r_{ij}).
$$

This is the first major conceptual step toward SchNet.

The convolution kernel is no longer constant.

Instead,

it becomes a continuous function.

---

## Step 4 — Continuous Neighbor Contributions

Suppose atom $i$ has several neighboring atoms.

Each neighboring atom possesses a feature vector

$$
\mathbf{x}_j.
$$

Rather than multiplying every neighbor by the same kernel,

each neighbor receives its own distance-dependent filter.

The contribution of neighbor $j$ becomes

$$
\mathbf{m}_{ij}
=

W(r_{ij})\mathbf{x}_j.
$$

Notice that two identical atoms located at different distances will produce different messages.

This behavior is impossible using conventional graph convolutions.

---

## Step 5 — Summing Information from All Neighbors

Every neighboring atom contributes information.

The complete message received by atom $i$ is therefore the sum of all neighbor contributions,

$$
\mathbf{m}_i
=

\sum_{j\in\mathcal{N}(i)}
W(r_{ij})
\mathbf{x}_j.
$$

This equation already resembles message passing in graph neural networks.

However,

the important difference is that every message is weighted by a continuously varying filter rather than a fixed edge weight.

---

## Step 6 — Updating the Atomic Representation

The purpose of message passing is to improve the representation of every atom.

Therefore,

the newly computed message is combined with the existing atomic features.

The simplest update rule is

$$
\mathbf{x}_i'
=
\mathbf{x}_i+\mathbf{m}_i.
$$

Substituting the previous equation gives

$$
\mathbf{x}_i'
=

\mathbf{x}*i+
\sum*{j\in\mathcal{N}(i)}
W(r_{ij})
\mathbf{x}_j.
$$

This equation captures the central philosophy of SchNet.

Every atom learns from its neighbors,

but the influence of each neighbor depends continuously on geometry.

---

## Step 7 — Introducing Learnable Feature Transformations

In practice,

SchNet does not apply the filter directly to the original feature vector.

Instead,

the neighboring features are first transformed using a learnable linear layer.

Let

$$
\mathbf{h}_j=L(\mathbf{x}_j),
$$

where $L(\cdot)$ denotes a linear transformation.

The message becomes

$$
\mathbf{m}_{ij}
=

W(r_{ij})
\odot
\mathbf{h}_j.
$$

Here,

$\odot$

denotes element-wise multiplication.

This allows the network to learn a more expressive feature representation before neighbor aggregation.

---

## Step 8 — Final CFConv Equation

Combining every step,

the Continuous-Filter Convolution layer can be written as

$$
\boxed{
\mathbf{x}_i'
=

\mathbf{x}*i+
\sum*{j\in\mathcal{N}(i)}
W(r_{ij})
\odot
L(\mathbf{x}_j)
}
$$

This is the complete mathematical form of the CFConv operator used in SchNet.

Every term has a clear physical interpretation.

* $\mathbf{x}_i$ is the current representation of atom $i$.
* $L(\mathbf{x}_j)$ transforms the neighboring atomic features.
* $W(r_{ij})$ generates a distance-dependent interaction filter.
* The summation aggregates information from the local atomic environment.

---

## Understanding the Equation Intuitively

Imagine standing in the center of a group of people.

Every person tells you some information.

However,

you naturally pay more attention to people standing nearby than to people standing farther away.

Your final understanding is therefore a weighted combination of everyone's contributions.

SchNet behaves in exactly the same way.

Each neighboring atom contributes information,

but its contribution is weighted according to its distance.

Closer atoms generally influence the representation more strongly,

while distant atoms contribute less.

Importantly,

the network **learns** these weighting functions from data instead of relying on manually defined physical equations.

---

## Why This Equation Is Powerful

The CFConv equation possesses several remarkable properties.

First,

it is **continuous**.

Small changes in atomic positions produce small changes in interaction strength.

Second,

it is **fully differentiable**.

Since every operation is differentiable,

the predicted energy can later be differentiated with respect to atomic coordinates to compute atomic forces.

Third,

it naturally supports molecules and crystals containing different numbers of atoms because the summation operates over the local neighborhood of each atom rather than requiring a fixed-size input.

Finally,

the interaction function is learned automatically.

Instead of assuming how atoms should interact,

SchNet discovers these relationships directly from quantum mechanical training data.

---

## Connection to Message Passing Neural Networks

Readers familiar with Message Passing Neural Networks (MPNNs) may recognize a similar structure.

A general message passing framework can be written as

$$
\mathbf{m}_{ij}=M(\mathbf{x}_i,\mathbf{x}*j,\mathbf{e}*{ij}),
$$

followed by

$$
\mathbf{x}_i'=U\left(\mathbf{x}*i,\sum_j\mathbf{m}*{ij}\right),
$$

where

* $M(\cdot)$ is the message function,
* $U(\cdot)$ is the update function,
* $\mathbf{e}_{ij}$ represents edge information.

SchNet is a specialized MPNN in which the edge information is not manually defined.

Instead,

the edge representation is generated directly from the **continuous interatomic distance**, making the message function significantly more expressive for atomistic systems.

---

## Summary

The Continuous-Filter Convolution equation is derived by progressively replacing the assumptions of classical convolution with operations that respect the geometry of atomistic systems. Fixed convolution kernels are replaced by learnable distance-dependent filters, discrete neighborhood indices are replaced by continuous interatomic distances, and neighbor contributions are aggregated through message passing. The resulting operator,

$$
\boxed{
\mathbf{x}_i'
=

\mathbf{x}*i+
\sum*{j\in\mathcal{N}(i)}
W(r_{ij})
\odot
L(\mathbf{x}_j)
}
$$

forms the mathematical heart of SchNet and enables the network to learn smooth, physically meaningful interactions directly from atomic geometry.

In the next section, **16.6.14 Filter-Generating Networks**, we will examine how the function $W(r_{ij})$ is actually learned, exploring the neural network architecture that transforms interatomic distances into continuous convolution filters.

# 16.6.14 Filter-Generating Network

In the previous section, we derived the Continuous-Filter Convolution (CFConv) equation and observed that its defining component is the **continuous filter**,

$$
W(r_{ij}).
$$

Unlike a conventional convolutional neural network, SchNet does not store a fixed convolution kernel. Instead, it **generates a new filter for every pair of neighboring atoms** based on their interatomic distance.

This raises a natural question:

> **How does SchNet generate these continuous filters?**

The answer lies in a small neural network called the **Filter-Generating Network (FGN)**.

This network transforms a scalar distance into a high-dimensional filter vector that is later used during message passing. Although relatively small compared to the overall SchNet architecture, the Filter-Generating Network is one of the most important components because it determines how atomic interactions vary with geometry.

---

# Why Do We Need a Filter-Generating Network?

Suppose two neighboring atoms are separated by

* $1.5$ Å,
* $2.1$ Å,
* $3.8$ Å.

If we used a fixed convolution kernel, all three neighbors would receive identical weights.

However, physics tells us that interactions depend strongly on distance.

For example,

* atoms that are very close experience strong electronic interactions,
* atoms farther apart interact more weakly,
* beyond a certain distance, interactions become negligible.

Therefore, the convolution filter should change continuously with distance.

Instead of storing one filter,

SchNet learns a function

$$
W:\mathbb{R}\rightarrow\mathbb{R}^{F},
$$

where

* the input is the interatomic distance,
* the output is an $F$-dimensional filter vector.

Thus, every unique distance produces its own convolution filter.

---

# The Overall Pipeline

The process of generating a filter can be summarized as

```text
Interatomic Distance
        │
        ▼
Radial Basis Expansion
        │
        ▼
Feedforward Neural Network
        │
        ▼
Continuous Filter
```

Notice that the Filter-Generating Network **does not receive atomic features**.

Its input consists only of the geometric information encoded by the interatomic distance.

---

# Step 1 — Distance as Input

For every neighboring pair of atoms,

SchNet first computes the Euclidean distance

$$
r_{ij}=|\mathbf{R}_i-\mathbf{R}_j|.
$$

Suppose

$$
r_{ij}=2.34\text{ Å}.
$$

At first glance, it might seem reasonable to feed this number directly into a neural network.

However, doing so creates an important problem.

---

# Why Raw Distances Are Difficult to Learn

Consider three neighboring atoms.

| Pair | Distance (Å) |
| ---- | -----------: |
| A–B  |         2.31 |
| A–C  |         2.34 |
| A–D  |         2.37 |

The numerical differences are very small.

Neural networks generally learn more effectively when the input representation is richer than a single scalar.

Furthermore,

many different interaction patterns may occur over similar distance ranges.

To improve learning,

SchNet first expands every distance into a higher-dimensional representation.

This process is called the **Radial Basis Function (RBF) Expansion**.

Mathematically,

$$
r_{ij}
\longrightarrow
\mathbf{e}_{ij}.
$$

Instead of one number,

the network now receives a feature vector.

The details of RBFs will be studied later in this chapter.

For now, simply view

$$
\mathbf{e}_{ij}
$$

as an encoded representation of distance.

---

# Step 2 — Passing Through a Neural Network

The RBF embedding is passed into a small feedforward neural network.

Suppose

$$
\mathbf{e}_{ij}\in\mathbb{R}^{K},
$$

where

$K$

is the number of radial basis functions.

The Filter-Generating Network computes

$$
\mathbf{W}_{ij}
===============

f_{\theta}(\mathbf{e}_{ij}),
$$

where

* $f_{\theta}$ denotes the neural network,
* $\theta$ represents its learnable parameters,
* $\mathbf{W}_{ij}$ is the generated filter.

Unlike a conventional convolution kernel,

this filter exists **only for the current pair of atoms**.

---

# Typical Architecture of the Filter Network

The original SchNet implementation uses a relatively small multilayer perceptron.

Conceptually,

```text
Distance Embedding
        │
        ▼
Linear Layer
        │
        ▼
Activation Function
        │
        ▼
Linear Layer
        │
        ▼
Continuous Filter
```

If the atomic embedding dimension is

$$
F=128,
$$

the output filter also has dimension

$$
128.
$$

Each component of the filter controls one feature channel during message passing.

---

# Mathematical Formulation

Assume that the RBF expansion produces

$$
\mathbf{e}_{ij}\in\mathbb{R}^{K}.
$$

The first linear layer computes

$$
\mathbf{h}_1
============

\mathbf{W}*1\mathbf{e}*{ij}
+
\mathbf{b}_1.
$$

Next,

a nonlinear activation function is applied,

$$
\mathbf{h}_2
============

\sigma(\mathbf{h}_1),
$$

where

$\sigma(\cdot)$

may denote the shifted softplus activation used in the original SchNet implementation.

Finally,

the second linear layer produces

$$
\mathbf{W}_{ij}
===============

\mathbf{W}_2\mathbf{h}_2
+
\mathbf{b}_2.
$$

The resulting vector

$$
\mathbf{W}_{ij}
$$

is the continuous convolution filter.

---

# Why Use a Neural Network?

One might ask,

why not simply use a fixed mathematical formula such as

$$
W(r)=\frac{1}{r}?
$$

Although many physical interactions approximately follow known equations,

real quantum mechanical interactions are considerably more complex.

For example,

interaction strength depends on

* atomic species,
* local coordination,
* electron density,
* chemical environment,
* hybridization.

Rather than forcing the model to follow one predefined equation,

SchNet allows the neural network to learn the interaction function directly from data.

This greatly increases the expressive power of the model.

---

# Dynamic Filter Generation

A remarkable property of SchNet is that filters are generated **dynamically**.

Suppose we have three neighboring atom pairs.

| Pair | Distance (Å) |
| ---- | -----------: |
| A–B  |         1.60 |
| A–C  |         2.45 |
| A–D  |         3.80 |

The Filter-Generating Network produces

$$
W_{AB},
$$

$$
W_{AC},
$$

and

$$
W_{AD},
$$

where

$$
W_{AB}
\neq
W_{AC}
\neq
W_{AD}.
$$

Thus,

every neighboring interaction receives its own learned convolution filter.

This behavior is fundamentally different from CNNs, where the same kernel is reused everywhere.

---

# Tensor Shapes Inside the Filter Network

Suppose

* Number of neighbor pairs = $E$
* Number of radial basis functions = $K$
* Atomic embedding dimension = $F$

The tensors have the following shapes.

| Quantity         | Shape   |
| ---------------- | ------- |
| Distance         | $(E)$   |
| RBF embedding    | $(E,K)$ |
| Hidden layer     | $(E,H)$ |
| Generated filter | $(E,F)$ |

Here,

$H$

denotes the hidden dimension of the filter network.

Each row of the output corresponds to one neighbor pair.

---

# PyTorch Implementation

A simplified implementation of the Filter-Generating Network is shown below.

```python
import torch
import torch.nn as nn

class FilterNetwork(nn.Module):

    def __init__(self, num_rbf, hidden_dim, embedding_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, rbf):
        return self.network(rbf)
```

Example usage:

```python
num_edges = 20
num_rbf = 64

rbf = torch.randn(num_edges, num_rbf)

filter_net = FilterNetwork(
    num_rbf=64,
    hidden_dim=128,
    embedding_dim=128
)

filters = filter_net(rbf)

print(filters.shape)
```

Output

```text
torch.Size([20, 128])
```

This means that the network has generated

* one filter for every neighboring atom pair,
* with each filter containing 128 learned values.

---

# Why the Filter Dimension Equals the Embedding Dimension

Recall that the neighboring atomic feature vector is

$$
\mathbf{x}_j
\in
\mathbb{R}^{F}.
$$

During CFConv,

the filter performs an element-wise multiplication,

$$
\mathbf{m}_{ij}
===============

\mathbf{W}_{ij}
\odot
\mathbf{x}_j.
$$

Since element-wise multiplication requires tensors of identical shape,

the generated filter must also belong to

$$
\mathbb{R}^{F}.
$$

This explains why the output dimension of the Filter-Generating Network matches the atomic embedding dimension.

---

# Physical Interpretation

The Filter-Generating Network can be viewed as a **learned interaction potential**.

Instead of explicitly programming how interaction strength varies with distance,

SchNet learns this relationship directly from quantum mechanical data.

During training,

the network gradually discovers

* which distances are most important,
* how interaction strength changes with geometry,
* how neighboring atoms influence one another.

Consequently,

the filter network becomes an implicit model of atomic interactions.

---

# Summary

The Filter-Generating Network is responsible for producing the continuous convolution filters that distinguish SchNet from conventional graph neural networks. Beginning with the interatomic distance, the model first constructs a higher-dimensional distance embedding using radial basis functions. This embedding is then processed by a small multilayer perceptron that generates an interaction-specific filter vector.

Unlike the fixed kernels used in classical convolutional neural networks, these filters are created dynamically for every neighboring atom pair. As a result, SchNet can model smooth, geometry-dependent interactions that closely reflect the underlying physics of molecules and crystalline materials while remaining fully differentiable and trainable end-to-end.

In the next section, **16.7 Interaction Blocks**, we will combine the Continuous-Filter Convolution layer, residual connections, atom-wise updates, and nonlinear transformations into the complete computational block that forms the backbone of the SchNet architecture.

# 16.7 Interaction Blocks

The **Interaction Block** is the fundamental building block of the SchNet architecture. If the Continuous-Filter Convolution (CFConv) layer is considered the "heart" of SchNet, then the Interaction Block can be viewed as its "heartbeat." Every time information flows through the network, it passes through one interaction block, allowing atoms to exchange information with their local environments and progressively learn richer chemical representations.

A single CFConv layer is capable of modeling only one round of interactions between neighboring atoms. However, the properties of real materials are determined by **multiple levels of atomic interactions**. An atom is influenced not only by its immediate neighbors but also indirectly by atoms that are several bonds away. To capture these long-range dependencies, SchNet stacks several interaction blocks together.

Each interaction block refines the atomic feature vectors while preserving the geometric information encoded by the continuous filters. As information propagates through multiple blocks, the atomic representations become increasingly informative, enabling accurate prediction of energies, forces, and other quantum mechanical properties.

---

# Why Is an Interaction Block Needed?

Suppose we have a simple crystal.

```text
Si —— Si —— Si —— Si
```

Consider the leftmost silicon atom.

Initially, it only has direct information about itself.

After one CFConv layer,

```text
Si ← Si
```

it receives information from its nearest neighbors.

After two interaction blocks,

```text
Si ← Si ← Si
```

information can propagate two hops away.

After three interaction blocks,

```text
Si ← Si ← Si ← Si
```

the atom has indirectly received information from atoms much farther away.

Thus, stacking interaction blocks increases the **receptive field** of the network.

---

# Information Flow Inside an Interaction Block

An interaction block performs four major operations.

```text
Atomic Features
        │
        ▼
Linear Transformation
        │
        ▼
Continuous-Filter Convolution
        │
        ▼
Nonlinear Transformation
        │
        ▼
Residual Addition
        │
        ▼
Updated Atomic Features
```

Each operation plays a distinct role in refining the atomic representations.

---

# Input to the Interaction Block

Assume the input feature matrix is

$$
\mathbf{X}
==========

\begin{bmatrix}
\mathbf{x}_1\
\mathbf{x}_2\
\vdots\
\mathbf{x}_N
\end{bmatrix},
$$

where

* $N$ is the number of atoms,
* each feature vector satisfies

$$
\mathbf{x}_i\in\mathbb{R}^{F}.
$$

The goal of the interaction block is to transform

$$
\mathbf{X}
\longrightarrow
\mathbf{X}',
$$

where the updated features contain richer information about each atom's local environment.

---

# Step 1 — Linear Transformation

Before message passing begins, the atomic features are projected into a new feature space.

This transformation is performed using a learnable linear layer,

$$
\mathbf{h}_i=L_1(\mathbf{x}_i).
$$

In matrix form,

$$
\mathbf{H}=L_1(\mathbf{X}).
$$

This projection allows the network to learn feature combinations that are more suitable for interaction modeling.

If the embedding dimension is

$$
F=128,
$$

then both the input and output tensors have shape

$$
(N,128).
$$

Although the dimensionality remains unchanged, the feature representation becomes more expressive.

---

# Step 2 — Continuous-Filter Convolution

The transformed features are passed through the CFConv layer.

Recall the CFConv equation derived earlier,

$$
\mathbf{m}_i
============

\sum_{j\in\mathcal{N}(i)}
W(r_{ij})
\odot
\mathbf{h}_j.
$$

This operation aggregates information from all neighboring atoms.

Unlike traditional graph convolutions,

the weighting function

$$
W(r_{ij})
$$

depends continuously on the interatomic distance.

The resulting message vector

$$
\mathbf{m}_i
$$

encodes information about the local atomic environment.

---

# Step 3 — Output Transformation

The aggregated message is passed through another learnable linear transformation,

$$
\mathbf{u}_i=L_2(\mathbf{m}_i).
$$

This layer mixes information across feature channels.

Without this transformation,

the expressive power of the interaction block would be significantly reduced.

The tensor shape remains

$$
(N,F).
$$

---

# Step 4 — Residual Connection

One of the most important innovations used in SchNet is the **residual connection**.

Instead of replacing the original atomic features,

the interaction block adds the learned update to the existing representation.

Mathematically,

$$
\mathbf{x}_i'
=============

\mathbf{x}_i+\mathbf{u}_i.
$$

In matrix notation,

$$
\boxed{
\mathbf{X}'
===========

\mathbf{X}
+
L_2
\left(
\mathrm{CFConv}
\left(
L_1(\mathbf{X})
\right)
\right)
}
$$

This equation defines a complete SchNet interaction block.

---

# Why Residual Connections Are Important

Suppose we stack six interaction blocks.

Without residual connections,

the network may suffer from

* vanishing gradients,
* unstable optimization,
* loss of previously learned information.

Residual learning solves these problems by allowing the model to learn **corrections** instead of entirely new representations.

Conceptually,

instead of learning

$$
\mathbf{x}_i',
$$

the network learns

$$
\Delta\mathbf{x}_i,
$$

such that

$$
\mathbf{x}_i'
=============

\mathbf{x}_i+\Delta\mathbf{x}_i.
$$

Learning small corrections is often much easier than learning a completely new representation.

---

# Information Propagation Across Multiple Blocks

Suppose SchNet contains three interaction blocks.

The computation proceeds as

$$
\mathbf{X}^{(0)}
\rightarrow
\mathbf{X}^{(1)}
\rightarrow
\mathbf{X}^{(2)}
\rightarrow
\mathbf{X}^{(3)}.
$$

Here,

* $\mathbf{X}^{(0)}$ denotes the initial atomic embeddings,
* $\mathbf{X}^{(3)}$ represents the final learned atomic representations.

Each interaction block increases the amount of structural information encoded within every atomic feature vector.

---

# Tensor Flow Through an Interaction Block

Assume

* Number of atoms = $N$
* Feature dimension = $F$

The tensor dimensions evolve as follows.

| Operation             | Tensor Shape |
| --------------------- | ------------ |
| Input features        | $(N,F)$      |
| Linear transformation | $(N,F)$      |
| CFConv output         | $(N,F)$      |
| Output linear layer   | $(N,F)$      |
| Residual addition     | $(N,F)$      |

Notice that the feature dimension remains constant throughout the interaction block.

This simplifies stacking multiple blocks.

---

# PyTorch Implementation

A simplified interaction block can be implemented as follows.

```python
import torch
import torch.nn as nn

class InteractionBlock(nn.Module):

    def __init__(self, hidden_dim, cfconv):
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.cfconv = cfconv
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):

        h = self.linear1(x)

        h = self.cfconv(h, edge_index, edge_attr)

        h = self.linear2(h)

        return x + h
```

This implementation clearly illustrates the residual structure.

The original features are added back to the transformed output.

---

# Stacking Multiple Interaction Blocks

The complete SchNet architecture consists of several interaction blocks arranged sequentially.

```text
Embedding Layer
       │
       ▼
Interaction Block 1
       │
       ▼
Interaction Block 2
       │
       ▼
Interaction Block 3
       │
       ▼
...
       │
       ▼
Atom-wise Prediction
```

The original SchNet paper typically employs six interaction blocks, although this number can be adjusted depending on the complexity of the problem.

Increasing the number of blocks generally enables the model to capture longer-range interactions but also increases computational cost.

---

# Physical Interpretation

Each interaction block can be interpreted as one "communication round" between atoms.

During one round,

* every atom gathers information from its neighbors,
* weighs their contributions according to distance,
* updates its internal representation,
* retains its previous knowledge through the residual connection.

After several rounds,

every atom possesses a learned representation that encodes not only its own chemical identity but also the geometric and chemical characteristics of its surrounding environment.

This learned representation forms the basis for predicting macroscopic material properties.

---

# Why Interaction Blocks Are Powerful

Interaction blocks combine several powerful ideas into a single computational unit.

* Continuous geometry-aware message passing.
* Learnable nonlinear feature transformations.
* Residual learning for stable optimization.
* Repeated information propagation across the atomic graph.
* End-to-end differentiable computation.

Together, these components enable SchNet to learn highly expressive atomic representations directly from structural data without relying on handcrafted descriptors.

---

# Summary

The Interaction Block is the core computational module of SchNet. It integrates linear feature transformations, Continuous-Filter Convolution, nonlinear processing, and residual learning into a single architecture that updates atomic representations through geometry-aware message passing. By stacking multiple interaction blocks, SchNet progressively enlarges the receptive field of each atom, allowing information to propagate across increasingly larger regions of a molecule or crystal.

The residual connections ensure stable optimization while preserving previously learned information, making deep SchNet models both effective and trainable. As a result, the final atomic feature vectors contain rich information about local chemical environments and long-range structural relationships.

In the next section, **16.8 Radial Basis Functions (RBFs)**, we will study one of the most important components of SchNet in detail. We will explain why raw interatomic distances are insufficient for learning, derive the Gaussian radial basis expansion mathematically, analyze its properties, and implement it from scratch in PyTorch.


# 16.8 Radial Basis Functions (RBFs)

In the previous sections, we discussed how SchNet generates continuous filters from interatomic distances. The Filter-Generating Network receives information about atomic separation and produces distance-dependent convolution filters.

However, directly using the raw distance value $r_{ij}$ as input to a neural network is not an efficient representation.

SchNet solves this problem using a technique called **Radial Basis Function (RBF) expansion**.

RBFs transform a single scalar distance into a higher-dimensional feature vector that allows the neural network to learn complex distance-dependent interactions more effectively.

The RBF representation is one of the most important components of SchNet because it converts continuous atomic geometry into a form that neural networks can process efficiently.

---

# 16.8.1 Why Distance Encoding Is Necessary

The fundamental input describing atomic interactions is the distance between two atoms:

$$
r_{ij}=||\mathbf{R}_i-\mathbf{R}_j||
$$

where

* $\mathbf{R}_i$ is the position of atom $i$,
* $\mathbf{R}_j$ is the position of atom $j$.

For example:

| Atomic pair | Distance |
| ----------- | -------: |
| Si–Si       |   2.35 Å |
| C–C         |   1.54 Å |
| O–H         |   0.96 Å |

The distance contains important physical information.

However, a single number is a very limited representation.

Consider two distances:

$$
r_1=1.50
$$

and

$$
r_2=2.50
$$

A neural network receiving only these numbers must learn the entire shape of the interaction function from very sparse information.

This creates a difficult learning problem.

---

# 16.8.2 The Idea Behind Radial Basis Expansion

The main idea of RBF expansion is:

Instead of representing a distance using one number, represent it using multiple basis functions.

A distance becomes a vector:

$$
r_{ij}
\rightarrow
\mathbf{e}_{ij}
$$

where

$$
\mathbf{e}_{ij}
===============

(e_1,e_2,\dots,e_K)
$$

and $K$ is the number of radial basis functions.

For example, instead of:

```text
Distance

2.35 Å
```

the network receives:

```text
Distance embedding

[0.02, 0.15, 0.73, 0.91, 0.34, ...]
```

Each component describes the similarity of the distance to a particular region.

---

# 16.8.3 Gaussian Radial Basis Functions

The original SchNet architecture uses Gaussian radial basis functions.

A Gaussian basis function is defined as:

$$
e_k(r)=
\exp
\left(
-\gamma(r-\mu_k)^2
\right)
$$

where

* $r$ is the interatomic distance,
* $\mu_k$ is the center of the $k$-th Gaussian,
* $\gamma$ controls the width of the Gaussian.

Each basis function responds strongly only to a specific distance range.

---

# Understanding Gaussian Expansion

Suppose we use three Gaussian functions.

Their centers are:

$$
\mu_1=1.0
$$

$$
\mu_2=2.0
$$

$$
\mu_3=3.0
$$

For a distance

$$
r=2.1
$$

the responses might be:

```text
Distance = 2.1 Å

Basis 1 → weak activation

Basis 2 → strong activation

Basis 3 → medium activation
```

The network now understands that the distance is close to the second interaction region.

---

# 16.8.4 Distance Embedding Matrix

For a system containing many neighboring pairs,

each distance is expanded separately.

Suppose we have $E$ atomic pairs.

The original distances are:

$$
[r_1,r_2,\dots,r_E]
$$

After RBF expansion:

$$
\mathbf{E}
==========

\begin{bmatrix}
e_1(r_1)&e_2(r_1)&...&e_K(r_1)\
e_1(r_2)&e_2(r_2)&...&e_K(r_2)\
\vdots&\vdots&&\vdots\
e_1(r_E)&e_2(r_E)&...&e_K(r_E)
\end{bmatrix}
$$

The resulting tensor has shape:

$$
(E,K)
$$

where

* $E$ = number of interacting atom pairs,
* $K$ = number of radial basis functions.

---

# 16.8.5 Why Multiple Basis Functions Help

A neural network learns by combining features.

A single distance value gives only one feature.

After RBF expansion:

$$
r
\rightarrow
(e_1,e_2,...,e_K)
$$

the network receives many distance-related features.

This allows it to learn functions such as:

* short-range repulsion,
* bonding regions,
* equilibrium distances,
* long-range decay.

For example:

A learned interaction may look like:

```text
Small distance

Strong repulsion

↓

Bonding distance

Stable interaction

↓

Large distance

Weak interaction
```

The RBF representation makes these patterns easier to learn.

---

# 16.8.6 Cutoff Function

In atomistic systems, interactions are usually limited to a cutoff radius.

For example:

$$
r_c=5\text{ Å}
$$

Atoms beyond this distance are ignored.

However, simply removing atoms at the cutoff creates a discontinuity.

Example:

$$
4.99\text{ Å}
$$

interaction exists.

But

$$
5.01\text{ Å}
$$

interaction disappears.

To avoid this problem, SchNet uses a smooth cutoff function.

A commonly used cutoff function is:

$$
f_c(r)=
\begin{cases}
0.5
\left(
\cos
\frac{\pi r}{r_c}
+1
\right)
&
r<r_c
\
0
&
r\geq r_c
\end{cases}
$$

This smoothly decreases the interaction strength to zero at the cutoff radius.

---

# 16.8.7 Applying the Cutoff to RBF Features

The final distance embedding becomes:

$$
e_k(r)
======

f_c(r)
\exp
(
-\gamma(r-\mu_k)^2
)
$$

The cutoff function ensures:

* nearby atoms contribute strongly,
* distant atoms contribute smoothly less,
* no artificial discontinuity appears.

---

# 16.8.8 PyTorch Implementation of RBF Expansion

A simple implementation:

```python
import torch
import torch.nn as nn


class GaussianRBF(nn.Module):

    def __init__(self, cutoff, num_rbf, gamma):
        super().__init__()

        self.cutoff = cutoff
        self.num_rbf = num_rbf

        centers = torch.linspace(
            0,
            cutoff,
            num_rbf
        )

        self.register_buffer(
            "centers",
            centers
        )

        self.gamma = gamma


    def forward(self, distances):

        diff = distances[:, None] - self.centers

        rbf = torch.exp(
            -self.gamma * diff**2
        )

        return rbf
```

Example:

```python
distances = torch.tensor(
    [1.5, 2.5, 3.5]
)

rbf = GaussianRBF(
    cutoff=5.0,
    num_rbf=64,
    gamma=10
)

features = rbf(distances)

print(features.shape)
```

Output:

```
torch.Size([3,64])
```

Each distance has now been converted into a 64-dimensional representation.

---

# 16.8.9 Connection with Materials Science

The RBF representation is especially important for materials because many properties depend strongly on atomic distances.

Examples:

## Bond Length

The equilibrium bond length determines:

* crystal stability,
* elastic properties,
* chemical bonding.

## Coordination Environment

The number and distance of neighboring atoms influence:

* oxidation state,
* electronic structure,
* magnetic behavior.

## Defects

Vacancies and substitutions modify local distance distributions.

The RBF expansion allows SchNet to capture these subtle structural variations automatically.

---

# Summary

Radial Basis Functions provide the bridge between continuous atomic geometry and neural network learning. Instead of supplying raw interatomic distances directly, SchNet expands each distance into a high-dimensional vector using Gaussian basis functions. This richer representation allows the Filter-Generating Network to learn complex distance-dependent interaction patterns.

The complete distance processing pipeline in SchNet is therefore:

```text
Atomic Coordinates

↓

Interatomic Distance

↓

Radial Basis Expansion

↓

Filter-Generating Network

↓

Continuous Interaction Filter

↓

CFConv Message Passing
```

This distance embedding mechanism is a key reason why SchNet can accurately model molecular and crystalline systems while maintaining smooth physical behavior.

Next, we will continue with **16.9 Interaction Block Mathematics**, where we will derive the complete mathematical operation of stacked SchNet interaction blocks and connect them to energy and force prediction.
