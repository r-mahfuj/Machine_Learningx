# Chapter 19 — CHGNet

# 19.1 Motivation

In the previous chapter, we studied **M3GNet**, one of the most influential universal graph neural networks for atomistic simulations.

M3GNet demonstrated that a machine learning model can simultaneously predict

* total energies,
* atomic forces,
* stress tensors,

with an accuracy approaching Density Functional Theory (DFT), while being several orders of magnitude faster.

This capability revolutionized computational materials science by enabling

* large-scale geometry optimization,
* long molecular dynamics simulations,
* high-throughput materials screening,
* accelerated materials discovery.

Despite these impressive achievements, M3GNet has important limitations.

Most notably,

**M3GNet learns only from atomic positions and atomic species.**

It does **not explicitly represent the electronic degrees of freedom** that govern chemical bonding.

As a result,

certain classes of materials remain difficult to model accurately, including

* strongly correlated materials,
* magnetic materials,
* transition-metal oxides,
* systems with significant charge transfer,
* mixed-valence compounds,
* spin-polarized systems.

These limitations motivated the development of a new generation of universal machine learning interatomic potentials.

One of the most important among them is

**CHGNet
(Charge-informed Graph Neural Network).**

---

# 19.2 Why M3GNet Is Not Always Enough

To understand why CHGNet was developed,

we must first understand the limitations of M3GNet.

Recall that M3GNet represents a crystal as a graph.

```text id="chg1"
Atoms

↓

Nodes

↓

Graph Neural Network

↓

Energy
```

The neural network receives information about

* atom types,
* atomic positions,
* local geometry.

However,

it has no explicit knowledge of

* electron density,
* local charge,
* magnetic moments,
* oxidation states.

Instead,

it must infer these quantities indirectly from the atomic arrangement.

For many materials,

this approximation works remarkably well.

For others,

it becomes insufficient.

---

# 19.3 The Importance of Electronic Structure

Chemical bonding is fundamentally determined by electrons.

Two structures may possess identical atomic coordinates,

yet differ dramatically because of

* different charge distributions,
* different oxidation states,
* different spin configurations.

For example,

consider iron.

```text id="chg2"
Fe²⁺

Fe³⁺
```

The two ions have

* different numbers of electrons,
* different magnetic moments,
* different bonding behavior.

If the model cannot distinguish these electronic states,

its predictions may become less accurate.

---

# 19.4 Charge Transfer in Materials

Many important materials exhibit significant charge transfer.

Examples include

* battery cathodes,
* solid electrolytes,
* transition-metal oxides,
* catalysts.

Consider a simple ionic crystal.

```text id="chg3"
Na

↓

Na⁺

Cl

↓

Cl⁻
```

Although sodium and chlorine begin as neutral atoms,

electrons transfer from sodium to chlorine.

The resulting electrostatic interactions dominate the material's properties.

A model that explicitly incorporates charge information is therefore expected to provide improved accuracy.

---

# 19.5 Magnetic Materials

Another challenge arises from magnetism.

Many technologically important materials are magnetic.

Examples include

* iron,
* cobalt,
* nickel,
* chromium oxides,
* manganese oxides,
* rare-earth compounds.

Their properties depend strongly on

* spin polarization,
* magnetic ordering,
* local magnetic moments.

Traditional graph neural networks generally do not represent these magnetic quantities explicitly.

CHGNet was designed to address this limitation.

---

# 19.6 Motivation Behind CHGNet

The central idea behind CHGNet is remarkably simple.

Instead of learning only atomic geometry,

the neural network should also learn information about

* local charge,
* magnetic moments,
* electronic environment.

Conceptually,

```text id="chg4"
Crystal Structure

↓

Graph Representation

↓

Charge Information

↓

Graph Neural Network

↓

Energy

Forces

Stress

Charges

Magnetic Moments
```

This richer representation enables the network to model more complicated chemical systems.

---

# 19.7 What Does CHGNet Stand For?

CHGNet stands for

**Charge-informed Graph Neural Network.**

The name reflects its primary innovation.

Instead of treating atoms purely as geometric objects,

CHGNet incorporates information related to the local electronic environment.

Importantly,

the model does **not** solve the Schrödinger equation.

Instead,

it learns electronic information directly from large DFT datasets.

---

# 19.8 The Main Goal of CHGNet

The primary objectives of CHGNet are

* improve energy prediction,
* improve force prediction,
* improve stress prediction,
* predict magnetic moments,
* capture charge redistribution,
* improve transferability across chemical space.

Thus,

CHGNet extends the philosophy of M3GNet rather than replacing it.

---

# 19.9 Comparison Between M3GNet and CHGNet

The conceptual difference can be summarized simply.

### M3GNet

```text id="chg5"
Atomic Structure

↓

Graph

↓

Energy

Forces

Stress
```

### CHGNet

```text id="chg6"
Atomic Structure

↓

Graph

↓

Charge-Aware Representation

↓

Energy

Forces

Stress

Magnetic Moments
```

Both models predict atomic interactions,

but CHGNet contains additional information describing the electronic state of the material.

---

# 19.10 Why Charge Information Improves Predictions

Suppose two neighboring atoms possess identical positions.

Without charge information,

the neural network only observes

```text id="chg7"
Distance

Element Types
```

With charge information,

it additionally knows

```text id="chg8"
Distance

Element Types

↓

Charge Distribution

↓

Magnetic State
```

This richer description allows the network to distinguish environments that appear geometrically identical but differ electronically.

Consequently,

energy and force predictions become more physically meaningful.

---

# 19.11 Relationship Between CHGNet and M3GNet

CHGNet should not be viewed as an entirely different model.

Instead,

it represents a natural evolution.

The progression is

```text id="chg9"
CGCNN

↓

MEGNet

↓

M3GNet

↓

CHGNet
```

Each generation incorporates increasingly sophisticated physical information.

* CGCNN introduced crystal graph neural networks.
* MEGNet incorporated global state variables.
* M3GNet introduced three-body interactions and universal interatomic potentials.
* CHGNet incorporated charge-informed representations and magnetic information.

Thus,

CHGNet builds upon the ideas developed throughout the evolution of graph neural networks for materials science.

---

# 19.12 Applications of CHGNet

Because of its improved representation,

CHGNet is particularly valuable for

* transition-metal oxides,
* battery materials,
* magnetic materials,
* ionic conductors,
* strongly correlated compounds,
* defect calculations,
* molecular dynamics,
* crystal relaxation,
* high-throughput materials discovery.

These applications will be explored in detail throughout the remainder of this chapter.

---

## Transition to the Next Section

We now understand **why CHGNet was developed** and how it extends the ideas introduced by M3GNet. The next step is to examine **the architecture of CHGNet itself**.

In the next section, we will study how CHGNet constructs its graph representation, incorporates charge and magnetic information, performs message passing, and predicts energies, forces, stresses, and magnetic moments within a unified graph neural network framework.

# 19.13 The Architecture of CHGNet

Having understood the motivation behind CHGNet, we now examine **how the model is constructed**.

Although CHGNet shares many similarities with M3GNet, several important architectural innovations distinguish it.

At a high level, the workflow is

```text id="chg_arch1"
Crystal Structure

↓

Graph Construction

↓

Atom Features

↓

Bond Features

↓

Angle Features

↓

Charge-aware Message Passing

↓

Atomic Representations

↓

Energy

Forces

Stress

Magnetic Moments
```

Like M3GNet,

CHGNet is an **atomistic graph neural network**.

However, unlike earlier models, it incorporates **latent electronic information** that allows the learned atomic representations to better reflect the local chemical environment.

---

# 19.14 Input to CHGNet

The input is the crystal structure itself.

This includes

* lattice vectors,
* atomic coordinates,
* atomic species,
* periodic boundary conditions.

Importantly,

the user **does not** need to supply

* atomic charges,
* oxidation states,
* magnetic moments.

Instead,

CHGNet learns these quantities implicitly during training.

Conceptually,

```text id="chg_arch2"
Crystal

↓

Atomic Positions

Atomic Species

↓

Graph
```

The electronic descriptors emerge naturally from the learned representations.

---

# 19.15 Graph Construction

As in previous graph neural networks,

the crystal is converted into a graph.

Every atom becomes a node.

```text id="chg_arch3"
Atom

↓

Node
```

Neighboring atoms are connected by edges.

```text id="chg_arch4"
Atom A

────────

Atom B

↓

Edge
```

Periodic boundary conditions are handled automatically,

allowing the graph to represent an infinite crystal.

---

# 19.16 Node Features

Each node initially represents an atom.

Its feature vector includes information related to

* atomic identity,
* chemical properties,
* learned embeddings.

Instead of using handcrafted descriptors,

CHGNet employs trainable embeddings.

Conceptually,

```text id="chg_arch5"
Element

↓

Embedding Vector

↓

Node Feature
```

During training,

these embeddings evolve so that chemically similar elements acquire similar numerical representations.

---

# 19.17 Edge Features

Each edge describes the relationship between neighboring atoms.

Typical edge information includes

* bond distance,
* relative atomic positions,
* geometric encoding.

For atoms

$i$

and

$j$,

the edge stores information derived from

$$
\mathbf{r}_{ij}
=

\mathbf{r}_j

\mathbf{r}_i.
$$

Distance information plays a central role in determining interatomic interactions.

---

# 19.18 Angular Features

Like M3GNet,

CHGNet includes three-body interactions.

Instead of considering only pairs of atoms,

it also examines bond angles.

For three neighboring atoms,

```text id="chg_arch6"
A

 \

  \

   B

  /

 /

C
```

the angle

$$
\theta_{ABC}
$$

provides additional geometric information.

These angular interactions improve the description of directional bonding.

---

# 19.19 Why Three-Body Information Matters

Many materials exhibit highly directional chemical bonds.

Examples include

* silicon,
* diamond,
* quartz,
* transition-metal oxides.

Two neighboring atoms may have identical bond lengths,

yet entirely different bond angles.

Since bond angles strongly influence chemical bonding,

incorporating three-body information significantly improves prediction accuracy.

---

# 19.20 Learned Atomic States

The most important concept in CHGNet is that **atomic feature vectors continuously evolve during message passing**.

Initially,

an atom is represented primarily by its chemical identity.

After several graph convolution layers,

its representation contains information about

* neighboring atoms,
* local coordination,
* chemical environment,
* electronic environment.

Conceptually,

```text id="chg_arch7"
Element

↓

Neighbor Information

↓

Local Environment

↓

Learned Atomic State
```

These learned representations become increasingly informative as message passing proceeds.

---

# 19.21 Message Passing in CHGNet

Graph neural networks operate by exchanging information between neighboring atoms.

One message-passing iteration consists of

```text id="chg_arch8"
Neighbor Features

↓

Message Construction

↓

Aggregation

↓

Node Update
```

This procedure is repeated multiple times.

Each iteration expands the amount of structural information available to every atom.

---

# 19.22 Charge-Informed Representations

This is where CHGNet differs fundamentally from M3GNet.

Instead of learning only geometric relationships,

the hidden atomic representations also encode information correlated with

* charge redistribution,
* oxidation state,
* magnetic behavior.

Importantly,

these quantities are **not manually specified**.

They emerge naturally from training against DFT reference data.

---

# 19.23 Latent Electronic Information

The term **latent** refers to information that is not directly observable but is represented internally by the neural network.

Conceptually,

```text id="chg_arch9"
Crystal

↓

Hidden Features

↓

Electronic Environment

↓

Predictions
```

Although users cannot directly inspect these hidden vectors,

they contain information that allows the model to distinguish chemically different atomic environments.

---

# 19.24 Multi-Task Learning

One of CHGNet's major innovations is its **multi-task learning** framework.

Instead of predicting only one quantity,

the model simultaneously learns several related physical properties.

These include

* total energy,
* atomic forces,
* stress tensor,
* magnetic moments.

Conceptually,

```text id="chg_arch10"
Shared Graph Network

↓

Energy

Forces

Stress

Magnetic Moments
```

The shared representation improves learning because these physical quantities are strongly related.

---

# 19.25 Why Multi-Task Learning Helps

Energy,

forces,

stress,

and magnetic moments are not independent.

For example,

changing atomic positions affects

* total energy,
* atomic forces,
* local magnetic moments.

Learning these quantities together allows the neural network to develop a more physically consistent internal representation.

As a result,

prediction accuracy often improves compared with training separate models for each property.

---

# 19.26 Predicting Total Energy

The primary prediction remains the total energy.

After message passing,

each atom possesses an updated feature vector.

An atomic energy contribution is predicted,

and the total energy is obtained by summing over all atoms.

Conceptually,

```text id="chg_arch11"
Atom 1

↓

Energy

+

Atom 2

↓

Energy

+

⋯

↓

Total Energy
```

This construction ensures that the total energy scales naturally with system size.

---

# 19.27 Predicting Atomic Forces

Once the energy has been determined,

atomic forces are obtained from the energy gradient.

$$
\mathbf{F}_i
============

*

\nabla_i
E.
$$

Because CHGNet is fully differentiable,

automatic differentiation computes these gradients efficiently.

Consequently,

force predictions remain physically consistent with the predicted energy surface.

---

# 19.28 Predicting the Stress Tensor

CHGNet also predicts the stress tensor,

which describes the response of the crystal to deformation.

The stress tensor enables

* constant-pressure molecular dynamics,
* elastic property calculations,
* structural optimization under applied pressure.

Stress prediction is one of the defining characteristics of modern universal interatomic potentials.

---

# 19.29 Predicting Magnetic Moments

Perhaps the most distinctive capability of CHGNet is its prediction of **atomic magnetic moments**.

For each atom,

the model estimates a local magnetic moment,

which provides information about the electronic spin state.

This capability is particularly valuable for

* magnetic materials,
* transition-metal oxides,
* spin-polarized systems.

Earlier universal graph neural networks generally lacked this functionality.

---

# 19.30 Overall Computational Pipeline

The complete CHGNet workflow can be summarized as

```text id="chg_arch12"
Crystal Structure

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

Angular Features

↓

Message Passing

↓

Charge-informed Representation

↓

Energy

Forces

Stress

Magnetic Moments
```

Every prediction originates from the same learned graph representation,

making CHGNet a unified model for multiple atomistic simulation tasks.

---

## Transition to the Next Section

Now that we understand the architecture of CHGNet, the next step is to examine **how the model is trained**. We will study the large DFT datasets used during training, the role of the **Materials Project trajectory (MPtrj) dataset**, the loss functions for energy, forces, stress, and magnetic moments, and how CHGNet learns transferable representations capable of generalizing across a wide range of materials.

# 19.31 Training CHGNet

A neural network is only as good as the data on which it is trained.

Even the most sophisticated architecture cannot produce reliable predictions if the training data are

* inaccurate,
* inconsistent,
* too small,
* or chemically unrepresentative.

For this reason,

the success of CHGNet depends not only on its architecture but also on the **large-scale Density Functional Theory (DFT) dataset** used during training.

Unlike traditional interatomic potentials that are developed for a single material,

CHGNet is trained on an enormous collection of materials covering a wide range of

* chemical compositions,
* crystal structures,
* atomic environments,
* magnetic configurations,
* lattice distortions.

This diversity allows the model to become a **universal interatomic potential**.

---

# 19.32 Why Large Training Datasets Are Necessary

Suppose we train a neural network using only silicon crystals.

The resulting model may perform well for

* silicon,
* germanium,

because of their similar crystal structures.

However,

it will almost certainly fail for

* iron oxides,
* lithium phosphates,
* transition-metal nitrides.

The reason is simple.

The model has never encountered these chemical environments during training.

A universal model therefore requires a **chemically diverse dataset**.

---

# 19.33 Density Functional Theory as the Source of Truth

CHGNet does not learn directly from experiments.

Instead,

it learns from quantum mechanical calculations.

Conceptually,

```text id="chg_train1"
Crystal Structures

↓

Density Functional Theory

↓

Reference Data

↓

CHGNet Training
```

The DFT calculations provide

* total energies,
* atomic forces,
* stress tensors,
* magnetic moments.

These quantities serve as the target values during supervised learning.

---

# 19.34 The Materials Project Trajectory Dataset (MPtrj)

One of the major innovations behind CHGNet is the use of the **Materials Project trajectory dataset**, commonly abbreviated as **MPtrj**.

Unlike earlier datasets that stored only fully relaxed crystal structures,

MPtrj contains the **entire relaxation trajectories** produced during DFT geometry optimization.

Each trajectory includes many intermediate structures rather than only the final equilibrium configuration.

This provides substantially richer information for training.

---

# 19.35 Why Relaxation Trajectories Are Valuable

Consider a DFT geometry optimization.

The calculation begins from an initial structure,

which is generally not at equilibrium.

During optimization,

the atoms gradually move toward their equilibrium positions.

Conceptually,

```text id="chg_train2"
Initial Structure

↓

Step 1

↓

Step 2

↓

Step 3

↓

Final Relaxed Structure
```

Every intermediate step contains valuable information about

* energies,
* forces,
* stress,
* atomic rearrangements.

Instead of discarding these intermediate configurations,

CHGNet uses them for training.

---

# 19.36 Advantages of Trajectory Training

Training on relaxation trajectories provides several important advantages.

First,

the neural network observes atoms

* far from equilibrium,
* near equilibrium,
* and at equilibrium.

Consequently,

the model learns the entire potential energy surface rather than only its minimum.

Second,

it encounters a much larger variety of local atomic environments.

Third,

force prediction improves because the model sees many more non-equilibrium force configurations.

---

# 19.37 Diversity of the Training Data

The MPtrj dataset contains an enormous diversity of materials.

Examples include

* elemental solids,
* binary compounds,
* ternary compounds,
* oxides,
* nitrides,
* sulfides,
* phosphates,
* transition-metal compounds,
* magnetic materials.

This chemical diversity is one of the primary reasons for the strong transferability of CHGNet.

---

# 19.38 Training Targets

Each training example contains several physical quantities.

These include

* total energy,
* atomic forces,
* stress tensor,
* magnetic moments.

Conceptually,

```text id="chg_train3"
Crystal

↓

DFT

↓

Energy

Forces

Stress

Magnetic Moments
```

The neural network attempts to reproduce all of these quantities simultaneously.

---

# 19.39 Multi-Task Supervised Learning

Because several physical quantities are predicted simultaneously,

CHGNet employs **multi-task supervised learning**.

The shared graph representation is optimized using information from multiple prediction tasks.

Conceptually,

```text id="chg_train4"
Graph Representation

↓

Energy Loss

Force Loss

Stress Loss

Magnetic Loss

↓

Combined Optimization
```

Learning multiple related tasks encourages the network to develop more physically meaningful internal representations.

---

# 19.40 Energy Loss

The first component of the training objective measures the difference between

* predicted energy,
* DFT energy.

A common loss function is the Mean Squared Error (MSE),

$$
L_E
===

\frac{1}{N}
\sum_{i=1}^{N}
(E_i^{\mathrm{pred}}
--------------------

E_i^{\mathrm{DFT}})^2.
$$

Reducing this loss improves the accuracy of total energy prediction.

---

# 19.41 Force Loss

For molecular dynamics,

force prediction is even more important than energy prediction.

The force loss is commonly written as

$$
L_F
===

\frac{1}{3N}
\sum_{i=1}^{N}
|
\mathbf{F}_i^{\mathrm{pred}}
----------------------------

\mathbf{F}_i^{\mathrm{DFT}}
|^2.
$$

Because every atom contributes three force components,

force data provide a very large number of training labels.

---

# 19.42 Stress Loss

The stress tensor also contributes to the overall objective.

A typical stress loss is

$$
L_S
===

|
\sigma^{\mathrm{pred}}
----------------------

\sigma^{\mathrm{DFT}}
|^2.
$$

Including stress improves

* cell optimization,
* pressure prediction,
* elastic property calculations,
* NPT molecular dynamics.

---

# 19.43 Magnetic Moment Loss

One unique aspect of CHGNet is its prediction of local magnetic moments.

The magnetic loss compares

* predicted magnetic moments,
* DFT magnetic moments.

Conceptually,

```text id="chg_train5"
Predicted Spin

↓

Compare

↓

DFT Spin

↓

Error
```

Learning magnetic information improves performance for spin-polarized materials.

---

# 19.44 Combined Loss Function

The complete training objective combines all prediction tasks.

Conceptually,

$$
L
=

w_E L_E
+
w_F L_F
+
w_S L_S
+
w_M L_M,
$$

where

* $L_E$ is the energy loss,
* $L_F$ is the force loss,
* $L_S$ is the stress loss,
* $L_M$ is the magnetic moment loss,

and

* $w_E$,
* $w_F$,
* $w_S$,
* $w_M$

are weighting coefficients controlling the relative importance of each task.

During training,

the neural network minimizes this combined loss.

---

# 19.45 Optimization of Network Parameters

Training proceeds using gradient-based optimization.

The workflow is

```text id="chg_train6"
Forward Pass

↓

Prediction

↓

Loss Calculation

↓

Backpropagation

↓

Parameter Update
```

This process is repeated for millions of crystal configurations until the loss converges.

---

# 19.46 Generalization Through Diversity

One of the most important lessons from CHGNet is that **generalization depends heavily on dataset diversity**.

A neural network trained on

* many crystal structures,
* many chemistries,
* many magnetic states,
* many distorted geometries,

develops more transferable representations than one trained on a narrow set of materials.

Thus,

the quality and diversity of the training data are just as important as the architecture itself.

---

# 19.47 Training Workflow Summary

The complete CHGNet training process can be summarized as

```text id="chg_train7"
Materials Project Structures

↓

DFT Relaxation Trajectories

↓

Energy

Forces

Stress

Magnetic Moments

↓

Graph Construction

↓

CHGNet

↓

Loss Calculation

↓

Backpropagation

↓

Trained Model
```

After training,

the model can predict atomistic properties for previously unseen materials in only a fraction of the time required for new DFT calculations.

---

## Transition to the Next Section

We have now examined how CHGNet is trained using large DFT trajectory datasets and multi-task learning. In the next section, we will explore **how CHGNet performs practical atomistic simulations**, including geometry optimization, molecular dynamics, charge prediction, magnetic property prediction, and high-throughput materials screening, and compare its practical performance with M3GNet.

# 19.48 Energy Prediction with CHGNet

After training,

CHGNet becomes a universal function that maps an atomic structure to its physical properties.

The first and most fundamental prediction is the **total energy** of the system.

Energy prediction forms the basis of almost every atomistic simulation because many other physical quantities can be derived from the energy.

The computational workflow is

```text id="chg_energy1"
Crystal Structure

↓

Graph Construction

↓

CHGNet

↓

Total Energy
```

Unlike conventional machine learning models that predict only scalar material properties,

CHGNet predicts the **potential energy surface (PES)** of atomic systems.

---

# 19.49 Why Energy is the Most Important Quantity

In atomistic simulations,

the total energy determines

* structural stability,
* phase stability,
* equilibrium geometry,
* atomic forces,
* thermodynamic behavior.

Many seemingly different materials problems ultimately reduce to comparing energies.

For example,

consider two crystal structures.

```text id="chg_energy2"
Structure A

↓

Energy = -42.8 eV

Structure B

↓

Energy = -43.5 eV
```

Since Structure B has the lower energy,

it is generally more stable under identical conditions.

---

# 19.50 Energy as a Function of Atomic Coordinates

The total energy is not a fixed number.

Instead,

it depends continuously on the atomic positions.

Mathematically,

the energy can be viewed as

$$
E
=

E(\mathbf{R}),
$$

where

$$
\mathbf{R}
==========

(\mathbf{r}_1,\mathbf{r}_2,\ldots,\mathbf{r}_N)
$$

represents the coordinates of all atoms.

Moving even one atom changes the total energy.

Consequently,

CHGNet learns the multidimensional energy landscape of materials.

---

# 19.51 The Potential Energy Surface

The relationship between atomic positions and energy is called the **potential energy surface (PES)**.

Conceptually,

```text id="chg_energy3"
Energy

      /\

     /  \

____/    \____

Atomic Configuration
```

Valleys correspond to stable structures,

while hills represent unstable configurations.

A universal interatomic potential such as CHGNet attempts to reproduce this surface as accurately as possible.

---

# 19.52 Atomic Energy Decomposition

Like many modern graph neural network potentials,

CHGNet predicts the total energy by summing contributions from individual atoms.

Conceptually,

```text id="chg_energy4"
Atom 1

↓

E₁

+

Atom 2

↓

E₂

+

⋯

↓

Total Energy
```

Mathematically,

$$
E_{\mathrm{total}}
==================

\sum_i
E_i.
$$

Although these atomic energies are learned quantities rather than directly measurable physical observables,

this decomposition provides an efficient and scalable framework for modeling large systems.

---

# 19.53 Why Atomic Energy Decomposition Works

An atom does not exist in isolation inside a crystal.

Its contribution to the total energy depends strongly on

* neighboring atoms,
* bond lengths,
* bond angles,
* local electronic environment.

Therefore,

the atomic energy

$$
E_i
$$

is not a fixed property of an element.

Instead,

it is a function of the atom's surrounding environment.

CHGNet learns this relationship automatically through message passing.

---

# 19.54 Energy Prediction During Geometry Optimization

One of the primary uses of energy prediction is geometry optimization.

The workflow is

```text id="chg_energy5"
Initial Structure

↓

Predict Energy

↓

Predict Forces

↓

Move Atoms

↓

Lower Energy

↓

Repeat
```

Each optimization step attempts to reduce the total energy until equilibrium is reached.

---

# 19.55 Energy Prediction During Molecular Dynamics

In molecular dynamics,

energy is evaluated at every time step.

The simulation proceeds as

```text id="chg_energy6"
Current Structure

↓

CHGNet

↓

Energy

↓

Forces

↓

Next Structure

↓

Repeat
```

Although forces determine atomic motion,

the energy provides the underlying potential energy surface from which those forces are derived.

---

# 19.56 Energy Prediction for Large Systems

One of the major advantages of CHGNet is its computational efficiency.

Consider three different approaches.

| Method                 |                     Typical System Size |
| ---------------------- | --------------------------------------: |
| DFT                    |                       Hundreds of atoms |
| Classical Force Fields |                       Millions of atoms |
| CHGNet                 | Thousands to tens of thousands of atoms |

CHGNet occupies an attractive middle ground,

providing near-DFT accuracy with computational costs much closer to classical interatomic potentials.

---

# 19.57 High-Throughput Energy Evaluation

Large materials databases often contain

* hundreds of thousands,
* or millions,

of candidate structures.

Evaluating each structure using DFT is computationally expensive.

Instead,

CHGNet enables rapid energy evaluation.

```text id="chg_energy7"
Database

↓

CHGNet

↓

Energy Ranking

↓

Stable Candidates
```

This capability forms the basis of modern high-throughput computational materials discovery.

---

# 19.58 Formation Energy Prediction

One important application of energy prediction is the estimation of **formation energies**.

The formation energy measures the stability of a compound relative to its constituent elements.

A commonly used expression is

$$
\Delta E_f
==========

## E_{\mathrm{compound}}

\sum_i
n_i
\mu_i,
$$

where

* $E_{\mathrm{compound}}$ is the total energy of the compound,
* $n_i$ is the number of atoms of element $i$,
* $\mu_i$ is the reference chemical potential of element $i$.

Lower formation energies generally indicate greater thermodynamic stability.

---

# 19.59 Energy Ranking in Materials Discovery

Suppose a researcher generates ten possible crystal structures for the same chemical composition.

Each structure is evaluated by CHGNet.

```text id="chg_energy8"
Candidate Structures

↓

CHGNet

↓

Energy

↓

Sort

↓

Most Stable Structure
```

Only the structures with the lowest energies proceed to more expensive DFT calculations.

This strategy dramatically reduces computational cost.

---

# 19.60 Accuracy of CHGNet Energy Predictions

Extensive benchmarking has shown that CHGNet achieves energy prediction errors that are typically close to those of state-of-the-art universal machine-learned interatomic potentials for many classes of crystalline materials.

Its performance is particularly strong because

* it is trained on a large and chemically diverse DFT dataset,
* it incorporates three-body geometric information,
* it learns charge-informed representations,
* it performs multi-task learning using energies, forces, stresses, and magnetic moments simultaneously.

The exact prediction error depends on the material class and whether similar chemical environments were represented during training.

---

# 19.61 Practical Applications of Energy Prediction

Energy prediction with CHGNet supports a wide variety of research applications.

These include

* geometry optimization,
* crystal structure prediction,
* high-throughput screening,
* phase stability analysis,
* defect calculations,
* molecular dynamics,
* phase transition studies,
* battery material discovery.

In nearly all of these applications,

the predicted energy serves as the foundation upon which further atomistic simulations are built.

---

## Transition to the Next Section

Although total energy is the most fundamental quantity predicted by CHGNet, **atomic forces** determine how atoms actually move and relax. In the next section, we will study **force prediction**, explain its relationship to the energy gradient, examine how CHGNet computes physically consistent forces through automatic differentiation, and explore why accurate force prediction is even more critical than energy prediction for geometry optimization and molecular dynamics.

# 19.62 Force Prediction with CHGNet

While total energy determines the stability of a material,

it is the **atomic forces** that determine how the atoms actually move.

Every geometry optimization,

every molecular dynamics simulation,

and every structural relaxation depends fundamentally on accurate force prediction.

For this reason,

modern machine-learned interatomic potentials are trained not only to reproduce DFT energies but also to reproduce **DFT atomic forces**.

Among all prediction targets,

force prediction is arguably the most important.

---

# 19.63 What Is an Atomic Force?

A force represents the tendency of an atom to move.

Consider an atom inside a crystal.

If the atom is exactly at its equilibrium position,

the surrounding atoms pull equally in every direction.

```text id="chg_force1"
      ●

   ●  ○  ●

      ●
```

The net force is

```text id="chg_force2"
0 eV/Å
```

The atom remains stationary.

Now suppose the atom is displaced slightly.

```text id="chg_force3"
      ●

   ●    ○

      ●
```

The neighboring atoms no longer pull equally.

A restoring force appears,

driving the atom back toward equilibrium.

---

# 19.64 Force as the Gradient of Energy

One of the most important relationships in atomistic simulations is

$$
\mathbf{F}_i
============

*

\nabla_i E.
$$

Here,

* $\mathbf{F}_i$ is the force acting on atom $i$,
* $E$ is the total potential energy,
* $\nabla_i$ denotes the gradient with respect to the coordinates of atom $i$.

This equation shows that

**forces are not independent quantities**.

They are determined entirely by the shape of the potential energy surface.

---

# 19.65 Physical Interpretation of the Negative Sign

The negative sign has a simple physical meaning.

Atoms naturally move toward **lower energy**.

Suppose moving an atom to the right increases the energy.

```text id="chg_force4"
Higher Energy

────────►
```

The force points in the opposite direction.

```text id="chg_force5"
◄────────

Force
```

Thus,

the force always acts to reduce the potential energy.

---

# 19.66 Why Accurate Forces Matter More Than Accurate Energies

It may seem surprising,

but in many atomistic simulations,

force accuracy is even more important than energy accuracy.

Consider two machine learning models.

### Model A

Very accurate energy

Poor force prediction

### Model B

Slightly less accurate energy

Excellent force prediction

For

* geometry optimization,
* molecular dynamics,
* phonon calculations,

Model B is generally the better choice.

The reason is that atomic motion depends directly on the forces,

not on the absolute energy value.

---

# 19.67 Force Prediction During Geometry Optimization

Geometry optimization proceeds by repeatedly following the predicted forces.

The workflow is

```text id="chg_force6"
Initial Structure

↓

CHGNet

↓

Forces

↓

Move Atoms

↓

Lower Energy

↓

Repeat
```

The optimization terminates when all atomic forces become sufficiently small.

---

# 19.68 Force Prediction During Molecular Dynamics

Molecular dynamics requires force evaluation at every integration step.

The sequence is

```text id="chg_force7"
Atomic Positions

↓

CHGNet

↓

Forces

↓

Newton's Equation

↓

Updated Positions

↓

Repeat
```

Since thousands or millions of force evaluations may be required,

both speed and accuracy are essential.

---

# 19.69 Force Prediction Through Automatic Differentiation

CHGNet does **not** train a completely separate neural network for forces.

Instead,

forces are obtained by differentiating the predicted energy.

Conceptually,

```text id="chg_force8"
Graph Neural Network

↓

Energy

↓

Automatic Differentiation

↓

Forces
```

Because the neural network is fully differentiable,

modern deep learning frameworks such as PyTorch compute these gradients automatically.

---

# 19.70 Why Automatic Differentiation Is Important

Automatic differentiation guarantees that

* energy,
* forces,

remain mathematically consistent.

If forces were predicted independently,

they might violate fundamental physical relationships.

For example,

the predicted force could point toward increasing energy,

which would be physically impossible.

Using

$$
\mathbf{F}
==========

*

\nabla E
$$

eliminates this problem.

---

# 19.71 Conservative Forces

Forces obtained from an energy gradient are called **conservative forces**.

Conservative forces satisfy several important physical properties.

They

* conserve energy in NVE simulations,
* produce stable molecular dynamics,
* correspond to a physically meaningful potential energy surface.

This consistency is one of the defining characteristics of modern machine-learned interatomic potentials.

---

# 19.72 Training Force Predictions

During training,

CHGNet compares its predicted forces with DFT reference forces.

For every atom,

the prediction error contributes to the force loss.

Conceptually,

```text id="chg_force9"
Predicted Force

↓

Compare

↓

DFT Force

↓

Error
```

The neural network parameters are updated until this error becomes as small as possible.

---

# 19.73 Why Force Data Are So Valuable

A single DFT calculation produces

* one total energy,

but

* three force components for every atom.

For a structure containing

```text id="chg_force10"
100 atoms
```

the calculation provides

```text id="chg_force11"
300 force values
```

in addition to

one energy.

Consequently,

force data greatly enrich the training dataset and improve the quality of the learned potential energy surface.

---

# 19.74 Force Prediction for Large Systems

One of CHGNet's major strengths is that force prediction scales efficiently with system size.

Instead of solving the Kohn–Sham equations at every molecular dynamics step,

the model performs only

* graph construction,
* message passing,
* automatic differentiation.

This dramatically reduces computational cost,

making simulations of thousands of atoms practical.

---

# 19.75 Typical Applications of Force Prediction

Accurate force prediction enables

* geometry optimization,
* molecular dynamics,
* defect relaxation,
* diffusion simulations,
* phonon calculations,
* phase transition studies,
* thermal expansion,
* crystal structure prediction.

In all of these applications,

the quality of the simulation depends directly on the accuracy of the predicted forces.

---

# 19.76 Force Prediction Example

Suppose a lithium atom occupies an unstable position inside a solid electrolyte.

Initially,

```text id="chg_force12"
Li

↓

Large Force
```

After several optimization steps,

the force decreases.

```text id="chg_force13"
Li

↓

Small Force
```

Eventually,

```text id="chg_force14"
Li

↓

≈ 0 eV/Å
```

The atom has reached its equilibrium position.

This simple example illustrates how force prediction drives structural relaxation.

---

# 19.77 Relationship Between Energy and Forces

Energy and forces should never be viewed as separate predictions.

Instead,

they describe different aspects of the same potential energy surface.

Conceptually,

```text id="chg_force15"
Potential Energy Surface

↓

Energy

↓

Gradient

↓

Forces

↓

Atomic Motion
```

A physically accurate machine-learned potential must reproduce both quantities simultaneously.

---

# 19.78 Advantages of CHGNet Force Prediction

Compared with repeated DFT calculations,

CHGNet provides several important advantages.

* Near-DFT force accuracy for many crystalline materials.
* Orders-of-magnitude faster force evaluation.
* Efficient simulations of large supercells.
* Consistent energy–force relationships through automatic differentiation.
* Practical molecular dynamics over nanosecond time scales.

These capabilities make CHGNet particularly valuable for large-scale atomistic simulations.

---

## Transition to the Next Section

We have now examined how CHGNet predicts **energies** and **forces**, the two quantities that define the potential energy surface and govern atomic motion. In the next section, we will study **stress prediction**, explaining how CHGNet computes the stress tensor, why stress is essential for pressure-controlled simulations, elastic property calculations, and variable-cell geometry optimization, and how it extends the capabilities of universal machine-learned interatomic potentials beyond simple force prediction.

# 19.79 Stress Prediction with CHGNet

While atomic forces describe the motion of individual atoms,

the **stress tensor** describes the mechanical state of the entire crystal.

Stress prediction is essential for simulations involving

* pressure,
* mechanical deformation,
* lattice optimization,
* elastic properties,
* phase transformations.

Without stress prediction,

the simulation cell itself cannot respond correctly to external loading.

For this reason,

stress prediction is one of the defining features of modern universal machine-learned interatomic potentials such as CHGNet.

---

# 19.80 What Is Stress?

Suppose a cube of material is compressed.

```text id="chg_stress1"
↓

██████

██████

██████

↑
```

The atoms become closer together,

and internal forces develop that oppose further compression.

These internal forces are collectively described by the **stress tensor**.

Unlike atomic forces,

which act on individual atoms,

stress is a property of the entire material.

---

# 19.81 Stress Versus Force

Although closely related,

force and stress are fundamentally different.

| Force                      | Stress                          |
| -------------------------- | ------------------------------- |
| Acts on an individual atom | Acts on the entire material     |
| Vector quantity            | Second-order tensor             |
| Units: eV/Å or N           | Units: GPa or Pa                |
| Controls atomic motion     | Controls mechanical deformation |

Both quantities are required for realistic atomistic simulations.

---

# 19.82 Mathematical Definition of Stress

For a simple uniaxial system,

stress is defined as

$$
\sigma
======

\frac{F}{A},
$$

where

* $F$ is the applied force,
* $A$ is the cross-sectional area.

This equation shows that the same force produces different stresses depending on the area over which it acts.

---

# 19.83 The Stress Tensor

Real crystals experience forces in three dimensions.

Consequently,

stress cannot be represented by a single number.

Instead,

it is written as a **tensor**

$$
\boldsymbol{\sigma}
===================

\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} & \sigma_{xz} \
\sigma_{yx} & \sigma_{yy} & \sigma_{yz} \
\sigma_{zx} & \sigma_{zy} & \sigma_{zz}
\end{bmatrix}.
$$

Each element describes how forces act along different directions.

---

# 19.84 Physical Meaning of the Tensor Components

The diagonal elements

$$
\sigma_{xx},
\quad
\sigma_{yy},
\quad
\sigma_{zz}
$$

represent **normal stresses**.

These correspond to

* compression,
* tension.

The off-diagonal elements

$$
\sigma_{xy},
\sigma_{xz},
\sigma_{yz}
$$

represent **shear stresses**.

These describe forces that tend to distort the crystal shape.

---

# 19.85 Why Stress Prediction Matters

Many important simulations require accurate stress prediction.

Examples include

* constant-pressure molecular dynamics,
* crystal structure relaxation,
* elastic constant calculations,
* thermal expansion,
* phase transitions,
* mechanical property prediction.

Without stress information,

the simulation cell would remain artificially fixed.

---

# 19.86 Variable Cell Relaxation

Geometry optimization involves two different types of relaxation.

### Atomic relaxation

Only atoms move.

```text id="chg_stress2"
Cell

Fixed

↓

Atoms Move
```

### Cell relaxation

Both

* atoms,
* lattice vectors,

change simultaneously.

```text id="chg_stress3"
Cell

↓

Changes Shape

↓

Atoms Relax
```

The second case requires stress prediction.

---

# 19.87 Constant-Pressure Simulations

Suppose we wish to simulate a material at

```text id="chg_stress4"
10 GPa
```

The simulation box must continuously expand or contract to maintain this pressure.

The workflow becomes

```text id="chg_stress5"
Atomic Positions

↓

CHGNet

↓

Stress Tensor

↓

Adjust Cell

↓

Repeat
```

This procedure is impossible without accurate stress prediction.

---

# 19.88 Stress During Molecular Dynamics

In NPT molecular dynamics,

both

* atomic positions,
* simulation cell,

evolve with time.

The algorithm proceeds as

```text id="chg_stress6"
Current Structure

↓

CHGNet

↓

Forces

Stress

↓

Update Atoms

↓

Update Cell

↓

Next Step
```

This enables realistic simulations under laboratory conditions where pressure remains constant.

---

# 19.89 Stress Prediction During Training

During supervised learning,

CHGNet compares its predicted stress tensor with the DFT stress tensor.

Conceptually,

```text id="chg_stress7"
Predicted Stress

↓

Compare

↓

DFT Stress

↓

Loss
```

Reducing this loss improves the model's ability to predict mechanical behavior.

---

# 19.90 Stress and Elastic Properties

Stress prediction enables direct calculation of elastic constants.

The workflow is

```text id="chg_stress8"
Relax Crystal

↓

Apply Small Strain

↓

Predict Stress

↓

Stress–Strain Curve

↓

Elastic Constants
```

From these calculations,

researchers obtain

* Young's modulus,
* bulk modulus,
* shear modulus,
* Poisson's ratio.

---

# 19.91 Stress During Phase Transitions

During a phase transition,

the crystal lattice often changes shape.

Examples include

* cubic → tetragonal,
* tetragonal → orthorhombic,
* solid → liquid.

Stress prediction allows the simulation cell to deform naturally during these transformations,

making the simulations much more realistic.

---

# 19.92 Internal Pressure

Even without external loading,

a crystal may possess **internal stress**.

For example,

after constructing a defective supercell,

the atoms are generally not in equilibrium.

```text id="chg_stress9"
Defect

↓

Internal Stress

↓

Relaxation
```

Geometry optimization removes these residual stresses,

leading to the equilibrium structure.

---

# 19.93 High-Pressure Materials Research

Many materials exist only under extreme pressures.

Examples include

* superhard materials,
* deep-Earth minerals,
* high-pressure ice phases,
* compressed hydrogen.

Studying these systems requires repeated stress calculations.

Since CHGNet predicts stress efficiently,

it enables rapid exploration of pressure-induced phase diagrams.

---

# 19.94 Mechanical Stability

Stress prediction also helps determine whether a crystal is mechanically stable.

A mechanically unstable structure may exhibit

* very large internal stresses,
* spontaneous lattice distortions,
* unstable phonon modes,
* structural collapse.

Monitoring the predicted stress tensor provides an early indication of such instabilities.

---

# 19.95 Practical Importance of Stress Prediction

The ability to predict stress efficiently benefits many research areas.

Examples include

* battery electrode expansion,
* pressure-induced phase transitions,
* thermal expansion,
* mechanical testing,
* thin-film strain engineering,
* crystal growth.

These applications extend far beyond simple energy prediction.

---

# 19.96 Advantages of CHGNet Stress Prediction

Compared with repeated DFT calculations,

CHGNet offers several advantages.

* Fast prediction of the complete stress tensor.
* Efficient variable-cell geometry optimization.
* Practical NPT molecular dynamics.
* Rapid evaluation of elastic properties.
* High-throughput mechanical screening.

Together with energy and force prediction,

stress prediction completes the essential ingredients required for realistic atomistic simulations.

---

## Transition to the Next Section

Energy, forces, and stress describe the mechanical behavior of materials. However, one of the defining innovations of CHGNet is its ability to learn **electronic information**, particularly **local magnetic moments** and **charge-related representations**. In the next section, we will examine how CHGNet predicts magnetic moments, why magnetism is crucial for many transition-metal compounds, and how charge-informed learning distinguishes CHGNet from earlier universal graph neural network potentials.

# 19.97 Charge-Informed Learning in CHGNet

The defining innovation of CHGNet is contained in its name:

**Charge-informed Graph Neural Network.**

Unlike earlier graph neural networks that primarily learn from

* atomic species,
* atomic positions,
* local geometry,

CHGNet is designed to learn representations that are closely related to the **electronic environment** of each atom.

This allows the model to better describe materials in which chemical bonding depends strongly on

* charge transfer,
* oxidation states,
* local magnetic moments,
* electron redistribution.

Rather than treating atoms simply as points connected by bonds,

CHGNet attempts to encode how electrons are distributed throughout the crystal.

---

# 19.98 What Does "Charge-Informed" Mean?

The term **charge-informed** does **not** mean that the user must provide atomic charges as input.

Instead,

the model learns internal representations that correlate with charge-related properties observed in DFT calculations.

Conceptually,

```text id="chg_charge1"
Crystal Structure

↓

Graph Neural Network

↓

Latent Electronic Representation

↓

Energy

Forces

Stress

Magnetic Moments
```

The electronic information is therefore learned rather than explicitly prescribed.

---

# 19.99 Charge Distribution in Materials

Electrons are rarely distributed uniformly throughout a material.

Instead,

they continuously rearrange in response to

* neighboring atoms,
* bond formation,
* crystal symmetry,
* local chemical environments.

For example,

consider sodium chloride.

Before bonding,

```text id="chg_charge2"
Na

Cl
```

After bonding,

```text id="chg_charge3"
Na⁺

Cl⁻
```

An electron has moved from sodium to chlorine.

This redistribution changes

* bonding,
* electrostatic interactions,
* stability,
* mechanical properties.

---

# 19.100 Why Geometry Alone Is Not Enough

Suppose two structures possess identical atomic positions.

```text id="chg_charge4"
Structure A

↓

Same Geometry

↓

Structure B
```

If one structure contains

* Fe²⁺

while the other contains

* Fe³⁺,

their properties may differ substantially.

Geometry alone cannot distinguish these situations.

The electronic state must also be represented.

This observation motivated the development of CHGNet.

---

# 19.101 Oxidation States and Bonding

The oxidation state of an atom influences

* bond strength,
* bond length,
* magnetic behavior,
* chemical reactivity.

For example,

iron can exist as

* Fe²⁺,
* Fe³⁺.

Although both are iron atoms,

their electronic configurations differ.

Consequently,

their interactions with neighboring oxygen atoms also differ.

A universal interatomic potential should therefore distinguish between these environments.

---

# 19.102 Local Chemical Environment

Every atom experiences a unique local environment.

Consider a lithium ion inside a battery cathode.

Its environment depends on

* neighboring oxygen atoms,
* neighboring transition metals,
* lattice distortions,
* local charge redistribution.

Even two lithium atoms within the same crystal may occupy chemically different environments.

CHGNet learns to represent these differences automatically.

---

# 19.103 Latent Electronic Features

The neural network does not explicitly compute

* electron density,
* wavefunctions,
* molecular orbitals.

Instead,

it constructs **latent feature vectors**.

These vectors contain numerical information that correlates strongly with

* charge transfer,
* oxidation state,
* magnetic behavior,
* local bonding.

Conceptually,

```text id="chg_charge5"
Atomic Environment

↓

Hidden Features

↓

Electronic Information
```

These hidden vectors become increasingly informative during training.

---

# 19.104 Charge Redistribution During Relaxation

As atoms move,

their electronic environment also changes.

For example,

during structure relaxation,

```text id="chg_charge6"
Initial Structure

↓

Atomic Motion

↓

Bond Rearrangement

↓

Charge Redistribution

↓

Relaxed Structure
```

A model that learns charge-aware representations can adapt naturally to these evolving electronic environments.

---

# 19.105 Why Charge Information Improves Energy Prediction

The total energy of a material depends on

* electrostatic interactions,
* covalent bonding,
* metallic bonding,
* ionic bonding.

All of these interactions involve electrons.

Consequently,

incorporating charge-related information enables more accurate prediction of the potential energy surface.

This is especially important for

* ionic compounds,
* transition-metal oxides,
* battery materials,
* catalysts.

---

# 19.106 Charge Transfer in Battery Materials

Rechargeable batteries provide an excellent example.

During charging,

lithium ions leave the cathode.

At the same time,

electrons redistribute throughout the transition-metal framework.

Conceptually,

```text id="chg_charge7"
Li Extraction

↓

Electron Redistribution

↓

Oxidation State Changes

↓

Energy Changes
```

Capturing these electronic changes is essential for accurate atomistic simulations.

---

# 19.107 Charge-Aware Message Passing

In conventional graph neural networks,

messages exchanged between neighboring atoms primarily describe

* geometry,
* distances,
* bond angles.

In CHGNet,

the learned representations also encode information related to the local electronic environment.

Conceptually,

```text id="chg_charge8"
Neighbor Atoms

↓

Geometry

+

Electronic Context

↓

Updated Node Features
```

This richer information improves the quality of the learned atomic embeddings.

---

# 19.108 Relationship Between Charge and Bond Strength

Bond strength depends strongly on electron distribution.

For example,

greater electron sharing often produces stronger covalent bonds,

while large charge separation leads to stronger ionic interactions.

Consequently,

two bonds of identical length may possess different strengths because their electronic environments differ.

A charge-informed model can distinguish between such cases.

---

# 19.109 Improved Transferability

One consequence of learning electronic information is improved transferability.

Suppose the neural network encounters a material it has never seen before.

Although the exact composition may be unfamiliar,

its electronic environment may resemble environments observed during training.

Because CHGNet learns these underlying electronic patterns,

it often generalizes more effectively across diverse chemical systems than models relying solely on geometric information.

---

# 19.110 Electronic Information Without Solving Quantum Mechanics

An important distinction should be emphasized.

CHGNet does **not** solve

* the Schrödinger equation,
* Kohn–Sham equations,
* Hartree–Fock equations.

Instead,

it learns statistical relationships between

* atomic structure,
* electronic environment,
* physical properties,

using DFT-generated reference data.

Thus,

CHGNet provides an efficient approximation to quantum mechanical behavior rather than replacing first-principles electronic structure calculations.

---

# 19.111 Advantages of Charge-Informed Learning

Charge-informed representations improve the description of

* ionic compounds,
* magnetic materials,
* transition-metal oxides,
* battery electrodes,
* catalysts,
* chemically complex materials.

Compared with purely geometry-based graph neural networks,

they provide a richer and more physically meaningful representation of atomic environments.

---

# 19.112 Summary of Charge-Informed Learning

The central idea behind CHGNet can be summarized as

```text id="chg_charge9"
Atomic Structure

↓

Geometry

+

Learned Electronic Environment

↓

Unified Atomic Representation

↓

Energy

Forces

Stress

Magnetic Moments
```

By combining structural and electronic information within a single graph neural network, CHGNet achieves greater accuracy and transferability across a wide range of materials systems.

---

## Transition to the Next Section

Charge and magnetism are closely related in many materials, particularly transition-metal compounds where unpaired electrons determine magnetic behavior. In the next section, we will examine **magnetic moment prediction in CHGNet**, explaining how local magnetic moments are learned, why magnetism is crucial for many technologically important materials, and how CHGNet extends universal interatomic potentials beyond purely structural predictions.

# 19.113 Magnetic Moment Prediction

One of the most distinctive capabilities of CHGNet is its ability to predict **local magnetic moments**.

Earlier universal graph neural networks such as

* CGCNN,
* MEGNet,
* M3GNet,

focused primarily on predicting

* energies,
* forces,
* stress.

CHGNet extends these capabilities by learning information about the **magnetic state of every atom**.

This is particularly important because many technologically important materials are magnetic.

---

# 19.114 What Is Magnetism?

Magnetism originates from the motion of electrons.

More specifically,

it arises primarily from

* electron spin,
* electron orbital motion.

At the atomic scale,

each electron behaves like a tiny magnet.

```text id="chg_mag1"
Electron

↓

Spin

↓

Tiny Magnet
```

When many electrons interact,

their magnetic moments combine to produce the overall magnetic behavior of the material.

---

# 19.115 Electron Spin

Every electron possesses an intrinsic property called **spin**.

The two possible spin states are commonly represented as

```text id="chg_mag2"
↑

↓

```

Although these symbols resemble rotation,

electron spin is a purely quantum mechanical property.

It has no classical equivalent.

---

# 19.116 Magnetic Moments

The magnetic strength of an atom is described by its **magnetic moment**.

It is commonly denoted by

$$
\mu.
$$

The SI unit is

```text id="chg_mag3"
A·m²
```

In computational materials science,

magnetic moments are usually reported in

```text id="chg_mag4"
Bohr Magnetons (μB)
```

where

$$
1~\mu_B
$$

is the **Bohr magneton**.

---

# 19.117 Origin of Local Magnetic Moments

Not every atom possesses a magnetic moment.

Atoms with completely filled electron shells generally have

```text id="chg_mag5"
μ ≈ 0
```

Atoms containing unpaired electrons often exhibit significant magnetism.

Examples include

* Fe,
* Co,
* Ni,
* Mn,
* Cr.

These elements play central roles in

* permanent magnets,
* batteries,
* catalysts,
* spintronic devices.

---

# 19.118 Why Transition Metals Are Magnetic

Transition metals contain partially filled

$d$

orbitals.

For example,

iron has several unpaired

$d$

electrons.

Conceptually,

```text id="chg_mag6"
d Orbitals

↓

Unpaired Electrons

↓

Magnetic Moment
```

The number and arrangement of these unpaired electrons determine the magnetic properties of the material.

---

# 19.119 Local Versus Global Magnetism

A distinction must be made between

* local magnetic moments,
* overall magnetic ordering.

Each atom may possess its own magnetic moment,

while the crystal as a whole may exhibit different magnetic phases.

Examples include

* ferromagnetism,
* antiferromagnetism,
* ferrimagnetism,
* paramagnetism.

CHGNet predicts the **local magnetic moment** associated with each atom.

---

# 19.120 Ferromagnetic Ordering

In a ferromagnetic material,

neighboring atomic moments align in the same direction.

```text id="chg_mag7"
↑

↑

↑

↑

↑
```

The individual moments reinforce one another,

producing a strong net magnetization.

Examples include

* iron,
* cobalt,
* nickel.

---

# 19.121 Antiferromagnetic Ordering

In an antiferromagnetic material,

neighboring moments point in opposite directions.

```text id="chg_mag8"
↑

↓

↑

↓

↑
```

Although individual atoms possess magnetic moments,

their contributions largely cancel,

producing little or no net magnetization.

Many transition-metal oxides exhibit antiferromagnetic ordering.

---

# 19.122 Why Magnetism Matters in Materials Science

Magnetism influences many important material properties.

Examples include

* electronic conductivity,
* catalytic activity,
* battery performance,
* phase stability,
* mechanical behavior,
* superconductivity.

Ignoring magnetic effects can lead to substantial errors in predicted energies and structures.

---

# 19.123 Magnetic Moments in Density Functional Theory

Spin-polarized DFT calculations naturally predict local magnetic moments.

These DFT values serve as reference labels during CHGNet training.

Conceptually,

```text id="chg_mag9"
Crystal

↓

Spin-Polarized DFT

↓

Magnetic Moments

↓

CHGNet Training
```

Thus,

the neural network learns to reproduce DFT magnetic behavior.

---

# 19.124 Multi-Task Learning of Magnetism

During training,

magnetic moments are learned simultaneously with

* energies,
* forces,
* stress.

The workflow is

```text id="chg_mag10"
Graph Representation

↓

Energy

Forces

Stress

Magnetic Moments

↓

Combined Loss
```

Because these physical quantities are closely related,

learning them together improves overall model performance.

---

# 19.125 Predicting Atomic Magnetic Moments

After training,

CHGNet predicts a magnetic moment for every atom.

For example,

```text
Atom      Magnetic Moment

Fe1       4.12 μB

Fe2       3.95 μB

O1        0.03 μB

O2        0.01 μB
```

These values describe the local magnetic environment within the crystal.

---

# 19.126 Why Magnetic Moment Prediction Improves Energy Prediction

Electronic spin influences chemical bonding.

Changing the magnetic state of a material often changes

* bond lengths,
* bond strengths,
* total energy.

Therefore,

predicting magnetic moments helps the neural network construct a more accurate potential energy surface.

This improvement is particularly significant for

* iron oxides,
* manganese oxides,
* chromium compounds,
* cobalt oxides,
* nickel-based materials.

---

# 19.127 Magnetism During Structure Relaxation

As atoms move,

their local electronic environments also change.

Consequently,

their magnetic moments may evolve.

Conceptually,

```text id="chg_mag11"
Atomic Motion

↓

Bond Changes

↓

Electronic Changes

↓

Magnetic Moment Changes
```

CHGNet naturally captures this coupling because magnetism is learned alongside structural information.

---

# 19.128 Applications of Magnetic Moment Prediction

The ability to predict magnetic moments enables research in

* magnetic materials,
* permanent magnets,
* battery cathodes,
* transition-metal oxides,
* spintronics,
* magnetic semiconductors,
* catalytic materials.

These applications extend beyond traditional atomistic simulations that consider only energy and forces.

---

# 19.129 Advantages Over Earlier Universal Potentials

Compared with previous graph neural network potentials,

CHGNet provides several additional capabilities.

* Local magnetic moment prediction.
* Improved treatment of spin-polarized materials.
* Better transferability for transition-metal compounds.
* More accurate descriptions of chemically complex systems.
* Unified prediction of structural and magnetic properties.

These improvements make CHGNet particularly attractive for studying modern functional materials.

---

# 19.130 Relationship Between Charge and Magnetism

Charge redistribution and magnetism are closely connected.

Changing the oxidation state of a transition-metal atom often changes

* the number of unpaired electrons,
* the local magnetic moment,
* the bonding behavior.

Conceptually,

```text id="chg_mag12"
Charge Redistribution

↓

Electronic Configuration

↓

Magnetic Moment

↓

Material Properties
```

By learning both charge-informed representations and magnetic moments, CHGNet captures these coupled physical effects more effectively than earlier universal graph neural networks.

---

## Transition to the Next Section

We have now studied the fundamental physical quantities predicted by CHGNet—**energy, forces, stress, charge-informed representations, and magnetic moments**. The next section moves from theory to **practical atomistic simulations**, where we will examine how CHGNet performs **geometry optimization (structure relaxation)**, why relaxation is essential before calculating material properties, and how the model efficiently locates equilibrium crystal structures before molecular dynamics or DFT refinement.

# 19.131 Structure Relaxation Using CHGNet

One of the most important applications of CHGNet is **structure relaxation** (also called **geometry optimization**).

Almost every computational materials science workflow begins with a structure that is **not** at equilibrium.

Before calculating

* electronic properties,
* phonons,
* elastic constants,
* diffusion,
* molecular dynamics,

the structure must first be relaxed.

CHGNet performs this relaxation rapidly while maintaining an accuracy close to DFT for many materials.

---

# 19.132 What Is Structure Relaxation?

Structure relaxation is the process of moving atoms until the crystal reaches its lowest-energy configuration.

Initially,

the atoms may occupy unstable positions.

```text id="chg_relax1"
Initial Structure

↓

Large Forces

↓

Atoms Move

↓

Lower Energy

↓

Relaxed Structure
```

The final relaxed structure corresponds to a local minimum on the potential energy surface.

---

# 19.133 Why Relaxation Is Necessary

Experimental crystal structures are not always perfectly suitable for simulations.

Similarly,

structures generated by

* crystal prediction algorithms,
* generative AI,
* atom substitutions,
* defect creation,

often contain unrealistic atomic positions.

These structures exhibit

* large forces,
* high internal stress,
* unstable bond lengths.

Relaxation removes these artifacts before further calculations.

---

# 19.134 The Potential Energy Surface

Structure relaxation can be understood using the concept of the **potential energy surface (PES)**.

Imagine plotting the total energy as a function of atomic positions.

```text id="chg_relax2"
Energy

        ●

      /

    /

___/__________

Atomic Configuration
```

The valleys represent stable configurations,

while the hills correspond to unstable arrangements.

Relaxation moves the structure downhill until a minimum is reached.

---

# 19.135 Role of Forces in Relaxation

The direction of atomic motion is determined by the predicted forces.

Recall that

$$
\mathbf{F}
==========

*

\nabla E.
$$

Since the force points toward decreasing energy,

moving atoms along the force direction gradually reduces the total energy.

Thus,

force prediction is the driving mechanism behind structure relaxation.

---

# 19.136 Relaxation Workflow

The complete relaxation process consists of repeated energy and force evaluations.

```text id="chg_relax3"
Initial Structure

↓

CHGNet

↓

Energy

Forces

↓

Move Atoms

↓

New Structure

↓

Repeat
```

This iterative procedure continues until the forces become sufficiently small.

---

# 19.137 Relaxation Convergence

A relaxation calculation does not continue indefinitely.

Instead,

it terminates once predefined convergence criteria are satisfied.

Typical stopping conditions include

* maximum force below a specified threshold,
* negligible change in total energy,
* negligible atomic displacement,
* maximum number of optimization steps reached.

A commonly used force criterion is

```text id="chg_relax4"
|F|max < 0.01 eV/Å
```

although the exact value depends on the application.

---

# 19.138 Atomic Relaxation Versus Cell Relaxation

Two types of relaxation are commonly performed.

### Atomic Relaxation

Only atomic positions change.

The lattice vectors remain fixed.

```text id="chg_relax5"
Cell

Fixed

↓

Atoms Move
```

This approach is appropriate when the experimental lattice constants are already known.

---

### Cell Relaxation

Both

* atomic positions,
* lattice vectors,

are optimized simultaneously.

```text id="chg_relax6"
Atoms Move

+

Cell Changes

↓

Fully Relaxed Crystal
```

Cell relaxation requires both **force** and **stress** predictions.

---

# 19.139 Local and Global Energy Minima

An important concept in relaxation is the distinction between **local** and **global** minima.

```text id="chg_relax7"
Energy

      \

       \__

          \____

               \___

Configuration
```

A relaxation algorithm generally finds the **nearest local minimum**, not necessarily the absolute lowest-energy structure.

Therefore,

different starting configurations may converge to different relaxed structures.

---

# 19.140 Optimization Algorithms

The atomic positions are updated using optimization algorithms.

Common choices include

* Gradient Descent,
* Conjugate Gradient,
* BFGS,
* L-BFGS,
* FIRE (Fast Inertial Relaxation Engine).

These algorithms use the predicted forces to determine how the atoms should move at each step.

CHGNet supplies the energies and forces, while the optimizer decides the update direction and step size.

---

# 19.141 Why CHGNet Makes Relaxation Faster

Traditional DFT relaxation requires a full self-consistent electronic structure calculation at every optimization step.

```text id="chg_relax8"
Structure

↓

DFT

↓

Energy

↓

Forces

↓

Repeat
```

Each iteration may take minutes or even hours, depending on the system size.

In contrast,

CHGNet replaces the expensive DFT calculation with a neural network evaluation.

```text id="chg_relax9"
Structure

↓

CHGNet

↓

Energy

↓

Forces

↓

Repeat
```

This significantly reduces the computational time while maintaining good accuracy for many materials.

---

# 19.142 Relaxation of Defective Structures

Many research problems involve crystals containing

* vacancies,
* interstitial atoms,
* substitutional dopants,
* grain boundaries,
* surfaces.

These modifications disturb the equilibrium positions of nearby atoms.

Structure relaxation allows the lattice to respond naturally to these defects before further analysis.

---

# 19.143 Relaxation in Battery Materials

Battery materials often experience significant structural changes during charging and discharging.

For example,

lithium extraction causes

* lattice distortion,
* bond rearrangement,
* volume change.

CHGNet enables rapid relaxation of these evolving structures, allowing researchers to investigate different states of charge efficiently.

---

# 19.144 Relaxation Before Molecular Dynamics

Running molecular dynamics on an unrelaxed structure can lead to

* unrealistically large forces,
* unstable trajectories,
* numerical integration errors.

Therefore,

a typical workflow is

```text id="chg_relax10"
Initial Structure

↓

Relaxation

↓

Equilibrium Structure

↓

Molecular Dynamics
```

The relaxed structure provides a physically meaningful starting point for dynamic simulations.

---

# 19.145 Relaxation Before DFT Refinement

CHGNet is frequently used as a **pre-relaxation tool** before performing expensive DFT calculations.

The workflow is

```text id="chg_relax11"
Initial Structure

↓

CHGNet Relaxation

↓

Near-Equilibrium Structure

↓

DFT Relaxation

↓

Final Structure
```

Because the CHGNet-relaxed structure is already close to equilibrium,

the subsequent DFT calculation often requires far fewer optimization steps, reducing the overall computational cost.

---

# 19.146 Advantages of CHGNet Relaxation

Using CHGNet for geometry optimization offers several practical benefits.

* Much faster than full DFT relaxation.
* Near-DFT accuracy for many crystalline materials.
* Supports both atomic and cell relaxation.
* Efficient handling of large supercells.
* Suitable for high-throughput structure optimization.
* Ideal as a pre-relaxation step before DFT.

These advantages make CHGNet a valuable tool in modern computational materials science workflows.

---

# 19.147 Relaxation Workflow Summary

The complete relaxation procedure can be summarized as

```text id="chg_relax12"
Initial Crystal

↓

Graph Construction

↓

CHGNet

↓

Energy + Forces + Stress

↓

Geometry Optimizer

↓

Updated Structure

↓

Convergence?

↓

No ───► Repeat

↓

Yes

↓

Relaxed Crystal
```

This iterative process transforms an arbitrary initial structure into a physically stable configuration suitable for further simulations and property calculations.

---

## Transition to the Next Section

Once a crystal has been relaxed, we often want to study **how it evolves with time** rather than just its equilibrium structure. The next section introduces **molecular dynamics (MD) using CHGNet**, where predicted energies and forces are used to simulate atomic motion, diffusion, thermal vibrations, phase transitions, and other finite-temperature phenomena over thousands or even millions of time steps.

# 19.148 Molecular Dynamics Using CHGNet

While structure relaxation determines the **equilibrium configuration** of a material,

it provides only a static picture.

Real materials are never perfectly stationary.

At finite temperature,

atoms continuously

* vibrate,
* diffuse,
* collide,
* rearrange.

To study these dynamic processes, we use **Molecular Dynamics (MD)**.

CHGNet enables molecular dynamics simulations with near-DFT accuracy while being several orders of magnitude faster than first-principles molecular dynamics.

---

# 19.149 What Is Molecular Dynamics?

Molecular Dynamics is a computational technique that predicts how atoms move over time.

Instead of finding only the lowest-energy structure,

MD simulates the continuous motion of atoms according to Newton's laws of motion.

The workflow is

```text id="chg_md1"
Initial Structure

↓

Energy & Forces

↓

Newton's Laws

↓

New Positions

↓

Repeat
```

The result is a trajectory describing the positions and velocities of all atoms as functions of time.

---

# 19.150 Why Molecular Dynamics Is Important

Many important material properties depend on atomic motion rather than static structures.

Examples include

* ionic diffusion,
* thermal expansion,
* phase transitions,
* melting,
* defect migration,
* phonon vibrations,
* thermal conductivity.

These phenomena cannot be studied using geometry optimization alone.

Instead,

they require simulations that evolve with time.

---

# 19.151 Newton's Second Law

The motion of every atom follows Newton's Second Law.

genui{"physics_motion_forces_learning_block":{"type_id":"NEWTON_SECOND_LAW"}}

Mathematically,

$$
\mathbf{F}
==========

m\mathbf{a},
$$

where

* $\mathbf{F}$ is the force,
* $m$ is the atomic mass,
* $\mathbf{a}$ is the acceleration.

Since CHGNet predicts the force,

the acceleration can be calculated directly.

---

# 19.152 Time Integration

Knowing the acceleration at one instant is not sufficient.

The equations of motion must be integrated over many small time steps.

For every step,

the simulation performs

```text id="chg_md2"
Current Positions

↓

CHGNet

↓

Forces

↓

Acceleration

↓

Velocity

↓

New Positions
```

This cycle is repeated thousands or millions of times.

---

# 19.153 Time Step

The simulation advances in very small increments of time.

A typical time step is

```text id="chg_md3"
1 femtosecond (1 fs)

=

10⁻¹⁵ seconds
```

Although extremely small,

millions of such steps can simulate events occurring over nanoseconds.

---

# 19.154 Molecular Dynamics Workflow

A complete CHGNet molecular dynamics simulation proceeds as follows.

```text id="chg_md4"
Relaxed Structure

↓

Assign Initial Velocities

↓

CHGNet

↓

Energy

Forces

Stress

↓

Integrate Motion

↓

Next Configuration

↓

Repeat
```

The resulting sequence of atomic configurations forms the molecular dynamics trajectory.

---

# 19.155 Initial Velocities

Atoms cannot begin moving without initial velocities.

These velocities are usually assigned according to the

**Maxwell–Boltzmann distribution** corresponding to the desired temperature.

For example,

```text id="chg_md5"
Temperature

↓

300 K

↓

Random Velocities
```

Different temperatures produce different velocity distributions.

---

# 19.156 Temperature in Molecular Dynamics

Temperature is directly related to the average kinetic energy of the atoms.

The kinetic energy of one atom is

genui{"physics_energy_fluids_machines_materials_learning_block":{"type_id":"KINETIC_ENERGY"}}

$$
KE
==

\frac{1}{2}mv^2.
$$

As temperature increases,

atomic velocities increase,

leading to more vigorous atomic motion.

---

# 19.157 Energy Conservation

In an ideal isolated system,

the total energy remains constant.

The total energy is

$$
E_{\text{total}}
================

PE
+
KE,
$$

where

* $PE$ is the potential energy,
* $KE$ is the kinetic energy.

CHGNet predicts the potential energy,

while the kinetic energy is computed from the atomic velocities.

---

# 19.158 Simulation Ensembles

Different molecular dynamics simulations maintain different physical quantities.

The most common ensembles are

| Ensemble | Constant Quantities                    |
| -------- | -------------------------------------- |
| NVE      | Number of atoms, Volume, Energy        |
| NVT      | Number of atoms, Volume, Temperature   |
| NPT      | Number of atoms, Pressure, Temperature |

Because CHGNet predicts

* energy,
* forces,
* stress,

it supports all of these commonly used ensembles.

---

# 19.159 Diffusion Simulations

One of the most important applications of CHGNet molecular dynamics is diffusion.

Consider lithium ions inside a battery cathode.

Initially,

```text id="chg_md6"
Li

↓

Site A
```

After sufficient thermal motion,

```text id="chg_md7"
Li

↓

Site B
```

Repeating this process produces ionic diffusion throughout the crystal.

The diffusion coefficient can then be calculated from the molecular dynamics trajectory.

---

# 19.160 Thermal Vibrations

Even at room temperature,

atoms vibrate around their equilibrium positions.

Conceptually,

```text id="chg_md8"
Equilibrium Position

↓

Small Oscillations

↓

Continuous Motion
```

These vibrations influence

* heat capacity,
* thermal conductivity,
* phonon spectra,
* lattice expansion.

Molecular dynamics naturally captures these effects.

---

# 19.161 Phase Transitions

Heating a material may produce structural transformations.

Examples include

* solid → liquid,
* crystalline → amorphous,
* one crystal phase → another.

A molecular dynamics simulation reveals how these transformations occur over time.

---

# 19.162 Defect Migration

Many functional materials contain

* vacancies,
* interstitial atoms,
* substitutional impurities.

These defects migrate through the crystal by thermal activation.

CHGNet molecular dynamics enables researchers to observe

* diffusion pathways,
* migration mechanisms,
* defect interactions,

without performing expensive DFT calculations at every time step.

---

# 19.163 Advantages Over Ab Initio Molecular Dynamics

Traditional **ab initio molecular dynamics (AIMD)** performs a DFT calculation at every time step.

The workflow is

```text id="chg_md9"
Structure

↓

DFT

↓

Forces

↓

Next Step
```

This approach is highly accurate but computationally demanding.

CHGNet replaces the DFT calculation with a neural network.

```text id="chg_md10"
Structure

↓

CHGNet

↓

Forces

↓

Next Step
```

As a result,

simulations become dramatically faster while retaining good accuracy for many materials.

---

# 19.164 Typical Simulation Lengths

Because CHGNet is computationally efficient,

researchers can simulate

* larger systems,
* longer time scales,
* more complex materials.

Typical simulations include

* thousands to tens of thousands of atoms,
* nanoseconds of simulation time,
* millions of integration steps,

which are often impractical using DFT alone.

---

# 19.165 Applications of CHGNet Molecular Dynamics

CHGNet molecular dynamics is widely used for studying

* lithium-ion diffusion,
* sodium-ion diffusion,
* thermal stability,
* defect migration,
* melting,
* solid-state phase transitions,
* ionic conductivity,
* thermal expansion,
* amorphization,
* high-temperature behavior.

These simulations provide insight into processes that are difficult or impossible to observe experimentally.

---

# 19.166 Complete Molecular Dynamics Workflow

The overall workflow can be summarized as

```text id="chg_md11"
Relaxed Crystal

↓

Assign Temperature

↓

Initialize Velocities

↓

CHGNet

↓

Energy

Forces

Stress

↓

Newtonian Integration

↓

Updated Structure

↓

Trajectory

↓

Property Analysis
```

The generated trajectory can then be analyzed to calculate physical properties such as diffusion coefficients, radial distribution functions, mean squared displacement, and temperature-dependent structural evolution.

---

## Transition to the Next Section

We have now explored the major simulation capabilities of CHGNet, including **energy prediction, force prediction, stress prediction, structure relaxation, and molecular dynamics**. In the next section, we will compare **CHGNet and M3GNet** in detail, examining their architectures, training datasets, prediction targets, computational performance, strengths, limitations, and the situations in which one model may be preferred over the other.

# 19.167 CHGNet Versus M3GNet

Both **M3GNet** and **CHGNet** represent major milestones in the development of universal graph neural networks for materials science.

They share many similarities.

Both models

* are graph neural networks,
* use message passing,
* predict energies and forces,
* support molecular dynamics,
* support structure relaxation,
* are trained using large DFT datasets.

However,

their design philosophies are different.

M3GNet focuses on learning a highly accurate universal interatomic potential from geometric information,

whereas CHGNet extends this idea by incorporating **charge-informed representations** and **magnetic information**.

---

# 19.168 Historical Development

The development of these models followed a natural progression.

```text id="chg_cmp1"
CGCNN

↓

MEGNet

↓

M3GNet

↓

CHGNet
```

Each generation introduced new physical information and improved predictive capability.

* **CGCNN** demonstrated that graph neural networks could learn crystal properties.
* **MEGNet** incorporated graph networks with global state information.
* **M3GNet** introduced three-body interactions and a universal interatomic potential.
* **CHGNet** added charge-informed learning and magnetic moment prediction.

---

# 19.169 Comparison of Model Inputs

Both models require only a crystal structure as input.

The user provides

* lattice vectors,
* atomic coordinates,
* atomic species.

Neither model requires manually specifying

* bond types,
* oxidation states,
* magnetic moments.

However,

their internal representations differ.

| Input                          | M3GNet | CHGNet |
| ------------------------------ | ------ | ------ |
| Atomic positions               | ✓      | ✓      |
| Atomic species                 | ✓      | ✓      |
| Bond distances                 | ✓      | ✓      |
| Bond angles                    | ✓      | ✓      |
| Charge-informed representation | ✗      | ✓      |
| Magnetic information           | ✗      | ✓      |

Thus,

CHGNet enriches the learned representation without requiring additional user input.

---

# 19.170 Three-Body Interactions

Both models explicitly incorporate three-body interactions.

Instead of considering only pairwise distances,

they also include bond angles.

```text id="chg_cmp2"
A

 \

  \

   B

  /

 /

C
```

This allows both models to describe

* directional covalent bonding,
* crystal geometry,
* angular interactions

much more accurately than pairwise potentials.

---

# 19.171 Treatment of Electronic Information

This is the most significant conceptual difference.

### M3GNet

Primarily learns from

* geometry,
* distances,
* angles.

### CHGNet

Learns from

* geometry,
* distances,
* angles,
* latent electronic representations,
* charge-related information,
* magnetic moments.

Consequently,

CHGNet provides a richer description of chemically complex materials.

---

# 19.172 Prediction Targets

Both models predict several physical quantities.

| Quantity         | M3GNet | CHGNet |
| ---------------- | ------ | ------ |
| Energy           | ✓      | ✓      |
| Forces           | ✓      | ✓      |
| Stress           | ✓      | ✓      |
| Magnetic moments | ✗      | ✓      |

Magnetic moment prediction is a defining capability of CHGNet.

---

# 19.173 Training Data

The training datasets also differ.

### M3GNet

Primarily trained using data derived from the Materials Project.

### CHGNet

Uses the **Materials Project Trajectory (MPtrj)** dataset,

which contains complete DFT relaxation trajectories.

Instead of learning only equilibrium structures,

CHGNet learns from

* equilibrium configurations,
* distorted structures,
* intermediate relaxation steps,
* nonequilibrium atomic environments.

This richer training data improves transferability.

---

# 19.174 Learning Strategy

Both models use supervised learning,

but CHGNet adopts a more comprehensive multi-task approach.

### M3GNet learns

* energy,
* forces,
* stress.

### CHGNet learns

* energy,
* forces,
* stress,
* magnetic moments.

Because these quantities are optimized simultaneously,

the learned representations become more physically meaningful.

---

# 19.175 Performance on Magnetic Materials

Magnetic materials remain challenging for machine learning potentials.

Examples include

* Fe₂O₃,
* MnO,
* NiO,
* CoO,
* Cr₂O₃.

For such systems,

local electronic structure strongly influences

* bonding,
* stability,
* lattice constants,
* phase behavior.

Since CHGNet explicitly learns magnetic information,

it generally provides a more realistic description of these materials.

---

# 19.176 Battery Materials

Battery materials frequently undergo

* oxidation,
* reduction,
* charge redistribution,
* magnetic changes.

Examples include

* LiFePO₄,
* LiMn₂O₄,
* LiCoO₂,
* NMC cathodes.

The charge-informed representations learned by CHGNet make it particularly well suited for these chemically evolving systems.

---

# 19.177 Computational Cost

Both models are highly efficient compared with DFT.

Their computational costs are generally similar because both rely on graph neural network message passing.

The additional charge-informed representations in CHGNet introduce only a modest increase in computational complexity while providing richer physical information.

Thus,

both remain practical for large-scale atomistic simulations.

---

# 19.178 Advantages of M3GNet

M3GNet offers several important strengths.

* Excellent universal interatomic potential.
* Strong geometric modeling.
* Efficient molecular dynamics.
* Accurate structure relaxation.
* Broad applicability across many crystalline materials.

It remains an outstanding general-purpose machine-learned potential.

---

# 19.179 Advantages of CHGNet

CHGNet builds upon these strengths while adding several new capabilities.

* Charge-informed representations.
* Local magnetic moment prediction.
* Improved treatment of transition-metal compounds.
* Better performance for many magnetic materials.
* Richer electronic representations.
* Multi-task learning including magnetic information.

These improvements are particularly valuable for functional materials containing strongly correlated transition metals.

---

# 19.180 Limitations of Both Models

Despite their impressive capabilities,

both models share several limitations.

Neither model

* replaces DFT for high-precision electronic structure calculations,
* predicts electronic band structures directly,
* computes electron densities,
* solves the Schrödinger equation,
* replaces many-body quantum chemistry methods.

Instead,

they provide efficient approximations to the DFT potential energy surface.

---

# 19.181 When Should You Use M3GNet?

M3GNet is an excellent choice when the primary goal is

* rapid structure relaxation,
* molecular dynamics,
* geometry optimization,
* high-throughput screening,
* general-purpose atomistic simulations,

especially when explicit magnetic information is not required.

---

# 19.182 When Should You Use CHGNet?

CHGNet is often preferred when studying

* transition-metal oxides,
* magnetic materials,
* battery electrodes,
* spin-polarized systems,
* chemically complex compounds,
* systems where local electronic environments play an important role.

Its charge-informed learning and magnetic moment prediction provide additional physical insight beyond purely geometric models.

---

# 19.183 Side-by-Side Summary

| Feature                                | M3GNet  | CHGNet   |
| -------------------------------------- | ------- | -------- |
| Graph neural network                   | ✓       | ✓        |
| Three-body interactions                | ✓       | ✓        |
| Universal interatomic potential        | ✓       | ✓        |
| Energy prediction                      | ✓       | ✓        |
| Force prediction                       | ✓       | ✓        |
| Stress prediction                      | ✓       | ✓        |
| Molecular dynamics                     | ✓       | ✓        |
| Structure relaxation                   | ✓       | ✓        |
| Charge-informed representation         | ✗       | ✓        |
| Magnetic moment prediction             | ✗       | ✓        |
| Better treatment of magnetic materials | Limited | Stronger |

---

# 19.184 Final Perspective

M3GNet and CHGNet should not be viewed as competing models but rather as successive stages in the evolution of machine-learned interatomic potentials.

M3GNet demonstrated that a universal graph neural network could accurately reproduce DFT-level energies, forces, and stresses across a broad range of materials.

CHGNet extended this foundation by incorporating physically meaningful electronic representations and magnetic information, improving its ability to model chemically and magnetically complex systems.

Together, these models have transformed atomistic simulations by enabling near-DFT accuracy at a fraction of the computational cost, making high-throughput materials discovery and large-scale molecular dynamics accessible to a much broader range of research problems.

---

## Transition to the Practical Implementation

With the theoretical foundations now complete, the remainder of this chapter will shift from **understanding CHGNet** to **using CHGNet in real research**.

The practical sections will include:

* Installation and environment setup
* Loading pretrained CHGNet models
* Reading crystal structures with `pymatgen`
* Predicting energies, forces, stresses, and magnetic moments
* Geometry optimization
* Molecular dynamics simulations
* High-throughput screening workflows
* Integration with ASE and Materials Project datasets
* Complete research examples suitable for publication-level materials informatics studies

These sections will provide fully executable Python code with detailed explanations, enabling readers to apply CHGNet directly to their own materials science research.

# 19.185 Installing CHGNet

In the previous sections, we studied the theoretical foundations of CHGNet.

Now we transition from theory to practice.

This section explains how to install CHGNet and prepare a Python environment suitable for

* energy prediction,
* force prediction,
* structure relaxation,
* molecular dynamics,
* materials research.

The goal is to build a stable scientific computing environment that can later be extended with

* PyTorch,
* ASE,
* pymatgen,
* matminer,
* Materials Project tools.

---

# 19.186 System Requirements

CHGNet can run on both

* CPUs,
* NVIDIA GPUs.

Although CPU execution is sufficient for learning and small systems,

GPU acceleration is strongly recommended for

* molecular dynamics,
* large supercells,
* high-throughput calculations,
* machine learning research.

Typical requirements are

| Component        | Recommended                                               |
| ---------------- | --------------------------------------------------------- |
| Python           | 3.10 or newer                                             |
| RAM              | ≥16 GB                                                    |
| Storage          | ≥10 GB free                                               |
| GPU              | NVIDIA CUDA-supported GPU (optional but recommended)      |
| Operating System | Linux, macOS, Windows (WSL recommended for Windows users) |

---

# 19.187 Creating a Virtual Environment

Installing scientific software directly into the system Python installation is not recommended.

Instead,

create an isolated virtual environment.

Using **conda**,

```python id="chg_install1"
conda create -n chgnet python=3.10
```

Activate the environment.

```python id="chg_install2"
conda activate chgnet
```

Now every package installed will remain isolated from other Python projects.

---

# 19.188 Installing PyTorch

CHGNet is built on top of **PyTorch**.

Install PyTorch before installing CHGNet.

For CPU-only systems,

```python id="chg_install3"
pip install torch torchvision torchaudio
```

For NVIDIA GPUs,

visit the official PyTorch installation page and choose the command matching your

* CUDA version,
* operating system,
* package manager.

Installing the correct CUDA-enabled version is important for maximizing performance.

---

# 19.189 Installing CHGNet

Once PyTorch has been installed,

install CHGNet directly from PyPI.

```python id="chg_install4"
pip install chgnet
```

This command automatically installs the core CHGNet package and its required dependencies.

---

# 19.190 Installing Pymatgen

CHGNet uses **pymatgen** to represent crystal structures.

Install it using

```python id="chg_install5"
pip install pymatgen
```

Pymatgen provides tools for

* reading CIF files,
* manipulating crystal structures,
* generating supercells,
* analyzing symmetry,
* preparing structures for machine learning.

---

# 19.191 Installing ASE

Many CHGNet workflows rely on the **Atomic Simulation Environment (ASE)**.

Install ASE using

```python id="chg_install6"
pip install ase
```

ASE provides

* geometry optimization,
* molecular dynamics,
* visualization,
* interfaces to numerous atomistic simulation packages.

CHGNet integrates naturally with ASE.

---

# 19.192 Optional Scientific Packages

Several additional packages are useful for research.

Install them simultaneously.

```python id="chg_install7"
pip install matplotlib pandas numpy scipy
```

These packages support

* plotting,
* numerical analysis,
* data manipulation,
* scientific computation.

---

# 19.193 Installing Matminer

Feature engineering and materials data analysis often require **matminer**.

Install it using

```python id="chg_install8"
pip install matminer
```

Although CHGNet itself does not require matminer,

it is valuable for many materials informatics workflows.

---

# 19.194 Checking the Installation

After installation,

verify that CHGNet can be imported successfully.

```python id="chg_install9"
import chgnet

print("CHGNet installed successfully!")
```

If no errors appear,

the installation is functioning correctly.

---

# 19.195 Checking the Installed Version

It is good practice to record the software version used in a research project.

```python id="chg_install10"
import chgnet

print(chgnet.__version__)
```

Documenting package versions improves

* reproducibility,
* collaboration,
* publication quality.

---

# 19.196 Testing PyTorch

Verify that PyTorch is working correctly.

```python id="chg_install11"
import torch

print(torch.__version__)
```

A successful output confirms that PyTorch has been installed properly.

---

# 19.197 Checking GPU Availability

If an NVIDIA GPU is available,

PyTorch should detect it automatically.

```python id="chg_install12"
import torch

print(torch.cuda.is_available())
```

Possible output

```text id="chg_install13"
True
```

indicates that GPU acceleration is available.

If the output is

```text id="chg_install14"
False
```

the calculations will run on the CPU.

---

# 19.198 Displaying GPU Information

To determine which GPU PyTorch is using,

execute

```python id="chg_install15"
import torch

print(torch.cuda.get_device_name(0))
```

Example output

```text id="chg_install16"
NVIDIA RTX 4090
```

or

```text id="chg_install17"
NVIDIA A100
```

depending on the available hardware.

---

# 19.199 Verifying Pymatgen

Ensure that pymatgen imports correctly.

```python id="chg_install18"
from pymatgen.core import Structure

print("Pymatgen works!")
```

Since nearly every CHGNet workflow begins with a crystal structure,

this verification is important.

---

# 19.200 Verifying ASE

Similarly,

test ASE.

```python id="chg_install19"
from ase import Atoms

print("ASE works!")
```

If this command executes successfully,

ASE is correctly installed.

---

# 19.201 Common Installation Problems

Several issues occur frequently during installation.

### Problem 1

```text id="chg_install20"
ModuleNotFoundError
```

Cause

The required package has not been installed.

Solution

```python id="chg_install21"
pip install package_name
```

---

### Problem 2

```text id="chg_install22"
CUDA not available
```

Cause

Incorrect PyTorch installation.

Solution

Install the CUDA-enabled version of PyTorch matching your GPU and CUDA toolkit.

---

### Problem 3

```text id="chg_install23"
Version conflicts
```

Cause

Mixing incompatible package versions.

Solution

Create a fresh virtual environment before installation.

---

# 19.202 Recommended Project Structure

A clean directory structure simplifies research.

```text id="chg_install24"
CHGNet_Project/

│

├── data/

├── cif/

├── scripts/

├── models/

├── results/

├── figures/

└── notebooks/
```

This organization keeps

* input structures,
* analysis scripts,
* simulation outputs,
* figures,

well organized.

---

# 19.203 Installation Checklist

Before proceeding,

confirm the following.

✓ Python installed

✓ Virtual environment created

✓ PyTorch installed

✓ CHGNet installed

✓ pymatgen installed

✓ ASE installed

✓ GPU detected (optional)

✓ Test imports successful

Once all items are complete,

the computational environment is ready for practical CHGNet simulations.

---

## Transition to the Next Section

The software environment is now fully prepared. In the next section, we will load a **pretrained CHGNet model**, examine its architecture programmatically, and perform our **first energy prediction** on a real crystal structure using only a few lines of Python. This marks the beginning of hands-on CHGNet applications for materials informatics research.

# 19.204 Loading a Pretrained CHGNet Model

One of the major advantages of CHGNet is that users **do not need to train the model from scratch**.

Training a universal interatomic potential requires

* millions of DFT calculations,
* thousands of GPU hours,
* large computational resources.

Instead,

the CHGNet developers provide **pretrained models** that can be downloaded and used immediately.

With only a few lines of Python, we can load a model capable of predicting

* total energy,
* atomic forces,
* stress tensor,
* magnetic moments

for a wide range of crystalline materials.

---

# 19.205 Why Use a Pretrained Model?

Training a deep graph neural network is an expensive process.

A simplified workflow is

```text id="chg_model1"
Millions of DFT Structures

↓

Graph Construction

↓

Neural Network Training

↓

Weeks of GPU Computation

↓

Pretrained Model
```

Once the model has been trained,

all users can simply download the learned parameters instead of repeating the entire training process.

This saves enormous computational effort.

---

# 19.206 Importing CHGNet

The first step is to import the CHGNet class.

```python id="chg_model2"
from chgnet.model import CHGNet
```

This statement imports the main neural network model used throughout the remainder of the chapter.

---

# 19.207 Loading the Default Model

The simplest way to load the pretrained model is

```python id="chg_model3"
from chgnet.model import CHGNet

model = CHGNet.load()
```

The `load()` function automatically

* loads the pretrained weights,
* initializes the neural network,
* prepares the model for inference.

No manual downloading is required.

---

# 19.208 Understanding What Happens Internally

Although the code appears simple,

many operations occur internally.

```text id="chg_model4"
Load Model

↓

Read Pretrained Weights

↓

Build Neural Network

↓

Load Parameters

↓

Ready for Prediction
```

These pretrained weights encode the knowledge learned from millions of DFT calculations.

---

# 19.209 The Model Object

The variable

```python id="chg_model5"
model
```

contains the complete neural network.

It stores

* network architecture,
* learned weights,
* prediction functions,
* graph construction methods.

All future predictions will use this object.

---

# 19.210 Inspecting the Model

We can display basic information about the loaded model.

```python id="chg_model6"
print(model)
```

The output summarizes the neural network architecture,

including

* embedding layers,
* message-passing blocks,
* prediction heads.

Although the exact output depends on the installed version of CHGNet, it typically lists the major components of the model.

---

# 19.211 Counting Trainable Parameters

Deep neural networks often contain millions of trainable parameters.

The following code calculates the total number.

```python id="chg_model7"
total_params = sum(
    p.numel() for p in model.parameters()
)

print(total_params)
```

Here,

* `parameters()` returns every learnable weight in the network.
* `numel()` returns the number of elements in each tensor.
* `sum()` adds them together.

The result indicates the overall complexity of the model.

---

# 19.212 Checking the Device

PyTorch models can run on either

* CPU,
* GPU.

Move the model to the desired device.

```python id="chg_model8"
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
```

If a CUDA-compatible GPU is available,

the model will execute much faster.

---

# 19.213 Evaluation Mode

Before making predictions,

the model should be switched to evaluation mode.

```python id="chg_model9"
model.eval()
```

Evaluation mode

* disables dropout,
* freezes batch normalization statistics,
* ensures deterministic inference.

This is standard practice for all pretrained PyTorch models.

---

# 19.214 Why Evaluation Mode Matters

Neural networks behave differently during

* training,
* inference.

During training,

certain layers intentionally introduce randomness to improve generalization.

During prediction,

this randomness should be disabled.

The workflow is

```text id="chg_model10"
Training

↓

Random Regularization

↓

Evaluation

↓

Deterministic Predictions
```

Using `model.eval()` ensures consistent results every time the model is used.

---

# 19.215 Verifying Successful Loading

A simple confirmation message is often useful.

```python id="chg_model11"
print("CHGNet model loaded successfully.")
```

If no error appears,

the model is ready for prediction.

---

# 19.216 What the Model Can Predict

Once loaded,

the pretrained model can estimate

* total energy,
* atomic forces,
* stress tensor,
* magnetic moments.

Conceptually,

```text id="chg_model12"
Crystal Structure

↓

CHGNet

↓

Energy

Forces

Stress

Magnetic Moments
```

These predictions require only the crystal structure as input.

---

# 19.217 No Retraining Required

Many beginners mistakenly believe that they must retrain the model before using it.

In most research workflows,

this is unnecessary.

The pretrained CHGNet model already captures a vast amount of information learned from DFT data.

Researchers typically

1. load the pretrained model,
2. perform predictions,
3. analyze the results.

Only specialized applications require additional fine-tuning.

---

# 19.218 When Fine-Tuning Is Needed

Although the pretrained model is highly transferable,

fine-tuning may improve performance for

* highly specialized materials,
* new chemical systems,
* custom DFT datasets,
* domain-specific research.

The workflow becomes

```text id="chg_model13"
Pretrained CHGNet

↓

Additional Training Data

↓

Fine-Tuning

↓

Specialized Model
```

This process is much faster than training an entirely new network from scratch.

---

# 19.219 Memory Usage

Loading the pretrained model requires memory to store

* network architecture,
* trainable weights,
* intermediate tensors.

Modern desktop computers with

```text id="chg_model14"
16–32 GB RAM
```

can typically run CHGNet comfortably for many practical materials science problems.

For very large supercells,

GPU memory becomes the primary limiting factor.

---

# 19.220 Complete Loading Script

The following script combines the essential steps into a single workflow.

```python id="chg_model15"
import torch
from chgnet.model import CHGNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CHGNet.load()
model = model.to(device)
model.eval()

print("Device:", device)
print("CHGNet is ready.")
```

This script serves as a standard starting point for most CHGNet projects.

---

# 19.221 Workflow Summary

The complete model-loading process can be summarized as

```text id="chg_model16"
Import CHGNet

↓

Load Pretrained Model

↓

Move to CPU/GPU

↓

Evaluation Mode

↓

Ready for Prediction
```

At this stage,

the neural network is fully initialized and ready to analyze real crystal structures.

---

## Transition to the Next Section

With the pretrained model successfully loaded, the next step is to provide it with a **real crystal structure**. In the following section, we will learn how to read **CIF files** using `pymatgen`, construct `Structure` objects, inspect lattice vectors and atomic coordinates, and prepare materials for CHGNet predictions. This is the first step toward performing energy, force, stress, and magnetic moment calculations on real crystalline materials.

# 19.222 Reading Crystal Structures with Pymatgen

A machine learning model cannot make predictions without input data.

For CHGNet, the required input is a **crystal structure**.

Unlike conventional machine learning models that accept vectors or tables,

CHGNet accepts a **pymatgen `Structure` object**, which contains complete crystallographic information.

In this section, we will learn how to

* read crystal structures,
* inspect lattice information,
* access atomic positions,
* prepare structures for CHGNet predictions.

---

# 19.223 Why Pymatgen?

**Pymatgen (Python Materials Genomics)** is one of the most widely used Python libraries in computational materials science.

It provides tools for

* reading crystal structures,
* writing structure files,
* symmetry analysis,
* crystal manipulation,
* defect generation,
* supercell construction,
* Materials Project integration.

Because of its robustness and widespread adoption, CHGNet uses pymatgen's `Structure` class as its primary input format.

---

# 19.224 Common Crystal Structure File Formats

Crystal structures are stored in several standard formats.

| Format  | Extension | Common Use                      |
| ------- | --------- | ------------------------------- |
| CIF     | `.cif`    | Experimental crystal structures |
| POSCAR  | `POSCAR`  | VASP calculations               |
| CONTCAR | `CONTCAR` | Relaxed VASP structures         |
| XYZ     | `.xyz`    | Molecules and small systems     |
| CSSR    | `.cssr`   | Crystallographic data           |
| JSON    | `.json`   | Serialized structures           |

Among these,

the **CIF (Crystallographic Information File)** is the most common starting point.

---

# 19.225 Importing the Structure Class

Begin by importing the `Structure` class.

```python id="chg_structure1"
from pymatgen.core import Structure
```

This class represents a periodic crystal, including

* lattice vectors,
* atomic species,
* fractional coordinates,
* Cartesian coordinates,
* symmetry information.

---

# 19.226 Loading a CIF File

Suppose we have a crystal structure named

```text id="chg_structure2"
LiFePO4.cif
```

Load it using

```python id="chg_structure3"
from pymatgen.core import Structure

structure = Structure.from_file("LiFePO4.cif")
```

The variable `structure` now contains the complete crystal.

---

# 19.227 Understanding the Structure Object

The `Structure` object stores all crystallographic information.

Conceptually,

```text id="chg_structure4"
Structure

├── Lattice

├── Atomic Species

├── Atomic Coordinates

├── Periodic Boundary Conditions

└── Crystal Symmetry
```

This object is passed directly to CHGNet.

---

# 19.228 Printing the Structure

To inspect the loaded crystal,

simply print it.

```python id="chg_structure5"
print(structure)
```

A typical output includes

* lattice parameters,
* lattice angles,
* number of atoms,
* atomic positions.

This provides a quick overview of the crystal.

---

# 19.229 Number of Atoms

The number of atoms in the unit cell is

```python id="chg_structure6"
print(len(structure))
```

Example output

```text id="chg_structure7"
28
```

The exact value depends on the crystal being loaded.

---

# 19.230 Accessing the Lattice

Every crystal possesses three lattice vectors.

Retrieve them using

```python id="chg_structure8"
print(structure.lattice)
```

This displays

* lattice lengths,
* lattice angles,
* lattice matrix.

The lattice defines the periodic simulation cell.

---

# 19.231 Lattice Parameters

The lattice constants are accessed individually.

```python id="chg_structure9"
print("a =", structure.lattice.a)
print("b =", structure.lattice.b)
print("c =", structure.lattice.c)
```

Similarly,

the lattice angles are

```python id="chg_structure10"
print("α =", structure.lattice.alpha)
print("β =", structure.lattice.beta)
print("γ =", structure.lattice.gamma)
```

These values completely define the unit cell geometry.

---

# 19.232 Atomic Species

The chemical elements present in the crystal are obtained using

```python id="chg_structure11"
print(structure.species)
```

Example output

```text id="chg_structure12"
Li
Fe
P
O
...
```

Each entry corresponds to one atomic site in the crystal.

---

# 19.233 Atomic Coordinates

The fractional coordinates are stored in

```python id="chg_structure13"
print(structure.frac_coords)
```

Example output

```text id="chg_structure14"
[[0.00 0.50 0.25]

 [0.50 0.00 0.75]

 ...
]
```

Fractional coordinates range from

```text id="chg_structure15"
0

to

1
```

along each lattice direction.

---

# 19.234 Cartesian Coordinates

Cartesian coordinates are obtained using

```python id="chg_structure16"
print(structure.cart_coords)
```

Unlike fractional coordinates,

Cartesian coordinates are expressed in

```text id="chg_structure17"
Å (Angstrom)
```

These coordinates represent actual distances in three-dimensional space.

---

# 19.235 Accessing Individual Atoms

Individual atomic sites can be accessed by index.

```python id="chg_structure18"
site = structure[0]

print(site)
```

The returned object contains

* element,
* coordinates,
* occupancy,
* additional crystallographic information.

---

# 19.236 Iterating Through Atoms

To examine every atom,

use a loop.

```python id="chg_structure19"
for site in structure:
    print(site.species_string, site.frac_coords)
```

This prints

* atomic element,
* fractional coordinates

for every atom in the crystal.

---

# 19.237 Chemical Formula

The chemical formula is available directly.

```python id="chg_structure20"
print(structure.formula)
```

Example output

```text id="chg_structure21"
Li4 Fe4 P4 O16
```

This confirms that the correct structure has been loaded.

---

# 19.238 Reduced Formula

The reduced chemical formula is

```python id="chg_structure22"
print(structure.composition.reduced_formula)
```

Example output

```text id="chg_structure23"
LiFePO4
```

This is often the preferred representation in publications.

---

# 19.239 Crystal Density

Pymatgen can also calculate the density.

```python id="chg_structure24"
print(structure.density)
```

Example output

```text id="chg_structure25"
3.6 g/cm³
```

The exact value depends on the material.

---

# 19.240 Visualizing the Structure (Optional)

Although CHGNet itself does not visualize crystals,

the loaded structure can be exported to visualization programs such as

* VESTA,
* OVITO,
* ASE GUI.

For example,

```python id="chg_structure26"
structure.to(filename="output.cif")
```

The exported file can then be opened in external visualization software.

---

# 19.241 Preparing the Structure for CHGNet

At this point,

the structure is fully prepared.

No additional preprocessing is required.

The workflow is

```text id="chg_structure27"
CIF File

↓

Pymatgen Structure

↓

CHGNet

↓

Predictions
```

This simplicity is one of the major strengths of CHGNet.

---

# 19.242 Complete Example

A complete script for loading a crystal structure is shown below.

```python id="chg_structure28"
from pymatgen.core import Structure

structure = Structure.from_file("LiFePO4.cif")

print("Formula:", structure.formula)
print("Number of atoms:", len(structure))
print("Lattice:")
print(structure.lattice)
```

This script provides all the essential information needed before making predictions.

---

# 19.243 Workflow Summary

The process of preparing a crystal for CHGNet can be summarized as

```text id="chg_structure29"
Crystal File

↓

Read with Pymatgen

↓

Structure Object

↓

Inspect Crystal

↓

Ready for CHGNet
```

Once the `Structure` object has been created, it can be passed directly to the pretrained model for inference.

---

## Transition to the Next Section

With both the **pretrained CHGNet model** and a **valid `Structure` object** prepared, we are now ready to perform our **first prediction**. In the next section, we will use CHGNet to predict the **total energy** of a real crystal, examine the returned prediction dictionary, interpret the output values, and understand how energy predictions are reported in practical materials informatics workflows.

# 19.244 First Energy Prediction with CHGNet

We have now completed the two essential preparation steps.

* The pretrained CHGNet model has been loaded.
* The crystal structure has been read using **pymatgen**.

We are now ready to perform our **first prediction**.

The objective is simple:

> Given a crystal structure, predict its total energy using CHGNet.

This is the fundamental operation upon which all later simulations—including relaxation, molecular dynamics, and high-throughput screening—are built.

---

# 19.245 Prediction Workflow

The complete inference pipeline is remarkably simple.

```text id="chg_predict1"
Crystal Structure

↓

CHGNet

↓

Graph Construction

↓

Message Passing

↓

Neural Network

↓

Energy Prediction
```

Although internally the model performs many sophisticated operations,

from the user's perspective only a few lines of Python are required.

---

# 19.246 Making the First Prediction

Assume that

* the model has already been loaded,
* the crystal structure has already been read.

The prediction is performed using

```python id="chg_predict2"
prediction = model.predict_structure(structure)
```

This single line

* constructs the crystal graph,
* performs message passing,
* computes atomic embeddings,
* predicts all requested physical quantities.

The results are stored in the variable

```python
prediction
```

---

# 19.247 Understanding the Prediction Object

The returned object is a Python dictionary.

Conceptually,

```text id="chg_predict3"
prediction

│

├── Energy

├── Forces

├── Stress

└── Magnetic Moments
```

Each quantity can be accessed independently.

---

# 19.248 Viewing Available Keys

To inspect the returned quantities,

execute

```python id="chg_predict4"
print(prediction.keys())
```

A typical output may resemble

```text id="chg_predict5"
dict_keys([
'e',
'f',
's',
'm'
])
```

where

* `e` → energy
* `f` → forces
* `s` → stress
* `m` → magnetic moments

Depending on the installed CHGNet version, additional metadata may also be present.

---

# 19.249 Extracting the Total Energy

The total energy is obtained using

```python id="chg_predict6"
energy = prediction["e"]

print(energy)
```

Example output

```text id="chg_predict7"
-124.873 eV
```

The numerical value will depend entirely on the crystal being analyzed.

---

# 19.250 What Does This Energy Mean?

The predicted energy represents the **total potential energy** of the entire crystal.

It includes contributions from

* chemical bonding,
* electrostatic interactions,
* local atomic environments,
* many-body interactions.

It is **not**

* energy per atom,
* formation energy,
* cohesive energy.

Instead,

it corresponds to the total energy of the structure supplied to the model.

---

# 19.251 Why Are Energies Usually Negative?

Many beginners are surprised when the predicted energy is negative.

For example,

```text id="chg_predict8"
-124.87 eV
```

This is perfectly normal.

In atomistic simulations,

the zero of energy is arbitrary.

A negative energy simply indicates that

the bonded crystal is more stable than the chosen reference state of isolated atoms.

---

# 19.252 Energy per Atom

Comparing total energies of different-sized structures can be misleading.

Instead,

researchers often compute the **energy per atom**.

The calculation is

$$
E_{\mathrm{atom}}
=================

\frac{E_{\mathrm{total}}}{N},
$$

where

* $E_{\mathrm{total}}$ is the predicted total energy,
* $N$ is the number of atoms.

---

# 19.253 Calculating Energy per Atom

Suppose the crystal contains

```text id="chg_predict9"
28 atoms
```

The energy per atom is

```python id="chg_predict10"
energy_per_atom = prediction["e"] / len(structure)

print(energy_per_atom)
```

Example output

```text id="chg_predict11"
-4.46 eV/atom
```

Energy per atom is much more useful for comparing different structures.

---

# 19.254 Why Energy per Atom Is Important

Consider two crystals.

| Structure | Total Energy |
| --------- | -----------: |
| A         |      -100 eV |
| B         |      -300 eV |

At first glance,

Structure B appears more stable.

However,

suppose

* Structure A contains 20 atoms.
* Structure B contains 80 atoms.

The energy per atom becomes

| Structure | Energy/Atom |
| --------- | ----------: |
| A         |     -5.0 eV |
| B         |    -3.75 eV |

Structure A is actually more stable on a per-atom basis.

---

# 19.255 Accessing Forces

Although we are currently focusing on energy,

the prediction already contains the atomic forces.

Retrieve them using

```python id="chg_predict12"
forces = prediction["f"]

print(forces)
```

The output is an array containing three force components for every atom.

We will study these in detail in the next section.

---

# 19.256 Accessing Stress

Similarly,

the stress tensor is available.

```python id="chg_predict13"
stress = prediction["s"]

print(stress)
```

The returned tensor can later be used for

* cell relaxation,
* pressure calculations,
* elastic property analysis.

---

# 19.257 Accessing Magnetic Moments

If the structure contains magnetic atoms,

their predicted magnetic moments can be obtained using

```python id="chg_predict14"
magmom = prediction["m"]

print(magmom)
```

Each entry corresponds to one atomic site.

---

# 19.258 Complete Prediction Script

The complete workflow from structure loading to energy prediction is shown below.

```python id="chg_predict15"
from pymatgen.core import Structure
from chgnet.model import CHGNet

structure = Structure.from_file("LiFePO4.cif")

model = CHGNet.load()

prediction = model.predict_structure(structure)

print("Total Energy:", prediction["e"], "eV")
print("Energy per Atom:",
      prediction["e"] / len(structure),
      "eV/atom")
```

This script represents the simplest practical CHGNet calculation.

---

# 19.259 Prediction Workflow Summary

The inference process can be summarized as

```text id="chg_predict16"
Load Structure

↓

Load CHGNet

↓

Predict

↓

Energy

↓

Analyze Results
```

Every subsequent CHGNet application—including relaxation, molecular dynamics, and high-throughput screening—builds upon this same prediction workflow.

---

# 19.260 Interpreting the Prediction Results

A successful energy prediction indicates that

* the crystal structure is valid,
* the pretrained model is functioning correctly,
* the computational environment has been configured properly.

At this stage,

we have verified the complete inference pipeline from **raw crystal structure** to **predicted physical properties**.

This marks the transition from software setup to genuine materials informatics research.

---

## Transition to the Next Section

While the total energy provides a measure of structural stability, it does not indicate **how the atoms should move**. The next section focuses on **force prediction**, where we will extract the force acting on every atom, interpret the force vectors, understand their units and physical meaning, visualize force arrays, and prepare these quantities for geometry optimization and molecular dynamics simulations.

# 19.261 Force Prediction with CHGNet

While the total energy tells us **how stable** a crystal is,

it does **not** tell us

* which atoms should move,
* how they should move,
* how quickly they should move.

These questions are answered by **atomic forces**.

Force prediction is one of the most important capabilities of CHGNet because it enables

* geometry optimization,
* molecular dynamics,
* defect relaxation,
* transition-state searches,
* lattice dynamics.

Without accurate forces, realistic atomistic simulations would not be possible.

---

# 19.262 From Energy to Force

The force acting on an atom is obtained from the gradient of the total energy.

Mathematically,

$$
\mathbf{F}
==========

*

\nabla E.
$$

This equation means that the force always points toward the direction in which the energy decreases most rapidly.

Consequently,

atoms naturally move toward lower-energy configurations.

---

# 19.263 Physical Meaning of Atomic Forces

Imagine placing a ball on a hillside.

```text id="chg_force1"
      ●

     /

    /

___/________
```

The ball rolls downhill because gravity acts on it.

Similarly,

an atom experiencing a force moves toward a lower-energy configuration.

In molecular simulations,

the "hill" is the **potential energy surface**, and the "rolling" is determined by the atomic forces.

---

# 19.264 Predicting Forces

CHGNet predicts the forces simultaneously with the total energy.

The prediction has already been computed as

```python id="chg_force2"
prediction = model.predict_structure(structure)
```

The force array is obtained using

```python id="chg_force3"
forces = prediction["f"]
```

The variable `forces` contains the force acting on every atom in the crystal.

---

# 19.265 Shape of the Force Array

The returned force array has dimensions

```text id="chg_force4"
(Number of Atoms) × 3
```

For example,

a crystal containing 28 atoms produces an array of shape

```text id="chg_force5"
(28, 3)
```

The three columns correspond to the

* x-component,
* y-component,
* z-component

of the force acting on each atom.

---

# 19.266 Displaying the Force Array

To inspect the complete array,

execute

```python id="chg_force6"
print(forces)
```

A typical output resembles

```text id="chg_force7"
[[ 0.01 -0.03  0.00]

 [-0.02  0.01  0.04]

 ...

 [ 0.00 -0.01  0.02]]
```

Each row corresponds to one atom.

---

# 19.267 Units of Force

The predicted forces are reported in

```text id="chg_force8"
eV/Å
```

This unit represents

* electron-volts of energy,
* per angstrom of displacement.

It is the standard force unit used by

* VASP,
* ASE,
* CHGNet,
* M3GNet,
* most atomistic simulation packages.

---

# 19.268 Accessing the Force on One Atom

The force acting on a single atom is easily extracted.

For example,

the first atom is

```python id="chg_force9"
print(forces[0])
```

Example output

```text id="chg_force10"
[0.01, -0.03, 0.00]
```

This vector represents the force components along the

* x,
* y,
* z

directions.

---

# 19.269 Interpreting Force Components

Suppose the predicted force is

```text id="chg_force11"
[0.20

-0.10

0.05]
```

This indicates

* a positive force along x,
* a negative force along y,
* a positive force along z.

Together,

these components define the direction in which the atom tends to move.

---

# 19.270 Force Magnitude

Sometimes we are interested only in the magnitude of the force.

For one atom,

the magnitude is

$$
|\mathbf{F}|
============

\sqrt{
F_x^2
+
F_y^2
+
F_z^2
}.
$$

This quantity measures the overall strength of the force regardless of direction.

---

# 19.271 Computing Force Magnitudes

Using NumPy,

the magnitude of every force vector is

```python id="chg_force12"
import numpy as np

force_magnitudes = np.linalg.norm(
    forces,
    axis=1
)

print(force_magnitudes)
```

The result is a one-dimensional array containing the force magnitude for every atom.

---

# 19.272 Maximum Force

Geometry optimization usually monitors the **largest force**.

It is computed as

```python id="chg_force13"
max_force = np.max(force_magnitudes)

print(max_force)
```

Example output

```text id="chg_force14"
0.083 eV/Å
```

This value is frequently used as the convergence criterion during relaxation.

---

# 19.273 Mean Force

The average force is also useful for diagnosing relaxation progress.

```python id="chg_force15"
mean_force = np.mean(force_magnitudes)

print(mean_force)
```

As the relaxation proceeds,

both

* the maximum force,
* the average force

should decrease steadily.

---

# 19.274 Why Forces Become Small

At equilibrium,

the crystal reaches a local minimum on the potential energy surface.

At this point,

```text id="chg_force16"
Energy

↓

Minimum

↓

Force ≈ 0
```

Thus,

small forces indicate that the structure is close to equilibrium.

---

# 19.275 Force Thresholds

Typical convergence thresholds are

| Maximum Force | Interpretation           |
| ------------- | ------------------------ |
| > 0.5 eV/Å    | Far from equilibrium     |
| 0.1–0.5 eV/Å  | Moderately relaxed       |
| < 0.05 eV/Å   | Well relaxed             |
| < 0.01 eV/Å   | High-accuracy relaxation |

The required threshold depends on the intended application.

---

# 19.276 Force Visualization

Conceptually,

forces can be visualized as arrows attached to each atom.

```text id="chg_force17"
O

↑

Fe → →

↓

Li ←
```

The

* direction

indicates where the atom tends to move,

while the

* arrow length

represents the force magnitude.

Programs such as

* OVITO,
* VESTA,
* ASE GUI

can display these vectors graphically.

---

# 19.277 Force Prediction in Molecular Dynamics

During molecular dynamics,

forces are computed at every time step.

The workflow is

```text id="chg_force18"
Atomic Positions

↓

CHGNet

↓

Forces

↓

Acceleration

↓

New Positions

↓

Repeat
```

This cycle determines the trajectory of every atom throughout the simulation.

---

# 19.278 Force Prediction During Relaxation

During geometry optimization,

forces determine how the atomic coordinates are updated.

```text id="chg_force19"
Current Structure

↓

CHGNet

↓

Forces

↓

Optimizer

↓

Updated Structure
```

When the forces become sufficiently small,

the optimization terminates.

---

# 19.279 Complete Example

The following script predicts the forces and analyzes their magnitudes.

```python id="chg_force20"
import numpy as np

forces = prediction["f"]

magnitudes = np.linalg.norm(
    forces,
    axis=1
)

print("Maximum Force:",
      np.max(magnitudes),
      "eV/Å")

print("Average Force:",
      np.mean(magnitudes),
      "eV/Å")
```

This simple analysis is routinely performed before and after geometry optimization.

---

# 19.280 Workflow Summary

The complete force prediction process is

```text id="chg_force21"
Crystal Structure

↓

CHGNet

↓

Force Array

↓

Magnitude

↓

Geometry Optimization

or

Molecular Dynamics
```

Force prediction is therefore the bridge between **static energy calculations** and **dynamic atomistic simulations**.

---

## Transition to the Next Section

We have now extracted the force acting on every atom in the crystal. The next practical step is to examine the **stress tensor prediction**, learning how to retrieve the stress matrix from CHGNet, interpret its components, understand its units, and use it for variable-cell relaxation, pressure calculations, and constant-pressure molecular dynamics simulations.

# 19.281 Stress Prediction with CHGNet

In the previous section, we studied **atomic forces**, which determine how individual atoms move.

However, crystals are not defined only by atomic positions.

The **simulation cell itself** can also deform.

It may

* expand,
* contract,
* shear,
* change shape.

These changes are governed by the **stress tensor**.

CHGNet predicts the stress tensor together with

* energy,
* forces,
* magnetic moments,

making it suitable for both atomic and cell relaxation.

---

# 19.282 What Is Stress?

Stress measures the **internal mechanical forces** acting within a material.

Suppose a crystal is compressed from both sides.

```text id="chg_stress1"
→ → →

██████

← ← ←
```

The atoms resist this compression.

This internal resistance is called **stress**.

Unlike force,

which acts on a single atom,

stress describes the mechanical state of the **entire crystal**.

---

# 19.283 Force Versus Stress

Although related,

force and stress are different physical quantities.

| Force                    | Stress                                     |
| ------------------------ | ------------------------------------------ |
| Acts on individual atoms | Acts on the crystal as a whole             |
| Vector quantity          | Second-order tensor                        |
| Unit: eV/Å               | Unit: GPa (or eV/Å³ depending on software) |
| Controls atomic motion   | Controls lattice deformation               |

Force determines how atoms move,

whereas stress determines how the simulation cell changes.

---

# 19.284 The Stress Tensor

Stress is represented by a **3 × 3 tensor**.

Mathematically,

$$
\boldsymbol{\sigma}
===================

\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} & \sigma_{xz} \
\sigma_{yx} & \sigma_{yy} & \sigma_{yz} \
\sigma_{zx} & \sigma_{zy} & \sigma_{zz}
\end{bmatrix}.
$$

The diagonal components describe **normal stresses**,

while the off-diagonal components describe **shear stresses**.

---

# 19.285 Physical Meaning of Tensor Components

The diagonal elements

* $\sigma_{xx}$,
* $\sigma_{yy}$,
* $\sigma_{zz}$

describe stretching or compression along the

* x-axis,
* y-axis,
* z-axis.

The off-diagonal elements

* $\sigma_{xy}$,
* $\sigma_{xz}$,
* $\sigma_{yz}$

describe shear deformation.

Conceptually,

```text id="chg_stress2"
Normal Stress

↓

Stretch

↓

Compression

+

Shear Stress

↓

Distortion
```

---

# 19.286 Predicting Stress

CHGNet predicts stress together with all other physical quantities.

The stress tensor is obtained using

```python id="chg_stress3"
stress = prediction["s"]
```

This variable contains the predicted stress tensor for the entire crystal.

---

# 19.287 Displaying the Stress Tensor

To inspect the prediction,

execute

```python id="chg_stress4"
print(stress)
```

Example output

```text id="chg_stress5"
[[ 0.42  0.01 -0.03]

 [ 0.01  0.39  0.02]

 [-0.03  0.02  0.45]]
```

The exact numerical values depend on the material.

---

# 19.288 Units of Stress

Different software packages report stress using different units.

Common units include

* GPa,
* eV/Å³.

ASE and CHGNet typically follow the conventions expected by atomistic simulation workflows.

Always verify the unit convention used by the software version and interface you are using before comparing with experimental or DFT results.

---

# 19.289 Hydrostatic Pressure

If the diagonal components are equal,

the crystal experiences **hydrostatic pressure**.

Conceptually,

```text id="chg_stress6"
↓

██████

↑

← →

Uniform Compression
```

This situation commonly occurs during high-pressure simulations.

---

# 19.290 Shear Stress

If the off-diagonal components are nonzero,

the crystal experiences **shear**.

```text id="chg_stress7"
██████

//////

██████
```

Instead of changing only its volume,

the simulation cell also changes shape.

---

# 19.291 Why Stress Matters

Stress prediction is essential for

* variable-cell relaxation,
* pressure calculations,
* elastic constant calculations,
* high-pressure simulations,
* NPT molecular dynamics.

Without stress,

the lattice vectors cannot be optimized correctly.

---

# 19.292 Variable-Cell Relaxation

During atomic relaxation,

only atoms move.

```text id="chg_stress8"
Atoms Move

↓

Cell Fixed
```

During variable-cell relaxation,

both atoms and the simulation cell are optimized.

```text id="chg_stress9"
Atoms Move

+

Cell Changes

↓

Lowest-Energy Structure
```

Stress determines how the lattice vectors are updated during this process.

---

# 19.293 Pressure from the Stress Tensor

Pressure is closely related to the average normal stress.

For isotropic systems,

the pressure is

$$
P
=

*

\frac{
\sigma_{xx}
+
\sigma_{yy}
+
\sigma_{zz}
}{3}.
$$

The negative sign reflects the convention that compression corresponds to positive pressure.

---

# 19.294 Residual Stress

After geometry optimization,

the stress should be close to zero.

Conceptually,

```text id="chg_stress10"
Large Stress

↓

Relaxation

↓

Small Stress
```

Residual stress indicates that the lattice has not yet reached mechanical equilibrium.

---

# 19.295 Stress During Molecular Dynamics

In constant-pressure molecular dynamics,

the simulation cell changes continuously.

The workflow is

```text id="chg_stress11"
Current Cell

↓

CHGNet

↓

Stress

↓

Cell Update

↓

Next Time Step
```

This enables realistic simulations under external pressure.

---

# 19.296 Applications of Stress Prediction

Stress prediction enables studies of

* lattice optimization,
* pressure-induced phase transitions,
* elastic deformation,
* mechanical stability,
* thermal expansion,
* equation of state calculations.

These applications are important in

* structural materials,
* battery materials,
* geophysics,
* high-pressure physics.

---

# 19.297 Complete Example

The following script retrieves and displays the stress tensor.

```python id="chg_stress12"
stress = prediction["s"]

print("Stress Tensor")

print(stress)
```

The returned matrix can then be passed to optimization algorithms or analyzed further.

---

# 19.298 Complete Prediction Pipeline

After one prediction,

CHGNet provides

```text id="chg_stress13"
Crystal Structure

↓

CHGNet

↓

Energy

↓

Forces

↓

Stress

↓

Magnetic Moments
```

These four quantities form the foundation for nearly all practical CHGNet simulations.

---

# 19.299 Workflow Summary

Stress prediction extends atomistic simulations beyond atomic motion.

The overall process is

```text id="chg_stress14"
Crystal

↓

CHGNet

↓

Stress Tensor

↓

Cell Relaxation

↓

Equilibrium Lattice
```

Together with force prediction,

stress enables full structural optimization of crystalline materials.

---

## Transition to the Next Section

We have now learned how to predict **energy**, **forces**, **stress**, and **magnetic moments** using CHGNet. The next section will combine these quantities in a complete **geometry optimization workflow using ASE**, where we will connect CHGNet to the Atomic Simulation Environment, perform automatic structure relaxation, monitor convergence, and obtain an equilibrium crystal suitable for further simulations such as molecular dynamics and high-throughput materials screening.

# 19.300 Geometry Optimization with ASE and CHGNet

One of the most common tasks in computational materials science is **geometry optimization**.

A newly generated crystal structure is rarely at its lowest-energy configuration.

The atoms may be

* slightly displaced,
* under internal stress,
* too close together,
* too far apart.

Geometry optimization automatically moves the atoms until the crystal reaches mechanical equilibrium.

CHGNet performs this optimization efficiently by providing accurate

* energies,
* forces,
* stresses

to the optimization algorithm.

---

# 19.301 Why Use ASE?

The **Atomic Simulation Environment (ASE)** is one of the most popular Python frameworks for atomistic simulations.

ASE provides

* geometry optimization,
* molecular dynamics,
* vibrational analysis,
* interfaces to many electronic structure codes,
* visualization tools.

CHGNet integrates directly with ASE, allowing researchers to use standard optimization algorithms without implementing them manually.

---

# 19.302 Overall Optimization Workflow

The complete workflow is

```text id="chg_ase1"
Crystal Structure

↓

Pymatgen

↓

ASE Atoms

↓

CHGNet Calculator

↓

Geometry Optimizer

↓

Relaxed Structure
```

Each component performs a specific task:

* **Pymatgen** reads the crystal.
* **ASE** manages the simulation.
* **CHGNet** predicts energies and forces.
* **The optimizer** updates atomic positions.

---

# 19.303 Importing Required Packages

Begin by importing the necessary modules.

```python id="chg_ase2"
from pymatgen.core import Structure
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
```

The `CHGNetCalculator` acts as a bridge between CHGNet and ASE.

---

# 19.304 Loading the Crystal

Read the crystal structure.

```python id="chg_ase3"
structure = Structure.from_file("LiFePO4.cif")
```

At this stage,

the crystal exists as a pymatgen `Structure` object.

---

# 19.305 Creating the CHGNet Model

Load the pretrained model.

```python id="chg_ase4"
model = CHGNet.load()
```

This model will provide

* energies,
* forces,
* stresses

during every optimization step.

---

# 19.306 Creating the Calculator

Next,

construct the ASE calculator.

```python id="chg_ase5"
calculator = CHGNetCalculator(model=model)
```

The calculator translates ASE requests into CHGNet predictions.

Conceptually,

```text id="chg_ase6"
ASE

↓

Calculator

↓

CHGNet

↓

Energy & Forces
```

---

# 19.307 Converting Pymatgen to ASE

ASE operates on an `Atoms` object,

while CHGNet predictions in previous sections used a pymatgen `Structure`.

Conversion is straightforward.

```python id="chg_ase7"
from pymatgen.io.ase import AseAtomsAdaptor

atoms = AseAtomsAdaptor.get_atoms(structure)
```

The variable

```python
atoms
```

now represents the crystal in ASE format.

---

# 19.308 Attaching the Calculator

Assign the calculator to the ASE object.

```python id="chg_ase8"
atoms.calc = calculator
```

From this point onward,

whenever ASE needs

* energy,
* forces,
* stress,

it automatically calls CHGNet.

---

# 19.309 Choosing an Optimizer

ASE provides several optimization algorithms.

Common choices include

* BFGS,
* LBFGS,
* FIRE,
* Quasi-Newton.

One of the most widely used is **BFGS**.

Import it using

```python id="chg_ase9"
from ase.optimize import BFGS
```

---

# 19.310 Creating the Optimizer

Construct the optimizer.

```python id="chg_ase10"
optimizer = BFGS(atoms)
```

The optimizer now controls how atomic positions are updated.

CHGNet supplies the forces,

while BFGS determines the next atomic configuration.

---

# 19.311 Running the Optimization

Start the relaxation using

```python id="chg_ase11"
optimizer.run(fmax=0.01)
```

The parameter

```text
fmax = 0.01
```

means

> Stop when the maximum force acting on any atom becomes smaller than **0.01 eV/Å**.

This is a commonly used convergence criterion.

---

# 19.312 What Happens Internally?

Although only one line of code is executed,

many operations occur repeatedly.

```text id="chg_ase12"
Current Structure

↓

CHGNet

↓

Energy

↓

Forces

↓

BFGS

↓

Move Atoms

↓

New Structure

↓

Repeat
```

This iterative process continues until convergence.

---

# 19.313 Monitoring Convergence

During optimization,

ASE prints information similar to

```text id="chg_ase13"
Step

Energy

Maximum Force
```

A typical optimization log looks like

```text
Step   Energy (eV)   Max Force (eV/Å)

0      -123.41       0.85

1      -124.07       0.42

2      -124.60       0.18

3      -124.82       0.05

4      -124.87       0.009
```

Notice that

* energy decreases,
* forces become smaller,

indicating convergence toward equilibrium.

---

# 19.314 Accessing the Relaxed Structure

After optimization,

the variable

```python
atoms
```

contains the relaxed crystal.

It now represents the equilibrium geometry predicted by CHGNet.

---

# 19.315 Saving the Relaxed Structure

Save the optimized structure as a CIF file.

```python id="chg_ase14"
from ase.io import write

write("relaxed.cif", atoms)
```

This file can be

* visualized,
* used in DFT,
* used for molecular dynamics,
* shared with collaborators.

---

# 19.316 Converting Back to Pymatgen

Many materials science workflows use pymatgen.

Convert the relaxed ASE object back to a `Structure`.

```python id="chg_ase15"
relaxed_structure = AseAtomsAdaptor.get_structure(atoms)
```

The optimized crystal is now compatible with

* pymatgen,
* Materials Project workflows,
* other machine learning tools.

---

# 19.317 Complete Geometry Optimization Script

The complete workflow is shown below.

```python id="chg_ase16"
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator

from ase.optimize import BFGS

structure = Structure.from_file("LiFePO4.cif")

atoms = AseAtomsAdaptor.get_atoms(structure)

model = CHGNet.load()

atoms.calc = CHGNetCalculator(model=model)

optimizer = BFGS(atoms)

optimizer.run(fmax=0.01)

write("relaxed.cif", atoms)
```

This script performs a complete CHGNet geometry optimization using ASE.

---

# 19.318 Choosing the Force Criterion

Different research problems require different convergence thresholds.

| Maximum Force | Typical Use                      |
| ------------- | -------------------------------- |
| 0.10 eV/Å     | Quick screening                  |
| 0.05 eV/Å     | Standard relaxation              |
| 0.02 eV/Å     | High-quality optimization        |
| 0.01 eV/Å     | Publication-quality calculations |
| 0.005 eV/Å    | Very high accuracy               |

Smaller thresholds improve accuracy but require more optimization steps.

---

# 19.319 Why CHGNet Is Faster Than DFT

With DFT,

every optimization step requires solving the electronic structure problem.

```text id="chg_ase17"
Structure

↓

DFT

↓

Energy

↓

Forces

↓

Next Step
```

With CHGNet,

the neural network directly predicts these quantities.

```text id="chg_ase18"
Structure

↓

CHGNet

↓

Energy

↓

Forces

↓

Next Step
```

This dramatically reduces computational cost while maintaining good accuracy for many systems.

---

# 19.320 Practical Applications

Geometry optimization using CHGNet is widely used for

* crystal structure refinement,
* defect relaxation,
* surface optimization,
* doped materials,
* battery electrodes,
* catalysts,
* high-throughput materials screening,
* preparing structures for DFT calculations.

In many research workflows,

CHGNet serves as a **pre-relaxation tool**, reducing the number of expensive DFT optimization steps.

---

# 19.321 Workflow Summary

The complete optimization process can be summarized as

```text id="chg_ase19"
Read Crystal

↓

Convert to ASE

↓

Attach CHGNet

↓

Run BFGS

↓

Relaxed Structure

↓

Save Results
```

This workflow forms the foundation of practical CHGNet simulations and is one of the most frequently used procedures in modern computational materials science.

---

## Transition to the Next Section

With geometry optimization complete, we are ready to simulate **finite-temperature atomic motion**. In the next section, we will integrate CHGNet with ASE's molecular dynamics engines, initialize atomic velocities, select simulation ensembles (NVE, NVT, and NPT), control temperature using thermostats, and perform realistic molecular dynamics simulations suitable for studying diffusion, thermal stability, and phase transitions.
# 19.322 Molecular Dynamics Simulations with CHGNet and ASE

Geometry optimization determines the **lowest-energy structure** of a crystal.

However,

real materials exist at finite temperature, where atoms are continuously moving.

To investigate these dynamic processes, we use **Molecular Dynamics (MD)**.

CHGNet enables molecular dynamics simulations by providing fast predictions of

* energies,
* forces,
* stresses,

allowing trajectories approaching DFT quality while being several orders of magnitude faster.

---

# 19.323 Why Perform Molecular Dynamics?

Static calculations answer questions such as

* What is the equilibrium crystal structure?
* What is the total energy?
* Is the material mechanically stable?

Dynamic simulations answer different questions.

For example,

* How do lithium ions diffuse?
* How does temperature affect the crystal?
* Will the structure melt?
* How do defects migrate?
* What happens during phase transitions?

These phenomena require the evolution of atomic positions over time.

---

# 19.324 Molecular Dynamics Workflow

The complete MD workflow is

```text id="chg_md_workflow1"
Initial Structure

↓

Assign Velocities

↓

CHGNet

↓

Energy & Forces

↓

Newton's Equations

↓

Updated Positions

↓

Repeat
```

This cycle is repeated thousands or millions of times.

---

# 19.325 ASE Molecular Dynamics

ASE provides several molecular dynamics engines.

Common choices include

* Velocity Verlet
* Langevin Dynamics
* Berendsen Dynamics
* Nose–Hoover Dynamics

Each algorithm corresponds to a different thermodynamic ensemble.

---

# 19.326 Common Thermodynamic Ensembles

| Ensemble | Constant Quantities           | Typical Use                      |
| -------- | ----------------------------- | -------------------------------- |
| NVE      | Number, Volume, Energy        | Energy-conserving simulations    |
| NVT      | Number, Volume, Temperature   | Constant-temperature simulations |
| NPT      | Number, Pressure, Temperature | Variable-cell simulations        |

Different scientific questions require different ensembles.

---

# 19.327 Preparing the Structure

Load the crystal.

```python id="chg_md_code1"
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

structure = Structure.from_file("LiFePO4.cif")

atoms = AseAtomsAdaptor.get_atoms(structure)
```

---

# 19.328 Loading CHGNet

Load the pretrained model.

```python id="chg_md_code2"
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator

model = CHGNet.load()

atoms.calc = CHGNetCalculator(model=model)
```

The calculator now provides forces whenever ASE requests them.

---

# 19.329 Assigning Initial Velocities

Before atoms can move,

they must be assigned initial velocities.

ASE generates velocities according to the Maxwell–Boltzmann distribution.

```python id="chg_md_code3"
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=300
)
```

This initializes the system at

```text id="chg_md_temp1"
300 K
```

---

# 19.330 Why Random Velocities?

At room temperature,

atoms move randomly because of thermal energy.

The Maxwell–Boltzmann distribution reproduces this physical behavior.

Conceptually,

```text id="chg_md_temp2"
Temperature

↓

Random Velocities

↓

Atomic Motion
```

Different temperatures produce different velocity distributions.

---

# 19.331 Choosing the Time Step

The integration time step must be very small.

A typical value is

```text id="chg_md_temp3"
1 fs

=

10⁻¹⁵ s
```

This is sufficiently small to resolve atomic vibrations.

---

# 19.332 Running an NVE Simulation

One of the simplest MD algorithms is Velocity Verlet.

```python id="chg_md_code4"
from ase.md.verlet import VelocityVerlet
from ase import units

dyn = VelocityVerlet(
    atoms,
    timestep=1 * units.fs
)
```

This performs a constant-energy simulation.

---

# 19.333 Running the Simulation

Execute the dynamics.

```python id="chg_md_code5"
dyn.run(1000)
```

This performs

```text id="chg_md_temp4"
1000 time steps
```

At a timestep of

```text
1 fs
```

the total simulated time becomes

```text id="chg_md_temp5"
1000 fs

=

1 ps
```

---

# 19.334 Monitoring the Simulation

During molecular dynamics,

it is useful to monitor

* energy,
* temperature,
* time.

A simple logging function is

```python id="chg_md_code6"
def print_status():

    print(
        atoms.get_potential_energy(),
        atoms.get_temperature()
    )
```

This function can be attached to the MD engine.

---

# 19.335 Attaching a Logger

ASE allows functions to execute automatically during the simulation.

```python id="chg_md_code7"
dyn.attach(
    print_status,
    interval=10
)
```

Every

```text id="chg_md_temp6"
10 steps
```

the current energy and temperature are printed.

---

# 19.336 Saving the Trajectory

Simulation trajectories should usually be saved.

```python id="chg_md_code8"
from ase.io.trajectory import Trajectory

traj = Trajectory(
    "md.traj",
    "w",
    atoms
)

dyn.attach(
    traj.write,
    interval=1
)
```

Every time step is now stored in

```text id="chg_md_temp7"
md.traj
```

---

# 19.337 Visualizing the Trajectory

Trajectory files can later be viewed using

* ASE GUI
* OVITO
* VMD

These programs animate atomic motion throughout the simulation.

---

# 19.338 Constant-Temperature Dynamics

Many simulations require a fixed temperature.

ASE provides Langevin dynamics.

```python id="chg_md_code9"
from ase.md.langevin import Langevin

dyn = Langevin(
    atoms,
    timestep=1 * units.fs,
    temperature_K=300,
    friction=0.01
)
```

This thermostat continuously regulates the temperature.

---

# 19.339 Why Use a Thermostat?

Without a thermostat,

small numerical errors may gradually change the temperature.

A thermostat removes or adds kinetic energy when necessary.

Conceptually,

```text id="chg_md_temp8"
Temperature

↓

Thermostat

↓

Target Temperature
```

---

# 19.340 Diffusion Example

Suppose lithium ions move through a cathode.

Initially,

```text id="chg_md_temp9"
Li

↓

Site A
```

Later,

```text id="chg_md_temp10"
Li

↓

Site B
```

Repeating this motion many times produces ionic diffusion.

The diffusion coefficient can then be computed from the trajectory.

---

# 19.341 Mean Squared Displacement

Diffusion is commonly analyzed using the **mean squared displacement (MSD)**.

For a particle,

$$
\mathrm{MSD}(t)
===============

\left<
\left|
\mathbf{r}(t)-\mathbf{r}(0)
\right|^2
\right>.
$$

A larger MSD indicates greater atomic mobility.

MSD is one of the most frequently calculated quantities from molecular dynamics simulations.

---

# 19.342 Radial Distribution Function

Another important analysis is the radial distribution function,

denoted

$$
g(r).
$$

It measures the probability of finding neighboring atoms at a distance

$$
r.
$$

The radial distribution function provides insight into

* crystal order,
* liquids,
* amorphous materials,
* melting.

---

# 19.343 Complete Molecular Dynamics Script

The following example performs a simple CHGNet molecular dynamics simulation.

```python id="chg_md_code10"
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=300
)

dyn = VelocityVerlet(
    atoms,
    timestep=1 * units.fs
)

dyn.run(1000)
```

This script represents the basic workflow used in many research projects.

---

# 19.344 Practical Applications

CHGNet molecular dynamics is widely applied to

* lithium diffusion,
* sodium diffusion,
* thermal stability,
* melting,
* phase transitions,
* defect migration,
* ionic conductivity,
* high-temperature crystal behavior,
* amorphization.

These simulations enable researchers to investigate atomistic processes that are difficult to observe experimentally.

---

# 19.345 Workflow Summary

The complete molecular dynamics process is

```text id="chg_md_summary1"
Load Structure

↓

Load CHGNet

↓

Assign Velocities

↓

Choose MD Algorithm

↓

Run Simulation

↓

Save Trajectory

↓

Analyze Diffusion

Temperature

Structure
```

Molecular dynamics transforms CHGNet from a static prediction model into a powerful simulation engine capable of investigating real-time atomic behavior over extended time scales.

---

## Transition to the Next Section

We have now completed the core practical capabilities of CHGNet, including **energy prediction, force prediction, stress prediction, geometry optimization, and molecular dynamics**. In the next section, we will build **high-throughput workflows**, where CHGNet automatically evaluates hundreds or thousands of crystal structures, filters candidates based on predicted stability, and accelerates large-scale materials discovery pipelines.

# 19.346 High-Throughput Materials Screening with CHGNet

One of the greatest advantages of machine learning interatomic potentials is that they enable **high-throughput materials screening**.

Instead of studying a single material,

we can automatically evaluate

* hundreds,
* thousands,
* or even millions

of candidate structures.

Because CHGNet predicts energies and forces in milliseconds to seconds (depending on system size and hardware), it is well suited for large-scale computational screening.

---

# 19.347 What Is High-Throughput Screening?

High-throughput screening (HTS) is the automated evaluation of a large number of candidate materials using a computational workflow.

Instead of manually studying one crystal at a time,

the computer repeatedly performs

1. Read a structure.
2. Predict its properties.
3. Store the results.
4. Move to the next structure.

Conceptually,

```text id="chg_ht1"
Structure 1

↓

Prediction

↓

Structure 2

↓

Prediction

↓

Structure 3

↓

Prediction

↓

...

↓

Database of Results
```

---

# 19.348 Why Use CHGNet for High-Throughput Studies?

Performing DFT calculations for thousands of materials is computationally expensive.

For example,

```text id="chg_ht2"
10,000 Structures

↓

DFT

↓

Months of Computing
```

With CHGNet,

the workflow becomes

```text id="chg_ht3"
10,000 Structures

↓

CHGNet

↓

Hours (or Less)
```

Although DFT remains more accurate,

CHGNet dramatically accelerates the initial screening stage.

---

# 19.349 Typical High-Throughput Workflow

A complete screening pipeline is

```text id="chg_ht4"
Candidate Structures

↓

CHGNet

↓

Energy

↓

Filter Stable Structures

↓

DFT Verification

↓

Final Candidates
```

Only the most promising materials proceed to expensive DFT calculations.

---

# 19.350 Organizing Structure Files

Suppose a directory contains many CIF files.

```text id="chg_ht5"
materials/

├── LiFePO4.cif

├── LiCoO2.cif

├── LiMn2O4.cif

├── NaFePO4.cif

├── NaMnO2.cif

└── ...
```

Our goal is to process every file automatically.

---

# 19.351 Reading Multiple Structures

Python's `glob` module can locate all CIF files.

```python id="chg_ht_code1"
from glob import glob

cif_files = glob("materials/*.cif")

print(len(cif_files))
```

The variable `cif_files` now contains the paths to every crystal structure.

---

# 19.352 Loading the CHGNet Model

The model should be loaded **once** before the loop.

```python id="chg_ht_code2"
from chgnet.model import CHGNet

model = CHGNet.load()
```

Loading the model repeatedly inside the loop would waste computation time.

---

# 19.353 Looping Through Structures

A simple screening loop is

```python id="chg_ht_code3"
from pymatgen.core import Structure

for file in cif_files:

    structure = Structure.from_file(file)

    prediction = model.predict_structure(structure)

    print(file, prediction["e"])
```

This predicts the total energy for every crystal.

---

# 19.354 Computing Energy per Atom

Different structures may contain different numbers of atoms.

Therefore,

energy per atom should usually be reported.

```python id="chg_ht_code4"
energy = prediction["e"]

energy_per_atom = energy / len(structure)
```

Energy per atom enables fair comparisons between structures of different sizes.

---

# 19.355 Storing the Results

The results can be collected in a list.

```python id="chg_ht_code5"
results = []

for file in cif_files:

    structure = Structure.from_file(file)

    prediction = model.predict_structure(structure)

    results.append({
        "file": file,
        "energy": prediction["e"],
        "energy_per_atom":
            prediction["e"] / len(structure)
    })
```

Each entry stores both the file name and its predicted energy.

---

# 19.356 Creating a DataFrame

For further analysis,

convert the results into a pandas DataFrame.

```python id="chg_ht_code6"
import pandas as pd

df = pd.DataFrame(results)

print(df)
```

The table can now be

* sorted,
* filtered,
* exported,
* visualized.

---

# 19.357 Saving the Results

Save the screening results as a CSV file.

```python id="chg_ht_code7"
df.to_csv(
    "screening_results.csv",
    index=False
)
```

The output file can later be opened in

* Excel,
* Python,
* R,
* MATLAB.

---

# 19.358 Ranking Materials

Sort the candidates according to energy per atom.

```python id="chg_ht_code8"
df = df.sort_values(
    "energy_per_atom"
)

print(df.head())
```

The structures with the lowest energy per atom appear at the top.

These are often the most stable candidates.

---

# 19.359 Filtering Stable Structures

Suppose we wish to keep only structures satisfying

```text id="chg_ht6"
Energy per Atom

<

-5 eV
```

This is easily done.

```python id="chg_ht_code9"
stable = df[
    df["energy_per_atom"] < -5
]
```

Only promising materials remain.

---

# 19.360 High-Throughput Relaxation

Instead of predicting only energies,

every structure can first be relaxed.

Conceptually,

```text id="chg_ht7"
Structure

↓

CHGNet Relaxation

↓

Relaxed Structure

↓

Energy

↓

Store Results
```

This generally improves the reliability of the screening process.

---

# 19.361 Parallel Screening

Large datasets benefit from parallel computation.

Instead of

```text id="chg_ht8"
Structure

↓

Structure

↓

Structure
```

multiple CPU cores or GPUs can process structures simultaneously.

```text id="chg_ht9"
GPU

↓

Structure 1

Structure 2

Structure 3

Structure 4

↓

Results
```

Parallelization greatly reduces total runtime.

---

# 19.362 Typical Screening Criteria

Researchers often filter candidates according to

* low energy,
* low residual force,
* low stress,
* desired magnetic moment,
* target composition,
* crystal symmetry.

Several criteria can be combined into one workflow.

---

# 19.363 Combining CHGNet with DFT

A common strategy is

```text id="chg_ht10"
100,000 Candidates

↓

CHGNet

↓

500 Candidates

↓

DFT

↓

20 Candidates

↓

Experiment
```

This hierarchical approach minimizes computational cost while maintaining high accuracy.

---

# 19.364 Real Research Applications

High-throughput CHGNet screening has applications in

* battery materials,
* catalysts,
* superconductors,
* thermoelectrics,
* solid electrolytes,
* hydrogen storage,
* photovoltaics,
* magnetic materials.

In each case,

CHGNet serves as a fast pre-screening tool before detailed first-principles calculations.

---

# 19.365 Complete Screening Script

A minimal end-to-end example is shown below.

```python id="chg_ht_code10"
from glob import glob

import pandas as pd

from pymatgen.core import Structure
from chgnet.model import CHGNet

model = CHGNet.load()

results = []

for file in glob("materials/*.cif"):

    structure = Structure.from_file(file)

    prediction = model.predict_structure(structure)

    results.append({
        "file": file,
        "energy": prediction["e"],
        "energy_per_atom":
            prediction["e"] / len(structure)
    })

df = pd.DataFrame(results)

df.sort_values(
    "energy_per_atom"
).to_csv(
    "results.csv",
    index=False
)
```

This script forms the basis of many automated materials discovery pipelines.

---

# 19.366 Workflow Summary

The complete high-throughput workflow is

```text id="chg_ht11"
Many Crystal Structures

↓

CHGNet

↓

Energy Prediction

↓

Rank Candidates

↓

Filter

↓

DFT Verification

↓

Experimental Validation
```

High-throughput screening is one of the primary reasons machine learning potentials have become indispensable in modern computational materials science. By rapidly narrowing vast chemical spaces to a manageable set of promising candidates, CHGNet enables researchers to focus expensive DFT calculations and experimental efforts where they are most likely to produce new materials discoveries.

---

## Transition to the Next Section

The practical CHGNet workflow is now complete. In the next section, we will explore **best practices, limitations, and common pitfalls** when using CHGNet in research. We will discuss where CHGNet performs exceptionally well, situations in which DFT should still be preferred, transferability limits, dataset bias, numerical stability, and recommendations for building reliable publication-quality materials informatics workflows.

# 19.367 Best Practices, Limitations, and Common Pitfalls of CHGNet

Throughout this chapter, we have learned how to use CHGNet for

* energy prediction,
* force prediction,
* stress prediction,
* geometry optimization,
* molecular dynamics,
* high-throughput screening.

Although CHGNet is a powerful model, it is **not a replacement for Density Functional Theory (DFT)** in every situation.

Understanding both its strengths and limitations is essential for conducting reliable scientific research.

---

# 19.368 When Should You Use CHGNet?

CHGNet performs exceptionally well for problems involving

* rapid structure evaluation,
* geometry optimization,
* molecular dynamics,
* large-scale screening,
* generating initial structures for DFT,
* exploratory materials discovery.

Whenever thousands of structures must be evaluated quickly,

CHGNet is often the preferred choice.

---

# 19.369 When Should You Still Use DFT?

Despite its impressive performance,

certain problems still require first-principles calculations.

Examples include

* highly accurate total energies,
* electronic band structures,
* density of states,
* charge density analysis,
* optical properties,
* phonon calculations,
* reaction barriers,
* systems very different from the training data.

For these applications,

DFT remains the reference method.

---

# 19.370 CHGNet as a DFT Accelerator

A useful way to think about CHGNet is

```text id="chg_limit1"
Machine Learning

↓

Fast Approximation

↓

Select Good Candidates

↓

DFT

↓

Final Verification
```

Rather than replacing DFT,

CHGNet significantly reduces the number of expensive DFT calculations required.

---

# 19.371 Understand the Training Domain

Like every machine learning model,

CHGNet learns patterns from data.

Its predictions are most reliable for materials that resemble those contained in its training dataset.

If a crystal is far outside this domain,

prediction accuracy may decrease.

Always ask

> Is my material similar to the data used during training?

---

# 19.372 Out-of-Distribution Materials

Examples of challenging systems include

* unusual crystal chemistries,
* extremely high-pressure phases,
* highly defective structures,
* exotic magnetic materials,
* structures with uncommon bonding environments.

These are called **out-of-distribution (OOD)** samples.

Machine learning models generally perform less reliably on OOD data.

---

# 19.373 Never Trust One Prediction Blindly

A single prediction should never be accepted without evaluation.

Good scientific practice includes

* comparing with DFT,
* checking literature values,
* validating against experiments,
* performing consistency checks.

Machine learning predictions should always be interpreted critically.

---

# 19.374 Verify Relaxed Structures

After geometry optimization,

always inspect the relaxed crystal.

Check for

* unrealistic bond lengths,
* overlapping atoms,
* broken structures,
* unreasonable lattice parameters.

Visualization software such as

* VESTA,
* OVITO,
* ASE GUI

is extremely useful for this purpose.

---

# 19.375 Monitor Force Convergence

Do not stop an optimization simply because the program finishes.

Instead,

verify that the maximum force satisfies the desired convergence criterion.

Typical values are

```text id="chg_limit2"
0.05 eV/Å

or

0.01 eV/Å
```

depending on the required accuracy.

---

# 19.376 Compare Energy per Atom

When comparing structures with different numbers of atoms,

never compare

```text id="chg_limit3"
Total Energy
```

Instead,

compare

```text id="chg_limit4"
Energy per Atom
```

This avoids misleading conclusions caused by differences in system size.

---

# 19.377 Check Structural Symmetry

Optimization occasionally breaks crystal symmetry.

After relaxation,

use pymatgen to analyze the symmetry.

```python id="chg_limit_code1"
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

analyzer = SpacegroupAnalyzer(relaxed_structure)

print(analyzer.get_space_group_symbol())
```

Comparing the space group before and after optimization can reveal unexpected structural changes.

---

# 19.378 Validate with DFT

For publication-quality research,

important CHGNet predictions should be confirmed using DFT.

Typical workflow

```text id="chg_limit5"
CHGNet

↓

Promising Candidate

↓

DFT Relaxation

↓

Property Calculation

↓

Publication
```

This approach combines the speed of machine learning with the accuracy of first-principles methods.

---

# 19.379 Beware of Extrapolation

Machine learning excels at **interpolation** but is less reliable for **extrapolation**.

```text id="chg_limit6"
Training Data

↓

Interpolation

✓ Reliable

Outside Training Data

↓

Extrapolation

⚠ Less Reliable
```

Avoid drawing strong conclusions from predictions made far outside the training domain.

---

# 19.380 Use Reasonable Simulation Conditions

When performing molecular dynamics,

avoid unrealistic parameters.

Examples include

* excessively large time steps,
* extremely high temperatures,
* unstable initial structures.

Reasonable simulation settings improve both numerical stability and physical realism.

---

# 19.381 Save Intermediate Results

Large simulations should save

* trajectories,
* relaxed structures,
* energies,
* log files.

This prevents data loss if the simulation is interrupted and allows later analysis without repeating expensive calculations.

---

# 19.382 Record Software Versions

Scientific reproducibility requires documenting

* CHGNet version,
* PyTorch version,
* ASE version,
* pymatgen version,
* Python version.

A simple example is

```python id="chg_limit_code2"
import chgnet
import torch
import ase
import pymatgen

print(chgnet.__version__)
print(torch.__version__)
print(ase.__version__)
print(pymatgen.__version__)
```

Recording versions makes it easier for others to reproduce your results.

---

# 19.383 Use Version Control

For research projects,

store scripts using version control systems such as Git.

A recommended project structure is

```text id="chg_limit7"
Project/

├── data/

├── scripts/

├── notebooks/

├── results/

├── figures/

├── README.md

└── requirements.txt
```

Good organization improves collaboration and reproducibility.

---

# 19.384 Common Mistakes Made by Beginners

Frequent mistakes include

* comparing total energies instead of energy per atom,
* forgetting to relax structures before analysis,
* using unrealistic molecular dynamics settings,
* ignoring force convergence,
* trusting predictions without validation,
* using incompatible software versions,
* failing to save intermediate results.

Avoiding these errors leads to more reliable scientific work.

---

# 19.385 Recommended Research Workflow

A practical workflow for materials discovery is

```text id="chg_limit8"
Generate Candidate Structures

↓

CHGNet Screening

↓

Geometry Optimization

↓

Property Prediction

↓

Select Best Candidates

↓

DFT Validation

↓

Experimental Verification
```

This workflow balances computational efficiency with scientific rigor.

---

# 19.386 Strengths of CHGNet

Major advantages include

* near-DFT accuracy for many systems,
* excellent force prediction,
* efficient molecular dynamics,
* magnetic moment prediction,
* rapid geometry optimization,
* compatibility with ASE,
* integration with pymatgen,
* suitability for high-throughput screening.

These features make CHGNet one of the most versatile machine learning interatomic potentials currently available.

---

# 19.387 Limitations of CHGNet

Important limitations include

* dependence on training data,
* reduced reliability for out-of-distribution systems,
* inability to directly replace electronic structure calculations,
* approximation errors relative to DFT,
* limited transferability to entirely new chemical spaces.

Recognizing these limitations is essential for responsible scientific use.

---

# 19.388 Chapter Summary

In this chapter, we have built a complete practical workflow using CHGNet.

We learned how to

* install CHGNet,
* load pretrained models,
* read crystal structures,
* predict energies,
* predict forces,
* predict stresses,
* perform geometry optimization,
* run molecular dynamics,
* conduct high-throughput screening,
* apply best practices for reliable research.

Together, these topics provide the foundation needed to use CHGNet effectively in modern materials informatics and computational materials science.

---

# 19.389 Looking Ahead

CHGNet represents one of the most advanced machine learning interatomic potentials available today. However, the field continues to evolve rapidly.

The next chapter introduces a broader class of models known as **Equivariant Graph Neural Networks (EGNNs)**. These architectures explicitly respect the rotational and translational symmetries of three-dimensional space, enabling even more accurate predictions of atomic interactions. We will study the mathematical foundations of equivariance, the symmetry groups **SO(3)** and **E(3)**, tensor-based message passing, and state-of-the-art architectures such as **e3nn**, **NequIP**, and **Allegro**, which are driving the next generation of machine-learning force fields for materials science.

