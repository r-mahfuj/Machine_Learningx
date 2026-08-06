Excellent. Since this chapter is large (likely **70–100 pages**), I'll write it in the same textbook style as the strongest chapters in the book, with deep explanations, derivations, and substantial code. We'll begin with the introduction and motivation.

---

# Chapter 18 — M3GNet: A Universal Graph Neural Network for Atomistic Materials Modeling

Until this point in the book, we have explored several graph neural network architectures that have transformed materials informatics. We began with **CGCNN**, which demonstrated that crystal structures could be represented as graphs and learned directly without handcrafted descriptors. We then studied **SchNet**, which introduced continuous-filter convolutions and enabled smooth learning from interatomic distances. Next, we examined **MEGNet**, which extended graph neural networks by jointly updating atoms, bonds, and global state variables. Finally, we explored **ALIGNN**, which incorporated explicit bond-angle information through line graphs, achieving state-of-the-art performance on many crystal property prediction tasks.

Although each of these architectures represents a significant advancement, they share a common limitation: **their primary objective is property prediction**. Given a crystal structure, they predict one or more material properties such as formation energy, band gap, elastic modulus, or dielectric constant. While this capability is extremely valuable, it is not sufficient for many problems in computational materials science.

Modern atomistic simulations require considerably more than scalar property prediction. Researchers often need to answer questions such as:

* How will the atoms move when heated?
* How does a crystal relax to its equilibrium structure?
* What are the forces acting on each atom?
* How does the lattice respond to external stress?
* Can we simulate thousands of time steps of molecular dynamics without performing expensive Density Functional Theory (DFT) calculations?

Answering these questions requires not only predicting the total energy of a system, but also accurately describing the **entire potential energy surface (PES)**. Once the potential energy surface is known, quantities such as atomic forces, stresses, equilibrium structures, diffusion pathways, and molecular dynamics trajectories can all be obtained through differentiation and numerical integration.

Traditional approaches rely on first-principles methods such as Density Functional Theory. While DFT provides remarkable accuracy, it is computationally expensive. Even a single geometry optimization for a moderately sized crystal may require hours or days on high-performance computing clusters. Molecular dynamics simulations requiring millions of force evaluations become prohibitively expensive when each force calculation is obtained from DFT.

Classical empirical force fields provide a faster alternative, but their predictive power is often limited because they rely on manually designed functional forms. These force fields are typically parameterized for a narrow class of materials and struggle to generalize to chemically diverse systems.

This gap between accuracy and computational efficiency motivated the development of **machine learning interatomic potentials (MLIPs)**. Instead of assuming a predefined analytical potential, MLIPs learn the relationship between atomic configurations and their energies directly from large datasets generated using first-principles calculations.

Among the many machine learning interatomic potentials proposed in recent years, **M3GNet (Materials Graph Neural Network)** represents one of the most influential and widely adopted architectures. Developed by researchers from the Materials Virtual Lab, M3GNet extends graph neural networks beyond property prediction and transforms them into a **universal atomistic modeling framework**.

Unlike earlier graph neural networks, M3GNet is capable of simultaneously predicting

* total energy,
* atomic forces,
* stress tensors,

using a single unified neural network.

This capability allows M3GNet to function as a machine learning interatomic potential suitable for

* geometry optimization,
* crystal relaxation,
* molecular dynamics,
* phonon calculations,
* diffusion studies,
* defect simulations,
* large-scale atomistic modeling.

One of the most important innovations introduced by M3GNet is its explicit incorporation of **three-body interactions**. Earlier graph neural networks primarily modeled pairwise interactions between neighboring atoms. However, many materials derive their structural stability and physical properties from angular relationships among groups of three atoms. By incorporating these three-body geometric interactions into the message-passing process, M3GNet achieves significantly higher accuracy across a broad range of crystalline materials.

Another major strength of M3GNet is its **universality**. Rather than being trained for a single property or a single material family, M3GNet is designed to learn a transferable representation that generalizes across the periodic table. A single trained model can be applied to thousands of different materials, making it particularly valuable for high-throughput materials discovery.

Throughout this chapter, we will study M3GNet from both theoretical and practical perspectives. We will begin by understanding why universal interatomic potentials are needed and how they differ from conventional graph neural networks. We will then examine the physical motivation for incorporating three-body interactions and explore how M3GNet represents crystal structures using atoms, bonds, and angular relationships.

Next, we will derive the complete mathematical formulation of the M3GNet architecture, including message-passing equations, energy prediction, force prediction through automatic differentiation, and stress computation. Building upon this theoretical foundation, we will implement M3GNet step by step using PyTorch, construct training pipelines, and learn how the model can be employed for geometry optimization, molecular dynamics, and large-scale materials simulations.

Finally, we will explore real-world applications of M3GNet in computational materials science and compare its capabilities with previously studied architectures such as CGCNN, SchNet, MEGNet, and ALIGNN.

By the end of this chapter, the reader will understand not only how M3GNet works internally but also why it has become one of the most powerful graph neural network architectures for atomistic materials modeling and one of the leading machine learning interatomic potentials available today.

---

# 18.1 Motivation

Machine learning has fundamentally changed the way researchers predict materials properties. Instead of performing expensive first-principles calculations for every candidate material, graph neural networks can learn the relationship between crystal structures and material properties directly from data. As we have seen in the previous chapters, this approach has dramatically accelerated tasks such as formation energy prediction, band gap estimation, elastic property prediction, and high-throughput materials screening.

However, predicting a single scalar property is only one aspect of computational materials science. Many scientific problems require a complete description of how atoms interact and evolve over time. A researcher interested in structural relaxation, defect migration, phase transitions, thermal transport, or diffusion cannot rely solely on predicted formation energies. These applications require access to the **potential energy surface**, from which forces and stresses can be derived.

The potential energy surface (PES) is a multidimensional function that assigns a total energy to every possible arrangement of atoms in a system. For a crystal containing $N$ atoms, the energy depends on the positions of all atoms,

$$
E = E(\mathbf{R}_1,\mathbf{R}_2,\ldots,\mathbf{R}_N),
$$

where $\mathbf{R}_i$ denotes the position vector of the $i^{\text{th}}$ atom.

Knowledge of this function is extraordinarily powerful. Once the total energy is known, atomic forces are obtained by differentiating the energy with respect to atomic positions, while stresses are obtained by differentiating with respect to lattice deformation. Consequently, the entire atomistic behavior of a material can be described using a single differentiable energy model.

Unfortunately, constructing an accurate potential energy surface is one of the most challenging problems in computational materials science. The function depends on an enormous number of atomic configurations, chemical environments, and geometric arrangements. Analytical force fields attempt to approximate this relationship using predefined equations, but these approximations often fail when applied to chemically diverse systems or complex bonding environments.

Density Functional Theory avoids predefined functional forms by solving quantum mechanical equations directly, but this accuracy comes at a substantial computational cost. For systems containing hundreds or thousands of atoms, repeated DFT calculations become impractical, particularly for long molecular dynamics simulations or large-scale structural optimization.

Machine learning interatomic potentials address this challenge by learning the potential energy surface directly from DFT-generated data. Instead of assuming a fixed mathematical form, the neural network discovers the underlying relationship between atomic structure and energy during training. Once trained, the model can evaluate energies and forces many orders of magnitude faster than DFT while maintaining near first-principles accuracy for systems similar to those represented in the training data.

M3GNet was developed precisely to bridge this gap. It combines the expressive power of graph neural networks with physically meaningful geometric representations, enabling accurate predictions of energies, forces, and stresses within a single unified architecture. More importantly, it extends graph neural networks beyond static property prediction into the broader domain of atomistic simulation, where learned interatomic potentials can replace expensive quantum mechanical calculations in many practical applications.

---

**Next section:** **18.2 From Property Prediction to Universal Interatomic Potentials**. This section will explain classical force fields (Lennard–Jones, EAM, Tersoff, Stillinger–Weber, ReaxFF), introduce machine learning interatomic potentials, and motivate why M3GNet represents a major step toward universal, transferable atomistic potentials.


# 18.2 From Property Prediction to Universal Interatomic Potentials

In the previous section, we learned that predicting a single material property is fundamentally different from describing the complete behavior of a material. A graph neural network capable of predicting formation energy or band gap is undoubtedly useful, but many problems in computational materials science require a much richer description of atomic interactions.

For example, consider the following research questions.

* How does a crystal deform under an external load?
* How do lithium ions diffuse through a solid electrolyte?
* How does a defect migrate through a crystal?
* How does a material melt when heated?
* How do atoms rearrange during phase transformation?

These problems cannot be solved by knowing only one scalar property.

Instead, we need a model capable of describing **how the energy changes as atoms move**.

This leads us to one of the central concepts of atomistic simulation:

> **the interatomic potential.**

Understanding interatomic potentials is essential for understanding why M3GNet was developed and why it represents such an important milestone in machine learning for materials science.

---

# 18.2.1 What Is an Interatomic Potential?

An interatomic potential is a mathematical function that describes the energy of a collection of atoms as a function of their positions.

Suppose a material contains

$N$

atoms located at positions

$$
\mathbf{R}_1,
\mathbf{R}_2,
\ldots,
\mathbf{R}_N.
$$

The total energy of the system can be written as

$$
E
=

E
(
\mathbf{R}_1,
\mathbf{R}_2,
\ldots,
\mathbf{R}_N
).
$$

This function is called the **interatomic potential** or the **potential energy function**.

Once this function is known,

almost every important physical quantity can be obtained.

For example,

atomic forces are

$$
\mathbf{F}_i
============

*

\frac{\partial E}
{\partial \mathbf{R}_i}.
$$

Similarly,

the stress tensor can be obtained by differentiating the energy with respect to lattice strain.

Therefore,

an accurate interatomic potential provides a complete description of atomic interactions.

---

# 18.2.2 Why Is the Potential Energy So Important?

The total energy determines the stability of every atomic configuration.

Imagine a ball rolling over a landscape.

```text id="m3gnet_potential1"
          Hill

         /  \

        /    \

 Valley      Valley
```

The valleys represent

low-energy configurations,

while the hills correspond to

unstable configurations.

Atoms behave similarly.

They naturally move toward configurations with lower potential energy.

Therefore,

understanding the energy landscape allows us to predict

* equilibrium structures,
* atomic motion,
* phase transitions,
* chemical reactions,
* diffusion pathways.

This multidimensional landscape is known as the **Potential Energy Surface (PES).**

---

# 18.2.3 The Potential Energy Surface

The Potential Energy Surface (PES) is one of the most important concepts in computational chemistry and materials science.

For a system containing

$N$

atoms,

the potential energy depends on

$3N$

independent spatial coordinates.

Mathematically,

$$
E
=

E
(
x_1,
y_1,
z_1,
\ldots,
x_N,
y_N,
z_N
).
$$

Thus,

the potential energy surface is not a simple two-dimensional curve.

Instead,

it is a hypersurface existing in a

$3N$

-dimensional space.

For even a relatively small crystal containing

100 atoms,

the energy depends on

300 coordinates.

Consequently,

the true potential energy surface is extraordinarily complex.

Learning this function accurately is one of the greatest challenges in atomistic machine learning.

---

# 18.2.4 Why Do We Need an Approximation?

Ideally,

every energy evaluation would be performed using quantum mechanics.

However,

this is computationally infeasible.

Suppose one DFT calculation requires

10 hours.

A molecular dynamics simulation may require

one million

energy evaluations.

The required computational time becomes

```text id="m3gnet_potential2"
1 Calculation

↓

10 Hours

↓

1,000,000 Calculations

↓

≈ 10 Million Hours
```

Clearly,

this is impossible for routine simulations.

Therefore,

researchers develop approximations to the true potential energy surface.

These approximations are called

interatomic potentials.

---

# 18.2.5 Classical Interatomic Potentials

Before machine learning became popular,

materials scientists relied on empirical force fields.

These force fields use manually designed mathematical equations containing adjustable parameters.

The parameters are fitted to experimental or first-principles data.

Several important classical potentials are widely used.

---

## Lennard–Jones Potential

One of the simplest interatomic potentials is the Lennard–Jones potential.

It is written as

$$
V(r)
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

where

* $r$ is the interatomic distance,
* $\varepsilon$ determines the well depth,
* $\sigma$ controls the equilibrium separation.

The first term represents strong short-range repulsion,

while the second describes long-range attraction.

Although historically important,

the Lennard–Jones potential is primarily suitable for noble gases and weakly interacting systems.

It cannot accurately describe the directional bonding found in most crystalline solids.

---

## Embedded Atom Method (EAM)

For metallic systems,

pairwise interactions alone are insufficient.

The Embedded Atom Method introduces many-body effects by considering the local electron density.

The total energy is written as

$$
E
=

\sum_i
F(\rho_i)
+
\frac12
\sum_{i\ne j}
\phi(r_{ij}),
$$

where

* $F(\rho_i)$ is the embedding energy,
* $\rho_i$ is the local electron density,
* $\phi(r_{ij})$ is the pair interaction.

EAM is highly successful for many metallic materials,

including

* aluminum,
* copper,
* nickel,
* iron.

However,

its applicability to covalent and ionic materials is limited.

---

## Tersoff Potential

Covalent materials such as

silicon

and

carbon

require bond-order potentials.

The Tersoff potential incorporates angular information through a bond-order term.

The energy takes the general form

$$
E
=

\frac12
\sum_{i\ne j}
f_C(r_{ij})
\left[
f_R(r_{ij})
+
b_{ij}
f_A(r_{ij})
\right].
$$

The bond-order coefficient

$b_{ij}$

depends on the local atomic environment,

allowing the strength of a bond to change according to neighboring atoms.

This was a major advance over purely pairwise models.

---

## Stillinger–Weber Potential

The Stillinger–Weber potential explicitly includes

three-body interactions.

Its total energy consists of

two-body

and

three-body

terms,

$$
E
=

\sum_{i<j}
V_2
+
\sum_{i<j<k}
V_3.
$$

The three-body contribution depends on

bond angles,

making the potential highly effective for tetrahedrally bonded materials such as silicon.

The inclusion of angular interactions is one of the key ideas that later inspired many modern graph neural networks, including M3GNet.

---

## ReaxFF

ReaxFF extends classical force fields by allowing

chemical bonds

to

form

and

break

during simulations.

Unlike conventional empirical potentials,

ReaxFF dynamically updates bond orders,

enabling simulations of

* combustion,
* chemical reactions,
* battery interfaces,
* catalytic processes.

Although remarkably versatile,

ReaxFF requires a large number of manually optimized parameters,

making its development both difficult and time-consuming.

---

# 18.2.6 Limitations of Classical Potentials

Despite decades of development,

classical force fields suffer from several fundamental limitations.

### Fixed Functional Form

The mathematical equations are specified before training.

If the true physics differs from the assumed equation,

the model cannot represent it accurately.

---

### Limited Transferability

A potential developed for silicon

may perform poorly for germanium.

Similarly,

a force field optimized for one crystal structure

often fails for another.

---

### Manual Parameterization

Developing a high-quality empirical potential requires

considerable human expertise.

Parameter fitting may take months or even years.

---

### Poor Generalization

Many classical potentials struggle to describe

* defects,
* surfaces,
* interfaces,
* disordered materials,
* high-pressure phases.

These limitations motivated the search for more flexible alternatives.

---

# 18.2.7 Machine Learning Interatomic Potentials

Machine learning interatomic potentials (MLIPs) replace manually designed equations with neural networks.

Instead of assuming the form of the potential,

the model learns it directly from data.

The workflow is

```text id="m3gnet_potential3"
DFT Calculations

↓

Atomic Structures

+

Energies

+

Forces

↓

Neural Network

↓

Learned Potential
```

The neural network approximates the potential energy surface,

allowing rapid predictions for previously unseen atomic configurations.

Because the model learns directly from data,

it is capable of representing far more complex interactions than traditional empirical force fields.

---

# 18.2.8 Advantages of Machine Learning Potentials

Compared with classical force fields,

MLIPs offer several important advantages.

* They automatically learn complex interaction functions.
* They do not require manually designed equations.
* They achieve near-DFT accuracy for many systems.
* They are computationally much faster than first-principles calculations.
* They can often generalize across chemically diverse materials when trained on sufficiently large datasets.

These advantages have made machine learning interatomic potentials one of the fastest-growing areas in computational materials science.

---

# 18.2.9 Position of M3GNet Among MLIPs

Numerous machine learning interatomic potentials have been proposed over the past decade,

including

* Behler–Parrinello Neural Networks,
* Deep Potential (DeepMD),
* SchNet,
* PhysNet,
* NequIP,
* Allegro,
* MACE,
* M3GNet.

Each architecture differs in

* atomic representation,
* geometric encoding,
* computational efficiency,
* scalability.

M3GNet distinguishes itself by combining

* graph neural networks,
* explicit three-body interactions,
* differentiable energy prediction,
* force prediction,
* stress prediction,

within a single unified architecture.

Moreover,

M3GNet was designed to serve as a **universal interatomic potential**, capable of modeling a broad range of crystalline materials rather than being restricted to a single chemical system.

---

# 18.2.10 From Property Prediction to Atomistic Simulation

The transition from earlier graph neural networks to M3GNet represents a conceptual shift.

Previous models answered questions such as

> *What is the band gap of this crystal?*

or

> *What is the formation energy of this material?*

M3GNet addresses a much broader question:

> **How will this material behave under arbitrary atomic motion?**

To answer this question,

the model must predict

* total energy,
* atomic forces,
* stress tensor,

while remaining fully differentiable.

This capability transforms the graph neural network from a property predictor into a complete atomistic simulation engine suitable for molecular dynamics, geometry optimization, and large-scale materials modeling.

---

## Looking Ahead

The discussion above naturally raises another important question:

**What makes M3GNet fundamentally different from previous graph neural networks?**

The answer lies in its treatment of **three-body interactions**. While earlier architectures primarily modeled pairwise relationships between neighboring atoms, M3GNet explicitly incorporates angular interactions that govern the stability and behavior of many crystalline materials.

In the next section, we will study **three-body interactions** in depth, understand their physical significance, examine why pairwise models are often insufficient, and see how M3GNet integrates angular information into its message-passing framework.

# 18.3 Three-Body Interactions

One of the defining innovations of M3GNet is its explicit incorporation of **three-body interactions** into the graph neural network. While previous models such as CGCNN and SchNet primarily learn from pairwise relationships between neighboring atoms, M3GNet recognizes that many physical properties of materials cannot be explained solely by considering atom pairs.

To understand why this is important, we must first examine how atoms interact in real materials.

Although the distance between two atoms certainly influences their interaction, **the presence and arrangement of surrounding atoms also changes the strength and nature of that interaction**. In other words, the energy associated with a bond depends not only on the bond length but also on the geometry of the local atomic environment.

This phenomenon is known as a **many-body interaction**.

Among many-body interactions, the simplest and most important is the **three-body interaction**, which depends on the relative arrangement of three atoms.

Understanding these interactions is essential for appreciating why M3GNet achieves higher accuracy than earlier graph neural networks on many atomistic simulation tasks.

---

# 18.3.1 Pairwise Interactions

Most classical interatomic potentials begin with the assumption that the total energy can be expressed as the sum of independent pair interactions.

For two atoms $i$ and $j$ separated by a distance $r_{ij}$,

the interaction energy is written as

$$
E_{ij}
======

V(r_{ij}),
$$

where

* $V$ is the pair potential,
* $r_{ij}$ is the distance between the atoms.

The total energy of the system becomes

$$
E
=

\frac12
\sum_{i\ne j}
V(r_{ij}).
$$

The factor of

$\frac12$

prevents each interaction from being counted twice.

This approach is attractive because it is mathematically simple and computationally efficient.

---

# 18.3.2 Physical Interpretation of Pair Potentials

Imagine two isolated atoms.

```text id="m3gnet_threebody1"
Atom A ------------- Atom B

Distance = r
```

Their interaction depends only on

$r$.

If the atoms move closer,

the energy changes.

If they move farther apart,

the energy changes again.

No other information is considered.

For noble gases,

this approximation often works well because their interactions are dominated by weak van der Waals forces.

However,

most engineering materials exhibit much more complicated bonding.

---

# 18.3.3 Why Pairwise Interactions Are Insufficient

Consider the following two atomic arrangements.

```text id="m3gnet_threebody2"
Configuration A

      C

      |

A ---------- B
```

```text id="m3gnet_threebody3"
Configuration B

A ---------- B

      |

      C
```

Suppose

the distance

between

A

and

B

is identical in both configurations.

A purely pairwise model predicts

the same interaction energy,

because

$r_{AB}$

is unchanged.

However,

real materials behave differently.

The location of atom

C

modifies

the electronic environment around the

A–B

bond.

Consequently,

the bond strength changes,

even though the bond length remains identical.

Pairwise models cannot describe this phenomenon.

---

# 18.3.4 Bond Angles

The missing quantity is

the bond angle.

Consider three atoms

$i$,

$j$,

and

$k$.

```text id="m3gnet_threebody4"
      j

     / \

    /θ  \

   i-----k
```

The angle

formed at atom

$j$

is

$$
\theta_{ijk}.
$$

Unlike pairwise interactions,

three-body interactions depend on

both

bond lengths

and

bond angles.

Therefore,

the local geometry becomes

an essential component

of the interaction energy.

---

# 18.3.5 Mathematical Description of Three-Body Interactions

A three-body contribution can be written as

$$
E_{ijk}
=======

V_3
(
r_{ij},
r_{jk},
\theta_{ijk}
).
$$

Notice that

the energy now depends on

three variables.

* First bond length

$r_{ij}$

* Second bond length

$r_{jk}$

* Bond angle

$\theta_{ijk}$

The total energy becomes

$$
E
=

\sum_{i<j}
V_2
+
\sum_{i<j<k}
V_3.
$$

This expression is considerably more expressive than a purely pairwise model.

---

# 18.3.6 Example: Silicon

Silicon provides one of the best examples of the importance of three-body interactions.

Each silicon atom forms

four covalent bonds.

The preferred bond angle is approximately

$$
109.47^\circ.
$$

```text id="m3gnet_threebody5"
          Si

        / | \

      Si Si Si

         |

        Si
```

If one bond angle changes,

the energy increases,

even if

all bond lengths remain unchanged.

Therefore,

the stability of silicon

cannot be described solely by pairwise distances.

Angular interactions are essential.

---

# 18.3.7 Example: Diamond

Diamond possesses the same tetrahedral geometry.

Each carbon atom prefers

approximately

$$
109.47^\circ.
$$

Changing

only

the bond angle

distorts the electronic orbitals,

raising the total energy.

Consequently,

an accurate potential

must recognize

angular deviations.

---

# 18.3.8 Example: Water

Although M3GNet focuses primarily on materials,

water provides an intuitive example.

The equilibrium H–O–H angle is approximately

$$
104.5^\circ.
$$

```text id="m3gnet_threebody6"
H

 \

  O

 /

H
```

If this angle changes,

the molecular energy changes,

despite

the O–H bond lengths remaining nearly constant.

Again,

pairwise interactions alone

cannot explain the observed behavior.

---

# 18.3.9 Three-Body Interactions in Crystalline Materials

Many technologically important materials

derive their properties

from directional bonding.

Examples include

* silicon,
* germanium,
* diamond,
* gallium nitride,
* silicon carbide,
* alumina,
* zirconia,
* many transition-metal oxides.

In these systems,

bond angles strongly influence

* structural stability,
* elastic constants,
* phonon spectra,
* defect formation,
* diffusion pathways.

Ignoring angular information

reduces predictive accuracy.

---

# 18.3.10 Why Earlier GNNs Struggle

Earlier graph neural networks

primarily perform

pairwise message passing.

```text id="m3gnet_threebody7"
Atom

↓

Neighbor

↓

Neighbor

↓

Update
```

Although repeated message passing

may eventually encode

some angular information,

the process is indirect.

The network must discover

bond-angle relationships

implicitly.

This requires

* more parameters,
* deeper networks,
* more training data.

Consequently,

learning becomes less efficient.

---

# 18.3.11 Explicit Angular Modeling

M3GNet adopts a different philosophy.

Instead of hoping

the network

discovers

angular relationships,

the architecture

provides them explicitly.

Every neighboring atom triplet

contributes

to

message passing.

```text id="m3gnet_threebody8"
Atom i

   \

    \

     Atom j

    /

   /

Atom k

↓

Angle

↓

Message
```

The network therefore receives

distance

and

angle

information simultaneously.

---

# 18.3.12 Three-Body Features

For each triplet,

M3GNet constructs

a feature vector containing

* bond length

$r_{ij}$

* bond length

$r_{jk}$

* bond angle

$\theta_{ijk}$

These quantities are transformed

into high-dimensional embeddings,

which are then processed

during message passing.

The resulting representations contain

substantially richer geometric information

than pairwise graphs.

---

# 18.3.13 Why Three-Body Interactions Improve Learning

Including angular information provides several important advantages.

### Better Physical Representation

The model more closely reflects

real atomic interactions.

---

### Improved Accuracy

Materials with directional bonding

are modeled

more accurately.

---

### Better Generalization

The network requires

less effort

to discover

angular relationships,

leading to improved transferability.

---

### Faster Convergence

Because physically meaningful information

is explicitly available,

optimization often converges

more rapidly.

---

# 18.3.14 Computational Cost

Three-body interactions

are considerably more expensive

than pairwise interactions.

Suppose

each atom has

$M$

neighbors.

Pairwise interactions scale approximately as

$$
O(M).
$$

Three-body interactions

consider

neighbor pairs,

leading to

approximately

$$
O(M^2).
$$

Consequently,

M3GNet requires

more computation

than

pairwise graph neural networks.

However,

the increase in computational cost

is generally outweighed

by the substantial improvement

in predictive accuracy.

---

# 18.3.15 Relationship to Earlier Models

It is useful to compare how different architectures handle geometry.

| Model      | Pairwise Distances | Bond Angles           | Three-Body Information |
| ---------- | ------------------ | --------------------- | ---------------------- |
| CGCNN      | ✓                  | ✗                     | ✗                      |
| SchNet     | ✓                  | Implicit              | Partial                |
| MEGNet     | ✓                  | Limited               | Partial                |
| ALIGNN     | ✓                  | Explicit (Line Graph) | ✓                      |
| **M3GNet** | ✓                  | Explicit              | ✓                      |

Both ALIGNN and M3GNet recognize the importance of angular information.

However,

their implementations differ.

* ALIGNN introduces a **line graph**, where bonds become graph nodes and neighboring bonds exchange messages.
* M3GNet directly constructs **three-body geometric features** from neighboring atom triplets and incorporates them into the message-passing process.

Although the implementation strategies differ, both architectures are motivated by the same physical principle: **the energy of a material depends on more than just pairwise distances**.

---

# 18.3.16 Summary

Three-body interactions represent one of the most important advances in modern atomistic machine learning.

Unlike pairwise models, which consider only distances between two atoms, three-body models account for the angular relationships among neighboring atoms. These angular interactions play a crucial role in determining the structure, stability, and properties of many crystalline materials.

By explicitly incorporating three-body information into its graph neural network architecture, M3GNet is able to learn a much richer representation of local atomic environments. This capability forms the foundation for its success as a universal machine learning interatomic potential capable of accurately predicting energies, forces, stresses, and atomistic dynamics.

---

## Looking Ahead

Understanding the importance of three-body interactions naturally leads to the next question:

**How does M3GNet represent a crystal so that atoms, bonds, and angular relationships can all be processed by the neural network?**

In the next section, **18.4 Graph Representation in M3GNet**, we will study the complete graph construction process, including node features, edge features, neighbor lists, cutoff radii, periodic boundary conditions, and the representation of atomic triplets used for three-body message passing.
