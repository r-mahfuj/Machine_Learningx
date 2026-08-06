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

## 18.3.17 Three-Body Interactions from a Quantum Mechanical Perspective

In the previous sections, we learned that pairwise interactions are often insufficient to describe real materials because the interaction between two atoms is influenced by their surrounding neighbors. We also introduced the concept of three-body interactions and showed that bond angles play a central role in determining the stability of many crystalline materials.

However, an important question remains unanswered.

**Why do three-body interactions exist in the first place?**

Are they merely a mathematical trick introduced to improve the accuracy of force fields, or do they have a genuine physical origin?

The answer lies in quantum mechanics.

Three-body interactions are not artificial constructs. They arise naturally from the collective behavior of electrons in many-atom systems. The electronic structure around an atom is influenced simultaneously by all of its neighboring atoms. Consequently, the energy associated with one bond cannot always be determined independently of the surrounding bonds.

Understanding this physical origin helps explain why M3GNet places such emphasis on angular information.

---

## 18.3.18 Electronic Origin of Three-Body Effects

Consider two isolated atoms approaching each other.

```text
Atom A ---------------- Atom B
```

As the atoms approach,

their electron clouds begin to overlap.

The overlap determines

* bond strength,
* equilibrium bond length,
* bond energy.

In this simple case,

the interaction depends almost entirely on the distance between the two atoms.

Now introduce a third atom.

```text
          C

         /

A ------ B
```

The electronic density surrounding atom **B** is no longer determined solely by atom **A**.

Instead,

the electrons simultaneously interact with

* atom A,
* atom B,
* atom C.

The electron density around atom B becomes distorted by the presence of atom C.

Consequently,

the interaction between atoms A and B changes,

even if the distance

$r_{AB}$

remains exactly the same.

This phenomenon is fundamentally quantum mechanical.

The energy depends on the collective electronic environment rather than independent atom pairs.

---

## 18.3.19 Directional Covalent Bonding

Three-body interactions become especially important in materials containing covalent bonds.

Unlike metallic bonding,

covalent bonding is highly directional.

The direction of neighboring atoms determines

* orbital overlap,
* electron localization,
* bond strength,
* crystal stability.

Consider the electronic configuration of silicon.

Each silicon atom possesses four valence electrons.

To minimize the total energy,

the atom forms four

$sp^3$

hybrid orbitals.

These orbitals point toward the corners of a tetrahedron.

```text
             Si

          /  |  \

       Si   Si   Si

            |

           Si
```

The preferred bond angle is

$$
109.47^\circ.
$$

This angle is not arbitrary.

It is the angle that maximizes orbital overlap while minimizing electron-electron repulsion.

Suppose one neighboring atom moves while all bond lengths remain constant.

```text
Before

      Si

     /

Si---Si


After

Si

 \

  Si---Si
```

Although every bond length remains unchanged,

the orbital overlap changes.

The electronic energy therefore changes.

Pairwise potentials cannot represent this effect because they only measure distances.

---

## 18.3.20 Example: Diamond

Diamond exhibits the same physical behavior.

Every carbon atom forms four

$sp^3$

covalent bonds.

The equilibrium geometry is again tetrahedral.

The remarkable hardness of diamond originates from

* strong covalent bonding,
* highly ordered bond angles,
* three-dimensional bonding network.

Suppose a carbon atom is displaced slightly.

The bond lengths may change only minimally,

yet the surrounding bond angles become distorted.

This distortion weakens orbital overlap,

increasing the total energy.

Therefore,

an accurate model must recognize angular deformation.

---

## 18.3.21 Ionic Crystals

One might assume that three-body interactions are important only for covalent materials.

In reality,

they also influence many ionic crystals.

Consider a ceramic such as alumina

$\mathrm{Al_2O_3}$.

Although the dominant interaction is electrostatic,

the local coordination geometry strongly influences

* crystal stability,
* elastic response,
* defect formation,
* diffusion pathways.

Neighboring oxygen atoms modify the electrostatic environment around aluminum atoms.

Thus,

local geometry still affects the total energy.

Three-body information therefore improves prediction accuracy even in predominantly ionic materials.

---

## 18.3.22 Metallic Systems

Metals behave differently.

Metallic bonding involves delocalized electrons rather than localized covalent bonds.

Consequently,

directionality is generally weaker.

This is why simple pair-based models such as the Embedded Atom Method (EAM) perform reasonably well for many metals.

However,

modern metallic alloys often possess

* complex crystal structures,
* chemical disorder,
* local lattice distortions.

In these systems,

neighbor geometry again becomes important.

Machine learning potentials therefore benefit from incorporating many-body information even for metallic materials.

---

## 18.3.23 Why Bond Angles Matter More Than Bond Lengths

An interesting observation from quantum chemistry is that

small angular distortions often produce surprisingly large energy changes.

Consider two hypothetical structures.

### Structure A

```text
      C

     /

A---B
```

### Structure B

```text
C

 \

  B---A
```

Suppose

the bond lengths satisfy

$$
r_{AB}^{(A)}
============

r_{AB}^{(B)},
$$

and

$$
r_{BC}^{(A)}
============

r_{BC}^{(B)}.
$$

A pairwise model predicts identical energies because every interatomic distance is unchanged.

However,

the bond angle

$$
\theta_{ABC}
$$

is different.

Quantum mechanically,

different bond angles correspond to different orbital overlap,

leading to different electronic energies.

This simple example illustrates why bond angles contain information that cannot be recovered from distances alone.

---

## 18.3.24 Three-Body Interactions and Chemical Environment

Another way to understand three-body interactions is through the concept of the **local chemical environment**.

Consider the following two situations.

### Environment 1

```text
      O

      |

Si ---- O
```

### Environment 2

```text
      N

      |

Si ---- O
```

Even if every bond length is identical,

the surrounding atoms differ.

The neighboring chemistry changes

* charge distribution,
* orbital hybridization,
* electron density.

The interaction between the silicon and oxygen atoms therefore changes.

Modern graph neural networks attempt to capture precisely these environment-dependent interactions.

Rather than assigning a fixed bond strength,

the network learns how bond strength varies according to the surrounding atomic arrangement.

---

## 18.3.25 Many-Body Expansion

Three-body interactions belong to a broader mathematical framework known as the **many-body expansion**.

Instead of expressing the total energy purely as a sum of pair interactions,

the energy is decomposed into contributions involving increasing numbers of atoms.

The general form is

$$
E
=

E^{(1)}
+
E^{(2)}
+
E^{(3)}
+
E^{(4)}
+\cdots,
$$

where

* $E^{(1)}$ represents one-body contributions,
* $E^{(2)}$ represents pair interactions,
* $E^{(3)}$ represents three-body interactions,
* $E^{(4)}$ represents four-body interactions,
* higher-order terms describe increasingly complex collective effects.

In practice,

higher-order terms become progressively more expensive to compute.

Consequently,

many atomistic models seek a balance between physical accuracy and computational efficiency.

M3GNet achieves this balance by explicitly incorporating **three-body interactions**, which capture a substantial portion of the missing physics beyond pairwise models while remaining computationally tractable.

---

### Transition to the Next Section

We now understand **why** three-body interactions are physically important and how they arise from the collective behavior of electrons in real materials.

The next question is a computational one:

> **How can a graph neural network represent atoms, bonds, and three-body geometric relationships in a form that a neural network can process?**

To answer this, we next examine **18.4 Graph Representation in M3GNet**, where we will construct the complete crystal graph, define node and edge features, generate atomic triplets, and prepare the geometric information used by the M3GNet architecture.


# 18.4 Graph Representation in M3GNet

In the previous section, we established the physical importance of three-body interactions and explained why bond angles must be considered to accurately model many crystalline materials. However, understanding the underlying physics is only the first step. A neural network cannot directly process atomic coordinates, chemical symbols, or crystal structures. Before learning can begin, the structural information must be transformed into a mathematical representation that preserves the essential physics while remaining suitable for graph-based computation.

This transformation is known as **graph construction**.

Graph construction is one of the most critical stages in any graph neural network. Regardless of how powerful the neural network architecture is, it cannot recover information that is lost during graph construction. If important geometric relationships are omitted or represented poorly, the model's predictive performance will inevitably suffer.

For this reason, the graph representation used in M3GNet is considerably richer than those employed in earlier graph neural networks.

Unlike conventional graph neural networks that primarily represent atoms and pairwise bonds, M3GNet constructs a graph capable of describing

* atomic identities,
* pairwise distances,
* local chemical environments,
* three-body geometric relationships,
* periodic crystal structures.

The resulting graph provides the neural network with a physically meaningful description of the material while remaining computationally efficient.

---

# 18.4.1 Why Represent a Crystal as a Graph?

A crystal is naturally described as a collection of atoms connected through chemical interactions.

From a mathematical perspective, this description closely resembles a graph.

Recall that a graph is defined as

$$
G = (V, E),
$$

where

* $V$ denotes the set of vertices (nodes),
* $E$ denotes the set of edges connecting those vertices.

In M3GNet,

the correspondence is straightforward.

| Graph Component | Physical Meaning               |
| --------------- | ------------------------------ |
| Node            | Atom                           |
| Edge            | Neighboring atomic interaction |
| Graph           | Entire crystal                 |

Thus, a crystal structure becomes a graph whose topology reflects the local atomic environment.

For example, consider a simple crystal containing four atoms.

```text id="m3gnet_graph1"
A -------- B
|          |
|          |
D -------- C
```

Each atom is represented as a graph node, while neighboring atoms are connected by edges.

Unlike an image or a regular grid, this graph representation naturally accommodates

* arbitrary crystal structures,
* different unit-cell sizes,
* varying coordination numbers,
* complex crystal symmetries.

This flexibility is one of the major reasons graph neural networks have become so successful in materials informatics.

---

# 18.4.2 Nodes: Representing Atoms

The most fundamental element of the graph is the **node**.

Every atom in the crystal corresponds to one node.

Suppose a crystal contains

$N$

atoms.

The graph therefore contains

$N$

nodes.

The node corresponding to atom

$i$

is associated with a feature vector

$$
\mathbf{v}_i.
$$

This vector does not initially contain learned information.

Instead, it encodes basic atomic properties.

Typical node features include

* atomic number,
* group in the periodic table,
* period,
* electronegativity,
* valence electron count,
* atomic radius,
* oxidation state (when available),
* embedding vector.

In modern graph neural networks, however, manually selecting dozens of atomic descriptors is often unnecessary.

Instead, the atomic number is first converted into a learnable embedding.

Suppose atom

$i$

has atomic number

$Z_i$.

An embedding layer maps

$$
Z_i
\longrightarrow
\mathbf{v}_i,
$$

where

$$
\mathbf{v}_i
\in
\mathbb{R}^d.
$$

Here,

$d$

is the embedding dimension.

Rather than treating atomic number as an ordinary integer,

the embedding layer learns a continuous vector representation during training.

As a result,

chemically similar elements gradually acquire similar embeddings.

For example,

silicon and germanium may occupy nearby locations in the learned embedding space because they exhibit similar chemical behavior.

---

# 18.4.3 Why Use Learnable Atomic Embeddings?

One might ask why we do not simply use the atomic number directly.

Suppose we represented three elements as

| Element   | Atomic Number |
| --------- | ------------- |
| Carbon    | 6             |
| Silicon   | 14            |
| Germanium | 32            |

A neural network receiving these integers would incorrectly assume that germanium is more than five times "larger" than carbon because

$$
32 > 6.
$$

Chemically, this interpretation is meaningless.

Instead, M3GNet learns an embedding vector for each element.

For example,

```text id="m3gnet_graph2"
Carbon

↓

Embedding Layer

↓

[0.21, -0.83, 0.56, ...]

Silicon

↓

Embedding Layer

↓

[0.24, -0.79, 0.60, ...]

Germanium

↓

Embedding Layer

↓

[0.30, -0.74, 0.58, ...]
```

Notice that silicon and germanium naturally acquire similar vector representations because their chemical environments and bonding characteristics are similar.

The network discovers these relationships automatically during training.

This approach is significantly more flexible than manually designing atomic descriptors.

---

# 18.4.4 Edges: Representing Atomic Interactions

Nodes alone are insufficient.

The graph must also describe how atoms interact with one another.

These interactions are represented by **edges**.

An edge is created whenever two atoms are considered neighbors.

Suppose atoms

$i$

and

$j$

are neighbors.

The graph contains an edge

$$
(i,j).
$$

Unlike social networks, where an edge merely indicates a friendship,

an edge in M3GNet represents a physical interaction.

Every edge stores geometric information about the two connected atoms.

The most important quantity is the interatomic distance

$$
r_{ij}.
$$

This distance is computed as

$$
r_{ij}
======

\left|
\mathbf{R}_i
------------

\mathbf{R}_j
\right|,
$$

where

* $\mathbf{R}_i$ is the position of atom $i$,
* $\mathbf{R}_j$ is the position of atom $j$.

The distance serves as one of the primary inputs to the message-passing process.

However, unlike classical force fields that use the raw distance directly, M3GNet later transforms this distance into a richer continuous representation using basis functions. We will study this encoding in detail in **Section 18.5**.

---

# 18.4.5 Directed and Undirected Edges

At first glance, one might expect each bond to correspond to a single edge.

However, message-passing neural networks often use **directed edges**.

Consider two neighboring atoms.

```text id="m3gnet_graph3"
Atom A -------- Atom B
```

Instead of storing one edge,

the graph stores two directed edges.

```text id="m3gnet_graph4"
A  ----->  B

A  <-----  B
```

Why?

Because the information flowing

from

A

to

B

may differ from the information flowing

from

B

to

A.

During message passing,

each directed edge carries its own messages,

allowing greater flexibility and expressive power.

This design also simplifies the implementation of graph convolution algorithms.

---

# 18.4.6 Edge Features

Just as each node possesses a feature vector,

each edge also possesses a feature vector.

For the edge connecting atoms

$i$

and

$j$,

the edge feature is written as

$$
\mathbf{e}_{ij}.
$$

At the graph construction stage,

this feature may include

* interatomic distance,
* relative displacement vector,
* periodic image information,
* basis-function encoding (later),
* learned edge embedding.

As training proceeds,

these edge features become increasingly rich,

capturing not only geometric information but also the evolving chemical environment around each bond.

---

At this point, we have established the two fundamental building blocks of the M3GNet graph:

1. **Nodes**, representing atoms through learnable embeddings.
2. **Edges**, representing neighboring atomic interactions through geometric features.

However, M3GNet differs from earlier graph neural networks because these components alone are **not enough**. The model must also represent **triplets of atoms** in order to capture bond-angle information and perform three-body message passing.

In the next part of this section, we will study **neighbor lists, cutoff radii, periodic boundary conditions, and the construction of atomic triplets**, which together enable M3GNet to incorporate three-body interactions into its graph representation.


Continuing **Section 18.4**.

---

## 18.4.7 Neighbor Lists

A real crystal may contain thousands or even millions of atoms. If every atom were connected to every other atom, the graph would contain an enormous number of edges.

For a crystal containing

$N$

atoms,

the total number of possible atom pairs is

$$
\frac{N(N-1)}{2}.
$$

This quantity grows quadratically with the number of atoms.

For example,

| Number of Atoms | Possible Atom Pairs |
| --------------: | ------------------: |
|             100 |               4,950 |
|           1,000 |             499,500 |
|          10,000 |          49,995,000 |

Clearly, constructing edges between every pair of atoms would make graph neural networks computationally infeasible.

Fortunately, this is also **physically unnecessary**.

In most materials, atoms interact most strongly with nearby neighbors. As the distance between two atoms increases, their interaction strength rapidly decreases.

Therefore, M3GNet does not connect every atom to every other atom. Instead, it constructs a **neighbor list**.

A neighbor list is simply the collection of atoms located sufficiently close to a given atom.

Suppose atom $i$ has neighboring atoms

$$
j,;k,;l,;m.
$$

Its neighbor list can be written as

$$
\mathcal{N}(i)
==============

{j,k,l,m}.
$$

Only atoms belonging to this neighbor list exchange messages with atom $i$.

---

### Example

Consider the following simplified crystal.

```text
          A

      B       C

          D

      E       F
```

Suppose atom **D** is chosen as the central atom.

Its neighbor list might be

```text
Neighbor(D)

↓

A

B

C

E

F
```

The graph therefore contains edges

```text
D ↔ A

D ↔ B

D ↔ C

D ↔ E

D ↔ F
```

Notice that atoms A and F are **not** directly connected unless they are also neighbors of each other.

This dramatically reduces the number of edges in the graph.

---

## 18.4.8 Cutoff Radius

The next question is

**How do we determine whether two atoms are neighbors?**

The simplest approach is to introduce a **cutoff radius**.

Suppose the cutoff distance is

$$
r_c.
$$

Two atoms are considered neighbors whenever

$$
r_{ij}
<
r_c.
$$

Otherwise,

no edge is created.

Mathematically,

$$
(i,j)\in E
\quad
\text{if}
\quad
r_{ij}<r_c.
$$

This simple rule defines the topology of the crystal graph.

---

### Physical Interpretation

Imagine drawing a sphere around every atom.

```text
             ●

        ***********

      **           **

     *      Atom     *

      **           **

        ***********
```

The sphere has radius

$r_c$.

Every atom lying inside the sphere becomes a neighbor.

Atoms outside the sphere are ignored.

The cutoff radius therefore determines the local chemical environment that each atom can "see."

---

## 18.4.9 Choosing the Cutoff Radius

Selecting an appropriate cutoff radius is extremely important.

If the cutoff is **too small**, important neighbors will be excluded.

```text
Too Small Cutoff

Atom

↓

Nearest Neighbor

↓

Second Neighbor (Ignored)
```

The graph loses valuable structural information.

Conversely,

if the cutoff is **too large**,

many weakly interacting atoms become connected.

```text
Large Cutoff

↓

Many Extra Neighbors

↓

Large Graph

↓

Higher Computational Cost
```

Although additional neighbors may contain some useful information,

they also increase

* memory usage,
* training time,
* computational complexity.

Consequently,

the cutoff radius must balance

* physical accuracy,
* computational efficiency.

Typical atomistic graph neural networks use cutoff radii between approximately **4 Å and 6 Å**, although the optimal value depends on the material system and the interaction range being modeled.

---

## 18.4.10 Why M3GNet Uses Local Environments

One might wonder why long-range interactions are ignored.

After all,

electrostatic interactions can extend over very large distances.

The key observation is that **most of the chemical information required for predicting local energies resides within the immediate neighborhood of each atom**.

The nearest atoms determine

* bond formation,
* coordination number,
* local symmetry,
* orbital overlap,
* bond angles.

These quantities dominate the local atomic energy.

Long-range interactions may still influence certain materials, but much of their effect can often be learned indirectly through multiple rounds of message passing.

Furthermore,

including every distant atom would make training prohibitively expensive.

Thus,

local environments provide an excellent compromise between

accuracy

and

efficiency.

---

## 18.4.11 Periodic Boundary Conditions

Unlike molecules,

crystals extend infinitely in space.

Of course,

a computer cannot simulate an infinite number of atoms.

Instead,

only a single **unit cell** is stored.

```text
□ □ □ □

□ □ □ □

□ □ □ □
```

This unit cell is repeated infinitely in every direction.

This assumption is known as a **Periodic Boundary Condition (PBC).**

Periodic boundary conditions are one of the foundations of computational materials science.

Without them,

the simulated crystal would possess artificial surfaces,

leading to incorrect physical behavior.

---

### Example

Suppose the unit cell contains only four atoms.

```text
+-----------+

A         B

C         D

+-----------+
```

Atom

A

appears to have no neighbor on the left.

However,

because the crystal repeats,

the atom immediately outside the left boundary is actually another periodic image of atom

B.

Likewise,

atoms located above,

below,

or beyond the unit cell boundaries also possess periodic images.

Therefore,

neighbor searching must include atoms from neighboring periodic cells.

---

## 18.4.12 Neighbor Search Under Periodic Boundary Conditions

When constructing the graph,

M3GNet does **not** search only within the original unit cell.

Instead,

it examines neighboring periodic images.

Conceptually,

the search region resembles

```text
□ □ □

□ ■ □

□ □ □
```

The central square represents the original unit cell.

The surrounding squares are periodic replicas.

During neighbor searching,

atoms from these neighboring images are considered whenever they fall within the cutoff radius.

As a result,

an atom located near the boundary correctly identifies neighbors across the periodic boundary.

This ensures that the graph accurately represents the infinite crystal rather than an isolated finite cluster.

---

## 18.4.13 The Minimum Image Convention

Periodic crystals introduce an additional complication.

Two atoms may possess multiple periodic images.

Which image should be used?

To answer this,

atomistic simulations employ the **minimum image convention**.

For every atom pair,

the algorithm selects the periodic image corresponding to the **shortest interatomic distance**.

Mathematically,

the distance becomes

$$
r_{ij}
======

\min_{\mathbf{T}}
\left|
\mathbf{R}_i
------------

(\mathbf{R}_j+\mathbf{T})
\right|,
$$

where

$\mathbf{T}$

is a lattice translation vector.

In other words,

rather than measuring the distance to the atom inside the original unit cell,

the algorithm measures the distance to whichever periodic image is closest.

This convention guarantees that every edge represents the physically nearest interaction.

---

## 18.4.14 Graph Construction Workflow

At this stage, we can summarize the graph construction process before introducing three-body triplets.

The procedure is

```text
Crystal Structure

↓

Atomic Coordinates

↓

Periodic Boundary Conditions

↓

Neighbor Search

↓

Cutoff Radius

↓

Node Construction

↓

Edge Construction

↓

Crystal Graph
```

This graph already contains

* atomic identities,
* neighbor relationships,
* interatomic distances.

However,

it is still insufficient for M3GNet.

The model also requires **bond-angle information**, which cannot be extracted from isolated edges alone.

To capture angular geometry, M3GNet constructs **atomic triplets**, transforming the graph from a simple collection of nodes and edges into a richer representation capable of modeling three-body interactions.

---

### Transition to the Next Part

In the next part of Section **18.4**, we will study **triplet generation**, **bond graph construction**, and how M3GNet transforms neighboring atom pairs into explicit three-body geometric features that form the foundation of its message-passing architecture.

Continuing **Section 18.4**.

---

# 18.4.15 From Pairwise Graphs to Three-Body Graphs

Up to this point, our crystal graph consists of only **nodes** (atoms) and **edges** (pairwise interactions). Such a graph is sufficient for architectures like CGCNN, where messages are exchanged only between neighboring atoms.

However, M3GNet goes one step further.

Recall from Section **18.3** that the energy of many materials depends not only on distances but also on **bond angles**. A conventional graph containing only nodes and edges cannot explicitly represent these angles.

To see why, consider the following three atoms.

```text
        k

       /

      /

i ---- j
```

The graph stores two edges

* $i \rightarrow j$
* $j \rightarrow k$

but nowhere does it explicitly store the angle

$$
\theta_{ijk}.
$$

A neural network receiving only pairwise edges would have to infer this angle indirectly after several rounds of message passing.

This is possible in principle, but inefficient.

M3GNet therefore **constructs the angular relationships explicitly** before message passing begins.

---

# 18.4.16 Atomic Triplets

The basic geometric object in M3GNet is no longer a pair of atoms.

Instead, it is an **ordered triplet** of atoms.

Suppose atom

$j$

is the central atom.

Two neighboring atoms,

$i$

and

$k$,

form the triplet

$$
(i,j,k).
$$

Graphically,

```text
        k

       /

      /

i ---- j
```

The central atom is always the atom where the angle is measured.

The corresponding bond angle is

$$
\theta_{ijk}.
$$

Notice that

the order of the atoms matters.

The triplet

$$
(i,j,k)
$$

is generally different from

$$
(k,j,i),
$$

because they correspond to different directed edge combinations during message passing.

---

# 18.4.17 Constructing Triplets

How are these triplets generated?

The procedure is remarkably simple.

Suppose the neighbor list of atom

$j$

is

$$
\mathcal{N}(j)
==============

{A,B,C,D}.
$$

Every **pair of neighbors** is combined with the central atom.

The resulting triplets are

```text
(A,j,B)

(A,j,C)

(A,j,D)

(B,j,C)

(B,j,D)

(C,j,D)
```

Each of these represents a unique angular interaction.

If an atom has

$M$

neighbors,

the number of possible triplets is

$$
\frac{M(M-1)}{2}.
$$

For example,

if an atom has

6

neighbors,

the number of triplets becomes

$$
\frac{6\times5}{2}
==================

15.

$$

Therefore,

a single atom may contribute many angular interactions to the graph.

---

# 18.4.18 Why the Central Atom Matters

Consider three atoms

```text
A ----- B ----- C
```

The angle

$$
\theta_{ABC}
$$

is measured at atom

$B$.

Now consider

```text
B ----- A ----- C
```

The angle

$$
\theta_{BAC}
$$

is measured at atom

$A$.

Although the same three atoms are involved,

the two angles are completely different.

Therefore,

triplets are **center-dependent**.

Every atom in the crystal can become the central atom of many different triplets.

---

# 18.4.19 Computing Bond Angles

Once a triplet has been identified,

its bond angle is calculated from the vectors joining the central atom to its neighbors.

Suppose

$$
\mathbf{r}_{ji}
===============

\mathbf{R}_i-\mathbf{R}_j
$$

and

$$
\mathbf{r}_{jk}
===============

\mathbf{R}_k-\mathbf{R}_j.
$$

The bond angle follows directly from the dot product.

$$
\cos\theta_{ijk}
================

\frac{
\mathbf{r}*{ji}
\cdot
\mathbf{r}*{jk}
}
{
\left|
\mathbf{r}*{ji}
\right|
,
\left|
\mathbf{r}*{jk}
\right|
}.
$$

Finally,

the angle itself is

$$
\theta_{ijk}
============

\cos^{-1}
\left(
\frac{
\mathbf{r}*{ji}
\cdot
\mathbf{r}*{jk}
}
{
\left|
\mathbf{r}*{ji}
\right|
,
\left|
\mathbf{r}*{jk}
\right|
}
\right).
$$

This equation is one of the most important geometric calculations performed during graph construction.

Every angular feature used by M3GNet originates from this computation.

---

# 18.4.20 Geometric Meaning of the Dot Product

The previous equation may appear to be just another mathematical formula, but it has a clear geometric interpretation.

Recall the dot product identity

$$
\mathbf{a}\cdot\mathbf{b}
=========================

|\mathbf{a}|
,
|\mathbf{b}|
\cos\theta.
$$

The dot product measures how closely two vectors point in the same direction.

Several important cases illustrate this relationship.

### Parallel Bonds

```text
A -------- B -------- C
```

The vectors point in nearly the same direction.

Therefore,

$$
\theta
\approx
0^\circ,
$$

and

$$
\cos\theta
\approx
1.
$$

---

### Perpendicular Bonds

```text
      C

      |

      |

A ---- B
```

The vectors are perpendicular.

Hence,

$$
\theta
======

90^\circ,
$$

and

$$
\cos\theta
==========

0.

$$

---

### Opposite Directions

```text
A -------- B -------- C
```

Now the vectors point in opposite directions.

Thus,

$$
\theta
======

180^\circ,
$$

and

$$
\cos\theta
==========

-1.
$$

These simple examples illustrate how the dot product converts geometric orientation into numerical information that a neural network can process.

---

# 18.4.21 Angular Features

Once the angle has been computed,

it becomes part of the feature representation for the triplet.

Conceptually,

every triplet contributes information such as

```text
Triplet

↓

Distance ij

↓

Distance jk

↓

Bond Angle

↓

Three-Body Feature Vector
```

Notice that the triplet is no longer described by a single scalar quantity.

Instead,

it contains multiple pieces of geometric information describing the local atomic environment.

These features will later be transformed into high-dimensional embeddings before entering the neural network.

---

# 18.4.22 Why Explicit Triplets Improve Learning

One might ask why M3GNet explicitly constructs triplets instead of allowing the neural network to infer angles from coordinates.

There are several reasons.

### Reduced Learning Burden

Without explicit angles,

the network must first discover the geometric relationship between neighboring edges.

Providing triplets directly removes this burden.

---

### Better Physical Inductive Bias

A machine-learning model performs best when its architecture reflects the underlying physics.

Since bond angles influence the energy,

they should appear explicitly in the graph representation.

---

### Faster Optimization

Because the relevant geometric information is already available,

the neural network spends less time learning basic geometry and more time learning the relationship between geometry and energy.

This often leads to faster convergence during training.

---

### Improved Generalization

Models built upon physically meaningful representations generally require less training data and extrapolate more reliably to unseen materials.

This principle is one of the major reasons modern atomistic graph neural networks increasingly incorporate explicit geometric information.

---

# 18.4.23 Graph Representation Before Neural Processing

At the end of graph construction, the crystal has been transformed into three interconnected levels of information.

```text
Crystal Structure

│

├── Nodes
│      ↓
│   Atomic Embeddings
│
├── Edges
│      ↓
│   Pairwise Distances
│
└── Triplets
       ↓
   Bond Angles
```

Together,

these components form the complete geometric representation used by M3GNet.

Unlike earlier graph neural networks, which operate primarily on nodes and edges, M3GNet begins with a representation that already contains the information necessary to model both pairwise and three-body interactions.

This richer representation is one of the key reasons why M3GNet achieves excellent accuracy across a wide variety of crystalline materials.

---

### Transition to Section 18.5

Although the graph now contains atoms, distances, and bond angles, these quantities are still expressed in their raw numerical form.

Neural networks generally do not learn effectively from raw distances or angles alone. Instead, these geometric quantities are transformed into smooth, continuous, high-dimensional representations using carefully designed basis functions.

In the next section, **18.5 Distance and Angular Encoding**, we will study how M3GNet converts interatomic distances and bond angles into learnable embeddings using radial basis functions, angular basis functions, and continuous geometric encodings before message passing begins.

# 18.5 Distance and Angular Encoding

In the previous section, we learned how M3GNet converts a crystal structure into a graph consisting of atoms, edges, and atomic triplets. At this stage, every edge stores an interatomic distance, and every triplet stores a bond angle. Although these quantities accurately describe the geometry of the crystal, they are **not yet suitable inputs for a deep neural network**.

One might naturally ask:

> **Why can't we simply feed the raw distances and angles directly into the neural network?**

For example, suppose two bonds have lengths

$$
2.31\ \text{Å}
$$

and

$$
2.32\ \text{Å}.
$$

From a physical perspective, these two bonds are almost identical. Their interaction energies should also be very similar. However, a neural network receiving these numbers directly has no built-in understanding that these distances represent nearly identical chemical environments.

Similarly, consider two bond angles,

$$
109.4^\circ
$$

and

$$
109.6^\circ.
$$

These angles differ by only

$$
0.2^\circ,
$$

yet a neural network processing raw numerical values may treat them as entirely independent inputs.

The challenge is even greater because atomic distances and angles vary continuously. Unlike atomic numbers, which belong to a finite set of elements, distances can assume infinitely many values. Consequently, the neural network must learn a smooth function over a continuous geometric space.

To overcome this problem, M3GNet first transforms raw geometric quantities into **continuous high-dimensional feature vectors**. This process is called **geometric encoding** or **basis expansion**.

Instead of representing a bond by a single number,

$$
r_{ij},
$$

the model represents it by an entire vector

$$
\mathbf{g}(r_{ij}),
$$

whose components smoothly vary with distance.

The same strategy is applied to bond angles.

These encodings provide the neural network with a much richer and more expressive description of the local atomic geometry.

---

# 18.5.1 Why Raw Distances Are Not Ideal Inputs

To appreciate the motivation behind geometric encoding, let us first examine what happens if raw distances are used directly.

Suppose the following four bonds exist in a dataset.

| Bond | Distance (Å) |
| ---- | -----------: |
| A–B  |         2.10 |
| C–D  |         2.12 |
| E–F  |         2.80 |
| G–H  |         5.60 |

If these values are passed directly to a neural network,

the model must simultaneously learn that

* 2.10 Å and 2.12 Å represent almost identical interactions,
* 5.60 Å corresponds to an almost negligible interaction,
* the dependence of energy on distance is highly nonlinear.

Learning this nonlinear relationship entirely from raw numerical values is difficult and often requires more training data.

A more effective approach is to transform each distance into a richer feature representation before it reaches the neural network.

---

# 18.5.2 Feature Expansion

The basic idea is remarkably simple.

Rather than describing a bond by one number,

we describe it using many numbers.

Instead of

$$
r_{ij},
$$

we compute

$$
\mathbf{g}(r_{ij})
==================

[g_1,g_2,\ldots,g_K].
$$

Here,

$K$

is the number of basis functions.

Each basis function responds differently to the input distance.

Together,

they provide a detailed description of the local geometry.

Conceptually,

the transformation is

```text id="m3gnet_encoding1"
Distance

↓

Basis Expansion

↓

High-Dimensional Vector

↓

Neural Network
```

Notice that the geometric information is not changed.

Only its representation changes.

---

# 18.5.3 Advantages of Basis Expansion

Expanding geometric quantities into high-dimensional feature vectors provides several important advantages.

### Smooth Learning

Nearby distances produce nearby feature vectors.

This allows the neural network to learn smooth physical relationships.

---

### Improved Numerical Stability

Feature expansion avoids abrupt changes in representation, making optimization more stable during training.

---

### Greater Expressive Power

Different basis functions emphasize different regions of the distance spectrum.

The resulting representation is far richer than a single scalar value.

---

### Better Generalization

Since neighboring geometries produce similar encodings,

the model generalizes more effectively to unseen atomic configurations.

---

# 18.5.4 Radial Basis Functions

The most common method for encoding interatomic distances is the **Radial Basis Function (RBF)** expansion.

Instead of representing a bond by

$r_{ij}$,

we evaluate several overlapping radial basis functions centered at different distances.

Graphically,

the idea can be visualized as

```text id="m3gnet_encoding2"
Basis 1

      /\

     /  \

Basis 2

        /\

       /  \

Basis 3

           /\

          /  \

-----------------------------> Distance
```

Every basis function responds strongly to a different range of distances.

A particular bond therefore activates several basis functions simultaneously.

This produces a smooth, informative representation.

---

# 18.5.5 Gaussian Radial Basis Functions

One of the most widely used radial basis functions is the **Gaussian basis**.

Each Gaussian is centered at a predefined location

$\mu_n$.

Its value is

$$
g_n(r)
======

\exp
\left[
------

\beta
(r-\mu_n)^2
\right],
$$

where

* $r$ is the interatomic distance,
* $\mu_n$ is the center of the basis function,
* $\beta$ controls the width of the Gaussian.

Each basis function reaches its maximum when

$$
r=\mu_n.
$$

As the distance moves away from the center,

its value gradually decreases.

---

## Physical Interpretation

Imagine placing several smooth "bells" along the distance axis.

```text id="m3gnet_encoding3"
      /\

     /  \

   /\    /\

  /  \  /  \

-----------------------------> Distance
```

When a bond length is measured,

it activates nearby bells.

The collection of activations forms the feature vector describing that bond.

Unlike a hard discretization,

Gaussian basis functions vary continuously,

making them ideal for neural networks.

---

# 18.5.6 Example of Distance Encoding

Suppose five Gaussian basis functions are centered at

* 1 Å,
* 2 Å,
* 3 Å,
* 4 Å,
* 5 Å.

Now consider a bond of length

$$
2.3\ \text{Å}.
$$

The encoded vector might resemble

```text id="m3gnet_encoding4"
Distance = 2.3 Å

↓

[0.02,
 0.81,
 0.54,
 0.06,
 0.00]
```

Notice that

the second and third basis functions respond most strongly because their centers are closest to the bond length.

Instead of a single scalar,

the neural network now receives a smooth five-dimensional description of the bond.

---

# 18.5.7 Why Overlapping Basis Functions Are Important

Suppose the basis functions did **not** overlap.

A bond of

2.99 Å

would activate only the basis centered at

3 Å,

whereas a bond of

3.01 Å

would suddenly activate another basis.

The representation would change abruptly,

even though the two bond lengths differ by only

0.02 Å.

Such discontinuities are undesirable because physical properties change smoothly with geometry.

Overlapping Gaussian basis functions eliminate this problem.

Neighboring distances produce similar feature vectors,

which greatly improves learning.

---

# 18.5.8 Number of Basis Functions

An important design choice is the number of basis functions,

denoted by

$K$.

If

$K$

is too small,

the geometric representation becomes coarse.

Important structural differences may be lost.

If

$K$

is excessively large,

the feature vectors become unnecessarily expensive to compute,

and the model may become more difficult to train.

Consequently,

the number of basis functions is treated as a hyperparameter chosen to balance expressiveness and computational efficiency.

---

# 18.5.9 Continuous Rather Than Discrete Geometry

One of the major strengths of basis expansion is that it preserves the continuous nature of atomic geometry.

Consider two bonds.

$$
r_1
===

2.30\ \text{Å},
$$

$$
r_2
===

2.31\ \text{Å}.
$$

After Gaussian expansion,

their feature vectors differ only slightly.

This smooth behavior reflects the underlying physics,

where tiny geometric changes generally produce tiny energy changes.

Such continuity is essential for accurately predicting **forces**, because forces are derivatives of the energy with respect to atomic positions.

Abrupt feature changes would lead to discontinuous energy predictions and unreliable force calculations.

---

### Transition to the Next Part

So far, we have focused on **distance encoding** using radial basis functions. However, M3GNet must also represent **bond angles**, which play a crucial role in three-body interactions.

In the next part of this section, we will study **angular basis functions**, **cutoff functions**, and how M3GNet combines encoded distances and encoded angles into the geometric feature vectors used during message passing.

Continuing **Section 18.5**.

---

# 18.5.10 Encoding Bond Angles

In the previous sections, we learned how interatomic distances are transformed into smooth, high-dimensional vectors using radial basis functions. Distance, however, is only one component of the local atomic geometry.

As discussed extensively in Section **18.3**, the energy of many materials also depends on **bond angles**.

Consider the following atomic triplet.

```text id="m3gnet_angle1"
          k

         /

        /

i ------ j
```

The angle formed at the central atom is

$$
\theta_{ijk}.
$$

This angle cannot simply be passed to the neural network as a single scalar.

For exactly the same reasons that raw distances are poor neural-network inputs, raw angles also present difficulties.

For example,

consider three bond angles

$$
108.9^\circ,
$$

$$
109.4^\circ,
$$

and

$$
109.9^\circ.
$$

These three angles describe nearly identical tetrahedral environments.

A neural network should therefore produce similar internal representations for them.

Instead of using the raw angle,

M3GNet transforms it into a continuous feature vector,

just as it does for interatomic distances.

---

# 18.5.11 Why Raw Angles Are Difficult to Learn

Suppose we provide only the angle

$$
109.47^\circ
$$

to the network.

From the network's perspective,

this is simply another floating-point number.

It has no built-in understanding that

* 109.47° is the ideal tetrahedral angle,
* 120° corresponds to trigonal coordination,
* 180° represents linear geometry.

The model would have to discover these relationships entirely from data.

This is possible,

but highly inefficient.

A much better approach is to transform the angle into a representation that varies smoothly over the angular space.

---

# 18.5.12 Angular Basis Functions

Just as distances are expanded into radial basis functions,

angles are expanded into **angular basis functions**.

Conceptually,

the process is identical.

Instead of

$$
\theta,
$$

the network receives

$$
\mathbf{a}(\theta)
==================

[a_1,a_2,\ldots,a_M].
$$

Each basis function responds most strongly to a particular angular region.

Graphically,

```text id="m3gnet_angle2"
Basis 1

      /\

Basis 2

          /\

Basis 3

               /\

---------------------------------> Angle
```

The resulting feature vector provides a detailed description of the local angular environment.

---

# 18.5.13 Periodicity of Angles

Angles possess an important property that ordinary distances do not.

They are **periodic geometric quantities**.

For example,

the difference between

$$
179^\circ
$$

and

$$
180^\circ
$$

is very small.

Likewise,

small angular distortions should produce only small changes in the feature representation.

Therefore,

angular encoding functions are designed to preserve this smooth periodic behavior.

This allows the neural network to learn continuous energy landscapes without introducing artificial discontinuities.

---

# 18.5.14 Combining Distance and Angle Information

A three-body interaction depends on

* the first bond length,
* the second bond length,
* the bond angle.

Therefore,

every atomic triplet ultimately produces a combined geometric feature vector.

Conceptually,

the process is

```text id="m3gnet_angle3"
Distance ij

↓

Radial Encoding

─────────────┐

             │

Distance jk  │

↓            │

Radial Encoding

─────────────┤

             │

Bond Angle

↓

Angular Encoding

─────────────┘

↓

Three-Body Geometric Feature
```

This feature vector becomes the input for the three-body message-passing operations performed by M3GNet.

Instead of reasoning directly with raw coordinates,

the neural network learns from these encoded geometric descriptors.

---

# 18.5.15 Cutoff Functions

Although neighboring atoms are selected using a cutoff radius,

simply removing all interactions beyond the cutoff introduces a serious numerical problem.

Suppose the cutoff radius is

$$
r_c=5\ \text{Å}.
$$

Now consider two bonds.

$$
r_1=4.99\ \text{Å},
$$

$$
r_2=5.01\ \text{Å}.
$$

Without additional processing,

the first bond contributes fully,

while the second contributes nothing.

This abrupt change is illustrated below.

```text id="m3gnet_cutoff1"
Interaction

│

│───────────────

│

│

└──────────────────────► Distance

                 Cutoff
```

Such discontinuities are physically unrealistic.

In reality,

interatomic interactions decay gradually rather than disappearing instantaneously.

Furthermore,

abrupt changes create discontinuous energy surfaces,

which in turn lead to unstable force predictions.

---

# 18.5.16 Smooth Cutoff Functions

To avoid these discontinuities,

M3GNet multiplies every geometric feature by a **smooth cutoff function**.

Rather than terminating interactions abruptly,

the cutoff function gradually decreases them to zero.

Conceptually,

the behavior becomes

```text id="m3gnet_cutoff2"
Interaction

│

│─────────────╲

│              ╲

│               ╲

└──────────────────────► Distance

                  Cutoff
```

Notice that

the interaction smoothly approaches zero exactly at the cutoff radius.

This ensures that

* the energy changes continuously,
* the force remains continuous,
* molecular dynamics simulations remain numerically stable.

---

# 18.5.17 Properties of a Good Cutoff Function

An effective cutoff function should satisfy several important conditions.

### Unity at Short Distances

Atoms that are close together should interact without attenuation.

Therefore,

$$
f_c(r)
\approx
1,
$$

for small

$r$.

---

### Zero at the Cutoff Radius

Exactly at

$r_c$,

the interaction should vanish smoothly.

$$
f_c(r_c)=0.
$$

---

### Continuous First Derivative

Because forces are derivatives of the energy,

the cutoff function should possess a continuous derivative.

Otherwise,

the predicted forces would contain artificial jumps.

---

### Numerical Stability

Smooth cutoff functions improve optimization,

reduce training instability,

and enhance molecular dynamics simulations.

---

# 18.5.18 Why Encoding Is Essential for Force Prediction

One of the major goals of M3GNet is not merely predicting material properties,

but predicting

* total energy,
* atomic forces,
* stress tensors.

Recall that atomic forces are obtained from the energy through differentiation.

$$
\mathbf{F}_i
============

*

\frac{\partial E}
{\partial \mathbf{R}_i}.
$$

For this derivative to exist and remain stable,

the energy function itself must vary smoothly with atomic positions.

If raw distances were used directly,

or if interactions ended abruptly at the cutoff radius,

the energy surface could contain sharp discontinuities.

These discontinuities would produce unstable force predictions,

making molecular dynamics simulations unreliable.

The combination of

* radial basis functions,
* angular basis functions,
* smooth cutoff functions,

ensures that the learned energy surface remains continuous and differentiable.

This is one of the key reasons M3GNet can accurately predict not only energies but also forces and stresses.

---

# 18.5.19 Summary

Distance and angular encoding form the bridge between crystal geometry and deep learning.

Rather than processing raw bond lengths and bond angles, M3GNet converts these geometric quantities into smooth, high-dimensional feature vectors using basis expansions. Smooth cutoff functions further ensure that interactions decay continuously near the cutoff radius, producing a differentiable potential energy surface suitable for predicting energies, forces, and stresses.

These encoded geometric features become the language through which the neural network understands the atomic environment.

---

## Transition to Section 18.6

At this point, we have all the ingredients required to construct the input representation for M3GNet:

* **Node features** representing atomic identities,
* **Edge features** representing encoded interatomic distances,
* **Triplet features** representing encoded bond angles,
* **Smooth cutoff functions** ensuring continuous interactions.

The next step is to understand **how the neural network processes these features**.

In **Section 18.6 – M3GNet Architecture**, we will build the complete network layer by layer, beginning with the embedding layer and progressing through message-passing blocks, three-body interaction modules, state updates, and the final readout network that predicts energies, forces, and stresses.

# 18.6 M3GNet Architecture

In the previous sections, we transformed a crystal structure into a rich graph representation. Every atom became a node, neighboring atoms were connected through edges, bond lengths were encoded using radial basis functions, and bond angles were represented through three-body geometric features. At this point, the crystal has been converted into a mathematical object that a neural network can process.

The next question is the central one of this chapter:

> **How does M3GNet process this graph to predict the energy, forces, and stresses of a material?**

The answer lies in its carefully designed neural network architecture.

Unlike conventional feedforward neural networks, which operate on fixed-size vectors, M3GNet is a **graph neural network**. Information flows through the graph by repeatedly exchanging messages between neighboring atoms. During this process, each atom gradually acquires information about its surrounding chemical environment, allowing the model to learn complex many-body interactions directly from data.

However, M3GNet extends the traditional graph neural network framework in several important ways.

Its architecture simultaneously updates

* atomic (node) features,
* bond (edge) features,
* global state features,

while explicitly incorporating **three-body interactions** through angular message passing.

This combination enables the network to construct a highly expressive representation of the local atomic environment while maintaining physical consistency.

---

# 18.6.1 High-Level Architecture

Before examining each component individually, it is useful to view the complete workflow.

```text id="m3gnet_arch1"
Crystal Structure

↓

Graph Construction

↓

Node Features

Edge Features

Triplet Features

↓

Embedding Layer

↓

Interaction Block 1

↓

Interaction Block 2

↓

Interaction Block 3

↓

⋮

↓

Readout Layer

↓

Total Energy

↓

Forces

Stress
```

The architecture can be divided into five major stages:

1. **Input graph construction**
2. **Embedding layer**
3. **Interaction blocks**
4. **Readout network**
5. **Differentiation to obtain forces and stresses**

Each stage performs a distinct task that contributes to the final prediction.

---

# 18.6.2 Input to the Network

The network receives three types of information.

### Node Features

Each atom

$i$

is represented by an embedding vector

$$
\mathbf{v}_i.
$$

Initially, this embedding depends primarily on the atomic species.

For example,

```text id="m3gnet_arch2"
Carbon

↓

Embedding

↓

v₁
```

```text id="m3gnet_arch3"
Silicon

↓

Embedding

↓

v₂
```

These embeddings are learned during training.

---

### Edge Features

Each neighboring atom pair

$(i,j)$

is associated with an encoded distance feature

$$
\mathbf{e}_{ij}.
$$

These features contain information about

* bond length,
* radial basis expansion,
* local geometric relationships.

---

### Triplet Features

Each atomic triplet

$$
(i,j,k)
$$

is represented by

$$
\mathbf{t}_{ijk},
$$

which contains encoded angular information.

These triplet features distinguish M3GNet from earlier pairwise graph neural networks.

---

# 18.6.3 Why an Embedding Layer Is Needed

The initial node and edge features contain only limited information.

For example,

an atomic number alone cannot capture

* oxidation state,
* bonding environment,
* coordination,
* electronic structure.

Therefore,

the first stage of the neural network transforms the raw inputs into higher-dimensional representations.

Suppose

the initial node feature is

$$
\mathbf{x}_i.
$$

The embedding layer computes

$$
\mathbf{h}_i^{(0)}
==================

f_{\text{embed}}
(
\mathbf{x}_i
),
$$

where

$$
\mathbf{h}_i^{(0)}
$$

is the initial hidden representation.

This vector becomes the starting point for all subsequent message-passing operations.

---

# 18.6.4 Hidden Feature Space

An important concept in deep learning is the **hidden feature space**.

Rather than working directly with atomic numbers or distances, the network converts every quantity into vectors in a high-dimensional space.

For example,

an atom may initially be represented as

```text id="m3gnet_arch4"
Silicon

↓

Atomic Number = 14
```

After embedding,

the representation becomes

```text id="m3gnet_arch5"
[-0.42,

 0.81,

 1.37,

 ...

 0.15]
```

Although these numerical values have no direct physical interpretation, they encode information useful for predicting material properties.

As training progresses, these embeddings evolve to capture increasingly sophisticated chemical relationships.

---

# 18.6.5 Message Passing

The heart of M3GNet is **message passing**.

The basic idea is simple.

Atoms do not make predictions independently.

Instead,

each atom gathers information from its neighboring atoms.

Suppose atom

$i$

has neighboring atoms

$j$

and

$k$.

```text id="m3gnet_arch6"
      j

      |

      |

i -------- k
```

Messages flow from

$j$

and

$k$

toward

$i$.

The received information is combined to update the representation of atom

$i$.

This process is repeated many times throughout the network.

Consequently,

each atom gradually acquires information about increasingly distant regions of the crystal.

---

# 18.6.6 What Is a Message?

A message is simply a vector of information transmitted along an edge.

For the interaction

$j \rightarrow i$,

the message is written as

$$
\mathbf{m}_{ji}.
$$

Rather than depending only on the neighboring atom,

the message is a function of both atoms and the connecting bond.

Conceptually,

$$
\mathbf{m}_{ji}
===============

\phi
(
\mathbf{h}_j,
\mathbf{h}*i,
\mathbf{e}*{ji}
),
$$

where

* $\mathbf{h}_j$ is the hidden feature of the sending atom,
* $\mathbf{h}_i$ is the hidden feature of the receiving atom,
* $\mathbf{e}_{ji}$ contains the encoded bond information,
* $\phi$ is a learnable neural network.

This formulation allows the transmitted information to depend on both chemical identity and geometry.

---

# 18.6.7 Message Aggregation

A single atom typically has several neighbors.

Suppose atom

$i$

receives messages from

* atom 1,
* atom 2,
* atom 3,
* atom 4.

These messages must be combined into one vector.

The aggregation operation is

$$
\mathbf{m}_i
============

\sum_{j\in\mathcal N(i)}
\mathbf{m}_{ji}.
$$

Graphically,

```text id="m3gnet_arch7"
Message 1

↓

Message 2

↓

Message 3

↓

Message 4

↓

Summation

↓

Combined Message
```

The summation operation has an important advantage.

It is **permutation invariant**.

In other words,

the order in which neighboring atoms are listed does not affect the result.

This property is essential because the physical properties of a crystal cannot depend on arbitrary indexing of atoms.

---

# 18.6.8 Node Update

Once the messages have been aggregated,

the atomic representation is updated.

Conceptually,

the update operation is

$$
\mathbf{h}_i^{(l+1)}
====================

U
(
\mathbf{h}_i^{(l)},
\mathbf{m}_i
),
$$

where

* $l$ denotes the interaction layer,
* $U$ is a learnable neural network.

The updated representation contains information from both the atom itself and its surrounding environment.

After several interaction layers,

each atom develops a highly informative representation describing its local chemistry and geometry.

---

# 18.6.9 Why Multiple Interaction Layers Are Needed

A single message-passing layer allows an atom to receive information only from its immediate neighbors.

Consider the following chain.

```text id="m3gnet_arch8"
A ----- B ----- C ----- D
```

After one interaction layer,

atom

A

knows about

B,

but not about

C

or

D.

After two layers,

A

receives information originating from

C.

After three layers,

information from

D

can also influence

A.

Thus,

deeper networks enable atoms to learn about increasingly distant regions of the crystal.

This hierarchical information propagation is one of the defining strengths of graph neural networks.

---

## Looking Ahead

So far, we have described the **general message-passing mechanism** used by M3GNet. However, the architecture contains a major innovation that distinguishes it from earlier graph neural networks.

Unlike CGCNN or standard message-passing networks, **M3GNet does not update only atomic (node) features**. It also updates **bond (edge) features** and incorporates **three-body angular interactions** during every interaction block.

These coupled updates are the core of the M3GNet architecture.

In the next subsection (**18.6.10**), we will study the complete **M3GNet Interaction Block**, examining how node, edge, and three-body features are updated simultaneously through learnable neural networks. This is the central architectural contribution of M3GNet and the key to its high predictive accuracy.

## 18.6.10 The M3GNet Interaction Block

The interaction block is the **fundamental computational unit** of M3GNet. Every prediction made by the network—whether it is the total energy, atomic forces, or stress tensor—depends on the repeated application of these interaction blocks.

Earlier graph neural networks, such as CGCNN, primarily update **node features** while treating edge features as fixed descriptors. M3GNet adopts a much richer strategy.

During each interaction block, **three different types of information are updated simultaneously**:

1. **Atomic (node) features**
2. **Bond (edge) features**
3. **Global state features** (when available)

Because the edge features themselves evolve during learning, the interaction between two atoms is no longer fixed. Instead, it adapts continuously according to the surrounding chemical environment.

This dynamic representation enables M3GNet to model complex many-body interactions much more effectively than earlier architectures.

---

## 18.6.11 Overall Structure of an Interaction Block

Conceptually, one interaction block can be viewed as the following sequence of operations.

```text id="m3gnet_block1"
Node Features

Edge Features

State Features

↓

Edge Update

↓

Three-Body Interaction

↓

Node Update

↓

State Update

↓

Updated Features
```

Notice that the information flows in stages.

The edge features are updated first because they define how neighboring atoms interact. These updated edges are then used to compute improved atomic representations. Finally, the global state is updated using information gathered from the entire graph.

Each interaction block therefore produces a more informative representation than the previous one.

---

## 18.6.12 Why Update Edge Features?

To understand why edge updates are important, consider two identical Si–O bonds.

```text id="m3gnet_block2"
Si -------- O
```

Suppose the bond length is

$$
1.62\ \text{Å}
$$

in both cases.

If edge features remain fixed, both bonds appear identical to the neural network.

However, imagine the surrounding environments are different.

### Environment A

```text id="m3gnet_block3"
      O

      |

Si ---- O
```

### Environment B

```text id="m3gnet_block4"
      N

      |

Si ---- O
```

Although the Si–O bond length is unchanged,

the neighboring atom differs.

This changes

* electron density,
* local charge distribution,
* bond strength,
* local energy.

Therefore,

the edge representing the Si–O bond should also change.

M3GNet allows this by updating edge features after every interaction block.

Consequently,

the same bond can acquire different representations depending on its local chemical environment.

---

## 18.6.13 Edge Update Network

Suppose the edge connecting atoms

$i$

and

$j$

has feature vector

$$
\mathbf{e}_{ij}^{(l)}
$$

at layer

$l$.

The interaction block computes a new edge representation

$$
\mathbf{e}_{ij}^{(l+1)}
=======================

\phi_e
\left(
\mathbf{h}_i^{(l)},
\mathbf{h}*j^{(l)},
\mathbf{e}*{ij}^{(l)}
\right),
$$

where

* $\mathbf{h}_i^{(l)}$ is the hidden feature of atom $i$,
* $\mathbf{h}_j^{(l)}$ is the hidden feature of atom $j$,
* $\phi_e$ is a learnable neural network.

Several important observations can be made.

First,

the updated bond depends on **both connected atoms**.

Second,

the bond also depends on its previous representation.

Thus,

edge features gradually evolve throughout the network, incorporating increasingly sophisticated chemical information.

---

## 18.6.14 Physical Interpretation of Edge Updates

Rather than thinking of an edge as a fixed bond,

it is more accurate to regard it as a **dynamic description of the interaction** between two atoms.

Initially,

an edge may contain only

* bond length,
* radial basis expansion.

After several interaction blocks,

the same edge also contains information about

* neighboring atoms,
* bond angles,
* coordination environment,
* chemical composition,
* local electronic environment.

In other words,

the edge becomes a learned representation of the bond itself.

This adaptive behavior is one of the major reasons M3GNet outperforms models that rely on fixed geometric descriptors.

---

## 18.6.15 Incorporating Three-Body Information

After updating the edge features,

M3GNet incorporates information from atomic triplets.

Recall that every triplet consists of

$$
(i,j,k),
$$

where

$j$

is the central atom.

```text id="m3gnet_block5"
        k

       /

      /

i ---- j
```

The triplet contains

* bond

$i-j$,

* bond

$j-k$,

* bond angle

$$
\theta_{ijk}.
$$

Rather than processing the two bonds independently,

the interaction block combines them into a unified three-body representation.

This allows the neural network to learn how bond angles influence the strength of neighboring interactions.

---

## 18.6.16 Three-Body Message Construction

Conceptually,

the three-body message is computed from

* the first edge,
* the second edge,
* the encoded bond angle.

Mathematically,

we may write

$$
\mathbf{m}_{ijk}
================

\phi_3
\left(
\mathbf{e}*{ij},
\mathbf{e}*{jk},
\mathbf{t}_{ijk}
\right),
$$

where

* $\mathbf{t}_{ijk}$ contains the encoded angular information,
* $\phi_3$ is a learnable neural network.

This equation illustrates one of the defining ideas of M3GNet.

The interaction between two atoms is influenced not only by the bond connecting them but also by the surrounding angular geometry.

---

## 18.6.17 Information Flow Through a Triplet

The interaction can be visualized as

```text id="m3gnet_block6"
Bond ij

↓

Encoded Distance

↓

┐

│

│

Bond jk

↓

Encoded Distance

↓

┤

│

│

Bond Angle

↓

Angular Encoding

↓

┘

↓

Three-Body Network

↓

Updated Bond Information
```

Instead of treating each bond separately,

the interaction block learns how neighboring bonds influence one another through the bond angle.

This allows the network to model directional chemical bonding directly.

---

## 18.6.18 Updating Atomic Features

Once the edge and three-body information have been updated,

the network computes new atomic representations.

Each neighboring bond sends information toward the central atom.

For atom

$i$,

the incoming messages are aggregated,

and the atomic representation becomes

$$
\mathbf{h}_i^{(l+1)}
====================

\phi_v
\left(
\mathbf{h}*i^{(l)},
\sum*{j\in\mathcal N(i)}
\mathbf{m}_{ji}
\right),
$$

where

$\phi_v$

is another learnable neural network.

Notice that the incoming messages already contain three-body information because the edge features have been updated using angular interactions.

Consequently,

the atomic representation also becomes angle-aware.

---

## 18.6.19 One Interaction Block Is Not Enough

A single interaction block allows information to travel only one neighborhood away.

After one block,

an atom understands only its immediate surroundings.

After two blocks,

it begins to receive information originating from neighbors of neighbors.

After three blocks,

its receptive field becomes even larger.

Graphically,

```text id="m3gnet_block7"
Block 1

A ← B

Block 2

A ← B ← C

Block 3

A ← B ← C ← D
```

Thus,

stacking multiple interaction blocks allows the model to construct increasingly global representations while preserving detailed local chemistry.

This hierarchical information propagation is analogous to how convolutional neural networks gradually increase their receptive field in image recognition tasks.

---

## 18.6.20 Residual Connections

One challenge encountered in very deep neural networks is the **vanishing information problem**.

If each interaction block completely replaces the previous representation,

important information from earlier layers may gradually disappear.

To avoid this problem,

M3GNet employs **residual connections**.

Instead of learning an entirely new representation,

the network learns only a correction.

Conceptually,

the update becomes

$$
\mathbf{h}^{(l+1)}
==================

\mathbf{h}^{(l)}
+
\Delta\mathbf{h}^{(l)}.
$$

Here,

$\Delta\mathbf{h}^{(l)}$

is the correction predicted by the interaction block.

This strategy offers several advantages:

* it improves gradient flow during backpropagation,
* enables much deeper networks,
* accelerates convergence,
* preserves useful information from earlier layers.

Residual learning has become a standard design principle in modern deep learning architectures, and it plays an important role in the stability and accuracy of M3GNet.

---

### Transition to the Next Subsection

We have now examined how a single interaction block updates **edge features**, incorporates **three-body interactions**, and produces improved **atomic representations**.

However, M3GNet still needs to combine the information from **all atoms in the crystal** to predict a single property such as the total energy.

In the next subsection (**18.6.21 Readout Network and Energy Prediction**), we will study how the final atomic representations are pooled into a graph-level representation and transformed into predictions of total energy, from which forces and stresses are subsequently obtained.

## 18.6.21 Readout Network and Energy Prediction

After several interaction blocks, every atom has accumulated information from its local environment. The hidden representation of an atom no longer reflects only its chemical identity; instead, it encodes information about

* neighboring atoms,
* bond lengths,
* bond angles,
* coordination environment,
* local crystal symmetry,
* chemical composition.

At this stage, the graph contains highly informative atomic representations.

The remaining task is to convert these atomic representations into physically meaningful quantities such as

* total energy,
* atomic forces,
* stress tensor.

This is the role of the **readout network**.

---

## 18.6.22 Local Energy Approximation

One of the most important physical assumptions made by M3GNet is that the total energy of a crystal can be decomposed into **atomic contributions**.

Instead of predicting the energy of the entire crystal directly, the model predicts the energy associated with each individual atom.

Mathematically,

$$
E
=

\sum_{i=1}^{N}
E_i,
$$

where

* $E$ is the total energy,
* $E_i$ is the energy contribution of atom $i$,
* $N$ is the number of atoms.

This assumption is widely used in modern machine learning interatomic potentials because the energy of an atom is primarily determined by its local environment.

Although an atom does not possess an independent physical energy in a strict quantum mechanical sense, this decomposition provides an accurate and computationally efficient approximation for learning potential energy surfaces.

---

## 18.6.23 Why Predict Atomic Energies?

Predicting atomic energies instead of directly predicting the total crystal energy offers several advantages.

### Scalability

Suppose a model is trained using unit cells containing

50

atoms.

If the model predicts only the total energy,

it cannot easily generalize to crystals containing

500

atoms.

However,

if it predicts atomic energies,

the total energy is simply

$$
E
=

E_1+E_2+\cdots+E_{500}.
$$

The model therefore scales naturally to systems of different sizes.

---

### Size Extensivity

A fundamental property of energy is **extensivity**.

If two identical crystals are placed together,

the total energy should double.

Suppose

Crystal A has energy

$$
E.
$$

Two identical copies should have energy

$$
2E.
$$

Because M3GNet sums atomic contributions,

this property is automatically satisfied.

This is known as **size extensivity**, and it is an essential requirement for any physically meaningful interatomic potential.

---

### Better Transferability

Since each atomic energy depends only on the local environment,

the model can be transferred to

* larger crystals,
* supercells,
* surfaces,
* grain boundaries,
* defects,

without requiring changes to the architecture.

---

## 18.6.24 Atomic Energy Prediction

Each interaction block produces a hidden feature vector

$$
\mathbf{h}_i.
$$

The readout network converts this feature vector into an atomic energy.

Conceptually,

$$
E_i
===

f_{\text{readout}}
(
\mathbf{h}_i
),
$$

where

$f_{\text{readout}}$

is a small multilayer neural network.

Graphically,

```text id="m3gnet_readout1"
Hidden Feature

↓

Fully Connected Layer

↓

Activation

↓

Fully Connected Layer

↓

Atomic Energy
```

The same readout network is applied to every atom.

Because the weights are shared,

the model can process crystals containing any number of atoms.

---

## 18.6.25 Pooling Operation

Once every atomic energy has been computed,

the network combines them using a **pooling operation**.

Unlike image classification,

where pooling often computes a maximum or an average,

energy prediction requires summation.

Therefore,

the pooling layer performs

$$
E
=

\sum_i
E_i.
$$

Graphically,

```text id="m3gnet_readout2"
Atom 1

↓

E₁

Atom 2

↓

E₂

Atom 3

↓

E₃

⋮

↓

Summation

↓

Total Energy
```

This simple operation guarantees that the predicted energy satisfies the physical requirement of extensivity.

---

## 18.6.26 Why Not Average the Atomic Energies?

One might wonder why we do not compute

$$
E
=

\frac{1}{N}
\sum_i
E_i.
$$

This would produce the average atomic energy.

Although averaging is useful for predicting intensive properties,

it is unsuitable for predicting the total energy.

Suppose a crystal is duplicated.

The average atomic energy remains unchanged,

whereas the total energy should double.

Consequently,

energy prediction requires summation rather than averaging.

For properties such as

* formation energy per atom,
* average magnetic moment,
* average charge,

mean pooling may be appropriate.

For total energy,

sum pooling is the physically correct choice.

---

## 18.6.27 Graph-Level Representation

Although M3GNet predicts atomic energies individually,

one can also view the summed representation as a **graph-level embedding**.

Conceptually,

the workflow is

```text id="m3gnet_readout3"
Atom Embeddings

↓

Atomic Energies

↓

Summation

↓

Crystal Representation

↓

Property Prediction
```

The graph representation contains information about the entire material.

Different output heads can then be attached for predicting different physical quantities.

For example,

```text id="m3gnet_readout4"
Graph Embedding

├── Energy Head

├── Band Gap Head

├── Elastic Property Head

├── Formation Energy Head

└── Other Properties
```

Although M3GNet is primarily designed as an interatomic potential,

its learned graph representations can also support many other prediction tasks.

---

## 18.6.28 Energy Conservation

A major advantage of predicting energy first is that it automatically enforces **energy conservation**.

In classical mechanics,

forces are not independent quantities.

Instead,

they arise from the potential energy surface.

Consequently,

if the neural network predicts a physically consistent energy,

all derived forces also satisfy the laws of conservative mechanics.

This is one of the most important conceptual differences between

* directly predicting forces,

and

* predicting energy and differentiating it.

M3GNet follows the second strategy.

As we shall see in the next section,

forces are obtained directly from the gradient of the predicted energy.

This guarantees consistency between

* energy,
* forces,
* stress.

---

## 18.6.29 Computational Workflow of the Readout Stage

The complete prediction pipeline can now be summarized.

```text id="m3gnet_readout5"
Crystal Structure

↓

Graph Construction

↓

Embedding Layer

↓

Interaction Blocks

↓

Updated Atomic Features

↓

Readout Network

↓

Atomic Energies

↓

Summation

↓

Total Energy
```

Notice that the neural network predicts **only one scalar quantity** at this stage:

the total energy.

All remaining physical quantities are derived from this energy using automatic differentiation.

---

## 18.6.30 Why Energy Is the Central Quantity

From a physical standpoint, energy occupies a unique position because many important material properties can be derived directly from it.

Once the total energy is known as a differentiable function of atomic coordinates and lattice vectors, one can compute:

* **Atomic forces** from the gradient of the energy with respect to atomic positions.
* **Stress tensors** from the derivative of the energy with respect to strain or lattice deformation.
* **Equilibrium structures** by minimizing the total energy.
* **Phonon properties** through higher-order derivatives of the energy.
* **Molecular dynamics trajectories** by integrating Newton's equations using the predicted forces.

This is why modern machine learning interatomic potentials focus on learning an accurate and differentiable energy function rather than attempting to predict every physical property independently.

---

### Transition to the Next Section

We have now completed the forward prediction pipeline of M3GNet, from graph construction to total energy prediction.

However, one of M3GNet's greatest strengths is that it predicts much more than energy. Because the predicted energy is fully differentiable, the network can obtain **atomic forces** and **stress tensors** through automatic differentiation, without requiring separate force or stress models.

In the next section (**18.7 Mathematical Formulation**), we will develop the complete mathematical framework of M3GNet, deriving the equations governing message passing, energy prediction, force calculation, and stress prediction in a rigorous, step-by-step manner. This section forms the theoretical foundation of the entire architecture and connects the neural network operations to the underlying physics.


# 18.7 Mathematical Formulation

The previous sections introduced the architecture of M3GNet from a conceptual perspective. We learned how crystal structures are converted into graphs, how geometric information is encoded, and how interaction blocks repeatedly update atomic representations before predicting the total energy.

While this provides an intuitive understanding of the model, modern materials informatics research requires a much deeper mathematical understanding.

To modify existing architectures, develop new models, or interpret research papers, it is essential to understand the equations governing every stage of the network.

In this section, we will derive the mathematical formulation of M3GNet step by step.

Unlike many research papers that present the equations in a compact form, we will carefully explain

* the meaning of every symbol,
* why each equation is needed,
* how the equations relate to the underlying physics,
* how the equations are implemented computationally.

Our objective is not merely to memorize formulas but to understand how geometry is transformed into predictions of energy, forces, and stresses.

---

# 18.7.1 Mathematical Representation of a Crystal

Everything begins with a crystal structure.

Suppose a crystal contains

$N$

atoms.

The crystal may be represented mathematically as

$$
\mathcal{C}
===========

{
(Z_i,\mathbf{R}*i)
}*{i=1}^{N},
$$

where

* $Z_i$ is the atomic number of atom $i$,
* $\mathbf{R}_i$ is the Cartesian coordinate of atom $i$.

The coordinate vector is

$$
\mathbf{R}_i
============

(x_i,y_i,z_i).
$$

Thus,

every atom contributes

* its chemical identity,
* its spatial position.

These quantities constitute the raw input to M3GNet.

---

# 18.7.2 Constructing the Graph

The crystal is converted into a graph

$$
G=(V,E),
$$

where

* $V$ is the set of atoms,
* $E$ is the set of neighboring interactions.

Each atom becomes one graph node.

Each neighboring atom pair becomes one graph edge.

If atoms

$i$

and

$j$

are separated by a distance smaller than the cutoff radius,

$$
r_{ij}<r_c,
$$

then

$$
(i,j)\in E.
$$

The graph therefore depends directly on the atomic geometry.

---

# 18.7.3 Node Features

Each atom is represented by a feature vector

$$
\mathbf{v}_i^{(0)}.
$$

Initially,

this feature is obtained from the atomic number through an embedding function.

Mathematically,

$$
\mathbf{v}_i^{(0)}
==================

\mathrm{Embed}(Z_i).
$$

The embedding layer converts discrete atomic identities into continuous vectors.

If the embedding dimension is

$d$,

then

$$
\mathbf{v}_i^{(0)}
\in
\mathbb{R}^{d}.
$$

The superscript

$(0)$

indicates that this is the initial representation before any message passing occurs.

---

# 18.7.4 Edge Features

For every neighboring atom pair,

the edge feature begins with the interatomic distance.

The distance between atoms

$i$

and

$j$

is

$$
r_{ij}
======

\left|
\mathbf{R}_i-\mathbf{R}_j
\right|.
$$

This scalar quantity is expanded using radial basis functions.

The encoded edge feature becomes

$$
\mathbf{e}_{ij}
===============

\mathrm{RBF}(r_{ij}).
$$

Instead of one scalar,

the network now receives a vector describing the bond.

Suppose

$K$

radial basis functions are used.

Then

$$
\mathbf{e}_{ij}
\in
\mathbb{R}^{K}.
$$

---

# 18.7.5 Triplet Features

For every atomic triplet

$$
(i,j,k),
$$

the bond angle is

$$
\theta_{ijk}.
$$

This angle is encoded using angular basis functions.

The resulting feature vector is

$$
\mathbf{t}_{ijk}
================

\mathrm{ABF}
(
\theta_{ijk}
).
$$

If

$M$

angular basis functions are used,

then

$$
\mathbf{t}_{ijk}
\in
\mathbb{R}^{M}.
$$

Consequently,

every triplet contributes an independent geometric descriptor.

---

# 18.7.6 Initial Graph Representation

At the beginning of the network,

the graph consists of three collections.

$$
G
=

(
V,
E,
T
),
$$

where

* $V$ contains node features,
* $E$ contains encoded edge features,
* $T$ contains encoded triplet features.

Thus,

the neural network begins with

```text id="m3gnet_math1"
Nodes

↓

Atomic Embeddings

Edges

↓

Encoded Distances

Triplets

↓

Encoded Angles
```

These three feature sets are repeatedly updated throughout the interaction blocks.

---

# 18.7.7 Edge Update Equation

The first computational step inside an interaction block is updating edge features.

The updated edge is

$$
\mathbf{e}_{ij}^{(l+1)}
=======================

\phi_e
\left(
\mathbf{v}_i^{(l)},
\mathbf{v}*j^{(l)},
\mathbf{e}*{ij}^{(l)}
\right),
$$

where

* $\phi_e$ is a neural network,
* $l$ denotes the interaction layer.

This equation states that the new bond representation depends on

* the current bond,
* the sending atom,
* the receiving atom.

Unlike fixed descriptors,

the edge evolves during learning.

---

# 18.7.8 Three-Body Update

M3GNet then incorporates angular information.

For every triplet,

the three-body message is

$$
\mathbf{m}_{ijk}
================

\phi_3
\left(
\mathbf{e}*{ij},
\mathbf{e}*{jk},
\mathbf{t}_{ijk}
\right).
$$

Notice that

three-body information depends simultaneously on

* two neighboring bonds,
* one bond angle.

This equation is one of the defining characteristics of M3GNet.

Earlier graph neural networks generally lack this explicit angular interaction.

---

# 18.7.9 Message Construction

Each neighboring atom sends a message toward the central atom.

The message is

$$
\mathbf{m}_{ji}
===============

\phi_m
\left(
\mathbf{v}*j,
\mathbf{e}*{ji}
\right).
$$

Because the edge already contains three-body information,

the message itself becomes geometry-aware.

Consequently,

information transmitted between atoms depends not only on distance but also on the surrounding angular environment.

---

# 18.7.10 Message Aggregation

Every atom receives multiple incoming messages.

Suppose

$$
\mathcal N(i)
$$

is the neighbor list of atom

$i$.

The aggregated message becomes

$$
\mathbf{m}_i
============

\sum_{j\in\mathcal N(i)}
\mathbf{m}_{ji}.
$$

This summation possesses an important mathematical property.

It is **permutation invariant**.

Changing the order of neighboring atoms does not change the result,

which is essential because physical predictions must be independent of atom indexing.

---

# 18.7.11 Node Update Equation

Finally,

the atomic representation is updated.

The new hidden feature becomes

$$
\mathbf{v}_i^{(l+1)}
====================

\phi_v
\left(
\mathbf{v}_i^{(l)},
\mathbf{m}_i
\right).
$$

This equation is repeated throughout every interaction block.

After multiple updates,

the hidden feature gradually accumulates information from larger regions of the crystal.

Initially,

the representation describes only the atom itself.

Later,

it describes

* neighboring atoms,
* neighboring bonds,
* bond angles,
* local crystal symmetry,
* chemical environment.

The hidden representation therefore becomes an increasingly complete description of the local atomic physics.

---

## Transition to the Next Part

So far, we have derived the mathematical equations governing **graph construction** and **message passing**.

However, the network has not yet produced a physical prediction.

In the next part of **Section 18.7**, we will derive the equations for

* atomic energy prediction,
* total energy calculation,
* automatic differentiation,
* force computation,
* stress tensor prediction,

showing how M3GNet transforms learned graph representations into a complete machine learning interatomic potential.

Continuing **Section 18.7**.

---

# 18.7.12 Atomic Energy Prediction

After the message-passing process has been repeated several times, each atom possesses a final hidden representation that summarizes its local chemical environment.

Suppose the final hidden representation of atom

$i$

is

$$
\mathbf{v}_i^{(L)},
$$

where

$L$

denotes the last interaction layer.

The purpose of the readout network is to convert this hidden representation into the atomic energy contribution.

Mathematically,

$$
E_i
===

\phi_r
\left(
\mathbf{v}_i^{(L)}
\right),
$$

where

* $E_i$ is the predicted energy contribution of atom $i$,
* $\phi_r$ is the readout neural network.

Unlike the interaction blocks,

the readout network is relatively simple.

It usually consists of several fully connected layers that transform the hidden feature vector into a single scalar.

Graphically,

```text id="m3gnet_math2"
Hidden Feature

↓

Linear Layer

↓

Activation

↓

Linear Layer

↓

Atomic Energy
```

The same readout network is applied to every atom.

Consequently,

the number of trainable parameters does not depend on the size of the crystal.

---

# 18.7.13 Total Energy

The total energy of the crystal is obtained by summing the atomic energy contributions.

The governing equation is

$$
E
=

\sum_{i=1}^{N}
E_i.
$$

Substituting the readout function,

we obtain

$$
E
=

\sum_{i=1}^{N}
\phi_r
\left(
\mathbf{v}_i^{(L)}
\right).
$$

This equation is one of the most important expressions in the entire M3GNet framework.

It tells us that the neural network does **not** predict the crystal energy directly.

Instead,

it predicts

many local atomic energies,

which are subsequently summed.

---

# 18.7.14 Why the Summation Is Physically Correct

One of the major requirements of an interatomic potential is **size extensivity**.

Suppose

Crystal A

contains

100

atoms,

and its predicted energy is

$$
E.
$$

Now suppose we duplicate the crystal,

forming a larger system containing

200

atoms.

The energy should become

$$
2E.
$$

Because M3GNet predicts

$$
E
=

\sum_iE_i,
$$

this requirement is satisfied automatically.

This property would not necessarily hold if the network directly predicted the total energy using a single graph embedding.

Thus,

the atomic energy decomposition is both computationally convenient and physically meaningful.

---

# 18.7.15 Energy as a Function of Atomic Coordinates

At first glance,

the energy equation appears to depend only on the hidden feature vectors.

However,

those hidden vectors themselves depend on

* atomic positions,
* bond lengths,
* bond angles.

Therefore,

the total energy is ultimately a function of all atomic coordinates.

More precisely,

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

This dependence is extremely important.

Because every operation inside M3GNet is differentiable,

the energy function is also differentiable.

This allows forces and stresses to be obtained directly through calculus.

---

# 18.7.16 Force Prediction

In classical mechanics,

the force acting on an atom is the negative gradient of the potential energy.

M3GNet follows exactly the same physical principle.

The force acting on atom

$i$

is

$$
\mathbf{F}_i
============

*

\frac{\partial E}
{\partial \mathbf{R}_i}.
$$

Here,

* $E$ is the total energy,
* $\mathbf{R}_i$ is the atomic position vector.

Notice that no separate force neural network is required.

Once the energy has been learned,

the forces follow automatically through differentiation.

This is one of the most elegant aspects of modern machine learning interatomic potentials.

---

# 18.7.17 Why Automatic Differentiation Works

The computation performed by M3GNet can be viewed as a long sequence of differentiable mathematical operations.

Conceptually,

the computational graph is

```text id="m3gnet_math3"
Atomic Coordinates

↓

Distances

↓

Basis Expansion

↓

Graph Features

↓

Interaction Blocks

↓

Atomic Energies

↓

Total Energy
```

Every step in this workflow is differentiable.

Therefore,

the chain rule can be applied automatically to compute

$$
\frac{\partial E}
{\partial \mathbf{R}_i}.
$$

Rather than deriving complicated analytical derivatives by hand,

deep learning frameworks such as PyTorch compute these gradients automatically using **automatic differentiation**.

This significantly simplifies both implementation and optimization.

---

# 18.7.18 The Chain Rule

To understand automatic differentiation,

consider the dependence of the energy on the hidden representations.

The energy depends on

* atomic features,
* edge features,
* distances,
* atomic positions.

Applying the chain rule,

the derivative becomes

$$
\frac{\partial E}
{\partial \mathbf{R}_i}
=======================

\frac{\partial E}
{\partial \mathbf{v}}
,
\frac{\partial \mathbf{v}}
{\partial \mathbf{e}}
,
\frac{\partial \mathbf{e}}
{\partial r}
,
\frac{\partial r}
{\partial \mathbf{R}_i}.
$$

Although this equation appears complicated,

modern deep learning software evaluates it automatically.

Researchers therefore implement only the forward computation.

The backward gradients are generated by the automatic differentiation engine.

---

# 18.7.19 Force Components

The force vector has three components.

$$
\mathbf{F}_i
============

(F_x,F_y,F_z).
$$

Equivalently,

$$
F_x
===

*

\frac{\partial E}
{\partial x_i},
$$

$$
F_y
===

*

\frac{\partial E}
{\partial y_i},
$$

$$
F_z
===

*

\frac{\partial E}
{\partial z_i}.
$$

Thus,

every atomic coordinate contributes independently to the force.

When the predicted energy surface is smooth,

the resulting force field is also smooth,

making molecular dynamics simulations numerically stable.

---

# 18.7.20 Why Predicting Forces Through Energy Is Better

An alternative approach would be to train a neural network that predicts forces directly.

Although possible,

this strategy has significant disadvantages.

First,

the predicted forces may violate energy conservation.

Specifically,

the predicted force field might not correspond to the gradient of any physically meaningful energy surface.

Such a force field is called **non-conservative**.

In contrast,

M3GNet predicts energy first.

Because forces are computed as

$$
\mathbf{F}
==========

*

\nabla E,
$$

the resulting force field is automatically conservative.

This guarantees consistency between

* energies,
* forces,
* structural relaxation,
* molecular dynamics.

For this reason,

nearly all modern machine learning interatomic potentials adopt the energy-first strategy.

---

# 18.7.21 Stress Tensor Prediction

In addition to atomic forces,

materials simulations often require the **stress tensor**.

Stress describes how the energy changes when the simulation cell is deformed.

Instead of differentiating with respect to atomic positions,

the derivative is taken with respect to strain.

The stress tensor is

$$
\sigma
======

\frac{1}{V}
\frac{\partial E}
{\partial \varepsilon},
$$

where

* $V$ is the cell volume,
* $\varepsilon$ is the strain tensor.

The stress tensor contains nine components,

usually arranged as

$$
\boldsymbol{\sigma}
===================

\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} & \sigma_{xz}\
\sigma_{yx} & \sigma_{yy} & \sigma_{yz}\
\sigma_{zx} & \sigma_{zy} & \sigma_{zz}
\end{bmatrix}.
$$

Because stress is also derived from the same energy function,

its prediction remains fully consistent with the predicted forces and energies.

---

# 18.7.22 Unified Prediction Framework

The remarkable feature of M3GNet is that **all major physical quantities originate from a single learned energy function**.

The relationships can be summarized as

```text id="m3gnet_math4"
Crystal Structure

↓

Graph Neural Network

↓

Total Energy

├── Gradient w.r.t. Atomic Positions

│         ↓

│      Atomic Forces

│

└── Gradient w.r.t. Cell Strain

          ↓

      Stress Tensor
```

Thus,

the network learns only one scalar function,

yet produces multiple physically meaningful outputs.

This unified framework is one of the defining characteristics of machine learning interatomic potentials and a major reason for the success of M3GNet in atomistic simulations.

---

### Transition to the Next Part

We have now established the mathematical foundation of M3GNet, including graph representation, message passing, energy prediction, force calculation, and stress prediction.

In the final part of **Section 18.7**, we will examine the **loss functions**, **multi-task training objective**, and **backpropagation equations** that enable M3GNet to learn accurate energies, forces, and stresses simultaneously from first-principles datasets. This completes the theoretical formulation of the model before moving on to its practical PyTorch implementation.

Continuing **Section 18.7**.

---

# 18.7.23 Training Objective

Up to this point, we have assumed that the neural network can predict the correct energy, forces, and stresses.

However, when training begins, the model knows nothing about materials.

Its parameters (weights and biases) are initialized randomly.

Consequently,

its initial predictions are essentially random.

For example,

suppose the true energy of a crystal is

$$
-128.63\ \text{eV}.
$$

At the beginning of training,

the network might predict

$$
52.84\ \text{eV}.
$$

The prediction is clearly incorrect.

The purpose of training is therefore to gradually adjust the neural network parameters until the predicted quantities closely match the reference values obtained from Density Functional Theory (DFT).

To achieve this,

we require a mathematical measure of prediction error.

This measure is called the **loss function**.

---

# 18.7.24 The Loss Function

A loss function measures the disagreement between the neural network prediction and the reference data.

Smaller loss values indicate better predictions.

Suppose

* the predicted energy is

$$
E_{\text{pred}},
$$

* the reference DFT energy is

$$
E_{\text{true}}.
$$

The prediction error is

$$
E_{\text{pred}}
---------------

E_{\text{true}}.
$$

Rather than using the raw error,

machine learning usually minimizes the squared error,

which is always non-negative.

The energy loss becomes

$$
L_E
===

\left(
E_{\text{pred}}
---------------

E_{\text{true}}
\right)^2.
$$

The squared error has several desirable properties.

* Positive and negative errors contribute equally.
* Large errors are penalized more heavily.
* The function is smooth and differentiable.

These properties make optimization much easier.

---

# 18.7.25 Mean Squared Error

Training is performed on many crystal structures simultaneously.

Suppose the training dataset contains

$N$

structures.

The average energy loss is

$$
L_E
===

\frac{1}{N}
\sum_{n=1}^{N}
\left(
E_n^{\text{pred}}
-----------------

E_n^{\text{true}}
\right)^2.
$$

This is called the **Mean Squared Error (MSE)**.

The optimizer attempts to minimize this quantity throughout training.

---

# 18.7.26 Force Loss

Energy alone is not sufficient for learning an accurate interatomic potential.

Remember that forces contain valuable information about the local shape of the potential energy surface.

Each atom possesses three force components.

Suppose

$$
\mathbf{F}_i^{\text{pred}}
$$

is the predicted force,

and

$$
\mathbf{F}_i^{\text{true}}
$$

is the DFT force.

The force loss is

$$
L_F
===

\frac{1}{N_{\text{atoms}}}
\sum_i
\left|
\mathbf{F}_i^{\text{pred}}
--------------------------

\mathbf{F}_i^{\text{true}}
\right|^2.
$$

Here,

$$
|\cdot|
$$

denotes the Euclidean norm.

Because every atom contributes three force components,

force datasets often contain much more information than energy datasets.

For this reason,

including force supervision dramatically improves model accuracy.

---

# 18.7.27 Stress Loss

For crystalline materials,

stress information is also extremely valuable.

Suppose

$$
\boldsymbol{\sigma}^{\text{pred}}
$$

is the predicted stress tensor,

and

$$
\boldsymbol{\sigma}^{\text{true}}
$$

is the DFT stress tensor.

The stress loss becomes

$$
L_S
===

\left|
\boldsymbol{\sigma}^{\text{pred}}
---------------------------------

\boldsymbol{\sigma}^{\text{true}}
\right|^2.
$$

Including stress information improves

* lattice optimization,
* elastic property prediction,
* pressure-dependent simulations.

---

# 18.7.28 Multi-Task Learning

Unlike ordinary regression models,

M3GNet predicts multiple physical quantities simultaneously.

Therefore,

the overall loss combines several individual losses.

The total loss is

$$
L
=

w_E L_E
+
w_F L_F
+
w_S L_S,
$$

where

* $w_E$ is the energy weight,
* $w_F$ is the force weight,
* $w_S$ is the stress weight.

These weights determine the relative importance of each prediction task.

---

## Why Are Weights Necessary?

Energy,

forces,

and stresses have different physical units and numerical magnitudes.

For example,

```text id="m3gnet_loss1"
Energy Error

↓

0.05 eV

Force Error

↓

0.30 eV/Å

Stress Error

↓

2.5 GPa
```

If all losses were added directly,

the largest numerical quantity would dominate optimization.

Weighting ensures that all physical quantities contribute appropriately during training.

Choosing suitable weights is therefore an important hyperparameter optimization problem.

---

# 18.7.29 Backpropagation

After computing the loss,

the neural network must determine how to modify its parameters.

Suppose the complete set of trainable parameters is

$$
\Theta.
$$

Training seeks the parameter values that minimize the loss function.

This optimization relies on the gradient

$$
\frac{\partial L}
{\partial \Theta}.
$$

This derivative indicates

* which parameters should increase,
* which should decrease,
* and by how much.

The gradients are computed automatically using backpropagation.

Because every operation inside M3GNet is differentiable,

the chain rule propagates gradients from the loss all the way back to the embedding layer.

---

# 18.7.30 Gradient Descent

Once the gradients have been computed,

the parameters are updated.

The simplest optimization rule is **gradient descent**.

The parameter update is

$$
\Theta_{\text{new}}
===================

## \Theta_{\text{old}}

\eta
\frac{\partial L}
{\partial \Theta},
$$

where

* $\eta$ is the learning rate.

The negative sign is important.

It ensures that the parameters move in the direction that reduces the loss.

Graphically,

```text id="m3gnet_loss2"
Current Parameters

↓

Compute Loss

↓

Compute Gradients

↓

Update Parameters

↓

Lower Loss
```

This procedure is repeated thousands of times during training.

---

# 18.7.31 Why Adam Is Usually Preferred

Although gradient descent illustrates the basic idea,

modern deep learning rarely uses plain gradient descent.

Instead,

M3GNet is typically trained using the **Adam optimizer**.

Adam improves convergence by maintaining adaptive learning rates for each parameter individually.

Compared with ordinary gradient descent,

Adam

* converges faster,
* handles noisy gradients,
* requires less manual tuning,
* performs well on very large neural networks.

For these reasons,

Adam has become the standard optimizer for most graph neural networks in materials informatics.

---

# 18.7.32 Complete Mathematical Pipeline

We can now summarize the complete mathematical workflow of M3GNet.

```text id="m3gnet_loss3"
Crystal Structure

↓

Graph Construction

↓

Distance Encoding

↓

Angle Encoding

↓

Embedding Layer

↓

Interaction Blocks

↓

Atomic Energies

↓

Total Energy

↓

Forces & Stress

↓

Loss Function

↓

Backpropagation

↓

Parameter Update

↓

Repeat
```

Every iteration through this workflow is called a **training step**.

After many thousands of training steps,

the model gradually learns the relationship between atomic structure and material properties.

---

# 18.7.33 Key Mathematical Equations

The entire mathematical framework of M3GNet can be summarized by a small set of fundamental equations.

### Graph Construction

$$
G=(V,E,T)
$$

---

### Distance

$$
r_{ij}
======

\left|
\mathbf{R}_i-\mathbf{R}_j
\right|
$$

---

### Edge Encoding

$$
\mathbf{e}_{ij}
===============

\mathrm{RBF}(r_{ij})
$$

---

### Angle Encoding

$$
\mathbf{t}_{ijk}
================

\mathrm{ABF}(\theta_{ijk})
$$

---

### Message Passing

$$
\mathbf{m}_i
============

\sum_{j\in\mathcal N(i)}
\mathbf{m}_{ji}
$$

---

### Node Update

$$
\mathbf{v}_i^{(l+1)}
====================

\phi_v
(
\mathbf{v}_i^{(l)},
\mathbf{m}_i
)
$$

---

### Atomic Energy

$$
E_i
===

\phi_r
(
\mathbf{v}_i^{(L)}
)
$$

---

### Total Energy

$$
E
=

\sum_i
E_i
$$

---

### Force

$$
\mathbf{F}_i
============

*

\frac{\partial E}
{\partial \mathbf{R}_i}
$$

---

### Stress

$$
\boldsymbol{\sigma}
===================

\frac{1}{V}
\frac{\partial E}
{\partial \boldsymbol{\varepsilon}}
$$

---

### Total Loss

$$
L
=

w_E L_E
+
w_F L_F
+
w_S L_S
$$

These equations form the mathematical backbone of M3GNet. Nearly every implementation, whether in the original paper or in open-source libraries such as MatGL, follows these principles, even if specific architectural details differ.

---

## Transition to Chapter 18.8

We have now completed the theoretical development of M3GNet. We understand how crystal structures are converted into graphs, how distances and angles are encoded, how message passing updates atomic representations, how energies, forces, and stresses are predicted, and how the model is trained through gradient-based optimization.

The next chapter, **18.8 PyTorch Implementation of M3GNet**, shifts from theory to practice. We will build an M3GNet model step by step using **PyTorch**, **DGL/PyTorch Geometric**, and **MatGL**, covering dataset preparation, graph construction, model definition, training loops, inference, structure relaxation, and molecular dynamics simulations. This implementation section will bridge the gap between mathematical understanding and real-world materials informatics research.

# 18.8 PyTorch Implementation of M3GNet

Up to this point, we have studied the complete theoretical framework of M3GNet. We understand why three-body interactions are important, how graph representations are constructed, how message passing works, and how the network predicts energies, forces, and stresses. However, understanding the theory alone is not sufficient for conducting research in materials informatics.

A researcher must also be able to implement, train, evaluate, and modify these models in practice.

In this section, we bridge the gap between theory and implementation.

Unlike previous sections, which focused primarily on mathematical concepts, this part of the chapter is highly practical. We will learn how to use the official implementation of M3GNet, prepare datasets, construct crystal graphs, train neural networks, predict material properties, relax crystal structures, and perform molecular dynamics simulations.

Our implementation will use **PyTorch** as the deep learning framework and **MatGL** as the primary graph neural network library.

---

# 18.8.1 Why Use MatGL Instead of Implementing M3GNet From Scratch?

A natural question is:

> **If we already know the mathematical equations of M3GNet, why not implement everything from scratch?**

From an educational perspective, implementing a graph neural network from scratch is extremely valuable because it reveals how every operation works internally.

However, modern research rarely starts from a blank file.

Instead, researchers typically build upon well-tested open-source implementations.

There are several reasons for this.

First,

M3GNet is a sophisticated architecture consisting of

* graph construction,
* radial basis functions,
* angular basis functions,
* multiple interaction blocks,
* automatic differentiation,
* energy, force, and stress prediction,
* crystal relaxation algorithms,
* molecular dynamics interfaces.

Implementing every component correctly requires thousands of lines of code.

Second,

research implementations are carefully optimized for

* computational efficiency,
* GPU utilization,
* numerical stability,
* compatibility with large datasets.

Reproducing these optimizations from scratch is both time-consuming and error-prone.

Finally,

using the official implementation ensures that experimental results are reproducible and comparable with published literature.

For these reasons, most researchers use the official implementation as a starting point and modify it according to their research objectives.

---

# 18.8.2 The Evolution from M3GNet to MatGL

When M3GNet was first introduced, its official implementation was released as a standalone Python package.

Although highly successful, maintaining a single model separately became increasingly difficult as newer graph neural network architectures emerged.

To address this issue, the developers introduced **MatGL (Materials Graph Library)**.

Rather than supporting only one architecture, MatGL provides a unified framework for multiple graph neural network models developed for materials science.

Today, MatGL includes implementations of models such as

* M3GNet,
* MEGNet,
* TensorNet,
* SO3Net,
* and additional architectures introduced in recent research.

Consequently, MatGL has become the recommended platform for developing graph neural network models for crystalline materials.

Instead of downloading separate repositories for each architecture, researchers can access them through a single, consistent interface.

---

# 18.8.3 What Is MatGL?

MatGL is an open-source Python library specifically designed for machine learning applications in materials science.

Conceptually, it occupies a position similar to that of

* **torchvision** for computer vision,
* **Hugging Face Transformers** for natural language processing,

but for **materials graph neural networks**.

MatGL provides

* pretrained graph neural network models,
* graph construction utilities,
* dataset handling,
* training pipelines,
* structure optimization tools,
* molecular dynamics interfaces.

Instead of implementing these components repeatedly for every project, researchers can focus directly on scientific problems.

---

# 18.8.4 Software Stack

The complete software stack used throughout this chapter is shown below.

```text id="m3gnet_impl1"
Python

↓

PyTorch

↓

DGL

↓

MatGL

↓

M3GNet Model
```

Each layer serves a specific purpose.

### Python

Python provides the programming language used throughout the implementation.

---

### PyTorch

PyTorch supplies

* tensors,
* automatic differentiation,
* neural network layers,
* optimization algorithms,
* GPU computation.

Every trainable parameter inside M3GNet is ultimately represented as a PyTorch tensor.

---

### DGL

MatGL uses the **Deep Graph Library (DGL)** to represent graph structures efficiently.

DGL provides

* graph storage,
* message passing operations,
* neighborhood aggregation,
* efficient GPU execution.

Rather than writing graph algorithms manually, we use DGL's optimized implementation.

---

### MatGL

MatGL builds upon DGL and PyTorch by providing materials-specific graph neural network models.

It handles

* crystal graph construction,
* radial basis expansion,
* angular features,
* pretrained M3GNet models,
* relaxation algorithms,
* molecular dynamics simulations.

---

# 18.8.5 Installation Requirements

Before implementing M3GNet, we must install several Python packages.

The most important dependencies are

* Python 3.10 or newer,
* PyTorch,
* DGL,
* MatGL,
* pymatgen,
* ASE,
* NumPy,
* SciPy.

The installation can be performed using `pip`.

```bash
pip install torch torchvision torchaudio
```

Next, install DGL.

For CPU execution,

```bash
pip install dgl
```

For CUDA-enabled GPUs, consult the official DGL installation instructions because the command depends on the installed CUDA version.

Finally, install MatGL.

```bash
pip install matgl
```

The remaining scientific libraries can be installed using

```bash
pip install pymatgen ase numpy scipy pandas matplotlib
```

After installation, verify that all packages import correctly.

```python
import torch
import dgl
import matgl
import pymatgen
import ase

print("All libraries imported successfully!")
```

If no error messages appear, the software environment has been configured correctly.

---

# 18.8.6 Checking GPU Availability

Graph neural networks are computationally intensive.

Training on CPUs is possible but can be extremely slow for large datasets.

PyTorch provides a simple method for checking whether a CUDA-enabled GPU is available.

```python
import torch

print(torch.cuda.is_available())
```

If the output is

```text
True
```

PyTorch has detected a compatible NVIDIA GPU.

The device can then be selected as

```python
device = torch.device("cuda")
```

Otherwise,

the computation will run on the CPU.

```python
device = torch.device("cpu")
```

A common implementation pattern is

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)
```

This code automatically chooses the GPU whenever one is available.

---

# 18.8.7 Verifying the Installation

Before proceeding to datasets and graph construction, it is good practice to verify that the installed libraries work together.

The following script prints the installed versions.

```python
import torch
import dgl
import matgl
import pymatgen

print("PyTorch :", torch.__version__)
print("DGL      :", dgl.__version__)
print("MatGL    :", matgl.__version__)
print("Pymatgen :", pymatgen.__version__)
```

A successful output indicates that the Python environment is correctly configured for the remainder of the chapter.

If import errors occur at this stage, they should be resolved before continuing.

---

# 18.8.8 Directory Structure for the Project

As the implementation grows, maintaining an organized project structure becomes increasingly important.

A recommended directory layout is

```text
M3GNet_Project/

│

├── data/

│   ├── raw/

│   ├── processed/

│   └── graphs/

│

├── models/

│

├── checkpoints/

│

├── notebooks/

│

├── scripts/

│

├── results/

│

└── train.py
```

Each directory has a clear purpose.

* **data/** stores crystal structures and datasets.
* **processed/** contains cleaned or converted data.
* **graphs/** stores graph representations if preprocessing is performed.
* **models/** contains custom neural network definitions.
* **checkpoints/** stores trained model weights.
* **results/** stores predictions, plots, and evaluation metrics.
* **scripts/** contains reusable utility programs.
* **notebooks/** contains Jupyter notebooks for experimentation.

Maintaining such an organization greatly simplifies debugging, collaboration, and reproducibility.

---

# 18.8.9 Implementation Roadmap

Before writing any substantial code, it is useful to understand the complete workflow that we will build over the next sections.

```text id="m3gnet_impl2"
Install Libraries

↓

Load Crystal Structures

↓

Convert Structures to Graphs

↓

Create Dataset

↓

Create DataLoader

↓

Load M3GNet

↓

Train Model

↓

Evaluate Model

↓

Predict Properties

↓

Relax Structures

↓

Run Molecular Dynamics
```

This roadmap reflects the workflow used in modern materials informatics research.

Every subsequent section of this chapter will focus on one stage of this pipeline, explaining not only **how** to implement it but also **why** each step is necessary.

---

### Transition to Section 18.8.10

The software environment is now ready. The next step is to obtain real crystal structures and convert them into graph representations that M3GNet can process.

In **Section 18.8.10**, we will begin working with **Materials Project data**, learn how crystal structures are represented using `pymatgen.Structure`, and prepare datasets suitable for graph neural network training. This marks the beginning of the complete end-to-end implementation pipeline.

# 18.8.10 Working with Crystal Structure Data

Machine learning models cannot operate directly on CIF files or crystal structure objects. Before an M3GNet model can make predictions, the crystal structure must first be represented in a form that the neural network understands.

In this section, we begin with the most fundamental object in computational materials science—the crystal structure—and learn how to manipulate it using **pymatgen**.

Although we introduced `pymatgen` briefly in **Chapter 0**, here we focus specifically on the functionality required for graph neural networks.

By the end of this section, you will be able to

* load crystal structures,
* inspect lattice information,
* access atomic coordinates,
* understand periodic boundary conditions,
* prepare structures for graph construction.

---

# 18.8.11 The `Structure` Object

In `pymatgen`, nearly every crystalline material is represented by the `Structure` class.

Conceptually,

```text id="m3gnet_data1"
Crystal

↓

Structure Object

↓

Python Manipulation
```

The `Structure` object stores all essential crystallographic information, including

* lattice vectors,
* atomic species,
* fractional coordinates,
* Cartesian coordinates,
* periodic boundary conditions,
* symmetry information.

Rather than manually storing these quantities in separate arrays, everything is organized inside a single object.

Import the class using

```python id="m3gnet_code1"
from pymatgen.core import Structure
```

---

# 18.8.12 Loading a CIF File

One of the most common crystal file formats is the **Crystallographic Information File (CIF)**.

Suppose the file

```text
Si.cif
```

contains the crystal structure of silicon.

It can be loaded using

```python id="m3gnet_code2"
from pymatgen.core import Structure

structure = Structure.from_file("Si.cif")
```

The variable

```python
structure
```

now contains the complete crystal.

---

# 18.8.13 Displaying the Structure

Simply printing the object provides useful information.

```python id="m3gnet_code3"
print(structure)
```

A typical output is

```text id="m3gnet_data2"
Full Formula (Si2)

Reduced Formula: Si

abc : 5.431 5.431 5.431

angles : 90.0 90.0 90.0

Sites (2)

0 Si ...

1 Si ...
```

Several pieces of information immediately become available.

* Chemical formula
* Lattice constants
* Cell angles
* Number of atomic sites
* Atomic positions

Without writing any additional code, the crystal has already been parsed into a structured representation.

---

# 18.8.14 Accessing Lattice Information

Every crystal possesses three lattice vectors

$$
\mathbf{a},
\mathbf{b},
\mathbf{c}.
$$

These vectors define the repeating unit cell.

In `pymatgen`,

the lattice object is accessed as

```python id="m3gnet_code4"
lattice = structure.lattice
```

To print the lattice,

```python id="m3gnet_code5"
print(lattice)
```

The lattice constants are

```python id="m3gnet_code6"
print(lattice.a)
print(lattice.b)
print(lattice.c)
```

Similarly,

the lattice angles are

```python id="m3gnet_code7"
print(lattice.alpha)
print(lattice.beta)
print(lattice.gamma)
```

These quantities define the geometry of the unit cell.

---

# 18.8.15 Lattice Matrix

Internally,

the lattice is stored as a

$$
3\times3
$$

matrix.

It can be accessed using

```python id="m3gnet_code8"
print(lattice.matrix)
```

For a cubic crystal,

the output may resemble

```text id="m3gnet_data3"
[[5.431 0.000 0.000]

 [0.000 5.431 0.000]

 [0.000 0.000 5.431]]
```

Each row corresponds to one lattice vector.

Mathematically,

$$
L
=

\begin{bmatrix}
\mathbf a\
\mathbf b\
\mathbf c
\end{bmatrix}.
$$

This lattice matrix is extensively used during graph construction and periodic image calculations.

---

# 18.8.16 Accessing Atomic Sites

Each atom in the crystal is represented by a **Site** object.

The total number of atoms is

```python id="m3gnet_code9"
print(len(structure))
```

To access the first atom,

```python id="m3gnet_code10"
site = structure[0]

print(site)
```

The output may resemble

```text id="m3gnet_data4"
[0.000 0.000 0.000] Si
```

Each site stores

* element,
* coordinates,
* occupancy,
* oxidation state (if available).

---

# 18.8.17 Atomic Species

The chemical element of each atom is obtained by

```python id="m3gnet_code11"
for site in structure:
    print(site.specie)
```

Possible output

```text id="m3gnet_data5"
Si

Si
```

For more complex materials,

the output could be

```text id="m3gnet_data6"
Li

Fe

P

O
```

These chemical identities are later converted into node embeddings by M3GNet.

---

# 18.8.18 Cartesian Coordinates

Each atom possesses Cartesian coordinates measured in Ångström.

They can be accessed using

```python id="m3gnet_code12"
for site in structure:
    print(site.coords)
```

Example output

```text id="m3gnet_data7"
[0.000 0.000 0.000]

[1.357 1.357 1.357]
```

Mathematically,

each coordinate is

$$
\mathbf R_i
===========

(x_i,y_i,z_i).
$$

These coordinates are used to compute

* interatomic distances,
* bond vectors,
* forces.

---

# 18.8.19 Fractional Coordinates

Crystallography usually stores atomic positions using **fractional coordinates** rather than Cartesian coordinates.

They can be accessed using

```python id="m3gnet_code13"
for site in structure:
    print(site.frac_coords)
```

Example output

```text id="m3gnet_data8"
[0.00 0.00 0.00]

[0.25 0.25 0.25]
```

Fractional coordinates express the position relative to the lattice vectors.

If

$$
(u,v,w)
$$

are the fractional coordinates,

the Cartesian position is

$$
\mathbf R
=========

u\mathbf a
+
v\mathbf b
+
w\mathbf c.
$$

This representation naturally accommodates periodic boundary conditions.

---

# 18.8.20 Why Fractional Coordinates Are Important

Suppose an atom has fractional coordinates

$$
(1.10,;0.50,;0.20).
$$

Although the first coordinate exceeds

1,

the atom is **not outside the crystal**.

Instead,

periodicity implies

$$
1.10
\equiv
0.10.
$$

Graphically,

```text id="m3gnet_data9"
Unit Cell

0 ------------------ 1

               |

             1.10

↓

Wrap Around

↓

0.10
```

Fractional coordinates therefore provide a natural way to describe infinite periodic crystals.

This property becomes extremely important during graph construction because neighboring atoms may lie in adjacent periodic images rather than within the original unit cell.

---

## Transition to the Next Section

We can now load crystal structures, inspect lattice parameters, access atomic species, and obtain both Cartesian and fractional coordinates.

However, M3GNet does not operate directly on these coordinates. The next step is to identify neighboring atoms under **periodic boundary conditions**, compute interatomic distances, and construct the crystal graph that serves as the input to the neural network.

In **Section 18.8.21**, we will learn how `pymatgen` finds neighboring atoms efficiently and how these neighbor lists are transformed into the graph representation used by M3GNet.

# 18.8.21 Finding Neighboring Atoms

A crystal structure by itself is **not yet a graph**.

Although the `Structure` object contains

* atomic positions,
* lattice vectors,
* chemical species,

it does **not explicitly specify which atoms interact with one another**.

To construct a graph, we must first determine the neighboring atoms for every atom in the crystal.

This process is called **neighbor searching**.

Neighbor searching is one of the most important preprocessing steps in graph neural networks for materials science because the graph connectivity determines how information propagates during message passing.

---

# 18.8.22 What Is a Neighbor?

Intuitively, two atoms are considered neighbors if they are sufficiently close to interact chemically.

For example, consider a silicon crystal.

```text id="m3gnet_neighbor1"
      Si

      |

Si —— Si —— Si

      |

      Si
```

The central silicon atom interacts most strongly with the surrounding silicon atoms.

These surrounding atoms become its graph neighbors.

Notice that atoms farther away are generally ignored because their interactions become much weaker.

The definition of "neighbor" therefore depends on a chosen **cutoff radius**.

---

# 18.8.23 Cutoff Radius

Suppose the distance between atoms

$i$

and

$j$

is

$$
r_{ij}.
$$

A neighbor relationship is defined as

$$
r_{ij}<r_c,
$$

where

$r_c$

is the cutoff radius.

If this condition is satisfied,

an edge is created between the two atoms.

Otherwise,

no edge is created.

Graphically,

```text id="m3gnet_neighbor2"
Distance

↓

Less than Cutoff?

↓

Yes → Create Edge

↓

No → Ignore
```

Thus,

the cutoff radius determines the connectivity of the graph.

---

# 18.8.24 Choosing the Cutoff Radius

Selecting an appropriate cutoff radius is an important design decision.

If the cutoff is **too small**,

important atomic interactions may be omitted.

For example,

```text id="m3gnet_neighbor3"
Cutoff Too Small

Atom A —— Atom B

Atom C

Ignored
```

The resulting graph becomes disconnected,

preventing information from flowing correctly.

---

If the cutoff is **too large**,

many unnecessary neighbors are included.

```text id="m3gnet_neighbor4"
Atom

↓

Hundreds of Neighbors

↓

Very Large Graph

↓

Slow Training
```

Excessively large graphs

* increase memory consumption,
* slow message passing,
* increase training time,
* may even introduce unnecessary noise.

Therefore,

the cutoff should be chosen carefully.

Typical cutoff values range from

$$
4\text{ Å}
\quad\text{to}\quad
6\text{ Å},
$$

although the optimal value depends on the material system and the model architecture.

---

# 18.8.25 Neighbor Search in `pymatgen`

Fortunately,

`pymatgen` provides highly optimized algorithms for finding neighbors.

The simplest approach uses

```python
structure.get_neighbors()
```

Suppose we wish to find all neighbors of the first atom within

5 Å.

```python
from pymatgen.core import Structure

structure = Structure.from_file("Si.cif")

neighbors = structure.get_neighbors(
    structure[0],
    r=5.0
)
```

The variable

```python
neighbors
```

contains all neighboring atoms located within the specified cutoff radius.

---

# 18.8.26 Inspecting Neighbor Information

Each neighbor object stores much more than the neighboring atom itself.

For example,

```python
for neighbor in neighbors:
    print(neighbor)
```

A typical output contains

* neighboring atom,
* Cartesian coordinates,
* interatomic distance,
* periodic image information.

This rich information is exactly what graph neural networks require.

---

# 18.8.27 Accessing Neighbor Distances

The distance between the central atom and each neighbor can be printed as

```python
for neighbor in neighbors:
    print(neighbor.nn_distance)
```

Example output

```text
2.35

2.35

2.35

2.35
```

These distances become the input for the radial basis expansion discussed earlier in this chapter.

Rather than using the raw distances directly,

M3GNet converts them into higher-dimensional feature vectors.

---

# 18.8.28 Neighbor Species

We can also inspect the chemical identity of neighboring atoms.

```python
for neighbor in neighbors:
    print(neighbor.specie)
```

Example output

```text
Si

Si

Si

Si
```

For more complex crystals,

the output could be

```text
O

Fe

Li

O

P
```

These chemical identities later become node embeddings.

---

# 18.8.29 Periodic Images

One feature that distinguishes crystalline materials from ordinary graphs is **periodicity**.

Consider the following one-dimensional crystal.

```text id="m3gnet_neighbor5"
|----Unit Cell----|

A -------- B

|----Next Cell----|

A -------- B
```

Suppose atom

A

lies near the left boundary of the unit cell.

Its closest neighbor may actually lie inside the **adjacent periodic image**, not inside the original cell.

Without periodic boundary conditions,

these atoms would appear to be far apart.

However,

crystallographically,

they are immediate neighbors.

This is why ordinary Euclidean neighbor searches are insufficient for crystals.

---

# 18.8.30 Periodic Boundary Conditions (PBC)

Periodic boundary conditions assume that the unit cell repeats infinitely in all directions.

Graphically,

```text id="m3gnet_neighbor6"
← Infinite Crystal →

□ □ □ □ □

□ □ □ □ □

□ □ □ □ □

□ □ □ □ □
```

The simulation cell is therefore only one small window into an infinite crystal.

Neighbor searching must consider atoms located in neighboring periodic images.

Fortunately,

`pymatgen` handles this automatically.

When

```python
structure.get_neighbors()
```

is called,

the algorithm searches not only inside the original unit cell but also inside the necessary periodic images.

No additional code is required.

---

# 18.8.31 Why Periodic Images Matter

Consider a cubic crystal.

```text id="m3gnet_neighbor7"
+-------------------+

A                 B

+-------------------+
```

Ignoring periodicity,

the distance between

A

and

B

appears very large.

With periodic boundary conditions,

the crystal repeats.

```text id="m3gnet_neighbor8"
... A | B ...
```

The true nearest distance becomes very small.

If periodicity were ignored,

the graph would contain incorrect edges,

leading to inaccurate predictions.

Therefore,

correct handling of periodic images is essential for any crystal graph neural network.

---

# 18.8.32 Neighbor Lists

After searching every atom,

we obtain a **neighbor list**.

Conceptually,

```text
Atom 0

↓

Neighbors

1

4

7

12

Atom 1

↓

Neighbors

0

2

8

10

⋮
```

This neighbor list is one of the primary inputs used during graph construction.

Each neighbor relationship eventually becomes a graph edge.

---

# 18.8.33 Computational Complexity

A naïve neighbor search compares every atom with every other atom.

For

$N$

atoms,

this requires

$$
N^2
$$

distance calculations.

For large crystals,

this quickly becomes computationally expensive.

Modern libraries therefore employ highly optimized spatial search algorithms,

including

* cell lists,
* linked lists,
* k-d trees,
* spatial partitioning methods.

These algorithms reduce the computational cost dramatically,

allowing neighbor searches for crystals containing thousands of atoms.

Because `pymatgen` already implements these optimizations internally, users rarely need to implement neighbor-search algorithms themselves.

---

# 18.8.34 Summary of Neighbor Searching

The complete neighbor-search workflow can be summarized as

```text id="m3gnet_neighbor9"
Crystal Structure

↓

Choose Cutoff Radius

↓

Apply Periodic Boundary Conditions

↓

Find Neighboring Atoms

↓

Compute Distances

↓

Store Neighbor List

↓

Construct Graph Edges
```

This neighbor list forms the foundation of the crystal graph.

Without it,

the graph neural network would have no information about which atoms should exchange messages.

---

## Transition to the Next Section

We now know how neighboring atoms are identified under periodic boundary conditions. However, M3GNet requires more than just a list of neighbors. Each neighbor relationship must be converted into **graph nodes, graph edges, edge attributes, and triplet (line graph) information** that can be processed efficiently by the neural network.

In **Section 18.8.35**, we will learn how **MatGL automatically converts a `pymatgen.Structure` into a DGL graph**, examine the internal graph representation, and inspect the node, edge, and three-body features that serve as the direct input to M3GNet.

# 18.8.35 From Crystal Structure to Graph

Up to this point, we have loaded a crystal structure and identified neighboring atoms using periodic boundary conditions.

However, M3GNet still cannot process this information directly.

Deep learning frameworks such as PyTorch expect numerical tensors rather than crystallographic objects. Likewise, graph neural network libraries expect graph objects instead of neighbor lists.

Therefore, an additional conversion step is required.

Conceptually,

```text id="m3gnet_graph1"
Crystal Structure

↓

Neighbor Search

↓

Graph Construction

↓

Graph Neural Network
```

This graph construction stage is one of the defining features of graph neural networks. It transforms a physical crystal into a mathematical object that can be processed efficiently by modern deep learning algorithms.

---

# 18.8.36 Why We Cannot Feed a CIF File Directly to M3GNet

A CIF file is essentially a text document.

For example,

```text id="m3gnet_graph2"
data_Si

_cell_length_a 5.431

_cell_length_b 5.431

_cell_length_c 5.431

...

Si 0.000 0.000 0.000

Si 0.250 0.250 0.250
```

Although this format is ideal for storing crystallographic information, it contains

* text,
* keywords,
* formatting rules,
* metadata.

Neural networks cannot interpret such information directly.

Instead, every crystal must ultimately become a collection of numerical tensors.

M3GNet therefore converts

```text id="m3gnet_graph3"
CIF

↓

Structure

↓

Graph

↓

Tensors

↓

Neural Network
```

---

# 18.8.37 Graph Representation of a Crystal

Mathematically,

the crystal graph is written as

$$
G=(V,E),
$$

where

* $V$ represents the set of nodes,
* $E$ represents the set of edges.

For M3GNet,

each node corresponds to one atom.

Each edge corresponds to one neighboring atomic pair.

For example,

consider a simple crystal containing four atoms.

```text id="m3gnet_graph4"
Atom 0

Atom 1

Atom 2

Atom 3
```

After graph construction,

```text id="m3gnet_graph5"
0 —— 1

|\    |

| \   |

2 —— 3
```

Only neighboring atoms are connected.

---

# 18.8.38 Node Features

Each node stores information describing an atom.

Initially,

this information is quite simple.

Typical node features include

* atomic number,
* atomic type,
* optional oxidation state,
* optional magnetic information.

For silicon,

the node feature might initially contain only

```text id="m3gnet_graph6"
Atomic Number

↓

14
```

Internally,

this integer is immediately transformed into a learnable embedding vector,

as discussed earlier in this chapter.

Thus,

although the original node feature is very small,

the neural network eventually converts it into a high-dimensional representation.

---

# 18.8.39 Edge Features

Edges contain information describing neighboring atomic pairs.

The most important quantity is the interatomic distance,

$$
r_{ij}.
$$

Instead of storing only the distance,

M3GNet converts it into radial basis features.

Conceptually,

```text id="m3gnet_graph7"
Distance

↓

Radial Basis Expansion

↓

Edge Feature Vector
```

This transformation enables the neural network to learn smooth relationships between distance and atomic interactions.

---

# 18.8.40 Triplet Features

Unlike CGCNN,

M3GNet also requires three-body information.

Suppose atoms

$i$,

$j$,

and

$k$

form a bond angle.

```text id="m3gnet_graph8"
i

 \

  j —— k
```

The angle

$$
\theta_{ijk}
$$

becomes an additional feature.

This information is stored separately from the ordinary graph.

Conceptually,

M3GNet processes

```text id="m3gnet_graph9"
Atoms

↓

Bonds

↓

Bond Angles
```

This additional angular information is one of the reasons M3GNet achieves much higher accuracy than earlier graph neural networks.

---

# 18.8.41 DGL Graph Objects

MatGL represents graphs using the **Deep Graph Library (DGL)**.

Instead of manually creating adjacency matrices,

we use DGL graph objects.

Conceptually,

```text id="m3gnet_graph10"
Python Objects

↓

DGL Graph

↓

GPU Tensor Representation
```

The DGL graph efficiently stores

* node indices,
* edge indices,
* graph connectivity,
* graph features.

Furthermore,

DGL automatically performs message passing on GPUs,

making training significantly faster.

---

# 18.8.42 Creating a Graph Converter

MatGL provides built-in graph conversion utilities.

Depending on the installed version of MatGL, the exact API may differ slightly, but the general workflow remains the same: create a graph converter (or graph builder) and use it to transform a `Structure` object into a graph.

Conceptually,

```python
graph_converter = GraphConverter(...)
```

The converter stores information such as

* cutoff radius,
* element types,
* graph construction rules,
* three-body interaction settings.

Once initialized,

the same converter can process thousands of crystal structures.

> **Note:** The exact class name and constructor arguments may change between MatGL releases. Always consult the version-specific documentation when implementing production code.

---

# 18.8.43 Converting a Structure into a Graph

After the converter has been created,

graph generation becomes straightforward.

Conceptually,

```python
graph = graph_converter.convert(structure)
```

Internally,

this single command performs numerous operations.

It

1. reads atomic coordinates,
2. identifies neighboring atoms,
3. applies periodic boundary conditions,
4. computes interatomic distances,
5. constructs graph edges,
6. prepares angular information,
7. stores everything inside a DGL graph.

Although the conversion appears simple,

it replaces hundreds of lines of graph construction code.

---

# 18.8.44 What Happens Internally?

The graph conversion pipeline can be visualized as

```text id="m3gnet_graph11"
Structure

↓

Read Atomic Coordinates

↓

Periodic Neighbor Search

↓

Distance Calculation

↓

Edge Construction

↓

Angle Construction

↓

Node Features

↓

Edge Features

↓

DGL Graph
```

Every crystal in the training dataset passes through this pipeline before entering the neural network.

---

# 18.8.45 Inspecting the Graph

After graph construction,

we can inspect basic graph properties.

Typical information includes

* number of nodes,
* number of edges,
* node feature dimensions,
* edge feature dimensions.

For example,

conceptually,

```python
print(graph)
```

may display information similar to

```text
Graph(
  num_nodes=64,
  num_edges=384
)
```

This immediately tells us

* how many atoms are present,
* how many neighboring interactions were identified.

Large supercells naturally produce larger graphs.

---

# 18.8.46 Number of Nodes

The number of graph nodes is simply the number of atoms.

If a crystal contains

$$
N
$$

atoms,

then

$$
|V|=N.
$$

For example,

```text id="m3gnet_graph12"
Crystal

↓

32 Atoms

↓

32 Nodes
```

Unlike image neural networks,

graph neural networks naturally handle graphs containing different numbers of nodes.

One crystal may contain

8

atoms,

while another contains

400

atoms.

The same M3GNet model can process both.

---

# 18.8.47 Number of Edges

The number of edges depends on

* cutoff radius,
* crystal structure,
* atomic density.

If each atom has approximately

12

neighbors,

then

the graph contains roughly

$$
12N
$$

directed edges,

or approximately

$$
6N
$$

undirected neighbor relationships.

Consequently,

edge counts vary significantly between different materials.

Dense metallic crystals typically contain more edges than open framework materials.

---

# 18.8.48 Directed vs. Undirected Edges

Although we often draw crystal graphs as undirected,

most graph neural network libraries internally store **directed edges**.

Suppose atoms

$i$

and

$j$

are neighbors.

Conceptually,

we draw

```text
i —— j
```

Internally,

DGL stores

```text
i → j

j → i
```

This duplication allows each atom to independently receive messages from its neighbors during message passing.

Therefore,

if a crystal has

100

undirected neighbor relationships,

the graph usually stores

200

directed edges.

This representation greatly simplifies message-passing algorithms and is standard practice in modern graph neural network implementations.

---

## Transition to the Next Section

We have now transformed a crystal structure into a graph and understood how nodes, edges, and graph connectivity are represented inside DGL.

However, the graph still contains only **basic structural information**. Before the graph can be passed into M3GNet, additional numerical features—such as atomic embeddings, radial basis expansions, and angular basis expansions—must be attached to the nodes and edges.

In **Section 18.8.49**, we will examine these graph features in detail and learn how MatGL prepares the tensors that become the actual input to the M3GNet neural network.

# 18.8.49 Graph Features in M3GNet

Constructing the graph is only the first step.

At this stage, the graph contains

* nodes,
* edges,
* connectivity information.

Although this representation captures **which atoms are connected**, it does not yet provide sufficient numerical information for deep learning.

A neural network cannot perform computations directly on atom labels or graph connectivity.

Instead, every node and every edge must be represented by numerical feature vectors.

These numerical representations are called **graph features**.

Conceptually,

```text id="m3gnet_feature1"
Crystal

↓

Graph

↓

Node Features

↓

Edge Features

↓

Triplet Features

↓

Neural Network
```

The quality of these features has a tremendous influence on model accuracy.

In fact, one of the major improvements introduced by M3GNet is its rich geometric feature representation.

---

# 18.8.50 Types of Features Used in M3GNet

Unlike earlier graph neural networks, M3GNet simultaneously employs three different categories of features.

These are

* node features,
* edge features,
* triplet (three-body) features.

Graphically,

```text id="m3gnet_feature2"
Crystal Graph

│

├── Node Features

│

├── Edge Features

│

└── Triplet Features
```

Each category describes a different aspect of the crystal.

| Feature Type | Physical Meaning    | Example       |
| ------------ | ------------------- | ------------- |
| Node         | Individual atom     | Atomic number |
| Edge         | Pair interaction    | Bond length   |
| Triplet      | Angular interaction | Bond angle    |

Together, these features provide a much richer description of the atomic environment than pairwise distances alone.

---

# 18.8.51 Node Features

Each node corresponds to one atom.

Initially,

the simplest descriptor of an atom is its atomic number,

$$
Z_i.
$$

For example,

| Element | Atomic Number |
| ------- | ------------- |
| H       | 1             |
| C       | 6             |
| O       | 8             |
| Si      | 14            |
| Fe      | 26            |

However,

feeding atomic numbers directly into a neural network is not ideal.

Consider silicon and phosphorus.

Their atomic numbers differ by only one,

yet their chemical behavior differs significantly.

Similarly,

carbon (6) is not "halfway" between hydrogen (1) and oxygen (8).

Atomic numbers are identifiers—not continuous physical variables.

Therefore,

M3GNet first converts each atomic number into a learnable vector representation.

---

# 18.8.52 Atomic Embedding

The embedding layer functions similarly to word embeddings in natural language processing.

Instead of representing an atom using a single integer,

it is represented by a high-dimensional vector.

Mathematically,

$$
\mathbf{v}_i^{(0)}
==================

\mathrm{Embedding}(Z_i).
$$

Suppose the embedding dimension is

128.

Instead of

```text id="m3gnet_feature3"
Silicon

↓

14
```

the network stores something conceptually like

```text id="m3gnet_feature4"
Silicon

↓

[-0.81,

 0.42,

 ...

 1.37]
```

This vector is **not manually designed**.

It is learned automatically during training.

Atoms with similar chemical behavior gradually acquire similar embeddings.

For example,

during training,

the embeddings of

* Na and K,
* Cl and Br,
* Fe and Co,

often become close together in the learned feature space because they exhibit similar chemistry.

This is one of the most powerful aspects of representation learning.

---

# 18.8.53 Why Learn Embeddings?

Suppose we attempted to predict material properties using only atomic numbers.

The neural network would have to learn all chemical relationships directly from integer values.

Instead,

the embedding layer learns these relationships automatically.

Conceptually,

```text id="m3gnet_feature5"
Atomic Number

↓

Embedding Layer

↓

Chemical Representation

↓

Neural Network
```

The embedding therefore acts as a bridge between discrete chemistry and continuous machine learning.

---

# 18.8.54 Edge Features

Node features describe individual atoms.

However,

chemical properties depend not only on atoms themselves but also on how atoms interact.

These interactions are represented by edges.

The simplest edge feature is the interatomic distance,

$$
r_{ij}
======

\left|
\mathbf{R}_i
------------

\mathbf{R}_j
\right|.
$$

For example,

```text id="m3gnet_feature6"
Atom A

|

2.35 Å

|

Atom B
```

The distance

2.35 Å

contains physically meaningful information about bond strength.

Shorter bonds generally correspond to stronger interactions,

although the relationship is highly nonlinear.

---

# 18.8.55 Why Distances Cannot Be Used Directly

At first glance,

it may seem sufficient to use

$$
r_{ij}
$$

as the edge feature.

However,

this approach has several limitations.

Suppose the bond length changes from

2.35 Å

to

2.36 Å.

Physically,

the interaction changes only slightly.

The neural network should therefore produce only a small change in its prediction.

Unfortunately,

raw distances do not always provide an ideal representation for learning such smooth relationships.

Instead,

modern graph neural networks transform distances into **continuous basis expansions**.

---

# 18.8.56 Radial Basis Expansion

M3GNet expands every distance into multiple radial basis functions.

Instead of one scalar,

the network receives a feature vector.

Mathematically,

$$
\mathbf{e}_{ij}
===============

\mathrm{RBF}(r_{ij}).
$$

Suppose

32

radial basis functions are used.

Then,

instead of

```text id="m3gnet_feature7"
2.35
```

the edge becomes

```text id="m3gnet_feature8"
[0.02,

0.15,

0.91,

...

0.01]
```

Each component measures how strongly the distance activates one basis function.

This transformation allows the neural network to learn much smoother distance-dependent interactions.

---

# 18.8.57 Gaussian Radial Basis Functions

One common choice is the Gaussian radial basis function.

Its mathematical form is

$$
\phi_n(r)
=========

\exp
\left[
------

\beta
(r-\mu_n)^2
\right],
$$

where

* $\mu_n$ is the center of the Gaussian,
* $\beta$ controls its width.

Each Gaussian is centered at a different distance.

For example,

```text id="m3gnet_feature9"
Center

↓

1.0 Å

↓

2.0 Å

↓

3.0 Å

↓

4.0 Å
```

When a bond length is close to one of these centers,

that basis function produces a large value.

When it is far away,

the value becomes very small.

Together,

the collection of Gaussian functions provides a smooth numerical description of bond length.

---

# 18.8.58 Why Basis Functions Improve Learning

Consider two bond lengths,

2.35 Å

and

2.36 Å.

If raw distances are used,

the neural network receives

```text id="m3gnet_feature10"
2.35

↓

2.36
```

The numerical difference is small, but the model must learn the nonlinear physics directly from these values.

After radial basis expansion,

both distances activate nearly the same set of Gaussian functions.

Their feature vectors become very similar.

Consequently,

small geometric changes produce small feature changes,

making optimization much more stable.

This smoothness is one reason radial basis expansions have become standard in modern atomistic neural networks.

---

# 18.8.59 Cutoff Function

Interactions should gradually vanish as atoms approach the cutoff radius.

If interactions were terminated abruptly,

the energy would become discontinuous,

which would lead to discontinuous forces.

To avoid this problem,

M3GNet multiplies radial basis functions by a **smooth cutoff function**.

Conceptually,

```text id="m3gnet_feature11"
Short Distance

↓

Large Weight

↓

Long Distance

↓

Small Weight

↓

Cutoff

↓

Zero
```

A common cosine cutoff function is

$$
f_c(r)=
\begin{cases}
\frac{1}{2}
\left[
\cos\left(\pi\frac{r}{r_c}\right)+1
\right],
&
r<r_c,
[1.2ex]
0,
&
r\ge r_c.
\end{cases}
$$

This function decreases smoothly to zero at the cutoff radius,

ensuring that both the predicted energy and its derivatives (forces) remain continuous.

---

## Transition to the Next Section

We have now examined **node features** and **edge features**, including atomic embeddings, radial basis expansions, and smooth cutoff functions.

However, M3GNet derives much of its power from a third category of features that most earlier graph neural networks ignored: **three-body (angular) features**. These features describe how neighboring bonds are oriented with respect to one another and are essential for accurately modeling covalent materials and complex crystal structures.

In **Section 18.8.60**, we will study angular basis functions, line graphs, and the construction of three-body features that enable M3GNet to capture bond-angle-dependent interactions.

# 18.8.60 Three-Body Features

One of the defining innovations of M3GNet is its ability to explicitly represent **three-body interactions**.

Earlier graph neural networks, such as CGCNN, primarily modeled pairwise interactions. In these models, information flows only along edges connecting two atoms.

However, many physical properties of materials cannot be accurately described using pairwise distances alone.

To understand why, consider two crystal structures with identical bond lengths but different bond angles.

Although the pairwise distances are identical, the total energies, electronic structures, and mechanical properties may differ substantially.

Therefore, a realistic atomistic potential must incorporate not only **who is connected**, but also **how those connections are arranged in space**.

---

# 18.8.61 A Simple Example

Consider three atoms.

```text id="m3gnet_angle1"
A

 \

  \

   B ------ C
```

The important geometric quantity is the angle

$$
\angle ABC.
$$

Suppose

* the bond length

$$
AB=2.0\ \text{Å},
$$

* the bond length

$$
BC=2.0\ \text{Å}.
$$

Now compare two situations.

### Structure 1

```text id="m3gnet_angle2"
A

|

|

B ------ C
```

The bond angle is

$$
90^\circ.
$$

---

### Structure 2

```text id="m3gnet_angle3"
A

\

 \

  B ------ C
```

The bond angle is

$$
120^\circ.
$$

Notice something important.

The two bond lengths are identical.

Only the angle changes.

Yet these two atomic environments are physically different.

Consequently,

their energies are different.

A graph neural network that only knows the bond lengths cannot distinguish these structures.

M3GNet solves this problem by explicitly including angular information.

---

# 18.8.62 Bond Angles

Suppose atom

$j$

is the central atom.

The vectors connecting the atoms are

$$
\mathbf{r}_{ji}
===============

\mathbf{R}_i-\mathbf{R}_j,
$$

and

$$
\mathbf{r}_{jk}
===============

\mathbf{R}_k-\mathbf{R}_j.
$$

The bond angle is computed using the dot product,

$$
\theta_{ijk}
============

\cos^{-1}
\left(
\frac{
\mathbf{r}*{ji}
\cdot
\mathbf{r}*{jk}
}{
|\mathbf{r}*{ji}|
,
|\mathbf{r}*{jk}|
}
\right).
$$

This is the same equation used in classical vector geometry.

Thus,

every triplet of neighboring atoms naturally defines an angle.

---

# 18.8.63 Why Bond Angles Matter

Bond angles strongly influence the stability of many materials.

For example,

in diamond,

each carbon atom forms four tetrahedral bonds.

The bond angle is approximately

$$
109.5^\circ.
$$

In graphite,

carbon atoms form planar hexagonal sheets.

The bond angle is

$$
120^\circ.
$$

Both materials consist entirely of carbon atoms.

Their bond lengths are also similar.

Nevertheless,

their

* hardness,
* electrical conductivity,
* density,
* crystal symmetry,

are completely different.

The principal difference lies in the angular arrangement of atoms.

This example illustrates why angular information is essential.

---

# 18.8.64 Three-Body Interactions

A **two-body interaction** depends only on

$$
r_{ij}.
$$

Mathematically,

$$
E
=

E(r_{ij}).
$$

A **three-body interaction** depends on

$$
r_{ij},
\quad
r_{jk},
\quad
\theta_{ijk}.
$$

Therefore,

$$
E
=

E
(
r_{ij},
r_{jk},
\theta_{ijk}
).
$$

Notice that the angle appears explicitly.

This additional information allows the potential energy surface to distinguish between atomic environments that would otherwise appear identical.

---

# 18.8.65 Enumerating Triplets

During graph construction,

MatGL searches for all valid atomic triplets.

Suppose atom

$j$

has four neighbors.

```text id="m3gnet_angle4"
      A

      |

D ---- B ---- C

      |

      E
```

The possible bond-angle triplets include

* A–B–C
* A–B–D
* A–B–E
* C–B–D
* C–B–E
* D–B–E

Each triplet contributes one angular interaction.

Thus,

the number of triplets grows much faster than the number of edges.

---

# 18.8.66 From Bond Graph to Line Graph

Representing thousands of triplets directly would be computationally inefficient.

Instead,

M3GNet introduces the concept of a **line graph**.

A line graph is not constructed from atoms.

Instead,

it is constructed from **bonds**.

The key idea is simple.

* In the original graph,

  * nodes are atoms,
  * edges are bonds.

* In the line graph,

  * nodes become bonds,
  * edges connect neighboring bonds.

Graphically,

Original graph

```text id="m3gnet_angle5"
Atom

↓

Bond

↓

Atom
```

becomes

Line graph

```text id="m3gnet_angle6"
Bond

↓

Bond

↓

Bond
```

This transformation allows angular information to be processed using ordinary message-passing algorithms.

---

# 18.8.67 Why the Line Graph Is Useful

Suppose the original graph contains

```text id="m3gnet_angle7"
A —— B —— C
```

There are two bonds,

$$
AB
\quad\text{and}\quad
BC.
$$

These bonds share the common atom

$$
B.
$$

Therefore,

the line graph contains

```text id="m3gnet_angle8"
AB

↓

BC
```

The connection between these bond-nodes represents the angle

$$
\theta_{ABC}.
$$

Consequently,

the line graph naturally encodes three-body interactions.

Instead of creating a specialized neural network for bond angles,

M3GNet simply performs message passing on a second graph.

This elegant design is one of the major architectural strengths of the model.

---

# 18.8.68 Angular Features

Each connection in the line graph stores angular information.

Instead of using the raw angle

$$
\theta,
$$

M3GNet expands it into an angular feature vector.

Conceptually,

```text id="m3gnet_angle9"
Bond Angle

↓

Angular Basis Expansion

↓

Feature Vector
```

This process is analogous to the radial basis expansion applied to bond lengths.

The resulting feature vector provides a smooth numerical representation of the angle.

---

# 18.8.69 Angular Basis Functions

Suppose the bond angle is

$$
\theta.
$$

Rather than using this scalar directly,

the model computes

$$
\mathbf{t}_{ijk}
================

\mathrm{ABF}(\theta_{ijk}),
$$

where

$$
\mathrm{ABF}
$$

denotes the angular basis function.

The output is a high-dimensional feature vector,

for example,

```text id="m3gnet_angle10"
[0.08,

0.44,

0.93,

...

0.12]
```

This vector is passed to the interaction blocks together with the radial edge features.

Thus,

every message exchanged between neighboring atoms depends not only on distance but also on local angular geometry.

---

# 18.8.70 Combined Geometric Representation

After graph construction,

every local atomic environment is described by three complementary components.

```text id="m3gnet_angle11"
Atom

↓

Atomic Embedding

↓

Neighbor Distance

↓

Radial Basis Features

↓

Bond Angle

↓

Angular Basis Features
```

Together,

these features capture

* atomic identity,
* pairwise geometry,
* angular geometry.

This rich representation enables M3GNet to model complex materials with significantly higher accuracy than graph neural networks based solely on pairwise interactions.

---

## Transition to the Next Section

We now understand how MatGL constructs node features, edge features, and three-body angular features. These tensors constitute the complete input to the M3GNet architecture.

The next step is to examine **how these graph features are passed through the neural network during the forward pass**. In **Section 18.8.71**, we will load a pretrained M3GNet model, inspect its architecture in PyTorch, and trace the complete flow of data from graph input to energy, force, and stress predictions.

# 18.8.71 Loading a Pretrained M3GNet Model

Constructing crystal graphs is only the first stage of the prediction pipeline.

The next step is to pass these graphs through the M3GNet neural network.

Fortunately, we do not always need to train a model from scratch.

The developers of MatGL provide **pretrained M3GNet models** that have already learned from millions of Density Functional Theory (DFT) calculations.

These pretrained models can immediately predict

* total energy,
* atomic forces,
* stress tensors,

for a wide variety of crystalline materials.

Using pretrained models offers several advantages.

* No expensive training is required.
* Predictions can be obtained within seconds.
* The models have already learned physically meaningful atomic representations.
* They provide an excellent starting point for transfer learning and fine-tuning.

In practical materials informatics research, pretrained models are often the first choice for rapid property prediction and structure optimization.

---

# 18.8.72 What Is a Pretrained Model?

A pretrained model is a neural network whose parameters have already been optimized using a large training dataset.

Recall from Section **18.7** that training involves repeatedly updating millions of parameters,

$$
\Theta,
$$

until the loss function is minimized.

Instead of initializing these parameters randomly,

a pretrained model provides values that have already been learned.

Conceptually,

```text id="m3gnet_pretrained1"
Random Parameters

↓

Months of Training

↓

Optimized Parameters

↓

Saved Model
```

When we load the pretrained model,

we are loading these optimized parameters directly.

---

# 18.8.73 Why Pretrained Models Work

Consider how humans learn.

Suppose a student has already completed several years of undergraduate materials science.

When beginning graduate research,

the student does not start learning from elementary chemistry again.

Instead,

the student builds upon existing knowledge.

Pretrained neural networks behave similarly.

Instead of beginning with random weights,

they already possess extensive knowledge about

* chemical bonding,
* crystal geometry,
* atomic interactions,
* energy landscapes.

Fine-tuning such models on a smaller dataset is therefore much more efficient than training from scratch.

---

# 18.8.74 Loading the Model

MatGL provides a convenient interface for loading pretrained models.

A typical workflow is

```python id="m3gnet_code14"
import matgl

model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
```

The string identifies a pretrained model distributed with MatGL.

After executing this command,

the neural network architecture and its learned parameters are loaded into memory.

Depending on the installed version of MatGL, the available pretrained model names may differ slightly. Always consult the documentation for your specific release.

---

# 18.8.75 What Does `load_model()` Actually Do?

Although loading the model requires only one line of Python,

many operations occur internally.

```text id="m3gnet_pretrained2"
Locate Model

↓

Download (If Necessary)

↓

Read Configuration

↓

Build Neural Network

↓

Load Trained Parameters

↓

Ready for Prediction
```

Specifically,

MatGL

1. identifies the requested model,
2. downloads it if it is not already cached,
3. reconstructs the network architecture,
4. loads the trained parameter values,
5. switches the model into inference mode (or prepares it for evaluation).

Thus,

a complex neural network containing millions of parameters becomes available through a single function call.

---

# 18.8.76 Understanding the Model Object

The returned object is an ordinary PyTorch neural network.

Therefore,

it behaves similarly to any other PyTorch model.

For example,

```python id="m3gnet_code15"
print(model)
```

prints the network architecture.

A simplified output might resemble

```text id="m3gnet_pretrained3"
M3GNet

├── Embedding Layer

├── Graph Convolution Blocks

├── Three-Body Interaction Blocks

├── Readout Network

└── Energy Head
```

The exact output depends on the MatGL version, but the essential components remain the same.

---

# 18.8.77 The Model Is a Computational Graph

It is useful to distinguish between two different meanings of the word **graph**.

The first graph is the **crystal graph**, which represents atoms and bonds.

The second graph is the **computational graph**, which represents mathematical operations inside the neural network.

Conceptually,

```text id="m3gnet_pretrained4"
Crystal

↓

Crystal Graph

↓

M3GNet

↓

Computational Graph

↓

Predictions
```

The crystal graph is the **input**.

The computational graph is the sequence of differentiable operations performed by PyTorch.

During backpropagation,

gradients flow through the computational graph—not the crystal graph itself.

Understanding this distinction helps avoid confusion when reading implementation code.

---

# 18.8.78 Switching Between Training and Evaluation Modes

PyTorch neural networks operate in two different modes.

* **Training mode**
* **Evaluation mode**

Training mode is activated using

```python id="m3gnet_code16"
model.train()
```

Evaluation mode is activated using

```python id="m3gnet_code17"
model.eval()
```

Although M3GNet does not rely heavily on layers such as dropout or batch normalization, it is still good practice to place the model in evaluation mode before making predictions.

This ensures that the network behaves consistently during inference.

---

# 18.8.79 Moving the Model to the GPU

Earlier,

we selected the computation device.

The model should also be transferred to the same device.

```python id="m3gnet_code18"
model = model.to(device)
```

If

```python
device = torch.device("cuda")
```

the entire neural network is moved to GPU memory.

If

```python
device = torch.device("cpu")
```

the model remains on the CPU.

An important rule in PyTorch is that

> **The model and all input tensors must reside on the same device.**

Otherwise,

PyTorch raises a runtime error.

---

# 18.8.80 Counting Trainable Parameters

One way to estimate the complexity of a neural network is to count the number of trainable parameters.

This can be done using

```python id="m3gnet_code19"
num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(num_parameters)
```

Here,

* `p.numel()` returns the number of elements in each parameter tensor.
* `requires_grad` ensures that only trainable parameters are counted.

Modern graph neural networks often contain millions of trainable parameters.

These parameters include

* embedding matrices,
* graph convolution weights,
* multilayer perceptron weights,
* readout layers.

The exact number depends on the chosen architecture and configuration.

---

# 18.8.81 Inspecting Individual Layers

PyTorch allows us to iterate through all modules in the network.

```python id="m3gnet_code20"
for name, module in model.named_modules():
    print(name, module)
```

This command reveals the hierarchical organization of the model.

Studying the layer structure is extremely helpful when

* modifying architectures,
* debugging implementations,
* implementing custom research ideas.

Rather than treating M3GNet as a "black box,"

researchers should become comfortable exploring its internal components.

---

# 18.8.82 Typical Forward Pipeline

Once the pretrained model has been loaded,

the complete inference pipeline is

```text id="m3gnet_pretrained5"
Structure

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

Triplet Features

↓

Embedding Layer

↓

Interaction Blocks

↓

Readout

↓

Predicted Energy

↓

Predicted Forces

↓

Predicted Stress
```

Everything after graph construction is performed automatically by the neural network.

The user only supplies the crystal structure.

---

# 18.8.83 Inference vs. Training

It is important to distinguish between **inference** and **training**.

During **training**,

the model

* predicts,
* computes the loss,
* calculates gradients,
* updates its parameters.

During **inference**,

the parameters remain fixed.

The model simply performs a forward pass to generate predictions.

Graphically,

```text id="m3gnet_pretrained6"
Training

Prediction

↓

Loss

↓

Backpropagation

↓

Parameter Update
```

versus

```text id="m3gnet_pretrained7"
Inference

Prediction

↓

Return Results
```

Most users of pretrained M3GNet models perform inference rather than training.

---

## Transition to the Next Section

We have now loaded a pretrained M3GNet model, examined its architecture, and prepared it for inference.

The next step is to **execute the forward pass**. In **Section 18.8.84**, we will feed crystal graphs into the model, trace how information flows through the embedding and interaction blocks, and obtain predictions for total energy, atomic forces, and stress tensors. This section will connect the theoretical architecture developed earlier in the chapter with the actual PyTorch implementation used in modern materials informatics research.

# 18.8.84 Performing a Forward Pass

The previous sections focused on preparing the input for M3GNet.

We have

* loaded crystal structures,
* converted them into graphs,
* generated node, edge, and angular features,
* loaded a pretrained M3GNet model.

Now we are ready to answer the most important practical question:

> **How does the model actually make a prediction?**

The process of passing input data through the neural network to obtain predictions is called the **forward pass**.

During the forward pass,

information flows through every layer of the network exactly once.

No parameters are updated.

No gradients are computed.

The network simply performs a sequence of mathematical operations to transform the input graph into predicted physical quantities.

---

# 18.8.85 The Complete Prediction Pipeline

Conceptually, the prediction process can be viewed as a sequence of transformations.

```text id="m3gnet_forward1"
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

Embedding Layer

↓

Graph Interaction Blocks

↓

Readout Layer

↓

Energy Prediction
```

Notice that the crystal itself never enters the neural network directly.

Only its graph representation is processed.

---

# 18.8.86 What Happens During the Forward Pass?

Suppose we denote the input graph by

$$
G.
$$

The neural network can be viewed as a mathematical function

$$
f_{\Theta},
$$

where

* $G$ is the input graph,
* $\Theta$ represents all trainable parameters.

The prediction is

$$
\hat{y}
=======

f_{\Theta}(G).
$$

Depending on the task,

$\hat{y}$ may represent

* total energy,
* atomic forces,
* stress tensor,
* another material property.

Although this equation appears simple,

the function

$$
f_{\Theta}
$$

contains millions of parameters and dozens of graph neural network operations.

---

# 18.8.87 Disabling Gradient Computation

When making predictions,

there is no need to compute gradients.

Gradient computation

* consumes memory,
* increases computation time.

PyTorch therefore provides the

```python
torch.no_grad()
```

context manager.

A typical inference workflow is

```python id="m3gnet_code21"
model.eval()

with torch.no_grad():
    prediction = model(graph)
```

This tells PyTorch that

* gradients are unnecessary,
* intermediate tensors need not be stored,
* inference should be optimized for speed.

For prediction tasks,

using `torch.no_grad()` is considered best practice.

---

# 18.8.88 Why Use `model.eval()`?

Earlier we discussed evaluation mode.

Before inference,

the model should always be switched to

```python
model.eval()
```

This ensures that any layers whose behavior differs between training and evaluation (such as dropout or batch normalization in other neural networks) operate correctly.

Although M3GNet relies less heavily on these layers than image classification networks,

calling

```python
model.eval()
```

remains the standard PyTorch workflow.

A typical inference template therefore becomes

```python id="m3gnet_code22"
model.eval()

with torch.no_grad():

    prediction = model(graph)
```

This small amount of code performs the complete forward propagation.

---

# 18.8.89 Input Tensors

Internally,

the graph contains several tensors.

Conceptually,

```text id="m3gnet_forward2"
Graph

│

├── Node Tensor

├── Edge Tensor

├── Angle Tensor

├── Connectivity

└── Lattice Information
```

During the forward pass,

all of these tensors are processed simultaneously.

Unlike ordinary neural networks,

graph neural networks do not process fixed-size matrices.

Instead,

they operate on graph structures whose connectivity may differ for every crystal.

---

# 18.8.90 The Embedding Layer

The first neural network layer encountered during the forward pass is usually the embedding layer.

Initially,

each atom is represented only by its atomic number.

For example,

```text id="m3gnet_forward3"
Atomic Number

↓

14
```

After the embedding layer,

the atom becomes

```text id="m3gnet_forward4"
Embedding Vector

↓

128 Numbers
```

Mathematically,

$$
\mathbf{h}_i^{(0)}
==================

\mathrm{Embedding}(Z_i).
$$

Here,

* $Z_i$ is the atomic number,
* $\mathbf{h}_i^{(0)}$ is the initial hidden representation of atom $i$.

Every subsequent interaction block updates this hidden representation.

---

# 18.8.91 Message Passing Begins

Once the initial node embeddings have been computed,

the graph neural network begins message passing.

For every edge,

information flows

from one atom

to

another atom.

Conceptually,

```text id="m3gnet_forward5"
Atom A

↓

Message

↓

Atom B
```

However,

the message depends on

* atomic embeddings,
* bond distance,
* bond angle,
* learnable neural network weights.

Thus,

messages are chemically informed rather than being simple numerical averages.

---

# 18.8.92 Computing Messages

Suppose

atoms

$i$

and

$j$

are neighbors.

The message generated along the edge can be written as

$$
\mathbf{m}_{ij}
===============

\phi
\left(
\mathbf{h}*i,
\mathbf{h}*j,
\mathbf{e}*{ij},
\mathbf{t}*{ijk}
\right),
$$

where

* $\mathbf{h}_i$ is the embedding of atom $i$,
* $\mathbf{h}_j$ is the embedding of atom $j$,
* $\mathbf{e}_{ij}$ represents radial edge features,
* $\mathbf{t}_{ijk}$ represents angular features,
* $\phi$ is a learnable neural network.

This equation illustrates an important idea:

The message exchanged between atoms depends not only on the atoms themselves but also on their geometric relationship.

---

# 18.8.93 Aggregating Messages

Each atom typically receives messages from many neighbors.

For example,

```text id="m3gnet_forward6"
Atom A

↙ ↓ ↘

Messages

↘ ↓ ↙

Central Atom
```

These incoming messages are aggregated into a single vector.

Mathematically,

$$
\mathbf{m}_i
============

\sum_{j\in\mathcal{N}(i)}
\mathbf{m}_{ij},
$$

where

$$
\mathcal{N}(i)
$$

denotes the neighbor set of atom

$i$.

The summation operation is permutation invariant,

meaning that changing the order of neighboring atoms does not alter the result.

This property is essential because atoms in a crystal have no intrinsic ordering.

---

# 18.8.94 Updating Atomic Representations

After aggregation,

the hidden representation of each atom is updated.

Conceptually,

```text id="m3gnet_forward7"
Old Embedding

↓

Incoming Messages

↓

Update Network

↓

New Embedding
```

Mathematically,

$$
\mathbf{h}_i^{(l+1)}
====================

\psi
\left(
\mathbf{h}_i^{(l)},
\mathbf{m}_i
\right),
$$

where

* $\psi$ is another neural network,
* $l$ denotes the interaction layer.

Each interaction block therefore refines the atomic representation using information from neighboring atoms.

---

# 18.8.95 Multiple Interaction Layers

The forward pass does not stop after a single message-passing step.

Instead,

M3GNet stacks several interaction blocks.

Conceptually,

```text id="m3gnet_forward8"
Initial Embedding

↓

Interaction Block 1

↓

Interaction Block 2

↓

Interaction Block 3

↓

...

↓

Final Atomic Representation
```

Each additional layer allows information to travel farther through the crystal.

For example,

after one interaction block,

an atom knows about its immediate neighbors.

After two interaction blocks,

it also receives information indirectly from neighbors of neighbors.

Thus,

deeper networks capture increasingly long-range structural information.

---

# 18.8.96 Hidden Representations Become Richer

As the forward pass progresses,

the hidden vectors gradually evolve.

Initially,

they mainly encode atomic identity.

After several interaction blocks,

they also encode

* local bonding,
* bond lengths,
* bond angles,
* chemical environment,
* coordination number,
* neighboring element types,
* medium-range structural information.

By the final interaction layer,

each atom possesses a learned representation that summarizes its local chemical environment.

These refined representations are then passed to the readout network, which converts atomic information into global material properties.

---

## Transition to Section 18.8.97

The interaction blocks produce highly informative atomic representations, but researchers are usually interested in **material-level properties** such as total energy rather than individual atomic embeddings.

In the next section, **18.8.97 Readout Layer and Property Prediction**, we will study how M3GNet combines atomic representations into predictions of total energy, atomic forces, and stress tensors, completing the forward-pass pipeline.

# 18.8.97 Readout Layer and Property Prediction

After passing through multiple interaction blocks, each atom possesses a highly informative hidden representation.

Initially, the hidden vector described only the identity of the atom.

After repeated message passing, however, it encodes

* the chemical identity of neighboring atoms,
* bond lengths,
* bond angles,
* local coordination,
* medium-range structural information,
* electronic environment learned during training.

At this stage, the neural network has extracted nearly all the information it needs from the crystal graph.

The remaining challenge is to convert these atomic representations into physically meaningful material properties.

This task is performed by the **readout layer**.

---

# 18.8.98 Why Is a Readout Layer Necessary?

Graph neural networks naturally operate at the **atomic level**.

After message passing, every atom has its own feature vector.

Suppose a crystal contains

$$
N
$$

atoms.

The output of the interaction blocks is

```text id="m3gnet_readout1"
Atom 1

↓

Embedding

Atom 2

↓

Embedding

⋮

Atom N

↓

Embedding
```

However,

most materials science problems require **crystal-level predictions** rather than atomic-level outputs.

For example,

we may wish to predict

* total energy,
* formation energy,
* band gap,
* elastic modulus.

Each crystal should produce **one prediction**, regardless of the number of atoms it contains.

The readout layer solves this problem.

---

# 18.8.99 Local Atomic Energy

Instead of predicting the total energy directly,

M3GNet predicts an **atomic energy contribution** for every atom.

Suppose

$$
\mathbf{h}_i
$$

is the final hidden representation of atom

$i$.

A small neural network predicts

$$
E_i
===

f_{\text{readout}}
(\mathbf{h}_i).
$$

Thus,

every atom contributes its own learned energy.

Conceptually,

```text id="m3gnet_readout2"
Atom

↓

Hidden Vector

↓

Readout Network

↓

Atomic Energy
```

This idea is inspired by classical interatomic potentials,

where the total energy is often written as a sum of local atomic contributions.

---

# 18.8.100 Total Energy Prediction

After computing atomic energies,

the crystal energy is obtained simply by summation.

Mathematically,

$$
E_{\text{total}}
================

\sum_{i=1}^{N}
E_i.
$$

Graphically,

```text id="m3gnet_readout3"
Atomic Energy 1

↓

Atomic Energy 2

↓

Atomic Energy 3

↓

⋮

↓

Atomic Energy N

↓

Summation

↓

Total Energy
```

This summation possesses an extremely important property.

The predicted energy is independent of the order in which atoms are listed.

Whether the atoms are indexed as

```text
1,2,3,...
```

or

```text
7,2,10,...
```

the total energy remains identical.

This property is called **permutation invariance**.

---

# 18.8.101 Why Summation Is Physically Meaningful

One might wonder why M3GNet predicts atomic energies rather than predicting the total energy directly.

There are several important reasons.

### Variable System Size

Different crystals contain different numbers of atoms.

A direct prediction network would struggle to accommodate arbitrary crystal sizes.

Summation naturally handles

* 8 atoms,
* 64 atoms,
* 512 atoms,

without changing the architecture.

---

### Extensivity

Many thermodynamic quantities are **extensive**.

If two identical crystals are combined,

their total energy approximately doubles.

The summation

$$
E_{\text{total}}
================

\sum_i E_i
$$

automatically satisfies this physical requirement.

---

### Local Physics

Atomic interactions are predominantly local.

Each atom contributes primarily according to its surrounding environment.

Predicting atomic energies therefore aligns naturally with the physics of condensed matter systems.

---

# 18.8.102 Predicting Atomic Forces

Energy prediction alone is insufficient for many applications.

For example,

* structure optimization,
* molecular dynamics,
* defect migration,
* diffusion studies,

all require **atomic forces**.

Fortunately,

once the energy function is differentiable,

forces follow directly from calculus.

The force acting on atom

$i$

is

$$
\mathbf{F}_i
============

*

\frac{\partial E}{\partial \mathbf{R}_i},
$$

where

* $E$ is the total energy,
* $\mathbf{R}_i$ is the atomic position.

This equation is identical to that used in Density Functional Theory.

The negative sign indicates that atoms move in the direction of decreasing energy.

---

# 18.8.103 Automatic Differentiation

Computing these derivatives manually would be extremely difficult.

Fortunately,

PyTorch automatically computes them using **automatic differentiation**.

Conceptually,

```text id="m3gnet_readout4"
Atomic Positions

↓

Neural Network

↓

Energy

↓

Automatic Differentiation

↓

Forces
```

Because every operation inside M3GNet is differentiable,

PyTorch can efficiently evaluate

$$
-\nabla E
$$

without requiring explicit derivative formulas for every layer.

This is one of the major advantages of deep learning frameworks.

---

# 18.8.104 Force Prediction Workflow

The complete force prediction process is

```text id="m3gnet_readout5"
Crystal Structure

↓

Graph

↓

M3GNet

↓

Total Energy

↓

Automatic Differentiation

↓

Atomic Forces
```

Notice that

**forces are not predicted independently.**

Instead,

they are obtained by differentiating the predicted energy.

This guarantees that

* energy,
* forces,

remain mathematically consistent.

Such consistency is essential for stable molecular dynamics simulations.

---

# 18.8.105 Predicting Stress Tensors

In addition to forces,

M3GNet predicts the **stress tensor**.

Stress measures the internal mechanical response of a material when the lattice is deformed.

The stress tensor is written as

$$
\boldsymbol{\sigma}
===================

\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} & \sigma_{xz} \
\sigma_{yx} & \sigma_{yy} & \sigma_{yz} \
\sigma_{zx} & \sigma_{zy} & \sigma_{zz}
\end{bmatrix}.
$$

Each component represents the force per unit area acting within the crystal.

Stress prediction is particularly important for

* geometry optimization,
* elastic constant calculations,
* pressure-dependent simulations,
* equation-of-state studies.

---

# 18.8.106 Stress from Energy Derivatives

Just as forces arise from derivatives with respect to atomic positions,

stress arises from derivatives with respect to lattice deformation.

Mathematically,

$$
\sigma_{\alpha\beta}
====================

\frac{1}{V}
\frac{\partial E}
{\partial \varepsilon_{\alpha\beta}},
$$

where

* $V$ is the cell volume,
* $\varepsilon_{\alpha\beta}$ is the strain tensor.

Thus,

M3GNet predicts stress using the same differentiable energy model.

Energy,

forces,

and stress therefore originate from a **single unified potential energy surface**.

---

# 18.8.107 Multiple Outputs from One Model

A remarkable feature of M3GNet is that one neural network simultaneously provides multiple physically related quantities.

Conceptually,

```text id="m3gnet_readout6"
Crystal Graph

↓

M3GNet

├── Total Energy

├── Atomic Forces

└── Stress Tensor
```

Instead of training separate models,

all predictions are generated from the same learned energy function.

This greatly improves physical consistency and computational efficiency.

---

# 18.8.108 Forward Pass Summary

The complete forward propagation can now be summarized as

```text id="m3gnet_readout7"
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

Embedding Layer

↓

Interaction Blocks

↓

Atomic Representations

↓

Readout Network

↓

Atomic Energies

↓

Summation

↓

Total Energy

↓

Automatic Differentiation

↓

Forces

↓

Stress
```

This pipeline represents the complete inference procedure of M3GNet.

Every prediction made by the model follows these steps.

---

# 18.8.109 A Complete Inference Example

The following conceptual code illustrates the complete workflow from crystal structure to prediction.

```python id="m3gnet_code23"
import torch
import matgl
from pymatgen.core import Structure

# Load pretrained model
model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model.eval()

# Load crystal structure
structure = Structure.from_file("Si.cif")

# Convert structure to graph
# (API may vary with MatGL version)
graph = graph_converter.convert(structure)

# Perform inference
with torch.no_grad():
    prediction = model(graph)

print(prediction)
```

This example omits some version-specific details (such as the exact graph converter API), but it illustrates the standard workflow used in practice.

---

## Transition to the Next Section

We now understand how M3GNet transforms a crystal graph into predictions of total energy, atomic forces, and stress tensors.

However, making predictions is only one aspect of using M3GNet. In research, we often need to **train or fine-tune** the model on new datasets—for example, to predict a specialized property or adapt the model to a new class of materials.

In **Section 18.8.110**, we will build a complete training pipeline, including dataset preparation, data loaders, loss functions, optimizers, training loops, validation, and checkpointing for M3GNet using PyTorch and MatGL.

# 18.8.110 Training M3GNet from Scratch and Fine-Tuning

Thus far, we have focused primarily on **using** a pretrained M3GNet model.

In practical research, however, there are many situations where a pretrained model is insufficient.

For example,

* you may have generated your own Density Functional Theory (DFT) dataset,
* you may wish to predict a new material property,
* you may be studying materials outside the domain of the pretrained model,
* you may want to improve prediction accuracy for a specialized class of materials.

In these situations, the model must be **trained** or **fine-tuned**.

Training is the process of adjusting the neural network parameters so that its predictions agree with reference data.

Mathematically,

training seeks to determine the optimal parameters

$$
\Theta^*
$$

that minimize a loss function,

$$
\Theta^*
========

\arg\min_{\Theta}
\mathcal{L}(\Theta).
$$

Here,

* $\Theta$ denotes all trainable parameters,
* $\mathcal{L}$ is the loss function.

Since M3GNet contains millions of parameters,

optimization is performed numerically using gradient-based methods.

---

# 18.8.111 Training vs. Fine-Tuning

Although the two procedures are closely related,

they are not identical.

## Training from Scratch

Training from scratch begins with randomly initialized parameters.

```text id="m3gnet_train1"
Random Parameters

↓

Training Dataset

↓

Millions of Updates

↓

Trained Model
```

This approach requires

* large datasets,
* significant computational resources,
* long training times.

It is generally used when no suitable pretrained model exists.

---

## Fine-Tuning

Fine-tuning begins with a pretrained model.

```text id="m3gnet_train2"
Pretrained Model

↓

Small Specialized Dataset

↓

Additional Training

↓

Improved Model
```

Only a relatively small number of updates are required because the network already possesses extensive chemical knowledge.

Fine-tuning is now the preferred strategy in many materials informatics applications.

---

# 18.8.112 Preparing the Dataset

Every supervised learning algorithm requires two components:

* input,
* target.

For M3GNet,

the input is a crystal structure,

while the targets may include

* total energy,
* atomic forces,
* stress tensor.

Conceptually,

```text id="m3gnet_train3"
Structure

↓

Graph

↓

Target Energy

↓

Target Forces

↓

Target Stress
```

Each training sample therefore consists of both structural information and reference physical quantities.

---

# 18.8.113 Dataset Organization

Suppose we have

1000

DFT calculations.

The dataset may be organized as

| Sample | Structure | Energy | Forces | Stress |
| ------ | --------- | ------ | ------ | ------ |
| 1      | CIF 1     | ✓      | ✓      | ✓      |
| 2      | CIF 2     | ✓      | ✓      | ✓      |
| ...    | ...       | ...    | ...    | ...    |
| 1000   | CIF1000   | ✓      | ✓      | ✓      |

Every row corresponds to one crystal.

The corresponding labels are usually obtained from first-principles calculations.

---

# 18.8.114 Splitting the Dataset

Machine learning models should never be evaluated on the same data used for training.

Instead,

the dataset is divided into three subsets.

```text id="m3gnet_train4"
Entire Dataset

↓

Training Set

↓

Validation Set

↓

Test Set
```

A common split is

* 80% training,
* 10% validation,
* 10% testing.

Each subset serves a different purpose.

---

### Training Set

Used for parameter optimization.

The model repeatedly updates its parameters using these samples.

---

### Validation Set

Used during training to monitor performance.

It helps

* tune hyperparameters,
* detect overfitting,
* select the best checkpoint.

---

### Test Set

Used only after training has finished.

It provides an unbiased estimate of the model's performance on unseen data.

The test set should **never** influence the training process.

---

# 18.8.115 Graph Conversion Before Training

Neural networks cannot train directly on crystal structures.

Every structure must first be converted into a graph.

Conceptually,

```text id="m3gnet_train5"
Crystal

↓

Graph Converter

↓

DGL Graph

↓

Training Sample
```

This conversion is usually performed

* before training,
* during data loading,
* or cached to disk for faster reuse.

For large datasets,

precomputing graphs often reduces overall training time.

---

# 18.8.116 Creating a Dataset Class

PyTorch organizes training data using dataset objects.

A custom dataset typically performs three tasks.

1. Load the crystal structure.
2. Convert it into a graph.
3. Return the corresponding target values.

Conceptually,

```python id="m3gnet_code24"
class CrystalDataset(Dataset):

    def __getitem__(self, index):

        structure = ...

        graph = ...

        target = ...

        return graph, target
```

The exact implementation depends on the data format and the MatGL version, but the overall logic remains the same.

Encapsulating data handling in a dataset class keeps the training code clean and reusable.

---

# 18.8.117 Using a DataLoader

Loading one crystal at a time would be inefficient.

PyTorch therefore provides the `DataLoader`, which automatically

* batches samples,
* shuffles the training data,
* loads data in parallel using multiple workers.

A typical setup is

```python id="m3gnet_code25"
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

Here,

* `batch_size=32` means that 32 graphs are processed together,
* `shuffle=True` randomizes the order of the training samples at the beginning of every epoch.

Shuffling helps prevent the model from learning spurious ordering patterns.

---

# 18.8.118 Why Mini-Batches?

Instead of processing the entire dataset at once,

modern deep learning uses **mini-batch training**.

Suppose the dataset contains

10,000

crystals.

Rather than computing gradients using all samples simultaneously,

the data are divided into smaller batches.

```text id="m3gnet_train6"
Dataset

↓

Batch 1

↓

Batch 2

↓

Batch 3

↓

⋮
```

Each batch produces one gradient update.

Mini-batch training

* reduces memory usage,
* accelerates optimization,
* improves GPU utilization,
* often leads to better convergence due to stochasticity.

---

# 18.8.119 Collating Variable-Sized Graphs

Unlike images,

crystals do not all contain the same number of atoms.

One graph may contain

16

nodes,

while another contains

128

nodes.

The DataLoader therefore cannot simply stack graphs into a fixed-size tensor.

Instead,

graph libraries such as DGL perform **graph batching**.

Conceptually,

```text id="m3gnet_train7"
Graph A

+

Graph B

+

Graph C

↓

Batched Graph
```

The batched graph preserves the boundaries between individual crystals while allowing all graphs to be processed efficiently in a single forward pass.

This is one of the major advantages of graph deep learning frameworks.

---

# 18.8.120 Preparing Targets

For each graph,

the corresponding target values must also be prepared.

Depending on the application,

the targets may include

* total energy,
* atomic forces,
* stress tensor,
* magnetic moments,
* other material properties.

These targets are converted into PyTorch tensors and moved to the same device as the model before training begins.

Ensuring that the graph data and target tensors reside on the same device (CPU or GPU) is essential for successful training.

---

## Transition to the Next Section

The training dataset is now fully prepared. We have organized the data, converted crystal structures into graphs, batched variable-sized graphs, and associated each graph with its target properties.

The next step is to define **how the model learns from these targets**. In **Section 18.8.121**, we will introduce the loss functions used by M3GNet for simultaneous prediction of energy, forces, and stress, explain why multi-task learning is beneficial, and derive the combined objective function optimized during training.

# 18.8.121 Loss Functions for M3GNet

Preparing the dataset is only one part of the training process.

The neural network must also know **how good or bad its predictions are**.

This is accomplished through a **loss function**.

The loss function measures the discrepancy between

* the predicted values,
* the reference values obtained from DFT calculations.

The objective of training is simple:

> **Minimize the loss function.**

As the loss decreases,

the predictions become increasingly accurate.

Mathematically,

training solves the optimization problem

$$
\Theta^*
========

\arg\min_{\Theta}
\mathcal{L}(\Theta),
$$

where

* $\Theta$ represents all trainable parameters,
* $\mathcal{L}$ is the loss function.

---

# 18.8.122 Why M3GNet Needs Multiple Losses

Unlike ordinary regression models,

M3GNet predicts multiple physical quantities simultaneously.

These include

* total energy,
* atomic forces,
* stress tensor.

Each prediction has its own error.

Therefore,

the model must optimize several objectives at the same time.

Conceptually,

```text id="m3gnet_loss1"
Predicted Energy

↓

Energy Error

Predicted Forces

↓

Force Error

Predicted Stress

↓

Stress Error

↓

Total Loss
```

This approach is called **multi-task learning**.

Instead of training separate neural networks,

one model learns several related tasks simultaneously.

---

# 18.8.123 Energy Loss

Suppose

$$
E^{\text{DFT}}
$$

is the reference energy,

and

$$
E^{\text{pred}}
$$

is the predicted energy.

The simplest energy loss is the Mean Squared Error (MSE),

$$
\mathcal{L}_E
=============

\frac{1}{N}
\sum_{i=1}^{N}
\left(
E_i^{\text{pred}}
-----------------

E_i^{\text{DFT}}
\right)^2.
$$

Here,

* $N$ is the number of training samples.

This loss penalizes large prediction errors much more strongly than small ones.

Consequently,

the network learns to reduce large mistakes rapidly.

---

# 18.8.124 Mean Absolute Error

Although MSE is common,

many materials science papers report the **Mean Absolute Error (MAE)**.

The MAE is

$$
\text{MAE}
==========

\frac{1}{N}
\sum_{i=1}^{N}
\left|
E_i^{\text{pred}}
-----------------

E_i^{\text{DFT}}
\right|.
$$

Unlike MSE,

the MAE grows linearly with the prediction error.

Therefore,

it is less sensitive to outliers.

It is important to distinguish between

* **training loss**, which may use MSE,
* **evaluation metric**, which is often MAE.

Many M3GNet studies optimize MSE during training while reporting MAE in their final results.

---

# 18.8.125 Force Loss

Forces are vector quantities.

Each atom has three force components,

$$
F_x,
\quad
F_y,
\quad
F_z.
$$

Suppose

$$
\mathbf{F}_i^{\text{pred}}
$$

is the predicted force,

and

$$
\mathbf{F}_i^{\text{DFT}}
$$

is the reference force.

The force loss is

$$
\mathcal{L}_F
=============

\frac{1}{3N_{\text{atoms}}}
\sum_i
\left|
\mathbf{F}_i^{\text{pred}}
--------------------------

\mathbf{F}_i^{\text{DFT}}
\right|^2.
$$

This expression averages the squared error across every force component.

Because forces contain much richer information than energies alone,

including force labels often improves the learned potential substantially.

---

# 18.8.126 Why Forces Are So Valuable

A single crystal provides only one total energy.

However,

if the crystal contains

100

atoms,

it also provides

300

force components.

Consequently,

force data supply a much richer training signal.

Graphically,

```text id="m3gnet_loss2"
One Structure

↓

1 Energy

+

300 Force Components

↓

Much More Information
```

For this reason,

modern machine-learned interatomic potentials almost always include force labels whenever they are available.

---

# 18.8.127 Stress Loss

The stress tensor contains nine components,

although symmetry often reduces the number of independent values.

Suppose

$$
\boldsymbol{\sigma}^{\text{pred}}
$$

and

$$
\boldsymbol{\sigma}^{\text{DFT}}
$$

denote the predicted and reference stress tensors.

The stress loss is

$$
\mathcal{L}_{\sigma}
====================

\frac{1}{9}
\left|
\boldsymbol{\sigma}^{\text{pred}}
---------------------------------

\boldsymbol{\sigma}^{\text{DFT}}
\right|^2.
$$

Including stress information enables the model to learn how the energy changes under lattice deformation.

This is particularly important for

* structural relaxation,
* equation-of-state calculations,
* pressure-dependent simulations.

---

# 18.8.128 Combined Loss Function

The total loss is formed by combining the individual losses.

Mathematically,

$$
\mathcal{L}
===========

w_E
\mathcal{L}*E
+
w_F
\mathcal{L}*F
+
w*{\sigma}
\mathcal{L}*{\sigma},
$$

where

* $w_E$ is the energy weight,
* $w_F$ is the force weight,
* $w_{\sigma}$ is the stress weight.

These coefficients control the relative importance of each prediction task.

---

# 18.8.129 Choosing the Loss Weights

Choosing appropriate loss weights is important because the numerical scales of the different targets are very different.

For example,

```text id="m3gnet_loss3"
Energy

↓

eV

Forces

↓

eV/Å

Stress

↓

GPa
```

If the losses were simply added together,

one quantity might dominate the optimization.

Therefore,

the weights are chosen so that

* energy,
* forces,
* stress,

all contribute meaningfully to the total objective.

The optimal weights depend on the dataset and the intended application.

---

# 18.8.130 Multi-Task Learning

Optimizing multiple physical quantities simultaneously is called **multi-task learning**.

Conceptually,

```text id="m3gnet_loss4"
Shared Neural Network

├── Energy

├── Forces

└── Stress
```

All three predictions share the same underlying atomic representations.

This offers several advantages.

* Better generalization.
* Improved data efficiency.
* Physically consistent predictions.
* Better transfer learning.

Because the three tasks are physically related,

learning one often improves the others.

---

# 18.8.131 Implementing the Loss in PyTorch

PyTorch provides built-in loss functions.

For example,

```python id="m3gnet_code26"
import torch.nn as nn

criterion = nn.MSELoss()
```

Energy loss can then be computed as

```python id="m3gnet_code27"
energy_loss = criterion(
    predicted_energy,
    target_energy
)
```

Similarly,

force and stress losses are computed independently.

Finally,

the total loss is assembled.

```python id="m3gnet_code28"
loss = (
    w_energy * energy_loss
    + w_force * force_loss
    + w_stress * stress_loss
)
```

This scalar value is the quantity minimized during training.

---

# 18.8.132 Monitoring the Loss

During optimization,

the loss is evaluated after every mini-batch.

A typical training curve looks like

```text id="m3gnet_loss5"
Loss

│\

│ \

│  \

│   \____

└────────────

Training Iterations
```

Initially,

the loss decreases rapidly.

Later,

the improvement becomes slower as the model approaches an optimum.

Monitoring both the training and validation losses is essential for detecting

* convergence,
* underfitting,
* overfitting.

---

# 18.8.133 When Does Training Stop?

Training usually continues until one of the following conditions is satisfied.

* The validation loss stops improving.
* A maximum number of epochs is reached.
* Early stopping is triggered.
* The learning rate becomes sufficiently small.

Stopping too early may produce an undertrained model,

whereas training for too long may lead to overfitting.

Selecting an appropriate stopping criterion is therefore an important practical aspect of model development.

---

## Transition to the Next Section

The loss function tells the neural network **what** should be minimized, but it does not explain **how** the parameters are updated.

The next step is to study the optimization algorithm itself. In **Section 18.8.134**, we will derive gradient descent, introduce the Adam optimizer used in most M3GNet implementations, explain backpropagation through graph neural networks, and build the complete PyTorch training loop for atomistic machine learning.

# 18.8.134 Optimization and Backpropagation

The loss function tells the neural network **how wrong** its predictions are.

However,

knowing the error alone is not sufficient.

The network must also determine **how to modify millions of parameters** so that future predictions become more accurate.

This is the role of the optimization algorithm.

Optimization is the process of adjusting the model parameters in order to reduce the loss function.

Mathematically,

training seeks

$$
\Theta^*
========

\arg\min_{\Theta}
\mathcal{L}(\Theta),
$$

where

* $\Theta$ denotes all trainable parameters,
* $\mathcal{L}$ is the total loss.

Finding the exact global minimum is generally impossible because modern neural networks contain millions of parameters.

Instead,

iterative optimization algorithms gradually improve the parameters over many training iterations.

---

# 18.8.135 The Optimization Cycle

Every training iteration follows the same sequence of operations.

```text id="m3gnet_opt1"
Input Graph

↓

Forward Pass

↓

Loss Computation

↓

Gradient Calculation

↓

Parameter Update

↓

Repeat
```

This cycle is repeated thousands or even millions of times until the model converges.

---

# 18.8.136 Gradient Descent

The simplest optimization algorithm is **Gradient Descent**.

Suppose the parameter vector is

$$
\Theta.
$$

The update rule is

$$
\Theta_{t+1}
============

## \Theta_t

\eta
\nabla_{\Theta}
\mathcal{L},
$$

where

* $\eta$ is the learning rate,
* $\nabla_{\Theta}\mathcal{L}$ is the gradient of the loss.

The gradient points in the direction of the steepest increase of the loss.

Therefore,

moving in the opposite direction reduces the loss.

---

# 18.8.137 Understanding the Gradient

Imagine standing on a mountain.

Your goal is to reach the valley.

The gradient tells you the direction of the steepest uphill climb.

To descend,

you simply walk in the opposite direction.

Graphically,

```text id="m3gnet_opt2"
       ▲

      / \

     /   \

    /     \

   ▼
```

Every optimization step moves the parameters slightly downhill on the loss surface.

Although neural network loss landscapes are vastly more complex than a simple mountain,

the underlying idea remains the same.

---

# 18.8.138 The Learning Rate

The learning rate,

$$
\eta,
$$

controls the size of each optimization step.

If the learning rate is too large,

the optimization may overshoot the minimum.

```text id="m3gnet_opt3"
Minimum

↓

Jump

↓

Jump

↓

Never Converges
```

If the learning rate is too small,

training becomes extremely slow.

```text id="m3gnet_opt4"
Tiny Steps

↓

Tiny Improvement

↓

Many Epochs
```

Therefore,

choosing an appropriate learning rate is one of the most important hyperparameters in deep learning.

Typical initial values are

```text id="m3gnet_opt5"
1e-3

or

5e-4

or

1e-4
```

The optimal value depends on the dataset and model architecture.

---

# 18.8.139 Why Ordinary Gradient Descent Is Rarely Used

Although gradient descent is easy to understand,

it has several disadvantages.

* Slow convergence.
* Sensitive to learning rate selection.
* Difficulty escaping flat regions.
* Inefficient on large datasets.

Modern deep learning therefore relies on more advanced optimization algorithms.

The most widely used optimizer for M3GNet is **Adam**.

---

# 18.8.140 The Adam Optimizer

Adam stands for

**Adaptive Moment Estimation**.

Instead of updating parameters using only the current gradient,

Adam also maintains moving averages of

* previous gradients,
* squared gradients.

These statistics allow Adam to adapt the learning rate individually for every parameter.

As a result,

Adam generally converges faster and more reliably than ordinary gradient descent.

Although the underlying equations are more complex,

the basic idea is intuitive:

> Parameters that consistently receive large gradients take smaller steps, while parameters with small gradients take larger steps.

---

# 18.8.141 Adam Update Equations

Adam computes the first moment estimate,

$$
m_t
===

\beta_1 m_{t-1}
+
(1-\beta_1)g_t,
$$

where

$$
g_t
===

\nabla_{\Theta}
\mathcal{L}.
$$

It also computes the second moment estimate,

$$
v_t
===

\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2.
$$

After bias correction,

the parameters are updated as

$$
\Theta_{t+1}
============

## \Theta_t

\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}.
$$

Here,

* $\beta_1$ controls momentum,
* $\beta_2$ controls adaptive scaling,
* $\epsilon$ prevents division by zero.

Although these equations may appear complicated,

PyTorch implements them automatically.

---

# 18.8.142 Creating an Adam Optimizer

Using Adam in PyTorch requires only a few lines of code.

```python id="m3gnet_code29"
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

Here,

* `model.parameters()` specifies the trainable parameters,
* `lr` defines the learning rate.

Other hyperparameters such as

* `betas`,
* `eps`,
* `weight_decay`

can also be customized,

although the default values are suitable for many applications.

---

# 18.8.143 Backpropagation

The optimizer cannot update parameters until it knows the gradient of the loss.

These gradients are computed using **backpropagation**.

Conceptually,

```text id="m3gnet_opt6"
Forward Pass

↓

Loss

↓

Backward Pass

↓

Gradients
```

During backpropagation,

the chain rule from calculus is applied repeatedly through every operation in the computational graph.

Since M3GNet is built entirely from differentiable operations,

PyTorch automatically computes these gradients.

---

# 18.8.144 Automatic Differentiation in PyTorch

Instead of deriving gradients manually,

PyTorch records every differentiable operation during the forward pass.

This creates a computational graph.

After computing the loss,

calling

```python id="m3gnet_code30"
loss.backward()
```

causes PyTorch to

1. traverse the computational graph in reverse,
2. apply the chain rule,
3. compute the gradient of every trainable parameter.

This automatic differentiation system is one of the core strengths of PyTorch.

---

# 18.8.145 Clearing Old Gradients

PyTorch accumulates gradients by default.

Therefore,

before computing new gradients,

the old ones must be cleared.

```python id="m3gnet_code31"
optimizer.zero_grad()
```

If this step is omitted,

gradients from multiple mini-batches accumulate,

leading to incorrect parameter updates.

A typical training iteration therefore begins with

```python id="08o6oq"
optimizer.zero_grad()
```

---

# 18.8.146 Updating Parameters

After computing gradients,

the optimizer updates every trainable parameter.

```python id="m3gnet_code32"
optimizer.step()
```

Internally,

Adam

* reads the gradients,
* computes adaptive updates,
* modifies every parameter tensor.

Once this step is complete,

the model has learned slightly from the current mini-batch.

---

# 18.8.147 One Complete Training Iteration

Putting everything together,

one optimization step follows this sequence.

```python id="m3gnet_code33"
optimizer.zero_grad()

prediction = model(graph)

loss = criterion(
    prediction,
    target
)

loss.backward()

optimizer.step()
```

Although only five lines of code are visible,

these commands execute millions of mathematical operations inside the neural network.

Every mini-batch processed during training follows this same pattern.

---

# 18.8.148 The Role of Epochs

One **epoch** corresponds to processing the entire training dataset once.

Suppose the dataset contains

10,000

structures,

and the batch size is

100.

Then,

one epoch consists of

$$
\frac{10000}{100}
=================

100
$$

mini-batch updates.

Training usually requires many epochs.

```text id="m3gnet_opt7"
Epoch 1

↓

Epoch 2

↓

Epoch 3

↓

⋮

↓

Epoch 200
```

As training progresses,

the loss generally decreases,

although the rate of improvement becomes slower near convergence.

---

# 18.8.149 Monitoring Optimization

During training,

it is common to record

* training loss,
* validation loss,
* learning rate.

A typical learning curve is

```text id="m3gnet_opt8"
Loss

│\

│ \

│  \

│   \__

│      \_

└──────────────

Epoch
```

If the training loss continues decreasing while the validation loss begins increasing,

the model is likely overfitting.

Monitoring these curves helps determine when training should stop.

---

## Transition to the Next Section

We now understand how M3GNet learns: the forward pass computes predictions, the loss measures prediction errors, backpropagation computes gradients, and the Adam optimizer updates the network parameters.

The final step is to combine these components into a **complete training pipeline**. In **Section 18.8.150**, we will implement the full PyTorch training loop, including batching, forward propagation, loss computation, backpropagation, validation, checkpoint saving, and learning rate scheduling exactly as performed in real M3GNet research workflows.

# 18.8.150 Complete M3GNet Training Pipeline

We have now developed nearly every component required to train an M3GNet model.

We understand

* crystal graph construction,
* atomic embeddings,
* radial features,
* three-body interactions,
* message passing,
* energy prediction,
* force calculation,
* stress calculation,
* multi-task loss functions,
* backpropagation,
* optimization.

The next task is to connect these ideas into a **complete training workflow**.

A simplified training pipeline can be represented as

```text
DFT Dataset
    ↓
Structures + Energies + Forces + Stresses
    ↓
Train / Validation / Test Split
    ↓
Graph Construction
    ↓
Mini-Batching
    ↓
M3GNet
    ↓
Energy Prediction
    ↓
Energy Derivatives
    ↓
Forces + Stress
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Parameter Update
    ↓
Validation
    ↓
Checkpoint
    ↓
Final Model
```

The important point is that these are not independent operations.

They form one continuous learning system.

---

# 18.8.151 What a Real Training Sample Contains

For an interatomic potential, one training sample is considerably richer than a conventional regression sample.

Suppose configuration $s$ contains $N_s$ atoms.

Its data may be written conceptually as

$$
\mathcal{D}_s
=============

\left{
\mathbf{R}_s,
\mathbf{L}_s,
\mathbf{Z}_s,
E_s,
\mathbf{F}_s,
\boldsymbol{\sigma}_s
\right},
$$

where

* $\mathbf{R}_s$ contains atomic coordinates,
* $\mathbf{L}_s$ contains lattice information,
* $\mathbf{Z}_s$ contains atomic identities,
* $E_s$ is the reference energy,
* $\mathbf{F}_s$ contains reference forces,
* $\boldsymbol{\sigma}_s$ contains reference stress.

Forces have shape approximately

$$
N_s\times3.
$$

Stress is generally represented by a $3\times3$ tensor or an equivalent six-component representation depending on convention and implementation.

Therefore, unlike ordinary tabular regression,

```text
features → target
```

the atomistic problem looks more like

```text
Variable-Sized Atomic Structure

        ↓

Energy + Forces + Stress
```

This difference is fundamental.

---

# 18.8.152 Where Training Data Usually Come From

For M3GNet research, reference labels generally come from quantum-mechanical calculations.

A typical workflow is

```text
Initial Structures

↓

DFT Calculations

↓

Energy

Forces

Stress

↓

Machine-Learning Dataset
```

For example, calculations performed using Quantum ESPRESSO, VASP, ABINIT, or another electronic-structure package can generate reference configurations.

If molecular dynamics or structural perturbations are performed,

a single material may produce hundreds or thousands of configurations.

The dataset might therefore contain

```text
Structure 1
Structure 2
Structure 3
...
Structure 500000
```

rather than simply one equilibrium structure for each compound.

This distinction is extremely important for interatomic potentials.

---

# 18.8.153 Why Equilibrium Structures Alone Are Not Enough

Suppose our dataset contains only fully relaxed structures.

At equilibrium,

atomic forces are approximately

$$
\mathbf{F}_i\approx0.
$$

Consequently, the training dataset tells the model primarily what the bottom of each energy basin looks like.

But consider molecular dynamics.

Atoms continuously move away from equilibrium.

The model must predict the energy landscape around equilibrium as well.

Therefore, useful training datasets contain

* equilibrium structures,
* distorted structures,
* strained structures,
* thermally displaced structures,
* high-energy configurations,
* different phases,
* different compositions.

Conceptually,

```text
            High Energy

               •

          •         •

       •               •

           \         /

            \       /

             \     /

              \   /

               \_/

            Equilibrium
```

Training only at the bottom of this landscape would provide very little information about its curvature or surrounding regions.

---

# 18.8.154 Dataset Diversity

Dataset size alone does not guarantee a good potential.

Consider two datasets.

### Dataset A

Contains

1,000,000

nearly identical silicon structures.

### Dataset B

Contains

100,000

structures spanning

* different compositions,
* different pressures,
* different temperatures,
* different coordination environments,
* different distortions.

For a universal model,

Dataset B may contain substantially more useful information despite being smaller.

The relevant concept is **coverage of configuration space**.

---

# 18.8.155 Structure-Level Data Splitting

We now split the dataset.

For example,

```python
from sklearn.model_selection import train_test_split

indices = list(range(len(structures)))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
)
```

This produces approximately

```text
80% Training

10% Validation

10% Testing
```

But researchers must be careful.

A purely random split can sometimes produce overly optimistic results.

---

# 18.8.156 Data Leakage in Atomistic ML

Suppose we perform molecular dynamics on one structure and obtain

```text
Frame 1
Frame 2
Frame 3
...
Frame 10000
```

Adjacent frames may be extremely similar.

If

```text
Frame 100
```

is placed in the training set,

while

```text
Frame 101
```

is placed in the test set,

the test structure is almost already represented in training.

The resulting test error may look excellent even though generalization is poor.

This is a form of **data leakage**.

---

# 18.8.157 Better Splitting Strategies

Depending on the research question, more rigorous splitting may be performed by

* composition,
* chemical system,
* prototype,
* trajectory,
* structure family,
* temperature,
* pressure,
* time block.

For example,

if we want to know whether the model generalizes to completely new compositions,

the split should be performed by composition rather than randomly by configuration.

The splitting strategy must therefore match the scientific question.

This principle extends far beyond M3GNet.

---

# 18.8.158 Energy Normalization

Raw total energies can vary enormously with system size.

Suppose one structure contains

10 atoms,

while another contains

100 atoms.

Their total energies may differ by hundreds or thousands of electron volts simply because one contains more atoms.

For model evaluation and sometimes preprocessing, it is therefore useful to consider energy per atom,

$$
\bar{E}
=======

\frac{E}{N}.
$$

However, one must distinguish carefully between

* how the training target is represented,
* how the model constructs an extensive energy,
* how errors are reported.

These choices depend on the training framework and scientific objective.

---

# 18.8.159 Elemental Energy References

An even more powerful strategy is to subtract elemental reference energies.

Suppose a structure contains elements indexed by $\alpha$.

We can express its energy schematically as

$$
E
=

\sum_{\alpha}
n_{\alpha}\epsilon_{\alpha}
+
\Delta E,
$$

where

* $n_{\alpha}$ is the number of atoms of element $\alpha$,
* $\epsilon_{\alpha}$ is a reference contribution,
* $\Delta E$ is the residual energy that the neural network must learn.

The model can then focus more strongly on chemically meaningful differences rather than spending capacity learning large elemental baseline contributions.

---

# 18.8.160 Why Energy Referencing Helps

Imagine two structures:

```text
Structure A

Total Energy = -8573 eV
```

and

```text
Structure B

Total Energy = -8575 eV
```

Their absolute energies are enormous,

but their difference is only

$$
2\ \text{eV}.
$$

Frequently, that difference is what matters physically.

Subtracting appropriate baselines can make the regression problem numerically better behaved.

---

# 18.8.161 Constructing Graphs

After preprocessing,

each structure must be transformed into the graph representation expected by the model.

Conceptually,

```text
pymatgen Structure

↓

Neighbor Search

↓

Periodic Edges

↓

Distance Calculation

↓

Three-Body Triplets

↓

Basis Expansion

↓

M3GNet Input
```

In a production workflow, this conversion should ideally be performed through the graph-building utilities provided by the specific MatGL version being used rather than by manually reimplementing periodic neighbor logic.

---

# 18.8.162 Why We Should Not Manually Build Everything

For learning purposes, manually constructing graphs is extremely useful.

It teaches us

* neighbor finding,
* periodic images,
* edge construction,
* triplet enumeration.

For actual research, however, a mature library implementation is usually preferable.

A small error involving periodic boundaries can silently corrupt the entire dataset.

For example,

```text
Atom

↓

Neighbor Across Cell Boundary
```

must be handled correctly.

Otherwise,

the model sees a physically incorrect atomic environment.

---

# 18.8.163 Dataset Objects

At a conceptual level, we can construct a dataset class like this:

```python
from torch.utils.data import Dataset


class M3GNetDataset(Dataset):

    def __init__(
        self,
        structures,
        energies,
        forces,
        stresses,
        graph_converter,
    ):
        self.structures = structures
        self.energies = energies
        self.forces = forces
        self.stresses = stresses
        self.graph_converter = graph_converter

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):

        structure = self.structures[idx]

        graph = self.graph_converter.convert(
            structure
        )

        labels = {
            "energy": self.energies[idx],
            "forces": self.forces[idx],
            "stress": self.stresses[idx],
        }

        return graph, labels
```

This code is intentionally conceptual because MatGL's concrete dataset/converter APIs can change between releases.

What matters here is understanding the architecture of the workflow.

---

# 18.8.164 Precomputing Graphs

Graph construction itself requires computational effort.

Suppose our dataset contains

500,000

structures.

If every graph is reconstructed during every epoch,

training becomes inefficient.

Instead,

we can compute graphs once.

```text
Structures

↓

Graph Construction

↓

Save Graphs

↓

Training Epoch 1

↓

Reuse

↓

Training Epoch 2

↓

Reuse

↓

...
```

This technique is known as **graph caching**.

It can substantially accelerate training.

---

# 18.8.165 Memory Trade-Off

Caching has a cost.

If the dataset is extremely large,

storing every graph in memory may require excessive RAM.

Therefore, researchers must choose between

```text
More Computation

vs.

More Memory
```

Possible strategies include

* caching in RAM,
* caching on disk,
* constructing graphs dynamically,
* hybrid caching.

The best solution depends on dataset size and available hardware.

---

# 18.8.166 Batching Crystal Graphs

Suppose a mini-batch contains four structures.

```text
Crystal A → 20 atoms

Crystal B → 45 atoms

Crystal C → 12 atoms

Crystal D → 68 atoms
```

They cannot be stacked like fixed-size images.

Instead, graph frameworks combine them into one disconnected graph.

Conceptually,

```text
Graph A     Graph B     Graph C     Graph D

   \           |           |          /

              Batch
```

There are no edges between the individual crystals.

The framework keeps track of which atoms belong to which graph.

---

# 18.8.167 Batched Readout

Suppose the model predicts atomic contributions

$$
E_i.
$$

For a batch containing multiple structures,

the readout operation must sum energies separately for every structure.

For structure $s$,

$$
E_s
===

\sum_{i\in s}
E_i.
$$

Therefore,

```text
Atoms 1–20

↓

Energy A


Atoms 21–65

↓

Energy B


Atoms 66–77

↓

Energy C
```

and so forth.

This allows structures of arbitrary size to coexist within one mini-batch.

---

# 18.8.168 A More Realistic Training Skeleton

We can now construct the conceptual core of the training loop.

```python
for epoch in range(num_epochs):

    model.train()

    for graph, labels in train_loader:

        optimizer.zero_grad()

        output = model(graph)

        energy_loss = energy_criterion(
            output["energy"],
            labels["energy"],
        )

        force_loss = force_criterion(
            output["forces"],
            labels["forces"],
        )

        stress_loss = stress_criterion(
            output["stress"],
            labels["stress"],
        )

        loss = (
            w_energy * energy_loss
            + w_force * force_loss
            + w_stress * stress_loss
        )

        loss.backward()

        optimizer.step()
```

Again, this illustrates the learning logic rather than promising exact drop-in compatibility with every MatGL release.

Now let us understand what each stage does.

---

# 18.8.169 Stage 1 — Enter Training Mode

At the beginning of every training epoch,

we call

```python
model.train()
```

This informs PyTorch that training is occurring.

---

# 18.8.170 Stage 2 — Obtain a Mini-Batch

The DataLoader provides

```python
graph, labels
```

where

`graph`

contains the batched crystal graphs,

while

`labels`

contains reference quantities.

Conceptually,

```text
Batch

├── Graphs
│
├── Energies
│
├── Forces
│
└── Stresses
```

---

# 18.8.171 Stage 3 — Clear Gradients

Before computing new gradients,

```python
optimizer.zero_grad()
```

is executed.

Otherwise,

PyTorch would accumulate gradients from previous iterations.

---

# 18.8.172 Stage 4 — Forward Propagation

Next,

```python
output = model(graph)
```

performs the forward pass.

Internally,

```text
Atomic Numbers

↓

Embeddings

↓

Radial Features

↓

Three-Body Features

↓

Message Passing

↓

Atomic Representations

↓

Readout

↓

Predictions
```

The output contains the physical quantities needed for loss calculation.

---

# 18.8.173 Stage 5 — Energy Loss

The predicted energy is compared with the reference energy.

```python
energy_loss = energy_criterion(
    output["energy"],
    labels["energy"],
)
```

Conceptually,

$$
E^{\text{pred}}
\longleftrightarrow
E^{\text{DFT}}.
$$

The difference contributes to the total loss.

---

# 18.8.174 Stage 6 — Force Loss

Likewise,

```python
force_loss = force_criterion(
    output["forces"],
    labels["forces"],
)
```

compares predicted and reference force components.

The force loss contains information about the **slope of the potential energy surface**.

This makes it extraordinarily valuable.

---

# 18.8.175 Stage 7 — Stress Loss

Similarly,

```python
stress_loss = stress_criterion(
    output["stress"],
    labels["stress"],
)
```

measures the error in the model's response to cell deformation.

Stress training becomes particularly important if the final potential will be used for

* variable-cell relaxation,
* high-pressure simulation,
* mechanical response,
* thermal expansion.

---

# 18.8.176 Stage 8 — Combined Loss

The losses are combined:

```python
loss = (
    w_energy * energy_loss
    + w_force * force_loss
    + w_stress * stress_loss
)
```

corresponding to

$$
\mathcal{L}
===========

w_E\mathcal{L}*E
+
w_F\mathcal{L}*F
+
w*{\sigma}\mathcal{L}*{\sigma}.
$$

This scalar summarizes the current model error.

---

# 18.8.177 Stage 9 — Backpropagation

Next,

```python
loss.backward()
```

calculates

$$
\frac{\partial\mathcal{L}}
{\partial\Theta}
$$

for all trainable parameters.

Gradients propagate backward through

```text
Loss

↓

Readout

↓

Message Passing

↓

Three-Body Interactions

↓

Edge Representations

↓

Atomic Embeddings
```

Consequently,

even the element embedding vectors themselves are optimized.

---

# 18.8.178 Stage 10 — Parameter Update

Finally,

```python
optimizer.step()
```

updates the parameters.

The model after this operation is slightly different from the model before it.

Repeated thousands of times,

these small changes gradually produce a useful interatomic potential.

---

# 18.8.179 Important Complication: Force Training Requires Derivatives

There is an important technical detail that must not be overlooked.

Earlier, we learned that

$$
\mathbf{F}_i
============

*

\frac{\partial E}
{\partial\mathbf{R}_i}.
$$

During training,

we want to minimize an error involving these forces.

Therefore, optimization may require derivatives of a quantity that is itself already a derivative of energy.

Schematically,

$$
\mathcal{L}_F
=============

\mathcal{L}
\left(
-\frac{\partial E}{\partial\mathbf{R}}
\right).
$$

Then parameter optimization requires

$$
\frac{\partial\mathcal{L}_F}
{\partial\Theta}.
$$

This involves differentiating through the force computation.

This is one reason force-field training is computationally more demanding than ordinary property regression.

---

# 18.8.180 Energy-Only Training vs Potential Training

This distinction is worth emphasizing.

Suppose we train a model only to predict formation energy.

Then the problem resembles ordinary supervised regression:

```text
Structure

↓

Energy
```

But training an interatomic potential involves

```text
Structure

↓

Differentiable Energy

↓

Forces

+

Stress
```

The latter places substantially stronger requirements on

* architecture,
* smoothness,
* differentiability,
* numerical stability.

A model with excellent energy MAE is therefore **not automatically a good interatomic potential**.

---

# 18.8.181 Validation Loop

After each training epoch,

we evaluate the model on the validation set.

Conceptually,

```python
model.eval()

validation_loss = 0.0

for graph, labels in val_loader:

    output = model(graph)

    # Compute the same required validation losses.
    # Details depend on whether forces/stresses
    # require autograd in the chosen implementation.

    validation_loss += ...
```

There is an important nuance here.

For ordinary energy-only regression, evaluation can usually be wrapped in

```python
torch.no_grad()
```

But if forces or stresses are being generated as derivatives of energy with respect to positions or strain, autograd may still be required during their evaluation.

Therefore, one should **not blindly disable gradient tracking for force/stress evaluation**.

The correct context depends on how the particular potential implementation computes these quantities.

---

# 18.8.182 Training Loss vs Validation Loss

Suppose after each epoch we obtain

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |          0.84 |            0.91 |
|    10 |          0.31 |            0.36 |
|    20 |          0.16 |            0.21 |
|    30 |          0.10 |            0.18 |
|    40 |          0.07 |            0.20 |
|    50 |          0.05 |            0.25 |

What happened?

Until approximately epoch 30,

both losses improved.

Afterward,

training loss continued decreasing,

but validation loss increased.

This is evidence of **overfitting**.

---

# 18.8.183 Early Stopping

Instead of blindly training for

500

epochs,

we can stop when validation performance stops improving.

Conceptually,

```text
Validation improves

↓

Save Model

↓

Validation improves

↓

Save Model

↓

No improvement

↓

Wait

↓

Still no improvement

↓

Stop Training
```

The waiting period is called **patience**.

For example,

```python
patience = 20
```

means that training may stop after 20 consecutive epochs without sufficient validation improvement.

---

# 18.8.184 Model Checkpointing

Whenever validation performance improves,

we save the model.

For example,

```python
torch.save(
    model.state_dict(),
    "best_m3gnet.pt",
)
```

This stores the parameter values.

Later,

they can be restored using

```python
model.load_state_dict(
    torch.load(
        "best_m3gnet.pt",
        weights_only=True,
    )
)
```

when supported by the installed PyTorch version.

Checkpointing ensures that we retain the best-performing model rather than simply the model from the final epoch.

---

# 18.8.185 Why Save the Best Model?

Suppose validation MAE behaves as

```text
Epoch 20 → 0.065

Epoch 30 → 0.051

Epoch 40 → 0.043

Epoch 50 → 0.047

Epoch 60 → 0.056
```

The best model occurred at epoch

40,

not epoch

60.

Without checkpointing,

we would accidentally keep the inferior model.

---

# 18.8.186 Learning Rate Scheduling

A constant learning rate is not always ideal.

At the beginning of training,

large steps are useful.

Near convergence,

smaller steps are preferable.

Conceptually,

```text
Early Training

Large Learning Rate

↓

Fast Progress


Late Training

Small Learning Rate

↓

Fine Adjustment
```

This motivates **learning rate scheduling**.

---

# 18.8.187 ReduceLROnPlateau

One useful PyTorch scheduler is

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
)
```

After validation,

we call

```python
scheduler.step(validation_loss)
```

If validation loss stops improving,

the learning rate decreases.

For example,

```text
1 × 10⁻³

↓

5 × 10⁻⁴

↓

2.5 × 10⁻⁴

↓

1.25 × 10⁻⁴
```

This allows increasingly fine optimization near convergence.

---

# 18.8.188 Gradient Clipping

Deep neural networks occasionally experience extremely large gradients.

This phenomenon can destabilize training.

One solution is **gradient clipping**.

For example,

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

This limits the overall gradient norm.

The training sequence becomes

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0,
)

optimizer.step()
```

Gradient clipping is not always necessary, but it can be useful when training becomes unstable.

---

# 18.8.189 Complete Research-Oriented Training Skeleton

We can now assemble a more complete conceptual implementation.

```python
import torch

num_epochs = 300
patience = 30

best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(num_epochs):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_train_loss = 0.0

    for graph, labels in train_loader:

        optimizer.zero_grad()

        output = model(graph)

        energy_loss = energy_criterion(
            output["energy"],
            labels["energy"],
        )

        force_loss = force_criterion(
            output["forces"],
            labels["forces"],
        )

        stress_loss = stress_criterion(
            output["stress"],
            labels["stress"],
        )

        loss = (
            w_energy * energy_loss
            + w_force * force_loss
            + w_stress * stress_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_train_loss += loss.item()

    train_loss = (
        running_train_loss
        / len(train_loader)
    )

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    running_val_loss = 0.0

    for graph, labels in val_loader:

        # Do not automatically use torch.no_grad()
        # if force/stress generation requires
        # derivatives with respect to coordinates
        # or strain.

        output = model(graph)

        energy_loss = energy_criterion(
            output["energy"],
            labels["energy"],
        )

        force_loss = force_criterion(
            output["forces"],
            labels["forces"],
        )

        stress_loss = stress_criterion(
            output["stress"],
            labels["stress"],
        )

        loss = (
            w_energy * energy_loss
            + w_force * force_loss
            + w_stress * stress_loss
        )

        running_val_loss += loss.item()

    val_loss = (
        running_val_loss
        / len(val_loader)
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler.step(val_loss)

    # -------------------------
    # Checkpoint
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_m3gnet.pt",
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    # -------------------------
    # Logging
    # -------------------------

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

    # -------------------------
    # Early stopping
    # -------------------------

    if epochs_without_improvement >= patience:

        print("Early stopping triggered.")

        break
```

This skeleton brings together nearly every training concept we have developed.

For an actual MatGL project, the library's current dataset, potential, trainer, graph conversion, and prediction APIs should be used where appropriate rather than assuming this conceptual dictionary-style output interface verbatim.

---

# 18.8.190 What Happens Inside One Epoch?

It is worth mentally tracing the entire operation.

Suppose the model sees a silicon structure.

### Step 1

Atomic numbers enter the embedding layer.

```text
Si

↓

Atomic Embedding
```

### Step 2

Neighbor distances are calculated.

```text
Si —— Si

2.35 Å
```

### Step 3

Distances are expanded into radial features.

### Step 4

Atomic triplets provide angular information.

```text
Si

 \

  Si —— Si
```

### Step 5

Three-body features are constructed.

### Step 6

Messages propagate between atoms.

### Step 7

Atomic hidden representations are updated.

### Step 8

Atomic energy contributions are predicted.

### Step 9

Atomic contributions are summed.

$$
E_{\text{pred}}
===============

\sum_i E_i.
$$

### Step 10

Energy derivatives produce forces.

$$
\mathbf{F}_i
============

*

\frac{\partial E}
{\partial\mathbf{R}_i}.
$$

### Step 11

Energy derivatives with respect to deformation provide stress according to the implementation's stress convention.

### Step 12

Predictions are compared against DFT labels.

### Step 13

The total loss is calculated.

### Step 14

Backpropagation calculates gradients.

### Step 15

Adam updates the network parameters.

Then another mini-batch enters.

This process repeats thousands of times.

---

# 18.8.191 Fine-Tuning a Pretrained M3GNet Model

Training from scratch is not always necessary.

Suppose we want a highly accurate potential specifically for

```text
Li–Fe–P–O
```

materials.

A pretrained universal model already understands many aspects of

* Li chemistry,
* Fe chemistry,
* P chemistry,
* O chemistry,
* bond lengths,
* coordination environments,
* angular relationships.

Throwing away this knowledge would be inefficient.

Instead,

we can **fine-tune** the pretrained network.

---

# 18.8.192 Fine-Tuning Workflow

The workflow becomes

```text
Universal Pretrained M3GNet

↓

Load Parameters

↓

Specialized Dataset

↓

Small Learning Rate

↓

Additional Training

↓

Domain-Specific Potential
```

This is transfer learning applied to atomistic machine learning.

---

# 18.8.193 Smaller Learning Rates for Fine-Tuning

Suppose training from scratch uses

```python
learning_rate = 1e-3
```

Fine-tuning might instead begin with something like

```python
learning_rate = 1e-4
```

or smaller, depending on the dataset and optimization behavior.

Why?

Because the pretrained parameters are already useful.

We do not want to destroy them with enormous updates.

Instead,

fine-tuning gently adjusts the learned representation toward the specialized domain.

---

# 18.8.194 Freezing Parameters

Sometimes,

we may choose to freeze part of the network.

For example,

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

would freeze everything.

We could then selectively unfreeze certain layers.

Conceptually,

```text
Atomic Embeddings

FROZEN

↓

Early Interaction Blocks

FROZEN

↓

Late Interaction Blocks

TRAINABLE

↓

Readout

TRAINABLE
```

Whether this strategy improves performance depends on the target task and dataset.

For potential fitting, full-network fine-tuning is often worth considering because geometric interaction representations themselves may need adaptation.

---

# 18.8.195 Catastrophic Forgetting

Fine-tuning introduces another phenomenon called **catastrophic forgetting**.

Suppose the universal model performs well on

100

chemical systems.

We fine-tune it exclusively on silicon.

Its silicon accuracy improves dramatically.

But its predictions for

* oxides,
* nitrides,
* metals,

may deteriorate.

The model has become specialized.

This is not necessarily a problem.

If our goal is

```text
Highly Accurate Silicon Potential
```

specialization is desirable.

But if we still want universality,

the fine-tuning dataset must remain diverse.

---

# 18.8.196 What Should Be Monitored During Training?

Loss alone is not enough.

A serious M3GNet training experiment should separately monitor quantities such as

$$
\mathrm{MAE}_E,
$$

$$
\mathrm{MAE}_F,
$$

and

$$
\mathrm{MAE}_{\sigma}.
$$

For example,

| Epoch | Energy MAE | Force MAE | Stress MAE |
| ----: | ---------: | --------: | ---------: |
|     1 |      0.142 |     0.481 |       1.83 |
|    20 |      0.061 |     0.193 |       0.91 |
|    50 |      0.039 |     0.112 |       0.62 |
|   100 |      0.032 |     0.086 |       0.55 |

The units and normalization must always be reported clearly.

---

# 18.8.197 Why Force MAE Deserves Special Attention

Imagine two potentials.

### Model A

Excellent energy prediction.

Poor force prediction.

### Model B

Slightly worse energy MAE.

Excellent force prediction.

If our goal is molecular dynamics,

Model B may be considerably more useful.

Why?

Because MD trajectories are generated from forces.

Newton's equation is

$$
m_i
\frac{d^2\mathbf{R}_i}{dt^2}
============================

\mathbf{F}_i.
$$

Errors in forces directly influence atomic motion.

---

# 18.8.198 Evaluating the Test Set

After training has completely finished,

we load the best checkpoint.

```python
model.load_state_dict(
    torch.load(
        "best_m3gnet.pt",
        weights_only=True,
    )
)

model.eval()
```

Then,

and only then,

we evaluate the test set.

The test set provides our final estimate of generalization performance.

It should not be repeatedly consulted while choosing hyperparameters.

---

# 18.8.199 Prediction Parity Analysis

One useful diagnostic is comparing predicted and reference quantities.

Ideally,

$$
y_{\text{pred}}
\approx
y_{\text{DFT}}.
$$

Therefore,

points should lie close to the parity line

$$
y=x.
$$

For energy prediction,

we compare

```text
DFT Energy

vs.

M3GNet Energy
```

For forces,

```text
DFT Force Components

vs.

M3GNet Force Components
```

Large systematic deviations from the parity line can reveal bias even when a single summary metric looks acceptable.

---

# 18.8.200 Error Distribution

Researchers should also inspect the distribution of residuals.

Define

$$
\Delta E
========

## E_{\text{pred}}

E_{\text{DFT}}.
$$

If errors are approximately centered around

$$
0,
$$

the model has little systematic bias.

If the residual distribution is shifted,

the model systematically

* overpredicts,
* or underpredicts.

The same analysis should be performed for force components and, where relevant, stress.

---

# 18.8.201 Evaluating Across Chemical Systems

A single global MAE can hide serious weaknesses.

Suppose the overall force MAE is

$$
0.08\ \text{eV/Å}.
$$

That sounds excellent.

But imagine the breakdown is

| Material Class   | Force MAE |
| ---------------- | --------: |
| Metals           | 0.04 eV/Å |
| Oxides           | 0.06 eV/Å |
| Semiconductors   | 0.05 eV/Å |
| Ionic conductors | 0.17 eV/Å |
| Surfaces         | 0.23 eV/Å |

The global average hides poor performance for surfaces and ionic conductors.

Therefore,

evaluation should be stratified according to scientifically meaningful categories whenever possible.

---

# 18.8.202 Evaluating Outside the Training Distribution

Perhaps the most important question is:

> What happens when M3GNet encounters structures unlike anything in its training dataset?

This is called **out-of-distribution generalization**.

Examples include

* new compositions,
* extreme pressures,
* unusual coordination,
* defects,
* surfaces,
* amorphous structures,
* very high temperatures.

Machine-learning potentials can become unreliable outside their training domain.

Therefore,

excellent interpolation performance does not guarantee reliable extrapolation.

---

# 18.8.203 The Central Lesson of Training

Training an M3GNet model is not simply

```python
model.fit(dataset)
```

It is a scientific workflow involving

```text
DFT Quality

↓

Dataset Diversity

↓

Configuration-Space Coverage

↓

Graph Construction

↓

Architecture

↓

Loss Weighting

↓

Optimization

↓

Validation

↓

Physical Testing

↓

Final Potential
```

Every stage influences the reliability of the final model.

A sophisticated neural network cannot compensate for a fundamentally poor dataset.

---

# 18.8.204 From a Trained Model to a Scientific Tool

Once the trained M3GNet potential has passed these tests,

something remarkable has happened.

We began with expensive quantum-mechanical calculations.

Those calculations taught the neural network the potential-energy landscape.

The neural network can now approximate that landscape much faster than running a new DFT calculation for every configuration.

The workflow becomes

```text
DFT

↓

Generate Training Data

↓

Train M3GNet

↓

Machine-Learned Potential

↓

Millions of Fast Predictions
```

This transformation is precisely what makes machine-learned interatomic potentials so powerful.

The model is no longer merely a predictor.

It becomes a computational engine for atomistic simulation.

---

# 18.8.205 Where We Go Next

Training the potential is not the end of the M3GNet workflow.

It is the beginning of its scientific use.

A trained potential can now be used to explore configurations that would be prohibitively expensive to investigate through DFT at every step.

The next major stage of this chapter is therefore **structure relaxation with M3GNet**.

We will move from

```text
Predicting Energy
```

to

```text
Using Energy + Forces + Stress

↓

Moving Atoms

↓

Changing the Lattice

↓

Finding Stable Structures
```

This connects the neural-network potential directly to one of the most important operations in computational materials science: **geometry optimization and crystal relaxation**.

# 18.9 Structure Relaxation with M3GNet

One of the most important applications of an interatomic potential is **structure relaxation**.

In computational materials science, the initial atomic positions obtained from an experiment, a crystal database, or a structure generation algorithm are often **not** the lowest-energy configuration.

Atoms may be

* slightly displaced,
* compressed,
* stretched,
* distorted,
* or arranged in a metastable configuration.

The purpose of structure relaxation is to move the atoms until the system reaches a stable equilibrium.

Unlike a simple energy prediction task,

relaxation is an **iterative optimization process**.

The model predicts the energy and forces, the atoms are moved accordingly, and the process is repeated until the forces become sufficiently small.

Conceptually,

```text id="m3g_relax1"
Initial Structure

↓

Predict Energy

↓

Predict Forces

↓

Move Atoms

↓

New Structure

↓

Repeat

↓

Relaxed Structure
```

This procedure is analogous to rolling a ball downhill until it settles at the bottom of a valley.

---

# 18.9.1 Why Relaxation Is Necessary

Suppose we construct a crystal by placing atoms according to an ideal prototype.

Although the arrangement may satisfy crystallographic symmetry, it may not correspond exactly to the minimum-energy structure.

Small deviations can arise from

* numerical approximations,
* thermal expansion,
* lattice strain,
* defects,
* vacancies,
* substitutions,
* manually generated structures.

If the structure is used directly for property calculations, the predicted properties may be inaccurate because the atoms are not at equilibrium.

Therefore, relaxation is almost always performed before calculating

* electronic properties,
* elastic properties,
* phonons,
* diffusion,
* molecular dynamics initial conditions.

---

# 18.9.2 The Potential Energy Surface

To understand relaxation, we must first understand the **Potential Energy Surface (PES)**.

The total energy of a material depends on the positions of all atoms.

For a system containing (N) atoms,

the energy is a function

$$
E
=

E(\mathbf{R}_1,\mathbf{R}_2,\ldots,\mathbf{R}_N),
$$

where

* $\mathbf{R}_i$ denotes the position of atom (i).

Since every atom has three coordinates,

the energy surface exists in a **3N-dimensional configuration space**.

For example,

* 10 atoms → 30-dimensional space,
* 100 atoms → 300-dimensional space,
* 1000 atoms → 3000-dimensional space.

Clearly,

this surface cannot be visualized directly.

---

# 18.9.3 A One-Dimensional Analogy

To develop intuition, imagine only one coordinate.

The energy may vary as

```text id="m3g_relax2"
Energy

│

│        ●

│      /

│    /

│  /

│/

└────────────────────

      Position
```

The lowest point corresponds to the stable equilibrium.

If the system starts at a higher position,

the force naturally drives it toward the minimum.

---

# 18.9.4 Local Minima and Global Minimum

Real materials rarely possess only one minimum.

Instead,

the potential energy surface contains many valleys.

```text id="m3g_relax3"
Energy

│       ●

│     /   \

│ ●_/     \__

│           \____

└────────────────────
```

Some valleys are

* shallow,
* deep,
* metastable,
* globally stable.

The deepest valley corresponds to the **global minimum**.

The others are **local minima**.

A relaxation algorithm generally converges to the nearest accessible minimum rather than the absolute global minimum.

Therefore,

the final structure depends on the starting configuration.

---

# 18.9.5 Forces Drive Relaxation

Earlier in this chapter we learned that atomic forces are obtained from the energy gradient,

$$
\mathbf{F}_i
============

*

\frac{\partial E}
{\partial \mathbf{R}_i}.
$$

This equation is the foundation of every geometry optimization algorithm.

The force always points toward decreasing energy.

Consequently,

if we repeatedly move atoms along the force direction,

the energy decreases.

This observation transforms relaxation into an optimization problem.

---

# 18.9.6 The Goal of Relaxation

A relaxed structure satisfies approximately

$$
\mathbf{F}_i
\approx
0
\qquad
\text{for every atom}.
$$

If cell relaxation is also performed,

the stress tensor should satisfy

$$
\boldsymbol{\sigma}
\approx
0,
$$

or the target external pressure.

Thus,

successful relaxation requires

* negligible forces,
* negligible stress (for variable-cell optimization),
* minimum total energy.

---

# 18.9.7 Energy Minimization vs Force Minimization

Students often wonder whether relaxation minimizes

* energy,
* or forces.

The answer is both.

Energy is the objective function,

while forces describe the local slope of that objective.

At the energy minimum,

the gradient becomes zero,

meaning

$$
\nabla E
========

0.

$$

Since

$$
\mathbf{F}
==========

*

\nabla E,
$$

zero gradient implies zero force.

Therefore,

energy minimization and force minimization are mathematically equivalent descriptions of the same process.

---

# 18.9.8 Relaxation Using M3GNet

With Density Functional Theory,

each optimization step requires a new self-consistent electronic calculation.

This is computationally expensive.

With M3GNet,

the workflow changes dramatically.

```text id="m3g_relax4"
Structure

↓

M3GNet

↓

Energy

+

Forces

↓

Geometry Optimizer

↓

Updated Structure

↓

Repeat
```

Instead of solving the Schrödinger equation at every step,

the neural network predicts energies and forces directly.

Consequently,

relaxation becomes orders of magnitude faster while often retaining high accuracy within the model's domain of validity.

---

# 18.9.9 The Iterative Relaxation Algorithm

A geometry optimization algorithm follows a repeated cycle.

1. Read the current structure.
2. Predict energy.
3. Predict forces.
4. Determine the displacement of each atom.
5. Update atomic coordinates.
6. Recalculate energy and forces.
7. Check convergence.
8. Stop if converged; otherwise repeat.

Conceptually,

```text id="m3g_relax5"
Structure

↓

Energy + Forces

↓

Move Atoms

↓

Converged?

↓

No ───────────────┐
                  │
Yes               │
↓                 │
Relaxed Structure │
                  │
└─────────────────┘
```

The neural network itself does **not** perform the optimization.

It supplies the energies and forces.

A geometry optimizer decides how the atoms should move.

---

# 18.9.10 Atomic Displacements

Suppose the current position of atom (i) is

$$
\mathbf{R}_i.
$$

The optimizer computes a displacement

$$
\Delta\mathbf{R}_i.
$$

The updated position becomes

$$
\mathbf{R}_i'
=============

\mathbf{R}_i
+
\Delta\mathbf{R}_i.
$$

The displacement depends on

* the force,
* the optimization algorithm,
* the optimization history.

Different optimizers compute different displacement vectors, even when the forces are identical.

---

# 18.9.11 Why We Cannot Simply Move Along the Force

One might imagine updating the coordinates using

$$
\mathbf{R}'
===========

\mathbf{R}
+
\alpha
\mathbf{F},
$$

where (\alpha) is a constant step size.

Although this idea resembles gradient descent,

it is usually inefficient for atomistic optimization.

If the step is too large,

atoms may overshoot the minimum.

If it is too small,

relaxation becomes unnecessarily slow.

Furthermore,

the energy surface often contains valleys with very different curvatures in different directions.

Modern optimizers therefore use information from previous iterations to choose more appropriate search directions and step sizes.

---

# 18.9.12 Convergence Criteria

The optimizer stops when predefined convergence conditions are satisfied.

Common criteria include

* maximum force,
* root-mean-square force,
* energy change,
* atomic displacement,
* stress magnitude.

For example,

a calculation may terminate when

$$
\max_i |\mathbf{F}_i|
<
0.01;
\text{eV/Å}.
$$

The exact threshold depends on the desired level of accuracy.

---

# 18.9.13 Energy During Relaxation

During optimization,

the energy generally decreases with each iteration.

A typical behavior is

```text id="m3g_relax6"
Energy

│●

│ \

│  \

│   \

│    \__

└────────────────────

Optimization Step
```

Initially,

large energy reductions occur.

As the structure approaches equilibrium,

the improvement becomes progressively smaller.

Eventually,

the energy changes become negligible, indicating convergence.

---

## Transition to the Next Section

So far, we have developed the physical principles behind geometry optimization: the potential energy surface, the relationship between forces and energy gradients, and the iterative relaxation procedure.

In the next section, we will study the **geometry optimization algorithms** themselves—including **Steepest Descent**, **Conjugate Gradient**, **BFGS**, and **LBFGS**—and explain why modern M3GNet workflows typically rely on quasi-Newton optimization methods for efficient and reliable structural relaxation.

# 18.9.14 Geometry Optimization Algorithms

The previous section established that relaxation consists of repeatedly moving atoms until the forces vanish.

A natural question now arises:

> **How should the atoms be moved?**

Although every optimizer uses the same information—primarily the energy and atomic forces—they differ in how they determine the next atomic positions.

Some methods

* move only in the direction of the current force,
* remember previous search directions,
* estimate the curvature of the energy surface,
* or approximate second derivatives.

The choice of optimizer has a major influence on

* convergence speed,
* numerical stability,
* computational efficiency.

---

# 18.9.15 The Optimization Problem

Suppose the atomic coordinates are collected into one vector

$$
\mathbf{R}
==========

(\mathbf{R}_1,\mathbf{R}_2,\ldots,\mathbf{R}_N).
$$

Relaxation seeks

$$
\mathbf{R}^*
============

\arg\min_{\mathbf{R}}
E(\mathbf{R}).
$$

Unlike neural-network training,

where millions of parameters are optimized,

geometry optimization changes only the atomic coordinates (and optionally the lattice vectors).

The neural network itself remains fixed.

---

# 18.9.16 Steepest Descent

The simplest geometry optimization algorithm is **Steepest Descent**.

The idea is straightforward.

Move atoms directly along the force direction.

Since

$$
\mathbf{F}
==========

*

\nabla E,
$$

the update becomes

$$
\mathbf{R}_{k+1}
================

\mathbf{R}_k
+
\alpha
\mathbf{F}_k,
$$

where

* $\alpha$ is the step size,
* $k$ denotes the optimization step.

Conceptually,

```text id="m3g_opt1"
Current Position

↓

Force

↓

Move

↓

New Position
```

The method always moves "downhill."

---

# 18.9.17 Advantages of Steepest Descent

Steepest Descent has several attractive properties.

* Very easy to implement.
* Requires only forces.
* Stable for poor initial structures.
* Robust during the early stages of optimization.

Because of these characteristics,

it is often used to remove extremely large forces before switching to a more sophisticated optimizer.

---

# 18.9.18 Why Steepest Descent Is Slow

Although Steepest Descent is simple,

it performs poorly near the energy minimum.

Imagine descending a narrow valley.

Instead of moving directly toward the minimum,

the algorithm repeatedly crosses from one side of the valley to the other.

```text id="m3g_opt2"
\        /

 \      /

  \    /

   \  /

    \/


Optimization Path

↘

↙

↘

↙

↘
```

This zigzag motion causes many unnecessary iterations.

Consequently,

Steepest Descent is rarely used for complete geometry optimization.

---

# 18.9.19 Conjugate Gradient

The **Conjugate Gradient (CG)** algorithm improves upon Steepest Descent.

Instead of using only the current force,

it also remembers the previous search direction.

The new search direction is

$$
\mathbf{p}_{k+1}
================

\mathbf{F}_{k+1}
+
\beta_k
\mathbf{p}_k,
$$

where

* $\mathbf{p}_k$ is the previous direction,
* $\beta_k$ determines how much of the previous direction is retained.

Unlike Steepest Descent,

the optimizer avoids repeatedly undoing previous progress.

---

# 18.9.20 Intuition Behind Conjugate Gradient

Suppose you are walking across a field.

Instead of forgetting every previous step,

you remember the overall direction in which the destination lies.

Consequently,

your path becomes much smoother.

Graphically,

```text id="m3g_opt3"
Steepest Descent

↘

↙

↘

↙

Conjugate Gradient

↘

 ↘

  ↘

   ↘
```

The number of optimization steps decreases substantially.

---

# 18.9.21 Advantages of Conjugate Gradient

Compared with Steepest Descent,

Conjugate Gradient

* converges faster,
* requires no second derivatives,
* is computationally inexpensive,
* performs well for medium-sized systems.

For many years,

CG was one of the most widely used geometry optimization algorithms in electronic-structure calculations.

---

# 18.9.22 Second Derivatives and Curvature

Both Steepest Descent and Conjugate Gradient rely primarily on first derivatives (forces).

However,

the shape of the energy surface also depends on its curvature.

The curvature is described mathematically by the **Hessian matrix**,

$$
\mathbf{H}
==========

\frac{\partial^2E}
{\partial\mathbf{R},\partial\mathbf{R}}.
$$

The Hessian tells us

* how rapidly the forces change,
* how steep the valley is,
* how strongly different atomic coordinates are coupled.

If the Hessian were known exactly,

geometry optimization could proceed much more efficiently.

---

# 18.9.23 Newton's Method

Newton's optimization method uses both

* first derivatives,
* second derivatives.

The update rule is

$$
\mathbf{R}_{k+1}
================

## \mathbf{R}_k

\mathbf{H}^{-1}
\nabla E.
$$

Since

$$
\nabla E
========

*

\mathbf{F},
$$

this may also be written as

$$
\mathbf{R}_{k+1}
================

\mathbf{R}_k
+
\mathbf{H}^{-1}
\mathbf{F}.
$$

Near the minimum,

Newton's method converges extremely rapidly.

---

# 18.9.24 Why Newton's Method Is Rarely Used Directly

Although Newton's method is mathematically elegant,

it has one major drawback.

The Hessian matrix is enormous.

For a system containing

(N)

atoms,

there are

(3N)

degrees of freedom.

The Hessian therefore has dimensions

$$
3N
\times
3N.
$$

For

1000

atoms,

the Hessian contains

$$
3000
\times
3000
====

9,000,000
$$

elements.

Computing and inverting such a matrix repeatedly is computationally expensive.

Consequently,

direct Newton optimization is impractical for most atomistic simulations.

---

# 18.9.25 Quasi-Newton Methods

Quasi-Newton methods provide an elegant solution.

Instead of calculating the exact Hessian,

they build an approximation using information collected during previous optimization steps.

Conceptually,

```text id="m3g_opt4"
Forces

↓

Estimate Curvature

↓

Approximate Hessian

↓

Better Step
```

This approximation captures much of the benefit of Newton's method while avoiding its computational cost.

---

# 18.9.26 The BFGS Algorithm

The most famous quasi-Newton algorithm is **BFGS**,

named after

* Broyden,
* Fletcher,
* Goldfarb,
* Shanno.

Rather than storing the exact Hessian,

BFGS continually updates an approximation based on

* atomic displacements,
* force changes.

After every optimization step,

the curvature estimate becomes more accurate.

Consequently,

the optimizer gradually learns the local shape of the potential energy surface.

---

# 18.9.27 Why BFGS Is Efficient

Imagine descending a mountain.

Instead of walking blindly downhill,

you gradually construct a mental map of the terrain.

Each new observation improves your estimate of the valley shape.

Eventually,

you know almost exactly where the minimum lies.

BFGS behaves similarly.

Rather than relying only on the current force,

it accumulates information from previous iterations.

---

# 18.9.28 The Hessian Approximation

Without deriving the complete algorithm,

the Hessian approximation satisfies

$$
\mathbf{H}_{k+1}
================

\mathbf{H}_k
+
\text{correction},
$$

where the correction depends on

* coordinate changes,
* gradient changes.

Each optimization step therefore improves the approximation.

Eventually,

the approximate Hessian becomes very similar to the true local curvature.

---

# 18.9.29 Limited-Memory BFGS (LBFGS)

For very large systems,

storing the complete approximate Hessian is still expensive.

The solution is **Limited-Memory BFGS (LBFGS)**.

Instead of storing the entire Hessian,

LBFGS stores only a small number of recent updates.

Conceptually,

```text id="m3g_opt5"
Previous Step

↓

Previous Step

↓

Previous Step

↓

Current Update
```

The optimizer reconstructs the search direction using this limited history.

Memory usage becomes almost independent of system size.

---

# 18.9.30 Why LBFGS Is Popular

LBFGS offers several advantages.

* Excellent convergence.
* Low memory usage.
* Suitable for large systems.
* No explicit Hessian calculation.
* Robust performance.

For these reasons,

LBFGS has become one of the standard optimizers in

* atomistic simulations,
* machine learning,
* scientific optimization.

Many M3GNet relaxation workflows employ LBFGS or closely related quasi-Newton methods because they combine rapid convergence with practical memory requirements.

---

# 18.9.31 Comparing Optimization Algorithms

The major optimization algorithms may be summarized as follows.

| Algorithm          | Uses Forces | Uses Hessian          | Memory Requirement | Typical Convergence    |
| ------------------ | ----------- | --------------------- | ------------------ | ---------------------- |
| Steepest Descent   | Yes         | No                    | Very Low           | Slow                   |
| Conjugate Gradient | Yes         | No                    | Low                | Moderate               |
| Newton             | Yes         | Exact Hessian         | Very High          | Very Fast Near Minimum |
| BFGS               | Yes         | Approximate Hessian   | Moderate           | Fast                   |
| LBFGS              | Yes         | Limited Approximation | Low                | Fast                   |

This comparison explains why quasi-Newton methods are widely preferred in practical geometry optimization.

---

# 18.9.32 Line Search

Regardless of the optimizer,

another important question remains.

Once a search direction has been chosen,

**how far should the atoms move?**

Moving too far may increase the energy.

Moving too little wastes computational effort.

Most optimizers therefore perform a **line search**.

Conceptually,

```text id="m3g_opt6"
Search Direction

↓

Trial Step

↓

Energy

↓

Too High?

↓

Adjust Step

↓

Accept
```

The line search determines a step length that sufficiently reduces the energy while maintaining numerical stability.

---

# 18.9.33 Convergence Near the Minimum

As relaxation approaches equilibrium,

both the forces and the atomic displacements become smaller.

A typical optimization history looks like

```text id="m3g_opt7"
Force

│●

│ \

│  \

│   \

│    \__

└────────────────────

Optimization Step
```

Eventually,

all convergence criteria are satisfied,

and the optimization terminates.

At this stage,

the structure is considered relaxed with respect to the chosen thresholds.

---

## Transition to the Next Section

Thus far, we have examined the mathematics and algorithms behind geometry optimization. However, in practice, researchers rarely implement these optimizers manually. Instead, M3GNet is commonly integrated with the **Atomic Simulation Environment (ASE)**, which provides robust implementations of BFGS, LBFGS, FIRE, and other optimizers.

In the next section, we will learn how **M3GNet interfaces with ASE**, how the neural network is exposed as an ASE calculator, and how complete crystal structure relaxations can be performed in just a few lines of Python while still understanding every computational step occurring behind the scenes.

# 18.9.34 M3GNet Integration with the Atomic Simulation Environment (ASE)

Training a high-quality M3GNet model is only the first step.

To perform practical atomistic simulations, the trained model must be connected to a simulation framework capable of

* storing atomic structures,
* performing geometry optimization,
* running molecular dynamics,
* computing physical properties,
* reading and writing common structure formats.

The most widely used framework for these tasks in Python is the **Atomic Simulation Environment (ASE)**.

ASE provides a common interface through which many computational engines—including Density Functional Theory codes, classical force fields, and machine-learned interatomic potentials such as M3GNet—can be used in exactly the same workflow.

This design allows researchers to switch between different computational methods with minimal changes to their simulation scripts.

---

# 18.9.35 What is ASE?

The **Atomic Simulation Environment (ASE)** is an open-source Python library for atomistic simulations.

It provides tools for

* constructing atomic structures,
* reading and writing crystal files,
* manipulating atoms,
* performing geometry optimization,
* running molecular dynamics,
* analyzing simulation results,
* interfacing with external calculators.

Rather than performing quantum-mechanical calculations itself, ASE acts as a **simulation framework** that communicates with external computational engines.

Conceptually,

```text id="m3g_ase1"
ASE

↓

Calculator

↓

Energy

Forces

Stress
```

The calculator may be

* VASP,
* Quantum ESPRESSO,
* GPAW,
* LAMMPS,
* M3GNet,
* or many others.

---

# 18.9.36 The ASE Architecture

ASE separates the simulation into two parts.

The first part stores the atomic structure.

The second part computes physical quantities.

```text id="m3g_ase2"
ASE Atoms Object

↓

Calculator

↓

Energy

↓

Forces

↓

Stress
```

This separation is extremely powerful because the same structure object can be evaluated using completely different computational methods.

For example,

the same crystal may first be evaluated using DFT and later using M3GNet without changing its atomic coordinates.

---

# 18.9.37 The Atoms Object

The central data structure in ASE is the **Atoms** object.

It stores all information required to describe a material,

including

* atomic species,
* atomic positions,
* lattice vectors,
* periodic boundary conditions.

Conceptually,

```text id="m3g_ase3"
Atoms

├── Elements

├── Coordinates

├── Cell

└── Periodicity
```

The Atoms object contains only structural information.

It does **not** know how to calculate energies or forces.

That responsibility belongs to the calculator.

---

# 18.9.38 What is a Calculator?

A **calculator** is an object that computes physical quantities for a given structure.

When ASE requests

```text id="m3g_ase4"
Energy
```

the calculator performs the necessary computation and returns the result.

Likewise,

ASE can request

```text id="m3g_ase5"
Forces
```

or

```text id="m3g_ase6"
Stress
```

without knowing how those quantities are calculated internally.

This abstraction is one of ASE's greatest strengths.

---

# 18.9.39 The Calculator Interface

Every ASE calculator provides a common interface.

Conceptually,

```text id="m3g_ase7"
Atoms

↓

Calculator

↓

get_potential_energy()

↓

get_forces()

↓

get_stress()
```

Regardless of whether the calculator is based on

* Density Functional Theory,
* empirical force fields,
* or graph neural networks,

ASE interacts with it in exactly the same way.

This standardization greatly simplifies simulation workflows.

---

# 18.9.40 Why This Design Is Powerful

Imagine writing a geometry optimization script.

Without ASE,

a different program would be required for every computational method.

With ASE,

the optimization code remains nearly identical.

Only the calculator changes.

Conceptually,

```text id="m3g_ase8"
ASE

↓

Calculator A

↓

Relaxation
```

can easily become

```text id="m3g_ase9"
ASE

↓

Calculator B

↓

Relaxation
```

The optimization algorithm itself does not need to know how the energy is computed.

---

# 18.9.41 Connecting M3GNet to ASE

To use M3GNet inside ASE,

the neural network is wrapped as an ASE calculator.

Conceptually,

```text id="m3g_ase10"
Crystal Structure

↓

ASE Atoms

↓

M3GNet Calculator

↓

Energy

Forces

Stress
```

Internally,

the calculator performs

1. graph construction,
2. message passing,
3. energy prediction,
4. automatic differentiation,
5. force calculation,
6. stress calculation.

From ASE's perspective,

however,

it simply receives numerical values.

---

# 18.9.42 Workflow During Relaxation

Suppose we relax a crystal using ASE and M3GNet.

Each optimization step proceeds as follows.

```text id="m3g_ase11"
Atoms

↓

ASE Optimizer

↓

Request Forces

↓

M3GNet Calculator

↓

Graph Construction

↓

Neural Network

↓

Energy

↓

Automatic Differentiation

↓

Forces

↓

Optimizer Updates Coordinates

↓

Repeat
```

Notice that ASE never constructs graphs itself.

That task belongs entirely to the M3GNet calculator.

---

# 18.9.43 Reading Crystal Structures

ASE can read many common structure formats,

including

* CIF,
* POSCAR,
* XYZ,
* PDB,
* and others.

For example,

```python id="m3gnet_code34"
from ase.io import read

atoms = read("Si.cif")
```

The resulting `atoms` object contains

* lattice vectors,
* atomic coordinates,
* chemical symbols,
* periodic boundary conditions.

At this stage,

no energy has yet been calculated.

---

# 18.9.44 Assigning the Calculator

Once the calculator has been created,

it is attached to the structure.

Conceptually,

```python id="m3gnet_code35"
atoms.calc = calculator
```

After this assignment,

the structure immediately gains the ability to compute

* energy,
* forces,
* stress.

The atomic coordinates themselves remain unchanged.

Only the computational engine has been attached.

---

# 18.9.45 Computing the Total Energy

The total potential energy is obtained by calling

```python id="m3gnet_code36"
energy = atoms.get_potential_energy()
```

Internally,

ASE performs the following sequence.

```text id="m3g_ase12"
Atoms

↓

Calculator

↓

Graph

↓

M3GNet

↓

Energy
```

To the user,

however,

this complexity is hidden behind a single function call.

---

# 18.9.46 Computing Atomic Forces

Similarly,

the forces are obtained using

```python id="m3gnet_code37"
forces = atoms.get_forces()
```

The returned array contains one three-dimensional force vector for every atom.

Conceptually,

```text id="m3g_ase13"
Atom 1

Fx Fy Fz

Atom 2

Fx Fy Fz

⋮

Atom N

Fx Fy Fz
```

These forces are precisely the quantities required by the geometry optimizer.

---

# 18.9.47 Computing the Stress Tensor

For variable-cell optimization,

the stress tensor can also be obtained.

```python id="m3gnet_code38"
stress = atoms.get_stress()
```

The returned values describe how the total energy changes with lattice deformation.

Stress information is essential for

* cell relaxation,
* pressure calculations,
* equation-of-state studies,
* elastic property evaluation.

---

# 18.9.48 Reusing the Calculator

One useful feature of ASE is that the same calculator can often be reused for multiple structures, depending on the workflow and implementation.

For example,

```text id="m3g_ase14"
Structure A

↓

Calculator

↓

Energy


Structure B

↓

Same Calculator

↓

Energy
```

This avoids repeatedly creating new calculator objects and simplifies large-scale screening studies.

---

# 18.9.49 Advantages of Using ASE with M3GNet

The combination of ASE and M3GNet provides several important benefits.

* Standardized simulation workflow.
* Support for many crystal formats.
* Access to robust optimization algorithms.
* Molecular dynamics capabilities.
* Trajectory storage.
* Analysis tools.
* Easy integration with other computational methods.

Most importantly,

the researcher can focus on the scientific problem rather than implementing low-level simulation algorithms.

---

# 18.9.50 Conceptual View of the Complete Workflow

The complete relationship between the different software components can now be summarized.

```text id="m3g_ase15"
Crystal File

↓

ASE

↓

Atoms Object

↓

M3GNet Calculator

↓

Graph Neural Network

↓

Energy

↓

Automatic Differentiation

↓

Forces + Stress

↓

ASE Optimizer

↓

Relaxed Structure
```

Each component has a clearly defined responsibility.

* ASE manages the simulation.
* M3GNet predicts the physics.
* The optimizer updates the atomic positions.

This modular design is one of the reasons why ASE has become the standard platform for atomistic simulations in Python.

---

# 18.9.51 Common Optimizers Available in ASE

ASE provides several optimization algorithms that can be used with M3GNet.

Some of the most commonly used are

| Optimizer | Characteristics                       | Typical Use                                    |
| --------- | ------------------------------------- | ---------------------------------------------- |
| BFGS      | Quasi-Newton, accurate                | Small and medium systems                       |
| LBFGS     | Limited-memory quasi-Newton           | Large systems                                  |
| FIRE      | Fast inertial relaxation              | Large atomic systems and MD-derived structures |
| MDMin     | Molecular-dynamics-based minimization | Initial relaxation                             |
| GPMin     | Gaussian process assisted             | Expensive calculators                          |

Among these,

**BFGS**, **LBFGS**, and **FIRE** are the most frequently used with machine-learned interatomic potentials such as M3GNet.

---

## Transition to the Next Section

We now understand how ASE and M3GNet work together: ASE manages the simulation, the M3GNet calculator predicts energies, forces, and stresses, and the optimizer updates the structure.

In the next section, we will perform a **complete crystal structure relaxation using M3GNet and ASE**, examining every line of Python code, explaining the role of each function, monitoring convergence, saving trajectories, and interpreting the final relaxed structure in a real research workflow.

# 18.9.52 Complete Crystal Structure Relaxation with M3GNet and ASE

Having understood the theory behind geometry optimization and the role of ASE, we are now ready to perform a complete crystal structure relaxation.

In this section, we will study the **entire workflow** used in practical research.

Rather than simply presenting a short script, we will examine every step in detail and explain what happens internally.

The overall workflow is

```text id="m3g_relax_workflow1"
Crystal Structure

↓

Read Structure

↓

Load M3GNet Model

↓

Create ASE Calculator

↓

Attach Calculator

↓

Choose Optimizer

↓

Run Relaxation

↓

Check Convergence

↓

Save Relaxed Structure
```

Although the Python code required for relaxation is relatively short, a large amount of computation occurs behind the scenes.

---

# 18.9.53 Step 1 — Import Required Libraries

The first step is importing the necessary libraries.

A typical workflow requires

* ASE,
* MatGL,
* PyTorch.

Conceptually,

```python id="m3gnet_code39"
from ase.io import read, write
from ase.optimize import LBFGS

import matgl
```

Depending on the installed MatGL version, additional imports may be required (for example, the calculator class or graph converter). Because the MatGL API evolves over time, you should always consult the documentation corresponding to your installed version.

Each imported package has a specific responsibility.

* ASE manages structures and optimization.
* MatGL provides the trained M3GNet model.
* PyTorch performs neural-network inference and automatic differentiation internally.

---

# 18.9.54 Step 2 — Load the Crystal Structure

Suppose we have a CIF file

```text id="m3g_relax_file1"
Si.cif
```

The structure can be read using ASE.

```python id="m3gnet_code40"
atoms = read("Si.cif")
```

Internally,

ASE reads

* lattice vectors,
* fractional coordinates,
* atomic species,
* periodic boundary conditions.

After loading,

the `atoms` object contains only structural information.

No energy has yet been calculated.

---

# 18.9.55 Internal Representation of the Structure

Conceptually,

the loaded structure contains

```text id="m3g_relax_structure1"
Atoms

├── Si

├── Si

├── Si

├── Si

Cell

↓

Lattice Vectors

↓

Periodic Boundary Conditions
```

At this stage,

the object knows **where the atoms are**,

but it does not yet know

* their energy,
* their forces,
* their stress.

Those quantities depend on the calculator.

---

# 18.9.56 Step 3 — Load the Pretrained M3GNet Model

Next,

the pretrained model is loaded.

Conceptually,

```python id="m3gnet_code41"
model = ...
```

The exact command depends on the MatGL version and on whether

* a pretrained universal model,
* a locally trained checkpoint,
* or a fine-tuned model

is being used.

Regardless of how the model is loaded,

the result is the same:

a neural network capable of predicting

* energy,
* forces,
* stress.

---

# 18.9.57 What Does the Loaded Model Contain?

The pretrained model contains millions of learned parameters.

Conceptually,

```text id="m3g_relax_model1"
Atomic Embeddings

↓

Interaction Blocks

↓

Three-Body Layers

↓

Readout Network

↓

Learned Parameters
```

These parameters were obtained during training using thousands or millions of DFT calculations.

During relaxation,

they remain fixed.

Only the atomic coordinates change.

---

# 18.9.58 Step 4 — Create the Calculator

The neural network must now be wrapped as an ASE calculator.

Conceptually,

```python id="m3gnet_code42"
calculator = ...
```

Internally,

the calculator connects

```text id="m3g_relax_calc1"
ASE

↓

Calculator

↓

M3GNet

↓

Energy

Forces

Stress
```

From ASE's perspective,

the calculator behaves exactly like any other computational engine.

---

# 18.9.59 Step 5 — Attach the Calculator

The calculator is attached to the structure.

```python id="m3gnet_code43"
atoms.calc = calculator
```

This single line establishes the connection between

* the crystal,
* the neural network.

Afterward,

ASE can request

```python
atoms.get_potential_energy()

atoms.get_forces()

atoms.get_stress()
```

without knowing anything about graph neural networks.

---

# 18.9.60 What Happens Internally?

Suppose ASE requests the total energy.

Internally,

the following sequence occurs.

```text id="m3g_relax_internal1"
ASE

↓

Atoms

↓

Graph Construction

↓

M3GNet

↓

Atomic Energies

↓

Total Energy
```

If forces are requested,

automatic differentiation computes

$$
\mathbf{F}
==========

*

\frac{\partial E}
{\partial\mathbf{R}}.
$$

The user never needs to perform these derivative calculations manually.

---

# 18.9.61 Step 6 — Choose the Optimizer

The next step is selecting a geometry optimizer.

For example,

```python id="m3gnet_code44"
optimizer = LBFGS(atoms)
```

or

```python id="m3gnet_code45"
optimizer = BFGS(atoms)
```

or

```python id="m3gnet_code46"
optimizer = FIRE(atoms)
```

Each optimizer uses the same

* energies,
* forces,

but chooses different atomic displacements.

The calculator remains unchanged.

---

# 18.9.62 Why LBFGS Is Often Preferred

For machine-learned interatomic potentials,

LBFGS is frequently an excellent choice because it

* converges rapidly,
* requires relatively little memory,
* handles large systems efficiently,
* approximates second-order optimization.

However,

no optimizer is universally best.

Some poorly conditioned systems converge more reliably with FIRE,

whereas small systems often relax efficiently using BFGS.

---

# 18.9.63 Step 7 — Running the Optimization

The relaxation is started by specifying a force convergence threshold.

Conceptually,

```python id="m3gnet_code47"
optimizer.run(
    fmax=0.01
)
```

The value

```text
0.01 eV/Å
```

means that relaxation stops when the largest atomic force becomes smaller than

$$
0.01
;
\text{eV/Å}.
$$

This criterion is commonly used for high-quality geometry optimization.

The exact threshold should be chosen based on the accuracy requirements of the study.

---

# 18.9.64 What Happens During Each Iteration?

Each optimization iteration follows the sequence

```text id="m3g_relax_iteration1"
Current Structure

↓

Energy

↓

Forces

↓

Optimizer

↓

Updated Structure

↓

Repeat
```

The optimizer repeatedly asks the calculator for new forces.

The calculator repeatedly evaluates the neural network.

This continues until convergence.

---

# 18.9.65 Example of Energy Convergence

Suppose the optimization history is

| Step | Energy (eV) |
| ---: | ----------: |
|    0 |     -125.41 |
|    5 |     -126.02 |
|   10 |     -126.27 |
|   15 |     -126.35 |
|   20 |     -126.36 |

Initially,

the energy decreases rapidly.

Near convergence,

the changes become much smaller.

This behavior is expected because the structure approaches the minimum of the potential energy surface.

---

# 18.9.66 Example of Force Convergence

Similarly,

the maximum force might decrease as

| Step | Maximum Force (eV/Å) |
| ---: | -------------------: |
|    0 |                 1.52 |
|    5 |                 0.43 |
|   10 |                 0.11 |
|   15 |                0.026 |
|   20 |                0.008 |

Once the maximum force falls below

```text
0.01 eV/Å
```

the optimization terminates.

Notice that the force decreases by nearly two orders of magnitude during the relaxation.

---

# 18.9.67 Monitoring the Relaxation

ASE prints useful information during optimization.

A typical log includes

* optimization step,
* total energy,
* maximum force,
* elapsed time.

Researchers monitor these quantities to ensure that

* the energy decreases smoothly,
* the forces approach zero,
* no numerical instability occurs.

Sudden increases in energy or oscillating forces may indicate poor convergence or an unsuitable optimizer.

---

# 18.9.68 Saving the Relaxed Structure

After convergence,

the relaxed structure should be written to disk.

For example,

```python id="m3gnet_code48"
write(
    "Si_relaxed.cif",
    atoms
)
```

The output file contains

* updated atomic coordinates,
* relaxed lattice (if variable-cell optimization was performed),
* optimized geometry.

This relaxed structure can then be used for

* DFT refinement,
* phonon calculations,
* molecular dynamics,
* electronic structure calculations,
* further machine-learning studies.

---

# 18.9.69 Saving the Optimization Trajectory

Rather than saving only the final structure,

it is often useful to save **every optimization step**.

This creates an optimization trajectory.

Conceptually,

```text id="m3g_relax_traj1"
Initial Structure

↓

Step 1

↓

Step 2

↓

Step 3

↓

⋮

↓

Final Structure
```

Trajectory files allow researchers to

* visualize atomic motion,
* analyze convergence,
* restart interrupted calculations,
* diagnose optimization failures.

ASE supports trajectory output through its `Trajectory` class or by attaching a trajectory writer to the optimizer. The exact syntax depends on the desired file format and ASE version.

---

# 18.9.70 Visualizing the Relaxation

Trajectory files can be visualized using

* ASE GUI,
* OVITO,
* VESTA (after exporting snapshots),
* or other visualization software.

Watching the optimization often reveals

* atomic rearrangements,
* bond formation,
* bond breaking,
* lattice deformation,
* defect migration.

Visualization therefore provides valuable physical insight beyond numerical convergence.

---

# 18.9.71 Variable-Cell Relaxation

So far,

we have mainly discussed moving atoms while keeping the lattice fixed.

However,

many materials require the lattice itself to relax.

In this case,

both

* atomic positions,
* lattice vectors

are optimized simultaneously.

Conceptually,

```text id="m3g_relax_cell1"
Atoms Move

+

Cell Changes

↓

Lower Energy
```

Variable-cell relaxation relies on both

* atomic forces,
* stress tensor.

---

# 18.9.72 Why Variable-Cell Relaxation Matters

Suppose the lattice constant is initially too small.

Even if all atomic forces become zero,

the crystal may still be under significant compressive stress.

Similarly,

an excessively large lattice may produce tensile stress.

Therefore,

equilibrium requires both

$$
\mathbf{F}_i
\approx
0
$$

and

$$
\boldsymbol{\sigma}
\approx
0,
$$

or the desired external pressure.

Optimizing only the atomic positions while neglecting lattice stress can therefore leave the structure in a mechanically non-equilibrium state.

---

## Transition to the Next Section

We have now completed a full M3GNet-based crystal relaxation workflow, from loading a structure and attaching the calculator to monitoring convergence and saving the optimized geometry.

In the next section, we will examine **cell optimization and stress-driven lattice relaxation in greater depth**, including how lattice vectors are updated, how external pressure is applied, and how M3GNet enables efficient variable-cell optimization comparable to first-principles calculations while remaining orders of magnitude faster.

# 18.9.73 Variable-Cell Relaxation

In the previous sections, we focused primarily on **atomic relaxation**, where only the atomic coordinates are optimized while the lattice vectors remain fixed.

However, many materials are not only out of equilibrium because of incorrect atomic positions, but also because of an incorrect **unit cell geometry**.

The lattice may be

* compressed,
* expanded,
* sheared,
* or distorted.

Even if every atom occupies its lowest-energy position within that fixed lattice, the crystal may still possess internal stress.

Therefore, many simulations require **variable-cell relaxation**, in which both the atomic positions and the lattice vectors are optimized simultaneously.

Conceptually,

```text id="m3g_varcell1"
Initial Cell

↓

Atoms Move

+

Cell Changes

↓

Lower Energy

↓

Equilibrium Structure
```

Unlike fixed-cell optimization, variable-cell relaxation searches for the lowest-energy combination of

* atomic coordinates,
* cell lengths,
* cell angles.

---

# 18.9.74 Atomic Relaxation vs Cell Relaxation

These two optimization problems are fundamentally different.

### Fixed-cell relaxation

Only atomic positions change.

```text id="m3g_varcell2"
Cell

──────────────

Unchanged

↓

Atoms Move
```

The lattice vectors remain constant.

---

### Variable-cell relaxation

Both the lattice and the atoms change.

```text id="m3g_varcell3"
Cell

↓

Expands

Contracts

Shears

↓

Atoms Move
```

This produces the true equilibrium crystal structure under the specified external conditions.

---

# 18.9.75 Why Cell Relaxation Is Necessary

Suppose a silicon crystal is constructed with a lattice constant that is 5% smaller than its equilibrium value.

Initially,

the atoms experience strong compressive forces.

Even after relaxing the atomic positions,

the crystal remains compressed because the lattice itself has not changed.

Consequently,

the structure still stores elastic energy.

Similarly,

if the lattice is too large,

the crystal experiences tensile stress.

Only by allowing the lattice vectors to change can the crystal reach mechanical equilibrium.

---

# 18.9.76 Internal Stress

The driving force behind cell relaxation is the **stress tensor**.

While atomic forces describe how the energy changes when atoms move,

the stress tensor describes how the energy changes when the lattice is deformed.

Mathematically,

stress is related to the derivative of the total energy with respect to strain.

A simplified expression is

$$
\sigma_{ij}
===========

\frac{1}{V}
\frac{\partial E}
{\partial \varepsilon_{ij}},
$$

where

* $V$ is the cell volume,
* $\varepsilon_{ij}$ is the strain tensor,
* $\sigma_{ij}$ is the stress tensor.

This equation shows that stress measures the energetic response to deformation of the simulation cell.

---

# 18.9.77 Physical Interpretation of Stress

Stress can be viewed as the **internal mechanical pressure** acting within the crystal.

For example,

### Positive stress

The crystal tends to expand.

```text id="m3g_varcell4"
←──── Cell ────→

Internal Pressure

↓

Expansion
```

---

### Negative stress

The crystal tends to contract.

```text id="m3g_varcell5"
Compression

↓

Smaller Cell
```

At equilibrium,

the internal stress balances the external pressure.

---

# 18.9.78 Stress Tensor Components

The stress tensor is a **second-order tensor** containing nine components,

$$
\boldsymbol{\sigma}
===================

\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} & \sigma_{xz} \
\sigma_{yx} & \sigma_{yy} & \sigma_{yz} \
\sigma_{zx} & \sigma_{zy} & \sigma_{zz}
\end{bmatrix}.
$$

For crystalline materials in equilibrium,

the tensor is symmetric,

meaning

$$
\sigma_{ij}
===========

\sigma_{ji}.
$$

Therefore,

only six independent components need to be determined.

These six quantities completely describe the internal mechanical state of the crystal.

---

# 18.9.79 Normal and Shear Stress

The stress tensor contains two different types of stress.

### Normal stress

Acts perpendicular to a surface.

Examples include

* compression,
* tension.

These correspond to

$$
\sigma_{xx},
\quad
\sigma_{yy},
\quad
\sigma_{zz}.
$$

---

### Shear stress

Acts parallel to a surface.

These correspond to

$$
\sigma_{xy},
\sigma_{xz},
\sigma_{yz}.
$$

Shear stresses distort the crystal shape without necessarily changing its volume.

---

# 18.9.80 Hydrostatic Pressure

A particularly important case is **hydrostatic pressure**.

Here,

all normal stresses are equal,

while all shear stresses vanish.

Thus,

$$
\sigma_{xx}
===========

# \sigma_{yy}

\sigma_{zz},
$$

and

$$
\sigma_{xy}
===========

# \sigma_{xz}

# \sigma_{yz}

0.

$$

Hydrostatic pressure changes only the volume of the crystal.

The crystal expands or contracts uniformly without changing its shape.

---

# 18.9.81 Relationship Between Stress and Cell Relaxation

The optimization algorithm attempts to eliminate internal stress.

Suppose the cell is compressed.

The stress tensor indicates that expansion lowers the total energy.

The optimizer therefore increases the lattice constants.

If the cell is too large,

the optimizer decreases the lattice constants.

Conceptually,

```text id="m3g_varcell6"
Stress

↓

Cell Update

↓

Lower Stress

↓

Repeat

↓

Zero Stress
```

The process continues until the stress reaches the desired target.

---

# 18.9.82 Variable-Cell Optimization Loop

During each optimization step,

both atomic coordinates and lattice vectors are updated.

The workflow becomes

```text id="m3g_varcell7"
Structure

↓

Energy

↓

Forces

+

Stress

↓

Optimizer

↓

New Cell

+

New Atomic Positions

↓

Repeat
```

Compared with fixed-cell relaxation,

additional calculations involving the stress tensor are required.

---

# 18.9.83 Role of M3GNet in Cell Relaxation

M3GNet predicts

* total energy,
* atomic forces,
* stress tensor,

within a single neural-network framework.

Thus,

variable-cell optimization requires no additional quantum-mechanical calculations.

The neural network evaluates the crystal,

automatic differentiation computes the required gradients,

and ASE uses those quantities to update

* the lattice,
* the atomic positions.

Consequently,

cell optimization using M3GNet is often hundreds or thousands of times faster than repeating DFT calculations at every optimization step.

---

# 18.9.84 Cell Degrees of Freedom

A three-dimensional crystal cell possesses six independent geometric degrees of freedom.

These are

* lattice length (a),
* lattice length (b),
* lattice length (c),
* angle (\alpha),
* angle (\beta),
* angle (\gamma).

During variable-cell optimization,

some or all of these quantities may change.

For highly symmetric crystals,

certain constraints may be imposed so that symmetry is preserved throughout the relaxation.

---

# 18.9.85 Example: Cubic Crystal Relaxation

Consider a cubic crystal.

Initially,

```text id="m3g_varcell8"
a = 5.20 Å
```

Suppose the equilibrium lattice constant is

```text id="m3g_varcell9"
a = 5.43 Å
```

The optimizer gradually increases the lattice parameter.

An example optimization history is

| Step | Lattice Constant (Å) |
| ---: | -------------------: |
|    0 |                 5.20 |
|    5 |                 5.31 |
|   10 |                 5.39 |
|   15 |                 5.42 |
|   20 |                 5.43 |

As the lattice constant approaches its equilibrium value,

the stress simultaneously approaches zero.

---

# 18.9.86 Cell Volume Evolution

During relaxation,

the unit-cell volume also changes.

The volume of a general unit cell is

$$
V
=

\mathbf{a}
\cdot
(\mathbf{b}
\times
\mathbf{c}),
$$

where

* $\mathbf{a}$,
* $\mathbf{b}$,
* $\mathbf{c}$

are the lattice vectors.

The optimizer continuously adjusts these vectors until the equilibrium volume is reached.

A decreasing stress is therefore usually accompanied by stabilization of the unit-cell volume.

---

# 18.9.87 Practical Applications of Variable-Cell Relaxation

Variable-cell optimization is essential in many areas of materials science.

Examples include

* predicting equilibrium crystal structures,
* determining lattice constants,
* calculating equations of state,
* studying materials under pressure,
* defect relaxation,
* surface reconstruction,
* phase-transition studies,
* high-pressure materials discovery.

Without cell relaxation, many computed structural and mechanical properties would be inaccurate because the crystal would not be in true mechanical equilibrium.

---

## Transition to the Next Section

We have now seen how M3GNet predicts both atomic forces and the stress tensor, enabling simultaneous optimization of atomic coordinates and lattice geometry. This makes it possible to obtain fully relaxed equilibrium structures with computational costs far below those of first-principles methods.

In the next section, we will extend these ideas to **Molecular Dynamics (MD)**, where atoms are no longer moved toward a minimum-energy configuration but instead evolve continuously in time according to Newton's equations of motion using the forces predicted by M3GNet.

# 18.10 Molecular Dynamics with M3GNet

Geometry optimization determines the **lowest-energy structure** of a material.

However, real materials rarely remain completely motionless.

At any temperature above absolute zero, atoms continuously vibrate because of their thermal energy.

These vibrations influence numerous material properties, including

* thermal expansion,
* diffusion,
* melting,
* phase transitions,
* ionic conductivity,
* mechanical behavior,
* heat capacity,
* thermal conductivity.

To study these dynamic phenomena, we require a simulation technique capable of following the motion of atoms as a function of time.

This technique is known as **Molecular Dynamics (MD)**.

Unlike geometry optimization, where atoms move only toward a minimum-energy configuration, molecular dynamics simulates the **actual physical motion** of atoms by solving Newton's equations of motion.

---

# 18.10.1 Geometry Optimization vs Molecular Dynamics

Although both methods use atomic forces, their objectives are fundamentally different.

### Geometry Optimization

Goal:

Find the lowest-energy configuration.

```text id="m3g_md1"
Initial Structure

↓

Energy Minimization

↓

Relaxed Structure
```

Atoms stop moving once equilibrium is reached.

---

### Molecular Dynamics

Goal:

Simulate the time evolution of atoms.

```text id="m3g_md2"
Initial Structure

↓

Time Evolution

↓

Atomic Trajectory
```

Atoms never stop moving unless the temperature is exactly

$$
0 ; \text{K}.
$$

---

# 18.10.2 What Molecular Dynamics Simulates

In molecular dynamics,

every atom behaves like a tiny particle obeying the laws of classical mechanics.

Each atom possesses

* position,
* velocity,
* acceleration,
* mass.

The simulation repeatedly computes

1. atomic forces,
2. accelerations,
3. velocities,
4. positions.

This process is repeated thousands or millions of times.

---

# 18.10.3 Newton's Second Law

The foundation of molecular dynamics is Newton's Second Law.

For atom (i),

$$
\mathbf{F}_i
============

m_i
\mathbf{a}_i,
$$

where

* $m_i$ is the atomic mass,
* $\mathbf{a}_i$ is the acceleration,
* $\mathbf{F}_i$ is the force.

Since M3GNet predicts the force,

the acceleration becomes

$$
\mathbf{a}_i
============

\frac{\mathbf{F}_i}{m_i}.
$$

Thus,

the neural network provides exactly the information required to simulate atomic motion.

---

# 18.10.4 Why Forces Are Central to MD

Notice an important distinction.

Geometry optimization uses forces to reduce energy.

Molecular dynamics uses forces to calculate acceleration.

The same predicted force therefore serves two completely different purposes.

During relaxation,

```text id="m3g_md3"
Force

↓

Move Toward Minimum
```

During MD,

```text id="m3g_md4"
Force

↓

Acceleration

↓

Velocity

↓

Position
```

The objective is not to minimize energy but to reproduce physically realistic trajectories.

---

# 18.10.5 The MD Time Step

Atomic motion is continuous.

Computers, however, can only perform discrete calculations.

Therefore, time is divided into very small intervals called **time steps**.

The simulation proceeds as

```text id="m3g_md5"
t

↓

t + Δt

↓

t + 2Δt

↓

t + 3Δt

↓

...
```

Typical MD time steps are

* 0.5 femtoseconds (fs),
* 1 femtosecond,
* 2 femtoseconds.

Since

$$
1;\text{fs}
===========

10^{-15};\text{s},
$$

millions of time steps may be required to simulate only a few nanoseconds of real physical time.

---

# 18.10.6 Updating Atomic Positions

Suppose the current atomic position is

$$
\mathbf{R}(t),
$$

and the velocity is

$$
\mathbf{v}(t).
$$

After a small time interval,

the position changes approximately as

$$
\mathbf{R}(t+\Delta t)
======================

\mathbf{R}(t)
+
\mathbf{v}(t)\Delta t.
$$

Likewise,

the velocity changes according to the acceleration.

Thus,

positions and velocities evolve together throughout the simulation.

---

# 18.10.7 The Molecular Dynamics Cycle

Every MD step follows the same sequence.

```text id="m3g_md6"
Atomic Positions

↓

M3GNet

↓

Energy

↓

Forces

↓

Acceleration

↓

Velocity

↓

New Positions

↓

Repeat
```

This loop may be executed

* thousands,
* millions,
* or even billions

of times depending on the length of the simulation.

---

# 18.10.8 Why Machine Learning Accelerates MD

Traditional first-principles molecular dynamics computes the forces using Density Functional Theory at every time step.

Suppose

100,000

time steps are required.

DFT therefore performs

100,000

electronic structure calculations.

This makes long simulations extremely expensive.

With M3GNet,

the workflow changes.

```text id="m3g_md7"
Atomic Positions

↓

Graph Construction

↓

Neural Network

↓

Forces
```

Instead of solving the Schrödinger equation,

the neural network predicts the forces directly.

The computational cost decreases dramatically while maintaining high accuracy within the domain represented by the training data.

---

# 18.10.9 Initial Conditions

Before starting an MD simulation,

initial conditions must be specified.

These include

* atomic positions,
* initial velocities,
* simulation temperature,
* simulation cell,
* time step.

The atomic positions often come from a relaxed crystal structure.

Initial velocities are usually assigned according to the desired temperature using a Maxwell–Boltzmann distribution.

---

# 18.10.10 Why Initial Velocities Matter

Suppose every atom begins with zero velocity.

Then

$$
K
=

0,
$$

where (K) is the kinetic energy.

The system therefore corresponds approximately to

```text id="m3g_md8"
0 K
```

because no thermal motion exists.

To simulate finite temperatures,

atoms must be assigned random velocities consistent with statistical mechanics.

---

# 18.10.11 Kinetic Energy

Each atom possesses kinetic energy

$$
K_i
===

\frac12
m_i
v_i^2.
$$

The total kinetic energy becomes

$$
K
=

\sum_i
\frac12
m_i
v_i^2.
$$

Unlike geometry optimization,

kinetic energy plays a central role in molecular dynamics because atoms are continuously moving.

---

# 18.10.12 Potential Energy

The neural network predicts the potential energy

$$
E_{\mathrm{pot}}.
$$

Therefore,

the total energy of the system is

$$
E_{\mathrm{total}}
==================

K
+
E_{\mathrm{pot}}.
$$

Both contributions evolve throughout the simulation.

As atoms move,

potential energy may decrease while kinetic energy increases,

or vice versa,

depending on the interaction between atoms.

---

# 18.10.13 Energy Exchange

Consider an atom moving toward a neighboring atom.

Initially,

its kinetic energy is high.

As the atoms approach each other,

repulsive interactions increase the potential energy.

Eventually,

the atom slows down.

Conceptually,

```text id="m3g_md9"
Kinetic Energy

↓

Potential Energy

↓

Kinetic Energy

↓

Potential Energy
```

Energy is continuously exchanged between kinetic and potential forms while the total energy remains conserved in an ideal isolated system.

---

# 18.10.14 Trajectories

The primary output of an MD simulation is not a single relaxed structure.

Instead,

the result is a trajectory.

```text id="m3g_md10"
Step 1

↓

Step 2

↓

Step 3

↓

Step 4

↓

⋮
```

Every frame contains

* atomic coordinates,
* velocities,
* energies,
* simulation time.

These trajectories allow researchers to analyze the dynamic behavior of materials rather than only their equilibrium configurations.

---

# 18.10.15 Visualization of Atomic Motion

Trajectory files can be visualized using software such as

* ASE GUI,
* OVITO,
* VMD,
* VESTA (for exported snapshots).

Animations reveal phenomena that are difficult to infer from numerical data alone, including

* lattice vibrations,
* atomic diffusion,
* defect migration,
* thermal disorder,
* structural transformations.

Such visualizations often provide intuitive insight into atomistic processes.

---

# 18.10.16 Why Molecular Dynamics Is Important in Materials Science

Many important physical phenomena cannot be understood using static crystal structures alone.

Examples include

* lithium-ion diffusion in battery materials,
* oxygen vacancy migration in solid oxides,
* thermal expansion of metals,
* melting and solidification,
* grain-boundary motion,
* crack propagation,
* ionic conductivity,
* phase transformations.

These processes inherently involve atomic motion over time and therefore require molecular dynamics simulations.

---

## Transition to the Next Section

We have established the fundamental principles of molecular dynamics and seen how M3GNet supplies the forces needed to propagate atomic motion efficiently.

In the next section, we will examine the **numerical integration algorithms** used in molecular dynamics—particularly the **Verlet**, **Velocity Verlet**, and **Leapfrog** methods—and explain how atomic positions and velocities are updated accurately and stably at every femtosecond time step.

# 18.10.17 Numerical Integration in Molecular Dynamics

The previous section established that molecular dynamics is governed by Newton's Second Law,

$$
\mathbf{F}
==========

m\mathbf{a}.
$$

Once the force acting on every atom is known, the acceleration can be calculated.

However, knowing the acceleration alone is **not sufficient**.

We must also determine

* the new velocities,
* the new atomic positions,

after every time step.

This requires solving the equations of motion.

For simple systems, analytical solutions may exist.

For realistic materials containing hundreds or thousands of atoms interacting through complex potentials, analytical solutions are impossible.

Instead, molecular dynamics uses **numerical integration**.

Numerical integration approximates continuous atomic motion using a sequence of very small time steps.

---

# 18.10.18 Why Numerical Integration Is Necessary

Suppose we know the force acting on every atom at the current time.

From Newton's Second Law,

$$
\mathbf{a}(t)
=============

\frac{\mathbf{F}(t)}{m}.
$$

The acceleration tells us how the velocity changes,

but it does not directly provide the atom's future position.

Therefore, we must integrate the equations of motion.

Conceptually,

```text id="m3g_md_int1"
Force

↓

Acceleration

↓

Velocity

↓

Position
```

This sequence is repeated for every atom at every time step.

---

# 18.10.19 Continuous Motion vs Discrete Simulation

Real atomic motion is continuous.

```text id="m3g_md_int2"
Continuous Time

────────────────────────────►
```

A computer, however, evaluates the system only at discrete times.

```text id="m3g_md_int3"
t0

↓

t1

↓

t2

↓

t3

↓

t4
```

The simulation therefore approximates continuous motion using a finite sequence of snapshots.

The smaller the time step,

the more accurately the simulation reproduces the true trajectory.

---

# 18.10.20 Taylor Series Expansion

Most molecular dynamics integration algorithms are derived from the Taylor series.

Expanding the atomic position about the current time gives

$$
\mathbf{R}(t+\Delta t)
======================

\mathbf{R}(t)
+
\mathbf{v}(t)\Delta t
+
\frac12
\mathbf{a}(t)
\Delta t^2
+
O(\Delta t^3),
$$

where

* $\mathbf{R}$ is position,
* $\mathbf{v}$ is velocity,
* $\mathbf{a}$ is acceleration,
* $\Delta t$ is the time step.

Similarly,

the velocity satisfies

$$
\mathbf{v}(t+\Delta t)
======================

\mathbf{v}(t)
+
\mathbf{a}(t)\Delta t
+
O(\Delta t^2).
$$

These expressions form the mathematical foundation of nearly all MD integration schemes.

---

# 18.10.21 Requirements of a Good Integrator

A useful molecular dynamics integrator should satisfy several important properties.

It should

* accurately reproduce atomic trajectories,
* conserve total energy over long simulations,
* remain numerically stable,
* require minimal computational cost,
* avoid accumulation of large rounding errors.

An algorithm that is highly accurate but unstable is unsuitable.

Likewise, an extremely stable algorithm that is computationally expensive may become impractical for large simulations.

---

# 18.10.22 The Euler Method

The simplest integration algorithm is the **Euler method**.

The updates are

$$
\mathbf{v}_{t+\Delta t}
=======================

\mathbf{v}_t
+
\mathbf{a}_t
\Delta t,
$$

and

$$
\mathbf{R}_{t+\Delta t}
=======================

\mathbf{R}_t
+
\mathbf{v}_t
\Delta t.
$$

Conceptually,

```text id="m3g_md_int4"
Current Position

↓

Current Velocity

↓

Move Forward
```

Although easy to understand,

Euler integration is rarely used in molecular dynamics.

---

# 18.10.23 Why Euler Integration Fails

The Euler method accumulates numerical errors rapidly.

Suppose an atom oscillates around its equilibrium position.

Instead of maintaining constant total energy,

Euler integration gradually increases or decreases the energy.

Consequently,

the simulated trajectory drifts away from the true physical trajectory.

This instability becomes severe during long simulations.

Therefore,

modern molecular dynamics almost never uses Euler integration.

---

# 18.10.24 The Verlet Algorithm

One of the most influential molecular dynamics algorithms is the **Verlet algorithm**.

Instead of updating positions using only the current velocity,

Verlet uses both the current and previous positions.

The update equation is

$$
\mathbf{R}(t+\Delta t)
======================

## 2\mathbf{R}(t)

\mathbf{R}(t-\Delta t)
+
\mathbf{a}(t)
\Delta t^2.
$$

Interestingly,

the velocity does not appear explicitly in this equation.

---

# 18.10.25 Why Verlet Is More Stable

The Verlet algorithm possesses several attractive properties.

It

* conserves energy well,
* is time reversible,
* remains stable over long simulations,
* is simple to implement.

Because of these advantages,

Verlet became one of the standard algorithms in atomistic simulations.

---

# 18.10.26 Intuition Behind Verlet

Imagine predicting tomorrow's position of a moving object.

Instead of using only today's information,

you also remember yesterday's position.

This additional information provides a better estimate of the future trajectory.

Conceptually,

```text id="m3g_md_int5"
Yesterday

↓

Today

↓

Tomorrow
```

The algorithm therefore captures the smoothness of atomic motion more accurately than Euler integration.

---

# 18.10.27 Velocity Verlet Algorithm

Modern molecular dynamics simulations usually employ the **Velocity Verlet algorithm**.

Unlike the original Verlet method,

Velocity Verlet updates both

* positions,
* velocities,

explicitly.

The position update is

$$
\mathbf{R}(t+\Delta t)
======================

\mathbf{R}(t)
+
\mathbf{v}(t)\Delta t
+
\frac12
\mathbf{a}(t)
\Delta t^2.
$$

After computing the new forces,

the velocity is updated using

$$
\mathbf{v}(t+\Delta t)
======================

\mathbf{v}(t)
+
\frac12
\left[
\mathbf{a}(t)
+
\mathbf{a}(t+\Delta t)
\right]
\Delta t.
$$

Notice that the velocity uses both the old and the new accelerations.

This improves accuracy while maintaining excellent numerical stability.

---

# 18.10.28 Velocity Verlet Workflow

Each MD step proceeds as follows.

```text id="m3g_md_int6"
Current Position

↓

Current Velocity

↓

Current Forces

↓

Update Position

↓

Compute New Forces

↓

Update Velocity

↓

Next Step
```

This sequence is repeated throughout the simulation.

---

# 18.10.29 Why Velocity Verlet Is Widely Used

Velocity Verlet offers an excellent balance between

* accuracy,
* computational efficiency,
* numerical stability.

Its major advantages include

* second-order accuracy,
* good energy conservation,
* explicit velocities,
* straightforward implementation,
* compatibility with thermostats and barostats.

Consequently,

it has become the default integrator in many molecular dynamics packages.

---

# 18.10.30 Leapfrog Integration

Another popular algorithm is the **Leapfrog integrator**.

In this method,

positions and velocities are evaluated at slightly different times.

Specifically,

velocities are computed at

$$
t+\frac12\Delta t,
$$

while positions remain defined at

$$
t.
$$

Conceptually,

```text id="m3g_md_int7"
Position

t

↓

t + Δt

↓

t + 2Δt



Velocity

t + Δt/2

↓

t + 3Δt/2

↓

t + 5Δt/2
```

The velocities therefore "leap over" the positions, giving the method its name.

---

# 18.10.31 Comparison of Common Integrators

The major integration algorithms can be summarized as follows.

| Integrator      | Accuracy | Energy Conservation | Typical Use                     |
| --------------- | -------- | ------------------- | ------------------------------- |
| Euler           | Low      | Poor                | Educational demonstrations      |
| Verlet          | High     | Very Good           | Classical MD                    |
| Velocity Verlet | High     | Excellent           | Most modern MD simulations      |
| Leapfrog        | High     | Excellent           | Biomolecular and large-scale MD |

Among these, **Velocity Verlet** is the most commonly used with machine-learned interatomic potentials, including M3GNet.

---

# 18.10.32 Integration Within an M3GNet Simulation

The complete molecular dynamics cycle now becomes

```text id="m3g_md_int8"
Current Structure

↓

Graph Construction

↓

M3GNet

↓

Energy

↓

Forces

↓

Acceleration

↓

Velocity Verlet

↓

Updated Structure

↓

Next Time Step
```

Notice that M3GNet itself **does not integrate the equations of motion**.

Its responsibility is to predict

* energies,
* forces,
* stresses.

The integration algorithm then uses those forces to propagate the atomic trajectories through time.

---

# 18.10.33 Choosing the Time Step

The choice of time step is critical.

If the time step is too large,

atoms may move unrealistically far during a single update, leading to numerical instability and poor energy conservation.

If the time step is too small,

the simulation becomes unnecessarily expensive because many more force evaluations are required.

For atomistic simulations involving vibrations of light elements, time steps of approximately **1–2 fs** are commonly used. Systems containing hydrogen or very high-frequency vibrations often require even smaller time steps.

---

# 18.10.34 Long-Time Stability

One of the primary reasons for using Verlet-family algorithms is their excellent long-term behavior.

Although small numerical errors are unavoidable,

these integrators tend to keep the total energy nearly constant over very long simulations of isolated systems.

This property is essential because molecular dynamics studies often require

* hundreds of thousands,
* millions,
* or even tens of millions

of integration steps.

Without good long-term stability, physically meaningful simulations would not be possible.

---

## Transition to the Next Section

We now understand how molecular dynamics advances atoms from one time step to the next using numerical integration algorithms such as Velocity Verlet.

However, real simulations are usually performed under specific thermodynamic conditions—for example, **constant temperature** or **constant pressure**. In the next section, we will study **statistical ensembles, thermostats, and barostats**, and explain how M3GNet-based molecular dynamics can accurately simulate materials under realistic experimental conditions such as room temperature or elevated pressure.

# 18.10.35 Statistical Ensembles in Molecular Dynamics

Thus far, we have discussed molecular dynamics as a purely mechanical simulation based on Newton's laws.

However, real materials are not isolated from their surroundings.

A crystal in an experiment may

* exchange heat with the environment,
* experience external pressure,
* change its volume,
* or exchange energy with neighboring materials.

Therefore, realistic simulations must reproduce not only atomic motion but also the correct **thermodynamic conditions**.

This is accomplished using **statistical ensembles**.

An ensemble specifies which macroscopic physical quantities remain constant throughout the simulation.

Examples include

* temperature,
* pressure,
* volume,
* total energy,
* number of atoms.

Different experimental situations require different ensembles.

---

# 18.10.36 What is a Statistical Ensemble?

A statistical ensemble is a large collection of hypothetical copies of the same physical system.

Each copy

* contains the same material,
* obeys the same physical laws,
* satisfies the same thermodynamic constraints,

but the exact atomic positions and velocities differ slightly.

Instead of analyzing one microscopic configuration,

statistical mechanics studies the average behavior of this ensemble.

In molecular dynamics,

we generate a trajectory whose long-time average approximates the ensemble average.

---

# 18.10.37 Microscopic and Macroscopic Properties

A material can be described from two different viewpoints.

### Microscopic description

Focuses on

* atomic positions,
* atomic velocities,
* individual forces,
* atomic trajectories.

### Macroscopic description

Focuses on measurable quantities such as

* temperature,
* pressure,
* density,
* internal energy,
* heat capacity.

One of the major goals of molecular dynamics is to calculate macroscopic properties from microscopic atomic motion.

---

# 18.10.38 Why Ensembles Matter

Suppose we wish to simulate silicon at

```text id="m3g_ensemble1"
300 K
```

If no temperature control is applied,

the system may gradually heat up or cool down due to numerical errors or external work.

The simulation would no longer represent silicon at room temperature.

Similarly,

suppose we wish to model a material under

```text id="m3g_ensemble2"
1 atm
```

Keeping the volume fixed would prevent the crystal from expanding or contracting in response to pressure.

Therefore,

the appropriate ensemble must be chosen according to the physical experiment being modeled.

---

# 18.10.39 The NVE Ensemble

The simplest molecular dynamics ensemble is the **NVE ensemble**.

The conserved quantities are

* Number of atoms (**N**),
* Volume (**V**),
* Total Energy (**E**).

Conceptually,

```text id="m3g_ensemble3"
N

Constant

+

V

Constant

+

E

Constant
```

Since no heat enters or leaves the system,

the NVE ensemble represents an isolated system.

---

# 18.10.40 Characteristics of the NVE Ensemble

In an ideal NVE simulation,

the total energy remains constant,

$$
E_{\text{total}}
================

K
+
E_{\text{pot}}
==============

\text{constant}.
$$

Although kinetic and potential energy continuously exchange,

their sum remains unchanged.

For example,

```text id="m3g_ensemble4"
Potential Energy

↓

Kinetic Energy

↓

Potential Energy

↓

Kinetic Energy
```

The total energy remains nearly constant throughout the simulation.

---

# 18.10.41 Advantages of NVE

The NVE ensemble has several important advantages.

* Simple implementation.
* No artificial temperature control.
* Conserves energy naturally.
* Suitable for studying intrinsic dynamics.

It is often used to verify the numerical stability of an interatomic potential.

If the total energy drifts significantly during an NVE simulation,

this may indicate

* an excessively large time step,
* poor numerical integration,
* inaccurate force calculations.

---

# 18.10.42 Limitations of NVE

Although physically meaningful,

the NVE ensemble does not represent many laboratory experiments.

Real materials frequently exchange heat with their surroundings.

Consequently,

their temperature remains approximately constant rather than their total energy.

To simulate such conditions,

another ensemble is required.

---

# 18.10.43 The NVT Ensemble

The **NVT ensemble** keeps constant

* Number of atoms (**N**),
* Volume (**V**),
* Temperature (**T**).

Conceptually,

```text id="m3g_ensemble5"
N

Constant

+

V

Constant

+

Temperature

Constant
```

Unlike NVE,

energy is allowed to fluctuate because heat can flow between the simulated system and an imaginary heat reservoir.

---

# 18.10.44 Why Constant Temperature is Important

Many material properties are temperature dependent.

Examples include

* thermal expansion,
* diffusion,
* phase transitions,
* ionic conductivity,
* defect migration.

To investigate these phenomena,

the simulation temperature must remain close to the desired value.

For example,

```text id="m3g_ensemble6"
300 K

600 K

1000 K
```

Each temperature corresponds to a different distribution of atomic velocities.

Maintaining these temperatures requires a **thermostat**.

---

# 18.10.45 The Role of a Thermostat

A thermostat is an algorithm that regulates the kinetic energy of the system.

It periodically adjusts atomic velocities so that the average temperature remains close to the target value.

Conceptually,

```text id="m3g_ensemble7"
Atomic Motion

↓

Temperature

↓

Thermostat

↓

Velocity Adjustment

↓

Target Temperature
```

Importantly,

the thermostat does not determine the atomic forces.

Those forces continue to be predicted by M3GNet.

---

# 18.10.46 Temperature and Kinetic Energy

Temperature is directly related to the average kinetic energy of the atoms.

For a classical system,

the equipartition theorem gives

$$
\left<
K
\right>
=======

\frac32
Nk_B T,
$$

where

* $N$ is the number of atoms,
* $k_B$ is the Boltzmann constant,
* $T$ is the absolute temperature.

This equation shows that increasing the temperature increases the average atomic velocities.

---

# 18.10.47 Maxwell–Boltzmann Distribution

Atoms in a material do not all move at the same speed.

Instead,

their velocities follow the **Maxwell–Boltzmann distribution**.

At low temperatures,

most atoms move slowly.

At higher temperatures,

the distribution broadens,

and a larger fraction of atoms possess high velocities.

Conceptually,

```text id="m3g_ensemble8"
Number of Atoms

│      /\

│     /  \

│    /    \

│___/______\____

        Velocity
```

As temperature increases,

the peak shifts toward higher velocities and becomes wider.

---

# 18.10.48 The NPT Ensemble

Many experiments are performed not at constant volume but at constant pressure.

The corresponding ensemble is the **NPT ensemble**.

The conserved quantities are

* Number of atoms (**N**),
* Pressure (**P**),
* Temperature (**T**).

Conceptually,

```text id="m3g_ensemble9"
N

Constant

+

Pressure

Constant

+

Temperature

Constant
```

Unlike NVT,

the simulation cell is allowed to change size and shape.

---

# 18.10.49 Why Constant Pressure Matters

Consider heating a crystal.

If the simulation volume is fixed,

the crystal cannot expand.

This produces unrealistic internal stresses.

In contrast,

an NPT simulation allows

* lattice expansion,
* lattice contraction,
* density changes,

just as occurs in real experiments.

This is particularly important for

* thermal expansion,
* phase transitions,
* high-pressure materials,
* liquids.

---

# 18.10.50 Thermostats and Barostats Together

In an NPT simulation,

both temperature and pressure must be controlled.

This requires two algorithms.

```text id="m3g_ensemble10"
Atomic Motion

↓

Thermostat

↓

Temperature Control

+

Barostat

↓

Pressure Control

↓

Realistic Conditions
```

The thermostat regulates atomic velocities,

while the barostat adjusts the simulation cell dimensions.

---

# 18.10.51 Which Ensemble Should Be Used?

The choice depends on the scientific problem.

| Ensemble | Constant Quantities | Typical Applications                                                  |
| -------- | ------------------- | --------------------------------------------------------------------- |
| NVE      | N, V, E             | Energy conservation studies, validation                               |
| NVT      | N, V, T             | Diffusion, equilibrium at fixed temperature                           |
| NPT      | N, P, T             | Thermal expansion, phase transitions, realistic laboratory conditions |

There is no universally "best" ensemble.

The correct choice depends entirely on the physical process being investigated.

---

# 18.10.52 M3GNet Within Different Ensembles

Regardless of the ensemble,

the role of M3GNet remains unchanged.

At every MD step,

M3GNet predicts

* total energy,
* atomic forces,
* stress tensor (when required).

The thermostat and barostat then modify

* atomic velocities,
* simulation cell,

to enforce the desired thermodynamic conditions.

Thus,

the machine-learning potential provides the underlying physics,

while the ensemble algorithms control the macroscopic state of the system.

---

## Transition to the Next Section

We have introduced the most important statistical ensembles—NVE, NVT, and NPT—and seen how they represent different experimental conditions. However, we have only briefly mentioned thermostats and barostats.

In the next section, we will study the **major thermostat algorithms**—including the **Berendsen**, **Andersen**, **Nosé–Hoover**, and **Langevin** thermostats—as well as common **barostats**, explaining their physical principles, mathematical foundations, advantages, limitations, and practical use with M3GNet-based molecular dynamics simulations.

# 18.10.53 Thermostats in Molecular Dynamics

In the previous section, we learned that the **NVT ensemble** requires the temperature to remain approximately constant throughout the simulation.

However, Newton's equations of motion alone cannot guarantee this.

If atoms simply evolve according to

$$
\mathbf{F}
==========

m\mathbf{a},
$$

the total energy is conserved (assuming an ideal isolated system), but the temperature may fluctuate or drift depending on the simulation conditions.

Therefore, molecular dynamics introduces a **thermostat**.

A thermostat is an algorithm that allows the simulated system to exchange heat with an imaginary thermal reservoir.

Its purpose is not to change the underlying physics but to ensure that the average temperature remains close to a desired target.

---

# 18.10.54 What Does a Thermostat Control?

A thermostat does **not** directly modify

* atomic positions,
* atomic forces,
* potential energy.

Instead, it regulates the **atomic velocities**.

Since temperature is proportional to the average kinetic energy,

controlling the velocities automatically controls the temperature.

Conceptually,

```text id="m3g_thermo1"
Atomic Velocities

↓

Kinetic Energy

↓

Temperature

↓

Thermostat

↓

Adjusted Velocities
```

The M3GNet model continues to predict forces exactly as before.

The thermostat simply modifies how the atoms move through time.

---

# 18.10.55 Temperature Fluctuations

Even in a perfectly equilibrated simulation,

the instantaneous temperature is not exactly constant.

Instead,

it fluctuates around the desired value.

For example,

during a

```text id="m3g_thermo2"
300 K
```

simulation,

the instantaneous temperature may be

| Time | Temperature (K) |
| ---- | --------------: |
| 0 ps |             298 |
| 1 ps |             304 |
| 2 ps |             301 |
| 3 ps |             297 |
| 4 ps |             303 |

Such fluctuations are physically expected.

The thermostat ensures that the **average** temperature remains close to the target value over time.

---

# 18.10.56 Velocity Rescaling

The simplest thermostat is **velocity rescaling**.

Suppose the current temperature is

$$
T_{\text{current}},
$$

while the desired temperature is

$$
T_{\text{target}}.
$$

The atomic velocities are multiplied by a scaling factor,

$$
\lambda
=======

\sqrt{
\frac
{T_{\text{target}}}
{T_{\text{current}}}
}.
$$

The updated velocities become

$$
\mathbf{v}'
===========

\lambda
\mathbf{v}.
$$

If the temperature is too high,

the velocities decrease.

If the temperature is too low,

the velocities increase.

---

# 18.10.57 Advantages and Limitations of Velocity Rescaling

Velocity rescaling is

* extremely simple,
* computationally inexpensive,
* useful during initial equilibration.

However,

it has an important drawback.

Because the velocities are modified artificially,

the resulting trajectory does not correctly sample the canonical (NVT) ensemble.

Therefore,

simple velocity rescaling is generally not recommended for production simulations.

---

# 18.10.58 The Berendsen Thermostat

One of the earliest practical thermostats is the **Berendsen thermostat**.

Instead of forcing the temperature to change instantaneously,

it relaxes the temperature gradually toward the target value.

Conceptually,

```text id="m3g_thermo3"
Current Temperature

↓

Slight Velocity Adjustment

↓

Closer to Target

↓

Repeat
```

This smooth relaxation improves numerical stability compared with simple velocity rescaling.

---

# 18.10.59 Temperature Relaxation

The Berendsen thermostat introduces a relaxation time,

usually denoted

$$
\tau.
$$

Large values of

$$
\tau
$$

produce weak temperature control.

Small values produce stronger coupling to the heat bath.

Thus,

the user can determine how rapidly the system approaches thermal equilibrium.

---

# 18.10.60 Advantages of the Berendsen Thermostat

The Berendsen thermostat offers several advantages.

* Stable.
* Simple.
* Rapid equilibration.
* Easy implementation.

Consequently,

it is frequently used during the early stages of molecular dynamics simulations.

---

# 18.10.61 Limitations of the Berendsen Thermostat

Despite its popularity,

the Berendsen thermostat has an important limitation.

It suppresses natural temperature fluctuations.

As a result,

the simulation does **not** generate the correct canonical ensemble.

Therefore,

many researchers use the Berendsen thermostat only during equilibration before switching to a more rigorous thermostat for production runs.

---

# 18.10.62 The Andersen Thermostat

The **Andersen thermostat** introduces stochastic collisions between atoms and an imaginary heat bath.

At random intervals,

selected atoms receive completely new velocities drawn from the Maxwell–Boltzmann distribution.

Conceptually,

```text id="m3g_thermo4"
Atomic Motion

↓

Random Collision

↓

New Velocity

↓

Continue Simulation
```

These random collisions mimic energy exchange with the environment.

---

# 18.10.63 Advantages of the Andersen Thermostat

The Andersen thermostat correctly samples the canonical ensemble.

It is

* conceptually simple,
* statistically rigorous,
* straightforward to implement.

Because velocities are repeatedly randomized,

the temperature remains close to the desired value.

---

# 18.10.64 Limitations of the Andersen Thermostat

Random velocity reassignment interrupts the natural motion of atoms.

Consequently,

properties depending on continuous particle motion,

such as

* diffusion coefficients,
* viscosity,
* transport phenomena,

may become inaccurate.

Therefore,

the Andersen thermostat is less suitable when dynamical properties are of primary interest.

---

# 18.10.65 The Nosé–Hoover Thermostat

The **Nosé–Hoover thermostat** is one of the most widely used thermostats in atomistic simulations.

Instead of applying random collisions,

it introduces an additional dynamical variable representing an ideal heat bath.

The thermostat becomes part of the equations of motion themselves.

Conceptually,

```text id="m3g_thermo5"
Atoms

↓

Nosé–Hoover Variable

↓

Temperature Control

↓

Continuous Dynamics
```

Unlike stochastic methods,

the atomic trajectories remain smooth.

---

# 18.10.66 Why Nosé–Hoover Is Popular

Nosé–Hoover possesses several important advantages.

* Generates the correct NVT ensemble.
* Preserves continuous atomic trajectories.
* Suitable for equilibrium simulations.
* Widely implemented in atomistic simulation software.

Because the thermostat becomes part of the equations of motion,

temperature control occurs naturally rather than through abrupt velocity changes.

---

# 18.10.67 Limitations of Nosé–Hoover

Although highly effective,

Nosé–Hoover is not perfect.

For some systems,

especially small systems,

the thermostat may not sample phase space efficiently.

To overcome this limitation,

many simulation packages implement **Nosé–Hoover chains**, which couple multiple thermostat variables together.

These extended methods improve sampling while preserving the desirable properties of the original algorithm.

---

# 18.10.68 The Langevin Thermostat

Another widely used approach is the **Langevin thermostat**.

This thermostat modifies Newton's equation by adding

* a frictional force,
* a random force.

Conceptually,

the equation becomes

$$
m
\mathbf{a}
==========

## \mathbf{F}

\gamma
\mathbf{v}
+
\mathbf{R}(t),
$$

where

* $\gamma$ is the friction coefficient,
* $\mathbf{R}(t)$ is a random force representing collisions with the surrounding environment.

The random force continuously injects thermal energy,

while the friction term removes energy.

Together,

they maintain the desired temperature.

---

# 18.10.69 Physical Interpretation of the Langevin Thermostat

The Langevin thermostat is particularly useful for modeling systems immersed in a surrounding medium.

Imagine a nanoparticle suspended in a liquid.

The surrounding solvent molecules continuously

* collide with the particle,
* transfer momentum,
* dissipate energy.

The friction and random-force terms imitate these microscopic interactions without explicitly simulating every solvent molecule.

---

# 18.10.70 Comparison of Common Thermostats

The most frequently used thermostat algorithms are summarized below.

| Thermostat         | Temperature Control     | Ensemble Accuracy | Typical Applications                              |
| ------------------ | ----------------------- | ----------------- | ------------------------------------------------- |
| Velocity Rescaling | Direct scaling          | Poor              | Initial equilibration                             |
| Berendsen          | Weak coupling           | Approximate       | Fast equilibration                                |
| Andersen           | Random collisions       | Good              | Equilibrium sampling                              |
| Nosé–Hoover        | Extended dynamics       | Excellent         | Production simulations                            |
| Langevin           | Friction + random force | Excellent         | Biological systems, diffusion, noisy environments |

Each thermostat has strengths and limitations.

The optimal choice depends on the scientific objective rather than on computational speed alone.

---

# 18.10.71 Thermostats with M3GNet

When using M3GNet,

the thermostat does **not** replace the neural network.

The workflow remains

```text id="m3g_thermo6"
Atomic Positions

↓

M3GNet

↓

Forces

↓

Integrator

↓

Thermostat

↓

Updated Velocities

↓

Next Time Step
```

M3GNet is responsible for computing the interatomic forces.

The thermostat modifies the velocities after integration so that the desired temperature is maintained.

Thus,

the neural network determines the microscopic interactions,

while the thermostat controls the macroscopic thermal environment.

---

# 18.10.72 Choosing a Thermostat in Practice

In practical materials simulations,

the following strategy is commonly adopted.

* **Berendsen thermostat**: Rapid equilibration of a newly initialized system.
* **Nosé–Hoover thermostat**: Long equilibrium production runs requiring correct canonical sampling.
* **Langevin thermostat**: Simulations where stochastic interactions or enhanced stability are beneficial, such as diffusion studies or systems coupled to an implicit environment.
* **Andersen thermostat**: Problems where accurate canonical sampling is more important than preserving continuous dynamical trajectories.

Researchers often begin with a Berendsen thermostat to bring the system to the target temperature and then switch to a Nosé–Hoover thermostat before collecting production data.

---

## Transition to the Next Section

We have now explored the major thermostat algorithms used to control temperature during molecular dynamics simulations. However, many real experiments are conducted not only at constant temperature but also at **constant pressure**.

In the next section, we will study **barostats**, including the **Berendsen**, **Parrinello–Rahman**, and related pressure-control algorithms, and explain how they adjust the simulation cell while working together with M3GNet's predicted stress tensor to perform realistic NPT molecular dynamics simulations.

# 18.10.73 Barostats in Molecular Dynamics

In the previous section, we studied **thermostats**, which maintain the temperature of a molecular dynamics simulation.

Many real experiments, however, are performed not only at constant temperature but also at constant **pressure**.

For example,

* materials are synthesized at atmospheric pressure,
* minerals form under extremely high pressures inside the Earth,
* batteries expand and contract during charging,
* crystals change their lattice parameters as temperature changes.

To simulate these situations realistically, molecular dynamics must allow the simulation cell to change its size and, in some cases, its shape.

This is the role of a **barostat**.

A barostat controls the pressure by continuously adjusting the simulation cell during the molecular dynamics simulation.

---

# 18.10.74 Why Pressure Control Is Necessary

Suppose a crystal is heated from

```text id="m3g_baro1"
300 K
```

to

```text id="m3g_baro2"
900 K
```

In reality,

the lattice expands because atomic vibrations become larger.

If the simulation cell is artificially kept fixed,

the crystal cannot expand.

Instead,

internal stresses build up,

leading to unrealistic results.

Allowing the simulation cell to change naturally removes these artificial stresses.

---

# 18.10.75 Internal and External Pressure

Pressure in molecular dynamics has two components.

### Internal pressure

Generated by

* atomic interactions,
* thermal motion,
* lattice strain.

### External pressure

Specified by the user.

Examples include

```text id="m3g_baro3"
0 GPa

1 GPa

10 GPa

100 GPa
```

The purpose of the barostat is to adjust the simulation cell until

$$
P_{\mathrm{internal}}
\approx
P_{\mathrm{external}}.
$$

---

# 18.10.76 Pressure and Stress

Pressure is closely related to the stress tensor.

For isotropic systems,

the pressure is

$$
P
=

*

\frac13
\left(
\sigma_{xx}
+
\sigma_{yy}
+
\sigma_{zz}
\right).
$$

The negative sign arises because compression corresponds to positive pressure but negative normal stress under the common stress convention used in atomistic simulations.

When M3GNet predicts the stress tensor,

the barostat uses this information to determine how the simulation cell should change.

---

# 18.10.77 How a Barostat Works

Conceptually,

a barostat operates as follows.

```text id="m3g_baro4"
Current Cell

↓

Compute Stress

↓

Compare with Target Pressure

↓

Expand or Contract Cell

↓

Repeat
```

If the internal pressure is

too high,

the simulation cell expands.

If the internal pressure is

too low,

the cell contracts.

This process continues throughout the simulation.

---

# 18.10.78 The Berendsen Barostat

The simplest pressure-control algorithm is the **Berendsen barostat**.

Like the Berendsen thermostat,

it gradually relaxes the pressure toward the desired value.

Instead of abruptly changing the lattice,

it scales the simulation cell smoothly.

Conceptually,

```text id="m3g_baro5"
Current Pressure

↓

Small Cell Adjustment

↓

Closer to Target

↓

Repeat
```

---

# 18.10.79 Advantages of the Berendsen Barostat

The Berendsen barostat offers several practical advantages.

* Stable.
* Easy to implement.
* Rapid equilibration.
* Smooth volume changes.

Because of these properties,

it is frequently used during the early stages of molecular dynamics simulations.

---

# 18.10.80 Limitations of the Berendsen Barostat

Despite its usefulness,

the Berendsen barostat has the same limitation as the Berendsen thermostat.

It does not generate the correct statistical fluctuations expected for the NPT ensemble.

Therefore,

while excellent for equilibration,

it is generally avoided during production simulations that require rigorous thermodynamic sampling.

---

# 18.10.81 The Parrinello–Rahman Barostat

One of the most important pressure-control algorithms is the **Parrinello–Rahman barostat**.

Unlike simpler methods,

it allows not only the volume but also the **shape** of the simulation cell to change.

This capability is essential for studying

* crystal phase transitions,
* structural transformations,
* ferroelastic materials,
* anisotropic mechanical deformation.

Conceptually,

```text id="m3g_baro6"
Initial Cell

↓

Expand

↓

Shear

↓

Rotate

↓

New Cell
```

The simulation cell evolves dynamically according to the internal stress.

---

# 18.10.82 Why Cell Shape Matters

Many materials do not deform uniformly.

For example,

during a structural phase transition,

a cubic crystal may transform into

* tetragonal,
* orthorhombic,
* monoclinic,

or another crystal system.

Such transformations involve changes not only in lattice lengths but also in lattice angles.

A simple volume-scaling barostat cannot describe these changes.

The Parrinello–Rahman approach can.

---

# 18.10.83 Coupling Between Stress and Cell Motion

During NPT molecular dynamics,

two types of motion occur simultaneously.

First,

atoms move within the unit cell.

Second,

the unit cell itself changes.

Conceptually,

```text id="m3g_baro7"
Atomic Motion

+

Cell Motion

↓

New Structure

↓

Repeat
```

The atomic coordinates are therefore continuously expressed relative to a changing lattice.

---

# 18.10.84 Role of the Stress Tensor

At every molecular dynamics step,

M3GNet predicts

* total energy,
* atomic forces,
* stress tensor.

The stress tensor then determines

* whether the cell should expand,
* whether it should contract,
* whether shear deformation is required.

Consequently,

stress prediction is essential for realistic NPT molecular dynamics.

Without stress,

pressure-controlled simulations would not be possible.

---

# 18.10.85 Pressure Equilibration

Suppose the target pressure is

```text id="m3g_baro8"
0 GPa
```

The simulation might evolve as

| Time (ps) | Pressure (GPa) |
| --------: | -------------: |
|         0 |            3.5 |
|         2 |            1.8 |
|         4 |            0.9 |
|         6 |            0.3 |
|         8 |            0.1 |
|        10 |            0.0 |

Initially,

the crystal is highly compressed.

The barostat gradually adjusts the simulation cell until the desired pressure is reached.

---

# 18.10.86 Volume Evolution

As the pressure approaches equilibrium,

the simulation cell volume also changes.

For example,

| Time (ps) | Volume (Å³) |
| --------: | ----------: |
|         0 |       150.2 |
|         2 |       153.8 |
|         4 |       156.0 |
|         6 |       157.1 |
|         8 |       157.6 |
|        10 |       157.7 |

Notice that

* pressure decreases,
* volume stabilizes,

simultaneously.

This indicates that the system has reached mechanical equilibrium.

---

# 18.10.87 NPT Molecular Dynamics Workflow

Combining M3GNet,

a thermostat,

and a barostat produces the following workflow.

```text id="m3g_baro9"
Atomic Positions

↓

M3GNet

↓

Energy

↓

Forces

↓

Stress

↓

Velocity Verlet

↓

Thermostat

↓

Barostat

↓

Updated Cell

↓

Next Time Step
```

Every component has a specific responsibility.

* M3GNet predicts the microscopic physics.
* The integrator advances the atoms in time.
* The thermostat regulates temperature.
* The barostat regulates pressure.

---

# 18.10.88 Choosing a Barostat

Several pressure-control algorithms are available.

| Barostat             | Characteristics                                      | Typical Applications                      |
| -------------------- | ---------------------------------------------------- | ----------------------------------------- |
| Berendsen            | Rapid equilibration                                  | Initial NPT equilibration                 |
| Parrinello–Rahman    | Correct cell fluctuations and shape changes          | Production simulations, phase transitions |
| Monte Carlo Barostat | Random volume moves accepted by statistical criteria | Monte Carlo and hybrid simulations        |

Among these,

the Parrinello–Rahman barostat is one of the most widely used for crystalline materials because it naturally accommodates anisotropic lattice deformations.

---

# 18.10.89 Practical Considerations

Pressure control should be applied carefully.

If the pressure coupling is too strong,

the simulation cell may oscillate excessively.

If it is too weak,

equilibration becomes very slow.

Similarly,

the barostat relaxation time should generally be larger than the thermostat relaxation time because the lattice evolves more slowly than atomic velocities.

Careful selection of these parameters improves numerical stability and yields more realistic structural fluctuations.

---

# 18.10.90 M3GNet-Based NPT Simulations

The ability of M3GNet to predict both **forces** and the full **stress tensor** makes it particularly well suited for NPT molecular dynamics.

Compared with first-principles molecular dynamics,

the workflow is conceptually identical,

but the force and stress evaluations are dramatically faster.

This enables simulations involving

* larger supercells,
* longer time scales,
* more extensive statistical sampling,

while retaining accuracy close to Density Functional Theory for systems represented in the training data.

---

## Transition to the Next Section

We have now developed all of the essential components required for molecular dynamics: force prediction using M3GNet, numerical integration, thermostats, and barostats.

In the next section, we will bring these components together by constructing a **complete M3GNet molecular dynamics simulation in Python**, explaining every line of code, initializing temperatures, selecting ensembles, running the simulation, storing trajectories, and analyzing the resulting atomic motion in a practical research workflow.

# 18.10.91 Complete Molecular Dynamics Workflow with M3GNet

Having studied

* Newton's equations of motion,
* numerical integration,
* statistical ensembles,
* thermostats,
* barostats,

we are now ready to perform a complete molecular dynamics simulation using M3GNet.

In this section, we will examine the entire workflow used in practical research.

Rather than simply presenting a short script, we will carefully explain every stage of the simulation and the role of each software component.

The complete workflow is

```text id="m3g_md_workflow1"
Crystal Structure

↓

Load M3GNet

↓

Create Calculator

↓

Initialize Velocities

↓

Choose Ensemble

↓

Run Molecular Dynamics

↓

Save Trajectory

↓

Analyze Results
```

Although the Python code may occupy only a few dozen lines, each line represents a sophisticated sequence of physical and computational operations.

---

# 18.10.92 Step 1 — Import Required Libraries

A typical M3GNet molecular dynamics simulation requires several Python packages.

Conceptually,

```python id="m3gnet_code49"
from ase.io import read
from ase.io import write

import matgl
```

Additional imports are generally required for

* molecular dynamics,
* velocity initialization,
* thermostats,
* trajectory storage,
* logging.

The exact import statements depend on the installed version of ASE and MatGL.

Because both projects continue to evolve, the API may differ slightly between releases.

---

# 18.10.93 Responsibilities of Each Package

Each package performs a different task.

| Package | Responsibility             |
| ------- | -------------------------- |
| ASE     | Simulation framework       |
| MatGL   | M3GNet neural network      |
| PyTorch | Neural-network computation |
| NumPy   | Numerical operations       |

ASE manages the molecular dynamics simulation,

while M3GNet predicts

* energy,
* forces,
* stress.

PyTorch performs the neural-network calculations behind the scenes.

---

# 18.10.94 Step 2 — Read the Crystal Structure

The simulation begins by reading a crystal structure.

```python id="m3gnet_code50"
atoms = read("Si.cif")
```

The resulting object contains

* lattice vectors,
* atomic positions,
* chemical species,
* periodic boundary conditions.

At this stage,

the structure contains no velocities.

---

# 18.10.95 Step 3 — Load the M3GNet Model

Next,

the pretrained M3GNet model is loaded.

Conceptually,

```python id="m3gnet_code51"
model = ...
```

The exact command depends on

* the installed MatGL version,
* the specific pretrained model,
* or a custom fine-tuned checkpoint.

Regardless of how it is loaded,

the model is capable of predicting

* energy,
* forces,
* stress.

---

# 18.10.96 Step 4 — Create the ASE Calculator

The neural network must now be connected to ASE.

Conceptually,

```python id="m3gnet_code52"
calculator = ...
```

The calculator converts

```text id="m3g_md_workflow2"
Atomic Structure

↓

Graph

↓

M3GNet

↓

Energy

Forces

Stress
```

ASE communicates only with the calculator.

It never interacts directly with the neural network.

---

# 18.10.97 Step 5 — Attach the Calculator

The calculator is attached to the structure.

```python id="m3gnet_code53"
atoms.calc = calculator
```

After this assignment,

the atoms object can immediately compute

```python id="m3gnet_code54"
atoms.get_potential_energy()

atoms.get_forces()

atoms.get_stress()
```

No molecular dynamics has yet begun.

The structure has merely been connected to the computational engine.

---

# 18.10.98 Step 6 — Assign Initial Velocities

Molecular dynamics requires both

* positions,
* velocities.

A relaxed crystal contains only positions.

Therefore,

initial velocities must be assigned.

These velocities are usually sampled from the Maxwell–Boltzmann distribution.

Conceptually,

```text id="m3g_md_workflow3"
Target Temperature

↓

Random Velocities

↓

Correct Distribution
```

The average kinetic energy then corresponds to the desired temperature.

---

# 18.10.99 Maxwell–Boltzmann Initialization

Suppose the simulation temperature is

```text id="m3g_md_workflow4"
300 K
```

The initialization algorithm

1. generates random velocities,
2. scales them appropriately,
3. ensures the correct average kinetic energy.

Initially,

individual atoms may move in completely different directions,

but the overall velocity distribution satisfies statistical mechanics.

---

# 18.10.100 Removing Net Momentum

Random initialization may produce a small net translation of the entire crystal.

For example,

every atom may acquire a slight velocity toward the positive

$x$

direction.

This causes the whole simulation box to drift.

To prevent this,

the center-of-mass momentum is removed.

Conceptually,

```text id="m3g_md_workflow5"
Random Velocities

↓

Average Momentum

↓

Subtract Average

↓

Zero Net Motion
```

After this correction,

the crystal vibrates without translating through space.

---

# 18.10.101 Step 7 — Select the Ensemble

The desired thermodynamic ensemble is now chosen.

Typical options include

* NVE,
* NVT,
* NPT.

The choice depends entirely on the scientific objective.

Examples include

| Ensemble | Example Application            |
| -------- | ------------------------------ |
| NVE      | Energy conservation            |
| NVT      | Diffusion at fixed temperature |
| NPT      | Thermal expansion              |

The molecular dynamics algorithm itself remains largely unchanged.

Only the thermostat and barostat differ.

---

# 18.10.102 Step 8 — Choose the Time Step

The time step determines how far atoms move during each integration step.

Typical values are

```text id="m3g_md_workflow6"
0.5 fs

1 fs

2 fs
```

A smaller time step

* improves numerical accuracy,
* increases computational cost.

A larger time step

* reduces computational cost,
* increases the risk of numerical instability.

The appropriate value depends on

* atomic masses,
* bond stiffness,
* simulation temperature,
* material type.

---

# 18.10.103 Step 9 — Create the Molecular Dynamics Object

The molecular dynamics engine combines

* the integrator,
* the thermostat,
* the barostat (if required),
* the calculator.

Conceptually,

```python id="m3gnet_code55"
md = ...
```

Internally,

this object manages

* force evaluation,
* integration,
* trajectory generation,
* temperature control,
* pressure control.

---

# 18.10.104 Internal Workflow During One Time Step

Each molecular dynamics step consists of several operations.

```text id="m3g_md_workflow7"
Current Positions

↓

Graph Construction

↓

M3GNet

↓

Energy

↓

Forces

↓

Velocity Verlet

↓

Thermostat

↓

Barostat

↓

Updated Positions
```

Notice that

M3GNet performs only one task:

predicting the physics.

The remaining operations are handled by ASE.

---

# 18.10.105 Step 10 — Run the Simulation

The simulation is started by specifying the number of integration steps.

Conceptually,

```python id="m3gnet_code56"
md.run(
    steps=10000
)
```

If the time step is

```text id="m3g_md_workflow8"
1 fs
```

then

```text id="m3g_md_workflow9"
10000 steps
```

correspond to

```text id="m3g_md_workflow10"
10 ps
```

of simulated physical time.

Longer simulations simply require more integration steps.

---

# 18.10.106 Saving the Trajectory

Rather than saving only the final structure,

molecular dynamics stores every configuration.

Conceptually,

```text id="m3g_md_workflow11"
Step 1

↓

Step 2

↓

Step 3

↓

⋮

↓

Final Step
```

The resulting trajectory contains

* atomic positions,
* velocities,
* energies,
* simulation time.

Trajectory files allow the complete atomic motion to be reconstructed.

---

# 18.10.107 Recording Simulation Data

In addition to trajectories,

researchers usually record

* temperature,
* potential energy,
* kinetic energy,
* total energy,
* pressure,
* volume.

A typical simulation log may appear as

| Time (ps) | Temperature (K) | Total Energy (eV) |
| --------: | --------------: | ----------------: |
|         0 |           300.2 |          -1256.84 |
|         2 |           299.8 |          -1256.79 |
|         4 |           301.1 |          -1256.82 |
|         6 |           300.5 |          -1256.80 |
|         8 |           299.9 |          -1256.81 |

Such logs are essential for verifying that the simulation has reached equilibrium and remains stable.

---

# 18.10.108 Monitoring Equilibration

The earliest portion of an MD simulation is known as the **equilibration period**.

During this stage,

the system adjusts to the desired thermodynamic conditions.

Researchers generally avoid collecting scientific data during equilibration because

* the temperature may still be changing,
* the pressure may not yet have stabilized,
* the structure may still be relaxing.

Only after these quantities fluctuate around stable average values is the system considered equilibrated.

---

# 18.10.109 Production Simulation

After equilibration,

the simulation enters the **production phase**.

This is the portion used for scientific analysis.

Typical quantities calculated from production trajectories include

* diffusion coefficients,
* radial distribution functions,
* mean squared displacement,
* thermal expansion,
* elastic properties,
* vibrational behavior.

The production trajectory should be sufficiently long to ensure statistically reliable averages.

---

# 18.10.110 Practical Workflow Summary

The complete research workflow can now be summarized as

```text id="m3g_md_workflow12"
Build Crystal

↓

Load M3GNet

↓

Attach Calculator

↓

Assign Velocities

↓

Choose Ensemble

↓

Run Molecular Dynamics

↓

Equilibrate

↓

Production Run

↓

Analyze Trajectory

↓

Calculate Material Properties
```

This sequence forms the basis of most practical molecular dynamics studies using machine-learned interatomic potentials.

---

# 18.10.111 Advantages of M3GNet Molecular Dynamics

Compared with first-principles molecular dynamics,

M3GNet-based MD provides several significant advantages.

* Force evaluations are dramatically faster.
* Larger simulation cells can be studied.
* Longer simulation times become feasible.
* Thermal and mechanical properties can be investigated efficiently.
* Near-DFT accuracy is achieved for systems represented in the training data.

These advantages make M3GNet an attractive tool for large-scale atomistic simulations that would otherwise be computationally prohibitive.

---

## Transition to Chapter 18.11

We have now completed the core theory and practical workflow of **M3GNet-based molecular dynamics**, from force prediction and numerical integration to thermostats, barostats, and complete simulation pipelines.

In the next chapter, we will explore **applications of M3GNet in materials science**, demonstrating how this framework is used in modern research for crystal structure prediction, large-scale geometry optimization, high-throughput materials screening, defect calculations, diffusion studies, phase transitions, and the discovery of novel materials.

# 18.11 Applications of M3GNet in Materials Science

Until now, we have concentrated on understanding **how M3GNet works**.

We studied

* graph neural networks,
* three-body interactions,
* energy prediction,
* force prediction,
* stress prediction,
* geometry optimization,
* molecular dynamics.

The natural question now is:

**Why has M3GNet become one of the most important machine learning models in computational materials science?**

The answer lies in its remarkable versatility.

Unlike many earlier machine learning models that were designed for a single task such as property prediction, M3GNet can be applied to a wide variety of atomistic simulations.

Because it predicts

* energies,
* forces,
* stresses,

it serves as a nearly complete replacement for Density Functional Theory (DFT) in many large-scale computational tasks while being several orders of magnitude faster.

This section explores the major research applications of M3GNet in modern materials science.

---

# 18.11.1 From Machine Learning Model to Scientific Tool

Many machine learning models produce only a single scalar property.

For example,

```text
Composition

↓

Machine Learning

↓

Band Gap
```

or

```text
Composition

↓

Machine Learning

↓

Formation Energy
```

Although useful,

such models cannot simulate atomic motion or optimize crystal structures.

M3GNet is fundamentally different.

It predicts the complete atomistic physics required for simulations.

```text id="m3g_app1"
Crystal Structure

↓

M3GNet

↓

Energy

↓

Forces

↓

Stress

↓

Atomistic Simulation
```

This transforms M3GNet from a prediction model into a **general computational engine**.

---

# 18.11.2 Major Application Areas

The applications of M3GNet can be broadly divided into several categories.

```text id="m3g_app2"
M3GNet

├── Geometry Optimization

├── Molecular Dynamics

├── Crystal Structure Prediction

├── High-Throughput Screening

├── Defect Calculations

├── Phase Stability

├── Diffusion Studies

├── Elastic Properties

└── Materials Discovery
```

Each of these applications has become an active area of research.

---

# 18.11.3 Geometry Optimization

Perhaps the most common application of M3GNet is crystal structure relaxation.

Instead of performing hundreds of expensive DFT calculations,

M3GNet predicts

* energies,
* forces,
* stresses

almost instantly.

The relaxation workflow is

```text id="m3g_app3"
Initial Structure

↓

M3GNet

↓

Forces

↓

Optimization

↓

Relaxed Structure
```

This enables researchers to optimize thousands or even millions of crystal structures.

---

# 18.11.4 Why Fast Relaxation Matters

Most crystal structures obtained from

* databases,
* crystal generators,
* substitution algorithms,
* evolutionary algorithms,

are not fully relaxed.

Their atomic coordinates may deviate significantly from equilibrium.

Before calculating

* electronic properties,
* mechanical properties,
* phonons,

these structures must first be optimized.

Using DFT,

this process may require

hours

or even

days

for a single structure.

M3GNet often reduces this time to seconds or minutes.

---

# 18.11.5 High-Throughput Structure Relaxation

Modern materials databases contain enormous numbers of structures.

Examples include

* hundreds of thousands,
* millions,

of candidate materials.

Relaxing every structure with DFT would require enormous computational resources.

Instead,

the workflow becomes

```text id="m3g_app4"
Database

↓

M3GNet Relaxation

↓

Stable Structures

↓

Further Analysis
```

Only the most promising candidates need subsequent DFT refinement.

---

# 18.11.6 High-Throughput Materials Screening

One of the most powerful applications of M3GNet is **high-throughput screening**.

Rather than studying one material at a time,

researchers investigate thousands or millions of candidates.

The workflow is

```text id="m3g_app5"
Millions of Materials

↓

M3GNet

↓

Energy Evaluation

↓

Ranking

↓

Top Candidates
```

This dramatically accelerates materials discovery.

---

# 18.11.7 Why Screening is Important

Suppose a researcher seeks a new battery cathode.

Possible chemical combinations may number

* hundreds of thousands,
* or even millions.

Synthesizing every candidate experimentally is impossible.

Even performing DFT calculations for every candidate may be prohibitively expensive.

Machine learning enables rapid elimination of poor candidates,

allowing computational and experimental effort to focus on the most promising materials.

---

# 18.11.8 Crystal Structure Prediction

Another major application is **crystal structure prediction**.

In this problem,

the chemical composition is known,

but the crystal structure is unknown.

For example,

suppose we know a compound has composition

```text id="m3g_app6"
AB₂
```

The question becomes

> Which crystal structure possesses the lowest energy?

M3GNet can rapidly evaluate many candidate structures.

---

# 18.11.9 Structure Search Workflow

A simplified workflow is

```text id="m3g_app7"
Generate Candidate Structures

↓

Relax with M3GNet

↓

Compare Energies

↓

Lowest-Energy Structure
```

The most stable structure is then validated using DFT.

---

# 18.11.10 Accelerating Evolutionary Algorithms

Crystal structure prediction algorithms,

such as

* genetic algorithms,
* particle swarm optimization,
* random structure search,

may generate tens of thousands of candidate structures.

Traditionally,

each candidate required a DFT calculation.

With M3GNet,

the evaluation becomes dramatically faster,

allowing many more candidates to be explored within the same computational budget.

---

# 18.11.11 Screening Metastable Structures

Not all technologically useful materials are perfectly stable.

Many important materials are **metastable**.

Examples include

* battery electrodes,
* catalysts,
* functional oxides.

M3GNet allows rapid comparison of relative energies,

helping identify structures that are sufficiently stable to exist experimentally,

even if they are not the absolute ground state.

---

# 18.11.12 Stability Prediction

A fundamental question in materials science is

> Will this crystal remain stable?

One important quantity is the formation energy.

Lower formation energies generally correspond to greater thermodynamic stability.

M3GNet enables rapid estimation of relative structural energies,

making it possible to identify unstable candidates before expensive first-principles calculations are performed.

---

# 18.11.13 Energy Ranking

Suppose ten candidate structures exist for the same composition.

The workflow becomes

```text id="m3g_app8"
Structure 1

↓

Energy

Structure 2

↓

Energy

⋮

↓

Rank Structures

↓

Lowest Energy
```

Only the lowest-energy structures are selected for detailed investigation.

---

# 18.11.14 Database Construction

Large computational databases require

* relaxed structures,
* energies,
* forces,
* stresses.

Because M3GNet performs these calculations rapidly,

it has become a valuable tool for generating large atomistic datasets.

These datasets may subsequently be used for

* DFT refinement,
* property prediction,
* additional machine learning,
* experimental planning.

---

# 18.11.15 Integration with Existing Databases

M3GNet integrates naturally with databases such as

* Materials Project,
* OQMD,
* JARVIS,
* NOMAD.

Researchers often retrieve an initial crystal structure,

relax it using M3GNet,

perform rapid screening,

and finally validate the most promising materials using first-principles calculations.

This hierarchical workflow combines the speed of machine learning with the accuracy of DFT.

---

# 18.11.16 Practical Research Workflow

A typical modern computational materials workflow is

```text id="m3g_app9"
Materials Database

↓

Candidate Selection

↓

M3GNet Relaxation

↓

Energy Ranking

↓

Top Candidates

↓

DFT Verification

↓

Experimental Validation
```

This workflow has become increasingly common in computational materials discovery because it dramatically reduces computational cost while maintaining scientific reliability.

---

## Transition to the Next Section

The applications discussed so far focus primarily on **crystal structures and thermodynamic stability**. However, one of the greatest strengths of M3GNet is its ability to predict **atomic forces**, enabling simulations of defects, diffusion, lattice dynamics, and temperature-dependent phenomena.

In the next section, we will explore these **dynamic applications**, including vacancy diffusion, interstitial migration, ionic conductivity, phase transitions, and large-scale molecular dynamics studies that were previously impractical using conventional Density Functional Theory alone.

# 18.11.17 Defect Calculations Using M3GNet

Real materials are never perfectly crystalline.

Even the highest-quality single crystals contain imperfections.

These imperfections, known as **crystal defects**, strongly influence the physical properties of materials.

For example,

* electrical conductivity,
* ionic conductivity,
* diffusion,
* mechanical strength,
* plastic deformation,
* catalytic activity,
* optical properties,

are often controlled more by defects than by the perfect crystal lattice.

Consequently,

defect calculations represent one of the most important applications of atomistic simulations.

Because M3GNet predicts

* energies,
* atomic forces,
* stress,

it enables efficient investigation of defect structures that would otherwise require thousands of expensive DFT calculations.

---

# 18.11.18 Why Defects Matter

Consider an ideal crystal.

```text id="m3g_defect1"
● ● ● ●

● ● ● ●

● ● ● ●

● ● ● ●
```

Every lattice site is occupied.

Now suppose one atom is missing.

```text id="m3g_defect2"
● ● ● ●

● ○ ● ●

● ● ● ●

● ● ● ●
```

The missing atom creates a **vacancy**.

Although only one atom has been removed,

the neighboring atoms experience different bonding environments.

Consequently,

their positions shift until a new equilibrium structure is reached.

---

# 18.11.19 Structural Relaxation Around Defects

Immediately after creating a defect,

the surrounding atoms are generally not in equilibrium.

Neighboring atoms experience

* unbalanced forces,
* modified bond lengths,
* altered bond angles.

Therefore,

the defective structure must be relaxed.

The workflow becomes

```text id="m3g_defect3"
Create Defect

↓

M3GNet

↓

Forces

↓

Geometry Optimization

↓

Relaxed Defect Structure
```

Because M3GNet predicts forces extremely quickly,

large defective supercells can be optimized much faster than with DFT.

---

# 18.11.20 Vacancy Defects

The simplest point defect is a vacancy.

A vacancy forms when an atom is removed from its lattice site.

Examples include

* lithium vacancies,
* oxygen vacancies,
* silicon vacancies,
* metal vacancies.

Vacancies often dominate

* diffusion,
* ionic conductivity,
* creep,
* high-temperature deformation.

---

# 18.11.21 Vacancy Formation Energy

One of the most important quantities is the **vacancy formation energy**.

Conceptually,

it measures the energy required to remove one atom from the crystal.

A commonly used expression is

$$
E_{\mathrm{vac}}
================

## E_{\mathrm{defect}}

E_{\mathrm{perfect}}
+
\mu,
$$

where

* $E_{\mathrm{defect}}$ is the relaxed defective crystal energy,
* $E_{\mathrm{perfect}}$ is the perfect crystal energy,
* $\mu$ is the chemical potential of the removed atom.

A lower vacancy formation energy generally indicates that vacancies form more readily.

---

# 18.11.22 Interstitial Defects

Instead of removing an atom,

another possibility is adding an extra atom.

```text id="m3g_defect4"
● ● ● ●

● ● ● ●

● ●●● ●

● ● ● ●
```

The additional atom occupies an **interstitial site**.

Interstitial atoms create significant lattice distortion because the crystal was not designed to accommodate an extra atom.

Consequently,

large atomic relaxations are usually required.

---

# 18.11.23 Substitutional Defects

Another common defect occurs when one atom replaces another.

For example,

```text id="m3g_defect5"
Si

↓

P
```

or

```text id="wq0mna"
Fe

↓

Cr
```

These are known as **substitutional defects**.

Substitutional doping is widely used in

* semiconductors,
* alloys,
* battery materials,
* catalysts.

M3GNet enables rapid relaxation of these doped structures.

---

# 18.11.24 Large Supercell Calculations

Defect calculations require large supercells.

Suppose a primitive cell contains

```text id="m3g_defect6"
8 atoms
```

A realistic defect calculation may require

```text id="m3g_defect7"
512 atoms
```

or even more.

Large supercells reduce artificial interactions between periodic images of the defect.

Unfortunately,

DFT computational cost increases rapidly with system size.

M3GNet, however,

scales much more favorably,

making these simulations practical.

---

# 18.11.25 Workflow for Defect Relaxation

A typical defect calculation proceeds as follows.

```text id="m3g_defect8"
Perfect Crystal

↓

Build Supercell

↓

Create Defect

↓

M3GNet Relaxation

↓

Relaxed Defect

↓

Property Calculation
```

Only after relaxation can reliable physical properties be calculated.

---

# 18.11.26 Diffusion of Defects

Defects are rarely stationary.

At finite temperature,

they migrate through the crystal.

Examples include

* lithium diffusion,
* oxygen vacancy migration,
* hydrogen diffusion,
* metal atom diffusion.

The movement of these defects determines many important material properties.

---

# 18.11.27 Vacancy Diffusion

Consider a vacancy.

Initially,

```text id="m3g_defect9"
A ● ○ ●
```

A neighboring atom may jump into the empty site.

```text id="m3g_defect10"
A ○ ● ●
```

The vacancy appears to move in the opposite direction.

Repeated atomic jumps produce long-range diffusion.

---

# 18.11.28 Why Diffusion Matters

Diffusion controls numerous technological processes.

Examples include

* battery charging,
* ionic conductivity,
* corrosion,
* precipitation,
* sintering,
* crystal growth,
* alloy homogenization.

Accurate force prediction is therefore essential for studying these processes.

---

# 18.11.29 Molecular Dynamics of Defects

Because M3GNet predicts atomic forces,

defect motion can be studied directly using molecular dynamics.

The workflow becomes

```text id="m3g_defect11"
Relaxed Defect

↓

Initialize Temperature

↓

Run MD

↓

Observe Atomic Migration

↓

Calculate Diffusion
```

Long simulations that would be prohibitively expensive using DFT become feasible.

---

# 18.11.30 Diffusion Coefficient

One important quantity obtained from molecular dynamics is the **diffusion coefficient**.

It is commonly determined from the **mean squared displacement (MSD)** using the Einstein relation,

$$
D
=

\frac{1}{6}
\lim_{t\to\infty}
\frac{d}{dt}
\left<
|\mathbf{r}(t)-\mathbf{r}(0)|^2
\right>,
$$

where

* $D$ is the diffusion coefficient,
* $\mathbf{r}(t)$ is the atomic position at time $t$,
* the angle brackets denote an average over atoms and time origins.

Because M3GNet enables long-time molecular dynamics,

the MSD can be sampled over sufficiently long trajectories to obtain reliable diffusion coefficients.

---

# 18.11.31 Ionic Conductivity

Battery materials often conduct ions rather than electrons.

Examples include

* lithium-ion conductors,
* sodium-ion conductors,
* solid electrolytes.

The ionic conductivity depends strongly on

* vacancy concentration,
* diffusion pathways,
* migration barriers.

M3GNet molecular dynamics provides an efficient way to investigate these transport mechanisms over large systems and long simulation times.

---

# 18.11.32 Defect Engineering

Modern materials science increasingly relies on **defect engineering**.

Instead of eliminating defects,

researchers intentionally introduce them to improve material performance.

Examples include

* increasing catalytic activity,
* enhancing ionic conductivity,
* strengthening alloys,
* tuning electronic properties.

Because M3GNet allows rapid relaxation and simulation of defective structures,

it has become an effective tool for exploring how different defect types influence material behavior before experimental synthesis.

---

# 18.11.33 Advantages of M3GNet for Defect Studies

Compared with conventional DFT,

M3GNet offers several important advantages for defect calculations.

* Efficient relaxation of very large supercells.
* Fast evaluation of defect structures.
* Long molecular dynamics simulations for defect migration.
* Prediction of forces and stresses for distorted environments.
* Practical screening of many defect configurations before higher-level calculations.

These advantages enable systematic studies of defect physics that would otherwise demand prohibitive computational resources.

---

## Transition to the Next Section

Point defects are only one aspect of atomistic materials behavior. Many technologically important phenomena involve **collective atomic motion**, such as melting, structural phase transitions, lattice vibrations, and thermal transport.

In the next section, we will examine how **M3GNet is used to study finite-temperature phenomena**, including phase transformations, thermal expansion, lattice dynamics, and temperature-dependent structural evolution through large-scale molecular dynamics simulations.

# 18.11.34 Phase Transitions and Finite-Temperature Materials Simulations

Thus far, we have examined applications of M3GNet involving

* crystal relaxation,
* defect calculations,
* diffusion,
* molecular dynamics.

These simulations are generally performed on a single crystal structure.

However, many important materials do **not** remain in the same crystal structure under all conditions.

As

* temperature,
* pressure,
* composition,

change, the crystal itself may transform into an entirely different phase.

These structural transformations are known as **phase transitions**.

Understanding phase transitions is one of the central goals of computational materials science.

Because M3GNet can perform long molecular dynamics simulations while predicting

* energies,
* forces,
* stresses,

it has become an extremely valuable tool for investigating phase transitions that are difficult to study using conventional Density Functional Theory.

---

# 18.11.35 What is a Phase Transition?

A phase transition occurs when a material changes from one thermodynamic state to another.

Examples include

* solid → liquid,
* liquid → gas,
* cubic → tetragonal,
* tetragonal → orthorhombic,
* ferromagnetic → paramagnetic (magnetic transition).

The transition is driven by changes in

* temperature,
* pressure,
* chemical composition,
* external fields.

---

# 18.11.36 Atomic View of a Phase Transition

From the atomic perspective,

a phase transition corresponds to a rearrangement of atoms.

Initially,

```text id="m3g_phase1"
Crystal Structure A

↓

Atoms Vibrate

↓

Increasing Temperature

↓

Atomic Rearrangement

↓

Crystal Structure B
```

Unlike geometry optimization,

which seeks a local energy minimum,

phase transitions involve movement between **different energy minima**.

---

# 18.11.37 Energy Landscape

The concept of the **potential energy landscape** provides a useful way to understand phase transitions.

Imagine the energy surface as a landscape containing hills and valleys.

```text id="m3g_phase2"
Energy

      /\

     /  \

____/    \____

 Valley A  Valley B
```

Each valley corresponds to a stable crystal structure.

To transform from one phase to another,

the system must overcome an energy barrier separating the valleys.

---

# 18.11.38 Thermal Energy and Phase Transitions

At low temperatures,

atoms possess relatively little kinetic energy.

Consequently,

they remain trapped within a single energy minimum.

As the temperature increases,

their kinetic energy increases.

Eventually,

the atoms acquire sufficient energy to cross the barrier separating two crystal structures.

This leads to a phase transition.

---

# 18.11.39 Why Molecular Dynamics is Needed

Static calculations cannot directly observe phase transitions.

A geometry optimization simply finds the nearest local minimum.

Instead,

we require molecular dynamics.

The workflow becomes

```text id="m3g_phase3"
Initial Crystal

↓

Increase Temperature

↓

Atomic Motion

↓

Structural Rearrangement

↓

New Phase
```

Only time-dependent simulations can capture the continuous atomic rearrangements involved in phase transformations.

---

# 18.11.40 Heating Simulations

One common molecular dynamics experiment involves gradually increasing the temperature.

For example,

```text id="m3g_phase4"
300 K

↓

500 K

↓

700 K

↓

900 K

↓

1100 K
```

At each temperature,

the atomic positions evolve naturally.

Researchers monitor

* lattice parameters,
* energy,
* density,
* crystal symmetry,

to determine whether a phase transition has occurred.

---

# 18.11.41 Cooling Simulations

The opposite process is also important.

Instead of heating,

the material is cooled.

```text id="m3g_phase5"
1200 K

↓

900 K

↓

600 K

↓

300 K
```

Cooling simulations are used to study

* crystallization,
* glass formation,
* ordering phenomena,
* phase stability.

The final structure may depend strongly on the cooling rate.

---

# 18.11.42 Melting Simulations

One of the simplest phase transitions is melting.

Initially,

atoms occupy well-defined lattice sites.

```text id="m3g_phase6"
Ordered Crystal
```

As the temperature increases,

atomic vibrations become larger.

Eventually,

the atoms lose their long-range order.

```text id="m3g_phase7"
Disordered Liquid
```

Molecular dynamics naturally reproduces this process.

---

# 18.11.43 Solidification

The reverse process,

solidification,

can also be investigated.

Initially,

atoms are randomly distributed.

```text id="m3g_phase8"
Liquid

↓

Cooling

↓

Nucleation

↓

Crystal Growth

↓

Solid
```

Observing crystallization directly using DFT is generally impossible because of the required simulation time.

Machine-learned interatomic potentials make such simulations practical.

---

# 18.11.44 Thermal Expansion

Almost every solid expands when heated.

The underlying reason is the anharmonicity of atomic vibrations.

As temperature increases,

the average interatomic distance becomes slightly larger.

Consequently,

the lattice expands.

This phenomenon is known as **thermal expansion**.

---

# 18.11.45 Measuring Thermal Expansion

An NPT molecular dynamics simulation allows the simulation cell to change naturally.

The lattice parameter might evolve as follows.

| Temperature (K) | Lattice Constant (Å) |
| --------------: | -------------------: |
|             300 |                5.431 |
|             500 |                5.437 |
|             700 |                5.445 |
|             900 |                5.456 |
|            1100 |                5.472 |

These values illustrate the gradual increase in lattice dimensions with temperature.

---

# 18.11.46 Coefficient of Thermal Expansion

The linear coefficient of thermal expansion is defined as

$$
\alpha
======

\frac{1}{L}
\frac{dL}{dT},
$$

where

* $L$ is a characteristic length (such as the lattice constant),
* $T$ is temperature.

By calculating lattice parameters at different temperatures using M3GNet molecular dynamics,

the coefficient of thermal expansion can be estimated efficiently.

---

# 18.11.47 Temperature-Dependent Structural Stability

Not all crystal structures remain stable at high temperatures.

A structure that is stable at

```text id="m3g_phase9"
300 K
```

may become unstable at

```text id="m3g_phase10"
1200 K
```

Conversely,

certain high-temperature phases cannot exist at room temperature.

Finite-temperature molecular dynamics provides direct insight into these stability changes.

---

# 18.11.48 Monitoring Structural Changes

Researchers often monitor

* lattice constants,
* unit-cell volume,
* radial distribution functions,
* coordination numbers,
* potential energy,

throughout a simulation.

Sudden changes in one or more of these quantities frequently indicate that a phase transition has occurred.

---

# 18.11.49 Pressure-Induced Phase Transitions

Temperature is not the only driving force.

Pressure can also transform crystal structures.

For example,

```text id="m3g_phase11"
Ambient Pressure

↓

Compression

↓

Higher Density Phase
```

Because M3GNet predicts the stress tensor,

NPT molecular dynamics can investigate pressure-induced structural transformations efficiently.

---

# 18.11.50 Large-Scale Phase Simulations

Many phase transitions involve collective atomic motion extending over hundreds or thousands of atoms.

Examples include

* martensitic transformations,
* ferroelastic switching,
* reconstructive phase transitions.

Capturing these processes requires large simulation cells and long trajectories.

These requirements make conventional DFT prohibitively expensive.

Machine-learned interatomic potentials overcome this limitation by dramatically reducing the computational cost of each force evaluation.

---

# 18.11.51 Combining Molecular Dynamics with Structural Analysis

A typical workflow is

```text id="m3g_phase12"
Initial Crystal

↓

M3GNet Molecular Dynamics

↓

Trajectory

↓

Structural Analysis

↓

Phase Identification
```

The trajectory is analyzed to determine

* whether a phase transition occurred,
* when it occurred,
* how the atomic arrangement changed,
* and whether the new phase is stable.

---

# 18.11.52 Advantages of M3GNet for Finite-Temperature Simulations

Compared with first-principles molecular dynamics,

M3GNet enables

* much larger simulation cells,
* significantly longer simulation times,
* efficient studies of melting and solidification,
* investigation of temperature-dependent structural evolution,
* practical exploration of pressure-induced phase transformations.

These capabilities make M3GNet an exceptionally powerful framework for studying the finite-temperature behavior of materials.

---

## Transition to the Next Section

The applications discussed so far have focused primarily on structural evolution and thermodynamic behavior. Another major research area is the prediction of **mechanical properties**, where stresses, strains, and elastic responses determine how materials deform under external loading.

In the next section, we will study how **M3GNet is applied to calculate elastic constants, stress–strain relationships, bulk modulus, shear modulus, Young's modulus, and other mechanical properties**, providing an efficient alternative to repeated first-principles calculations for mechanical characterization.

# 18.11.53 Mechanical Property Prediction Using M3GNet

One of the most important objectives of computational materials science is to determine **how a material responds to mechanical loading**.

When a force is applied,

a material may

* stretch,
* compress,
* bend,
* twist,
* fracture,
* or permanently deform.

These responses determine whether a material is suitable for engineering applications.

For example,

* aerospace alloys require high strength,
* battery materials must withstand repeated expansion and contraction,
* semiconductor devices require mechanically stable crystals,
* structural ceramics must resist deformation under extreme temperatures.

Because M3GNet predicts both **forces** and the **stress tensor**, it can efficiently calculate many mechanical properties that traditionally require numerous Density Functional Theory calculations.

---

# 18.11.54 Stress and Strain

Mechanical behavior is fundamentally described by two quantities:

* **stress**
* **strain**

Stress describes the internal force acting within a material.

Strain describes the resulting deformation.

Conceptually,

```text id="m3g_mech1"
Applied Force

↓

Stress

↓

Deformation

↓

Strain
```

Understanding the relationship between stress and strain is central to mechanical property prediction.

---

# 18.11.55 What is Stress?

Stress represents force acting over an area.

Mathematically,

$$
\sigma
======

\frac{F}{A},
$$

where

* $F$ is the applied force,
* $A$ is the cross-sectional area.

The SI unit of stress is

```text id="m3g_mech2"
Pascal (Pa)
```

although computational materials science often reports stress in

```text id="m3g_mech3"
GPa
```

because atomic-scale stresses are very large.

---

# 18.11.56 What is Strain?

Strain measures the relative deformation of a material.

For uniaxial loading,

$$
\varepsilon
===========

\frac{\Delta L}{L_0},
$$

where

* $L_0$ is the original length,
* $\Delta L$ is the change in length.

Unlike stress,

strain has **no units** because it is the ratio of two lengths.

---

# 18.11.57 Elastic Deformation

Initially,

most materials deform elastically.

```text id="m3g_mech4"
Force Applied

↓

Material Deforms

↓

Force Removed

↓

Original Shape Restored
```

During elastic deformation,

atomic bonds stretch slightly but return to their original configuration after the load is removed.

---

# 18.11.58 Plastic Deformation

If the applied stress becomes sufficiently large,

the material may deform permanently.

```text id="m3g_mech5"
Large Force

↓

Permanent Atomic Rearrangement

↓

Permanent Shape Change
```

This is known as **plastic deformation**.

Atomistic simulations typically investigate the elastic regime first because it is directly related to the crystal's intrinsic mechanical properties.

---

# 18.11.59 Hooke's Law

Within the elastic regime,

stress and strain are approximately proportional.

This relationship is known as Hooke's Law.

For one-dimensional loading,

$$
\sigma
======

E
\varepsilon,
$$

where

* $E$ is Young's modulus.

This equation states that stiffer materials require greater stress to produce the same strain.

---

# 18.11.60 Stress–Strain Curve

The mechanical response of a material is commonly represented using a stress–strain curve.

Conceptually,

```text id="m3g_mech6"
Stress

│

│          /

│        /

│      /

│    /

│__/

──────────────

Strain
```

The initial slope corresponds to Young's modulus.

As deformation increases,

the material may eventually yield or fracture.

---

# 18.11.61 How M3GNet Calculates Mechanical Properties

The general workflow is

```text id="m3g_mech7"
Relaxed Crystal

↓

Apply Small Strain

↓

M3GNet

↓

Stress Tensor

↓

Elastic Constants
```

Instead of computing stresses using DFT,

M3GNet predicts them directly.

---

# 18.11.62 Applying Small Deformations

Mechanical calculations typically involve applying small strains,

for example,

```text id="m3g_mech8"
0.5%

1%

2%
```

After each deformation,

the atomic positions are relaxed while maintaining the imposed strain.

The resulting stress tensor is then calculated.

---

# 18.11.63 Stress–Strain Data

A typical calculation may produce

| Strain | Stress (GPa) |
| -----: | -----------: |
|  0.000 |         0.00 |
|  0.005 |         0.82 |
|  0.010 |         1.64 |
|  0.015 |         2.47 |
|  0.020 |         3.29 |

The nearly linear relationship indicates elastic behavior.

The slope of this line determines the elastic modulus.

---

# 18.11.64 Elastic Constants

Real crystals deform in three dimensions.

Consequently,

their mechanical behavior is described not by a single constant,

but by the **elastic stiffness tensor**

$$
C_{ij}.
$$

The generalized form of Hooke's Law is

$$
\sigma_i
========

\sum_j
C_{ij}
\varepsilon_j.
$$

The matrix

$$
C_{ij}
$$

contains the elastic constants of the crystal.

These constants characterize the material's resistance to deformation.

---

# 18.11.65 Crystal Symmetry and Elastic Constants

The number of independent elastic constants depends on crystal symmetry.

For example,

| Crystal System | Independent Elastic Constants |
| -------------- | ----------------------------: |
| Cubic          |                             3 |
| Hexagonal      |                             5 |
| Orthorhombic   |                             9 |
| Monoclinic     |                            13 |
| Triclinic      |                            21 |

Higher symmetry reduces the number of independent elastic constants.

---

# 18.11.66 Young's Modulus

Young's modulus measures the stiffness of a material during tensile loading.

It is defined as

$$
E
=

\frac{\sigma}{\varepsilon}.
$$

Large values indicate stiff materials.

Examples include

* diamond,
* tungsten,
* silicon carbide.

Lower values correspond to softer materials such as polymers.

---

# 18.11.67 Bulk Modulus

The **bulk modulus** measures resistance to uniform compression.

It is defined as

$$
K
=

*

V
\frac{dP}{dV},
$$

where

* $V$ is volume,
* $P$ is pressure.

Materials with large bulk moduli resist compression strongly.

Diamond is a classic example.

---

# 18.11.68 Shear Modulus

The **shear modulus** describes resistance to shape changes under shear loading.

It is defined as

$$
G
=

\frac{\tau}{\gamma},
$$

where

* $\tau$ is shear stress,
* $\gamma$ is shear strain.

Shear modulus plays an important role in understanding plastic deformation and dislocation motion.

---

# 18.11.69 Poisson's Ratio

When a material is stretched,

it usually contracts in the perpendicular direction.

This behavior is quantified by **Poisson's ratio**,

$$
\nu
===

*

\frac{\varepsilon_{\mathrm{transverse}}}
{\varepsilon_{\mathrm{longitudinal}}}.
$$

Most engineering materials have positive Poisson's ratios,

although certain metamaterials exhibit negative values.

---

# 18.11.70 Workflow for Mechanical Property Prediction

A complete computational workflow is

```text id="m3g_mech9"
Relax Crystal

↓

Apply Strain

↓

Relax Atomic Positions

↓

Predict Stress (M3GNet)

↓

Repeat for Multiple Strains

↓

Fit Stress–Strain Curve

↓

Calculate Elastic Constants
```

Because M3GNet predicts stresses rapidly,

many strain states can be evaluated efficiently.

---

# 18.11.71 High-Throughput Mechanical Screening

One powerful application is the rapid screening of mechanical properties.

Instead of computing elastic constants for a handful of materials,

researchers can evaluate thousands of compounds.

The workflow becomes

```text id="m3g_mech10"
Materials Database

↓

Automatic Deformation

↓

M3GNet Stress Prediction

↓

Elastic Properties

↓

Ranking
```

This approach accelerates the discovery of

* lightweight structural materials,
* superhard materials,
* mechanically stable battery electrodes,
* high-strength alloys.

---

# 18.11.72 Advantages of M3GNet for Mechanical Calculations

Compared with conventional DFT,

M3GNet provides several significant advantages.

* Rapid stress prediction.
* Efficient evaluation of multiple strain states.
* Practical calculations for large supercells.
* High-throughput mechanical screening.
* Integration with molecular dynamics for finite-temperature mechanical behavior.

These capabilities enable researchers to investigate mechanical properties on a much larger scale than was previously practical.

---

## Transition to the Final Chapter

We have now explored the major scientific applications of M3GNet, including structure relaxation, molecular dynamics, defect calculations, phase transitions, and mechanical property prediction.

In the **final chapter**, we will examine the **limitations, challenges, future directions, and best practices** for using M3GNet in materials informatics. We will discuss when M3GNet can replace Density Functional Theory, when DFT remains essential, strategies for validating machine-learning predictions, uncertainty considerations, transferability, and the future evolution of universal machine-learned interatomic potentials.

# Chapter 19 — Limitations, Best Practices, and Future Directions of M3GNet

---

# 19.1 Why Understanding Limitations is Essential

Throughout this book, M3GNet has been presented as a remarkably powerful machine learning model for atomistic simulations.

We have seen that it can

* predict total energies,
* compute atomic forces,
* estimate stress tensors,
* perform geometry optimization,
* run molecular dynamics,
* accelerate materials discovery.

Given these impressive capabilities, one might ask:

> **Can M3GNet completely replace Density Functional Theory (DFT)?**

The answer is **no**.

Although M3GNet achieves impressive accuracy and speed, it remains a **machine-learned approximation**. Like every machine learning model, its predictions depend on

* the quality of its training data,
* the diversity of structures it has seen,
* the physical assumptions built into the model,
* and the domain in which it is applied.

Understanding these limitations is just as important as understanding the model's strengths.

A good computational materials scientist knows **when to trust M3GNet, when to validate its predictions, and when higher-level quantum mechanical calculations are still necessary**.

---

# 19.2 Machine Learning Models Do Not Discover New Physics

One of the most common misconceptions is that machine learning "understands" materials.

It does not.

M3GNet does not solve

* the Schrödinger equation,
* the Kohn–Sham equations,
* or quantum mechanical wavefunctions.

Instead,

it learns patterns from previously computed data.

Conceptually,

```text id="limit1"
DFT Data

↓

Training

↓

M3GNet

↓

Predictions
```

The model approximates the relationship between

* atomic structure,
* energy,
* forces,
* stress.

It does **not** derive these relationships from first principles.

---

# 19.3 Generalization Depends on Training Data

The predictive ability of M3GNet depends strongly on the diversity of its training data.

Suppose the training set contains

* oxides,
* nitrides,
* metals,
* semiconductors.

The model is likely to perform well on materials similar to these.

However,

consider a completely new material class that never appeared during training.

The prediction uncertainty increases significantly.

Machine learning models are generally excellent at **interpolation** but less reliable for **extrapolation**.

---

# 19.4 Interpolation vs Extrapolation

Interpolation means making predictions between examples already represented in the training data.

```text id="limit2"
Training Data

● ● ● ● ●

Prediction

      ●
```

Extrapolation means predicting beyond the region represented during training.

```text id="limit3"
Training Data

● ● ● ●

                     Prediction
```

Most machine learning models,

including M3GNet,

perform much better during interpolation than extrapolation.

---

# 19.5 Why DFT Remains Important

Despite the success of machine learning,

DFT remains indispensable for several reasons.

DFT

* is based directly on quantum mechanics,
* predicts electronic structure,
* calculates charge density,
* computes magnetic properties,
* evaluates electronic band structures,
* provides reference data for training machine learning models.

M3GNet accelerates many atomistic simulations,

but it does **not** replace the underlying quantum mechanical theory.

---

# 19.6 M3GNet and DFT Should Work Together

Rather than viewing M3GNet and DFT as competitors,

they should be viewed as complementary tools.

A modern workflow is

```text id="limit4"
Candidate Materials

↓

M3GNet Screening

↓

Promising Candidates

↓

High-Accuracy DFT

↓

Experimental Validation
```

This hierarchical strategy combines

* machine learning speed,
* quantum mechanical accuracy,
* experimental reliability.

It has become the standard workflow in modern computational materials science.

---

# 19.7 Sources of Prediction Error

Prediction errors may arise from several sources.

### Training data limitations

If important chemical environments are absent,

the model cannot learn them.

### Dataset noise

Errors present in the reference DFT calculations propagate into the trained model.

### Model approximation

Neural networks approximate complicated functions,

but no approximation is perfect.

### Numerical precision

Floating-point computations introduce small numerical errors.

Understanding these sources helps researchers interpret predictions appropriately.

---

# 19.8 Transferability

A major challenge in machine-learned interatomic potentials is **transferability**.

Transferability refers to the ability of a model trained on one set of structures to perform well on previously unseen structures.

High transferability requires

* chemically diverse datasets,
* physically meaningful graph representations,
* robust neural network architectures.

One reason M3GNet became so influential is that it exhibits substantially better transferability than many earlier machine-learned potentials.

---

# 19.9 Out-of-Distribution Structures

A model occasionally encounters structures that differ substantially from anything in its training data.

Examples include

* unusual crystal symmetries,
* extreme pressures,
* very high temperatures,
* exotic chemical compositions.

These are called **out-of-distribution (OOD)** structures.

Predictions for OOD systems should always be interpreted cautiously and, whenever possible, validated using first-principles calculations.

---

# 19.10 Validation is Essential

No computational prediction should be accepted blindly.

Researchers typically validate M3GNet predictions by comparing them with

* DFT calculations,
* experimental measurements,
* previous theoretical studies,
* independent computational methods.

Agreement among multiple approaches increases confidence in the results.

Disagreement often indicates that additional investigation is necessary.

---

# 19.11 Best Practices for Using M3GNet

The following guidelines are recommended for practical research.

1. Begin with well-relaxed crystal structures whenever possible.
2. Verify that the material is reasonably represented by the model's training domain.
3. Validate key predictions using DFT before publication.
4. Use sufficiently large supercells for defect and molecular dynamics simulations.
5. Monitor energy, temperature, and pressure throughout molecular dynamics.
6. Compare predictions with experimental data whenever available.
7. Report the computational methodology clearly for reproducibility.

Following these practices improves both scientific reliability and reproducibility.

---

# 19.12 Future Directions of Machine-Learned Interatomic Potentials

Machine-learned interatomic potentials continue to evolve rapidly.

Current research focuses on

* larger and more diverse training datasets,
* uncertainty quantification,
* active learning,
* better long-range interaction modeling,
* improved treatment of charged systems,
* multi-fidelity learning,
* foundation models for materials science.

These advances aim to improve both accuracy and generalization.

---

# 19.13 Beyond M3GNet

Although M3GNet is one of the most influential universal interatomic potentials,

research continues to produce new architectures.

Examples include

* graph transformer models,
* equivariant graph neural networks,
* foundation models trained on millions of crystal structures,
* multi-modal materials AI systems.

Future models will likely integrate

* crystal structures,
* compositions,
* electronic properties,
* experimental data,
* literature knowledge,

within a unified framework.

---

# 19.14 Artificial Intelligence in Materials Discovery

The role of artificial intelligence is expanding beyond individual property prediction.

Modern AI systems increasingly assist researchers in

* proposing new compounds,
* planning experiments,
* optimizing synthesis conditions,
* analyzing microscopy images,
* interpreting spectroscopy,
* accelerating autonomous laboratories.

Rather than replacing scientists,

AI serves as an intelligent assistant that greatly increases research productivity.

---

# 19.15 Final Perspective

Materials informatics represents a fundamental shift in the way materials are discovered and studied.

Traditional computational materials science relied almost exclusively on solving physical equations from first principles.

Modern materials informatics combines

* physics,
* chemistry,
* computer science,
* statistics,
* and artificial intelligence

to solve increasingly complex scientific problems.

Models such as M3GNet demonstrate that machine learning can accurately reproduce many quantum mechanical calculations while reducing computational cost by orders of magnitude.

This capability enables simulations that were previously impractical because of their size or time scale.

However, the most successful researchers do not rely solely on machine learning or solely on first-principles calculations.

Instead, they integrate

* domain knowledge,
* physical theory,
* machine learning,
* high-performance computing,
* and experimental validation

into a unified research workflow.

The future of computational materials science will almost certainly be driven by this integration, where AI accelerates discovery while physics ensures scientific reliability.

---

# Chapter Summary

In this final chapter, we learned that

* M3GNet is a powerful approximation to quantum mechanical calculations, not a replacement for them.
* The accuracy of machine learning models depends strongly on their training data and their ability to generalize.
* DFT remains essential for high-accuracy electronic structure calculations and for validating machine learning predictions.
* M3GNet and DFT are most effective when used together in a hierarchical workflow.
* Understanding model limitations, validating predictions, and following best practices are essential for reliable scientific research.
* Future developments in graph neural networks, foundation models, and AI-assisted materials discovery will continue to transform computational materials science.

---

# Concluding Remarks

You have now completed **Materials Informatics Machine Learning: From Theory to Research Implementation**.

Beginning with the mathematical foundations of machine learning, progressing through classical algorithms, deep learning, graph neural networks, CGCNN, MEGNet, ALIGNN, and M3GNet, this book has developed the knowledge required to apply modern artificial intelligence to materials science research.

The ultimate goal of this book has not been simply to teach algorithms, but to enable you to think like a **materials informatics researcher**—someone who can combine materials science, physics, machine learning, and scientific programming to solve real scientific problems.

The field is advancing rapidly, and the tools described here will continue to evolve. However, the underlying principles—careful scientific reasoning, rigorous validation, and a deep understanding of both materials science and machine learning—will remain the foundation of impactful research for years to come.

