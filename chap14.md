# Chapter 14
# Crystal Graph Convolutional Neural Networks (CGCNN): Learning Directly from Crystal Structures

---

> "The transition from handcrafted descriptors to graph neural networks represents one of the most significant paradigm shifts in the history of materials informatics."

---

# Learning Objectives

After completing this chapter, readers will be able to

- Understand why conventional machine learning reaches its limits for crystal materials.
- Explain why a crystal structure is naturally represented as a graph.
- Construct crystal graphs directly from CIF files using Pymatgen.
- Engineer node and edge features suitable for graph neural networks.
- Understand the mathematical foundations of Crystal Graph Convolutional Neural Networks (CGCNN).
- Implement the original CGCNN architecture from scratch using PyTorch and PyTorch Geometric.
- Train CGCNN models for predicting materials properties such as formation energy, band gap, elastic modulus, and bulk modulus.
- Optimize hyperparameters and evaluate model performance.
- Interpret learned crystal representations and understand their physical significance.
- Apply CGCNN to real materials informatics research problems.

---

# 14.1 Introduction

Machine learning has transformed numerous scientific disciplines by enabling computers to discover patterns directly from data.

In materials science, machine learning offers an exciting possibility:

> Can a computer learn the relationship between crystal structure and material properties without explicitly solving the quantum mechanical equations governing the system?

For decades, answering this question relied almost entirely on **Density Functional Theory (DFT)** and experimental measurements.

Although remarkably successful, both approaches have significant limitations.

Experimental measurements are

- expensive,
- time consuming,
- resource intensive,
- sometimes impossible for unstable materials.

Similarly, Density Functional Theory calculations often require

- large computational resources,
- significant memory,
- long simulation times,
- careful convergence studies.

For a single crystal containing only a few dozen atoms, a DFT calculation may require several hours or even days depending on the computational settings.

Now consider a modern materials database containing

- one million candidate materials.

Performing first-principles calculations for every material becomes computationally prohibitive.

Consequently, researchers began searching for computational methods capable of predicting material properties much more rapidly.

Machine learning emerged as one of the most promising solutions.

Instead of solving the electronic Schrödinger equation for every new material, a machine learning model attempts to learn the relationship

```
Crystal Structure

↓

Machine Learning Model

↓

Material Property
```

Once trained, the model can often make predictions within milliseconds.

This capability has fundamentally changed the workflow of computational materials discovery.

Instead of performing expensive calculations on every candidate material, researchers can first screen thousands or even millions of compounds using machine learning and reserve DFT calculations only for the most promising candidates.

This strategy is now one of the cornerstones of modern materials informatics.

---

# 14.2 The Evolution of Machine Learning in Materials Science

The development of machine learning for materials science did not occur overnight.

It evolved through several distinct stages, each addressing limitations of the previous generation.

Understanding this evolution is important because it explains why Crystal Graph Convolutional Neural Networks were developed.

The progression can be summarized as

```
Experiments

↓

Physics-Based Simulation

↓

Descriptor-Based Machine Learning

↓

Deep Learning

↓

Graph Neural Networks
```

Each stage represents an important milestone in the evolution of computational materials science.

---

## Stage 1: Experimental Materials Science

Historically, new materials were discovered almost entirely through laboratory experiments.

The workflow was relatively straightforward.

```
Design Material

↓

Synthesize Material

↓

Characterize Material

↓

Measure Properties

↓

Modify Composition

↓

Repeat
```

Although this approach has produced many remarkable discoveries, it suffers from several disadvantages.

Experiments require

- specialized equipment,
- trained personnel,
- expensive materials,
- significant time.

Furthermore, exploring the enormous chemical space experimentally is practically impossible.

---

## Stage 2: Computational Materials Science

The development of Density Functional Theory introduced a revolutionary alternative.

Instead of synthesizing every material experimentally, researchers could compute

- total energy,
- electronic structure,
- charge density,
- elastic constants,
- phonon spectra,

directly from quantum mechanics.

The workflow became

```
Crystal Structure

↓

Density Functional Theory

↓

Material Property
```

DFT dramatically accelerated materials research and remains one of the most important tools in computational materials science.

However, DFT calculations remain computationally expensive.

High-throughput screening of millions of materials is still impractical using DFT alone.

---

## Stage 3: Descriptor-Based Machine Learning

The next major breakthrough was the application of machine learning.

Instead of directly solving the quantum mechanical equations, researchers represented each material using numerical descriptors.

Examples include

- atomic number,
- electronegativity,
- ionic radius,
- valence electron concentration,
- atomic mass,
- composition-based descriptors,
- structural descriptors generated using Matminer.

These descriptors formed a feature vector.

```
Crystal Structure

↓

Feature Engineering

↓

Feature Vector

↓

Machine Learning

↓

Prediction
```

Algorithms such as

- Random Forest,
- Gradient Boosting,
- Support Vector Regression,
- XGBoost,

were then trained to predict material properties.

This approach significantly reduced computational cost and often achieved excellent predictive accuracy.

However, it introduced a new challenge.

The quality of the prediction depended heavily on the quality of the handcrafted descriptors.

If important structural information was not encoded in the descriptors, the machine learning model had no way of discovering it.

This limitation motivated the next stage in the evolution of materials informatics.

---

# 14.3 Why Descriptor Engineering Reaches Its Limits

Feature engineering has been one of the central themes throughout this book.

In previous chapters, we learned how to generate descriptors using tools such as Matminer.

These descriptors summarize the chemical and structural characteristics of a material in numerical form.

For example, a crystal may be represented by a vector containing hundreds of engineered features.

```
[

Atomic Number,

Mean Electronegativity,

Atomic Radius,

Density,

Packing Fraction,

...

]
```

The machine learning algorithm receives only this numerical vector.

It never directly observes the crystal structure itself.

This creates an important limitation.

Consider the two crystal structures shown conceptually below.

```
Material A

Si

Si

Si

Si
```

and

```
Material B

Si

Si

Si

Si
```

Both materials contain exactly the same chemical composition.

However, their atomic arrangements are different.

If the engineered descriptors fail to capture this structural difference, the machine learning model will treat the two materials as being essentially identical.

This problem becomes increasingly severe for properties that depend strongly on local atomic environments, including

- band gap,
- ionic conductivity,
- defect formation energy,
- catalytic activity,
- diffusion barriers.

In these situations, manually designed descriptors may discard information that is essential for accurate prediction.

This observation raises an important question.

> Rather than designing descriptors manually, can we allow the neural network to learn directly from the crystal structure itself?

Answering this question led to one of the most important developments in modern materials informatics—the Crystal Graph Convolutional Neural Network.

---

**End of Section 14.3**

# 14.4 From Handcrafted Features to Representation Learning

The limitations of descriptor engineering naturally lead to one of the most fundamental ideas in modern deep learning:

> **Instead of manually telling a machine what features are important, allow the machine to learn those features directly from data.**

This concept is known as **representation learning**.

Representation learning is one of the defining characteristics of deep learning and is the primary reason why deep neural networks have achieved remarkable success in computer vision, natural language processing, speech recognition, and, more recently, materials informatics.

To appreciate why representation learning is so powerful, consider how humans recognize materials.

A materials scientist examining a crystal structure does not consciously calculate hundreds of numerical descriptors before making observations.

Instead, the scientist naturally notices patterns such as

- the coordination environment,
- bond lengths,
- bond angles,
- symmetry,
- repeating structural motifs,
- connectivity between atoms.

These observations emerge from examining the structure itself rather than from predefined mathematical descriptors.

Deep learning attempts to mimic this process.

Instead of relying on manually engineered descriptors, the neural network receives the raw structural information and gradually learns increasingly abstract representations that are useful for predicting material properties.

This transition fundamentally changes the philosophy of machine learning.

---

## Feature Engineering

```
Crystal Structure

↓

Human Designs Features

↓

Machine Learning Model

↓

Prediction
```

---

## Representation Learning

```
Crystal Structure

↓

Deep Neural Network

↓

Automatically Learned Features

↓

Prediction
```

Notice the important difference.

In descriptor-based machine learning,

the researcher determines which information should be preserved.

In deep learning,

the network determines which information is useful.

This seemingly small difference has profound consequences.

---

# 14.5 Why Deep Learning Changed Materials Informatics

Deep learning differs from classical machine learning in one crucial aspect.

Classical algorithms such as Random Forest or Support Vector Machines do not create new representations of the data.

Instead, they attempt to discover relationships between already existing descriptors.

Suppose we calculate

```text
Atomic Radius

Electronegativity

Density

Packing Fraction
```

A Random Forest learns relationships among these descriptors.

However, it cannot invent entirely new descriptors beyond those already provided.

Deep neural networks behave differently.

Every hidden layer transforms the input representation into a new representation.

Conceptually,

```
Input Features

↓

Hidden Representation 1

↓

Hidden Representation 2

↓

Hidden Representation 3

↓

Prediction
```

Each successive layer extracts increasingly complex information.

For crystal materials, this means the network can gradually learn

- local atomic environments,
- chemical interactions,
- coordination motifs,
- structural patterns,

without these concepts being explicitly programmed.

This ability to discover useful representations automatically is one of the main reasons deep learning has become increasingly successful in materials science.

---

# 14.6 Why Images Inspired Graph Neural Networks

The rapid success of deep learning first occurred in computer vision.

Images possess an important property.

Pixels have spatial relationships.

For example,

```
□ □ □

□ □ □

□ □ □
```

Neighboring pixels are closely related.

A Convolutional Neural Network (CNN) exploits these spatial relationships by applying convolutional filters across neighboring pixels.

This approach revolutionized image recognition.

Researchers naturally wondered whether a similar idea could be applied to crystalline materials.

Unfortunately,

crystals are fundamentally different from images.

Images exist on a regular two-dimensional grid.

Every pixel has fixed neighbors.

For example,

```
Pixel

↓

Up

Down

Left

Right
```

Every image follows the same regular structure.

Crystal structures do not.

Atoms occupy arbitrary positions in three-dimensional space.

The number of neighboring atoms is not constant.

Consider two atoms.

One atom may have

```
4 neighbors
```

while another has

```
12 neighbors.
```

Unlike images,

crystals cannot be represented naturally using a fixed rectangular grid.

Consequently,

ordinary Convolutional Neural Networks are not well suited for crystalline materials.

A different mathematical representation is required.

---

# 14.7 Why Crystals Are Naturally Graphs

Although crystals are poorly represented as images,

they possess another important property.

They consist of

- atoms,
- interactions between atoms.

This immediately suggests another mathematical structure.

A graph.

A graph consists of

- nodes,
- edges.

Remarkably,

a crystal structure consists of exactly the same components.

| Graph Theory | Crystal Structure |
|--------------|------------------|
| Node | Atom |
| Edge | Atomic interaction or bond |
| Graph | Crystal |
| Connectivity | Neighbor relationships |

This correspondence is almost perfect.

Instead of forcing crystals into an image-like representation,

we simply represent them using graphs.

For example,

consider sodium chloride.

```
Na —— Cl

|       |

Cl —— Na
```

This structure can immediately be interpreted as a graph.

```
Node

↓

Na
```

```
Node

↓

Cl
```

```
Edge

↓

Na—Cl interaction
```

The graph preserves the connectivity of the crystal without requiring a regular grid.

This observation forms the conceptual foundation of Crystal Graph Convolutional Neural Networks.

---

# 14.8 Advantages of Graph Representations

Representing crystals as graphs provides several major advantages over handcrafted descriptors.

## 1. Structural Information Is Preserved

Unlike descriptor vectors,

graphs explicitly retain the relationships between atoms.

The network knows

- which atoms are connected,
- how atoms are arranged,
- which neighbors surround each atom.

No structural information is discarded during feature engineering.

---

## 2. Flexible Number of Atoms

Machine learning models based on fixed-length feature vectors often require every material to be represented by vectors of identical size.

Graphs naturally avoid this limitation.

A graph containing

```
20 atoms
```

and another containing

```
200 atoms
```

can both be processed using the same neural network architecture.

This flexibility is particularly important for materials science because crystal unit cells vary enormously in size.

---

## 3. Local Chemical Environments Are Naturally Represented

Many material properties depend primarily on local atomic environments.

Examples include

- coordination geometry,
- nearest-neighbor interactions,
- bond lengths,
- local symmetry.

Graphs explicitly preserve these relationships.

This allows Graph Neural Networks to learn local chemistry directly from the crystal structure.

---

## 4. Better Generalization

Because graph representations retain much more structural information,

they often generalize better to previously unseen materials than models relying solely on handcrafted descriptors.

Instead of memorizing specific feature combinations,

the network learns general principles governing atomic interactions.

---

# 14.9 The Birth of Crystal Graph Convolutional Neural Networks

By approximately 2017,

several developments had converged.

First,

large open materials databases such as the Materials Project had made tens of thousands of Density Functional Theory calculations publicly available.

Second,

deep learning frameworks such as PyTorch and TensorFlow had matured considerably.

Third,

Graph Neural Networks had begun demonstrating impressive results in chemistry and molecular property prediction.

Researchers recognized that crystalline materials could also be represented naturally as graphs.

This realization led to the development of the **Crystal Graph Convolutional Neural Network (CGCNN)** by **Tian Xie** and **Jeffrey C. Grossman**.

Rather than describing crystals using handcrafted descriptors,

CGCNN introduced a fundamentally different approach.

The model receives the crystal structure itself.

It then learns useful atomic representations through repeated graph convolutions.

The overall workflow becomes

```
Crystal Structure

↓

Crystal Graph

↓

Graph Convolution Layers

↓

Crystal Representation

↓

Property Prediction
```

This architecture represented a major milestone in materials informatics because it demonstrated that deep learning could learn directly from crystal structures with minimal manual feature engineering.

It also established the foundation upon which many later graph neural network architectures—including MEGNet, ALIGNN, M3GNet, and several equivariant graph neural networks—would be built.

Before studying the architecture of CGCNN itself, however, we must first understand the mathematical language used to describe graphs.

The next section therefore introduces the essential concepts of graph theory required to understand graph neural networks for crystalline materials.

# 14.10 Why Graph Theory Matters in Materials Science

Graph theory is a branch of mathematics that studies **objects** and the **relationships between them**.

Although graph theory was originally developed as a purely mathematical discipline, it has become one of the most important foundations of modern machine learning.

Today, graph theory is used in

- social networks,
- transportation systems,
- recommendation systems,
- molecular chemistry,
- protein structure prediction,
- materials science.

The reason is simple.

Many real-world systems are naturally described as **collections of interacting entities** rather than as regular grids or tables.

Crystalline materials belong to this category.

A crystal is not merely a collection of atoms.

Instead, it is a network of interacting atoms whose collective behavior determines the physical properties of the material.

Consequently, graph theory provides a natural mathematical language for describing crystalline solids.

Before studying Crystal Graph Convolutional Neural Networks, we must therefore establish the graph theoretic concepts upon which they are built.

Although the mathematical definitions introduced in this section are general, every concept will immediately be interpreted from the perspective of crystalline materials.

---

# 14.11 What Is a Graph?

Mathematically, a graph is defined as

$$
G=(V,E)
$$

where

- \(V\) is the set of vertices (or nodes),
- \(E\) is the set of edges connecting those vertices.

This compact notation describes an enormous variety of systems.

For example,

consider four cities connected by roads.

```
City A ----- City B

 |             |

 |             |

City C ----- City D
```

The cities are nodes.

The roads are edges.

Now consider a crystal.

```
Li ----- O

 |       |

 |       |

Co ----- O
```

Exactly the same mathematical representation can be used.

The atoms become nodes.

The interactions between neighboring atoms become edges.

Although one system represents transportation and the other represents chemistry,

both are mathematically identical graphs.

This observation is one of the most important ideas underlying graph neural networks.

The neural network does not need to know that the nodes represent atoms.

It simply learns patterns from graph connectivity.

The physical meaning is supplied by the way we construct the graph.

---

# 14.12 Nodes: Representing Atoms

The first component of every crystal graph is the node.

In graph theory,

a node is simply an individual object.

For crystalline materials,

each node corresponds to one atom.

Consider crystalline silicon.

```
Si

Si

Si

Si
```

Each silicon atom becomes one node.

Similarly,

for sodium chloride,

```
Na

Cl

Na

Cl
```

four atoms produce four graph nodes.

Notice an important point.

Nodes do not represent elements.

They represent **individual atoms**.

If a crystal contains

```
20 oxygen atoms
```

then the graph contains

```
20 separate oxygen nodes.
```

Although these atoms belong to the same chemical element,

their local environments may differ.

Consequently,

their learned representations may also become different during graph neural network training.

This distinction is essential because atoms occupying different crystallographic sites often contribute differently to material properties.

---

# 14.13 Node Features

Simply creating nodes is not sufficient.

Each node must also contain numerical information describing the corresponding atom.

This information is stored as a **feature vector**.

Suppose we represent every atom using

- atomic number,
- electronegativity,
- atomic radius,
- atomic mass.

The feature vector for one atom may be written as

$$
x_i=
[x_1,x_2,x_3,x_4]
$$

where

- \(x_1\) is the atomic number,
- \(x_2\) is electronegativity,
- \(x_3\) is atomic radius,
- \(x_4\) is atomic mass.

Conceptually,

```
Li

↓

Atomic Number

↓

Electronegativity

↓

Atomic Radius

↓

Atomic Mass
```

These numerical values become the initial representation of the atom.

During graph neural network training,

these initial features are gradually transformed into much richer learned representations.

Unlike descriptor-based machine learning,

these initial atomic features are only the starting point.

The network continuously updates them as information flows through the crystal graph.

---

# 14.14 Choosing Node Features

Selecting appropriate node features is an important aspect of graph construction.

The features should satisfy two important requirements.

First,

they should uniquely characterize the chemical identity of the atom.

Second,

they should contain information that is potentially useful for predicting material properties.

Common atomic descriptors include

- atomic number,
- group number,
- period,
- atomic mass,
- covalent radius,
- atomic radius,
- ionic radius,
- Pauling electronegativity,
- electron affinity,
- first ionization energy,
- valence electron count,
- number of s electrons,
- number of p electrons,
- number of d electrons,
- number of f electrons.

Not every application requires every descriptor.

The optimal feature set depends on the scientific problem.

For example,

predicting elastic properties may require somewhat different information than predicting catalytic activity.

One of the strengths of graph neural networks, however, is that they are generally less sensitive to feature engineering than classical machine learning methods because much of the representation is learned automatically.

---

# 14.15 Example: Creating Node Features with Pymatgen

Pymatgen provides convenient access to many elemental properties.

The following example illustrates how atomic information can be extracted directly from a crystal structure.

```python
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

structure = Structure.from_file("LiCoO2.cif")

for site in structure:

    element = Element(site.specie.symbol)

    features = {

        "atomic_number": element.Z,

        "atomic_mass": float(element.atomic_mass),

        "electronegativity": element.X,

        "atomic_radius": element.atomic_radius

    }

    print(site.specie.symbol, features)
```

Typical output may resemble

```text
Li {'atomic_number': 3,
    'atomic_mass': 6.941,
    'electronegativity': 0.98,
    'atomic_radius': 1.45}

Co {'atomic_number': 27,
    'atomic_mass': 58.933,
    'electronegativity': 1.88,
    'atomic_radius': 1.52}

O {'atomic_number': 8,
   'atomic_mass': 15.999,
   'electronegativity': 3.44,
   'atomic_radius': 0.60}
```

Notice that Pymatgen automatically retrieves physically meaningful atomic properties.

These values will later become numerical node features used by the graph neural network.

However, before they can be processed by PyTorch, they must be converted into tensors.

This conversion will be performed during graph construction later in the chapter.

# 14.16 Why Edges Are More Important Than Nodes

Although atoms define the building blocks of a crystal, **materials properties are rarely determined by isolated atoms**.

Instead, they arise from **interactions between atoms**.

Consider diamond.

Every carbon atom is identical.

If we only looked at the atomic number,

```
Carbon

↓

Atomic Number = 6
```

every carbon atom would appear exactly the same.

However, diamond exhibits

- extremely high hardness,
- high thermal conductivity,
- wide band gap,

not because carbon has atomic number 6,

but because of **how carbon atoms are connected**.

Now consider graphite.

Every atom is still carbon.

```
Carbon

↓

Atomic Number = 6
```

Yet graphite is

- electrically conductive,
- soft,
- easily cleaved,
- lubricating.

The atomic composition is identical.

The difference lies entirely in the **arrangement and connectivity** of the atoms.

This simple example illustrates one of the central ideas of graph neural networks.

Nodes describe **what the atoms are**.

Edges describe **how the atoms interact**.

In many materials science problems,

the edge information is just as important as the node information.

---

# 14.17 What Is an Edge?

In graph theory,

an edge represents a relationship between two nodes.

Mathematically,

an edge connecting nodes

$$
i
$$

and

$$
j
$$

is written as

$$
(i,j)
$$

For crystalline materials,

an edge represents an interaction between neighboring atoms.

The interaction may correspond to

- a chemical bond,
- a nearest-neighbor interaction,
- an atomic interaction within a cutoff radius.

Unlike molecules,

crystals often do not have well-defined chemical bonds.

Consequently,

most crystal graph neural networks—including CGCNN—define edges using **neighbor searching** rather than explicit chemical bond detection.

Conceptually,

```
Atom i

↓

Neighbor Search

↓

Atom j

↓

Edge
```

If atom

$$
j
$$

is sufficiently close to atom

$$
i,
$$

an edge is created.

Otherwise,

no edge exists.

---

# 14.18 Neighbor Relationships in Crystal Structures

Determining neighboring atoms is one of the most important steps in graph construction.

Suppose we have the following simplified crystal.

```
          O

          |

Li ------- Co

          |

          O
```

The cobalt atom has

- one lithium neighbor,
- two oxygen neighbors.

Therefore,

three graph edges are created.

```
Co —— Li

Co —— O

Co —— O
```

Notice that edges describe **local atomic environments**.

When the graph neural network later updates the representation of the cobalt atom,

it will receive information only from these neighboring atoms.

The quality of the graph therefore depends heavily on how neighboring atoms are defined.

---

# 14.19 How Do We Decide Which Atoms Are Neighbors?

Unlike molecules,

crystals extend infinitely in three dimensions.

Consequently,

every atom has infinitely many periodic images.

Clearly,

we cannot connect every atom to every other atom.

Instead,

graph neural networks restrict interactions to nearby atoms.

Several neighbor-finding strategies exist.

The most common are

- fixed cutoff radius,
- fixed number of nearest neighbors,
- Voronoi tessellation,
- bond-based methods.

Each approach has advantages and disadvantages.

The original CGCNN uses a **fixed cutoff radius combined with a maximum number of neighbors**, making graph construction both physically meaningful and computationally efficient.

The choice of neighbor definition strongly influences model performance and should therefore be regarded as an important hyperparameter.

---

# 14.20 Fixed Cutoff Radius

The simplest approach is to connect atoms whose distance is smaller than a specified cutoff radius.

Suppose the cutoff radius is

$$
r_c = 6\ \text{Å}
$$

Then,

for every atom,

all neighboring atoms within

6 Å

are connected.

Conceptually,

```
           O

      ○────────○

        \      /

         \    /

          \  /

           Li

Radius = 6 Å
```

Every atom inside the sphere becomes a neighbor.

Every atom outside the sphere is ignored.

Mathematically,

an edge exists if

$$
d_{ij} \le r_c
$$

where

- \(d_{ij}\) is the distance between atoms \(i\) and \(j\),
- \(r_c\) is the cutoff radius.

This criterion is simple,

fast,

and widely used in graph neural networks for materials science.

---

# 14.21 Choosing an Appropriate Cutoff Radius

The cutoff radius determines how much of the local crystal environment is included in the graph.

A cutoff that is too small causes important interactions to be ignored.

For example,

```
Cutoff = 2 Å
```

may miss

- second-nearest neighbors,
- longer chemical interactions,
- important coordination information.

The resulting graph becomes incomplete.

Conversely,

a cutoff that is too large creates unnecessary edges.

```
Cutoff = 15 Å
```

may connect atoms that interact only weakly.

This increases

- graph size,
- memory usage,
- computational cost,

while providing little additional physical information.

Selecting an appropriate cutoff therefore requires balancing

- physical realism,
- computational efficiency.

Typical values used in crystal graph neural networks range from

```
5 Å

to

8 Å.
```

The optimal value depends on the material system and the property being predicted.

---

# 14.22 Maximum Number of Neighbors

Some crystal structures contain extremely dense atomic environments.

Suppose one atom has

```
28 neighboring atoms
```

inside the cutoff radius.

Processing every neighbor increases computational cost.

The original CGCNN therefore introduces another constraint.

Only the closest

$$
N_{\max}
$$

neighbors are retained.

For example,

if

$$
N_{\max}=12,
$$

the graph construction algorithm

- computes all neighboring atoms,
- sorts them according to distance,
- keeps only the nearest twelve.

Conceptually,

```
All Neighbors

↓

Sort by Distance

↓

Keep Closest N

↓

Create Edges
```

This strategy produces graphs of approximately consistent size,

making neural network training more efficient.

---

# 14.23 Finding Neighbors Using Pymatgen

Pymatgen provides several methods for identifying neighboring atoms.

One of the most commonly used is

```python
Structure.get_all_neighbors()
```

The following example identifies all neighboring atoms within a cutoff radius of

6 Å.

```python
from pymatgen.core import Structure

structure = Structure.from_file("LiCoO2.cif")

cutoff = 6.0

all_neighbors = structure.get_all_neighbors(cutoff)

for atom_index, neighbors in enumerate(all_neighbors):

    print(f"\nAtom {atom_index}")

    for neighbor in neighbors:

        print(

            neighbor.specie,

            round(neighbor.nn_distance, 3),

            "Å"

        )
```

A typical output may resemble

```text
Atom 0

O 2.091 Å

O 2.091 Å

Co 2.816 Å

Co 2.816 Å

Li 4.957 Å
```

For each atom,

Pymatgen returns

- neighboring atom,
- interatomic distance,
- periodic image information,
- atomic coordinates.

These neighbor lists form the foundation of crystal graph construction.

In the next section,

we will convert these neighbor relationships into the graph connectivity format required by PyTorch Geometric.

# 14.24 Periodic Boundary Conditions: The Foundation of Crystal Graphs

One of the most important differences between molecular graphs and crystal graphs is the presence of **periodic boundary conditions (PBCs)**.

Ignoring periodicity produces an incorrect graph and can significantly reduce prediction accuracy.

Therefore, before constructing a crystal graph, we must understand how periodic boundary conditions work.

---

## Why Periodic Boundary Conditions Are Needed

Unlike molecules, crystals are **not isolated objects**.

A crystal extends infinitely in all three spatial directions.

Of course, a computer cannot store an infinite number of atoms.

Instead, crystallography represents an infinite crystal using a **unit cell**.

The unit cell contains the smallest repeating arrangement of atoms that can generate the entire crystal through translation.

Conceptually,

```
Unit Cell

↓

Repeat Along x

↓

Repeat Along y

↓

Repeat Along z

↓

Infinite Crystal
```

Every atom inside the unit cell has neighboring atoms that may actually belong to neighboring copies of the unit cell.

Consequently, limiting neighbor searches to atoms inside a single unit cell produces incorrect atomic environments.

---

## Example: Sodium Chloride

Consider a simplified sodium chloride crystal.

```
+-----------+
| Na     Cl |
|           |
| Cl     Na |
+-----------+
```

Suppose we examine the sodium atom located near the right boundary.

If we ignore periodicity, we might conclude that it has only two nearby chlorine atoms.

However, this conclusion is incorrect.

The crystal continues beyond the right boundary.

Immediately outside the unit cell is another identical unit cell.

```
+-----------+-----------+
| Na     Cl | Na     Cl |
|           |           |
| Cl     Na | Cl     Na |
+-----------+-----------+
```

The sodium atom interacts not only with atoms inside its own unit cell but also with atoms inside neighboring cells.

Ignoring these interactions destroys the true crystal environment.

---

## The Infinite Crystal Concept

Periodic boundary conditions allow the computer to behave as though the crystal continues forever.

Instead of explicitly storing infinitely many atoms,

the software stores only

```
One Unit Cell
```

and automatically generates neighboring image atoms whenever necessary.

Conceptually,

```
Stored Unit Cell

↓

Generate Periodic Images

↓

Neighbor Search

↓

Construct Graph
```

The generated image atoms are **not new physical atoms**.

They simply represent atoms belonging to neighboring copies of the unit cell.

---

# 14.25 Image Atoms

Suppose an atom lies very close to the left boundary of the unit cell.

```
+-----------------+

Na

|

|

Left Boundary

+-----------------+
```

Its nearest neighbor may actually lie inside the unit cell immediately to the left.

Although this neighboring atom is physically identical to an atom already stored in memory,

its translated position must be considered during neighbor searching.

This translated copy is called an **image atom**.

Conceptually,

```
Original Atom

↓

Translate by Lattice Vector

↓

Image Atom
```

The graph construction algorithm treats image atoms exactly like ordinary neighboring atoms when determining local atomic environments.

After graph construction,

only the connectivity information is retained.

The duplicated image atoms themselves are not permanently stored in the graph.

---

# 14.26 Lattice Translation Vectors

Periodic images are generated using the crystal lattice vectors.

Suppose the lattice vectors are

$$
\mathbf{a},
\mathbf{b},
\mathbf{c}.
$$

Every periodic image can be obtained through

$$
\mathbf{R}
=
n_a\mathbf{a}
+
n_b\mathbf{b}
+
n_c\mathbf{c}
$$

where

$$
n_a,n_b,n_c
$$

are integers.

For example,

```
(0,0,0)

↓

Original Unit Cell
```

```
(1,0,0)

↓

One Cell Along +x
```

```
(-1,0,0)

↓

One Cell Along -x
```

```
(0,1,0)

↓

One Cell Along +y
```

Each translated copy is physically identical.

Only its spatial position changes.

---

# 14.27 Why Periodicity Changes Neighbor Relationships

Consider two atoms.

```
Atom A

....................................

Atom B
```

Inside a single unit cell,

their direct distance appears very large.

However,

because the crystal repeats,

an image of Atom B may lie immediately beside Atom A.

```
Atom A

Atom B (Image)
```

The true nearest-neighbor distance therefore becomes much smaller.

Graph construction must always use the **minimum-image distance** rather than the direct distance inside a single unit cell.

Failing to do so produces incorrect edge connections.

---

# 14.28 Neighbor Search with Periodicity in Pymatgen

Fortunately,

Pymatgen automatically handles periodic boundary conditions during neighbor searching.

The user does not need to manually generate image atoms.

Consider the following example.

```python
from pymatgen.core import Structure

structure = Structure.from_file("LiCoO2.cif")

neighbors = structure.get_all_neighbors(

    r=6.0

)
```

Even though only one unit cell is stored,

Pymatgen internally

- generates image atoms,
- computes minimum-image distances,
- identifies neighboring atoms,
- returns the correct local environment.

This is one of the major reasons Pymatgen is widely used in materials informatics.

---

# 14.29 Inspecting Periodic Image Information

Each neighboring atom returned by Pymatgen contains additional information describing its periodic image.

The following example prints the image vector associated with every neighbor.

```python
from pymatgen.core import Structure

structure = Structure.from_file("LiCoO2.cif")

neighbors = structure.get_all_neighbors(

    r=6.0

)

for atom_neighbors in neighbors:

    for neighbor in atom_neighbors:

        print(

            "Species:",

            neighbor.specie

        )

        print(

            "Distance:",

            round(neighbor.nn_distance, 3)

        )

        print(

            "Image:",

            neighbor.image

        )

        print()
```

A typical output may resemble

```text
Species: O

Distance: 2.091

Image: (0, 0, 0)

Species: Co

Distance: 2.816

Image: (1, 0, 0)

Species: Li

Distance: 4.957

Image: (-1, 0, 0)
```

The image vector indicates which translated unit cell contains the neighboring atom.

For example,

```
(1,0,0)
```

means that the neighboring atom belongs to the unit cell translated by one lattice vector in the positive x-direction.

Similarly,

```
(-1,0,0)
```

corresponds to the neighboring cell in the negative x-direction.

Although these image atoms are physically identical to atoms inside the original unit cell,

their translated positions are essential for constructing the correct crystal graph.

---

# 14.30 Why Periodic Boundary Conditions Are Essential for CGCNN

Every message-passing operation in CGCNN depends on the local atomic neighborhood.

If the neighborhood is incorrect,

the learned atomic representations will also be incorrect.

Ignoring periodic boundary conditions can produce

- missing neighbors,
- incorrect coordination numbers,
- distorted bond distances,
- inaccurate local environments.

As a result,

the graph neural network learns an incorrect representation of the crystal.

This ultimately reduces prediction accuracy for properties such as

- formation energy,
- band gap,
- elastic constants,
- magnetic properties.

For this reason,

every modern graph neural network developed for crystalline materials—including CGCNN, MEGNet, ALIGNN, and M3GNet—constructs graphs using periodic boundary conditions.

Periodic neighbor searching is therefore not an implementation detail.

It is a fundamental physical requirement for accurately representing crystalline materials as graphs.

# 14.31 From Neighbor Lists to Graph Connectivity

After identifying neighboring atoms, we still do not have a graph.

At this stage, we only possess a collection of neighbor lists.

For example,

```
Atom 0

↓

Neighbors

1

3

5

7
```

```
Atom 1

↓

Neighbors

0

2

4
```

```
Atom 2

↓

Neighbors

1

5

6
```

Although this information describes the local environment of every atom, graph neural networks cannot directly process neighbor lists.

Instead, they require a mathematical representation of graph connectivity.

In PyTorch Geometric, this representation is called **edge connectivity** and is stored using a tensor named

```text
edge_index
```

Understanding how neighbor lists become `edge_index` is one of the most important steps in graph construction.

---

# 14.32 What Is `edge_index`?

Most graph neural network libraries represent graph connectivity using a **Coordinate List (COO)** format.

Instead of storing an adjacency matrix, every edge is represented by two integers.

The first integer denotes the source node.

The second integer denotes the destination node.

For example,

```
Atom 0

↓

Atom 1
```

is represented as

```
0 → 1
```

Similarly,

```
Atom 0

↓

Atom 3
```

becomes

```
0 → 3
```

Rather than storing these edges separately, PyTorch Geometric combines them into a two-row tensor.

Conceptually,

```
Source Nodes

↓

0

0

1

2

3
```

```
Destination Nodes

↓

1

3

2

5

4
```

Together,

these two rows form

```text
edge_index
```

which completely describes the connectivity of the graph.

---

# 14.33 Building `edge_index` Step by Step

Suppose we have the following graph.

```
      Atom 1

     /      \

Atom 0 ---- Atom 2

     \

      Atom 3
```

The graph contains four atoms.

The connections are

```
0 → 1

0 → 2

0 → 3

1 → 2
```

The corresponding `edge_index` becomes

```text
[[0, 0, 0, 1],
 [1, 2, 3, 2]]
```

Notice that

- the first row contains source atoms,
- the second row contains destination atoms.

Every column corresponds to one edge.

This compact representation allows graph neural networks to efficiently perform message passing.

---

# 14.34 Undirected Crystal Graphs

Atomic interactions are generally considered **undirected**.

If

```
Atom A

↓

Atom B
```

is a valid interaction,

then

```
Atom B

↓

Atom A
```

should also exist.

Consequently,

every edge must appear twice.

For example,

instead of

```text
0 → 1
```

we include

```text
0 → 1

1 → 0
```

Likewise,

```
2 → 5
```

becomes

```
2 → 5

5 → 2
```

This symmetric representation allows information to flow in both directions during message passing.

Most crystal graph neural networks therefore construct **bidirectional edges** even though the underlying interaction is physically the same.

---

# 14.35 Constructing `edge_index` from Neighbor Lists

Suppose Pymatgen has returned the following neighbor information.

```
Atom 0

↓

1

3

5
```

```
Atom 1

↓

0

2
```

We can convert these neighbor relationships into graph connectivity using Python.

```python
edge_list = []

for source_atom, neighbors in enumerate(all_neighbors):

    for neighbor in neighbors:

        destination_atom = neighbor.index

        edge_list.append(

            [source_atom, destination_atom]

        )
```

At this stage,

`edge_list` contains

```text
[[0,1],

 [0,3],

 [0,5],

 [1,0],

 [1,2],

 ...]
```

However,

PyTorch Geometric expects a tensor rather than a Python list.

---

# 14.36 Converting to a PyTorch Tensor

The edge list is converted into a tensor using PyTorch.

```python
import torch

edge_index = torch.tensor(

    edge_list,

    dtype=torch.long

).t().contiguous()
```

Let us examine this code carefully.

First,

```python
torch.tensor(...)
```

converts the Python list into a tensor.

The tensor initially has shape

```
(Number of Edges, 2)
```

For example,

```
[[0,1],

 [0,3],

 [1,2]]
```

has shape

```
(3,2)
```

However,

PyTorch Geometric requires the opposite arrangement.

The transpose operation

```python
.t()
```

changes the shape to

```
(2, Number of Edges)
```

Finally,

```python
.contiguous()
```

ensures that the tensor occupies contiguous memory,

which improves computational efficiency during graph operations.

---

# 14.37 Understanding the Shape of `edge_index`

Suppose the graph contains

```
150 edges
```

The resulting tensor has shape

```python
torch.Size([2,150])
```

The first row stores every source node.

The second row stores every destination node.

Conceptually,

```
edge_index

↓

Row 0

↓

Sources
```

```
Row 1

↓

Destinations
```

Each column therefore represents one graph connection.

This representation is extremely memory efficient because it stores only existing edges.

Unlike an adjacency matrix,

memory usage grows approximately linearly with the number of edges rather than quadratically with the number of nodes.

This is one of the reasons modern graph neural network libraries use the COO representation.

---

# 14.38 Visualizing `edge_index`

Consider the following crystal graph.

```
      1

     / \

    /   \

   0-----2

    \

     \

      3
```

The bidirectional graph produces

```text
edge_index

[[0,1,

  0,2,

  0,3,

  1,0,

  1,2,

  2,0,

  2,1,

  3,0],


 [1,0,

  2,0,

  3,0,

  0,1,

  2,1,

  0,2,

  1,2,

  0,3]]
```

Although this tensor appears simple,

it completely specifies the topology of the graph.

Every graph convolution performed by CGCNN relies on this connectivity information.

Without `edge_index`,

the neural network has no knowledge of which atoms should exchange information.

---

# 14.39 Verifying Graph Connectivity

Before training a graph neural network,

it is good practice to verify that the constructed graph is correct.

Several simple checks can be performed.

First,

confirm that every atom has at least one neighbor.

```python
num_atoms = len(structure)

connected_atoms = set(

    edge_index[0].tolist()

)

print(

    "Connected:",

    len(connected_atoms),

    "/",

    num_atoms

)
```

Second,

verify that every edge has a corresponding reverse edge.

Third,

ensure that no atom has an unexpectedly large or unexpectedly small number of neighbors.

Finally,

visual inspection of a few crystal graphs is often useful for confirming that graph construction has behaved as expected.

These simple validation steps can prevent subtle graph construction errors that may otherwise remain unnoticed throughout model training.

---

# 14.40 What Comes After `edge_index`?

At this stage,

our graph contains

- nodes,
- node features,
- edges,
- graph connectivity.

However,

one essential component is still missing.

Knowing that two atoms are connected is not sufficient.

The neural network must also understand **how they are connected**.

For example,

two neighboring atoms may be separated by

```
2.0 Å
```

while another pair may be separated by

```
5.6 Å.
```

These interactions clearly should not be treated as identical.

Therefore,

each edge must also contain numerical information describing the interaction between neighboring atoms.

These quantities are called **edge features**.

In the next section, we will construct physically meaningful edge features, beginning with the most fundamental descriptor in crystalline materials—the interatomic distance.

# 14.41 Why Edge Features Are Necessary

Up to this point, our graph contains

- nodes,
- node features,
- graph connectivity (`edge_index`).

Although this information defines the topology of the crystal graph, it is still incomplete.

Consider the following two situations.

```
Carbon -------- Carbon

Distance = 1.54 Å
```

and

```
Carbon ----------------------------- Carbon

Distance = 5.80 Å
```

Both pairs of atoms are connected by an edge.

If the graph only stores connectivity,

the neural network sees these two interactions as identical.

However, from a physical perspective, they are fundamentally different.

Atoms separated by

```
1.54 Å
```

interact much more strongly than atoms separated by

```
5.80 Å.
```

Therefore, every edge must contain additional numerical information describing the interaction between neighboring atoms.

This information is stored as **edge features**.

---

# 14.42 What Are Edge Features?

An edge feature is a numerical vector describing the relationship between two connected atoms.

Mathematically,

for an edge connecting atoms

$$
i
$$

and

$$
j,
$$

the edge feature is written as

$$
e_{ij}.
$$

Unlike node features,

which describe individual atoms,

edge features describe **pairwise atomic interactions**.

Typical edge features include

- interatomic distance,
- bond length,
- bond type,
- bond order (for molecules),
- relative atomic position,
- Gaussian distance expansion.

For crystalline materials,

the most important edge feature is the **interatomic distance**.

Consequently, the original CGCNN constructs edge features primarily from neighboring atomic distances.

---

# 14.43 Computing Interatomic Distances

Suppose two atoms occupy Cartesian coordinates

$$
(x_i,y_i,z_i)
$$

and

$$
(x_j,y_j,z_j).
$$

Their Euclidean distance is

$$
d_{ij}
=
\sqrt{
(x_i-x_j)^2
+
(y_i-y_j)^2
+
(z_i-z_j)^2
}.
$$

This quantity measures the separation between the two atoms.

In crystalline materials,

the distance is always computed using the **minimum-image convention**, ensuring that periodic boundary conditions are correctly respected.

Fortunately,

Pymatgen performs these calculations automatically during neighbor searching.

Each neighbor object returned by

```python
get_all_neighbors()
```

already contains the shortest periodic distance.

---

# 14.44 Accessing Distances in Pymatgen

The interatomic distance is available through the

```python
nn_distance
```

attribute.

The following example extracts the distance associated with every neighboring atom.

```python
from pymatgen.core import Structure

structure = Structure.from_file("LiCoO2.cif")

neighbors = structure.get_all_neighbors(

    r=6.0

)

for atom_neighbors in neighbors:

    for neighbor in atom_neighbors:

        distance = neighbor.nn_distance

        print(distance)
```

A typical output may resemble

```text
2.091

2.816

4.957

5.631

...
```

These values are measured in **angstroms (Å)** and represent the physical separation between neighboring atoms.

The distances will later become part of the edge feature tensor.

---

# 14.45 Why Raw Distances Are Not Ideal

One might expect that the interatomic distance itself could be used directly as an edge feature.

For example,

```
Edge

↓

Distance = 2.15 Å
```

Although this seems reasonable,

using raw distances introduces several difficulties.

First,

small changes in atomic position produce only small numerical differences.

```
2.11 Å

↓

2.12 Å

↓

2.13 Å
```

A neural network may struggle to distinguish subtle structural differences from such scalar values alone.

Second,

a single number provides only limited expressive power.

The network must infer complex nonlinear relationships directly from one scalar quantity.

Finally,

raw distances provide poor localization.

Atoms separated by

```
2.0 Å

and

5.0 Å
```

produce very different interactions,

yet the network receives only two numerical values.

Researchers discovered that transforming distances into a richer representation significantly improves learning.

This transformation is called **Gaussian distance expansion**.

---

# 14.46 The Idea Behind Gaussian Distance Expansion

Instead of representing every edge using a single distance,

CGCNN represents each distance using a vector of overlapping Gaussian functions.

Conceptually,

instead of

```
Distance

↓

2.43 Å
```

we generate

```
Distance

↓

[

0.02,

0.15,

0.61,

0.97,

0.72,

0.31,

0.05,

...

]
```

The distance has now become a high-dimensional feature vector.

Each element measures how strongly the distance activates a particular Gaussian basis function.

This richer representation enables the neural network to distinguish atomic environments much more effectively than using raw distances alone.

---

# 14.47 Why Gaussian Functions?

To understand the motivation,

consider how color is represented in digital images.

A single grayscale value contains relatively little information.

Instead,

modern imaging often represents color using multiple channels such as

- red,
- green,
- blue.

Similarly,

CGCNN expands one scalar distance into multiple numerical channels.

Each Gaussian function responds strongly only to distances near its center.

Conceptually,

```
Distance

↓

Gaussian 1

↓

Gaussian 2

↓

Gaussian 3

↓

...

↓

Feature Vector
```

Nearby distances activate similar Gaussian functions,

whereas distant values produce very different activation patterns.

This produces a smooth, continuous representation of atomic interactions.

---

# 14.48 Mathematical Definition of Gaussian Expansion

Suppose the interatomic distance is

$$
d.
$$

The response of one Gaussian basis function centered at

$$
\mu_k
$$

is

$$
g_k(d)
=
\exp
\left(
-
\frac{(d-\mu_k)^2}
{\sigma^2}
\right).
$$

where

- \(d\) is the interatomic distance,
- \(\mu_k\) is the center of the \(k\)-th Gaussian,
- \(\sigma\) controls the width of the Gaussian.

Rather than computing only one Gaussian,

CGCNN evaluates many Gaussian functions with different centers.

The resulting edge feature becomes

$$
e_{ij}
=
[g_1,g_2,\ldots,g_K].
$$

This vector replaces the original scalar distance.

---

# 14.49 Understanding Gaussian Expansion Intuitively

Suppose we create Gaussian functions centered at

```
0 Å

1 Å

2 Å

3 Å

4 Å

5 Å

6 Å
```

Now consider an edge with distance

```
2.2 Å.
```

The Gaussian centered at

```
2 Å
```

will produce a large value.

The Gaussian centered at

```
3 Å
```

will also respond,

although less strongly.

The Gaussian centered at

```
6 Å
```

will produce a value very close to zero.

Instead of describing the edge using

```
2.2
```

the graph neural network receives a smooth activation pattern spanning multiple basis functions.

This richer encoding substantially improves the ability of the network to learn complex structural relationships.

In the next section, we will implement Gaussian distance expansion from scratch in Python and use it to construct the complete edge feature tensor required by the Crystal Graph Convolutional Neural Network.

# 14.50 Implementing Gaussian Distance Expansion from Scratch

Now that we understand the mathematical motivation behind Gaussian distance expansion, we can implement it ourselves.

Rather than relying on a prewritten library function, we will construct the Gaussian basis expansion step by step.

The implementation follows exactly the mathematical definition introduced in the previous section.

We begin by importing PyTorch.

```python
import torch
```

Next, we define a function that transforms one interatomic distance into its Gaussian representation.

```python
import torch

def gaussian_expansion(

    distance,

    centers,

    width

):

    distance = torch.tensor(distance)

    gaussian = torch.exp(

        -((distance - centers) ** 2) /

        (width ** 2)

    )

    return gaussian
```

Although the function contains only a few lines, every statement performs an important mathematical operation.

Let us examine each step carefully.

---

# 14.51 Understanding the Implementation

The first argument,

```python
distance
```

is the interatomic distance between two neighboring atoms.

For example,

```python
distance = 2.43
```

The second argument,

```python
centers
```

contains the centers of every Gaussian basis function.

For example,

```python
centers = torch.arange(

    0,

    6.5,

    0.2

)
```

produces

```text
0.0

0.2

0.4

0.6

...

6.0
```

Each of these values corresponds to one Gaussian basis function.

Finally,

```python
width
```

determines the width of every Gaussian.

Typical values lie between

```
0.2 Å

and

0.5 Å.
```

Smaller widths produce narrower Gaussian peaks,

whereas larger widths generate smoother overlapping basis functions.

---

# 14.52 Creating the Gaussian Centers

The Gaussian centers are usually distributed uniformly across the cutoff radius.

Suppose the cutoff radius is

```
6 Å.
```

The following code constructs equally spaced centers.

```python
cutoff = 6.0

step = 0.2

centers = torch.arange(

    0,

    cutoff + step,

    step

)

print(centers)
```

Typical output

```text
tensor([

0.0,

0.2,

0.4,

0.6,

...

5.8,

6.0

])
```

In this example,

the graph neural network uses

31 Gaussian basis functions.

Every interatomic distance will therefore be represented by a feature vector of length

```
31.
```

---

# 14.53 Applying Gaussian Expansion

Suppose two neighboring atoms are separated by

```
2.35 Å.
```

Applying Gaussian expansion is straightforward.

```python
distance = 2.35

width = 0.2

edge_feature = gaussian_expansion(

    distance,

    centers,

    width

)

print(edge_feature)
```

The resulting tensor may resemble

```text
tensor([

0.0000,

0.0000,

0.0001,

...

0.7548,

0.9692,

0.8353,

0.4827,

0.1865,

...

0.0000

])
```

Notice that only Gaussian functions whose centers lie close to

```
2.35 Å
```

produce significant values.

The remaining basis functions are nearly zero.

Consequently,

the resulting feature vector is localized around the true interatomic distance.

---

# 14.54 Visualizing the Gaussian Representation

Suppose the cutoff radius is

```
6 Å
```

and the Gaussian centers are spaced every

```
0.2 Å.
```

The distance

```
2.35 Å
```

activates several nearby basis functions.

Conceptually,

```text
Distance

↓

2.35 Å

↓

Gaussian Basis

↓

0.00

0.01

0.08

0.42

0.91

0.98

0.67

0.24

0.03

0.00

...
```

Rather than encoding the interaction using one number,

the neural network now receives a smooth, continuous feature vector.

This richer representation allows neighboring distances to produce similar feature vectors,

making learning significantly easier.

---

# 14.55 Constructing Edge Features for an Entire Crystal

A crystal contains many neighboring atom pairs.

Therefore,

Gaussian expansion must be applied to every edge.

The following implementation performs this operation.

```python
edge_features = []

for atom_neighbors in neighbors:

    for neighbor in atom_neighbors:

        distance = neighbor.nn_distance

        feature = gaussian_expansion(

            distance,

            centers,

            width

        )

        edge_features.append(feature)
```

Each neighboring atom contributes one edge feature vector.

If the crystal graph contains

```
850 edges
```

then

```
850 Gaussian vectors
```

are generated.

These vectors collectively form the edge feature matrix.

---

# 14.56 Converting Edge Features into a Tensor

The edge feature vectors stored in the Python list must now be combined into a single tensor.

PyTorch provides the

```python
torch.stack()
```

function for this purpose.

```python
edge_attr = torch.stack(

    edge_features

)
```

Suppose

- the graph contains

```
850 edges
```

and

- each Gaussian expansion contains

```
31 basis functions.
```

The resulting tensor has shape

```python
torch.Size([850,31])
```

The first dimension corresponds to graph edges.

The second dimension corresponds to Gaussian basis functions.

Each row therefore represents one neighboring atomic interaction.

---

# 14.57 Understanding `edge_attr`

PyTorch Geometric stores edge features using the variable

```python
edge_attr
```

Conceptually,

```text
edge_attr

↓

Edge 1

↓

31 Features
```

```text
Edge 2

↓

31 Features
```

```text
Edge 3

↓

31 Features
```

Every edge possesses its own feature vector.

During graph convolution,

the neural network uses both

- node features,

and

- edge features

to compute messages exchanged between neighboring atoms.

Unlike traditional graph convolutional networks,

CGCNN explicitly incorporates edge information into message passing.

This is one of the major reasons why it performs well for crystalline materials.

---

# 14.58 Verifying Edge Features

Before proceeding further,

it is useful to verify that edge features have been generated correctly.

Several simple checks can be performed.

First,

inspect the tensor shape.

```python
print(edge_attr.shape)
```

Typical output

```text
torch.Size([850,31])
```

Second,

inspect one edge feature.

```python
print(edge_attr[0])
```

Third,

confirm that no feature contains invalid numerical values.

```python
print(

    torch.isnan(edge_attr).any()

)
```

The output should be

```text
False
```

Finally,

verify that every edge listed in

```python
edge_index
```

has a corresponding row in

```python
edge_attr.
```

Maintaining this one-to-one correspondence is essential.

If the ordering of edge features does not exactly match the ordering of graph edges,

the graph neural network will associate incorrect interactions with neighboring atoms,

leading to erroneous message passing.

At this point, we have successfully constructed

- node features,
- graph connectivity,
- edge features.

These three components together constitute the complete crystal graph representation required as input to the Crystal Graph Convolutional Neural Network.

The next step is to assemble these components into the `Data` object used by PyTorch Geometric.

# 14.59 The PyTorch Geometric `Data` Object

Up to this point, we have constructed the three fundamental components of a crystal graph:

- node features,
- graph connectivity,
- edge features.

Although these components exist individually, they must now be combined into a single data structure that can be processed by a Graph Neural Network.

PyTorch Geometric accomplishes this using the **`Data` object**.

The `Data` object serves as a container that stores every piece of information associated with a graph.

Rather than passing multiple tensors separately,

```text
Node Features

↓

Edge Connectivity

↓

Edge Features
```

they are grouped together into one object.

Conceptually,

```text
Crystal Graph

↓

Data Object

↓

Graph Neural Network
```

This unified representation greatly simplifies graph processing and enables PyTorch Geometric to perform batching, message passing, pooling, and graph-level learning efficiently.

---

# 14.60 Anatomy of a `Data` Object

A `Data` object may contain many different attributes.

For CGCNN, the most important ones are

| Attribute | Description | Shape |
|-----------|-------------|-------|
| `x` | Node feature matrix | `(N, F_node)` |
| `edge_index` | Graph connectivity | `(2, E)` |
| `edge_attr` | Edge feature matrix | `(E, F_edge)` |
| `y` | Target property | Depends on task |

where

- \(N\) is the number of atoms,
- \(E\) is the number of graph edges,
- \(F_{node}\) is the number of node features,
- \(F_{edge}\) is the number of edge features.

For example,

suppose a crystal contains

- 40 atoms,
- 480 edges,
- 92 node features,
- 41 Gaussian edge features.

Then

```
x

↓

(40,92)
```

```
edge_index

↓

(2,480)
```

```
edge_attr

↓

(480,41)
```

Together, these tensors completely describe the crystal graph.

---

# 14.61 Creating a `Data` Object

The `Data` class is imported from PyTorch Geometric.

```python
from torch_geometric.data import Data
```

Suppose we have already constructed

- `x`,
- `edge_index`,
- `edge_attr`,
- `target`.

The graph is created using

```python
graph = Data(

    x=x,

    edge_index=edge_index,

    edge_attr=edge_attr,

    y=target

)
```

This single object now contains the entire crystal graph.

Every graph neural network layer in PyTorch Geometric expects its input in this format.

---

# 14.62 Understanding Each Component

Let us examine the constructor carefully.

```python
x=x
```

stores the node feature matrix.

Each row corresponds to one atom.

For example,

```
Atom 1

↓

92 Features
```

```
Atom 2

↓

92 Features
```

The matrix therefore contains one feature vector per atom.

---

```python
edge_index=edge_index
```

stores graph connectivity.

Every column represents one graph edge.

For example,

```
0 → 5
```

means that atom

```
0
```

can exchange information with atom

```
5
```

during message passing.

---

```python
edge_attr=edge_attr
```

stores the Gaussian-expanded edge features.

Every row corresponds to one graph edge.

For example,

```
Edge

↓

31 Gaussian Features
```

These edge features describe the interaction between neighboring atoms.

---

```python
y=target
```

stores the property that the network should predict.

For regression problems,

this may be

- formation energy,
- band gap,
- bulk modulus,
- shear modulus,
- elastic tensor components.

For classification problems,

it may represent

- metal vs semiconductor,
- stable vs unstable,
- magnetic vs non-magnetic.

---

# 14.63 Example: Constructing a Complete Crystal Graph

The following simplified example illustrates the complete workflow.

```python
from torch_geometric.data import Data

graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_attr,

    y=torch.tensor(

        [-2.41],

        dtype=torch.float

    )

)

print(graph)
```

Typical output

```text
Data(

x=[40,92],

edge_index=[2,480],

edge_attr=[480,31],

y=[1]

)
```

This summary immediately provides useful information.

The graph contains

- 40 atoms,
- 480 edges,
- 92 node features,
- 31 edge features,
- one target value.

Although compact,

this object contains everything required for graph neural network training.

---

# 14.64 Inspecting Individual Graph Components

The stored tensors remain accessible as normal attributes.

For example,

the node feature matrix is obtained using

```python
print(graph.x)
```

Graph connectivity is obtained using

```python
print(graph.edge_index)
```

Edge features are obtained using

```python
print(graph.edge_attr)
```

The target property is accessed using

```python
print(graph.y)
```

Since every attribute is a PyTorch tensor,

standard tensor operations can be applied directly.

For example,

```python
print(graph.x.shape)

print(graph.edge_attr.shape)

print(graph.edge_index.shape)
```

This makes debugging much easier before training begins.

---

# 14.65 Why the `Data` Object Is So Powerful

At first glance,

the `Data` object appears to be little more than a dictionary containing tensors.

In reality,

it is the foundation upon which the entire PyTorch Geometric ecosystem is built.

Because every graph shares the same standardized format,

the same neural network can process

- small graphs,
- large graphs,
- molecular graphs,
- crystal graphs,
- protein graphs,
- social networks.

The architecture itself remains unchanged.

Only the graph data changes.

This abstraction allows researchers to focus on designing graph neural networks rather than repeatedly writing custom graph-processing code.

---

# 14.66 Verifying the Constructed Graph

Before training any graph neural network,

the graph should always be validated.

Several simple checks are recommended.

First,

confirm the number of atoms.

```python
print(

    graph.num_nodes

)
```

Next,

confirm the number of edges.

```python
print(

    graph.num_edges

)
```

Then,

verify the dimensions of the feature matrices.

```python
print(graph.x.shape)

print(graph.edge_attr.shape)
```

Finally,

ensure that the target property has been loaded correctly.

```python
print(graph.y)
```

These simple validation steps often reveal mistakes during graph construction that would otherwise be difficult to diagnose during model training.

---

# 14.67 One Crystal Is Not Enough

The graph constructed so far represents **only one crystal**.

Machine learning, however, requires many training examples.

Suppose we have

```
Crystal 1

↓

Graph 1
```

```
Crystal 2

↓

Graph 2
```

```
Crystal 3

↓

Graph 3
```

Eventually,

our dataset may contain

```
50,000

or even

100,000
```

individual crystal graphs.

Managing such large collections manually would be impractical.

PyTorch Geometric therefore provides specialized dataset classes and data loaders that automatically organize, batch, and efficiently feed crystal graphs to the neural network during training.

Constructing this scalable data pipeline is the next major step toward implementing the Crystal Graph Convolutional Neural Network from scratch.

# 14.68 Building a Crystal Graph Dataset

A graph neural network cannot be trained using a single crystal.

Like every machine learning model, CGCNN learns by observing many examples.

Suppose our goal is to predict the formation energy of crystalline materials.

Instead of one crystal,

we may have

```
Crystal 1

Formation Energy = -3.12 eV/atom
```

```
Crystal 2

Formation Energy = -2.54 eV/atom
```

```
Crystal 3

Formation Energy = -1.87 eV/atom
```

```
...

50,000 Crystals
```

Every crystal must undergo exactly the same preprocessing pipeline.

```
Crystal Structure

↓

Read CIF File

↓

Neighbor Search

↓

Construct Graph

↓

Generate Node Features

↓

Generate Edge Features

↓

Create Data Object

↓

Dataset
```

The graph neural network never processes raw CIF files directly.

Instead,

it receives a collection of graph objects.

---

# 14.69 Why We Need a Dataset Class

Imagine manually loading

```
50,000
```

crystal graphs during every training epoch.

The workflow would quickly become unmanageable.

Instead,

PyTorch Geometric provides dataset classes that

- organize graph data,
- load graphs automatically,
- support indexing,
- enable efficient batching,
- integrate seamlessly with DataLoader.

Conceptually,

```
Thousands of Graphs

↓

Dataset

↓

DataLoader

↓

Graph Neural Network
```

The dataset acts as an interface between stored graph files and the neural network.

---

# 14.70 Organizing the Raw Data

Before writing code,

it is useful to organize the project directory.

A common structure is

```text
CGCNN_Project/

│

├── data/

│   ├── cif/

│   │     LiCoO2.cif

│   │     Si.cif

│   │     NaCl.cif

│   │     ...

│   │

│   └── targets.csv

│

├── graphs/

│

├── models/

│

├── train.py

│

└── dataset.py
```

The directory

```
cif/
```

contains the crystal structures.

The file

```
targets.csv
```

stores the material properties that will become prediction targets.

For example,

```text
material_id,formation_energy

LiCoO2,-2.41

Si,-5.42

NaCl,-3.67
```

Each row links one crystal structure with its corresponding property.

---

# 14.71 Creating a Custom Dataset

PyTorch Geometric datasets inherit from

```python
InMemoryDataset
```

or

```python
Dataset
```

For educational purposes,

we begin with the simpler

```python
Dataset
```

class.

```python
from torch_geometric.data import Dataset

class CrystalDataset(Dataset):

    def __init__(

        self,

        root,

        transform=None,

        pre_transform=None

    ):

        super().__init__(

            root,

            transform,

            pre_transform

        )
```

Although this class appears simple,

it provides the framework for managing thousands of crystal graphs efficiently.

---

# 14.72 Required Methods

Every custom dataset must implement several methods.

The first specifies how many samples exist.

```python
def len(self):

    return len(self.graph_files)
```

Suppose

```
25,000
```

graph files have been generated.

Then

```python
dataset.len()
```

returns

```text
25000
```

This information allows PyTorch to determine the dataset size automatically.

---

The second required method retrieves one graph.

```python
def get(self, index):

    graph = torch.load(

        self.graph_files[index]

    )

    return graph
```

Whenever the neural network requests

```
Graph 157
```

the dataset loads only that graph from disk.

This lazy loading strategy dramatically reduces memory usage for very large datasets.

---

# 14.73 Complete Dataset Skeleton

Combining these methods produces a functional dataset.

```python
import os

import torch

from torch_geometric.data import Dataset

class CrystalDataset(Dataset):

    def __init__(

        self,

        graph_dir

    ):

        super().__init__()

        self.graph_dir = graph_dir

        self.graph_files = sorted(

            [

                os.path.join(

                    graph_dir,

                    file

                )

                for file in os.listdir(graph_dir)

                if file.endswith(".pt")

            ]

        )

    def len(self):

        return len(self.graph_files)

    def get(

        self,

        idx

    ):

        return torch.load(

            self.graph_files[idx]

        )
```

This class assumes that every crystal graph has already been saved as a PyTorch file.

Each call to

```python
dataset[i]
```

returns one complete `Data` object.

---

# 14.74 Saving Crystal Graphs

After constructing a graph,

it is convenient to save it for future use.

```python
torch.save(

    graph,

    "graphs/LiCoO2.pt"

)
```

The graph is serialized into a binary file.

Later,

it can be reloaded instantly.

```python
graph = torch.load(

    "graphs/LiCoO2.pt"

)
```

This approach avoids reconstructing graphs every time the model is trained.

Since graph construction can be computationally expensive,

saving processed graphs significantly accelerates repeated experiments.

---

# 14.75 Loading the Dataset

Once the dataset class has been defined,

loading every graph becomes straightforward.

```python
dataset = CrystalDataset(

    graph_dir="graphs"

)

print(

    len(dataset)

)
```

Suppose

```
18,420
```

graphs have been stored.

The output becomes

```text
18420
```

Individual graphs can now be accessed by index.

```python
graph = dataset[0]

print(graph)
```

Typical output

```text
Data(

x=[36,92],

edge_index=[2,428],

edge_attr=[428,31],

y=[1]

)
```

Each sample returned by the dataset is already prepared for graph neural network training.

---

# 14.76 Inspecting Dataset Samples

Before beginning model training,

it is good practice to inspect several randomly selected graphs.

```python
for i in range(5):

    graph = dataset[i]

    print(

        "Graph",

        i

    )

    print(

        "Nodes:",

        graph.num_nodes

    )

    print(

        "Edges:",

        graph.num_edges

    )

    print(

        "Target:",

        graph.y

    )

    print()
```

A possible output is

```text
Graph 0

Nodes: 40

Edges: 482

Target: tensor([-2.41])

Graph 1

Nodes: 24

Edges: 290

Target: tensor([-5.42])

Graph 2

Nodes: 32

Edges: 376

Target: tensor([-3.67])
```

Notice that different crystals naturally contain different numbers of atoms and edges.

Unlike conventional neural networks,

graph neural networks can process graphs of varying sizes without requiring every sample to have identical dimensions.

This flexibility is one of the major strengths of graph-based deep learning.

---

# 14.77 Preparing for Mini-Batch Training

Although the dataset now contains thousands of crystal graphs,

training the neural network one graph at a time would be inefficient.

Modern GPUs achieve their highest performance by processing many training samples simultaneously.

Consequently,

graph neural networks use **mini-batch training**.

Instead of

```
One Graph

↓

Forward Pass

↓

Backward Pass
```

the model processes

```
Graph 1

Graph 2

Graph 3

...

Graph 32

↓

One Mini-Batch

↓

Forward Pass

↓

Backward Pass
```

Batching graph data is considerably more complex than batching images because every graph contains a different number of nodes and edges.

PyTorch Geometric provides specialized data loaders that automatically merge multiple graphs into a single batch while preserving their individual identities.

Understanding this batching mechanism is essential before implementing the Crystal Graph Convolutional Neural Network itself.

# 14.78 Mini-Batch Training for Graph Neural Networks

In conventional deep learning, mini-batch training is straightforward because every sample has exactly the same dimensions.

For example, consider image classification.

Suppose every image has a resolution of

```
224 × 224 × 3
```

A batch containing

```
32 images
```

can simply be stacked into a tensor with shape

```text
(32, 224, 224, 3)
```

Every image has identical dimensions, so batching is trivial.

Crystal graphs are fundamentally different.

One crystal may contain

```
18 atoms
```

Another may contain

```
76 atoms
```

A third may contain

```
240 atoms
```

Their graph sizes are completely different.

Consequently, they cannot simply be stacked like images.

Graph neural network libraries therefore use a completely different batching strategy.

---

# 14.79 The Idea Behind Graph Batching

Instead of padding graphs until they have identical sizes, PyTorch Geometric constructs one **large disconnected graph**.

Suppose we have three graphs.

```
Graph A

3 Nodes
```

```
Graph B

5 Nodes
```

```
Graph C

4 Nodes
```

Instead of processing them separately,

PyTorch Geometric combines them into

```
Large Graph

↓

Graph A

Graph B

Graph C
```

Importantly,

no edges are created between different graphs.

Each original graph remains completely isolated.

Conceptually,

```
Graph A

○──○

│

○



Graph B

○──○──○

│

○──○



Graph C

○──○

│

○──○
```

After batching,

all three graphs coexist inside one larger graph,

but there are **no connections between them**.

The graph neural network therefore processes many samples simultaneously while preserving the integrity of every crystal.

---

# 14.80 Why Disconnected Graphs Work

Graph neural networks exchange information only through graph edges.

Since no edge connects

```
Graph A
```

to

```
Graph B,
```

messages cannot propagate between different crystals.

Consequently,

every graph behaves exactly as though it were processed independently.

This clever idea eliminates the need for

- graph padding,
- resizing,
- truncation.

Instead,

graphs of arbitrary size can be processed efficiently in parallel.

---

# 14.81 The Batch Vector

After multiple graphs are merged,

the neural network must still know which atoms belong to which crystal.

PyTorch Geometric stores this information in a tensor called

```python
batch
```

Suppose we batch three graphs.

```
Graph 1

3 atoms
```

```
Graph 2

2 atoms
```

```
Graph 3

4 atoms
```

The resulting batch vector becomes

```text
tensor([

0,

0,

0,

1,

1,

2,

2,

2,

2

])
```

Each number identifies the graph to which an atom belongs.

The first three atoms belong to graph

```
0.
```

The next two atoms belong to graph

```
1.
```

The final four atoms belong to graph

```
2.
```

Although all atoms are stored inside one tensor,

the batch vector preserves the identity of every individual crystal.

---

# 14.82 Creating a DataLoader

PyTorch Geometric provides a specialized DataLoader for batching graph data.

It is imported using

```python
from torch_geometric.loader import DataLoader
```

Suppose

```python
dataset
```

contains

```
18,420
```

crystal graphs.

A DataLoader is created as follows.

```python
from torch_geometric.loader import DataLoader

train_loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=True

)
```

This loader automatically

- selects graphs,
- merges them,
- constructs the batch vector,
- feeds the batch into the neural network.

The user does not need to implement batching manually.

---

# 14.83 Understanding the DataLoader Parameters

The first argument,

```python
dataset
```

specifies the collection of crystal graphs.

The parameter

```python
batch_size=32
```

requests that

```
32 graphs
```

be processed simultaneously during every optimization step.

Larger batch sizes generally improve GPU utilization,

although they also require more memory.

The parameter

```python
shuffle=True
```

randomizes the graph order at the beginning of every training epoch.

Random shuffling prevents the neural network from learning undesirable ordering patterns and generally improves optimization.

---

# 14.84 Inspecting a Mini-Batch

Let us inspect the contents of one mini-batch.

```python
batch = next(

    iter(train_loader)

)

print(batch)
```

A possible output is

```text
DataBatch(

x=[1348,92],

edge_index=[2,16542],

edge_attr=[16542,31],

y=[32],

batch=[1348]

)
```

Although the batch contains

```
32 crystal graphs,
```

PyTorch Geometric presents them as one large disconnected graph.

The output indicates

- 1348 atoms,
- 16,542 graph edges,
- 32 target values.

The

```python
batch
```

tensor records which atom belongs to which crystal.

---

# 14.85 Understanding the Shapes

Suppose the output is

```text
x = [1348,92]
```

This means

- 1348 atoms,
- 92 node features per atom.

Similarly,

```text
edge_attr = [16542,31]
```

means

- 16,542 graph edges,
- 31 Gaussian edge features.

Finally,

```text
y = [32]
```

contains

```
32 target values,
```

one for each crystal.

Notice that the target tensor is indexed by graphs,

whereas node and edge tensors are indexed by atoms and edges.

This distinction is essential when implementing graph-level prediction models such as CGCNN.

---

# 14.86 Accessing the Batch Vector

The batch assignment tensor can be examined directly.

```python
print(batch.batch)
```

A typical output may resemble

```text
tensor([

0,

0,

0,

0,

...

1,

1,

1,

...

2,

2,

...

31

])
```

Every entry corresponds to one atom.

If

```python
batch.batch[150]
```

returns

```text
7
```

then

```
Atom 150
```

belongs to

```
Graph 7.
```

This information becomes essential during graph pooling,

where atomic representations must be combined into one crystal representation.

---

# 14.87 Moving Batches to the GPU

Training large graph neural networks on the CPU is often prohibitively slow.

Fortunately,

the entire batch can be transferred to the GPU exactly like an ordinary PyTorch tensor.

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

batch = batch.to(device)
```

This command moves

- node features,
- graph connectivity,
- edge features,
- batch assignments,
- target values

to GPU memory simultaneously.

No additional code is required.

---

# 14.88 The Complete Data Pipeline

At this stage,

we have constructed the complete preprocessing pipeline required before graph neural network training.

The workflow can now be summarized as

```text
Crystal Structure (.cif)

↓

Read with Pymatgen

↓

Neighbor Search

↓

Periodic Boundary Conditions

↓

Node Features

↓

Edge Construction

↓

Gaussian Distance Expansion

↓

Data Object

↓

Dataset

↓

DataLoader

↓

Mini-Batch

↓

Graph Neural Network
```

Every CGCNN implementation follows this overall sequence.

Although implementation details may differ,

the underlying data pipeline remains essentially the same.

The next major stage of this chapter begins the neural network itself.

We will start by revisiting the mathematical foundations of graph convolution before deriving the original Crystal Graph Convolutional Network proposed by Xie and Grossman.

# 14.89 From Crystal Graphs to Graph Convolution

At this point, we have completed the entire graph construction pipeline.

Starting from a crystal structure, we have learned how to

- represent atoms as nodes,
- construct node features,
- identify neighboring atoms,
- account for periodic boundary conditions,
- create graph connectivity,
- generate edge features,
- organize graphs into datasets,
- prepare mini-batches for training.

Although this is a significant accomplishment, the graph itself does not perform any prediction.

A graph is simply a structured representation of the crystal.

The crucial question now becomes

> **How does a neural network learn from this graph?**

The answer lies in the concept of **graph convolution**.

Graph convolution is the fundamental operation that allows information to flow through the graph.

Just as convolutional neural networks learn from neighboring pixels in an image, graph neural networks learn from neighboring nodes in a graph.

Understanding graph convolution is essential because it forms the computational core of the Crystal Graph Convolutional Neural Network.

---

# 14.90 Why Ordinary Neural Networks Cannot Process Graphs

Before introducing graph convolution, it is useful to understand why a conventional neural network is insufficient.

Consider a standard feedforward neural network.

```
Input

↓

Hidden Layer

↓

Hidden Layer

↓

Output
```

The network assumes that every input sample has the same fixed structure.

For example,

```
Image

↓

224 × 224 × 3
```

or

```
Feature Vector

↓

150 Features
```

Every sample contains exactly the same number of input values.

Crystal graphs violate this assumption.

One material may contain

```
18 atoms
```

while another contains

```
240 atoms.
```

Similarly,

the number of neighboring atoms varies from one crystal to another.

Because both the number of nodes and the graph topology change from sample to sample,

there is no fixed input arrangement that a conventional neural network can process.

A new computational framework is therefore required.

---

# 14.91 Revisiting Convolution in Images

To understand graph convolution, let us first recall how convolution operates in images.

Consider a small image.

```text
8   6   5

7   9   2

4   3   1
```

A convolutional neural network applies a small filter,

for example,

```text
1   0   1

0   1   0

1   0   1
```

The filter moves across the image,

combining information from neighboring pixels.

Each output pixel depends only on its local neighborhood.

This process allows the CNN to detect

- edges,
- textures,
- corners,
- shapes,

and progressively more complex visual features.

The key idea is that **neighboring pixels exchange information**.

---

# 14.92 The Key Observation

Suppose we ignore the image itself and focus only on pixel connectivity.

Every interior pixel communicates with its surrounding pixels.

Conceptually,

```text
        ○

      ○ ○ ○

        ●

      ○ ○ ○

        ○
```

The central pixel receives information from its neighbors.

Now consider a crystal graph.

```text
        O

      /   \

Li —— Co —— O

      \   /

        O
```

The cobalt atom also has neighboring atoms.

Although the geometry differs,

the underlying idea is identical.

The representation of the central atom should depend on the information contained in neighboring atoms.

This observation leads directly to graph convolution.

---

# 14.93 The Philosophy of Graph Convolution

Graph convolution replaces the concept of a moving image filter with a more general operation.

Instead of scanning across pixels,

the neural network repeatedly performs the following steps.

1. Gather information from neighboring atoms.

2. Combine this information with the current atom.

3. Update the atom's representation.

This process is repeated for every atom in the graph.

Conceptually,

```text
Neighbor Information

↓

Aggregate

↓

Combine with Current Atom

↓

Updated Atom Representation
```

After one graph convolution,

every atom contains information about its immediate neighbors.

After two graph convolutions,

every atom indirectly contains information about neighbors of neighbors.

After several graph convolutions,

each atom gradually acquires knowledge about an increasingly larger region of the crystal.

---

# 14.94 Atomic Feature Evolution

Initially,

every atom is represented only by its handcrafted input features.

For example,

```
Lithium

↓

Atomic Number

↓

Atomic Radius

↓

Electronegativity
```

After the first graph convolution,

its representation changes.

```
Lithium

↓

Own Features

+

Neighbor Information

↓

Updated Representation
```

After another graph convolution,

the representation changes again.

```
Updated Representation

+

Second-Hop Neighbor Information

↓

New Representation
```

This iterative refinement is one of the defining characteristics of graph neural networks.

Instead of remaining fixed,

the feature vector associated with every atom evolves throughout the network.

---

# 14.95 Message Passing: The Core Idea

Graph convolution is often described using the language of **message passing**.

The terminology is highly intuitive.

Every neighboring atom sends a **message** to the central atom.

The central atom collects these messages,

combines them,

and updates its internal representation.

Conceptually,

```text
Neighbor 1

↓

Message

↘

      Central Atom

↗

Message

↓

Neighbor 2

↖

Message

↓

Neighbor 3

↓

Updated Representation
```

The messages are not literal text.

They are numerical vectors computed from

- node features,
- edge features,
- learned neural network parameters.

The neural network automatically learns what information should be transmitted.

---

# 14.96 Why Multiple Graph Convolution Layers Are Needed

Suppose we use only one graph convolution layer.

Each atom can communicate only with its immediate neighbors.

For example,

```text
A —— B —— C
```

After one convolution,

```
B
```

knows about

```
A

and

C.
```

However,

```
A
```

still knows nothing about atoms farther away.

Now suppose we apply a second graph convolution.

During the first layer,

```
B
```

learned information from

```
C.
```

During the second layer,

```
A
```

receives information from

```
B,
```

which already contains information about

```
C.
```

Consequently,

```
A
```

has now indirectly learned about

```
C.
```

This phenomenon is known as **receptive field expansion**.

Each additional graph convolution layer increases the region of the crystal that influences every atom.

---

# 14.97 Information Flow in Crystal Graphs

Consider the following simplified crystal graph.

```text
       O

      / \

Li —— Co —— O

      \ /

       O
```

Initially,

the cobalt atom contains only its own atomic features.

After one graph convolution,

its representation incorporates information from

- lithium,
- oxygen,
- oxygen,
- oxygen.

After a second graph convolution,

the cobalt representation also includes information from atoms connected to those neighboring atoms.

As graph convolution layers accumulate,

each atomic representation gradually captures both

- local chemistry,
- larger structural context.

This ability to combine local and long-range information is one of the primary reasons graph neural networks are so effective for crystalline materials.

---

# 14.98 Toward the Original CGCNN Equation

Although the concept of message passing is intuitive,

a neural network ultimately requires mathematical equations.

The original Crystal Graph Convolutional Neural Network introduced a specific update rule describing

- how messages are computed,
- how neighboring information is aggregated,
- how atomic representations are updated.

Understanding this update equation is essential because it distinguishes CGCNN from other graph neural network architectures.

In the next section, we will derive the original graph convolution equation proposed by Xie and Grossman, examine every mathematical term in detail, and implement it step by step in PyTorch.

# 14.99 The Original CGCNN Convolution Equation

We have now reached the mathematical core of the Crystal Graph Convolutional Neural Network.

Everything discussed so far—including graph construction, node features, edge features, and batching—exists to support one operation:

**graph convolution**.

The original CGCNN paper by **Tian Xie and Jeffrey C. Grossman (2018)** introduced a graph convolution operator specifically designed for crystalline materials.

Unlike traditional Graph Convolutional Networks (GCNs), the CGCNN convolution explicitly incorporates

- neighboring atoms,
- edge features (interatomic distances),
- nonlinear neural networks,
- gating mechanisms.

The graph convolution layer repeatedly updates the representation of every atom until it contains enough structural information to predict material properties.

---

# 14.100 The Goal of Graph Convolution

Suppose every atom initially has a feature vector

$$
\mathbf{v}_i^{(0)}
$$

where

- \(i\) denotes the atom,
- \(0\) denotes the initial layer.

Initially,

this vector contains only handcrafted atomic descriptors such as

- atomic number,
- electronegativity,
- atomic radius,
- valence electrons.

Our objective is to transform

$$
\mathbf{v}_i^{(0)}
$$

into a richer representation

$$
\mathbf{v}_i^{(1)}
$$

that includes information from neighboring atoms.

After another graph convolution,

we obtain

$$
\mathbf{v}_i^{(2)}
$$

which contains information from a larger neighborhood.

Repeating this process several times allows each atom to gradually encode increasingly complex structural information.

Conceptually,

```text
Initial Atom Features

↓

Graph Convolution

↓

Updated Features

↓

Graph Convolution

↓

Higher-Level Features

↓

Graph Convolution

↓

Learned Atomic Representation
```

---

# 14.101 Notation Used in CGCNN

Before examining the equation, we must define the symbols used throughout the paper.

Let

$$
\mathbf{v}_i^{(t)}
$$

denote the feature vector of atom

$$
i
$$

after

$$
t
$$

graph convolution layers.

Suppose atom

$$
i
$$

has neighboring atoms

$$
j \in N(i),
$$

where

$$
N(i)
$$

represents the neighborhood of atom

$$
i.
$$

For every neighboring pair,

the edge feature is

$$
\mathbf{u}_{ij},
$$

which in CGCNN is primarily derived from the Gaussian-expanded interatomic distance.

Finally,

the graph convolution computes a new feature vector

$$
\mathbf{v}_i^{(t+1)}.
$$

The entire purpose of graph convolution is therefore to transform

$$
\mathbf{v}_i^{(t)}
$$

into

$$
\mathbf{v}_i^{(t+1)}.
$$

---

# 14.102 The Original CGCNN Update Equation

The update rule proposed in the original paper is

$$
\boxed{
\mathbf{v}_i^{(t+1)}
=
\mathbf{v}_i^{(t)}
+
\sum_{j \in N(i)}
\sigma(\mathbf{z}_{ij}^{(t)}W_f+\mathbf{b}_f)
\odot
g(\mathbf{z}_{ij}^{(t)}W_s+\mathbf{b}_s)
}
$$

Although this equation appears intimidating at first glance, it is simply a structured way of combining information from neighboring atoms.

Each component has a specific physical and computational purpose.

The remainder of this section explains every term in detail.

---

# 14.103 Constructing the Neighbor Feature Vector

The first step is to combine the information associated with an edge.

CGCNN forms a concatenated feature vector

$$
\mathbf{z}_{ij}^{(t)}
=
\mathbf{v}_i^{(t)}
\oplus
\mathbf{v}_j^{(t)}
\oplus
\mathbf{u}_{ij},
$$

where

$$
\oplus
$$

denotes vector concatenation.

This means that the network joins

- the current atom,
- the neighboring atom,
- the edge connecting them,

into one feature vector.

Conceptually,

```text
Current Atom Features

↓

Neighbor Features

↓

Edge Features

↓

Concatenate

↓

zᵢⱼ
```

Instead of processing node and edge information separately,

CGCNN processes them together.

This allows the neural network to learn interaction patterns that depend simultaneously on

- atomic identities,
- neighboring chemistry,
- interatomic distances.

---

# 14.104 Why Concatenation Is Important

Suppose we wish to evaluate the interaction between

```
Lithium

↓

Oxygen
```

If the network receives only the lithium feature vector,

it cannot distinguish whether lithium interacts with

- oxygen,
- cobalt,
- fluorine.

Similarly,

if the network receives only the oxygen feature vector,

it lacks information about the central atom.

Finally,

knowing only the two atoms is insufficient because

```
Li—O

Distance = 1.9 Å
```

and

```
Li—O

Distance = 4.8 Å
```

represent very different interactions.

By concatenating

- both node feature vectors,
- the edge feature vector,

CGCNN provides the neural network with all relevant local information.

---

# 14.105 The Filter Network

Once the concatenated vector

$$
\mathbf{z}_{ij}
$$

has been created,

it is passed through the first neural network,

called the **filter network**.

Mathematically,

$$
\mathbf{f}_{ij}
=
\sigma
(
\mathbf{z}_{ij}W_f
+
\mathbf{b}_f
),
$$

where

- \(W_f\) is a learnable weight matrix,
- \(\mathbf{b}_f\) is a learnable bias vector,
- \(\sigma\) is the sigmoid activation function.

The sigmoid function produces values between

```
0

and

1.
```

Consequently,

the filter network acts as a **gate**.

It determines how much information should be transmitted through each edge.

Large values correspond to important interactions,

whereas values near zero suppress weak or irrelevant interactions.

---

# 14.106 The Core Network

The concatenated feature vector is also passed through a second neural network,

called the **core network**.

Its output is

$$
\mathbf{s}_{ij}
=
g
(
\mathbf{z}_{ij}W_s
+
\mathbf{b}_s
),
$$

where

- \(W_s\) and \(\mathbf{b}_s\) are learnable parameters,
- \(g\) is typically the Softplus activation function in the original CGCNN implementation.

Unlike the filter network,

which decides **how much** information should pass,

the core network determines **what information** should be transmitted.

It generates the candidate message associated with the neighboring atom.

---

# 14.107 Combining the Two Networks

The outputs of the filter network and the core network are multiplied element-wise.

Mathematically,

$$
\mathbf{m}_{ij}
=
\mathbf{f}_{ij}
\odot
\mathbf{s}_{ij},
$$

where

$$
\odot
$$

denotes the Hadamard (element-wise) product.

Conceptually,

```text
Filter Output

↓

Importance Weight

×

Core Output

↓

Candidate Message

↓

Final Message
```

If the filter output is close to zero,

the message is almost completely suppressed.

If the filter output is close to one,

the message passes through almost unchanged.

Thus,

the filter network acts as an adaptive attention mechanism that controls information flow between neighboring atoms.

---

# 14.108 Why CGCNN Uses Two Neural Networks

One might ask why the model employs two separate neural networks instead of a single multilayer perceptron.

The answer lies in the different roles they play.

The filter network decides

> **Should this neighboring interaction be important?**

The core network decides

> **What information should be sent if it is important?**

Separating these responsibilities allows the model to learn both

- the strength of atomic interactions,
- the content of atomic interactions,

independently.

This design significantly improves the expressive power of the graph convolution and is one of the distinctive characteristics of the original CGCNN architecture.

In the next section, we will complete the convolution operation by examining how messages from all neighboring atoms are aggregated, how the residual update is performed, and how the new atomic feature vector is produced.

# 14.109 Aggregating Messages from Neighboring Atoms

In the previous section, we derived the message associated with a single neighboring atom.

For one neighboring atom \(j\), the message is

$$
\mathbf{m}_{ij}
=
\mathbf{f}_{ij}
\odot
\mathbf{s}_{ij}.
$$

However, a crystal atom rarely has only one neighbor.

For example,

```
           O

         /   \

      O  Co   O

         \   /

          Li
```

The cobalt atom simultaneously interacts with several neighboring atoms.

Each neighboring atom generates its own message.

Therefore,

the neural network must combine all of these messages into a single representation.

This process is called **message aggregation**.

---

# 14.110 Why Aggregation Is Necessary

Suppose atom \(i\) has four neighbors.

```
Neighbor 1

↓

Message 1
```

```
Neighbor 2

↓

Message 2
```

```
Neighbor 3

↓

Message 3
```

```
Neighbor 4

↓

Message 4
```

The neural network cannot update the atom using four separate vectors.

Instead,

these vectors must first be combined into one.

Conceptually,

```text
Message 1

↓

Message 2

↓

Message 3

↓

Message 4

↓

Aggregation

↓

Single Vector
```

Only after aggregation can the atom update its internal representation.

---

# 14.111 Aggregation in the Original CGCNN

The original CGCNN performs aggregation by **summing** all neighboring messages.

Mathematically,

$$
\mathbf{M}_i
=
\sum_{j\in N(i)}
\mathbf{m}_{ij}.
$$

Substituting the message equation,

$$
\mathbf{M}_i
=
\sum_{j\in N(i)}
\sigma(\mathbf{z}_{ij}W_f+\mathbf{b}_f)
\odot
g(\mathbf{z}_{ij}W_s+\mathbf{b}_s).
$$

This summation produces one aggregated message vector for every atom.

Notice that the order of neighboring atoms does not matter.

Whether neighbors are processed as

```
A

↓

B

↓

C
```

or

```
C

↓

A

↓

B
```

the final sum remains identical.

This property is known as **permutation invariance**, which is an essential requirement for graph neural networks.

---

# 14.112 Why Sum Aggregation Works Well

Several aggregation operators are possible.

For example,

- sum,
- mean,
- maximum,
- weighted attention.

The original CGCNN uses **sum aggregation** because it naturally captures the cumulative influence of neighboring atoms.

Suppose two neighboring oxygen atoms contribute similar messages.

Using summation,

both contributions strengthen the final representation.

In contrast,

taking only the maximum would ignore one of the neighbors.

Likewise,

averaging could reduce the magnitude of important interactions.

Sum aggregation therefore preserves the combined effect of the local atomic environment.

---

# 14.113 Residual Feature Update

After aggregation,

the atom possesses a combined neighboring message,

$$
\mathbf{M}_i.
$$

A simple approach would replace the current atomic feature vector with this new information.

However,

doing so would discard valuable information already contained in the atom itself.

Instead,

CGCNN uses a **residual update**.

The new feature vector becomes

$$
\boxed{
\mathbf{v}_i^{(t+1)}
=
\mathbf{v}_i^{(t)}
+
\mathbf{M}_i
}
$$

or equivalently,

$$
\boxed{
\mathbf{v}_i^{(t+1)}
=
\mathbf{v}_i^{(t)}
+
\sum_{j\in N(i)}
\sigma(\mathbf{z}_{ij}W_f+\mathbf{b}_f)
\odot
g(\mathbf{z}_{ij}W_s+\mathbf{b}_s)
}
$$

This is the complete graph convolution equation used in the original CGCNN.

---

# 14.114 Why Residual Connections Are Important

Residual connections were originally popularized in deep convolutional neural networks through ResNet.

CGCNN adopts the same principle.

Instead of computing

```text
New Features

=

Neighbor Messages
```

the model computes

```text
New Features

=

Old Features

+

Neighbor Messages
```

This seemingly simple modification provides several advantages.

First,

the original atomic information is preserved.

Second,

the network learns **corrections** to the existing representation rather than completely replacing it.

Third,

residual connections improve gradient flow during backpropagation,

making it possible to train deeper graph neural networks.

Without residual connections,

information from the original atomic descriptors could gradually disappear after many graph convolution layers.

---

# 14.115 Step-by-Step View of One CGCNN Layer

The entire computation performed by one CGCNN layer can now be summarized as follows.

```text
Current Atom Features

↓

Find Neighboring Atoms

↓

Concatenate

(Current Atom,

 Neighbor Atom,

 Edge Features)

↓

Filter Network

↓

Core Network

↓

Element-wise Multiplication

↓

Neighbor Message

↓

Sum Over All Neighbors

↓

Residual Addition

↓

Updated Atom Features
```

Every atom in the graph performs this computation simultaneously.

Because all weight matrices are shared across every edge,

the model can generalize to crystals containing different numbers of atoms.

---

# 14.116 Example: Updating One Atom

Suppose atom \(i\) has three neighbors.

The three message vectors are

```text
Neighbor 1

↓

[0.20, 0.31, 0.15]
```

```text
Neighbor 2

↓

[0.42, 0.18, 0.26]
```

```text
Neighbor 3

↓

[0.11, 0.40, 0.22]
```

Summing them gives

```text
Aggregated Message

↓

[0.73, 0.89, 0.63]
```

Assume the current atomic feature vector is

```text
Current Features

↓

[1.25, 0.94, 1.12]
```

Applying the residual update,

```text
Updated Features

=

Current Features

+

Aggregated Message

=

[1.98, 1.83, 1.75]
```

The updated representation now contains information from both the atom itself and its neighboring environment.

---

# 14.117 Layer-by-Layer Information Propagation

One graph convolution layer allows each atom to communicate with its immediate neighbors.

```
Layer 1

A ←→ B ←→ C
```

After the first layer,

atom

```
B
```

contains information from

```
A

and

C.
```

Applying a second layer,

```
Layer 2
```

allows information from

```
A
```

to reach

```
C
```

through

```
B.
```

As additional graph convolution layers are stacked,

each atom gradually incorporates information from increasingly distant regions of the crystal.

This expanding receptive field enables the model to capture both

- local chemical bonding,

and

- larger structural motifs.

---

# 14.118 Completing One Graph Convolution Layer

At the end of one CGCNN layer,

every atom has received,

filtered,

aggregated,

and incorporated information from its neighboring atoms.

The graph itself remains unchanged.

Only the node feature matrix changes.

Conceptually,

```text
Graph Topology

↓

Unchanged
```

```text
Node Features

↓

Updated
```

```text
Edge Features

↓

Unchanged
```

This process is repeated several times,

allowing atomic representations to become progressively richer.

After the final graph convolution layer,

each atom possesses a learned embedding that captures both its intrinsic properties and its structural environment.

These learned atomic embeddings form the basis for predicting crystal properties.

The next step is to implement the complete CGCNN convolution layer from scratch in PyTorch, translating every mathematical equation derived above into executable code.

# 14.119 Implementing the CGCNN Convolution Layer in PyTorch

We have now derived the complete mathematical formulation of the Crystal Graph Convolutional Neural Network.

The next step is to translate these equations into executable PyTorch code.

Rather than using a prebuilt implementation, we will construct the convolution layer ourselves so that every mathematical operation is completely transparent.

Our implementation will closely follow the original CGCNN paper while using the modern PyTorch Geometric framework.

---

# 14.120 Reviewing the Computation Pipeline

Before writing code, let us summarize the computation performed by one CGCNN layer.

For every edge in the graph,

the layer performs the following sequence of operations.

```text
Current Atom Features

↓

Neighbor Atom Features

↓

Edge Features

↓

Concatenate

↓

Filter Network

↓

Core Network

↓

Element-wise Multiplication

↓

Neighbor Message

↓

Aggregate Messages

↓

Residual Update

↓

Updated Atom Features
```

Every line of code that follows corresponds directly to one step in this workflow.

---

# 14.121 Creating a Custom CGCNN Layer

In PyTorch Geometric, most graph convolution layers inherit from

```python
MessagePassing
```

This base class provides the infrastructure required for

- message passing,
- aggregation,
- graph traversal.

We begin by importing the required libraries.

```python
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
```

Next, we define our own convolution layer.

```python
class CGCNNConv(MessagePassing):

    def __init__(

        self,

        node_dim,

        edge_dim

    ):

        super().__init__(

            aggr="add"

        )
```

Notice that

```python
aggr="add"
```

implements the summation operator used in the original CGCNN equation.

---

# 14.122 Defining the Learnable Networks

The original CGCNN contains two neural networks.

The first is the filter network.

The second is the core network.

Both receive the concatenated feature vector

$$
z_{ij}.
$$

The concatenated vector contains

- current atom features,
- neighboring atom features,
- edge features.

Its dimension is therefore

```text
2 × Node Features

+

Edge Features
```

The filter network is implemented as

```python
self.filter_network = nn.Sequential(

    nn.Linear(

        2 * node_dim + edge_dim,

        node_dim

    ),

    nn.Sigmoid()

)
```

The sigmoid activation ensures that every output value lies between

```
0

and

1,
```

allowing the network to act as an adaptive gate.

---

The core network is

```python
self.core_network = nn.Sequential(

    nn.Linear(

        2 * node_dim + edge_dim,

        node_dim

    ),

    nn.Softplus()

)
```

The Softplus activation is identical to that used in the original CGCNN implementation.

Unlike ReLU,

Softplus is smooth and continuously differentiable,

which helps stabilize optimization.

---

# 14.123 The Forward Function

Every PyTorch module requires a

```python
forward()
```

method.

Its purpose is to initiate message passing.

```python
def forward(

    self,

    x,

    edge_index,

    edge_attr

):

    return self.propagate(

        edge_index,

        x=x,

        edge_attr=edge_attr

    )
```

At first glance,

this function appears surprisingly short.

The reason is that

```python
propagate()
```

automatically performs

- neighborhood traversal,
- message computation,
- aggregation,
- update.

We only need to specify how individual messages should be computed.

---

# 14.124 Understanding `propagate()`

Suppose the graph contains

```
400 edges.
```

Calling

```python
propagate()
```

does **not** simply loop through the edges in Python.

Internally,

PyTorch Geometric performs highly optimized tensor operations.

Conceptually,

the process is

```text
Read edge_index

↓

Identify Every Neighbor Pair

↓

Call message()

↓

Aggregate Messages

↓

Call update()

↓

Return Updated Features
```

This abstraction allows graph neural networks to process very large graphs efficiently.

---

# 14.125 The Message Function

The most important part of the implementation is the

```python
message()
```

function.

It computes the message transmitted across every graph edge.

```python
def message(

    self,

    x_i,

    x_j,

    edge_attr

):
```

PyTorch Geometric automatically supplies the required tensors.

Here,

- `x_i` represents the destination atom,
- `x_j` represents the source (neighboring) atom,
- `edge_attr` contains the Gaussian-expanded edge features.

The user does not need to manually index neighboring atoms.

The framework performs this automatically.

---

# 14.126 Constructing the Concatenated Feature Vector

Inside the message function,

the first operation is concatenation.

```python
z = torch.cat(

    [

        x_i,

        x_j,

        edge_attr

    ],

    dim=1

)
```

Suppose

- node features contain

```
92 features,
```

and

- edge features contain

```
31 features.
```

The concatenated vector therefore has

```text
92

+

92

+

31

=

215 features.
```

Every graph edge produces one such feature vector.

---

# 14.127 Computing the Filter Output

The concatenated vector is passed through the filter network.

```python
filter_output = self.filter_network(

    z

)
```

The output has the same dimensionality as the node feature vector.

Every value lies between

```
0

and

1.
```

Conceptually,

```text
Concatenated Features

↓

Filter Network

↓

Importance Weights
```

These weights determine how much information should pass through the edge.

---

# 14.128 Computing the Core Output

The same concatenated vector is passed through the second neural network.

```python
core_output = self.core_network(

    z

)
```

Unlike the filter network,

the core network generates the candidate message itself.

Conceptually,

```text
Concatenated Features

↓

Core Network

↓

Candidate Message
```

The filter network controls **how much** of this message should be transmitted.

---

# 14.129 Constructing the Final Message

The outputs of the two networks are multiplied element-wise.

```python
message = filter_output * core_output
```

This corresponds exactly to the Hadamard product

$$
\mathbf{f}_{ij}
\odot
\mathbf{s}_{ij}.
$$

The resulting tensor is the final message transmitted from the neighboring atom to the current atom.

---

# 14.130 Automatic Aggregation

Once every edge message has been computed,

PyTorch Geometric automatically aggregates them.

Because we specified

```python
aggr="add"
```

during initialization,

the framework computes

```text
Message 1

+

Message 2

+

Message 3

+

...
```

for every atom.

No additional code is required.

The aggregation exactly matches the summation appearing in the original CGCNN equation.

---

# 14.131 The Update Function

After aggregation,

the framework calls the

```python
update()
```

function.

The aggregated messages are passed as input.

To implement the residual connection,

we write

```python
def update(

    self,

    aggr_out,

    x

):

    return x + aggr_out
```

This directly implements

$$
\mathbf{v}^{(t+1)}
=
\mathbf{v}^{(t)}
+
\mathbf{M}.
$$

Every atom therefore retains its previous representation while incorporating newly aggregated neighborhood information.

---

# 14.132 Complete CGCNN Layer

Combining every component yields a complete graph convolution layer.

```python
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing


class CGCNNConv(MessagePassing):

    def __init__(

        self,

        node_dim,

        edge_dim

    ):

        super().__init__(

            aggr="add"

        )

        input_dim = (

            2 * node_dim

            + edge_dim

        )

        self.filter_network = nn.Sequential(

            nn.Linear(

                input_dim,

                node_dim

            ),

            nn.Sigmoid()

        )

        self.core_network = nn.Sequential(

            nn.Linear(

                input_dim,

                node_dim

            ),

            nn.Softplus()

        )

    def forward(

        self,

        x,

        edge_index,

        edge_attr

    ):

        return self.propagate(

            edge_index,

            x=x,

            edge_attr=edge_attr

        )

    def message(

        self,

        x_i,

        x_j,

        edge_attr

    ):

        z = torch.cat(

            [

                x_i,

                x_j,

                edge_attr

            ],

            dim=1

        )

        filter_output = self.filter_network(

            z

        )

        core_output = self.core_network(

            z

        )

        return filter_output * core_output

    def update(

        self,

        aggr_out,

        x

    ):

        return x + aggr_out
```

This implementation faithfully reproduces the core graph convolution mechanism introduced in the original CGCNN paper.

---

# 14.133 What We Have Implemented

Our custom layer now performs every essential operation required for one CGCNN graph convolution.

Given

- node features,
- edge connectivity,
- edge features,

it automatically

1. Traverses every graph edge.
2. Builds concatenated feature vectors.
3. Computes filter coefficients.
4. Computes candidate messages.
5. Applies gated message passing.
6. Aggregates neighboring messages.
7. Performs a residual update.

After a single forward pass,

every atom possesses an updated embedding that incorporates information from its immediate neighbors.

However,

one graph convolution layer is rarely sufficient to capture the complexity of crystalline materials.

In practice,

multiple CGCNN layers are stacked together to progressively enlarge the receptive field and learn richer atomic representations.

The next section constructs the complete CGCNN architecture by combining multiple graph convolution layers with graph pooling and a prediction network for crystal property prediction.

# 14.134 Building the Complete Crystal Graph Convolutional Neural Network

A single CGCNN convolution layer updates atomic feature vectors using information from neighboring atoms.

Although this operation is powerful, it is not sufficient to predict material properties.

A complete neural network requires several additional components.

The overall CGCNN architecture consists of

1. Input layer
2. Multiple graph convolution layers
3. Graph pooling (readout)
4. Fully connected neural network
5. Output layer

Conceptually,

```text
Crystal Structure

↓

Crystal Graph

↓

Node Features

↓

CGCNN Layer 1

↓

CGCNN Layer 2

↓

CGCNN Layer 3

↓

Graph Pooling

↓

Crystal Representation

↓

Fully Connected Layers

↓

Property Prediction
```

Each component performs a distinct role.

Together they transform a crystal structure into a predicted material property.

---

# 14.135 Why Multiple Graph Convolution Layers?

Suppose we use only one graph convolution layer.

```
A —— B —— C
```

After one layer,

atom

```
B
```

learns information from

```
A

and

C.
```

However,

atom

```
A
```

still has no information about

```
C.
```

Now suppose we add another graph convolution layer.

```
Layer 1

A ←→ B ←→ C

↓

Layer 2

Information propagates further
```

During the second layer,

atom

```
A
```

receives information from

```
B,
```

which already contains information from

```
C.
```

Consequently,

atom

```
A
```

now indirectly learns about

```
C.
```

Each additional graph convolution layer expands the receptive field of every atom.

This enables the model to capture increasingly long-range structural information.

---

# 14.136 Choosing the Number of Graph Convolution Layers

The original CGCNN paper typically employed

```
3

to

6
```

graph convolution layers.

Using too few layers causes the receptive field to remain small.

Important long-range interactions may never be incorporated.

Conversely,

using too many layers introduces new challenges.

Very deep graph neural networks often suffer from

- oversmoothing,
- vanishing gradients,
- increased computational cost.

Oversmoothing occurs when repeated message passing causes neighboring node embeddings to become nearly identical.

Eventually,

the network loses its ability to distinguish different atoms.

For most materials science applications,

three to six graph convolution layers provide a good balance between expressive power and computational efficiency.

---

# 14.137 Implementing Multiple CGCNN Layers

PyTorch provides the

```python
ModuleList
```

container for storing multiple neural network layers.

The following implementation creates several graph convolution layers.

```python
self.convs = nn.ModuleList(

    [

        CGCNNConv(

            hidden_dim,

            edge_dim

        )

        for _ in range(

            num_conv_layers

        )

    ]

)
```

Suppose

```python
num_conv_layers = 4
```

Then the network automatically creates

```text
CGCNN Layer 1

↓

CGCNN Layer 2

↓

CGCNN Layer 3

↓

CGCNN Layer 4
```

Every layer has independent learnable parameters.

---

# 14.138 Passing the Graph Through the Layers

Once the layers have been created,

the graph is processed sequentially.

```python
for conv in self.convs:

    x = conv(

        x,

        edge_index,

        edge_attr

    )
```

The first layer updates the initial atomic descriptors.

The second layer refines those updated representations.

The process continues until every graph convolution layer has been applied.

After the final layer,

each atom possesses a learned embedding that summarizes information from its surrounding crystal environment.

---

# 14.139 The Need for Graph Pooling

After the final graph convolution layer,

the network still contains one feature vector **per atom**.

Suppose a crystal contains

```
48 atoms.
```

The output may have shape

```text
(48,128)
```

This means

- 48 atoms,
- 128 learned features per atom.

However,

our prediction target is usually a **crystal property**, such as

- formation energy,
- band gap,
- bulk modulus,
- elastic constant.

These properties describe the **entire crystal**, not individual atoms.

Therefore,

the atomic embeddings must be combined into a single crystal representation.

This operation is called **graph pooling** or **graph readout**.

---

# 14.140 Global Pooling

Graph pooling combines all atomic embeddings into one feature vector.

Conceptually,

```text
Atom 1

↓

Atom 2

↓

Atom 3

↓

...

↓

Atom N

↓

Pooling

↓

Crystal Embedding
```

The resulting vector summarizes the entire crystal.

It is this crystal embedding—not the individual atomic embeddings—that is passed to the final prediction network.

---

# 14.141 Pooling Methods

Several graph pooling strategies are commonly used.

### Global Sum Pooling

All atomic feature vectors are summed.

$$
\mathbf{h}
=
\sum_{i=1}^{N}
\mathbf{v}_i.
$$

This preserves the cumulative contribution of all atoms.

---

### Global Mean Pooling

The average atomic feature vector is computed.

$$
\mathbf{h}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{v}_i.
$$

Mean pooling reduces sensitivity to the number of atoms in the crystal.

---

### Global Max Pooling

Each feature dimension retains only its maximum value.

This emphasizes the strongest atomic responses.

---

The original CGCNN primarily uses **average pooling**, allowing crystals with different numbers of atoms to be compared more naturally.

---

# 14.142 Global Mean Pooling in PyTorch Geometric

PyTorch Geometric provides several ready-made pooling functions.

For CGCNN,

global mean pooling is implemented as

```python
from torch_geometric.nn import global_mean_pool
```

During the forward pass,

pooling is performed using

```python
crystal_embedding = global_mean_pool(

    x,

    batch

)
```

The argument

```python
x
```

contains the atomic embeddings produced by the graph convolution layers.

The

```python
batch
```

tensor specifies which atoms belong to which crystal.

The output contains

```
One Feature Vector

↓

Per Crystal
```

instead of one feature vector per atom.

---

# 14.143 Why the Batch Vector Is Needed

Recall that mini-batch training merges many crystal graphs into one disconnected graph.

Suppose we process three crystals simultaneously.

```text
Crystal A

↓

20 atoms
```

```text
Crystal B

↓

35 atoms
```

```text
Crystal C

↓

18 atoms
```

The combined graph contains

```
73 atoms.
```

During pooling,

the neural network must compute

- one embedding for Crystal A,
- one embedding for Crystal B,
- one embedding for Crystal C.

The `batch` tensor tells the pooling operation exactly which atoms belong to each crystal, ensuring that embeddings are computed independently for every graph in the mini-batch.

---

# 14.144 The Fully Connected Prediction Network

After pooling,

each crystal is represented by a single feature vector.

This vector is passed through a conventional feedforward neural network.

For example,

```python
self.mlp = nn.Sequential(

    nn.Linear(

        hidden_dim,

        hidden_dim

    ),

    nn.ReLU(),

    nn.Linear(

        hidden_dim,

        hidden_dim // 2

    ),

    nn.ReLU(),

    nn.Linear(

        hidden_dim // 2,

        1

    )

)
```

The multilayer perceptron transforms the learned crystal embedding into the desired prediction.

Depending on the task,

the final output may represent

- formation energy,
- band gap,
- elastic modulus,
- classification probability.

---

# 14.145 The Overall Forward Pass

The complete forward pass of the network now becomes

```text
Input Crystal Graph

↓

Node Features

↓

Graph Convolution Layers

↓

Updated Atomic Embeddings

↓

Global Mean Pooling

↓

Crystal Embedding

↓

Fully Connected Network

↓

Predicted Property
```

Although the architecture appears relatively simple,

each stage plays a crucial role.

The graph convolution layers learn local atomic environments,

the pooling layer summarizes the entire crystal,

and the fully connected network converts this learned representation into a physically meaningful prediction.

---

# 14.146 Toward a Complete CGCNN Model

We now possess every building block required to construct the original Crystal Graph Convolutional Neural Network.

Specifically, we have implemented

- crystal graph construction,
- node features,
- edge features,
- Gaussian distance expansion,
- message passing,
- graph convolution,
- residual updates,
- graph pooling,
- prediction layers.

In the next section, we will combine these components into a single `CGCNN` class implemented entirely in PyTorch Geometric, producing a complete end-to-end model that accepts crystal structures as input and predicts material properties through graph neural network learning.

# 14.147 Implementing the Complete CGCNN Model in PyTorch

We now have every component required to build the complete Crystal Graph Convolutional Neural Network.

Previously, we implemented only a single graph convolution layer.

Now we will assemble those layers into a complete end-to-end deep learning model capable of predicting material properties directly from crystal graphs.

The complete model consists of

- an input projection layer,
- multiple CGCNN convolution layers,
- graph pooling,
- fully connected prediction layers.

Conceptually,

```text
Crystal Graph

↓

Input Projection

↓

CGCNN Layer 1

↓

CGCNN Layer 2

↓

CGCNN Layer 3

↓

CGCNN Layer 4

↓

Global Mean Pooling

↓

Fully Connected Layers

↓

Prediction
```

Unlike the previous sections, which examined each component separately, we now integrate them into one neural network.

---

# 14.148 Why an Input Projection Layer?

Recall that our node features are handcrafted descriptors.

For example,

```
92 atomic descriptors
```

These descriptors are informative but may not match the hidden feature dimension used inside the neural network.

Suppose we want every hidden representation to contain

```
128 features.
```

The network therefore begins by projecting

```
92

↓

128
```

using a linear transformation.

This operation allows the graph convolution layers to operate within a consistent feature space.

---

The input projection layer is implemented as

```python
self.embedding = nn.Linear(

    node_dim,

    hidden_dim

)
```

Suppose

```python
node_dim = 92

hidden_dim = 128
```

Then every atom is transformed from

```text
92 Features

↓

128 Features
```

before graph convolution begins.

---

# 14.149 Constructing the Graph Convolution Stack

The hidden representations are refined using multiple CGCNN layers.

```python
self.convs = nn.ModuleList(

    [

        CGCNNConv(

            hidden_dim,

            edge_dim

        )

        for _ in range(

            num_conv_layers

        )

    ]

)
```

Each graph convolution layer has its own learnable parameters.

If

```python
num_conv_layers = 4
```

then information propagates through four successive rounds of message passing.

Each round expands the receptive field of every atom.

---

# 14.150 Adding Dropout for Regularization

Deep neural networks often memorize the training data instead of learning generalizable patterns.

This phenomenon is known as **overfitting**.

A common solution is **dropout**.

During training,

dropout randomly deactivates a fraction of neurons.

For example,

with

```python
dropout = 0.2
```

approximately

```
20%
```

of hidden neurons are temporarily ignored during each training iteration.

This encourages the network to learn more robust representations.

The dropout layer is

```python
self.dropout = nn.Dropout(

    p=0.2

)
```

Dropout is active only during training.

It is automatically disabled during model evaluation.

---

# 14.151 Constructing the Prediction Head

After graph pooling,

the crystal embedding is passed through a multilayer perceptron.

A simple prediction head is

```python
self.predictor = nn.Sequential(

    nn.Linear(

        hidden_dim,

        hidden_dim

    ),

    nn.ReLU(),

    nn.Dropout(

        0.2

    ),

    nn.Linear(

        hidden_dim,

        hidden_dim // 2

    ),

    nn.ReLU(),

    nn.Linear(

        hidden_dim // 2,

        output_dim

    )

)
```

For regression tasks,

```python
output_dim = 1
```

For classification,

the output dimension equals the number of classes.

---

# 14.152 The Complete Forward Pass

The forward function combines every component of the network.

```python
def forward(

    self,

    data

):

    x = data.x

    edge_index = data.edge_index

    edge_attr = data.edge_attr

    batch = data.batch
```

The graph information stored inside the `Data` object is extracted.

Next,

the node features are projected into the hidden space.

```python
x = self.embedding(

    x

)
```

At this point,

every atom has a hidden feature vector of dimension

```text
hidden_dim.
```

---

# 14.153 Applying Graph Convolution Layers

The hidden representations are updated repeatedly.

```python
for conv in self.convs:

    x = conv(

        x,

        edge_index,

        edge_attr

    )
```

Each iteration performs

- message construction,
- message aggregation,
- residual update.

After the final graph convolution,

every atom contains information from a substantial portion of the crystal.

---

# 14.154 Pooling Atomic Embeddings

The atomic embeddings must now be combined into one crystal representation.

```python
from torch_geometric.nn import global_mean_pool

x = global_mean_pool(

    x,

    batch

)
```

Suppose the mini-batch contains

```
32 crystals.
```

After pooling,

the output tensor has shape

```text
(32, hidden_dim)
```

Each row now corresponds to one crystal rather than one atom.

---

# 14.155 Predicting Material Properties

The pooled crystal embedding is finally passed through the prediction network.

```python
x = self.dropout(

    x

)

prediction = self.predictor(

    x

)

return prediction
```

The returned tensor contains the predicted material property.

For example,

```text
tensor([

[-2.54],

[-3.71],

[-1.89],

...

])
```

Each value corresponds to one crystal in the mini-batch.

---

# 14.156 Complete CGCNN Implementation

The following code assembles every component into one complete model.

```python
import torch
import torch.nn as nn

from torch_geometric.nn import (

    global_mean_pool

)


class CGCNN(nn.Module):

    def __init__(

        self,

        node_dim,

        edge_dim,

        hidden_dim=128,

        num_conv_layers=4,

        output_dim=1

    ):

        super().__init__()

        self.embedding = nn.Linear(

            node_dim,

            hidden_dim

        )

        self.convs = nn.ModuleList(

            [

                CGCNNConv(

                    hidden_dim,

                    edge_dim

                )

                for _ in range(

                    num_conv_layers

                )

            ]

        )

        self.dropout = nn.Dropout(

            0.2

        )

        self.predictor = nn.Sequential(

            nn.Linear(

                hidden_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Dropout(

                0.2

            ),

            nn.Linear(

                hidden_dim,

                hidden_dim // 2

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim // 2,

                output_dim

            )

        )

    def forward(

        self,

        data

    ):

        x = self.embedding(

            data.x

        )

        edge_index = data.edge_index

        edge_attr = data.edge_attr

        batch = data.batch

        for conv in self.convs:

            x = conv(

                x,

                edge_index,

                edge_attr

            )

        x = global_mean_pool(

            x,

            batch

        )

        x = self.dropout(

            x

        )

        return self.predictor(

            x

        )
```

Although concise,

this implementation faithfully reproduces the overall architecture of the original CGCNN.

Every major stage discussed throughout this chapter is now represented in executable code.

---

# 14.157 Understanding the Flow of Data

During one forward pass,

the data flows through the model as follows.

```text
Node Features

↓

Linear Embedding

↓

Graph Convolution 1

↓

Graph Convolution 2

↓

Graph Convolution 3

↓

Graph Convolution 4

↓

Global Mean Pooling

↓

Dropout

↓

Fully Connected Network

↓

Predicted Property
```

Each stage transforms the representation into a progressively more informative form.

Initially,

the model only knows basic atomic descriptors.

After multiple rounds of graph convolution,

it learns complex structural patterns that are useful for predicting material properties.

---

# 14.158 Model Summary

At this point,

our implementation contains every essential architectural component of the original Crystal Graph Convolutional Neural Network:

- Input embedding layer
- Multiple gated CGCNN convolution layers
- Residual message passing
- Gaussian-expanded edge features
- Global graph pooling
- Fully connected prediction head
- End-to-end differentiable architecture

However, a neural network is useful only after it has been trained.

The next stage of this chapter focuses on the complete training pipeline.

We will implement

- loss functions,
- optimizers,
- learning-rate scheduling,
- forward and backward propagation,
- parameter updates,
- validation,
- checkpointing,

and train the CGCNN model on a real materials dataset using PyTorch.

# 14.159 Training the Crystal Graph Convolutional Neural Network

Constructing a neural network architecture is only half of the machine learning process.

A newly initialized CGCNN contains millions of parameters whose values are random.

Consequently,

its predictions are essentially random as well.

The purpose of training is to gradually adjust these parameters so that the predicted material properties become increasingly close to the true experimental or computational values.

Training is an optimization problem.

The neural network repeatedly

- makes predictions,
- measures the prediction error,
- computes gradients,
- updates its parameters.

After thousands of iterations,

the model learns meaningful relationships between crystal structures and material properties.

---

# 14.160 Overview of the Training Pipeline

Training a CGCNN follows a well-defined sequence.

```text
Crystal Dataset

↓

DataLoader

↓

Mini-Batch

↓

Forward Pass

↓

Prediction

↓

Loss Calculation

↓

Backpropagation

↓

Gradient Computation

↓

Parameter Update

↓

Next Mini-Batch
```

This process is repeated for many epochs until the model converges.

Although the workflow appears simple,

each stage involves important mathematical and computational concepts.

---

# 14.161 Preparing the Dataset Splits

Before training,

the dataset must be divided into separate subsets.

A common strategy is

- Training Set
- Validation Set
- Test Set

For example,

```text
Total Dataset

↓

100,000 Crystals

↓

Training

80%

↓

Validation

10%

↓

Test

10%
```

The training set is used to optimize the neural network parameters.

The validation set is used to monitor performance during training and tune hyperparameters.

The test set is used only after training has finished to estimate the model's performance on completely unseen data.

Keeping these datasets separate is essential to obtain an unbiased evaluation.

---

# 14.162 Splitting the Dataset in PyTorch

Suppose

```python
dataset
```

contains every processed crystal graph.

The dataset can be divided using

```python
from torch.utils.data import random_split

dataset_size = len(dataset)

train_size = int(

    0.8 * dataset_size

)

validation_size = int(

    0.1 * dataset_size

)

test_size = (

    dataset_size

    - train_size

    - validation_size

)

train_dataset, validation_dataset, test_dataset = random_split(

    dataset,

    [

        train_size,

        validation_size,

        test_size

    ]

)
```

The three subsets now contain independent crystal graphs.

---

# 14.163 Creating DataLoaders

Each dataset is wrapped inside a DataLoader.

```python
from torch_geometric.loader import DataLoader

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True

)

validation_loader = DataLoader(

    validation_dataset,

    batch_size=32,

    shuffle=False

)

test_loader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False

)
```

Notice that only the training loader uses

```python
shuffle=True
```

Randomizing the order of training samples improves optimization.

Validation and test datasets remain fixed to ensure reproducible evaluation.

---

# 14.164 Creating the Model

The CGCNN model can now be instantiated.

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

model = CGCNN(

    node_dim=92,

    edge_dim=31,

    hidden_dim=128,

    num_conv_layers=4,

    output_dim=1

)

model = model.to(device)
```

The model is transferred to the GPU whenever one is available.

Training on a GPU is typically several orders of magnitude faster than training on a CPU.

---

# 14.165 Choosing the Loss Function

The loss function measures the difference between predictions and target values.

For regression tasks such as

- formation energy,
- band gap,
- elastic modulus,

the most common choice is **Mean Squared Error (MSE)**.

The loss function is

```python
criterion = nn.MSELoss()
```

Mathematically,

$$
\mathcal{L}
=
\frac{1}{N}
\sum_{i=1}^{N}

(y_i-\hat{y}_i)^2.
$$

Large prediction errors contribute disproportionately to the loss,

encouraging the network to reduce major mistakes.

For classification problems,

other loss functions such as Cross Entropy Loss are typically used instead.

---

# 14.166 Choosing the Optimizer

The optimizer determines how the model parameters are updated.

The original CGCNN implementation employs the Adam optimizer.

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3

)
```

Adam combines

- momentum,
- adaptive learning rates,
- efficient gradient updates.

These properties make it particularly well suited for training deep neural networks.

The learning rate

```python
1e-3
```

is a common starting point,

although it may require adjustment depending on the dataset and model.

---

# 14.167 The Forward Pass

During each training iteration,

a mini-batch of crystal graphs is loaded.

```python
for batch in train_loader:

    batch = batch.to(device)

    prediction = model(batch)
```

The model receives a complete mini-batch of graphs and returns one prediction for each crystal.

Suppose

```
Batch Size = 32
```

The output tensor has shape

```text
(32,1)
```

Each element corresponds to one predicted material property.

---

# 14.168 Computing the Loss

The predicted values are compared with the ground truth.

```python
loss = criterion(

    prediction,

    batch.y

)
```

Suppose the model predicts

```text
Formation Energy

↓

-2.81 eV
```

while the true value is

```text
-2.45 eV.
```

The loss function quantifies this discrepancy.

Smaller losses indicate better predictions.

The optimizer will attempt to minimize this value during training.

---

# 14.169 Clearing Previous Gradients

PyTorch accumulates gradients by default.

Therefore,

the gradients from the previous iteration must be cleared before computing new ones.

```python
optimizer.zero_grad()
```

This prevents gradients from multiple batches from being added together unintentionally.

Gradient accumulation is useful in certain advanced training strategies,

but standard supervised learning clears gradients at every iteration.

---

# 14.170 Backpropagation

Once the loss has been computed,

PyTorch automatically calculates gradients.

```python
loss.backward()
```

This single line performs the entire backpropagation algorithm.

Using the computational graph constructed during the forward pass,

PyTorch applies the chain rule to compute

$$
\frac{\partial \mathcal{L}}
{\partial \theta}
$$

for every learnable parameter

$$
\theta
$$

in the network.

These gradients indicate how each parameter should change to reduce the loss.

---

# 14.171 Updating the Parameters

After the gradients have been computed,

the optimizer updates the parameters.

```python
optimizer.step()
```

Conceptually,

every parameter is adjusted according to

```text
New Parameter

=

Old Parameter

−

Learning Rate

×

Gradient
```

Although Adam uses a more sophisticated update rule internally,

the underlying objective remains the same:

reduce the loss by moving the parameters toward better values.

---

# 14.172 One Complete Training Iteration

Combining every step yields one complete optimization iteration.

```python
batch = batch.to(device)

optimizer.zero_grad()

prediction = model(batch)

loss = criterion(

    prediction,

    batch.y

)

loss.backward()

optimizer.step()
```

This sequence is repeated once for every mini-batch in the training dataset.

After all mini-batches have been processed,

one **training epoch** has been completed.

---

# 14.173 The Complete Training Loop

Training continues for many epochs.

A complete training loop is

```python
num_epochs = 100

for epoch in range(

    num_epochs

):

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        prediction = model(batch)

        loss = criterion(

            prediction,

            batch.y

        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (

        total_loss

        / len(train_loader)

    )

    print(

        f"Epoch {epoch+1}: "

        f"{average_loss:.6f}"

    )
```

The printed loss should gradually decrease as training progresses.

A steadily decreasing training loss indicates that the network is successfully learning meaningful relationships from the crystal graphs.

---

# 14.174 What Happens During Training?

Although the code appears simple,

an enormous amount of computation occurs internally.

For every mini-batch,

the network

- performs graph convolutions,
- exchanges messages between neighboring atoms,
- computes crystal embeddings,
- predicts material properties,
- evaluates prediction error,
- computes gradients,
- updates millions of learnable parameters.

This entire process repeats thousands of times.

Over successive epochs,

the initially random atomic embeddings evolve into highly informative representations that capture chemical bonding, crystal symmetry, and local atomic environments.

However,

training loss alone is not sufficient to judge model quality.

A model may achieve extremely low training loss simply by memorizing the training data.

To determine whether the model has truly learned generalizable patterns,

its performance must be evaluated on unseen validation data.

The next section introduces validation, evaluation metrics, early stopping, and learning-rate scheduling, which are essential components of a robust CGCNN training pipeline.

# 14.175 Model Validation and Performance Evaluation

A decreasing training loss does **not** necessarily indicate that a model has learned meaningful physical relationships.

A neural network with sufficient capacity can simply memorize the training data.

Such a model performs extremely well on previously seen crystals but fails when presented with new materials.

This phenomenon is known as **overfitting**.

The purpose of model validation is to measure how well the trained CGCNN generalizes to unseen crystal structures.

Rather than evaluating the model on the training data,

we evaluate it using an independent validation dataset that was never used during optimization.

Only if the model performs well on both training and validation data can we conclude that it has learned meaningful structure-property relationships.

---

# 14.176 The Difference Between Training, Validation, and Testing

The three dataset splits serve different purposes.

```text
Entire Dataset

↓

Training Set

↓

Learn Parameters

────────────────────

Validation Set

↓

Tune Model

────────────────────

Test Set

↓

Final Evaluation
```

The **training set** is used to optimize the neural network weights.

The **validation set** is used during training to monitor generalization performance and guide decisions such as

- selecting the learning rate,
- choosing the number of graph convolution layers,
- determining when to stop training.

The **test set** remains untouched until the very end.

It provides an unbiased estimate of the final model performance.

Using the test set repeatedly during development can unintentionally leak information into the model selection process, leading to overly optimistic performance estimates.

---

# 14.177 Evaluation Mode

Neural networks behave differently during training and evaluation.

Certain layers,

such as

- Dropout,
- Batch Normalization,

change their behavior depending on whether the model is training.

Before evaluating the validation set,

the model must be switched into evaluation mode.

```python
model.eval()
```

This command

- disables dropout,
- uses stored Batch Normalization statistics,
- ensures deterministic predictions.

Failing to call

```python
model.eval()
```

may produce inconsistent validation results.

---

# 14.178 Disabling Gradient Computation

During validation,

no parameters are updated.

Therefore,

gradient computation is unnecessary.

PyTorch provides a special context manager for this purpose.

```python
with torch.no_grad():

    prediction = model(batch)
```

Disabling gradient computation provides several advantages.

- Lower GPU memory usage
- Faster inference
- Reduced computational overhead

This is the standard approach for validation and testing.

---

# 14.179 Validation Loop

The complete validation loop closely resembles the training loop,

except that

- gradients are disabled,
- parameters are not updated.

```python
model.eval()

validation_loss = 0.0

with torch.no_grad():

    for batch in validation_loader:

        batch = batch.to(device)

        prediction = model(batch)

        loss = criterion(

            prediction,

            batch.y

        )

        validation_loss += loss.item()

validation_loss /= len(validation_loader)

print(

    validation_loss

)
```

This loop computes the average validation loss after each training epoch.

Monitoring this quantity provides valuable insight into the model's generalization ability.

---

# 14.180 Interpreting Training and Validation Loss

During successful training,

both losses should decrease.

Conceptually,

```text
Epoch

↓

Training Loss

↓

Validation Loss
```

Initially,

both values decrease rapidly.

Eventually,

the training loss may continue decreasing while the validation loss stops improving.

This behavior is illustrated below.

```text
Epoch

↓

Training Loss

██████████

███████

████

██

█

Validation Loss

██████████

██████

███

██

██

███
```

Notice that after a certain point,

the validation loss begins increasing even though the training loss continues decreasing.

This indicates overfitting.

The network is memorizing the training data rather than learning generalizable physical relationships.

---

# 14.181 Early Stopping

A common strategy for preventing overfitting is **early stopping**.

Instead of training for a predetermined number of epochs,

training terminates automatically once the validation loss stops improving.

Conceptually,

```text
Validation Loss

↓

Improves

↓

Improves

↓

Improves

↓

Stops Improving

↓

Training Stops
```

Early stopping prevents unnecessary optimization that would otherwise reduce generalization performance.

---

# 14.182 Implementing Early Stopping

A simple implementation is shown below.

```python
best_validation_loss = float("inf")

patience = 10

counter = 0

for epoch in range(num_epochs):

    train()

    validation_loss = validate()

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        counter = 0

        torch.save(

            model.state_dict(),

            "best_model.pt"

        )

    else:

        counter += 1

    if counter >= patience:

        print(

            "Early stopping."

        )

        break
```

Whenever validation performance improves,

the model is saved.

If no improvement occurs for

```
10 consecutive epochs,
```

training terminates automatically.

This approach is widely used in modern deep learning.

---

# 14.183 Learning Rate Scheduling

The learning rate determines how large each optimization step is.

A learning rate that is too large may prevent convergence.

A learning rate that is too small may cause extremely slow training.

Instead of keeping the learning rate fixed,

many researchers gradually reduce it during training.

PyTorch provides several schedulers for this purpose.

One commonly used scheduler is

```python
ReduceLROnPlateau
```

which reduces the learning rate whenever validation performance stops improving.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=5

)
```

Here,

the learning rate is reduced by a factor of

```
0.5
```

after

```
5
```

epochs without validation improvement.

---

# 14.184 Updating the Scheduler

After each validation epoch,

the scheduler receives the latest validation loss.

```python
scheduler.step(

    validation_loss

)
```

Suppose the initial learning rate is

```text
0.001
```

After several unsuccessful epochs,

it becomes

```text
0.0005
```

Later,

it may decrease further.

```text
0.00025
```

Smaller learning rates allow the optimizer to perform finer adjustments near the optimum.

---

# 14.185 Evaluation Metrics for Regression

Loss functions guide optimization,

but they are not always the most interpretable way to evaluate a regression model.

Several standard metrics are commonly reported in materials informatics.

### Mean Absolute Error (MAE)

$$
\text{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|.
$$

MAE measures the average prediction error in the original physical units.

For example,

```
MAE = 0.08 eV/atom
```

means that,

on average,

the predicted formation energies differ from the true values by

```
0.08 eV/atom.
```

---

### Root Mean Squared Error (RMSE)

$$
\text{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}.
$$

RMSE penalizes large prediction errors more strongly than MAE.

It is therefore more sensitive to outliers.

---

### Coefficient of Determination (\(R^2\))

$$
R^2
=
1
-
\frac
{\sum (y-\hat y)^2}
{\sum (y-\bar y)^2}.
$$

The \(R^2\) score measures how much of the variance in the target property is explained by the model.

- \(R^2 = 1\): perfect prediction.
- \(R^2 = 0\): no improvement over predicting the mean.
- \(R^2 < 0\): worse than predicting the mean.

For materials property prediction,

MAE and RMSE are often the most informative metrics because they retain physical units.

---

# 14.186 Computing Evaluation Metrics

The metrics can be computed using scikit-learn.

```python
from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)

mae = mean_absolute_error(

    targets,

    predictions

)

rmse = mean_squared_error(

    targets,

    predictions,

    squared=False

)

r2 = r2_score(

    targets,

    predictions

)

print(

    "MAE:",

    mae

)

print(

    "RMSE:",

    rmse

)

print(

    "R²:",

    r2

)
```

These metrics should always be computed on the validation or test set,

not on the training set.

---

# 14.187 Saving the Best Model

The model that achieves the lowest validation loss should be preserved.

Saving only the final epoch is not always desirable,

because the best-performing model may have occurred much earlier.

PyTorch saves the model parameters using

```python
torch.save(

    model.state_dict(),

    "best_model.pt"

)
```

The model can later be restored using

```python
model.load_state_dict(

    torch.load(

        "best_model.pt"

    )

)
```

This ensures that subsequent predictions use the best-performing network rather than simply the last network trained.

---

# 14.188 A Complete Training Strategy

Combining the ideas introduced in this section,

a robust CGCNN training workflow becomes

```text
Initialize Model

↓

Training Epoch

↓

Validation

↓

Compute Metrics

↓

Save Best Model

↓

Adjust Learning Rate

↓

Early Stopping Check

↓

Next Epoch
```

This training strategy is widely adopted in state-of-the-art graph neural network research because it balances optimization efficiency with strong generalization performance.

At this point,

we have built and trained a complete Crystal Graph Convolutional Neural Network.

The final part of this chapter focuses on applying the trained model to real materials science problems, including making predictions for unseen crystals, interpreting learned representations, and discussing the strengths, limitations, and practical considerations of CGCNN in modern materials informatics research.

# 14.189 Making Predictions with a Trained CGCNN Model

After the Crystal Graph Convolutional Neural Network has been successfully trained, it can be used to predict the properties of **previously unseen materials**.

This stage is called **inference**.

Unlike training,

during inference

- the model parameters remain fixed,
- no gradients are computed,
- no optimization occurs.

The model simply receives a crystal graph and produces a prediction.

Conceptually,

```text
New Crystal

↓

Graph Construction

↓

Trained CGCNN

↓

Predicted Property
```

Inference is ultimately the purpose of the entire training process.

A well-trained CGCNN should generalize to new crystal structures that were never encountered during training.

---

# 14.190 Loading the Trained Model

Before making predictions,

the trained parameters must be restored.

Suppose the best model was saved during training as

```
best_model.pt
```

The model is reconstructed using exactly the same architecture.

```python
model = CGCNN(

    node_dim=92,

    edge_dim=31,

    hidden_dim=128,

    num_conv_layers=4,

    output_dim=1

)
```

The learned parameters are then loaded.

```python
model.load_state_dict(

    torch.load(

        "best_model.pt",

        map_location=device

    )

)

model.to(device)

model.eval()
```

Calling

```python
model.eval()
```

is essential because it disables dropout and ensures deterministic predictions.

---

# 14.191 Preparing a New Crystal

Suppose a researcher has synthesized a new crystal and wishes to estimate its formation energy before performing expensive Density Functional Theory (DFT) calculations.

The crystal exists as a CIF file.

```text
NewMaterial.cif
```

The preprocessing pipeline is identical to the one used during training.

```text
Read CIF

↓

Pymatgen Structure

↓

Neighbor Search

↓

Node Features

↓

Edge Features

↓

Graph Construction

↓

Data Object
```

The trained network expects the same graph representation used during training.

Any inconsistency between training and inference preprocessing can significantly reduce prediction accuracy.

---

# 14.192 Constructing the Graph

Suppose the graph construction function developed earlier is

```python
build_graph()
```

The new graph is created as

```python
structure = Structure.from_file(

    "NewMaterial.cif"

)

graph = build_graph(

    structure

)
```

The resulting object is a standard PyTorch Geometric `Data` object.

However,

the model expects mini-batches rather than individual graphs.

Therefore,

the graph must be wrapped inside a DataLoader.

---

# 14.193 Creating a DataLoader for Inference

A single graph can be processed using

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(

    [graph],

    batch_size=1,

    shuffle=False

)
```

Although only one graph is present,

the DataLoader automatically creates the required

```python
batch
```

tensor.

This ensures that the forward pass is identical to training.

---

# 14.194 Making a Prediction

Prediction requires only a few lines of code.

```python
with torch.no_grad():

    for batch in loader:

        batch = batch.to(device)

        prediction = model(batch)
```

The output tensor contains one predicted value.

For example,

```text
tensor([[-2.81]])
```

The prediction can be converted into a Python scalar.

```python
predicted_value = prediction.item()

print(

    predicted_value

)
```

Possible output

```text
-2.81
```

If the target property is formation energy,

the predicted value is interpreted as

```text
Formation Energy

=

−2.81 eV/atom
```

---

# 14.195 Predicting Multiple Crystals

In practical materials informatics,

researchers rarely evaluate a single crystal.

Instead,

thousands of candidate materials may be screened.

Suppose a directory contains many CIF files.

```text
candidate_materials/

↓

Material1.cif

Material2.cif

Material3.cif

...

Material5000.cif
```

Every structure can be converted into a graph,

forming a dataset.

```python
graphs = []

for filename in cif_files:

    structure = Structure.from_file(

        filename

    )

    graph = build_graph(

        structure

    )

    graphs.append(

        graph

    )
```

The graphs are then processed in batches.

```python
loader = DataLoader(

    graphs,

    batch_size=32,

    shuffle=False

)
```

Batch inference is considerably faster than predicting one crystal at a time,

particularly when using a GPU.

---

# 14.196 Collecting Predictions

Predictions from every mini-batch can be accumulated into a list.

```python
predictions = []

with torch.no_grad():

    for batch in loader:

        batch = batch.to(device)

        output = model(batch)

        predictions.extend(

            output.cpu()

            .numpy()

            .flatten()

        )
```

The resulting list contains one predicted property for every crystal.

```text
Crystal 1

↓

−2.81
```

```text
Crystal 2

↓

−1.94
```

```text
Crystal 3

↓

−3.27
```

These predictions can then be exported for further analysis.

---

# 14.197 Saving Predictions

Predicted properties are often stored in a CSV file.

```python
import pandas as pd

results = pd.DataFrame(

    {

        "material":

        material_names,

        "prediction":

        predictions

    }

)

results.to_csv(

    "predictions.csv",

    index=False

)
```

The resulting file may appear as

| Material | Predicted Formation Energy (eV/atom) |
|-----------|--------------------------------------:|
| LiCoO₂ | -2.81 |
| NaCl | -3.62 |
| Si | -5.39 |
| MgO | -4.76 |

Such files can be directly incorporated into high-throughput materials screening pipelines.

---

# 14.198 High-Throughput Materials Screening

One of the greatest strengths of CGCNN is its ability to evaluate enormous numbers of candidate materials.

Traditional DFT calculations may require

```
Several Hours

↓

One Crystal
```

In contrast,

a trained CGCNN can often predict

```
Thousands of Crystals

↓

Minutes
```

This dramatic speedup enables researchers to rapidly identify promising materials before committing computational resources to expensive first-principles calculations.

A common workflow is

```text
Millions of Hypothetical Structures

↓

CGCNN Prediction

↓

Top 1%

↓

DFT Verification

↓

Experimental Validation
```

Rather than replacing DFT,

CGCNN acts as a fast screening tool that narrows the search space.

---

# 14.199 Understanding the Limitations of Predictions

Although CGCNN is a powerful predictive model,

its predictions should not be interpreted as exact physical truth.

Several factors influence prediction reliability.

First,

the model cannot reliably extrapolate far beyond the distribution of the training data.

For example,

a network trained exclusively on inorganic oxides may perform poorly on metal-organic frameworks or polymers.

Second,

prediction quality depends heavily on dataset quality.

Incorrect labels,

limited chemical diversity,

or systematic biases in the training data directly affect the learned model.

Finally,

CGCNN predicts correlations learned from data.

It does not replace first-principles calculations or experiments when definitive physical validation is required.

---

# 14.200 Practical Applications of CGCNN

Since its introduction,

CGCNN has been applied to a wide range of materials science problems.

Representative applications include

- formation energy prediction,
- band gap prediction,
- elastic modulus prediction,
- bulk modulus prediction,
- shear modulus prediction,
- thermal conductivity estimation,
- battery electrode discovery,
- catalyst screening,
- superconducting material exploration,
- thermoelectric material discovery,
- photovoltaic material screening.

Many of these applications rely on large DFT databases such as the Materials Project,

where CGCNN serves as a surrogate model for expensive quantum-mechanical calculations.

---

# 14.201 Completing the CGCNN Workflow

At this stage,

we have implemented the complete end-to-end workflow of the Crystal Graph Convolutional Neural Network.

```text
Crystal Structure

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

CGCNN Layers

↓

Graph Pooling

↓

Prediction Network

↓

Training

↓

Validation

↓

Inference

↓

Materials Discovery
```

Every stage has been implemented in code and connected into a unified machine learning pipeline.

You now possess the knowledge required to reproduce the original CGCNN architecture and apply it to real materials informatics research.

However,

CGCNN is only the beginning of graph neural networks for crystalline materials.

Subsequent architectures—including **MEGNet**, **SchNet**, **ALIGNN**, and **M3GNet**—introduce more sophisticated message-passing mechanisms, richer edge representations, and improved physical inductive biases.

The next chapter builds upon the foundations established here and explores these modern graph neural network architectures, highlighting how they extend the ideas introduced by CGCNN to achieve state-of-the-art performance in materials property prediction.

# 14.202 Computational Complexity and Scalability of CGCNN

Understanding how CGCNN learns is only one aspect of designing an effective graph neural network.

Another equally important consideration is **computational efficiency**.

Modern materials databases often contain

- hundreds of thousands of crystal structures,
- millions of atoms,
- tens of millions of graph edges.

Consequently,

a practical researcher must understand

- computational complexity,
- memory consumption,
- GPU utilization,
- scalability.

These considerations determine whether a model can be trained efficiently on real-world datasets.

---

# 14.203 Time Complexity of Graph Construction

Before training even begins,

every crystal must be converted into a graph.

The graph construction process consists of

```text
Read CIF

↓

Build Crystal Structure

↓

Neighbor Search

↓

Construct Edges

↓

Generate Node Features

↓

Generate Edge Features
```

The most computationally expensive step is usually the neighbor search.

Suppose

```
N
```

atoms exist inside the crystal.

A naive algorithm compares every atom with every other atom.

```text
Atom 1

↓

Compare with

↓

All Atoms

────────────────

Atom 2

↓

Compare with

↓

All Atoms

...

Atom N
```

The computational complexity becomes

$$
O(N^2).
$$

Fortunately,

Pymatgen employs efficient spatial search algorithms,

reducing the practical computational cost considerably.

Nevertheless,

neighbor searching remains a significant preprocessing expense for very large structures.

---

# 14.204 Complexity of Message Passing

Once the graph has been constructed,

the graph convolution layer operates primarily on graph edges.

Suppose

```
N

=

Number of Atoms
```

and

```
E

=

Number of Graph Edges.
```

Every graph convolution layer performs message computation for every edge.

Therefore,

the computational complexity is approximately

$$
O(E).
$$

Since crystalline materials usually have a bounded coordination number,

the number of edges grows approximately linearly with the number of atoms.

Consequently,

graph convolution scales much more favorably than algorithms with quadratic complexity.

---

# 14.205 Memory Requirements

The neural network must store several tensors simultaneously.

These include

- node features,
- edge indices,
- edge features,
- intermediate activations,
- gradients,
- optimizer states.

Suppose

```
100 atoms
```

are represented using

```
128-dimensional
```

hidden features.

The node feature matrix contains

```text
100 × 128
```

floating-point numbers.

Similarly,

edge features may require

```text
800 × 31
```

values,

depending on the number of neighbors.

During backpropagation,

additional memory is required to store gradients and intermediate activations.

Consequently,

GPU memory consumption increases approximately linearly with

- the number of atoms,
- the number of graph edges,
- the hidden feature dimension,
- the number of graph convolution layers.

---

# 14.206 Influence of Hidden Dimension

One of the most important architectural hyperparameters is the hidden feature dimension.

Suppose we compare three models.

| Hidden Dimension | Relative Memory | Relative Computation |
|-----------------:|----------------:|---------------------:|
| 64 | Low | Low |
| 128 | Moderate | Moderate |
| 256 | High | High |

Doubling the hidden dimension does not merely double computation.

Many internal matrix multiplications scale with the feature dimension,

making larger models substantially more computationally demanding.

Researchers therefore choose the hidden dimension by balancing

- predictive accuracy,
- computational efficiency.

---

# 14.207 Influence of Graph Depth

Adding more graph convolution layers also increases computational cost.

Suppose one graph convolution layer requires

```
1 unit
```

of computation.

Then

| Number of Layers | Relative Cost |
|-----------------:|--------------:|
| 2 | 2 |
| 4 | 4 |
| 8 | 8 |

Increasing graph depth enlarges the receptive field,

but also increases

- computation,
- memory consumption,
- training time.

Moreover,

very deep graph neural networks may suffer from oversmoothing,

reducing predictive performance.

Consequently,

modern CGCNN implementations typically employ only a moderate number of graph convolution layers.

---

# 14.208 Batch Size and GPU Utilization

The batch size determines how many crystal graphs are processed simultaneously.

Suppose we compare different batch sizes.

| Batch Size | GPU Utilization | Memory Usage |
|------------:|----------------:|-------------:|
| 8 | Low | Low |
| 32 | Moderate | Moderate |
| 128 | High | High |

Larger batches generally improve computational efficiency because modern GPUs are optimized for parallel processing.

However,

GPU memory eventually becomes the limiting factor.

The optimal batch size depends on

- graph size,
- hidden dimension,
- available GPU memory.

---

# 14.209 Training Time

The total training time depends upon several factors.

```text
Training Time

↓

Dataset Size

↓

Graph Size

↓

Number of Layers

↓

Hidden Dimension

↓

Batch Size

↓

GPU Performance
```

Suppose

```
Dataset

=

60,000 crystals
```

Using

```
Batch Size = 32
```

one training epoch requires

```text
60,000 / 32

≈

1,875 mini-batches.
```

If each mini-batch requires

```
0.05 seconds,
```

one epoch takes approximately

```
94 seconds.
```

Training for

```
100 epochs
```

would therefore require roughly

```
2.6 hours.
```

This example illustrates why efficient implementations are important for large datasets.

---

# 14.210 Advantages of GPU Acceleration

Graph neural networks involve many matrix operations.

Modern GPUs execute these operations far more efficiently than CPUs.

For example,

a CPU processes operations sequentially using relatively few cores.

A GPU executes thousands of mathematical operations simultaneously.

Conceptually,

```text
CPU

↓

Few Powerful Cores

↓

Sequential Computation
```

```text
GPU

↓

Thousands of Processing Cores

↓

Massively Parallel Computation
```

For large materials datasets,

GPU acceleration often reduces training time from several days to a few hours.

---

# 14.211 Scaling to Large Materials Databases

One of the strengths of CGCNN is its ability to scale to modern materials databases.

Examples include

- Materials Project,
- Open Quantum Materials Database (OQMD),
- AFLOW,
- JARVIS.

These repositories contain hundreds of thousands—or even millions—of crystal structures and associated properties.

By converting each structure into a graph,

CGCNN can learn from these large datasets and develop highly generalizable representations of crystalline materials.

As dataset size increases,

prediction accuracy often improves because the network encounters a wider variety of chemical environments.

---

# 14.212 Practical Strategies for Efficient Training

Researchers commonly employ several techniques to improve computational efficiency.

- **Precompute graph objects:** Construct graphs once and save them as `.pt` files to avoid repeated preprocessing.
- **Use mini-batch training:** Process multiple crystal graphs simultaneously to maximize GPU utilization.
- **Choose appropriate batch sizes:** Increase the batch size until GPU memory becomes a constraint.
- **Use mixed precision training:** Employ half-precision floating-point arithmetic when supported by modern GPUs to reduce memory usage and accelerate computation.
- **Monitor GPU utilization:** Tools such as `nvidia-smi` help identify memory bottlenecks and underutilized hardware.
- **Cache frequently used datasets:** Storing processed datasets on fast SSDs reduces input/output overhead during training.

These practices can substantially reduce training time without changing the underlying model architecture.

---

# 14.213 Computational Characteristics of CGCNN

The computational behavior of CGCNN can be summarized as follows.

| Aspect | Characteristics |
|---------|-----------------|
| Graph Construction | Moderate preprocessing cost due to neighbor search |
| Graph Convolution | Approximately linear in the number of graph edges |
| Memory Usage | Depends on graph size, hidden dimension, and network depth |
| GPU Efficiency | Excellent for mini-batch training |
| Scalability | Suitable for large materials databases |

Compared with conventional fully connected neural networks,

CGCNN requires more sophisticated preprocessing.

However,

its graph representation enables the model to capture structural information that conventional descriptor-based methods cannot.

This additional computational cost is often justified by the significant improvement in predictive accuracy.

---

# 14.214 Concluding Remarks on CGCNN

The Crystal Graph Convolutional Neural Network represented a major milestone in materials informatics.

For the first time,

it demonstrated that crystal structures could be learned directly as graphs,

eliminating the need for extensive manual feature engineering.

Throughout this chapter,

we have developed CGCNN from first principles.

Beginning with crystal graph construction,

we systematically explored

- node representations,
- edge representations,
- Gaussian distance expansion,
- message passing,
- graph convolution,
- gated update mechanisms,
- graph pooling,
- network architecture,
- PyTorch implementation,
- model training,
- validation,
- inference,
- computational efficiency.

You are now equipped to implement, train, evaluate, and deploy CGCNN models for a wide range of materials property prediction tasks.

More importantly,

you now possess the conceptual foundation required to understand the next generation of graph neural networks.

In the following chapter, we will study **MEGNet (Materials Graph Networks)**, which extends the ideas of CGCNN by incorporating graph-level state attributes, more expressive update functions, and a unified message-passing framework that achieves even higher predictive performance across diverse materials science applications.

# 14.215 Strengths, Limitations, and Future Directions of CGCNN

By this point, we have studied every major component of the Crystal Graph Convolutional Neural Network, from graph construction to training and inference.

Before moving to more advanced graph neural network architectures, it is important to critically evaluate CGCNN itself.

No machine learning model is universally optimal.

Every model is built upon a set of assumptions that determine

- where it performs exceptionally well,
- where it struggles,
- how it can be improved.

Understanding these strengths and limitations is essential for conducting high-quality materials informatics research.

Rather than treating CGCNN as a "black box," researchers should understand **why** it succeeds, **where** it fails, and **what motivated the development of later architectures such as MEGNet, ALIGNN, SchNet, and M3GNet.**

---

# 14.216 Major Strengths of CGCNN

CGCNN introduced several groundbreaking ideas that fundamentally changed machine learning for crystalline materials.

### 1. Direct Learning from Crystal Structures

Perhaps the greatest innovation of CGCNN is that it learns directly from the crystal graph.

Traditional machine learning required researchers to manually compute descriptors such as

- atomic radius,
- electronegativity,
- packing fraction,
- density,
- handcrafted structural descriptors.

The workflow was

```text
Crystal Structure

↓

Feature Engineering

↓

Machine Learning
```

The quality of the final model depended heavily on the quality of the handcrafted features.

CGCNN changed this paradigm.

Instead,

the workflow became

```text
Crystal Structure

↓

Crystal Graph

↓

Graph Neural Network

↓

Learned Representation

↓

Prediction
```

The network automatically discovers useful structural representations during training.

This significantly reduces the need for manual feature engineering.

---

### 2. Natural Representation of Crystals

Crystals are naturally composed of

- atoms,
- chemical bonds,
- local atomic environments.

Graphs represent exactly these relationships.

Unlike fixed-length feature vectors,

graphs preserve

- connectivity,
- local coordination,
- crystal topology.

Consequently,

CGCNN models crystalline materials more naturally than conventional machine learning algorithms.

---

### 3. Parameter Sharing

The same graph convolution layer is applied to every atom and every edge.

Suppose the network learns how lithium interacts with oxygen.

The same learned parameters can immediately be applied to another crystal containing lithium and oxygen.

This parameter sharing enables CGCNN to generalize across materials with

- different crystal structures,
- different unit-cell sizes,
- different numbers of atoms.

---

### 4. Variable-Size Crystal Structures

Conventional neural networks require fixed-size inputs.

Crystal structures vary enormously.

Some structures contain

```
8 atoms.
```

Others contain

```
300 atoms.
```

CGCNN naturally accommodates graphs of different sizes through

- message passing,
- graph pooling.

No padding or truncation is required.

---

### 5. High Predictive Accuracy

Large benchmark studies have demonstrated that CGCNN substantially outperforms many descriptor-based machine learning models for predicting

- formation energy,
- band gap,
- elastic properties,
- magnetic properties,
- thermodynamic quantities.

Although newer architectures have surpassed CGCNN in certain benchmarks,

it remains a highly competitive baseline.

---

# 14.217 Limitations of CGCNN

Despite its success,

CGCNN also has important limitations.

Understanding these weaknesses explains why newer graph neural network architectures were developed.

---

### 1. Limited Edge Representation

In the original CGCNN,

edge information primarily consists of

- interatomic distance,
- Gaussian distance expansion.

While distance is an important descriptor,

real chemical interactions also depend on

- bond angles,
- local geometry,
- orbital interactions,
- higher-order structural relationships.

These effects are not explicitly represented.

Consequently,

CGCNN may fail to capture complex directional bonding.

---

### 2. No Explicit Angular Information

Consider two neighboring atoms.

```
A

↓

B

↓

C
```

The bond angle

```
∠ABC
```

is often crucial in determining

- covalent bonding,
- crystal stability,
- electronic structure.

The original CGCNN ignores bond angles.

Only pairwise distances are considered.

Later architectures,

such as ALIGNN,

explicitly incorporate angular information through line graphs.

---

### 3. Fixed Neighbor Cutoff

Graph construction depends on a predefined cutoff radius.

For example,

```
8 Å
```

Atoms beyond this cutoff are ignored.

Choosing the cutoff is challenging.

If it is too small,

important interactions may be omitted.

If it is too large,

the graph becomes unnecessarily dense,

increasing computational cost.

This introduces a hyperparameter that influences model performance.

---

### 4. Limited Long-Range Interactions

Each graph convolution layer exchanges information only between neighboring atoms.

Long-range interactions require stacking multiple graph convolution layers.

Very deep graph neural networks,

however,

may suffer from

- oversmoothing,
- vanishing gradients,
- increased computational cost.

Consequently,

capturing long-range physics remains challenging.

---

### 5. Dependence on Training Data

Like all supervised learning algorithms,

CGCNN can only learn patterns present in its training dataset.

If certain chemistries,

crystal structures,

or property ranges are absent,

the model may struggle to make reliable predictions for those materials.

Extrapolation beyond the training distribution remains one of the central challenges in machine learning.

---

# 14.218 Comparison with Traditional Machine Learning

The following table summarizes the differences between descriptor-based machine learning and CGCNN.

| Traditional Machine Learning | CGCNN |
|------------------------------|--------|
| Requires handcrafted descriptors | Learns representations automatically |
| Fixed-length feature vectors | Variable-size crystal graphs |
| Manual feature engineering | End-to-end representation learning |
| Limited structural information | Explicit crystal connectivity |
| Simpler models | Deep graph neural networks |

The transition from handcrafted descriptors to learned graph representations represents one of the most significant advances in modern materials informatics.

---

# 14.219 Why New Graph Neural Networks Were Developed

Although CGCNN represented a major breakthrough,

researchers soon recognized several opportunities for improvement.

This led to the development of newer architectures.

| Model | Major Improvement |
|--------|-------------------|
| SchNet | Continuous-filter convolutions for atomistic systems |
| MEGNet | Graph-level state features and more expressive update functions |
| ALIGNN | Explicit bond-angle information through line graphs |
| M3GNet | Three-body interactions and universal interatomic potentials |

Each of these models builds upon the same fundamental message-passing framework introduced by CGCNN while addressing one or more of its limitations.

For example,

ALIGNN improves directional bonding,

while M3GNet captures higher-order interactions and enables accurate prediction of energies, forces, and stresses.

---

# 14.220 When Should You Use CGCNN?

Despite the availability of newer models,

CGCNN remains an excellent choice in many situations.

It is particularly suitable when

- learning the fundamentals of graph neural networks,
- establishing baseline performance,
- working with moderate-sized datasets,
- predicting scalar material properties,
- developing new graph-based workflows.

Its relatively simple architecture also makes it easier to modify and extend for research purposes.

Many new graph neural network architectures continue to use concepts originally introduced by CGCNN.

---

# 14.221 Key Lessons from This Chapter

Throughout this chapter, we have transformed the original CGCNN paper into a complete, implementation-oriented understanding.

We have learned

- why crystal structures can be represented as graphs,
- how to construct crystal graphs from CIF files,
- how node and edge features are generated,
- why Gaussian distance expansion is used,
- how message passing operates,
- how gated graph convolution updates atomic embeddings,
- how graph pooling creates crystal-level representations,
- how to implement CGCNN using PyTorch Geometric,
- how to train, validate, and evaluate the model,
- how to deploy it for inference on unseen materials,
- how to analyze its computational complexity,
- what its strengths and limitations are.

At this stage, you should be capable of reading the original CGCNN paper with confidence, implementing the model from scratch, reproducing published experiments, and adapting the architecture for your own research projects.

---

# 14.222 Looking Ahead

CGCNN established the foundation for graph neural networks in materials science.

However,

it is no longer the endpoint of the field.

Modern materials informatics increasingly relies on architectures that incorporate

- richer geometric information,
- more expressive message-passing schemes,
- physically informed inductive biases,
- multi-task learning,
- force and stress prediction,
- large-scale pretraining.

These developments have dramatically expanded the capabilities of graph neural networks beyond scalar property prediction.

In the next chapter, we will build directly upon the concepts introduced here by studying **MEGNet (Materials Graph Networks)**.

Rather than treating MEGNet as an entirely new model, we will analyze it as a natural evolution of CGCNN, highlighting exactly what changes, why those changes improve performance, and how to implement the complete architecture in PyTorch Geometric.


# 14.223 Complete End-to-End CGCNN Research Project

Throughout this chapter, we have studied every individual component of the Crystal Graph Convolutional Neural Network.

However, research is rarely performed by implementing isolated components.

Instead, researchers develop an **entire machine learning pipeline** that begins with raw crystal structures and ends with scientific conclusions.

In this section, we will assemble every concept introduced in this chapter into a complete research workflow.

The objective is to reproduce a realistic materials informatics project that predicts the formation energy of crystalline materials.

Rather than treating the individual pieces independently, we will connect them into one unified system.

The workflow we will implement is

```text
Materials Project Dataset

↓

Download Crystal Structures

↓

Download Target Properties

↓

Preprocess Structures

↓

Construct Crystal Graphs

↓

Generate Node Features

↓

Generate Edge Features

↓

Create PyTorch Dataset

↓

Split Dataset

↓

Train CGCNN

↓

Validate Model

↓

Evaluate Test Performance

↓

Predict New Materials

↓

Screen Candidate Structures

↓

Scientific Analysis
```

This pipeline closely resembles the workflow used in many published materials informatics studies.

---

# 14.224 Project Directory Structure

A well-organized project is essential for reproducibility.

As machine learning projects grow, storing every function inside a single Python file quickly becomes difficult to maintain.

Instead, researchers separate different tasks into dedicated modules.

A recommended project structure is

```text
CGCNN_Project/

│

├── data/

│   ├── raw/

│   ├── processed/

│   └── splits/

│

├── models/

│   ├── cgcnn.py

│   └── layers.py

│

├── datasets/

│   └── crystal_dataset.py

│

├── preprocessing/

│   ├── graph_builder.py

│   ├── node_features.py

│   └── edge_features.py

│

├── training/

│   ├── train.py

│   ├── validate.py

│   └── test.py

│

├── inference/

│   └── predict.py

│

├── utils/

│   ├── metrics.py

│   ├── visualization.py

│   └── checkpoint.py

│

├── config.py

│

├── requirements.txt

│

└── README.md
```

Each folder has a specific responsibility.

Separating functionality in this way improves readability, debugging, and collaboration.

Large research projects almost always follow a similar modular design.

---

# 14.225 Responsibilities of Each Module

Each directory contributes to one stage of the machine learning pipeline.

### data/

Stores

- downloaded CIF files,
- processed graph objects,
- train-validation-test splits.

---

### preprocessing/

Contains all routines responsible for converting crystal structures into graph representations.

Typical tasks include

- reading CIF files,
- neighbor searching,
- Gaussian distance expansion,
- graph construction.

---

### datasets/

Defines custom PyTorch datasets.

This module determines how graph objects are loaded during training.

---

### models/

Contains

- graph convolution layers,
- complete CGCNN architecture,
- auxiliary neural network modules.

Separating model definitions from training code improves maintainability.

---

### training/

Implements

- training loops,
- validation,
- testing,
- checkpointing.

No model architecture should be defined here.

Instead,

the training code simply imports the model.

---

### inference/

Contains scripts used after training.

Typical tasks include

- loading trained weights,
- predicting properties,
- screening candidate materials.

---

### utils/

Stores reusable helper functions such as

- evaluation metrics,
- plotting utilities,
- checkpoint management,
- random seed initialization.

---

### config.py

Contains all hyperparameters in one location.

Instead of modifying values throughout the project,

every important parameter can be edited here.

For example,

```python
HIDDEN_DIM = 128

NUM_LAYERS = 4

LEARNING_RATE = 1e-3

BATCH_SIZE = 32

CUTOFF_RADIUS = 8.0
```

Centralizing configuration improves reproducibility and simplifies experimentation.

---

# 14.226 Installing Required Libraries

A research project should specify every required dependency.

A typical

```
requirements.txt
```

file may contain

```text
torch

torch-geometric

numpy

scipy

pandas

matplotlib

scikit-learn

pymatgen

matminer

tqdm

networkx
```

Using a requirements file ensures that other researchers can reproduce the computational environment with a single installation command.

---

# 14.227 Reproducibility

Scientific machine learning requires reproducible experiments.

Every run should produce identical results whenever possible.

PyTorch allows random seeds to be fixed.

```python
import random

import numpy as np

import torch

random.seed(42)

np.random.seed(42)

torch.manual_seed(42)

torch.cuda.manual_seed_all(42)
```

Setting the random seed reduces variation caused by random initialization and dataset shuffling.

Although complete determinism is not always possible on every GPU,

fixing random seeds greatly improves reproducibility.

---

# 14.228 Logging Experiments

During research,

hundreds of experiments may be performed.

Researchers therefore record

- learning rate,
- hidden dimension,
- validation MAE,
- training time,
- dataset version,
- random seed.

A simple experiment log may appear as

| Experiment | Hidden Dimension | Layers | MAE |
|------------|-----------------:|-------:|----:|
| Exp-01 | 64 | 3 | 0.067 |
| Exp-02 | 128 | 4 | 0.054 |
| Exp-03 | 256 | 6 | 0.051 |

Maintaining detailed experiment logs is essential for comparing model configurations and reproducing published results.

---

# 14.229 The Complete Research Pipeline

At this point, every component developed throughout the chapter can be viewed as part of a single integrated workflow.

```text
Download Dataset

↓

Read Crystal Structures

↓

Construct Crystal Graphs

↓

Generate Features

↓

Build Dataset

↓

Train CGCNN

↓

Validate Performance

↓

Save Best Model

↓

Evaluate Test Set

↓

Predict New Materials

↓

Analyze Results

↓

Publish Research
```

This pipeline represents the practical workflow followed in many contemporary materials informatics studies.

By integrating graph construction, graph neural networks, optimization, evaluation, and inference into a single reproducible framework, researchers can efficiently accelerate the discovery of new materials while reducing dependence on computationally expensive first-principles calculations.

# 14.230 Reproducing the Original CGCNN Paper

One of the defining characteristics of good scientific research is **reproducibility**.

A machine learning model is not truly understood until its published results can be independently reproduced.

In this section, we examine how the original CGCNN study can be reproduced using modern tools such as **PyTorch Geometric** and **Pymatgen**.

Rather than simply implementing the architecture, we now focus on reproducing the scientific methodology presented in the original paper.

This exercise teaches several important research skills.

- Reading scientific papers critically.
- Translating mathematical equations into code.
- Reproducing published experiments.
- Comparing reproduced results with reported benchmarks.
- Identifying sources of experimental variation.

Learning how to reproduce published work is one of the most valuable skills for graduate students and researchers.

---

# 14.231 Overview of the Original CGCNN Study

The original CGCNN paper addressed a fundamental problem in materials science.

> Can a neural network predict material properties directly from crystal structures without manually engineered descriptors?

To answer this question, the authors constructed crystal graphs from experimentally and computationally determined crystal structures and trained graph neural networks to predict several material properties.

The study demonstrated that graph neural networks could automatically learn chemically meaningful representations and outperform many conventional machine learning methods.

This work established graph neural networks as one of the dominant approaches in modern materials informatics.

---

# 14.232 Target Properties Predicted

The original study investigated several different regression tasks.

Representative target properties included

- Formation energy
- Band gap
- Fermi energy
- Bulk modulus
- Shear modulus
- Elastic properties

Each target property represents a separate supervised learning problem.

For example,

```text
Crystal Structure

↓

CGCNN

↓

Formation Energy
```

or

```text
Crystal Structure

↓

CGCNN

↓

Band Gap
```

The same neural network architecture can be trained for different properties simply by changing the target labels.

---

# 14.233 Typical Dataset Workflow

A simplified reproduction workflow is

```text
Crystal Database

↓

Download CIF Files

↓

Download Material Properties

↓

Construct Crystal Graphs

↓

Split Dataset

↓

Train CGCNN

↓

Evaluate Predictions

↓

Compare with Published Results
```

Notice that the neural network never receives handcrafted descriptors.

The only inputs are

- crystal structures,
- target property values.

Everything else is learned automatically.

---

# 14.234 Hyperparameters Used in Practice

Although exact values may vary depending on the implementation and dataset version, a typical CGCNN training configuration resembles the following.

| Hyperparameter | Typical Value |
|---------------|--------------:|
| Hidden Dimension | 64–128 |
| Graph Convolution Layers | 3–6 |
| Batch Size | 32–256 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Epochs | 100–300 |
| Dropout | 0.0–0.2 |

These values provide a strong baseline for reproducing many published CGCNN experiments.

In practice, researchers often perform hyperparameter optimization to determine the best configuration for a specific dataset.

---

# 14.235 Matching the Original Training Procedure

To reproduce published results faithfully, several aspects of the training pipeline must match the original methodology.

These include

- graph construction procedure,
- neighbor cutoff radius,
- node feature definitions,
- Gaussian distance expansion,
- optimizer,
- learning rate,
- batch size,
- data split strategy,
- evaluation metric.

Even seemingly minor differences can produce measurable changes in prediction accuracy.

For this reason, reproducibility requires careful attention to every stage of the computational workflow.

---

# 14.236 Sources of Variation

Suppose two researchers independently implement CGCNN.

Even if both implementations are correct, their reported MAE values may differ slightly.

Several factors contribute to this variation.

### Random Initialization

Neural network weights are initialized randomly.

Different initial weights lead to slightly different optimization trajectories.

---

### Dataset Splits

Changing the train-validation-test split changes the samples seen during training.

Some splits are naturally more difficult than others.

---

### Random Shuffling

Mini-batches are randomly shuffled every epoch.

Different mini-batch orders influence optimization.

---

### Floating-Point Arithmetic

Different hardware

- CPUs,
- NVIDIA GPUs,
- AMD GPUs,

may produce tiny numerical differences because floating-point arithmetic is not perfectly identical across architectures.

Although these differences are usually negligible,

they may accumulate over many optimization steps.

---

# 14.237 Evaluating Reproduction Quality

Successful reproduction does **not** require obtaining exactly the same numerical values reported in the original paper.

Instead,

researchers generally expect

- similar prediction accuracy,
- similar convergence behavior,
- similar learning curves,
- similar qualitative conclusions.

For example,

suppose the published MAE is

```
0.039 eV/atom.
```

A reproduced result of

```
0.041 eV/atom
```

would generally be considered an excellent reproduction.

Scientific reproducibility focuses on consistency of conclusions rather than identical floating-point values.

---

# 14.238 Comparing Predictions

A useful way to evaluate the trained model is to compare predicted values with ground-truth values.

Ideally,

all points should lie near the diagonal line.

```text
Predicted

↑

│                         •

│                    •

│               •

│          •

│     •

│•

└──────────────────────────→

Actual
```

Large deviations from the diagonal indicate prediction errors.

Such plots provide an intuitive visualization of model accuracy.

In practice,

researchers often accompany these plots with quantitative metrics such as MAE, RMSE, and \(R^2\).

---

# 14.239 Analyzing Prediction Errors

After reproducing the model,

the next step is to analyze where errors occur.

Questions commonly investigated include

- Which crystal systems produce the largest errors?
- Do certain chemical elements contribute disproportionately to prediction uncertainty?
- Does prediction accuracy decrease for very large unit cells?
- Are metallic materials more difficult to predict than semiconductors?
- Which structural motifs are consistently mispredicted?

Answering these questions provides insight into the strengths and limitations of the learned representation.

Error analysis is often as scientifically valuable as the prediction accuracy itself.

---

# 14.240 Extending the Original Model

Once the original CGCNN has been successfully reproduced,

it becomes a foundation for further research.

Possible extensions include

- increasing graph depth,
- introducing attention mechanisms,
- incorporating bond-angle information,
- learning more expressive edge representations,
- predicting multiple material properties simultaneously,
- incorporating uncertainty estimation,
- combining graph neural networks with transformer architectures.

Most modern graph neural network research follows exactly this strategy:

1. Reproduce an existing model.
2. Introduce a novel modification.
3. Compare the new model against the original baseline.

Without a strong baseline implementation, meaningful scientific comparisons are impossible.

---

# 14.241 Lessons Learned from Reproducing CGCNN

Reproducing the original CGCNN teaches far more than how to implement a single graph neural network.

It develops essential research skills, including

- interpreting scientific literature,
- translating equations into software,
- constructing reproducible machine learning pipelines,
- evaluating experimental results critically,
- identifying sources of variability,
- designing fair baseline comparisons.

These skills are transferable to virtually every modern graph neural network architecture.

Whether implementing MEGNet, SchNet, ALIGNN, M3GNet, or future models, the same principles of careful experimentation and reproducible research continue to apply.

With the completion of this section, the reader has progressed beyond simply *using* CGCNN and has acquired the ability to **reproduce, evaluate, and extend** one of the foundational graph neural network models in materials science—an essential milestone on the path toward independent research.

# 14.242 Common Problems and Debugging CGCNN Models

Building a Crystal Graph Convolutional Neural Network is considerably more complex than implementing a conventional neural network.

A CGCNN pipeline consists of many interconnected stages.

```text
Crystal Structure

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

Graph Neural Network

↓

Training

↓

Prediction
```

An error in **any** of these stages can lead to poor performance or complete training failure.

Professional machine learning researchers often spend more time debugging than writing new code.

Therefore, learning how to diagnose and solve common problems is an essential research skill.

This section discusses the most frequently encountered issues when implementing CGCNN models and presents systematic strategies for identifying and resolving them.

---

# 14.243 Problem 1: Graph Construction Fails

One of the earliest problems occurs during graph generation.

Typical error messages include

```text
IndexError

RuntimeError

ValueError
```

or the generated graph contains

- zero edges,
- isolated atoms,
- incorrect neighbor relationships.

The most common causes are

- invalid CIF files,
- incorrect lattice information,
- failed neighbor searches,
- inappropriate cutoff radii.

For example,

if the cutoff radius is chosen too small,

```text
Cutoff = 1.5 Å
```

many atoms may have **no neighbors**, producing disconnected graphs.

Conversely,

an excessively large cutoff

```text
Cutoff = 15 Å
```

may create an unrealistically dense graph, increasing memory consumption and introducing noisy interactions.

A useful debugging strategy is to inspect the number of neighbors for several randomly selected atoms.

```python
for i in range(len(structure)):

    neighbors = structure.get_neighbors(

        structure[i],

        8.0

    )

    print(

        i,

        len(neighbors)

    )
```

Unexpectedly small or extremely large neighbor counts usually indicate an inappropriate cutoff radius.

---

# 14.244 Problem 2: Incorrect Edge Indices

PyTorch Geometric expects edge indices to satisfy

```text
0 ≤ index < Number of Nodes
```

Suppose a graph contains

```
20 atoms.
```

Valid node indices are therefore

```text
0

↓

19
```

An edge index of

```text
25
```

is invalid.

Such errors often produce messages like

```text
RuntimeError:

index out of bounds
```

Before constructing the graph,

verify the edge index tensor.

```python
print(

    edge_index.min(),

    edge_index.max()

)

print(

    x.shape[0]

)
```

The maximum edge index should always be smaller than the number of nodes.

---

# 14.245 Problem 3: NaN Loss

One of the most alarming situations during training is observing

```text
Loss

↓

NaN
```

Once the loss becomes NaN,

training can no longer continue.

Common causes include

- excessively large learning rates,
- numerical overflow,
- invalid feature values,
- division by zero,
- corrupted datasets.

The first step is to inspect the input tensors.

```python
print(

    torch.isnan(x).any()

)

print(

    torch.isnan(edge_attr).any()

)

print(

    torch.isnan(batch.y).any()

)
```

Every statement should return

```text
False
```

If NaN values are present before training,

the dataset itself must be corrected.

---

# 14.246 Problem 4: Exploding Gradients

Suppose the training loss behaves as follows.

```text
0.82

↓

0.67

↓

0.59

↓

18.2

↓

9134

↓

NaN
```

This pattern usually indicates **exploding gradients**.

Very large gradients produce enormous parameter updates,

causing numerical instability.

One common solution is gradient clipping.

```python
torch.nn.utils.clip_grad_norm_(

    model.parameters(),

    max_norm=5.0

)
```

Gradient clipping limits the magnitude of parameter updates while preserving the gradient direction.

It is widely used in deep neural networks.

---

# 14.247 Problem 5: Training Loss Does Not Decrease

Another common issue is

```text
Epoch

↓

Training Loss

↓

1.21

↓

1.20

↓

1.21

↓

1.19

↓

1.20
```

The loss fluctuates without improving.

Possible causes include

- learning rate too small,
- incorrect graph construction,
- poor node features,
- insufficient model capacity,
- optimization bugs.

A useful debugging approach is to intentionally overfit a very small dataset.

For example,

train the model using only

```
20 crystals.
```

A correctly implemented network should eventually memorize such a small dataset.

If it cannot,

there is likely an implementation error.

---

# 14.248 Problem 6: Severe Overfitting

Suppose the training loss continues decreasing,

while validation loss increases.

```text
Training Loss

↓

0.50

↓

0.32

↓

0.18

↓

0.08
```

```text
Validation Loss

↓

0.54

↓

0.46

↓

0.53

↓

0.71
```

This indicates overfitting.

The network is memorizing the training set rather than learning general physical relationships.

Possible solutions include

- early stopping,
- dropout,
- weight decay,
- larger datasets,
- stronger data diversity,
- reducing model complexity.

---

# 14.249 Problem 7: Underfitting

The opposite situation is underfitting.

Both training and validation losses remain high.

```text
Training Loss

↓

1.20

↓

1.14

↓

1.11

↓

1.09
```

```text
Validation Loss

↓

1.25

↓

1.20

↓

1.16

↓

1.14
```

Neither improves significantly.

Possible causes include

- insufficient hidden dimension,
- too few graph convolution layers,
- inadequate training time,
- poor feature representation.

Increasing model capacity often improves performance.

---

# 14.250 Problem 8: GPU Out of Memory

Large crystal graphs require substantial GPU memory.

A typical error message is

```text
CUDA out of memory
```

Possible solutions include

- reducing batch size,
- decreasing hidden dimension,
- reducing graph depth,
- using mixed-precision training,
- processing smaller graphs.

For example,

changing

```python
batch_size = 64
```

to

```python
batch_size = 16
```

often resolves memory issues.

---

# 14.251 Problem 9: Extremely Slow Training

Suppose one training epoch requires

```
45 minutes.
```

Several factors may contribute.

- Graphs are reconstructed every epoch.
- Data loading is inefficient.
- CPU becomes the bottleneck.
- GPU utilization is low.

Useful optimization strategies include

- precomputing graph objects,
- increasing DataLoader workers,
- storing processed graphs on SSDs,
- batching multiple graphs together.

GPU utilization can be monitored using

```text
nvidia-smi
```

If GPU utilization remains very low,

the bottleneck usually lies in data preprocessing rather than neural network computation.

---

# 14.252 Problem 10: Poor Generalization

The model performs well on the training dataset but poorly on new materials.

This often occurs when

- training data lack chemical diversity,
- crystal systems are highly imbalanced,
- certain elements rarely appear,
- the model is evaluated on materials outside the training distribution.

Generalization improves by

- increasing dataset diversity,
- incorporating additional chemistries,
- expanding structural diversity,
- collecting higher-quality labels.

No neural network can reliably predict materials fundamentally different from those encountered during training.

---

# 14.253 A Systematic Debugging Strategy

Rather than modifying many components simultaneously,

researchers typically isolate one stage at a time.

A recommended debugging workflow is

```text
Check CIF Files

↓

Verify Crystal Structures

↓

Inspect Neighbor Search

↓

Visualize Graph

↓

Verify Node Features

↓

Verify Edge Features

↓

Test Forward Pass

↓

Train Small Dataset

↓

Scale to Full Dataset
```

This step-by-step procedure greatly simplifies the identification of implementation errors.

Attempting to debug every component simultaneously often makes the true source of the problem difficult to identify.

---

# 14.254 Best Practices for Stable CGCNN Training

The following recommendations significantly improve the reliability of CGCNN experiments.

- Fix random seeds before training.
- Save checkpoints frequently.
- Monitor both training and validation loss.
- Plot learning curves after every experiment.
- Verify graph construction visually for several structures.
- Inspect prediction distributions.
- Start with a small subset before scaling to the full dataset.
- Record every hyperparameter and software version.
- Save the best-performing model rather than the final epoch.
- Test inference using unseen crystal structures before deploying the model.

Following these practices reduces debugging time and improves the reproducibility of materials informatics research.

---

# 14.255 Final Remarks on Debugging

Every experienced machine learning researcher encounters failed experiments.

Training instability, graph construction errors, poor convergence, and unexpected prediction behavior are normal parts of developing graph neural networks.

The difference between a beginner and an experienced researcher is **not** the absence of errors—it is the ability to diagnose them systematically.

A careful debugging strategy, combined with rigorous validation and reproducible workflows, ensures that a CGCNN implementation is not only functional but also scientifically reliable.

With this section, the reader now possesses not only the theoretical knowledge and implementation skills required to build a Crystal Graph Convolutional Neural Network, but also the practical expertise needed to troubleshoot, optimize, and deploy CGCNN models in real-world materials informatics research.

# 14.256 Best Practices for CGCNN Research

Implementing a Crystal Graph Convolutional Neural Network is only the first step toward conducting high-quality materials informatics research.

Producing scientifically meaningful results requires far more than obtaining a low prediction error.

Researchers must design experiments that are

- reproducible,
- unbiased,
- statistically reliable,
- computationally efficient,
- scientifically interpretable.

In this final section, we discuss a collection of best practices that have become standard in modern graph neural network research.

Following these recommendations will significantly improve both the quality and credibility of future research projects.

---

# 14.257 Design the Dataset Carefully

The quality of a machine learning model is fundamentally limited by the quality of its training data.

A common misconception is that larger datasets always produce better models.

In reality,

dataset quality is often more important than dataset size.

A good materials dataset should possess

- accurate target labels,
- chemically diverse compounds,
- structurally diverse crystals,
- balanced property distributions,
- minimal duplicate structures.

For example,

suppose a formation-energy dataset contains

```text
95%

Oxides
```

and

```text
5%

Everything Else
```

The trained model may perform well on oxides while generalizing poorly to sulfides, nitrides, carbides, halides, or intermetallic compounds.

A chemically balanced dataset generally produces more robust models.

---

# 14.258 Prevent Data Leakage

One of the most serious mistakes in machine learning is **data leakage**.

Data leakage occurs when information from the test set indirectly influences model training.

Examples include

- normalizing the entire dataset before splitting,
- selecting hyperparameters using the test set,
- accidentally including duplicate crystal structures across splits.

The correct workflow is

```text
Entire Dataset

↓

Train

↓

Validation

↓

Test

↓

Normalize Using

Training Statistics Only
```

The test set must remain completely unseen until the final evaluation.

Violating this principle produces unrealistically optimistic performance estimates.

---

# 14.259 Record Every Experiment

Scientific results should always be reproducible.

For every experiment,

record

- dataset version,
- random seed,
- software versions,
- learning rate,
- hidden dimension,
- graph depth,
- cutoff radius,
- optimizer,
- training time,
- validation metrics,
- test metrics.

A useful experiment log might look like

| Experiment | Seed | Hidden Dim | Layers | MAE |
|------------|-----:|-----------:|-------:|----:|
| Run-01 | 42 | 128 | 4 | 0.053 |
| Run-02 | 7 | 128 | 4 | 0.055 |
| Run-03 | 42 | 256 | 6 | 0.049 |

Without detailed records,

reproducing successful experiments becomes extremely difficult.

---

# 14.260 Use Multiple Random Seeds

Training deep neural networks involves randomness.

Different random initializations may produce slightly different models.

Therefore,

reporting the result from only one training run may be misleading.

Instead,

repeat training several times using different random seeds.

For example,

```text
Seed

↓

42

↓

7

↓

123

↓

999

↓

2024
```

Report

- mean MAE,
- standard deviation.

For instance,

```text
MAE

=

0.051 ± 0.002 eV/atom
```

This provides a more statistically reliable estimate of model performance.

---

# 14.261 Compare Against Strong Baselines

Every proposed improvement should be compared with existing methods.

Typical baseline models include

- Linear Regression,
- Random Forest,
- XGBoost,
- Fully Connected Neural Networks,
- Original CGCNN.

If a new architecture does not outperform a strong baseline,

its practical value may be limited.

Benchmarking against established methods is a fundamental principle of scientific machine learning.

---

# 14.262 Visualize Learning Behavior

Numerical metrics alone rarely reveal the complete picture.

Researchers routinely visualize

- training loss,
- validation loss,
- prediction errors,
- residual distributions,
- predicted-versus-actual plots,
- embedding spaces.

For example,

a training curve may reveal

```text
Training Loss

↓

Validation Loss

↓

Overfitting Begins
```

long before the final evaluation metric changes significantly.

Visualization is therefore an important diagnostic tool.

---

# 14.263 Analyze Failure Cases

Rather than focusing exclusively on successful predictions,

carefully investigate failures.

Questions worth asking include

- Which crystal systems are consistently mispredicted?
- Which chemical families produce the largest errors?
- Are prediction errors correlated with unit-cell size?
- Are metallic systems more difficult than semiconductors?
- Are highly unstable materials predicted less accurately?

Understanding failures often leads to new scientific insights and future model improvements.

---

# 14.264 Interpret the Model Scientifically

Machine learning should support scientific understanding,

not merely generate numerical predictions.

Whenever possible,

attempt to interpret

- learned atomic embeddings,
- crystal embeddings,
- clustering behavior,
- latent feature spaces,
- chemically similar materials.

For example,

if lithium-containing compounds cluster together in the learned embedding space,

the model may have discovered meaningful chemical relationships.

Interpretability strengthens confidence in the learned representation.

---

# 14.265 Validate with Physics

Machine learning predictions should be consistent with known physical principles.

For example,

suppose a model predicts

```text
Formation Energy

=

+25 eV/atom
```

for a stable oxide.

Such a prediction is physically unrealistic.

Similarly,

predicted elastic constants,

band gaps,

or densities should remain within physically reasonable ranges.

Whenever possible,

compare predictions with

- Density Functional Theory,
- experimental measurements,
- established literature.

Machine learning should complement,

not replace,

physical reasoning.

---

# 14.266 Publish Reproducible Research

Modern computational science emphasizes openness and reproducibility.

Whenever possible,

publish

- source code,
- trained models,
- preprocessing scripts,
- hyperparameters,
- dataset splits,
- software versions.

A reproducible project allows other researchers to

- verify published results,
- identify potential errors,
- extend existing work,
- build upon previous discoveries.

Open science accelerates progress across the entire research community.

---

# 14.267 Continuous Learning

Graph neural networks continue to evolve rapidly.

After mastering CGCNN,

researchers should continue exploring

- MEGNet,
- SchNet,
- ALIGNN,
- M3GNet,
- Equivariant Graph Neural Networks,
- Graph Transformers,
- Foundation Models for Materials Science.

Many modern architectures inherit core ideas introduced by CGCNN while incorporating richer physical information and more expressive message-passing mechanisms.

Keeping pace with these developments is essential for conducting cutting-edge materials informatics research.

---

# 14.268 Final Conclusion of Chapter 14

The Crystal Graph Convolutional Neural Network represents one of the most influential developments in computational materials science.

It demonstrated that crystal structures could be represented as graphs and that graph neural networks could automatically learn meaningful structural representations without relying on handcrafted descriptors.

Throughout this chapter, we progressed from the fundamental concepts of crystal graph construction to the complete implementation, training, validation, evaluation, deployment, and critical analysis of CGCNN.

More importantly, we learned not only **how** to use CGCNN, but also **why** it works, **when** it should be applied, **where** it may fail, and **how** to improve upon it.

With this foundation, you are now prepared to understand and implement the next generation of graph neural network architectures for materials science. The concepts of graph construction, message passing, graph pooling, representation learning, and end-to-end optimization developed in this chapter will reappear repeatedly in more advanced models.

The next chapter begins with **Materials Graph Networks (MEGNet)**, which extends the CGCNN framework by introducing graph-level state attributes, more expressive update functions, and a unified message-passing formulation that has become one of the cornerstones of modern materials informatics.
