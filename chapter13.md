# Chapter 13 — Graph Neural Networks for Crystal Materials

---

# Learning Objectives

After completing this chapter, you will be able to

- Explain why crystal structures are naturally represented as graphs.
- Understand the basic concepts of graph theory used in materials informatics.
- Construct crystal graphs from crystal structures using **pymatgen**.
- Identify nodes, edges, node features, and edge features.
- Understand how Graph Neural Networks (GNNs) differ from fully connected neural networks.
- Explain the concept of message passing in GNNs.
- Build graph datasets suitable for deep learning.
- Prepare crystal structures for Graph Neural Network models.
- Understand how graph representations preserve structural information that descriptor-based methods cannot.

By the end of this chapter, you will have the theoretical and practical foundation necessary to study advanced materials GNN architectures such as **CGCNN**, **MEGNet**, **M3GNet**, and **ALIGNN** in the following chapters.

---

# 13.1 Introduction

In the previous chapter, we built fully connected neural networks capable of predicting various materials properties from numerical descriptors.

Our workflow looked like

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

Descriptor Vector

↓

Neural Network

↓

Property Prediction
```

This approach has proven remarkably successful for many materials informatics problems.

However, it also revealed an important limitation.

Before the neural network could begin learning, the crystal structure first had to be converted into a fixed-length vector of handcrafted descriptors.

These descriptors summarized the material using quantities such as

- average atomic number,
- average electronegativity,
- average atomic radius,
- density,
- packing fraction,
- oxidation states,
- structural fingerprints.

Although these descriptors capture important information, they do not preserve the complete crystal structure.

The neural network never directly observes

- atoms,
- chemical bonds,
- local atomic environments,
- crystal connectivity,
- or the topology of the crystal lattice.

Instead, it only receives a list of numbers.

As computational materials science advanced and larger materials databases became available, researchers began asking an important question.

> **Can a neural network learn directly from the crystal structure itself without relying on handcrafted descriptors?**

The answer is **yes**.

The key idea is surprisingly simple.

Instead of representing a crystal as a vector,

we represent it as a **graph**.

---

# 13.2 Why Graphs?

To understand why graphs are so powerful, let us first consider what a crystal actually is.

A crystal consists of

- atoms,
- chemical bonds,
- neighboring atoms,
- repeating unit cells,
- three-dimensional connectivity.

Notice something interesting.

Every atom is connected to other atoms.

This immediately resembles another familiar mathematical object:

a graph.

Instead of viewing a crystal as a collection of descriptors,

we can view it as a network.

Conceptually,

```text
Crystal

↓

Network of Atoms

↓

Graph
```

This simple observation has transformed modern materials informatics.

---

## 13.2.1 Revisiting Descriptor-Based Machine Learning

Let us briefly review the workflow used throughout the previous chapters.

```text
Crystal Structure

↓

Descriptor Engineering

↓

Machine Learning

↓

Predicted Property
```

The success of this workflow depends heavily on the quality of the engineered descriptors.

If important structural information is missing,

the machine learning model cannot recover it.

No matter how powerful the neural network becomes,

it cannot learn information that was discarded before training began.

---

## 13.2.2 A Different Perspective

Suppose instead that we keep the crystal structure intact.

Rather than summarizing it,

we allow the neural network to observe

- every atom,
- every neighboring atom,
- every chemical bond,
- every local environment.

The workflow becomes

```text
Crystal Structure

↓

Graph Representation

↓

Graph Neural Network

↓

Property Prediction
```

Instead of asking

> "Which descriptors should we calculate?"

we ask

> "How can the neural network learn directly from the crystal?"

This shift represents one of the most important conceptual changes in modern materials informatics.

---

## 13.2.3 Why This Matters

Consider two silicon crystals.

Both contain exactly the same chemical composition.

However,

one contains

- vacancies,

while the other contains

- perfect atomic ordering.

A descriptor vector may treat these structures as being very similar.

A graph representation, however,

explicitly preserves the atomic connectivity and local environments, allowing the neural network to distinguish between them more naturally.

Similarly,

small structural changes such as

- dopants,
- defects,
- distortions,
- surface reconstructions,

can significantly alter material properties.

Graph representations preserve these structural differences much better than handcrafted descriptor vectors.

---

# 13.3 A Crystal Is Naturally a Graph

This idea is surprisingly intuitive.

Imagine a simple crystal.

```
      Si

     /  \

   Si----Si

     \  /

      Si
```

A materials scientist immediately recognizes this as a collection of atoms connected through neighboring interactions.

A graph theorist sees exactly the same picture.

The translation is almost immediate.

```text
Atom

↓

Node
```

```text
Neighbor Relationship

↓

Edge
```

Therefore,

every crystal can naturally be converted into a graph.

Unlike descriptor vectors,

this graph preserves the structural relationships between atoms.

---

## 13.3.1 Atoms Become Nodes

In graph theory,

the fundamental object is the **node**.

For crystalline materials,

the correspondence is straightforward.

Every atom becomes one node.

For example,

```
NaCl

↓

Na

↓

Node
```

```
Cl

↓

Node
```

Likewise,

a silicon crystal containing eight atoms becomes

```
8 Atoms

↓

8 Nodes
```

The neural network will eventually learn a representation for every node.

---

### Small Code Example

Suppose we read a crystal structure using **pymatgen**.

```python
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
```

Each atomic site can be accessed using a simple loop.

```python
for atom in structure:
    print(atom.species_string)
```

Possible output

```text
Si
Si
Si
Si
Si
Si
Si
Si
```

Each printed atom will eventually correspond to one node in the graph.

Notice that we are **not** calculating descriptors here.

Instead,

we are reading the actual crystal structure.

---

## 13.3.2 Bonds Become Edges

Nodes alone are not enough.

A graph also requires **edges**.

For crystalline materials,

edges represent neighboring atomic interactions.

Conceptually,

```
Si —— Si
```

becomes

```text
Node

↓

Edge

↓

Node
```

Unlike social networks, where an edge might represent friendship,

in a crystal graph an edge represents

- neighboring atoms,
- chemical bonding,
- local coordination,
- atomic interaction.

These edges allow information to flow through the graph during neural network training.

---

### Why Neighboring Atoms?

One important question naturally arises.

> **How do we know which atoms should be connected?**

Unlike molecules,

crystals rarely contain explicitly stored chemical bonds.

Instead,

neighbor relationships are usually determined using

- cutoff distances,
- nearest-neighbor algorithms,
- Voronoi tessellations,
- coordination analysis.

In materials informatics,

distance-based neighbor searching is one of the most common approaches.

In the next section, we will use **pymatgen** to automatically identify neighboring atoms and construct the edges of a crystal graph.

## 13.3.3 From Crystal Structures to Graphs

At this point, the overall idea behind Graph Neural Networks can be summarized very simply.

Instead of converting a crystal into hundreds of handcrafted descriptors,

we convert it into a graph.

The entire workflow becomes

```text
Crystal Structure

↓

Read Structure

↓

Identify Atoms

↓

Find Neighboring Atoms

↓

Construct Graph

↓

Graph Neural Network

↓

Property Prediction
```

Notice how different this workflow is from the descriptor-based approach studied in the previous chapter.

Previously,

```text
Crystal

↓

Descriptors

↓

Neural Network
```

Now,

```text
Crystal

↓

Graph

↓

Graph Neural Network
```

This may appear to be a small modification,

but in reality it fundamentally changes how the neural network learns about materials.

Instead of learning from manually engineered descriptors,

the network now learns directly from the crystal itself.

---

## 13.3.4 What Information Does the Graph Preserve?

Consider a simple silicon crystal.

```text
        Si

      /    \

    Si ---- Si

      \    /

        Si
```

When represented as a graph,

the following information is naturally preserved.

- Every atom
- Neighboring atoms
- Connectivity
- Local coordination
- Atomic interactions

Unlike descriptor vectors,

the graph retains the relationships between atoms.

For example,

the graph knows that

```text
Atom 0

↓

Connected to

↓

Atom 1

Atom 2

Atom 5
```

Those relationships remain available throughout the learning process.

---

## 13.3.5 What Information Will Each Node Store?

A node is much more than simply an atom label.

Every node stores numerical information describing that atom.

For example,

a silicon atom may initially contain

```text
Atomic Number

↓

14
```

```text
Atomic Mass

↓

28.085
```

```text
Electronegativity

↓

1.90
```

```text
Covalent Radius

↓

111 pm
```

During training,

these values become the initial representation of the node.

Later,

the Graph Neural Network will continuously improve this representation by learning from neighboring atoms.

---

### Small Code Example

Suppose we inspect one atom using **pymatgen**.

```python
atom = structure[0]

print(atom.species_string)
print(atom.specie.Z)
print(atom.specie.atomic_mass)
```

Example output

```text
Si
14
28.085 amu
```

Here,

- `species_string` gives the chemical symbol.
- `Z` is the atomic number.
- `atomic_mass` returns the atomic mass.

These quantities may become part of the node features used by a Graph Neural Network.

---

## 13.3.6 Node Features

The numerical information associated with each node is called its **node feature vector**.

Instead of storing only

```text
Si
```

the node stores a collection of numerical values.

Conceptually,

```text
Si

↓

Node Features

↓

[14,
28.085,
1.90,
111,
4]
```

where the entries might represent

- atomic number,
- atomic mass,
- electronegativity,
- covalent radius,
- number of valence electrons.

The exact choice of node features depends on the model.

Some Graph Neural Networks use only the atomic number.

Others include dozens of atomic properties.

Some modern models even learn the node features automatically.

---

### Creating a Simple Node Feature

Suppose we want every node to store only the atomic number.

```python
node_features = []

for atom in structure:

    node_features.append(
        atom.specie.Z
    )

print(node_features)
```

Possible output

```text
[14, 14, 14, 14, 14, 14, 14, 14]
```

Each value corresponds to one node in the graph.

Although extremely simple,

this already forms a valid node feature representation.

---

## 13.3.7 Why Numerical Features Are Necessary

A neural network cannot process text labels directly.

For example,

the following representation is not suitable for training.

```text
Node 0

↓

Si
```

Instead,

the information must be converted into numerical form.

One possibility is

```text
Si

↓

14
```

or

```text
Si

↓

[14,
28.085,
1.90,
111]
```

Graph Neural Networks perform mathematical operations on numbers,

not chemical symbols.

Therefore,

every atom must ultimately be represented numerically.

---

## 13.3.8 What Information Does an Edge Store?

Just as nodes have features,

edges may also contain important information.

For a pair of neighboring atoms,

the edge may store

- bond distance,
- bond type,
- periodic image,
- relative atomic position,
- coordination information.

Conceptually,

```text
Si -------- Si

↓

Distance

↓

2.35 Å
```

Unlike descriptor-based machine learning,

Graph Neural Networks can directly use this local structural information during training.

---

### Small Code Example

The distance between neighboring atoms can be obtained directly from **pymatgen**.

```python
neighbors = structure.get_neighbors(
    structure[0],
    3.0
)

for neighbor in neighbors:

    print(
        neighbor.index,
        neighbor.nn_distance
    )
```

Example output

```text
1 2.35
2 2.35
5 2.35
7 2.35
```

Here,

- `neighbor.index` identifies the neighboring atom.
- `neighbor.nn_distance` gives the distance between the atoms in ångströms.

These distances frequently become edge features in Graph Neural Networks.

---

## 13.3.9 Constructing the First Edges

We now have enough information to begin constructing a graph.

The first step is to identify neighboring atoms.

For every atom,

we search within a specified cutoff radius.

Conceptually,

```text
Atom

↓

Neighbor Search

↓

Neighbor List

↓

Edges
```

This procedure is repeated for every atom in the crystal.

---

### Small Code Example

The following code begins constructing an edge list.

```python
edge_list = []

cutoff = 3.0

for atom_index in range(len(structure)):

    neighbors = structure.get_neighbors(
        structure[atom_index],
        cutoff
    )

    for neighbor in neighbors:

        edge_list.append(
            (
                atom_index,
                neighbor.index
            )
        )
```

At this stage,

`edge_list` contains pairs of connected atoms.

For example,

```text
(0, 1)

(0, 2)

(1, 3)

(2, 5)
```

Each pair represents one edge in the crystal graph.

---

## 13.3.10 Visualizing the Graph

After identifying nodes and edges,

the crystal can now be viewed as a graph.

Instead of imagining a periodic crystal,

the Graph Neural Network sees

```text
Node 0 -------- Node 1

  |               |

  |               |

Node 2 -------- Node 3
```

Each node stores atomic information.

Each edge stores neighboring relationships.

Information will later flow through these edges during message passing,

allowing every atom to learn from its local chemical environment.

This ability to exchange information between neighboring atoms is the defining characteristic of Graph Neural Networks and is one of the primary reasons they have become so successful for crystal property prediction.

In the next section, we will learn how neighboring atoms are identified efficiently and how **pymatgen** constructs crystal graphs from real crystal structures.

# 13.4 Finding Neighboring Atoms Using pymatgen

In the previous section, we learned that

- atoms become **nodes**,
- neighboring atoms become **edges**.

This naturally raises an important question.

> **How does a computer determine which atoms are neighbors?**

For molecules, this question is often answered by examining chemical bonds.

Crystals, however, are different.

Most crystal structure files such as

- CIF,
- POSCAR,
- CONTCAR,

do **not** explicitly store chemical bonds.

Instead, they store

- lattice vectors,
- atomic species,
- atomic coordinates.

Therefore,

before constructing a graph, we must first determine which atoms interact with one another.

This process is known as **neighbor finding**.

Neighbor finding is one of the most important preprocessing steps in Graph Neural Networks for materials science.

Without it,

there would be no graph,

and therefore no Graph Neural Network.

---

## 13.4.1 What Does a Crystal File Contain?

Suppose we load a silicon crystal.

```python
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
```

At this stage,

the structure object contains

- lattice information,
- atomic positions,
- atomic species,
- periodic boundary conditions.

However,

notice what is missing.

There is **no list of bonds**.

Conceptually,

```text
POSCAR

↓

Lattice

Atomic Positions

Atomic Species

✓
```

but

```text
Chemical Bonds

✗
```

The Graph Neural Network cannot determine connectivity until we compute it.

---

## 13.4.2 Why Not Connect Every Atom?

One possible idea is to connect every atom to every other atom.

Conceptually,

```text
Atom 0

↓

Connect to

↓

Atom 1

Atom 2

Atom 3

Atom 4

...

Atom N
```

Although this seems reasonable,

it creates several problems.

First,

the number of edges grows extremely rapidly.

If a crystal contains

```text
100 atoms
```

connecting every atom to every other atom creates approximately

```text
100 × 99

=

9,900

connections
```

For larger supercells,

the number of connections becomes enormous.

Second,

many atoms are too far apart to interact significantly.

Including distant atoms introduces unnecessary complexity and increases computational cost.

Instead,

we usually connect only **nearby atoms**.

---

## 13.4.3 Distance-Based Neighbor Search

The simplest approach is to define a **cutoff radius**.

Only atoms within this radius are considered neighbors.

Conceptually,

```text
Central Atom

↓

Search Radius

↓

Nearby Atoms

↓

Neighbors
```

Imagine drawing a sphere around an atom.

Every atom inside that sphere becomes a neighboring node.

Every atom outside the sphere is ignored.

For example,

```text
Cutoff Radius

↓

3.0 Å
```

If another atom lies

```text
2.45 Å
```

away,

it becomes a neighbor.

If it lies

```text
4.10 Å
```

away,

it is excluded.

---

## 13.4.4 Searching for Neighbors in pymatgen

The **Structure** class provides a convenient method for neighbor searching.

```python
neighbors = structure.get_neighbors(
    structure[0],
    3.0
)
```

This single line performs several operations.

It

- selects the first atom,
- draws a sphere with a radius of **3.0 Å**,
- identifies all atoms inside the sphere,
- returns the neighboring atoms.

Notice how compact the implementation is.

Instead of manually calculating hundreds of interatomic distances,

**pymatgen** performs the search automatically.

---

## 13.4.5 Understanding the Code

Let us examine the previous code line carefully.

```python
structure[0]
```

This refers to

```text
First Atom

↓

Central Atom
```

The second argument

```python
3.0
```

represents

```text
Search Radius

↓

3.0 Å
```

Finally,

```python
neighbors
```

contains every neighboring atom found within the cutoff radius.

---

## 13.4.6 Examining the Neighbor List

The returned object behaves like a Python list.

We can inspect it directly.

```python
print(neighbors)
```

The output is typically a collection of neighbor objects containing information such as

- neighboring atom,
- distance,
- periodic image,
- coordinates.

Although the printed output can appear complicated,

each entry simply describes one neighboring atom.

---

## 13.4.7 Looping Through the Neighbors

A more useful approach is to inspect each neighbor individually.

```python
for neighbor in neighbors:

    print(neighbor)
```

Every iteration corresponds to one neighboring atom.

Conceptually,

```text
Neighbor List

↓

Neighbor 1

Neighbor 2

Neighbor 3

Neighbor 4
```

This makes it easy to extract specific information.

---

## 13.4.8 Retrieving Neighbor Indices

Every neighboring atom has an index within the crystal structure.

The index identifies which atom participates in the edge.

```python
for neighbor in neighbors:

    print(neighbor.index)
```

Example output

```text
1

2

5

7
```

This tells us that

```text
Atom 0

↓

Connected To

↓

Atom 1

Atom 2

Atom 5

Atom 7
```

These connections will later become graph edges.

---

## 13.4.9 Measuring Bond Distances

Besides identifying neighboring atoms,

we also need the distance between them.

This information is frequently used as an **edge feature**.

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

Each value represents an interatomic distance measured in ångströms.

In many Graph Neural Networks,

bond distances are supplied directly to the model.

---

## 13.4.10 Printing Neighbor Information

We can combine the atom index and distance.

```python
for neighbor in neighbors:

    print(
        f"Neighbor: {neighbor.index}, "
        f"Distance: {neighbor.nn_distance:.2f} Å"
    )
```

Example output

```text
Neighbor: 1, Distance: 2.35 Å

Neighbor: 2, Distance: 2.35 Å

Neighbor: 5, Distance: 2.35 Å

Neighbor: 7, Distance: 2.35 Å
```

This is often the first step in constructing a crystal graph.

---

## 13.4.11 Searching Every Atom

So far,

we have examined only one atom.

A Graph Neural Network, however,

requires the neighbors of **every atom**.

The workflow becomes

```text
Atom 0

↓

Neighbors

↓

Edges

────────────

Atom 1

↓

Neighbors

↓

Edges

────────────

Atom 2

↓

Neighbors

↓

Edges

────────────

Repeat Until

↓

Last Atom
```

---

### Small Code Example

```python
cutoff = 3.0

for atom_index in range(len(structure)):

    neighbors = structure.get_neighbors(

        structure[atom_index],

        cutoff

    )

    print(

        f"Atom {atom_index} "

        f"has {len(neighbors)} neighbors."

    )
```

Example output

```text
Atom 0 has 4 neighbors.

Atom 1 has 4 neighbors.

Atom 2 has 4 neighbors.

...
```

This confirms that every atom can independently discover its local chemical environment.

---

## 13.4.12 Why the Cutoff Radius Matters

Choosing the cutoff radius is an important scientific decision.

If the cutoff is **too small**,

important neighboring atoms may be excluded.

```text
Small Cutoff

↓

Missing Neighbors

↓

Incomplete Graph
```

If the cutoff is **too large**,

the graph may contain unnecessary long-range interactions.

```text
Large Cutoff

↓

Too Many Edges

↓

Higher Computational Cost
```

Therefore,

the cutoff radius should be selected based on

- chemical bonding,
- coordination environment,
- crystal structure,
- recommendations from previous literature.

Many published Graph Neural Network models use cutoff radii between **4 Å** and **8 Å**, although the optimal value depends on the specific material system and model architecture.

---

## 13.4.13 Neighbor Finding in Real Materials Informatics Workflows

In practical research,

neighbor searching is only one step in a much larger pipeline.

```text
Quantum ESPRESSO

↓

Structure Relaxation

↓

POSCAR / CIF

↓

pymatgen

↓

Neighbor Search

↓

Node Features

↓

Edge Features

↓

Crystal Graph

↓

Graph Neural Network
```

Notice that graph construction begins immediately after reading the crystal structure.

Everything that follows depends on correctly identifying neighboring atoms.

---

## 13.4.14 Summary

Neighbor finding transforms a collection of atomic coordinates into a connected crystal graph.

Using **pymatgen**, we can

- identify neighboring atoms,
- compute interatomic distances,
- determine atomic connectivity,
- prepare the information required for Graph Neural Networks.

The neighbor list forms the foundation of graph construction.

Once neighboring atoms have been identified,

we are ready to convert these relationships into **graph edges** and build the complete graph representation required for deep learning.

In the next section, we will construct the edge list, define node and edge features, and assemble our first crystal graph suitable for Graph Neural Network models.

# 13.5 Constructing the Crystal Graph

In the previous section, we learned how to identify neighboring atoms using **pymatgen**.

Neighbor searching provides an important piece of information.

For every atom, we now know

- which atoms surround it,
- how far away they are,
- and how they are connected.

The next step is to convert this information into a mathematical graph that can be processed by a Graph Neural Network.

This process is called **graph construction**.

Graph construction transforms a crystal structure into a collection of

- nodes,
- edges,
- node features,
- edge features.

Together, these four components completely describe the graph.

---

## 13.5.1 Components of a Crystal Graph

Every crystal graph contains four fundamental components.

```text
Crystal Graph

↓

Nodes

↓

Edges

↓

Node Features

↓

Edge Features
```

Each component serves a different purpose.

| Component | Represents |
|-----------|------------|
| Nodes | Atoms |
| Edges | Neighbor relationships |
| Node Features | Atomic properties |
| Edge Features | Information about atomic interactions |

These four objects become the input to a Graph Neural Network.

---

## 13.5.2 Step 1 — Create the Nodes

The easiest component to construct is the node list.

Every atom corresponds to exactly one node.

Suppose our crystal contains

```text
Atom 0

Atom 1

Atom 2

Atom 3
```

The graph therefore contains

```text
Node 0

Node 1

Node 2

Node 3
```

No information is lost during this step.

The graph simply assigns each atom a node identifier.

---

### Small Code Example

The total number of nodes is simply the number of atoms.

```python
num_nodes = len(structure)

print(num_nodes)
```

Example output

```text
8
```

This means the graph will contain eight nodes.

---

## 13.5.3 Assigning Node IDs

Internally,

every node is assigned an integer index.

```text
Node ID

↓

0

1

2

3

...

N-1
```

These identifiers allow edges to reference specific atoms.

For example,

```text
Edge

↓

(2,5)
```

means

```text
Node 2

↓

Connected To

↓

Node 5
```

The node ID itself carries no physical meaning.

It simply identifies a location within the graph.

---

## 13.5.4 Step 2 — Build the Edge List

Once the nodes exist,

we must determine how they are connected.

This information comes directly from the neighbor search performed in the previous section.

Conceptually,

```text
Atom

↓

Neighbor Search

↓

Neighbor List

↓

Edge List
```

The edge list records every connection in the crystal.

---

### Small Code Example

We begin with an empty list.

```python
edge_list = []
```

Next,

we examine every atom.

```python
cutoff = 3.0

for atom_index in range(len(structure)):

    neighbors = structure.get_neighbors(

        structure[atom_index],

        cutoff

    )
```

At this stage,

`neighbors` contains every neighboring atom.

Now we convert each neighbor into an edge.

```python
for neighbor in neighbors:

    edge_list.append(

        (

            atom_index,

            neighbor.index

        )

    )
```

Each tuple represents one graph edge.

---

## 13.5.5 Understanding the Edge List

Suppose the resulting edge list is

```text
(0,1)

(0,2)

(0,5)

(1,3)

(2,6)
```

Each row describes one connection.

For example,

```text
(0,2)
```

means

```text
Node 0

↓

Connected To

↓

Node 2
```

Notice that no chemical equations are involved.

The graph simply records which atoms interact.

---

## 13.5.6 Printing the Edge List

The edge list can be inspected directly.

```python
for edge in edge_list:

    print(edge)
```

Example output

```text
(0, 1)

(0, 2)

(0, 5)

(1, 3)

(2, 6)
```

Although simple,

this list completely defines the graph connectivity.

---

## 13.5.7 Directed and Undirected Edges

An important question now arises.

Should

```text
(0,1)
```

also imply

```text
(1,0)
```

In graph theory,

this distinction determines whether the graph is

- directed,
- or undirected.

Most crystal Graph Neural Networks treat neighboring relationships as **bidirectional**.

If

```text
Atom A

↓

Neighbor

↓

Atom B
```

then

```text
Atom B

↓

Neighbor

↓

Atom A
```

Both directions are stored.

---

### Small Code Example

A bidirectional graph can be constructed by adding two edges.

```python
edge_list.append(

    (

        atom_index,

        neighbor.index

    )

)

edge_list.append(

    (

        neighbor.index,

        atom_index

    )

)
```

This ensures that information can flow in both directions during message passing.

---

## 13.5.8 Avoiding Duplicate Edges

Neighbor searching may produce duplicate connections.

For example,

```text
(0,1)

(1,0)

(0,1)
```

contains repeated information.

Duplicates increase memory usage and unnecessary computation.

One simple solution is to remove repeated edges.

```python
edge_list = list(

    set(edge_list)

)
```

Although many graph libraries automatically handle duplicate edges,

it is useful to understand how they arise.

---

## 13.5.9 Constructing Node Features

The graph topology alone is insufficient.

Every node also requires numerical features.

Earlier,

we learned that useful atomic properties include

- atomic number,
- atomic mass,
- electronegativity,
- covalent radius,
- number of valence electrons.

We now convert these into numerical vectors.

---

### Small Code Example

The simplest feature is the atomic number.

```python
node_features = []

for atom in structure:

    node_features.append(

        atom.specie.Z

    )
```

Printing the result

```python
print(node_features)
```

may produce

```text
[14, 14, 14, 14, 14, 14, 14, 14]
```

Each value corresponds to one node.

---

## 13.5.10 Multiple Node Features

Real Graph Neural Networks rarely rely on a single feature.

Instead,

each node stores several atomic properties.

Conceptually,

```text
Node

↓

Atomic Number

↓

Atomic Mass

↓

Electronegativity

↓

Covalent Radius

↓

Valence Electrons
```

The resulting feature vector may look like

```text
[14,

28.085,

1.90,

111,

4]
```

Each node possesses its own feature vector.

---

### Small Code Example

```python
node_features = []

for atom in structure:

    node_features.append(

        [

            atom.specie.Z,

            float(atom.specie.atomic_mass)

        ]

    )
```

Now every node contains two features instead of one.

Later,

additional atomic properties can easily be added.

---

## 13.5.11 Constructing Edge Features

Edges may also contain numerical information.

The most common edge feature is the interatomic distance.

Conceptually,

```text
Node A

↓

Distance

↓

Node B
```

Unlike node features,

edge features describe the relationship between two atoms.

---

### Small Code Example

```python
edge_features = []

for atom_index in range(len(structure)):

    neighbors = structure.get_neighbors(

        structure[atom_index],

        cutoff

    )

    for neighbor in neighbors:

        edge_features.append(

            [

                neighbor.nn_distance

            ]

        )
```

Every edge now stores its bond distance.

---

## 13.5.12 Keeping Edges and Edge Features Consistent

Notice something important.

Every edge must have a corresponding edge feature.

For example,

```text
Edge

↓

(0,1)

↓

Distance

↓

2.35 Å
```

If the graph contains

```text
150 edges
```

then it must also contain

```text
150 edge feature vectors.
```

The two lists must always remain synchronized.

---

## 13.5.13 Organizing the Complete Graph

At this stage,

our graph consists of four separate objects.

```text
Nodes

↓

8
```

```text
Edges

↓

28
```

```text
Node Features

↓

8 Feature Vectors
```

```text
Edge Features

↓

28 Feature Vectors
```

Together,

these completely describe the crystal graph.

---

## 13.5.14 A Research Workflow

The graph construction process now fits naturally into a materials informatics workflow.

```text
Quantum ESPRESSO

↓

Structure Relaxation

↓

POSCAR

↓

pymatgen

↓

Read Structure

↓

Neighbor Search

↓

Nodes

↓

Edges

↓

Node Features

↓

Edge Features

↓

Crystal Graph
```

Notice that no machine learning has occurred yet.

Everything so far has been devoted to preparing the graph.

Only after graph construction is complete can the Graph Neural Network begin learning.

---

## 13.5.15 Why Graph Construction Matters

Graph construction is often overlooked,

yet it has a major influence on model performance.

Poorly constructed graphs may

- omit important neighboring atoms,
- introduce unnecessary edges,
- lose chemical information,
- reduce prediction accuracy.

Conversely,

a carefully designed graph preserves the local chemical environment and provides the Graph Neural Network with meaningful structural information.

For this reason,

graph construction is considered one of the most important preprocessing stages in modern materials informatics.

---

## 13.5.16 Summary

In this section, we transformed a crystal structure into a mathematical graph.

We learned how to

- create nodes from atoms,
- build edges from neighboring atoms,
- construct node features,
- define edge features,
- assemble the complete graph.

At this point,

we have successfully represented the crystal as a graph.

However,

a new question naturally arises.

> **How can a neural network perform calculations on a graph instead of a fixed-length feature vector?**

The answer requires a new type of deep learning architecture specifically designed for graph-structured data.

In the next section, we will introduce the fundamental concepts of **graph theory** that underpin every modern Graph Neural Network.

# 13.6 Graph Theory Fundamentals for Materials Scientists

Now that we know how to construct a crystal graph,

we need to understand the mathematical language used to describe graphs.

Fortunately,

the graph theory required for Graph Neural Networks is much simpler than many readers initially expect.

In this section,

we will introduce only the concepts that are essential for modern materials informatics.

Rather than studying abstract mathematical graphs,

we will interpret every concept using crystal structures.

---

# 13.6.1 What Is Graph Theory?

Graph theory is a branch of mathematics that studies relationships between objects.

A graph consists of only two fundamental components.

```text
Objects

↓

Nodes
```

```text
Relationships

↓

Edges
```

Everything else in graph theory is built from these two simple ideas.

For crystalline materials,

the correspondence is straightforward.

```text
Crystal

↓

Atoms

↓

Nodes
```

```text
Crystal

↓

Neighbor Relationships

↓

Edges
```

Unlike descriptor-based machine learning,

graph theory allows us to preserve how atoms are connected to one another.

---

# 13.6.2 Graph Notation

Mathematically,

a graph is usually written as

```text
G = (V, E)
```

where

```text
G

↓

Entire Graph
```

```text
V

↓

Vertices (Nodes)
```

```text
E

↓

Edges
```

In materials science,

this notation becomes

```text
G

↓

Crystal Graph
```

```text
V

↓

Atoms
```

```text
E

↓

Neighbor Connections
```

For example,

a silicon unit cell may be represented as

```text
G = (Atoms, Neighbor Relationships)
```

Although the notation appears mathematical,

its physical interpretation is very intuitive.

---

# 13.6.3 Vertices (Nodes)

The terms **vertex** and **node** mean exactly the same thing.

Computer scientists often use the word

```text
Node
```

while mathematicians often use

```text
Vertex
```

Throughout this book,

we will primarily use the term

```text
Node
```

because it is more common in Graph Neural Network literature.

Each node corresponds to one atom.

Conceptually,

```text
Crystal

↓

Li

↓

Node 0
```

```text
Crystal

↓

O

↓

Node 1
```

```text
Crystal

↓

O

↓

Node 2
```

Every atom becomes one node regardless of the chemical species.

---

### Small Code Example

The number of graph nodes is simply

```python
num_nodes = len(structure)

print(num_nodes)
```

For a structure containing twenty atoms,

the output is

```text
20
```

The graph therefore contains twenty nodes.

---

# 13.6.4 Edges

Edges define relationships between nodes.

For crystal graphs,

an edge usually represents neighboring atoms.

Conceptually,

```text
Node A

↓

Neighbor

↓

Node B
```

An edge does **not** necessarily mean that a traditional covalent bond exists.

Instead,

it simply indicates that two atoms are considered neighbors according to the graph construction algorithm.

This distinction is important.

Graph Neural Networks operate on neighboring relationships,

not necessarily on textbook chemical bonds.

---

### Example

Suppose atom 4 is within the cutoff radius of atom 9.

The graph stores

```text
(4,9)
```

or, for an undirected graph,

```text
(4,9)

(9,4)
```

These pairs define the connectivity of the graph.

---

# 13.6.5 Paths

A **path** is a sequence of connected nodes.

Suppose we have

```text
Li

↓

O

↓

Ti

↓

O
```

The information can travel along this sequence.

Graphically,

```text
Node 0

↓

Node 1

↓

Node 2

↓

Node 3
```

This becomes extremely important for Graph Neural Networks.

During message passing,

information propagates along these paths,

allowing distant atoms to indirectly influence one another.

---

# 13.6.6 Neighborhood

The **neighborhood** of a node consists of every node directly connected to it.

Suppose we focus on one silicon atom.

```text
          Si

        / | \

      Si Si Si

        \ | /

          Si
```

The neighboring atoms form the local chemical environment.

Graphically,

```text
Central Node

↓

Neighbor Nodes
```

Every Graph Neural Network repeatedly gathers information from these neighboring nodes.

Neighborhood information is one of the fundamental ideas behind message passing.

---

### Small Code Example

We already know how to retrieve the neighborhood using **pymatgen**.

```python
neighbors = structure.get_neighbors(

    structure[0],

    3.0

)
```

The returned list represents the neighborhood of the first atom.

---

# 13.6.7 Node Degree

One of the simplest graph properties is the **degree** of a node.

The degree is simply the number of edges connected to that node.

Conceptually,

```text
Central Atom

↓

4 Neighboring Atoms

↓

Degree = 4
```

For example,

a tetrahedrally coordinated silicon atom has

```text
Degree

↓

4
```

The degree provides a simple measure of local connectivity.

---

### Small Code Example

```python
degree = len(neighbors)

print(degree)
```

Possible output

```text
4
```

This indicates that the selected atom has four neighboring atoms within the chosen cutoff radius.

---

# 13.6.8 Connected Graphs

A graph is called **connected** if every node can be reached from every other node.

Conceptually,

```text
Node

↓

Node

↓

Node

↓

Node
```

All nodes communicate through some sequence of edges.

If portions of the graph become isolated,

information cannot flow between them.

For crystal structures,

we generally expect the graph to remain connected,

especially when an appropriate cutoff radius is selected.

---

# 13.6.9 Cycles

A **cycle** occurs when a path eventually returns to its starting node.

Example

```text
Node A

↓

Node B

↓

Node C

↓

Node A
```

Cycles appear naturally in many crystal structures.

For example,

hexagonal rings occur in

- graphene,
- graphite,
- hexagonal boron nitride.

These repeating structural motifs influence many physical properties.

Graph Neural Networks naturally capture such connectivity patterns.

---

# 13.6.10 Adjacency

Perhaps the most important concept in graph theory is **adjacency**.

Two nodes are adjacent if they share an edge.

Conceptually,

```text
Node 0

↓

Connected

↓

Node 1
```

Therefore,

Node 0 and Node 1 are adjacent.

If no edge exists,

the nodes are not adjacent.

Adjacency determines how information flows during message passing.

Only adjacent nodes exchange information directly.

---

# 13.6.11 Adjacency Matrix

Instead of storing edges as a list,

a graph can also be represented using an **adjacency matrix**.

Suppose our graph contains four nodes.

```text
      0 1 2 3

0     0 1 1 0

1     1 0 0 1

2     1 0 0 1

3     0 1 1 0
```

Each row represents one node.

Each column represents another node.

A value of

```text
1
```

means

```text
Connected
```

A value of

```text
0
```

means

```text
Not Connected
```

Although modern Graph Neural Networks usually use edge lists rather than adjacency matrices,

understanding adjacency matrices provides valuable intuition.

---

### Small Code Example

A simple adjacency matrix can be created using NumPy.

```python
import numpy as np

adjacency = np.array(

    [

        [0,1,1,0],

        [1,0,0,1],

        [1,0,0,1],

        [0,1,1,0]

    ]

)

print(adjacency)
```

Later,

PyTorch Geometric stores this information in a more memory-efficient format.

---

# 13.6.12 Sparse Graphs

Crystal graphs are typically **sparse**.

This means

most pairs of atoms are **not** directly connected.

Consider a crystal containing

```text
100 Atoms
```

Only a small number of nearby atoms interact with any given atom.

Most entries in the adjacency matrix are therefore

```text
0
```

rather than

```text
1
```

This sparsity allows Graph Neural Networks to scale efficiently to much larger crystal structures.

---

# 13.6.13 Why Graph Theory Matters

At first glance,

graph theory may seem like an abstract branch of mathematics.

In reality,

it provides the language required to describe crystal structures.

Every important concept in Graph Neural Networks—

- nodes,
- neighborhoods,
- edges,
- connectivity,
- adjacency,
- paths,

originates from graph theory.

Understanding these ideas now will make message passing and graph convolutions much easier to understand in the following sections.

---

# 13.6.14 Materials Science Perspective

Notice how naturally graph theory aligns with crystalline materials.

| Graph Theory | Materials Science |
|--------------|-------------------|
| Node | Atom |
| Edge | Neighbor relationship |
| Degree | Coordination number (approximate) |
| Neighborhood | Local atomic environment |
| Path | Chain of interacting atoms |
| Connected Graph | Connected crystal network |
| Adjacency Matrix | Atomic connectivity matrix |

Unlike descriptor vectors,

graphs preserve both

- the properties of individual atoms,
- and the relationships between them.

This preservation of structural information is the key reason Graph Neural Networks have become so successful in computational materials science.

---

# 13.6.15 Summary

Graph theory provides the mathematical foundation for Graph Neural Networks.

In this section,

we introduced the concepts of

- nodes,
- edges,
- neighborhoods,
- degree,
- paths,
- adjacency,
- adjacency matrices,
- connected graphs,
- sparse graphs.

Although these ideas originated in mathematics,

they correspond naturally to the language of crystalline materials.

We now have everything needed to describe a crystal as a graph.

The next challenge is even more interesting.

> **Once a crystal has been converted into a graph, how does a neural network actually learn from it?**

The answer lies in the defining operation of every Graph Neural Network:

**message passing**.

# 13.7 Message Passing — The Heart of Graph Neural Networks

Up to this point, we have successfully converted a crystal structure into a graph.

We now have

- nodes representing atoms,
- edges representing neighboring relationships,
- node features describing atomic properties,
- edge features describing atomic interactions.

However, an important question remains unanswered.

> **How does a Graph Neural Network actually use this graph to make predictions?**

Unlike a fully connected neural network, a Graph Neural Network cannot simply process each atom independently.

Instead,

every atom must be able to **communicate** with its neighboring atoms.

This communication process is known as **message passing**.

Message passing is the defining operation of nearly every modern Graph Neural Network, including

- CGCNN,
- MEGNet,
- M3GNet,
- ALIGNN,
- GraphSAGE,
- Graph Attention Networks (GAT),
- Graph Isomorphism Networks (GIN).

Although different architectures implement message passing in different ways, the underlying idea remains the same.

Every node repeatedly gathers information from its neighbors, combines that information with its own features, and updates its representation.

---

# 13.7.1 Why Do Atoms Need to Exchange Information?

Suppose we wish to predict the formation energy of silicon.

Consider one silicon atom.

```text
      Si
```

If we only know

- its atomic number,
- atomic mass,
- electronegativity,

can we determine the formation energy of the entire crystal?

Of course not.

Formation energy depends on

- neighboring atoms,
- crystal structure,
- bond lengths,
- coordination environment,
- long-range interactions.

One atom alone contains only local information.

To understand the material,

each atom must also learn about its surroundings.

---

## Materials Science Example

Imagine two carbon atoms.

One belongs to

```text
Diamond
```

The other belongs to

```text
Graphite
```

Both atoms have

```text
Atomic Number = 6
```

If a neural network only looked at atomic number,

both atoms would appear identical.

However,

their neighboring atoms are arranged very differently.

```text
Diamond

↓

Tetrahedral coordination
```

```text
Graphite

↓

Planar coordination
```

The local environment changes the material properties dramatically.

Message passing allows each atom to learn this environment automatically.

---

# 13.7.2 The Basic Idea of Message Passing

Imagine that every atom can send information to its neighboring atoms.

Conceptually,

```text
Atom A

↓

Send Information

↓

Atom B
```

At the same time,

Atom B also sends information back.

```text
Atom B

↓

Send Information

↓

Atom A
```

Every neighboring atom exchanges information simultaneously.

This process repeats throughout the crystal.

After several rounds,

each atom possesses information about not only itself but also its surrounding environment.

---

## Information Flow

The workflow is remarkably simple.

```text
Current Node Features

↓

Receive Neighbor Information

↓

Combine Information

↓

Update Node Features

↓

Repeat
```

Every Graph Neural Network follows this general strategy.

The only difference lies in

- how messages are computed,
- how messages are combined,
- how node features are updated.

---

# 13.7.3 A Silicon Crystal Example

Consider the following simplified crystal graph.

```text
        Si

      /    \

    Si ---- Si

      \    /

        Si
```

Focus on the upper silicon atom.

Initially,

its feature vector might be

```text
[14,
28.085,
1.90]
```

representing

- atomic number,
- atomic mass,
- electronegativity.

During message passing,

the neighboring atoms send their own information.

Conceptually,

```text
Neighbor 1

↓

Neighbor 2

↓

Central Atom

↑

Neighbor 3
```

The central atom then combines all received information into a richer representation.

Instead of knowing only

```text
"I am silicon."
```

it now also learns

```text
"My neighbors are silicon."

"My coordination is tetrahedral."

"My bond lengths are approximately 2.35 Å."
```

This richer representation is much more informative for predicting material properties.

---

# 13.7.4 Message Passing Is Not Physical Communication

The phrase **message passing** can sometimes be misleading.

Atoms are **not** physically sending messages.

Instead,

the Graph Neural Network performs mathematical operations that combine information from neighboring nodes.

Conceptually,

```text
Neighbor Features

↓

Mathematical Computation

↓

Updated Node Features
```

The "messages" are simply vectors of numerical values.

---

# 13.7.5 One Round of Message Passing

A single message-passing iteration consists of four steps.

```text
Node Features

↓

Collect Neighbor Features

↓

Aggregate Information

↓

Update Node Features
```

Every node performs these operations simultaneously.

After one iteration,

every atom has incorporated information from its immediate neighbors.

---

# 13.7.6 Looking at the Code

Most Graph Neural Network libraries hide the mathematical complexity.

A complete message-passing step may appear surprisingly simple.

```python
x = conv(x, edge_index)
```

Although this is only one line of code,

it performs many operations internally.

The layer

- identifies neighboring nodes,
- gathers their feature vectors,
- computes messages,
- aggregates those messages,
- updates every node feature.

This is why Graph Neural Networks are so powerful.

A single line of code performs an entire neighborhood aggregation operation.

---

## Understanding the Variables

Let us examine the previous code carefully.

```python
x
```

contains

```text
Node Features
```

For example,

```text
Node 0

↓

[14, 28.085]
```

```text
Node 1

↓

[8, 15.999]
```

The variable

```python
edge_index
```

stores the graph connectivity.

Conceptually,

```text
Node 0

↓

Connected To

↓

Node 1
```

```text
Node 1

↓

Connected To

↓

Node 2
```

The object

```python
conv
```

represents a graph convolution layer.

It performs message passing automatically.

---

# 13.7.7 How Far Can Information Travel?

After one message-passing step,

each node only knows about its immediate neighbors.

```text
Step 1

↓

Immediate Neighbors
```

After two message-passing steps,

information can travel one level farther.

```text
Node

↓

Neighbor

↓

Neighbor's Neighbor
```

After three layers,

the receptive field grows even larger.

```text
Central Atom

↓

First Shell

↓

Second Shell

↓

Third Shell
```

The deeper the network,

the larger the portion of the crystal each atom can "see."

---

## Example

Suppose we stack three graph convolution layers.

```python
x = conv1(x, edge_index)

x = conv2(x, edge_index)

x = conv3(x, edge_index)
```

Although each layer only exchanges information between neighboring atoms,

stacking layers allows information to propagate across increasingly larger regions of the crystal.

---

# 13.7.8 Aggregating Neighbor Information

When a node receives messages from multiple neighbors,

it must combine them into a single representation.

This process is called **aggregation**.

Conceptually,

```text
Neighbor A

↓

Neighbor B

↓

Neighbor C

↓

Aggregate

↓

Updated Node
```

Common aggregation methods include

- sum,
- mean,
- maximum.

Different Graph Neural Networks choose different aggregation strategies.

---

### Simple Python Example

Suppose a node receives three scalar messages.

```python
messages = [2.1, 1.8, 2.4]
```

Using the average,

```python
average = sum(messages) / len(messages)

print(average)
```

Output

```text
2.1
```

Real Graph Neural Networks perform this operation on multidimensional feature vectors rather than single numbers.

---

# 13.7.9 Updating the Node Representation

After aggregation,

the node combines the received information with its own features.

Conceptually,

```text
Current Features

+

Neighbor Information

↓

Updated Features
```

The updated representation now contains richer structural information.

For example,

instead of

```text
[14]
```

the node may evolve into

```text
[-0.82,
1.15,
0.47,
...
]
```

Notice something interesting.

These values no longer correspond to familiar physical quantities.

Instead,

they are **learned representations** generated by the neural network.

These learned features often contain much more predictive information than the original atomic descriptors.

---

# 13.7.10 Why Multiple Message Passing Layers Are Useful

One round of message passing captures only local information.

Many material properties,

however,

depend on interactions extending beyond the first coordination shell.

By stacking multiple Graph Neural Network layers,

information propagates farther through the crystal.

```text
Layer 1

↓

Nearest Neighbors
```

```text
Layer 2

↓

Neighbors of Neighbors
```

```text
Layer 3

↓

Larger Local Environment
```

This progressively enlarges the receptive field of the model.

---

# 13.7.11 Message Passing in Materials Science

Let us revisit the complete workflow.

```text
Crystal Structure

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

Message Passing

↓

Updated Node Features

↓

Pooling

↓

Property Prediction
```

Notice that message passing occurs **before** the final property prediction.

Its purpose is not to make predictions directly.

Instead,

it transforms simple atomic features into rich representations that encode the local chemical environment.

These learned representations become the foundation for predicting

- band gaps,
- formation energies,
- elastic constants,
- dielectric properties,
- adsorption energies,
- battery voltages,
- thermal conductivity,
- and many other materials properties.

---

# 13.7.12 Summary

Message passing is the fundamental operation that distinguishes Graph Neural Networks from traditional neural networks.

Instead of processing atoms independently,

each node repeatedly gathers information from its neighboring atoms, combines that information with its own features, and updates its representation.

Through repeated message-passing iterations, every atom gradually learns about its local chemical environment, allowing the network to build increasingly informative representations of the crystal structure.

However, one important question still remains.

> **After every atom has learned a rich representation of its local environment, how do we combine all of those atomic representations into a single prediction for the entire crystal?**

The answer lies in the next stage of a Graph Neural Network:

**graph pooling and graph-level representations**.

# 13.8 Graph Pooling — From Atomic Features to Material Properties

After several rounds of message passing,

every atom in the crystal has learned a rich representation of its local chemical environment.

At this stage,

each node contains much more information than when the graph was first constructed.

However,

another important challenge appears.

> **Most materials properties belong to the entire crystal, not to individual atoms.**

For example,

we usually wish to predict

- formation energy,
- band gap,
- elastic modulus,
- bulk modulus,
- density,
- dielectric constant,

for the **entire material**.

A Graph Neural Network therefore needs a way to combine information from all atoms into a single representation of the crystal.

This operation is called **graph pooling** (also known as **graph readout**).

---

# 13.8.1 Why Pooling Is Necessary

Suppose we have a silicon crystal containing eight atoms.

After message passing,

the Graph Neural Network produces eight learned feature vectors.

Conceptually,

```text
Node 0

↓

Feature Vector
```

```text
Node 1

↓

Feature Vector
```

```text
...

```

```text
Node 7

↓

Feature Vector
```

But our target is

```text
Band Gap

↓

One Number
```

or

```text
Formation Energy

↓

One Number
```

How do we convert

```text
Eight Feature Vectors

↓

One Crystal Representation
```

The answer is graph pooling.

---

# 13.8.2 The Basic Idea

Graph pooling gathers information from every node and combines it into a single graph-level feature vector.

Conceptually,

```text
Node Features

↓

Pooling

↓

Crystal Feature Vector

↓

Prediction
```

Instead of treating atoms independently,

the neural network summarizes the entire crystal into one numerical representation.

---

# 13.8.3 Pooling Is Similar to a Team Report

Imagine a research group consisting of several scientists.

Each scientist studies one aspect of a material.

For example,

Scientist 1 studies

```text
Crystal Structure
```

Scientist 2 studies

```text
Electronic Structure
```

Scientist 3 studies

```text
Mechanical Properties
```

Before publishing a paper,

their individual findings are combined into one report.

Graph pooling performs a similar task.

Each node contributes information,

and the pooling operation combines those contributions into one graph representation.

---

# 13.8.4 Pooling Workflow

The complete workflow becomes

```text
Crystal Structure

↓

Graph

↓

Message Passing

↓

Updated Node Features

↓

Pooling

↓

Crystal Representation

↓

Prediction
```

Notice that pooling occurs **after** message passing.

The nodes must first learn meaningful representations before they are combined.

---

# 13.8.5 Sum Pooling

The simplest pooling method is **sum pooling**.

Suppose four nodes contain the following scalar features.

```text
1.2

2.0

0.8

1.5
```

Sum pooling simply adds them together.

```text
1.2

+

2.0

+

0.8

+

1.5

↓

5.5
```

For multidimensional feature vectors,

the addition is performed element by element.

---

### Small Code Example

Using PyTorch,

sum pooling can be demonstrated with a simple example.

```python
import torch

node_features = torch.tensor(

    [

        [1.2, 0.5],

        [2.0, 1.0],

        [0.8, 0.3]

    ]

)

graph_feature = node_features.sum(dim=0)

print(graph_feature)
```

Output

```text
tensor([4.0000, 1.8000])
```

Each feature has been summed across all nodes.

---

# 13.8.6 Mean Pooling

Instead of summing,

we may compute the average.

Conceptually,

```text
All Node Features

↓

Average

↓

Graph Feature
```

Mean pooling is useful when graphs contain different numbers of atoms.

Without averaging,

larger crystals naturally produce larger summed values.

---

### Small Code Example

```python
graph_feature = node_features.mean(dim=0)

print(graph_feature)
```

Output

```text
tensor([1.3333, 0.6000])
```

The graph representation is now independent of the number of atoms.

---

# 13.8.7 Max Pooling

Another common strategy is **max pooling**.

Instead of averaging,

the largest value from each feature dimension is selected.

Conceptually,

```text
Feature 1

↓

Largest Value
```

```text
Feature 2

↓

Largest Value
```

This emphasizes the strongest signal among all atoms.

---

### Small Code Example

```python
graph_feature = node_features.max(dim=0).values

print(graph_feature)
```

Output

```text
tensor([2.0000, 1.0000])
```

Only the largest value from each feature column is retained.

---

# 13.8.8 Which Pooling Method Is Best?

Different Graph Neural Networks use different pooling strategies.

Each has its advantages.

| Pooling Method | Characteristics |
|---------------|-----------------|
| Sum | Preserves contributions from every atom |
| Mean | Independent of graph size |
| Max | Emphasizes dominant features |

There is no universally best choice.

The optimal pooling strategy depends on

- the prediction task,
- the dataset,
- the neural network architecture.

Many crystal Graph Neural Networks successfully use **sum pooling** because many extensive material properties naturally increase with the number of atoms.

---

# 13.8.9 Pooling in PyTorch Geometric

Fortunately,

modern graph libraries perform pooling automatically.

For example,

PyTorch Geometric provides

```python
from torch_geometric.nn import global_mean_pool
```

A graph-level representation can be obtained using

```python
graph_embedding = global_mean_pool(

    x,

    batch

)
```

Here,

```python
x
```

contains the updated node features,

while

```python
batch
```

indicates which nodes belong to each graph when multiple crystal structures are processed simultaneously.

---

## Understanding the `batch` Tensor

Suppose we train using two crystal structures.

The first contains

```text
4 Atoms
```

The second contains

```text
3 Atoms
```

The corresponding batch tensor is

```python
batch = torch.tensor(

    [0, 0, 0, 0, 1, 1, 1]

)
```

This tells PyTorch Geometric

- nodes 0–3 belong to graph 0,
- nodes 4–6 belong to graph 1.

The pooling function computes one graph representation for each crystal automatically.

---

# 13.8.10 From Graph Embedding to Prediction

After pooling,

we finally have one feature vector describing the entire crystal.

Conceptually,

```text
Graph Embedding

↓

Fully Connected Layer

↓

Predicted Property
```

For regression,

the output might be

```text
Predicted Formation Energy

↓

-3.82 eV/atom
```

or

```text
Predicted Band Gap

↓

1.64 eV
```

The graph embedding contains the structural information required for these predictions.

---

### Small Code Example

A simple prediction head may look like

```python
import torch.nn as nn

predictor = nn.Linear(

    64,

    1

)

prediction = predictor(graph_embedding)
```

Here,

the graph embedding contains 64 learned features,

which are transformed into a single predicted material property.

---

# 13.8.11 Complete Graph Neural Network Workflow

We can now summarize the entire Graph Neural Network pipeline.

```text
Crystal Structure

↓

pymatgen

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

Graph Convolution Layer

↓

Message Passing

↓

Updated Node Features

↓

Graph Pooling

↓

Crystal Embedding

↓

Prediction Layer

↓

Material Property
```

Notice how every stage contributes to the final prediction.

Unlike descriptor-based machine learning,

the neural network has learned directly from the crystal structure itself.

---

# 13.8.12 Materials Science Example

Suppose we wish to predict the band gap of several semiconductor materials.

The workflow becomes

```text
POSCAR

↓

Read Structure

↓

Neighbor Search

↓

Crystal Graph

↓

Graph Neural Network

↓

Graph Pooling

↓

Band Gap Prediction
```

For formation energy prediction,

only the final target changes.

```text
Crystal Graph

↓

Graph Neural Network

↓

Graph Pooling

↓

Formation Energy Prediction
```

Exactly the same architecture can also be used for predicting

- elastic constants,
- adsorption energies,
- dielectric constants,
- thermal conductivity,
- battery voltages,

simply by changing the training labels.

---

# 13.8.13 Code Fragment — A Minimal GNN Forward Pass

The following simplified code demonstrates how all the components discussed so far fit together.

```python
class SimpleGNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 64)

        self.linear = nn.Linear(64, 1)

    def forward(self, data):

        x = self.conv1(
            data.x,
            data.edge_index
        )

        x = x.relu()

        x = self.conv2(
            x,
            data.edge_index
        )

        x = global_mean_pool(
            x,
            data.batch
        )

        prediction = self.linear(x)

        return prediction
```

Even though this model contains only a few lines of code,

it performs

- graph convolutions,
- message passing,
- graph pooling,
- and property prediction.

In later chapters,

we will replace these generic graph convolution layers with specialized architectures such as **CGCNN**, **MEGNet**, **M3GNet**, and **ALIGNN**, which are specifically designed for crystalline materials.

---

# 13.8.14 Summary

Graph pooling bridges the gap between atomic information and material properties.

After message passing has enriched every node representation,

pooling combines those atomic representations into a single graph-level embedding that describes the entire crystal.

This graph embedding is then used to predict material properties such as band gaps, formation energies, elastic constants, and dielectric properties.

At this point, we have completed the core workflow of a Graph Neural Network:

- construct a crystal graph,
- perform message passing,
- aggregate node information through pooling,
- predict material properties.

In the next section, we will examine how these ideas are implemented in real deep learning frameworks, beginning with **PyTorch Geometric**, the most widely used library for Graph Neural Networks in computational materials science.

# 13.9 Introduction to PyTorch Geometric for Materials Science

So far, we have learned the theoretical foundations of Graph Neural Networks.

We now understand

- how crystals are converted into graphs,
- how neighboring atoms become edges,
- how message passing updates atomic representations,
- and how graph pooling produces crystal-level features.

The next step is to implement these ideas in code.

Fortunately, we do **not** need to build Graph Neural Networks from scratch.

Modern deep learning libraries provide highly optimized implementations of graph operations.

The most widely used library is **PyTorch Geometric (PyG)**.

Throughout the remainder of this book, we will use **PyTorch Geometric** because

- it is built on top of PyTorch,
- it is widely adopted in academic research,
- it supports many Graph Neural Network architectures,
- it provides efficient GPU implementations,
- and it integrates naturally with materials science workflows.

By the end of this section, you will understand how crystal graphs are represented inside PyTorch Geometric and how they are prepared for training.

---

# 13.9.1 What Is PyTorch Geometric?

PyTorch Geometric is an extension of PyTorch designed specifically for graph-structured data.

Instead of working only with

- images,
- text,
- or tabular data,

it provides tools for working with

- social networks,
- molecular graphs,
- protein structures,
- transportation networks,
- and crystal structures.

For materials science,

PyTorch Geometric allows us to represent

```text
Crystal

↓

Graph

↓

Graph Neural Network

↓

Property Prediction
```

without manually implementing graph algorithms.

---

# 13.9.2 Installing PyTorch Geometric

The installation depends on the version of PyTorch already installed.

A common installation procedure is

```bash
pip install torch-geometric
```

Depending on your operating system and CUDA version,

additional packages may also be required.

In a research environment,

it is good practice to verify the installation before beginning model development.

---

### Checking the Installation

```python
import torch
import torch_geometric

print(torch.__version__)
print(torch_geometric.__version__)
```

If both version numbers are displayed,

the installation has been completed successfully.

---

# 13.9.3 The Data Object

The central object in PyTorch Geometric is the **Data** class.

Instead of storing a crystal using multiple unrelated variables,

everything is packaged into a single object.

Conceptually,

```text
Data Object

↓

Node Features

↓

Edges

↓

Edge Features

↓

Target Property
```

This organization makes graph processing much easier.

---

### Creating a Data Object

```python
from torch_geometric.data import Data
```

Suppose we already have

- node features,
- graph connectivity,
- edge features.

They can be combined as

```python
graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_features

)
```

Here,

```python
x
```

stores node features,

```python
edge_index
```

stores graph connectivity,

and

```python
edge_attr
```

stores edge features.

---

# 13.9.4 Node Features (`x`)

The attribute

```python
graph.x
```

contains the feature vector for every node.

Conceptually,

```text
Node

↓

Atomic Features
```

Suppose each atom stores

- atomic number,
- atomic mass,
- electronegativity.

Then

```python
graph.x
```

might look like

```python
tensor(

    [

        [14, 28.085, 1.90],

        [14, 28.085, 1.90],

        [14, 28.085, 1.90]

    ]

)
```

Each row corresponds to one atom.

Each column corresponds to one feature.

---

### Code Fragment

The tensor can be created directly.

```python
import torch

x = torch.tensor(

    [

        [14.0, 28.085],

        [14.0, 28.085],

        [14.0, 28.085]

    ],

    dtype=torch.float

)
```

Notice that PyTorch expects floating-point values for neural network computations.

---

# 13.9.5 Graph Connectivity (`edge_index`)

Perhaps the most important object in PyTorch Geometric is

```python
edge_index
```

Unlike an adjacency matrix,

PyTorch Geometric stores graph connectivity as a compact list of edges.

Suppose we have

```text
Node 0

↓

Node 1
```

and

```text
Node 0

↓

Node 2
```

The graph is represented as

```python
edge_index = torch.tensor(

    [

        [0, 0],

        [1, 2]

    ]

)
```

Each column represents one edge.

---

### Understanding the Shape

Notice something important.

```python
edge_index.shape
```

returns

```text
(2, Number of Edges)
```

The first row stores

```text
Source Nodes
```

The second row stores

```text
Destination Nodes
```

This representation is far more memory-efficient than storing an entire adjacency matrix.

---

# 13.9.6 Bidirectional Edges

Most crystal graphs are undirected.

Therefore,

both directions are usually stored.

Instead of

```text
0 → 1
```

we store

```text
0 → 1

1 → 0
```

The corresponding tensor becomes

```python
edge_index = torch.tensor(

    [

        [0,1,0,2,1,0,2,0],

        [1,0,2,0,0,1,0,2]

    ]

)
```

Although this increases the number of stored edges,

it enables information to flow in both directions during message passing.

---

# 13.9.7 Edge Features (`edge_attr`)

Many Graph Neural Networks also require information about each edge.

The most common edge feature is

```text
Bond Distance
```

Suppose four edges have distances

```text
2.35 Å

2.35 Å

2.48 Å

2.52 Å
```

These become

```python
edge_attr = torch.tensor(

    [

        [2.35],

        [2.35],

        [2.48],

        [2.52]

    ]

)
```

Notice that every edge has its own feature vector.

---

# 13.9.8 Target Property (`y`)

Machine learning always requires a target.

For materials science,

the target may be

- band gap,
- formation energy,
- bulk modulus,
- elastic constant,
- dielectric constant.

Suppose we wish to predict formation energy.

```python
y = torch.tensor(

    [-3.82],

    dtype=torch.float

)
```

The complete graph object becomes

```python
graph = Data(

    x=x,

    edge_index=edge_index,

    edge_attr=edge_attr,

    y=y

)
```

Everything describing one crystal is now contained in a single object.

---

# 13.9.9 Inspecting a Graph

PyTorch Geometric provides a convenient summary.

```python
print(graph)
```

A typical output is

```text
Data(

    x=[8, 10],

    edge_index=[2, 32],

    edge_attr=[32, 1],

    y=[1]

)
```

This output immediately tells us

- 8 atoms,
- 10 node features,
- 32 edges,
- one edge feature,
- one target property.

Being able to interpret this summary is an essential debugging skill when building Graph Neural Networks.

---

# 13.9.10 Building a Crystal Graph from pymatgen

We can now connect everything we have learned.

```text
POSCAR

↓

pymatgen

↓

Read Structure

↓

Neighbor Search

↓

Node Features

↓

Edge List

↓

Edge Features

↓

PyTorch Geometric Data Object
```

This pipeline forms the basis of nearly every modern crystal Graph Neural Network.

---

### Simplified Implementation

```python
structure = Structure.from_file("POSCAR")

node_features = ...

edge_index = ...

edge_features = ...

graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_features

)
```

The omitted sections (`...`) represent the preprocessing steps developed in the previous sections.

---

# 13.9.11 Processing Multiple Crystals

A machine learning model is trained on many crystal structures rather than a single graph.

For example,

```text
Crystal 1

↓

Graph
```

```text
Crystal 2

↓

Graph
```

```text
Crystal 3

↓

Graph
```

PyTorch Geometric combines these individual graphs into batches for efficient GPU training.

Fortunately,

this batching process is handled automatically.

The researcher simply prepares one `Data` object for each crystal.

---

### Code Fragment

```python
dataset = [

    graph1,

    graph2,

    graph3

]
```

Later,

a `DataLoader` combines these graphs into mini-batches during training.

We will study batching in detail in the next section.

---

# 13.9.12 Why PyTorch Geometric Is Important for Materials Science

Although it is possible to implement Graph Neural Networks from scratch,

doing so is rarely necessary.

PyTorch Geometric provides

- efficient graph data structures,
- optimized graph convolution layers,
- GPU acceleration,
- automatic batching,
- pooling operations,
- and many state-of-the-art Graph Neural Network architectures.

As a result,

researchers can focus on

- crystal representation,
- model design,
- feature engineering,
- and scientific interpretation,

rather than low-level graph algorithms.

For this reason,

PyTorch Geometric has become one of the standard tools in computational materials science.

---

# 13.9.13 Summary

PyTorch Geometric provides a practical framework for implementing Graph Neural Networks.

In this section, we learned how crystal graphs are represented using the `Data` object and how node features, edge connectivity, edge features, and target properties are organized for deep learning.

These data structures provide the foundation for building complete Graph Neural Network models.

In the next section, we will bring everything together by constructing a **complete crystal graph dataset** from multiple crystal structures and preparing it for model training using PyTorch Geometric's `Dataset` and `DataLoader` classes.

# 13.10 Building a Crystal Graph Dataset for Materials Science

In the previous section, we learned how a **single crystal structure** can be represented as a graph using PyTorch Geometric.

A single graph, however, is not enough to train a Graph Neural Network.

Machine learning requires **many examples**.

Instead of one crystal,

we may have

- 500 crystal structures,
- 5,000 crystal structures,
- 50,000 crystal structures,
- or even millions of crystal structures.

Each crystal must be converted into its own graph before training can begin.

This section explains how an entire materials dataset is transformed into a collection of graph objects suitable for Graph Neural Networks.

The workflow developed here is almost identical to that used in modern materials informatics research.

---

# 13.10.1 The Dataset Workflow

Suppose we have a directory containing relaxed crystal structures.

```text
Materials Dataset

│

├── Material_001

│      POSCAR

│      band_gap.txt

│

├── Material_002

│      POSCAR

│      band_gap.txt

│

├── Material_003

│      POSCAR

│      band_gap.txt

│

└── ...
```

For every material,

we perform the same sequence of operations.

```text
Crystal Structure

↓

Read Structure

↓

Neighbor Search

↓

Node Features

↓

Edge Features

↓

Graph Object

↓

Dataset
```

This process is repeated until every crystal has been converted into a graph.

---

# 13.10.2 One Graph per Material

An important idea to remember is

> **One crystal structure corresponds to one graph.**

For example,

```text
Silicon

↓

Graph 1
```

```text
Gallium Arsenide

↓

Graph 2
```

```text
Titanium Dioxide

↓

Graph 3
```

The complete dataset therefore becomes

```text
Graph 1

Graph 2

Graph 3

...

Graph N
```

Each graph stores

- crystal structure,
- atomic features,
- neighboring relationships,
- target property.

---

# 13.10.3 Organizing the Dataset

In Python,

the simplest dataset is just a list.

```python
dataset = []
```

Every graph created during preprocessing is appended to this list.

```python
dataset.append(graph)
```

After processing every material,

the dataset might contain

```text
500 graphs
```

or

```text
25,000 graphs
```

depending on the size of the materials database.

---

# 13.10.4 Reading Multiple Crystal Structures

Suppose every material has its own POSCAR file.

The following simplified code loops over all structures.

```python
from pathlib import Path

data_folder = Path("dataset")

for material in data_folder.iterdir():

    poscar = material / "POSCAR"

    structure = Structure.from_file(poscar)

    print(structure.formula)
```

Notice that the workflow remains exactly the same regardless of whether we process one crystal or ten thousand crystals.

---

# 13.10.5 Converting Each Crystal into a Graph

Inside the loop,

each crystal is converted into a graph.

Conceptually,

```text
Structure

↓

Node Features

↓

Edge Index

↓

Edge Features

↓

Graph Object
```

Although we already know how to construct these objects,

the important point is that the same preprocessing pipeline is repeated for every material.

---

### Code Fragment

```python
for material in data_folder.iterdir():

    structure = Structure.from_file(

        material / "POSCAR"

    )

    node_features = build_node_features(structure)

    edge_index, edge_attr = build_edges(structure)

    graph = Data(

        x=node_features,

        edge_index=edge_index,

        edge_attr=edge_attr

    )

    dataset.append(graph)
```

Here,

`build_node_features()` and `build_edges()` represent helper functions that implement the procedures developed earlier in this chapter.

---

# 13.10.6 Adding the Target Property

Machine learning requires both

- inputs,
- outputs.

The graph is the input.

The material property is the output.

Suppose we wish to predict the band gap.

```python
band_gap = 1.42
```

The graph becomes

```python
graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_attr,

    y=torch.tensor(

        [band_gap],

        dtype=torch.float

    )

)
```

Now the graph contains both

- the crystal representation,
- and the correct answer.

During training,

the Graph Neural Network learns to predict `y` from the graph.

---

# 13.10.7 Storing Additional Information

Besides the target property,

it is often useful to retain metadata.

For example,

```python
graph.material_id = "mp-149"

graph.formula = "Si"

graph.spacegroup = 227
```

Although these attributes are not used directly during training,

they are extremely useful when

- debugging,
- visualizing results,
- interpreting predictions,
- comparing with experimental data.

Many research projects store additional metadata alongside each graph for later analysis.

---

# 13.10.8 Splitting the Dataset

Before training,

the dataset must be divided into three parts.

```text
Entire Dataset

↓

Training Set

↓

Validation Set

↓

Test Set
```

Each subset serves a different purpose.

| Dataset | Purpose |
|----------|---------|
| Training | Learn model parameters |
| Validation | Tune hyperparameters |
| Test | Evaluate final model |

Keeping the test set completely unseen is essential for obtaining an unbiased estimate of model performance.

---

### Example

Suppose we have

```text
10,000 Crystal Structures
```

A common split is

```text
Training

80%

↓

8,000
```

```text
Validation

10%

↓

1,000
```

```text
Test

10%

↓

1,000
```

Different projects may use slightly different ratios,

but the underlying idea remains the same.

---

# 13.10.9 Creating DataLoaders

Processing every graph individually would be inefficient.

Instead,

graphs are grouped into **mini-batches**.

PyTorch Geometric provides the `DataLoader` class for this purpose.

```python
from torch_geometric.loader import DataLoader
```

Creating a training loader is straightforward.

```python
train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True

)
```

Similarly,

validation and test loaders can be created.

```python
valid_loader = DataLoader(

    valid_dataset,

    batch_size=32

)

test_loader = DataLoader(

    test_dataset,

    batch_size=32

)
```

The `DataLoader` automatically combines graphs of different sizes into batches, allowing efficient GPU training.

---

# 13.10.10 Inspecting a Batch

Each batch contains multiple crystal graphs.

Conceptually,

```text
Crystal A

↓

Graph
```

```text
Crystal B

↓

Graph
```

```text
Crystal C

↓

Graph
```

↓

```text
Mini-Batch
```

The batch object stores

- node features,
- graph connectivity,
- edge features,
- graph assignments,
- target values.

---

### Code Fragment

```python
for batch in train_loader:

    print(batch)

    break
```

Typical output

```text
DataBatch(

    x=[512, 32],

    edge_index=[2, 2944],

    edge_attr=[2944, 1],

    batch=[512],

    y=[32]

)
```

From this summary we can immediately determine

- the total number of atoms in the batch,
- the number of edges,
- the number of node features,
- and the number of graphs in the batch.

Learning to interpret this output is an important practical skill.

---

# 13.10.11 Understanding the `batch` Tensor

One of the most confusing concepts for beginners is the `batch` tensor.

Suppose our mini-batch contains three crystal structures.

```text
Crystal 1

↓

4 Atoms
```

```text
Crystal 2

↓

3 Atoms
```

```text
Crystal 3

↓

5 Atoms
```

PyTorch Geometric stores all atoms in one large tensor.

The `batch` tensor records which graph each atom belongs to.

Example

```python
batch = torch.tensor(

    [

        0,0,0,0,

        1,1,1,

        2,2,2,2,2

    ]

)
```

This tells the Graph Neural Network

- atoms 0–3 belong to Crystal 1,
- atoms 4–6 belong to Crystal 2,
- atoms 7–11 belong to Crystal 3.

During graph pooling,

this tensor ensures that each crystal receives its own graph-level representation.

---

# 13.10.12 Complete Dataset Pipeline

We can now summarize the entire preprocessing workflow.

```text
Quantum ESPRESSO

↓

Relaxed Structure

↓

POSCAR / CIF

↓

pymatgen

↓

Neighbor Search

↓

Node Features

↓

Edge Features

↓

Graph Object

↓

Dataset

↓

Training / Validation / Test Split

↓

DataLoader

↓

Graph Neural Network
```

This is the complete pipeline followed by many modern Graph Neural Network studies in computational materials science.

---

# 13.10.13 Best Practices for Materials Datasets

When building a graph dataset for research,

it is important to follow several best practices.

- Verify that every crystal structure can be parsed successfully.
- Use a consistent neighbor-finding strategy across the entire dataset.
- Normalize target properties when appropriate.
- Remove duplicate or corrupted structures.
- Keep preprocessing scripts under version control for reproducibility.
- Store material identifiers alongside each graph for traceability.
- Record preprocessing parameters such as cutoff radius and selected node features.

Careful dataset preparation often has a greater impact on model performance than increasing model complexity.

---

# 13.10.14 Summary

In this section, we transformed a collection of crystal structures into a complete graph dataset suitable for deep learning.

We learned how to

- convert each crystal into a graph,
- attach target properties,
- organize graphs into datasets,
- split the dataset into training, validation, and test sets,
- and prepare mini-batches using PyTorch Geometric's `DataLoader`.

At this point, we have everything required to train a Graph Neural Network on materials data.

In the next section, we will build our **first complete Graph Neural Network model** for crystal property prediction and follow the entire training workflow from initialization to evaluation.

# 13.11 Building Your First Graph Neural Network for Crystal Property Prediction

We have now completed the entire data preparation pipeline.

Starting from crystal structures, we have

- converted crystals into graphs,
- generated node features,
- generated edge features,
- created PyTorch Geometric `Data` objects,
- organized the graphs into datasets,
- and prepared mini-batches using `DataLoader`.

The final step is to build a complete Graph Neural Network and train it to predict a materials property.

In this section, we will construct a simple but complete Graph Neural Network using **PyTorch Geometric**.

Although the model presented here is intentionally simple, it contains the essential building blocks used in many research papers.

In the next chapter, we will replace these generic graph convolution layers with specialized architectures such as **CGCNN**.

---

# 13.11.1 The Overall Workflow

Before writing any code, let us review the complete workflow.

```text
Crystal Structure

↓

Crystal Graph

↓

Node Features

↓

Graph Convolution Layers

↓

Message Passing

↓

Graph Pooling

↓

Fully Connected Layer

↓

Predicted Property
```

During training,

the prediction is compared with the true material property.

```text
Prediction

↓

Loss Function

↓

Backpropagation

↓

Optimizer

↓

Updated Model
```

This process repeats over many epochs until the model learns meaningful relationships between crystal structures and material properties.

---

# 13.11.2 Importing the Required Libraries

Our first step is to import the required Python libraries.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
```

Here,

- `torch` provides tensor operations,
- `nn` contains neural network layers,
- `GCNConv` implements graph convolution,
- `global_mean_pool` creates graph-level representations.

---

# 13.11.3 Defining the Neural Network

Like every PyTorch model,

our Graph Neural Network inherits from `nn.Module`.

```python
class CrystalGNN(nn.Module):

    def __init__(

        self,

        input_dim,

        hidden_dim

    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.output = nn.Linear(
            hidden_dim,
            1
        )
```

This model contains

- two graph convolution layers,
- one prediction layer.

Although simple,

this architecture is sufficient for demonstrating the complete workflow.

---

# 13.11.4 Writing the Forward Function

The `forward()` function describes how data moves through the network.

```python
def forward(self, data):

    x = data.x

    edge_index = data.edge_index

    batch = data.batch

    x = self.conv1(

        x,

        edge_index

    )

    x = F.relu(x)

    x = self.conv2(

        x,

        edge_index

    )

    x = F.relu(x)

    x = global_mean_pool(

        x,

        batch

    )

    prediction = self.output(x)

    return prediction
```

Notice how closely the code matches the workflow introduced earlier.

```text
Node Features

↓

Graph Convolution

↓

ReLU

↓

Graph Convolution

↓

Pooling

↓

Prediction
```

Understanding this correspondence between theory and implementation is essential for becoming comfortable with Graph Neural Networks.

---

# 13.11.5 Creating the Model

Once the class has been defined,

creating the model is straightforward.

```python
model = CrystalGNN(

    input_dim=32,

    hidden_dim=64

)
```

Here,

`32` represents the number of node features,

while

`64` is the hidden embedding dimension.

These values depend on how the graph was constructed.

---

# 13.11.6 Choosing the Loss Function

Most materials property prediction tasks involve regression.

Typical targets include

- band gap,
- formation energy,
- elastic modulus,
- bulk modulus.

For regression,

Mean Squared Error (MSE) is commonly used.

```python
criterion = nn.MSELoss()
```

The loss measures how far the predicted property is from the true value.

For example,

```text
True Band Gap

↓

1.42 eV
```

```text
Predicted Band Gap

↓

1.65 eV
```

The larger the difference,

the larger the loss.

---

# 13.11.7 Choosing an Optimizer

Next,

we select an optimizer.

The most common choice for Graph Neural Networks is **Adam**.

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001

)
```

Here,

```text
Learning Rate

↓

0.001
```

This value is often a good starting point for many materials informatics problems,

although it should always be tuned experimentally.

---

# 13.11.8 One Training Step

The core of every neural network training loop consists of a few familiar steps.

```python
batch = next(iter(train_loader))

prediction = model(batch)

loss = criterion(

    prediction,

    batch.y

)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Although this code contains only a few lines,

it performs

- forward propagation,
- loss computation,
- backpropagation,
- gradient descent.

Everything discussed in Chapter 12 now comes together.

---

## Understanding Each Line

```python
prediction = model(batch)
```

The Graph Neural Network predicts a material property.

```python
loss = criterion(
    prediction,
    batch.y
)
```

The prediction is compared with the true value.

```python
loss.backward()
```

Backpropagation computes gradients.

```python
optimizer.step()
```

Gradient descent updates the model parameters.

---

# 13.11.9 Writing the Training Loop

Instead of training on only one mini-batch,

we repeat the process for every batch in every epoch.

```python
num_epochs = 100

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        prediction = model(batch)

        loss = criterion(

            prediction,

            batch.y

        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(

        f"Epoch {epoch+1}",

        total_loss

    )
```

This loop forms the foundation of almost every Graph Neural Network training script.

---

# 13.11.10 Evaluating the Model

After training,

we switch the model into evaluation mode.

```python
model.eval()
```

Gradients are no longer required.

```python
with torch.no_grad():

    for batch in test_loader:

        prediction = model(batch)
```

Using `torch.no_grad()` reduces memory usage and speeds up inference.

---

# 13.11.11 Measuring Model Performance

For regression problems,

several evaluation metrics are commonly used.

| Metric | Interpretation |
|---------|----------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

In materials science,

**MAE** is one of the most frequently reported metrics.

For example,

```text
Band Gap MAE

↓

0.18 eV
```

or

```text
Formation Energy MAE

↓

0.042 eV/atom
```

Reporting physically meaningful units helps researchers understand the practical significance of model performance.

---

### Computing MAE

```python
mae = torch.mean(

    torch.abs(

        prediction -

        batch.y

    )

)

print(mae)
```

---

# 13.11.12 Visualizing Predictions

Numerical metrics are useful,

but visualization often provides deeper insight.

A common approach is to compare predicted values with reference values.

Conceptually,

```text
True Property

↓

Scatter Plot

↓

Predicted Property
```

If the model performs perfectly,

all points lie along the diagonal.

Although plotting will be covered in later chapters,

this visualization is standard in nearly every materials informatics paper.

---

### Code Fragment

```python
import matplotlib.pyplot as plt

plt.scatter(

    batch.y.numpy(),

    prediction.numpy()

)

plt.xlabel(

    "True Band Gap"

)

plt.ylabel(

    "Predicted Band Gap"

)
```

For real projects,

remember to move tensors to the CPU before converting them to NumPy arrays if training on a GPU.

---

# 13.11.13 Complete Training Pipeline

At this point,

we can summarize the entire Graph Neural Network workflow.

```text
Crystal Structures

↓

pymatgen

↓

Graph Construction

↓

PyTorch Geometric Dataset

↓

DataLoader

↓

Graph Neural Network

↓

Training

↓

Evaluation

↓

Predicted Material Properties
```

This is the complete pipeline followed in many modern Graph Neural Network studies.

Although individual research papers introduce more sophisticated architectures,

the overall workflow remains remarkably similar.

---

# 13.11.14 Limitations of This Simple Model

The model developed in this section is intentionally simple.

Several important aspects of state-of-the-art crystal Graph Neural Networks are missing.

For example,

- edge features are not explicitly used,
- periodic boundary conditions are not considered,
- bond distances are ignored,
- angular information is absent,
- only a basic graph convolution layer is used.

These limitations motivate the development of specialized architectures.

Researchers have designed models that incorporate far richer information about crystal structures.

Among the earliest and most influential is the

**Crystal Graph Convolutional Neural Network (CGCNN).**

---

# 13.11.15 From Generic GNNs to Crystal-Specific GNNs

The Graph Neural Network presented here demonstrates the general principles of graph-based learning.

However,

crystalline materials possess unique characteristics.

For example,

they exhibit

- periodic atomic arrangements,
- well-defined coordination environments,
- chemically meaningful bond distances,
- symmetry relationships,
- long-range structural order.

Generic graph convolution layers were not originally designed with these characteristics in mind.

To address this,

materials scientists developed specialized Graph Neural Networks tailored specifically for crystalline materials.

The first major breakthrough was the **Crystal Graph Convolutional Neural Network (CGCNN)**.

CGCNN introduced a graph construction strategy and convolution operation specifically designed for crystal structures, making it one of the foundational models in materials informatics.

The next chapter is devoted entirely to understanding how CGCNN works, how it differs from generic Graph Neural Networks, and how to implement it for real materials science applications.

---

# 13.11.16 Summary

In this section, we built our first complete Graph Neural Network for crystal property prediction.

Starting with crystal graphs, we defined a neural network architecture, implemented graph convolution layers, trained the model using gradient descent, evaluated its performance, and interpreted the results using common regression metrics.

Although this model is intentionally simple, it demonstrates every essential stage of the Graph Neural Network workflow and provides the foundation needed to understand more advanced crystal-specific architectures.

In the next chapter, we move beyond generic Graph Neural Networks and study the **Crystal Graph Convolutional Neural Network (CGCNN)**—one of the most influential deep learning models developed specifically for crystalline materials.

# 13.12 Building Node Features for Crystal Graphs

A Graph Neural Network cannot work directly with atoms.

Although a crystal graph tells us **which atoms are connected**, the neural network also needs to know **what each atom is**.

This information is stored in the **node features**.

Recall from earlier sections that each atom in the crystal becomes one node in the graph.

```text
Crystal

↓

Atoms

↓

Nodes

↓

Node Features
```

Node features are one of the most important components of any Graph Neural Network.

Even if two crystal graphs have identical connectivity,

their predictions may differ dramatically because of the information stored inside each node.

For example,

consider two crystal structures.

```text
Silicon

↓

Si
```

and

```text
Germanium

↓

Ge
```

The graph topology may be similar,

but the atomic properties are different.

Without node features,

the Graph Neural Network would not know the difference between silicon and germanium.

---

# 13.12.1 What Are Node Features?

A node feature is simply a numerical description of an atom.

Instead of storing

```text
Atom

↓

Si
```

we store

```text
Atom

↓

Feature Vector
```

For example,

```text
Silicon

↓

[14,
28.085,
1.90,
111.0,
4]
```

Each number represents one physical property of the atom.

Collectively,

these numbers form the node feature vector.

---

# 13.12.2 Why Are Node Features Necessary?

Imagine that every atom were represented only by its index.

```text
Atom 0

↓

0
```

```text
Atom 1

↓

1
```

```text
Atom 2

↓

2
```

These numbers carry no chemical meaning.

The Graph Neural Network cannot determine

- which atom is oxygen,
- which atom is lithium,
- which atom is titanium.

Instead,

each node should contain chemically meaningful information.

For example,

```text
Lithium

↓

Atomic Number = 3
```

```text
Oxygen

↓

Atomic Number = 8
```

```text
Titanium

↓

Atomic Number = 22
```

The network can now distinguish different elements.

---

# 13.12.3 Common Atomic Features

Many atomic properties can be used as node features.

Some of the most common are

| Feature | Physical Meaning |
|---------|------------------|
| Atomic number | Identity of the element |
| Atomic mass | Mass of the atom |
| Group | Position in periodic table |
| Period | Periodic table row |
| Electronegativity | Ability to attract electrons |
| Atomic radius | Approximate atomic size |
| Covalent radius | Typical bond length contribution |
| Valence electrons | Number of outer-shell electrons |
| Electron affinity | Ability to gain electrons |
| Ionization energy | Energy required to remove an electron |

Notice that these are all **intrinsic atomic properties**.

They do not depend on the crystal structure.

---

# 13.12.4 Node Features in Materials Science

Unlike social networks,

where node features may represent

- age,
- occupation,
- location,

materials science uses physical and chemical descriptors.

Conceptually,

```text
Atom

↓

Physical Properties

↓

Node Features
```

The goal is to provide the Graph Neural Network with enough information to distinguish one chemical element from another.

---

# 13.12.5 Obtaining Atomic Properties with pymatgen

One of the greatest advantages of **pymatgen** is that it provides convenient access to elemental properties.

Suppose we create a silicon element.

```python
from pymatgen.core import Element

si = Element("Si")
```

We can now inspect several properties.

```python
print(si.Z)
```

Output

```text
14
```

The atomic number is stored in the attribute

```python
Z
```

---

### Atomic Mass

```python
print(si.atomic_mass)
```

Example output

```text
28.085 amu
```

---

### Electronegativity

```python
print(si.X)
```

Output

```text
1.90
```

where

```text
X

↓

Pauling Electronegativity
```

---

### Periodic Table Group

```python
print(si.group)
```

Output

```text
14
```

---

### Period

```python
print(si.row)
```

Output

```text
3
```

---

# 13.12.6 Creating a Feature Vector

Once the properties have been collected,

they can be combined into one feature vector.

```python
from pymatgen.core import Element

element = Element("Si")

features = [

    element.Z,

    float(element.atomic_mass),

    element.X,

    element.group,

    element.row

]

print(features)
```

Example output

```text
[14,
28.085,
1.9,
14,
3]
```

This vector represents one silicon atom.

---

# 13.12.7 Building Features for an Entire Crystal

A crystal usually contains many atoms.

Suppose our structure contains

```text
Si

Si

Si

Si

Si

Si

Si

Si
```

Each atom receives its own feature vector.

The workflow becomes

```text
Crystal

↓

Loop Over Atoms

↓

Feature Vector

↓

Node Feature Matrix
```

---

### Code Example

```python
from pymatgen.core import Structure
from pymatgen.core import Element

structure = Structure.from_file("POSCAR")

node_features = []

for site in structure:

    element = Element(site.specie.symbol)

    features = [

        element.Z,

        float(element.atomic_mass),

        element.X,

        element.group,

        element.row

    ]

    node_features.append(features)
```

After the loop,

`node_features`

contains one feature vector for every atom.

---

# 13.12.8 Converting to a PyTorch Tensor

Neural networks operate on tensors,

not Python lists.

The conversion is straightforward.

```python
import torch

x = torch.tensor(

    node_features,

    dtype=torch.float

)

print(x.shape)
```

Suppose the crystal contains

```text
16 Atoms
```

and we use

```text
5 Features
```

The output becomes

```text
torch.Size([16, 5])
```

Meaning

```text
16 Nodes

×

5 Features
```

This tensor is exactly what PyTorch Geometric expects.

---

# 13.12.9 Inspecting the Feature Matrix

Suppose the first three atoms are

```text
Si

Si

O
```

The tensor may appear as

```text
tensor(

[
 [14.0000,28.0850,1.9000,14.0000,3.0000],

 [14.0000,28.0850,1.9000,14.0000,3.0000],

 [ 8.0000,15.9990,3.4400,16.0000,2.0000]

]

)
```

Notice that

- the silicon atoms share identical intrinsic atomic properties,
- oxygen has a different feature vector.

The Graph Neural Network can therefore distinguish different chemical species immediately.

---

# 13.12.10 Choosing Good Node Features

More features do **not** always produce better models.

A useful feature should

- have physical meaning,
- distinguish different elements,
- contribute relevant chemical information,
- be available for every atom.

Poorly chosen features can

- increase computational cost,
- introduce redundancy,
- reduce generalization.

Feature engineering therefore remains an important aspect of materials informatics.

---

# 13.12.11 Learned Embeddings vs Handcrafted Features

Earlier Graph Neural Networks often relied heavily on handcrafted atomic descriptors.

For example,

```text
Atomic Number

Atomic Radius

Electronegativity

Valence Electrons
```

Modern architectures increasingly learn atomic representations automatically.

Instead of manually designing dozens of features,

the model begins with a small amount of elemental information and learns richer atomic embeddings during training.

This is similar to how word embeddings are learned in natural language processing.

We will encounter this idea again when studying **CGCNN** and **M3GNet**.

---

# 13.12.12 Materials Science Perspective

Choosing node features is fundamentally different from traditional descriptor engineering.

In descriptor-based machine learning,

we computed one feature vector for the **entire crystal**.

For example,

```text
Crystal

↓

SOAP Descriptor

↓

Machine Learning Model
```

In Graph Neural Networks,

we instead compute features for **every atom**.

```text
Crystal

↓

Each Atom

↓

Node Feature Matrix

↓

Graph Neural Network
```

The network then learns how these atomic features interact through message passing.

This is one of the major conceptual shifts from classical machine learning to graph-based deep learning.

---

# 13.12.13 Summary

Node features provide the Graph Neural Network with the chemical identity of every atom in a crystal.

Using **pymatgen**, we can extract physically meaningful atomic properties such as atomic number, atomic mass, electronegativity, group, and period, and organize them into a node feature matrix suitable for PyTorch Geometric.

This matrix forms the foundation of every crystal graph.

However, nodes alone are not enough.

To accurately model crystalline materials, we must also describe the **relationships between atoms**.

These relationships are encoded through **edge features**, which will be the focus of the next section.

# 13.13 Building Edge Features for Crystal Graphs

In the previous section, we learned how to construct **node features**, which describe the properties of individual atoms.

However,

knowing the atoms alone is not sufficient.

Consider the following question.

> **How does the Graph Neural Network know which atoms interact with one another?**

The answer lies in the **edges**.

Edges describe the relationships between atoms,

but modern Graph Neural Networks require more than simply knowing that two atoms are connected.

They also need information **about the connection itself**.

This information is stored as **edge features**.

For crystal Graph Neural Networks,

edge features are just as important as node features.

In fact,

models such as **CGCNN**, **MEGNet**, and **M3GNet** achieve much of their success because they incorporate rich edge information during message passing.

---

# 13.13.1 What Are Edge Features?

Recall that every edge represents a neighboring relationship between two atoms.

```text
Atom A

↓

Neighbor Relationship

↓

Atom B
```

Instead of storing only

```text
Connected
```

we also store numerical information describing that connection.

For example,

```text
Si ---- Si

↓

Bond Distance = 2.35 Å
```

This numerical value becomes an edge feature.

---

# 13.13.2 Why Are Edge Features Important?

Imagine two silicon atoms.

One pair is separated by

```text
2.35 Å
```

Another pair is separated by

```text
3.20 Å
```

If the Graph Neural Network only knows that the atoms are connected,

both relationships appear identical.

However,

from a physical perspective,

they are very different.

Bond length influences

- bond strength,
- orbital overlap,
- electronic structure,
- elastic behavior,
- and many other material properties.

Edge features allow the neural network to learn these differences.

---

# 13.13.3 Common Edge Features

Several quantities can be stored for every edge.

| Edge Feature | Physical Meaning |
|--------------|------------------|
| Bond distance | Distance between atoms |
| Displacement vector | Relative position of neighboring atoms |
| Periodic image | Neighboring unit cell information |
| Bond angle* | Angular relationship (used in some models) |
| Radial basis expansion | Encoded distance representation |

\*Bond angles are generally introduced in more advanced architectures such as ALIGNN.

Among these,

the most fundamental edge feature is the **interatomic distance**.

---

# 13.13.4 Obtaining Neighbor Distances with pymatgen

Fortunately,

**pymatgen** can compute neighboring atoms and their distances automatically.

Suppose we load a crystal.

```python
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
```

We now search for neighboring atoms within a cutoff radius.

```python
neighbors = structure.get_neighbors(

    structure[0],

    r=3.0

)
```

Each neighbor returned contains valuable information.

---

### Inspecting One Neighbor

```python
neighbor = neighbors[0]

print(neighbor)
```

Besides the neighboring atom,

the object also stores

- distance,
- coordinates,
- periodic image.

These values are essential for constructing edge features.

---

# 13.13.5 Extracting Bond Distances

The simplest edge feature is the interatomic distance.

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

These values correspond to the four nearest silicon neighbors.

Each distance becomes one edge feature.

---

# 13.13.6 Building the Edge Feature Matrix

Suppose every edge stores only one feature.

```text
Distance
```

The feature matrix becomes

```text
Edge 0

↓

2.35
```

```text
Edge 1

↓

2.48
```

```text
Edge 2

↓

2.35
```

---

### Code Example

```python
edge_attr = []

for neighbor in neighbors:

    edge_attr.append(

        [

            neighbor.nn_distance

        ]

    )
```

Notice that each edge feature is itself stored as a list,

because PyTorch expects

```text
Number of Edges

×

Number of Edge Features
```

---

# 13.13.7 Converting Edge Features to a Tensor

Once all edge features have been collected,

they are converted into a PyTorch tensor.

```python
import torch

edge_attr = torch.tensor(

    edge_attr,

    dtype=torch.float

)

print(edge_attr.shape)
```

Suppose the graph contains

```text
48 Edges
```

Each edge stores

```text
1 Feature
```

The tensor shape becomes

```text
torch.Size([48,1])
```

---

# 13.13.8 Building the Edge Index

The Graph Neural Network also needs to know

which atoms each edge connects.

This information is stored in

```python
edge_index
```

Suppose

```text
Atom 0

↓

Atom 1
```

The edge is stored as

```python
[0,1]
```

If

```text
Atom 0

↓

Atom 2
```

we store

```python
[0,2]
```

---

### Complete Code

```python
edge_index = []

for i, site in enumerate(structure):

    neighbors = structure.get_neighbors(

        site,

        r=3.0

    )

    for neighbor in neighbors:

        j = neighbor.index

        edge_index.append(

            [i, j]

        )
```

Notice that every neighboring pair generates one edge.

---

# 13.13.9 Converting the Edge Index

PyTorch Geometric expects a tensor with shape

```text
(2, Number of Edges)
```

Therefore,

we transpose the edge list.

```python
edge_index = torch.tensor(

    edge_index,

    dtype=torch.long

).t().contiguous()
```

The transpose converts

```text
[[0,1],

 [0,2],

 [1,0]]
```

into

```text
[[0,0,1],

 [1,2,0]]
```

which is the format required by PyTorch Geometric.

---

# 13.13.10 Building `edge_index` and `edge_attr` Together

In practice,

both tensors are constructed simultaneously.

```python
edge_index = []

edge_attr = []

for i, site in enumerate(structure):

    neighbors = structure.get_neighbors(

        site,

        r=3.0

    )

    for neighbor in neighbors:

        edge_index.append(

            [i, neighbor.index]

        )

        edge_attr.append(

            [

                neighbor.nn_distance

            ]

        )
```

Finally,

```python
edge_index = torch.tensor(

    edge_index,

    dtype=torch.long

).t().contiguous()

edge_attr = torch.tensor(

    edge_attr,

    dtype=torch.float

)
```

At this point,

both graph connectivity and edge features are complete.

---

# 13.13.11 Periodic Boundary Conditions

Crystal structures extend infinitely in space.

As a result,

a neighboring atom may belong to a **periodic image** rather than the original unit cell.

Conceptually,

```text
Unit Cell

↓

Neighbor

↓

Periodic Image
```

Fortunately,

**pymatgen** automatically accounts for periodic boundary conditions during neighbor searches.

This means

```python
structure.get_neighbors()
```

returns the physically correct neighboring atoms,

even when they lie outside the original unit cell.

This is one of the major advantages of using specialized materials science libraries.

---

# 13.13.12 Why Distance Alone Is Sometimes Not Enough

Although bond distance is extremely informative,

modern Graph Neural Networks often require richer edge representations.

For example,

two bonds may have similar lengths but very different chemical environments.

Researchers therefore transform bond distances into higher-dimensional feature vectors before passing them to the neural network.

One of the most common techniques is **Gaussian basis expansion**.

---

# 13.13.13 Gaussian Distance Expansion

Instead of representing a bond using

```text
2.35 Å
```

we represent it as

```text
[0.02,

0.18,

0.76,

0.91,

0.43,

...]

```

Each value measures how close the bond distance is to a predefined Gaussian center.

This transformation gives the neural network a smoother and more expressive representation of interatomic distances.

CGCNN uses this idea extensively.

We will study Gaussian distance expansion in detail in the next chapter.

---

# 13.13.14 Complete Graph Construction

At this point,

we have everything required to build a graph.

```text
Crystal Structure

↓

Node Features

↓

Edge Index

↓

Edge Features

↓

Graph Object
```

The corresponding code becomes

```python
from torch_geometric.data import Data

graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_attr

)
```

This object now contains

- atomic features,
- graph connectivity,
- edge features,

and is ready for input into a Graph Neural Network.

---

# 13.13.15 Debugging Edge Construction

When building crystal graphs,

it is important to verify that the graph has been constructed correctly.

A few useful checks include

```python
print(edge_index.shape)

print(edge_attr.shape)

print(node_features.shape)
```

Typical output might be

```text
torch.Size([2, 192])

torch.Size([192, 1])

torch.Size([64, 5])
```

From these shapes we immediately know

- the graph contains 64 atoms,
- 192 edges,
- one edge feature per edge.

Inspecting tensor shapes is one of the simplest ways to detect preprocessing errors.

---

# 13.13.16 Materials Science Perspective

Unlike traditional descriptor-based machine learning,

Graph Neural Networks learn from **relationships** between atoms.

Those relationships are encoded by the edges.

Node features answer the question

> **What is this atom?**

Edge features answer the question

> **How is this atom related to its neighbors?**

Together,

they provide the Graph Neural Network with a rich representation of the crystal structure.

This combination enables modern Graph Neural Networks to capture structural information that would be difficult to express using handcrafted descriptors alone.

---

# 13.13.17 Summary

Edge features describe the relationships between neighboring atoms in a crystal graph.

Using **pymatgen**, we learned how to identify neighboring atoms, compute interatomic distances, construct the `edge_index` tensor, and build the `edge_attr` matrix required by PyTorch Geometric.

These tensors, together with the node feature matrix, form the complete graph representation used by modern crystal Graph Neural Networks.

However,

one important question still remains.

> **How should neighboring atoms be selected in the first place?**

Should we use a fixed cutoff distance?

A fixed number of neighbors?

Or more sophisticated chemistry-aware algorithms?

The next section explores the most widely used **neighbor-finding algorithms** in computational materials science and explains when each method should be used.

# 13.14 Neighbor Finding Algorithms for Crystal Graph Construction

In the previous section, we learned how to build node features and edge features.

However, one important question still remains.

> **How do we decide which atoms should be connected by edges?**

At first glance, this may seem like a simple problem.

After all,

couldn't we just connect atoms that are close together?

Unfortunately,

real crystal structures are much more complicated.

Different materials have

- different bond lengths,
- different coordination environments,
- different crystal symmetries,
- different atomic densities.

A single neighbor-finding strategy does not work equally well for every material.

For this reason,

neighbor finding is one of the most important preprocessing steps in Graph Neural Networks for materials science.

An incorrectly constructed graph can significantly reduce prediction accuracy,

regardless of how sophisticated the neural network itself may be.

In this section,

we will study the most commonly used neighbor-finding algorithms and learn how they are implemented using **pymatgen**.

---

# 13.14.1 Why Neighbor Finding Matters

Consider two different crystals.

```text
Diamond

↓

Each Carbon

↓

4 Neighbors
```

Now consider a face-centered cubic metal.

```text
Copper

↓

Each Copper Atom

↓

12 Neighbors
```

If we always connect exactly four neighbors,

the copper graph will lose important structural information.

If we always connect twelve neighbors,

diamond will gain many unrealistic long-distance interactions.

Therefore,

the choice of neighbor-finding algorithm directly affects the graph that the Graph Neural Network learns from.

---

# 13.14.2 The General Workflow

Regardless of the algorithm,

the overall workflow remains the same.

```text
Crystal Structure

↓

Neighbor Finding Algorithm

↓

Neighbor List

↓

Graph Construction

↓

Graph Neural Network
```

Only the second step changes.

---

# 13.14.3 Method 1 — Fixed Cutoff Radius

The simplest approach is to connect atoms within a specified distance.

For example,

```text
Cutoff Radius

↓

3.0 Å
```

Any atom located within

```text
3.0 Å
```

is considered a neighbor.

Conceptually,

```text
Central Atom

↓

Search Sphere

↓

Neighboring Atoms
```

---

## Code Example

```python
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")

neighbors = structure.get_neighbors(

    structure[0],

    r=3.0

)

print(len(neighbors))
```

The output tells us how many neighboring atoms were found within the cutoff radius.

---

## Advantages

- Simple implementation.
- Fast computation.
- Widely used in early crystal GNN models.
- Easy to understand.

---

## Disadvantages

The same cutoff may not work for every material.

For example,

```text
Very Small Cutoff

↓

Missing Neighbors
```

or

```text
Very Large Cutoff

↓

Too Many Edges
```

Choosing an appropriate cutoff often requires domain knowledge.

---

# 13.14.4 Visualizing the Cutoff Radius

Imagine placing a sphere around one atom.

```text
            ○

       ○         ○

          \     /

            ●

          /     \

       ○         ○
```

The central atom is shown as

```text
●
```

Every atom inside the sphere becomes a neighbor.

Atoms outside the sphere are ignored.

---

# 13.14.5 Method 2 — k-Nearest Neighbors (kNN)

Instead of using a fixed distance,

another approach is to select the nearest **k** atoms.

For example,

```text
k = 6
```

means

```text
Choose

↓

Six Closest Atoms
```

No matter how dense or sparse the crystal is,

every atom receives exactly six neighbors.

---

### Simplified Example

Suppose the distances are

```text
1.8 Å

2.0 Å

2.2 Å

2.4 Å

2.6 Å

2.8 Å

3.3 Å

3.5 Å
```

If

```text
k = 4
```

only the first four atoms are selected.

---

## Advantages

- Every node has the same number of neighbors.
- Useful for deep learning.
- Produces graphs with predictable size.

---

## Disadvantages

Some selected neighbors may be physically unrealistic,

especially in low-density materials.

Conversely,

important chemical neighbors may be excluded if they lie just beyond the chosen value of **k**.

---

# 13.14.6 Method 3 — MinimumDistanceNN

One of the neighbor-finding classes provided by **pymatgen** is

`MinimumDistanceNN`.

Instead of selecting neighbors using a manually chosen cutoff,

it estimates neighbors based on atomic distances.

---

### Code Example

```python
from pymatgen.analysis.local_env import MinimumDistanceNN

nn = MinimumDistanceNN()

info = nn.get_nn_info(

    structure,

    0

)

print(len(info))
```

Each element in

```python
info
```

contains information about one neighboring atom.

---

### Inspecting the Results

```python
for neighbor in info:

    print(

        neighbor["site"].specie,

        neighbor["site"].coords

    )
```

The returned dictionary contains

- neighboring site,
- image information,
- and additional metadata.

---

# 13.14.7 Method 4 — Voronoi Neighbors

A more sophisticated approach is based on **Voronoi tessellation**.

Instead of using only distance,

space around each atom is divided into polyhedra.

Two atoms become neighbors if their Voronoi cells share a face.

Conceptually,

```text
Crystal

↓

Voronoi Cells

↓

Shared Face

↓

Neighbor
```

This method adapts naturally to different crystal structures.

---

### Code Example

```python
from pymatgen.analysis.local_env import VoronoiNN

nn = VoronoiNN()

neighbors = nn.get_nn_info(

    structure,

    0

)
```

The workflow is almost identical to the previous example.

Only the neighbor-finding algorithm changes.

---

## Advantages

- Geometry-aware.
- Works well for many crystal structures.
- Frequently used in structural analysis.

---

## Limitations

For highly distorted structures,

Voronoi cells may become sensitive to small geometric changes.

---

# 13.14.8 Method 5 — CrystalNN

One of the most powerful neighbor-finding algorithms available in **pymatgen** is **CrystalNN**.

Unlike simple distance-based methods,

CrystalNN combines

- interatomic distance,
- Voronoi geometry,
- chemical information,
- coordination environment.

As a result,

it often produces chemically meaningful neighbor lists.

---

### Code Example

```python
from pymatgen.analysis.local_env import CrystalNN

cnn = CrystalNN()

neighbors = cnn.get_nn_info(

    structure,

    0

)
```

Each neighbor returned contains

- neighboring atom,
- image,
- weight,
- and coordination information.

---

### Inspecting Neighbor Weights

```python
for neighbor in neighbors:

    print(

        neighbor["site"].specie,

        neighbor["weight"]

    )
```

Unlike cutoff methods,

CrystalNN assigns confidence weights to neighboring atoms.

These weights can sometimes be incorporated into graph construction.

---

## Why CrystalNN Is Popular

CrystalNN was specifically designed for crystalline materials.

It performs well across

- ionic compounds,
- covalent solids,
- metallic systems,
- mixed bonding environments.

For this reason,

many materials science workflows use CrystalNN during preprocessing.

---

# 13.14.9 Method 6 — EconNN

Another available algorithm is **EconNN**,

which estimates coordination environments using effective coordination numbers.

Although less common in Graph Neural Networks,

it is useful for analyzing coordination chemistry.

---

### Code Example

```python
from pymatgen.analysis.local_env import EconNN

nn = EconNN()

neighbors = nn.get_nn_info(

    structure,

    0
)
```

The interface remains nearly identical,

making it easy to experiment with different neighbor definitions.

---

# 13.14.10 Comparing Neighbor-Finding Algorithms

Each method has strengths and weaknesses.

| Method | Main Idea | Advantages | Limitations |
|---------|-----------|------------|-------------|
| Fixed Cutoff | Distance threshold | Simple, fast | Sensitive to cutoff choice |
| kNN | Fixed number of neighbors | Uniform graph size | May ignore chemistry |
| MinimumDistanceNN | Distance-based heuristic | Automatic | Less chemistry-aware |
| VoronoiNN | Shared Voronoi faces | Geometry-aware | Sensitive to distortions |
| CrystalNN | Distance + chemistry + geometry | Highly accurate | More computationally expensive |
| EconNN | Effective coordination | Useful for coordination analysis | Less common in GNNs |

No single algorithm is best for every application.

The choice depends on

- material type,
- dataset size,
- computational resources,
- Graph Neural Network architecture.

---

# 13.14.11 Which Algorithms Are Used in Research?

Different Graph Neural Networks often employ different graph construction strategies.

For example,

| Model | Typical Neighbor Strategy |
|-------|---------------------------|
| Early GCN models | Fixed cutoff radius |
| CGCNN | Fixed cutoff with distance expansion |
| MEGNet | Distance-based neighbor graph |
| M3GNet | Radius graph with learned edge features |
| ALIGNN | Radius graph plus line graph construction |

Although the architectures differ,

every model begins with the same fundamental task:

constructing a graph from a crystal structure.

---

# 13.14.12 Comparing Neighbor Counts

Suppose we analyze the same crystal using three different algorithms.

```python
cutoff_neighbors = structure.get_neighbors(

    structure[0],

    r=3.0

)

print(

    "Cutoff:",

    len(cutoff_neighbors)

)
```

```python
cnn = CrystalNN()

print(

    "CrystalNN:",

    len(

        cnn.get_nn_info(

            structure,

            0

        )

    )

)
```

Running several neighbor-finding methods on the same structure is an excellent way to understand how graph construction changes.

---

# 13.14.13 Practical Guidelines

When building crystal graphs,

the following recommendations are useful.

- Begin with a physically reasonable cutoff radius.
- Visualize several neighbor lists before processing an entire dataset.
- Check coordination numbers against known crystal structures.
- Use the same neighbor-finding strategy for the entire dataset.
- Record preprocessing parameters for reproducibility.
- Avoid changing neighbor definitions midway through a project.

Small preprocessing changes can produce noticeably different Graph Neural Networks.

---

# 13.14.14 Materials Science Perspective

Neighbor finding is not merely a programming task.

It represents a scientific assumption about

> **Which atoms interact strongly enough to influence material properties?**

Different neighbor definitions produce different graphs,

and different graphs may lead to different learned representations.

Therefore,

graph construction should always be considered part of the scientific modeling process,

not just data preprocessing.

---

# 13.14.15 Summary

Neighbor finding is one of the most critical steps in constructing crystal graphs for Graph Neural Networks.

We explored several widely used algorithms, including fixed cutoff radius, k-nearest neighbors, MinimumDistanceNN, VoronoiNN, CrystalNN, and EconNN, and learned how each can be implemented using **pymatgen**.

Understanding the strengths and limitations of these algorithms allows researchers to construct graph representations that more faithfully capture the underlying chemistry and physics of crystalline materials.

With node features, edge features, and neighbor-finding methods now in place, we have all the components needed to build complete crystal graphs.

In the next section, we will combine everything we have learned and implement a **complete crystal graph construction pipeline** starting from a real POSCAR or CIF file and ending with a ready-to-train PyTorch Geometric graph object.

# 13.15 Complete Crystal Graph Construction Pipeline

So far, we have studied each component of a crystal graph individually.

We learned how to

- read crystal structures,
- generate node features,
- identify neighboring atoms,
- construct edge indices,
- compute edge features.

While understanding these individual components is important,

real materials informatics research requires combining them into a single preprocessing pipeline.

In practice, researchers rarely construct graphs manually.

Instead, they build an automated pipeline capable of converting thousands—or even millions—of crystal structures into graph objects suitable for Graph Neural Networks.

This section develops such a pipeline step by step.

By the end of this section, you will understand how a raw crystal structure becomes a graph that can be directly used for training a Graph Neural Network.

---

# 13.15.1 The Complete Workflow

Every crystal follows the same sequence of preprocessing steps.

```text
POSCAR / CIF

↓

Read Crystal Structure

↓

Extract Atomic Features

↓

Find Neighboring Atoms

↓

Generate Edge Features

↓

Construct Graph

↓

PyTorch Geometric Data Object

↓

Graph Neural Network
```

This workflow forms the backbone of almost every modern crystal Graph Neural Network.

---

# 13.15.2 Reading the Crystal Structure

The first step is loading the crystal.

```python
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
```

Let us inspect some basic information.

```python
print(structure.formula)

print(len(structure))
```

Example output

```text
Si8

8
```

The structure contains

- chemical composition,
- lattice vectors,
- atomic coordinates,
- periodic boundary conditions.

---

# 13.15.3 Inspecting the Crystal

Before constructing the graph,

it is always good practice to inspect the structure.

```python
print(structure)
```

or

```python
for site in structure:

    print(

        site.specie,

        site.frac_coords

    )
```

Example

```text
Si

[0.0, 0.0, 0.0]
```

```text
Si

[0.25, 0.25, 0.25]
```

Checking the structure before preprocessing helps detect corrupted or incorrectly formatted files.

---

# 13.15.4 Building the Node Feature Matrix

We now convert every atom into a numerical feature vector.

```python
from pymatgen.core import Element

node_features = []

for site in structure:

    element = Element(

        site.specie.symbol

    )

    features = [

        element.Z,

        float(element.atomic_mass),

        element.X,

        element.group,

        element.row

    ]

    node_features.append(features)
```

The resulting list contains one feature vector per atom.

Convert it into a tensor.

```python
import torch

x = torch.tensor(

    node_features,

    dtype=torch.float

)
```

Inspect its dimensions.

```python
print(x.shape)
```

Example

```text
torch.Size([8,5])
```

Meaning

```text
8 Atoms

×

5 Features
```

---

# 13.15.5 Finding Neighboring Atoms

Next,

we determine which atoms are connected.

For this example,

we use a cutoff radius.

```python
cutoff = 3.0
```

We initialize empty lists.

```python
edge_index = []

edge_attr = []
```

Loop over every atom.

```python
for i, site in enumerate(structure):

    neighbors = structure.get_neighbors(

        site,

        r=cutoff

    )

    print(

        f"Atom {i}",

        len(neighbors)

    )
```

This allows us to verify that neighboring atoms are being identified correctly.

---

# 13.15.6 Constructing Graph Connectivity

For every neighboring atom,

we store the connection.

```python
for i, site in enumerate(structure):

    neighbors = structure.get_neighbors(

        site,

        r=cutoff

    )

    for neighbor in neighbors:

        edge_index.append(

            [

                i,

                neighbor.index

            ]

        )
```

Conceptually,

```text
Atom

↓

Neighbor

↓

Edge
```

Repeated over the entire crystal,

this produces the graph connectivity.

---

# 13.15.7 Computing Edge Features

For every edge,

we also compute the bond distance.

```python
for i, site in enumerate(structure):

    neighbors = structure.get_neighbors(

        site,

        r=cutoff

    )

    for neighbor in neighbors:

        edge_attr.append(

            [

                neighbor.nn_distance

            ]

        )
```

Each edge now stores

```text
Bond Distance
```

as its feature.

---

# 13.15.8 Converting to Tensors

Convert both lists into tensors.

```python
edge_index = torch.tensor(

    edge_index,

    dtype=torch.long

).t().contiguous()

edge_attr = torch.tensor(

    edge_attr,

    dtype=torch.float
)
```

Inspect their dimensions.

```python
print(edge_index.shape)

print(edge_attr.shape)
```

Typical output

```text
torch.Size([2,64])

torch.Size([64,1])
```

This means

- 64 edges,
- one feature per edge.

---

# 13.15.9 Creating the Graph Object

Now we combine everything into one graph.

```python
from torch_geometric.data import Data

graph = Data(

    x=x,

    edge_index=edge_index,

    edge_attr=edge_attr

)
```

The graph now contains

- node features,
- graph connectivity,
- edge features.

---

# 13.15.10 Adding Target Properties

Supervised learning also requires labels.

Suppose we want to predict

```text
Band Gap
```

The graph becomes

```python
band_gap = 1.12

graph.y = torch.tensor(

    [band_gap],

    dtype=torch.float

)
```

Similarly,

formation energy could be stored as

```python
formation_energy = -5.34

graph.y = torch.tensor(

    [

        formation_energy

    ],

    dtype=torch.float

)
```

Only the target property changes.

The graph construction process remains identical.

---

# 13.15.11 Inspecting the Finished Graph

PyTorch Geometric provides a convenient summary.

```python
print(graph)
```

Example

```text
Data(

    x=[8,5],

    edge_index=[2,64],

    edge_attr=[64,1],

    y=[1]

)
```

From a single line,

we immediately know

- number of atoms,
- number of edges,
- node feature dimension,
- edge feature dimension,
- target dimension.

---

# 13.15.12 Packaging Everything into a Function

In real research,

we rarely write preprocessing code repeatedly.

Instead,

we place the entire pipeline inside a reusable function.

```python
def structure_to_graph(

    structure,

    cutoff=3.0

):

    ...

    return graph
```

The function can then be called repeatedly.

```python
graph = structure_to_graph(

    structure
)
```

This modular design makes preprocessing cleaner, easier to test, and reusable across projects.

---

# 13.15.13 Processing an Entire Dataset

Suppose a folder contains hundreds of crystal structures.

```text
Dataset

├── Material_001

├── Material_002

├── Material_003

└── ...
```

We simply repeat the graph construction function.

```python
dataset = []

for file in files:

    structure = Structure.from_file(

        file

    )

    graph = structure_to_graph(

        structure

    )

    dataset.append(graph)
```

After the loop,

the dataset consists entirely of graph objects.

---

# 13.15.14 Saving Graphs

Graph construction can be computationally expensive.

Rather than rebuilding graphs every time,

researchers often save the processed dataset.

Using PyTorch,

this is straightforward.

```python
torch.save(

    dataset,

    "graphs.pt"

)
```

The saved dataset can later be loaded.

```python
dataset = torch.load(

    "graphs.pt"
)
```

This avoids repeating preprocessing and speeds up experimentation.

---

# 13.15.15 A More Complete Research Pipeline

The preprocessing workflow in a research project is often more extensive.

```text
Quantum ESPRESSO

↓

Relaxed Structure

↓

POSCAR

↓

pymatgen

↓

Crystal Graph

↓

Dataset

↓

DataLoader

↓

Graph Neural Network

↓

Training

↓

Evaluation

↓

Predicted Property
```

Every stage contributes to the final model performance.

Errors introduced during preprocessing can propagate throughout the entire machine learning workflow.

---

# 13.15.16 Common Debugging Checks

Before training a Graph Neural Network,

verify that the graph has been constructed correctly.

Useful checks include

```python
print(graph.x.shape)

print(graph.edge_index.shape)

print(graph.edge_attr.shape)

print(graph.y)
```

Also inspect

```python
print(graph.num_nodes)

print(graph.num_edges)
```

Unexpected values often indicate

- incorrect cutoff radius,
- missing neighbors,
- duplicated edges,
- improperly generated feature vectors.

Debugging preprocessing is often easier than debugging the neural network itself.

---

# 13.15.17 Materials Science Perspective

Graph construction is not merely a programming exercise.

It is the stage where physical knowledge is translated into a machine-readable representation.

Every design choice—

- node features,
- neighbor-finding strategy,
- edge features,
- target properties—

reflects assumptions about the underlying materials system.

Well-designed graph representations allow Graph Neural Networks to learn meaningful physical relationships rather than simply memorizing data.

---

# 13.15.18 Summary

In this section, we assembled all the preprocessing steps into a complete crystal graph construction pipeline.

Starting from a POSCAR or CIF file, we extracted atomic features, identified neighboring atoms, computed edge features, constructed graph connectivity, created a PyTorch Geometric `Data` object, and prepared the graph for supervised learning.

This pipeline represents the standard workflow used in many modern materials informatics studies and forms the foundation for implementing advanced crystal Graph Neural Networks.

Although we can now construct crystal graphs successfully, understanding their internal structure is equally important.

In the next section, we will learn how to **visualize crystal graphs**, inspect node and edge connectivity, and verify that the generated graphs accurately represent the underlying crystal structures before training a Graph Neural Network.

# 13.16 Visualizing Crystal Graphs

Constructing a crystal graph is only the first step.

Before training a Graph Neural Network, it is good scientific practice to **verify that the graph accurately represents the underlying crystal structure**.

Visualization plays a crucial role in this process.

By visualizing the graph, we can answer questions such as

- Are the correct atoms connected?
- Are any atoms isolated?
- Are there too many neighbors?
- Is the cutoff radius reasonable?
- Does the graph resemble the expected crystal structure?

In research, graph visualization is frequently used during preprocessing to detect mistakes before expensive model training begins.

---

# 13.16.1 Why Visualize Crystal Graphs?

Imagine constructing a graph containing several thousand edges.

Without visualization,

it is difficult to determine whether the graph is correct.

Consider two situations.

Correct graph

```text
Si —— Si

 \    /

   Si
```

Incorrect graph

```text
Si

     Si —— Si
```

The second graph contains an isolated atom.

Although the code may execute without errors,

the graph no longer represents the actual crystal.

Visualization helps detect these problems immediately.

---

# 13.16.2 Crystal Structure vs Crystal Graph

It is important to distinguish between

the crystal structure

and

its graph representation.

```text
Crystal Structure

↓

Atomic Coordinates

↓

Neighbor Search

↓

Crystal Graph
```

The crystal structure contains

- lattice vectors,
- atomic coordinates,
- periodicity.

The graph contains only

- nodes,
- edges,
- node features,
- edge features.

The graph is therefore an abstract representation of the crystal.

---

# 13.16.3 Visualizing Graph Connectivity

Suppose we have four atoms.

```text
Atom 0

↓

Atom 1
```

```text
Atom 0

↓

Atom 2
```

```text
Atom 1

↓

Atom 3
```

The graph becomes

```text
      2

      |

      |

0 ---- 1

      |

      |

      3
```

This representation ignores the exact atomic positions.

Instead,

it focuses on connectivity.

---

# 13.16.4 Converting a PyTorch Geometric Graph into a NetworkX Graph

PyTorch Geometric provides a convenient conversion utility.

```python
from torch_geometric.utils import to_networkx
```

Suppose we already have

```python
graph
```

We can convert it into a NetworkX graph.

```python
network = to_networkx(

    graph,

    to_undirected=True

)
```

The resulting graph can be visualized using standard NetworkX tools.

---

# 13.16.5 Plotting the Graph

NetworkX works well with Matplotlib.

```python
import networkx as nx
import matplotlib.pyplot as plt
```

A simple visualization can be created using

```python
nx.draw(

    network,

    with_labels=True

)

plt.show()
```

The labels correspond to node indices,

allowing us to verify graph connectivity.

---

# 13.16.6 Improving the Layout

Different layouts emphasize different structural properties.

For example,

the spring layout places connected nodes closer together.

```python
position = nx.spring_layout(

    network,

    seed=42

)

nx.draw(

    network,

    position,

    with_labels=True

)

plt.show()
```

Although this layout is not physically accurate,

it often makes graph connectivity much easier to understand.

---

# 13.16.7 Inspecting Individual Edges

Sometimes,

we want to inspect the neighbors of a specific atom.

NetworkX provides convenient methods.

```python
print(

    list(

        network.neighbors(0)

    )

)
```

Example output

```text
[1, 2, 5, 7]
```

This tells us that

Atom 0

is connected to

Atoms

1,

2,

5,

and

7.

---

# 13.16.8 Counting Nodes and Edges

Several useful properties can be inspected directly.

```python
print(

    network.number_of_nodes()

)

print(

    network.number_of_edges()

)
```

Example

```text
64

192
```

These values should agree with

```python
graph.num_nodes

graph.num_edges
```

Checking consistency between different representations helps identify preprocessing errors.

---

# 13.16.9 Visualizing Node Features

Node colors can represent atomic species.

Suppose our crystal contains

- silicon,
- oxygen,
- aluminum.

First,

generate a color list.

```python
colors = []

for site in structure:

    if site.specie.symbol == "Si":

        colors.append("blue")

    elif site.specie.symbol == "O":

        colors.append("red")

    else:

        colors.append("gray")
```

Then draw the graph.

```python
nx.draw(

    network,

    node_color=colors,

    with_labels=True

)

plt.show()
```

Although simplified,

this visualization immediately distinguishes different atomic species.

---

# 13.16.10 Labeling Nodes with Chemical Symbols

Instead of displaying node indices,

we can display element symbols.

```python
labels = {}

for i, site in enumerate(structure):

    labels[i] = site.specie.symbol
```

Draw the graph.

```python
nx.draw(

    network,

    labels=labels,

    with_labels=True

)

plt.show()
```

Example

```text
Si —— Si

 \    /

   O
```

This representation is often easier to interpret than numerical indices.

---

# 13.16.11 Visualizing Neighbor Relationships

Sometimes,

we want to inspect only the neighbors of one atom.

```python
atom = 0

neighbors = structure.get_neighbors(

    structure[atom],

    r=3.0
)

for neighbor in neighbors:

    print(

        neighbor.index,

        neighbor.nn_distance

    )
```

Example output

```text
1   2.35

2   2.35

5   2.35

7   2.35
```

Comparing these values with the graph helps verify that graph edges correspond to actual neighboring atoms.

---

# 13.16.12 Comparing Different Cutoff Radii

Visualization is particularly useful when selecting a cutoff radius.

Suppose we try

```text
2.0 Å
```

The graph may become

```text
Disconnected
```

Increasing the cutoff to

```text
3.0 Å
```

may produce

```text
Well Connected
```

Choosing

```text
6.0 Å
```

could generate

```text
Too Many Edges
```

Visual inspection helps determine whether the graph is chemically reasonable.

---

# 13.16.13 Visualizing the Original Crystal

Graph visualization should always be complemented by visualization of the original crystal.

For example,

using **pymatgen**, we can inspect the structure.

```python
print(structure)
```

For interactive visualization,

researchers often export the structure to external visualization software such as

- VESTA,
- OVITO,
- CrystalMaker.

Comparing the crystal with its graph provides confidence that preprocessing has been performed correctly.

---

# 13.16.14 Debugging Graph Construction

Graph visualization is one of the fastest debugging tools available.

Common issues include

- isolated atoms,
- duplicated edges,
- disconnected components,
- incorrect neighbor assignments,
- missing atomic species,
- unexpected graph density.

Whenever model performance is unexpectedly poor,

one of the first steps should be to inspect several crystal graphs visually.

Many preprocessing mistakes become obvious immediately after visualization.

---

# 13.16.15 Materials Science Perspective

Unlike social or citation networks,

crystal graphs are derived from well-defined physical structures.

This provides an important advantage.

Whenever the generated graph appears inconsistent with known crystal chemistry,

the graph construction pipeline should be questioned before modifying the neural network architecture.

In many research projects,

careful graph preprocessing contributes more to model performance than increasing network depth or adding additional graph convolution layers.

---

# 13.16.16 Summary

Visualization provides an essential bridge between crystal structures and graph representations.

By converting PyTorch Geometric graphs into NetworkX objects, inspecting node connectivity, labeling atomic species, and comparing graph topology with the original crystal structure, researchers can verify that preprocessing has been performed correctly.

Graph visualization is therefore not simply a presentation tool—it is an important part of scientific validation and debugging in materials informatics.

In the next section, we will put everything together by building a **complete end-to-end mini project** that starts from a real crystal structure, constructs a graph, feeds it into a Graph Neural Network, and predicts a materials property.

# 13.17 Mini Project: Predicting the Band Gap of Silicon Using a Graph Neural Network

Throughout this chapter, we have studied the individual components of Graph Neural Networks for crystalline materials.

We learned how to

- read crystal structures,
- generate node features,
- construct edge indices,
- compute edge features,
- build graph datasets,
- train simple Graph Neural Networks.

In this section, we combine everything into one complete workflow.

The objective is not to obtain a highly accurate predictive model.

Instead, the goal is to understand how every preprocessing and deep learning component fits together in a realistic materials informatics project.

By the end of this section, you will have implemented a complete Graph Neural Network pipeline starting from a crystal structure and ending with a predicted material property.

---

# 13.17.1 Project Overview

Our project consists of the following steps.

```text
Silicon POSCAR

↓

Read Crystal Structure

↓

Generate Node Features

↓

Find Neighbors

↓

Build Graph

↓

Graph Neural Network

↓

Predict Band Gap
```

Although we use silicon as an example,

the same workflow applies to

- semiconductors,
- metals,
- oxides,
- battery materials,
- catalysts,
- superconductors.

---

# 13.17.2 Required Libraries

First, import the required libraries.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymatgen.core import Structure
from pymatgen.core import Element

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
```

These libraries provide everything required for graph construction and neural network training.

---

# 13.17.3 Loading the Crystal Structure

Load the silicon crystal.

```python
structure = Structure.from_file(

    "POSCAR"

)
```

Inspect the structure.

```python
print(structure.formula)

print(len(structure))
```

Example

```text
Si8

8
```

The structure contains

eight silicon atoms.

---

# 13.17.4 Constructing Node Features

Each silicon atom is converted into a feature vector.

```python
node_features = []

for site in structure:

    atom = Element(

        site.specie.symbol

    )

    node_features.append(

        [

            atom.Z,

            float(atom.atomic_mass),

            atom.X,

            atom.group,

            atom.row

        ]

    )
```

Convert the list into a tensor.

```python
x = torch.tensor(

    node_features,

    dtype=torch.float

)
```

Inspect the dimensions.

```python
print(x.shape)
```

Output

```text
torch.Size([8,5])
```

---

# 13.17.5 Constructing the Graph Edges

Choose a cutoff radius.

```python
cutoff = 3.0
```

Initialize empty lists.

```python
edge_index = []

edge_attr = []
```

Loop over all atoms.

```python
for i, site in enumerate(structure):

    neighbors = structure.get_neighbors(

        site,

        r=cutoff

    )

    for neighbor in neighbors:

        edge_index.append(

            [

                i,

                neighbor.index

            ]

        )

        edge_attr.append(

            [

                neighbor.nn_distance

            ]

        )
```

Convert both lists into tensors.

```python
edge_index = torch.tensor(

    edge_index,

    dtype=torch.long

).t().contiguous()

edge_attr = torch.tensor(

    edge_attr,

    dtype=torch.float

)
```

Inspect the shapes.

```python
print(edge_index.shape)

print(edge_attr.shape)
```

---

# 13.17.6 Creating the Graph Object

Suppose the experimental band gap of silicon is approximately

```text
1.12 eV
```

Create the graph.

```python
graph = Data(

    x=x,

    edge_index=edge_index,

    edge_attr=edge_attr,

    y=torch.tensor(

        [1.12],

        dtype=torch.float

    )

)
```

The graph now contains

- node features,
- graph connectivity,
- edge features,
- target property.

---

# 13.17.7 Building the Graph Neural Network

Define a simple model.

```python
class SiliconGNN(

    nn.Module

):

    def __init__(

        self,

        input_dim,

        hidden_dim

    ):

        super().__init__()

        self.conv1 = GCNConv(

            input_dim,

            hidden_dim

        )

        self.conv2 = GCNConv(

            hidden_dim,

            hidden_dim

        )

        self.fc = nn.Linear(

            hidden_dim,

            1

        )
```

---

### Forward Propagation

```python
    def forward(

        self,

        data

    ):

        x = self.conv1(

            data.x,

            data.edge_index

        )

        x = F.relu(x)

        x = self.conv2(

            x,

            data.edge_index

        )

        x = F.relu(x)

        batch = torch.zeros(

            x.size(0),

            dtype=torch.long

        )

        x = global_mean_pool(

            x,

            batch

        )

        return self.fc(x)
```

Since our project contains only one graph,

the batch vector consists entirely of zeros.

---

# 13.17.8 Initializing the Model

Create the model.

```python
model = SiliconGNN(

    input_dim=5,

    hidden_dim=64

)
```

Choose the optimizer.

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001

)
```

Choose the loss function.

```python
criterion = nn.MSELoss()
```

---

# 13.17.9 Training the Model

Train for several epochs.

```python
for epoch in range(100):

    optimizer.zero_grad()

    prediction = model(graph)

    loss = criterion(

        prediction,

        graph.y

    )

    loss.backward()

    optimizer.step()

    print(

        epoch,

        loss.item()

    )
```

Every iteration performs

```text
Forward Propagation

↓

Loss

↓

Backpropagation

↓

Gradient Descent

↓

Updated Parameters
```

---

# 13.17.10 Making a Prediction

After training,

switch to evaluation mode.

```python
model.eval()

with torch.no_grad():

    prediction = model(graph)

print(

    prediction.item()

)
```

Example output

```text
1.08
```

The prediction should become progressively closer to

```text
1.12 eV
```

during training.

---

# 13.17.11 Understanding the Learning Process

Although this example contains only one crystal,

the Graph Neural Network still performs the same operations used in large research datasets.

```text
Node Features

↓

Graph Convolutions

↓

Node Embeddings

↓

Graph Pooling

↓

Prediction
```

The only difference is that real datasets contain thousands of crystal graphs instead of one.

---

# 13.17.12 Extending to Multiple Materials

A real materials informatics project would contain many graphs.

Instead of

```text
Silicon
```

we might have

```text
Silicon

Gallium Arsenide

Titanium Dioxide

Lithium Iron Phosphate

Graphite

...
```

The workflow becomes

```text
Crystal Dataset

↓

Graph Dataset

↓

DataLoader

↓

Training

↓

Prediction
```

The code changes very little.

Instead of training on one graph,

we iterate over a `DataLoader`.

---

### Code Fragment

```python
for batch in train_loader:

    prediction = model(

        batch

    )

    loss = criterion(

        prediction,

        batch.y

    )

    loss.backward()

    optimizer.step()
```

This is the standard training procedure used in modern Graph Neural Networks.

---

# 13.17.13 Possible Improvements

The model developed here is intentionally simple.

Many improvements are possible.

For example,

- use richer node features,
- include Gaussian-expanded edge features,
- add more graph convolution layers,
- increase the hidden embedding size,
- introduce dropout,
- normalize node features,
- perform hyperparameter optimization,
- train on thousands of materials rather than a single crystal.

Each improvement can potentially increase predictive performance.

---

# 13.17.14 Relation to Real Research

Although simplified,

this mini project follows the same sequence of operations found in many published Graph Neural Network studies.

Researchers typically begin with

```text
Materials Project

↓

Crystal Structures

↓

Graph Construction

↓

Training Dataset

↓

Graph Neural Network

↓

Property Prediction
```

The major difference is scale.

Instead of one silicon crystal,

research datasets may contain

- 20,000,
- 100,000,
- or even millions

of crystal structures.

The preprocessing pipeline, however, remains fundamentally the same.

---

# 13.17.15 Key Lessons

This project demonstrates that a complete Graph Neural Network workflow consists of several interconnected stages.

```text
Read Crystal

↓

Construct Graph

↓

Generate Features

↓

Build Dataset

↓

Train GNN

↓

Predict Material Property
```

Every stage is essential.

Errors introduced during preprocessing can significantly affect the final prediction accuracy.

Conversely,

carefully designed graph representations often improve performance more than simply increasing model complexity.

---

# 13.17.16 Summary

In this mini project, we developed a complete end-to-end Graph Neural Network workflow for crystalline materials.

Starting from a silicon crystal structure, we generated node features, constructed graph connectivity, computed edge features, created a PyTorch Geometric graph object, defined a Graph Neural Network, trained the model, and predicted the band gap.

Although the example is intentionally simple, it captures the complete workflow used in modern graph-based materials informatics.

Before moving to specialized crystal Graph Neural Networks such as CGCNN, it is important to recognize that building a correct graph is only part of the challenge.

Real-world projects often encounter issues such as disconnected graphs, incorrect neighbor selection, tensor shape mismatches, and GPU-related errors.

The next section focuses on these practical challenges and presents common debugging strategies for Graph Neural Network development in computational materials science.

# Supplement to Chapter 13

# Practical PyTorch Geometric for Materials Informatics

---

# S13.1 Why This Supplement Exists

By the end of Chapter 13, we have learned the theoretical foundations of Graph Neural Networks for crystalline materials. We know how to

- represent crystals as graphs,
- generate node features,
- generate edge features,
- construct graph connectivity,
- create PyTorch Geometric `Data` objects,
- train a simple Graph Neural Network.

These examples are intentionally small because their purpose is to explain the underlying concepts.

However, real materials informatics research is very different.

Instead of processing a single silicon crystal, researchers often work with datasets containing

- thousands,
- tens of thousands,
- or even millions

of crystal structures.

Managing datasets of this scale introduces many practical challenges that are rarely discussed in research papers.

For example,

- How should graph datasets be organized?
- How can thousands of CIF files be converted automatically into graphs?
- How should graphs be loaded efficiently during training?
- How can models be saved and resumed?
- How should GPU resources be used?
- How can training be monitored and debugged?

These questions are not specific to Graph Neural Networks.

Rather,

they concern the engineering practices required to transform a proof-of-concept model into a robust research workflow.

This supplement focuses entirely on these practical implementation skills.

It bridges the gap between educational examples and research-quality software development.

---

# S13.1.1 From Educational Code to Research Code

Educational examples are intentionally simple.

For example,

our earlier workflow consisted of

```text
Read One Crystal

↓

Construct One Graph

↓

Train One Model
```

Although useful for learning,

this workflow cannot scale to modern materials databases.

A research workflow looks very different.

```text
Thousands of Crystal Structures

↓

Automatic Graph Construction

↓

Dataset Management

↓

Mini-Batch Training

↓

Model Evaluation

↓

Checkpoint Saving

↓

Property Prediction
```

Notice that most of the workflow concerns data management rather than neural network design.

---

# S13.1.2 Why Published Papers Rarely Show This Code

Many students are surprised after reading research papers.

The paper may describe

- the neural network,
- the dataset,
- the optimization algorithm,

yet omit thousands of lines of preprocessing code.

The reason is simple.

Scientific papers focus on

> **the scientific contribution**

rather than

> **software engineering details.**

As a result,

important implementation details such as

- graph preprocessing,
- dataset classes,
- batching,
- checkpoint management,
- GPU training,

are often omitted completely.

Learning these practical skills therefore requires studying software libraries and open-source implementations in addition to reading research papers.

---

# S13.1.3 The Goal of This Supplement

The objective of this supplement is to teach the practical skills required to build Graph Neural Network projects for computational materials science.

By the end of this supplement,

you should be able to

- organize a professional project,
- automatically construct graph datasets,
- train models efficiently,
- save and resume experiments,
- debug graph preprocessing,
- evaluate trained models,
- perform inference on new crystal structures.

These skills are essential for reproducing modern materials informatics research.

---

# S13.1.4 Learning Philosophy

Throughout this supplement,

we will continue following the same philosophy used throughout this book.

Every concept will include

- an explanation,
- small code fragments,
- complete implementations,
- and their application to materials science.

Rather than treating programming as a separate subject,

we view code as another language for expressing scientific ideas.

Every line of code should correspond to a meaningful operation in the materials informatics workflow.

---

# S13.1.5 How This Supplement Fits into the Book

The relationship between the previous chapter and this supplement can be summarized as

```text
Chapter 13

↓

Graph Theory

↓

Crystal Graph Construction

↓

Basic Graph Neural Networks

↓

Supplement

↓

Research-Quality Implementation

↓

Next Chapter

↓

Crystal Graph Convolutional Neural Networks (CGCNN)
```

The main chapter teaches **what** Graph Neural Networks are.

This supplement teaches **how** researchers implement them in practice.

Together,

they provide the foundation needed for studying specialized crystal Graph Neural Network architectures in the following chapters.

---

# S13.1.6 Summary

Chapter 13 introduced the concepts underlying Graph Neural Networks for crystalline materials.

This supplement extends those concepts by focusing on practical implementation.

The goal is not simply to write code that works.

Instead,

the objective is to write code that is

- modular,
- reusable,
- scalable,
- reproducible,
- and suitable for research.

These engineering practices form the backbone of modern computational materials science and will be used repeatedly throughout the remaining chapters of this book.

# S13.2 Organizing a Real Materials Informatics Project

As Graph Neural Network projects grow in complexity, writing all code in a single Python script quickly becomes unmanageable.

For example, a small educational example might consist of only one file.

```text
main.py
```

This approach works for simple demonstrations.

However, real materials informatics projects often involve

- thousands of crystal structures,
- multiple Graph Neural Network models,
- data preprocessing,
- model evaluation,
- visualization,
- checkpoint management,
- experiment tracking.

Keeping everything inside one file becomes difficult to maintain and debug.

Professional research software is therefore organized into multiple modules, each with a well-defined responsibility.

---

# S13.2.1 Why Project Organization Matters

Good project organization provides several advantages.

It

- improves readability,
- reduces duplicated code,
- simplifies debugging,
- encourages code reuse,
- supports collaboration,
- makes experiments reproducible.

Imagine returning to a project six months after publishing a paper.

A well-organized project allows you to understand and reproduce your work much more easily than a single large script.

---

# S13.2.2 A Typical Research Project Structure

A common directory structure for Graph Neural Network research is shown below.

```text
MaterialsGNN/

│

├── data/

│   ├── raw/

│   ├── processed/

│   └── external/

│

├── models/

│

├── training/

│

├── utils/

│

├── checkpoints/

│

├── results/

│

├── notebooks/

│

├── configs/

│

├── main.py

│

└── requirements.txt
```

Each directory serves a different purpose.

Instead of placing every file together,

related functionality is grouped into dedicated modules.

---

# S13.2.3 The data Directory

The `data` directory stores every dataset used in the project.

A common organization is

```text
data/

├── raw/

├── processed/

└── external/
```

The three folders have different roles.

---

## raw

The `raw` directory contains the original crystal structures.

Examples include

```text
raw/

├── Si.cif

├── LiFePO4.cif

├── POSCAR_001

├── POSCAR_002
```

These files should never be modified.

They represent the original input data.

---

## processed

The `processed` directory stores graph objects generated from the raw crystal structures.

For example,

```text
processed/

├── graph_0001.pt

├── graph_0002.pt

├── graph_0003.pt
```

These files are generated automatically during preprocessing.

Saving processed graphs avoids reconstructing them every time the model is trained.

---

## external

Sometimes datasets are downloaded from external sources.

Examples include

- Materials Project
- OQMD
- AFLOW
- NOMAD

Downloaded files are often stored inside

```text
external/
```

before being cleaned and processed.

---

# S13.2.4 The models Directory

The `models` directory contains Graph Neural Network architectures.

Example

```text
models/

├── gcn.py

├── cgcnn.py

├── megnet.py

├── alignn.py
```

Each model is implemented in a separate file.

For example,

```python
from models.cgcnn import CGCNN
```

Keeping model definitions separate from training code improves modularity.

---

# S13.2.5 The training Directory

Training scripts are placed inside

```text
training/
```

For example,

```text
training/

├── train.py

├── validate.py

├── test.py
```

Instead of mixing

- preprocessing,
- model definition,
- training,

each stage is implemented independently.

---

# S13.2.6 The utils Directory

Many functions are reused throughout a project.

Examples include

- graph construction,
- feature generation,
- evaluation metrics,
- visualization,
- logging.

Rather than copying code repeatedly,

these functions are placed inside

```text
utils/
```

Example

```text
utils/

├── graph_utils.py

├── metrics.py

├── visualization.py

├── seed.py
```

Then,

they can be imported whenever needed.

```python
from utils.metrics import mae
```

This approach greatly reduces duplicated code.

---

# S13.2.7 The checkpoints Directory

Training large Graph Neural Networks may require

- hours,
- days,
- or even weeks.

If training stops unexpectedly,

all progress could be lost.

To prevent this,

trained model parameters are periodically saved.

```text
checkpoints/

├── epoch10.pt

├── epoch20.pt

├── best_model.pt
```

The model can later resume training from one of these checkpoints.

---

# S13.2.8 The results Directory

Experimental outputs are usually stored separately.

For example,

```text
results/

├── loss_curve.png

├── predictions.csv

├── parity_plot.png

├── metrics.txt
```

Separating results from source code keeps the project organized.

---

# S13.2.9 The notebooks Directory

Jupyter notebooks are useful for

- exploratory analysis,
- quick experiments,
- visualization,
- debugging.

Example

```text
notebooks/

├── data_analysis.ipynb

├── graph_visualization.ipynb

├── feature_engineering.ipynb
```

Once an idea is validated,

the implementation should usually be transferred into reusable Python modules.

---

# S13.2.10 The configs Directory

As projects become larger,

many hyperparameters must be managed.

Instead of hardcoding them,

they can be stored in configuration files.

Example

```text
configs/

├── train.yaml

├── cgcnn.yaml

├── experiment1.yaml
```

A configuration file might specify

```text
Learning Rate

↓

0.001
```

```text
Batch Size

↓

32
```

```text
Epochs

↓

200
```

Using configuration files makes experiments easier to reproduce.

---

# S13.2.11 The Main Script

The project usually contains one entry point.

```text
main.py
```

Instead of implementing every operation,

its job is simply to coordinate the workflow.

Conceptually,

```text
Load Configuration

↓

Load Dataset

↓

Create Model

↓

Train

↓

Evaluate

↓

Save Results
```

The actual implementations remain inside dedicated modules.

---

# S13.2.12 Example Import Structure

Once the project is organized,

the main script becomes much cleaner.

```python
from models.cgcnn import CGCNN

from training.train import train

from training.validate import validate

from utils.metrics import mae

from dataset import CrystalDataset
```

Notice that each module has a single responsibility.

This design follows good software engineering practices.

---

# S13.2.13 Recommended Workflow

A typical research workflow now becomes

```text
Raw Crystal Structures

↓

Preprocessing

↓

Processed Graph Dataset

↓

Training

↓

Validation

↓

Testing

↓

Prediction

↓

Analysis

↓

Publication
```

Because each stage is implemented independently,

the workflow is easy to modify.

For example,

a different Graph Neural Network architecture can be introduced without changing the preprocessing code.

---

# S13.2.14 Materials Science Perspective

Large-scale materials informatics projects often involve collaborations between

- computational materials scientists,
- machine learning researchers,
- software engineers.

A well-organized project structure allows each researcher to work on a specific component without interfering with the rest of the codebase.

Moreover,

many published Graph Neural Network repositories follow a similar organization,

making it easier to understand and extend existing research software.

---

# S13.2.15 Summary

A well-organized project structure is an essential part of research-quality Graph Neural Network development.

Separating datasets, preprocessing, model architectures, training routines, utilities, checkpoints, and experimental results makes the code easier to understand, maintain, and reproduce.

Although this organization may seem unnecessary for small examples, it becomes indispensable when working with large crystal datasets and complex Graph Neural Network architectures.

In the next section, we will implement one of the most important components of a research workflow by creating a **custom PyTorch Geometric dataset** capable of automatically loading and preprocessing thousands of crystal structures.

# S13.3 Creating a Custom PyTorch Geometric Dataset

In the previous section, we organized our materials informatics project into multiple directories.

The next step is to automate dataset creation.

So far in this book, we have manually created graph objects.

For example,

```python
graph = Data(

    x=x,

    edge_index=edge_index,

    edge_attr=edge_attr,

    y=target

)
```

This approach is useful for learning,

but it becomes impractical when working with large datasets.

Imagine processing

- 5,000 crystal structures,
- 20,000 Materials Project entries,
- or 100,000 OQMD structures.

Manually constructing every graph would be impossible.

PyTorch Geometric solves this problem using **Dataset** classes.

Instead of manually creating graph objects,

we create a dataset object that automatically

- loads structures,
- converts them into graphs,
- stores the processed graphs,
- and returns them whenever they are needed during training.

This automation forms the foundation of every large-scale Graph Neural Network project.

---

# S13.3.1 Why Use a Dataset Class?

Suppose we have a folder containing thousands of crystal structures.

```text
raw/

├── material_001.cif

├── material_002.cif

├── material_003.cif

...

├── material_5000.cif
```

Without a dataset class,

we would repeatedly write code like

```python
structure = Structure.from_file(file)

graph = structure_to_graph(structure)

dataset.append(graph)
```

inside every training script.

This duplicates code and makes maintenance difficult.

Instead,

we encapsulate the entire preprocessing pipeline inside a reusable dataset class.

Conceptually,

```text
Raw Crystal Files

↓

Dataset Class

↓

Automatic Graph Construction

↓

Graph Dataset
```

---

# S13.3.2 Dataset vs InMemoryDataset

PyTorch Geometric provides two commonly used base classes.

```text
Dataset
```

and

```text
InMemoryDataset
```

The difference lies in how graphs are stored.

### Dataset

Graphs are loaded from disk whenever they are needed.

```text
Disk

↓

Load One Graph

↓

Training
```

Advantages

- low memory usage,
- suitable for extremely large datasets.

Disadvantages

- slower training because graphs must be loaded repeatedly.

---

### InMemoryDataset

Graphs are loaded once and stored in RAM.

```text
Disk

↓

Load All Graphs

↓

Memory

↓

Training
```

Advantages

- much faster training,
- efficient for medium-sized datasets.

Disadvantages

- requires more system memory.

For many materials informatics projects containing several thousand crystal structures,

`InMemoryDataset` is an excellent choice.

---

# S13.3.3 Importing the Required Libraries

Begin by importing the necessary modules.

```python
from torch_geometric.data import InMemoryDataset

from torch_geometric.data import Data

import torch

from pymatgen.core import Structure
```

These modules provide

- dataset management,
- graph representation,
- tensor operations,
- crystal structure parsing.

---

# S13.3.4 Creating the Dataset Class

A custom dataset is created by inheriting from `InMemoryDataset`.

```python
class CrystalDataset(

    InMemoryDataset

):

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

        self.load(

            self.processed_paths[0]

        )
```

This class will eventually manage every graph in our dataset.

---

# S13.3.5 Understanding the Constructor

The constructor initializes the dataset.

```python
def __init__(

    self,

    root,

    transform=None,

    pre_transform=None

):
```

The arguments specify

- where the dataset is stored,
- optional graph transformations,
- optional preprocessing operations.

The command

```python
self.load(

    self.processed_paths[0]

)
```

loads the processed dataset if it already exists.

If no processed dataset is available,

PyTorch Geometric automatically calls the preprocessing routine.

---

# S13.3.6 Defining Raw Files

The dataset must know which raw files exist.

This is achieved using

```python
@property

def raw_file_names(

    self

):

    return [

        "material_001.cif",

        "material_002.cif"

    ]
```

In practice,

this list can be generated automatically by scanning the directory.

For example,

```python
from pathlib import Path

files = list(

    Path(

        self.raw_dir

    ).glob("*.cif")
)
```

This approach automatically detects every CIF file inside the dataset.

---

# S13.3.7 Defining Processed Files

After graph construction,

PyTorch Geometric stores the processed dataset.

We specify the output filename.

```python
@property

def processed_file_names(

    self

):

    return [

        "graphs.pt"

    ]
```

This file contains all processed graph objects.

Future training sessions can load it directly,

avoiding repeated preprocessing.

---

# S13.3.8 The Download Method

Some datasets are downloaded automatically.

PyTorch Geometric therefore defines

```python
def download(

    self

):

    pass
```

For most materials informatics projects,

the crystal structures are already available locally,

so this method often remains empty.

If desired,

it could later be extended to download structures from online repositories.

---

# S13.3.9 The Process Method

The most important function is

```python
process()
```

This method converts raw crystal structures into graph objects.

Conceptually,

```text
Raw Crystal

↓

Read Structure

↓

Construct Graph

↓

Save Graph

↓

Repeat
```

The processing routine is executed only once.

---

# S13.3.10 Reading Every Crystal Structure

Inside the processing loop,

each crystal is loaded individually.

```python
for file in self.raw_paths:

    structure = Structure.from_file(

        file

    )
```

Each iteration loads one crystal structure.

---

# S13.3.11 Converting Structures into Graphs

Rather than rewriting preprocessing code,

we call a reusable function.

```python
graph = structure_to_graph(

    structure
)
```

Notice how modular programming simplifies the implementation.

The dataset class does not need to know

how graph construction works.

It only needs to call the appropriate function.

---

# S13.3.12 Collecting Graphs

Create an empty list.

```python
data_list = []
```

After constructing each graph,

append it.

```python
data_list.append(

    graph
)
```

After processing every crystal,

the list contains the complete graph dataset.

---

# S13.3.13 Saving the Dataset

Finally,

save the processed graphs.

```python
self.save(

    data_list,

    self.processed_paths[0]

)
```

The next time the dataset is loaded,

PyTorch Geometric reads the processed graphs directly,

eliminating the need for repeated graph construction.

---

# S13.3.14 Complete Dataset Skeleton

Combining the previous sections,

a simplified dataset class becomes

```python
class CrystalDataset(

    InMemoryDataset

):

    @property
    def raw_file_names(self):

        ...

    @property
    def processed_file_names(self):

        ...

    def download(self):

        pass

    def process(self):

        data_list = []

        for file in self.raw_paths:

            structure = Structure.from_file(

                file

            )

            graph = structure_to_graph(

                structure

            )

            data_list.append(

                graph

            )

        self.save(

            data_list,

            self.processed_paths[0]

        )
```

Although simplified,

this illustrates the overall structure used in many Graph Neural Network projects.

---

# S13.3.15 Loading the Dataset

Using the custom dataset is remarkably simple.

```python
dataset = CrystalDataset(

    root="data"
)
```

Inspect its size.

```python
print(

    len(dataset)

)
```

Example

```text
5000
```

Access an individual graph.

```python
graph = dataset[0]

print(graph)
```

Example

```text
Data(

    x=[64,5],

    edge_index=[2,192],

    edge_attr=[192,1],

    y=[1]
)
```

The dataset behaves like a standard Python collection,

making it easy to integrate with training pipelines.

---

# S13.3.16 Materials Science Perspective

Modern crystal Graph Neural Network studies often involve preprocessing tens of thousands of crystal structures.

Using a custom dataset class provides several important advantages.

It

- standardizes preprocessing,
- ensures every crystal is converted consistently,
- avoids duplicated code,
- speeds up subsequent experiments,
- supports reproducible research.

Once the dataset class has been implemented,

future projects require only changing the graph construction function or target property,

while the overall data management workflow remains unchanged.

---

# S13.3.17 Summary

Custom dataset classes are one of the most powerful features of PyTorch Geometric.

By encapsulating graph construction inside an `InMemoryDataset`, we can automatically convert thousands of crystal structures into graph objects while keeping preprocessing code modular and reusable.

This approach is widely used in modern materials informatics because it separates graph construction from model training, resulting in cleaner, more maintainable, and more reproducible research software.

In the next section, we will extend this idea by **automatically processing entire directories of crystal structures**, including robust error handling, progress monitoring, and efficient preprocessing workflows suitable for large materials databases.

# S13.4 Processing Thousands of Crystal Structures Automatically

The custom dataset developed in the previous section provides an organized framework for managing graph datasets.

However, one important task still remains.

We must convert **thousands of crystal structures** into graph objects automatically.

For small examples,

manually loading a few crystal structures is sufficient.

For example,

```python
structure = Structure.from_file(

    "Si.cif"

)

graph = structure_to_graph(

    structure

)
```

Unfortunately,

this approach does not scale.

Modern materials informatics datasets often contain

- 5,000 structures,
- 20,000 structures,
- 100,000 structures,
- or even millions of crystal structures.

Clearly,

manual preprocessing is impossible.

Instead,

we need an automated preprocessing pipeline.

---

# S13.4.1 The Automated Workflow

The preprocessing workflow can be summarized as

```text
Folder of Crystal Structures

↓

Read One Structure

↓

Construct Graph

↓

Save Graph

↓

Next Structure

↓

Repeat Until Finished
```

This process is performed only once.

After preprocessing,

all future experiments use the saved graph dataset.

---

# S13.4.2 Organizing the Raw Dataset

Suppose our project contains the following directory.

```text
data/

└── raw/

    ├── Si.cif

    ├── GaAs.cif

    ├── TiO2.cif

    ├── LiFePO4.cif

    ├── Graphite.cif

    └── ...
```

Each file represents one crystal structure.

Our goal is to convert every file into a graph automatically.

---

# S13.4.3 Finding All Crystal Files

Python provides several methods for locating files.

One convenient approach uses the `pathlib` module.

```python
from pathlib import Path

raw_dir = Path(

    "data/raw"

)

files = list(

    raw_dir.glob("*.cif")

)
```

The variable

```python
files
```

contains every CIF file inside the directory.

Inspect the result.

```python
print(

    len(files)

)
```

Example

```text
5348
```

The preprocessing pipeline now knows exactly how many crystal structures must be processed.

---

# S13.4.4 Processing Every Crystal

We simply iterate over the list.

```python
for file in files:

    structure = Structure.from_file(

        file

    )

    graph = structure_to_graph(

        structure

    )
```

Conceptually,

```text
Crystal 1

↓

Graph 1
```

```text
Crystal 2

↓

Graph 2
```

```text
Crystal 3

↓

Graph 3
```

Eventually,

every crystal becomes a graph.

---

# S13.4.5 Building the Dataset

Create an empty list.

```python
graphs = []
```

Append every graph.

```python
for file in files:

    structure = Structure.from_file(

        file

    )

    graph = structure_to_graph(

        structure

    )

    graphs.append(

        graph

    )
```

After the loop,

```python
graphs
```

contains the complete dataset.

---

# S13.4.6 Monitoring Progress

Large datasets may require

- several minutes,
- several hours,
- or even days

to preprocess.

Printing progress is therefore useful.

```python
for i, file in enumerate(files):

    print(

        f"Processing {i+1}/{len(files)}"

    )
```

Example output

```text
Processing 1/5348

Processing 2/5348

Processing 3/5348

...
```

This simple addition allows us to estimate preprocessing progress.

---

# S13.4.7 Using tqdm Progress Bars

A more professional solution uses the `tqdm` library.

```python
from tqdm import tqdm
```

Replace the loop.

```python
for file in tqdm(files):

    structure = Structure.from_file(

        file

    )

    graph = structure_to_graph(

        structure

    )
```

The terminal now displays a progress bar similar to

```text
███████████████ 72%
```

This makes long preprocessing jobs much easier to monitor.

---

# S13.4.8 Handling Corrupted Crystal Files

Real datasets are rarely perfect.

Occasionally,

a crystal file may

- be corrupted,
- contain missing atoms,
- have invalid lattice vectors,
- fail to load.

If preprocessing stops after the first error,

hours of computation may be lost.

---

## A Better Strategy

Wrap preprocessing inside a

```python
try
```

block.

```python
graphs = []

for file in files:

    try:

        structure = Structure.from_file(

            file

        )

        graph = structure_to_graph(

            structure

        )

        graphs.append(

            graph

        )

    except Exception:

        print(

            "Skipping",

            file

        )
```

Instead of crashing,

the pipeline skips problematic structures and continues processing.

---

# S13.4.9 Recording Failed Structures

Simply skipping errors may not be sufficient.

Researchers often keep a record of failed files.

```python
failed = []
```

During preprocessing,

store problematic filenames.

```python
except Exception:

    failed.append(

        file

    )
```

After preprocessing,

inspect them.

```python
print(

    failed

)
```

This makes it easy to investigate problematic structures later.

---

# S13.4.10 Counting Successful Graphs

At the end of preprocessing,

summarize the results.

```python
print(

    len(graphs)

)
```

Example

```text
5294
```

Suppose

54 structures failed.

Then

```python
print(

    len(failed)

)
```

might produce

```text
54
```

Keeping these statistics is good research practice.

---

# S13.4.11 Saving the Graph Dataset

After preprocessing,

save the entire dataset.

```python
torch.save(

    graphs,

    "graphs.pt"

)
```

Later,

the dataset can be restored instantly.

```python
graphs = torch.load(

    "graphs.pt"

)
```

This avoids repeating expensive preprocessing.

---

# S13.4.12 Saving Individual Graphs

Some researchers prefer storing graphs separately.

For example,

```python
torch.save(

    graph,

    f"graph_{i}.pt"

)
```

This produces

```text
processed/

├── graph_0001.pt

├── graph_0002.pt

├── graph_0003.pt
```

Individual graph files are useful when datasets become extremely large.

---

# S13.4.13 Logging Preprocessing Information

Keeping a preprocessing log improves reproducibility.

For example,

record

- preprocessing date,
- cutoff radius,
- node features,
- edge features,
- neighbor-finding algorithm,
- number of successful graphs,
- number of failed structures.

Conceptually,

```text
Preprocessing Log

↓

Parameters

↓

Dataset Statistics

↓

Saved with Project
```

Such records make future experiments easier to reproduce.

---

# S13.4.14 Improving Performance

Graph construction is usually CPU-intensive.

Several strategies can improve preprocessing speed.

- Use multiple CPU cores.
- Avoid repeatedly loading identical structures.
- Save processed graphs immediately.
- Minimize unnecessary computations.
- Profile preprocessing bottlenecks.

Although optimization is important,

correct graph construction should always take priority over speed.

---

# S13.4.15 Complete Automated Pipeline

The complete preprocessing workflow now becomes

```text
Crystal Dataset

↓

Locate All Files

↓

Load Crystal

↓

Generate Node Features

↓

Find Neighbors

↓

Generate Edge Features

↓

Construct Graph

↓

Store Graph

↓

Repeat

↓

Save Processed Dataset
```

This workflow is nearly identical to those used in many published Graph Neural Network studies.

---

# S13.4.16 Materials Science Perspective

Automated preprocessing is indispensable in modern computational materials science.

Databases such as

- Materials Project,
- OQMD,
- AFLOW,
- NOMAD,

contain tens or hundreds of thousands of crystal structures.

Without automated graph construction,

training Graph Neural Networks on datasets of this scale would be impractical.

Consequently,

robust preprocessing pipelines are considered an essential component of research-quality materials informatics software.

---

# S13.4.17 Summary

In this section, we developed an automated preprocessing pipeline capable of converting thousands of crystal structures into graph objects.

We learned how to locate crystal files, iterate through entire datasets, monitor preprocessing progress, handle corrupted structures gracefully, record failed files, save processed graphs, and maintain reproducible preprocessing logs.

These techniques transform graph construction from a manual task into an efficient and scalable workflow suitable for large materials databases.

In the next section, we will learn how to **load these graph datasets efficiently during training** using PyTorch Geometric's `DataLoader`, enabling mini-batch Graph Neural Network training on thousands of crystal graphs.

# S13.5 Building DataLoaders for Crystal Graph Datasets

In the previous section, we automatically converted thousands of crystal structures into graph objects.

However, another challenge now arises.

Suppose our dataset contains

```text
50,000 Crystal Graphs
```

Should we train the Graph Neural Network using all 50,000 graphs at once?

The answer is

> **No.**

Loading an entire dataset into the neural network simultaneously would require an enormous amount of memory and would be computationally inefficient.

Instead,

modern deep learning frameworks divide the dataset into **mini-batches**.

PyTorch Geometric provides the **DataLoader** class to perform this task automatically.

The DataLoader is one of the most important components of every Graph Neural Network training pipeline.

---

# S13.5.1 What Is a DataLoader?

A DataLoader is an object that automatically

- reads graph datasets,
- groups graphs into mini-batches,
- optionally shuffles the data,
- feeds batches into the Graph Neural Network.

Conceptually,

```text
Graph Dataset

↓

DataLoader

↓

Mini-Batches

↓

Graph Neural Network
```

Rather than processing one graph at a time,

the model receives many graphs together.

---

# S13.5.2 Why Mini-Batches Are Necessary

Suppose we have

```text
10,000 Crystal Graphs
```

Training all graphs simultaneously would require

- large amounts of RAM,
- large GPU memory,
- long computation times.

Instead,

the dataset is divided into smaller groups.

For example,

```text
Graph Dataset

↓

Batch 1

↓

32 Graphs
```

```text
Graph Dataset

↓

Batch 2

↓

32 Graphs
```

```text
Graph Dataset

↓

Batch 3

↓

32 Graphs
```

This continues until the entire dataset has been processed.

Mini-batches make training both memory-efficient and computationally efficient.

---

# S13.5.3 Importing the DataLoader

PyTorch Geometric provides its own DataLoader implementation.

Import it as follows.

```python
from torch_geometric.loader import DataLoader
```

This DataLoader understands graph data,

which is different from ordinary tensors.

---

# S13.5.4 Creating a DataLoader

Suppose we have already created

```python
dataset = CrystalDataset(
    root="data"
)
```

Creating a DataLoader requires only one line.

```python
train_loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=True

)
```

This object automatically generates batches during training.

---

# S13.5.5 Understanding the Parameters

The constructor contains three important arguments.

```python
DataLoader(

    dataset,

    batch_size=32,

    shuffle=True

)
```

### dataset

Specifies the graph dataset.

```python
dataset
```

---

### batch_size

Determines how many graphs are processed together.

For example,

```text
Batch Size

↓

32 Graphs
```

Other common values include

- 16
- 32
- 64
- 128

The optimal batch size depends on

- GPU memory,
- graph size,
- model complexity.

---

### shuffle

When

```python
shuffle=True
```

the graph order changes every epoch.

Conceptually,

Epoch 1

```text
Graph 5

Graph 2

Graph 17

...
```

Epoch 2

```text
Graph 12

Graph 8

Graph 1

...
```

Random shuffling generally improves training by preventing the model from learning unwanted ordering patterns.

---

# S13.5.6 Inspecting the DataLoader

The DataLoader behaves like an iterator.

```python
for batch in train_loader:

    print(batch)

    break
```

Example output

```text
DataBatch(

    x=[2150,5],

    edge_index=[2,10384],

    edge_attr=[10384,1],

    batch=[2150],

    y=[32]
)
```

Notice that the DataLoader no longer returns a single graph.

Instead,

it returns one large graph representing an entire mini-batch.

---

# S13.5.7 How Graphs Are Combined

Suppose we have three crystal graphs.

```text
Graph A

↓

5 Atoms
```

```text
Graph B

↓

8 Atoms
```

```text
Graph C

↓

6 Atoms
```

The DataLoader combines them into

```text
One Large Graph

↓

19 Nodes

↓

Combined Edge List

↓

Batch Vector
```

Internally,

PyTorch Geometric keeps track of which nodes belong to each original graph.

This allows multiple graphs to be processed simultaneously.

---

# S13.5.8 Iterating Through Batches

Training proceeds batch by batch.

```python
for batch in train_loader:

    prediction = model(

        batch

    )
```

Instead of

```text
One Graph

↓

Prediction
```

we now have

```text
32 Graphs

↓

32 Predictions
```

This significantly improves GPU utilization.

---

# S13.5.9 Understanding the Number of Batches

Suppose

```text
Dataset Size

↓

1000 Graphs
```

and

```text
Batch Size

↓

32
```

The number of batches is approximately

```text
1000 ÷ 32

↓

32 Batches
```

The final batch may contain fewer graphs if the dataset size is not exactly divisible by the batch size.

---

# S13.5.10 Accessing Individual Batches

Each batch contains multiple tensors.

For example,

```python
print(batch.x.shape)

print(batch.edge_index.shape)

print(batch.edge_attr.shape)

print(batch.y.shape)
```

Typical output

```text
torch.Size([2200,5])

torch.Size([2,10480])

torch.Size([10480,1])

torch.Size([32])
```

Notice that

the node count depends on the combined size of all graphs in the mini-batch.

---

# S13.5.11 Separating Training, Validation, and Test Sets

Before creating DataLoaders,

the dataset is usually divided into

- training,
- validation,
- testing.

Conceptually,

```text
Complete Dataset

↓

Training Set

↓

Validation Set

↓

Test Set
```

Each subset receives its own DataLoader.

```python
train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True

)

val_loader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False

)

test_loader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False

)
```

Notice that

validation and test datasets are generally **not shuffled**.

---

# S13.5.12 Choosing an Appropriate Batch Size

There is no universally optimal batch size.

Small batches

```text
8–16 Graphs
```

Advantages

- lower memory usage,
- noisier gradients,
- sometimes better generalization.

Large batches

```text
64–256 Graphs
```

Advantages

- faster GPU computation,
- smoother gradients,
- fewer optimizer updates.

The best choice depends on

- available hardware,
- graph complexity,
- model architecture.

Experimentation is often necessary.

---

# S13.5.13 Monitoring Batch Processing

During training,

it is often useful to inspect progress.

```python
for i, batch in enumerate(train_loader):

    print(

        f"Batch {i+1}"

    )
```

Example

```text
Batch 1

Batch 2

Batch 3
```

This simple diagnostic can help identify slow batches or preprocessing issues.

---

# S13.5.14 Materials Science Perspective

Graph batching is especially important in materials informatics because crystal structures vary significantly in size.

For example,

one graph may represent

```text
12 Atoms
```

while another contains

```text
180 Atoms
```

Unlike conventional machine learning datasets,

where every sample often has the same dimensions,

graph datasets naturally contain variable-sized structures.

PyTorch Geometric's DataLoader automatically handles these differences,

allowing crystals with different numbers of atoms and bonds to be trained together in the same mini-batch.

This capability is one of the major advantages of graph-based deep learning frameworks.

---

# S13.5.15 Summary

The DataLoader is responsible for efficiently feeding graph datasets into a Graph Neural Network.

Rather than loading one graph at a time, it combines multiple graphs into mini-batches, manages data shuffling, and supports efficient GPU training.

Understanding the DataLoader is essential because nearly every Graph Neural Network training loop relies on it.

However, one part of the DataLoader output often confuses new users:

```python
batch
```

This tensor determines which nodes belong to which original graph after batching.

In the next section, we will study the **batch vector** in detail and understand how PyTorch Geometric keeps multiple crystal graphs separate while processing them together.

# S13.6 Understanding the `batch` Vector in PyTorch Geometric

In the previous section, we learned how the `DataLoader` combines multiple crystal graphs into a single mini-batch.

However, this raises an important question.

If every graph is merged into one large graph,

**how does PyTorch Geometric know which atoms belong to which crystal?**

The answer lies in one of the most important tensors produced by the `DataLoader`:

```python
batch
```

Understanding the `batch` vector is essential because almost every graph-level prediction model uses it during pooling.

Fortunately,

the concept is much simpler than it first appears.

---

# S13.6.1 Inspecting a Mini-Batch

Suppose we already have

```python
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)
```

Retrieve one mini-batch.

```python
batch = next(iter(train_loader))
```

Now inspect it.

```python
print(batch)
```

Example output

```text
DataBatch(
    x=[148,5],
    edge_index=[2,624],
    edge_attr=[624,1],
    y=[4],
    batch=[148]
)
```

Notice the additional tensor

```text
batch=[148]
```

This tensor did not exist when we worked with individual graphs.

---

# S13.6.2 What Does the Batch Vector Represent?

Suppose our mini-batch contains four crystal structures.

```text
Graph 0

↓

25 atoms
```

```text
Graph 1

↓

40 atoms
```

```text
Graph 2

↓

33 atoms
```

```text
Graph 3

↓

50 atoms
```

PyTorch Geometric merges them into

```text
25

+

40

+

33

+

50

=

148 nodes
```

Instead of storing four separate graphs,

it stores one larger graph.

The `batch` vector records which crystal every atom originally belonged to.

---

# S13.6.3 A Small Example

Suppose our batch contains only three crystal structures.

```text
Crystal A

↓

3 atoms
```

```text
Crystal B

↓

2 atoms
```

```text
Crystal C

↓

4 atoms
```

The combined graph contains

```text
9 nodes
```

The batch vector becomes

```python
tensor([

0,0,0,

1,1,

2,2,2,2

])
```

Interpretation

Node

```text
0
```

belongs to

```text
Crystal 0
```

Node

```text
1
```

belongs to

```text
Crystal 0
```

Node

```text
3
```

belongs to

```text
Crystal 1
```

Node

```text
8
```

belongs to

```text
Crystal 2
```

Nothing more complicated than that.

---

# S13.6.4 Visualizing the Batch Vector

Conceptually,

```text
Merged Nodes

↓

0

1

2

3

4

5

6

7

8
```

Batch tensor

```text
0

0

0

1

1

2

2

2

2
```

Every integer identifies one crystal graph.

---

# S13.6.5 Inspecting the Batch Tensor

Simply print it.

```python
print(batch.batch)
```

Example

```text
tensor(

[0,0,0,...,1,1,...,2,2,...]

)
```

Determine its size.

```python
print(batch.batch.shape)
```

Output

```text
torch.Size([148])
```

The length always equals the total number of nodes.

---

# S13.6.6 How Many Graphs Are in the Batch?

One useful debugging trick is

```python
print(

batch.batch.max()+1

)
```

Example

```text
4
```

The maximum graph index plus one equals the number of graphs in the mini-batch.

---

# S13.6.7 Counting Nodes Per Crystal

Suppose we want to know how many atoms belong to each crystal.

PyTorch provides

```python
import torch

counts = torch.bincount(

batch.batch

)

print(counts)
```

Example output

```text
tensor([25,40,33,50])
```

This confirms that

- Crystal 0 has 25 atoms.
- Crystal 1 has 40 atoms.
- Crystal 2 has 33 atoms.
- Crystal 3 has 50 atoms.

This is an excellent debugging technique.

---

# S13.6.8 Why Pooling Needs the Batch Vector

Recall that node embeddings are computed during message passing.

Suppose the GNN produces

```text
148 Node Embeddings
```

However,

our task is to predict

```text
Formation Energy
```

There should be only

```text
4 Predictions
```

One prediction for each crystal.

The model therefore needs to know

which nodes belong together.

That information comes directly from

```python
batch
```

---

# S13.6.9 Global Mean Pooling

A common operation is

```python
from torch_geometric.nn import global_mean_pool
```

Pooling is performed as

```python
graph_embedding = global_mean_pool(

node_embeddings,

batch.batch

)
```

Notice that

```python
batch.batch
```

is passed explicitly.

Without it,

the pooling layer would not know where one crystal ends and the next begins.

---

# S13.6.10 Shape Before and After Pooling

Suppose

```text
148 Nodes

↓

64 Features
```

The node embedding matrix has shape

```text
(148,64)
```

After pooling,

we obtain

```text
4 Graph Embeddings
```

with shape

```text
(4,64)
```

The GNN has transformed

many atom-level representations

into

one crystal-level representation per graph.

---

# S13.6.11 A Complete Example

```python
batch = next(iter(train_loader))

print(batch.x.shape)

print(batch.batch.shape)

node_embeddings = model.conv1(

batch.x,

batch.edge_index

)

print(node_embeddings.shape)

graph_embeddings = global_mean_pool(

node_embeddings,

batch.batch

)

print(graph_embeddings.shape)
```

Possible output

```text
torch.Size([148,5])

torch.Size([148])

torch.Size([148,64])

torch.Size([4,64])
```

This is exactly what we expect.

---

# S13.6.12 Common Beginner Mistake

A frequent error is forgetting to pass the batch vector.

Incorrect

```python
graph_embedding = global_mean_pool(

node_embeddings

)
```

Correct

```python
graph_embedding = global_mean_pool(

node_embeddings,

batch.batch

)
```

Always remember

the pooling layer requires graph membership information.

---

# S13.6.13 Debugging Tip

When debugging a new Graph Neural Network,

inspect the batch tensor.

```python
print(batch.batch)

print(batch.batch.shape)

print(torch.bincount(batch.batch))
```

If the number of nodes per graph looks unreasonable,

there may be an error in

- graph construction,
- batching,
- preprocessing.

Checking the batch vector early can save hours of debugging later.

---

# S13.6.14 Materials Science Example

Suppose our mini-batch contains

```text
Silicon

↓

64 atoms
```

```text
LiFePO₄

↓

112 atoms
```

```text
SrTiO₃

↓

40 atoms
```

The merged graph contains

```text
216 nodes
```

The batch vector tells the pooling layer

which atoms belong to

- silicon,
- LiFePO₄,
- SrTiO₃.

After pooling,

the network produces

```text
Formation Energy (Si)

Formation Energy (LiFePO₄)

Formation Energy (SrTiO₃)
```

rather than

216 atom-level predictions.

---

# Research Tip

Whenever you build a new crystal graph dataset, inspect the `batch` tensor before training. If the number of nodes assigned to each graph does not match the corresponding crystal structures, investigate your graph construction or batching pipeline first. Many training failures that appear to be model problems are actually caused by incorrectly batched graphs.

---

# S13.6.15 Summary

The `batch` vector is one of the most important tensors generated by the PyTorch Geometric `DataLoader`.

It records the graph membership of every node after multiple crystal graphs have been merged into a single mini-batch. This information enables pooling layers to transform atom-level embeddings into crystal-level representations, making graph-level property prediction possible.

In the next section, we will use everything developed so far—datasets, DataLoaders, batching, pooling, and Graph Neural Networks—to implement a complete **training loop** for predicting material properties.

# S13.7 Building the Complete Training Loop for Crystal Graph Neural Networks

Up to this point, we have developed every major component required for training a Graph Neural Network.

We have learned how to

- represent crystal structures as graphs,
- construct graph datasets,
- create mini-batches using the `DataLoader`,
- understand the `batch` vector,
- and prepare graph-level targets.

The final step is to connect these components into a complete training loop.

This training loop is the heart of every Graph Neural Network.

Whether predicting

- formation energy,
- band gap,
- bulk modulus,
- elastic constants,
- dielectric constant,
- magnetic moment,

the overall workflow remains nearly identical.

---

# S13.7.1 The Complete Training Pipeline

A typical training iteration follows the sequence

```text
Crystal Graph Batch

↓

Graph Neural Network

↓

Predicted Property

↓

Loss Function

↓

Backpropagation

↓

Optimizer

↓

Updated Parameters
```

Every epoch repeats this process until the model converges.

---

# S13.7.2 Preparing the Model

Suppose we have already implemented a Graph Neural Network.

```python
model = CrystalGNN(
    node_features=5,
    hidden_channels=64
)
```

Move the model to the selected device.

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

model = model.to(device)
```

Always move the model to the same device as the data.

---

# S13.7.3 Choosing the Loss Function

Formation energy prediction is a regression problem.

A common choice is Mean Squared Error.

```python
criterion = torch.nn.MSELoss()
```

For band gap prediction,

the same loss is often used.

```python
criterion = torch.nn.MSELoss()
```

If the target property has significant outliers,

Mean Absolute Error may be preferable.

```python
criterion = torch.nn.L1Loss()
```

The choice of loss function depends on the scientific problem.

---

# S13.7.4 Creating the Optimizer

The optimizer updates the neural network parameters.

A common choice is Adam.

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3

)
```

Notice

```python
model.parameters()
```

Every trainable weight inside the Graph Neural Network is passed to the optimizer.

---

# S13.7.5 Switching to Training Mode

Before training,

activate training mode.

```python
model.train()
```

This is important because some layers behave differently during training and evaluation.

For example,

- Dropout randomly disables neurons during training.
- Batch Normalization updates running statistics.

Always call

```python
model.train()
```

before the training loop.

---

# S13.7.6 Iterating Over Mini-Batches

Training begins by looping through the DataLoader.

```python
for batch in train_loader:

    ...
```

Each iteration returns one mini-batch of crystal graphs.

Move the batch to the same device.

```python
batch = batch.to(device)
```

If the model is on the GPU,

the data must also be on the GPU.

---

# S13.7.7 Resetting the Gradients

Before computing a new gradient,

clear the old gradients.

```python
optimizer.zero_grad()
```

Why?

PyTorch accumulates gradients by default.

Without resetting,

gradients from previous mini-batches would continue accumulating,

leading to incorrect parameter updates.

---

# S13.7.8 Forward Propagation

Generate predictions.

```python
prediction = model(batch)
```

The input

```python
batch
```

contains

- node features,
- edge indices,
- edge features,
- batch vector.

The output might represent

```text
Predicted Formation Energy
```

for every crystal in the mini-batch.

---

# S13.7.9 Inspecting the Predictions

Always verify tensor dimensions.

```python
print(prediction.shape)
```

Example

```text
torch.Size([32])
```

If

```python
batch.y.shape
```

is also

```text
torch.Size([32])
```

then the prediction dimensions are correct.

Checking tensor shapes early prevents many debugging problems.

---

# S13.7.10 Computing the Loss

Compare predictions with experimental or DFT targets.

```python
loss = criterion(

    prediction,

    batch.y

)
```

Conceptually,

```text
Prediction

↓

Loss Function

↓

Training Error
```

Lower loss indicates better predictions.

---

# S13.7.11 Inspecting the Loss

During development,

print the loss.

```python
print(

    loss.item()

)
```

Example

```text
0.284
```

The

```python
.item()
```

method converts a PyTorch tensor into a Python number.

---

# S13.7.12 Backpropagation

Compute gradients.

```python
loss.backward()
```

This single line performs backpropagation throughout the entire Graph Neural Network.

Internally,

PyTorch computes

```text
∂Loss

──────

∂Weight
```

for every trainable parameter.

Fortunately,

automatic differentiation performs these calculations automatically.

---

# S13.7.13 Updating the Parameters

After computing gradients,

update the parameters.

```python
optimizer.step()
```

Conceptually,

```text
Current Parameters

↓

Gradient

↓

Optimizer

↓

Updated Parameters
```

This is the learning step.

---

# S13.7.14 One Complete Mini-Batch

Combining everything,

one training iteration becomes

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

Although short,

this code performs

- forward propagation,
- loss computation,
- gradient computation,
- parameter optimization.

Nearly every deep learning project follows this pattern.

---

# S13.7.15 Training for Multiple Epochs

Training requires repeating the mini-batch loop many times.

```python
epochs = 100

for epoch in range(epochs):

    model.train()

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
```

Each epoch processes every graph in the training dataset once.

---

# S13.7.16 Tracking the Average Loss

Rather than printing every mini-batch loss,

compute the epoch average.

```python
epoch_loss = 0.0

for batch in train_loader:

    ...

    epoch_loss += loss.item()
```

Compute the mean.

```python
epoch_loss /= len(train_loader)
```

Display the result.

```python
print(

    f"Epoch Loss: {epoch_loss:.4f}"

)
```

Example

```text
Epoch Loss: 0.1264
```

Tracking average loss makes training progress much easier to interpret.

---

# S13.7.17 Watching the Model Learn

Suppose training produces

```text
Epoch 1

Loss = 1.842
```

```text
Epoch 10

Loss = 0.621
```

```text
Epoch 30

Loss = 0.214
```

```text
Epoch 80

Loss = 0.081
```

A steadily decreasing loss usually indicates successful learning.

If the loss increases dramatically or becomes

```text
NaN
```

there is likely a problem with

- preprocessing,
- learning rate,
- model architecture,
- or numerical stability.

---

# S13.7.18 Complete Training Script

A simplified training script is shown below.

```python
model.train()

for epoch in range(100):

    epoch_loss = 0

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

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)

    print(

        f"Epoch {epoch+1}: "

        f"{epoch_loss:.4f}"

    )
```

This compact loop forms the foundation of most Graph Neural Network training pipelines.

---

# S13.7.19 Materials Science Example

Suppose our dataset contains

- 18,000 crystal structures,
- DFT-calculated formation energies,
- atom features,
- bond distances.

The training loop proceeds as

```text
Mini-Batch

↓

Crystal Graph Neural Network

↓

Formation Energy Prediction

↓

MSE Loss

↓

Backpropagation

↓

Parameter Update

↓

Repeat
```

After many epochs,

the network gradually learns the relationship between crystal structure and formation energy.

Exactly the same training loop can be used for

- band gap prediction,
- elastic modulus,
- thermal conductivity,
- dielectric constant,
- adsorption energy,

provided the dataset and target values are changed.

---

# Research Tip

Monitor both the training loss and the validation loss throughout training. A continuously decreasing training loss accompanied by an increasing validation loss is often a sign of **overfitting**. Saving the model with the lowest validation error is generally preferable to simply using the final training epoch.

---

# S13.7.20 Summary

In this section, we assembled every component developed so far into a complete Graph Neural Network training loop. We learned how to switch the model into training mode, iterate over graph mini-batches, perform forward propagation, compute the loss, apply backpropagation, update the model parameters, and monitor the training process.

This training loop serves as the foundation for virtually every crystal Graph Neural Network architecture discussed later in the book, including CGCNN, MEGNet, M3GNet, and ALIGNN.

In the next section, we will extend this workflow by implementing a **validation and testing pipeline**, allowing us to evaluate the predictive performance of our models on previously unseen crystal structures.

# S13.8 Building the Validation and Testing Pipeline

Training loss tells us how well the Graph Neural Network fits the training data.

However,

a low training loss **does not necessarily mean the model has learned to generalize**.

A model may simply memorize the training dataset while performing poorly on new crystal structures.

For this reason,

every machine learning model should be evaluated on data that was **never seen during training**.

This evaluation is performed using

- the validation dataset,
- and the test dataset.

Validation allows us to monitor training and tune hyperparameters,

while the test dataset provides the final estimate of model performance.

---

# S13.8.1 Splitting the Dataset

A typical materials informatics workflow divides the complete dataset into three subsets.

```text
Complete Dataset

↓

Training Set

↓

Validation Set

↓

Test Set
```

A common split is

```text
Training

↓

80%
```

```text
Validation

↓

10%
```

```text
Test

↓

10%
```

The exact percentages vary,

but the underlying principle remains the same.

Each crystal structure should belong to **only one subset**.

---

# S13.8.2 Why Validation Is Necessary

Suppose we train a Graph Neural Network for 500 epochs.

Should we always use the model from Epoch 500?

Not necessarily.

Consider the following example.

```text
Epoch

↓

Validation MAE
```

```text
20

↓

0.145 eV
```

```text
60

↓

0.094 eV
```

```text
120

↓

0.082 eV
```

```text
250

↓

0.088 eV
```

Although the training continues,

the validation performance begins to deteriorate after Epoch 120.

The model from Epoch 120 is therefore preferable.

---

# S13.8.3 Switching to Evaluation Mode

Before evaluating the model,

switch it to evaluation mode.

```python
model.eval()
```

Unlike

```python
model.train()
```

evaluation mode

- disables dropout,
- freezes Batch Normalization statistics,
- ensures deterministic predictions.

Always remember

```python
model.eval()
```

before validation or testing.

---

# S13.8.4 Disabling Gradient Computation

Gradients are unnecessary during evaluation.

PyTorch therefore provides

```python
with torch.no_grad():
```

Example

```python
model.eval()

with torch.no_grad():

    prediction = model(batch)
```

Disabling gradients

- reduces memory usage,
- speeds up inference,
- prevents unnecessary computations.

---

# S13.8.5 The Validation Loop

The validation loop resembles the training loop,

but two important operations are omitted.

- No backpropagation.
- No optimizer updates.

A simplified validation loop is

```python
model.eval()

validation_loss = 0

with torch.no_grad():

    for batch in val_loader:

        batch = batch.to(device)

        prediction = model(batch)

        loss = criterion(

            prediction,

            batch.y

        )

        validation_loss += loss.item()

validation_loss /= len(val_loader)
```

The model parameters remain unchanged.

---

# S13.8.6 Comparing Training and Validation

Suppose we record the following losses.

```text
Epoch

↓

Training Loss

↓

Validation Loss
```

```text
20

↓

0.210

↓

0.225
```

```text
60

↓

0.102

↓

0.110
```

```text
120

↓

0.051

↓

0.059
```

```text
220

↓

0.012

↓

0.140
```

Notice

training loss continues decreasing,

while validation loss begins increasing.

This behavior indicates

```text
Overfitting
```

---

# S13.8.7 Evaluating on the Test Dataset

After selecting the best model,

evaluate it once using the test dataset.

The procedure is identical.

```python
model.eval()

test_loss = 0

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        prediction = model(batch)

        loss = criterion(

            prediction,

            batch.y

        )

        test_loss += loss.item()

test_loss /= len(test_loader)
```

The test dataset should not influence model development.

It is reserved for the final performance assessment.

---

# S13.8.8 Mean Absolute Error (MAE)

For materials property prediction,

Mean Absolute Error is one of the most widely reported metrics.

Example

```python
from torch.nn import L1Loss

mae = L1Loss()

error = mae(

    prediction,

    batch.y

)
```

Interpretation

```text
MAE = 0.05 eV
```

means

the average prediction differs from the reference value by approximately

```text
0.05 eV
```

MAE is commonly reported for

- formation energy,
- band gap,
- adsorption energy,
- voltage prediction.

---

# S13.8.9 Root Mean Squared Error (RMSE)

Another common metric is

```text
Root Mean Squared Error
```

It penalizes large prediction errors more strongly than MAE.

Example

```python
mse = criterion(

    prediction,

    batch.y

)

rmse = torch.sqrt(

    mse

)
```

RMSE is particularly useful when large prediction errors are undesirable.

---

# S13.8.10 Coefficient of Determination (R²)

Another useful metric is

```text
R²
```

which measures how well predictions explain the variation in the data.

Using Scikit-learn,

```python
from sklearn.metrics import r2_score

score = r2_score(

    y_true,

    y_pred

)
```

Typical interpretation

```text
R² = 1.0

Perfect Prediction
```

```text
R² = 0.0

No Better Than Predicting the Mean
```

Higher values indicate better predictive performance.

---

# S13.8.11 Collecting Predictions

To compute evaluation metrics,

collect predictions from every mini-batch.

```python
predictions = []

targets = []
```

Inside the evaluation loop,

```python
predictions.extend(

    prediction.cpu().tolist()

)

targets.extend(

    batch.y.cpu().tolist()

)
```

After evaluation,

both lists contain predictions for the entire dataset.

---

# S13.8.12 Inspecting Individual Predictions

Rather than looking only at summary statistics,

inspect individual samples.

```python
for i in range(5):

    print(

        targets[i],

        predictions[i]

    )
```

Example

```text
-3.82  -3.76

-2.14  -2.18

-4.91  -4.88
```

This simple check often reveals systematic prediction errors.

---

# S13.8.13 Creating a Parity Plot

One of the most common figures in materials informatics papers is the parity plot.

Conceptually,

```text
True Property

↓

Predicted Property
```

Ideal predictions lie along

```text
y = x
```

A simple implementation is

```python
import matplotlib.pyplot as plt

plt.scatter(

    targets,

    predictions

)

plt.xlabel(

    "DFT Formation Energy"

)

plt.ylabel(

    "Predicted Formation Energy"

)

plt.show()
```

A tight cluster around the diagonal indicates accurate predictions.

---

# S13.8.14 Saving the Best Model

Suppose the validation MAE improves.

Save the model immediately.

```python
torch.save(

    model.state_dict(),

    "best_model.pt"

)
```

This ensures that the best-performing model is preserved,

even if later epochs begin to overfit.

---

# S13.8.15 Monitoring Validation Performance

A simple training script might produce

```text
Epoch 1

Train Loss: 0.612

Validation MAE: 0.423
```

```text
Epoch 25

Train Loss: 0.144

Validation MAE: 0.102
```

```text
Epoch 60

Train Loss: 0.071

Validation MAE: 0.068
```

```text
Epoch 120

Train Loss: 0.033

Validation MAE: 0.084
```

Although the training loss continues decreasing,

validation performance has begun to worsen.

The model from Epoch 60 should therefore be selected.

---

# S13.8.16 Materials Science Example

Suppose we train a Graph Neural Network to predict formation energies using 40,000 Materials Project crystal structures.

After training,

we evaluate the model on previously unseen materials.

Example results

```text
Validation MAE

↓

0.041 eV/atom
```

```text
Test MAE

↓

0.043 eV/atom
```

```text
R²

↓

0.986
```

These metrics indicate that the model generalizes well beyond the training dataset.

Such evaluation procedures are standard in modern materials informatics research.

---

# Research Tip

Never report only the training error in a scientific publication. Always evaluate the final model using an independent test set and report multiple performance metrics, such as MAE, RMSE, and R². Whenever possible, include a parity plot and clearly state how the dataset was divided into training, validation, and test subsets to ensure reproducibility.

---

# S13.8.17 Summary

A well-designed validation and testing pipeline is essential for developing reliable Graph Neural Networks.

In this section, we learned how to switch a model into evaluation mode, disable gradient computation, calculate validation and test losses, compute commonly used regression metrics such as MAE, RMSE, and R², collect predictions, generate parity plots, and save the best-performing model.

These evaluation techniques are used throughout computational materials science to assess the predictive performance of models before they are applied to new crystal structures or reported in scientific publications.

In the next section, we will learn how to **train Graph Neural Networks efficiently on GPUs**, including moving graph data to CUDA devices, managing GPU memory, and accelerating training for large crystal graph datasets.


# S13.9 Accelerating Crystal Graph Neural Networks with GPUs

The training loop developed in the previous section works correctly on a CPU.

However,

modern Graph Neural Networks are computationally demanding.

A single model may contain

- millions of trainable parameters,
- millions of graph edges,
- and datasets containing tens of thousands of crystal structures.

Training such models on a CPU alone can require

- several hours,
- several days,
- or even weeks.

For this reason,

most modern Graph Neural Network research is performed using **Graphics Processing Units (GPUs)**.

Fortunately,

PyTorch makes GPU acceleration remarkably straightforward.

---

# S13.9.1 Why GPUs Are Faster

A CPU is designed to perform a wide variety of tasks efficiently.

A GPU, on the other hand,

is designed to perform thousands of mathematical operations simultaneously.

Deep learning relies heavily on

- matrix multiplication,
- tensor operations,
- vector arithmetic.

These operations can be executed much faster on a GPU.

Conceptually,

```text
Crystal Graph Batch

↓

Matrix Operations

↓

GPU

↓

Parallel Computation

↓

Faster Training
```

---

# S13.9.2 Checking GPU Availability

Before using a GPU,

verify that PyTorch can detect one.

```python
import torch

print(

    torch.cuda.is_available()

)
```

Example output

```text
True
```

If

```text
False
```

is returned,

PyTorch cannot currently access a CUDA-compatible GPU.

---

# S13.9.3 Selecting the Device

Rather than writing separate code for CPUs and GPUs,

define a device object.

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)
```

Inspect the result.

```python
print(device)
```

Possible outputs

```text
cuda
```

or

```text
cpu
```

From this point onward,

the same code works on either device.

---

# S13.9.4 Moving the Model to the GPU

Creating the model does **not** automatically place it on the GPU.

Suppose we define

```python
model = CrystalGNN(

    node_features=5,

    hidden_channels=64

)
```

Move it explicitly.

```python
model = model.to(

    device

)
```

Now every trainable parameter resides on the selected device.

---

# S13.9.5 Moving Mini-Batches to the GPU

Moving only the model is not enough.

Every graph mini-batch must also be transferred.

Inside the training loop,

write

```python
for batch in train_loader:

    batch = batch.to(

        device

    )
```

Both

- the model,
- and the graph data

must reside on the same device.

---

# S13.9.6 A Common Beginner Error

Consider the following situation.

```text
Model

↓

GPU
```

```text
Graph Batch

↓

CPU
```

Attempting

```python
prediction = model(batch)
```

will produce an error because tensors are stored on different devices.

Always ensure

```text
Model

↓

GPU
```

and

```text
Graph Batch

↓

GPU
```

before performing a forward pass.

---

# S13.9.7 Inspecting the Device

You can verify the device used by a tensor.

```python
print(

    batch.x.device

)
```

Example

```text
cuda:0
```

Likewise,

inspect the model parameters.

```python
print(

    next(

        model.parameters()

    ).device

)
```

Example

```text
cuda:0
```

These simple checks are extremely useful during debugging.

---

# S13.9.8 Complete GPU Training Loop

The complete training loop differs only slightly from the CPU version.

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

model = model.to(device)

for epoch in range(epochs):

    model.train()

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
```

Notice that

only two additional steps are required.

- Move the model.
- Move each mini-batch.

The remaining training loop is unchanged.

---

# S13.9.9 Evaluating on the GPU

Validation follows the same pattern.

```python
model.eval()

with torch.no_grad():

    for batch in val_loader:

        batch = batch.to(device)

        prediction = model(batch)
```

The evaluation pipeline is identical,

except that gradients are disabled.

---

# S13.9.10 Returning Predictions to the CPU

Many Python libraries,

such as NumPy and Matplotlib,

expect CPU tensors.

Before plotting or computing certain metrics,

move predictions back to the CPU.

```python
prediction = prediction.cpu()
```

Convert to a NumPy array if needed.

```python
prediction = prediction.numpy()
```

This is commonly required when generating parity plots or exporting predictions.

---

# S13.9.11 Monitoring GPU Memory

GPU memory is limited.

During development,

inspect memory usage.

```python
print(

    torch.cuda.memory_allocated()

)
```

or

```python
print(

    torch.cuda.memory_reserved()

)
```

These values help diagnose memory-related problems.

---

# S13.9.12 Handling Out-of-Memory Errors

One of the most common training errors is

```text
CUDA out of memory
```

Typical causes include

- batch size too large,
- very large crystal graphs,
- deep Graph Neural Networks,
- multiple models occupying GPU memory.

The simplest solution is to reduce

```python
batch_size
```

For example,

change

```python
batch_size=128
```

to

```python
batch_size=32
```

or

```python
batch_size=16
```

Smaller batches require less GPU memory.

---

# S13.9.13 Clearing Unused GPU Memory

During experimentation,

unused GPU memory may remain allocated.

PyTorch provides

```python
torch.cuda.empty_cache()
```

This releases cached memory back to the GPU memory manager.

Although it does not increase total GPU memory,

it can help reduce fragmentation during interactive experimentation.

---

# S13.9.14 Choosing an Appropriate Batch Size

The optimal batch size depends on

- GPU memory,
- graph size,
- model complexity.

For example,

a dataset containing

```text
Small Unit Cells

↓

Batch Size 128
```

may fit comfortably in memory.

However,

datasets containing

```text
Large Supercells

↓

Batch Size 16
```

may require much smaller batches.

There is no universal choice.

Always experiment.

---

# S13.9.15 Mixed Precision Training

Modern GPUs support mixed precision training,

where some calculations use

```text
float16
```

instead of

```text
float32
```

Advantages include

- faster computation,
- lower memory usage,
- larger batch sizes.

PyTorch supports this through automatic mixed precision (AMP).

A simplified example is

```python
from torch.cuda.amp import autocast

with autocast():

    prediction = model(batch)

    loss = criterion(

        prediction,

        batch.y

    )
```

Mixed precision is particularly beneficial when training large Graph Neural Networks on modern NVIDIA GPUs.

---

# S13.9.16 Multi-GPU Training

Very large crystal datasets may exceed the capabilities of a single GPU.

PyTorch supports distributing training across multiple GPUs.

Conceptually,

```text
Mini-Batch

↓

GPU 1

GPU 2

GPU 3

GPU 4

↓

Gradient Synchronization

↓

Parameter Update
```

Although beyond the scope of this chapter,

multi-GPU training is widely used for state-of-the-art Graph Neural Networks.

---

# S13.9.17 Materials Science Example

Suppose we train a CGCNN model using

- 45,000 crystal structures,
- 200 training epochs,
- batch size of 64.

Approximate training times might resemble

```text
CPU

↓

18 hours
```

```text
Single GPU

↓

2 hours
```

The exact speedup depends on

- hardware,
- graph size,
- model architecture,

but GPU acceleration typically reduces training time dramatically.

This efficiency enables researchers to perform extensive hyperparameter tuning and large-scale experiments that would otherwise be impractical.

---

# Research Tip

Before launching a long training job, verify that both the model and the graph mini-batches are on the same device by checking `next(model.parameters()).device` and `batch.x.device`. This simple verification can prevent many of the most common CUDA-related errors encountered during Graph Neural Network development.

---

# S13.9.18 Summary

GPU acceleration is a fundamental component of modern Graph Neural Network research.

In this section, we learned how to detect CUDA devices, move models and graph batches to the GPU, manage memory, resolve common device-related errors, use automatic mixed precision, and understand the principles of multi-GPU training.

These techniques allow crystal Graph Neural Networks to be trained efficiently on large materials datasets, making modern materials informatics research both practical and scalable.

In the next section, we will learn how to **save trained models, resume interrupted experiments, and perform inference on previously unseen crystal structures**.

# S13.11 Common Errors and Debugging Strategies in Crystal Graph Neural Networks

Building a Graph Neural Network for crystal materials involves several interconnected stages.

The complete workflow contains

```text
Crystal Structure

↓

Graph Construction

↓

Dataset

↓

DataLoader

↓

Model

↓

Training

↓

Evaluation
```

A problem in any one of these stages can cause the final model to fail.

Many beginners assume that poor performance always means

> "The neural network architecture is not good enough."

However,

in practical materials informatics research,

many failures are caused by

- incorrect graph construction,
- wrong tensor dimensions,
- incorrect target values,
- data preprocessing mistakes,
- device mismatches,
- numerical instability.

Therefore,

learning how to debug a Graph Neural Network is as important as learning how to build one.

---

# S13.11.1 The Debugging Philosophy

A good debugging strategy follows the same direction as the data flow.

Instead of immediately modifying the model,

inspect each stage.

```text
Crystal File

↓

Check Structure

↓

Graph

↓

Check Nodes and Edges

↓

Batch

↓

Check Tensor Shapes

↓

Model

↓

Check Output

↓

Loss

↓

Check Training
```

Never assume that an error comes from the last component you modified.

---

# S13.11.2 Debugging Crystal Structures

The first possible source of error is the crystal structure itself.

Before converting structures into graphs,

inspect the input.

Example:

```python
print(structure)
```

Possible output:

```text
Full Formula (Li4 P2 S7)

Sites (13)
```

Check

- chemical composition,
- number of atoms,
- lattice parameters,
- periodic boundary conditions.

A corrupted CIF file can produce incorrect graphs.

---

# S13.11.3 Checking Pymatgen Structure Objects

For materials workflows,

Pymatgen is usually responsible for reading crystal structures.

Example:

```python
from pymatgen.core import Structure

structure = Structure.from_file(

    "material.cif"

)
```

Inspect the number of atoms.

```python
print(

    len(structure)

)
```

Example:

```text
40
```

If a material expected to contain 40 atoms returns

```text
0
```

or

```text
4000
```

there is likely a preprocessing problem.

---

# S13.11.4 Debugging Graph Construction

After converting a crystal into a graph,

always inspect the graph object.

Example:

```python
graph = structure_to_graph(

    structure

)

print(graph)
```

Expected output:

```text
Data(
x=[40,5],
edge_index=[2,240],
edge_attr=[240,1],
y=[1]
)
```

This tells us:

```text
40 nodes

↓

5 node features
```

and

```text
240 connections

↓

1 edge feature
```

---

# S13.11.5 Common Graph Construction Errors

## Error 1: No Edges

Output:

```text
edge_index=[2,0]
```

This means the graph contains no connections.

Possible causes:

- cutoff radius too small,
- periodic neighbors not included,
- incorrect distance calculation.

Solution:

Increase the neighbor cutoff.

Example:

```python
cutoff = 5.0
```

instead of

```python
cutoff = 2.0
```

---

## Error 2: Too Many Edges

Output:

```text
edge_index=[2,50000]
```

Possible causes:

- cutoff radius too large,
- duplicate neighbors,
- incorrect periodic expansion.

A very dense graph increases memory usage and slows training.

---

# S13.11.6 Debugging Node Features

Node features represent atomic information.

Example:

```python
print(graph.x.shape)
```

Expected:

```text
torch.Size([40,5])
```

Meaning:

```text
40 atoms

↓

5 features per atom
```

---

Incorrect:

```text
torch.Size([40])
```

This means the feature dimension is missing.

The model expects

```text
(number of atoms, number of features)
```

not a simple list.

---

# S13.11.7 Debugging Edge Index

The edge index is one of the most important graph tensors.

Example:

```python
print(graph.edge_index)
```

Expected format:

```text
tensor([

[0,1,2,3,...],

[1,0,3,2,...]

])
```

The shape should always be

```python
[2, number_of_edges]
```

Check:

```python
print(

graph.edge_index.shape

)
```

Example:

```text
torch.Size([2,240])
```

---

# S13.11.8 Understanding Edge Index Errors

A common error:

```text
IndexError:
index 45 is out of bounds
```

This means an edge refers to a node that does not exist.

Example:

The graph contains

```text
40 nodes
```

but an edge points to

```text
node 45
```

The edge construction algorithm is incorrect.

---

# S13.11.9 Debugging DataLoader Batches

After batching,

inspect the output.

```python
batch = next(iter(loader))

print(batch)
```

Example:

```text
DataBatch(
x=[200,5],
edge_index=[2,1200],
y=[16],
batch=[200]
)
```

Interpretation:

```text
200 total atoms

↓

16 crystal structures
```

---

Check the number of graphs.

```python
print(

batch.num_graphs

)
```

Output:

```text
16
```

If this does not match your batch size,

investigate the dataset.

---

# S13.11.10 Tensor Shape Debugging

Tensor shapes are the most common source of deep learning errors.

Always print shapes.

Example:

```python
print(batch.x.shape)

print(batch.edge_index.shape)

print(batch.y.shape)
```

Expected:

```text
Node Features

↓

[Total Nodes, Features]
```

```text
Edges

↓

[2, Total Edges]
```

```text
Targets

↓

[Number of Graphs]
```

---

# S13.11.11 Debugging Model Output

After forward propagation:

```python
prediction = model(batch)
```

Inspect:

```python
print(

prediction.shape

)
```

For regression,

the output should usually match the target shape.

Example:

Prediction:

```text
torch.Size([32])
```

Target:

```text
torch.Size([32])
```

Correct.

---

Incorrect:

Prediction:

```text
torch.Size([32,1])
```

Target:

```text
torch.Size([32])
```

This mismatch may cause loss calculation problems.

A simple solution:

```python
prediction = prediction.squeeze()
```

---

# S13.11.12 NaN Loss Problems

One serious training problem is

```text
Loss = NaN
```

Possible causes:

- learning rate too large,
- invalid input values,
- exploding gradients,
- division by zero,
- incorrect normalization.

---

# S13.11.13 Checking for Invalid Values

Inspect your data.

```python
torch.isnan(

batch.x

).any()
```

If output:

```text
True
```

your features contain invalid values.

Similarly,

check targets.

```python
torch.isnan(

batch.y

).any()
```

---

# S13.11.14 Reducing Learning Rate

If training becomes unstable,

reduce the learning rate.

Example:

Before:

```python
lr=0.01
```

After:

```python
lr=0.001
```

A smaller learning rate produces more stable optimization.

---

# S13.11.15 Gradient Explosion

Sometimes gradients become extremely large.

Check gradient magnitude.

Example:

```python
for parameter in model.parameters():

    if parameter.grad is not None:

        print(

        parameter.grad.norm()

        )
```

Very large values indicate unstable training.

---

# S13.11.16 Gradient Clipping

A common solution is gradient clipping.

```python
torch.nn.utils.clip_grad_norm_(

    model.parameters(),

    max_norm=1.0

)
```

Place it before

```python
optimizer.step()
```

Example:

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(

    model.parameters(),

    1.0

)

optimizer.step()
```

This limits extreme updates.

---

# S13.11.17 GPU Debugging

Common error:

```text
Expected all tensors to be on the same device
```

Solution:

Check:

```python
print(batch.x.device)

print(next(model.parameters()).device)
```

Both should return:

```text
cuda:0
```

---

# S13.11.18 GPU Memory Problems

Error:

```text
CUDA out of memory
```

Solutions:

Reduce:

```python
batch_size
```

Example:

```python
batch_size=32
```

instead of

```python
batch_size=128
```

Other solutions:

- reduce hidden dimension,
- reduce number of layers,
- use mixed precision.

---

# S13.11.19 Overfitting Debugging

A common materials ML problem:

Training error:

```text
Very Low
```

Testing error:

```text
Very High
```

This indicates overfitting.

Possible solutions:

- collect more training structures,
- reduce model complexity,
- apply dropout,
- use regularization,
- stop training earlier.

---

# S13.11.20 Underfitting Debugging

The opposite problem:

Training error:

```text
High
```

Validation error:

```text
High
```

The model has not learned enough.

Solutions:

- increase model capacity,
- train longer,
- improve features,
- reduce excessive regularization.

---

# S13.11.21 Materials Science Specific Debugging

Materials datasets contain unique challenges.

## Problem: Similar Structures in Train and Test

Random splitting may place chemically similar structures in both sets.

This can produce unrealistically good results.

Solution:

Use

- composition-based splitting,
- structure-based splitting,
- time-based splitting.

---

## Problem: DFT Data Quality

Machine learning cannot correct inaccurate reference data.

If DFT calculations contain inconsistent settings,

the model learns those inconsistencies.

Always check:

- exchange-correlation functional,
- convergence criteria,
- computational settings.

---

# S13.11.22 A Practical Debugging Checklist

Before training:

```text
✓ CIF files load correctly

✓ Structures are reasonable

✓ Graphs contain nodes

✓ Graphs contain edges

✓ Features have correct dimensions

✓ Targets are correct
```

During training:

```text
✓ Loss decreases

✓ No NaN values

✓ Gradients are reasonable

✓ GPU usage is correct
```

Before publication:

```text
✓ Test set is independent

✓ Metrics are reported

✓ Predictions are inspected

✓ Results are reproducible
```

---

# Research Tip

When a Graph Neural Network fails, do not immediately change the architecture. First verify the data pipeline. In materials informatics, incorrect crystal preprocessing is often a much larger source of error than the choice between different neural network architectures.

---

# S13.11.23 Summary

Debugging is an essential research skill when developing Graph Neural Networks for materials science.

In this section, we learned how to inspect crystal structures, verify graph construction, debug tensor shapes, identify batching problems, solve GPU errors, handle NaN losses, prevent overfitting, and diagnose materials-specific issues.

A successful materials informatics researcher does not only know how to build models.

They know how to determine **why a model fails** and how to systematically fix it.

In the next section, we will combine everything developed throughout this supplement into a complete end-to-end crystal Graph Neural Network project.

# S13.12 End-to-End Crystal Graph Neural Network Project

In the previous sections,

we developed every individual component required for building a Graph Neural Network for materials prediction.

We learned how to

- convert crystal structures into graphs,
- create PyTorch Geometric datasets,
- batch multiple crystals,
- build Graph Neural Network architectures,
- train models,
- validate performance,
- use GPUs,
- save models,
- and debug common problems.

However,

a researcher does not work with isolated pieces.

A real materials informatics project requires connecting all these components into one complete workflow.

This section presents a complete end-to-end example:

```text
Crystal Structure

↓

Pymatgen

↓

Graph Representation

↓

PyTorch Geometric Dataset

↓

Graph Neural Network

↓

Training

↓

Validation

↓

Model Saving

↓

Prediction of New Materials
```

This workflow represents the foundation of modern crystal graph machine learning research.

---

# S13.12.1 Research Problem Definition

Every machine learning project begins with a scientific question.

Example:

> Can we predict the formation energy of a crystal structure directly from its atomic arrangement?

The objective is:

Input:

```text
Crystal Structure
```

Output:

```text
Formation Energy
```

Mathematically,

\[
f(\text{Crystal Structure}) \rightarrow E_f
\]

where

\[
E_f
\]

is the formation energy.

---

# S13.12.2 Dataset Preparation

A materials dataset contains three important components.

## 1. Crystal Structures

Usually stored as

```text
CIF files
```

Example:

```text
LiCoO2.cif

Si.cif

SrTiO3.cif
```

---

## 2. Target Properties

Calculated using

- DFT,
- experiments,
- databases.

Example:

```text
LiCoO2

Formation Energy

↓

-7.42 eV
```

---

## 3. Metadata

Additional information may include:

- material ID,
- chemical composition,
- calculation method,
- reference source.

---

# S13.12.3 Project Folder Organization

A professional project may look like:

```text
Crystal_GNN_Project/

│

├── data/

│   ├── structures/

│   │   ├── material1.cif

│   │   ├── material2.cif

│   │

│   └── targets.csv

│

├── dataset.py

├── model.py

├── train.py

├── evaluate.py

├── predict.py

└── checkpoints/

```

Separating components makes experiments easier to reproduce.

---

# S13.12.4 Installing Required Libraries

A typical environment requires:

```bash
pip install torch

pip install torch-geometric

pip install pymatgen

pip install pandas

pip install scikit-learn

pip install matplotlib
```

These libraries provide

- graph computation,
- crystal handling,
- data processing,
- evaluation tools.

---

# S13.12.5 Reading Crystal Structures Using Pymatgen

The first step is loading a crystal.

```python
from pymatgen.core import Structure


structure = Structure.from_file(

    "LiCoO2.cif"

)


print(structure)
```

Example output:

```text
Full Formula (LiCoO2)

Sites (12)
```

The structure object now contains:

- atomic species,
- coordinates,
- lattice information.

---

# S13.12.6 Converting a Crystal Into a Graph

The crystal must now become a graph.

The conversion follows:

```text
Atom

↓

Node
```

and

```text
Atomic Neighbor Relationship

↓

Edge
```

Example:

```python
from torch_geometric.data import Data


graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_features,

    y=target

)
```

The graph now contains all information required by the GNN.

---

# S13.12.7 Creating a Dataset Class

A PyTorch Geometric dataset manages multiple crystal graphs.

Example:

```python
from torch_geometric.data import Dataset


class CrystalDataset(Dataset):

    def __init__(self, structures, targets):

        self.structures = structures

        self.targets = targets


    def len(self):

        return len(self.structures)


    def get(self, idx):

        structure = self.structures[idx]

        target = self.targets[idx]


        graph = structure_to_graph(

            structure,

            target

        )


        return graph
```

The dataset automatically provides graphs during training.

---

# S13.12.8 Creating Training and Testing Sets

Split the dataset.

```python
from sklearn.model_selection import train_test_split


train_data, test_data = train_test_split(

    dataset,

    test_size=0.1,

    random_state=42

)
```

A validation set can also be created.

Example:

```text
Training

↓

80%

Validation

↓

10%

Testing

↓

10%
```

---

# S13.12.9 Creating DataLoaders

The DataLoader creates mini-batches.

```python
from torch_geometric.loader import DataLoader


train_loader = DataLoader(

    train_data,

    batch_size=32,

    shuffle=True

)


test_loader = DataLoader(

    test_data,

    batch_size=32

)
```

Each batch contains multiple crystal graphs.

---

# S13.12.10 Building the Crystal GNN Model

A simple crystal GNN contains:

```text
Node Features

↓

Graph Convolution

↓

Graph Convolution

↓

Pooling

↓

Prediction Layer
```

Example:

```python
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class CrystalGNN(torch.nn.Module):

    def __init__(self):

        super().__init__()


        self.conv1 = GCNConv(

            5,

            64

        )


        self.conv2 = GCNConv(

            64,

            64

        )


        self.fc = torch.nn.Linear(

            64,

            1

        )


    def forward(self, data):

        x = self.conv1(

            data.x,

            data.edge_index

        )


        x = torch.relu(x)


        x = self.conv2(

            x,

            data.edge_index

        )


        x = global_mean_pool(

            x,

            data.batch

        )


        x = self.fc(x)


        return x.squeeze()
```

This is a complete graph-level regression model.

---

# S13.12.11 Initializing Training Components

Create:

```python
model = CrystalGNN()
```

Loss function:

```python
criterion = torch.nn.MSELoss()
```

Optimizer:

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001

)
```

---

# S13.12.12 Complete Training Loop

The complete training process:

```python
epochs = 100


for epoch in range(epochs):


    model.train()


    total_loss = 0


    for batch in train_loader:


        optimizer.zero_grad()


        prediction = model(batch)


        loss = criterion(

            prediction,

            batch.y

        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()



    print(

        epoch,

        total_loss

    )
```

This loop teaches the GNN the relationship between crystal structures and formation energies.

---

# S13.12.13 Validation Step

After training,

evaluate the model.

```python
model.eval()


predictions = []

targets = []


with torch.no_grad():


    for batch in test_loader:


        prediction = model(batch)


        predictions.extend(

            prediction.tolist()

        )


        targets.extend(

            batch.y.tolist()

        )
```

Now the model predictions can be compared with reference values.

---

# S13.12.14 Calculating Performance Metrics

Example:

```python
from sklearn.metrics import mean_absolute_error


mae = mean_absolute_error(

    targets,

    predictions

)


print(

    "MAE:",

    mae

)
```

Output:

```text
MAE: 0.052 eV
```

The model prediction error is approximately

```text
0.052 eV
```

---

# S13.12.15 Saving the Trained Model

After obtaining good performance:

```python
torch.save(

    model.state_dict(),

    "formation_energy_gnn.pt"

)
```

The trained model can now be reused.

---

# S13.12.16 Predicting a New Material

Suppose we discover a new crystal.

```text
New_Candidate.cif
```

Load it:

```python
new_structure = Structure.from_file(

    "New_Candidate.cif"

)
```

Convert:

```python
new_graph = structure_to_graph(

    new_structure

)
```

Predict:

```python
model.eval()


with torch.no_grad():

    energy = model(

        new_graph

    )


print(energy)
```

The GNN provides a rapid prediction.

---

# S13.12.17 Complete Materials Discovery Workflow

The complete research workflow becomes:

```text
Large Materials Database

↓

Crystal Structures

↓

Graph Conversion

↓

Graph Neural Network

↓

Property Prediction

↓

Ranking Candidates

↓

DFT Verification

↓

Experimental Testing
```

This is the central idea behind modern materials informatics.

---

# S13.12.18 Example Research Scenario

Imagine screening battery materials.

Dataset:

```text
200,000 crystal structures
```

Target:

```text
Lithium diffusion energy barrier
```

Traditional approach:

```text
200,000 DFT Calculations
```

would be extremely expensive.

Machine learning approach:

```text
Train GNN

↓

Predict all candidates

↓

Select top 100

↓

Perform accurate DFT calculations
```

The GNN acts as a fast screening tool.

---

# S13.12.19 Important Research Considerations

A working model is not automatically a reliable scientific model.

A researcher must consider:

## Data Quality

Are the reference calculations consistent?

---

## Data Distribution

Does the training set represent the materials being predicted?

---

## Physical Meaning

Does the model learn chemically meaningful patterns?

---

## Generalization

Can the model predict completely new materials?

---

# S13.12.20 Final Complete Workflow Summary

The complete crystal Graph Neural Network pipeline is:

```text
1. Define Materials Problem

↓

2. Collect Crystal Dataset

↓

3. Read Structures Using Pymatgen

↓

4. Convert Structures Into Graphs

↓

5. Create PyTorch Geometric Dataset

↓

6. Split Dataset

↓

7. Build GNN Model

↓

8. Train Using Backpropagation

↓

9. Validate Performance

↓

10. Save Best Model

↓

11. Predict New Materials
```

This workflow represents the foundation for advanced crystal graph architectures discussed in later chapters.

---

# Research Tip

The most important skill in materials informatics is not memorizing a particular neural network architecture. It is learning how to transform a scientific materials problem into a machine learning workflow: defining the target property, representing the crystal correctly, selecting an appropriate model, evaluating performance honestly, and interpreting predictions scientifically.

---

# S13.12.21 Summary

This supplement completed the practical implementation pathway for Crystal Graph Neural Networks.

Starting from raw crystal structures, we built a complete machine learning workflow involving Pymatgen, graph construction, PyTorch Geometric, model training, evaluation, saving, and prediction.

After completing this chapter, a materials science researcher should understand not only the theory behind Graph Neural Networks but also how these models are implemented in actual computational materials research.

In the next chapter, we will move from general Graph Neural Networks to specialized crystal architectures, beginning with one of the most influential models in materials informatics:

**Crystal Graph Convolutional Neural Network (CGCNN).**