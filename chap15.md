# Part IV — Advanced Graph Neural Networks for Materials Science

# Chapter 15 — Materials Graph Networks (MEGNet)

---

# Learning Objectives

After completing this chapter, the reader will be able to

- Understand why the Crystal Graph Convolutional Neural Network (CGCNN) required further improvements.
- Explain the motivation behind the development of Materials Graph Networks (MEGNet).
- Understand the Graph Networks framework proposed by Battaglia *et al.* and its adaptation to materials science.
- Represent crystalline materials using node, edge, and global state attributes.
- Derive every mathematical component of the MEGNet architecture.
- Implement every component of MEGNet from scratch using PyTorch and PyTorch Geometric.
- Construct datasets suitable for MEGNet training.
- Train, validate, and evaluate MEGNet models for property prediction.
- Compare MEGNet with CGCNN and understand when each architecture should be preferred.
- Apply MEGNet to modern materials informatics research problems.

---

# Introduction

The Crystal Graph Convolutional Neural Network introduced in the previous chapter demonstrated a revolutionary idea:

> A crystal structure can be treated as a graph, and graph neural networks can automatically learn meaningful representations for predicting material properties.

Instead of manually designing descriptors such as

- average electronegativity,
- average atomic radius,
- packing fraction,
- coordination number,

CGCNN learned structural representations directly from atomic connectivity.

This represented a major shift in materials informatics.

However, shortly after the publication of CGCNN, researchers began identifying several limitations.

These limitations inspired the development of a more general graph neural network architecture known as the **Materials Graph Network (MEGNet)**.

Unlike CGCNN, which was specifically designed for crystalline materials, MEGNet was based on a much broader mathematical framework known as the **Graph Network (GN)** framework.

This seemingly small conceptual shift dramatically increased the flexibility of graph neural networks.

Rather than designing update equations specifically for crystals, MEGNet formulated materials learning as repeated information exchange between three different entities:

- atoms,
- bonds,
- the entire crystal.

This unified representation made it possible to incorporate not only atomic and bond information but also global state variables such as

- temperature,
- pressure,
- magnetic field,
- processing conditions,
- external thermodynamic parameters.

Consequently, MEGNet became one of the first graph neural networks capable of learning material behavior under varying physical environments.

Throughout this chapter, we will study MEGNet from first principles.

Rather than treating MEGNet as simply another graph neural network, we will derive every mathematical component, understand the physical motivation behind each design decision, and implement the complete architecture from scratch.

---

# 15.1 Why CGCNN Was Not Enough

Before introducing a new model, it is important to understand why researchers felt that another architecture was necessary.

Scientific progress almost always follows the same pattern.

```text
Existing Model

↓

Strengths

↓

Weaknesses

↓

New Architecture

↓

Improved Performance
```

Therefore, the first question we should ask is

> **What limitations prevented CGCNN from becoming a universal graph neural network for materials science?**

The answer lies in the design philosophy of CGCNN itself.

CGCNN was specifically designed for crystalline solids.

Its message-passing mechanism was highly effective for learning structural representations of crystals.

However, several important challenges remained unresolved.

---

## Limitation 1 — Architecture Specialized for Crystal Property Prediction

The original CGCNN architecture was developed primarily for supervised prediction tasks such as

- formation energy,
- band gap,
- elastic modulus,
- bulk modulus.

The workflow can be summarized as

```text
Crystal Structure

↓

Crystal Graph

↓

CGCNN

↓

Material Property
```

Although this workflow achieved excellent predictive performance, the architecture itself was closely tied to this specific problem.

It was not originally formulated as a general graph network capable of representing arbitrary physical systems.

Consequently, extending CGCNN to other domains often required redesigning portions of the architecture.

Researchers therefore sought a more universal formulation.

---

## Limitation 2 — No Explicit Global State Information

Consider two experiments performed on the same crystal.

Experiment A

```text
Temperature = 300 K
Pressure = 1 atm
```

Experiment B

```text
Temperature = 900 K
Pressure = 20 GPa
```

The atomic arrangement may initially appear identical.

However,

their physical behavior is not.

Properties such as

- free energy,
- phase stability,
- diffusion,
- magnetic ordering,

depend strongly on external conditions.

CGCNN receives only the crystal graph.

Its inputs consist of

- atoms,
- neighboring atoms,
- bond distances.

There is no mechanism for representing global environmental information.

Mathematically,

CGCNN models

$$
\text{Property}=f(\text{Crystal Graph})
$$

The external state of the system is absent.

This omission limits the applicability of the model whenever material properties depend on thermodynamic conditions.

---

## Limitation 3 — Limited Information Flow

In CGCNN,

information propagates primarily through neighboring atoms.

Suppose we examine the following portion of a crystal.

```text
A —— B —— C —— D
```

During one graph convolution,

Atom B receives information from

- Atom A
- Atom C

Atom D is invisible to Atom B.

After two graph convolutions,

information from Atom D can finally reach Atom B.

This gradual propagation is common to message-passing neural networks.

However,

CGCNN lacks an explicit mechanism allowing information describing the entire crystal to influence every atom simultaneously.

For many physical systems,

global information plays an essential role.

Examples include

- total charge,
- external electric field,
- pressure,
- crystal symmetry,
- processing conditions.

These quantities cannot naturally be represented using only local neighbor interactions.

---

## Limitation 4 — Fixed Update Strategy

The CGCNN update equation combines neighboring information using a fixed architecture.

Although extremely successful,

the update mechanism is largely specialized for crystal graphs.

Researchers wondered whether a more general update rule could be constructed.

Instead of designing equations specifically for atoms,

could we create one mathematical framework applicable to

- molecules,
- crystals,
- polymers,
- batteries,
- catalysts,
- biological systems?

This question motivated the Graph Network framework.

---

## Limitation 5 — Weak Separation Between Graph Components

A graph naturally contains several different types of information.

Node information

```text
Atom

↓

Atomic Number

↓

Electronegativity

↓

Atomic Radius
```

Edge information

```text
Bond

↓

Distance

↓

Bond Type

↓

Neighbor Relationship
```

Global information

```text
Crystal

↓

Temperature

↓

Pressure

↓

External Field

↓

Composition
```

CGCNN primarily focuses on nodes and edges.

Global information is not treated as a first-class component of the graph.

As a result,

representing changes in environmental conditions becomes difficult.

---

## Limitation 6 — Limited Generalization to Arbitrary Graph Problems

Although CGCNN is extremely effective for crystalline materials,

its architecture is not intended to serve as a universal graph neural network.

Modern machine learning increasingly favors general frameworks that can be adapted to many different applications.

Researchers therefore desired an architecture possessing

- greater flexibility,
- modular design,
- reusable update functions,
- extensibility.

These requirements eventually led to the adoption of the Graph Networks formalism.

---

## Summary of CGCNN Limitations

The major limitations of CGCNN can be summarized as follows.

| Limitation | Consequence |
|------------|-------------|
| No global state representation | Cannot naturally model temperature, pressure, or external fields |
| Crystal-specific design | Difficult to generalize to broader graph problems |
| Fixed message-passing formulation | Less architectural flexibility |
| Primarily local interactions | Limited incorporation of global physical information |
| Specialized update equations | Harder to extend and modify |

It is important to emphasize that these limitations do **not** imply that CGCNN is a poor model.

On the contrary,

CGCNN remains one of the foundational graph neural networks in materials science and continues to be widely used.

Rather,

its limitations motivated researchers to ask a deeper question:

> Can we develop a mathematically general graph neural network framework that treats nodes, edges, and global state variables in a unified manner while remaining applicable to materials science?

The answer to this question was **MEGNet**.

In the next section, we will explore the historical development of MEGNet, examine the scientific ideas that inspired it, and understand how the Graph Network framework transformed graph neural networks from specialized architectures into a general computational framework for learning on graphs.

# 15.2 Historical Development of MEGNet

Scientific breakthroughs rarely emerge in isolation.

Instead, they are usually the result of years of incremental progress, where each new discovery builds upon previous ideas while addressing their limitations.

The development of the **Materials Graph Network (MEGNet)** is an excellent example of this process.

To fully appreciate MEGNet, it is important to understand not only *what* it is, but also *why* it was developed and *how* it evolved from earlier work in graph neural networks.

The history of MEGNet lies at the intersection of two rapidly advancing fields:

- Graph Neural Networks (Machine Learning)
- Computational Materials Science

Although these disciplines developed independently for many years, their convergence has fundamentally transformed modern materials informatics.

---

# 15.2.1 Before Deep Learning in Materials Science

Before graph neural networks became popular, most machine learning methods for materials science relied heavily on **handcrafted descriptors**.

Researchers first converted each material into a numerical feature vector before applying traditional machine learning algorithms.

A typical workflow looked like

```text
Crystal Structure

↓

Feature Engineering

↓

Feature Vector

↓

Machine Learning Model

↓

Material Property
```

Common descriptors included

- average atomic number,
- average atomic radius,
- average electronegativity,
- average valence electron count,
- density,
- packing fraction,
- coordination number,
- bond statistics.

For example,

a binary compound

```text
MgO
```

might be represented as

| Feature | Value |
|---------|-------:|
| Average Atomic Number | 10 |
| Average Electronegativity | 2.38 |
| Density | 3.58 g/cm³ |
| Average Radius | 1.15 Å |

Although these descriptors often produced reasonable predictions,

they suffered from several major disadvantages.

---

## Dependence on Human Expertise

Feature engineering required extensive domain knowledge.

Researchers had to determine

- which descriptors were important,
- how to calculate them,
- whether they were physically meaningful.

Different research groups often designed completely different descriptor sets for the same problem.

As a result,

the success of a machine learning model depended heavily on the quality of manually designed features.

---

## Information Loss

Another major problem was that handcrafted descriptors often discarded important structural information.

Consider the following two crystal structures.

```text
Material A

A

/ \

B   C
```

```text
Material B

A

|

B

|

C
```

Both materials may have identical average atomic numbers,

yet their atomic arrangements are fundamentally different.

Traditional descriptors may fail to distinguish them.

This information loss limits prediction accuracy.

---

## Poor Generalization

Descriptors designed for one class of materials often performed poorly on another.

For example,

descriptors optimized for

- oxides

might not adequately describe

- metallic alloys,
- semiconductors,
- layered materials.

Researchers therefore desired a representation capable of adapting automatically to different classes of materials.

---

# 15.2.2 The Rise of Representation Learning

Around the same time,

deep learning was revolutionizing fields such as

- computer vision,
- speech recognition,
- natural language processing.

A key idea behind deep learning was

> Let the neural network learn useful representations automatically.

Instead of manually engineering features,

the model learns them directly from data.

This philosophy became known as **representation learning**.

The workflow changed dramatically.

Instead of

```text
Raw Data

↓

Manual Features

↓

Machine Learning
```

deep learning introduced

```text
Raw Data

↓

Neural Network

↓

Learned Representation

↓

Prediction
```

The neural network itself became responsible for discovering useful features.

This idea naturally inspired materials scientists.

Could a neural network learn useful representations directly from crystal structures?

---

# 15.2.3 Crystal Graph Convolutional Neural Network (CGCNN)

This question led to one of the most influential developments in materials informatics:

the **Crystal Graph Convolutional Neural Network (CGCNN)**.

CGCNN represented crystals as graphs.

```text
Atoms

↓

Nodes

Neighbor Relationships

↓

Edges

↓

Graph Neural Network

↓

Property Prediction
```

Rather than relying on handcrafted descriptors,

CGCNN learned atomic representations through message passing.

Its success demonstrated that graph neural networks were exceptionally well suited for crystalline materials.

For many benchmark datasets,

CGCNN outperformed conventional machine learning methods.

However,

its architecture remained specialized for crystal property prediction.

Researchers soon began asking

> Can we create a graph neural network that is more general, more modular, and more physically expressive?

---

# 15.2.4 The Graph Network Revolution

While CGCNN was gaining popularity,

graph neural network research in the broader machine learning community was advancing rapidly.

A particularly influential contribution came from

**Battaglia et al.**

who proposed the **Graph Networks (GN) framework**.

Instead of designing graph neural networks for individual applications,

the GN framework introduced a unified mathematical description of graph computation.

Every graph was viewed as consisting of three fundamental components.

```text
Graph

↓

Nodes

Edges

Global State
```

Each component possessed

- its own attributes,
- its own update function,
- its own aggregation mechanism.

This modular formulation transformed graph neural networks from application-specific architectures into a general computational framework.

The implications for materials science were immediately apparent.

---

# 15.2.5 Adapting the Graph Network Framework to Materials

Researchers recognized that crystalline materials naturally fit the Graph Network formalism.

A crystal can be interpreted as

```text
Graph

↓

Atoms

↓

Nodes
```

```text
Chemical Bonds

↓

Edges
```

```text
Entire Crystal

↓

Global State
```

Unlike CGCNN,

the Graph Network framework allowed the entire material to possess its own learnable representation.

This opened the possibility of incorporating

- temperature,
- pressure,
- magnetic field,
- external stress,
- synthesis conditions,

directly into the graph.

Rather than treating these quantities separately,

they became part of the learning process.

---

# 15.2.6 Birth of MEGNet

Building upon the Graph Network framework,

researchers at **Lawrence Berkeley National Laboratory** developed the **Materials Graph Network (MEGNet)**.

The primary objective was straightforward.

> Build a graph neural network specifically for materials science while retaining the mathematical flexibility of the general Graph Network framework.

Instead of inventing an entirely new graph neural network,

MEGNet specialized the existing Graph Network architecture for crystalline materials.

The resulting workflow became

```text
Crystal Structure

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

Global State Features

↓

MEGNet Blocks

↓

Pooling

↓

Property Prediction
```

This architecture generalized the ideas introduced by CGCNN while remaining firmly grounded in modern graph neural network theory.

---

# 15.2.7 Why MEGNet Was a Major Step Forward

MEGNet introduced several important conceptual advances.

First,

it treated

- atoms,
- bonds,
- global state,

as equally important components of the graph.

Second,

its update functions became modular.

Instead of one specialized convolution equation,

MEGNet defined separate update functions for

- edges,
- nodes,
- global state.

This modularity made the architecture significantly easier to extend.

Third,

the Graph Network formulation provided a unified framework applicable to

- molecules,
- crystals,
- amorphous materials,
- batteries,
- catalysts,
- polymers.

Rather than designing different neural networks for different systems,

the same mathematical framework could be adapted to many scientific problems.

---

# 15.2.8 Scientific Impact of MEGNet

Since its introduction,

MEGNet has become one of the foundational architectures in materials informatics.

It has been successfully applied to

- formation energy prediction,
- band gap prediction,
- elastic property prediction,
- thermodynamic property prediction,
- molecular property prediction,
- catalyst discovery,
- battery materials,
- high-throughput materials screening.

More importantly,

its modular Graph Network formulation influenced many later architectures,

including

- M3GNet,
- MatGL,
- CHGNet,

and numerous other graph neural networks used in computational materials science today.

Rather than replacing CGCNN,

MEGNet expanded the conceptual foundations of graph neural networks by demonstrating how physically meaningful global information could be incorporated into message passing.

This marked an important transition from **graph convolution architectures** toward **general graph network architectures**, laying the groundwork for many of the models that followed.

In the next section, we will examine the **original MEGNet paper** in detail, exploring its research objectives, architectural innovations, experimental methodology, and the scientific contributions that established MEGNet as one of the most influential graph neural networks in modern materials informatics.

# 15.4 Graph Networks Framework

---

Before we can understand MEGNet, we must first understand the mathematical framework upon which it is built.

Many beginners incorrectly assume that MEGNet is simply an improved version of CGCNN.

This is not true.

CGCNN is a specific graph neural network architecture developed specifically for crystalline materials.

MEGNet, on the other hand, is an implementation of a much broader and mathematically rigorous framework known as the **Graph Network (GN) Framework**.

This distinction is extremely important.

Without understanding Graph Networks, MEGNet appears to be nothing more than another message passing neural network.

After understanding Graph Networks, however, it becomes clear that MEGNet is actually a specialization of a universal computational framework capable of operating on **any graph**, regardless of whether the graph represents

- crystals,
- molecules,
- proteins,
- social networks,
- transportation systems,
- electrical circuits,
- knowledge graphs.

Therefore, before implementing MEGNet, we must first build a solid understanding of Graph Networks from first principles.

This section is one of the most important sections in the entire book.

Many modern graph neural networks—including

- MEGNet,
- M3GNet,
- CHGNet,
- MatGL,

either directly or indirectly inherit ideas from this framework.

Mastering Graph Networks means understanding the mathematical language spoken by nearly every modern graph neural network.

---

# 15.4.1 Why Was the Graph Network Framework Introduced?

To appreciate the Graph Network framework, we must briefly examine the state of graph neural network research before its introduction.

Early graph neural networks were usually designed independently.

Researchers working on chemistry designed one architecture.

Researchers working on computer vision designed another.

Researchers studying recommender systems developed yet another.

Although all these models operated on graphs, they often differed substantially.

For example,

Graph Convolutional Networks (GCN)

used one update rule.

Graph Attention Networks (GAT)

used another.

Message Passing Neural Networks (MPNN)

used yet another.

Each architecture introduced its own notation, update equations, terminology, and implementation details.

As the field grew,

graph neural networks became increasingly fragmented.

Researchers recognized that most of these architectures shared common computational principles.

Instead of continuously inventing new graph architectures,

could there exist a single mathematical framework capable of describing all of them?

This question motivated Battaglia and colleagues to propose the Graph Network framework.

Rather than defining one particular neural network,

they defined a **computational language** for graph-based learning.

Every graph neural network could then be interpreted as a particular implementation of this language.

This idea is remarkably similar to the relationship between

Newton's Laws

and

specific engineering applications.

Newton's laws are universal.

Different engineering systems simply apply those laws differently.

Similarly,

Graph Networks define universal graph computations.

Specific architectures such as MEGNet simply instantiate them.

---

# 15.4.2 What Is a Graph?

Everything in Graph Networks begins with one mathematical object:

the graph.

In mathematics,

a graph is **not** a plot with x and y axes.

Instead,

a graph is a collection of

- objects,
- relationships.

The objects are called

**nodes** (or vertices).

The relationships are called

**edges**.

Mathematically,

a graph is written as

$$
G=(V,E)
$$

where

- $G$ denotes the graph,
- $V$ is the set of vertices,
- $E$ is the set of edges.

For example,

consider three atoms.

```text
Carbon

Hydrogen

Oxygen
```

Suppose carbon is bonded to both hydrogen and oxygen.

The corresponding graph becomes

```text
H

 \

  C

 /

O
```

Here,

the nodes are

$$
V=\{C,H,O\}
$$

The edges are

$$
E=\{(C,H),(C,O)\}
$$

Notice that the graph contains only connectivity.

Nothing has yet been said about

- atomic number,
- electronegativity,
- bond distance,
- crystal symmetry.

Those quantities will later become **attributes**.

---

# 15.4.3 Why Are Graphs Ideal for Materials?

Graphs are extraordinarily well suited for describing materials because materials are fundamentally collections of interacting atoms.

A crystal naturally consists of

```text
Atoms

↓

Interactions

↓

Crystal
```

This is exactly the same structure as

```text
Nodes

↓

Edges

↓

Graph
```

The correspondence is almost perfect.

| Materials Science | Graph Theory |
|-------------------|--------------|
| Atom | Node |
| Bond | Edge |
| Crystal | Graph |
| Material Property | Graph Label |

This correspondence allows us to translate physical systems directly into mathematical graphs.

No handcrafted descriptors are required.

Instead,

the crystal itself becomes the input.

---

# 15.4.4 Why Not Use Images Instead?

A natural question arises.

Could we simply convert crystal structures into images and use convolutional neural networks?

The answer is generally **no**.

Images possess a very regular structure.

Each pixel always has neighbors located in fixed positions.

For example,

```text
□ □ □

□ □ □

□ □ □
```

Every interior pixel has

- one pixel above,
- one below,
- one left,
- one right,
- four diagonal neighbors.

The neighborhood is fixed.

Crystals are fundamentally different.

One atom may have

4 neighbors.

Another

6.

Another

12.

Another

14.

The neighborhood size is not constant.

Graph neural networks naturally handle this irregular connectivity.

Traditional convolutional neural networks do not.

---

# 15.4.5 Graphs Preserve Physical Connectivity

Consider two crystals containing exactly the same atoms.

Crystal A

```text
A —— B —— C
```

Crystal B

```text
A

|

B

|

C
```

Both crystals contain identical atomic compositions.

Traditional feature vectors might produce nearly identical descriptors.

Graphs immediately distinguish them because

their connectivity differs.

Graph representations therefore preserve structural information that would otherwise be lost.

This ability to encode topology is one of the primary reasons graph neural networks have transformed computational materials science.

---

# 15.4.6 The Evolution of Graph Representation

The simplest graph contains only

- nodes,
- edges.

Mathematically,

$$
G=(V,E)
$$

However,

this representation is insufficient for scientific applications.

Suppose we know two atoms are connected.

Questions immediately arise.

Which elements are they?

How far apart are they?

What type of bond connects them?

Is the graph describing

- graphite,
- diamond,
- silicon,
- germanium?

Connectivity alone cannot answer these questions.

We therefore enrich the graph.

Instead of merely storing nodes,

each node stores information.

Instead of merely storing edges,

each edge also stores information.

The graph evolves into

$$
G=(V,E)
$$

where

each node possesses attributes

and

each edge possesses attributes.

Finally,

Graph Networks introduce one additional component.

Global information.

The graph now becomes

$$
G=(V,E,U)
$$

where

- $V$ represents node attributes,
- $E$ represents edge attributes,
- $U$ represents global graph attributes.

This seemingly small modification fundamentally changes the expressive power of the graph.

Instead of representing only local atomic environments,

the graph can now also represent information describing the **entire physical system**.

This is the defining idea behind the Graph Network framework and the conceptual leap that distinguishes MEGNet from CGCNN.

In the following sections, we will examine each of these three components—nodes, edges, and global state—in depth, derive their mathematical representations, discuss their physical interpretation in crystalline materials, and then implement them step by step in PyTorch before constructing the complete MEGNet architecture.

# 15.4.7 From Mathematical Graphs to Scientific Graphs

At this point, we know that a graph consists of

$$
G=(V,E,U)
$$

However, this notation is still too abstract to be useful.

When a mathematician writes

$$
G=(V,E)
$$

they usually mean

- a collection of vertices,
- a collection of edges.

Nothing more.

A graph in pure mathematics does not know

- what an atom is,
- what a bond is,
- what energy is,
- what temperature is.

It simply represents connectivity.

Machine learning, however, is fundamentally different.

A neural network cannot learn useful representations from connectivity alone.

The network needs **numerical information**.

Therefore, every object inside the graph must contain data.

This leads to one of the most important ideas in modern graph neural networks.

> **Graphs are no longer collections of points and lines. They become collections of learnable feature vectors.**

This idea completely changes how we think about graphs.

Instead of

```text
Node
```

we now have

```text
Node

↓

Feature Vector
```

Instead of

```text
Edge
```

we have

```text
Edge

↓

Feature Vector
```

Instead of

```text
Entire Graph
```

we have

```text
Graph

↓

Global Feature Vector
```

Graph neural networks never operate directly on atoms or bonds.

They operate on **vectors** describing atoms and bonds.

Understanding these vectors is one of the keys to understanding MEGNet.

---

# 15.4.8 Nodes Are Feature Vectors

Consider a silicon atom.

To a human,

we immediately recognize

```text
Element

↓

Silicon
```

A neural network does not.

The computer only understands numbers.

Therefore,

every atom must be converted into a numerical representation.

Suppose we choose four atomic properties.

| Property | Symbol |
|-----------|---------|
| Atomic Number | $Z$ |
| Group | $G$ |
| Period | $P$ |
| Electronegativity | $\chi$ |

Silicon becomes

$$
[14,\;14,\;3,\;1.90]
$$

Carbon becomes

$$
[6,\;14,\;2,\;2.55]
$$

Oxygen becomes

$$
[8,\;16,\;2,\;3.44]
$$

Each atom therefore becomes a vector.

Instead of writing

```text
Si
```

the neural network sees

$$
\mathbf{v}_i=
\begin{bmatrix}
14\\
14\\
3\\
1.90
\end{bmatrix}
$$

This vector is called the **node feature vector**.

The graph therefore stores

```text
Atom

↓

Node Feature Vector
```

instead of simply storing

```text
Atom Name
```

---

# 15.4.9 What Can Be Stored Inside Node Features?

There is no universal choice.

Different researchers include different information depending on the scientific problem.

Typical atomic descriptors include

- atomic number,
- atomic mass,
- electronegativity,
- electron affinity,
- ionization energy,
- covalent radius,
- van der Waals radius,
- valence electrons,
- oxidation state,
- periodic table group,
- periodic table period,
- atomic volume,
- magnetic moment.

Some datasets use

20 features.

Others use

90.

Some modern graph neural networks even begin with only the atomic number and allow the network to learn everything else.

This flexibility is one reason graph neural networks are so powerful.

---

# 15.4.10 Example: Building Node Features in Python

Before using PyTorch,

let us first build node feature vectors using ordinary Python.

```python
import numpy as np

# Example atomic information

silicon = {
    "atomic_number": 14,
    "group": 14,
    "period": 3,
    "electronegativity": 1.90
}

carbon = {
    "atomic_number": 6,
    "group": 14,
    "period": 2,
    "electronegativity": 2.55
}

oxygen = {
    "atomic_number": 8,
    "group": 16,
    "period": 2,
    "electronegativity": 3.44
}

def atom_to_vector(atom):

    return np.array([
        atom["atomic_number"],
        atom["group"],
        atom["period"],
        atom["electronegativity"]
    ], dtype=float)

print(atom_to_vector(silicon))
print(atom_to_vector(carbon))
print(atom_to_vector(oxygen))
```

Output

```text
[14.   14.    3.    1.90]

[ 6.   14.    2.    2.55]

[ 8.   16.    2.    3.44]
```

Although this example is extremely simple,

it demonstrates an important principle.

The neural network never sees

```text
Si
```

It only receives

numbers.

---

# 15.4.11 Why Raw Atomic Properties Are Often Not Enough

Suppose two researchers choose different node features.

Researcher A uses

```text
Atomic Number
```

Researcher B uses

```text
Atomic Number

Atomic Radius

Electronegativity

Ionization Energy

Electron Affinity
```

Which representation is better?

The answer is

neither.

Modern graph neural networks rarely use these raw vectors directly.

Instead,

they transform them into **learnable embeddings**.

The raw feature vector

$$
\mathbf{x}
$$

is passed through a neural network

$$
\mathbf{h}=f(\mathbf{x})
$$

where

$f$

is usually a multilayer perceptron.

The network learns which combinations of atomic properties are most useful for the prediction task.

Consequently,

the initial atomic vector is only the starting point.

The learned representation eventually becomes much richer.

Later in this chapter,

we will implement these embedding layers from scratch.

---

# 15.4.12 Edge Features

Nodes describe atoms.

Edges describe the relationships between atoms.

Suppose two atoms are bonded.

```text
C —— O
```

Connectivity alone is insufficient.

Many questions immediately arise.

How far apart are the atoms?

What is the bond order?

Is the bond covalent?

Metallic?

Ionic?

Hydrogen?

Every bond therefore receives its own feature vector.

Mathematically,

the edge connecting

atom $i$

to

atom $j$

is represented as

$$
\mathbf{e}_{ij}
$$

Unlike nodes,

edge vectors describe relationships rather than objects.

Typical edge information includes

- bond length,
- bond type,
- bond direction,
- interatomic distance,
- coordination relationship,
- Gaussian basis expansion of distance.

Later,

we will see that MEGNet relies heavily on edge updates.

In fact,

the first operation inside every MEGNet block updates the edges before updating the nodes.

This design choice is one of the major differences between MEGNet and many earlier graph neural networks.

---

# 15.4.13 Example: Constructing Edge Features

Suppose we have

```text
Si —— O
```

with

distance

$$
1.62\;\AA
$$

A simple edge feature vector might be

$$
\mathbf{e}_{ij}
=
[1.62]
$$

If additional information is available,

the vector may become

$$
\mathbf{e}_{ij}
=
[
1.62,
1,
0,
0
]
$$

where

- first entry = bond distance,
- second entry = covalent,
- third entry = ionic,
- fourth entry = metallic.

Different graph neural networks choose different edge representations.

One of the strengths of MEGNet is that its architecture is flexible enough to accommodate many different choices without changing the underlying mathematical framework.

# 15.4.14 Why Distance Alone Is Not Sufficient

At first glance, it may appear that the distance between two atoms contains all the information necessary to describe their interaction.

For example,

consider two pairs of atoms separated by exactly the same distance.

```text
Pair A

Na -------- Cl

Distance = 2.82 Å
```

```text
Pair B

C -------- C

Distance = 2.82 Å
```

Both edges have the same geometric length.

If the graph stores only

$$
\mathbf{e}_{ij}=[2.82]
$$

then the neural network initially sees these two bonds as identical.

From a physical perspective, however, they are completely different.

The sodium–chlorine interaction is predominantly ionic.

The carbon–carbon interaction is predominantly covalent.

Even though the distance is identical,

their

- electron distributions,
- bond strengths,
- orbital interactions,
- mechanical behavior,
- electronic properties,

are fundamentally different.

Therefore,

distance alone cannot uniquely describe an interaction.

This observation motivates richer edge representations.

---

# 15.4.15 Engineering Better Edge Features

Different graph neural networks encode bond information differently.

A simple representation may contain

$$
\mathbf{e}_{ij}
=
[d_{ij}]
$$

where

$d_{ij}$

is the interatomic distance.

A richer representation may include

$$
\mathbf{e}_{ij}
=
[d_{ij},
Z_i,
Z_j]
$$

where

- $Z_i$ is the atomic number of atom $i$,
- $Z_j$ is the atomic number of atom $j$.

An even richer representation may contain

- bond length,
- bond direction,
- periodic image,
- coordination information,
- local symmetry,
- Gaussian basis expansion,
- radial basis expansion.

Modern graph neural networks often avoid manually selecting bond descriptors.

Instead,

they transform simple physical quantities into high-dimensional learned representations.

This philosophy mirrors what we discussed for node features.

Raw physical information becomes the input.

Learned representations become the internal language of the neural network.

---

# 15.4.16 Example: Building Edge Features

Suppose we know

```text
Si -------- O

Distance = 1.62 Å
```

A simple Python implementation might be

```python
import numpy as np

edge_distance = 1.62

edge_feature = np.array([edge_distance], dtype=float)

print(edge_feature)
```

Output

```text
[1.62]
```

Suppose we additionally know

- Silicon atomic number = 14
- Oxygen atomic number = 8

Then

```python
import numpy as np

edge_feature = np.array([
    1.62,
    14,
    8
], dtype=float)

print(edge_feature)
```

Output

```text
[ 1.62 14.00 8.00 ]
```

Although this representation is still extremely simple,

it illustrates an important principle.

Edges,

just like nodes,

are numerical feature vectors.

---

# 15.4.17 Global Features

We have now discussed

- nodes,
- edges.

The final component introduced by the Graph Network framework is something entirely new.

The **global feature vector**.

This is the defining idea that distinguishes Graph Networks—and therefore MEGNet—from many earlier graph neural network architectures.

Imagine performing two experiments on the same crystal.

Experiment 1

```text
Temperature = 300 K

Pressure = 1 atm
```

Experiment 2

```text
Temperature = 1500 K

Pressure = 25 GPa
```

The crystal may initially contain exactly the same atoms connected by exactly the same bonds.

Therefore,

the node features are identical.

The edge features are identical.

Yet,

the material behaves differently.

For example,

its

- free energy,
- entropy,
- phase stability,
- diffusion coefficient,
- lattice expansion,

may all change.

Where should this information be stored?

Neither

- nodes,

nor

- edges,

describe properties of the entire system.

The Graph Network framework therefore introduces a third object

$$
\mathbf{u}
$$

called the **global feature vector**.

Instead of belonging to one atom,

or one bond,

it belongs to the graph itself.

---

# 15.4.18 Physical Meaning of the Global State

The global feature vector represents information shared by every part of the graph.

Examples include

- temperature,
- pressure,
- external electric field,
- magnetic field,
- synthesis conditions,
- pH,
- total system charge,
- simulation timestep,
- experimental conditions.

Unlike node features,

which describe individual atoms,

or edge features,

which describe pairwise interactions,

the global state describes the physical environment of the entire material.

This is one of the most important conceptual advances introduced by the Graph Network framework.

Instead of treating environmental variables separately,

they become part of the graph itself.

---

# 15.4.19 Example Global Feature Vector

Suppose a material is studied under

```text
Temperature = 300 K

Pressure = 1 atm
```

The global vector could be

$$
\mathbf{u}
=
[300,\;1]
$$

If additional information is available,

we may instead use

$$
\mathbf{u}
=
[
300,
1,
0,
7
]
$$

where

the entries represent

- temperature,
- pressure,
- magnetic field,
- pH.

There is no universal definition.

The global vector is designed according to the scientific problem.

---

# 15.4.20 Why the Global State Is So Powerful

Suppose we want to predict

the Gibbs free energy

of a material.

From thermodynamics,

we know

$$
G
=
G(T,P)
$$

The free energy depends explicitly on

- temperature,
- pressure.

If the neural network receives only

- atoms,
- bonds,

it has no way to distinguish

```text
300 K
```

from

```text
1200 K
```

because the graph appears identical.

Adding

$$
\mathbf{u}
$$

solves this problem.

Now,

every message-passing step can incorporate

global thermodynamic information.

This makes the graph much more expressive.

Instead of learning only

$$
f(V,E)
$$

the model learns

$$
f(V,E,U)
$$

This seemingly minor change greatly expands the range of physical phenomena that can be modeled.

---

# 15.4.21 The Complete Graph Representation

We can now combine everything we have learned.

A Graph Network graph consists of three interacting components.

```text
                    Graph

                      │

      ┌───────────────┼────────────────┐

      │               │                │

   Nodes           Edges          Global State

      │               │                │

 Atom Features   Bond Features   System Features

      │               │                │

 Atomic No.      Distance         Temperature

 Radius          Bond Type        Pressure

 Group           Connectivity     External Field

 Electrons       Geometry         Total Charge

 ...
```

This representation is considerably richer than the simple graph

$$
G=(V,E)
$$

introduced in elementary graph theory.

Instead,

every component of the graph carries meaningful physical information.

Graph neural networks operate on these feature vectors,

continually updating them during message passing until they become increasingly informative representations of the underlying material.

---

# 15.4.22 Formal Mathematical Definition of a Graph Network

We are now ready to write the complete mathematical definition.

A Graph Network graph is represented as

$$
G=(V,E,U)
$$

where

$$
V=
\{
\mathbf{v}_1,
\mathbf{v}_2,
\ldots,
\mathbf{v}_{N_v}
\}
$$

is the collection of node feature vectors.

Similarly,

$$
E=
\{
(\mathbf{e}_k,r_k,s_k)
\}_{k=1}^{N_e}
$$

is the collection of edge feature vectors.

Each edge stores

- its feature vector $\mathbf{e}_k$,
- its receiver node $r_k$,
- its sender node $s_k$.

Finally,

$$
U=\mathbf{u}
$$

represents the global feature vector.

Notice something important.

Edges are **not** defined only by their features.

They must also know

- where they start,
- where they end.

Without this information,

message passing would be impossible.

This seemingly simple observation forms the mathematical foundation of every Graph Network computation that follows.

In the next part of this section, we will move from graph representation to **graph computation**, introducing the concepts of update functions, aggregation functions, and the complete computational cycle that powers every MEGNet block.


# 15.4.23 Understanding the Mathematical Representation of a Graph Network

The mathematical notation

$$
G=(V,E,U)
$$

appears deceptively simple.

Many readers memorize this equation and move on.

However, every symbol in this equation represents a large amount of information.

To truly understand Graph Networks, we must examine every component in detail.

---

## The Node Set

Recall that

$$
V=
\{
\mathbf{v}_1,
\mathbf{v}_2,
\ldots,
\mathbf{v}_{N_v}
\}
$$

This notation tells us that the graph contains

$$
N_v
$$

nodes.

Each node has its own feature vector.

Suppose we have the following crystal fragment.

```text
Si -------- O

|           |

O -------- Si
```

There are four atoms.

Therefore,

$$
N_v=4
$$

The node feature matrix becomes

$$
V=
\{
\mathbf{v}_1,
\mathbf{v}_2,
\mathbf{v}_3,
\mathbf{v}_4
\}
$$

Notice something important.

The nodes do **not** need to have different feature dimensions.

Every node vector has the same length.

For example,

suppose every atom is represented by

- atomic number
- electronegativity
- atomic radius
- valence electrons

Then

$$
\mathbf{v}_i
=
\begin{bmatrix}
Z_i\\
\chi_i\\
r_i\\
n_i
\end{bmatrix}
$$

Every atom now has a four-dimensional feature vector.

If later we decide to include

- ionization energy,
- electron affinity,
- oxidation state,

then every node vector becomes larger.

The important point is that **all nodes share the same feature dimension**.

If

$$
d_v=64
$$

then every node embedding has dimension

64.

The graph may contain

10 atoms,

100 atoms,

or

100000 atoms,

but every node embedding still has dimension

64.

---

## Node Feature Matrix

Instead of writing node vectors individually,

machine learning implementations usually store them inside one large matrix.

Suppose

we have

five atoms,

each described using

four features.

The feature matrix becomes

$$
X
=
\begin{bmatrix}
14 & 1.90 & 1.11 & 4\\
8 & 3.44 & 0.66 & 6\\
8 & 3.44 & 0.66 & 6\\
14 & 1.90 & 1.11 & 4\\
8 & 3.44 & 0.66 & 6
\end{bmatrix}
$$

The dimensions are

$$
N_v
\times
d_v
$$

where

- $N_v$ is the number of atoms,

- $d_v$ is the node feature dimension.

In deep learning,

this matrix is usually stored as a tensor.

---

## PyTorch Representation

Suppose our graph contains

five atoms

and

each atom has

four features.

The corresponding tensor is

```python
import torch

node_features = torch.tensor([
    [14.0, 1.90, 1.11, 4],
    [8.0, 3.44, 0.66, 6],
    [8.0, 3.44, 0.66, 6],
    [14.0, 1.90, 1.11, 4],
    [8.0, 3.44, 0.66, 6]
])

print(node_features.shape)
```

Output

```text
torch.Size([5,4])
```

The first dimension

corresponds to

the number of atoms.

The second dimension

corresponds to

the number of features describing each atom.

This representation is used by nearly every graph neural network library,

including

- PyTorch Geometric,
- DGL,
- Spektral.

---

# 15.4.24 Understanding the Edge Set

The edge set is slightly more complicated.

Recall

$$
E=
\{
(\mathbf{e}_k,r_k,s_k)
\}_{k=1}^{N_e}
$$

Most beginners focus only on

$$
\mathbf{e}_k
$$

and ignore

$$
r_k
$$

and

$$
s_k.
$$

This is a serious mistake.

An edge has

three components.

First,

its feature vector.

Second,

its sender.

Third,

its receiver.

Without the sender and receiver,

the neural network would have no idea where messages should travel.

---

## Example

Suppose

```text
Atom 0 -------- Atom 1
```

The edge

contains

```text
Sender = Atom 0

Receiver = Atom 1

Distance = 2.31 Å
```

Mathematically,

$$
(
\mathbf{e},
r,
s
)
=
(
2.31,
1,
0
)
$$

Notice that

the edge knows

- what it contains,

and

- where it goes.

This information becomes crucial during message passing.

---

## Why Are Sender and Receiver Needed?

Suppose we remove

the sender and receiver information.

The graph now stores only

```text
2.31 Å

1.67 Å

2.12 Å

2.54 Å
```

What do these distances describe?

Which atoms are connected?

Which direction should information flow?

The neural network cannot answer these questions.

Therefore,

every edge must explicitly specify

its two endpoints.

---

# 15.4.25 Edge Index Representation

Instead of storing edges as mathematical tuples,

PyTorch Geometric stores connectivity inside a matrix called

```text
edge_index
```

Suppose our graph is

```text
0 -------- 1

|

2
```

Node

0

is connected to

1

and

2.

The edge index becomes

```python
edge_index = torch.tensor([
    [0,0],
    [1,2]
]).t()
```

Notice the transpose.

Printing

```python
print(edge_index)
```

gives

```text
tensor([
 [0,1],
 [0,2]
])
```

Internally,

PyTorch Geometric stores

```python
edge_index = edge_index.t().contiguous()
```

giving

```text
tensor([
 [0,0],
 [1,2]
])
```

The first row

contains

senders.

The second row

contains

receivers.

Graph libraries prefer this representation because

it enables efficient parallel computation on GPUs.

---

# 15.4.26 Complete Example

Consider

the following crystal fragment.

```text
0 -------- 1

|          |

2 -------- 3
```

The graph contains

four nodes.

Each atom has

three features.

Suppose

```python
node_features = torch.tensor([
    [14,1.9,4],
    [8,3.44,6],
    [8,3.44,6],
    [14,1.9,4]
],dtype=torch.float)
```

Connectivity

```python
edge_index = torch.tensor([
    [0,0,1,2],
    [1,2,3,3]
],dtype=torch.long)
```

Distances

```python
edge_features = torch.tensor([
    [2.10],
    [2.25],
    [2.18],
    [2.31]
])
```

Now

the graph contains

everything needed for message passing.

Each edge knows

- where it starts,

- where it ends,

- what information it carries.

Each node knows

its own atomic properties.

Nothing else is required to begin graph computation.

---

# 15.4.27 The Global Feature Vector Revisited

Earlier,

we introduced

the global feature vector

$$
\mathbf{u}
$$

only conceptually.

Now,

let us examine it mathematically.

Unlike

nodes

or

edges,

there is

only

one

global feature vector

for the entire graph.

Suppose

our prediction depends upon

temperature

pressure

and

magnetic field.

The global vector becomes

$$
\mathbf{u}
=
\begin{bmatrix}
300\\
1\\
0
\end{bmatrix}
$$

Every node

can access

this same vector.

Every edge

can also access

this same vector.

This means

global information can influence

every message passing operation.

This is one of the defining ideas behind the Graph Network framework.

Later,

when we derive the edge update equation,

you will see that

the global feature vector directly participates in computing new edge embeddings.

Unlike CGCNN,

global physics is therefore integrated into every layer of the network.

---

At this point, we have fully described the **static structure** of a Graph Network. We know what information is stored in nodes, edges, and the global state, and how these are represented mathematically and in code. The next step is to answer the central question of Graph Networks:

> **How does information actually flow through this graph?**

To answer that, we will introduce the three core computational components of every Graph Network:

1. **Update functions ($\phi$)** — which learn how to transform nodes, edges, and the global state.
2. **Aggregation functions ($\rho$)** — which combine information from multiple graph elements while preserving permutation invariance.
3. **The complete Graph Network computational cycle** — the sequence of edge updates, node updates, and global updates that forms the foundation of every MEGNet block.

# 15.4.28 Graph Computation Begins

Up to this point, our graph has been nothing more than a sophisticated data structure.

It stores

- atoms,
- bonds,
- global information,

but **nothing has been computed yet**.

A graph without computation is like a textbook that has never been read.

It contains information,

but that information has not yet been transformed into knowledge.

Graph Neural Networks are fundamentally **computational systems**.

Their purpose is not merely to store graphs.

Their purpose is to **transform** graphs.

This transformation occurs repeatedly.

Initially,

the graph contains raw physical information.

```text
Atomic Number

↓

Electronegativity

↓

Bond Distance

↓

Temperature
```

After one Graph Network block,

these raw quantities become richer representations.

After several Graph Network blocks,

the graph no longer represents simple physical quantities.

Instead,

it contains **learned latent representations** that encode highly complex chemical and structural information.

Understanding how this transformation occurs is the central goal of the Graph Network framework.

---

# 15.4.29 Computation as Repeated State Transformation

Suppose the graph at layer

$l$

is written as

$$
G^{(l)}
=
(V^{(l)},E^{(l)},U^{(l)})
$$

Initially,

the graph contains

$$
G^{(0)}
$$

where

the node features are simply

raw atomic properties,

the edge features are

raw bond descriptors,

and the global feature vector contains

physical state variables.

The Graph Network applies one computational block,

producing

$$
G^{(1)}
$$

Another block produces

$$
G^{(2)}
$$

Another

$$
G^{(3)}
$$

Eventually,

after

$L$

blocks,

we obtain

$$
G^{(L)}
$$

whose node, edge, and global representations have become highly informative.

Notice something important.

The graph itself does not change.

The connectivity remains identical.

Only

the feature vectors evolve.

This is an essential idea.

Graph Networks do **not**

continually rebuild the graph.

They continually update the **information stored inside the graph**.

---

# 15.4.30 A Simple Analogy

Imagine a classroom.

Each student possesses

some initial knowledge.

```text
Student A

↓

Knows Mathematics
```

```text
Student B

↓

Knows Physics
```

```text
Student C

↓

Knows Chemistry
```

Suppose they begin discussing a research problem.

Student A shares mathematics.

Student B contributes physics.

Student C contributes chemistry.

After one discussion,

every student knows more than before.

Another discussion occurs.

Knowledge continues to spread.

Eventually,

every student possesses a much richer understanding.

Graph Networks operate in exactly this manner.

Nodes exchange information.

Edges exchange information.

The global state exchanges information.

Knowledge gradually propagates through the graph.

This process is called

**message passing**.

---

# 15.4.31 Why Updating Nodes Alone Is Not Enough

Many early graph neural networks focused almost entirely on updating node representations.

The workflow looked like

```text
Nodes

↓

Neighbor Aggregation

↓

Updated Nodes
```

Edges remained fixed.

The graph itself remained fixed.

MEGNet adopts a fundamentally different philosophy.

It asks

> Why should only atoms learn?

Bonds also contain information.

Furthermore,

the entire crystal contains information.

Therefore,

MEGNet updates

```text
Edges

↓

Nodes

↓

Global State
```

during every Graph Network block.

This idea may appear obvious,

but it significantly increases the expressive power of the model.

---

# 15.4.32 Two Fundamental Operations

Every Graph Network performs only two types of computation.

Everything else is built from these two ideas.

The first operation is

**update**.

The second operation is

**aggregation**.

Understanding these two operations is equivalent to understanding the Graph Network framework.

---

## Update

Suppose an atom currently possesses the representation

$$
\mathbf{v}
$$

After learning from neighboring atoms,

its representation changes.

Mathematically,

$$
\mathbf{v}
\longrightarrow
\mathbf{v}'
$$

The same idea applies to

edges

and

the global state.

Every object inside the graph evolves.

This evolution is called

an **update**.

---

## Aggregation

Suppose one atom has

six neighbors.

Each neighbor sends information.

The receiving atom now has

six different incoming messages.

How should these messages be combined?

Simply concatenating them is impossible,

because another atom may have

twelve neighbors.

The neural network therefore requires a mathematical operation that combines

an arbitrary number of messages

into

one vector.

This operation is called

**aggregation**.

---

# 15.4.33 The Two Families of Functions

The Graph Network framework introduces two categories of functions.

Update functions

are represented by

$$
\phi
$$

Aggregation functions

are represented by

$$
\rho
$$

These symbols appear throughout the original Graph Networks paper.

They are not arbitrary notation.

Each represents an entire class of neural network operations.

---

## Update Functions

The symbol

$$
\phi
$$

means

> "Learn how to transform information."

There are three update functions.

Edge update

$$
\phi_e
$$

Node update

$$
\phi_v
$$

Global update

$$
\phi_u
$$

Each update function is usually implemented using

a multilayer perceptron (MLP).

Later,

we will implement every one of these from scratch.

---

## Aggregation Functions

The symbol

$$
\rho
$$

means

> "Combine many vectors into one vector."

Examples include

sum

$$
\sum
$$

mean

$$
\frac{1}{N}\sum
$$

maximum

$$
\max
$$

Later,

we will discuss why these functions must satisfy an important mathematical property called

**permutation invariance**.

---

# 15.4.34 Why Separate Update and Aggregation?

A beginner may wonder

Why not simply use one neural network?

Why introduce two different operations?

The reason becomes obvious when we examine what each operation accomplishes.

Aggregation performs

no learning.

For example,

suppose three neighboring atoms send messages.

```text
2

5

8
```

The aggregation function

may compute

their sum.

```text
2 + 5 + 8 = 15
```

No parameters were learned.

The operation is purely mathematical.

An update function,

however,

contains trainable parameters.

Suppose the aggregated message is

15.

A neural network transforms

15

into

a richer representation.

Unlike aggregation,

this transformation is learned during training.

Therefore,

aggregation

collects information.

Update

interprets information.

The distinction is extremely important.

---

# 15.4.35 The Complete Computational Pipeline

Every Graph Network block follows the same high-level workflow.

```text
Current Graph

↓

Update Edges

↓

Aggregate Edge Information

↓

Update Nodes

↓

Aggregate Node Information

↓

Update Global State

↓

Updated Graph
```

This sequence is repeated

again

and

again

throughout the network.

Notice that

information flows upward.

Edges are updated first.

Nodes use the updated edges.

The global state then uses the updated nodes and updated edges.

The ordering is deliberate.

Later,

we will derive the mathematical justification for this sequence.

---

# 15.4.36 Why Edge Updates Come First

This is one of the most elegant design choices in the Graph Network framework.

Consider

```text
Si -------- O
```

Suppose

the bond between

Si

and

O

changes.

Should the atoms know this?

Obviously,

yes.

If the bond becomes stronger,

the neighboring atoms should receive updated information.

Therefore,

before updating atoms,

the graph first updates

the bond itself.

Only after

the edge contains new information

do the atoms read it.

This mirrors physical intuition.

Interactions change first.

Atoms respond to those interactions.

The Graph Network framework faithfully captures this process.

---

At this stage we have established the overall computational philosophy of Graph Networks. We now know **what** update functions and aggregation functions are and **why** both are necessary. The next step is to derive them mathematically, beginning with the **edge update function ($\phi_e$)**, which is the first learnable operation performed inside every MEGNet block.

# 15.4.37 The Edge Update Function ($\phi_e$)

We now arrive at the first learnable operation in the Graph Network framework.

The **edge update**.

Although many introductions to Graph Neural Networks begin by updating nodes, the Graph Network framework deliberately starts with the edges.

This is not an arbitrary implementation choice.

It reflects an important physical observation.

> **Atoms interact through bonds. Therefore, before updating the atoms themselves, we should first update the interactions between them.**

This simple idea distinguishes Graph Networks from many earlier GNN architectures.

To appreciate why this is important, let us examine the process carefully.

---

# 15.4.38 What Does an Edge Represent?

An edge is often introduced as

> "A connection between two nodes."

Although this definition is mathematically correct,

it is physically incomplete.

In materials science,

an edge represents **an interaction**.

Depending on the problem,

the interaction may correspond to

- chemical bonding,
- electrostatic interaction,
- neighbor relationship,
- interatomic potential,
- atomic communication pathway.

Therefore,

an edge should not be viewed merely as a line joining two atoms.

Instead,

it is an object possessing its own information and its own learnable representation.

Consider

```text
Si ---------------- O
```

The edge may initially contain

- distance,
- bond order,
- periodic image,
- bond direction,
- radial basis expansion.

Initially,

its feature vector may be

$$
\mathbf{e}_{ij}
=
\begin{bmatrix}
2.35
\end{bmatrix}
$$

or perhaps

$$
\mathbf{e}_{ij}
=
\begin{bmatrix}
2.35\\
1\\
0\\
0
\end{bmatrix}
$$

After one Graph Network layer,

this vector no longer simply represents

distance.

Instead,

it becomes a learned latent representation describing the interaction between the two atoms.

---

# 15.4.39 Why Should Edges Learn?

Suppose we have

```text
Carbon -------- Carbon
```

Initially,

the edge knows only

```text
Bond Length = 1.54 Å
```

After training,

the neural network may discover

that

this particular interaction is characteristic of

- diamond,
- high stiffness,
- large elastic modulus.

The edge representation gradually evolves.

Instead of storing

one number,

it eventually stores

dozens

or even

hundreds

of learned features.

For example,

after several Graph Network layers,

the edge embedding may become

$$
\mathbf{e}_{ij}
=
[
-0.42,
0.73,
1.18,
-0.91,
\ldots
]
$$

These numbers have no direct physical interpretation individually.

Collectively,

however,

they encode highly sophisticated information learned from data.

This is one of the defining characteristics of deep learning.

The network invents its own internal representation.

---

# 15.4.40 What Information Is Needed to Update an Edge?

Imagine two atoms

```text
Atom i -------- Atom j
```

Can we update the bond using only

its previous bond feature?

Probably not.

Suppose

Atom

$i$

changes dramatically.

The bond should also change.

Similarly,

if

Atom

$j$

changes,

the bond should again change.

Furthermore,

suppose

temperature increases.

Bond behavior may also change.

Therefore,

the edge update should depend upon

four sources of information.

1. The current edge.

2. The sender atom.

3. The receiver atom.

4. The global state.

Graphically,

```text
Sender Node

      │

      ▼

Current Edge -----> Updated Edge

      ▲

      │

Receiver Node

      ▲

      │

Global State
```

This diagram captures one of the central ideas of the Graph Network framework.

The updated edge is influenced by

both local information

(nodes)

and

global information.

---

# 15.4.41 Mathematical Form of the Edge Update

Suppose

edge

$k$

connects

sender node

$s_k$

to

receiver node

$r_k$.

The Graph Network defines the updated edge as

$$
\mathbf{e}_k'
=
\phi_e
(
\mathbf{e}_k,
\mathbf{v}_{r_k},
\mathbf{v}_{s_k},
\mathbf{u}
)
$$

This equation deserves careful study.

The update function

$\phi_e$

receives

four inputs.

Current edge

$$
\mathbf{e}_k
$$

Receiver node

$$
\mathbf{v}_{r_k}
$$

Sender node

$$
\mathbf{v}_{s_k}
$$

Global state

$$
\mathbf{u}
$$

and produces

a new edge representation

$$
\mathbf{e}_k'
$$

Notice something remarkable.

The edge no longer depends solely on itself.

Instead,

it learns by combining

its own information,

the information stored in both connected atoms,

and

the state of the entire graph.

This is a much richer update mechanism than simply updating edges using bond distances alone.

---

# 15.4.42 Why Include Both Sender and Receiver?

A common question is

Why include

both

the sender

and

the receiver?

Wouldn't one atom be sufficient?

The answer depends on the physical problem.

Consider

```text
Na -------- Cl
```

Suppose we swap the atoms.

```text
Cl -------- Na
```

In an undirected graph,

these represent the same bond.

However,

many graph computations internally treat edges as directed.

Information flows

from

one node

to

another.

The edge update therefore needs access to

both endpoint representations.

Even when the graph is undirected,

most graph libraries internally store

two directed edges.

For example,

```text
0 -------- 1
```

is represented as

```text
0 → 1

1 → 0
```

This allows message passing to proceed efficiently.

---

# 15.4.43 Why Include the Global State?

Suppose

temperature changes.

The atoms remain identical.

The bond length remains identical.

Yet

the interaction energy changes.

If the edge update ignored

the global state,

the neural network would be unable to distinguish

```text
300 K
```

from

```text
1500 K.
```

Including

$$
\mathbf{u}
$$

solves this problem.

The updated edge now becomes

a function of

both

local

and

global

physics.

This is one of the major innovations introduced by the Graph Network framework and inherited by MEGNet.

---

# 15.4.44 Implementing the Edge Update Function

The Graph Network framework does not specify how

$\phi_e$

should be implemented.

It simply specifies

what information it receives.

In practice,

MEGNet implements

$\phi_e$

using a multilayer perceptron.

The first step is to concatenate

all input vectors.

Suppose

```text
Edge Dimension = 16

Node Dimension = 64

Global Dimension = 32
```

The concatenated vector has dimension

$$
16
+
64
+
64
+
32
=
176.
$$

The neural network then transforms

this

176-dimensional vector

into

a new edge embedding.

---

## Step 1 — Concatenate Inputs

```python
import torch

edge = torch.randn(16)

sender = torch.randn(64)

receiver = torch.randn(64)

global_state = torch.randn(32)

edge_input = torch.cat([
    edge,
    sender,
    receiver,
    global_state
])

print(edge_input.shape)
```

Output

```text
torch.Size([176])
```

This concatenated vector contains

every piece of information needed to update the edge.

---

## Step 2 — Build the Edge MLP

```python
import torch
import torch.nn as nn

class EdgeUpdate(nn.Module):

    def __init__(
        self,
        edge_dim,
        node_dim,
        global_dim,
        hidden_dim
    ):

        super().__init__()

        input_dim = (
            edge_dim
            + 2 * node_dim
            + global_dim
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                edge_dim
            )
        )

    def forward(
        self,
        edge,
        sender,
        receiver,
        global_state
    ):

        x = torch.cat([
            edge,
            sender,
            receiver,
            global_state
        ])

        return self.mlp(x)
```

Notice that the output dimension is

`edge_dim`.

The edge is **updated**, not replaced by a completely different type of object.

Its representation changes,

but it remains an edge feature vector.

---

## Step 3 — Running the Edge Update

```python
edge_update = EdgeUpdate(
    edge_dim=16,
    node_dim=64,
    global_dim=32,
    hidden_dim=128
)

updated_edge = edge_update(
    edge,
    sender,
    receiver,
    global_state
)

print(updated_edge.shape)
```

Output

```text
torch.Size([16])
```

The network has transformed

the original edge embedding

into

a new learned edge embedding.

At this point,

we have implemented the first learnable component of the Graph Network framework.

However,

this implementation updates **only one edge at a time**.

A real crystal may contain

thousands

or even

millions

of edges.

Updating edges one by one would be computationally impractical.

In the next part, we will develop the **vectorized edge update**, showing how all edges in an entire graph are updated simultaneously using tensor operations and how this formulation naturally integrates with PyTorch Geometric's message-passing framework.

# 15.4.45 Why the Previous Implementation Is Not Enough

In the previous section, we implemented the edge update function for **a single edge**.

Although this implementation clearly illustrates the mathematics, it is **not** how modern graph neural networks are actually implemented.

Consider a realistic crystal.

Suppose a crystal contains

- 128 atoms.

If each atom has approximately

12 nearest neighbors,

the graph already contains approximately

$$
\frac{128\times12}{2}
=
768
$$

undirected edges.

Since PyTorch Geometric internally represents undirected edges using **two directed edges**, the graph actually contains

$$
1536
$$

directed edges.

Updating every edge individually would require

```python
for edge in edges:

    updated_edge = edge_update(...)
```

This approach has several major problems.

---

## Problem 1 — Python Loops Are Slow

Python is an interpreted language.

Every iteration introduces overhead.

Suppose our graph contains

```text
1,500 edges
```

The loop executes

1,500 times.

Now suppose we process a batch of

32 crystals.

The total number of edge updates becomes

```text
48,000
```

Now consider a dataset containing

100,000 crystals.

The computational cost quickly becomes enormous.

Modern deep learning avoids Python loops whenever possible.

---

## Problem 2 — GPUs Prefer Matrix Operations

Modern GPUs are designed to perform

millions

or

billions

of arithmetic operations simultaneously.

They are **not** optimized for repeatedly executing small Python functions.

Instead,

they achieve maximum performance when computations are expressed as

large tensor operations.

For example,

instead of computing

```python
edge1

edge2

edge3

...

edge1500
```

one after another,

the GPU performs

```text
Edge Matrix

↓

Neural Network

↓

Updated Edge Matrix
```

Everything happens simultaneously.

This is called **vectorization**.

---

## Problem 3 — Automatic Parallelization

Suppose each edge update requires

200 floating-point operations.

Updating

1500 edges

requires

$$
1500\times200
=
300000
$$

operations.

A GPU can perform these operations simultaneously because

every edge update is mathematically independent.

Instead of

```text
Edge 1

↓

Edge 2

↓

Edge 3

↓

...
```

the GPU performs

```text
Edge 1

Edge 2

Edge 3

...

Edge 1500

↓

Updated Together
```

This is one of the reasons graph neural networks scale efficiently to large systems.

---

# 15.4.46 Representing All Edges as Matrices

Instead of storing

one edge,

we store

all edges together.

Suppose

our graph contains

four edges.

Each edge has

three features.

The edge feature matrix becomes

$$
E
=
\begin{bmatrix}
2.10 & 1 & 0\\
2.25 & 1 & 0\\
1.95 & 0 & 1\\
2.30 & 1 & 0
\end{bmatrix}
$$

Its dimensions are

$$
N_e
\times
d_e
$$

where

- $N_e$ is the number of edges,

- $d_e$ is the edge feature dimension.

Notice the similarity with the node feature matrix.

Every graph component is represented using matrices.

This makes tensor computation possible.

---

# 15.4.47 Sender and Receiver Features for Every Edge

Recall the edge update equation

$$
\mathbf{e}_k'
=
\phi_e
(
\mathbf{e}_k,
\mathbf{v}_{r_k},
\mathbf{v}_{s_k},
\mathbf{u}
)
$$

The challenge is

how do we obtain

the sender and receiver features

for every edge

without writing loops?

The answer lies in **tensor indexing**.

Suppose

our graph contains

four atoms.

```text
Node 0

Node 1

Node 2

Node 3
```

The node feature matrix is

```python
node_features = torch.tensor([
    [14.0,1.90],
    [8.0,3.44],
    [8.0,3.44],
    [14.0,1.90]
])
```

Now suppose the graph connectivity is

```text
0 → 1

0 → 2

1 → 3

2 → 3
```

The edge index becomes

```python
edge_index = torch.tensor([
    [0,0,1,2],
    [1,2,3,3]
])
```

The first row stores

the sender indices.

The second row stores

the receiver indices.

---

# 15.4.48 Extracting Sender Features

PyTorch allows indexing using entire tensors.

Instead of

```python
sender0 = node_features[0]

sender1 = node_features[0]

sender2 = node_features[1]

sender3 = node_features[2]
```

we simply write

```python
senders = node_features[
    edge_index[0]
]
```

Printing

```python
print(senders)
```

produces

```text
tensor([
 [14.00,1.90],
 [14.00,1.90],
 [8.00,3.44],
 [8.00,3.44]
])
```

Every sender node has been collected

using one indexing operation.

No loops are required.

---

# 15.4.49 Extracting Receiver Features

Exactly the same idea applies.

```python
receivers = node_features[
    edge_index[1]
]
```

Output

```text
tensor([
 [8.00,3.44],
 [8.00,3.44],
 [14.00,1.90],
 [14.00,1.90]
])
```

Again,

every receiver feature vector is extracted simultaneously.

---

# 15.4.50 Expanding the Global State

Suppose

the global feature vector is

```python
global_state = torch.tensor([
    300.0,
    1.0
])
```

Shape

```text
torch.Size([2])
```

However,

the edge update requires

one global vector

for every edge.

If there are

four edges,

the tensor should become

```text
Edge 1 → [300,1]

Edge 2 → [300,1]

Edge 3 → [300,1]

Edge 4 → [300,1]
```

PyTorch provides

`repeat()`

and

`expand()`

for exactly this purpose.

```python
global_edges = global_state.unsqueeze(0)

global_edges = global_edges.repeat(
    edge_features.size(0),
    1
)

print(global_edges.shape)
```

Output

```text
torch.Size([4,2])
```

Now every edge has access to

the same global state.

This is mathematically equivalent to

passing

$$
\mathbf{u}
$$

into every edge update equation.

---

# 15.4.51 Constructing the Complete Edge Input Matrix

We now possess

every component needed for the edge update.

```python
edge_features

senders

receivers

global_edges
```

The complete input matrix is obtained using

```python
edge_inputs = torch.cat(

    [
        edge_features,
        senders,
        receivers,
        global_edges
    ],

    dim=1

)
```

Suppose

```text
Edge Features = 16

Sender Features = 64

Receiver Features = 64

Global Features = 32
```

The resulting matrix has dimensions

$$
N_e
\times
176
$$

Every row corresponds to

one edge.

Every column corresponds to

one feature.

The matrix might conceptually look like

$$
\begin{bmatrix}
e_1 & s_1 & r_1 & u\\
e_2 & s_2 & r_2 & u\\
e_3 & s_3 & r_3 & u\\
\vdots & \vdots & \vdots & \vdots\\
e_{N_e} & s_{N_e} & r_{N_e} & u
\end{bmatrix}
$$

where each row is processed independently by the neural network.

---

# 15.4.52 Updating Every Edge Simultaneously

The most elegant aspect of deep learning is that

a multilayer perceptron naturally operates on

an entire matrix.

Suppose

```python
edge_inputs.shape
```

returns

```text
torch.Size([1536,176])
```

Passing this tensor through the MLP

```python
updated_edges = self.edge_mlp(
    edge_inputs
)
```

automatically produces

```text
torch.Size([1536,16])
```

Notice what happened.

One forward pass

updated

every edge

in the graph.

No loops.

No manual iteration.

No repeated function calls.

The GPU performs all computations simultaneously.

---

# 15.4.53 Complete Vectorized Edge Update Module

Putting everything together,

our edge update module becomes

```python
class EdgeUpdate(torch.nn.Module):

    def __init__(
        self,
        edge_dim,
        node_dim,
        global_dim,
        hidden_dim
    ):

        super().__init__()

        input_dim = (
            edge_dim
            + 2*node_dim
            + global_dim
        )

        self.edge_mlp = torch.nn.Sequential(

            torch.nn.Linear(
                input_dim,
                hidden_dim
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                hidden_dim,
                edge_dim
            )
        )

    def forward(

        self,

        node_features,

        edge_features,

        edge_index,

        global_state

    ):

        senders = node_features[
            edge_index[0]
        ]

        receivers = node_features[
            edge_index[1]
        ]

        global_edges = global_state.unsqueeze(0)

        global_edges = global_edges.repeat(
            edge_features.size(0),
            1
        )

        edge_inputs = torch.cat(

            [
                edge_features,
                senders,
                receivers,
                global_edges
            ],

            dim=1

        )

        updated_edges = self.edge_mlp(
            edge_inputs
        )

        return updated_edges
```

Although this code is relatively compact,

it implements the complete mathematical equation

$$
\mathbf{e}_k'
=
\phi_e
(
\mathbf{e}_k,
\mathbf{v}_{r_k},
\mathbf{v}_{s_k},
\mathbf{u}
)
$$

for **every edge in the graph simultaneously**.

This is essentially how modern graph neural network libraries achieve high computational efficiency.

However, the edge update is only the **first** stage of a Graph Network block. After every edge has learned a new representation, the next question naturally arises:

> **How do these updated edge representations influence the atoms themselves?**

To answer this, we must introduce one of the most fundamental ideas in graph neural networks: **aggregation**. Before a node can be updated, it must first collect information from all of its neighboring edges. This seemingly simple operation leads us to the mathematical concepts of aggregation functions ($\rho$), permutation invariance, and message passing, which form the backbone of every Graph Network and every MEGNet block.

# 15.4.54 Why Aggregation Is Necessary

At this point in the Graph Network computation, every edge has already been updated.

The graph now contains

- updated edge representations,
- original node representations,
- the current global representation.

The next objective is to update the nodes.

A natural question immediately arises.

> **How should a node learn from all of its neighboring edges?**

This question is much deeper than it first appears.

Consider the following crystal fragment.

```text
        O

        |

Si ----- Si ----- O

        |

       Al
```

Focus on the central silicon atom.

It has

four neighboring atoms.

Consequently,

it also has

four neighboring edges.

Each edge now contains a learned representation produced by the edge update function.

Suppose the updated edge embeddings are

$$
\mathbf{e}_1',
\mathbf{e}_2',
\mathbf{e}_3',
\mathbf{e}_4'
$$

The silicon atom cannot update itself by looking at only one of these edges.

It must somehow combine information from **all four**.

This combination process is called **aggregation**.

---

# 15.4.55 A Real Materials Science Perspective

Imagine standing at one atom inside a crystal.

Every neighboring atom exerts an influence.

Some neighbors

- donate electrons,
- withdraw electrons,
- strengthen bonding,
- weaken bonding,
- distort the lattice,
- contribute magnetic interactions.

No single neighbor completely determines the atomic environment.

Instead,

the local environment emerges from the combined influence of **all neighbors**.

Graph aggregation mirrors this physical intuition.

Instead of considering neighbors independently,

the neural network combines all neighboring information into one representation.

This aggregated representation becomes the "chemical environment" perceived by the atom.

---

# 15.4.56 Why We Cannot Simply Concatenate Neighbor Features

A beginner's first idea might be

> "Why not concatenate all neighboring edge vectors?"

Suppose every edge embedding has dimension

16.

A node with

three neighbors

would receive

$$
3\times16=48
$$

features.

A node with

eight neighbors

would receive

$$
8\times16=128
$$

features.

A node with

twelve neighbors

would receive

$$
12\times16=192
$$

features.

Immediately,

a problem appears.

The neural network now receives

feature vectors of different sizes.

Neural networks require

**fixed input dimensions**.

Variable-length vectors cannot be directly processed by ordinary fully connected layers.

Therefore,

concatenation is not a practical solution.

---

# 15.4.57 Another Problem: Different Crystals Have Different Coordination Numbers

Consider several materials.

| Material | Typical Coordination Number |
|-----------|----------------------------:|
| Diamond | 4 |
| Sodium Chloride | 6 |
| FCC Metal | 12 |
| BCC Metal | 8 |

If the neural network depended on concatenation,

every material would produce

different input sizes.

The model would have to be redesigned for every crystal structure.

Clearly,

this is impossible.

We therefore require a mathematical operation that satisfies two conditions.

First,

it must accept

any number of neighboring vectors.

Second,

it must always produce

one fixed-size vector.

Aggregation satisfies exactly these requirements.

---

# 15.4.58 What Is Aggregation?

Aggregation is a mathematical function

that receives

many vectors

and returns

one vector.

Suppose a node receives

four incoming messages.

$$
\mathbf{m}_1,
\mathbf{m}_2,
\mathbf{m}_3,
\mathbf{m}_4
$$

Aggregation computes

$$
\bar{\mathbf{m}}
=
\rho
(
\mathbf{m}_1,
\mathbf{m}_2,
\mathbf{m}_3,
\mathbf{m}_4
)
$$

where

$\rho$

denotes an aggregation function.

Regardless of whether the node receives

2,

5,

20,

or

100 neighbors,

the output is always

one vector.

This property makes Graph Networks scalable to graphs of arbitrary size.

---

# 15.4.59 Aggregation Is Not Learning

One of the most common misconceptions among beginners is that aggregation is itself a neural network.

It is not.

Aggregation contains **no trainable parameters**.

Suppose we compute

the sum

of three vectors.

$$
\mathbf{m}_1
+
\mathbf{m}_2
+
\mathbf{m}_3
$$

Nothing is learned.

No weights are updated.

No gradients are stored.

Aggregation is simply a mathematical operation.

Learning occurs **after aggregation**, when the aggregated vector is passed through an update function such as

$$
\phi_v.
$$

Keeping these two concepts separate is extremely important.

| Operation | Learns Parameters? |
|------------|-------------------|
| Aggregation ($\rho$) | No |
| Update Function ($\phi$) | Yes |

---

# 15.4.60 The Three Most Common Aggregation Functions

Most Graph Neural Networks rely on one of three aggregation operations.

## Sum Aggregation

The simplest aggregation is

$$
\bar{\mathbf{m}}
=
\sum_i
\mathbf{m}_i
$$

Every neighboring message is added together.

Example

Suppose

$$
\mathbf{m}_1=[1,2]
$$

$$
\mathbf{m}_2=[3,1]
$$

$$
\mathbf{m}_3=[2,5]
$$

Then

$$
\bar{\mathbf{m}}
=
[6,8]
$$

Sum aggregation preserves information about

the total contribution of all neighbors.

Many materials graph neural networks,

including MEGNet,

make extensive use of summation because many physical quantities are naturally additive.

---

## Mean Aggregation

Instead of adding,

we compute the average.

$$
\bar{\mathbf{m}}
=
\frac{1}{N}
\sum_i
\mathbf{m}_i
$$

For the previous example,

$$
\bar{\mathbf{m}}
=
[2,\;2.67]
$$

Mean aggregation removes the dependence on the number of neighbors.

This can improve stability when graphs have widely varying coordination numbers.

---

## Max Aggregation

Another possibility is

taking the largest value

in every feature dimension.

Suppose

$$
\mathbf{m}_1=[1,4]
$$

$$
\mathbf{m}_2=[5,2]
$$

$$
\mathbf{m}_3=[3,7]
$$

Then

$$
\bar{\mathbf{m}}
=
[5,7]
$$

Max aggregation focuses on the strongest signal rather than the combined signal.

It is common in computer vision applications,

although it is less common in atomistic graph neural networks.

---

# 15.4.61 Implementing Aggregation in PyTorch

Aggregation is surprisingly easy to implement.

Suppose

three neighboring edges produce

```python
import torch

messages = torch.tensor([

    [1.0,2.0],

    [3.0,1.0],

    [2.0,5.0]

])
```

---

## Sum Aggregation

```python
aggregated = messages.sum(dim=0)

print(aggregated)
```

Output

```text
tensor([6.,8.])
```

---

## Mean Aggregation

```python
aggregated = messages.mean(dim=0)

print(aggregated)
```

Output

```text
tensor([2.0000,2.6667])
```

---

## Maximum Aggregation

```python
aggregated,_ = messages.max(dim=0)

print(aggregated)
```

Output

```text
tensor([5.,7.])
```

Notice that

the output dimension remains

two

regardless of the number of incoming messages.

Whether the node has

3 neighbors,

30 neighbors,

or

300 neighbors,

aggregation always produces

one feature vector with the same dimensionality.

This fixed-size representation is exactly what allows the subsequent node update function to operate on graphs with arbitrary connectivity.

---

# 15.4.62 Aggregation in Real Crystal Graphs

Consider a silicon atom inside crystalline silicon.

```text
          Si

           \

Si -------- Si -------- Si

           /

         Si
```

The central atom receives messages from

four neighboring bonds.

The Graph Network computes

```text
Edge 1 Message

↓

Edge 2 Message

↓

Edge 3 Message

↓

Edge 4 Message

↓

Aggregation

↓

Local Environment Representation
```

This aggregated vector is no longer associated with any single bond.

Instead,

it summarizes the **entire local chemical environment** surrounding the atom.

Only after this aggregation step does the Graph Network proceed to update the node itself.

This brings us to the next major component of the Graph Network framework: the **node update function ($\phi_v$)**, where the atom combines its own current representation with the aggregated information from all neighboring interactions to produce a richer atomic embedding.

# 15.4.63 The Node Update Function ($\phi_v$)

At this stage of the Graph Network computation,

every edge has already been updated.

The graph now contains

- updated edge embeddings,
- original node embeddings,
- the current global state.

The next objective is to update every node.

Unlike the edge update, which considers only one edge at a time, the node update must integrate information arriving from **all neighboring edges**.

This operation transforms an atom from an isolated entity into a representation of its **local chemical environment**.

This idea is one of the central principles behind modern graph neural networks.

An atom should not be represented only by its intrinsic properties.

Instead,

it should be represented by

- its own atomic characteristics,
- the interactions with neighboring atoms,
- the physical state of the entire system.

---

# 15.4.64 Physical Interpretation

Consider a silicon atom inside crystalline silicon.

```text
            Si

             |

Si -------- Si -------- Si

             |

            Si
```

Initially,

the central silicon atom may only know

```text
Atomic Number = 14

Electronegativity = 1.90

Radius = 1.11 Å
```

This information describes an isolated silicon atom.

However,

an atom inside a crystal is never isolated.

Its behavior depends strongly on

- neighboring atoms,
- neighboring bonds,
- crystal geometry,
- external conditions.

Therefore,

after one Graph Network layer,

the atomic representation should describe

not only the atom itself,

but also

its surrounding environment.

Instead of representing

"silicon",

the embedding gradually represents

"silicon inside this particular crystal environment."

This distinction is extremely important.

Graph Networks learn **contextual atomic representations**, not merely atomic descriptors.

---

# 15.4.65 Information Available for Updating a Node

Suppose

node

$i$

is connected to several neighboring atoms.

By the time the node update begins,

the following information is already available.

### 1. The current node embedding

$$
\mathbf{v}_i
$$

This contains the current representation of the atom.

---

### 2. Updated neighboring edge embeddings

Every edge connected to the atom has already been transformed by

$$
\phi_e.
$$

Suppose the neighboring edges are

$$
\mathbf{e}_1',
\mathbf{e}_2',
\mathbf{e}_3',
\ldots
$$

These edge embeddings contain information about

- bond strength,
- neighboring chemistry,
- geometric relationships,
- learned interaction patterns.

---

### 3. The global feature vector

The node also has access to

$$
\mathbf{u}
$$

which may contain

- temperature,
- pressure,
- magnetic field,
- external conditions.

Thus,

the updated atomic representation can depend on

both

local

and

global

physics.

---

# 15.4.66 Why Aggregation Happens Before the Node Update

Suppose one atom has

six neighboring edges.

The edge embeddings are

$$
\mathbf{e}_1',
\mathbf{e}_2',
\ldots,
\mathbf{e}_6'
$$

The node update function cannot directly process

six independent vectors,

because another atom may have

four neighbors,

or

twelve neighbors.

Therefore,

the Graph Network first performs aggregation.

Mathematically,

the aggregated edge information becomes

$$
\bar{\mathbf{e}}_i
=
\rho_{e\rightarrow v}
\left(
\{
\mathbf{e}_k'
:
r_k=i
\}
\right)
$$

Let us carefully interpret this equation.

The notation

$$
r_k=i
$$

means

> "Select every edge whose receiver node is node $i$."

The aggregation function

$$
\rho_{e\rightarrow v}
$$

then combines all of those updated edge embeddings into a single vector.

This aggregated vector summarizes the complete local interaction environment surrounding the atom.

---

# 15.4.67 Understanding the Aggregation Equation

Suppose

node

$i$

receives four incoming edges.

```text
Edge A

↓

Edge B

↓

Node i

↑

Edge C

↑

Edge D
```

Each edge sends an updated embedding.

The aggregation operation computes

$$
\bar{\mathbf{e}}_i
=
\mathbf{e}_A'
+
\mathbf{e}_B'
+
\mathbf{e}_C'
+
\mathbf{e}_D'
$$

if summation is used.

If mean aggregation is chosen,

then

$$
\bar{\mathbf{e}}_i
=
\frac{
\mathbf{e}_A'
+
\mathbf{e}_B'
+
\mathbf{e}_C'
+
\mathbf{e}_D'
}{4}
$$

Notice something important.

The node never sees the individual edge embeddings.

Instead,

it receives

one summarized representation

describing

its neighboring interactions.

---

# 15.4.68 The Node Update Equation

After aggregation,

the Graph Network updates the node.

The mathematical equation is

$$
\mathbf{v}_i'
=
\phi_v
(
\bar{\mathbf{e}}_i,
\mathbf{v}_i,
\mathbf{u}
)
$$

This equation mirrors the edge update equation,

but now the inputs are different.

The node update receives

- the aggregated neighboring edge information,
- the current node representation,
- the global state.

It produces

the updated node embedding

$$
\mathbf{v}_i'.
$$

Notice the elegant symmetry.

The edge update depends on

- edges,
- nodes,
- global state.

The node update depends on

- aggregated edges,
- nodes,
- global state.

Each stage gradually propagates information through the graph.

---

# 15.4.69 Why the Original Node Representation Is Still Needed

One might wonder

why

$$
\mathbf{v}_i
$$

is included in the update equation.

Why not compute

the new node representation

using only

the neighboring edges?

The answer is simple.

A node should never lose its own identity.

Suppose

a silicon atom

and

an oxygen atom

have identical neighboring environments.

If we ignored the original node embedding,

both atoms might become indistinguishable.

Including

$$
\mathbf{v}_i
$$

ensures that the updated representation preserves

the intrinsic identity of the atom

while incorporating information from its surroundings.

This is analogous to human learning.

Your knowledge changes through interaction with others,

but you do not lose your own identity.

The updated node embedding reflects both

who the atom is

and

what environment it experiences.

---

## **MEGNet Perspective**

This node update equation is implemented directly inside every MEGNet block.

Each atom begins with an initial embedding derived from its atomic attributes.

As message passing proceeds,

the embedding evolves into a representation of the atom **within its crystal environment**.

After multiple MEGNet blocks,

the node embedding no longer represents an isolated chemical element.

Instead,

it encodes

- local coordination,
- bonding environment,
- neighboring chemistry,
- structural motifs,
- long-range information propagated through repeated message passing.

This progressive enrichment of atomic representations is one of the key reasons MEGNet achieves significantly higher predictive accuracy than models based solely on handcrafted descriptors or fixed atomic features.

# 15.4.70 Implementing the Node Update Function

We now translate the mathematical equation

$$
\mathbf{v}_i'
=
\phi_v
(
\bar{\mathbf{e}}_i,
\mathbf{v}_i,
\mathbf{u}
)
$$

into PyTorch.

Unlike the edge update,

the node update cannot begin immediately.

Before updating a node,

we must first compute

$$
\bar{\mathbf{e}}_i,
$$

the aggregated information arriving from all neighboring edges.

This aggregation step is one of the defining characteristics of every message passing neural network.

Only after every node has collected messages from its neighbors can the node embedding be updated.

---

# 15.4.71 Computing Messages for Every Node

Suppose our graph contains

```text
5 Nodes

8 Directed Edges
```

After the edge update,

the tensor

```python
updated_edge_features
```

may have shape

```text
torch.Size([8, 64])
```

This means

- there are 8 updated edges,
- every edge embedding has dimension 64.

Our objective is

to convert these

8 edge embeddings

into

5 aggregated node embeddings.

Graphically,

```text
Edge Embeddings

↓

Group by Receiver Node

↓

Aggregate

↓

One Vector Per Node
```

This is precisely the operation represented mathematically by

$$
\rho_{e\rightarrow v}.
$$

---

# 15.4.72 Which Edges Belong to Which Node?

Recall that every edge stores

```text
Sender

Receiver
```

inside

```python
edge_index
```

Suppose

```python
edge_index = torch.tensor([
    [0,0,1,2,3,4],
    [1,2,2,3,4,0]
])
```

The first row contains

sender nodes.

The second row contains

receiver nodes.

Therefore,

the receiver indices are

```python
receivers = edge_index[1]
```

Output

```text
tensor([1,2,2,3,4,0])
```

These indices tell us exactly

which node should receive

each updated edge embedding.

---

# 15.4.73 Why Ordinary PyTorch Is Not Enough

Suppose we try

```python
for edge in edges:

    node += edge
```

This works,

but is extremely slow.

Modern graph neural network libraries instead perform

parallel aggregation.

The most common implementation uses

**scatter operations**.

Scatter operations distribute information

from many edges

into their corresponding destination nodes.

Conceptually,

they perform

```text
Edge 1 → Node 2

Edge 2 → Node 5

Edge 3 → Node 2

Edge 4 → Node 1

↓

Automatically Sum

↓

Node Representations
```

This operation executes efficiently on GPUs.

---

# 15.4.74 Aggregating Edge Embeddings Using PyTorch Geometric

PyTorch Geometric provides

powerful scatter utilities.

One commonly used function is

```python
from torch_geometric.utils import scatter
```

Suppose

```python
updated_edge_features.shape

torch.Size([8,64])
```

The aggregation becomes

```python
from torch_geometric.utils import scatter

aggregated_edges = scatter(

    updated_edge_features,

    receivers,

    dim=0,

    reduce="sum"

)
```

The resulting tensor may have shape

```text
torch.Size([5,64])
```

Every node now possesses

one aggregated edge embedding.

No loops.

No manual indexing.

Everything is computed simultaneously.

---

# 15.4.75 Understanding Scatter Aggregation

Suppose we have

three edge embeddings.

```text
Edge 1 → Node 0

Edge 2 → Node 1

Edge 3 → Node 0
```

Receiver indices become

```python
receivers = torch.tensor([0,1,0])
```

Edge embeddings

```python
messages = torch.tensor([

    [1.,2.],

    [3.,1.],

    [2.,5.]

])
```

Applying

```python
aggregated = scatter(

    messages,

    receivers,

    dim=0,

    reduce="sum"

)
```

produces

```text
Node 0

↓

[3.,7.]

Node 1

↓

[3.,1.]
```

because

Node 0 receives

Edge 1

and

Edge 3,

while

Node 1 receives

only

Edge 2.

Scatter automatically performs

the grouping,

aggregation,

and storage.

---

# 15.4.76 Preparing Inputs for the Node Update

After aggregation,

every node has

three sources of information.

First,

its current embedding

```python
node_features
```

Shape

```text
[N_nodes, node_dim]
```

Second,

its aggregated neighboring information

```python
aggregated_edges
```

Shape

```text
[N_nodes, edge_dim]
```

Third,

the global state.

Suppose

```python
global_state.shape

torch.Size([32])
```

We must copy

the global vector

for every node.

```python
global_nodes = global_state.unsqueeze(0)

global_nodes = global_nodes.repeat(

    node_features.size(0),

    1

)
```

Shape

```text
[N_nodes, global_dim]
```

Every node now has access to

the same global information.

---

# 15.4.77 Constructing the Node Input Matrix

We concatenate

all three tensors.

```python
node_inputs = torch.cat(

    [

        node_features,

        aggregated_edges,

        global_nodes

    ],

    dim=1

)
```

Suppose

```text
Node Dimension = 64

Edge Dimension = 64

Global Dimension = 32
```

The resulting tensor has dimensions

$$
N_v
\times
160
$$

where

$$
160
=
64
+
64
+
32.
$$

Each row now contains

everything needed

to update one node.

---

# 15.4.78 Building the Node Update Network

The update function

$$
\phi_v
$$

is implemented using

a multilayer perceptron.

```python
import torch
import torch.nn as nn

class NodeUpdate(nn.Module):

    def __init__(

        self,

        node_dim,

        edge_dim,

        global_dim,

        hidden_dim

    ):

        super().__init__()

        input_dim = (

            node_dim

            + edge_dim

            + global_dim

        )

        self.node_mlp = nn.Sequential(

            nn.Linear(

                input_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                node_dim

            )

        )

    def forward(

        self,

        node_features,

        aggregated_edges,

        global_state

    ):

        global_nodes = global_state.unsqueeze(0)

        global_nodes = global_nodes.repeat(

            node_features.size(0),

            1

        )

        node_inputs = torch.cat(

            [

                node_features,

                aggregated_edges,

                global_nodes

            ],

            dim=1

        )

        updated_nodes = self.node_mlp(

            node_inputs

        )

        return updated_nodes
```

Notice that

the output dimension is

again

`node_dim`.

The representation changes,

but the object remains

a node embedding.

---

# 15.4.79 Running the Node Update

Once the aggregation has been computed,

the update becomes straightforward.

```python
node_update = NodeUpdate(

    node_dim=64,

    edge_dim=64,

    global_dim=32,

    hidden_dim=128

)

updated_nodes = node_update(

    node_features,

    aggregated_edges,

    global_state

)

print(updated_nodes.shape)
```

Output

```text
torch.Size([N_nodes,64])
```

Every atom in the crystal has now received information from

its neighboring interactions

and

the global physical state.

Its embedding is no longer a simple description of an isolated atom.

Instead,

it has become a learned representation of

the atom within its surrounding crystal environment.

---

## **MEGNet Perspective**

This is the second major computation inside every MEGNet block.

The workflow is now

```text
Raw Node Features

↓

Edge Update

↓

Updated Edge Embeddings

↓

Edge Aggregation

↓

Node Update

↓

Updated Node Embeddings
```

Notice that the node embeddings now contain information extending beyond the individual atom.

After several MEGNet blocks,

information propagates farther and farther through the crystal.

An atom eventually learns not only about its nearest neighbors,

but also about second-nearest,

third-nearest,

and more distant atomic environments.

This progressive expansion of the receptive field is one of the fundamental reasons MEGNet can accurately predict complex material properties.

However, the Graph Network computation is still not complete.

We have updated

- edges,
- nodes.

The final stage is to update the **global state ($\phi_u$)**, allowing the graph itself to learn a richer representation of the entire material system by combining information from all updated nodes and all updated edges.

# 15.4.80 The Global Update Function ($\phi_u$)

We have now completed two of the three update operations in the Graph Network framework.

First,

the edges were updated.

Second,

the nodes were updated.

The only remaining object is

the **global state**.

This is one of the defining innovations of the Graph Network framework and, consequently, of MEGNet.

Earlier graph neural networks such as GCN and GraphSAGE generally focused only on

- nodes,
- edges.

MEGNet, however, introduces a third level of representation.

The **entire graph** itself becomes a learnable object.

Instead of merely storing global information,

the network continuously refines its understanding of the complete material system.

---

# 15.4.81 What Does the Global State Represent?

At the beginning of computation,

the global feature vector may contain explicit physical quantities such as

- temperature,
- pressure,
- magnetic field,
- strain,
- applied electric field.

For example,

$$
\mathbf{u}
=
\begin{bmatrix}
300\\
1.0\\
0
\end{bmatrix}
$$

might represent

```text
Temperature = 300 K

Pressure = 1 atm

Magnetic Field = 0 T
```

However,

after several Graph Network blocks,

the global state evolves into something much richer.

Instead of storing only experimental conditions,

it becomes a **learned embedding of the entire crystal**.

In other words,

the global vector gradually answers questions such as

- What type of crystal is this?
- How are atoms arranged?
- What bonding patterns dominate?
- How complex is the local chemistry?
- What information is important for predicting the target property?

Thus,

the global state changes from

**physical metadata**

into

**a learned representation of the complete material**.

---

# 15.4.82 Why Do We Need a Global Representation?

Imagine reading a novel.

Each word represents

a node.

Each sentence represents

a local neighborhood.

However,

understanding the novel requires more than understanding individual sentences.

You must understand

the overall story.

The global state plays exactly this role.

Nodes understand

local atomic environments.

Edges understand

pairwise interactions.

The global vector understands

the entire crystal.

This hierarchical organization is one of the reasons Graph Networks are so powerful.

---

# 15.4.83 Information Available During the Global Update

By the time the global update begins,

the graph already contains

- updated edge embeddings,

$$
\mathbf{e}_k'
$$

- updated node embeddings,

$$
\mathbf{v}_i'
$$

- the previous global state,

$$
\mathbf{u}
$$

The objective is

to combine all of this information into

a new global representation,

$$
\mathbf{u}'.
$$

Unlike node updates,

which focus on local neighborhoods,

the global update considers

the **entire graph simultaneously**.

---

# 15.4.84 The Need for Two Aggregation Operations

The graph may contain

- hundreds of nodes,
- thousands of edges.

The global update cannot process every node and edge individually.

Instead,

it first aggregates

all updated node embeddings

into

one vector,

and

all updated edge embeddings

into

another vector.

Mathematically,

the node aggregation is

$$
\bar{\mathbf{v}}
=
\rho_{v\rightarrow u}
\left(
\{
\mathbf{v}_i'
\}
\right)
$$

Similarly,

the edge aggregation is

$$
\bar{\mathbf{e}}
=
\rho_{e\rightarrow u}
\left(
\{
\mathbf{e}_k'
\}
\right)
$$

Notice something important.

These aggregations are performed

over the **entire graph**.

There is no longer any distinction between neighboring atoms.

Everything contributes to the global representation.

---

# 15.4.85 Understanding the Global Aggregation

Suppose

our graph contains

four updated node embeddings.

```text
Node 1

↓

Node 2

↓

Node 3

↓

Node 4
```

Applying

sum aggregation

produces

$$
\bar{\mathbf{v}}
=
\mathbf{v}_1'
+
\mathbf{v}_2'
+
\mathbf{v}_3'
+
\mathbf{v}_4'
$$

Likewise,

if the graph contains

six updated edge embeddings,

their aggregation becomes

$$
\bar{\mathbf{e}}
=
\sum_k
\mathbf{e}_k'
$$

The graph has now been compressed into

two vectors.

One summarizes

all atomic information.

The other summarizes

all interaction information.

---

# 15.4.86 The Global Update Equation

The Graph Network framework now computes

the updated global state using

$$
\mathbf{u}'
=
\phi_u
(
\bar{\mathbf{e}},
\bar{\mathbf{v}},
\mathbf{u}
)
$$

Let us examine each input.

The first input

$$
\bar{\mathbf{e}}
$$

contains

information from

every interaction

inside the graph.

The second input

$$
\bar{\mathbf{v}}
$$

contains

information from

every atom.

The third input

$$
\mathbf{u}
$$

contains

the previous global representation.

Together,

these three vectors completely summarize

the state of the graph.

The neural network

$\phi_u$

learns how to combine them into

a richer global embedding.

---

# 15.4.87 Why Include the Previous Global State?

A natural question arises.

Why does the update function receive

$$
\mathbf{u}
$$

if

all node

and

edge information

has already been aggregated?

The answer is

memory.

Suppose

the original global vector contains

temperature.

Temperature may not appear

inside

the node features

or

edge features.

If we discarded

$$
\mathbf{u},
$$

that information would be permanently lost.

Including

the previous global state

ensures that

external physical conditions remain available

throughout every Graph Network block.

---

## **MEGNet Perspective**

This is one of the defining differences between MEGNet and earlier graph neural networks.

Most early GNNs assume that the graph contains only

nodes

and

edges.

MEGNet introduces

a third information pathway:

the **global state**.

This allows the model to naturally incorporate external variables such as

- temperature,
- pressure,
- entropy,
- synthesis conditions,
- experimental parameters,

without artificially attaching them to every atom.

As a result,

MEGNet can learn how global physical conditions modify local atomic interactions,

making it especially powerful for predicting material properties that depend on external thermodynamic conditions.

---

# 15.4.88 Implementing Global Aggregation

Before implementing

$$
\phi_u,
$$

we must first aggregate

all updated node embeddings

and

all updated edge embeddings.

Suppose

```python
updated_nodes.shape

torch.Size([128,64])
```

and

```python
updated_edges.shape

torch.Size([768,64])
```

Since the graph represents

a **single crystal**,

the aggregation is straightforward.

```python
global_nodes = updated_nodes.sum(

    dim=0,

    keepdim=True

)

global_edges = updated_edges.sum(

    dim=0,

    keepdim=True

)
```

The resulting tensors become

```text
global_nodes

torch.Size([1,64])

global_edges

torch.Size([1,64])
```

Each tensor now summarizes

the entire graph.

For batched graphs,

PyTorch Geometric instead performs this aggregation separately for each graph using the `batch` vector and global pooling operations. We will implement this more general approach later when constructing the complete MEGNet architecture.

---

# 15.4.89 Constructing the Global Input Vector

The inputs are concatenated exactly as before.

```python
global_input = torch.cat(

    [

        global_edges,

        global_nodes,

        global_state.unsqueeze(0)

    ],

    dim=1

)
```

Suppose

```text
Edge Dimension = 64

Node Dimension = 64

Global Dimension = 32
```

The resulting tensor has shape

```text
torch.Size([1,160])
```

This vector contains

all information required to update

the global representation.

---

# 15.4.90 Building the Global Update Network

The implementation closely resembles

the edge

and

node

update networks.

```python
import torch
import torch.nn as nn

class GlobalUpdate(nn.Module):

    def __init__(

        self,

        edge_dim,

        node_dim,

        global_dim,

        hidden_dim

    ):

        super().__init__()

        input_dim = (

            edge_dim

            + node_dim

            + global_dim

        )

        self.global_mlp = nn.Sequential(

            nn.Linear(

                input_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_dim,

                global_dim

            )

        )

    def forward(

        self,

        global_edges,

        global_nodes,

        global_state

    ):

        x = torch.cat(

            [

                global_edges,

                global_nodes,

                global_state.unsqueeze(0)

            ],

            dim=1

        )

        updated_global = self.global_mlp(x)

        return updated_global.squeeze(0)
```

This completes the implementation of

$$
\phi_u.
$$

At this point,

we have successfully implemented

all three learnable update functions of the Graph Network framework:

- Edge update ($\phi_e$)
- Node update ($\phi_v$)
- Global update ($\phi_u$)

However,

these functions have been studied independently.

The next step is significantly more important.

We will combine them into a **single Graph Network block**, tracing the complete data flow from raw graph inputs to updated edges, updated nodes, and the updated global state. This integrated computation forms the fundamental building block of every MEGNet model, and stacking multiple such blocks enables the network to learn increasingly complex representations of crystal structures.

# 15.4.91 Constructing a Complete Graph Network Block

We have now studied every learnable component of the Graph Network framework individually.

Specifically,

we have implemented

- the edge update function,
- the node update function,
- the global update function.

Although each function is relatively simple on its own,

a Graph Network derives its power from **combining these functions into a single computational unit**.

This computational unit is called a **Graph Network Block**.

The Graph Network Block is the fundamental building block of MEGNet.

Just as a convolutional neural network repeatedly stacks convolution layers,

MEGNet repeatedly stacks Graph Network blocks.

Everything learned by MEGNet emerges from repeatedly applying this block to the crystal graph.

---

# 15.4.92 A Graph Network Block Is One Iteration of Graph Reasoning

One Graph Network block can be viewed as

one complete reasoning step over the graph.

Initially,

the graph contains

```text
Nodes

Edges

Global State
```

After one Graph Network block,

every component has learned something new.

```text
Updated Nodes

Updated Edges

Updated Global State
```

If another block is applied,

these updated representations become the inputs.

The graph continues learning.

After many Graph Network blocks,

the representations become increasingly informative.

This iterative refinement is one of the defining characteristics of deep graph neural networks.

---

# 15.4.93 Data Flow Inside the Block

The computation always follows the same sequence.

```text
Initial Graph

↓

Edge Update

↓

Updated Edges

↓

Edge Aggregation

↓

Node Update

↓

Updated Nodes

↓

Global Aggregation

↓

Global Update

↓

Updated Graph
```

Notice the strict ordering.

Edges are updated first.

Nodes depend on updated edges.

The global state depends on both updated nodes and updated edges.

This ordering is not arbitrary.

Each stage builds upon the information computed in the previous stage.

---

# 15.4.94 Why This Ordering Makes Physical Sense

Suppose we are simulating a crystal.

Interactions between atoms determine

how atoms influence one another.

Therefore,

the interactions should be updated first.

Once the interactions are known,

each atom can update its own representation.

Finally,

once every atom has updated,

the model can summarize

the entire crystal.

Thus,

the ordering

```text
Edges

↓

Nodes

↓

Global
```

matches the natural hierarchy of physical systems.

Interactions influence atoms.

Atoms collectively determine the material.

---

# 15.4.95 Mathematical Representation of the Entire Block

The Graph Network block consists of three consecutive equations.

First,

every edge is updated.

$$
\mathbf{e}_k'
=
\phi_e
(
\mathbf{e}_k,
\mathbf{v}_{r_k},
\mathbf{v}_{s_k},
\mathbf{u}
)
$$

Second,

edge information is aggregated,

and every node is updated.

$$
\bar{\mathbf{e}}_i
=
\rho_{e\rightarrow v}
(
\{
\mathbf{e}_k'
\}
)
$$

$$
\mathbf{v}_i'
=
\phi_v
(
\bar{\mathbf{e}}_i,
\mathbf{v}_i,
\mathbf{u}
)
$$

Finally,

the graph aggregates

all updated edges

and

all updated nodes.

$$
\bar{\mathbf{e}}
=
\rho_{e\rightarrow u}
(
\{
\mathbf{e}_k'
\}
)
$$

$$
\bar{\mathbf{v}}
=
\rho_{v\rightarrow u}
(
\{
\mathbf{v}_i'
\}
)
$$

The global state becomes

$$
\mathbf{u}'
=
\phi_u
(
\bar{\mathbf{e}},
\bar{\mathbf{v}},
\mathbf{u}
)
$$

These equations completely describe one Graph Network block.

Every MEGNet layer performs exactly this sequence of computations.

---

# 15.4.96 Information Propagation Through the Graph

To understand why Graph Networks work so well,

consider the following crystal.

```text
A ----- B ----- C ----- D
```

Initially,

atom

A

knows nothing about

D.

After one Graph Network block,

information propagates only one step.

```text
A ←→ B

B ←→ C

C ←→ D
```

After the second block,

information propagates further.

```text
A ←→ B ←→ C

B ←→ C ←→ D
```

After the third block,

atom

A

has indirectly received information originating from

D.

Thus,

stacking Graph Network blocks gradually increases the **receptive field** of every node.

Instead of observing only immediate neighbors,

nodes begin to learn long-range structural information.

This is one of the major reasons deep graph neural networks outperform shallow models.

---

# 15.4.97 Receptive Field Expansion

Suppose every node initially sees

only

its first neighbors.

```text
Radius = 1
```

After one block,

the receptive field remains

one hop.

After two blocks,

each node has indirectly incorporated information from

second-nearest neighbors.

```text
Radius = 2
```

After three blocks,

third-nearest neighbors contribute.

```text
Radius = 3
```

In general,

after

$L$

Graph Network blocks,

a node has access to information approximately

$L$

hops away in the graph.

This progressive expansion enables MEGNet to learn

medium-range

and

long-range

structural effects.

---

# 15.4.98 Computational View of One Block

The entire computation can be summarized as

```text
Input Graph

Nodes

Edges

Global

↓

Edge Neural Network

↓

Updated Edges

↓

Aggregation

↓

Node Neural Network

↓

Updated Nodes

↓

Global Aggregation

↓

Global Neural Network

↓

Updated Graph
```

Every stage is fully differentiable.

Consequently,

gradients can flow through

the entire Graph Network block

during backpropagation.

All update networks are optimized simultaneously.

---

# 15.4.99 Implementing the Complete Graph Network Block

We now combine the previously developed modules into a single PyTorch implementation.

```python
import torch
import torch.nn as nn

class GraphNetworkBlock(nn.Module):

    def __init__(

        self,

        edge_update,

        node_update,

        global_update

    ):

        super().__init__()

        self.edge_update = edge_update

        self.node_update = node_update

        self.global_update = global_update

    def forward(

        self,

        node_features,

        edge_features,

        edge_index,

        global_state

    ):

        # Step 1
        updated_edges = self.edge_update(

            node_features,

            edge_features,

            edge_index,

            global_state

        )

        # Step 2
        # Aggregate edge information
        # (implementation shown previously)

        aggregated_edges = ...

        # Step 3
        updated_nodes = self.node_update(

            node_features,

            aggregated_edges,

            global_state

        )

        # Step 4
        # Aggregate nodes and edges
        # (implementation shown previously)

        global_nodes = ...

        global_edges = ...

        # Step 5
        updated_global = self.global_update(

            global_edges,

            global_nodes,

            global_state

        )

        return (

            updated_nodes,

            updated_edges,

            updated_global

        )
```

This implementation highlights the logical organization of the computation.

In the next section, we will replace the placeholder (`...`) operations with fully vectorized implementations using PyTorch Geometric's scatter and global pooling functions.

---

# 15.4.100 Why the Graph Network Block Is Modular

One of the greatest strengths of the Graph Network framework is its modularity.

Each update function is independent.

For example,

the edge update MLP can be replaced by

- deeper neural networks,
- residual networks,
- attention mechanisms,
- transformer blocks,

without changing the remainder of the framework.

Similarly,

the aggregation function can be

- sum,
- mean,
- max,
- attention-weighted aggregation.

This modular design has enabled the development of numerous successors to MEGNet,

including

- M3GNet,
- ALIGNN,
- CHGNet,
- MatGL,

all of which retain the same fundamental Graph Network philosophy while modifying specific components.

---

## **MEGNet Perspective**

At this point, we have mathematically reconstructed the complete computational core of MEGNet.

Every MEGNet layer is essentially a Graph Network block that performs

1. **Edge update** to learn interaction representations,
2. **Node update** to learn atomic representations,
3. **Global update** to learn crystal-level representations.

The remarkable predictive power of MEGNet does not arise from a single sophisticated equation.

Instead, it emerges from repeatedly applying this relatively simple Graph Network block across the crystal graph.

Understanding this block is therefore the key to understanding not only MEGNet, but also nearly every modern materials graph neural network developed after it.

The next section moves beyond the abstract Graph Network framework and begins examining **how MEGNet specifically constructs these blocks**, including the architectural modifications, residual connections, embedding layers, and design choices introduced in the original MEGNet paper.

# 15.4.101 Why One Graph Network Block Is Usually Not Enough

We have now built a complete Graph Network block.

A natural question immediately follows.

> **If one Graph Network block can update the graph, why does MEGNet stack multiple Graph Network blocks?**

The answer lies in how information propagates through a graph.

A single Graph Network block allows an atom to directly learn only from its **immediate neighbors**.

Many material properties, however, depend on interactions extending far beyond the first coordination shell.

Consequently,

one Graph Network block is rarely sufficient.

Instead,

MEGNet repeatedly applies the same computational process,

allowing information to propagate progressively farther through the crystal.

---

# 15.4.102 Information Propagation After One Block

Consider a one-dimensional chain of atoms.

```text
A ---- B ---- C ---- D ---- E
```

Initially,

every atom only possesses its own feature vector.

```text
A

B

C

D

E
```

During the first Graph Network block,

messages travel only across directly connected edges.

```text
A ⇄ B

B ⇄ C

C ⇄ D

D ⇄ E
```

After the first update,

Atom C has learned about

- Atom B
- Atom D

but has learned nothing directly about

- Atom A
- Atom E

The receptive field extends only

one edge away.

---

# 15.4.103 Information Propagation After Two Blocks

Now apply a second Graph Network block.

The updated node embeddings from the first block become the inputs to the second block.

```text
First Block

↓

Updated Nodes

↓

Second Block
```

During the second round of message passing,

information already stored inside neighboring nodes continues propagating.

Now,

Atom C indirectly receives information originating from

```text
A

↓

B

↓

C
```

and simultaneously

```text
E

↓

D

↓

C
```

Its receptive field has expanded.

Instead of understanding only

its nearest neighbors,

it now incorporates

second-nearest neighbors.

---

# 15.4.104 Information Propagation After Three Blocks

Applying a third Graph Network block expands the receptive field even further.

```text
Block 1

↓

Block 2

↓

Block 3
```

Information originating from one end of the graph can now travel much farther.

For sufficiently deep networks,

an atom gradually accumulates knowledge about

the entire crystal.

Graphically,

```text
One Block

A → B

Two Blocks

A → B → C

Three Blocks

A → B → C → D

Four Blocks

A → B → C → D → E
```

Each additional Graph Network block increases the amount of structural context available to every atom.

---

# 15.4.105 Why Long-Range Information Matters in Materials Science

Many material properties are not determined solely by nearest-neighbor interactions.

For example,

electronic properties often depend upon

- extended bonding networks,
- crystal symmetry,
- long-range electrostatic interactions.

Mechanical properties may depend on

- grain connectivity,
- crystal packing,
- network rigidity.

Thermal conductivity depends upon

collective lattice vibrations,

which extend throughout the crystal.

Magnetic ordering frequently depends on

exchange interactions spanning multiple atomic distances.

Consequently,

restricting a model to one-hop neighborhoods severely limits its predictive capability.

Repeated message passing allows MEGNet to gradually capture these longer-range phenomena.

---

# 15.4.106 Receptive Field Versus Computational Cost

Increasing the number of Graph Network blocks improves the receptive field,

but it also increases computational cost.

Suppose

one Graph Network block requires

$$
T
$$

seconds.

Two blocks require approximately

$$
2T.
$$

Five blocks require

$$
5T.
$$

Furthermore,

deeper models require

more memory,

more parameters,

and longer training times.

Therefore,

choosing the number of Graph Network blocks becomes a balance between

- expressive power,
- computational efficiency.

---

# 15.4.107 Why Not Use Hundreds of Graph Network Blocks?

One might think

"If deeper is better,

why not stack one hundred Graph Network blocks?"

Unfortunately,

deep graph neural networks encounter several important difficulties.

## 1. Over-smoothing

As message passing continues,

neighboring node embeddings gradually become increasingly similar.

Eventually,

different atoms become difficult to distinguish.

This phenomenon is called

**over-smoothing**.

Instead of preserving local chemical identities,

all node embeddings begin converging toward nearly identical vectors.

---

## 2. Vanishing Gradients

Very deep neural networks become increasingly difficult to optimize.

Gradients must travel through many successive update functions,

causing them to shrink during backpropagation.

Training therefore becomes unstable.

---

## 3. Increased Computational Cost

Each additional Graph Network block performs

- edge updates,
- node updates,
- global updates,
- aggregation operations.

Training time grows approximately linearly with the number of blocks.

Therefore,

extremely deep Graph Networks become computationally expensive.

---

# 15.4.108 The Practical Design Used by MEGNet

Rather than stacking dozens of Graph Network blocks,

the original MEGNet architecture typically employs only a small number.

Conceptually,

its computation resembles

```text
Crystal Graph

↓

Embedding Layer

↓

Graph Network Block

↓

Graph Network Block

↓

Graph Network Block

↓

Readout

↓

Property Prediction
```

Although only a few Graph Network blocks are used,

each block significantly enriches the node,

edge,

and global representations.

This design provides an effective compromise between

accuracy

and

computational efficiency.

---

# 15.4.109 Residual Information Flow

Another important design consideration is

preserving information learned in earlier layers.

Suppose

the first Graph Network block learns

excellent atomic representations.

If later layers completely overwrite these representations,

valuable information may be lost.

Modern graph neural networks therefore frequently employ

**residual (skip) connections**,

allowing information from earlier blocks to flow directly into later blocks.

Instead of computing

```text
Input

↓

Graph Block

↓

Output
```

the computation becomes

```text
Input

├──────────────┐

│              │

▼              │

Graph Block    │

│              │

▼              │

Addition ◄─────┘

↓

Output
```

Residual connections improve

- gradient flow,
- optimization stability,
- representation preservation.

---

# 15.4.110 MEGNet Perspective

The original MEGNet architecture incorporates multiple Graph Network blocks arranged sequentially.

Each block performs exactly the sequence we have derived:

1. Edge update
2. Edge aggregation
3. Node update
4. Global aggregation
5. Global update

The output graph from one block becomes the input graph for the next block.

Consequently,

the crystal representation becomes progressively richer with every layer.

By the final Graph Network block,

the node embeddings no longer represent isolated atoms,

nor merely local coordination environments.

Instead,

they encode information propagated across large portions of the crystal,

capturing complex structural and chemical relationships that are highly informative for predicting material properties.

At this point, we have completed the complete mathematical and algorithmic description of the **Graph Network framework** underlying MEGNet.

The next major section transitions from the general framework to the **actual MEGNet architecture proposed by Chen et al.**, where we will examine how embedding layers, Graph Network blocks, residual connections, readout operations, and prediction heads are assembled into the complete neural network used for state-of-the-art materials property prediction.

# 15.5 The Complete MEGNet Architecture

We have now completely understood the **Graph Network (GN) framework** that serves as the theoretical foundation of MEGNet.

However, understanding the Graph Network equations alone is **not** sufficient to implement MEGNet.

The Graph Network framework is a **general computational framework**.

MEGNet is a **specific neural network architecture** built on top of that framework.

This distinction is extremely important.

Many beginners incorrectly assume that

> Graph Network = MEGNet.

This is not true.

The relationship is

```text
Graph Network Framework

↓

General Mathematical Framework

↓

MEGNet

↓

One Specific Architecture
```

Exactly the same Graph Network framework can be modified to create

- MEGNet
- M3GNet
- CHGNet
- ALIGNN
- MatGL

These models differ

not because the Graph Network equations change,

but because the architecture surrounding those equations changes.

Therefore,

our next objective is to understand

**how the original MEGNet paper transforms the Graph Network framework into a practical deep learning model for materials property prediction.**

---

# 15.5.1 High-Level Overview of MEGNet

Suppose we want to predict

the formation energy

of a crystal.

The input is

```text
Crystal Structure
```

The output is

```text
Formation Energy
```

Everything occurring between these two stages

constitutes the MEGNet architecture.

The complete workflow is

```text
Crystal Structure

↓

Graph Construction

↓

Initial Node Features

↓

Initial Edge Features

↓

Global State

↓

Embedding Networks

↓

Multiple Graph Network Blocks

↓

Readout Layer

↓

Prediction Network

↓

Material Property
```

Every one of these stages is essential.

Removing any component significantly reduces the model's performance.

---

# 15.5.2 The Five Major Components of MEGNet

The architecture can be divided into five major modules.

```text
1.

Graph Construction

↓

2.

Embedding Layer

↓

3.

Graph Network Blocks

↓

4.

Readout

↓

5.

Prediction Head
```

Each module performs a different task.

---

## Module 1 — Graph Construction

Graph construction converts

the crystal structure

into

a mathematical graph.

Specifically,

it defines

- nodes,
- edges,
- global state.

This stage is responsible for converting

physical information

into

machine learning data.

Without graph construction,

the neural network has no understanding of atomic connectivity.

---

## Module 2 — Embedding Layer

The raw atomic descriptors

are not immediately suitable for deep learning.

For example,

suppose

the node feature is

```text
Atomic Number = 14
```

Feeding

14

directly into a neural network

is generally ineffective.

Instead,

MEGNet transforms

every raw feature

into a higher-dimensional learned representation.

For example,

```text
14

↓

Embedding

↓

64-dimensional vector
```

This process is called

**feature embedding**.

Embedding layers are learned during training.

They enable the network to discover

useful numerical representations

rather than relying on manually designed descriptors.

---

## Module 3 — Graph Network Blocks

This is the computational core of MEGNet.

Each block performs

```text
Edge Update

↓

Node Update

↓

Global Update
```

using exactly the equations

derived in the previous section.

Multiple Graph Network blocks

are stacked sequentially.

Information propagates farther through the graph

after every block.

---

## Module 4 — Readout

After message passing,

every node possesses

a high-dimensional embedding.

However,

the prediction target

such as

formation energy

is associated with

the entire crystal,

not with individual atoms.

Therefore,

MEGNet must combine

all node information

into

one graph-level representation.

This operation is called

the **readout layer**.

The readout converts

```text
Many Node Embeddings

↓

One Crystal Embedding
```

---

## Module 5 — Prediction Head

Finally,

the graph embedding

is passed through

a standard multilayer perceptron.

The neural network predicts

```text
Formation Energy

Band Gap

Elastic Modulus

Bulk Modulus

Shear Modulus

Density of States

...

```

depending upon

the training dataset.

---

# 15.5.3 Overall Computational Pipeline

Putting everything together,

the entire MEGNet architecture becomes

```text
Crystal Structure

↓

Neighbor Search

↓

Graph Construction

↓

Initial Features

↓

Embedding Networks

↓

Graph Network Block

↓

Graph Network Block

↓

Graph Network Block

↓

Readout

↓

Fully Connected Neural Network

↓

Predicted Property
```

Notice something important.

Only

the Graph Network blocks

perform message passing.

The remaining modules perform

- preprocessing,
- feature transformation,
- graph summarization,
- prediction.

Each module has a distinct responsibility.

---

# 15.5.4 Data Dimensions Throughout the Network

To understand the architecture more clearly,

let us follow the dimensions of the tensors.

Suppose

our crystal contains

```text
100 Atoms

600 Directed Edges
```

Initially,

the graph contains

```text
Nodes

↓

100 × F_node

Edges

↓

600 × F_edge

Global

↓

1 × F_global
```

After the embedding layer,

the dimensions become

```text
Nodes

↓

100 × 64

Edges

↓

600 × 64

Global

↓

1 × 32
```

Notice that

raw descriptors

have now become

dense latent vectors.

After every Graph Network block,

the dimensions remain identical.

```text
Nodes

↓

100 × 64

Edges

↓

600 × 64

Global

↓

1 × 32
```

The numerical values change,

but the tensor shapes remain constant.

This greatly simplifies

deep network construction.

---

# 15.5.5 Why Keep the Dimensions Constant?

Suppose

the first Graph Network block

outputs

64-dimensional node embeddings.

If the second block expected

128-dimensional inputs,

the architecture would become unnecessarily complicated.

Instead,

MEGNet uses

the same hidden dimensions

throughout

all Graph Network blocks.

This allows

one block

to feed directly into

the next block.

Graphically,

```text
64

↓

64

↓

64

↓

64
```

rather than

```text
64

↓

128

↓

256

↓

512
```

Keeping a constant hidden dimension

improves

implementation simplicity,

computational efficiency,

and parameter sharing.

---

# 15.5.6 Why Embeddings Instead of Raw Features?

Consider two atoms.

```text
Hydrogen

Atomic Number = 1
```

and

```text
Uranium

Atomic Number = 92
```

If these numbers are used directly,

the neural network may incorrectly assume

that uranium is

"92 times larger"

than hydrogen

in some meaningful mathematical sense.

This interpretation is physically meaningless.

Instead,

the embedding layer learns

a vector representation

for every element.

For example,

```text
Hydrogen

↓

[-0.72, 0.18, ..., 0.41]
```

```text
Carbon

↓

[ 0.65,-0.24, ..., 0.83]
```

```text
Silicon

↓

[-0.13, 0.57, ..., 0.09]
```

These vectors are learned automatically during training.

Elements exhibiting similar chemical behavior often develop similar embeddings,

allowing the network to capture relationships that raw atomic numbers cannot express.

---

## **MEGNet Perspective**

Everything we have studied so far—the Graph Network equations, message passing, aggregation, and update functions—constitutes only the **computational engine** of MEGNet.

The architecture surrounding this engine is equally important.

The original MEGNet model first transforms raw materials data into learned embeddings, repeatedly refines these embeddings through stacked Graph Network blocks, compresses the entire crystal into a graph-level representation using a readout operation, and finally predicts material properties with a multilayer perceptron.

In the following sections, we will begin examining each architectural component in detail, starting with the **embedding layer**, one of the most overlooked yet fundamentally important parts of the MEGNet model.

# 15.5.7 The Embedding Layer

The first learnable component of the MEGNet architecture is the **embedding layer**.

Although it occupies only a small portion of the overall network,

it plays a crucial role.

Without embeddings,

the Graph Network would operate on raw numerical descriptors,

which are often poor representations of chemical information.

The purpose of the embedding layer is to transform

low-dimensional,

human-designed descriptors

into

high-dimensional,

learnable feature representations.

Rather than forcing the model to interpret raw values,

we allow the neural network to discover its own internal representation of chemistry.

This idea is one of the major reasons modern deep learning models outperform traditional machine learning approaches based on handcrafted descriptors.

---

# 15.5.8 Why Raw Features Are Not Enough

Suppose we represent an atom only by its atomic number.

```text
Hydrogen

Atomic Number = 1
```

```text
Carbon

Atomic Number = 6
```

```text
Silicon

Atomic Number = 14
```

```text
Iron

Atomic Number = 26
```

These numbers uniquely identify the elements,

but they do **not** express chemical similarity.

For example,

consider

```text
Carbon = 6

Silicon = 14
```

Chemically,

carbon and silicon belong to the same group in the periodic table.

They exhibit

- similar valence configurations,
- comparable bonding behavior,
- analogous crystal structures.

However,

their atomic numbers differ by

8.

Now consider

```text
Iron = 26

Nickel = 28
```

Their atomic numbers differ by only

2,

yet their chemical behavior differs significantly in many applications.

The numerical distance between atomic numbers therefore has **no physical meaning**.

A neural network trained directly on atomic numbers may incorrectly interpret

```text
28

>

14

>

6

>

1
```

as an ordered numerical relationship,

which is not how chemistry works.

---

# 15.5.9 The Fundamental Idea Behind Embeddings

Instead of feeding raw atomic numbers into the Graph Network,

MEGNet first converts every element into a **dense vector**.

Conceptually,

the transformation is

```text
Atomic Number

↓

Embedding Layer

↓

High-Dimensional Vector
```

Instead of

```text
Silicon

↓

14
```

the network learns

```text
Silicon

↓

[-0.42,
 0.83,
-0.15,
...
 1.07]
```

This vector may contain

64,

128,

or even

256

learnable values.

Initially,

these values are random.

During training,

gradient descent continuously adjusts them until they encode chemically useful information.

---

# 15.5.10 What Does an Embedding Actually Learn?

This is one of the most important concepts in deep learning.

The embedding vector is **not manually designed**.

No scientist specifies

```text
Dimension 1 = Electronegativity

Dimension 2 = Atomic Radius

Dimension 3 = Valence Electrons
```

Instead,

the neural network discovers these representations automatically.

For example,

after sufficient training,

the learned embedding may implicitly encode

- atomic size,
- electronegativity,
- oxidation behavior,
- bonding preference,
- metallic character,
- electronic structure,
- periodic trends,

without ever being explicitly instructed to do so.

These learned representations are often called

**latent features**.

---

# 15.5.11 A Geometric Interpretation

Imagine that every chemical element is represented as a point in a

64-dimensional space.

Initially,

the locations are random.

```text
H

Li

C

O

Si

Fe

Ni
```

During training,

the optimizer gradually moves these points.

Eventually,

elements with similar chemical behavior become close together.

For example,

the learned space might resemble

```text
      C

         Si

O

           Ge

Fe

Co

Ni
```

Notice that

carbon,

silicon,

and germanium,

all members of Group 14,

may naturally cluster together.

Similarly,

transition metals often form another cluster.

The network discovers these relationships entirely from data.

---

# 15.5.12 Embedding as a Lookup Table

Mathematically,

an embedding layer is simply a learnable matrix.

Suppose

there are

95

chemical elements.

Suppose

the embedding dimension is

64.

The embedding matrix has dimensions

$$
95 \times 64.
$$

Each row corresponds to one element.

For example,

```text
Element

↓

Embedding Vector
```

```text
Hydrogen

↓

Row 1
```

```text
Carbon

↓

Row 6
```

```text
Iron

↓

Row 26
```

```text
Gold

↓

Row 79
```

During training,

every row of this matrix is updated independently.

Thus,

each element develops its own learned representation.

---

# 15.5.13 Mathematical Representation

Suppose

the embedding matrix is

$$
\mathbf{E}
\in
\mathbb{R}^{N_e \times d},
$$

where

- $N_e$ is the number of elements,
- $d$ is the embedding dimension.

If an atom has atomic number

$Z$,

its embedding is simply

$$
\mathbf{x}_Z
=
\mathbf{E}[Z].
$$

This notation means

"take the $Z$-th row of the embedding matrix."

Unlike ordinary matrix multiplication,

this operation is simply

an indexing operation.

It is therefore computationally inexpensive.

---

# 15.5.14 Implementing an Embedding Layer in PyTorch

PyTorch provides a dedicated module for embeddings.

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(

    num_embeddings=95,

    embedding_dim=64

)
```

Here,

```python
num_embeddings=95
```

means

the embedding table contains

95 rows,

one for each chemical element.

```python
embedding_dim=64
```

means

every element is represented by

a 64-dimensional vector.

Initially,

the embedding weights are random.

```python
print(embedding.weight.shape)
```

Output

```text
torch.Size([95, 64])
```

---

# 15.5.15 Looking Up Atomic Embeddings

Suppose our crystal contains

```text
Si

O

O

Mg
```

Their atomic numbers are

```python
atomic_numbers = torch.tensor([14, 8, 8, 12])
```

Passing them through the embedding layer is straightforward.

```python
node_features = embedding(

    atomic_numbers

)

print(node_features.shape)
```

Output

```text
torch.Size([4,64])
```

Notice what happened.

The input contained

only four integers.

The output contains

four learned feature vectors,

each with

64 dimensions.

No manual feature engineering was required.

---

# 15.5.16 Understanding the Output

Suppose

the first atom is silicon.

Its embedding might look like

```text
[-0.52,
 0.84,
-0.17,
...
 0.91]
```

The second atom,

oxygen,

may produce

```text
[ 1.23,
-0.44,
 0.18,
...
-0.61]
```

These numbers have **no direct physical interpretation individually**.

Their importance lies in their collective representation.

The Graph Network learns how to use these vectors during message passing.

Over time,

the embeddings evolve into highly informative representations of elemental chemistry.

---

# 15.5.17 Why Learned Embeddings Outperform Handcrafted Features

Traditional materials informatics often relies on manually selected descriptors such as

- atomic radius,
- electronegativity,
- ionization energy,
- electron affinity,
- valence electron count.

Although useful,

these descriptors assume that researchers already know which properties are important.

Deep learning follows a different philosophy.

Rather than prescribing the representation,

the network learns it directly from data.

This allows MEGNet to discover subtle relationships that may not be captured by predefined descriptors.

For sufficiently large datasets,

learned embeddings frequently outperform handcrafted feature engineering.

---

## **MEGNet Perspective**

In the original MEGNet model,

every atom first passes through an embedding layer before any message passing occurs.

Consequently,

the Graph Network blocks never operate directly on atomic numbers.

Instead,

they operate on learned latent vectors representing each chemical element.

These embeddings are optimized jointly with the rest of the network through backpropagation.

As training progresses,

the embedding space becomes an increasingly accurate representation of chemical similarity,

providing the Graph Network with a much richer starting point for learning atomic interactions.

In the next section, we will examine how **bond (edge) features** are embedded in MEGNet, which is substantially different from node embeddings because edge features are continuous quantities (such as interatomic distances) rather than discrete identifiers like atomic numbers.

# 15.5.18 Edge Embedding in MEGNet

In the previous section,

we learned how MEGNet transforms **atomic features** into dense vector representations using an embedding layer.

However,

atoms are only one part of a crystal graph.

The second essential component is the **edge**.

Recall that

an edge represents the interaction between two atoms.

```text
Atom A -------- Atom B
```

The quality of the learned node representations depends heavily on how accurately these interactions are represented.

Consequently,

MEGNet devotes considerable attention to constructing informative edge features.

Unlike node embeddings,

which operate on **discrete atomic identities**,

edge embeddings operate on **continuous geometric information**.

This difference fundamentally changes how the embedding process is designed.

---

# 15.5.19 Why Edge Features Are Different from Node Features

Nodes represent atoms.

Atoms belong to a finite set of chemical elements.

For example,

```text
H

He

Li

...

Fe

Cu

Zn
```

There are only a limited number of possible atomic identities.

Therefore,

a lookup table (embedding matrix) is sufficient.

Edges,

however,

represent relationships.

Relationships are not discrete.

Suppose two silicon atoms are separated by

```text
2.30 Å
```

Another pair may be separated by

```text
2.36 Å
```

Another by

```text
2.41 Å
```

There are infinitely many possible interatomic distances.

A lookup table cannot represent

an infinite number of values.

Therefore,

MEGNet must use a different strategy.

---

# 15.5.20 What Information Does an Edge Contain?

Consider two atoms.

```text
Si -------- O
```

The edge connecting them may contain

- bond distance,
- bond direction,
- bond order (if available),
- periodic image information,
- local geometric descriptors.

Among these,

the **interatomic distance** is by far the most important feature used in the original MEGNet architecture.

Suppose

the silicon-oxygen distance is

$$
d = 1.62\ \text{Å}
$$

This single number forms the basis of the edge representation.

---

# 15.5.21 Why We Cannot Feed Raw Distances Directly

One might ask,

> Why not simply use the bond distance itself?

For example,

```text
Edge Feature

↓

1.62
```

Although this appears reasonable,

it has several limitations.

First,

a single scalar provides only limited expressive power.

Second,

small numerical changes in distance can produce highly nonlinear changes in physical interactions.

For example,

the difference between

```text
1.60 Å

and

1.70 Å
```

may be chemically much more significant than

the difference between

```text
4.60 Å

and

4.70 Å.
```

A neural network operating directly on raw distances must learn these nonlinear relationships from scratch.

This makes training more difficult.

Instead,

MEGNet transforms every distance into a richer numerical representation before message passing begins.

---

# 15.5.22 Continuous Feature Expansion

Unlike atomic numbers,

continuous variables cannot be embedded using lookup tables.

Instead,

MEGNet expands each scalar distance into

a high-dimensional feature vector.

Conceptually,

the transformation is

```text
Distance

↓

Basis Expansion

↓

High-Dimensional Vector
```

Instead of

```text
1.62 Å

↓

1 Number
```

the network receives

```text
[

0.13,

0.82,

0.91,

0.27,

...

0.04

]
```

This richer representation allows the neural network to model complex distance-dependent interactions much more effectively.

---

# 15.5.23 The Gaussian Basis Expansion

The original MEGNet paper represents interatomic distances using a **Gaussian basis expansion**.

Rather than treating a distance as a single scalar,

MEGNet measures how strongly that distance activates several Gaussian functions positioned along the distance axis.

Graphically,

```text
Distance Axis

0Å--------------------------------6Å

      /\      /\      /\      /\

     /  \    /  \    /  \    /  \

    G1   G2   G3   G4   ...
```

Each Gaussian responds strongly only when the bond length lies close to its center.

Instead of producing

one value,

the distance generates

many responses.

These responses become the edge feature vector.

---

# 15.5.24 Mathematical Definition of a Gaussian Basis Function

Suppose

one Gaussian is centered at

$$
\mu_i.
$$

Its response to a distance

$d$

is

$$
e_i(d)
=
\exp
\left(
-
\frac{(d-\mu_i)^2}
{2\sigma^2}
\right),
$$

where

- $\mu_i$ is the center of the Gaussian,
- $\sigma$ controls its width.

Each Gaussian produces

one feature.

Using

$K$

Gaussians

produces

a

$K$

-dimensional edge representation.

---

# 15.5.25 Understanding the Equation

Consider

a Gaussian centered at

$$
\mu = 2.0\ \text{Å}.
$$

Suppose the bond length is

$$
d=2.0\ \text{Å}.
$$

Then

$$
(d-\mu)^2=0,
$$

and therefore

$$
e(d)=1.
$$

The Gaussian reaches its maximum value.

Now consider

$$
d=3.5\ \text{Å}.
$$

The distance lies far from the center.

Consequently,

$$
(d-\mu)^2
$$

becomes much larger,

causing the exponential to approach zero.

Thus,

each Gaussian responds only to a limited range of bond distances.

This localized response makes the representation much more expressive than using the raw distance directly.

---

# 15.5.26 Why Use Many Gaussians?

Suppose we use only one Gaussian.

```text
Distance

↓

One Response
```

The representation remains very limited.

Instead,

MEGNet employs multiple Gaussian basis functions.

For example,

their centers might be placed at

```text
0 Å

0.2 Å

0.4 Å

...

5.8 Å

6.0 Å
```

When a bond distance is evaluated,

every Gaussian produces an activation.

The resulting feature vector captures

where the bond lies

along the distance axis.

This process is analogous to expressing a signal in terms of basis functions,

a common idea in mathematics and physics.

---

# 15.5.27 Visualizing Gaussian Expansion

Suppose the bond distance is

```text
2.35 Å
```

The Gaussian responses might resemble

```text
Center

Response

0.0 Å

0.00

0.5 Å

0.01

1.0 Å

0.08

1.5 Å

0.42

2.0 Å

0.91

2.5 Å

0.87

3.0 Å

0.38

3.5 Å

0.06

4.0 Å

0.00
```

Instead of representing the bond using

```text
2.35
```

we now represent it using

a high-dimensional vector containing all of these responses.

This richer representation provides the Graph Network with much more information.

---

# 15.5.28 Implementing Gaussian Basis Expansion in PyTorch

We can implement the Gaussian basis expansion directly.

```python
import torch

class GaussianExpansion:

    def __init__(

        self,

        dmin,

        dmax,

        step,

        width

    ):

        self.centers = torch.arange(

            dmin,

            dmax + step,

            step

        )

        self.width = width

    def __call__(

        self,

        distances

    ):

        distances = distances.view(-1,1)

        return torch.exp(

            -((distances - self.centers)**2)

            /

            (2 * self.width**2)

        )
```

This class converts

a vector of bond distances

into

their Gaussian basis representations.

---

# 15.5.29 Applying Gaussian Expansion

Suppose

our graph contains

four bond distances.

```python
distances = torch.tensor([

    1.62,

    2.10,

    2.87,

    3.45

])
```

Create the Gaussian basis.

```python
gaussian = GaussianExpansion(

    dmin=0,

    dmax=5,

    step=0.2,

    width=0.2

)
```

Now expand the distances.

```python
edge_features = gaussian(

    distances

)

print(edge_features.shape)
```

Output

```text
torch.Size([4,26])
```

Each bond is no longer represented by

a single scalar.

Instead,

it is represented by

26 continuous features,

one from each Gaussian basis function.

These expanded edge features become the input to the Graph Network blocks.

---

## **MEGNet Perspective**

The original MEGNet architecture does **not** feed raw interatomic distances directly into the edge update network.

Instead,

every distance undergoes a Gaussian basis expansion,

producing a smooth, high-dimensional representation of bond length.

This transformation greatly improves the network's ability to learn subtle distance-dependent interactions,

which are essential for accurately predicting material properties such as formation energy, band gap, elastic constants, and phonon-related quantities.

In the next section, we will examine **why Gaussian basis functions were chosen**, how the parameters (centers, spacing, and width) influence the representation, and why this seemingly simple preprocessing step has such a profound impact on the performance and stability of MEGNet.

# 15.5.30 Why Gaussian Basis Functions?

In the previous section, we learned that MEGNet does **not** use raw interatomic distances directly.

Instead,

every bond distance is transformed using a **Gaussian basis expansion**.

At first glance,

this may appear to be merely a mathematical trick.

However,

there are deep mathematical,

physical,

and machine learning reasons behind this design choice.

Understanding these reasons is essential because almost every modern materials graph neural network—including **MEGNet, M3GNet, CHGNet, MatGL, and several molecular graph networks**—uses some form of continuous basis expansion.

Therefore,

Gaussian expansion is not simply a preprocessing step.

It is one of the foundational ideas of modern atomistic deep learning.

---

# 15.5.31 The Fundamental Problem

Suppose two bonds have lengths

$$
2.00\ \text{Å}
$$

and

$$
2.01\ \text{Å}.
$$

Physically,

these two bonds are almost identical.

Now consider

$$
2.00\ \text{Å}
$$

and

$$
5.50\ \text{Å}.
$$

These represent completely different atomic interactions.

A neural network receiving raw distances

```text
2.00

2.01

5.50
```

must somehow discover

- similarity,
- locality,
- smoothness,

entirely by itself.

This is an unnecessarily difficult learning problem.

Instead,

we would like the input representation itself to encode these properties.

Gaussian basis functions accomplish exactly that.

---

# 15.5.32 Smooth Representations

Consider a Gaussian centered at

$$
2.0\ \text{Å}.
$$

Suppose

the distance changes only slightly.

```text
2.00 Å

↓

2.01 Å
```

The Gaussian response changes only slightly.

Likewise,

```text
2.01 Å

↓

2.02 Å
```

again produces only a very small change.

Graphically,

```text
Response

1.0 |          /\

    |         /  \

    |        /    \

0.5 |-------/------\-------

    |      /        \

0.0 +---------------------------->

        Distance
```

The curve changes continuously.

There are no sudden jumps.

This smoothness is extremely important because

most physical properties also vary smoothly with atomic positions.

---

# 15.5.33 Physical Motivation

Imagine stretching a bond.

```text
Si ---- O
```

becomes

```text
Si ----- O
```

The bond energy does **not** suddenly jump.

Instead,

it changes gradually.

Similarly,

electronic structure,

force,

and local potential

also change continuously.

If our input representation changes smoothly,

the neural network can learn these physical relationships much more efficiently.

Thus,

Gaussian basis functions naturally reflect

the continuity of real atomic interactions.

---

# 15.5.34 Locality

Another important property of Gaussian functions is

**locality**.

Suppose

one Gaussian is centered at

$$
2.0\ \text{Å}.
$$

It responds strongly only near

$$
2.0\ \text{Å}.
$$

Distances far away contribute almost nothing.

Graphically,

```text
Center

↓

2.0 Å

↓

Large Response
```

```text
5.0 Å

↓

Almost Zero Response
```

Therefore,

each Gaussian specializes in

a particular region of distance space.

The collection of many Gaussians forms

a set of local detectors.

This idea is analogous to

convolutional filters in image processing,

where each filter detects

particular local patterns.

---

# 15.5.35 Overlapping Basis Functions

One Gaussian alone cannot represent every distance.

Therefore,

multiple Gaussians overlap.

Graphically,

```text
            G1

          /\

         /  \

        /    \

      /\      /\

     /  \    /  \

    /    \  /    \

---/------\/------\----------------

   G2      G3      G4
```

Suppose

the distance is

$$
2.3\ \text{Å}.
$$

Then

- Gaussian centered at 2.2 Å responds strongly.
- Gaussian centered at 2.4 Å also responds strongly.
- Nearby Gaussians respond moderately.
- Distant Gaussians contribute almost nothing.

Instead of assigning the bond to

one discrete category,

the representation smoothly interpolates between neighboring basis functions.

This greatly improves numerical stability.

---

# 15.5.36 Why Not Use One-Hot Encoding?

One alternative would be

distance bins.

For example,

```text
0–1 Å

1–2 Å

2–3 Å

3–4 Å
```

A bond at

2.1 Å

might become

```text
[0,0,1,0]
```

while

2.9 Å

also becomes

```text
[0,0,1,0]
```

These two distances are treated as identical.

Even worse,

consider

```text
1.99 Å

↓

[0,1,0,0]
```

and

```text
2.01 Å

↓

[0,0,1,0]
```

Although the distances differ by only

0.02 Å,

their representations become completely different.

This discontinuity makes learning much more difficult.

Gaussian basis expansion avoids this problem entirely.

---

# 15.5.37 Gaussian Expansion Produces Continuous Features

Suppose

the distance gradually changes

```text
2.00

↓

2.05

↓

2.10

↓

2.15
```

The Gaussian feature vector changes

smoothly.

Every feature evolves continuously.

This continuity is especially important for

- energy prediction,
- force prediction,
- molecular dynamics,

where tiny atomic displacements occur constantly.

Indeed,

later models such as

M3GNet

use this smooth behavior to predict

atomic forces,

which require differentiating the neural network output with respect to atomic positions.

Without smooth basis functions,

this would be extremely difficult.

---

# 15.5.38 Influence of the Gaussian Width ($\sigma$)

Recall the Gaussian equation

$$
e_i(d)
=
\exp
\left(
-
\frac{(d-\mu_i)^2}
{2\sigma^2}
\right).
$$

The parameter

$$
\sigma
$$

controls

the width of each Gaussian.

---

## Small Width

Suppose

$$
\sigma=0.05.
$$

Each Gaussian becomes

very narrow.

```text
        /\

       /  \

------/----\------------
```

Only distances extremely close to the center produce large responses.

Advantages

- high resolution,
- precise localization.

Disadvantages

- sparse features,
- sensitive to small numerical noise,
- harder optimization.

---

## Large Width

Suppose

$$
\sigma=1.0.
$$

Now each Gaussian becomes

very broad.

```text
      ________

    /          \

---/------------\------------
```

Advantages

- smoother features,
- better numerical stability.

Disadvantages

- neighboring distances become difficult to distinguish,
- reduced spatial resolution.

---

# 15.5.39 Choosing the Gaussian Centers

Another important design decision concerns

the Gaussian centers.

Suppose

the cutoff radius is

$$
5\ \text{Å}.
$$

The centers may be

```text
0.0

0.2

0.4

...

4.8

5.0
```

Uniform spacing ensures

that every physically relevant bond length

activates several nearby Gaussians.

If the spacing becomes too large,

important geometric information may be lost.

If it becomes too small,

the feature dimension increases unnecessarily,

raising computational cost without significant performance gains.

Thus,

the spacing,

number of Gaussians,

and Gaussian width

must be chosen together.

---

# 15.5.40 Implementing Gaussian Expansion from Scratch

The previous implementation used broadcasting.

Let us examine the computation explicitly.

```python
import torch

distances = torch.tensor([

    1.6,

    2.3,

    3.1

])

centers = torch.arange(

    0,

    5.2,

    0.2

)

sigma = 0.2

expanded = []

for d in distances:

    responses = torch.exp(

        -((d - centers)**2)

        /

        (2 * sigma**2)

    )

    expanded.append(

        responses

    )

expanded = torch.stack(

    expanded

)

print(expanded.shape)
```

Output

```text
torch.Size([3,26])
```

Although this implementation is slower than the vectorized version,

it clearly illustrates

how every distance is compared with

every Gaussian center.

---

# 15.5.41 Computational Interpretation

Suppose

there are

1000 edges

and

100 Gaussian basis functions.

The expansion produces

```text
1000

↓

100000 Gaussian Responses
```

Although this seems computationally expensive,

modern tensor libraries perform these operations very efficiently on GPUs.

Furthermore,

this preprocessing cost is negligible compared with the computational cost of the Graph Network blocks themselves.

---

## **MEGNet Perspective**

The Gaussian basis expansion is one of the signature design choices of the original MEGNet architecture.

Rather than forcing the neural network to infer nonlinear distance relationships directly from raw bond lengths,

MEGNet provides a smooth, localized, high-dimensional representation of geometric information.

This representation significantly improves optimization,

captures the continuous nature of atomic interactions,

and enables the Graph Network blocks to learn chemically meaningful distance-dependent message passing.

Although later models such as M3GNet replace Gaussian basis functions with more sophisticated basis expansions (including spherical and directional bases),

the underlying principle remains exactly the same:

**continuous geometric information should first be transformed into a richer latent representation before message passing begins.**

The next section will integrate everything we have learned so far by constructing the **complete input graph for MEGNet**, combining node embeddings, Gaussian-expanded edge features, and the global state into the data structure that enters the first Graph Network block.

# 15.5.42 Constructing the Complete Input Graph

At this point, we have studied every individual component that forms the input of MEGNet.

Specifically, we now understand

- how atoms become node features,
- how bond distances become edge features,
- how global information is represented,
- how Graph Network blocks update these representations.

The next objective is to assemble all of these pieces into **one complete graph**.

This graph is the actual object that enters the first Graph Network block.

Everything we have studied so far converges here.

---

# 15.5.43 From Crystal Structure to Graph

Suppose we begin with a crystal structure.

```text
Crystal

↓

Atomic Coordinates

↓

Lattice Vectors

↓

Atomic Species
```

This information alone is not directly usable by a Graph Neural Network.

The crystal must first be converted into a graph.

Graph construction consists of four major steps.

```text
Crystal

↓

Determine Nodes

↓

Determine Edges

↓

Compute Features

↓

Graph Object
```

Each step will now be studied in detail.

---

# 15.5.44 Step 1 — Creating the Nodes

Every atom becomes one node.

Suppose we have magnesium oxide.

```text
Mg

O

O

Mg
```

The graph therefore contains

```text
Node 0 → Mg

Node 1 → O

Node 2 → O

Node 3 → Mg
```

If the crystal contains

100 atoms,

the graph contains

100 nodes.

Nothing more complicated than that.

---

# 15.5.45 Step 2 — Determining the Edges

Next,

we determine

which atoms interact.

The original MEGNet uses a distance cutoff.

Suppose

the cutoff radius is

$$
r_c=4.0\ \text{Å}.
$$

Every atom whose distance is less than

$$
4.0\ \text{Å}
$$

is connected by an edge.

Graphically,

```text
Mg -------- O

Distance = 2.05 Å

↓

Edge Exists
```

```text
Mg ----------------------- O

Distance = 6.20 Å

↓

No Edge
```

Thus,

the crystal graph stores only local interactions.

This greatly reduces computational cost.

---

# 15.5.46 Directed Edges

Recall that MEGNet represents every bond using **two directed edges**.

Instead of

```text
Mg -------- O
```

the graph stores

```text
Mg → O

O → Mg
```

Why?

Because

during message passing,

every edge updates

the receiver node.

Having both directions allows

both atoms

to receive information.

Suppose

our crystal contains

50 undirected bonds.

The graph actually stores

100 directed edges.

---

# 15.5.47 Constructing the Edge Index

PyTorch Geometric stores graph connectivity using

```python
edge_index
```

Suppose

our graph contains

```text
0 → 1

1 → 0

1 → 2

2 → 1
```

The tensor becomes

```python
edge_index = torch.tensor([

    [0,1,1,2],

    [1,0,2,1]

])
```

The first row

contains senders.

The second row

contains receivers.

This compact representation allows

efficient GPU computation.

---

# 15.5.48 Step 3 — Computing Node Features

Each node stores

its atomic identity.

Initially,

suppose

```python
atomic_numbers = torch.tensor([

    12,

    8,

    8,

    12

])
```

representing

```text
Mg

O

O

Mg
```

These atomic numbers pass through

the embedding layer.

```python
node_features = node_embedding(

    atomic_numbers

)
```

Suppose

the embedding dimension is

64.

Then

```text
node_features

↓

torch.Size([4,64])
```

Every atom now possesses

a learned

64-dimensional feature vector.

---

# 15.5.49 Step 4 — Computing Edge Features

Suppose

the bond distances are

```python
distances = torch.tensor([

    2.05,

    2.05,

    2.10,

    2.10

])
```

Each distance undergoes

Gaussian basis expansion.

```python
edge_features = gaussian(

    distances

)
```

Suppose

26 Gaussian basis functions are used.

The resulting tensor becomes

```text
torch.Size([4,26])
```

Notice

the edge dimension

is different from

the node dimension.

This is perfectly acceptable.

The edge update network

learns how to combine them.

---

# 15.5.50 Step 5 — Initializing the Global State

Suppose

temperature

and

pressure

are available.

The global vector might be

```python
global_state = torch.tensor([

    300.0,

    1.0

])
```

Alternatively,

if no global information exists,

MEGNet simply initializes

the global state as

```python
global_state = torch.zeros(

    global_dim

)
```

The Graph Network blocks will learn

a meaningful global representation

during training.

---

# 15.5.51 Combining Everything

At this stage,

the graph contains

four components.

```text
Node Features

↓

[N_nodes,node_dim]
```

```text
Edge Features

↓

[N_edges,edge_dim]
```

```text
Edge Index

↓

[2,N_edges]
```

```text
Global State

↓

[global_dim]
```

Together,

these completely define

the graph.

---

# 15.5.52 Creating a PyTorch Geometric Data Object

PyTorch Geometric stores graphs using

the

`Data`

class.

```python
from torch_geometric.data import Data

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

stores node embeddings.

```python
edge_index
```

stores connectivity.

```python
edge_attr
```

stores edge embeddings.

Notice that

the standard

`Data`

object has

no dedicated field

for the global state.

---

# 15.5.53 Storing the Global State

We can simply attach

the global vector

as another attribute.

```python
graph.u = global_state
```

Now the graph contains

```python
graph.x

graph.edge_index

graph.edge_attr

graph.u
```

These correspond exactly to

$$
\mathbf{v},
\mathbf{e},
\mathbf{u}
$$

from the Graph Network equations.

---

# 15.5.54 Inspecting the Graph

Printing the graph

may produce

```text
Data(

x=[4,64],

edge_index=[2,4],

edge_attr=[4,26],

u=[2]

)
```

This concise summary contains

all information needed

by the first Graph Network block.

---

# 15.5.55 Complete Graph Construction Pipeline

The entire preprocessing pipeline can now be summarized.

```text
Crystal Structure

↓

Neighbor Search

↓

Edge Construction

↓

Atomic Numbers

↓

Node Embedding

↓

Node Features

↓

Bond Distances

↓

Gaussian Expansion

↓

Edge Features

↓

Global Features

↓

Graph Object

↓

Graph Network Block
```

This pipeline converts

raw crystallographic information

into

the mathematical representation

used throughout MEGNet.

---

# 15.5.56 End-to-End Graph Construction Example

The following code combines every step discussed so far.

```python
import torch

from torch_geometric.data import Data

# Atomic numbers

atomic_numbers = torch.tensor([

    12,

    8,

    8,

    12

])

# Node embedding

node_features = node_embedding(

    atomic_numbers

)

# Connectivity

edge_index = torch.tensor([

    [0,1,1,2],

    [1,0,2,1]

])

# Bond distances

distances = torch.tensor([

    2.05,

    2.05,

    2.10,

    2.10

])

# Gaussian basis expansion

edge_features = gaussian(

    distances

)

# Global state

global_state = torch.tensor([

    300.0,

    1.0

])

# Graph

graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_features

)

graph.u = global_state

print(graph)
```

Output

```text
Data(

x=[4,64],

edge_index=[2,4],

edge_attr=[4,26],

u=[2]

)
```

Although this example is intentionally small,

exactly the same workflow scales to crystals containing

hundreds

or

thousands

of atoms.

The only difference is the size of the tensors.

The mathematical formulation remains unchanged.

---

## **MEGNet Perspective**

At this stage, we have completely constructed the input expected by the MEGNet architecture.

Starting from a crystal structure, we

1. identify neighboring atoms,
2. build the graph connectivity,
3. embed atomic identities,
4. expand bond distances using Gaussian basis functions,
5. initialize the global state,
6. package everything into a graph object.

This graph is the direct input to the first Graph Network block.

From this point onward, the network performs only neural computations—message passing, feature updates, aggregation, and prediction.

In the next section, we will begin constructing the **complete MEGNet neural network architecture in PyTorch**, assembling embedding layers, Graph Network blocks, residual connections, readout operations, and prediction heads into a single trainable model.

# 15.6 Building the Complete MEGNet Neural Network in PyTorch

Everything we have learned so far has been studied as independent components.

We have implemented

- node embeddings,
- Gaussian-expanded edge features,
- edge update networks,
- node update networks,
- global update networks,
- Graph Network blocks.

However,

none of these components alone constitute MEGNet.

A neural network becomes useful only after every component is connected into one computational graph.

This section focuses on constructing the **entire MEGNet architecture** exactly as it operates during training and inference.

Instead of studying isolated modules,

we will now assemble a complete deep learning model.

---

# 15.6.1 The Complete Data Flow Inside MEGNet

Before writing code,

it is important to understand the overall computation.

Suppose the input is a crystal structure.

The forward propagation follows

```text
Crystal Structure

↓

Graph Construction

↓

Node Embedding

↓

Edge Gaussian Expansion

↓

Graph Network Block 1

↓

Graph Network Block 2

↓

Graph Network Block 3

↓

Readout Layer

↓

Prediction Head

↓

Material Property
```

Every arrow represents a tensor transformation.

Nothing is skipped.

Every component participates in gradient computation.

---

# 15.6.2 Breaking the Network into Modules

Rather than writing one enormous neural network,

MEGNet is naturally divided into independent modules.

```text
MEGNet

├── Node Embedding

├── Edge Expansion

├── Graph Block 1

├── Graph Block 2

├── Graph Block 3

├── Readout

└── Prediction Head
```

This modular design offers several advantages.

- Easier debugging.
- Better code readability.
- Independent testing.
- Reusable components.
- Simpler research modifications.

Modern deep learning libraries encourage this modular programming style.

---

# 15.6.3 Choosing Network Dimensions

Every neural network requires hidden dimensions.

Suppose we choose

```text
Node Dimension

64

Edge Dimension

64

Global Dimension

32

Hidden Dimension

128
```

These dimensions remain constant throughout the network.

Keeping dimensions fixed greatly simplifies implementation because every Graph Network block expects identical input sizes.

---

# 15.6.4 The Complete Architecture Class

Every neural network in PyTorch inherits from

```python
nn.Module
```

The complete architecture begins as

```python
import torch
import torch.nn as nn

class MEGNet(nn.Module):

    def __init__(

        self,

        num_elements,

        node_dim,

        edge_dim,

        global_dim,

        hidden_dim,

        num_blocks

    ):

        super().__init__()
```

Everything that follows inside

`__init__`

defines the trainable components of the network.

---

# 15.6.5 Creating the Node Embedding Layer

The first learnable component is

the atomic embedding.

```python
self.node_embedding = nn.Embedding(

    num_embeddings=num_elements,

    embedding_dim=node_dim

)
```

Suppose

```text
num_elements = 95

node_dim = 64
```

The embedding matrix becomes

```text
95 × 64
```

Every chemical element has its own learnable vector.

---

# 15.6.6 Why Edge Features Do Not Need an Embedding Matrix

Notice that we do **not** create

```python
nn.Embedding
```

for the edge features.

Why?

Because

edge features are already continuous vectors after Gaussian expansion.

Suppose

26 Gaussian basis functions are used.

The edge feature already has the form

```text
[

0.02,

0.11,

0.58,

...

0.01

]
```

Instead of a lookup table,

MEGNet projects these vectors into the hidden feature space using a small multilayer perceptron.

---

# 15.6.7 Edge Projection Network

```python
self.edge_projection = nn.Sequential(

    nn.Linear(

        26,

        hidden_dim

    ),

    nn.ReLU(),

    nn.Linear(

        hidden_dim,

        edge_dim

    )

)
```

Suppose

```text
26

↓

128

↓

64
```

Every bond is now represented by

64 learned features,

matching the dimensionality expected by the Graph Network blocks.

---

# 15.6.8 Global Feature Projection

The global vector may initially contain only a few physical quantities.

For example,

```text
Temperature

Pressure
```

Only

two numbers.

The Graph Network,

however,

expects

a higher-dimensional global representation.

Therefore,

MEGNet also projects the global features.

```python
self.global_projection = nn.Sequential(

    nn.Linear(

        2,

        hidden_dim

    ),

    nn.ReLU(),

    nn.Linear(

        hidden_dim,

        global_dim

    )

)
```

This transforms

```text
2

↓

128

↓

32
```

producing the initial global embedding.

---

# 15.6.9 Why Every Feature Type Is Projected

Notice the symmetry.

Nodes

↓

Embedding Layer

Edges

↓

Projection Network

Global State

↓

Projection Network

Eventually,

every object has a compatible latent representation.

```text
Nodes

↓

64

Edges

↓

64

Global

↓

32
```

This greatly simplifies later computations.

Every Graph Network block receives tensors with identical dimensions.

---

# 15.6.10 Constructing Multiple Graph Network Blocks

The heart of MEGNet is

not one Graph Network block,

but several.

Instead of writing

```python
self.block1

self.block2

self.block3
```

manually,

PyTorch provides

`ModuleList`.

```python
self.blocks = nn.ModuleList(

    [

        GraphNetworkBlock(

            edge_update=EdgeUpdate(

                node_dim,

                edge_dim,

                global_dim,

                hidden_dim

            ),

            node_update=NodeUpdate(

                node_dim,

                edge_dim,

                global_dim,

                hidden_dim

            ),

            global_update=GlobalUpdate(

                edge_dim,

                node_dim,

                global_dim,

                hidden_dim

            )

        )

        for _ in range(num_blocks)

    ]

)
```

Suppose

```text
num_blocks = 3
```

The architecture automatically constructs

```text
Graph Block 1

Graph Block 2

Graph Block 3
```

Each block has

its own trainable parameters.

Although the architecture is identical,

the weights are different.

---

# 15.6.11 Why Use ModuleList?

Suppose we instead wrote

```python
self.block1

self.block2

self.block3
```

The code would quickly become difficult to maintain.

Imagine experimenting with

```text
8 Blocks

12 Blocks

16 Blocks
```

The architecture would need to be rewritten every time.

Instead,

`ModuleList`

allows

```python
for block in self.blocks:
```

making the implementation completely scalable.

This design is standard in modern deep learning.

---

# 15.6.12 The Architecture Constructed So Far

At this point,

our network contains

```text
MEGNet

├── Node Embedding

├── Edge Projection

├── Global Projection

├── Graph Block 1

├── Graph Block 2

├── Graph Block 3
```

Notice something important.

The network still cannot make predictions.

After the Graph Network blocks,

we must still

- summarize the graph,
- produce one graph embedding,
- predict the target property.

These remaining components constitute

the second half of the MEGNet architecture.

---

## **MEGNet Perspective**

The initialization phase of the MEGNet model defines **what the network is made of**, not **how it computes**.

At this stage, we have created all of the trainable building blocks:

- an embedding table for atoms,
- projection networks for bond and global features,
- a configurable stack of Graph Network blocks.

Nothing has yet passed through the network.

The actual computation begins in the **forward()** method, where tensors flow through these components in sequence.

The next section will implement the complete `forward()` function, showing exactly how data moves through the entire MEGNet architecture during prediction.

# 15.6.13 Implementing the Forward Pass

The `__init__()` function defines **what layers exist** in the network.

However,

a neural network cannot perform any computation until we define its **forward pass**.

The `forward()` function describes

how data flows

through every layer of the network.

In other words,

`__init__()` builds the machine,

while `forward()` explains

how the machine operates.

Every time we execute

```python
prediction = model(graph)
```

PyTorch automatically calls

```python
forward()
```

Therefore,

this function is the computational heart of the MEGNet implementation.

---

# 15.6.14 Inputs to the Forward Function

Our graph contains four major components.

```python
graph.x
```

Node information.

```python
graph.edge_index
```

Graph connectivity.

```python
graph.edge_attr
```

Edge information.

```python
graph.u
```

Global state.

Therefore,

our forward function begins as

```python
def forward(

    self,

    graph

):
```

The graph object already contains everything required for prediction.

---

# 15.6.15 Extracting Graph Components

The first operation is simply unpacking the graph.

```python
x = graph.x

edge_index = graph.edge_index

edge_attr = graph.edge_attr

u = graph.u
```

At this stage,

suppose the tensor dimensions are

```text
x

↓

[N_nodes]
```

```text
edge_attr

↓

[N_edges,26]
```

```text
edge_index

↓

[2,N_edges]
```

```text
u

↓

[2]
```

These are still

the raw graph features.

The next step transforms them into latent representations.

---

# 15.6.16 Embedding the Nodes

The atomic numbers stored inside

```python
graph.x
```

must first be converted into learned embeddings.

```python
x = self.node_embedding(

    x

)
```

Suppose

the crystal contains

120 atoms.

The tensor changes

from

```text
[120]
```

to

```text
[120,64]
```

Every atom is now represented

by a

64-dimensional learned vector.

---

# 15.6.17 Projecting Edge Features

The Gaussian-expanded edge features

are passed through

the projection network.

```python
edge_attr = self.edge_projection(

    edge_attr

)
```

Suppose

the Gaussian expansion produced

26 features.

The projection performs

```text
26

↓

128

↓

64
```

Consequently,

the edge tensor changes

from

```text
[N_edges,26]
```

to

```text
[N_edges,64]
```

Now

both nodes

and

edges

occupy

the same latent feature space.

---

# 15.6.18 Projecting the Global State

The global vector

is processed similarly.

```python
u = self.global_projection(

    u

)
```

Suppose

the original global vector contains

```text
Temperature

Pressure
```

The projection transforms

```text
2

↓

128

↓

32
```

The graph now possesses

```text
Nodes

64

Edges

64

Global

32
```

These dimensions match

the expectations of the Graph Network blocks.

---

# 15.6.19 Passing Through Graph Network Blocks

Now begins

the actual message passing.

Instead of calling

each block individually,

we iterate through

the

`ModuleList`.

```python
for block in self.blocks:

    x,

    edge_attr,

    u = block(

        x,

        edge_attr,

        edge_index,

        u

    )
```

Each iteration performs

```text
Edge Update

↓

Node Update

↓

Global Update
```

After the first iteration,

the graph becomes

```text
Updated Graph
```

This updated graph

becomes

the input

to the second Graph Network block.

This process repeats

until all blocks have been executed.

---

# 15.6.20 Understanding the Iterative Refinement

Suppose

three Graph Network blocks are used.

The computation becomes

```text
Initial Graph

↓

Graph Block 1

↓

Graph Block 2

↓

Graph Block 3

↓

Final Graph
```

Notice

that the tensor dimensions remain unchanged.

For example,

```text
Nodes

120 × 64

↓

120 × 64

↓

120 × 64

↓

120 × 64
```

Only the **values**

change.

Every Graph Network block

learns progressively richer representations.

---

# 15.6.21 What Changes Inside the Blocks?

Although the dimensions remain fixed,

the information contained inside the tensors becomes increasingly sophisticated.

Initially,

a node embedding mainly represents

its own atomic identity.

After one Graph Network block,

the node also contains information about

its nearest neighbors.

After two Graph Network blocks,

the node additionally contains

information propagated from

second-nearest neighbors.

After three Graph Network blocks,

even more distant regions of the crystal

influence the representation.

Thus,

the node embeddings become

progressively more informative

after every layer.

---

# 15.6.22 The Graph After Message Passing

After all Graph Network blocks,

our graph contains

```text
Node Embeddings

↓

Chemically Enriched
```

```text
Edge Embeddings

↓

Interaction Enriched
```

```text
Global Embedding

↓

Crystal-Level Representation
```

These tensors now encode

far more information

than the original graph.

However,

the network still has one remaining challenge.

Suppose

the crystal contains

```text
250 atoms.
```

The network still possesses

250 node embeddings.

Yet

formation energy

is

a **single number**.

Therefore,

the graph must now be compressed.

---

# 15.6.23 Why Compression Is Necessary

Imagine predicting

formation energy.

The desired output is

```text
-3.42 eV/atom
```

Not

```text
250 separate predictions.
```

Therefore,

the network must transform

```text
250 Node Embeddings

↓

One Crystal Embedding
```

This operation is called

the **readout**.

Without a readout layer,

graph-level property prediction would be impossible.

---

# 15.6.24 The State of the Network Before Readout

Immediately before the readout operation,

the network looks like

```text
Crystal Graph

↓

Embedding

↓

Graph Block 1

↓

Graph Block 2

↓

Graph Block 3

↓

Updated Node Features

Updated Edge Features

Updated Global State
```

Everything required for prediction has already been learned.

The only remaining task is

to summarize

the graph

into

a single fixed-size representation.

---

## **MEGNet Perspective**

The `forward()` function transforms a raw crystal graph into a hierarchy of increasingly informative latent representations.

First,

raw atomic identities,

bond distances,

and global descriptors are embedded into continuous feature spaces.

Next,

multiple Graph Network blocks iteratively refine these representations through message passing.

By the end of this stage,

every node embedding contains information that extends well beyond its local atomic environment,

capturing complex structural and chemical relationships throughout the crystal.

The next component of the architecture—the **readout layer**—will aggregate these learned node, edge, and global representations into a single crystal-level vector suitable for predicting material properties.

# 15.6.13 Forward Propagation in MEGNet

In the previous sections, we built the entire MEGNet architecture by defining its individual components. We created

- the node embedding layer,
- the edge projection network,
- the global projection network,
- multiple Graph Network blocks.

However, defining these components is only the first half of constructing a neural network.

A neural network is not merely a collection of layers.

It is a **computational graph**.

The architecture defines **what computations are available**.

The forward pass defines **how those computations are executed**.

Without a forward pass, the network has no behavior.

It is simply a collection of disconnected mathematical operations.

Therefore, before implementing the `forward()` function, we must understand what **forward propagation** actually means inside a Graph Neural Network.

---

# 15.6.14 What Is Forward Propagation?

Forward propagation is the process of transforming an input into an output by passing data through every layer of the neural network.

For a conventional multilayer perceptron (MLP), the process is straightforward.

```text
Input

↓

Linear Layer

↓

Activation

↓

Linear Layer

↓

Prediction
```

Each layer receives the output of the previous layer.

The data always flows in one direction.

This process is called **forward propagation** because information propagates from the input toward the output.

---

In convolutional neural networks (CNNs), the idea is identical.

```text
Image

↓

Convolution

↓

Pooling

↓

Convolution

↓

Fully Connected Layer

↓

Prediction
```

Although the operations differ, the computational philosophy remains the same.

Every layer transforms the representation produced by the previous layer.

---

Graph Neural Networks follow exactly the same principle.

The difference is that the data is no longer arranged as vectors or images.

Instead, the input is a graph.

Consequently, every layer must update

- node features,
- edge features,
- global features,

while respecting the graph connectivity.

---

# 15.6.15 Forward Propagation Inside MEGNet

Forward propagation in MEGNet consists of repeatedly transforming a crystal graph into increasingly informative latent representations.

The entire computation can be summarized as

```text
Crystal Structure

↓

Graph Construction

↓

Node Embedding

↓

Edge Projection

↓

Global Projection

↓

Graph Network Block

↓

Graph Network Block

↓

Graph Network Block

↓

Readout

↓

Prediction Head

↓

Material Property
```

Notice that the crystal itself never changes.

Only its numerical representation changes.

Initially,

the graph contains raw physical information.

Eventually,

it contains highly abstract learned representations suitable for property prediction.

---

# 15.6.16 What Changes During Forward Propagation?

Suppose we begin with a silicon crystal.

Initially, every node stores only

```text
Atomic Number
```

Every edge stores only

```text
Bond Distance
```

The global state may contain only

```text
Temperature

Pressure
```

At this stage, the graph contains almost no learned information.

It merely stores physical descriptors extracted from the crystal.

---

After the embedding layers,

the graph changes dramatically.

Instead of

```text
Si

↓

14
```

the node becomes

```text
Si

↓

[-0.34,

 0.81,

 ...

 0.27]
```

Similarly,

every bond distance

```text
2.35 Å
```

becomes

a high-dimensional latent vector.

The graph now consists entirely of learned numerical representations.

---

Next,

the Graph Network blocks begin exchanging information between neighboring atoms.

Initially,

an atom knows only about itself.

```text
Si
```

After one Graph Network block,

the atom also contains information about

its nearest neighbors.

```text
O

↓

Si

↓

O
```

After two Graph Network blocks,

information from atoms two hops away begins to influence the representation.

```text
Mg

↓

O

↓

Si

↓

O

↓

Mg
```

With every Graph Network block,

the receptive field expands.

The graph representation becomes increasingly rich.

---

# 15.6.17 The Computational Graph

Although we often draw the architecture as a sequence of layers,

PyTorch internally represents the computation differently.

Every operation becomes a node in a computational graph.

For example,

```python
x = self.node_embedding(x)

edge_attr = self.edge_projection(edge_attr)

u = self.global_projection(u)
```

creates a graph similar to

```text
Atomic Numbers

↓

Embedding

↓

Node Features

↓

Graph Block

↓

Readout

↓

Prediction

↓

Loss
```

Every operation records

- its input,
- its output,
- the mathematical operation performed.

PyTorch stores this information automatically.

This stored computation is called the **dynamic computational graph**.

---

# 15.6.18 Why the Computational Graph Is Important

The computational graph serves one essential purpose.

It enables **automatic differentiation**.

Suppose the network predicts

```text
Formation Energy

=

−3.12 eV
```

while the true value is

```text
−3.45 eV.
```

The prediction error is

```text
Loss
```

Backpropagation must determine

how every parameter contributed to this error.

Without remembering every operation performed during forward propagation,

this would be impossible.

The computational graph stores exactly the information required to compute derivatives.

---

# 15.6.19 Data Dependency

One of the most important ideas in deep learning is that every layer depends on the output of previous layers.

For example,

the second Graph Network block cannot begin until the first block has finished.

```text
Block 1

↓

Block 2

↓

Block 3
```

Similarly,

the readout layer cannot operate until every Graph Network block has updated the graph.

Finally,

the prediction head cannot make a prediction until the readout layer has produced a graph-level representation.

Every stage depends on all preceding stages.

---

# 15.6.20 Forward Propagation During Training

During training,

the forward pass performs two tasks simultaneously.

First,

it computes the prediction.

Second,

it records every mathematical operation inside the computational graph.

Conceptually,

training consists of

```text
Forward Pass

↓

Prediction

↓

Loss

↓

Backpropagation

↓

Parameter Update
```

The forward pass therefore prepares everything needed for gradient computation.

---

# 15.6.21 Forward Propagation During Inference

Inference differs from training.

During inference,

the network still performs

```text
Forward Pass

↓

Prediction
```

However,

no gradients are required.

No computational graph needs to be stored.

Consequently,

inference requires significantly less memory than training.

This is why PyTorch provides

```python
with torch.no_grad():
```

during model evaluation.

Disabling gradient tracking reduces memory consumption and improves prediction speed.

---

# 15.6.22 Why Understanding the Forward Pass Matters

Many beginners think that implementing a neural network is equivalent to defining several layers inside `__init__()`.

This is incorrect.

The intelligence of a neural network lies in the **flow of information**.

The forward pass determines

- which tensors interact,
- when message passing occurs,
- how representations evolve,
- when aggregation happens,
- when predictions are produced.

Understanding this flow is essential for

- debugging,
- modifying architectures,
- implementing new research models,
- interpreting intermediate representations.

Almost every new Graph Neural Network proposed in the literature modifies the **forward computation** rather than inventing entirely new layers.

Consequently, mastering the forward propagation process is one of the most valuable skills for a materials informatics researcher.

---

## MEGNet Perspective

The MEGNet architecture is fundamentally a sequence of differentiable tensor transformations operating on a crystal graph.

The forward pass begins with raw atomic identities, bond distances, and global descriptors.

These quantities are transformed into learned latent representations through embedding and projection networks.

Multiple Graph Network blocks then iteratively refine these representations by exchanging information across the graph.

Finally, the refined graph is compressed into a single crystal-level representation from which the target material property is predicted.

Every operation performed during this forward propagation is automatically recorded by PyTorch's dynamic computational graph, enabling efficient gradient computation during backpropagation.

With this conceptual understanding established, we are now ready to implement the complete `forward()` function of MEGNet line by line, explaining every tensor transformation in detail.

# 15.6.23 The Complete Forward Pass Flow Diagram

Now that we understand the conceptual meaning of forward propagation, we are ready to examine the **entire forward pass** of MEGNet from beginning to end.

Before writing a single line of code, we should first understand **how every tensor moves through the network**.

This is one of the most important habits in deep learning research.

Experienced researchers rarely begin by writing code.

Instead, they first answer the following questions:

- What are the inputs?
- What is the shape of every tensor?
- Which operation changes each tensor?
- Which tensors remain unchanged?
- Which operations create new tensors?
- Which operations are differentiable?
- Which tensors participate in gradient computation?

Once these questions are answered, implementing the network becomes almost mechanical.

---

# 15.6.24 The Entire Computational Pipeline

The complete forward propagation of MEGNet can be visualized as

```text
Crystal Structure

↓

Graph Construction

↓

Atomic Numbers
Bond Distances
Global Features

↓

Node Embedding
Gaussian Expansion
Global Projection

↓

Initial Graph

↓

Graph Network Block 1

↓

Graph Network Block 2

↓

Graph Network Block 3

↓

Updated Graph

↓

Readout Layer

↓

Graph Embedding

↓

Prediction Head

↓

Predicted Material Property
```

Notice something important.

The crystal itself never changes.

Only its mathematical representation evolves.

---

# 15.6.25 Tracking Tensor Shapes

Suppose our crystal contains

- 120 atoms
- 720 directed edges

Assume

- node dimension = 64
- edge dimension = 64
- global dimension = 32

The tensors entering the network are

| Tensor | Shape | Description |
|---------|-------|-------------|
| Atomic Numbers | `[120]` | Integer atomic identities |
| Edge Distances | `[720]` | Raw bond lengths |
| Global Features | `[2]` | Temperature and pressure |

These are **raw physical descriptors**.

They have not yet entered the neural network.

---

# 15.6.26 After the Embedding Stage

The node embedding converts

```text
[120]
```

into

```text
[120,64]
```

Each atom now owns

64 learned features.

Likewise,

Gaussian expansion converts

```text
[720]
```

into

```text
[720,26]
```

The edge projection network then transforms

```text
[720,26]

↓

[720,64]
```

The global projection converts

```text
[2]

↓

[32]
```

At this point,

the graph contains

| Tensor | Shape |
|---------|-------|
| Node Features | `[120,64]` |
| Edge Features | `[720,64]` |
| Global State | `[32]` |

The graph is now fully embedded.

---

# 15.6.27 Entering the First Graph Network Block

The first Graph Network block receives

```text
Nodes

↓

[120,64]

Edges

↓

[720,64]

Global

↓

[32]
```

Inside the block,

three update operations occur sequentially.

```text
Edge Update

↓

Node Update

↓

Global Update
```

Importantly,

none of these operations change the tensor dimensions.

Instead,

they change only the numerical values stored inside each tensor.

---

# 15.6.28 After the First Graph Network Block

The output becomes

```text
Updated Nodes

↓

[120,64]

Updated Edges

↓

[720,64]

Updated Global State

↓

[32]
```

Although the dimensions remain unchanged,

every feature vector has been modified.

Each atom now contains information gathered from its immediate neighbors.

Each bond contains richer interaction information.

The global state summarizes the updated crystal.

---

# 15.6.29 After the Second Graph Network Block

The second block receives exactly the tensors produced by the first block.

```text
Graph Block 1

↓

Graph Block 2
```

Again,

the dimensions remain

```text
Nodes

↓

[120,64]

Edges

↓

[720,64]

Global

↓

[32]
```

However,

the information stored inside these tensors becomes significantly richer.

Atoms now contain information propagated across approximately two graph hops.

This means that an atom can begin to learn about atoms that are **not directly bonded**, but are connected through neighboring atoms.

---

# 15.6.30 After the Third Graph Network Block

Repeating the process once more produces

```text
Graph Block 3
```

The output tensors remain

```text
Nodes

↓

[120,64]

Edges

↓

[720,64]

Global

↓

[32]
```

By now,

the node embeddings no longer represent isolated atoms.

Instead,

each node contains information aggregated from a substantial portion of the crystal.

This phenomenon is often described by saying that the **receptive field** of each node has expanded.

---

# 15.6.31 Evolution of Node Representations

The transformation of a single node throughout the network can be summarized as

```text
Atomic Number

↓

Embedding

↓

Chemical Representation

↓

Neighbor-Aware Representation

↓

Crystal-Aware Representation
```

Initially,

the node knows only

"What atom am I?"

After message passing,

it gradually learns

- who its neighbors are,
- what chemical environments surround it,
- how those environments influence the target property.

Thus,

the node embedding evolves from a simple atomic identity into a highly informative latent representation.

---

# 15.6.32 Evolution of Edge Representations

Edges undergo a similar transformation.

Initially,

an edge contains only

```text
Bond Distance
```

After projection,

it becomes

```text
Distance Embedding
```

After multiple Graph Network blocks,

the edge embedding also contains

- information from neighboring bonds,
- information from adjacent atoms,
- information propagated through the global state.

Thus,

the edge representation gradually evolves into a learned description of the local chemical interaction.

---

# 15.6.33 Evolution of the Global State

The global state follows a different evolution.

Initially,

it may contain only

```text
Temperature

Pressure
```

or even

zeros.

After every Graph Network block,

information from

- every node,
- every edge,

is aggregated into the global representation.

Consequently,

the global state gradually becomes a summary of the **entire crystal**.

By the final Graph Network block,

it often contains surprisingly rich information about the overall material.

---

# 15.6.34 The Final Graph Before Readout

Immediately before the readout operation,

the graph consists of

| Component | Meaning |
|-----------|---------|
| Node Features | Final atomic representations |
| Edge Features | Final interaction representations |
| Global State | Final crystal representation |

Every component has been refined through several rounds of message passing.

The network has extracted as much structural information as possible without yet producing a prediction.

The next task is to combine these distributed representations into a **single fixed-size graph embedding** suitable for regression or classification.

---

## **MEGNet Perspective**

The forward propagation of MEGNet is best understood as a sequence of representation transformations rather than a sequence of mathematical layers.

Each stage preserves the graph structure while progressively enriching the information stored within nodes, edges, and the global state.

The tensor shapes remain nearly constant throughout the Graph Network blocks, but their semantic meaning changes dramatically—from raw physical descriptors to high-level latent representations encoding local chemistry, long-range interactions, and global crystal characteristics.

Only after this iterative refinement is complete does the network perform the readout operation, which converts the distributed graph information into a single vector that can be used to predict material properties.

In the next section, we will begin implementing the **actual `forward()` method in PyTorch**, explaining every line of code, every tensor operation, every shape transformation, and every differentiable computation in detail.

# 15.6.35 Implementing the `forward()` Method Line by Line

After understanding the complete computational pipeline, we are finally ready to implement the actual `forward()` method.

Unlike many tutorials that simply present the entire function at once, we will build it **one statement at a time**, carefully examining

- what the statement does,
- why it is necessary,
- how tensor shapes change,
- whether new memory is allocated,
- whether gradients are tracked,
- how the operation contributes to the computational graph.

By the end of this section, every line inside the `forward()` function will be completely understood.

---

# 15.6.36 The Function Declaration

The forward function begins with

```python
def forward(self, graph):
```

This tells PyTorch

> "Whenever the model receives an input graph, execute the following computations."

Unlike ordinary Python functions,

this function is automatically called whenever we execute

```python
prediction = model(graph)
```

Internally,

PyTorch converts this statement into

```python
prediction = model.forward(graph)
```

Although we almost never call `forward()` directly,

it is always this function that performs the computation.

---

# 15.6.37 Understanding the Input Graph

Recall that the graph object was constructed during preprocessing.

Suppose we created

```python
graph = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_features

)

graph.u = global_state
```

The graph now contains

```text
graph

├── x

├── edge_index

├── edge_attr

└── u
```

Each field stores a different component of the crystal graph.

---

## Node Features

```python
graph.x
```

contains one feature vector for every atom.

For example,

```text
120 atoms

↓

graph.x

↓

[120]
```

before embedding,

or

```text
[120,64]
```

after embedding.

---

## Edge Connectivity

```python
graph.edge_index
```

stores

which atoms are connected.

Its shape is

```text
[2,N_edges]
```

The first row contains

the source nodes.

The second row contains

the destination nodes.

Suppose

```python
edge_index =

[[0,1,1,2],

 [1,0,2,1]]
```

This represents

```text
0 → 1

1 → 0

1 → 2

2 → 1
```

---

## Edge Features

```python
graph.edge_attr
```

stores

one feature vector

for every directed edge.

After Gaussian expansion and projection,

its shape may be

```text
[720,64]
```

Every bond therefore possesses

its own latent representation.

---

## Global State

```python
graph.u
```

contains

the graph-level features.

Suppose

```text
Temperature

Pressure
```

After projection,

its shape becomes

```text
[32]
```

Unlike node and edge tensors,

there is usually only

one global vector

per graph.

---

# 15.6.38 Extracting the Graph Components

The first lines inside `forward()` simply unpack the graph.

```python
x = graph.x

edge_index = graph.edge_index

edge_attr = graph.edge_attr

u = graph.u
```

Nothing is computed here.

No neural network layers are applied.

No tensor values change.

These statements merely create references to the tensors stored inside the graph object.

---

# 15.6.39 Why Assign Local Variables?

One might ask,

> Why not simply write

```python
graph.x
```

everywhere?

For example,

instead of

```python
x = graph.x
```

we could repeatedly write

```python
graph.x = self.node_embedding(graph.x)
```

Although this works,

it makes the code

- harder to read,
- harder to debug,
- more verbose.

Using local variables

```python
x

edge_attr

u
```

produces cleaner implementations.

Nearly every PyTorch project follows this convention.

---

# 15.6.40 Tensor Shapes Immediately After Extraction

Suppose

our graph contains

120 atoms

and

720 directed edges.

Immediately after extraction,

the tensors have shapes

```text
x

↓

[120]
```

```text
edge_index

↓

[2,720]
```

```text
edge_attr

↓

[720,26]
```

```text
u

↓

[2]
```

Notice

that these are still

the **raw graph features**.

The neural network has not yet transformed them.

---

# 15.6.41 Does This Create New Memory?

Consider

```python
x = graph.x
```

A common misconception is that this copies the tensor.

It does **not**.

Instead,

both variables reference the same underlying tensor.

Conceptually,

```text
graph.x

──────┐

      ▼

Tensor

      ▲

──────┘

x
```

No additional memory is allocated.

This operation is therefore extremely inexpensive.

---

# 15.6.42 Are Gradients Tracked?

Suppose

```python
graph.x
```

already requires gradients.

Then

```python
x
```

also requires gradients,

because both variables reference

the same tensor.

No new computational graph node is created.

The assignment itself is not a differentiable operation.

Only later neural network layers,

such as

```python
self.node_embedding(x)
```

become part of the computational graph.

---

# 15.6.43 Why We Extract Everything Before Computation

Notice that we extract

all graph components

before performing any neural computations.

This produces a logical separation between

**data access**

and

**data transformation**.

Conceptually,

the forward pass now consists of two phases.

```text
Phase 1

↓

Extract Tensors
```

followed by

```text
Phase 2

↓

Transform Tensors
```

Keeping these phases separate makes the implementation much easier to understand and maintain.

---

# 15.6.44 Summary of the Extraction Stage

At the end of the extraction stage,

nothing about the graph has changed.

We have simply prepared the tensors for subsequent computation.

The state of the network can now be summarized as

| Tensor | Shape | Description |
|---------|--------|-------------|
| `x` | `[120]` | Raw atomic numbers |
| `edge_index` | `[2,720]` | Graph connectivity |
| `edge_attr` | `[720,26]` | Gaussian-expanded bond features |
| `u` | `[2]` | Raw global descriptors |

The next operation will perform the **first learnable transformation** inside the network:

the **node embedding layer**, where discrete atomic identities are converted into continuous latent feature vectors suitable for message passing.

# 15.6.45 Applying the Node Embedding Layer

The first learnable operation inside the forward pass is the **node embedding layer**.

At this point,

the tensor

```python
x
```

contains only

atomic identities.

For example,

suppose our crystal contains

```text
Mg

O

O

Mg
```

The corresponding tensor is

```python
x = torch.tensor([

    12,

    8,

    8,

    12

])
```

Each number is an **atomic number**.

These numbers uniquely identify the elements,

but they are **not** suitable inputs for a neural network.

Therefore,

the first task of MEGNet is to transform these discrete identifiers into continuous feature vectors.

---

# 15.6.46 The Node Embedding Operation

The transformation is performed using

```python
x = self.node_embedding(x)
```

Although this statement consists of only one line,

a surprisingly large amount of computation occurs internally.

Conceptually,

the operation can be represented as

```text
Atomic Numbers

↓

Embedding Matrix Lookup

↓

Dense Feature Vectors
```

Each atomic number is used as an index into the embedding matrix.

The corresponding row is then returned.

---

# 15.6.47 Revisiting the Embedding Matrix

Recall that the embedding layer was defined during initialization.

```python
self.node_embedding = nn.Embedding(

    num_embeddings=95,

    embedding_dim=64

)
```

Internally,

PyTorch creates a learnable matrix

$$
\mathbf{E}
\in
\mathbb{R}^{95\times64}.
$$

Conceptually,

the matrix looks like

```text
Element

Embedding Vector

H

[...]

He

[...]

Li

[...]

...

Fe

[...]

Cu

[...]

Zn

[...]
```

Every row corresponds to one chemical element.

Every column represents one learned latent feature.

Initially,

every value inside this matrix is randomly initialized.

During training,

gradient descent continuously updates these values.

---

# 15.6.48 What Happens Internally?

Suppose

the first atom is magnesium.

Its atomic number is

```text
12
```

PyTorch performs

```text
Embedding Matrix

↓

Row 12

↓

Returned Vector
```

Suppose

the embedding dimension is

64.

The returned vector becomes

```text
[

0.42,

-0.18,

0.77,

...

-0.31

]
```

The same process occurs independently

for every atom in the graph.

No interaction between atoms occurs yet.

Each lookup is completely independent.

---

# 15.6.49 Tensor Shape Before and After Embedding

Suppose

our graph contains

120 atoms.

Before embedding,

the tensor has shape

```text
[120]
```

Each element is

a single integer.

```text
Atom 1

↓

12

Atom 2

↓

8

Atom 3

↓

8

...
```

After embedding,

the tensor becomes

```text
[120,64]
```

Each atom is now represented by

64 floating-point numbers.

Graphically,

```text
Before

↓

[120]

↓

Embedding

↓

After

↓

[120,64]
```

Notice that

the number of atoms remains unchanged.

Only the feature dimension increases.

---

# 15.6.50 Memory Layout After Embedding

After embedding,

the tensor can be visualized as a matrix.

```text
Atom

Feature Vector

Mg

[64 values]

O

[64 values]

O

[64 values]

Mg

[64 values]

...
```

Every row corresponds to one atom.

Every column corresponds to one learned feature.

Unlike handcrafted descriptors,

these features have **no predefined physical meaning**.

Their meaning emerges automatically during training.

---

# 15.6.51 Is the Embedding Layer a Neural Network?

Strictly speaking,

an embedding layer is **not** a traditional neural network layer.

There is

- no matrix multiplication,
- no activation function,
- no nonlinear transformation.

Instead,

it performs a **lookup operation**.

Mathematically,

if an atom has atomic number

$Z$,

its embedding is

$$
\mathbf{x}_Z
=
\mathbf{E}[Z].
$$

This simply selects

the

$Z$

th row of the embedding matrix.

Although simple,

this lookup operation is fully differentiable with respect to the embedding matrix.

---

# 15.6.52 Why Is the Embedding Matrix Trainable?

Suppose

the network predicts

formation energy.

Initially,

the embedding for oxygen might be

```text
[

0.11,

-0.42,

...

0.67

]
```

After computing the loss,

backpropagation determines

how each value contributed to the prediction error.

Gradient descent then updates

every component of the oxygen embedding.

After many optimization steps,

the embedding gradually evolves into a chemically meaningful representation.

This learning process occurs independently

for every chemical element.

---

# 15.6.53 Gradient Flow Through the Embedding Layer

A common question is

> If embedding performs only indexing,

how can gradients be computed?

The answer lies in the embedding matrix.

The atomic numbers themselves

are integers.

They never change.

Only the selected rows inside the embedding matrix are updated.

Suppose

our graph contains

```text
Mg

O

O

Mg
```

During backpropagation,

only

the rows corresponding to

magnesium

and

oxygen

receive gradients.

Rows corresponding to

iron,

gold,

or uranium

remain unchanged for this training example.

This makes embedding layers computationally efficient,

especially when the number of possible elements is large.

---

# 15.6.54 Inspecting the Embedded Tensor

After the embedding operation,

we can verify the tensor dimensions.

```python
print(x.shape)
```

Output

```text
torch.Size([120,64])
```

Each atom now possesses

a 64-dimensional latent representation.

This tensor becomes the node feature matrix used throughout the remainder of the network.

---

# 15.6.55 Why Embedding Comes Before Message Passing

One might ask,

> Why not perform message passing directly on atomic numbers?

Suppose

an atom sends

```text
14
```

to its neighbors.

What does the number

14

mean mathematically?

Nothing.

It is simply an identifier.

Message passing relies on

vector operations,

including

- addition,
- concatenation,
- matrix multiplication,
- nonlinear transformations.

These operations require continuous feature vectors,

not discrete identifiers.

Therefore,

embedding is an essential preprocessing step.

Without embedding,

Graph Network blocks cannot perform meaningful computations.

---

# 15.6.56 State of the Network After Node Embedding

Immediately after applying

```python
x = self.node_embedding(x)
```

the network has transformed

raw atomic identities

into

continuous latent representations.

The current state of the graph is

| Component | Shape | Status |
|-----------|--------|--------|
| Node Features | `[120,64]` | Embedded |
| Edge Features | `[720,26]` | Not yet projected |
| Global Features | `[2]` | Not yet projected |

Only the node features have entered the learned feature space.

The edge and global tensors still remain in their original representations.

These will be transformed next.

---

## **MEGNet Perspective**

The node embedding layer is the first trainable component encountered during the forward pass.

Although mathematically simple, it performs a critical role by converting discrete atomic identities into continuous latent vectors that can participate in differentiable message passing.

Each embedding vector is learned directly from data through gradient descent, allowing the network to discover chemically meaningful representations without relying on handcrafted atomic descriptors.

With the node features now embedded, the next stage of the forward pass projects the Gaussian-expanded edge features into the same latent feature space, enabling nodes and edges to interact effectively within the Graph Network blocks.

# 15.6.57 Projecting the Edge Features

After embedding the node features, the next step in the forward pass is to transform the **edge features**.

Recall that every edge currently contains the output of the Gaussian basis expansion.

Although these features already provide a rich description of bond distances, they are **not yet in the latent feature space used by the Graph Network blocks**.

Therefore, before message passing can begin, every edge feature vector must be projected into the same hidden representation used by the node features.

The operation is

```python
edge_attr = self.edge_projection(edge_attr)
```

Like the node embedding layer,

this statement transforms the representation of the graph.

Unlike the embedding layer,

however,

this transformation is performed using a neural network.

---

# 15.6.58 What Does `edge_attr` Contain?

Suppose

our crystal graph contains

720 directed edges.

After Gaussian basis expansion,

the edge tensor may have the shape

```text
[720,26]
```

Each row corresponds to

one bond.

For example,

```text
Edge 1

↓

[

0.02,

0.15,

0.71,

0.93,

...

0.00

]
```

```text
Edge 2

↓

[

0.00,

0.03,

0.42,

0.88,

...

0.01

]
```

Each value represents the response of one Gaussian basis function.

These vectors encode geometric information,

but they are still relatively low-dimensional.

---

# 15.6.59 Why Project the Edge Features?

A natural question is

> Why not feed the Gaussian vectors directly into the Graph Network?

There are several reasons.

First,

the node features have dimension

```text
64
```

while the Gaussian expansion may produce only

```text
26
```

The update networks inside the Graph Network combine

- node features,
- edge features,
- global features.

Working with different feature dimensions would require additional transformations inside every update function.

Instead,

MEGNet first projects all feature types into compatible latent spaces.

This simplifies the architecture considerably.

---

# 15.6.60 The Edge Projection Network

Recall that the projection network was defined as

```python
self.edge_projection = nn.Sequential(

    nn.Linear(

        26,

        hidden_dim

    ),

    nn.ReLU(),

    nn.Linear(

        hidden_dim,

        edge_dim

    )

)
```

Suppose

```text
hidden_dim = 128

edge_dim = 64
```

The transformation becomes

```text
26

↓

128

↓

64
```

Unlike the embedding layer,

this network performs

learnable nonlinear transformations.

---

# 15.6.61 The First Linear Layer

The first operation is

```python
nn.Linear(

    26,

    128

)
```

Suppose

one bond has the feature vector

```text
[

0.02,

0.15,

0.71,

...

0.01

]
```

The linear layer computes

$$
\mathbf{h}
=
\mathbf{W}_1
\mathbf{e}
+
\mathbf{b}_1,
$$

where

- $\mathbf{e}$ is the 26-dimensional Gaussian feature vector,
- $\mathbf{W}_1$ is a learnable weight matrix,
- $\mathbf{b}_1$ is a learnable bias vector.

The output becomes

a

128-dimensional hidden representation.

---

# 15.6.62 Why Increase the Dimension?

Notice that the first layer expands

```text
26

↓

128
```

This is a common strategy in deep learning.

A larger hidden space allows the network to learn

more complex nonlinear combinations of the input features.

The Gaussian basis functions themselves are relatively simple.

The projection network combines them into much richer representations.

Conceptually,

the network is learning

which combinations of Gaussian responses are most informative for predicting material properties.

---

# 15.6.63 The ReLU Activation

After the first linear transformation,

MEGNet applies

```python
nn.ReLU()
```

The Rectified Linear Unit is defined as

$$
\mathrm{ReLU}(x)
=
\max(0,x).
$$

Graphically,

```text
Output

^

|

|       /

|      /

|     /

|____/____________>

     0
```

Negative values become zero.

Positive values remain unchanged.

This introduces nonlinearity,

allowing the projection network to learn far more expressive transformations than a single linear layer.

---

# 15.6.64 The Second Linear Layer

Finally,

the hidden representation is projected into the edge latent space.

```python
nn.Linear(

    128,

    64

)
```

Mathematically,

$$
\mathbf{e}'
=
\mathbf{W}_2
\mathbf{h}
+
\mathbf{b}_2.
$$

The output is

a

64-dimensional edge embedding.

Thus,

every bond is represented by

a learned latent vector

compatible with the node feature dimension.

---

# 15.6.65 Tensor Shape Transformation

Suppose

the graph contains

720 directed edges.

The tensor evolves as follows.

```text
Before Projection

↓

[720,26]
```

```text
After First Linear Layer

↓

[720,128]
```

```text
After ReLU

↓

[720,128]
```

```text
After Second Linear Layer

↓

[720,64]
```

Only the feature dimension changes.

The number of edges remains constant.

---

# 15.6.66 Vectorized Computation

Although we often describe the projection

one edge at a time,

PyTorch performs the computation for **all edges simultaneously**.

Instead of executing

```python
for edge in edges:

    project(edge)
```

the entire tensor

```text
[720,26]
```

is processed in one highly optimized matrix operation.

This vectorized implementation is substantially faster,

especially on GPUs,

where thousands of edges can be processed in parallel.

---

# 15.6.67 Gradient Flow Through the Projection Network

Unlike the embedding layer,

the edge projection network contains

ordinary neural network parameters.

During backpropagation,

gradients are computed for

- the first weight matrix,
- the first bias vector,
- the second weight matrix,
- the second bias vector.

Every edge contributes to the optimization of these shared parameters.

Consequently,

the projection network learns a general transformation that works across all bonds in the training dataset.

---

# 15.6.68 Why Use a Neural Network Instead of Another Embedding?

Recall that node embeddings operate on

discrete identifiers.

Edges,

however,

contain **continuous values**.

Continuous variables cannot be indexed into a lookup table.

Instead,

they must be transformed through differentiable functions.

A multilayer perceptron naturally fulfills this role.

This distinction is fundamental.

- Nodes use an **embedding lookup** because atomic identities are discrete.
- Edges use a **projection network** because bond descriptors are continuous.

---

# 15.6.69 State of the Network After Edge Projection

Immediately after

```python
edge_attr = self.edge_projection(edge_attr)
```

the graph contains

| Component | Shape | Status |
|-----------|--------|--------|
| Node Features | `[120,64]` | Embedded |
| Edge Features | `[720,64]` | Projected |
| Global Features | `[2]` | Raw |

Both nodes and edges now occupy compatible latent feature spaces.

The only remaining raw component is the global state.

This will be transformed next.

---

## **MEGNet Perspective**

The edge projection network converts the Gaussian-expanded bond descriptors into high-dimensional latent representations suitable for message passing.

Unlike node embeddings, which rely on discrete lookup operations, the edge projection network is a fully trainable multilayer perceptron that learns nonlinear combinations of geometric information.

By projecting both node and edge features into compatible latent spaces, MEGNet enables efficient information exchange within the Graph Network blocks, where atomic identities and interatomic interactions are integrated to produce increasingly informative representations of the crystal structure.

The next stage of the forward pass applies an analogous projection to the **global state**, completing the initialization of every feature type before message passing begins.

# 15.6.70 Projecting the Global State

At this point in the forward pass,

both the node features and the edge features have been transformed into learned latent representations.

The only remaining component that still exists in its original form is the **global state**.

Recall that the global state represents information describing the **entire graph** rather than individual atoms or individual bonds.

Before message passing begins,

the global state must also be transformed into the same latent feature space as the rest of the network.

This transformation is performed by

```python
u = self.global_projection(u)
```

Although this statement appears simple,

it plays a crucial role in allowing the Graph Network to incorporate crystal-level information into every message-passing step.

---

# 15.6.71 Revisiting the Meaning of the Global State

Unlike node features,

which belong to individual atoms,

and edge features,

which belong to individual bonds,

the global state belongs to the **entire crystal**.

Graphically,

```text
Crystal

├── Node 1

├── Node 2

├── Node 3

├── ...

└── Global State
```

There is only **one global vector** for each graph.

Every Graph Network block updates this vector as information flows through the network.

---

# 15.6.72 Examples of Global Features

The original Graph Networks framework allows the global state to contain any graph-level descriptors.

Examples include

```text
Temperature

Pressure

Density

External Electric Field

External Magnetic Field

Applied Stress

Applied Strain

Chemical Potential

Experimental Conditions
```

For molecular systems,

the global state might instead contain

```text
Solvent Information

pH

Reaction Temperature
```

For crystalline materials,

it is common to initialize the global state using

external thermodynamic conditions.

However,

many datasets do not provide any graph-level descriptors.

---

# 15.6.73 What If No Global Features Exist?

One of the elegant aspects of MEGNet is that

the global state does **not** need to contain meaningful information initially.

Suppose

the dataset contains only crystal structures.

No temperature.

No pressure.

No external conditions.

We may simply initialize

```python
u = torch.zeros(

    global_dim

)
```

Initially,

the global vector contains no information.

However,

during message passing,

the Graph Network blocks continuously update

the global representation.

Eventually,

this vector becomes a learned summary of the entire crystal.

Thus,

the global state evolves from

```text
Nothing

↓

Everything
```

through repeated message passing.

---

# 15.6.74 Why Project the Global State?

Suppose

our raw global vector is

```python
u = torch.tensor([

    300.0,

    1.0

])
```

representing

```text
Temperature = 300 K

Pressure = 1 atm
```

This tensor has shape

```text
[2]
```

The Graph Network,

however,

expects a much richer representation.

Instead of using only two numbers,

we transform them into

a high-dimensional latent vector.

Conceptually,

the transformation is

```text
Raw Global Features

↓

Projection Network

↓

Latent Global Representation
```

---

# 15.6.75 The Global Projection Network

Recall its definition.

```python
self.global_projection = nn.Sequential(

    nn.Linear(

        2,

        hidden_dim

    ),

    nn.ReLU(),

    nn.Linear(

        hidden_dim,

        global_dim

    )

)
```

Suppose

```text
hidden_dim = 128

global_dim = 32
```

The transformation becomes

```text
2

↓

128

↓

32
```

Notice that the architecture is remarkably similar to

the edge projection network.

The only difference is

the input dimension.

---

# 15.6.76 Mathematical Formulation

Let the raw global feature vector be

$$
\mathbf{u}
\in
\mathbb{R}^{2}.
$$

The first linear layer computes

$$
\mathbf{h}
=
\mathbf{W}_1
\mathbf{u}
+
\mathbf{b}_1.
$$

After applying the ReLU activation,

the second linear layer computes

$$
\mathbf{u}'
=
\mathbf{W}_2
\mathbf{h}
+
\mathbf{b}_2.
$$

The resulting vector

$$
\mathbf{u}'
$$

is the initial global embedding.

This vector becomes the graph-level feature used throughout message passing.

---

# 15.6.77 Tensor Shape Evolution

Suppose

our graph contains

only two global descriptors.

The tensor evolves as

```text
Before Projection

↓

[2]
```

```text
After First Linear Layer

↓

[128]
```

```text
After ReLU

↓

[128]
```

```text
After Second Linear Layer

↓

[32]
```

Unlike node and edge tensors,

there is only **one** such vector per graph.

---

# 15.6.78 Why Learn a Higher-Dimensional Global Representation?

At first glance,

it may seem unnecessary to transform

two numbers

into

thirty-two numbers.

However,

this higher-dimensional representation allows the network to learn

complex interactions between

- thermodynamic conditions,
- crystal chemistry,
- local atomic environments.

For example,

the influence of temperature on a material property is rarely linear.

By embedding the global state into a latent space,

the network can learn highly nonlinear relationships.

---

# 15.6.79 Interaction with Message Passing

An important feature of MEGNet is that

the global state is **not isolated**.

During every Graph Network block,

the global state participates in

- edge updates,
- node updates,
- global updates.

Consequently,

the global vector influences

every atom

and

every bond.

Likewise,

information from

every atom

and

every bond

flows back into the global state.

This bidirectional exchange is one of the defining characteristics of the Graph Networks framework.

---

# 15.6.80 Gradient Flow Through the Global Projection

The global projection network is fully differentiable.

During backpropagation,

gradients are computed for

- the first linear layer,
- the ReLU activation,
- the second linear layer.

These gradients update the projection network so that

the latent global representation becomes increasingly informative for the prediction task.

If the initial global descriptors contain useful physical information,

the network learns how to exploit them.

If the initial global vector is simply zeros,

the projection still learns an appropriate initialization that can be refined through message passing.

---

# 15.6.81 Comparing the Three Feature Transformations

At this stage,

every feature type has undergone its initial transformation.

| Feature Type | Initial Representation | Transformation | Latent Dimension |
|--------------|-----------------------|----------------|------------------|
| Nodes | Atomic Numbers | Embedding Layer | 64 |
| Edges | Gaussian Basis Expansion | Projection Network | 64 |
| Global State | Physical Descriptors | Projection Network | 32 |

Although the transformations differ,

their objective is identical:

convert raw physical information into learned latent representations suitable for message passing.

---

# 15.6.82 State of the Network Before Message Passing

Immediately before entering the first Graph Network block,

the graph has the following form.

| Component | Shape | Description |
|-----------|--------|-------------|
| Node Features | `[120,64]` | Embedded atomic representations |
| Edge Features | `[720,64]` | Projected bond representations |
| Global State | `[32]` | Projected graph representation |

Every component now resides in a learned feature space.

The graph is fully initialized.

From this point onward,

the network performs **message passing** rather than feature initialization.

No additional embedding or projection operations occur.

Instead,

the existing representations are repeatedly refined through interactions among nodes, edges, and the global state.

---

## **MEGNet Perspective**

The projection of the global state completes the initialization stage of the MEGNet forward pass.

At this moment,

every component of the crystal graph—nodes, edges, and the graph-level descriptor—has been transformed into a continuous latent representation.

Unlike traditional graph neural networks that operate only on nodes and edges,

MEGNet maintains a dedicated global representation that both influences and is influenced by every message-passing step.

This design enables the model to naturally incorporate external physical conditions and to learn a holistic representation of the crystal during training.

With all feature types initialized, the network is now ready to enter the first **Graph Network block**, where the actual iterative message-passing process begins.

# 15.6.83 Entering the Graph Network Blocks

At this stage,

the initialization phase of the forward pass is complete.

Every component of the graph has been transformed into a learned latent representation.

The graph now consists of

- embedded node features,
- projected edge features,
- projected global features.

These tensors are now ready for the **core computation** of the MEGNet architecture.

This computation is performed by the **Graph Network blocks**.

Unlike the embedding and projection layers,

which transform raw input data only once,

the Graph Network blocks repeatedly refine the graph representation through message passing.

This iterative refinement is where most of the learning occurs.

---

# 15.6.84 The Main Computation Loop

The Graph Network blocks were stored inside

```python
self.blocks
```

using

```python
nn.ModuleList
```

Therefore,

instead of calling every block individually,

we simply iterate through them.

```python
for block in self.blocks:

    x, edge_attr, u = block(

        x,

        edge_attr,

        edge_index,

        u

    )
```

Although this loop contains only a few lines of code,

it performs the majority of the computation inside MEGNet.

---

# 15.6.85 Understanding the Loop

Suppose

the network contains

three Graph Network blocks.

Conceptually,

the loop expands into

```python
x, edge_attr, u = block1(

    x,

    edge_attr,

    edge_index,

    u

)

x, edge_attr, u = block2(

    x,

    edge_attr,

    edge_index,

    u

)

x, edge_attr, u = block3(

    x,

    edge_attr,

    edge_index,

    u

)
```

The output of one block immediately becomes the input of the next block.

No intermediate processing is required.

---

# 15.6.86 Why Reassign the Variables?

Notice that

the same variables

```python
x

edge_attr

u
```

appear on both sides of the assignment.

For example,

```python
x = updated_x
```

is written simply as

```python
x = ...
```

This does **not** mean that the old tensor is modified in place.

Instead,

each Graph Network block returns **new tensors**.

Conceptually,

the computation is

```text
Old Node Features

↓

Graph Network Block

↓

New Node Features
```

The variable

```python
x
```

is then updated to reference the new tensor.

The previous tensor remains part of the computational graph until backpropagation is complete.

---

# 15.6.87 Tensor Shapes Throughout the Loop

Suppose

the graph initially contains

```text
Nodes

↓

[120,64]

Edges

↓

[720,64]

Global

↓

[32]
```

After the first Graph Network block,

the shapes remain

```text
Nodes

↓

[120,64]

Edges

↓

[720,64]

Global

↓

[32]
```

Exactly the same happens after

the second

and

third

Graph Network blocks.

The dimensions never change.

Only the numerical values stored inside the tensors evolve.

This is an intentional design choice.

Keeping the feature dimensions constant greatly simplifies stacking multiple Graph Network blocks.

---

# 15.6.88 What Happens Inside Each Iteration?

Although the loop appears simple,

each iteration performs three separate neural computations.

```text
Input Graph

↓

Edge Update

↓

Updated Edges

↓

Node Update

↓

Updated Nodes

↓

Global Update

↓

Updated Global State
```

Thus,

one iteration of the loop corresponds to one complete message-passing cycle.

When the loop executes three times,

the graph undergoes three complete rounds of information exchange.

---

# 15.6.89 Information Flow During Iteration

Consider one silicon atom inside the crystal.

Initially,

its embedding contains only information learned from

its atomic identity.

```text
Si
```

After the first Graph Network block,

the embedding also contains information from neighboring atoms.

```text
O

↓

Si

↓

O
```

After the second block,

information from atoms that are two graph hops away begins to arrive.

```text
Mg

↓

O

↓

Si

↓

O

↓

Mg
```

After the third block,

the receptive field expands even further.

The atom gradually develops an increasingly complete understanding of its local chemical environment.

---

# 15.6.90 Evolution of the Edge Features

The edge features also evolve during every iteration.

Initially,

an edge contains primarily geometric information.

```text
Bond Distance

↓

Gaussian Features

↓

Projected Features
```

After one Graph Network block,

the edge also incorporates

- information from its connected atoms,
- information from the global state.

After multiple iterations,

each edge becomes a learned description of the local chemical interaction rather than merely a bond distance.

---

# 15.6.91 Evolution of the Global State

The global state changes simultaneously.

Initially,

it may contain

```text
Temperature

Pressure
```

or simply

zeros.

After every Graph Network block,

information from

every node

and

every edge

is aggregated into the global representation.

Consequently,

the global vector becomes

an increasingly accurate summary

of the entire crystal.

Unlike node features,

which describe individual atoms,

the global state represents

the graph as a whole.

---

# 15.6.92 Why Multiple Graph Network Blocks?

One Graph Network block allows information to travel only a limited distance through the graph.

Suppose

Atom A

is connected to

Atom B,

which is connected to

Atom C.

```text
A

↓

B

↓

C
```

After one Graph Network block,

Atom A receives information from

Atom B,

but not directly from

Atom C.

After the second Graph Network block,

information originating from

Atom C

can now influence

Atom A.

Thus,

stacking multiple Graph Network blocks progressively increases the range over which information can propagate.

This is analogous to stacking convolutional layers in convolutional neural networks,

where deeper layers possess larger receptive fields.

---

# 15.6.93 Computational Cost of the Loop

Suppose

the graph contains

- 120 nodes,
- 720 edges,
- 3 Graph Network blocks.

Each Graph Network block performs

- one edge update,
- one node update,
- one global update.

The complete forward pass therefore performs

```text
3 Edge Updates

3 Node Updates

3 Global Updates
```

In general,

if

$L$

Graph Network blocks are used,

the total number of update operations is

```text
L Edge Updates

L Node Updates

L Global Updates
```

Consequently,

the computational cost increases approximately linearly with the number of Graph Network blocks.

Increasing the network depth usually improves representational power,

but also increases training time and memory consumption.

---

# 15.6.94 State of the Graph After the Loop

Once the loop terminates,

the graph has been fully refined.

The tensors now represent

| Component | Description |
|-----------|-------------|
| Node Features | Final atomic embeddings |
| Edge Features | Final interaction embeddings |
| Global State | Final crystal embedding |

No further message passing occurs.

The graph now contains the richest internal representation that the network can produce.

However,

the prediction cannot yet be made.

The network still possesses

one feature vector

for every atom

and

one feature vector

for every bond.

The next challenge is to convert these distributed representations into **a single fixed-length vector** representing the entire crystal.

This operation is performed by the **readout layer**.

---

## **MEGNet Perspective**

The Graph Network blocks form the computational core of MEGNet.

Each block performs one complete message-passing cycle, successively updating edges, nodes, and the global state while preserving the graph topology.

By stacking multiple blocks, information propagates across increasingly larger regions of the crystal, enabling each atom to encode not only its own chemical identity but also the structural and chemical context of its surrounding environment.

After the final Graph Network block, the graph contains highly expressive latent representations that integrate local interactions, long-range structural effects, and global crystal information.

The next stage of the forward pass applies the **readout operation**, which compresses these distributed representations into a single graph-level embedding suitable for predicting material properties.

# 15.6.95 The Readout Operation

After the final Graph Network block,

the network has completed all message passing.

Every atom,

every bond,

and the global state now contain highly refined latent representations.

However,

the network still faces one fundamental problem.

The graph contains **many feature vectors**,

while the prediction requires **only one**.

Suppose we wish to predict

- formation energy,
- band gap,
- bulk modulus,
- elastic modulus,
- dielectric constant.

Each of these quantities is a **graph-level property**.

There is one prediction per crystal,

not one prediction per atom.

Therefore,

before making a prediction,

the network must convert the distributed graph representation into a **single fixed-length vector**.

This transformation is called the **readout operation**.

---

# 15.6.96 Why Is a Readout Necessary?

Suppose our crystal contains

120 atoms.

After message passing,

the node feature matrix has shape

```text
[120,64]
```

This means

the network possesses

120 different feature vectors.

Graphically,

```text
Atom 1

↓

64 Features

Atom 2

↓

64 Features

...

Atom 120

↓

64 Features
```

But the desired output is

```text
Formation Energy

↓

One Number
```

The prediction head cannot process

120 independent vectors.

It requires

one graph representation.

Therefore,

all node information must first be aggregated.

---

# 15.6.97 What Makes a Good Readout?

A readout function must satisfy several important requirements.

First,

it must accept graphs containing **different numbers of atoms**.

For example,

```text
Graph A

↓

42 atoms
```

```text
Graph B

↓

315 atoms
```

Both graphs must ultimately produce

one vector of identical size.

Second,

the readout must be **permutation invariant**.

Changing the order in which atoms are stored should never change the predicted material property.

For example,

these two graphs represent exactly the same crystal.

```text
Mg

O

O

Mg
```

and

```text
O

Mg

Mg

O
```

Although the atom ordering differs,

the predicted formation energy must remain identical.

A valid readout function must satisfy this property.

---

# 15.6.98 Permutation Invariance

Mathematically,

suppose

the node embeddings are

$$
\{

\mathbf{x}_1,

\mathbf{x}_2,

...,

\mathbf{x}_N

\}.
$$

A permutation changes their order.

$$
\{

\mathbf{x}_3,

\mathbf{x}_1,

\mathbf{x}_N,

...,

\mathbf{x}_2

\}.
$$

Although the ordering changes,

the crystal itself does not.

Therefore,

the readout function

$R$

must satisfy

$$
R(\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N)

=

R(\mathbf{x}_{\pi(1)},\mathbf{x}_{\pi(2)},\ldots,\mathbf{x}_{\pi(N)}),
$$

where

$\pi$

represents any permutation.

This property is essential for every graph neural network.

---

# 15.6.99 Common Readout Functions

Several aggregation functions satisfy permutation invariance.

The most common are

```text
Sum Pooling

Mean Pooling

Max Pooling
```

Each produces

one graph embedding,

but they summarize information differently.

Modern graph neural networks often combine several pooling operations,

or use learned attention-based pooling,

but the basic principles remain the same.

---

# 15.6.100 Sum Pooling

The simplest readout is

the sum of all node embeddings.

Mathematically,

$$
\mathbf{g}

=

\sum_{i=1}^{N}

\mathbf{x}_i.
$$

Conceptually,

```text
Node 1

↓

Node 2

↓

Node 3

↓

...

↓

Node N

↓

Sum

↓

Graph Embedding
```

The output dimension remains

64,

regardless of

how many atoms are present.

---

# 15.6.101 Advantages of Sum Pooling

Sum pooling preserves information about

the size of the graph.

Suppose

two crystals have identical local environments,

but one contains twice as many atoms.

Their summed embeddings will generally differ.

Consequently,

sum pooling naturally captures

extensive properties,

such as

- total energy,
- total mass,
- total charge.

This makes it particularly useful for predicting quantities that scale with system size.

---

# 15.6.102 Mean Pooling

Another popular choice is

mean pooling.

Instead of summing the node embeddings,

the network computes their average.

Mathematically,

$$
\mathbf{g}

=

\frac{1}{N}

\sum_{i=1}^{N}

\mathbf{x}_i.
$$

Conceptually,

```text
Node Embeddings

↓

Average

↓

Graph Embedding
```

Again,

the output dimension remains fixed,

independent of the number of atoms.

---

# 15.6.103 Advantages of Mean Pooling

Mean pooling removes the direct influence of graph size.

Suppose

two crystals have identical local environments,

but different numbers of atoms.

Their average embeddings may become very similar.

This behavior is particularly useful for predicting

intensive properties,

such as

- band gap,
- density,
- elastic modulus per unit volume.

These quantities do not necessarily scale with the number of atoms.

---

# 15.6.104 Max Pooling

Max pooling operates differently.

Instead of adding or averaging,

it selects the largest value in each feature dimension.

Mathematically,

$$
g_j

=

\max_i

x_{ij},
$$

where

$x_{ij}$

is

the

$j$

th feature of

the

$i$

th node.

Conceptually,

```text
Feature 1

↓

Largest Value

Feature 2

↓

Largest Value

...

↓

Graph Embedding
```

This emphasizes the strongest activations in the graph.

---

# 15.6.105 Comparing the Three Pooling Methods

The behavior of the three pooling strategies differs significantly.

| Readout | Operation | Typical Applications |
|----------|-----------|----------------------|
| Sum | Adds all node embeddings | Extensive properties |
| Mean | Computes the average | Intensive properties |
| Max | Selects largest activation | Feature detection |

Each method is permutation invariant.

Each produces

a fixed-length graph embedding.

The choice depends on

the physical quantity being predicted.

---

# 15.6.106 Readout in the Original MEGNet

The original MEGNet architecture does not rely on

a single pooling operation alone.

Instead,

it aggregates information from

- node representations,
- edge representations,
- the global state.

These aggregated vectors are then concatenated

to produce the final graph embedding.

Conceptually,

```text
Node Readout

↓

Graph Vector

Edge Readout

↓

Graph Vector

Global State

↓

Graph Vector

↓

Concatenation

↓

Final Graph Embedding
```

This richer representation allows the prediction head to utilize

information from every component of the graph,

rather than relying solely on node embeddings.

---

# 15.6.107 Why Aggregate Edges as Well?

Many graph neural networks perform readout using only node features.

MEGNet extends this idea.

Consider

two crystals

with identical atomic compositions

but different bonding environments.

The node embeddings alone may not fully distinguish them.

The edge embeddings,

however,

contain explicit information about

interatomic interactions.

Aggregating both nodes and edges therefore provides

a more complete description of the crystal.

---

# 15.6.108 The Goal of the Readout Stage

After the readout operation,

the network no longer stores

hundreds of node vectors

or

hundreds of edge vectors.

Instead,

it possesses

one vector

representing

the entire crystal.

Graphically,

```text
Node Features

↓

Readout

↓

Node Summary

Edge Features

↓

Readout

↓

Edge Summary

Global State

↓

Already Global

↓

Concatenate

↓

Graph Embedding
```

This graph embedding becomes the direct input to the prediction head.

---

## **MEGNet Perspective**

The readout operation forms the bridge between graph representation learning and property prediction.

While the Graph Network blocks distribute information throughout the crystal,

the readout gathers this distributed knowledge into a single fixed-length vector that summarizes the entire material.

Unlike many graph neural networks that aggregate only node features,

MEGNet incorporates information from nodes, edges, and the global state, producing a richer and more physically meaningful graph representation.

In the next section, we will implement this complete readout procedure in PyTorch, including node pooling, edge pooling, feature concatenation, and the construction of the final graph embedding used for prediction.

# 15.6.109 Implementing the Readout Layer

We now arrive at one of the most important parts of the MEGNet forward pass.

After several rounds of message passing,

the graph contains

- refined node embeddings,
- refined edge embeddings,
- a refined global representation.

However,

these representations are still distributed throughout the graph.

The prediction head expects a **single feature vector** for each crystal.

Therefore,

the first task of the readout layer is to aggregate information from every component of the graph.

Unlike the Graph Network blocks,

which operate locally,

the readout layer performs a **global aggregation**.

---

# 15.6.110 Batching in PyTorch Geometric

Before implementing the readout,

we must understand an important concept.

During training,

we rarely process one graph at a time.

Instead,

multiple graphs are combined into a **mini-batch**.

Suppose our batch contains three crystals.

```text
Crystal A

42 atoms
```

```text
Crystal B

91 atoms
```

```text
Crystal C

135 atoms
```

PyTorch Geometric does **not** store these graphs separately.

Instead,

it merges them into one large disconnected graph.

Conceptually,

```text
Batch

↓

Crystal A

Crystal B

Crystal C
```

The node feature matrix becomes

```text
[268,64]
```

because

```text
42

+

91

+

135

=

268
```

There is still only **one** tensor.

The framework keeps track of graph membership using another tensor called

```python
batch
```

---

# 15.6.111 The Batch Vector

The batch vector assigns every node to its corresponding graph.

For example,

```python
batch =
[
0,0,0,...,

1,1,1,...,

2,2,2,...
]
```

If

```python
batch[17] = 0
```

then

node 17 belongs to

Crystal A.

If

```python
batch[180] = 2
```

then

node 180 belongs to

Crystal C.

Thus,

the batch vector tells PyTorch Geometric

which nodes should be pooled together.

---

# 15.6.112 Accessing the Batch Tensor

The batch information is stored inside

```python
graph.batch
```

Therefore,

the first step is

```python
batch = graph.batch
```

Its shape is

```text
[N_nodes]
```

Suppose

268 nodes exist.

Then

```text
batch

↓

[268]
```

Each element stores

the graph index

to which the corresponding node belongs.

---

# 15.6.113 Global Mean Pooling

The simplest readout operation is

global mean pooling.

PyTorch Geometric provides

a highly optimized implementation.

```python
from torch_geometric.nn import global_mean_pool
```

The readout becomes

```python
node_graph = global_mean_pool(

    x,

    batch

)
```

This operation computes

the average node embedding

for every graph independently.

---

# 15.6.114 Understanding the Pooling Operation

Suppose

Crystal A contains

four atoms.

Their embeddings are

```text
Atom 1

↓

64 values

Atom 2

↓

64 values

Atom 3

↓

64 values

Atom 4

↓

64 values
```

Global mean pooling computes

$$
\mathbf{g}_{node}

=

\frac{1}{4}

(\mathbf{x}_1

+

\mathbf{x}_2

+

\mathbf{x}_3

+

\mathbf{x}_4).
$$

The result is

one

64-dimensional vector

representing the entire crystal.

The same computation is performed independently

for every graph in the batch.

---

# 15.6.115 Tensor Shapes After Node Pooling

Suppose

our mini-batch contains

16 crystals.

Before pooling,

```text
x

↓

[268,64]
```

After pooling,

```text
node_graph

↓

[16,64]
```

Notice the dramatic reduction.

Hundreds of atomic feature vectors have been compressed into

one feature vector per graph.

---

# 15.6.116 Pooling the Edge Features

MEGNet also aggregates

edge representations.

However,

edge pooling requires a different batch assignment,

because

edges,

not nodes,

must be grouped by graph.

The graph membership of each edge is determined from its source node.

```python
edge_batch = batch[

    edge_index[0]

]
```

Let us understand why this works.

---

# 15.6.117 Understanding `edge_batch`

Recall that

```python
edge_index
```

has shape

```text
[2,N_edges]
```

The first row contains

source nodes.

For example,

```python
edge_index =

[[0,1,2,5],

 [1,2,5,7]]
```

Selecting

```python
edge_index[0]
```

returns

```python
[0,1,2,5]
```

These are

the source nodes.

Looking up

```python
batch[edge_index[0]]
```

returns

the graph index

for every edge.

Consequently,

every edge is assigned to the correct crystal.

---

# 15.6.118 Global Mean Pooling of Edge Features

The edge embeddings are then pooled

exactly like the node embeddings.

```python
edge_graph = global_mean_pool(

    edge_attr,

    edge_batch

)
```

Suppose

before pooling,

```text
edge_attr

↓

[1608,64]
```

After pooling,

```text
edge_graph

↓

[16,64]
```

Every graph now possesses

one summarized edge representation.

---

# 15.6.119 Preparing the Global State

Unlike nodes and edges,

the global state already exists

at the graph level.

Suppose

our batch contains

16 graphs.

The projected global tensor already has shape

```text
[16,32]
```

No pooling is required.

The global representation can therefore be used directly.

```python
global_graph = u
```

---

# 15.6.120 State of the Readout Components

After all aggregation operations,

we possess three graph-level tensors.

| Tensor | Shape | Meaning |
|---------|--------|----------|
| `node_graph` | `[16,64]` | Average node representation |
| `edge_graph` | `[16,64]` | Average edge representation |
| `global_graph` | `[16,32]` | Global state representation |

Every tensor now corresponds to

entire crystals,

not individual atoms or bonds.

The next step is to combine these three sources of information into one unified graph embedding.

---

## **MEGNet Perspective**

The readout stage converts distributed graph information into compact graph-level representations suitable for prediction.

Node embeddings are aggregated using graph-wise pooling,

edge embeddings are pooled separately using the graph assignment derived from the source nodes,

and the global state is already available as a graph-level feature.

At this stage,

the network has successfully transformed a variable-sized crystal graph into three fixed-size representations,

laying the foundation for constructing the final graph embedding that will be passed to the prediction head.

# 15.6.121 Constructing the Final Graph Embedding

At the end of the readout stage,

we possess three independent graph-level representations.

```text
Node Representation

↓

[Batch Size,64]
```

```text
Edge Representation

↓

[Batch Size,64]
```

```text
Global Representation

↓

[Batch Size,32]
```

Although each tensor contains useful information,

none of them alone completely describes the crystal.

Instead,

MEGNet combines these three complementary representations into a **single graph embedding**.

This embedding becomes the input to the prediction network.

---

# 15.6.122 Why Combine Three Representations?

Each representation captures a different aspect of the material.

The node representation summarizes

the atoms.

The edge representation summarizes

the chemical interactions.

The global representation summarizes

graph-level information accumulated during message passing.

Conceptually,

```text
Nodes

↓

"What atoms exist?"
```

```text
Edges

↓

"How do atoms interact?"
```

```text
Global State

↓

"What does the entire crystal look like?"
```

A high-quality prediction requires all three.

Ignoring any one of them would discard valuable information.

---

# 15.6.123 Feature Concatenation

The simplest way to combine multiple feature vectors is **concatenation**.

PyTorch provides the function

```python
torch.cat()
```

The implementation is

```python
graph_embedding = torch.cat(

    [

        node_graph,

        edge_graph,

        global_graph

    ],

    dim=1

)
```

The argument

```python
dim=1
```

means

concatenate along the feature dimension,

not the batch dimension.

---

# 15.6.124 Understanding Concatenation

Suppose

one crystal has

```text
Node Representation

↓

64 values
```

```text
Edge Representation

↓

64 values
```

```text
Global Representation

↓

32 values
```

Concatenation produces

```text
64

+

64

+

32

=

160
```

features.

Graphically,

```text
Node Features

[xxxxxxxxxxxxxxxx]

+

Edge Features

[yyyyyyyyyyyyyyyy]

+

Global Features

[zzzzzzzz]

↓

Graph Embedding

[xxxxxxxxxxxxxxxx

 yyyyyyyyyyyyyyyy

 zzzzzzzz]
```

No information is averaged,

summed,

or discarded.

The vectors are simply placed side by side.

---

# 15.6.125 Tensor Shapes During Concatenation

Suppose

our batch contains

16 crystals.

Before concatenation,

```text
node_graph

↓

[16,64]
```

```text
edge_graph

↓

[16,64]
```

```text
global_graph

↓

[16,32]
```

After concatenation,

```text
graph_embedding

↓

[16,160]
```

The batch dimension remains unchanged.

Only the feature dimension increases.

---

# 15.6.126 Why Use `dim=1`?

Consider

```python
torch.cat(

    [

        node_graph,

        edge_graph

    ],

    dim=1

)
```

Suppose

both tensors have shape

```text
[16,64]
```

The result becomes

```text
[16,128]
```

Each graph still occupies

one row.

Only the number of features increases.

If instead we used

```python
dim=0
```

the output would become

```text
[32,64]
```

This would incorrectly double

the number of graphs.

Therefore,

feature concatenation must always occur along

the feature dimension.

---

# 15.6.127 Why Not Add the Three Vectors?

One might wonder

why we do not simply compute

```python
graph_embedding = (

    node_graph

    +

    edge_graph

)
```

or

```python
node_graph

+

edge_graph

+

global_graph
```

Addition requires

all vectors to have

the same dimension.

Even if they did,

addition would mix

different types of information.

For example,

```text
Node Feature 7

+

Edge Feature 7
```

has

no clear physical interpretation.

Concatenation avoids this problem.

Every feature remains distinct,

allowing the prediction network to decide

how much importance to assign to each component.

---

# 15.6.128 The Graph Embedding as a Learned Material Descriptor

The concatenated vector

is much more than

a collection of numbers.

It is

a learned descriptor

of the entire material.

Traditional materials informatics often relies on

handcrafted descriptors,

such as

- average electronegativity,
- atomic radius statistics,
- packing fraction,
- valence electron concentration.

MEGNet learns these descriptors automatically.

Instead of manually designing features,

the network discovers

the representations

that best predict the target property.

This learned graph embedding is therefore

one of the most valuable outputs of the network.

It can even be reused

for downstream tasks,

such as

- clustering materials,
- dimensionality reduction,
- transfer learning,
- visualization of materials space.

---

# 15.6.129 Gradient Flow Through Concatenation

Unlike

linear layers

or

activation functions,

concatenation contains

no learnable parameters.

Nevertheless,

it is fully differentiable.

Suppose

the prediction loss produces gradients

with respect to

the graph embedding.

During backpropagation,

PyTorch automatically separates these gradients into

```text
Node Gradient

↓

Node Representation
```

```text
Edge Gradient

↓

Edge Representation
```

```text
Global Gradient

↓

Global Representation
```

Each branch then propagates

its gradients independently

through the remainder of the computational graph.

Thus,

concatenation does not interrupt

gradient flow.

---

# 15.6.130 Inspecting the Final Graph Embedding

After concatenation,

we can verify

the tensor dimensions.

```python
print(graph_embedding.shape)
```

Suppose

the batch size is

16.

The output becomes

```text
torch.Size([16,160])
```

Every crystal is now represented

by

one

160-dimensional feature vector.

Regardless of

whether the crystal contains

20 atoms,

100 atoms,

or

500 atoms,

the graph embedding always has

the same dimensionality.

This fixed-size representation makes it possible

to use ordinary fully connected neural networks

for the final prediction.

---

# 15.6.131 State of the Network Before Prediction

The forward pass has now completed

every graph-specific operation.

The current state is

| Component | Shape | Status |
|-----------|--------|--------|
| Graph Embedding | `[Batch Size,160]` | Complete material representation |

All structural information has been compressed

into a fixed-length vector.

From this point onward,

the network behaves

like a conventional multilayer perceptron.

The remaining task is straightforward:

transform

the graph embedding

into

the desired material property.

---

## **MEGNet Perspective**

The concatenation stage creates the final graph embedding by combining complementary information extracted from nodes, edges, and the global state.

Unlike handcrafted feature engineering, this representation is learned entirely from data and adapts automatically to the prediction task.

Because the graph embedding has a fixed dimensionality regardless of crystal size, it provides a universal representation that can be processed by standard feed-forward neural networks.

The next stage of the forward pass implements the **prediction head**, which transforms this learned graph embedding into the final predicted material property.

# 15.6.132 The Prediction Head

The graph embedding constructed in the previous section represents the **entire crystal** as a single fixed-length vector.

At this point,

all graph-specific computations have been completed.

There will be

- no more message passing,
- no more edge updates,
- no more node updates,
- no more graph aggregation.

From this point onward,

the problem becomes identical to a standard supervised learning task.

The network now possesses one feature vector for every crystal.

Its remaining task is simply to map that feature vector to the desired target property.

This mapping is performed by the **prediction head**.

---

# 15.6.133 Why Is a Prediction Head Needed?

Suppose

the graph embedding has dimension

```text
160
```

This means

each crystal is represented by

```text
[

x₁,

x₂,

x₃,

...

x₁₆₀

]
```

These numbers contain rich information about

- atomic environments,
- bonding,
- crystal structure,
- long-range interactions,
- global material characteristics.

However,

they are **latent features**.

They are not yet

- formation energies,
- band gaps,
- elastic constants,
- bulk moduli.

A neural network must still learn

how these latent features relate to the target property.

---

# 15.6.134 The Prediction Network

Recall the prediction network defined during initialization.

```python
self.predictor = nn.Sequential(

    nn.Linear(

        160,

        128

    ),

    nn.ReLU(),

    nn.Linear(

        128,

        64

    ),

    nn.ReLU(),

    nn.Linear(

        64,

        output_dim

    )

)
```

Suppose

```text
output_dim = 1
```

for formation energy prediction.

The computation becomes

```text
160

↓

128

↓

64

↓

1
```

Unlike the Graph Network blocks,

this is simply

a multilayer perceptron (MLP).

---

# 15.6.135 Executing the Prediction Head

The forward pass requires only one statement.

```python
prediction = self.predictor(

    graph_embedding

)
```

Although this appears to be a single operation,

PyTorch automatically executes

every layer

inside the sequential module.

Conceptually,

the computation becomes

```text
Graph Embedding

↓

Linear Layer

↓

ReLU

↓

Linear Layer

↓

ReLU

↓

Linear Layer

↓

Prediction
```

---

# 15.6.136 The First Linear Layer

Suppose

the graph embedding has shape

```text
[16,160]
```

The first layer computes

$$
\mathbf{h}_1

=

\mathbf{W}_1

\mathbf{g}

+

\mathbf{b}_1,
$$

where

- $\mathbf{g}$ is the graph embedding,
- $\mathbf{W}_1$ has dimensions

$$
128\times160,
$$

- $\mathbf{b}_1$

is the bias vector.

The output becomes

```text
[16,128]
```

Each graph is now represented by

128 learned features.

---

# 15.6.137 First Nonlinear Transformation

The network immediately applies

```python
nn.ReLU()
```

which computes

$$
\mathrm{ReLU}(x)

=

\max(0,x).
$$

This operation

introduces nonlinearity.

Without nonlinear activation functions,

the entire prediction network would collapse into

one large linear transformation,

greatly limiting its expressive power.

---

# 15.6.138 The Second Hidden Layer

The second hidden layer computes

```text
128

↓

64
```

Mathematically,

$$
\mathbf{h}_2

=

\mathbf{W}_2

\mathbf{h}_1

+

\mathbf{b}_2.
$$

The output tensor becomes

```text
[16,64]
```

The network has now compressed

the learned material representation

into

64 high-level predictive features.

---

# 15.6.139 The Output Layer

Finally,

the last linear layer computes

```text
64

↓

1
```

Mathematically,

$$
\hat{y}

=

\mathbf{W}_3

\mathbf{h}_2

+

\mathbf{b}_3.
$$

The resulting tensor has shape

```text
[16,1]
```

Each row corresponds to

one crystal.

Each value corresponds to

one predicted property.

For example,

```text
Crystal 1

↓

−3.42 eV
```

```text
Crystal 2

↓

−2.87 eV
```

```text
Crystal 3

↓

−1.65 eV
```

---

# 15.6.140 Why Is There No Activation After the Output Layer?

Notice

that the final layer

contains

no ReLU,

no sigmoid,

and

no softmax.

This is intentional.

Formation energy,

band gap,

and many other material properties

are continuous variables.

They may be

positive,

negative,

or

zero.

Applying

ReLU

would incorrectly force

all predictions

to be nonnegative.

Therefore,

regression models typically leave

the final layer

unconstrained.

---

# 15.6.141 Prediction Shapes for Different Tasks

The output dimension depends entirely

on the learning problem.

### Single-property regression

```text
Formation Energy

↓

[Batch Size,1]
```

---

### Multi-property regression

Suppose

the network predicts

- formation energy,
- band gap,
- bulk modulus.

Then

```text
output_dim = 3
```

The prediction tensor becomes

```text
[Batch Size,3]
```

Every graph now produces

three predicted values simultaneously.

---

### Binary classification

Suppose

we wish to predict

whether a material is stable.

The network may output

```text
[Batch Size,1]
```

representing

a logit.

A sigmoid activation

is then applied

during loss computation.

---

### Multi-class classification

Suppose

we classify

crystal structures into

five categories.

Then

```text
output_dim = 5
```

The prediction tensor becomes

```text
[Batch Size,5]
```

Each column corresponds

to one class score.

---

# 15.6.142 Why Use an MLP Instead of Another Graph Network?

At this point,

all graph information

has already been integrated

into

the graph embedding.

Additional message passing

would no longer provide

new structural information.

Instead,

the task becomes

ordinary function approximation.

Multilayer perceptrons

are extremely effective

at learning mappings

between

high-dimensional feature vectors

and

target variables.

Therefore,

MEGNet concludes

with

a conventional feed-forward neural network.

---

# 15.6.143 Tensor Evolution Throughout the Prediction Head

Suppose

our batch contains

16 crystals.

The tensor evolves as

```text
Graph Embedding

↓

[16,160]
```

```text
Linear

↓

[16,128]
```

```text
ReLU

↓

[16,128]
```

```text
Linear

↓

[16,64]
```

```text
ReLU

↓

[16,64]
```

```text
Output Layer

↓

[16,1]
```

This final tensor

contains

the predicted material properties.

---

# 15.6.144 Returning the Prediction

The last statement

inside the forward pass

is

```python
return prediction
```

This returns

the prediction tensor

to the caller.

When the user executes

```python
output = model(graph)
```

the variable

```python
output
```

receives

the tensor returned

by

```python
return prediction
```

This marks

the completion

of the forward pass.

---

# 15.6.145 The Entire Forward Pass at a Glance

The complete computation performed by MEGNet can now be summarized as

```text
Crystal Structure

↓

Graph Construction

↓

Node Embedding

↓

Edge Projection

↓

Global Projection

↓

Graph Network Block 1

↓

Graph Network Block 2

↓

Graph Network Block 3

↓

Node Readout

↓

Edge Readout

↓

Graph Embedding

↓

Prediction Head

↓

Predicted Material Property
```

Every stage is differentiable.

Every parameter is optimized simultaneously through backpropagation.

Together,

these components enable MEGNet to learn complex relationships between crystal structure and material properties directly from data.

---

## **MEGNet Perspective**

The prediction head is the final stage of the MEGNet forward pass.

After the graph embedding has condensed the structural, chemical, and global information of the crystal into a fixed-length representation, the prediction network transforms this learned descriptor into the target material property using a conventional multilayer perceptron.

Although architecturally simple compared to the Graph Network blocks, the prediction head is responsible for learning the final mapping between the latent material representation and the desired output.

With the completion of the `return prediction` statement, the forward propagation of MEGNet is fully defined. The next major topic is the **backward pass and training process**, where gradients are computed through every stage of this forward computation and the network parameters are optimized using gradient descent.

# 15.6.146 The Complete `forward()` Method

After studying every individual component in detail,

we are finally ready to assemble the complete `forward()` method.

Nothing in the following implementation should appear mysterious anymore.

Every line has already been explained mathematically,

algorithmically,

and physically.

Instead of seeing isolated operations,

we can now understand the forward pass as one continuous computational pipeline.

---

# 15.6.147 Complete Implementation

```python
def forward(self, graph):

    # ---------------------------------------------------
    # Extract graph components
    # ---------------------------------------------------

    x = graph.x

    edge_index = graph.edge_index

    edge_attr = graph.edge_attr

    u = graph.u

    batch = graph.batch

    # ---------------------------------------------------
    # Initial feature transformations
    # ---------------------------------------------------

    x = self.node_embedding(x)

    edge_attr = self.edge_projection(edge_attr)

    u = self.global_projection(u)

    # ---------------------------------------------------
    # Graph Network Blocks
    # ---------------------------------------------------

    for block in self.blocks:

        x, edge_attr, u = block(

            x,

            edge_attr,

            edge_index,

            u

        )

    # ---------------------------------------------------
    # Graph Readout
    # ---------------------------------------------------

    node_graph = global_mean_pool(

        x,

        batch

    )

    edge_batch = batch[edge_index[0]]

    edge_graph = global_mean_pool(

        edge_attr,

        edge_batch

    )

    global_graph = u

    graph_embedding = torch.cat(

        [

            node_graph,

            edge_graph,

            global_graph

        ],

        dim=1

    )

    # ---------------------------------------------------
    # Property Prediction
    # ---------------------------------------------------

    prediction = self.predictor(

        graph_embedding

    )

    return prediction
```

This is the entire forward computation performed by the MEGNet model.

Everything else inside the network exists only to support this computation.

---

# 15.6.148 Understanding the Forward Pass as a Pipeline

Although the implementation occupies fewer than fifty lines,

it performs a remarkably sophisticated computation.

Conceptually,

the network executes the following sequence.

```text
Raw Crystal

↓

Graph Construction

↓

Extract Graph Components

↓

Node Embedding

↓

Edge Projection

↓

Global Projection

↓

Graph Network Block 1

↓

Graph Network Block 2

↓

Graph Network Block 3

↓

Node Readout

↓

Edge Readout

↓

Concatenate Features

↓

Prediction Network

↓

Material Property
```

Notice that

every stage builds upon the previous one.

No stage operates independently.

---

# 15.6.149 Following One Atom Through the Entire Network

To better understand the complete forward pass,

let us follow a single atom from the beginning to the end.

Initially,

the atom is represented only by

its atomic number.

```text
Fe

↓

26
```

The embedding layer converts it into

```text
26

↓

64-dimensional embedding
```

Initially,

this embedding contains only information about iron itself.

During the first Graph Network block,

the atom receives information from its neighboring atoms.

```text
Neighbor Atoms

↓

Iron Atom

↓

Updated Embedding
```

During the second Graph Network block,

information originating from neighbors of neighbors also reaches the atom.

As additional Graph Network blocks are applied,

the atom gradually develops a representation of its wider chemical environment.

Eventually,

its embedding no longer represents only an isolated iron atom.

Instead,

it represents

> an iron atom located within a specific crystal environment.

This distinction is fundamental.

---

# 15.6.150 Following One Bond Through the Network

The edge features undergo a similar transformation.

Initially,

an edge contains only

the Gaussian-expanded bond distance.

```text
Distance

↓

Gaussian Basis

↓

26 Features
```

The projection network transforms this into

a latent representation.

During message passing,

the edge receives information from

its two connected atoms

and

the global state.

Consequently,

the edge embedding evolves from

a purely geometric description

into

a learned representation of the chemical interaction between the two atoms.

---

# 15.6.151 Following the Global State

The evolution of the global state is even more interesting.

Initially,

it may contain

```text
Temperature

Pressure
```

or simply

zeros.

After every Graph Network block,

information from

every atom

and

every bond

is aggregated into the global vector.

Eventually,

this vector becomes

a compressed summary

of the entire crystal.

Unlike node embeddings,

which describe local environments,

the global state represents

the material as a whole.

---

# 15.6.152 What Does the Graph Embedding Actually Represent?

After pooling,

the network constructs

the graph embedding.

This vector is not merely

a mathematical object.

It is a learned descriptor of the material.

Each dimension of the graph embedding captures

some aspect of the crystal,

such as

- chemical composition,
- local coordination,
- bonding patterns,
- long-range structural effects,
- interactions among different atomic environments.

Importantly,

these features are **not manually engineered**.

They emerge naturally during training as the network minimizes the prediction loss.

---

# 15.6.153 Why the Entire Pipeline Is Differentiable

One of the greatest strengths of MEGNet is that

every operation in the forward pass is differentiable.

The embedding layer,

projection networks,

Graph Network blocks,

pooling operations,

concatenation,

and prediction head

all participate in the computational graph.

Consequently,

when the prediction error is computed,

gradients can flow continuously from the output layer back to the earliest embedding layer.

This allows the network to optimize all parameters simultaneously.

---

# 15.6.154 Computational Graph of the Forward Pass

The complete computational graph can be visualized as

```text
Atomic Numbers ───────────────┐
                              │
                              ▼
                      Node Embedding
                              │
                              │
Bond Distances ───────────────┐
                              ▼
                      Edge Projection
                              │
                              │
Global Features ──────────────┐
                              ▼
                    Global Projection
                              │
                              ▼
                  Graph Network Block 1
                              │
                              ▼
                  Graph Network Block 2
                              │
                              ▼
                  Graph Network Block 3
                              │
          ┌──────────┬─────────┘
          ▼          ▼
    Node Pool   Edge Pool
          │          │
          └────┬─────┘
               ▼
      Concatenate with Global State
               │
               ▼
        Graph Embedding
               │
               ▼
       Prediction Network
               │
               ▼
     Predicted Material Property
```

This computational graph is exactly what PyTorch records automatically during the forward pass.

No manual gradient calculations are required.

---

# 15.6.155 Why Understanding the Forward Pass Matters

Many researchers learn to use MEGNet simply by importing the model and calling

```python
prediction = model(graph)
```

While this is sufficient for using the software,

it is insufficient for conducting research.

A researcher must understand

- how information flows,
- why every tensor exists,
- where gradients originate,
- how architectural modifications influence learning,
- how to debug unexpected behavior,
- how to extend the model.

Only by understanding the forward pass in detail can one confidently develop new graph neural network architectures for materials science.

---

# 15.6.156 Summary of the Complete Forward Pass

The MEGNet forward pass transforms a raw crystal structure into a predicted material property through a sequence of well-defined stages:

1. **Graph construction** converts the crystal into nodes, edges, and a global state.
2. **Embedding and projection layers** transform raw features into latent representations.
3. **Graph Network blocks** iteratively exchange information among nodes, edges, and the global state.
4. **Readout operations** aggregate distributed information into graph-level representations.
5. **Concatenation** combines node, edge, and global summaries into a single graph embedding.
6. **The prediction head** maps this embedding to the target material property.

Each stage contributes a specific function, and together they form one end-to-end differentiable model capable of learning complex structure–property relationships directly from crystal data.

---

## **MEGNet Perspective**

The `forward()` method represents the complete inference procedure of the MEGNet architecture. It integrates graph construction, feature initialization, iterative message passing, graph-level aggregation, and supervised prediction into a unified computational framework. Although the implementation is compact, it encapsulates the central ideas of modern graph neural networks for materials science.

With the forward pass now fully understood, the next major section of this chapter will shift from **how MEGNet computes predictions** to **how MEGNet learns**. We will examine the backward pass, automatic differentiation, loss computation, gradient propagation through Graph Network blocks, parameter updates using optimizers, and the complete end-to-end training algorithm that enables MEGNet to discover structure–property relationships from large materials datasets.

# 15.7 Training MEGNet: From Forward Propagation to Learning

Until now, we have focused exclusively on **forward propagation**.

During the forward pass,

the network receives a crystal graph,

processes it through multiple Graph Network blocks,

constructs a graph embedding,

and finally predicts a material property.

However,

at this stage,

the network **does not actually know anything**.

Every parameter inside

- the embedding layer,
- the edge projection network,
- the global projection network,
- every Graph Network block,
- the prediction head,

was initially assigned random values.

Consequently,

the first prediction produced by the network is usually very poor.

For example,

suppose the true formation energy of a crystal is

```text
−3.52 eV
```

Immediately after random initialization,

MEGNet may predict

```text
4.81 eV
```

or

```text
−0.63 eV
```

or even

```text
12.74 eV
```

These predictions are incorrect because the network has not yet learned the relationship between crystal structure and material properties.

The central question of deep learning is therefore

> **How does a neural network transform random parameters into a model capable of making accurate scientific predictions?**

Answering this question requires understanding the **training process**.

---

# 15.7.1 What Does It Mean for a Neural Network to Learn?

Humans learn through experience.

A child gradually learns to recognize

- faces,
- animals,
- objects,

after observing many examples.

Similarly,

a neural network learns by repeatedly observing

examples consisting of

```text
Input

↓

Correct Output
```

For MEGNet,

the input is

```text
Crystal Graph
```

while the correct output is

```text
Material Property
```

For example,

```text
Crystal Structure

↓

Formation Energy

↓

−3.52 eV
```

or

```text
Crystal Structure

↓

Band Gap

↓

1.84 eV
```

The network repeatedly compares its own prediction with the correct answer.

Whenever the prediction is incorrect,

the parameters are adjusted slightly.

After thousands of such adjustments,

the model gradually learns the underlying relationship between structure and property.

Learning is therefore **nothing more than the repeated adjustment of parameters in order to reduce prediction error**.

---

# 15.7.2 The Four Fundamental Stages of Training

Every deep learning algorithm,

regardless of architecture,

follows the same four-stage cycle.

```text
Forward Propagation

↓

Loss Computation

↓

Backward Propagation

↓

Parameter Update
```

After the parameters are updated,

the cycle begins again.

```text
Forward

↓

Loss

↓

Backward

↓

Update

↓

Forward

↓

Loss

↓

Backward

↓

Update

↓

...
```

This process continues

for hundreds or even thousands of epochs.

Eventually,

the model converges toward parameters that minimize the prediction error.

---

# 15.7.3 Stage 1 — Forward Propagation

The first stage is the forward pass,

which we have already studied in detail.

Given a crystal graph,

MEGNet computes

```python
prediction = model(graph)
```

Internally,

this performs

```text
Graph Construction

↓

Embedding

↓

Projection

↓

Graph Network Blocks

↓

Readout

↓

Prediction Head

↓

Predicted Property
```

Suppose

the predicted formation energy is

```text
−2.94 eV
```

The network has now produced an answer.

However,

it has no way of knowing

whether this answer is correct.

The next stage measures its quality.

---

# 15.7.4 Stage 2 — Loss Computation

The loss function measures

how different the prediction is from the true value.

Suppose

```text
Prediction

↓

−2.94 eV
```

while the true value is

```text
−3.52 eV
```

The prediction error is

```text
0.58 eV
```

Rather than adjusting parameters directly from this error,

deep learning defines a mathematical function called the **loss function**.

The loss converts prediction errors into a single scalar value that quantifies model performance.

A smaller loss indicates better predictions.

A larger loss indicates poorer predictions.

The optimization process aims to minimize this loss.

---

# 15.7.5 Stage 3 — Backward Propagation

Once the loss has been computed,

the network asks

> **Which parameters caused this error?**

This is the purpose of backpropagation.

Backpropagation computes

the gradient of the loss

with respect to **every trainable parameter** in the network.

For MEGNet,

this includes

- embedding vectors,
- projection layer weights,
- Graph Network block parameters,
- prediction head weights.

Each parameter receives its own gradient,

indicating

- the direction in which it should change,
- and how strongly it should change.

Backpropagation is therefore the mechanism through which the network learns from its mistakes.

---

# 15.7.6 Stage 4 — Parameter Update

Once the gradients have been computed,

an optimization algorithm,

such as Adam,

updates the parameters.

Conceptually,

every parameter moves

slightly

in the direction that reduces the loss.

After the update,

the network performs another forward pass.

The new prediction is usually

slightly more accurate

than before.

Repeating this process thousands of times gradually transforms

random parameters

into

scientifically meaningful representations.

---

# 15.7.7 The Training Loop

Combining the four stages produces the complete training algorithm.

```text
Initialize Parameters

↓

Forward Pass

↓

Prediction

↓

Compute Loss

↓

Compute Gradients

↓

Update Parameters

↓

Repeat
```

This loop is executed

for every mini-batch

throughout the training process.

Eventually,

the network converges to a set of parameters capable of accurately predicting material properties from crystal structures.

---

# 15.7.8 Why the Training Process Is More Important Than the Architecture

Many beginners believe that choosing the right neural network architecture is the most important part of deep learning.

In reality,

the architecture merely defines

**what the model can compute**.

The training algorithm determines

**what the model actually learns**.

A sophisticated architecture trained poorly will perform worse than a simpler architecture trained correctly.

Therefore,

understanding

- loss functions,
- gradient computation,
- optimization algorithms,
- learning rate scheduling,

is just as important as understanding Graph Network blocks themselves.

---

## **MEGNet Perspective**

The forward pass defines how MEGNet transforms a crystal graph into a predicted material property.

The training process defines how MEGNet improves this transformation over time.

Every training iteration consists of four tightly connected stages: forward propagation, loss computation, backward propagation, and parameter optimization.

Together, these stages allow the network to gradually discover complex relationships between crystal structure, atomic interactions, and material properties without explicitly programming those relationships.

In the following sections, we will examine each stage of this learning process in depth, beginning with the mathematical foundations of **loss functions** and how prediction errors are quantified.

# 15.7.9 Loss Functions: Measuring Prediction Error

A neural network can only improve if it knows **how wrong** its predictions are.

Suppose that MEGNet predicts

```text
Formation Energy

↓

−2.85 eV
```

while the correct value is

```text
Formation Energy

↓

−3.52 eV
```

Clearly,

the prediction is incorrect.

However,

the neural network cannot understand words such as

> "good"

or

> "bad."

Instead,

it requires a precise mathematical quantity that measures the quality of its prediction.

This quantity is called the **loss**.

The loss converts prediction quality into a single numerical value that optimization algorithms can minimize.

Without a loss function,

deep learning would be impossible.

---

# 15.7.10 Prediction Error Versus Loss

Many beginners confuse

**prediction error**

with

**loss**.

Although they are closely related,

they are not identical.

Suppose

the true value is

```text
−3.52 eV
```

and the prediction is

```text
−2.85 eV
```

The prediction error is

$$
\hat{y}-y
=
-2.85-(-3.52)
=
0.67\ \text{eV}.
$$

This quantity simply measures the numerical difference.

The loss function,

however,

transforms this error into a value suitable for optimization.

Different loss functions treat errors differently.

Therefore,

the same prediction error may produce different loss values depending on the chosen loss function.

---

# 15.7.11 Desired Properties of a Loss Function

A useful loss function should satisfy several important properties.

First,

the loss should be

```text
Zero
```

when the prediction is perfect.

Second,

the loss should increase as the prediction becomes less accurate.

Third,

the loss should be differentiable,

allowing gradients to be computed during backpropagation.

Finally,

the loss should provide a smooth optimization landscape,

making gradient descent stable.

These requirements eliminate many seemingly reasonable error measures.

---

# 15.7.12 Regression Versus Classification Losses

The appropriate loss function depends on the prediction task.

For **regression** problems,

such as predicting

- formation energy,
- band gap,
- elastic modulus,

the output is continuous.

Common regression losses include

- Mean Squared Error (MSE),
- Mean Absolute Error (MAE),
- Huber Loss.

For **classification** problems,

such as predicting

- stable versus unstable,
- crystal system,
- material class,

the output represents discrete categories.

Classification commonly uses

- Binary Cross Entropy,
- Cross Entropy Loss,
- Focal Loss.

Since MEGNet is most frequently used for predicting continuous material properties,

our primary focus will be regression losses.

---

# 15.7.13 Why Mean Squared Error Is the Standard Choice

The most widely used regression loss in deep learning is the

**Mean Squared Error (MSE)**.

The idea is simple.

Instead of using the raw prediction error,

we square every error.

Squaring has two important consequences.

First,

all errors become positive.

For example,

errors of

```text
+2
```

and

```text
−2
```

both contribute equally.

Second,

large errors receive much greater penalties than small ones.

This encourages the network to reduce particularly inaccurate predictions.

---

# 15.7.14 Mathematical Definition of MSE

Suppose

a mini-batch contains

$N$

training examples.

The Mean Squared Error is

$$
\boxed{
L_{\mathrm{MSE}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
}
$$

where

- $y_i$ is the true material property,
- $\hat{y}_i$ is the predicted property,
- $N$ is the number of samples in the batch.

The square ensures

that positive and negative errors contribute equally.

---

# 15.7.15 Example: Computing MSE by Hand

Suppose

MEGNet predicts the formation energies of four crystals.

| Crystal | True Value (eV) | Prediction (eV) |
|---------|----------------:|----------------:|
| A | −3.20 | −3.10 |
| B | −2.80 | −2.50 |
| C | −1.50 | −1.90 |
| D | −4.10 | −3.70 |

First,

compute the prediction errors.

| Crystal | Error |
|---------|-------:|
| A | 0.10 |
| B | 0.30 |
| C | −0.40 |
| D | 0.40 |

Notice that

Crystal C

has a negative error.

If we simply averaged these errors,

positive and negative values could cancel each other.

This would incorrectly suggest that the model performs well.

---

# 15.7.16 Squaring the Errors

The next step is

to square every error.

| Crystal | Error | Squared Error |
|---------|------:|--------------:|
| A | 0.10 | 0.0100 |
| B | 0.30 | 0.0900 |
| C | −0.40 | 0.1600 |
| D | 0.40 | 0.1600 |

Every contribution is now positive.

Large mistakes also contribute much more strongly than small ones.

For example,

an error of

```text
0.40
```

produces

```text
0.16
```

while an error of

```text
0.10
```

produces only

```text
0.01
```

The larger error contributes

16 times more.

---

# 15.7.17 Computing the Mean

Finally,

average the squared errors.

$$
L_{\mathrm{MSE}}
=
\frac{
0.01
+
0.09
+
0.16
+
0.16
}{4}
=
0.105.
$$

Therefore,

the Mean Squared Error for this mini-batch is

```text
0.105
```

This single scalar summarizes

the quality of all predictions in the batch.

---

# 15.7.18 Interpreting the MSE

The absolute value of the MSE

depends on the units of the target property.

Suppose

formation energies are measured in

electron volts.

Then

the MSE has units of

$$
\text{eV}^2.
$$

Because of the squared units,

MSE is primarily an optimization objective,

not a directly interpretable scientific quantity.

For this reason,

research papers often report

**Mean Absolute Error (MAE)**

alongside MSE,

since MAE retains the original physical units.

---

# 15.7.19 Computing MSE in PyTorch

Fortunately,

there is no need to implement MSE manually.

PyTorch provides

a built-in loss function.

```python
import torch.nn as nn

criterion = nn.MSELoss()
```

Computing the loss requires only

```python
loss = criterion(

    prediction,

    target

)
```

Suppose

```python
prediction.shape

↓

[32,1]
```

and

```python
target.shape

↓

[32,1]
```

The output is

```python
tensor(0.0837)
```

This single value represents

the average squared prediction error

for the entire mini-batch.

---

# 15.7.20 Why MSE Works Well for Materials Property Prediction

Material property prediction often requires

high numerical precision.

For example,

an error of

0.8 eV

in band-gap prediction

is much more serious than

an error of

0.05 eV.

Because MSE squares prediction errors,

large mistakes receive disproportionately large penalties.

Consequently,

during optimization,

the network focuses strongly on correcting its worst predictions.

This behavior has made MSE

the standard loss function

for many materials informatics regression tasks,

including

- formation energy prediction,
- adsorption energy prediction,
- bulk modulus estimation,
- elastic constant prediction,
- dielectric constant prediction.

---

## **MEGNet Perspective**

The loss function provides the quantitative measure that drives learning.

For regression problems, the Mean Squared Error transforms prediction differences into a smooth, differentiable objective that can be minimized through gradient descent.

Although the forward pass computes predictions, the loss function determines **how good those predictions are**.

Its numerical value serves as the starting point for backpropagation, allowing gradients to flow backward through every Graph Network block and ultimately update the parameters of the MEGNet model.

In the next section, we will study **Mean Absolute Error (MAE)**, compare it mathematically with MSE, and explain why modern materials informatics papers frequently report MAE as their primary evaluation metric while still training with MSE.

# 15.7.21 Mean Absolute Error (MAE)

Although Mean Squared Error (MSE) is the most common loss function used during training,

it is **not** always the best metric for evaluating a model.

In materials science literature,

you will frequently encounter statements such as

```text
Formation Energy MAE

↓

0.028 eV/atom
```

or

```text
Band Gap MAE

↓

0.31 eV
```

Notice that researchers usually report **MAE**, not MSE.

This naturally raises two important questions.

1. Why do researchers train with MSE but report MAE?
2. What makes MAE easier to interpret scientifically?

To answer these questions,

we must first understand what Mean Absolute Error actually measures.

---

# 15.7.22 The Idea Behind MAE

Suppose

the true formation energy is

```text
−3.52 eV
```

and

MEGNet predicts

```text
−2.85 eV
```

The prediction error is

$$
\hat{y}-y
=
0.67\ \text{eV}.
$$

Instead of squaring this error,

MAE simply takes its absolute value.

$$
|0.67|
=
0.67.
$$

Likewise,

if the prediction error had been

```text
−0.67 eV
```

its absolute value would still be

```text
0.67 eV.
```

Therefore,

positive and negative errors contribute equally,

but large errors are **not exaggerated** as they are in MSE.

---

# 15.7.23 Mathematical Definition of MAE

Suppose

a mini-batch contains

$N$

samples.

The Mean Absolute Error is

$$
\boxed{
L_{\mathrm{MAE}}
=
\frac{1}{N}
\sum_{i=1}^{N}
|\hat{y}_i-y_i|
}
$$

where

- $y_i$ is the true property,
- $\hat{y}_i$ is the predicted property.

Unlike MSE,

the error is not squared.

Instead,

only its magnitude is considered.

---

# 15.7.24 Computing MAE by Hand

Consider the same four-crystal example used earlier.

| Crystal | True (eV) | Prediction (eV) |
|---------|----------:|----------------:|
| A | −3.20 | −3.10 |
| B | −2.80 | −2.50 |
| C | −1.50 | −1.90 |
| D | −4.10 | −3.70 |

The prediction errors are

| Crystal | Error |
|---------|------:|
| A | 0.10 |
| B | 0.30 |
| C | −0.40 |
| D | 0.40 |

Taking absolute values gives

| Crystal | Absolute Error |
|---------|---------------:|
| A | 0.10 |
| B | 0.30 |
| C | 0.40 |
| D | 0.40 |

Finally,

compute the average.

$$
L_{\mathrm{MAE}}
=
\frac{
0.10
+
0.30
+
0.40
+
0.40
}{4}
=
0.30.
$$

Therefore,

the MAE equals

```text
0.30 eV.
```

---

# 15.7.25 Comparing MAE and MSE

Notice the difference.

For the same predictions,

we obtained

```text
MAE

↓

0.30
```

and

```text
MSE

↓

0.105
```

These numbers are **not directly comparable**

because

they measure prediction error differently.

MAE measures

the average magnitude of the errors.

MSE measures

the average squared magnitude.

---

# 15.7.26 Why MSE Penalizes Large Errors More Strongly

Consider two prediction errors.

```text
Error A

↓

0.5 eV
```

```text
Error B

↓

2.0 eV
```

For MAE,

their contributions are simply

```text
0.5

and

2.0
```

The larger error contributes

four times more.

For MSE,

the contributions become

$$
0.5^2
=
0.25
$$

and

$$
2^2
=
4.
$$

Now,

the larger error contributes

**sixteen times more**.

Thus,

MSE strongly emphasizes large mistakes,

whereas MAE treats every error in direct proportion to its size.

---

# 15.7.27 Visual Comparison

Suppose

the prediction error gradually increases.

The corresponding penalties behave differently.

| Error | MAE | MSE |
|------:|----:|----:|
| 0.1 | 0.1 | 0.01 |
| 0.5 | 0.5 | 0.25 |
| 1.0 | 1.0 | 1.00 |
| 2.0 | 2.0 | 4.00 |
| 3.0 | 3.0 | 9.00 |

Notice that

MSE grows much faster.

This is why optimization using MSE focuses strongly on correcting very inaccurate predictions.

---

# 15.7.28 Physical Interpretation of MAE

One of the greatest advantages of MAE

is that

its units remain identical to the target property.

Suppose

formation energy

is measured in

electron volts.

Then

MAE is also measured in

electron volts.

Therefore,

an MAE of

```text
0.05 eV
```

means

> On average, the model makes an error of approximately **0.05 eV**.

This interpretation is immediately meaningful to materials scientists.

By contrast,

an MSE of

```text
0.0025 eV²
```

is much harder to interpret physically.

---

# 15.7.29 Why Materials Informatics Papers Report MAE

Most research papers evaluate models using MAE because

it answers a practical scientific question.

Instead of asking

> "What is the average squared error?"

researchers ask

> "How far, on average, are the predictions from the true values?"

For example,

consider these two statements.

```text
MSE = 0.0025 eV²
```

and

```text
MAE = 0.05 eV
```

The second statement is immediately understandable.

A scientist can directly judge whether

an average error of

0.05 eV

is acceptable for a particular application.

---

# 15.7.30 Computing MAE in PyTorch

PyTorch also provides

a built-in implementation.

```python
import torch.nn as nn

criterion = nn.L1Loss()
```

Here,

"L1"

refers to

the

$L_1$

norm,

which computes

absolute differences.

The loss is calculated as

```python
loss = criterion(

    prediction,

    target

)
```

If

```python
prediction.shape

↓

[32,1]
```

and

```python
target.shape

↓

[32,1]
```

the output is

a single scalar,

for example,

```python
tensor(0.047)
```

representing

the average absolute prediction error

for the mini-batch.

---

# 15.7.31 Should We Train with MAE or MSE?

This is a common question.

There is no universal answer,

but modern materials informatics typically follows the pattern

| Purpose | Common Choice |
|----------|---------------|
| Training | MSE |
| Evaluation | MAE |

The reason is practical.

MSE has smooth derivatives,

which generally make optimization easier.

MAE,

on the other hand,

provides a physically meaningful performance metric.

Consequently,

many MEGNet implementations optimize

MSE,

while reporting

MAE

on the validation and test sets.

---

# 15.7.32 Choosing Between MSE and MAE

The choice depends on the application.

Use **MSE** when

- large errors must be heavily penalized,
- smooth optimization is important,
- accurate regression is the primary goal.

Use **MAE** when

- interpretability is important,
- the reported error should have physical units,
- comparisons with published materials informatics benchmarks are required.

In practice,

many researchers monitor **both** throughout training.

---

## **MEGNet Perspective**

Mean Absolute Error complements Mean Squared Error by providing a directly interpretable measure of prediction quality.

While MSE is often preferred during optimization because of its mathematical properties,

MAE expresses the average prediction error in the same physical units as the target property,

making it particularly valuable for reporting scientific results.

For this reason,

materials informatics studies commonly **train MEGNet using MSE** while **evaluating performance using MAE**.

The next section will introduce the **Huber Loss**, a hybrid loss function that combines the stability of MSE with the robustness of MAE, making it especially useful when datasets contain noisy measurements or outliers.

# 15.7.33 Huber Loss: Combining the Advantages of MSE and MAE

We have now studied the two most important regression loss functions.

**Mean Squared Error (MSE)**

penalizes large errors very strongly.

**Mean Absolute Error (MAE)**

treats all errors proportionally.

Both approaches have advantages,

but both also have limitations.

This naturally raises an important question.

> Can we design a loss function that behaves like MSE for small errors while behaving like MAE for very large errors?

The answer is

**yes**.

This is precisely the motivation behind the **Huber Loss**.

Huber Loss combines the smooth optimization properties of MSE with the robustness of MAE.

For this reason,

it is widely used in scientific machine learning,

computer vision,

robotics,

and increasingly in materials informatics when datasets contain noisy measurements or experimental uncertainty.

---

# 15.7.34 Why MSE Can Become Problematic

Suppose

our training dataset contains

1000 crystals.

Most formation energies are measured accurately,

but one experimental measurement contains a large error.

For example,

suppose the prediction errors are

```text
0.04

0.08

0.06

0.05

...

7.5
```

The final error

```text
7.5
```

is an outlier.

Using MSE,

its contribution becomes

$$
7.5^2
=
56.25.
$$

Meanwhile,

an ordinary prediction error of

```text
0.05
```

contributes only

$$
0.05^2
=
0.0025.
$$

The outlier therefore dominates the loss.

During optimization,

the neural network may spend most of its effort correcting this single unusual sample,

instead of improving predictions for the remaining dataset.

---

# 15.7.35 Why MAE Can Also Be Difficult

MAE solves the outlier problem because

errors increase only linearly.

The previous outlier contributes

```text
7.5
```

instead of

```text
56.25.
```

However,

MAE introduces another challenge.

Its derivative changes abruptly at zero.

Mathematically,

the absolute value function

$$
|x|
$$

is not differentiable at

$$
x=0.
$$

Although modern optimization algorithms can still work with MAE,

its optimization landscape is generally less smooth than MSE.

This sometimes leads to slower convergence during training.

---

# 15.7.36 The Motivation Behind Huber Loss

Huber Loss was designed to combine

the best characteristics of both loss functions.

Its behavior is

- quadratic for small errors,
- linear for large errors.

Consequently,

small prediction errors are optimized like MSE,

while large outliers are treated more gently,

similar to MAE.

This balance often produces more stable training on noisy datasets.

---

# 15.7.37 Mathematical Definition of Huber Loss

Let

the prediction error be

$$
e
=
\hat{y}-y.
$$

Huber Loss is defined as

$$
\boxed{
L(e)
=
\begin{cases}
\frac{1}{2}e^2,
&
|e|
\le
\delta,
\\
\\
\delta
\left(
|e|
-
\frac{\delta}{2}
\right),
&
|e|
>
\delta.
\end{cases}
}
$$

Here,

$\delta$

is called the **transition threshold**.

It determines

where the loss changes

from quadratic behavior

to linear behavior.

---

# 15.7.38 Understanding the Two Regions

Huber Loss has two distinct operating modes.

### Region 1

Small errors

$$
|e|
\le
\delta.
$$

The loss becomes

$$
L
=
\frac12e^2.
$$

This is simply

Mean Squared Error

up to a constant factor.

Consequently,

small prediction errors are optimized smoothly.

---

### Region 2

Large errors

$$
|e|
>
\delta.
$$

The loss becomes

linear.

Instead of growing quadratically,

it increases approximately

proportionally to

the magnitude of the error.

Large outliers therefore cannot dominate the optimization process.

---

# 15.7.39 Example Calculation

Suppose

the transition threshold is

$$
\delta=1.
$$

Consider

two prediction errors.

### Example A

```text
Error

↓

0.40
```

Since

$$
0.40<1,
$$

Huber Loss becomes

$$
L
=
\frac12
(0.40)^2
=
0.08.
$$

This behaves exactly like MSE.

---

### Example B

```text
Error

↓

3.00
```

Since

$$
3>1,
$$

Huber Loss becomes

$$
L
=
1
\left(
3-\frac12
\right)
=
2.5.
$$

Notice the difference.

Using MSE,

the loss would have been

$$
3^2=9.
$$

Huber Loss produces only

```text
2.5.
```

Thus,

the influence of the outlier is dramatically reduced.

---

# 15.7.40 Comparing MSE, MAE, and Huber Loss

Consider several prediction errors.

| Error | MAE | MSE | Huber ($\delta=1$) |
|------:|----:|----:|-------------------:|
| 0.2 | 0.20 | 0.04 | 0.02 |
| 0.5 | 0.50 | 0.25 | 0.125 |
| 1.0 | 1.00 | 1.00 | 0.50 |
| 2.0 | 2.00 | 4.00 | 1.50 |
| 3.0 | 3.00 | 9.00 | 2.50 |

Several important observations emerge.

For small errors,

Huber Loss resembles MSE.

For large errors,

its growth becomes nearly linear,

similar to MAE.

---

# 15.7.41 Visual Interpretation

If we plotted

loss

against

prediction error,

the three curves would appear very different.

```text
Loss

^

|

|                         MSE

|                      /

|                   /

|                /

|             /

|          /

|       /

|----/---------------------------->

|   /

|  /

| /

|/

Prediction Error
```

MAE

would be

a V-shaped straight line.

Huber Loss

would begin as a smooth parabola

near zero,

then gradually transition into

straight lines.

Thus,

Huber Loss provides

the smoothness of MSE

near the optimum

while avoiding excessive penalties

for large outliers.

---

# 15.7.42 Computing Huber Loss in PyTorch

PyTorch includes

Huber Loss

as a built-in function.

```python
import torch.nn as nn

criterion = nn.HuberLoss(

    delta=1.0

)
```

The loss is computed exactly as before.

```python
loss = criterion(

    prediction,

    target

)
```

Changing

```python
delta
```

changes the point

where the loss switches

from quadratic

to linear.

---

# 15.7.43 Choosing the Value of $\delta$

The parameter

$\delta$

controls the behavior of Huber Loss.

A small

$\delta$

causes the loss

to behave like MAE

for most prediction errors.

A very large

$\delta$

makes the loss

behave almost identically to MSE.

Therefore,

$\delta$

must be chosen

based on

the expected scale

of prediction errors.

For many applications,

the default value

```text
1.0
```

works well,

although it can be tuned as a hyperparameter.

---

# 15.7.44 When Should Materials Scientists Use Huber Loss?

Huber Loss is particularly useful when

the dataset contains

- experimental measurement errors,
- noisy labels,
- occasional incorrect database entries,
- rare outlier materials.

Examples include

- experimentally measured band gaps,
- thermal conductivity datasets,
- mechanical property databases,
- high-throughput experimental measurements.

In contrast,

large DFT databases,

such as

Materials Project,

often contain relatively consistent labels,

making MSE a perfectly reasonable choice.

---

# 15.7.45 Comparing the Three Regression Losses

The three major regression losses can now be summarized.

| Property | MAE | MSE | Huber |
|-----------|-----|-----|--------|
| Penalizes large errors strongly | No | Yes | Moderately |
| Robust to outliers | Excellent | Poor | Good |
| Smooth optimization | Moderate | Excellent | Excellent |
| Easy physical interpretation | Excellent | Moderate | Moderate |
| Common in materials ML | Evaluation | Training | Noisy datasets |

Each loss has its own strengths.

There is no universally superior choice.

The appropriate loss depends on

the scientific problem,

the quality of the dataset,

and the optimization objectives.

---

## **MEGNet Perspective**

Huber Loss provides a practical compromise between Mean Squared Error and Mean Absolute Error.

By behaving quadratically for small prediction errors and linearly for large ones,

it combines stable optimization with robustness against noisy or incorrect labels.

Although MSE remains the most common training objective for large computational materials databases,

Huber Loss is often advantageous when working with experimental datasets where measurement uncertainty and outliers are unavoidable.

Having established how prediction error is quantified, we are now ready to study **backpropagation**, the algorithm that uses these loss values to compute gradients and teach MEGNet how to improve its predictions.

# 15.7.46 Backpropagation: How MEGNet Learns

We have now completed the first two stages of the training process.

```text
Crystal Graph

↓

Forward Pass

↓

Prediction

↓

Loss
```

At this point,

MEGNet knows

**how wrong** its prediction is.

For example,

suppose the true formation energy is

```text
−3.52 eV
```

while the network predicts

```text
−2.91 eV.
```

The loss function computes

a numerical penalty,

perhaps

```text
0.37
```

or

```text
0.12,
```

depending on the chosen loss function.

However,

knowing the loss alone is **not enough**.

The neural network still faces a much more difficult question.

> **Which of the millions of parameters caused this error?**

Even more importantly,

> **How should each parameter change to reduce the error?**

Answering these questions is the purpose of **backpropagation**.

Backpropagation is the mathematical algorithm that transforms a single loss value into millions of parameter updates.

Without backpropagation,

deep learning would not exist.

---

# 15.7.47 Why Updating Every Parameter Is Difficult

Consider a simplified MEGNet model.

Suppose it contains

```text
Node Embedding

↓

4,000 parameters
```

```text
Edge Projection

↓

2,000 parameters
```

```text
Graph Network Blocks

↓

850,000 parameters
```

```text
Prediction Head

↓

30,000 parameters
```

The entire model therefore contains

```text
886,000
```

trainable parameters.

Now suppose

the prediction is incorrect.

Which parameter should change?

Should

parameter

```text
#17
```

increase?

Should

parameter

```text
#251,843
```

decrease?

Should

every parameter change?

If so,

by how much?

These questions cannot be answered by trial and error.

A systematic mathematical procedure is required.

---

# 15.7.48 The Central Idea Behind Backpropagation

Imagine standing on a mountain.

Your goal is

to reach the lowest point in the valley.

Before taking a step,

you need to know

which direction goes downhill.

Backpropagation performs exactly this task.

Instead of mountains,

the landscape is

the **loss function**.

Instead of physical position,

the coordinates are

the neural network parameters.

Backpropagation determines

the downhill direction

for every parameter simultaneously.

Optimization algorithms then move

slightly

in that direction.

Repeating this process

gradually reduces the loss.

---

# 15.7.49 Gradients: The Language of Learning

The information that tells the optimizer

how to change a parameter

is called its **gradient**.

Mathematically,

the gradient measures

how sensitive the loss is

to changes in a parameter.

Suppose

one parameter is called

$$
w.
$$

The corresponding gradient is

$$
\boxed{
\frac{\partial L}{\partial w}
}
$$

where

- $L$ is the loss,
- $w$ is one trainable parameter.

This quantity answers an important question.

> **If we change the parameter slightly, how much will the loss change?**

Every trainable parameter inside MEGNet has its own gradient.

---

# 15.7.50 Interpreting the Gradient

Suppose

the gradient equals

$$
\frac{\partial L}{\partial w}
=
5.
$$

A positive gradient means

that increasing

$w$

causes

the loss to increase.

Therefore,

the optimizer should

**decrease**

$w$.

Now suppose

$$
\frac{\partial L}{\partial w}
=
-3.
$$

A negative gradient indicates

that increasing

$w$

reduces

the loss.

Therefore,

the optimizer should

**increase**

$w$.

Thus,

the **sign**

of the gradient determines

the direction of movement.

---

# 15.7.51 The Magnitude of the Gradient

The magnitude of the gradient is equally important.

Consider two parameters.

Parameter A

has gradient

$$
0.002.
$$

Parameter B

has gradient

$$
8.5.
$$

Parameter B has a much larger influence

on the loss.

Consequently,

the optimizer should adjust

Parameter B

much more aggressively

than

Parameter A.

Therefore,

the gradient provides

both

- direction,
- importance.

---

# 15.7.52 A Simple Example

Suppose

our neural network contains

only one parameter,

$$
w.
$$

Further suppose

the loss function is

$$
L(w)
=
(w-4)^2.
$$

The graph of this function

is

a parabola.

The minimum occurs at

$$
w=4.
$$

If

the current parameter value is

$$
w=8,
$$

then

the loss equals

$$
16.
$$

Clearly,

the parameter is not optimal.

How can the network discover

the correct direction?

---

# 15.7.53 Computing the Gradient

Differentiate

the loss with respect to

$w$.

$$
\frac{dL}{dw}
=
2(w-4).
$$

Substituting

$$
w=8
$$

gives

$$
\frac{dL}{dw}
=
8.
$$

The gradient is positive.

Therefore,

the optimizer knows

that

$w$

should decrease.

Suppose instead

$$
w=2.
$$

Then

$$
\frac{dL}{dw}
=
-4.
$$

Now the gradient is negative.

The optimizer therefore increases

$w$.

Without ever seeing the graph,

the network always knows

which direction leads toward the minimum.

---

# 15.7.54 Extending This Idea to Millions of Parameters

The previous example involved

only one parameter.

MEGNet,

however,

contains hundreds of thousands

or even millions

of parameters.

Backpropagation simply repeats

the same principle

for every trainable parameter.

Conceptually,

it computes

```text
Loss

↓

Gradient of Parameter 1

↓

Gradient of Parameter 2

↓

Gradient of Parameter 3

↓

...

↓

Gradient of Parameter N
```

Each parameter receives

its own individualized update direction.

---

# 15.7.55 Why Manual Gradient Calculation Is Impossible

Imagine attempting to compute

the gradients

for a modern MEGNet architecture by hand.

Every Graph Network block contains

- multiple linear layers,
- activation functions,
- concatenation operations,
- pooling operations,
- residual connections.

These blocks are stacked repeatedly.

The resulting computational graph may contain

millions of mathematical operations.

Manually differentiating such a graph

would require

thousands of pages of symbolic mathematics.

Fortunately,

this is unnecessary.

Modern deep learning frameworks,

including PyTorch,

perform **automatic differentiation**.

Backpropagation is executed automatically,

regardless of the network's complexity.

---

# 15.7.56 Backpropagation Is an Application of the Chain Rule

Although PyTorch performs backpropagation automatically,

the underlying mathematics is remarkably elegant.

Every operation in the forward pass

depends on previous operations.

For example,

```text
Atomic Features

↓

Embedding

↓

Graph Network Block

↓

Readout

↓

Prediction

↓

Loss
```

The loss depends on

the prediction.

The prediction depends on

the graph embedding.

The graph embedding depends on

the Graph Network blocks.

The Graph Network blocks depend on

the embedding layer.

Calculating gradients through this chain

requires one of the most important ideas in calculus:

the **chain rule**.

Backpropagation is essentially

the systematic application of the chain rule

to an extremely large computational graph.

---

## **MEGNet Perspective**

Backpropagation transforms prediction errors into learning signals.

Starting from a single scalar loss,

it computes the gradient of that loss with respect to every trainable parameter in the MEGNet architecture.

These gradients quantify how each parameter influences the prediction error,

providing the information required by optimization algorithms such as Adam to improve the model.

In the next section, we will study the **chain rule of calculus** in detail and show how it enables gradients to propagate backward through every layer of the MEGNet computational graph, from the prediction head all the way back to the atomic embedding layer.

# 15.7.57 The Chain Rule: The Mathematical Foundation of Backpropagation

In the previous section,

we learned that backpropagation computes

the gradient of the loss

with respect to every trainable parameter.

However,

we have not yet answered

the most important mathematical question.

> **How can gradients travel backward through dozens of neural network layers?**

The answer lies in one of the most fundamental concepts in calculus:

the **Chain Rule**.

Every deep learning framework,

including PyTorch,

TensorFlow,

and JAX,

implements backpropagation by repeatedly applying the chain rule.

Without the chain rule,

modern deep learning would be mathematically impossible.

---

# 15.7.58 Understanding Composite Functions

The chain rule applies whenever

one function depends on another function.

Suppose we have

three functions.

$$
x
\rightarrow
u(x)
\rightarrow
v(u)
\rightarrow
L(v)
$$

Graphically,

the computation looks like

```text
Input

↓

Function 1

↓

Intermediate Variable

↓

Function 2

↓

Intermediate Variable

↓

Function 3

↓

Loss
```

Each stage depends on

the previous stage.

The loss does not depend directly on

the input.

Instead,

its dependence is indirect.

This is exactly how neural networks operate.

---

# 15.7.59 A Neural Network Is a Giant Composite Function

Consider

the MEGNet forward pass.

```text
Atomic Features

↓

Embedding Layer

↓

Graph Network Block

↓

Graph Embedding

↓

Prediction Head

↓

Prediction

↓

Loss
```

Every stage depends on

the output of the previous stage.

Mathematically,

we can write

$$
L
=
L(P(G(E(x))))
$$

where

- $x$ represents the input graph,
- $E$ is the embedding layer,
- $G$ represents all Graph Network blocks,
- $P$ is the prediction head,
- $L$ is the loss function.

Notice that

the embedding layer does **not** directly influence the loss.

Instead,

its influence passes through

every later stage.

The chain rule allows us to compute

this indirect influence.

---

# 15.7.60 A Simple Numerical Example

Before studying neural networks,

let us examine

a much simpler example.

Suppose

$$
u
=
2x
$$

and

$$
L
=
u^2.
$$

The complete computation is

```text
x

↓

Multiply by 2

↓

u

↓

Square

↓

Loss
```

Suppose

$$
x=3.
$$

Then

$$
u=6
$$

and

$$
L=36.
$$

Now ask

> **How sensitive is the loss to changes in $x$?**

---

# 15.7.61 Direct Differentiation

One possibility is

to substitute

$$
u=2x
$$

directly.

Then

$$
L=(2x)^2=4x^2.
$$

Differentiating gives

$$
\frac{dL}{dx}
=
8x.
$$

Substituting

$$
x=3
$$

produces

$$
\frac{dL}{dx}
=
24.
$$

This is the correct gradient.

However,

deep neural networks contain millions of intermediate variables.

Direct substitution becomes impossible.

Instead,

we apply the chain rule.

---

# 15.7.62 Applying the Chain Rule

The chain rule states

$$
\boxed{
\frac{dL}{dx}
=
\frac{dL}{du}
\cdot
\frac{du}{dx}
}
$$

Rather than differentiating the entire expression at once,

we differentiate

each operation separately.

First,

differentiate

$$
L=u^2.
$$

This gives

$$
\frac{dL}{du}
=
2u.
$$

Next,

differentiate

$$
u=2x.
$$

This gives

$$
\frac{du}{dx}
=
2.
$$

Finally,

multiply the two derivatives.

Since

$$
u=6,
$$

we obtain

$$
\frac{dL}{dx}
=
(2\times6)
\times2
=
24.
$$

The result is identical to

direct differentiation.

---

# 15.7.63 The Key Insight

Notice something remarkable.

Neither derivative

"knows"

about the complete computation.

The derivative

$$
\frac{dL}{du}
$$

depends only on

the squaring operation.

The derivative

$$
\frac{du}{dx}
$$

depends only on

the multiplication by two.

Each operation computes

its own local derivative.

The overall gradient is obtained simply by multiplying these local derivatives together.

This is the central idea of backpropagation.

---

# 15.7.64 Extending to More Layers

Suppose the computation contains

five stages.

```text
x

↓

A

↓

B

↓

C

↓

Prediction

↓

Loss
```

The chain rule becomes

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial P}
\cdot
\frac{\partial P}{\partial C}
\cdot
\frac{\partial C}{\partial B}
\cdot
\frac{\partial B}{\partial A}
\cdot
\frac{\partial A}{\partial x}.
$$

Each layer contributes

one local derivative.

The complete gradient is simply

the product of all local derivatives.

---

# 15.7.65 Applying the Chain Rule to MEGNet

Now consider

the simplified MEGNet pipeline.

```text
Node Embedding

↓

Graph Network Blocks

↓

Readout

↓

Prediction Head

↓

Loss
```

Suppose we wish to compute

the gradient

of the loss

with respect to

the embedding layer parameters.

The chain rule gives

$$
\frac{\partial L}
{\partial \theta_{\text{embed}}}
=
\frac{\partial L}
{\partial P}
\cdot
\frac{\partial P}
{\partial R}
\cdot
\frac{\partial R}
{\partial G}
\cdot
\frac{\partial G}
{\partial \theta_{\text{embed}}},
$$

where

- $P$ represents the prediction head,
- $R$ represents the readout,
- $G$ represents the Graph Network blocks,
- $\theta_{\text{embed}}$ denotes the embedding parameters.

Thus,

even though the embedding layer is far from the output,

its gradients can still be computed.

---

# 15.7.66 Why the Name "Backpropagation"?

During the forward pass,

information flows

from

the input

toward

the prediction.

```text
Crystal

↓

Embedding

↓

Graph Network

↓

Prediction

↓

Loss
```

During gradient computation,

the direction reverses.

```text
Loss

↓

Prediction Head

↓

Graph Network

↓

Embedding
```

The gradients therefore propagate

**backward**

through the computational graph.

This reverse flow

gives the algorithm its name:

**backpropagation**.

---

# 15.7.67 Why Local Derivatives Are Enough

One of the greatest strengths

of the chain rule

is that

every layer only needs to know

how to differentiate itself.

For example,

the embedding layer

does not need to understand

the prediction head.

The prediction head

does not need to understand

Graph Network blocks.

Every operation simply computes

its own local derivative.

PyTorch automatically multiplies

these local derivatives together

to obtain the complete gradient.

This modularity is what allows

deep neural networks

with hundreds of layers

to remain computationally tractable.

---

# 15.7.68 The Chain Rule in Automatic Differentiation

During the forward pass,

PyTorch records

every mathematical operation

performed by the network.

Internally,

it builds

a computational graph.

Each node of this graph

stores

- its output,
- its input,
- and the rule required to compute its derivative.

When

```python
loss.backward()
```

is executed,

PyTorch traverses this graph

in reverse order,

applying the chain rule

at every operation.

The resulting gradients are accumulated automatically,

without requiring the user to derive any equations manually.

---

# 15.7.69 Why the Chain Rule Is the Heart of Deep Learning

Every optimization algorithm,

whether

- stochastic gradient descent,
- Adam,
- RMSProp,
- AdamW,

depends on gradients.

Every gradient in a neural network

is ultimately computed

using the chain rule.

Consequently,

although the chain rule is introduced in elementary calculus,

it remains one of the most important mathematical ideas

in modern artificial intelligence.

Without it,

training models such as

MEGNet,

CGCNN,

ALIGNN,

or

M3GNet

would be impossible.

---

## **MEGNet Perspective**

Backpropagation is fundamentally an application of the chain rule to the computational graph created during the forward pass.

Rather than differentiating the entire network as one enormous equation,

each layer computes only its own local derivative.

These local derivatives are multiplied together as gradients propagate backward from the loss to the earliest layers.

This elegant mathematical principle enables MEGNet to efficiently compute gradients for millions of parameters, making end-to-end learning of complex crystal structure–property relationships computationally feasible.

In the next section, we will examine **automatic differentiation in PyTorch**, exploring how the computational graph is constructed during the forward pass and how `loss.backward()` traverses that graph to compute gradients automatically.

# 15.7.70 Automatic Differentiation in PyTorch

After understanding the mathematics of the chain rule,

we are ready to answer an important practical question.

> **How does PyTorch actually compute millions of gradients automatically?**

The answer is

**Automatic Differentiation**,

usually called

**Autograd**.

Autograd is one of the most important components of PyTorch.

Without it,

every neural network researcher would have to manually derive and implement millions of gradient equations.

Modern deep learning would become practically impossible.

Fortunately,

PyTorch performs all of these calculations automatically.

---

# 15.7.71 What Is Automatic Differentiation?

Automatic differentiation is a computational technique that evaluates derivatives exactly by systematically applying the chain rule.

It is important to distinguish automatic differentiation from two other approaches.

### Numerical differentiation

Approximates derivatives using finite differences.

For example,

$$
\frac{df}{dx}
\approx
\frac{f(x+h)-f(x)}{h}.
$$

Although conceptually simple,

this approach suffers from

- approximation errors,
- numerical instability,
- extremely high computational cost.

It is unsuitable for modern deep learning.

---

### Symbolic differentiation

Manipulates mathematical expressions algebraically.

Computer algebra systems such as Mathematica perform symbolic differentiation.

While mathematically exact,

symbolic differentiation becomes extremely complicated for neural networks containing millions of operations.

---

### Automatic differentiation

Automatic differentiation combines the advantages of both approaches.

It computes **exact derivatives up to machine precision**

while remaining computationally efficient.

This is precisely the method used by PyTorch.

---

# 15.7.72 The Computational Graph

The key idea behind Autograd is the

**computational graph**.

Every operation executed during the forward pass becomes

a node

inside a directed graph.

Consider a simple computation.

```python
x = torch.tensor(2.0, requires_grad=True)

y = 3 * x

z = y ** 2
```

The corresponding computational graph is

```text
x

↓

Multiply by 3

↓

y

↓

Square

↓

z
```

Notice that

every variable knows

where it came from.

This information allows PyTorch to reconstruct

the entire sequence of computations.

---

# 15.7.73 Recording Operations During the Forward Pass

Suppose

the MEGNet forward pass executes

```python
x = self.node_embedding(x)

x = self.block(x)

graph = global_mean_pool(x)

prediction = self.predictor(graph)
```

While these statements appear to perform only numerical calculations,

Autograd records every operation.

Internally,

PyTorch stores information such as

```text
Input Tensor

↓

Linear Layer

↓

ReLU

↓

Linear Layer

↓

Pooling

↓

Prediction
```

Each node stores

- its inputs,
- its outputs,
- the mathematical operation,
- the rule for computing its derivative.

This stored graph is called

the **dynamic computational graph**.

---

# 15.7.74 Why PyTorch Uses a Dynamic Graph

Unlike some older deep learning frameworks,

PyTorch builds the computational graph

**during execution**.

This means

every forward pass constructs

a fresh graph.

For example,

suppose an

`if`

statement changes the network behavior.

```python
if training:

    x = layer1(x)

else:

    x = layer2(x)
```

PyTorch records

only the operations that were actually executed.

This makes PyTorch extremely flexible.

Researchers can create

- variable-sized graphs,
- conditional architectures,
- recursive models,
- graph neural networks,

without manually defining static computation graphs.

This flexibility is one reason why PyTorch has become the dominant framework for graph neural network research.

---

# 15.7.75 The Role of `requires_grad`

Not every tensor needs gradients.

Only trainable parameters require differentiation.

PyTorch uses

```python
requires_grad=True
```

to indicate

that a tensor should participate in automatic differentiation.

For example,

```python
x = torch.tensor(

    2.0,

    requires_grad=True

)
```

This tells Autograd

to record every operation involving

`x`.

If

```python
requires_grad=False
```

the tensor behaves as

ordinary numerical data,

and no gradients are stored.

---

# 15.7.76 Example: Building a Computational Graph

Consider

```python
x = torch.tensor(

    2.0,

    requires_grad=True

)

y = x * 5

z = y + 3

loss = z ** 2
```

The graph becomes

```text
x

↓

×5

↓

y

↓

+3

↓

z

↓

Square

↓

Loss
```

Although the code contains only four lines,

Autograd has already stored

the complete computational graph.

Nothing has yet been differentiated.

The graph is simply waiting

for the backward pass.

---

# 15.7.77 The `grad_fn` Attribute

Every tensor produced by an operation

stores information about

how it was created.

For example,

```python
print(loss.grad_fn)
```

might produce

```text
<PowBackward0>
```

Similarly,

```python
print(z.grad_fn)
```

might display

```text
<AddBackward0>
```

and

```python
print(y.grad_fn)
```

may return

```text
<MulBackward0>
```

These objects represent

the local derivative rules

required during backpropagation.

They are the building blocks

of the computational graph.

---

# 15.7.78 Executing `loss.backward()`

Once the forward pass has finished,

the loss has been computed.

Training now proceeds with

```python
loss.backward()
```

This single command performs

all gradient calculations.

Internally,

PyTorch

1. starts at the loss,
2. traverses the computational graph backward,
3. applies the chain rule at every node,
4. accumulates gradients for every trainable parameter.

Conceptually,

the process is

```text
Loss

↓

Square

↓

Addition

↓

Multiplication

↓

Input Tensor
```

Every gradient is computed automatically.

---

# 15.7.79 Where Are the Gradients Stored?

After calling

```python
loss.backward()
```

every trainable parameter contains

its corresponding gradient.

For example,

```python
print(x.grad)
```

might output

```text
tensor(250.)
```

Similarly,

for a neural network,

every layer stores gradients inside

its parameters.

```python
model.layer.weight.grad
```

```python
model.layer.bias.grad
```

```python
embedding.weight.grad
```

Each gradient tells the optimizer

how that parameter should change.

---

# 15.7.80 Autograd Inside MEGNet

The same process occurs

inside MEGNet,

although on a much larger scale.

During the forward pass,

PyTorch records

every operation,

including

- node embeddings,
- edge projections,
- Graph Network updates,
- message aggregation,
- pooling,
- concatenation,
- prediction layers.

After

```python
loss.backward()
```

Autograd computes gradients for

every parameter,

including

- atomic embedding vectors,
- edge projection weights,
- node update networks,
- edge update networks,
- global update networks,
- prediction head weights.

Even though

the network may contain

millions of parameters,

the user never writes

a single derivative equation.

---

# 15.7.81 Why Automatic Differentiation Is Revolutionary

Before automatic differentiation became standard,

implementing new neural network architectures required

manual derivation

of every gradient.

Researchers often spent

more time deriving equations

than designing models.

Autograd completely changed this workflow.

Today,

a researcher can design

an entirely new graph neural network,

implement only the forward pass,

and allow PyTorch

to derive

all backward computations automatically.

This capability has dramatically accelerated research in

deep learning,

scientific machine learning,

and materials informatics.

---

# 15.7.82 A Common Misconception

Many beginners believe

that

```python
loss.backward()
```

updates the neural network.

This is incorrect.

`loss.backward()`

only computes gradients.

The parameters themselves remain unchanged.

Parameter updates occur later,

when the optimizer executes

```python
optimizer.step()
```

Thus,

the training process consists of two distinct stages.

```text
loss.backward()

↓

Compute Gradients
```

followed by

```text
optimizer.step()

↓

Update Parameters
```

Understanding this distinction is essential for correctly implementing training loops.

---

## **MEGNet Perspective**

Automatic differentiation is the computational engine that makes training modern graph neural networks practical.

By recording every operation performed during the forward pass and applying the chain rule in reverse, PyTorch automatically computes exact gradients for every trainable parameter in the MEGNet architecture.

This allows researchers to focus on designing better models rather than manually deriving complex gradient equations.

In the next section, we will examine **gradient propagation through the MEGNet architecture**, following how the loss gradient flows backward from the prediction head through the readout layer, Graph Network blocks, and finally to the atomic embedding layer.

# 15.7.83 Gradient Propagation Through the MEGNet Architecture

After executing

```python
loss.backward()
```

PyTorch begins traversing the computational graph

from the loss

toward the earliest trainable parameters.

For a small neural network,

this backward journey is relatively simple.

For MEGNet,

however,

the process is considerably more sophisticated.

Unlike conventional feed-forward neural networks,

MEGNet contains

- node features,
- edge features,
- global state vectors,
- message passing,
- graph pooling,
- multiple Graph Network blocks.

Consequently,

gradients must flow through several interconnected computational paths before reaching the earliest parameters.

Understanding this gradient flow is essential for understanding **how MEGNet actually learns**.

---

# 15.7.84 The Direction of Information Flow

During the forward pass,

information flows

from the crystal

toward the prediction.

```text
Crystal Structure

↓

Graph Construction

↓

Node Embedding

↓

Graph Network Block 1

↓

Graph Network Block 2

↓

Graph Network Block 3

↓

Readout

↓

Prediction Head

↓

Prediction

↓

Loss
```

Backpropagation reverses this direction.

The gradient begins

at the loss

and propagates backward.

```text
Loss

↓

Prediction Head

↓

Readout

↓

Graph Network Block 3

↓

Graph Network Block 2

↓

Graph Network Block 1

↓

Node Embedding

↓

Atomic Features
```

Every layer receives

a learning signal

from the layer above it.

---

# 15.7.85 Step 1 — Gradient at the Loss

Suppose

the prediction is

$$
\hat{y}
=
-2.91
\text{ eV}
$$

while

the true value is

$$
y
=
-3.52
\text{ eV}.
$$

Using Mean Squared Error,

the loss is

$$
L
=
(\hat y-y)^2.
$$

The first gradient computed is

$$
\frac{\partial L}
{\partial\hat y}.
$$

This derivative answers

the question

> **How should the prediction change to reduce the loss?**

At this stage,

only the prediction value is involved.

The network parameters are not yet considered.

---

# 15.7.86 Step 2 — Gradient Through the Prediction Head

The prediction

is produced by

the multilayer perceptron.

```text
Graph Embedding

↓

Linear

↓

ReLU

↓

Linear

↓

Prediction
```

The chain rule computes

$$
\frac{\partial L}
{\partial\theta_{\text{predictor}}},
$$

where

$\theta_{\text{predictor}}$

represents

all weights and biases

inside the prediction head.

Each linear layer receives

its own gradient.

Each bias receives

its own gradient.

The prediction head is therefore

the first component to learn.

---

# 15.7.87 Step 3 — Gradient Reaches the Graph Embedding

The prediction head receives

the graph embedding

as its input.

Therefore,

the chain rule next computes

$$
\frac{\partial L}
{\partial\mathbf g},
$$

where

$$
\mathbf g
$$

is the graph embedding.

This gradient answers

> **How should the graph representation change to reduce the prediction error?**

Notice

that

the graph embedding

is not a trainable parameter.

Instead,

it is an intermediate tensor.

Nevertheless,

its gradient is required,

because earlier layers depend upon it.

---

# 15.7.88 Step 4 — Backpropagation Through Concatenation

Recall that

the graph embedding was created using

```python
graph_embedding = torch.cat(

    [

        node_graph,

        edge_graph,

        global_graph

    ],

    dim=1

)
```

Concatenation itself contains

no trainable parameters.

However,

during backpropagation,

PyTorch automatically splits

the incoming gradient.

Conceptually,

the gradient

```text
Graph Embedding Gradient

↓

Split

↓

Node Gradient

↓

Edge Gradient

↓

Global Gradient
```

Each component now continues

along its own computational path.

---

# 15.7.89 Step 5 — Backpropagation Through Graph Readout

Consider

the node pooling operation.

```python
node_graph = global_mean_pool(

    x,

    batch

)
```

Suppose

a graph contains

five atoms.

During the forward pass,

their embeddings were averaged.

```text
Atom 1

Atom 2

Atom 3

Atom 4

Atom 5

↓

Mean

↓

Graph Vector
```

During the backward pass,

the incoming gradient is distributed

back to

all contributing atoms.

Every node embedding therefore receives

part of the graph-level learning signal.

Exactly the same idea applies

to edge pooling.

---

# 15.7.90 Step 6 — Gradient Through the Final Graph Network Block

The pooled node,

edge,

and global gradients

now enter

the final Graph Network block.

Recall

that this block contains

three update networks.

```text
Edge Update

↓

Node Update

↓

Global Update
```

Backpropagation computes gradients for

every parameter inside these update networks.

At the same time,

it computes gradients for

the updated

node,

edge,

and global tensors,

allowing learning signals to continue propagating

toward earlier blocks.

---

# 15.7.91 Step 7 — Gradient Through Multiple Graph Network Blocks

Suppose

MEGNet contains

three Graph Network blocks.

The backward computation becomes

```text
Loss

↓

Prediction Head

↓

Graph Block 3

↓

Graph Block 2

↓

Graph Block 1

↓

Embedding Layer
```

Each block receives gradients

from the block above it.

Each block computes

its own local derivatives.

The chain rule then propagates

the gradients further backward.

This process continues

until the earliest block

has been reached.

---

# 15.7.92 Step 8 — Gradient Reaches the Embedding Layer

Eventually,

the learning signal arrives

at the embedding layer.

Recall

that

the atomic number

```text
26
```

was transformed into

a learnable vector.

```text
Atomic Number

↓

Embedding Vector
```

The gradient now determines

how this embedding vector

should change

to reduce future prediction errors.

For example,

the learned representation of iron

may shift slightly

toward values

that improve formation energy prediction.

Thus,

even the atomic embeddings

are learned automatically.

---

# 15.7.93 Every Parameter Receives Its Own Gradient

By the time

backpropagation finishes,

every trainable parameter

inside MEGNet possesses

its own gradient.

This includes

- node embedding weights,
- edge projection weights,
- global projection weights,
- edge update networks,
- node update networks,
- global update networks,
- prediction head weights,
- biases.

Conceptually,

the result is

```text
Parameter

↓

Gradient
```

for every trainable tensor.

These gradients are stored internally

until the optimizer uses them.

---

# 15.7.94 Gradient Flow Is Not Information Flow

An important distinction should be made.

During the forward pass,

the network exchanges

chemical and structural information.

During the backward pass,

the network exchanges

learning signals.

Forward propagation answers

> **What property does this crystal have?**

Backpropagation answers

> **How should every parameter change to improve future predictions?**

Although both processes travel through

the same computational graph,

they serve completely different purposes.

---

# 15.7.95 The Complete Backward Pipeline

The entire backward computation

can now be summarized as

```text
Loss

↓

Prediction Gradient

↓

Prediction Head

↓

Graph Embedding Gradient

↓

Split Gradient

↓

Node Gradient

↓

Edge Gradient

↓

Global Gradient

↓

Graph Network Block 3

↓

Graph Network Block 2

↓

Graph Network Block 1

↓

Embedding Layer

↓

Atomic Embeddings
```

Every arrow in this diagram

represents

an application

of the chain rule.

Together,

these operations compute

millions of gradients

within a fraction of a second.

---

## **MEGNet Perspective**

Backpropagation through MEGNet is considerably more complex than in ordinary feed-forward neural networks because learning signals must propagate through graph pooling, multiple Graph Network blocks, and three interacting feature types: nodes, edges, and the global state.

Despite this complexity, PyTorch automatically computes every required gradient by repeatedly applying the chain rule to the computational graph created during the forward pass.

Once gradients have been computed for every trainable parameter, the learning process is still incomplete. The gradients merely indicate **how** the parameters should change. The actual parameter updates are performed by an optimization algorithm.

In the next section, we will study **gradient descent and optimization**, beginning with the mathematical principles that convert gradients into parameter updates and ultimately enable MEGNet to improve its predictions over successive training iterations.

# 15.7.96 Gradient Descent: Turning Gradients into Learning

At this point,

the backward pass has finished.

Every trainable parameter inside MEGNet now possesses a gradient.

For example,

```text
Node Embedding Weight

↓

Gradient = 0.0034
```

```text
Edge Update Weight

↓

Gradient = -0.128
```

```text
Prediction Layer Weight

↓

Gradient = 1.742
```

These gradients contain valuable information.

They tell us

- which direction reduces the loss,
- how sensitive the loss is to each parameter,
- which parameters should change the most.

However,

**nothing has actually learned yet.**

The parameters inside the network are still exactly the same as before the backward pass.

The gradients merely provide instructions.

Someone—or something—must still apply those instructions.

This is the role of the **optimizer**.

The mathematical principle behind nearly every optimizer is called **Gradient Descent**.

---

# 15.7.97 What Is Gradient Descent?

Gradient descent is one of the most important optimization algorithms in machine learning.

Its objective is remarkably simple.

> **Repeatedly adjust the model parameters so that the loss becomes smaller.**

Suppose

the neural network contains only one parameter,

$$
w.
$$

Imagine plotting

the loss

as a function of

$w$.

```text
Loss

^

|

|                    •

|                 •

|              •

|           •

|        •

|     •

|  •

|____________________________________>

                     Parameter w
```

The curve has

a minimum.

Our goal is to move

the parameter

toward

this minimum.

Gradient descent provides

a systematic procedure

for doing exactly that.

---

# 15.7.98 Why the Gradient Indicates the Best Direction

Recall that

the gradient

is

$$
\frac{\partial L}{\partial w}.
$$

This derivative measures

how rapidly

the loss changes

when the parameter changes.

Suppose

$$
\frac{\partial L}{\partial w}
=
5.
$$

A positive gradient means

that increasing

$w$

causes

the loss to increase.

Therefore,

to reduce the loss,

we should

decrease

$w$.

Now suppose

$$
\frac{\partial L}{\partial w}
=
-3.
$$

A negative gradient indicates

that increasing

$w$

reduces

the loss.

Therefore,

the parameter should move upward.

The gradient therefore tells us

both

the direction

and

the steepness

of the loss surface.

---

# 15.7.99 The Gradient Descent Update Rule

The parameter update equation is

$$
\boxed{
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{\partial L}{\partial w}
}
$$

where

- $w$ is the parameter,
- $\eta$ is the learning rate,
- $\frac{\partial L}{\partial w}$ is the gradient.

This deceptively simple equation

is responsible for nearly all modern deep learning.

Every optimizer,

including

Adam,

AdamW,

RMSProp,

and SGD,

is built upon this fundamental idea.

---

# 15.7.100 Understanding the Minus Sign

One of the most common beginner questions is

> **Why is there a minus sign?**

The answer comes directly from calculus.

The gradient points

toward

the direction

of **greatest increase**.

Our objective,

however,

is

to **decrease**

the loss.

Therefore,

we move

in the opposite direction.

Suppose

the gradient equals

```text
+4
```

Then

```text
New Parameter

↓

Old Parameter − Learning Rate × 4
```

The parameter decreases.

Conversely,

if the gradient equals

```text
−4,
```

then

subtracting a negative number

causes

the parameter to increase.

Thus,

the minus sign always moves

the parameters downhill.

---

# 15.7.101 The Learning Rate

The symbol

$$
\eta
$$

is called

the **learning rate**.

It determines

how far

the optimizer moves

during each update.

Think of descending a mountain.

You can

- take tiny steps,
- medium-sized steps,
- or enormous jumps.

The learning rate controls

the size of these steps.

Typical values include

```text
0.1

0.01

0.001

0.0001
```

For deep graph neural networks,

learning rates around

```text
0.001
```

are particularly common.

---

# 15.7.102 Example Calculation

Suppose

a parameter currently equals

$$
w
=
2.0.
$$

Suppose

the gradient is

$$
\frac{\partial L}{\partial w}
=
5.
$$

Finally,

suppose

the learning rate is

$$
\eta
=
0.1.
$$

Applying gradient descent,

$$
w_{\text{new}}
=
2
-
0.1
\times
5
=
1.5.
$$

The parameter has moved

closer

to the minimum.

---

# 15.7.103 What Happens After One Update?

A common misconception is that

one update

completes training.

In reality,

a single update changes

the parameters only slightly.

The workflow is

```text
Forward Pass

↓

Loss

↓

Backward Pass

↓

Gradient Descent

↓

Updated Parameters
```

Immediately afterward,

the network performs

another forward pass.

Because the parameters have changed,

the prediction also changes.

The new prediction is

usually

slightly better.

The process then repeats.

---

# 15.7.104 Training Is Thousands of Small Improvements

Suppose

the first prediction is

```text
Band Gap

↓

2.81 eV
```

while the true value is

```text
2.15 eV.
```

After one update,

the prediction might become

```text
2.74 eV.
```

After

100 updates,

perhaps

```text
2.29 eV.
```

After

10,000 updates,

perhaps

```text
2.16 eV.
```

The network rarely learns

through dramatic changes.

Instead,

learning emerges

from

thousands

or even millions

of tiny parameter updates.

---

# 15.7.105 The Loss Landscape

Although we often illustrate

the loss

using

a simple parabola,

real neural networks are far more complex.

A modern MEGNet model

may contain

over one million parameters.

The loss therefore exists

not on a two-dimensional curve,

but within

a space containing

millions of dimensions.

Each parameter defines

one coordinate axis.

Consequently,

the optimizer is effectively navigating

an enormous,

high-dimensional surface

in search of

a low-loss region.

---

# 15.7.106 Why Finding the Global Minimum Is Difficult

Unlike simple quadratic functions,

deep neural network loss surfaces

contain

- valleys,
- ridges,
- plateaus,
- saddle points,
- local minima.

A conceptual illustration is

```text
Loss

^

|        /\

|   ___ /  \__

|__/          \____

|__________________________>

        Parameters
```

Fortunately,

deep learning does not usually require

finding the absolute global minimum.

Instead,

finding

a sufficiently good minimum

often produces excellent predictive performance.

---

# 15.7.107 Gradient Descent for Millions of Parameters

Everything discussed for

a single parameter

extends naturally

to

millions of parameters.

Suppose

MEGNet contains

the parameter vector

$$
\mathbf{\theta}.
$$

The update equation becomes

$$
\boxed{
\mathbf{\theta}_{\text{new}}
=
\mathbf{\theta}_{\text{old}}
-
\eta
\nabla_{\mathbf{\theta}}L
}
$$

where

$$
\nabla_{\mathbf{\theta}}L
$$

represents

the gradient vector

containing

the derivative

of the loss

with respect to

every trainable parameter.

Instead of updating

one variable,

the optimizer updates

all parameters

simultaneously.

---

# 15.7.108 Why Plain Gradient Descent Is Rarely Used

Although gradient descent provides

the fundamental optimization principle,

modern deep learning rarely uses

the simplest version directly.

Instead,

researchers typically employ

more advanced optimizers,

including

- Stochastic Gradient Descent (SGD),
- Momentum,
- RMSProp,
- Adam,
- AdamW.

These methods

improve convergence,

increase stability,

and often produce better final models.

Among them,

**Adam**

has become

the most widely used optimizer

for graph neural networks,

including

CGCNN,

MEGNet,

M3GNet,

ALIGNN,

and CHGNet.

---

## **MEGNet Perspective**

Gradient descent converts the gradients produced during backpropagation into actual parameter updates.

By repeatedly moving every trainable parameter in the direction that reduces the loss, the optimizer gradually transforms randomly initialized weights into meaningful representations of atomic environments, chemical bonds, and crystal structures.

Although the basic gradient descent equation is simple, modern graph neural networks almost never rely on it directly. Instead, they employ adaptive optimization algorithms that accelerate convergence and improve training stability.

In the next section, we will study the **Adam optimizer** in depth, examining why it has become the standard optimization algorithm for MEGNet and most contemporary materials graph neural networks, and deriving its update equations step by step from first principles.

# 15.7.109 The Adam Optimizer: The Standard Optimizer for MEGNet

In the previous section,

we introduced the basic gradient descent update rule

$$
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{\partial L}{\partial w}.
$$

Although elegant,

this algorithm has a significant limitation.

Every parameter

moves using

the same learning rate.

Regardless of whether

a parameter is

- changing rapidly,
- changing slowly,
- frequently reversing direction,
- almost converged,

gradient descent treats them all identically.

For small optimization problems,

this approach is often sufficient.

However,

modern graph neural networks,

including MEGNet,

contain hundreds of thousands or millions of trainable parameters.

Different parameters learn at very different speeds.

Some parameters require large updates,

while others require only tiny adjustments.

Using a single learning rate for every parameter can therefore lead to

- slow convergence,
- unstable training,
- oscillations,
- poor final performance.

To overcome these limitations,

modern deep learning uses **adaptive optimization algorithms**.

The most successful and widely used of these algorithms is **Adam**.

---

# 15.7.110 What Does "Adam" Mean?

Adam stands for

**Adaptive Moment Estimation**.

The name reflects its two fundamental ideas.

1. **Adaptive**

Each parameter receives its own effective learning rate.

Parameters that change frequently receive smaller updates,

while parameters that change slowly receive larger updates.

2. **Moment Estimation**

Instead of relying only on the current gradient,

Adam also remembers

the history of previous gradients.

Consequently,

parameter updates become

more stable,

less noisy,

and significantly faster.

---

# 15.7.111 Why Ordinary Gradient Descent Can Oscillate

Suppose

we are minimizing a loss function

whose surface looks like

a narrow valley.

```text
Loss

^

|

|      \        /

|       \______/

|

+------------------------>

      Parameter
```

The gradient near the valley walls

is very steep.

As a result,

ordinary gradient descent

may repeatedly overshoot the minimum.

```text
Left

↓

Right

↓

Left

↓

Right

↓

Left
```

Instead of moving smoothly toward the optimum,

the optimizer oscillates.

Training becomes inefficient.

---

# 15.7.112 Using Previous Gradients

Humans rarely make decisions

based only on the most recent observation.

Instead,

we remember

what happened previously.

Adam applies the same idea.

Instead of considering

only the current gradient,

Adam computes

an exponentially weighted average

of previous gradients.

Consequently,

parameter updates become smoother.

Random fluctuations

are reduced.

Training becomes more stable.

---

# 15.7.113 The First Moment: Momentum

Suppose

the current gradient is

$$
g_t.
$$

Adam computes

the **first moment**

using

$$
\boxed{
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
}
$$

where

- $m_t$ is the running average of gradients,
- $g_t$ is the current gradient,
- $\beta_1$ controls how much previous gradients are remembered.

Typically,

$$
\beta_1=0.9.
$$

This means

approximately

90%

of the previous momentum

is retained,

while

10%

comes from the newest gradient.

---

# 15.7.114 Intuition Behind Momentum

Imagine

pushing a heavy cart.

If you push once,

the cart moves only slightly.

However,

if you continue pushing

in the same direction,

the cart gains momentum.

Likewise,

if gradients repeatedly point

toward

the same direction,

Adam gradually increases

its confidence

in that direction.

Instead of reacting strongly

to every small fluctuation,

the optimizer develops

a smoother trajectory.

Momentum therefore reduces

zigzag motion

during optimization.

---

# 15.7.115 The Second Moment: Measuring Gradient Magnitude

Momentum alone

does not solve every optimization problem.

Some parameters consistently receive

large gradients.

Others receive

very small gradients.

Ideally,

parameters with

large gradients

should take smaller steps,

while parameters with

small gradients

should take larger steps.

To accomplish this,

Adam estimates

the **second moment**.

$$
\boxed{
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
}
$$

where

- $v_t$ is the running average of squared gradients,
- $\beta_2$

is usually

$$
0.999.
$$

Notice that

the gradient is squared.

Consequently,

the second moment

measures

the typical magnitude

of recent gradients.

---

# 15.7.116 Why Squared Gradients Matter

Suppose

two parameters behave differently.

Parameter A

receives gradients

```text
0.001

0.002

0.001

0.002
```

Parameter B

receives gradients

```text
5.1

4.9

5.4

5.0
```

Clearly,

Parameter B

already changes rapidly.

It should therefore

move more cautiously.

Because Adam stores

the running average

of squared gradients,

Parameter B

automatically receives

smaller effective updates.

Meanwhile,

Parameter A

can still move

relatively quickly.

This adaptive behavior

is one of Adam's greatest strengths.

---

# 15.7.117 Bias Correction

At the beginning of training,

both

$m_t$

and

$v_t$

are initialized to zero.

Consequently,

their early estimates

are biased toward zero.

Adam corrects this problem

using

bias correction.

The corrected first moment is

$$
\boxed{
\hat m_t
=
\frac{m_t}
{1-\beta_1^t}
}
$$

Similarly,

the corrected second moment is

$$
\boxed{
\hat v_t
=
\frac{v_t}
{1-\beta_2^t}
}
$$

These corrections

are particularly important

during the first few optimization steps.

Without them,

parameter updates

would be systematically underestimated.

---

# 15.7.118 The Complete Adam Update Equation

Combining

momentum,

adaptive learning rates,

and bias correction

produces

the Adam update rule.

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\varepsilon}
}
$$

where

- $\theta$ represents the parameters,
- $\eta$ is the learning rate,
- $\hat m_t$ is the corrected first moment,
- $\hat v_t$ is the corrected second moment,
- $\varepsilon$ is a tiny constant preventing division by zero.

Typically,

$$
\varepsilon
=
10^{-8}.
$$

Although this equation appears more complicated than ordinary gradient descent,

its purpose is simple.

Each parameter

receives

its own adaptive step size

based on

its optimization history.

---

# 15.7.119 Why Adam Works So Well

Adam combines

three desirable properties.

First,

momentum smooths noisy gradients.

Second,

adaptive learning rates

allow each parameter

to learn at an appropriate speed.

Third,

bias correction

improves the accuracy

of early optimization steps.

Together,

these features make Adam

both

fast

and

stable.

Consequently,

it usually converges

more rapidly

than ordinary stochastic gradient descent.

---

# 15.7.120 Adam in PyTorch

Using Adam

requires only a few lines of code.

```python
import torch.optim as optim

optimizer = optim.Adam(

    model.parameters(),

    lr=1e-3

)
```

Here,

```python
model.parameters()
```

provides

all trainable parameters

inside the MEGNet model.

The learning rate

is set to

```python
1e-3
```

which corresponds to

$$
0.001.
$$

The optimizer automatically manages

all moment calculations,

bias corrections,

and parameter updates.

---

# 15.7.121 The Role of `optimizer.step()`

After gradients have been computed,

the optimizer performs

the actual update.

```python
loss.backward()

optimizer.step()
```

Internally,

Adam

1. reads every parameter gradient,
2. updates its first moment,
3. updates its second moment,
4. applies bias correction,
5. computes the adaptive step,
6. updates the parameter.

Once

`optimizer.step()`

finishes,

the neural network has officially

**learned**

from the current mini-batch.

---

# 15.7.122 Typical Hyperparameters

The default Adam hyperparameters

have proven effective

for many deep learning applications.

| Hyperparameter | Typical Value | Purpose |
|---------------|--------------:|---------|
| Learning rate ($\eta$) | 0.001 | Base step size |
| $\beta_1$ | 0.9 | Momentum coefficient |
| $\beta_2$ | 0.999 | Squared-gradient averaging |
| $\varepsilon$ | $10^{-8}$ | Numerical stability |

These defaults

are widely used

in

computer vision,

natural language processing,

graph neural networks,

and materials informatics.

---

# 15.7.123 Adam in Materials Informatics

Nearly every modern materials graph neural network

uses Adam

or one of its close variants.

Examples include

- CGCNN,
- MEGNet,
- M3GNet,
- ALIGNN,
- CHGNet.

The reason is straightforward.

Materials datasets

often contain

millions of parameters,

heterogeneous feature scales,

and highly nonlinear optimization landscapes.

Adam's adaptive learning strategy

handles these challenges

more effectively

than simple gradient descent.

---

## **MEGNet Perspective**

The Adam optimizer is the standard choice for training MEGNet because it extends gradient descent with momentum, adaptive learning rates, and bias correction. Rather than updating every parameter identically, Adam adjusts each parameter according to both its recent gradient direction and its historical gradient magnitude.

This adaptive behavior enables faster and more stable convergence when learning complex structure–property relationships from crystal graphs.

With the optimizer now fully understood, we are ready to combine all components of the learning process into a **complete MEGNet training loop**, integrating forward propagation, loss computation, backpropagation, optimization, validation, and checkpointing exactly as they are implemented in real materials informatics research.

# 15.7.124 The Complete MEGNet Training Pipeline

We have now studied every major component involved in training a MEGNet model.

Individually,

we understand

- the forward pass,
- the loss function,
- backpropagation,
- the chain rule,
- automatic differentiation,
- gradient descent,
- the Adam optimizer.

However,

these components do not operate independently.

They work together

inside a carefully organized training loop.

This training loop is executed

thousands

or even hundreds of thousands

of times

until the neural network converges.

Understanding this pipeline is one of the most important milestones in learning graph neural networks.

After this section,

you should be able to read

and understand

the training code used in most MEGNet research papers.

---

# 15.7.125 The Overall Workflow

Training proceeds through

a repeated sequence of operations.

```text
Dataset

↓

Mini-Batch

↓

Forward Pass

↓

Prediction

↓

Loss Calculation

↓

Backward Pass

↓

Gradient Computation

↓

Optimizer Step

↓

Updated Parameters

↓

Next Mini-Batch
```

This sequence

is repeated continuously

throughout training.

One complete pass through

the entire dataset

is called

an **epoch**.

---

# 15.7.126 Step 1 — Loading a Mini-Batch

Suppose

our training dataset contains

```text
120,000 crystals.
```

Instead of processing

all crystals simultaneously,

the dataset is divided

into small batches.

For example,

```text
Batch Size

↓

32
```

The first iteration processes

crystals

```text
1–32.
```

The second iteration processes

```text
33–64.
```

The process continues

until

every crystal

has been seen once.

Using mini-batches

reduces memory usage,

accelerates training,

and introduces beneficial stochasticity into optimization.

---

# 15.7.127 Step 2 — Resetting Previous Gradients

Before computing new gradients,

the gradients from

the previous iteration

must be cleared.

Otherwise,

PyTorch will accumulate gradients.

The first statement inside every training iteration is therefore

```python
optimizer.zero_grad()
```

This command sets

every stored gradient

to zero.

Conceptually,

```text
Previous Gradients

↓

Delete

↓

Ready for New Gradients
```

Failing to call

`zero_grad()`

is one of the most common mistakes made by beginners.

---

# 15.7.128 Step 3 — Forward Pass

The mini-batch

is passed through

the MEGNet model.

```python
prediction = model(batch)
```

Internally,

this executes

the complete forward pipeline.

```text
Crystal Graph

↓

Embedding

↓

Graph Network Blocks

↓

Readout

↓

Prediction Head

↓

Predicted Property
```

At this point,

the network has produced

its predictions,

but no learning has occurred yet.

---

# 15.7.129 Step 4 — Compute the Loss

The predictions

are compared

with

the true target values.

```python
loss = criterion(

    prediction,

    target

)
```

For regression problems,

the criterion is often

```python
nn.MSELoss()
```

or

```python
nn.L1Loss()
```

The result is

a single scalar,

for example,

```text
Loss = 0.018
```

This value summarizes

the prediction error

for the entire mini-batch.

---

# 15.7.130 Step 5 — Backpropagation

Next,

PyTorch computes

the gradients.

```python
loss.backward()
```

Internally,

Autograd

- traverses the computational graph,
- applies the chain rule,
- computes gradients,
- stores gradients

inside every trainable parameter.

At the end of this step,

every parameter possesses

its own gradient.

---

# 15.7.131 Step 6 — Parameter Update

Once gradients have been computed,
the optimizer updates
the network parameters.

```python
optimizer.step()
```

For Adam,
this involves

- momentum,
- adaptive learning rates,
- bias correction,
- parameter updates.

The model
has now learned
from
the current mini-batch.

---

# 15.7.132 Step 7 — Repeat for Every Mini-Batch

The previous six steps
are repeated
for
every mini-batch
inside the dataset.

```text
Batch 1

↓

Learn

↓

Batch 2

↓

Learn

↓

Batch 3

↓

Learn

↓

...

↓

Final Batch
```

When the final mini-batch finishes,
one epoch
has been completed.

---

# 15.7.133 The Complete Training Loop in PyTorch

The complete training loop
looks remarkably concise.

```python
for batch in train_loader:

    optimizer.zero_grad()

    prediction = model(batch)

    loss = criterion(

        prediction,

        batch.y

    )

    loss.backward()

    optimizer.step()
```

Although only a few lines long,
this code performs
millions of mathematical operations,
including

- graph construction,
- message passing,
- matrix multiplication,
- automatic differentiation,
- parameter optimization.

This simplicity
is one of the greatest strengths

of PyTorch.

---

# 15.7.134 Why Training Requires Many Epochs

One epoch
rarely produces
an accurate model.
Instead,
the network gradually improves.

For example,

| Epoch | Training Loss |
|------:|--------------:|
| 1 | 1.92 |
| 10 | 0.81 |
| 25 | 0.29 |
| 50 | 0.11 |
| 100 | 0.05 |

Each epoch
slightly improves
the learned parameters.
Eventually,
the loss stabilizes,
indicating that
the mdoel has largely converged.

---

# 15.7.135 Monitoring Training Progress

During training,
researchers usually record

- training loss,
- validation loss,
- learning rate,
- MAE,
- RMSE.

For example,

```text
Epoch 25

Training Loss

↓

0.083

Validation MAE

↓

0.034 eV/atom
```

Monitoring these metrics
helps determine
whether
the model is
learning effectively
or beginning to overfit.

---

# 15.7.136 Why Validation Is Necessary

A decreasing training loss
does **not**
guarantee
that the model generalizes well.
The network may simply memorize
the training data.
Therefore,
after each epoch,
the model is evaluated
on
a separate validation dataset.
Importantly,
during validation,
the parameters
are **not updated**.
The validation step measures
how well the model predicts
previously unseen crystals.

---

# 15.7.137 Saving the Best Model

Suppose
validation MAE
improves.

```text
Epoch 18

↓

Validation MAE

↓

0.031
```

This model
should be saved.

```python
torch.save(

    model.state_dict(),

    "best_model.pth"

)
```

Later,
if performance deteriorates,
the best-performing checkpoint
can be restored.
Model checkpointing
is standard practice
in almost every deep learning project.

---

# 15.7.138 The Full Training Lifecycle

The complete lifecycle
of a MEGNet training experiment
can now be summarized.

```text
Prepare Dataset

↓

Construct Graphs

↓

Create DataLoader

↓

Initialize Model

↓

Initialize Loss Function

↓

Initialize Optimizer

↓

Epoch Loop

↓

Mini-Batch Loop

↓

Forward Pass

↓

Loss

↓

Backward Pass

↓

Optimizer Step

↓

Validation

↓

Save Best Model

↓

Training Complete
```

This pipeline
forms the foundation
of nearly every graph neural network training workflow.

---

# 15.7.139 Connecting the Mathematics to the Code

At first glance,
the training loop appears deceptively simple.

```python
prediction = model(batch)

loss = criterion(prediction, batch.y)

loss.backward()

optimizer.step()
```

Behind these four statements,
PyTorch performs

- tensor operations,
- computational graph construction,
- chain rule differentiation,
- gradient accumulation,
- adaptive optimization.

Every mathematical concept
introduced throughout this chapter
is executed automatically
inside these few lines of code.
Understanding this correspondence
between theory and implementation
is one of the defining characteristics
of an expert machine learning practitioner.

---

## **MEGNet Perspective**

The complete training loop integrates every concept developed throughout this chapter into a single, coherent workflow. Each mini-batch undergoes forward propagation, loss evaluation, backpropagation, and parameter optimization before the process repeats for the next batch. Over many epochs, these small parameter updates gradually transform randomly initialized weights into chemically meaningful representations capable of accurately predicting materials properties.

At this point, you understand the complete learning mechanism of MEGNet—from raw crystal graphs to optimized neural network parameters.

The next major section moves beyond optimization itself and focuses on **evaluating trained models**, introducing validation strategies, learning curves, overfitting detection, early stopping, checkpointing, and best practices for ensuring reliable and reproducible performance in materials informatics research.

# 15.8 Model Evaluation, Validation, and Generalization

Up to this point,
our objective has been
to **train** a MEGNet model.
Training, however,
is only half of the machine learning pipeline.
A model that performs extremely well on the training dataset
is **not necessarily** a useful scientific model.
The true objective of machine learning is not
to memorize the training data,
but to discover underlying physical relationships that can be applied to **previously unseen materials**.
Suppose we train a MEGNet model
using
120,000 crystal structures
from the Materials Project.
After training,
the model predicts every training crystal
almost perfectly.
Does this mean
the model has learned materials science?
Not necessarily.
The model may simply have memorized
the training dataset.
To determine whether the model has genuinely learned
the relationship between crystal structure and material properties,
we must evaluate it
using crystals that it has **never encountered before**.
This process is called **model validation**.

---

# 15.8.1 Why Training Accuracy Is Misleading

Imagine preparing for an examination.
A student receives
100 practice questions.
Instead of understanding the concepts,
the student memorizes every answer.
On the examination,
the teacher asks
those exact same questions.
The student scores

100%.

Now consider a different examination.
The questions are new,
although they test the same concepts.
The memorizing student now performs poorly.
Clearly,
the student did not truly understand the subject.
Neural networks behave in exactly the same way
If a model merely memorizes
the training crystals,
it may achieve an extremely small training loss,
yet completely fai
when predicting new materials.
Therefore,
training performance alon
is not a reliable measure
of scientific usefulness.

---

# 15.8.2 The Goal of Generalization

The fundamental goal of machine learning
is called
**generalization**.
Generalization means
the ability of a model
to make accurate predictions
for data that were **not used during training**.
For MEGNet,
generalization means
predicting the properties of
new crystal structures,
new chemical compositions,
or newly synthesized materials.
A model with good generalization
has learned
the underlying structure–property relationship,
rather than simply memorizing examples.

---

# 15.8.3 The Three Dataset Splits

To measure generalization objectively,
the dataset is divided
into three independent subsets.

```text
Complete Dataset

↓

Training Set

↓

Validation Set

↓

Test Set
```

Each subset has
a different purpose.

---

### Training Set

The training set
is used
to update
the neural network parameters.
Only this subset
participates in
backpropagation
and
optimization.

---

### Validation Set

The validation set
is used
to monitor
training progress.
It helps answer questions such as

- Is the model improving?
- Is overfitting beginning?
- Should training stop?
- Which hyperparameters work best?

Importantly,
validation data
are **never used**
to update parameters.

---

### Test Set

The test set
is used
only once,
after all training decisions
have been completed.
Its purpose is
to provide
an unbiased estimate
of the final model performance.
The test set
should remain untouched
throughout model development.

---

# 15.8.4 A Typical Dataset Split

Suppose
our dataset contains
100,000 crystals.
A common split is

| Dataset | Percentage | Samples |
|---------|-----------:|---------:|
| Training | 80% | 80,000 |
| Validation | 10% | 10,000 |
| Test | 10% | 10,000 |

Other ratios,
such as

70–15–15
or

90–5–5,

are also common.
The exact choice
depends on
the dataset size
and
the research objective.

---

# 15.8.5 Why Data Leakage Must Be Avoided

One of the most serious mistakes
in machine learning
is

**data leakage**.

Data leakage occurs
when information from
the validation
or
test dataset

accidentally influences

the training process.

For example,

suppose

a crystal appears

in both

the training set

and

the test set.

The model has already seen

this crystal during training.

Naturally,

its prediction will appear

extremely accurate.

However,

this apparent accuracy

is misleading.

The model has not generalized.

It has merely remembered

the crystal.

Therefore,

the three datasets

must remain

strictly independent.

---

# 15.8.6 Training Versus Validation

The workflow

during an epoch

can be summarized as

```text
Training Dataset

↓

Forward Pass

↓

Loss

↓

Backward Pass

↓

Optimizer Step
```

followed by

```text
Validation Dataset

↓

Forward Pass

↓

Prediction

↓

Performance Metrics

↓

No Parameter Updates
```

Notice the crucial difference.

Training changes

the model.

Validation only measures

the model.

---

# 15.8.7 Why Validation Uses `model.eval()`

PyTorch provides

two operating modes

for neural networks.

```python
model.train()
```

and

```python
model.eval()
```

During training,

certain layers,

such as

Dropout

and

Batch Normalization,

behave differently.

Before evaluating

the validation dataset,

the model should therefore switch

to evaluation mode.

```python
model.eval()
```

This ensures

that predictions

are consistent

and reproducible.

---

# 15.8.8 Disabling Gradient Computation

Validation does not require

backpropagation.

Consequently,

gradient computation

should be disabled.

PyTorch provides

the following context manager.

```python
with torch.no_grad():

    prediction = model(batch)
```

Using

`torch.no_grad()`

reduces

memory usage,

improves inference speed,

and prevents unnecessary construction

of computational graphs.

This is standard practice

during validation

and testing.

---

# 15.8.9 A Typical Validation Loop

The validation procedure

resembles the training loop,

but with two important differences.

- No gradients are computed.
- No optimizer updates occur.

```python
model.eval()

validation_loss = 0.0

with torch.no_grad():

    for batch in validation_loader:

        prediction = model(batch)

        loss = criterion(

            prediction,

            batch.y

        )

        validation_loss += loss.item()
```

After processing

every validation batch,

the average validation loss

is computed.

This value

is then recorded

for later analysis.

---

# 15.8.10 Why Validation Is Performed Every Epoch

Suppose

training lasts

100 epochs.

After each epoch,

we evaluate

the validation dataset.

This produces

a sequence of validation losses.

For example,

| Epoch | Validation Loss |
|------:|----------------:|
| 1 | 0.84 |
| 10 | 0.29 |
| 25 | 0.12 |
| 40 | 0.08 |
| 60 | 0.07 |
| 80 | 0.09 |
| 100 | 0.13 |

Initially,

the validation loss decreases,

indicating

improving generalization.

Eventually,

however,

the validation loss begins increasing.

This phenomenon

is one of the clearest indicators

of **overfitting**,

which we will study in the next section.

---

## **MEGNet Perspective**

Validation is the bridge between optimization and scientific reliability.

While training teaches the MEGNet model using known crystal structures, validation continuously measures whether the learned representations generalize to unseen materials.

A well-designed validation strategy ensures that improvements in performance reflect genuine learning of crystal chemistry rather than memorization of the training dataset.

In the next section, we will study **overfitting and underfitting**, learning how to recognize these behaviors from training and validation curves and how modern deep learning techniques prevent them during MEGNet training.

