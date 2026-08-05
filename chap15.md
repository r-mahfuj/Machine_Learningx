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

## 15.8.11 Overfitting: When MEGNet Memorizes Instead of Learning (Part 1)

One of the most important challenges in training any machine learning model is not making the model powerful enough—it is preventing the model from becoming **too powerful**.

This statement often surprises beginners.

Throughout this book, we have continuously increased the complexity of our models.

We moved from

* Linear Regression,
* Decision Trees,
* Random Forests,
* Gradient Boosting,
* Neural Networks,

and finally arrived at graph neural networks such as MEGNet.

Each successive model is capable of learning increasingly complex relationships.

It would therefore seem natural to assume that a more powerful model must always produce better predictions.

Unfortunately, this is not true.

A model with excessive capacity can begin to learn not only the true physical relationship between crystal structure and material properties but also the accidental details, random fluctuations, measurement noise, and peculiarities of the training dataset.

When this occurs, the model appears to perform exceptionally well during training while performing poorly on previously unseen materials.

This phenomenon is called **overfitting**.

Overfitting is one of the central problems in modern machine learning and is especially important in materials informatics because available datasets are often much smaller than those used in computer vision or natural language processing.

---

### What Does "Overfitting" Actually Mean?

Suppose we wish to predict the formation energy of crystalline materials.

Our training dataset contains

```text
80,000 crystal structures
```

Each crystal has

* atomic positions,
* lattice parameters,
* chemical composition,
* neighbor information,
* experimentally or computationally determined formation energy.

During training,

MEGNet attempts to discover a function

$$
f(G;\theta)
$$

that maps each crystal graph

$$
G
$$

to its corresponding property.

Ideally,

the learned function should approximate the true physical relationship

$$
f_{\text{physics}}(G).
$$

Instead,

an overfitted model learns something closer to

$$
f_{\text{training}}(G),
$$

which contains not only genuine physical trends but also random characteristics unique to the training dataset.

The model therefore becomes extremely specialized to the training data.

---

### An Everyday Analogy

Imagine two students preparing for an advanced materials science examination.

The first student studies

* thermodynamics,
* crystallography,
* diffusion,
* phase diagrams,
* electronic structure.

The second student memorizes

every question from the previous year's examination.

Suppose the instructor repeats the same examination.

The second student scores

100%.

However,

if the instructor writes a completely new examination,

the second student performs poorly.

The first student,

although perhaps not perfect,

performs consistently well because genuine understanding has been developed.

Machine learning models behave in exactly the same manner.

Learning the underlying scientific principles corresponds to **generalization**.

Memorizing individual examples corresponds to **overfitting**.

---

### Overfitting from the Perspective of Materials Science

Consider two crystals.

The first crystal belongs to the training dataset.

```text
LiFePO₄
```

The second crystal is newly synthesized.

```text
LiMnPO₄
```

Although these materials are chemically related,

the second crystal was never observed during training.

A properly trained MEGNet model should recognize that

both crystals share

* similar crystal topology,
* similar bonding environments,
* similar coordination geometries,

and therefore produce a physically reasonable prediction.

An overfitted model,

however,

may rely excessively on the exact atomic configurations encountered during training.

Instead of recognizing chemical similarity,

it effectively asks

> "Have I seen this exact crystal before?"

If the answer is no,

prediction quality deteriorates rapidly.

---

### Why Deep Neural Networks Are Especially Vulnerable

Overfitting becomes increasingly severe as model capacity increases.

Capacity refers to the ability of a model to represent complicated mathematical functions.

A simple linear regression model contains relatively few parameters.

A decision tree contains more.

A random forest contains thousands.

A modern graph neural network may contain

```text
500,000

↓

2,000,000

↓

10,000,000
```

trainable parameters.

Each parameter increases the flexibility of the model.

This flexibility is extremely valuable because it allows MEGNet to learn highly nonlinear relationships between crystal structures and material properties.

However,

the same flexibility also enables the network to memorize the training dataset.

Consequently,

the very property that makes deep learning successful also creates its greatest weakness.

---

### Capacity and Generalization

To understand this idea more formally,

suppose our neural network contains

$$
P
$$

trainable parameters.

As

$$
P
$$

increases,

the family of functions that the network can represent also expands.

For very small

$$
P,
$$

the model may be incapable of representing the true physical relationship.

For extremely large

$$
P,
$$

the network can represent almost any mapping,

including mappings that simply memorize every training example.

Therefore,

increasing model complexity initially improves prediction accuracy,

but eventually begins to reduce the model's ability to generalize.

This trade-off between

* model complexity,
* training accuracy,
* and generalization

is one of the central ideas in statistical learning theory.

---

### Why Overfitting Is Dangerous in Scientific Machine Learning

In many commercial applications,

a small decrease in prediction accuracy may simply reduce user satisfaction.

In scientific research,

the consequences can be much more serious.

Suppose a materials discovery project screens

one million hypothetical compounds.

The top

100

predicted candidates are selected for expensive DFT calculations.

If the neural network is overfitted,

many of these predictions may be incorrect.

Researchers may then spend

weeks or months

performing simulations on materials that are actually poor candidates.

Even worse,

truly promising compounds may be discarded because the model underestimated their properties.

Thus,

overfitting wastes

* computational resources,
* experimental effort,
* research funding,
* and scientific time.

For this reason,

controlling overfitting is not merely a machine learning concern—it is essential for conducting reliable computational materials research.

---

### The Hidden Nature of Overfitting

One reason overfitting is particularly dangerous is that it is often invisible during training.

Imagine observing only the training loss.

It decreases steadily.

```text
Epoch 1

↓

1.82

Epoch 10

↓

0.47

Epoch 30

↓

0.09

Epoch 80

↓

0.006
```

Everything appears excellent.

The optimizer is working.

The network is learning.

Training seems highly successful.

However,

suppose we now examine the validation loss.

```text
Epoch 1

↓

1.95

Epoch 10

↓

0.62

Epoch 30

↓

0.21

Epoch 50

↓

0.19

Epoch 80

↓

0.44
```

A completely different story emerges.

Although training loss continues decreasing,

validation loss begins increasing.

The model is no longer learning better physical representations.

Instead,

it is memorizing increasingly specific details of the training dataset.

This divergence between training and validation performance is the characteristic signature of overfitting.

---

In the next part, we will examine **how overfitting develops mathematically**, introduce the concepts of **empirical risk** and **generalization error**, and analyze why overfitting is particularly challenging for graph neural networks such as MEGNet trained on materials datasets.

## 15.8.11 Overfitting: When MEGNet Memorizes Instead of Learning (Part 2)

In the previous section, we developed an intuitive understanding of overfitting. We saw that a neural network can gradually shift from learning genuine physical relationships to memorizing the individual examples contained within the training dataset.

While this intuition is important, modern machine learning studies overfitting from a much more rigorous perspective. To understand why overfitting occurs and how it can be detected, we must first distinguish between two different types of prediction error.

---

### Training Error and Generalization Error

During training, the optimizer repeatedly minimizes the loss computed on the training dataset.

Suppose the training dataset is

$$
D_{train}
=========

{(G_i,y_i)}_{i=1}^{N}.
$$

The average loss over this dataset is called the **training error** or **empirical risk**.

Mathematically,

$$
R_{train}
=========

\frac{1}{N}
\sum_{i=1}^{N}
L
\left(
f(G_i;\theta),
y_i
\right),
$$

where

* $L$ is the loss function,
* $f(G_i;\theta)$ is the prediction of the neural network,
* $y_i$ is the true target property.

This quantity is exactly what the optimizer minimizes.

Notice an important point.

The optimizer never sees any data outside the training dataset.

Therefore, from the optimizer's perspective,

reducing

$$
R_{train}
$$

is the only objective.

Unfortunately,

our actual scientific objective is different.

---

### The Quantity We Really Care About

Suppose another crystal exists,

$$
G_{new},
$$

which was never included in the training data.

The prediction error for this crystal contributes to what is called the **generalization error**.

Ideally, we would like to evaluate the model over **every possible crystal that could exist**.

Conceptually,

the desired objective is

$$
R_{true}
========

\mathbb{E}
\left[
L(f(G;\theta),y)
\right],
$$

where the expectation is taken over the entire population of possible materials.

This quantity is sometimes called

* expected risk,
* population risk,
* true error.

Unlike the training error,

it cannot be computed directly,

because we do not possess data for every possible crystal in nature.

Instead,

we estimate it using validation and test datasets.

---

### Why These Two Errors Are Different

If the training dataset were infinitely large and perfectly representative of all materials,

then

$$
R_{train}
\approx
R_{true}.
$$

In practice,

materials datasets are finite.

Consequently,

the optimizer only learns from a limited sample.

As the neural network becomes increasingly flexible,

it begins fitting characteristics that exist only inside that sample.

The result is

$$
R_{train}
\downarrow
$$

while

$$
R_{true}
\uparrow.
$$

This divergence is precisely the mathematical definition of overfitting.

---

### The Generalization Gap

The difference between training performance and unseen-data performance is called the **generalization gap**.

It is defined conceptually as

$$
\text{Generalization Gap}
=

R_{true}

-R_{train}.
$$

A small generalization gap indicates that the model behaves similarly on both training data and unseen data.

A large generalization gap indicates that the model has memorized the training dataset.

For example,

| Model   | Training MAE | Validation MAE | Generalization |
| ------- | -----------: | -------------: | -------------- |
| Model A |     0.025 eV |       0.031 eV | Excellent      |
| Model B |     0.004 eV |       0.182 eV | Poor           |

Although Model B achieves a much lower training error,

its validation performance is dramatically worse.

The larger gap immediately suggests severe overfitting.

---

### Why Graph Neural Networks Can Overfit

Graph neural networks possess enormous representational power.

A MEGNet model simultaneously learns

* atomic embeddings,
* bond representations,
* local environments,
* long-range interactions,
* graph-level representations,
* nonlinear mappings between structure and property.

Each Graph Network block introduces additional trainable parameters.

Each multilayer perceptron introduces even more.

As additional blocks are stacked together,

the network becomes increasingly expressive.

This expressiveness is beneficial because materials exhibit highly nonlinear behavior.

However,

it also enables the network to reproduce extremely complicated patterns that may exist only within the training dataset.

Consequently,

a sufficiently large GNN can almost perfectly reproduce the training data without discovering the true underlying physical relationship.

---

### Noise Exists in Materials Data

Another important source of overfitting is **noise**.

Real materials datasets are rarely perfect.

Even computational databases contain uncertainties arising from

* DFT approximations,
* exchange-correlation functionals,
* convergence tolerances,
* numerical precision,
* structural relaxation procedures.

Experimental datasets contain additional uncertainties,

including

* measurement error,
* impurities,
* synthesis variability,
* instrument limitations.

An ideal machine learning model should learn the true physical trend while ignoring random noise.

An overfitted model does the opposite.

It begins treating noise as though it were a meaningful scientific signal.

As training continues,

the network becomes increasingly specialized to accidental variations that have no physical significance.

---

### A Visual Interpretation

Imagine plotting the true relationship between a structural descriptor and formation energy.

The true relationship is smooth.

The measured data fluctuate slightly because of noise.

A well-generalized model follows the overall trend.

An overfitted model attempts to pass through every individual point.

Conceptually,

```text
True Physical Trend

──────────────

Measured Samples

•   •      •    •

  •     •      •

Well-Generalized Model

~~~~~~~~~~~~~~~

Overfitted Model

/\/\/\/\/\/\/\/\
```

The overfitted curve achieves a lower training error,

yet it provides a poorer approximation of the underlying physical law.

This illustrates one of the most fundamental principles of scientific machine learning:

**The model with the lowest training error is not necessarily the model that best represents reality.**

---

### Overfitting Is Usually Gradual

Many beginners imagine overfitting as a sudden event.

In reality,

it develops progressively.

During the early stages of training,

the network learns broad physical relationships.

For example,

it may first recognize that

* bond length influences bond strength,
* coordination environment influences stability,
* electronegativity affects charge distribution.

These relationships improve both training and validation performance.

As optimization continues,

the remaining errors become increasingly difficult to reduce.

Instead of discovering new physical principles,

the optimizer begins exploiting peculiarities unique to individual training samples.

At this point,

training loss continues to decrease,

but validation performance no longer improves.

Eventually,

validation performance begins to deteriorate.

This marks the transition from **learning** to **memorization**.

---

### Why More Training Is Not Always Better

A common misconception is that additional training must always improve the model.

This assumption is incorrect.

Suppose we continue training long after the model has reached its optimal generalization point.

The optimizer continues minimizing the training loss,

but every additional epoch increases memorization.

Consequently,

the final model may actually perform worse than an earlier checkpoint.

For this reason,

modern deep learning does not simply train for as many epochs as possible.

Instead,

training is carefully monitored,

and the model is often stopped before severe overfitting develops.

This strategy is known as **early stopping**, which we will examine later in this chapter.

---

At this stage, we understand **what overfitting is and why it occurs**. However, recognizing overfitting in practice requires careful analysis of the model's behavior during training.

In the next section, we will study **learning curves**, examining how training and validation losses evolve over time and how these curves provide one of the most powerful diagnostic tools for identifying overfitting in MEGNet and other graph neural networks.

## 15.8.11 Overfitting: When MEGNet Memorizes Instead of Learning (Part 3)

Up to this point, we have explained overfitting conceptually and mathematically. However, when training a real MEGNet model, researchers do not directly observe the internal parameters or the mathematical function being learned.

Instead, they observe **training statistics**.

These statistics provide indirect evidence about how the neural network is learning.

Among all diagnostic tools used in deep learning, none is more informative than the **learning curve**.

A properly interpreted learning curve can reveal

* whether the optimizer is functioning correctly,
* whether the learning rate is appropriate,
* whether the model is underfitting,
* whether the model is overfitting,
* whether additional training is beneficial,
* whether early stopping should be applied.

For this reason, experienced researchers examine learning curves after nearly every training experiment.

---

### What Is a Learning Curve?

A learning curve is simply a graph that shows how model performance changes as training progresses.

The horizontal axis represents the number of training epochs.

The vertical axis usually represents

* loss,
* MAE,
* RMSE,
* or another evaluation metric.

The two most important curves are

* the **training curve**, and
* the **validation curve**.

Conceptually,

```text
Performance Metric

^

|

|  Training Curve

|

|

|

|  Validation Curve

|

+------------------------------------>

             Epoch
```

Although this figure appears simple,

it summarizes everything that happens during training.

---

### How the Curves Are Produced

Recall the complete training loop discussed earlier.

For every epoch,

the following sequence occurs.

1. Train on every mini-batch.
2. Compute the average training loss.
3. Evaluate the validation dataset.
4. Compute the validation loss.
5. Store both values.

After repeating this process for many epochs,

we obtain two sequences.

For example,

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |          2.18 |            2.25 |
|     2 |          1.74 |            1.82 |
|     3 |          1.39 |            1.51 |
|   ... |           ... |             ... |
|   100 |          0.06 |            0.08 |

Plotting these values produces the learning curves.

---

### The Ideal Learning Curve

In an ideal training process,

both curves decrease together.

Conceptually,

```text
Loss

^

|\
| \
|  \
|   \
|    \
|     \
|      \____

+------------------------------>

          Epoch
```

Initially,

both losses decrease rapidly because the network learns the most important physical relationships.

As training continues,

the rate of improvement slows.

Eventually,

both curves stabilize.

When

* the training loss is small,
* the validation loss is also small,
* and the two curves remain close,

the model has likely learned useful representations without memorizing the data.

---

### What Happens During Early Training?

At the beginning of optimization,

the network parameters are random.

Consequently,

predictions are essentially random.

Suppose the task is predicting formation energy.

The first predictions may differ from the true values by several electron volts.

The initial loss is therefore very large.

After a few epochs,

the optimizer begins discovering simple physical relationships.

For example,

the network may first learn

* that neighboring atoms influence each other,
* that atomic species matter,
* that bond environments affect stability.

These broad trends significantly reduce both training and validation loss.

This phase corresponds to genuine learning.

---

### Why Validation Improves Initially

An important observation is that

during the early stages,

improvements on the training dataset also improve predictions on unseen crystals.

This occurs because

the network is learning general physical principles rather than individual examples.

For instance,

the model may discover that

higher coordination numbers often stabilize crystal structures.

This relationship applies not only to crystals in the training dataset,

but also to entirely new materials.

Consequently,

validation performance improves alongside training performance.

---

### The Turning Point

Eventually,

the optimizer reaches a critical stage.

Most large-scale physical relationships have already been learned.

The remaining training errors become increasingly difficult to reduce.

Instead of discovering additional physical laws,

the optimizer begins fitting subtle details that exist only within the training dataset.

At this point,

the learning curves begin to separate.

The training loss continues decreasing,

while the validation loss decreases much more slowly.

This divergence marks the beginning of overfitting.

---

### The Signature of Overfitting

The characteristic pattern of overfitting appears as

```text
Loss

^

|\
| \
|  \
|   \
|    \
|     \____________________

|

|        \

|         \

|          \______

|                 \

|                  \

+---------------------------------->

                Epoch
```

The lower curve represents the training loss.

The upper curve represents the validation loss.

Notice three distinct phases.

**Phase 1**

Both curves decrease together.

The model is learning.

---

**Phase 2**

Training loss continues decreasing.

Validation loss decreases only slightly.

The model is approaching its optimal generalization point.

---

**Phase 3**

Training loss continues decreasing.

Validation loss begins increasing.

The model has entered the overfitting regime.

Additional optimization no longer improves scientific performance.

Instead,

the network memorizes increasingly specific details of the training dataset.

---

### Why Training Loss Never Reveals Overfitting

Suppose we examine only the training loss.

It might appear as

| Epoch | Training Loss |
| ----: | ------------: |
|    10 |          0.72 |
|    20 |          0.38 |
|    40 |          0.15 |
|    80 |          0.05 |
|   120 |          0.01 |

Everything looks excellent.

Every epoch improves performance.

Nothing appears problematic.

However,

training loss alone provides no information about generalization.

A continuously decreasing training loss is expected,

even for a severely overfitted model.

Therefore,

training loss should never be interpreted in isolation.

---

### Why Validation Loss Is the Key Metric

Now consider the validation loss.

| Epoch | Validation Loss |
| ----: | --------------: |
|    10 |            0.81 |
|    20 |            0.45 |
|    40 |            0.19 |
|    60 |            0.17 |
|    80 |            0.20 |
|   120 |            0.34 |

This table immediately reveals the problem.

The lowest validation loss occurs near

Epoch 60.

After this point,

the network continues improving on the training dataset,

yet performs progressively worse on unseen materials.

Consequently,

the model obtained at

Epoch 120

is actually inferior to the model obtained at

Epoch 60,

despite having a much smaller training loss.

This illustrates one of the most important lessons in deep learning:

> **The best model is determined by validation performance, not by training performance.**

---

### The Optimal Stopping Point

If we overlay the two curves,

the optimal model corresponds approximately to the point where

the validation loss reaches its minimum.

Conceptually,

```text
Loss

^

|\
| \
|  \
|   \
|    \
|     \_________________

|

|      \

|       \_____

|             \

|              \

+--------------------●---------------->

                     Best Epoch
```

The highlighted point represents the epoch at which the model generalizes most effectively.

Training beyond this point generally increases memorization without improving predictive capability.

---

### Why This Matters for Materials Informatics

In materials science,

the objective is rarely to reproduce known database values.

Instead,

we hope to predict the properties of

* hypothetical compounds,
* newly synthesized crystals,
* unexplored chemical compositions,
* or entirely new classes of materials.

These systems resemble the validation and test datasets—not the training dataset.

Therefore,

the model selected for publication or deployment should always correspond to the epoch with the best validation performance.

This principle is followed in virtually every high-quality MEGNet, CGCNN, ALIGNN, M3GNet, and CHGNet study.

---

At this point, we can recognize overfitting by examining learning curves. The next question naturally follows:

**Why do some models overfit much more easily than others?**

To answer this, we will study the factors that control overfitting, including **model capacity, dataset size, noise, feature dimensionality, graph depth, and parameter count**, and explain why graph neural networks for materials science require particularly careful regularization.

## 15.8.11 Overfitting: When MEGNet Memorizes Instead of Learning (Part 4)

In the previous section, we learned how to **detect** overfitting by examining learning curves. However, detecting overfitting is only the first step. A much deeper question remains:

> **Why do some neural networks overfit while others generalize well?**

This question has occupied machine learning researchers for decades.

Although there is no single universal answer, experience and statistical learning theory show that overfitting is primarily controlled by the interaction between three factors:

1. **Model capacity**
2. **Dataset complexity and size**
3. **Training duration**

The relationship between these factors determines whether a model successfully learns the underlying physical laws or merely memorizes the available data.

---

### Model Capacity

The first and perhaps most important factor is **model capacity**.

Capacity refers to the ability of a model to represent different mathematical functions.

A model with low capacity can represent only relatively simple relationships.

A model with high capacity can represent extremely complicated nonlinear mappings.

For example,

a linear regression model has very low capacity because it can only learn relationships of the form

$$
y = wx + b.
$$

Regardless of how long it is trained, it cannot represent highly nonlinear behavior.

A decision tree has greater capacity because it partitions the feature space into many regions.

Random forests and gradient boosting increase capacity even further by combining multiple decision trees.

Deep neural networks—and especially graph neural networks—possess extraordinarily high capacity.

A modern MEGNet model may contain hundreds of thousands or even millions of trainable parameters.

Such a model can approximate highly complex functions relating atomic environments to material properties.

This expressive power is precisely what makes deep learning successful.

However, it also introduces the possibility of memorization.

---

### A Simple Analogy

Imagine giving two artists the same assignment.

The first artist receives only a pencil.

The second artist receives a complete professional digital drawing studio.

The first artist has limited expressive ability.

The second artist can reproduce even the smallest details.

If both artists are asked to copy a noisy photograph,

the first artist will naturally capture only the main shapes and important structures.

The second artist can reproduce every shadow, scratch, dust particle, and camera artifact.

Machine learning models behave similarly.

A low-capacity model captures only the dominant physical relationships.

A high-capacity model is capable of reproducing every detail—including noise.

---

### Capacity Is Not the Enemy

At this point, it may appear that we should always use smaller models.

This conclusion is incorrect.

A model with insufficient capacity cannot represent the true relationship between crystal structure and material properties.

Suppose the true mapping between crystal graphs and formation energy is highly nonlinear.

A simple linear regression model cannot capture

* bond-angle effects,
* coordination environments,
* long-range interactions,
* many-body chemistry,
* nonlinear electronic behavior.

Its predictions will remain poor regardless of how much data are available.

Thus,

too little capacity produces **underfitting**.

Too much capacity may produce **overfitting**.

The objective is therefore **not** to minimize capacity.

Instead, we seek an appropriate balance between flexibility and generalization.

---

### Dataset Size

Capacity alone does not determine whether overfitting occurs.

Dataset size is equally important.

Consider two training scenarios.

#### Scenario A

A MEGNet model with one million parameters is trained using

```text
1,500 crystal structures.
```

#### Scenario B

The same model is trained using

```text
1,500,000 crystal structures.
```

The architecture is identical.

The optimizer is identical.

The learning rate is identical.

Yet the likelihood of overfitting is dramatically different.

With only

1,500 samples,

the network can easily memorize individual crystals.

With

1.5 million samples,

memorization becomes much more difficult because the dataset contains vastly greater chemical diversity.

In general,

larger datasets encourage the network to learn broad physical principles rather than individual examples.

This is one of the primary reasons why large-scale databases such as the Materials Project have revolutionized materials informatics.

---

### Why Materials Science Suffers More Than Computer Vision

Researchers in computer vision often train neural networks using datasets containing

* ten million,
* fifty million,
* or even hundreds of millions

of images.

By comparison,

materials science datasets are usually much smaller.

Typical public datasets contain

| Dataset           |                    Approximate Size |
| ----------------- | ----------------------------------: |
| QM9               |                  ~134,000 molecules |
| Materials Project |         ~150,000 crystal structures |
| OQMD              |             ~1,000,000 calculations |
| JARVIS            | Hundreds of thousands of structures |

Although these databases are impressive by materials science standards,

they are still modest compared with datasets used in many other machine learning domains.

Consequently,

graph neural networks for materials science operate in a relatively **data-limited regime**.

This makes overfitting a much more significant concern.

---

### Dataset Diversity Matters More Than Dataset Size

A large dataset is not necessarily a diverse dataset.

Imagine training a MEGNet model using one million crystal structures.

At first,

this sounds ideal.

However,

suppose

95%

of those structures belong to only a few closely related oxide families.

The effective diversity of the dataset is much lower than the raw sample count suggests.

Now consider a smaller dataset containing

100,000 structures

carefully selected from many different chemistries,

including

* oxides,
* nitrides,
* carbides,
* sulfides,
* halides,
* phosphides,
* intermetallic compounds,
* layered materials,
* perovskites,
* spinels,
* Heusler alloys.

Although the second dataset is numerically smaller,

it may provide better generalization because it exposes the neural network to a much broader range of atomic environments.

For materials informatics,

chemical diversity is often as important as sample size.

---

### Noise and Label Quality

Another major contributor to overfitting is imperfect target data.

Suppose two datasets are available.

Dataset A

contains formation energies computed using carefully converged DFT calculations.

Dataset B

contains values obtained using inconsistent computational settings.

Even though both datasets contain the same number of structures,

Dataset B includes greater numerical uncertainty.

An expressive neural network may begin learning these inconsistencies.

Instead of approximating the true physical relationship,

it approximates the noise present in the labels.

Consequently,

better labels often produce larger improvements than more sophisticated neural network architectures.

This principle is particularly important in scientific machine learning:

> **A neural network cannot learn information that is absent from the data.**

Likewise,

it cannot distinguish genuine physical patterns from systematic errors unless the dataset itself is reliable.

---

### Feature Complexity

Graph neural networks receive much richer information than conventional machine learning models.

A MEGNet graph contains

* node features,
* edge features,
* global state variables,
* connectivity information,
* neighborhood relationships.

As additional descriptors are introduced,

the feature space becomes increasingly high-dimensional.

High-dimensional feature spaces provide the network with greater flexibility,

but they also increase the possibility of learning accidental correlations.

For example,

suppose one particular coordination environment appears only in a small subset of training crystals.

A large neural network may incorrectly conclude that this environment uniquely determines the target property,

even though the apparent relationship arose purely by chance.

Such spurious correlations often contribute to overfitting.

---

### Graph Depth

Another important factor specific to graph neural networks is **graph depth**.

Increasing the number of message-passing layers allows each atom to gather information from progressively more distant neighbors.

A shallow network may consider only local bonding environments.

A deeper network incorporates increasingly long-range structural information.

Initially,

this improves predictive performance.

However,

beyond a certain depth,

additional layers may begin fitting increasingly subtle details specific to the training dataset.

Moreover,

very deep graph neural networks can suffer from additional problems such as

* oversmoothing,
* oversquashing,
* optimization instability.

Therefore,

adding more graph layers does not guarantee better generalization.

The optimal depth depends on both the dataset and the target property.

---

### The Balance Between Capacity and Data

We can now summarize the central principle governing overfitting.

A neural network should possess **enough capacity to represent the underlying physics**, but **not so much capacity that it memorizes the training data**.

Whether this balance is achieved depends on

* the number of trainable parameters,
* the quantity of available data,
* the diversity of chemical space,
* the quality of the target labels,
* the architecture of the graph neural network,
* and the duration of training.

Finding this balance is one of the primary responsibilities of a machine learning researcher.

It is also one of the reasons why successful materials informatics models require careful experimental design rather than simply increasing network size.

---

At this stage, we understand **what causes overfitting**. The next logical question is:

**How can we prevent it?**

In the following sections, we will study the major regularization techniques used in modern graph neural networks, including **weight decay (L2 regularization), dropout, early stopping, data augmentation (where applicable), and model checkpointing**, explaining both their mathematical foundations and their practical implementation in MEGNet.

## 15.8.12 Preventing Overfitting: Regularization in MEGNet (Part 2)

### L2 Regularization (Weight Decay)

Among all regularization techniques developed for machine learning, **L2 regularization** is one of the oldest, most thoroughly studied, and most widely used.

If you examine the implementation details of modern graph neural networks—including

* MEGNet,
* CGCNN,
* M3GNet,
* ALIGNN,
* CHGNet,

you will almost always find some form of weight decay being used during optimization.

Despite its widespread use, the underlying idea is remarkably simple.

Instead of allowing the optimizer to choose arbitrarily large parameter values, we encourage it to prefer **smaller and smoother parameter values** whenever possible.

The important phrase here is **"whenever possible."**

Weight decay does **not** force every parameter to become zero.

Instead, it asks the optimizer:

> *"If two different parameter values produce nearly identical prediction accuracy, prefer the smaller one."*

This seemingly small modification has profound consequences for generalization.

---

### Why Large Weights Lead to Complex Functions

To understand weight decay, let us first examine the role of a weight in a neural network.

Consider a single neuron

$$
z = wx + b.
$$

Suppose

$$
w = 1.
$$

A change of

$$
\Delta x = 0.1
$$

produces

$$
\Delta z = 0.1.
$$

Now suppose

$$
w = 100.
$$

The same input change produces

$$
\Delta z = 10.
$$

The neuron has become one hundred times more sensitive.

As additional nonlinear activation functions are applied,

this sensitivity propagates through the network.

Eventually,

tiny differences between two crystal structures may produce enormous differences in prediction.

Such behavior often indicates that the network has begun fitting accidental details rather than robust physical relationships.

---

### Smooth Functions Versus Highly Sensitive Functions

Imagine two MEGNet models trained to predict band gaps.

The first model changes its prediction gradually as bond lengths change.

```text
Bond Length

↓

Prediction

↓

Smooth Change
```

The second model changes its prediction dramatically after extremely small geometric changes.

```text
Bond Length

↓

Prediction

↓

Large Oscillations
```

Which behavior is more physically reasonable?

Materials properties usually vary continuously.

Small structural perturbations rarely produce enormous discontinuities.

Therefore,

the smoother model is generally more consistent with physical reality.

Weight decay encourages this smoother behavior.

---

### Constructing the Regularization Term

Suppose the neural network contains

$$
n
$$

trainable parameters

$$
w_1,w_2,\ldots,w_n.
$$

A natural measure of the overall parameter magnitude is

$$
w_1^2+w_2^2+\cdots+w_n^2.
$$

Using summation notation,

this becomes

$$
\sum_{i=1}^{n}w_i^2.
$$

This quantity increases whenever parameter magnitudes become large.

Consequently,

it provides a convenient measure of model complexity.

We therefore define the L2 penalty as

$$
\boxed{
L_{L2}
======

\sum_{i=1}^{n}
w_i^2
}
$$

Notice that every trainable parameter contributes to the penalty.

Large parameters contribute much more strongly because their values are squared.

---

### Why Do We Square the Parameters?

A natural question arises.

Why use

$$
w^2
$$

instead of simply

$$
|w|
$$

or

$$
w?
$$

Several reasons make squaring particularly attractive.

#### Positive Contributions

If we summed the raw parameter values,

positive and negative weights would cancel each other.

For example,

$$
5+(-5)=0.
$$

Clearly,

this does not indicate a simple model.

Squaring eliminates this problem because

$$
5^2=25,
$$

$$
(-5)^2=25.
$$

Both contribute equally.

---

#### Larger Penalties for Larger Weights

Squaring increases rapidly.

Consider

| Weight | Square |
| -----: | -----: |
|    0.5 |   0.25 |
|      1 |      1 |
|      2 |      4 |
|      5 |     25 |
|     10 |    100 |

Small parameters receive only modest penalties.

Extremely large parameters receive very strong penalties.

This behavior is precisely what we desire.

---

#### Mathematical Convenience

The derivative of

$$
w^2
$$

is

$$
2w,
$$

which is continuous,

smooth,

and computationally efficient.

This makes optimization straightforward.

---

### The Complete Loss Function

Without regularization,

the optimization objective is simply

$$
L_{prediction}.
$$

After introducing L2 regularization,

the objective becomes

$$
\boxed{
L_{total}
=========

L_{prediction}
+
\lambda
\sum_{i=1}^{n}
w_i^2
}
$$

where

$$
\lambda
$$

is called the **regularization coefficient**.

This equation is one of the most important equations in supervised machine learning.

It tells the optimizer to minimize

both

* prediction error,
* parameter magnitude.

---

### The Role of the Regularization Coefficient

The coefficient

$$
\lambda
$$

determines how strongly the model is penalized for large weights.

Suppose

$$
\lambda=0.
$$

Then

$$
L_{total}
=========

L_{prediction}.
$$

No regularization occurs.

The optimizer behaves exactly as before.

Now suppose

$$
\lambda
=======

100.

$$

The regularization term dominates.

The optimizer aggressively forces parameters toward zero.

The resulting model may become too simple,

leading to underfitting.

Therefore,

choosing

$$
\lambda
$$

requires careful balance.

In practice,

common values include

$$
10^{-5},
10^{-4},
10^{-3},
10^{-2}.
$$

The optimal value depends on

* dataset size,
* network architecture,
* prediction task.

---

### A Numerical Example

Suppose

the prediction loss equals

$$
0.025.
$$

The sum of squared parameters is

$$
420.
$$

If

$$
\lambda=10^{-4},
$$

then

$$
L_{regularization}
==================

10^{-4}
\times420
=========

0.042.
$$

The optimizer therefore minimizes

$$
L_{total}
=========

0.025
+
0.042
=====

0.067.
$$

Notice that

improving prediction accuracy is no longer sufficient.

Increasing parameter magnitudes also increases the total loss.

The optimizer must therefore find a compromise between accuracy and simplicity.

---

### How Weight Decay Influences Optimization

Recall the ordinary gradient descent update

$$
w_{new}
=======

## w_{old}

\eta
\frac{\partial L}{\partial w}.
$$

When L2 regularization is added,

the gradient now contains an additional contribution from

$$
\lambda
w^2.
$$

Since

$$
\frac{d}{dw}(w^2)=2w,
$$

the optimizer effectively experiences a small force that continuously pulls parameters toward zero.

This does **not** abruptly eliminate parameters.

Instead,

every optimization step slightly discourages unnecessarily large weights.

Consequently,

parameter growth remains controlled throughout training.

---

### Why It Is Called "Weight Decay"

The phrase **weight decay** arises because parameter magnitudes gradually decrease during optimization.

Suppose a parameter currently equals

$$
w=8.
$$

During successive optimization steps,

the regularization term gently nudges it toward smaller values.

Conceptually,

```text
8.0

↓

7.8

↓

7.5

↓

7.2

↓

6.9

↓

...
```

This gradual shrinking is called **decay**.

Importantly,

weights are not forced to zero unless doing so also minimizes the prediction loss.

Instead,

only unnecessary parameter growth is discouraged.

---

### Physical Interpretation for MEGNet

From a materials science perspective,

weight decay encourages the neural network to learn **stable and physically meaningful relationships** rather than highly specialized mappings that depend on accidental characteristics of the training dataset.

Because neighboring crystal structures often possess similar physical properties,

smooth predictive functions are generally more consistent with real chemistry than functions that fluctuate dramatically in response to tiny structural changes.

Weight decay therefore acts as a mathematical expression of an important scientific principle:

> **Prefer the simplest explanation that adequately describes the observed data.**

---

In the next part, we will move beyond the mathematics and examine **how weight decay is implemented in PyTorch**, explain why the `weight_decay` parameter in the Adam optimizer corresponds to L2 regularization, discuss practical hyperparameter selection, and clarify which parameters should—and should not—receive weight decay during MEGNet training.


## 15.8.12 Preventing Overfitting: Regularization in MEGNet (Part 3)

In the previous section, we developed the mathematical foundation of L2 regularization. We derived the regularized loss function and showed that large parameter values are penalized during optimization.

However, researchers rarely implement L2 regularization by explicitly adding

$$
\lambda \sum_i w_i^2
$$

to the loss function.

Instead, modern deep learning frameworks implement weight decay directly inside the optimizer.

Understanding this implementation is important because almost every MEGNet training script uses the optimizer interface rather than manually modifying the loss function.

---

### Weight Decay in PyTorch

The simplest implementation of weight decay in PyTorch looks like

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3,

    weight_decay=1e-5

)
```

At first glance,

this appears surprisingly simple.

Only one additional argument has been introduced:

```python
weight_decay=1e-5
```

Behind this single line,

PyTorch automatically performs L2 regularization throughout optimization.

The user does not need to modify the loss function manually.

---

### What Happens Internally?

Suppose the optimizer computes the gradient of the prediction loss.

Without weight decay,

the update direction depends only on

$$
\frac{\partial L_{\text{prediction}}}{\partial w}.
$$

With weight decay,

an additional gradient term is included.

Conceptually,

the optimizer now behaves as though it were minimizing

$$
L_{\text{prediction}}
+
\lambda
\sum_i
w_i^2.
$$

Therefore,

each parameter update contains two contributions.

The first contribution attempts to improve prediction accuracy.

The second contribution attempts to keep the parameters reasonably small.

The optimizer balances these two objectives automatically.

---

### Why Optimizer-Based Regularization Is Preferred

One might wonder why PyTorch implements weight decay inside the optimizer rather than requiring users to modify the loss function themselves.

There are several reasons.

First,

the optimizer-based implementation is concise.

Instead of writing additional mathematical expressions,

the user specifies only one hyperparameter.

Second,

the implementation is computationally efficient.

Modern optimizers perform the necessary calculations directly during parameter updates.

Third,

this approach reduces programming errors.

Researchers can easily enable or disable regularization without modifying the training loop.

Consequently,

optimizer-based weight decay has become the standard practice throughout deep learning.

---

### Choosing an Appropriate Weight Decay

Selecting the weight decay coefficient is an example of **hyperparameter optimization**.

There is no universally optimal value.

Instead,

the best choice depends on

* the dataset,
* the neural network architecture,
* the target property,
* and the amount of available training data.

Nevertheless,

certain ranges have become common in practice.

| Weight Decay | Typical Interpretation     |
| ------------ | -------------------------- |
| 0            | No regularization          |
| $10^{-6}$    | Very weak regularization   |
| $10^{-5}$    | Common for many GNNs       |
| $10^{-4}$    | Moderate regularization    |
| $10^{-3}$    | Strong regularization      |
| $10^{-2}$    | Very strong regularization |

These values should not be viewed as rigid rules.

Instead,

they provide reasonable starting points for experimentation.

---

### What Happens If Weight Decay Is Too Small?

Suppose

```python
weight_decay = 0
```

or

```python
weight_decay = 1e-8
```

The regularization term becomes almost negligible.

The optimizer focuses almost entirely on minimizing the prediction loss.

Large parameter values are no longer discouraged.

If the network possesses high capacity,

memorization becomes increasingly likely.

Learning curves may eventually exhibit the characteristic divergence associated with overfitting.

---

### What Happens If Weight Decay Is Too Large?

Now suppose

```python
weight_decay = 0.1
```

or even

```python
weight_decay = 1
```

The optimizer now spends most of its effort shrinking parameters.

Prediction accuracy becomes a secondary objective.

Many useful parameters are forced toward zero before they have an opportunity to learn meaningful chemical relationships.

Training loss remains relatively high.

Validation performance also suffers.

This situation corresponds to **underfitting** rather than overfitting.

Thus,

both extremes should be avoided.

---

### A Conceptual View of the Trade-Off

Weight decay introduces a balance between two competing objectives.

On one side,

the optimizer seeks maximum prediction accuracy.

On the other,

it seeks minimum model complexity.

Conceptually,

```text
Prediction Accuracy

←──────────────────────→

Model Simplicity
```

Increasing weight decay shifts the optimization toward simplicity.

Reducing weight decay shifts the optimization toward prediction accuracy.

The optimal model lies somewhere between these extremes.

---

### Should Every Parameter Receive Weight Decay?

An interesting practical question now arises.

Should weight decay be applied to **every trainable parameter**?

The answer is

**not always**.

Modern deep learning often distinguishes between different types of parameters.

For example,

consider

* weights of linear layers,
* weights of graph convolution layers,
* bias parameters,
* Batch Normalization parameters,
* Layer Normalization parameters.

Many researchers apply weight decay only to the weight matrices while excluding

* biases,
* normalization parameters.

The reason is that these parameters play different roles during optimization.

Excessively shrinking normalization parameters may actually reduce training stability.

---

### Selective Weight Decay in PyTorch

PyTorch allows parameters to be grouped so that different optimization settings can be applied.

A simplified example is

```python
optimizer = torch.optim.Adam(

    [

        {

            "params": weight_parameters,

            "weight_decay": 1e-5

        },

        {

            "params": bias_parameters,

            "weight_decay": 0.0

        }

    ],

    lr=1e-3

)
```

Here,

ordinary weight matrices receive regularization,

whereas bias parameters do not.

Large-scale research codebases frequently employ this strategy.

---

### Weight Decay and Adam

Historically,

many researchers combined weight decay directly with the Adam optimizer.

Later research demonstrated that the original implementation was **not mathematically identical** to true weight decay.

This observation led to the development of **AdamW**, an optimizer that decouples weight decay from the adaptive gradient update.

Today,

many modern deep learning libraries recommend

```python
torch.optim.AdamW
```

instead of

```python
torch.optim.Adam
```

when weight decay is required.

Although both optimizers often produce similar results,

AdamW provides a cleaner theoretical formulation and has become the preferred choice in many state-of-the-art models.

---

### Adam Versus AdamW

The practical difference can be summarized as follows.

| Optimizer      | Weight Decay Implementation           |
| -------------- | ------------------------------------- |
| Adam           | Coupled with adaptive gradients       |
| AdamW          | Decoupled weight decay                |
| Recommendation | Prefer AdamW for modern deep learning |

Many recent graph neural network implementations,

including advanced materials informatics models,

use AdamW for this reason.

---

### Practical Recommendations for MEGNet

When training a MEGNet model,

a reasonable starting configuration is

```python
optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-3,

    weight_decay=1e-5

)
```

This configuration

* provides adaptive optimization,
* controls parameter growth,
* reduces the risk of overfitting,
* and works well for many regression tasks involving crystal graphs.

Nevertheless,

the optimal hyperparameters should always be determined experimentally using the validation dataset rather than copied blindly from previous studies.

---

### Limitations of Weight Decay

Although weight decay is extremely useful,

it is **not a complete solution** to overfitting.

Even with carefully chosen regularization,

a sufficiently large neural network may still memorize the training dataset.

Additional techniques are therefore required.

One of the most effective of these techniques is **dropout**, which approaches regularization from an entirely different perspective.

Rather than penalizing large parameters,

dropout temporarily removes portions of the neural network during training, forcing the remaining neurons to learn more robust and independent representations.

In the next section, we will study **dropout from first principles**, understand why randomly disabling neurons improves generalization, derive its mathematical interpretation, and implement dropout layers within a MEGNet architecture using PyTorch.

## 15.8.12 Preventing Overfitting: Regularization in MEGNet (Part 4)

### Dropout: Preventing Neurons from Becoming Overly Dependent

Weight decay regularizes a neural network by discouraging excessively large parameter values.

Dropout approaches the same problem from an entirely different perspective.

Instead of restricting the magnitude of the weights,

dropout modifies the **network architecture during training**.

More specifically,

dropout temporarily removes a randomly selected subset of neurons from the computation graph during each forward pass.

At first glance, this idea appears counterintuitive.

Why would deliberately removing information improve the performance of a neural network?

Surprisingly,

this simple idea has become one of the most influential regularization techniques in deep learning.

Although dropout is used less aggressively in graph neural networks than in image classification, it remains an important component of many GNN architectures, including numerous variants used in materials informatics.

---

### The Fundamental Problem: Co-Adaptation

To understand dropout, we must first understand a phenomenon known as **co-adaptation**.

Consider two neurons inside a multilayer perceptron.

Suppose the first neuron always activates whenever a certain crystal environment is observed.

The second neuron gradually learns to rely almost entirely on the first neuron.

Instead of learning an independent representation,

it simply assumes that the first neuron will always provide the necessary information.

This creates a dependency.

Eventually,

many neurons begin relying on one another in increasingly specialized ways.

The network becomes highly effective for the training dataset,

but these fragile dependencies often fail to generalize.

This behavior is called **co-adaptation**.

Dropout was specifically designed to reduce this phenomenon.

---

### An Analogy

Imagine a research laboratory with five scientists.

Initially,

every scientist understands the complete research project.

Now suppose one scientist becomes exceptionally skilled.

Gradually,

the remaining four scientists stop learning certain tasks because they know the expert will always perform them.

The laboratory now depends heavily on one individual.

If that researcher becomes unavailable,

the entire project suffers.

A better strategy is to require every scientist to understand the project independently.

Dropout applies exactly this philosophy to neural networks.

During training,

some neurons are temporarily removed.

The remaining neurons must continue solving the prediction task without relying on those missing neurons.

As a result,

every neuron learns a stronger and more independent representation.

---

### The Basic Idea

Consider a hidden layer containing eight neurons.

Normally,

every neuron participates in the forward pass.

```text id="k3vxp2"
Input

↓

● ● ● ● ● ● ● ●

↓

Output
```

Now suppose dropout uses a probability

$$
p = 0.5.
$$

During one training iteration,

approximately half of the neurons are randomly disabled.

```text id="n7hd4c"
Input

↓

● ○ ● ○ ● ● ○ ●

↓

Output
```

The circles marked

○

represent neurons that have been temporarily removed.

These neurons contribute nothing during this forward pass.

On the next mini-batch,

a completely different subset of neurons is removed.

```text id="qv5p8e"
Input

↓

○ ● ● ● ○ ● ● ○

↓

Output
```

Every mini-batch therefore trains a slightly different neural network.

---

### Why Random Removal Helps

Suppose a particular neuron has become extremely important.

Without dropout,

many other neurons may learn to depend on it.

Now imagine that dropout suddenly removes this neuron.

The remaining network must continue making accurate predictions without its assistance.

Consequently,

other neurons are forced to develop alternative representations.

Over many iterations,

the network becomes much more robust.

Instead of relying on a few dominant neurons,

information becomes distributed across many neurons.

This distributed representation usually generalizes much better.

---

### Dropout Can Be Viewed as Training Many Networks

One of the most elegant interpretations of dropout is that it approximately trains an enormous collection of different neural networks simultaneously.

Consider a layer containing only four neurons.

Each neuron may be

* active,
* inactive.

Therefore,

the number of possible subnetworks is

$$
2^4 = 16.
$$

Now imagine a layer containing

512 neurons.

The number of possible subnetworks becomes

$$
2^{512},
$$

which is astronomically large.

Of course,

we never explicitly train all these networks.

Instead,

every mini-batch randomly samples one particular subnetwork.

After thousands of optimization steps,

the learned parameters perform well across many different subnetworks.

This is one reason dropout improves robustness.

---

### Mathematical Description

Suppose the activation of a neuron is

$$
h.
$$

Introduce a random variable

$$
m,
$$

called the dropout mask.

The mask follows a Bernoulli distribution.

Specifically,

$$
m =
\begin{cases}
1, & \text{with probability } 1-p,\
0, & \text{with probability } p.
\end{cases}
$$

The output after dropout becomes

$$
\tilde{h}
=========

m h.
$$

If

$$
m=1,
$$

the neuron remains active.

If

$$
m=0,
$$

the neuron is completely removed during that forward pass.

This operation is repeated independently for every neuron in the dropout layer.

---

### The Meaning of the Dropout Probability

The parameter

$$
p
$$

controls the fraction of neurons removed.

Common values include

| Dropout Probability | Interpretation          |
| ------------------: | ----------------------- |
|                 0.0 | No dropout              |
|                 0.1 | Very weak dropout       |
|                 0.2 | Light regularization    |
|                 0.3 | Moderate regularization |
|                 0.5 | Strong regularization   |

A larger value increases regularization,

but excessive dropout may remove too much information,

making optimization difficult.

Thus,

dropout probability must be selected carefully.

---

### What Happens During Every Mini-Batch?

Suppose training proceeds for

100 epochs.

Each epoch contains

500 mini-batches.

During every mini-batch,

a new dropout mask is generated.

Consequently,

the network architecture changes

50,000 times throughout training.

Importantly,

the parameters are shared among all these temporary subnetworks.

This continual architectural variation prevents the network from memorizing highly specific neuron interactions.

---

### Why Dropout Is Disabled During Evaluation

A very important question now arises.

Should neurons continue disappearing when the model predicts new materials?

The answer is

**no**.

During validation and testing,

we want deterministic and reproducible predictions.

Randomly disabling neurons would cause predictions to fluctuate.

Therefore,

dropout is used **only during training**.

During evaluation,

every neuron participates.

This is one of the reasons PyTorch distinguishes between

```python
model.train()
```

and

```python
model.eval()
```

When

```python
model.train()
```

is active,

dropout randomly removes neurons.

When

```python
model.eval()
```

is called,

dropout is automatically disabled,

and the complete network is used for inference.

This behavior occurs automatically for every `nn.Dropout` layer.

---

### Why Dropout Works Well

Although dropout appears almost random,

its effect is remarkably systematic.

It

* reduces co-adaptation,
* distributes learned information,
* increases robustness,
* decreases memorization,
* improves generalization.

However,

dropout also has limitations.

In graph neural networks,

particularly those operating on relatively small crystal graphs,

aggressive dropout may unintentionally remove important structural information.

Consequently,

dropout probabilities used in GNNs are often smaller than those commonly employed in computer vision.

In the next section, we will move from the underlying theory to **practical dropout implementation in MEGNet**, examining exactly where dropout layers should be inserted, where they should be avoided, and how different dropout placements influence predictive performance in graph neural networks for materials science.

## 15.8.12 Preventing Overfitting: Regularization in MEGNet (Part 5)

In the previous section, we introduced the fundamental idea of dropout and explained why randomly removing neurons during training improves generalization.

However, an important practical question remains.

> **Where should dropout actually be placed inside a MEGNet model?**

The answer is more subtle than many beginners expect.

Unlike a conventional multilayer perceptron, MEGNet contains several different computational stages.

Each stage processes different types of information.

Applying dropout indiscriminately may actually reduce predictive performance.

Understanding where dropout is appropriate—and where it should be avoided—is therefore an important aspect of designing high-quality graph neural networks.

---

### Revisiting the MEGNet Architecture

Recall the overall architecture developed earlier in this chapter.

```text
Crystal Structure

↓

Graph Construction

↓

Node Features

↓

Edge Features

↓

Global Features

↓

Embedding Layers

↓

MEGNet Block 1

↓

MEGNet Block 2

↓

...

↓

Readout

↓

Fully Connected Layers

↓

Property Prediction
```

Each of these components performs a distinct function.

Some layers learn local chemical representations.

Others combine information across the graph.

Still others transform the graph representation into the final prediction.

Dropout affects each stage differently.

---

### Should Dropout Be Applied to the Input Features?

One possible idea is to apply dropout immediately after the input features are created.

For example,

suppose each atom is represented by

* atomic number,
* electronegativity,
* covalent radius,
* valence electron count.

Should we randomly remove some of these descriptors?

Generally,

the answer is **no**.

These features represent genuine physical information.

Randomly deleting them may remove important chemical knowledge before the network has an opportunity to learn.

Unlike images,

where individual pixels contain highly redundant information,

many atomic descriptors are fundamentally important.

Consequently,

input dropout is rarely used in materials graph neural networks.

---

### Dropout Inside the Message-Passing Layers

The next possibility is applying dropout inside the Graph Network blocks.

Recall that each MEGNet block performs

* edge updates,
* node updates,
* global state updates.

These computations are responsible for learning the chemical interactions within the crystal.

Conceptually,

```text
Edge Update

↓

Node Update

↓

Global Update
```

Should dropout be introduced here?

The answer is

**sometimes—but with caution**.

Message passing is the heart of the graph neural network.

Removing too much information during this stage may interrupt the propagation of chemical information between neighboring atoms.

If the dropout probability is too large,

important interaction pathways may disappear.

As a result,

the network may struggle to learn meaningful atomic representations.

For this reason,

many MEGNet implementations either

* avoid dropout inside message-passing blocks entirely,
* or use only very small dropout probabilities.

---

### Why Message Passing Is Different from Ordinary Neural Networks

In a conventional multilayer perceptron,

each neuron processes information independently.

Randomly removing several neurons generally has only a local effect.

Graph neural networks behave differently.

Every node communicates with its neighbors.

Removing information from one node can indirectly influence many other nodes through repeated message passing.

Consider a crystal graph.

```text id="8x74gn"
Atom A

↓

Atom B

↓

Atom C

↓

Atom D
```

Suppose dropout removes information associated with Atom B.

Now,

Atom C receives incomplete information.

During the next message-passing layer,

Atom D also receives degraded information.

Thus,

the effect propagates through the graph.

Because of this cascading behavior,

dropout inside graph convolutions must be used carefully.

---

### Dropout After Readout

One of the safest locations for dropout is **after graph pooling (readout)**.

Recall that the readout layer converts the variable-sized crystal graph into a fixed-length vector.

Conceptually,

```text
Crystal Graph

↓

Readout

↓

Graph Embedding

↓

Fully Connected Network
```

At this stage,

the graph representation has already been constructed.

The remaining computations resemble an ordinary feed-forward neural network.

Applying dropout here is often very effective.

The graph representation remains intact,

while the prediction head is prevented from overfitting.

---

### Dropout in the Prediction Head

Suppose the prediction head consists of

```text
Graph Embedding

↓

Linear Layer

↓

ReLU

↓

Linear Layer

↓

Prediction
```

A common architecture introduces dropout between the fully connected layers.

```text
Graph Embedding

↓

Linear

↓

ReLU

↓

Dropout

↓

Linear

↓

Prediction
```

This is one of the most widely used dropout configurations in graph neural networks.

Because the message-passing process has already finished,

dropout no longer interferes with information propagation across the crystal.

Instead,

it regularizes only the final regression network.

---

### PyTorch Implementation

PyTorch provides dropout through the module

```python
nn.Dropout
```

A simple prediction head might be written as

```python
self.predictor = nn.Sequential(

    nn.Linear(hidden_dim, hidden_dim),

    nn.ReLU(),

    nn.Dropout(p=0.2),

    nn.Linear(hidden_dim, 1)

)
```

During training,

approximately

20%

of the hidden activations are randomly removed.

During evaluation,

every neuron remains active automatically.

No additional code is required.

---

### Multiple Dropout Layers

Large neural networks sometimes employ several dropout layers.

For example,

```text
Linear

↓

ReLU

↓

Dropout

↓

Linear

↓

ReLU

↓

Dropout

↓

Linear

↓

Prediction
```

Each dropout layer generates an independent random mask.

Consequently,

different portions of the prediction head are regularized simultaneously.

This strategy is common in deep multilayer perceptrons.

---

### Choosing the Dropout Probability

The choice of dropout probability depends on

* dataset size,
* model complexity,
* amount of available training data.

Typical values for graph neural networks include

| Dropout Probability | Interpretation                  |
| ------------------: | ------------------------------- |
|                 0.0 | Disabled                        |
|                0.05 | Very light                      |
|                0.10 | Light                           |
|                0.20 | Common                          |
|                0.30 | Moderate                        |
|                0.50 | Usually too aggressive for GNNs |

Unlike image classification,

where dropout values of

$$
0.5
$$

are common,

graph neural networks frequently use smaller probabilities because graph datasets are relatively small and graph representations are more sensitive to missing information.

---

### Combining Weight Decay and Dropout

An important question is whether dropout replaces weight decay.

The answer is

**no**.

These techniques address overfitting in different ways.

Weight decay discourages excessively large parameter values.

Dropout discourages excessive dependence between neurons.

Because they operate through different mechanisms,

they complement one another.

Many successful MEGNet training configurations therefore combine

* AdamW optimization,
* weight decay,
* dropout,
* and early stopping.

Together,

these techniques provide substantially better generalization than any single method alone.

---

### A Practical Observation

If you examine published implementations of

* MEGNet,
* M3GNet,
* CHGNet,
* ALIGNN,

you will notice an interesting trend.

The graph representation itself is often treated carefully,

with relatively little dropout during message passing,

while the final multilayer perceptron receives stronger regularization.

This reflects an important principle:

> The learned crystal representation is valuable and should be preserved, whereas the final prediction head is comparatively more prone to memorizing the training data.

This design philosophy has proven effective across many state-of-the-art materials informatics models.

---

At this point, we have examined the two most common parameter-based regularization techniques:

* **Weight Decay (L2 Regularization)**
* **Dropout**

However, among all regularization methods used in practical deep learning, the one that often has the greatest impact is neither of these.

Instead, it is **Early Stopping**—a strategy that recognizes the precise moment at which a model has learned as much as it can before memorization begins.

In the next section, we will study **Early Stopping** in depth, beginning with its underlying intuition before moving to its mathematical interpretation and PyTorch implementation for MEGNet training.


## 15.8.12 Preventing Overfitting: Regularization in MEGNet (Part 6)

### Early Stopping: Stopping Training Before Memorization Begins

Among all the regularization techniques discussed so far,

* weight decay,
* dropout,
* architectural design,

one method stands out because of its remarkable simplicity and effectiveness.

That method is **Early Stopping**.

Interestingly, early stopping does not modify

* the neural network architecture,
* the optimizer,
* the loss function,
* or the dataset.

Instead, it changes **when training ends**.

This simple idea often produces larger improvements than making the neural network itself more sophisticated.

Consequently, early stopping has become a standard component of nearly every modern deep learning workflow, including MEGNet, M3GNet, ALIGNN, CHGNet, and many other graph neural network models.

---

### The Fundamental Idea

Recall the learning curves discussed in the previous section.

Initially,

both training loss and validation loss decrease together.

```text id="es1"
Training Loss

↓

Validation Loss

↓

Both Improve
```

As optimization continues,

the model reaches a point where it has learned most of the meaningful physical relationships contained within the data.

Beyond this point,

training continues to improve,

but validation performance begins to deteriorate.

Conceptually,

```text id="es2"
Training Loss

↓

Continues Decreasing

Validation Loss

↓

Begins Increasing
```

This marks the beginning of overfitting.

The key insight behind early stopping is remarkably simple:

> **Why continue training after validation performance has already begun to worsen?**

Instead of allowing memorization to continue,

we simply stop training.

---

### The Best Model Is Rarely the Final Model

A common misconception among beginners is that the model obtained after the final epoch must be the best model.

In practice,

this is often false.

Suppose training lasts for

200 epochs.

Validation performance might evolve as follows.

| Epoch | Validation MAE (eV) |
| ----: | ------------------: |
|    20 |               0.115 |
|    40 |               0.081 |
|    60 |           **0.064** |
|    80 |               0.068 |
|   100 |               0.076 |
|   150 |               0.102 |
|   200 |               0.145 |

The best-performing model was obtained at

**Epoch 60**.

Continuing to Epoch 200 only increased memorization.

If we publish the final model instead of the best validation model,

our reported results will actually be worse.

Therefore,

modern deep learning almost never uses the final epoch automatically.

Instead,

it keeps track of the best validation performance throughout training.

---

### An Analogy

Imagine preparing for a marathon.

Initially,

each day of training improves your physical fitness.

Eventually,

you reach peak condition.

If training continues without sufficient recovery,

fatigue begins accumulating.

Performance gradually declines.

At this stage,

continuing to train harder no longer improves athletic ability.

The optimal strategy is to stop before excessive fatigue develops.

Neural networks behave similarly.

Initially,

additional epochs improve learning.

Eventually,

the network reaches peak generalization.

Beyond this point,

additional optimization increases memorization rather than understanding.

Early stopping simply recognizes this optimal moment.

---

### Why Early Stopping Works

Recall that the optimizer minimizes only the training loss.

It has no direct understanding of generalization.

Consequently,

the optimizer will continue reducing training error indefinitely if allowed.

Validation performance,

however,

provides information about unseen data.

By monitoring validation loss,

we indirectly estimate the true generalization error.

When validation performance stops improving,

it suggests that further optimization is no longer discovering meaningful physical relationships.

Instead,

the optimizer is fitting characteristics unique to the training dataset.

Stopping at this point preserves the best balance between learning and memorization.

---

### Monitoring Validation Loss

Suppose validation loss is recorded after every epoch.

```text id="es3"
Epoch

↓

Validation Loss

↓

Store Result
```

The algorithm compares the current validation loss with the best value observed so far.

If the new value is lower,

the current model becomes the new best model.

If not,

training continues,

but a counter is increased.

This process repeats after every epoch.

---

### The Concept of Patience

Validation loss does not always decrease smoothly.

Because optimization is stochastic,

small fluctuations are normal.

For example,

suppose validation loss evolves as

| Epoch | Validation Loss |
| ----: | --------------: |
|    50 |           0.082 |
|    51 |           0.081 |
|    52 |           0.082 |
|    53 |           0.080 |
|    54 |           0.081 |

Stopping immediately after Epoch 52 would be a mistake.

The increase from

0.081

to

0.082

is simply random variation.

To avoid premature termination,

early stopping introduces the concept of **patience**.

Patience specifies the number of consecutive epochs during which validation performance is allowed to remain unimproved.

For example,

```text id="es4"
Patience = 20
```

means that training continues until

20 consecutive epochs

fail to improve validation performance.

Only then is training terminated.

---

### Choosing the Patience Value

The patience parameter depends on

* dataset size,
* optimizer,
* learning rate,
* batch size,
* model complexity.

Common values include

| Patience | Typical Usage                        |
| -------: | ------------------------------------ |
|        5 | Small experiments                    |
|       10 | Moderate training                    |
|       20 | Common for GNNs                      |
|    30–50 | Large datasets with slow convergence |

Graph neural networks often require relatively large patience values because validation loss may improve slowly over many epochs.

Stopping too early may prevent the network from reaching its optimal solution.

---

### The Early Stopping Algorithm

Conceptually,

the procedure is straightforward.

```text id="es5"
Initialize

↓

Best Validation Loss = ∞

↓

Train One Epoch

↓

Evaluate Validation Loss

↓

Improved?

↓

Yes

↓

Save Model

↓

Reset Counter

↓

Continue

↓

No

↓

Increase Counter

↓

Counter > Patience?

↓

Yes

↓

Stop Training
```

Although simple,

this algorithm has become one of the most widely used regularization strategies in deep learning.

---

### Why Early Stopping Is Especially Effective for Materials GNNs

Materials datasets are typically much smaller than datasets used in natural language processing or computer vision.

Consequently,

graph neural networks often begin overfitting relatively early.

Suppose a MEGNet model is trained using

50,000 crystal structures.

The model may already reach its optimal generalization performance after

80 epochs.

Continuing to

300 epochs

rarely improves predictive accuracy.

Instead,

the network gradually memorizes increasingly specific structural patterns.

Early stopping prevents this unnecessary optimization.

---

### A Common Misunderstanding

Many beginners assume that early stopping is merely a method for reducing training time.

This interpretation is incomplete.

Reducing computation is a useful side effect,

but it is **not** the primary objective.

The true purpose is

> **to preserve the model corresponding to the best generalization performance.**

Even if computational resources were unlimited,

continuing optimization beyond the optimal validation epoch would still produce an inferior scientific model.

Therefore,

early stopping should be viewed as a regularization technique,

not simply as a computational optimization.

---

### Combining Early Stopping with Weight Decay and Dropout

One of the reasons modern graph neural networks generalize so well is that they rarely rely on a single regularization technique.

Instead,

multiple complementary methods are used simultaneously.

A typical MEGNet training pipeline may include

* AdamW optimization,
* weight decay,
* dropout,
* early stopping,
* model checkpointing.

Each technique addresses a different aspect of overfitting.

Together,

they produce a model that is significantly more robust than one trained without regularization.

---

At this point, we understand **why early stopping works**. The next practical question is equally important:

> **How do we implement early stopping in a real MEGNet training loop?**

In the next section, we will build a complete early stopping utility in PyTorch, integrate it into the MEGNet training process, and explain how professional research code automatically saves and restores the best-performing model during optimization.

## 15.8.12 Preventing Overfitting: Regularization in MEGNet (Part 7)

### Implementing Early Stopping in PyTorch for MEGNet

In the previous section, we studied the theory behind early stopping. We learned that training should terminate when validation performance no longer improves, rather than after an arbitrary number of epochs.

Now we will implement this idea in code.

This section is particularly important because nearly every professional deep learning project—including those in materials informatics—uses some variation of this procedure.

---

## A Common Beginner Mistake

Many beginners write training loops like this:

```python
for epoch in range(200):

    train(...)

    validate(...)
```

After 200 epochs,

they simply save the final model.

```python
torch.save(model.state_dict(), "final_model.pth")
```

This approach is easy to understand,

but it contains a serious flaw.

There is absolutely no guarantee that

**Epoch 200**

contains the best model.

The best validation performance may have occurred at

* Epoch 47,
* Epoch 83,
* Epoch 126,

or any other point during optimization.

Saving only the final model may therefore discard the best-performing network.

---

# The Correct Strategy

Instead of saving only the last model,

we should save the model **whenever validation performance improves.**

Conceptually,

```text
Epoch 1

↓

Validation Loss = 0.92

↓

Save Model

↓

Epoch 2

↓

Validation Loss = 0.81

↓

Save Model

↓

Epoch 3

↓

Validation Loss = 0.78

↓

Save Model

↓

...

↓

Epoch 48

↓

Validation Loss = 0.064

↓

Save Model

↓

Epoch 49

↓

Validation Loss = 0.065

↓

Do Not Save

↓

Epoch 50

↓

Validation Loss = 0.067

↓

Do Not Save
```

Notice something important.

The best model is always preserved.

Even if training later becomes unstable,

the optimal checkpoint already exists.

---

# Tracking the Best Validation Loss

The simplest implementation begins by storing

```python
best_val_loss = float("inf")
```

Initially,

the best validation loss is set to positive infinity.

Since every real validation loss is smaller,

the first epoch automatically becomes the best model.

After each epoch,

we compare

```python
if val_loss < best_val_loss:
```

If the new validation loss is smaller,

we update

```python
best_val_loss = val_loss
```

and save the model.

---

# Saving the Best Model

Saving the model is straightforward.

```python
torch.save(

    model.state_dict(),

    "best_megnet_model.pth"

)
```

The file

```text
best_megnet_model.pth
```

contains only the learned parameters.

Whenever validation performance improves,

this file is overwritten with the new best model.

At the end of training,

it automatically contains the parameters corresponding to the lowest validation loss.

---

# Counting Consecutive Failures

Saving the best model is only half of early stopping.

We also need to determine

when training should terminate.

To accomplish this,

we introduce a counter.

Initially,

```python
counter = 0
```

Whenever validation improves,

the counter is reset.

```python
counter = 0
```

Whenever validation fails to improve,

the counter increases.

```python
counter += 1
```

If the counter becomes larger than the chosen patience,

training stops.

---

# The Complete Logic

The decision process can be summarized as

```text
Validation Improved?

↓

Yes

↓

Save Model

↓

Reset Counter

↓

Continue Training

-------------------

Validation Improved?

↓

No

↓

Increase Counter

↓

Counter > Patience?

↓

Yes

↓

Stop Training
```

Although simple,

this algorithm is remarkably effective.

---

# A Complete Early Stopping Class

Rather than scattering this logic throughout the training loop,

professional code usually encapsulates it inside a reusable class.

```python
class EarlyStopping:

    def __init__(self, patience=20):

        self.patience = patience

        self.counter = 0

        self.best_loss = float("inf")

        self.stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            self.counter = 0

            return True

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True

            return False
```

This class stores

* the best validation loss,
* the patience,
* the failure counter,
* whether training should stop.

---

# Using the Class

Training becomes much cleaner.

```python
early_stopping = EarlyStopping(

    patience=20

)
```

After every validation step,

```python
save_model = early_stopping(val_loss)
```

If

```python
save_model
```

is

```python
True
```

the current parameters are saved.

```python
if save_model:

    torch.save(

        model.state_dict(),

        "best_model.pth"

    )
```

Finally,

after each epoch,

```python
if early_stopping.stop:

    break
```

The training loop immediately terminates.

---

# Complete Training Flow

The overall procedure now becomes

```text
Initialize Model

↓

Initialize Optimizer

↓

Initialize EarlyStopping

↓

Epoch

↓

Training

↓

Validation

↓

Validation Improved?

↓

Yes

↓

Save Model

↓

Reset Counter

↓

Continue

----------------

No

↓

Increase Counter

↓

Counter ≥ Patience?

↓

Yes

↓

Terminate Training
```

This workflow is almost identical to that used in production research code.

---

# Restoring the Best Model

One subtle but extremely important point remains.

Suppose training terminates at

Epoch 143.

The best validation performance actually occurred at

Epoch 117.

The parameters currently stored inside

```python
model
```

correspond to

Epoch 143,

not

Epoch 117.

Therefore,

before testing or inference,

we must reload the saved checkpoint.

```python
model.load_state_dict(

    torch.load(

        "best_model.pth"

    )

)
```

Now,

the model once again contains the parameters associated with the lowest validation loss.

Only after this step should we evaluate the test dataset.

This detail is frequently overlooked by beginners but is essential for obtaining correct results.

---

# Why Checkpointing and Early Stopping Work Together

Notice that early stopping alone is insufficient.

Imagine the following sequence.

| Epoch | Validation MAE |
| ----: | -------------: |
|    95 |          0.052 |
|    96 |          0.051 |
|    97 |      **0.049** |
|    98 |          0.050 |
|    99 |          0.052 |
|   100 |          0.054 |
|   ... |            ... |
|   117 |           Stop |

Early stopping eventually terminates at

Epoch 117.

However,

the best model existed at

Epoch 97.

Without checkpointing,

that model would be lost forever.

This is why professional deep learning always combines

* **early stopping** and
* **model checkpointing**.

Checkpointing preserves the best parameters.

Early stopping decides when to terminate optimization.

Together,

they ensure that the final model corresponds to the highest validation performance.

---

# Practical Advice for MEGNet Training

For most MEGNet regression tasks, a solid starting configuration is:

| Parameter        |  Recommended Starting Value |
| ---------------- | --------------------------: |
| Optimizer        |                       AdamW |
| Learning rate    |            $1\times10^{-3}$ |
| Weight decay     |            $1\times10^{-5}$ |
| Dropout          |                     0.1–0.2 |
| Patience         |                20–30 epochs |
| Model checkpoint | Save lowest validation loss |

These are **starting points**, not universal rules. The final values should always be chosen based on validation performance for the specific dataset and prediction task.

---

At this point, we have covered the major regularization techniques used in MEGNet:

* L2 Regularization (Weight Decay)
* Dropout
* Early Stopping
* Model Checkpointing

The next major topic in Chapter 15 should naturally be **15.9 Training a Complete MEGNet Model**, where we integrate everything learned so far into a complete, research-quality training pipeline, including data loading, optimizer setup, learning rate scheduling, mixed precision (optional), logging, checkpointing, validation, testing, and final inference. This section will serve as the bridge between understanding MEGNet and using it for real materials informatics research.

# 15.9 Training a Complete MEGNet Model

At this point in the chapter, we have learned nearly every individual component required to build a MEGNet model.

We understand

* crystal graph construction,
* node, edge, and global features,
* message passing,
* graph updates,
* pooling,
* prediction heads,
* loss functions,
* optimizers,
* regularization,
* and early stopping.

However, these concepts have been studied independently.

In real research, they must work together as one complete machine learning pipeline.

This section integrates everything developed throughout the chapter into a complete MEGNet training workflow.

The goal is not simply to write code that runs.

The goal is to understand **every operation performed during training**, why it exists, and how it contributes to learning meaningful materials representations.

---

# From Crystal Structure to a Trained Neural Network

When reading a published MEGNet paper, the training process is often summarized in only a few sentences.

For example,

> "The model was trained using Adam with a learning rate of $10^{-3}$ for 300 epochs."

To a beginner, this sounds deceptively simple.

In reality,

those few words hide hundreds of computational operations.

A complete training workflow actually looks like

```text
Crystal Structures

↓

Graph Construction

↓

Dataset

↓

Train–Validation–Test Split

↓

DataLoader

↓

Initialize MEGNet

↓

Initialize Optimizer

↓

Initialize Scheduler

↓

Initialize Early Stopping

↓

Training Loop

↓

Validation

↓

Checkpoint Saving

↓

Learning Rate Adjustment

↓

Early Stopping Decision

↓

Best Model

↓

Test Evaluation

↓

Inference on New Materials
```

Every box in this workflow corresponds to dozens or even hundreds of lines of code.

Professional research software simply automates these operations.

Understanding them individually is what allows us to modify, debug, and improve the model.

---

# The Overall Philosophy of Training

Many beginners believe that training means

> "Give the neural network some data."

This is incorrect.

Training is actually an **optimization problem**.

Suppose the neural network contains

$$
\theta
$$

trainable parameters.

Initially,

these parameters are random.

Therefore,

the prediction

$$
\hat y=f(G;\theta)
$$

is also essentially random.

Training repeatedly modifies

$$
\theta
$$

until

$$
f(G;\theta)
$$

approximates the unknown physical function

$$
f_{\text{physics}}(G).
$$

Notice something important.

The dataset never changes.

The crystal structures remain exactly the same.

Only the neural network parameters change.

Training is therefore the process of **searching parameter space**.

---

# What Happens During One Epoch?

The word **epoch** appears constantly in machine learning literature.

Unfortunately,

many students use the term without understanding its precise meaning.

An epoch is defined as

> **One complete pass through the entire training dataset.**

Suppose the training dataset contains

```text
50,000 crystal structures.
```

If every crystal is processed exactly once,

one epoch has been completed.

This definition is independent of

* model architecture,
* optimizer,
* batch size.

Only the dataset matters.

---

# Why We Do Not Process the Entire Dataset at Once

A natural question arises.

Why not feed all

50,000

crystals into the neural network simultaneously?

The answer is computational.

Consider a realistic MEGNet dataset.

Each crystal graph contains

* atomic features,
* bond features,
* neighbor lists,
* graph connectivity,
* global attributes.

Different crystals contain different numbers of atoms.

Some may have

```text
8 atoms.
```

Others may contain

```text
120 atoms.
```

Processing all graphs simultaneously would require enormous GPU memory.

Instead,

the dataset is divided into **mini-batches**.

---

# Mini-Batch Training

Suppose the batch size equals

```text
32
```

The dataset is divided as

```text
50,000 Crystals

↓

32

↓

32

↓

32

↓

...

↓

Final Batch
```

Each mini-batch contains only a small subset of crystals.

The optimizer updates the parameters after processing each batch.

This approach provides several advantages.

First,

GPU memory requirements become manageable.

Second,

parameter updates occur much more frequently.

Third,

the slight randomness introduced by mini-batches often improves optimization.

Mini-batch gradient descent has therefore become the standard approach for training deep neural networks.

---

# Determining the Number of Mini-Batches

Suppose

the training dataset contains

$$
N
$$

samples,

and the batch size is

$$
B.
$$

The number of batches per epoch is approximately

$$
\frac{N}{B}.
$$

For example,

$$
N=50,000,
$$

$$
B=32.
$$

Then

$$
\frac{50000}{32}
\approx1563.
$$

Thus,

one epoch consists of approximately

1,563

forward and backward passes.

Notice how different this is from the beginner intuition that

> "One epoch means one optimization step."

In reality,

one epoch often contains thousands of optimization steps.

---

# One Mini-Batch Inside MEGNet

Consider one mini-batch containing

32 crystal graphs.

Conceptually,

```text
Batch

├── Crystal 1

├── Crystal 2

├── Crystal 3

├── ...

└── Crystal 32
```

Each graph possesses

* different numbers of atoms,
* different numbers of edges,
* different crystal symmetries.

PyTorch Geometric combines these variable-sized graphs into a single large disconnected graph.

Conceptually,

```text
Large Batch Graph

────────────────────────

Crystal A

(no connections)

Crystal B

(no connections)

Crystal C

(no connections)

────────────────────────
```

Notice that

the crystals are **not connected**.

They simply share one computational graph,

allowing GPU operations to process them efficiently.

This batching strategy is one of the key innovations that makes graph neural networks practical.

---

# Operations Performed for Every Mini-Batch

Every batch undergoes exactly the same sequence of operations.

```text
Mini-Batch

↓

Move to GPU

↓

Forward Pass

↓

Loss Calculation

↓

Gradient Computation

↓

Parameter Update

↓

Next Batch
```

Although this diagram appears simple,

each step contains substantial computation.

In the following sections,

we will analyze every operation individually,

beginning with **moving graph data to the GPU**, followed by the complete forward pass through the MEGNet architecture, gradient computation via backpropagation, optimizer updates, and finally the validation stage.

By the end of this section, you will understand not only **how** to train a MEGNet model, but **why every line of the training loop exists**, enabling you to confidently modify and extend research implementations rather than treating them as black boxes.

## 15.9 Training a Complete MEGNet Model (Part 2)

### Understanding the Training Loop: The Heart of Deep Learning

Every machine learning paper contains a sentence similar to

> "The model was trained for 300 epochs."

Although this statement appears simple,

it hides the most important algorithm in deep learning:

the **training loop**.

Everything that a neural network learns occurs inside this loop.

Whether the model is

* Linear Regression,
* ResNet,
* Transformer,
* CGCNN,
* MEGNet,
* M3GNet,
* CHGNet,

the overall training philosophy remains remarkably similar.

The architecture changes.

The optimizer changes.

The data change.

The training loop remains almost unchanged.

For this reason,

understanding the training loop is one of the most valuable skills in machine learning.

---

# A Bird's-Eye View

Before examining code,

consider the complete process.

```text
Initialize Model

↓

Initialize Optimizer

↓

Initialize Loss Function

↓

for each epoch

    for each mini-batch

        Forward Pass

        Compute Loss

        Backpropagation

        Update Parameters

    Validate

    Save Best Model

    Early Stopping Check

Training Complete
```

Although this algorithm contains only a few steps,

every modern deep learning framework—including PyTorch, TensorFlow, and JAX—implements some variation of this process.

---

# Step 1 — Loading a Mini-Batch

The first operation inside the loop is obtaining a mini-batch from the DataLoader.

```python
for batch in train_loader:
```

This single line performs a surprising amount of work.

The DataLoader

* reads graph objects from memory,
* groups them into a mini-batch,
* creates the batch index,
* constructs edge connectivity,
* prepares tensors for GPU computation.

For PyTorch Geometric,

the variable

```python
batch
```

is not a simple tensor.

Instead,

it is a **Batch object** containing several tensors.

Conceptually,

```text
Batch

├── x

├── edge_index

├── edge_attr

├── batch

├── global_features

└── y
```

Each of these tensors serves a specific purpose.

---

# Understanding Each Tensor

Let us examine them individually.

### Node Features

```python
batch.x
```

contains the feature vector of every atom.

Suppose three crystals are batched together.

```text
Crystal A

3 atoms

Crystal B

5 atoms

Crystal C

4 atoms
```

The batch contains

```text
12 atoms
```

Therefore,

```python
batch.x
```

has shape

$$
(12,F_n)
$$

where

$$
F_n
$$

is the number of node features.

Notice that

PyTorch Geometric stores

**all atoms together**

rather than storing each crystal separately.

---

### Edge Connectivity

```python
batch.edge_index
```

contains the graph connectivity.

Its shape is

$$
(2,E)
$$

where

$$
E
$$

is the number of edges.

Each column represents one bond.

For example,

```text
0 → 1

1 → 2

2 → 0
```

means

Atom 0 sends a message to Atom 1,

Atom 1 sends a message to Atom 2,

Atom 2 sends a message back to Atom 0.

---

### Edge Features

```python
batch.edge_attr
```

contains bond information.

Typical edge descriptors include

* bond distance,
* bond type,
* Gaussian basis expansion,
* relative position vectors.

If there are

400 edges,

then

$$
batch.edge_attr
\in
\mathbb{R}^{400\times F_e}
$$

where

$$
F_e
$$

is the edge feature dimension.

---

### Batch Index

Perhaps the most mysterious tensor is

```python
batch.batch
```

Many beginners ignore this variable,

yet it is one of the most important tensors in graph neural networks.

Recall that

PyTorch Geometric combines multiple graphs into one large disconnected graph.

The tensor

```python
batch.batch
```

records

which atom belongs to which crystal.

For example,

```text
Atom

0

1

2

3

4

5

6

7
```

may correspond to

```text
Batch Index

0

0

0

1

1

1

2

2
```

This tells the readout layer

Atoms

0–2

belong to Crystal 0,

Atoms

3–5

belong to Crystal 1,

Atoms

6–7

belong to Crystal 2.

Without this tensor,

graph pooling would be impossible.

---

### Target Values

Finally,

```python
batch.y
```

contains the target property.

For formation energy prediction,

it may look like

```text
[-3.24,

-1.87,

-2.41,

...]

```

Its shape is

$$
(B,1)
$$

where

$$
B
$$

is the batch size.

---

# Step 2 — Moving Data to the GPU

Modern graph neural networks are almost always trained on GPUs.

Before computation,

every tensor must be transferred from CPU memory.

The standard PyTorch command is

```python
batch = batch.to(device)
```

where

```python
device
```

is typically

```python
device = torch.device(

    "cuda"

)
```

if a GPU is available,

or

```python
device = torch.device(

    "cpu"

)
```

otherwise.

This command moves

* node features,
* edge indices,
* edge attributes,
* targets,
* batch indices,

simultaneously.

After this operation,

all computations occur on the GPU.

---

# Why GPUs Matter

Consider a MEGNet model containing

2 million parameters.

Training on a CPU may require

hours

or even

days.

A modern GPU performs

thousands of floating-point operations simultaneously.

This parallelism dramatically accelerates

* matrix multiplication,
* message passing,
* gradient computation.

For large materials datasets,

GPU acceleration is not merely convenient—

it is practically essential.

---

# Step 3 — Resetting the Gradients

Before computing new gradients,

the old gradients must be removed.

This is done using

```python
optimizer.zero_grad()
```

This line often confuses beginners.

Why must gradients be reset?

The answer lies in how PyTorch computes derivatives.

Unlike many mathematical descriptions,

PyTorch **accumulates gradients** by default.

Suppose

after one batch,

the gradient equals

$$
0.8.
$$

If gradients are not cleared,

the next batch may produce

$$
0.6.
$$

The stored value becomes

$$
1.4,
$$

rather than

$$
0.6.
$$

After hundreds of batches,

the accumulated gradients become meaningless.

Therefore,

every iteration begins by clearing the previous gradients.

Conceptually,

```text
Previous Batch

↓

Gradient = 0

↓

Forward Pass

↓

Backward Pass

↓

New Gradient
```

This ensures that every optimization step depends only on the current mini-batch.

---

# Summary of the First Three Steps

At this point,

every iteration of the training loop has performed three essential operations:

1. **Load one mini-batch** from the DataLoader.
2. **Move the graph data to the computation device** (GPU or CPU).
3. **Clear any previously accumulated gradients**.

Only after these preparation steps is the model ready to perform its first real computation:

the **forward pass**.

The forward pass is where the crystal graph enters the MEGNet architecture, messages propagate through the graph, node, edge, and global features are updated, and the network ultimately predicts a material property.

In the next section, we will dissect the forward pass line by line, tracing a single batch of crystal graphs through every stage of the MEGNet model until the final prediction is produced.

## 15.9 Training a Complete MEGNet Model (Part 3)

### Step 4 — The Forward Pass: How MEGNet Makes Predictions

The preparation stage of the training loop is now complete.

For every mini-batch, we have

* loaded the crystal graphs,
* moved them to the GPU,
* cleared the previous gradients.

Now comes the most important computation in the entire neural network:

> **The forward pass.**

The forward pass is where the neural network transforms a crystal structure into a predicted material property.

Everything that we have studied throughout this chapter—including

* graph construction,
* embeddings,
* edge updates,
* node updates,
* global state updates,
* graph pooling,

is executed during this stage.

---

# What Is the Forward Pass?

Mathematically,

the forward pass is simply the evaluation of the neural network function

$$
\hat y=f(G;\theta),
$$

where

* $G$ is the crystal graph,
* $\theta$ represents all trainable parameters,
* $\hat y$ is the predicted property.

Although this equation appears compact,

the computation hidden inside

$$
f(G;\theta)
$$

is extremely complex.

A single forward pass through MEGNet may involve

* millions of matrix multiplications,
* nonlinear activation functions,
* message passing,
* aggregation operations,
* pooling,
* fully connected layers.

Yet PyTorch allows all of this to be performed using a single line.

---

# The Entire Forward Pass in PyTorch

The complete forward pass usually looks like

```python
prediction = model(batch)
```

This single statement triggers every operation defined inside

```python
model.forward()
```

To a beginner,

this may appear almost magical.

However,

what actually happens is a carefully organized sequence of computations.

---

# Looking Inside `model.forward()`

Conceptually,

the forward function might resemble

```python
def forward(self, batch):

    node_features = self.node_embedding(batch.x)

    edge_features = self.edge_embedding(batch.edge_attr)

    global_features = self.global_embedding(batch.u)

    node_features, edge_features, global_features = self.megnet_block(

        node_features,

        edge_features,

        global_features,

        batch.edge_index,

        batch.batch

    )

    graph_embedding = self.readout(

        node_features,

        batch.batch

    )

    prediction = self.predictor(

        graph_embedding

    )

    return prediction
```

This is a simplified version,

but it illustrates the complete computational pipeline.

Every line corresponds to concepts developed earlier in this chapter.

---

# Step 4.1 — Embedding the Input Features

Recall that the raw graph contains

* atomic numbers,
* bond distances,
* global descriptors.

These quantities are meaningful to humans,

but they are not immediately suitable for deep learning.

The first task is therefore to transform them into continuous vector representations.

For atoms,

we compute

$$
\mathbf{h}_v^{(0)}
==================

E_v(\mathbf{x}_v),
$$

where

* $\mathbf{x}_v$ is the original atomic feature vector,
* $E_v$ is the node embedding network.

Similarly,

edge features become

$$
\mathbf{e}_{ij}^{(0)}
=====================

E_e(\mathbf{x}_{ij}),
$$

and global features become

$$
\mathbf{u}^{(0)}
================

E_u(\mathbf{x}_u).
$$

After embedding,

all three feature types share a common latent space.

This makes message passing much more effective.

---

# Step 4.2 — Passing Through the MEGNet Blocks

Next,

the embedded graph enters the first Graph Network block.

Conceptually,

```text
Embedded Graph

↓

MEGNet Block 1

↓

Updated Graph

↓

MEGNet Block 2

↓

Updated Graph

↓

...

↓

Final Graph Representation
```

Each block performs three operations.

---

### Edge Update

Each bond gathers information from

* the source atom,
* the destination atom,
* the previous edge representation,
* the global state.

Mathematically,

$$
\mathbf{e}_{ij}^{(t+1)}
=======================

\phi_e
\left(
\mathbf{h}_i^{(t)},
\mathbf{h}*j^{(t)},
\mathbf{e}*{ij}^{(t)},
\mathbf{u}^{(t)}
\right).
$$

The edge representation now contains richer chemical information.

---

### Node Update

Every atom collects messages from neighboring bonds.

The aggregated message is

$$
\mathbf{m}_i
============

\sum_{j\in\mathcal N(i)}
\mathbf{e}_{ij}.
$$

The node representation is updated as

$$
\mathbf{h}_i^{(t+1)}
====================

\phi_v
\left(
\mathbf{h}_i^{(t)},
\mathbf{m}_i,
\mathbf{u}^{(t)}
\right).
$$

Each atom therefore acquires information about its local chemical environment.

---

### Global Update

Finally,

the entire crystal representation is updated.

The global state combines information from

* all nodes,
* all edges,
* the previous global state.

Conceptually,

$$
\mathbf{u}^{(t+1)}
==================

\phi_u
(
\mathbf{u}^{(t)},
\bar{\mathbf h},
\bar{\mathbf e}
).
$$

At this point,

every part of the graph has exchanged information.

---

# Repeating the Process

One MEGNet block rarely provides sufficient expressive power.

Instead,

multiple blocks are stacked.

Suppose the model contains four blocks.

The information flow becomes

```text
Initial Graph

↓

Block 1

↓

Block 2

↓

Block 3

↓

Block 4

↓

Final Graph
```

Every block allows information to propagate farther through the crystal.

After several iterations,

an atom may indirectly receive information from distant regions of the structure.

This progressive information exchange is one of the defining characteristics of graph neural networks.

---

# Step 4.3 — Readout (Pooling)

After message passing,

every atom possesses its own learned representation.

Suppose the graph contains

100 atoms.

We therefore have

100 feature vectors.

However,

the prediction task concerns

**the crystal**,

not individual atoms.

We therefore require a fixed-length graph representation.

This is achieved through the readout layer.

Conceptually,

```text
Atom Features

↓

Pooling

↓

Graph Embedding
```

Mathematically,

the graph embedding is

$$
\mathbf{g}
==========

R
(
\mathbf{h}_1,
\mathbf{h}_2,
\ldots,
\mathbf{h}_N
),
$$

where

$R$

is a permutation-invariant pooling operator,

such as

* sum,
* mean,
* or set2set pooling.

Regardless of whether the crystal contains

10 atoms

or

500 atoms,

the output always has the same dimension.

This property allows graphs of different sizes to be processed by the same neural network.

---

# Step 4.4 — Prediction Head

The graph embedding is now passed through a multilayer perceptron.

Conceptually,

```text
Graph Embedding

↓

Linear Layer

↓

ReLU

↓

Dropout

↓

Linear Layer

↓

Predicted Property
```

The mathematical operation can be written as

$$
\hat y
======

f_{\text{MLP}}
(
\mathbf g
).
$$

Depending on the task,

the output may represent

* formation energy,
* band gap,
* bulk modulus,
* elastic tensor component,
* dielectric constant,
* magnetic moment,

or any other graph-level material property.

---

# The Output Tensor

Suppose the batch size equals

32.

The forward pass returns

```python
prediction = model(batch)
```

The shape of

```python
prediction
```

is

$$
(32,1)
$$

Each row corresponds to one crystal.

For example,

```text
Crystal 1 → -2.34 eV

Crystal 2 → -1.67 eV

Crystal 3 → -3.02 eV

...

Crystal 32 → -2.48 eV
```

Notice that the model predicts all crystals simultaneously.

This parallel computation is one of the reasons GPUs are so effective for deep learning.

---

# Why the Forward Pass Is Differentiable

Every operation inside the forward pass—

* linear layers,
* message passing,
* pooling,
* activation functions,

is differentiable.

Consequently,

PyTorch automatically constructs a **computational graph** while the forward pass executes.

This graph records every mathematical operation.

It is **not** the crystal graph.

Instead,

it is a graph describing the sequence of computations required to produce the prediction.

This computational graph is what makes automatic differentiation possible.

Without it,

backpropagation could not compute parameter gradients.

---

# Forward Pass vs. Inference

An important distinction should be made.

The forward pass occurs both during

* training,
* and inference.

The computation itself is almost identical.

The difference lies in what happens **after** the forward pass.

During training,

the prediction is compared with the true target,

the loss is computed,

and gradients are propagated backward.

During inference,

no gradients are calculated.

The prediction is simply returned to the user.

Thus,

the forward pass is the common foundation of both training and prediction.

---

# Summary

At this point in every training iteration,

MEGNet has successfully transformed a batch of crystal structures into predicted material properties.

The next question naturally follows:

> **How does the neural network know whether these predictions are good or bad?**

To answer this,

we must compare the predictions with the true material properties using a **loss function**.

The loss converts prediction errors into a single numerical quantity that tells the optimizer how well the model is performing.

In the next section, we will study **loss computation in MEGNet**, derive the most commonly used regression loss functions (MAE, MSE, and Huber loss), explain when each should be used in materials informatics, and show exactly how they are implemented in PyTorch.

## 15.9 Training a Complete MEGNet Model (Part 4)

### Step 5 — Computing the Loss: Measuring Prediction Error

The forward pass has now produced predictions for every crystal in the mini-batch.

Suppose the model predicts formation energies.

For one mini-batch, the predictions may be

```text
Crystal          Predicted Formation Energy (eV)

1                         -2.15

2                         -1.84

3                         -3.47

...

32                        -2.73
```

However,

these numbers alone tell us nothing.

Are these predictions good?

Are they terrible?

Should the neural network change its parameters?

The neural network itself has no intuition about chemistry.

It does not know whether

* a prediction error of 0.01 eV is excellent,
* a prediction error of 2 eV is disastrous.

It needs a mathematical measure of performance.

That measure is called the **loss function**.

---

# Why a Loss Function Is Necessary

Imagine teaching a student to solve mathematics problems.

After every answer,

you compare the student's solution with the correct answer.

If the answer is wrong,

you explain the mistake.

The student gradually improves.

Without feedback,

learning would be impossible.

A neural network learns in exactly the same way.

The true material properties serve as the "correct answers."

The predictions are the student's answers.

The loss function measures the disagreement between them.

The optimizer then uses this information to improve the model.

Thus,

the loss function acts as the **teacher** of the neural network.

---

# Predictions and Targets

Suppose one mini-batch contains four crystals.

The predictions are

$$
\hat{\mathbf y}
=

[-2.41,,-1.82,,-3.05,,-0.91].
$$

The true DFT values are

$$
\mathbf y
=

[-2.35,,-1.90,,-2.96,,-1.02].
$$

Clearly,

the predictions are not identical.

The question is

**How should we measure the difference?**

Many possibilities exist.

Some penalize large errors heavily.

Others treat all errors equally.

Choosing an appropriate loss function is therefore an important modeling decision.

---

# Requirements of a Good Loss Function

A useful loss function should satisfy several properties.

It should

* produce small values for accurate predictions,
* produce large values for inaccurate predictions,
* be differentiable,
* provide meaningful gradients,
* be computationally efficient.

If these conditions are satisfied,

gradient-based optimization becomes possible.

---

# Regression Versus Classification

Before discussing specific loss functions,

it is useful to distinguish between two major categories of machine learning.

### Classification

Classification predicts categories.

Examples include

* metal vs insulator,
* magnetic vs non-magnetic,
* stable vs unstable.

Typical outputs are

```text
Class A

Class B

Class C
```

Classification commonly uses

* Cross Entropy Loss,
* Binary Cross Entropy.

---

### Regression

MEGNet is usually used for **regression**.

Regression predicts continuous numerical values.

Examples include

* formation energy,
* band gap,
* elastic modulus,
* lattice constant,
* dielectric constant,
* heat capacity.

These quantities are real numbers.

Consequently,

MEGNet generally employs regression loss functions.

---

# The Prediction Error

The simplest quantity is the prediction error.

For one sample,

$$
e
=

\hat y-y.
$$

Suppose

True formation energy

$$-2.50\text{ eV}.$$

Prediction

$$

-2.20
\text{ eV}.
$$

Then

$$
e
=
 -2.20-(-2.50)

0.30
\text{ eV}.
$$

The prediction is

0.30 eV

too high.

---

Suppose instead

Prediction

$$

-2.80
\text{ eV}.
$$

Then

$$
e
=

 -2.80-(-2.50)

-0.30
\text{ eV}.
$$

Now the prediction is

0.30 eV

too low.

Notice something important.

Both predictions are equally inaccurate.

Yet

their errors have opposite signs.

Therefore,

we cannot simply average

$$
e.
$$

Positive and negative errors would cancel each other.

---

# Removing the Sign

There are two common ways to eliminate this cancellation.

The first is

the **absolute value**

$$
|e|.
$$

The second is

the **square**

$$
e^2.
$$

These lead to the two most widely used regression loss functions.

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

We will study each individually.

---

# Mean Absolute Error (MAE)

The absolute error for one prediction is

$$
|y-\hat y|.
$$

For an entire dataset containing

$$
N
$$

samples,

the Mean Absolute Error is

$$
\boxed{
MAE
===

\frac1N
\sum_{i=1}^{N}
|y_i-\hat y_i|
}
$$

This is one of the most popular evaluation metrics in materials informatics.

---

### Example

Suppose

| True | Predicted |
| ---: | --------: |
| -2.5 |      -2.2 |
| -1.8 |      -1.9 |
| -3.1 |      -2.8 |

Absolute errors become

| Error |
| ----: |
|   0.3 |
|   0.1 |
|   0.3 |

Therefore,

$$
MAE
=

\frac{0.3+0.1+0.3}{3}

0.233
\text{ eV}.
$$

This means

the average prediction error is approximately

0.233 eV.

---

# Why Researchers Like MAE

MAE has several attractive properties.

First,

it has the same physical units as the target.

For formation energy,

the MAE is measured in

eV.

For bulk modulus,

it is measured in

GPa.

For band gap,

again,

eV.

This makes interpretation straightforward.

If a paper reports

```text
MAE = 0.03 eV
```

the meaning is immediately clear.

---

Second,

every error contributes linearly.

A prediction error of

2 eV

is treated as twice as bad as

1 eV.

This makes MAE relatively robust against occasional outliers.

---

# Limitations of MAE

Although MAE is easy to interpret,

its mathematical properties are not ideal.

The absolute value function is not differentiable at

$$
e=0.
$$

Although modern optimization algorithms can still handle this,

gradient-based methods generally prefer smoother functions.

This motivates another important loss function.

---

# Mean Squared Error (MSE)

Instead of taking the absolute value,

we square each error.

The Mean Squared Error becomes

$$
\boxed{
MSE
===

\frac1N
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}
$$

Notice the difference.

Large errors now receive much larger penalties.

---

### Example

Consider three prediction errors.

```text
0.1

0.2

1.0
```

For MAE,

the contributions are

```text
0.1

0.2

1.0
```

For MSE,

the contributions become

```text
0.01

0.04

1.00
```

Observe that

the largest error now dominates the loss.

MSE therefore strongly encourages the optimizer to eliminate large prediction errors.

---

# Why MSE Is So Popular

MSE possesses several important mathematical advantages.

Its derivative is continuous.

The gradient is smooth.

Optimization becomes more stable.

Consequently,

MSE has historically become one of the most widely used loss functions for regression neural networks.

Many early graph neural networks,

including some MEGNet implementations,

used MSE during training.

---

# A Trade-Off Between MAE and MSE

At this point,

we can compare the two losses.

| Property                | MAE             | MSE            |
| ----------------------- | --------------- | -------------- |
| Interpretation          | Excellent       | Less intuitive |
| Units                   | Same as target  | Squared units  |
| Sensitivity to Outliers | Low             | High           |
| Gradient Smoothness     | Lower           | Higher         |
| Optimization            | Slightly harder | Easier         |

Neither loss is universally superior.

The choice depends on the problem being solved.

---

# Loss Function vs Evaluation Metric

An important distinction should now be emphasized.

Many papers state

> "The model was trained using MSE loss and evaluated using MAE."

This is **not** a contradiction.

Training and evaluation serve different purposes.

The optimizer prefers MSE because of its smooth gradients.

Researchers prefer MAE because it is easier to interpret physically.

Thus,

it is common to

* optimize MSE,
* report MAE.

Understanding this distinction helps explain many materials informatics publications.

---

At this point, we have introduced the two most fundamental regression loss functions.

However, modern materials informatics increasingly employs another loss that combines the strengths of both:

**Huber Loss (Smooth L1 Loss).**

Huber loss behaves like MSE for small errors, providing smooth optimization, while behaving like MAE for large errors, making it much more robust to outliers.

In the next section, we will derive the Huber loss mathematically, explain why it is often preferred for noisy materials datasets, visualize its behavior, and implement it in PyTorch for MEGNet training.

## 15.9 Training a Complete MEGNet Model (Part 5)

### Huber Loss: Combining the Advantages of MAE and MSE

In the previous section, we introduced the two most fundamental regression loss functions:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

Each possesses important strengths.

Each also has important weaknesses.

Before proceeding further, let us briefly summarize what we have learned.

---

### Mean Absolute Error

MAE computes

$$
MAE
===

\frac1N
\sum_{i=1}^{N}
|y_i-\hat y_i|.
$$

Advantages

* Easy to interpret
* Same units as the target
* Robust against outliers

Disadvantages

* Absolute value is not perfectly smooth
* Optimization is somewhat more difficult

---

### Mean Squared Error

MSE computes

$$
MSE
===

\frac1N
\sum_{i=1}^{N}
(y_i-\hat y_i)^2.
$$

Advantages

* Smooth derivatives
* Stable optimization
* Excellent for gradient descent

Disadvantages

* Extremely sensitive to large errors
* Outliers dominate the loss

Neither loss is perfect.

This naturally raises an important question.

> **Can we combine the strengths of both MAE and MSE into a single loss function?**

The answer is yes.

That loss function is called the **Huber Loss**.

It is also known in PyTorch as **Smooth L1 Loss**.

---

# Motivation

Suppose we are training MEGNet on a dataset of formation energies.

Most DFT calculations are highly reliable.

However,

a few structures contain

* unconverged calculations,
* incorrect magnetic states,
* numerical instabilities,
* data entry errors.

These samples produce unusually large prediction errors.

Suppose the prediction errors are

```text
0.02

0.04

0.03

0.05

2.15
```

Notice that

the final sample is dramatically different from the others.

---

### What Happens with MSE?

Squaring these errors gives

```text
0.0004

0.0016

0.0009

0.0025

4.6225
```

The final sample dominates the optimization.

The neural network spends much of its effort attempting to fit this one abnormal example.

This may actually reduce generalization.

---

### What Happens with MAE?

Using MAE,

the errors become

```text
0.02

0.04

0.03

0.05

2.15
```

Now,

the outlier still contributes,

but it no longer overwhelms the optimization.

Unfortunately,

we lose the smooth optimization properties of MSE.

---

# The Idea Behind Huber Loss

Huber loss combines these two behaviors.

The idea is remarkably elegant.

For **small errors**,

use the squared loss

because it is smooth.

For **large errors**,

switch to the absolute loss

because it is robust.

Thus,

the optimizer enjoys the best characteristics of both methods.

---

# Mathematical Definition

Define the prediction error

$$
e
=

y-\hat y.
$$

Choose a threshold

$$
\delta.
$$

The Huber loss is defined as

$$
\boxed{
L(e)=
\begin{cases}
\frac12e^2,
&
|e|\le\delta,
[0.4cm]
\delta
\left(
|e|-\frac12\delta
\right),
&
|e|>\delta.
\end{cases}
}
$$

This equation deserves careful examination.

It consists of

**two different mathematical expressions**.

The loss automatically chooses between them depending on the prediction error.

---

# Region 1 — Small Errors

Suppose

$$
|e|
<
\delta.
$$

The loss becomes

$$
L(e)
====

\frac12e^2.
$$

This is essentially MSE.

Near the correct solution,

optimization behaves exactly like squared error minimization.

The gradients remain smooth,

allowing efficient optimization.

---

### Example

Suppose

$$
\delta=1.
$$

Prediction error

$$
e=0.3.
$$

Since

$$
0.3<1,
$$

Huber loss becomes

$$
L
=

\frac12
(0.3)^2
=======

0.045.
$$

The behavior is almost identical to MSE.

---

# Region 2 — Large Errors

Now suppose

$$
|e|

>

\delta.
$$

Instead of continuing to square the error,

Huber loss changes to

$$
L(e)
====

\delta
\left(
|e|-\frac12\delta
\right).
$$

Notice something remarkable.

The loss now grows

**linearly**,

just like MAE.

Large prediction errors therefore cannot dominate the optimization.

---

### Example

Again,

let

$$
\delta=1.
$$

Suppose

$$
e=3.
$$

Then

$$
L
=

1
\left(
3-\frac12
\right)
=======

2.5.
$$

Compare this with MSE.

For MSE,

$$
3^2
===

9.

$$

Huber loss

$$

2.5.
$$

The outlier is still penalized,

but much less aggressively.

---

# Visualizing the Three Loss Functions

Conceptually,

the three losses behave differently.

```text
Prediction Error

↓

MAE

Linear

────────────

MSE

Quadratic

╭───────╮

Huber

Quadratic

↓

Linear
```

Huber loss starts like MSE.

After reaching

$$
\delta,
$$

it gradually transitions into MAE.

This smooth transition is the key to its success.

---

# Why Is the Transition Smooth?

An important design goal of Huber loss is that

there should be **no sudden jump** at

$$
|e|
===

\delta.
$$

The two equations are carefully constructed so that

* the loss is continuous,
* the gradient is continuous.

This smooth transition makes optimization stable while still protecting against outliers.

This is one reason why Huber loss is widely used in modern deep learning.

---

# Choosing the Threshold

The parameter

$$
\delta
$$

controls

where the transition occurs.

Suppose

$$
\delta
======

0.5.
$$

Then

all errors larger than

0.5

behave like MAE.

Now suppose

$$
\delta
======

5.

$$

Almost every prediction behaves like MSE.

Thus,

the threshold determines how sensitive the loss is to outliers.

---

# Typical Values

Many machine learning libraries use

```text
δ = 1
```

as the default.

For normalized regression problems,

this often performs well.

However,

materials informatics frequently predicts quantities with very different numerical scales.

For example,

| Property          | Typical Magnitude |
| ----------------- | ----------------: |
| Band gap          |            0–8 eV |
| Formation energy  |       −10 to 5 eV |
| Bulk modulus      |        10–400 GPa |
| Elastic constants |   Hundreds of GPa |

Consequently,

the optimal

$$
\delta
$$

should be selected according to the scale of the target property.

---

# PyTorch Implementation

PyTorch implements Huber loss through

```python
criterion = torch.nn.HuberLoss(

    delta=1.0

)
```

or equivalently,

```python
criterion = torch.nn.SmoothL1Loss()
```

Training then proceeds exactly as before.

```python
loss = criterion(

    prediction,

    target

)
```

From the user's perspective,

changing from MSE to Huber loss requires only one line of code.

Internally,

however,

the optimization behavior changes substantially.

---

# Which Loss Should You Use for MEGNet?

There is no universal answer.

The best choice depends on

* dataset quality,
* target property,
* presence of outliers,
* numerical noise.

Nevertheless,

general recommendations can be made.

| Situation                        | Recommended Loss |
| -------------------------------- | ---------------- |
| Very clean DFT dataset           | MSE              |
| Dataset with occasional outliers | Huber            |
| Reporting prediction accuracy    | MAE              |
| Noisy experimental measurements  | Huber            |

Many recent materials informatics studies prefer Huber loss because real datasets often contain

* noisy measurements,
* imperfect DFT calculations,
* inconsistent experimental values.

Huber loss provides a good balance between optimization stability and robustness.

---

# Computing the Loss in the Training Loop

Regardless of which loss function is selected,

the training loop remains almost identical.

```python
prediction = model(batch)

loss = criterion(

    prediction,

    batch.y

)
```

At this moment,

the neural network has completed its forward computation.

The loss is now a **single scalar value**.

For example,

```text
Loss = 0.0247
```

This number summarizes the prediction quality of the entire mini-batch.

However,

the optimizer still does not know

**how each parameter contributed to this error**.

To improve the model,

it must determine how changing every weight would affect the loss.

This is the purpose of **backpropagation**.

Backpropagation is arguably the most important algorithm in modern deep learning.

It computes the gradient of the loss with respect to **every trainable parameter** in the MEGNet model, enabling gradient-based optimization.

In the next section, we will study backpropagation from first principles, beginning with the chain rule of calculus and progressing to PyTorch's automatic differentiation system (`autograd`), showing exactly how gradients flow backward through every MEGNet block.

## 15.9 Training a Complete MEGNet Model (Part 6)

# Step 6 — Backpropagation: How MEGNet Learns from Its Mistakes

At this point in the training loop, something remarkable has happened.

The neural network has

* received a batch of crystal graphs,
* converted them into graph representations,
* predicted material properties,
* computed the prediction loss.

Suppose the loss is

$$
L=0.026.
$$

The model now knows

> "I am wrong."

However,

this information alone is not enough.

The optimizer needs to answer a much more difficult question:

> **Which of the millions of parameters caused this error?**

More importantly,

> **How should each parameter change to reduce the loss?**

Answering these questions is the purpose of **backpropagation**.

Backpropagation is the algorithm that transforms prediction errors into learning.

Without backpropagation,

deep learning would not exist.

---

# Learning Requires Blame Assignment

Imagine a research group publishing an incorrect scientific result.

The paper contains contributions from

* the experimental team,
* the computational team,
* the data analysts,
* the principal investigator.

The final result is incorrect.

Who is responsible?

Perhaps

* the experiment contained measurement errors,
* the simulation used an incorrect potential,
* the data preprocessing introduced mistakes,
* or the statistical analysis was flawed.

To improve future work,

we must determine

**how much responsibility belongs to each contributor.**

Backpropagation performs exactly this task for neural networks.

Instead of scientists,

it analyzes

* weights,
* biases,
* embedding layers,
* message passing functions,
* prediction layers.

Every parameter receives its share of responsibility for the prediction error.

---

# The Objective of Backpropagation

Suppose the MEGNet model contains

two million trainable parameters.

Denote them collectively by

$$
\theta.
$$

The optimizer wishes to minimize

$$
L(\theta).
$$

To accomplish this,

it must determine

how sensitive the loss is to every parameter.

Mathematically,

it computes

$$
\frac{\partial L}{\partial\theta}.
$$

This quantity is called the **gradient**.

The gradient tells us

how much the loss changes

if a parameter changes slightly.

---

# Understanding a Gradient

Consider a simple function

$$
L(w)=w^2.
$$

Its derivative is

$$
\frac{dL}{dw}=2w.
$$

Suppose

$$
w=5.
$$

Then

$$
\frac{dL}{dw}=10.
$$

A positive gradient means

increasing

$$
w
$$

increases the loss.

Therefore,

the optimizer should decrease

$$
w.
$$

Now suppose

$$
w=-4.
$$

The gradient becomes

$$
-8.
$$

A negative gradient means

increasing

$$
w
$$

reduces the loss.

The optimizer therefore moves in the opposite direction.

This simple example illustrates the fundamental idea behind gradient descent.

---

# Gradients in MEGNet

Of course,

MEGNet is far more complicated than

$$
w^2.
$$

The loss depends on

millions of parameters simultaneously.

These include

* embedding weights,
* edge update weights,
* node update weights,
* global update weights,
* linear layer weights,
* bias parameters.

Conceptually,

```text
Loss

↑

Prediction Layer

↑

Readout

↑

MEGNet Block

↑

Embedding Layer

↑

Input Graph
```

Every layer contributes to the final prediction.

Therefore,

every layer contributes to the loss.

Backpropagation computes gradients for **all** of them.

---

# Why It Is Called Backpropagation

Recall the direction of information flow during the forward pass.

```text
Crystal

↓

Embedding

↓

MEGNet Block

↓

Readout

↓

Prediction

↓

Loss
```

Information moves

**forward**

through the network.

Backpropagation reverses this direction.

```text
Loss

↓

Prediction Layer

↓

Readout

↓

MEGNet Block

↓

Embedding

↓

Parameters
```

The prediction error travels backward,

gradually determining how every parameter contributed to the final mistake.

Hence the name

**back-propagation**.

---

# The Chain Rule

The mathematical foundation of backpropagation is

the **chain rule** from calculus.

Suppose

$$
L=f(g(w)).
$$

The derivative is

$$
\boxed{
\frac{dL}{dw}
=============

\frac{dL}{dg}
\cdot
\frac{dg}{dw}
}
$$

Instead of computing one enormous derivative,

the chain rule decomposes it into a sequence of simpler derivatives.

Deep neural networks are essentially gigantic compositions of functions.

Consequently,

the chain rule provides an efficient method for computing gradients.

---

# Applying the Chain Rule to MEGNet

Recall the computational pipeline

```text
Node Features

↓

Embedding

↓

Message Passing

↓

Pooling

↓

Prediction Head

↓

Loss
```

The loss depends on

the prediction.

The prediction depends on

the pooled graph representation.

The pooled representation depends on

the node embeddings.

The node embeddings depend on

the message-passing layers.

The message-passing layers depend on

the original parameters.

The chain rule connects all of these relationships.

Rather than treating the network as one enormous equation,

backpropagation computes gradients one layer at a time.

---

# A Simple Example

Suppose

$$
y=2x.
$$

Then

$$
z=y+3.
$$

Finally,

$$
L=z^2.
$$

The computation proceeds

```text
x

↓

y = 2x

↓

z = y + 3

↓

L = z²
```

Using the chain rule,

$$
\frac{dL}{dx}
=============

\frac{dL}{dz}
\cdot
\frac{dz}{dy}
\cdot
\frac{dy}{dx}.
$$

Each derivative is simple.

Together,

they produce the complete gradient.

Deep neural networks follow exactly the same principle,

only with millions of intermediate operations instead of three.

---

# Why Manual Differentiation Is Impossible

Imagine attempting to compute gradients manually for MEGNet.

The network may contain

* 30 linear layers,
* multiple Graph Network blocks,
* activation functions,
* pooling operations,
* normalization layers.

The resulting derivative would occupy hundreds of pages.

Even a small programming error would invalidate the calculation.

Fortunately,

modern deep learning frameworks perform this differentiation automatically.

---

# PyTorch Autograd

PyTorch includes an automatic differentiation engine called

```text
Autograd
```

During the forward pass,

Autograd records

every differentiable operation.

Conceptually,

```text
Forward Pass

↓

Operation 1

↓

Operation 2

↓

Operation 3

↓

Loss

↓

Computational Graph
```

This computational graph stores

* tensor operations,
* dependencies,
* required gradients.

When backpropagation begins,

Autograd traverses this graph in reverse order,

applying the chain rule automatically.

---

# Triggering Backpropagation

The remarkable aspect of PyTorch is

that the entire gradient computation requires only one command.

```python
loss.backward()
```

This single line computes

the gradient of the loss

with respect to

**every trainable parameter**.

No explicit derivative calculations are required.

Internally,

PyTorch performs millions of mathematical operations,

but the user sees only one function call.

---

# What Happens After `loss.backward()`?

Suppose the model contains

```python
self.linear.weight
```

After

```python
loss.backward()
```

its gradient becomes available.

```python
print(

    self.linear.weight.grad

)
```

Every trainable parameter now possesses a corresponding gradient.

Conceptually,

```text
Weight Matrix

↓

Gradient Matrix
```

These gradients are **not** new parameter values.

Instead,

they describe

how the parameters should change

to reduce the loss.

The optimizer will use these gradients in the next step.

---

# Important Observation

Notice that

```python
loss.backward()
```

does **not**

change the neural network.

It merely computes gradients.

The parameters remain exactly the same.

Learning has not yet occurred.

The actual learning happens during the optimizer update.

The optimizer reads the gradients,

calculates new parameter values,

and modifies the neural network accordingly.

Only after this update has the model actually learned from the current mini-batch.

---

# Summary

At this stage of the training loop,

we have

1. performed the forward pass,
2. computed the prediction loss,
3. calculated gradients for every trainable parameter.

The neural network now knows **how every parameter should change** to reduce the prediction error.

However, the parameters themselves remain unchanged.

The final step is to use these gradients to update the model.

This is the responsibility of the **optimizer**.

In the next section, we will study the optimizer update in depth, derive the gradient descent equation, explain how Adam and AdamW modify the standard algorithm, and trace exactly how millions of MEGNet parameters are updated after every mini-batch.

## 15.9 Training a Complete MEGNet Model (Part 7)

# Step 7 — Optimizer Step: Updating the Neural Network Parameters

At the end of the previous section, every trainable parameter in the MEGNet model has an associated gradient.

For example,

```python
embedding.weight.grad
edge_update.weight.grad
node_update.weight.grad
global_update.weight.grad
predictor.weight.grad
```

Each gradient answers the following question:

> **"If this parameter changes slightly, how will the loss change?"**

However, the neural network has **not learned anything yet**.

The gradients merely provide directions.

Someone still needs to use those directions to modify the parameters.

That "someone" is the **optimizer**.

The optimizer is the component responsible for transforming gradients into actual learning.

---

# From Gradients to Learning

Suppose a parameter has the value

$$
w = 2.50.
$$

After backpropagation,

its gradient is

$$
\frac{\partial L}{\partial w}=0.80.
$$

The optimizer now asks

> "Should I increase this parameter or decrease it?"

Since the gradient is positive,

increasing

$$
w
$$

would increase the loss.

Therefore,

the optimizer decreases

$$
w.
$$

The new parameter becomes

$$
w_{\text{new}}
==============

## 2.50

\eta
\times
0.80,
$$

where

$$
\eta
$$

is the learning rate.

This is the fundamental idea behind **gradient descent**.

---

# The Gradient Descent Equation

The simplest optimization algorithm is

Gradient Descent.

Its update equation is

$$
\boxed{
\theta_{\text{new}}
===================

## \theta_{\text{old}}

\eta
\nabla_{\theta}L
}
$$

This equation is one of the most important equations in machine learning.

Let us examine each component carefully.

---

### Parameters

The symbol

$$
\theta
$$

represents

**all trainable parameters**

inside MEGNet.

These include

* embedding matrices,
* neural network weights,
* bias vectors,
* update function parameters,
* prediction layer weights.

Instead of updating one parameter,

the optimizer updates **millions** of parameters simultaneously.

---

### Learning Rate

The symbol

$$
\eta
$$

(pronounced "eta")

is the learning rate.

It determines

how large each optimization step should be.

A small learning rate results in

small,

careful updates.

A large learning rate produces

large,

aggressive updates.

The choice of learning rate has an enormous influence on training stability.

---

### Gradient

The quantity

$$
\nabla_{\theta}L
$$

is the gradient computed by backpropagation.

It tells us

which direction increases the loss most rapidly.

To reduce the loss,

we therefore move in the opposite direction.

Hence the negative sign in the update equation.

---

# An Intuitive Analogy

Imagine hiking down a mountain while blindfolded.

Your objective is to reach the lowest point in the valley.

You cannot see the landscape.

However,

at every step,

someone tells you

which direction points uphill.

Naturally,

you walk in the opposite direction.

Eventually,

you reach the valley.

Gradient descent works in exactly the same manner.

The loss function defines the landscape.

The gradient indicates the uphill direction.

The optimizer walks downhill.

The minimum of the loss corresponds to the valley.

---

# Why Not Take Huge Steps?

Suppose the learning rate is extremely large.

Instead of taking careful steps,

the optimizer leaps enormous distances.

Conceptually,

```text
Valley

↓

Huge Jump

↓

Other Side

↓

Huge Jump Back

↓

Other Side Again
```

The optimizer repeatedly overshoots the minimum.

Training becomes unstable.

Sometimes,

the loss even diverges.

---

Now suppose the learning rate is extremely small.

```text
Tiny Step

↓

Tiny Step

↓

Tiny Step

↓

Tiny Step
```

Training becomes painfully slow.

Reaching convergence may require thousands of epochs.

Therefore,

an appropriate learning rate balances

* stability,
* speed,
* convergence.

---

# Why We Use Adam Instead of Simple Gradient Descent

Although gradient descent is easy to understand,

it is rarely used for deep neural networks.

Modern models contain

millions of parameters.

Different parameters may require

different update magnitudes.

Simple gradient descent uses

the same learning rate

for every parameter.

This is inefficient.

Instead,

MEGNet typically employs

**Adam**

or

**AdamW**.

---

# Adam: Adaptive Moment Estimation

Adam stands for

**Adaptive Moment Estimation**.

Rather than using only the current gradient,

Adam also remembers

previous gradients.

This allows the optimizer to estimate

* the average gradient,
* the variability of the gradient.

Consequently,

parameter updates become

more stable,

more adaptive,

and generally faster.

Conceptually,

Adam maintains

```text
Current Gradient

+

Past Gradients

↓

Adaptive Update
```

Instead of treating every parameter equally,

Adam automatically adjusts the effective learning rate for each parameter.

---

# Why Adam Works Well for MEGNet

Graph neural networks produce complex optimization landscapes.

Different layers behave very differently.

For example,

the embedding layer may receive

small,

stable gradients.

Meanwhile,

the prediction layer may experience

large,

rapidly changing gradients.

Adam automatically adapts to these differences.

This makes optimization

more efficient

and more stable.

Consequently,

Adam has become the default optimizer for many graph neural networks.

---

# AdamW: The Modern Improvement

Although Adam performs well,

it has one important weakness.

Weight decay,

which serves as L2 regularization,

was originally incorporated into Adam in a way that does not perfectly match the intended regularization behavior.

AdamW modifies this procedure.

Instead of mixing weight decay with the gradient,

AdamW applies weight decay separately.

This seemingly small modification produces

better generalization,

particularly for deep neural networks.

Therefore,

recent materials informatics research often prefers

**AdamW**

over standard Adam.

---

# Initializing the Optimizer

In PyTorch,

AdamW is initialized as

```python
optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-3,

    weight_decay=1e-5

)
```

Notice that

```python
model.parameters()
```

automatically collects

every trainable parameter

inside the MEGNet model.

No manual specification is required.

---

# Performing the Update

Once gradients have been computed,

the optimizer performs one update step.

The entire parameter update requires only

```python
optimizer.step()
```

This deceptively simple command

* reads every stored gradient,
* computes the AdamW update,
* applies weight decay,
* modifies every trainable parameter.

After this operation,

the neural network has officially learned from the current mini-batch.

---

# What Changes After `optimizer.step()`?

Before the update,

suppose one parameter equals

$$
0.842731.
$$

After the optimizer step,

it might become

$$
0.841902.
$$

The change is very small.

In fact,

most parameter updates are tiny.

Yet,

millions of such small adjustments accumulate over thousands of mini-batches.

Gradually,

the network transforms from

random initialization

into

a model capable of accurately predicting material properties.

This gradual refinement is one of the defining characteristics of deep learning.

---

# The Complete Sequence So Far

At this stage,

the training loop for one mini-batch consists of

```text
Load Mini-Batch

↓

Move to GPU

↓

Zero Gradients

↓

Forward Pass

↓

Compute Loss

↓

Backpropagation

↓

Optimizer Step
```

After the optimizer step,

one training iteration is complete.

The DataLoader then provides the next mini-batch,

and the entire process repeats.

---

# One Epoch Is Thousands of Learning Steps

Recall that an epoch consists of

all mini-batches in the training dataset.

Suppose

* Training samples = 64,000
* Batch size = 64

Then

$$
\frac{64,000}{64}=1,000
$$

mini-batches.

This means

one epoch contains

1,000

forward passes,

1,000

backward passes,

and

1,000

optimizer updates.

If training lasts

200 epochs,

the optimizer performs

200,000

parameter updates.

This illustrates why deep learning models can gradually learn highly complex relationships despite each individual update being very small.

---

# Have We Finished Training?

Not yet.

After completing one epoch,

the model has learned from the training data.

However,

we still do not know whether it has become **better**.

To answer this question,

we must evaluate the model on data that it has **never seen during optimization**.

This is the purpose of the **validation phase**.

Validation allows us to monitor generalization, detect overfitting, determine whether the model is improving, decide when to save checkpoints, adjust the learning rate, and trigger early stopping.

In the next section, we will examine the complete validation procedure, explain why gradients are disabled during evaluation using `torch.no_grad()`, discuss the difference between `model.train()` and `model.eval()`, and show how validation guides the entire training process.

## 15.9 Training a Complete MEGNet Model (Part 8)

# Step 8 — Validation: Measuring the True Performance of MEGNet

After completing one epoch, the model has learned from every mini-batch in the training dataset.

At first glance, it might seem reasonable to measure the model's performance using the same training data.

However, this approach can be dangerously misleading.

A neural network can become extremely good at predicting the samples it has already seen while performing poorly on completely new crystal structures.

This phenomenon is known as **overfitting**, and it is one of the central challenges in machine learning.

To determine whether the model has truly learned general chemical and structural relationships—or has merely memorized the training data—we evaluate it on a **validation dataset**.

---

# Why We Need Validation

Consider two students preparing for an examination.

The first student memorizes the exact questions from previous years.

The second student studies the underlying concepts.

Suppose the examination contains entirely new questions.

The first student struggles because the questions differ from those that were memorized.

The second student performs well because the concepts have been understood.

A neural network behaves in exactly the same way.

If it memorizes the training data,

it may produce very low training error,

yet fail on new materials.

Validation allows us to distinguish between

**memorization**

and

**generalization**.

---

# The Role of the Validation Dataset

The complete dataset is typically divided into three parts.

```text
Entire Dataset

│

├── Training Set

├── Validation Set

└── Test Set
```

Each subset serves a different purpose.

### Training Set

The optimizer updates the model using this data.

Gradients are computed.

Weights are modified.

Learning occurs.

---

### Validation Set

The optimizer **never** updates parameters using validation data.

Instead,

the validation set answers questions such as

* Is the model improving?
* Has overfitting begun?
* Should training stop?
* Should this checkpoint be saved?

Validation therefore guides the training process.

---

### Test Set

The test set is used only after training is completely finished.

It provides an unbiased estimate of the model's final performance.

Ideally,

the test set remains untouched until every modeling decision has been completed.

---

# The Validation Workflow

After finishing one epoch,

the workflow becomes

```text
Training Epoch Complete

↓

Switch to Evaluation Mode

↓

Disable Gradient Computation

↓

Predict Validation Set

↓

Compute Validation Loss

↓

Compute Validation Metrics

↓

Compare with Previous Epoch

↓

Save Best Model?
```

Notice that

validation does **not**

modify the model.

It simply measures performance.

---

# Step 1 — Switching to Evaluation Mode

Before validating,

the model must be switched from training mode to evaluation mode.

In PyTorch,

this is done using

```python
model.eval()
```

This line is easy to overlook,

yet it is extremely important.

Many neural network layers behave differently during training and evaluation.

Examples include

* Dropout
* Batch Normalization

During training,

Dropout randomly disables neurons.

During validation,

we want every neuron to contribute.

Calling

```python
model.eval()
```

automatically changes the behavior of these layers.

---

# What Happens If We Forget `model.eval()`?

Suppose the network contains dropout.

During validation,

random neurons continue to disappear.

Consequently,

the predictions become inconsistent.

Running validation twice on the same dataset may produce different results.

The measured validation error no longer reflects the true model performance.

For this reason,

forgetting

```python
model.eval()
```

is one of the most common beginner mistakes in PyTorch.

---

# Step 2 — Disabling Gradient Computation

Validation does not require learning.

Therefore,

there is no reason to compute gradients.

PyTorch provides the context manager

```python
with torch.no_grad():
```

The validation loop typically appears as

```python
model.eval()

with torch.no_grad():

    for batch in val_loader:

        prediction = model(batch)

        ...
```

This simple statement produces several important benefits.

---

# Why Disable Gradients?

During training,

PyTorch stores every intermediate computation because it will later need them for backpropagation.

These stored tensors consume a considerable amount of GPU memory.

During validation,

backpropagation never occurs.

Therefore,

storing these tensors is unnecessary.

Using

```python
torch.no_grad()
```

offers several advantages.

* Reduces GPU memory usage
* Speeds up computation
* Prevents accidental gradient accumulation

For large MEGNet models,

the memory savings can be substantial.

---

# Step 3 — Making Predictions

The forward pass during validation is almost identical to the training forward pass.

```python
prediction = model(batch)
```

The important difference is that

no computational graph is constructed for gradient computation.

The network simply produces predictions.

For example,

```text
Crystal 1

↓

Predicted Formation Energy

↓

−2.43 eV
```

This process repeats for every crystal in the validation dataset.

---

# Step 4 — Computing Validation Loss

After obtaining predictions,

we compute the validation loss using exactly the same loss function employed during training.

```python
loss = criterion(

    prediction,

    batch.y

)
```

Suppose the validation dataset contains

500 crystals.

Each mini-batch produces its own loss.

These batch losses are then averaged to obtain the overall validation loss.

For example,

```text
Batch 1

↓

Loss = 0.021

Batch 2

↓

Loss = 0.018

Batch 3

↓

Loss = 0.024

...

↓

Average Validation Loss

↓

0.020
```

This single number summarizes the model's performance on unseen data.

---

# Step 5 — Computing Evaluation Metrics

Although the optimizer only needs the loss,

researchers usually compute additional metrics.

For regression tasks,

common choices include

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination ($R^2$)

For example,

```text
Validation Results

Loss : 0.019

MAE  : 0.041 eV

RMSE : 0.062 eV

R²   : 0.987
```

Each metric provides different information about model performance.

MAE is particularly popular in materials informatics because it is easy to interpret physically.

---

# Tracking Validation Performance

Validation metrics are recorded after every epoch.

Suppose the validation MAE evolves as follows.

| Epoch | Validation MAE (eV) |
| ----: | ------------------: |
|     1 |               0.214 |
|     5 |               0.132 |
|    10 |               0.081 |
|    20 |               0.046 |
|    30 |               0.039 |
|    40 |               0.041 |
|    50 |               0.045 |

Observe the trend.

Initially,

the model improves rapidly.

Later,

the validation error reaches a minimum.

After that,

performance begins to deteriorate.

This is the signature of overfitting.

---

# Comparing Training and Validation Loss

One of the most informative diagnostic tools is to compare

training loss

and

validation loss

over time.

Several characteristic patterns may emerge.

### Healthy Training

```text
Epoch →

Training Loss

↓

Validation Loss

↓

Both decrease together
```

The model is learning useful representations that generalize to unseen data.

---

### Underfitting

```text
Training Loss

High

Validation Loss

High
```

The model cannot even fit the training data.

Possible reasons include

* insufficient model capacity,
* poor feature representation,
* inadequate training.

---

### Overfitting

```text
Training Loss

↓

↓

↓

Validation Loss

↓

↓

↑

↑
```

Training performance continues to improve,

but validation performance worsens.

The model is memorizing the training data.

---

# Why Validation Guides Training

Validation results influence nearly every important decision made during training.

They determine

* whether the current checkpoint should be saved,
* whether the learning rate should be reduced,
* whether early stopping should terminate training,
* which epoch represents the best-performing model.

Without validation,

these decisions would rely on training performance alone,

which often leads to models that fail to generalize.

Validation is therefore not merely an evaluation step—

it is the central feedback mechanism that guides the entire optimization process.

---

# Summary

At the end of each epoch,

the MEGNet training pipeline performs a complete evaluation on previously unseen crystal structures.

Unlike the training phase,

validation

* uses `model.eval()`,
* disables gradient computation with `torch.no_grad()`,
* performs only forward passes,
* computes loss and evaluation metrics,
* never updates model parameters.

These validation results determine whether the model is genuinely improving or beginning to overfit.

The next step is to use this information intelligently.

Instead of simply recording validation metrics,

we can use them to automatically

* save the best-performing model,
* reduce the learning rate when progress stalls,
* terminate training when further improvement becomes unlikely.

In the next section, we will integrate **learning rate scheduling, model checkpointing, and early stopping** into a single, complete training pipeline, producing a research-quality MEGNet training workflow suitable for real materials informatics applications.

## 15.9 Training a Complete MEGNet Model (Part 9)

# Step 9 — Model Checkpointing: Saving the Best MEGNet Model

After every validation phase, an important question arises:

> **Should we save the current model?**

A beginner might answer,

> "Just save the model after the last epoch."

Unfortunately, this strategy often produces inferior models.

The model obtained after the final epoch is **not necessarily the best model**.

In many cases, the neural network reaches its highest validation performance long before training ends.

Consequently, modern deep learning does not simply save the final model.

Instead, it saves the **best-performing model** during training.

This process is known as **model checkpointing**.

---

# Why Saving Only the Final Model Is Dangerous

Suppose we train MEGNet for

100 epochs.

The validation MAE evolves as follows.

| Epoch | Validation MAE (eV) |
| ----: | ------------------: |
|    10 |               0.091 |
|    20 |               0.058 |
|    30 |               0.041 |
|    40 |               0.034 |
|    50 |               0.031 |
|    60 |               0.029 |
|    70 |               0.028 |
|    80 |               0.030 |
|    90 |               0.034 |
|   100 |               0.039 |

Notice what happened.

The model improved continuously until

Epoch 70.

After that,

the validation error increased.

The model began to overfit.

If we simply save the final epoch,

we obtain

```text
Epoch 100

MAE = 0.039 eV
```

However,

the best model actually occurred at

```text
Epoch 70

MAE = 0.028 eV
```

The difference is substantial.

---

# The Purpose of a Checkpoint

A checkpoint is a snapshot of the training process.

It records the current state of the model so that it can be restored later.

Depending on the application,

a checkpoint may include

* model parameters,
* optimizer state,
* learning rate scheduler state,
* epoch number,
* validation metrics.

In its simplest form,

a checkpoint stores only the trained neural network weights.

---

# The Checkpoint Decision

After each validation phase,

the algorithm compares

```text
Current Validation Loss
```

with

```text
Best Validation Loss So Far
```

Two outcomes are possible.

---

### Case 1 — Performance Improved

Suppose

```text
Best Loss

0.026
```

Current loss

```text
0.023
```

Since

```text
0.023 < 0.026
```

the model has improved.

The checkpoint should be updated.

Conceptually,

```text
Current Model

↓

Better than Previous?

↓

YES

↓

Save Model
```

The current model now becomes

the new best model.

---

### Case 2 — Performance Did Not Improve

Suppose

```text
Best Loss

0.023
```

Current loss

```text
0.027
```

Since performance has deteriorated,

the checkpoint remains unchanged.

```text
Current Model

↓

Better than Previous?

↓

NO

↓

Keep Old Checkpoint
```

Thus,

the best model is never overwritten by a worse one.

---

# Implementing Checkpointing in PyTorch

The logic is surprisingly simple.

```python
if val_loss < best_val_loss:

    best_val_loss = val_loss

    torch.save(

        model.state_dict(),

        "best_megnet_model.pth"

    )
```

Only two operations occur.

First,

the best validation loss is updated.

Second,

the model parameters are written to disk.

---

# Understanding `state_dict()`

Many beginners wonder

why we save

```python
model.state_dict()
```

instead of

```python
model
```

The answer is efficiency.

A PyTorch model contains

* network architecture,
* parameters,
* Python objects,
* methods.

The architecture is already defined in the source code.

The only information that changes during training is

the parameter values.

The

```python
state_dict()
```

contains precisely these trainable tensors.

Consequently,

saving the state dictionary is

smaller,

faster,

and more portable.

---

# Loading a Saved Model

Once training is complete,

the checkpoint can be restored.

```python
model.load_state_dict(

    torch.load(

        "best_megnet_model.pth"

    )

)
```

After this command,

the neural network returns to

its best recorded state.

Notice that

training does not resume automatically.

Only the parameters are restored.

---

# What Information Can a Research Checkpoint Contain?

Professional research projects often save considerably more than the model weights.

A comprehensive checkpoint may include

```python
checkpoint = {

    "epoch": epoch,

    "model_state_dict": model.state_dict(),

    "optimizer_state_dict":

        optimizer.state_dict(),

    "scheduler_state_dict":

        scheduler.state_dict(),

    "validation_loss": val_loss

}
```

Saving the optimizer state allows training to resume later without losing momentum estimates or adaptive learning-rate information.

---

# Naming Checkpoints

Research projects often save checkpoints using descriptive filenames.

Examples include

```text
best_model.pth

best_validation_model.pth

megnet_bandgap_best.pth

formation_energy_checkpoint.pth
```

Meaningful filenames become particularly valuable when multiple experiments are conducted simultaneously.

---

# Saving Multiple Checkpoints

Sometimes researchers save

more than one checkpoint.

For example,

```text
Latest Model

↓

Saved Every Epoch
```

and

```text
Best Model

↓

Saved Only When Validation Improves
```

The latest model allows interrupted training to resume.

The best model provides the highest validation performance.

Maintaining both checkpoints is common in large-scale research projects.

---

# Checkpoint Frequency

Should we save after every mini-batch?

Usually not.

Saving to disk is relatively slow.

Instead,

most implementations save checkpoints

once per epoch,

after validation has been completed.

This strikes a good balance between computational efficiency and data safety.

---

# Why Validation Is Used Instead of Training Loss

An important question arises.

Why compare validation loss rather than training loss?

Suppose

Training Loss

```text
0.002
```

Validation Loss

```text
0.061
```

The training performance is excellent.

However,

validation performance is poor.

Saving this model would preserve an overfitted network.

Checkpointing based on validation performance ensures that the selected model generalizes well to unseen materials.

---

# The Training Pipeline So Far

Our complete training workflow has now expanded.

```text
Initialize Model

↓

Train One Epoch

↓

Validation

↓

Better Than Previous?

↓

Yes

↓

Save Checkpoint

↓

Next Epoch
```

This simple decision process ensures that the best-performing model is never lost during training.

---

# Is Saving the Best Model Enough?

Checkpointing protects the best model,

but it does not solve another important problem.

Suppose the validation loss stops improving for many consecutive epochs.

The optimizer may continue training

for hours,

or even days,

without achieving any meaningful improvement.

This wastes computational resources.

A more intelligent strategy is to monitor the validation performance and automatically terminate training when improvement has stalled.

This technique is called **early stopping**.

In the next section, we will study early stopping in detail, explain the concept of **patience**, discuss how it prevents overfitting, and integrate it into the complete MEGNet training pipeline to produce a robust, research-grade training workflow.

## 15.9 Training a Complete MEGNet Model (Part 10)

# Step 10 — Early Stopping: Preventing Overfitting Automatically

Modern deep learning models are often trained for hundreds of epochs.

However, training for more epochs does **not** always produce a better model.

After a certain point, the neural network begins to memorize the training data instead of learning the underlying physical relationships.

As a result,

* training loss continues to decrease,
* validation loss begins to increase.

This phenomenon is called **overfitting**.

Rather than forcing the user to manually monitor training, modern machine learning uses an automatic mechanism called **early stopping**.

Early stopping terminates training when the validation performance has stopped improving.

It is one of the simplest and most effective regularization techniques in deep learning.

---

# Why More Training Can Hurt Performance

Suppose we train MEGNet for 200 epochs.

The validation MAE evolves as follows.

| Epoch | Validation MAE (eV) |
| ----: | ------------------: |
|    10 |               0.182 |
|    20 |               0.104 |
|    30 |               0.061 |
|    40 |               0.038 |
|    50 |               0.031 |
|    60 |               0.028 |
|    70 |               0.027 |
|    80 |               0.028 |
|    90 |               0.030 |
|   100 |               0.033 |
|   150 |               0.044 |
|   200 |               0.057 |

Notice what happens.

The model reaches its best performance around

Epoch 70.

After that,

training continues,

but validation performance gradually deteriorates.

If we continue training until Epoch 200,

we obtain a worse model than the one already available at Epoch 70.

Early stopping prevents this unnecessary training.

---

# The Basic Idea

Instead of asking

> "How many epochs should I train?"

we ask

> "Has the validation performance improved recently?"

If the answer is

"No"

for many consecutive epochs,

training stops automatically.

Conceptually,

```text
Validation Improves

↓

Continue Training

Validation Stops Improving

↓

Wait

↓

Still No Improvement?

↓

Stop Training
```

---

# Patience

The central idea behind early stopping is the **patience parameter**.

Patience specifies

how many consecutive epochs

without improvement

are allowed before training terminates.

Suppose

```text
Patience = 10
```

This means

training continues

for ten additional epochs

after the last improvement.

If validation performance still does not improve,

training stops.

---

# Why Patience Is Necessary

Suppose validation loss behaves as follows.

| Epoch | Validation Loss |
| ----: | --------------: |
|    40 |           0.041 |
|    41 |           0.042 |
|    42 |           0.041 |
|    43 |           0.040 |
|    44 |           0.039 |

Notice that

validation performance briefly worsens

before improving again.

If we stopped immediately after Epoch 41,

we would miss the better model found later.

Patience prevents premature termination.

It allows the optimizer time to escape temporary plateaus.

---

# Tracking Improvement

Early stopping maintains two variables.

```text
Best Validation Loss

Counter
```

Initially,

```text
Best Loss = ∞

Counter = 0
```

After each validation phase,

the algorithm compares

the current validation loss

with

the best validation loss observed so far.

---

### Case 1 — Improvement

Suppose

```text
Best Loss = 0.031

Current Loss = 0.029
```

Since performance improved,

```text
Best Loss

↓

Updated to 0.029

Counter

↓

Reset to 0
```

Training continues.

---

### Case 2 — No Improvement

Suppose

```text
Best Loss = 0.029

Current Loss = 0.031
```

Performance did not improve.

Therefore,

```text
Counter

↓

Counter + 1
```

Training still continues,

provided that the counter has not exceeded the patience value.

---

# When Does Training Stop?

Suppose

```text
Patience = 5
```

Validation losses evolve as follows.

| Epoch | Validation Loss | Counter |
| ----: | --------------: | ------: |
|    50 |           0.028 |       0 |
|    51 |           0.029 |       1 |
|    52 |           0.030 |       2 |
|    53 |           0.031 |       3 |
|    54 |           0.030 |       4 |
|    55 |           0.032 |       5 |

At Epoch 55,

the patience limit has been reached.

Training terminates automatically.

Notice that

the best model remains

the checkpoint saved at

Epoch 50.

---

# Implementing Early Stopping

A simplified implementation appears below.

```python
if val_loss < best_val_loss:

    best_val_loss = val_loss

    patience_counter = 0

    torch.save(

        model.state_dict(),

        "best_model.pth"

    )

else:

    patience_counter += 1

if patience_counter >= patience:

    print("Early stopping")

    break
```

Although only a few lines long,

this algorithm is used in thousands of modern research projects.

---

# Early Stopping and Checkpointing Work Together

Checkpointing

and

early stopping

serve different purposes.

Checkpointing answers

> "Which model should be saved?"

Early stopping answers

> "When should training stop?"

Together,

they produce an efficient training workflow.

```text
Training

↓

Validation

↓

Improved?

↓

Yes

↓

Save Checkpoint

↓

Reset Patience

↓

Continue

────────────

Improved?

↓

No

↓

Increase Counter

↓

Counter ≥ Patience?

↓

Yes

↓

Stop Training
```

---

# Choosing the Patience Value

The appropriate patience depends on

* dataset size,
* optimizer,
* learning rate,
* model complexity.

Typical values include

| Dataset Size | Typical Patience |
| ------------ | ---------------: |
| Small        |     10–20 epochs |
| Medium       |     20–40 epochs |
| Large        |    30–100 epochs |

For many MEGNet applications,

patience values between

20

and

50 epochs

provide good results.

---

# Advantages of Early Stopping

Early stopping offers several important benefits.

### Prevents Overfitting

Training stops before the model begins memorizing the training data.

---

### Saves Computational Time

Large graph neural networks may require

many hours

or even

days

to train.

Early stopping avoids wasting computation once improvement has ceased.

---

### Requires No Architectural Changes

Unlike dropout or weight decay,

early stopping does not modify the neural network itself.

It simply determines

when training should end.

---

### Improves Generalization

In practice,

the model selected by early stopping often performs better on completely unseen materials than the final epoch.

---

# Limitations

Early stopping is highly effective,

but it is not perfect.

If the validation dataset is very small,

random fluctuations may trigger premature stopping.

Similarly,

if the patience value is too small,

training may terminate before reaching the optimal solution.

Consequently,

the validation curve should always be interpreted alongside domain knowledge and other evaluation metrics.

---

# The Complete Research Training Pipeline

At this stage,

our MEGNet training workflow has become considerably more sophisticated.

```text
Initialize Model

↓

Initialize Optimizer

↓

Train One Epoch

↓

Validation

↓

Improved?

↓

Yes

↓

Save Best Model

↓

Reset Patience

↓

Continue Training

────────────

No

↓

Increase Patience Counter

↓

Counter ≥ Patience?

↓

Yes

↓

Stop Training
```

This pipeline is representative of modern deep learning practice and forms the backbone of many published materials informatics studies.

---

# What Is Still Missing?

Although the training pipeline is now robust,

one important component remains.

Suppose validation performance improves rapidly at the beginning of training,

then reaches a plateau.

Using the same learning rate throughout training is often inefficient.

Instead,

modern optimizers automatically **reduce the learning rate** when progress slows,

allowing the model to make increasingly fine parameter updates near the optimum.

This strategy is implemented through a **learning rate scheduler**.

In the next section, we will study learning rate scheduling in depth, explain why reducing the learning rate often produces substantial improvements in MEGNet performance, compare common scheduling strategies (StepLR, Cosine Annealing, OneCycleLR, and ReduceLROnPlateau), and demonstrate how schedulers are integrated into a complete, publication-quality MEGNet training pipeline.

## 15.9 Training a Complete MEGNet Model (Part 11)

# Step 11 — Learning Rate Scheduling: Making Optimization More Efficient

In the previous sections, we learned that the **learning rate** controls how large each parameter update is during optimization.

Recall the gradient descent update equation

[
\theta_{\text{new}}
===================

## \theta_{\text{old}}

\eta
\nabla_{\theta}L,
]

where

* (\theta) represents the trainable parameters,
* (\nabla_{\theta}L) is the gradient,
* (\eta) is the learning rate.

Throughout our earlier discussion, we treated the learning rate as a fixed number.

For example,

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

This means that every optimizer step uses exactly the same learning rate.

Although this approach works,

it is rarely the best strategy.

Modern deep learning almost always changes the learning rate **during training**.

This process is called **learning rate scheduling**.

---

# Why Should the Learning Rate Change?

Imagine driving from one city to another.

At the beginning of the journey,

the road is empty.

Driving quickly is efficient.

However,

as you approach your destination,

continuing at high speed becomes dangerous.

Instead,

you gradually slow down.

Optimization behaves in a very similar way.

During the early stages of training,

the neural network is far from the optimum.

Large parameter updates allow rapid learning.

Near convergence,

the model requires much smaller updates to fine-tune the parameters.

A scheduler automatically performs this adjustment.

---

# Constant Learning Rate vs Scheduled Learning Rate

Suppose the learning rate remains fixed.

```text
Epoch

↓

Learning Rate

0.001

0.001

0.001

0.001

0.001
```

Every optimizer step has the same magnitude.

Now consider a scheduled learning rate.

```text
Epoch

↓

0.001

↓

0.0005

↓

0.0001

↓

0.00005
```

The optimizer starts aggressively,

then gradually becomes more precise.

This simple idea often improves both convergence speed and final model accuracy.

---

# An Intuitive Picture

Imagine the loss function as a valley.

Initially,

the optimizer is far from the minimum.

Large steps help reach the valley quickly.

```text
Large Steps

↓

↓

↓

Valley
```

Near the minimum,

large steps may overshoot the optimum.

Instead,

small careful steps are preferable.

```text
Near Minimum

↓

Small Step

↓

Smaller Step

↓

Optimal Solution
```

Learning rate scheduling automatically performs this transition.

---

# Benefits of Learning Rate Scheduling

A well-designed scheduler provides several advantages.

### Faster Initial Learning

Large learning rates allow the optimizer to explore the parameter space efficiently.

---

### Improved Stability

Reducing the learning rate later prevents oscillations around the optimum.

---

### Better Final Accuracy

Small updates near convergence allow the model to refine its parameters more precisely.

---

### Reduced Training Time

Because optimization becomes more efficient,

fewer epochs are often required.

---

# Types of Learning Rate Schedulers

PyTorch provides many scheduling strategies.

Some of the most widely used include

* StepLR
* MultiStepLR
* ExponentialLR
* CosineAnnealingLR
* OneCycleLR
* ReduceLROnPlateau

Each follows a different philosophy.

We will examine the most important ones individually.

---

# StepLR

The simplest scheduler reduces the learning rate after a fixed number of epochs.

Suppose

* initial learning rate = 0.001
* step size = 30 epochs
* decay factor = 0.1

The schedule becomes

| Epoch | Learning Rate |
| ----: | ------------: |
|  1–30 |         0.001 |
| 31–60 |        0.0001 |
| 61–90 |       0.00001 |

Every thirty epochs,

the learning rate decreases by a factor of ten.

---

### PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)
```

At the end of every epoch,

we update the scheduler.

```python
scheduler.step()
```

Although simple,

StepLR has been successfully used in many deep learning applications.

---

# Exponential Learning Rate Decay

Instead of reducing the learning rate abruptly,

ExponentialLR decreases it gradually after every epoch.

Conceptually,

```text
Epoch

↓

0.001

↓

0.00095

↓

0.00090

↓

0.00086

↓

...
```

The learning rate changes smoothly,

avoiding sudden jumps.

Implementation is equally straightforward.

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95
)
```

Here,

the learning rate decreases by approximately

5%

after every epoch.

---

# Cosine Annealing

One of the most popular schedulers in modern deep learning is

**Cosine Annealing**.

Instead of decreasing the learning rate linearly,

it follows a cosine curve.

Conceptually,

```text
Learning Rate

High

│\
│ \
│  \
│   \
│    \____

Epoch
```

The learning rate decreases slowly at first,

then more rapidly,

before flattening near the end of training.

Cosine Annealing often provides smoother convergence than StepLR.

---

### PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100
)
```

Here,

(T_{\text{max}})

represents the number of epochs required to complete one cosine cycle.

Cosine scheduling has become especially popular for transformer models and graph neural networks.

---

# ReduceLROnPlateau

Unlike previous schedulers,

ReduceLROnPlateau does not depend on the epoch number.

Instead,

it monitors

the validation performance.

Suppose validation loss stops improving.

Rather than terminating training,

the scheduler first reduces the learning rate.

Conceptually,

```text
Validation Improves

↓

Keep Learning Rate

Validation Plateaus

↓

Reduce Learning Rate

↓

Continue Training
```

This allows the optimizer to make smaller,

more precise updates.

---

### PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)
```

Unlike most schedulers,

the validation loss must be supplied.

```python
scheduler.step(val_loss)
```

If validation loss fails to improve for five epochs,

the learning rate is reduced by half.

---

# Why ReduceLROnPlateau Is Popular in Materials Informatics

Materials datasets often exhibit

long periods of slow improvement.

A fixed learning rate may become too large to achieve further refinement.

ReduceLROnPlateau automatically detects this situation.

Instead of stopping immediately,

it decreases the learning rate,

allowing optimization to continue.

Consequently,

this scheduler is widely used in

* CGCNN,
* MEGNet,
* ALIGNN,
* M3GNet,

and many other materials informatics models.

---

# Combining Scheduling with Early Stopping

Learning rate scheduling

and

early stopping

work together.

Suppose validation loss stops improving.

The scheduler first reduces the learning rate.

If performance subsequently improves,

training continues.

If improvement still does not occur,

early stopping eventually terminates training.

Conceptually,

```text
Validation Plateau

↓

Reduce Learning Rate

↓

Performance Improves?

↓

Yes

↓

Continue Training

────────────

No

↓

Early Stopping
```

This combination is considerably more effective than early stopping alone.

---

# Monitoring the Learning Rate

During training,

it is useful to record the current learning rate.

PyTorch allows this through

```python
current_lr = optimizer.param_groups[0]["lr"]

print(current_lr)
```

A typical training log might appear as

```text
Epoch 10

Learning Rate = 0.001

Validation MAE = 0.051

-------------------------

Epoch 30

Learning Rate = 0.0005

Validation MAE = 0.038

-------------------------

Epoch 60

Learning Rate = 0.00025

Validation MAE = 0.031
```

Such logs help researchers understand how the scheduler influences optimization.

---

# Recommended Scheduler for MEGNet

Several schedulers perform well with graph neural networks,

but practical experience suggests the following recommendations.

| Scenario                    | Recommended Scheduler           |
| --------------------------- | ------------------------------- |
| Small datasets              | ReduceLROnPlateau               |
| Medium datasets             | CosineAnnealingLR               |
| Large datasets              | OneCycleLR or CosineAnnealingLR |
| Simple baseline experiments | StepLR                          |

For many MEGNet applications,

**ReduceLROnPlateau** is an excellent default choice because it responds directly to validation performance rather than relying on a predetermined epoch schedule.

---

# Summary

Learning rate scheduling is a simple yet powerful technique for improving optimization.

Instead of using one fixed learning rate throughout training,

the scheduler automatically adjusts the step size as learning progresses.

Different schedulers employ different strategies,

but all share the same objective:

* accelerate early learning,
* stabilize late optimization,
* improve final model performance.

When combined with

* checkpointing,
* early stopping,
* AdamW optimization,

learning rate scheduling forms an essential component of a modern, research-quality MEGNet training pipeline.

In the next section, we will combine **every component studied so far** into a **complete end-to-end MEGNet training program**, integrating data loading, model initialization, optimization, validation, checkpointing, learning rate scheduling, and early stopping into a single, publication-quality training script suitable for real-world materials informatics research.

## 15.9 Training a Complete MEGNet Model (Part 12)

# Step 12 — Building a Complete Research-Grade MEGNet Training Pipeline

Over the previous sections, we have studied every component of the MEGNet training process individually.

We learned

* how data are loaded,
* how mini-batches are created,
* how the forward pass works,
* how the loss is computed,
* how backpropagation calculates gradients,
* how AdamW updates parameters,
* how validation measures generalization,
* how checkpointing saves the best model,
* how early stopping prevents overfitting,
* how learning rate scheduling improves optimization.

Individually,

each of these components is relatively simple.

The true power of modern deep learning emerges when they are combined into a single training pipeline.

In this section, we will assemble every component into a research-quality workflow similar to those used in published materials informatics studies.

---

# The Complete Training Pipeline

At a high level,

training a MEGNet model follows the sequence below.

```text
Load Dataset

↓

Split Dataset

↓

Create DataLoaders

↓

Initialize Model

↓

Initialize Optimizer

↓

Initialize Scheduler

↓

Training Loop

↓

Validation Loop

↓

Checkpoint

↓

Early Stopping

↓

Testing

↓

Deployment
```

Every research project based on MEGNet follows this general structure.

The specific implementation details may differ,

but the overall workflow remains essentially the same.

---

# Step 1 — Loading the Dataset

Training begins by loading the crystal dataset.

The dataset may originate from

* Materials Project,
* OQMD,
* JARVIS,
* AFLOW,
* or a custom DFT database.

Each sample contains

* atomic species,
* atomic coordinates,
* lattice vectors,
* target properties.

Conceptually,

```text
Crystal Structure

↓

Graph Construction

↓

Crystal Graph Dataset
```

At this stage,

no neural network computation has yet occurred.

---

# Step 2 — Splitting the Dataset

Before training,

the dataset is divided into three subsets.

```text
Entire Dataset

↓

Training Set

Validation Set

Test Set
```

A common split is

* 80% training,
* 10% validation,
* 10% testing.

The validation and test sets must remain completely independent of training.

---

# Step 3 — Creating DataLoaders

Each subset is wrapped inside a DataLoader.

```python
train_loader

val_loader

test_loader
```

The DataLoader

* creates mini-batches,
* shuffles the training data,
* manages parallel loading,
* feeds batches to the GPU.

This allows efficient training even for datasets containing hundreds of thousands of crystals.

---

# Step 4 — Initializing the Model

The MEGNet model is now created.

```python
model = MEGNet(...)
```

Initially,

all trainable parameters are randomly initialized.

At this point,

the model has absolutely no knowledge of chemistry,

crystal structures,

or material properties.

Every prediction is essentially random.

Training will gradually transform these random parameters into meaningful representations.

---

# Step 5 — Moving the Model to the GPU

If a GPU is available,

the model is transferred to GPU memory.

```python
device = torch.device("cuda")

model.to(device)
```

This allows the large matrix operations inside MEGNet to execute much more efficiently.

For modern graph neural networks,

GPU acceleration is almost essential.

---

# Step 6 — Initializing the Optimizer

The optimizer determines

how parameters are updated.

A typical choice is

```python
optimizer = AdamW(...)
```

The optimizer stores

* learning rate,
* weight decay,
* momentum information.

These quantities remain available throughout training.

---

# Step 7 — Initializing the Loss Function

Next,

the regression loss is selected.

Examples include

```python
MSELoss()

HuberLoss()

SmoothL1Loss()
```

The choice depends on

* dataset quality,
* noise level,
* target property.

For many materials informatics applications,

Huber loss provides an excellent balance between stability and robustness.

---

# Step 8 — Initializing the Scheduler

The learning-rate scheduler is then created.

For example,

```python
ReduceLROnPlateau(...)
```

or

```python
CosineAnnealingLR(...)
```

The scheduler automatically adjusts the learning rate during optimization.

---

# Step 9 — The Epoch Loop

Training now begins.

Conceptually,

```text
Epoch 1

↓

Epoch 2

↓

Epoch 3

↓

...

↓

Epoch N
```

Each epoch processes

every training crystal exactly once.

---

# Step 10 — The Mini-Batch Loop

Inside every epoch,

the DataLoader produces mini-batches.

```text
Mini-batch 1

↓

Mini-batch 2

↓

Mini-batch 3

↓

...
```

Each mini-batch follows the complete optimization cycle.

---

# Step 11 — Training One Mini-Batch

For every mini-batch,

the following operations occur.

```text
Move Batch to GPU

↓

Zero Gradients

↓

Forward Pass

↓

Compute Loss

↓

Backpropagation

↓

Optimizer Step
```

This sequence is repeated thousands of times during training.

---

# Step 12 — Validation

After every epoch,

training temporarily pauses.

The model switches to evaluation mode.

```python
model.eval()
```

Gradients are disabled.

```python
torch.no_grad()
```

The validation dataset is processed,

and metrics such as

* validation loss,
* MAE,
* RMSE

are computed.

These metrics estimate

how well the model generalizes to unseen crystals.

---

# Step 13 — Scheduler Update

Once validation is complete,

the scheduler examines the validation results.

If progress has slowed,

the learning rate is reduced.

Conceptually,

```text
Validation Plateau

↓

Reduce Learning Rate

↓

Continue Optimization
```

This often allows additional improvements.

---

# Step 14 — Save the Best Model

If validation performance improves,

the current model is saved.

```text
Validation Improved

↓

Save Checkpoint
```

The checkpoint contains

the best-performing parameters observed so far.

---

# Step 15 — Early Stopping Check

If validation performance has not improved for many epochs,

the patience counter increases.

Eventually,

```text
Counter ≥ Patience

↓

Stop Training
```

The optimization process terminates automatically.

---

# Step 16 — Load the Best Checkpoint

Once training finishes,

the best checkpoint is restored.

```python
model.load_state_dict(...)
```

Notice that

the final epoch

may not correspond

to the best-performing model.

Checkpointing ensures that

the optimal parameters are recovered.

---

# Step 17 — Final Testing

Only now

is the test dataset evaluated.

The test set has never influenced

* optimization,
* checkpointing,
* scheduler decisions,
* early stopping.

Consequently,

its performance provides

an unbiased estimate

of the model's real predictive capability.

---

# The Entire Training Pipeline

Putting everything together,

the complete workflow becomes

```text
Load Dataset

↓

Split Dataset

↓

Create DataLoaders

↓

Initialize MEGNet

↓

Initialize Optimizer

↓

Initialize Scheduler

↓

For Each Epoch

    ↓

    Train

        ↓

        Forward

        ↓

        Loss

        ↓

        Backward

        ↓

        Optimizer Step

    ↓

    Validation

    ↓

    Scheduler

    ↓

    Save Best Model

    ↓

    Early Stopping

↓

Load Best Checkpoint

↓

Evaluate on Test Set

↓

Deploy Model
```

This flowchart summarizes the entire lifecycle of training a modern graph neural network for materials property prediction.

---

# From Theory to Research Practice

At the beginning of this chapter,

MEGNet may have appeared to be a highly complex architecture.

However,

after studying every stage individually,

its training process can now be understood as a sequence of well-defined operations.

Every published MEGNet study,

regardless of the target property,

follows this same fundamental workflow.

Whether predicting

* formation energies,
* band gaps,
* elastic constants,
* dielectric properties,
* thermal conductivity,
* battery voltages,

or any other materials property,

the underlying optimization pipeline remains essentially unchanged.

The differences lie primarily in

* the dataset,
* the target property,
* the graph construction,
* the hyperparameters,

not in the core training procedure.

---

# Conclusion of Section 15.9

We have now completed a comprehensive study of the MEGNet training pipeline, progressing from the loading of crystal datasets to the final evaluation of a trained model.

More importantly, we have connected the mathematical concepts of gradients, optimization, loss functions, and graph neural networks to their practical implementation in a research workflow.

At this point, you should understand **how** a MEGNet model learns.

However, understanding how a model learns is only one part of becoming a successful materials informatics researcher.

Equally important is understanding **why a model makes a particular prediction**.

Deep neural networks are often criticized for behaving as "black boxes."

The next major section addresses this challenge.

We will begin **Section 15.10 — Interpreting MEGNet Predictions**, where we will explore feature importance, learned atomic embeddings, latent representations, attention and message-passing analysis, visualization techniques, uncertainty estimation, and explainable AI methods that reveal what the model has actually learned about crystal chemistry and materials physics.

## 15.10 Interpreting MEGNet Predictions

### Why Interpretability Matters in Materials Informatics

By this point, we have learned how MEGNet

* represents crystal structures as graphs,
* performs message passing,
* updates atomic, bond, and global features,
* predicts material properties,
* learns through backpropagation,
* is trained using modern optimization techniques.

If the training process is successful, the model may achieve excellent predictive accuracy.

For example,

| Property         |           MAE |
| ---------------- | ------------: |
| Formation Energy | 0.023 eV/atom |
| Band Gap         |       0.19 eV |
| Bulk Modulus     |       8.5 GPa |

These numbers are impressive.

However, an important scientific question immediately arises.

> **Why did the model make these predictions?**

Unlike classical physics models, deep neural networks rarely provide explicit equations.

For example,

a traditional regression model might produce

[
E = 0.52x_1 - 1.43x_2 + 0.81x_3.
]

From this equation, we can immediately determine

* which variables are important,
* whether each variable increases or decreases the prediction,
* how strongly each variable contributes.

A deep neural network such as MEGNet does not provide such a simple equation.

Instead, it consists of

* millions of trainable parameters,
* multiple message-passing layers,
* nonlinear activation functions,
* graph pooling operations,
* high-dimensional latent representations.

Consequently, the relationship between the input crystal structure and the predicted property is much more difficult to understand.

---

# Accuracy Is Not Enough

Suppose a MEGNet model predicts that a newly designed alloy has

```text
Formation Energy = −2.81 eV/atom
```

The prediction may be highly accurate.

However, a materials scientist immediately asks

* Which atoms contributed most?
* Which chemical bonds were important?
* Did the model recognize local coordination?
* Was the prediction dominated by composition or crystal structure?
* Can this prediction be trusted?

A prediction without an explanation is often insufficient for scientific discovery.

Scientists seek understanding,

not merely accurate numbers.

---

# The Black Box Problem

Deep neural networks are often described as **black boxes**.

This means

we know

* the input,
* the output,

but the internal reasoning remains difficult to interpret.

Conceptually,

```text
Crystal Structure

↓

???

↓

Predicted Band Gap
```

The question marks represent

hundreds of nonlinear mathematical operations.

Understanding these operations is the goal of model interpretability.

---

# Why Interpretability Is Especially Important in Materials Science

Interpretability is valuable in every application of machine learning,

but it is particularly important in materials science.

Unlike image classification,

materials informatics is not merely interested in prediction.

Its ultimate goal is

**scientific discovery**.

A successful machine learning model should help answer questions such as

* Why is diamond harder than silicon?
* Why do certain crystal structures become superconducting?
* Which atomic environments stabilize a material?
* Which chemical substitutions improve ionic conductivity?

If the model cannot provide insight,

its scientific usefulness becomes limited.

---

# Interpretability Builds Scientific Confidence

Imagine two models predicting

the formation energy

of the same crystal.

Model A predicts

```text
−3.12 eV/atom
```

without any explanation.

Model B predicts

```text
−3.10 eV/atom
```

and additionally explains

* oxygen atoms dominate the prediction,
* octahedral coordination is highly influential,
* long metal–oxygen bonds reduce stability,
* strong covalent bonding lowers the energy.

Although the numerical predictions are similar,

most researchers would trust

Model B

far more.

Interpretability increases confidence.

---

# Interpretability Helps Detect Errors

Understanding model behavior also helps identify mistakes.

Suppose a band-gap model consistently focuses on

lattice constants,

while ignoring

chemical composition.

This observation may indicate

* insufficient training data,
* incorrect graph construction,
* missing node features,
* biased optimization.

Interpretability therefore serves as a powerful debugging tool.

---

# Interpretability Supports Materials Discovery

Suppose researchers wish to design a new battery cathode.

A MEGNet model predicts

excellent voltage.

Interpretability may reveal

that

* transition-metal oxidation state,
* oxygen coordination,
* local bond geometry,

are primarily responsible.

These insights immediately suggest

new candidate materials with similar structural characteristics.

Thus,

interpretability transforms

machine learning

into

a scientific discovery tool.

---

# Levels of Interpretability

Interpretability can be studied at several different levels.

### 1. Global Interpretability

Global interpretability seeks to understand

how the model behaves overall.

Questions include

* Which elements are generally important?
* Which crystal structures are easily recognized?
* What chemical trends has the model learned?

This provides a broad understanding of the trained network.

---

### 2. Local Interpretability

Local interpretability focuses on

one individual prediction.

For example,

consider a single crystal.

We may ask

* Which atoms contributed most?
* Which bonds were most influential?
* Why was this particular band gap predicted?

Local explanations are especially useful for materials design.

---

### 3. Latent Representation Analysis

Instead of studying predictions directly,

we examine

the learned internal representations.

Questions include

* Do chemically similar elements cluster together?
* Are crystal families separated automatically?
* Has the model discovered periodic trends?

This analysis provides insight into

what the neural network has learned internally.

---

# Sources of Information Inside MEGNet

Unlike many machine learning models,

MEGNet contains several different types of learned information.

These include

```text
Atomic Embeddings

↓

Bond Embeddings

↓

Global State Embeddings

↓

Message Passing Features

↓

Graph Representation

↓

Final Prediction
```

Each stage contains valuable scientific information.

Consequently,

interpretability can be performed at multiple levels of the architecture.

---

# Major Interpretability Techniques for MEGNet

Throughout this section,

we will study several complementary approaches.

| Technique                      | Primary Question                                     |
| ------------------------------ | ---------------------------------------------------- |
| Atomic embedding visualization | What chemical relationships has the model learned?   |
| Latent-space visualization     | How are materials organized internally?              |
| Message-passing analysis       | How does information flow through the crystal graph? |
| Feature attribution            | Which atoms and bonds are most influential?          |
| Saliency analysis              | Which structural changes most affect predictions?    |
| SHAP analysis                  | How does each feature contribute to the prediction?  |
| Uncertainty estimation         | How confident is the model?                          |

Each technique provides a different perspective on the learned model.

No single method tells the complete story.

Instead,

they complement one another.

---

# Interpretability Is an Active Research Area

Although interpretability has become increasingly important,

it remains an active area of research.

Graph neural networks are significantly more difficult to interpret than

* linear regression,
* decision trees,
* random forests.

Their predictions depend on

complex interactions among

* atomic environments,
* neighboring atoms,
* bond features,
* graph topology,
* nonlinear transformations.

Developing reliable interpretation methods for graph neural networks is therefore an important research frontier in artificial intelligence.

---

# Roadmap for This Section

In the remainder of Section 15.10, we will gradually uncover what MEGNet has learned during training.

We will proceed in the following order:

1. **Learned Atomic Embeddings** – understanding how MEGNet represents chemical elements.
2. **Visualizing Latent Space** – exploring how crystals organize themselves in the learned feature space.
3. **Message Passing Interpretation** – analyzing how information propagates through crystal graphs.
4. **Feature Attribution Methods** – identifying the atoms, bonds, and structural motifs that most influence predictions.
5. **Explainable AI Techniques** – applying saliency maps, Integrated Gradients, GNNExplainer, and SHAP to graph neural networks.
6. **Uncertainty Estimation** – determining when the model is confident and when its predictions should be treated cautiously.
7. **Case Studies** – interpreting real MEGNet predictions for formation energy, band gap, and elastic properties.

By the end of this section, you will not only know **how MEGNet makes predictions**, but also **how to investigate the scientific reasoning behind those predictions**, enabling you to use graph neural networks as tools for both accurate prediction and meaningful materials discovery.

---

### Next Section

**15.10.1 Learned Atomic Embeddings: How MEGNet Discovers the Periodic Table Automatically**

## 15.10.1 Learned Atomic Embeddings: How MEGNet Discovers the Periodic Table Automatically

One of the most fascinating aspects of MEGNet is that it is **never explicitly taught chemistry**.

No one tells the model that

* sodium belongs to the alkali metals,
* fluorine is highly electronegative,
* silicon and germanium belong to the same group,
* oxygen forms strong bonds with transition metals.

The model receives only

* crystal structures,
* target material properties.

Yet, after training, MEGNet frequently develops internal representations that closely resemble the organization of the periodic table.

This remarkable behavior emerges automatically through learning.

---

# What Is an Atomic Embedding?

Earlier in this chapter, we learned that neural networks cannot process element symbols directly.

For example,

consider the crystal

```text
SiO₂
```

The neural network cannot perform calculations using the strings

```text
"Si"

"O"
```

Instead,

each element is converted into a numerical vector called an **embedding**.

Suppose the embedding dimension is

64.

Then

```text
Si

↓

[0.24, -0.17, 0.91, ..., 0.38]
```

Similarly,

```text
O

↓

[-0.73, 0.45, 0.18, ..., -0.61]
```

Each element is therefore represented by a point in a **64-dimensional vector space**.

These vectors become the initial node features of the crystal graph.

---

# Random at the Beginning

When training begins,

the embeddings contain no chemical information.

For example,

the embeddings might look like

```text
Hydrogen

↓

[0.18, -0.41, 0.72, ...]

Carbon

↓

[-0.36, 0.83, -0.27, ...]

Iron

↓

[0.55, -0.09, 0.14, ...]
```

These values are initialized randomly.

At this stage,

the model has no concept of

* valence electrons,
* electronegativity,
* atomic radius,
* oxidation state,
* periodic trends.

Every element is simply a random vector.

---

# How Do Embeddings Learn?

Suppose MEGNet is trained to predict

formation energy.

During training,

backpropagation updates not only

* convolution weights,
* message-passing functions,
* prediction layers,

but also

the atomic embeddings.

Conceptually,

```text
Random Embeddings

↓

Forward Pass

↓

Prediction Error

↓

Backpropagation

↓

Updated Embeddings
```

This process repeats

millions of times.

Gradually,

the embedding vectors evolve into meaningful chemical representations.

---

# Why Similar Elements Become Similar

Consider two alkali metals,

sodium (Na)

and

potassium (K).

Both

* possess one valence electron,
* commonly form +1 ions,
* exhibit similar bonding behavior,
* appear in chemically similar compounds.

Because they often play similar roles during prediction,

the optimizer discovers that

their embedding vectors should also become similar.

Mathematically,

their positions in embedding space move closer together.

Conceptually,

```text
Beginning

Na •                 K •

Far Apart

↓

Training

↓

Na • • K

Close Together
```

The optimizer is never instructed to do this.

It emerges naturally because similar embeddings reduce prediction error.

---

# A Geometric Interpretation

Suppose every element is represented as a point in a high-dimensional space.

Initially,

the arrangement resembles random noise.

```text
Random Space

H        Fe

      O

Si

          Na
```

After training,

elements with similar chemistry begin forming clusters.

```text
Learned Space

Li Na K

↓

Mg Ca Sr

↓

O S Se

↓

F Cl Br

↓

Fe Co Ni
```

This organization closely resembles the periodic table.

---

# Why Does This Happen?

The neural network optimizes only one objective:

reduce prediction error.

Suppose replacing

sodium

with

potassium

rarely changes the target property.

The optimizer gradually learns

that their embeddings should become nearly identical.

Conversely,

replacing

oxygen

with

gold

dramatically changes the prediction.

Therefore,

their embeddings remain far apart.

The embedding space naturally organizes itself according to chemical similarity.

---

# Embeddings Encode Hidden Chemical Knowledge

Although the embedding vectors consist only of numerical values,

they implicitly encode numerous chemical properties.

Researchers have observed that learned embeddings often correlate with

* atomic number,
* atomic radius,
* electronegativity,
* ionization energy,
* oxidation state,
* valence electron count,
* periodic group,
* periodic period.

Remarkably,

none of these quantities were explicitly supplied during training.

The neural network discovered them because they help predict material properties.

---

# Visualizing Atomic Embeddings

A 64-dimensional vector cannot be plotted directly.

To visualize the learned embeddings,

we first reduce their dimensionality.

Popular techniques include

* Principal Component Analysis (PCA),
* t-SNE,
* UMAP.

Suppose PCA projects the embeddings into two dimensions.

The resulting visualization might resemble

```text
                Halogens

                   •

             •

Transition Metals

• • • • •

             •

Alkali Metals

• • •
```

Clusters corresponding to chemical families often emerge automatically.

---

# Example: Alkali Metals

Suppose we examine

* Li
* Na
* K
* Rb
* Cs

After training,

their embedding vectors frequently occupy neighboring regions.

Why?

Because these elements

* possess one valence electron,
* exhibit similar oxidation states,
* participate in similar crystal environments.

The optimizer recognizes these similarities through the prediction task.

---

# Example: Oxygen and Sulfur

Consider

oxygen

and

sulfur.

Although different,

they both belong to

Group 16.

Both frequently

* accept electrons,
* form oxides and sulfides,
* bond strongly with metals.

Consequently,

their learned embeddings often lie relatively close together.

Again,

this relationship was not manually programmed.

---

# What Do Individual Dimensions Mean?

Suppose the embedding dimension equals

64.

Does

Dimension 17

represent electronegativity?

Does

Dimension 42

represent atomic radius?

Usually,

no.

Unlike handcrafted descriptors,

individual embedding dimensions rarely possess simple physical meanings.

Instead,

chemical information is distributed across many dimensions simultaneously.

For example,

electronegativity may influence

* Dimension 4,
* Dimension 12,
* Dimension 27,
* Dimension 51,

all at once.

This phenomenon is known as a **distributed representation**.

---

# Advantages of Learned Embeddings

Compared with manually designed descriptors,

learned embeddings provide several advantages.

### Automatically Optimized

The vectors are optimized specifically for the prediction task.

---

### Capture Complex Relationships

They learn nonlinear chemical relationships that handcrafted descriptors may miss.

---

### Adapt to Different Properties

Embeddings trained for

formation energy

may differ from those trained for

band gap

or

elastic modulus.

The representation adapts to the scientific problem.

---

### Reduce Manual Feature Engineering

Instead of designing dozens of chemical descriptors,

the neural network learns its own representation directly from data.

---

# Limitations

Despite their power,

atomic embeddings also have limitations.

First,

they require

large,

high-quality datasets.

Small datasets may produce unstable embeddings.

Second,

the learned vectors depend on

the prediction task.

An embedding optimized for

formation energy

may not be optimal for predicting

thermal conductivity.

Finally,

individual embedding dimensions are difficult to interpret physically,

making direct scientific interpretation more challenging than with handcrafted descriptors.

---

# Scientific Importance

The discovery that graph neural networks automatically learn chemically meaningful embeddings was one of the major breakthroughs in materials informatics.

It demonstrated that deep learning is capable of extracting fundamental chemical knowledge directly from crystal structures,

without requiring manually engineered descriptors.

This finding has inspired numerous subsequent architectures,

including

* CGCNN,
* MEGNet,
* ALIGNN,
* M3GNet,

all of which rely on learned atomic representations rather than fixed chemical feature vectors.

---

# Summary

Learned atomic embeddings form the foundation of MEGNet's understanding of chemistry.

Although initialized randomly,

they gradually evolve during training into structured representations that often reflect the organization of the periodic table.

Chemically similar elements naturally acquire similar embedding vectors because doing so reduces prediction error.

These embeddings capture rich chemical information in a high-dimensional latent space and serve as the starting point for all subsequent message-passing operations.

However, atomic embeddings describe **individual elements**.

A material is much more than a collection of isolated atoms.

The next question is therefore:

> **How does MEGNet combine these atomic embeddings to create a representation of an entire crystal?**

In the next section, **15.10.2 Latent Crystal Representations**, we will explore how message passing transforms atomic embeddings into high-dimensional representations of complete crystal structures and how these latent representations reveal hidden organization within materials space.

## 15.10.2 Latent Crystal Representations: How MEGNet Learns the Materials Space

In the previous section, we learned that MEGNet assigns every chemical element a learned embedding vector.

For example,

```text
Silicon

↓

64-dimensional embedding
```

and

```text
Oxygen

↓

64-dimensional embedding
```

These embeddings provide numerical representations of **individual atoms**.

However, a crystal is not simply a collection of isolated atoms.

Its properties depend on

* atomic composition,
* crystal structure,
* chemical bonding,
* local coordination,
* long-range interactions.

Therefore, the neural network must combine thousands of atomic features into **one representation describing the entire crystal**.

This representation is called the **latent crystal representation**.

It is one of the most important concepts in graph neural networks.

---

# What Does "Latent" Mean?

The word **latent** means

> **hidden or internal.**

A latent representation is therefore an internal numerical description learned by the neural network.

Unlike

* density,
* lattice constant,
* electronegativity,

latent features are **not directly measured physical quantities**.

Instead,

they are automatically learned because they help predict the target property.

---

# From Atoms to a Crystal Representation

Consider a simple crystal.

```text
Si

O

O
```

Initially,

each atom possesses its own embedding.

```text
Si

↓

[0.41, -0.22, ..., 0.18]

O

↓

[-0.73, 0.52, ..., -0.44]

O

↓

[-0.73, 0.52, ..., -0.44]
```

After several message-passing layers,

every atom has incorporated information from its neighbors.

The updated node features now encode

* atomic identity,
* local environment,
* neighboring atoms,
* bond geometry.

The graph pooling layer then combines all node features into a single vector.

Conceptually,

```text
Updated Atom Features

↓

Graph Pooling

↓

Crystal Representation

↓

[0.26, -0.41, ..., 0.83]
```

This vector represents the **entire material**.

---

# Why Is a Single Crystal Vector Necessary?

The prediction layer expects a fixed-size input.

However,

different crystals contain different numbers of atoms.

For example,

```text
Diamond

8 atoms

↓

?

Silicon Supercell

128 atoms

↓

?

Perovskite

40 atoms

↓

?
```

The prediction network cannot directly process graphs of arbitrary size.

Pooling solves this problem.

Regardless of whether a crystal contains

8 atoms

or

800 atoms,

pooling produces

one fixed-length vector.

For example,

```text
Crystal

↓

Pooling

↓

128-dimensional vector
```

The prediction layer always receives vectors of identical size.

---

# What Information Does the Latent Representation Contain?

The latent crystal representation is expected to summarize everything relevant for prediction.

For example,

it may encode information about

* elemental composition,
* average bond strength,
* crystal symmetry,
* coordination environments,
* electronic interactions,
* structural motifs.

Importantly,

these quantities are **not stored separately**.

Instead,

they are distributed throughout the latent vector.

---

# Learning Without Explicit Rules

Suppose we train MEGNet to predict

formation energy.

Nobody tells the model

* which coordination numbers matter,
* which bond angles are important,
* how electronegativity affects stability.

Instead,

the optimizer gradually adjusts the latent representation so that crystals with similar formation energies occupy similar regions of latent space.

This organization emerges automatically.

---

# Understanding Latent Space

Imagine that every crystal is represented by one point in a high-dimensional space.

Initially,

the points are randomly distributed.

```text
Random Space

•

      •

            •

   •

          •
```

After training,

the organization becomes much more meaningful.

Crystals with similar chemistry and properties begin clustering together.

```text
Learned Space

Oxides

••••

Semiconductors

••••

Metals

••••

Layered Materials

••••
```

The neural network has effectively organized the entire materials database.

---

# Why Similar Materials Cluster Together

Suppose two materials are

```text
MgO

CaO
```

Although they are different compounds,

both are

* ionic oxides,
* wide-band-gap insulators,
* rocksalt structures.

Because they often exhibit similar target properties,

their latent representations become similar.

Now consider

```text
Graphite

Diamond
```

Both consist entirely of carbon,

yet their crystal structures are completely different.

Consequently,

their latent vectors remain separated.

The latent representation therefore captures

both composition

and

structure.

---

# Visualizing Latent Space

A latent vector may contain

128

or

256 dimensions.

Humans cannot visualize such spaces directly.

Therefore,

dimensionality reduction techniques are commonly used.

Popular choices include

* Principal Component Analysis (PCA),
* t-SNE,
* UMAP.

These methods project the high-dimensional latent vectors into

two

or

three dimensions,

allowing researchers to inspect the learned organization.

---

# Example Using PCA

Suppose PCA projects thousands of crystal representations into two dimensions.

The visualization may resemble

```text
                   Wide Band Gap

                        ••••

              Oxides

             •••••••

Metals

••••••

Layered Materials

           •••••

Semiconductors

                 •••••
```

Each point represents

one crystal.

Nearby points correspond to materials with similar learned representations.

---

# Scientific Meaning of Clusters

Clusters in latent space often correspond to meaningful material families.

Researchers have observed clusters representing

* oxides,
* nitrides,
* carbides,
* sulfides,
* metallic alloys,
* semiconductors,
* layered materials.

Remarkably,

the neural network is never explicitly instructed to create these categories.

They emerge because similar materials require similar internal representations for accurate prediction.

---

# Discovering New Materials

Latent space is valuable not only for visualization,

but also for materials discovery.

Suppose a new crystal is projected into latent space.

If it lies close to

known high-temperature superconductors,

it may exhibit similar behavior.

Likewise,

a candidate battery material located near successful cathode materials may deserve experimental investigation.

Thus,

latent space provides a powerful tool for

**similarity search**.

---

# Interpolation in Latent Space

Because latent space is continuous,

it becomes possible to interpolate between materials.

Conceptually,

```text
Material A

•

↓

Intermediate Region

↓

•

↓

Material B
```

Materials located between known compounds may possess intermediate properties.

This idea has inspired many generative models for inverse materials design.

---

# Latent Space Reflects the Training Objective

An important point is that latent representations are **task-dependent**.

Suppose we train one MEGNet model to predict

formation energy,

and another to predict

band gap.

Although both models analyze the same crystals,

their latent spaces may differ substantially.

The formation-energy model emphasizes

thermodynamic stability.

The band-gap model emphasizes

electronic structure.

Consequently,

each model organizes materials differently.

---

# Limitations of Latent Representations

Although latent space is extremely informative,

it also has limitations.

First,

individual dimensions rarely possess clear physical interpretations.

Unlike handcrafted descriptors,

latent features are distributed representations.

Second,

visualizations produced by PCA,

t-SNE,

or UMAP inevitably lose information because they compress hundreds of dimensions into only two or three.

Finally,

the organization of latent space depends strongly on

* dataset quality,
* model architecture,
* training objective.

Therefore,

latent-space visualizations should be interpreted carefully.

---

# Latent Space in Modern Materials Informatics

Latent crystal representations have become central to many areas of materials informatics.

They are widely used for

* clustering material families,
* identifying chemically similar compounds,
* anomaly detection,
* transfer learning,
* active learning,
* inverse materials design,
* generative crystal models.

Many recent graph neural network architectures build directly upon these learned representations.

---

# Summary

The latent crystal representation is MEGNet's internal numerical description of an entire material.

Beginning with learned atomic embeddings,

successive message-passing layers integrate information from neighboring atoms until each node encodes its local chemical environment.

Graph pooling then combines these updated node features into a single fixed-length vector representing the complete crystal.

Crystals with similar chemistry and physical behavior naturally cluster together in this latent space, making it a valuable tool for visualization, similarity search, and materials discovery.

However, latent representations describe **what** the network has learned—not **how** information moves through the crystal during learning.

To understand that process, we must examine the flow of information itself.

In the next section, **15.10.3 Message Passing Interpretation**, we will investigate how atomic information propagates across crystal graphs, how neighboring atoms influence one another during prediction, and how successive message-passing layers progressively construct a chemically meaningful representation of the entire crystal.

## 15.10.3 Message Passing Interpretation: Understanding Information Flow in MEGNet

So far, we have learned that MEGNet represents a crystal as a graph and gradually transforms this graph into a latent crystal representation.

However, one important question remains unanswered:

> **How does information actually travel through the graph?**

This question is central to understanding Graph Neural Networks.

Unlike conventional neural networks, where information flows through fixed layers, MEGNet continuously exchanges information **between neighboring atoms** through a process known as **message passing**.

Understanding message passing is essential because it reveals **how local atomic environments influence the final prediction**.

---

# Revisiting the Crystal Graph

Consider a simple crystal fragment.

```text id="mp01"
      O
     /
Si —— O
     \
      O
```

Each atom is represented as a node.

Each chemical bond (or neighboring interaction) is represented as an edge.

Initially,

every node contains only its own atomic embedding.

For example,

```text id="mp02"
Silicon

↓

Embedding

Oxygen

↓

Embedding
```

At this point,

the silicon atom has **no information** about the surrounding oxygen atoms.

Similarly,

each oxygen atom knows nothing about silicon.

The graph consists of isolated node features connected only by edges.

---

# Why Message Passing Is Necessary

Suppose we want to predict

the band gap of silicon dioxide.

The band gap is **not** determined by silicon alone.

It depends on

* the neighboring oxygen atoms,
* bond lengths,
* bond angles,
* local coordination,
* crystal symmetry.

If silicon ignored its neighboring atoms,

accurate prediction would be impossible.

Therefore,

the node features must exchange information.

This exchange is message passing.

---

# Information Flow During One Message Passing Layer

Consider again the SiO₂ fragment.

Initially,

```text id="mp03"
Si

↓

Own Features Only
```

After one message-passing step,

each neighboring oxygen sends information to silicon.

Conceptually,

```text id="mp04"
      O

      ↓

O → Si ← O
```

The silicon node now combines

* its own embedding,
* information received from every neighboring oxygen.

Its feature vector becomes

```text id="mp05"
Updated Silicon Features

=

Original Silicon Features

+

Neighbor Information
```

The same process occurs simultaneously for every atom in the graph.

---

# Every Atom Learns From Its Neighbors

Message passing is **symmetric**.

Silicon learns from oxygen,

but oxygen also learns from silicon.

For example,

```text id="mp06"
Oxygen

↓

Original Features

+

Neighbor Silicon Features

↓

Updated Oxygen Features
```

Thus,

every node continuously updates its representation based on its local chemical environment.

---

# One Layer Means One-Hop Communication

An important observation is that

one message-passing layer only exchanges information between **direct neighbors**.

Suppose the graph is

```text id="mp07"
A — B — C
```

After one layer,

A receives information from B,

and C receives information from B.

However,

A has **not yet received information from C**.

The communication distance is only

one edge.

---

# Two Message Passing Layers

Now consider two consecutive message-passing layers.

### Layer 1

```text id="mp08"
A ←→ B ←→ C
```

After Layer 1,

B contains information from both A and C.

### Layer 2

During the second layer,

B sends its updated information back.

```text id="mp09"
A ← B → C
```

Since B already knows about C,

A now indirectly receives information originating from C.

Consequently,

after two layers,

A has learned about atoms two hops away.

---

# Information Propagation

As additional message-passing layers are added,

information spreads progressively farther through the crystal.

```text id="mp10"
Layer 1

Immediate Neighbors

↓

Layer 2

Neighbors of Neighbors

↓

Layer 3

Three-Hop Neighborhood

↓

...

↓

Entire Crystal
```

Eventually,

each atom contains information describing a significant portion of the crystal.

---

# Chemical Interpretation

Consider a transition-metal oxide.

```text id="mp11"
O

↓

Fe

↓

O
```

Initially,

the iron atom knows only that it is iron.

After message passing,

its feature vector incorporates information about

* surrounding oxygen atoms,
* local bond distances,
* coordination geometry,
* neighboring transition metals.

Consequently,

the updated iron representation becomes

a description of its **chemical environment**,

rather than merely its atomic identity.

---

# Local Environment Is More Important Than Atomic Identity

Two atoms of the same element can have very different environments.

For example,

consider two carbon atoms.

One belongs to

diamond.

The other belongs to

graphite.

Initially,

both possess identical atomic embeddings.

```text id="mp12"
Carbon

↓

Embedding
```

After message passing,

their updated representations become very different.

Diamond carbon receives messages corresponding to

tetrahedral covalent bonding.

Graphite carbon receives messages corresponding to

planar sp² bonding.

Although both atoms are carbon,

their final node representations are no longer identical.

The network has learned the importance of local structure.

---

# Message Passing Is Repeated

A single message-passing layer captures only local information.

Therefore,

MEGNet stacks multiple graph network blocks.

Conceptually,

```text id="mp13"
Input Graph

↓

Message Passing

↓

Updated Graph

↓

Message Passing

↓

Updated Graph

↓

Message Passing

↓

Final Graph Representation
```

Each successive layer increases the receptive field of every atom.

---

# What Information Is Actually Passed?

A common misconception is that

atoms simply transmit their embedding vectors.

In reality,

the transmitted message depends on

* node features,
* edge features,
* global state features.

Conceptually,

```text id="mp14"
Message

=

f(

Atom Features,

Bond Features,

Global Features

)
```

Therefore,

the information exchanged already reflects

chemical bonding

and

the overall crystal environment.

This makes MEGNet significantly more expressive than models using only node features.

---

# Which Neighbors Matter Most?

Not every neighboring atom contributes equally.

Consider

```text id="mp15"
Central Atom

↓

Neighbor A

Neighbor B

Neighbor C
```

Some neighbors may exert

strong influence,

while others contribute relatively little.

Later sections on

* attention mechanisms,
* saliency maps,
* GNNExplainer,

will show how researchers identify the most influential neighbors.

---

# Message Passing Learns Chemical Rules

Although no chemical rules are explicitly programmed,

message passing gradually learns relationships such as

* oxygen strongly influences transition metals,
* tetrahedral coordination differs from octahedral coordination,
* long bonds contribute differently from short bonds,
* local geometry affects electronic structure.

These relationships emerge because they improve prediction accuracy.

Thus,

message passing serves as the mechanism through which chemical knowledge is learned.

---

# Visualizing Message Passing

Researchers often visualize message passing using colored graphs.

For example,

important atoms may be highlighted.

```text id="mp16"
Large Contribution

🔴

Medium Contribution

🟠

Small Contribution

🔵
```

Similarly,

important bonds can be displayed using thicker edges.

Such visualizations help explain why a particular prediction was made.

---

# Limitations of Message Passing Interpretation

Although message passing provides valuable insight,

interpreting it remains challenging.

Several difficulties arise.

First,

messages are high-dimensional vectors,

not simple numerical values.

Second,

multiple message-passing layers interact in complex nonlinear ways.

Third,

neighboring atoms influence one another simultaneously,

making it difficult to isolate individual contributions.

Consequently,

additional explainability techniques are often required.

---

# Why Understanding Message Passing Matters

Message passing is the mechanism through which MEGNet transforms

isolated atomic embeddings

into chemically meaningful crystal representations.

Without message passing,

every atom would remain independent,

and the model could never learn

* chemical bonding,
* local coordination,
* structural motifs,
* long-range interactions.

Understanding message passing therefore provides insight into **how MEGNet reasons about crystal structures**.

---

# Summary

Message passing is the core computational process that enables MEGNet to learn from crystal graphs.

During each message-passing layer,

every atom exchanges information with its neighbors,

allowing node features to evolve from simple atomic embeddings into rich descriptions of local chemical environments.

By stacking multiple graph network layers,

information propagates progressively farther through the crystal,

ultimately enabling the model to build a comprehensive representation of the entire material.

However,

message passing alone does not tell us **which atoms or bonds were most responsible for a specific prediction**.

To answer that question,

we require methods that quantify the importance of different parts of the crystal graph.

In the next section, **15.10.4 Feature Attribution Methods**, we will study techniques that identify the atoms, bonds, and structural motifs that contribute most strongly to MEGNet's predictions, providing a direct explanation for individual material property predictions.

## 15.10.4 Feature Attribution Methods: Identifying the Most Important Atoms and Bonds

In the previous section, we learned that information flows through a crystal graph via message passing.

Every atom exchanges information with its neighbors, gradually building a representation of the entire crystal.

Although this explains **how** information propagates, it does not answer another important question:

> **Which atoms were actually responsible for the final prediction?**

Similarly,

we may ask

* Which chemical bonds were most influential?
* Which local atomic environments dominated the prediction?
* Which part of the crystal should a scientist modify to improve a material property?

Answering these questions requires **feature attribution methods**.

These methods attempt to measure the contribution of different parts of the input graph to the model's prediction.

---

# What Is Feature Attribution?

Feature attribution is the process of assigning an **importance score** to each input feature.

For an ordinary machine learning model,

the input features might be

* density,
* atomic radius,
* electronegativity,
* lattice constant.

Feature attribution determines

how much each feature contributed to the prediction.

Graph neural networks are different.

The input is not a simple feature vector.

Instead, the input consists of

* nodes,
* edges,
* graph connectivity,
* node features,
* edge features,
* global features.

Therefore,

feature attribution becomes considerably more challenging.

---

# A Simple Example

Suppose MEGNet predicts

```text id="fa01"
Band Gap

=

2.31 eV
```

The crystal contains

```text id="fa02"
Silicon

Oxygen

Oxygen
```

A feature attribution method may produce

| Atom | Importance |
| ---- | ---------: |
| Si   |       0.48 |
| O₁   |       0.29 |
| O₂   |       0.23 |

This result suggests that

the silicon atom contributed most strongly to the prediction.

Notice that

the importance scores sum to

1.0,

making them easy to interpret.

---

# Bond Attribution

Sometimes,

the most important information lies not in the atoms,

but in the bonds connecting them.

Suppose a crystal contains

```text id="fa03"
Fe — O

Fe — O

Fe — Fe
```

A feature attribution algorithm might produce

| Bond  | Importance |
| ----- | ---------: |
| Fe–O  |       0.41 |
| Fe–O  |       0.38 |
| Fe–Fe |       0.21 |

This result suggests that

metal–oxygen interactions dominate the prediction.

---

# Local Environment Attribution

Instead of evaluating atoms or bonds individually,

some methods evaluate complete local environments.

For example,

consider

```text id="fa04"
Central Atom

↓

Six Neighboring Oxygen Atoms

↓

Octahedral Coordination
```

The attribution algorithm may conclude

that the **entire octahedral environment**

is primarily responsible for the predicted formation energy.

This type of interpretation is particularly useful in crystallography.

---

# Importance Scores

Feature attribution methods typically assign a numerical importance score.

For example,

| Feature | Score |
| ------- | ----: |
| Atom A  |  0.63 |
| Atom B  |  0.24 |
| Atom C  |  0.13 |

Higher scores indicate

greater influence on the prediction.

A score near zero suggests

that the corresponding atom had little effect.

---

# Positive and Negative Contributions

Not all features increase the prediction.

Some decrease it.

Suppose MEGNet predicts

formation energy.

A feature attribution analysis might reveal

| Feature                  | Contribution |
| ------------------------ | -----------: |
| Strong Metal–Oxygen Bond |        −0.81 |
| Long Bond Length         |        +0.44 |
| Vacancy Defect           |        +0.29 |

Negative contributions reduce the predicted energy,

often indicating increased stability.

Positive contributions increase the predicted energy,

often indicating destabilization.

Thus,

feature attribution provides not only importance,

but also direction.

---

# Visualizing Feature Importance

Importance scores are usually displayed directly on the crystal structure.

For example,

important atoms may appear

larger,

brighter,

or warmer in color.

Conceptually,

```text id="fa05"
High Importance

🔴

Medium Importance

🟠

Low Importance

🔵
```

Similarly,

important bonds may be displayed using thicker lines.

These visualizations allow researchers to identify chemically significant regions immediately.

---

# Example: Lithium-Ion Battery Material

Suppose MEGNet predicts

a high intercalation voltage

for a lithium transition-metal oxide.

Feature attribution may reveal

```text id="fa06"
High Importance

↓

Transition Metal

↓

Neighboring Oxygen

↓

Li Pathway
```

This suggests

that

the transition-metal–oxygen network

is primarily responsible for the voltage prediction.

Such information guides future material design.

---

# Feature Attribution Is Local

An important property of feature attribution methods is that

they usually explain

**one prediction at a time**.

For example,

consider two crystals.

```text id="fa07"
Crystal A

↓

Band Gap

2.1 eV

Crystal B

↓

Band Gap

3.8 eV
```

The important atoms in Crystal A

may be completely different

from those in Crystal B.

Consequently,

feature attribution provides

**local explanations**,

not global chemical rules.

---

# Different Attribution Methods

Many algorithms have been developed for neural network interpretation.

Some of the most widely used include

* Gradient Saliency
* Integrated Gradients
* Grad-CAM (adapted for graphs)
* SHAP
* LIME
* GNNExplainer
* GraphMask

Each method computes importance differently.

Some analyze gradients,

others perturb the input,

while others learn explanatory subgraphs.

---

# Perturbation-Based Attribution

One intuitive approach is

to remove part of the input

and observe what happens.

Suppose an oxygen atom is removed.

If the prediction changes dramatically,

the oxygen atom was probably important.

Conceptually,

```text id="fa08"
Original Crystal

↓

Prediction

↓

Remove Atom

↓

New Prediction

↓

Large Difference

↓

Important Atom
```

This approach is easy to understand,

although computationally expensive.

---

# Gradient-Based Attribution

Another common strategy uses gradients.

Instead of removing atoms,

we ask

> **How sensitive is the prediction to small changes in each input feature?**

Large gradients indicate

that small changes strongly influence the prediction.

Small gradients indicate

relatively unimportant features.

Gradient-based methods are generally much faster than perturbation methods.

---

# Scientific Applications

Feature attribution has become increasingly important in materials informatics.

Researchers use it to investigate

* catalytic active sites,
* defect chemistry,
* dopant effectiveness,
* ionic diffusion pathways,
* structural instabilities,
* superconducting motifs,
* magnetic interactions.

Rather than simply predicting a property,

the model begins to explain

**why** the property exists.

---

# Limitations

Feature attribution methods are extremely useful,

but they are not perfect.

Several limitations should be recognized.

First,

different attribution algorithms may produce different explanations for the same prediction.

Second,

importance scores indicate

correlation,

not necessarily causation.

Finally,

graph neural networks contain highly nonlinear interactions,

making exact attribution inherently difficult.

Consequently,

feature attribution should always be interpreted alongside chemical knowledge and experimental evidence.

---

# From Attribution to Explainable AI

Feature attribution represents one important branch of explainable artificial intelligence.

However,

modern explainability extends beyond assigning importance scores.

Researchers also wish to

* identify the smallest subgraph responsible for a prediction,
* understand how gradients propagate,
* quantify uncertainty,
* explain predictions in terms of human-understandable concepts.

These objectives require more advanced explainability techniques.

---

# Summary

Feature attribution methods attempt to identify the atoms, bonds, and local structural environments that contribute most strongly to a MEGNet prediction.

Unlike message-passing analysis, which explains how information flows through the graph, feature attribution focuses on **which parts of the graph are most responsible for a specific prediction**.

These methods provide valuable scientific insight by highlighting chemically significant regions of a crystal, supporting hypothesis generation, materials design, and model validation.

However, feature attribution alone cannot fully explain the reasoning of a graph neural network.

Modern explainable AI includes additional techniques that reveal influential subgraphs, trace gradient flow, and provide more faithful interpretations of complex graph models.

In the next section, **15.10.5 Explainable AI for Graph Neural Networks**, we will explore advanced interpretation methods including **Saliency Maps, Integrated Gradients, GNNExplainer, GraphMask, and SHAP**, and learn how these techniques provide deeper insight into MEGNet's predictions while preserving scientific interpretability.

## 15.10.6 Uncertainty Estimation: Knowing When MEGNet Can Be Trusted

Throughout this chapter, we have focused on improving the predictive accuracy of MEGNet.

Suppose our trained model predicts

```text id="unc01"
Formation Energy

=

−2.84 eV/atom
```

The prediction appears reasonable.

However, an important question remains:

> **How confident is the model in this prediction?**

A prediction without an estimate of uncertainty can be misleading.

Two predictions may have identical values,

yet one may be highly reliable while the other may be little more than an educated guess.

Understanding prediction uncertainty is therefore a critical aspect of modern materials informatics.

---

# Why Uncertainty Matters

Consider two hypothetical crystals.

### Crystal A

```text id="unc02"
Predicted Band Gap

=

2.10 eV
```

Estimated uncertainty

```text id="unc03"
±0.03 eV
```

### Crystal B

```text id="unc04"
Predicted Band Gap

=

2.10 eV
```

Estimated uncertainty

```text id="unc05"
±1.25 eV
```

Although both predictions are identical,

their reliability is dramatically different.

The first prediction is highly trustworthy.

The second should be treated with considerable caution.

Thus,

a complete machine learning prediction consists of

```text id="unc06"
Prediction

+

Uncertainty
```

rather than

```text id="unc07"
Prediction Only
```

---

# What Is Uncertainty?

Uncertainty measures

how unsure a model is about its prediction.

In machine learning,

uncertainty generally reflects one of two situations.

1. The model has **insufficient knowledge**.

2. The data themselves contain **noise or ambiguity**.

These two situations correspond to two different types of uncertainty.

---

# Aleatoric Uncertainty

The first type is called **aleatoric uncertainty**.

Aleatoric uncertainty originates from the data themselves.

Examples include

* experimental measurement error,
* thermal fluctuations,
* instrument noise,
* imperfect DFT calculations,
* inconsistent labels.

Even a perfect machine learning model cannot eliminate this uncertainty because it is inherent in the observations.

---

### Example

Suppose several laboratories measure

the thermal conductivity

of the same material.

Their results might be

| Laboratory | Measured Value (W·m⁻¹·K⁻¹) |
| ---------- | -------------------------: |
| A          |                        152 |
| B          |                        149 |
| C          |                        154 |
| D          |                        151 |

The variation arises from experimental uncertainty,

not from the neural network.

This is aleatoric uncertainty.

---

# Epistemic Uncertainty

The second type is **epistemic uncertainty**.

This uncertainty reflects

the model's lack of knowledge.

It occurs because

* the training dataset is too small,
* certain crystal types are missing,
* the model has never encountered similar materials.

Unlike aleatoric uncertainty,

epistemic uncertainty **can decrease** as more training data become available.

---

### Example

Suppose the training dataset contains

thousands of oxides

but very few nitrides.

When predicting a new nitride,

MEGNet may be uncertain because

it has limited prior experience with this class of materials.

The uncertainty reflects

missing knowledge,

not noisy measurements.

---

# Comparing the Two Types

| Property                  | Aleatoric          | Epistemic            |
| ------------------------- | ------------------ | -------------------- |
| Source                    | Noisy data         | Limited knowledge    |
| Reduced with more data?   | No                 | Yes                  |
| Present in perfect model? | Yes                | No                   |
| Typical Cause             | Experimental error | Sparse training data |

Understanding this distinction is essential when interpreting model confidence.

---

# Why Materials Informatics Needs Uncertainty

Materials discovery often involves exploring

previously unknown compounds.

For many candidate materials,

the model operates far from its training distribution.

Without uncertainty estimation,

the model may produce

confident-looking predictions

that are actually unreliable.

Uncertainty estimation helps researchers distinguish between

```text id="unc08"
Reliable Prediction

↓

Experimental Validation
```

and

```text id="unc09"
Highly Uncertain Prediction

↓

Collect More Data First
```

This saves both time and experimental resources.

---

# Confidence Intervals

A common way to express uncertainty is through a confidence interval.

For example,

```text id="unc10"
Bulk Modulus

=

165 ± 4 GPa
```

The predicted value is

165 GPa,

while the model estimates that the true value is likely to lie within

approximately

161–169 GPa.

Confidence intervals communicate both

the prediction

and

its reliability.

---

# Monte Carlo Dropout

One of the simplest uncertainty estimation techniques is

**Monte Carlo Dropout**.

Normally,

dropout is disabled during inference.

Monte Carlo Dropout intentionally leaves dropout active.

The same crystal is evaluated multiple times.

```text id="unc11"
Crystal

↓

Prediction 1

↓

Prediction 2

↓

Prediction 3

↓

Prediction 4

↓

Prediction 5
```

Because different neurons are randomly deactivated,

the predictions vary slightly.

Large variation indicates

high uncertainty.

Small variation indicates

high confidence.

---

### Example

Suppose five predictions are obtained.

| Run | Band Gap (eV) |
| --: | ------------: |
|   1 |          1.95 |
|   2 |          1.98 |
|   3 |          1.97 |
|   4 |          1.94 |
|   5 |          1.96 |

The predictions are tightly clustered.

The model appears confident.

Now consider

| Run | Band Gap (eV) |
| --: | ------------: |
|   1 |          0.82 |
|   2 |          2.74 |
|   3 |          1.35 |
|   4 |          3.01 |
|   5 |          1.89 |

The predictions vary dramatically,

indicating substantial uncertainty.

---

# Deep Ensembles

Another highly effective approach is

**Deep Ensembles**.

Instead of training

one MEGNet model,

we train

multiple independent models.

```text id="unc12"
MEGNet 1

↓

MEGNet 2

↓

MEGNet 3

↓

MEGNet 4

↓

Average Prediction
```

Each model begins with different random initialization.

Agreement among the models implies

high confidence.

Disagreement indicates

greater uncertainty.

---

# Advantages of Deep Ensembles

Deep ensembles are widely regarded as one of the most reliable uncertainty estimation methods because they

* capture model uncertainty effectively,
* require no architectural modifications,
* perform well across many applications.

The primary disadvantage is computational cost,

since multiple models must be trained.

---

# Bayesian Neural Networks

A more advanced approach is the

**Bayesian Neural Network (BNN)**.

In ordinary neural networks,

each parameter has

one fixed value.

For example,

```text id="unc13"
Weight

=

0.42
```

In Bayesian neural networks,

every parameter is treated as

a probability distribution.

```text id="unc14"
Weight

~

Normal Distribution
```

Rather than predicting

one value,

the model predicts

a distribution of possible values.

This naturally provides uncertainty estimates.

---

# Out-of-Distribution Detection

One major challenge in materials informatics is

predicting materials that differ substantially from those in the training set.

Such materials are called

**out-of-distribution (OOD)** samples.

For example,

suppose MEGNet was trained almost exclusively on

* oxides,
* nitrides,
* carbides.

Now it encounters

a complex high-entropy alloy.

Because this material lies outside the training distribution,

the model should report

high uncertainty.

OOD detection is therefore an important safeguard against overconfident predictions.

---

# Uncertainty in Active Learning

Uncertainty estimation is a key component of

**active learning**.

Instead of randomly selecting new materials for expensive DFT calculations,

the algorithm identifies

the materials with the highest uncertainty.

Conceptually,

```text id="unc15"
Candidate Materials

↓

Predict Properties

↓

Estimate Uncertainty

↓

Select Most Uncertain Materials

↓

Perform DFT

↓

Retrain Model
```

This strategy maximizes information gain while minimizing computational cost.

---

# Scientific Applications

Uncertainty estimation has become increasingly important in

* autonomous materials discovery,
* high-throughput screening,
* inverse materials design,
* Bayesian optimization,
* experimental planning,
* robotics-driven laboratories.

Rather than treating every prediction equally,

scientists prioritize predictions

that are both

accurate

and

confident.

---

# Limitations

Although uncertainty estimation is powerful,

it is not perfect.

Several challenges remain.

First,

different uncertainty estimation methods may produce different confidence estimates.

Second,

uncertainty itself must be calibrated.

A model claiming

95% confidence

should indeed be correct

approximately 95% of the time.

Finally,

accurate uncertainty estimation often increases computational cost.

Consequently,

practical applications require balancing

accuracy,

interpretability,

and

efficiency.

---

# From Prediction to Scientific Decision-Making

Traditional machine learning answers

> **What is the predicted property?**

Modern materials informatics asks two additional questions.

> **Why was this prediction made?**

and

> **How confident is the prediction?**

Together,

prediction,

interpretability,

and

uncertainty

form the foundation of trustworthy AI for scientific research.

---

# Summary

Uncertainty estimation quantifies the confidence of a MEGNet prediction, enabling researchers to distinguish reliable predictions from uncertain ones.

Two principal forms of uncertainty exist:

* **Aleatoric uncertainty**, arising from inherent noise in the data.
* **Epistemic uncertainty**, arising from limited model knowledge and reducible through additional training data.

Methods such as Monte Carlo Dropout, Deep Ensembles, Bayesian Neural Networks, and out-of-distribution detection allow MEGNet to estimate prediction confidence and support safer decision-making in materials discovery.

Combined with interpretability techniques, uncertainty estimation transforms graph neural networks into trustworthy scientific tools suitable for high-throughput screening, autonomous experimentation, and AI-assisted materials design.

However, theoretical understanding alone is not sufficient for research.

To fully appreciate these concepts, we must see them applied to real materials.

In the next section, **15.10.7 Case Studies: Interpreting Real MEGNet Predictions**, we will analyze published examples involving **formation energy, band gap, elastic properties, and battery materials**, demonstrating how interpretability and uncertainty estimation work together to reveal the scientific reasoning behind MEGNet's predictions in practical materials informatics research.

## 15.10.7 Case Studies: Interpreting Real MEGNet Predictions

Throughout this chapter, we have explored the theoretical foundations of interpreting Graph Neural Networks.

We learned

* how atomic embeddings are learned,
* how latent crystal representations emerge,
* how message passing propagates chemical information,
* how feature attribution identifies important atoms,
* how explainable AI techniques interpret predictions,
* how uncertainty estimation measures model confidence.

Although these concepts are important individually,

their real value becomes apparent only when they are applied to actual materials science problems.

In this section, we will examine several representative case studies that demonstrate how researchers use MEGNet to gain scientific insight rather than merely produce accurate predictions.

The goal is not only to understand **what** the model predicts, but also **why** it makes those predictions and **how much confidence** we should place in them.

---

# Case Study 1 — Formation Energy Prediction

Formation energy is one of the most common targets in materials informatics because it is directly related to thermodynamic stability.

Suppose a trained MEGNet model predicts the formation energy of a previously unseen oxide.

```text id="cs01"
Predicted Formation Energy

=

−3.12 eV/atom
```

with an uncertainty estimate of

```text id="cs02"
±0.04 eV/atom
```

The small uncertainty indicates that the prediction is highly reliable.

However, the numerical value alone provides little scientific understanding.

---

## Step 1 — Feature Attribution

Feature attribution highlights the atoms contributing most strongly to the prediction.

| Atom             | Importance |
| ---------------- | ---------: |
| Transition Metal |       0.53 |
| Oxygen 1         |       0.19 |
| Oxygen 2         |       0.16 |
| Oxygen 3         |       0.12 |

The transition-metal atom contributes more than half of the prediction.

This suggests that

the local transition-metal environment largely determines the crystal's stability.

---

## Step 2 — Message Passing Interpretation

Analyzing the message-passing layers reveals that

the strongest information exchange occurs between

```text id="cs03"
Transition Metal

↓

Nearest Oxygen Atoms
```

Long-range interactions contribute relatively little.

This indicates that

local coordination dominates the formation energy.

---

## Step 3 — Scientific Interpretation

From a materials science perspective,

this result is reasonable.

Formation energy depends strongly on

* bond strength,
* oxidation state,
* local coordination,
* crystal chemistry.

The explanation therefore agrees with established physical principles.

---

# Case Study 2 — Band Gap Prediction

Electronic band gap is considerably more complex than formation energy because it depends on

* orbital hybridization,
* crystal symmetry,
* chemical bonding,
* long-range electronic interactions.

Suppose MEGNet predicts

```text id="cs04"
Band Gap

=

2.84 eV
```

---

## Atomic Importance

Feature attribution identifies

| Region                      | Relative Importance |
| --------------------------- | ------------------: |
| Transition Metal d Orbitals |                0.47 |
| Oxygen p Orbitals           |                0.35 |
| Remaining Atoms             |                0.18 |

Although MEGNet never explicitly receives orbital information,

it has learned representations that correlate with these electronic interactions.

---

## Latent Representation

Projecting the latent crystal representation using UMAP reveals

that the material lies close to

known semiconductors.

Conceptually,

```text id="cs05"
Known Semiconductors

•••••

↓

New Material

•

↓

Wide Band Gap Oxides

•••••
```

This suggests that

the model recognizes similarities with previously studied semiconductor families.

---

## Scientific Conclusion

The prediction is not merely numerical.

The latent representation indicates

that the new material belongs to a chemically meaningful family.

This increases confidence in the prediction.

---

# Case Study 3 — Elastic Modulus Prediction

Elastic properties depend strongly on

* bond stiffness,
* crystal packing,
* coordination geometry.

Suppose MEGNet predicts

```text id="cs06"
Bulk Modulus

=

215 GPa
```

---

## Graph Interpretation

GNNExplainer identifies the following explanatory subgraph.

```text id="cs07"
Central Metal Atom

↓

Six Strong Covalent Bonds

↓

Nearest Neighbor Network
```

Remarkably,

only a small fraction of the entire crystal graph is required to reproduce the prediction.

---

## Physical Interpretation

The identified subgraph corresponds to

the primary load-bearing region of the crystal.

This agrees with classical elasticity theory,

which predicts that

strong directional bonds dominate elastic stiffness.

---

# Case Study 4 — Battery Cathode Materials

Suppose MEGNet is trained to predict

average lithium intercalation voltage.

A candidate material produces

```text id="cs08"
Predicted Voltage

=

4.18 V
```

---

## Explainability Analysis

Integrated Gradients highlight

```text id="cs09"
Transition Metal

↓

Oxygen Network

↓

Lithium Diffusion Path
```

The lithium atoms themselves contribute surprisingly little.

Instead,

the transition-metal–oxygen framework dominates the prediction.

---

## Scientific Insight

This observation agrees with battery chemistry.

The voltage is controlled primarily by

changes in transition-metal oxidation states,

rather than by lithium alone.

Thus,

the explanation reinforces existing electrochemical theory.

---

# Case Study 5 — Detecting an Unreliable Prediction

Suppose MEGNet predicts

```text id="cs10"
Thermal Conductivity

=

920 W/m·K
```

At first glance,

this appears impressive.

However,

uncertainty estimation produces

```text id="cs11"
±430 W/m·K
```

The uncertainty is extremely large.

---

## Why Is the Uncertainty High?

Inspection of the dataset reveals

that

the crystal belongs to

a family almost absent from the training database.

Consequently,

the model is making an extrapolation.

---

## Recommended Action

Instead of trusting the prediction immediately,

researchers should

* perform DFT calculations,
* obtain experimental measurements,
* expand the training dataset.

This example illustrates why uncertainty estimation is essential for responsible materials discovery.

---

# Case Study 6 — Discovering Similar Materials Through Latent Space

Suppose researchers develop

a new thermoelectric material.

After projection into latent space,

the crystal appears close to

several well-known thermoelectric compounds.

```text id="cs12"
Known Thermoelectrics

••••

↓

New Material

•

↓

Similar Compounds

•••
```

Although the material has never been experimentally tested,

its location within latent space suggests

that it may possess similar transport properties.

This approach is increasingly used for

candidate screening

before expensive calculations.

---

# Case Study 7 — Detecting Dataset Bias

Interpretability can also reveal problems with the training data.

Suppose saliency maps consistently emphasize

crystal volume,

while largely ignoring

chemical composition.

This observation raises important questions.

Perhaps

* the dataset contains mostly one chemical family,
* composition varies little,
* volume accidentally correlates with the target property.

The model may therefore have learned

a shortcut

rather than genuine chemistry.

Interpretability helps detect such hidden biases before deployment.

---

# Lessons from These Case Studies

Although the predicted properties differ,

the workflow remains remarkably consistent.

```text id="cs13"
Prediction

↓

Interpretation

↓

Uncertainty

↓

Scientific Reasoning

↓

Materials Decision
```

Every successful materials informatics study follows this general strategy.

The machine learning model becomes

not merely a predictor,

but a scientific assistant.

---

# Combining Multiple Interpretation Techniques

In practice,

researchers rarely rely on a single explainability method.

Instead,

they combine several complementary approaches.

| Technique                  | Purpose                            |
| -------------------------- | ---------------------------------- |
| Atomic Embeddings          | Understand learned chemistry       |
| Latent Space Visualization | Identify similar materials         |
| Message Passing Analysis   | Study information flow             |
| Feature Attribution        | Identify important atoms and bonds |
| Integrated Gradients       | Measure feature sensitivity        |
| GNNExplainer               | Find explanatory subgraphs         |
| SHAP                       | Quantify feature contributions     |
| Uncertainty Estimation     | Assess prediction confidence       |

Using multiple methods provides

more reliable scientific conclusions.

---

# From Prediction to Scientific Discovery

The true strength of MEGNet lies not in achieving a slightly lower prediction error,

but in enabling scientific discovery.

A successful model should help researchers

* understand chemical bonding,
* recognize structural motifs,
* identify important atomic environments,
* estimate confidence,
* prioritize experiments,
* accelerate materials discovery.

When combined with domain knowledge,

graph neural networks become partners in scientific research rather than simple prediction engines.

---

# Summary

The case studies presented in this section demonstrate how MEGNet predictions can be interpreted using atomic embeddings, latent representations, message-passing analysis, feature attribution, explainable AI methods, and uncertainty estimation.

Across diverse applications—including formation energy, band gap, elastic properties, battery materials, and thermoelectric compounds—the same interpretability framework consistently transforms numerical predictions into scientifically meaningful insights.

Rather than asking only **"What property does the model predict?"**, researchers now ask three complementary questions:

* **What is the predicted property?**
* **Why did the model make this prediction?**
* **How confident is the prediction?**

Answering these questions enables graph neural networks to support trustworthy, explainable, and scientifically grounded materials discovery.

---

# Chapter 15 Conclusion

With the completion of **Section 15.10**, we have finished our comprehensive study of **MEGNet (Materials Graph Network)**.

Beginning with graph construction, we explored message passing, graph network blocks, training strategies, optimization methods, interpretability, explainable AI, and uncertainty estimation.

You should now understand not only how to train a MEGNet model for materials property prediction, but also how to interpret its predictions, evaluate its reliability, and extract meaningful scientific insights from its learned representations.

This knowledge forms a strong foundation for modern graph-based materials informatics.

In the next chapter, we will build upon this foundation by studying **M3GNet (Materials 3-Body Graph Network)**, a next-generation graph neural network that extends MEGNet by incorporating explicit three-body interactions, enabling significantly improved prediction of structural, energetic, and dynamical properties while providing a unified framework for interatomic potentials and materials property prediction.

## 15.11 Chapter Summary

In this chapter, we conducted a comprehensive study of the **Materials Graph Network (MEGNet)**, one of the most influential graph neural network architectures developed for materials informatics. Unlike traditional machine learning methods that rely on handcrafted descriptors, MEGNet learns directly from crystal structures by representing materials as graphs and applying message-passing operations to capture complex atomic interactions.

The chapter began by introducing the motivation behind graph neural networks in materials science. We discussed the limitations of conventional descriptor-based machine learning and explained why crystal structures are naturally represented as graphs, where atoms correspond to nodes, chemical bonds correspond to edges, and the entire crystal is represented by global state variables.

We then examined the architecture of MEGNet in detail. We learned how node features, edge features, and global state features are initialized and updated through successive graph network blocks. Unlike earlier graph neural network models, MEGNet updates all three types of features simultaneously, enabling the model to capture not only local atomic environments but also global material characteristics. We also studied the mathematical foundations of message passing and understood how repeated information exchange allows each atom to gradually build a richer representation of its surrounding chemical environment.

A significant portion of the chapter focused on graph pooling. Since crystals contain different numbers of atoms, graph pooling converts variable-sized crystal graphs into fixed-length vectors suitable for prediction tasks. We explored several pooling strategies, including sum pooling, mean pooling, max pooling, and set2set pooling, and discussed their advantages and limitations for different materials science applications.

The training procedure of MEGNet was then presented step by step. We examined dataset preparation, graph construction, batching, forward propagation, loss computation, backpropagation, and parameter optimization. We discussed commonly used loss functions for regression and classification problems, optimization algorithms such as Adam, learning-rate scheduling, early stopping, and techniques for preventing overfitting. These concepts provide the practical knowledge required to train MEGNet models on real materials datasets.

The chapter also introduced transfer learning within the MEGNet framework. We learned how models pretrained on large datasets, such as formation-energy databases, can be fine-tuned for smaller property-prediction tasks. This approach enables researchers to achieve high predictive accuracy even when only limited experimental data are available.

A major emphasis of the chapter was model interpretability. We investigated how MEGNet automatically learns chemically meaningful atomic embeddings that often reflect periodic trends without explicitly encoding chemical knowledge. We studied latent crystal representations, which organize materials into meaningful regions of high-dimensional feature space, allowing researchers to identify chemically similar compounds and discover hidden relationships among materials.

The chapter further explored message-passing interpretation, demonstrating how information propagates through crystal graphs and how local atomic environments influence final predictions. Building on this foundation, we introduced feature attribution methods that identify the atoms, bonds, and structural motifs most responsible for a given prediction.

Modern explainable artificial intelligence techniques were then discussed in detail. We examined gradient-based methods such as Saliency Maps and Integrated Gradients, graph-specific approaches including GNNExplainer and GraphMask, and game-theoretic methods such as SHAP. Together, these approaches transform MEGNet from a purely predictive model into a scientifically interpretable tool capable of revealing the reasoning behind its predictions.

Recognizing that prediction accuracy alone is insufficient for scientific applications, we devoted considerable attention to uncertainty estimation. We distinguished between aleatoric uncertainty, arising from noisy observations, and epistemic uncertainty, resulting from limited model knowledge. We also explored practical uncertainty estimation techniques, including Monte Carlo Dropout, Deep Ensembles, Bayesian Neural Networks, and out-of-distribution detection, all of which help determine whether a prediction should be trusted.

Finally, we studied several realistic case studies involving formation energy, band gap, elastic properties, battery materials, thermoelectric compounds, and dataset bias detection. These examples demonstrated how interpretability, explainability, and uncertainty estimation work together to transform machine learning predictions into scientifically meaningful insights that support materials discovery.

Overall, MEGNet represents a major milestone in the evolution of materials informatics. By combining graph representations, message passing, learned atomic embeddings, global state modeling, transfer learning, explainable AI, and uncertainty estimation, MEGNet provides a powerful framework for predicting and understanding material properties directly from crystal structures. Its influence extends beyond property prediction, serving as the conceptual foundation for many subsequent graph neural network architectures, including **M3GNet**, **ALIGNN**, and several modern universal interatomic potential models.

After completing this chapter, the reader should be able to:

* Explain why graph neural networks are particularly suitable for materials science.
* Represent crystal structures as graphs containing node, edge, and global features.
* Describe the architecture and workflow of the MEGNet model.
* Understand the mathematical principles of message passing and graph pooling.
* Train and optimize MEGNet models for materials property prediction.
* Apply transfer learning to improve performance on small datasets.
* Interpret learned atomic embeddings and latent crystal representations.
* Analyze message passing and perform feature attribution.
* Use explainable AI techniques to interpret graph neural network predictions.
* Estimate prediction uncertainty and assess model reliability.
* Apply MEGNet to practical materials informatics research problems.

With a thorough understanding of MEGNet, the reader is now prepared to study more advanced graph neural network architectures that incorporate richer physical information. In the next chapter, we will explore **M3GNet (Materials Three-Body Graph Network)**, which extends the MEGNet framework by explicitly modeling three-body interactions and angular information, significantly improving the prediction of structural, energetic, mechanical, and dynamical properties while enabling state-of-the-art universal interatomic potentials for atomistic simulations.


