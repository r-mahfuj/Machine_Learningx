# Chapter 23 — Representation Learning for Materials

## 23.1 Introduction

Machine learning models are only as good as the information they receive. In materials science, the raw input is typically a crystal structure, a molecule, or an atomic configuration. However, machine learning algorithms cannot directly understand atoms, bonds, lattice vectors, or periodic crystal structures. Before a model can learn relationships between structure and properties, the material must first be represented in a mathematical form.

This mathematical representation is one of the most important concepts in modern artificial intelligence.

Historically, materials informatics relied on manually designed descriptors such as atomic radius, electronegativity, coordination number, density, lattice parameters, bond lengths, or handcrafted crystal fingerprints. These descriptors were carefully engineered by domain experts using physical intuition and chemical knowledge. While successful for many classical machine learning algorithms, handcrafted features possess several important limitations. They often fail to capture complex interactions, require extensive expert knowledge, and are difficult to generalize across different classes of materials.

The emergence of deep learning fundamentally changed this paradigm.

Instead of manually designing features, modern neural networks learn useful representations directly from data. During training, the network automatically discovers hidden patterns that best describe atomic environments, crystal structures, chemical interactions, and material properties. This automatic discovery of informative features is known as **representation learning**.

Representation learning has become one of the defining characteristics of modern artificial intelligence. Rather than relying on human-designed descriptors, deep neural networks construct hierarchical representations that become increasingly abstract as information propagates through the network. Early layers typically learn simple local geometric relationships, while deeper layers gradually capture chemical environments, bonding characteristics, crystal symmetry, long-range interactions, and eventually high-level physical concepts relevant to the prediction task.

For materials science, this capability is revolutionary.

Instead of asking

> "Which descriptors should we compute?"

researchers now ask

> "Can the neural network learn the descriptors automatically?"

The answer is yes.

Graph Neural Networks (GNNs), Crystal Graph Convolutional Neural Networks (CGCNN), MEGNet, ALIGNN, MACE, Allegro, and many other modern architectures all rely on learned representations rather than handcrafted descriptors. Their impressive predictive performance arises not simply because they contain millions of parameters, but because they learn physically meaningful embeddings of atomic structures during training.

Representation learning also provides several advantages beyond property prediction.

A learned representation can be reused for many downstream tasks, including

- property prediction,
- crystal classification,
- clustering,
- visualization,
- anomaly detection,
- similarity search,
- inverse materials design,
- generative modeling,
- active learning,
- autonomous materials discovery.

Consequently, a single representation may become a universal description of materials that can support multiple machine learning tasks simultaneously.

This idea is closely related to the concept of **foundation models**, which have transformed natural language processing and computer vision. Just as large language models learn universal representations of language, foundation models for materials aim to learn universal representations of crystal structures that can later be fine-tuned for specific scientific problems.

Throughout this chapter, we will study how modern machine learning algorithms learn these representations, how latent spaces emerge during training, how crystal embeddings can be visualized and interpreted, and how representation learning serves as the foundation for generative AI, active learning, explainable artificial intelligence, and autonomous materials discovery.

By the end of this chapter, the reader will understand

- why representations determine machine learning performance,
- how neural networks construct crystal embeddings,
- how latent spaces encode chemical knowledge,
- how self-supervised learning learns representations without labels,
- how contrastive learning improves material embeddings,
- how embedding visualization reveals hidden material families,
- how clustering discovers chemically similar compounds,
- how learned representations enable foundation models for materials science.

These concepts form the intellectual bridge between property prediction models discussed in previous chapters and the generative and autonomous AI systems that will be explored in the following chapters.

## 23.2 Why Representation Learning Matters

The success of every machine learning model ultimately depends on how information is represented. Regardless of the complexity of the learning algorithm, if the input representation fails to capture the underlying characteristics of a material, the model cannot accurately predict its properties. Consequently, representation learning has become one of the central research topics in modern artificial intelligence and materials informatics.

To understand its importance, consider the following question.

Suppose we wish to predict the band gap of silicon and diamond.

Although these materials have similar chemical compositions, their electronic properties differ significantly because of differences in crystal structure and atomic arrangement. If a machine learning model receives only the chemical formula,

```text
Si

C
```

it has almost no information regarding

- crystal symmetry,
- bond angles,
- coordination environments,
- local atomic geometry,
- long-range periodicity.

Without this structural information, accurate prediction becomes nearly impossible.

This illustrates a fundamental principle of machine learning:

> **The quality of a model is fundamentally limited by the quality of its representation.**

Historically, researchers attempted to overcome this limitation by manually designing descriptors based on chemical intuition. Examples include

- atomic number,
- electronegativity,
- atomic radius,
- ionization energy,
- coordination number,
- lattice parameters,
- density,
- bond length,
- bond angle,
- packing fraction.

These handcrafted descriptors transformed materials into numerical vectors suitable for machine learning algorithms.

For example, a material might be represented as

```text
Material

↓

[Atomic Number,

Atomic Radius,

Electronegativity,

Density,

Band Filling,

Coordination Number]
```

Such feature vectors enabled classical machine learning algorithms such as

- Linear Regression,
- Support Vector Machines,
- Random Forest,
- Gradient Boosting,
- XGBoost,

to predict material properties with reasonable success.

However, handcrafted descriptors suffer from several important limitations.

First, they require extensive domain expertise. Designing useful descriptors often demands years of experience in chemistry, crystallography, and materials science.

Second, handcrafted descriptors may omit important physical information. While researchers can encode known relationships, they cannot easily capture complex many-body interactions that emerge naturally during learning.

Third, descriptors designed for one class of materials may perform poorly for another. Features that successfully describe metallic alloys may be ineffective for ceramics, semiconductors, polymers, or molecular crystals.

Finally, manually engineered descriptors limit scientific discovery because the machine learning model can only learn relationships among variables explicitly provided by the researcher.

Deep learning addresses these limitations through automatic representation learning.

Instead of manually specifying which descriptors should be used, neural networks learn the representations directly from data during training.

The workflow changes fundamentally.

Traditional machine learning follows

```text
Material

↓

Handcrafted Features

↓

Machine Learning Model

↓

Prediction
```

Deep learning instead performs

```text
Material

↓

Neural Network

↓

Learned Representation

↓

Prediction
```

The representation itself becomes part of the optimization process.

During training, the network continuously modifies its internal parameters to construct increasingly informative representations of the input structures. These learned representations are often called **embeddings** or **latent features**.

One of the remarkable characteristics of representation learning is its hierarchical nature.

Early neural network layers typically identify simple local information.

For crystal structures, these may include

- neighboring atoms,
- local coordination,
- bond distances,
- bond orientations.

As information propagates through deeper layers, increasingly abstract concepts emerge.

Intermediate layers may learn

- chemical environments,
- coordination polyhedra,
- local symmetry,
- crystal motifs.

The deepest layers often encode highly abstract information related to

- electronic structure,
- thermodynamic stability,
- mechanical behavior,
- defect chemistry,
- material functionality.

This hierarchical learning process is illustrated conceptually below.

```text
Crystal Structure

        │

        ▼

Atomic Coordinates

        │

        ▼

Local Atomic Environments

        │

        ▼

Chemical Bonding Patterns

        │

        ▼

Crystal Motifs

        │

        ▼

Material Representation

        │

        ▼

Property Prediction
```

Each successive layer extracts more meaningful information than the previous one.

This automatic feature extraction explains why deep learning models often outperform classical machine learning methods on complex materials datasets.

Another major advantage of learned representations is their reusability.

Once a neural network has learned a meaningful representation of crystal structures, that representation can often be transferred to entirely different prediction tasks.

For example, a representation learned while predicting formation energy may also prove useful for

- band gap prediction,
- elastic constants,
- thermal conductivity,
- dielectric properties,
- magnetic ordering,
- defect formation energy.

This concept forms the basis of **transfer learning**, which will be discussed later in this chapter.

Representation learning also enables entirely new forms of scientific analysis.

Instead of using neural networks solely for prediction, researchers can examine the learned embedding space itself.

Materials with similar crystal structures often appear close together within the learned latent space.

Similarly,

- chemically related compounds,
- similar crystal prototypes,
- materials sharing electronic properties,

naturally cluster together even though the neural network was never explicitly instructed to organize them in this manner.

This phenomenon reveals that representation learning captures underlying physical relationships rather than simply memorizing training examples.

Such learned embeddings have become powerful scientific tools for

- materials visualization,
- similarity search,
- clustering,
- anomaly detection,
- generative design,
- autonomous materials discovery.

In recent years, representation learning has become the foundation of nearly every state-of-the-art machine learning architecture in materials science, including

- CGCNN,
- MEGNet,
- SchNet,
- ALIGNN,
- MACE,
- Allegro,
- Equiformer,
- MatterSim,
- foundation models for crystalline materials.

Although these models differ significantly in architecture, they all share one common objective:

> **To automatically learn meaningful representations of atomic structures directly from data instead of relying on manually engineered descriptors.**

This shift from handcrafted features to learned representations represents one of the most significant paradigm changes in computational materials science and serves as the foundation for modern deep learning approaches discussed throughout the remainder of this book.

## 23.3 What is a Representation?

To understand representation learning, we must first define what a **representation** actually is.

In machine learning, a **representation** is a mathematical description of an object that preserves the information necessary for solving a particular task.

For materials science, the object may be

- an atom,
- a molecule,
- a crystal,
- an alloy,
- a defect,
- an interface,
- or an entire material.

Since machine learning algorithms operate on numerical data, every material must ultimately be converted into a numerical representation before it can be processed by a model.

This conversion process is illustrated below.

```text
Real Material

        │

        ▼

Mathematical Representation

        │

        ▼

Machine Learning Model

        │

        ▼

Property Prediction
```

The representation acts as the communication language between the physical material and the learning algorithm.

Without an appropriate representation, even the most sophisticated machine learning model cannot extract meaningful physical relationships.

---

### 23.3.1 Characteristics of a Good Representation

An effective representation should satisfy several important properties.

First, it should preserve the essential physics of the material.

For example,

if two crystal structures differ only by a rigid translation,

their representations should remain identical because the underlying material has not changed.

Similarly,

rotating the crystal should not alter its physical properties.

Therefore, useful representations should possess appropriate invariance properties.

Second,

the representation should distinguish physically different structures.

For example,

diamond and graphite both consist entirely of carbon atoms.

Although their chemical composition is identical,

their atomic arrangements are fundamentally different.

A useful representation must therefore assign different numerical descriptions to these two structures.

Third,

the representation should be continuous.

Small changes in atomic positions should produce only small changes in the representation.

This continuity allows neural networks to learn smooth relationships between structure and properties.

Finally,

the representation should be computationally efficient.

Representations requiring excessive computational cost become impractical for large-scale datasets containing millions of structures.

---

### 23.3.2 Representation as a Feature Vector

The simplest representation is a numerical feature vector.

Suppose we wish to describe silicon using several manually selected descriptors.

```text
Silicon

↓

[

Atomic Number,

Atomic Radius,

Electronegativity,

Melting Point,

Density

]
```

Numerically,

this becomes

```text
[

14,

111,

1.90,

1687,

2.33

]
```

Each number represents one measurable property.

The complete vector becomes the machine learning input.

This type of representation was widely used in classical machine learning.

---

### 23.3.3 Representation as a Matrix

Sometimes a single vector is insufficient.

Instead,

materials may be represented using matrices.

For example,

an atomic coordinate matrix may be written as

```text
Atom

x

y

z

Si₁

0.00

0.00

0.00

Si₂

0.25

0.25

0.25

Si₃

0.50

0.50

0.50
```

Similarly,

distance matrices,

adjacency matrices,

or Coulomb matrices are commonly used representations.

Matrices naturally preserve pairwise relationships between atoms.

---

### 23.3.4 Representation as a Graph

Modern materials informatics increasingly represents crystals as graphs.

In a crystal graph,

```text
Atoms

↓

Nodes
```

and

```text
Chemical Bonds

↓

Edges
```

The resulting graph may be visualized as

```text
        Si

      /    \

    Si ---- Si

      \    /

        Si
```

Unlike fixed-length feature vectors,

graphs naturally represent materials containing different numbers of atoms.

Graph representations also preserve local atomic connectivity,

making them particularly suitable for Graph Neural Networks.

---

### 23.3.5 Representation as an Embedding

Deep learning introduces an entirely different concept.

Instead of manually defining the representation,

the neural network learns it automatically.

Suppose the input crystal is

```text
Silicon
```

Rather than assigning handcrafted descriptors,

the neural network transforms the crystal into a high-dimensional vector.

Example

```text
Silicon

↓

[

-1.37,

0.82,

2.14,

...

0.51

]
```

The numerical values themselves have no obvious physical interpretation.

Instead,

they collectively encode the information required for the prediction task.

This learned vector is called an **embedding**.

---

### 23.3.6 Representations Change During Learning

One of the defining characteristics of deep learning is that representations evolve throughout training.

Initially,

the neural network produces essentially random embeddings.

```text
Crystal

↓

Random Representation
```

As optimization proceeds,

the embeddings gradually become more meaningful.

```text
Crystal

↓

Improved Representation
```

Eventually,

similar materials begin to occupy nearby regions of the learned feature space,

while dissimilar materials become separated.

This continuous refinement of internal representations is the essence of representation learning.

---

### 23.3.7 Multiple Representations of the Same Material

A single material may have many valid representations.

For example,

silicon may be represented as

**Chemical Formula**

```text
Si
```

**Atomic Coordinates**

```text
(x,y,z)
```

**Crystal Graph**

```text
Nodes

+

Edges
```

**Distance Matrix**

```text
NxN Matrix
```

**Handcrafted Descriptor Vector**

```text
[Feature₁,

Feature₂,

...

Featureₙ]
```

**Learned Embedding**

```text
[

0.23,

-1.14,

...

2.08

]
```

Each representation emphasizes different aspects of the material.

Choosing an appropriate representation has a profound influence on prediction accuracy.

---

### 23.3.8 Why Learned Representations Are Powerful

Traditional descriptors rely entirely on human intuition.

Researchers must decide

- which properties matter,
- which descriptors should be included,
- how they should be calculated.

Deep learning removes this requirement.

Instead,

the neural network automatically discovers the most informative representation by minimizing the prediction error.

Consequently,

the learned embedding often contains subtle structural and chemical information that would be extremely difficult to engineer manually.

For example,

without being explicitly instructed,

a Graph Neural Network may learn representations encoding

- local coordination,
- hybridization,
- crystal symmetry,
- bond polarity,
- electronic environment,
- atomic packing,
- defect chemistry,

simply because these characteristics improve predictive performance.

---

### 23.3.9 Representation Learning as Feature Discovery

A useful way to interpret representation learning is as **automatic feature discovery**.

Instead of manually designing descriptors,

the optimization algorithm continuously searches for internal representations that best explain the training data.

The workflow becomes

```text
Crystal Structure

        │

        ▼

Neural Network

        │

        ▼

Automatically Learned Features

        │

        ▼

Prediction
```

Thus,

representation learning eliminates one of the largest bottlenecks in classical machine learning:

the need for handcrafted feature engineering.

This capability has become one of the primary reasons why deep learning methods consistently outperform traditional approaches across a wide range of materials informatics applications.

## 23.4 Handcrafted Features vs Learned Representations

One of the most significant paradigm shifts in machine learning has been the transition from **handcrafted feature engineering** to **automatic representation learning**. This transition has fundamentally changed how computational models are developed for materials science.

For decades, machine learning models relied almost entirely on manually designed descriptors. Researchers carefully selected measurable physical and chemical properties that they believed were relevant to a particular prediction problem. These descriptors were then supplied to classical machine learning algorithms such as Support Vector Machines, Random Forests, or Gradient Boosting models.

Modern deep learning follows a completely different philosophy.

Instead of asking

> **"Which features should we provide to the model?"**

we instead ask

> **"Can the model learn the features automatically?"**

Representation learning demonstrates that the answer is yes.

---

### 23.4.1 Handcrafted Features

Handcrafted features are descriptors explicitly designed by human experts before model training.

The workflow is

```text
Material

        │

        ▼

Feature Engineering

        │

        ▼

Feature Vector

        │

        ▼

Machine Learning Model

        │

        ▼

Prediction
```

The machine learning algorithm never observes the original crystal structure.

Instead, it only receives the engineered numerical descriptors.

For example, suppose we wish to predict the formation energy of an oxide.

A researcher might manually construct the following feature vector.

| Feature | Example |
|----------|---------|
| Atomic Number | 22 |
| Atomic Radius | 147 pm |
| Electronegativity | 1.54 |
| Oxidation State | +4 |
| Density | 4.23 g/cm³ |
| Lattice Constant | 4.59 Å |
| Coordination Number | 6 |

The resulting input becomes

```text
[

22,

147,

1.54,

4,

4.23,

4.59,

6

]
```

This vector is then supplied to the learning algorithm.

---

### 23.4.2 Advantages of Handcrafted Features

Handcrafted descriptors possess several important strengths.

First,

they are usually physically interpretable.

Every feature has a clear scientific meaning.

For example,

```text
Electronegativity

↓

Chemical Reactivity
```

or

```text
Atomic Radius

↓

Atomic Size
```

Second,

feature engineering incorporates decades of accumulated scientific knowledge.

Experienced materials scientists often know which physical quantities strongly influence particular properties.

Third,

classical machine learning algorithms generally require relatively small datasets.

Well-designed descriptors often enable accurate predictions even when only a few hundred training examples are available.

---

### 23.4.3 Limitations of Handcrafted Features

Despite their usefulness,

handcrafted descriptors suffer from several important limitations.

#### Dependence on Expert Knowledge

Designing effective descriptors requires extensive domain expertise.

Researchers must decide

- which physical quantities are important,
- which descriptors should be calculated,
- how those descriptors should be combined.

This process can be time-consuming and subjective.

---

#### Limited Expressiveness

Many complex physical interactions cannot be adequately represented using simple numerical descriptors.

For example,

electronic interactions,

many-body effects,

orbital hybridization,

and long-range crystal correlations are extremely difficult to encode manually.

---

#### Task Specificity

Descriptors that perform well for one prediction problem may perform poorly for another.

For example,

features useful for predicting

```text
Formation Energy
```

may not be sufficient for predicting

```text
Band Gap
```

or

```text
Thermal Conductivity.
```

Consequently,

new descriptors often need to be engineered for every new application.

---

#### Information Loss

Feature engineering inevitably compresses the original crystal structure.

During this process,

valuable structural information may be discarded.

Once information is lost,

the machine learning model cannot recover it.

---

### 23.4.4 Learned Representations

Deep learning removes the need for manual feature engineering.

Instead,

the neural network directly receives the raw material representation.

Examples include

- atomic coordinates,
- crystal graphs,
- neighbor lists,
- atomic species.

The network then automatically constructs internal representations.

The workflow becomes

```text
Material

        │

        ▼

Neural Network

        │

        ▼

Learned Representation

        │

        ▼

Prediction
```

The representation is optimized simultaneously with the prediction task.

No explicit feature engineering is required.

---

### 23.4.5 Hierarchical Feature Learning

One of the greatest strengths of deep neural networks is their ability to learn hierarchical representations.

Different layers learn different levels of abstraction.

```text
Input Crystal

        │

        ▼

Atomic Positions

        │

        ▼

Local Neighborhoods

        │

        ▼

Chemical Bonds

        │

        ▼

Coordination Environments

        │

        ▼

Crystal Motifs

        │

        ▼

Material Embedding

        │

        ▼

Property Prediction
```

Each successive layer extracts increasingly complex physical information.

Unlike handcrafted descriptors,

these representations emerge automatically during optimization.

---

### 23.4.6 Example: Band Gap Prediction

Suppose we wish to predict the band gap.

Using classical machine learning,

the workflow might be

```text
Crystal

↓

Density

Atomic Radius

Electronegativity

Packing Fraction

↓

Random Forest

↓

Band Gap
```

Using deep learning,

the workflow becomes

```text
Crystal Graph

↓

Graph Neural Network

↓

Learned Crystal Embedding

↓

Band Gap
```

The second approach allows the neural network to discover descriptors that may never have been considered by human researchers.

---

### 23.4.7 Comparison Between the Two Approaches

| Aspect | Handcrafted Features | Learned Representations |
|--------|----------------------|-------------------------|
| Feature Design | Manual | Automatic |
| Domain Expertise | High | Lower |
| Interpretability | High | Moderate |
| Information Loss | Possible | Minimal |
| Adaptability | Limited | High |
| Generalization | Moderate | Excellent |
| Suitable for Deep Learning | Limited | Excellent |

---

### 23.4.8 Why Learned Representations Often Perform Better

Learned representations possess several important advantages over manually engineered descriptors.

First,

they are optimized directly for the prediction objective.

Rather than relying on human intuition,

the network discovers the representations that minimize prediction error.

Second,

they capture highly nonlinear relationships between atomic environments.

Complex interactions that are impossible to encode manually may naturally emerge during training.

Third,

they continuously improve throughout optimization.

Early in training,

the representations may appear random.

As optimization proceeds,

they become increasingly organized,

eventually grouping chemically and structurally similar materials together.

This adaptive behavior allows deep learning models to outperform traditional descriptor-based methods on many challenging materials science problems.

---

### 23.4.9 When Are Handcrafted Features Still Useful?

Despite the success of deep learning,

handcrafted descriptors remain valuable in several situations.

They are often preferred when

- datasets are very small,
- computational resources are limited,
- interpretability is essential,
- rapid prototyping is required,
- classical machine learning algorithms are sufficient.

In practice,

many modern materials informatics workflows combine both approaches,

using handcrafted descriptors alongside learned embeddings to exploit the advantages of each.

---

### 23.4.10 Transition Toward Foundation Models

The evolution from handcrafted descriptors to learned representations has ultimately led to the development of foundation models for materials science.

Instead of constructing task-specific descriptors,

large neural networks now learn universal representations that can be reused across numerous downstream applications.

This progression can be summarized as

```text
Handcrafted Features

        ↓

Machine Learning

        ↓

Deep Learning

        ↓

Representation Learning

        ↓

Foundation Models
```

This evolution represents one of the most important developments in modern computational materials science and forms the basis for the advanced methods discussed throughout the remainder of this chapter.

## 23.5 Crystal Representations

The representation of a material determines how effectively a machine learning model can learn the relationship between atomic structure and material properties. Unlike images or text, crystalline materials possess unique characteristics such as periodicity, symmetry, variable numbers of atoms, and complex chemical environments. Consequently, designing suitable crystal representations has been one of the central challenges in materials informatics.

Over the past two decades, researchers have developed a wide variety of crystal representations. These range from simple manually engineered descriptor vectors to sophisticated graph-based and neural network embeddings learned automatically from data.

The evolution of crystal representations closely mirrors the evolution of machine learning itself.

```text
Chemical Formula

        ↓

Handcrafted Descriptors

        ↓

Crystal Fingerprints

        ↓

Graph Representations

        ↓

Learned Crystal Embeddings

        ↓

Foundation Representations
```

Each successive representation captures increasingly rich structural and chemical information.

---

### 23.5.1 Requirements of an Ideal Crystal Representation

An ideal crystal representation should satisfy several important properties.

It should

- uniquely identify different crystal structures,
- preserve atomic connectivity,
- represent periodic boundary conditions,
- distinguish different chemical species,
- remain invariant under translation,
- remain invariant under rotation whenever appropriate,
- scale efficiently to large systems,
- support gradient-based optimization.

These properties ensure that the representation faithfully describes the underlying material while remaining suitable for machine learning algorithms.

---

### 23.5.2 Chemical Formula Representation

The simplest possible representation is the chemical formula.

Examples include

```text
Si

Fe₂O₃

Al₂O₃

LiFePO₄

MoS₂
```

Although useful for identifying a material,

the chemical formula contains very little structural information.

For example,

diamond and graphite are both represented as

```text
C
```

despite possessing completely different crystal structures and physical properties.

Consequently,

chemical formulas alone are insufficient for most prediction tasks.

---

### 23.5.3 Composition-Based Feature Vectors

A more informative representation summarizes elemental properties.

For example,

a compound may be represented using

- average atomic number,
- average atomic mass,
- average electronegativity,
- average covalent radius,
- valence electron count,
- atomic fraction.

Example

```text
Material

↓

[

Average Atomic Number,

Average Atomic Radius,

Average Electronegativity,

Average Valence Electrons,

Density

]
```

These descriptors ignore atomic arrangement but often provide reasonable performance for composition-based property prediction.

---

### 23.5.4 Crystal Structure Descriptors

To incorporate structural information,

additional descriptors may be computed.

Typical quantities include

- lattice constants,
- lattice angles,
- unit-cell volume,
- density,
- space group,
- crystal system,
- coordination number,
- bond lengths,
- bond angles.

Example

```text
Crystal

↓

[

a,

b,

c,

α,

β,

γ,

Volume,

Density,

Coordination Number

]
```

Such descriptors provide significantly richer structural information than chemical composition alone.

---

### 23.5.5 Local Atomic Environment

Many material properties depend primarily on the immediate neighborhood surrounding each atom.

The local atomic environment typically includes

- neighboring atoms,
- bond distances,
- bond angles,
- coordination geometry,
- local chemical composition.

For example,

consider one silicon atom.

```text
          Si

       /  |  \

    Si  Si  Si

       \  |  /

          Si
```

Rather than describing the entire crystal,

the representation focuses on this local neighborhood.

Most modern machine learning potentials rely heavily on local atomic environments.

---

### 23.5.6 Neighbor List Representation

A convenient way to represent local environments is through neighbor lists.

Example

| Central Atom | Neighbor | Distance (Å) |
|--------------|----------|-------------:|
| Si₁ | Si₂ | 2.35 |
| Si₁ | Si₃ | 2.35 |
| Si₁ | Si₄ | 2.36 |
| Si₁ | Si₅ | 2.35 |

The neighbor list preserves local connectivity while remaining computationally efficient.

Nearly all modern graph neural networks begin by constructing such neighbor lists.

---

### 23.5.7 Distance Matrix Representation

Another representation stores pairwise atomic distances.

Example

```text
      A₁   A₂   A₃

A₁   0   2.35 3.84

A₂ 2.35   0   2.35

A₃ 3.84 2.35   0
```

The distance matrix preserves geometric relationships between atoms.

However,

it becomes increasingly expensive for very large systems because its size grows quadratically with the number of atoms.

---

### 23.5.8 Graph Representation

Graph representations have become the dominant representation for crystalline materials.

In a crystal graph,

```text
Atoms

↓

Nodes
```

and

```text
Chemical Bonds

↓

Edges
```

A simple graph appears as

```text
        Si

      /    \

    Si ---- Si

      \    /

        Si
```

Each node stores atomic information,

while each edge stores interaction information.

This representation naturally supports crystals containing different numbers of atoms.

---

### 23.5.9 Node Features

Each node contains information describing an individual atom.

Typical node features include

- atomic number,
- atomic mass,
- electronegativity,
- covalent radius,
- valence electrons,
- oxidation state,
- atomic embedding.

Example

```text
Node

↓

[

Atomic Number,

Atomic Radius,

Electronegativity,

Valence Electrons

]
```

These features become the initial inputs to a Graph Neural Network.

---

### 23.5.10 Edge Features

Edges describe interactions between neighboring atoms.

Common edge features include

- bond distance,
- bond direction,
- bond angle,
- periodic image information,
- radial basis expansion,
- spherical harmonic encoding.

Example

```text
Edge

↓

[

Distance,

Direction,

Periodic Offset

]
```

During message passing,

both node and edge features evolve together.

---

### 23.5.11 Periodic Crystal Representation

Unlike molecules,

crystals extend infinitely in space.

Therefore,

representations must account for periodic boundary conditions.

Instead of representing only the atoms inside one unit cell,

neighbor relationships include periodic images.

Conceptually,

```text
Unit Cell

↓

Periodic Replication

↓

Infinite Crystal
```

This ensures that atoms near the unit-cell boundary interact correctly with neighboring images.

Most crystal graph neural networks automatically incorporate periodic boundary conditions during graph construction.

---

### 23.5.12 Learned Crystal Representations

Rather than manually defining node and edge features,

modern deep learning models transform them into high-dimensional embeddings.

For example,

an atom initially described by

```text
[

Atomic Number,

Radius,

Electronegativity

]
```

may become

```text
[

-1.24,

0.58,

2.31,

...

0.91

]
```

after several graph neural network layers.

These learned representations encode substantially more information than the original handcrafted descriptors.

---

### 23.5.13 Comparing Crystal Representations

Different representations capture different aspects of crystalline materials.

| Representation | Structural Information | Suitable for Deep Learning |
|---------------|-----------------------|----------------------------|
| Chemical Formula | Very Low | No |
| Composition Vector | Low | Limited |
| Crystal Descriptors | Moderate | Moderate |
| Distance Matrix | High | Moderate |
| Neighbor List | High | Yes |
| Crystal Graph | Very High | Excellent |
| Learned Embedding | Extremely High | Excellent |

The progression clearly shows the increasing expressive power of modern representations.

---

### 23.5.14 Why Graph Representations Dominate Modern Materials AI

Graph representations possess several major advantages.

They naturally

- represent arbitrary crystal sizes,
- preserve atomic connectivity,
- encode local chemistry,
- incorporate periodic boundary conditions,
- support message passing,
- learn hierarchical atomic interactions.

Consequently,

nearly every state-of-the-art materials graph neural network—including CGCNN, MEGNet, ALIGNN, MACE, Allegro, and Equiformer—uses graph representations as the foundation for representation learning.

These graph representations provide the starting point from which increasingly informative crystal embeddings emerge during neural network training.

## 23.6 Embedding Spaces

Modern deep learning models do not directly make predictions from raw crystal structures. Instead, they first transform every material into a numerical representation known as an **embedding**. These embeddings exist in a high-dimensional mathematical space called the **embedding space** or **feature space**.

The embedding space is one of the most important concepts in representation learning because it is where the neural network organizes its understanding of materials. During training, the network gradually arranges similar materials close together while separating materials with different structural or chemical characteristics.

Unlike handcrafted descriptors, embedding spaces are not explicitly designed by researchers. Instead, they emerge automatically through optimization.

---

### 23.6.1 What is an Embedding?

An embedding is a dense numerical vector learned by a neural network.

Instead of representing silicon using manually engineered descriptors,

```text
Silicon

↓

[

Atomic Number,

Atomic Radius,

Electronegativity,

Density

]
```

a deep learning model may represent it as

```text
Silicon

↓

[

-0.83,

1.42,

0.57,

...

-2.16

]
```

Each number individually has no obvious physical meaning.

Instead,

the entire vector collectively captures the structural, chemical, and physical characteristics that are useful for prediction.

This vector is called the **embedding**.

---

### 23.6.2 High-Dimensional Feature Space

Every embedding corresponds to one point in a high-dimensional mathematical space.

For example,

if the embedding contains

```text
128
```

numbers,

then every material occupies one point in a

```text
128-dimensional space.
```

Conceptually,

```text
Material A

↓

●
```

```text
Material B

↓

●
```

```text
Material C

↓

●
```

Each point represents one crystal structure.

The neural network learns where these points should be located.

---

### 23.6.3 Why High Dimensions?

One may ask why embeddings require hundreds of dimensions instead of only two or three.

The reason is that crystalline materials possess extremely complex relationships involving

- chemistry,
- bonding,
- crystal symmetry,
- coordination,
- electronic structure,
- periodicity,
- local environments.

A low-dimensional space cannot adequately represent all of these factors simultaneously.

Higher-dimensional spaces provide the flexibility necessary for separating materials with subtle structural differences.

For example,

an embedding dimension of

```text
128
```

or

```text
256
```

is common in modern Graph Neural Networks.

---

### 23.6.4 Learning the Embedding Space

Initially,

the neural network assigns essentially random embeddings.

```text
Before Training

Material A

●

Material B

●

Material C

●
```

No meaningful organization exists.

During optimization,

gradient descent continuously adjusts the neural network parameters.

As training progresses,

materials with similar properties gradually move closer together.

```text
After Training

Metals

● ● ●

Semiconductors

● ● ●

Insulators

● ● ●
```

The embedding space becomes increasingly organized.

---

### 23.6.5 Similar Materials Occupy Nearby Regions

One of the remarkable properties of learned embedding spaces is that chemically similar materials naturally cluster together.

For example,

```text
Diamond

↓

●
```

```text
Silicon

↓

●
```

```text
Germanium

↓

●
```

may appear close together because they share

- tetrahedral coordination,
- covalent bonding,
- similar crystal symmetry.

Likewise,

alkali halides,

perovskites,

or transition-metal oxides often form their own clusters.

Importantly,

the neural network discovers these relationships automatically.

No explicit clustering information is provided during training.

---

### 23.6.6 Distance Represents Similarity

The distance between two embeddings reflects their similarity.

Small distance

```text
Material A ●────● Material B

↓

Highly Similar
```

Large distance

```text
Material A ●──────────────● Material B

↓

Very Different
```

Thus,

distance within the embedding space becomes a quantitative measure of chemical similarity.

---

### 23.6.7 Measuring Similarity Between Embeddings

Several distance metrics are commonly used.

#### Euclidean Distance

The straight-line distance between two embedding vectors.

Small Euclidean distance generally indicates similar materials.

---

#### Cosine Similarity

Rather than measuring absolute distance,

cosine similarity measures the angle between two vectors.

Values close to

```text
1
```

indicate highly similar embeddings.

Values near

```text
0
```

indicate unrelated materials.

Negative values indicate opposite directions.

Cosine similarity is widely used for representation learning because it is less sensitive to vector magnitude.

---

#### Dot Product

Some neural networks compare embeddings using the dot product.

Larger values indicate stronger similarity.

This measure is frequently used in contrastive learning.

---

### 23.6.8 Evolution of the Embedding Space During Training

The embedding space continuously evolves.

Initially,

```text
Random

↓

Disorganized
```

After several training epochs,

```text
Partial Organization
```

Finally,

```text
Well-Separated Material Families
```

The gradual emergence of organized clusters indicates that the network has learned meaningful structural relationships.

---

### 23.6.9 Embedding Space as a Scientific Map

The embedding space can be viewed as a map of materials.

Instead of geographic locations,

the coordinates represent learned structural relationships.

For example,

neighboring points may correspond to

- similar crystal symmetry,
- similar electronic properties,
- similar bonding environments,
- similar thermodynamic stability.

Researchers can therefore explore the embedding space to discover relationships that may not be immediately apparent from conventional descriptors.

---

### 23.6.10 Embedding Arithmetic

Although less common than in natural language processing,

embedding vectors sometimes exhibit meaningful arithmetic relationships.

Conceptually,

```text
Embedding

↓

Chemical Information

+

Structural Information

+

Electronic Information
```

The embedding combines multiple physical characteristics into a single numerical representation.

Different regions of the embedding space may therefore correspond to different classes of materials.

---

### 23.6.11 Visualizing High-Dimensional Embeddings

Humans cannot directly visualize

```text
128-dimensional
```

spaces.

Therefore,

dimensionality reduction techniques are used to project embeddings into

- two dimensions,
- three dimensions.

Common visualization methods include

- Principal Component Analysis (PCA),
- t-distributed Stochastic Neighbor Embedding (t-SNE),
- Uniform Manifold Approximation and Projection (UMAP).

These methods will be discussed later in this chapter.

---

### 23.6.12 Why Embedding Spaces Matter

Embedding spaces are far more than intermediate neural network outputs.

They enable

- property prediction,
- similarity search,
- clustering,
- anomaly detection,
- transfer learning,
- generative modeling,
- active learning,
- foundation models.

Many modern materials informatics workflows rely more heavily on the learned embedding itself than on the final prediction layer.

Consequently,

understanding embedding spaces has become one of the central objectives of representation learning and forms the basis for nearly every advanced deep learning method used in contemporary computational materials science.

## 23.7 Latent Space

One of the most powerful ideas in modern deep learning is the concept of the **latent space**. While an embedding is a numerical representation of a material, the latent space is the mathematical space in which all these learned representations exist.

Every material processed by a neural network is ultimately mapped to one point inside this latent space. During training, the neural network continuously reorganizes this space so that materials with similar structural, chemical, or physical characteristics become close together, while dissimilar materials move farther apart.

The latent space therefore serves as the neural network's internal understanding of the materials world.

---

### 23.7.1 What is a Latent Space?

The word **latent** means *hidden*.

A latent space is therefore a hidden feature space learned automatically by the neural network.

The overall workflow can be viewed as

```text
Crystal Structure

        │

        ▼

Encoder Network

        │

        ▼

Latent Representation

        │

        ▼

Prediction Head

        │

        ▼

Material Property
```

Unlike handcrafted descriptors,

the latent variables are not directly observable.

Instead,

they emerge automatically during optimization.

---

### 23.7.2 Why Learn a Latent Space?

Suppose we represent every crystal directly by its atomic coordinates.

```text
Crystal

↓

Thousands of Coordinates
```

This representation is

- extremely high-dimensional,
- highly redundant,
- difficult for machine learning algorithms to process.

Instead,

the encoder compresses the crystal into a compact vector.

```text
Crystal

↓

Encoder

↓

128-D Latent Vector
```

This latent vector contains only the information necessary for the prediction task.

Thus,

the latent space acts as a compressed summary of the material.

---

### 23.7.3 Compression Through an Encoder

Most deep learning architectures contain an **encoder**.

Its purpose is to transform complex input data into a compact representation.

Conceptually,

```text
Large Crystal Graph

        │

        ▼

Graph Neural Network

        │

        ▼

256 Features

        │

        ▼

128 Features

        │

        ▼

64 Features

        │

        ▼

Latent Representation
```

As information flows through the encoder,

irrelevant details are discarded,

while useful physical information is preserved.

---

### 23.7.4 Bottleneck Learning

Many neural networks intentionally contain a narrow hidden layer called the **bottleneck**.

```text
Input

↓

512

↓

256

↓

128

↓

32

↓

128

↓

256

↓

Output
```

The narrow layer forces the network to compress information.

Because only a limited number of variables can pass through the bottleneck,

the network learns highly informative latent features.

This principle forms the basis of

- Autoencoders,
- Variational Autoencoders,
- Representation Learning,
- Foundation Models.

---

### 23.7.5 Latent Variables

Suppose the bottleneck contains

```text
64
```

neurons.

The resulting latent vector becomes

```text
z = [

0.41,

-1.83,

0.92,

...

1.12

]
```

Each number is called a **latent variable**.

Unlike handcrafted descriptors,

these variables usually have no direct physical interpretation.

Instead,

they collectively encode

- crystal symmetry,
- bonding environments,
- local coordination,
- chemistry,
- electronic information,

in a compressed mathematical form.

---

### 23.7.6 Latent Space Organization

Initially,

the latent space is random.

```text
Random Initialization

●

●

●

●

●
```

After training,

materials naturally organize themselves.

```text
Metals

● ● ●

Semiconductors

● ● ●

Ceramics

● ● ●

Perovskites

● ●
```

The neural network was never explicitly instructed to create these clusters.

Instead,

they emerge because similar materials require similar internal representations.

---

### 23.7.7 Example: Learning Band Gap Representations

Suppose the network is trained to predict band gaps.

Although the training labels contain only

```text
Band Gap
```

the latent space often learns much richer information.

Materials with similar

- bonding,
- coordination,
- crystal symmetry,
- electronic structure,

naturally become neighbors.

Thus,

the latent space captures much more than the target property itself.

---

## 23.7.8 Code Example — Extracting Latent Embeddings from a Neural Network

The following PyTorch example demonstrates how intermediate latent representations can be extracted from a trained neural network.

```python
import torch
import torch.nn as nn

class MaterialEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(100,256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,64)

        )

        self.predictor = nn.Linear(64,1)

    def forward(self,x):

        z = self.encoder(x)

        y = self.predictor(z)

        return y

    def get_embedding(self,x):

        return self.encoder(x)
```

The encoder produces a **64-dimensional latent representation**, while the predictor uses this representation to estimate the material property.

---

### 23.7.9 Generating Latent Embeddings

Suppose we have feature vectors describing several materials.

```python
model = MaterialEncoder()

x = torch.randn(10,100)

embeddings = model.get_embedding(x)

print(embeddings.shape)
```

Output

```text
torch.Size([10, 64])
```

Interpretation

- 10 materials were processed.
- Each material is represented by a 64-dimensional embedding.
- These embeddings are points inside the learned latent space.

---

### 23.7.10 Computing Pairwise Similarity

Once embeddings have been generated,

their similarity can be quantified.

```python
import torch
import torch.nn.functional as F

similarity = F.cosine_similarity(

    embeddings[0].unsqueeze(0),

    embeddings[1].unsqueeze(0)

)

print(similarity)
```

Possible output

```text
tensor([0.94])
```

A cosine similarity close to

```text
1
```

indicates that the two materials occupy nearby regions of the latent space and therefore possess similar learned representations.

---

### 23.7.11 Measuring Distances in Latent Space

Euclidean distance provides another measure of similarity.

```python
distance = torch.norm(

    embeddings[0] -

    embeddings[1]

)

print(distance)
```

Smaller values indicate greater similarity.

This principle forms the basis of

- similarity search,
- clustering,
- nearest-neighbor retrieval,
- recommendation systems.

---

### 23.7.12 Why Latent Spaces Matter in Materials Informatics

Latent spaces provide much more than compact feature vectors.

They enable researchers to

- visualize millions of materials,
- discover hidden material families,
- identify outliers,
- perform transfer learning,
- retrieve chemically similar compounds,
- build foundation models,
- generate entirely new materials.

Indeed,

most modern advances in materials AI—including contrastive learning, self-supervised learning, generative models, diffusion models, and active learning—operate primarily within the latent space rather than on raw crystal structures.

Understanding the latent space is therefore essential for understanding modern representation learning and serves as the foundation for the remaining topics in this chapter.

## 23.8 Representation Learning in Graph Neural Networks

Graph Neural Networks (GNNs) have become the dominant deep learning architecture for crystalline materials because they naturally represent atomic structures as graphs. Unlike traditional neural networks, which operate on fixed-size vectors, GNNs directly learn representations from atoms and their interactions.

The fundamental objective of a Graph Neural Network is not merely to predict a material property. Instead, its primary goal is to **learn increasingly informative representations of atoms, bonds, and entire crystal structures**.

Every forward pass through a GNN gradually transforms simple atomic information into rich chemical and structural embeddings that are useful for downstream prediction tasks.

---

### 23.8.1 Representation Learning as Message Passing

Graph Neural Networks learn representations through a process called **message passing**.

Initially,

each atom contains only basic information.

```text
Atom

↓

Atomic Number

Atomic Mass

Electronegativity

Atomic Radius
```

This information alone is insufficient for understanding the complete crystal.

During message passing,

neighboring atoms continuously exchange information.

```text
Atom A

↔

Atom B

↔

Atom C

↔

Atom D
```

Each interaction allows the atom to update its internal representation.

After multiple message-passing layers,

every atom gradually develops an understanding of its surrounding chemical environment.

---

### 23.8.2 Initial Node Features

Each node in the crystal graph begins with an initial feature vector.

For example,

```text
Carbon

↓

[

Atomic Number,

Atomic Mass,

Electronegativity,

Valence Electrons

]
```

Numerically,

```text
[

6,

12.01,

2.55,

4

]
```

These handcrafted values serve only as the starting point.

During training,

they are transformed into learned embeddings.

---

### 23.8.3 Initial Edge Features

Edges describe relationships between neighboring atoms.

Typical edge information includes

- bond distance,
- bond direction,
- periodic image,
- radial basis encoding.

Example

```text
Edge

↓

Distance = 2.35 Å

Direction = (0.57,0.57,0.57)
```

These edge features determine how information flows between neighboring atoms.

---

### 23.8.4 Node Embedding Update

After receiving messages from neighboring atoms,

the node embedding is updated.

Conceptually,

```text
Old Node Representation

        +

Neighbor Information

↓

Updated Node Representation
```

Every layer produces a more informative embedding than the previous one.

For example,

Layer 0

```text
Si

↓

[

Atomic Number,

Radius,

Electronegativity

]
```

Layer 1

```text
↓

Local Coordination
```

Layer 2

```text
↓

Bonding Environment
```

Layer 3

```text
↓

Crystal Motif
```

Layer 4

```text
↓

Learned Atomic Embedding
```

Thus,

the representation becomes increasingly abstract.

---

### 23.8.5 Message Passing Process

A Graph Neural Network repeatedly performs three operations.

```text
Receive Messages

↓

Aggregate Messages

↓

Update Representation
```

This procedure is repeated for every node.

After one iteration,

an atom knows about its immediate neighbors.

After two iterations,

it receives indirect information from neighbors of neighbors.

After several layers,

the embedding captures increasingly larger portions of the crystal.

---

### 23.8.6 Graph Embeddings

Eventually,

the atomic embeddings must be combined into a single representation describing the entire crystal.

```text
Atom Embeddings

↓

Pooling

↓

Crystal Embedding
```

This graph embedding becomes the input to the final prediction network.

The resulting representation summarizes

- crystal geometry,
- chemistry,
- bonding,
- coordination,
- local environments,

within one fixed-length vector.

---

### 23.8.7 Readout Functions

The process of converting atomic embeddings into one crystal embedding is called the **readout function**.

Several readout strategies are commonly used.

#### Sum Pooling

```text
Crystal Embedding

=

Atom₁

+

Atom₂

+

...

+

Atomₙ
```

---

#### Mean Pooling

```text
Crystal Embedding

=

Average

(

Atom Embeddings

)
```

---

#### Max Pooling

Each feature takes the maximum value among all atoms.

---

#### Attention Pooling

More important atoms contribute more strongly to the final embedding.

This approach is widely used in modern graph transformers.

---

### 23.8.8 Hierarchical Representation Learning

Graph Neural Networks learn representations hierarchically.

```text
Raw Atoms

↓

Atomic Features

↓

Local Neighborhood

↓

Chemical Environment

↓

Crystal Motifs

↓

Crystal Representation

↓

Material Property
```

Each layer captures increasingly complex structural information.

This hierarchy resembles feature extraction in convolutional neural networks for images.

---

## 23.8.9 Code Example — Learning Node Embeddings with PyTorch Geometric

The following example demonstrates a simple Graph Neural Network that learns node embeddings.

```python
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class CrystalGNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(16,64)

        self.conv2 = GCNConv(64,128)

        self.conv3 = GCNConv(128,64)

    def forward(self,x,edge_index):

        x = self.conv1(x,edge_index)

        x = F.relu(x)

        x = self.conv2(x,edge_index)

        x = F.relu(x)

        x = self.conv3(x,edge_index)

        return x
```

Notice that the network returns

```python
x
```

rather than a prediction.

These are the learned **node embeddings**.

---

### 23.8.10 Running the Network

```python
from torch_geometric.data import Data

x = torch.randn(12,16)

edge_index = torch.tensor(

    [

        [0,1,1,2,2,3,3,4],

        [1,0,2,1,3,2,4,3]

    ],

    dtype=torch.long

)

data = Data(

    x=x,

    edge_index=edge_index

)

model = CrystalGNN()

node_embeddings = model(

    data.x,

    data.edge_index

)

print(node_embeddings.shape)
```

Output

```text
torch.Size([12, 64])
```

Interpretation

- The crystal contains **12 atoms**.
- Each atom is now represented by a **64-dimensional learned embedding**.

---

### 23.8.11 Constructing a Graph Embedding

A graph-level representation can be obtained through global pooling.

```python
from torch_geometric.nn import global_mean_pool

batch = torch.zeros(

    data.num_nodes,

    dtype=torch.long

)

graph_embedding = global_mean_pool(

    node_embeddings,

    batch

)

print(graph_embedding.shape)
```

Output

```text
torch.Size([1, 64])
```

Instead of twelve atomic embeddings,

the entire crystal is now represented by **one 64-dimensional vector**.

This graph embedding can subsequently be used for

- formation energy prediction,
- band gap prediction,
- elastic property prediction,
- stability prediction,
- similarity search,
- transfer learning.

---

### 23.8.12 Why Representation Learning in GNNs is Powerful

Unlike classical descriptor-based methods,

Graph Neural Networks do not require manually engineered crystal fingerprints.

Instead,

they automatically learn

- atomic representations,
- bond representations,
- neighborhood representations,
- crystal representations,

directly from data through message passing.

This automatic hierarchical representation learning is one of the primary reasons why Graph Neural Networks consistently outperform traditional machine learning approaches across a wide range of materials informatics applications.

## 23.9 Self-Supervised Learning for Materials

One of the greatest challenges in materials informatics is the limited availability of labeled data. While millions of crystal structures can be generated or collected from databases such as the Materials Project, OQMD, AFLOW, and NOMAD, only a relatively small fraction possess experimentally measured or Density Functional Theory (DFT) computed properties.

Deep learning models trained using supervised learning depend heavily on these labels.

For example,

```text
Crystal

↓

Formation Energy
```

or

```text
Crystal

↓

Band Gap
```

Obtaining such labels often requires expensive first-principles calculations or laboratory experiments.

Self-supervised learning provides a powerful solution to this problem.

Instead of learning from externally provided labels, the model constructs its own learning objectives directly from the input data. In other words, the data itself generates the supervision signal.

Consequently, enormous collections of unlabeled crystal structures can be used to pretrain neural networks before fine-tuning them on much smaller labeled datasets.

This idea has revolutionized natural language processing through models such as BERT and GPT, computer vision through masked image modeling, and is now becoming increasingly important in materials informatics.

---

### 23.9.1 Supervised vs Self-Supervised Learning

The difference between supervised and self-supervised learning can be illustrated as follows.

**Supervised Learning**

```text
Crystal Structure

        │

        ▼

Band Gap Label

        │

        ▼

Neural Network

        │

        ▼

Prediction
```

The model directly learns from labeled properties.

---

**Self-Supervised Learning**

```text
Crystal Structure

        │

        ▼

Automatically Generated Task

        │

        ▼

Neural Network

        │

        ▼

Learn Representation
```

No external labels are required.

Instead,

the learning task is created automatically from the crystal itself.

---

### 23.9.2 Why Self-Supervised Learning?

Modern materials databases contain

- millions of crystal structures,
- millions of atomic configurations,
- enormous molecular datasets,

but only a relatively small percentage contain high-quality DFT labels.

For example,

```text
Available Crystal Structures

↓

10,000,000
```

```text
Labeled Formation Energies

↓

250,000
```

Discarding the remaining structures wastes an enormous amount of useful structural information.

Self-supervised learning enables the neural network to exploit all available crystal structures, regardless of whether property labels exist.

---

### 23.9.3 Pretext Tasks

A self-supervised model learns by solving an artificial prediction problem called a **pretext task**.

The pretext task should force the neural network to understand the crystal structure.

Typical pretext tasks include

- masked atom prediction,
- masked bond prediction,
- neighbor prediction,
- crystal reconstruction,
- contrastive representation learning.

Although these tasks are artificial,

they encourage the network to learn chemically meaningful representations.

---

### 23.9.4 Masked Atom Prediction

One of the simplest self-supervised learning strategies is **masked atom prediction**.

Suppose the crystal contains

```text
Si

Si

Si

Si
```

Randomly,

one atom is hidden.

```text
Si

[MASK]

Si

Si
```

The neural network must determine the missing atomic species using the surrounding environment.

This task requires the model to understand

- local chemistry,
- crystal symmetry,
- neighboring atoms,
- bonding environments.

Rather than memorizing atom types,

the model develops chemically meaningful crystal embeddings.

---

### 23.9.5 Masked Bond Prediction

Instead of hiding atoms,

the model may hide bond information.

Example

```text
Si

────

?

────

Si
```

The neural network predicts

- bond distance,
- bond existence,
- bond type,

using neighboring structural information.

This forces the model to learn geometric relationships within crystals.

---

### 23.9.6 Crystal Reconstruction

Another common objective asks the encoder to compress the crystal into a latent representation and then reconstruct the original crystal.

```text
Crystal

↓

Encoder

↓

Latent Space

↓

Decoder

↓

Reconstructed Crystal
```

The reconstruction error becomes the learning objective.

Successful reconstruction indicates that the latent representation preserves essential structural information.

---

### 23.9.7 Context Prediction

Instead of predicting missing atoms,

the model predicts whether two atomic environments belong to the same crystal.

For example,

```text
Environment A

↓

Same Crystal ?

↓

Yes / No

↓

Environment B
```

This encourages embeddings from identical crystals to become similar while embeddings from unrelated materials become different.

---

## 23.9.8 Code Example — Randomly Masking Atomic Features

The following example demonstrates one of the simplest self-supervised learning tasks.

Random node features are masked before being supplied to a Graph Neural Network.

```python
import torch

def mask_node_features(

    x,

    mask_probability=0.15

):

    x_masked = x.clone()

    num_nodes = x.shape[0]

    mask = torch.rand(num_nodes) < mask_probability

    x_masked[mask] = 0

    return x_masked, mask
```

Suppose

```python
node_features = torch.randn(20,16)

masked_features, mask = mask_node_features(

    node_features,

    mask_probability=0.20

)

print(mask.sum())
```

Possible output

```text
tensor(4)
```

Interpretation

Four atomic feature vectors have been masked.

The neural network must reconstruct these missing features.

---

### 23.9.9 Predicting Masked Nodes

A simple decoder can be attached to predict the masked node features.

```python
import torch.nn as nn

class MaskedNodeDecoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(

            64,

            16

        )

    def forward(self,x):

        return self.linear(x)
```

The decoder receives learned node embeddings and attempts to reconstruct the original atomic features.

---

### 23.9.10 Computing the Reconstruction Loss

Only masked nodes contribute to the loss.

```python
criterion = nn.MSELoss()

loss = criterion(

    predicted_features[mask],

    original_features[mask]

)

print(loss)
```

By minimizing this reconstruction error,

the encoder gradually learns informative crystal representations without requiring any external property labels.

---

### 23.9.11 Advantages of Self-Supervised Learning

Self-supervised learning offers several important advantages for materials informatics.

It

- uses unlabeled crystal databases,
- reduces dependence on expensive DFT calculations,
- improves generalization,
- produces transferable crystal embeddings,
- enables foundation model training,
- significantly improves downstream performance after fine-tuning.

These advantages make self-supervised learning one of the fastest-growing research areas in modern materials artificial intelligence.

In the next section, we will examine **contrastive learning**, one of the most successful self-supervised learning strategies for learning highly discriminative crystal representations.

## 23.10 Contrastive Learning

One of the most influential developments in modern representation learning is **contrastive learning**. Rather than learning directly from property labels, contrastive learning trains a neural network to distinguish between **similar** and **dissimilar** examples. The objective is simple:

- Similar materials should have **similar embeddings**.
- Different materials should have **different embeddings**.

Although conceptually straightforward, this learning paradigm has dramatically improved representation quality in computer vision, natural language processing, graph learning, and, more recently, materials informatics.

For crystal materials, contrastive learning enables Graph Neural Networks to learn chemically meaningful embeddings without requiring formation energies, band gaps, elastic constants, or any other property labels.

---

### 23.10.1 Basic Idea

Suppose we have a crystal structure.

```text
Silicon Crystal
```

Instead of using the crystal only once, we generate two different views of the same structure.

```text
Original Crystal

        │

        ├──────────────┐

        ▼              ▼

View 1             View 2
```

These two views should represent the same material.

The neural network processes both views independently.

```text
View 1

↓

Encoder

↓

Embedding z₁
```

```text
View 2

↓

Encoder

↓

Embedding z₂
```

Since both originate from the same crystal,

their embeddings should become nearly identical.

---

### 23.10.2 Positive and Negative Pairs

Contrastive learning relies on two kinds of sample pairs.

#### Positive Pair

Two different views generated from the **same** material.

Example

```text
Si Crystal

↓

View A

↓

Embedding A
```

```text
Si Crystal

↓

View B

↓

Embedding B
```

The learning objective is

```text
Embedding A

≈

Embedding B
```

---

#### Negative Pair

Two embeddings originating from different materials.

Example

```text
Silicon

↓

Embedding A
```

```text
Aluminum Oxide

↓

Embedding B
```

The objective becomes

```text
Embedding A

≠

Embedding B
```

Thus,

positive pairs are pulled together,

while negative pairs are pushed apart.

---

### 23.10.3 Geometry of Contrastive Learning

Initially,

embeddings are randomly distributed.

```text
●

●

●

●

●
```

After contrastive training,

positive samples move closer together.

```text
Si

●●

Fe

●●

Al₂O₃

●●
```

Different material families become increasingly separated.

The embedding space therefore acquires meaningful chemical structure.

---

### 23.10.4 Data Augmentation for Crystal Graphs

Unlike images,

crystals cannot simply be rotated or cropped arbitrarily.

Instead,

graph-specific augmentations are used.

Common augmentations include

- random edge removal,
- node masking,
- feature masking,
- subgraph sampling,
- edge perturbation,
- coordinate perturbation,
- neighbor dropout.

These augmentations create different views of the same crystal while preserving its overall identity.

---

### 23.10.5 Graph Contrastive Learning

Graph contrastive learning extends these ideas to crystal graphs.

```text
Crystal Graph

        │

        ├───────────────┐

        ▼               ▼

Augmented Graph 1   Augmented Graph 2

        │               │

        ▼               ▼

Graph Encoder    Graph Encoder

        │               │

        ▼               ▼

Embedding z₁     Embedding z₂
```

The encoder learns graph representations that remain consistent despite graph perturbations.

---

### 23.10.6 InfoNCE Loss

The most widely used contrastive objective is the **InfoNCE Loss**.

Its purpose is to

- maximize similarity between positive pairs,
- minimize similarity between negative pairs.

Intuitively,

the optimization attempts to produce

```text
Positive Similarity

↓

Large
```

and

```text
Negative Similarity

↓

Small
```

The resulting embedding space becomes highly discriminative.

Modern methods such as

- SimCLR,
- GraphCL,
- MoCo,
- BYOL,
- BGRL,

all rely on this fundamental principle.

---

## 23.10.7 Code Example — Computing Cosine Similarity

The first step in contrastive learning is measuring similarity between embeddings.

```python
import torch
import torch.nn.functional as F

z1 = torch.randn(64)

z2 = torch.randn(64)

similarity = F.cosine_similarity(

    z1.unsqueeze(0),

    z2.unsqueeze(0)

)

print(similarity)
```

Possible output

```text
tensor([0.83])
```

Values closer to

```text
1
```

indicate highly similar representations.

---

### 23.10.8 Creating Positive Pairs

Suppose

```python
graph = data
```

We generate two augmented versions.

```python
graph_view1 = random_edge_dropout(graph)

graph_view2 = random_node_mask(graph)
```

Although structurally different,

both graphs still represent the same crystal.

Therefore,

their embeddings should become similar.

---

### 23.10.9 Graph Encoder

A simple graph encoder may be written as

```python
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

import torch.nn.functional as F
import torch.nn as nn

class GraphEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(16,64)

        self.conv2 = GCNConv(64,128)

    def forward(

        self,

        x,

        edge_index,

        batch

    ):

        x = self.conv1(x,edge_index)

        x = F.relu(x)

        x = self.conv2(x,edge_index)

        x = global_mean_pool(

            x,

            batch

        )

        return x
```

This encoder converts an entire crystal graph into one embedding vector.

---

### 23.10.10 Contrastive Loss Implementation

A simplified InfoNCE-style loss can be implemented using cosine similarity.

```python
def contrastive_loss(

    z1,

    z2,

    temperature=0.5

):

    similarity = F.cosine_similarity(

        z1,

        z2

    )

    loss = -torch.log(

        torch.exp(similarity/temperature)

    )

    return loss.mean()
```

In practice,

large batches containing many negative samples are used.

---

### 23.10.11 Training Loop

```python
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3

)

for epoch in range(100):

    optimizer.zero_grad()

    z1 = model(

        graph_view1.x,

        graph_view1.edge_index,

        graph_view1.batch

    )

    z2 = model(

        graph_view2.x,

        graph_view2.edge_index,

        graph_view2.batch

    )

    loss = contrastive_loss(

        z1,

        z2

    )

    loss.backward()

    optimizer.step()

    print(

        epoch,

        loss.item()

    )
```

During training,

the encoder gradually learns crystal embeddings that remain stable under graph augmentations.

---

### 23.10.12 Applications in Materials Informatics

Contrastive learning has rapidly become one of the most promising techniques for materials representation learning.

It has been successfully applied to

- crystal graph pretraining,
- molecular representation learning,
- transfer learning,
- property prediction,
- similarity search,
- active learning,
- foundation models,
- generative materials design.

Rather than learning only one specific property,

contrastive learning produces general-purpose crystal embeddings that can later be fine-tuned for numerous downstream tasks.

This ability to learn universal representations makes contrastive learning one of the foundational techniques underlying modern materials foundation models.

## 23.11 Transfer Learning

One of the major limitations in materials informatics is the scarcity of high-quality labeled data. While millions of crystal structures are available in public databases, only a relatively small subset has reliable Density Functional Theory (DFT) calculations or experimentally measured properties. Training a deep neural network from scratch on a small dataset often results in poor generalization and overfitting.

Transfer learning provides an effective solution to this problem.

Instead of training an entirely new neural network for every prediction task, transfer learning begins with a model that has already learned useful crystal representations from a large dataset. These learned representations are then reused and adapted for a new task using a much smaller dataset.

This approach has become one of the most successful paradigms in modern deep learning and is increasingly important in computational materials science.

---

### 23.11.1 What is Transfer Learning?

Transfer learning is the process of transferring knowledge learned from one problem to another related problem.

Instead of

```text
Random Initialization

↓

Train Entire Network

↓

Prediction
```

the workflow becomes

```text
Pretrained Model

↓

Fine-Tuning

↓

Prediction
```

The pretrained model already understands many aspects of crystal chemistry and atomic interactions.

Only a small amount of additional training is required to adapt it to a new application.

---

### 23.11.2 Source Task and Target Task

Transfer learning involves two separate learning problems.

The **source task** is the original problem used to train the model.

Example

```text
Source Dataset

↓

Formation Energy

↓

Pretrained Crystal Encoder
```

The **target task** is the new prediction problem.

Example

```text
Target Dataset

↓

Band Gap

↓

Fine-Tuned Model
```

Although the prediction targets differ,

both tasks involve learning relationships between crystal structure and material properties.

Therefore,

the learned crystal representations remain useful.

---

### 23.11.3 Why Transfer Learning Works

During pretraining,

the neural network learns fundamental concepts such as

- local coordination,
- chemical bonding,
- crystal symmetry,
- atomic environments,
- periodic interactions.

These concepts are useful for many different materials science problems.

For example,

a model trained to predict formation energy has already learned how atoms interact inside crystals.

The same knowledge can be reused when predicting

- elastic constants,
- thermal conductivity,
- dielectric constant,
- magnetic moment,
- diffusion barriers,
- defect formation energies.

Instead of learning these concepts repeatedly,

the model simply adapts its existing knowledge.

---

### 23.11.4 Feature Reuse

Transfer learning can be interpreted as feature reuse.

```text
Crystal

↓

Pretrained Encoder

↓

Crystal Embedding

↓

New Prediction Head

↓

Target Property
```

The encoder remains largely unchanged,

while only the prediction layer learns the new task.

This dramatically reduces training time.

---

### 23.11.5 Fine-Tuning

The most common transfer learning strategy is **fine-tuning**.

Initially,

the pretrained parameters are loaded.

```text
Pretrained Weights

↓

Encoder
```

A new prediction layer is then attached.

```text
Crystal Embedding

↓

New Linear Layer

↓

Target Prediction
```

Training proceeds using the smaller target dataset.

Depending on the application,

researchers may

- train only the new prediction head,
- train the final few layers,
- fine-tune the entire network.

---

### 23.11.6 Freezing Layers

Sometimes,

the encoder is frozen.

```text
Encoder

↓

Frozen

↓

Prediction Head

↓

Trainable
```

Only the prediction head updates during optimization.

This approach is useful when

- the target dataset is very small,
- overfitting is likely,
- pretrained representations are already highly informative.

---

### 23.11.7 Full Fine-Tuning

Alternatively,

every network parameter may continue updating.

```text
Encoder

↓

Trainable

↓

Prediction Head

↓

Trainable
```

This allows the pretrained representations to adapt completely to the new task.

Full fine-tuning generally requires larger datasets.

---

## 23.11.8 Code Example — Loading a Pretrained Model

Suppose a pretrained encoder has been saved.

```python
import torch

model = CrystalGNN()

model.load_state_dict(

    torch.load(

        "pretrained_encoder.pth"

    )

)
```

The encoder now contains representations learned from a large crystal dataset.

---

### 23.11.9 Freezing the Encoder

The pretrained encoder can be frozen using

```python
for parameter in model.parameters():

    parameter.requires_grad = False
```

Now,

only newly added layers will be updated.

---

### 23.11.10 Adding a New Prediction Head

```python
import torch.nn as nn

model.predictor = nn.Sequential(

    nn.Linear(64,32),

    nn.ReLU(),

    nn.Linear(32,1)

)
```

The encoder continues producing crystal embeddings,

while the prediction head learns the new property.

---

### 23.11.11 Optimizing Only Trainable Parameters

```python
optimizer = torch.optim.Adam(

    filter(

        lambda p: p.requires_grad,

        model.parameters()

    ),

    lr=1e-3

)
```

Only parameters marked as trainable participate in optimization.

---

### 23.11.12 Fine-Tuning Loop

```python
criterion = nn.MSELoss()

for epoch in range(100):

    optimizer.zero_grad()

    prediction = model(

        data.x,

        data.edge_index,

        data.batch

    )

    loss = criterion(

        prediction,

        target

    )

    loss.backward()

    optimizer.step()

    print(

        epoch,

        loss.item()

    )
```

Compared with training from scratch,

the model typically converges much faster.

---

### 23.11.13 Advantages of Transfer Learning

Transfer learning provides numerous benefits.

It

- reduces training time,
- improves convergence,
- reduces overfitting,
- requires fewer labeled samples,
- produces better representations,
- enables foundation models,
- improves generalization.

These advantages become increasingly important as neural networks grow larger.

---

### 23.11.14 Transfer Learning in Materials Science

Modern materials AI increasingly relies on transfer learning.

Typical applications include

- formation energy → band gap prediction,
- molecular datasets → crystal datasets,
- small DFT datasets → experimental datasets,
- one crystal family → another crystal family,
- pretrained foundation models → specialized materials applications.

Rather than training every model from the beginning,

researchers now develop increasingly powerful pretrained crystal encoders that can be reused across numerous downstream tasks.

This shift toward reusable representations is one of the defining characteristics of modern materials foundation models and represents an important step toward universal artificial intelligence for materials discovery.

## 23.12 Embedding Visualization

The representations learned by modern Graph Neural Networks often exist in extremely high-dimensional spaces. Typical crystal embeddings contain between 64 and 1024 numerical features. Although these high-dimensional representations are highly informative for machine learning algorithms, they cannot be directly interpreted by humans.

Embedding visualization attempts to solve this problem by projecting high-dimensional crystal embeddings into two or three dimensions while preserving as much structural information as possible.

Visualization serves several important purposes in materials informatics.

It allows researchers to

- inspect learned representations,
- identify material families,
- discover hidden clusters,
- detect outliers,
- analyze chemical similarity,
- evaluate representation quality,
- interpret neural network behavior.

Rather than viewing the neural network as a black box, embedding visualization provides a window into its internal understanding of materials.

---

### 23.12.1 Why Visualize Embeddings?

Suppose a Graph Neural Network produces a

```text
256-dimensional
```

embedding for every crystal.

One embedding might appear as

```text
[

-0.84,

1.37,

0.25,

...

-2.11

]
```

Another material might produce

```text
[

-0.79,

1.44,

0.19,

...

-2.03

]
```

Simply reading these numbers provides little insight.

However,

if the embeddings are projected into two dimensions,

their relationships become visible.

For example,

```text
Perovskites

● ● ●

          Metals

          ● ● ●

                Semiconductors

                ● ● ●

Oxides

● ● ●
```

Meaningful material families often emerge naturally.

---

### 23.12.2 High-Dimensional Geometry

Each crystal embedding is a point in a high-dimensional vector space.

For example,

```text
Crystal

↓

Embedding

↓

256-dimensional Vector
```

Thousands of materials therefore produce thousands of points.

```text
Crystal 1

↓

●

Crystal 2

↓

●

Crystal 3

↓

●
```

Humans cannot visualize spaces beyond three dimensions.

Therefore,

dimensionality reduction techniques are required.

---

### 23.12.3 Goals of Dimensionality Reduction

A visualization algorithm attempts to preserve

- neighborhood relationships,
- cluster structure,
- local similarity,
- global geometry,

while reducing

```text
256 Dimensions

↓

2 Dimensions
```

No visualization method preserves every property perfectly.

Different algorithms emphasize different aspects of the embedding space.

---

### 23.12.4 Visualization Pipeline

The complete workflow is

```text
Crystal Structures

↓

Graph Neural Network

↓

Crystal Embeddings

↓

Dimensionality Reduction

↓

2D Coordinates

↓

Visualization
```

The dimensionality reduction stage may use

- PCA,
- t-SNE,
- UMAP.

Each method will be discussed in the following sections.

---

## 23.12.5 Code Example — Extracting Crystal Embeddings

Suppose a trained Graph Neural Network has already been constructed.

The graph embedding can be extracted directly.

```python
import torch

model.eval()

embeddings = []

labels = []

with torch.no_grad():

    for data in loader:

        z = model.encoder(

            data.x,

            data.edge_index,

            data.batch

        )

        embeddings.append(

            z.cpu()

        )

        labels.append(

            data.y.cpu()

        )
```

After processing every crystal,

the embeddings are combined.

```python
embeddings = torch.cat(

    embeddings,

    dim=0

)

labels = torch.cat(

    labels,

    dim=0
)

print(

    embeddings.shape

)
```

Example output

```text
torch.Size([2500, 128])
```

This indicates

- 2500 crystals
- each represented by a 128-dimensional embedding.

---

### 23.12.6 Converting to NumPy

Most visualization libraries operate on NumPy arrays.

```python
embedding_array = embeddings.numpy()

print(

    embedding_array.shape

)
```

Output

```text
(2500, 128)
```

This array becomes the input for PCA,

t-SNE,

or UMAP.

---

### 23.12.7 Visualizing Raw Embedding Statistics

Before applying dimensionality reduction,

it is often useful to inspect the embeddings.

```python
print(

    embedding_array.mean()

)

print(

    embedding_array.std()

)

print(

    embedding_array.min()

)

print(

    embedding_array.max()

)
```

These statistics help identify

- exploding embeddings,
- collapsed representations,
- numerical instability.

---

### 23.12.8 Computing Pairwise Similarity Matrix

A similarity matrix provides another useful visualization.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(

    embedding_array

)

print(

    similarity_matrix.shape

)
```

Output

```text
(2500, 2500)
```

Each element represents the cosine similarity between two crystal embeddings.

---

### 23.12.9 Heatmap Visualization

The similarity matrix can be visualized as a heatmap.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.imshow(

    similarity_matrix,

    cmap="viridis"

)

plt.colorbar(

    label="Cosine Similarity"

)

plt.title(

    "Embedding Similarity Matrix"

)

plt.xlabel(

    "Crystal Index"

)

plt.ylabel(

    "Crystal Index"

)

plt.show()
```

The heatmap often reveals

- material families,
- repeated structures,
- hidden clusters,
- outlier crystals.

---

### 23.12.10 Visualizing Embedding Norms

The magnitude of each embedding can also be analyzed.

```python
import numpy as np

norms = np.linalg.norm(

    embedding_array,

    axis=1

)

plt.figure(figsize=(7,5))

plt.hist(

    norms,

    bins=40

)

plt.xlabel(

    "Embedding Norm"

)

plt.ylabel(

    "Frequency"

)

plt.title(

    "Distribution of Embedding Magnitudes"

)

plt.show()
```

A narrow distribution generally indicates stable representation learning,

whereas extremely large variations may indicate optimization problems.

---

### 23.12.11 Why Embedding Visualization Matters

Visualization transforms abstract numerical embeddings into interpretable scientific information.

Instead of analyzing hundreds of numerical dimensions,

researchers can directly observe

- crystal clusters,
- chemical families,
- structural similarity,
- anomalous materials,
- latent organization.

The following sections introduce three of the most widely used dimensionality reduction techniques—

- Principal Component Analysis (PCA),
- t-distributed Stochastic Neighbor Embedding (t-SNE),
- Uniform Manifold Approximation and Projection (UMAP)—

and demonstrate how they reveal the hidden structure of learned crystal representations.

## 23.13 Principal Component Analysis (PCA)

One of the earliest and most widely used techniques for visualizing high-dimensional data is **Principal Component Analysis (PCA)**. Although originally developed in statistics, PCA has become an indispensable tool in machine learning, data mining, and materials informatics.

When a Graph Neural Network generates crystal embeddings containing hundreds of dimensions, it becomes impossible to visualize their distribution directly. PCA addresses this problem by projecting the embeddings onto a lower-dimensional space while preserving as much of the original variance as possible.

Unlike nonlinear visualization techniques such as t-SNE and UMAP, PCA performs a **linear transformation** of the data. Despite its simplicity, it remains one of the most useful methods for exploring learned crystal representations.

---

### 23.13.1 Motivation

Suppose a trained Graph Neural Network produces a

```text
256-dimensional
```

embedding for every crystal.

```text
Crystal

↓

256-D Embedding
```

Humans cannot visualize a 256-dimensional space.

PCA reduces the dimensionality

```text
256 Dimensions

↓

2 Dimensions
```

while attempting to preserve the maximum possible information.

The resulting two-dimensional coordinates can then be plotted.

---

### 23.13.2 Intuition Behind PCA

Imagine a cloud of crystal embeddings distributed in a high-dimensional space.

```text
●

      ●

           ●

     ●

               ●
```

Rather than examining every dimension individually,

PCA searches for the directions along which the data varies the most.

These directions are called **principal components**.

The first principal component captures the largest variance.

The second principal component captures the largest remaining variance while remaining perpendicular to the first.

Each additional component explains progressively less variation.

---

### 23.13.3 Principal Components

Suppose the original embedding has

```text
128
```

features.

PCA computes

```text
PC₁

PC₂

PC₃

...

PC₁₂₈
```

where

- PC₁ explains the largest variance,
- PC₂ explains the second-largest variance,
- PC₃ explains the third-largest variance,

and so on.

For visualization,

only

```text
PC₁

and

PC₂
```

are usually retained.

---

### 23.13.4 Variance Explained

Every principal component explains a certain percentage of the total variance.

Example

| Principal Component | Variance Explained |
|----------------------|------------------:|
| PC₁ | 42% |
| PC₂ | 21% |
| PC₃ | 11% |
| Remaining Components | 26% |

Thus,

the first two components explain

```text
63%
```

of the information contained in the original embedding space.

---

### 23.13.5 PCA Workflow

The complete workflow becomes

```text
Crystal Structures

↓

Graph Neural Network

↓

128-D Crystal Embeddings

↓

Principal Component Analysis

↓

PC₁

PC₂

↓

2D Scatter Plot
```

This provides a simple visualization of the learned embedding space.

---

## 23.13.6 Code Example — Performing PCA

The first step is importing the required libraries.

```python
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
```

Assume

```python
embedding_array
```

contains

```text
2500 × 128
```

crystal embeddings.

---

### 23.13.7 Creating the PCA Model

```python
pca = PCA(

    n_components=2

)
```

Here,

the embeddings will be projected onto two principal components.

---

### 23.13.8 Transforming the Embeddings

```python
embedding_pca = pca.fit_transform(

    embedding_array

)

print(

    embedding_pca.shape

)
```

Output

```text
(2500, 2)
```

Each crystal is now represented by only two coordinates.

---

### 23.13.9 Visualizing the Embeddings

```python
plt.figure(

    figsize=(8,6)

)

plt.scatter(

    embedding_pca[:,0],

    embedding_pca[:,1],

    s=18,

    alpha=0.7

)

plt.xlabel(

    "Principal Component 1"

)

plt.ylabel(

    "Principal Component 2"

)

plt.title(

    "PCA Projection of Crystal Embeddings"

)

plt.grid(True)

plt.show()
```

The resulting scatter plot often reveals

- material families,
- dense regions,
- isolated structures,
- outliers.

---

### 23.13.10 Coloring by Material Property

Embedding visualization becomes much more informative when points are colored according to a physical property.

Suppose

```python
band_gap
```

contains the measured band gaps.

```python
plt.figure(

    figsize=(8,6)

)

scatter = plt.scatter(

    embedding_pca[:,0],

    embedding_pca[:,1],

    c=band_gap,

    cmap="viridis",

    s=20

)

plt.xlabel(

    "PC1"

)

plt.ylabel(

    "PC2"

)

plt.title(

    "PCA Colored by Band Gap"

)

plt.colorbar(

    scatter,

    label="Band Gap (eV)"

)

plt.show()
```

Clusters with similar colors indicate that the neural network has organized materials according to electronic properties.

---

### 23.13.11 Explained Variance Ratio

One advantage of PCA is that it quantifies how much information is preserved.

```python
print(

    pca.explained_variance_ratio_

)
```

Example output

```text
[0.42 0.21]
```

This indicates

- PC₁ explains 42% of the variance.
- PC₂ explains 21% of the variance.

Together,

they preserve

```text
63%
```

of the original information.

---

### 23.13.12 Cumulative Explained Variance

The cumulative variance can also be plotted.

```python
pca_full = PCA()

pca_full.fit(

    embedding_array

)

cumulative_variance = np.cumsum(

    pca_full.explained_variance_ratio_

)

plt.figure(

    figsize=(8,5)

)

plt.plot(

    cumulative_variance,

    linewidth=2

)

plt.xlabel(

    "Number of Principal Components"

)

plt.ylabel(

    "Cumulative Explained Variance"

)

plt.grid(True)

plt.show()
```

This figure helps determine how many principal components are required to preserve most of the information.

---

### 23.13.13 Interpreting PCA Plots

When analyzing PCA visualizations,

researchers should look for

- well-separated material families,
- overlapping crystal classes,
- isolated outliers,
- continuous property gradients,
- dense embedding regions.

These patterns provide valuable insight into the quality of the learned representations.

---

### 23.13.14 Advantages of PCA

PCA offers several important strengths.

It

- is computationally efficient,
- scales well to large datasets,
- preserves global variance,
- provides deterministic results,
- quantifies explained variance,
- is easy to interpret.

Because of these advantages,

PCA is often the first visualization technique applied to learned crystal embeddings.

---

### 23.13.15 Limitations of PCA

Despite its usefulness,

PCA has important limitations.

It

- assumes linear relationships,
- cannot preserve complex nonlinear manifolds,
- may fail to separate highly nonlinear material families,
- often compresses local neighborhood information.

Consequently,

modern representation learning frequently supplements PCA with nonlinear visualization techniques such as **t-distributed Stochastic Neighbor Embedding (t-SNE)** and **Uniform Manifold Approximation and Projection (UMAP)**, which will be discussed in the following sections.

## 23.14 t-distributed Stochastic Neighbor Embedding (t-SNE)

While Principal Component Analysis is an excellent tool for exploring global trends in crystal embeddings, it has an important limitation: it assumes that the underlying data are linearly distributed. Learned representations generated by deep neural networks, however, rarely satisfy this assumption.

Crystal embeddings often lie on highly nonlinear manifolds shaped by complex atomic interactions, chemical environments, and crystal symmetries. Capturing these nonlinear relationships requires more sophisticated visualization techniques.

One of the most successful nonlinear dimensionality reduction algorithms is **t-distributed Stochastic Neighbor Embedding (t-SNE)**.

Unlike PCA, t-SNE focuses on preserving **local neighborhood relationships**. Materials that are close together in the original high-dimensional embedding space remain close after projection into two dimensions.

Because of this property, t-SNE has become one of the most widely used visualization techniques in deep learning research.

---

### 23.14.1 Motivation

Suppose a trained Graph Neural Network produces

```text
512-dimensional
```

crystal embeddings.

Although PCA can reduce these embeddings to two dimensions,

it may merge multiple material families into overlapping clusters.

```text
PCA

↓

Large Overlapping Clusters
```

t-SNE instead attempts to preserve local similarities.

```text
t-SNE

↓

Distinct Local Clusters
```

This often reveals hidden structures that PCA cannot detect.

---

### 23.14.2 Basic Idea

Instead of preserving variance,

t-SNE attempts to preserve **neighbor relationships**.

Suppose two crystals are very similar in the original embedding space.

```text
Embedding Space

Crystal A ●──● Crystal B
```

After dimensionality reduction,

they should remain close.

```text
2D Visualization

Crystal A ●──● Crystal B
```

Likewise,

crystals that are far apart should generally remain separated.

---

### 23.14.3 High-Dimensional Similarity

The first step in t-SNE is computing pairwise similarities between embeddings.

Suppose

```text
Crystal A

Crystal B

Crystal C
```

If

```text
A

and

B
```

are chemically similar,

their similarity probability becomes large.

If

```text
A

and

C
```

are unrelated,

their similarity probability becomes small.

The algorithm therefore builds a probability distribution describing neighbor relationships.

---

### 23.14.4 Low-Dimensional Mapping

Next,

t-SNE constructs a two-dimensional representation.

Initially,

points are placed randomly.

```text
Random Initialization

●

●

●

●
```

During optimization,

the points gradually move until their neighborhood relationships resemble those in the original embedding space.

Eventually,

similar materials cluster together.

---

### 23.14.5 Why Student's t-Distribution?

Ordinary Gaussian distributions compress distant points too aggressively.

To avoid this problem,

t-SNE uses a Student's t-distribution in the low-dimensional space.

The heavier tails of the distribution

- reduce crowding,
- improve cluster separation,
- preserve neighborhood structure.

This characteristic gives the algorithm its name:

**t-distributed Stochastic Neighbor Embedding.**

---

### 23.14.6 Characteristics of t-SNE

Compared with PCA,

t-SNE

- preserves local neighborhoods,
- separates clusters more effectively,
- captures nonlinear structures,
- produces visually appealing embeddings.

However,

it does **not** preserve global distances.

Consequently,

the distance between two distant clusters in a t-SNE plot should not be interpreted quantitatively.

---

## 23.14.7 Code Example — Applying t-SNE

First,

import the required libraries.

```python
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
```

Assume

```python
embedding_array
```

contains

```text
2500 × 128
```

crystal embeddings.

---

### 23.14.8 Creating the t-SNE Model

```python
tsne = TSNE(

    n_components=2,

    perplexity=30,

    learning_rate="auto",

    init="pca",

    random_state=42

)
```

Important parameters include

- `n_components`
- `perplexity`
- `learning_rate`
- `init`

These significantly influence the visualization quality.

---

### 23.14.9 Computing the Projection

```python
embedding_tsne = tsne.fit_transform(

    embedding_array

)

print(

    embedding_tsne.shape

)
```

Output

```text
(2500, 2)
```

Each crystal now possesses

two visualization coordinates.

---

### 23.14.10 Visualizing the Embeddings

```python
plt.figure(

    figsize=(8,6)

)

plt.scatter(

    embedding_tsne[:,0],

    embedding_tsne[:,1],

    s=18,

    alpha=0.7

)

plt.title(

    "t-SNE Projection of Crystal Embeddings"

)

plt.xlabel(

    "Component 1"

)

plt.ylabel(

    "Component 2"

)

plt.grid(True)

plt.show()
```

Compared with PCA,

clusters are often considerably more distinct.

---

### 23.14.11 Coloring by Material Property

A much more informative visualization colors each point using a physical property.

```python
plt.figure(

    figsize=(8,6)

)

scatter = plt.scatter(

    embedding_tsne[:,0],

    embedding_tsne[:,1],

    c=band_gap,

    cmap="plasma",

    s=20

)

plt.colorbar(

    scatter,

    label="Band Gap (eV)"

)

plt.title(

    "t-SNE Colored by Band Gap"

)

plt.xlabel(

    "Component 1"

)

plt.ylabel(

    "Component 2"

)

plt.show()
```

Clusters exhibiting similar colors indicate that the Graph Neural Network has successfully organized materials according to electronic structure.

---

### 23.14.12 Effect of Perplexity

One of the most important hyperparameters is

```text
Perplexity
```

Small values

```text
5–15
```

produce

- many small clusters,
- highly localized neighborhoods.

Large values

```text
40–80
```

produce

- smoother global organization,
- fewer isolated clusters.

Typical values for materials datasets range from

```text
20

to

50.
```

---

### 23.14.13 Comparing PCA and t-SNE

| Property | PCA | t-SNE |
|----------|-----|--------|
| Linear | Yes | No |
| Preserves Variance | Yes | No |
| Preserves Local Neighborhoods | Moderate | Excellent |
| Preserves Global Structure | Good | Limited |
| Computational Cost | Low | High |
| Cluster Separation | Moderate | Excellent |

---

### 23.14.14 Advantages

t-SNE offers several important advantages.

It

- captures nonlinear relationships,
- reveals hidden material families,
- preserves neighborhood structure,
- produces intuitive visualizations,
- identifies crystal clusters,
- is widely used in deep learning research.

---

### 23.14.15 Limitations

Despite its popularity,

t-SNE also has several limitations.

It

- is computationally expensive,
- scales poorly to millions of samples,
- depends strongly on hyperparameters,
- does not preserve global distances,
- produces different visualizations with different random initializations.

Because of these limitations,

modern materials informatics increasingly employs **Uniform Manifold Approximation and Projection (UMAP)**, which preserves both local and global structures while scaling much more efficiently to large crystal datasets. The next section explores UMAP and its applications in materials representation learning.

## 23.15 Uniform Manifold Approximation and Projection (UMAP)

While t-distributed Stochastic Neighbor Embedding (t-SNE) has become one of the most popular visualization techniques for deep learning embeddings, it also has several important limitations. It is computationally expensive, scales poorly to very large datasets, and often struggles to preserve the global structure of the embedding space.

To address these limitations, researchers increasingly employ **Uniform Manifold Approximation and Projection (UMAP)**. UMAP is a modern nonlinear dimensionality reduction algorithm that is significantly faster than t-SNE while often producing equally informative—or even superior—visualizations.

Because of its computational efficiency and excellent ability to preserve both local and global relationships, UMAP has become one of the preferred visualization techniques in modern materials informatics.

---

### 23.15.1 Motivation

Suppose a Graph Neural Network produces

```text
512-dimensional
```

crystal embeddings for

```text
500,000
```

materials.

Applying PCA may preserve global variance but fail to separate nonlinear material families.

Applying t-SNE may reveal local clusters but require hours of computation.

UMAP seeks to provide both

- efficient computation,
- meaningful visualization.

The workflow becomes

```text
Crystal Embeddings

↓

UMAP

↓

2D Representation
```

---

### 23.15.2 Basic Idea

UMAP assumes that high-dimensional data lie on a lower-dimensional manifold.

Its objective is to preserve this manifold during dimensionality reduction.

Instead of simply maximizing variance,

or preserving local probabilities,

UMAP attempts to preserve both

- local neighborhoods,
- global topology.

Consequently,

the resulting visualization often retains more meaningful relationships between different material families.

---

### 23.15.3 Local and Global Structure

Suppose two crystals possess similar local atomic environments.

```text
Crystal A

↓

●

Crystal B

↓

●
```

UMAP attempts to keep these materials close together.

Similarly,

suppose two crystal families are fundamentally different.

```text
Perovskites

↓

● ● ●
```

```text
Oxides

↓

● ● ●
```

UMAP generally preserves the separation between these larger groups better than t-SNE.

---

### 23.15.4 Graph Interpretation of UMAP

Interestingly,

UMAP itself begins by constructing a graph.

```text
Embedding Space

↓

Nearest Neighbor Graph

↓

Graph Optimization

↓

2D Embedding
```

This graph-based formulation makes UMAP particularly compatible with Graph Neural Network embeddings.

---

### 23.15.5 Advantages Over t-SNE

Compared with t-SNE,

UMAP

- scales to much larger datasets,
- executes significantly faster,
- preserves more global structure,
- produces stable visualizations,
- supports transformation of new data,
- requires less computational memory.

These advantages explain its growing popularity in representation learning.

---

## 23.15.6 Code Example — Installing UMAP

If UMAP is not already installed,

it can be installed using

```bash
pip install umap-learn
```

---

### 23.15.7 Importing UMAP

```python
import umap

import matplotlib.pyplot as plt
```

Assume

```python
embedding_array
```

contains

```text
2500 × 128
```

crystal embeddings.

---

### 23.15.8 Creating the UMAP Model

```python
reducer = umap.UMAP(

    n_neighbors=15,

    min_dist=0.1,

    n_components=2,

    metric="euclidean",

    random_state=42

)
```

Important hyperparameters include

- `n_neighbors`
- `min_dist`
- `metric`

These control how the embedding is constructed.

---

### 23.15.9 Computing the Projection

```python
embedding_umap = reducer.fit_transform(

    embedding_array

)

print(

    embedding_umap.shape

)
```

Output

```text
(2500, 2)
```

Each crystal now has two visualization coordinates.

---

### 23.15.10 Visualizing the Embedding

```python
plt.figure(

    figsize=(8,6)

)

plt.scatter(

    embedding_umap[:,0],

    embedding_umap[:,1],

    s=18,

    alpha=0.75

)

plt.title(

    "UMAP Projection of Crystal Embeddings"

)

plt.xlabel(

    "UMAP Dimension 1"

)

plt.ylabel(

    "UMAP Dimension 2"

)

plt.grid(True)

plt.show()
```

The resulting visualization frequently reveals

- crystal families,
- structural motifs,
- chemical clusters,
- isolated materials.

---

### 23.15.11 Coloring by Band Gap

Property coloring often reveals hidden relationships.

```python
plt.figure(

    figsize=(8,6)

)

scatter = plt.scatter(

    embedding_umap[:,0],

    embedding_umap[:,1],

    c=band_gap,

    cmap="viridis",

    s=18

)

plt.colorbar(

    scatter,

    label="Band Gap (eV)"

)

plt.title(

    "UMAP Colored by Band Gap"

)

plt.xlabel(

    "UMAP 1"

)

plt.ylabel(

    "UMAP 2"

)

plt.show()
```

Clusters with similar colors indicate that the learned embeddings successfully organize materials according to electronic properties.

---

### 23.15.12 Visualizing Material Classes

Suppose every crystal belongs to one of several material classes.

```python
material_classes = [

    "Metal",

    "Semiconductor",

    "Oxide",

    "Perovskite"

]
```

Using categorical coloring,

different crystal families become immediately visible.

```python
import seaborn as sns

plt.figure(

    figsize=(8,6)

)

sns.scatterplot(

    x=embedding_umap[:,0],

    y=embedding_umap[:,1],

    hue=material_labels,

    palette="tab10",

    s=25

)

plt.title(

    "UMAP of Crystal Families"

)

plt.show()
```

This visualization often reveals remarkable separation between chemically distinct materials.

---

### 23.15.13 Transforming New Materials

Unlike t-SNE,

UMAP allows new crystal embeddings to be projected into an existing visualization.

Suppose

```python
new_embedding
```

contains embeddings from previously unseen materials.

```python
new_coordinates = reducer.transform(

    new_embedding

)

print(

    new_coordinates.shape

)
```

The new materials appear within the existing embedding map without recomputing the entire visualization.

This capability is extremely useful for

- online learning,
- active learning,
- autonomous materials discovery.

---

### 23.15.14 Comparing PCA, t-SNE, and UMAP

| Property | PCA | t-SNE | UMAP |
|----------|-----|--------|------|
| Linear | Yes | No | No |
| Preserves Local Structure | Moderate | Excellent | Excellent |
| Preserves Global Structure | Good | Limited | Good |
| Computational Speed | Very Fast | Slow | Fast |
| Large Dataset Support | Excellent | Moderate | Excellent |
| Transform New Samples | Yes | No | Yes |
| Widely Used in Deep Learning | Moderate | High | Very High |

---

### 23.15.15 Applications in Materials Informatics

UMAP has become a standard tool for analyzing learned crystal embeddings.

Typical applications include

- visualization of Graph Neural Network embeddings,
- exploration of chemical space,
- identification of crystal families,
- discovery of anomalous materials,
- active learning dataset analysis,
- foundation model evaluation,
- transfer learning diagnostics,
- generative model latent space visualization.

Because of its combination of speed, scalability, and visualization quality, UMAP is now one of the most widely adopted dimensionality reduction techniques in modern materials informatics.

In the next section, we will move beyond visualization and use these learned embeddings to perform **clustering**, allowing machine learning algorithms to automatically discover previously unknown groups of chemically similar materials.

## 23.16 Clustering in Representation Space

One of the greatest advantages of learned crystal representations is that they naturally organize chemically and structurally similar materials close together in the embedding space. Once these embeddings have been learned by a Graph Neural Network, researchers can analyze them using **unsupervised clustering algorithms**.

Unlike supervised learning, clustering does not require property labels such as band gap or formation energy. Instead, it automatically groups materials according to similarities discovered directly from their learned representations.

Clustering has become an important tool for

- discovering unknown material families,
- exploring chemical space,
- identifying structural motifs,
- detecting outliers,
- accelerating materials discovery.

---

### 23.16.1 Why Cluster Learned Embeddings?

Suppose a Graph Neural Network generates

```text
128-dimensional
```

embeddings for

```text
50,000
```

crystals.

Each crystal occupies one point inside the learned embedding space.

```text
Embedding Space

↓

●

●

●

●

●
```

Rather than manually inspecting thousands of materials,

a clustering algorithm automatically partitions them into groups.

```text
Cluster 1

● ● ●
```

```text
Cluster 2

● ● ●
```

```text
Cluster 3

● ●
```

These clusters frequently correspond to meaningful material classes.

---

### 23.16.2 What Does a Cluster Represent?

A cluster consists of materials whose embeddings are similar.

This often implies similarity in

- chemistry,
- crystal symmetry,
- local coordination,
- electronic structure,
- bonding environment.

For example,

one cluster might primarily contain

```text
Perovskites
```

while another contains

```text
Layered Oxides
```

and another

```text
Transition Metal Dichalcogenides.
```

Importantly,

the clustering algorithm is never explicitly told these categories.

They emerge automatically from the learned representation.

---

### 23.16.3 Applications of Clustering

Embedding clustering has numerous applications.

It can be used for

- materials database organization,
- chemical similarity analysis,
- candidate material selection,
- anomaly detection,
- active learning,
- dataset visualization,
- discovering novel crystal families.

Large materials databases frequently use clustering before performing expensive DFT calculations.

---

### 23.16.4 Workflow

The complete workflow becomes

```text
Crystal Structures

↓

Graph Neural Network

↓

Crystal Embeddings

↓

Clustering Algorithm

↓

Material Families
```

The quality of the resulting clusters depends strongly on the quality of the learned embeddings.

---

### 23.16.5 Distance Measures

Clustering algorithms compare embeddings using distance metrics.

Common choices include

- Euclidean distance,
- cosine distance,
- Manhattan distance.

For learned crystal embeddings,

cosine similarity is often particularly effective because it focuses on embedding direction rather than magnitude.

---

## 23.16.6 Code Example — Loading Learned Embeddings

Suppose

```python
embedding_array
```

contains

```text
2500 × 128
```

learned crystal embeddings.

```python
print(

    embedding_array.shape

)
```

Output

```text
(2500, 128)
```

Each row corresponds to one material.

---

### 23.16.7 K-Means Clustering

One of the simplest clustering algorithms is K-Means.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(

    n_clusters=6,

    random_state=42

)

cluster_labels = kmeans.fit_predict(

    embedding_array

)

print(

    cluster_labels[:10]

)
```

Possible output

```text
[2 2 5 1 0 4 4 2 1 5]
```

Each crystal is assigned to one cluster.

---

### 23.16.8 Visualizing the Clusters

The clusters become much easier to interpret after dimensionality reduction.

Suppose

```python
embedding_umap
```

has already been computed.

```python
import matplotlib.pyplot as plt

plt.figure(

    figsize=(8,6)

)

plt.scatter(

    embedding_umap[:,0],

    embedding_umap[:,1],

    c=cluster_labels,

    cmap="tab10",

    s=20

)

plt.xlabel(

    "UMAP Dimension 1"

)

plt.ylabel(

    "UMAP Dimension 2"

)

plt.title(

    "K-Means Clustering of Crystal Embeddings"

)

plt.colorbar(

    label="Cluster"

)

plt.show()
```

Distinct colors reveal the automatically discovered material groups.

---

### 23.16.9 Cluster Centers

K-Means computes one representative embedding for each cluster.

```python
centers = kmeans.cluster_centers_

print(

    centers.shape

)
```

Output

```text
(6, 128)
```

Each center represents the average embedding of one material family.

These centers can be interpreted as prototype materials.

---

### 23.16.10 Cluster Sizes

Cluster populations can be inspected.

```python
import numpy as np

unique,

counts = np.unique(

    cluster_labels,

    return_counts=True

)

for u,c in zip(

    unique,

    counts

):

    print(

        f"Cluster {u}: {c}"

    )
```

Example output

```text
Cluster 0 : 381

Cluster 1 : 462

Cluster 2 : 524

Cluster 3 : 294

Cluster 4 : 395

Cluster 5 : 444
```

Large clusters often correspond to common material families.

Very small clusters may represent rare materials or outliers.

---

### 23.16.11 Evaluating Cluster Quality

One useful metric is the **Silhouette Score**.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(

    embedding_array,

    cluster_labels

)

print(

    score

)
```

Interpretation

```text
Score ≈ 1

↓

Excellent Separation
```

```text
Score ≈ 0

↓

Overlapping Clusters
```

```text
Score < 0

↓

Poor Clustering
```

---

### 23.16.12 Finding Similar Materials

After clustering,

retrieving chemically similar materials becomes straightforward.

Suppose

```python
target_cluster = cluster_labels[25]

similar_materials = np.where(

    cluster_labels == target_cluster

)[0]

print(

    similar_materials[:20]

)
```

Every returned material belongs to the same learned representation cluster.

This approach is frequently used for similarity search in materials databases.

---

### 23.16.13 Advantages of Clustering Learned Representations

Clustering learned embeddings offers several important benefits.

It

- requires no property labels,
- reveals hidden material families,
- accelerates database exploration,
- supports similarity search,
- improves candidate selection,
- identifies anomalous structures,
- provides interpretable organization of large materials datasets.

Because modern Graph Neural Networks learn chemically meaningful embeddings, clustering often discovers scientifically meaningful crystal groups that closely correspond to known structural and chemical classes.

In the following sections, we extend these ideas to more advanced clustering techniques such as **DBSCAN** and **hierarchical clustering**, which do not require the number of clusters to be specified beforehand and can better capture complex structures within the learned representation space.

## 23.17 Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

One limitation of K-Means clustering is that the number of clusters must be specified before training begins. In many materials informatics problems, however, the true number of material families is unknown. Furthermore, real materials datasets often contain rare compounds, defective structures, metastable phases, and anomalous crystal configurations that should not necessarily belong to any cluster.

To address these challenges, density-based clustering algorithms have become increasingly popular. Among these, the most widely used method is **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**.

Unlike K-Means, DBSCAN identifies clusters by searching for regions of high sample density while simultaneously detecting isolated points as noise or outliers.

For materials informatics, this property is extremely valuable because unusual crystal structures frequently correspond to scientifically interesting materials.

---

### 23.17.1 Motivation

Consider a learned embedding space containing thousands of crystal structures.

```text
Embedding Space

↓

● ● ● ●

● ● ●

                ●

                ● ●

                ● ● ●

                         ●
```

Several dense regions are visible,

while a few isolated points lie far away from every cluster.

K-Means would force every point into one cluster,

including the isolated samples.

DBSCAN instead identifies

- dense material families,
- isolated outliers.

---

### 23.17.2 Basic Idea

DBSCAN defines clusters according to **sample density**.

Instead of assuming spherical clusters,

it searches for regions where many neighboring samples occur within a specified radius.

Conceptually,

```text
Dense Region

↓

Cluster
```

```text
Sparse Region

↓

Noise
```

Therefore,

clusters may possess highly irregular shapes.

---

### 23.17.3 Core Concepts

DBSCAN introduces three important concepts.

### Core Point

A point containing many nearby neighbors.

```text
●

● ● ●

●
```

Core points form the centers of clusters.

---

### Border Point

A point located near a dense region but containing relatively few neighbors.

```text
●

↓

Border
```

Border points become members of nearby clusters.

---

### Noise Point

A completely isolated point.

```text
                ●
```

Noise points do not belong to any cluster.

These points frequently correspond to

- unusual materials,
- defective structures,
- rare crystal phases,
- potentially novel compounds.

---

### 23.17.4 DBSCAN Parameters

DBSCAN contains two important hyperparameters.

#### ε (epsilon)

Defines the neighborhood radius.

```text
Point

↓

Search Radius
```

Nearby samples inside this radius are considered neighbors.

---

#### Minimum Samples

Specifies the minimum number of neighbors required for a point to become a core point.

Example

```text
Minimum Samples = 5
```

If fewer than five neighbors exist,

the point cannot initiate a cluster.

---

### 23.17.5 Advantages Over K-Means

Compared with K-Means,

DBSCAN

- automatically determines the number of clusters,
- identifies outliers,
- supports irregular cluster shapes,
- requires no centroid initialization.

These properties make DBSCAN particularly useful for scientific datasets.

---

## 23.17.6 Code Example — Performing DBSCAN

Import the required library.

```python
from sklearn.cluster import DBSCAN
```

Assume

```python
embedding_array
```

contains learned crystal embeddings.

```python
dbscan = DBSCAN(

    eps=0.8,

    min_samples=10

)

cluster_labels = dbscan.fit_predict(

    embedding_array

)
```

Each material is assigned a cluster label.

---

### 23.17.7 Inspecting the Cluster Labels

```python
print(

    cluster_labels[:20]

)
```

Possible output

```text
[

0

0

0

1

1

-1

2

2

2

-1

...
]
```

Notice

```text
-1
```

These samples represent noise points.

---

### 23.17.8 Counting the Clusters

Unlike K-Means,

DBSCAN automatically determines the number of clusters.

```python
import numpy as np

n_clusters = len(

    set(cluster_labels)

) - (

    -1 in cluster_labels

)

print(

    "Clusters:",

    n_clusters

)
```

Example output

```text
Clusters: 8
```

---

### 23.17.9 Counting Outliers

```python
noise_points = np.sum(

    cluster_labels == -1

)

print(

    "Noise Samples:",

    noise_points

)
```

Example output

```text
Noise Samples: 74
```

These materials deserve closer scientific investigation because they differ significantly from the majority of the dataset.

---

### 23.17.10 Visualizing DBSCAN Clusters

Suppose UMAP coordinates have already been computed.

```python
plt.figure(

    figsize=(8,6)

)

plt.scatter(

    embedding_umap[:,0],

    embedding_umap[:,1],

    c=cluster_labels,

    cmap="tab20",

    s=18

)

plt.title(

    "DBSCAN Clustering of Crystal Embeddings"

)

plt.xlabel(

    "UMAP 1"

)

plt.ylabel(

    "UMAP 2"

)

plt.colorbar(

    label="Cluster"

)

plt.show()
```

Noise samples appear with the label

```text
-1
```

and are clearly separated from dense crystal families.

---

### 23.17.11 Highlighting Outliers

Outliers can be visualized independently.

```python
noise_mask = cluster_labels == -1

plt.figure(

    figsize=(8,6)

)

plt.scatter(

    embedding_umap[:,0],

    embedding_umap[:,1],

    color="lightgray",

    s=15,

    alpha=0.4

)

plt.scatter(

    embedding_umap[noise_mask,0],

    embedding_umap[noise_mask,1],

    color="red",

    s=35,

    label="Outliers"

)

plt.legend()

plt.title(

    "Outlier Detection with DBSCAN"

)

plt.show()
```

This visualization immediately identifies unusual crystal structures.

---

### 23.17.12 Applications in Materials Discovery

DBSCAN has numerous applications in computational materials science.

It is frequently used for

- detecting novel crystal structures,
- identifying anomalous compounds,
- removing noisy data,
- discovering rare material families,
- exploring latent embedding spaces,
- preprocessing datasets before supervised learning.

Because the algorithm naturally separates dense material groups from isolated structures, it is particularly valuable for discovering potentially interesting compounds hidden within massive materials databases.

---

### 23.17.13 Advantages and Limitations

#### Advantages

DBSCAN

- automatically determines the number of clusters,
- identifies outliers,
- discovers arbitrarily shaped clusters,
- requires no centroid initialization,
- performs well for nonlinear embedding spaces.

#### Limitations

DBSCAN

- is sensitive to the choice of `eps`,
- struggles when cluster densities vary significantly,
- may merge nearby clusters if `eps` is too large,
- may split clusters if `eps` is too small.

Despite these limitations, DBSCAN remains one of the most useful clustering algorithms for analyzing learned crystal embeddings because of its ability to identify both meaningful material families and scientifically interesting outliers.

In the next section, we examine **Hierarchical Clustering**, which constructs a tree of material relationships and provides a multiscale view of chemical similarity within the learned representation space.

## 23.18 Hierarchical Clustering

While algorithms such as K-Means and DBSCAN assign each material directly to a single cluster, they do not reveal how different clusters are related to one another. In many materials science problems, understanding these relationships is just as important as identifying the clusters themselves.

For example, two groups of oxide materials may be more similar to each other than either is to metallic compounds. Similarly, different families of perovskites may form larger groups before eventually merging with other oxide materials.

Hierarchical clustering addresses this problem by constructing a **tree-like hierarchy** that represents the relationships between all materials and material families.

Instead of producing only clusters,

hierarchical clustering produces an entire hierarchy of similarity.

---

### 23.18.1 Motivation

Suppose we have learned crystal embeddings for several thousand materials.

Instead of asking

```text
Which cluster does this material belong to?
```

we ask

```text
How similar are all materials to one another?
```

The answer is represented as a hierarchical tree.

```text
All Materials

│

├── Metals

│

├── Transition Metals

│

└── Noble Metals

│

├── Oxides

│

├── Perovskites

│

└── Spinels
```

This hierarchy provides much richer scientific information than a flat clustering algorithm.

---

### 23.18.2 Bottom-Up Clustering

The most common hierarchical clustering method is **Agglomerative Clustering**.

Initially,

every material forms its own cluster.

```text
●

●

●

●

●
```

The algorithm repeatedly merges the two most similar clusters.

```text
Step 1

↓

●●

●

●

●
```

```text
Step 2

↓

●●

●●

●
```

Eventually,

all materials merge into one large hierarchy.

---

### 23.18.3 Dendrogram

The hierarchy is visualized using a **dendrogram**.

Example

```text
             ───────────────

         ────

     ────

───

A    B    C    D    E
```

The vertical height at which two branches merge represents their dissimilarity.

Materials connected near the bottom are highly similar.

Materials connected near the top are much more different.

---

### 23.18.4 Linkage Methods

Different strategies exist for measuring distances between clusters.

#### Single Linkage

Uses the closest pair of points.

```text
Minimum Distance
```

---

#### Complete Linkage

Uses the farthest pair of points.

```text
Maximum Distance
```

---

#### Average Linkage

Uses the average distance between all pairs.

```text
Average Distance
```

---

#### Ward Linkage

Minimizes the increase in cluster variance after merging.

Ward linkage is one of the most popular choices for materials embeddings because it often produces compact and meaningful material families.

---

### 23.18.5 Advantages

Hierarchical clustering provides several unique benefits.

It

- does not require specifying the number of clusters initially,
- reveals multiscale material relationships,
- produces interpretable dendrograms,
- allows clusters to be selected at different similarity levels.

---

## 23.18.6 Code Example — Agglomerative Clustering

Import the clustering library.

```python
from sklearn.cluster import AgglomerativeClustering
```

Assume

```python
embedding_array
```

contains the learned crystal embeddings.

```python
hierarchical = AgglomerativeClustering(

    n_clusters=8,

    linkage="ward"

)

cluster_labels = hierarchical.fit_predict(

    embedding_array

)
```

Each material now belongs to one hierarchical cluster.

---

### 23.18.7 Inspecting Cluster Labels

```python
print(

    cluster_labels[:20]

)
```

Possible output

```text
[

2

2

2

0

0

5

5

1

3

3

...
]
```

These labels correspond to the final partition chosen from the hierarchy.

---

### 23.18.8 Visualizing Hierarchical Clusters

Suppose UMAP coordinates have already been computed.

```python
import matplotlib.pyplot as plt

plt.figure(

    figsize=(8,6)

)

plt.scatter(

    embedding_umap[:,0],

    embedding_umap[:,1],

    c=cluster_labels,

    cmap="tab20",

    s=18

)

plt.xlabel(

    "UMAP Dimension 1"

)

plt.ylabel(

    "UMAP Dimension 2"

)

plt.title(

    "Hierarchical Clustering of Crystal Embeddings"

)

plt.colorbar(

    label="Cluster"

)

plt.show()
```

The resulting visualization often reveals chemically meaningful material families.

---

### 23.18.9 Constructing a Dendrogram

A dendrogram provides a complete visualization of the hierarchy.

```python
from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

linkage_matrix = linkage(

    embedding_array,

    method="ward"

)

plt.figure(

    figsize=(12,6)

)

dendrogram(

    linkage_matrix,

    truncate_mode="level",

    p=5

)

plt.title(

    "Hierarchical Clustering Dendrogram"

)

plt.xlabel(

    "Crystal Samples"

)

plt.ylabel(

    "Ward Distance"

)

plt.show()
```

The dendrogram reveals how crystal families merge as similarity decreases.

---

### 23.18.10 Selecting the Number of Clusters

Unlike K-Means,

hierarchical clustering allows researchers to choose the number of clusters **after** constructing the hierarchy.

For example,

a horizontal cut through the dendrogram determines the final grouping.

```text
                ─────────────

           ─────

      ─────

──────

──────────────

Cut Here

──────────────
```

A higher cut produces fewer, larger clusters.

A lower cut produces many smaller clusters.

---

### 23.18.11 Cluster Statistics

The number of materials in each cluster can be inspected.

```python
import numpy as np

unique,

counts = np.unique(

    cluster_labels,

    return_counts=True

)

for cluster,

count in zip(

    unique,

    counts

):

    print(

        f"Cluster {cluster}: {count}"

    )
```

Example output

```text
Cluster 0 : 386

Cluster 1 : 441

Cluster 2 : 372

Cluster 3 : 298

Cluster 4 : 351

Cluster 5 : 276

Cluster 6 : 219

Cluster 7 : 157
```

---

### 23.18.12 Applications in Materials Informatics

Hierarchical clustering is widely used for

- identifying crystal families,
- organizing materials databases,
- discovering structural motifs,
- analyzing learned embeddings,
- studying chemical similarity,
- transfer learning dataset analysis,
- foundation model evaluation.

Because it provides relationships at multiple similarity scales, hierarchical clustering is particularly valuable for exploring complex materials datasets where no obvious cluster boundaries exist.

---

### 23.18.13 Comparison with Other Clustering Methods

| Method | Need Number of Clusters | Detect Outliers | Hierarchical Structure | Handles Irregular Shapes |
|---------|------------------------|-----------------|------------------------|--------------------------|
| K-Means | Yes | No | No | No |
| DBSCAN | No | Yes | No | Yes |
| Hierarchical Clustering | Optional (chosen later) | Limited | Yes | Moderate |

Hierarchical clustering complements K-Means and DBSCAN by providing an interpretable multiscale view of learned crystal representations.

In the next section, we will use these clustered embeddings to perform **chemical similarity analysis**, enabling the retrieval of structurally and chemically related materials directly from the learned representation space.

## 23.19 Chemical Similarity in Learned Representation Space

One of the most powerful applications of representation learning is **chemical similarity search**. Instead of comparing materials using handcrafted descriptors or manually engineered fingerprints, modern Graph Neural Networks learn representations in which chemically and structurally similar materials naturally lie close together in the latent space.

Once a high-quality embedding space has been learned, researchers can rapidly retrieve materials with similar crystal structures, bonding environments, electronic characteristics, or physical properties. This capability forms the foundation of recommendation systems, materials retrieval engines, active learning pipelines, and inverse materials design.

Rather than asking

> *Which material has exactly these descriptors?*

we instead ask

> *Which materials are most similar according to the learned representation?*

This shift from handcrafted similarity to learned similarity is one of the defining characteristics of modern materials informatics.

---

### 23.19.1 Why Chemical Similarity Matters

Chemical similarity plays an important role throughout materials science.

Researchers often wish to answer questions such as

- Which materials resemble silicon?
- Which compounds are structurally similar to LiFePO₄?
- Which materials have bonding environments similar to graphene?
- Which known compounds resemble a newly discovered crystal?

Traditional descriptor-based methods often fail because they cannot capture the full complexity of crystal chemistry.

Learned embeddings overcome this limitation.

---

### 23.19.2 Similarity in Latent Space

Suppose a Graph Neural Network generates the following embeddings.

```text
Silicon

↓

Embedding A
```

```text
Germanium

↓

Embedding B
```

```text
Graphite

↓

Embedding C
```

Because silicon and germanium possess similar crystal structures and electronic properties,

their embeddings become close.

```text
Embedding Space

Si ●──● Ge

            Graphite ●
```

The learned representation therefore reflects chemical relationships automatically.

---

### 23.19.3 Distance as Similarity

The simplest similarity measure is distance.

Small distance

```text
↓

Highly Similar Materials
```

Large distance

```text
↓

Chemically Different Materials
```

Numerous distance metrics can be employed.

Common choices include

- Euclidean distance
- Cosine similarity
- Manhattan distance
- Mahalanobis distance

Among these,

cosine similarity is particularly popular because it compares embedding direction rather than magnitude.

---

### 23.19.4 Nearest Neighbor Search

Once every material has an embedding,

retrieval becomes straightforward.

```text
Query Material

↓

Embedding

↓

Nearest Neighbor Search

↓

Most Similar Materials
```

This operation can be performed in milliseconds, even for databases containing hundreds of thousands of crystals.

---

### 23.19.5 Applications

Similarity search is useful for

- discovering related compounds,
- identifying candidate materials,
- transfer learning,
- recommendation systems,
- database exploration,
- inverse materials design,
- active learning,
- experimental planning.

Modern materials databases increasingly provide embedding-based search rather than traditional descriptor matching.

---

## 23.19.6 Code Example — Computing Cosine Similarity

Assume

```python
embedding_array
```

contains learned crystal embeddings.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(

    embedding_array

)

print(

    similarity_matrix.shape

)
```

Example output

```text
(2500, 2500)
```

Each element

```text
(i,j)
```

contains the similarity between two crystal embeddings.

---

### 23.19.7 Finding the Most Similar Materials

Suppose material

```python
42
```

is selected as the query.

```python
query_index = 42

similarities = similarity_matrix[query_index]

ranking = similarities.argsort()[::-1]
```

The ranking now contains the indices of materials ordered by decreasing similarity.

---

### 23.19.8 Retrieving the Top Neighbors

```python
top_k = 10

nearest_neighbors = ranking[1:top_k+1]

print(

    nearest_neighbors

)
```

Possible output

```text
[

391

847

120

58

774

...

]
```

These materials possess the most similar learned crystal representations.

---

### 23.19.9 Displaying Similarity Scores

```python
for idx in nearest_neighbors:

    print(

        idx,

        similarities[idx]

    )
```

Example output

```text
391 0.992

847 0.988

120 0.986

58 0.984

774 0.982
```

Values close to

```text
1
```

indicate highly similar materials.

---

### 23.19.10 Euclidean Distance Search

Instead of cosine similarity,

Euclidean distance may also be used.

```python
from sklearn.metrics import pairwise_distances

distance_matrix = pairwise_distances(

    embedding_array,

    metric="euclidean"

)

distances = distance_matrix[query_index]

ranking = distances.argsort()

nearest_neighbors = ranking[1:11]

print(

    nearest_neighbors

)
```

Smaller distances correspond to greater similarity.

---

### 23.19.11 Fast Nearest Neighbor Search

Large materials databases often contain hundreds of thousands of embeddings.

Efficient search algorithms become essential.

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(

    n_neighbors=10,

    metric="cosine"

)

nn.fit(

    embedding_array

)

distances,

indices = nn.kneighbors(

    embedding_array[query_index].reshape(1,-1)

)

print(indices)
```

This approach is significantly faster than computing the full similarity matrix.

---

### 23.19.12 Visualizing Similar Materials

Suppose UMAP coordinates have already been computed.

```python
plt.figure(figsize=(8,6))

plt.scatter(

    embedding_umap[:,0],

    embedding_umap[:,1],

    color="lightgray",

    s=15,

    alpha=0.5

)

plt.scatter(

    embedding_umap[nearest_neighbors,0],

    embedding_umap[nearest_neighbors,1],

    color="red",

    s=60,

    label="Nearest Neighbors"

)

plt.scatter(

    embedding_umap[query_index,0],

    embedding_umap[query_index,1],

    color="blue",

    s=100,

    label="Query Material"

)

plt.legend()

plt.title(

    "Nearest Neighbor Search in Embedding Space"

)

plt.show()
```

The nearest neighbors typically form a compact region surrounding the query material.

---

### 23.19.13 Practical Applications

Embedding-based similarity search has numerous applications in materials informatics.

Researchers use it for

- recommending candidate materials,
- discovering chemically related compounds,
- selecting DFT calculations,
- identifying substitute materials,
- screening large databases,
- accelerating experimental discovery,
- building intelligent search engines for materials repositories.

Because the learned embeddings encode rich structural and chemical information, similarity retrieval frequently identifies meaningful relationships that are difficult to capture using traditional handcrafted descriptors.

Chemical similarity search therefore represents one of the most practical and scientifically valuable applications of modern representation learning in materials science.

## 23.20 Foundation Embeddings for Materials

Representation learning has evolved far beyond task-specific neural networks. Modern deep learning research is moving toward **foundation models**—large neural networks trained on enormous collections of unlabeled data that learn universal representations transferable to numerous downstream tasks.

In materials informatics, this idea has given rise to **foundation embeddings**. Rather than training separate models for predicting formation energy, band gap, elasticity, or stability, a single large model learns general crystal representations from millions of materials. These embeddings can then be adapted to virtually any materials science application.

Foundation embeddings represent one of the most important recent developments in artificial intelligence for materials discovery and are expected to play a central role in next-generation autonomous materials research.

---

### 23.20.1 What Are Foundation Embeddings?

A foundation embedding is a learned vector representation generated by a large pretrained model that captures fundamental information about a material.

Instead of learning

```text
Crystal

↓

Band Gap
```

or

```text
Crystal

↓

Formation Energy
```

the model learns

```text
Crystal

↓

Universal Representation
```

This representation can later be reused for many different prediction tasks.

---

### 23.20.2 Motivation

Traditional supervised learning follows the workflow

```text
Task

↓

Train Model

↓

Prediction
```

Every new task requires training a completely new neural network.

Foundation models instead perform

```text
Millions of Crystals

↓

Large Pretraining

↓

Universal Embeddings

↓

Many Downstream Tasks
```

The expensive pretraining process is performed only once.

The learned embeddings become reusable across many applications.

---

### 23.20.3 Learning Universal Representations

During pretraining,

the model encounters enormous structural diversity.

For example,

```text
Metals

↓

Oxides

↓

Perovskites

↓

Layered Materials

↓

Polymers

↓

MOFs

↓

Semiconductors
```

Rather than memorizing one property,

the neural network gradually learns

- atomic environments,
- bonding patterns,
- crystal symmetry,
- coordination chemistry,
- periodic interactions,
- long-range structural relationships.

These learned concepts become embedded inside a universal latent space.

---

### 23.20.4 Foundation Model Workflow

The complete workflow becomes

```text
Millions of Crystal Structures

↓

Self-Supervised Learning

↓

Large Graph Neural Network

↓

Foundation Embeddings

↓

Fine-Tuning

↓

Specific Materials Task
```

The same pretrained encoder can therefore be used for

- band gap prediction,
- formation energy,
- elastic constants,
- dielectric properties,
- diffusion,
- catalysis,
- defect prediction.

---

### 23.20.5 Transferability

A key characteristic of foundation embeddings is transferability.

Suppose the encoder has been pretrained using millions of crystal structures.

Later,

only a small dataset may be available for predicting

```text
Thermal Conductivity
```

Instead of training a completely new network,

the pretrained embedding is reused.

```text
Foundation Embedding

↓

Small Prediction Head

↓

Thermal Conductivity
```

This dramatically reduces the amount of labeled data required.

---

### 23.20.6 Embedding Space

Every crystal is mapped into a common latent space.

```text
Embedding Space

↓

Metals

● ● ●

Oxides

● ● ●

Perovskites

● ● ●

MOFs

● ● ●
```

Materials with similar chemistry naturally occupy nearby regions.

This organization emerges automatically during large-scale pretraining.

---

## 23.20.7 Code Example — Using a Pretrained Crystal Encoder

Suppose a pretrained encoder has already been downloaded.

```python
import torch

encoder = CrystalEncoder()

encoder.load_state_dict(

    torch.load(

        "foundation_encoder.pth"

    )

)

encoder.eval()
```

The encoder now produces foundation embeddings for any crystal.

---

### 23.20.8 Generating Foundation Embeddings

```python
with torch.no_grad():

    embedding = encoder(

        data.x,

        data.edge_index,

        data.batch

    )

print(

    embedding.shape

)
```

Example output

```text
torch.Size([1, 256])
```

The entire crystal has been converted into a

```text
256-dimensional
```

foundation embedding.

---

### 23.20.9 Saving Embeddings

Large datasets often precompute embeddings.

```python
embeddings = []

with torch.no_grad():

    for data in loader:

        z = encoder(

            data.x,

            data.edge_index,

            data.batch

        )

        embeddings.append(

            z.cpu()

        )

embeddings = torch.cat(

    embeddings,

    dim=0

)

torch.save(

    embeddings,

    "foundation_embeddings.pt"

)
```

The saved embeddings can later be reused without rerunning the Graph Neural Network.

---

### 23.20.10 Using Embeddings for Downstream Prediction

Instead of training an entire GNN,

only a small prediction network is required.

```python
import torch.nn as nn

predictor = nn.Sequential(

    nn.Linear(256,128),

    nn.ReLU(),

    nn.Linear(128,1)

)

prediction = predictor(

    embedding

)
```

This greatly reduces computational cost.

---

### 23.20.11 Similarity Search with Foundation Embeddings

Foundation embeddings naturally support similarity search.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(

    embedding_database,

    embedding

)

nearest = similarity.argsort(

    axis=0

)[::-1][:10]

print(

    nearest

)
```

The retrieved materials are the most chemically similar compounds according to the pretrained model.

---

### 23.20.12 Fine-Tuning Foundation Models

Rather than training from scratch,

the pretrained encoder can be fine-tuned.

```python
for parameter in encoder.parameters():

    parameter.requires_grad = True

optimizer = torch.optim.Adam(

    encoder.parameters(),

    lr=1e-5
)
```

A very small learning rate is typically sufficient because the encoder already contains extensive chemical knowledge.

---

### 23.20.13 Advantages of Foundation Embeddings

Foundation embeddings provide numerous advantages.

They

- require fewer labeled samples,
- improve prediction accuracy,
- accelerate convergence,
- support transfer learning,
- enable similarity search,
- improve active learning,
- generalize across many materials tasks,
- reduce computational cost.

These properties make foundation embeddings one of the most promising directions in modern materials artificial intelligence.

---

### 23.20.14 Current Foundation Models in Materials Science

Several modern materials AI models employ foundation representations.

Examples include

- large pretrained Graph Neural Networks,
- self-supervised crystal encoders,
- multimodal materials foundation models,
- language-guided materials models,
- graph transformer foundation models.

These models continue to evolve rapidly as larger datasets and more powerful computational resources become available.

---

### 23.20.15 Future Outlook

Foundation embeddings represent a major shift in computational materials science. Rather than developing separate machine learning models for every individual property, researchers are increasingly moving toward universal pretrained crystal encoders that learn general chemical and structural knowledge from massive datasets.

These universal representations form the basis for modern transfer learning, active learning, autonomous laboratories, inverse materials design, and generative materials discovery. As foundation models continue to improve, they are expected to become the core computational engine behind next-generation materials informatics and AI-driven materials discovery.

## 23.21 Chapter Summary

In this chapter, we explored one of the most fundamental concepts in modern machine learning for materials science—**representation learning**. Rather than relying on manually engineered descriptors, representation learning enables neural networks to automatically learn meaningful numerical representations of crystal structures directly from data.

We began by understanding why representations are central to modern artificial intelligence. Classical machine learning algorithms depend heavily on handcrafted features such as elemental properties, composition vectors, or crystal descriptors. Although these descriptors have achieved considerable success, they often fail to capture the full complexity of atomic interactions and crystal chemistry. Deep learning overcomes this limitation by learning hierarchical representations directly from raw crystal graphs.

We then examined **crystal embeddings**, where every material is represented as a dense numerical vector within a latent space. During Graph Neural Network training, atoms, bonds, local environments, and entire crystal structures are progressively transformed into increasingly informative embeddings. These learned representations naturally organize chemically and structurally similar materials close together while separating unrelated compounds.

The concept of **latent space** was introduced as the hidden feature space learned by deep neural networks. Although latent dimensions do not correspond directly to physical quantities such as electronegativity or atomic radius, they encode complex combinations of structural and chemical information that prove highly useful for downstream prediction tasks.

Next, we studied **representation learning within Graph Neural Networks**, where message passing allows neighboring atoms to exchange information and gradually build increasingly expressive atomic and crystal embeddings. Pooling operations convert these atomic embeddings into graph-level crystal representations suitable for predicting material properties.

The chapter then introduced **self-supervised learning**, one of the most important advances in modern deep learning. Instead of relying on expensive DFT calculations or experimental labels, self-supervised learning constructs learning objectives directly from unlabeled crystal structures. Tasks such as masked atom prediction, masked bond prediction, and crystal reconstruction enable neural networks to learn general-purpose crystal representations from massive materials databases.

Building upon self-supervised learning, we explored **contrastive learning**, where the neural network learns by bringing representations of similar materials closer together while pushing different materials farther apart. By generating multiple augmented views of the same crystal and optimizing similarity between their embeddings, contrastive learning produces highly discriminative representations suitable for transfer learning and downstream prediction.

The chapter continued with **transfer learning**, demonstrating how pretrained crystal encoders can be adapted to new prediction tasks using relatively small labeled datasets. Rather than training a new Graph Neural Network for every property, pretrained representations can be reused and fine-tuned for predicting formation energies, band gaps, elastic properties, thermal conductivity, and numerous other material characteristics.

Because learned embeddings exist in high-dimensional spaces, we next investigated methods for **embedding visualization**. Visualization allows researchers to inspect learned representations, discover material families, identify outliers, and evaluate representation quality. Three major dimensionality reduction techniques were discussed:

- **Principal Component Analysis (PCA)**, which performs linear dimensionality reduction while preserving maximum variance.
- **t-distributed Stochastic Neighbor Embedding (t-SNE)**, which preserves local neighborhood relationships and reveals nonlinear material clusters.
- **Uniform Manifold Approximation and Projection (UMAP)**, which combines computational efficiency with excellent preservation of both local and global embedding structures.

Each visualization technique was accompanied by complete Python implementations demonstrating how learned crystal embeddings can be projected into two-dimensional spaces for scientific interpretation.

After visualization, we explored **clustering** within learned representation spaces. Clustering algorithms enable automatic discovery of chemically similar materials without requiring property labels. Three important clustering methods were examined:

- **K-Means Clustering**, which partitions embedding space into predefined clusters.
- **DBSCAN**, a density-based algorithm capable of discovering irregularly shaped material families while simultaneously identifying outlier materials.
- **Hierarchical Clustering**, which constructs multiscale dendrograms revealing relationships among crystal families.

These clustering techniques provide powerful tools for organizing large materials databases, discovering novel material classes, and exploring chemical similarity.

Building upon clustering, we investigated **chemical similarity search**, where learned crystal embeddings replace traditional handcrafted descriptors. Cosine similarity, Euclidean distance, nearest-neighbor search, and embedding retrieval were demonstrated through complete Python implementations. Learned representations enable rapid retrieval of structurally and chemically related materials from databases containing hundreds of thousands of compounds.

Finally, the chapter introduced **foundation embeddings**, representing one of the newest directions in materials informatics. Large pretrained Graph Neural Networks trained on millions of crystal structures can produce universal crystal embeddings that transfer across numerous downstream tasks. Rather than constructing separate neural networks for every property prediction problem, a single foundation model learns general chemical and structural knowledge that can be fine-tuned for formation energy prediction, band gap estimation, elasticity, diffusion, catalysis, and many other applications.

Throughout this chapter, extensive code implementations demonstrated how modern representation learning techniques can be applied using Python, PyTorch, PyTorch Geometric, Scikit-learn, UMAP, and related scientific libraries. These implementations illustrated the complete workflow from crystal embeddings and self-supervised learning to visualization, clustering, similarity search, and foundation model inference.

Representation learning has fundamentally transformed materials informatics by replacing manually engineered descriptors with automatically learned crystal representations. These learned embeddings have become the foundation for modern Graph Neural Networks, self-supervised learning, contrastive learning, transfer learning, active learning, generative AI, and autonomous materials discovery. As materials foundation models continue to improve, representation learning will remain one of the central technologies driving the future of artificial intelligence in computational materials science.

## 23.22 Code Implementation — Building a Complete Representation Learning Pipeline

In the previous sections, we discussed representation learning, crystal embeddings, self-supervised learning, contrastive learning, visualization, clustering, similarity search, and foundation embeddings individually. In practice, however, these components are combined into a single pipeline.

This section demonstrates how a modern representation learning workflow is implemented using PyTorch Geometric.

The pipeline consists of

```text
Crystal Structures

↓

Graph Construction

↓

Graph Neural Network Encoder

↓

Crystal Embeddings

↓

Embedding Storage

↓

Visualization

↓

Clustering

↓

Similarity Search

↓

Downstream Prediction
```

---

### 23.22.1 Import Required Libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import numpy as np
```

---

## 23.22.2 Graph Encoder

This encoder converts an atomic graph into a crystal embedding.

```python
class CrystalEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(16,64)

        self.conv2 = GCNConv(64,128)

        self.conv3 = GCNConv(128,256)

    def forward(

        self,

        x,

        edge_index,

        batch

    ):

        x = self.conv1(x,edge_index)

        x = F.relu(x)

        x = self.conv2(x,edge_index)

        x = F.relu(x)

        x = self.conv3(x,edge_index)

        embedding = global_mean_pool(

            x,

            batch

        )

        return embedding
```

---

## 23.22.3 Generate Crystal Embeddings

```python
encoder = CrystalEncoder()

encoder.eval()

embeddings = []

with torch.no_grad():

    for data in loader:

        z = encoder(

            data.x,

            data.edge_index,

            data.batch

        )

        embeddings.append(

            z.cpu()

        )

embeddings = torch.cat(

    embeddings,

    dim=0

)

print(

    embeddings.shape

)
```

Example

```text
torch.Size([5000,256])
```

---

## 23.22.4 Save Learned Embeddings

```python
torch.save(

    embeddings,

    "crystal_embeddings.pt"

)
```

Later,

the embeddings can be reloaded without running the neural network again.

```python
embeddings = torch.load(

    "crystal_embeddings.pt"

)
```

---

## 23.22.5 Convert to NumPy

```python
embedding_array = embeddings.numpy()

print(

    embedding_array.shape

)
```

Output

```text
(5000,256)
```

---

## 23.22.6 PCA Visualization

```python
pca = PCA(

    n_components=2

)

embedding_pca = pca.fit_transform(

    embedding_array

)

plt.figure(figsize=(8,6))

plt.scatter(

    embedding_pca[:,0],

    embedding_pca[:,1],

    s=15

)

plt.title(

    "PCA Projection"

)

plt.show()
```

---

## 23.22.7 K-Means Clustering

```python
kmeans = KMeans(

    n_clusters=8,

    random_state=42

)

cluster_labels = kmeans.fit_predict(

    embedding_array

)

print(

    cluster_labels[:20]

)
```

---

## 23.22.8 Plot Clustered Embeddings

```python
plt.figure(figsize=(8,6))

plt.scatter(

    embedding_pca[:,0],

    embedding_pca[:,1],

    c=cluster_labels,

    cmap="tab10",

    s=20

)

plt.title(

    "Material Clusters"

)

plt.colorbar()

plt.show()
```

---

## 23.22.9 Similarity Search

```python
similarity = cosine_similarity(

    embedding_array

)

query = 100

ranking = similarity[query].argsort()[::-1]

top_neighbors = ranking[1:11]

print(

    top_neighbors

)
```

Example

```text
[91 233 851 402 771 15 918 311 74 128]
```

---

## 23.22.10 Fine-Tuning on a Property Prediction Task

```python
class PropertyPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(

            256,

            128

        )

        self.fc2 = nn.Linear(

            128,

            1

        )

    def forward(

        self,

        x

    ):

        x = F.relu(

            self.fc1(x)

        )

        return self.fc2(x)
```

---

## 23.22.11 Training the Prediction Head

```python
predictor = PropertyPredictor()

optimizer = torch.optim.Adam(

    predictor.parameters(),

    lr=1e-3

)

criterion = nn.MSELoss()

for epoch in range(100):

    optimizer.zero_grad()

    prediction = predictor(

        embeddings

    )

    loss = criterion(

        prediction,

        targets

    )

    loss.backward()

    optimizer.step()

    print(

        epoch,

        loss.item()

    )
```

---

## 23.22.12 Predicting New Materials

```python
encoder.eval()

predictor.eval()

with torch.no_grad():

    embedding = encoder(

        new_data.x,

        new_data.edge_index,

        new_data.batch

    )

    property_prediction = predictor(

        embedding

    )

print(

    property_prediction

)
```

---

## 23.22.13 End-to-End Representation Learning Pipeline

The complete workflow implemented in this chapter can be summarized as

```text
Crystal Structure

↓

Graph Construction

↓

Graph Neural Network Encoder

↓

Crystal Embedding

↓

├──────────────┬──────────────┬──────────────┬──────────────┐

▼              ▼              ▼              ▼

PCA          Clustering   Similarity Search  Fine-Tuning

▼              ▼              ▼              ▼

Visualization Material Families Retrieval Property Prediction
```

This pipeline represents the standard workflow used in modern materials informatics research. Whether the final objective is property prediction, similarity retrieval, clustering, active learning, or generative AI, nearly all contemporary materials foundation models first learn high-quality crystal embeddings and then reuse these representations for downstream scientific tasks.

This concludes the practical implementation of representation learning for materials.

## 23.23 Exercises

The following exercises are designed to reinforce both the theoretical concepts and practical implementation of representation learning in materials informatics. They progress from conceptual understanding to programming implementation and finally to research-oriented applications.

---

# Conceptual Questions

### Question 1

Why are learned crystal representations generally more powerful than handcrafted descriptors?

---

### Question 2

Explain the difference between

- feature engineering
- feature learning
- representation learning.

---

### Question 3

What is meant by a crystal embedding?

How does it differ from the original crystal graph?

---

### Question 4

Define latent space.

Why is latent space important for Graph Neural Networks?

---

### Question 5

Why do similar materials tend to occupy nearby locations in the embedding space?

---

### Question 6

What is self-supervised learning?

Why is it particularly valuable for materials science?

---

### Question 7

Explain the difference between

- supervised learning,
- self-supervised learning,
- contrastive learning.

---

### Question 8

What are positive pairs and negative pairs in contrastive learning?

---

### Question 9

Explain how transfer learning reduces the amount of labeled data required for materials prediction.

---

### Question 10

Compare PCA, t-SNE, and UMAP.

Under what situations would each visualization technique be preferred?

---

### Question 11

What is the purpose of clustering learned crystal embeddings?

---

### Question 12

Compare

- K-Means,
- DBSCAN,
- Hierarchical Clustering.

Discuss their advantages and limitations.

---

### Question 13

Why is cosine similarity commonly used for learned crystal embeddings?

---

### Question 14

What are foundation embeddings?

How are they changing modern materials informatics?

---

# Programming Exercises

---

## Exercise 1

Generate random crystal embeddings.

```python
import torch

embeddings = torch.randn(

    1000,

    128

)
```

Tasks

- Compute the embedding mean.
- Compute the standard deviation.
- Compute the embedding norms.
- Plot the norm distribution.

---

## Exercise 2

Perform PCA visualization.

Requirements

- Reduce the embeddings to two dimensions.
- Create a scatter plot.
- Label both principal components.
- Report the explained variance ratio.

---

## Exercise 3

Apply t-SNE.

Experiment with

```text
Perplexity

=

10

20

40

60
```

Compare the resulting visualizations.

Discuss the effect of perplexity.

---

## Exercise 4

Perform UMAP visualization.

Experiment with

```text
n_neighbors

=

10

20

50
```

Discuss how neighborhood size influences the embedding structure.

---

## Exercise 5

Cluster the embeddings using K-Means.

Perform clustering with

```text
k

=

4

6

8

10
```

Compare

- cluster size,
- silhouette score,
- visualization quality.

---

## Exercise 6

Apply DBSCAN.

Experiment with

```text
eps

=

0.3

0.5

0.8

1.2
```

Determine

- number of clusters,
- number of outliers,
- cluster stability.

---

## Exercise 7

Construct a dendrogram using hierarchical clustering.

Identify

- large material families,
- small clusters,
- isolated branches.

---

## Exercise 8

Perform similarity search.

Select one material.

Retrieve

```text
Top 10

Most Similar Materials
```

using

- cosine similarity,
- Euclidean distance.

Compare the retrieved neighbors.

---

## Exercise 9

Implement a simple contrastive learning objective.

Hints

- Generate two augmented graphs.
- Compute graph embeddings.
- Compute cosine similarity.
- Minimize the contrastive loss.

---

## Exercise 10

Freeze the encoder of a pretrained Graph Neural Network.

Train only a new prediction head.

Compare

- training speed,
- prediction accuracy,

against a model trained entirely from scratch.

---

# Research Exercises

---

## Exercise 11

Download a crystal dataset from the Materials Project.

Train a Graph Neural Network encoder.

Generate crystal embeddings.

Visualize the embeddings using

- PCA,
- t-SNE,
- UMAP.

Compare all three visualizations.

---

## Exercise 12

Cluster the generated embeddings.

Investigate whether the discovered clusters correspond to

- crystal systems,
- chemical composition,
- band gap,
- formation energy.

---

## Exercise 13

Perform nearest-neighbor retrieval for ten randomly selected crystals.

Evaluate whether the retrieved materials possess similar

- compositions,
- structures,
- physical properties.

---

## Exercise 14

Train a self-supervised encoder using masked node prediction.

Compare downstream property prediction performance with

- random initialization,
- supervised-only training.

---

## Exercise 15

Implement contrastive pretraining.

Fine-tune the encoder on

- band gap prediction,
- formation energy prediction.

Compare the results with a randomly initialized encoder.

---

## Exercise 16

Construct a foundation embedding database for a large crystal dataset.

Develop a recommendation engine capable of retrieving the most chemically similar materials given a query crystal.

---

# Programming Challenge

Develop a complete representation learning framework for crystal materials.

Your implementation should include

- Graph Neural Network encoder
- Crystal embeddings
- Self-supervised pretraining
- Contrastive learning
- Transfer learning
- PCA visualization
- t-SNE visualization
- UMAP visualization
- K-Means clustering
- DBSCAN clustering
- Hierarchical clustering
- Similarity search
- Foundation embedding generation
- Downstream property prediction

The final framework should accept crystal structures as input and produce learned crystal representations that can be reused for visualization, clustering, similarity retrieval, and materials property prediction.

---

# Chapter Takeaways

After completing this chapter, the reader should be able to

- explain the importance of representation learning in materials informatics,
- understand crystal embeddings and latent spaces,
- implement self-supervised and contrastive learning methods,
- apply transfer learning to pretrained crystal encoders,
- visualize embeddings using PCA, t-SNE, and UMAP,
- cluster learned crystal representations using multiple algorithms,
- perform chemical similarity search,
- generate and utilize foundation embeddings,
- build complete representation learning pipelines for modern materials AI research.

With these concepts mastered, the reader is now prepared to study **Chapter 24 — Generative AI for Materials Discovery**, where learned crystal representations become the foundation for inverse design, crystal generation, diffusion models, graph VAEs, and AI-driven materials discovery.

