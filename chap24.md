# Chapter 24 — Generative AI for Materials Discovery

## 24.1 Introduction

- Why generative AI?
- The challenge of discovering new materials
- Forward prediction vs inverse design
- Why representation learning naturally leads to generation
- Modern AI-driven materials discovery

---

## 24.2 Motivation

- Immense size of chemical space
- Limitations of high-throughput DFT
- Cost of experiments
- Autonomous materials discovery
- AI-assisted materials design
- Industrial applications

---

## 24.3 Forward Design vs Inverse Design

- Traditional workflow
- Inverse design philosophy
- Property-conditioned generation
- Closed-loop optimization
- AI design cycle

### Code Implementation
- Visualizing forward vs inverse workflows
- Simple inverse optimization example

---

## 24.4 Fundamentals of Generative Models

- Discriminative vs generative models
- Probability distributions
- Latent-variable models
- Sampling
- Likelihood
- Maximum likelihood estimation
- Latent spaces revisited

### Code Implementation
- Sampling Gaussian latent variables
- Visualizing latent distributions

---

## 24.5 Autoencoders

- Encoder
- Decoder
- Reconstruction
- Bottleneck
- Feature compression
- Latent representation
- Reconstruction loss

### Mathematics

- Encoder mapping
- Decoder mapping
- Reconstruction objective

### Code Implementation

- Autoencoder architecture
- Training
- Latent extraction
- Reconstruction
- Visualization

---

## 24.6 Variational Autoencoders (VAEs)

- Motivation
- Latent probability distributions
- Gaussian latent variables
- KL divergence
- ELBO
- Reparameterization trick
- Sampling
- Latent interpolation
- Crystal generation

### Mathematics

- ELBO derivation
- KL divergence
- Reparameterization

### Code Implementation

- Complete VAE
- Training
- Sampling
- Crystal generation
- Latent interpolation
- Property interpolation

---

## 24.7 Graph Variational Autoencoders

- Why GraphVAE
- Graph encoder
- Graph decoder
- Graph reconstruction
- Crystal graph generation
- Loss functions

### Mathematics

- Graph reconstruction loss
- Latent sampling

### Code Implementation

- GraphVAE using PyTorch Geometric
- Training
- Sampling
- Crystal reconstruction

---

## 24.8 Generative Adversarial Networks (GANs)

- Generator
- Discriminator
- Minimax optimization
- Nash equilibrium
- Generator loss
- Discriminator loss
- Training instability
- Mode collapse

### Mathematics

- GAN objective
- Minimax optimization

### Code Implementation

- Generator
- Discriminator
- Training loop
- Crystal descriptor generation

---

## 24.9 Graph GANs

- Graph generation
- Node generation
- Edge generation
- Crystal graph synthesis
- Applications

### Code Implementation

- Graph Generator
- Graph Discriminator
- Training
- Crystal generation

---

## 24.10 Diffusion Models

- Motivation
- Forward diffusion
- Reverse diffusion
- Noise schedule
- Denoising
- DDPM
- DDIM
- Score matching
- Sampling

### Mathematics

- Forward process
- Reverse process
- Noise prediction
- Diffusion loss

### Code Implementation

- Forward diffusion
- Reverse diffusion
- Noise scheduler
- DDPM implementation
- Sampling

---

## 24.11 Crystal Diffusion Models

- DiffCSP
- Crystal Diffusion
- Periodic crystals
- Fractional coordinates
- Lattice prediction
- Symmetry preservation
- Periodic boundary conditions

### Code Implementation

- Crystal diffusion workflow
- Sampling new crystals
- Lattice generation
- Coordinate prediction

---

## 24.12 Score-Based Diffusion Models

- Score matching
- Langevin dynamics
- Noise-conditioned score networks
- Sampling

### Mathematics

- Score function
- Langevin dynamics

### Code Implementation

- Score network
- Langevin sampler

---

## 24.13 Property-Conditioned Generation

- Conditional VAE
- Conditional GAN
- Conditional Diffusion
- Band-gap-conditioned generation
- Formation-energy-conditioned generation
- Multi-property optimization

### Code Implementation

- Conditional embeddings
- Conditional VAE
- Conditional diffusion

---

## 24.14 Latent Space Optimization

- Searching latent space
- Bayesian optimization
- Gradient optimization
- Evolutionary optimization
- Multi-objective optimization

### Code Implementation

- Latent optimization
- Property maximization
- Multi-objective search

---

## 24.15 Evaluating Generated Materials

- Validity
- Novelty
- Diversity
- Uniqueness
- Stability
- Synthesizability
- DFT validation
- Experimental validation

### Code Implementation

- Novelty calculation
- Diversity metrics
- Validity checks
- Duplicate detection

---

## 24.16 Modern Foundation Generative Models

- Large pretrained crystal generators
- Foundation models
- Crystal language models
- Large Graph Transformers
- Zero-shot generation
- Few-shot generation

### Code Implementation

- Using pretrained generative models
- Sampling
- Fine-tuning

---

## 24.17 Text-Guided Materials Generation

- Natural-language-guided generation
- Large Language Models
- Multimodal foundation models
- Prompt engineering
- AI-assisted discovery
- Future outlook

### Code Implementation

- Prompt encoding
- Text-conditioned generation (illustrative workflow)

---

## 24.18 Complete Materials Generation Pipeline

A complete end-to-end project

```text
Crystal Dataset

↓

Graph Construction

↓

Graph Encoder

↓

Latent Representation

↓

VAE / GraphVAE / Diffusion

↓

Generate Candidate Crystal

↓

Property Predictor

↓

Filter Candidates

↓

DFT Validation

↓

Experimental Validation

↓

Novel Material Discovery
```

### Complete Code Implementation

- Dataset preparation
- Graph construction
- Training generative model
- Sampling
- Property prediction
- Ranking generated materials
- Saving generated structures

---

## 24.19 Chapter Summary

---

## 24.20 Code Implementation Summary

A complete repository-style implementation combining every major model developed in the chapter.

---

## 24.21 Exercises

### Conceptual Questions

### Programming Exercises

### Research Exercises

### End-to-End Project


# Chapter 24 — Generative AI for Materials Discovery

## 24.1 Introduction

Machine learning has revolutionized materials science by enabling accurate prediction of material properties directly from crystal structures. Throughout the previous chapters, we focused primarily on **predictive models**, where the objective was to estimate properties such as formation energy, band gap, elastic modulus, dielectric constant, or thermal conductivity from an existing material.

While these predictive models are extremely useful, they address only one side of the materials discovery problem.

A much more ambitious scientific question is

> **Can artificial intelligence discover completely new materials that have never existed before?**

This question represents the central objective of **Generative Artificial Intelligence (Generative AI)** in materials science.

Instead of predicting the properties of existing materials, generative models learn the underlying distribution of crystal structures and use this knowledge to create entirely new candidate materials. These generated materials can subsequently be evaluated using density functional theory (DFT), molecular dynamics, or experimental synthesis.

Generative AI therefore shifts the focus from

```text
Prediction
```

to

```text
Creation.
```

This transition represents one of the most important developments in modern computational materials science.

---

## 24.1.1 From Prediction to Generation

Throughout this book, most machine learning models have followed a supervised learning paradigm.

```text
Crystal Structure

↓

Machine Learning Model

↓

Predicted Property
```

For example,

```text
Crystal

↓

Graph Neural Network

↓

Band Gap
```

or

```text
Crystal

↓

Random Forest

↓

Formation Energy
```

In every case,

the material already exists.

The machine learning model simply predicts one or more properties.

Generative AI reverses this process.

Instead of predicting properties,

the objective becomes

```text
Desired Property

↓

Generative Model

↓

New Crystal Structure
```

This process is known as **inverse materials design**, because the desired properties are specified first, and the material is generated afterward.

---

## 24.1.2 Why Generative AI?

The number of theoretically possible crystalline materials is astronomically large.

Even if only a small number of chemical elements are considered,

the possible combinations of

- chemical composition,
- crystal symmetry,
- atomic coordinates,
- lattice parameters,
- defects,
- dopants,

produce an effectively infinite design space.

For example,

```text
118 Elements

↓

Countless Compositions

↓

Countless Crystal Structures

↓

Enormous Chemical Space
```

Exploring this space experimentally is impossible.

Even high-throughput DFT calculations require enormous computational resources.

Generative AI offers a fundamentally different strategy.

Instead of exhaustively searching all possibilities,

the model learns the statistical distribution of existing materials and proposes only promising candidates.

---

## 24.1.3 Learning the Distribution of Materials

Unlike supervised learning,

generative models attempt to learn

```text
P(Material)
```

or

```text
P(Crystal Structure)
```

rather than

```text
P(Property | Material)
```

In other words,

the neural network attempts to understand

- how atoms are arranged,
- which crystal structures are physically meaningful,
- what combinations of elements are chemically plausible,
- how real materials are distributed within chemical space.

Once this distribution has been learned,

the model can sample from it to generate new structures.

---

## 24.1.4 Relationship with Representation Learning

Chapter 23 introduced one of the most important concepts in deep learning:

```text
Crystal

↓

Graph Neural Network

↓

Crystal Embedding
```

These learned embeddings capture

- chemistry,
- bonding,
- local environments,
- crystal symmetry,
- structural similarity.

Generative AI builds directly upon these representations.

Instead of stopping after learning the embedding,

the model learns how to generate entirely new embeddings,

or alternatively,

how to decode embeddings back into crystal structures.

The workflow therefore becomes

```text
Crystal

↓

Encoder

↓

Latent Representation

↓

Decoder

↓

Generated Crystal
```

This encoder–decoder framework forms the basis of many modern generative models.

---

## 24.1.5 Types of Generative Models

Several classes of generative models have been successfully applied to materials science.

These include

- Autoencoders
- Variational Autoencoders (VAEs)
- Graph Variational Autoencoders
- Generative Adversarial Networks (GANs)
- Graph GANs
- Diffusion Models
- Score-Based Generative Models
- Large Foundation Models

Each of these approaches generates materials in a different manner.

Some generate directly in latent space,

while others iteratively construct crystal structures through denoising or adversarial learning.

The following chapters explore each of these methods in detail.

---

## 24.1.6 Applications in Materials Science

Generative AI has rapidly become one of the most active research areas in computational materials science.

Current applications include

- battery materials,
- solid electrolytes,
- catalysts,
- superconductors,
- thermoelectric materials,
- photovoltaic materials,
- magnetic materials,
- metal-organic frameworks,
- polymers,
- two-dimensional materials.

Rather than screening millions of candidates,

researchers can generate a relatively small number of highly promising materials that satisfy specific design objectives.

---

## 24.1.7 Advantages Over Conventional Discovery

Compared with traditional high-throughput screening,

generative models provide several important advantages.

They

- reduce the search space,
- accelerate materials discovery,
- lower computational cost,
- identify novel compounds,
- support inverse design,
- integrate naturally with DFT,
- enable autonomous discovery pipelines.

Consequently,

many modern autonomous laboratories combine

- generative AI,
- property prediction,
- uncertainty estimation,
- active learning,
- robotic experimentation

into a closed-loop discovery system.

---

## 24.1.8 Chapter Roadmap

This chapter presents a comprehensive introduction to generative artificial intelligence for materials discovery.

We begin with the mathematical foundations of generative modeling before introducing progressively more advanced models, including Variational Autoencoders, Graph Variational Autoencoders, Generative Adversarial Networks, Graph GANs, Diffusion Models, and Crystal Diffusion Models.

Throughout the chapter, every major concept is accompanied by complete Python implementations using PyTorch and PyTorch Geometric. By the end of this chapter, the reader will understand how modern generative models learn crystal representations, generate entirely new candidate materials, and integrate seamlessly with property prediction models to create complete AI-driven materials discovery pipelines.


## 24.2 Motivation

The discovery of new materials has always been one of the primary goals of materials science. Every technological revolution—from the Bronze Age and the Iron Age to modern semiconductors, lithium-ion batteries, and quantum materials—has been driven by the discovery of new materials possessing superior properties.

Historically, discovering these materials required decades of experimental effort. Researchers synthesized thousands of compounds, characterized their structures, measured their properties, and gradually improved candidate materials through trial-and-error experimentation.

Although computational materials science has significantly accelerated this process through Density Functional Theory (DFT), molecular dynamics, and high-throughput simulations, the search space remains overwhelmingly large.

Generative Artificial Intelligence has emerged as a promising solution to this challenge. Rather than exhaustively searching all possible materials, generative models learn how existing materials are organized and then intelligently propose entirely new candidate materials that are likely to possess desirable properties.

---

## 24.2.1 The Immense Size of Chemical Space

The number of possible crystalline materials is extraordinarily large.

A material is determined by numerous variables, including

- elemental composition,
- stoichiometric ratios,
- crystal symmetry,
- lattice parameters,
- atomic coordinates,
- defects,
- dopants,
- pressure,
- temperature.

Even when restricting ourselves to stable inorganic compounds, the number of possible combinations becomes enormous.

For example,

```text
118 Chemical Elements

↓

Millions of Possible Compositions

↓

Billions of Crystal Structures

↓

Astronomical Design Space
```

It is impossible to enumerate and experimentally synthesize every possible material.

---

## 24.2.2 Limitations of Experimental Discovery

Traditional materials discovery is expensive.

A typical workflow involves

```text
Material Synthesis

↓

Crystal Growth

↓

Characterization

↓

Property Measurement

↓

Analysis
```

Each step requires

- specialized equipment,
- trained personnel,
- laboratory resources,
- significant time.

Many synthesized compounds ultimately fail to exhibit useful properties, resulting in considerable wasted effort.

Furthermore,

some materials require

- extremely high temperatures,
- high pressures,
- specialized atmospheres,

making experimental screening even more challenging.

---

## 24.2.3 Limitations of High-Throughput DFT

Density Functional Theory has revolutionized computational materials science.

Instead of synthesizing every candidate,

researchers can compute

- formation energy,
- band gap,
- elastic constants,
- dielectric properties,
- phonons,

before experiments begin.

The workflow becomes

```text
Candidate Material

↓

DFT Simulation

↓

Predicted Properties

↓

Experimental Validation
```

Although much faster than laboratory experiments,

DFT remains computationally expensive.

For many complex materials,

a single calculation may require

- several CPU hours,
- hundreds of CPU hours,
- or even several days on high-performance computing systems.

Screening millions of hypothetical materials therefore becomes impractical.

---

## 24.2.4 Why Exhaustive Search Is Impossible

Suppose we attempt to evaluate every possible material.

```text
10,000,000 Materials

×

2 Hours per DFT Calculation

=

20,000,000 CPU Hours
```

Even using a large supercomputer,

this computation may require months or years.

Moreover,

many generated materials may be physically impossible,

chemically unstable,

or experimentally unsynthesizable.

Consequently,

blind enumeration is highly inefficient.

---

## 24.2.5 Intelligent Search Instead of Exhaustive Search

Generative AI changes the search strategy.

Rather than evaluating every possible material,

the model first learns the statistical distribution of known materials.

```text
Known Crystal Database

↓

Learn Distribution

↓

Generate Promising Candidates

↓

DFT Validation

↓

Experimental Testing
```

Only a small fraction of highly promising materials require expensive simulations.

This dramatically reduces computational cost.

---

## 24.2.6 Accelerating Materials Discovery

Generative models act as intelligent proposal engines.

Instead of producing random crystal structures,

they generate materials that resemble realistic compounds while simultaneously exploring previously unknown regions of chemical space.

The discovery pipeline therefore becomes

```text
Existing Materials

↓

Generative Model

↓

Novel Candidates

↓

Property Prediction

↓

DFT Verification

↓

Laboratory Synthesis
```

This workflow has become increasingly common in modern computational materials science.

---

## 24.2.7 Inverse Materials Design

Traditional materials discovery follows a forward process.

```text
Material

↓

Property Prediction
```

Generative AI enables the reverse process.

```text
Desired Property

↓

Material Generation
```

For example,

instead of asking

```text
What is the band gap of this material?
```

we ask

```text
Generate a material with

Band Gap = 1.5 eV
```

Similarly,

the objective may become

```text
Generate

↓

High Ionic Conductivity

Low Formation Energy

Large Stability

High Capacity
```

This approach is known as **inverse materials design**, one of the primary motivations behind generative artificial intelligence.

---

## 24.2.8 Industrial Importance

Industries increasingly rely on accelerated materials discovery.

Applications include

- lithium-ion batteries,
- sodium-ion batteries,
- hydrogen storage,
- solar cells,
- fuel cells,
- catalysts,
- aerospace alloys,
- biomedical implants,
- thermoelectric materials,
- semiconductor devices.

Reducing discovery time from

```text
Years

↓

Months
```

or even

```text
Months

↓

Weeks
```

can provide enormous economic advantages.

---

## 24.2.9 AI-Driven Materials Discovery

Modern AI pipelines combine multiple machine learning components.

```text
Materials Database

↓

Representation Learning

↓

Generative Model

↓

Property Prediction

↓

Uncertainty Estimation

↓

Active Learning

↓

DFT

↓

Experiment
```

Rather than replacing scientists,

these models function as intelligent assistants that rapidly narrow the search space and recommend the most promising candidates for further investigation.

---

## 24.2.10 Transition to Generative Models

Having established the motivation for generative artificial intelligence, the next step is to understand the mathematical foundations of generative modeling.

Unlike predictive machine learning models that estimate material properties, generative models attempt to learn the probability distribution of crystal structures themselves. Once this distribution has been learned, entirely new materials can be generated by sampling from the learned latent space.

In the following section, we introduce the fundamental principles of **generative models**, providing the theoretical foundation upon which Variational Autoencoders, Generative Adversarial Networks, Diffusion Models, and modern crystal generation frameworks are built.

## 24.3 Forward Design vs Inverse Design

For decades, materials science has primarily relied on **forward design**, where researchers first propose or synthesize a material and then determine its physical or chemical properties through experiments or computational simulations.

Modern artificial intelligence introduces a fundamentally different philosophy known as **inverse design**. Instead of beginning with a material, inverse design begins with the desired properties and attempts to generate materials that satisfy those requirements.

This shift represents one of the most significant conceptual changes in computational materials science and forms the foundation of generative AI.

---

## 24.3.1 Forward Design

Forward design follows a straightforward sequence.

A researcher first selects a material based on prior knowledge or intuition.

The material is then synthesized or simulated, and its properties are measured.

The workflow can be summarized as

```text
Material

↓

Simulation / Experiment

↓

Properties
```

For example,

```text
LiFePO₄

↓

DFT

↓

Band Gap
```

or

```text
Crystal Structure

↓

Graph Neural Network

↓

Formation Energy
```

The material is already known.

The objective is simply to predict its behavior.

---

## 24.3.2 Characteristics of Forward Design

Forward design possesses several advantages.

It

- is conceptually simple,
- is compatible with existing simulation techniques,
- allows accurate property prediction,
- integrates naturally with machine learning regression models.

However,

its major limitation is that researchers must already possess a candidate material before prediction begins.

If the chosen material performs poorly,

the entire process must be repeated using another candidate.

---

## 24.3.3 Limitations of Forward Design

Suppose a researcher wishes to discover a battery cathode with

- high energy density,
- low cost,
- excellent thermal stability,
- long cycle life.

Using forward design,

the workflow becomes

```text
Candidate 1

↓

Evaluate

↓

Reject
```

```text
Candidate 2

↓

Evaluate

↓

Reject
```

```text
Candidate 3

↓

Evaluate

↓

Accept
```

Thousands of unsuccessful evaluations may occur before a satisfactory material is discovered.

This trial-and-error strategy is both computationally and experimentally expensive.

---

## 24.3.4 Inverse Design

Inverse design reverses the discovery process.

Instead of asking

```text
What properties does this material possess?
```

we ask

```text
Which material possesses these properties?
```

The workflow becomes

```text
Desired Properties

↓

Generative AI

↓

Candidate Materials
```

For example,

```text
Band Gap = 1.4 eV

High Stability

Low Density

↓

Generate Crystal
```

Instead of searching randomly,

the model directly proposes materials likely to satisfy the design objectives.

---

## 24.3.5 Comparison

The philosophical difference between the two approaches is illustrated below.

### Forward Design

```text
Material

↓

Property Prediction
```

### Inverse Design

```text
Desired Properties

↓

Material Generation
```

Forward design answers

> **What does this material do?**

Inverse design answers

> **Which material should I build?**

---

## 24.3.6 Role of Machine Learning

Predictive machine learning models naturally support forward design.

For example,

```text
Crystal

↓

CGCNN

↓

Formation Energy
```

or

```text
Crystal

↓

M3GNet

↓

Elastic Modulus
```

Generative AI instead learns

```text
Desired Property

↓

Latent Space Search

↓

Generated Crystal
```

The neural network therefore acts as a material designer rather than merely a property predictor.

---

## 24.3.7 Optimization in Inverse Design

Inverse design is fundamentally an optimization problem.

Suppose we seek a material satisfying

```text
High Band Gap

High Stability

Low Density
```

These objectives may compete with one another.

The generative model therefore searches for crystal structures that simultaneously optimize multiple properties.

Conceptually,

```text
Design Objectives

↓

Optimization

↓

Best Crystal Candidates
```

This process is known as **multi-objective optimization** and will be discussed further in later chapters.

---

## 24.3.8 AI-Assisted Inverse Design Workflow

Modern AI-driven materials discovery typically combines several machine learning components.

```text
Desired Properties

↓

Generative Model

↓

Candidate Crystal

↓

Property Predictor

↓

Uncertainty Estimation

↓

DFT

↓

Experiment
```

Each component contributes to reducing the number of expensive simulations required.

---

## 24.3.9 Example

Suppose researchers wish to discover a new photovoltaic material.

Instead of evaluating millions of crystals,

they specify

```text
Band Gap

≈

1.4 eV

High Stability

Low Toxicity
```

The generative model produces

```text
Crystal A

Crystal B

Crystal C

Crystal D
```

A property prediction model estimates the electronic properties of each candidate.

Only the most promising structures proceed to DFT calculations and experimental synthesis.

This dramatically accelerates the discovery process.

---

## 24.3.10 Why Inverse Design Requires Generative Models

Predictive models alone cannot perform inverse design.

A regression model can estimate

```text
Crystal

↓

Band Gap
```

but it cannot answer

```text
Band Gap

↓

Crystal
```

Generating entirely new materials requires models capable of learning the probability distribution of crystal structures.

These are known as **generative models**.

The remainder of this chapter introduces the major classes of generative models used in materials informatics, beginning with the mathematical foundations of probabilistic generation before progressing to Variational Autoencoders, Generative Adversarial Networks, Diffusion Models, and modern crystal generation frameworks.

---

## 24.3.11 Code Implementation — Forward vs Inverse Design

The following example illustrates the conceptual difference between predictive and generative workflows.

```python
# -------------------------------
# Forward Design
# -------------------------------

crystal = load_crystal("LiFePO4.cif")

band_gap = property_predictor(crystal)

print(f"Predicted Band Gap: {band_gap:.2f} eV")
```

Output

```text
Predicted Band Gap: 3.82 eV
```

Here, the material is already known, and the model predicts one of its properties.

---

The inverse design workflow is fundamentally different.

```python
# -------------------------------
# Inverse Design
# -------------------------------

target_properties = {

    "band_gap": 1.4,

    "formation_energy": -2.5,

    "density": 4.8

}

generated_crystal = generator(

    target_properties

)

print(generated_crystal)
```

Output

```text
Generated Crystal Structure

Space Group : Pnma

Atoms       : 28

Lattice     : Generated

Status      : Candidate Material
```

Unlike the predictive model, the generative model creates a **new candidate crystal** that satisfies the requested design objectives. This capability forms the basis of modern AI-driven inverse materials design and motivates the probabilistic generative models introduced in the following section.

## 24.4 Fundamentals of Generative Models

The previous section introduced the concept of inverse materials design, where the objective is to generate new materials that satisfy desired physical or chemical properties. Achieving this objective requires a fundamentally different class of machine learning algorithms known as **generative models**.

Unlike predictive models, which learn relationships between inputs and outputs, generative models learn the underlying probability distribution of the data itself. Once this distribution has been learned, the model can generate entirely new samples that resemble the training data while still exhibiting novelty.

Generative models therefore serve as the computational engine behind modern AI-driven materials discovery.

---

## 24.4.1 Predictive Models vs Generative Models

Before studying generative models, it is important to distinguish them from predictive (discriminative) models.

Suppose we have a crystal structure

```text
LiFePO₄
```

A predictive model attempts to answer

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

Mathematically, the model learns

```text
P(y | x)
```

where

- **x** represents the crystal,
- **y** represents the target property.

The objective is property prediction.

---

A generative model solves a completely different problem.

Instead of predicting properties,

it learns

```text
P(x)
```

where

**x** is the crystal structure itself.

After learning this probability distribution,

the model can generate

```text
New Crystal

↓

Predicted Crystal Structure
```

rather than

```text
Existing Crystal

↓

Predicted Property
```

---

## 24.4.2 What Does "Learning a Distribution" Mean?

One of the most common misunderstandings about generative AI is the phrase

> "The model learns the data distribution."

To understand this idea, consider a very simple dataset.

Suppose our training dataset contains only circles.

```text
○

○

○

○

○
```

A predictive model may classify these circles or estimate one of their properties.

A generative model instead asks

> What characteristics define a circle?

After training,

it learns

- average radius,
- variation,
- position,
- overall shape.

Once these patterns are learned,

the model can generate

```text
○

○

○
```

that were **never present in the training data** but still resemble realistic circles.

The same principle applies to materials.

---

## 24.4.3 Learning the Distribution of Crystal Structures

Instead of circles,

our dataset contains crystals.

```text
Crystal Database

↓

Graph Neural Network

↓

Statistical Patterns
```

During training,

the model gradually learns

- preferred bond lengths,
- coordination environments,
- lattice symmetry,
- atomic connectivity,
- common crystal motifs,
- chemical stability.

After training,

the neural network possesses an internal statistical model describing realistic crystal structures.

Sampling from this model produces entirely new candidate materials.

---

## 24.4.4 Probability Distribution of Materials

Suppose every crystal is represented by

```text
x
```

The training dataset contains

```text
x₁

x₂

x₃

...

xₙ
```

Generative learning attempts to approximate

```text
P(x)
```

This probability distribution describes

> How likely is a crystal to exist?

Realistic materials correspond to

```text
High Probability
```

Impossible structures correspond to

```text
Very Low Probability
```

The objective of the generative model is therefore to sample primarily from regions of high probability.

---

## 24.4.5 Sampling

Once the distribution has been learned,

the model performs

```text
Sampling
```

Sampling simply means generating new examples from the learned distribution.

Conceptually,

```text
Learn Distribution

↓

Sample

↓

New Material
```

Every generated sample is

- realistic,
- chemically meaningful,
- statistically consistent,

while still being different from every training example.

---

## 24.4.6 Latent Variable Models

Directly generating crystal structures is extremely difficult because crystal graphs are highly complex.

Instead,

most modern generative models introduce a hidden variable

```text
z
```

called the **latent variable**.

The workflow becomes

```text
Crystal

↓

Encoder

↓

Latent Variable z

↓

Decoder

↓

Generated Crystal
```

Rather than generating crystals directly,

the model first generates

```text
z
```

then reconstructs a crystal from this latent representation.

This idea underlies

- Variational Autoencoders,
- Graph VAEs,
- Diffusion Models,
- Foundation Generative Models.

---

## 24.4.7 Why Latent Spaces Are Useful

Latent spaces possess several desirable properties.

Nearby points often represent

- similar chemistry,
- similar crystal structures,
- similar bonding,
- similar properties.

For example,

```text
Latent Space

● LiFePO₄

● LiMnPO₄

● LiCoPO₄
```

Because these compounds are chemically related,

their embeddings naturally cluster together.

Sampling between them may generate entirely new phosphate materials.

---

## 24.4.8 Maximum Likelihood Learning

Most generative models attempt to maximize the probability of observing the training data.

Conceptually,

```text
Observed Materials

↓

Increase Probability

↓

Better Model
```

The model gradually adjusts its parameters until the generated distribution closely resembles the true distribution of materials.

Although different generative models optimize different objective functions, nearly all are based on this general principle.

---

## 24.4.9 Major Classes of Generative Models

Modern materials informatics primarily uses four families of generative models.

### Autoencoder-Based Models

```text
Crystal

↓

Encoder

↓

Latent Space

↓

Decoder

↓

Crystal
```

Examples

- Autoencoders
- Variational Autoencoders
- Graph VAEs

---

### Adversarial Models

```text
Generator

↓

Fake Crystal

↓

Discriminator

↓

Real or Fake
```

Examples

- GAN
- Graph GAN

---

### Diffusion Models

```text
Crystal

↓

Add Noise

↓

Learn Reverse Process

↓

Remove Noise

↓

Generated Crystal
```

Examples

- DDPM
- Crystal Diffusion
- DiffCSP

---

### Autoregressive Models

```text
Generate

↓

Atom 1

↓

Atom 2

↓

Atom 3

↓

...

↓

Complete Crystal
```

Although less common than diffusion models in current materials research, autoregressive methods remain an active research area.

---

## 24.4.10 Why Generative Models Are Difficult

Generating crystal structures is considerably more difficult than predicting material properties.

A successful generative model must simultaneously satisfy

- chemical validity,
- charge neutrality,
- physically reasonable bond lengths,
- crystal symmetry,
- lattice consistency,
- structural stability.

Failure to satisfy any of these constraints may produce physically impossible materials.

Consequently, modern generative models often incorporate crystallographic knowledge, graph representations, symmetry constraints, and physics-based loss functions to guide generation.

---

## 24.4.11 Transition to Autoencoders

The next logical question is

> **How can a neural network learn a latent representation from which new materials can be generated?**

The simplest answer is provided by the **Autoencoder**.

An autoencoder learns to compress a crystal into a compact latent representation and then reconstruct the original crystal from that representation. Although originally developed for dimensionality reduction, autoencoders provide the conceptual foundation for Variational Autoencoders, Graph VAEs, and many modern generative models used in materials discovery.

In the next section, we begin by studying the architecture, mathematics, and implementation of **Autoencoders** before progressing to more powerful probabilistic generative models.

---

## 24.4.12 Code Implementation — Sampling from a Latent Distribution

The following example demonstrates the basic idea of sampling from a latent space. Instead of generating crystals directly, we first sample latent vectors from a standard normal distribution.

```python
import torch
import matplotlib.pyplot as plt

# Generate 1,000 latent vectors of dimension 2
z = torch.randn(1000, 2)

print(z.shape)
```

Output

```text
torch.Size([1000, 2])
```

Each row represents one point in the latent space.

---

We can visualize these latent samples.

```python
plt.figure(figsize=(6,6))

plt.scatter(

    z[:,0],

    z[:,1],

    s=10,

    alpha=0.6

)

plt.xlabel("Latent Dimension 1")

plt.ylabel("Latent Dimension 2")

plt.title("Samples from Latent Space")

plt.grid(True)

plt.show()
```

The points approximately follow a two-dimensional Gaussian distribution. In later sections, these latent vectors will be transformed by a decoder into complete crystal structures, illustrating the central idea behind modern generative models.

## 24.5 Autoencoders

Autoencoders are among the earliest deep learning models developed for **representation learning**. Although they were originally designed for dimensionality reduction and data compression, they later became the conceptual foundation of many modern generative models, including Variational Autoencoders (VAEs), Graph Variational Autoencoders, and numerous latent-variable generative frameworks.

In materials informatics, an autoencoder learns how to compress a crystal structure into a compact numerical representation and then reconstruct the original crystal from this compressed representation. During this process, the neural network discovers meaningful structural patterns, chemical relationships, and latent representations without requiring property labels.

Although a standard autoencoder is **not** a true generative model, understanding its architecture is essential before studying probabilistic models such as VAEs.

---

## 24.5.1 Motivation

Suppose each crystal is described using

```text
512 Features
```

These features may include

- elemental descriptors,
- crystal graph embeddings,
- local atomic environments,
- structural fingerprints.

Many of these features are redundant.

Instead of storing

```text
512 Numbers
```

the neural network learns a compressed representation

```text
512

↓

64

↓

512
```

If the original crystal can be accurately reconstructed,

the 64-dimensional representation must contain the essential information.

This compressed representation is called the **latent representation** or **latent code**.

---

## 24.5.2 Autoencoder Architecture

A standard autoencoder consists of two neural networks.

### Encoder

Compresses the input.

```text
Input Crystal

↓

Encoder

↓

Latent Vector
```

---

### Decoder

Reconstructs the input.

```text
Latent Vector

↓

Decoder

↓

Reconstructed Crystal
```

Together,

the complete architecture becomes

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

The neural network learns by minimizing the difference between the input and the reconstructed output.

---

## 24.5.3 The Bottleneck

The most important component of an autoencoder is the **bottleneck layer**.

Example

```text
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

512
```

The narrow layer

```text
32
```

forces the network to compress the information.

Because the network cannot simply memorize every input feature,

it must learn meaningful representations.

---

## 24.5.4 Learning Useful Representations

Suppose two crystal structures are chemically similar.

```text
LiFePO₄

↓

Encoder

↓

Latent Vector A
```

```text
LiMnPO₄

↓

Encoder

↓

Latent Vector B
```

Because both compounds possess similar crystal structures,

their latent vectors often become nearby points.

Thus,

the encoder naturally organizes similar materials together.

---

## 24.5.5 Reconstruction

After encoding,

the decoder reconstructs the original crystal representation.

```text
Crystal

↓

Encoder

↓

Compressed Representation

↓

Decoder

↓

Reconstructed Crystal
```

The closer the reconstruction is to the original,

the better the learned representation.

---

## 24.5.6 Reconstruction Loss

Training is driven by the reconstruction error.

The objective is

```text
Input

≈

Output
```

For continuous features,

the Mean Squared Error (MSE) is commonly used.

```text
Original Features

↓

Compare

↓

Reconstructed Features

↓

Loss
```

The neural network minimizes this reconstruction loss during training.

---

## 24.5.7 Why Autoencoders Learn Useful Features

One may ask

> Why doesn't the network simply memorize every crystal?

The answer lies in the bottleneck.

Because information must pass through a much smaller latent representation,

the network is forced to discover

- structural regularities,
- atomic patterns,
- bonding relationships,
- chemical similarities.

Consequently,

the latent representation often becomes significantly more informative than the original handcrafted descriptors.

---

## 24.5.8 Applications in Materials Science

Although standard autoencoders are rarely used directly for crystal generation,

they remain useful for

- feature compression,
- dimensionality reduction,
- representation learning,
- anomaly detection,
- visualization,
- denoising,
- preprocessing large datasets.

More importantly,

they provide the conceptual basis for Variational Autoencoders, which will be introduced in the next section.

---

## 24.5.9 Autoencoder Workflow

The complete workflow is

```text
Crystal Descriptor

↓

Encoder

↓

Latent Representation

↓

Decoder

↓

Reconstructed Descriptor
```

After training,

only the encoder is often retained.

```text
Crystal

↓

Encoder

↓

Latent Embedding
```

These embeddings can then be used for

- clustering,
- visualization,
- similarity search,
- transfer learning.

---

## 24.5.10 Limitations

Despite their usefulness,

standard autoencoders possess important limitations.

They

- do not learn probabilistic latent spaces,
- cannot easily generate new materials,
- often memorize training data,
- provide no principled sampling mechanism.

Consequently,

sampling random latent vectors usually produces meaningless outputs.

This limitation motivates the development of **Variational Autoencoders**, which transform the latent space into a continuous probability distribution suitable for generation.

---

## 24.5.11 Code Implementation — Building a Simple Autoencoder

The following implementation demonstrates a fully connected autoencoder for crystal feature vectors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Linear(512,256),

            nn.ReLU(),

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Linear(128,64)

        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(64,128),

            nn.ReLU(),

            nn.Linear(128,256),

            nn.ReLU(),

            nn.Linear(256,512)

        )

    def forward(self,x):

        z = self.encoder(x)

        reconstruction = self.decoder(z)

        return reconstruction
```

---

## 24.5.12 Creating the Model

```python
model = AutoEncoder()

print(model)
```

The network compresses

```text
512

↓

64

↓

512
```

through the bottleneck.

---

## 24.5.13 Loss Function

The reconstruction loss is computed using Mean Squared Error.

```python
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3

)
```

---

## 24.5.14 Training Loop

```python
epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()

    reconstruction = model(features)

    loss = criterion(

        reconstruction,

        features

    )

    loss.backward()

    optimizer.step()

    print(

        f"Epoch {epoch+1}: {loss.item():.6f}"

    )
```

As training progresses,

the reconstruction loss should decrease.

---

## 24.5.15 Extracting Latent Embeddings

After training,

only the encoder is needed.

```python
with torch.no_grad():

    latent_vectors = model.encoder(

        features

    )

print(

    latent_vectors.shape

)
```

Example output

```text
torch.Size([5000,64])
```

These 64-dimensional embeddings summarize the essential structural information contained in the original 512-dimensional descriptors.

---

## 24.5.16 Visualizing the Latent Space

The learned latent vectors can be visualized using PCA.

```python
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

latent = latent_vectors.numpy()

pca = PCA(

    n_components=2

)

projection = pca.fit_transform(

    latent

)

plt.figure(figsize=(8,6))

plt.scatter(

    projection[:,0],

    projection[:,1],

    s=12,

    alpha=0.7

)

plt.title(

    "Latent Space Learned by Autoencoder"

)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.grid(True)

plt.show()
```

Clusters emerging in this latent space often correspond to chemically or structurally related materials.

---

## 24.5.17 Transition to Variational Autoencoders

Although autoencoders successfully learn compressed crystal representations, they cannot reliably generate new materials because the latent space lacks a well-defined probabilistic structure. Random sampling generally produces invalid latent vectors that decode into unrealistic outputs.

To overcome this limitation, **Variational Autoencoders (VAEs)** impose a probability distribution on the latent space, allowing meaningful sampling and interpolation. This seemingly simple modification transforms the autoencoder from a representation learning model into a true generative model capable of creating entirely new crystal candidates.

## 24.6 Variational Autoencoders (VAEs)

The standard autoencoder introduced in the previous section is highly effective for learning compressed representations of crystal structures. However, despite its ability to reconstruct training data accurately, it possesses one fundamental weakness:

> **A standard autoencoder cannot reliably generate new materials.**

This limitation is precisely why **Variational Autoencoders (VAEs)** were developed.

Variational Autoencoders transform the deterministic latent space of an ordinary autoencoder into a **continuous probabilistic latent space** from which entirely new samples can be generated. Instead of merely compressing existing materials, a VAE learns how crystal structures are statistically distributed and then samples new structures from this learned distribution.

For this reason, VAEs represent one of the earliest successful deep generative models and remain an important foundation for modern crystal generation techniques.

---

## 24.6.1 Why Standard Autoencoders Cannot Generate New Materials

Consider the latent space learned by an ordinary autoencoder.

```text
Crystal

↓

Encoder

↓

Latent Vector

↓

Decoder

↓

Reconstructed Crystal
```

During training,

the encoder learns a mapping

```text
Crystal A

↓

(2.15, -1.82, 0.91, ...)
```

```text
Crystal B

↓

(-0.74, 0.36, 2.84, ...)
```

```text
Crystal C

↓

(5.92, -7.41, 1.26, ...)
```

Notice that the latent vectors may occupy completely arbitrary locations.

There is **no guarantee** that nearby points correspond to meaningful materials.

Even worse,

large regions of latent space contain no training examples.

For example,

```text
Latent Space

●

        ●


                    ●



      ×


×

                 ×
```

The circles represent latent vectors learned during training.

The crosses represent random points.

If we sample one of these random points,

```text
Random Latent Vector

↓

Decoder

↓

?
```

the decoder has never encountered such an input before.

Consequently,

the reconstructed crystal is usually meaningless.

---

## 24.6.2 The Missing Property

The latent space of a standard autoencoder lacks an important property:

> **Continuity**

Suppose two materials are chemically similar.

```text
LiFePO₄

↓

Latent Vector A
```

```text
LiMnPO₄

↓

Latent Vector B
```

Ideally,

every point between A and B should also correspond to a physically meaningful crystal.

Unfortunately,

ordinary autoencoders provide no such guarantee.

The latent space often contains

- holes,
- discontinuities,
- isolated clusters,
- empty regions.

Random sampling therefore becomes unreliable.

---

## 24.6.3 Desired Properties of a Generative Latent Space

An ideal latent space should satisfy several important properties.

### Smoothness

Nearby points should correspond to similar materials.

```text
Material A

↓

•

↓

•

↓

•

↓

Material B
```

Every intermediate point should decode into a realistic crystal.

---

### Continuity

There should not be large empty regions.

```text
Good

•

•

•

•

•

•

```

rather than

```text
Bad

•




•






•
```

---

### Sampleability

Any randomly selected point should decode into a realistic material.

```text
Random Point

↓

Decoder

↓

Valid Crystal
```

This is impossible with ordinary autoencoders.

---

## 24.6.4 The Key Idea Behind Variational Autoencoders

Instead of assigning each crystal a single fixed latent vector,

VAEs assign

> **a probability distribution.**

Instead of

```text
Crystal

↓

Latent Vector
```

the encoder produces

```text
Crystal

↓

Probability Distribution
```

For example,

instead of

```text
(2.31, -0.64)
```

the encoder predicts

```text
Mean

=

(2.31, -0.64)

Variance

=

(0.18, 0.27)
```

Rather than representing a crystal by one point,

it is represented by an entire Gaussian distribution.

---

## 24.6.5 From Points to Distributions

The difference can be illustrated visually.

### Ordinary Autoencoder

```text
Crystal

↓

•

```

One point.

---

### Variational Autoencoder

```text
Crystal

↓

      *****
   ***********
 ***************
*****************
 ***************
   ***********
      *****
```

A probability distribution.

This seemingly small modification changes everything.

Instead of memorizing isolated latent vectors,

the model learns a **continuous probability landscape**.

---

## 24.6.6 Gaussian Latent Variables

Most VAEs assume that latent variables follow a multivariate Gaussian distribution.

Instead of predicting

```text
z
```

the encoder predicts

```text
μ

and

σ
```

where

- μ is the mean,
- σ is the standard deviation.

The latent variable is then sampled from

```text
z ~ N(μ,σ²)
```

Rather than encoding one deterministic representation,

the encoder learns an entire family of possible representations.

---

## 24.6.7 Encoder Output

Unlike ordinary autoencoders,

the encoder no longer produces a single vector.

Instead,

it produces

```text
Crystal

↓

Encoder

↓

Mean Vector

↓

μ
```

and simultaneously

```text
Crystal

↓

Encoder

↓

Standard Deviation

↓

σ
```

These two vectors completely define the latent probability distribution.

---

## 24.6.8 Sampling from the Distribution

Once μ and σ have been predicted,

a latent vector is sampled.

```text
μ

σ

↓

Sampling

↓

z

↓

Decoder
```

Every forward pass therefore produces a slightly different latent representation.

Consequently,

the decoder becomes robust to small perturbations and learns a smooth latent space.

---

## 24.6.9 Why Random Sampling Now Works

Because every crystal contributes a probability distribution,

the latent space becomes densely populated.

```text
**************

*******************

**********************

*******************

**************
```

Instead of isolated points,

the distributions overlap.

Randomly selecting

```text
z
```

from this space is therefore much more likely to produce a meaningful crystal.

This property makes VAEs true generative models.

---

## 24.6.10 Overall Architecture

The architecture of a Variational Autoencoder differs slightly from a conventional autoencoder.

```text
Crystal

↓

Encoder

↓

μ

σ

↓

Sampling

↓

Latent Variable

↓

Decoder

↓

Generated Crystal
```

The additional sampling step is the defining feature of the VAE.

Although simple,

it transforms the network from a reconstruction model into a probabilistic generative model.

---

## 24.6.11 Why VAEs Are Important for Materials Science

Suppose our training database contains

```text
LiFePO₄

LiMnPO₄

LiCoPO₄

LiNiPO₄
```

A VAE may learn a continuous latent representation connecting these compounds.

Sampling intermediate latent vectors could generate

previously unknown phosphate materials that have never been experimentally synthesized.

This capability forms the basis of AI-driven inverse materials design.

Instead of searching billions of crystal structures,

the model intelligently explores a smooth latent space containing chemically meaningful candidates.

In the following sections, we will derive the mathematical foundations of Variational Autoencoders, beginning with latent probability distributions, Gaussian assumptions, and the reparameterization trick that enables efficient neural network training.

## 24.6.12 Latent Probability Distributions

In the previous section, we saw that a Variational Autoencoder does not represent a crystal by a single latent vector. Instead, each crystal is represented by a **probability distribution**.

This seemingly simple modification is the defining characteristic of the VAE and is responsible for its ability to generate new materials.

To understand why this works, we must first understand the role of probability distributions in machine learning.

---

## 24.6.12.1 Why Introduce Probability?

Consider two crystal structures.

```text
LiFePO₄

LiMnPO₄
```

Although these materials are not identical, they share many similarities.

Both possess

- similar crystal symmetry,
- similar atomic arrangements,
- similar coordination environments,
- similar phosphate frameworks.

Because of these similarities, we would like their latent representations to also be similar.

Instead of assigning

```text
LiFePO₄

↓

(2.13, -1.84)
```

and

```text
LiMnPO₄

↓

(8.41, 6.92)
```

which are far apart,

we prefer

```text
LiFePO₄

↓

Around (2,1)
```

```text
LiMnPO₄

↓

Around (2.3,1.2)
```

The keyword is

> **Around**

rather than

> **Exactly**

This naturally leads to probability distributions.

---

## 24.6.12.2 Deterministic Representation

A standard autoencoder produces

```text
Crystal

↓

Encoder

↓

Latent Vector
```

Mathematically,

```text
x

↓

z
```

Every input always produces exactly the same output.

If

```text
LiFePO₄

↓

(1.94,0.51)
```

then every forward pass returns

```text
(1.94,0.51)
```

No uncertainty exists.

---

## 24.6.12.3 Probabilistic Representation

A Variational Autoencoder instead produces

```text
Crystal

↓

Encoder

↓

Distribution
```

Rather than predicting

```text
z
```

it predicts

```text
μ

σ
```

which define

```text
z

~

N(μ,σ²)
```

Now,

every forward pass produces a slightly different sample.

For example,

the encoder may predict

```text
μ

=

(2.0,1.5)
```

```text
σ

=

(0.2,0.3)
```

One forward pass might generate

```text
(2.03,1.47)
```

Another

```text
(1.91,1.74)
```

Another

```text
(2.18,1.56)
```

Every sample belongs to the same probability distribution.

---

## 24.6.12.4 Visual Interpretation

Instead of one point,

a crystal occupies a small region.

Ordinary Autoencoder

```text
        •

```

Variational Autoencoder

```text
      *****

   ***********

 ***************

*****************

 ***************

   ***********

      *****
```

The center corresponds to

```text
μ
```

while the spread is controlled by

```text
σ
```

---

## 24.6.12.5 Mean Vector

The mean vector determines the center of the latent distribution.

Suppose

```text
μ

=

(2.1,-0.8)
```

This means

the probability distribution is centered around

```text
(2.1,-0.8)
```

The decoder therefore expects samples to lie near this location.

---

## 24.6.12.6 Standard Deviation

The standard deviation controls the spread.

Small

```text
σ
```

produces

```text
•

```

A narrow distribution.

Large

```text
σ
```

produces

```text
*************

*******************

***********************
```

A wide distribution.

The network learns how uncertain each latent variable should be.

---

## 24.6.12.7 Gaussian Distribution

Nearly all Variational Autoencoders assume a Gaussian latent space.

A Gaussian distribution is chosen because

- it is mathematically convenient,
- it is continuous,
- it is differentiable,
- sampling is straightforward,
- optimization becomes stable.

For one latent variable,

the distribution resembles

```text
          *

        *****

      *********

    *************

  *****************

*********************

  *****************

    *************

      *********

        *****

          *
```

The peak occurs at

```text
μ
```

while the width is determined by

```text
σ
```

---

## 24.6.12.8 Multidimensional Gaussian

Real VAEs rarely use only one latent variable.

Instead,

the latent vector may contain

```text
64

128

256

512
```

dimensions.

Each latent variable possesses its own

```text
μ

σ
```

For example,

```text
μ

=

[

1.52,

-0.84,

0.37,

...

]

σ

=

[

0.24,

0.18,

0.31,

...

]
```

The complete latent distribution is therefore a multivariate Gaussian.

---

## 24.6.12.9 Why Gaussian?

Suppose we randomly sample

```text
z

~

N(0,1)
```

The sampled values naturally cluster near zero.

Very large values become increasingly unlikely.

This behavior produces

- smooth latent spaces,
- continuous interpolation,
- stable optimization,
- meaningful sampling.

Consequently,

new materials generated from nearby latent vectors tend to possess similar crystal structures.

---

## 24.6.12.10 Materials Science Interpretation

Suppose a latent dimension encodes

```text
Average Bond Length
```

Instead of storing

```text
2.03 Å
```

the model stores

```text
2.03 Å

±

0.08 Å
```

Similarly,

another latent dimension may encode

```text
Octahedral Distortion

±

Small Variation
```

Another

```text
Lattice Symmetry

±

Small Variation
```

Rather than memorizing one exact crystal,

the model learns a family of chemically similar crystals.

This flexibility enables the generation of previously unseen materials.

---

## 24.6.12.11 Transition to Sampling

Representing crystals as Gaussian distributions raises an important question.

Suppose the encoder predicts

```text
μ

σ
```

How do we actually generate

```text
z
```

from these two quantities?

At first glance,

one might simply sample

```text
z

~

N(μ,σ²)
```

Unfortunately,

this sampling operation is not directly differentiable, preventing gradients from flowing through the neural network during training.

The next section introduces one of the most elegant ideas in deep learning—the **Reparameterization Trick**—which makes Variational Autoencoders trainable using standard backpropagation while preserving their probabilistic nature.

## 24.6.13 The Reparameterization Trick

In the previous section, we introduced the central idea behind Variational Autoencoders: instead of representing a crystal by a single latent vector, the encoder predicts a probability distribution parameterized by its mean and standard deviation.

Mathematically,

```text
Encoder

↓

μ

σ
```

The next step appears straightforward.

We simply sample

```text
z

~

N(μ,σ²)
```

and pass the sampled latent vector to the decoder.

Unfortunately, this seemingly simple operation creates one of the biggest challenges in training Variational Autoencoders.

---

## 24.6.13.1 The Problem

Neural networks are trained using **backpropagation**.

Backpropagation computes gradients by following the computational graph from the output back to every trainable parameter.

A simplified training pipeline is

```text
Input

↓

Encoder

↓

Latent Variable

↓

Decoder

↓

Loss

↓

Backpropagation
```

Every operation inside this pipeline must be differentiable.

Unfortunately,

random sampling is **not differentiable**.

---

## 24.6.13.2 Why Random Sampling Breaks Backpropagation

Suppose the encoder predicts

```text
μ = 2.5

σ = 0.4
```

A naïve implementation would be

```python
z = random_sample(

    mean=mu,

    std=sigma

)
```

The problem is that

```text
Random Sampling

↓

Random Number Generator
```

is independent of

```text
μ

σ
```

During backpropagation,

the neural network cannot determine

```text
How changing μ affects z
```

or

```text
How changing σ affects z
```

Consequently,

the gradients become

```text
Undefined
```

and training fails.

---

## 24.6.13.3 Computational Graph Failure

Consider the following computational graph.

```text
Crystal

↓

Encoder

↓

μ

σ

↓

Random Sampling

↓

Decoder

↓

Loss
```

Backpropagation attempts to compute

```text
Loss

↓

Decoder

↓

Sampling

↓

Encoder
```

However,

the sampling operation acts like a barrier.

```text
Loss

↓

Decoder

↓

×

↓

Encoder
```

The gradients cannot pass through the random number generator.

Therefore,

the encoder cannot learn.

---

## 24.6.13.4 The Brilliant Solution

The key insight behind the Variational Autoencoder is surprisingly elegant.

Instead of sampling

```text
z

~

N(μ,σ²)
```

we first sample from a **fixed standard normal distribution**

```text
ε

~

N(0,1)
```

and then transform it.

The latent variable is computed as

```text
z = μ + σ ε
```

This simple equation is called the **Reparameterization Trick**.

Instead of sampling from

```text
N(μ,σ²)
```

we sample from

```text
N(0,1)
```

and move the randomness outside the neural network.

---

## 24.6.13.5 Intuition

Imagine we wish to generate numbers from

```text
Mean = 5

Standard Deviation = 2
```

Instead of directly sampling

```text
5 ± 2
```

we first generate

```text
ε

=

0.31
```

from

```text
N(0,1)
```

Then compute

```text
z

=

5

+

2 × 0.31

=

5.62
```

Another sample

```text
ε

=

−0.82
```

gives

```text
z

=

5

+

2 × (−0.82)

=

3.36
```

Every sample still follows

```text
N(5,2²)
```

but the computation is now differentiable.

---

## 24.6.13.6 Why This Works

Notice that

```text
ε
```

is completely independent of the neural network.

The neural network predicts only

```text
μ

σ
```

The latent variable is then computed using ordinary arithmetic.

```text
μ

↓

+

↓

σ × ε

↓

z
```

Addition

Multiplication

Scaling

are all differentiable operations.

Therefore,

gradients can flow through

```text
z

↓

μ

σ
```

during training.

---

## 24.6.13.7 New Computational Graph

After reparameterization,

the computational graph becomes

```text
Crystal

↓

Encoder

↓

μ

σ

↓

μ + σ ε

↓

Latent Vector

↓

Decoder

↓

Loss
```

Now,

backpropagation proceeds normally.

```text
Loss

↓

Decoder

↓

Latent Vector

↓

μ

σ

↓

Encoder
```

The network can successfully optimize both

```text
μ
```

and

```text
σ
```

using gradient descent.

---

## 24.6.13.8 Mathematical Interpretation

The reparameterization equation is

```text
z = μ + σ ε
```

where

```text
ε ~ N(0,1)
```

This equation has an important interpretation.

```text
μ
```

determines

> the center of the distribution.

```text
σ
```

determines

> the spread.

```text
ε
```

provides

> random variation.

Together,

they reconstruct the desired Gaussian distribution.

---

## 24.6.13.9 Visualization

Without reparameterization,

sampling appears as

```text
μ

σ

↓

Random Generator

↓

z
```

With reparameterization,

the workflow becomes

```text
μ

↓

+

↓

σ × ε

↓

z
```

The random component

```text
ε
```

is external to the neural network.

Everything else remains differentiable.

---

## 24.6.13.10 Materials Science Perspective

Suppose the latent representation describes a crystal.

The encoder predicts

```text
μ

↓

Average Crystal Representation
```

and

```text
σ

↓

Allowed Structural Variation
```

Sampling different

```text
ε
```

produces

```text
Crystal A'

Crystal A''

Crystal A'''
```

These generated crystals are not identical.

Instead,

they represent slightly different but chemically plausible variations of the same underlying material.

This capability is essential for discovering previously unseen compounds.

---

## 24.6.13.11 PyTorch Implementation

The reparameterization trick can be implemented in only a few lines of code.

```python
import torch

def reparameterize(

    mu,

    log_var

):

    std = torch.exp(

        0.5 * log_var

    )

    epsilon = torch.randn_like(

        std

    )

    z = mu + epsilon * std

    return z
```

Notice that

```python
torch.randn_like(std)
```

samples

```text
ε ~ N(0,1)
```

The latent vector is then computed using

```python
z = mu + epsilon * std
```

which remains fully differentiable.

---

## 24.6.13.12 Why Predict log Variance Instead of σ?

Most practical VAE implementations do **not** predict the standard deviation directly.

Instead,

the encoder predicts

```text
log(σ²)
```

commonly written as

```text
log_var
```

The standard deviation is then recovered using

```python
std = torch.exp(

    0.5 * log_var

)
```

This approach offers several numerical advantages.

It

- guarantees positive variance,
- improves optimization stability,
- avoids negative standard deviations,
- reduces numerical overflow during training.

Consequently,

nearly all modern VAE implementations predict

```text
μ

and

log_var
```

rather than

```text
μ

and

σ.
```

---

## 24.6.13.13 Transition to the Decoder

After obtaining the sampled latent vector

```text
z
```

the final step is to reconstruct a material from this compressed probabilistic representation.

The decoder transforms latent vectors back into crystal descriptors, crystal graphs, or atomic structures. Understanding the decoder completes the encoder–latent–decoder architecture of the Variational Autoencoder and prepares us for deriving the complete VAE objective function.

## 24.6.14 The Decoder

In the previous sections, we studied how the encoder transforms a crystal into a probability distribution over the latent space and how the reparameterization trick enables differentiable sampling from that distribution.

The next step is to transform the sampled latent vector back into a meaningful crystal representation.

This task is performed by the **decoder**.

The decoder is responsible for converting a low-dimensional latent representation into a high-dimensional crystal representation that closely resembles the original input.

Mathematically, the decoder approximates the inverse transformation of the encoder.

---

## 24.6.14.1 Role of the Decoder

Recall the complete VAE architecture.

```text
Crystal

↓

Encoder

↓

μ, σ

↓

Sampling

↓

Latent Vector z

↓

Decoder

↓

Reconstructed Crystal
```

The encoder compresses information,

whereas the decoder reconstructs it.

If the encoder answers

> "How can I summarize this crystal?"

the decoder answers

> "How can I reconstruct the crystal from this summary?"

---

## 24.6.14.2 Decoder as a Function

Suppose the latent vector is

```text
z
```

The decoder learns another nonlinear mapping

```text
Decoder

:

z

→

x̂
```

where

```text
x̂
```

denotes the reconstructed crystal representation.

Unlike the encoder,

which reduces dimensionality,

the decoder gradually expands the information.

---

## 24.6.14.3 Information Expansion

Suppose the latent vector contains only

```text
64 Features
```

The decoder gradually reconstructs

```text
64

↓

128

↓

256

↓

512
```

Eventually,

the output has the same dimensionality as the original crystal descriptor.

---

## 24.6.14.4 Intuition

Consider image compression.

A compressed JPEG file contains far fewer bits than the original image.

When opened,

the decoder reconstructs the image from this compressed representation.

A Variational Autoencoder performs a similar task.

```text
Crystal Descriptor

↓

Compressed Representation

↓

Crystal Descriptor
```

The compressed representation contains only the essential structural information required for reconstruction.

---

## 24.6.14.5 Decoder in Materials Informatics

Depending on the application,

the decoder may reconstruct

- composition vectors,
- crystal descriptors,
- graph representations,
- lattice parameters,
- atomic coordinates,
- complete crystal structures.

For simple educational implementations,

the decoder usually reconstructs feature vectors.

More advanced crystal generation models reconstruct

- atomic positions,
- lattice vectors,
- bonding topology,
- periodic crystal graphs.

---

## 24.6.14.6 Example

Suppose the encoder compresses

```text
LiFePO₄
```

into

```text
z

=

[-0.82,

1.41,

0.67,

...]

```

The decoder receives

```text
z
```

and predicts

```text
Crystal Descriptor

↓

Band Structure Features

↓

Local Environments

↓

Reconstructed Crystal
```

Ideally,

the reconstructed crystal should closely resemble the original LiFePO₄ structure.

---

## 24.6.14.7 Decoder Architecture

The decoder is usually another feed-forward neural network.

For example,

```text
64

↓

128

↓

256

↓

512
```

Each hidden layer gradually reconstructs more detailed structural information.

The final layer outputs

```text
512 Features
```

which can be compared directly with the original input.

---

## 24.6.14.8 Reconstruction Quality

The decoder is trained to minimize reconstruction error.

If reconstruction is poor,

important information has been lost during encoding.

If reconstruction is accurate,

the latent representation successfully captures the essential characteristics of the crystal.

The objective is therefore

```text
Input

≈

Reconstruction
```

rather than

```text
Input

=

Reconstruction
```

Small reconstruction errors are expected because the latent representation is compressed.

---

## 24.6.14.9 Decoder During Generation

The decoder plays a second, even more important role.

After training,

we no longer require an encoder.

Instead,

we can simply sample

```text
z

~

N(0,I)
```

and decode it.

```text
Random Latent Vector

↓

Decoder

↓

Generated Material
```

This ability distinguishes Variational Autoencoders from ordinary autoencoders.

The decoder becomes a **material generator** rather than merely a reconstruction network.

---

## 24.6.14.10 Interpolation in Latent Space

One remarkable property of VAEs is smooth interpolation.

Suppose

```text
z₁

↓

LiFePO₄
```

and

```text
z₂

↓

LiMnPO₄
```

Intermediate latent vectors

```text
0.25z₁ + 0.75z₂
```

may decode into

```text
Li(Fe,Mn)PO₄
```

or another chemically meaningful intermediate structure.

This smooth interpolation arises because the latent space is continuous.

---

## 24.6.14.11 Decoder Training

The decoder receives gradients through the reconstruction loss.

Its objective is

```text
Generate

↓

Reconstruction

↓

Compare

↓

Original Crystal

↓

Loss
```

During training,

both encoder and decoder improve simultaneously.

The encoder learns better latent representations,

while the decoder learns how to accurately reconstruct crystals from these representations.

---

## 24.6.14.12 PyTorch Implementation

A simple decoder can be implemented as

```python
class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(64,128),

            nn.ReLU(),

            nn.Linear(128,256),

            nn.ReLU(),

            nn.Linear(256,512)

        )

    def forward(

        self,

        z

    ):

        reconstruction = self.network(

            z

        )

        return reconstruction
```

The decoder accepts a latent vector of dimension

```text
64
```

and reconstructs a feature vector of dimension

```text
512.
```

---

## 24.6.14.13 Complete VAE Architecture in PyTorch

Combining the encoder, reparameterization step, and decoder gives the complete forward pass.

```python
class VariationalAutoEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

    def reparameterize(

        self,

        mu,

        log_var

    ):

        std = torch.exp(

            0.5 * log_var

        )

        epsilon = torch.randn_like(

            std

        )

        return mu + epsilon * std

    def forward(

        self,

        x

    ):

        mu, log_var = self.encoder(

            x

        )

        z = self.reparameterize(

            mu,

            log_var

        )

        reconstruction = self.decoder(

            z

        )

        return reconstruction, mu, log_var
```

Notice that the forward function returns

- reconstructed input,
- latent mean,
- latent log variance.

These quantities are all required for computing the VAE loss.

---

## 24.6.14.14 From Architecture to Learning Objective

The encoder and decoder together define the structure of the Variational Autoencoder. However, architecture alone is insufficient. We still need an objective function that teaches the model

1. how to reconstruct crystal structures accurately, and
2. how to organize the latent space into a smooth Gaussian distribution suitable for generation.

Unlike ordinary autoencoders, VAEs optimize **two objectives simultaneously**. The next section derives this learning objective, beginning with the reconstruction loss before introducing the Kullback–Leibler divergence that regularizes the latent space.

## 24.6.15 Reconstruction Loss

In the previous section, we completed the architecture of the Variational Autoencoder by introducing the encoder, probabilistic latent space, reparameterization trick, and decoder.

The remaining question is

> **How do we train the entire model?**

Like every deep learning algorithm, a Variational Autoencoder requires a **loss function** that measures how well the network performs.

Unlike an ordinary neural network, however, the VAE has **two learning objectives**.

The first objective is identical to that of a conventional autoencoder:

> **The reconstructed crystal should be as similar as possible to the original crystal.**

This objective is quantified by the **reconstruction loss**.

---

## 24.6.15.1 Why Reconstruction Matters

Consider the complete VAE pipeline.

```text
Crystal

↓

Encoder

↓

Latent Distribution

↓

Sampling

↓

Decoder

↓

Reconstructed Crystal
```

Suppose the input crystal is

```text
LiFePO₄
```

After encoding and decoding, we obtain

```text
LiFePO₄'
```

If

```text
LiFePO₄'

≈

LiFePO₄
```

then the latent representation successfully preserves the important structural information.

However, if the decoder produces

```text
NaCl
```

instead,

the learned representation has failed.

Therefore,

the reconstructed crystal should closely resemble the original crystal.

---

## 24.6.15.2 Comparing the Original and Reconstruction

Suppose the original crystal descriptor is

```text
x
```

and the decoder produces

```text
x̂
```

The reconstruction loss measures the difference between these two vectors.

Conceptually,

```text
Original Crystal

↓

Compare

↓

Reconstructed Crystal

↓

Difference

↓

Loss
```

A smaller loss indicates a better reconstruction.

---

## 24.6.15.3 Feature Vector Example

Assume the original crystal descriptor is

```text
Original

[

0.82,

1.45,

0.61,

2.18

]
```

The decoder predicts

```text
Reconstruction

[

0.80,

1.42,

0.67,

2.11

]
```

The two vectors are very similar.

Therefore,

the reconstruction loss is small.

If the decoder instead predicts

```text
[

5.72,

−2.14,

8.41,

1.03

]
```

the reconstruction loss becomes much larger.

---

## 24.6.15.4 Mean Squared Error (MSE)

For continuous descriptors,

the most common reconstruction loss is the **Mean Squared Error (MSE).**

Suppose

```text
Original

↓

x
```

and

```text
Prediction

↓

x̂
```

The reconstruction error is obtained by

1. computing the difference,
2. squaring every element,
3. averaging all squared differences.

Large errors are penalized much more strongly than small errors because of the squaring operation.

---

## 24.6.15.5 Why Squared Error?

Suppose two prediction errors exist.

```text
Error A

=

0.1
```

```text
Error B

=

2.0
```

After squaring,

```text
0.1²

=

0.01
```

```text
2.0²

=

4
```

The larger error receives a dramatically larger penalty.

Consequently,

the neural network focuses primarily on correcting large reconstruction mistakes.

---

## 24.6.15.6 Interpretation in Materials Science

Suppose one feature corresponds to

```text
Average Bond Length
```

Original

```text
2.02 Å
```

Prediction

```text
2.01 Å
```

Difference

```text
0.01 Å
```

Very small reconstruction error.

Now suppose another feature represents

```text
Lattice Constant
```

Original

```text
5.62 Å
```

Prediction

```text
7.84 Å
```

Difference

```text
2.22 Å
```

This large error contributes significantly to the reconstruction loss.

During training,

the decoder gradually learns to reproduce increasingly accurate crystal descriptors.

---

## 24.6.15.7 Binary Cross Entropy

Not every dataset contains continuous values.

Suppose the input consists of binary variables.

Examples include

- atom present or absent,
- occupied or unoccupied sites,
- graph adjacency matrices.

In these situations,

the reconstruction objective becomes a binary classification problem.

Rather than using Mean Squared Error,

Binary Cross Entropy (BCE) is often employed.

Therefore,

the choice of reconstruction loss depends on the nature of the data.

| Data Type | Typical Reconstruction Loss |
|-----------|----------------------------|
| Continuous descriptors | Mean Squared Error (MSE) |
| Binary variables | Binary Cross Entropy (BCE) |
| Multi-class outputs | Cross Entropy Loss |

Most crystal descriptor applications use **MSE**.

---

## 24.6.15.8 Decoder Perspective

The decoder can be viewed as attempting to solve

```text
Latent Vector

↓

Decoder

↓

Original Crystal
```

The reconstruction loss simply measures

> **How successful was the decoder?**

A perfectly reconstructed crystal would produce

```text
Loss

=

0
```

Although perfect reconstruction is rarely achieved,

the objective is to minimize the reconstruction error as much as possible.

---

## 24.6.15.9 Reconstruction Alone Is Not Enough

Suppose we ignore the probabilistic nature of the VAE and optimize only the reconstruction loss.

The network could simply memorize the training dataset.

Its latent space would resemble that of an ordinary autoencoder.

```text
Crystal

↓

Encoder

↓

Arbitrary Latent Space

↓

Decoder

↓

Crystal
```

Although reconstruction would be excellent,

generation would fail.

Random latent vectors would still decode into meaningless outputs.

Therefore,

another objective is required.

The latent space itself must also be regularized.

---

## 24.6.15.10 PyTorch Implementation

Computing the reconstruction loss is straightforward.

```python
criterion = nn.MSELoss()

reconstruction_loss = criterion(

    reconstruction,

    x

)
```

where

```python
x
```

is the original crystal descriptor,

and

```python
reconstruction
```

is the decoder output.

---

## 24.6.15.11 Training Example

```python
reconstruction,

mu,

log_var = model(

    x

)

reconstruction_loss = F.mse_loss(

    reconstruction,

    x

)

print(

    reconstruction_loss.item()

)
```

Typical output

```text
0.00418
```

As training progresses,

this value should gradually decrease.

---

## 24.6.15.12 Why the Reconstruction Loss Is Only Half the Story

Although reconstruction loss teaches the decoder to reproduce the training data accurately, it says **nothing** about the organization of the latent space.

Two crystals that are chemically similar may still be mapped to completely unrelated latent vectors.

Likewise, random latent vectors may still decode into physically meaningless structures.

To transform the latent space into a smooth Gaussian distribution suitable for generation, the Variational Autoencoder introduces a second objective known as the **Kullback–Leibler (KL) Divergence**. Unlike reconstruction loss, which measures how accurately the decoder reproduces individual crystals, the KL divergence measures how closely the learned latent distribution matches a desired probability distribution.

Together, these two objectives form the complete learning objective of the Variational Autoencoder.

## 24.6.16 Kullback–Leibler (KL) Divergence

In the previous section, we introduced the reconstruction loss, whose objective is to ensure that the decoder accurately reconstructs the input crystal from its latent representation.

However, reconstruction loss alone is insufficient for building a useful generative model.

Suppose we train a Variational Autoencoder using only reconstruction loss.

The encoder could simply memorize every crystal by assigning each one an arbitrary latent vector.

```text
Crystal A

↓

(-25.7, 18.1)
```

```text
Crystal B

↓

(103.6, -91.4)
```

```text
Crystal C

↓

(-310.2, 41.7)
```

Although reconstruction would be nearly perfect, the latent space would become highly irregular and impossible to sample from.

To transform this irregular latent space into a smooth and continuous probability distribution, Variational Autoencoders introduce a second loss function called the **Kullback–Leibler (KL) Divergence**.

---

## 24.6.16.1 Why Regularize the Latent Space?

Suppose the encoder is allowed complete freedom.

The resulting latent vectors might appear as

```text
•






•











•
```

Large empty regions separate the encoded materials.

Now imagine sampling a random latent vector.

```text
×

```

This point is located inside an empty region.

The decoder has never observed similar latent vectors during training.

Consequently,

```text
Random Latent Vector

↓

Decoder

↓

Invalid Crystal
```

Generation fails.

---

## 24.6.16.2 The Desired Latent Space

Ideally,

the latent space should resemble a smooth Gaussian distribution.

```text
**************

*******************

**********************

*******************

**************
```

Every region contains meaningful information.

Random sampling now becomes possible.

```text
Random Sample

↓

Decoder

↓

Valid Crystal
```

This is precisely the purpose of the KL divergence.

---

## 24.6.16.3 What Does KL Divergence Measure?

KL divergence measures

> **how different two probability distributions are.**

Suppose

```text
Learned Distribution

↓

q(z|x)
```

and

```text
Target Distribution

↓

N(0,I)
```

The KL divergence measures the distance between these two distributions.

If the distributions are nearly identical,

```text
KL

↓

Small
```

If they differ significantly,

```text
KL

↓

Large
```

The objective is therefore

```text
Minimize KL Divergence
```

---

## 24.6.16.4 Target Distribution

Variational Autoencoders assume that every latent variable should approximately follow

```text
Standard Normal Distribution

N(0,I)
```

where

```text
Mean

=

0
```

and

```text
Variance

=

1
```

Rather than allowing arbitrary latent distributions,

the encoder is encouraged to remain close to this standard Gaussian.

---

## 24.6.16.5 Why Choose a Standard Normal Distribution?

The standard normal distribution offers several advantages.

It is

- continuous,
- mathematically simple,
- easy to sample,
- differentiable,
- symmetric,
- stable during optimization.

More importantly,

sampling becomes trivial.

```python
z = torch.randn(

    batch_size,

    latent_dimension

)
```

Every generated latent vector automatically follows

```text
N(0,I)
```

---

## 24.6.16.6 Visual Interpretation

Without KL regularization,

latent vectors appear scattered.

```text
•






•









•

```

With KL regularization,

they gradually organize into

```text
**************

*******************

**********************

*******************

**************
```

The latent space becomes smooth,

continuous,

and suitable for generation.

---

## 24.6.16.7 Effect on Similar Materials

Suppose the dataset contains

```text
LiFePO₄

LiMnPO₄

LiCoPO₄
```

Without KL divergence,

their latent vectors may be

```text
LiFePO₄

↓

(-32,91)
```

```text
LiMnPO₄

↓

(118,-204)
```

```text
LiCoPO₄

↓

(-271,13)
```

These locations possess no meaningful geometric relationship.

After KL regularization,

the same materials may become

```text
LiFePO₄

↓

(-0.8,0.4)
```

```text
LiMnPO₄

↓

(-0.6,0.5)
```

```text
LiCoPO₄

↓

(-0.5,0.2)
```

Now,

their proximity reflects their chemical similarity.

---

## 24.6.16.8 Latent Space Interpolation

Because the latent space becomes continuous,

interpolation becomes meaningful.

```text
LiFePO₄

↓

•

↓

•

↓

•

↓

LiMnPO₄
```

Intermediate latent vectors often decode into chemically plausible intermediate structures.

This capability is impossible with ordinary autoencoders.

---

## 24.6.16.9 Mathematical Expression

For a Gaussian latent distribution,

the KL divergence has a closed-form solution.

Rather than estimating it numerically,

the value can be computed directly from

```text
μ

and

σ
```

or equivalently

```text
μ

and

log_var.
```

This analytical solution makes Variational Autoencoders computationally efficient.

---

## 24.6.16.10 Intuition Behind the Formula

The KL divergence penalizes two undesirable situations.

### Mean Too Far From Zero

If

```text
μ

=

8
```

the latent distribution lies far from the origin.

The penalty increases.

---

### Variance Too Large

If

```text
σ

=

15
```

the latent distribution becomes extremely wide.

Again,

the penalty increases.

---

### Variance Too Small

If

```text
σ

≈

0
```

the latent distribution collapses into a nearly deterministic point.

This also increases the penalty.

Consequently,

the encoder learns to balance

- meaningful latent representations,
- smooth probabilistic organization.

---

## 24.6.16.11 Reconstruction Loss vs KL Divergence

The two loss terms perform completely different roles.

| Reconstruction Loss | KL Divergence |
|---------------------|---------------|
| Reconstructs the input crystal | Organizes the latent space |
| Improves reconstruction quality | Improves generation quality |
| Trains the decoder | Regularizes the encoder |
| Preserves structural information | Preserves probabilistic continuity |

Both objectives are equally important.

Without reconstruction loss,

the decoder cannot learn.

Without KL divergence,

generation becomes impossible.

---

## 24.6.16.12 PyTorch Implementation

Suppose the encoder predicts

```python
mu

log_var
```

The KL divergence can be computed as

```python
kl_loss = -0.5 * torch.mean(

    1

    + log_var

    - mu.pow(2)

    - log_var.exp()

)
```

This compact equation is one of the defining characteristics of Variational Autoencoders.

---

## 24.6.16.13 Example

```python
reconstruction,

mu,

log_var = model(

    x

)

reconstruction_loss = F.mse_loss(

    reconstruction,

    x

)

kl_loss = -0.5 * torch.mean(

    1

    + log_var

    - mu.pow(2)

    - log_var.exp()

)

print(

    reconstruction_loss.item(),

    kl_loss.item()

)
```

Typical output

```text
Reconstruction Loss : 0.0038

KL Loss             : 0.0126
```

During training,

both losses gradually decrease until the model achieves a balance between accurate reconstruction and a well-organized latent space.

---

## 24.6.16.14 Why KL Divergence Makes Generation Possible

The KL divergence transforms the latent space into a smooth approximation of a standard Gaussian distribution.

As a result,

sampling becomes extremely simple.

```python
z = torch.randn(

    1,

    latent_dimension

)
```

Passing this latent vector through the decoder

```python
generated = decoder(

    z

)
```

produces a completely new sample.

For materials informatics,

this sample may represent

- a new crystal descriptor,
- a new crystal graph,
- or, in more advanced models, an entirely new crystal structure.

This simple sampling procedure is the foundation of generative materials discovery.

---

## 24.6.16.15 Transition to the Complete VAE Loss

The Variational Autoencoder now possesses two independent learning objectives.

The reconstruction loss teaches the decoder to accurately reproduce crystal structures, while the KL divergence organizes the latent space into a smooth Gaussian distribution suitable for generation.

The next section combines these two objectives into a single optimization function known as the **Evidence Lower Bound (ELBO)**, which serves as the complete training objective for Variational Autoencoders.

## 24.6.17 The Evidence Lower Bound (ELBO)

The previous two sections introduced the two fundamental objectives of a Variational Autoencoder.

The first objective ensures that the reconstructed crystal is as similar as possible to the original crystal.

The second objective forces the latent representations to follow a smooth Gaussian distribution.

These objectives appear to compete with one another.

- Perfect reconstruction encourages the network to memorize every training sample.
- KL regularization encourages the latent representations to remain close to a standard normal distribution.

A successful Variational Autoencoder must satisfy **both** objectives simultaneously.

This balance is achieved through a single optimization objective known as the **Evidence Lower Bound (ELBO).**

The ELBO is one of the most important concepts in probabilistic deep learning and serves as the complete training objective for Variational Autoencoders.

---

## 24.6.17.1 Why Do We Need a Combined Objective?

Suppose we optimize **only** the reconstruction loss.

```text
Crystal

↓

Encoder

↓

Decoder

↓

Crystal
```

The network may simply memorize every crystal.

Its latent space becomes

```text
•







•










•
```

Generation becomes impossible.

---

Now suppose we optimize **only** the KL divergence.

The encoder forces every latent vector toward

```text
N(0,I)
```

Unfortunately,

the decoder receives almost identical latent vectors for every crystal.

The reconstructed outputs become nearly identical.

```text
Crystal A

↓

Same Output
```

```text
Crystal B

↓

Same Output
```

```text
Crystal C

↓

Same Output
```

Reconstruction quality collapses.

Neither objective alone is sufficient.

---

## 24.6.17.2 Balancing Two Competing Objectives

The VAE must learn

1. accurate reconstruction,

and simultaneously

2. smooth latent organization.

Conceptually,

```text
Good Reconstruction

+

Smooth Latent Space

↓

Successful Generative Model
```

The ELBO combines these two goals into a single optimization problem.

---

## 24.6.17.3 Components of the ELBO

The ELBO consists of two parts.

### Reconstruction Term

Measures

```text
Original Crystal

↓

Decoder Output

↓

Difference
```

Smaller values indicate better reconstruction.

---

### KL Regularization

Measures

```text
Learned Distribution

↓

Compare

↓

Standard Gaussian
```

Smaller values indicate a smoother latent space.

---

Together,

the objective becomes

```text
ELBO

=

Reconstruction

+

KL Regularization
```

Since neural networks minimize loss,

the optimization seeks

- low reconstruction error,
- low KL divergence.

---

## 24.6.17.4 Interpretation

Imagine a student learning chemistry.

One objective is

```text
Memorize Facts
```

Another objective is

```text
Understand Concepts
```

Memorizing alone produces poor generalization.

Understanding alone without remembering details is also insufficient.

A successful student balances both.

The Variational Autoencoder behaves similarly.

It balances

```text
Reconstruction

and

Generalization
```

through the ELBO.

---

## 24.6.17.5 Visualizing the Trade-Off

Consider two extreme situations.

### Perfect Reconstruction

```text
Reconstruction Loss

↓

Very Small
```

```text
KL Loss

↓

Very Large
```

The model memorizes the training set.

Poor generation.

---

### Perfect Gaussian Distribution

```text
KL Loss

↓

Very Small
```

```text
Reconstruction Loss

↓

Very Large
```

The latent space is beautiful,

but reconstruction is terrible.

---

The optimal solution lies between these extremes.

```text
Moderate Reconstruction Loss

+

Moderate KL Loss

↓

Best Overall Model
```

---

## 24.6.17.6 β-VAE

In many practical applications,

researchers introduce a weighting parameter

```text
β
```

to control the influence of the KL divergence.

The optimization objective becomes

```text
Reconstruction

+

β × KL
```

Different values of

```text
β
```

produce different behaviors.

### Small β

```text
Excellent Reconstruction

Poor Latent Organization
```

---

### Large β

```text
Excellent Latent Organization

Poor Reconstruction
```

---

Choosing an appropriate value of

```text
β
```

is an important hyperparameter optimization problem.

---

## 24.6.17.7 β-VAE in Materials Science

Suppose we are learning crystal representations.

Small

```text
β
```

may produce

```text
Very Accurate Crystal Reconstructions
```

but poor interpolation.

Large

```text
β
```

may produce

```text
Smooth Chemical Space
```

but inaccurate crystal descriptors.

Researchers therefore select

```text
β
```

according to the application.

For

- representation learning,

larger

```text
β
```

is often preferred.

For

- crystal reconstruction,

smaller values may perform better.

---

## 24.6.17.8 Loss Landscape

Training gradually improves both objectives.

Early epochs

```text
High Reconstruction Loss

High KL Loss
```

↓

Middle epochs

```text
Lower Reconstruction

Lower KL
```

↓

Final epochs

```text
Balanced Solution
```

The optimization process seeks a point where both losses are simultaneously minimized.

---

## 24.6.17.9 PyTorch Implementation

The complete VAE loss can be implemented very simply.

```python
reconstruction_loss = F.mse_loss(

    reconstruction,

    x

)

kl_loss = -0.5 * torch.mean(

    1

    + log_var

    - mu.pow(2)

    - log_var.exp()

)

loss = reconstruction_loss + kl_loss
```

The optimizer minimizes

```python
loss
```

rather than the individual terms.

---

## 24.6.17.10 β-VAE Implementation

Introducing the weighting factor requires only one additional parameter.

```python
beta = 0.1

loss = reconstruction_loss + beta * kl_loss
```

Researchers often experiment with

```text
β

=

0.01

0.05

0.1

0.5

1

2

4
```

depending on the application.

---

## 24.6.17.11 Complete Training Step

A typical training iteration becomes

```python
optimizer.zero_grad()

reconstruction, mu, log_var = model(x)

reconstruction_loss = F.mse_loss(

    reconstruction,

    x

)

kl_loss = -0.5 * torch.mean(

    1

    + log_var

    - mu.pow(2)

    - log_var.exp()

)

loss = reconstruction_loss + kl_loss

loss.backward()

optimizer.step()
```

This single optimization simultaneously trains

- the encoder,
- the latent distribution,
- the decoder.

---

## 24.6.17.12 Monitoring Training

During training,

it is useful to record each component separately.

```python
print(

    f"Reconstruction: {reconstruction_loss:.4f}"

)

print(

    f"KL: {kl_loss:.4f}"

)

print(

    f"Total Loss: {loss:.4f}"

)
```

Typical output

```text
Epoch 50

Reconstruction : 0.0038

KL             : 0.0112

Total          : 0.0150
```

Monitoring the individual components helps diagnose problems such as

- posterior collapse,
- excessive KL regularization,
- poor reconstruction.

---

## 24.6.17.13 Materials Informatics Interpretation

For crystal generation,

the ELBO has an intuitive physical interpretation.

The reconstruction term ensures that the latent representation preserves

- crystal symmetry,
- bonding environments,
- atomic arrangements,
- compositional information.

The KL divergence ensures that these representations occupy a smooth, continuous chemical space from which entirely new crystal structures can be generated by random sampling.

Thus, the ELBO simultaneously preserves **structural fidelity** and **generative capability**, making Variational Autoencoders suitable for inverse materials design.

---

## 24.6.17.14 Transition to Latent Space Interpolation

Once the Variational Autoencoder has been trained using the ELBO objective, the latent space becomes continuous and well organized. One of the most remarkable consequences is that meaningful paths can be drawn between different materials within this latent space.

Instead of jumping abruptly from one crystal to another, the model can smoothly interpolate between chemically related materials, generating plausible intermediate structures. This property makes VAEs particularly valuable for exploring unexplored regions of chemical space and will be examined in the next section on **Latent Space Interpolation**.

## 24.6.18 Latent Space Interpolation

One of the most remarkable properties of Variational Autoencoders is that the learned latent space is **continuous**. Unlike ordinary autoencoders, where latent vectors may be scattered arbitrarily, the probabilistic regularization imposed by the KL divergence produces a smooth latent manifold.

This smoothness enables one of the most powerful capabilities of generative models:

> **Interpolation between materials.**

Instead of generating materials completely at random, we can smoothly move through latent space from one material to another and observe how crystal representations gradually change.

This capability has profound implications for inverse materials design because it allows researchers to explore entirely new regions of chemical space while maintaining physically meaningful intermediate structures.

---

## 24.6.18.1 What is Interpolation?

Interpolation simply means generating intermediate points between two known latent vectors.

Suppose

```text
Material A

↓

Latent Vector

↓

z₁
```

and

```text
Material B

↓

Latent Vector

↓

z₂
```

Instead of decoding only

```text
z₁
```

or

```text
z₂
```

we generate intermediate vectors

```text
z₁

↓

↓

↓

↓

z₂
```

Each intermediate vector is decoded into a new material.

---

## 24.6.18.2 Why Does This Work?

Recall that the VAE forces the latent space to approximate

```text
N(0,I)
```

This regularization creates a smooth probability landscape.

Nearby latent vectors correspond to

- similar crystal structures,
- similar chemistry,
- similar bonding,
- similar properties.

Therefore,

moving gradually through latent space also produces gradual changes in the generated material.

---

## 24.6.18.3 Ordinary Autoencoder vs VAE

### Ordinary Autoencoder

```text
•









•

```

No meaningful path exists.

Interpolation often produces

```text
Invalid Crystal
```

---

### Variational Autoencoder

```text
**************

*******************

**********************

*******************

**************
```

Every intermediate point belongs to the learned distribution.

Interpolation therefore remains meaningful.

---

## 24.6.18.4 Linear Interpolation

The simplest interpolation method is linear interpolation.

Suppose

```text
z₁
```

and

```text
z₂
```

represent two crystals.

We generate

```text
z

=

(1−α)z₁

+

αz₂
```

where

```text
α
```

varies between

```text
0

and

1
```

When

```text
α = 0
```

the result is

```text
z₁
```

When

```text
α = 1
```

the result is

```text
z₂
```

Intermediate values generate intermediate materials.

---

## 24.6.18.5 Visualization

Suppose

```text
z₁

↓

LiFePO₄
```

and

```text
z₂

↓

LiMnPO₄
```

Interpolation produces

```text
LiFePO₄

↓

↓

↓

↓

LiMnPO₄
```

The decoder generates a sequence of intermediate latent representations.

Conceptually,

```text
LiFePO₄

↓

Candidate 1

↓

Candidate 2

↓

Candidate 3

↓

LiMnPO₄
```

Some intermediate structures may correspond to previously unknown compounds.

---

## 24.6.18.6 Materials Science Interpretation

Suppose two battery cathode materials differ only in their transition metal.

```text
LiFePO₄

↓

Encoder

↓

z₁
```

```text
LiMnPO₄

↓

Encoder

↓

z₂
```

Interpolation explores latent vectors lying between

```text
z₁

and

z₂
```

These vectors may decode into

- mixed transition-metal compounds,
- partially substituted crystals,
- previously unknown compositions.

Such candidates can subsequently be evaluated using

- DFT,
- molecular dynamics,
- experimental synthesis.

---

## 24.6.18.7 Property Evolution

Interpolation often produces gradual changes in material properties.

For example,

```text
Band Gap

↓

3.8

↓

3.5

↓

3.2

↓

2.9

↓

2.6
```

Similarly,

other properties may evolve smoothly.

```text
Density

↓

Elastic Modulus

↓

Formation Energy

↓

Magnetic Moment
```

This continuous variation makes latent interpolation a valuable tool for inverse design.

---

## 24.6.18.8 Chemical Space Exploration

Instead of searching randomly,

researchers can intentionally explore specific regions.

```text
Known Material

↓

Interpolation

↓

Nearby Candidates

↓

Property Prediction

↓

DFT Validation
```

This approach greatly reduces computational cost compared with random enumeration.

---

## 24.6.18.9 PyTorch Implementation

Suppose

```python
z1
```

and

```python
z2
```

are latent vectors.

Linear interpolation can be implemented as

```python
import torch

steps = 10

interpolated = []

for alpha in torch.linspace(

    0,

    1,

    steps

):

    z = (1-alpha) * z1 + alpha * z2

    interpolated.append(z)
```

The list

```python
interpolated
```

contains latent vectors smoothly connecting

```text
z₁

and

z₂.
```

---

## 24.6.18.10 Decoding the Interpolation

Each interpolated vector can be passed through the decoder.

```python
generated = []

with torch.no_grad():

    for z in interpolated:

        x = decoder(

            z.unsqueeze(0)

        )

        generated.append(x)
```

The resulting outputs represent a sequence of gradually changing crystal descriptors.

---

## 24.6.18.11 Visualizing Latent Trajectories

Suppose the latent space has been reduced to two dimensions using PCA.

Interpolation appears as

```text
•

↓

•

↓

•

↓

•

↓

•
```

rather than abrupt jumps

```text
•









•
```

The smooth trajectory indicates that the latent space has learned meaningful chemical relationships.

---

## 24.6.18.12 Practical Applications

Latent interpolation has numerous applications in materials informatics.

These include

- discovering intermediate compounds,
- exploring alloy compositions,
- studying substitution effects,
- optimizing crystal chemistry,
- visualizing chemical space,
- generating candidate materials for DFT screening.

Rather than relying on random search,

researchers can deliberately move toward desired regions of latent space.

---

## 24.6.18.13 Limitations

Although interpolation is powerful,

it does not guarantee physically stable materials.

Intermediate latent vectors may decode into

- unstable structures,
- chemically invalid compounds,
- unrealistic atomic arrangements.

Therefore,

generated materials should always undergo

- property prediction,
- stability analysis,
- DFT relaxation,
- experimental validation.

Interpolation generates **candidates**, not verified materials.

---

## 24.6.18.14 Transition to Crystal Generation

Interpolation demonstrates that the latent space learned by a Variational Autoencoder is smooth and chemically meaningful. However, interpolation still depends on two existing materials.

The true power of generative models lies in their ability to create **entirely new materials** by sampling directly from the latent distribution without requiring any starting crystal.

In the next section, we use the trained Variational Autoencoder to perform **crystal generation**, demonstrating how completely new candidate materials can be synthesized computationally by sampling latent vectors from a standard Gaussian distribution.

## 24.6.19 Crystal Generation Using Variational Autoencoders

The ultimate objective of a Variational Autoencoder is not merely to reconstruct existing crystal structures but to generate entirely **new materials** that have never appeared in the training dataset.

This capability distinguishes generative models from predictive machine learning algorithms.

Predictive models answer

```text
Known Crystal

↓

Property
```

Generative models answer

```text
Random Latent Vector

↓

New Crystal
```

Once a VAE has learned a smooth latent representation of chemical space, generating new materials becomes remarkably simple.

---

## 24.6.19.1 From Reconstruction to Generation

During training, the workflow is

```text
Crystal

↓

Encoder

↓

Latent Distribution

↓

Sampling

↓

Decoder

↓

Reconstruction
```

Notice that both the encoder and decoder are required.

After training, however, the encoder is no longer necessary.

Instead,

we directly sample from the learned latent distribution.

```text
Random z

↓

Decoder

↓

Generated Crystal
```

The decoder effectively becomes a **crystal generator**.

---

## 24.6.19.2 Sampling from Latent Space

Because the latent space has been regularized to follow

```text
N(0,I)
```

new latent vectors are generated simply by sampling from a standard normal distribution.

```text
z

~

N(0,I)
```

Every sampled vector corresponds to a different point in latent space.

Each point potentially represents a different crystal.

---

## 24.6.19.3 Generation Pipeline

The complete generation pipeline is

```text
Random Number Generator

↓

Latent Vector

↓

Decoder

↓

Crystal Descriptor

↓

Crystal Structure

↓

DFT Validation

↓

Experiment
```

Unlike traditional computational screening,

no existing crystal is required.

---

## 24.6.19.4 Example

Suppose the latent dimension equals

```text
64
```

A randomly sampled vector might be

```text
[

0.18,

−0.63,

1.24,

...

]

```

Passing this vector through the decoder produces

```text
Crystal Descriptor

↓

Formation Energy

↓

Atomic Environment

↓

Structural Features
```

A post-processing algorithm can then reconstruct a crystal candidate.

---

## 24.6.19.5 Diversity of Generated Materials

Every random sample produces a different latent vector.

```text
z₁

↓

Crystal A
```

```text
z₂

↓

Crystal B
```

```text
z₃

↓

Crystal C
```

Even though all latent vectors originate from the same Gaussian distribution,

their decoded structures may differ significantly.

Consequently,

one trained VAE can generate thousands or even millions of candidate materials.

---

## 24.6.19.6 Novelty

An important question naturally arises.

> Does the VAE simply memorize the training set?

Ideally,

the answer is **no**.

Because the decoder receives previously unseen latent vectors,

it generates

- new compositions,
- new structural combinations,
- new crystal representations,

rather than exact copies of the training data.

The generated materials should be

- chemically plausible,
- statistically similar to known materials,
- yet sufficiently different to represent novel discoveries.

---

## 24.6.19.7 Crystal Validity

Not every generated sample is physically meaningful.

Some decoded structures may violate

- charge neutrality,
- crystallographic symmetry,
- coordination chemistry,
- bond length constraints,
- atomic overlap rules.

Therefore,

generated crystals must undergo additional validation before being considered realistic candidates.

---

## 24.6.19.8 Validation Pipeline

A typical workflow becomes

```text
Generated Crystal

↓

Chemical Validation

↓

Structure Relaxation

↓

DFT Calculation

↓

Property Prediction

↓

Experimental Synthesis
```

Only a small fraction of generated crystals ultimately satisfy all physical constraints.

---

## 24.6.19.9 Filtering Generated Materials

Several filtering criteria are commonly applied.

### Chemical Validity

```text
Valid Stoichiometry

✓
```

---

### Charge Neutrality

```text
Net Charge

=

0

✓
```

---

### Structural Stability

```text
Reasonable Bond Lengths

✓
```

---

### Thermodynamic Stability

```text
Formation Energy

↓

Low
```

---

### Mechanical Stability

```text
Elastic Constants

↓

Stable
```

Only candidates satisfying these criteria proceed to expensive DFT calculations.

---

## 24.6.19.10 Materials Discovery Workflow

Modern AI-assisted materials discovery often follows

```text
Materials Database

↓

Train VAE

↓

Generate 100,000 Candidates

↓

Property Prediction

↓

Top 500 Candidates

↓

DFT Screening

↓

Top 20 Candidates

↓

Experimental Validation
```

Instead of performing DFT on hundreds of thousands of structures,

researchers evaluate only the most promising candidates.

This dramatically reduces computational cost.

---

## 24.6.19.11 PyTorch Implementation

Generating new latent vectors requires only one command.

```python
latent_dimension = 64

z = torch.randn(

    1,

    latent_dimension

)
```

The sampled vector follows

```text
N(0,I)
```

---

The decoder then generates a new crystal representation.

```python
with torch.no_grad():

    generated = decoder(

        z

    )

print(

    generated.shape

)
```

Example output

```text
torch.Size([1,512])
```

The decoder has reconstructed a new 512-dimensional crystal descriptor.

---

## 24.6.19.12 Generating Multiple Materials

Generating many candidates is equally straightforward.

```python
num_samples = 1000

z = torch.randn(

    num_samples,

    latent_dimension

)

with torch.no_grad():

    generated = decoder(

        z

    )
```

Output

```text
1000 Candidate Materials
```

Each row represents a different generated material.

---

## 24.6.19.13 Predicting Properties

Generated descriptors can immediately be passed to a property prediction model.

```python
predicted_band_gap = predictor(

    generated

)

predicted_energy = formation_model(

    generated

)
```

This allows rapid ranking of generated materials before expensive quantum-mechanical simulations.

---

## 24.6.19.14 Integration with DFT

A practical materials discovery pipeline often combines

```text
VAE

↓

Candidate Generation

↓

Property Prediction

↓

DFT

↓

Experimental Validation
```

Instead of replacing first-principles simulations,

the VAE serves as an intelligent proposal engine.

It dramatically reduces the search space by suggesting chemically meaningful candidate structures.

---

## 24.6.19.15 Limitations

Although Variational Autoencoders are historically important,

they exhibit several limitations.

They often produce

- blurry latent representations,
- averaged crystal descriptors,
- limited structural diversity,
- lower-quality samples compared to modern diffusion models.

These limitations motivated the development of

- Graph VAEs,
- Diffusion Models,
- Score-Based Models,
- Crystal Diffusion Networks.

Nevertheless,

the VAE remains the conceptual foundation upon which many modern generative models are built.

---

## 24.6.19.16 Transition to Materials-Specific VAEs

The Variational Autoencoder discussed thus far operates on generic feature vectors. While useful for understanding probabilistic generation, real materials are naturally represented as **graphs**, where atoms form nodes and chemical bonds define edges.

Representing crystals as graphs enables the model to preserve local atomic environments, coordination geometry, and periodic connectivity much more faithfully than fixed-length descriptors.

The next section extends the VAE framework to **Graph Variational Autoencoders (Graph VAEs)**, which adapt the probabilistic principles of VAEs to graph-structured crystal data and form the basis of many modern generative models for inorganic materials.

## 24.6.20 Graph Variational Autoencoders (Graph VAEs)

The Variational Autoencoder introduced in the previous sections assumes that every material can be represented as a fixed-length numerical vector.

For some applications, this assumption is reasonable.

For example,

- composition descriptors,
- matminer features,
- Magpie descriptors,
- SOAP vectors,
- crystal fingerprints.

However,

crystals are **not naturally vectors**.

They are fundamentally

```text
Atoms

+

Chemical Bonds

+

Periodic Connectivity
```

This naturally leads to a graph representation.

Consequently,

modern generative materials models replace the fully connected encoder and decoder with **Graph Neural Networks (GNNs)**.

When Variational Autoencoders are combined with Graph Neural Networks, the resulting architecture is called a **Graph Variational Autoencoder (Graph VAE).**

Graph VAEs have become one of the earliest successful deep generative models for crystal structures.

---

## 24.6.20.1 Why Ordinary VAEs Are Limited

Suppose we represent a crystal using

```text
512 Numerical Features
```

Although these descriptors contain useful information,

they often lose

- local bonding,
- neighborhood information,
- atomic connectivity,
- coordination geometry,
- periodic graph topology.

For example,

consider diamond.

```text
Carbon

↓

Descriptor

↓

512 Numbers
```

The descriptor may summarize

- density,
- lattice constant,
- atomic radius,

but it no longer explicitly represents

```text
Which carbon atom is bonded to which?
```

This information is extremely important.

---

## 24.6.20.2 Graph Representation

Instead of descriptors,

we represent the crystal directly.

```text
Crystal

↓

Graph
```

where

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

Each node contains

- atomic number,
- electronegativity,
- atomic radius,
- valence electrons,

while edges contain

- bond length,
- neighbor distance,
- coordination information.

---

## 24.6.20.3 Graph VAE Architecture

The overall architecture becomes

```text
Crystal Graph

↓

Graph Encoder

↓

μ

σ

↓

Reparameterization

↓

Latent Graph Vector

↓

Graph Decoder

↓

Generated Crystal Graph
```

Notice that

the probabilistic framework remains unchanged.

Only the encoder and decoder become graph neural networks.

---

## 24.6.20.4 Graph Encoder

Instead of using

```text
Linear Layers
```

the encoder now uses

```text
Graph Convolutions
```

Workflow

```text
Crystal Graph

↓

Message Passing

↓

Node Embeddings

↓

Graph Embedding

↓

μ

σ
```

The graph embedding summarizes the entire crystal.

---

## 24.6.20.5 Graph Decoder

The decoder performs the reverse operation.

```text
Latent Vector

↓

Graph Decoder

↓

Atoms

↓

Edges

↓

Crystal Graph
```

Instead of reconstructing

```text
512 Numbers
```

it reconstructs

- atom identities,
- atomic positions,
- bonding relationships,
- crystal connectivity.

---

## 24.6.20.6 Learning Crystal Topology

The Graph VAE learns much richer information than ordinary VAEs.

For example,

instead of learning

```text
Average Bond Length
```

it learns

```text
Complete Bond Network
```

Instead of learning

```text
Average Coordination Number
```

it learns

```text
Actual Neighbor Relationships
```

Consequently,

generated structures preserve much more realistic chemistry.

---

## 24.6.20.7 Message Passing During Encoding

Suppose each node initially stores

```text
Atomic Features
```

Graph convolution updates these features.

```text
Node Features

↓

Neighbor Aggregation

↓

Updated Features

↓

Repeat

↓

Graph Embedding
```

After several message-passing layers,

the embedding contains information about the entire crystal.

---

## 24.6.20.8 Materials Science Example

Consider

```text
LiFePO₄
```

Graph representation

```text
Li

↓

O

↓

P

↓

Fe

↓

O
```

Message passing allows

- oxygen atoms,
- iron atoms,
- phosphorus atoms,

to exchange information.

Eventually,

the graph embedding captures

- local environments,
- crystal symmetry,
- bonding topology,
- chemical composition.

---

## 24.6.20.9 Graph Decoder Challenges

Encoding graphs is relatively straightforward.

Decoding graphs is much harder.

The decoder must predict

- number of atoms,
- atom identities,
- edge connectivity,
- periodic lattice,
- crystal symmetry.

Unlike images,

graphs possess variable size.

Different materials contain

- different numbers of atoms,
- different bonding patterns,
- different lattice structures.

Graph generation is therefore significantly more difficult than image generation.

---

## 24.6.20.10 PyTorch Geometric Example

A Graph VAE encoder typically begins with graph convolution layers.

```python
from torch_geometric.nn import GCNConv

import torch.nn.functional as F

class GraphEncoder(nn.Module):

    def __init__(

        self,

        input_dim,

        hidden_dim,

        latent_dim

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

        self.mu_layer = nn.Linear(

            hidden_dim,

            latent_dim

        )

        self.logvar_layer = nn.Linear(

            hidden_dim,

            latent_dim

        )
```

---

## 24.6.20.11 Forward Pass

```python
def forward(

    self,

    x,

    edge_index

):

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

    graph_embedding = x.mean(

        dim=0

    )

    mu = self.mu_layer(

        graph_embedding

    )

    log_var = self.logvar_layer(

        graph_embedding

    )

    return mu, log_var
```

Notice that

graph convolution replaces the dense encoder introduced earlier.

---

## 24.6.20.12 Applications of Graph VAEs

Graph VAEs have been applied to

- molecular generation,
- inorganic crystal generation,
- catalyst discovery,
- battery materials,
- porous materials,
- metal-organic frameworks,
- polymers.

Their ability to preserve graph topology makes them considerably more powerful than descriptor-based VAEs.

---

## 24.6.20.13 Limitations

Despite their success,

Graph VAEs still exhibit important limitations.

They often struggle with

- periodic boundary conditions,
- variable lattice parameters,
- symmetry preservation,
- generation of large crystals,
- decoding complex graph structures.

These limitations motivated the development of more powerful generative architectures based on **diffusion models**, which currently represent the state of the art in crystal generation.

---

## 24.6.20.14 Transition to Diffusion Models

Variational Autoencoders were the first deep generative models to successfully learn continuous latent spaces for materials generation. However, their generated samples are often overly smooth and may lack the structural fidelity required for high-quality crystal design.

Recent advances in generative AI have largely shifted toward **Diffusion Models**, which generate materials by gradually removing noise from random latent states. These models have dramatically improved sample quality and now power many state-of-the-art crystal generation frameworks such as **DiffCSP**, **CDVAE extensions**, and other score-based crystal generation models.

In the next major section, we transition from latent-variable generative models to **Diffusion Models**, which currently represent one of the most powerful approaches for AI-driven materials discovery.

## 24.7 Diffusion Models

Variational Autoencoders represented one of the first successful deep generative models for materials discovery. They introduced the idea of learning a continuous latent space from which new materials could be generated through probabilistic sampling.

Despite their success, VAEs possess important limitations.

They often produce

- blurry latent representations,
- averaged structures,
- reduced structural diversity,
- lower-quality generated samples.

Over the last few years, a new class of generative models has largely replaced VAEs for many applications in computer vision, molecular generation, protein design, and increasingly, materials science.

These models are known as **Diffusion Models**.

Today, diffusion models represent the state of the art in generative AI and have become one of the most promising approaches for computational materials discovery.

---

## 24.7.1 Motivation

Consider image generation.

Suppose we begin with

```text
Completely Random Noise
```

```text
□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□
```

A diffusion model gradually removes the noise until a meaningful image appears.

```text
Noise

↓

Less Noise

↓

Recognizable Shape

↓

Clear Image
```

The remarkable idea is that

> **Generation starts from pure noise.**

The same principle can be applied to crystal structures.

Instead of generating images,

we generate

- crystal graphs,
- atomic coordinates,
- lattice parameters,
- complete crystal structures.

---

## 24.7.2 Basic Idea

Diffusion models consist of two processes.

### Forward Process

Gradually destroys information.

```text
Crystal

↓

Add Noise

↓

More Noise

↓

More Noise

↓

Pure Gaussian Noise
```

---

### Reverse Process

Learns to reverse this corruption.

```text
Pure Noise

↓

Remove Noise

↓

Better Crystal

↓

Remove Noise

↓

Final Crystal
```

The reverse process is learned by a neural network.

---

## 24.7.3 Why Learn to Remove Noise?

Learning to directly generate crystal structures is extremely difficult.

Instead,

the neural network solves a much simpler problem.

At every step,

it answers

> "Given a slightly noisy crystal, what noise should be removed?"

This simpler learning objective dramatically improves training stability.

---

## 24.7.4 Forward Diffusion Process

Suppose the original crystal representation is

```text
x₀
```

The forward process repeatedly adds Gaussian noise.

```text
x₀

↓

x₁

↓

x₂

↓

x₃

↓

...

↓

xₜ
```

Eventually,

```text
xₜ
```

becomes indistinguishable from random Gaussian noise.

---

## 24.7.5 Visual Interpretation

Imagine a crystal gradually disappearing.

```text
Perfect Crystal

↓

Slightly Corrupted

↓

Moderately Corrupted

↓

Highly Corrupted

↓

Random Noise
```

Each step destroys only a tiny amount of information.

After many steps,

all structural information disappears.

---

## 24.7.6 Reverse Diffusion

Now imagine reversing the movie.

```text
Random Noise

↓

Weak Crystal Pattern

↓

Atomic Arrangement

↓

Crystal Framework

↓

Complete Crystal
```

Instead of destroying information,

the neural network reconstructs it.

---

## 24.7.7 Why This Works

The forward process is known exactly.

We intentionally add

known Gaussian noise.

Therefore,

the neural network can learn

how to remove that same noise.

Rather than generating crystals directly,

it repeatedly performs

```text
Noise Prediction
```

until the crystal emerges.

---

## 24.7.8 Difference from VAEs

Variational Autoencoder

```text
Random Latent Vector

↓

Decoder

↓

Crystal
```

Only one decoding step.

---

Diffusion Model

```text
Noise

↓

Denoise

↓

Denoise

↓

Denoise

↓

...

↓

Crystal
```

Hundreds or even thousands of denoising steps.

Although slower,

the generated samples are usually much higher quality.

---

## 24.7.9 Why Diffusion Models Produce Better Samples

Every denoising step makes only a small correction.

Instead of attempting to generate an entire crystal in one pass,

the model gradually refines

- atomic positions,
- lattice vectors,
- local environments,
- bonding topology.

This iterative refinement significantly improves sample quality.

---

## 24.7.10 Applications in Materials Science

Diffusion models are increasingly used for

- crystal generation,
- inverse materials design,
- catalyst discovery,
- battery materials,
- molecular generation,
- porous framework generation,
- alloy discovery.

Many recent state-of-the-art materials generation papers now employ diffusion-based architectures.

---

## 24.7.11 Workflow

A simplified workflow is

```text
Training Crystal

↓

Forward Diffusion

↓

Random Noise

↓

Neural Network Learns

↓

Reverse Diffusion

↓

Recovered Crystal
```

Once training is complete,

generation begins directly from

```text
Random Gaussian Noise
```

---

## 24.7.12 Advantages

Compared with previous generative models,

diffusion models provide

- higher sample quality,
- more stable training,
- better diversity,
- improved structural realism,
- superior generation of complex materials,
- state-of-the-art performance across numerous scientific domains.

These advantages explain why diffusion models have rapidly become one of the dominant approaches in modern generative AI.

---

## 24.7.13 Limitations

Despite their impressive performance,

diffusion models also possess several disadvantages.

Generation is

- computationally expensive,
- slower than VAEs,
- memory intensive,
- often requires hundreds of denoising iterations.

Nevertheless,

the quality improvements usually outweigh these additional computational costs.

---

## 24.7.14 Transition

Understanding diffusion models begins with understanding how noise is gradually added to data during the forward process.

The next section derives the **forward diffusion process** mathematically and explains how Gaussian noise progressively transforms an ordered crystal into pure random noise while preserving a tractable probabilistic framework for learning.

## 24.7.15 Forward Diffusion Process

The forward diffusion process is the first mathematical component of a diffusion model.

Its purpose is deliberately simple:

> **Take a real crystal and gradually destroy its information by adding controlled Gaussian noise.**

The process begins with a clean crystal representation and progressively transforms it into random noise.

```text
Clean Crystal

↓

Small Noise

↓

More Noise

↓

Heavy Noise

↓

Pure Gaussian Noise
```

The remarkable aspect of diffusion models is that this forward process does **not** need to be learned.

We define it mathematically.

The neural network is trained to learn the **reverse process**.

---

## 24.7.15.1 Starting with a Clean Crystal

Let the original crystal representation be

```text
x₀
```

For example,

```text
x₀

=

Crystal Structure
```

Depending on the model,

`x₀` may represent

- atomic coordinates,
- lattice parameters,
- crystal graph features,
- atom embeddings,
- periodic fractional coordinates,
- a continuous crystal representation.

The diffusion process gradually transforms

```text
x₀
```

into

```text
xₜ
```

where `t` represents the diffusion timestep.

---

## 24.7.15.2 Adding Noise

At each timestep,

a small amount of Gaussian noise is added.

Conceptually,

```text
x₀

↓

Add small noise

↓

x₁
```

Then

```text
x₁

↓

Add small noise

↓

x₂
```

and so on.

Eventually,

```text
xₜ

≈

Gaussian Noise
```

The model may use hundreds or thousands of timesteps.

---

## 24.7.15.3 Noise Schedule

The amount of noise added at each timestep is controlled by a **noise schedule**.

Define

```text
β₁

β₂

β₃

...

βₜ
```

where each

```text
βₜ
```

controls how much noise is introduced at timestep `t`.

Usually,

```text
0 < βₜ < 1
```

and the noise gradually increases.

For example,

```text
β₁ = 0.0001
```

```text
β₂ = 0.0002
```

```text
β₃ = 0.0003
```

and so forth.

The exact schedule depends on the diffusion architecture.

---

## 24.7.15.4 Why Use Small Noise Steps?

Suppose we completely destroy the crystal in one step.

```text
Crystal

↓

Random Noise
```

The model would have to learn how to reconstruct the entire structure in one operation.

This is extremely difficult.

Instead,

we divide the transformation into many small steps.

```text
Crystal

↓

98% Crystal + 2% Noise

↓

95% Crystal + 5% Noise

↓

90% Crystal + 10% Noise

↓

...

↓

100% Noise
```

Each individual transformation becomes much easier to model.

---

## 24.7.15.5 The Forward Transition

The forward diffusion process is commonly represented as

```text
q(xₜ | xₜ₋₁)
```

This means

> the probability of obtaining `xₜ` given the previous state `xₜ₋₁`.

The transition is Gaussian.

Conceptually,

```text
xₜ

=

slightly modified xₜ₋₁

+

Gaussian Noise
```

---

## 24.7.15.6 Mathematical Form

The forward transition can be written as

```text
q(xₜ | xₜ₋₁)

=

N(
√(1−βₜ)xₜ₋₁,

βₜI
)
```

This equation contains two important components.

The first is

```text
√(1−βₜ)xₜ₋₁
```

which preserves part of the original signal.

The second is

```text
βₜI
```

which controls the Gaussian noise.

---

## 24.7.15.7 Intuition

Suppose

```text
βₜ
```

is very small.

Then

```text
√(1−βₜ)
```

is close to

```text
1
```

Therefore,

most of the original crystal remains.

```text
Original Crystal

████████████████

↓

After Small Noise

██████████████░░
```

As `t` increases,

```text
βₜ
```

becomes larger.

The original signal gradually disappears.

---

## 24.7.15.8 Cumulative Noise

Applying the transition repeatedly would require calculating

```text
x₁

↓

x₂

↓

x₃

↓

...

↓

xₜ
```

during every training iteration.

This would be inefficient.

Fortunately,

the forward diffusion process has a closed-form expression.

Define

```text
αₜ = 1 − βₜ
```

and

```text
ᾱₜ = α₁ α₂ ... αₜ
```

Then the noisy sample can be obtained directly from the original crystal.

```text
xₜ

=

√ᾱₜ x₀

+

√(1−ᾱₜ) ε
```

where

```text
ε ~ N(0,I)
```

This equation is fundamental to diffusion models.

---

## 24.7.15.9 Meaning of the Equation

The equation

```text
xₜ

=

√ᾱₜ x₀

+

√(1−ᾱₜ) ε
```

contains two competing terms.

### Original Crystal

```text
√ᾱₜ x₀
```

This preserves information from the original crystal.

---

### Noise

```text
√(1−ᾱₜ) ε
```

This introduces random Gaussian noise.

---

Therefore,

```text
Crystal Signal

+

Noise

=

Noisy Crystal
```

---

## 24.7.15.10 Early Timesteps

At small `t`,

```text
ᾱₜ ≈ 1
```

Therefore,

```text
√ᾱₜ ≈ 1
```

and

```text
√(1−ᾱₜ) ≈ 0
```

so

```text
xₜ ≈ x₀
```

The crystal remains almost unchanged.

---

## 24.7.15.11 Late Timesteps

At large `t`,

```text
ᾱₜ ≈ 0
```

Therefore,

```text
√ᾱₜ ≈ 0
```

and

```text
√(1−ᾱₜ) ≈ 1
```

so

```text
xₜ ≈ ε
```

The crystal becomes almost pure Gaussian noise.

---

## 24.7.15.12 Complete Diffusion Trajectory

The entire process can therefore be visualized as

```text
t = 0

████████████████████
Crystal
```

```text
t = 100

████████████████░░░░
Mostly Crystal
```

```text
t = 300

██████████░░░░░░░░░░
Partially Noisy
```

```text
t = 600

████░░░░░░░░░░░░░░░░
Mostly Noise
```

```text
t = 1000

░░░░░░░░░░░░░░░░░░░░
Gaussian Noise
```

The exact number of timesteps is architecture-dependent.

---

## 24.7.15.13 PyTorch Implementation

A simple implementation of the forward diffusion process is

```python
import torch


def add_noise(

    x0,

    t,

    alpha_bar

):

    noise = torch.randn_like(

        x0

    )

    signal_scale = torch.sqrt(

        alpha_bar[t]

    )

    noise_scale = torch.sqrt(

        1 - alpha_bar[t]

    )

    xt = (

        signal_scale * x0

        +

        noise_scale * noise

    )

    return xt, noise
```

The function returns

```text
xₜ
```

and the exact noise

```text
ε
```

that was added.

The noise is important because the neural network will later be trained to predict it.

---

## 24.7.15.14 Constructing a Noise Schedule

A simple linear schedule can be implemented as

```python
T = 1000

beta = torch.linspace(

    1e-4,

    0.02,

    T

)
```

Then

```python
alpha = 1.0 - beta
```

and

```python
alpha_bar = torch.cumprod(

    alpha,

    dim=0

)
```

These quantities define the complete forward diffusion process.

---

## 24.7.15.15 Example

Suppose the crystal representation is

```python
x0 = torch.randn(

    1,

    128

)
```

We choose a random timestep.

```python
t = torch.randint(

    0,

    T,

    (1,)

)
```

Then add noise.

```python
xt, noise = add_noise(

    x0,

    t,

    alpha_bar

)
```

Now,

```text
x0

↓

Clean Crystal
```

and

```text
xt

↓

Noisy Crystal
```

represent two different states of the same diffusion trajectory.

---

## 24.7.15.16 Why Store the Noise?

The diffusion model knows exactly which noise was added.

Suppose

```text
Clean Crystal

+

Noise

↓

Noisy Crystal
```

During training,

the neural network receives the noisy crystal and timestep.

```text
xₜ

+

t

↓

Neural Network

↓

Predicted Noise
```

The target is the actual noise

```text
ε
```

used during the forward process.

Therefore,

the training objective becomes

```text
Predicted Noise

≈

Actual Noise
```

This is considerably easier than directly asking the network to reconstruct the complete crystal.

---

## 24.7.15.17 Why This Is Powerful for Crystal Generation

The forward process converts a complicated crystal-generation problem into a controlled noise-prediction problem.

The model does not initially need to understand

```text
Complete Crystal
```

all at once.

Instead,

it learns how crystal information disappears under controlled noise.

Later,

the same knowledge is used in reverse.

```text
Noise

↓

Remove a little noise

↓

Remove a little more

↓

...

↓

Crystal
```

---

## 24.7.15.18 Important Materials-Specific Consideration

The forward diffusion process described above is mathematically straightforward for continuous variables.

However,

crystals contain several different types of information.

For example,

```text
Atomic Coordinates
```

are continuous,

while

```text
Element Identity
```

is categorical.

Similarly,

```text
Lattice Parameters
```

are continuous,

while

```text
Crystal Symmetry
```

may be represented discretely.

Therefore,

real crystal diffusion models cannot always apply identical Gaussian noise to every component.

Different representations may require different diffusion mechanisms.

For example,

```text
Coordinates

↓

Continuous Diffusion
```

while

```text
Atomic Species

↓

Categorical Diffusion
```

and

```text
Crystal Composition

↓

Discrete or Conditional Modeling
```

This is one reason crystal diffusion is significantly more complicated than image diffusion.

---

## 24.7.15.19 Transition to Reverse Diffusion

The forward process is completely known.

We choose

```text
βₜ
```

and can therefore generate

```text
xₜ
```

for any timestep.

The difficult part is the reverse direction.

We must now train a neural network to learn

```text
xₜ

↓

xₜ₋₁
```

or equivalently,

```text
Noisy Crystal

+

Timestep

↓

Predicted Noise

↓

Less Noisy Crystal
```

Once the neural network learns this reverse transformation, generation can begin from pure Gaussian noise and proceed step by step toward a complete crystal structure.

The next section develops the **reverse diffusion process**, including the noise-prediction network, timestep embeddings, denoising equations, and the connection between diffusion models and score-based generative modeling.

## 24.7.16 Reverse Diffusion Process

The forward diffusion process gradually transforms a clean crystal into Gaussian noise.

The generation problem requires the opposite operation.

Starting from random noise,

```text
Random Noise

↓

Less Noise

↓

More Structure

↓

Crystal Framework

↓

Complete Crystal

## 24.7.17 Score-Based Diffusion Models

The noise-prediction formulation introduced in the previous section provides a practical way to train diffusion models. A deeper mathematical interpretation is obtained by considering the **score function** of the probability distribution.

This perspective is particularly important for modern generative modeling because it connects diffusion models with

- probability density estimation,
- stochastic differential equations,
- Langevin dynamics,
- continuous-time diffusion,
- score matching.

For materials science, the score-based formulation provides a powerful framework for learning the probability distribution of crystal structures and generating new structures by moving noisy samples toward regions of high probability.

---

## 24.7.17.1 What Is a Score Function?

Consider a probability distribution

```text
p(x)

## 24.7.18 Crystal Diffusion

The general diffusion framework provides a powerful method for generating data from noise. However, crystals are fundamentally different from ordinary images or simple vectors.

A crystal is a periodic three-dimensional arrangement of atoms governed by strict physical and geometric constraints.

A complete crystal representation contains

```text
Atomic Species

+

Atomic Coordinates

+

Lattice Vectors

+

Periodic Boundary Conditions

+

Crystal Symmetry
```

Therefore, a successful crystal diffusion model must learn more than statistical patterns.

It must also respect the geometry and symmetries of crystalline materials.

This has led to the development of **crystal-specific diffusion models**, which adapt diffusion principles to the unique structure of materials.

---

## 24.7.18.1 Why Crystal Generation Is Difficult

Consider a simple image.

An image consists of

```text
Pixels

↓

Continuous Values
```

A crystal contains multiple interacting variables.

```text
Crystal

├── Elements
├── Positions
├── Lattice
├── Periodicity
└── Symmetry
```

Each variable has different mathematical properties.

For example,

```text
Atomic Position

↓

Continuous
```

while

```text
Element Identity

↓

Discrete
```

and

```text
Crystal Symmetry

↓

Structured / Discrete
```

A generative model must handle these different representations simultaneously.

---

## 24.7.18.2 Crystal as a Mathematical Object

A crystal can be represented by

```text
C = (A, X, L)
```

where

```text
A
```

represents atomic species,

```text
X
```

represents atomic coordinates,

and

```text
L
```

represents the lattice.

Conceptually,

```text
Crystal

=

Atoms

+

Coordinates

+

Lattice
```

A crystal diffusion model attempts to learn

```text
p(A, X, L)
```

or a conditional distribution such as

```text
p(A, X, L | target_property)
```

The second formulation is particularly important for inverse materials design.

---

## 24.7.18.3 Periodic Boundary Conditions

One of the most important differences between ordinary graphs and crystals is **periodicity**.

A crystal is not an isolated collection of atoms.

Instead,

```text
Unit Cell

↓

Repeated Infinitely
```

For example,

```text
┌─────────────┐
│   Crystal   │
│   Unit Cell │
└─────────────┘
      ↓
Repeated
      ↓
Infinite Crystal
```

The atoms at one boundary interact with atoms in neighboring periodic images.

A generative model that ignores this behavior can generate physically meaningless structures.

---

## 24.7.18.4 Fractional Coordinates

Crystal structures are often represented using fractional coordinates.

Suppose

```text
fᵢ = (uᵢ, vᵢ, wᵢ)
```

is the fractional coordinate of atom `i`.

The Cartesian coordinate is obtained from the lattice matrix.

```text
rᵢ = L fᵢ
```

where

```text
L
```

contains the lattice vectors.

This representation is particularly useful for diffusion models because fractional coordinates naturally describe positions inside the periodic unit cell.

---

## 24.7.18.5 Why Cartesian Coordinates Are Not Always Ideal

Suppose a crystal is rotated.

Its Cartesian coordinates change.

```text
Crystal

↓

Rotate

↓

Different Coordinates
```

But physically,

```text
Crystal

=

Same Crystal
```

This means a model operating directly on Cartesian coordinates must learn rotational symmetry from data.

That is inefficient.

Instead,

modern crystal diffusion models often use representations and architectures that explicitly respect geometric symmetry.

---

## 24.7.18.6 Translational Symmetry

Crystals are invariant under translations by lattice vectors.

If

```text
r
```

is an atomic position,

then

```text
r + n₁a + n₂b + n₃c
```

represents an equivalent periodic position, where

```text
n₁, n₂, n₃ ∈ Z
```

and

```text
a, b, c
```

are the lattice vectors.

Therefore,

a crystal diffusion model must understand that these positions are physically equivalent under periodic translations.

---

## 24.7.18.7 Rotational Symmetry

The physical properties of a crystal should not depend on the coordinate system used to describe it.

For example,

```text
Crystal

↓

Rotate

↓

Same Physical Structure
```

This motivates **rotation-equivariant** neural networks.

If the input coordinates rotate,

the predicted geometric quantities should transform consistently.

---

## 24.7.18.8 Permutation Symmetry

Atoms of the same chemical species are physically indistinguishable.

Consider

```text
Atom 1 = Fe

Atom 2 = Fe
```

Swapping their labels should not change the physical crystal.

Therefore,

the model should be invariant to permutations of equivalent atoms.

This property is known as **permutation invariance**.

---

## 24.7.18.9 E(3)-Equivariance

A particularly important concept in geometric deep learning is **E(3)-equivariance**.

E(3) represents transformations involving

- translations,
- rotations,
- reflections.

An equivariant model ensures that geometric predictions transform correctly when the input structure is transformed.

Conceptually,

```text
Transform Input

↓

Neural Network

↓

Transform Output
```

rather than

```text
Transform Input

↓

Completely Different Prediction
```

This is extremely important for crystal generation.

---

## 24.7.18.10 Crystal Diffusion Representation

A simplified crystal diffusion model may operate on

```text
X = Atomic Coordinates
```

while separately modeling

```text
A = Atomic Species
```

and

```text
L = Lattice
```

The complete generative process becomes

```text
Random Noise

↓

Generate Lattice

↓

Generate Atomic Species

↓

Generate Atomic Coordinates

↓

Apply Periodicity

↓

Relax Crystal

↓

Valid Crystal
```

Different models may change this order or jointly generate all components.

---

## 24.7.18.11 Continuous Variables

Several crystal variables are naturally continuous.

Examples include

```text
Lattice Lengths

Angles

Atomic Coordinates

Atomic Displacements
```

Gaussian diffusion can therefore be applied naturally.

For example,

```text
xₜ

=

√ᾱₜ x₀

+

√(1−ᾱₜ) ε
```

can be applied to continuous coordinates.

---

## 24.7.18.12 Discrete Variables

Atomic species are different.

Consider

```text
Li

Fe

P

O
```

These are categorical values.

Adding ordinary Gaussian noise to an atomic number does not make physical sense.

For example,

```text
Fe

+

Gaussian Noise

=

Fe + 0.73
```

does not correspond to a valid chemical element.

Therefore,

categorical diffusion or another discrete generative mechanism may be required.

---

## 24.7.18.13 Joint Crystal Generation

A sophisticated model may generate

```text
Composition

↓

Lattice

↓

Coordinates
```

jointly.

Conceptually,

```text
Noise

↓

┌─────────────────┐
│ Crystal Model   │
├─────────────────┤
│ Composition     │
│ Lattice         │
│ Coordinates     │
└─────────────────┘

↓

Crystal
```

The model learns correlations between these components.

For example,

composition influences

- lattice size,
- coordination,
- bond lengths,
- symmetry.

---

## 24.7.18.14 Conditional Crystal Generation

The model can also be conditioned on a target property.

Suppose we want

```text
Band Gap

≈

2.0 eV
```

The generation process becomes

```text
Random Noise

+

Target Band Gap

↓

Crystal Diffusion Model

↓

Candidate Crystal
```

Similarly,

we may condition on

```text
Formation Energy

Bulk Modulus

Magnetic Moment

Ionic Conductivity

Thermal Conductivity
```

This turns crystal diffusion into an inverse design framework.

---

## 24.7.18.15 Example

Suppose the target is

```text
High Bulk Modulus

+

Low Formation Energy
```

The model receives

```python
condition = torch.tensor([

    target_bulk_modulus,

    target_formation_energy

])
```

The condition is embedded and supplied to the denoising network.

```python
condition_embedding = condition_encoder(

    condition

)

predicted_noise = model(

    noisy_structure,

    time_embedding,

    condition_embedding

)
```

The denoising process is therefore guided toward materials consistent with the desired properties.

---

## 24.7.18.16 Simplified Crystal Diffusion Architecture

A conceptual architecture can be written as

```text
                 Random Noise
                      │
                      ▼
              ┌──────────────┐
              │ Crystal       │
              │ Diffusion     │
              │ Network       │
              └──────────────┘
                 │    │    │
                 ▼    ▼    ▼
              Species Lattice Coordinates
                 │    │    │
                 └────┼────┘
                      ▼
                Crystal Structure
```

The network repeatedly updates the noisy structure until a valid crystal emerges.

---

## 24.7.18.17 PyTorch Representation

A simplified crystal representation might contain

```python
class CrystalState:

    def __init__(

        self,

        atomic_numbers,

        fractional_coords,

        lattice

    ):

        self.atomic_numbers = atomic_numbers

        self.fractional_coords = fractional_coords

        self.lattice = lattice
```

The diffusion model can then operate on the continuous components.

```python
state = CrystalState(

    atomic_numbers,

    fractional_coords,

    lattice

)
```

---

## 24.7.18.18 Lattice Generation

A lattice can be represented by a `3 × 3` matrix.

```python
lattice = torch.tensor([

    [a1, a2, a3],

    [b1, b2, b3],

    [c1, c2, c3]

])
```

The diffusion model can learn to generate the lattice parameters along with atomic coordinates.

For a physically valid structure,

the lattice must define a non-degenerate unit cell.

Therefore,

validation is required after generation.

---

## 24.7.18.19 Converting Fractional Coordinates

After generation,

fractional coordinates can be converted to Cartesian coordinates.

```python
cartesian_coords = (

    fractional_coords

    @ lattice

)
```

This produces the physical positions of atoms inside the unit cell.

---

## 24.7.18.20 Periodic Neighbor Construction

Once coordinates have been generated,

periodic neighbors can be constructed.

A simplified workflow is

```text
Generated Crystal

↓

Periodic Images

↓

Neighbor Search

↓

Crystal Graph
```

In practical implementations,

tools such as `pymatgen` can be used to construct periodic structures and analyze their local environments.

For example,

```python
from pymatgen.core import Structure, Lattice

structure = Structure(

    lattice,

    species,

    fractional_coords

)
```

The generated object can then be inspected using standard materials-science workflows.

---

## 24.7.18.21 Structural Validation

A generated crystal should not immediately be considered a valid material.

At minimum,

the structure should be checked for

```text
✓ Valid Composition

✓ Reasonable Atomic Distances

✓ Valid Lattice

✓ No Severe Atomic Overlap

✓ Periodic Consistency
```

Additional checks may include

```text
✓ Charge Balance

✓ Coordination Chemistry

✓ Symmetry

✓ Thermodynamic Stability
```

---

## 24.7.18.22 Structure Relaxation

A diffusion model generates a candidate.

It does not necessarily generate the exact equilibrium geometry.

Therefore,

a common workflow is

```text
Generated Structure

↓

Geometry Optimization

↓

Relaxed Structure

↓

Energy Calculation
```

Density Functional Theory can then be used to determine whether the candidate is energetically favorable.

---

## 24.7.18.23 DFT Integration

A practical computational workflow can be organized as

```text
Crystal Database

↓

Train Diffusion Model

↓

Generate 100,000 Candidates

↓

Basic Structural Filtering

↓

Machine Learning Property Prediction

↓

Select Top Candidates

↓

DFT Relaxation

↓

Formation Energy

↓

Stability Analysis
```

This combines the speed of machine learning with the physical accuracy of first-principles calculations.

---

## 24.7.18.24 Example Materials Informatics Pipeline

Suppose the research goal is to discover a semiconductor with a target band gap.

The workflow becomes

```text
Materials Dataset

↓

Crystal Diffusion Model

↓

Generate Candidates

↓

Band Gap Predictor

↓

Filter:

1.5 eV < Eg < 2.0 eV

↓

Formation Energy Model

↓

Select Stable Candidates

↓

DFT

↓

Final Candidates
```

The diffusion model acts as the **candidate generator**, while predictive models and DFT act as **candidate evaluators**.

---

## 24.7.18.25 Important Limitation

A generated structure can look chemically plausible while still being unstable.

For example,

```text
Generated Structure

✓ Valid Geometry

✓ Reasonable Bond Lengths

✓ Valid Composition
```

does not necessarily imply

```text
Thermodynamically Stable
```

Therefore,

generation and validation must remain separate stages.

```text
Generation

≠

Validation
```

A good generative model proposes candidates.

Physics determines whether those candidates are actually viable.

---

## 24.7.18.26 Current Research Direction

Modern crystal-generation research increasingly combines

```text
Diffusion Models

+

Equivariant Neural Networks

+

Periodic Geometry

+

Property Conditioning

+

DFT Validation
```

This combination allows models to move beyond simple crystal reconstruction toward genuine inverse materials design.

The long-term objective is

```text
Desired Property

↓

AI Model

↓

Novel Crystal

↓

DFT

↓

Experiment
```

This is one of the central goals of modern Materials Informatics.

---

## 24.7.18.27 Transition to Score-Based Crystal Generation

Crystal diffusion demonstrates how the general diffusion framework must be modified to handle the chemical and geometric structure of materials.

However, modern crystal generation does not rely on a single universal representation. Different approaches model coordinates, compositions, lattices, and crystal graphs in different ways.

A deeper understanding comes from examining the specific diffusion architectures developed for crystal generation.

The next section therefore focuses on **score-based crystal generation**, where geometric score networks guide noisy crystal configurations toward chemically and structurally plausible regions of materials space.

## 24.7.19 Score-Based Crystal Generation

The previous section introduced the general idea of crystal diffusion and showed why crystal structures require specialized treatment of

- atomic species,
- atomic coordinates,
- lattice parameters,
- periodicity,
- geometric symmetry.

A deeper formulation of this problem is obtained by using **score-based generative modeling**.

Instead of directly asking a neural network to generate a crystal, we train it to estimate the direction in which a noisy crystal should move to become more probable under the learned materials distribution.

This provides a natural connection between

```text
Crystal Structure

↓

Probability Distribution

↓

Score Function

↓

Denoising

↓

Generated Crystal

## 24.7.20 Modern Crystal Diffusion Architectures

The previous sections developed the mathematical foundation of score-based crystal generation.

The central idea was

```text
Noise

↓

Score Network

↓

Iterative Denoising

↓

Crystal

````markdown id="61482"
## 24.7.21 Property-Conditioned Crystal Generation

A major advantage of generative AI for materials discovery is the ability to generate structures according to a desired scientific objective.

Unconditional generation asks:

```text
What crystals look like the training distribution?
````

Property-conditioned generation asks:

```text
Which crystals could have the property I want?
```

This distinction changes generative AI from a structure-generation technique into an **inverse materials design framework**.

The overall problem becomes

```text
Target Property

↓

Generative Model

↓

Crystal Structure

↓

Property Evaluation
```

For example,

a researcher may want a material with

```text
Band Gap ≈ 2.0 eV
```

or

```text
High Ionic Conductivity
```

or

```text
Low Formation Energy
```

or

```text
High Bulk Modulus
```

The diffusion model attempts to generate structures satisfying these requirements.

---

## 24.7.21.1 Forward Design vs Inverse Design

Traditional materials research often follows a forward process.

```text
Composition

↓

Crystal Structure

↓

Electronic Structure

↓

Properties
```

The researcher chooses a material and calculates its properties.

This is called **forward design**.

The inverse problem reverses the direction.

```text
Desired Property

↓

Unknown Structure

↓

Candidate Material
```

This is substantially more difficult because many different structures can produce similar properties.

Therefore,

the inverse mapping is generally one-to-many.

```text
Property

        ↙
       ↙
Structure A
       ↘
        ↘
Structure B
        ↘
         Structure C
```

A generative model is particularly suitable for this problem because it can generate multiple candidates rather than searching for only one solution.

---

## 24.7.21.2 Mathematical Formulation

Let

```text
C
```

represent a crystal structure and

```text
y
```

represent its target property.

Unconditional generation models

```text
p(C)
```

whereas property-conditioned generation models

```text
p(C | y)
```

The objective is therefore to learn the probability distribution of crystals given a desired property.

For multiple properties,

```text
y = [y₁, y₂, ..., yₖ]
```

and the model learns

```text
p(C | y₁,y₂,...,yₖ)
```

---

## 24.7.21.3 Example: Band Gap

Suppose a dataset contains

```text
Crystal A → Eg = 0.8 eV

Crystal B → Eg = 1.4 eV

Crystal C → Eg = 2.1 eV

Crystal D → Eg = 3.5 eV
```

The model learns the relationship between

```text
Crystal Structure

↕

Band Gap
```

During generation,

we provide

```text
Target:

Eg = 2.0 eV
```

The diffusion model then attempts to generate structures from

```text
p(C | Eg = 2.0)
```

rather than simply sampling from

```text
p(C)
```

---

## 24.7.21.4 Conditioning Mechanism

A simple architecture is

```text
Noisy Crystal
      │
      ▼
Crystal Encoder
      │
      ▼
Crystal Features
      │
      ├───────────────┐
      │               │
      ▼               ▼
Time Embedding   Property Embedding
      │               │
      └───────┬───────┘
              ▼
       Diffusion Network
              │
              ▼
        Predicted Score
```

The target property is converted into a learned embedding.

```python
property_embedding = property_encoder(
    target_property
)
```

The diffusion network uses this embedding while predicting the denoising direction.

---

## 24.7.21.5 Property Encoder

Suppose the model receives three target properties.

```python
target = torch.tensor([
    2.0,      # band gap
    -2.5,     # formation energy
    150.0     # bulk modulus
])
```

A simple encoder can be written as

```python
class PropertyEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

    def forward(self, x):

        return self.network(x)
```

The output becomes a latent representation of the desired material properties.

---

## 24.7.21.6 Conditioning the Denoising Network

The diffusion network can receive

```text
Crystal Features

+

Time Embedding

+

Property Embedding
```

A simplified implementation is

```python
class ConditionalDiffusionModel(
    nn.Module
):

    def __init__(
        self,
        crystal_dim,
        property_dim,
        hidden_dim,
        time_dim
    ):

        super().__init__()

        self.crystal_encoder = nn.Linear(
            crystal_dim,
            hidden_dim
        )

        self.property_encoder = nn.Linear(
            property_dim,
            hidden_dim
        )

        self.time_encoder = nn.Linear(
            time_dim,
            hidden_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                crystal_dim
            )
        )

    def forward(
        self,
        crystal,
        time_embedding,
        property_condition
    ):

        h = self.crystal_encoder(
            crystal
        )

        p = self.property_encoder(
            property_condition
        )

        t = self.time_encoder(
            time_embedding
        )

        h = h + p + t

        return self.network(h)
```

This simple architecture demonstrates the basic conditioning mechanism.

---

## 24.7.21.7 Why Conditioning Is Difficult

The property is not independent of the structure.

For example,

band gap depends on

```text
Composition

+

Crystal Structure

+

Atomic Arrangement

+

Bonding

+

Electronic Structure
```

Therefore,

the diffusion model must learn a complicated mapping:

```text
Desired Property

↓

Structural Constraints

↓

Chemical Constraints

↓

Generated Crystal
```

A simple neural network may not capture these relationships accurately.

This is why property-conditioned materials generation generally benefits from strong structural representations and large datasets.

---

## 24.7.21.8 Continuous Property Conditioning

Some properties are continuous.

Examples include

```text
Band Gap

Formation Energy

Density

Bulk Modulus

Elastic Modulus

Magnetic Moment
```

These can be directly supplied as numerical values.

For example,

```python
target_band_gap = torch.tensor([
    [2.0]
])
```

The property encoder transforms the scalar into a learned feature vector.

```python
embedding = property_encoder(
    target_band_gap
)
```

---

## 24.7.21.9 Categorical Conditioning

Other properties or material classes may be categorical.

For example,

```text
Crystal System

Cubic

Tetragonal

Orthorhombic

Hexagonal
```

These can be represented using embeddings.

```python
class_embedding = nn.Embedding(
    num_classes,
    embedding_dim
)
```

Then

```python
system_embedding = class_embedding(
    crystal_system
)
```

The model can condition generation on the desired crystal class.

---

## 24.7.21.10 Composition Conditioning

The model can also be conditioned on composition.

For example,

```text
Target System:

Li-Fe-P-O
```

The model may be asked to generate structures within this chemical space.

Conceptually,

```text
Chemical System

+

Target Property

↓

Diffusion Model

↓

Crystal
```

This is useful when the researcher already knows which elements are experimentally accessible.

---

## 24.7.21.11 Partial Conditioning

It is not always necessary to specify every property.

For example,

```text
Composition:

Li-Fe-P-O

Band Gap:

1.5–2.0 eV

Other Properties:

Unknown
```

The model can generate structures satisfying the known constraints while allowing other characteristics to vary.

This provides a flexible design framework.

---

## 24.7.21.12 Property Ranges

In real materials discovery,

the desired property is often a range rather than an exact value.

For example,

```text
1.5 eV ≤ Eg ≤ 2.0 eV
```

Instead of conditioning on one value,

the researcher can sample target values inside the desired range.

```python
target_band_gap = torch.empty(
    batch_size,
    1
).uniform_(
    1.5,
    2.0
)
```

Each generated candidate is therefore associated with a target selected from the acceptable design window.

---

## 24.7.21.13 Multi-Objective Generation

Materials design is usually multi-objective.

For example,

a researcher may want

```text
High Band Gap

+

Low Formation Energy

+

High Bulk Modulus
```

These objectives can conflict.

For example,

increasing one property may decrease another.

Therefore,

the desired design space can be represented as

```text
Property 1

↕

Property 2
```

and the goal is to find a region where multiple objectives are acceptable.

---

## 24.7.21.14 Pareto Optimality

For multiple competing objectives,

the **Pareto frontier** is useful.

Suppose we want

```text
Low Formation Energy

High Bulk Modulus
```

A candidate is Pareto optimal if no other candidate is simultaneously

```text
Better in Every Objective
```

The generated materials can therefore be ranked according to Pareto dominance.

Conceptually,

```text
Bulk Modulus
     ↑
     │          ●
     │       ●
     │    ●
     │  ●
     │ ●
     └────────────────→
       Stability
```

The boundary represents a trade-off between competing objectives.

---

## 24.7.21.15 Weighted Conditioning

One simple strategy is to combine normalized objectives.

Suppose

```text
y₁ = Band Gap

y₂ = Stability

y₃ = Bulk Modulus
```

A combined score can be defined as

```text
S = w₁y₁ + w₂y₂ + w₃y₃
```

where

```text
w₁,w₂,w₃
```

represent the importance of each objective.

In code,

```python
score = (
    w_band_gap * band_gap
    +
    w_stability * stability
    +
    w_bulk * bulk_modulus
)
```

However,

weighted sums should be used carefully because the objectives may have different physical scales.

---

## 24.7.21.16 Normalization of Properties

Before conditioning,

properties should generally be normalized.

For example,

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

properties_scaled = scaler.fit_transform(
    properties
)
```

This produces approximately

```text
Mean ≈ 0

Standard Deviation ≈ 1
```

Normalization prevents a large numerical property from dominating smaller numerical properties.

---

## 24.7.21.17 Property Prediction Model

Property-conditioned generation requires reliable property information.

A separate predictive model can be trained.

For example,

```text
Crystal

↓

Graph Neural Network

↓

Predicted Band Gap
```

A simple architecture could be

```python
class PropertyPredictor(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(self, x):

        return self.network(x)
```

This model can be used to evaluate generated candidates.

---

## 24.7.21.18 Generator and Predictor

The generative and predictive models can be connected.

```text
                  Target Property
                        │
                        ▼
                ┌───────────────┐
                │ Diffusion     │
                │ Generator     │
                └───────┬───────┘
                        │
                        ▼
                  Crystal
                        │
                        ▼
                ┌───────────────┐
                │ Property      │
                │ Predictor     │
                └───────┬───────┘
                        │
                        ▼
                  Predicted
                    Property
```

This allows the researcher to check whether the generated structure actually satisfies the requested objective.

---

## 24.7.21.19 Generation-Verification Loop

Instead of generating one structure,

we generate many.

```python
candidates = []

for _ in range(
    10000
):

    crystal = sample_crystal(
        model,
        condition
    )

    candidates.append(
        crystal
    )
```

The candidates are then evaluated.

```python
predicted_properties = predictor(
    candidates
)
```

The researcher can filter the results.

```python
selected = candidates[
    (
        predicted_band_gap > 1.5
    )
    &
    (
        predicted_band_gap < 2.0
    )
]
```

This simple loop is one of the most useful practical patterns in generative materials research.

---

## 24.7.21.20 Classifier Guidance

Another approach to conditional generation is **classifier guidance**.

Instead of embedding the target property directly into the diffusion model,

a separate predictor or classifier estimates whether a structure satisfies the desired condition.

The diffusion score can then be modified.

Conceptually,

```text
Original Score

+

Condition Gradient

↓

Guided Score
```

The condition gradient pushes the generation trajectory toward desired structures.

---

## 24.7.21.21 Conceptual Classifier Guidance

Suppose

```text
p(y | x)
```

is a property predictor.

The conditional score can be related conceptually to

```text
∇x log p(x | y)
```

which can be decomposed into terms involving

```text
∇x log p(x)
```

and

```text
∇x log p(y | x)
```

The first term represents the learned distribution of realistic crystals.

The second term encourages the desired property.

This provides an elegant connection between generative modeling and property prediction.

---

## 24.7.21.22 Why Classifier Guidance Is Attractive

It allows the property predictor to be trained separately.

For example,

```text
Crystal Dataset
      │
      ├───────────────┐
      ▼               ▼
Diffusion Model   Property Model
      │               │
      └───────┬───────┘
              ▼
        Guided Generation
```

This modularity can be useful when an existing high-quality property predictor is already available.

---

## 24.7.21.23 Classifier-Free Guidance

Another important technique is **classifier-free guidance**.

During training,

the model sometimes receives the property condition and sometimes receives no condition.

Conceptually,

```text
Training Sample

        │
        ├── Conditioned
        │
        └── Unconditioned
```

The same network learns both behaviors.

During sampling,

the conditioned and unconditioned predictions can be combined.

A simplified formulation is

```text
guided

=

unconditional

+

w(
conditional
−
unconditional
)
```

where

```text
w
```

controls the strength of conditioning.

---

## 24.7.21.24 Guidance Strength

If

```text
w = 0
```

the generation is essentially unconditional.

As

```text
w
```

increases,

the model is pushed more strongly toward the target.

Conceptually,

```text
Low Guidance

↓

More Diversity

```

while

```text
High Guidance

↓

Stronger Property Satisfaction
```

However,

very strong guidance can reduce diversity or produce unrealistic structures.

Therefore,

guidance strength should be treated as a tunable parameter.

---

## 24.7.21.25 Diversity vs Property Accuracy

A fundamental trade-off exists between

```text
Diversity
```

and

```text
Target Accuracy
```

Suppose the model generates 1,000 structures.

Weak conditioning may produce

```text
900 diverse structures

+

100 target-satisfying structures
```

Strong conditioning may produce

```text
700 target-satisfying structures

+

300 diverse structures
```

but those 700 structures may be very similar to each other.

Therefore,

a useful evaluation should consider both.

---

## 24.7.21.26 Measuring Diversity

Generated candidates can be compared using

```text
Composition

Structural Fingerprint

Graph Embedding

Latent Representation
```

For example,

if 1,000 generated structures collapse into only 20 unique structures,

the model may be suffering from mode collapse or excessive guidance.

A simple structural diversity metric can be based on pairwise similarity.

```python
similarity_matrix = compute_similarity(
    generated_structures
)
```

Candidates with extremely high similarity can be removed.

---

## 24.7.21.27 Duplicate Removal

Generated datasets should be deduplicated.

A simple workflow is

```text
Generated Structures

↓

Canonical Representation

↓

Hash / Fingerprint

↓

Remove Duplicates
```

For example,

```python
unique_structures = {}

for structure in candidates:

    key = structure.composition.reduced_formula

    if key not in unique_structures:

        unique_structures[key] = structure
```

Composition alone is not sufficient for true structural deduplication, but it illustrates the concept.

A stronger implementation should compare structure-aware fingerprints.

---

## 24.7.21.28 Chemical Constraints

A generative model can produce structures that violate basic chemical principles.

For example,

```text
Impossible Oxidation States

Impossible Coordination

Severe Atomic Overlap

Unreasonable Density
```

Therefore,

chemistry-aware filtering should be integrated into the generation pipeline.

A useful first-stage filter is

```text
Generated Crystal

↓

Composition Check

↓

Distance Check

↓

Charge / Valence Check

↓

Coordination Check

↓

Accept / Reject
```

---

## 24.7.21.29 Oxidation-State Analysis

`pymatgen` can assist with oxidation-state analysis.

For example,

oxidation states may be assigned or estimated depending on the structure and available chemical information.

Conceptually,

```python
structure = structure.add_oxidation_state_by_guess()

print(
    structure
)
```

A researcher can then inspect whether the generated composition admits a chemically reasonable oxidation-state assignment.

This should be treated as a screening tool rather than an absolute proof of chemical validity.

---

## 24.7.21.30 Formation Energy Screening

Suppose the generative model produces

```text
10,000 candidates
```

A machine-learning model can estimate formation energies.

```python
formation_energy = formation_energy_model(
    candidates
)
```

The candidates can then be sorted.

```python
ranking = torch.argsort(
    formation_energy
)
```

Only the most promising candidates are sent to expensive DFT calculations.

---

## 24.7.21.31 Band Gap Screening

Similarly,

a band-gap model can be used.

```python
band_gap = band_gap_model(
    candidates
)
```

A target filter can then be applied.

```python
mask = (
    (band_gap >= 1.5)
    &
    (band_gap <= 2.0)
)
```

This converts a large generated dataset into a much smaller candidate set.

---

## 24.7.21.32 Combining Multiple Predictors

A realistic screening pipeline may use several models.

```text
Generated Crystal
       │
       ├──► Formation Energy Model
       │
       ├──► Band Gap Model
       │
       ├──► Bulk Modulus Model
       │
       ├──► Density Model
       │
       └──► Stability Model
```

The outputs are combined into a candidate table.

```text
Crystal | Eg | Eform | Bulk | Density
---------------------------------------
A       | 1.8| -2.7  | 210  | 5.2
B       | 2.4| -1.1  | 180  | 4.1
C       | 1.7| -3.2  | 240  | 5.8
```

Candidate `C` may be more attractive if the design objective favors all four properties.

---

## 24.7.21.33 Example Research Objective

Consider a photovoltaic materials discovery problem.

The desired material should satisfy

```text
1.3 eV ≤ Band Gap ≤ 1.8 eV

Low Formation Energy

Suitable Density

No Severe Structural Instability
```

The complete pipeline becomes

```text
Crystal Dataset

↓

Train Conditional Diffusion Model

↓

Generate 100,000 Crystals

↓

Remove Invalid Structures

↓

Remove Duplicates

↓

Band Gap Prediction

↓

Formation Energy Prediction

↓

Density Filtering

↓

Top 1,000 Candidates

↓

DFT

↓

Top 20 Candidates

↓

Experimental Investigation
```

This is a realistic Materials Informatics workflow.

---

## 24.7.21.34 Conditional Generation with a Materials Dataset

Suppose the training data is stored in a dataframe.

```python
import pandas as pd

df = pd.read_csv(
    "materials_dataset.csv"
)

print(
    df.head()
)
```

A typical dataset may contain

```text
structure

composition

band_gap

formation_energy

density

bulk_modulus
```

The structures are converted into model inputs.

```python
structures = df[
    "structure"
].tolist()

properties = df[
    [
        "band_gap",
        "formation_energy",
        "density",
        "bulk_modulus"
    ]
].values
```

The property matrix can then be normalized.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

properties_scaled = scaler.fit_transform(
    properties
)
```

---

## 24.7.21.35 Dataset Class

A simplified PyTorch dataset is

```python
from torch.utils.data import Dataset


class CrystalDataset(Dataset):

    def __init__(
        self,
        crystals,
        properties
    ):

        self.crystals = crystals

        self.properties = properties

    def __len__(self):

        return len(
            self.crystals
        )

    def __getitem__(self, idx):

        crystal = self.crystals[idx]

        condition = self.properties[idx]

        return {
            "crystal": crystal,
            "condition": condition
        }
```

A production dataset would additionally construct

```text
Node Features

Edge Index

Periodic Offsets

Distances

Lattice Features
```

during preprocessing.

---

## 24.7.21.36 Training Strategy

The conditional diffusion model can be trained using

```text
1. Sample Crystal

2. Sample Noise Level

3. Add Noise

4. Sample Condition

5. Predict Score

6. Calculate Loss

7. Backpropagate

8. Update Model
```

Conceptually,

```python
for batch in dataloader:

    crystal = batch["crystal"]

    condition = batch["condition"]

    sigma = sample_sigma(
        crystal.size(0)
    )

    noise = torch.randn_like(
        crystal
    )

    noisy = (
        crystal
        +
        sigma * noise
    )

    score = model(
        noisy,
        sigma,
        condition
    )

    target = -noise / sigma

    loss = F.mse_loss(
        score,
        target
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

---

## 24.7.21.37 Generation Experiment

After training,

we can select a desired target.

```python
target = torch.tensor([
    1.6,    # band gap
    -3.0,   # formation energy
    5.0,    # density
    200.0   # bulk modulus
])
```

The condition is repeated for a batch.

```python
condition = target.repeat(
    1000,
    1
)
```

Then generate

```python
generated = sample_crystal(
    model=model,
    condition=condition
)
```

The result is a collection of candidate structures.

---

## 24.7.21.38 Important Scientific Warning

The generated structure should **not** be assumed to possess the target property simply because the model was conditioned on it.

For example,

```text
Condition:

Band Gap = 1.6 eV
```

does not guarantee

```text
Actual Band Gap = 1.6 eV
```

The diffusion model has learned a statistical relationship.

Prediction error remains possible.

Therefore,

the condition should be interpreted as a generation preference rather than a physical guarantee.

---

## 24.7.21.39 Surrogate Model Verification

A fast surrogate model can estimate whether the generated structure satisfies the target.

```text
Generated Crystal

↓

Surrogate Predictor

↓

Estimated Property
```

If

```text
Predicted Eg = 1.58 eV
```

for a target of

```text
1.60 eV
```

the candidate can be considered promising.

However,

the surrogate model itself has uncertainty.

Therefore,

high-value candidates should eventually be evaluated using higher-fidelity calculations.

---

## 24.7.21.40 DFT Verification

For the final candidates,

the workflow becomes

```text
Generated Crystal

↓

DFT Geometry Relaxation

↓

SCF Calculation

↓

Electronic Structure

↓

Band Gap

↓

Formation Energy
```

The DFT results provide a stronger test of the generative model.

The difference between the requested and calculated property can also be measured.

```python
error = abs(
    dft_band_gap
    -
    target_band_gap
)
```

---

## 24.7.21.41 Experimental Validation

The ultimate validation is experimental.

A successful pipeline is therefore

```text
AI Generation

↓

ML Screening

↓

DFT

↓

Synthesis

↓

Characterization

↓

Measured Properties
```

This closes the gap between computational generation and real materials discovery.

---

## 24.7.21.42 Feedback from New Data

Experimental results can be added back to the dataset.

```text
Old Dataset

+

New Experimental Data

↓

Updated Dataset

↓

Retrain Model
```

The workflow becomes iterative.

```text
Generate

↓

Screen

↓

Calculate

↓

Experiment

↓

Learn

↓

Generate Again
```

This concept connects generative AI with **active learning**, which will become particularly important in the later chapter on autonomous materials discovery.

---

## 24.7.21.43 Property-Conditioned Generation as an Optimization Problem

Property-conditioned generation can also be viewed as optimization.

Suppose

```text
C
```

is a candidate crystal and

```text
f(C)
```

is a predicted property.

The goal is to find

```text
C*
```

such that

```text
f(C*) ≈ y_target
```

subject to constraints

```text
Chemical Validity

Structural Validity

Thermodynamic Stability

Manufacturability
```

Therefore,

the true problem is

```text
Find Crystal C

such that

Target Property is satisfied

while

Physical Constraints are satisfied.
```

Generative models provide a powerful way of searching this enormous space.

---

## 24.7.21.44 Why Generative Models Are Powerful for Materials

The number of possible materials is enormous.

Even restricting the search to a small set of elements,

the number of possible

```text
Compositions

+

Atomic Arrangements

+

Lattice Configurations
```

is extremely large.

Exhaustive enumeration is impossible.

Generative models learn the statistical structure of known materials and use it to propose new candidates.

```text
Known Materials

↓

Learn Materials Distribution

↓

Generate New Samples

↓

Search Unexplored Regions
```

This is the fundamental reason generative AI is attractive for materials discovery.

---

## 24.7.21.45 The Role of Domain Knowledge

Purely data-driven generation is not sufficient.

Materials science knowledge can dramatically improve the workflow.

Useful constraints include

```text
Element Availability

Oxidation States

Coordination Chemistry

Crystal Symmetry

Density

Formation Energy

Phase Stability

Processing Constraints
```

The best systems therefore combine

```text
Machine Learning

+

Materials Science

+

Physics
```

rather than treating the problem as a purely statistical generation task.

---

## 24.7.21.46 Recommended Research Architecture

For a Materials Informatics research project,

a robust architecture can be organized as

```text
                    ┌───────────────────┐
                    │ Materials Dataset │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Crystal Encoder   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Conditional       │
                    │ Diffusion Model   │
                    └─────────┬─────────┘
                              │
                     Target Properties
                              │
                              ▼
                    ┌───────────────────┐
                    │ Candidate         │
                    │ Generation        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Structural        │
                    │ Filtering         │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ ML Property       │
                    │ Prediction        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ DFT Validation    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Experimental      │
                    │ Validation        │
                    └───────────────────┘
```

This architecture can be adapted to different materials-discovery problems.

---

## 24.7.21.47 Practical Implementation Strategy

A researcher should not begin by attempting to implement the most complicated crystal diffusion model from scratch.

A better progression is

```text
Step 1

Implement Basic Diffusion
```

then

```text
Step 2

Implement Conditional Diffusion
```

then

```text
Step 3

Replace Vector Network with Graph Network
```

then

```text
Step 4

Add Periodic Boundary Conditions
```

then

```text
Step 5

Add Equivariance
```

then

```text
Step 6

Generate Composition + Lattice + Coordinates
```

then

```text
Step 7

Add Property Prediction
```

then

```text
Step 8

Integrate DFT
```

This progression makes debugging and scientific validation considerably easier.

---

## 24.7.21.48 Debugging a Crystal Generator

When a model generates poor structures,

the problem should be isolated systematically.

### Check 1: Dataset

```text
Are the training structures valid?
```

### Check 2: Representation

```text
Are coordinates and lattice correctly encoded?
```

### Check 3: Periodicity

```text
Are periodic neighbors correctly represented?
```

### Check 4: Diffusion

```text
Does the model successfully denoise simple structures?
```

### Check 5: Conditioning

```text
Does changing the target property change generation?
```

### Check 6: Chemistry

```text
Are generated compositions reasonable?
```

### Check 7: Physics

```text
Are candidates stable after relaxation?
```

This layered debugging process is much more reliable than changing the neural network architecture blindly.

---

## 24.7.21.49 Core Research Insight

Property-conditioned generation demonstrates an important principle of Materials Informatics:

```text
The AI model does not discover a material by itself.
```

Instead,

the workflow distributes the task across multiple models and computational methods.

```text
Generative AI
    ↓
Proposes structures

Machine Learning
    ↓
Predicts properties

Physics
    ↓
Tests stability

DFT
    ↓
Provides higher-fidelity calculations

Experiment
    ↓
Tests reality
```

Each component addresses a different part of the scientific problem.

---

## 24.7.21.50 Summary

Property-conditioned crystal generation transforms diffusion models into inverse-design tools.

The essential workflow is

```text
Target Property

↓

Property Embedding

↓

Conditional Diffusion

↓

Candidate Crystal

↓

Structural Validation

↓

Property Prediction

↓

DFT

↓

Experimental Validation
```

The most important concepts are

```text
p(C | y)

Property Conditioning

Classifier Guidance

Classifier-Free Guidance

Multi-Objective Generation

Diversity

Chemical Constraints

Surrogate Prediction

DFT Validation
```

The central research objective is not simply to generate realistic crystals.

It is to generate **novel, physically plausible, and scientifically useful crystals that satisfy a specific materials-design objective**.

This leads naturally to the next major component of generative materials discovery: **evaluating and filtering the enormous number of structures produced by generative models**. In practical research, generation alone is insufficient; candidate ranking, novelty assessment, stability prediction, and computational screening determine which generated structures are worth further investigation.

# Chapter 24 — Generative AI for Materials Discovery

## 24.8 Advanced Crystal Generation

The previous sections established the fundamental idea of generative materials discovery.

A generative model learns a distribution of materials and then samples new candidates from that learned distribution.

For crystal materials, however, generation is considerably more complicated than generating ordinary images, text, or tabular data.

A crystal is not simply a collection of independent atoms.

It contains:

* Atomic species
* Atomic coordinates
* Unit-cell vectors
* Periodic boundary conditions
* Bonding relationships
* Symmetry
* Composition
* Local coordination environments
* Long-range structural interactions

Therefore, a realistic crystal generator must preserve several constraints simultaneously.

The overall problem can be represented as

```text
Composition
     +
Crystal Structure
     +
Periodic Lattice
     +
Chemical Constraints
     +
Physical Constraints
          ↓
   Generative Model
          ↓
   Candidate Crystal
```

The objective is not merely to generate something that looks mathematically valid.

The generated structure should ideally be:

```text
Valid
+
Chemically plausible
+
Structurally meaningful
+
Novel
+
Stable
+
Target-property compatible
```

This distinction is fundamental in Materials Informatics.

A generative model can produce a mathematically valid object that is chemically impossible.

Therefore, **generation and validation must be treated as separate but connected stages of the materials discovery workflow.**

---

## 24.8.1 Why Crystal Generation Is Difficult

Consider a simple crystal containing `N` atoms.

A naïve representation could store the structure as

```text
Atom 1 → (x₁, y₁, z₁)
Atom 2 → (x₂, y₂, z₂)
...
Atom N → (xₙ, yₙ, zₙ)
```

However, this representation is incomplete.

The coordinates are defined relative to a unit cell:

```text
a
b
c
```

where the three lattice vectors define the periodic simulation cell.

Thus, a crystal can be represented approximately as

```text
C = {Z, X, L}
```

where

```text
Z = atomic species

X = atomic coordinates

L = lattice parameters
```

The lattice itself can be written as

```text
L =
[a₁ a₂ a₃
 b₁ b₂ b₃
 c₁ c₂ c₃]
```

The complete generated object therefore contains both discrete and continuous variables.

```text
Atomic species
      ↓
Discrete variables

Atomic coordinates
      ↓
Continuous variables

Lattice parameters
      ↓
Continuous variables
```

This creates an important challenge.

A generative model must learn a distribution over a mixed representation:

```text
p(Z, X, L)
```

or, for conditional generation,

```text
p(Z, X, L | y)
```

where `y` represents desired material properties.

---

## 24.8.2 Crystal Representation Choices

Before implementing a generative model, the researcher must decide how a crystal will be represented.

Several representations are possible.

### Coordinate Representation

The simplest representation is

```text
Atom types
+
Cartesian coordinates
+
Lattice
```

For example,

```python
structure = {
    "species": [
        "Li",
        "Fe",
        "P",
        "O",
        "O",
        "O",
        "O"
    ],

    "coords": [
        [0.00, 0.00, 0.00],
        [0.50, 0.50, 0.50],
        [0.25, 0.25, 0.25],
        [0.10, 0.10, 0.10],
        [0.60, 0.10, 0.10],
        [0.10, 0.60, 0.10],
        [0.10, 0.10, 0.60]
    ],

    "lattice": [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ]
}
```

This representation is intuitive and easy to manipulate.

However, directly generating Cartesian coordinates has several disadvantages.

The model must learn:

* Translation invariance
* Rotation invariance
* Periodicity
* Atom permutation symmetry
* Crystal symmetry

Therefore, direct coordinate generation requires careful architectural design.

---

## 24.8.3 Fractional Coordinates

For periodic crystals, fractional coordinates are often more convenient.

A position is represented relative to the lattice vectors:

```text
r = x a + y b + z c
```

where

```text
0 ≤ x,y,z < 1
```

for atoms inside the conventional fractional representation of the unit cell.

For example,

```python
fractional_coords = [
    [0.00, 0.00, 0.00],
    [0.50, 0.50, 0.50],
    [0.25, 0.25, 0.25]
]
```

The actual Cartesian positions are obtained from the lattice.

Using `pymatgen`:

```python
from pymatgen.core import Lattice
from pymatgen.core import Structure

lattice = Lattice.cubic(5.0)

species = [
    "Li",
    "Fe",
    "P"
]

frac_coords = [
    [0.00, 0.00, 0.00],
    [0.50, 0.50, 0.50],
    [0.25, 0.25, 0.25]
]

structure = Structure(
    lattice,
    species,
    frac_coords
)

print(structure)
```

Fractional coordinates have an important advantage.

The lattice and atomic positions become separate components:

```text
Crystal
   │
   ├── Lattice
   │
   └── Fractional coordinates
```

This makes periodic generation easier to formulate.

---

## 24.8.4 Lattice Representation

A crystal generator must also generate an appropriate lattice.

A lattice can be described using

```text
a
b
c
α
β
γ
```

where

```text
a,b,c
```

are lattice lengths and

```text
α,β,γ
```

are the interaxial angles.

Thus,

```text
L = (a,b,c,α,β,γ)
```

is a compact representation.

For example,

```python
lattice_parameters = torch.tensor([
    5.2,
    5.2,
    7.4,
    90.0,
    90.0,
    120.0
])
```

A generator can therefore contain separate prediction heads:

```text
Latent Representation
        │
        ├──────► Atomic Species
        │
        ├──────► Fractional Coordinates
        │
        └──────► Lattice Parameters
```

This decomposition is particularly useful for modular architectures.

---

## 24.8.5 Variable Number of Atoms

Another major difficulty is that crystals do not necessarily contain the same number of atoms.

One structure may contain

```text
4 atoms
```

while another contains

```text
20 atoms
```

and another may contain

```text
80 atoms
```

A conventional neural network expects tensors with fixed dimensions.

Therefore, the generator must handle variable-sized structures.

Several strategies exist.

### Strategy 1 — Fixed Maximum Number of Atoms

Choose a maximum number:

```text
N_max = 64
```

and pad smaller structures.

For example,

```text
Crystal A → 12 atoms + 52 padding positions

Crystal B → 30 atoms + 34 padding positions
```

A mask identifies real atoms:

```python
mask = torch.tensor([
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 0, 0, 0,
    0, 0, 0, 0, 0
])
```

This approach simplifies batching but introduces unused capacity.

### Strategy 2 — Autoregressive Generation

Atoms can be generated sequentially.

```text
Start
 ↓
Generate atom 1
 ↓
Generate atom 2
 ↓
Generate atom 3
 ↓
...
 ↓
Stop
```

This naturally supports variable-sized structures.

However, sequential generation can be computationally expensive.

### Strategy 3 — Graph Generation

The crystal is represented as a graph.

```text
Atoms → Nodes
Interactions → Edges
```

Graph neural networks naturally support variable numbers of nodes.

This approach becomes particularly important when moving from simple coordinate generation toward crystal graph generation.

---

## 24.8.6 Crystal as a Graph

Let

```text
G = (V,E)
```

represent a crystal graph.

Each node corresponds to an atom.

```text
V = {v₁,v₂,...,vₙ}
```

Each edge represents a local interaction.

```text
E = {(i,j)}
```

A node feature vector may contain:

```text
Atomic number
Period
Group
Electronegativity
Atomic radius
Valence information
```

An edge can contain:

```text
Distance
Periodic image information
Bond-related features
```

A simplified graph representation is therefore

```text
Crystal
   ↓
Graph
   ↓
Nodes + Edges
   ↓
GNN
   ↓
Latent Representation
```

This representation connects naturally with the GNN methods developed earlier in the book.

---

## 24.8.7 Why Graph Representations Help Generative Models

Graph representations solve several problems associated with ordinary coordinate vectors.

A graph does not require a fixed number of atoms.

For example,

```text
Graph A → 8 nodes

Graph B → 20 nodes

Graph C → 50 nodes
```

can all be processed by the same message-passing architecture.

A graph neural network can compute

```text
hᵢ' = UPDATE(
    hᵢ,
    AGGREGATE(
        hⱼ
    )
)
```

where neighboring atoms contribute information to each atomic representation.

For crystal generation, the latent representation can therefore capture both:

```text
Chemical Information
```

and

```text
Structural Information
```

---

## 24.8.8 A General Crystal Generative Architecture

A practical high-level architecture can be written as

```text
Crystal Dataset
      │
      ▼
Crystal Representation
      │
      ▼
Graph / Coordinate Encoder
      │
      ▼
Latent Representation
      │
      ▼
Generative Model
      │
      ├───────────────┐
      │               │
      ▼               ▼
Composition        Structure
      │               │
      └───────┬───────┘
              ▼
        Crystal Decoder
              │
              ▼
       Generated Crystal
              │
              ▼
         Validation
```

This architecture is more realistic for materials generation than treating a crystal as an ordinary feature vector.

---

## 24.8.9 Latent Representation of a Crystal

Suppose a crystal encoder produces

```python
z = encoder(structure)
```

where

```text
z ∈ R^d
```

For example,

```python
z.shape
```

may return

```text
torch.Size([256])
```

meaning that the crystal has been compressed into a 256-dimensional latent representation.

The latent vector might encode information associated with:

```text
Composition
Local environments
Connectivity
Symmetry
Density
Bonding patterns
```

The latent representation does not necessarily correspond directly to a single physical property.

Instead, it is a learned representation optimized for the generative task.

---

## 24.8.10 Crystal Encoder

A simplified encoder can be constructed using a graph neural network.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class CrystalEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim
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
            latent_dim
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        z = self.fc(x)

        return z
```

This encoder performs the following operations:

```text
Atomic Features
      ↓
Graph Convolution
      ↓
Graph Convolution
      ↓
Pooling
      ↓
Latent Vector
```

The pooling operation converts a variable-sized graph into a fixed-dimensional crystal representation.

---

## 24.8.11 Why Global Pooling Is Important

Suppose one crystal contains

```text
10 atoms
```

and another contains

```text
30 atoms
```

After message passing, the node representations have different sizes.

For example,

```text
Crystal A:
10 × 128

Crystal B:
30 × 128
```

A neural network cannot directly compare these tensors as fixed-size crystal representations.

Global pooling solves this problem.

For mean pooling,

```text
z = (1/N) Σ hᵢ
```

where `hᵢ` is the representation of atom `i`.

In PyTorch Geometric:

```python
x = global_mean_pool(
    x,
    batch
)
```

The output becomes

```text
Number of crystals × latent dimension
```

regardless of the number of atoms in each crystal.

---

## 24.8.12 Limitations of Simple Pooling

Global mean pooling is useful but loses structural information.

Consider two crystals:

```text
Crystal A
```

and

```text
Crystal B
```

that contain similar atomic features but very different spatial arrangements.

Mean pooling may produce similar representations.

Therefore,

```text
Mean Pooling
```

does not fully preserve:

* Spatial geometry
* Long-range interactions
* Symmetry
* Exact atomic arrangement

More sophisticated architectures can use:

```text
Attention
+
Edge features
+
3D geometric information
+
Equivariant message passing
```

These methods are especially important for high-quality crystal generation.

---

## 24.8.13 Equivariance in Crystal Generation

A fundamental physical requirement is that a crystal should not change its physical identity when the entire structure is rotated.

Suppose

```text
X
```

is a set of atomic coordinates.

After a rotation,

```text
X' = RX
```

where `R` is a rotation matrix.

A physical property such as formation energy should remain unchanged:

```text
E(X') = E(X)
```

This is rotational invariance.

For vector quantities, however, the output should transform with the coordinates.

This is equivariance.

Therefore, a generative model must distinguish between:

```text
Invariant quantities
```

and

```text
Equivariant quantities
```

This becomes one of the central design principles of modern 3D materials generative models.

---

## 24.8.14 Invariance vs Equivariance

Consider a crystal rotated by a transformation `R`.

For an invariant function:

```text
f(RX) = f(X)
```

For an equivariant function:

```text
f(RX) = Rf(X)
```

For example,

```text
Formation Energy
```

should generally be invariant.

But an atomic displacement vector should transform under rotation.

Therefore,

```text
Energy
→ Invariant

Force
→ Equivariant

Position
→ Equivariant

Atomic species
→ Invariant
```

Understanding this distinction is essential when designing generative architectures.

---

## 24.8.15 Why Equivariance Matters

Without equivariance, a model may need to learn the same physical relationship repeatedly under many different orientations.

For example,

```text
Crystal
```

can be rotated by

```text
0°
30°
60°
90°
...
```

and each orientation may appear different numerically even though it represents the same physical structure.

An equivariant model builds the symmetry into the architecture.

This can provide:

* Better data efficiency
* Better physical consistency
* Improved generalization
* More stable geometric generation

Therefore, modern crystal generation increasingly relies on geometric deep learning rather than ordinary fully connected networks.

---

## 24.8.16 Generating Coordinates with a Neural Network

A simplified generator can predict atomic coordinates from a latent vector.

```python
class CoordinateGenerator(nn.Module):

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        num_atoms
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                num_atoms * 3
            )
        )

        self.num_atoms = num_atoms

    def forward(self, z):

        coords = self.network(z)

        coords = coords.view(
            -1,
            self.num_atoms,
            3
        )

        return coords
```

The output is

```text
Batch × Number of Atoms × 3
```

For example,

```python
z = torch.randn(
    32,
    128
)

generator = CoordinateGenerator(
    latent_dim=128,
    hidden_dim=256,
    num_atoms=20
)

coords = generator(z)

print(coords.shape)
```

Output:

```text
torch.Size([32, 20, 3])
```

This means the model has generated coordinates for

```text
32 crystals
```

with

```text
20 atoms
```

each.

However, this implementation is only a conceptual demonstration.

It does **not** yet enforce periodicity, chemical validity, minimum interatomic distances, symmetry, or rotational equivariance.

Those constraints must be incorporated into a realistic materials generator.

---

## 24.8.17 Generating Atomic Species

Atomic coordinates alone are insufficient.

The model must also determine which elements occupy the generated positions.

Suppose the allowed elements are

```python
elements = [
    "Li",
    "Na",
    "K",
    "Fe",
    "Co",
    "Ni",
    "P",
    "O"
]
```

The model can output logits for each atomic position.

```python
class SpeciesGenerator(nn.Module):

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        num_atoms,
        num_elements
    ):

        super().__init__()

        self.num_atoms = num_atoms

        self.num_elements = num_elements

        self.network = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                num_atoms * num_elements
            )
        )

    def forward(self, z):

        logits = self.network(z)

        logits = logits.view(
            -1,
            self.num_atoms,
            self.num_elements
        )

        return logits
```

During training, the logits can be converted into probabilities:

```python
probabilities = torch.softmax(
    logits,
    dim=-1
)
```

For one atom, the model might predict

```text
Li → 0.05
Na → 0.02
K  → 0.01
Fe → 0.65
Co → 0.15
Ni → 0.08
P  → 0.01
O  → 0.03
```

The model therefore predicts that the atom is most likely to be Fe.

---

## 24.8.18 Joint Crystal Generation

A complete simplified generator can combine

```text
Latent Vector
      │
      ├───────────────┐
      ▼               ▼
Species Head     Coordinate Head
      │               │
      ▼               ▼
Elements         Coordinates
      │               │
      └───────┬───────┘
              ▼
           Lattice
              │
              ▼
        Crystal Structure
```

A conceptual implementation is

```python
class CrystalGenerator(nn.Module):

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        num_atoms,
        num_elements
    ):

        super().__init__()

        self.num_atoms = num_atoms
        self.num_elements = num_elements

        self.shared = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU()
        )

        self.species_head = nn.Linear(
            hidden_dim,
            num_atoms * num_elements
        )

        self.coord_head = nn.Linear(
            hidden_dim,
            num_atoms * 3
        )

        self.lattice_head = nn.Linear(
            hidden_dim,
            6
        )

    def forward(self, z):

        h = self.shared(z)

        species = self.species_head(h)

        coords = self.coord_head(h)

        lattice = self.lattice_head(h)

        species = species.view(
            -1,
            self.num_atoms,
            self.num_elements
        )

        coords = coords.view(
            -1,
            self.num_atoms,
            3
        )

        lattice = lattice.view(
            -1,
            6
        )

        return (
            species,
            coords,
            lattice
        )
```

This architecture illustrates the basic idea of joint crystal generation.

The latent vector contains information about the entire candidate material.

Different output heads translate this latent representation into the structural components.

---

## 24.8.19 Constraining Fractional Coordinates

If fractional coordinates are used, the generated values should generally lie within the unit-cell interval.

A simple approach is to apply a sigmoid:

```python
fractional_coords = torch.sigmoid(
    raw_coords
)
```

This produces

```text
0 < x < 1
0 < y < 1
0 < z < 1
```

For example,

```python
raw_coords = torch.randn(
    4,
    20,
    3
)

fractional_coords = torch.sigmoid(
    raw_coords
)

print(
    fractional_coords.min(),
    fractional_coords.max()
)
```

The output will lie approximately within

```text
[0, 1]
```

However, this alone does not solve periodic duplication.

Two atoms near opposite boundaries may actually be very close through periodic images.

Therefore, distance calculations must use periodic boundary conditions.

---

## 24.8.20 Periodic Distance

For a crystal, the distance between two atoms is not necessarily the ordinary Euclidean distance.

The nearest periodic image must be considered.

Conceptually,

```text
Atom A
  │
  │
  │ periodic boundary
  │
Atom B'
```

where `B'` may be a periodic image of atom B.

The minimum-image distance is therefore

```text
d_min = min |rᵢ - rⱼ + n₁a + n₂b + n₃c|
```

over relevant integer translations.

This is essential for crystal generation because a model might otherwise place atoms apparently far apart while their periodic images overlap.

---

## 24.8.21 Constructing Generated Structures with pymatgen

After generation, the numerical outputs can be converted into a `pymatgen` structure.

```python
from pymatgen.core import Lattice
from pymatgen.core import Structure

lattice = Lattice.from_parameters(
    a=5.2,
    b=5.2,
    c=7.4,
    alpha=90,
    beta=90,
    gamma=120
)

species = [
    "Li",
    "Fe",
    "P",
    "O"
]

coords = [
    [0.00, 0.00, 0.00],
    [0.50, 0.50, 0.50],
    [0.25, 0.25, 0.25],
    [0.10, 0.10, 0.10]
]

structure = Structure(
    lattice,
    species,
    coords
)

print(structure)
```

This conversion is an important bridge between machine learning and conventional computational materials science.

The neural network generates numerical representations.

`pymatgen` then converts those representations into a scientifically meaningful crystal object.

---

## 24.8.22 Structural Validation

The generated structure should never be accepted automatically.

A validation stage should inspect the structure.

A practical pipeline is

```text
Generated Structure
        ↓
Can pymatgen parse it?
        ↓
Composition valid?
        ↓
Atomic distances valid?
        ↓
Density reasonable?
        ↓
Coordination reasonable?
        ↓
Chemical constraints satisfied?
        ↓
Property prediction
```

For example:

```python
def validate_structure(structure):

    if structure is None:
        return False

    if len(structure) == 0:
        return False

    return True
```

This is only the simplest possible validation.

A research implementation should contain much stronger checks.

---

## 24.8.23 Minimum Interatomic Distance

One of the simplest structural filters is a minimum-distance constraint.

Suppose

```text
d_min = 1.2 Å
```

is selected as a conservative threshold for a particular screening task.

A candidate can be rejected if

```text
min(i,j) dᵢⱼ < d_min
```

A simplified implementation can inspect pairwise distances:

```python
import numpy as np


def minimum_distance(structure):

    distances = []

    for i in range(
        len(structure)
    ):

        for j in range(
            i + 1,
            len(structure)
        ):

            d = structure.get_distance(
                i,
                j,
                jimage=None
            )

            distances.append(d)

    return min(distances)
```

Then:

```python
d_min = minimum_distance(
    structure
)

if d_min < 1.2:

    print(
        "Structure rejected"
    )
```

For periodic crystals, however, the distance calculation must account properly for periodic images. A production implementation should use the periodic-distance functionality provided by the crystal-structure toolkit rather than relying on a naïve nonperiodic calculation.

---

## 24.8.24 Density Screening

Density is another useful first-stage filter.

For a structure,

```text
ρ = mass / volume
```

`pymatgen` can calculate the density:

```python
density = structure.density

print(
    "Density:",
    density,
    "g/cm³"
)
```

A researcher may impose a broad screening interval:

```python
if (
    density < 0.5
    or density > 15.0
):

    reject = True
```

The exact limits should be selected according to the chemical system.

The important principle is that **validation thresholds should be scientifically justified rather than chosen arbitrarily.**

---

## 24.8.25 Structure Validity Is Not Stability

A crucial distinction must be made between

```text
Structural Validity
```

and

```text
Thermodynamic Stability
```

A structure can be geometrically valid while being highly unstable.

For example,

```text
Valid coordinates
        ↓
Valid lattice
        ↓
No atomic overlap
```

does not imply

```text
Low formation energy
```

or

```text
Stable against decomposition
```

Therefore, the screening pipeline should be

```text
Geometry Validation
        ↓
Chemical Validation
        ↓
ML Property Prediction
        ↓
Thermodynamic Screening
        ↓
DFT Validation
```

This distinction is critical in real materials discovery.

---

## 24.8.26 Generative Models as Candidate Generators

A useful conceptual change is to stop viewing a generative model as a machine that produces the final answer.

Instead,

```text
Generative Model
      ↓
Candidate Generator
      ↓
Large Candidate Pool
      ↓
Scientific Screening
      ↓
High-Quality Candidates
```

For example, a model may generate

```text
100,000
```

structures.

After structural validation:

```text
100,000
 ↓
70,000
```

After chemical filtering:

```text
70,000
 ↓
30,000
```

After property prediction:

```text
30,000
 ↓
2,000
```

After stability screening:

```text
2,000
 ↓
100
```

After DFT:

```text
100
 ↓
10
```

These final candidates may then be considered for experimental synthesis.

This staged reduction is one of the most practical ways to use generative AI in computational materials research.

---

## 24.8.27 Generation-to-DFT Workflow

The complete computational workflow can therefore be written as

```text
Materials Dataset
       ↓
Representation Learning
       ↓
Train Generative Model
       ↓
Sample New Crystals
       ↓
Structural Validation
       ↓
Chemical Filtering
       ↓
ML Property Prediction
       ↓
Stability Screening
       ↓
Candidate Ranking
       ↓
DFT Calculations
       ↓
Relaxed Structures
       ↓
Accurate Properties
       ↓
Experimental Candidates
```

The DFT stage is particularly important because machine-learning predictions are ultimately approximations.

A generated structure that looks promising according to an ML surrogate should be subjected to higher-fidelity calculations before being treated as a serious materials candidate.

---

## 24.8.28 The Role of Structural Relaxation

A generated structure may not initially correspond to a local energy minimum.

Therefore,

```text
Generated Crystal
       ↓
DFT Relaxation
       ↓
Relaxed Crystal
```

is often necessary.

The relaxation process modifies

```text
Atomic positions
```

and potentially

```text
Lattice parameters
```

to reduce the total energy.

A candidate can therefore be evaluated before and after relaxation.

```text
Generated Structure
       │
       ├── ML prediction
       │
       └── DFT relaxation
                ↓
          Relaxed Structure
                ↓
          Final properties
```

This provides an important test of whether the generated material corresponds to a physically meaningful configuration.

---

## 24.8.29 Generated vs Relaxed Structure

Suppose a model generates

```text
Crystal A
```

with predicted formation energy

```text
-2.8 eV/atom
```

After DFT relaxation, the structure may become

```text
Crystal A'
```

with

```text
-2.4 eV/atom
```

The difference indicates that the original structure was not fully optimized.

Therefore, generative-model evaluation should distinguish between:

```text
As-generated properties
```

and

```text
Relaxed properties
```

This distinction becomes especially important when comparing generative models.

---

## 24.8.30 Research-Level Generative Pipeline

A more complete research architecture is

```text
                         Materials Dataset
                                │
                                ▼
                      Structure Representation
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        Crystal Encoder                  Property Encoder
                │                               │
                └───────────────┬───────────────┘
                                ▼
                         Latent Representation
                                │
                                ▼
                      Conditional Generator
                                │
                                ▼
                       Generated Crystals
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        Geometry Check    Chemical Check    Duplicate Check
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                         ML Surrogate Models
                                │
                                ▼
                        Candidate Ranking
                                │
                                ▼
                              DFT
                                │
                                ▼
                         Relaxed Structures
                                │
                                ▼
                       Final Candidate Set
```

This architecture integrates nearly every major component developed throughout the Materials Informatics book:

```text
Materials Science
+
Machine Learning
+
Deep Learning
+
Graph Neural Networks
+
Generative Modeling
+
pymatgen
+
DFT
```

The generative model is therefore not an isolated algorithm.

It becomes one component inside a complete computational materials discovery system.

## 24.8.31 From Candidate Generation to Materials Discovery

The most important idea in generative materials discovery is that **generation is only the beginning of the discovery process**.

A generative model does not automatically produce experimentally useful materials.

Instead, it produces a large search space of possible candidates.

The scientific workflow then determines which candidates deserve further investigation.

The complete process can therefore be represented as

```text
Generative Model
       ↓
Candidate Structures
       ↓
Validity Screening
       ↓
Chemical Screening
       ↓
Property Prediction
       ↓
Stability Screening
       ↓
Ranking
       ↓
DFT
       ↓
Experimental Validation
```

This creates a fundamental distinction between

```text
Generation
```

and

```text
Discovery
```

Generation asks:

```text
What structures can the model produce?
```

Discovery asks:

```text
Which generated structures are scientifically useful?
```

A successful materials-generation system must address both questions.

---

## 24.8.32 Candidate Ranking

After generating a large number of structures, the candidates must be ranked.

Suppose the model produces

```python
candidates = generate_crystals(
    model,
    num_samples=10000
)
```

Each candidate can then be evaluated using several predictive models.

```python
band_gap = band_gap_model(
    candidates
)

formation_energy = formation_energy_model(
    candidates
)

bulk_modulus = bulk_modulus_model(
    candidates
)
```

The results can be stored in a table.

```python
results = pd.DataFrame({

    "band_gap": band_gap,

    "formation_energy":
        formation_energy,

    "bulk_modulus":
        bulk_modulus
})
```

The resulting dataset might look like

```text
Candidate    Band Gap    Formation Energy    Bulk Modulus
----------------------------------------------------------
C001         1.72        -3.14               218
C002         0.84        -2.10               145
C003         1.91        -3.52               260
C004         2.83        -1.74               120
```

The ranking procedure depends on the scientific objective.

For example, a photovoltaic material may require a particular band-gap range rather than simply maximizing band gap.

Therefore, ranking should reflect the actual research objective.

---

## 24.8.33 Target Distance

For a desired property, one simple ranking method is to calculate the distance between the predicted property and the target.

Suppose the desired band gap is

```text
Eg,target = 1.8 eV
```

For candidate `i`,

```text
dᵢ = |Eg,i - Eg,target|
```

A smaller value indicates a closer match.

In Python:

```python
target_band_gap = 1.8

results["band_gap_error"] = (
    results["band_gap"]
    - target_band_gap
).abs()
```

The candidates can then be sorted:

```python
ranked = results.sort_values(
    "band_gap_error"
)
```

The best candidates are those with the smallest target error.

This simple strategy is useful when the objective is to approach a specific target rather than maximize or minimize a property.

---

## 24.8.34 Constraint-Based Screening

Often, a researcher does not need an exact target.

Instead, the material must satisfy a set of constraints.

For example,

```text
1.5 eV ≤ Band Gap ≤ 2.0 eV

Formation Energy < -2.5 eV/atom

Bulk Modulus > 180 GPa
```

This can be implemented directly.

```python
mask = (

    (results["band_gap"] >= 1.5)

    &

    (results["band_gap"] <= 2.0)

    &

    (results["formation_energy"] < -2.5)

    &

    (results["bulk_modulus"] > 180)
)

selected = results[
    mask
]
```

The resulting dataset contains only candidates satisfying all constraints.

This approach is often more scientifically meaningful than ranking according to a single numerical score.

---

## 24.8.35 Soft Constraints

Hard constraints can sometimes be too restrictive.

Suppose the desired band gap is

```text
1.8 eV
```

A candidate with

```text
1.79 eV
```

may be excellent.

A candidate with

```text
1.81 eV
```

may also be excellent.

However, a strict equality condition would reject both.

Soft constraints provide a more flexible alternative.

For example,

```python
band_gap_error = (
    results["band_gap"]
    - 1.8
).abs()

results["band_gap_score"] = (
    torch.exp(
        -band_gap_error
    )
)
```

The closer a candidate is to the target, the larger its score.

In practice, the exact scoring function should be selected according to the scientific problem.

---

## 24.8.36 Multi-Objective Candidate Scoring

Suppose the research objective contains three properties:

```text
Band Gap
Formation Energy
Bulk Modulus
```

A normalized score can be constructed.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled = scaler.fit_transform(
    results[
        [
            "band_gap",
            "formation_energy",
            "bulk_modulus"
        ]
    ]
)
```

The normalized values can then be combined.

For example,

```python
results["score"] = (

    0.4 * results["band_gap_score"]

    +

    0.4 * results["formation_energy_score"]

    +

    0.2 * results["bulk_modulus_score"]
)
```

The weights represent the relative importance of each objective.

However, weighted scoring should not replace proper multi-objective analysis when the objectives strongly conflict.

---

## 24.8.37 Pareto-Based Candidate Selection

For multi-objective materials discovery, Pareto optimality provides a more principled strategy.

Suppose the researcher wants

```text
Low Formation Energy
```

and

```text
High Bulk Modulus
```

Candidate A is dominated if another candidate is

```text
Lower in Formation Energy
```

and

```text
Higher in Bulk Modulus
```

at the same time.

A Pareto-optimal candidate cannot be improved in one objective without sacrificing at least one other objective.

The workflow becomes

```text
Generated Candidates
        ↓
Property Prediction
        ↓
Multi-Objective Evaluation
        ↓
Remove Dominated Candidates
        ↓
Pareto Frontier
```

The remaining candidates represent scientifically meaningful trade-offs.

---

## 24.8.38 Diversity-Preserving Selection

Selecting only the top-scoring candidates can create another problem.

Suppose the top 100 candidates are all structurally almost identical.

The discovery process has then explored only a tiny region of materials space.

Therefore, candidate selection should often balance

```text
Property Quality
```

with

```text
Structural Diversity
```

A useful strategy is

```text
High Property Score
+
High Structural Diversity
```

rather than

```text
Highest Property Score Only
```

For example:

```python
selected = []

for candidate in ranked_candidates:

    if sufficiently_different(
        candidate,
        selected
    ):

        selected.append(
            candidate
        )
```

The exact definition of `sufficiently_different` depends on the structural representation.

Possible representations include:

```text
Composition
Crystal Fingerprint
Graph Embedding
SOAP Descriptor
Latent Embedding
```

---

## 24.8.39 Structural Fingerprints

A crystal fingerprint converts a structure into a numerical representation suitable for similarity calculations.

Conceptually,

```text
Crystal
   ↓
Fingerprint
   ↓
Similarity
```

Suppose

```python
fp_a = fingerprint(
    crystal_a
)

fp_b = fingerprint(
    crystal_b
)
```

A similarity measure can then be calculated.

```python
similarity = compute_similarity(
    fp_a,
    fp_b
)
```

Highly similar candidates can be grouped together.

This is useful for detecting:

* Duplicates
* Near-duplicates
* Structural families
* Novel structures
* Mode collapse

---

## 24.8.40 Latent-Space Analysis

Generative models often provide a latent representation.

Suppose

```python
z = encoder(
    structures
)
```

produces

```text
N × d
```

latent vectors.

These vectors can be visualized after dimensionality reduction.

```text
Crystal Dataset
      ↓
Encoder
      ↓
Latent Space
      ↓
PCA / t-SNE / UMAP
      ↓
2D Visualization
```

The visualization can reveal whether generated structures occupy:

```text
Known Materials Space
```

or extend into

```text
New Chemical / Structural Regions
```

This provides a useful method for analyzing the behavior of a generative model.

---

## 24.8.41 PCA of Generated Crystal Embeddings

PCA can be applied to latent vectors.

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=2
)

z_2d = pca.fit_transform(
    z
)
```

The result can be visualized.

```python
import matplotlib.pyplot as plt

plt.scatter(
    z_2d[:, 0],
    z_2d[:, 1]
)

plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.title(
    "Generated Crystal Latent Space"
)

plt.show()
```

If both training and generated structures are embedded using the same encoder, their distributions can be compared.

```text
Training Materials
        ● ● ●
      ● ● ● ●
    ● ● ●

Generated Materials
          ○ ○
        ○ ○ ○
          ○
```

A generator that produces candidates far outside the learned materials distribution may require additional validation.

---

## 24.8.42 Detecting Mode Collapse

Mode collapse occurs when a generative model repeatedly produces similar outputs.

For example,

```text
Sample 1 → Structure A
Sample 2 → Structure A'
Sample 3 → Structure A
Sample 4 → Structure A'
...
```

Even if the generated structures satisfy the target property, this may represent poor generative diversity.

A simple diagnostic is to calculate pairwise similarity.

```python
similarities = compute_pairwise_similarity(
    generated_structures
)
```

If the average similarity is extremely high, the model may be suffering from insufficient diversity.

Therefore, a generative model should be evaluated using both:

```text
Validity
```

and

```text
Diversity
```

---

## 24.8.43 Novelty Analysis

A generated structure should ideally not simply reproduce a training example.

Novelty can be evaluated by comparing generated structures against the training dataset.

The workflow is

```text
Generated Crystal
       ↓
Compare with Training Dataset
       ↓
Similarity
       ↓
Novel / Known / Near-Duplicate
```

For example,

```python
nearest_similarity = find_nearest_training_similarity(
    generated_structure,
    training_structures
)
```

A candidate with extremely high similarity to a known structure may not represent meaningful discovery.

However, novelty alone is not sufficient.

A completely novel but chemically impossible structure is not useful.

Therefore,

```text
Novelty
+
Validity
+
Stability
+
Desired Properties
```

must be considered together.

---

## 24.8.44 Chemical Space Coverage

A strong generative model should ideally explore regions of materials space that are relevant but not excessively represented in the training data.

For example,

```text
Training Dataset

A A A A
B B B
C C
D
```

A generator might produce

```text
A A
B B
C C
D D
E
F
```

The new `E` and `F` regions may indicate exploration beyond the original distribution.

However, extrapolation is inherently risky.

The further a candidate moves from the training distribution, the greater the uncertainty of the predictive models may become.

This connects generative modeling directly to uncertainty quantification, which will be discussed later in the book.

---

## 24.8.45 Surrogate Models for Generated Materials

DFT calculations are computationally expensive.

Therefore, generative workflows often use machine-learning surrogate models.

The architecture becomes

```text
Generated Crystal
       ↓
Crystal Representation
       ↓
ML Property Predictor
       ↓
Predicted Property
```

For example,

```python
predicted_energy = energy_model(
    crystal_graph
)
```

and

```python
predicted_band_gap = band_gap_model(
    crystal_graph
)
```

The surrogate model may evaluate thousands or millions of candidates much faster than direct DFT.

This enables large-scale computational screening.

---

## 24.8.46 Why Surrogate Error Matters

A predictive model is not perfect.

Suppose the actual band gap is

```text
2.1 eV
```

but the surrogate predicts

```text
1.7 eV
```

The candidate may incorrectly pass or fail a design constraint.

Therefore, generated-material screening should ideally include uncertainty estimates.

Instead of predicting only

```text
Eg = 1.7 eV
```

the model could provide

```text
Eg = 1.7 ± 0.2 eV
```

This allows the researcher to distinguish between confident and uncertain predictions.

This concept becomes particularly important when generated candidates lie outside the training distribution.

---

## 24.8.47 DFT as the High-Fidelity Filter

After ML screening, a smaller candidate set can be sent to DFT.

For example,

```text
1,000,000 generated structures
            ↓
ML screening
            ↓
10,000 candidates
            ↓
Advanced ML screening
            ↓
500 candidates
            ↓
DFT
            ↓
50 promising materials
```

The role of DFT is not necessarily to replace the machine-learning model.

Instead,

```text
ML
→ High-throughput approximate screening

DFT
→ High-fidelity validation
```

This combination provides a practical balance between computational cost and scientific accuracy.

---

## 24.8.48 DFT Feedback

The workflow can become iterative.

Suppose the generator produces candidates:

```text
C1
C2
C3
...
C100
```

DFT calculations provide accurate properties.

These new results can then be added to the training dataset.

```text
Original Dataset
      +
DFT Results
      ↓
Updated Dataset
      ↓
Retrain Model
      ↓
Improved Generator
```

This creates a feedback loop.

```text
Generate
   ↓
Screen
   ↓
DFT
   ↓
Add Data
   ↓
Retrain
   ↓
Generate Again
```

This idea connects generative AI with active learning and autonomous materials discovery.

---

## 24.8.49 Complete Generative Materials Discovery Loop

The full system can now be expressed as

```text
                    Materials Dataset
                           │
                           ▼
                  Representation Learning
                           │
                           ▼
                   Generative Model
                           │
                           ▼
                  Generate Candidates
                           │
                           ▼
                 Structural Validation
                           │
                           ▼
                  Chemical Validation
                           │
                           ▼
                 Duplicate Removal
                           │
                           ▼
                 Property Prediction
                           │
                           ▼
                 Uncertainty Analysis
                           │
                           ▼
                  Candidate Ranking
                           │
                           ▼
                         DFT
                           │
                           ▼
                  Experimental Testing
                           │
                           ▼
                    New Materials Data
                           │
                           └───────────────┐
                                           │
                                           ▼
                                  Model Improvement
                                           │
                                           └──────►
```

This is the foundation of a closed-loop generative materials discovery system.

---

## 24.8.50 Implementation Strategy

A researcher should not attempt to build the entire system simultaneously.

A practical implementation can be developed in stages.

### Stage 1 — Dataset

Prepare a dataset containing:

```text
Structure
Composition
Properties
```

For example:

```python
df = pd.read_csv(
    "materials_dataset.csv"
)
```

---

### Stage 2 — Structure Processing

Convert structures into a consistent representation.

```python
from pymatgen.core import Structure

structures = []

for text in df["structure"]:

    structure = Structure.from_str(
        text,
        fmt="cif"
    )

    structures.append(
        structure
    )
```

---

### Stage 3 — Crystal Encoding

Convert each crystal into a machine-learning representation.

```text
Crystal
   ↓
Pymatgen
   ↓
Graph
   ↓
Node Features
   ↓
Edge Features
```

---

### Stage 4 — Train Representation Model

The encoder learns a compact representation.

```python
z = encoder(
    crystal_graph
)
```

---

### Stage 5 — Train Generative Model

The generative model learns the distribution of crystal representations.

```python
loss = diffusion_loss(
    noisy_crystal,
    time,
    condition
)
```

---

### Stage 6 — Generate Candidates

```python
generated = sample(
    model,
    num_samples=10000
)
```

---

### Stage 7 — Validate

```python
valid = []

for structure in generated:

    if validate_structure(
        structure
    ):

        valid.append(
            structure
        )
```

---

### Stage 8 — Predict Properties

```python
properties = predictor(
    valid
)
```

---

### Stage 9 — Rank

```python
ranked = rank_candidates(
    valid,
    properties
)
```

---

### Stage 10 — DFT

The highest-ranked candidates are converted into DFT input structures.

```text
Generated Crystal
       ↓
POSCAR / CIF
       ↓
DFT Calculation
       ↓
Relaxation
       ↓
Electronic / Mechanical / Magnetic Properties
```

---

## 24.8.51 Practical Research Architecture

A practical implementation may eventually contain separate modules:

```text
materials_generation/
│
├── data/
│   ├── structures/
│   ├── properties/
│   └── processed/
│
├── representations/
│   ├── crystal_graph.py
│   ├── node_features.py
│   └── edge_features.py
│
├── models/
│   ├── encoder.py
│   ├── vae.py
│   ├── diffusion.py
│   ├── conditional.py
│   └── predictor.py
│
├── generation/
│   ├── sampler.py
│   ├── constraints.py
│   └── validation.py
│
├── screening/
│   ├── properties.py
│   ├── ranking.py
│   └── diversity.py
│
├── dft/
│   ├── input_generation.py
│   ├── relaxation.py
│   └── parsing.py
│
└── experiments/
    ├── train.py
    ├── generate.py
    └── evaluate.py
```

This modular organization becomes increasingly important as the project grows.

A research codebase should separate:

```text
Data
Models
Generation
Validation
Screening
DFT
Evaluation
```

rather than placing the entire workflow into a single Python script.

---

## 24.8.52 A Minimal End-to-End Prototype

The following simplified pseudocode illustrates the complete concept.

```python
# Load data

dataset = load_materials(
    "materials_dataset.csv"
)


# Encode structures

representations = encoder(
    dataset.structures
)


# Train generative model

for epoch in range(
    num_epochs
):

    loss = train_step(
        representations,
        dataset.properties
    )


# Generate candidates

candidates = generate(
    model,
    num_samples=10000
)


# Validate

valid = [
    crystal
    for crystal in candidates
    if validate(crystal)
]


# Predict properties

predictions = predictor(
    valid
)


# Filter

selected = filter_candidates(
    valid,
    predictions
)


# Rank

ranked = rank_candidates(
    selected
)


# Select top candidates

top_candidates = ranked[
    :100
]


# DFT

run_dft(
    top_candidates
)
```

Although simplified, this code captures the central architecture of generative materials discovery.

The difficult part of research implementation is not writing the individual function calls.

The difficult part is ensuring that every component is scientifically valid and communicates correctly with the next component.

---

## 24.8.53 Key Lessons

Advanced crystal generation requires much more than a neural network capable of producing coordinates.

A practical system must consider:

```text
Crystal Representation
+
Variable Number of Atoms
+
Periodic Boundary Conditions
+
Chemical Validity
+
Geometric Validity
+
Symmetry
+
Equivariance
+
Property Conditioning
+
Diversity
+
Novelty
+
Uncertainty
+
DFT Validation
```

The most important conceptual workflow is therefore

```text
Generate
   ↓
Validate
   ↓
Predict
   ↓
Rank
   ↓
Verify
```

rather than

```text
Generate
   ↓
Done
```

This distinction separates a demonstration of generative AI from a genuine Materials Informatics research workflow.

The ultimate objective is not to generate the largest number of crystals.

It is to efficiently navigate an enormous materials design space and identify a small number of candidates that are:

```text
Chemically plausible
+
Structurally valid
+
Novel
+
Stable
+
Property-compatible
+
Computationally verifiable
+
Experimentally meaningful
```

The next major step is therefore to examine **how modern generative models represent periodic crystal structures and how diffusion processes can operate directly on crystal geometry**.

## 24.8.54 Crystal-Specific Constraints in Generative Models

The previous sections described the general generative materials-discovery pipeline.

However, crystal generation introduces constraints that do not appear in ordinary molecular or image generation.

A crystal is not simply a collection of atoms.

It contains

```text
Atoms
+
Periodic Lattice
+
Atomic Coordinates
+
Chemical Species
+
Symmetry
+
Interatomic Distances
+
Coordination Environment
```

Therefore, a generative model must produce structures that are simultaneously valid in several different spaces.

Conceptually,

```text
Latent Representation
        ↓
Crystal Generation
        ↓
┌─────────────────────────┐
│ Composition             │
│ Lattice                 │
│ Coordinates             │
│ Periodicity             │
│ Geometry                │
│ Chemistry               │
│ Symmetry                │
└─────────────────────────┘
        ↓
Valid Crystal
```

A model that generates reasonable atomic coordinates but an invalid lattice has failed.

Similarly, a model that generates a chemically impossible composition has also failed.

Therefore, crystal generation should be treated as a **constrained generative modeling problem**.

---

## 24.8.55 Periodic Boundary Conditions

The most important difference between a finite molecular structure and a crystal is periodicity.

A crystal can be represented by a unit cell:

```text
Lattice
+
Atomic fractional coordinates
+
Atomic species
```

The unit cell is periodically repeated in three dimensions.

If the lattice vectors are

```text
a
b
c
```

then a periodic atomic position can be written as

```text
r = x a + y b + z c
```

where

```text
0 ≤ x,y,z < 1
```

are fractional coordinates.

This representation is particularly convenient for machine learning because the structure can be reconstructed from the lattice and fractional coordinates.

In Python:

```python
from pymatgen.core import Lattice
from pymatgen.core import Structure

lattice = Lattice.cubic(
    5.5
)

structure = Structure(
    lattice,

    ["Si", "Si"],

    [
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25]
    ]
)

print(structure)
```

The model therefore needs to generate both

```text
Lattice
```

and

```text
Atomic Positions
```

rather than generating Cartesian coordinates alone.

---

## 24.8.56 Lattice Generation

The lattice can be described by six conventional parameters:

```text
a
b
c
α
β
γ
```

where

```text
a, b, c
```

are lattice lengths and

```text
α, β, γ
```

are lattice angles.

A simple generative model could therefore predict

```python
lattice_parameters = model(
    latent_vector
)
```

with an output such as

```text
[a, b, c, alpha, beta, gamma]
```

However, unconstrained generation can produce physically unreasonable values.

For example:

```text
a = -2.0 Å
```

is impossible.

Similarly,

```text
alpha = 0°
```

would produce a degenerate cell.

Therefore, lattice generation requires constraints.

---

## 24.8.57 Constraining Lattice Parameters

One simple approach is to transform unconstrained neural-network outputs.

Suppose the network produces

```python
raw_a = model_output[:, 0]
```

A positive lattice parameter can be obtained using a positive-valued transformation.

```python
import torch

a = torch.nn.functional.softplus(
    raw_a
)
```

The same procedure can be applied to

```text
b
c
```

For angles, a bounded transformation can be used.

```python
raw_alpha = model_output[:, 3]

alpha = (
    torch.sigmoid(raw_alpha)
    * 180.0
)
```

However, this alone does not guarantee a valid three-dimensional lattice.

A production-quality implementation should therefore validate the resulting lattice matrix.

---

## 24.8.58 Lattice Matrix Representation

Instead of directly generating six lattice parameters, a model can generate the lattice matrix.

The matrix can be written as

```text
L =
| a₁  a₂  a₃ |
| b₁  b₂  b₃ |
| c₁  c₂  c₃ |
```

The determinant represents the unit-cell volume.

```text
V = |det(L)|
```

A valid crystal requires a positive, nonzero volume.

In Python:

```python
volume = torch.det(
    lattice_matrix
).abs()
```

A very small volume indicates an invalid or highly compressed structure.

Therefore:

```python
valid_volume = (
    volume > minimum_volume
)
```

can be used as a first screening condition.

---

## 24.8.59 Atomic Overlap

One of the most common failure modes in unconstrained crystal generation is atomic overlap.

Suppose two generated atoms have

```text
distance = 0.2 Å
```

This is generally physically unreasonable.

Therefore, a minimum-distance constraint is useful.

For atoms `i` and `j`:

```text
dᵢⱼ > d_min
```

The exact threshold depends on the atomic species.

A simple implementation is:

```python
def valid_distances(
    structure,
    minimum_distance=1.0
):

    distances = structure.distance_matrix

    mask = (
        distances > minimum_distance
    )

    return mask.all()
```

A more chemically meaningful implementation would use species-dependent minimum distances.

For example:

```python
def minimum_distance(
    element_a,
    element_b
):
    ...
```

could estimate a physically reasonable lower bound based on atomic radii or learned statistics.

---

## 24.8.60 Coordination Constraints

Distance validity alone is insufficient.

A structure may contain reasonable interatomic distances while producing chemically unreasonable coordination environments.

For example:

```text
Generated Structure

Metal
 ├── O
 ├── O
 ├── O
 ├── O
 ├── O
 ├── O
 ├── O
 └── O
```

An eight-coordinate environment may be valid for some elements but completely unreasonable for others.

Therefore, coordination should be analyzed after generation.

A simplified workflow is:

```text
Generated Structure
       ↓
Neighbor Finding
       ↓
Coordination Number
       ↓
Chemical Plausibility
```

With pymatgen, local environments can be analyzed using structure-analysis tools.

The important principle is that the validity criteria should depend on the chemistry of the generated structure.

---

## 24.8.61 Composition Constraints

Composition is another major constraint.

Suppose the model is intended to generate compounds in the chemical system

```text
Li-Fe-P-O
```

The generator should not suddenly produce

```text
Li-Fe-P-O-Cl-N
```

unless such elements are explicitly allowed.

A simple composition filter can therefore be implemented.

```python
allowed_elements = {
    "Li",
    "Fe",
    "P",
    "O"
}

def valid_elements(
    structure
):

    elements = {
        str(site.specie)
        for site in structure
    }

    return elements.issubset(
        allowed_elements
    )
```

This provides a basic chemical-space constraint.

---

## 24.8.62 Stoichiometric Constraints

Sometimes the allowed elements are known but the stoichiometry must also be constrained.

For example, the researcher may want compounds approximately belonging to

```text
LiFePO4
```

The generated composition should then preserve an appropriate elemental relationship.

Composition can be represented as a vector:

```text
Li   Fe   P   O

1    1    1   4
```

A generative model could produce composition logits:

```python
composition_logits = model(
    latent
)
```

which are converted into predicted elemental proportions.

For example:

```python
composition_probabilities = torch.softmax(
    composition_logits,
    dim=-1
)
```

These probabilities can then be converted into candidate stoichiometries.

---

## 24.8.63 Composition and Structure Should Not Be Separated Completely

A major challenge is that composition and structure are coupled.

For example, changing

```text
Na → Li
```

can alter:

```text
Atomic Size
Bond Length
Coordination
Lattice Parameters
Symmetry
Electronic Structure
```

Therefore, a model that independently generates composition and structure may produce inconsistent combinations.

Conceptually:

```text
Composition
     │
     ├──────────────┐
     ▼              ▼
Lattice         Coordination
     │              │
     └──────┬───────┘
            ▼
      Crystal Structure
```

This motivates joint generative models.

---

## 24.8.64 Joint Crystal Representation

A more advanced model can represent the complete crystal as

```text
C = {
    composition,
    lattice,
    coordinates
}
```

The generative objective then becomes

```text
p(C)
=
p(
    composition,
    lattice,
    coordinates
)
```

For conditional generation:

```text
p(
    composition,
    lattice,
    coordinates
    |
    target_properties
)
```

This is a much richer formulation than generating only atomic positions.

---

## 24.8.65 Graph Representation of Generated Crystals

Once a crystal has been generated, it can be converted into a graph.

```text
Atoms → Nodes

Interactions → Edges
```

For example:

```text
       O
      / \
     /   \
    Fe---O
     \
      O
```

The graph representation allows a GNN to evaluate the structure.

A typical workflow becomes:

```text
Generated Crystal
       ↓
Neighbor Search
       ↓
Crystal Graph
       ↓
GNN
       ↓
Property Prediction
```

This creates a natural connection between the generative models discussed in this chapter and the GNN architectures developed earlier in the book.

---

## 24.8.66 Periodic Neighbor Construction

For a crystal graph, neighbors must be determined under periodic boundary conditions.

A naive Euclidean distance calculation using only atoms inside the unit cell is insufficient.

An atom near one boundary may interact with an atom across the opposite boundary.

Therefore:

```text
Unit Cell
┌───────────────┐
│ A             │
│               │
│               │
│             B │
└───────────────┘

Periodic copies:

... A | Cell | B ...
```

The neighbor search must account for periodic images.

This is one reason why crystal graph construction is more complicated than ordinary molecular graph construction.

Pymatgen provides tools for periodic neighbor analysis.

For example:

```python
neighbors = structure.get_neighbors(
    structure[0],
    r=5.0
)
```

The returned neighbors can then be converted into graph edges.

---

## 24.8.67 Structure Validation Pipeline

A practical generated-crystal validator can combine multiple checks.

```python
def validate_crystal(
    structure
):

    if not valid_lattice(
        structure
    ):
        return False

    if not valid_elements(
        structure
    ):
        return False

    if not valid_distances(
        structure
    ):
        return False

    if not valid_density(
        structure
    ):
        return False

    if not valid_coordination(
        structure
    ):
        return False

    return True
```

The generation workflow then becomes:

```python
valid_candidates = []

for crystal in generated:

    if validate_crystal(
        crystal
    ):

        valid_candidates.append(
            crystal
        )
```

This simple structure makes the validation system modular.

Each scientific constraint can be improved independently.

---

## 24.8.68 Density Screening

Generated structures may contain unrealistic densities.

Density is calculated from mass and unit-cell volume:

```text
ρ = m / V
```

Pymatgen can calculate density directly.

```python
density = structure.density
```

A simple filter could be:

```python
def valid_density(
    structure,
    minimum=0.5,
    maximum=20.0
):

    density = structure.density

    return (
        minimum
        <= density
        <= maximum
    )
```

The actual limits should not be chosen arbitrarily for a research study.

They should be based on the relevant chemical domain.

For example, density limits appropriate for organic crystals may be inappropriate for heavy inorganic compounds.

---

## 24.8.69 Symmetry Considerations

Crystal symmetry provides another important source of information.

Many crystals belong to recognized space groups.

A generative model may therefore be conditioned on a desired crystal system or space group.

For example:

```text
Cubic
Tetragonal
Orthorhombic
Monoclinic
Triclinic
```

The generation process could become

```text
Target Property
      +
Crystal System
      ↓
Generative Model
      ↓
Crystal
```

This is useful when experimental or physical constraints strongly favor a particular symmetry.

---

## 24.8.70 Symmetry Validation

After generation, symmetry can be analyzed.

Pymatgen provides access to symmetry analysis through `SpacegroupAnalyzer`.

```python
from pymatgen.symmetry.analyzer import (
    SpacegroupAnalyzer
)

analyzer = SpacegroupAnalyzer(
    structure
)

space_group = (
    analyzer.get_space_group_symbol()
)

space_group_number = (
    analyzer.get_space_group_number()
)

print(
    space_group,
    space_group_number
)
```

This information can be used for filtering.

For example:

```python
desired_space_group = 225

if (
    space_group_number
    == desired_space_group
):

    accepted.append(
        structure
    )
```

Symmetry should nevertheless be interpreted carefully because generated structures may be numerically close to a higher-symmetry configuration without being exactly symmetric.

---

## 24.8.71 Relaxation as a Validation Step

A generated crystal is usually not guaranteed to be at an energy minimum.

The structure may contain:

```text
Slightly incorrect
atomic positions
```

or

```text
Non-optimal lattice parameters
```

Therefore, structural relaxation is often necessary.

The workflow becomes:

```text
Generated Structure
       ↓
Initial Validation
       ↓
Geometry Optimization
       ↓
Relaxed Structure
       ↓
Energy Evaluation
```

A typical DFT workflow might use Quantum ESPRESSO.

Conceptually:

```text
Generated CIF
     ↓
Convert to QE input
     ↓
vc-relax
     ↓
Relaxed Structure
     ↓
scf
     ↓
Properties
```

This is particularly important because the generated structure should ultimately be judged after physically meaningful relaxation.

---

## 24.8.72 Generation and Relaxation Are Different Problems

It is important not to confuse

```text
Generation
```

with

```text
Relaxation
```

A generative model attempts to propose a candidate structure.

A relaxation calculation attempts to find a nearby lower-energy configuration.

Therefore:

```text
Generator:
"Here is a possible structure."

DFT:
"Here is the physically optimized structure near that configuration."
```

A successful generator should produce candidates that remain meaningful after relaxation.

If nearly every generated structure collapses into an unrelated structure during relaxation, the generator may not be learning useful structural information.

---

## 24.8.73 Relaxation Stability Metric

A useful research metric is the fraction of generated structures that remain structurally meaningful after relaxation.

Define:

```text
Relaxation Success Rate
=
Number of Successfully Relaxed Candidates
/
Number of Generated Valid Candidates
```

For example:

```python
relaxation_success_rate = (
    successful_relaxations
    /
    valid_candidates
)
```

A model that generates 10,000 structures but only 50 successfully relaxable candidates may require significant improvement.

Therefore, relaxation success should be reported alongside generation quality.

---

## 24.8.74 Energy Above Hull

For thermodynamic screening, formation energy alone may not be sufficient.

A compound can have a negative formation energy but still be unstable relative to competing phases.

The energy-above-hull concept addresses this issue.

Conceptually:

```text
Formation Energy
       ↓
Competing Phases
       ↓
Convex Hull
       ↓
Energy Above Hull
```

A candidate on the convex hull is thermodynamically stable relative to the considered competing phases.

A candidate above the hull may be metastable or unstable depending on the energy difference.

Therefore, generative materials screening should ideally consider phase stability rather than relying only on formation energy.

---

## 24.8.75 Materials Project Integration

Large materials databases can provide reference structures and calculated properties for training and screening.

A typical workflow can be:

```text
Materials Database
       ↓
Download Structures
       ↓
Clean Dataset
       ↓
Train Generative Model
       ↓
Generate Candidates
       ↓
Compare Against Known Materials
```

For research, database information should be carefully versioned.

The dataset should record:

```text
Structure ID
Composition
Structure
Calculated Properties
Source
Database Version
```

This makes experiments reproducible.

---

## 24.8.76 Dataset Leakage in Generative Materials Research

Data leakage is particularly important in generative materials research.

Suppose a generated structure is extremely similar to a structure present in the training dataset.

If it is later presented as a newly discovered candidate, the discovery claim may be misleading.

Therefore, train/test splitting should consider structural similarity.

A random split may not be sufficient.

For example:

```text
Training
├── Structure A
├── Structure A'
└── Structure A''

Test
└── Structure A'''
```

may produce artificially optimistic performance because all structures belong to nearly identical structural families.

A better strategy may involve grouping related structures before splitting.

---

## 24.8.77 Composition-Based Splitting

One possible strategy is to split by chemical systems.

For example:

```text
Training:
Li-Fe-O
Na-Fe-O
Li-Co-O

Testing:
Na-Co-O
```

The model must then generalize to a less familiar chemical combination.

Another strategy is to hold out entire structural families.

The appropriate split depends on the scientific question.

If the research goal is interpolation:

```text
Random / Family-Aware Split
```

may be appropriate.

If the goal is extrapolation:

```text
Chemical-System Holdout
```

may be more informative.

---

## 24.8.78 Evaluation of a Crystal Generator

A serious evaluation should not rely on a single metric.

At minimum, the generator should be evaluated in several categories.

### Validity

```text
Are generated structures physically and chemically valid?
```

### Uniqueness

```text
Are generated structures different from one another?
```

### Novelty

```text
Are generated structures meaningfully different from the training data?
```

### Property Accuracy

```text
Do generated structures satisfy the target properties?
```

### Diversity

```text
Does the generator explore multiple regions of materials space?
```

### Stability

```text
Do generated structures survive high-fidelity validation?
```

These metrics should be considered together.

---

## 24.8.79 Generator Evaluation Table

A useful evaluation table is:

```text
Metric                  Result
------------------------------------
Validity                91.4%
Uniqueness              84.7%
Novelty                 72.3%
Target Satisfaction     63.8%
Relaxation Success      48.1%
DFT Stability           31.7%
```

The exact values above are illustrative.

The important point is the structure of the evaluation.

A model with 95% validity but only 5% target satisfaction may not be useful for property-directed discovery.

Conversely, a model with excellent target satisfaction but extremely poor validity may also be unusable.

---

## 24.8.80 From Generative Model to Autonomous Discovery

The ultimate extension of this framework is an autonomous materials-discovery loop.

The system can repeatedly:

```text
Generate
   ↓
Screen
   ↓
Select
   ↓
Calculate
   ↓
Learn
   ↓
Generate Again
```

A more complete architecture is:

```text
                 ┌─────────────────────┐
                 │ Materials Database  │
                 └──────────┬──────────┘
                            ↓
                    Training Dataset
                            ↓
                 ┌─────────────────────┐
                 │ Generative Model    │
                 └──────────┬──────────┘
                            ↓
                    Crystal Candidates
                            ↓
                 ┌─────────────────────┐
                 │ Validity Filtering  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ ML Property Models  │
                 └──────────┬──────────┘
                            ↓
                    Candidate Ranking
                            ↓
                 ┌─────────────────────┐
                 │ DFT / High Fidelity │
                 └──────────┬──────────┘
                            ↓
                    New Reliable Data
                            │
                            └───────────────┐
                                            ↓
                                   Dataset Update
                                            ↓
                                      Model Retraining
                                            │
                                            └──────►
```

This architecture represents the transition from a static generative model to a continuously improving materials-discovery system.

The generator proposes.

The surrogate models screen.

DFT verifies.

New calculations improve the models.

The process then repeats.

That closed loop is one of the most promising directions for Materials Informatics.

---

## 24.8.81 Research Implementation Checklist

Before using a generative crystal model for an actual research project, the following questions should be answered.

```text
[ ] Is the crystal representation physically appropriate?

[ ] Are periodic boundary conditions handled correctly?

[ ] Can the model generate valid lattice parameters?

[ ] Can the model generate valid atomic coordinates?

[ ] Are atomic overlaps detected?

[ ] Are chemical constraints enforced?

[ ] Are composition constraints enforced?

[ ] Is symmetry analyzed?

[ ] Are duplicate structures removed?

[ ] Is structural diversity measured?

[ ] Is novelty measured?

[ ] Are target properties predicted independently?

[ ] Is prediction uncertainty estimated?

[ ] Are generated structures relaxed?

[ ] Is thermodynamic stability evaluated?

[ ] Are high-quality candidates verified with DFT?

[ ] Is the dataset protected against leakage?

[ ] Is the complete workflow reproducible?
```

A model should not be considered successful simply because its generated structures look plausible.

The real scientific test is whether the generated candidates survive increasingly strict levels of validation.

---

## 24.8.82 The Hierarchy of Trust

Generated materials can be viewed through a hierarchy of increasing confidence.

```text
Level 1
Generated
Structure

        ↓

Level 2
Geometrically
Valid

        ↓

Level 3
Chemically
Plausible

        ↓

Level 4
ML Property
Compatible

        ↓

Level 5
DFT Relaxed

        ↓

Level 6
Thermodynamically
Competitive

        ↓

Level 7
Experimentally
Validated
```

This hierarchy is important because a prediction at one level should not automatically be interpreted as evidence at a higher level.

For example:

```text
ML predicted stable
```

does not mean

```text
DFT confirmed stable
```

and

```text
DFT stable
```

does not necessarily mean

```text
Experimentally synthesizable
```

Generative Materials Informatics must therefore maintain a clear distinction between prediction, validation, and experimental evidence.

---

## 24.8.83 Toward Research-Grade Generative Materials Informatics

At this point, the generative framework can be summarized as a complete research architecture.

```text
                    MATERIALS DATA
                          │
                          ▼
                Crystal Representation
                          │
                          ▼
                  Generative Learning
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
       Unconditional              Conditional
        Generation                Generation
             │                         │
             └────────────┬────────────┘
                          ▼
                  Crystal Candidates
                          │
                          ▼
                  Validity Filtering
                          │
                          ▼
                 Chemical Filtering
                          │
                          ▼
                  Deduplication
                          │
                          ▼
              Property Prediction
                          │
                          ▼
              Multi-Objective Ranking
                          │
                          ▼
                    DFT Validation
                          │
                          ▼
                Experimental Testing
                          │
                          ▼
                  New Materials
                          │
                          ▼
                    New Data
                          │
                          └──────────────►
                         Retraining
```

The important conceptual transition is now complete:

```text
Machine Learning
        ↓
Generative Modeling
        ↓
Crystal Generation
        ↓
Property-Conditioned Generation
        ↓
Candidate Screening
        ↓
High-Fidelity Validation
        ↓
Materials Discovery
```

Generative AI therefore becomes most powerful when it is integrated with the rest of the Materials Informatics ecosystem rather than treated as an isolated neural-network technique.

The next step is to examine the **practical implementation of crystal diffusion models**, including how lattice parameters, fractional coordinates, atomic identities, periodicity, noise schedules, denoising networks, and sampling procedures can be implemented computationally.

## 24.8.84 Practical Crystal Diffusion Model Architecture

The previous sections established the scientific requirements of generative crystal discovery.

We can now move from the conceptual framework to the implementation of a crystal diffusion model.

A practical crystal diffusion model must answer several questions:

```text
What exactly is being diffused?

How is a crystal represented?

How are atomic species represented?

How are coordinates represented?

How is the lattice represented?

How is periodicity handled?

How is noise added?

How is noise removed?

How is the target property incorporated?

How is a valid crystal reconstructed?
```

A simplified architecture is

```text
Crystal
   │
   ├── Composition
   │
   ├── Fractional Coordinates
   │
   └── Lattice
   │
   ▼
Continuous / Discrete Representation
   │
   ▼
Forward Diffusion
   │
   ▼
Noisy Crystal
   │
   ▼
Denoising Network
   │
   ▼
Reverse Diffusion
   │
   ▼
Generated Crystal
```

Unlike image diffusion, the object being generated is structured and periodic.

Therefore, the implementation must preserve physical relationships throughout the generation process.

---

## 24.8.85 Crystal State Representation

A crystal can be represented as

```text
C = (Z, X, L)
```

where

```text
Z = atomic species

X = atomic fractional coordinates

L = lattice
```

For example, a crystal containing four atoms could be represented as

```python
atomic_numbers = torch.tensor([
    8,
    8,
    26,
    26
])
```

The fractional coordinates could be

```python
fractional_coords = torch.tensor([

    [0.00, 0.00, 0.00],

    [0.50, 0.50, 0.00],

    [0.25, 0.25, 0.25],

    [0.75, 0.75, 0.75]

])
```

and the lattice matrix could be

```python
lattice = torch.tensor([

    [5.0, 0.0, 0.0],

    [0.0, 5.0, 0.0],

    [0.0, 0.0, 5.0]

])
```

The complete crystal state is therefore

```python
crystal = {

    "atomic_numbers":
        atomic_numbers,

    "frac_coords":
        fractional_coords,

    "lattice":
        lattice
}
```

This representation separates the discrete and continuous parts of the crystal.

---

## 24.8.86 Why Fractional Coordinates Are Useful

Diffusing Cartesian coordinates directly can create difficulties because Cartesian coordinates depend on the lattice.

For example:

```text
Cartesian Coordinates
        +
Lattice
        ↓
Crystal Geometry
```

If the lattice changes, the Cartesian positions must change accordingly.

Fractional coordinates separate these quantities.

For an atom with fractional coordinate

```text
f = (x,y,z)
```

the Cartesian position is

```text
r = fL
```

where `L` is the lattice matrix.

In Python:

```python
cart_coords = (
    fractional_coords
    @ lattice
)
```

Therefore, the model can generate:

```text
Fractional Coordinates
+
Lattice
```

and reconstruct Cartesian coordinates afterward.

---

## 24.8.87 Diffusing Fractional Coordinates

Fractional coordinates lie in the periodic interval

```text
[0,1)
```

This creates a special problem for diffusion.

Consider two coordinates:

```text
0.99
```

and

```text
0.01
```

They are numerically far apart:

```text
|0.99 - 0.01| = 0.98
```

but physically they are very close because of periodicity.

The periodic distance is approximately

```text
0.02
```

Therefore, ordinary Euclidean distance is not sufficient.

A periodic coordinate difference can be calculated using

```python
def periodic_difference(
    x1,
    x2
):

    delta = x1 - x2

    return (
        delta
        - torch.round(delta)
    )
```

For example:

```python
x1 = torch.tensor(
    [0.99]
)

x2 = torch.tensor(
    [0.01]
)

delta = periodic_difference(
    x1,
    x2
)

print(delta)
```

The resulting difference is approximately

```text
-0.02
```

rather than `0.98`.

This periodic treatment is essential for crystal diffusion.

---

## 24.8.88 Periodic Coordinate Noise

A naive diffusion process might use

```python
x_t = (
    alpha_t.sqrt() * x_0
    +
    (1 - alpha_t).sqrt()
    * noise
)
```

However, directly adding Gaussian noise to fractional coordinates can move them outside `[0,1)`.

For example:

```text
x = 0.95

noise = +0.20

x' = 1.15
```

The coordinate should instead wrap around periodically:

```text
1.15 → 0.15
```

This can be implemented as

```python
x_noisy = (
    x_clean
    +
    sigma * noise
) % 1.0
```

This simple operation ensures that the coordinate remains inside the unit cell.

However, periodic wrapping alone does not completely solve the problem because the underlying diffusion geometry is also periodic.

More advanced models therefore use periodic representations or wrapped distributions designed specifically for angular variables.

---

## 24.8.89 Noise Schedule

The forward diffusion process gradually corrupts the crystal.

A common formulation is

```text
q(x_t | x_0)
=
N(
    √ᾱ_t x_0,
    (1-ᾱ_t)I
)
```

where

```text
ᾱ_t
```

controls the amount of noise.

A simple linear beta schedule can be implemented as

```python
import torch


def linear_beta_schedule(
    num_steps,
    beta_start=1e-4,
    beta_end=0.02
):

    return torch.linspace(
        beta_start,
        beta_end,
        num_steps
    )
```

The cumulative product is then

```python
betas = linear_beta_schedule(
    1000
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)
```

The resulting values determine how much information about the original crystal remains at each timestep.

---

## 24.8.90 Forward Diffusion Function

A standard forward diffusion function can be written as

```python
def q_sample(
    x0,
    t,
    alpha_bars
):

    alpha_bar = (
        alpha_bars[t]
    )

    noise = torch.randn_like(
        x0
    )

    xt = (
        alpha_bar.sqrt()
        * x0
        +
        (1.0 - alpha_bar).sqrt()
        * noise
    )

    return xt, noise
```

For crystal coordinates, periodic handling can be added:

```python
def q_sample_fractional(
    x0,
    t,
    alpha_bars
):

    alpha_bar = (
        alpha_bars[t]
    )

    noise = torch.randn_like(
        x0
    )

    xt = (
        alpha_bar.sqrt()
        * x0
        +
        (1.0 - alpha_bar).sqrt()
        * noise
    )

    xt = xt % 1.0

    return xt, noise
```

This illustrates the basic implementation idea.

A production model should use a diffusion formulation designed explicitly for periodic coordinates rather than relying solely on modulo wrapping.

---

## 24.8.91 Timestep Embedding

The denoising network must know the current diffusion timestep.

For example,

```text
t = 10
```

means the structure is only slightly noisy.

While

```text
t = 900
```

may correspond to a highly corrupted structure.

A sinusoidal timestep embedding can be implemented as

```python
import math


def timestep_embedding(
    timesteps,
    dim
):

    half = dim // 2

    frequencies = torch.exp(
        -math.log(10000)
        *
        torch.arange(
            half,
            device=timesteps.device
        )
        /
        (half - 1)
    )

    angles = (
        timesteps[:, None]
        *
        frequencies[None, :]
    )

    embedding = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles)
        ],
        dim=-1
    )

    return embedding
```

The embedding is then supplied to the denoising network.

---

## 24.8.92 Atomic Identity Representation

Atomic species are discrete.

For example:

```text
H
Li
C
O
Fe
Ni
...
```

They cannot be treated exactly like ordinary continuous coordinates.

A common approach is an embedding layer.

```python
class ElementEmbedding(
    nn.Module
):

    def __init__(
        self,
        num_elements,
        embedding_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_elements,
            embedding_dim
        )

    def forward(
        self,
        atomic_numbers
    ):

        return self.embedding(
            atomic_numbers
        )
```

For example:

```python
element_encoder = ElementEmbedding(
    num_elements=119,
    embedding_dim=128
)
```

The model then converts each atomic identity into a learned vector.

---

## 24.8.93 Combining Atomic and Coordinate Features

Each atom can have an initial feature representation:

```text
Element Embedding
       +
Fractional Coordinate
       +
Local Structural Information
       +
Timestep Embedding
```

A simple implementation is

```python
class AtomInputEncoder(
    nn.Module
):

    def __init__(
        self,
        num_elements,
        hidden_dim
    ):

        super().__init__()

        self.element_embedding = (
            nn.Embedding(
                num_elements,
                hidden_dim
            )
        )

        self.coord_encoder = nn.Sequential(

            nn.Linear(
                3,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

    def forward(
        self,
        atomic_numbers,
        frac_coords
    ):

        element_features = (
            self.element_embedding(
                atomic_numbers
            )
        )

        coordinate_features = (
            self.coord_encoder(
                frac_coords
            )
        )

        return (
            element_features
            +
            coordinate_features
        )
```

This is still a simplified architecture.

A research-grade model should additionally incorporate periodic neighborhood information.

---

## 24.8.94 Periodic Graph Construction

The crystal graph can be constructed from neighboring atoms.

Conceptually:

```text
Atom i
  │
  ├──── Atom j
  │
  ├──── Atom k
  │
  └──── Periodic image of Atom l
```

The edge features can contain information such as

```text
Distance
Direction
Periodic Offset
Bond-like Information
```

For a periodic edge, the neighbor may belong to an adjacent unit-cell image.

The edge can therefore store an integer lattice translation:

```python
offset = torch.tensor([
    1,
    0,
    0
])
```

which means that the neighbor lies in the unit cell translated by one lattice vector in the `a` direction.

---

## 24.8.95 Edge Feature Construction

A simple edge representation can include distance.

```python
def edge_distance(
    frac_i,
    frac_j,
    lattice,
    offset
):

    delta_frac = (
        frac_j
        +
        offset
        -
        frac_i
    )

    delta_cart = (
        delta_frac
        @
        lattice
    )

    distance = torch.linalg.norm(
        delta_cart
    )

    return distance
```

This is an important calculation because the physical distance depends on both:

```text
Fractional Coordinates
```

and

```text
Lattice
```

The lattice therefore directly participates in the graph representation.

---

## 24.8.96 Gaussian Distance Expansion

Instead of supplying a single distance value, many GNNs expand the distance into a basis.

For example:

```python
class GaussianDistanceExpansion(
    nn.Module
):

    def __init__(
        self,
        cutoff=5.0,
        num_basis=64
    ):

        super().__init__()

        centers = torch.linspace(
            0.0,
            cutoff,
            num_basis
        )

        self.register_buffer(
            "centers",
            centers
        )

        self.width = (
            centers[1]
            - centers[0]
        )

    def forward(
        self,
        distances
    ):

        return torch.exp(
            -(
                distances[..., None]
                - self.centers
            ) ** 2
            /
            self.width ** 2
        )
```

The result converts a scalar distance into a high-dimensional representation.

This can help the network learn nonlinear relationships between distance and interactions.

---

## 24.8.97 Message Passing Network

A crystal diffusion network can use message passing.

For each atom:

```text
Neighbor Features
       ↓
Edge Features
       ↓
Message
       ↓
Aggregation
       ↓
Updated Atom Feature
```

A simplified layer is

```python
class MessagePassingLayer(
    nn.Module
):

    def __init__(
        self,
        hidden_dim,
        edge_dim
    ):

        super().__init__()

        self.message = nn.Sequential(

            nn.Linear(
                hidden_dim + edge_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.update = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

    def forward(
        self,
        node_features,
        edge_index,
        edge_features
    ):

        source = edge_index[0]

        target = edge_index[1]

        messages = self.message(
            torch.cat(
                [
                    node_features[source],
                    edge_features
                ],
                dim=-1
            )
        )

        aggregated = torch.zeros_like(
            node_features
        )

        aggregated.index_add_(
            0,
            target,
            messages
        )

        updated = self.update(
            torch.cat(
                [
                    node_features,
                    aggregated
                ],
                dim=-1
            )
        )

        return updated
```

This simplified layer demonstrates how local crystal environments can influence atom representations.

---

## 24.8.98 Adding Timestep Conditioning

The message-passing network should also know the diffusion timestep.

A simple approach is

```python
h = h + time_embedding
```

where the timestep embedding has been projected to the hidden dimension.

For example:

```python
self.time_projection = nn.Linear(
    time_dim,
    hidden_dim
)
```

Then:

```python
time_features = (
    self.time_projection(
        time_embedding
    )
)

node_features = (
    node_features
    +
    time_features[:, None, :]
)
```

The model can therefore behave differently at different diffusion stages.

---

## 24.8.99 Property Conditioning

Property conditioning can be added in the same latent space.

Suppose the target contains:

```text
Band Gap = 1.8 eV

Formation Energy = -3.0 eV/atom

Bulk Modulus = 220 GPa
```

The condition vector is:

```python
condition = torch.tensor([
    1.8,
    -3.0,
    220.0
])
```

A property encoder can transform it:

```python
class PropertyEncoder(
    nn.Module
):

    def __init__(
        self,
        property_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                property_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

    def forward(
        self,
        properties
    ):

        return self.network(
            properties
        )
```

The resulting vector can be injected into the diffusion network.

---

## 24.8.100 Conditional Crystal Diffusion Network

A simplified research prototype can now combine:

```text
Element Embedding
+
Coordinate Encoding
+
Graph Message Passing
+
Time Embedding
+
Property Embedding
```

A conceptual implementation is:

```python
class CrystalDiffusionModel(
    nn.Module
):

    def __init__(
        self,
        num_elements,
        hidden_dim,
        edge_dim,
        property_dim,
        time_dim
    ):

        super().__init__()

        self.element_embedding = (
            nn.Embedding(
                num_elements,
                hidden_dim
            )
        )

        self.coord_encoder = nn.Sequential(

            nn.Linear(
                3,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.time_encoder = nn.Sequential(

            nn.Linear(
                time_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.property_encoder = nn.Sequential(

            nn.Linear(
                property_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.message_layer = (
            MessagePassingLayer(
                hidden_dim,
                edge_dim
            )
        )

        self.output = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                3
            )
        )

    def forward(
        self,
        atomic_numbers,
        frac_coords,
        edge_index,
        edge_features,
        time_embedding,
        property_condition
    ):

        h_element = (
            self.element_embedding(
                atomic_numbers
            )
        )

        h_coord = (
            self.coord_encoder(
                frac_coords
            )
        )

        h_time = (
            self.time_encoder(
                time_embedding
            )
        )

        h_property = (
            self.property_encoder(
                property_condition
            )
        )

        h = (
            h_element
            +
            h_coord
            +
            h_time
            +
            h_property
        )

        h = self.message_layer(
            h,
            edge_index,
            edge_features
        )

        predicted_noise = (
            self.output(h)
        )

        return predicted_noise
```

This is not a complete state-of-the-art crystal diffusion implementation.

It is a **research-oriented prototype architecture** demonstrating how the components fit together.

---

## 24.8.101 Predicting Coordinate Noise

The network can be trained to predict the noise added during the forward process.

Given

```text
x₀
```

the clean fractional coordinates,

we sample a timestep:

```python
t = torch.randint(
    0,
    num_steps,
    (batch_size,)
)
```

Then generate noisy coordinates:

```python
x_t, noise = q_sample_fractional(
    x0,
    t,
    alpha_bars
)
```

The network predicts:

```python
predicted_noise = model(
    atomic_numbers,
    x_t,
    edge_index,
    edge_features,
    time_embedding,
    property_condition
)
```

The training loss can be

```python
loss = (
    predicted_noise
    - noise
).pow(2).mean()
```

This is the fundamental diffusion training objective.

---

## 24.8.102 Training Step

A simplified training step is:

```python
def training_step(
    model,
    batch,
    alpha_bars
):

    atomic_numbers = (
        batch.atomic_numbers
    )

    x0 = (
        batch.frac_coords
    )

    properties = (
        batch.properties
    )

    t = torch.randint(
        0,
        len(alpha_bars),
        (1,),
        device=x0.device
    )

    xt, noise = (
        q_sample_fractional(
            x0,
            t,
            alpha_bars
        )
    )

    time_embedding = (
        timestep_embedding(
            t,
            dim=128
        )
    )

    predicted_noise = model(
        atomic_numbers,
        xt,
        batch.edge_index,
        batch.edge_features,
        time_embedding,
        properties
    )

    loss = (
        predicted_noise
        - noise
    ).pow(2).mean()

    return loss
```

The optimizer can then update the model:

```python
optimizer.zero_grad()

loss = training_step(
    model,
    batch,
    alpha_bars
)

loss.backward()

optimizer.step()
```

This gives the basic training loop.

---

## 24.8.103 Complete Training Loop

A practical training loop can be written as:

```python
for epoch in range(
    num_epochs
):

    model.train()

    total_loss = 0.0

    for batch in dataloader:

        batch = batch.to(device)

        optimizer.zero_grad()

        loss = training_step(
            model,
            batch,
            alpha_bars
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

    average_loss = (
        total_loss
        /
        len(dataloader)
    )

    print(
        f"Epoch {epoch}: "
        f"Loss = {average_loss:.6f}"
    )
```

Gradient clipping can help prevent unstable updates during training.

---

## 24.8.104 Why This Prototype Is Not Yet Sufficient

The implementation above is useful for understanding the architecture, but several important scientific issues remain.

A serious crystal diffusion model must consider:

```text
Periodic Geometry
+
Permutation Symmetry
+
Rotational Equivariance
+
Translational Invariance
+
Variable Number of Atoms
+
Discrete Atomic Species
+
Lattice Generation
+
Crystal Symmetry
```

A simple multilayer perceptron does not automatically satisfy these requirements.

For example, rotating a physical crystal should not change a scalar material property.

Similarly, translating every atom by the same amount should not change the physical identity of the crystal.

Therefore, symmetry-aware architectures are essential.

---

## 24.8.105 Permutation Invariance

Atoms of the same physical structure may appear in different orders.

For example:

```text
Structure A:

Fe
O
O
O


Structure B:

O
Fe
O
O
```

These are the same physical structure if the atomic coordinates correspond appropriately.

The model should therefore not depend on arbitrary atom ordering.

Message-passing GNNs naturally help with this because aggregation operations such as summation are permutation invariant.

For example:

```python
aggregated.index_add_(
    0,
    target,
    messages
)
```

does not require a particular ordering of the incoming neighbors.

---

## 24.8.106 Translational Invariance

A crystal can be translated without changing its physical properties.

If every atom is shifted by the same vector:

```text
rᵢ' = rᵢ + t
```

the physical crystal remains equivalent.

Therefore, property predictors should satisfy

```text
f(
    {rᵢ + t}
)
=
f(
    {rᵢ}
)
```

This is another reason fractional coordinates and relative displacement vectors are useful.

Instead of learning from absolute positions alone, the model can learn from periodic relative geometry.

---

## 24.8.107 Rotational Equivariance

Suppose a crystal is rotated.

Its scalar properties should remain unchanged:

```text
E(RX) = E(X)
```

where `R` is a rotation matrix.

For vector quantities, the output should rotate consistently.

This property is called equivariance.

A model satisfying rotational symmetry is therefore preferable for physical crystal generation.

This motivates architectures based on:

```text
E(3)-equivariance
```

or

```text
SE(3)-equivariance
```

depending on the precise representation and physical requirements.

---

## 24.8.108 Equivariant Message Passing

A simplified conceptual architecture is:

```text
Scalar Features
       +
Vector Features
       +
Periodic Geometry
       ↓
Equivariant Message Passing
       ↓
Updated Scalar / Vector Features
```

Instead of predicting coordinates using an arbitrary neural network, the model predicts geometrically consistent updates.

Conceptually:

```python
delta_r = equivariant_network(
    positions,
    node_features,
    edge_vectors
)
```

The output transforms consistently when the input crystal is rotated.

This is one of the major differences between a basic educational prototype and a research-grade crystal diffusion model.

---

## 24.8.109 Variable Number of Atoms

Another major challenge is that crystals can contain different numbers of atoms.

For example:

```text
Crystal A → 4 atoms

Crystal B → 8 atoms

Crystal C → 32 atoms

Crystal D → 96 atoms
```

A fixed-size neural network cannot directly assume that every crystal has the same number of atoms.

Graph neural networks naturally handle variable numbers of nodes.

```text
Graph A → 4 nodes

Graph B → 8 nodes

Graph C → 32 nodes
```

The same message-passing architecture can operate on all of them.

This is one reason graph-based diffusion models are attractive for crystal generation.

---

## 24.8.110 Generating the Number of Atoms

The model must also decide how many atoms to generate.

One simple strategy is to generate atom count separately.

```python
num_atoms = count_model(
    latent_vector
)
```

Then:

```text
Latent Vector
      ↓
Atom Count
      ↓
Generate N Atoms
```

A more advanced model can jointly model composition and structure.

For example:

```text
Latent Representation
        │
        ├──► Atom Count
        │
        ├──► Element Identity
        │
        ├──► Coordinates
        │
        └──► Lattice
```

This joint formulation is more expressive but also substantially more difficult to train.

---

## 24.8.111 Variable-Length Generation

A practical strategy is to use a maximum number of atoms.

For example:

```python
max_atoms = 64
```

and create a mask:

```python
atom_mask = torch.zeros(
    batch_size,
    max_atoms
)

atom_mask[
    :num_atoms
] = 1
```

The network can then ignore padded atoms.

Conceptually:

```text
Real atoms:

● ● ● ● ● ●

Padding:

○ ○ ○ ○ ○
```

The mask prevents padding tokens from contributing to the physical calculation.

This is a common computational strategy, although it introduces additional complexity for training and sampling.

---

## 24.8.112 Discrete Diffusion for Atomic Species

Coordinates are continuous, but atomic identities are discrete.

Therefore, applying ordinary Gaussian diffusion directly to atomic numbers is inappropriate.

For example:

```text
Fe = 26
O  = 8
```

Adding Gaussian noise to `26` does not produce a meaningful intermediate element.

Instead, discrete diffusion approaches can corrupt categorical identities probabilistically.

For example:

```text
Fe
 ↓
Fe  70%
Co  10%
Mn   8%
Ni   7%
O    5%
```

The model then learns to reconstruct the original categorical distribution.

Conceptually:

```text
Clean Element
      ↓
Discrete Forward Process
      ↓
Noisy Category
      ↓
Neural Network
      ↓
Original Element
```

This creates a mixed continuous-discrete diffusion problem.

---

## 24.8.113 Mixed Continuous-Discrete Crystal Diffusion

A complete crystal diffusion model may therefore contain separate diffusion processes:

```text
Crystal
 │
 ├── Coordinates
 │       ↓
 │   Continuous Diffusion
 │
 ├── Lattice
 │       ↓
 │   Continuous Diffusion
 │
 └── Atomic Species
         ↓
    Discrete Diffusion
```

The denoising model combines all three.

```text
Noisy Species
      +
Noisy Coordinates
      +
Noisy Lattice
      +
Time
      +
Condition
      ↓
Crystal Denoising Network
```

This is a much more realistic representation of the generative problem.

---

## 24.8.114 The Research-Level Architecture

A conceptual research-grade architecture can now be summarized as:

```text
                    Target Properties
                           │
                           ▼
                  Property Embedding
                           │
                           │
Noisy Crystal ─────────────┼─────────────►
                           │
                           ▼
                 Equivariant GNN
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       Species Head   Coordinate Head   Lattice Head
            │              │              │
            ▼              ▼              ▼
      Element Update   Position Update   Cell Update
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    Denoised Crystal
```

This architecture combines the major ideas introduced throughout the chapter:

```text
Generative Modeling
+
Crystal Graphs
+
Diffusion
+
Property Conditioning
+
Physical Symmetry
```

It provides the foundation for implementing modern crystal-generation systems.

---

## 24.8.115 Minimal Research Project

A student implementing this framework should not begin with a massive database and a complicated diffusion architecture.

A much better research strategy is to build the system incrementally.

### Phase 1

Start with a small dataset.

```text
100–1,000 crystals
```

Focus on:

```text
Structure Parsing
Graph Construction
Property Prediction
```

### Phase 2

Implement unconditional generation.

```text
Crystal
 ↓
Diffusion
 ↓
Crystal
```

### Phase 3

Add property conditioning.

```text
Target Property
      +
Diffusion
      ↓
Crystal
```

### Phase 4

Add structural validation.

```text
Crystal
 ↓
Validity Checks
```

### Phase 5

Add ML screening.

```text
Crystal
 ↓
GNN Predictor
 ↓
Target Filtering
```

### Phase 6

Add DFT validation.

```text
Crystal
 ↓
Relaxation
 ↓
Accurate Properties
```

### Phase 7

Build the closed-loop system.

```text
Generate
 ↓
Screen
 ↓
DFT
 ↓
Retrain
 ↓
Generate
```

This incremental approach makes debugging and scientific interpretation much easier.

---

## 24.8.116 What a Researcher Should Report

A generative materials paper should report more than a sample image of generated crystals.

At minimum, it should describe:

```text
Dataset
Representation
Train/Test Split
Diffusion Schedule
Model Architecture
Conditioning Method
Training Procedure
Sampling Procedure
Validity Metrics
Uniqueness
Novelty
Diversity
Property Accuracy
Stability
DFT Validation
```

The number of generated structures should also be clearly reported.

For example:

```text
Generated:
100,000

Valid:
82,400

Unique:
76,200

Target-satisfying:
8,500

DFT-tested:
500

Stable after DFT:
72
```

This provides a much more meaningful picture of the generator's performance.

---

## 24.8.117 Final Implementation Perspective

The central implementation idea of crystal diffusion can now be summarized as:

```text
Represent Crystal
        ↓
Add Noise
        ↓
Predict Noise / Score
        ↓
Remove Noise
        ↓
Repeat
        ↓
Recover Crystal
```

For a real materials system, this becomes:

```text
Composition
+
Lattice
+
Periodic Coordinates
        ↓
Noising Process
        ↓
Equivariant Crystal Network
        ↑
Time Embedding
        ↑
Property Embedding
        ↓
Denoising
        ↓
Generated Crystal
        ↓
Chemical Validation
        ↓
Property Prediction
        ↓
DFT
```

The most important lesson is that **crystal diffusion is not simply image diffusion applied to atomic coordinates**.

The model must understand:

```text
Periodic Geometry
Chemical Identity
Crystal Symmetry
Variable Composition
Variable Atom Count
Interatomic Interactions
Physical Constraints
```

These requirements make crystal generation one of the most challenging applications of generative machine learning in Materials Informatics.

The next section will move from the architecture to the **actual dataset construction and preprocessing pipeline**, including structure parsing, canonicalization, atom-count handling, lattice normalization, graph construction, property normalization, train/validation/test splitting, and preparation of PyTorch/PyTorch Geometric datasets for crystal diffusion training.


## 24.8.118 Crystal Diffusion Dataset Construction

A generative model is only as useful as the dataset used to train it.

For crystal generation, dataset preparation is more complicated than preparing an ordinary tabular machine-learning dataset.

A conventional ML dataset may look like:

```text
Feature 1
Feature 2
Feature 3
      ↓
Target
```

A crystal-generation dataset instead contains structured scientific objects:

```text
Composition
     +
Crystal Structure
     +
Lattice
     +
Atomic Coordinates
     +
Material Properties
```

A useful dataset record can therefore be represented as

```python
sample = {

    "structure": structure,

    "atomic_numbers": atomic_numbers,

    "frac_coords": frac_coords,

    "lattice": lattice,

    "num_atoms": num_atoms,

    "properties": properties
}
```

For example:

```python
sample = {

    "structure": "mp-1234",

    "atomic_numbers": [
        3,
        26,
        15,
        8
    ],

    "frac_coords": [
        [0.00, 0.00, 0.00],
        [0.50, 0.50, 0.50],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75]
    ],

    "lattice": [
        [5.1, 0.0, 0.0],
        [0.0, 5.1, 0.0],
        [0.0, 0.0, 5.1]
    ],

    "num_atoms": 4,

    "properties": {
        "band_gap": 1.82,
        "formation_energy": -2.71,
        "density": 5.24
    }
}
```

This representation provides everything required for the subsequent preprocessing stages.

---

## 24.8.119 Loading Crystal Structures with pymatgen

In a Materials Informatics workflow, `pymatgen` can be used to parse crystal structures.

```python
from pymatgen.core import Structure
```

A structure can be loaded from a CIF file:

```python
structure = Structure.from_file(
    "structure.cif"
)

print(structure)
```

Important information can then be extracted:

```python
print(
    structure.composition
)

print(
    structure.lattice
)

print(
    structure.frac_coords
)
```

The atomic species can be obtained using:

```python
species = [
    site.specie
    for site in structure
]
```

Atomic numbers can be extracted as:

```python
atomic_numbers = [
    site.specie.Z
    for site in structure
]
```

The fractional coordinates are:

```python
frac_coords = (
    structure.frac_coords
)
```

and the lattice matrix is:

```python
lattice = (
    structure.lattice.matrix
)
```

Therefore:

```python
sample = {

    "atomic_numbers":
        atomic_numbers,

    "frac_coords":
        frac_coords,

    "lattice":
        lattice
}
```

---

## 24.8.120 Extracting Material Properties

Suppose the dataset contains a CSV file:

```text
materials.csv
```

with columns:

```text
material_id
cif_path
band_gap
formation_energy
density
bulk_modulus
```

The file can be loaded using:

```python
import pandas as pd

df = pd.read_csv(
    "materials.csv"
)

print(
    df.head()
)
```

The property columns can be selected:

```python
property_columns = [

    "band_gap",

    "formation_energy",

    "density",

    "bulk_modulus"
]

properties = df[
    property_columns
]
```

The resulting matrix has the form:

```text
                Eg     Eform     Density     Bulk
Material 1      ...      ...        ...        ...
Material 2      ...      ...        ...        ...
Material 3      ...      ...        ...        ...
```

This property matrix will later be used for conditional generation.

---

## 24.8.121 Handling Missing Properties

Real materials datasets frequently contain missing values.

For example:

```python
print(
    df[property_columns].isna().sum()
)
```

A possible output is:

```text
band_gap             42
formation_energy      5
density               0
bulk_modulus        183
```

A simple filtering strategy is:

```python
df_clean = df.dropna(
    subset=property_columns
)
```

However, this may remove a large fraction of the dataset.

For example:

```text
Original dataset
      ↓
100,000 structures

After complete-property filtering
      ↓
61,000 structures
```

This may be undesirable.

Therefore, the correct strategy depends on the generation objective.

If the model must condition simultaneously on all four properties, complete records may be required.

If only band gap is used for conditioning, it may be better to retain structures for which band-gap information exists.

For example:

```python
df_bandgap = df.dropna(
    subset=["band_gap"]
)
```

This preserves more training data.

---

## 24.8.122 Removing Invalid Crystal Structures

Before training, structures should be checked for basic validity.

A first check is:

```python
def is_valid_structure(
    structure
):

    if structure is None:
        return False

    if len(structure) == 0:
        return False

    if structure.lattice is None:
        return False

    return True
```

The dataset can then be filtered:

```python
valid_structures = []

for structure in structures:

    if is_valid_structure(
        structure
    ):

        valid_structures.append(
            structure
        )
```

This is only a basic validity check.

A physically meaningful dataset requires additional checks.

---

## 24.8.123 Detecting Severe Atomic Overlap

Generated and experimental structures may occasionally contain atoms that are unrealistically close.

A simple minimum-distance check can be performed using periodic distances.

With pymatgen:

```python
def minimum_distance(
    structure
):

    distances = (
        structure.distance_matrix
    )

    distances = distances[
        distances > 1e-8
    ]

    if len(distances) == 0:
        return None

    return distances.min()
```

A simple screening condition might be:

```python
min_dist = minimum_distance(
    structure
)

if min_dist is not None:

    if min_dist < 0.5:

        print(
            "Suspicious structure"
        )
```

The exact threshold should **not** be treated as universal.

Different element pairs have different physically reasonable distances.

Therefore, a research-grade implementation should use chemically informed distance criteria.

---

## 24.8.124 Canonicalizing Structures

The same physical crystal can appear in different representations.

For example, two files may differ in:

```text
Atom ordering
Unit-cell choice
Origin choice
Coordinate representation
Symmetry setting
```

Therefore, simply comparing raw arrays is insufficient.

Canonicalization attempts to transform equivalent structures into a consistent representation.

Conceptually:

```text
Raw Structure
      ↓
Standardization
      ↓
Canonical Structure
```

This is particularly important for:

```text
Duplicate Removal
Dataset Splitting
Evaluation
Novelty Measurement
```

Without canonicalization, the model may accidentally see effectively identical structures in both training and test sets.

---

## 24.8.125 Structure Standardization

A structure can be processed using pymatgen's structure tools.

For example:

```python
from pymatgen.symmetry.analyzer import (
    SpacegroupAnalyzer
)
```

Then:

```python
analyzer = SpacegroupAnalyzer(
    structure
)
```

A conventional standard structure can be obtained using:

```python
standard_structure = (
    analyzer.get_conventional_standard_structure()
)
```

Alternatively:

```python
primitive_structure = (
    analyzer.get_primitive_standard_structure()
)
```

The choice depends on the modeling strategy.

Primitive structures generally contain fewer atoms, which can reduce computational cost.

However, conventional cells can make crystallographic interpretation easier.

---

## 24.8.126 Primitive vs Conventional Cells

Consider a crystal with:

```text
Conventional cell:
32 atoms
```

and

```text
Primitive cell:
8 atoms
```

A generative model operating on the primitive representation may require substantially less computation.

The general workflow is:

```text
CIF
 ↓
Structure
 ↓
Primitive Cell
 ↓
Model Representation
```

However, primitive-cell conversion must be handled carefully.

Two structures that are physically equivalent can have different primitive-cell representations.

Therefore, primitive-cell conversion should be part of a clearly defined preprocessing protocol.

---

## 24.8.127 Removing Duplicate Structures

Duplicate structures can severely distort generative-model evaluation.

Suppose:

```text
Dataset:

100,000 structures
```

but after deduplication:

```text
83,000 unique structures
```

Then the effective dataset size is only 83,000.

A very simple first-stage check can use composition:

```python
formula = (
    structure.composition
    .reduced_formula
)
```

However, composition is not enough.

For example:

```text
Fe2O3 structure A
Fe2O3 structure B
```

may represent different polymorphs.

Therefore:

```text
Composition
```

should not be used as the final structural identity.

A stronger strategy uses:

```text
Composition
+
Lattice
+
Atomic Coordinates
```

or a structure fingerprint.

---

## 24.8.128 Structure Fingerprints

A structure fingerprint converts a crystal into a numerical representation.

Conceptually:

```text
Crystal
   ↓
Fingerprint
   ↓
Vector
```

For example:

```python
fingerprint = structure_fingerprint(
    structure
)
```

Two structures can then be compared:

```python
similarity = cosine_similarity(
    fingerprint_a,
    fingerprint_b
)
```

A high similarity indicates structurally similar materials.

This becomes useful not only for duplicate removal but also for measuring generated diversity.

---

## 24.8.129 Composition Encoding

The composition can also be represented numerically.

For example:

```text
Li2FePO4
```

can be represented using elemental fractions.

```python
from pymatgen.core import Composition

composition = Composition(
    "Li2FePO4"
)

print(
    composition.fractional_composition
)
```

The resulting representation contains normalized elemental fractions.

For a fixed element vocabulary, the composition can be converted into a vector:

```text
Li   Fe   P    O
----------------
0.25 0.125 0.125 0.50
```

This representation can be useful for auxiliary conditioning.

---

## 24.8.130 Element Vocabulary

A generative model must define which elements it can generate.

For example:

```python
element_list = [

    "H",
    "Li",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Fe",
    "Co",
    "Ni",
    "Cu"
]
```

A mapping can then be created:

```python
element_to_index = {

    element: i

    for i, element
    in enumerate(element_list)
}
```

For example:

```python
print(
    element_to_index["Fe"]
)
```

The reverse mapping is also required:

```python
index_to_element = {

    i: element

    for element, i
    in element_to_index.items()
}
```

This allows the model to convert between:

```text
Element
↔
Integer Token
```

---

## 24.8.131 Unknown Elements

If a structure contains an element not included in the vocabulary:

```python
if element not in element_to_index:

    raise ValueError(
        f"Unknown element: {element}"
    )
```

This should generally be handled explicitly.

Silently mapping an unknown element to an arbitrary category can corrupt the training data.

For a research project, the element vocabulary should therefore be defined before training.

---

## 24.8.132 Coordinate Tensor Construction

The fractional coordinates can be converted into PyTorch tensors.

```python
import torch

frac_coords = torch.tensor(
    structure.frac_coords,
    dtype=torch.float32
)
```

Atomic numbers:

```python
atomic_numbers = torch.tensor(
    [
        site.specie.Z
        for site in structure
    ],
    dtype=torch.long
)
```

Lattice:

```python
lattice = torch.tensor(
    structure.lattice.matrix,
    dtype=torch.float32
)
```

The resulting tensors are:

```python
print(
    atomic_numbers.shape
)

print(
    frac_coords.shape
)

print(
    lattice.shape
)
```

For a crystal containing `N` atoms:

```text
atomic_numbers → [N]

frac_coords    → [N, 3]

lattice        → [3, 3]
```

---

## 24.8.133 Property Normalization Pipeline

The property matrix should generally be normalized before conditioning.

For example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

properties_scaled = (
    scaler.fit_transform(
        properties
    )
)
```

The scaler should be fitted **only on the training set**.

This is extremely important.

The correct procedure is:

```text
Training Properties
        ↓
Fit Scaler
        ↓
Transform Train
        ↓
Transform Validation
        ↓
Transform Test
```

Not:

```text
Train + Validation + Test
        ↓
Fit Scaler
```

The second approach introduces information from the validation/test distribution into training.

---

## 24.8.134 Saving the Property Scaler

The trained scaler should be saved.

```python
import joblib

joblib.dump(
    scaler,
    "property_scaler.pkl"
)
```

Later:

```python
scaler = joblib.load(
    "property_scaler.pkl"
)
```

This is essential during generation.

Suppose the user requests:

```text
Band Gap = 2.0 eV
```

The target must be transformed using the same scaler used during training:

```python
target = scaler.transform(
    [[2.0]]
)
```

The model should never receive a differently normalized target.

---

## 24.8.135 Train/Validation/Test Splitting

A standard split may be:

```text
Training:
80%

Validation:
10%

Test:
10%
```

For example:

```python
from sklearn.model_selection import train_test_split

train_df, temp_df = (
    train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42
    )
)

val_df, test_df = (
    train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )
)
```

This gives approximately:

```text
80% training
10% validation
10% test
```

The random seed ensures reproducibility.

---

## 24.8.136 Why Random Splitting Can Be Dangerous

Random splitting is not always sufficient for materials datasets.

Suppose the dataset contains many chemically related materials:

```text
LiFePO4
LiMnPO4
LiCoPO4
LiNiPO4
```

Random splitting could place highly related structures in all three partitions.

The resulting test performance may appear excellent even though the model has not truly learned to generalize to new chemical systems.

Therefore, more challenging evaluation strategies can include:

```text
Composition-based split
Element-based split
Chemical-family split
Structure-family split
Time-based split
```

For example:

```text
Training:
Li-containing compounds

Test:
Na-containing compounds
```

This measures a substantially harder form of generalization.

---

## 24.8.137 Materials-Aware Data Splitting

A composition-aware split can be constructed by grouping structures.

Conceptually:

```python
groups = df[
    "chemical_system"
]
```

Then a grouped splitter can be used.

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=groups
    )
)
```

This prevents structures belonging to the same predefined group from being distributed arbitrarily across the datasets.

The grouping strategy should be selected according to the scientific question.

---

## 24.8.138 PyTorch Dataset

A custom dataset can now be created.

```python
from torch.utils.data import Dataset


class CrystalDiffusionDataset(
    Dataset
):

    def __init__(
        self,
        structures,
        properties
    ):

        self.structures = structures

        self.properties = properties

    def __len__(self):

        return len(
            self.structures
        )

    def __getitem__(
        self,
        index
    ):

        structure = (
            self.structures[index]
        )

        property_vector = (
            self.properties[index]
        )

        atomic_numbers = torch.tensor(

            [
                site.specie.Z
                for site in structure
            ],

            dtype=torch.long
        )

        frac_coords = torch.tensor(

            structure.frac_coords,

            dtype=torch.float32
        )

        lattice = torch.tensor(

            structure.lattice.matrix,

            dtype=torch.float32
        )

        properties = torch.tensor(

            property_vector,

            dtype=torch.float32
        )

        return {

            "atomic_numbers":
                atomic_numbers,

            "frac_coords":
                frac_coords,

            "lattice":
                lattice,

            "properties":
                properties
        }
```

This provides a clean interface between the materials dataset and PyTorch.

---

## 24.8.139 The Variable-Length Batch Problem

There is an important issue with the dataset above.

Suppose one sample has:

```text
4 atoms
```

while another has:

```text
12 atoms
```

Their coordinate tensors have different shapes:

```text
[4,3]

[12,3]
```

PyTorch cannot simply stack them into:

```python
torch.stack(...)
```

because their dimensions differ.

A custom `collate_fn` is therefore required.

---

## 24.8.140 Custom Collate Function

A simple collate function can preserve each crystal separately:

```python
def crystal_collate_fn(
    batch
):

    atomic_numbers = [
        item["atomic_numbers"]
        for item in batch
    ]

    frac_coords = [
        item["frac_coords"]
        for item in batch
    ]

    lattices = torch.stack(
        [
            item["lattice"]
            for item in batch
        ]
    )

    properties = torch.stack(
        [
            item["properties"]
            for item in batch
        ]
    )

    return {

        "atomic_numbers":
            atomic_numbers,

        "frac_coords":
            frac_coords,

        "lattice":
            lattices,

        "properties":
            properties
    }
```

The dataloader can then use:

```python
from torch.utils.data import DataLoader

loader = DataLoader(

    dataset,

    batch_size=16,

    shuffle=True,

    collate_fn=
        crystal_collate_fn
)
```

This allows each batch to contain crystals with different numbers of atoms.

---

## 24.8.141 Padding-Based Representation

An alternative is to pad every structure to a fixed maximum number of atoms.

Suppose:

```python
max_atoms = 64
```

A crystal with 10 atoms receives:

```text
10 real atoms
+
54 padding atoms
```

The coordinate tensor becomes:

```text
[64,3]
```

and a mask identifies the real atoms:

```python
mask = torch.zeros(
    64,
    dtype=torch.bool
)

mask[:10] = True
```

The model can then ignore padded atoms.

This approach simplifies batching but increases memory usage.

---

## 24.8.142 Building Periodic Neighbor Graphs

The next major preprocessing stage is graph construction.

For every atom, neighboring atoms must be identified.

Conceptually:

```text
Crystal
   ↓
Periodic Neighbor Search
   ↓
Edges
   ↓
Distances
   ↓
Edge Features
```

A cutoff radius can be selected:

```python
cutoff = 5.0
```

meaning that atoms within approximately 5 Å are considered neighbors.

A research implementation should use a periodic neighbor-search algorithm rather than computing only distances inside the original unit cell.

---

## 24.8.143 Why Periodic Neighbor Search Matters

Consider an atom near the boundary:

```text
|-------------------|
|                   |
|                ●  |
|                   |
| ●                 |
|-------------------|
```

The two atoms may be physically close through periodic wrapping even though they appear far apart inside the displayed unit cell.

A non-periodic graph would miss this interaction.

The correct graph includes:

```text
Atom
  │
  └── Periodic Image
```

Therefore:

```text
Periodic Boundary Conditions
```

must be incorporated into graph construction.

---

## 24.8.144 Graph Data Structure

A graph sample can contain:

```python
graph = {

    "atomic_numbers":
        atomic_numbers,

    "frac_coords":
        frac_coords,

    "lattice":
        lattice,

    "edge_index":
        edge_index,

    "edge_vectors":
        edge_vectors,

    "edge_distances":
        edge_distances,

    "properties":
        properties
}
```

The important dimensions are approximately:

```text
atomic_numbers → [N]

frac_coords    → [N,3]

lattice        → [3,3]

edge_index     → [2,E]

edge_vectors   → [E,3]

edge_distances → [E]

properties     → [P]
```

where:

```text
N = number of atoms

E = number of edges

P = number of conditioning properties
```

---

## 24.8.145 Converting Fractional Coordinates to Cartesian Coordinates

Periodic graph construction generally requires Cartesian displacement vectors.

For two atoms:

```text
fᵢ
fⱼ
```

and a periodic offset:

```text
n = (n₁,n₂,n₃)
```

the fractional displacement is:

```text
Δf = fⱼ + n - fᵢ
```

The Cartesian displacement is:

```text
Δr = Δf L
```

In Python:

```python
def periodic_displacement(
    frac_i,
    frac_j,
    lattice,
    offset
):

    delta_frac = (
        frac_j
        + offset
        - frac_i
    )

    delta_cart = (
        delta_frac
        @
        lattice
    )

    return delta_cart
```

The distance is:

```python
distance = torch.linalg.norm(
    delta_cart
)
```

This calculation is central to periodic crystal graph construction.

---

## 24.8.146 Building a Complete Dataset Pipeline

The complete preprocessing workflow can now be written as:

```text
Raw Materials Dataset
        ↓
Load Structures
        ↓
Parse with pymatgen
        ↓
Validate Structures
        ↓
Standardize Structures
        ↓
Remove Duplicates
        ↓
Extract Composition
        ↓
Extract Atomic Species
        ↓
Extract Fractional Coordinates
        ↓
Extract Lattice
        ↓
Build Periodic Graph
        ↓
Extract Target Properties
        ↓
Normalize Properties
        ↓
Train/Validation/Test Split
        ↓
PyTorch Dataset
        ↓
PyTorch Geometric Batch
```

This pipeline is the foundation for the actual diffusion-training system.

---

## 24.8.147 Saving Preprocessed Data

Preprocessing can be computationally expensive.

Therefore, processed samples should generally be cached rather than regenerated every training epoch.

For example:

```python
import torch

torch.save(
    processed_data,
    "crystal_dataset.pt"
)
```

The dataset can later be loaded using:

```python
processed_data = torch.load(
    "crystal_dataset.pt"
)
```

For very large datasets, more scalable formats may be preferable.

The important principle is:

```text
Expensive preprocessing
        ↓
Perform once
        ↓
Cache
        ↓
Reuse during training
```

---

## 24.8.148 Dataset Quality Control

Before beginning diffusion training, several statistics should be examined.

For example:

```python
print(
    df["num_atoms"].describe()
)
```

The researcher should inspect:

```text
Number of structures
Number of unique compositions
Number of unique chemical systems
Atom-count distribution
Element frequency
Lattice-parameter distribution
Density distribution
Property distributions
Missing-value fraction
Duplicate fraction
```

For example:

```python
print(
    df["band_gap"].describe()
)

print(
    df["density"].describe()
)
```

Visualization is also valuable.

```python
import matplotlib.pyplot as plt

plt.hist(
    df["band_gap"],
    bins=50
)

plt.xlabel(
    "Band gap (eV)"
)

plt.ylabel(
    "Number of structures"
)

plt.show()
```

A diffusion model trained on a severely biased property distribution may generate correspondingly biased materials.

---

## 24.8.149 Inspecting Element Distribution

Element frequency can be calculated:

```python
from collections import Counter

counter = Counter()

for structure in structures:

    for site in structure:

        counter[
            site.specie.symbol
        ] += 1
```

The most common elements can then be inspected:

```python
print(
    counter.most_common(20)
)
```

This can reveal strong dataset biases.

For example:

```text
O     180,000
Si    120,000
Fe     95,000
Al     91,000
...
```

If some elements dominate the dataset, the generator may learn to reproduce those biases.

---

## 24.8.150 Atom-Count Distribution

The distribution of atoms per structure should also be examined.

```python
atom_counts = [
    len(structure)
    for structure in structures
]
```

Then:

```python
plt.hist(
    atom_counts,
    bins=50
)

plt.xlabel(
    "Number of atoms"
)

plt.ylabel(
    "Number of structures"
)

plt.show()
```

This reveals whether the dataset mostly contains:

```text
Small structures
```

or

```text
Large structures
```

If the dataset contains structures ranging from 2 atoms to 200 atoms, the model must be designed to handle this variability.

---

## 24.8.151 Property Distribution and Conditioning Bias

Suppose the training dataset contains:

```text
Band gap:

0–1 eV      → 60%
1–2 eV      → 30%
2–4 eV      → 10%
```

If the researcher asks the model to generate:

```text
Eg = 3.5 eV
```

the model is being asked to operate in a relatively underrepresented region.

This is an important limitation.

Property-conditioned generation does not magically create information that was absent from the training data.

Therefore:

```text
Training Distribution
        ↓
Defines
        ↓
Learned Generative Support
```

A target far outside the training distribution may lead to poor or unrealistic generation.

---

## 24.8.152 Conditioning Target Selection

During training, target properties come from real structures.

For example:

```python
property_condition = (
    batch["properties"]
)
```

During generation, however, the researcher chooses the condition.

For example:

```python
target = torch.tensor([

    2.0,
    -3.0,
    200.0

])
```

The target must be normalized:

```python
target_scaled = (
    scaler.transform(
        target.reshape(1, -1)
    )
)
```

Then converted into a tensor:

```python
target_scaled = torch.tensor(
    target_scaled,
    dtype=torch.float32
)
```

This is the bridge between the scientific design objective and the trained diffusion model.

---

## 24.8.153 Dataset Preparation Checklist

Before training, the following questions should have clear answers:

```text
[ ] What structures are included?

[ ] Which elements are allowed?

[ ] Are structures standardized?

[ ] Are duplicates removed?

[ ] Are invalid structures removed?

[ ] How are periodic neighbors generated?

[ ] How is the lattice represented?

[ ] How are coordinates represented?

[ ] How are atomic species encoded?

[ ] How are variable atom counts handled?

[ ] Which properties are used for conditioning?

[ ] How are properties normalized?

[ ] Was the scaler fitted only on training data?

[ ] How was the train/test split performed?

[ ] Is chemical-family leakage controlled?

[ ] Are processed graphs cached?

[ ] Are dataset statistics documented?
```

These steps may appear like preprocessing details, but they directly influence the scientific validity of the resulting generative model.

---

## 24.8.154 From Dataset to Diffusion Training

After preprocessing, the data flow becomes:

```text
Crystal Dataset
      │
      ▼
PyTorch Dataset
      │
      ▼
Periodic Graph
      │
      ▼
Batch
      │
      ├───────────────┐
      │               │
      ▼               ▼
Clean Crystal     Properties
      │               │
      ▼               ▼
Forward Diffusion  Condition Encoder
      │               │
      └───────┬───────┘
              ▼
       Crystal Diffusion
            Network
              │
              ▼
        Noise Prediction
              │
              ▼
             Loss
              │
              ▼
       Backpropagation
```

This is the complete bridge between a materials database and a trainable generative model.

The next step is to implement the **actual periodic crystal graph construction with PyTorch Geometric**, including `Data` objects, edge indices, periodic image offsets, edge vectors, batching variable-size crystals, and the graph representation required by the diffusion network.

## 24.8.155 Periodic Crystal Graph Construction with PyTorch Geometric

The previous section converted crystal structures into the conceptual components required by a generative model.

The next step is to construct an actual graph representation that can be processed by a graph neural network.

For a crystal,

```text
Atoms → Nodes

Periodic Neighbor Relationships → Edges

Atomic Properties → Node Features

Distances / Directions → Edge Features
```

A crystal graph can therefore be represented as

```text
             Atom 2
             /    \
            /      \
       Atom 1 ----- Atom 3
          \          /
           \        /
             Atom 4
```

However, unlike an ordinary molecular graph, a crystal graph must respect **periodic boundary conditions**.

This distinction is fundamental.

A molecule is usually represented as a finite graph:

```text
Molecule
   ↓
Finite atoms
   ↓
Finite edges
```

A crystal is conceptually infinite:

```text
        Cell
         ↓
   ┌─────────────┐
   │             │
   │   Crystal   │
   │             │
   └─────────────┘
       ↕   ↕
   Periodic copies
```

Therefore, graph construction must include neighbors belonging to periodic images of the unit cell.

---

## 24.8.156 PyTorch Geometric Representation

PyTorch Geometric represents graphs using a `Data` object.

```python
from torch_geometric.data import Data
```

A simple graph can be constructed as:

```python
data = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_features
)
```

For crystal generation, additional information can be stored:

```python
data = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_features,

    pos=cartesian_positions,

    lattice=lattice,

    atomic_numbers=atomic_numbers
)
```

The important fields are:

```text
x
↓
Node features

edge_index
↓
Connectivity

edge_attr
↓
Edge features

pos
↓
Atomic positions

lattice
↓
Unit-cell geometry
```

---

## 24.8.157 Atomic Node Features

The simplest node feature is the atomic number.

For example:

```text
Li → 3
Fe → 26
P  → 15
O  → 8
```

The corresponding tensor is:

```python
import torch

atomic_numbers = torch.tensor(
    [3, 26, 15, 8],
    dtype=torch.long
)
```

However, directly using atomic numbers as continuous features is generally not ideal.

The difference between:

```text
Fe = 26
Co = 27
Ni = 28
```

does not imply that these elements should have a simple linear relationship.

Instead, atomic identities are usually mapped into learned embeddings.

---

## 24.8.158 Element Embedding

A simple embedding layer can be created using:

```python
import torch.nn as nn

element_embedding = nn.Embedding(

    num_embeddings=119,

    embedding_dim=64
)
```

The atomic numbers can then be embedded:

```python
node_features = element_embedding(
    atomic_numbers
)
```

For four atoms:

```text
atomic_numbers
      ↓
[3, 26, 15, 8]
      ↓
Embedding
      ↓
[4, 64]
```

Thus each atom receives a 64-dimensional learned representation.

---

## 24.8.159 Why Embeddings Are Useful

The model can learn representations such as:

```text
Chemical Similarity
Oxidation Behavior
Atomic Size
Electronegativity
Bonding Characteristics
```

without requiring the researcher to manually specify all relationships.

The embedding initially contains random values.

During training:

```text
Prediction Error
      ↓
Backpropagation
      ↓
Embedding Update
      ↓
Better Chemical Representation
```

Therefore, the model learns an element representation optimized for the generative task.

---

## 24.8.160 One-Hot Element Encoding

An alternative is one-hot encoding.

For example, suppose the vocabulary is:

```text
Li
Fe
P
O
```

Then:

```text
Li → [1,0,0,0]

Fe → [0,1,0,0]

P  → [0,0,1,0]

O  → [0,0,0,1]
```

This can be implemented using:

```python
import torch.nn.functional as F

one_hot = F.one_hot(

    atomic_numbers,

    num_classes=119
).float()
```

However, one-hot vectors do not contain learned relationships between elements.

For deep generative models, learned embeddings are generally more flexible.

---

## 24.8.161 Extracting Periodic Neighbors with pymatgen

`pymatgen` provides tools for finding neighboring sites.

A useful method is:

```python
neighbors = structure.get_all_neighbors(
    r=5.0
)
```

Here:

```text
r = cutoff radius
```

For example:

```python
neighbors = structure.get_all_neighbors(
    r=5.0
)
```

returns neighboring sites for each atom.

The resulting information can be inspected:

```python
for i, atom_neighbors in enumerate(
    neighbors
):

    print(
        "Atom:",
        i
    )

    for neighbor in atom_neighbors:

        print(
            neighbor
        )
```

The exact returned object contains information such as:

```text
Neighbor site
Distance
Image information
```

The image information is important for periodic crystals.

---

## 24.8.162 Periodic Image Offsets

Suppose atom `j` belongs to a neighboring periodic cell.

Its periodic image can be described using an integer translation vector:

```text
n = (nx, ny, nz)
```

For example:

```text
(0,0,0)
```

means the original unit cell.

Whereas:

```text
(1,0,0)
```

means the image translated by one lattice vector along the first lattice direction.

Similarly:

```text
(0,-1,0)
```

means a periodic image translated by one negative lattice vector.

The graph therefore needs to store not only:

```text
i → j
```

but also:

```text
i → j + periodic_image
```

---

## 24.8.163 Edge Index

PyTorch Geometric represents graph connectivity using an `edge_index` tensor.

For example:

```python
edge_index = torch.tensor(

    [
        [0, 1, 2, 2],
        [1, 2, 0, 3]
    ],

    dtype=torch.long
)
```

The shape is:

```text
[2, E]
```

where:

```text
E = number of directed edges
```

The first row contains source nodes:

```text
0 1 2 2
```

and the second row contains destination nodes:

```text
1 2 0 3
```

Therefore:

```text
0 → 1
1 → 2
2 → 0
2 → 3
```

For crystal graphs, periodic image information must additionally be stored.

---

## 24.8.164 Periodic Edge Representation

A useful representation is:

```python
edge_index
edge_shift
```

where:

```text
edge_index
↓
Which atoms are connected

edge_shift
↓
Which periodic image contains the neighbor
```

For example:

```python
edge_index = torch.tensor(

    [
        [0, 0, 1],
        [1, 2, 2]
    ],

    dtype=torch.long
)
```

and:

```python
edge_shift = torch.tensor(

    [
        [0, 0, 0],
        [1, 0, 0],
        [0,-1, 0]
    ],

    dtype=torch.long
)
```

The second edge therefore represents a neighbor in the periodic image translated by:

```text
(1,0,0)
```

---

## 24.8.165 Calculating Periodic Displacement Vectors

Let:

```text
f_i
```

be the fractional coordinate of the source atom.

Let:

```text
f_j
```

be the fractional coordinate of the destination atom.

Let:

```text
n_ij
```

be the periodic image shift.

Then:

```text
Δf_ij = f_j + n_ij - f_i
```

The corresponding Cartesian displacement is:

```text
Δr_ij = Δf_ij L
```

where `L` is the lattice matrix.

In Python:

```python
def displacement_vector(

    frac_i,

    frac_j,

    lattice,

    image_shift

):

    delta_frac = (

        frac_j

        + image_shift

        - frac_i
    )

    delta_cart = (

        delta_frac
        @
        lattice
    )

    return delta_cart
```

The distance is:

```python
def displacement_distance(

    frac_i,

    frac_j,

    lattice,

    image_shift

):

    vector = displacement_vector(

        frac_i,

        frac_j,

        lattice,

        image_shift
    )

    return torch.linalg.norm(
        vector
    )
```

---

## 24.8.166 Constructing a Periodic Crystal Graph

We can now construct a basic graph.

```python
def build_crystal_graph(

    structure,

    cutoff=5.0

):

    neighbors = (
        structure.get_all_neighbors(
            r=cutoff
        )
    )

    edge_sources = []

    edge_targets = []

    edge_shifts = []

    edge_distances = []

    frac_coords = torch.tensor(

        structure.frac_coords,

        dtype=torch.float32
    )

    lattice = torch.tensor(

        structure.lattice.matrix,

        dtype=torch.float32
    )

    for i, atom_neighbors in enumerate(
        neighbors
    ):

        for neighbor in atom_neighbors:

            j = neighbor.index

            image = neighbor.image

            distance = neighbor.nn_distance

            edge_sources.append(i)

            edge_targets.append(j)

            edge_shifts.append(image)

            edge_distances.append(
                distance
            )

    edge_index = torch.tensor(

        [
            edge_sources,
            edge_targets
        ],

        dtype=torch.long
    )

    edge_shifts = torch.tensor(

        edge_shifts,

        dtype=torch.long
    )

    edge_distances = torch.tensor(

        edge_distances,

        dtype=torch.float32
    )

    return (

        edge_index,

        edge_shifts,

        edge_distances
    )
```

This is the first practical implementation of a periodic crystal graph.

---

## 24.8.167 Verifying the Graph

After constructing the graph:

```python
edge_index, edge_shifts, edge_distances = (
    build_crystal_graph(
        structure
    )
)
```

we can inspect:

```python
print(
    edge_index.shape
)

print(
    edge_shifts.shape
)

print(
    edge_distances.shape
)
```

For example:

```text
[2, 48]
[48, 3]
[48]
```

This means the graph contains:

```text
48 directed edges
```

and each edge has:

```text
3-dimensional periodic shift
```

and:

```text
1 scalar distance
```

---

## 24.8.168 Edge Vectors

Distances alone may not contain sufficient geometric information.

Two neighboring atoms may have identical distances but different spatial directions.

Therefore, the edge vector is useful.

For every edge:

```text
Δr = [dx, dy, dz]
```

can be calculated.

```python
def compute_edge_vectors(

    frac_coords,

    lattice,

    edge_index,

    edge_shifts

):

    src = edge_index[0]

    dst = edge_index[1]

    src_frac = frac_coords[src]

    dst_frac = frac_coords[dst]

    delta_frac = (

        dst_frac

        + edge_shifts.float()

        - src_frac
    )

    edge_vectors = (
        delta_frac
        @
        lattice
    )

    return edge_vectors
```

The resulting tensor has shape:

```text
[E,3]
```

---

## 24.8.169 Edge Distance Verification

The distance can be recomputed:

```python
edge_distances = torch.linalg.norm(

    edge_vectors,

    dim=1
)
```

This provides a useful consistency check.

```python
print(
    edge_distances
)
```

The values should agree with the neighbor distances obtained during graph construction within numerical tolerance.

---

## 24.8.170 Radial Basis Encoding

A neural network often benefits from converting distances into smooth basis functions.

Instead of feeding:

```text
2.81 Å
```

directly into the network, the distance can be expanded using Gaussian radial basis functions.

For basis center `μ_k`:

```text
φ_k(r)
=
exp(
    -γ(r - μ_k)^2
)
```

A simple implementation is:

```python
class GaussianRBF(
    nn.Module
):

    def __init__(

        self,

        num_basis=32,

        cutoff=5.0

    ):

        super().__init__()

        centers = torch.linspace(

            0.0,

            cutoff,

            num_basis
        )

        self.register_buffer(

            "centers",

            centers
        )

        self.gamma = (
            10.0
            / cutoff**2
        )

    def forward(self, distances):

        return torch.exp(

            -self.gamma
            *
            (
                distances.unsqueeze(-1)
                -
                self.centers
            )**2
        )
```

For:

```text
E = 100
```

edges and:

```text
32
```

basis functions:

```text
distance
   ↓
RBF
   ↓
[100,32]
```

---

## 24.8.171 Cutoff Envelope

A hard cutoff can introduce discontinuities.

For example:

```text
r < rc → neighbor

r ≥ rc → ignored
```

The network may therefore see an abrupt change around the cutoff.

A smooth cutoff function can be used.

One simple form is:

```text
f_c(r)
=
1/2[
cos(πr/r_c)+1
]
```

for:

```text
r < r_c
```

and:

```text
f_c(r)=0
```

otherwise.

A PyTorch implementation is:

```python
def cosine_cutoff(

    distances,

    cutoff

):

    values = torch.zeros_like(
        distances
    )

    mask = distances < cutoff

    values[mask] = 0.5 * (

        torch.cos(

            torch.pi
            *
            distances[mask]
            /
            cutoff
        )

        + 1.0
    )

    return values
```

This creates a smooth transition toward zero.

---

## 24.8.172 Combining RBF and Cutoff

The final edge representation can be:

```python
rbf = GaussianRBF(
    num_basis=32,
    cutoff=5.0
)

radial_features = rbf(
    edge_distances
)

cutoff_values = cosine_cutoff(

    edge_distances,

    5.0
)

radial_features = (

    radial_features
    *
    cutoff_values.unsqueeze(-1)
)
```

The resulting tensor is:

```text
[E,32]
```

This can be supplied to the message-passing network.

---

## 24.8.173 Crystal Graph Data Object

We can now construct a more complete PyTorch Geometric object.

```python
from torch_geometric.data import Data


def structure_to_graph(

    structure,

    cutoff=5.0

):

    atomic_numbers = torch.tensor(

        [
            site.specie.Z
            for site in structure
        ],

        dtype=torch.long
    )

    frac_coords = torch.tensor(

        structure.frac_coords,

        dtype=torch.float32
    )

    lattice = torch.tensor(

        structure.lattice.matrix,

        dtype=torch.float32
    )

    (
        edge_index,
        edge_shifts,
        edge_distances
    ) = build_crystal_graph(

        structure,

        cutoff=cutoff
    )

    edge_vectors = (
        compute_edge_vectors(

            frac_coords,

            lattice,

            edge_index,

            edge_shifts
        )
    )

    data = Data(

        atomic_numbers=
            atomic_numbers,

        frac_coords=
            frac_coords,

        lattice=
            lattice,

        edge_index=
            edge_index,

        edge_shifts=
            edge_shifts,

        edge_vectors=
            edge_vectors,

        edge_distances=
            edge_distances
    )

    return data
```

Now:

```python
data = structure_to_graph(
    structure
)
```

and:

```python
print(data)
```

may produce a structure similar to:

```text
Data(
    atomic_numbers=[N],
    frac_coords=[N,3],
    lattice=[3,3],
    edge_index=[2,E],
    edge_shifts=[E,3],
    edge_vectors=[E,3],
    edge_distances=[E]
)
```

---

## 24.8.174 Adding Property Conditions

The crystal graph can also contain the target properties.

For example:

```python
property_vector = torch.tensor(

    [
        1.82,
        -2.71,
        5.24,
        210.0
    ],

    dtype=torch.float32
)
```

Then:

```python
data.properties = (
    property_vector
)
```

The graph now represents:

```text
Crystal
+
Target Properties
```

For training:

```text
Clean Crystal
       +
Property Condition
       ↓
Forward Diffusion
```

During generation:

```text
Desired Properties
       ↓
Diffusion Model
       ↓
Generated Crystal
```

---

## 24.8.175 Adding Atom Masks

For some architectures, it is useful to know which nodes are active.

For example:

```python
num_atoms = len(
    structure
)

atom_mask = torch.ones(

    num_atoms,

    dtype=torch.bool
)
```

Then:

```python
data.atom_mask = (
    atom_mask
)
```

For padded representations, the mask becomes more important.

For example:

```text
Maximum atoms = 64

Real atoms:
0–11

Padding:
12–63
```

The mask would be:

```text
[1,1,1,1,1,1,1,1,1,1,1,1,
 0,0,0,...,0]
```

---

## 24.8.176 Variable-Size Graphs in PyTorch Geometric

One major advantage of PyTorch Geometric is that graphs do not need to have the same number of nodes.

Suppose:

```text
Crystal A → 8 atoms

Crystal B → 12 atoms

Crystal C → 24 atoms
```

They can still be combined into a batch.

PyTorch Geometric internally creates a disconnected graph:

```text
Graph A          Graph B          Graph C

●──●             ●──●             ●──●
│  │             │  │             │  │
●──●             ●──●             ●──●
                                  │
                                  ●
```

There are no edges between different crystals.

The batching system also keeps track of which graph each node belongs to.

---

## 24.8.177 PyTorch Geometric Batch

```python
from torch_geometric.loader import DataLoader
```

A loader can be created:

```python
loader = DataLoader(

    dataset,

    batch_size=16,

    shuffle=True
)
```

Then:

```python
for batch in loader:

    print(
        batch
    )

    break
```

PyTorch Geometric automatically combines graph structures.

A `batch` vector identifies the graph associated with each node.

For example:

```text
batch =
[0,0,0,0,
 1,1,1,
 2,2,2,2,2]
```

means:

```text
Nodes 0–3
→ Crystal 0

Nodes 4–6
→ Crystal 1

Nodes 7–11
→ Crystal 2
```

---

## 24.8.178 Why the Batch Vector Matters

Suppose a message-passing network generates node representations:

```text
h₁
h₂
...
h_N
```

The model may need to convert these node-level features into a crystal-level representation.

For example:

```text
Node Features
      ↓
Global Pooling
      ↓
Crystal Representation
```

The `batch` vector tells the pooling function which nodes belong to which crystal.

For example:

```python
from torch_geometric.nn import global_mean_pool

crystal_embedding = (
    global_mean_pool(
        node_embeddings,
        batch.batch
    )
)
```

If the batch contains 16 crystals, the output might be:

```text
[16, hidden_dim]
```

---

## 24.8.179 Graph Neural Network Message Passing

The fundamental GNN operation can be written conceptually as:

```text
Node i
  │
  ├── Neighbor j
  ├── Neighbor k
  └── Neighbor l
       ↓
Messages
       ↓
Aggregation
       ↓
Updated Node Representation
```

A generic message-passing equation is:

```text
m_ij = φ(h_i, h_j, e_ij)
```

where:

```text
h_i
```

is the representation of atom `i`,

```text
h_j
```

is the representation of neighboring atom `j`,

and:

```text
e_ij
```

contains geometric information.

The node update is:

```text
h_i' =
ψ(
h_i,
Σ_j m_ij
)
```

This is the basic mechanism by which the network learns local chemical environments.

---

## 24.8.180 Simple Crystal Message-Passing Layer

A simplified implementation can be written as:

```python
class CrystalMessageLayer(
    nn.Module
):

    def __init__(

        self,

        hidden_dim,

        edge_dim

    ):

        super().__init__()

        self.message_mlp = nn.Sequential(

            nn.Linear(

                hidden_dim * 2
                + edge_dim,

                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(

                hidden_dim,

                hidden_dim
            )
        )

        self.update_mlp = nn.Sequential(

            nn.Linear(

                hidden_dim * 2,

                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(

                hidden_dim,

                hidden_dim
            )
        )

    def forward(

        self,

        x,

        edge_index,

        edge_attr

    ):

        src = edge_index[0]

        dst = edge_index[1]

        messages = self.message_mlp(

            torch.cat(

                [
                    x[src],
                    x[dst],
                    edge_attr
                ],

                dim=-1
            )
        )

        aggregation = torch.zeros_like(
            x
        )

        aggregation.index_add_(

            0,

            dst,

            messages
        )

        updated = self.update_mlp(

            torch.cat(

                [
                    x,
                    aggregation
                ],

                dim=-1
            )
        )

        return updated
```

This is a simplified implementation intended to expose the mechanism.

Production crystal models usually use more sophisticated equivariant architectures.

---

## 24.8.181 Building a Crystal Encoder

Several message-passing layers can be stacked.

```python
class CrystalEncoder(
    nn.Module
):

    def __init__(

        self,

        hidden_dim=128,

        edge_dim=32,

        num_layers=4

    ):

        super().__init__()

        self.embedding = nn.Embedding(

            119,

            hidden_dim
        )

        self.layers = nn.ModuleList(

            [

                CrystalMessageLayer(

                    hidden_dim,

                    edge_dim

                )

                for _ in range(
                    num_layers
                )

            ]
        )

    def forward(

        self,

        atomic_numbers,

        edge_index,

        edge_attr

    ):

        x = self.embedding(
            atomic_numbers
        )

        for layer in self.layers:

            x = x + layer(

                x,

                edge_index,

                edge_attr
            )

        return x
```

This encoder transforms:

```text
Atomic Numbers
+
Periodic Graph
+
Edge Features
```

into learned atomic representations.

---

## 24.8.182 From Crystal Encoder to Diffusion Model

The encoder can now be placed inside the generative architecture.

```text
Crystal
   │
   ▼
Periodic Graph
   │
   ▼
Crystal Encoder
   │
   ▼
Latent Node Representation
   │
   ├──────────────┐
   │              │
   ▼              ▼
Time            Property
Embedding       Embedding
   │              │
   └──────┬───────┘
          ▼
     Diffusion
      Network
          │
          ▼
   Noise / Score
   Prediction
```

This represents the central architecture of a property-conditioned crystal diffusion model.

---

## 24.8.183 Time Embedding

Diffusion models require information about the current noise level.

The timestep:

```text
t
```

is therefore converted into an embedding.

A standard sinusoidal embedding is:

```python
import math


def timestep_embedding(

    timesteps,

    dim

):

    half = dim // 2

    frequencies = torch.exp(

        -math.log(10000)
        *
        torch.arange(
            half,
            device=timesteps.device
        )
        /
        half
    )

    angles = (
        timesteps[:, None]
        *
        frequencies[None, :]
    )

    embedding = torch.cat(

        [
            torch.sin(angles),
            torch.cos(angles)
        ],

        dim=-1
    )

    return embedding
```

For example:

```python
t = torch.tensor(
    [0, 100, 500, 999]
)

embedding = timestep_embedding(
    t,
    128
)
```

produces:

```text
[4,128]
```

---

## 24.8.184 Property Embedding

The property vector is separately encoded.

```python
class PropertyEncoder(
    nn.Module
):

    def __init__(

        self,

        input_dim,

        hidden_dim

    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(

                input_dim,

                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(

                hidden_dim,

                hidden_dim
            )
        )

    def forward(self, x):

        return self.network(x)
```

For four properties:

```python
property_encoder = PropertyEncoder(

    input_dim=4,

    hidden_dim=128
)
```

Then:

```python
property_embedding = (
    property_encoder(
        properties
    )
)
```

produces:

```text
[batch_size,128]
```

---

## 24.8.185 Combining Time and Property Conditions

The time and property embeddings can be combined.

```python
condition = (

    time_embedding
    +
    property_embedding
)
```

The condition can then be injected into the node representations.

For example:

```python
condition = condition[
    batch.batch
]
```

so that each atom receives the condition associated with its crystal.

Then:

```python
x = x + condition
```

This gives every atom access to:

```text
Current Diffusion Time
+
Desired Material Properties
```

---

## 24.8.186 Conditional Crystal Diffusion Architecture

The complete simplified model becomes:

```python
class ConditionalCrystalDiffusion(
    nn.Module
):

    def __init__(

        self,

        hidden_dim=128,

        property_dim=4,

        edge_dim=32

    ):

        super().__init__()

        self.atom_embedding = nn.Embedding(

            119,

            hidden_dim
        )

        self.property_encoder = (
            PropertyEncoder(

                property_dim,

                hidden_dim
            )
        )

        self.time_projection = nn.Linear(

            hidden_dim,

            hidden_dim
        )

        self.message_layer = (
            CrystalMessageLayer(

                hidden_dim,

                edge_dim
            )
        )

        self.output_layer = nn.Linear(

            hidden_dim,

            3
        )

    def forward(

        self,

        atomic_numbers,

        edge_index,

        edge_attr,

        batch_index,

        property_condition,

        time_embedding

    ):

        x = self.atom_embedding(
            atomic_numbers
        )

        property_embedding = (
            self.property_encoder(
                property_condition
            )
        )

        time_embedding = (
            self.time_projection(
                time_embedding
            )
        )

        condition = (

            property_embedding
            +
            time_embedding
        )

        condition = condition[
            batch_index
        ]

        x = x + condition

        x = x + self.message_layer(

            x,

            edge_index,

            edge_attr
        )

        output = self.output_layer(
            x
        )

        return output
```

This is still a simplified model.

Its purpose is to demonstrate the complete information flow:

```text
Atomic Identity
+
Crystal Geometry
+
Periodic Neighborhood
+
Diffusion Time
+
Target Properties
        ↓
Neural Network
        ↓
Predicted Denoising Quantity
```

---

## 24.8.187 What Should the Model Predict?

The exact prediction target depends on the diffusion formulation.

Possible targets include:

```text
Noise ε
```

or:

```text
Score ∇x log p_t(x)
```

or:

```text
Velocity v
```

A common DDPM-style formulation predicts the injected noise.

Suppose:

```text
x₀
```

is the clean crystal representation.

The forward process generates:

```text
x_t
```

The model receives:

```text
x_t
+
t
+
condition
```

and predicts:

```text
ε̂
```

which approximates the actual noise:

```text
ε
```

The training objective is then:

```text
L =
||ε - ε̂||²
```

This connects the crystal graph representation to the mathematical diffusion process.

---

## 24.8.188 Important Representation Issue

At this stage, an important scientific limitation must be emphasized.

A crystal is not simply:

```text
Node Features
+
Edges
```

A generative model must ultimately represent quantities such as:

```text
Atomic Species
Atomic Positions
Lattice
Periodic Symmetry
Composition
```

and these variables have different mathematical properties.

For example:

```text
Atomic species
→ discrete

Fractional coordinates
→ continuous

Lattice
→ continuous matrix

Number of atoms
→ discrete
```

Therefore, a complete crystal diffusion model must decide how to jointly generate these different variables.

This is one of the central difficulties of crystal generative modeling.

---

## 24.8.189 Discrete and Continuous Variables

The generation problem can be represented as:

```text
Crystal
 │
 ├── Atomic Species
 │       ↓
 │    Discrete
 │
 ├── Coordinates
 │       ↓
 │    Continuous
 │
 ├── Lattice
 │       ↓
 │    Continuous
 │
 └── Number of Atoms
         ↓
      Discrete
```

A simple diffusion process operating only on Cartesian coordinates therefore does not solve the complete crystal-generation problem.

The model must additionally determine:

```text
Which elements exist?

How many atoms exist?

Where are they located?

What is the unit cell?

```

This motivates more advanced crystal generative architectures.

---

## 24.8.190 The Complete Research Pipeline

At this point, the implementation pipeline can be summarized as:

```text
Materials Database
        ↓
pymatgen
        ↓
Structure Standardization
        ↓
Periodic Neighbor Search
        ↓
Crystal Graph
        ↓
PyTorch Geometric Data
        ↓
Node Embeddings
        ↓
Edge/Radial Features
        ↓
Crystal Encoder
        ↓
Time Conditioning
        ↓
Property Conditioning
        ↓
Diffusion Network
        ↓
Noise / Score Prediction
        ↓
Diffusion Training
        ↓
Crystal Generation
        ↓
Chemical Validation
        ↓
Property Prediction
        ↓
DFT Verification
```

This pipeline is the practical connection between the theoretical diffusion framework and an actual Materials Informatics implementation.

The next stage is therefore to move from **graph construction** into the **forward diffusion process itself**: defining the noise schedule, adding noise to crystal representations, implementing the diffusion objective, constructing the training loop, and monitoring whether the model learns the crystal distribution.

## 24.8.191 Forward Diffusion for Crystal Representations

The previous section established the complete crystal-graph representation.

The next step is to implement the diffusion process itself.

The fundamental idea of diffusion modeling is to transform a structured data distribution into a simple noise distribution through a sequence of controlled perturbations.

For a crystal representation,

```text
Clean Crystal
     ↓
Small Noise
     ↓
More Noise
     ↓
More Noise
     ↓
Almost Random
```

At the end of the forward process, the original crystal information is largely destroyed.

The neural network is then trained to reverse this process.

```text
Random Noise
     ↓
Denoising Step
     ↓
Less Noise
     ↓
Crystal-like Structure
     ↓
Valid Crystal
```

This gives the central generative mechanism:

```text
Forward Process:

Crystal → Noise

Reverse Process:

Noise → Crystal
```

---

## 24.8.192 Mathematical Definition of the Forward Process

Let

```text
x₀
```

represent the original crystal representation.

A diffusion process gradually adds Gaussian noise.

At timestep `t`:

```text
x_t
```

is the noisy representation.

The forward transition can be written as

```text
q(x_t | x_{t-1})
=
N(
√(1-β_t)x_{t-1},
β_t I
)
```

where:

```text
β_t
```

is the noise variance at timestep `t`.

Define:

```text
α_t = 1 - β_t
```

and the cumulative product:

```text
ᾱ_t = ∏_{s=1}^{t} α_s
```

Then the noisy sample can be generated directly from the original crystal:

```text
x_t
=
√ᾱ_t x₀
+
√(1-ᾱ_t) ε
```

where:

```text
ε ~ N(0,I)
```

This equation is extremely important because it means that we do not need to apply every previous diffusion step to obtain `x_t`.

We can sample `x_t` directly.

---

## 24.8.193 Why the Closed-Form Equation Matters

Suppose the diffusion process contains:

```text
T = 1000
```

timesteps.

A naive implementation would require:

```text
x₀
 ↓
x₁
 ↓
x₂
 ↓
...
 ↓
x₅₀₀
```

to obtain `x₅₀₀`.

This would be computationally inefficient.

Instead, the closed-form expression allows:

```python
x_t = (
    sqrt_alpha_bar_t * x_0
    +
    sqrt_one_minus_alpha_bar_t * noise
)
```

Therefore:

```text
x₀
+
t
+
random noise
↓
x_t
```

can be computed directly.

---

## 24.8.194 Implementing the Noise Schedule

The first component required by the diffusion process is a noise schedule.

A simple linear schedule is:

```python
import torch


def linear_beta_schedule(
    timesteps,
    beta_start=1e-4,
    beta_end=0.02
):

    return torch.linspace(
        beta_start,
        beta_end,
        timesteps
    )
```

For example:

```python
T = 1000

betas = linear_beta_schedule(
    T
)

print(
    betas.shape
)
```

Output:

```text
torch.Size([1000])
```

The first timestep has a small amount of noise:

```text
β₁ ≈ 0.0001
```

while later timesteps have progressively larger noise.

---

## 24.8.195 Computing α and ᾱ

We define:

```python
alphas = 1.0 - betas
```

Then:

```python
alpha_bars = torch.cumprod(
    alphas,
    dim=0
)
```

The complete initialization becomes:

```python
T = 1000

betas = linear_beta_schedule(
    T
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)
```

The resulting quantities represent:

```text
betas
 ↓
Noise added at each step

alphas
 ↓
Signal retained at each step

alpha_bars
 ↓
Total signal retained from x₀
```

---

## 24.8.196 Inspecting the Noise Schedule

It is useful to inspect the schedule.

```python
print(
    betas[:10]
)

print(
    alpha_bars[:10]
)

print(
    alpha_bars[-1]
)
```

Typically:

```text
Early timesteps
→ ᾱ_t close to 1

Late timesteps
→ ᾱ_t much smaller
```

Therefore:

```text
Early t:

x_t ≈ x₀

Late t:

x_t ≈ noise
```

This is the fundamental behavior required for diffusion training.

---

## 24.8.197 Visualizing the Signal-to-Noise Transition

Conceptually:

```text
Signal
1.0 │████████████████████
    │██████████████████
    │████████████████
    │████████████
    │████████
    │████
0.0 └──────────────────────
       Diffusion Time
```

At low `t`, the crystal remains recognizable.

At high `t`, the crystal representation becomes increasingly noisy.

For a materials model, this means that the network must learn denoising across a wide range of structural corruption levels.

---

## 24.8.198 Extracting Timestep Coefficients

During training, each sample in a batch may use a different timestep.

For example:

```text
Crystal 1 → t = 20
Crystal 2 → t = 410
Crystal 3 → t = 875
Crystal 4 → t = 120
```

We therefore need a function to extract the appropriate coefficient.

```python
def extract(
    values,
    timesteps,
    shape
):

    batch_size = timesteps.shape[0]

    out = values.gather(
        0,
        timesteps
    )

    return out.reshape(
        batch_size,
        *((1,) * (len(shape) - 1))
    )
```

This allows us to obtain:

```text
ᾱ₂₀
ᾱ₄₁₀
ᾱ₈₇₅
ᾱ₁₂₀
```

for the four samples simultaneously.

---

## 24.8.199 Forward Noising Function

We can now implement the complete forward diffusion equation.

```python
def q_sample(
    x_start,
    timesteps,
    alpha_bars,
    noise=None
):

    if noise is None:

        noise = torch.randn_like(
            x_start
        )

    sqrt_alpha_bar = torch.sqrt(
        alpha_bars
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bars
    )

    sqrt_alpha_bar_t = extract(
        sqrt_alpha_bar,
        timesteps,
        x_start.shape
    )

    sqrt_one_minus_alpha_bar_t = extract(
        sqrt_one_minus_alpha_bar,
        timesteps,
        x_start.shape
    )

    noisy_sample = (
        sqrt_alpha_bar_t
        * x_start
        +
        sqrt_one_minus_alpha_bar_t
        * noise
    )

    return (
        noisy_sample,
        noise
    )
```

This function returns both:

```text
x_t
```

and:

```text
ε
```

The original noise is needed because the model will later be trained to predict it.

---

## 24.8.200 Testing the Forward Process

Suppose we have a simple crystal feature tensor:

```python
x = torch.randn(
    8,
    32
)
```

where:

```text
8
↓
number of nodes

32
↓
feature dimension
```

We can select timesteps:

```python
timesteps = torch.randint(
    0,
    T,
    (8,)
)
```

Then:

```python
x_noisy, noise = q_sample(

    x,

    timesteps,

    alpha_bars
)
```

Inspect:

```python
print(
    x.shape
)

print(
    x_noisy.shape
)

print(
    noise.shape
)
```

All three should have:

```text
[8,32]
```

---

## 24.8.201 Comparing Different Diffusion Timesteps

We can investigate the effect of timestep.

```python
x = torch.randn(
    1,
    32
)
```

Then create:

```python
timesteps = torch.tensor([
    0
])
```

and:

```python
x_0_noisy, _ = q_sample(
    x,
    timesteps,
    alpha_bars
)
```

For a later timestep:

```python
timesteps = torch.tensor([
    500
])
```

we obtain:

```python
x_500_noisy, _ = q_sample(
    x,
    timesteps,
    alpha_bars
)
```

And near the end:

```python
timesteps = torch.tensor([
    999
])
```

we obtain:

```python
x_999_noisy, _ = q_sample(
    x,
    timesteps,
    alpha_bars
)
```

Conceptually:

```text
x₀
↓
Strong crystal information

x₅₀₀
↓
Partially corrupted representation

x₉₉₉
↓
Almost pure noise
```

---

## 24.8.202 Diffusion Training Objective

The diffusion model receives the noisy sample:

```text
x_t
```

along with:

```text
t
```

and the desired property:

```text
y
```

The model predicts the original noise:

```text
ε̂θ(x_t,t,y)
```

The training target is the actual noise:

```text
ε
```

The standard objective is:

```text
L =
E[
||ε - ε̂θ(x_t,t,y)||²
]
```

In PyTorch:

```python
loss = torch.mean(
    (
        predicted_noise
        -
        true_noise
    ) ** 2
)
```

This is the central training loss of the simplified conditional diffusion model.

---

## 24.8.203 Why Predicting Noise Works

At first this may seem surprising.

The network is not directly told:

```text
"Generate crystal X."
```

Instead, it learns:

```text
"What noise was added to this crystal?"
```

If the model becomes good at predicting the noise, the reverse process can gradually remove it.

Therefore:

```text
Noise Prediction
        ↓
Denoising
        ↓
Data Generation
```

This is the key conceptual connection between training and generation.

---

## 24.8.204 Noise Prediction Network

A simple baseline network can be constructed as:

```python
class NoisePredictor(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        property_dim,
        time_dim
    ):

        super().__init__()

        self.input_layer = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.time_layer = nn.Linear(
            time_dim,
            hidden_dim
        )

        self.property_layer = nn.Linear(
            property_dim,
            hidden_dim
        )

        self.network = nn.Sequential(

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            )
        )

    def forward(
        self,
        x,
        time_embedding,
        property_condition
    ):

        h = self.input_layer(
            x
        )

        t = self.time_layer(
            time_embedding
        )

        p = self.property_layer(
            property_condition
        )

        h = h + t + p

        return self.network(
            h
        )
```

This is a simple baseline.

The architecture can later be replaced with a graph-based or equivariant network.

---

## 24.8.205 Adding Graph Message Passing

For crystals, the network should account for neighboring atoms.

A graph-based denoising model can therefore be written as:

```python
class GraphNoisePredictor(
    nn.Module
):

    def __init__(
        self,
        hidden_dim,
        edge_dim,
        property_dim,
        time_dim
    ):

        super().__init__()

        self.atom_embedding = nn.Embedding(
            119,
            hidden_dim
        )

        self.property_encoder = (
            PropertyEncoder(
                property_dim,
                hidden_dim
            )
        )

        self.time_encoder = nn.Linear(
            time_dim,
            hidden_dim
        )

        self.message_layer_1 = (
            CrystalMessageLayer(
                hidden_dim,
                edge_dim
            )
        )

        self.message_layer_2 = (
            CrystalMessageLayer(
                hidden_dim,
                edge_dim
            )
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            3
        )

    def forward(
        self,
        atomic_numbers,
        edge_index,
        edge_attr,
        batch_index,
        property_condition,
        time_embedding
    ):

        x = self.atom_embedding(
            atomic_numbers
        )

        property_embedding = (
            self.property_encoder(
                property_condition
            )
        )

        time_embedding = (
            self.time_encoder(
                time_embedding
            )
        )

        condition = (
            property_embedding
            +
            time_embedding
        )

        condition = condition[
            batch_index
        ]

        x = x + condition

        x = x + self.message_layer_1(
            x,
            edge_index,
            edge_attr
        )

        x = x + self.message_layer_2(
            x,
            edge_index,
            edge_attr
        )

        noise_prediction = (
            self.output_layer(x)
        )

        return noise_prediction
```

This architecture explicitly combines:

```text
Atomic Identity
+
Neighborhood Information
+
Geometry
+
Diffusion Time
+
Property Condition
```

---

## 24.8.206 What Does the Network Predict for a Crystal?

This question depends on the chosen representation.

If the diffusion variable is atomic Cartesian position:

```text
x = [x,y,z]
```

then the model may predict:

```text
ε_x
ε_y
ε_z
```

for every atom.

Therefore:

```text
N atoms
×
3 coordinates
```

gives:

```text
[N,3]
```

noise predictions.

For example:

```python
predicted_noise = model(
    ...
)

print(
    predicted_noise.shape
)
```

could produce:

```text
torch.Size([32,3])
```

for a crystal containing 32 atoms.

---

## 24.8.207 Coordinate Diffusion

A simplified coordinate diffusion process can therefore be written as:

```text
Original Coordinates

R₀
 ↓
Forward Diffusion
 ↓
R_t
 ↓
Noise Predictor
 ↓
ε̂
 ↓
Reverse Diffusion
 ↓
R_{t-1}
```

The model gradually learns how atomic positions should move toward chemically plausible configurations.

However, coordinate diffusion alone has an important limitation.

The generated atoms may move into physically impossible configurations.

For example:

```text
Atom A
   ●
   ↓
Atom B
   ●
```

could collapse to:

```text
●●
```

with an unrealistically small interatomic distance.

Therefore, geometric and chemical constraints are essential.

---

## 24.8.208 Periodic Coordinate Diffusion

Crystal coordinates are periodic.

If fractional coordinates are used:

```text
x ∈ [0,1)
y ∈ [0,1)
z ∈ [0,1)
```

then a coordinate near:

```text
x = 0.99
```

is physically close to:

```text
x = 0.01
```

because of periodicity.

A conventional Euclidean representation does not automatically capture this.

For example:

```text
|0.99 - 0.01|
=
0.98
```

but under periodic boundary conditions the minimum wrapped distance is:

```text
0.02
```

Therefore, periodic geometry must be handled explicitly.

---

## 24.8.209 Minimum-Image Convention

A common periodic operation is:

```python
def minimum_image(
    delta
):

    return (
        delta
        -
        torch.round(delta)
    )
```

For a fractional displacement:

```python
delta = (
    frac_j
    -
    frac_i
)
```

we can write:

```python
delta_wrapped = minimum_image(
    delta
)
```

For example:

```text
0.98
```

becomes:

```text
-0.02
```

because:

```text
0.98 - round(0.98)
=
-0.02
```

This operation is extremely important when working directly in fractional coordinates.

---

## 24.8.210 Periodic Distance from Fractional Coordinates

Given fractional displacement:

```text
Δf
```

the minimum-image displacement is:

```python
delta_frac = (
    frac_j
    -
    frac_i
)

delta_frac = (
    delta_frac
    -
    torch.round(
        delta_frac
    )
)
```

Then:

```python
delta_cart = (
    delta_frac
    @
    lattice
)
```

and:

```python
distance = torch.linalg.norm(
    delta_cart
)
```

This produces the periodic minimum distance.

For non-orthogonal lattices, however, more careful periodic neighbor handling may be required than simply applying a component-wise rounding operation.

This is one reason why structure libraries such as `pymatgen` are valuable during preprocessing.

---

## 24.8.211 Training Data Preparation

A practical dataset may contain:

```text
structure
band_gap
formation_energy
density
bulk_modulus
```

For example:

```python
import pandas as pd

df = pd.read_csv(
    "materials_dataset.csv"
)

print(
    df.columns
)
```

Possible output:

```text
[
    "structure",
    "band_gap",
    "formation_energy",
    "density",
    "bulk_modulus"
]
```

The structures can be converted to `pymatgen` objects.

```python
from pymatgen.core import Structure

structures = [

    Structure.from_dict(
        structure_dict
    )

    for structure_dict
    in df["structure"]
]
```

---

## 24.8.212 Building the Graph Dataset

We can convert each structure into a graph.

```python
graphs = []

for structure in structures:

    graph = structure_to_graph(
        structure,
        cutoff=5.0
    )

    graphs.append(
        graph
    )
```

Then attach properties:

```python
property_columns = [

    "band_gap",
    "formation_energy",
    "density",
    "bulk_modulus"
]

for graph, row in zip(
    graphs,
    df[property_columns].values
):

    graph.properties = torch.tensor(

        row,

        dtype=torch.float32
    )
```

The resulting dataset contains both:

```text
Crystal Graph
+
Scientific Targets
```

---

## 24.8.213 Property Normalization

The properties should generally be standardized before conditioning.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

properties = df[
    property_columns
].values

properties_scaled = (
    scaler.fit_transform(
        properties
    )
)
```

The normalized values can then be attached:

```python
for graph, condition in zip(
    graphs,
    properties_scaled
):

    graph.properties = torch.tensor(

        condition,

        dtype=torch.float32
    )
```

The scaler must be saved.

```python
import joblib

joblib.dump(
    scaler,
    "property_scaler.pkl"
)
```

This is important because the same transformation must be applied during inference.

---

## 24.8.214 Why the Scaler Must Be Saved

Suppose the training dataset has:

```text
Band Gap
mean = 2.1 eV

standard deviation = 1.2 eV
```

A target:

```text
2.5 eV
```

is converted using:

```text
z =
(2.5 - 2.1)
/
1.2
```

During generation, the model receives the normalized value.

After generation, the researcher may want the original physical units.

The inverse transformation is:

```python
predicted_properties = (
    scaler.inverse_transform(
        predicted_scaled
    )
)
```

Therefore:

```text
Physical Property
       ↓
Scaler
       ↓
Normalized Property
       ↓
Diffusion Model
```

and later:

```text
Predicted Property
       ↓
Inverse Scaler
       ↓
Physical Units
```

---

## 24.8.215 PyTorch Dataset for Crystal Diffusion

A complete dataset class can be written as:

```python
from torch.utils.data import Dataset


class CrystalDiffusionDataset(
    Dataset
):

    def __init__(
        self,
        graphs
    ):

        self.graphs = graphs

    def __len__(self):

        return len(
            self.graphs
        )

    def __getitem__(
        self,
        index
    ):

        return self.graphs[index]
```

Then:

```python
dataset = (
    CrystalDiffusionDataset(
        graphs
    )
)
```

and:

```python
print(
    len(dataset)
)
```

---

## 24.8.216 PyTorch Geometric DataLoader

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(

    dataset,

    batch_size=16,

    shuffle=True
)
```

Then:

```python
for batch in loader:

    print(
        batch
    )

    break
```

The batch now contains multiple crystals.

---

## 24.8.217 Sampling Diffusion Timesteps During Training

Each training example should usually receive a randomly selected timestep.

```python
batch_size = (
    batch.properties.shape[0]
)

timesteps = torch.randint(

    low=0,

    high=T,

    size=(
        batch_size,
    ),

    device=batch.properties.device
)
```

This produces something like:

```text
[12, 847, 391, 4, 723, ...]
```

Each crystal therefore experiences a different amount of corruption.

---

## 24.8.218 Constructing Noisy Coordinates

Suppose:

```python
clean_coordinates = (
    batch.pos
)
```

We sample noise:

```python
noise = torch.randn_like(
    clean_coordinates
)
```

Then:

```python
noisy_coordinates, true_noise = (
    q_sample(

        clean_coordinates,

        timesteps[
            batch.batch
        ],

        alpha_bars,

        noise=noise
    )
)
```

Notice the use of:

```python
timesteps[
    batch.batch
]
```

because `batch.batch` maps every atom to its crystal.

If crystal 0 uses:

```text
t = 100
```

then every atom belonging to crystal 0 must receive the same diffusion timestep.

This is essential.

---

## 24.8.219 Training Step

A simplified training step is:

```python
optimizer.zero_grad()

predicted_noise = model(

    atomic_numbers=batch.atomic_numbers,

    edge_index=batch.edge_index,

    edge_attr=batch.edge_attr,

    batch_index=batch.batch,

    property_condition=batch.properties,

    time_embedding=time_embedding
)

loss = torch.mean(

    (
        predicted_noise
        -
        true_noise
    ) ** 2
)

loss.backward()

optimizer.step()
```

The sequence is:

```text
Crystal
 ↓
Add Noise
 ↓
Noisy Crystal
 ↓
Model
 ↓
Predicted Noise
 ↓
Compare with True Noise
 ↓
MSE Loss
 ↓
Backpropagation
 ↓
Parameter Update
```

---

## 24.8.220 Complete Simplified Training Loop

A complete baseline training loop can be written as:

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = GraphNoisePredictor(

    hidden_dim=128,

    edge_dim=32,

    property_dim=4,

    time_dim=128

).to(device)

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-4,

    weight_decay=1e-6
)
```

Training:

```python
num_epochs = 100

for epoch in range(
    num_epochs
):

    model.train()

    epoch_loss = 0.0

    for batch in loader:

        batch = batch.to(
            device
        )

        optimizer.zero_grad()

        batch_size = (
            batch.properties.shape[0]
        )

        timesteps = torch.randint(

            0,

            T,

            (
                batch_size,
            ),

            device=device
        )

        atom_timesteps = (
            timesteps[
                batch.batch
            ]
        )

        noise = torch.randn_like(
            batch.pos
        )

        noisy_pos, true_noise = (
            q_sample(

                batch.pos,

                atom_timesteps,

                alpha_bars.to(device),

                noise=noise
            )
        )

        time_emb = timestep_embedding(

            timesteps,

            128
        )

        predicted_noise = model(

            atomic_numbers=
                batch.atomic_numbers,

            edge_index=
                batch.edge_index,

            edge_attr=
                batch.edge_attr,

            batch_index=
                batch.batch,

            property_condition=
                batch.properties,

            time_embedding=
                time_emb
        )

        loss = torch.mean(

            (
                predicted_noise
                -
                true_noise
            ) ** 2
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += (
            loss.item()
        )

    epoch_loss /= len(
        loader
    )

    print(

        f"Epoch "
        f"{epoch + 1:03d} "
        f"| Loss "
        f"{epoch_loss:.6f}"
    )
```

This is a complete conceptual baseline for diffusion training.

It is intentionally simplified.

A production-quality crystal diffusion system requires more sophisticated treatment of periodic geometry, equivariance, lattice generation, variable atom counts, discrete species, and chemical validity.

---

## 24.8.221 Why Gradient Clipping Is Used

Generative models can occasionally produce large gradients.

This may destabilize training.

The following operation:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

limits the gradient norm.

Conceptually:

```text
Very Large Gradient
       ↓
Gradient Clipping
       ↓
Controlled Parameter Update
```

This can improve numerical stability, particularly in deeper generative networks.

---

## 24.8.222 Monitoring Training

The training loss should be recorded.

```python
loss_history = []

for epoch in range(
    num_epochs
):

    ...

    loss_history.append(
        epoch_loss
    )
```

The values can then be inspected.

```python
print(
    loss_history[-10:]
)
```

A simple plot can be produced using Matplotlib:

```python
import matplotlib.pyplot as plt

plt.plot(
    loss_history
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Noise Prediction Loss"
)

plt.title(
    "Crystal Diffusion Training"
)

plt.show()
```

The goal is not simply to obtain a low training loss.

The generated structures must eventually be evaluated.

Therefore:

```text
Training Loss
      ↓
Model Learning
      ↓
Generation Quality
      ↓
Chemical Validity
      ↓
Property Accuracy
```

are all separate questions.

---

## 24.8.223 Saving the Trained Model

After training:

```python
torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            num_epochs,

        "loss":
            epoch_loss

    },

    "crystal_diffusion_model.pt"
)
```

The property scaler should also be stored:

```python
joblib.dump(

    scaler,

    "property_scaler.pkl"
)
```

The diffusion schedule should also be saved or reproducibly reconstructed.

For example:

```python
torch.save(

    {

        "betas":
            betas,

        "alphas":
            alphas,

        "alpha_bars":
            alpha_bars

    },

    "diffusion_schedule.pt"
)
```

This makes the experiment reproducible.

---

## 24.8.224 Loading the Model

Later, the model can be restored:

```python
checkpoint = torch.load(

    "crystal_diffusion_model.pt",

    map_location=device
)

model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)

optimizer.load_state_dict(

    checkpoint[
        "optimizer_state_dict"
    ]
)
```

The model should then be placed into evaluation mode:

```python
model.eval()
```

---

## 24.8.225 From Training to Generation

Training teaches the model:

```text
How to Remove Noise
```

Generation reverses the process.

We begin with:

```text
Random Noise
```

and repeatedly apply the learned denoising model.

Conceptually:

```text
x_T
 ↓
x_{T-1}
 ↓
x_{T-2}
 ↓
...
 ↓
x_2
 ↓
x_1
 ↓
x_0
```

where:

```text
x_T
```

is approximately Gaussian noise and:

```text
x_0
```

should resemble a valid crystal representation.

---

## 24.8.226 Reverse Diffusion Equation

For a DDPM-style model, the reverse transition can be written conceptually as:

```text
pθ(x_{t-1}|x_t)
=
N(
μθ(x_t,t),
Σ_t
)
```

The model predicts the mean of the reverse distribution.

When the model predicts noise:

```text
ε̂θ
```

the reverse mean can be constructed from the predicted noise and the diffusion coefficients.

A commonly used form is:

```text
μθ(x_t,t)
=
1/√α_t
[
x_t
-
(1-α_t)
/
√(1-ᾱ_t)
ε̂θ
]
```

The implementation should follow the exact diffusion parameterization used during training.

This is important because mixing equations from different diffusion parameterizations can lead to an incorrect sampler.

---

## 24.8.227 Implementing One Reverse Step

A simplified DDPM reverse step can be implemented as:

```python
def p_sample(

    model,

    x_t,

    timestep,

    betas,

    alphas,

    alpha_bars,

    predicted_noise

):

    beta_t = betas[timestep]

    alpha_t = alphas[timestep]

    alpha_bar_t = (
        alpha_bars[timestep]
    )

    mean = (

        1.0
        /
        torch.sqrt(alpha_t)

    ) * (

        x_t

        -
        (
            beta_t
            /
            torch.sqrt(
                1.0
                -
                alpha_bar_t
            )
        )
        *
        predicted_noise
    )

    if timestep == 0:

        return mean

    noise = torch.randn_like(
        x_t
    )

    variance = torch.sqrt(
        beta_t
    )

    return (

        mean
        +
        variance
        * noise
    )
```

This illustrates the structure of a reverse diffusion step.

For a research implementation, the exact posterior variance and parameterization should be implemented carefully rather than relying on this simplified expression.

---

## 24.8.228 Conditional Generation

The major difference between unconditional and conditional generation occurs in the model input.

Unconditional:

```text
Noise
+
Time
 ↓
Model
```

Conditional:

```text
Noise
+
Time
+
Target Property
 ↓
Model
```

For example:

```python
target = torch.tensor(

    [
        [
            1.70,
            -2.80,
            5.00,
            220.0
        ]
    ],

    dtype=torch.float32,

    device=device
)
```

The target must first be transformed using the same scaler used during training.

```python
target_scaled = (
    scaler.transform(
        target.cpu().numpy()
    )
)
```

Then:

```python
target_scaled = torch.tensor(

    target_scaled,

    dtype=torch.float32,

    device=device
)
```

The model now receives:

```text
Target:

Band Gap = 1.70 eV
Formation Energy = -2.80
Density = 5.00
Bulk Modulus = 220
```

in normalized form.

---

## 24.8.229 Generating Multiple Candidates

Materials discovery rarely requires only one candidate.

Suppose:

```python
num_candidates = 1000
```

Then:

```python
for i in range(
    num_candidates
):

    crystal = generate_crystal(
        target_condition
    )

    candidates.append(
        crystal
    )
```

The result is:

```text
Target Condition
       ↓
       ↓
1000 Independent Samples
       ↓
Candidate Materials
```

This is one of the major advantages of generative models over deterministic prediction.

---

## 24.8.230 Why Multiple Samples Matter

Suppose the target is:

```text
Eg = 1.8 eV
```

There may be many possible structures:

```text
Candidate A → 1.79 eV
Candidate B → 1.82 eV
Candidate C → 1.76 eV
Candidate D → 1.81 eV
```

A generative model should ideally explore this design space rather than repeatedly returning nearly identical structures.

Therefore, the evaluation must consider:

```text
Property Accuracy
+
Chemical Validity
+
Structural Diversity
```

rather than property accuracy alone.

---

## 24.8.231 Random Seeds and Reproducibility

Generative models are stochastic.

For reproducible experiments:

```python
import random
import numpy as np
import torch


seed = 42

random.seed(
    seed
)

np.random.seed(
    seed
)

torch.manual_seed(
    seed
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        seed
    )
```

However, complete bitwise reproducibility on GPUs may require additional deterministic settings and can reduce computational performance.

The important scientific principle is:

```text
Randomness should be controlled
when comparing experiments.
```

---

## 24.8.232 The First Complete Conditional Diffusion Workflow

The complete implementation developed so far can be summarized as:

```text
Materials Dataset
        ↓
pymatgen Structure
        ↓
Periodic Neighbor Search
        ↓
Crystal Graph
        ↓
PyTorch Geometric Data
        ↓
Property Normalization
        ↓
Random Diffusion Timestep
        ↓
Forward Noise Addition
        ↓
Noisy Crystal
        ↓
Graph Neural Network
        ↓
Time + Property Conditioning
        ↓
Predicted Noise
        ↓
MSE Loss
        ↓
Backpropagation
        ↓
Trained Diffusion Model
        ↓
Reverse Diffusion
        ↓
Generated Candidate
        ↓
Chemical Validation
        ↓
Property Prediction
        ↓
DFT
```

This represents a complete **baseline research implementation pipeline**.

---

## 24.8.233 Important Limitation of the Baseline

The implementation above should not be mistaken for a production crystal generator.

It demonstrates the architecture and training mechanics, but several difficult problems remain.

The baseline currently assumes that the crystal representation is already available and that its graph structure can be constructed.

A complete generative system must additionally generate:

```text
1. Number of atoms

2. Atomic species

3. Atomic positions

4. Lattice parameters

5. Periodic structure

6. Chemically valid bonding

7. Physically reasonable density

8. Stable or metastable structures
```

Therefore:

```text
Graph Diffusion
```

is only one component of the full problem.

---

## 24.8.234 The Crystal Generation Bottleneck

A conventional image diffusion model generates:

```text
Pixel
Pixel
Pixel
...
```

all inside a fixed rectangular domain.

Crystal generation is considerably more complicated:

```text
Crystal
 │
 ├── Variable number of atoms
 │
 ├── Discrete atom identities
 │
 ├── Continuous coordinates
 │
 ├── Periodic boundary conditions
 │
 ├── Variable lattice
 │
 └── Symmetry constraints
```

Therefore, the model must learn a structured object with mixed discrete and continuous variables.

This is why materials generative modeling remains an active research area.

---

## 24.8.235 Research-Level Architecture

A more complete crystal diffusion architecture can be conceptualized as:

```text
                 Target Properties
                        │
                        ▼
                Property Encoder
                        │
                        ▼
Random Noise ─────► Diffusion Model
                        │
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
        Species      Positions    Lattice
        Generator    Generator    Generator
            │           │           │
            └───────────┼───────────┘
                        ▼
                 Crystal Structure
                        │
                        ▼
                 Chemical Filters
                        │
                        ▼
                Property Predictors
                        │
                        ▼
                     DFT
```

This is much closer to the complete inverse-design problem.

---

## 24.8.236 From Baseline to Research Implementation

The baseline implementation is valuable because it establishes the core mechanics:

```text
Forward Diffusion
Reverse Diffusion
Noise Prediction
Graph Representation
Property Conditioning
Training
Sampling
```

The next research-level improvements should address:

```text
Periodic Equivariance

Lattice Generation

Atomic Species Generation

Variable Atom Counts

Symmetry

Chemical Validity

Energy-Based Filtering

Property Guidance

Classifier-Free Guidance

Sampling Efficiency
```

These components transform a simple educational diffusion model into a more realistic Materials Informatics generative framework.

---

## 24.8.237 Practical Research Checklist

Before claiming that a conditional crystal diffusion model works, the researcher should verify:

```text
[ ] Crystal structures are standardized

[ ] Periodic neighbors are correctly constructed

[ ] Edge distances are correct

[ ] Periodic image shifts are preserved

[ ] Property values are normalized

[ ] Training and validation sets are separated

[ ] Diffusion schedule is reproducible

[ ] Noise prediction loss decreases

[ ] Validation loss is monitored

[ ] Generated structures are chemically valid

[ ] Duplicate structures are removed

[ ] Property predictions are accurate

[ ] Structural diversity is measured

[ ] Generated candidates are compared
    against the training distribution

[ ] Promising candidates are verified
    using higher-fidelity calculations
```

This checklist prevents a common mistake in generative materials research: interpreting a decreasing neural-network loss as proof that useful materials have been generated.

A generative model is successful only when the generated structures are simultaneously:

```text
Valid
+
Novel
+
Diverse
+
Property-Relevant
+
Physically Plausible
```

---

## 24.8.238 The Central Research Principle

The most important lesson from this implementation is that generative materials discovery is not simply:

```text
Train Model
↓
Generate Materials
```

Instead, it is an iterative scientific pipeline:

```text
              ┌─────────────────────┐
              │ Materials Dataset   │
              └──────────┬──────────┘
                         ↓
                 Representation
                         ↓
                Generative Model
                         ↓
                 Candidate Crystal
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
       Validity Check        Property Prediction
              │                     │
              └──────────┬──────────┘
                         ↓
                   Candidate Ranking
                         ↓
                         DFT
                         ↓
                 Experimental Testing
                         ↓
                  New Materials Data
                         │
                         └──────────────►
                              Retraining
```

The final arrow is especially important.

The discovery process can become a **closed-loop Materials Informatics system**.

New calculations and experiments produce new data.

That data improves the predictive models.

Improved models guide generation.

New generation produces better candidates.

Therefore:

```text
Generation
    ↓
Prediction
    ↓
DFT
    ↓
Experiment
    ↓
New Data
    ↓
Retraining
    ↓
Improved Generation
```

This closed-loop concept connects generative AI with active learning, autonomous materials discovery, and ultimately self-improving computational materials research.

The next section should therefore move beyond the basic DDPM implementation and develop **equivariant and periodic-aware crystal diffusion**, where atomic coordinates, lattice geometry, and symmetry are treated according to their physical transformation properties rather than as ordinary Euclidean feature vectors.

## 24.8.239 Equivariance in Crystal Diffusion

The baseline diffusion model developed in the previous sections treats crystal coordinates as ordinary numerical vectors.

This is useful for understanding diffusion mechanics, but it is not sufficient for a physically rigorous crystal-generation model.

The central reason is that crystal structures obey geometric symmetries.

If the entire crystal is rotated in space, its physical properties do not change.

For example,

```text
Original Crystal

      ●
     / \
    ●---●


Rotated Crystal

        ●
       / \
      ●---●
```

The Cartesian coordinates change.

However,

```text
Composition
Bond lengths
Bond angles
Energy
Band gap
Density
```

do not fundamentally change simply because the coordinate system was rotated.

A physically meaningful model should therefore understand that these two structures represent the same physical configuration under a rotation.

This requirement leads to the concept of **equivariance**.

---

## 24.8.240 Invariance vs Equivariance

These two concepts are closely related but should not be confused.

### Invariance

A function is invariant if its output does not change under a transformation.

For a material property:

```text
Crystal
   ↓
Property
```

rotating the crystal should produce the same scalar property.

Mathematically,

```text
f(Rx) = f(x)
```

where `R` represents a rotation.

Examples include:

```text
Formation Energy
Band Gap
Density
Bulk Modulus
```

These are scalar quantities.

---

### Equivariance

A function is equivariant if its output transforms consistently with the input.

For example, suppose a neural network predicts an atomic force vector.

If the crystal is rotated:

```text
Original force
      ↓
Rotate crystal
      ↓
Rotated force
```

The predicted force should rotate in exactly the same way.

Mathematically:

```text
f(Rx) = Rf(x)
```

Therefore:

```text
Invariant:

f(Rx) = f(x)


Equivariant:

f(Rx) = Rf(x)
```

This distinction is fundamental for crystal diffusion.

---

## 24.8.241 Why Equivariance Matters for Coordinates

Suppose an atom has Cartesian position:

```python
r = torch.tensor([
    1.0,
    0.0,
    0.0
])
```

Now rotate the structure by 90 degrees around the z-axis.

The position becomes approximately:

```text
[0, 1, 0]
```

The numerical representation changed.

However, the physical relationship between the atoms did not.

A neural network that treats:

```text
x-coordinate
y-coordinate
z-coordinate
```

as unrelated scalar features may accidentally learn coordinate-system-specific patterns.

This creates an undesirable behavior:

```text
Same physical crystal
        ↓
Different orientation
        ↓
Different prediction
```

An equivariant model instead learns:

```text
Same physical crystal
        ↓
Different orientation
        ↓
Correspondingly transformed representation
```

This is much more consistent with physics.

---

## 24.8.242 Rotation Matrix

A three-dimensional rotation can be represented by a matrix:

```text
R ∈ SO(3)
```

For example, rotation around the z-axis by angle `θ` is:

```text
R =
[
 cosθ   -sinθ   0
 sinθ    cosθ   0
   0       0    1
]
```

In PyTorch:

```python
import torch


def rotation_matrix_z(theta):

    c = torch.cos(theta)
    s = torch.sin(theta)

    return torch.tensor([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
```

For a set of coordinates:

```python
positions = torch.tensor([

    [1.0, 0.0, 0.0],

    [0.0, 1.0, 0.0],

    [0.0, 0.0, 1.0]

])
```

we can apply:

```python
R = rotation_matrix_z(
    torch.tensor(
        torch.pi / 2
    )
)

rotated_positions = (
    positions @ R.T
)
```

The coordinates change, but the internal structure remains physically equivalent under the rotation.

---

## 24.8.243 Translational Symmetry

Rotation is not the only important symmetry.

Crystals also possess translational periodicity.

Suppose an atom is located at:

```text
r
```

and the lattice translation vector is:

```text
T
```

Then:

```text
r
```

and:

```text
r + T
```

represent equivalent periodic positions.

Therefore, a crystal model should not interpret an arbitrary translation of the entire structure as a new material.

Conceptually:

```text
Crystal A

●────●────●


Translated Crystal

    ●────●────●
```

These structures represent the same periodic crystal.

---

## 24.8.244 Periodic Equivariance

For crystal generation, the model therefore needs to respect at least:

```text
Translation
Rotation
Permutation of equivalent atoms
Periodic boundary conditions
```

A more complete symmetry-aware formulation is:

```text
Crystal Representation
        ↓
Symmetry-aware Network
        ↓
Prediction / Denoising
```

This is substantially more appropriate than treating a crystal as an ordinary point cloud.

---

## 24.8.245 Why Ordinary MLPs Are Insufficient

Consider the baseline model:

```python
self.network = nn.Sequential(

    nn.Linear(
        input_dim,
        hidden_dim
    ),

    nn.SiLU(),

    nn.Linear(
        hidden_dim,
        hidden_dim
    ),

    nn.SiLU(),

    nn.Linear(
        hidden_dim,
        output_dim
    )
)
```

This network has no explicit knowledge of:

```text
Distances
Angles
Rotations
Periodic images
Symmetry
```

It can theoretically learn some of these relationships from data.

However, that is inefficient.

The network must discover physical symmetries from examples rather than having them built into its architecture.

This can increase:

```text
Data requirements
Training difficulty
Generalization error
Computational cost
```

Equivariant architectures solve part of this problem structurally.

---

## 24.8.246 Relative Atomic Geometry

The most basic geometric quantity between two atoms is the relative vector:

```text
r_ij = r_j - r_i
```

In code:

```python
relative_vector = (
    pos[edge_index[1]]
    -
    pos[edge_index[0]]
)
```

For an edge:

```text
i → j
```

this produces:

```text
r_ij
```

The corresponding distance is:

```python
distance = torch.linalg.norm(
    relative_vector,
    dim=-1,
    keepdim=True
)
```

This gives:

```text
d_ij = ||r_ij||
```

Distances are rotationally invariant.

If the entire crystal is rotated:

```text
r_ij → Rr_ij
```

then:

```text
||Rr_ij|| = ||r_ij||
```

because rotations preserve Euclidean distance.

This makes distances particularly useful for crystal graph construction.

---

## 24.8.247 Gaussian Distance Expansion

Instead of providing only one scalar distance, many graph neural networks expand distances into a richer basis.

A simple Gaussian basis is:

```text
φ_k(d)
=
exp(
-γ(d-μ_k)²
)
```

where:

```text
μ_k
```

represents the center of basis function `k`.

A PyTorch implementation is:

```python
class GaussianDistanceExpansion(
    nn.Module
):

    def __init__(
        self,
        cutoff,
        num_basis,
        gamma=10.0
    ):

        super().__init__()

        centers = torch.linspace(
            0.0,
            cutoff,
            num_basis
        )

        self.register_buffer(
            "centers",
            centers
        )

        self.gamma = gamma

    def forward(
        self,
        distances
    ):

        return torch.exp(

            -self.gamma
            *
            (
                distances
                -
                self.centers
            ) ** 2

        )
```

For example:

```python
expansion = (
    GaussianDistanceExpansion(
        cutoff=5.0,
        num_basis=64
    )
)

edge_features = expansion(
    distance
)
```

The result is:

```text
distance
   ↓
Gaussian Basis
   ↓
64-dimensional Edge Feature
```

This provides the network with a smooth representation of local geometry.

---

## 24.8.248 Periodic Neighbor Vectors

For a periodic crystal, the relative vector must include lattice translations.

Let:

```text
S_ij
```

be the integer periodic image shift.

Then the Cartesian relative vector can be expressed as:

```text
r_ij
=
r_j
-
r_i
+
L S_ij
```

where:

```text
L
```

is the lattice matrix.

In code:

```python
def periodic_relative_vector(
    pos_i,
    pos_j,
    lattice,
    image_shift
):

    return (
        pos_j
        -
        pos_i
        +
        image_shift @ lattice
    )
```

The important point is that the periodic image information must not be discarded.

---

## 24.8.249 Structure-to-Graph with Periodic Information

A practical graph representation should store:

```text
Node:
Atomic Number

Edge:
Neighbor Index
Distance
Relative Vector
Periodic Image
```

Conceptually:

```text
Atom i
  │
  ├── Atom j
  │
  ├── Distance
  │
  ├── Direction
  │
  └── Periodic Shift
```

A graph conversion function can therefore return:

```python
def structure_to_graph(
    structure,
    cutoff=5.0
):

    ...

    return {
        "atomic_numbers":
            atomic_numbers,

        "positions":
            positions,

        "edge_index":
            edge_index,

        "edge_vectors":
            edge_vectors,

        "edge_distances":
            edge_distances,

        "image_shifts":
            image_shifts,

        "lattice":
            lattice
    }
```

This richer representation is much more useful for equivariant crystal models.

---

## 24.8.250 Equivariant Message Passing

A basic graph neural network performs:

```text
Neighbor Features
       ↓
Message
       ↓
Aggregation
       ↓
Node Update
```

For example:

```python
message = message_network(
    node_i,
    node_j,
    edge_features
)
```

An equivariant network additionally tracks geometric quantities.

A conceptual message is:

```text
m_ij
=
f(
h_i,
h_j,
d_ij,
r_ij
)
```

where:

```text
h_i
```

and:

```text
h_j
```

are scalar node features,

```text
d_ij
```

is invariant distance,

and:

```text
r_ij
```

is an equivariant vector.

The network must combine them in a way that preserves the correct transformation rules.

---

## 24.8.251 Scalar and Vector Features

A simple equivariant model can maintain two types of representations.

Scalar features:

```text
h_i ∈ R^F
```

Vector features:

```text
v_i ∈ R^(F×3)
```

The scalar features remain unchanged under rotation:

```text
h_i' = h_i
```

while vector features transform as:

```text
v_i' = Rv_i
```

Therefore:

```text
Scalar Features
+
Vector Features
↓
Equivariant Representation
```

This is an important conceptual step toward modern E(3)-equivariant neural networks.

---

## 24.8.252 A Simple Vector Message Layer

A simplified implementation can demonstrate the idea.

```python
class EquivariantMessageLayer(
    nn.Module
):

    def __init__(
        self,
        hidden_dim
    ):

        super().__init__()

        self.scalar_network = nn.Sequential(

            nn.Linear(
                hidden_dim * 2 + 1,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.vector_gate = nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def forward(
        self,
        scalar_features,
        vector_features,
        edge_index,
        edge_vectors,
        edge_distances
    ):

        row, col = edge_index

        h_i = scalar_features[
            row
        ]

        h_j = scalar_features[
            col
        ]

        message_input = torch.cat(

            [
                h_i,
                h_j,
                edge_distances
            ],

            dim=-1
        )

        scalar_message = (
            self.scalar_network(
                message_input
            )
        )

        gate = torch.sigmoid(
            self.vector_gate(
                scalar_message
            )
        )

        vector_message = (

            gate.unsqueeze(-1)

            *
            edge_vectors.unsqueeze(1)
        )

        return (
            scalar_message,
            vector_message
        )
```

This is still a simplified educational implementation.

However, it illustrates the central idea:

```text
Scalar information
+
Geometric vectors
↓
Equivariant messages
```

---

## 24.8.253 Why the Vector Representation Is Equivariant

Suppose:

```text
r_ij
```

is rotated:

```text
r_ij' = Rr_ij
```

The vector message is constructed from:

```text
scalar_gate × r_ij
```

The scalar gate is invariant.

Therefore:

```text
m_vector'
=
gate × Rr_ij
```

which gives:

```text
m_vector'
=
R m_vector
```

Thus the vector message transforms consistently under rotation.

This is the basic mechanism behind equivariant geometric neural networks.

---

## 24.8.254 Centering Atomic Coordinates

Before coordinate diffusion, it is often useful to remove arbitrary global translations.

For a molecular or finite point cloud, one simple operation is:

```python
center = positions.mean(
    dim=0,
    keepdim=True
)

positions_centered = (
    positions
    -
    center
)
```

Then:

```text
mean(position) ≈ 0
```

For periodic crystals, however, naive Cartesian centering is not always appropriate because the crystal is defined by its lattice and periodic topology.

Therefore, periodic crystal generation requires more careful treatment.

This distinction is important:

```text
Molecule:
Translation can often be removed
by centering coordinates.

Crystal:
Translation is part of the
periodic representation.
```

---

## 24.8.255 Fractional vs Cartesian Coordinates

Crystal structures can be represented using either:

```text
Cartesian coordinates
```

or:

```text
Fractional coordinates
```

Fractional coordinates are defined relative to the lattice.

If:

```text
f
```

is a fractional coordinate and:

```text
L
```

is the lattice matrix, then:

```text
r = fL
```

depending on the row/column convention used by the implementation.

The advantage of fractional coordinates is that the periodic unit cell is naturally represented.

The disadvantage is that the geometry becomes coupled to the lattice.

Therefore, a complete generator may need to model:

```text
Fractional Coordinates
+
Lattice
```

jointly.

---

## 24.8.256 Joint Lattice and Coordinate Generation

A more realistic crystal diffusion model can be written as:

```text
             Noise
               │
               ▼
       ┌───────────────┐
       │ Diffusion     │
       │ Network       │
       └───────┬───────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   Fractional        Lattice
   Coordinates       Parameters
       │                │
       └───────┬────────┘
               ▼
        Crystal Structure
```

The model therefore generates both:

```text
Atomic arrangement
```

and:

```text
Unit-cell geometry
```

This is substantially closer to the actual crystal-generation problem.

---

## 24.8.257 Lattice Parameter Representation

A general crystal lattice can be represented using:

```text
a
b
c
α
β
γ
```

where:

```text
a,b,c
```

are lattice lengths and:

```text
α,β,γ
```

are lattice angles.

A simple tensor representation is:

```python
lattice_parameters = torch.tensor([

    a,
    b,
    c,
    alpha,
    beta,
    gamma

])
```

However, directly diffusing these six values has limitations.

For example:

```text
a > 0
b > 0
c > 0
```

must remain positive.

Likewise, the angles must remain physically meaningful.

A better approach may therefore transform the lattice into a representation with fewer constraints or use constrained parameterizations.

---

## 24.8.258 Positive Lattice Parameters

One simple technique is to model the logarithm of lattice lengths.

Instead of predicting:

```text
a
b
c
```

predict:

```text
log(a)
log(b)
log(c)
```

Then:

```python
lengths = torch.exp(
    log_lengths
)
```

This guarantees:

```text
a > 0
b > 0
c > 0
```

Similarly:

```python
a = torch.exp(
    log_a
)
```

prevents the diffusion model from producing negative lattice lengths.

---

## 24.8.259 Lattice Matrix Representation

An alternative is to directly represent the lattice as a `3 × 3` matrix:

```text
L =
[
a₁ a₂ a₃
b₁ b₂ b₃
c₁ c₂ c₃
]
```

Then:

```python
lattice = torch.tensor([
    [a1, a2, a3],
    [b1, b2, b3],
    [c1, c2, c3]
])
```

The lattice matrix contains both:

```text
Lengths
+
Angles
```

implicitly.

However, arbitrary diffusion of a `3 × 3` matrix may generate degenerate lattices.

For example:

```text
det(L) ≈ 0
```

would correspond to a collapsed unit cell.

Therefore, lattice validity must be explicitly monitored.

---

## 24.8.260 Cell Volume as a Validity Constraint

The unit-cell volume is:

```text
V = |det(L)|
```

In PyTorch:

```python
volume = torch.abs(
    torch.linalg.det(
        lattice
    )
)
```

A generated structure with:

```text
V ≈ 0
```

is physically meaningless.

A simple validity check is:

```python
valid_volume = (
    volume > 1e-3
)
```

A realistic implementation should use chemically and dataset-informed limits rather than an arbitrary universal threshold.

---

## 24.8.261 Atom-Atom Distance Constraints

A second critical validity check is the minimum interatomic distance.

```python
def minimum_distance(
    positions
):

    distances = torch.cdist(
        positions,
        positions
    )

    mask = (
        distances
        > 1e-8
    )

    return distances[
        mask
    ].min()
```

Then:

```python
min_dist = minimum_distance(
    positions
)
```

A generated structure can be rejected if:

```text
minimum distance
```

is unrealistically small.

However, a universal distance threshold should not be used because:

```text
Li-Li
C-C
C-H
Fe-Fe
O-O
```

have different physically reasonable distance ranges.

A better implementation uses element-dependent constraints.

---

## 24.8.262 Element-Dependent Distance Filtering

Suppose:

```python
minimum_distance_table = {

    ("Li", "Li"): 2.0,

    ("Li", "O"): 1.5,

    ("Fe", "O"): 1.6,

    ("O", "O"): 1.2

}
```

Then the generated structure can be checked pairwise.

```python
def check_pair_distance(
    element_i,
    element_j,
    distance,
    table
):

    key = tuple(sorted([
        element_i,
        element_j
    ]))

    threshold = table.get(
        key,
        1.0
    )

    return distance > threshold
```

This is only an illustrative screening system.

Real research should use chemically meaningful distance statistics or validated structural filters.

---

## 24.8.263 Equivariant Diffusion Objective

For coordinate diffusion, the model predicts noise vectors:

```text
ε ∈ R^(N×3)
```

The predicted noise should transform equivariantly.

If the crystal is rotated:

```text
x' = Rx
```

then the noise should transform as:

```text
ε' = Rε
```

Therefore:

```text
εθ(Rx,t)
=
R εθ(x,t)
```

This is the central equivariance requirement for coordinate diffusion.

It means that the model does not need to memorize separate behavior for every possible orientation.

---

## 24.8.264 Equivariance Test

A practical implementation should explicitly test equivariance.

First generate:

```python
x = torch.randn(
    10,
    3
)
```

Generate a random rotation:

```python
R = random_rotation_matrix()
```

Rotate:

```python
x_rotated = (
    x @ R.T
)
```

Run the model:

```python
pred_original = model(
    x
)

pred_rotated = model(
    x_rotated
)
```

The expected relationship is:

```python
expected = (
    pred_original @ R.T
)
```

Then compare:

```python
error = torch.mean(

    (
        pred_rotated
        -
        expected
    ) ** 2
)
```

A properly equivariant model should produce a very small error, subject to numerical precision and implementation details.

This is an important research-level diagnostic.

---

## 24.8.265 Testing Rotation Equivariance Automatically

A reusable test function can be written:

```python
def test_rotation_equivariance(
    model,
    positions,
    rotation
):

    model.eval()

    with torch.no_grad():

        original_output = model(
            positions
        )

        rotated_positions = (
            positions
            @
            rotation.T
        )

        rotated_output = model(
            rotated_positions
        )

        expected_output = (
            original_output
            @
            rotation.T
        )

        error = torch.mean(

            (
                rotated_output
                -
                expected_output
            ) ** 2
        )

    return error.item()
```

Then:

```python
error = (
    test_rotation_equivariance(
        model,
        positions,
        rotation
    )
)

print(
    "Equivariance error:",
    error
)
```

This converts an abstract theoretical requirement into a measurable engineering test.

---

## 24.8.266 Why Equivariance Reduces Data Requirements

Consider a conventional model.

If the training dataset contains:

```text
Crystal A
```

in one orientation, the model may need to learn that:

```text
Crystal A rotated by 30°
```

is physically equivalent.

An equivariant architecture builds this relationship into the model.

Therefore:

```text
One Physical Example
        ↓
Many Orientations
        ↓
Automatically Related
```

This can improve sample efficiency.

In Materials Informatics, where high-quality DFT datasets can be expensive, this property is especially valuable.

---

## 24.8.267 E(3) and SE(3) Symmetry

Modern geometric deep-learning literature often discusses:

```text
E(3)
```

and:

```text
SE(3)
```

groups.

Very broadly:

```text
SE(3)
=
Rotations
+
Translations
```

while:

```text
E(3)
```

also includes reflections.

For crystal generation, the exact symmetry group used by a model depends on the representation and physical problem.

The important idea is:

```text
The network should respect
the geometric transformations
that should not change the physical meaning
of the structure.
```

---

## 24.8.268 Reflection Symmetry

A reflection changes handedness.

For many scalar material properties, reflection may still preserve the property.

However, vector and pseudovector quantities can behave differently.

Therefore, researchers must distinguish between:

```text
SO(3)
```

rotations and the larger:

```text
O(3)
```

transformation group.

This becomes particularly important for:

```text
Magnetism
Chirality
Spin
Polarization
Optical properties
```

A model should not blindly assume that all properties are invariant under every geometric transformation.

---

## 24.8.269 Crystal Symmetry Beyond Global Rotation

Crystal symmetry is richer than simply rotating the entire structure.

A crystal may possess:

```text
Translation Symmetry
Rotation Symmetry
Mirror Symmetry
Inversion Symmetry
Screw Symmetry
Glide Symmetry
```

These are described by crystallographic space groups.

For example:

```text
P1
P2₁/c
Fm-3m
Pnma
R-3m
```

A generative model does not necessarily need to explicitly generate a space-group label.

Instead, symmetry can emerge from the generated structure.

Nevertheless, the generated structure should be checked for consistency with crystallographic constraints.

---

## 24.8.270 Space-Group Validation with pymatgen

After generating a candidate structure, `pymatgen` can be used to analyze symmetry.

```python
from pymatgen.symmetry.analyzer import (
    SpacegroupAnalyzer
)

analyzer = (
    SpacegroupAnalyzer(
        structure
    )
)

space_group = (
    analyzer.get_space_group_symbol()
)

space_group_number = (
    analyzer.get_space_group_number()
)

print(
    space_group
)

print(
    space_group_number
)
```

This gives a useful post-generation diagnostic.

For example:

```text
Space group:
Fm-3m

Number:
225
```

The exact result depends on the generated structure and symmetry tolerance.

---

## 24.8.271 Symmetry Tolerance

Generated structures contain numerical noise.

Therefore, symmetry detection depends on tolerance parameters.

For example:

```python
analyzer = SpacegroupAnalyzer(

    structure,

    symprec=1e-2,

    angle_tolerance=5
)
```

Changing these values can change the detected symmetry.

Therefore, symmetry analysis should be treated as a numerical procedure rather than an absolute truth.

A researcher should report the tolerance settings when symmetry classification is important to the scientific conclusion.

---

## 24.8.272 The Complete Equivariant Crystal Diffusion Architecture

We can now expand the previous baseline architecture.

```text
                  Target Property
                        │
                        ▼
                Property Encoder
                        │
                        ▼
Noise ───────────► Equivariant
                   Diffusion Network
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
          Species   Coordinates   Lattice
             │          │          │
             └──────────┼──────────┘
                        ▼
                 Crystal Structure
                        │
                        ▼
              Periodic Validation
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Distance   Symmetry    Chemistry
           Check      Check       Check
             │          │          │
             └──────────┼──────────┘
                        ▼
                Property Predictor
                        │
                        ▼
                       DFT
```

This is a much more realistic conceptual architecture for Materials Informatics generative modeling.

---

## 24.8.273 What Has Been Achieved

At this point, the implementation has progressed from a basic diffusion model to a physically informed framework.

We now have:

```text
✓ Forward diffusion

✓ Noise schedules

✓ Noise prediction

✓ Conditional diffusion

✓ Property embeddings

✓ Crystal graph representation

✓ Periodic neighbor information

✓ Coordinate diffusion

✓ Lattice representation

✓ Equivariant modeling

✓ Chemical validity checks

✓ Symmetry analysis

✓ Property verification
```

However, one major challenge remains.

The number of atoms in a generated crystal is not necessarily known beforehand.

For example:

```text
Li₂O
```

contains three atoms per formula unit.

Another candidate may contain:

```text
64 atoms
```

in its unit cell.

Therefore, a realistic generative model must handle **variable-size crystal structures**.

This leads to the next major problem:

```text
How can a diffusion model generate
the number of atoms and their
chemical identities before generating
their positions?
```

The solution requires moving beyond purely continuous coordinate diffusion and introducing **discrete and variable-cardinality generation**.

That problem forms the next stage of the crystal-generation implementation.

## 24.8.274 Variable-Size Crystal Generation

A major limitation of the previous diffusion architecture is that the number of atoms is assumed to be known.

For example, the model may receive:

```python
num_atoms = 32
```

and generate coordinates for exactly 32 atoms.

This is convenient computationally, but it is not a complete crystal-generation framework.

In real materials discovery, the model may need to generate:

```text
Number of atoms
+
Element identities
+
Atomic positions
+
Lattice
```

Therefore, the complete generation problem becomes

```text
Crystal
│
├── Number of atoms
│
├── Atomic species
│
├── Atomic coordinates
│
└── Lattice
```

The model must learn all four components.

---

## 24.8.275 Why the Number of Atoms Matters

Consider two candidate structures:

```text
Candidate A

Li
O
O
```

and

```text
Candidate B

Li
Fe
P
O
O
O
O
```

The two structures have different:

```text
Composition
Stoichiometry
Number of atoms
Graph size
```

A diffusion model designed for a fixed number of nodes cannot directly move between these configurations.

Therefore, variable-size generation requires an additional mechanism.

There are several possible strategies:

```text
Strategy 1
Generate atom count first

Strategy 2
Generate composition first

Strategy 3
Generate a maximum-size structure
and mask unused atoms

Strategy 4
Use an autoregressive generator

Strategy 5
Use a joint discrete-continuous model
```

Each strategy has advantages and limitations.

---

## 24.8.276 Hierarchical Crystal Generation

A useful design is to generate the crystal hierarchically.

Instead of asking one neural network to generate everything simultaneously, the process can be decomposed.

```text
Material Objective
       │
       ▼
Composition
       │
       ▼
Number of Atoms
       │
       ▼
Lattice
       │
       ▼
Atomic Positions
       │
       ▼
Crystal Structure
```

This resembles the way a researcher may formulate a materials-design problem.

For example:

```text
Target:

Band Gap = 1.5–2.0 eV

       ↓

Chemical System:

Li-Fe-P-O

       ↓

Candidate Composition

       ↓

Crystal Structure

       ↓

Property Verification
```

This hierarchical approach reduces the dimensionality of each individual generation task.

---

## 24.8.277 Composition as a Discrete Variable

Atomic species are inherently discrete.

For example:

```text
Li
Fe
P
O
```

cannot be represented naturally as a continuous coordinate.

A simple numerical representation is an atomic-number encoding:

```python
atomic_numbers = torch.tensor([
    3,
    26,
    15,
    8
])
```

where:

```text
Li → 3
Fe → 26
P  → 15
O  → 8
```

However, directly treating atomic numbers as continuous values can be problematic.

For example, the network may produce:

```text
17.43
```

which does not correspond to a valid element.

Therefore, discrete chemical identities require a discrete representation.

---

## 24.8.278 Element Embeddings

One common approach is to represent each element using a learned embedding.

```python
class ElementEmbedding(
    nn.Module
):

    def __init__(
        self,
        num_elements,
        embedding_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_elements,
            embedding_dim
        )

    def forward(self, atomic_numbers):

        return self.embedding(
            atomic_numbers
        )
```

For example:

```python
embedding = ElementEmbedding(
    num_elements=119,
    embedding_dim=128
)
```

Then:

```python
z = embedding(
    atomic_numbers
)
```

produces:

```text
Atomic Number
      ↓
Embedding Vector
```

Instead of forcing the model to interpret:

```text
Fe = 26
```

as simply a number, the network learns a high-dimensional representation.

---

## 24.8.279 Chemical Embedding Space

After training, the embedding space may organize chemically related elements in useful ways.

Conceptually:

```text
Embedding Space

       O
      / \
     S   Se

 Li       Na
  \       /
   K     Rb
```

The network may learn similarities related to:

```text
Atomic size
Electronegativity
Valence
Bonding behavior
Periodic position
Chemical environment
```

The model is not explicitly required to learn these relationships independently.

They can emerge from the training objective.

---

## 24.8.280 Predicting Element Probabilities

Instead of directly predicting an element identity, the network can predict logits.

```python
logits = model(
    hidden_state
)
```

Suppose:

```text
logits =
[
  1.2,
  0.4,
  3.1,
  0.8,
  ...
]
```

Convert them into probabilities:

```python
probabilities = torch.softmax(
    logits,
    dim=-1
)
```

Then:

```python
element = torch.multinomial(
    probabilities,
    num_samples=1
)
```

The generation process becomes:

```text
Hidden Representation
        ↓
Element Logits
        ↓
Softmax
        ↓
Element Probability
        ↓
Sample Element
```

This is a discrete generation process.

---

## 24.8.281 Chemical Composition Generation

A composition can be represented as:

```text
Li₂FePO₄
```

which can be converted to a sequence:

```text
Li Li Fe P O O O O
```

or equivalently:

```text
Element:
Li Fe P O

Count:
2  1  1  4
```

A composition generator can therefore predict both:

```text
Element
```

and:

```text
Stoichiometric Count
```

---

## 24.8.282 Composition Generator

A simple neural network can predict element probabilities.

```python
class CompositionGenerator(
    nn.Module
):

    def __init__(
        self,
        condition_dim,
        hidden_dim,
        num_elements
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                condition_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                num_elements
            )
        )

    def forward(
        self,
        condition
    ):

        return self.network(
            condition
        )
```

The input condition may contain:

```text
Target property
Chemical constraints
Crystal system
Experimental restrictions
```

For example:

```python
condition = torch.tensor([
    2.0,     # target band gap
    -2.5,    # target formation energy
    150.0    # target bulk modulus
])
```

Then:

```python
logits = generator(
    condition
)
```

---

## 24.8.283 Masking Chemically Forbidden Elements

Suppose the researcher only wants:

```text
Li-Fe-P-O
```

The model should not generate:

```text
Au
Pt
Hg
U
```

if these elements are outside the allowed chemical system.

A simple mask can be constructed.

```python
allowed_elements = [
    3,      # Li
    26,     # Fe
    15,     # P
    8       # O
]
```

Then:

```python
mask = torch.full(
    (119,),
    float("-inf")
)

mask[
    allowed_elements
] = 0.0
```

Apply the mask:

```python
masked_logits = (
    logits + mask
)
```

Then:

```python
probabilities = torch.softmax(
    masked_logits,
    dim=-1
)
```

The forbidden elements effectively receive zero probability.

This is an example of **hard chemical conditioning**.

---

## 24.8.284 Hard Constraints vs Soft Constraints

Not every constraint should necessarily be implemented as a hard mask.

For example:

```text
Allowed elements
```

can be a hard constraint.

But:

```text
Preferred oxidation state
```

may be better represented as a soft preference.

Conceptually:

```text
Hard constraint

Forbidden
   ↓
Probability = 0
```

whereas:

```text
Soft constraint

Unfavorable
   ↓
Probability reduced
```

This distinction is important in generative materials design.

---

## 24.8.285 Maximum-Atom Masking

Another practical approach is to define:

```python
max_atoms = 64
```

and represent every structure using 64 possible atom slots.

A structure containing only 20 atoms becomes:

```text
Atom 1
Atom 2
...
Atom 20
Atom 21 → MASK
...
Atom 64 → MASK
```

The model therefore always operates on:

```text
64 positions
```

but only some positions correspond to real atoms.

A mask tensor can be used:

```python
atom_mask = torch.zeros(
    64,
    dtype=torch.bool
)

atom_mask[:20] = True
```

Then:

```python
valid_positions = positions[
    atom_mask
]
```

This is computationally convenient.

---

## 24.8.286 Advantages of Masked Generation

The maximum-size strategy simplifies batching.

For example:

```text
Structure A → 16 atoms
Structure B → 32 atoms
Structure C → 48 atoms
```

can all be represented using:

```text
max_atoms = 64
```

and batched into:

```text
(batch_size, 64, features)
```

This is convenient for GPU computation.

However, the disadvantage is memory waste.

If most structures contain only 10 atoms:

```text
10 valid atoms
+
54 masked atoms
```

are processed.

For large crystal datasets, this can become expensive.

---

## 24.8.287 Variable-Length Autoregressive Generation

A different strategy is to generate atoms sequentially.

The model begins with:

```text
Start
```

and predicts:

```text
Atom 1
```

then:

```text
Atom 2
```

then:

```text
Atom 3
```

until it predicts:

```text
STOP
```

The generation process is:

```text
START
  ↓
Atom
  ↓
Atom
  ↓
Atom
  ↓
...
  ↓
STOP
```

This naturally supports variable-size structures.

---

## 24.8.288 Autoregressive Crystal Generation

A simplified model could be written:

```python
class CrystalAutoregressiveModel(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_elements
    ):

        super().__init__()

        self.encoder = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.gru = nn.GRUCell(
            hidden_dim,
            hidden_dim
        )

        self.element_head = nn.Linear(
            hidden_dim,
            num_elements
        )

        self.stop_head = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        x,
        hidden
    ):

        h = self.encoder(x)

        hidden = self.gru(
            h,
            hidden
        )

        element_logits = (
            self.element_head(
                hidden
            )
        )

        stop_logit = (
            self.stop_head(
                hidden
            )
        )

        return (
            element_logits,
            stop_logit,
            hidden
        )
```

This is only a conceptual implementation.

A research-quality crystal autoregressive model must additionally handle:

```text
Periodic geometry
Atom ordering
Symmetry
Lattice
Chemical validity
```

---

## 24.8.289 Why Atom Ordering Is a Problem

Consider:

```text
Li Fe O O O
```

and:

```text
O Li O Fe O
```

These can represent the same composition and potentially the same structure if positions are associated appropriately.

An autoregressive model, however, sees them as different sequences.

This creates a **permutation problem**.

The network may waste capacity learning arbitrary ordering differences.

Therefore, atom-order invariance is important.

---

## 24.8.290 Permutation Invariance

Suppose the atom feature matrix is:

```text
X =
[
x₁
x₂
x₃
]
```

A permutation changes the order:

```text
X' =
[
x₃
x₁
x₂
]
```

A property predictor should satisfy:

```text
f(X') = f(X)
```

for permutation-invariant properties.

This requirement is naturally handled by graph neural networks using aggregation operations such as:

```python
sum()
mean()
max()
```

For example:

```python
graph_embedding = node_features.sum(
    dim=0
)
```

Because addition is commutative:

```text
x₁ + x₂ + x₃
=
x₃ + x₁ + x₂
```

the resulting graph representation is independent of node ordering.

---

## 24.8.291 Set-Based Crystal Representation

A crystal can therefore be considered as a set:

```text
Crystal =
{
Atom₁,
Atom₂,
...,
Atom_N
}
```

rather than a sequence.

A set-based model should satisfy:

```text
f(
{
A,B,C
}
)
=
f(
{
C,A,B
}
)
```

This property is particularly useful for crystal generation and prediction.

---

## 24.8.292 Deep Sets

A simple permutation-invariant architecture follows the Deep Sets formulation.

Conceptually:

```text
Crystal
  ↓
Atom Encoder
  ↓
Element-wise Features
  ↓
Permutation-Invariant Aggregation
  ↓
Crystal Representation
```

Mathematically:

```text
z =
ρ(
Σ φ(x_i)
)
```

where:

```text
φ
```

encodes individual atoms and:

```text
ρ
```

maps the aggregated representation to the final output.

A simple implementation is:

```python
class DeepSetEncoder(
    nn.Module
):

    def __init__(
        self,
        atom_dim,
        hidden_dim
    ):

        super().__init__()

        self.phi = nn.Sequential(

            nn.Linear(
                atom_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.rho = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

    def forward(
        self,
        atoms
    ):

        atom_features = self.phi(
            atoms
        )

        pooled = atom_features.sum(
            dim=1
        )

        return self.rho(
            pooled
        )
```

This provides a simple permutation-invariant crystal representation.

---

## 24.8.293 Why Graphs Are Better for Crystal Geometry

Although Deep Sets handle permutation invariance, they do not explicitly represent local bonding structure.

Consider:

```text
Crystal
   ↓
Atoms
   ↓
All atoms treated equally
```

This loses explicit information about:

```text
Nearest neighbors
Bond lengths
Coordination
Local geometry
Periodic connectivity
```

Graph representations solve this by introducing edges.

```text
Atoms → Nodes

Neighbor relationships → Edges
```

Therefore:

```text
Set Representation
```

can be extended to:

```text
Graph Representation
```

which is substantially more informative for crystal generation.

---

## 24.8.294 Joint Discrete-Continuous Diffusion

We can now formulate the full generation problem.

Some variables are continuous:

```text
Coordinates
Lattice parameters
Distances
Angles
```

Other variables are discrete:

```text
Element identity
Number of atoms
Crystal class
Composition
```

Therefore, the complete state is:

```text
x =
(
x_continuous,
x_discrete
)
```

A realistic generator must learn both.

This leads to:

```text
Discrete + Continuous Generative Modeling
```

---

## 24.8.295 Continuous Diffusion Variables

The continuous variables may include:

```python
positions
lattice
```

For example:

```python
continuous_state = torch.cat(
    [
        positions.reshape(
            batch_size,
            -1
        ),

        lattice.reshape(
            batch_size,
            -1
        )
    ],
    dim=-1
)
```

Gaussian diffusion can then be applied.

```python
noise = torch.randn_like(
    continuous_state
)

x_t = (
    sqrt_alpha
    * continuous_state
    +
    sqrt_one_minus_alpha
    * noise
)
```

---

## 24.8.296 Discrete Diffusion Variables

Element identities cannot simply receive Gaussian noise.

Instead, a categorical diffusion process can progressively corrupt element identities.

For example:

```text
Original:

O

↓

Corrupted:

O / N / F / C / ...

↓

More Noise:

Random Element

↓

Final:

Nearly Uniform Distribution
```

The model then learns to reconstruct the original categorical distribution.

This is fundamentally different from ordinary Gaussian coordinate diffusion.

---

## 24.8.297 Categorical Noise Transition

Suppose there are:

```text
K = 100
```

possible atomic categories.

A categorical transition matrix can be represented as:

```text
Q_t ∈ R^(K×K)
```

where:

```text
Q_t(i,j)
```

represents the probability of transitioning from category `i` to category `j`.

The forward process becomes:

```text
x₀
 ↓
x₁
 ↓
x₂
 ↓
...
 ↓
x_T
```

where each state is categorical.

The reverse model learns:

```text
pθ(x_{t-1} | x_t)
```

rather than:

```text
pθ(x_{t-1} | x_t)
```

under a Gaussian assumption.

The mathematical structure is therefore different from continuous DDPMs.

---

## 24.8.298 Hybrid Crystal State

A complete crystal state may therefore be represented as:

```text
Crystal State
│
├── Element identities
│      ↓
│   Discrete diffusion
│
├── Coordinates
│      ↓
│   Continuous diffusion
│
└── Lattice
       ↓
   Continuous diffusion
```

The reverse process reconstructs all three simultaneously.

Conceptually:

```text
Noisy Elements
       │
Noisy Coordinates
       │
Noisy Lattice
       │
       ▼
Joint Diffusion Network
       │
       ▼
Cleaner Crystal
       │
       ▼
Valid Crystal
```

This is much closer to the full crystal-generation problem.

---

## 24.8.299 Joint Denoising Network

A simplified architecture is:

```python
class JointCrystalDiffusion(
    nn.Module
):

    def __init__(
        self,
        num_elements,
        element_dim,
        hidden_dim,
        lattice_dim=9
    ):

        super().__init__()

        self.element_embedding = (
            nn.Embedding(
                num_elements,
                element_dim
            )
        )

        self.position_encoder = (
            nn.Linear(
                3,
                hidden_dim
            )
        )

        self.lattice_encoder = (
            nn.Linear(
                lattice_dim,
                hidden_dim
            )
        )

        self.time_encoder = (
            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.network = nn.Sequential(

            nn.Linear(
                element_dim
                + hidden_dim
                + hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.position_head = nn.Linear(
            hidden_dim,
            3
        )

        self.element_head = nn.Linear(
            hidden_dim,
            num_elements
        )

        self.lattice_head = nn.Linear(
            hidden_dim,
            lattice_dim
        )

    def forward(
        self,
        atomic_numbers,
        positions,
        lattice,
        time_embedding
    ):

        element_features = (
            self.element_embedding(
                atomic_numbers
            )
        )

        position_features = (
            self.position_encoder(
                positions
            )
        )

        lattice_features = (
            self.lattice_encoder(
                lattice
            )
        )

        h = torch.cat(
            [
                element_features,
                position_features,
                lattice_features
            ],
            dim=-1
        )

        h = self.network(h)

        predicted_positions = (
            self.position_head(h)
        )

        predicted_elements = (
            self.element_head(h)
        )

        predicted_lattice = (
            self.lattice_head(
                h.mean(dim=1)
            )
        )

        return (
            predicted_elements,
            predicted_positions,
            predicted_lattice
        )
```

This implementation is intentionally simplified.

It demonstrates the architecture rather than claiming to be a production crystal diffusion system.

---

## 24.8.300 Limitations of the Simple Joint Model

The preceding implementation contains several serious limitations.

First:

```text
The lattice is broadcast
to every atom.
```

Second:

```text
The coordinate encoder is not
rotation equivariant.
```

Third:

```text
Element identities are treated
through ordinary embeddings rather
than a discrete diffusion process.
```

Fourth:

```text
Periodic neighbors are not explicitly
represented.
```

Fifth:

```text
The number of atoms is fixed.
```

Sixth:

```text
Chemical validity is not guaranteed.
```

Therefore, the model should be considered a conceptual bridge toward a proper architecture.

A research implementation must address each limitation explicitly.

---

## 24.8.301 Variable Atom Count Through a Count Predictor

One practical solution is to predict the number of atoms before generation.

Suppose:

```python
num_atoms_model = nn.Sequential(

    nn.Linear(
        condition_dim,
        hidden_dim
    ),

    nn.SiLU(),

    nn.Linear(
        hidden_dim,
        max_atoms
    )
)
```

The output represents:

```text
P(N = 1)
P(N = 2)
P(N = 3)
...
P(N = max_atoms)
```

Then:

```python
count_logits = (
    num_atoms_model(
        condition
    )
)

count_probability = torch.softmax(
    count_logits,
    dim=-1
)

num_atoms = torch.multinomial(
    count_probability,
    1
)
```

The overall generation becomes:

```text
Target Property
      ↓
Count Predictor
      ↓
N atoms
      ↓
Composition Generator
      ↓
Elements
      ↓
Diffusion
      ↓
Coordinates + Lattice
```

---

## 24.8.302 Composition Before Geometry

A chemically sensible generation order is often:

```text
Target
  ↓
Composition
  ↓
Structure
```

rather than:

```text
Target
  ↓
Random Coordinates
  ↓
Try to determine chemistry
```

For example:

```text
Target:

Band Gap = 2.0 eV

        ↓

Candidate composition:

Li₂FePO₄

        ↓

Generate possible structures

        ↓

Evaluate properties
```

This separates:

```text
Chemical search
```

from:

```text
Structural search
```

which can make the overall problem easier.

---

## 24.8.303 Composition-Conditioned Crystal Diffusion

The diffusion model can receive a composition embedding.

```python
composition_embedding = (
    composition_encoder(
        composition
    )
)
```

The complete condition becomes:

```python
condition = torch.cat(
    [
        property_embedding,
        composition_embedding
    ],
    dim=-1
)
```

Then:

```python
crystal = diffusion_model(
    noisy_crystal,
    condition
)
```

The model is therefore asked to generate:

```text
Structure | Composition, Property
```

rather than:

```text
Structure | Property
```

alone.

---

## 24.8.304 Example: Li-Fe-P-O Design

Suppose the researcher specifies:

```python
target_band_gap = 2.0

allowed_elements = [
    "Li",
    "Fe",
    "P",
    "O"
]
```

The generation pipeline becomes:

```text
Target Band Gap
      │
      ▼
Property Encoder
      │
      ├───────────────┐
      │               │
      ▼               ▼
Composition       Diffusion
Constraint        Network
      │               │
      └───────┬───────┘
              ▼
        Crystal Candidate
```

A practical implementation might first generate candidate compositions and then run a structural diffusion model for each composition.

---

## 24.8.305 Composition Sampling

A simple composition sampler could generate integer counts.

```python
def sample_composition(
    element_probabilities,
    total_atoms
):

    elements = torch.multinomial(
        element_probabilities,
        total_atoms,
        replacement=True
    )

    return elements
```

For example:

```python
elements = sample_composition(
    probabilities,
    total_atoms=16
)
```

might produce:

```text
Li Li Li Fe Fe P O O O O ...
```

The result must then be converted into a chemically meaningful stoichiometry.

Random independent element sampling is generally insufficient because it can generate unrealistic compositions.

---

## 24.8.306 Charge-Balance Constraints

Suppose the composition contains:

```text
Li⁺
Fe²⁺
P⁵⁺
O²⁻
```

The total oxidation-state charge should approximately satisfy:

```text
Σ n_i q_i = 0
```

For a composition:

```text
Li_a Fe_b P_c O_d
```

the charge-balance condition could be:

```text
a(+1)
+
b(q_Fe)
+
c(+5)
+
d(-2)
≈ 0
```

This provides a useful chemical constraint.

A generative model can therefore incorporate charge balance either:

```text
During generation
```

or:

```text
During post-generation filtering
```

---

## 24.8.307 Chemical Validity Pipeline

A generated structure can pass through several filters.

```text
Generated Crystal
       ↓
Composition Validity
       ↓
Charge Balance
       ↓
Minimum Distance
       ↓
Density
       ↓
Coordination
       ↓
Symmetry
       ↓
Duplicate Removal
       ↓
Property Prediction
       ↓
DFT
```

This demonstrates an important principle:

> A generative model should not be treated as a final scientific decision-maker.

Instead, generation should be embedded inside a larger validation and screening pipeline.

---

## 24.8.308 Generation Is Not Validation

A diffusion model may produce a mathematically valid output while producing a physically unreasonable material.

For example:

```text
Neural Network Output
        ↓
Valid Tensor
        ↓
Invalid Crystal
```

Therefore:

```text
Generation
≠
Validation
```

and:

```text
Prediction
≠
Physical Truth
```

A candidate becomes scientifically interesting only after passing increasingly rigorous validation stages.

---

## 24.8.309 Multi-Stage Validation

A realistic workflow should use increasingly expensive tests.

### Stage 1 — Mathematical validity

Check:

```text
NaN
Inf
Tensor shape
```

### Stage 2 — Geometric validity

Check:

```text
Minimum distances
Cell volume
Lattice validity
```

### Stage 3 — Chemical validity

Check:

```text
Composition
Charge balance
Oxidation states
Coordination
```

### Stage 4 — Machine-learning screening

Predict:

```text
Formation energy
Band gap
Density
Elastic properties
```

### Stage 5 — Physics-based validation

Run:

```text
DFT relaxation
Electronic structure
Phonons
```

### Stage 6 — Experimental validation

Finally:

```text
Synthesis
Characterization
Property measurement
```

Thus:

```text
AI Generation
      ↓
Chemical Screening
      ↓
ML Screening
      ↓
DFT
      ↓
Experiment
```

is much more realistic than:

```text
AI
 ↓
Material
```

---

## 24.8.310 Generated Crystal Screening Code

A simple screening framework can be written as:

```python
def screen_crystal(
    structure,
    property_predictor
):

    # --------------------------------
    # 1. Basic validity
    # --------------------------------

    if structure is None:
        return False

    # --------------------------------
    # 2. Cell validity
    # --------------------------------

    volume = (
        structure.volume
    )

    if volume <= 0:
        return False

    # --------------------------------
    # 3. Density validity
    # --------------------------------

    density = (
        structure.density
    )

    if density <= 0:
        return False

    # --------------------------------
    # 4. Property prediction
    # --------------------------------

    predicted = (
        property_predictor(
            structure
        )
    )

    band_gap = predicted[
        "band_gap"
    ]

    formation_energy = predicted[
        "formation_energy"
    ]

    # --------------------------------
    # 5. Target filtering
    # --------------------------------

    if not (
        1.5
        <= band_gap
        <= 2.0
    ):
        return False

    if formation_energy > -2.0:
        return False

    return True
```

The function returns:

```text
True
```

only when the candidate satisfies the screening criteria.

In practice, each screening criterion should be implemented carefully and validated against known materials.

---

## 24.8.311 Generation at Scale

Once the model is trained, large-scale generation becomes possible.

```python
candidates = []

for i in range(
    100_000
):

    condition = sample_condition()

    crystal = generate_crystal(
        model,
        condition
    )

    candidates.append(
        crystal
    )
```

However, generating 100,000 structures is not automatically useful.

The generated dataset may contain:

```text
Invalid structures
Duplicates
Unstable structures
Poor-property materials
Out-of-distribution structures
```

Therefore, the computational pipeline should be designed as:

```text
Generate Many
      ↓
Cheap Filters
      ↓
Generate Fewer
      ↓
ML Prediction
      ↓
Generate/Select Fewer
      ↓
DFT
      ↓
Experimental Candidates
```

This is a fundamental principle of computational materials discovery.

---

## 24.8.312 Cost-Aware Candidate Selection

Suppose:

```text
100,000
```

structures are generated.

Running DFT on all of them may be computationally prohibitive.

Instead:

```text
100,000
   ↓
Geometric filtering
   ↓
50,000
   ↓
Chemical filtering
   ↓
20,000
   ↓
ML property prediction
   ↓
2,000
   ↓
DFT
   ↓
100
   ↓
High-accuracy DFT
   ↓
20
   ↓
Experiment
```

This creates a **funnel-shaped discovery pipeline**.

The key idea is:

```text
Cheap models
screen large spaces.

Expensive models
evaluate small spaces.
```

---

## 24.8.313 Out-of-Distribution Detection

Generated structures may lie far outside the training distribution.

For example:

```text
Training Materials
      ↓
Common Chemistry

Generated Material
      ↓
Unusual Chemistry
```

The property predictor may make unreliable predictions in such regions.

Therefore, uncertainty estimation is important.

A simple approach is an ensemble.

```python
predictions = []

for model in ensemble:

    prediction = model(
        crystal
    )

    predictions.append(
        prediction
    )

predictions = torch.stack(
    predictions
)

mean_prediction = (
    predictions.mean(
        dim=0
    )
)

uncertainty = (
    predictions.std(
        dim=0
    )
)
```

A candidate with:

```text
Excellent predicted property
```

but:

```text
Very high uncertainty
```

should not automatically be treated as a top candidate.

---

## 24.8.314 Active Learning Loop

The generation system can be connected to active learning.

```text
Generate
   ↓
Predict
   ↓
Estimate Uncertainty
   ↓
Select Informative Candidates
   ↓
DFT
   ↓
Add to Dataset
   ↓
Retrain
   ↓
Generate Again
```

This creates an iterative research workflow.

```python
for iteration in range(
    num_iterations
):

    candidates = generate(
        generator
    )

    predictions = predict(
        candidates
    )

    selected = select_informative(
        candidates,
        predictions
    )

    dft_results = run_dft(
        selected
    )

    dataset.add(
        dft_results
    )

    retrain(
        dataset
    )
```

This is one of the most powerful ways to connect generative AI with Materials Informatics.

---

## 24.8.315 Closed-Loop Materials Discovery

The ultimate architecture is therefore not simply a generative model.

It is a closed scientific loop:

```text
                 ┌───────────────┐
                 │ Materials     │
                 │ Dataset       │
                 └───────┬───────┘
                         │
                         ▼
                 Train ML Models
                         │
                         ▼
                 Generative Model
                         │
                         ▼
                 Candidate Crystals
                         │
                         ▼
                 Fast Screening
                         │
                         ▼
                 Property Models
                         │
                         ▼
                       DFT
                         │
                         ▼
                  Experimental
                    Validation
                         │
                         ▼
                  New Materials
                         │
                         ▼
                  New Training Data
                         │
                         └───────────────►
```

This changes generative AI from a standalone machine-learning model into a component of an autonomous materials-discovery workflow.

---

## 24.8.316 From Crystal Generation to Autonomous Discovery

The long-term goal is to automate the complete loop:

```text
Scientific Objective
        ↓
Generate Candidates
        ↓
Evaluate Candidates
        ↓
Select Best Candidates
        ↓
Run DFT
        ↓
Learn From Results
        ↓
Generate Improved Candidates
```

The system can progressively explore increasingly promising regions of materials space.

This is the conceptual foundation of autonomous materials discovery.

---

## 24.8.317 Practical Research Architecture

A realistic research codebase may eventually be organized as:

```text
materials_generation/
│
├── data/
│   ├── structures.py
│   ├── compositions.py
│   └── preprocessing.py
│
├── representations/
│   ├── crystal_graph.py
│   ├── periodic_neighbors.py
│   └── embeddings.py
│
├── diffusion/
│   ├── schedule.py
│   ├── forward.py
│   ├── reverse.py
│   ├── conditional.py
│   └── equivariant.py
│
├── models/
│   ├── property_predictor.py
│   ├── composition_model.py
│   └── crystal_model.py
│
├── constraints/
│   ├── chemistry.py
│   ├── geometry.py
│   └── symmetry.py
│
├── generation/
│   ├── sampler.py
│   ├── filtering.py
│   └── deduplication.py
│
├── evaluation/
│   ├── validity.py
│   ├── diversity.py
│   ├── novelty.py
│   └── uncertainty.py
│
└── experiments/
    ├── train.py
    ├── generate.py
    └── evaluate.py
```

This type of organization becomes increasingly important as the project moves from educational experiments toward publishable research.

---

## 24.8.318 Reproducible Generation

Generative experiments should be reproducible.

Set random seeds:

```python
import random
import numpy as np
import torch


SEED = 42

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )
```

However, complete reproducibility may still depend on:

```text
CUDA version
PyTorch version
GPU architecture
Numerical kernels
Data preprocessing
Model checkpoint
Sampling configuration
```

Therefore, a research experiment should record all relevant configuration information.

---

## 24.8.319 Configuration-Based Experiments

Instead of hard-coding parameters throughout the project, use a configuration dictionary.

```python
config = {

    "seed": 42,

    "batch_size": 64,

    "hidden_dim": 256,

    "num_layers": 8,

    "learning_rate": 1e-4,

    "diffusion_steps": 1000,

    "cutoff": 5.0,

    "max_atoms": 64,

    "guidance_strength": 2.0,

    "target_property": "band_gap"

}
```

Then:

```python
model = build_model(
    config
)
```

and:

```python
train(
    model,
    config
)
```

This makes experiments easier to reproduce and compare.

---

## 24.8.320 Saving Generated Structures

Generated structures should be saved in standard materials-science formats.

For example, `pymatgen` can write structures to CIF.

```python
structure.to(
    filename="candidate_001.cif"
)
```

A complete generation loop may therefore be:

```python
for i, structure in enumerate(
    generated_structures
):

    filename = (
        f"candidate_{i:06d}.cif"
    )

    structure.to(
        filename=filename
    )
```

This creates:

```text
candidate_000000.cif
candidate_000001.cif
candidate_000002.cif
...
```

These files can subsequently be inspected, visualized, relaxed, or submitted to DFT workflows.

---

## 24.8.321 Storing Metadata

The structure file alone is not enough.

Each generated candidate should also store metadata such as:

```text
Generation seed
Target properties
Guidance strength
Model checkpoint
Predicted properties
Uncertainty
Validity status
DFT status
```

A simple record can be represented as:

```python
candidate_record = {

    "id":
        "candidate_000001",

    "target_band_gap":
        2.0,

    "predicted_band_gap":
        1.87,

    "predicted_formation_energy":
        -2.91,

    "uncertainty":
        0.12,

    "guidance_strength":
        2.0,

    "valid":
        True

}
```

These records can be stored in:

```text
CSV
JSON
Parquet
Database
```

depending on project scale.

---

## 24.8.322 Generation Dataset

Eventually, the output becomes a new materials dataset.

For example:

```text
ID | Composition | Band Gap | Eform | Density | Uncertainty
-------------------------------------------------------------
001| Li2FePO4    | 1.82     | -2.91 | 4.8     | 0.12
002| LiFePO4     | 1.65     | -3.10 | 4.6     | 0.08
003| Li3FeP2O8   | 2.04     | -2.20 | 5.1     | 0.31
```

This table becomes the basis for:

```text
Ranking
Visualization
DFT selection
Active learning
Experimental selection
```

---

## 24.8.323 The Researcher’s Decision Layer

The final decision should not be made by a single predicted number.

Instead, candidates should be evaluated using multiple criteria.

For example:

```text
Candidate
│
├── Property performance
├── Stability
├── Uncertainty
├── Novelty
├── Diversity
├── Chemical accessibility
├── Synthetic feasibility
└── Computational cost
```

A candidate with an excellent band gap but impossible chemistry should be rejected.

Likewise, a candidate with excellent predicted stability but extremely high model uncertainty should receive lower priority.

This introduces the concept of **multi-criteria candidate selection**.

---

## 24.8.324 A Complete Candidate Ranking Function

A simplified ranking score could be:

```python
score = (

    w_property
    * property_score

    +

    w_stability
    * stability_score

    +

    w_novelty
    * novelty_score

    +

    w_diversity
    * diversity_score

    -

    w_uncertainty
    * uncertainty

)
```

Then:

```python
ranking = torch.argsort(
    score,
    descending=True
)
```

The highest-ranked candidates can be selected for expensive evaluation.

Again, the weights should be scientifically justified rather than chosen arbitrarily.

---

## 24.8.325 Diversity-Aware Selection

Simply selecting the top 100 predicted structures can produce highly similar materials.

For example:

```text
Top 100 candidates

      ↓

95 are essentially the same
chemical family
```

This reduces the scientific value of the generated library.

A better strategy is:

```text
High property score
+
High structural diversity
```

For example:

```text
Candidate A
Candidate B
Candidate C
Candidate D
```

should ideally represent distinct regions of materials space.

This creates a balance between:

```text
Exploitation
```

and:

```text
Exploration
```

---

## 24.8.326 Exploitation vs Exploration

The distinction is fundamental.

### Exploitation

Search near known high-performing materials.

```text
Known good region
       ↓
More candidates nearby
```

### Exploration

Search unfamiliar regions.

```text
Known region
     ↓
Unknown chemical space
```

A generative model with excessively strong conditioning may focus almost entirely on exploitation.

A model with weak conditioning may explore broadly but fail to satisfy the target.

Therefore:

```text
Successful Materials Discovery
=
Exploration
+
Exploitation
```

---

## 24.8.327 Why This Matters for Generative Materials AI

At this stage, the central idea of the chapter becomes clear.

Generative AI does not simply answer:

```text
"What materials exist?"
```

It attempts to answer:

```text
"What materials could exist
that satisfy a scientific objective?"
```

The workflow therefore becomes:

```text
Scientific Objective
        ↓
Generative Model
        ↓
Candidate Structures
        ↓
Validity Filtering
        ↓
Property Prediction
        ↓
Uncertainty Estimation
        ↓
DFT Validation
        ↓
Experimental Validation
```

This is the transition from **materials prediction** to **materials design**.

---

## 24.8.328 The Complete Crystal-Generation Problem

The crystal-generation problem can now be expressed as a joint distribution:

```text
p(
    N,
    Z,
    X,
    L
    |
    y
)
```

where:

```text
N
=
number of atoms

Z
=
atomic identities

X
=
atomic coordinates

L
=
lattice

y
=
desired material properties
```

The goal is therefore to learn:

```text
p(
    N,Z,X,L
    |
    y
)
```

rather than simply:

```text
p(X)
```

This is a much more complete formulation of property-conditioned crystal generation.

---

## 24.8.329 Research-Level Architecture

The final conceptual model is:

```text
                         Target Properties
                               │
                               ▼
                       Property Encoder
                               │
                               ▼
                    ┌────────────────────┐
                    │ Conditional        │
                    │ Crystal Generator  │
                    └─────────┬──────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ▼              ▼              ▼
          Atom Count       Elements       Lattice
               │              │              │
               └──────────────┼──────────────┘
                              │
                              ▼
                    Equivariant Diffusion
                              │
                              ▼
                     Atomic Coordinates
                              │
                              ▼
                       Crystal Structure
                              │
                              ▼
                  ┌──────────────────────┐
                  │ Chemical Validation  │
                  └──────────┬───────────┘
                             │
                             ▼
                    Property Predictors
                             │
                             ▼
                      Uncertainty Model
                             │
                             ▼
                            DFT
                             │
                             ▼
                       Experiment
```

This architecture captures the major components required for a modern Materials Informatics generative system.

---

## 24.8.330 From Educational Diffusion to Research Diffusion

The progression developed throughout this chapter can now be summarized as:

```text
Basic Neural Network
        ↓
Autoencoder
        ↓
Variational Autoencoder
        ↓
Graph VAE
        ↓
Diffusion Model
        ↓
Crystal Diffusion
        ↓
Property-Conditioned Diffusion
        ↓
Equivariant Diffusion
        ↓
Discrete + Continuous Generation
        ↓
Chemical Constraints
        ↓
Property Screening
        ↓
DFT Validation
        ↓
Active Learning
        ↓
Closed-Loop Materials Discovery
```

Each step removes another limitation of the previous model.

The important lesson is that a research-quality generative materials system is not obtained simply by applying a fashionable generative architecture.

It requires the integration of:

```text
Machine Learning
+
Materials Science
+
Crystallography
+
Geometry
+
Chemistry
+
Physics
+
Scientific Computing
```

The strength of Materials Informatics lies precisely in this integration.

---

## 24.8.331 Implementation Milestone

At this point, a researcher should be able to implement the conceptual pipeline:

```python
dataset = load_materials_dataset()

conditions = extract_target_properties(
    dataset
)

model = build_conditional_crystal_diffusion(
    dataset
)

train(
    model,
    dataset
)

generated = generate_crystals(
    model,
    target_conditions
)

valid = validate_structures(
    generated
)

predictions = property_models(
    valid
)

ranked = rank_candidates(
    predictions
)

selected = select_for_dft(
    ranked
)
```

The implementation is deliberately modular.

Each component can be independently improved:

```text
Better representation
        ↓
Better diffusion model
        ↓
Better property predictor
        ↓
Better chemical constraints
        ↓
Better candidate selection
        ↓
Better materials discovery
```

This modularity is extremely useful for research.

---

## 24.8.332 What Remains

The crystal-generation framework is now substantially developed, but several advanced topics remain before the generative-AI portion of the chapter is complete.

The next stages should address:

```text
Advanced crystal diffusion architectures
```

```text
Score-based generative modeling
```

```text
Continuous-time diffusion
```

```text
SDE-based crystal generation
```

```text
Probability-flow ODEs
```

```text
Advanced equivariant networks
```

```text
Crystal-specific diffusion objectives
```

```text
Evaluation of generated materials
```

```text
Validity
```

```text
Novelty
```

```text
Diversity
```

```text
Property accuracy
```

```text
Distribution matching
```

```text
Benchmarking against known materials
```

```text
DFT-based validation
```

```text
Failure analysis
```

```text
Reproducible generative-materials experiments
```

These topics will move the discussion from a conceptual crystal generator toward a **research-grade generative materials discovery framework**.

# 24.9 — Advanced Crystal Diffusion and Score-Based Generative Modeling

The previous sections developed the representation of periodic crystals and established the basic framework for diffusion-based crystal generation.

The next step is to move from a discrete, finite-step diffusion description toward the more general mathematical framework of **score-based generative modeling**.

This transition is important because modern generative models increasingly formulate diffusion as a continuous stochastic process rather than simply as a fixed sequence of denoising steps.

The central idea is:

```text
Data Distribution
      ↓
Add Noise Continuously
      ↓
Continuous Probability Process
      ↓
Learn the Score
      ↓
Reverse the Process
      ↓
Generate New Samples
```

For materials science, the corresponding process becomes

```text
Real Crystal
      ↓
Continuous Perturbation
      ↓
Noisy Crystal
      ↓
Score Network
      ↓
Reverse Generation
      ↓
Generated Crystal
```

The advantage of this formulation is that it provides a unified mathematical language for

* diffusion models,
* score matching,
* stochastic differential equations,
* reverse-time stochastic processes,
* probability-flow ODEs,
* continuous-time sampling,
* and advanced conditional generation.

For crystal generation, these ideas provide the theoretical foundation for constructing models that operate directly on

```text
Atomic Coordinates
+
Lattice
+
Atomic Species
+
Periodic Geometry
+
Material Conditions
```

rather than treating generation as a simple vector-denoising problem.

---

## 24.9.1 From Discrete Diffusion to Continuous Diffusion

A conventional diffusion model uses a finite number of time steps.

For example,

```text
x₀
↓
x₁
↓
x₂
↓
...
↓
xₜ
```

where

```text
t = 1,2,...,T
```

At each step, a small amount of noise is introduced.

The forward process can be written as

```text
q(xₜ | xₜ₋₁)
```

and the complete forward process becomes

```text
q(x₁:T | x₀)
=
∏ₜ q(xₜ | xₜ₋₁)
```

The reverse model attempts to learn

```text
pθ(xₜ₋₁ | xₜ)
```

so that generation can begin from noise and proceed toward data.

This formulation is extremely useful.

However, it introduces an arbitrary choice:

> How many diffusion steps should be used?

One model might use

```text
T = 1000
```

while another may use

```text
T = 100
```

or

```text
T = 50
```

The continuous-time formulation removes this discrete restriction.

Instead of defining only

```text
t = 1,2,...,T
```

we allow

```text
t ∈ [0,1]
```

or more generally

```text
t ∈ [0,T]
```

The diffusion process then evolves continuously.

---

## 24.9.2 Continuous-Time Crystal Diffusion

Let

```text
C₀
```

represent a crystal.

A continuous diffusion process can be written as

```text
C(t)
```

where

```text
t = 0
```

corresponds approximately to the original crystal distribution and

```text
t = 1
```

corresponds to a highly noisy distribution.

Conceptually,

```text
t = 0

Real Crystal
     ↓
   slight noise
     ↓
   more noise
     ↓
   stronger noise
     ↓
t = 1

Randomized Crystal
```

The model therefore learns a continuous transformation

```text
p₀(C)
      →
pₜ(C)
      →
p₁(C)
```

where `p₀` is the data distribution and `p₁` is chosen to be a simple distribution such as a Gaussian.

For crystal generation,

```text
p₀(C)
```

represents the distribution of physically observed or computationally generated materials.

The model learns how this distribution evolves as noise is progressively introduced.

---

## 24.9.3 The Score Function

The central object in score-based generative modeling is the **score function**.

For a probability density

```text
p(x)
```

the score is

```text
∇ₓ log p(x)
```

It describes the direction in which the log probability increases most rapidly.

In other words, it tells us

> In which direction should we move a noisy sample to make it more likely under the data distribution?

For a crystal representation `C`, the corresponding score is

```text
∇C log p(C)
```

This can be interpreted as a vector field over crystal space.

Conceptually,

```text
             High probability
                    ●
                 ↗  ↑  ↖
              ↗     │     ↖
           ↗        │        ↖
        noisy crystal
```

The score points toward regions of higher probability.

This is extremely useful for generation.

Instead of directly asking the neural network

```text
"What is the clean crystal?"
```

we can ask

```text
"What direction should this noisy crystal move
to become more likely under the crystal distribution?"
```

That direction is represented by the score.

---

## 24.9.4 Score-Based Generation

Suppose a noisy crystal is represented by

```text
Cₜ
```

and the model estimates

```text
sθ(Cₜ,t)
≈
∇Cₜ log pₜ(Cₜ)
```

The neural network therefore receives

```text
Noisy Crystal
+
Noise Level
```

and predicts a vector field.

Conceptually,

```text
Cₜ
 │
 ▼
Crystal Encoder
 │
 ├── Noise-Level Embedding
 │
 ▼
Score Network
 │
 ▼
Estimated Score
```

The estimated score tells the sampler how to modify the current structure.

This produces a trajectory

```text
Random Structure
      ↓
Less Random Structure
      ↓
More Crystal-Like Structure
      ↓
Chemically Plausible Structure
      ↓
Generated Crystal
```

---

## 24.9.5 Score Network for Crystal Structures

A score network can be represented abstractly as

```python
class CrystalScoreNetwork(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            )
        )

    def forward(
        self,
        x,
        t
    ):

        return self.network(x)
```

This is only a conceptual implementation.

A real crystal score network should generally include

```text
Atomic Embeddings
+
Periodic Neighbor Information
+
Lattice Information
+
Geometric Features
+
Time Embedding
+
Equivariant Operations
```

Therefore, a research-grade architecture will be substantially more sophisticated.

---

# 24.9.6 Noise-Level Conditioning

The score depends on the noise level.

Therefore,

```text
sθ(C)
```

is insufficient.

The model should instead estimate

```text
sθ(C,t)
```

The same structure can have very different denoising directions depending on the amount of noise.

For example,

```text
Low Noise
```

requires fine structural corrections.

The model may need to distinguish

```text
slightly incorrect bond length
```

from

```text
slightly incorrect atomic position
```

At high noise, however, the model must recover much more global information.

Therefore,

```text
High Noise
→
Global Structural Reconstruction
```

while

```text
Low Noise
→
Fine Structural Refinement
```

This is why the noise level is an essential input.

---

# 24.9.7 Time Embeddings

The scalar time variable can be converted into a high-dimensional embedding.

A common approach is sinusoidal embedding.

```python
import torch
import math


def time_embedding(
    t,
    dim
):

    half = dim // 2

    frequencies = torch.exp(
        -math.log(10000)
        *
        torch.arange(
            half,
            device=t.device
        )
        /
        half
    )

    angles = (
        t[:, None]
        *
        frequencies[None, :]
    )

    return torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles)
        ],
        dim=-1
    )
```

The resulting vector allows the network to distinguish different diffusion stages.

For example,

```text
t = 0.01
```

and

```text
t = 0.90
```

produce substantially different embeddings.

The network can therefore learn different denoising behavior at different noise levels.

---

# 24.9.8 Noise-Conditioned Crystal Network

A more appropriate conceptual architecture is

```python
class NoiseConditionedCrystalModel(
    nn.Module
):

    def __init__(
        self,
        crystal_dim,
        hidden_dim,
        time_dim
    ):

        super().__init__()

        self.crystal_encoder = nn.Linear(
            crystal_dim,
            hidden_dim
        )

        self.time_encoder = nn.Linear(
            time_dim,
            hidden_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                crystal_dim
            )
        )

    def forward(
        self,
        crystal,
        t
    ):

        crystal_features = (
            self.crystal_encoder(
                crystal
            )
        )

        t_features = (
            self.time_encoder(
                t
            )
        )

        h = (
            crystal_features
            +
            t_features
        )

        return self.network(h)
```

Again, this should be regarded as a pedagogical implementation.

For actual crystal generation, the internal representation should be replaced with a periodic graph or equivariant representation.

---

# 24.9.9 Score Matching

A central challenge is that the true score

```text
∇ₓ log p(x)
```

is generally unknown.

The model therefore needs a training objective that allows the score to be learned indirectly.

This leads to **score matching**.

The basic objective is related to minimizing the difference between the learned score and the true score.

Conceptually,

```text
Learned Score
      ↓
sθ(x)
      ↓
compare
      ↑
True Score
      ↓
∇x log p(x)
```

In practice, the true data-density gradient is difficult to calculate directly.

Denoising score matching provides a practical solution.

---

# 24.9.10 Denoising Score Matching

Instead of training directly on

```text
∇x log p(x)
```

we perturb the data with known noise.

Let

```text
x₀
```

be a clean sample.

Generate

```text
xₜ
```

using a known perturbation process.

The conditional score

```text
∇xₜ log p(xₜ | x₀)
```

can often be calculated analytically.

The model is trained to reproduce this score.

The objective becomes conceptually

```text
L =
E[
||
sθ(xₜ,t)
-
target_score
||²
]
```

This is particularly attractive for materials because clean training structures are available from

```text
Materials databases
DFT calculations
Experimental crystal structures
```

and noisy versions can be generated artificially.

---

# 24.9.11 Gaussian Perturbation

Suppose

```text
xₜ
=
x₀
+
σₜ ε
```

where

```text
ε ~ N(0,I)
```

The conditional score of the Gaussian perturbation is

```text
∇xₜ log p(xₜ|x₀)
=
-
(xₜ-x₀)
/
σₜ²
```

The model therefore learns

```text
sθ(xₜ,t)
≈
-
(xₜ-x₀)
/
σₜ²
```

This gives the network a direct training target.

For crystal coordinates,

```text
x
```

may correspond to

```text
fractional coordinates
```

or an appropriate transformed geometric representation.

---

# 24.9.12 Crystal Coordinate Score Matching

Suppose the fractional coordinates are

```python
coordinates
```

Noise can be generated using

```python
noise = torch.randn_like(
    coordinates
)
```

Then

```python
sigma = 0.1

noisy_coordinates = (
    coordinates
    +
    sigma * noise
)
```

The target score is

```python
target_score = (
    -noise
    / sigma
)
```

because

```text
xₜ - x₀
=
σ ε
```

and therefore

```text
-(xₜ-x₀)/σ²
=
-ε/σ
```

The training loss becomes

```python
predicted_score = model(
    noisy_coordinates,
    sigma
)

loss = torch.mean(
    (
        predicted_score
        -
        target_score
    ) ** 2
)
```

This provides a simple demonstration of score matching.

For periodic crystals, however, the perturbation must respect periodic geometry.

---

# 24.9.13 Periodic Score Matching

A naïve Gaussian perturbation can create problems near periodic boundaries.

Consider

```text
x = 0.99
```

and a noisy perturbation that produces

```text
x = 1.04
```

The equivalent periodic coordinate is

```text
x = 0.04
```

Therefore, the score should be defined consistently under

```text
x
≡
x + n
```

for integer lattice translations.

The model should therefore satisfy an appropriate periodicity condition.

Conceptually,

```text
s(x)
=
s(x+n)
```

for integer periodic translations under the chosen coordinate convention.

This requirement becomes especially important when the model operates directly in fractional coordinates.

---

# 24.9.14 Noise Schedules

The amount of noise applied during diffusion is controlled by a noise schedule.

A simple schedule might gradually increase

```text
σ
```

from

```text
σmin
```

to

```text
σmax
```

For example,

```python
sigma_min = 0.01
sigma_max = 10.0
```

A logarithmic schedule can be useful:

```python
def sigma_schedule(
    t,
    sigma_min,
    sigma_max
):

    return (
        sigma_min
        *
        (
            sigma_max
            /
            sigma_min
        ) ** t
    )
```

At

```text
t = 0
```

the noise level is approximately

```text
σmin
```

while at

```text
t = 1
```

it reaches

```text
σmax
```

This provides a continuous range of noise scales.

---

# 24.9.15 Why Multiple Noise Scales Are Important

Crystal structures contain information at different spatial scales.

At small noise levels, the model may learn

```text
Bond lengths
Bond angles
Local coordination
```

At intermediate noise levels, it may learn

```text
Local motifs
Coordination environments
Polyhedral arrangements
```

At large noise levels, the model must recover

```text
Composition
Crystal topology
Lattice geometry
Global structural organization
```

Therefore, score-based modeling naturally creates a hierarchy:

```text
Small Noise
    ↓
Local Structure
    ↓
Medium Noise
    ↓
Medium-Range Structure
    ↓
Large Noise
    ↓
Global Crystal Organization
```

This multiscale behavior is particularly valuable for materials generation.

---

# 24.9.16 Noise-Conditional Training

During training, a noise level can be sampled randomly.

```python
batch_size = coordinates.shape[0]

t = torch.rand(
    batch_size,
    device=coordinates.device
)

sigma = sigma_schedule(
    t,
    sigma_min=0.01,
    sigma_max=10.0
)
```

Noise is then generated:

```python
noise = torch.randn_like(
    coordinates
)
```

and the noisy coordinates are constructed:

```python
noisy = (
    coordinates
    +
    sigma[:, None, None]
    * noise
)
```

The model predicts the score:

```python
predicted_score = model(
    noisy,
    t
)
```

The target is

```python
target_score = (
    -noise
    /
    sigma[:, None, None]
)
```

and the loss is

```python
loss = torch.mean(
    (
        predicted_score
        -
        target_score
    ) ** 2
)
```

This is the basic computational structure of a noise-conditioned score model.

---

# 24.9.17 Weighted Score Matching

Different noise levels can have very different score magnitudes.

At small `σ`,

```text
1/σ
```

can become very large.

Therefore, directly averaging losses across all noise levels may cause some regions of the noise schedule to dominate training.

A weighted objective can instead be used:

```text
L =
E[
λ(σ)
||
sθ(x,σ)
-
target
||²
]
```

where

```text
λ(σ)
```

controls the contribution of each noise level.

The exact weighting strategy depends on the score parameterization and diffusion formulation.

This is an important implementation detail because poor weighting can make training unstable.

---

# 24.9.18 Score-Based Crystal Sampling

After training, the model can generate structures starting from a highly noisy distribution.

Conceptually,

```text
Random Noise
     ↓
Score Network
     ↓
Update
     ↓
Score Network
     ↓
Update
     ↓
...
     ↓
Crystal
```

At each step, the score provides a direction in which the current sample should move.

A simple Euler-like update can be written conceptually as

```text
xnew
=
xold
+
Δt × update_direction
```

For a basic score-driven sampler,

```python
x = torch.randn(
    shape,
    device=device
)

for sigma in sigmas:

    score = model(
        x,
        sigma
    )

    step = step_size(
        sigma
    )

    x = (
        x
        +
        step * score
    )
```

This is only a simplified illustration.

Real score-based samplers use carefully derived stochastic or deterministic updates.

---

# 24.9.19 Langevin Dynamics

One important score-based sampling method is Langevin dynamics.

The basic idea combines

```text
Score Direction
+
Random Noise
```

so that the sample moves toward high-probability regions while retaining stochasticity.

A simplified update is

```text
xₖ₊₁
=
xₖ
+
ε sθ(xₖ)
+
√(2ε) zₖ
```

where

```text
zₖ ~ N(0,I)
```

The first term

```text
ε sθ(xₖ)
```

moves toward higher probability.

The second term

```text
√(2ε)zₖ
```

introduces stochastic exploration.

For materials generation, this means the model can explore multiple plausible structures rather than following one deterministic trajectory.

---

# 24.9.20 Annealed Langevin Dynamics

Using one noise scale is often insufficient.

Instead, the model can use a sequence

```text
σ₁ > σ₂ > ... > σL
```

and perform Langevin sampling at each level.

Conceptually,

```text
High Noise
    ↓
Langevin Sampling
    ↓
Lower Noise
    ↓
Langevin Sampling
    ↓
Lower Noise
    ↓
...
    ↓
Low Noise
    ↓
Final Crystal
```

This is called **annealed Langevin dynamics**.

It is particularly intuitive for crystal generation.

At high noise,

```text
global exploration
```

is emphasized.

At low noise,

```text
structural refinement
```

becomes dominant.

---

# 24.9.21 Crystal Annealed Langevin Sampling

A simplified implementation is

```python
x = torch.randn(
    crystal_shape,
    device=device
)

for sigma in reversed(
    sigma_levels
):

    for _ in range(
        num_steps
    ):

        score = model(
            x,
            sigma
        )

        noise = torch.randn_like(
            x
        )

        step_size = (
            0.01
            *
            sigma ** 2
        )

        x = (
            x
            +
            step_size * score
            +
            torch.sqrt(
                2 * step_size
            )
            * noise
        )
```

This code is intentionally simplified.

A real implementation must additionally handle

```text
periodic coordinates
lattice constraints
atomic species
equivariance
chemical validity
```

and other crystal-specific requirements.

---

# 24.9.22 Stochastic Differential Equations

The continuous diffusion process can be formulated using a stochastic differential equation.

A general SDE is

```text
dx
=
f(x,t)dt
+
g(t)dW
```

where

```text
f(x,t)
```

is the drift term,

```text
g(t)
```

is the diffusion coefficient,

and

```text
W
```

is a Wiener process.

The first component

```text
f(x,t)dt
```

describes deterministic evolution.

The second component

```text
g(t)dW
```

introduces stochasticity.

This gives a powerful mathematical framework for diffusion models.

---

# 24.9.23 Forward SDE

The forward diffusion process can therefore be represented as

```text
dC
=
f(C,t)dt
+
g(t)dW
```

where `C` represents the crystal state.

For example,

```text
C =
(L,Z,R)
```

under an appropriate continuous representation.

The forward process gradually destroys structural information.

Conceptually,

```text
Real Crystal
     ↓
Slightly Noisy Crystal
     ↓
Noisier Crystal
     ↓
Highly Noisy Crystal
```

At the terminal time, the distribution should become sufficiently simple for sampling.

---

# 24.9.24 Variance-Preserving SDE

One important family is the variance-preserving SDE.

A common form is

```text
dx
=
-
1/2 β(t)x dt
+
√β(t)dW
```

where

```text
β(t)
```

controls the noise schedule.

The process gradually reduces the signal contribution while injecting Gaussian noise.

This provides a continuous-time analogue of commonly used discrete diffusion processes.

For crystal generation, the same conceptual framework can be applied to continuous structural representations.

---

# 24.9.25 Variance-Exploding SDE

Another important family is the variance-exploding SDE.

A simplified form is

```text
dx
=
g(t)dW
```

where the noise variance increases substantially over time.

The resulting distribution becomes increasingly broad.

This formulation is particularly natural for score-based models trained over many noise scales.

The model learns

```text
sθ(x,t)
```

across the entire noise range.

---

# 24.9.26 Why SDEs Are Useful for Materials Generation

The SDE formulation provides several advantages.

First, it gives a continuous description of the generation process.

Second, it separates the physical representation from the numerical integration method.

Third, it provides access to mathematically derived reverse-time processes.

Fourth, it allows different samplers to be constructed from the same trained score model.

This is valuable for research because the same trained model may support

```text
Stochastic Sampling
```

and

```text
Deterministic Sampling
```

through different numerical procedures.

---

# 24.9.27 Reverse-Time SDE

The most important result in score-based generative modeling is that the forward SDE can be reversed.

Suppose the forward process is

```text
dx
=
f(x,t)dt
+
g(t)dW
```

Then the reverse-time process contains a term involving the score:

```text
dx
=
[
f(x,t)
-
g(t)^2
∇x log pₜ(x)
]
dt
+
g(t)dW̄
```

with time interpreted in the reverse direction.

The key term is

```text
∇x log pₜ(x)
```

which is exactly what the score network learns.

Therefore,

```text
Forward SDE
       ↓
Data → Noise

Reverse SDE
       ↑
Noise → Data
```

The trained score network makes the reverse process possible.

---

# 24.9.28 Reverse-Time Crystal Generation

For crystals, the conceptual process is

```text
Random Crystal Representation
             ↓
      Reverse SDE
             ↓
      Score Network
             ↓
     Periodic Structure
             ↓
      Chemical Structure
             ↓
       Final Crystal
```

At each integration step, the model evaluates

```text
sθ(C,t)
```

and uses it to modify the current crystal state.

This means the score network effectively acts as a learned vector field over crystal configuration space.

---

# 24.9.29 Crystal Score as a Vector Field

Imagine a simplified two-dimensional crystal configuration space.

Each point represents a possible structure.

The score defines a vector:

```text
        ↗  ↑  ↖
      →   ●   ←
        ↘  ↓  ↙
```

A generated structure follows this field.

In the actual problem, however, the space is enormously larger.

A crystal may contain

```text
N atoms
```

with three coordinates per atom, plus lattice variables and discrete chemical identities.

The effective configuration space can therefore have hundreds or thousands of dimensions.

The neural network approximates the score field in this high-dimensional space.

This is one reason why powerful geometric architectures are required.

---

# 24.9.30 Probability-Flow ODE

A remarkable result is that the same score function can define a deterministic ordinary differential equation.

This is called the **probability-flow ODE**.

Instead of the stochastic reverse SDE,

```text
dW
```

is removed.

The resulting process has the general form

```text
dx
=
[
f(x,t)
-
1/2 g(t)^2
∇x log pₜ(x)
]
dt
```

The exact sign and time orientation depend on the integration convention.

The important conceptual distinction is:

```text
Reverse SDE
→ stochastic

Probability-flow ODE
→ deterministic
```

Both can produce samples from the same target distribution when solved appropriately.

---

# 24.9.31 Why Probability-Flow ODEs Matter

The probability-flow ODE has several important uses.

It can provide

```text
Deterministic generation
```

from a fixed initial condition.

It can also be used for

```text
Likelihood estimation
```

under appropriate assumptions.

Furthermore, it allows the generation trajectory to be interpreted as a continuous deterministic path.

For research, this provides an alternative to stochastic diffusion sampling.

---

# 24.9.32 SDE vs ODE Sampling

The difference can be summarized as

```text
Reverse SDE

Noise
 ↓
Stochastic trajectory
 ↓
Crystal
```

while

```text
Probability-flow ODE

Noise
 ↓
Deterministic trajectory
 ↓
Crystal
```

The stochastic formulation can encourage exploration.

The deterministic formulation can provide reproducibility for a fixed initial state and numerical solver.

Therefore, both are useful tools.

---

# 24.9.33 Numerical Integration

The SDE or ODE itself is not directly solved symbolically in practical generative modeling.

Instead, numerical integration is used.

For an ODE,

```text
dx/dt = F(x,t)
```

a simple Euler update is

```text
xₜ₊Δt
=
xₜ
+
Δt F(xₜ,t)
```

In Python:

```python
x = x + dt * velocity
```

More advanced methods include

```text
Euler
Heun
Runge–Kutta
Adaptive solvers
```

The choice of solver affects computational cost and sampling quality.

---

# 24.9.34 ODE Sampling Example

A conceptual implementation is

```python
def ode_step(
    x,
    t,
    dt,
    model
):

    score = model(
        x,
        t
    )

    velocity = (
        drift(
            x,
            t
        )
        -
        0.5
        *
        diffusion(
            t
        ) ** 2
        *
        score
    )

    return (
        x
        +
        dt * velocity
    )
```

A complete sampler can then integrate from high noise toward low noise.

```python
x = initial_noise

for t in time_grid:

    x = ode_step(
        x,
        t,
        dt,
        model
    )
```

For crystal generation, the state `x` would represent a structured crystal state rather than a simple vector.

---

# 24.9.35 SDE Solvers

For an SDE,

```text
dx
=
f(x,t)dt
+
g(t)dW
```

the numerical procedure must account for stochastic increments.

A simple Euler–Maruyama step is

```text
xₜ₊Δt
=
xₜ
+
f(xₜ,t)Δt
+
g(t)√Δt ε
```

where

```text
ε ~ N(0,I)
```

In code:

```python
noise = torch.randn_like(x)

x = (
    x
    +
    drift * dt
    +
    diffusion
    * torch.sqrt(
        torch.tensor(dt)
    )
    * noise
)
```

Again, the actual crystal implementation requires careful treatment of geometry and periodicity.

---

# 24.9.36 Conditional Score-Based Generation

The score framework also supports property conditioning.

Suppose

```text
y
```

is a target material property.

The model learns

```text
sθ(C,t,y)
```

which approximates

```text
∇C log pₜ(C|y)
```

The generation process becomes

```text
Random Noise
      ↓
Target Property
      ↓
Conditional Score
      ↓
Reverse SDE / ODE
      ↓
Generated Crystal
```

For example,

```text
y = 2.0 eV
```

may represent a target band gap.

The score network then attempts to guide generation toward structures compatible with that property.

---

# 24.9.37 Multi-Property Score Conditioning

The condition can be a vector:

```text
y =
[
Band Gap,
Formation Energy,
Bulk Modulus,
Density
]
```

The conditional score becomes

```text
sθ(C,t,y)
```

The property vector is embedded:

```python
property_embedding = property_encoder(
    properties
)
```

and combined with the crystal representation.

Conceptually,

```text
Crystal
   +
Time
   +
Property
   ↓
Conditional Score Network
   ↓
Score
```

This provides a continuous-time formulation of multi-objective crystal generation.

---

# 24.9.38 Classifier Guidance in the Score Framework

Suppose a separate property predictor estimates

```text
p(y|C)
```

Then its gradient with respect to the crystal representation is

```text
∇C log p(y|C)
```

This gradient can guide the generation trajectory.

Conceptually,

```text
Base Score
     +
Property Gradient
     ↓
Guided Score
```

A simplified guided score can be represented as

```text
guided_score
=
base_score
+
w × property_gradient
```

where

```text
w
```

controls guidance strength.

This provides a direct connection between

```text
Property Prediction
```

and

```text
Crystal Generation
```

---

# 24.9.39 Classifier-Free Score Guidance

Classifier-free guidance avoids requiring a separate classifier.

During training, the condition is randomly removed for some samples.

The model therefore learns both

```text
sθ(C,t,y)
```

and

```text
sθ(C,t)
```

within the same architecture.

During sampling,

```text
guided
=
unconditional
+
w(
conditional
-
unconditional
)
```

The parameter

```text
w
```

controls the strength of conditioning.

This is particularly attractive for materials generation because it avoids maintaining a separate property classifier for every generation task.

---

# 24.9.40 Guidance Strength and Materials Diversity

Increasing guidance strength generally makes the generation process more strongly aligned with the requested condition.

However,

```text
Higher Guidance
```

does not necessarily mean

```text
Better Materials
```

Very strong guidance may cause

```text
Reduced Diversity
```

or

```text
Unphysical Structures
```

Therefore, guidance should be evaluated jointly using

```text
Property Accuracy
+
Validity
+
Novelty
+
Diversity
```

This will become particularly important in Section 24.12.

---

# 24.9.41 Crystal-Specific Score Decomposition

A crystal contains multiple components.

Therefore, the score can conceptually be decomposed into

```text
Score(C)
=
Score(L)
+
Score(R)
+
Score(Z)
```

where

```text
L = lattice
R = coordinates
Z = species
```

This decomposition is conceptual rather than necessarily representing independent probability distributions.

In a real architecture, these components may interact strongly.

For example,

```text
Composition
```

influences

```text
Lattice
```

and

```text
Lattice
```

influences

```text
Atomic Distances
```

Therefore, the model should generally learn their joint relationships.

---

# 24.9.42 Joint Crystal Score Model

A more realistic representation is

```text
                  Crystal
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
     Lattice       Species    Coordinates
        │            │            │
        └────────────┼────────────┘
                     ↓
             Joint Representation
                     ↓
               Score Network
                     ↓
               Joint Score
```

The score network can therefore update the different components coherently.

This is essential.

It would be undesirable for the model to produce

```text
chemically plausible composition
```

but

```text
geometrically impossible structure
```

or

```text
reasonable coordinates
```

with

```text
an invalid lattice.
```

---

# 24.9.43 Energy-Aware Score Modeling

A particularly interesting extension is to incorporate physical energy.

Suppose an energy model provides

```text
E(C)
```

for a crystal.

The probability distribution may conceptually be related to

```text
p(C)
∝
exp(
-βE(C)
)
```

under a simplified equilibrium interpretation.

The corresponding energy contribution to the score is related to

```text
-β∇C E(C)
```

This suggests a connection between

```text
Generative Modeling
```

and

```text
Physical Energy Landscapes
```

A generative model can therefore potentially learn to generate structures that occupy physically favorable regions of configuration space.

---

# 24.9.44 Combining Learned and Physical Guidance

A conceptual guided score can be written as

```text
Total Score
=
Learned Score
+
Property Guidance
+
Energy Guidance
+
Constraint Guidance
```

For example,

```text
              Learned Crystal Score
                       │
                       +
                Property Gradient
                       │
                       +
                 Energy Gradient
                       │
                       +
                Constraint Signal
                       │
                       ▼
                  Guided Score
                       │
                       ▼
                Crystal Sampling
```

This is an important direction for materials generative modeling because purely statistical generation can produce structures that look plausible to the model but violate physical constraints.

---

# 24.9.45 Differentiable Property Guidance

If the property predictor is differentiable,

```text
ŷ = fθ(C)
```

then the gradient

```text
∇C ŷ
```

can be calculated.

This gradient indicates how the crystal representation should change to alter the predicted property.

For example,

```python
predicted_band_gap = predictor(
    crystal
)

gradient = torch.autograd.grad(
    predicted_band_gap,
    crystal
)[0]
```

This gradient can potentially be incorporated into a generation procedure.

However, gradients from imperfect predictors must be used carefully.

A property model may have high predictive accuracy on known materials but behave unpredictably far outside its training distribution.

This is one reason that generated candidates must eventually be validated using higher-fidelity methods.

---

# 24.9.46 Uncertainty-Aware Guidance

A property predictor should ideally provide not only

```text
Prediction
```

but also

```text
Uncertainty
```

For example,

```text
Band Gap
=
1.8 ± 0.2 eV
```

A generation system can then penalize candidates for which the predictor is highly uncertain.

Conceptually,

```text
Generated Crystal
       ↓
Property Predictor
       ↓
Prediction + Uncertainty
       ↓
Candidate Ranking
```

This can prevent the system from exploiting weaknesses in the predictor.

---

# 24.9.47 Predictor Exploitation Problem

Suppose a property predictor has learned

```text
f(C)
```

but its training data covers only a limited region of materials space.

The generator may discover a structure far outside that region where

```text
f(C)
```

predicts an exceptionally attractive property.

However, this prediction may be unreliable.

The generator has effectively learned to exploit the predictor's weakness.

This phenomenon is sometimes described as

```text
Model Exploitation
```

or

```text
Reward Hacking
```

in broader generative-learning contexts.

For materials research, this is a serious concern.

Therefore,

```text
High Predicted Performance
```

should never automatically imply

```text
High Physical Performance
```

---

# 24.9.48 Out-of-Distribution Detection

A generated crystal should therefore be compared against the training distribution.

Possible representations include

```text
Composition Descriptors
Structural Fingerprints
Graph Embeddings
Latent Representations
```

The distance from the training distribution can be estimated.

For example,

```python
distance = torch.norm(
    generated_embedding
    -
    training_centroid
)
```

A simple distance threshold can identify potentially unusual structures.

More sophisticated approaches include

```text
Mahalanobis distance
Density estimation
Nearest-neighbor distance
Embedding-space density
Ensemble uncertainty
```

These methods can be incorporated into candidate screening.

---

# 24.9.49 Crystal Generation as Energy-Landscape Exploration

The score perspective gives a useful interpretation.

The crystal space can be imagined as a highly complex landscape.

```text
Probability
   ↑
   │          ●
   │       ●     ●
   │    ●           ●
   │  ●               ●
   └────────────────────→
          Crystal Space
```

High-probability regions correspond to structures frequently represented in the training distribution.

Low-probability regions correspond to unusual or poorly represented structures.

The score points toward increasing probability.

Generation therefore becomes a process of moving through this landscape.

The challenge is to reach regions that are simultaneously

```text
Realistic
+
Novel
+
Chemically Valid
+
Property-Optimal
```

This is the fundamental objective of generative materials discovery.

---

# 24.9.50 Advanced Crystal Diffusion Architectures

A research-grade crystal diffusion model can now be conceptualized as a combination of several components.

```text
                 Crystal
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
    Species      Coordinates    Lattice
       │            │            │
       └────────────┼────────────┘
                    ▼
          Periodic Graph Builder
                    │
                    ▼
            Equivariant Encoder
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
        Time     Property   Energy
      Embedding  Embedding  Signal
          │         │         │
          └─────────┼─────────┘
                    ▼
             Score Network
                    │
                    ▼
             Crystal Score
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Reverse SDE        Probability ODE
          │                   │
          └─────────┬─────────┘
                    ▼
             Generated Crystal
```

This architecture combines the major concepts introduced in this section.

---

# 24.9.51 Minimal Research Implementation

A simplified research prototype can begin with a continuous representation.

```python
import torch
import torch.nn as nn


class ScoreNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        time_dim
    ):

        super().__init__()

        self.time_encoder = nn.Sequential(

            nn.Linear(
                time_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.input_encoder = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            )
        )

    def forward(
        self,
        x,
        time_embedding
    ):

        h = self.input_encoder(x)

        t = self.time_encoder(
            time_embedding
        )

        h = h + t

        return self.network(h)
```

This is not yet a crystal-specific architecture.

It is a minimal framework for understanding the computational principle.

---

# 24.9.52 Training Loop for a Score Model

A simplified training loop is:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


for epoch in range(
    num_epochs
):

    for crystal in dataloader:

        x0 = crystal.to(device)

        t = torch.rand(
            x0.shape[0],
            device=device
        )

        sigma = sigma_schedule(
            t,
            0.01,
            10.0
        )

        noise = torch.randn_like(
            x0
        )

        xt = (
            x0
            +
            sigma[:, None]
            * noise
        )

        target = (
            -noise
            /
            sigma[:, None]
        )

        t_embed = time_embedding(
            t,
            time_dim
        )

        predicted = model(
            xt,
            t_embed
        )

        loss = torch.mean(
            (
                predicted
                -
                target
            ) ** 2
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

The implementation captures the essential structure:

```text
Clean Data
   ↓
Sample Noise Level
   ↓
Add Noise
   ↓
Predict Score
   ↓
Compare with Target
   ↓
Backpropagation
```

For crystal materials, however, the next implementation level must replace the simple vector `x0` with a physically meaningful crystal representation.

---

# 24.9.53 From Prototype to Research Model

The transition from the minimal implementation to a research-grade model requires several upgrades.

### Prototype

```text
Vector
+
MLP
+
Gaussian Noise
```

### Research Model

```text
Periodic Crystal
+
Element Embeddings
+
Periodic Graph
+
Lattice Representation
+
Equivariant Network
+
Continuous Noise Conditioning
+
Score Objective
+
Property Conditioning
+
Physical Constraints
```

The difference is substantial.

The prototype teaches the mathematics.

The research model addresses the actual materials problem.

---

# 24.9.54 Recommended Crystal Diffusion Architecture

For a serious implementation, the model can be organized into the following modules.

```text
Crystal Dataset
      ↓
Structure Parser
      ↓
Periodic Graph Construction
      ↓
Element Embedding
      ↓
Lattice Encoder
      ↓
Coordinate Representation
      ↓
Equivariant Message Passing
      ↓
Time Embedding
      ↓
Property Conditioning
      ↓
Score Head
      ↓
Reverse SDE / ODE
      ↓
Generated Structure
```

Each module should have a clearly defined scientific role.

This modularity is important for research because it allows individual components to be replaced and compared.

---

# 24.9.55 Experimental Ablation Strategy

A research project should not immediately begin with the most complicated architecture.

Instead, models can be developed incrementally.

### Model A — MLP baseline

```text
Crystal descriptors
→ MLP
→ Diffusion
```

### Model B — Graph model

```text
Crystal
→ Periodic Graph
→ GNN
→ Diffusion
```

### Model C — Equivariant model

```text
Crystal
→ Periodic Graph
→ Equivariant GNN
→ Diffusion
```

### Model D — Conditional model

```text
Crystal
+
Target Property
→
Equivariant Diffusion
```

### Model E — Physics-guided model

```text
Crystal
+
Property
+
Energy
+
Constraints
→
Equivariant Diffusion
```

This sequence allows the researcher to determine which components actually improve generation quality.

---

# 24.9.56 Research Questions

The advanced diffusion framework creates several research questions.

For example:

1. Does an equivariant architecture improve structural validity?

2. Does continuous-time diffusion improve sampling efficiency?

3. Does classifier-free guidance improve property accuracy?

4. Does stronger guidance reduce crystal diversity?

5. Does energy guidance improve DFT stability?

6. Does periodic representation improve generation compared with Cartesian coordinates?

7. Does an SDE sampler outperform an ODE sampler?

8. How does the noise schedule affect crystal novelty?

9. How many generated candidates are required to discover useful structures?

10. Does the model generate materials outside the training distribution?

These questions transform the generative model from a programming exercise into a genuine research framework.

---

# 24.9.57 Computational Cost

Generative crystal models can be computationally expensive.

The total cost may involve

```text
Training
+
Sampling
+
Structure Validation
+
ML Property Prediction
+
DFT Validation
```

If the model generates

```text
100,000
```

candidate crystals, even inexpensive validation becomes computationally significant.

Therefore, an efficient pipeline should progressively reduce the candidate set.

For example,

```text
100,000 Generated
        ↓
Validity Filter
        ↓
60,000
        ↓
Duplicate Removal
        ↓
40,000
        ↓
ML Property Screening
        ↓
2,000
        ↓
Multi-Objective Ranking
        ↓
200
        ↓
DFT
        ↓
20
```

This hierarchical screening strategy is essential for practical materials discovery.

---

# 24.9.58 The Role of DFT in the Generative Loop

The final objective is not merely to generate structures that resemble the training data.

The objective is to discover physically meaningful materials.

Therefore,

```text
Generative AI
```

should eventually connect to

```text
First-Principles Calculations
```

The complete loop becomes

```text
Generate
   ↓
Validate
   ↓
Predict
   ↓
Rank
   ↓
DFT
   ↓
Update Knowledge
   ↓
Generate Again
```

This creates the foundation for an iterative materials-discovery system.

---

# 24.9.59 Summary of the Advanced Diffusion Framework

The transition from discrete diffusion to score-based continuous-time modeling provides a powerful theoretical framework for crystal generation.

The key concepts are:

```text
Discrete Diffusion
        ↓
Continuous Diffusion
        ↓
Score Function
        ↓
Score Matching
        ↓
SDE
        ↓
Reverse-Time SDE
        ↓
Probability-Flow ODE
        ↓
Continuous Sampling
        ↓
Crystal Generation
```

For materials science, this framework must additionally account for

```text
Periodic Geometry
+
Atomic Species
+
Lattice
+
Symmetry
+
Chemical Constraints
+
Material Properties
```

The final research problem is therefore not simply

```text
Learn p(C)
```

but rather

```text
Learn a physically meaningful generative process
for periodic crystal structures.
```

---

# 24.9.60 Transition to the Next Section

The score-based framework provides the mathematical machinery required for continuous-time generation.

However, one major problem remains.

A crystal is not an ordinary vector.

Its representation must respect the fundamental symmetries of three-dimensional space.

These include

```text
Translation
Rotation
Reflection
Permutation
```

A model that ignores these symmetries may waste computational capacity learning equivalent representations of the same physical structure.

More importantly, an inappropriate architecture can produce predictions that are not physically consistent under coordinate transformations.

Therefore, the next section focuses on one of the most important architectural ideas in modern materials machine learning:

```text
Equivariance
```

The next stage will develop **E(3)- and SE(3)-equivariant neural networks**, explain why they are important for crystal generation, and show how equivariance can be integrated with diffusion models.

This leads directly to:

# 24.10 — Equivariant Generative Models for Crystals

# 24.10 Equivariant Generative Models for Crystals

The previous section developed the score-based framework for crystal generation.

We saw that a diffusion model can be interpreted as learning a vector field over crystal configuration space.

The generation process can therefore be represented as

```text
Random Crystal
      ↓
Score Network
      ↓
Reverse Diffusion
      ↓
Structured Crystal
```

However, one fundamental problem remains.

A crystal is not an ordinary vector.

Its representation contains

```text
Atomic Species
+
Atomic Coordinates
+
Lattice
+
Periodic Geometry
+
Symmetry
```

and these quantities are strongly constrained by the geometry of three-dimensional space.

If the entire crystal is translated, rotated, or reflected, the physical material does not suddenly become a different material.

Therefore, a generative model should respect these transformations.

This motivates **equivariant generative modeling**.

The central idea is:

> The neural network should transform its internal geometric representations consistently with the physical transformation applied to the crystal.

This provides a natural connection between

```text
Crystal Geometry
        +
Group Symmetry
        +
Equivariant Neural Networks
        +
Diffusion Models
```

The resulting framework is particularly powerful for generative materials discovery because it allows the model to learn the probability distribution of physically meaningful structures without wasting capacity learning coordinate-system redundancies.

---

# 24.10.1 Why Equivariance Is Necessary

Consider a crystal structure represented by atomic coordinates

```text
X
```

Suppose we rotate the entire structure using a rotation matrix

```text
R
```

The new coordinates are

```text
X' = RX
```

Physically,

```text
X
```

and

```text
X'
```

represent the same material configuration under a different coordinate system.

For a scalar property such as formation energy,

```text
E(X') = E(X)
```

The quantity is invariant.

For a vector quantity such as an atomic displacement,

```text
v' = Rv
```

The quantity is equivariant.

Therefore, a model must distinguish between two different requirements.

```text
Invariant

Input transforms
↓
Output remains unchanged
```

and

```text
Equivariant

Input transforms
↓
Output transforms predictably
```

Both concepts are important in crystal generation.

---

# 24.10.2 Invariance and Equivariance

Suppose

```text
f(X)
```

is a neural network.

For an invariant property,

```text
f(RX) = f(X)
```

For an equivariant prediction,

```text
f(RX) = Rf(X)
```

These equations express two different physical behaviors.

For example,

```text
Formation Energy
Band Gap
Density
```

are generally invariant with respect to a global rotation.

On the other hand,

```text
Atomic Displacement
Force
Position Difference
```

are geometric quantities and should transform with the structure.

Therefore, a crystal diffusion model may contain both invariant and equivariant components.

---

# 24.10.3 Symmetries of a Crystal Representation

A crystal model must generally consider several transformations.

These include

```text
Translation
Rotation
Reflection
Permutation
Periodic Translation
```

Each represents a different form of representation redundancy.

For example, translating every atom by the same vector does not change the physical structure.

```text
r_i' = r_i + t
```

where

```text
t
```

is a translation vector.

Similarly, a global rotation gives

```text
r_i' = Rr_i
```

A permutation of atom indices also should not change the physical crystal.

For example,

```text
Atom 1 = Fe
Atom 2 = O
```

and

```text
Atom 1 = O
Atom 2 = Fe
```

can represent the same structure if the associated coordinates are permuted consistently.

Therefore, the model should not depend on arbitrary atom indexing.

---

# 24.10.4 Euclidean Group E(3)

The combination of translations, rotations, and reflections in three-dimensional space is associated with the Euclidean group

```text
E(3)
```

A transformation can be written conceptually as

```text
x' = Rx + t
```

where

```text
R
```

represents a rotation or reflection and

```text
t
```

represents translation.

For a physically consistent geometric model, transformations of this form should be handled systematically.

An E(3)-equivariant network is constructed so that its intermediate representations transform consistently under these operations.

---

# 24.10.5 SE(3) Equivariance

Another important group is

```text
SE(3)
```

which consists of proper rotations and translations.

The transformation is

```text
x' = Rx + t
```

with

```text
det(R) = 1
```

This excludes reflections.

SE(3)-equivariant architectures are widely used in three-dimensional molecular and materials machine learning.

For crystal generation, these architectures are useful because the model can directly operate on geometric information while respecting the underlying spatial symmetries.

---

# 24.10.6 Why Data Augmentation Is Not Enough

A simple strategy for handling rotations is data augmentation.

Suppose the training dataset contains

```text
Crystal A
```

We can generate additional examples:

```text
Crystal A
Rotated Crystal A
Another Rotated Crystal A
Reflected Crystal A
```

The model can then learn from these examples.

However, this approach has limitations.

The network is still learning symmetry from data.

It does not mathematically guarantee the desired transformation behavior.

If the training dataset contains only a limited number of orientations, the model may not generalize perfectly to unseen transformations.

Equivariant networks solve this problem differently.

Instead of saying

```text
"Learn rotational symmetry from examples."
```

we say

```text
"Build rotational symmetry directly into the architecture."
```

This is a much stronger constraint.

---

# 24.10.7 Equivariant Diffusion

The same principle applies to diffusion models.

Suppose the crystal at diffusion time t is

```text
C_t
```

and the score network is

```text
s_θ(C_t,t)
```

For an equivariant score model, the score should transform consistently with the crystal.

Conceptually,

```text
s_θ(RC_t,t)
=
R s_θ(C_t,t)
```

This means that if the crystal is rotated, the predicted denoising direction rotates in exactly the same way.

This is essential.

Otherwise, the model might produce different physical denoising behavior merely because the researcher chose a different coordinate system.

---

# 24.10.8 Equivariant Score Field

The score function introduced earlier is

```text
∇C log p_t(C)
```

For a crystal representation, this score can contain geometric information about how the structure should change during reverse diffusion.

An equivariant score field therefore behaves as

```text
Crystal
   ↓
Geometric Representation
   ↓
Equivariant Score Network
   ↓
Equivariant Score
```

The score can contain different components.

```text
Coordinate Score
+
Lattice Score
+
Chemical Information
```

The coordinate component must respect spatial transformations.

The lattice component must transform consistently with the unit-cell geometry.

The chemical component may be represented through invariant embeddings.

---

# 24.10.9 Equivariant Representation of Atomic Coordinates

Let

```text
r_i
```

represent the position of atom i.

A naive network may directly process

```text
(r_x,r_y,r_z)
```

using ordinary linear layers.

For example,

```python
self.layer = nn.Linear(3, 128)
```

Although this is mathematically valid as a neural network operation, it does not automatically preserve rotational symmetry.

A rotation of the input coordinates does not necessarily produce the corresponding rotation of the output representation.

An equivariant architecture instead constructs features according to their transformation type.

For example,

```text
Scalar Features
+
Vector Features
+
Higher-Order Features
```

can be processed separately and combined using equivariant operations.

---

# 24.10.10 Scalar Features

Scalar features do not change under rotation.

Examples include

```text
Atomic Number
Atomic Mass
Electronegativity
Learned Chemical Embedding
Local Scalar Descriptor
```

If a scalar feature is

```text
s
```

then under rotation

```text
s' = s
```

Therefore, scalar features are naturally suitable for chemical information.

For example,

```python
atomic_number = torch.tensor([
    3,
    26,
    8
])
```

can be mapped to learned embeddings.

```python
embedding = nn.Embedding(
    119,
    hidden_dim
)

chemical_features = embedding(
    atomic_number
)
```

These features do not need to rotate.

---

# 24.10.11 Vector Features

Vector features transform with the coordinate system.

If

```text
v
```

is a vector, then after rotation

```text
v' = Rv
```

Examples include

```text
Atomic Displacement
Bond Direction
Force
Position Difference
```

A crystal network can therefore maintain vector channels that explicitly store directional information.

For example,

```text
Atom i
   │
   │ r_ij
   ▼
Atom j
```

The relative displacement

```text
r_ij = r_j - r_i
```

is a vector.

If the crystal rotates,

```text
r_ij' = Rr_ij
```

The network must preserve this relationship.

---

# 24.10.12 Higher-Order Features

Scalar and vector representations are not always sufficient.

Complex local environments may contain angular correlations that require higher-order representations.

These can be represented using irreducible representations associated with angular momentum

```text
l = 0,1,2,...
```

where

```text
l = 0
```

corresponds to scalar-like information,

```text
l = 1
```

corresponds to vector-like information,

and higher values describe increasingly complex angular structures.

Conceptually,

```text
l = 0
↓
Scalar

l = 1
↓
Vector

l = 2
↓
Quadrupole-like representation

l = 3
↓
Higher-order angular information
```

This provides the mathematical foundation for highly expressive three-dimensional geometric networks.

---

# 24.10.13 Spherical Harmonics

A central mathematical tool in equivariant neural networks is the **spherical harmonic**.

For a direction

```text
r̂
```

the spherical harmonics can be written as

```text
Y_l^m(r̂)
```

where

```text
l
```

controls the angular order and

```text
m
```

indexes the components.

They provide a systematic basis for describing angular information.

For example,

```text
Distance
```

can describe

```text
how far
```

while spherical harmonics describe

```text
which direction
```

Together they provide a powerful representation of local crystal geometry.

---

# 24.10.14 Radial and Angular Information

For two atoms,

```text
r_ij = r_j - r_i
```

we can separate the information into

```text
Distance:

d_ij = ||r_ij||
```

and

```text
Direction:

r̂_ij = r_ij / ||r_ij||
```

A radial basis can encode

```text
d_ij
```

while spherical harmonics encode

```text
r̂_ij
```

Conceptually,

```text
Relative Position
       │
       ├──────────────┐
       ↓              ↓
   Distance        Direction
       │              │
       ↓              ↓
Radial Basis     Spherical Harmonics
       │              │
       └───────┬──────┘
               ↓
      Equivariant Features
```

This separation is fundamental to many modern geometric architectures.

---

# 24.10.15 Equivariant Message Passing

A crystal can be represented as a periodic graph.

For atom i, let

```text
N(i)
```

denote its neighboring atoms.

A conventional graph network computes messages such as

```text
m_ij = φ(h_i,h_j,e_ij)
```

and aggregates them:

```text
m_i = Σ_j m_ij
```

An equivariant network extends this idea by explicitly incorporating geometric quantities.

Conceptually,

```text
m_ij
=
φ(
h_i,
h_j,
d_ij,
Y_l^m(r̂_ij)
)
```

The resulting message contains both

```text
Chemical Information
+
Geometric Information
```

while preserving the appropriate transformation laws.

---

# 24.10.16 Equivariant Tensor Products

Chemical and geometric information cannot always be combined through ordinary addition.

Instead, equivariant tensor products can couple different representation types.

Conceptually,

```text
Chemical Feature
        ×
Geometric Feature
        ↓
Tensor Product
        ↓
Equivariant Feature
```

For example,

```text
Scalar
×
Vector
```

produces a vector-like representation.

More complicated products can combine higher-order representations.

These operations allow the model to construct rich geometric features without destroying equivariance.

---

# 24.10.17 Crystal Diffusion Architecture

A research-grade equivariant crystal diffusion model can therefore be represented as

```text
Crystal Structure
       │
       ├───────────────┐
       │               │
       ▼               ▼
Atomic Species      Coordinates
       │               │
       ▼               ▼
Chemical           Geometric
Embedding          Representation
       │               │
       └───────┬───────┘
               ↓
       Periodic Graph
               ↓
       Equivariant GNN
               ↑
               │
        Time Embedding
               │
               +
       Property Condition
               ↓
       Equivariant Score
               ↓
       Reverse Diffusion
               ↓
       Generated Crystal
```

This architecture combines the concepts developed throughout the chapter.

---

# 24.10.18 Conditioning an Equivariant Diffusion Model

Property conditioning can also be integrated into the equivariant architecture.

Suppose the target property is

```text
Band Gap = 2.0 eV
```

The property is first encoded:

```python
property_embedding = property_encoder(
    target_band_gap
)
```

The resulting representation can be injected into invariant portions of the network.

Conceptually,

```text
Property
   ↓
Property Encoder
   ↓
Condition Embedding
   │
   ├───────────────┐
   │               │
   ↓               ↓
Chemical Features  Time Features
   │               │
   └───────┬───────┘
           ↓
   Equivariant Network
           ↑
           │
      Crystal Geometry
```

The important principle is that the scalar condition itself does not need to transform under rotation.

It can therefore be used to modulate invariant parts of the network while geometric features remain equivariant.

---

# 24.10.19 Equivariant Noise Prediction

During training, noise is added to the crystal representation.

The model then predicts the appropriate denoising quantity.

For coordinates,

```text
X_t
```

may be perturbed.

The model predicts

```text
ε_θ(X_t,t)
```

or the corresponding score.

For an equivariant architecture,

```text
ε_θ(RX_t,t)
=
R ε_θ(X_t,t)
```

for the geometric component.

This ensures that the predicted denoising direction follows the same transformation as the crystal.

---

# 24.10.20 Periodic Equivariance

Ordinary rotational equivariance is not sufficient for crystals.

A crystal also has periodicity.

Consider

```text
r_i
```

and a periodic image

```text
r_i + n_1a_1 + n_2a_2 + n_3a_3
```

where

```text
n_1,n_2,n_3 ∈ Z
```

represent integer translations of the unit cell.

These positions correspond to equivalent periodic images.

Therefore, the crystal model must combine

```text
Spatial Equivariance
+
Periodic Geometry
```

This is one of the major differences between molecular and crystal generative modeling.

---

# 24.10.21 Periodic Neighbor Construction

For each atom, neighboring periodic images can be identified using the lattice.

Conceptually,

```python
for i in atoms:

    for j in atoms:

        for offset in periodic_offsets:

            displacement = (
                positions[j]
                + offset @ lattice
                - positions[i]
            )

            distance = torch.linalg.norm(
                displacement
            )
```

Only neighbors satisfying a cutoff condition are retained.

```python
if distance < cutoff:
    add_edge(i, j, offset)
```

The resulting graph contains both local geometry and periodic information.

---

# 24.10.22 Why Periodic Edges Matter

Consider an atom near the boundary of a unit cell.

Its nearest neighbor may appear on the opposite side of the cell.

Without periodic edges, the model may incorrectly conclude that the atom has no nearby neighbor.

For example,

```text
|------------------|
| A              B |
|------------------|
```

If A is near the left boundary and B is near the right boundary, they may actually be close under periodic boundary conditions.

Therefore,

```text
Unit Cell Boundary
```

should not be interpreted as a physical boundary.

The crystal is continuous.

This makes periodic graph construction essential for accurate crystal generation.

---

# 24.10.23 Equivariance and Lattice Generation

A more advanced model may generate not only atomic coordinates but also the lattice.

The generated crystal becomes

```text
C = (L,X,Z)
```

where

```text
L = lattice
X = coordinates
Z = species
```

The lattice itself transforms under rotation:

```text
L' = RL
```

Therefore, the model's lattice prediction must also respect the appropriate geometric transformation.

This creates a coupled generative problem:

```text
Lattice
   ↕
Coordinates
   ↕
Chemical Identity
```

The model must generate all three consistently.

---

# 24.10.24 Joint Equivariant Crystal Generation

A complete generative process can therefore be represented as

```text
              Crystal
                 │
      ┌──────────┼──────────┐
      ↓          ↓          ↓
   Lattice    Coordinates  Species
      │          │          │
      └──────────┼──────────┘
                 ↓
        Joint Representation
                 ↓
        Equivariant Network
                 ↑
          Time + Condition
                 ↓
          Score Prediction
                 ↓
         Reverse Diffusion
                 ↓
          Generated Crystal
```

The coupling is important because these variables cannot be generated independently.

For example,

```text
Composition
```

influences

```text
Lattice
```

and the lattice influences

```text
Atomic Distances
```

which influence

```text
Chemical Bonding
```

which ultimately influences

```text
Material Properties
```

Therefore, a joint model is scientifically more meaningful than treating each component as an unrelated variable.

---

# 24.10.25 Equivariant Diffusion vs Conventional Diffusion

The difference can be summarized as follows.

| Property                | Conventional Diffusion | Equivariant Crystal Diffusion  |
| ----------------------- | ---------------------- | ------------------------------ |
| Coordinates             | Ordinary vectors       | Geometric representations      |
| Rotation handling       | Learned / augmented    | Built into architecture        |
| Periodicity             | Often external         | Explicitly represented         |
| Directional information | Limited                | Explicit                       |
| Symmetry                | Learned                | Encoded mathematically         |
| Crystal geometry        | Indirect               | Central to model               |
| Physical consistency    | Not guaranteed         | Stronger structural constraint |

The purpose of equivariance is therefore not simply to make the architecture more complicated.

Its purpose is to incorporate known physical structure into the learning problem.

---

# 24.10.26 Conceptual PyTorch Architecture

A simplified research-oriented architecture can be written as

```python
import torch
import torch.nn as nn


class EquivariantCrystalDiffusion(
    nn.Module
):

    def __init__(
        self,
        atomic_dim,
        hidden_dim,
        property_dim,
        time_dim
    ):

        super().__init__()

        self.atomic_encoder = nn.Embedding(
            119,
            atomic_dim
        )

        self.property_encoder = nn.Sequential(
            nn.Linear(
                property_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(
                time_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            3
        )

    def forward(
        self,
        atomic_numbers,
        coordinates,
        time_embedding,
        property_condition
    ):

        atomic_features = (
            self.atomic_encoder(
                atomic_numbers
            )
        )

        property_features = (
            self.property_encoder(
                property_condition
            )
        )

        time_features = (
            self.time_encoder(
                time_embedding
            )
        )

        # A real implementation would
        # replace this simplified operation
        # with periodic equivariant message
        # passing.

        h = atomic_features.mean(
            dim=1
        )

        h = h + property_features
        h = h + time_features

        return self.output_layer(h)
```

This implementation is intentionally simplified.

It demonstrates the software organization:

```text
Atomic Embedding
+
Property Embedding
+
Time Embedding
+
Geometric Model
```

but it is **not itself an actual E(3)-equivariant diffusion network**.

A research implementation must replace the ordinary linear coordinate processing with an appropriate equivariant architecture.

---

# 24.10.27 Using e3nn

A practical implementation can use the `e3nn` ecosystem to construct equivariant operations.

The conceptual workflow is

```text
PyTorch
   ↓
e3nn
   ↓
Irreducible Representations
   ↓
Spherical Harmonics
   ↓
Tensor Products
   ↓
Equivariant Message Passing
   ↓
Crystal Diffusion
```

The advantage of using a specialized equivariant library is that the researcher does not need to manually implement the underlying representation theory.

Instead, the architecture can be constructed from validated mathematical components.

This significantly reduces the possibility of introducing symmetry-breaking operations accidentally.

---

# 24.10.28 Equivariant Diffusion Training Loop

The training process can be summarized as

```python
for crystal in dataloader:

    structure = crystal.structure

    t = sample_time(
        batch_size=len(structure)
    )

    noisy_structure = diffuse(
        structure,
        t
    )

    score = model(
        noisy_structure,
        t
    )

    loss = score_matching_loss(
        score,
        structure,
        noisy_structure,
        t
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

A real implementation must additionally handle

```text
Periodic Graph Construction
+
Equivariant Features
+
Lattice Variables
+
Atomic Species
+
Conditioning
+
Noise Schedule
```

Therefore, the simplified loop should be understood as the computational skeleton rather than a complete production implementation.

---

# 24.10.29 Sampling from the Equivariant Model

Once training is complete, generation begins from a noisy representation.

```python
crystal = initialize_noise(
    batch_size
)

for t in reversed(time_steps):

    score = model(
        crystal,
        t
    )

    crystal = sampler_step(
        crystal,
        score,
        t
    )
```

The crucial difference is that the score network now understands the geometric structure of the crystal.

The resulting trajectory becomes

```text
Noise
  ↓
Equivariant Score
  ↓
Geometric Refinement
  ↓
Chemical Refinement
  ↓
Periodic Structure
  ↓
Candidate Crystal
```

---

# 24.10.30 Why Equivariance Improves Data Efficiency

Suppose the training dataset contains a crystal in one orientation.

A conventional model may need many differently oriented examples to learn that these structures are equivalent.

An equivariant model receives a stronger inductive bias.

It already knows that

```text
Rotated Structure
```

is related to

```text
Original Structure
```

through the transformation rules of the architecture.

Therefore, the model can devote more of its capacity to learning

```text
Chemical Relationships
+
Structural Relationships
+
Property Relationships
```

rather than relearning coordinate-system transformations.

This can improve data efficiency.

---

# 24.10.31 Equivariance and Generalization

Equivariance can also improve generalization.

Suppose the model encounters a crystal orientation that was rare in the training data.

A symmetry-aware architecture can still process it consistently because the transformation rule is built into the model.

Therefore,

```text
Training Distribution
        ↓
Learn Physical Rules
        ↓
Apply to New Orientations
```

is preferable to

```text
Training Distribution
        ↓
Memorize Coordinate Patterns
        ↓
Hope for Generalization
```

This distinction becomes especially important for generated structures because sampling can explore configurations that were not explicitly represented in the training dataset.

---

# 24.10.32 Equivariance Does Not Guarantee Chemical Validity

Equivariance solves an important geometric problem.

It does not automatically solve every materials problem.

A model can be perfectly equivariant and still generate

```text
Unstable Structures
+
Impossible Oxidation States
+
Atomic Overlap
+
Unrealistic Density
+
Unfavorable Formation Energy
```

Therefore,

```text
Equivariance
```

should be viewed as one component of the full generative framework.

The complete system remains

```text
Equivariant Generation
        ↓
Chemical Validation
        ↓
Structural Validation
        ↓
Property Prediction
        ↓
DFT
```

This distinction is critical.

---

# 24.10.33 Equivariance and Property Conditioning

The combination of equivariance and property conditioning creates a particularly powerful framework.

Suppose the researcher specifies

```text
Band Gap = 1.5–2.0 eV
```

The model receives

```text
Crystal Geometry
+
Property Condition
```

and learns

```text
p(C | y)
```

while respecting spatial symmetry.

Conceptually,

```text
Target Property
      ↓
Condition Embedding
      │
      ▼
Equivariant Diffusion Model
      ↑
      │
Crystal Geometry
      ↓
Generated Crystal
```

The model therefore combines two forms of prior knowledge:

```text
Physical Symmetry
+
Scientific Objective
```

This is one of the most promising directions in generative materials discovery.

---

# 24.10.34 Toward Physics-Guided Equivariant Generation

The framework can be extended further.

Instead of conditioning only on a target property,

the model may receive

```text
Target Property
+
Chemical Constraints
+
Energy Information
+
Structural Constraints
```

The architecture becomes

```text
              Target Property
                     │
              Chemical Constraints
                     │
                Energy Model
                     │
                     ↓
              Equivariant
             Diffusion Model
                     ↑
                     │
             Crystal Geometry
                     ↓
              Generated Crystal
```

This creates a hierarchy of constraints.

The generative model must satisfy not only the statistical distribution learned from data but also increasingly strong scientific requirements.

---

# 24.10.35 The Research-Grade Architecture

The complete conceptual framework developed so far can now be summarized as

```text
                    Materials Dataset
                           ↓
                  Crystal Representation
                           ↓
                  Periodic Graph
                           ↓
                Chemical Embeddings
                           +
                Geometric Embeddings
                           ↓
                  Equivariant Network
                           ↑
                ┌──────────┴──────────┐
                │                     │
         Time Embedding        Property Condition
                │                     │
                └──────────┬──────────┘
                           ↓
                     Score Function
                           ↓
                    Reverse Diffusion
                           ↓
                  Candidate Crystal
                           ↓
                 Chemical Validation
                           ↓
                 Structural Validation
                           ↓
                  Property Prediction
                           ↓
                         DFT
```

This architecture represents a major step beyond the simple diffusion models introduced earlier.

The model is no longer simply learning

```text
p(C)
```

or even

```text
p(C|y)
```

in an abstract vector space.

Instead, the goal is to learn a physically structured generative process for

```text
Periodic
+
Chemical
+
Geometric
+
Symmetry-Constrained
```

crystal configurations.

---

# 24.10.36 Key Takeaways

Equivariance provides one of the most important architectural principles for generative materials models.

The key ideas are:

```text
Crystal Structures
        ↓
Three-Dimensional Geometry
        ↓
Symmetry
        ↓
Equivariant Representations
        ↓
Equivariant Score Network
        ↓
Diffusion
        ↓
Generated Crystal
```

The major concepts introduced in this section are

* invariance,
* equivariance,
* E(3),
* SE(3),
* scalar features,
* vector features,
* higher-order representations,
* spherical harmonics,
* radial representations,
* equivariant message passing,
* tensor products,
* periodic equivariance,
* equivariant score functions,
* property-conditioned equivariant diffusion,
* and physics-guided extensions.

The central lesson is:

> **A generative model for crystals should respect the geometry and symmetry of the physical system it is attempting to generate.**

This is not merely an architectural preference.

It is a way of embedding known physical structure into the machine-learning problem.

---

## Transition to Section 24.11

The previous sections developed the mathematical and architectural foundation for modern crystal generation.

We have now moved from

```text
Basic Diffusion
```

to

```text
Score-Based Diffusion
```

and then to

```text
Equivariant Crystal Diffusion
```

The next challenge is to determine whether the structures produced by these models are actually useful.

A generative model may produce thousands of structures.

But how many are

```text
Valid?
Novel?
Diverse?
Stable?
Property-compatible?
```

Therefore, generation must be followed by rigorous evaluation.

The next stage will focus on the scientific evaluation of generated materials:

```text
Generated Crystals
        ↓
Validity
        ↓
Novelty
        ↓
Diversity
        ↓
Property Accuracy
        ↓
Distribution Matching
        ↓
Benchmarking
        ↓
DFT Validation
```

This leads to:

# 24.11 — Evaluation of Generated Materials
# 24.11 Evaluation of Generated Materials

Generating a crystal structure is not the final objective of a generative materials model.

A model may produce thousands or even millions of candidate structures, but the existence of a generated structure does not imply that the structure is

```text
Chemically Valid
        ↓
Structurally Valid
        ↓
Novel
        ↓
Diverse
        ↓
Stable
        ↓
Property-Compatible
        ↓
Experimentally Useful
```

Therefore, generative materials discovery requires a rigorous evaluation framework.

This is one of the most important differences between a demonstration of generative AI and a genuine Materials Informatics research workflow.

A generative model should not be judged only by whether it can produce visually plausible structures.

It should be evaluated according to scientific criteria.

The central question becomes:

> **Does the model generate new materials that are physically meaningful, chemically reasonable, statistically diverse, and useful for the intended materials-design objective?**

A useful evaluation framework therefore considers several dimensions:

```text
Validity
Novelty
Diversity
Property Accuracy
Distribution Matching
Stability
Chemical Plausibility
Structural Plausibility
```

For a research-grade system, these evaluations should be performed before expensive first-principles calculations.

The overall workflow is

```text
Generative Model
      ↓
Generate Candidates
      ↓
Validity Filtering
      ↓
Duplicate Removal
      ↓
Novelty Analysis
      ↓
Diversity Analysis
      ↓
Property Prediction
      ↓
Stability Screening
      ↓
Distribution Comparison
      ↓
DFT Validation
      ↓
Experimental Candidates
```

---

# 24.11.1 Why Evaluation Is Difficult

Generative materials models are more difficult to evaluate than conventional supervised-learning models.

For supervised learning, evaluation often uses a relatively simple procedure.

```text
Training Data
      ↓
Model
      ↓
Test Data
      ↓
Prediction
      ↓
MAE / RMSE / R²
```

A generative model has a fundamentally different objective.

It does not simply predict a known answer.

Instead, it produces new samples from a learned distribution.

Therefore, several questions must be answered simultaneously.

```text
Are the structures valid?

Are they realistic?

Are they new?

Are they sufficiently diverse?

Do they resemble real materials?

Do they satisfy the requested properties?

Are they energetically reasonable?

Can they survive DFT relaxation?
```

No single metric can answer all of these questions.

Consequently, generative materials research requires a **multi-dimensional evaluation framework**.

---

# 24.11.2 The Four Fundamental Evaluation Dimensions

A useful starting point is to divide evaluation into four major categories.

```text
                Generated Materials
                       │
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Validity         Novelty         Diversity
       │               │                │
       └───────────────┼────────────────┘
                       ↓
                Property Accuracy
```

These dimensions answer different questions.

### Validity

Does the generated structure satisfy basic structural and chemical constraints?

### Novelty

Is the generated structure genuinely different from the training data?

### Diversity

Does the model produce many different structures or repeatedly generate nearly identical candidates?

### Property Accuracy

Does the generated structure actually possess the desired properties?

A model can perform well on one dimension and poorly on another.

For example,

```text
High Validity
+
Low Novelty
```

means the model generates realistic structures but mostly reproduces known materials.

Another model may have

```text
High Novelty
+
Low Validity
```

meaning it generates unusual structures but many are chemically or structurally meaningless.

A useful generative model must achieve a reasonable balance.

---

# 24.11.3 Validity

The first evaluation step is validity.

A generated structure should satisfy fundamental requirements before any expensive property calculation is performed.

For a crystal, validity may include

```text
Valid Composition
+
Valid Atomic Coordinates
+
Reasonable Interatomic Distances
+
Valid Periodic Structure
+
Reasonable Density
+
No Severe Atomic Overlap
```

A basic validity pipeline is

```text
Generated Crystal
       ↓
Parse Structure
       ↓
Check Composition
       ↓
Check Coordinates
       ↓
Check Periodicity
       ↓
Check Distances
       ↓
Accept / Reject
```

If a model generates 10,000 structures and 8,500 pass these tests, then the basic validity rate is

```text
Validity Rate = 8500 / 10000
              = 0.85
              = 85%
```

The validity rate is therefore an important first metric.

---

# 24.11.4 Structural Parsing

The first practical test is whether the generated representation can actually be converted into a valid structure object.

For example, with `pymatgen`:

```python
from pymatgen.core import Structure


def check_structure(structure):

    try:

        if not isinstance(
            structure,
            Structure
        ):
            return False

        if len(structure) == 0:
            return False

        return True

    except Exception:

        return False
```

A batch of generated structures can then be evaluated.

```python
valid_structures = []

for structure in generated_structures:

    if check_structure(structure):

        valid_structures.append(
            structure
        )
```

This is only the first level of validity.

A structure may be syntactically valid while still being chemically unreasonable.

---

# 24.11.5 Atomic Overlap

One of the most common failure modes of generated structures is severe atomic overlap.

Suppose two atoms are generated at nearly the same position.

```text
Atom A
   ●
   ●
Atom B
```

Such a structure is generally physically meaningless.

The minimum pairwise distance can therefore be checked.

```python
def minimum_distance(structure):

    distances = structure.distance_matrix

    nonzero = distances[
        distances > 1e-8
    ]

    if len(nonzero) == 0:
        return 0.0

    return nonzero.min()
```

A simple screening rule might be

```python
min_distance = minimum_distance(
    structure
)

if min_distance < 0.7:

    reject = True
```

The exact threshold should **not** be treated as universal.

It depends on

```text
Elements
+
Bonding Environment
+
Pressure
+
Material Class
```

Therefore, a research implementation should use chemically informed criteria rather than blindly applying one arbitrary distance threshold to every material.

---

# 24.11.6 Composition Validity

Generated compositions should also be inspected.

For example, the model may generate

```text
LiFePO4
```

which is chemically interpretable.

But it may also generate a composition containing unexpected or unsupported elemental combinations.

A simple composition inspection can be performed with `pymatgen`.

```python
composition = structure.composition

print(
    composition.reduced_formula
)
```

The reduced formula provides a normalized representation of the composition.

For example,

```text
Li2Fe2P2O8
```

may reduce to

```text
LiFePO4
```

This is useful for composition-level analysis and duplicate detection.

---

# 24.11.7 Oxidation-State Plausibility

Composition alone is insufficient.

A generated material may contain a chemically possible set of elements but require implausible oxidation states.

For example,

```text
Cation Charge
+
Anion Charge
```

must be approximately charge balanced in an ordinary ionic interpretation.

`pymatgen` can assist with oxidation-state assignment.

```python
from pymatgen.analysis.bond_valence import (
    BVAnalyzer
)


analyzer = BVAnalyzer()

try:

    structure_with_oxi = (
        analyzer.get_oxi_state_decorated_structure(
            structure
        )
    )

    valid_oxidation_states = True

except Exception:

    valid_oxidation_states = False
```

This provides a useful screening mechanism.

However, oxidation-state assignment should not be interpreted as absolute proof of chemical stability.

Many real materials exhibit

```text
Mixed Valence
+
Covalent Bonding
+
Delocalized Electrons
+
Unusual Coordination
```

Therefore, oxidation-state analysis is best treated as a **chemical plausibility filter**.

---

# 24.11.8 Coordination Environment

The generated crystal should also have reasonable local coordination.

For each atom,

```text
Central Atom
     │
 ┌───┼───┐
 ↓   ↓   ↓
Neighboring Atoms
```

the model should produce a physically meaningful local environment.

Coordination analysis can help detect

```text
Impossible Coordination
+
Severe Undercoordination
+
Unrealistic Bonding
```

For example, a generated structure containing a highly coordinated hydrogen atom should immediately attract attention.

The evaluation should therefore consider element-specific chemical behavior.

---

# 24.11.9 Density Screening

Density provides another inexpensive structural filter.

The density is

```text
ρ = m / V
```

where

```text
m = mass
V = volume
```

For a generated crystal, `pymatgen` can calculate the density.

```python
density = structure.density

print(
    density
)
```

Extremely unusual densities may indicate

```text
Incorrect Lattice
+
Atomic Overlap
+
Unphysical Cell Volume
```

Density alone cannot determine stability, but it is a useful early screening feature.

---

# 24.11.10 Validity Rate

For a generated set containing

```text
N_total
```

structures, let

```text
N_valid
```

be the number passing the validity filters.

Then

```text
Validity Rate = N_valid / N_total
```

For example,

```text
Generated = 10,000

Valid = 8,200
```

gives

```text
Validity Rate = 0.82
```

or

```text
82%
```

This metric should always be reported when comparing generative models.

A model that generates only 10 valid structures from 10,000 attempts should not be considered equivalent to a model that generates 9,000 valid structures.

---

# 24.11.11 Validity Should Be Hierarchical

Validity should not be treated as a single binary test.

A more useful framework is hierarchical.

```text
Level 1
Representation Validity

        ↓

Level 2
Geometric Validity

        ↓

Level 3
Chemical Validity

        ↓

Level 4
Energetic Plausibility

        ↓

Level 5
DFT Stability
```

For example:

### Level 1 — Representation

Can the structure be parsed?

### Level 2 — Geometry

Are atoms reasonably separated?

### Level 3 — Chemistry

Are composition and coordination plausible?

### Level 4 — Energy

Is the predicted formation energy reasonable?

### Level 5 — First-Principles Validation

Does the structure remain meaningful after DFT relaxation?

This hierarchy prevents inexpensive and expensive tests from being mixed together.

---

# 24.11.12 Novelty

Validity asks:

```text
Is the generated structure reasonable?
```

Novelty asks:

```text
Is it actually new?
```

This distinction is crucial.

A generative model could achieve 100% validity simply by memorizing the training dataset.

Such a model would have limited value for materials discovery.

Therefore, generated candidates must be compared against the training set.

---

# 24.11.13 Composition-Level Novelty

The simplest novelty measure compares compositions.

Suppose the training dataset contains

```text
LiFePO4
NaFePO4
LiCoO2
```

and the generator produces

```text
LiMnPO4
NaCoO2
LiFePO4
```

The first two compositions are new relative to the training set.

The third is not.

A simple implementation is

```python
training_formulas = set(
    structure.composition.reduced_formula
    for structure in training_structures
)


novel_structures = []

for structure in generated_structures:

    formula = (
        structure
        .composition
        .reduced_formula
    )

    if formula not in training_formulas:

        novel_structures.append(
            structure
        )
```

This provides **composition-level novelty**.

However, it is not sufficient.

---

# 24.11.14 Structural Novelty

Two structures can have the same composition while having completely different crystal structures.

For example,

```text
Composition
   ↓
AB
```

may occur in multiple structure types.

Therefore,

```text
Same Composition
≠
Same Crystal Structure
```

A stronger novelty evaluation must compare

```text
Composition
+
Lattice
+
Atomic Arrangement
```

Possible representations include

```text
Crystal Fingerprints
SOAP
Graph Embeddings
Structure Descriptors
Local Environment Descriptors
```

The generated structure can then be compared against the nearest training structures.

---

# 24.11.15 Nearest-Neighbor Novelty

Suppose a structure is represented by a feature vector

```text
z
```

and the training dataset contains

```text
z₁,z₂,...,z_N
```

The nearest-neighbor distance can be defined as

```text
d_min = min_i d(z,z_i)
```

A large

```text
d_min
```

indicates that the generated structure is far from the training examples in the selected representation space.

However, large distance does not automatically mean scientific novelty.

It may also indicate an invalid structure.

Therefore,

```text
Novelty
```

must always be interpreted together with

```text
Validity
```

---

# 24.11.16 Novelty–Validity Trade-Off

A useful conceptual plot is

```text
Validity
   ↑
   │        ●
   │      ●
   │    ●
   │
   │  ●
   │ ●
   └────────────────→
        Novelty
```

A good generator should ideally produce structures that are both

```text
Highly Valid
+
Highly Novel
```

A generator producing highly novel but invalid structures is not useful.

Likewise, a generator producing highly valid but completely memorized structures provides limited discovery capability.

---

# 24.11.17 Diversity

Novelty measures whether structures differ from the training set.

Diversity measures whether generated structures differ from **each other**.

This distinction is important.

Suppose a model generates

```text
10,000 structures
```

but 9,800 are nearly identical.

The model technically generates many structures, but the actual design-space coverage is poor.

This phenomenon is related to

```text
Mode Collapse
```

or excessive concentration of the generated distribution.

---

# 24.11.18 Composition Diversity

The simplest diversity measure counts unique compositions.

```python
unique_formulas = set(
    structure
    .composition
    .reduced_formula
    for structure in generated_structures
)

print(
    len(unique_formulas)
)
```

If

```text
10,000
```

structures produce only

```text
100
```

unique compositions, the composition-level diversity is relatively low.

A simple ratio is

```text
Composition Diversity
=
Number of Unique Compositions
/
Number of Generated Structures
```

This is easy to calculate but does not capture structural diversity within the same composition.

---

# 24.11.19 Structural Diversity

A stronger approach is to represent each crystal using a structural fingerprint.

Let the fingerprints be

```text
z₁,z₂,...,z_N
```

Then pairwise distances can be calculated.

```python
from sklearn.metrics import pairwise_distances

distance_matrix = pairwise_distances(
    embeddings
)
```

A larger average pairwise distance indicates greater structural diversity in the selected representation space.

For large datasets, calculating all pairwise distances can be computationally expensive.

Therefore, approximate methods or random subsets may be used.

---

# 24.11.20 Diversity in Latent Space

If the generative model has a latent representation,

```text
z
```

generated materials can be projected into that space.

For example,

```text
Crystal
   ↓
Encoder
   ↓
Latent Vector
   ↓
PCA / UMAP
   ↓
Materials Space
```

A diverse generator should cover a meaningful region of the learned materials space.

Visualization can reveal

```text
Clusters
+
Gaps
+
Mode Collapse
+
Outliers
```

However, dimensionality-reduction plots should be interpreted carefully because the visualization itself can distort distances.

---

# 24.11.21 Property Diversity

Structural diversity is not the only relevant form of diversity.

Generated materials should ideally explore different regions of property space.

Suppose the generated band gaps are

```text
0.5
0.6
0.7
0.7
0.8
...
```

The model may be structurally diverse but property-space diversity may be low.

A useful evaluation therefore examines

```text
Structure Space
+
Composition Space
+
Property Space
```

simultaneously.

---

# 24.11.22 Property Accuracy

For property-conditioned generation, property accuracy is one of the most important metrics.

Suppose the target is

```text
Band Gap = 2.0 eV
```

and the model generates structures with predicted values

```text
1.95
2.02
2.08
1.87
2.14
```

The generated properties should be compared against the target.

The error can be defined as

```text
Error = |y_generated - y_target|
```

For multiple generated structures, the mean absolute target error is

```text
MAE_target
=
(1/N)
Σ |y_i - y_target|
```

A lower value indicates stronger property targeting.

---

# 24.11.23 Property Success Rate

Often the researcher cares about a design window rather than an exact value.

Suppose the target is

```text
1.5 eV ≤ Eg ≤ 2.0 eV
```

Define

```python
mask = (
    (band_gaps >= 1.5)
    &
    (band_gaps <= 2.0)
)

success_rate = mask.mean()
```

If 4,000 out of 10,000 candidates satisfy the target,

```text
Success Rate = 40%
```

This can be more scientifically meaningful than MAE when the objective is a practical design constraint.

---

# 24.11.24 Multi-Property Accuracy

Real materials design often involves multiple objectives.

Suppose the desired constraints are

```text
1.5 ≤ Band Gap ≤ 2.0

Formation Energy < -2.0 eV/atom

Bulk Modulus > 150 GPa
```

A candidate is successful only if all constraints are satisfied.

```python
mask = (
    (band_gap >= 1.5)
    &
    (band_gap <= 2.0)
    &
    (formation_energy < -2.0)
    &
    (bulk_modulus > 150)
)
```

The fraction satisfying all conditions becomes the multi-objective success rate.

This metric is particularly useful for practical materials discovery.

---

# 24.11.25 Property Predictor Error

Property-conditioned evaluation depends strongly on the property predictor.

Suppose a machine-learning model predicts the band gap of generated structures.

The prediction itself contains uncertainty.

Therefore,

```text
Generated Property
```

should not automatically be treated as

```text
True Property
```

Instead,

```text
Generated Crystal
      ↓
Property Predictor
      ↓
Predicted Property
      ↓
DFT / Experiment
      ↓
Validated Property
```

The predictor should itself be evaluated on an independent test set.

Metrics may include

```text
MAE
RMSE
R²
Calibration
Uncertainty
```

A weak property predictor can make an apparently successful generator look better than it actually is.

---

# 24.11.26 Distribution Matching

A generative model should reproduce important characteristics of the real materials distribution.

Suppose the training dataset has distributions for

```text
Density
Band Gap
Formation Energy
Lattice Parameters
Composition
Number of Atoms
```

The generated dataset should be compared with these distributions.

For example,

```text
Training Distribution
        ↓
██████████████

Generated Distribution
        ↓
████████████
```

If the two distributions are dramatically different, the generator may have learned an unrealistic distribution.

---

# 24.11.27 Property Distribution Comparison

Suppose the training band-gap distribution is

```text
Eg
↑
│        ███
│      ███████
│    ███████████
│  █████████████
└────────────────→
```

The generated distribution can be plotted alongside it.

A useful comparison can reveal

```text
Mean Shift
Variance Change
Missing Regions
Excessive Concentration
Unexpected Outliers
```

These differences provide clues about the behavior of the generative model.

---

# 24.11.28 Statistical Distribution Metrics

Several statistical measures can compare generated and reference distributions.

For one-dimensional distributions, examples include

```text
Mean
Variance
Median
Quantiles
Kolmogorov–Smirnov Distance
Wasserstein Distance
```

For more complex distributions, one may compare

```text
Embedding Distributions
Kernel Distances
Maximum Mean Discrepancy
```

The appropriate metric depends on the representation and scientific question.

---

# 24.11.29 Wasserstein Distance

The Wasserstein distance provides an intuitive measure of how much one distribution must be transported to become another.

Conceptually,

```text
Training Distribution
        ↓
     Transport
        ↓
Generated Distribution
```

A small Wasserstein distance indicates that the distributions are relatively similar.

However, distribution matching should not become the only objective.

A perfect generator that simply reproduces the training distribution may have excellent distribution similarity while generating little novelty.

Therefore, evaluation should balance

```text
Distribution Matching
+
Novelty
+
Diversity
```

---

# 24.11.30 Precision and Recall for Generative Materials

Generative-model evaluation can also be interpreted through precision and recall.

Conceptually,

### Precision

Among generated structures, how many are close to the valid materials distribution?

### Recall

How much of the valid materials distribution does the generator cover?

This produces a useful conceptual distinction.

```text
High Precision
```

means the generator produces realistic structures.

```text
High Recall
```

means it explores a broad region of the valid materials space.

An ideal generator seeks both.

---

# 24.11.31 Benchmarking Against Known Materials

Generated materials should be compared against known materials databases where appropriate.

For example,

```text
Generated Materials
        ↓
Compare with Reference Database
        ↓
Known?
Novel?
Similar?
```

Potential reference datasets include curated computational materials databases.

The purpose is not simply to determine whether a structure already exists.

It is also useful to determine whether generated candidates occupy

```text
Known Chemical Space
```

or extend into

```text
Underexplored Chemical Space
```

---

# 24.11.32 Database Matching

A generated structure can be compared using

```text
Composition
+
Lattice
+
Structure Fingerprint
+
Atomic Environment
```

A practical workflow is

```text
Generated Crystal
      ↓
Canonical Structure
      ↓
Fingerprint
      ↓
Database Search
      ↓
Nearest Known Materials
```

The nearest known structure can provide valuable scientific context.

For example,

```text
Generated Candidate
        ↓
Nearest Known Structure
        ↓
Structural Similarity
        ↓
Property Comparison
```

This can help determine whether the candidate represents a genuinely new region of materials space or a small modification of an existing structure.

---

# 24.11.33 Stability Evaluation

Validity does not imply stability.

A generated structure can be

```text
Geometrically Valid
+
Chemically Plausible
```

while still being energetically unfavorable.

Therefore, stability evaluation is necessary.

Important quantities may include

```text
Formation Energy
Energy Above Hull
Decomposition Energy
Relaxation Energy
Phonon Stability
```

The exact hierarchy depends on the research problem.

---

# 24.11.34 Formation Energy Screening

A machine-learning model can provide an inexpensive first estimate.

```python
formation_energy = (
    formation_energy_model(
        generated_structures
    )
)
```

The candidates can then be ranked.

```python
ranking = torch.argsort(
    formation_energy
)
```

Only the most promising structures need to proceed to expensive calculations.

This produces a hierarchical screening pipeline:

```text
10,000 Generated
        ↓
ML Formation Energy
        ↓
1,000 Candidates
        ↓
DFT
        ↓
100 Candidates
```

The numbers are illustrative.

The actual reduction depends on the research problem.

---

# 24.11.35 Energy Above the Convex Hull

For thermodynamic stability, formation energy alone may not be sufficient.

A structure can have a favorable formation energy but still decompose into other compounds.

The energy above the convex hull provides a more meaningful stability measure.

Conceptually,

```text
Energy
  ↑
  │       ● metastable
  │
  │────────────── Hull
  │
  └────────────────→ Composition
```

The distance from the convex hull represents the thermodynamic driving force toward decomposition under the assumptions of the computational framework.

Therefore,

```text
Formation Energy
```

and

```text
Energy Above Hull
```

should not be treated as interchangeable quantities.

---

# 24.11.36 DFT-Based Validation

Machine-learning predictions are useful for screening, but high-value candidates should eventually be evaluated using first-principles calculations.

The workflow becomes

```text
Generated Crystal
       ↓
ML Screening
       ↓
Selected Candidates
       ↓
DFT Relaxation
       ↓
Electronic Structure
       ↓
Energy / Stability
       ↓
Final Candidate
```

For example, a candidate may be passed to a DFT workflow using the researcher's established Quantum ESPRESSO pipeline.

The important principle is:

> **Generative AI proposes candidates; first-principles calculations provide higher-fidelity scientific validation.**

---

# 24.11.37 DFT Relaxation Success

One particularly useful metric is the fraction of generated structures that successfully undergo structural relaxation.

Suppose

```text
1,000 candidates
```

are sent to DFT.

If

```text
850
```

successfully converge to physically meaningful relaxed structures, the relaxation success rate is

```text
85%
```

This provides a much stronger evaluation than simply checking whether the generated coordinates can be parsed.

---

# 24.11.38 Structural Change After Relaxation

The generated structure and relaxed structure can be compared.

Define

```text
Generated Structure
        ↓
DFT Relaxation
        ↓
Relaxed Structure
```

If the structure changes dramatically, the generator may be producing configurations far from physically stable minima.

Useful quantities include changes in

```text
Lattice Parameters
Bond Lengths
Angles
Volume
Atomic Positions
```

A large relaxation displacement can therefore provide evidence that the original generated structure was physically implausible.

---

# 24.11.39 Generation-to-Relaxation Distance

For a generated structure X and its relaxed structure X_relaxed, one can define a structural difference metric

```text
D(X, X_relaxed)
```

using an appropriate structural representation.

A small value suggests that the generated structure was already near a stable configuration.

A large value suggests substantial structural correction during relaxation.

This metric can therefore be useful when evaluating the physical quality of a generative model.

---

# 24.11.40 Failure Analysis

A research-grade generative-materials study should not report only successful candidates.

Failures should also be analyzed.

Possible failure categories include

```text
Invalid Structure
Atomic Overlap
Unreasonable Density
Chemical Imbalance
Unstable Composition
Duplicate
Low Novelty
Low Diversity
Poor Property Accuracy
DFT Failure
Large Relaxation
```

A useful failure-analysis table is

| Failure Type        | Number | Fraction |
| ------------------- | -----: | -------: |
| Parsing failure     |    ... |      ... |
| Atomic overlap      |    ... |      ... |
| Chemical invalidity |    ... |      ... |
| Duplicate           |    ... |      ... |
| Property failure    |    ... |      ... |
| DFT failure         |    ... |      ... |

This makes the generative pipeline scientifically transparent.

---

# 24.11.41 Evaluation Dashboard

For a large generation experiment, the results can be summarized using a compact evaluation table.

| Metric                   |  Result |
| ------------------------ | ------: |
| Generated structures     | 100,000 |
| Valid structures         |     ... |
| Validity rate            |     ... |
| Unique compositions      |     ... |
| Novel structures         |     ... |
| Novelty rate             |     ... |
| Diversity score          |     ... |
| Property success rate    |     ... |
| ML-stable candidates     |     ... |
| DFT-relaxed candidates   |     ... |
| DFT-confirmed candidates |     ... |

This provides a much clearer picture than reporting only one success metric.

---

# 24.11.42 A Complete Evaluation Pipeline

The complete evaluation workflow can now be represented as

```text
                    Generated Crystals
                           │
                           ▼
                  Representation Check
                           │
                           ▼
                  Geometric Validation
                           │
                           ▼
                   Chemical Validation
                           │
                           ▼
                    Duplicate Removal
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
          Novelty                    Diversity
              │                         │
              └────────────┬────────────┘
                           ↓
                   Property Prediction
                           ↓
                   Target Filtering
                           ↓
                  Stability Screening
                           ↓
                  Distribution Analysis
                           ↓
                    Database Matching
                           ↓
                     DFT Validation
                           ↓
                Experimental Candidates
```

This transforms generation from a single neural-network operation into a complete scientific workflow.

---

# 24.11.43 Evaluating Conditional Generation

Conditional generation requires an additional question.

Suppose the model was conditioned on

```text
Band Gap = 2.0 eV
```

A successful evaluation must determine whether the generated materials actually satisfy that condition.

Therefore, evaluation should compare

```text
Target Property
       ↕
Generated Property
```

For each generated candidate,

```python
target = 2.0

error = abs(
    predicted_band_gap - target
)
```

The distribution of these errors can be analyzed.

```text
Target
  │
  │     ●
  │   ● ● ●
  │ ● ● ● ●
  │
  └────────────────→
       Candidates
```

A well-conditioned model should concentrate candidates around the desired target while maintaining meaningful structural diversity.

---

# 24.11.44 Conditional Generation Success Matrix

For multiple properties, evaluation can be organized into a matrix.

| Candidate | Band Gap | Formation Energy | Bulk Modulus | Valid | Stable |
| --------- | -------: | ---------------: | -----------: | ----- | ------ |
| A         |      1.8 |             -3.1 |          220 | Yes   | Yes    |
| B         |      2.4 |             -3.0 |          190 | Yes   | Yes    |
| C         |      1.7 |             -1.2 |          250 | Yes   | No     |
| D         |      1.9 |             -3.4 |          230 | Yes   | Yes    |

Candidate D may be preferred because it satisfies multiple objectives simultaneously.

This table is often more scientifically informative than a single aggregate metric.

---

# 24.11.45 Diversity–Quality Trade-Off

One of the central challenges in generative modeling is balancing

```text
Quality
```

and

```text
Diversity
```

A model that generates only highly probable structures may produce excellent validity but poor exploration.

A model that explores aggressively may discover unusual candidates but produce more invalid structures.

Conceptually,

```text
Quality
  ↑
  │ ●
  │   ●
  │     ●
  │       ●
  │         ●
  └────────────────→
          Diversity
```

The ideal operating region depends on the scientific objective.

For discovery, some degree of exploration is essential.

For targeted optimization, stronger exploitation may be preferred.

---

# 24.11.46 Guidance Strength and Evaluation

The guidance strength introduced earlier directly affects this trade-off.

For classifier-free guidance,

```text
guided
=
unconditional
+
w(
conditional
−
unconditional
)
```

where `w` controls the conditioning strength.

Small values may produce

```text
Higher Diversity
+
Lower Target Accuracy
```

while large values may produce

```text
Higher Target Accuracy
+
Lower Diversity
```

Therefore, a generative-materials experiment should evaluate several guidance strengths rather than selecting one value arbitrarily.

For example,

```text
w = 0.5
w = 1.0
w = 2.0
w = 4.0
w = 6.0
```

can be compared using

```text
Validity
+
Novelty
+
Diversity
+
Property Success
```

---

# 24.11.47 Pareto Evaluation of Generative Models

The generator itself can be evaluated using multiple objectives.

For example,

```text
Objective 1 → Validity
Objective 2 → Novelty
Objective 3 → Diversity
Objective 4 → Property Success
```

A model configuration that improves one objective while damaging another may be Pareto-dominated.

Therefore, rather than selecting a model based on one metric, researchers can identify a Pareto-efficient set of model configurations.

This is particularly useful when comparing

```text
Guidance Strength
Noise Schedule
Model Size
Sampling Steps
Conditioning Strategy
```

---

# 24.11.48 Reproducibility

Generative experiments must also be reproducible.

The following should be recorded:

```text
Dataset Version
Random Seed
Model Architecture
Training Epochs
Optimizer
Learning Rate
Batch Size
Noise Schedule
Sampling Algorithm
Sampling Steps
Guidance Strength
Property Predictor Version
Filtering Thresholds
DFT Settings
```

A minimal configuration object might be stored as

```python
config = {

    "seed": 42,

    "batch_size": 64,

    "learning_rate": 1e-4,

    "diffusion_steps": 1000,

    "sampling_steps": 100,

    "guidance_scale": 2.0,

    "cutoff": 5.0

}
```

Without such information, reproducing a generative materials experiment can become extremely difficult.

---

# 24.11.49 Recommended Research Protocol

A robust experiment can follow the following sequence.

```text
1. Define Scientific Objective
          ↓
2. Prepare Reference Dataset
          ↓
3. Train Generative Model
          ↓
4. Generate Large Candidate Set
          ↓
5. Measure Basic Validity
          ↓
6. Remove Duplicates
          ↓
7. Measure Novelty
          ↓
8. Measure Diversity
          ↓
9. Predict Target Properties
          ↓
10. Apply Design Constraints
          ↓
11. Screen Thermodynamic Stability
          ↓
12. Compare with Known Materials
          ↓
13. Select Top Candidates
          ↓
14. Perform DFT Validation
          ↓
15. Analyze Failures
          ↓
16. Report All Metrics
```

This procedure ensures that the generative model is evaluated as a scientific discovery tool rather than simply as a neural-network demonstration.

---

# 24.11.50 Research-Grade Evaluation Philosophy

The central philosophy can be summarized as

```text
Generation ≠ Discovery
```

Generation produces candidate structures.

Discovery requires determining whether those candidates are scientifically meaningful.

Therefore,

```text
Generative AI
      ↓
Candidate Generation
      ↓
Scientific Evaluation
      ↓
Physical Validation
      ↓
Materials Discovery
```

The final objective is not to maximize the number of generated crystals.

It is to maximize the number of **useful, valid, novel, diverse, property-compatible, and physically credible candidates**.

This distinction should guide the entire evaluation strategy.

---

# 24.11.51 Summary

The evaluation of generated materials requires substantially more than checking whether a neural network can produce crystal-like coordinates.

A rigorous evaluation framework should examine:

```text
Validity
+
Novelty
+
Diversity
+
Property Accuracy
+
Distribution Matching
+
Chemical Plausibility
+
Thermodynamic Stability
+
DFT Compatibility
```

Validity determines whether the generated structures are structurally and chemically meaningful.

Novelty determines whether the model goes beyond simply reproducing its training data.

Diversity determines whether the generator explores multiple regions of materials space.

Property accuracy determines whether conditional generation actually satisfies the requested scientific objective.

Distribution matching determines whether the generated dataset remains connected to realistic materials space.

Stability analysis determines whether apparently valid structures are energetically meaningful.

Finally, DFT provides a higher-fidelity validation stage for the most promising candidates.

The complete principle is therefore:

```text
Generate
   ↓
Validate
   ↓
Compare
   ↓
Filter
   ↓
Predict
   ↓
Rank
   ↓
DFT
   ↓
Discover
```

A generative materials model should ultimately be judged not by how impressive its generated structures appear, but by whether it can reliably produce **scientifically useful candidates that survive increasingly rigorous validation**.

The next stage is therefore to move from evaluation metrics toward the practical integration of the complete generative pipeline with Materials Informatics workflows, including candidate ranking, active learning, DFT feedback, and closed-loop materials discovery.

# 24.12 Closed-Loop Generative Materials Discovery

The previous sections established how generative models can create candidate crystal structures and how those candidates can be evaluated for validity, novelty, diversity, property accuracy, and stability.

However, a complete Materials Informatics discovery system should not stop after generating and evaluating one batch of structures.

A more powerful framework is a **closed-loop discovery system**.

Instead of

```text
Dataset
   ↓
Train Model
   ↓
Generate Materials
   ↓
Stop
```

the system repeatedly learns from its own discoveries.

The overall concept becomes

```text
Existing Materials Data
        ↓
Generative Model
        ↓
Candidate Materials
        ↓
ML Screening
        ↓
DFT Validation
        ↓
New High-Quality Data
        ↓
Retrain / Update Models
        ↓
Generate Better Candidates
        ↺
```

This creates a feedback loop between

```text
Generation
+
Prediction
+
Simulation
+
Learning
```

The ultimate objective is therefore not merely to build a generative model.

The objective is to build a system capable of **iteratively exploring materials space and improving its knowledge as new information becomes available**.

---

# 24.12.1 From Generative Modeling to Materials Discovery

A generative model answers:

> What structures can be generated from the learned distribution?

A materials discovery system asks a more ambitious question:

> Which generated structures should be investigated next?

This distinction is fundamental.

Suppose a model generates

```text
100,000
```

candidate crystals.

It is computationally impossible to perform expensive DFT calculations on every candidate in many realistic research settings.

Therefore, the workflow must decide which candidates deserve further evaluation.

The process becomes

```text
100,000 Generated
       ↓
ML Filtering
       ↓
10,000 Candidates
       ↓
Stability Filtering
       ↓
1,000 Candidates
       ↓
DFT
       ↓
100 Candidates
       ↓
Detailed Analysis
       ↓
10–20 High-Value Candidates
```

This is the role of **candidate selection**.

---

# 24.12.2 Candidate Ranking

After generation, each candidate can be represented by a collection of predicted properties.

For example:

| Candidate | Band Gap | Formation Energy | Bulk Modulus | Density |
| --------- | -------: | ---------------: | -----------: | ------: |
| A         |     1.72 |             -3.1 |          215 |     5.2 |
| B         |     2.41 |             -2.8 |          240 |     4.8 |
| C         |     1.61 |             -3.4 |          235 |     5.5 |
| D         |     1.84 |             -2.1 |          180 |     4.9 |

Suppose the objective is

```text
1.5 ≤ Eg ≤ 2.0 eV

Low Formation Energy

High Bulk Modulus
```

Candidate C may therefore receive a high ranking.

A simple scoring function can be constructed after normalization.

```python
score = (
    w_gap * gap_score
    +
    w_energy * energy_score
    +
    w_bulk * bulk_score
)
```

The important principle is that the individual properties should generally be transformed into comparable scales before combining them.

---

# 24.12.3 From Hard Filtering to Ranking

A binary filter produces

```text
Accept
Reject
```

For example,

```python
if band_gap >= 1.5 and band_gap <= 2.0:
    accept = True
```

This can be useful, but it discards information.

Suppose two materials have

```text
Candidate A → Eg = 1.51 eV
Candidate B → Eg = 1.95 eV
```

Both satisfy the same constraint.

However, depending on the scientific objective, one may be preferable.

A ranking-based approach preserves more information.

```text
Target = 1.75 eV

Candidate A
Error = |1.51 - 1.75|

Candidate B
Error = |1.95 - 1.75|
```

The candidates can then be ranked according to their distance from the desired target.

---

# 24.12.4 Multi-Objective Candidate Ranking

Materials design rarely has only one objective.

Consider a photovoltaic material.

The researcher may desire

```text
Band Gap ≈ 1.5 eV
Low Formation Energy
High Absorption
Low Toxicity
High Structural Stability
```

These objectives may conflict.

For example,

```text
Higher Stability
      ↕
Electronic Performance
```

may not always improve simultaneously.

Therefore, a single scalar score may oversimplify the problem.

A better framework is multi-objective optimization.

```text
Candidate Space
       ↓
Evaluate Multiple Objectives
       ↓
Pareto Analysis
       ↓
Select Non-Dominated Candidates
```

---

# 24.12.5 Pareto-Based Candidate Selection

Suppose two properties are being optimized:

```text
Formation Energy
Bulk Modulus
```

with objectives

```text
Lower Formation Energy
Higher Bulk Modulus
```

Candidate A dominates candidate B if A is at least as good as B in every objective and strictly better in at least one.

The set of candidates that are not dominated forms the Pareto frontier.

```text
Bulk Modulus
     ↑
     │                 ●
     │             ●
     │          ●
     │       ●
     │    ●
     │ ●
     └────────────────────→
       Stability
```

Rather than choosing one arbitrary candidate, the researcher can investigate several Pareto-optimal candidates.

This is especially useful when the final scientific objective contains unavoidable trade-offs.

---

# 24.12.6 Uncertainty-Aware Candidate Selection

Predicted properties are not exact.

Suppose a model predicts

```text
Band Gap = 1.82 eV
```

but its uncertainty is large.

Another candidate may have

```text
Band Gap = 1.79 eV
```

with much lower uncertainty.

The second candidate may be scientifically safer.

Therefore, a discovery system should ideally consider

```text
Prediction
+
Uncertainty
```

rather than prediction alone.

A predictive model may provide

```text
ŷ ± σ
```

where

```text
ŷ
```

is the predicted property and

```text
σ
```

represents predictive uncertainty.

---

# 24.12.7 Why Uncertainty Matters

Suppose the target is

```text
Eg = 1.8 eV
```

and two candidates are predicted as

```text
Candidate A:
1.79 ± 0.03 eV

Candidate B:
1.81 ± 0.40 eV
```

Both have similar predicted values.

However, Candidate B is much more uncertain.

This means that the system has less confidence in the prediction.

Such candidates can be valuable for exploration because they may contain information that the current model does not understand well.

This leads directly to **active learning**.

---

# 24.12.8 Active Learning

Active learning is a strategy in which the machine-learning system chooses which new data points should be labeled or evaluated next.

In Materials Informatics, labeling may mean

```text
DFT Calculation
```

or

```text
Experimental Measurement
```

The basic workflow is

```text
Initial Dataset
      ↓
Train ML Model
      ↓
Predict Unlabeled Materials
      ↓
Estimate Uncertainty
      ↓
Select Informative Candidates
      ↓
DFT / Experiment
      ↓
Add New Data
      ↓
Retrain Model
      ↺
```

This is especially valuable because high-fidelity materials calculations can be expensive.

---

# 24.12.9 Active Learning with Generative Models

Generative models and active learning can be combined.

The generator proposes structures.

The active-learning system decides which structures should be evaluated.

```text
Generative Model
       ↓
Millions of Candidates
       ↓
Active Learning
       ↓
Select Informative Candidates
       ↓
DFT
       ↓
New Training Data
       ↓
Update Models
```

This creates a powerful division of labor.

The generator explores.

The active-learning algorithm prioritizes.

DFT provides high-quality labels.

The updated model learns from the new information.

---

# 24.12.10 Exploration vs Exploitation

Active learning must balance two competing goals.

### Exploitation

Investigate materials that are already predicted to be highly promising.

### Exploration

Investigate materials about which the model is uncertain.

These can be represented as

```text
Exploitation
    ↓
Find the Best Known Region
```

and

```text
Exploration
    ↓
Discover Unknown Regions
```

A useful discovery system requires both.

If the system only exploits known regions, it may become trapped near already understood materials.

If it only explores uncertain regions, it may spend resources on poor candidates.

---

# 24.12.11 Acquisition Functions

An acquisition function determines which candidate should be evaluated next.

Conceptually,

```text
Candidate
   ↓
Predicted Mean
+
Predicted Uncertainty
   ↓
Acquisition Score
```

A simple uncertainty-based acquisition function is

```python
acquisition_score = uncertainty
```

which prioritizes candidates about which the model is least certain.

A simple exploitation-based score is

```python
acquisition_score = predicted_property
```

for a property where higher values are desirable.

A combined strategy can be

```python
acquisition_score = (
    predicted_property
    +
    beta * uncertainty
)
```

where `beta` controls the exploration strength.

This is a simplified concept, but it illustrates the central idea.

---

# 24.12.12 Generative Active Learning Loop

A practical generative active-learning loop can be implemented conceptually as

```python
for iteration in range(
    num_iterations
):

    candidates = generate_materials(
        generator,
        n_samples
    )

    predictions, uncertainty = (
        predictor.predict_with_uncertainty(
            candidates
        )
    )

    scores = (
        predictions
        +
        beta * uncertainty
    )

    selected = select_top_candidates(
        candidates,
        scores,
        budget
    )

    dft_results = run_dft(
        selected
    )

    dataset.add(
        selected,
        dft_results
    )

    predictor = retrain_predictor(
        dataset
    )
```

This creates a genuine feedback loop.

---

# 24.12.13 The Role of DFT in the Loop

DFT acts as a high-fidelity oracle.

The machine-learning models are inexpensive but approximate.

DFT is expensive but generally provides substantially higher-fidelity information for the modeled physical system.

Therefore,

```text
ML
↓
Fast Approximate Screening

DFT
↓
Expensive High-Fidelity Evaluation
```

The discovery framework uses ML to reduce the number of DFT calculations required.

Instead of

```text
100,000 candidates
       ↓
100,000 DFT calculations
```

the workflow attempts to achieve

```text
100,000 generated
       ↓
ML screening
       ↓
1,000 selected
       ↓
1,000 DFT calculations
```

The exact reduction depends on the scientific problem.

---

# 24.12.14 DFT Feedback

Once DFT calculations are completed, their results should not simply be stored and forgotten.

They can be used to improve the models.

For example:

```text
Generated Structure
        ↓
ML Prediction
        ↓
DFT Result
        ↓
Prediction Error
```

If the model predicted

```text
Formation Energy = -2.8 eV/atom
```

but DFT produced

```text
Formation Energy = -2.4 eV/atom
```

the discrepancy becomes new information.

The updated dataset therefore contains

```text
Structure
+
ML Prediction
+
DFT Ground Truth
```

which can improve future predictions.

---

# 24.12.15 Correcting Model Bias

Suppose the original dataset contains mostly

```text
Oxides
```

and relatively few

```text
Sulfides
```

The generative model may learn a strong bias toward oxides.

If the model begins generating large numbers of oxide structures, an active-learning system may deliberately select promising sulfide candidates for DFT.

The new DFT data can then reduce the imbalance.

This creates a feedback mechanism for correcting weaknesses in the training distribution.

---

# 24.12.16 Closed-Loop Materials Informatics

The complete system can now be represented as

```text
                 ┌──────────────────┐
                 │ Existing Dataset │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Generative Model │
                 └────────┬─────────┘
                          ↓
                  Generated Crystals
                          ↓
                 ┌──────────────────┐
                 │ ML Predictors    │
                 └────────┬─────────┘
                          ↓
                   Candidate Ranking
                          ↓
                 ┌──────────────────┐
                 │ Active Learning  │
                 └────────┬─────────┘
                          ↓
                   Selected Crystals
                          ↓
                 ┌──────────────────┐
                 │ DFT Calculation  │
                 └────────┬─────────┘
                          ↓
                    New High-
                    Quality Data
                          │
                          └──────────────┐
                                         ↓
                                  Model Updating
                                         │
                                         └──────↺
```

This is a central concept in modern computational materials discovery.

---

# 24.12.17 Practical Dataset Updating

Suppose the original dataset is stored in a dataframe.

```python
import pandas as pd

dataset = pd.read_csv(
    "materials_dataset.csv"
)
```

After DFT evaluation, new results may be stored as

```python
new_results = pd.DataFrame({

    "structure": structures,

    "band_gap": dft_band_gaps,

    "formation_energy":
        dft_formation_energies,

    "density": densities

})
```

The datasets can then be combined.

```python
updated_dataset = pd.concat(
    [
        dataset,
        new_results
    ],
    ignore_index=True
)
```

The updated dataset becomes the basis for the next training cycle.

---

# 24.12.18 Retraining the Property Predictor

After adding DFT results, the property predictor can be retrained.

```python
X = build_features(
    updated_dataset
)

y = updated_dataset[
    "formation_energy"
].values
```

The model can then be trained again.

```python
predictor.fit(
    X,
    y
)
```

The new model should be evaluated on a carefully separated validation or test set.

It is important not to evaluate the model only on the same data used for retraining.

---

# 24.12.19 Updating the Generative Model

The generative model can also be updated.

There are several possible strategies.

### Full retraining

Train the generative model again using the expanded dataset.

### Fine-tuning

Continue training the existing model using the new data.

### Incremental learning

Update the model periodically as new high-quality data become available.

The choice depends on

```text
Dataset Size
+
Model Architecture
+
Computational Cost
+
Frequency of New Data
```

---

# 24.12.20 Avoiding Feedback Loops That Amplify Bias

Closed-loop systems can introduce new problems.

Suppose the generator repeatedly produces materials from one narrow region.

The active-learning system selects only those candidates.

DFT evaluates them.

The new training data therefore becomes even more concentrated in the same region.

After several iterations, the model may become increasingly biased.

Conceptually,

```text
Initial Bias
    ↓
Generation Bias
    ↓
Selection Bias
    ↓
New Dataset Bias
    ↓
Stronger Generation Bias
    ↺
```

This is a dangerous feedback loop.

Therefore, diversity constraints should be included in candidate selection.

---

# 24.12.21 Diversity-Aware Active Learning

Candidate selection can combine

```text
Property Quality
+
Uncertainty
+
Structural Diversity
```

For example,

```python
score = (
    quality_score
    +
    beta * uncertainty
    +
    gamma * diversity_score
)
```

where

```text
beta
```

controls exploration through uncertainty and

```text
gamma
```

controls exploration through diversity.

This encourages the system to select candidates that are

```text
Promising
+
Informative
+
Structurally Diverse
```

---

# 24.12.22 Batch Selection

DFT calculations are often performed in batches.

Suppose the computational budget allows

```text
100 DFT calculations
```

per iteration.

The system should not simply select the top 100 candidates if those candidates are nearly identical.

Instead, it may select

```text
Top Candidates
+
Diversity Constraint
```

For example:

```text
Candidate Ranking
       ↓
Select Best Candidate
       ↓
Remove Highly Similar Structures
       ↓
Select Next Candidate
       ↓
Repeat
```

This produces a more informative DFT batch.

---

# 24.12.23 Example Batch Selection

A simplified implementation can use embeddings.

```python
selected = []

for candidate in ranked_candidates:

    if len(selected) >= budget:
        break

    if is_sufficiently_different(
        candidate,
        selected
    ):

        selected.append(
            candidate
        )
```

The similarity criterion may be based on

```text
Composition
+
Structure Fingerprint
+
Graph Embedding
+
Latent Representation
```

The precise implementation depends on the model and materials representation.

---

# 24.12.24 Generative Model as a Proposal Engine

At this stage, it is useful to reinterpret the generative model.

The generator does not necessarily need to produce the final material directly.

Instead, it can act as a **proposal engine**.

```text
Generative Model
       ↓
Large Candidate Pool
       ↓
Scientific Selection
       ↓
High-Value Candidates
```

This is often a more realistic way to use generative AI in materials discovery.

The generator explores possibilities.

Scientific models decide which possibilities deserve further investigation.

---

# 24.12.25 Integration with Existing Materials Informatics Models

The generative model does not need to operate independently.

It can be integrated with models developed elsewhere in the Materials Informatics workflow.

For example,

```text
Crystal Generator
       ↓
CGCNN / GNN
       ↓
Formation Energy
       ↓
Band Gap Model
       ↓
Elastic Property Model
       ↓
Stability Model
```

This creates a modular architecture.

Different predictive models can be replaced or improved independently.

---

# 24.12.26 Integration with CGCNN

A crystal graph neural network can act as a property predictor.

```text
Crystal Structure
       ↓
Graph Construction
       ↓
CGCNN
       ↓
Material Representation
       ↓
Property Prediction
```

The generator provides the structure.

The CGCNN evaluates the structure.

The resulting prediction is then used for candidate selection.

This demonstrates how generative AI connects naturally with the graph-based models developed earlier in this book.

---

# 24.12.27 Integration with M3GNet and Other Foundation Models

More advanced materials models can also be used.

For example, a generated structure could be evaluated using a pretrained materials model for

```text
Energy
+
Forces
+
Stress
+
Structure Relaxation
```

Such models may provide much faster approximate evaluation than a full DFT calculation.

The resulting hierarchy becomes

```text
Generative Model
       ↓
Fast ML Screening
       ↓
Advanced Materials Model
       ↓
DFT
       ↓
Experiment
```

Each stage becomes progressively more expensive and more accurate.

---

# 24.12.28 Hierarchical Materials Screening

This leads to a general principle:

> **Use inexpensive models to eliminate obviously poor candidates before applying expensive methods.**

For example,

```text
100,000 Generated
        ↓
Validity Filter
        ↓
50,000
        ↓
Property Predictor
        ↓
10,000
        ↓
Stability Model
        ↓
2,000
        ↓
Advanced ML Relaxation
        ↓
500
        ↓
DFT
        ↓
50
        ↓
Experiment
```

This hierarchy can dramatically reduce computational cost.

---

# 24.12.29 Cost-Aware Discovery

Every stage has an associated computational cost.

Conceptually,

```text
Validity Check
       ↓
Very Cheap

ML Prediction
       ↓
Cheap

Advanced ML Simulation
       ↓
Moderate

DFT
       ↓
Expensive

Experiment
       ↓
Very Expensive
```

Therefore, candidate selection should account for both

```text
Scientific Value
+
Computational Cost
```

The most valuable candidate is not necessarily the one with the highest predicted property.

It may be the candidate that provides the greatest scientific value per unit computational cost.

---

# 24.12.30 Discovery Efficiency

A useful concept is

```text
Discovery Efficiency
=
Useful Validated Candidates
/
Computational Cost
```

For example, if one workflow produces

```text
20 validated candidates
```

using

```text
10,000 CPU-hours
```

while another produces

```text
15 validated candidates
```

using

```text
2,000 CPU-hours
```

the second workflow may be more computationally efficient despite generating fewer candidates.

This becomes increasingly important for large-scale materials discovery.

---

# 24.12.31 Closed-Loop Example

Consider a photovoltaic-material discovery problem.

The design objectives are

```text
1.3 ≤ Eg ≤ 1.8 eV

Low Formation Energy

High Stability
```

The workflow begins with a dataset of known crystals.

```text
Materials Dataset
       ↓
Train Generative Model
       ↓
Generate 100,000 Candidates
```

The first filter removes invalid structures.

```text
100,000
   ↓
80,000 Valid
```

A band-gap predictor is applied.

```text
80,000
   ↓
15,000 Target-Compatible
```

Formation-energy prediction is then applied.

```text
15,000
   ↓
3,000 Promising
```

Diversity-aware selection reduces the set.

```text
3,000
   ↓
500 Diverse Candidates
```

DFT calculations are performed.

```text
500
   ↓
120 Stable
```

The new DFT results are added to the dataset.

```text
120 New High-Quality Data
       ↓
Updated Dataset
       ↓
Retraining
```

The generator is then used again.

```text
Iteration 1
      ↓
Iteration 2
      ↓
Iteration 3
      ↓
...
```

The goal is for the quality of discovered candidates to improve over successive iterations.

---

# 24.12.32 Measuring Improvement Across Iterations

A closed-loop system should be evaluated after every iteration.

For example:

| Iteration | Validity | Novelty | Property Success | DFT Success |
| --------- | -------: | ------: | ---------------: | ----------: |
| 0         |      ... |     ... |              ... |         ... |
| 1         |      ... |     ... |              ... |         ... |
| 2         |      ... |     ... |              ... |         ... |
| 3         |      ... |     ... |              ... |         ... |

This makes it possible to determine whether the feedback loop is actually improving the system.

Possible trends include

```text
Increasing Property Success
+
Increasing Stability
+
Maintained Diversity
```

which would indicate successful learning.

---

# 24.12.33 When the Loop Should Stop

A closed-loop system requires a stopping criterion.

Possible criteria include

```text
Target Number of Candidates Found
```

or

```text
No Significant Improvement
```

or

```text
Computational Budget Exhausted
```

or

```text
Desired Property Region Saturated
```

For example,

```python
if improvement < tolerance:

    stop = True
```

Another criterion may be

```text
No new high-value candidates
for N consecutive iterations
```

This is useful when the search has reached a region of diminishing returns.

---

# 24.12.34 Convergence of Materials Discovery

Unlike conventional optimization, materials discovery may not have a simple mathematical optimum.

The system may instead converge toward

```text
A Stable High-Quality Candidate Distribution
```

where additional iterations produce fewer genuinely new discoveries.

This can be interpreted as saturation of the explored materials space.

However, apparent convergence may also indicate

```text
Mode Collapse
+
Over-Exploitation
+
Insufficient Exploration
```

Therefore, convergence should always be interpreted together with diversity metrics.

---

# 24.12.35 Reproducible Closed-Loop Experiments

A complete experiment should record every iteration.

For each cycle, store

```text
Model Version
Dataset Version
Random Seed
Generated Structures
Filtering Results
Selected Candidates
DFT Results
Updated Dataset
```

A possible directory structure is

```text
project/
│
├── data/
│   ├── initial/
│   ├── iteration_01/
│   ├── iteration_02/
│   └── iteration_03/
│
├── models/
│   ├── generator/
│   └── predictors/
│
├── generation/
│
├── screening/
│
├── dft/
│
├── analysis/
│
└── configs/
```

This becomes especially important when experiments involve thousands or millions of generated structures.

---

# 24.12.36 The Complete Generative Materials Discovery Architecture

The ideas developed throughout this chapter can now be combined.

```text
                    MATERIALS DATABASE
                           │
                           ↓
                 Structure Representation
                           │
                           ↓
                Generative Crystal Model
                           │
                           ↓
                  Candidate Generation
                           │
                           ↓
                  Structural Validation
                           │
                           ↓
                  Chemical Validation
                           │
                           ↓
                  Duplicate Removal
                           │
             ┌─────────────┴─────────────┐
             ↓                           ↓
         Novelty                     Diversity
             │                           │
             └─────────────┬─────────────┘
                           ↓
                  Property Prediction
                           ↓
                  Uncertainty Estimation
                           ↓
                   Candidate Ranking
                           ↓
                  Active Learning
                           ↓
                    DFT Evaluation
                           ↓
                    New Knowledge
                           │
                           ↓
                     Model Update
                           │
                           └───────────↺
```

This is the foundation of a **closed-loop generative Materials Informatics system**.

---

# 24.12.37 From Candidate Generation to Autonomous Discovery

The ultimate extension of this framework is an increasingly autonomous discovery system.

A researcher specifies

```text
Scientific Objective
```

and the system performs

```text
Generation
      ↓
Screening
      ↓
Selection
      ↓
Simulation
      ↓
Learning
      ↓
Regeneration
```

with progressively less manual intervention.

Conceptually,

```text
Human Objective
      ↓
AI Generation
      ↓
AI Screening
      ↓
Automated DFT
      ↓
AI Learning
      ↓
Next Generation
```

This does not eliminate the role of the materials scientist.

Instead, it changes the role.

The researcher increasingly focuses on

```text
Problem Definition
+
Physical Constraints
+
Interpretation
+
Validation
+
Experimental Strategy
```

while computational systems handle large-scale candidate exploration.

---

# 24.12.38 Human-in-the-Loop Materials Discovery

Despite the increasing automation, human expertise remains essential.

A materials scientist may identify constraints that a purely statistical model cannot understand.

For example,

```text
Toxic Elements
+
Rare Elements
+
Experimental Inaccessibility
+
Synthesis Difficulty
+
Air Sensitivity
+
Cost
+
Environmental Constraints
```

may all influence whether a generated material is practically useful.

Therefore, a realistic framework is not

```text
AI → Material
```

but

```text
Scientist
   ↓
Design Objective
   ↓
AI Generation
   ↓
Scientific Screening
   ↓
Scientist Interpretation
   ↓
DFT / Experiment
   ↓
New Knowledge
```

The strongest systems will therefore combine machine intelligence with domain expertise.

---

# 24.12.39 From Computational Discovery to Experimental Discovery

The ultimate purpose of computational materials discovery is not to generate computer files.

The purpose is to identify materials worth investigating experimentally.

The complete pathway is therefore

```text
Generative AI
      ↓
Computational Candidates
      ↓
ML Screening
      ↓
DFT Validation
      ↓
Synthesis Feasibility
      ↓
Experimental Synthesis
      ↓
Experimental Characterization
      ↓
New Materials Knowledge
```

Experimental results can then become new training data.

This creates an even larger feedback loop:

```text
Computation
     ↕
Experiment
```

rather than computation operating independently from experiment.

---

# 24.12.40 Final Closed-Loop Principle

The central principle of this section can be summarized as

```text
Generate
   ↓
Predict
   ↓
Rank
   ↓
Select
   ↓
Simulate
   ↓
Validate
   ↓
Learn
   ↓
Generate Again
```

This transforms generative AI from a one-time structure-generation method into an iterative materials-discovery engine.

The key idea is not simply to generate more structures.

It is to generate **better candidates, learn from expensive evaluations, and progressively improve the search process**.

Therefore, a mature generative Materials Informatics system can be viewed as

```text
                ┌─────────────────────┐
                │ Scientific Objective│
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Generative Model    │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Candidate Materials │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ ML + AI Screening   │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Active Selection    │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ DFT / Experiment    │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ New Scientific Data │
                └──────────┬──────────┘
                           │
                           └──────────↺
```

The result is a continuously improving computational discovery framework.

This closes the conceptual transition from **generative modeling** to **generative Materials Informatics**.

The remaining development of the chapter can now move toward the advanced research frontier: integrating these generative systems with autonomous optimization, multimodal scientific models, text-guided materials generation, synthesis-aware design, and future AI-driven materials discovery platforms.

# 24.13 Autonomous Materials Discovery and Synthesis-Aware Generative Design

The closed-loop framework developed in Section 24.12 establishes an important principle:

```text
Generate
   ↓
Predict
   ↓
Select
   ↓
DFT / Experiment
   ↓
Learn
   ↺
```

However, a generated crystal is not automatically a useful material.

A computationally attractive structure may still be

```text
Difficult to Synthesize
Chemically Unstable
Expensive
Toxic
Metastable Under Real Conditions
Sensitive to Processing Conditions
```

Therefore, the next step toward a research-grade generative materials system is to incorporate **synthesis feasibility and experimental constraints directly into the discovery process**.

The objective changes from

```text
Generate a theoretically interesting crystal
```

to

```text
Generate a crystal that is

scientifically desirable

+

chemically plausible

+

structurally valid

+

thermodynamically reasonable

+

synthesizable

+

experimentally useful
```

This represents a major transition from **computational materials generation** toward **practical materials discovery**.

---

# 24.13.1 Why Synthesis Matters

Consider a generative model that produces a material with

```text
Excellent Band Gap
Low Formation Energy
High Stability
High Bulk Modulus
```

From a purely computational perspective, this may appear to be an excellent candidate.

However, suppose the material requires

```text
Extremely High Pressure
Rare Elements
Toxic Precursors
Highly Reactive Atmosphere
Difficult Temperature Control
```

The material may have little practical value.

Therefore, materials discovery contains at least two distinct questions:

```text
Can this material exist?
```

and

```text
Can we actually make it?
```

A successful generative system should eventually address both.

---

# 24.13.2 Computational Stability vs Experimental Stability

These concepts should not be confused.

A structure may be computationally stable under a particular theoretical approximation but experimentally inaccessible.

Similarly, a metastable material may not be the lowest-energy phase but may still be experimentally synthesizable.

Therefore,

```text
Lowest Energy
≠
Only Synthesizable Material
```

Materials synthesis depends on

```text
Thermodynamics
+
Kinetics
+
Temperature
+
Pressure
+
Chemical Environment
+
Reaction Pathway
```

A realistic discovery framework should therefore consider more than formation energy alone.

---

# 24.13.3 Thermodynamic Stability

Suppose a generated structure has formation energy

```text
Eform = -2.5 eV/atom
```

This alone does not guarantee stability against decomposition.

A material may decompose into a combination of other phases with lower total energy.

Therefore, phase stability is often evaluated relative to the **convex hull**.

Conceptually,

```text
Composition
      ↓
Competing Phases
      ↓
Energy Comparison
      ↓
Distance to Convex Hull
```

A structure close to the convex hull is generally more thermodynamically competitive.

---

# 24.13.4 Energy Above Hull

A useful quantity is

```text
Energy Above Hull
```

which measures how far a candidate lies above the thermodynamic stability boundary.

Conceptually,

```text
Eabove_hull = Ecandidate - Ehull
```

If

```text
Eabove_hull = 0
```

the structure lies on the convex hull.

If

```text
Eabove_hull > 0
```

the structure is thermodynamically metastable relative to competing phases.

The exact interpretation depends on the chemical system and computational methodology.

---

# 24.13.5 Stability-Aware Generation

The generative model can therefore include stability as another design objective.

Instead of

```text
Target Band Gap
      ↓
Generator
```

we can use

```text
Target Band Gap
+
Target Stability
      ↓
Generator
```

or

```text
Generator
      ↓
Candidate Structures
      ↓
Stability Predictor
      ↓
Filtering
```

A more sophisticated architecture can incorporate stability during generation itself.

---

# 24.13.6 Stability Conditioning

Suppose the target material should satisfy

```text
Band Gap ≈ 1.5 eV

Energy Above Hull < 0.05 eV/atom
```

The condition vector may become

```python
condition = torch.tensor([
    1.5,
    0.05
])
```

The property encoder can transform these values into a latent condition representation.

```python
condition_embedding = (
    property_encoder(condition)
)
```

The diffusion model can then generate structures conditioned on both objectives.

---

# 24.13.7 Synthesis Feasibility as a Property

Synthesis feasibility can itself be treated as a prediction problem.

For example, a model could estimate

```text
Probability of Successful Synthesis
```

from

```text
Composition
+
Structure
+
Known Precursors
+
Reaction Conditions
+
Temperature
+
Pressure
+
Atmosphere
```

The prediction could be represented as

```text
P(synthesis success | material, conditions)
```

This transforms synthesis from an afterthought into a measurable design objective.

---

# 24.13.8 Synthesis-Aware Generative Design

The design problem can therefore be expressed as

```text
Desired Properties
        +
Chemical Constraints
        +
Stability Constraints
        +
Synthesis Constraints
        ↓
Generative Model
        ↓
Candidate Crystal
```

For example:

```text
Band Gap: 1.3–1.8 eV

Energy Above Hull: < 0.05 eV/atom

No Highly Toxic Elements

Synthesis Temperature: < 1000 K
```

The generated material must satisfy all relevant constraints.

This is significantly more realistic than optimizing a single material property.

---

# 24.13.9 Reaction Conditions

A material does not exist independently of its synthesis environment.

Important variables may include

```text
Temperature
Pressure
Time
Atmosphere
Precursor Ratio
Solvent
pH
Cooling Rate
Heating Rate
```

Therefore, an advanced generative framework may eventually generate not only

```text
Crystal Structure
```

but also

```text
Synthesis Conditions
```

The problem becomes

```text
Target Material
       ↓
Structure
       +
Synthesis Route
       ↓
Experimental Candidate
```

---

# 24.13.10 Joint Structure–Synthesis Generation

An ambitious architecture could model

```text
p(C, R | y)
```

where

```text
C = Crystal Structure

R = Synthesis Route

y = Desired Properties
```

The model therefore attempts to generate both a material and a plausible pathway for producing it.

Conceptually,

```text
Target Properties
       ↓
 ┌───────────────┐
 │ Generative AI │
 └───────┬───────┘
         ↓
 ┌───────┴────────┐
 ↓                ↓
Crystal         Synthesis
Structure        Route
 ↓                ↓
 └───────┬────────┘
         ↓
 Experimental Candidate
```

This represents an important direction for future Materials Informatics research.

---

# 24.13.11 Why Synthesis Generation Is Difficult

Synthesis is fundamentally different from static structure prediction.

A crystal structure describes a final state.

A synthesis route describes a **process**.

Processes depend on

```text
Time
+
Temperature
+
Chemical Environment
+
Reaction Pathways
+
Kinetics
```

Therefore, synthesis-aware generation requires models capable of representing sequences and transformations.

This creates a connection between materials generation and

```text
Sequence Modeling
+
Graph Generation
+
Reaction Prediction
+
Process Optimization
```

---

# 24.13.12 Materials Reaction Graphs

A synthesis process can be represented as a graph.

For example,

```text
Precursor A ──┐
              ├──► Intermediate ──► Product
Precursor B ──┘
```

More complex reactions may contain multiple stages.

```text
A + B
 ↓
Intermediate 1
 ↓
Intermediate 2
 ↓
Final Crystal
```

A generative system could therefore learn reaction pathways as graph transformations.

---

# 24.13.13 Connecting Crystal Generation with Reaction Prediction

The complete workflow can become

```text
Target Properties
       ↓
Crystal Generator
       ↓
Candidate Crystal
       ↓
Reaction / Synthesis Model
       ↓
Possible Synthesis Routes
       ↓
Feasibility Prediction
       ↓
Candidate Ranking
```

This provides an important bridge between computational materials discovery and experimental planning.

---

# 24.13.14 Cost-Constrained Materials Discovery

Not all promising materials are equally valuable.

Suppose two candidates have identical predicted properties.

Candidate A requires

```text
Common Elements
+
Cheap Precursors
+
Low Temperature
```

Candidate B requires

```text
Rare Element
+
Expensive Precursor
+
Extreme Pressure
```

Candidate A may be more attractive experimentally.

Therefore, candidate ranking should eventually consider

```text
Scientific Performance
+
Stability
+
Synthesis Feasibility
+
Cost
+
Availability
```

---

# 24.13.15 Element Availability

Element availability can be incorporated into the design objective.

For example,

```text
Abundant Elements
```

may be preferred over

```text
Critical / Rare Elements
```

depending on the application.

A simple penalty function could be introduced.

```python
availability_penalty = (
    calculate_element_scarcity(
        composition
    )
)
```

The final candidate score could then include

```python
score = (
    property_score
    - stability_penalty
    - synthesis_penalty
    - availability_penalty
)
```

The exact form depends on the research objective.

---

# 24.13.16 Toxicity Constraints

A theoretically excellent candidate may be unsuitable because of toxicity.

For example, an application may require

```text
No Pb
No Hg
No Cd
```

or other application-specific restrictions.

This can be implemented as a hard constraint.

```python
if contains_restricted_element(
    composition
):
    reject(candidate)
```

Alternatively, toxicity can be incorporated as a soft optimization objective.

---

# 24.13.17 Application-Specific Constraints

The constraints depend strongly on the target application.

### Battery materials

Important considerations may include

```text
Ionic Conductivity
Voltage
Capacity
Stability
Element Availability
```

### Photovoltaic materials

Important considerations may include

```text
Band Gap
Absorption
Stability
Defect Tolerance
Toxicity
```

### Structural materials

Important considerations may include

```text
Strength
Bulk Modulus
Fracture Resistance
Density
Temperature Stability
```

Therefore, generative materials discovery should be **application-aware** rather than property-agnostic.

---

# 24.13.18 Application-Specific Objective Functions

A photovoltaic discovery problem may define

```python
objective = (
    gap_score
    + stability_score
    + absorption_score
    - toxicity_penalty
)
```

A battery-material problem may instead use

```python
objective = (
    capacity_score
    + voltage_score
    + conductivity_score
    + stability_score
)
```

The same generative architecture can therefore support very different scientific objectives.

---

# 24.13.19 Constraint Hierarchy

Not every constraint should necessarily be treated equally.

A useful hierarchy is

```text
Hard Constraints
      ↓
Soft Constraints
      ↓
Optimization Objectives
```

For example:

### Hard

```text
Invalid Structure → Reject
```

### Soft

```text
Moderate Instability → Penalize
```

### Objective

```text
Higher Conductivity → Prefer
```

This distinction makes candidate selection more scientifically meaningful.

---

# 24.13.20 Hard Constraints

Examples include

```text
Atomic Overlap
Invalid Composition
Impossible Geometry
Forbidden Elements
Invalid Oxidation State
```

These candidates can be removed immediately.

```python
valid = (
    structure_is_valid(candidate)
    and
    composition_is_allowed(candidate)
)
```

Hard filtering is computationally efficient because it prevents expensive downstream calculations.

---

# 24.13.21 Soft Constraints

Some properties should not necessarily cause immediate rejection.

For example,

```text
Energy Above Hull = 0.08 eV/atom
```

may be acceptable for a metastable material depending on the scientific context.

Therefore, instead of

```python
if energy_above_hull > threshold:
    reject()
```

one may assign a penalty.

```python
penalty = (
    alpha *
    energy_above_hull
)
```

This allows the system to retain potentially interesting metastable candidates.

---

# 24.13.22 Constrained Generative Modeling

An advanced generative model can attempt to generate only structures satisfying known constraints.

Conceptually,

```text
p(C | y, constraints)
```

rather than simply

```text
p(C | y)
```

The constraint information can enter through

```text
Condition Embeddings
+
Guidance
+
Energy Functions
+
Rejection Sampling
+
Post-Generation Filtering
```

Different strategies offer different computational and modeling advantages.

---

# 24.13.23 Rejection Sampling

The simplest constrained generation strategy is rejection sampling.

Generate a structure.

Check constraints.

If invalid, reject it.

```python
while len(valid_candidates) < target:

    candidate = generate()

    if satisfies_constraints(
        candidate
    ):

        valid_candidates.append(
            candidate
        )
```

This is easy to implement but can become inefficient when valid structures are rare.

---

# 24.13.24 Constraint-Guided Diffusion

A more advanced approach incorporates constraints during the diffusion trajectory.

Conceptually,

```text
Unconstrained Score
        +
Constraint Guidance
        ↓
Modified Score
        ↓
Constrained Sampling
```

The goal is to prevent the trajectory from moving toward obviously invalid regions.

This is particularly attractive for crystal generation because structural validity can be difficult to enforce after generation.

---

# 24.13.25 Energy-Based Guidance

A differentiable energy function can be used to guide generation.

Suppose

```text
E(C)
```

represents an energy-like objective.

The model can use the gradient

```text
∇C E(C)
```

to influence the generation trajectory.

Conceptually,

```text
Generative Score
      +
Energy Gradient
      ↓
Guided Generation
```

This creates a connection between

```text
Generative Modeling
```

and

```text
Physics-Based Optimization
```

---

# 24.13.26 Physics-Guided Generation

Materials are governed by physical laws.

Therefore, generative models can potentially benefit from physical constraints.

Examples include

```text
Charge Neutrality
Periodic Boundary Conditions
Atomic Distances
Symmetry
Conservation Laws
Energy Constraints
```

A physics-aware generator should therefore not treat materials as arbitrary data points.

It should respect the structure of the physical problem.

---

# 24.13.27 Charge Neutrality

For ionic materials, generated compositions should generally satisfy charge neutrality.

Suppose a composition contains

```text
A²⁺
B³⁺
O²⁻
```

The total charge must be consistent with a physically meaningful composition.

A simple screening function could be

```python
charge = calculate_total_charge(
    composition
)

if abs(charge) > tolerance:
    reject(candidate)
```

Such rules can remove obviously problematic candidates before expensive calculations.

---

# 24.13.28 Geometric Constraints

Generated crystals should also satisfy basic geometric requirements.

Examples include

```text
Minimum Interatomic Distance
Reasonable Coordination
Non-Degenerate Unit Cell
Reasonable Density
```

A simple distance check may be written conceptually as

```python
for pair in atomic_pairs:

    if pair.distance < minimum_distance:

        valid = False
```

This is an important first-stage structural validation.

---

# 24.13.29 Symmetry Constraints

Crystal symmetry provides another source of prior knowledge.

A generator may be conditioned on

```text
Space Group
Crystal System
Point Group
```

For example,

```text
Target:

Cubic Structure
```

can be incorporated as a categorical condition.

This may significantly reduce the search space.

---

# 24.13.30 Symmetry-Aware Generation

The generation process can therefore become

```text
Composition
+
Space Group
+
Target Properties
      ↓
Generative Model
      ↓
Crystal Structure
```

This is particularly useful when the researcher knows that a desired material class should possess a particular symmetry.

---

# 24.13.31 Structure–Property–Process Design

At the most advanced level, materials discovery becomes a three-way problem.

```text
Structure
    ↕
Properties
    ↕
Process
```

The structure determines properties.

The synthesis process determines which structures can actually be produced.

The process can also influence defects, phase composition, microstructure, and metastability.

Therefore,

```text
Structure
+
Property
+
Synthesis
```

should eventually be treated as a coupled system.

---

# 24.13.32 Toward Autonomous Laboratories

The natural endpoint of this development is an autonomous materials laboratory.

The computational system proposes candidates.

Automated equipment synthesizes them.

Characterization systems measure the resulting materials.

The measurements are returned to the AI system.

The loop continues.

Conceptually,

```text
                  AI
                   ↓
             Candidate Design
                   ↓
             Automated Synthesis
                   ↓
              Characterization
                   ↓
              Experimental Data
                   ↓
                  AI
                   ↺
```

This is sometimes described as a **self-driving laboratory**.

---

# 24.13.33 Self-Driving Materials Research

A self-driving materials workflow may contain

```text
Generative AI
      ↓
Experiment Planner
      ↓
Robotic Synthesis
      ↓
Automated Characterization
      ↓
Data Analysis
      ↓
Active Learning
      ↓
Next Experiment
```

The human researcher defines the scientific objective and constraints.

The automated system performs repeated cycles of hypothesis generation and testing.

---

# 24.13.34 Role of the Materials Scientist

Automation does not remove the need for scientific understanding.

The researcher remains responsible for

```text
Defining the Scientific Problem
Selecting Meaningful Objectives
Choosing Appropriate Constraints
Evaluating Physical Plausibility
Interpreting Results
Designing Experiments
Understanding Failure
```

AI can accelerate search.

It does not replace scientific judgment.

---

# 24.13.35 The Researcher–AI Partnership

A useful conceptual model is

```text
Human
  │
  │ Scientific Knowledge
  ↓
AI
  │
  │ Large-Scale Exploration
  ↓
Candidate Materials
  │
  │ Scientific Validation
  ↓
Human
```

The strongest workflow therefore combines

```text
Human Scientific Reasoning
+
Machine Learning
+
Simulation
+
Automation
```

rather than relying on any one component independently.

---

# 24.13.36 Practical Research Pipeline

A research-grade generative materials project can now be organized as

```text
1. Define Scientific Objective
          ↓
2. Assemble Materials Dataset
          ↓
3. Represent Crystal Structures
          ↓
4. Train Generative Model
          ↓
5. Generate Candidate Structures
          ↓
6. Validate Structural Integrity
          ↓
7. Apply Chemical Constraints
          ↓
8. Remove Duplicates
          ↓
9. Predict Material Properties
          ↓
10. Estimate Prediction Uncertainty
          ↓
11. Rank Candidates
          ↓
12. Apply Diversity Constraints
          ↓
13. Evaluate Stability
          ↓
14. Evaluate Synthesis Feasibility
          ↓
15. Select DFT Candidates
          ↓
16. Perform DFT
          ↓
17. Compare ML Predictions with DFT
          ↓
18. Update Dataset
          ↓
19. Retrain Models
          ↓
20. Repeat
```

This pipeline integrates essentially every major concept developed in the generative-materials portion of the chapter.

---

# 24.13.37 A Research-Grade Candidate Record

For reproducibility, each generated material should ideally be stored together with its complete computational history.

A candidate record may contain

```python
candidate = {

    "structure": structure,

    "composition":
        structure.composition.formula,

    "generation_model":
        model_version,

    "generation_seed":
        random_seed,

    "target_properties":
        target_properties,

    "predicted_properties":
        predictions,

    "prediction_uncertainty":
        uncertainties,

    "formation_energy":
        formation_energy,

    "energy_above_hull":
        energy_above_hull,

    "validity":
        validity,

    "novelty":
        novelty,

    "diversity_score":
        diversity,

    "synthesis_score":
        synthesis_score,

    "selection_iteration":
        iteration

}
```

This record makes it possible to trace a candidate from its generation to its final validation.

---

# 24.13.38 Candidate Provenance

Provenance is particularly important in generative materials research.

For every structure, researchers should be able to answer:

```text
Which model generated it?

Which checkpoint?

Which random seed?

Which target condition?

Which dataset?

Which preprocessing pipeline?

Which screening models?

Which DFT settings?

Which final result?
```

Without provenance, reproducing a discovery can become extremely difficult.

---

# 24.13.39 Reproducibility Checklist

A reproducible generative-materials experiment should record at least:

```text
Dataset Version
Model Architecture
Model Parameters
Training Configuration
Random Seeds
Software Versions
Python Environment
Generation Conditions
Filtering Criteria
Prediction Models
DFT Parameters
Selection Rules
Evaluation Metrics
```

The computational environment should also be preserved whenever possible.

For example,

```text
requirements.txt
environment.yml
Dockerfile
```

can help reconstruct the software environment.

---

# 24.13.40 Research-Level Evaluation

A generative model should not be judged by generated structures alone.

A research-grade evaluation should examine

```text
Validity
Novelty
Uniqueness
Diversity
Property Accuracy
Stability
Distribution Matching
Condition Satisfaction
Synthesis Feasibility
Computational Efficiency
```

The final evaluation should therefore answer several different questions:

```text
Can the model generate valid crystals?
```

```text
Can it generate genuinely new crystals?
```

```text
Can it satisfy the requested properties?
```

```text
Are the generated materials physically meaningful?
```

```text
Can they potentially be synthesized?
```

```text
Does the model outperform simpler baselines?
```

---

# 24.13.41 Benchmarking Against Baselines

A generative model should not be evaluated in isolation.

Possible baselines include

```text
Random Sampling
Composition-Based Search
Evolutionary Algorithms
Bayesian Optimization
Existing Generative Models
Unconditional Diffusion
Conditional Diffusion
```

The comparison should use the same evaluation criteria and computational budget whenever possible.

---

# 24.13.42 Why Baselines Matter

Suppose a generative model discovers

```text
50 promising materials
```

This number alone says little.

If random search discovers

```text
55
```

under the same budget, the generative model may not provide an advantage.

Therefore, the meaningful question is

```text
How much better is the generative approach
than a reasonable alternative?
```

This is a fundamental principle of scientific machine-learning research.

---

# 24.13.43 Generative AI as an Optimization Engine

At this stage, the role of generative AI can be viewed more broadly.

It is not merely

```text
A Crystal Generator
```

but potentially

```text
A Search-Space Model
```

that learns where scientifically meaningful materials are likely to exist.

This is why generative modeling is particularly attractive for inverse materials design.

Instead of searching every possible crystal individually, the model learns a probability distribution over plausible structures.

---

# 24.13.44 The Central Research Vision

The long-term vision can therefore be summarized as

```text
Scientific Objective
        ↓
Learned Materials Distribution
        ↓
Generative Search
        ↓
Physics-Aware Filtering
        ↓
Property Prediction
        ↓
Active Learning
        ↓
DFT
        ↓
Experiment
        ↓
New Knowledge
        ↺
```

The system continuously transforms

```text
Data
```

into

```text
Knowledge
```

and then uses that knowledge to guide the next generation of materials.

---

# 24.13.45 Summary

This section extended the closed-loop generative framework into a more realistic materials-discovery system.

The major ideas were:

```text
Synthesis Feasibility
```

```text
Thermodynamic Stability
```

```text
Energy Above Hull
```

```text
Chemical Constraints
```

```text
Physics-Guided Generation
```

```text
Cost-Aware Candidate Selection
```

```text
Application-Specific Objectives
```

```text
Structure–Property–Process Design
```

```text
Autonomous Experimentation
```

```text
Self-Driving Laboratories
```

The central transition is

```text
Crystal Generation
        ↓
Materials Discovery
        ↓
Synthesis-Aware Discovery
        ↓
Autonomous Materials Research
```

A mature generative Materials Informatics framework therefore needs to connect **generation, prediction, simulation, optimization, synthesis, experimentation, and learning** into a single scientific loop.

The next stage can move deeper into the **research frontier of multimodal generative AI**, where crystal structures, compositions, properties, scientific literature, synthesis procedures, images, and experimental measurements are represented jointly rather than as isolated data types.

# 24.14 Multimodal Generative AI for Materials Discovery

The previous sections developed generative models capable of producing crystal structures and incorporating increasingly realistic constraints such as

```text
Target Properties
+
Chemical Constraints
+
Thermodynamic Stability
+
Synthesis Feasibility
```

However, real materials research does not consist of crystal structures alone.

A modern materials-science dataset may contain

```text
Crystal Structures
Compositions
DFT Results
Experimental Measurements
Synthesis Procedures
Scientific Papers
Microscopy Images
Diffraction Patterns
Spectroscopy
Temperature
Pressure
Processing Conditions
```

These different forms of information describe different aspects of the same physical system.

Therefore, a major direction in modern AI research is **multimodal learning**.

The central idea is to learn representations that connect different scientific modalities.

Conceptually,

```text
                    Materials Knowledge
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
      Structure          Text          Experimental Data
          │                │                │
          └────────────────┼────────────────┘
                           ↓
                    Multimodal Model
                           ↓
                  Materials Understanding
```

The objective is no longer simply

```text
Generate a crystal
```

but rather

```text
Understand the material from many forms of scientific evidence
```

and eventually

```text
Generate new materials using all available information.
```

---

# 24.14.1 What Is Multimodal Materials AI?

A modality is a particular form in which information is represented.

For materials science, important modalities include

```text
Structure
Composition
Text
Images
Graphs
Numerical Properties
Spectra
Processing Conditions
```

A multimodal model attempts to learn relationships between these representations.

For example,

```text
Crystal Structure
        ↕
Band Gap
        ↕
Scientific Description
        ↕
Experimental Measurement
```

The same material can therefore be represented from multiple perspectives.

---

# 24.14.2 Why Multimodal Learning Is Important

Materials databases are highly heterogeneous.

One database may contain

```text
Composition + Formation Energy
```

while another contains

```text
Crystal Structure + Band Gap
```

and scientific literature may contain

```text
Synthesis Procedure + Experimental Properties
```

A human researcher naturally combines these sources.

A traditional machine-learning model usually cannot.

Therefore, multimodal AI attempts to construct a unified representation.

---

# 24.14.3 The Materials Knowledge Graph

One useful conceptual representation is a materials knowledge graph.

For example,

```text
Material
   │
   ├── Composition
   │
   ├── Crystal Structure
   │
   ├── Band Gap
   │
   ├── Formation Energy
   │
   ├── Synthesis Method
   │
   ├── Temperature
   │
   ├── Experimental Result
   │
   └── Scientific Literature
```

Relationships between these entities can also be represented.

```text
Material A
   │
   ├── has_structure → Structure A
   ├── has_property → Band Gap
   ├── synthesized_by → Method X
   └── reported_in → Paper Y
```

This creates a bridge between machine learning and structured scientific knowledge.

---

# 24.14.4 Structure as One Modality

Crystal structures can be represented using several approaches.

For example,

```text
Atomic Coordinates
```

```text
Crystal Graph
```

```text
Periodic Graph
```

```text
Voxel Representation
```

```text
Continuous Equivariant Representation
```

The choice depends on the model architecture.

For materials generation, graph-based and coordinate-based representations are particularly important because they preserve structural information.

---

# 24.14.5 Composition as a Modality

Composition provides another representation.

For example,

```text
LiFePO₄
```

can be represented as

```text
Li : 1
Fe : 1
P  : 1
O  : 4
```

or as an elemental feature vector.

A composition encoder can transform this information into a latent representation.

Conceptually,

```text
Composition
     ↓
Element Encoder
     ↓
Composition Embedding
```

The composition embedding can then interact with a structural representation.

---

# 24.14.6 Property Vectors

A material can also be represented by a property vector.

For example,

```python
properties = [
    band_gap,
    formation_energy,
    density,
    bulk_modulus,
    magnetic_moment
]
```

The numerical vector can be passed through a neural network.

```python
property_embedding = property_encoder(
    properties
)
```

This provides another modality for the model.

---

# 24.14.7 Scientific Text as a Modality

Scientific literature contains enormous amounts of materials knowledge.

A paper may describe

```text
Composition
Crystal Structure
Synthesis Conditions
Band Gap
Magnetic Behavior
Mechanical Properties
Experimental Limitations
```

Much of this information may not exist in a structured database.

Therefore, language models can potentially extract useful information from scientific text.

For example,

```text
Paper
 ↓
Scientific Language Model
 ↓
Material Information
 ↓
Structured Representation
```

---

# 24.14.8 Text-to-Materials Representation

Suppose a paper contains a statement such as

```text
A perovskite oxide synthesized at elevated temperature
exhibits a band gap near 2 eV.
```

A language model could potentially extract

```text
Material Class:
Perovskite Oxide

Processing:
High Temperature

Band Gap:
≈ 2 eV
```

The extracted information can then be connected to a materials database.

This is an important application of language models in Materials Informatics.

---

# 24.14.9 Scientific Language Models

A scientific language model can be trained or adapted using

```text
Research Papers
Patents
Materials Databases
Technical Reports
Experimental Records
```

The model learns scientific terminology and relationships.

Examples of information it may learn include

```text
material → property
material → synthesis
composition → phase
structure → behavior
processing → microstructure
```

However, language-model predictions should not automatically be treated as experimentally verified facts.

Scientific validation remains essential.

---

# 24.14.10 Joint Embedding

One strategy for multimodal learning is to map different modalities into a common latent space.

For example,

```text
Crystal Structure ──► Structure Encoder ──┐
                                         │
Composition ───────► Composition Encoder ├──► Shared Latent Space
                                         │
Text ──────────────► Text Encoder ───────┤
                                         │
Properties ────────► Property Encoder ───┘
```

The goal is for representations describing the same material to become close in latent space.

---

# 24.14.11 Shared Latent Space

Suppose

```text
Structure A
```

and

```text
Text Description A
```

describe the same material.

The model should ideally produce

```text
z_structure ≈ z_text
```

in the shared latent space.

Similarly,

```text
z_composition
z_property
z_experiment
```

may encode complementary information about the same material.

---

# 24.14.12 Contrastive Learning

A powerful method for learning multimodal representations is contrastive learning.

The basic idea is simple.

Matching pairs should have similar embeddings.

Non-matching pairs should have different embeddings.

For example,

```text
Correct Structure ↔ Correct Text
```

should be close.

While

```text
Structure A ↔ Text B
```

should be farther apart.

Conceptually,

```text
Positive Pair
      ↓
Bring Together

Negative Pair
      ↓
Push Apart
```

---

# 24.14.13 Contrastive Objective

Let

```text
z_s
```

represent a structure embedding and

```text
z_t
```

represent a text embedding.

A similarity function can be defined as

```text
sim(z_s, z_t)
```

The training objective encourages high similarity for matching pairs and low similarity for non-matching pairs.

A temperature parameter is often used to control the sharpness of the distribution.

The exact loss depends on the chosen contrastive-learning formulation.

---

# 24.14.14 Materials Structure–Text Alignment

A structure–text model could learn relationships such as

```text
Crystal Structure
       ↕
"Layered oxide with strong anisotropic conductivity"
```

This enables a researcher to search a materials database using natural language.

For example,

```text
Find layered materials with high ionic conductivity.
```

The model could retrieve structures whose learned representations are consistent with the description.

---

# 24.14.15 Text-Guided Materials Generation

This idea can be extended from retrieval to generation.

Instead of providing a numerical condition,

```text
Band Gap = 2.0 eV
```

the researcher could provide a textual specification:

```text
Generate a stable semiconductor with
a band gap near 2 eV and high thermal stability.
```

The language representation becomes the conditioning signal.

Conceptually,

```text
Text Prompt
     ↓
Text Encoder
     ↓
Condition Embedding
     ↓
Crystal Generative Model
     ↓
Candidate Structures
```

This is one of the most interesting future directions in generative Materials Informatics.

---

# 24.14.16 Numerical Conditioning vs Text Conditioning

Numerical conditioning is precise.

For example,

```text
Band Gap = 1.8 eV
```

Text conditioning is more flexible.

For example,

```text
A semiconductor suitable for visible-light
optoelectronic applications.
```

The second description may contain information that is difficult to encode as a single numerical vector.

Therefore, the two approaches can complement each other.

---

# 24.14.17 Hybrid Conditioning

An advanced system could accept both.

```text
Text:
High-temperature structural material

Band Gap:
Not important

Bulk Modulus:
> 250 GPa

Density:
< 6 g/cm³
```

The conditioning vector could contain

```text
Text Embedding
+
Numerical Property Embedding
+
Composition Embedding
```

The combined representation is then supplied to the generative model.

---

# 24.14.18 Multimodal Conditioning Architecture

A conceptual architecture is

```text
                    ┌───────────────┐
                    │ Text Encoder  │
                    └───────┬───────┘
                            │
                    Text Embedding
                            │
                            ▼
Structure ──► Structure Encoder ──┐
                                 │
Composition ─► Composition Encoder
                                 │
Properties ─► Property Encoder ──┤
                                 ▼
                         Fusion Network
                                 │
                                 ▼
                         Diffusion Model
                                 │
                                 ▼
                         Crystal Structure
```

The fusion network combines information from different modalities.

---

# 24.14.19 Early Fusion

In early fusion, different representations are combined near the beginning of the model.

For example,

```python
combined = torch.cat(
    [
        structure_embedding,
        composition_embedding,
        property_embedding
    ],
    dim=-1
)
```

The combined representation is then passed through the remaining network.

The advantage is simplicity.

The limitation is that the model must learn how to integrate heterogeneous representations from an early stage.

---

# 24.14.20 Late Fusion

In late fusion, each modality is processed independently for a longer portion of the architecture.

Conceptually,

```text
Structure ──► Encoder ──┐
                        │
Text ───────► Encoder ──┤
                        ├──► Fusion
Properties ─► Encoder ──┤
                        │
Composition ─► Encoder ─┘
```

This allows each encoder to specialize in its modality.

The resulting representations are combined later.

---

# 24.14.21 Cross-Attention

A more powerful mechanism is cross-attention.

One modality can attend to another.

For example,

```text
Crystal Representation
        ↕
Text Representation
```

The structure representation can selectively use information from the text representation.

Conceptually,

```text
Structure Tokens
       │
       ▼
Cross Attention
       ▲
       │
Text Tokens
```

This is particularly useful when the relationship between modalities is complex.

---

# 24.14.22 Why Attention Is Useful for Materials

A scientific description may contain many pieces of information.

For example,

```text
The material is stable below 800 K,
contains layered Fe-O units,
and exhibits a band gap of approximately 1.9 eV.
```

Different parts of the sentence correspond to different physical constraints.

Attention mechanisms can learn which pieces of information are relevant to the generation task.

---

# 24.14.23 Structure–Text Cross-Attention

Suppose the structure model receives

```text
Atomic Representation
```

and the language model receives

```text
Scientific Description
```

Cross-attention allows the structure representation to incorporate relevant textual information.

The architecture becomes

```text
Structure Features
       ↓
Cross-Attention
       ↑
Text Features
       ↓
Conditioned Structure Representation
```

This can be integrated into a diffusion network.

---

# 24.14.24 Multimodal Diffusion

The same principle can be extended to diffusion models.

Instead of

```text
Noise
+
Property Condition
```

the diffusion model receives

```text
Noise
+
Structure Context
+
Composition
+
Properties
+
Text
```

The denoising network learns

```text
εθ(x_t, t, c)
```

where `c` can contain multimodal conditioning information.

The generative process therefore becomes

```text
Multimodal Scientific Condition
              ↓
        Diffusion Model
              ↓
        Crystal Structure
```

---

# 24.14.25 Materials Image Generation and Interpretation

Images are another important modality.

Materials research routinely produces

```text
SEM Images
TEM Images
AFM Images
Optical Microscopy
XRD Patterns
Electron Diffraction
Spectroscopy
```

AI can process these images to infer structural or physical information.

For example,

```text
SEM Image
   ↓
Vision Encoder
   ↓
Microstructure Representation
```

This representation can be connected to composition and processing information.

---

# 24.14.26 Microstructure as a Modality

For many materials,

```text
Atomic Structure
```

is not sufficient.

Macroscopic properties may depend strongly on

```text
Grain Size
Porosity
Defect Density
Phase Distribution
Grain Boundaries
Texture
```

Therefore, a multimodal materials model could represent

```text
Crystal Structure
+
Microstructure
+
Processing
+
Properties
```

This creates a connection between Materials Informatics and materials processing.

---

# 24.14.27 XRD as a Modality

X-ray diffraction provides information about crystal structure.

An XRD pattern can be represented as a numerical sequence:

```python
xrd_pattern = [
    intensity_1,
    intensity_2,
    intensity_3,
    ...
]
```

A neural encoder can transform the pattern into a latent representation.

```python
xrd_embedding = xrd_encoder(
    xrd_pattern
)
```

This embedding can then be compared with a generated or predicted crystal structure.

---

# 24.14.28 Structure-to-XRD Consistency

A generated crystal should produce a physically consistent diffraction pattern.

Conceptually,

```text
Generated Crystal
       ↓
Simulated XRD
       ↓
Compare with Experimental XRD
```

This creates an additional validation layer.

A candidate that satisfies property constraints but produces an inconsistent diffraction signature may require further investigation.

---

# 24.14.29 Spectroscopy as a Modality

Other measurements can also be incorporated.

Examples include

```text
Raman Spectra
IR Spectra
XPS
UV–Vis
NMR
Magnetic Measurements
```

The same general framework applies:

```text
Measurement
   ↓
Encoder
   ↓
Latent Representation
```

The latent representation can then be fused with structural information.

---

# 24.14.30 Experimental Data Fusion

Consider a material for which we have

```text
Composition
Crystal Structure
XRD
SEM
Band Gap
Synthesis Temperature
```

A multimodal model can combine all of these.

```text
Composition ───────┐
Structure ─────────┤
XRD ───────────────┤
SEM ───────────────┤
Properties ────────┤
Processing ────────┘
          ↓
   Multimodal Encoder
          ↓
   Material Representation
```

This representation may be substantially richer than any individual modality.

---

# 24.14.31 Missing Modalities

Real datasets are rarely complete.

One material may have

```text
Structure + DFT
```

but no experiment.

Another may have

```text
Experimental Data
```

but no complete crystal structure.

A practical multimodal system must therefore tolerate missing information.

For example,

```text
Structure ✓
Text ✓
XRD ✗
SEM ✗
Properties ✓
```

The model should still be able to operate.

---

# 24.14.32 Masked-Modality Training

One strategy is to randomly hide modalities during training.

For example,

```text
Training Sample

Structure ✓
Text ✗
Properties ✓
XRD ✓
```

The model learns to operate with incomplete information.

During another training step,

```text
Structure ✗
Text ✓
Properties ✓
XRD ✗
```

This improves robustness.

---

# 24.14.33 Cross-Modal Prediction

Multimodal models can also predict one modality from another.

For example,

```text
Crystal Structure
      ↓
Predicted XRD
```

or

```text
Composition
      ↓
Predicted Crystal Representation
```

or

```text
Scientific Text
      ↓
Estimated Material Properties
```

These tasks can provide useful auxiliary training objectives.

---

# 24.14.34 Multitask Learning

A single model can perform several tasks simultaneously.

For example,

```text
Input:
Crystal Structure
```

Outputs:

```text
Band Gap
Formation Energy
Density
Magnetic Moment
Stability
```

The total loss can be written conceptually as

```text
Ltotal =
λ₁Lbandgap
+
λ₂Lformation
+
λ₃Ldensity
+
λ₄Lmagnetic
+
λ₅Lstability
```

The weights control the contribution of each task.

Multitask learning can encourage the model to learn a more general representation of materials.

---

# 24.14.35 Generative Multitask Learning

The same idea can be incorporated into generation.

The model may simultaneously learn

```text
Structure Generation
+
Property Prediction
+
Text Alignment
+
Composition Prediction
```

This creates a richer internal representation.

However, multitask systems are more difficult to train and require careful balancing of objectives.

---

# 24.14.36 Materials Foundation Models

The long-term goal of multimodal learning is a **materials foundation model**.

Such a model would be pretrained on very large and diverse collections of materials information.

Conceptually,

```text
Large Materials Corpus
        ↓
Pretraining
        ↓
Materials Foundation Model
        ↓
┌───────────────┬───────────────┬───────────────┐
↓               ↓               ↓
Prediction     Retrieval      Generation
```

The same pretrained representation could then be adapted to many downstream tasks.

---

# 24.14.37 Pretraining Data

A materials foundation model could potentially use

```text
Crystal Databases
DFT Databases
Scientific Papers
Patents
Experimental Measurements
Spectroscopy
Microscopy
Synthesis Records
```

The challenge is that these sources differ greatly in

```text
Quality
Scale
Format
Reliability
Terminology
```

Therefore, dataset curation becomes extremely important.

---

# 24.14.38 Data Quality

A large dataset is not necessarily a good dataset.

Materials data may contain

```text
Duplicate Structures
Incorrect Labels
Inconsistent Units
Different Calculation Methods
Experimental Noise
Missing Metadata
Contradictory Measurements
```

A foundation model trained on poorly curated data may learn these inconsistencies.

Therefore,

```text
Data Quality
```

is as important as

```text
Model Architecture
```

---

# 24.14.39 Data Provenance

Every scientific datum should ideally have provenance.

For example,

```text
Band Gap = 1.85 eV

Source:
DFT calculation

Method:
PBE

Dataset:
Database X

Version:
2026.1
```

or

```text
Band Gap = 1.92 eV

Source:
Experiment

Measurement:
UV–Vis

Temperature:
300 K
```

These two values should not automatically be treated as equivalent.

---

# 24.14.40 Computational and Experimental Labels

Materials models must distinguish between

```text
Predicted Property
```

and

```text
Measured Property
```

For example,

```text
DFT Band Gap
```

may systematically differ from

```text
Experimental Band Gap
```

depending on the material and computational method.

A multimodal model should preserve this distinction.

---

# 24.14.41 Uncertainty in Multimodal Models

Predictions should ideally include uncertainty.

For example,

```text
Predicted Band Gap

1.82 ± 0.15 eV
```

rather than simply

```text
1.82 eV
```

Uncertainty is particularly important when generated candidates are selected for expensive experiments.

---

# 24.14.42 Human-in-the-Loop Generation

A powerful workflow allows the researcher to interact with the generative model.

For example,

```text
Researcher
   ↓
Natural-Language Objective
   ↓
AI Generates Candidates
   ↓
Researcher Reviews
   ↓
Adjusts Constraints
   ↓
AI Generates Again
```

The human remains inside the discovery loop.

This can be especially valuable when the objective contains scientific knowledge that is difficult to formalize mathematically.

---

# 24.14.43 Natural-Language Materials Search

A multimodal materials model could support queries such as

```text
Find stable oxide materials with
band gaps between 1.5 and 2.0 eV.
```

or

```text
Find layered materials with strong
ionic conductivity and low density.
```

or

```text
Generate a semiconductor that is
stable at high temperature and does
not contain toxic heavy metals.
```

The system translates the natural-language objective into structured computational constraints.

---

# 24.14.44 Natural Language to Computational Constraints

Conceptually,

```text
Natural Language
       ↓
Language Model
       ↓
Structured Constraints
       ↓
Materials Generator
```

For example,

```text
"high thermal stability"
```

might be mapped to an application-specific criterion such as

```text
decomposition temperature > threshold
```

while

```text
"low density"
```

might become

```text
density < threshold
```

The exact mapping must be scientifically defined rather than blindly inferred by the language model.

---

# 24.14.45 Multimodal Candidate Ranking

Generated materials can be ranked using multiple information sources.

For example,

```text
Candidate
   │
   ├── Predicted Properties
   ├── Stability
   ├── Structure Quality
   ├── Textual Scientific Evidence
   ├── Experimental Similarity
   └── Synthesis Feasibility
```

A multimodal ranking model can combine these signals.

This allows candidate selection to move beyond a single numerical property.

---

# 24.14.46 Retrieval-Augmented Materials AI

A language model should not be expected to remember every materials-science fact.

Instead, it can retrieve relevant scientific information from external databases or literature.

Conceptually,

```text
Research Question
       ↓
Information Retrieval
       ↓
Relevant Papers / Materials
       ↓
Language Model
       ↓
Scientific Response
```

This is commonly referred to as **retrieval-augmented generation**.

For Materials Informatics, retrieval can connect generative AI with continuously updated scientific databases.

---

# 24.14.47 Retrieval Before Generation

A particularly useful strategy is

```text
User Objective
      ↓
Retrieve Known Materials
      ↓
Learn Existing Design Space
      ↓
Generate New Candidates
      ↓
Compare with Retrieved Materials
```

This reduces the risk of generating candidates that are already known.

It also provides a meaningful baseline for evaluating novelty.

---

# 24.14.48 Literature-Guided Generation

Scientific literature can provide constraints that are difficult to encode numerically.

For example, papers may indicate that

```text
A particular structure is unstable above a temperature.
```

or

```text
A particular synthesis route requires oxygen-rich conditions.
```

Such information can potentially guide generation.

The resulting workflow becomes

```text
Literature
   ↓
Knowledge Extraction
   ↓
Constraints
   ↓
Generative Model
```

---

# 24.14.49 From Generative Model to Materials Copilot

Combining the concepts developed throughout this chapter suggests a broader system.

A materials AI assistant could

```text
Read Literature
      ↓
Understand Scientific Objective
      ↓
Search Materials Databases
      ↓
Generate Candidate Structures
      ↓
Predict Properties
      ↓
Check Stability
      ↓
Estimate Synthesis Feasibility
      ↓
Rank Candidates
      ↓
Prepare DFT Calculations
      ↓
Interpret Results
      ↓
Suggest Experiments
```

Such a system would function as a **Materials Research Copilot**.

---

# 24.14.50 Important Limitation

Despite these capabilities, multimodal generative AI should not be treated as an autonomous source of scientific truth.

A generated structure may be

```text
Mathematically Valid
```

but

```text
Physically Implausible
```

A language model may produce

```text
Scientifically Plausible Text
```

that is nevertheless incorrect.

A predicted property may have

```text
Large Uncertainty
```

Therefore,

```text
AI Generation
≠
Scientific Validation
```

The correct workflow remains

```text
AI
 ↓
Hypothesis
 ↓
Physics
 ↓
Simulation
 ↓
Experiment
 ↓
Validation
```

---

# 24.14.51 The Future of Multimodal Materials Generation

The long-term direction can be summarized as

```text
                    Materials AI
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   Structures          Text          Experiments
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                  Shared Representation
                         ↓
                Generative Foundation Model
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   New Structures    New Properties   New Designs
                         │
                         ↓
                   DFT / Experiment
                         │
                         ↓
                    New Knowledge
                         ↺
```

The ultimate objective is not simply to generate more structures.

It is to build AI systems capable of reasoning across the different representations used by materials scientists.

---

# 24.14.52 Summary

This section introduced multimodal generative AI as a natural extension of crystal-generation models.

The major concepts were:

```text
Multimodal Materials Representation
```

```text
Structure–Text Alignment
```

```text
Composition Embeddings
```

```text
Property Embeddings
```

```text
Scientific Language Models
```

```text
Contrastive Learning
```

```text
Shared Latent Spaces
```

```text
Cross-Attention
```

```text
Multimodal Diffusion
```

```text
Experimental Data Fusion
```

```text
XRD and Spectroscopy Integration
```

```text
Materials Foundation Models
```

```text
Retrieval-Augmented Materials AI
```

```text
Natural-Language Materials Design
```

The conceptual evolution is therefore

```text
Crystal Generation
        ↓
Property-Conditioned Generation
        ↓
Synthesis-Aware Generation
        ↓
Closed-Loop Materials Discovery
        ↓
Multimodal Materials AI
        ↓
Materials Foundation Models
        ↓
Autonomous Scientific Discovery
```

The next stage should move from multimodal representation toward the **complete research implementation framework**: how these generative systems are actually trained, evaluated, benchmarked, reproduced, and connected to DFT and experimental validation in a real Materials Informatics research project.

# 24.15 Training and Evaluation of Generative Materials Models

The previous sections established the conceptual progression from crystal generation to multimodal and synthesis-aware materials discovery.

The complete framework can now be viewed as

```text
Materials Data
      ↓
Representation
      ↓
Generative Model
      ↓
Candidate Generation
      ↓
Validity Checking
      ↓
Property Prediction
      ↓
Stability Analysis
      ↓
Synthesis Feasibility
      ↓
DFT Validation
      ↓
Experimental Validation
      ↓
New Data
      ↺
```

However, constructing such a system requires more than selecting a generative architecture.

A research-grade model must be

```text
Correctly Trained
Properly Validated
Scientifically Benchmarked
Reproducible
Physically Interpretable
Computationally Efficient
```

This section therefore focuses on the practical methodology required to train and evaluate generative models for materials discovery.

---

# 24.15.1 Why Evaluation Is Difficult

Traditional machine-learning problems often have a straightforward evaluation metric.

For example,

```text
Classification
    ↓
Accuracy

Regression
    ↓
MAE / RMSE / R²
```

Generative materials models are more complicated.

A generated crystal can be

```text
Structurally Valid
```

but not

```text
Novel
```

or

```text
Novel
```

but not

```text
Stable
```

or

```text
Stable
```

but not

```text
Target-Satisfying
```

or

```text
Target-Satisfying
```

but experimentally impossible.

Therefore, no single metric is sufficient.

---

# 24.15.2 Multiple Evaluation Dimensions

A useful evaluation framework should consider

```text
Validity
Novelty
Uniqueness
Diversity
Property Accuracy
Stability
Condition Satisfaction
Distribution Matching
Synthesis Feasibility
Computational Cost
```

These dimensions answer different scientific questions.

---

# 24.15.3 Validity

The first question is:

```text
Is the generated structure actually a valid crystal?
```

Validity may include

```text
Valid Composition
Valid Lattice
Valid Atomic Coordinates
No Severe Atomic Overlap
Reasonable Density
Periodic Consistency
Chemical Plausibility
```

A simple validity ratio can be defined as

```text
Validity Rate =
Number of Valid Structures
/
Total Generated Structures
```

For example, if

```text
10,000
```

structures are generated and

```text
8,700
```

pass the validity checks,

then

```text
Validity Rate = 0.87
```

or

```text
87%
```

---

# 24.15.4 Structural Validity

A crystal-generation pipeline should first verify the structure itself.

For example,

```python
def basic_structure_check(structure):

    if structure is None:
        return False

    if len(structure) == 0:
        return False

    if structure.lattice.volume <= 0:
        return False

    return True
```

This is only a minimal check.

A real implementation should perform substantially more detailed validation.

---

# 24.15.5 Interatomic Distance Validation

Generated structures may contain unrealistic atomic overlaps.

For example,

```text
Atom A
   ↓
0.05 Å
   ↓
Atom B
```

would generally indicate an invalid structure.

A minimum-distance criterion can therefore be applied.

```python
def check_distances(
    structure,
    minimum_distance
):

    distances = structure.distance_matrix

    for i in range(len(structure)):

        for j in range(i + 1, len(structure)):

            if distances[i, j] < minimum_distance:
                return False

    return True
```

The exact threshold should be chemically meaningful rather than arbitrarily chosen.

---

# 24.15.6 Chemical Validity

Structural geometry alone is insufficient.

A generated material may have

```text
Impossible Composition
```

or

```text
Unreasonable Oxidation States
```

Therefore, chemical analysis should be included.

A possible workflow is

```text
Generated Structure
      ↓
Composition Analysis
      ↓
Oxidation-State Analysis
      ↓
Coordination Analysis
      ↓
Chemical Plausibility
```

Such checks should be regarded as screening tools rather than absolute proofs of physical existence.

---

# 24.15.7 Uniqueness

Suppose a model generates

```text
10,000
```

structures.

If only

```text
200
```

are unique, then the generator is repeatedly producing the same structures.

The uniqueness ratio can be defined conceptually as

```text
Uniqueness =
Number of Unique Structures
/
Number of Valid Structures
```

High uniqueness is generally desirable.

However, uniqueness alone does not guarantee useful diversity.

---

# 24.15.8 Duplicate Detection

Duplicate detection should ideally use structure-aware representations.

Possible approaches include

```text
Composition
Canonical Structure Representation
Structure Fingerprints
Graph Representations
Local Environment Descriptors
```

A simplistic composition-based approach is

```python
seen = set()

unique = []

for structure in candidates:

    key = structure.composition.reduced_formula

    if key not in seen:

        seen.add(key)
        unique.append(structure)
```

However, two structurally different materials can have the same composition.

Therefore, composition alone is not sufficient for scientific deduplication.

---

# 24.15.9 Structure Fingerprints

A stronger method represents a structure using a fingerprint.

Conceptually,

```text
Crystal
   ↓
Structure Fingerprint
   ↓
Vector
```

Two structures can then be compared using a similarity metric.

For example,

```python
similarity = compare_fingerprints(
    fingerprint_a,
    fingerprint_b
)
```

A threshold can then determine whether two candidates should be considered effectively identical.

---

# 24.15.10 Novelty

Novelty asks:

```text
Is the generated material genuinely new?
```

This requires comparison with known materials.

The workflow becomes

```text
Generated Structures
       ↓
Known Materials Database
       ↓
Structural Comparison
       ↓
Novelty Classification
```

A structure that differs only slightly from an existing material should not automatically be described as completely novel.

---

# 24.15.11 Database-Based Novelty Checking

Suppose a generated candidate is compared against

```text
Materials Project
ICSD
OQMD
JARVIS
AFLOW
```

or another appropriate reference collection.

The candidate can be assigned a similarity score to its nearest known structure.

Conceptually,

```text
Novelty Score =
1 − Maximum Similarity
```

The exact metric depends on the representation.

---

# 24.15.12 Novelty Is Not the Same as Value

A highly novel material is not necessarily useful.

Consider

```text
Candidate A
Novelty = High
Stability = Very Low
```

and

```text
Candidate B
Novelty = Moderate
Stability = High
Target Property = Excellent
```

Candidate B may be scientifically more valuable.

Therefore,

```text
Novelty
```

should be treated as one evaluation dimension rather than the final objective.

---

# 24.15.13 Diversity

Diversity measures how broadly the generator explores the materials space.

For example,

```text
1,000 structures
```

could all belong to the same structural family.

The model may therefore have high uniqueness but low diversity.

Diversity can be measured in spaces such as

```text
Composition Space
Structure Fingerprint Space
Graph Embedding Space
Latent Space
Property Space
```

---

# 24.15.14 Composition Diversity

A simple composition-diversity analysis can examine the elemental combinations present in the generated set.

For example,

```text
Oxides
Sulfides
Halides
Nitrides
Intermetallics
```

can be counted separately.

This gives a coarse view of chemical exploration.

---

# 24.15.15 Latent-Space Diversity

A learned embedding can provide a more sophisticated measure.

Suppose the generator produces embeddings

```text
z₁, z₂, ..., zₙ
```

Pairwise distances can be computed.

```python
distance = torch.cdist(
    embeddings,
    embeddings
)
```

A high average pairwise distance indicates broader exploration in the selected representation space.

However, diversity depends strongly on the representation used.

---

# 24.15.16 Property Accuracy

For conditional generation, the most important question may be:

```text
Did the generated materials actually satisfy the requested property?
```

Suppose the target is

```text
Band Gap = 2.0 eV
```

and the generated candidates have predicted values

```text
1.9
2.1
2.0
2.4
1.8
```

The model can be evaluated using the error between target and predicted property.

For a candidate,

```text
Property Error =
|Predicted Property − Target Property|
```

---

# 24.15.17 Condition Satisfaction Rate

A useful metric is

```text
Condition Satisfaction Rate =
Number of Candidates Satisfying Target
/
Number of Valid Candidates
```

For example,

```text
Target:
1.5 ≤ Eg ≤ 2.0 eV
```

If

```text
6,000
```

of

```text
8,000
```

valid structures satisfy the condition,

then

```text
Satisfaction Rate = 75%
```

This is often more informative than reporting only the mean property value.

---

# 24.15.18 Property Error Distribution

The entire error distribution should be analyzed.

For example,

```text
Target = 2.0 eV

Errors:
0.02
0.08
0.15
0.30
0.45
...
```

A histogram or cumulative distribution can show how strongly the generator concentrates around the requested property.

The evaluation should therefore consider both

```text
Mean Error
```

and

```text
Tail Behavior
```

---

# 24.15.19 Multi-Objective Satisfaction

Suppose the target requires

```text
1.5 ≤ Band Gap ≤ 2.0 eV

Energy Above Hull < 0.05 eV/atom

Bulk Modulus > 200 GPa
```

A candidate satisfies the design objective only if all conditions are satisfied.

```python
mask = (
    (band_gap >= 1.5)
    &
    (band_gap <= 2.0)
    &
    (energy_above_hull < 0.05)
    &
    (bulk_modulus > 200)
)
```

The fraction of candidates satisfying the complete set provides a useful measure of multi-objective generation performance.

---

# 24.15.20 Distribution Matching

A generative model should not only produce valid individual structures.

It should also learn the underlying distribution of the training data.

Suppose the training dataset contains

```text
Crystal Systems
```

distributed approximately as

```text
Cubic        30%
Orthorhombic 25%
Tetragonal   20%
Hexagonal    15%
Other        10%
```

A generator producing

```text
Cubic = 95%
```

has clearly failed to reproduce the broader distribution.

---

# 24.15.21 Distribution Comparison

Different statistical methods can be used to compare generated and reference distributions.

Possible metrics include

```text
KL Divergence
Jensen–Shannon Divergence
Wasserstein Distance
Maximum Mean Discrepancy
```

The appropriate metric depends on the representation and scientific question.

---

# 24.15.22 Property Distribution Matching

The same idea can be applied to material properties.

For example,

```text
Training Band-Gap Distribution
```

can be compared with

```text
Generated Band-Gap Distribution
```

A generator should reproduce relevant features of the reference distribution while still producing novel candidates.

This creates an important balance between

```text
Distribution Fidelity
```

and

```text
Novelty
```

---

# 24.15.23 Precision and Recall for Generative Models

Generative-model evaluation can also use concepts analogous to precision and recall.

Conceptually,

```text
Precision
```

asks:

```text
How many generated samples resemble the desired data distribution?
```

while

```text
Recall
```

asks:

```text
How much of the relevant data distribution does the model cover?
```

A model with high precision but low recall may generate realistic but narrow samples.

A model with high recall but low precision may explore broadly but generate many unrealistic structures.

---

# 24.15.24 The Diversity–Quality Trade-Off

Generative modeling often involves a trade-off.

Increasing guidance may improve

```text
Target Property Accuracy
```

but reduce

```text
Diversity
```

Weak guidance may produce

```text
High Diversity
```

but poor

```text
Condition Satisfaction
```

Therefore, model evaluation should consider both.

Conceptually,

```text
             Property Accuracy
                    ↑
                    │
               ●    │
            ●       │
         ●          │
      ●             │
────────────────────────────→
          Diversity
```

The preferred operating point depends on the scientific objective.

---

# 24.15.25 Pareto Evaluation of Generative Models

The same concept of Pareto optimality introduced earlier can be applied to model evaluation.

For example, compare models according to

```text
Validity
Diversity
Condition Satisfaction
```

A model may be considered better if it improves one metric without degrading the others.

This produces a model-level Pareto frontier.

---

# 24.15.26 Baseline Models

A new generative architecture should be compared against simpler approaches.

Possible baselines include

```text
Random Sampling
Nearest-Neighbor Retrieval
Variational Autoencoder
GAN
Unconditional Diffusion
Conditional Diffusion
Evolutionary Search
Bayesian Optimization
```

The baseline should be scientifically reasonable.

A sophisticated model should not be compared only against a deliberately weak method.

---

# 24.15.27 Ablation Studies

Ablation studies determine which components actually contribute to performance.

Suppose the proposed model contains

```text
Structure Encoder
Property Conditioning
Classifier-Free Guidance
Chemical Constraints
Stability Predictor
```

Ablation experiments can remove one component at a time.

```text
Full Model

Full − Property Conditioning

Full − Guidance

Full − Chemical Constraints

Full − Stability Predictor
```

The resulting performance differences show the contribution of each component.

---

# 24.15.28 Why Ablation Studies Matter

Suppose the full model achieves

```text
85% Condition Satisfaction
```

Removing the stability predictor gives

```text
84%
```

while removing property conditioning gives

```text
52%
```

This indicates that property conditioning is substantially more important than the stability predictor for the selected objective.

Without ablation studies, such conclusions cannot be established reliably.

---

# 24.15.29 Random Seeds

Generative models are stochastic.

Different random seeds can produce different results.

Therefore, a single run is insufficient for strong scientific claims.

For example,

```python
seeds = [
    0,
    1,
    2,
    3,
    4
]
```

The experiment should be repeated for each seed.

---

# 24.15.30 Reporting Mean and Variance

Suppose five experiments produce validity rates

```text
0.81
0.84
0.83
0.80
0.85
```

The final report should ideally include

```text
Mean Validity
+
Standard Deviation
```

rather than reporting only

```text
0.85
```

This gives a more reliable estimate of model performance.

---

# 24.15.31 Training–Validation–Test Splits

Generative models also require careful dataset splitting.

A simple approach is

```text
Dataset
   ↓
Training
Validation
Test
```

For example,

```text
70% Training
15% Validation
15% Test
```

However, random splitting may be inappropriate for materials datasets.

---

# 24.15.32 Composition-Based Splits

Suppose structurally similar materials appear in both training and test sets.

The model may appear to generalize well simply because it has already seen extremely similar examples.

A stronger evaluation can hold out entire chemical systems.

For example,

```text
Training:
Li-Fe-O
Na-Fe-O
K-Fe-O

Test:
Li-Co-O
```

This provides a more difficult test of chemical generalization.

---

# 24.15.33 Structure-Based Splits

Another strategy is to hold out structural families.

For example,

```text
Training:
Perovskites
Spinels
Rocksalt

Test:
Layered Structures
```

This evaluates whether the model can extrapolate beyond the dominant structural patterns in the training data.

---

# 24.15.34 Time-Based Splits

For literature-driven systems, a temporal split may be more realistic.

For example,

```text
Older Papers
     ↓
Training

Recent Papers
     ↓
Test
```

This simulates a realistic scientific scenario:

```text
Learn from existing knowledge
       ↓
Predict future discoveries
```

---

# 24.15.35 Data Leakage

Data leakage is a major concern.

Suppose a generated candidate appears in the training dataset under a slightly different representation.

The model may appear to discover a new material even though it has effectively memorized it.

Therefore, novelty evaluation should account for

```text
Exact Duplicates
Structural Near-Duplicates
Composition Matches
Database Overlap
```

---

# 24.15.36 Memorization vs Generalization

A generative model can produce impressive-looking results through memorization.

For example,

```text
Training Materials
       ↓
Memorization
       ↓
Near-Identical Generated Structures
```

This is not equivalent to discovering new regions of materials space.

A strong model should demonstrate both

```text
Distribution Learning
```

and

```text
Generalization
```

---

# 24.15.37 Training the Generative Model

A simplified training loop for a diffusion model may look like

```python
for batch in dataloader:

    crystal = batch["crystal"]

    condition = batch["properties"]

    t = sample_timesteps(
        crystal.shape[0]
    )

    noise = torch.randn_like(
        crystal
    )

    noisy_crystal = q_sample(
        crystal,
        t,
        noise
    )

    predicted_noise = model(
        noisy_crystal,
        t,
        condition
    )

    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

This is a simplified illustration.

Real crystal-diffusion systems require carefully designed representations and equivariant architectures.

---

# 24.15.38 Validation During Training

Validation should occur throughout training.

For example,

```python
for epoch in range(
    num_epochs
):

    train_one_epoch()

    validation_loss = (
        evaluate_validation_set()
    )

    print(
        epoch,
        validation_loss
    )
```

However, validation loss alone is insufficient for generative materials models.

Generated samples should also be periodically evaluated.

---

# 24.15.39 Periodic Sampling

During training, the researcher can generate a fixed number of structures.

```python
samples = sample_model(
    model,
    num_samples=100
)
```

These can then be evaluated for

```text
Validity
Diversity
Novelty
Property Accuracy
```

Tracking these quantities across training can reveal whether the model is actually improving.

---

# 24.15.40 Early Stopping

If validation performance stops improving, training can be stopped.

For example,

```python
if validation_loss < best_loss:

    best_loss = validation_loss
    save_checkpoint(model)

else:

    patience_counter += 1
```

Early stopping can reduce overfitting and unnecessary computation.

---

# 24.15.41 Checkpointing

Important checkpoints should be saved.

```python
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    },
    "checkpoint.pt"
)
```

This allows the experiment to be resumed and provides a reproducible model state.

---

# 24.15.42 Experiment Tracking

A research project should track

```text
Experiment ID
Dataset Version
Model Version
Hyperparameters
Random Seed
Training Loss
Validation Metrics
Generated Sample Statistics
Hardware
Software Version
```

Tools such as experiment-tracking platforms can automate much of this process.

The important principle is not the specific tool but the preservation of experimental history.

---

# 24.15.43 Hardware Considerations

Generative materials models can be computationally expensive.

Training may require

```text
GPU
High Memory
Fast Storage
Parallel Data Loading
```

Crystal structures may also contain variable numbers of atoms, making batching more complicated than ordinary fixed-size tensors.

Efficient graph batching and memory management are therefore important.

---

# 24.15.44 Computational Scaling

Suppose generation of one structure takes

```text
0.1 seconds
```

Then generating

```text
1,000,000
```

structures would require approximately

```text
100,000 seconds
```

on a single processing stream.

This demonstrates why efficient generation becomes important when large candidate spaces are explored.

Parallel generation can significantly reduce wall-clock time.

---

# 24.15.45 Candidate Budget

The number of generated candidates should be treated as part of the experimental design.

For example,

```text
10³ candidates
10⁴ candidates
10⁵ candidates
10⁶ candidates
```

may produce very different discovery outcomes.

A model that performs well with one million samples but poorly with ten thousand may not be computationally practical.

Therefore, evaluation should report performance as a function of generation budget.

---

# 24.15.46 Sample Efficiency

An important question is:

```text
How many generated candidates are required
to discover one excellent material?
```

Suppose

```text
100 candidates
→ 1 promising material
```

versus

```text
10,000 candidates
→ 1 promising material
```

The first system may be much more efficient.

This motivates the concept of **sample efficiency** in generative materials discovery.

---

# 24.15.47 Discovery Efficiency

A useful practical metric is the number of high-quality candidates discovered per computational budget.

Conceptually,

```text
Discovery Efficiency =
Number of Promising Candidates
/
Computational Cost
```

The cost may include

```text
Generation Time
Property Prediction
DFT Calculations
Experimental Measurements
```

This is more meaningful than generation quality alone.

---

# 24.15.48 DFT Validation

Machine-learning predictions should eventually be tested using higher-fidelity calculations.

A typical pipeline is

```text
Generated Structure
      ↓
ML Screening
      ↓
Top Candidates
      ↓
DFT Geometry Optimization
      ↓
Electronic / Mechanical / Magnetic Calculations
      ↓
Validated Properties
```

The DFT stage acts as a higher-fidelity filter.

---

# 24.15.49 ML-to-DFT Consistency

Suppose the ML model predicts

```text
Band Gap = 2.0 eV
```

but DFT optimization produces

```text
Band Gap = 1.2 eV
```

The discrepancy should be analyzed.

Possible causes include

```text
Model Error
Distribution Shift
Structural Relaxation
Representation Limitations
DFT Method Dependence
```

Such disagreements are scientifically valuable because they reveal weaknesses in the model.

---

# 24.15.50 Geometry Relaxation

Generated structures may not initially be at an equilibrium geometry.

Therefore,

```text
Generated Crystal
      ↓
DFT Relaxation
      ↓
Relaxed Crystal
      ↓
Property Calculation
```

is often necessary.

A generated structure that changes dramatically during relaxation may indicate that the original structure was physically implausible or unstable.

---

# 24.15.51 Structural Drift

Define a structural difference between

```text
Generated Structure
```

and

```text
DFT-Relaxed Structure
```

A large structural change can indicate poor generation quality.

Therefore, one useful evaluation quantity is

```text
Structural Drift =
Distance(
    Generated,
    Relaxed
)
```

The exact metric should account for periodicity and atom correspondence.

---

# 24.15.52 DFT Validation as an Oracle

In an active-learning framework, DFT can be treated as an expensive oracle.

```text
AI Generator
      ↓
Candidate
      ↓
DFT Oracle
      ↓
Accurate Label
      ↓
Training Dataset
```

Only a small number of candidates need to be evaluated with DFT.

This dramatically reduces computational cost compared with calculating every generated structure.

---

# 24.15.53 Active Learning

The model can choose which candidates are most valuable to evaluate.

For example, it may select candidates with

```text
High Predicted Performance
```

or

```text
High Uncertainty
```

or

```text
High Diversity
```

or a combination of these.

The workflow becomes

```text
Generate
   ↓
Predict
   ↓
Select Informative Candidates
   ↓
DFT
   ↓
Update Model
   ↺
```

---

# 24.15.54 Exploitation vs Exploration

Active learning contains a fundamental trade-off.

### Exploitation

Choose materials predicted to be excellent.

```text
Find the best known region.
```

### Exploration

Choose materials about which the model is uncertain.

```text
Discover new regions.
```

A useful strategy balances both.

```text
Selection Score
=
Performance
+
Uncertainty
+
Diversity
```

The exact weighting depends on the research objective.

---

# 24.15.55 Uncertainty Sampling

Suppose three generated candidates have predictions

```text
Candidate A:
2.0 ± 0.02 eV

Candidate B:
2.1 ± 0.08 eV

Candidate C:
2.0 ± 0.50 eV
```

Candidate C has substantially greater uncertainty.

Evaluating it with DFT may provide more information to the model.

This is one motivation for uncertainty-based active learning.

---

# 24.15.56 Diversity-Aware Active Learning

Selecting only high-performing candidates may cause the dataset to become concentrated in one region.

Therefore, candidate selection can include diversity.

For example,

```text
Top Property Candidates
        +
Uncertain Candidates
        +
Diverse Candidates
```

This produces a more informative DFT batch.

---

# 24.15.57 Batch Active Learning

In practical research, DFT calculations are often performed in batches.

For example,

```text
Generate 100,000
       ↓
Select 100
       ↓
DFT
       ↓
Add 100 labels
       ↓
Retrain
```

The process repeats.

This is more realistic than retraining after every individual calculation.

---

# 24.15.58 Closed-Loop Training

The complete active-learning cycle becomes

```text
Initial Dataset
      ↓
Train Generative Model
      ↓
Generate Candidates
      ↓
ML Screening
      ↓
Select Candidates
      ↓
DFT / Experiment
      ↓
Obtain New Labels
      ↓
Expand Dataset
      ↓
Retrain
      ↺
```

This creates a self-improving materials-discovery system.

---

# 24.15.59 Research Benchmark Table

A final study should ideally compare models using a table such as

| Model                 | Validity | Novelty | Diversity | Condition Satisfaction | Stability | DFT Success |
| --------------------- | -------: | ------: | --------: | ---------------------: | --------: | ----------: |
| VAE                   |        — |       — |         — |                      — |         — |           — |
| GAN                   |        — |       — |         — |                      — |         — |           — |
| Diffusion             |        — |       — |         — |                      — |         — |           — |
| Conditional Diffusion |        — |       — |         — |                      — |         — |           — |
| Proposed Model        |        — |       — |         — |                      — |         — |           — |

The values must come from the actual experiment.

They should never be invented simply to make the proposed method appear superior.

---

# 24.15.60 Reproducible Evaluation Protocol

A strong research protocol should specify:

```text
Dataset
↓
Preprocessing
↓
Train/Validation/Test Split
↓
Model Configuration
↓
Random Seeds
↓
Training Procedure
↓
Sampling Procedure
↓
Validity Checks
↓
Novelty Evaluation
↓
Property Evaluation
↓
Stability Evaluation
↓
DFT Validation
```

Every step should be documented.

---

# 24.15.61 A Complete Experimental Configuration

A configuration file may contain

```python
config = {

    "dataset":
        "materials_dataset_v1",

    "model":
        "conditional_diffusion",

    "epochs":
        500,

    "batch_size":
        64,

    "learning_rate":
        1e-4,

    "num_samples":
        100000,

    "random_seed":
        42,

    "property_condition":
        [
            "band_gap",
            "formation_energy"
        ],

    "guidance_scale":
        2.0
}
```

Keeping configuration separate from source code improves reproducibility.

---

# 24.15.62 End-to-End Research Implementation

A simplified high-level implementation may look like

```python
# 1. Load dataset

dataset = load_materials_dataset(
    "materials.csv"
)


# 2. Preprocess structures

structures = preprocess_structures(
    dataset
)


# 3. Extract target properties

properties = extract_properties(
    dataset
)


# 4. Train generative model

model = train_diffusion_model(
    structures,
    properties
)


# 5. Generate candidates

candidates = generate_candidates(
    model,
    target_properties,
    num_samples=100000
)


# 6. Validate

valid = validate_structures(
    candidates
)


# 7. Remove duplicates

unique = deduplicate(
    valid
)


# 8. Predict properties

predictions = predictor(
    unique
)


# 9. Rank

selected = rank_candidates(
    unique,
    predictions
)


# 10. DFT

results = run_dft(
    selected
)


# 11. Update dataset

dataset = update_dataset(
    dataset,
    results
)
```

This simplified code captures the structure of an actual research workflow.

---

# 24.15.63 The Difference Between a Demonstration and Research

A short notebook that generates a few crystals is a useful demonstration.

It is not automatically a research framework.

A research implementation requires

```text
Controlled Dataset
+
Reproducible Training
+
Quantitative Evaluation
+
Baselines
+
Ablation Studies
+
Uncertainty Analysis
+
High-Fidelity Validation
```

This distinction is critical for scientific machine learning.

---

# 24.15.64 Common Failure Modes

Generative materials projects can fail in several ways.

### Failure 1: Invalid Structures

```text
Generator
↓
Unphysical Geometry
```

### Failure 2: Memorization

```text
Generator
↓
Known Structures
```

### Failure 3: Mode Collapse

```text
Many Samples
↓
Few Distinct Structures
```

### Failure 4: Poor Conditioning

```text
Target = 2 eV
↓
Generated Mean = 0.8 eV
```

### Failure 5: Distribution Shift

```text
Training Distribution
↓
Very Different Generated Structures
↓
Poor Predictor Accuracy
```

### Failure 6: False Scientific Confidence

```text
AI Prediction
↓
Assumed to be True
```

These failures must be explicitly tested.

---

# 24.15.65 Failure Analysis

When a model fails, the objective should not simply be to report the failure.

The failure should be analyzed.

For example,

```text
Invalid Structures
        ↓
Distance Analysis
        ↓
Chemical Analysis
        ↓
Model Representation
        ↓
Identify Failure Mechanism
```

This can lead directly to architectural improvements.

---

# 24.15.66 Scientific Debugging

Scientific machine learning requires two types of debugging.

### Software Debugging

```text
Tensor Shape
GPU Memory
NaN Values
Data Loading
Gradient Errors
```

### Scientific Debugging

```text
Physical Plausibility
Data Leakage
Incorrect Units
Invalid Labels
Distribution Shift
Incorrect Evaluation
```

A model can be computationally correct but scientifically wrong.

Both forms of debugging are therefore essential.

---

# 24.15.67 Units and Physical Scales

Materials datasets often combine quantities with very different scales.

For example,

```text
Band Gap:
eV

Density:
g/cm³

Bulk Modulus:
GPa

Formation Energy:
eV/atom
```

These should be handled carefully.

Normalization should be performed using training-set statistics.

```python
scaler.fit(
    training_properties
)

train_scaled = scaler.transform(
    training_properties
)

validation_scaled = scaler.transform(
    validation_properties
)

test_scaled = scaler.transform(
    test_properties
)
```

The validation and test sets must not influence the fitted scaler.

---

# 24.15.68 Evaluation Without Data Leakage

A critical rule is:

```text
Never use test-set information
to train or tune the model.
```

For example, the following is incorrect:

```python
scaler.fit(
    all_properties
)
```

if `all_properties` includes the test set.

Instead,

```python
scaler.fit(
    train_properties
)
```

must be used.

The same principle applies to

```text
Feature Selection
Hyperparameter Optimization
Threshold Selection
Model Selection
```

---

# 24.15.69 Hyperparameter Optimization

Generative models can contain many hyperparameters.

Examples include

```text
Learning Rate
Batch Size
Hidden Dimension
Number of Layers
Noise Schedule
Diffusion Steps
Guidance Strength
Embedding Dimension
Dropout
Weight Decay
```

A systematic search can be performed.

```text
Configuration
      ↓
Training
      ↓
Validation
      ↓
Evaluation
      ↓
Next Configuration
```

However, optimization should be performed using the validation set rather than the test set.

---

# 24.15.70 Final Test Evaluation

The test set should ideally be used only after

```text
Architecture Selection
+
Hyperparameter Selection
+
Training Decisions
```

have been completed.

The final test evaluation then provides an approximately unbiased estimate of generalization performance.

---

# 24.15.71 Reporting Computational Cost

Research papers should report computational requirements.

For example,

```text
GPU:
NVIDIA GPU

Training Time:
X hours

Generation:
Y structures/second

DFT:
Z CPU/GPU hours
```

This allows other researchers to evaluate whether the method is practically reproducible.

---

# 24.15.72 Carbon and Resource Considerations

Large generative models can require substantial computational resources.

Therefore, responsible Materials Informatics research should consider

```text
Energy Consumption
Compute Time
Hardware Requirements
Number of DFT Calculations
```

A method that achieves only a small improvement at enormous computational cost may not be practically preferable.

---

# 24.15.73 Final Research Workflow

The complete research-grade pipeline developed throughout the chapter can now be summarized as

```text
                  Scientific Objective
                          ↓
                  Materials Dataset
                          ↓
              Structure Representation
                          ↓
               Generative Model Training
                          ↓
                Conditional Generation
                          ↓
                  Validity Filtering
                          ↓
                   Novelty Checking
                          ↓
                  Diversity Analysis
                          ↓
                 Property Prediction
                          ↓
                  Stability Screening
                          ↓
               Synthesis Feasibility
                          ↓
                    Candidate Ranking
                          ↓
                     Active Learning
                          ↓
                         DFT
                          ↓
                    Experiment
                          ↓
                  New Experimental Data
                          ↓
                    Dataset Expansion
                          ↓
                     Model Retraining
                          ↺
```

This is no longer merely a generative-AI workflow.

It is a complete **Materials Informatics discovery framework**.

---

# 24.15.74 Final Perspective

The central lesson of generative materials research is that generation itself is only one component of discovery.

A model that produces thousands of crystals is not necessarily useful.

A useful system must produce candidates that are

```text
Valid
Novel
Diverse
Property-Relevant
Stable
Synthesizable
Experimentally Testable
```

The ultimate scientific objective is therefore

```text
Generate
   +
Predict
   +
Validate
   +
Learn
```

repeated continuously.

The most powerful future systems will combine

```text
Generative AI
+
Materials Science
+
Physics-Based Simulation
+
Multimodal Learning
+
Active Learning
+
Automation
```

to create increasingly autonomous discovery pipelines.

The researcher remains responsible for defining meaningful scientific questions, choosing appropriate constraints, validating computational results, and interpreting physical behavior.

AI provides the ability to explore enormous design spaces that would be impossible to search manually.

The resulting paradigm can be summarized as

```text
Human Scientific Reasoning
             +
       Generative AI
             +
      Physics Simulation
             +
         Experiment
             ↓
       New Materials
             ↓
       New Knowledge
             ↺
```

This completes the transition from **machine-learning-assisted materials prediction** to **AI-driven materials discovery**.



## 24.16 — From Generative Models to Scientific Discovery

The generative models developed throughout this chapter provide a mechanism for exploring a space of possible materials.

However, generation alone does not constitute discovery.

Suppose a model generates

```text
10,000 crystal structures
```

The existence of these structures is not yet a scientific result.

The researcher still needs to determine

```text
Which structures are physically meaningful?

Which structures are genuinely new?

Which structures satisfy the scientific objective?

Which structures are sufficiently stable?

Which structures can actually be synthesized?

Which structures are worth evaluating with DFT?
```

This distinction is fundamental.

A generative model produces **possibilities**.

Materials science determines **meaning**.

Therefore, a complete generative-discovery system must contain several layers of scientific reasoning.

```text
                    Scientific Objective
                           ↓
                    Candidate Generation
                           ↓
                    Structural Validation
                           ↓
                    Chemical Validation
                           ↓
                    Property Prediction
                           ↓
                    Stability Evaluation
                           ↓
                    Candidate Prioritization
                           ↓
                         DFT
                           ↓
                      Experiment
```

The generator is therefore only one component of a larger scientific system.

---

## 24.16.1 Generation Is Not Prediction

It is useful to distinguish two fundamentally different machine-learning tasks.

A conventional predictive model solves a forward problem.

```text
Crystal Structure
        ↓
Predictive Model
        ↓
Material Property
```

For example,

```text
Crystal
   ↓
GNN
   ↓
Band Gap
```

The model receives an existing material and predicts a property.

A generative model performs a fundamentally different operation.

```text
Desired Condition
        ↓
Generative Model
        ↓
Crystal Structure
```

The model attempts to construct a new object that satisfies the requested constraints.

The distinction can therefore be summarized as

```text
Prediction:

Structure → Property


Generation:

Property → Structure
```

This reversal is what makes generative materials discovery an inverse-design problem.

---

## 24.16.2 The Search Space Is Enormous

The number of possible crystalline materials is extraordinarily large.

Even if we restrict ourselves to a small number of chemical elements, the number of possible combinations of

```text
Composition
Atomic Arrangement
Lattice Parameters
Space Group
Atomic Positions
Defect Concentration
Ordering
```

can become enormous.

For a crystal containing `N` atoms, even the atomic coordinates alone contain approximately

```text
3N
```

continuous degrees of freedom before considering additional variables such as the lattice.

The lattice itself introduces additional degrees of freedom.

A general three-dimensional lattice can be represented by

```text
A =
[ a₁  a₂  a₃ ]
```

where the three vectors define the periodic unit cell.

Therefore, a crystal-generation model is not simply choosing a few categorical labels.

It is exploring a highly structured, high-dimensional space.

---

## 24.16.3 Why Random Search Is Inefficient

Suppose the complete candidate space contains

```text
10^12
```

possible structures.

Assume that only

```text
10^4
```

of these structures are scientifically interesting.

The fraction of useful candidates is then extremely small.

A random search would therefore spend most of its computational effort exploring irrelevant regions.

Generative models attempt to solve this problem by learning the distribution of physically observed or computationally generated materials.

Instead of searching blindly,

```text
Entire Chemical Space
```

the model learns to concentrate sampling around

```text
Chemically and structurally plausible regions
```

of that space.

This is one of the major reasons generative modeling is attractive for materials discovery.

---

## 24.16.4 Learning the Materials Distribution

Let `C` denote a crystal structure.

An unconditional generative model attempts to approximate

```text
p(C)
```

whereas a property-conditioned model attempts to approximate

```text
p(C | y)
```

where `y` represents the desired property.

The difference is scientifically important.

For unconditional generation,

```text
Sample:

C ~ p(C)
```

For conditional generation,

```text
Sample:

C ~ p(C | y_target)
```

The second formulation allows the researcher to direct the generation process.

For example,

```text
y_target = 2.0 eV
```

could represent a desired band gap.

The generator then preferentially explores structures associated with that region of materials space.

---

## 24.16.5 The Generator Learns a Distribution, Not a Formula

A common misunderstanding is to think that the generative model learns a deterministic equation such as

```text
Band Gap = f(Composition)
```

and then simply solves that equation backward.

In reality, materials behavior is much more complicated.

A property may depend simultaneously on

```text
Composition
Crystal Structure
Bond Length
Bond Angle
Coordination
Symmetry
Electronic Configuration
Defects
Magnetic Ordering
```

Therefore, multiple structures may produce similar properties.

Mathematically,

```text
y → C
```

is generally not a one-to-one mapping.

Instead,

```text
p(C | y)
```

may contain many plausible structures.

This is one of the fundamental reasons probabilistic generative models are appropriate for inverse materials design.

---

## 24.16.6 Multiple Valid Solutions

Consider a target band gap of approximately

```text
2.0 eV
```

Suppose the generator produces

```text
Crystal A → 1.95 eV

Crystal B → 2.02 eV

Crystal C → 2.08 eV

Crystal D → 1.98 eV
```

All four structures may satisfy the original electronic objective.

However, they may differ substantially in

```text
Formation Energy
Density
Mechanical Properties
Chemical Stability
Synthesis Feasibility
```

Therefore, the best candidate cannot be selected using band gap alone.

The discovery process must evaluate several dimensions of scientific usefulness.

---

## 24.16.7 Candidate Ranking

Suppose the generated candidates have predicted properties

| Crystal | Band Gap (eV) | Formation Energy | Bulk Modulus (GPa) |
| ------- | ------------: | ---------------: | -----------------: |
| A       |          1.96 |             -1.8 |                140 |
| B       |          2.01 |             -3.1 |                210 |
| C       |          2.08 |             -2.4 |                190 |
| D       |          1.73 |             -3.5 |                230 |

If the objective is

```text
Band Gap ≈ 2 eV
Low Formation Energy
High Bulk Modulus
```

candidate B may be more attractive than candidate A even though both satisfy the band-gap requirement.

This illustrates an important principle:

> Materials discovery requires ranking candidates according to the complete scientific objective rather than evaluating a single property in isolation.

---

## 24.16.8 Hard Constraints and Soft Objectives

Not every requirement should be treated in the same way.

Suppose a researcher wants

```text
Band Gap:
1.5–2.0 eV

Formation Energy:
< 0

Density:
< 6 g/cm³
```

The first two conditions might be treated as hard constraints.

A candidate outside these regions may simply be rejected.

Other objectives may be softer.

For example,

```text
Higher Bulk Modulus
```

may be preferred but not absolutely required.

The resulting design problem can therefore contain both

```text
Hard Constraints
```

and

```text
Optimization Objectives
```

A practical workflow is

```text
Generated Candidate
        ↓
Hard Constraint Filtering
        ↓
Soft Objective Ranking
        ↓
High-Priority Candidates
```

This is more realistic than treating every property identically.

---

## 24.16.9 The Importance of Uncertainty

A prediction from a machine-learning model is not necessarily exact.

Suppose a property predictor estimates

```text
Band Gap = 1.92 eV
```

The model should ideally also provide information about its confidence.

For example,

```text
Band Gap = 1.92 ± 0.15 eV
```

The uncertainty becomes particularly important when the generated structure lies outside the training distribution.

A model may confidently predict a familiar material while being highly uncertain for a chemically unusual structure.

Therefore, generative materials discovery should consider both

```text
Prediction
```

and

```text
Prediction Uncertainty
```

---

## 24.16.10 Distribution Shift

Suppose the training dataset contains mostly

```text
Oxides
```

and the generator produces an unusual

```text
Sulfide
```

The property predictor may still return a numerical value.

However, that numerical value may not be reliable.

This is an example of distribution shift.

The training distribution is

```text
p_train(C)
```

while the generated candidate may belong to a different distribution

```text
p_generated(C)
```

If these distributions differ substantially, prediction uncertainty should increase.

This is one of the most important issues in research-grade generative materials workflows.

---

## 24.16.11 Generation and Verification Must Remain Separate

A useful architecture therefore separates

```text
Generation
```

from

```text
Verification
```

The generator proposes candidates.

Independent models and physics-based methods evaluate them.

```text
                 Generator
                     ↓
              Candidate Crystal
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
   Structure      Property       Stability
   Validation     Prediction     Evaluation
       │             │             │
       └─────────────┼─────────────┘
                     ↓
                 Candidate
                  Ranking
                     ↓
                    DFT
```

This separation is scientifically valuable.

If the same model both generates a structure and declares that the structure is successful, there is a risk of introducing correlated errors.

Independent verification provides a stronger test.

---

## 24.16.12 The Role of DFT

Density Functional Theory remains one of the most important verification tools in the workflow.

Machine-learning models are extremely useful for rapid screening.

However, a generated structure may ultimately require a higher-fidelity calculation.

The workflow therefore becomes

```text
Generative Model
        ↓
10⁵ Candidates
        ↓
ML Screening
        ↓
10³ Candidates
        ↓
DFT Screening
        ↓
10² Candidates
        ↓
Detailed DFT
        ↓
10 Candidates
        ↓
Experiment
```

The exact numbers will depend on the research problem.

The principle is more important than the numerical values:

> Expensive calculations should be concentrated on the most promising candidates.

---

## 24.16.13 Geometry Optimization as Verification

A generated structure is not necessarily at an equilibrium configuration.

Its initial atomic positions may contain small or large deviations from a stable geometry.

Therefore, a DFT workflow may begin with structural relaxation.

```text
Generated Structure
        ↓
DFT Geometry Optimization
        ↓
Relaxed Structure
        ↓
Energy Evaluation
        ↓
Property Calculation
```

This creates an important distinction between

```text
Generated Geometry
```

and

```text
Relaxed Geometry
```

A candidate that looks promising before relaxation may become unstable after optimization.

Conversely, a slightly imperfect generated geometry may relax into a physically meaningful structure.

---

## 24.16.14 Energy Above Hull

For many materials-discovery problems, formation energy alone is insufficient.

Thermodynamic stability can be evaluated relative to competing phases.

The relevant quantity is often the energy above the convex hull.

Conceptually,

```text
E_hull
```

represents the lowest-energy combination of competing phases at the same overall composition.

A candidate with

```text
E_above_hull = 0
```

lies on the convex hull and is thermodynamically stable relative to the considered competing phases.

A positive value indicates metastability or instability relative to those phases.

This makes energy-above-hull screening particularly important for generated crystals.

---

## 24.16.15 Generated Structures Must Be Compared with the Materials Distribution

A generative model should not merely produce valid structures.

It should also produce structures that occupy scientifically meaningful regions of materials space.

Suppose the training dataset contains

```text
Oxides
Nitrides
Sulfides
Halides
```

and the generated dataset contains only

```text
Oxides
```

even though the model was expected to explore the broader chemical space.

This may indicate excessive concentration or mode collapse.

Therefore, generated structures should be compared with the training distribution in terms of

```text
Composition
Crystal System
Space Group
Density
Local Coordination
Structural Fingerprints
Latent Embeddings
```

---

## 24.16.16 Latent-Space Analysis

A useful way to compare generated and known materials is through a learned representation.

Suppose a materials encoder maps every structure to

```text
z ∈ R^d
```

The resulting vectors can be visualized using dimensionality-reduction methods.

For example,

```text
Crystal Structures
        ↓
Materials Encoder
        ↓
High-Dimensional Embeddings
        ↓
PCA / UMAP
        ↓
2D Materials Map
```

The researcher can then inspect whether generated structures

```text
remain inside known materials regions
```

or

```text
explore previously underrepresented regions.
```

Both behaviors can be scientifically meaningful.

---

## 24.16.17 Novelty Must Be Distinguished from Error

A generated structure that differs from the training data is not automatically a valuable discovery.

It may be

```text
Novel and Valid
```

or

```text
Novel and Physically Impossible
```

Therefore,

```text
Novelty
```

must always be considered together with

```text
Validity
```

and

```text
Scientific Utility
```

A useful candidate occupies a region of materials space that is sufficiently new while remaining physically plausible.

---

## 24.16.18 The Discovery Triangle

The three fundamental requirements can be summarized as

```text
                 Validity
                    ▲
                   / \
                  /   \
                 /     \
                /       \
          Novelty ───── Utility
```

A strong generative system should balance all three.

A model that maximizes validity may reproduce known structures.

A model that maximizes novelty may generate unrealistic structures.

A model that maximizes a single property may produce highly repetitive candidates.

The scientific challenge is to identify the region where

```text
Validity
+
Novelty
+
Utility
```

are simultaneously high.

---

## 24.16.19 From Candidate Generation to Candidate Selection

At this point, the generative process produces a candidate pool.

For example,

```python
candidates = []

for _ in range(10000):

    crystal = generate_crystal(
        model,
        condition
    )

    candidates.append(crystal)
```

The candidate pool can then be processed.

```python
valid_candidates = []

for crystal in candidates:

    if check_structure(crystal):

        valid_candidates.append(
            crystal
        )
```

A property predictor can then evaluate the valid structures.

```python
predicted_band_gap = band_gap_model(
    valid_candidates
)
```

Additional properties can be predicted in the same way.

```python
predicted_energy = formation_energy_model(
    valid_candidates
)

predicted_bulk = bulk_modulus_model(
    valid_candidates
)
```

The result is a structured candidate table.

```text
Crystal | Band Gap | Formation Energy | Bulk Modulus
----------------------------------------------------
A       | 1.82      | -2.7             | 190
B       | 2.04      | -3.1             | 220
C       | 2.61      | -1.4             | 170
```

The generative problem has now become a **decision problem**.

---

## 24.16.20 Why Decision-Making Becomes the Next Problem

Suppose there are

```text
100,000
```

valid candidates.

Only

```text
100
```

can be evaluated using expensive DFT.

The researcher must therefore decide which 100 structures deserve further investigation.

Random selection wastes information.

Selecting only the structures with the highest predicted property may also be dangerous because the model may be uncertain about those predictions.

A better strategy should consider

```text
Predicted Performance
+
Prediction Uncertainty
+
Candidate Diversity
+
Scientific Constraints
```

This is the point where the generative framework naturally connects to sequential decision-making.

---

## 24.16.21 The Scientific Feedback Loop

Once DFT or experiment is performed, the new results become additional training data.

Suppose the original dataset contains

```text
D₀
```

The generative model produces candidates.

After verification, new observations are obtained:

```text
D_new
```

The updated dataset becomes

```text
D₁ = D₀ ∪ D_new
```

The model can then be retrained.

```text
D₀
 ↓
Train Model
 ↓
Generate Candidates
 ↓
DFT / Experiment
 ↓
D_new
 ↓
D₁ = D₀ ∪ D_new
 ↓
Retrain
 ↺
```

This transforms a static machine-learning workflow into a continuously improving discovery system.

---

## 24.16.22 Human Scientific Reasoning Remains Essential

Automation does not eliminate the researcher.

The researcher must still decide

```text
What property matters?

What constraints are scientifically justified?

Which candidates are chemically meaningful?

Which computational method is appropriate?

Which DFT approximation is reliable?

Which experimental route is realistic?
```

AI can accelerate exploration.

It cannot automatically determine the scientific importance of every generated structure.

The strongest framework is therefore not

```text
Human vs AI
```

but

```text
Human Scientific Reasoning
+
Machine Learning
+
Generative Modeling
+
Physics-Based Simulation
+
Experiment
```

---

## 24.16.23 Toward Closed-Loop Materials Discovery

The complete system can now be represented as

```text
                 Scientific Objective
                         ↓
                 Generative Model
                         ↓
                 Candidate Pool
                         ↓
                Fast Prediction
                         ↓
                Candidate Selection
                         ↓
                       DFT
                         ↓
                    Experiment
                         ↓
                    New Data
                         ↓
                 Model Updating
                         ↓
                Improved Generation
                         ↺
```

The important feature of this architecture is the feedback loop.

The system does not generate materials once and stop.

Instead,

```text
Generation
→ Evaluation
→ Learning
→ Generation
```

is repeated.

Each cycle has the potential to improve the model's understanding of the relevant region of materials space.

---

## 24.16.24 Transition to Sequential Materials Learning

The framework developed in this chapter has therefore reached an important boundary.

Generative AI answers:

```text
What materials could exist?
```

Property prediction answers:

```text
What might their properties be?
```

Physics-based simulation answers:

```text
Are those predictions physically credible?
```

Experiment answers:

```text
Can the material actually be realized?
```

The remaining question is:

```text
Which candidate should we evaluate next?
```

That question requires a different class of methods based on **uncertainty, sequential decision-making, and efficient selection of new information**.

Those methods form the subject of the next chapter.

# Transition to Chapter 25 — Active Learning

The generative framework developed in Chapter 24 can produce an enormous number of candidate materials. The central difficulty is no longer simply generating possibilities. It is deciding where limited computational and experimental resources should be invested.

A researcher may have

```text
100,000 generated candidates
```

but enough computational resources to perform detailed calculations on only

```text
100
```

of them.

The selection of those 100 candidates can determine whether the discovery campaign succeeds or fails.

Rather than choosing candidates randomly, the next chapter will develop a systematic framework in which the current model determines which new observation would be most informative or most promising.

The resulting loop is

```text
Current Dataset
      ↓
Train Model
      ↓
Estimate Predictions
      ↓
Estimate Uncertainty
      ↓
Select Candidate
      ↓
DFT / Experiment
      ↓
New Observation
      ↓
Update Dataset
      ↓
Retrain Model
      ↺
```

# 24.17 — Advanced Diffusion Models for Crystal Generation

The diffusion framework introduced earlier provides the basic mechanism for generating crystal structures by gradually transforming structured data into noise and then learning the reverse process.

For simple demonstrations, a discrete-time formulation is sufficient.

However, research-grade crystal generation requires a more general treatment.

Real crystal structures contain multiple interacting representations:

```text
Composition
     +
Lattice
     +
Atomic Coordinates
     +
Periodic Geometry
     +
Chemical Identity
```

These variables do not all behave in the same mathematical way.

Atomic coordinates are continuous.

Element identities are categorical.

The lattice is continuous but constrained by periodicity.

Crystal symmetries impose geometric restrictions.

Consequently, an advanced crystal diffusion model must account for both the **probabilistic nature of diffusion** and the **physical structure of crystalline materials**.

A useful research-level architecture can therefore be represented as

```text
                    Crystal
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
      Composition    Lattice      Coordinates
          │            │            │
          └────────────┼────────────┘
                       ↓
                Equivariant Network
                       ↓
                Score / Noise Model
                       ↓
                Diffusion Process
                       ↓
                Generated Crystal
```

The central objective remains unchanged:

> Learn how to transform a simple stochastic distribution into a distribution of chemically and structurally meaningful crystals.

---

# 24.17.1 Why Standard Diffusion Is Not Enough

A conventional diffusion model may operate on a vector

```text
x ∈ R^d
```

where every component can be perturbed using Gaussian noise.

A crystal cannot always be represented this simply.

Suppose the atomic coordinates are

```text
r₁, r₂, ..., r_N
```

and the lattice matrix is

```text
L
```

The crystal is represented by

```text
C = (Z, L, R)
```

where

```text
Z = atomic identities

L = lattice parameters

R = atomic coordinates
```

The model must preserve physically meaningful relationships among these quantities.

For example, translating every atom by the same vector should not change the physical identity of the crystal.

Likewise, rotating the entire crystal should not create a fundamentally different material.

This leads directly to the requirement for **symmetry-aware and equivariant neural networks**.

---

# 24.17.2 Score-Based Generative Modeling

A useful alternative formulation of diffusion models is based on the **score function**.

For a probability density `p(x)`, the score is defined as

```text
s(x) = ∇ₓ log p(x)
```

The score describes the direction in which the probability density increases most rapidly.

Conceptually,

```text
Low Probability
       ↑
       │
       │   Score
       │    ↗
       │   /
       │  /
       ●────────→
          High Probability
```

For materials generation, the score can be interpreted as information about how the current noisy structure should move toward regions of higher probability under the learned materials distribution.

The generative process therefore becomes a trajectory through configuration space.

```text
Noise
  ↓
Noisy Crystal
  ↓
Score Guidance
  ↓
Less Noisy Crystal
  ↓
Score Guidance
  ↓
...
  ↓
Crystal
```

---

# 24.17.3 The Score of a Crystal Distribution

Let the crystal representation be

```text
x
```

and let the distribution of valid crystals be

```text
p_data(x)
```

The ideal score function would be

```text
s*(x) = ∇ₓ log p_data(x)
```

In practice, the true distribution is unknown.

A neural network therefore learns an approximation:

```text
s_θ(x,t) ≈ ∇ₓ log p_t(x)
```

where `p_t(x)` is the distribution of crystals after a particular amount of noise has been added.

The time variable `t` is important because the correct score depends on the noise level.

At high noise,

```text
p_t(x)
```

contains very little structural information.

At low noise,

```text
p_t(x)
```

is much closer to the actual crystal distribution.

Therefore, the score network must learn behavior across many noise levels.

---

# 24.17.4 Noise-Conditioned Score Networks

A simplified score network can be written as

```python
class ScoreNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        time_dim
    ):

        super().__init__()

        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            )
        )

    def forward(
        self,
        x,
        time_embedding
    ):

        t = self.time_encoder(
            time_embedding
        )

        h = self.network[0](x)

        h = h + t

        h = self.network[1](h)
        h = self.network[2](h)
        h = self.network[3](h)
        h = self.network[4](h)

        return h
```

This simplified model is not yet a crystal-aware architecture.

Its purpose is to illustrate the central idea:

```text
Current Structure
       +
Noise Level
       ↓
Score Network
       ↓
Direction Toward Higher Probability
```

A research implementation would replace the simple vector network with an architecture capable of processing periodic crystal geometry.

---

# 24.17.5 Denoising Score Matching

The score can be learned using denoising score matching.

Suppose a clean structure is

```text
x₀
```

and a noisy structure is generated according to

```text
x_t = x₀ + σ(t) ε
```

where

```text
ε ~ N(0,I)
```

The model attempts to learn the score associated with the noisy distribution.

A simplified objective is

```text
L = E [
    λ(t)
    ||s_θ(x_t,t) - s_target(x_t,x₀,t)||²
]
```

where

```text
s_θ
```

is the neural-network prediction and

```text
s_target
```

is the analytically known score for the chosen perturbation process.

The weighting function

```text
λ(t)
```

controls how different noise levels contribute to the training objective.

---

# 24.17.6 Why the Score Formulation Is Useful for Materials

The score formulation provides a useful interpretation of the generative process.

Instead of asking the network to directly predict the final crystal, the model learns a local direction field.

Conceptually,

```text
Current Crystal
      ↓
Where should it move?
      ↓
Score
      ↓
Small update
      ↓
Improved Crystal
```

Repeated application of these updates can transform random noise into a structured material.

This is particularly useful for high-dimensional crystal representations because the network does not have to directly memorize every possible crystal.

It learns the local geometry of the probability distribution.

---

# 24.17.7 Continuous-Time Diffusion

Discrete diffusion uses a finite sequence of noise levels:

```text
t₀
t₁
t₂
...
t_T
```

A continuous-time formulation instead treats time as a continuous variable:

```text
t ∈ [0,1]
```

The noisy crystal evolves continuously.

The forward process can be represented by a stochastic differential equation:

```text
dx = f(x,t)dt + g(t)dW
```

where

```text
f(x,t)
```

is the drift term,

```text
g(t)
```

controls the diffusion strength,

and

```text
dW
```

represents Brownian motion.

This formulation provides a mathematical bridge between diffusion models and stochastic processes.

---

# 24.17.8 Stochastic Differential Equations

The forward SDE progressively destroys information in the original material structure.

Conceptually,

```text
Crystal
   ↓
Small Perturbation
   ↓
More Noise
   ↓
More Noise
   ↓
Random Distribution
```

The reverse process attempts to reconstruct the crystal.

If the forward process is

```text
dx = f(x,t)dt + g(t)dW
```

the reverse-time process depends on the score.

Conceptually,

```text
Forward SDE:

Crystal → Noise


Reverse SDE:

Noise → Crystal
```

The learned score provides the information required to reverse the diffusion process.

---

# 24.17.9 Reverse-Time Generation

The reverse stochastic process can be written conceptually as

```text
dx =
[
f(x,t)
-
g(t)² ∇ₓ log p_t(x)
]dt
+
g(t)dW̄
```

where the score

```text
∇ₓ log p_t(x)
```

provides information about the data distribution.

The exact sign conventions depend on the direction in which time is parameterized.

The important conceptual result is that the score modifies the reverse dynamics so that random samples are gradually transformed into samples from the learned crystal distribution.

---

# 24.17.10 Probability-Flow ODE

An important consequence of the continuous-time formulation is that a corresponding deterministic ordinary differential equation can be constructed.

This is called the probability-flow ODE.

Conceptually,

```text
Stochastic Reverse Process
          │
          ↓
Probability-Flow ODE
          │
          ↓
Deterministic Trajectory
```

The ODE has the general form

```text
dx =
[
f(x,t)
-
1/2 g(t)² s_θ(x,t)
]dt
```

The stochastic term disappears.

This provides an alternative route for generation.

Instead of sampling random noise at every reverse step, the model can integrate a deterministic trajectory from the initial noise distribution toward the learned data distribution.

---

# 24.17.11 Why Probability-Flow ODEs Matter

Probability-flow ODEs are useful for several reasons.

They provide

```text
Deterministic trajectories

Alternative sampling methods

Potential likelihood estimation

Continuous-time interpretation

Connections to numerical differential equations
```

For materials researchers, this is especially interesting because numerical integration methods become part of the generative process.

The generator is no longer viewed merely as a neural network.

It becomes a learned dynamical system.

---

# 24.17.12 Numerical Integration

The probability-flow ODE must be solved numerically.

A simple Euler update has the form

```python
x_next = (
    x
    +
    dt * velocity
)
```

where

```python
velocity = ode_model(
    x,
    t
)
```

More sophisticated solvers may provide greater accuracy.

The general workflow becomes

```text
Initial Noise
     ↓
ODE Solver
     ↓
t = 1 → 0
     ↓
Generated Crystal
```

The number of integration steps influences both computational cost and generation quality.

---

# 24.17.13 SDE-Based Crystal Generation

For crystal generation, the continuous representation may be applied to atomic coordinates.

Suppose

```text
R =
[r₁,
 r₂,
 ...
 r_N]
```

represents the atomic coordinates.

A diffusion process can perturb

```text
R
```

while the neural network learns how to reverse that perturbation.

However, ordinary Euclidean coordinates introduce a major issue.

If the entire crystal is translated,

```text
rᵢ' = rᵢ + a
```

the physical structure is unchanged.

Likewise, a global rotation should not change the physical identity.

This motivates equivariant architectures.

---

# 24.17.14 Equivariance

A function is equivariant when transforming the input produces a corresponding transformation of the output.

Suppose

```text
R
```

is rotated by a transformation

```text
Q
```

such that

```text
R' = QR
```

An equivariant model should satisfy a relationship of the form

```text
f(QR) = Qf(R)
```

for vector-valued outputs.

For invariant quantities, the desired relationship is instead

```text
f(QR) = f(R)
```

For example,

```text
Energy
```

should remain unchanged under a global rotation.

Therefore,

```text
Energy(QR) = Energy(R)
```

while a predicted force should rotate with the structure.

---

# 24.17.15 Why Equivariance Is Essential for Crystal Generation

Suppose two structures differ only by a global rotation.

Physically, they represent the same structure.

A naive neural network may nevertheless treat their Cartesian coordinates as different patterns.

This forces the model to learn rotational invariance from data.

An equivariant architecture incorporates this symmetry directly.

Instead of learning

```text
Rotation invariance
```

from thousands of examples, the architecture enforces it mathematically.

This can improve

```text
Data efficiency

Physical consistency

Generalization

Training stability
```

---

# 24.17.16 Message Passing in an Equivariant Crystal Network

A crystal can be represented as a graph

```text
G = (V,E)
```

where

```text
V = atoms
E = interactions between atoms
```

Each atom contains features such as

```text
Atomic Number
Element Embedding
Charge
Local Environment
```

Edges can contain

```text
Distance
Relative Position
Periodic Image Information
```

A message-passing layer can therefore use

```text
h_i
h_j
r_ij
```

to construct an interaction message.

Conceptually,

```text
m_ij = φ(h_i,h_j,r_ij)
```

and

```text
h_i' =
ψ(
h_i,
Σ_j m_ij
)
```

An equivariant model additionally ensures that vector and tensor quantities transform correctly under symmetry operations.

---

# 24.17.17 Crystal-Specific Diffusion Objectives

A generic diffusion objective may not be optimal for crystals.

A crystal-generation model can instead combine several objectives.

For example,

```text
L_total =
L_diffusion
+
λ₁ L_geometry
+
λ₂ L_chemistry
+
λ₃ L_symmetry
+
λ₄ L_property
```

where

```text
L_diffusion
```

trains the generative process,

```text
L_geometry
```

encourages physically reasonable geometry,

```text
L_chemistry
```

encourages chemically meaningful structures,

```text
L_symmetry
```

encourages appropriate crystalline symmetry,

and

```text
L_property
```

encourages target-property satisfaction.

The coefficients

```text
λ₁, λ₂, λ₃, λ₄
```

control the relative importance of these objectives.

---

# 24.17.18 Geometry Constraints

Generated crystals may contain

```text
Very Short Bonds

Atomic Overlap

Extremely Large Voids

Unrealistic Cell Dimensions
```

A geometry penalty can therefore be introduced.

For example, if two atoms are closer than a minimum acceptable distance,

```text
d_ij < d_min
```

a penalty can be applied.

A simplified penalty is

```text
L_overlap =
Σ_ij max(0, d_min - d_ij)²
```

This does not replace physics-based validation.

It simply gives the generator an additional signal discouraging obviously unreasonable structures.

---

# 24.17.19 Chemical Constraints

Chemical validity is more difficult than geometric validity.

A generated structure may have acceptable distances but chemically implausible bonding.

For example,

```text
Unreasonable Coordination

Unbalanced Charge

Unlikely Oxidation States

Chemically Implausible Composition
```

can occur.

Chemical constraints may therefore be incorporated through

```text
Composition Filters

Oxidation-State Analysis

Coordination Analysis

Valence Rules

Chemical-Potential Constraints
```

The model should not be expected to discover all chemical rules automatically.

Domain knowledge remains valuable.

---

# 24.17.20 Crystal-Symmetry Constraints

Crystal structures are associated with space groups and symmetry operations.

A generated structure may approximately resemble a known symmetry class while containing small coordinate deviations.

After generation, symmetry analysis can therefore be performed.

A practical workflow is

```text
Generated Structure
        ↓
Structure Standardization
        ↓
Symmetry Detection
        ↓
Space-Group Identification
        ↓
Consistency Check
```

Tools such as `pymatgen` and related crystallographic libraries can assist in this stage.

The important point is that symmetry should be treated as a scientific property rather than merely a visualization feature.

---

# 24.17.21 Evaluation of Generated Materials

A generative model should never be evaluated using a single metric.

A research-grade evaluation should examine at least

```text
Validity
Novelty
Diversity
Property Accuracy
Distribution Similarity
Stability
```

These quantities answer different questions.

```text
Validity:
Are the structures physically meaningful?

Novelty:
Are they genuinely different from known materials?

Diversity:
Does the model produce many different solutions?

Property Accuracy:
Do they satisfy the requested objectives?

Distribution Similarity:
Does generation remain chemically realistic?

Stability:
Can the structures survive physical validation?
```

---

# 24.17.22 Validity

Validity is the first requirement.

A generated structure should satisfy basic structural conditions.

For example,

```text
No severe atomic overlap

Valid lattice

Valid periodic structure

Reasonable density

Meaningful composition
```

A simple validity rate can be defined as

```text
Validity Rate =
Valid Structures
----------------
Generated Structures
```

For example, if

```text
10,000
```

structures are generated and

```text
8,700
```

pass the structural filters,

then

```text
Validity Rate = 0.87
```

or

```text
87%
```

A high validity rate is desirable, but it does not guarantee scientific usefulness.

---

# 24.17.23 Novelty

Novelty measures whether generated structures differ from known structures.

A naive approach is to compare generated compositions with the training dataset.

However, composition-level novelty is insufficient.

Two structures may have the same composition but different

```text
Crystal Structure

Space Group

Atomic Arrangement

Polymorph
```

Therefore, structural fingerprints should preferably be used.

A general workflow is

```text
Generated Crystal
        ↓
Structural Fingerprint
        ↓
Compare with Training Set
        ↓
Nearest Known Structure
        ↓
Novelty Score
```

---

# 24.17.24 Diversity

Diversity measures how broadly the generator explores the candidate space.

Suppose 10,000 structures are generated.

If almost all structures are nearly identical, the model has low diversity.

A simple conceptual metric is the average pairwise distance:

```text
Diversity =
Average(
distance(C_i,C_j)
)
```

where the distance is calculated using an appropriate structural representation.

For large datasets, exact pairwise comparison may be expensive.

Approximate nearest-neighbor methods or embedding-space statistics can therefore be used.

---

# 24.17.25 Property Accuracy

For property-conditioned generation, the requested property must be evaluated.

Suppose the target is

```text
Eg = 2.0 eV
```

and generated candidates have predicted values

```text
1.97
2.03
2.11
1.91
```

The model can be evaluated using target error.

A simple metric is

```text
MAE =
1/N Σ |y_pred - y_target|
```

The lower the error, the more accurately the generator satisfies the condition.

For property ranges, the evaluation can instead measure the fraction of candidates inside the acceptable interval.

---

# 24.17.26 Distribution Matching

A strong generator should reproduce important statistical properties of real materials.

For example,

```text
Element Frequencies

Composition Statistics

Density Distribution

Lattice Parameters

Coordination Numbers

Space-Group Distribution
```

can be compared between

```text
Training Data
```

and

```text
Generated Data
```

Large deviations may indicate that the generator is producing unrealistic or biased structures.

---

# 24.17.27 Benchmarking Against Known Materials

Generated materials should be compared with existing databases or literature.

For example, a candidate can be compared against known structures in terms of

```text
Composition

Formation Energy

Band Gap

Density

Crystal System

Space Group
```

This allows the researcher to determine whether the generated candidates are

```text
Known Materials

Known Polymorphs

Near-Duplicates

Potentially Novel Structures
```

Benchmarking is therefore essential before claiming novelty.

---

# 24.17.28 DFT-Based Validation

Machine-learning predictions are useful for rapid screening, but final validation should use appropriate physics-based calculations.

A candidate may therefore pass through

```text
Structure Validation
        ↓
ML Property Prediction
        ↓
ML Stability Prediction
        ↓
DFT Relaxation
        ↓
DFT Energy
        ↓
DFT Properties
```

The exact calculations depend on the research objective.

For an electronic-materials problem, one may evaluate

```text
Band Structure

Density of States

Band Gap

Formation Energy
```

For mechanical materials,

```text
Elastic Constants

Bulk Modulus

Shear Modulus

Young's Modulus
```

may be more appropriate.

---

# 24.17.29 DFT as an Independent Test

The most important role of DFT in this framework is not merely to produce another prediction.

It provides an approximately independent physical evaluation.

Suppose the machine-learning predictor estimates

```text
Band Gap = 2.0 eV
```

but DFT produces

```text
Band Gap = 0.8 eV
```

The disagreement reveals a failure of the screening model.

Such failures are scientifically valuable because they identify regions where the machine-learning model needs improvement.

---

# 24.17.30 Failure Analysis

A research-grade generative model must be analyzed when it fails.

Common failure modes include

```text
Invalid Geometry

Chemical Impossibility

Duplicate Generation

Mode Collapse

Poor Property Control

Unstable Structures

Out-of-Distribution Generation

Prediction Error

Sampling Instability
```

Each failure mode requires a different response.

For example,

```text
Invalid Geometry
        ↓
Improve geometric representation
```

whereas

```text
Poor Property Control
        ↓
Improve conditioning mechanism
```

and

```text
Mode Collapse
        ↓
Improve diversity or sampling strategy
```

---

# 24.17.31 Mode Collapse

Mode collapse occurs when the generator repeatedly produces very similar structures.

Suppose 10,000 samples contain only

```text
25
```

meaningfully distinct structures.

The generator has failed to adequately represent the diversity of the target distribution.

This can occur when

```text
Conditioning Is Too Strong

Sampling Is Poor

Training Data Is Too Limited

Model Capacity Is Inappropriate
```

The problem is particularly important in materials discovery because repeated generation of the same structure provides little additional scientific information.

---

# 24.17.32 Property-Conditioning Failure

A generator may produce chemically valid structures but fail to satisfy the target property.

For example,

```text
Target Band Gap = 2.0 eV
```

but generated structures cluster around

```text
0.5–1.0 eV
```

This indicates that the conditioning mechanism is not sufficiently controlling the generation process.

Possible causes include

```text
Weak Property Representation

Poor Predictor

Insufficient Training Data

Incorrect Property Scaling

Weak Guidance

Conflicting Objectives
```

Property conditioning should therefore be evaluated independently from structural validity.

---

# 24.17.33 Reproducibility

Generative materials research is particularly sensitive to randomness.

A single experiment may depend on

```text
Random Seed

Dataset Split

Model Initialization

Noise Schedule

Sampling Steps

Guidance Strength

Software Version

Hardware
```

Therefore, a research result should not depend on a single arbitrary run.

A reproducible experiment should record these parameters explicitly.

For example,

```python
config = {

    "seed": 42,

    "epochs": 500,

    "batch_size": 64,

    "learning_rate": 1e-4,

    "sampling_steps": 1000,

    "guidance_strength": 2.0
}
```

The configuration should be stored together with the trained model.

---

# 24.17.34 Multiple Random Seeds

A robust experiment should ideally be repeated with multiple random seeds.

For example,

```python
seeds = [
    0,
    1,
    2,
    3,
    4
]

for seed in seeds:

    train_model(
        seed=seed
    )
```

Performance can then be reported as

```text
Mean ± Standard Deviation
```

rather than as a single number.

This reduces the possibility that an unusually favorable random initialization is mistaken for genuine model performance.

---

# 24.17.35 Experimental Record

A complete generative-materials experiment should record

```text
Dataset Version

Preprocessing

Feature Representation

Model Architecture

Hyperparameters

Training Configuration

Random Seeds

Sampling Configuration

Generated Candidates

Filtering Rules

Prediction Models

DFT Settings

Final Selection Criteria
```

This information allows another researcher to reproduce the study.

Reproducibility is not an optional software practice.

It is part of scientific validity.

---

# 24.17.36 A Research-Grade Generative Materials Pipeline

The complete framework can now be summarized.

```text
                    Scientific Objective
                           ↓
                    Materials Dataset
                           ↓
                 Representation Learning
                           ↓
                Crystal Generative Model
                           ↓
                    Candidate Sampling
                           ↓
                  Structural Validation
                           ↓
                   Chemical Validation
                           ↓
                  Duplicate Removal
                           ↓
                  Property Prediction
                           ↓
                Uncertainty Estimation
                           ↓
                 Candidate Ranking
                           ↓
                        DFT
                           ↓
                  Stability Evaluation
                           ↓
                    Experiment
                           ↓
                     New Data
                           ↓
                    Model Updating
                           ↺
```

This is substantially more powerful than treating generative AI as a system that simply produces crystal structures.

It transforms generation into a complete scientific workflow.

---

# 24.17.37 From Generative AI to Autonomous Discovery

The ultimate objective is not merely to generate more structures.

The objective is to create a system capable of repeatedly performing

```text
Generate
Evaluate
Select
Simulate
Learn
Generate Again
```

The system gradually concentrates computational and experimental effort in promising regions of materials space.

This creates a closed-loop architecture.

```text
             ┌──────────────────────┐
             │ Scientific Objective │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Generative Model     │
             └──────────┬───────────┘
                        ↓
                  Candidates
                        ↓
             ┌──────────────────────┐
             │ Predictive Models    │
             └──────────┬───────────┘
                        ↓
                  Candidate Set
                        ↓
             ┌──────────────────────┐
             │ DFT / Experiment     │
             └──────────┬───────────┘
                        ↓
                    New Data
                        ↓
             ┌──────────────────────┐
             │ Model Update         │
             └──────────┬───────────┘
                        │
                        └──────────────→
```

The next question is therefore not simply

```text
What can we generate?
```

but

```text
What should we evaluate next?
```

That distinction marks the transition from **generative modeling** to **active scientific decision-making**.

---

# 24.18 — Chapter 24 Summary

Generative AI provides a fundamentally different approach to materials discovery.

Instead of beginning with a predefined material and predicting its properties, the researcher can specify a scientific objective and ask a generative model to propose candidate structures.

The central progression developed in this chapter is

```text
Materials Distribution
        ↓
Generative Representation
        ↓
Crystal Generation
        ↓
Property Conditioning
        ↓
Candidate Screening
        ↓
Physics-Based Validation
        ↓
Scientific Discovery
```

Variational autoencoders demonstrated how materials can be represented in continuous latent spaces.

Graph-based VAEs extended this concept to structures whose atomic connectivity is naturally represented as a graph.

Diffusion models provided a powerful framework for learning complex distributions through controlled noise injection and denoising.

Score-based formulations showed that generation can be interpreted through the gradient of the learned probability density.

Continuous-time diffusion and SDE formulations generalized the process beyond a finite sequence of discrete noise steps.

Equivariant neural networks introduced an essential physical principle: the representation should respect the symmetry of the underlying crystal.

Property conditioning transformed unconditional generation into an inverse-design framework.

The generator can therefore target quantities such as

```text
Band Gap

Formation Energy

Density

Bulk Modulus

Elastic Properties

Magnetic Properties
```

while chemical and structural constraints can restrict the candidate space.

However, generation alone is insufficient.

A generated crystal must be evaluated for

```text
Validity

Novelty

Diversity

Property Accuracy

Distribution Similarity

Stability
```

Machine-learning predictors can provide rapid screening, while DFT can provide a higher-fidelity physical validation stage.

The complete framework is therefore

```text
Generate
   ↓
Validate
   ↓
Predict
   ↓
Rank
   ↓
DFT
   ↓
Experiment
```

and, importantly,

```text
Experiment
   ↓
New Data
   ↓
Model Update
   ↓
Generate Again
```

This creates the foundation for a closed-loop materials-discovery system.

The remaining challenge is resource allocation.

When thousands or millions of possible candidates exist, only a small fraction can be evaluated using expensive DFT calculations or physical experiments.

The scientific problem therefore becomes a sequential decision problem:

```text
Which candidate should be evaluated next?
```

Answering this question requires methods that explicitly consider prediction, uncertainty, information gain, and the value of new observations.

These methods form the foundation of **Active Learning**, which is developed in the next chapter.

---

# Chapter 24 — Final Perspective

Generative AI should not be viewed simply as another machine-learning algorithm.

Its importance in Materials Informatics comes from the fact that it changes the direction of the scientific workflow.

Traditional Materials Informatics often follows

```text
Structure
   ↓
Features
   ↓
Machine Learning
   ↓
Property Prediction
```

Generative Materials Informatics adds the reverse direction:

```text
Scientific Objective
        ↓
Generative Model
        ↓
Candidate Structure
        ↓
Property Evaluation
```

The combination creates a powerful framework:

```text
                  MATERIALS INFORMATICS

                       Prediction
                           ↑
                           │
                 Structure → Property
                           │
                           │
                           ↓
                     Generation
                           │
                    Property Target
```

The long-term vision is larger still.

```text
Human Objective
      ↓
Generative AI
      ↓
Candidate Materials
      ↓
Machine Learning
      ↓
DFT
      ↓
Experiment
      ↓
New Knowledge
      ↓
Model Update
      ↺
```

At this point, generative AI becomes part of an adaptive scientific system rather than an isolated neural network.

The generator proposes.

The predictive models estimate.

Physics evaluates.

The experiment verifies.

The learning system updates.

And the cycle continues.

This is the foundation upon which autonomous materials discovery can be constructed.

The next chapter therefore moves from **generation** to **selection**.

The central question changes from

```text
"What material can the model generate?"
```

to

```text
"Which material should the scientist investigate next?"
```

That is the central problem of **Active Learning**.


# 24.19 — Text-Guided Materials Generation

The approaches developed throughout this chapter condition generative models using numerical properties, compositions, structural information, and learned representations.

A natural extension is to condition generation using **human language**.

Instead of providing a numerical target such as

```text
Band Gap = 2.0 eV
```

a researcher could provide a scientific description such as

```text
Generate a stable semiconductor with a band gap
near 2 eV, low formation energy, and a dense
crystal structure suitable for photovoltaic applications.
```

The generative system would then attempt to translate the textual description into a set of scientifically meaningful constraints.

Conceptually,

```text
Natural-Language Objective
          ↓
      Language Model
          ↓
   Scientific Constraints
          ↓
   Materials Generator
          ↓
    Crystal Candidates
          ↓
 Property / Stability Evaluation
```

This represents a significant change in the interaction between researchers and generative materials systems.

---

# 24.19.1 From Numerical Conditioning to Language Conditioning

Earlier, property-conditioned generation was formulated as

```text
p(C | y)
```

where

```text
C
```

represents the crystal and

```text
y
```

represents the desired property.

For language-guided generation, the conditioning variable becomes a text description.

Let

```text
T
```

represent a natural-language instruction.

The generative objective can then be conceptualized as

```text
p(C | T)
```

For example,

```text
T =
"Generate a semiconductor with a band gap
between 1.5 and 2.0 eV."
```

The model attempts to sample structures from

```text
p(C | T)
```

rather than

```text
p(C)
```

The text therefore acts as a high-level specification of the desired material.

---

# 24.19.2 Why Text Conditioning Is Attractive

Numerical conditioning is precise but limited.

A numerical condition might specify

```text
Band Gap = 2.0 eV
Formation Energy < -2 eV
Density = 5 g/cm³
```

However, a researcher often thinks in terms of broader scientific objectives.

For example,

```text
"Find a lightweight material suitable for
high-temperature structural applications."
```

This statement contains information about

```text
Density

Mechanical Strength

Thermal Stability

Operating Temperature
```

without explicitly specifying numerical values for every quantity.

Natural language therefore provides a flexible interface for expressing scientific intent.

---

# 24.19.3 Language as a High-Level Design Interface

A future materials-generation system could accept instructions such as

```text
Generate a stable oxide semiconductor
with a moderate band gap and strong resistance
to thermal degradation.
```

The system could translate this into

```text
Material Class
        ↓
Chemical Constraints
        ↓
Structural Constraints
        ↓
Property Targets
        ↓
Generative Model
```

The language model would therefore function as an interface between human scientific reasoning and numerical generative models.

The researcher would not necessarily need to specify every mathematical condition manually.

---

# 24.19.4 Text-to-Property Translation

One practical architecture is to first convert language into structured property constraints.

For example,

```text
Text:

"Generate a semiconductor with a band gap
around 2 eV and high mechanical stiffness."
```

could be translated into

```python
constraints = {

    "band_gap": {
        "target": 2.0,
        "tolerance": 0.2
    },

    "bulk_modulus": {
        "minimum": 200
    }
}
```

The generative model can then use these numerical conditions.

The overall architecture becomes

```text
Text
 ↓
Language Model
 ↓
Structured Constraints
 ↓
Property-Conditioned Generator
 ↓
Crystal
```

This approach has an important advantage.

The language model does not need to generate the crystal directly.

Instead, it interprets the scientific request and converts it into a representation that an established materials generator can understand.

---

# 24.19.5 Text-to-Structure Generation

A more ambitious approach attempts to map language directly to a structural representation.

Conceptually,

```text
Text
 ↓
Text Encoder
 ↓
Shared Latent Representation
 ↓
Crystal Decoder
 ↓
Crystal Structure
```

The text encoder converts the scientific description into a vector representation.

The crystal generator then uses that representation to construct a candidate structure.

If

```text
z_T
```

represents the text embedding and

```text
z_C
```

represents the crystal latent representation, the system attempts to learn a meaningful relationship between them.

Ideally,

```text
Similar Scientific Descriptions
        ↓
Similar Latent Representations
        ↓
Related Materials
```

---

# 24.19.6 Joint Text–Materials Representation

A powerful future direction is to place language and materials representations in a shared latent space.

Conceptually,

```text
                  Shared Latent Space

        Text                     Materials
         ●                           ●
       "oxide"                    Oxide A
          ●                         ●
      "perovskite"              Perovskite B
             ●                   ●
       "semiconductor"       Semiconductor C
```

The model learns relationships between scientific descriptions and material representations.

A contrastive objective could encourage matching text–material pairs to become close in latent space while unrelated pairs become distant.

Conceptually,

```text
Matching Pair

Text: "perovskite semiconductor"
                ↕
         Perovskite Crystal

        → Small Distance


Non-Matching Pair

Text: "metallic alloy"
                ↕
         Oxide Crystal

        → Large Distance
```

This creates a foundation for semantic materials search and generation.

---

# 24.19.7 Scientific Text as Training Data

Large amounts of materials knowledge already exist in textual form.

Examples include

```text
Research Papers

Materials Databases

Experimental Reports

DFT Publications

Crystal-Structure Descriptions

Synthesis Procedures

Property Tables

Review Articles
```

A future materials foundation model could potentially learn from these heterogeneous sources.

The training data could contain relationships such as

```text
Material
+
Composition
+
Crystal Structure
+
Property
+
Synthesis Conditions
+
Scientific Description
```

The model would therefore learn not only what structures exist, but also how researchers describe their properties and applications.

---

# 24.19.8 Combining Text with Crystal Graphs

Text-based representations and graph-based representations can be combined.

For example,

```text
Scientific Text
       ↓
Text Encoder
       ↓
Text Embedding
       │
       ├──────────────┐
       │              │
       ↓              ↓
Crystal Graph     Property Model
       ↓              ↓
Graph Encoder    Property Embedding
       │              │
       └──────┬───────┘
              ↓
       Multimodal Representation
              ↓
        Crystal Generator
```

This is a multimodal materials model.

The same system can potentially reason over

```text
Language

Composition

Crystal Structure

Properties

Experimental Information
```

---

# 24.19.9 Multimodal Materials Generation

The future generator may not accept text alone.

A researcher could provide several forms of information simultaneously.

For example,

```text
Composition:
Li-Fe-P-O

Target:
Band Gap ≈ 2 eV

Crystal System:
Orthorhombic

Application:
Electrode material

Reference Structure:
[uploaded structure]
```

The model could combine all of these constraints.

Conceptually,

```text
Text
   +
Composition
   +
Structure
   +
Property
   ↓
Multimodal Generator
   ↓
Candidate Crystals
```

This is potentially much more powerful than any single conditioning mechanism.

---

# 24.19.10 Text-Guided Inverse Design

Text-guided generation can therefore be viewed as an extension of inverse design.

Traditional inverse design asks

```text
Desired Property
       ↓
Structure
```

Text-guided inverse design asks

```text
Scientific Objective
       ↓
Interpreted Constraints
       ↓
Material Structure
```

The first step becomes semantic interpretation.

The second becomes physical generation.

The third becomes scientific validation.

Therefore,

```text
Language
  ↓
Intent
  ↓
Constraints
  ↓
Generation
  ↓
Validation
```

---

# 24.19.11 The Importance of Scientific Grounding

A major challenge is that a language model can generate scientifically plausible-sounding statements without those statements being physically correct.

For example, a model might claim

```text
"This structure should have a high band gap
because of its strong ionic bonding."
```

Such a statement is not sufficient evidence.

The generated material must still be evaluated using appropriate computational or experimental methods.

Therefore,

```text
Language Model
       ↓
Hypothesis
       ↓
Generative Model
       ↓
Physical Evaluation
       ↓
Evidence
```

is safer than

```text
Language Model
       ↓
"Correct" Material
```

The language model should therefore be regarded as a reasoning and interface component rather than a substitute for physical validation.

---

# 24.19.12 Hallucination in Materials Generation

Text-guided systems introduce another failure mode: hallucination.

A language model may generate

```text
Nonexistent Materials

Incorrect Properties

Unsupported Chemical Claims

Invented References

Physically Inconsistent Descriptions
```

This is especially dangerous in scientific applications.

For this reason, generated claims should be separated from verified results.

A useful principle is

```text
Language Generation
        ≠
Scientific Verification
```

The latter requires computational or experimental evidence.

---

# 24.19.13 Retrieval-Augmented Materials Generation

One way to improve reliability is to connect the language model to external materials databases.

The workflow becomes

```text
Research Question
       ↓
Information Retrieval
       ↓
Materials Database
       ↓
Relevant Structures / Properties
       ↓
Language Model
       ↓
Generative Model
       ↓
New Candidates
```

The retrieval system provides factual grounding.

The generative system then operates using known scientific information rather than relying entirely on parameters learned from language data.

---

# 24.19.14 Database-Grounded Generation

Suppose a researcher asks:

```text
Find possible oxide materials related to
known perovskite semiconductors with band gaps
near the visible-light range.
```

The system could first retrieve known materials.

For example,

```text
Database
   ↓
Known Oxides
   ↓
Relevant Band-Gap Range
   ↓
Structural Families
   ↓
Candidate Chemical Spaces
```

The generative model could then explore nearby but previously unknown structures.

This provides a connection between

```text
Known Materials
        ↓
Learned Distribution
        ↓
Novel Candidates
```

---

# 24.19.15 Text-Guided Materials Discovery as a Future Architecture

A long-term architecture may therefore look like

```text
                    Researcher
                         │
                         ▼
                 Natural Language
                         │
                         ▼
               Scientific Interpreter
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
    Database Search   Property Model   Knowledge
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                Materials Generator
                         ↓
                  Crystal Candidates
                         ↓
              Structural Validation
                         ↓
               Property Prediction
                         ↓
                         DFT
                         ↓
                   Experiment
```

This architecture combines language models, generative models, predictive models, databases, simulation, and experiments.

It represents a plausible direction toward integrated AI-assisted materials discovery.

---

# 24.19.16 Human Oversight

Even highly automated systems should retain human scientific oversight.

A researcher should be able to inspect

```text
Generated Structures

Property Predictions

Uncertainty

DFT Results

Chemical Validity

Selection Criteria
```

The final decision should not necessarily be delegated blindly to the generative model.

A useful framework is

```text
AI proposes
   ↓
AI evaluates
   ↓
Physics verifies
   ↓
Scientist decides
```

This is especially important when experiments are expensive or safety-critical.

---

# 24.19.17 Toward Materials Foundation Models

The concepts discussed throughout this chapter point toward a broader idea: **materials foundation models**.

A foundation model could potentially learn simultaneously from

```text
Crystal Structures

Composition

Atomic Graphs

DFT Data

Experimental Data

Scientific Literature

Images

Spectra

Processing Conditions
```

Rather than training a completely separate model for every individual property, a large pretrained representation could serve as a foundation for many downstream tasks.

The workflow could become

```text
Large Materials Dataset
        ↓
Pretraining
        ↓
Materials Foundation Model
        ↓
Fine-Tuning
        ↓
Specific Research Problem
```

Possible downstream applications include

```text
Property Prediction

Structure Generation

Materials Search

Synthesis Planning

Defect Prediction

Phase Stability

Electronic Structure Prediction

Experimental Design
```

---

# 24.19.18 From Models to Scientific Agents

The ultimate extension is beyond individual models.

A scientific AI agent could coordinate several specialized components.

For example,

```text
                 Materials AI Agent
                        │
        ┌───────────────┼────────────────┐
        ↓               ↓                ↓
   Literature       Generator        Predictor
   Retrieval            │                │
        │               ↓                │
        └────────→ Candidate ←───────────┘
                         │
                         ↓
                        DFT
                         │
                         ↓
                    Experimental
                         │
                         ↓
                    New Knowledge
```

The agent would not itself need to perform every calculation.

Instead, it would coordinate specialized tools.

This is conceptually similar to how a human research group operates:

```text
Literature Review
       ↓
Hypothesis
       ↓
Computational Design
       ↓
Simulation
       ↓
Experiment
       ↓
Analysis
```

AI systems may eventually coordinate these stages automatically.

---

# 24.19.19 Limitations of Generative Materials AI

Despite the rapid development of these methods, major limitations remain.

### Data limitations

High-quality materials datasets are often much smaller than datasets used in mainstream generative AI.

Important properties may also be inconsistently calculated or measured.

### Computational limitations

Generating a structure is inexpensive compared with validating it using high-level DFT or experiment.

Therefore, generation can create a bottleneck downstream rather than eliminating one.

### Chemical validity

Neural networks can generate mathematically valid structures that are chemically unreasonable.

### Distribution bias

A model can only learn effectively from the regions represented in its training data.

### Property-prediction error

If the screening model is inaccurate, the generator may optimize toward incorrect predictions.

### Experimental accessibility

A theoretically promising material may still be difficult or impossible to synthesize.

Therefore,

```text
Generative Success
```

does not automatically imply

```text
Experimental Success
```

---

# 24.19.20 The Real Meaning of "Discovery"

A generated crystal is not automatically a discovered material.

A meaningful discovery requires a chain of evidence.

```text
Generated
   ↓
Structurally Valid
   ↓
Chemically Plausible
   ↓
Property-Promising
   ↓
Computationally Validated
   ↓
Potentially Synthesizable
   ↓
Experimentally Verified
```

Only after sufficient validation should a candidate be regarded as a serious materials-discovery result.

This distinction is essential when evaluating claims made by generative AI systems.

---

# 24.19.21 Future Research Directions

Several important research directions remain open.

Future models may improve

```text
Crystal Symmetry Handling

Periodic Boundary Representation

Long-Range Interactions

Defect Generation

Multi-Component Materials

Metastable Structures

Synthesis-Aware Generation

Uncertainty Quantification

Experimental Feasibility Prediction

Text-Guided Generation

Multimodal Learning
```

Defect generation is particularly important because real materials are rarely perfect crystals.

A future generator may need to produce

```text
Vacancies

Interstitials

Substitutional Defects

Antisite Defects

Dislocations
```

rather than only ideal periodic structures.

---

# 24.19.22 Synthesis-Aware Generation

One of the most important future developments is the incorporation of synthesis constraints.

A structure may be thermodynamically attractive but experimentally inaccessible.

A truly useful generator should therefore consider

```text
Composition
   +
Structure
   +
Stability
   +
Synthesis Conditions
```

The objective becomes

```text
Generate a material that is not only
theoretically promising but experimentally accessible.
```

This requires models that understand relationships among

```text
Precursors

Temperature

Pressure

Atmosphere

Reaction Time

Processing Route
```

and final structure.

---

# 24.19.23 Toward Closed-Loop Generative Discovery

The final vision developed in this chapter is a closed-loop system.

```text
                 Scientific Objective
                         ↓
                  Language Interface
                         ↓
                 Generative Model
                         ↓
                  Crystal Candidates
                         ↓
               Structural Screening
                         ↓
                Property Prediction
                         ↓
                Candidate Selection
                         ↓
                       DFT
                         ↓
                    Experiment
                         ↓
                    New Data
                         ↓
                  Model Updating
                         │
                         └──────────→
```

The system continuously improves its understanding of materials space.

This is the point where generative AI connects naturally to the next major concept in Materials Informatics:

**active learning**.

Generation answers:

```text
"What could we make?"
```

Active learning asks:

```text
"What should we evaluate next?"
```

Together they create a much more efficient scientific search strategy.

---

# 24.19.24 Final Perspective on Generative AI

Generative AI represents a transition from passive materials prediction toward active materials design.

The traditional workflow begins with known structures:

```text
Known Material
      ↓
Representation
      ↓
Machine Learning
      ↓
Predicted Property
```

Generative materials discovery reverses the direction:

```text
Desired Outcome
      ↓
Generative Model
      ↓
Candidate Structure
      ↓
Property Evaluation
```

The next generation of systems will combine both directions.

```text
                 Materials Knowledge
                        ↓
              ┌─────────┴─────────┐
              ↓                   ↓
          Prediction          Generation
              ↓                   ↓
              └─────────┬─────────┘
                        ↓
                  Candidate Search
                        ↓
                       DFT
                        ↓
                    Experiment
                        ↓
                  New Knowledge
                        ↓
                  Model Update
```

The ultimate goal is not to replace materials scientists.

It is to expand the number of scientifically meaningful hypotheses that researchers can investigate.

Generative AI can explore enormous regions of materials space that would be impossible to examine manually.

But the scientific workflow remains grounded in the same fundamental principles:

```text
Physical Reasoning
+
Chemical Knowledge
+
Mathematical Modeling
+
Machine Learning
+
Simulation
+
Experiment
```

Generative AI becomes most powerful when these components are integrated rather than treated independently.

With this final perspective, the generative-AI framework developed in Chapter 24 is complete.

The next chapter turns to the problem of **deciding which candidate to investigate next**.

That problem leads directly to **Active Learning**, Bayesian Optimization, uncertainty-aware selection, and closed-loop materials discovery.


