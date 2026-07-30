# Chapter 12 — Neural Networks for Materials Science

---

# 12.1 Introduction

Machine learning has transformed computational materials science by enabling researchers to predict material properties directly from data instead of performing expensive quantum mechanical calculations for every new material.

Throughout the previous chapters, we studied several powerful machine learning algorithms, including

- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost

These algorithms are collectively referred to as **classical machine learning methods**.

They have become indispensable tools in materials informatics because they are

- accurate,
- computationally efficient,
- relatively easy to interpret,
- and require comparatively small datasets.

For many materials science problems, classical machine learning remains the preferred choice.

However, as materials databases have grown from thousands of compounds to millions of structures, researchers have encountered increasingly complex problems that classical machine learning struggles to solve.

Examples include

- predicting multiple material properties simultaneously,
- learning directly from crystal structures,
- discovering complex nonlinear relationships,
- generating meaningful material representations,
- and modeling atomic interactions without manually engineered descriptors.

These challenges motivated the development of **deep learning**.

Unlike classical machine learning, deep learning automatically learns useful representations directly from data, reducing the need for extensive manual feature engineering.

Today, deep learning has become one of the most active research areas in computational materials science and serves as the foundation for modern Graph Neural Networks (GNNs), Crystal Graph Convolutional Neural Networks (CGCNN), MEGNet, M3GNet, ALIGNN, and many other state-of-the-art materials prediction models.

In this chapter, we will develop a complete understanding of neural networks—from the simplest artificial neuron to training deep neural networks using PyTorch.

This knowledge will provide the mathematical and computational foundation required for the graph-based deep learning models introduced in the following chapters.

---

## 12.1.1 Learning Objectives

After completing this chapter, you should be able to

- explain why deep learning was developed,
- identify the limitations of classical machine learning,
- understand the architecture of artificial neural networks,
- describe the role of neurons, weights, and biases,
- understand activation functions and nonlinear modeling,
- explain forward propagation,
- understand loss functions and optimization,
- derive the intuition behind backpropagation,
- build neural networks using PyTorch,
- train neural networks for materials property prediction,
- and evaluate when deep learning is preferable to traditional machine learning methods.

---

## 12.1.2 Position of This Chapter in the Book

The previous chapters focused on supervised learning using manually engineered descriptors.

A typical workflow looked like

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

Hundreds of Descriptors

↓

Machine Learning Model

↓

Property Prediction
```

This workflow has proven extremely successful for many materials science applications.

However, one important question remains.

> **What if the descriptors themselves are not sufficient?**

Or even more fundamentally,

> **What if we do not want to manually design descriptors at all?**

Deep learning addresses these questions by allowing the model to automatically learn useful representations directly from data.

This represents one of the biggest conceptual shifts in modern machine learning.

---

## 12.1.3 From Feature Engineering to Representation Learning

One of the central themes of this book has been **feature engineering**.

Using libraries such as

- `pymatgen`
- `matminer`

we converted crystal structures into numerical descriptors.

For example,

```text
Crystal Structure

↓

Average Atomic Number

↓

Electronegativity Difference

↓

Packing Fraction

↓

Atomic Radius Statistics

↓

Band Center

↓

Hundreds of Numerical Features
```

These descriptors become the input for machine learning algorithms.

Feature engineering is extremely powerful because it incorporates scientific knowledge into the model.

The descriptors often encode

- chemistry,
- crystallography,
- electronic structure,
- and atomic interactions.

Consequently,

algorithms such as Random Forest and XGBoost can produce highly accurate predictions using relatively small datasets.

However,

this approach has an important limitation.

The quality of the model depends heavily on the quality of the descriptors.

If important physical information is missing,

the machine learning model cannot recover it automatically.

In other words,

```text
Poor Features

↓

Poor Model
```

regardless of which machine learning algorithm is used.

Deep learning introduces a different philosophy.

Instead of requiring researchers to manually design informative descriptors,

the neural network learns useful internal representations directly from the training data.

This process is called

**representation learning**.

Rather than relying entirely on handcrafted descriptors,

the network gradually constructs increasingly informative representations during training.

Conceptually,

the workflow becomes

```text
Raw Input

↓

Neural Network

↓

Learn Internal Representations

↓

Predict Material Property
```

Representation learning is one of the defining characteristics of deep learning and is the reason why neural networks have become so successful in fields such as computer vision, natural language processing, speech recognition, and increasingly, materials science.

---

## 12.1.4 Why Deep Learning Became Necessary

Classical machine learning algorithms have achieved remarkable success in materials informatics.

For many prediction tasks,

Random Forest,

Gradient Boosting,

and XGBoost

remain excellent choices.

So why did researchers invest enormous effort into developing deep learning?

The answer lies in the complexity of modern scientific datasets.

Materials science now routinely produces

- millions of DFT calculations,
- extremely high-dimensional datasets,
- complex crystal structures,
- atomistic simulations,
- and experimental databases containing vast amounts of information.

These datasets contain relationships that are often too complicated to describe using manually engineered descriptors alone.

Consider predicting the formation energy of a crystal.

The prediction depends on

- atomic species,
- coordination environments,
- bond lengths,
- bond angles,
- crystal symmetry,
- local chemical environments,
- long-range interactions,
- and many other factors.

Designing descriptors that perfectly capture all of these effects is extremely difficult.

Deep learning addresses this challenge by allowing the model itself to discover the most useful representations during training.

Rather than relying exclusively on human-designed descriptors,

the model learns which patterns are most important for the prediction task.

---

## 12.1.5 Evolution of Machine Learning in Materials Science

The development of machine learning in materials science can be viewed as a gradual progression.

```text
Physics-Based Models

↓

Empirical Models

↓

Classical Machine Learning

↓

Deep Learning

↓

Graph Neural Networks

↓

Foundation Models
```

Each stage represents an increase in the ability of models to automatically learn complex relationships from data.

Initially,

researchers relied almost entirely on

- analytical equations,
- empirical relationships,
- and physical intuition.

Later,

classical machine learning enabled prediction using engineered descriptors.

Today,

deep learning and graph neural networks allow models to learn directly from crystal structures with minimal manual feature engineering.

This progression reflects one of the most important trends in modern computational materials science.

---

## 12.1.6 Deep Learning in Modern Materials Informatics

Deep learning now supports a wide variety of materials science applications.

Examples include

- formation energy prediction,
- band gap prediction,
- elastic property prediction,
- density prediction,
- thermal conductivity prediction,
- battery material discovery,
- catalyst screening,
- interatomic potential development,
- crystal structure generation,
- defect prediction,
- phase stability prediction,
- and inverse materials design.

Many state-of-the-art models used by leading research groups are based on deep neural networks.

Examples include

- CGCNN,
- MEGNet,
- M3GNet,
- ALIGNN,
- CHGNet,
- MatterSim,
- and other graph-based architectures.

Although these models differ significantly in their architectures,

they all rely on the same fundamental principles introduced in this chapter.

Understanding artificial neural networks is therefore essential before studying graph neural networks.

---

## 12.1.7 Roadmap for This Chapter

This chapter gradually builds from the simplest concepts to complete deep learning models.

We will begin by understanding the biological inspiration behind artificial neural networks and then study how an artificial neuron performs computation.

Next,

we will investigate

- network architectures,
- activation functions,
- forward propagation,
- loss functions,
- optimization algorithms,
- gradient descent,
- and backpropagation.

Finally,

we will implement neural networks using **PyTorch** and apply them to materials property prediction problems.

By the end of the chapter, you will possess the theoretical foundation required to understand the graph neural network architectures presented in the following chapters, where crystal structures themselves become the input to deep learning models rather than manually engineered descriptors.

## 12.2 Why Deep Learning?

Deep learning is one of the most significant developments in modern artificial intelligence. It has transformed fields such as computer vision, natural language processing, speech recognition, robotics, and increasingly, computational materials science.

Before studying the mathematics of neural networks, it is important to answer a fundamental question.

> **Why do we need deep learning when classical machine learning algorithms such as Random Forest and XGBoost already perform so well?**

The answer is not that deep learning is always better.

In fact, for many materials science problems, **classical machine learning remains the preferred choice**.

Instead, deep learning was developed to solve problems that become difficult—or impossible—for traditional machine learning algorithms.

To understand this transition, we first need to examine the strengths and limitations of classical machine learning.

---

## 12.2.1 Classical Machine Learning Has Been Extremely Successful

Throughout this book, we have used algorithms such as

- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost

These algorithms have been successfully applied to predict

- band gaps,
- formation energies,
- elastic moduli,
- thermal conductivity,
- dielectric constants,
- and many other material properties.

A typical workflow looks like

```text
Crystal Structure

↓

Descriptor Generation

↓

Machine Learning Model

↓

Property Prediction
```

For many datasets,

this workflow produces excellent results.

In fact,

Random Forest and XGBoost often outperform neural networks when

- datasets are relatively small,
- descriptors are carefully engineered,
- and interpretability is important.

This is why classical machine learning remains widely used in materials informatics.

---

## 12.2.2 The Feature Engineering Bottleneck

Although classical machine learning algorithms are powerful,

they depend heavily on one critical component:

**feature engineering**.

Recall our previous workflow.

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

300 Descriptors

↓

Random Forest

↓

Band Gap
```

Notice something important.

The machine learning model never sees the crystal structure directly.

Instead,

it only sees the descriptors that we provide.

If the descriptors contain sufficient information,

the model performs well.

If important information is missing,

the model cannot recover it.

For example,

suppose we compute

```text
Average Atomic Number

Atomic Radius

Density

Packing Fraction
```

but ignore

```text
Bond Angles

Local Coordination

Atomic Connectivity
```

If these missing quantities strongly influence the target property,

the model will struggle regardless of how sophisticated the algorithm is.

This illustrates one of the greatest limitations of classical machine learning.

```text
Prediction Quality

↓

Feature Quality

↓

Human Knowledge
```

The model depends on human-designed features.

---

## 12.2.3 Manual Feature Engineering Requires Scientific Expertise

Designing good descriptors is far from trivial.

Consider the prediction of formation energy.

Which descriptors should we include?

Possible choices include

- atomic number,
- atomic mass,
- electronegativity,
- ionic radius,
- covalent radius,
- oxidation state,
- coordination number,
- lattice volume,
- density,
- packing fraction,
- bond statistics,
- orbital information,
- and hundreds of others.

Each descriptor reflects a hypothesis about which physical quantities influence the property.

Designing these descriptors requires

- chemistry,
- crystallography,
- solid-state physics,
- and domain expertise.

Consequently,

feature engineering is both

- scientifically valuable,
- and time consuming.

---

## 12.2.4 There Is No Perfect Descriptor Set

Even experienced researchers cannot guarantee that their descriptor set contains all relevant information.

Suppose two materials have

- identical average atomic numbers,
- similar densities,
- similar lattice constants,

but completely different atomic arrangements.

Their properties may differ dramatically.

If the descriptors fail to capture these structural differences,

the machine learning model cannot distinguish the materials.

The problem is not the learning algorithm.

The problem is that important information has already been discarded before training even begins.

---

## 12.2.5 Increasing Dataset Complexity

Modern materials databases are growing rapidly.

Examples include

- Materials Project,
- Open Quantum Materials Database (OQMD),
- AFLOW,
- NOMAD,
- and many institutional databases.

These repositories contain

- hundreds of thousands,
- or even millions,

of crystal structures.

Each structure may contain

- dozens of atoms,
- complex geometries,
- multiple chemical species,
- and intricate local environments.

Representing such complexity using only handcrafted descriptors becomes increasingly difficult.

---

## 12.2.6 Complex Relationships Are Difficult to Engineer

Many material properties depend on interactions occurring at multiple scales.

For example,

electronic properties depend on

- local atomic environments,
- bonding,
- crystal symmetry,
- orbital interactions,
- and long-range structural effects.

These relationships are highly nonlinear.

Trying to summarize all of this information with a fixed list of descriptors is an enormous challenge.

Deep learning was developed to reduce this dependence on manual descriptor design.

---

## 12.2.7 From Feature Engineering to Feature Learning

The key conceptual difference between classical machine learning and deep learning can be summarized in one diagram.

### Classical Machine Learning

```text
Crystal Structure

↓

Human Designs Features

↓

Machine Learning Model

↓

Prediction
```

Most of the intelligence lies in the feature engineering stage.

---

### Deep Learning

```text
Raw Input

↓

Neural Network

↓

Automatically Learn Features

↓

Prediction
```

Here,

the neural network learns useful internal representations directly from data.

Instead of requiring researchers to specify every important descriptor,

the model gradually discovers informative patterns during training.

This process is called

**feature learning**

or

**representation learning**.

---

## 12.2.8 Learning Hierarchical Representations

One of the defining characteristics of deep learning is that representations are learned in multiple stages.

Instead of learning a single transformation,

the network gradually builds increasingly abstract representations.

Conceptually,

the process resembles

```text
Input

↓

Simple Patterns

↓

Intermediate Patterns

↓

Complex Patterns

↓

Prediction
```

For image recognition,

this hierarchy might become

```text
Pixels

↓

Edges

↓

Shapes

↓

Objects

↓

Classification
```

For materials science,

the hierarchy might resemble

```text
Atomic Information

↓

Local Environments

↓

Bonding Patterns

↓

Structural Motifs

↓

Material Properties
```

Although these internal representations are learned automatically,

they often correspond to meaningful physical concepts.

---

## 12.2.9 Why "Deep" Learning?

The word

**deep**

refers to the number of computational layers inside the neural network.

A shallow network may contain

```text
Input

↓

Hidden Layer

↓

Output
```

A deep network contains many hidden layers.

```text
Input

↓

Hidden Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Output
```

Each additional layer allows the network to construct increasingly sophisticated internal representations.

This layered representation learning is one of the reasons deep learning can model extremely complex relationships.

---

## 12.2.10 Is Deep Learning Always Better?

A common misconception is that deep learning always outperforms classical machine learning.

This is **not true**.

For many materials informatics problems,

Random Forest or XGBoost may produce

- higher accuracy,
- faster training,
- lower computational cost,
- and better interpretability.

Deep learning generally becomes advantageous when

- datasets are very large,
- relationships are highly complex,
- raw structured data are available,
- or manual feature engineering becomes difficult.

Choosing between classical machine learning and deep learning should therefore depend on

- the dataset,
- the problem,
- the available computational resources,
- and the scientific objective.

---

## 12.2.11 Deep Learning in Materials Science

Modern deep learning has enabled entirely new approaches to materials discovery.

Instead of relying exclusively on handcrafted descriptors,

researchers can train neural networks directly on

- crystal structures,
- atomic coordinates,
- neighbor relationships,
- graphs,
- electron density,
- microscopy images,
- and even text extracted from scientific literature.

Examples include

- predicting formation energies directly from crystal graphs,
- learning interatomic potentials,
- discovering new battery materials,
- predicting catalytic activity,
- generating hypothetical crystal structures,
- and accelerating high-throughput materials screening.

These advances would be extremely difficult using traditional descriptor-based approaches alone.

---

## 12.2.12 Deep Learning as the Foundation for Graph Neural Networks

Although this chapter focuses on conventional neural networks,

its true purpose extends beyond simple property prediction.

Graph Neural Networks,

which we will study in the next chapters,

are themselves specialized neural networks.

They inherit fundamental concepts such as

- neurons,
- weights,
- activation functions,
- forward propagation,
- loss functions,
- optimization,
- and backpropagation.

The primary difference is that their input is no longer a vector of handcrafted descriptors.

Instead,

they operate directly on crystal graphs.

Understanding deep learning is therefore an essential prerequisite for understanding modern graph-based materials informatics.

---

## 12.2.13 Summary

Deep learning was not developed to replace classical machine learning.

Rather,

it was developed to overcome the limitations associated with manual feature engineering and to enable models to learn useful representations directly from data.

As datasets become larger and more complex,

the ability to automatically learn representations becomes increasingly valuable.

This transition from handcrafted descriptors to learned representations marks one of the most important conceptual shifts in modern artificial intelligence.

In the next section, we begin studying the fundamental building block of every neural network: the **artificial neuron**. From this simple computational unit, increasingly complex neural network architectures can be constructed.


## 12.3 Artificial Neural Networks

Before we study deep neural networks, we must first understand their smallest building block.

Just as every crystal is built from atoms,

every neural network is built from **artificial neurons**.

Regardless of whether the network contains

- 10 neurons,
- 1,000 neurons,
- or millions of neurons,

the basic computation performed by each neuron is essentially the same.

Understanding how a single neuron works is therefore the foundation for understanding every deep learning model, including Graph Neural Networks (GNNs), Crystal Graph Convolutional Neural Networks (CGCNN), MEGNet, M3GNet, and ALIGNN.

---

### 12.3.1 Inspiration from Biology

The idea of artificial neural networks was inspired by the human brain.

The human brain contains approximately

```text
86 Billion Neurons
```

These neurons communicate through electrical and chemical signals.

A biological neuron receives signals from many neighboring neurons, processes the information, and decides whether to send a signal to other neurons.

A simplified biological neuron looks like

```text
        Inputs

          │
      Dendrites
          │
          ▼
    +-------------+
    |   Cell Body |
    +-------------+
          │
        Axon
          │
          ▼
        Output
```

Although the human brain is extraordinarily complex,

this basic idea inspired the development of artificial neurons.

It is important to understand that **artificial neural networks are only loosely inspired by biology**.

They are mathematical models rather than accurate simulations of biological brains.

---

### 12.3.2 The Artificial Neuron

An artificial neuron performs a very simple computation.

It

1. receives numerical inputs,
2. assigns importance to each input,
3. combines the information,
4. applies a mathematical transformation,
5. produces an output.

Conceptually,

```text
Inputs

↓

Weighted Combination

↓

Activation Function

↓

Output
```

Every neuron in a neural network follows this same sequence.

---

### 12.3.3 Inputs

Suppose we want to predict the formation energy of a material.

The inputs might be descriptors such as

- average atomic number,
- density,
- average electronegativity,
- lattice volume,
- packing fraction.

These inputs can be represented as

```text
x1

x2

x3

x4

...

xn
```

where

```text
n
```

is the number of input features.

For example,

```text
x1 = Average Atomic Number

x2 = Density

x3 = Average Electronegativity

x4 = Packing Fraction
```

Each input provides one piece of information about the material.

---

### 12.3.4 Weights

Not every feature is equally important.

For predicting band gaps,

electronegativity may be more informative than density.

For predicting elastic properties,

bond strength may be more important than atomic mass.

The neural network learns this automatically using **weights**.

Each input has an associated weight.

```text
Input

↓

Weight

↓

Contribution
```

or

```text
x1 → w1

x2 → w2

x3 → w3

...

xn → wn
```

Large weights indicate that a feature has a strong influence on the prediction.

Small weights indicate that the feature contributes relatively little.

Unlike handcrafted rules,

the weights are **learned automatically** during training.

---

### 12.3.5 The Weighted Sum

The neuron combines all inputs using their corresponding weights.

Conceptually,

```text
Output Before Activation

=

(w1 × x1)

+

(w2 × x2)

+

(w3 × x3)

+

...

+

(wn × xn)
```

This quantity is often called the

- weighted sum,
- linear combination,
- or pre-activation value.

Notice that each feature contributes according to both

- its numerical value,
- and its learned weight.

---

### 12.3.6 Bias

Most neurons also include an additional parameter called the **bias**.

The bias acts like an adjustable offset.

Instead of computing only

```text
Weighted Sum
```

the neuron computes

```text
Weighted Sum

+

Bias
```

Conceptually,

```text
Output Before Activation

=

(w1 × x1)

+

(w2 × x2)

+

...

+

Bias
```

The bias provides additional flexibility by allowing the neuron to shift its output independently of the inputs.

Without the bias,

the neuron would be unnecessarily restricted in the functions it can learn.

---

### 12.3.7 Why Are Weights and Biases Needed?

Consider predicting band gap.

Suppose two descriptors are available.

```text
Density

Electronegativity
```

If electronegativity is much more important,

the network should assign

```text
Large Weight

↓

Electronegativity
```

and

```text
Small Weight

↓

Density
```

Similarly,

the bias allows the prediction to shift upward or downward even when all inputs remain unchanged.

During training,

both the weights and the bias are continuously adjusted to minimize prediction errors.

---

### 12.3.8 The Activation Function

After computing the weighted sum,

the neuron passes the result through another mathematical operation called the **activation function**.

The complete workflow becomes

```text
Inputs

↓

Weighted Sum

↓

Bias

↓

Activation Function

↓

Output
```

The activation function introduces **nonlinearity** into the neural network.

Without activation functions,

even very deep neural networks would behave like simple linear models.

We will study activation functions in detail in the next section.

---

### 12.3.9 Complete Structure of an Artificial Neuron

A complete neuron can be represented as

```text
             x1
              │
            × w1
              │

             x2
              │
            × w2
              │

             x3
              │
            × w3
              │

              ▼

      Weighted Sum

              │

           + Bias

              │

              ▼

    Activation Function

              │

              ▼

           Output
```

Although this diagram appears simple,

every modern deep learning model is built from millions of these basic computational units.

---

### 12.3.10 A Materials Science Example

Suppose a neural network is predicting formation energy.

The input descriptors might be

```text
Average Atomic Number

↓

Density

↓

Electronegativity

↓

Average Atomic Radius

↓

Packing Fraction
```

Each descriptor enters the neuron.

The neuron learns

- which descriptors are important,
- which descriptors should be ignored,
- and how they should be combined.

After training,

the neuron automatically determines relationships that would be difficult to design manually.

---

### 12.3.11 Learning the Weights

One of the most remarkable properties of neural networks is that the weights are **not chosen by the researcher**.

Instead,

they are learned from data.

Initially,

the weights are assigned random values.

```text
Random Weights

↓

Prediction

↓

Error

↓

Update Weights

↓

Better Prediction

↓

Repeat
```

This learning process continues until the prediction error becomes sufficiently small.

The algorithm responsible for updating the weights is called **backpropagation**, which we will study later in this chapter.

---

### 12.3.12 Artificial Neurons in Deep Learning

A single neuron is capable of performing only a simple computation.

However,

when thousands or millions of neurons are connected together,

they become capable of learning extremely complex relationships.

For example,

modern materials deep learning models may contain

```text
Input Features

↓

Hundreds of Neurons

↓

Thousands of Neurons

↓

Graph Layers

↓

Prediction
```

Although these networks appear highly sophisticated,

they are ultimately composed of the same simple neuron that we have introduced in this section.

Understanding this basic computational unit is therefore the first step toward understanding every deep learning architecture used in modern materials informatics.

---

### 12.3.13 Summary

An artificial neuron is the fundamental computational unit of a neural network.

Each neuron

- receives numerical inputs,
- assigns a weight to each input,
- computes a weighted sum,
- adds a bias,
- applies an activation function,
- and produces an output.

During training,

the neuron automatically learns the values of its weights and bias so that prediction errors become progressively smaller.

Although a single neuron performs only a simple calculation, connecting many neurons together enables deep neural networks to model the highly complex relationships encountered in materials science.

In the next section, we will learn how individual neurons are connected to form **layers** and how multiple layers combine to create a complete artificial neural network.

## 12.4 Neural Network Architecture

In the previous section, we studied the **artificial neuron**, the fundamental computational unit of every neural network.

A single neuron, however, has very limited learning capability.

It can only perform one relatively simple transformation on the input data.

Real-world materials science problems are far more complicated.

For example, predicting

- formation energy,
- band gap,
- elastic modulus,
- dielectric constant,
- thermal conductivity,

requires understanding highly nonlinear relationships between hundreds of input features.

To learn these complex relationships, neurons are **connected together into layers**, and multiple layers form a **neural network**.

Understanding how these layers work is the next step toward understanding deep learning.

---

### 12.4.1 From a Single Neuron to a Network

Imagine trying to solve a difficult engineering problem using only one scientist.

That scientist can only perform a limited amount of work.

Now imagine assembling an entire research team.

Each researcher specializes in one task.

The output of one researcher becomes the input of another.

Together, they can solve problems that are impossible for one person alone.

Neural networks operate in exactly the same way.

Instead of researchers,

they consist of neurons.

Instead of discussions,

they exchange numerical values.

Instead of scientific conclusions,

they produce predictions.

Conceptually,

```text
Neuron

↓

Small Computation

↓

Many Neurons

↓

Complex Computation
```

---

### 12.4.2 What Is a Layer?

A **layer** is simply a collection of neurons that perform computations simultaneously.

For example,

```text
○   ○   ○   ○
```

represents one layer containing four neurons.

Each neuron receives inputs,

performs its own computation,

and produces one output.

These outputs become the inputs for the next layer.

---

### 12.4.3 The Three Main Types of Layers

Most neural networks consist of three basic types of layers.

```text
Input Layer

↓

Hidden Layer(s)

↓

Output Layer
```

Each layer has a different purpose.

---

### 12.4.4 Input Layer

The input layer is the entry point of the neural network.

It does **not** perform complicated calculations.

Its primary purpose is to receive the input features and pass them to the first hidden layer.

Suppose we are predicting band gap using five descriptors.

The input layer may contain

```text
Average Atomic Number

Density

Average Electronegativity

Packing Fraction

Average Atomic Radius
```

Diagrammatically,

```text
Input Layer

○

○

○

○

○
```

Each neuron in the input layer represents one feature.

If the dataset contains

```text
250 descriptors
```

then the input layer contains

```text
250 input neurons.
```

The number of input neurons therefore equals the number of input features.

---

### 12.4.5 Hidden Layers

The hidden layers perform almost all of the learning.

They receive information from the previous layer,

combine it,

transform it,

and pass it to the next layer.

Example

```text
Input Layer

↓

○ ○ ○ ○ ○

↓

Hidden Layer

↓

○ ○ ○ ○

↓

Hidden Layer

↓

○ ○ ○

↓

Output
```

Notice that the hidden layers are located

between

the input

and

the output.

This is why they are called **hidden** layers.

The values inside these layers are not directly observed by the user.

Instead,

they represent internal representations learned automatically by the network.

---

### 12.4.6 Why Multiple Hidden Layers?

Suppose we use only one hidden layer.

The network can certainly learn many useful relationships.

However,

some scientific problems require learning increasingly abstract concepts.

Multiple hidden layers allow the network to build knowledge gradually.

Conceptually,

```text
Input Features

↓

Simple Relationships

↓

Intermediate Relationships

↓

Complex Relationships

↓

Prediction
```

Each layer extracts information from the previous one.

For materials science,

this might resemble

```text
Atomic Properties

↓

Local Chemistry

↓

Bonding Environment

↓

Crystal Characteristics

↓

Material Property
```

Although the network is never explicitly programmed to perform these steps,

it often learns similar hierarchical representations automatically.

---

### 12.4.7 Why Is It Called "Deep" Learning?

The word

**deep**

refers to the number of hidden layers.

A shallow neural network might look like

```text
Input

↓

Hidden Layer

↓

Output
```

A deep neural network may contain

```text
Input

↓

Hidden Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Output
```

Modern deep learning models may contain

- dozens,
- hundreds,
- or even thousands

of layers.

Graph Neural Networks also become "deep" by stacking many graph convolution layers.

---

### 12.4.8 Output Layer

The final layer of the network is the **output layer**.

Its purpose is to produce the final prediction.

The number of output neurons depends on the problem.

---

#### Regression

If we are predicting

```text
Band Gap
```

only one value is required.

Therefore,

```text
Output Layer

↓

○
```

One neuron produces

```text
Predicted Band Gap
```

---

#### Multiple Property Prediction

Suppose we wish to predict

- band gap,
- formation energy,
- bulk modulus.

The output layer becomes

```text
○

○

○
```

Each neuron predicts one property.

---

#### Classification

Suppose we want to classify materials into

- Metal
- Semiconductor
- Insulator

The output layer contains

```text
○

○

○
```

where each neuron corresponds to one class.

---

### 12.4.9 Fully Connected Layers

The most common neural network architecture is called a

**fully connected neural network**.

In this architecture,

every neuron connects to every neuron in the next layer.

Example

```text
Input Layer

○     ○     ○

 \   /|\   /

  \ / | \ /

   ○  ○  ○

  /|\ | /|\

 / | \|/ | \

○   ○   ○
```

Although this produces many connections,

it allows information from every input feature to influence every neuron in the next layer.

Later,

we will see that Graph Neural Networks use a different connection strategy.

---

### 12.4.10 Information Flow

Information always moves in one direction during prediction.

```text
Input

↓

Hidden Layer

↓

Hidden Layer

↓

Output
```

This process is called

**forward propagation**.

Each layer receives information,

processes it,

and passes the transformed representation forward.

No information moves backward during prediction.

The backward movement occurs only during training when the network updates its weights.

---

### 12.4.11 Example Neural Network for Materials Property Prediction

Suppose we wish to predict formation energy using

20 descriptors generated by `matminer`.

A possible network architecture is

```text
20 Input Features

↓

64 Neurons

↓

32 Neurons

↓

16 Neurons

↓

1 Output

(Formation Energy)
```

This architecture is often written as

```text
20 → 64 → 32 → 16 → 1
```

Each arrow represents a fully connected layer.

---

### 12.4.12 How Does Information Change Across Layers?

The information entering the network gradually changes as it passes through each layer.

```text
Input Descriptors

↓

Hidden Layer 1

↓

Hidden Layer 2

↓

Hidden Layer 3

↓

Prediction
```

The first hidden layer may learn simple combinations of descriptors.

The second layer combines those representations into more meaningful patterns.

The third layer learns even more abstract representations.

Finally,

the output layer converts the learned representation into the predicted material property.

This gradual refinement of information is one of the defining characteristics of deep learning.

---

### 12.4.13 Choosing the Number of Hidden Layers

One of the most common questions is

> How many hidden layers should a neural network contain?

Unfortunately,

there is no universal answer.

The optimal architecture depends on

- dataset size,
- problem complexity,
- computational resources,
- and prediction accuracy.

General guidelines include

- Small datasets often perform well with one or two hidden layers.
- Larger datasets may benefit from deeper networks.
- Extremely deep networks require much more computational power.
- Adding more layers does not always improve performance.

In practice,

the architecture is treated as a **hyperparameter** and is optimized experimentally.

---

### 12.4.14 Choosing the Number of Neurons

Each hidden layer also requires selecting the number of neurons.

For example,

```text
20 → 32 → 1

20 → 64 → 32 → 1

20 → 128 → 64 → 32 → 1
```

All three are valid architectures.

Increasing the number of neurons increases the network's capacity to learn complex relationships.

However,

larger networks

- require more memory,
- train more slowly,
- and are more likely to overfit when datasets are small.

Choosing the appropriate network size is therefore an important design decision.

---

### 12.4.15 Neural Network Architecture in Materials Science

A typical descriptor-based workflow in materials informatics looks like

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

StandardScaler

↓

Neural Network

↓

Predicted Property
```

Later in this book,

we will replace the descriptor generation step with Graph Neural Networks.

Instead of manually computed descriptors,

the network will receive the crystal structure itself as input.

That transition represents one of the biggest advances in modern materials informatics.

---

### 12.4.16 Summary

A neural network is constructed by connecting many artificial neurons into layers.

Most networks consist of

- an input layer,
- one or more hidden layers,
- and an output layer.

The hidden layers perform the learning by transforming the input data into increasingly informative internal representations.

The depth of the network is determined by the number of hidden layers, while the learning capacity depends on both the number of layers and the number of neurons within each layer.

Understanding this layered architecture prepares us for the next essential component of deep learning: **activation functions**. Without activation functions, even the deepest neural network would behave like a simple linear model, severely limiting its ability to learn the complex nonlinear relationships that govern material properties.

## 12.5 Activation Functions

In the previous sections, we learned that a neuron performs three basic operations.

1. It receives inputs.
2. It computes a weighted sum.
3. It produces an output.

At first glance, this appears sufficient for building a neural network.

However, there is a fundamental problem.

Suppose we simply connect many neurons together without performing any additional operation.

```text
Input

↓

Weighted Sum

↓

Weighted Sum

↓

Weighted Sum

↓

Output
```

Surprisingly,

this entire network is mathematically equivalent to **one single linear transformation**.

In other words,

adding more layers would not make the network any more powerful.

To overcome this limitation, neural networks introduce an additional mathematical operation known as the **activation function**.

Activation functions are one of the most important components of deep learning because they enable neural networks to learn **nonlinear relationships**.

Without activation functions,

deep learning would not exist.

---

### 12.5.1 Why Are Activation Functions Necessary?

Consider predicting the band gap of a material.

Band gap depends on many interacting factors, including

- atomic composition,
- crystal symmetry,
- orbital overlap,
- local coordination,
- bond lengths,
- bond angles,
- electronic structure.

These relationships are highly nonlinear.

For example,

doubling one descriptor does not necessarily double the band gap.

Similarly,

changing one atom may produce a completely different electronic structure.

These complex behaviors cannot be accurately represented using only linear equations.

Therefore,

the neural network must be able to learn nonlinear functions.

Activation functions provide this capability.

Conceptually,

```text
Weighted Sum

↓

Activation Function

↓

Nonlinear Output
```

---

### 12.5.2 Linear Models Cannot Learn Complex Relationships

Imagine attempting to fit the following data.

```text
Property

^

|

|          ●

|      ●

|   ●

| ●

+---------------------------->

          Descriptor
```

A straight line may describe some datasets reasonably well.

However,

many materials science relationships resemble

```text
Property

^

|

|       ●

|    ●

|  ●

|      ●

|           ●

+---------------------------->

          Descriptor
```

These nonlinear patterns cannot be captured by a purely linear model.

Even if we stack many linear layers,

the result remains linear.

Activation functions solve this problem by introducing nonlinearity after every layer.

---

### 12.5.3 What Does an Activation Function Do?

An activation function receives the weighted sum from a neuron and transforms it into another value.

Conceptually,

```text
Inputs

↓

Weighted Sum

↓

Activation Function

↓

Neuron Output
```

Every neuron performs this transformation independently.

Although the mathematical forms of different activation functions vary,

their purpose is the same.

They allow neural networks to approximate highly complex functions.

---

### 12.5.4 Characteristics of a Good Activation Function

A useful activation function should

- introduce nonlinearity,
- be computationally efficient,
- allow stable training,
- support gradient-based optimization,
- and work reliably for deep networks.

Different activation functions satisfy these requirements to different degrees.

Consequently,

many activation functions have been developed over the history of deep learning.

The most important ones are

- Linear
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- GELU
- Softmax

We will study each of these individually.

---

### 12.5.5 Linear Activation

The simplest activation function is the **linear activation**.

Conceptually,

```text
Input

↓

Output
```

Nothing changes.

The output remains proportional to the input.

Graphically,

```text
Output

^

|

|        /

|      /

|    /

|  /

|/

+---------------------------->

            Input
```

Although linear activation is simple,

it has one major limitation.

Multiple linear layers are mathematically equivalent to one linear layer.

Therefore,

linear activation is rarely used in hidden layers.

It is commonly used only in the **output layer** of regression problems.

Examples include

- band gap prediction,
- formation energy prediction,
- density prediction,
- elastic modulus prediction.

---

### 12.5.6 Sigmoid Activation Function

Historically,

one of the earliest activation functions was the **sigmoid** function.

The sigmoid transforms any input into a value between

```text
0

and

1
```

Graphically,

```text
Output

1.0 |                  _______

    |               __/

0.5 |------------__/

    |         _/

0.0 |_______/

    +---------------------------->

               Input
```

Because its output resembles an "S" shape,

it is sometimes called the

**logistic function**.

The sigmoid was widely used in early neural networks.

---

### 12.5.7 Advantages of Sigmoid

The sigmoid function has several useful properties.

- Smooth output
- Continuous
- Differentiable
- Produces probabilities between 0 and 1

Because of these properties,

sigmoid is still used in

- binary classification,
- probability estimation,
- logistic regression,
- certain output layers.

---

### 12.5.8 Limitations of Sigmoid

Despite its historical importance,

sigmoid has significant disadvantages.

When the input becomes very large or very small,

the output changes very little.

The curve becomes almost flat.

```text
Large Negative Input

↓

Output ≈ 0
```

```text
Large Positive Input

↓

Output ≈ 1
```

In these flat regions,

learning becomes extremely slow.

This phenomenon contributes to the

**vanishing gradient problem**,

which we will study later when discussing backpropagation.

Because of this limitation,

sigmoid is rarely used in modern hidden layers.

---

### 12.5.9 Hyperbolic Tangent (Tanh)

The **tanh** activation function is closely related to sigmoid.

Instead of producing outputs between

```text
0

and

1
```

it produces outputs between

```text
-1

and

1
```

Graphically,

```text
Output

 1 |               ______

   |            __/

 0 |---------__/

   |      _/

-1 |_____/

   +---------------------------->

             Input
```

Unlike sigmoid,

tanh produces both positive and negative outputs.

This often improves learning because the data remain centered around zero.

---

### 12.5.10 Advantages of Tanh

Compared with sigmoid,

tanh offers

- zero-centered output,
- smoother optimization,
- improved convergence,
- stronger gradients near zero.

For many years,

tanh was preferred over sigmoid for hidden layers.

However,

it still suffers from the vanishing gradient problem for very large positive or negative inputs.

---

### 12.5.11 Rectified Linear Unit (ReLU)

The most influential activation function in modern deep learning is the

**Rectified Linear Unit (ReLU).**

ReLU follows a remarkably simple rule.

```text
Negative Input

↓

Output = 0
```

```text
Positive Input

↓

Output = Input
```

Graphically,

```text
Output

^

|

|          /

|        /

|      /

|_____/____________________>

      0
```

This simple behavior revolutionized deep learning.

Today,

ReLU is the default activation function in many neural network architectures.

---

### 12.5.12 Why ReLU Became So Popular

ReLU has several important advantages.

- Very simple computation
- Fast training
- Efficient gradient propagation
- Reduces the vanishing gradient problem
- Works well for deep neural networks

Because of these advantages,

ReLU became the standard activation function for

- computer vision,
- natural language processing,
- speech recognition,
- and many materials science applications.

Most modern deep learning libraries,

including PyTorch,

use ReLU extensively.

---

### 12.5.13 The Dying ReLU Problem

Although ReLU performs extremely well,

it has one weakness.

Suppose a neuron receives only negative inputs.

```text
Input

↓

Negative

↓

Output = 0
```

If this situation continues during training,

the neuron may stop learning completely.

This phenomenon is known as the

**dying ReLU problem**.

Several newer activation functions were developed to address this limitation.

---

### 12.5.14 Leaky ReLU

Leaky ReLU modifies the behavior of ReLU for negative inputs.

Instead of producing exactly zero,

it allows a small negative output.

Conceptually,

```text
Negative Input

↓

Small Negative Output
```

```text
Positive Input

↓

Normal Positive Output
```

Graphically,

```text
Output

^

|

|          /

|        /

|      /

|_____/__________________>

    /

  /
```

Because neurons continue producing small outputs,

they remain capable of learning.

Leaky ReLU is commonly used when dying ReLU becomes problematic.

---

### 12.5.15 GELU

Another modern activation function is

**Gaussian Error Linear Unit (GELU).**

Unlike ReLU,

which abruptly switches between zero and the input,

GELU changes smoothly.

Conceptually,

```text
Small Inputs

↓

Partially Activated
```

```text
Large Inputs

↓

Fully Activated
```

GELU has become increasingly popular in

- transformer models,
- foundation models,
- and advanced deep learning architectures.

Several modern materials AI models also employ GELU.

---

### 12.5.16 Softmax Activation

The **Softmax** activation function is primarily used for

**multi-class classification**.

Suppose a model must classify a material as

- Metal
- Semiconductor
- Insulator

The output layer may produce

```text
Metal

↓

0.15
```

```text
Semiconductor

↓

0.80
```

```text
Insulator

↓

0.05
```

These values represent probabilities.

The predicted class is simply the one with the highest probability.

Softmax is therefore commonly used in the output layer of classification networks.

---

### 12.5.17 Choosing an Activation Function

Different activation functions are appropriate for different situations.

| Activation | Typical Use |
|------------|-------------|
| Linear | Regression output layer |
| Sigmoid | Binary classification output |
| Tanh | Some hidden layers |
| ReLU | Most hidden layers |
| Leaky ReLU | Hidden layers when dying ReLU is a concern |
| GELU | Modern deep learning architectures |
| Softmax | Multi-class classification output |

No activation function is universally best.

The appropriate choice depends on

- the task,
- the network architecture,
- and the training behavior.

---

### 12.5.18 Activation Functions in Materials Science

Most neural networks for materials property prediction use

```text
Hidden Layers

↓

ReLU
```

or

```text
Hidden Layers

↓

GELU
```

If the task is predicting a continuous property such as

- formation energy,
- band gap,
- bulk modulus,

the output layer usually employs a

**linear activation**.

If the task involves classifying materials,

such as

- metal,
- semiconductor,
- insulator,

the output layer often uses

**Softmax**.

Choosing suitable activation functions is therefore an important aspect of neural network design.

---

### 12.5.19 Summary

Activation functions transform the output of a neuron into a nonlinear representation, allowing neural networks to learn the complex relationships that characterize real-world materials data.

Without activation functions,

even extremely deep neural networks would reduce to simple linear models with very limited predictive power.

Among the many activation functions developed over the years,

ReLU has become the standard choice for hidden layers because of its simplicity and excellent training behavior, while Linear, Sigmoid, and Softmax remain important for specific output-layer tasks.

With activation functions now understood, we are ready to examine how information actually moves through an entire neural network. In the next section, we will study **forward propagation**, where inputs travel layer by layer through the network to produce a prediction.

## 12.6 Forward Propagation

So far, we have learned about

- artificial neurons,
- neural network architecture,
- and activation functions.

We now have all the components required to understand how a neural network actually makes a prediction.

The process by which information moves through a neural network from the input layer to the output layer is called **forward propagation** or **forward pass**.

Every prediction made by a neural network, whether it is predicting

- band gap,
- formation energy,
- elastic modulus,
- battery voltage,
- or any other material property,

begins with forward propagation.

It is the computational process that transforms raw input features into a final prediction.

---

### 12.6.1 What Is Forward Propagation?

Forward propagation is the sequential movement of information through a neural network.

The process always follows the same direction.

```text
Input Layer

↓

Hidden Layer 1

↓

Hidden Layer 2

↓

...

↓

Output Layer

↓

Prediction
```

Notice that information moves only **forward**.

No weights are updated during this stage.

The network simply performs calculations using its current weights and biases.

---

### 12.6.2 Why Is It Called "Propagation"?

The word

**propagation**

means

> passing information from one location to another.

In a neural network,

the output produced by one layer becomes the input to the next layer.

For example,

```text
Layer 1 Output

↓

Layer 2 Input

↓

Layer 2 Output

↓

Layer 3 Input
```

This process continues until the final prediction is produced.

Each layer transforms the information into a more useful internal representation.

---

### 12.6.3 Overall Workflow

The complete forward propagation process can be summarized as

```text
Input Features

↓

Weighted Sum

↓

Bias

↓

Activation Function

↓

Hidden Representation

↓

Repeat for Every Hidden Layer

↓

Output Layer

↓

Prediction
```

Every hidden layer performs the same sequence of operations.

Only the learned weights and biases differ.

---

### 12.6.4 Step 1 — Receive the Input Features

The process begins with the input layer.

Suppose we are predicting the formation energy of a material using five descriptors.

```text
Average Atomic Number

Density

Average Electronegativity

Packing Fraction

Average Atomic Radius
```

The input layer simply stores these numerical values.

No learning occurs here.

The input layer acts as a gateway between the dataset and the neural network.

---

### 12.6.5 Step 2 — Compute the Weighted Sum

Each neuron in the first hidden layer receives every input feature.

The neuron multiplies each input by its corresponding weight.

Conceptually,

```text
Input

↓

Multiply by Weight

↓

Add Together

↓

Weighted Sum
```

Using simple notation,

```text
Weighted Sum

=

(w1 × x1)

+

(w2 × x2)

+

...

+

(wn × xn)

+

Bias
```

This quantity represents the neuron's raw output before activation.

---

### 12.6.6 Step 3 — Apply the Activation Function

The weighted sum is then transformed using an activation function.

```text
Weighted Sum

↓

Activation Function

↓

Neuron Output
```

For example,

if ReLU is used,

```text
Negative Value

↓

0
```

```text
Positive Value

↓

Remain Positive
```

This introduces nonlinearity into the network.

Without this step,

deep learning would not be possible.

---

### 12.6.7 Step 4 — Pass Information to the Next Layer

The outputs of the first hidden layer become the inputs for the second hidden layer.

```text
Hidden Layer 1

↓

Hidden Layer 2
```

Exactly the same sequence is repeated.

```text
Inputs

↓

Weighted Sum

↓

Bias

↓

Activation

↓

Outputs
```

Each layer learns progressively more informative representations.

---

### 12.6.8 Step 5 — Produce the Final Prediction

Eventually,

the information reaches the output layer.

Suppose the network predicts band gap.

```text
Input Features

↓

Hidden Layers

↓

Output Layer

↓

2.18 eV
```

The value

```text
2.18 eV
```

is the network's prediction.

At this stage,

the prediction may or may not be correct.

The network has not yet learned anything.

It has only performed calculations using its current weights.

---

### 12.6.9 A Complete Forward Pass

The complete process can be visualized as

```text
Input Features

      │

      ▼

Input Layer

      │

      ▼

Hidden Layer 1

      │

      ▼

Activation

      │

      ▼

Hidden Layer 2

      │

      ▼

Activation

      │

      ▼

Output Layer

      │

      ▼

Predicted Property
```

Every prediction follows this sequence.

---

### 12.6.10 A Numerical Example

Suppose a neuron receives three input features.

```text
x1 = 2

x2 = 4

x3 = 1
```

Assume the corresponding weights are

```text
w1 = 0.5

w2 = 0.2

w3 = 0.8
```

and the bias is

```text
Bias = 0.3
```

The weighted sum becomes

```text
(0.5 × 2)

+

(0.2 × 4)

+

(0.8 × 1)

+

0.3
```

Computing each term,

```text
1.0

+

0.8

+

0.8

+

0.3

=

2.9
```

If ReLU is used,

the output becomes

```text
ReLU(2.9)

↓

2.9
```

because the value is positive.

This output is then passed to the next layer.

---

### 12.6.11 Information Becomes More Abstract

One of the most remarkable aspects of forward propagation is that the information gradually changes.

The input features may represent simple physical quantities.

```text
Density

Electronegativity

Atomic Radius

Packing Fraction
```

The first hidden layer combines these descriptors.

The second hidden layer combines those combinations.

The third hidden layer combines even more complex representations.

Conceptually,

```text
Raw Descriptors

↓

Simple Patterns

↓

Intermediate Patterns

↓

Complex Representations

↓

Prediction
```

The network automatically discovers these representations during training.

---

### 12.6.12 Forward Propagation in Matrix Form

Modern neural networks rarely process one neuron at a time.

Instead,

entire layers are computed simultaneously using matrix operations.

Conceptually,

```text
Input Matrix

↓

Weight Matrix

↓

Bias Vector

↓

Activation Function

↓

Output Matrix
```

This allows thousands of neurons to be evaluated efficiently using optimized numerical libraries.

One of the reasons GPUs are so effective for deep learning is that they perform these matrix operations extremely quickly.

---

### 12.6.13 Forward Propagation for Multiple Samples

During training,

the network usually processes many materials simultaneously.

Instead of

```text
One Material

↓

Prediction
```

the workflow becomes

```text
Batch of Materials

↓

Neural Network

↓

Batch of Predictions
```

For example,

```text
32 Materials

↓

Forward Propagation

↓

32 Predicted Band Gaps
```

Processing multiple samples together improves computational efficiency and leads to more stable training.

We will discuss batches in detail later in this chapter.

---

### 12.6.14 Forward Propagation During Inference

Once a neural network has been trained,

forward propagation is the only computation required.

The workflow becomes

```text
New Material

↓

Generate Features

↓

Forward Propagation

↓

Predicted Property
```

No weight updates occur during inference.

The network simply applies its learned knowledge to previously unseen materials.

---

### 12.6.15 Forward Propagation in Materials Science

Consider a neural network trained to predict formation energy.

The complete prediction pipeline may look like

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

Feature Vector

↓

Neural Network

↓

Forward Propagation

↓

Predicted Formation Energy
```

Every new material passes through exactly the same sequence of calculations.

Only the input descriptors change.

The network architecture and learned parameters remain fixed.

---

### 12.6.16 Forward Propagation Versus Training

It is important to distinguish between

**prediction**

and

**learning**.

Forward propagation performs prediction.

```text
Input

↓

Prediction
```

Learning requires an additional stage.

```text
Prediction

↓

Calculate Error

↓

Update Weights

↓

Better Prediction
```

The process of updating the weights is called

**backpropagation**,

which will be studied later in this chapter.

Forward propagation answers the question

> "What does the network predict?"

Backpropagation answers the question

> "How should the network change its weights to improve future predictions?"

Together,

these two processes form the foundation of neural network training.

---

### 12.6.17 Summary

Forward propagation is the process by which information flows through a neural network to produce a prediction.

Starting with the input features, each layer computes weighted sums, adds biases, applies activation functions, and passes the resulting outputs to the next layer until the final prediction is generated.

During forward propagation,

- no learning occurs,
- no weights are updated,
- and no optimization is performed.

The network simply uses its current parameters to make predictions.

Once a prediction has been produced, the next step is to determine **how good that prediction is**. To do this, we need a quantitative measure of prediction error.

In the next section, we will introduce **loss functions**, which measure the difference between the predicted values and the true material properties and provide the objective that the neural network seeks to minimize during training.

## 12.7 Loss Functions

In the previous section, we learned how a neural network performs **forward propagation** to generate predictions.

However, making a prediction is only the first step.

The next question is far more important.

> **How does the neural network know whether its prediction is good or bad?**

Consider the following example.

Suppose the actual band gap of a material is

```text
Actual Band Gap = 2.50 eV
```

After forward propagation, the neural network predicts

```text
Predicted Band Gap = 2.31 eV
```

Is this prediction acceptable?

What if another model predicts

```text
2.48 eV
```

Which model is better?

To answer these questions, we need a quantitative measure of prediction error.

This measure is called the **loss function**.

The loss function is one of the most important concepts in deep learning because it tells the neural network **how wrong its predictions are**.

Training a neural network is essentially the process of minimizing the loss function.

---

### 12.7.1 What Is a Loss Function?

A **loss function** is a mathematical function that measures the difference between

- the predicted value,
- and the true value.

Conceptually,

```text
Prediction

↓

Compare with True Value

↓

Compute Error

↓

Loss
```

A small loss indicates that the prediction is close to the correct answer.

A large loss indicates that the prediction is poor.

Therefore,

```text
Small Loss

↓

Good Model
```

```text
Large Loss

↓

Poor Model
```

The objective of neural network training is

```text
Minimize Loss
```

---

### 12.7.2 Why Is the Loss Function Necessary?

Imagine trying to improve your performance on an exam without knowing your score.

You would have no idea whether your study methods were working.

Similarly,

a neural network cannot improve unless it receives feedback about the quality of its predictions.

The loss function provides this feedback.

The overall training process becomes

```text
Input Features

↓

Forward Propagation

↓

Prediction

↓

Loss Function

↓

Measure Error

↓

Update Weights

↓

Better Prediction
```

Without a loss function,

the neural network would have no direction for learning.

---

### 12.7.3 Loss Versus Error

Although the words

**error**

and

**loss**

are often used interchangeably,

they are not exactly the same.

An **error** refers to the difference between

```text
Prediction

and

Actual Value
```

A **loss function** converts that error into a numerical quantity that the optimization algorithm can minimize.

For example,

```text
Actual Band Gap

↓

2.50 eV
```

```text
Predicted Band Gap

↓

2.20 eV
```

The prediction error is

```text
0.30 eV
```

The loss function determines how this error contributes to the overall training objective.

---

### 12.7.4 Regression and Classification Have Different Loss Functions

The appropriate loss function depends on the type of machine learning problem.

For **regression** tasks,

examples include

- band gap prediction,
- formation energy prediction,
- density prediction,
- elastic modulus prediction.

The output is a continuous numerical value.

Common regression loss functions include

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss

For **classification** tasks,

examples include

- metal vs semiconductor,
- crystal system prediction,
- phase classification.

The output represents categories.

Common classification loss functions include

- Binary Cross Entropy
- Categorical Cross Entropy

We begin with regression because it is the most common problem in materials informatics.

---

### 12.7.5 Mean Squared Error (MSE)

The most widely used regression loss function is

**Mean Squared Error (MSE).**

The computation follows three simple steps.

```text
Prediction Error

↓

Square the Error

↓

Average Over All Samples
```

The mathematical expression is

```text
MSE = Average[(Prediction − Actual)^2]
```

Notice the square.

Squaring ensures that

- positive and negative errors do not cancel,
- and larger errors receive much greater penalties.

---

### 12.7.6 Example of Mean Squared Error

Suppose we have predictions for three materials.

| Material | Actual (eV) | Predicted (eV) |
|----------|-------------:|---------------:|
| A | 2.00 | 1.80 |
| B | 3.50 | 3.70 |
| C | 1.20 | 1.00 |

First,

compute the prediction errors.

| Material | Error |
|----------|------:|
| A | -0.20 |
| B | 0.20 |
| C | -0.20 |

Next,

square each error.

| Material | Squared Error |
|----------|--------------:|
| A | 0.04 |
| B | 0.04 |
| C | 0.04 |

Finally,

take the average.

```text
(0.04 + 0.04 + 0.04)

÷

3

=

0.04
```

Therefore,

```text
MSE = 0.04
```

---

### 12.7.7 Why Square the Errors?

Suppose we have two prediction errors.

```text
0.2 eV
```

and

```text
2.0 eV
```

If we simply average the errors,

large mistakes receive relatively little additional penalty.

After squaring,

```text
0.2

↓

0.04
```

```text
2.0

↓

4.00
```

The larger error now contributes much more strongly to the loss.

Consequently,

MSE encourages the neural network to eliminate large prediction errors.

---

### 12.7.8 Mean Absolute Error (MAE)

Another common regression loss is

**Mean Absolute Error (MAE).**

Instead of squaring the errors,

MAE uses their absolute values.

The mathematical expression is

```text
MAE = Average[ |Prediction − Actual| ]
```

Conceptually,

```text
Prediction Error

↓

Take Absolute Value

↓

Average
```

Unlike MSE,

MAE treats all errors proportionally.

---

### 12.7.9 Example of Mean Absolute Error

Using the previous example,

the errors are

```text
0.20

0.20

0.20
```

The average becomes

```text
(0.20 + 0.20 + 0.20)

÷

3

=

0.20
```

Therefore,

```text
MAE = 0.20
```

Unlike MSE,

large errors are not emphasized as strongly.

---

### 12.7.10 MSE Versus MAE

Both loss functions are widely used.

However,

they behave differently.

| Mean Squared Error (MSE) | Mean Absolute Error (MAE) |
|---------------------------|---------------------------|
| Squares errors | Uses absolute values |
| Penalizes large errors strongly | Penalizes all errors equally |
| Smooth optimization | More robust to outliers |
| Common for neural network training | Often used as an evaluation metric |

In many deep learning applications,

training is performed using MSE,

while performance is reported using MAE or RMSE.

---

### 12.7.11 Root Mean Squared Error (RMSE)

Although RMSE is commonly reported,

it is usually considered an **evaluation metric** rather than a training loss.

RMSE is obtained by taking the square root of MSE.

```text
RMSE = Square Root of MSE
```

The advantage is that RMSE has the same physical units as the target property.

For example,

if band gap is measured in

```text
eV
```

then

```text
RMSE

↓

eV
```

This makes RMSE easier to interpret scientifically.

---

### 12.7.12 Huber Loss

Huber Loss combines the advantages of both

- MSE
- MAE

For small errors,

it behaves similarly to MSE.

For very large errors,

it behaves more like MAE.

Conceptually,

```text
Small Error

↓

MSE Behavior
```

```text
Large Error

↓

MAE Behavior
```

Huber Loss is useful when datasets contain outliers that should not dominate training.

---

### 12.7.13 Loss Functions for Classification

Regression predicts numerical values.

Classification predicts categories.

Suppose a neural network classifies materials as

- Metal
- Semiconductor
- Insulator

The output might be

| Class | Probability |
|--------|------------:|
| Metal | 0.12 |
| Semiconductor | 0.81 |
| Insulator | 0.07 |

The network predicts

```text
Semiconductor
```

Classification problems use specialized loss functions,

primarily

- Binary Cross Entropy
- Cross Entropy Loss

These compare predicted probabilities with the correct class labels.

We will revisit these loss functions when studying classification networks.

---

### 12.7.14 Loss During Training

Initially,

the neural network begins with random weights.

Consequently,

its predictions are often poor.

```text
Epoch 1

↓

Large Loss
```

As training continues,

the weights improve.

```text
Epoch 20

↓

Smaller Loss
```

Eventually,

the loss reaches a relatively stable value.

```text
Epoch 100

↓

Very Small Loss
```

A successful training process therefore resembles

```text
Loss

^

|

|\
| \
|  \
|   \
|    \_____

+---------------------------->

          Epoch
```

A steadily decreasing loss generally indicates that the network is learning useful patterns.

---

### 12.7.15 Training Loss and Validation Loss

In practice,

we monitor two different losses.

```text
Training Dataset

↓

Training Loss
```

```text
Validation Dataset

↓

Validation Loss
```

Training loss measures how well the model fits the data used for learning.

Validation loss measures how well the model generalizes to unseen data.

If

```text
Training Loss

↓

Very Small
```

but

```text
Validation Loss

↓

Large
```

the model is likely **overfitting**.

We will study overfitting and regularization later in this chapter.

---

### 12.7.16 Loss Functions in Materials Science

Most neural network models for materials property prediction use

```text
Regression

↓

MSE Loss
```

or

```text
Regression

↓

MAE Loss
```

Examples include predicting

- band gaps,
- formation energies,
- elastic constants,
- bulk moduli,
- thermal conductivities,
- dielectric constants.

Classification problems,

such as identifying crystal phases or classifying materials,

typically employ cross-entropy-based losses.

Choosing an appropriate loss function is therefore closely tied to the scientific objective.

---

### 12.7.17 Summary

A loss function provides the numerical feedback that allows a neural network to learn.

After forward propagation generates a prediction, the loss function compares that prediction with the true value and quantifies the prediction error.

For regression problems, the most common loss functions are MSE, MAE, and Huber Loss, while classification problems typically use cross-entropy-based losses.

The ultimate objective of neural network training is simple:

```text
Find the weights that produce the smallest possible loss.
```

However,

knowing the loss alone is not enough.

The network must also determine **how to change its weights** in order to reduce that loss.

The optimization algorithm responsible for this process begins with one of the most fundamental ideas in machine learning:

**gradient descent**.

In the next section, we will study how gradient descent systematically adjusts neural network parameters to minimize the loss function.

## 12.8 Gradient Descent

In the previous section, we introduced the concept of a **loss function**, which measures how far the neural network's predictions are from the correct values.

However, calculating the loss is only the beginning.

Once the neural network knows that its predictions are incorrect, an important question arises.

> **How should the network change its weights to reduce the loss?**

Suppose a neural network predicts the band gap of a material.

```text
Actual Band Gap

↓

2.50 eV
```

```text
Predicted Band Gap

↓

1.85 eV
```

The prediction is clearly inaccurate.

Should the weights

- increase,
- decrease,
- or remain unchanged?

How much should they change?

Gradient descent is the optimization algorithm that answers these questions.

It is the fundamental algorithm used to train almost every modern neural network.

---

### 12.8.1 What Is Gradient Descent?

Gradient descent is an optimization algorithm that finds the set of weights and biases that minimize the loss function.

Conceptually,

```text
Initial Weights

↓

Prediction

↓

Loss

↓

Adjust Weights

↓

New Prediction

↓

Smaller Loss

↓

Repeat
```

This cycle continues until the loss can no longer be significantly reduced.

The ultimate objective is

```text
Minimum Loss
```

---

### 12.8.2 The Mountain Analogy

One of the best ways to understand gradient descent is through a simple analogy.

Imagine standing somewhere on a mountain during heavy fog.

You cannot see the valley below.

Your objective is to reach the lowest point.

How would you proceed?

Most people would

- look around,
- determine which direction slopes downward,
- take a small step,
- and repeat the process.

Eventually,

they would arrive near the bottom of the valley.

Gradient descent follows exactly the same strategy.

Instead of descending a mountain,

it descends the **loss surface**.

```text
Large Loss

      ▲

     / \

    /   \

   /     \

__/       \____

             ▼

        Minimum Loss
```

Each optimization step moves the neural network toward lower loss.

---

### 12.8.3 The Loss Landscape

A neural network may contain

- hundreds,
- thousands,
- or even millions

of trainable parameters.

Each possible combination of weights corresponds to a different loss.

If we imagine plotting all possible losses,

we obtain a complex surface called the **loss landscape**.

Conceptually,

```text
Weights

↓

Loss Surface

↓

Lowest Point

↓

Best Parameters
```

The goal of optimization is to locate this lowest region.

---

### 12.8.4 Why Random Weight Updates Do Not Work

Suppose we updated weights randomly.

```text
Random Weight

↓

Prediction

↓

Random Weight

↓

Prediction

↓

Random Weight

↓

Prediction
```

Sometimes the prediction would improve.

Sometimes it would become worse.

Overall,

training would be extremely inefficient.

Gradient descent avoids random guessing by using information from the loss function to determine the best direction for updating the weights.

---

### 12.8.5 What Is a Gradient?

A **gradient** measures how rapidly the loss changes when a parameter changes.

Conceptually,

it answers the question

> **If this weight changes slightly, how much will the loss change?**

For every trainable parameter,

the gradient provides two pieces of information.

- Which direction reduces the loss?
- How rapidly is the loss changing?

This information allows the optimizer to update the weights intelligently rather than randomly.

---

### 12.8.6 Understanding the Direction of the Gradient

Consider a simple hill.

```text
        ▲

      /   \

    /       \

__/           \__
```

If you stand on the left side,

the downhill direction points to the left.

If you stand on the right side,

the downhill direction points to the right.

The gradient identifies the uphill direction.

Gradient descent simply moves in the opposite direction.

```text
Gradient

↓

Steepest Increase
```

```text
Gradient Descent

↓

Steepest Decrease
```

---

### 12.8.7 Weight Update Concept

Each training iteration follows the same idea.

```text
Current Weight

↓

Compute Gradient

↓

Move Opposite Gradient

↓

Updated Weight
```

This process is repeated for every trainable parameter in the neural network.

Over many iterations,

the weights gradually approach values that minimize the loss.

---

### 12.8.8 The Learning Rate

Knowing the direction of the update is not enough.

We must also decide

> **How large should each update be?**

This is determined by the **learning rate**.

The learning rate controls the size of every optimization step.

```text
Gradient

↓

Learning Rate

↓

Weight Update
```

The learning rate is one of the most important hyperparameters in deep learning.

---

### 12.8.9 Large Learning Rate

Suppose the learning rate is very large.

The optimizer takes huge steps.

```text
Current Position

↓

Huge Step

↓

Past Minimum

↓

Huge Step

↓

Past Minimum Again
```

Instead of converging,

the optimizer repeatedly jumps over the minimum.

Conceptually,

```text
Loss

^

|

|      \      /

|       \    /

|        \  /

|         \/

|         /\

+------------------------>

      Training
```

The loss oscillates rather than decreasing smoothly.

Training may even diverge completely.

---

### 12.8.10 Small Learning Rate

Now suppose the learning rate is extremely small.

```text
Current Position

↓

Tiny Step

↓

Tiny Step

↓

Tiny Step
```

Eventually,

the optimizer reaches the minimum,

but training becomes extremely slow.

Conceptually,

```text
Loss

^

|

|\

| \

|  \

|   \

|    \

+------------------------>

     Very Long Training
```

Although stable,

this approach may require an impractically large number of training iterations.

---

### 12.8.11 Choosing a Good Learning Rate

An effective learning rate balances

- stability,
- convergence speed,
- and optimization accuracy.

Generally,

```text
Too Small

↓

Slow Learning
```

```text
Too Large

↓

Unstable Learning
```

```text
Appropriate Value

↓

Fast and Stable Learning
```

Selecting a suitable learning rate is therefore one of the most important aspects of training a neural network.

---

### 12.8.12 One Complete Gradient Descent Cycle

A complete optimization cycle consists of the following steps.

```text
Input Features

↓

Forward Propagation

↓

Prediction

↓

Loss Function

↓

Compute Gradient

↓

Update Weights

↓

Repeat
```

Notice that the prediction changes after every weight update.

Ideally,

each update produces a smaller loss than the previous iteration.

---

### 12.8.13 Local and Global Minima

The loss landscape of a deep neural network is highly complex.

Instead of one valley,

there may be many valleys.

Conceptually,

```text
Loss

^

|     /\

|    /  \__

|___/       \__

|              \____

+---------------------------->

        Parameters
```

The deepest valley represents the **global minimum**.

Smaller valleys represent **local minima**.

In practice,

modern neural networks often perform well even if optimization converges to a local minimum rather than the absolute global minimum.

---

### 12.8.14 Batch Gradient Descent

The simplest version of gradient descent computes the loss using the **entire training dataset** before updating the weights.

Workflow

```text
Entire Dataset

↓

Forward Propagation

↓

Compute Loss

↓

Update Weights
```

Advantages

- Stable optimization
- Accurate gradient estimates

Disadvantages

- Computationally expensive
- Slow for very large datasets

Modern materials databases often contain hundreds of thousands of structures, making this approach impractical.

---

### 12.8.15 Stochastic Gradient Descent (SGD)

Instead of processing the entire dataset,

Stochastic Gradient Descent updates the weights after processing **one training example**.

```text
One Material

↓

Prediction

↓

Loss

↓

Weight Update
```

Advantages

- Very fast updates
- Low memory usage

Disadvantages

- Noisy optimization
- Less stable convergence

Despite its noise,

SGD remains one of the most influential optimization methods in machine learning.

---

### 12.8.16 Mini-Batch Gradient Descent

Most modern neural networks use **mini-batch gradient descent**.

Instead of using

- one sample,
- or the entire dataset,

the optimizer processes a small group of samples together.

For example,

```text
32 Materials

↓

Forward Propagation

↓

Loss

↓

Weight Update
```

or

```text
64 Materials

↓

Forward Propagation

↓

Loss

↓

Weight Update
```

Mini-batch gradient descent combines the advantages of both previous methods.

It

- trains efficiently,
- provides stable gradients,
- utilizes GPUs effectively,
- and has become the standard approach in deep learning.

---

### 12.8.17 Gradient Descent in Materials Science

Suppose we are training a neural network to predict formation energies.

The optimization workflow becomes

```text
Crystal Structures

↓

Descriptors

↓

Neural Network

↓

Predicted Formation Energy

↓

Loss Function

↓

Gradient Descent

↓

Updated Weights
```

After thousands of optimization steps,

the neural network gradually learns the relationship between material descriptors and formation energy.

The same optimization strategy is used for predicting

- band gaps,
- elastic constants,
- dielectric properties,
- battery voltages,
- thermal conductivities,
- and many other materials properties.

---

### 12.8.18 Limitations of Basic Gradient Descent

Although gradient descent is conceptually simple,

basic gradient descent has several limitations.

- Choosing a suitable learning rate can be difficult.
- Training may become slow.
- Optimization can stall in flat regions.
- Convergence may oscillate.
- Large datasets make full-batch training impractical.

To overcome these challenges,

more advanced optimization algorithms have been developed,

including

- Momentum,
- RMSProp,
- Adam,
- AdamW.

These optimizers are widely used in modern deep learning frameworks such as PyTorch.

---

### 12.8.19 Summary

Gradient descent is the optimization algorithm that enables neural networks to learn.

Rather than randomly changing weights, it uses information from the gradient of the loss function to determine both the direction and magnitude of each update.

By repeatedly performing

```text
Prediction

↓

Loss

↓

Gradient

↓

Weight Update
```

the neural network gradually reduces its prediction error and learns increasingly accurate representations of the data.

However, one important question remains unanswered.

How does the neural network actually compute the gradients needed for these updates?

The answer lies in one of the most important algorithms in deep learning:

**backpropagation**.

In the next section, we will study backpropagation in detail and see how neural networks efficiently compute gradients for millions of trainable parameters.

## 12.8 Gradient Descent

In the previous section, we introduced the concept of a **loss function**, which measures how far the neural network's predictions are from the correct values.

However, calculating the loss is only the beginning.

Once the neural network knows that its predictions are incorrect, an important question arises.

> **How should the network change its weights to reduce the loss?**

Suppose a neural network predicts the band gap of a material.

```text
Actual Band Gap

↓

2.50 eV
```

```text
Predicted Band Gap

↓

1.85 eV
```

The prediction is clearly inaccurate.

Should the weights

- increase,
- decrease,
- or remain unchanged?

How much should they change?

Gradient descent is the optimization algorithm that answers these questions.

It is the fundamental algorithm used to train almost every modern neural network.

---

### 12.8.1 What Is Gradient Descent?

Gradient descent is an optimization algorithm that finds the set of weights and biases that minimize the loss function.

Conceptually,

```text
Initial Weights

↓

Prediction

↓

Loss

↓

Adjust Weights

↓

New Prediction

↓

Smaller Loss

↓

Repeat
```

This cycle continues until the loss can no longer be significantly reduced.

The ultimate objective is

```text
Minimum Loss
```

---

### 12.8.2 The Mountain Analogy

One of the best ways to understand gradient descent is through a simple analogy.

Imagine standing somewhere on a mountain during heavy fog.

You cannot see the valley below.

Your objective is to reach the lowest point.

How would you proceed?

Most people would

- look around,
- determine which direction slopes downward,
- take a small step,
- and repeat the process.

Eventually,

they would arrive near the bottom of the valley.

Gradient descent follows exactly the same strategy.

Instead of descending a mountain,

it descends the **loss surface**.

```text
Large Loss

      ▲

     / \

    /   \

   /     \

__/       \____

             ▼

        Minimum Loss
```

Each optimization step moves the neural network toward lower loss.

---

### 12.8.3 The Loss Landscape

A neural network may contain

- hundreds,
- thousands,
- or even millions

of trainable parameters.

Each possible combination of weights corresponds to a different loss.

If we imagine plotting all possible losses,

we obtain a complex surface called the **loss landscape**.

Conceptually,

```text
Weights

↓

Loss Surface

↓

Lowest Point

↓

Best Parameters
```

The goal of optimization is to locate this lowest region.

---

### 12.8.4 Why Random Weight Updates Do Not Work

Suppose we updated weights randomly.

```text
Random Weight

↓

Prediction

↓

Random Weight

↓

Prediction

↓

Random Weight

↓

Prediction
```

Sometimes the prediction would improve.

Sometimes it would become worse.

Overall,

training would be extremely inefficient.

Gradient descent avoids random guessing by using information from the loss function to determine the best direction for updating the weights.

---

### 12.8.5 What Is a Gradient?

A **gradient** measures how rapidly the loss changes when a parameter changes.

Conceptually,

it answers the question

> **If this weight changes slightly, how much will the loss change?**

For every trainable parameter,

the gradient provides two pieces of information.

- Which direction reduces the loss?
- How rapidly is the loss changing?

This information allows the optimizer to update the weights intelligently rather than randomly.

---

### 12.8.6 Understanding the Direction of the Gradient

Consider a simple hill.

```text
        ▲

      /   \

    /       \

__/           \__
```

If you stand on the left side,

the downhill direction points to the left.

If you stand on the right side,

the downhill direction points to the right.

The gradient identifies the uphill direction.

Gradient descent simply moves in the opposite direction.

```text
Gradient

↓

Steepest Increase
```

```text
Gradient Descent

↓

Steepest Decrease
```

---

### 12.8.7 Weight Update Concept

Each training iteration follows the same idea.

```text
Current Weight

↓

Compute Gradient

↓

Move Opposite Gradient

↓

Updated Weight
```

This process is repeated for every trainable parameter in the neural network.

Over many iterations,

the weights gradually approach values that minimize the loss.

---

### 12.8.8 The Learning Rate

Knowing the direction of the update is not enough.

We must also decide

> **How large should each update be?**

This is determined by the **learning rate**.

The learning rate controls the size of every optimization step.

```text
Gradient

↓

Learning Rate

↓

Weight Update
```

The learning rate is one of the most important hyperparameters in deep learning.

---

### 12.8.9 Large Learning Rate

Suppose the learning rate is very large.

The optimizer takes huge steps.

```text
Current Position

↓

Huge Step

↓

Past Minimum

↓

Huge Step

↓

Past Minimum Again
```

Instead of converging,

the optimizer repeatedly jumps over the minimum.

Conceptually,

```text
Loss

^

|

|      \      /

|       \    /

|        \  /

|         \/

|         /\

+------------------------>

      Training
```

The loss oscillates rather than decreasing smoothly.

Training may even diverge completely.

---

### 12.8.10 Small Learning Rate

Now suppose the learning rate is extremely small.

```text
Current Position

↓

Tiny Step

↓

Tiny Step

↓

Tiny Step
```

Eventually,

the optimizer reaches the minimum,

but training becomes extremely slow.

Conceptually,

```text
Loss

^

|

|\

| \

|  \

|   \

|    \

+------------------------>

     Very Long Training
```

Although stable,

this approach may require an impractically large number of training iterations.

---

### 12.8.11 Choosing a Good Learning Rate

An effective learning rate balances

- stability,
- convergence speed,
- and optimization accuracy.

Generally,

```text
Too Small

↓

Slow Learning
```

```text
Too Large

↓

Unstable Learning
```

```text
Appropriate Value

↓

Fast and Stable Learning
```

Selecting a suitable learning rate is therefore one of the most important aspects of training a neural network.

---

### 12.8.12 One Complete Gradient Descent Cycle

A complete optimization cycle consists of the following steps.

```text
Input Features

↓

Forward Propagation

↓

Prediction

↓

Loss Function

↓

Compute Gradient

↓

Update Weights

↓

Repeat
```

Notice that the prediction changes after every weight update.

Ideally,

each update produces a smaller loss than the previous iteration.

---

### 12.8.13 Local and Global Minima

The loss landscape of a deep neural network is highly complex.

Instead of one valley,

there may be many valleys.

Conceptually,

```text
Loss

^

|     /\

|    /  \__

|___/       \__

|              \____

+---------------------------->

        Parameters
```

The deepest valley represents the **global minimum**.

Smaller valleys represent **local minima**.

In practice,

modern neural networks often perform well even if optimization converges to a local minimum rather than the absolute global minimum.

---

### 12.8.14 Batch Gradient Descent

The simplest version of gradient descent computes the loss using the **entire training dataset** before updating the weights.

Workflow

```text
Entire Dataset

↓

Forward Propagation

↓

Compute Loss

↓

Update Weights
```

Advantages

- Stable optimization
- Accurate gradient estimates

Disadvantages

- Computationally expensive
- Slow for very large datasets

Modern materials databases often contain hundreds of thousands of structures, making this approach impractical.

---

### 12.8.15 Stochastic Gradient Descent (SGD)

Instead of processing the entire dataset,

Stochastic Gradient Descent updates the weights after processing **one training example**.

```text
One Material

↓

Prediction

↓

Loss

↓

Weight Update
```

Advantages

- Very fast updates
- Low memory usage

Disadvantages

- Noisy optimization
- Less stable convergence

Despite its noise,

SGD remains one of the most influential optimization methods in machine learning.

---

### 12.8.16 Mini-Batch Gradient Descent

Most modern neural networks use **mini-batch gradient descent**.

Instead of using

- one sample,
- or the entire dataset,

the optimizer processes a small group of samples together.

For example,

```text
32 Materials

↓

Forward Propagation

↓

Loss

↓

Weight Update
```

or

```text
64 Materials

↓

Forward Propagation

↓

Loss

↓

Weight Update
```

Mini-batch gradient descent combines the advantages of both previous methods.

It

- trains efficiently,
- provides stable gradients,
- utilizes GPUs effectively,
- and has become the standard approach in deep learning.

---

### 12.8.17 Gradient Descent in Materials Science

Suppose we are training a neural network to predict formation energies.

The optimization workflow becomes

```text
Crystal Structures

↓

Descriptors

↓

Neural Network

↓

Predicted Formation Energy

↓

Loss Function

↓

Gradient Descent

↓

Updated Weights
```

After thousands of optimization steps,

the neural network gradually learns the relationship between material descriptors and formation energy.

The same optimization strategy is used for predicting

- band gaps,
- elastic constants,
- dielectric properties,
- battery voltages,
- thermal conductivities,
- and many other materials properties.

---

### 12.8.18 Limitations of Basic Gradient Descent

Although gradient descent is conceptually simple,

basic gradient descent has several limitations.

- Choosing a suitable learning rate can be difficult.
- Training may become slow.
- Optimization can stall in flat regions.
- Convergence may oscillate.
- Large datasets make full-batch training impractical.

To overcome these challenges,

more advanced optimization algorithms have been developed,

including

- Momentum,
- RMSProp,
- Adam,
- AdamW.

These optimizers are widely used in modern deep learning frameworks such as PyTorch.

---

### 12.8.19 Summary

Gradient descent is the optimization algorithm that enables neural networks to learn.

Rather than randomly changing weights, it uses information from the gradient of the loss function to determine both the direction and magnitude of each update.

By repeatedly performing

```text
Prediction

↓

Loss

↓

Gradient

↓

Weight Update
```

the neural network gradually reduces its prediction error and learns increasingly accurate representations of the data.

However, one important question remains unanswered.

How does the neural network actually compute the gradients needed for these updates?

The answer lies in one of the most important algorithms in deep learning:

**backpropagation**.

In the next section, we will study backpropagation in detail and see how neural networks efficiently compute gradients for millions of trainable parameters.

# 12.9 Backpropagation — How Neural Networks Learn

In the previous section, we introduced gradient descent.

We learned that gradient descent updates neural network parameters using information from the gradient of the loss function.

However, an important question remains:

> How does the neural network calculate the gradient of every weight?

A modern neural network may contain:

```text
Thousands

↓

Millions

↓

Billions

of parameters
```

Manually calculating how every single parameter affects the final error would be impossible.

The solution is:

```text
Backpropagation
```

Backpropagation is the algorithm that efficiently calculates gradients throughout a neural network.

It is the mechanism that allows deep learning models to learn from data.

---

# 12.9.1 The Basic Idea of Backpropagation

A neural network has two major computational directions.

## Forward Direction

Information moves from input to output.

```text
Material Descriptors

↓

Neural Network

↓

Prediction

↓

Loss
```

This calculates:

```text
How wrong is the model?
```

---

## Backward Direction

The error signal moves backward.

```text
Loss

↓

Output Layer

↓

Hidden Layers

↓

Input Connections
```

This calculates:

```text
How much did each parameter contribute to the error?
```

---

The complete training cycle is:

```text
Input Data

↓

Forward Propagation

↓

Prediction

↓

Loss Calculation

↓

Backpropagation

↓

Gradient Calculation

↓

Optimizer Update

↓

New Parameters
```

---

# 12.9.2 Why Backpropagation Is Necessary

Consider a simple neural network:

```text
Input

↓

Hidden Layer

↓

Output

```

Suppose the network predicts:

```text
Predicted Band Gap

↓

1.8 eV
```

but the actual value is:

```text
Actual Band Gap

↓

2.5 eV
```

The loss is large.

The network must answer:

```text
Which weight caused this error?

How much should each weight change?
```

Backpropagation provides this information.

---

# 12.9.3 A Simple Neural Network Example

Consider a single neuron:

```text
Input

x

↓

Weight

w

↓

Output

y
```

The prediction is:

```text
y = x × w
```

Suppose:

```text
Input descriptor:

x = 5
```

Weight:

```text
w = 0.4
```

Prediction:

```text
y = 5 × 0.4

y = 2
```

If the target is:

```text
Target = 3
```

then the error is:

```text
Prediction

↓

2


Target

↓

3
```

The weight must increase.

---

# 12.9.4 The Chain Rule

Backpropagation is based on a fundamental mathematical concept:

```text
Chain Rule
```

The chain rule tells us how a final output depends on intermediate variables.

A neural network is a chain of operations.

Example:

```text
Input

↓

Weight Multiplication

↓

Activation

↓

Output

↓

Loss
```

The final loss depends indirectly on every weight.

The chain rule allows us to trace this dependency backward.

---

# 12.9.5 Understanding Gradient Flow

Consider:

```text
Weight

↓

Neuron Output

↓

Prediction

↓

Loss
```

Backpropagation calculates:

```text
How much does this weight affect the loss?
```

Mathematically:

```text
Change in Loss

----------------

Change in Weight
```

This quantity is the gradient.

---

# 12.9.6 Forward Propagation Example in Python

Before calculating gradients,

we need a forward pass.

Using PyTorch:

```python
import torch


# Material descriptor input

x = torch.tensor(
    [5.0]
)


# Weight

w = torch.tensor(
    [0.4],
    requires_grad=True
)


# Forward calculation

prediction = x * w


print(
    prediction
)
```

Output:

```text
tensor([2.])
```

The neural network has produced a prediction.

---

# 12.9.7 Calculating Loss

Now compare prediction with the target.

Example:

```python
target = torch.tensor(
    [3.0]
)
```

Use mean squared error:

```python
loss_function = torch.nn.MSELoss()


loss = loss_function(
    prediction,
    target
)


print(loss)
```

The loss measures prediction error.

---

# 12.9.8 Computing Gradients Automatically

PyTorch can calculate gradients automatically.

Use:

```python
loss.backward()
```

Example:

```python
loss.backward()


print(
    w.grad
)
```

Output:

```text
tensor([-10.])
```

The gradient tells us:

```text
How much the weight contributed to the error
```

---

# 12.9.9 Updating the Weight Manually

Gradient descent updates parameters:

```text
New Weight

=

Old Weight

-

Learning Rate × Gradient
```

Example:

```python
learning_rate = 0.01


with torch.no_grad():

    w -= learning_rate * w.grad
```

Now the weight has moved in the direction that reduces the loss.

---

# 12.9.10 Complete Manual Training Loop

We can train this simple neuron.

```python
import torch


x = torch.tensor(
    [5.0]
)


target = torch.tensor(
    [3.0]
)


w = torch.tensor(
    [0.1],
    requires_grad=True
)


learning_rate = 0.01



for epoch in range(20):


    prediction = x * w


    loss = (
        prediction - target
    ) ** 2


    loss.backward()


    with torch.no_grad():

        w -= learning_rate * w.grad


    w.grad.zero_()


    print(
        epoch,
        loss.item(),
        w.item()
    )
```

The training process:

```text
Initial Weight

↓

Prediction

↓

Loss

↓

Gradient

↓

Weight Update

↓

Improved Prediction
```

---

# 12.9.11 What Happens Inside a Real Neural Network?

A real materials neural network contains many layers.

Example:

```text
200 Material Descriptors

↓

Linear Layer

↓

Activation

↓

Linear Layer

↓

Activation

↓

Prediction
```

During training:

Forward pass:

```text
Descriptors

↓

Hidden Representations

↓

Prediction
```

Backward pass:

```text
Loss

↓

Output Layer Gradients

↓

Hidden Layer Gradients

↓

Input Layer Gradients
```

---

# 12.9.12 PyTorch Automatic Differentiation

Calculating millions of gradients manually is impossible.

PyTorch provides:

```text
Autograd
```

Autograd automatically builds a computational graph.

Example:

```python
x

↓

Linear Layer

↓

Activation

↓

Output

↓

Loss
```

PyTorch remembers every operation.

When:

```python
loss.backward()
```

is called,

the gradients are calculated automatically.

---

# 12.9.13 Example With a Neural Network Layer

```python
import torch


import torch.nn as nn



layer = nn.Linear(
    10,
    1
)


x = torch.randn(
    1,
    10
)


prediction = layer(
    x
)


target = torch.tensor(
    [[2.0]]
)


loss_function = nn.MSELoss()


loss = loss_function(
    prediction,
    target
)


loss.backward()
```

After:

```python
loss.backward()
```

every parameter has gradients.

Check:

```python
for parameter in layer.parameters():

    print(
        parameter.grad
    )
```

---

# 12.9.14 Backpropagation in Materials Property Prediction

Consider predicting formation energy.

Input:

```text
Material Descriptors

Density

Composition

Atomic Properties

Structure Features
```

Forward pass:

```text
Descriptors

↓

Neural Network

↓

Predicted Formation Energy
```

Loss:

```text
Predicted Energy

vs

DFT Energy
```

Backward pass:

```text
Energy Error

↓

Gradient Calculation

↓

Update Neural Network Weights
```

After thousands of iterations:

```text
Neural Network

↓

Learns Materials-Property Relationship
```

---

# 12.9.15 Vanishing Gradient Problem

Deep networks contain many layers.

During backpropagation,

gradients travel backward through these layers.

Sometimes gradients become extremely small.

This is called:

```text
Vanishing Gradient
```

Conceptually:

```text
Loss

↓

Layer 5

Gradient = 0.01

↓

Layer 4

Gradient = 0.001

↓

Layer 3

Gradient = 0.00001
```

Early layers learn extremely slowly.

---

# 12.9.16 Exploding Gradient Problem

The opposite problem can also occur.

Gradients become extremely large.

Example:

```text
Gradient

↓

100

↓

10000

↓

1000000
```

This causes unstable training.

Solutions include:

- proper initialization,
- normalization,
- suitable activation functions,
- gradient clipping.

---

# 12.9.17 Backpropagation and Deep Learning Success

Backpropagation was one of the major breakthroughs that enabled modern deep learning.

Combined with:

```text
Large Datasets

+

Powerful Hardware

+

Better Architectures

+

Optimization Algorithms
```

it allows neural networks to learn complex patterns.

In materials science,

this enables models that learn relationships between:

```text
Structure

↓

Representation

↓

Property
```

without requiring every relationship to be manually programmed.

---

# 12.9.18 Summary

Backpropagation is the learning mechanism of neural networks.

The complete process is:

```text
Input Materials Data

↓

Forward Propagation

↓

Prediction

↓

Loss

↓

Backpropagation

↓

Gradient Calculation

↓

Weight Update
```

Using PyTorch:

```python
loss.backward()
```

performs the complex mathematics required to calculate gradients.

The optimizer then uses these gradients to improve the model.

Now that we understand how neural networks learn,

the next step is to understand the practical tools used to build them.

In the next section:

# 12.10 Introduction to PyTorch for Materials Machine Learning

we will begin building complete neural networks using PyTorch.

# 12.10 Introduction to PyTorch for Materials Machine Learning

In the previous sections, we learned the theory behind neural networks.

We now understand:

- neurons,
- layers,
- activation functions,
- loss functions,
- gradient descent,
- backpropagation.

However, implementing all of these algorithms manually for large neural networks would be extremely difficult.

Fortunately, modern deep learning frameworks automate most of this work.

The most popular frameworks are:

- PyTorch
- TensorFlow
- JAX

In this book, we will use **PyTorch** because it is

- easy to learn,
- highly flexible,
- widely used in scientific research,
- the foundation of many state-of-the-art materials informatics libraries.

By the end of this chapter, you will be able to build, train, evaluate, and save neural networks entirely using PyTorch.

---

# 12.10.1 Why PyTorch?

Suppose we want to predict the band gap of materials using a neural network.

Without PyTorch, we would need to manually implement:

- forward propagation,
- activation functions,
- loss functions,
- gradient calculations,
- backpropagation,
- parameter updates,
- optimization algorithms.

This would require hundreds or even thousands of lines of code.

PyTorch allows us to write

```python
loss.backward()

optimizer.step()
```

instead of implementing the complete mathematics ourselves.

This allows researchers to focus on

- model architecture,
- feature engineering,
- scientific interpretation,

rather than low-level mathematical implementation.

---

# 12.10.2 Installing PyTorch

The recommended installation is through pip.

```bash
pip install torch torchvision torchaudio
```

Verify the installation.

```python
import torch

print(torch.__version__)
```

Example output

```text
2.8.0
```

---

# 12.10.3 Checking GPU Availability

Deep learning often requires GPUs.

PyTorch allows us to check whether CUDA is available.

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

or

```text
False
```

If a GPU is available,

its name can be printed.

```python
print(
    torch.cuda.get_device_name(0)
)
```

Example

```text
NVIDIA RTX 4090
```

---

# 12.10.4 Selecting the Computing Device

A common PyTorch pattern is

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print(device)
```

Output

```text
cuda
```

or

```text
cpu
```

Later,

every tensor and model will be moved to this device.

---

# 12.10.5 What Is a Tensor?

The fundamental data structure in PyTorch is the **tensor**.

A tensor is similar to a NumPy array,

but with additional capabilities:

- automatic differentiation,
- GPU acceleration,
- deep learning operations.

Conceptually,

```text
Scalar

↓

Vector

↓

Matrix

↓

Tensor
```

---

# 12.10.6 Creating Scalars

```python
import torch

x = torch.tensor(
    5
)

print(x)
```

Output

```text
tensor(5)
```

---

# 12.10.7 Creating Vectors

```python
vector = torch.tensor(

    [1,2,3,4]

)

print(vector)
```

Output

```text
tensor([1,2,3,4])
```

---

# 12.10.8 Creating Matrices

```python
matrix = torch.tensor(

    [

        [1,2],

        [3,4]

    ]

)

print(matrix)
```

Output

```text
tensor(

[[1,2],

 [3,4]]

)
```

---

# 12.10.9 Tensor Dimensions

We can inspect tensor shapes.

```python
print(
    matrix.shape
)
```

Output

```text
torch.Size([2,2])
```

Similarly,

```python
print(
    vector.shape
)
```

Output

```text
torch.Size([4])
```

Understanding tensor shapes is essential because neural networks expect inputs with specific dimensions.

---

# 12.10.10 Creating Random Tensors

Machine learning often starts with randomly initialized parameters.

PyTorch provides several convenient functions.

Random numbers between 0 and 1.

```python
x = torch.rand(
    3,
    4
)

print(x)
```

Random values from a normal distribution.

```python
x = torch.randn(
    3,
    4
)

print(x)
```

Random integers.

```python
x = torch.randint(

    low=0,

    high=10,

    size=(3,4)

)

print(x)
```

---

# 12.10.11 Creating Common Tensors

Zeros

```python
zeros = torch.zeros(
    5,
    5
)

print(zeros)
```

Ones

```python
ones = torch.ones(
    3,
    4
)

print(ones)
```

Identity matrix

```python
eye = torch.eye(
    4
)

print(eye)
```

---

# 12.10.12 Tensor Data Types

PyTorch supports multiple numerical types.

Check the type.

```python
x = torch.tensor(
    [1,2,3]
)

print(
    x.dtype
)
```

Output

```text
torch.int64
```

Create floating-point tensors.

```python
x = torch.tensor(

    [1.0,2.0,3.0],

    dtype=torch.float32

)

print(
    x.dtype
)
```

Output

```text
torch.float32
```

Most neural network computations use

```text
float32
```

---

# 12.10.13 Moving Tensors to GPU

If CUDA is available,

tensors can be transferred.

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

x = torch.randn(
    5,
    5
)

x = x.to(device)

print(
    x.device
)
```

Output

```text
cuda:0
```

or

```text
cpu
```

---

# 12.10.14 Basic Tensor Operations

Addition

```python
a = torch.tensor(
    [1,2,3]
)

b = torch.tensor(
    [4,5,6]
)

print(
    a + b
)
```

Subtraction

```python
print(
    a - b
)
```

Multiplication

```python
print(
    a * b
)
```

Division

```python
print(
    a / b
)
```

---

# 12.10.15 Matrix Multiplication

Neural networks rely heavily on matrix multiplication.

PyTorch provides

```python
torch.matmul()
```

Example

```python
A = torch.randn(
    3,
    5
)

B = torch.randn(
    5,
    2
)

C = torch.matmul(
    A,
    B
)

print(
    C.shape
)
```

Output

```text
torch.Size([3,2])
```

This operation forms the basis of every fully connected neural network layer.

---

# 12.10.16 Tensor Reshaping

Changing tensor dimensions is extremely common.

Example

```python
x = torch.arange(
    12
)

print(x)
```

Reshape

```python
x = x.reshape(
    3,
    4
)

print(x)
```

Output

```text
tensor(

[[0,1,2,3],

 [4,5,6,7],

 [8,9,10,11]]

)
```

---

# 12.10.17 Converting Between NumPy and PyTorch

NumPy to PyTorch

```python
import numpy as np

array = np.array(
    [1,2,3]
)

tensor = torch.from_numpy(
    array
)

print(tensor)
```

PyTorch to NumPy

```python
array = tensor.numpy()

print(array)
```

This is particularly useful when working with

- NumPy,
- pandas,
- scikit-learn,

alongside PyTorch.

---

# 12.10.18 Creating Tensors from Materials Data

Suppose we generated descriptors using matminer.

```python
import pandas as pd

df = pd.read_csv(
    "materials_descriptors.csv"
)

print(df.head())
```

Assume the dataframe contains

```text
density

volume

atomic_mass

electronegativity

band_gap
```

Separate inputs and targets.

```python
X = df.drop(

    columns=["band_gap"]

)

y = df["band_gap"]
```

Convert to tensors.

```python
X_tensor = torch.tensor(

    X.values,

    dtype=torch.float32

)

y_tensor = torch.tensor(

    y.values,

    dtype=torch.float32

).reshape(
    -1,
    1
)
```

Check the shapes.

```python
print(
    X_tensor.shape
)

print(
    y_tensor.shape
)
```

Example output

```text
torch.Size([5000,4])

torch.Size([5000,1])
```

Now the materials descriptors are ready to be used by a neural network.

---

# 12.10.19 Automatic Differentiation

One of PyTorch's most powerful features is automatic differentiation.

Create a tensor that tracks gradients.

```python
x = torch.tensor(

    2.0,

    requires_grad=True

)

y = x ** 2

print(y)
```

Compute gradients.

```python
y.backward()

print(
    x.grad
)
```

Output

```text
tensor(4.)
```

PyTorch automatically computed

```text
Gradient

=

4
```

without requiring us to derive the expression manually.

---

# 12.10.20 Summary

PyTorch is the primary framework we will use throughout the remainder of this book.

In this section, we learned how to

- install PyTorch,
- create tensors,
- inspect tensor shapes,
- perform mathematical operations,
- move data to GPUs,
- convert NumPy arrays into tensors,
- prepare materials descriptors for neural networks,
- use automatic differentiation.

These operations form the foundation of every deep learning workflow.

In the next section, we will begin constructing our first neural network using PyTorch by learning about the `torch.nn` module and fully connected layers.

# 12.11 Building Neural Networks with `torch.nn`

In the previous section, we learned about PyTorch tensors, automatic differentiation, and how to prepare materials descriptors for machine learning.

However, creating an entire neural network by manually defining every weight and bias would quickly become impractical.

For example, suppose our input consists of **200 material descriptors** generated using matminer.

If our first hidden layer contains **128 neurons**, then the first layer alone requires

```text
200 × 128

=

25,600 weights
```

In addition,

each neuron has its own bias.

Managing thousands or millions of trainable parameters manually would be extremely difficult.

Fortunately, PyTorch provides the **torch.nn** module, which automatically creates and manages neural network layers.

---

# 12.11.1 What Is `torch.nn`?

`torch.nn` is a package that provides ready-made building blocks for neural networks.

Instead of manually implementing every neuron,

we simply combine layers.

Conceptually,

```text
Input Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Output Layer
```

becomes

```python
nn.Linear()

↓

nn.ReLU()

↓

nn.Linear()

↓

Prediction
```

---

# 12.11.2 Importing the Neural Network Module

```python
import torch
import torch.nn as nn
```

Almost every PyTorch neural network begins with

```python
import torch.nn as nn
```

The alias

```python
nn
```

is used throughout the deep learning community.

---

# 12.11.3 The Fully Connected Layer

The most commonly used layer is

```python
nn.Linear()
```

A fully connected layer means

```text
Every neuron

↓

Connected to

↓

Every neuron in the next layer
```

Example

```python
layer = nn.Linear(

    in_features=4,

    out_features=8

)
```

This creates

```text
4 input features

↓

8 output neurons
```

---

# 12.11.4 Understanding Input and Output Features

Suppose our materials dataset contains

```text
Density

Atomic Radius

Electronegativity

Volume
```

There are

```text
4 input features
```

If we choose

```python
nn.Linear(4,8)
```

the layer learns

```text
4 Inputs

↓

8 Learned Features
```

Every output neuron receives information from all four inputs.

---

# 12.11.5 Visualizing a Fully Connected Layer

Conceptually,

```text
Density  ------------\

Radius   -------------\

Electronegativity ----- > Hidden Neuron 1

Volume   -------------/


Density  ------------\

Radius   -------------\

Electronegativity ----- > Hidden Neuron 2

Volume   -------------/

...

↓

8 Hidden Neurons
```

Every connection has

- one weight,
- one contribution to the final prediction.

---

# 12.11.6 Number of Trainable Parameters

Consider

```python
layer = nn.Linear(
    4,
    8
)
```

Weights

```text
4 × 8

=

32
```

Biases

```text
8
```

Total trainable parameters

```text
32

+

8

=

40
```

PyTorch creates all of these automatically.

---

# 12.11.7 Inspecting Layer Parameters

Every layer contains weights and biases.

```python
layer = nn.Linear(
    4,
    8
)

print(layer.weight)
```

Example output

```text
Parameter containing:

tensor(...)
```

View the bias.

```python
print(
    layer.bias
)
```

Check parameter shapes.

```python
print(
    layer.weight.shape
)

print(
    layer.bias.shape
)
```

Output

```text
torch.Size([8,4])

torch.Size([8])
```

---

# 12.11.8 Passing Data Through a Layer

Create sample input.

```python
x = torch.tensor(

    [

        [1.2,4.3,2.1,5.8]

    ],

    dtype=torch.float32

)
```

Pass it through the layer.

```python
output = layer(
    x
)

print(output)
```

Output

```text
tensor([[...]])
```

The result contains

```text
8 output values
```

because the layer has

```text
8 neurons
```

---

# 12.11.9 Understanding Tensor Shapes

Suppose

```python
x.shape
```

returns

```text
torch.Size([32,4])
```

Interpretation

```text
32 materials

↓

Each material has

↓

4 descriptors
```

Passing through

```python
nn.Linear(4,8)
```

produces

```text
torch.Size([32,8])
```

Meaning

```text
32 materials

↓

8 learned features
```

Understanding tensor shapes is one of the most important debugging skills in PyTorch.

---

# 12.11.10 Stacking Multiple Layers

Neural networks become powerful when layers are combined.

Example

```python
model = nn.Sequential(

    nn.Linear(
        4,
        16
    ),

    nn.ReLU(),

    nn.Linear(
        16,
        8
    ),

    nn.ReLU(),

    nn.Linear(
        8,
        1
    )

)
```

Architecture

```text
4 Descriptors

↓

16 Neurons

↓

8 Neurons

↓

1 Prediction
```

---

# 12.11.11 Why Use Multiple Layers?

Each layer learns increasingly complex representations.

Example

First layer

```text
Raw Descriptors

↓

Basic Relationships
```

Second layer

```text
Interactions

↓

Higher-Level Patterns
```

Third layer

```text
Learned Representation

↓

Property Prediction
```

Instead of manually designing relationships,

the network learns them automatically.

---

# 12.11.12 Building a Custom Neural Network Class

Although `nn.Sequential` is convenient,

most research models use custom classes.

Example

```python
import torch.nn as nn


class MaterialsNN(nn.Module):


    def __init__(self):

        super().__init__()


        self.fc1 = nn.Linear(
            4,
            16
        )


        self.fc2 = nn.Linear(
            16,
            8
        )


        self.fc3 = nn.Linear(
            8,
            1
        )


        self.relu = nn.ReLU()


    def forward(self, x):


        x = self.relu(
            self.fc1(x)
        )


        x = self.relu(
            self.fc2(x)
        )


        x = self.fc3(x)


        return x
```

---

# 12.11.13 Creating the Model

Instantiate the model.

```python
model = MaterialsNN()

print(model)
```

Example output

```text
MaterialsNN(

(fc1): Linear(...)

(fc2): Linear(...)

(fc3): Linear(...)
)
```

This summary is useful for verifying the architecture.

---

# 12.11.14 Making Predictions

Generate sample data.

```python
sample = torch.randn(
    5,
    4
)
```

Five materials,

each with four descriptors.

Predict.

```python
prediction = model(
    sample
)

print(
    prediction
)
```

Output shape

```text
torch.Size([5,1])
```

Meaning

```text
5 materials

↓

5 predicted properties
```

---

# 12.11.15 Counting Trainable Parameters

Knowing model size is important.

```python
total = sum(

    parameter.numel()

    for parameter in model.parameters()

)

print(total)
```

Count only trainable parameters.

```python
trainable = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad

)

print(trainable)
```

For larger research models,

this number may reach millions.

---

# 12.11.16 Viewing Individual Parameters

Loop through parameters.

```python
for name, parameter in model.named_parameters():

    print(name)

    print(parameter.shape)

    print(parameter)

    print()
```

Example output

```text
fc1.weight

torch.Size([16,4])

...

fc1.bias

torch.Size([16])

...
```

This is useful when debugging or inspecting models.

---

# 12.11.17 Moving the Model to GPU

If CUDA is available,

move the model.

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

model = model.to(
    device
)
```

Remember:

Input tensors must also be moved.

```python
sample = sample.to(
    device
)
```

Otherwise,

PyTorch raises an error because the model and data are on different devices.

---

# 12.11.18 Example Materials Property Prediction Model

Suppose we generated

```text
145 Magpie descriptors
```

using matminer.

Our architecture could be

```python
class BandGapNN(nn.Module):


    def __init__(self):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                145,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
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
                1
            )

        )


    def forward(self, x):

        return self.network(x)
```

Architecture

```text
145 Descriptors

↓

256 Neurons

↓

128 Neurons

↓

64 Neurons

↓

Band Gap
```

---

# 12.11.19 Good Practices When Designing Networks

Some general guidelines.

Choose the input layer to match the number of descriptors.

Example

```text
200 descriptors

↓

Input layer = 200 neurons
```

For regression problems,

the output layer usually contains

```text
1 neuron
```

Avoid making networks unnecessarily deep for small datasets.

Always monitor

- training loss,
- validation loss,
- overfitting.

---

# 12.11.20 Summary

The `torch.nn` module provides powerful building blocks for constructing neural networks.

In this section, we learned how to

- create fully connected layers,
- inspect trainable parameters,
- build models using `nn.Sequential`,
- create custom neural network classes,
- move models to GPUs,
- design networks for materials property prediction.

At this point, we have successfully built a neural network architecture.

However, the model still contains randomly initialized weights and cannot make meaningful predictions.

The next step is to understand **how data flows through the network during a forward pass**, producing predictions that can later be used for training.

In the next section, we will study:

# 12.12 Forward Propagation in Neural Networks

# 12.12 Forward Propagation in Neural Networks

In the previous section, we built complete neural network architectures using the `torch.nn` module.

Although the model now contains layers, weights, biases, and activation functions, it still does not perform any useful task until data passes through it.

The process of moving data from the input layer to the output layer is called **forward propagation** (or simply the **forward pass**).

During the forward pass, the neural network transforms raw input features into a prediction.

For materials informatics, the workflow becomes

```text
Material Descriptors

↓

Input Layer

↓

Hidden Layer 1

↓

Activation Function

↓

Hidden Layer 2

↓

Activation Function

↓

Output Layer

↓

Predicted Property
```

Unlike backpropagation, no learning occurs during the forward pass.

The network simply performs calculations using its current weights and biases.

---

# 12.12.1 Purpose of Forward Propagation

The objective of forward propagation is simple.

Given an input,

```text
Material Features
```

produce an output,

```text
Predicted Property
```

Examples include

```text
Descriptors

↓

Band Gap
```

```text
Descriptors

↓

Formation Energy
```

```text
Descriptors

↓

Elastic Modulus
```

```text
Descriptors

↓

Bulk Modulus
```

Every neural network prediction begins with a forward pass.

---

# 12.12.2 Flow of Information

Consider a neural network with two hidden layers.

```text
Input Features

↓

Linear Layer

↓

ReLU

↓

Linear Layer

↓

ReLU

↓

Output Layer

↓

Prediction
```

Notice that information always moves

```text
Left

↓

Right
```

or

```text
Input

↓

Output
```

There is no feedback during forward propagation.

---

# 12.12.3 Forward Propagation Inside One Neuron

Consider one neuron.

```text
Input Features

↓

Weighted Sum

↓

Bias

↓

Activation Function

↓

Output
```

The neuron performs three operations.

Step 1

Compute the weighted sum.

```text
Weighted Sum

=

(Input × Weight)

+

Bias
```

Step 2

Apply the activation function.

```text
Activation

↓

ReLU

↓

Sigmoid

↓

Tanh
```

Step 3

Pass the result to the next layer.

---

# 12.12.4 Forward Pass Through Multiple Layers

Suppose our neural network is

```text
4 Inputs

↓

8 Neurons

↓

4 Neurons

↓

1 Output
```

Forward propagation becomes

```text
Input

↓

Linear(4 → 8)

↓

ReLU

↓

Linear(8 → 4)

↓

ReLU

↓

Linear(4 → 1)

↓

Prediction
```

Every layer transforms the representation learned by the previous layer.

---

# 12.12.5 Example Using Material Descriptors

Suppose we extracted four descriptors.

```text
Density

Volume

Average Atomic Mass

Electronegativity
```

The feature vector becomes

```text
[

5.21,

63.8,

34.5,

2.18

]
```

This vector is the input to the neural network.

---

# 12.12.6 Creating the Input Tensor

```python
import torch

x = torch.tensor(

    [

        [5.21,63.8,34.5,2.18]

    ],

    dtype=torch.float32

)

print(x)
```

Output

```text
tensor([[ 5.2100, 63.8000, 34.5000, 2.1800]])
```

Shape

```python
print(
    x.shape
)
```

Output

```text
torch.Size([1,4])
```

Interpretation

```text
1 material

↓

4 descriptors
```

---

# 12.12.7 Passing Data Through One Layer

Create a simple layer.

```python
import torch.nn as nn

layer = nn.Linear(

    4,

    3

)
```

Pass the input.

```python
output = layer(x)

print(output)
```

Example output

```text
tensor([[0.42,-1.38,2.14]])
```

Shape

```python
print(
    output.shape
)
```

Output

```text
torch.Size([1,3])
```

The layer has transformed four descriptors into three learned features.

---

# 12.12.8 Adding an Activation Function

Neural networks rarely use only linear layers.

Instead,

they combine them with activation functions.

```python
relu = nn.ReLU()

activated = relu(output)

print(activated)
```

Example output

```text
tensor([[0.42,0.00,2.14]])
```

Negative values become zero.

---

# 12.12.9 Building a Complete Forward Pass

Create a small model.

```python
model = nn.Sequential(

    nn.Linear(
        4,
        8
    ),

    nn.ReLU(),

    nn.Linear(
        8,
        4
    ),

    nn.ReLU(),

    nn.Linear(
        4,
        1
    )

)
```

Pass the material descriptors through the network.

```python
prediction = model(x)

print(prediction)
```

Example output

```text
tensor([[1.83]])
```

This is the predicted material property.

---

# 12.12.10 Forward Propagation Using a Custom Model

Using the custom model from the previous section,

```python
model = MaterialsNN()

prediction = model(x)

print(prediction)
```

Notice something important.

We never explicitly call

```python
model.forward(x)
```

Instead,

we simply write

```python
model(x)
```

PyTorch automatically calls the `forward()` method internally.

---

# 12.12.11 Batch Predictions

Neural networks usually process many materials simultaneously.

Example

```python
batch = torch.tensor(

    [

        [5.2,63.8,34.5,2.18],

        [7.1,58.2,40.1,1.95],

        [3.9,72.5,27.3,2.75]

    ],

    dtype=torch.float32

)

print(batch.shape)
```

Output

```text
torch.Size([3,4])
```

Predict.

```python
prediction = model(batch)

print(prediction)
```

Output shape

```text
torch.Size([3,1])
```

Interpretation

```text
3 materials

↓

3 predictions
```

This is far more efficient than predicting one material at a time.

---

# 12.12.12 Inspecting Intermediate Outputs

Sometimes we want to observe what happens after each layer.

Instead of using `nn.Sequential`,

we can execute each layer individually.

```python
fc1 = nn.Linear(
    4,
    8
)

relu = nn.ReLU()

fc2 = nn.Linear(
    8,
    1
)

x1 = fc1(x)

print(x1)

x2 = relu(x1)

print(x2)

prediction = fc2(x2)

print(prediction)
```

This approach is extremely useful for debugging.

---

# 12.12.13 Printing Tensor Shapes During Forward Propagation

A common debugging technique is printing tensor shapes.

```python
class DebugNN(nn.Module):


    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            4,
            8
        )

        self.fc2 = nn.Linear(
            8,
            1
        )


    def forward(self, x):

        print(
            "Input:",
            x.shape
        )

        x = self.fc1(x)

        print(
            "After fc1:",
            x.shape
        )

        x = torch.relu(x)

        x = self.fc2(x)

        print(
            "Output:",
            x.shape
        )

        return x
```

Run

```python
model = DebugNN()

prediction = model(batch)
```

Example output

```text
Input: torch.Size([3,4])

After fc1: torch.Size([3,8])

Output: torch.Size([3,1])
```

Printing shapes is one of the easiest ways to locate dimension mismatch errors.

---

# 12.12.14 Forward Propagation During Training

During training,

every mini-batch follows exactly the same sequence.

```text
Mini-Batch

↓

Forward Pass

↓

Prediction

↓

Loss Function

↓

Backpropagation

↓

Optimizer Update
```

Notice that

```text
Forward Pass

↓

Prediction
```

always happens before

```text
Loss Calculation
```

because the loss cannot be computed until predictions exist.

---

# 12.12.15 Forward Propagation During Inference

After training,

the workflow becomes much simpler.

```text
New Material

↓

Descriptors

↓

Neural Network

↓

Prediction
```

No gradients are calculated.

No weights are updated.

The model simply predicts.

---

# 12.12.16 Disabling Gradient Calculation

During inference,

gradient calculation wastes memory and computation.

PyTorch provides

```python
torch.no_grad()
```

Example

```python
model.eval()

with torch.no_grad():

    prediction = model(batch)

print(prediction)
```

Advantages

- Faster prediction
- Lower memory usage
- No unnecessary gradient computation

This is the standard approach for deploying trained models.

---

# 12.12.17 Example Materials Workflow

A complete inference pipeline for band gap prediction might look like

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

Descriptors

↓

PyTorch Tensor

↓

Neural Network

↓

Predicted Band Gap
```

Code

```python
structure_features = torch.tensor(

    descriptor_vector,

    dtype=torch.float32

).reshape(
    1,
    -1
)

model.eval()

with torch.no_grad():

    band_gap = model(
        structure_features
    )

print(band_gap.item())
```

---

# 12.12.18 Common Mistakes During Forward Propagation

### Incorrect Input Shape

Wrong

```python
x = torch.randn(4)
```

Correct

```python
x = torch.randn(
    1,
    4
)
```

The first dimension represents the batch size.

---

### Incorrect Data Type

Wrong

```python
torch.tensor(
    [1,2,3]
)
```

This creates integers.

Correct

```python
torch.tensor(

    [1,2,3],

    dtype=torch.float32

)
```

Neural networks typically require floating-point inputs.

---

### Forgetting Evaluation Mode

Always write

```python
model.eval()
```

before inference,

especially when using

- Dropout
- Batch Normalization

---

# 12.12.19 Summary

Forward propagation is the process of transforming input features into predictions using the current neural network parameters.

During a forward pass,

information flows through

```text
Input

↓

Linear Layers

↓

Activation Functions

↓

Output Layer

↓

Prediction
```

Using PyTorch,

a complete prediction is often as simple as

```python
prediction = model(x)
```

Behind this single line,

PyTorch performs all matrix multiplications, bias additions, and activation function evaluations required to generate the output.

The forward pass produces predictions,

but it does not measure how good those predictions are.

The next step is to compare the predictions with the true target values.

This is the role of **loss functions**, which quantify prediction error and provide the objective that the neural network must minimize during training.

In the next section, we will study:

# 12.13 Loss Functions in Deep Learning

# 12.13 Loss Functions in Deep Learning

In the previous section, we learned how a neural network performs **forward propagation** to generate predictions.

However, producing a prediction is only part of the learning process.

After making a prediction, the neural network must answer an important question.

> **How good is this prediction?**

To answer this question, we compare the predicted value with the actual value.

The numerical measure of this difference is called the **loss**.

The objective of training is simple.

```text
Prediction

↓

Loss Function

↓

Loss Value

↓

Backpropagation

↓

Update Weights

↓

Better Prediction
```

Without a loss function, the neural network has no way of determining whether its predictions are improving or becoming worse.

---

# 12.13.1 What Is a Loss Function?

A **loss function** is a mathematical function that measures the error between the predicted output of a model and the true target value.

Conceptually,

```text
Prediction

↓

Compare

↓

Actual Value

↓

Loss
```

The loss is a single numerical value.

```text
Small Loss

↓

Good Prediction
```

```text
Large Loss

↓

Poor Prediction
```

During training, the optimizer attempts to reduce the loss as much as possible.

---

# 12.13.2 Loss Versus Error

Although the words **error** and **loss** are often used interchangeably, they have slightly different meanings.

An error is simply the difference between prediction and target.

```text
Error

=

Prediction

-

Target
```

A loss function transforms this error into a positive numerical value that can be optimized.

For example,

```text
Prediction

↓

Error

↓

Loss Function

↓

Loss
```

The optimizer always minimizes the loss rather than the raw prediction error.

---

# 12.13.3 Example: Band Gap Prediction

Suppose we are predicting the band gap of a semiconductor.

Actual value

```text
2.80 eV
```

Prediction

```text
2.55 eV
```

Prediction error

```text
2.55

-

2.80

=

-0.25 eV
```

The loss function converts this difference into a value that can be minimized during training.

---

# 12.13.4 Regression and Classification Loss Functions

The choice of loss function depends on the machine learning task.

For regression problems,

examples include

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss

For classification problems,

examples include

- Binary Cross Entropy
- Cross Entropy Loss

Most materials property prediction problems are regression problems.

Examples include

- band gap prediction,
- formation energy prediction,
- elastic modulus prediction,
- bulk modulus prediction,
- dielectric constant prediction.

Therefore,

Mean Squared Error is one of the most commonly used loss functions in materials informatics.

---

# 12.13.5 Mean Squared Error (MSE)

Mean Squared Error measures the average squared difference between predictions and targets.

Conceptually,

```text
Prediction

↓

Difference

↓

Square

↓

Average

↓

MSE
```

Because the error is squared,

large prediction errors receive much larger penalties than small errors.

---

# 12.13.6 Manual Example of MSE

Suppose we have three materials.

| Material | Actual Band Gap | Predicted Band Gap |
|----------|----------------:|-------------------:|
| A | 2.5 | 2.2 |
| B | 1.8 | 2.0 |
| C | 3.4 | 3.1 |

Calculate the prediction errors.

```text
Material A

↓

2.2 - 2.5

↓

-0.3
```

```text
Material B

↓

2.0 - 1.8

↓

0.2
```

```text
Material C

↓

3.1 - 3.4

↓

-0.3
```

Square each error.

```text
0.09

0.04

0.09
```

Average

```text
(0.09 + 0.04 + 0.09)

/

3

=

0.0733
```

The Mean Squared Error is

```text
0.0733
```

---

# 12.13.7 Computing MSE in PyTorch

PyTorch already implements Mean Squared Error.

```python
import torch
import torch.nn as nn


prediction = torch.tensor(

    [[2.2],
     [2.0],
     [3.1]]

)

target = torch.tensor(

    [[2.5],
     [1.8],
     [3.4]]

)


loss_function = nn.MSELoss()

loss = loss_function(

    prediction,

    target

)

print(loss)
```

Output

```text
tensor(0.0733)
```

---

# 12.13.8 Mean Absolute Error (MAE)

Another popular regression loss is Mean Absolute Error.

Instead of squaring the error,

it calculates the absolute difference.

Conceptually,

```text
Prediction

↓

Difference

↓

Absolute Value

↓

Average
```

Unlike MSE,

MAE does not strongly penalize very large prediction errors.

---

# 12.13.9 Computing MAE in PyTorch

```python
loss_function = nn.L1Loss()

loss = loss_function(

    prediction,

    target

)

print(loss)
```

Example output

```text
tensor(0.2667)
```

This value represents the average prediction error.

---

# 12.13.10 Comparing MSE and MAE

Suppose one prediction is extremely poor.

```text
Actual

↓

2.5
```

Prediction

```text
12.5
```

Error

```text
10
```

For MAE,

```text
Loss

↓

10
```

For MSE,

```text
Loss

↓

100
```

Because the error is squared,

MSE penalizes large mistakes much more aggressively.

---

# 12.13.11 Huber Loss

Huber Loss combines the advantages of both MSE and MAE.

For small errors,

it behaves like MSE.

For large errors,

it behaves like MAE.

This makes it less sensitive to outliers.

PyTorch implementation

```python
loss_function = nn.HuberLoss()

loss = loss_function(

    prediction,

    target

)

print(loss)
```

Huber Loss is often useful when experimental datasets contain noisy measurements.

---

# 12.13.12 Loss During Training

During every training iteration,

the workflow is

```text
Mini-Batch

↓

Forward Pass

↓

Prediction

↓

Loss Function

↓

Loss Value
```

Only after the loss is computed can backpropagation begin.

---

# 12.13.13 Example Training Step

Suppose we already have

- a neural network,
- input descriptors,
- target values.

```python
prediction = model(

    X_batch

)

loss_function = nn.MSELoss()

loss = loss_function(

    prediction,

    y_batch

)

print(loss.item())
```

Example output

```text
0.1824
```

This value tells us how well the model predicts the current mini-batch.

---

# 12.13.14 Backpropagation Begins from the Loss

The loss is the starting point of learning.

```text
Prediction

↓

Loss

↓

loss.backward()

↓

Gradients

↓

Optimizer
```

In PyTorch,

the entire backward pass begins with

```python
loss.backward()
```

This single line computes gradients for every trainable parameter.

---

# 12.13.15 Monitoring Loss During Training

One of the most important indicators of successful learning is the training loss.

A typical training loop prints the loss after every epoch.

```python
for epoch in range(10):

    prediction = model(

        X_train

    )

    loss = loss_function(

        prediction,

        y_train

    )

    print(

        epoch,

        loss.item()

    )
```

Example output

```text
Epoch 0  2.85

Epoch 1  1.74

Epoch 2  1.09

Epoch 3  0.72

Epoch 4  0.41

Epoch 5  0.28
```

A steadily decreasing loss usually indicates that the model is learning.

---

# 12.13.16 Visualizing the Loss Curve

Training loss is often plotted against the epoch number.

```text
Loss

^

|

| *

|  *

|    *

|      *

|         *

|            *

+---------------------------->

          Epoch
```

A decreasing curve generally indicates successful optimization.

Later in this book, we will generate these plots using Matplotlib.

---

# 12.13.17 Recording Loss During Training

Instead of printing the loss,

it is common to store it.

```python
loss_history = []

for epoch in range(100):

    prediction = model(

        X_train

    )

    loss = loss_function(

        prediction,

        y_train

    )

    loss_history.append(

        loss.item()

    )
```

The stored values can later be visualized.

---

# 12.13.18 Plotting the Loss Curve

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(

    loss_history

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Loss"

)

plt.title(

    "Training Loss"

)

plt.grid(True)

plt.show()
```

This plot helps determine whether the model is improving over time.

---

# 12.13.19 Loss Functions in Materials Informatics

Different materials science problems may use different loss functions.

| Task | Typical Loss |
|------|--------------|
| Band gap prediction | MSE |
| Formation energy prediction | MSE |
| Elastic modulus prediction | MSE |
| Thermal conductivity prediction | MSE |
| Crystal classification | Cross Entropy |
| Phase classification | Cross Entropy |

Regression problems dominate materials property prediction,

making MSE one of the most frequently used choices.

---

# 12.13.20 Choosing a Loss Function

General guidelines

Use **MSE** when

- large errors should be strongly penalized,
- the dataset contains relatively few outliers.

Use **MAE** when

- robustness to outliers is important,
- average prediction error is the primary concern.

Use **Huber Loss** when

- the dataset contains some noisy or unreliable measurements,
- a balance between MSE and MAE is desired.

---

# 12.13.21 Summary

A loss function measures how well a neural network performs on a given prediction.

During every training iteration,

the workflow is

```text
Input Data

↓

Forward Propagation

↓

Prediction

↓

Loss Function

↓

Loss

↓

Backpropagation

↓

Gradient Calculation

↓

Weight Update
```

PyTorch provides several built-in loss functions, including

- `nn.MSELoss()`
- `nn.L1Loss()`
- `nn.HuberLoss()`
- `nn.CrossEntropyLoss()`

Among these,

**Mean Squared Error** is the most commonly used loss function for regression tasks in materials informatics.

Now that we know how to generate predictions and measure their quality,

the next step is to update the model parameters efficiently.

In the next section, we will study **optimizers** in PyTorch and learn how algorithms such as **SGD** and **Adam** automatically update neural network weights during training.

# 12.15 Building a Complete Neural Network Training Pipeline

In the previous sections, we learned the individual components required to train a neural network.

We now understand

- neural network architectures,
- forward propagation,
- loss functions,
- backpropagation,
- optimizers.

The next step is to combine all of these components into a complete training pipeline.

This section demonstrates how a real deep learning project is organized.

Although we will use a relatively simple regression problem, the same workflow is used for much larger materials informatics models, including CGCNN, MEGNet, and M3GNet.

The overall workflow is

```text
Raw Dataset

↓

Train / Validation Split

↓

PyTorch Dataset

↓

DataLoader

↓

Neural Network

↓

Loss Function

↓

Optimizer

↓

Training Loop

↓

Validation

↓

Save Best Model
```

This workflow is the foundation of nearly every deep learning project.

---

# 12.15.1 Preparing the Dataset

Assume we already generated descriptors using matminer.

Our CSV file contains

| Density | Volume | Mean Atomic Mass | Electronegativity | Band Gap |
|----------|--------|------------------|-------------------|----------|

Load the dataset.

```python
import pandas as pd

df = pd.read_csv(
    "materials_descriptors.csv"
)

print(df.head())
```

Separate the features from the target.

```python
X = df.drop(
    columns=["band_gap"]
)

y = df["band_gap"]
```

---

# 12.15.2 Splitting the Dataset

Before training,

we divide the dataset into

- training data
- validation data

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)
```

Example

```text
5000 Materials

↓

4000 Training

↓

1000 Validation
```

---

# 12.15.3 Converting Data to PyTorch Tensors

```python
import torch

X_train = torch.tensor(

    X_train.values,

    dtype=torch.float32

)

X_valid = torch.tensor(

    X_valid.values,

    dtype=torch.float32

)

y_train = torch.tensor(

    y_train.values,

    dtype=torch.float32

).reshape(-1,1)

y_valid = torch.tensor(

    y_valid.values,

    dtype=torch.float32

).reshape(-1,1)
```

Check the shapes.

```python
print(X_train.shape)
print(y_train.shape)
```

Example

```text
torch.Size([4000,145])

torch.Size([4000,1])
```

---

# 12.15.4 Creating a PyTorch Dataset

Although tensors can be used directly,

PyTorch recommends wrapping them inside a Dataset.

```python
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(

    X_train,

    y_train

)

valid_dataset = TensorDataset(

    X_valid,

    y_valid

)
```

Each sample now consists of

```text
(Input Features,

Target)
```

---

# 12.15.5 Creating DataLoaders

DataLoader automatically

- creates mini-batches,
- shuffles data,
- efficiently loads samples.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(

    train_dataset,

    batch_size=64,

    shuffle=True

)

valid_loader = DataLoader(

    valid_dataset,

    batch_size=64,

    shuffle=False

)
```

Workflow

```text
Dataset

↓

DataLoader

↓

Mini-Batches

↓

Neural Network
```

---

# 12.15.6 Building the Neural Network

Suppose our descriptors contain

```text
145 features
```

A simple regression network could be

```python
import torch.nn as nn

class BandGapNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(145,256),

            nn.ReLU(),

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Linear(128,64),

            nn.ReLU(),

            nn.Linear(64,1)

        )

    def forward(self,x):

        return self.network(x)
```

Create the model.

```python
model = BandGapNN()
```

---

# 12.15.7 Selecting the Device

```python
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

model = model.to(device)
```

Every batch must also be moved to the same device.

---

# 12.15.8 Defining the Loss Function

For regression,

Mean Squared Error is commonly used.

```python
loss_function = nn.MSELoss()
```

---

# 12.15.9 Defining the Optimizer

```python
import torch.optim as optim

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001

)
```

---

# 12.15.10 The Structure of a Training Loop

Every epoch follows the same workflow.

```text
Mini-Batch

↓

Forward Pass

↓

Loss

↓

Backpropagation

↓

Optimizer Step

↓

Next Mini-Batch
```

---

# 12.15.11 Writing the Training Loop

```python
epochs = 50

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        prediction = model(

            X_batch

        )

        loss = loss_function(

            prediction,

            y_batch

        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(

        f"Epoch {epoch+1}",

        f"Loss = {epoch_loss:.5f}"

    )
```

Notice that every epoch consists of many mini-batches rather than the entire dataset.

---

# 12.15.12 Why Use `model.train()`?

PyTorch models have two operating modes.

Training mode

```python
model.train()
```

Evaluation mode

```python
model.eval()
```

Training mode activates layers such as

- Dropout
- Batch Normalization

These layers behave differently during evaluation.

Always switch to training mode before training begins.

---

# 12.15.13 Evaluating on the Validation Set

After each epoch,

evaluate the model.

```python
model.eval()

validation_loss = 0.0

with torch.no_grad():

    for X_batch, y_batch in valid_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        prediction = model(

            X_batch

        )

        loss = loss_function(

            prediction,

            y_batch

        )

        validation_loss += loss.item()

validation_loss /= len(valid_loader)

print(

    "Validation Loss:",

    validation_loss

)
```

Notice that

```python
torch.no_grad()
```

is used because gradients are unnecessary during evaluation.

---

# 12.15.14 Recording the Training History

Instead of printing only the latest loss,

store the history.

```python
training_history = []

validation_history = []
```

Inside the loop

```python
training_history.append(

    epoch_loss

)

validation_history.append(

    validation_loss

)
```

This information can later be visualized.

---

# 12.15.15 Plotting Learning Curves

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(

    training_history,

    label="Training"

)

plt.plot(

    validation_history,

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Learning Curve")

plt.legend()

plt.grid(True)

plt.show()
```

A typical successful training curve looks like

```text
Loss

^

| *

|  *

|   *

|    *

|      *

|        *

+---------------------------->

        Epoch
```

---

# 12.15.16 Saving the Best Model

Suppose we wish to save the model whenever the validation loss improves.

```python
best_loss = float("inf")
```

Inside the training loop

```python
if validation_loss < best_loss:

    best_loss = validation_loss

    torch.save(

        model.state_dict(),

        "best_model.pth"

    )
```

This prevents accidentally saving a poorly performing model.

---

# 12.15.17 Loading the Saved Model

```python
model.load_state_dict(

    torch.load(

        "best_model.pth"

    )

)

model.eval()
```

The trained model is now ready for inference.

---

# 12.15.18 Making Predictions

Predict new materials.

```python
with torch.no_grad():

    prediction = model(

        X_valid.to(device)

    )
```

Convert predictions back to NumPy.

```python
prediction = prediction.cpu().numpy()
```

These values can now be compared with experimental or DFT results.

---

# 12.15.19 Complete Workflow

The entire pipeline can now be summarized.

```text
Descriptors

↓

Train/Test Split

↓

TensorDataset

↓

DataLoader

↓

Mini-Batch

↓

Neural Network

↓

Prediction

↓

Loss Function

↓

Backpropagation

↓

Optimizer

↓

Updated Parameters

↓

Validation

↓

Save Best Model
```

This workflow is essentially identical to the one used in modern deep learning research papers.

---

# 12.15.20 Improving the Pipeline

The pipeline we built is intentionally simple.

Real research projects often include additional components.

Examples include

- feature normalization,
- learning rate schedulers,
- early stopping,
- gradient clipping,
- mixed precision training,
- checkpoint saving,
- experiment logging,
- hyperparameter optimization.

These techniques improve training stability and model performance.

---

# 12.15.21 Example: Predicting Band Gaps from Materials Descriptors

A practical materials informatics workflow might look like

```text
Crystal Structures

↓

Quantum ESPRESSO

↓

Optimized Structures

↓

pymatgen

↓

Structure Objects

↓

matminer

↓

145 Material Descriptors

↓

Train / Validation Split

↓

PyTorch Neural Network

↓

Band Gap Prediction

↓

Model Evaluation

↓

Deployment
```

Notice how PyTorch represents only one component of a much larger materials informatics workflow.

---

# 12.15.22 Common Beginner Mistakes

### Forgetting `model.train()`

This can produce incorrect behavior when using Dropout or Batch Normalization.

---

### Forgetting `model.eval()`

Predictions may become inconsistent during evaluation.

---

### Forgetting `torch.no_grad()`

Inference becomes slower and consumes unnecessary memory.

---

### Forgetting to Move Data to the GPU

Wrong

```python
prediction = model(X_batch)
```

Correct

```python
X_batch = X_batch.to(device)

prediction = model(X_batch)
```

The model and input tensors must always reside on the same device.

---

### Using an Extremely Large Batch Size

Very large batches

- require more GPU memory,
- may reduce generalization,
- can cause out-of-memory errors.

Typical values include

```text
32

64

128

256
```

---

# 12.15.23 Summary

In this section, we assembled all of the individual components introduced throughout this chapter into a complete neural network training pipeline.

We learned how to

- prepare datasets,
- create `TensorDataset` objects,
- build `DataLoader` instances,
- define neural networks,
- train using mini-batches,
- evaluate on validation data,
- record learning curves,
- save the best-performing model,
- load trained models for inference.

This workflow represents the standard training pipeline used in modern PyTorch projects and forms the foundation for more advanced architectures such as Graph Neural Networks.

At this point, you have all the fundamental knowledge required to build, train, and evaluate fully connected neural networks for materials property prediction.

In the next section, we will explore **techniques for improving neural network performance**, including regularization methods such as **Dropout**, **Batch Normalization**, **Early Stopping**, and **Learning Rate Scheduling**, which are essential for training reliable deep learning models on real-world materials datasets.

# 12.16 Improving Neural Network Performance

The neural network developed in the previous section is capable of learning from data and making predictions.

However, simply increasing the number of training epochs does not always improve model performance.

In practice, neural networks often suffer from problems such as

- overfitting,
- unstable training,
- slow convergence,
- poor generalization,
- exploding gradients,
- vanishing gradients.

Modern deep learning addresses these issues using a collection of training techniques known as **regularization** and **optimization improvements**.

These techniques do not fundamentally change the neural network architecture.

Instead, they improve **how the model learns**.

This section introduces the most widely used techniques in modern deep learning, many of which appear in research papers on materials informatics.

---

# 12.16.1 Why Neural Networks Overfit

Recall from earlier chapters that overfitting occurs when a model memorizes the training data rather than learning general patterns.

Conceptually,

```text
Training Loss

↓

Very Small
```

```text
Validation Loss

↓

Large
```

The model performs well on materials it has already seen,

but poorly on new materials.

Suppose we train a neural network to predict band gaps using 5,000 materials.

If the model memorizes these structures instead of learning the relationship between composition and band gap,

it may fail when predicting the properties of completely new compounds.

---

# 12.16.2 Training and Validation Curves

A common way to detect overfitting is by monitoring both training and validation loss.

Ideal training

```text
Loss

^

|

| *

|  *

|   *

|    *

|      *

|        *

+---------------------------->

          Epoch
```

Overfitting

```text
Loss

^

|

| *

|  *

|   *

|    *

|     *\

|      \____

+---------------------------->

          Epoch

Training Loss

↓

Continues Decreasing

Validation Loss

↓

Starts Increasing
```

When validation loss begins increasing while training loss continues decreasing,

the model is beginning to overfit.

---

# 12.16.3 Dropout

One of the simplest methods for reducing overfitting is **Dropout**.

Instead of using every neuron during every training iteration,

Dropout randomly disables some neurons.

Conceptually,

Without Dropout

```text
Neuron

↓

Neuron

↓

Neuron

↓

Neuron
```

With Dropout

```text
Neuron

↓

OFF

↓

Neuron

↓

OFF

↓

Neuron
```

Because different neurons are temporarily disabled during training,

the network cannot rely too heavily on any single pathway.

This encourages the model to learn more robust representations.

---

# 12.16.4 Using Dropout in PyTorch

PyTorch provides the

```python
nn.Dropout()
```

layer.

Example

```python
import torch.nn as nn

model = nn.Sequential(

    nn.Linear(145,256),

    nn.ReLU(),

    nn.Dropout(

        p=0.3

    ),

    nn.Linear(256,128),

    nn.ReLU(),

    nn.Dropout(

        p=0.3

    ),

    nn.Linear(128,1)

)
```

Here,

```python
p=0.3
```

means

```text
30%

↓

Probability that a neuron is temporarily disabled during training.
```

---

# 12.16.5 Understanding the Dropout Probability

Common dropout probabilities include

| Probability | Typical Usage |
|-------------|---------------|
| 0.1 | Very small regularization |
| 0.2 | Light regularization |
| 0.3 | Moderate regularization |
| 0.5 | Strong regularization |

Large dropout values may prevent the network from learning effectively.

Small values may not sufficiently reduce overfitting.

---

# 12.16.6 Dropout During Training and Evaluation

An important characteristic of Dropout is that it behaves differently depending on the model mode.

Training mode

```python
model.train()
```

Dropout is active.

Some neurons are randomly disabled.

Evaluation mode

```python
model.eval()
```

Dropout is automatically disabled.

Every neuron participates in prediction.

PyTorch manages this behavior automatically.

---

# 12.16.7 Batch Normalization

Another important technique is **Batch Normalization**.

Instead of changing the architecture,

Batch Normalization normalizes the outputs of intermediate layers.

Conceptually,

```text
Layer Output

↓

Normalize

↓

Stable Distribution

↓

Next Layer
```

Benefits include

- faster convergence,
- improved numerical stability,
- reduced sensitivity to initialization,
- slightly improved generalization.

---

# 12.16.8 Using Batch Normalization

PyTorch implementation

```python
model = nn.Sequential(

    nn.Linear(

        145,

        256

    ),

    nn.BatchNorm1d(

        256

    ),

    nn.ReLU(),

    nn.Linear(

        256,

        128

    ),

    nn.BatchNorm1d(

        128

    ),

    nn.ReLU(),

    nn.Linear(

        128,

        1

    )

)
```

Notice that

```python
nn.BatchNorm1d()
```

is typically placed

```text
After Linear

↓

Before ReLU
```

---

# 12.16.9 Early Stopping

Sometimes the model begins to overfit after a certain number of epochs.

Continuing training only worsens validation performance.

Early stopping automatically terminates training once validation performance stops improving.

Conceptually,

```text
Epoch

↓

Validation Loss Improves

↓

Save Model

↓

Validation Loss Stops Improving

↓

Stop Training
```

This prevents unnecessary training and reduces overfitting.

---

# 12.16.10 Simple Early Stopping Implementation

```python
best_loss = float("inf")

patience = 10

counter = 0
```

Inside the training loop

```python
if validation_loss < best_loss:

    best_loss = validation_loss

    counter = 0

else:

    counter += 1

if counter >= patience:

    print(

        "Early stopping."

    )

    break
```

If the validation loss does not improve for

```text
10 epochs
```

training automatically stops.

---

# 12.16.11 Learning Rate Scheduling

The learning rate does not need to remain constant throughout training.

Many deep learning models begin with a relatively large learning rate.

As training progresses,

the learning rate is gradually reduced.

Conceptually,

```text
Training Begins

↓

Large Learning Rate

↓

Fast Learning

↓

Reduce Learning Rate

↓

Fine-Tuning
```

Learning rate scheduling often produces better final models.

---

# 12.16.12 Step Learning Rate Scheduler

PyTorch provides many learning rate schedulers.

One example is

```python
StepLR
```

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(

    optimizer,

    step_size=20,

    gamma=0.5

)
```

Interpretation

Every

```text
20 epochs
```

the learning rate becomes

```text
50%

↓

of its previous value.
```

---

# 12.16.13 Updating the Scheduler

Inside the training loop,

call

```python
scheduler.step()
```

Example

```python
for epoch in range(100):

    ...

    optimizer.step()

    scheduler.step()
```

The learning rate automatically changes throughout training.

---

# 12.16.14 Viewing the Current Learning Rate

```python
print(

    optimizer.param_groups[0]["lr"]

)
```

Example output

```text
0.0005
```

This is useful for debugging and monitoring the scheduler.

---

# 12.16.15 Weight Decay

Another popular regularization method is **weight decay**.

Instead of allowing weights to grow without limit,

weight decay encourages smaller parameter values.

PyTorch implementation

```python
optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=0.001,

    weight_decay=1e-5

)
```

Weight decay is widely used in modern deep learning because it often improves generalization.

---

# 12.16.16 Gradient Clipping

Occasionally,

very large gradients cause unstable training.

Gradient clipping limits their magnitude.

PyTorch implementation

```python
torch.nn.utils.clip_grad_norm_(

    model.parameters(),

    max_norm=1.0

)
```

Place gradient clipping after

```python
loss.backward()
```

and before

```python
optimizer.step()
```

Example

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(

    model.parameters(),

    max_norm=1.0

)

optimizer.step()
```

Gradient clipping is especially useful for very deep neural networks.

---

# 12.16.17 Combining Multiple Techniques

A modern training loop often combines several improvements.

Example

```python
model.train()

prediction = model(

    X_batch

)

loss = loss_function(

    prediction,

    y_batch

)

optimizer.zero_grad()

loss.backward()

torch.nn.utils.clip_grad_norm_(

    model.parameters(),

    max_norm=1.0

)

optimizer.step()

scheduler.step()
```

This training loop includes

- gradient clipping,
- optimization,
- learning rate scheduling.

---

# 12.16.18 Example Research Architecture

A typical neural network used in materials informatics might be

```python
model = nn.Sequential(

    nn.Linear(145,256),

    nn.BatchNorm1d(256),

    nn.ReLU(),

    nn.Dropout(0.2),

    nn.Linear(256,128),

    nn.BatchNorm1d(128),

    nn.ReLU(),

    nn.Dropout(0.2),

    nn.Linear(128,64),

    nn.ReLU(),

    nn.Linear(64,1)

)
```

Combined with

```python
optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=0.001,

    weight_decay=1e-5

)
```

this represents a strong baseline model for many descriptor-based materials prediction tasks.

---

# 12.16.19 Which Techniques Should You Use?

General recommendations

| Technique | Recommended? | Typical Usage |
|------------|--------------|---------------|
| Dropout | Yes | Reduce overfitting |
| Batch Normalization | Yes | Stabilize training |
| AdamW | Yes | Default optimizer |
| Weight Decay | Yes | Better generalization |
| Learning Rate Scheduler | Yes | Long training runs |
| Gradient Clipping | Sometimes | Deep or unstable networks |
| Early Stopping | Yes | Prevent overfitting |

Not every project requires every technique,

but understanding when to apply them is an important research skill.

---

# 12.16.20 Summary

Training a neural network involves much more than defining an architecture and selecting an optimizer.

Modern deep learning models rely on several complementary techniques to improve stability and generalization.

In this section, we introduced

- Dropout,
- Batch Normalization,
- Early Stopping,
- Learning Rate Scheduling,
- Weight Decay,
- Gradient Clipping.

Together, these methods make neural networks more reliable, more stable, and better able to generalize to unseen materials.

At this point, we have covered nearly every fundamental component required to build and train fully connected neural networks using PyTorch.

The final section of this chapter will bring everything together by developing a **complete end-to-end materials informatics project**, starting from material descriptors and ending with a trained neural network capable of predicting a real materials property.

# 12.17 Model Evaluation for Neural Networks in Materials Informatics

In the previous sections, we successfully built and trained neural networks using PyTorch. During training, we repeatedly observed the value of the loss function decreasing.

However, an important question remains.

> **How do we know whether our neural network is actually good?**

A low training loss alone does **not** necessarily mean that the model has learned useful scientific relationships.

A model may simply memorize the training data without learning patterns that generalize to new materials.

Therefore, after training a neural network, we must evaluate its performance using independent validation or test data.

For materials informatics, model evaluation is especially important because our ultimate goal is rarely to predict the properties of materials that already exist in our dataset.

Instead, we want to predict the properties of **new materials** that have never been synthesized or simulated before.

The evaluation workflow is

```text
Training Dataset

↓

Train Neural Network

↓

Validation Dataset

↓

Performance Metrics

↓

Interpret Results

↓

Predict New Materials
```

Throughout this section, we will focus on evaluation methods commonly used in computational materials science literature.

---

# 12.17.1 Why Model Evaluation Is Important

Suppose we train a neural network to predict the formation energies of materials contained in the Materials Project database.

If the model memorizes those materials instead of learning the underlying relationship between composition, crystal structure, and formation energy,

it may perform perfectly on the training data while completely failing on new compounds.

For example,

```text
Training MAE

↓

0.01 eV/atom
```

looks impressive.

However,

```text
Validation MAE

↓

0.42 eV/atom
```

indicates that the model has poor predictive ability.

This is why every materials machine learning paper reports evaluation metrics on unseen data.

---

# 12.17.2 Train, Validation, and Test Sets

A typical dataset is divided into three parts.

```text
Entire Dataset

↓

Training Set

↓

Validation Set

↓

Test Set
```

Each serves a different purpose.

### Training Set

Used to update the neural network weights.

```text
Training Data

↓

Backpropagation

↓

Weight Updates
```

---

### Validation Set

Used during development to

- monitor performance,
- detect overfitting,
- tune hyperparameters,
- select the best model.

No weight updates occur during validation.

---

### Test Set

Used only once after training is complete.

The test set provides an unbiased estimate of how well the model will perform on completely unseen materials.

Good practice is

```text
Training

↓

Validation

↓

Final Test
```

The test set should never influence model design.

---

# 12.17.3 Regression Metrics

Most materials informatics problems involve regression.

Examples include predicting

- band gap,
- formation energy,
- bulk modulus,
- elastic constants,
- dielectric constant,
- thermal conductivity,
- battery voltage.

The most common regression metrics are

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

These metrics summarize different aspects of model performance.

---

# 12.17.4 Mean Absolute Error (MAE)

MAE measures the average magnitude of prediction errors.

Conceptually,

```text
Prediction

↓

Difference

↓

Absolute Value

↓

Average
```

Lower MAE indicates more accurate predictions.

For example,

```text
MAE

↓

0.05 eV
```

means that, on average,

the predicted band gaps differ from the reference values by

```text
0.05 eV
```

MAE is widely reported because it is easy to interpret physically.

---

# 12.17.5 Calculating MAE Using scikit-learn

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(

    y_true,

    y_pred

)

print(

    f"MAE: {mae:.4f}"

)
```

Example output

```text
MAE: 0.0725
```

---

# 12.17.6 Mean Squared Error (MSE)

MSE squares each prediction error before averaging.

Conceptually,

```text
Prediction

↓

Difference

↓

Square

↓

Average
```

Because the errors are squared,

large mistakes receive much greater penalties.

MSE is frequently used as the training loss for regression models.

---

# 12.17.7 Computing MSE

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(

    y_true,

    y_pred

)

print(

    f"MSE: {mse:.5f}"

)
```

Example

```text
MSE: 0.01284
```

---

# 12.17.8 Root Mean Squared Error (RMSE)

RMSE is simply the square root of MSE.

An important advantage is that RMSE has the same physical units as the target property.

Suppose we predict

```text
Band Gap
```

measured in

```text
eV
```

Then

```text
RMSE

↓

eV
```

making interpretation much easier.

---

# 12.17.9 Computing RMSE

```python
import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(

    mean_squared_error(

        y_true,

        y_pred

    )

)

print(

    f"RMSE: {rmse:.4f}"

)
```

---

# 12.17.10 Coefficient of Determination (R²)

The R² score measures how well the model explains the variation in the target values.

Typical interpretation

| R² | Interpretation |
|----|----------------|
| 1.0 | Perfect prediction |
| 0.9 | Excellent model |
| 0.8 | Very good |
| 0.5 | Moderate |
| 0.0 | No predictive power |
| < 0 | Worse than predicting the mean |

For materials informatics,

high-quality neural network models often achieve

```text
R²

>

0.90
```

depending on the dataset and property being predicted.

---

# 12.17.11 Computing R²

```python
from sklearn.metrics import r2_score

r2 = r2_score(

    y_true,

    y_pred

)

print(

    f"R²: {r2:.4f}"

)
```

Example

```text
R²: 0.9481
```

---

# 12.17.12 Evaluating a PyTorch Model

After training,

switch the model into evaluation mode.

```python
model.eval()

predictions = []

targets = []

with torch.no_grad():

    for X_batch, y_batch in valid_loader:

        X_batch = X_batch.to(device)

        output = model(

            X_batch

        )

        predictions.extend(

            output.cpu().numpy()

        )

        targets.extend(

            y_batch.numpy()

        )
```

Convert to NumPy arrays.

```python
predictions = np.array(

    predictions

).flatten()

targets = np.array(

    targets

).flatten()
```

Now compute evaluation metrics.

```python
mae = mean_absolute_error(

    targets,

    predictions

)

rmse = np.sqrt(

    mean_squared_error(

        targets,

        predictions

    )

)

r2 = r2_score(

    targets,

    predictions

)

print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
```

---

# 12.17.13 Visualizing Predicted Versus Actual Values

Numbers alone rarely tell the complete story.

One of the most common figures in materials machine learning papers is the **Predicted vs Actual** plot.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(

    targets,

    predictions,

    alpha=0.6

)

plt.plot(

    [

        targets.min(),

        targets.max()

    ],

    [

        targets.min(),

        targets.max()

    ],

    color="red"

)

plt.xlabel(

    "Actual Band Gap (eV)"

)

plt.ylabel(

    "Predicted Band Gap (eV)"

)

plt.title(

    "Predicted vs Actual"

)

plt.show()
```

A well-trained model produces points close to the diagonal line.

Conceptually,

```text
Predicted

^

|

|        *

|      *

|    *

|  *

|*

+------------------------>

Actual
```

---

# 12.17.14 Residual Analysis

Residuals represent the prediction errors.

```text
Residual

=

Prediction

-

Actual
```

Plotting residuals helps identify

- systematic bias,
- outliers,
- regions where the model performs poorly.

```python
residuals = predictions - targets

plt.figure(figsize=(7,5))

plt.scatter(

    targets,

    residuals,

    alpha=0.6

)

plt.axhline(

    0,

    color="red"

)

plt.xlabel(

    "Actual Band Gap (eV)"

)

plt.ylabel(

    "Residual"

)

plt.title(

    "Residual Plot"

)

plt.show()
```

Ideally,

the residuals should be randomly distributed around zero.

---

# 12.17.15 Learning Curves

During training,

we recorded

- training loss,
- validation loss.

Plotting both curves provides valuable insight into model behavior.

```python
plt.figure(figsize=(8,5))

plt.plot(

    training_history,

    label="Training"

)

plt.plot(

    validation_history,

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()
```

Interpretation

Training and validation curves decrease together

```text
↓

Good learning
```

Training decreases while validation increases

```text
↓

Overfitting
```

Both remain large

```text
↓

Underfitting
```

---

# 12.17.16 Evaluating Models for Different Materials Properties

Different properties naturally produce different error magnitudes.

Examples

| Property | Typical Evaluation Metric |
|----------|---------------------------|
| Band Gap | MAE (eV), RMSE (eV), R² |
| Formation Energy | MAE (eV/atom), RMSE |
| Bulk Modulus | MAE (GPa), R² |
| Elastic Modulus | MAE (GPa), R² |
| Thermal Conductivity | MAE, RMSE |
| Battery Voltage | MAE (V), RMSE |

Always report the units together with the metric.

For example,

```text
MAE = 0.08 eV
```

is far more informative than

```text
MAE = 0.08
```

---

# 12.17.17 Comparing Neural Networks with Classical Machine Learning

One of the most common tasks in materials informatics is comparing multiple models.

For example,

| Model | MAE (eV) | RMSE (eV) | R² |
|--------|---------:|----------:|----:|
| Linear Regression | 0.42 | 0.57 | 0.73 |
| Random Forest | 0.16 | 0.24 | 0.94 |
| XGBoost | 0.11 | 0.17 | 0.97 |
| Neural Network | 0.10 | 0.15 | 0.98 |

Such comparisons help determine whether the increased complexity of a neural network is justified.

In many descriptor-based materials datasets, tree-based methods such as XGBoost remain highly competitive, especially for smaller datasets.

---

# 12.17.18 Best Practices for Materials Machine Learning Evaluation

When reporting neural network performance,

always

- evaluate on unseen data,
- report multiple metrics,
- include physical units,
- visualize predictions,
- inspect residuals,
- compare with baseline models,
- discuss scientific significance rather than only numerical accuracy.

Remember that the goal is not simply to achieve the lowest possible error.

The goal is to build a model that can reliably guide materials discovery.

---

# 12.17.19 Summary

Model evaluation is a critical step in every materials informatics workflow.

After training a neural network, we must verify that it can accurately predict the properties of materials it has never encountered before.

In this section, we learned how to

- evaluate neural networks using MAE, MSE, RMSE, and R²,
- generate predictions using PyTorch,
- visualize predicted versus actual values,
- analyze residuals,
- interpret learning curves,
- compare neural networks with classical machine learning models.

These evaluation techniques are standard practice in modern computational materials science and should accompany every neural network model developed for property prediction.

In the next section, we will learn how to **save, load, and deploy trained PyTorch models**, allowing us to reuse neural networks for predicting the properties of entirely new materials without retraining the model.

# 12.18 Saving, Loading, and Deploying Neural Network Models for Materials Informatics

Training a neural network can require minutes, hours, or even days depending on

- the dataset size,
- the model architecture,
- the available hardware.

After successfully training a model, we do **not** want to repeat the entire training process every time we need to make predictions.

Instead, we save the trained model to disk and reload it whenever needed.

For materials informatics, this is particularly important because a trained neural network can be reused to rapidly predict the properties of thousands or even millions of candidate materials.

The deployment workflow is

```text
Train Neural Network

↓

Save Model

↓

Load Model

↓

Predict New Materials

↓

Accelerate Materials Discovery
```

This workflow is widely used in computational materials science research.

---

# 12.18.1 Why Save a Trained Model?

Suppose we trained a neural network to predict the band gaps of semiconductor materials.

Training required

- 100 epochs,
- 25,000 materials,
- several hours of GPU computation.

Once training is complete,

the learned parameters represent valuable scientific knowledge extracted from the dataset.

Saving the model allows us to

- reuse it later,
- share it with collaborators,
- deploy it in scientific software,
- perform high-throughput screening,
- reproduce published results.

---

# 12.18.2 What Does a Neural Network Learn?

Recall that during training,

the optimizer updates

- weights,
- biases.

Conceptually,

```text
Random Parameters

↓

Training

↓

Optimized Parameters

↓

Save to Disk
```

These optimized parameters are what PyTorch saves.

The architecture itself is **not** stored inside the parameter file.

---

# 12.18.3 Understanding `state_dict()`

Every PyTorch model contains a collection of learnable parameters.

PyTorch stores these parameters in a dictionary called the **state dictionary**.

View it using

```python
print(

    model.state_dict()

)
```

Example output

```text
OrderedDict([

('network.0.weight', ...),

('network.0.bias', ...),

('network.2.weight', ...),

('network.2.bias', ...),

...

])
```

The state dictionary contains

- weights,
- biases,
- BatchNorm statistics,
- other learnable parameters.

---

# 12.18.4 Saving Model Parameters

The recommended approach is to save only the state dictionary.

```python
import torch

torch.save(

    model.state_dict(),

    "band_gap_model.pth"

)
```

The file

```text
band_gap_model.pth
```

contains all learned parameters.

Saving only the parameters produces smaller files and greater flexibility.

---

# 12.18.5 Loading a Saved Model

Before loading the parameters,

we must recreate the same model architecture.

```python
model = BandGapNN()
```

Next,

load the saved parameters.

```python
model.load_state_dict(

    torch.load(

        "band_gap_model.pth"

    )

)

model.eval()
```

Calling

```python
model.eval()
```

switches the network into inference mode.

This is especially important when using

- Dropout,
- Batch Normalization.

---

# 12.18.6 Why `model.eval()` Is Important

Training mode

```python
model.train()
```

Evaluation mode

```python
model.eval()
```

During inference,

Dropout should not randomly disable neurons,

and Batch Normalization should use the learned statistics instead of recalculating them.

Therefore,

always execute

```python
model.eval()
```

before making predictions.

---

# 12.18.7 Making Predictions with a Saved Model

Suppose we have descriptors for several new materials.

```python
with torch.no_grad():

    prediction = model(

        X_new

    )
```

Example output

```text
tensor(

[[1.42],

 [2.81],

 [0.93]]

)
```

These values represent the predicted material property,

such as

- band gap (eV),
- formation energy (eV/atom),
- bulk modulus (GPa),

depending on the target used during training.

---

# 12.18.8 Converting Predictions to NumPy

Most scientific Python libraries use NumPy arrays.

Convert the predictions.

```python
prediction = prediction.cpu().numpy()

print(prediction)
```

Example

```text
[[1.42]

 [2.81]

 [0.93]]
```

These values can now be analyzed using

- NumPy,
- pandas,
- matplotlib,
- scikit-learn.

---

# 12.18.9 Predicting a Single Material

Suppose we generated descriptors for one new crystal.

Convert them into a tensor.

```python
new_material = torch.tensor(

    descriptor_vector,

    dtype=torch.float32

).unsqueeze(0)
```

Predict.

```python
model.eval()

with torch.no_grad():

    band_gap = model(

        new_material

    )

print(

    band_gap.item()

)
```

Example output

```text
2.17
```

The predicted band gap is

```text
2.17 eV
```

---

# 12.18.10 High-Throughput Materials Screening

One of the major advantages of machine learning is the ability to screen thousands of materials rapidly.

Workflow

```text
Crystal Database

↓

pymatgen

↓

Descriptor Generation

↓

PyTorch Model

↓

Predicted Properties

↓

Rank Candidate Materials
```

Instead of performing expensive Density Functional Theory calculations for every compound,

the trained neural network can estimate properties in seconds.

---

# 12.18.11 Example: Screening Thousands of Materials

Suppose we generated descriptors for

```text
50,000

candidate materials.
```

Prediction becomes

```python
model.eval()

with torch.no_grad():

    predicted_band_gaps = model(

        X_candidates

    )
```

Every material is evaluated in a single forward pass or in batches, depending on available memory.

---

# 12.18.12 Exporting Predictions

Store predictions in a pandas DataFrame.

```python
import pandas as pd

results = pd.DataFrame(

    {

        "Material": material_names,

        "Predicted Band Gap": predicted_band_gaps.cpu().numpy().flatten()

    }

)
```

Save to a CSV file.

```python
results.to_csv(

    "predicted_band_gaps.csv",

    index=False

)
```

Example

| Material | Predicted Band Gap (eV) |
|----------|------------------------:|
| Si | 1.12 |
| GaAs | 1.43 |
| ZnO | 3.28 |

---

# 12.18.13 Saving Complete Training Checkpoints

Sometimes we want to continue training later.

Instead of saving only the model,

save the optimizer and training progress.

```python
torch.save(

    {

        "epoch": epoch,

        "model_state": model.state_dict(),

        "optimizer_state": optimizer.state_dict(),

        "loss": loss.item()

    },

    "checkpoint.pth"

)
```

This file stores nearly everything needed to resume training.

---

# 12.18.14 Loading a Checkpoint

```python
checkpoint = torch.load(

    "checkpoint.pth"

)

model.load_state_dict(

    checkpoint["model_state"]

)

optimizer.load_state_dict(

    checkpoint["optimizer_state"]

)

start_epoch = checkpoint["epoch"]
```

Training can now continue from the saved epoch instead of restarting.

---

# 12.18.15 Loading Models on Different Devices

Suppose the model was trained on a GPU,

but predictions will be made on a CPU.

```python
checkpoint = torch.load(

    "band_gap_model.pth",

    map_location="cpu"

)

model.load_state_dict(

    checkpoint
)
```

Likewise,

a CPU-trained model can later be loaded onto a GPU if one is available.

This flexibility makes PyTorch models portable across different computing environments.

---

# 12.18.16 Organizing a Research Project

A well-organized project directory might look like

```text
BandGapProject/

│

├── data/

│     descriptors.csv

│

├── models/

│     best_model.pth

│     checkpoint.pth

│

├── predictions/

│     predicted_band_gaps.csv

│

├── train.py

├── predict.py

├── evaluate.py

└── notebook.ipynb
```

Maintaining a clear project structure improves reproducibility and collaboration.

---

# 12.18.17 Complete Inference Script

The following script loads a trained model and predicts the band gaps of new materials.

```python
import torch

model = BandGapNN()

model.load_state_dict(

    torch.load(

        "band_gap_model.pth"

    )

)

model.eval()

with torch.no_grad():

    predictions = model(

        X_new

    )

print(

    predictions
)
```

Only a few lines of code are required because all training has already been completed.

---

# 12.18.18 Integrating the Model into a Materials Informatics Workflow

A realistic workflow for descriptor-based materials informatics is

```text
Crystal Structure

↓

Quantum ESPRESSO

↓

Relaxed Structure

↓

pymatgen

↓

Crystal Object

↓

matminer

↓

Descriptor Vector

↓

Saved PyTorch Model

↓

Predicted Property

↓

Materials Ranking

↓

Experimental Validation
```

The trained neural network becomes one component of a much larger computational pipeline.

---

# 12.18.19 Best Practices

When saving PyTorch models,

always

- save the `state_dict()` rather than the entire model,
- switch to `model.eval()` before inference,
- use `torch.no_grad()` during prediction,
- save checkpoints during long training runs,
- maintain versioned model files,
- record the descriptor generation procedure,
- document the dataset and preprocessing steps.

Reproducibility is a fundamental requirement of scientific computing.

---

# 12.18.20 Summary

Saving and loading trained neural networks is an essential step in every materials informatics project.

Rather than retraining a model each time,

we can preserve its learned parameters and reuse them to predict the properties of new materials quickly and efficiently.

In this section, we learned how to

- save neural network parameters using `state_dict()`,
- reload trained models,
- perform inference,
- export predictions,
- save complete training checkpoints,
- continue interrupted training,
- deploy neural networks within descriptor-based materials informatics workflows.

At this stage, you have acquired the practical skills needed to build, train, evaluate, save, and deploy fully connected neural networks for materials property prediction.

In the next section, we will examine **how neural networks are used in modern materials informatics research**, including applications to band gap prediction, formation energy estimation, elastic properties, battery materials, and high-throughput materials discovery.


# 12.19 Neural Networks in Materials Informatics

In the previous sections, we developed the theoretical and practical foundations of neural networks using PyTorch. We learned how to

- build neural network architectures,
- define loss functions,
- train models using backpropagation,
- optimize parameters,
- evaluate performance,
- save trained models.

However, an important question remains.

> **How are neural networks actually used in modern materials science research?**

The answer depends on the scientific problem.

Unlike many introductory machine learning examples that focus on images or text, materials informatics involves learning relationships between a material's composition, crystal structure, processing conditions, and physical properties.

Neural networks have become powerful tools because they can model highly nonlinear relationships that are difficult to capture using traditional empirical equations.

Throughout this section, we will examine several important applications of neural networks in materials informatics and show how they fit into a complete computational materials science workflow.

---

# 12.19.1 The Role of Neural Networks in Materials Discovery

Traditional materials discovery often follows a slow and expensive process.

```text
Design Material

↓

Synthesize

↓

Characterize

↓

Measure Properties

↓

Modify Composition

↓

Repeat
```

Each iteration may require

- days,
- weeks,
- or even months.

Computational methods such as Density Functional Theory (DFT) significantly reduce experimental effort but remain computationally expensive for very large material spaces.

Neural networks provide another level of acceleration.

Instead of performing expensive quantum mechanical calculations for every candidate material, a trained neural network can rapidly estimate important properties.

The modern workflow becomes

```text
Known Materials

↓

DFT Calculations

↓

Materials Database

↓

Neural Network Training

↓

Predict Properties of New Materials

↓

Select Best Candidates

↓

Experimental Validation
```

The neural network does not replace physics.

Instead, it learns from physics-based simulations and experimental measurements to make rapid predictions for previously unseen materials.

---

# 12.19.2 Descriptor-Based Neural Networks

The neural networks studied in this chapter belong to the category of **descriptor-based neural networks**.

Their input consists of numerical features extracted from crystal structures.

Typical workflow

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

Descriptors

↓

Neural Network

↓

Property Prediction
```

The descriptors may include

- average atomic number,
- average electronegativity,
- atomic radius,
- packing fraction,
- density,
- oxidation states,
- structural fingerprints,
- orbital features,
- compositional statistics.

A single material may be represented by

```text
50

↓

200

↓

500

↓

1000+

descriptors
```

These numerical features become the input to the neural network.

---

# 12.19.3 Band Gap Prediction

One of the most common applications of machine learning in materials science is predicting the electronic band gap.

The band gap determines whether a material behaves as

- a metal,
- a semiconductor,
- or an insulator.

Accurate band gap prediction is important for

- solar cells,
- LEDs,
- photodetectors,
- transparent conductors,
- semiconductor devices.

Traditional workflow

```text
Crystal Structure

↓

DFT Band Structure Calculation

↓

Band Gap
```

Machine learning workflow

```text
Crystal Structure

↓

Descriptors

↓

Neural Network

↓

Predicted Band Gap
```

Once trained,

the neural network can estimate band gaps for thousands of candidate materials in only a few seconds.

---

# 12.19.4 Example: Band Gap Prediction Pipeline

Suppose we obtained relaxed crystal structures from Quantum ESPRESSO.

The workflow might be

```text
Quantum ESPRESSO

↓

Relaxed Structure

↓

pymatgen

↓

Structure Object

↓

matminer

↓

145 Descriptors

↓

PyTorch Neural Network

↓

Predicted Band Gap
```

Example implementation

```python
model.eval()

with torch.no_grad():

    predicted_band_gap = model(

        descriptor_tensor

    )

print(

    f"Predicted Band Gap: {predicted_band_gap.item():.3f} eV"

)
```

The predicted value can then be compared with DFT or experimental measurements.

---

# 12.19.5 Formation Energy Prediction

Another important property is the **formation energy**.

Formation energy measures the thermodynamic stability of a material.

Stable materials generally have lower formation energies.

Machine learning workflow

```text
Crystal Structure

↓

Descriptors

↓

Neural Network

↓

Formation Energy
```

Applications include

- discovering stable compounds,
- alloy design,
- high-throughput screening,
- identifying synthesizable materials.

Training code is identical to the band gap example.

Only the target variable changes.

```python
y = df["formation_energy"]
```

Everything else in the PyTorch pipeline remains unchanged.

---

# 12.19.6 Elastic Property Prediction

Mechanical properties are critical for structural materials.

Examples include

- Young's modulus,
- bulk modulus,
- shear modulus,
- Poisson's ratio.

Workflow

```text
Crystal Structure

↓

Descriptors

↓

Neural Network

↓

Elastic Modulus
```

Instead of performing expensive stress-strain DFT calculations,

a trained neural network can estimate elastic properties almost instantly.

---

# 12.19.7 Predicting Thermal Properties

Neural networks are also used to predict

- thermal conductivity,
- heat capacity,
- thermal expansion,
- Debye temperature.

These properties influence

- thermoelectric materials,
- thermal barrier coatings,
- electronic cooling,
- aerospace materials.

Example workflow

```text
Descriptors

↓

Neural Network

↓

Thermal Conductivity
```

Regression techniques remain exactly the same.

Only the target property differs.

---

# 12.19.8 Battery Materials

Machine learning has become increasingly important in battery research.

Important prediction targets include

- intercalation voltage,
- ionic conductivity,
- diffusion barriers,
- specific capacity,
- cycle stability.

Workflow

```text
Crystal Structure

↓

Descriptors

↓

Neural Network

↓

Predicted Voltage
```

Candidate materials with promising predictions can then undergo more expensive DFT calculations and experimental testing.

---

# 12.19.9 Catalyst Discovery

Catalyst development often requires evaluating enormous chemical spaces.

Neural networks help predict

- adsorption energies,
- reaction barriers,
- catalytic activity,
- selectivity,
- stability.

Workflow

```text
Surface Structure

↓

Descriptors

↓

Neural Network

↓

Catalytic Property
```

Machine learning enables researchers to narrow the search space before performing detailed first-principles calculations.

---

# 12.19.10 High-Entropy Alloys

High-entropy alloys contain multiple principal elements,

making traditional empirical design difficult.

Neural networks can predict

- phase stability,
- hardness,
- yield strength,
- corrosion resistance.

Example workflow

```text
Composition

↓

Descriptors

↓

Neural Network

↓

Mechanical Properties
```

Such models assist researchers in exploring extremely large compositional spaces.

---

# 12.19.11 Metal-Organic Frameworks (MOFs)

Metal-organic frameworks are widely studied for

- gas storage,
- gas separation,
- catalysis,
- carbon capture.

Neural networks can predict

- surface area,
- pore volume,
- gas adsorption capacity,
- adsorption selectivity.

Workflow

```text
MOF Structure

↓

Descriptors

↓

Neural Network

↓

Gas Uptake Prediction
```

---

# 12.19.12 Two-Dimensional Materials

The discovery of graphene stimulated enormous interest in two-dimensional materials.

Examples include

- transition metal dichalcogenides,
- MXenes,
- hexagonal boron nitride.

Neural networks can predict

- band gap,
- carrier mobility,
- magnetic behavior,
- optical absorption.

Such predictions guide computational screening before expensive simulations.

---

# 12.19.13 Multi-Property Prediction

Instead of predicting only one property,

a neural network may predict several properties simultaneously.

Example

```text
Crystal Structure

↓

Descriptors

↓

Shared Neural Network

↓

Band Gap

Formation Energy

Bulk Modulus

Density
```

This approach is known as **multi-task learning**.

Example architecture

```python
self.output = nn.Linear(

    128,

    4

)
```

Each output neuron predicts a different material property.

---

# 12.19.14 A Complete Descriptor-Based Materials Informatics Pipeline

The knowledge developed throughout this book now forms a complete workflow.

```text
Quantum ESPRESSO

↓

Crystal Structure

↓

pymatgen

↓

Structure Analysis

↓

matminer

↓

Descriptor Generation

↓

Feature Matrix

↓

PyTorch Neural Network

↓

Training

↓

Model Evaluation

↓

Property Prediction

↓

Candidate Ranking
```

Every stage plays an important role.

Notice that PyTorch is only one component of a much larger materials informatics ecosystem.

---

# 12.19.15 When Should We Use Neural Networks?

Neural networks perform best when

- datasets are relatively large,
- nonlinear relationships exist,
- sufficient computational resources are available,
- feature engineering captures meaningful physics.

For small descriptor-based datasets,

tree-based methods such as Random Forest and XGBoost often remain competitive.

As the dataset size increases,

neural networks frequently become more effective because they can learn increasingly complex nonlinear relationships.

Choosing the appropriate model depends on the scientific problem rather than simply selecting the most sophisticated algorithm.

---

# 12.19.16 Current Limitations

Although fully connected neural networks are powerful,

they still depend heavily on manually engineered descriptors.

Consider the workflow

```text
Crystal Structure

↓

Hundreds of Handcrafted Descriptors

↓

Neural Network
```

An important question arises.

> **Are these descriptors preserving all the structural information contained within the crystal?**

Unfortunately,

the answer is often **no**.

Descriptor engineering inevitably compresses a complex three-dimensional crystal into a fixed-length numerical vector.

Some information is preserved,

while some is inevitably lost.

This limitation motivates the development of more advanced deep learning methods.

---

# 12.19.17 Summary

Neural networks have become an important tool for modern materials informatics.

They enable researchers to learn complex nonlinear relationships between material descriptors and physical properties, making it possible to rapidly screen vast numbers of candidate materials before performing expensive first-principles calculations or experiments.

In this section, we examined how descriptor-based neural networks are applied to

- band gap prediction,
- formation energy estimation,
- elastic property prediction,
- thermal property prediction,
- battery materials,
- catalysts,
- high-entropy alloys,
- metal-organic frameworks,
- two-dimensional materials.

Despite their success, these models still rely on manually engineered descriptors generated using tools such as **pymatgen** and **matminer**.

This naturally leads to an important question.

> **Can a neural network learn directly from the crystal structure itself instead of relying on handcrafted descriptors?**

The answer to this question marks one of the most significant developments in modern materials informatics.

In the next section, we will explore why fully connected neural networks have fundamental limitations for crystalline materials and how these limitations motivated the development of **Graph Neural Networks (GNNs)**.


# 12.20 Limitations of Fully Connected Neural Networks and the Need for Graph Neural Networks

Throughout this chapter, we have developed neural network models that accept numerical descriptors as input and predict various materials properties.

These models have demonstrated remarkable capabilities for learning complex nonlinear relationships between material descriptors and target properties such as

- band gap,
- formation energy,
- elastic modulus,
- thermal conductivity,
- battery voltage.

However, despite their success, fully connected neural networks possess several fundamental limitations when applied to crystalline materials.

Understanding these limitations is essential because they motivate one of the most important developments in modern materials informatics:

> **Graph Neural Networks (GNNs).**

This section explains why descriptor-based neural networks eventually reach their limits and why representing crystals as graphs provides a more natural and physically meaningful approach.

---

# 12.20.1 A Crystal Is More Than a List of Numbers

Throughout this chapter, every neural network received an input similar to

```text
Density

Volume

Average Atomic Mass

Mean Electronegativity

Packing Fraction

Average Covalent Radius

...

↓

145 Numerical Features
```

From the perspective of the neural network,

this input is simply a vector of numbers.

```text
[0.42, 7.31, 2.18, 5.94, ...]
```

The model has no direct knowledge that these numbers originated from

- atoms,
- chemical bonds,
- crystal symmetry,
- atomic neighborhoods,
- periodic structures.

It only sees numerical values.

Consequently,

much of the structural information present in the original crystal has already been compressed before the neural network begins learning.

---

# 12.20.2 Descriptor Engineering Depends on Human Expertise

Descriptor-based machine learning requires researchers to decide

- which descriptors to calculate,
- which descriptors to remove,
- which descriptors to normalize,
- which descriptors are scientifically meaningful.

The workflow becomes

```text
Crystal Structure

↓

Researcher's Knowledge

↓

Feature Engineering

↓

Descriptor Matrix

↓

Neural Network
```

If important descriptors are omitted,

the neural network cannot recover the missing information.

Its predictive ability is limited by the quality of the engineered features.

This dependency on manual feature engineering is one of the largest limitations of classical machine learning methods.

---

# 12.20.3 Information Is Lost During Feature Engineering

Consider two crystal structures.

Although they may possess different atomic arrangements,

their calculated descriptor vectors may appear very similar.

```text
Crystal A

↓

Descriptor Vector

↓

[2.31, 5.12, 0.74, ...]
```

```text
Crystal B

↓

Descriptor Vector

↓

[2.29, 5.15, 0.76, ...]
```

The descriptor vectors differ only slightly,

even though the underlying crystal structures may exhibit significantly different bonding environments.

Once structural information has been compressed into descriptors,

the original atomic arrangement cannot generally be reconstructed.

Some information has been permanently lost.

---

# 12.20.4 Crystal Structures Are Naturally Three-Dimensional

Materials are inherently three-dimensional.

Atoms occupy specific positions within a crystal lattice.

Neighboring atoms interact through chemical bonds and local coordination environments.

Conceptually,

```text
Atom

↓

Neighboring Atoms

↓

Bond Network

↓

Entire Crystal
```

A descriptor vector cannot explicitly preserve every atomic relationship within such a complex structure.

Instead,

it summarizes the structure using statistical quantities.

While these summaries are often useful,

they cannot completely represent the richness of the crystal.

---

# 12.20.5 Atomic Connectivity Is Ignored

Suppose we examine a silicon crystal.

Conceptually,

```text
Si

↓

Connected to

↓

Four Neighboring Si Atoms
```

The arrangement of neighboring atoms strongly influences

- electronic structure,
- mechanical properties,
- thermal transport,
- diffusion behavior.

However,

descriptor-based neural networks receive only numerical summaries.

They do not explicitly know

- which atoms are connected,
- how many neighbors each atom has,
- the distances between neighboring atoms,
- the geometry of the local environment.

These relationships must be approximated indirectly through engineered descriptors.

---

# 12.20.6 Local Atomic Environments Are Extremely Important

Many materials properties depend primarily on the local atomic environment rather than the overall chemical composition.

For example,

changing only one atom in a crystal may

- alter the local coordination,
- modify chemical bonding,
- introduce defects,
- change electronic properties.

Descriptor-based models often struggle to capture these subtle local effects.

Graph-based methods, however,

can naturally represent local neighborhoods because they explicitly store atomic connectivity.

---

# 12.20.7 Crystals Have Variable Numbers of Atoms

Different crystal structures contain different numbers of atoms.

For example,

```text
Silicon Unit Cell

↓

2 Atoms
```

```text
Perovskite

↓

5 Atoms
```

```text
Complex Oxide

↓

40 Atoms
```

A fully connected neural network expects a fixed-length input vector.

Regardless of whether the crystal contains

- 2 atoms,
- 20 atoms,
- or 200 atoms,

the descriptor vector must always contain exactly the same number of features.

Creating fixed-length representations for variable-sized crystal structures is challenging and often requires substantial feature engineering.

---

# 12.20.8 Symmetry Information Is Difficult to Preserve

Crystal symmetry plays a fundamental role in materials science.

Properties such as

- electronic band structure,
- phonon dispersion,
- elastic behavior,
- optical response,

are strongly influenced by crystal symmetry.

Although descriptors can encode some symmetry information,

they rarely capture it completely.

Consequently,

descriptor-based neural networks may overlook important structural relationships that arise from crystal symmetry.

---

# 12.20.9 Long-Range Atomic Interactions

Some materials properties depend not only on immediate neighbors but also on atoms located farther away.

Examples include

- electrostatic interactions,
- magnetic ordering,
- long-range strain,
- polarization.

Representing these interactions using handcrafted descriptors is difficult.

As the complexity of the material increases,

the number of required descriptors also increases.

---

# 12.20.10 Descriptor Engineering Becomes Increasingly Difficult

As researchers attempt to improve prediction accuracy,

they often calculate increasingly large descriptor sets.

Typical workflow

```text
Crystal

↓

50 Descriptors

↓

Prediction
```

Later,

```text
Crystal

↓

150 Descriptors

↓

Prediction
```

Eventually,

```text
Crystal

↓

800 Descriptors

↓

Prediction
```

Adding more descriptors increases

- computational cost,
- memory requirements,
- feature redundancy,
- preprocessing complexity.

More descriptors do not necessarily produce better models.

---

# 12.20.11 The Neural Network Learns Only After Feature Generation

Notice where learning begins.

```text
Crystal Structure

↓

Descriptor Generation

↓

Neural Network Learns
```

Everything before descriptor generation is fixed.

The neural network cannot improve the descriptors themselves.

It can only learn relationships between the provided features and the target property.

This limitation prevents the model from discovering new structural representations automatically.

---

# 12.20.12 What Would an Ideal Model Do?

Instead of relying on handcrafted descriptors,

an ideal model would learn directly from the crystal structure.

Conceptually,

```text
Crystal Structure

↓

Neural Network

↓

Learn Structural Representation

↓

Predict Property
```

Such a model would automatically determine

- which atoms are important,
- which bonds matter,
- which local environments influence the target property.

No manual descriptor engineering would be required.

---

# 12.20.13 Representing Crystals as Graphs

Fortunately,

crystal structures naturally resemble graphs.

A graph consists of

- nodes,
- edges.

For crystalline materials,

the correspondence is straightforward.

```text
Atoms

↓

Nodes
```

```text
Chemical Bonds

↓

Edges
```

For example,

```text
Si —— Si

│      │

Si —— Si
```

can be represented as

```text
Nodes

↓

Silicon Atoms

Edges

↓

Neighbor Relationships
```

This representation preserves the connectivity between atoms instead of reducing the crystal to a fixed-length descriptor vector.

---

# 12.20.14 Learning Directly from Crystal Structures

Using graph representations,

the workflow changes fundamentally.

Descriptor-based workflow

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

Descriptors

↓

Neural Network

↓

Prediction
```

Graph-based workflow

```text
Crystal Structure

↓

pymatgen

↓

Crystal Graph

↓

Graph Neural Network

↓

Prediction
```

Instead of manually defining hundreds of descriptors,

the Graph Neural Network learns useful structural representations during training.

---

# 12.20.15 Advantages of Graph-Based Learning

Compared with fully connected neural networks,

Graph Neural Networks offer several important advantages.

They can

- preserve atomic connectivity,
- represent local environments,
- process crystals containing different numbers of atoms,
- learn structural features automatically,
- reduce dependence on handcrafted descriptors,
- better capture the physics of crystalline materials.

These advantages have made graph neural networks the dominant deep learning approach for many materials science applications.

---

# 12.20.16 From Classical Deep Learning to Graph Neural Networks

This chapter has focused on fully connected neural networks because they introduce the fundamental concepts of deep learning.

We learned about

- neurons,
- activation functions,
- forward propagation,
- loss functions,
- backpropagation,
- optimization,
- model evaluation,
- PyTorch implementation.

These concepts remain essential.

Graph Neural Networks use exactly the same training principles.

The primary difference lies in how the input data are represented.

Instead of numerical descriptor vectors,

they operate directly on graphs constructed from crystal structures.

---

# 12.20.17 Transition to the Next Chapter

We are now ready to move beyond descriptor-based deep learning.

The next major question is

> **How can a neural network learn directly from atoms and their bonding relationships without relying on handcrafted descriptors?**

The answer lies in **Graph Neural Networks (GNNs)**.

Rather than treating a crystal as a fixed-length numerical vector,

GNNs treat it as a network of interconnected atoms.

Information flows through this network as neighboring atoms exchange messages, allowing the model to learn rich structural representations that are closely aligned with the underlying physics of crystalline materials.

This approach has transformed modern materials informatics and forms the foundation of many state-of-the-art models, including

- Crystal Graph Convolutional Neural Networks (CGCNN),
- Materials Graph Networks (MEGNet),
- M3GNet,
- Atomistic Line Graph Neural Networks (ALIGNN).

These models have significantly advanced the prediction of materials properties and accelerated computational materials discovery.

---

# 12.20.18 Chapter Summary

In this chapter, we built a complete foundation in neural networks for materials informatics.

Beginning with the biological inspiration for artificial neurons, we developed increasingly sophisticated concepts, including multilayer neural networks, activation functions, forward propagation, loss functions, gradient descent, backpropagation, optimization, and practical implementation using PyTorch.

We then learned how to

- prepare materials datasets,
- build neural network architectures,
- train and evaluate models,
- improve model performance,
- save and deploy trained models,
- apply neural networks to real materials science problems.

Finally, we examined the limitations of descriptor-based neural networks and recognized that crystalline materials possess rich structural information that is difficult to capture using handcrafted descriptors alone.

This realization motivates the next stage of our journey.

In **Chapter 13**, we will replace fixed-length descriptor vectors with **crystal graphs**, allowing neural networks to learn directly from atomic structures. This transition marks one of the most important advances in modern materials informatics and opens the door to state-of-the-art graph-based deep learning methods for materials discovery.

