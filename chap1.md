
# Chapter 1
# Introduction to Machine Learning

---

# Learning Objectives

After completing this chapter, you will be able to

- define machine learning precisely,
- explain the difference between traditional programming and machine learning,
- distinguish artificial intelligence, machine learning, and deep learning,
- understand different categories of machine learning,
- explain supervised, unsupervised, semi-supervised, and reinforcement learning,
- understand datasets, samples, features, and target variables,
- explain the complete machine learning workflow,
- understand training, validation, and testing,
- implement your first machine learning model in Python using Scikit-learn,
- interpret predictions made by a machine learning model.

By the end of this chapter, you will understand **how machines learn from data** and will write your first complete machine learning program.

---

# 1.1 Why Do We Need Machine Learning?

Imagine you are a materials scientist searching for a new alloy suitable for aerospace applications.

Suppose your laboratory has measured the properties of **100 materials**.

Finding useful relationships among these materials is manageable.

Now imagine that instead of 100 materials, you have

- 100,000 materials,
- 500 measured properties for each material,
- experimental data collected over several decades.

Could a human examine every possible relationship?

Almost certainly not.

The number of possible combinations is simply too large.

Modern materials databases such as those generated through high-throughput experiments and computational simulations contain millions of measurements.

Hidden within these measurements are relationships that can lead to

- stronger alloys,
- better battery materials,
- improved semiconductors,
- more efficient solar cells,
- lighter structural materials,
- high-temperature superconductors.

Finding these relationships manually is practically impossible.

This is where machine learning becomes indispensable.

Rather than programming explicit rules,

we allow the computer to discover patterns directly from data.

Machine learning enables computers to analyze vast quantities of information, identify complex relationships, and make predictions that would be difficult or impossible for humans to derive manually.

---

# 1.2 A Historical Perspective

For many decades, computer programs relied entirely on manually written rules.

If we wanted a program to perform a task, we had to tell it exactly what to do.

For example,

suppose we wanted to classify a material as "high density."

A traditional program might contain rules such as

```text
IF density > 7.5

THEN High Density

ELSE Low Density
```

This approach works for simple problems.

However, many scientific relationships are far more complicated.

For example,

predicting

- fracture toughness,
- fatigue life,
- corrosion resistance,
- superconducting transition temperature,

cannot usually be expressed using a few simple rules.

Instead,

these properties depend upon many interacting variables.

Machine learning allows computers to discover these relationships automatically.

Instead of receiving rules,

the computer receives examples.

From these examples,

it learns the underlying patterns.

---

# 1.3 A Simple Analogy

Imagine a child learning to recognize cats.

The child is not given a mathematical definition of a cat.

Instead,

the child sees hundreds of examples.

Over time,

the brain learns common characteristics.

Eventually,

the child can recognize a new cat that has never been seen before.

Machine learning operates similarly.

Instead of being programmed with explicit rules,

the algorithm studies examples and gradually learns patterns.

Later,

when presented with new data,

it applies those learned patterns to make predictions.

This ability to generalize from previous experience is the defining characteristic of machine learning.

---

# 1.4 The Formal Definition of Machine Learning

One widely accepted definition was proposed by computer scientist **Tom M. Mitchell** in 1997:

> "A computer program is said to learn from experience **E** with respect to some task **T** and performance measure **P**, if its performance at task **T**, as measured by **P**, improves with experience **E**."

Although this definition may appear abstract, each component has a precise meaning.

- **Task (T)** — the problem we want to solve.
- **Experience (E)** — the data used for learning.
- **Performance (P)** — the metric used to evaluate success.

For example, suppose we wish to predict the hardness of metallic alloys.

- **Task:** Predict hardness.
- **Experience:** Previously measured alloy data.
- **Performance:** Prediction error on unseen alloys.

As the model receives more high-quality training data, its predictive performance generally improves.

This improvement through experience is the essence of machine learning.

---

# 1.5 Machine Learning in Materials Informatics

Machine learning has transformed materials science.

Instead of relying solely on expensive experiments or computationally intensive simulations, researchers increasingly use machine learning to accelerate materials discovery.

Some applications include

- predicting crystal structures,
- estimating elastic constants,
- discovering battery electrode materials,
- predicting corrosion resistance,
- identifying catalysts,
- optimizing additive manufacturing parameters,
- forecasting mechanical properties,
- predicting thermal conductivity,
- estimating electronic bandgaps,
- screening millions of hypothetical compounds.

Rather than replacing experiments,

machine learning helps researchers decide **which experiments are most promising**.

Consequently,

research time and cost can be reduced dramatically.

---


In the next section, we will study the relationship between **Artificial Intelligence, Machine Learning, Data Science, and Deep Learning**, clarifying one of the most commonly misunderstood topics in modern computing.
# 1.6 Artificial Intelligence, Machine Learning, Data Science, and Deep Learning

Terms such as

- Artificial Intelligence (AI),
- Machine Learning (ML),
- Deep Learning (DL),
- Data Science,

are frequently used in books, research papers, news articles, and social media.

These terms are closely related, but **they do not mean the same thing**.

Understanding the relationship between them is essential before studying machine learning algorithms.

Many beginners incorrectly assume that AI, machine learning, and deep learning are interchangeable.

They are not.

This section explains how these fields differ and how they overlap.

---

# 1.7 Artificial Intelligence (AI)

Artificial Intelligence is the broadest field.

Its goal is to develop machines capable of performing tasks that normally require human intelligence.

Examples include

- understanding language,
- recognizing images,
- planning actions,
- solving problems,
- making decisions,
- playing games,
- controlling robots.

Notice that this definition says nothing about **how** these tasks are accomplished.

A system can be considered artificial intelligence even if it does not use machine learning.

For example, an expert system that follows thousands of manually written rules is an AI system, even though it never learns from data.

Therefore,

**Artificial Intelligence is the science of creating intelligent systems.**

Machine learning is only one approach to achieving this goal.

---

# 1.8 Machine Learning (ML)

Machine Learning is a subfield of Artificial Intelligence.

Instead of programming every rule manually,

we provide examples,

and the computer learns patterns from those examples.

For instance,

suppose we want to predict the hardness of a material.

Instead of writing hundreds of physical rules,

we collect experimental measurements.

The algorithm studies these measurements and learns the relationship between

- composition,
- processing conditions,
- microstructure,
- hardness.

Later,

it predicts the hardness of materials it has never seen before.

This ability to improve through experience distinguishes machine learning from traditional programming.

---

# 1.9 Deep Learning (DL)

Deep Learning is a specialized branch of machine learning.

Instead of using relatively simple models,

deep learning uses **artificial neural networks** containing many layers.

These networks are inspired by the structure of the human brain,

although they are much simpler than biological neurons.

Deep learning excels at problems involving

- images,
- speech,
- natural language,
- video,
- complex scientific data.

Examples include

- recognizing defects in microscope images,
- identifying phases from diffraction patterns,
- predicting crystal structures,
- analyzing electron microscopy images.

Deep learning usually requires

- very large datasets,
- powerful computers,
- graphics processing units (GPUs),
- longer training times.

For many materials science problems with limited experimental data,

classical machine learning methods often perform just as well or even better.

---

# 1.10 Data Science

Data Science is broader than machine learning.

A data scientist does much more than train models.

Typical responsibilities include

- collecting data,
- cleaning data,
- organizing databases,
- visualizing information,
- performing statistical analysis,
- building machine learning models,
- communicating results.

In practice,

a large portion of a data scientist's time is spent preparing data rather than developing algorithms.

Machine learning is therefore one important tool within the larger discipline of data science.

---

# 1.11 Relationship Among the Fields

The relationship among these disciplines can be represented conceptually as

```text
Artificial Intelligence
│
├── Rule-Based Systems
│
├── Robotics
│
├── Expert Systems
│
└── Machine Learning
      │
      ├── Linear Regression
      ├── Decision Trees
      ├── Random Forests
      ├── Support Vector Machines
      ├── XGBoost
      └── Deep Learning
            │
            ├── Convolutional Neural Networks
            ├── Recurrent Neural Networks
            └── Transformers
```

Notice that

- Deep Learning is part of Machine Learning.
- Machine Learning is part of Artificial Intelligence.
- Artificial Intelligence contains many techniques besides Machine Learning.

---

# 1.12 Which Field Does This Book Cover?

This book focuses primarily on **machine learning for materials informatics**.

You will learn

- Linear Regression,
- Decision Trees,
- Random Forests,
- Gradient Boosting,
- XGBoost,
- LightGBM,
- CatBoost,
- Support Vector Machines,
- K-Nearest Neighbors,
- Clustering,
- Dimensionality Reduction,
- Model Evaluation,
- Feature Engineering,
- Hyperparameter Optimization.

Later chapters also introduce deep learning concepts relevant to materials science.

However,

a strong understanding of classical machine learning is essential before studying deep learning.

---

# 1.13 A Real Materials Science Example

Suppose researchers want to discover a new lightweight alloy for aerospace applications.

Different approaches might be

**Traditional Programming**

A programmer writes hundreds of fixed rules based on known physical relationships.

**Machine Learning**

A model learns from thousands of previously tested alloys and predicts promising new compositions.

**Deep Learning**

A neural network learns directly from large datasets containing compositions, crystal structures, and processing histories, discovering complex patterns automatically.

Each approach aims to solve the same problem,

but they use different methods.

---

# 1.14 Common Misconceptions

Many beginners have incorrect ideas about AI and machine learning.

Here are some common misconceptions.

### Misconception 1

**"AI and Machine Learning are the same."**

Incorrect.

Machine learning is only one branch of AI.

---

### Misconception 2

**"Deep Learning is completely different from Machine Learning."**

Incorrect.

Deep learning is a specialized type of machine learning.

---

### Misconception 3

**"Machine learning always outperforms traditional methods."**

Incorrect.

For small datasets or simple relationships,

classical statistical models or physics-based approaches may perform better.

The best method depends on the problem.

---

### Misconception 4

**"Deep learning is always better than classical machine learning."**

Incorrect.

Deep learning often requires large datasets and significant computational resources.

For many materials science datasets, algorithms such as Random Forests or XGBoost can achieve excellent performance with much less data.

---

# 1.15 Why This Distinction Matters

As you read research papers,

you will frequently encounter statements such as

> "Artificial intelligence was used to accelerate materials discovery."

In many cases,

the actual method employed is machine learning.

Similarly,

papers may claim to use "AI"

when they specifically use deep learning.

Understanding the hierarchy among these fields allows you to interpret scientific literature accurately.

---

In the next section, we will compare **traditional programming and machine learning** in detail, showing exactly how a computer transitions from following manually written rules to learning directly from data.
# 1.16 Traditional Programming vs Machine Learning

To truly understand machine learning, we must first understand how it differs from traditional programming.

Both approaches use computers to solve problems, but the way they solve problems is fundamentally different.

Traditional programming depends on humans writing explicit instructions.

Machine learning allows computers to discover patterns from examples.

---

# 1.17 Traditional Programming

In traditional programming, a human programmer analyzes a problem and writes a set of rules.

The computer simply follows these instructions.

The basic structure is:

```text
Rules + Data

↓

Computer Program

↓

Output
```

The programmer provides both

- the data,
- the logic required to process the data.

---

# 1.18 Example of Traditional Programming

Suppose we want to classify materials as high density or low density.

A simple rule might be:

```text
IF density > 5 g/cm³

THEN classify as High Density

ELSE classify as Low Density
```

The programmer has already decided the rule.

The computer does not learn anything.

It only executes the instructions.

For example:

Input:

```text
Material: Aluminum

Density: 2.70 g/cm³
```

The program checks:

```text
2.70 > 5 ?
```

The answer is false.

Therefore:

```text
Classification = Low Density
```

The computer followed a human-defined rule.

---

# 1.19 Limitations of Traditional Programming

Traditional programming works extremely well when the rules are known and simple.

Examples include

- calculators,
- accounting software,
- sorting algorithms,
- database systems.

However, many scientific problems do not have simple rules.

Consider predicting the mechanical strength of an alloy.

Strength depends on many factors:

- chemical composition,
- crystal structure,
- grain size,
- heat treatment,
- defects,
- processing history.

Writing every possible rule manually would be nearly impossible.

The number of interactions between variables becomes enormous.

This is where machine learning becomes useful.

---

# 1.20 Machine Learning Approach

Machine learning changes the direction of the problem.

Instead of humans writing rules,

we provide examples.

The algorithm discovers the rules automatically.

The structure becomes:

```text
Data + Known Answers

↓

Machine Learning Algorithm

↓

Learned Model
```

The learned model can then make predictions for new data.

---

# 1.21 Example of Machine Learning

Suppose we have experimental alloy data.

| Alloy | Carbon (%) | Chromium (%) | Heat Treatment | Hardness |
|------|------------|--------------|----------------|----------|
| A | 0.2 | 5 | Annealed | 180 |
| B | 0.5 | 10 | Quenched | 450 |
| C | 0.8 | 12 | Tempered | 620 |

The machine learning algorithm receives

## Features (Input)

- Carbon content
- Chromium content
- Processing condition

## Target (Output)

- Hardness

The algorithm studies many examples and learns the relationship.

Later, if we provide a new alloy composition,

the model can estimate its hardness.

---

# 1.22 The Difference in Learning

The fundamental difference is:

## Traditional Programming

Human:

> "Here are the rules. Apply them."

Computer:

> "I will execute the rules."

---

## Machine Learning

Human:

> "Here are examples. Find the hidden patterns."

Computer:

> "I will learn the relationship between inputs and outputs."

---

# 1.23 Mathematical View

Traditional programming can be represented as:

\[
Output = Rules(Input)
\]

The programmer creates the function.

For example:

\[
Density\ Classification = f(Density)
\]

where the function is manually defined.

---

Machine learning can be represented as:

\[
Output = Model(Input)
\]

The algorithm learns the function:

\[
f(X) \rightarrow y
\]

where

- \(X\) = input features,
- \(y\) = target output.

The goal of machine learning is to find a function that accurately maps inputs to outputs.

---

# 1.24 Another Materials Science Example

Imagine predicting the melting temperature of materials.

A traditional approach would require a scientist to write rules like:

```text
IF atomic bonding is strong
AND atomic size is small
AND crystal packing is efficient

THEN melting temperature is high
```

However, real materials contain complicated interactions.

Machine learning instead uses thousands of examples:

| Material | Features | Melting Temperature |
|-|-|-|
| Alloy 1 | Composition + Structure | 1200 K |
| Alloy 2 | Composition + Structure | 1600 K |
| Alloy 3 | Composition + Structure | 900 K |

The model learns patterns that may not be obvious to humans.

---

# 1.25 Machine Learning Does Not Replace Scientific Knowledge

An important point:

Machine learning does not eliminate the need for physics, chemistry, or materials science.

A common misunderstanding is that machine learning is a replacement for scientific understanding.

It is not.

The strongest materials research combines:

- domain knowledge,
- experimental understanding,
- computational methods,
- machine learning.

For example,

a materials scientist decides:

- which properties are important,
- which features represent the material,
- which predictions are physically meaningful.

The machine learning algorithm helps discover complex relationships within the data.

---

# 1.26 When Should We Use Machine Learning?

Machine learning is especially useful when:

## 1. Large Amounts of Data Exist

Example:

Millions of materials in computational databases.

---

## 2. Relationships Are Complex

Example:

Predicting battery lifetime from composition and operating conditions.

---

## 3. Exact Physical Models Are Difficult

Example:

Predicting corrosion behavior under many environmental conditions.

---

## 4. Fast Predictions Are Needed

Example:

Screening thousands of candidate materials before laboratory testing.

---

# 1.27 When Should We Not Use Machine Learning?

Machine learning is not always the best solution.

Avoid unnecessary machine learning when:

- the problem has a simple mathematical solution,
- there is insufficient data,
- the relationship is already completely understood,
- interpretability is more important than prediction accuracy.

For example,

calculating stress using Hooke's law does not require machine learning.

The equation already exists.

---

# 1.28 Complete Comparison

| Aspect | Traditional Programming | Machine Learning |
|---|---|---|
| Rules | Written by humans | Learned from data |
| Input | Data + Rules | Data + Examples |
| Output | Answer | Learned model |
| Flexibility | Limited by rules | Improves with more data |
| Best for | Clearly defined problems | Complex pattern recognition |
| Human role | Create instructions | Provide data and guidance |

---

In the next section, we will introduce the **four major types of machine learning: supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning**. Understanding these categories is essential before studying individual algorithms.


# 1.29 Types of Machine Learning

Machine learning algorithms can be divided into several categories based on **how they learn from data**.

The four major categories are:

1. Supervised Learning
2. Unsupervised Learning
3. Semi-Supervised Learning
4. Reinforcement Learning

Understanding these categories is important because each type solves different kinds of problems.

Before selecting an algorithm, a researcher must first understand what type of learning problem they have.

---

# 1.30 Supervised Learning

Supervised learning is the most commonly used type of machine learning.

In supervised learning, the algorithm learns from examples where the correct answer is already known.

The dataset contains:

- Input variables (features)
- Desired output (target)

The model learns the relationship between them.

The basic idea is:

```text
Input Data + Correct Answers

          ↓

Machine Learning Algorithm

          ↓

Learned Relationship

          ↓

Prediction for New Data
```

---

# 1.31 Example of Supervised Learning in Materials Science

Suppose we want to predict the hardness of an alloy.

We collect experimental data:

| Carbon (%) | Chromium (%) | Heat Treatment | Hardness (HV) |
|------------|--------------|----------------|---------------|
| 0.2 | 5 | Annealed | 180 |
| 0.5 | 10 | Quenched | 450 |
| 0.8 | 12 | Tempered | 620 |

Here:

## Features (Inputs)

The information given to the model:

- Carbon percentage
- Chromium percentage
- Heat treatment condition

Represented mathematically as:

\[
X
\]

---

## Target (Output)

The value we want to predict:

- Hardness

Represented mathematically as:

\[
y
\]

---

The model learns:

\[
X \rightarrow y
\]

After training, we can provide a new alloy composition:

```text
Carbon = 0.6%
Chromium = 11%
Heat treatment = Quenched
```

The model predicts:

```text
Estimated hardness = 520 HV
```

The model has learned from previous examples.

---

# 1.32 Two Main Types of Supervised Learning

Supervised learning is divided into:

1. Regression
2. Classification

The difference depends on the type of output.

---

# 1.33 Regression

Regression predicts a continuous numerical value.

Examples:

Predicting:

- hardness,
- melting temperature,
- elastic modulus,
- thermal conductivity,
- battery capacity,
- bandgap energy.

The output is a number.

Example:

Input:

```text
Material composition
```

Output:

```text
Melting temperature = 1450 K
```

Mathematically:

\[
f(X)=y
\]

where:

$$y \in \mathbb{R}$$

meaning the output can be any real number.

---

# 1.34 Classification

Classification predicts categories or labels.

Examples:

Predicting whether a material is:

- metal,
- semiconductor,
- insulator.

Or:

- corrosion resistant,
- corrosion susceptible.

The output is a class.

Example:

Input:

```text
Bandgap = 1.8 eV
```

Output:

```text
Semiconductor
```

Mathematically:

\[
y \in \{Class_1,Class_2,...,Class_n\}
\]

---

# 1.35 Regression vs Classification Example

Consider predicting the behavior of a material.

## Regression Problem

Question:

> What is the electrical conductivity?

Possible output:

```text
5.8 × 10^7 S/m
```

This is a continuous number.

Therefore:

Regression.

---

## Classification Problem

Question:

> Is this material conductive or non-conductive?

Possible output:

```text
Conductive
```

This is a category.

Therefore:

Classification.

---

# 1.36 Common Supervised Learning Algorithms

In this book, we will study many supervised learning algorithms.

Examples include:

## Linear Regression

Used for predicting continuous values.

Example:

Predicting elastic modulus.

---

## Decision Trees

Models that learn decision rules.

Example:

Classifying materials based on properties.

---

## Random Forests

Collections of many decision trees.

Example:

Predicting mechanical properties from multiple features.

---

## Support Vector Machines

Finding optimal boundaries between classes.

Example:

Classifying materials into different groups.

---

## Gradient Boosting Algorithms

Including:

- Gradient Boosting,
- XGBoost,
- LightGBM,
- CatBoost.

These are among the most powerful algorithms for structured scientific data.

---

# 1.37 Unsupervised Learning

In unsupervised learning, the dataset does not contain known answers.

The algorithm receives only input data.

The goal is to discover hidden structures.

The basic idea:

```text
Input Data

     ↓

Machine Learning Algorithm

     ↓

Discover Hidden Patterns
```

---

# 1.38 Example of Unsupervised Learning

Imagine we have thousands of materials.

For each material we know:

- atomic radius,
- density,
- electronegativity,
- crystal information.

However, we do not know their categories.

An unsupervised algorithm may discover groups such as:

```text
Group 1:
Similar metallic materials

Group 2:
Similar semiconductor materials

Group 3:
Similar ceramic materials
```

The algorithm finds patterns without being told the answers.

---

# 1.39 Common Unsupervised Learning Tasks

## Clustering

Groups similar data points together.

Examples:

- discovering material families,
- grouping similar crystal structures.

Algorithms:

- K-Means,
- Hierarchical Clustering,
- DBSCAN.

---

## Dimensionality Reduction

Reduces the number of variables while preserving important information.

Examples:

- visualizing high-dimensional materials databases,
- removing redundant features.

Algorithms:

- PCA,
- t-SNE,
- UMAP.

---

# 1.40 Semi-Supervised Learning

Semi-supervised learning combines supervised and unsupervised learning.

Many real scientific datasets contain:

- a small amount of labeled data,
- a large amount of unlabeled data.

For example:

A laboratory may have experimentally measured properties for only 500 materials.

However, computational databases may contain millions of materials without experimental labels.

Semi-supervised learning uses both sources.

The idea:

```text
Small labeled dataset

+

Large unlabeled dataset

↓

Improved Learning
```

This approach is especially valuable in materials discovery because experimental measurements are expensive.

---

# 1.41 Reinforcement Learning

Reinforcement learning is based on learning through interaction.

Instead of receiving a correct answer,

the algorithm receives rewards or penalties.

The system learns which actions produce better outcomes.

The basic structure:

```text
Agent

↓

Action

↓

Environment

↓

Reward / Penalty

↓

Improved Strategy
```

---

# 1.42 Example of Reinforcement Learning

Imagine designing a material processing method.

An AI system chooses:

- temperature,
- pressure,
- cooling rate.

The resulting material quality determines the reward.

Over many attempts,

the system learns processing conditions that maximize performance.

Applications include:

- autonomous laboratories,
- experimental optimization,
- robotics,
- process control.

---

# 1.43 Comparing the Four Types

| Type | Data Required | Goal | Example |
|---|---|---|---|
| Supervised | Labeled data | Predict outputs | Predict hardness |
| Unsupervised | Unlabeled data | Find patterns | Group materials |
| Semi-supervised | Few labels + many unlabeled samples | Improve learning | Materials discovery |
| Reinforcement | Interaction + reward | Learn decisions | Optimize experiments |

---

# 1.44 Which Type Will We Focus On?

The majority of this book focuses on:

## Supervised Learning

because it is currently the most widely used approach for materials property prediction.

You will learn algorithms such as:

- Linear Regression,
- Decision Trees,
- Random Forests,
- Gradient Boosting,
- XGBoost.

We will also cover important unsupervised techniques such as:

- clustering,
- dimensionality reduction.

Later chapters will introduce advanced methods relevant to modern materials informatics.

---

In the next section, we will learn the fundamental language of machine learning: **datasets, samples, features, targets, and labels**. These concepts appear in every algorithm and every research paper.

# 1.45 Understanding Datasets, Samples, Features, and Targets

Before studying machine learning algorithms, we must learn the language used throughout machine learning.

Almost every research paper, textbook, and software library uses terms such as

- dataset,
- sample,
- feature,
- target,
- label,
- observation,
- instance.

These words appear repeatedly in every chapter of this book.

Understanding them now will make learning machine learning much easier.

---

# 1.46 What Is a Dataset?

A **dataset** is a collection of data organized in a structured form.

In machine learning, a dataset is usually represented as a table.

Each row represents one observation.

Each column represents one property.

For example,

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |
| Aluminum | 2.70 | 0.00 | 167 |
| Silicon | 2.33 | 1.12 | 1150 |
| Copper | 8.96 | 0.00 | 369 |

This entire table is called a **dataset**.

Datasets may contain

- tens of rows,
- thousands of rows,
- or even millions of rows.

The quality of a machine learning model depends heavily on the quality of its dataset.

---

# 1.47 What Is a Sample?

A **sample** is a single row in a dataset.

Each sample represents one observation or one example.

Looking at our dataset,

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |

This single row is one sample.

Similarly,

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Silicon | 2.33 | 1.12 | 1150 |

is another sample.

A dataset is simply a collection of many samples.

---

# 1.48 Other Names for a Sample

Different books use different terminology.

The following words often mean the same thing:

| Term | Meaning |
|------|---------|
| Sample | One data point |
| Observation | One data point |
| Instance | One data point |
| Record | One row of data |
| Example | One data point |

Although the names differ,

they all refer to a single row in the dataset.

Throughout this book,

we will primarily use the term **sample**.

---

# 1.49 What Is a Feature?

A **feature** is an input variable that describes a sample.

Features are the information used by the machine learning model to make predictions.

For our example,

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |

Possible features are

- Density
- Bandgap

These properties help describe the material.

The model studies these features to learn patterns.

Mathematically,

all features are represented by

\[
X
\]

which is called the **feature matrix**.

---

# 1.50 Feature Matrix (X)

Suppose we use only Density and Bandgap as inputs.

The feature matrix becomes

| Density | Bandgap |
|---------:|---------:|
| 7.85 | 0.00 |
| 2.70 | 0.00 |
| 2.33 | 1.12 |
| 8.96 | 0.00 |

Notice that

the Material column is not included because it is simply the material's name.

The Hardness column is not included because it is the value we want to predict.

Therefore,

$$X=
\begin{bmatrix}
7.85 & 0.00\\
2.70 & 0.00\\
2.33 & 1.12\\
8.96 & 0.00
\end{bmatrix}$$

Every row corresponds to one sample.

Every column corresponds to one feature.

---

# 1.51 What Is the Target?

The **target** is the value that we want the machine learning model to predict.

In our example,

the target is

| Hardness |
|----------:|
| 250 |
| 167 |
| 1150 |
| 369 |

The target is represented mathematically by

$$y$$

This is often called the **target vector**.

The objective of supervised learning is to learn the relationship


$$X \rightarrow y$$


---

# 1.52 What Is a Label?

The word **label** is commonly used in **classification** problems.

Suppose our dataset is

| Density | Bandgap | Material Type |
|---------:|---------:|---------------|
| 7.85 | 0.00 | Metal |
| 2.33 | 1.12 | Semiconductor |
| 1.80 | 5.40 | Insulator |

Here,

the output column

```text
Material Type
```

contains categories rather than numbers.

These categories are called **labels**.

Examples include

- Metal,
- Semiconductor,
- Insulator.

Thus,

- Regression predicts numerical targets.
- Classification predicts labels.

---

# 1.53 Features vs Target

Students often confuse these two concepts.

Remember:

| Features | Target |
|----------|--------|
| Used as input | Predicted by the model |
| Known before prediction | Unknown before prediction |
| Represented by \(X\) | Represented by \(y\) |

Think of it this way:

Features describe the material.

The target is the property we want to estimate.

---

# 1.54 A Complete Example

Suppose we wish to predict hardness.

Our dataset is

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |
| Aluminum | 2.70 | 0.00 | 167 |
| Silicon | 2.33 | 1.12 | 1150 |
| Copper | 8.96 | 0.00 | 369 |

Then

### Features

```text
Density

Bandgap
```

### Target

```text
Hardness
```

The machine learning algorithm receives

```text
Density

Bandgap
```

and learns to predict

```text
Hardness
```

---

# 1.55 Converting a Dataset into X and y Using Pandas

Now let's see how this is done in Python.

```python
import pandas as pd

data = {
    "Material": ["Steel", "Aluminum", "Silicon", "Copper"],
    "Density": [7.85, 2.70, 2.33, 8.96],
    "Bandgap": [0.00, 0.00, 1.12, 0.00],
    "Hardness": [250, 167, 1150, 369]
}

df = pd.DataFrame(data)

X = df[["Density", "Bandgap"]]

y = df["Hardness"]

print(X)

print(y)
```

---

## Line-by-Line Explanation

### Line 1

```python
import pandas as pd
```

Imports the Pandas library.

---

### Lines 3–8

```python
data = {
    ...
}
```

Creates a Python dictionary containing the dataset.

---

### Line 10

```python
df = pd.DataFrame(data)
```

Converts the dictionary into a DataFrame.

---

### Line 12

```python
X = df[["Density", "Bandgap"]]
```

Selects two columns from the DataFrame.

These columns become the feature matrix.

Notice the use of **double square brackets**.

They return a DataFrame containing multiple columns.

---

### Line 14

```python
y = df["Hardness"]
```

Selects the target column.

Since only one column is selected,

Pandas returns a Series.

This Series becomes the target vector.

---

### Lines 16–18

```python
print(X)

print(y)
```

Display the feature matrix and target vector.

These variables are now ready to be used by a machine learning algorithm.

---

# 1.56 Why Choosing Good Features Matters

Not every column in a dataset should be used as a feature.

For example,

suppose our dataset contains

- Material ID,
- Laboratory notebook number,
- Date of measurement.

These columns usually have no physical relationship with hardness.

Including irrelevant features may confuse the model and reduce prediction accuracy.

Good feature selection is therefore one of the most important tasks in machine learning.

Later chapters will discuss feature engineering and feature selection in detail.

---

# 1.57 Practice Exercise

Consider the following dataset.

| Material | Density | Elastic Modulus | Thermal Conductivity |
|-----------|---------:|----------------:|---------------------:|
| Steel | 7.85 | 210 | 50 |
| Copper | 8.96 | 110 | 401 |
| Aluminum | 2.70 | 69 | 237 |
| Titanium | 4.51 | 116 | 22 |

Answer the following questions.

1. How many samples are in the dataset?
2. Which columns could be used as features?
3. Which column would you choose as the target if you wanted to predict thermal conductivity?
4. Write Python code to create `X` and `y`.

Try solving these questions before reading further.

Understanding datasets and features is essential for every machine learning algorithm.

---

In the next section, we will learn about the **machine learning workflow**, following a dataset from raw data to a trained model and finally to predictions on unseen materials.


# 1.58 The Complete Machine Learning Workflow

Now that we understand

- datasets,
- samples,
- features,
- targets,

we can study the **complete machine learning workflow**.

One of the biggest mistakes beginners make is believing that machine learning consists only of training a model.

In reality,

training the model is only one step in a much larger process.

Professional machine learning projects follow a systematic workflow that ensures the resulting model is reliable, accurate, and applicable to real-world problems.

Throughout the rest of this book, we will repeatedly follow this workflow.

By the end, it should become second nature.

---

# 1.59 Overview of the Workflow

A typical supervised machine learning project follows these steps.

```text
Define the Problem
        │
        ▼
Collect Data
        │
        ▼
Explore the Data
        │
        ▼
Clean and Preprocess the Data
        │
        ▼
Split the Dataset
        │
        ▼
Choose a Machine Learning Model
        │
        ▼
Train the Model
        │
        ▼
Evaluate the Model
        │
        ▼
Improve the Model
        │
        ▼
Make Predictions
```

Each step has a specific purpose.

Skipping any of them can reduce the quality of the final model.

---

# 1.60 Step 1 — Define the Problem

Every machine learning project begins with a clearly defined question.

For example,

instead of saying

> "I want to use machine learning."

we should ask a specific scientific question such as

- Can we predict the hardness of a material?
- Can we estimate the bandgap of a semiconductor?
- Can we classify materials as metallic or non-metallic?
- Can we identify promising battery materials?

Notice that machine learning is **not the goal**.

The goal is to solve a scientific or engineering problem.

Machine learning is the tool.

---

# 1.61 Step 2 — Collect the Data

Once the problem is defined,

we gather data.

Possible sources include

- laboratory experiments,
- published scientific literature,
- computational simulations,
- online databases,
- previous research projects.

For materials informatics,

common information includes

- composition,
- density,
- crystal structure,
- grain size,
- elastic modulus,
- hardness,
- thermal conductivity,
- bandgap.

The quality of the data strongly influences the quality of the final model.

A useful principle is

> **Better data usually produce better models.**

---

# 1.62 Step 3 — Explore the Data

Before training any model,

we must understand the dataset.

Typical questions include

- How many samples are available?
- Are there missing values?
- Are there obvious errors?
- What are the ranges of the variables?
- Are there unusual outliers?
- Are some features strongly correlated?

During this stage,

we commonly use

- `head()`,
- `info()`,
- `describe()`,
- histograms,
- scatter plots,
- correlation analysis.

Exploratory Data Analysis (EDA) often reveals problems that are invisible at first glance.

---

# 1.63 Step 4 — Clean and Preprocess the Data

Real-world data are rarely perfect.

Common preprocessing tasks include

- removing duplicate rows,
- handling missing values,
- correcting incorrect entries,
- selecting useful features,
- scaling numerical features,
- encoding categorical variables.

A machine learning model should never be trained on poor-quality data.

Good preprocessing often improves performance more than changing the algorithm itself.

---

# 1.64 Step 5 — Split the Dataset

This is one of the most important steps in machine learning.

Instead of training on the entire dataset,

we divide it into separate parts.

Typically,

- one part is used for learning,
- another part is reserved for evaluation.

A common split is

```text
Training Set

↓

Model Learns

Testing Set

↓

Model Evaluation
```

The testing data must remain unseen during training.

Otherwise,

we cannot determine whether the model has actually learned or simply memorized the data.

In a later section, we will also introduce the **validation set**, which is used to tune machine learning models.

---

# 1.65 Step 6 — Choose a Machine Learning Algorithm

The choice of algorithm depends on the problem.

Examples include

| Problem | Possible Algorithm |
|----------|--------------------|
| Predict hardness | Linear Regression |
| Predict elastic modulus | Random Forest |
| Predict bandgap | XGBoost |
| Classify materials | Decision Tree |
| Group similar materials | K-Means |

There is no universally best algorithm.

Different problems require different approaches.

One of the goals of this book is to help you understand when each algorithm is appropriate.

---

# 1.66 Step 7 — Train the Model

Training is the stage where learning occurs.

The algorithm studies the relationship between

- features (`X`),
- targets (`y`).

During training,

the model adjusts its internal parameters to reduce prediction errors.

After training,

the model has learned a mathematical relationship from the data.

It can now make predictions for new samples.

---

# 1.67 Step 8 — Evaluate the Model

Training accuracy alone is not enough.

A model might perform perfectly on the training data yet fail completely on new data.

Therefore,

we evaluate the model using data that it has never seen before.

Depending on the problem,

we use evaluation metrics such as

- Mean Absolute Error (MAE),
- Mean Squared Error (MSE),
- Root Mean Squared Error (RMSE),
- Accuracy,
- Precision,
- Recall,
- F1-score,
- ROC-AUC.

Each metric measures a different aspect of model performance.

Later chapters will explain these metrics in detail.

---

# 1.68 Step 9 — Improve the Model

The first model is rarely the best model.

After evaluation,

we often improve performance by

- collecting more data,
- selecting better features,
- tuning hyperparameters,
- trying different algorithms,
- reducing overfitting,
- improving preprocessing.

Machine learning is an iterative process.

Researchers repeatedly improve models until satisfactory performance is achieved.

---

# 1.69 Step 10 — Make Predictions

Once we are satisfied with the model,

it can be used on new, unseen data.

For example,

suppose a new alloy has

- Density = 7.6 g/cm³
- Bandgap = 0.2 eV

The trained model predicts

```text
Estimated Hardness = 340 HV
```

Notice that this alloy was never included in the training dataset.

The ability to generalize to unseen samples is the ultimate objective of machine learning.

---

# 1.70 A Materials Informatics Example

Suppose researchers wish to discover new high-strength alloys.

The workflow would be

```text
Research Question

↓

Collect Experimental Data

↓

Clean Dataset

↓

Visualize Data

↓

Choose Features

↓

Train Random Forest

↓

Evaluate Predictions

↓

Predict Properties of New Alloys

↓

Experimentally Test Only the Most Promising Candidates
```

Instead of testing thousands of alloys experimentally,

researchers can focus only on the most promising predictions.

This significantly reduces cost and development time.

---

# 1.71 Why This Workflow Matters

Many beginners focus entirely on selecting sophisticated algorithms.

In practice,

the success of a machine learning project often depends more on

- high-quality data,
- careful preprocessing,
- proper evaluation,
- thoughtful feature selection,

than on choosing a complex model.

A simple algorithm trained on excellent data often outperforms a sophisticated algorithm trained on poor data.

---

# 1.72 Practice Exercise

Suppose you want to build a machine learning model to predict the thermal conductivity of materials.

Answer the following questions.

1. What is the problem you are trying to solve?
2. What type of data would you collect?
3. Which columns might be useful as features?
4. What would be the target variable?
5. Why should you split the dataset before training?
6. How would you determine whether your model performs well?

Think through each step carefully.

Being able to design a workflow is just as important as writing code.

---


In the next section, we will study **training sets, validation sets, and test sets** in detail. You will learn why machine learning datasets are divided into multiple parts and how this helps us measure a model's ability to generalize to unseen data.


# 1.73 Training Set, Validation Set, and Test Set

One of the most important ideas in machine learning is that **a model should be evaluated on data it has never seen before**.

Many beginners believe that if a model predicts the training data perfectly, then it must be a good model.

Unfortunately, this is often false.

A model can simply **memorize** the training data instead of learning the underlying relationships.

To prevent this problem, machine learning datasets are divided into separate subsets.

These subsets allow us to train, improve, and fairly evaluate a model.

---

# 1.74 Why Can't We Use the Entire Dataset for Training?

Imagine that you are preparing for an examination.

Suppose your teacher gives you the exact exam paper one week before the exam.

You memorize every answer.

On exam day, you score 100%.

Does this mean you understand the subject deeply?

Not necessarily.

You may have memorized the answers rather than learned the concepts.

Machine learning models can behave in exactly the same way.

If a model is evaluated using the same data it learned from, a high score may simply indicate memorization.

Our real goal is to determine whether the model can make accurate predictions for **new, unseen data**.

---

# 1.75 The Three Parts of a Dataset

A dataset is commonly divided into three subsets.

```text
Entire Dataset

        │

        ├──────────────┐

        ▼              ▼

Training Set      Remaining Data

                      │

          ┌───────────┴───────────┐

          ▼                       ▼

Validation Set              Test Set
```

Each subset serves a different purpose.

---

# 1.76 Training Set

The **training set** is used to teach the machine learning model.

The algorithm examines the features (`X`) and targets (`y`) and learns the relationship between them.

During training,

the model continuously adjusts its internal parameters to reduce prediction errors.

This is the only dataset from which the model is allowed to learn.

Think of the training set as the textbook from which a student studies.

---

# 1.77 Validation Set

After training,

we often need to make decisions such as

- Which algorithm should we use?
- How deep should a decision tree be?
- How many trees should a random forest contain?
- Which hyperparameters produce the best performance?

To answer these questions,

we use the **validation set**.

The validation set is **not** used to teach the model.

Instead,

it helps us compare different versions of the model.

We repeatedly evaluate candidate models on the validation set and choose the one that performs best.

Think of the validation set as a series of practice tests that help a student improve before the final examination.

---

# 1.78 Test Set

The **test set** is used only once,

after all training and model selection have been completed.

It provides an unbiased estimate of how well the final model will perform on completely unseen data.

The model must never learn from the test set.

If we repeatedly examine the test results while modifying the model,

the test set gradually becomes part of the training process,

making the evaluation unfair.

Think of the test set as the final examination.

You should not practice using the final exam questions.

---

# 1.79 Summary of the Three Sets

| Dataset | Purpose | Used for Learning? |
|----------|---------|-------------------|
| Training Set | Learn patterns from data | Yes |
| Validation Set | Select models and tune hyperparameters | No |
| Test Set | Measure final performance | No |

Remember:

Only the training set teaches the model.

The validation and test sets measure performance.

---

# 1.80 Typical Dataset Splits

There is no universal rule for dividing a dataset.

However, several splits are commonly used.

| Training | Validation | Test |
|-----------|-----------|------|
| 60% | 20% | 20% |
| 70% | 15% | 15% |
| 80% | 10% | 10% |

The appropriate choice depends on

- the size of the dataset,
- the complexity of the problem,
- the amount of available data.

Large datasets often devote a greater percentage to training.

Small datasets may require different evaluation strategies, such as cross-validation, which we will study later.

---

# 1.81 A Materials Science Example

Suppose we have experimental measurements for **10,000 alloys**.

A possible split might be

| Dataset | Number of Alloys |
|----------|-----------------:|
| Training Set | 8,000 |
| Validation Set | 1,000 |
| Test Set | 1,000 |

The workflow becomes

```text
8,000 Alloys

↓

Train the Model

↓

Evaluate Using 1,000 Validation Alloys

↓

Choose the Best Model

↓

Evaluate Once Using 1,000 Test Alloys

↓

Report Final Performance
```

This procedure gives us confidence that the reported performance reflects the model's ability to generalize.

---

# 1.82 Training, Validation, and Testing in Python

Scikit-learn provides a convenient function for splitting datasets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This example creates a training set and a test set.

Later chapters will show how to create a separate validation set and how to use cross-validation.

---

## Line-by-Line Explanation

### Line 1

```python
from sklearn.model_selection import train_test_split
```

Imports the `train_test_split()` function from Scikit-learn.

This function randomly divides a dataset into separate subsets.

---

### Line 3

```python
X_train, X_test, y_train, y_test =
```

The function returns four variables.

- `X_train` contains the training features.
- `X_test` contains the testing features.
- `y_train` contains the training targets.
- `y_test` contains the testing targets.

Using clear variable names makes machine learning code easier to read.

---

### Lines 4–8

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Let's examine each argument.

#### `X`

The feature matrix.

---

#### `y`

The target vector.

---

#### `test_size=0.2`

This tells Scikit-learn that **20% of the samples** should be placed into the test set.

The remaining **80%** become the training set.

---

#### `random_state=42`

The dataset is shuffled before splitting.

Setting `random_state` to a fixed value ensures that the same random split is produced every time the program is executed.

This improves reproducibility.

The specific value `42` has no special mathematical meaning.

Any fixed integer could be used.

---

# 1.83 What Happens Internally?

Suppose we have ten samples.

```text
1
2
3
4
5
6
7
8
9
10
```

After shuffling,

Scikit-learn might produce

```text
7
2
9
1
10
4
6
3
8
5
```

If `test_size=0.2`,

then

```text
Training Set

7
2
9
1
10
4
6
3
```

and

```text
Test Set

8
5
```

Notice that the samples are selected randomly rather than simply taking the last two rows.

Random splitting helps prevent systematic bias.

---

# 1.84 Common Beginner Mistakes

Several mistakes occur frequently.

### Mistake 1

Training and testing on the same dataset.

Result:

Unrealistically optimistic performance.

---

### Mistake 2

Looking at the test results repeatedly while improving the model.

Result:

The test set is no longer independent.

---

### Mistake 3

Using a very small training set.

Result:

The model may not have enough information to learn meaningful patterns.

---

### Mistake 4

Forgetting to shuffle the data before splitting.

Result:

The training and testing sets may not represent the same data distribution.

Fortunately,

`train_test_split()` shuffles the data by default.

---

# 1.85 Practice Exercise

Suppose you have a dataset containing **5,000 material samples**.

Answer the following questions.

1. How many samples would be in the training and testing sets if `test_size=0.2`?
2. Why should the test set remain unseen during training?
3. What is the purpose of `random_state`?
4. Why is random splitting generally better than simply taking the last rows of the dataset?

Try answering these questions before moving to the next section.

Understanding dataset splitting is essential because every supervised machine learning model in this book will use this procedure.

---

In the next section, we will study one of the most important concepts in machine learning: **overfitting and underfitting**. You will learn why some models memorize the training data, why others fail to learn meaningful patterns, and how to recognize the difference.


# 1.86 Overfitting and Underfitting

One of the primary goals of machine learning is **generalization**.

Generalization means that a model should perform well not only on the data it has already seen but also on **new, unseen data**.

A model that memorizes the training data without learning the underlying patterns is not useful.

Similarly, a model that fails to learn even the basic relationships in the data is also ineffective.

These two situations are known as

- **Underfitting**
- **Overfitting**

Finding the balance between them is one of the central challenges in machine learning.

---

# 1.87 What Is Underfitting?

A model is said to **underfit** when it is too simple to capture the underlying relationship in the data.

As a result,

- it performs poorly on the training data,
- it also performs poorly on new data.

In other words,

the model has **not learned enough**.

Imagine trying to describe the Earth's surface using a perfectly flat sheet.

The flat sheet ignores mountains, valleys, rivers, and hills.

It is too simple to represent reality.

An underfitted machine learning model behaves in the same way.

---

# 1.88 Example of Underfitting

Suppose the true relationship between density and hardness is complex.

A very simple model might predict

```text
Every material has a hardness of 300 HV.
```

Regardless of the input,

the prediction never changes.

For example,

| Density | Actual Hardness | Model Prediction |
|---------:|----------------:|-----------------:|
| 2.70 | 167 | 300 |
| 4.51 | 349 | 300 |
| 7.85 | 250 | 300 |
| 8.96 | 369 | 300 |

Clearly,

the model has failed to learn the relationship.

---

# 1.89 Characteristics of Underfitting

An underfitted model usually has

- high training error,
- high testing error,
- poor predictive ability,
- insufficient complexity.

The model ignores important information contained in the dataset.

---

# 1.90 What Is Overfitting?

Overfitting is the opposite problem.

An overfitted model learns the training data **too well**.

Instead of learning general patterns,

it memorizes individual examples,

including

- random noise,
- measurement errors,
- accidental variations.

As a result,

the model performs extremely well on the training data,

but performs poorly on unseen data.

---

# 1.91 Student Analogy

Imagine two students preparing for an examination.

### Student A

Studies the concepts,

understands the ideas,

and can solve new problems.

This student represents a **well-trained model**.

---

### Student B

Memorizes every answer from one practice paper.

During the actual examination,

the questions are slightly different.

The student becomes confused.

This student represents an **overfitted model**.

The student memorized instead of understanding.

Machine learning models can make the same mistake.

---

# 1.92 Example of Overfitting

Suppose a decision tree continues splitting until every training sample is perfectly classified.

Training accuracy becomes

```text
100%
```

This sounds impressive.

However,

when tested on new materials,

accuracy may drop dramatically because the model has memorized the training data rather than learned general rules.

High training accuracy does **not** always indicate a good model.

---

# 1.93 Characteristics of Overfitting

An overfitted model usually has

- extremely low training error,
- much higher testing error,
- poor generalization,
- excessive complexity.

The model is trying to learn details that are unique to the training dataset rather than meaningful scientific relationships.

---

# 1.94 A Balanced Model

The ideal model lies between underfitting and overfitting.

Such a model

- captures important relationships,
- ignores random noise,
- performs well on unseen data.

Its objective is not to memorize the dataset.

Its objective is to learn the underlying pattern.

This ability to perform well on new data is called **generalization**.

---

# 1.95 Visualizing the Three Cases

Imagine we are fitting a curve through experimental data.

### Underfitting

```text
Data:      •   •    •      •

Model: -----------------------
```

The model is too simple.

It misses the overall trend.

---

### Good Fit

```text
Data:      •   •    •      •

Model:  ~~~~~~~~~~~~~~~~~~~~~
```

The model captures the main relationship without following every small fluctuation.

---

### Overfitting

```text
Data:      •   •    •      •

Model: /\/\/\_/\/\__/\/\/\/\
```

The model follows every individual point,

including measurement noise.

This reduces its ability to predict new observations.

---

# 1.96 Training Error vs Testing Error

As model complexity increases,

training error usually decreases.

However,

testing error behaves differently.

```text
Error

^

|

| \

|  \            Testing Error

|   \         /\
|    \       /  \

|     \_____/    \

|------------------------------> Model Complexity

       Training Error
```

Notice that

- training error continuously decreases,
- testing error first decreases,
- then begins increasing.

The lowest point of the testing error corresponds to the best model.

---

# 1.97 Why Does Overfitting Occur?

Several factors can cause overfitting.

## Too Complex a Model

A very flexible model may memorize the training data.

---

## Too Little Training Data

With only a small number of examples,

the model may mistake random noise for meaningful patterns.

---

## Noisy Data

Experimental errors and measurement uncertainty can mislead the model.

---

## Too Many Features

Using unnecessary variables increases the risk of learning accidental relationships.

Feature selection helps reduce this problem.

---

# 1.98 How Can We Reduce Overfitting?

Common strategies include

- collecting more training data,
- simplifying the model,
- removing irrelevant features,
- using cross-validation,
- applying regularization,
- pruning decision trees,
- limiting model complexity,
- stopping training at the appropriate time.

These techniques will be introduced throughout later chapters.

---

# 1.99 How Can We Reduce Underfitting?

If a model is underfitting,

possible solutions include

- using a more powerful algorithm,
- increasing model complexity,
- adding more informative features,
- training for longer (when applicable),
- reducing excessive regularization.

The appropriate solution depends on the algorithm being used.

---

# 1.100 Materials Science Example

Suppose we wish to predict the yield strength of steel alloys.

### Underfitted Model

Uses only one feature:

- Density.

Many important variables such as composition and heat treatment are ignored.

Prediction accuracy is poor.

---

### Overfitted Model

Uses hundreds of features,

including several irrelevant or noisy variables.

The model memorizes the training alloys but performs poorly on newly developed alloys.

---

### Well-Generalized Model

Uses physically meaningful features such as

- carbon content,
- chromium content,
- grain size,
- heat treatment,
- phase fraction.

The model captures the important relationships while ignoring random noise.

Such a model is more likely to make reliable predictions for previously unseen materials.

---

# 1.101 Practice Exercise

A machine learning model achieves

- Training Accuracy = **99.8%**
- Test Accuracy = **72.5%**

1. Is the model likely underfitting or overfitting?
2. Why is the test accuracy much lower than the training accuracy?
3. Suggest at least three methods to improve the model.

Next, consider another model with

- Training Accuracy = **63%**
- Test Accuracy = **61%**

1. Is this model underfitting or overfitting?
2. What changes might improve its performance?

These examples illustrate why both training and testing performance must always be considered together.

---

In the next section, we will study **bias and variance**, the theoretical concepts that explain *why* underfitting and overfitting occur and how machine learning practitioners balance them when designing models.


# 1.102 Bias and Variance

In the previous section, we learned that a machine learning model can suffer from

- underfitting,
- overfitting.

But **why** do these problems occur?

The answer lies in two fundamental concepts:

- **Bias**
- **Variance**

These concepts help us understand the strengths and weaknesses of a machine learning model.

Nearly every machine learning algorithm can be analyzed in terms of its bias and variance.

Understanding this topic will help you choose appropriate models and improve their performance.

---

# 1.103 What Is Bias?

**Bias** measures how much a model's predictions differ from the true underlying relationship because the model is too simple.

A model with **high bias** makes strong assumptions about the data.

As a result,

it cannot capture complex patterns.

High bias usually leads to **underfitting**.

---

# 1.104 A Simple Analogy

Imagine an archer aiming at the center of a target.

Suppose every arrow lands far to the left of the center.

The arrows are close to each other,

but all of them miss the target.

```text
Target Center

        X

●●●
```

The archer is consistently making the same mistake.

This is similar to **high bias**.

The predictions are consistently incorrect because the model is too simple.

---

# 1.105 High Bias in Machine Learning

Suppose the true relationship between density and hardness is curved.

However,

our model always assumes the relationship is a straight line.

No matter how much data we collect,

the model cannot represent the true pattern.

Its assumptions are too restrictive.

This is a high-bias model.

---

# 1.106 What Is Variance?

**Variance** measures how much a model changes when the training data change slightly.

A model with **high variance** is very sensitive to the training dataset.

Even small changes in the data may produce a very different model.

High variance usually leads to **overfitting**.

---

# 1.107 Another Analogy

Consider the same archer.

This time,

the arrows are scattered everywhere.

```text
      ●

           ●

  ●

                 ●

        X

     ●
```

Some arrows are left,

others are right,

some are above,

and some are below the target.

The archer is inconsistent.

This resembles a high-variance machine learning model.

The model changes dramatically depending on the training data.

---

# 1.108 High Variance in Machine Learning

Suppose we train a decision tree.

If we remove just one training sample,

the tree changes completely.

Its predictions become very different.

This indicates that the model is highly sensitive to the training data.

Such models often memorize the training set rather than learning general patterns.

---

# 1.109 Relationship Between Bias and Variance

Bias and variance are closely connected.

In general,

- increasing model complexity decreases bias,
- increasing model complexity increases variance.

Conversely,

- simplifying a model increases bias,
- simplifying a model reduces variance.

This creates a trade-off.

---

# 1.110 The Bias–Variance Trade-Off

One of the central goals of machine learning is to balance bias and variance.

```text
Model Complexity
Low -------------------------------------- High

High Bias
Low Variance
(Underfitting)

                ▼ Best Balance ▼

Low Bias
High Variance
(Overfitting)
```

Neither extreme is desirable.

The objective is to find a model that is complex enough to learn important patterns but simple enough to generalize well.

---

# 1.111 Connecting the Concepts

The relationship can be summarized as follows.

| Situation | Bias | Variance |
|-----------|------|----------|
| Underfitting | High | Low |
| Good Fit | Moderate | Moderate |
| Overfitting | Low | High |

Notice that

underfitting is mainly a bias problem,

whereas overfitting is mainly a variance problem.

---

# 1.112 Examples Using Common Algorithms

Different machine learning algorithms naturally exhibit different levels of bias and variance.

| Algorithm | Typical Bias | Typical Variance |
|-----------|--------------|------------------|
| Linear Regression | High | Low |
| Decision Tree (deep) | Low | High |
| Random Forest | Low | Moderate |
| XGBoost | Low | Moderate |
| Support Vector Machine | Depends on parameters | Depends on parameters |

This table provides only general tendencies.

The actual behavior depends on the dataset and the chosen hyperparameters.

---

# 1.113 Materials Science Example

Suppose we want to predict the tensile strength of aluminum alloys.

### Model A

Uses only density.

The predictions are consistently poor because important variables such as alloy composition and heat treatment are ignored.

This model has

- high bias,
- low variance.

---

### Model B

Uses hundreds of features,

including many irrelevant measurements.

It predicts the training data almost perfectly,

but performs poorly on new alloys.

This model has

- low bias,
- high variance.

---

### Model C

Uses carefully selected physical features,

such as

- alloy composition,
- grain size,
- precipitate fraction,
- heat treatment.

It performs well on both training and testing data.

This model achieves a good balance between bias and variance.

---

# 1.114 How Can We Reduce Bias?

If a model has high bias,

possible solutions include

- using a more flexible algorithm,
- adding informative features,
- increasing model complexity,
- reducing excessive regularization,
- improving feature engineering.

These changes allow the model to learn more complex relationships.

---

# 1.115 How Can We Reduce Variance?

If a model has high variance,

possible solutions include

- collecting more training data,
- removing irrelevant features,
- simplifying the model,
- using regularization,
- pruning decision trees,
- using ensemble methods such as Random Forests,
- applying cross-validation during model selection.

These techniques improve the model's ability to generalize.

---

# 1.116 Why This Matters

When building a machine learning model,

our objective is not to achieve the highest possible training accuracy.

Instead,

we want a model that performs well on **new, unseen data**.

Balancing bias and variance is one of the key principles that allows machine learning models to generalize successfully.

Throughout this book,

whenever we study a new algorithm,

we will discuss

- its typical bias,
- its typical variance,
- situations where it performs well,
- situations where it performs poorly.

By understanding these characteristics,

you will be able to select algorithms more intelligently rather than relying on trial and error.

---

# 1.117 Practice Exercise

A researcher trains two models to predict the hardness of metallic alloys.

**Model A**

- Training Error: High
- Test Error: High

**Model B**

- Training Error: Very Low
- Test Error: Much Higher than Training Error

Answer the following questions.

1. Which model likely has high bias?
2. Which model likely has high variance?
3. Which model is underfitting?
4. Which model is overfitting?
5. Suggest one improvement for each model.

Thinking through these questions will strengthen your understanding of the bias–variance trade-off before we begin studying actual machine learning algorithms.

---

In the next section, we will introduce **regression and classification**, the two fundamental problem types in supervised machine learning. Every supervised algorithm you learn later in this book is designed to solve one or both of these problems.


# 1.118 Regression and Classification

In Chapter 1, we briefly introduced **regression** and **classification** as the two major categories of supervised learning.

Now that you understand

- datasets,
- features,
- targets,
- training and testing,
- overfitting,
- bias and variance,

we are ready to study these two problem types in greater detail.

Almost every supervised machine learning algorithm is designed to solve either

- a regression problem,
- a classification problem,
- or both.

Therefore,

before learning any algorithm,

you must first identify **what type of problem you are trying to solve**.

Choosing the wrong type of algorithm is one of the most common beginner mistakes.

---

# 1.119 What Is a Regression Problem?

A **regression problem** is one in which the machine learning model predicts a **continuous numerical value**.

The output can take any value within a range.

For example,

suppose we want to predict

- hardness,
- density,
- elastic modulus,
- melting temperature,
- thermal conductivity,
- electrical conductivity,
- yield strength,
- bandgap energy.

Each of these properties is measured using numbers.

Therefore,

they are regression problems.

---

# 1.120 Examples of Regression

Suppose a model predicts the hardness of an alloy.

Input

```text
Carbon = 0.6%

Chromium = 11%

Heat Treatment = Quenched
```

Output

```text
Hardness = 518.7 HV
```

Notice that

518.7 is a numerical value.

The prediction could also have been

- 517.2,
- 519.8,
- 520.1,

or any other real number.

There is no fixed set of possible outputs.

---

# 1.121 Mathematical Representation

In regression,

the model learns a function

\[
f(X)=y
\]

where

- \(X\) represents the features,
- \(y\) represents a continuous numerical value.

Examples include

\[
y=425.6
\]

or

\[
y=1387.4
\]

The output belongs to the set of real numbers,

written mathematically as

\[
y\in\mathbb{R}
\]

---

# 1.122 Real Materials Science Regression Problems

Regression is extremely common in materials informatics.

Examples include predicting

| Input Features | Output |
|----------------|--------|
| Composition | Hardness |
| Composition | Yield Strength |
| Crystal Structure | Bandgap |
| Processing Parameters | Tensile Strength |
| Temperature | Thermal Expansion |
| Composition + Processing | Elastic Modulus |

Notice that every output is measured using numbers.

---

# 1.123 What Is a Classification Problem?

A **classification problem** predicts categories rather than numbers.

Instead of asking

> "How much?"

classification asks

> "Which category?"

The model assigns each sample to one of several predefined classes.

---

# 1.124 Examples of Classification

Suppose we wish to classify materials.

Possible outputs include

```text
Metal
```

```text
Semiconductor
```

```text
Insulator
```

The model chooses one category.

It does not predict a numerical value.

---

Another example

Predict whether a material will fail during service.

Possible outputs

```text
Safe
```

or

```text
Unsafe
```

Again,

the outputs are categories.

---

# 1.125 Mathematical Representation

In classification,

the output belongs to a finite collection of classes.

For example,

\[
y\in
\{
Metal,
Semiconductor,
Insulator
\}
\]

Unlike regression,

the model cannot produce intermediate values such as

```text
Metal.73
```

The prediction must belong to one of the predefined categories.

---

# 1.126 Binary Classification

Some classification problems contain only **two classes**.

These are called **binary classification** problems.

Examples include

| Problem | Possible Classes |
|----------|------------------|
| Material Defective? | Yes / No |
| Corrosion Resistant? | Yes / No |
| Battery Safe? | Safe / Unsafe |
| Conductive? | Conductive / Non-Conductive |

Binary classification is one of the most common tasks in machine learning.

---

# 1.127 Multi-Class Classification

Some problems contain more than two classes.

These are called **multi-class classification** problems.

Examples

| Problem | Possible Classes |
|----------|------------------|
| Crystal System | Cubic, Tetragonal, Hexagonal, Orthorhombic, Monoclinic, Triclinic |
| Material Type | Metal, Ceramic, Polymer, Composite |
| Failure Mode | Brittle, Ductile, Fatigue, Creep |

The model selects one class from several possibilities.

---

# 1.128 Regression vs Classification

The following table summarizes the differences.

| Regression | Classification |
|------------|----------------|
| Predicts numbers | Predicts categories |
| Continuous output | Discrete output |
| Example: Hardness | Example: Metal or Ceramic |
| Example: Density | Example: Safe or Unsafe |
| Example: Yield Strength | Example: Corrosion Resistant |

A useful question is

> **Can the output take any numerical value?**

If the answer is **yes**,

the problem is probably regression.

If the answer is **no**,

the problem is probably classification.

---

# 1.129 Can Regression Predictions Be Negative?

Sometimes,

yes.

For example,

predicting

- temperature,
- electrical potential,
- stress,

may legitimately produce negative values.

However,

some quantities such as

- density,
- hardness,
- absolute temperature,

cannot be negative.

The machine learning algorithm itself does not automatically know these physical constraints.

As materials scientists,

we must interpret predictions using scientific knowledge.

This is another reason why domain expertise remains essential.

---

# 1.130 Which Algorithms Solve Which Problems?

Some algorithms are designed specifically for one problem type,

while others have separate versions for both.

| Algorithm | Regression | Classification |
|-----------|:----------:|:--------------:|
| Linear Regression | ✓ | ✗ |
| Logistic Regression | ✗ | ✓ |
| Decision Tree | ✓ | ✓ |
| Random Forest | ✓ | ✓ |
| Support Vector Machine | ✓ | ✓ |
| XGBoost | ✓ | ✓ |
| LightGBM | ✓ | ✓ |
| CatBoost | ✓ | ✓ |
| K-Nearest Neighbors | ✓ | ✓ |

Throughout this book,

we will clearly distinguish between the regression and classification versions whenever applicable.

---

# 1.131 A Practical Example

Suppose a researcher has measured the following properties.

| Density | Grain Size | Carbon (%) | Hardness |
|---------:|-----------:|-----------:|---------:|
| 7.8 | 15 | 0.40 | 320 |
| 7.7 | 18 | 0.35 | 295 |
| 7.9 | 12 | 0.45 | 360 |

If the goal is

> Predict hardness,

this is a **regression** problem because hardness is numerical.

Now consider another dataset.

| Density | Grain Size | Carbon (%) | Material Class |
|---------:|-----------:|-----------:|----------------|
| 7.8 | 15 | 0.40 | Steel |
| 2.7 | 30 | 0.00 | Aluminum |
| 4.5 | 20 | 0.00 | Titanium |

If the goal is

> Predict the material class,

this is a **classification** problem because the output is categorical.

---

# 1.132 Common Beginner Mistakes

### Mistake 1

Using a regression algorithm to predict categories.

Example

Trying to predict

```text
Metal

Ceramic

Polymer
```

using Linear Regression.

This is incorrect.

---

### Mistake 2

Using a classification algorithm to predict continuous values.

Example

Trying to predict

```text
Hardness = 425.8 HV
```

using a classification model.

This is also incorrect.

---

### Mistake 3

Choosing an algorithm before understanding the problem.

Always identify the problem type first.

Only then should you choose the appropriate machine learning algorithm.

---

# 1.133 Python Example: Regression vs Classification

Suppose we have a Pandas DataFrame named `df`.

### Regression Target

```python
X = df[["Density", "Carbon", "Grain_Size"]]

y = df["Hardness"]
```

Here,

`y` contains numerical values,

so this is a regression problem.

---

### Classification Target

```python
X = df[["Density", "Carbon", "Grain_Size"]]

y = df["Material_Class"]
```

Here,

`y` contains category labels,

so this is a classification problem.

Notice that the feature matrix remains the same.

Only the target changes.

This demonstrates that the same dataset can sometimes be used to solve different machine learning problems depending on the prediction objective.

---

The next section introduces your **first complete machine learning program**. You will load a dataset, split it into training and testing sets, train a Linear Regression model using Scikit-learn, make predictions, and understand every single line of code in detail.


# 1.134 Your First Complete Machine Learning Program

Throughout this chapter, we have discussed machine learning from a theoretical perspective.

Now it is time to write our **first complete machine learning program**.

Do not worry if the code looks unfamiliar.

We will explain **every single line**, every function, and every parameter.

Our objective is not simply to run the program.

Our objective is to understand **exactly why each line is written**.

By the end of this section, you will have trained your first machine learning model using **Scikit-learn**, the most widely used machine learning library in Python.

---

# 1.135 The Problem

Suppose we have experimental data for several metallic alloys.

For each alloy we know

- Density
- Carbon percentage

and we want to predict

- Hardness

Since hardness is a continuous numerical value,

this is a **regression problem**.

We will solve it using **Linear Regression**.

---

# 1.136 The Complete Program

```python
# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Create a dataset
data = {
    "Density": [7.85, 7.60, 7.95, 7.70, 7.50, 8.00],
    "Carbon": [0.20, 0.35, 0.50, 0.40, 0.15, 0.60],
    "Hardness": [210, 260, 340, 290, 180, 380]
}

# Step 3: Convert into a DataFrame
df = pd.DataFrame(data)

# Step 4: Select features and target
X = df[["Density", "Carbon"]]
y = df["Hardness"]

# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 6: Create the model
model = LinearRegression()

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Make predictions
predictions = model.predict(X_test)

# Step 9: Display the predictions
print(predictions)
```

Although this program is fewer than 40 lines long,

it contains almost the complete workflow of supervised machine learning.

Now let us understand every line.

---

# 1.137 Step 1 — Import Libraries

```python
import pandas as pd
```

### What is a library?

A **library** is a collection of pre-written code.

Instead of writing everything ourselves,

we use code written and tested by experts.

Think of a library as a toolbox.

Instead of making your own screwdriver,

you simply take one from the toolbox.

---

### What is Pandas?

Pandas is a Python library for working with data.

It allows us to

- read datasets,
- organize tables,
- select columns,
- filter rows,
- clean data.

Almost every machine learning project begins with Pandas.

---

### Why `as pd`?

```python
import pandas as pd
```

The keyword `as` creates an alias.

Instead of writing

```python
pandas.DataFrame(...)
```

we can simply write

```python
pd.DataFrame(...)
```

This saves typing and is the standard convention used throughout the Python community.

---

# 1.138 Importing `train_test_split`

```python
from sklearn.model_selection import train_test_split
```

This line imports the `train_test_split()` function.

Its purpose is to divide a dataset into

- training data,
- testing data.

Instead of writing the splitting algorithm ourselves,

Scikit-learn provides a reliable implementation.

---

# 1.139 Importing Linear Regression

```python
from sklearn.linear_model import LinearRegression
```

This imports the **LinearRegression** class.

Think of a class as a blueprint.

Later,

we will create an actual regression model from this blueprint.

---

# 1.140 Creating the Dataset

```python
data = {
    "Density": [7.85, 7.60, 7.95, 7.70, 7.50, 8.00],
    "Carbon": [0.20, 0.35, 0.50, 0.40, 0.15, 0.60],
    "Hardness": [210, 260, 340, 290, 180, 380]
}
```

This creates a Python **dictionary**.

A dictionary stores information using

```text
Key : Value
```

For example,

```python
"Density"
```

is a key.

Its corresponding values are

```python
[7.85, 7.60, 7.95, ...]
```

Each list becomes one column in the dataset.

---

# 1.141 Converting to a DataFrame

```python
df = pd.DataFrame(data)
```

This converts the dictionary into a Pandas DataFrame.

The result looks like

| Density | Carbon | Hardness |
|---------:|--------:|---------:|
| 7.85 | 0.20 | 210 |
| 7.60 | 0.35 | 260 |
| 7.95 | 0.50 | 340 |
| ... | ... | ... |

A DataFrame is the most common way to store tabular data in Python.

---

# 1.142 Selecting the Features

```python
X = df[["Density", "Carbon"]]
```

This line selects two columns.

These columns become the feature matrix.

Notice the **double square brackets**.

```python
[["Density", "Carbon"]]
```

Why two brackets?

The outer brackets indicate that we are providing a list.

The inner brackets contain the names of the columns.

If we wrote

```python
df["Density"]
```

we would obtain only one column.

If we write

```python
df[["Density", "Carbon"]]
```

we obtain multiple columns.

---

# 1.143 Selecting the Target

```python
y = df["Hardness"]
```

This selects only one column.

The hardness values become the target vector.

The model will learn

```text
Density

Carbon

↓

Hardness
```

---

# 1.144 Splitting the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This function divides the data into

- training features,
- testing features,
- training targets,
- testing targets.

---

### `X`

Contains the input features.

---

### `y`

Contains the target values.

---

### `test_size=0.2`

Twenty percent of the data are reserved for testing.

The remaining eighty percent are used for training.

---

### `random_state=42`

Ensures that the split is reproducible.

Every time the code is executed,

the same random split will be generated.

This is useful for experiments and scientific reproducibility.

---

# 1.145 Creating the Model

```python
model = LinearRegression()
```

This creates a **Linear Regression object**.

Notice that

```python
LinearRegression
```

is the class,

while

```python
LinearRegression()
```

creates an actual model.

Think of it like this.

A blueprint describes a house.

The constructed house is the object created from that blueprint.

---

# 1.146 Training the Model

```python
model.fit(X_train, y_train)
```

This is one of the most important lines in machine learning.

The method

```python
fit()
```

means

> "Learn from the training data."

The model studies

- the input features,
- the target values,

and computes the mathematical relationship between them.

After this line finishes,

the model has learned from the training dataset.

---

# 1.147 Making Predictions

```python
predictions = model.predict(X_test)
```

The method

```python
predict()
```

asks the trained model to estimate the target values for new data.

Notice that we use

```python
X_test
```

and **not**

```python
y_test
```

Why?

Because the purpose of prediction is to estimate unknown values.

The model receives only the features.

It does **not** receive the correct answers.

---

# 1.148 Displaying the Predictions

```python
print(predictions)
```

This displays the predicted hardness values.

For example,

the output might look like

```text
[274.3 352.8]
```

These are the model's estimates.

To know how accurate they are,

we compare them with the actual values stored in

```python
y_test
```

Learning how to evaluate these predictions is the next important step in machine learning.

We will study evaluation metrics in later chapters.

---

# 1.149 What Happened Behind the Scenes?

Although the program appears simple,

the following operations occurred internally.

```text
Create Dataset
       │
       ▼
Select Features (X)
       │
       ▼
Select Target (y)
       │
       ▼
Split Dataset
       │
       ▼
Create Linear Regression Model
       │
       ▼
Train the Model
       │
       ▼
Predict New Values
```

This is the foundation of almost every supervised machine learning project.

Whether you later use

- Decision Trees,
- Random Forests,
- Support Vector Machines,
- XGBoost,
- LightGBM,
- CatBoost,

the overall workflow will remain remarkably similar.

Only the model itself changes.

The surrounding workflow—preparing the data, training, and making predictions—stays largely the same.


# 1.150 Evaluating Your First Machine Learning Model

In the previous section, we trained a Linear Regression model and used it to predict hardness values.

However, simply obtaining predictions is not enough.

We must answer an important question:

> **How good are these predictions?**

A machine learning model is useful only if its predictions are reasonably close to the true values.

Therefore, every machine learning project includes a **model evaluation** step.

---

# 1.151 Comparing Predictions with Actual Values

Suppose our model produced the following predictions.

| Actual Hardness (HV) | Predicted Hardness (HV) |
|----------------------:|------------------------:|
| 210 | 215 |
| 260 | 252 |
| 340 | 348 |
| 290 | 284 |

Notice that the predictions are not exactly equal to the actual values.

This is completely normal.

Machine learning models almost always make some prediction error.

The goal is to make that error as small as possible.

---

# 1.152 What Is Prediction Error?

Prediction error is simply the difference between the actual value and the predicted value.

Mathematically,

$$\text{Error} = \text{Actual Value} - \text{Predicted Value}$$


For the first sample,

Actual = 210

Predicted = 215

Therefore,


$$210 - 215 = -5$$


The negative sign tells us the prediction is slightly higher than the true value.

Often, we are interested only in the **size** of the error, not its direction.

---

# 1.153 Why Can't We Just Count Correct Predictions?

For regression problems, predictions are continuous numbers.

Suppose the true hardness is

```text
300 HV
```

and the model predicts

```text
299 HV
```

Should this be considered incorrect?

What if the prediction is

```text
301 HV?
```

Unlike classification,

there is no simple "correct" or "incorrect."

Instead, we measure **how close** the prediction is to the actual value.

This is why regression uses error metrics instead of accuracy.

---

# 1.154 Mean Absolute Error (MAE)

One of the simplest evaluation metrics is the **Mean Absolute Error (MAE)**.

The word **absolute** means that we ignore positive and negative signs.

For example,

| Actual | Predicted | Error | Absolute Error |
|---------:|----------:|------:|---------------:|
| 210 | 215 | -5 | 5 |
| 260 | 252 | 8 | 8 |
| 340 | 348 | -8 | 8 |
| 290 | 284 | 6 | 6 |

The MAE is simply the average of these absolute errors.


$$
4\text{MAE}
=
\frac{5+8+8+6}{4}
=
6.75
$$


This means that, on average, the model's predictions differ from the true values by **6.75 HV**.

A smaller MAE indicates better predictions.

---

# 1.155 Calculating MAE in Python

Scikit-learn provides a function for calculating MAE.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print(mae)
```

---

## Line-by-Line Explanation

### Line 1

```python
from sklearn.metrics import mean_absolute_error
```

Imports the `mean_absolute_error()` function from Scikit-learn.

This function calculates the average absolute prediction error.

---

### Line 3

```python
mae = mean_absolute_error(y_test, predictions)
```

The function compares

- `y_test` (the true values),
- `predictions` (the model's predicted values).

It computes the Mean Absolute Error and stores the result in the variable `mae`.

---

### Line 5

```python
print(mae)
```

Displays the calculated MAE.

For example,

```text
6.75
```

---

# 1.156 Mean Squared Error (MSE)

Another common evaluation metric is the **Mean Squared Error (MSE)**.

Instead of taking the absolute value,

MSE squares each error.

$$
\text{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

Squaring has an important effect.

Large errors become much larger.

For example,

| Error | Squared Error |
|-------:|--------------:|
| 2 | 4 |
| 5 | 25 |
| 10 | 100 |

This means MSE penalizes large mistakes more heavily than MAE.

---

# 1.157 Calculating MSE in Python

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)

print(mse)
```

The usage is almost identical to MAE.

The only difference is the function being called.

---

# 1.158 Root Mean Squared Error (RMSE)

Although MSE is useful,

its unit is squared.

For hardness,

MSE would have units of

```text
(HV)²
```

This is difficult to interpret.

Taking the square root returns the error to the original units.

$$
\text{RMSE}
=
\sqrt{\text{MSE}}
$$

If hardness is measured in HV,

RMSE is also measured in HV.

Many researchers prefer RMSE because it is easier to interpret.

---

# 1.159 Calculating RMSE in Python

```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(
    y_test,
    predictions,
    squared=False
)

print(rmse)
```

Here,

the argument

```python
squared=False
```

tells Scikit-learn to return the square root of the MSE.

---

# 1.160 Which Metric Should You Use?

Each metric has advantages.

| Metric | Interpretation | Sensitive to Large Errors? |
|--------|----------------|----------------------------|
| MAE | Average error | No |
| MSE | Average squared error | Yes |
| RMSE | Error in original units | Yes |

In materials science,

it is common to report more than one metric.

Doing so provides a more complete picture of model performance.

---

# 1.161 Complete Evaluation Example

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions,
    squared=False
)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
```

This code evaluates the model using three widely used regression metrics.

As you progress through this book, you will encounter these metrics repeatedly when comparing different machine learning algorithms.

---

# 1.162 A Note on Model Performance

Suppose two models produce the following MAE values.

| Model | MAE (HV) |
|--------|---------:|
| Model A | 5.8 |
| Model B | 12.4 |

Model A has the smaller average prediction error.

Therefore,

Model A performs better according to MAE.

However,

it is often wise to examine multiple evaluation metrics before deciding which model is best.

Different metrics emphasize different aspects of prediction quality.

---

In the next section, we will learn how to **save a trained machine learning model and load it later**. This allows you to train a model once and reuse it without repeating the entire training process, an essential skill for real-world machine learning applications.


# 1.163 Saving and Loading Machine Learning Models

Imagine you spent several hours training a machine learning model on a large dataset.

After training finishes,

you close Python.

The next day,

you want to use the trained model to predict the hardness of a new alloy.

Should you train the model again from the beginning?

Fortunately,

the answer is **no**.

Machine learning models can be **saved** to a file and **loaded** later.

This is exactly how machine learning systems are used in real-world applications.

A model is usually trained once and then reused many times.

---

# 1.164 Why Save a Model?

Training a machine learning model can be expensive.

For example,

training may require

- several minutes,
- several hours,
- or even several days,

depending on

- dataset size,
- algorithm complexity,
- computer hardware.

Instead of repeating this process every time,

we save the trained model.

Later,

we simply load it and immediately make predictions.

This saves both time and computational resources.

---

# 1.165 The General Workflow

The workflow becomes

```text
Collect Data

↓

Train Model

↓

Save Model to Disk

↓

Close Python

↓

Open Python Later

↓

Load Saved Model

↓

Predict New Data
```

Notice that the model is trained only once.

---

# 1.166 Saving Models in Python

The most common way to save Scikit-learn models is by using the **joblib** library.

If you do not already have it installed,

you can install it using

```bash
pip install joblib
```

Most Python distributions used for machine learning already include it.

---

# 1.167 Saving a Trained Model

Suppose we have already trained a model.

```python
model.fit(X_train, y_train)
```

Now we save it.

```python
import joblib

joblib.dump(model, "hardness_model.pkl")
```

This creates a file named

```text
hardness_model.pkl
```

on your computer.

The `.pkl` extension stands for **pickle**, a common Python format for storing objects.

---

# 1.168 Line-by-Line Explanation

### Line 1

```python
import joblib
```

Imports the Joblib library.

Joblib specializes in efficiently saving and loading Python objects,

especially machine learning models.

---

### Line 3

```python
joblib.dump(model, "hardness_model.pkl")
```

Let's examine the arguments.

### `model`

This is the trained machine learning model.

It contains everything learned during training,

including its parameters.

---

### `"hardness_model.pkl"`

This is the filename.

You may choose any valid filename.

For example,

```text
linear_model.pkl
```

or

```text
yield_strength_model.pkl
```

The file is saved in the current working directory unless another path is specified.

---

# 1.169 Loading a Saved Model

Suppose we close Python,

return the next day,

and want to continue using the model.

We simply load it.

```python
import joblib

model = joblib.load("hardness_model.pkl")
```

Now the variable `model` contains the trained machine learning model again.

No retraining is necessary.

---

# 1.170 Making Predictions After Loading

Once the model has been loaded,

we use it exactly as before.

```python
new_data = [[7.75, 0.45]]

prediction = model.predict(new_data)

print(prediction)
```

The model immediately predicts the hardness of the new alloy.

Notice that the prediction process is identical to the one used immediately after training.

---

# 1.171 Explaining the Prediction Code

### Line 1

```python
new_data = [[7.75, 0.45]]
```

Creates a new sample.

The values represent

- Density = 7.75
- Carbon = 0.45

Notice the **double square brackets**.

Why?

The model expects a collection of samples.

Even if we have only one sample,

it must still be represented as a two-dimensional structure.

The outer list represents the dataset.

The inner list represents one sample.

---

### Line 3

```python
prediction = model.predict(new_data)
```

Uses the loaded model to estimate the hardness.

Since the model was already trained,

no learning occurs here.

The model simply applies what it learned previously.

---

### Line 5

```python
print(prediction)
```

Displays the predicted hardness.

For example,

```text
[327.6]
```

The brackets indicate that the prediction is returned as a NumPy array.

---

# 1.172 Saving More Than One Model

A project may contain several trained models.

For example,

```text
linear_regression.pkl

decision_tree.pkl

random_forest.pkl

xgboost.pkl
```

Each file stores a different machine learning model.

Later,

we can load whichever model we need.

---

# 1.173 Where Are Models Used?

Saving models is essential in real applications.

Examples include

### Materials Science

A trained model predicts the strength of newly designed alloys.

---

### Manufacturing

A quality-control system predicts defective products.

---

### Healthcare

A diagnostic model assists physicians in identifying diseases.

---

### Finance

A credit-risk model evaluates loan applications.

---

### Autonomous Laboratories

A machine learning model suggests the next experiment without retraining after every decision.

In each case,

the model is trained once,

saved,

and then reused repeatedly.

---

# 1.174 Common Beginner Mistakes

### Mistake 1

Trying to use a model before training it.

Incorrect

```python
model.predict(X_test)
```

before calling

```python
model.fit(...)
```

The model has not learned anything yet.

---

### Mistake 2

Loading the wrong model file.

For example,

accidentally loading

```text
decision_tree.pkl
```

instead of

```text
random_forest.pkl
```

Always use clear filenames.

---

### Mistake 3

Changing the order of the input features.

Suppose the model was trained using

```text
Density

Carbon
```

Later,

you provide

```text
Carbon

Density
```

The model assumes the feature order is unchanged.

Changing the order can produce incorrect predictions.

---

### Mistake 4

Using different preprocessing.

If the training data were scaled or encoded,

the new data must undergo **exactly the same preprocessing** before prediction.

Otherwise,

the predictions may be unreliable.

We will learn how to save preprocessing steps together with the model in later chapters.

---

# 1.175 Complete Example

```python
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Training data
data = {
    "Density": [7.85, 7.60, 7.95, 7.70, 7.50],
    "Carbon": [0.20, 0.35, 0.50, 0.40, 0.15],
    "Hardness": [210, 260, 340, 290, 180]
}

df = pd.DataFrame(data)

X = df[["Density", "Carbon"]]
y = df["Hardness"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "hardness_model.pkl")

# Load model
loaded_model = joblib.load("hardness_model.pkl")

# Predict new sample
new_material = [[7.80, 0.42]]

prediction = loaded_model.predict(new_material)

print(prediction)
```

This short program demonstrates the complete process:

1. Create a dataset.
2. Train a machine learning model.
3. Save the trained model.
4. Load the saved model.
5. Predict the property of a new material.

This workflow is used in countless real-world machine learning applications.

---

The next section concludes Chapter 1 with a **mini end-to-end project**. You will combine everything learned so far—loading data, selecting features, splitting the dataset, training a model, making predictions, evaluating performance, and saving the trained model—into a single complete machine learning pipeline.


# 1.176 Chapter 1 Mini Project — Building Your First Complete Machine Learning Pipeline

Congratulations!

You have now learned the fundamental concepts of machine learning, including

- datasets,
- features,
- targets,
- regression,
- training and testing,
- model evaluation,
- saving trained models.

Now it is time to combine everything into one complete project.

This project follows the same workflow used by data scientists and machine learning engineers.

Don't worry if it seems long.

Every line of code has already been explained in previous sections.

Here, our goal is to see how everything fits together.

---

# 1.177 Project Objective

We want to build a machine learning model that predicts the **hardness of an alloy** using two features:

- Density
- Carbon Percentage

Our workflow will be

```text
Create Dataset

↓

Select Features and Target

↓

Split Dataset

↓

Train Model

↓

Predict Test Data

↓

Evaluate Model

↓

Save Model

↓

Load Model

↓

Predict New Material
```

---

# 1.178 Complete Project Code

```python
# ==========================================
# Step 1: Import libraries
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==========================================
# Step 2: Create dataset
# ==========================================

data = {
    "Density": [7.85, 7.60, 7.95, 7.70, 7.50, 8.00, 7.82, 7.65],
    "Carbon": [0.20, 0.35, 0.50, 0.40, 0.15, 0.60, 0.30, 0.25],
    "Hardness": [210, 260, 340, 290, 180, 380, 245, 230]
}

df = pd.DataFrame(data)

# ==========================================
# Step 3: Select features and target
# ==========================================

X = df[["Density", "Carbon"]]

y = df["Hardness"]

# ==========================================
# Step 4: Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ==========================================
# Step 5: Create model
# ==========================================

model = LinearRegression()

# ==========================================
# Step 6: Train model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# Step 7: Make predictions
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# Step 8: Evaluate model
# ==========================================

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# ==========================================
# Step 9: Save model
# ==========================================

joblib.dump(model, "hardness_model.pkl")

# ==========================================
# Step 10: Load model
# ==========================================

loaded_model = joblib.load("hardness_model.pkl")

# ==========================================
# Step 11: Predict new material
# ==========================================

new_material = [[7.75, 0.42]]

predicted_hardness = loaded_model.predict(new_material)

print("Predicted Hardness:", predicted_hardness)
```

This program represents a complete machine learning workflow.

Every supervised learning project follows approximately the same structure.

---

# 1.179 Understanding the Workflow

Let's review what happened.

### Step 1

```python
import ...
```

We imported all the libraries needed for

- data manipulation,
- model creation,
- evaluation,
- model saving.

---

### Step 2

```python
data = {...}
```

We created our dataset.

Normally, datasets are loaded from CSV files or databases.

Here we created a small dataset manually to keep the example simple.

---

### Step 3

```python
X

y
```

We separated

- the input variables,
- the output variable.

This is required by almost every Scikit-learn algorithm.

---

### Step 4

```python
train_test_split()
```

We divided the dataset into

- training data,
- testing data.

This allows us to evaluate the model fairly.

---

### Step 5

```python
LinearRegression()
```

We created the machine learning model.

At this point,

the model knows nothing.

It has not learned yet.

---

### Step 6

```python
fit()
```

This is where learning occurs.

The algorithm studies

- Density,
- Carbon,

and learns how they relate to Hardness.

---

### Step 7

```python
predict()
```

The trained model estimates hardness values for samples it has never seen before.

---

### Step 8

```python
mean_absolute_error()
```

We compare the predictions with the actual values.

This tells us how well the model performs.

---

### Step 9

```python
joblib.dump()
```

The trained model is saved.

Future programs can reuse it without retraining.

---

### Step 10

```python
joblib.load()
```

The saved model is restored from disk.

Everything learned during training is recovered.

---

### Step 11

```python
predict(new_material)
```

We use the loaded model to predict the hardness of a completely new alloy.

This is the ultimate goal of supervised machine learning.

---

# 1.180 What Happens Inside the Computer?

Although the code looks simple,

many computations happen internally.

```text
Training Data

↓

Linear Regression Algorithm

↓

Find Best Mathematical Relationship

↓

Store Learned Parameters

↓

Receive New Material

↓

Calculate Predicted Hardness
```

The programmer never writes the mathematical equation directly.

The algorithm discovers it automatically.

---

# 1.181 What You Have Learned So Far

By completing Chapter 1, you have already learned skills that beginners often take weeks to understand.

You now know how to

✓ Understand supervised learning

✓ Distinguish regression from classification

✓ Create datasets

✓ Select features and targets

✓ Split datasets

✓ Train machine learning models

✓ Make predictions

✓ Evaluate predictions

✓ Save trained models

✓ Load trained models

These ideas form the foundation of every machine learning project.

Whether you later study

- Decision Trees,
- Random Forests,
- Support Vector Machines,
- Neural Networks,
- XGBoost,
- Graph Neural Networks,

the same overall workflow will still apply.

Only the learning algorithm changes.

---

# 1.182 Exercises

Complete the following exercises without looking at previous examples.

### Exercise 1

Create a dataset with

- Density
- Grain Size
- Hardness

Train a Linear Regression model.

---

### Exercise 2

Change

```python
test_size=0.25
```

to

```python
test_size=0.40
```

Observe how the number of training and testing samples changes.

---

### Exercise 3

Add another feature called

```text
Chromium
```

Train the model again.

Does the prediction change?

---

### Exercise 4

Save the trained model using a different filename.

Load it again.

Verify that the predictions remain the same.

---

### Exercise 5

Create three new alloy samples.

Predict their hardness using your saved model.

---

# 1.183 Chapter 1 Summary

In this chapter, you built a strong foundation in machine learning.

You learned not only the terminology but also how a complete machine learning pipeline works in Python.

Most importantly, you wrote and understood your first end-to-end machine learning program.

From this point onward, the book will focus on individual algorithms in much greater depth.

Every algorithm will be taught using the same approach:

1. Intuition and theory.
2. Mathematical foundation.
3. Advantages and limitations.
4. Step-by-step Python implementation.
5. Line-by-line code explanation.
6. Visualization of results.
7. Materials science examples.
8. Practical exercises.

By the end of the book, you will not only understand how machine learning algorithms work—you will also be able to implement them confidently from scratch using Python and apply them to real materials informatics problems.

---

# Looking Ahead

**Chapter 2: Linear Regression** begins your journey into the first machine learning algorithm.

Unlike the brief introduction in this chapter, Chapter 2 will explore Linear Regression in depth.

You will learn:

- the mathematics behind the algorithm,
- the cost function and loss minimization,
- gradient descent intuition,
- model coefficients and intercepts,
- assumptions of linear regression,
- visualization of fitted lines,
- multiple linear regression,
- performance evaluation,
- complete Python implementations with line-by-line explanations,
- real-world materials science case studies,
- and practical exercises.

By the end of Chapter 2, you won't just know how to use Linear Regression—you will understand exactly how and why it works.
