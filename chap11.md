# Chapter 11 — Dimensionality Reduction and Unsupervised Learning

## Chapter Overview

In the previous chapters, we focused exclusively on **supervised machine learning**, where the objective was to predict a known property from a set of input descriptors. We learned how algorithms such as Linear Regression, Decision Trees, Random Forest, Gradient Boosting, and XGBoost build predictive models for materials properties including formation energy, band gap, elastic modulus, Curie temperature, and many others.

A typical supervised learning workflow looks like this:

```text
Crystal Structure
        │
        ▼
Feature Engineering
(pymatgen + matminer)
        │
        ▼
Descriptor Matrix
        │
        ▼
Machine Learning Model
        │
        ▼
Predicted Property
```

In every supervised learning problem, the dataset contains two essential components:

- A set of **input features (descriptors)**.
- A **known target property (label)**.

For example,

| Material | Descriptors | Target |
|-----------|-------------|---------|
| Silicon | 300 descriptors | Band Gap |
| Fe₂O₃ | 300 descriptors | Formation Energy |
| LiFePO₄ | 300 descriptors | Average Voltage |

The objective is to learn the relationship

$$
X \rightarrow y
$$

where

- **X** is the descriptor matrix.
- **y** is the target property.

However, many problems in materials science do **not** involve predicting a known property.

Instead, researchers often have a large collection of materials with hundreds of computed descriptors but **no labels at all**.

For example, imagine downloading 50,000 crystal structures from the Materials Project and computing hundreds of descriptors using **matminer**.

After feature generation, the dataset may look like this:

| Material | Descriptor 1 | Descriptor 2 | ... | Descriptor 450 |
|-----------|-------------:|-------------:|----:|---------------:|
| Material 1 | ... | ... | ... | ... |
| Material 2 | ... | ... | ... | ... |
| Material 3 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

Notice that no property such as formation energy or band gap is included.

The question is no longer

> "Can we predict a property?"

Instead, the questions become

- Which materials are chemically similar?
- Which materials belong to the same family?
- Can we visualize this enormous dataset?
- Are there hidden groups of materials?
- Which descriptors actually contain useful information?
- Which materials are unusual or outliers?
- Which regions of chemical space remain unexplored?

These are fundamentally different questions.

Rather than making predictions, we are attempting to **discover hidden structure within the data itself**.

This is the goal of **unsupervised learning**.

---

## Why Unsupervised Learning Matters in Materials Informatics

Modern materials informatics rarely works with only a handful of descriptors.

Instead, feature engineering libraries such as **pymatgen** and **matminer** can generate hundreds of numerical features from a single crystal structure.

These descriptors describe various physical, chemical, and structural characteristics of a material, including

- elemental properties,
- crystal symmetry,
- atomic packing,
- bonding environments,
- electronic structure,
- orbital occupations,
- local coordination,
- statistical distributions of atomic properties,
- Voronoi geometry,
- and many other structural descriptors.

For example, combining several common matminer featurizers can easily generate

| Featurizer | Approximate Number of Features |
|------------|-------------------------------:|
| ElementProperty | 100+ |
| Stoichiometry | 20+ |
| DensityFeatures | 5–10 |
| StructuralHeterogeneity | 10+ |
| ChemicalOrdering | 10+ |
| BondFractions | 100+ |
| OrbitalFieldMatrix | 32–100 |
| Site Statistics | 50+ |

As a result, a dataset containing 20,000 materials may contain over 400 descriptors for every material.

Mathematically, the dataset can be represented as

$$
X \in \mathbb{R}^{20000 \times 450}
$$

where

- each row corresponds to one material,
- each column corresponds to one descriptor.

Every material is therefore represented as a single point in a **450-dimensional space**.

---

## The Challenge of High-Dimensional Data

Working with hundreds of descriptors presents several challenges.

First, humans cannot visualize more than three dimensions.

If every material is described using 450 numerical variables, there is no direct way to understand how materials are distributed throughout the dataset.

Second, many descriptors are strongly correlated.

For example,

- atomic number and atomic mass tend to increase together,
- atomic radius and covalent radius often follow similar trends,
- several electronegativity descriptors contain overlapping information,
- multiple structural descriptors may describe the same underlying geometry.

Consequently, many descriptors provide redundant information.

Third, machine learning algorithms often become less efficient as the number of descriptors increases.

High-dimensional datasets typically require

- more memory,
- longer training times,
- more computational resources,
- and larger datasets to avoid overfitting.

Finally, understanding the relationships between hundreds of descriptors becomes nearly impossible through manual inspection alone.

---

## The Goal of Dimensionality Reduction

Dimensionality reduction attempts to solve these problems by transforming a large set of descriptors into a much smaller set of variables while preserving the essential information contained within the original dataset.

Instead of describing every material using hundreds of descriptors,

```text
450 Original Descriptors
           │
           ▼
Dimensionality Reduction
           │
           ▼
10–20 Informative Components
```

the algorithm constructs a lower-dimensional representation that captures the dominant patterns present in the data.

Ideally, these new variables preserve

- the overall structure of the dataset,
- similarities between materials,
- important physical trends,
- and the majority of the information contained in the original descriptors.

This allows researchers to visualize and analyze extremely complex datasets that would otherwise be impossible to interpret.

---

## Principal Component Analysis (PCA)

The first dimensionality reduction technique studied in this chapter is **Principal Component Analysis (PCA)**.

PCA is one of the most widely used statistical techniques in science, engineering, physics, chemistry, biology, and materials science.

Its objective is remarkably simple:

> Find a new coordinate system that captures the maximum amount of variation in the dataset using as few variables as possible.

Instead of using the original descriptors,

```text
x₁
x₂
x₃
⋮
x₄₅₀
```

PCA creates a new set of variables called **principal components**.

```text
PC₁
PC₂
PC₃
⋮
```

These principal components are combinations of the original descriptors and are ordered according to the amount of information they capture.

Typically,

- PC₁ captures the largest variation,
- PC₂ captures the second largest variation,
- PC₃ captures the third largest variation,
- and so on.

Very often,

the first two or three principal components already explain most of the variability within a materials dataset.

This allows hundreds of descriptors to be visualized in only two dimensions.

```text
Hundreds of Descriptors
           │
           ▼
          PCA
           │
           ▼
      PC₁ and PC₂
           │
           ▼
Visualization of Materials Space
```

---

## Clustering

Dimensionality reduction tells us **how materials are distributed**.

Clustering goes one step further by automatically identifying groups of similar materials.

Unlike classification,

where predefined labels already exist,

clustering attempts to discover groups directly from the data.

For example,

instead of telling the algorithm

- these are oxides,
- these are nitrides,
- these are Heusler alloys,

the algorithm examines the descriptors and identifies groups entirely on its own.

Materials that are chemically or structurally similar naturally appear in the same cluster.

Throughout this chapter, we will study three widely used clustering algorithms:

- **K-Means**
- **Hierarchical Clustering**
- **DBSCAN**

Each algorithm approaches clustering differently and is suitable for different types of materials datasets.

---

## Applications in Materials Science

The combination of dimensionality reduction and clustering has become one of the most powerful exploratory tools in modern materials informatics.

Typical applications include

- visualizing chemical space,
- discovering previously unknown material families,
- identifying structurally similar compounds,
- grouping crystal structures,
- detecting anomalous materials,
- exploring high-throughput computational databases,
- accelerating materials discovery,
- selecting representative training datasets,
- and understanding descriptor relationships.

These methods are widely applied to datasets obtained from

- Materials Project,
- OQMD,
- AFLOW,
- NOMAD,
- JARVIS,
- and many other computational materials databases.

---

## What You Will Learn in This Chapter

This chapter is organized into two major parts.

The first part introduces **Principal Component Analysis (PCA)**.

We begin with the mathematical concepts that make PCA possible, including

- variance,
- eigenvectors,
- eigenvalues,
- and projection.

Once these concepts are understood, we apply PCA to real descriptor datasets generated using **pymatgen** and **matminer** to visualize high-dimensional materials spaces.

The second part introduces **clustering algorithms**, including

- K-Means,
- Hierarchical Clustering,
- and DBSCAN.

Finally, we demonstrate how these techniques are used to identify material families, explore chemical space, and reveal hidden structures within large materials databases.

By the end of this chapter, you will understand not only how these algorithms work mathematically, but also how they are applied in real-world materials informatics research to transform hundreds of descriptors into meaningful scientific insight.

## 11.2 Principal Component Analysis (PCA)

Among all dimensionality reduction techniques, **Principal Component Analysis (PCA)** is by far the most widely used and one of the most important algorithms in scientific data analysis. Since its introduction by Karl Pearson in 1901, PCA has become a standard tool in statistics, physics, chemistry, biology, image processing, finance, and more recently, materials informatics.

In materials science, datasets frequently contain hundreds of descriptors for every material. Although these descriptors provide valuable information, they also create several practical and computational challenges. PCA addresses these challenges by transforming a high-dimensional dataset into a new coordinate system where most of the important information is concentrated into only a few variables.

Rather than analyzing hundreds of descriptors independently, PCA identifies the most informative directions within the dataset and expresses every material using these new directions.

---

### The Central Idea Behind PCA

Suppose we generate descriptors for thousands of materials using **matminer**.

Our dataset might contain

- elemental descriptors,
- structural descriptors,
- bonding descriptors,
- electronic descriptors,
- geometric descriptors,
- thermodynamic descriptors.

Altogether, each material may be represented using hundreds of numerical values.

```text
Material 1 → [x₁ x₂ x₃ ... x₄₅₀]

Material 2 → [x₁ x₂ x₃ ... x₄₅₀]

Material 3 → [x₁ x₂ x₃ ... x₄₅₀]

...

Material N → [x₁ x₂ x₃ ... x₄₅₀]
```

Mathematically, the dataset is written as

$$
X \in \mathbb{R}^{N \times M}
$$

where

- **N** is the number of materials.
- **M** is the number of descriptors.

For example,

$$
X \in \mathbb{R}^{20000 \times 450}
$$

means

- 20,000 materials,
- 450 descriptors for each material.

Although the dataset contains 450 features, not all of them provide independent information.

Many descriptors are correlated.

For example,

- atomic number and atomic mass increase together,
- density and unit-cell volume are often related,
- several electronegativity descriptors measure similar chemical behavior,
- different radius descriptors frequently exhibit similar trends.

As a result, much of the information is duplicated.

Instead of storing the same information repeatedly, PCA compresses these correlated descriptors into a much smaller number of new variables called **principal components**.

---

### A Change of Perspective

One of the biggest misconceptions about PCA is that it removes descriptors.

It does **not**.

Instead, PCA changes the coordinate system used to describe the data.

Imagine looking at a crystal from different viewing directions.

The crystal itself never changes.

Only your viewpoint changes.

```text
           Crystal

      View 1  →  □

      View 2  →  ◇

      View 3  →  ▱
```

Each view contains the same object but reveals different information.

PCA works in exactly the same way.

The dataset remains unchanged.

Only the coordinate axes are rotated until they align with the directions containing the greatest variation.

Instead of describing a material using

```text
Density

Electronegativity

Atomic Radius

Packing Fraction

...

Descriptor 450
```

PCA describes the same material using

```text
PC₁

PC₂

PC₃

...

PC₄₅₀
```

The principal components are simply a better coordinate system for describing the same data.

---

### Why Is This Useful?

Suppose two descriptors contain almost identical information.

```text
Average Atomic Number

Average Atomic Mass
```

These descriptors are highly correlated because heavier elements generally have larger atomic numbers.

Keeping both descriptors contributes relatively little new information.

PCA recognizes this redundancy.

Instead of storing both variables separately, PCA combines them into a single direction representing their shared trend.

The same process occurs for hundreds of descriptors simultaneously.

Eventually,

450 correlated descriptors

may become

20 informative principal components

without losing much information.

---

### Principal Components

A **principal component** is a new variable constructed as a weighted combination of the original descriptors.

For example,

the first principal component is

$$
PC_1 = w_1x_1 + w_2x_2 + \cdots + w_Mx_M
$$

where

- **x₁, x₂, ..., xₘ** are the original descriptors,
- **w₁, w₂, ..., wₘ** are numerical weights learned by PCA.

These weights determine how strongly each descriptor contributes to the principal component.

Unlike the original descriptors,

principal components are

- independent,
- orthogonal,
- ordered by importance.

The first component captures the greatest amount of variation.

The second captures the greatest remaining variation while remaining orthogonal to the first.

The third captures the next largest variation while remaining orthogonal to both previous components.

This process continues until every descriptor has been transformed into a principal component.

---

### The PCA Workflow

The complete PCA workflow consists of several mathematical steps.

```text
Original Descriptor Matrix
          │
          ▼
Feature Scaling
          │
          ▼
Compute Variance
          │
          ▼
Compute Covariance Matrix
          │
          ▼
Find Eigenvalues
and Eigenvectors
          │
          ▼
Construct Principal Components
          │
          ▼
Project Data
          │
          ▼
Low-Dimensional Representation
```

Each step has a precise mathematical purpose.

During the next sections of this chapter, we will study every step in detail.

---

### Mathematical Roadmap

To fully understand PCA, we must first understand several fundamental concepts from statistics and linear algebra.

These concepts are introduced gradually throughout the next sections.

1. Variance
2. Covariance
3. Covariance Matrix
4. Eigenvectors
5. Eigenvalues
6. Principal Components
7. Projection
8. Explained Variance

Each concept builds upon the previous one.

For example,

without understanding variance,

it is impossible to understand covariance.

Without covariance,

the covariance matrix cannot be constructed.

Without the covariance matrix,

eigenvectors and eigenvalues cannot be computed.

Without eigenvectors,

principal components do not exist.

Therefore, although PCA appears to be a single algorithm, it is actually a beautiful combination of statistics and linear algebra.

---

### PCA in Materials Informatics

PCA has become one of the most widely used exploratory techniques in computational materials science.

Typical applications include

- visualization of high-dimensional descriptor datasets,
- identifying chemically similar materials,
- reducing descriptor redundancy,
- preprocessing data before machine learning,
- detecting outliers,
- understanding descriptor relationships,
- exploring crystal structure databases,
- selecting representative training data,
- accelerating feature engineering.

A common workflow is

```text
Crystal Structures
        │
        ▼
pymatgen
        │
        ▼
matminer Featurizers
        │
        ▼
450–1000 Descriptors
        │
        ▼
Principal Component Analysis
        │
        ▼
PC₁, PC₂
        │
        ▼
2D Visualization of Materials Space
```

After PCA, researchers can often observe that

- oxides occupy one region,
- nitrides occupy another,
- intermetallic compounds form separate clusters,
- battery materials group together,
- superconductors occupy distinct regions,
- and anomalous materials appear as isolated points.

This provides valuable scientific insight long before any supervised machine learning model is trained.

---

In the next section, we begin the mathematical foundation of PCA by studying the most fundamental quantity in statistics and machine learning:

**variance**.

## 11.2 Variance and Covariance — The Mathematical Foundation of PCA

Before we can understand **Principal Component Analysis (PCA)**, we must first understand the statistical concepts upon which it is built. Although PCA is often introduced as a machine learning algorithm, it is fundamentally a statistical technique that uses ideas from linear algebra to analyze the structure of data.

Every step of PCA is driven by one central question:

> **How does the data vary?**

If there were no variation in the data, there would be nothing to learn, nothing to compress, and nothing to visualize. Consequently, the entire PCA algorithm begins by measuring variation within the dataset.

This section introduces the two most important statistical quantities used in PCA:

- **Variance**
- **Covariance**

Variance measures how much a single variable changes.

Covariance measures how two variables change together.

These two concepts eventually lead to the construction of the **covariance matrix**, from which PCA derives its principal components.

---

## Why Does PCA Care About Variation?

Imagine collecting information about several thousand materials from the Materials Project.

For each material, you compute descriptors using **matminer**.

The resulting dataset might look like this:

| Material | Density | Atomic Radius | Electronegativity | Volume | Band Center | ... |
|-----------|---------|---------------|-------------------|---------|-------------|-----|
| Material 1 | ... | ... | ... | ... | ... | ... |
| Material 2 | ... | ... | ... | ... | ... | ... |
| Material 3 | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

Each column represents a descriptor.

Some descriptors vary significantly across materials.

Others barely change.

Suppose one descriptor has almost the same value for every material.

```text
5.01
5.00
5.02
5.01
5.00
5.01
5.02
```

This descriptor contains very little information because it barely distinguishes one material from another.

Now consider another descriptor.

```text
0.8
2.1
4.7
6.3
8.5
9.1
11.4
```

Here the values spread over a much larger range.

This descriptor contains considerably more information because it separates materials more effectively.

PCA therefore asks a very natural question:

> **Which directions in the dataset contain the greatest variation?**

To answer that question, we must first define what "variation" actually means.

---

# Understanding Variation

Variation simply describes how much numerical values differ from one another.

If every observation is identical,

```text
5
5
5
5
5
5
```

there is **no variation**.

Every sample is exactly the same.

On the other hand,

```text
1
4
7
9
12
15
```

shows substantial variation because the observations are spread over a wide range.

Variation is therefore a measure of **spread**.

The greater the spread,

the larger the variation.

---

## Why Variation Matters in Machine Learning

Machine learning algorithms learn by finding differences between samples.

Suppose every material in a dataset had exactly the same density.

```text
Density

7.84
7.84
7.84
7.84
7.84
```

Density would provide no useful information.

Knowing the density would not help distinguish one material from another.

Now consider a second dataset.

```text
Density

2.5
3.7
5.9
7.8
10.6
12.4
```

Density now varies substantially.

Different materials possess different densities.

This descriptor now carries useful information.

In general,

descriptors with larger variation tend to contain more information than descriptors that remain nearly constant.

This is one of the key ideas behind PCA.

---

# Measuring Variation

Simply looking at numbers is not enough.

We need a mathematical quantity that measures how spread out a variable is.

That quantity is called the **variance**.

Variance measures

> **How far observations are distributed around their average value.**

---

## Step 1 — Compute the Mean

Before measuring variation,

we first determine the center of the data.

The center is represented by the arithmetic mean.

For a variable

$$
x_1,x_2,x_3,\ldots,x_N
$$

the mean is

$$
\mu=\frac{1}{N}\sum_{i=1}^{N}x_i
$$

where

- **N** is the number of observations.
- **μ** represents the average value.

---

### Example

Suppose five materials have densities

| Material | Density (g/cm³) |
|-----------|----------------:|
| A | 4 |
| B | 5 |
| C | 6 |
| D | 7 |
| E | 8 |

The average density is

$$
\mu=\frac{4+5+6+7+8}{5}=6
$$

The mean therefore represents the central value around which the observations are distributed.

---

## Step 2 — Measure the Distance from the Mean

Knowing the mean is not sufficient.

We want to know how far each observation lies from the average.

Subtract the mean from every observation.

| Density | Mean | Difference |
|---------:|-----:|-----------:|
| 4 | 6 | -2 |
| 5 | 6 | -1 |
| 6 | 6 | 0 |
| 7 | 6 | 1 |
| 8 | 6 | 2 |

These differences are called **deviations from the mean**.

Negative values indicate observations below the average.

Positive values indicate observations above the average.

---

## Why Can't We Simply Add the Deviations?

Suppose we add all the deviations together.

$$
(-2)+(-1)+0+1+2=0
$$

The answer is always zero.

This happens because positive and negative deviations cancel each other.

Therefore,

simply summing deviations tells us nothing about the spread of the data.

We need another approach.

---

## Squaring the Deviations

Instead of adding the deviations directly,

we square them.

| Difference | Squared Difference |
|------------|-------------------:|
| -2 | 4 |
| -1 | 1 |
| 0 | 0 |
| 1 | 1 |
| 2 | 4 |

Squaring has two important effects.

### First

Negative numbers become positive.

Therefore,

positive and negative deviations no longer cancel.

### Second

Large deviations become much larger.

For example,

| Difference | Squared Value |
|------------|--------------:|
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 5 | 25 |

Consequently,

observations that lie far from the mean contribute much more strongly to the variance.

This makes variance highly sensitive to unusually large deviations.

---

## Definition of Variance

The variance of a variable is the average of the squared deviations from the mean.

For an entire population,

the variance is

$$
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
$$

where

- **σ²** is the variance,
- **μ** is the mean,
- **N** is the number of observations.

When working with a sample rather than an entire population, we usually divide by

$$
N-1
$$

instead of

$$
N.
$$

Thus,

the sample variance becomes

$$
s^2=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\mu)^2
$$

Since most machine learning datasets are considered samples,

this is the form commonly used in practice.

---

## Computing Variance Step by Step

Returning to our density example,

| Density | Difference | Squared Difference |
|---------:|-----------:|-------------------:|
| 4 | -2 | 4 |
| 5 | -1 | 1 |
| 6 | 0 | 0 |
| 7 | 1 | 1 |
| 8 | 2 | 4 |

The sum of squared deviations is

$$
4+1+0+1+4=10
$$

The sample variance is

$$
s^2=\frac{10}{5-1}=2.5
$$

The larger this number,

the more spread out the observations are.

---

# Interpreting Variance

Variance always satisfies

$$
Variance \ge 0
$$

because squared numbers can never be negative.

Different values of variance convey different meanings.

### Small Variance

```text
5.1
5.0
5.2
5.1
5.0
```

All observations lie close together.

The descriptor changes very little.

---

### Large Variance

```text
1
5
9
14
21
```

The observations are widely dispersed.

The descriptor changes significantly across materials.

---

## Variance in Descriptor Space

Imagine calculating the variance of every descriptor produced by **matminer**.

```text
Density                 Variance = 1.9

Atomic Radius           Variance = 0.4

Electronegativity       Variance = 2.7

Packing Fraction        Variance = 0.1

Band Center             Variance = 5.6

...
```

Descriptors with extremely small variance contribute very little information because they barely change from one material to another.

Conversely,

descriptors with large variance often contain richer information about differences between materials.

However,

variance alone is not enough.

Knowing how each descriptor varies individually tells us nothing about how descriptors are related to one another.

For example,

suppose density increases whenever atomic radius increases.

Or perhaps density decreases whenever electronegativity increases.

Variance cannot describe these relationships.

To understand how two variables change together,

we introduce the second fundamental concept of PCA:

**covariance**.

Covariance reveals whether two descriptors tend to increase together, decrease together, or vary independently. As we will see in the next section, the covariance matrix constructed from these pairwise relationships forms the mathematical foundation upon which the entire PCA algorithm is built.

## 11.3 Covariance — Measuring Relationships Between Variables

In the previous section, we learned that **variance** measures how much a **single variable** changes. Variance tells us whether a descriptor contains significant information by quantifying how widely its values are distributed around the mean.

However, PCA is not interested only in how individual descriptors vary.

It is equally interested in **how different descriptors vary together**.

Consider two descriptors commonly encountered in materials informatics.

- Average atomic number
- Average atomic mass

Both descriptors generally increase together because heavier elements usually have larger atomic numbers.

Now consider another pair of descriptors.

- Density
- Unit cell volume

Depending on the class of materials, these descriptors may show either a positive or a negative relationship.

If two descriptors exhibit similar trends, they likely contain overlapping information. If they behave independently, they provide different types of information.

To measure these relationships quantitatively, we introduce **covariance**.

Covariance is one of the most important concepts in statistics and forms the mathematical backbone of Principal Component Analysis.

---

# From Variance to Covariance

Variance answers the question

> **How much does one variable vary?**

Covariance answers a different question

> **How do two variables vary together?**

Suppose we have two descriptors,

- Density
- Average electronegativity

Instead of analyzing them separately, we now ask

- Do they increase together?
- Does one increase while the other decreases?
- Are they completely unrelated?

Covariance provides the answer.

---

# Understanding Covariance Intuitively

Imagine observing two quantities simultaneously.

Suppose every time one quantity increases, the other also increases.

```text
Variable X

1
2
3
4
5

Variable Y

2
4
6
8
10
```

The two variables clearly move together.

When **X increases**, **Y also increases**.

This is an example of **positive covariance**.

---

Now consider another dataset.

```text
Variable X

1
2
3
4
5

Variable Y

10
8
6
4
2
```

As **X increases**, **Y decreases**.

The two variables move in opposite directions.

This is an example of **negative covariance**.

---

Finally, consider

```text
Variable X

1
2
3
4
5

Variable Y

7
2
9
3
6
```

There is no obvious relationship.

Sometimes both increase.

Sometimes one decreases.

Sometimes nothing consistent happens.

This corresponds to covariance near zero.

---

Therefore,

covariance tells us whether two variables

- increase together,
- decrease together,
- or behave independently.

---

# A Materials Science Example

Consider several metallic elements.

| Element | Atomic Number | Atomic Mass |
|----------|--------------:|------------:|
| Al | 13 | 26.98 |
| Ti | 22 | 47.87 |
| Fe | 26 | 55.85 |
| Cu | 29 | 63.55 |
| Ag | 47 | 107.87 |

As atomic number increases,

atomic mass also increases.

The two descriptors exhibit a strong positive relationship.

Consequently,

their covariance is positive.

---

Now consider another hypothetical dataset.

| Material | Density | Lattice Constant |
|----------|--------:|-----------------:|
| A | 9.2 | 3.1 |
| B | 8.4 | 3.4 |
| C | 7.3 | 3.8 |
| D | 6.1 | 4.2 |
| E | 5.4 | 4.8 |

Here,

larger lattice constants correspond to lower densities.

The descriptors move in opposite directions.

Their covariance is negative.

---

# Why PCA Needs Covariance

Suppose we generated 450 descriptors using matminer.

Some descriptors may contain almost identical information.

For example,

```text
Average Atomic Number

Average Atomic Mass

Average Nuclear Charge
```

All three tend to increase together.

If PCA treated them as completely independent descriptors, it would count the same information multiple times.

Instead,

PCA measures the covariance between every pair of descriptors.

Highly correlated descriptors are recognized as carrying similar information and are combined into common directions known as principal components.

Without covariance,

PCA would not know which descriptors are redundant.

---

# Computing Covariance

The computation of covariance is remarkably similar to that of variance.

Recall that variance measures the squared deviation from the mean.

Instead of squaring the deviation of one variable,

covariance multiplies the deviations of **two different variables**.

Suppose we have two variables,

$$
X=(x_1,x_2,\ldots,x_N)
$$

and

$$
Y=(y_1,y_2,\ldots,y_N)
$$

First compute their means.

$$
\mu_X=\frac{1}{N}\sum_{i=1}^{N}x_i
$$

$$
\mu_Y=\frac{1}{N}\sum_{i=1}^{N}y_i
$$

Then compute the covariance.

For a sample,

$$
Cov(X,Y)=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\mu_X)(y_i-\mu_Y)
$$

Notice how similar this equation is to the variance formula.

The only difference is that

instead of multiplying

$$
(x_i-\mu)^2
$$

we multiply

$$
(x_i-\mu_X)
$$

by

$$
(y_i-\mu_Y)
$$

This simple change allows us to measure how two variables move together.

---

# Understanding the Formula

The covariance equation may appear complicated initially, but each part has a clear interpretation.

### Step 1

Compute the average of each variable.

This establishes the center of both datasets.

---

### Step 2

Measure how far every observation lies from its average.

Positive deviations indicate observations above the mean.

Negative deviations indicate observations below the mean.

---

### Step 3

Multiply the deviations.

This multiplication is the key idea behind covariance.

Consider four possible situations.

| X Deviation | Y Deviation | Product |
|-------------|------------|---------|
| Positive | Positive | Positive |
| Negative | Negative | Positive |
| Positive | Negative | Negative |
| Negative | Positive | Negative |

Notice an important pattern.

If both variables move in the same direction,

their product is positive.

If they move in opposite directions,

their product is negative.

Therefore,

the sign of the covariance naturally tells us whether the variables move together or apart.

---

# Example Calculation

Suppose we have the following data.

| Sample | X | Y |
|---------|--:|--:|
| 1 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 3 | 6 |
| 4 | 4 | 8 |
| 5 | 5 | 10 |

The averages are

$$
\mu_X=3
$$

$$
\mu_Y=6
$$

Now compute the deviations.

| X | X−μₓ | Y | Y−μᵧ | Product |
|--:|------:|--:|------:|--------:|
|1|-2|2|-4|8|
|2|-1|4|-2|2|
|3|0|6|0|0|
|4|1|8|2|2|
|5|2|10|4|8|

The sum of the products is

$$
8+2+0+2+8=20
$$

Since

$$
N=5
$$

the sample covariance is

$$
Cov(X,Y)=\frac{20}{4}=5
$$

The covariance is positive,

indicating that both variables increase together.

---

# Negative Covariance

Now consider

| X | Y |
|--:|--:|
|1|10|
|2|8|
|3|6|
|4|4|
|5|2|

When X is above its average,

Y is below its average.

Consequently,

many products become negative.

The covariance is therefore negative,

indicating an inverse relationship.

---

# Zero Covariance

Suppose two descriptors behave completely independently.

```text
Density

7.2
6.5
9.1
8.3
7.4

Band Gap

3.8
1.2
5.4
2.7
4.1
```

If no consistent relationship exists,

positive and negative products tend to cancel.

The covariance approaches zero.

A covariance near zero suggests that the two variables do not exhibit a linear relationship.

---

# Interpreting Covariance

The sign of the covariance is far more important than its numerical magnitude.

| Covariance | Interpretation |
|------------|----------------|
| Positive | Variables increase together. |
| Negative | One variable increases while the other decreases. |
| Approximately Zero | No significant linear relationship exists. |

One important limitation should be noted.

The numerical value of covariance depends on the units of the variables.

For example,

- density may be measured in g/cm³,
- atomic radius in Å,
- elastic modulus in GPa.

Changing the units changes the magnitude of the covariance.

Therefore,

covariance values from different descriptor pairs cannot be compared directly.

This limitation is one reason why descriptor scaling is usually performed before PCA.

---

# From Covariance to the Covariance Matrix

Calculating the covariance between two descriptors is useful,

but materials datasets often contain hundreds of descriptors.

Suppose a dataset contains

- Density
- Volume
- Atomic Radius
- Electronegativity
- Packing Fraction
- Formation Energy
- Band Center
- Orbital Occupancy
- ...

Instead of computing one covariance,

we compute **every possible pair**.

The results are organized into a square matrix called the **covariance matrix**.

```text
                Density  Radius  EN  Volume  ...

Density            ●

Radius             ●       ●

EN                 ●       ●    ●

Volume             ●       ●    ●      ●

...
```

Every element of this matrix represents the covariance between two descriptors.

The covariance matrix summarizes the relationships among **all descriptors simultaneously**.

It is one of the most information-rich objects in statistics.

More importantly,

it serves as the direct input to the next stage of PCA.

Once the covariance matrix has been constructed, PCA performs an eigenvalue decomposition to determine the directions along which the dataset varies the most.

Those directions become the **principal components**.

In the next section, we will study the **covariance matrix** in detail and understand why it is the starting point for discovering the principal axes of variation in high-dimensional materials datasets.

## 11.4 The Covariance Matrix — Capturing the Structure of the Entire Dataset

In the previous section, we learned that covariance measures the relationship between **two variables**. If the covariance is positive, the variables tend to increase together. If it is negative, one variable tends to increase while the other decreases. If the covariance is close to zero, the variables are approximately independent in a linear sense.

However, real materials informatics datasets rarely contain only two descriptors.

A typical dataset generated using **pymatgen** and **matminer** may contain

- 150 descriptors,
- 300 descriptors,
- 500 descriptors,
- or even more than 1,000 descriptors.

Suppose our dataset contains 450 descriptors.

We could compute the covariance between

- Density and Atomic Radius,
- Density and Electronegativity,
- Density and Packing Fraction,
- Density and Volume,
- ...

But after computing one covariance, we still have hundreds of descriptor pairs remaining.

PCA therefore needs a way to summarize **all pairwise relationships simultaneously**.

This is accomplished using the **covariance matrix**.

The covariance matrix is one of the most important objects in statistics, data science, and machine learning because it describes the entire structure of the dataset.

---

# Why Do We Need a Matrix?

Imagine a dataset containing only four descriptors.

- Density
- Atomic Radius
- Electronegativity
- Unit Cell Volume

The possible covariance calculations are

```text
Density ↔ Density

Density ↔ Atomic Radius

Density ↔ Electronegativity

Density ↔ Volume

Atomic Radius ↔ Electronegativity

Atomic Radius ↔ Volume

Electronegativity ↔ Volume

...
```

Even with only four descriptors, several covariance values must be calculated.

Now imagine a dataset with

450 descriptors.

The number of pairwise relationships becomes enormous.

Instead of storing these values separately,

PCA organizes them into a single square matrix.

---

# Structure of the Covariance Matrix

Suppose our dataset contains three descriptors.

- Density
- Atomic Radius
- Electronegativity

The covariance matrix looks like

```text
                     Density   Radius   Electronegativity

Density            Cov(D,D)   Cov(D,R)   Cov(D,E)

Radius             Cov(R,D)   Cov(R,R)   Cov(R,E)

Electronegativity  Cov(E,D)   Cov(E,R)   Cov(E,E)
```

Every row represents one descriptor.

Every column represents another descriptor.

Each cell contains the covariance between the corresponding pair.

For a dataset containing **M descriptors**, the covariance matrix has dimensions

$$
M \times M
$$

Therefore,

if our dataset contains

450 descriptors,

the covariance matrix has dimensions

$$
450 \times 450
$$

---

# Understanding the Diagonal

One of the first things to notice is the main diagonal.

```text
                     Density   Radius   EN

Density              ★

Radius                        ★

EN                                   ★
```

Each diagonal element represents

```text
Cov(X,X)
```

But what happens if we compute the covariance of a variable with itself?

Recall the covariance equation.

$$
Cov(X,Y)=\frac{1}{N-1}\sum(x_i-\mu_X)(y_i-\mu_Y)
$$

If

$$
X=Y
$$

then the equation becomes

$$
Cov(X,X)=\frac{1}{N-1}\sum(x_i-\mu)^2
$$

This is exactly the definition of **variance**.

Therefore,

the diagonal elements of the covariance matrix are simply the variances of each descriptor.

For example,

```text
                Density   Radius   EN

Density          Var(D)

Radius                     Var(R)

EN                                  Var(E)
```

This is an important observation.

**Every covariance matrix automatically contains the variance of every descriptor.**

---

# Understanding the Off-Diagonal Elements

The remaining elements describe relationships between different descriptors.

For example,

```text
Cov(Density, Radius)
```

tells us whether density and atomic radius increase together.

Similarly,

```text
Cov(Radius, EN)
```

describes the relationship between atomic radius and electronegativity.

These off-diagonal elements contain the information PCA uses to discover redundant descriptors.

---

# Example Covariance Matrix

Suppose the covariance matrix of a small dataset is

```text
                 Density   Radius   EN

Density            2.8      1.9    -0.3

Radius             1.9      1.5    -0.2

EN                -0.3     -0.2     0.8
```

Notice several interesting observations.

### Density and Radius

The covariance is

```text
1.9
```

This positive value indicates that these descriptors generally increase together.

---

### Density and Electronegativity

The covariance is

```text
-0.3
```

The negative sign indicates an inverse relationship.

---

### Radius and Electronegativity

Again,

the covariance is negative.

These descriptors tend to move in opposite directions.

---

### Diagonal

The diagonal entries

```text
2.8

1.5

0.8
```

represent the variances of

- Density,
- Atomic Radius,
- Electronegativity.

---

# The Covariance Matrix Is Symmetric

One remarkable property of the covariance matrix is that it is always **symmetric**.

Why?

Because

```text
Cov(X,Y)=Cov(Y,X)
```

Changing the order of the variables does not change the covariance.

Therefore,

```text
Covariance Matrix

        Density Radius EN

Density    ●      ●     ●

Radius     ●      ●     ●

EN         ●      ●     ●
```

is identical above and below the diagonal.

Mathematically,

$$
C=C^T
$$

where

- **C** is the covariance matrix,
- **Cᵀ** is its transpose.

This symmetry is extremely important because symmetric matrices possess several beautiful mathematical properties.

In particular,

they always have

- real eigenvalues,
- orthogonal eigenvectors.

These properties make PCA mathematically possible.

---

# Computing the Covariance Matrix

Suppose our original descriptor matrix is

```text
                Density Radius EN

Material 1

Material 2

Material 3

...

Material N
```

The computation proceeds through several steps.

### Step 1

Compute the mean of every descriptor.

---

### Step 2

Subtract the mean from every observation.

This process is called **centering the data**.

After centering,

every descriptor has an average value of zero.

---

### Step 3

Compute the covariance between every pair of descriptors.

---

### Step 4

Arrange all covariance values into the covariance matrix.

This matrix now summarizes the statistical relationships among every descriptor in the dataset.

---

# Matrix Representation

Although covariance can be computed pair by pair,

matrix algebra provides a much more elegant formulation.

Suppose

- **X** is the centered data matrix,
- **N** is the number of samples.

The covariance matrix is

$$
C=\frac{1}{N-1}X^TX
$$

This equation appears frequently in machine learning, statistics, and linear algebra.

Understanding each component is important.

### X

The centered descriptor matrix.

Every column has zero mean.

---

### Xᵀ

The transpose of the descriptor matrix.

Rows become columns.

Columns become rows.

---

### XᵀX

Computes every pairwise covariance simultaneously.

Instead of calculating hundreds of covariance values individually,

matrix multiplication computes them all in one operation.

---

### Dividing by N−1

Produces the sample covariance matrix.

This is the version used by most machine learning libraries, including **scikit-learn**.

---

# Why Centering Is Necessary

One question naturally arises.

Why must we subtract the mean before computing the covariance matrix?

Suppose two descriptors have very large average values.

Without centering,

their averages would dominate the calculations.

The covariance would then reflect the magnitude of the descriptors rather than how they vary.

Centering removes this bias.

After centering,

only the fluctuations around the average remain.

This ensures that PCA analyzes variation rather than absolute values.

For this reason,

**centering is an essential preprocessing step in PCA.**

---

# Covariance Matrix in Materials Informatics

Consider a descriptor matrix generated using matminer.

```text
Material

↓

ElementProperty

↓

Stoichiometry

↓

Bond Fractions

↓

Structural Descriptors

↓

Orbital Features

↓

450 Descriptors
```

The covariance matrix summarizes how every descriptor relates to every other descriptor.

Examples include

- Density ↔ Packing Fraction
- Atomic Radius ↔ Unit Cell Volume
- Electronegativity ↔ Ionic Radius
- Average Atomic Mass ↔ Atomic Number
- Covalent Radius ↔ Metallic Radius

Some descriptor pairs exhibit

- strong positive covariance,
- strong negative covariance,
- or almost no covariance.

PCA analyzes this entire matrix simultaneously to determine which combinations of descriptors explain the largest variations across all materials.

---

# The Covariance Matrix as a Geometric Object

Although we often think of the covariance matrix as simply a table of numbers, it has a much deeper meaning.

Imagine a cloud of points representing thousands of materials in a high-dimensional descriptor space.

```text
            •      •
       •
                  •
   •
              •
         •
```

The covariance matrix describes

- the overall shape of this cloud,
- the direction in which it stretches,
- the direction in which it compresses,
- and how different dimensions interact with one another.

If the cloud stretches strongly along a particular direction,

that direction contains a large amount of variation.

If the cloud is very narrow along another direction,

that direction contains relatively little information.

The job of PCA is to discover these special directions automatically.

---

# Why the Covariance Matrix Is the Heart of PCA

Every concept we have studied so far has been preparing us for one crucial step.

- Variance measures the spread of a single descriptor.
- Covariance measures the relationship between two descriptors.
- The covariance matrix summarizes the relationships among **all descriptors simultaneously**.

Once this matrix has been constructed, the statistical part of PCA is complete.

The remaining task belongs to **linear algebra**.

The next step is to ask a profound question:

> **Can we find the directions along which this covariance matrix exhibits its greatest variation?**

The answer comes from one of the most elegant ideas in linear algebra—the concepts of **eigenvectors** and **eigenvalues**.

These mathematical objects reveal the natural axes of the data and ultimately become the **principal components** that define the new coordinate system used by PCA.

## 11.5 Eigenvectors and Eigenvalues — Discovering the Natural Directions of the Data

Up to this point, we have focused entirely on statistics.

We began by understanding **variance**, which measures how much a single descriptor varies. We then introduced **covariance**, which measures how two descriptors vary together. Finally, we combined all pairwise covariances into the **covariance matrix**, a compact representation of the statistical relationships within the entire dataset.

At this stage, we have successfully summarized the structure of the dataset.

However, PCA is not interested merely in storing this information.

Its true objective is to answer a much deeper question.

> **What are the most important directions in which the data vary?**

Finding these directions requires a transition from statistics to **linear algebra**.

This transition introduces two of the most important mathematical concepts used throughout science and engineering:

- Eigenvectors
- Eigenvalues

Although these concepts often appear abstract when introduced in mathematics courses, their interpretation within PCA is surprisingly intuitive.

---

# Why Do We Need Eigenvectors?

Imagine standing in the middle of a forest.

You want to determine the direction in which the forest extends the farthest.

You could walk

- north,
- south,
- east,
- west,
- northeast,
- southwest,

or in infinitely many other directions.

Some directions quickly lead to the edge of the forest.

Others allow you to travel much farther.

The longest direction naturally describes the overall orientation of the forest.

Now replace the forest with a cloud of data points.

```text
                 •

            •        •

        •

     •

          •

                 •

                      •
```

The cloud clearly stretches more in one direction than another.

PCA asks

> **Can we automatically find the direction along which the data are stretched the most?**

That direction is called the **first principal direction**.

Finding it is exactly the job of an eigenvector.

---

# An Intuitive Picture

Suppose we plot two descriptors.

- Density
- Average Atomic Radius

Instead of forming a circular cloud,

the data may form an elongated ellipse.

```text
Radius

 ^
 |                       •
 |                  •
 |             •
 |        •
 |    •
 | •
 +---------------------------------------> Density
```

Notice that the data are not spread equally in every direction.

The cloud has a clear orientation.

The longest axis represents the direction of maximum variation.

The shortest axis represents the direction of minimum variation.

PCA attempts to discover these axes automatically.

These axes are precisely the **eigenvectors** of the covariance matrix.

---

# Rotating the Coordinate System

The original descriptors define one coordinate system.

```text
          Radius

             ^
             |

             |

             |

-------------+----------------> Density
```

Unfortunately,

the data cloud may not align with these axes.

Instead,

it may be tilted.

```text
             •

         •

      •

   •

•

      •

           •
```

Rather than forcing the data to fit the existing coordinate system,

PCA rotates the coordinate system until it aligns with the natural orientation of the data.

```text
Original Axes

      ↑
      │
      │
──────┼──────►

↓

Rotate Axes

         ↗ PC₁

      •

   •

•

         ↘ PC₂
```

The data itself never changes.

Only the coordinate system changes.

This simple geometric idea lies at the heart of PCA.

---

# What Is an Eigenvector?

An **eigenvector** is a special direction associated with a matrix.

Most vectors change both their **length** and **direction** when multiplied by a matrix.

An eigenvector is different.

After multiplication,

its direction remains unchanged.

Only its length changes.

Mathematically,

an eigenvector satisfies

$$
A\mathbf{v}=\lambda\mathbf{v}
$$

where

- **A** is a matrix,
- **v** is an eigenvector,
- **λ** (lambda) is the corresponding eigenvalue.

This equation says something remarkable.

Multiplying the matrix by the eigenvector produces another vector pointing in exactly the same direction.

The only difference is its length.

---

# Understanding the Equation

The equation

$$
A\mathbf{v}=\lambda\mathbf{v}
$$

looks intimidating at first, but every symbol has a simple interpretation.

### A

The matrix.

In PCA,

this matrix is the **covariance matrix**.

---

### v

A direction in space.

Not every direction is an eigenvector.

Only a few special directions satisfy the equation.

---

### λ

A scaling factor.

It tells us how strongly the matrix stretches or compresses the corresponding eigenvector.

---

Suppose

$$
\lambda=5
$$

Then

```text
v

---->

↓

After multiplication

------------------------->
```

The vector becomes five times longer.

Its direction remains unchanged.

---

Suppose instead

$$
\lambda=0.2
$$

The vector becomes much shorter.

```text
---------->

↓

-->

```

Again,

the direction remains the same.

Only the magnitude changes.

---

# A Physical Analogy

Imagine stretching a rubber sheet.

Some directions stretch a great deal.

Other directions hardly stretch at all.

```text
Before Stretching

□□□□□□

↓

After Stretching

□□□□□□□□□□□□□□
```

The direction that stretches the most corresponds to the largest eigenvalue.

The direction that stretches the least corresponds to the smallest eigenvalue.

PCA uses exactly this idea.

The covariance matrix "stretches" vectors mathematically.

The eigenvectors reveal the preferred stretching directions.

---

# What Is an Eigenvalue?

If an eigenvector tells us **where** an important direction exists,

the corresponding eigenvalue tells us **how important** that direction is.

Every eigenvector has one associated eigenvalue.

The eigenvalue measures the amount of variation captured along that direction.

Large eigenvalue

↓

Large variance

↓

Important direction

Small eigenvalue

↓

Small variance

↓

Less important direction

This relationship is fundamental to PCA.

The algorithm always orders the principal components according to decreasing eigenvalues.

---

# Interpreting Eigenvalues in PCA

Suppose PCA computes the following eigenvalues.

| Principal Direction | Eigenvalue |
|---------------------|-----------:|
| PC₁ | 18.4 |
| PC₂ | 7.6 |
| PC₃ | 1.8 |
| PC₄ | 0.3 |

Immediately we learn several things.

The first principal direction explains much more variation than any other direction.

The second direction is also important.

The remaining directions contribute progressively less information.

Consequently,

PCA often keeps only the first few components.

The remaining components contribute very little additional information.

---

# Why Are There Multiple Eigenvectors?

Suppose our dataset has only two descriptors.

The covariance matrix has dimensions

$$
2\times2
$$

Such a matrix possesses

- two eigenvectors,
- two eigenvalues.

Graphically,

```text
          PC₂

           ↑

           │

     •

  •

•

──────────────→ PC₁
```

The first eigenvector points along the direction of maximum variation.

The second eigenvector is perpendicular to the first.

Together,

they form a completely new coordinate system.

---

If the dataset contains

450 descriptors,

the covariance matrix has dimensions

$$
450\times450
$$

Therefore,

PCA computes

- 450 eigenvectors,
- 450 eigenvalues.

Fortunately,

only a handful usually explain most of the variation.

---

# Orthogonality of Eigenvectors

One remarkable property of the covariance matrix is that its eigenvectors are **orthogonal**.

Orthogonal simply means

**mutually perpendicular**.

In two dimensions,

the eigenvectors intersect at

90°.

```text
          PC₂

           ↑

           │

-----------+------------→ PC₁
```

In higher dimensions,

the same idea applies.

Orthogonality is extremely important because it guarantees that each principal component captures **new information**.

No principal component duplicates information already explained by another component.

---

# Ordering the Eigenvectors

PCA does not use eigenvectors in an arbitrary order.

Instead,

they are sorted according to their eigenvalues.

```text
Largest Eigenvalue

↓

First Principal Component

↓

Second Largest Eigenvalue

↓

Second Principal Component

↓

Third Largest Eigenvalue

↓

Third Principal Component
```

The ordering ensures that

- PC₁ captures the maximum possible variance,
- PC₂ captures the maximum remaining variance,
- PC₃ captures the next largest remaining variance,

and so on.

This is what makes PCA an optimal linear dimensionality reduction technique.

---

# A Materials Informatics Example

Suppose we generate 450 descriptors using matminer for 20,000 crystal structures.

After computing the covariance matrix,

PCA calculates

```text
450 Eigenvectors

450 Eigenvalues
```

The first few eigenvalues might look like

| Component | Eigenvalue |
|-----------|-----------:|
| PC₁ | 152.6 |
| PC₂ | 73.4 |
| PC₃ | 28.1 |
| PC₄ | 9.2 |
| PC₅ | 4.6 |
| Remaining 445 Components | Very Small |

This immediately tells us that the majority of the information contained within 450 descriptors is concentrated in only a few directions.

Instead of analyzing all 450 descriptors,

we may need only

- PC₁,
- PC₂,
- PC₃,

to capture most of the variation in the dataset.

This dramatic reduction is what makes PCA so powerful for high-dimensional materials data.

---

# Visualizing Eigenvectors

Imagine a cloud of materials plotted in descriptor space.

```text
                    •

               •

          •

      •

   •

         •

             •

                 •
```

The covariance matrix analyzes this cloud.

The largest eigenvector aligns with the longest dimension of the cloud.

```text
                 PC₁

        ↗↗↗↗↗↗↗↗↗

            •

         •

      •

   •

```

The second eigenvector lies perpendicular to the first.

```text
              ↑ PC₂

              │

              │

──────────────┼────────────→ PC₁
```

These two directions now become the first two principal components.

Instead of describing every material using the original descriptors,

PCA describes every material by its position along these newly discovered axes.

---

# From Eigenvectors to Principal Components

At this point, the most difficult mathematics behind PCA has been introduced.

We now understand that

- the covariance matrix summarizes relationships between descriptors,
- eigenvectors identify the natural directions of variation,
- eigenvalues measure the importance of those directions.

However, one important question still remains.

> **Once the eigenvectors have been found, how do we use them to transform the original dataset?**

The answer lies in the process of **projection**.

Projection converts every material from the original descriptor space into the new coordinate system defined by the eigenvectors.

The resulting coordinates are called the **principal components**, and they form the reduced representation of the original high-dimensional dataset that makes PCA such a powerful tool for visualization, data compression, and exploratory analysis.

## 11.6 Principal Components and Projection — Transforming Data into a New Coordinate System

In the previous section, we learned how PCA analyzes the covariance matrix to compute its **eigenvectors** and **eigenvalues**.

At first glance, it may seem that PCA is complete once these quantities have been calculated.

However, the eigenvectors themselves are **not** the final output of PCA.

Instead, they define a **new coordinate system**.

The next step is to express every material in terms of this new coordinate system. This transformation is called **projection**, and the resulting coordinates are known as the **principal components**.

This section explains exactly what principal components are, how projection works, and why this transformation allows us to reduce hundreds of descriptors to just a few informative variables.

---

# A Change of Coordinates

Imagine locating a city on Earth.

One way is to use

- latitude,
- longitude.

Another way is to describe the same location relative to another city.

The physical location has not changed.

Only the coordinate system has changed.

Similarly, suppose a material is originally described using three descriptors.

```text
Density

Atomic Radius

Electronegativity
```

These descriptors define one coordinate system.

```text
          Electronegativity

                 ^

                 |

                 |

                 |

                 |

                 +-------------> Density
                /
               /
              /
     Atomic Radius
```

After PCA,

the coordinate axes become

```text
PC₁

PC₂

PC₃
```

These new axes are simply rotated versions of the original descriptor axes.

The material itself remains exactly the same.

Only its coordinates change.

---

# What Is a Principal Component?

A **principal component** is the coordinate of a data point after it has been projected onto an eigenvector.

This definition is extremely important.

An eigenvector represents a **direction**.

A principal component represents the **position of a material along that direction**.

For example,

suppose the first eigenvector points along the direction of maximum variation.

Projecting every material onto this direction produces

```text
PC₁
```

Projecting onto the second eigenvector produces

```text
PC₂
```

Projecting onto the third eigenvector produces

```text
PC₃
```

Therefore,

each principal component is simply a new numerical feature.

Instead of describing a material using hundreds of original descriptors,

we describe it using

```text
PC₁

PC₂

PC₃

...
```

---

# An Analogy: Shadows

Projection is easier to understand using the idea of a shadow.

Imagine shining a flashlight on an object.

```text
      Light

        *

       \

        \

         □ Object

          \

           \

------------+----------------

          Shadow
```

The shadow is a projection of the object onto the ground.

Although some information is lost,

the shadow still captures important aspects of the object's shape.

PCA performs a similar operation.

Instead of projecting a three-dimensional object onto a floor,

it projects high-dimensional data onto carefully chosen directions.

These directions are the eigenvectors.

---

# Why Projection Works

Suppose a dataset forms an elongated cloud.

```text
                    •

                •

             •

         •

      •

   •

```

Most of the variation lies along one direction.

Instead of recording the position of every point in every direction,

we can record only its position along the longest axis.

```text
                    •

                •

             •

         •

      •

   •

------------------------------- PC₁
```

Almost all important information is preserved.

Very little information exists in directions perpendicular to this axis.

This is why PCA can reduce dimensionality without losing much information.

---

# Mathematical View of Projection

Suppose

- **x** is a centered data point,
- **v₁** is the first eigenvector.

The first principal component is obtained by projecting the data point onto the eigenvector.

Mathematically,

$$
PC_1 = x \cdot v_1
$$

where

- **x** is the original descriptor vector,
- **v₁** is the first eigenvector,
- **·** represents the dot product.

Similarly,

$$
PC_2 = x \cdot v_2
$$

$$
PC_3 = x \cdot v_3
$$

Each principal component is therefore the dot product between the original descriptor vector and one eigenvector.

---

# Matrix Form of the Projection

Suppose

- **X** is the centered descriptor matrix,
- **V** contains the selected eigenvectors.

The complete projection is

$$
Z = XV
$$

where

- **X** is the original centered dataset,
- **V** is the matrix of eigenvectors,
- **Z** is the transformed dataset.

The matrix **Z** contains the principal components.

Instead of

```text
20,000 Materials

×

450 Descriptors
```

we might obtain

```text
20,000 Materials

×

10 Principal Components
```

The dataset has become dramatically smaller while retaining most of its information.

---

# Selecting Only the Most Important Components

One of the greatest strengths of PCA is that we do not have to keep every principal component.

Suppose PCA computes

450 principal components.

They are ordered according to their eigenvalues.

```text
PC₁

↓

Most Important

PC₂

↓

Very Important

PC₃

↓

Important

...

PC₄₅₀

↓

Least Important
```

Rather than keeping all 450,

we may retain only

```text
PC₁

PC₂

PC₃

PC₄

PC₅
```

The remaining components contribute relatively little information and can often be discarded.

This is the essence of dimensionality reduction.

---

# A Numerical Example

Suppose a dataset originally contains

```text
450 descriptors
```

After PCA,

the explained variance may look like this.

| Component | Variance Explained |
|-----------|-------------------:|
| PC₁ | 38% |
| PC₂ | 24% |
| PC₃ | 15% |
| PC₄ | 8% |
| PC₅ | 5% |
| Remaining 445 Components | 10% |

Adding the first five components gives

```text
38 + 24 + 15 + 8 + 5 = 90%
```

This means

only five principal components preserve approximately **90%** of the total variation present in the original 450 descriptors.

The remaining 445 components collectively contribute only about 10% of the variation.

For many applications,

discarding these components has little impact on the overall structure of the data.

---

# Geometric Interpretation

Imagine looking at a three-dimensional object.

```text
          z

          ^

         /|

        / |

       /  |

      •---+------> x

     /

    /

   y
```

Suppose nearly all of the variation occurs within a flat plane.

```text
      • • • • •

   • • • • •

• • • • •
```

The third dimension contributes very little.

Instead of storing

```text
x

y

z
```

we can safely ignore the third direction and keep only

```text
PC₁

PC₂
```

The data now lie in two dimensions while preserving nearly all meaningful variation.

---

# Projection in High Dimensions

Although we can visualize only two or three dimensions,

the same idea extends naturally to hundreds of descriptors.

Suppose every material is represented by

```text
450 descriptors.
```

Each material is therefore a single point in a 450-dimensional space.

PCA computes

450 orthogonal eigenvectors.

These eigenvectors define

450 new coordinate axes.

Every material is then projected onto these axes.

Instead of recording

450 coordinates,

we might keep only

```text
PC₁

PC₂

PC₃

PC₄

PC₅
```

The remaining coordinates are discarded because they contain relatively little information.

---

# Why Principal Components Are Better Than the Original Features

The original descriptors often contain

- redundant information,
- correlated variables,
- unnecessary complexity.

For example,

```text
Average Atomic Number

Average Atomic Mass

Average Nuclear Charge
```

all describe related physical properties.

PCA combines these correlated descriptors into a smaller number of independent principal components.

As a result,

the transformed features have several desirable properties.

- They are uncorrelated.
- They are ordered by importance.
- They capture the maximum possible variance.
- They eliminate much of the redundancy present in the original descriptors.
- They simplify visualization and machine learning.

These properties explain why PCA is frequently used as a preprocessing step before training machine learning models.

---

# PCA as Data Compression

Another useful way to think about PCA is as a form of **lossy data compression**.

Imagine compressing a high-resolution image.

A small amount of fine detail may be lost,

but the overall appearance remains almost unchanged.

Similarly,

PCA compresses a high-dimensional dataset.

Instead of storing every descriptor,

it stores only the most informative combinations of descriptors.

The goal is not to preserve every tiny fluctuation.

The goal is to preserve the overall structure of the data while greatly reducing its dimensionality.

---

# Projection in Materials Informatics

A common workflow in materials informatics is

```text
Crystal Structures

        │

        ▼

pymatgen

        │

        ▼

matminer

        │

        ▼

450–1000 Descriptors

        │

        ▼

Feature Scaling

        │

        ▼

PCA

        │

        ▼

Principal Components

        │

        ▼

PC₁ vs PC₂ Scatter Plot

        │

        ▼

Visualization of Materials Space
```

Instead of attempting to interpret hundreds of descriptor columns,

researchers examine a simple two-dimensional plot of the first two principal components.

Remarkably, materials with similar chemistry or crystal structures often appear close together, while different material classes naturally separate into distinct regions.

---

# From Projection to Visualization

At this point, the complete mathematical framework of PCA is almost complete.

We now understand

- how variance measures the spread of a descriptor,
- how covariance measures relationships between descriptors,
- how the covariance matrix summarizes the entire dataset,
- how eigenvectors identify the principal directions,
- how eigenvalues measure the importance of those directions,
- and how projection transforms every material into the new coordinate system defined by the principal components.

One important question still remains.

> **How do we decide how many principal components should be kept?**

Keeping too many components defeats the purpose of dimensionality reduction.

Keeping too few may discard valuable information.

To answer this question, we introduce the concept of **explained variance**, which quantifies how much of the original information is preserved by each principal component and provides a principled way to choose the optimal number of components.

## 11.7 Explained Variance — How Many Principal Components Should We Keep?

After computing the principal components, PCA provides a new representation of the dataset. If the original dataset contained **450 descriptors**, PCA also produces **450 principal components**.

At first, this may seem disappointing.

We started with 450 variables.

After applying PCA, we still have 450 variables.

Has anything actually been reduced?

The answer is **not yet**.

The true power of PCA comes from the fact that **the principal components are not equally important**. Some components capture a tremendous amount of information, while others contribute almost nothing.

The challenge is therefore no longer

> "How do we compute the principal components?"

Instead, it becomes

> **"How many principal components should we keep?"**

Answering this question requires understanding **explained variance**.

---

# Information Is Not Distributed Equally

Imagine reading a 500-page textbook.

Suppose

- the first 50 pages introduce all the major concepts,
- the next 100 pages provide detailed explanations,
- the remaining 350 pages contain examples and additional discussions.

Although every page contributes something, the first part of the book contains most of the essential ideas.

The same principle applies to PCA.

Not every principal component contributes equally.

For example,

| Principal Component | Importance |
|---------------------|-----------:|
| PC₁ | Very High |
| PC₂ | High |
| PC₃ | Moderate |
| PC₄ | Small |
| PC₅ | Very Small |
| Remaining Components | Minimal |

The first few components often contain most of the information in the dataset.

---

# What Does "Explained Variance" Mean?

Recall that PCA searches for directions with the largest variance.

Each principal component therefore captures a certain amount of the total variance present in the original dataset.

The amount of variance captured by a component is called its **explained variance**.

Simply put,

> **Explained variance measures how much of the original information is preserved by a principal component.**

A component with large explained variance contains a great deal of information.

A component with very small explained variance contributes relatively little.

---

# Relationship Between Eigenvalues and Explained Variance

Earlier, we learned that every principal component has an associated eigenvalue.

Those eigenvalues are much more than mathematical quantities.

In PCA,

they directly measure the variance captured by each principal component.

Larger eigenvalue

↓

More variance captured

↓

More information preserved

Smaller eigenvalue

↓

Less variance captured

↓

Less information preserved

Therefore,

the eigenvalues determine the importance of each principal component.

---

# Calculating Explained Variance Ratio

Suppose PCA computes the following eigenvalues.

| Component | Eigenvalue |
|-----------|-----------:|
| PC₁ | 18 |
| PC₂ | 10 |
| PC₃ | 6 |
| PC₄ | 4 |
| PC₅ | 2 |

The total variance is simply the sum of all eigenvalues.

$$
18+10+6+4+2=40
$$

The explained variance ratio of each component is

$$
\frac{\text{Eigenvalue of Component}}{\text{Total Eigenvalues}}
$$

For PC₁,

$$
\frac{18}{40}=0.45
$$

or

45%.

Similarly,

| Component | Explained Variance |
|-----------|-------------------:|
| PC₁ | 45% |
| PC₂ | 25% |
| PC₃ | 15% |
| PC₄ | 10% |
| PC₅ | 5% |

Notice that

```text
45%

+

25%

+

15%

+

10%

+

5%

=

100%
```

All principal components together explain **100%** of the variance in the original dataset.

---

# Cumulative Explained Variance

While the explained variance of individual components is useful, we are usually more interested in the **combined contribution** of several components.

This is called the **cumulative explained variance**.

For the previous example,

| Components Kept | Cumulative Variance |
|-----------------|--------------------:|
| PC₁ | 45% |
| PC₁ + PC₂ | 70% |
| PC₁ + PC₂ + PC₃ | 85% |
| PC₁ + PC₂ + PC₃ + PC₄ | 95% |
| All Components | 100% |

This table immediately tells us that

keeping only the first four principal components preserves **95%** of the information contained in the original dataset.

Instead of working with five variables,

we only need four.

For a real materials dataset containing hundreds of descriptors, the reduction is often much more dramatic.

---

# Why Does PCA Work So Well?

Many descriptors generated by **matminer** are correlated.

Consider the following descriptors.

- Average atomic number
- Average atomic mass
- Average nuclear charge
- Average number of electrons

Although these descriptors measure different physical quantities, they often change together because they all depend on elemental composition.

Consequently,

they contain overlapping information.

Instead of preserving each descriptor individually,

PCA combines them into a single principal component.

This allows a large amount of information to be compressed into only a few variables.

---

# A Materials Informatics Example

Suppose we compute

450 descriptors

for

15,000 materials.

After PCA,

the explained variance may look like this.

| Component | Explained Variance |
|-----------|-------------------:|
| PC₁ | 31% |
| PC₂ | 21% |
| PC₃ | 14% |
| PC₄ | 9% |
| PC₅ | 6% |
| Remaining 445 Components | 19% |

The cumulative variance becomes

| Components | Cumulative Variance |
|------------|--------------------:|
| PC₁ | 31% |
| PC₁–PC₂ | 52% |
| PC₁–PC₃ | 66% |
| PC₁–PC₄ | 75% |
| PC₁–PC₅ | 81% |
| All Components | 100% |

Although the original dataset contained **450 descriptors**, the first five principal components already preserve more than **80%** of the information.

For visualization, this level of compression is often sufficient.

---

# Choosing the Number of Components

One of the most common questions when applying PCA is

> **How many principal components should be retained?**

Unfortunately,

there is no universal answer.

The appropriate number depends on the objective of the analysis.

### Case 1 — Data Visualization

If the goal is visualization,

we usually keep

- PC₁
- PC₂

or

- PC₁
- PC₂
- PC₃

because humans can only visualize two or three dimensions.

Even if these components explain only 60–80% of the variance, they often reveal meaningful patterns.

---

### Case 2 — Machine Learning Preprocessing

If PCA is used before training a machine learning model,

we typically retain enough components to preserve

- 90%
- 95%
- or even 99%

of the total variance.

This minimizes information loss while significantly reducing dimensionality.

---

### Case 3 — Data Compression

When storage efficiency or computational speed is important,

a smaller number of components may be selected to achieve a balance between compression and information preservation.

---

# The Scree Plot

One of the most widely used tools for selecting the number of principal components is the **Scree Plot**.

A Scree Plot displays the explained variance of each principal component.

```text
Explained
Variance

^

| *

| * *

| * *

| * *

| * *

| * *

| * *

| * *

| * *

| * *

|______________________________>

 PC₁ PC₂ PC₃ PC₄ PC₅ ...
```

The height of each point represents the variance explained by that component.

Typically,

the first few components explain a large amount of variance,

after which the curve gradually levels off.

---

# The Elbow Method

The Scree Plot often contains a noticeable bend called the **elbow**.

```text
Variance

^

| *

| * *

| *  *

| *   *

| *     *

| *       *

| *         * * * * *

+---------------------------------->

          Elbow
```

The elbow indicates the point beyond which additional principal components contribute relatively little new information.

A common strategy is to retain all components before this point.

Although the elbow method is heuristic rather than mathematically exact, it is widely used because it provides an intuitive balance between dimensionality reduction and information retention.

---

# Trade-Off Between Compression and Information

Dimensionality reduction always involves a compromise.

Keeping more principal components

- preserves more information,
- but reduces dimensionality less.

Keeping fewer components

- greatly simplifies the dataset,
- but discards more information.

The goal is therefore to identify the smallest number of principal components that still captures the essential structure of the data.

---

# Explained Variance in Materials Informatics

Consider a high-throughput materials dataset.

```text
20,000 Crystal Structures

↓

matminer

↓

600 Descriptors

↓

PCA

↓

600 Principal Components

↓

Keep First 15 Components

↓

95% Variance Preserved
```

Instead of performing machine learning using 600 correlated descriptors,

we can work with only 15 independent principal components.

This dramatically reduces

- computational cost,
- memory requirements,
- training time,
- and descriptor redundancy,

while preserving nearly all of the meaningful information contained in the original dataset.

---

# Preparing for Visualization

At this point, we have completed the mathematical foundation of PCA.

We now understand

- variance,
- covariance,
- the covariance matrix,
- eigenvectors,
- eigenvalues,
- projection,
- principal components,
- and explained variance.

The next step is to put all of these ideas together into a complete PCA algorithm and apply it to real materials datasets.

We will first examine the PCA workflow step by step and then use **scikit-learn** to transform hundreds of descriptors generated with **pymatgen** and **matminer** into two- and three-dimensional representations that reveal the hidden structure of materials space.

## 11.8 The Principal Component Analysis Algorithm — Step-by-Step

After studying the mathematical foundations of PCA, it is useful to step back and see how all of the individual concepts fit together into a complete algorithm.

So far, we have learned

- why variance is important,
- how covariance measures relationships between descriptors,
- how the covariance matrix summarizes those relationships,
- how eigenvectors identify important directions,
- how eigenvalues measure the importance of those directions,
- and how projection transforms the data into a new coordinate system.

These concepts are not independent ideas.

Each one is a necessary step in the PCA algorithm.

The complete workflow can be summarized as

```text
Original Dataset
        │
        ▼
Standardize Features
        │
        ▼
Center the Data
        │
        ▼
Compute Covariance Matrix
        │
        ▼
Compute Eigenvalues
and Eigenvectors
        │
        ▼
Sort Components
by Eigenvalue
        │
        ▼
Select Top Components
        │
        ▼
Project Data
        │
        ▼
Reduced-Dimensional Dataset
```

Although this workflow appears simple, every step has a specific mathematical purpose. Skipping or incorrectly performing any one of them can produce misleading results.

---

# Step 1 — Construct the Descriptor Matrix

Everything begins with a dataset.

Suppose we have calculated descriptors for a collection of materials using **matminer**.

The descriptor matrix might look like this.

| Material | Density | Avg Atomic Radius | Electronegativity | Packing Fraction | ... |
|-----------|---------:|------------------:|------------------:|-----------------:|----:|
| Material 1 | 5.32 | 1.28 | 1.87 | 0.69 | ... |
| Material 2 | 7.91 | 1.44 | 2.13 | 0.74 | ... |
| Material 3 | 3.46 | 1.12 | 1.54 | 0.63 | ... |
| ... | ... | ... | ... | ... | ... |

If there are

- **N materials**
- **M descriptors**

then the dataset can be represented as

$$
X \in \mathbb{R}^{N \times M}
$$

For example,

```text
20,000 Materials

×

450 Descriptors
```

Every row corresponds to one material.

Every column corresponds to one descriptor.

---

# Step 2 — Standardize the Features

One of the most important preprocessing steps in PCA is **feature scaling**.

Different descriptors often have completely different numerical ranges.

For example,

| Descriptor | Typical Values |
|------------|---------------:|
| Density | 2–20 |
| Atomic Radius | 0.5–3 |
| Atomic Number | 1–118 |
| Formation Energy | -8 to 3 |
| Unit Cell Volume | 20–1000 |

Notice that these descriptors are measured on vastly different scales.

If PCA were applied directly,

descriptors with larger numerical values would dominate the covariance matrix.

For example,

Unit Cell Volume

might contribute much more strongly than

Electronegativity,

not because it is more informative,

but simply because its numerical values are larger.

To avoid this problem,

each descriptor is standardized.

For every descriptor,

the mean is subtracted,

and the result is divided by the standard deviation.

After standardization,

every descriptor has

- mean = 0
- standard deviation = 1

This ensures that all descriptors contribute fairly.

---

# Why Standardization Is Essential

Consider two descriptors.

```text
Density

3
5
7
9
```

and

```text
Atomic Number

10
30
60
90
```

Although both descriptors vary,

Atomic Number has much larger numerical values.

Without scaling,

PCA would incorrectly conclude that Atomic Number is much more important.

Standardization removes this artificial bias.

---

# Step 3 — Center the Dataset

Although standardization already centers the data,

it is useful to understand why centering is required.

Suppose one descriptor has an average value of

```text
150
```

Another has an average of

```text
2
```

If we compute covariance directly,

the averages influence the calculations.

Instead,

PCA focuses only on deviations from the average.

After centering,

every descriptor oscillates around zero.

```text
Before Centering

150

152

149

151

↓

After Centering

-1

1

-2

0
```

The covariance matrix now reflects variation rather than absolute magnitude.

---

# Step 4 — Compute the Covariance Matrix

Once the dataset has been centered,

PCA computes the covariance matrix.

If there are

450 descriptors,

the covariance matrix has dimensions

```text
450 × 450
```

The covariance matrix contains

- the variance of every descriptor,
- the covariance between every pair of descriptors.

Conceptually,

it summarizes

> **How every descriptor is related to every other descriptor.**

This is the statistical description of the dataset.

---

# Step 5 — Compute Eigenvalues and Eigenvectors

The next step belongs entirely to linear algebra.

The covariance matrix is decomposed into

- eigenvectors,
- eigenvalues.

```text
Covariance Matrix

↓

Eigenvalue Decomposition

↓

Eigenvectors

+

Eigenvalues
```

Each eigenvector represents

a direction in descriptor space.

Each eigenvalue measures

how much variance exists along that direction.

---

# Step 6 — Sort the Components

The eigenvectors are not used in arbitrary order.

Instead,

they are sorted according to their eigenvalues.

Largest eigenvalue

↓

PC₁

Second largest eigenvalue

↓

PC₂

Third largest eigenvalue

↓

PC₃

...

Smallest eigenvalue

↓

Last principal component

This ordering guarantees that the first principal component captures the maximum possible variance.

---

# Step 7 — Choose the Number of Components

Suppose PCA computes

450 principal components.

Keeping all of them defeats the purpose of dimensionality reduction.

Instead,

we examine the explained variance.

For example,

| Component | Cumulative Variance |
|-----------|--------------------:|
| PC₁ | 38% |
| PC₁–PC₂ | 61% |
| PC₁–PC₃ | 76% |
| PC₁–PC₅ | 88% |
| PC₁–PC₁₂ | 96% |

If our goal is

95% information retention,

we keep

the first

12 principal components.

The remaining

438 components

are discarded.

---

# Step 8 — Project the Dataset

Once the important eigenvectors have been selected,

every material is projected onto them.

Originally,

one material might be represented as

```text
Density

Radius

Volume

Packing Fraction

Electronegativity

...

450 Descriptors
```

After projection,

the same material becomes

```text
PC₁

PC₂

PC₃

...

PC₁₂
```

Nothing about the material has changed.

Only its numerical representation has changed.

This new representation is much smaller,

contains less redundancy,

and is often easier for machine learning algorithms to process.

---

# Step 9 — Analyze the Reduced Dataset

The transformed dataset can now be used for a wide variety of applications.

For example,

```text
Principal Components

↓

Visualization

↓

Cluster Analysis

↓

Outlier Detection

↓

Machine Learning

↓

Chemical Space Exploration
```

The reduced representation often reveals patterns that were impossible to detect within hundreds of original descriptors.

---

# Computational Complexity

PCA is computationally efficient for many materials informatics problems, but its cost increases as the number of descriptors grows.

Suppose

- **N** = number of materials
- **M** = number of descriptors

The major computational tasks include

- computing the covariance matrix,
- performing eigenvalue decomposition,
- projecting the data.

For datasets containing thousands of descriptors,

the eigenvalue decomposition becomes the most computationally expensive step.

Fortunately,

most materials informatics datasets contain

- thousands to hundreds of thousands of materials,

but only

- hundreds of descriptors,

making PCA computationally practical on modern computers.

---

# Strengths of PCA

PCA has become one of the most widely used dimensionality reduction techniques because it offers several important advantages.

- Reduces hundreds of descriptors to a manageable number.
- Removes redundant information.
- Produces uncorrelated features.
- Speeds up machine learning algorithms.
- Enables visualization of high-dimensional datasets.
- Often improves model stability.
- Requires no labeled data.
- Is mathematically well understood and computationally efficient.

Because of these advantages,

PCA is frequently used as the first exploratory analysis performed on a newly generated materials dataset.

---

# Limitations of PCA

Despite its usefulness,

PCA also has important limitations.

First,

PCA is a **linear** method.

It can only discover linear relationships between descriptors.

If the true structure of the data is highly nonlinear,

PCA may fail to capture it.

Second,

the principal components are combinations of many original descriptors.

This can make them difficult to interpret physically.

For example,

a principal component may combine

- density,
- electronegativity,
- atomic radius,
- and packing fraction

into a single variable.

Although mathematically meaningful,

its physical interpretation may not be immediately obvious.

Third,

PCA is sensitive to feature scaling.

Failing to standardize descriptors before applying PCA can produce misleading results.

Finally,

PCA assumes that directions with the largest variance are the most informative.

While this assumption is often valid, it is not universally true.

In some applications, important scientific information may exist in low-variance directions.

---

# PCA in the Complete Materials Informatics Workflow

The role of PCA within a modern materials informatics pipeline can now be understood.

```text
Crystal Structures
        │
        ▼
pymatgen
        │
        ▼
matminer
        │
        ▼
Descriptor Matrix
(300–1000 Features)
        │
        ▼
Feature Scaling
        │
        ▼
Principal Component Analysis
        │
        ▼
Reduced Feature Space
(2–20 Components)
        │
        ├──────────────► Visualization
        │
        ├──────────────► Clustering
        │
        ├──────────────► Outlier Detection
        │
        └──────────────► Machine Learning
```

At this point, we have completed the theoretical foundation of Principal Component Analysis.

We understand not only **what PCA does**, but **why every step is necessary** and **how the mathematics translates into practical data analysis**.

The next section moves from theory to practice. We will implement PCA using **scikit-learn**, apply it to descriptor datasets generated with **pymatgen** and **matminer**, and learn how to visualize high-dimensional materials space using the first few principal components.

## 11.9 Implementing Principal Component Analysis with Scikit-learn

After developing the mathematical theory behind Principal Component Analysis (PCA), the next step is to apply it to real materials datasets.

Fortunately, we do not need to manually

- compute covariance matrices,
- solve eigenvalue problems,
- sort eigenvectors,
- or perform matrix projections.

The Python machine learning library **scikit-learn** implements all of these mathematical operations efficiently through its `PCA` class.

Although the library performs the calculations automatically, understanding the mathematics remains essential. Without understanding what PCA is doing internally, it becomes difficult to interpret the results correctly.

In this section, we will learn how to apply PCA to descriptor datasets generated using **pymatgen** and **matminer**.

---

# The PCA Workflow in Python

The complete workflow consists of only a few steps.

```text
Descriptor Data
        │
        ▼
Load Dataset
        │
        ▼
Separate Features
        │
        ▼
Standardize Features
        │
        ▼
Apply PCA
        │
        ▼
Transform Dataset
        │
        ▼
Analyze Principal Components
        │
        ▼
Visualize Materials Space
```

Although the code is short, each step corresponds to an important mathematical operation discussed in the previous sections.

---

# Required Python Libraries

The most commonly used libraries are

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
```

Each library has a specific purpose.

| Library | Purpose |
|----------|---------|
| pandas | Reading and manipulating datasets |
| NumPy | Numerical computations |
| StandardScaler | Feature standardization |
| PCA | Principal Component Analysis |
| Matplotlib | Visualization |

---

# Loading the Dataset

Suppose we have already generated descriptors using **matminer**.

The dataset might be stored as

```text
materials_descriptors.csv
```

It can be loaded using

```python
df = pd.read_csv("materials_descriptors.csv")
```

Suppose the dataframe contains

| Formula | Density | Radius | Electronegativity | Packing Fraction | Band Gap |
|-----------|---------:|--------:|------------------:|-----------------:|----------:|
| Si | ... | ... | ... | ... | ... |
| Fe2O3 | ... | ... | ... | ... | ... |
| LiFePO4 | ... | ... | ... | ... | ... |

Notice that

- Formula is not a numerical descriptor.
- Band Gap may be the target property.

PCA should only be applied to numerical descriptor columns.

---

# Selecting the Feature Matrix

Suppose the first column contains material names and the last column contains the target property.

The descriptor matrix becomes

```python
X = df.iloc[:, 1:-1]
```

If the dataframe contains only descriptors,

then

```python
X = df
```

can be used directly.

Always verify the dataframe before applying PCA.

```python
print(X.head())
```

---

# Why Standardization Comes First

One of the most common mistakes beginners make is applying PCA directly to raw descriptors.

Consider two descriptors.

| Descriptor | Typical Range |
|------------|--------------:|
| Density | 2–20 |
| Atomic Number | 1–118 |

Atomic Number has much larger numerical values.

Without scaling,

it dominates the covariance matrix.

The resulting principal components become biased toward descriptors with large numerical ranges.

To avoid this,

we standardize every feature.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

This performs two operations.

First,

the mean of every descriptor becomes zero.

Second,

the standard deviation becomes one.

After scaling,

all descriptors contribute equally.

---

# Creating the PCA Model

Once the descriptors have been standardized,

creating the PCA model requires only one line.

```python
pca = PCA()
```

At this stage,

no calculations have been performed.

We have only created a PCA object.

---

# Computing the Principal Components

The PCA model learns from the dataset using

```python
X_pca = pca.fit_transform(X_scaled)
```

This single command performs every mathematical step discussed earlier.

Internally,

scikit-learn

- centers the data,
- computes the covariance matrix,
- calculates eigenvalues,
- calculates eigenvectors,
- sorts the principal components,
- projects every material into the new coordinate system.

The result,

```python
X_pca
```

contains the transformed dataset.

---

# Understanding the Output

Suppose the original dataset contained

```text
500 Materials

×

300 Descriptors
```

After applying PCA,

the transformed matrix still has

```text
500 Materials

×

300 Principal Components
```

Initially,

nothing has been discarded.

Every principal component has been computed.

The reduction happens only after selecting a subset of components.

---

# Keeping Only Two Principal Components

If the goal is visualization,

we usually keep only

- PC₁
- PC₂

This can be done directly.

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

Now,

the transformed dataset contains

```text
500 Materials

×

2 Components
```

instead of

```text
500 Materials

×

300 Descriptors
```

This is a dramatic reduction in dimensionality.

---

# Keeping a Fixed Number of Components

Suppose we want to preserve

the first

10 principal components.

```python
pca = PCA(n_components=10)

X_pca = pca.fit_transform(X_scaled)
```

The transformed matrix now contains

```text
PC₁

PC₂

...

PC₁₀
```

---

# Keeping Components Based on Explained Variance

Instead of specifying the exact number of components,

we can specify the amount of variance to preserve.

For example,

to retain

95%

of the variance,

```python
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
```

Scikit-learn automatically determines the minimum number of principal components required to preserve at least 95% of the total variance.

This is one of the most useful features of the PCA implementation.

---

# Explained Variance Ratio

After fitting the model,

the explained variance ratio can be accessed through

```python
pca.explained_variance_ratio_
```

Example output

```python
array([0.41,
       0.22,
       0.11,
       0.08,
       0.05,
       ...])
```

This means

| Component | Explained Variance |
|-----------|-------------------:|
| PC₁ | 41% |
| PC₂ | 22% |
| PC₃ | 11% |
| PC₄ | 8% |
| PC₅ | 5% |

The first two components together explain

```text
63%
```

of the total variance.

---

# Computing Cumulative Explained Variance

The cumulative explained variance is easily calculated.

```python
cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print(cumulative_variance)
```

Example output

```python
[0.41
 0.63
 0.74
 0.82
 0.87
 0.91
 0.94
 0.96
 ...]
```

This tells us

- PC₁ explains 41%
- PC₁–PC₂ explain 63%
- PC₁–PC₅ explain 87%
- PC₁–PC₈ explain 96%

---

# Accessing the Principal Components

Each row of

```python
X_pca
```

represents one material.

For example,

```python
print(X_pca[:5])
```

might produce

```text
PC1      PC2

-2.14    0.82

-1.37   -0.54

 0.61    1.28

 2.44   -1.11

 0.32    0.74
```

These values are the coordinates of each material in the new PCA coordinate system.

---

# Accessing the Eigenvectors

The principal directions are stored as

```python
pca.components_
```

Each row corresponds to one principal component.

Each column corresponds to one original descriptor.

Example

```text
PC1

Density             0.42

Atomic Radius       0.37

Volume             -0.11

Packing Fraction    0.29

...
```

These numbers are called **loadings**.

They indicate how strongly each original descriptor contributes to a principal component.

Large positive values indicate strong positive contributions.

Large negative values indicate strong negative contributions.

Values close to zero contribute very little.

Later in this chapter, we will use these loadings to interpret the physical meaning of the principal components.

---

# Accessing the Eigenvalues

The eigenvalues are available as

```python
pca.explained_variance_
```

Example

```python
array([
18.5,
9.7,
4.2,
2.8,
...
])
```

Recall that

larger eigenvalues correspond to

greater variance

and therefore

more important principal components.

---

# Checking the Shape of the Transformed Dataset

It is always good practice to verify the dimensionality reduction.

```python
print(X.shape)

print(X_pca.shape)
```

Example output

```text
Original Dataset

(500, 300)

Reduced Dataset

(500, 10)
```

Notice that

the number of materials remains unchanged,

while

the number of descriptors has been reduced dramatically.

---

# Common Mistakes When Applying PCA

Several mistakes occur frequently when PCA is applied to materials datasets.

### Applying PCA Before Scaling

Incorrect workflow

```text
Descriptors

↓

PCA
```

Correct workflow

```text
Descriptors

↓

StandardScaler

↓

PCA
```

Always standardize descriptors unless there is a strong scientific reason not to.

---

### Including Non-Numerical Columns

Columns such as

- Formula
- Material ID
- Space Group Symbol

should not be included in the descriptor matrix.

Only numerical descriptors should be passed to PCA.

---

### Including the Target Property

Suppose the dataset contains

```text
Band Gap
```

or

```text
Formation Energy
```

These are labels for supervised learning.

They should not be included when computing principal components.

Only input descriptors belong in the PCA feature matrix.

---

### Assuming PC₁ Has a Direct Physical Meaning

The first principal component is usually a weighted combination of many descriptors.

It rarely corresponds to a single physical property.

Interpreting principal components requires examining their loadings and understanding which descriptors contribute most strongly.

---

# PCA Is Now Ready for Exploration

At this point, we have transformed a high-dimensional descriptor matrix into a much smaller set of principal components.

However, the numerical values alone provide limited insight.

The true strength of PCA becomes apparent when these principal components are visualized.

By plotting **PC₁ against PC₂**, or **PC₁, PC₂, and PC₃** together, we can often reveal hidden patterns in the data. Materials with similar chemistry, crystal structures, or functional properties frequently cluster together, while unusual compounds appear as isolated points.

In the next section, we will use these principal components to visualize **materials space**, interpret the resulting plots, and explore how PCA reveals the underlying organization of large materials databases.

## 11.10 Visualizing High-Dimensional Materials Space Using PCA

One of the greatest strengths of Principal Component Analysis is that it allows us to **visualize datasets that would otherwise be impossible to understand**.

Humans can naturally visualize

- one-dimensional data,
- two-dimensional data,
- and three-dimensional data.

However, a typical materials informatics dataset may contain

- 200 descriptors,
- 500 descriptors,
- or even more than 1000 descriptors.

Such datasets exist in hundreds of dimensions, making direct visualization impossible.

PCA solves this problem by projecting the original descriptor space into a small number of principal components while preserving as much information as possible.

Instead of visualizing

```text
500 Descriptors
```

we visualize

```text
PC₁

PC₂
```

or

```text
PC₁

PC₂

PC₃
```

This allows us to explore the structure of the dataset and discover hidden relationships between materials.

---

# What Is Materials Space?

Before creating PCA plots, it is important to understand the idea of **materials space**.

Suppose every material is described using four descriptors.

- Density
- Atomic Radius
- Electronegativity
- Unit Cell Volume

Each material can then be represented as a single point in a four-dimensional space.

```text
Material A

↓

(5.1, 1.25, 1.87, 84)

Material B

↓

(7.8, 1.43, 2.05, 61)

Material C

↓

(2.9, 0.98, 1.52, 103)
```

Each coordinate corresponds to one descriptor.

If the dataset contains

450 descriptors,

every material becomes a point in a **450-dimensional space**.

Although we cannot visualize such a space directly, mathematically it is no different from ordinary three-dimensional space.

PCA simply finds a lower-dimensional representation of this high-dimensional materials space.

---

# From Descriptor Space to PCA Space

The transformation performed by PCA can be summarized as

```text
Original Descriptor Space

Density

Radius

Volume

Packing Fraction

...

450 Descriptors

        │

        ▼

Principal Component Analysis

        │

        ▼

Reduced PCA Space

PC₁

PC₂

PC₃
```

The original descriptors disappear.

Instead,

each material is described using its coordinates along the principal components.

---

# Creating a Two-Dimensional PCA Plot

Suppose we keep only two principal components.

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

The transformed dataset contains

```text
PC₁

PC₂
```

These coordinates can be plotted.

```python
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1]
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA of Materials Dataset")

plt.show()
```

This produces a scatter plot where

- every point represents one material,
- the x-axis corresponds to PC₁,
- the y-axis corresponds to PC₂.

---

# Interpreting the Scatter Plot

Suppose the PCA visualization looks like

```text
PC₂

^

|                     ● ●

|                 ● ● ●

|              ●

|        ● ●

|   ● ●

|________________________________________>

               PC₁
```

Several observations can immediately be made.

Materials located close together have similar descriptor values.

Materials separated by large distances are statistically different.

Dense regions indicate common classes of materials.

Sparse regions may indicate unusual or rare compounds.

---

# Distance in PCA Space

One of the most important ideas in PCA visualization is **distance**.

Suppose two materials appear very close together.

```text
●

 ●
```

These materials have similar descriptor combinations.

They often exhibit

- similar compositions,
- similar crystal structures,
- similar chemical environments,
- or similar physical properties.

Now consider two materials that are widely separated.

```text
●




                         ●
```

Their descriptors differ substantially.

Consequently,

their chemistry or structure may also differ significantly.

Although PCA reduces dimensionality,

it attempts to preserve these relative distances as much as possible.

---

# Understanding the Principal Axes

The axes in a PCA plot are not physical quantities.

The x-axis is

```text
PC₁
```

The y-axis is

```text
PC₂
```

Each principal component is a weighted combination of many descriptors.

For example,

PC₁ might be influenced by

- density,
- average atomic mass,
- packing fraction,
- atomic radius,
- electronegativity.

Therefore,

moving along the horizontal axis simultaneously changes many underlying descriptors.

Unlike a plot of

Density vs Band Gap,

the PCA axes do not correspond to single measurable properties.

---

# Coloring the Points

Scatter plots become much more informative when materials are colored according to a known property.

For example,

suppose we color points by

- band gap,
- formation energy,
- density,
- crystal system,
- or magnetic ordering.

Example code

```python
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Band Gap"],
    cmap="viridis"
)

plt.colorbar(label="Band Gap (eV)")

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.show()
```

Instead of displaying identical points,

the plot now reveals how the chosen property varies across materials space.

---

# Visualizing Different Material Classes

Suppose the dataset contains

- metals,
- semiconductors,
- insulators.

The PCA plot might resemble

```text
PC₂

^

|          ● ● ●

|       ● ● ● ●

|

|

|                      ▲ ▲ ▲

|                    ▲ ▲ ▲

|

|

|                                  ■ ■ ■

|                               ■ ■ ■

+-------------------------------------------------->

                    PC₁
```

where

```text
● Metals

▲ Semiconductors

■ Insulators
```

Notice that

different material classes naturally occupy different regions.

This separation occurs even though PCA never received class labels.

It is an entirely unsupervised technique.

---

# Discovering Hidden Structure

One of the primary goals of PCA visualization is exploration.

Rather than answering a predefined question,

PCA allows researchers to discover unexpected patterns.

For example,

the visualization may reveal

- natural clusters,
- continuous trends,
- isolated outliers,
- overlapping material families,
- unexplored regions of chemical space.

These discoveries often motivate further computational or experimental investigation.

---

# Identifying Outliers

PCA is also useful for detecting unusual materials.

Suppose most materials occupy a compact region.

```text
● ● ● ● ●

● ● ● ●

● ● ●
```

However,

one point lies far away.

```text
● ● ● ●

● ● ●

●




                             ★
```

The isolated point may represent

- a rare crystal structure,
- an unusual composition,
- a data entry error,
- a failed simulation,
- or a genuinely novel material.

Outlier detection is therefore one of the most valuable applications of PCA in high-throughput materials screening.

---

# Three-Dimensional PCA Visualization

Sometimes,

two principal components do not capture enough information.

In such cases,

three components can be used.

```python
pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)
```

The transformed data now contain

```text
PC₁

PC₂

PC₃
```

These can be visualized using a three-dimensional scatter plot.

Although more informative than a two-dimensional plot,

3D visualizations are generally more difficult to interpret and often require interactive plotting tools for effective exploration.

---

# How Much Information Does a PCA Plot Show?

Suppose

PC₁ explains

42%

of the variance,

and

PC₂ explains

23%.

The two-dimensional plot therefore represents

```text
42%

+

23%

=

65%
```

of the total variation in the original dataset.

This means

35%

of the information exists in higher principal components that are not shown.

Consequently,

a PCA plot is always a simplified representation of the original high-dimensional dataset.

This simplification is usually worthwhile because it enables intuitive visualization while preserving the dominant trends.

---

# PCA Visualization in Materials Informatics

A typical materials informatics workflow may proceed as follows.

```text
Crystal Structures

        │

        ▼

pymatgen

        │

        ▼

matminer

        │

        ▼

600 Descriptors

        │

        ▼

StandardScaler

        │

        ▼

Principal Component Analysis

        │

        ▼

PC₁ and PC₂

        │

        ▼

Scatter Plot

        │

        ▼

Explore Materials Space
```

Instead of examining hundreds of descriptor columns individually,

researchers gain an immediate overview of the entire dataset.

Patterns that are invisible in tabular data often become obvious in just a single figure.

---

# Example: Exploring a Materials Database

Imagine applying PCA to a database containing

```text
25,000 inorganic compounds.
```

After computing the first two principal components,

the scatter plot reveals

- one dense cluster of oxides,
- another cluster containing sulfides,
- a smaller region populated by nitrides,
- and several isolated materials located far from all major groups.

Without any supervision,

PCA has organized the dataset according to similarities in descriptor space.

A researcher can now investigate

- why certain materials cluster together,
- why others appear isolated,
- and whether unexplored regions of the plot contain promising candidates for future study.

---

# Strengths and Limitations of PCA Visualization

PCA visualization offers several important advantages.

**Advantages**

- Makes high-dimensional datasets interpretable.
- Preserves the dominant variation in the data.
- Reveals natural groupings of materials.
- Helps identify outliers and anomalies.
- Useful for exploratory data analysis before machine learning.

However,

it also has limitations.

**Limitations**

- Only linear relationships are preserved.
- Two-dimensional plots cannot capture all of the information in high-dimensional datasets.
- Principal components may be difficult to interpret physically.
- Different clusters may overlap after projection, even if they are separable in higher dimensions.

Understanding these limitations helps prevent overinterpreting PCA visualizations.

---

# Transition to Clustering

PCA provides an informative map of materials space, allowing us to visualize similarities and differences among materials.

However, visualization alone does not automatically identify groups.

When thousands or even millions of materials are present, manually inspecting a scatter plot is neither practical nor objective.

The next step is to use **clustering algorithms**, which automatically identify groups of similar materials based on their positions in descriptor space or PCA space. These algorithms form the foundation of **unsupervised learning**, enabling the discovery of material families, chemical trends, and hidden patterns without requiring labeled training data.

# 11.11 Introduction to Clustering — Finding Hidden Groups in Materials Data

In the first half of this chapter, we studied **Principal Component Analysis (PCA)**, one of the most widely used techniques for **dimensionality reduction**.

PCA answers the question

> **How can we represent hundreds of descriptors using only a few new variables while preserving most of the information?**

However, dimensionality reduction is only one type of unsupervised learning.

Another equally important problem is

> **Can a computer automatically discover groups of similar materials without being told what those groups are?**

This problem is known as **clustering**.

Clustering is one of the fundamental tasks of unsupervised machine learning and plays an important role in materials informatics. Instead of predicting a property such as band gap or formation energy, clustering attempts to discover the natural organization of materials based solely on their descriptors.

---

# What Is Clustering?

Suppose we have a dataset containing thousands of materials.

Each material is described using hundreds of descriptors generated with **matminer**.

Although the dataset contains no labels such as

- Metal
- Semiconductor
- Ceramic
- Battery Material

we would still like to know whether some materials are naturally similar to one another.

Clustering attempts to answer exactly this question.

Instead of requiring labeled examples,

the algorithm searches for patterns within the data itself.

Materials with similar descriptors are grouped together,

while materials with very different descriptors are placed into different groups.

---

# An Everyday Analogy

Imagine entering a large library where none of the books have been organized.

Thousands of books are scattered randomly across the floor.

No labels exist.

No shelves exist.

No librarian tells you

- which books are physics,
- which books are chemistry,
- which books are mathematics.

Instead,

you begin examining the books.

You notice that

- many books discuss quantum mechanics,
- some discuss organic chemistry,
- others focus on machine learning,
- while another collection contains biology textbooks.

Without being told anything beforehand,

you naturally begin creating groups.

```text
Books

↓

Look for Similarity

↓

Group Similar Books Together

↓

Physics

Chemistry

Mathematics

Biology
```

This is exactly what a clustering algorithm does.

Instead of books,

it examines materials.

Instead of chapter titles,

it examines descriptors.

---

# Clustering Is Unsupervised Learning

Recall the distinction between supervised and unsupervised learning.

In supervised learning,

we have

```text
Descriptors

+

Target Property
```

For example,

```text
Descriptors

↓

Random Forest

↓

Band Gap
```

The model learns the relationship between descriptors and the known target.

Clustering is fundamentally different.

Here,

we have only

```text
Descriptors
```

No labels exist.

No target property exists.

The algorithm must discover structure entirely on its own.

---

# Supervised vs Unsupervised Learning

The difference can be summarized as

```text
Supervised Learning

Descriptors

+

Known Labels

↓

Learn Mapping

↓

Predict Labels
```

versus

```text
Unsupervised Learning

Descriptors

↓

Discover Hidden Patterns

↓

Groups

Clusters

Relationships
```

This distinction is one of the defining characteristics of clustering.

---

# What Is a Cluster?

A **cluster** is a collection of data points that are more similar to each other than to points outside the group.

Suppose we plot two descriptors.

```text
^

|          ● ● ●

|        ● ● ●

|

|                          ▲ ▲ ▲

|                        ▲ ▲ ▲

|

|                                      ■ ■

|                                    ■ ■

+-------------------------------------------->
```

Even without labels,

most people immediately recognize

three separate groups.

These groups are called

**clusters**.

The purpose of clustering algorithms is to detect these groups automatically.

---

# Similarity Is the Key Idea

Every clustering algorithm depends on one fundamental concept:

**similarity**.

Materials that are similar should belong to the same cluster.

Materials that are dissimilar should belong to different clusters.

The challenge is therefore to define what "similar" actually means.

---

# Measuring Similarity

In machine learning,

similarity is usually measured using **distance**.

The closer two materials are,

the more similar they are assumed to be.

Suppose we have two materials represented in PCA space.

```text
Material A

↓

PC₁ = 1.2

PC₂ = 0.8

Material B

↓

PC₁ = 1.5

PC₂ = 0.6
```

These points lie close together.

They are therefore considered similar.

Now consider

```text
Material C

↓

PC₁ = 9.8

PC₂ = -4.1
```

Material C is much farther away.

It is likely to belong to a different cluster.

---

# Euclidean Distance

The most common distance measure is the **Euclidean distance**.

It is simply the ordinary straight-line distance between two points.

For two points

```text
(x₁, y₁)

and

(x₂, y₂)
```

the Euclidean distance is

```text
Distance = sqrt((x₂ - x₁)^2 + (y₂ - y₁)^2)
```

Although this equation is written for two dimensions,

the same idea extends naturally to hundreds of descriptors.

In PCA space,

the coordinates become

```text
PC₁

PC₂

PC₃

...

PCₙ
```

The Euclidean distance is then computed across all retained principal components.

---

# Why PCA Is Often Used Before Clustering

Recall that PCA removes

- redundant descriptors,
- correlated variables,
- unnecessary dimensions.

Instead of clustering

```text
600 Descriptors
```

we often cluster

```text
PC₁

PC₂

...

PC₁₅
```

This offers several advantages.

- Faster computation.
- Less noise.
- Reduced redundancy.
- Easier visualization.
- Better cluster separation in many datasets.

The workflow therefore becomes

```text
Crystal Structures

↓

Descriptor Generation

↓

Standardization

↓

PCA

↓

Clustering
```

This pipeline is widely used in modern materials informatics.

---

# What Makes a Good Cluster?

Although different clustering algorithms define clusters differently,

good clusters usually share several characteristics.

### Small Within-Cluster Distance

Points inside the same cluster should lie close together.

```text
● ● ●

● ●

●
```

The materials within the cluster are highly similar.

---

### Large Between-Cluster Distance

Different clusters should be well separated.

```text
● ● ●




                 ▲ ▲ ▲




                           ■ ■ ■
```

Large separation makes the clusters easier to distinguish.

---

### High Internal Similarity

Materials within a cluster often share

- similar compositions,
- similar structures,
- similar descriptor values,
- or similar physical behavior.

---

### Low External Similarity

Materials belonging to different clusters should differ significantly.

---

# Clustering Does Not Know Chemistry

One important fact is often overlooked.

Clustering algorithms know **nothing** about

- atoms,
- crystal structures,
- electronic structure,
- thermodynamics,
- or chemistry.

The algorithm sees only numbers.

For example,

instead of recognizing

```text
LiFePO₄
```

it sees

```text
1.42

0.83

5.17

...

432 Descriptor Values
```

The algorithm groups materials purely according to numerical similarity.

If chemically meaningful clusters emerge,

that information comes from the descriptors themselves—not from prior chemical knowledge embedded in the clustering algorithm.

---

# Applications of Clustering in Materials Informatics

Clustering has numerous applications throughout computational materials science.

Some common examples include

- discovering families of chemically similar materials,
- grouping crystal structures,
- identifying polymorphs,
- exploring chemical composition space,
- finding unusual materials,
- detecting failed simulations,
- organizing large materials databases,
- selecting representative compounds for further calculations.

Because clustering requires no labeled data,

it is particularly valuable when studying newly generated or poorly understood datasets.

---

# Example Workflow

A realistic materials informatics workflow might proceed as follows.

```text
Materials Project Database

        │

        ▼

Crystal Structures

        │

        ▼

pymatgen

        │

        ▼

matminer

        │

        ▼

550 Numerical Descriptors

        │

        ▼

Feature Scaling

        │

        ▼

Principal Component Analysis

        │

        ▼

First 20 Principal Components

        │

        ▼

Clustering Algorithm

        │

        ▼

Material Families

        │

        ├────────► Oxide Group

        ├────────► Sulfide Group

        ├────────► Intermetallic Group

        └────────► Novel Candidate Group
```

Notice that no labels were provided.

The clustering algorithm discovered these groups entirely from the descriptor data.

---

# Different Clustering Algorithms

Unfortunately,

there is no single clustering algorithm that works best for every dataset.

Different algorithms make different assumptions about the shape and distribution of the data.

Some algorithms assume

- spherical clusters,

others allow

- elongated clusters,

while others can identify

- clusters of arbitrary shape.

In this chapter,

we will study three of the most widely used clustering algorithms.

```text
Clustering Algorithms

│

├── K-Means

│      Best for compact,
│      spherical clusters

│

├── Hierarchical Clustering

│      Reveals nested
│      relationships

│

└── DBSCAN

       Detects arbitrary
       cluster shapes and
       identifies outliers
```

Each algorithm has different strengths and weaknesses, making them suitable for different materials informatics applications.

---

# From Visualization to Automatic Discovery

PCA allowed us to visualize the structure of high-dimensional materials space.

However, visualization alone requires human interpretation.

Clustering takes the next step.

Instead of asking researchers to manually inspect thousands of points on a scatter plot, clustering algorithms automatically partition the data into meaningful groups based on statistical similarity.

These groups often correspond to material families, chemical trends, structural classes, or entirely new regions of materials space that warrant further investigation.

In the next section, we begin with the simplest and most widely used clustering algorithm: **K-Means Clustering**, which partitions data into a user-defined number of compact clusters by iteratively assigning points to the nearest cluster center.

## 11.12 K-Means Clustering — The Most Popular Clustering Algorithm

Among all clustering algorithms, **K-Means** is undoubtedly the most widely used. It is simple, computationally efficient, easy to implement, and performs remarkably well on many real-world datasets.

Because of these advantages, K-Means is often the first clustering algorithm applied when exploring a new materials dataset.

However, despite its simplicity, K-Means is based on a powerful optimization principle.

Rather than randomly dividing the data into groups, K-Means repeatedly adjusts the cluster centers until the overall similarity within each cluster is maximized.

---

# The Goal of K-Means

Suppose we have thousands of materials represented in PCA space.

```text
PC₂

^

|

|      ● ● ●

|    ● ● ● ●

|

|

|                   ● ● ●

|                 ● ● ● ●

|

|

|                                   ● ● ●

|                                 ● ● ●

+------------------------------------------------->

                     PC₁
```

Even without labels,

it appears that three separate groups exist.

Our objective is to let the computer automatically identify these groups.

Instead of assigning labels manually,

we simply specify

```text
Number of Clusters = 3
```

The K-Means algorithm determines

- which materials belong together,
- where the center of each cluster should be,
- and how every material should be assigned.

---

# Why Is It Called "K-Means"?

The name comes from two ideas.

## K

Represents the number of clusters.

For example,

```text
K = 2

↓

Two clusters
```

```text
K = 5

↓

Five clusters
```

The user must choose this value before running the algorithm.

---

## Means

Each cluster is represented by its **mean**.

The mean is simply the average position of all points belonging to that cluster.

This average point is called the

**centroid**.

---

# What Is a Centroid?

Suppose five materials belong to the same cluster.

```text
●

●

●

●

●
```

The centroid is their average location.

```text
●

●

★

●

●
```

The star represents the center of the cluster.

Mathematically,

the centroid is computed by averaging every coordinate.

For example,

if a cluster contains

```text
PC₁

1

2

4

5
```

the centroid coordinate becomes

```text
(1 + 2 + 4 + 5) / 4

=

3
```

The same calculation is performed for every principal component.

---

# The Basic Idea

K-Means follows an extremely intuitive strategy.

```text
Guess Centers

↓

Assign Points

↓

Move Centers

↓

Assign Again

↓

Move Again

↓

Repeat

↓

Clusters Stop Changing
```

Eventually,

the cluster centers stabilize,

and the algorithm terminates.

---

# Step 1 — Choose the Number of Clusters

The first input is

```text
K
```

Suppose

```text
K = 3
```

Our goal is to divide the dataset into

three clusters.

---

# Step 2 — Initialize the Centroids

Initially,

the cluster centers are unknown.

K-Means therefore starts with an initial guess.

```text
PC₂

^

|          ★

|

|                      ★

|

|

|                                 ★

+----------------------------------------------->

               PC₁
```

These initial centroids may be

- selected randomly,
- or generated using more sophisticated methods such as K-Means++.

The choice of initialization can influence the final solution.

---

# Step 3 — Assign Every Point

Once the initial centroids have been chosen,

every material is assigned to the nearest centroid.

Suppose a material lies here.

```text
●
```

The distances to each centroid are computed.

```text
Distance to Cluster 1

↓

2.1

Distance to Cluster 2

↓

5.8

Distance to Cluster 3

↓

8.4
```

Since

Cluster 1

is closest,

the material is assigned to

Cluster 1.

This process is repeated for every material.

---

# Step 4 — Update the Centroids

After all materials have been assigned,

the centroid positions are recalculated.

Suppose Cluster 1 contains

```text
●

●

●

●

●
```

Its centroid becomes

```text
★

```

located at the average position of all points.

Notice that

the centroid usually moves.

```text
Old Center

★

↓

New Center

★
```

The movement reflects the current membership of the cluster.

---

# Step 5 — Repeat

Once the centroids move,

some materials may now be closer to different clusters.

Therefore,

the algorithm repeats the assignment step.

```text
Assign Points

↓

Update Centers

↓

Assign Again

↓

Update Again
```

This cycle continues until

the centroids no longer move significantly.

At this point,

the algorithm has converged.

---

# Visualizing the Iterations

### Initial Guess

```text
● ● ●

★

                 ● ● ●

             ★

                              ● ● ●

                        ★
```

---

### After One Iteration

```text
● ● ●

   ★

                ● ● ●

                   ★

                             ● ● ●

                                ★
```

---

### Final Solution

```text
● ● ●

  ★

                ● ● ●

                  ★

                              ● ● ●

                                ★
```

Notice that

the centroids gradually move toward the centers of the clusters.

---

# The Optimization Objective

K-Means attempts to minimize the total distance between every point and its assigned centroid.

More precisely,

it minimizes the **Within-Cluster Sum of Squares (WCSS)**.

The objective is

```text
For Every Cluster

↓

Compute Distance

↓

Square Distance

↓

Add All Distances

↓

Minimize Total
```

Smaller WCSS means

- tighter clusters,
- greater similarity within each cluster,
- better clustering.

---

# Why Squared Distance?

Instead of simply adding distances,

K-Means squares them.

For example,

```text
Distance

2

↓

Squared Distance

4
```

```text
Distance

5

↓

Squared Distance

25
```

Large distances become much more heavily penalized.

This encourages compact clusters.

---

# Geometric Interpretation

Imagine stretching rubber bands from every point to the cluster centroid.

```text
●------★

●------★

●------★

●------★
```

The total length of all rubber bands should be as small as possible.

The centroid automatically moves to minimize this total distance.

This provides an intuitive picture of why K-Means works.

---

# Example in Two Dimensions

Suppose we have

twelve materials.

```text
PC₂

^

|      ● ● ●

|      ● ● ●

|

|

|                 ● ● ●

|                 ● ● ●

+---------------------------------------->

                 PC₁
```

Choosing

```text
K = 2
```

produces

```text
Cluster 1

● ● ●

● ● ●

★

```

and

```text
Cluster 2

● ● ●

● ● ●

★

```

Every material belongs to exactly one cluster.

---

# K-Means in High Dimensions

Although the illustrations show only two dimensions,

K-Means works in any number of dimensions.

For example,

instead of clustering

```text
PC₁

PC₂
```

we may cluster

```text
PC₁

PC₂

PC₃

...

PC₁₅
```

The distance calculations simply extend into higher-dimensional space.

This is why PCA is often applied before K-Means.

Instead of clustering hundreds of correlated descriptors,

we cluster a smaller set of informative principal components.

---

# Applying K-Means in Materials Informatics

A common workflow is

```text
Crystal Structures

        │

        ▼

Descriptor Generation

        │

        ▼

Standardization

        │

        ▼

PCA

        │

        ▼

First 10 Principal Components

        │

        ▼

K-Means Clustering

        │

        ▼

Material Families
```

Each cluster often contains materials with similar

- compositions,
- crystal structures,
- electronic properties,
- or bonding environments.

---

# Advantages of K-Means

K-Means remains popular because of several important strengths.

- Simple to understand and implement.
- Fast even for very large datasets.
- Scales well to thousands or millions of samples.
- Works efficiently with PCA-reduced data.
- Produces compact, easy-to-interpret clusters.
- Available in virtually every machine learning library.

For many exploratory analyses,

K-Means is an excellent first choice.

---

# Limitations of K-Means

Despite its popularity,

K-Means has several important limitations.

First,

the number of clusters

```text
K
```

must be specified before the algorithm begins.

If the wrong value is chosen,

the resulting clusters may not reflect the true structure of the data.

Second,

K-Means assumes clusters are roughly

- spherical,
- compact,
- and similar in size.

Datasets containing elongated or irregularly shaped clusters may not be represented accurately.

Third,

the algorithm is sensitive to outliers.

A single extreme point can pull the centroid away from the true center of the cluster.

Finally,

different random initializations may produce different solutions.

For this reason,

modern implementations often use **K-Means++** initialization and perform multiple runs before selecting the best result.

---

# From Theory to Implementation

We now understand the complete logic behind K-Means.

1. Choose the number of clusters.
2. Initialize the centroids.
3. Assign every material to its nearest centroid.
4. Recalculate the centroids.
5. Repeat until convergence.

The algorithm is conceptually simple, yet remarkably effective for many materials informatics problems.

In the next section, we will implement **K-Means using scikit-learn**, learn how to choose an appropriate value of **K**, visualize the resulting clusters, and apply the algorithm to PCA-transformed materials descriptor datasets.

## 11.13 Implementing K-Means Clustering with Scikit-learn

In the previous section, we learned the theory behind the K-Means algorithm.

We now understand that K-Means

- starts with initial cluster centers,
- assigns every data point to its nearest center,
- updates the cluster centers,
- and repeats the process until the clusters no longer change.

Fortunately, just as with PCA, we do not need to implement this algorithm ourselves.

The **scikit-learn** library provides a highly optimized implementation that can cluster datasets containing thousands or even millions of samples efficiently.

In this section, we will learn how to apply K-Means to materials descriptor datasets and visualize the resulting clusters.

---

# Typical Workflow

In materials informatics, K-Means is rarely applied directly to hundreds of raw descriptors.

Instead, the workflow usually looks like

```text
Crystal Structures

        │

        ▼

pymatgen

        │

        ▼

matminer

        │

        ▼

Descriptor Matrix

        │

        ▼

StandardScaler

        │

        ▼

PCA

        │

        ▼

First 10 Principal Components

        │

        ▼

K-Means Clustering

        │

        ▼

Cluster Labels
```

Reducing the dimensionality before clustering often produces more meaningful and computationally efficient results.

---

# Importing the Required Libraries

The primary library is

```python
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

If PCA has already been performed,

our input dataset will usually be

```python
X_pca
```

instead of the original descriptor matrix.

---

# Creating the K-Means Model

Suppose we wish to divide the dataset into

three clusters.

Creating the model requires only one line.

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)
```

The parameter

```python
n_clusters
```

specifies

```text
K
```

the desired number of clusters.

The parameter

```python
random_state
```

ensures that the algorithm produces the same result every time the code is executed.

This improves reproducibility.

---

# Fitting the Model

The clustering algorithm is trained using

```python
kmeans.fit(X_pca)
```

Unlike supervised learning,

there are no target labels.

Only the feature matrix is required.

Internally,

the algorithm

- initializes the centroids,
- assigns every material,
- updates the centroids,
- repeats until convergence.

---

# Predicting Cluster Membership

After fitting,

every material receives a cluster label.

These labels are obtained using

```python
labels = kmeans.labels_
```

Example output

```python
array([
0,
0,
1,
2,
1,
2,
...
])
```

Each number identifies the cluster to which a material belongs.

For example,

| Material | Cluster |
|-----------|---------|
| Material 1 | 0 |
| Material 2 | 0 |
| Material 3 | 1 |
| Material 4 | 2 |
| Material 5 | 1 |

Notice that

the labels themselves have no physical meaning.

Cluster

```text
0
```

is not necessarily better or more important than

Cluster

```text
2
```

The numbering is completely arbitrary.

---

# Obtaining the Cluster Centers

The coordinates of the centroids are stored as

```python
kmeans.cluster_centers_
```

Example output

```text
Cluster 1

PC1 = -2.3

PC2 = 0.8

Cluster 2

PC1 = 1.4

PC2 = -1.7

Cluster 3

PC1 = 4.8

PC2 = 2.6
```

These coordinates represent the centers of the clusters in PCA space.

---

# Visualizing the Clusters

One of the greatest advantages of combining PCA with K-Means is that the results can be visualized directly.

```python
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap="tab10"
)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("K-Means Clustering")

plt.show()
```

Each color represents one cluster.

The resulting plot may resemble

```text
PC₂

^

|      ● ● ●

|    ● ● ●

|

|

|                     ▲ ▲ ▲

|                  ▲ ▲ ▲

|

|

|                                  ■ ■ ■

|                               ■ ■ ■

+-------------------------------------------------->

                    PC₁
```

where

```text
● Cluster 0

▲ Cluster 1

■ Cluster 2
```

---

# Plotting the Centroids

The cluster centers can also be displayed.

```python
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap="tab10"
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker="X",
    s=250,
    color="black",
    label="Centroids"
)

plt.legend()

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.show()
```

The visualization now becomes

```text
PC₂

^

|      ● ● ●

|    ● X ●

|

|

|                     ▲ ▲

|                  ▲ X ▲

|

|

|                               ■ X ■

|                                ■ ■

+-------------------------------------------------->

                    PC₁
```

The large

```text
X
```

marks the center of each cluster.

---

# Adding Cluster Labels to the DataFrame

Suppose we want to save the cluster assignments alongside the original data.

This is easily accomplished.

```python
df["Cluster"] = labels
```

The dataframe now becomes

| Formula | Density | Radius | Band Gap | Cluster |
|----------|---------:|--------:|----------:|---------:|
| Si | ... | ... | ... | 1 |
| Fe₂O₃ | ... | ... | ... | 2 |
| LiFePO₄ | ... | ... | ... | 0 |

The cluster labels can now be used for further analysis.

---

# Examining Cluster Sizes

It is often useful to determine how many materials belong to each cluster.

```python
df["Cluster"].value_counts()
```

Example output

```text
Cluster 0

523 Materials

Cluster 1

612 Materials

Cluster 2

465 Materials
```

Large differences in cluster size may indicate that

- one material family dominates the dataset,
- or the chosen value of K is not ideal.

---

# Computing the Inertia

One important quantity produced by K-Means is

```python
kmeans.inertia_
```

The inertia is the

**Within-Cluster Sum of Squares (WCSS)**.

Example

```python
print(kmeans.inertia_)
```

Output

```text
824.37
```

Smaller values indicate

- tighter clusters,
- lower average distance,
- better compactness.

However,

a smaller inertia does **not** always imply a better clustering because inertia always decreases as the number of clusters increases.

This leads to an important question.

> **How do we determine the optimal number of clusters?**

---

# Choosing Different Values of K

Suppose we try several values.

```python
K = 2

↓

Two clusters
```

```python
K = 4

↓

Four clusters
```

```python
K = 8

↓

Eight clusters
```

Each choice produces a different partition of the data.

Choosing

too few clusters

may combine unrelated materials,

while

too many clusters

may split natural material families into unnecessarily small groups.

---

# Example Materials Informatics Workflow

Consider a dataset of

30,000 crystal structures.

The workflow might be

```text
Materials Project

        │

        ▼

Crystal Structures

        │

        ▼

matminer

        │

        ▼

700 Descriptors

        │

        ▼

StandardScaler

        │

        ▼

PCA

        │

        ▼

15 Principal Components

        │

        ▼

K-Means

(K = 6)

        │

        ▼

Six Material Families
```

Each cluster can then be analyzed independently.

Researchers might compute

- average band gap,
- average formation energy,
- dominant crystal system,
- elemental composition,

for each cluster to understand the characteristics of the discovered material families.

---

# Interpreting the Clusters

Once clustering is complete,

the next step is scientific interpretation.

For example,

suppose

Cluster 0

contains mostly

- oxides,
- high band-gap materials,
- ionic compounds.

Cluster 1

contains

- transition-metal alloys,
- metallic compounds,
- high densities.

Cluster 2

contains

- layered materials,
- low-dimensional compounds,
- small formation energies.

Although K-Means itself has no knowledge of chemistry,

examining the materials assigned to each cluster often reveals meaningful physical trends.

---

# Limitations of the Implementation

Although scikit-learn makes K-Means extremely easy to use,

the quality of the results still depends on the dataset.

Poor clustering may occur if

- the descriptors are not informative,
- the features were not standardized,
- the chosen value of K is inappropriate,
- the clusters have irregular shapes,
- or the dataset contains many outliers.

Therefore,

successful clustering depends not only on the algorithm but also on careful data preparation and thoughtful interpretation.

---

# Preparing to Choose the Optimal Number of Clusters

Throughout this section, we assumed that the number of clusters

```text
K
```

was already known.

In practice,

this is rarely the case.

For most materials datasets,

the correct number of clusters is unknown before the analysis begins.

Fortunately,

machine learning provides several techniques for estimating an appropriate value of **K**.

The most widely used approach is the **Elbow Method**, which analyzes how the clustering error changes as the number of clusters increases. This method provides an intuitive way to balance model complexity with cluster quality and is often the first step when applying K-Means to an unfamiliar materials dataset.

## 11.14 Choosing the Number of Clusters — The Elbow Method

One of the biggest limitations of K-Means is that the user must specify the number of clusters,

```text
K
```

before the algorithm begins.

Unfortunately,

for most real materials datasets,

the correct value of

```text
K
```

is unknown.

Suppose we have a database containing

20,000 materials.

Should we divide them into

```text
2 Clusters?

5 Clusters?

10 Clusters?

25 Clusters?
```

There is usually no obvious answer.

Choosing too few clusters combines distinct material families together,

while choosing too many clusters splits natural groups into many smaller clusters.

The challenge, therefore, is to estimate a reasonable value of **K** before performing the final clustering.

One of the most widely used techniques for this purpose is the **Elbow Method**.

---

# Why Increasing K Always Improves the Fit

Recall that K-Means minimizes the

**Within-Cluster Sum of Squares (WCSS)**,

also called the **inertia**.

Smaller WCSS indicates that points lie closer to their cluster centers.

Suppose we begin with

```text
K = 1
```

Every material belongs to a single cluster.

```text
● ● ● ● ● ● ● ● ●

        ★
```

Since one centroid must represent the entire dataset,

many points are far away.

The WCSS is therefore very large.

---

Now suppose

```text
K = 2
```

```text
● ● ● ●

    ★




               ● ● ● ●

                  ★
```

Each centroid represents a smaller region.

The average distance decreases,

so WCSS becomes smaller.

---

If we continue increasing

```text
K
```

the clusters become even more compact.

```text
K = 3

↓

Lower WCSS

K = 4

↓

Even Lower WCSS

K = 5

↓

Even Lower WCSS
```

In fact,

adding more clusters **always** reduces WCSS.

This creates an important problem.

If we simply minimize WCSS,

the best solution would be

```text
K = Number of Materials
```

where every material forms its own cluster.

Clearly,

this is not useful.

---

# The Idea Behind the Elbow Method

Instead of looking for the smallest WCSS,

we examine **how quickly WCSS decreases**.

Initially,

adding extra clusters greatly improves the clustering.

Later,

adding more clusters provides only small improvements.

This transition often appears as an

**elbow**

in the curve.

---

# Visualizing the Elbow

Imagine plotting

```text
Number of Clusters

↓

K
```

on the horizontal axis,

and

```text
WCSS
```

on the vertical axis.

A typical curve looks like

```text
WCSS

^

|

| *

|  *

|    *

|       *

|          *

|            *

|             *

|              *

+--------------------------------------------->

   1   2   3   4   5   6   7   8

              K
```

Notice that

the curve decreases rapidly at first,

then gradually becomes flatter.

The location where the curve begins to flatten is called the

**elbow**.

---

# Why Is It Called the Elbow?

The graph resembles a bent arm.

```text
WCSS

^

|

| *

|  *

|    *

|      *

|       *

|        *

|         *

+------------------------------>

           ▲

         Elbow
```

Before the elbow,

each additional cluster provides a substantial improvement.

After the elbow,

the improvements become relatively small.

The elbow therefore represents a good balance between

- simplicity,
- and clustering quality.

---

# Computing the Elbow Curve

The Elbow Method simply repeats K-Means for several values of

```text
K
```

For example,

```text
K = 1

↓

Run K-Means

↓

Record WCSS

↓

K = 2

↓

Run Again

↓

Record WCSS

↓

...

↓

K = 10
```

Finally,

all WCSS values are plotted.

---

# Implementing the Elbow Method

Suppose we already have

```python
X_pca
```

The following code computes the WCSS for different values of K.

```python
wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(X_pca)

    wcss.append(kmeans.inertia_)
```

At the end,

the list

```python
wcss
```

contains one inertia value for every choice of

```text
K
```

---

# Plotting the Elbow Curve

The results are visualized using

```python
plt.figure(figsize=(8,6))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()
```

The resulting graph may resemble

```text
WCSS

^

| *

|  *

|    *

|       *

|          *

|            *

|             *

+--------------------------------------------->

 1  2  3  4  5  6  7  8  9 10

              ▲

            Elbow
```

Suppose the elbow occurs at

```text
K = 4
```

We would then choose

```python
KMeans(n_clusters=4)
```

for the final clustering.

---

# Example

Suppose the computed inertia values are

| K | WCSS |
|---:|------:|
| 1 | 2450 |
| 2 | 1320 |
| 3 | 760 |
| 4 | 520 |
| 5 | 470 |
| 6 | 435 |
| 7 | 410 |
| 8 | 392 |

Notice the pattern.

From

```text
K = 1

↓

K = 2
```

the improvement is very large.

From

```text
K = 2

↓

K = 3
```

the improvement is still significant.

However,

after

```text
K = 4
```

the reductions become much smaller.

This suggests that

```text
K = 4
```

captures most of the meaningful structure in the dataset.

---

# Interpreting the Elbow

The elbow should not be viewed as an exact mathematical answer.

Instead,

it provides a reasonable estimate.

Sometimes,

the elbow is very clear.

```text
*

 *

   *

      *

          *

            *

             *

```

Sometimes,

the curve decreases smoothly.

```text
*

 *

  *

   *

    *

     *

      *

```

In such cases,

choosing the optimal number of clusters becomes more subjective.

Additional evaluation methods may be required.

---

# Materials Informatics Example

Suppose we generate

500 descriptors

for

15,000 inorganic compounds.

After applying PCA,

the first

20 principal components

are clustered using K-Means.

Running the Elbow Method produces the following observation.

```text
K = 2

↓

Very Large Improvement

K = 3

↓

Large Improvement

K = 4

↓

Moderate Improvement

K = 5

↓

Very Small Improvement

K = 6

↓

Tiny Improvement
```

The elbow appears near

```text
K = 4
```

This suggests that the descriptor space naturally separates into approximately four major material families.

These clusters might later correspond to

- oxides,
- sulfides,
- intermetallic compounds,
- and layered materials,

although the clustering algorithm itself has no knowledge of chemistry.

---

# Advantages of the Elbow Method

The Elbow Method is popular because it is

- simple,
- intuitive,
- easy to visualize,
- computationally inexpensive,
- supported by virtually every machine learning library.

It provides a useful first estimate of the appropriate number of clusters.

---

# Limitations of the Elbow Method

Despite its popularity,

the Elbow Method is not perfect.

One limitation is that

the elbow is not always clearly visible.

Some datasets produce smooth curves with no obvious turning point.

Another limitation is that

the method considers only the compactness of clusters.

It does not measure

- how well separated the clusters are,
- whether clusters overlap,
- or whether the chosen clusters are scientifically meaningful.

Consequently,

the Elbow Method should be viewed as a practical guideline rather than a definitive rule.

---

# Beyond the Elbow Method

Because the Elbow Method sometimes produces ambiguous results,

other cluster evaluation techniques have been developed.

These include

- the **Silhouette Score**,
- the **Calinski-Harabasz Index**,
- the **Davies-Bouldin Index**,
- and several stability-based approaches.

Among these,

the **Silhouette Score** is the most widely used because it measures both

- cluster compactness,
- and cluster separation.

In the next section, we will study the **Silhouette Score**, understand its mathematical intuition, implement it using scikit-learn, and learn how it complements the Elbow Method when selecting the optimal number of clusters for materials informatics datasets.

## 11.15 Evaluating Cluster Quality Using the Silhouette Score

In the previous section, we learned how the **Elbow Method** helps estimate the number of clusters by analyzing the **Within-Cluster Sum of Squares (WCSS)**.

Although the Elbow Method is simple and widely used, it has one important limitation.

Many real datasets do **not** produce a clear elbow.

For example,

```text
WCSS

^

|

| *

|  *

|   *

|    *

|      *

|        *

|          *

+--------------------------------------------->

   1   2   3   4   5   6   7
```

Where is the elbow?

There is no obvious answer.

In situations like this, we need another way to evaluate clustering quality.

One of the most popular methods is the **Silhouette Score**.

Unlike the Elbow Method, which only measures how compact the clusters are, the Silhouette Score evaluates both

- how close a material is to other members of its own cluster, and
- how far it is from materials in neighboring clusters.

For this reason, it is often considered a more reliable measure of clustering quality.

---

# The Intuition Behind the Silhouette Score

Imagine you are standing inside a group of people.

There are two possibilities.

### Situation 1

Everyone around you belongs to your own group.

The nearest people from another group are far away.

```text
● ● ●

● ★ ●

● ● ●




               ▲ ▲ ▲

               ▲ ▲ ▲
```

You clearly belong to your current group.

This is a good clustering result.

---

### Situation 2

You stand halfway between two groups.

```text
● ●

 ●

     ★

         ▲

       ▲ ▲
```

Now it is difficult to decide which group you belong to.

This is a poor clustering result.

---

The Silhouette Score measures exactly this idea.

A good cluster has

- small distances within the cluster,
- and large distances to neighboring clusters.

---

# Two Important Distances

For every material,

the Silhouette Score considers two distances.

---

## Distance A

Distance

```text
A
```

is the average distance from a material to all other materials in the **same cluster**.

```text
Cluster

● ●

● ★ ●

● ●
```

If the material lies near the center,

Distance A is small.

If it lies near the edge,

Distance A becomes larger.

Therefore,

smaller values of

```text
A
```

are better.

---

## Distance B

Distance

```text
B
```

is the average distance from the material to the **nearest neighboring cluster**.

```text
Cluster 1

● ●

● ★




Cluster 2

▲ ▲ ▲

▲ ▲
```

If the neighboring cluster is far away,

Distance B is large.

Large values of

```text
B
```

are desirable.

---

# The Silhouette Score Formula

Using these two distances,

the Silhouette Score for one material is

```text
Silhouette Score = (B - A) / max(A, B)
```

where

```text
A

↓

Average distance to points
inside the same cluster
```

and

```text
B

↓

Average distance to points
in the nearest neighboring cluster
```

Fortunately,

scikit-learn computes this automatically,

so we rarely calculate it manually.

Understanding the meaning of the equation is much more important than memorizing it.

---

# Understanding the Score

The Silhouette Score always lies between

```text
-1

and

1
```

Every value has a useful interpretation.

---

## Score Close to +1

```text
Silhouette Score

≈ 1
```

The material is

- very close to its own cluster,
- very far from neighboring clusters.

This indicates excellent clustering.

---

## Score Close to 0

```text
Silhouette Score

≈ 0
```

The material lies near the boundary between two clusters.

Its assignment is uncertain.

---

## Score Less Than 0

```text
Silhouette Score

< 0
```

The material is actually closer to another cluster than its assigned cluster.

This usually indicates

- an incorrect assignment,
- overlapping clusters,
- or a poor choice of K.

---

# Visual Interpretation

Excellent clustering

```text
Cluster 1

● ● ●

● ★ ●

● ●




Cluster 2

▲ ▲ ▲

▲ ▲ ▲
```

Silhouette Score

```text
≈ 0.9
```

---

Borderline material

```text
● ●

 ★

      ▲ ▲
```

Silhouette Score

```text
≈ 0
```

---

Incorrect assignment

```text
●




        ★

      ▲ ▲ ▲
```

Silhouette Score

```text
< 0
```

---

# Average Silhouette Score

The overall clustering quality is measured by averaging the scores of all materials.

Example

| Average Score | Interpretation |
|--------------:|---------------|
| 0.80–1.00 | Excellent clustering |
| 0.60–0.80 | Very good clustering |
| 0.40–0.60 | Reasonable clustering |
| 0.20–0.40 | Weak clustering |
| Less than 0.20 | Poor clustering |

These ranges are not strict rules,

but they provide useful guidelines.

---

# Computing the Silhouette Score

Scikit-learn provides a built-in function.

```python
from sklearn.metrics import silhouette_score
```

Suppose we already have

```python
X_pca
```

and

```python
labels
```

The average score is computed using

```python
score = silhouette_score(
    X_pca,
    labels
)

print(score)
```

Example output

```text
0.71
```

A score of

```text
0.71
```

indicates that the clusters are

- compact,
- well separated,
- and generally well defined.

---

# Comparing Different Values of K

The Silhouette Score becomes particularly useful when comparing different choices of

```text
K
```

Example

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = kmeans.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        labels
    )

    print(k, score)
```

Example output

```text
K = 2

0.53

K = 3

0.67

K = 4

0.74

K = 5

0.61

K = 6

0.55
```

The highest score occurs at

```text
K = 4
```

suggesting that four clusters provide the best separation.

---

# Elbow Method vs Silhouette Score

Although both methods help choose

```text
K
```

they evaluate different aspects of clustering.

| Elbow Method | Silhouette Score |
|--------------|------------------|
| Uses WCSS | Uses average similarity |
| Measures cluster compactness | Measures compactness and separation |
| Visual inspection required | Produces a numerical score |
| Sometimes ambiguous | Easier to compare different values of K |

In practice,

many researchers use **both methods together**.

If both methods suggest the same value of

```text
K
```

confidence in the clustering result increases.

---

# Example from Materials Informatics

Suppose we compute

400 descriptors

for

12,000 crystalline materials.

After PCA,

the first

15 principal components

are clustered.

The evaluation produces

| K | WCSS | Silhouette Score |
|---:|------:|----------------:|
| 2 | High | 0.51 |
| 3 | Lower | 0.63 |
| 4 | Lower | 0.76 |
| 5 | Slightly Lower | 0.65 |
| 6 | Lower | 0.58 |

The Elbow Method indicates that improvements become small after

```text
K = 4
```

The Silhouette Score also reaches its maximum at

```text
K = 4
```

Because both methods agree,

choosing

```text
K = 4
```

is well justified.

---

# Limitations of the Silhouette Score

Although the Silhouette Score is extremely useful,

it is not perfect.

One limitation is that it generally favors

- compact,
- well-separated,
- nearly spherical clusters.

Datasets containing

- elongated clusters,
- curved clusters,
- or clusters with varying densities

may receive lower scores even if the clustering is scientifically meaningful.

Another limitation is computational cost.

Computing the Silhouette Score requires many distance calculations,

making it slower than simply evaluating inertia,

especially for very large datasets.

Finally,

a high Silhouette Score does not automatically imply scientific significance.

A statistically good cluster may still lack meaningful chemical or physical interpretation.

Therefore,

cluster evaluation should always be combined with domain knowledge.

---

# Practical Guidelines

When applying K-Means to a materials dataset,

a common strategy is

```text
Generate Descriptors

        │

        ▼

Standardize Features

        │

        ▼

Apply PCA

        │

        ▼

Run K-Means
for Several Values of K

        │

        ├────────► Compute WCSS

        │

        ├────────► Plot Elbow Curve

        │

        ├────────► Compute Silhouette Score

        │

        ▼

Choose Best K

        │

        ▼

Interpret Material Families
```

This workflow combines multiple evaluation techniques to produce more reliable clustering results.

---

# Beyond K-Means

K-Means assumes that clusters are

- compact,
- roughly spherical,
- and similar in size.

Many materials datasets violate these assumptions.

For example,

materials may form

- elongated trends,
- nested groups,
- irregularly shaped clusters,
- or isolated outliers.

In these situations,

other clustering algorithms often perform much better.

The next algorithm we will study is **Hierarchical Clustering**, which builds a tree of relationships between materials instead of forcing them into a fixed number of clusters. This approach provides a richer view of the organization of materials space and is especially useful for exploring similarities between material families.

## 11.16 Hierarchical Clustering — Building a Tree of Materials

So far, we have studied **K-Means Clustering**, one of the most widely used clustering algorithms.

K-Means is fast, simple, and effective, but it has several important limitations.

For example,

- the number of clusters must be specified in advance,
- clusters are assumed to be roughly spherical,
- and every material is assigned directly to a cluster.

However, many materials datasets do not naturally satisfy these assumptions.

Instead of forming clearly separated spherical groups,

materials often exhibit

- nested families,
- gradual transitions,
- multiple levels of similarity,
- or hierarchical relationships.

To analyze these kinds of datasets, we use **Hierarchical Clustering**.

Unlike K-Means,

Hierarchical Clustering does not simply divide the data into clusters.

Instead,

it builds a **tree-like structure** that describes how materials are related to one another.

---

# The Main Idea

Imagine you have collected

20 different materials.

Rather than immediately dividing them into

four clusters,

you first identify

the two most similar materials.

These become a small group.

Next,

another pair of similar materials is grouped together.

Eventually,

small groups merge into larger groups.

Those larger groups merge again.

The process continues until

every material belongs to one large tree.

```text
Individual Materials

↓

Small Groups

↓

Larger Groups

↓

Material Families

↓

Entire Dataset
```

Instead of producing only the final clusters,

Hierarchical Clustering records **every merging step**.

---

# An Everyday Analogy

Imagine organizing photographs.

You begin with hundreds of pictures.

First,

you group

family photographs.

Then,

vacation photographs.

Then,

work photographs.

Later,

you merge

family and vacation pictures

into

personal photographs.

Finally,

all photographs belong to one complete collection.

```text
Photos

↓

Family

Vacation

Work

↓

Personal

Professional

↓

Complete Collection
```

This gradual merging process is exactly how hierarchical clustering operates.

---

# Hierarchical Structure

Unlike K-Means,

which produces only a single partition,

Hierarchical Clustering creates multiple levels of organization.

```text
Entire Dataset

│

├────────────── Group A

│               │

│               ├──── Subgroup A1

│               └──── Subgroup A2

│

└────────────── Group B

                │

                ├──── Subgroup B1

                └──── Subgroup B2
```

Notice that

large groups contain smaller groups,

which themselves contain even smaller groups.

This hierarchical organization often reflects natural relationships between materials.

---

# Two Types of Hierarchical Clustering

There are two main approaches.

```text
Hierarchical Clustering

│

├── Agglomerative

└── Divisive
```

---

## Agglomerative Clustering

Agglomerative clustering begins with

every material as its own cluster.

Then,

clusters are gradually merged.

```text
20 Materials

↓

20 Clusters

↓

15 Clusters

↓

10 Clusters

↓

5 Clusters

↓

1 Cluster
```

This is the most commonly used version and the one implemented in most machine learning libraries.

---

## Divisive Clustering

Divisive clustering works in the opposite direction.

The algorithm starts with

one giant cluster.

It repeatedly divides that cluster into smaller groups.

```text
1 Cluster

↓

2 Clusters

↓

4 Clusters

↓

8 Clusters

↓

Individual Groups
```

Although conceptually interesting,

divisive clustering is less commonly used because it is computationally more expensive.

In this chapter,

we focus on **Agglomerative Hierarchical Clustering**.

---

# Agglomerative Clustering Step by Step

The algorithm follows a simple sequence.

```text
Every Material
↓

Individual Cluster
↓

Find Closest Pair
↓

Merge Them
↓

Update Distances
↓

Repeat
↓

Single Tree
```

Every iteration reduces

the number of clusters

by exactly one.

---

# Step 1 — Start with Individual Clusters

Suppose we have six materials.

Initially,

each material forms its own cluster.

```text
A

B

C

D

E

F
```

At this stage,

there are

```text
6 Clusters
```

---

# Step 2 — Compute Distances

The distance between every pair of clusters is calculated.

Conceptually,

the distance matrix might look like

| | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 2 | 7 | 8 | 10 | 12 |
| B | 2 | 0 | 6 | 9 | 11 | 13 |
| C | 7 | 6 | 0 | 3 | 8 | 9 |
| D | 8 | 9 | 3 | 0 | 4 | 6 |
| E |10 |11 |8 |4 |0 |2 |
| F |12 |13 |9 |6 |2 |0 |

Smaller numbers indicate greater similarity.

---

# Step 3 — Merge the Closest Pair

Suppose

A and B

have the smallest distance.

They merge.

```text
(A B)

C

D

E

F
```

The number of clusters decreases.

```text
6

↓

5
```

---

# Step 4 — Update the Distances

Since

A and B

have become one cluster,

the distance matrix must be recalculated.

The algorithm determines

how far

Cluster (A B)

is from

C,

D,

E,

and F.

This calculation depends on the **linkage method**, which we will study shortly.

---

# Step 5 — Repeat

The algorithm again finds

the closest pair.

Suppose

E and F

are now closest.

```text
(A B)

C

D

(E F)
```

Now there are

```text
4 Clusters
```

The process continues.

Eventually,

the clusters become

```text
((A B) C)

(D (E F))
```

Finally,

everything merges into one tree.

```text
(((A B) C) (D (E F)))
```

---

# The Dendrogram

The tree produced by Hierarchical Clustering is called a

**dendrogram**.

A dendrogram records

every merging operation.

Example

```text
|

|          -------- A

|

|     -----|

|     |     -------- B

|

|-----------|

|           |--------- C

|

|----------------------|

|                      |--------- D

|                      |

|                      |----- E

|                            |

|                            F
```

Although this diagram may initially appear unfamiliar,

it contains an enormous amount of information.

---

# Reading a Dendrogram

Every leaf represents

one material.

```text
A

B

C

D

E

F
```

Moving upward,

branches join together.

The height of each merge represents

the distance between the clusters.

For example,

if

A

and

B

merge near the bottom,

they are highly similar.

If

Cluster (A B)

joins

C

much higher,

then

C

is less similar.

---

# What Does the Height Mean?

The vertical axis represents

the distance

or

dissimilarity

between clusters.

```text
High Merge

↓

Clusters are very different
```

```text
Low Merge

↓

Clusters are very similar
```

Therefore,

materials that merge near the bottom of the dendrogram are closely related.

Materials that merge only near the top are much less similar.

---

# Cutting the Dendrogram

One of the most useful features of hierarchical clustering is that

we do **not** have to decide the number of clusters before running the algorithm.

Instead,

we simply cut the dendrogram at a chosen height.

Example

```text
|

|          -------- A

|

|     -----|

|     |     -------- B

|--------------------------- Cut Here

|-----------|

|           |--------- C

|

|----------------------|

|                      |--------- D

|                      |

|                      |----- E

|                            |

|                            F
```

Everything connected below the cut becomes one cluster.

Changing the cut height changes the number of clusters.

---

# Advantages Over K-Means

This flexibility is one of the major advantages of hierarchical clustering.

Instead of repeating the algorithm for

```text
K = 2

K = 3

K = 4

K = 5
```

the dendrogram already contains

all of these possibilities.

A single clustering run can produce

- two clusters,
- five clusters,
- ten clusters,

simply by selecting different cut heights.

---

# Applications in Materials Informatics

Hierarchical clustering is especially useful when studying

- chemically related compounds,
- crystal structure families,
- polymorphs,
- alloy systems,
- compositional trends,
- and evolutionary relationships between materials.

For example,

suppose we analyze descriptors for

thousands of oxides.

The dendrogram may reveal

```text
Oxides

│

├── Transition Metal Oxides

│      ├── Perovskites

│      ├── Spinels

│      └── Rutile Structures

│

└── Main Group Oxides

       ├── Silicates

       ├── Borates

       └── Phosphates
```

This multi-level organization is difficult to obtain using K-Means.

---

# Advantages of Hierarchical Clustering

Hierarchical clustering offers several important benefits.

- The number of clusters does not need to be specified beforehand.
- Produces an interpretable tree of relationships.
- Reveals nested material families.
- Useful for exploratory analysis.
- Works well for small and medium-sized datasets.
- Allows clustering at multiple levels of similarity.

These characteristics make it particularly valuable when researchers wish to understand the structure of a dataset rather than simply partition it.

---

# Limitations

Despite its strengths,

Hierarchical Clustering also has limitations.

- It is computationally more expensive than K-Means.
- Large datasets can produce very large dendrograms that are difficult to interpret.
- Once two clusters are merged, the decision cannot be undone.
- Results depend strongly on the chosen linkage method and distance metric.

For datasets containing hundreds of thousands of materials,

K-Means is often significantly faster.

However,

for moderate-sized datasets where understanding relationships is important,

Hierarchical Clustering is frequently the preferred choice.

---

# The Importance of Linkage Methods

So far,

we have repeatedly stated that the algorithm merges

the **closest clusters**.

But this raises an important question.

> **How do we define the distance between two clusters containing many materials?**

For individual materials,

distance is easy to calculate.

For clusters,

there are several possible definitions.

The answer depends on the **linkage method**.

Different linkage methods can produce very different dendrograms, even when applied to exactly the same dataset.

In the next section, we will study the most common linkage methods—**single linkage**, **complete linkage**, **average linkage**, and **Ward linkage**—and learn how each one influences the clustering of materials data.

## 11.17 Linkage Methods in Hierarchical Clustering

In the previous section, we learned that Agglomerative Hierarchical Clustering repeatedly merges the **closest pair of clusters** until every material belongs to one large hierarchical tree.

However, this immediately raises an important question.

Suppose Cluster A contains

```text
5 Materials
```

and Cluster B contains

```text
8 Materials.
```

How do we define the **distance** between these two clusters?

Unlike individual materials, which have a single position in descriptor space, clusters contain many points.

There is no unique way to measure the distance between them.

Instead, Hierarchical Clustering provides several different definitions, known as **linkage methods**.

The choice of linkage method has a significant impact on the final dendrogram and the resulting clusters.

---

# Why Linkage Matters

Consider two clusters.

```text
Cluster A

● ● ●

● ●



Cluster B

                ▲ ▲ ▲

                ▲ ▲
```

The question is

> **How far apart are these two clusters?**

Possible answers include

- the distance between the nearest materials,
- the distance between the farthest materials,
- the average distance,
- or the distance between the cluster centers.

Each definition produces different clustering behavior.

---

# The Four Most Common Linkage Methods

The most widely used linkage methods are

```text
Hierarchical Clustering

│

├── Single Linkage

├── Complete Linkage

├── Average Linkage

└── Ward Linkage
```

Each method defines cluster distance differently.

---

# 1. Single Linkage

Single linkage defines the distance between two clusters as

> **the smallest distance between any pair of materials belonging to the two clusters.**

In other words,

the algorithm searches for the two closest materials.

---

## Example

```text
Cluster A

● ● ●

●



                   ▲

                ▲ ▲ ▲

               Cluster B
```

Only the closest pair matters.

```text
● ---------------- ▲
```

Even if all other materials are much farther apart,

the clusters may still merge because of this single short distance.

---

## Characteristics

Single linkage tends to create

- long,
- thin,
- chain-like clusters.

For this reason,

it is sometimes called the

**nearest-neighbor method**.

---

## Advantages

- Can detect elongated clusters.
- Can discover irregular shapes.
- Simple to understand.

---

## Disadvantages

Single linkage often suffers from the

**chaining effect**.

Suppose materials form a long chain.

```text
● ● ● ● ● ● ● ● ●
```

Instead of identifying separate groups,

single linkage may merge them into one enormous cluster.

This behavior is often undesirable.

---

# 2. Complete Linkage

Complete linkage takes the opposite approach.

Instead of using the nearest materials,

it uses the **farthest pair**.

The cluster distance becomes

> **the largest distance between any pair of materials in the two clusters.**

---

## Example

```text
Cluster A

● ● ●



                     ▲ ▲ ▲

                     ▲
```

Now the algorithm considers

```text
● -------------------------- ▲
```

the maximum separation.

---

## Characteristics

Complete linkage produces

- compact,
- well-separated,
- relatively spherical clusters.

Because it considers the worst-case distance,

clusters do not merge unless **every** member is reasonably close.

---

## Advantages

- Produces compact clusters.
- Less sensitive to chaining.
- Often yields cleaner separation.

---

## Disadvantages

- Sensitive to outliers.
- May split elongated clusters into multiple pieces.

---

# 3. Average Linkage

Average linkage provides a compromise between

single

and

complete linkage.

Instead of considering only one pair,

it computes the **average distance** between every pair of materials belonging to the two clusters.

---

## Example

Suppose

Cluster A

contains

```text
3 Materials
```

and

Cluster B

contains

```text
4 Materials.
```

The algorithm computes

every possible pairwise distance,

then calculates the average.

```text
Average Distance

↓

Cluster Distance
```

---

## Characteristics

Average linkage generally produces

- balanced clusters,
- moderate compactness,
- moderate flexibility.

It is less affected by

- extreme outliers,
- and chaining,

than the previous methods.

---

## Advantages

- More stable than single linkage.
- Less sensitive to outliers than complete linkage.
- Produces balanced dendrograms.

---

## Disadvantages

- Computationally more expensive.
- May not perform well when cluster densities differ greatly.

---

# 4. Ward Linkage

Ward linkage is fundamentally different.

Instead of directly measuring distances between clusters,

it measures

**how much the total variance increases when two clusters are merged.**

The algorithm always chooses the merge that causes the smallest increase in variance.

Conceptually,

it asks

> **Which merge keeps the clusters as compact as possible?**

---

## Intuition

Suppose we have two possible merges.

### Option 1

Produces

very compact clusters.

```text
● ●

● ●



▲ ▲

▲ ▲
```

---

### Option 2

Produces

widely scattered clusters.

```text
●      ●

      ●




▲

          ▲ ▲
```

Ward linkage chooses

Option 1

because it minimizes the increase in cluster variance.

---

## Characteristics

Ward linkage generally produces

- compact clusters,
- approximately spherical groups,
- balanced cluster sizes.

For many scientific datasets,

including materials informatics,

it is often the preferred linkage method.

---

## Advantages

- Produces compact clusters.
- Often gives excellent separation.
- Works well with Euclidean distance.
- Widely used in scientific research.

---

## Disadvantages

- Favors spherical clusters.
- Less effective for highly irregular cluster shapes.

---

# Comparing the Linkage Methods

The four methods can be summarized as follows.

| Linkage Method | Distance Definition | Typical Cluster Shape |
|----------------|--------------------|-----------------------|
| Single | Minimum distance | Long, chain-like |
| Complete | Maximum distance | Compact |
| Average | Mean distance | Balanced |
| Ward | Minimum increase in variance | Compact and balanced |

Each method emphasizes a different aspect of similarity.

---

# Visual Comparison

Suppose the same dataset is clustered using different linkage methods.

---

### Single Linkage

```text
● ● ● ● ● ● ● ●
```

One long chain may form.

---

### Complete Linkage

```text
● ● ●

● ●




▲ ▲ ▲

▲ ▲
```

Produces compact groups.

---

### Average Linkage

```text
● ● ●

● ●




▲ ▲

▲ ▲ ▲
```

Produces intermediate behavior.

---

### Ward Linkage

```text
● ●

● ●




▲ ▲

▲ ▲
```

Produces compact,

well-balanced clusters.

---

# Choosing a Linkage Method

There is no universally best linkage method.

The appropriate choice depends on

- the dataset,
- the descriptor distribution,
- and the scientific objective.

General guidelines include

| Goal | Recommended Linkage |
|------|----------------------|
| Detect elongated clusters | Single |
| Obtain compact groups | Complete |
| General-purpose clustering | Average |
| Materials informatics, PCA data | Ward |

In practice,

**Ward linkage** is often the default choice for descriptor-based materials datasets because it tends to produce stable and interpretable clusters.

---

# Implementing Hierarchical Clustering in Scikit-learn

Hierarchical clustering is implemented in scikit-learn through the

```python
AgglomerativeClustering
```

class.

For example,

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)
```

The model is then fitted using

```python
labels = model.fit_predict(X_pca)
```

The resulting

```python
labels
```

contain the cluster assignment for every material,

just as in K-Means.

Unlike K-Means,

however,

the clustering process is based on progressively merging clusters rather than repeatedly updating centroids.

---

# Linkage Methods in Materials Informatics

Suppose we compute

600 descriptors

for

10,000 inorganic compounds,

reduce them to

20 principal components,

and apply Hierarchical Clustering.

Different linkage methods may reveal different scientific structures.

For example,

- **Single linkage** may connect materials through gradual compositional changes.
- **Complete linkage** may isolate highly distinct material families.
- **Average linkage** may reveal moderate-scale chemical groupings.
- **Ward linkage** may separate the dataset into compact families of structurally and chemically similar compounds.

Comparing the results obtained using different linkage methods often provides additional insight into the organization of materials space.

---

# Transition to Density-Based Clustering

Both K-Means and Hierarchical Clustering work well for many datasets.

However,

they share an important limitation.

Both tend to perform best when clusters are relatively compact.

Real materials datasets do not always satisfy this assumption.

Some material families form

- curved structures,
- irregular shapes,
- clusters with different densities,
- or contain isolated outliers.

For these situations,

a fundamentally different approach is needed.

Instead of grouping materials based on distances to cluster centers or hierarchical merges, the next algorithm—**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**—identifies clusters based on **local point density**. This enables it to discover arbitrarily shaped clusters while naturally identifying outliers as noise, making it one of the most powerful unsupervised learning methods for exploratory materials informatics.

## 11.18 DBSCAN — Density-Based Clustering of Applications with Noise

In the previous sections, we studied two of the most widely used clustering algorithms.

- **K-Means**
- **Hierarchical Clustering**

Although both algorithms are extremely useful, they share several important limitations.

For example,

- K-Means assumes clusters are approximately spherical.
- Hierarchical clustering often favors compact groups depending on the linkage method.
- Both algorithms struggle when clusters have irregular shapes.
- Neither algorithm naturally identifies outliers.

Many real materials datasets violate these assumptions.

For example,

- layered materials may form elongated distributions,
- composition gradients may produce curved structures,
- defect-containing materials may appear as isolated points,
- and high-throughput databases often contain erroneous calculations or unusual compounds.

To analyze these datasets effectively, we need a different approach.

Instead of defining clusters by their centers or by repeatedly merging clusters, **DBSCAN** defines clusters using **density**.

---

# What Is DBSCAN?

DBSCAN stands for

```text
Density-Based Spatial Clustering
of
Applications with Noise
```

Although the name appears complicated,

the underlying idea is remarkably simple.

Instead of asking

> Which points are closest to a centroid?

DBSCAN asks

> Which points belong to a dense neighborhood?

If many materials are packed closely together,

they form a cluster.

If a material lies alone,

far from all other materials,

it is treated as **noise**.

---

# An Everyday Analogy

Imagine looking at a night sky.

Some stars form dense constellations.

Other stars appear isolated.

```text
*  *  *

 *  *




                 *  *

               *   *




                              *
```

Without knowing astronomy,

you would naturally identify

- the first dense group,
- the second dense group,

while ignoring

the isolated star.

This is exactly how DBSCAN works.

---

# The Fundamental Idea

DBSCAN searches for regions where points are densely packed.

```text
High Density

↓

Cluster
```

```text
Low Density

↓

Noise
```

Unlike K-Means,

the algorithm never asks

```text
How many clusters exist?
```

Instead,

clusters emerge naturally from the data.

---

# A Density-Based View

Consider the following dataset.

```text
PC₂

^

|

|      ● ● ●

|     ● ● ●

|

|

|                     ▲ ▲ ▲

|                   ▲ ▲ ▲

|

|

|                                  ■

+-------------------------------------------------->

                     PC₁
```

The dense groups become

clusters.

The isolated point

```text
■
```

is classified as

noise.

---

# Two Important Parameters

DBSCAN has only two main parameters.

```text
DBSCAN

│

├── eps

└── min_samples
```

Understanding these parameters is the key to understanding the algorithm.

---

# Parameter 1 — eps

The parameter

```text
eps
```

(pronounced "epsilon")

defines the

**neighborhood radius**.

Imagine drawing a circle around every material.

```text
          ●

      *********

    ***       ***

   **     ●     **

    ***       ***

      *********
```

The radius of this circle is

```text
eps
```

Any material inside the circle is considered a neighbor.

---

# Small eps

If

```text
eps
```

is very small,

each point has very few neighbors.

```text
●     ●      ●

     ●

          ●
```

Most materials appear isolated.

The algorithm may identify

many small clusters

or

only noise.

---

# Large eps

If

```text
eps
```

is very large,

many neighborhoods overlap.

```text
● ● ● ● ● ● ●

● ● ● ● ● ● ●
```

Almost every material becomes connected.

The algorithm may merge unrelated groups into one enormous cluster.

Therefore,

choosing an appropriate value of

```text
eps
```

is essential.

---

# Parameter 2 — min_samples

The second parameter is

```text
min_samples
```

This specifies

the minimum number of neighboring points required to form a dense region.

Suppose

```text
min_samples = 5
```

A point must have

at least

five neighboring points

within the

```text
eps
```

radius

to qualify as part of a dense region.

---

# Why Do We Need min_samples?

Consider two situations.

---

### Dense Region

```text
● ● ●

● ● ●

● ●
```

Many neighbors exist.

This should become a cluster.

---

### Sparse Region

```text
●




      ●




            ●
```

Very few neighbors exist.

This should not become a cluster.

The

```text
min_samples
```

parameter allows DBSCAN to distinguish between these situations.

---

# Three Types of Points

Every point in DBSCAN belongs to one of three categories.

```text
DBSCAN Points

│

├── Core Point

├── Border Point

└── Noise Point
```

Understanding these point types is the heart of the algorithm.

---

# Core Points

A **core point** has

at least

```text
min_samples
```

neighbors

inside its

```text
eps
```

radius.

Example

```text
● ● ●

● ★ ●

● ●
```

The star

has many nearby neighbors.

It becomes

a core point.

Core points form the backbone of every cluster.

---

# Border Points

A border point has

fewer than

```text
min_samples
```

neighbors,

but it lies close to a core point.

Example

```text
● ● ●

● ★ ●

      ○
```

The circle

belongs to the cluster,

even though it is not dense enough to become a core point.

---

# Noise Points

A noise point

has too few neighbors

and

is not connected to any core point.

```text
● ● ●

● ★ ●





                   ×
```

The isolated point

```text
×
```

is labeled

noise.

Unlike K-Means,

DBSCAN naturally detects outliers without requiring a separate anomaly detection algorithm.

---

# How DBSCAN Builds Clusters

The clustering process is straightforward.

```text
Choose a Point

↓

Find Its Neighbors

↓

Is It a Core Point?

↓

Yes

↓

Create New Cluster

↓

Expand Cluster

↓

Repeat
```

If the point is not dense enough,

the algorithm checks whether it belongs to another cluster.

If not,

it becomes

noise.

---

# Cluster Expansion

Suppose a core point is discovered.

```text
● ● ●

● ★ ●

● ●
```

Every neighboring point is examined.

If a neighbor is also a core point,

its neighbors are added as well.

```text
● ● ● ●

● ★ ★ ●

● ● ● ●
```

This process continues until

the entire dense region has been explored.

---

# A Simple Example

Consider the following dataset.

```text
PC₂

^

|

|      ● ● ●

|     ● ● ●

|

|

|                  ▲ ▲ ▲

|                  ▲ ▲

|

|

|                                  ×

+-------------------------------------------------->

                     PC₁
```

DBSCAN identifies

```text
Cluster 1

↓

●
```

```text
Cluster 2

↓

▲
```

```text
Noise

↓

×
```

Notice that

the algorithm automatically detects

both clusters

without specifying

```text
K = 2
```

---

# DBSCAN Does Not Require K

One of the greatest advantages of DBSCAN is that

the number of clusters is **not** provided by the user.

Compare the algorithms.

| Algorithm | Need Number of Clusters? |
|------------|--------------------------|
| K-Means | Yes |
| Hierarchical | Optional (chosen by cutting the dendrogram) |
| DBSCAN | No |

Instead,

the data determine

how many clusters exist.

---

# Detecting Arbitrary Shapes

Perhaps the most important strength of DBSCAN is that

clusters do **not** have to be spherical.

For example,

consider a curved distribution.

```text
●

 ●

  ●

    ●

      ●

        ●

          ●
```

K-Means would likely divide this structure into several clusters.

DBSCAN,

however,

can identify it as

one continuous cluster

because the points remain densely connected.

---

# Advantages of DBSCAN

DBSCAN offers several important benefits.

- No need to specify the number of clusters.
- Naturally detects outliers.
- Can identify irregularly shaped clusters.
- Works well for exploratory analysis.
- Effective for noisy datasets.
- Handles arbitrary cluster geometries.

These characteristics make DBSCAN one of the most powerful density-based clustering algorithms.

---

# Limitations of DBSCAN

Despite its strengths,

DBSCAN is not universally applicable.

Its performance depends heavily on choosing appropriate values of

```text
eps
```

and

```text
min_samples.
```

Another limitation is that

DBSCAN assumes clusters have approximately similar densities.

Suppose one cluster is extremely dense,

while another is much more diffuse.

```text
Dense Cluster

● ● ● ●

● ● ●




Sparse Cluster

▲     ▲

   ▲

      ▲
```

A single value of

```text
eps
```

may not work well for both regions.

Finally,

DBSCAN becomes less effective in very high-dimensional spaces because distances between points become less informative.

For this reason,

materials scientists often apply

**PCA**

before using DBSCAN,

reducing hundreds of descriptors to a smaller number of principal components.

---

# DBSCAN in Materials Informatics

DBSCAN is particularly useful in situations where the dataset contains

- unusual compounds,
- computational failures,
- defect structures,
- metastable materials,
- or previously unknown material families.

A typical workflow is

```text
Crystal Structures

        │

        ▼

Descriptor Generation

        │

        ▼

Standardization

        │

        ▼

PCA

        │

        ▼

DBSCAN

        │

        ├────────► Cluster 1

        ├────────► Cluster 2

        ├────────► Cluster 3

        └────────► Noise Materials
```

The materials labeled as **noise** are often scientifically interesting. They may represent rare chemistries, unique crystal structures, calculation errors, or potential candidates for further investigation.

---

# From Theory to Implementation

We now understand the fundamental concepts behind DBSCAN.

- Clusters are defined by density rather than centroids.
- Every point is classified as a core point, border point, or noise point.
- The algorithm automatically determines the number of clusters.
- Outliers are detected naturally during clustering.

In the next section, we will implement **DBSCAN using scikit-learn**, learn how to choose suitable values for `eps` and `min_samples`, visualize density-based clusters, and compare the results directly with those produced by K-Means and Hierarchical Clustering.

## 11.19 Implementing DBSCAN with Scikit-learn

In the previous section, we learned the theory behind **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**.

Unlike K-Means,

DBSCAN

- does not require the number of clusters,
- can identify clusters with arbitrary shapes,
- naturally detects outliers,
- and groups materials based on local density.

Fortunately, implementing DBSCAN in Python is just as straightforward as implementing K-Means.

The **scikit-learn** library provides a highly optimized implementation that requires only a few lines of code.

In this section, we will learn how to apply DBSCAN to materials datasets, visualize the resulting clusters, and interpret the output.

---

# Typical Workflow

As with the previous clustering algorithms, DBSCAN is usually applied after feature preprocessing.

A typical materials informatics workflow is

```text
Crystal Structures

        │

        ▼

Descriptor Generation

        │

        ▼

Feature Matrix

        │

        ▼

Standardization

        │

        ▼

PCA

        │

        ▼

First 10–20 Principal Components

        │

        ▼

DBSCAN

        │

        ▼

Clusters + Noise Points
```

Applying PCA before DBSCAN often improves clustering quality because it removes redundant information and reduces noise in high-dimensional descriptor space.

---

# Importing the Required Libraries

The DBSCAN implementation is available in the

```python
sklearn.cluster
```

module.

```python
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

Assume that

```python
X_pca
```

contains the PCA-transformed descriptor matrix.

---

# Creating the DBSCAN Model

A DBSCAN model is created by specifying

- the neighborhood radius,
- and the minimum number of neighboring points.

Example

```python
dbscan = DBSCAN(
    eps=0.6,
    min_samples=5
)
```

Here,

```python
eps
```

defines the search radius,

while

```python
min_samples
```

defines the minimum number of nearby points required to create a dense region.

---

# Fitting the Model

Unlike supervised learning,

DBSCAN requires only the feature matrix.

```python
labels = dbscan.fit_predict(X_pca)
```

This single command

- fits the model,
- identifies dense regions,
- expands clusters,
- and labels every material.

The resulting labels are stored in

```python
labels
```

---

# Understanding the Labels

Suppose the output is

```python
array([
0,
0,
1,
1,
2,
2,
-1,
-1
])
```

Unlike K-Means,

DBSCAN uses

```text
-1
```

to represent

**noise points**.

The labels therefore have the following meaning.

| Label | Interpretation |
|-------:|----------------|
| 0 | Cluster 1 |
| 1 | Cluster 2 |
| 2 | Cluster 3 |
| -1 | Noise (Outlier) |

Every negative label corresponds to a point that does not belong to any cluster.

---

# Adding Labels to the DataFrame

As before,

cluster assignments can be stored directly in the dataset.

```python
df["Cluster"] = labels
```

The dataframe might now look like

| Formula | Band Gap | Density | Cluster |
|----------|----------:|---------:|---------:|
| Si | ... | ... | 0 |
| Fe₂O₃ | ... | ... | 1 |
| LiFePO₄ | ... | ... | 2 |
| Unknown Compound | ... | ... | -1 |

Notice that the final material has been identified as an outlier.

---

# Counting the Clusters

Unlike K-Means,

the cluster numbers are not specified beforehand.

Therefore,

we often determine how many clusters were discovered.

```python
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(n_clusters)
```

Example output

```text
3
```

The subtraction removes the noise label

```text
-1
```

from the cluster count.

---

# Counting Noise Points

Similarly,

the number of detected outliers can be computed.

```python
n_noise = np.sum(labels == -1)

print(n_noise)
```

Example output

```text
42
```

This means that

42 materials

were classified as noise.

---

# Visualizing the Clusters

Visualization is straightforward after PCA.

```python
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap="tab10"
)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("DBSCAN Clustering")

plt.show()
```

A typical result may resemble

```text
PC₂

^

|

|      ● ● ●

|    ● ● ●

|

|

|                     ▲ ▲ ▲

|                  ▲ ▲ ▲

|

|

|                                ×

+-------------------------------------------------->

                     PC₁
```

where

```text
● Cluster 0

▲ Cluster 1

× Noise
```

---

# Coloring Noise Separately

In scientific figures,

noise points are often displayed in black.

One possible implementation is

```python
plt.figure(figsize=(8,6))

unique_labels = np.unique(labels)

for label in unique_labels:

    mask = labels == label

    if label == -1:

        plt.scatter(
            X_pca[mask,0],
            X_pca[mask,1],
            color="black",
            label="Noise"
        )

    else:

        plt.scatter(
            X_pca[mask,0],
            X_pca[mask,1],
            label=f"Cluster {label}"
        )

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.legend()

plt.show()
```

The resulting visualization clearly separates

clusters

from

outliers.

---

# Choosing eps

Selecting

```text
eps
```

is the most important decision when using DBSCAN.

Consider three possibilities.

---

## eps Too Small

```text
●      ●

    ●

         ●
```

Almost every point appears isolated.

Result

```text
Many Noise Points

Many Tiny Clusters
```

---

## eps Too Large

```text
● ● ● ● ● ● ●

● ● ● ● ● ● ●
```

Most points become connected.

Result

```text
One Giant Cluster
```

---

## Appropriate eps

```text
● ● ●

● ●




▲ ▲ ▲

▲ ▲
```

Dense groups remain separated.

This usually produces meaningful clusters.

---

# Choosing min_samples

The second important parameter is

```text
min_samples
```

Small values

```text
2

or

3
```

allow even tiny groups to become clusters.

Large values

```text
10

or

20
```

require much denser neighborhoods.

General guidelines are

- Small datasets → smaller values.
- Large datasets → larger values.
- Higher-dimensional data often requires larger values.

However,

there is no universally optimal choice.

Parameter tuning depends on the dataset.

---

# Example Workflow

Suppose we have

8,000 materials

represented by

450 descriptors.

The complete workflow might be

```text
Crystal Structures

        │

        ▼

matminer

        │

        ▼

450 Descriptors

        │

        ▼

StandardScaler

        │

        ▼

PCA

(15 Components)

        │

        ▼

DBSCAN

(eps = 0.7,
 min_samples = 8)

        │

        ├────────► Cluster 0

        ├────────► Cluster 1

        ├────────► Cluster 2

        ├────────► Cluster 3

        └────────► Noise
```

Each cluster can then be analyzed independently to determine whether it corresponds to

- a crystal family,
- a compositional trend,
- or a unique bonding environment.

---

# Comparing DBSCAN with K-Means

The differences between the two algorithms are substantial.

| Feature | K-Means | DBSCAN |
|---------|----------|---------|
| Number of clusters required | Yes | No |
| Finds arbitrary shapes | No | Yes |
| Detects outliers | No | Yes |
| Assumes spherical clusters | Yes | No |
| Uses centroids | Yes | No |
| Based on density | No | Yes |

Neither algorithm is universally better.

Each is designed for different types of data.

---

# When Should You Use DBSCAN?

DBSCAN is particularly useful when

- the number of material families is unknown,
- unusual compounds are expected,
- outlier detection is important,
- clusters have irregular shapes,
- or exploratory analysis is the primary goal.

Examples include

- searching for novel materials,
- identifying failed DFT calculations,
- detecting unusual crystal structures,
- and exploring large descriptor spaces without prior assumptions.

---

# Limitations in High Dimensions

Although DBSCAN is powerful,

its performance decreases as dimensionality increases.

Suppose we attempt to cluster

```text
600 Descriptors
```

directly.

In high-dimensional space,

most points become nearly equally distant from one another.

This phenomenon,

known as the **curse of dimensionality**,

makes density estimation much more difficult.

For this reason,

DBSCAN is usually applied after

- feature selection,
- feature engineering,
- or PCA.

Reducing the data to

10–30 principal components

often leads to much better clustering performance.

---

# From Individual Algorithms to Practical Materials Analysis

At this point, we have studied three major clustering algorithms.

- **K-Means** groups materials around centroids.
- **Hierarchical Clustering** builds a tree of material relationships.
- **DBSCAN** discovers dense regions and identifies outliers.

Each algorithm approaches clustering from a different perspective, and each excels under different conditions.

In the next section, we will compare these three algorithms directly, examining their strengths, weaknesses, computational characteristics, and typical applications in materials informatics. This comparison will help establish practical guidelines for selecting the most appropriate clustering technique for a given materials science problem.

## 11.20 Comparing Clustering Algorithms for Materials Informatics

Throughout this chapter, we have studied three of the most important clustering algorithms in machine learning.

- K-Means
- Hierarchical Clustering
- DBSCAN

Although all three algorithms belong to the category of **unsupervised learning**, they solve the clustering problem using completely different strategies.

Understanding these differences is essential because **there is no universally best clustering algorithm**.

The most appropriate choice depends on

- the structure of the dataset,
- the scientific objective,
- the dimensionality of the descriptors,
- the presence of outliers,
- and the computational resources available.

In this section, we compare these algorithms from both a machine learning and materials informatics perspective.

---

# Three Different Ways of Thinking About Clusters

The three algorithms answer the question

> **"What is a cluster?"**

in very different ways.

### K-Means

A cluster consists of points that are closest to a common centroid.

```text
        ★

    ● ● ●

  ● ● ● ●

   ● ● ●
```

The centroid defines the cluster.

---

### Hierarchical Clustering

A cluster is built by repeatedly merging similar groups.

```text
Materials

↓

Small Groups

↓

Larger Groups

↓

Hierarchy
```

The hierarchy defines the clusters.

---

### DBSCAN

A cluster is simply a dense region of points.

```text
● ● ●

● ● ●

● ●
```

Density defines the cluster.

---

Although these ideas seem similar,

they often produce very different clustering results.

---

# Comparison of Basic Characteristics

The table below summarizes the major characteristics of each algorithm.

| Property | K-Means | Hierarchical | DBSCAN |
|-----------|----------|--------------|---------|
| Supervised? | No | No | No |
| Requires K? | Yes | No (tree can be cut later) | No |
| Uses Centroids | Yes | No | No |
| Density Based | No | No | Yes |
| Detects Outliers | No | No | Yes |
| Produces Dendrogram | No | Yes | No |
| Handles Arbitrary Shapes | Poorly | Moderately | Very Well |

Even this simple comparison reveals that the algorithms are designed for different types of problems.

---

# Cluster Shape

Perhaps the biggest difference lies in the shapes of clusters that each algorithm can discover.

---

## K-Means

K-Means performs best when clusters are

- compact,
- spherical,
- and approximately equal in size.

Example

```text
● ● ●

● ● ●




▲ ▲ ▲

▲ ▲ ▲
```

This is the ideal situation for K-Means.

---

## Hierarchical Clustering

Hierarchical clustering is somewhat more flexible.

Depending on the linkage method,

it can identify

- compact clusters,
- nested groups,
- or moderately irregular structures.

---

## DBSCAN

DBSCAN can identify clusters with almost any shape.

```text
●

 ●

  ●

    ●

      ●

        ●
```

or

```text
● ● ●

●     ●

●     ●

● ● ●
```

As long as the points remain densely connected,

DBSCAN recognizes them as one cluster.

---

# Handling Outliers

Outliers are common in materials datasets.

Examples include

- failed DFT calculations,
- unstable structures,
- incorrect descriptors,
- rare compounds,
- measurement errors.

How do the algorithms handle these points?

---

## K-Means

Every point must belong to a cluster.

Even obvious outliers are forced into the nearest cluster.

```text
Cluster

● ● ●

● ●





×

↓

Assigned Anyway
```

---

## Hierarchical Clustering

Outliers eventually merge into the hierarchy.

Although they may appear as isolated branches,

they are still assigned to the tree.

---

## DBSCAN

DBSCAN naturally identifies outliers.

```text
● ● ●

● ●





×

↓

Noise
```

No additional outlier detection algorithm is needed.

---

# Computational Cost

The algorithms also differ in computational efficiency.

---

## K-Means

Very fast.

Efficient for

- tens of thousands,
- hundreds of thousands,
- or even millions of samples.

---

## Hierarchical Clustering

Much slower.

The repeated merging process becomes expensive for large datasets.

Consequently,

hierarchical clustering is often limited to

small

or

medium-sized datasets.

---

## DBSCAN

Intermediate.

Performance depends strongly on

- dataset size,
- neighborhood search,
- and dimensionality.

For moderate datasets,

it is generally practical.

---

# Ease of Interpretation

Different algorithms provide different kinds of scientific insight.

---

## K-Means

Produces

simple,

easy-to-understand

clusters.

```text
Cluster 1

Cluster 2

Cluster 3
```

---

## Hierarchical Clustering

Provides much richer information.

Instead of only clusters,

it reveals

relationships

between clusters.

```text
Materials

↓

Families

↓

Subfamilies

↓

Individual Compounds
```

This makes it particularly useful for exploratory studies.

---

## DBSCAN

Highlights

- dense regions,
- isolated compounds,
- and unusual materials.

This is valuable for discovering

novel candidates

or

rare material families.

---

# Performance on High-Dimensional Data

Materials informatics datasets often contain

```text
200–1000 Descriptors
```

How do the algorithms perform?

---

## K-Means

Generally performs well

after

feature scaling

and

PCA.

---

## Hierarchical Clustering

Can also work well,

but computation becomes increasingly expensive.

---

## DBSCAN

More sensitive to dimensionality.

Distance-based density estimation becomes difficult as the number of dimensions increases.

For this reason,

DBSCAN is almost always applied

after PCA.

---

# Typical Materials Informatics Applications

Different scientific problems naturally favor different algorithms.

---

## K-Means

Best suited for

- grouping similar compounds,
- organizing large databases,
- rapid exploratory analysis,
- preprocessing before supervised learning.

Example

```text
Materials Project

↓

500 Descriptors

↓

PCA

↓

K-Means

↓

Eight Material Families
```

---

## Hierarchical Clustering

Useful for

- studying chemical relationships,
- identifying structural hierarchies,
- comparing crystal families,
- understanding similarities between compounds.

Example

```text
Oxides

↓

Perovskites

↓

Cubic Perovskites

↓

Specific Materials
```

---

## DBSCAN

Ideal for

- discovering unusual compounds,
- identifying outliers,
- detecting metastable materials,
- exploring irregular descriptor distributions.

Example

```text
Descriptor Space

↓

DBSCAN

↓

Dense Material Families

+

Rare Compounds
```

---

# Strengths and Weaknesses

The major strengths of each algorithm are summarized below.

| Algorithm | Major Strength |
|------------|----------------|
| K-Means | Fast and scalable |
| Hierarchical | Reveals hierarchical relationships |
| DBSCAN | Finds arbitrary clusters and detects outliers |

Similarly,

their major limitations are

| Algorithm | Main Limitation |
|------------|----------------|
| K-Means | Requires K and assumes spherical clusters |
| Hierarchical | Computationally expensive |
| DBSCAN | Sensitive to parameter selection and varying densities |

---

# Which Algorithm Should You Choose?

There is no single answer.

Instead,

the choice depends on the research question.

### If the goal is

rapid clustering of a large materials database

↓

Choose

**K-Means**

---

### If the goal is

understanding relationships between material families

↓

Choose

**Hierarchical Clustering**

---

### If the goal is

discovering novel compounds

or

identifying unusual materials

↓

Choose

**DBSCAN**

---

In many research projects,

multiple algorithms are applied to the same dataset.

Comparing their results often provides a deeper understanding of the underlying materials space.

---

# Combining PCA with Clustering

One of the most common workflows in modern materials informatics combines PCA with one of the clustering algorithms discussed in this chapter.

```text
Crystal Structures

        │

        ▼

pymatgen

        │

        ▼

matminer

        │

        ▼

Hundreds of Descriptors

        │

        ▼

StandardScaler

        │

        ▼

Principal Component Analysis

        │

        ▼

Reduced Feature Space

        │

        ├────────► K-Means

        │

        ├────────► Hierarchical Clustering

        │

        └────────► DBSCAN

        │

        ▼

Visualization

Material Families

Chemical Trends

Outlier Detection
```

This workflow has become standard practice in many materials informatics studies because it combines

- dimensionality reduction,
- visualization,
- and unsupervised learning

into a single analytical pipeline.

---

# Practical Recommendations

The following guidelines summarize common best practices.

| Situation | Recommended Algorithm |
|------------|----------------------|
| Large descriptor datasets | K-Means |
| Hierarchical relationships | Hierarchical Clustering |
| Unknown number of clusters | DBSCAN |
| Outlier detection | DBSCAN |
| Publication-quality dendrograms | Hierarchical Clustering |
| Fast exploratory analysis | K-Means |
| Novel materials discovery | DBSCAN |
| PCA-reduced descriptor space | All three |

These recommendations should be viewed as starting points rather than strict rules.

Real-world materials datasets often benefit from experimenting with multiple clustering approaches.

---

# Looking Beyond Traditional Clustering

The clustering algorithms introduced in this chapter form the foundation of unsupervised learning.

However,

modern materials informatics increasingly employs more advanced techniques for exploring high-dimensional descriptor spaces.

Examples include

- Gaussian Mixture Models (GMM),
- Spectral Clustering,
- t-SNE (t-distributed Stochastic Neighbor Embedding),
- UMAP (Uniform Manifold Approximation and Projection),
- Self-Organizing Maps (SOMs),
- and graph-based clustering methods.

Many of these methods combine dimensionality reduction with clustering to reveal complex relationships that are difficult to detect using traditional algorithms.

---

# Chapter Summary

In this chapter, we explored two major topics in unsupervised machine learning: **dimensionality reduction** and **clustering**.

We began by studying **Principal Component Analysis (PCA)** and learned how high-dimensional descriptor spaces can be transformed into a small number of principal components while preserving most of the information contained in the original data. We examined the concepts of variance, covariance, eigenvectors, eigenvalues, principal components, explained variance, and projection, and saw how PCA enables visualization of materials described by hundreds of descriptors generated using `pymatgen` and `matminer`.

We then introduced clustering as a method for discovering hidden patterns without labeled data. Three of the most widely used clustering algorithms were presented.

- **K-Means**, which partitions materials around centroids and is well suited for large, compact datasets.
- **Hierarchical Clustering**, which builds a dendrogram that reveals nested relationships between material families.
- **DBSCAN**, which identifies dense regions in descriptor space while naturally detecting outliers and irregularly shaped clusters.

We also learned how to evaluate clustering quality using the **Elbow Method** and the **Silhouette Score**, implemented all three clustering algorithms using **scikit-learn**, and compared their strengths, limitations, and practical applications in materials informatics.

Together, PCA and clustering provide powerful tools for exploring large materials databases, organizing chemical space, identifying material families, discovering unusual compounds, and generating new scientific hypotheses before supervised machine learning models are developed.

With this chapter, we complete the core concepts of **unsupervised learning** that are most commonly used in computational materials science. The next chapters will build upon these ideas as we move toward advanced topics such as model interpretation, deep learning, graph neural networks, and modern AI methods for materials discovery.

# Chapter 11A — Practical Implementation of Dimensionality Reduction and Unsupervised Learning

## From Theory to Research-Ready Python Workflows

---

# 11A.1 Introduction

In Chapter 11, we developed the theoretical foundations of **dimensionality reduction** and **unsupervised learning**. We learned why Principal Component Analysis (PCA) reduces high-dimensional descriptor spaces, how clustering algorithms identify hidden patterns, and how these techniques enable visualization and exploration of chemical and materials spaces.

However, understanding the theory alone is not enough.

A computational materials scientist must also know **how to implement these methods using Python and modern machine learning libraries**.

This supplementary chapter bridges the gap between theory and implementation.

Rather than introducing new algorithms, the purpose of this chapter is to demonstrate how the concepts developed in Chapter 11 are applied in real materials informatics workflows.

By the end of this chapter, you will be able to

- load a materials dataset,
- prepare descriptor matrices,
- standardize features correctly,
- perform Principal Component Analysis,
- visualize high-dimensional materials spaces,
- interpret principal components,
- perform clustering using multiple algorithms,
- evaluate clustering quality,
- and build complete unsupervised learning pipelines suitable for research.

Throughout this chapter, we will use Python together with

- NumPy
- pandas
- Matplotlib
- scikit-learn
- SciPy

Although the examples are intentionally simple, the workflows are essentially identical to those used in modern materials informatics research.

---

# 11A.2 Learning Objectives

After completing this chapter, you should be able to

- Prepare descriptor matrices for unsupervised learning.
- Properly standardize numerical features.
- Perform PCA using scikit-learn.
- Determine the appropriate number of principal components.
- Interpret explained variance ratios.
- Visualize materials in two- and three-dimensional PCA space.
- Apply K-Means clustering.
- Apply Hierarchical Clustering.
- Apply DBSCAN.
- Compare different clustering algorithms.
- Build complete unsupervised learning workflows for materials datasets.

---

# 11A.3 Software Requirements

Install the required Python packages before beginning.

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

If you are using Jupyter Notebook,

```bash
pip install notebook
```

or

```bash
pip install jupyterlab
```

---

# 11A.4 Complete Workflow

Throughout this chapter we will repeatedly construct the following workflow.

```text
Materials Dataset

↓

Load Dataset

↓

Inspect Data

↓

Select Descriptors

↓

Feature Scaling

↓

Principal Component Analysis

↓

Visualization

↓

Clustering

↓

Interpretation

↓

Scientific Conclusions
```

Although individual research projects differ,

this pipeline forms the basis of many published materials informatics studies.

---

# 11A.5 Organization of This Supplement

This supplement is organized as a sequence of progressively more advanced implementation exercises.

| Section | Topic |
|---------|------|
| 11A.6 | Loading Materials Data |
| 11A.7 | Preparing Descriptor Matrices |
| 11A.8 | Feature Scaling |
| 11A.9 | Principal Component Analysis |
| 11A.10 | Interpreting PCA Results |
| 11A.11 | Visualizing Materials Space |
| 11A.12 | K-Means Clustering |
| 11A.13 | Choosing the Number of Clusters |
| 11A.14 | Hierarchical Clustering |
| 11A.15 | DBSCAN |
| 11A.16 | Comparing Clustering Algorithms |
| 11A.17 | Complete Materials Informatics Workflow |
| 11A.18 | Exercises |

Each section introduces new implementation details while building toward a complete research-level workflow.

---

# 11A.6 Dataset Used Throughout This Chapter

For simplicity, we assume that a descriptor dataset has already been generated.

A typical dataset may contain

| Feature | Description |
|----------|-------------|
| density | Material density |
| volume | Unit-cell volume |
| mean_atomic_mass | Average atomic mass |
| mean_atomic_radius | Average atomic radius |
| mean_electronegativity | Average electronegativity |
| packing_fraction | Atomic packing fraction |
| band_gap | Target property (optional) |

In practice, these descriptors may have been generated using

- pymatgen
- matminer
- Quantum ESPRESSO outputs
- Materials Project data

Later chapters will demonstrate how such datasets are constructed automatically.

For this supplement, our focus is on applying machine learning techniques to an existing descriptor matrix.

---

# 11A.7 What Makes This Chapter Different?

Unlike Chapter 11, which focused on

- mathematical intuition,
- geometric interpretation,
- algorithmic understanding,
- and scientific motivation,

this chapter focuses on implementation.

Almost every section contains executable Python code together with explanations of

- what each line does,
- why it is necessary,
- common mistakes,
- and how the code relates to the underlying mathematics.

The goal is not simply to copy code.

The goal is to understand **why** every step appears in a modern materials informatics workflow.

---

# 11A.8 Recommended Workflow for Readers

To obtain the greatest benefit from this chapter,

follow each section in the following order.

```text
Read Theory

↓

Run Code

↓

Modify Code

↓

Observe Results

↓

Repeat
```

Do not simply read the source code.

Execute every program,

modify parameters,

change datasets,

experiment with different algorithms,

and observe how the outputs change.

This active learning process is essential for developing practical machine learning skills.

---

# 11A.9 Before You Begin

Before continuing, ensure that you are comfortable with

- Python variables
- lists
- dictionaries
- functions
- NumPy arrays
- pandas DataFrames

You should also have completed the earlier chapters covering

- feature engineering,
- preprocessing,
- model evaluation,
- and validation,

since many of those concepts will be used throughout this supplement.

---

# 11A.10 Summary

This supplementary chapter serves as the practical companion to Chapter 11.

Rather than introducing new theory, it demonstrates how dimensionality reduction and clustering algorithms are implemented using modern Python libraries.

By working through the code examples in this chapter, you will gain the practical experience needed to apply PCA and clustering techniques to real materials datasets and prepare for the deep learning workflows introduced in the following chapters.

In the next section, we begin by loading a materials dataset, inspecting its structure, and preparing it for machine learning analysis.

# 11A.6 Loading and Inspecting a Materials Dataset

Before applying PCA or clustering algorithms, the first step in any machine learning workflow is understanding the dataset.

Machine learning algorithms do not understand materials, atoms, or crystal structures directly.

They only understand numerical matrices.

Therefore, the first challenge in materials informatics is converting scientific information into a clean numerical representation.

The general transformation is

```text
Crystal Structure

↓

pymatgen

↓

matminer descriptors

↓

Numerical Feature Matrix

↓

Machine Learning Algorithm
```

By the time PCA or clustering is applied, the dataset should already be represented as a table of numbers.

---

# 11A.6.1 Importing Required Libraries

We begin by importing the essential Python libraries.

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
```

These libraries provide the fundamental tools for handling scientific datasets.

## pandas

`pandas` is used for

- loading datasets,
- organizing tabular data,
- filtering columns,
- handling missing values.

The main object in pandas is the

```text
DataFrame
```

which represents a table similar to a spreadsheet.

---

## NumPy

`numpy` provides efficient numerical operations.

Machine learning libraries internally represent data as NumPy arrays.

For example,

```python
[1.5, 2.3, 4.8]
```

becomes

```python
array([1.5, 2.3, 4.8])
```

which allows fast mathematical operations.

---

## Matplotlib

`matplotlib` is used for visualization.

Throughout this chapter we will use it to create

- PCA plots,
- cluster visualizations,
- variance plots,
- dendrograms.

---

# 11A.6.2 Loading a CSV Materials Dataset

Assume we have a descriptor file:

```text
materials_descriptors.csv
```

The file may have been generated from

- Materials Project structures,
- pymatgen calculations,
- matminer featurizers,
- DFT calculations.

We load it using pandas.

```python
df = pd.read_csv(
    "materials_descriptors.csv"
)
```

The variable

```python
df
```

now contains the complete dataset.

---

# 11A.6.3 Viewing the First Few Samples

The first step after loading any dataset is inspection.

```python
df.head()
```

Example output:

```text
        density   volume   atomic_mass   electronegativity

0       5.43      160.2       28.09            1.90

1       6.21      145.8       32.06            2.05

2       7.12      132.4       55.85            1.83
```

Each row represents one material.

Each column represents one descriptor.

---

# 11A.6.4 Understanding Dataset Dimensions

The shape of the dataset is extremely important.

```python
df.shape
```

Example output:

```text
(5000, 120)
```

This means:

```text
5000 materials

×

120 descriptors
```

In materials informatics terminology,

the dataset contains

```text
5000 samples

120-dimensional feature space
```

This high-dimensional feature space is exactly where PCA becomes useful.

---

# 11A.6.5 Checking Dataset Information

Use

```python
df.info()
```

Example output:

```text
<class 'pandas.core.frame.DataFrame'>

5000 entries

120 columns

float64 values
```

This provides information about

- number of samples,
- number of features,
- data types,
- missing values.

---

# 11A.6.6 Checking Statistical Properties

Machine learning workflows require understanding the numerical distribution of features.

Use

```python
df.describe()
```

Example output:

```text
               density       volume

count          5000       5000

mean            6.21      145.8

std             1.45       35.2

min             1.20       40.5

max            15.8      400.2
```

Important quantities include

## Mean

Average value.

Example:

```text
Average density of materials
```

---

## Standard deviation

Measures how spread out the values are.

Large standard deviation means large variation.

---

## Minimum and Maximum

Useful for identifying

- unrealistic values,
- unit errors,
- outliers.

---

# 11A.6.7 Checking Missing Values

Real materials datasets are rarely perfect.

Missing values may occur because of

- failed DFT calculations,
- incomplete experimental measurements,
- unavailable descriptors.

Check missing values using:

```python
df.isnull().sum()
```

Example:

```text
density                 0

volume                 12

electronegativity       3
```

This indicates

- density has no missing values,
- volume has 12 missing values,
- electronegativity has 3 missing values.

---

# 11A.6.8 Removing Missing Data

The simplest solution is removing incomplete samples.

```python
df_clean = df.dropna()
```

This removes rows containing missing values.

However,

this should be used carefully.

If the dataset is small,

removing samples may significantly reduce the available data.

---

# 11A.6.9 Filling Missing Values

Another approach is replacing missing values.

For example,

replace missing values with the column average.

```python
df_filled = df.fillna(
    df.mean()
)
```

Mathematically,

a missing descriptor value becomes

```text
Missing Value

↓

Average Descriptor Value
```

This approach preserves the number of samples.

However,

it introduces assumptions about the missing data.

---

# 11A.6.10 Materials Science Considerations

Handling missing values requires scientific judgment.

For example,

suppose a descriptor represents

```text
Band gap
```

A missing value may indicate

- a failed calculation,
- metallic behavior,
- unavailable experimental data.

Blindly filling the value may introduce incorrect physical information.

Therefore,

data cleaning is not only a programming task.

It requires understanding the materials science behind the data.

---

# 11A.6.11 Separating Features and Targets

In supervised learning,

we usually separate

- input descriptors,
- target properties.

Example:

Predicting band gap.

```text
Inputs:

density

atomic radius

electronegativity

volume


Target:

band gap
```

In Python:

```python
X = df.drop(
    "band_gap",
    axis=1
)

y = df["band_gap"]
```

Here,

`X` contains the descriptors.

`y` contains the property we want to predict.

---

# 11A.6.12 For Unsupervised Learning

PCA and clustering do not require target labels.

The algorithms only analyze the structure of the feature space.

Therefore,

we usually use only

```python
X
```

Example:

```python
X = df
```

or

```python
X = df[feature_columns]
```

The goal is not prediction.

The goal is discovering hidden patterns.

---

# 11A.6.13 Selecting Descriptor Columns

A materials dataset may contain non-feature columns.

Examples:

```text
material_id

formula

structure_name

band_gap

formation_energy
```

These should usually not be included in PCA.

Select only numerical descriptors.

Example:

```python
features = [
    "density",
    "volume",
    "atomic_mass",
    "electronegativity",
    "packing_fraction"
]


X = df[features]
```

Now,

`X` contains only the variables used for analysis.

---

# 11A.6.14 Checking the Final Feature Matrix

Before applying PCA,

always inspect the final matrix.

```python
X.head()
```

and

```python
X.shape
```

Example:

```text
(5000, 50)
```

Meaning:

```text
5000 materials

50 descriptors
```

This is the actual input that will enter PCA.

---

# 11A.6.15 Converting pandas DataFrame to NumPy

Most scikit-learn algorithms accept both pandas DataFrames and NumPy arrays.

Conversion:

```python
X_array = X.values
```

Example:

Before:

```text
pandas DataFrame
```

After:

```text
numpy array
```

However,

keeping the DataFrame format is often preferable because column names are useful for interpretation.

---

# 11A.6.16 Complete Dataset Preparation Example

A complete basic workflow:

```python
import pandas as pd


# Load dataset

df = pd.read_csv(
    "materials_descriptors.csv"
)


# Remove missing values

df = df.dropna()


# Select descriptors

features = [
    "density",
    "volume",
    "atomic_mass",
    "electronegativity",
    "packing_fraction"
]


X = df[features]


# Inspect dataset

print(X.head())

print(X.shape)
```

Output:

```text
        density    volume    atomic_mass

0       5.43       160.2       28.09

1       6.21       145.8       32.06


(5000,5)
```

---

# 11A.6.17 Connection With Materials Informatics

In a real research workflow,

the previous code represents the transition:

```text
Crystal Data

↓

pymatgen

↓

matminer

↓

Descriptor DataFrame

↓

X matrix

↓

PCA / Clustering
```

The quality of PCA and clustering depends strongly on the quality of this input matrix.

Poor descriptors,

incorrect units,

missing values,

or meaningless features

will produce meaningless results.

---

# 11A.6.18 Summary

Before applying dimensionality reduction or clustering algorithms, the materials dataset must be carefully prepared.

The essential steps are:

```text
Load Dataset

↓

Inspect Structure

↓

Check Missing Values

↓

Select Descriptors

↓

Create Feature Matrix
```

At this stage, we have transformed a materials dataset into a numerical representation suitable for machine learning.

In the next section, we will discuss one of the most important preprocessing steps in PCA and clustering:

**feature scaling and normalization.**

# 11A.7 Feature Scaling and Standardization Before PCA

After preparing the materials descriptor matrix, the next essential step is **feature scaling**.

Many beginners skip this step because the dataset already appears numerical.

However, for PCA and clustering algorithms, scaling is not optional.

It is one of the most important preprocessing operations.

A materials dataset may contain descriptors with completely different numerical ranges.

For example:

| Descriptor | Typical Range |
|------------|---------------|
| Density | 1 – 20 g/cm³ |
| Atomic Mass | 1 – 240 amu |
| Volume | 50 – 1000 Å³ |
| Electronegativity | 0.5 – 4 |
| Band Gap | 0 – 10 eV |

Although all these values describe materials,

their numerical scales are very different.

Machine learning algorithms do not understand physical importance.

They only see numbers.

Therefore,

a feature with a larger numerical magnitude can dominate the calculation even if it is not scientifically more important.

---

# 11A.7.1 Why Scaling Is Necessary for PCA

Principal Component Analysis is based on variance.

Recall from Chapter 11:

```text
PCA

↓

Find directions of maximum variance
```

The problem is that variance depends strongly on numerical scale.

Consider two descriptors:

```text
Feature A:

Atomic Mass

Range:

1 - 240
```

and

```text
Feature B:

Electronegativity

Range:

0.5 - 4
```

Even if electronegativity contains important chemical information,

atomic mass may dominate PCA simply because its numbers are larger.

The PCA algorithm may conclude:

```text
Large Numerical Variation

↓

Important Feature
```

even when this is not physically correct.

Therefore,

we scale all descriptors to comparable ranges before PCA.

---

# 11A.7.2 Example Without Scaling

Consider a simple dataset.

```python
import pandas as pd


data = {
    "atomic_mass": [20, 50, 100, 200],
    "electronegativity": [1.5, 2.0, 2.5, 3.0]
}


df = pd.DataFrame(data)

print(df)
```

Output:

```text
   atomic_mass   electronegativity

0       20              1.5

1       50              2.0

2      100              2.5

3      200              3.0
```

The numerical scale of atomic mass is much larger.

If PCA is applied directly,

the atomic mass feature will dominate the principal components.

---

# 11A.7.3 Standardization

The most common scaling method in machine learning is **standardization**.

Standardization transforms each feature so that it has

```text
Mean = 0

Standard Deviation = 1
```

The transformation is:

```text
Standardized Value

=

(Value - Mean)

÷

Standard Deviation
```

Conceptually:

```text
Original Feature

↓

Subtract Average

↓

Divide by Spread

↓

Scaled Feature
```

After standardization,

all features are placed on a comparable scale.

---

# 11A.7.4 Using StandardScaler in Scikit-Learn

Scikit-learn provides a convenient implementation.

Import:

```python
from sklearn.preprocessing import StandardScaler
```

Create the scaler:

```python
scaler = StandardScaler()
```

Apply transformation:

```python
X_scaled = scaler.fit_transform(X)
```

Now,

`X_scaled` contains standardized descriptor values.

---

# 11A.7.5 Understanding fit()

The first operation is

```python
scaler.fit(X)
```

The scaler calculates the statistics of each feature.

For every descriptor,

it calculates:

```text
Mean

and

Standard Deviation
```

Example:

For density:

```text
Mean density

=

6.5 g/cm³
```

```text
Standard deviation

=

2.1
```

These values are stored internally.

---

# 11A.7.6 Understanding transform()

The second operation is

```python
scaler.transform(X)
```

The previously calculated mean and standard deviation are used to transform the data.

Example:

Original value:

```text
Density = 8.0
```

Mean:

```text
6.5
```

Standard deviation:

```text
2.1
```

Transformation:

```text
(8.0 - 6.5)

÷

2.1
```

Result:

```text
0.71
```

The descriptor is now standardized.

---

# 11A.7.7 fit_transform()

Most workflows combine both steps.

Instead of:

```python
scaler.fit(X)

X_scaled = scaler.transform(X)
```

we write:

```python
X_scaled = scaler.fit_transform(X)
```

This performs:

```text
Calculate Mean and Standard Deviation

↓

Apply Scaling

↓

Return Transformed Data
```

---

# 11A.7.8 Complete Scaling Example

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler


data = {
    "density": [5.4, 6.2, 7.1, 8.0],
    "atomic_mass": [28, 55, 85, 120],
    "electronegativity": [1.8, 2.0, 2.4, 3.1]
}


df = pd.DataFrame(data)


scaler = StandardScaler()


X_scaled = scaler.fit_transform(df)


print(X_scaled)
```

Output:

```text
[[-1.2 -1.1 -1.0]

 [-0.4 -0.5 -0.6]

 [ 0.4  0.2  0.2]

 [ 1.2  1.4  1.4]]
```

Notice that the original scales are gone.

All features now have comparable distributions.

---

# 11A.7.9 Checking the Result

Convert back to a DataFrame:

```python
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=df.columns
)


print(X_scaled_df.describe())
```

Expected output:

```text
                 density   atomic_mass

mean               0.0        0.0

std                1.0        1.0
```

This confirms that standardization worked correctly.

---

# 11A.7.10 Scaling Before Clustering

Scaling is not only important for PCA.

It is also critical for clustering algorithms.

Consider K-Means.

K-Means groups points based on distance.

The distance formula is:

```text
Distance

=

Difference between coordinates
```

If one feature has a much larger scale,

it dominates the distance calculation.

Example:

Without scaling:

```text
Atomic Mass

↓

Dominates Distance
```

```text
Electronegativity

↓

Almost Ignored
```

After scaling:

```text
All Features

↓

Equal Contribution
```

---

# 11A.7.11 Scaling and Chemical Meaning

A common misunderstanding is:

> "If a descriptor is physically important, should we avoid scaling it?"

No.

Scaling does not remove physical meaning.

It only changes the numerical representation.

For example:

Before scaling:

```text
Density

↓

5.4 g/cm³
```

After scaling:

```text
Density

↓

-0.35
```

The physical property has not changed.

Only the mathematical representation used by the algorithm has changed.

---

# 11A.7.12 When Should We NOT Scale?

Although scaling is usually required for PCA and clustering,

it is not always necessary.

For example:

Tree-based models such as

- Random Forest
- Decision Trees
- XGBoost

do not require feature scaling.

Why?

Because trees split data based on thresholds rather than distances.

However,

for methods based on

- variance,
- distance,
- gradients,

scaling is usually essential.

Examples:

| Algorithm | Scaling Needed |
|-----------|----------------|
| PCA | Yes |
| K-Means | Yes |
| DBSCAN | Yes |
| Neural Networks | Usually yes |
| Random Forest | No |
| XGBoost | Usually no |

---

# 11A.7.13 Materials Informatics Example

Suppose we generate 200 descriptors using matminer.

The dataset may contain:

```text
Density

Volume

Atomic Mass

Bond Length

Coordination Number

Electronegativity

Covalent Radius

Packing Fraction

...
```

The workflow becomes:

```text
matminer Descriptors

↓

Feature Matrix

↓

StandardScaler

↓

PCA

↓

Materials Space Visualization
```

Scaling is the bridge between scientific descriptors and machine learning algorithms.

---

# 11A.7.14 Complete Preprocessing Pipeline

A typical preprocessing workflow:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load data

df = pd.read_csv(
    "materials_descriptors.csv"
)


# Select descriptors

X = df[
    [
        "density",
        "volume",
        "atomic_mass",
        "electronegativity"
    ]
]


# Scale features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


print(X_scaled.shape)
```

Output:

```text
(5000, 4)
```

The resulting matrix is now ready for PCA.

---

# 11A.7.15 Common Mistakes

## Mistake 1: Applying PCA Before Scaling

Incorrect:

```python
pca.fit_transform(X)
```

when features have different units.

Correct:

```python
X_scaled = scaler.fit_transform(X)

X_pca = pca.fit_transform(X_scaled)
```

---

## Mistake 2: Scaling the Target Variable

For unsupervised learning,

there is no target.

For supervised learning,

usually scale only the input descriptors.

Example:

Correct:

```text
Descriptors

↓

Scaler

↓

Model
```

---

## Mistake 3: Scaling Categorical Variables

Do not standardize variables such as:

```text
Crystal System

Space Group

Material Class
```

These require different encoding methods.

---

# 11A.7.16 Summary

Feature scaling is a critical preprocessing step before PCA and clustering.

The main reasons are:

```text
Different Units

↓

Different Numerical Ranges

↓

Bias in Algorithms

↓

Incorrect Results
```

Standardization solves this problem by transforming every descriptor to a common scale:

```text
Mean = 0

Standard Deviation = 1
```

In practical materials informatics workflows:

```text
pymatgen / matminer descriptors

↓

Feature Matrix

↓

StandardScaler

↓

PCA / Clustering
```

After scaling, the dataset is finally ready for dimensionality reduction.

In the next section, we will implement **Principal Component Analysis (PCA) in Python using scikit-learn** and analyze how hundreds of materials descriptors can be compressed into a low-dimensional materials space.

# 11A.8 Principal Component Analysis (PCA) Implementation Using Python

After preparing and scaling the descriptor matrix, we are ready to perform Principal Component Analysis (PCA).

In Chapter 11, we studied the theoretical foundations of PCA:

- variance,
- principal directions,
- eigenvectors,
- eigenvalues,
- projection,
- dimensionality reduction.

Now we will implement these concepts using Python and scikit-learn.

The goal is to transform a high-dimensional materials descriptor space into a lower-dimensional representation that can be visualized and analyzed.

The complete workflow is:

```text
Materials Descriptors

↓

Feature Scaling

↓

PCA

↓

Principal Components

↓

Visualization

↓

Scientific Interpretation
```

---

# 11A.8.1 Why Use PCA in Materials Informatics?

Modern materials databases can contain hundreds or thousands of descriptors.

For example:

```text
Crystal Structure

↓

pymatgen

↓

matminer

↓

200+ descriptors
```

These descriptors may include:

- atomic fractions,
- average atomic properties,
- structural features,
- bonding descriptors,
- electronic descriptors.

Although these descriptors contain valuable information,

visualizing a 200-dimensional space is impossible.

Humans can only directly visualize

- one dimension,
- two dimensions,
- three dimensions.

PCA solves this problem by finding a smaller representation.

Conceptually:

```text
200-dimensional Materials Space

↓

PCA

↓

2-dimensional Visualization Space
```

---

# 11A.8.2 Importing PCA

PCA is available inside scikit-learn.

```python
from sklearn.decomposition import PCA
```

The PCA class provides all necessary mathematical operations:

- covariance calculation,
- eigenvalue decomposition,
- projection,
- transformation.

---

# 11A.8.3 Preparing the Input Matrix

Assume we already created and scaled our descriptor matrix.

Example:

```python
X_scaled
```

has the form:

```text
Material 1:

[0.25, -0.41, 1.20, ...]


Material 2:

[-0.13, 0.82, -0.55, ...]
```

Mathematically:

```text
X_scaled

=

Number of Materials

×

Number of Descriptors
```

Example:

```text
5000 materials

×

200 descriptors
```

This is the input for PCA.

---

# 11A.8.4 Creating a PCA Object

To perform PCA:

```python
pca = PCA(
    n_components=2
)
```

The parameter

```python
n_components
```

controls the number of principal components we want to keep.

Here:

```text
200 original features

↓

2 principal components
```

The result can be plotted on a 2D graph.

---

# 11A.8.5 Performing PCA Transformation

Now apply PCA:

```python
X_pca = pca.fit_transform(
    X_scaled
)
```

This single command performs two operations.

---

## Step 1: Fit

The algorithm learns the principal directions.

Mathematically:

```text
Feature Matrix

↓

Covariance Matrix

↓

Eigenvectors

↓

Principal Components
```

---

## Step 2: Transform

The original data are projected onto the new coordinate system.

Conceptually:

```text
Original Space

↓

Projection

↓

PCA Space
```

---

# 11A.8.6 Checking the PCA Output

Inspect the dimensions:

```python
X_pca.shape
```

Example:

```text
(5000, 2)
```

Meaning:

Before:

```text
5000 materials

×

200 descriptors
```

After PCA:

```text
5000 materials

×

2 principal components
```

The number of samples remains unchanged.

Only the number of features is reduced.

---

# 11A.8.7 Understanding Principal Components

The transformed variables are called

```text
Principal Components
```

They are new artificial coordinates.

For example:

Original descriptors:

```text
Density

Volume

Atomic Radius

Electronegativity

Bond Length

...
```

become:

```text
PC1

PC2
```

Important:

Principal components are not physical properties.

They are mathematical combinations of the original descriptors.

---

# 11A.8.8 Viewing Principal Component Values

Convert PCA results into a DataFrame:

```python
pca_df = pd.DataFrame(
    X_pca,
    columns=[
        "PC1",
        "PC2"
    ]
)

print(pca_df.head())
```

Example output:

```text
        PC1       PC2

0      1.52     -0.44

1      0.82      1.10

2     -2.05      0.34
```

Each material now has two new coordinates.

These coordinates define its position in the reduced materials space.

---

# 11A.8.9 Visualizing PCA Materials Space

Now we can create a two-dimensional materials map.

```python
import matplotlib.pyplot as plt


plt.figure(
    figsize=(8,6)
)


plt.scatter(
    X_pca[:,0],
    X_pca[:,1]
)


plt.xlabel(
    "Principal Component 1"
)


plt.ylabel(
    "Principal Component 2"
)


plt.title(
    "PCA Materials Space"
)


plt.show()
```

The result is a map where:

- each point represents one material,
- nearby points have similar descriptor patterns,
- distant points have different characteristics.

---

# 11A.8.10 Interpreting the PCA Plot

A PCA plot does not directly show:

```text
Material A

↓

Material B
```

Instead,

it shows similarity in descriptor space.

If two materials appear close together:

```text
Material A

        ●

        ●

Material B
```

they likely have similar:

- composition,
- structure,
- atomic properties,
- descriptor values.

If they are far apart:

```text
●                         ●
Material A                Material B
```

their descriptor fingerprints are very different.

---

# 11A.8.11 Coloring PCA by Material Properties

PCA becomes more scientifically useful when combined with known properties.

Example:

Color points by band gap.

```python
plt.figure(
    figsize=(8,6)
)


plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["band_gap"]
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.colorbar(
    label="Band Gap (eV)"
)


plt.show()
```

Now the materials map contains additional scientific information.

We can visually examine whether:

- similar regions have similar band gaps,
- clusters of materials exist,
- unusual materials appear as outliers.

---

# 11A.8.12 Number of Components and Information Loss

Reducing dimensions always causes some information loss.

For example:

```text
200 descriptors

↓

2 components
```

cannot preserve everything.

Therefore,

we need to measure how much information remains.

PCA provides this through:

```python
explained_variance_ratio_
```

---

# 11A.8.13 Explained Variance Ratio

Use:

```python
pca.explained_variance_ratio_
```

Example output:

```text
[
0.42,

0.21
]
```

Meaning:

```text
PC1 explains 42% of variance

PC2 explains 21% of variance
```

Together:

```text
42% + 21%

=

63%
```

of the original information is preserved.

---

# 11A.8.14 Total Explained Variance

Calculate:

```python
total_variance = (
    pca.explained_variance_ratio_
    .sum()
)


print(total_variance)
```

Example:

```text
0.63
```

Meaning:

```text
Two principal components preserve 63% of the original variance.
```

---

# 11A.8.15 Increasing the Number of Components

Sometimes two dimensions are not enough.

For example:

```python
pca = PCA(
    n_components=10
)


X_pca = pca.fit_transform(
    X_scaled
)
```

Now:

```text
200 descriptors

↓

10 principal components
```

The model retains more information.

---

# 11A.8.16 Automatic Component Selection

Instead of manually choosing the number of components,

we can preserve a target percentage of variance.

Example:

```python
pca = PCA(
    n_components=0.95
)
```

This means:

```text
Keep enough components to preserve 95% variance
```

The algorithm automatically determines the required number of components.

---

# 11A.8.17 Finding the Required Number of Components

After fitting:

```python
pca.n_components_
```

Example:

```text
35
```

Interpretation:

```text
200 original descriptors

↓

35 principal components

↓

95% information retained
```

---

# 11A.8.18 Complete PCA Workflow

A complete materials PCA implementation:

```python
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


# Load dataset

df = pd.read_csv(
    "materials_descriptors.csv"
)


# Select descriptors

X = df[
    [
        "density",
        "volume",
        "atomic_mass",
        "electronegativity",
        "packing_fraction"
    ]
]


# Scale features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# PCA

pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X_scaled
)


print(
    X_pca.shape
)


print(
    pca.explained_variance_ratio_
)
```

---

# 11A.8.19 Materials Informatics Interpretation

The computational workflow is:

```text
Crystal Structures

↓

pymatgen

↓

matminer descriptors

↓

200-dimensional feature space

↓

StandardScaler

↓

PCA

↓

2D Materials Map
```

The final PCA plot allows researchers to explore:

- material families,
- descriptor similarity,
- unusual structures,
- chemical trends,
- potential candidates for discovery.

---

# 11A.8.20 Common PCA Mistakes

## Mistake 1: Using PCA Without Scaling

Incorrect:

```python
PCA().fit_transform(X)
```

when descriptors have different units.

Correct:

```python
X_scaled = scaler.fit_transform(X)

X_pca = PCA().fit_transform(X_scaled)
```

---

## Mistake 2: Including Material IDs

Incorrect:

```text
material_id

density

volume
```

IDs have no physical meaning.

---

## Mistake 3: Treating Principal Components as Physical Variables

PC1 is not:

```text
Density
```

or

```text
Composition
```

It is a mathematical combination of many descriptors.

---

# 11A.8.21 Summary

In this section, we implemented PCA using Python.

The complete workflow was:

```text
Descriptor Matrix

↓

Standardization

↓

PCA

↓

Principal Components

↓

Visualization
```

We learned how to:

- create PCA models,
- reduce dimensionality,
- calculate explained variance,
- visualize materials spaces,
- interpret PCA coordinates.

PCA transforms hundreds of descriptors into a compact representation that allows scientists to explore hidden structures inside complex materials datasets.

In the next section, we will examine the internal information learned by PCA by analyzing:

**principal component loadings and descriptor contributions.**

# 11A.9 Understanding PCA Loadings and Descriptor Contributions

In the previous section, we successfully transformed a high-dimensional materials descriptor space into a low-dimensional representation using PCA.

However, an important scientific question remains:

> **What do the principal components actually represent?**

A PCA plot can show that materials are separated into different regions, but it does not immediately explain **why** they are separated.

For materials scientists, this interpretation is extremely important.

A machine learning model is useful not only because it predicts or separates data, but because we can understand the scientific factors controlling the behavior.

PCA loadings provide this interpretation.

---

# 11A.9.1 What Are PCA Loadings?

The principal components are mathematical combinations of the original descriptors.

For example,

suppose our dataset contains:

```text
Density

Atomic Radius

Electronegativity

Volume

Bond Length
```

PCA creates:

```text
PC1

PC2

PC3
```

However,

PC1 is not a new physical property.

It is created from a combination of the original descriptors.

Conceptually:

```text
PC1

=

(weight1 × Density)

+

(weight2 × Atomic Radius)

+

(weight3 × Electronegativity)

+

...
```

The weights are called:

```text
PCA Loadings
```

---

# 11A.9.2 Mathematical Meaning of Loadings

A principal component can be written as:

```text
PC1

=

a1X1

+

a2X2

+

a3X3

+

...

+
anXn
```

where:

```text
X

=

Original descriptor
```

and

```text
a

=

Loading coefficient
```

The loading tells us how strongly each original descriptor contributes to the principal component.

---

# 11A.9.3 Accessing PCA Loadings in Python

Scikit-learn stores the loading matrix in:

```python
pca.components_
```

Example:

```python
print(pca.components_)
```

Output:

```text
[
 [ 0.45, 0.38, -0.12, 0.71 ],

 [-0.22, 0.81, 0.55, -0.09 ]
]
```

Interpretation:

Rows:

```text
Principal Components
```

Columns:

```text
Original Features
```

For example:

```text
PC1:

0.45 × Density

+

0.38 × Volume

-

0.12 × Atomic Mass

+

0.71 × Electronegativity
```

---

# 11A.9.4 Understanding the Shape of the Loading Matrix

Suppose:

```python
X.shape
```

returns:

```text
(5000, 20)
```

Meaning:

```text
5000 materials

20 descriptors
```

and:

```python
pca = PCA(
    n_components=3
)
```

Then:

```python
pca.components_.shape
```

returns:

```text
(3,20)
```

Meaning:

```text
3 principal components

×

20 descriptor contributions
```

---

# 11A.9.5 Creating a Loading DataFrame

Raw arrays are difficult to interpret.

A DataFrame is easier.

```python
loading_df = pd.DataFrame(
    pca.components_,
    columns=X.columns
)


print(loading_df)
```

Example:

```text
        density   volume   radius   electronegativity

PC1      0.45     0.38     0.21        0.71

PC2     -0.22     0.81     0.55       -0.09
```

Now we can directly connect:

```text
Principal Component

↓

Descriptor Contribution
```

---

# 11A.9.6 Identifying Important Descriptors

To find the strongest contributors,

we examine the absolute loading values.

Example:

```python
pc1_loadings = loading_df.loc[0]


pc1_importance = (
    pc1_loadings
    .abs()
    .sort_values(
        ascending=False
    )
)


print(pc1_importance)
```

Output:

```text
electronegativity     0.71

density               0.45

volume                0.38

radius                0.21
```

Interpretation:

For PC1,

electronegativity has the strongest influence.

---

# 11A.9.7 Visualizing PCA Loadings

A bar plot is often useful.

```python
import matplotlib.pyplot as plt


plt.figure(
    figsize=(10,5)
)


plt.bar(
    loading_df.columns,
    loading_df.iloc[0]
)


plt.xticks(
    rotation=90
)


plt.xlabel(
    "Descriptor"
)


plt.ylabel(
    "Loading"
)


plt.title(
    "PC1 Descriptor Contributions"
)


plt.show()
```

This plot shows which descriptors dominate PC1.

---

# 11A.9.8 Positive and Negative Loadings

Loadings can be positive or negative.

Example:

```text
Density

+

Electronegativity

-

Atomic Radius
```

A positive loading means the descriptor increases the value of the principal component.

A negative loading means the descriptor decreases it.

Important:

The sign itself is not physically absolute.

The direction of a principal component can be reversed mathematically.

For example:

```text
PC1

=

0.5 Density

+

0.3 Volume
```

is equivalent to:

```text
PC1

=

-0.5 Density

-

0.3 Volume
```

Both describe the same direction.

The important information is the relative contribution.

---

# 11A.9.9 PCA Loadings in Materials Science

Loadings help answer scientific questions.

Example:

Suppose we analyze oxide materials.

PCA reveals:

```text
PC1 strongly depends on:

- oxygen fraction
- electronegativity
- bond length
```

This suggests that PC1 represents a chemical bonding trend.

Another example:

For metallic materials:

```text
PC1 strongly depends on:

- atomic radius
- density
- atomic volume
```

This may represent an atomic size or packing trend.

---

# 11A.9.10 Example: Interpreting a Materials Map

Suppose PCA creates:

```text
PC1

↓

Atomic size contribution


PC2

↓

Chemical bonding contribution
```

The PCA map may show:

```text
        PC2

         ↑

 Ceramic ●

          \

           \

            \

Metal ●--------------→ PC1

```

Interpretation:

Materials separate because of differences in:

- atomic size,
- bonding characteristics,
- composition.

PCA converts a complicated descriptor space into interpretable scientific trends.

---

# 11A.9.11 Complete PCA Interpretation Workflow

A complete workflow:

```python
# Fit PCA

pca = PCA(
    n_components=3
)


X_pca = pca.fit_transform(
    X_scaled
)


# Extract loadings

loadings = pd.DataFrame(
    pca.components_,
    columns=X.columns
)


# Analyze PC1

pc1 = (
    loadings
    .iloc[0]
    .abs()
    .sort_values(
        ascending=False
    )
)


print(pc1)
```

---

# 11A.9.12 Combining PCA Coordinates With Materials Information

Often we want to combine PCA results with the original dataset.

Example:

```python
pca_results = pd.DataFrame(
    X_pca,
    columns=[
        "PC1",
        "PC2",
        "PC3"
    ]
)


final_df = pd.concat(
    [
        df,
        pca_results
    ],
    axis=1
)
```

Now each material has:

```text
Original Descriptors

+

PCA Coordinates
```

Example:

```text
Material

Density

Volume

Band Gap

PC1

PC2
```

This is useful for:

- visualization,
- clustering,
- searching similar materials.

---

# 11A.9.13 PCA for Descriptor Selection

PCA can also reveal redundancy.

Materials datasets often contain many correlated descriptors.

Example:

```text
Atomic Radius

Atomic Volume

Bond Length

```

may contain overlapping information.

PCA detects this because these descriptors contribute to similar directions.

Therefore,

PCA can help identify:

- redundant features,
- correlated descriptors,
- unnecessary variables.

---

# 11A.9.14 PCA Does Not Replace Physical Understanding

Although PCA is powerful,

it does not discover scientific laws automatically.

A principal component is:

```text
Mathematical Pattern

↓

Not necessarily Physical Mechanism
```

The scientist must interpret the results using knowledge of:

- chemistry,
- crystallography,
- physics,
- materials science.

Machine learning provides patterns.

Scientific reasoning provides meaning.

---

# 11A.9.15 Common Mistakes When Interpreting Loadings

## Mistake 1: Treating PC1 as a Real Property

Incorrect:

```text
PC1 = Band Gap
```

Correct:

```text
PC1 = Combination of Descriptor Trends
```

---

## Mistake 2: Ignoring Feature Scaling

Loadings become misleading if PCA was performed on unscaled features.

Always:

```text
Scale

↓

PCA

↓

Interpret Loadings
```

---

## Mistake 3: Looking Only at PCA Plots

A plot shows separation.

Loadings explain the reason.

Both are required.

---

# 11A.9.16 Summary

PCA loadings provide the connection between mathematical dimensionality reduction and scientific interpretation.

The complete workflow is:

```text
Materials Descriptors

↓

Standardization

↓

PCA

↓

Principal Components

↓

Loadings

↓

Scientific Interpretation
```

Using PCA loadings, researchers can identify:

- important descriptors,
- hidden chemical trends,
- correlated variables,
- dominant factors controlling materials spaces.

This step transforms PCA from a visualization method into a scientific analysis tool.

In the next section, we will build complete PCA visualization workflows and learn how to create publication-quality materials maps.

# 11A.10 Visualizing Materials Space Using PCA

After performing PCA, the next important step is visualization.

A major motivation behind dimensionality reduction is that it allows scientists to see patterns that are hidden inside high-dimensional descriptor spaces.

A materials dataset containing hundreds of descriptors cannot be directly visualized.

For example:

```text
Material Descriptor Space

Dimensions:

Density

Volume

Atomic Radius

Electronegativity

Bond Length

Coordination Number

...

200+ features
```

Humans cannot directly interpret this space.

PCA transforms this high-dimensional space into a smaller representation.

The workflow becomes:

```text
200-dimensional Materials Space

↓

PCA

↓

2-dimensional Materials Map

↓

Visualization

↓

Scientific Interpretation
```

Each point in the PCA plot represents a material.

The location of the point represents its similarity to other materials based on the descriptors.

---

# 11A.10.1 Why Visualization Matters in Materials Informatics

Visualization is not only for making attractive figures.

It helps answer scientific questions.

Examples:

### Question 1

Do chemically similar materials occupy similar regions?

```text
Composition

↓

Descriptor Similarity

↓

Spatial Clustering
```

---

### Question 2

Are there naturally occurring material families?

```text
Materials Dataset

↓

PCA

↓

Groups of Similar Materials
```

---

### Question 3

Are there unusual materials?

A material far away from the main distribution may indicate:

- unusual chemistry,
- rare structures,
- possible outliers,
- interesting candidates for discovery.

---

# 11A.10.2 Basic PCA Scatter Plot

Assume PCA has already been performed.

```python
X_pca
```

contains:

```text
PC1

PC2
```

coordinates.

We can plot them using matplotlib.

```python
import matplotlib.pyplot as plt


plt.figure(
    figsize=(8,6)
)


plt.scatter(
    X_pca[:,0],
    X_pca[:,1]
)


plt.xlabel(
    "Principal Component 1"
)


plt.ylabel(
    "Principal Component 2"
)


plt.title(
    "Materials Space Using PCA"
)


plt.show()
```

The resulting plot is a two-dimensional representation of the original descriptor space.

---

# 11A.10.3 Understanding the PCA Axes

Unlike ordinary plots,

the axes are not physical quantities.

The x-axis:

```text
PC1
```

is a combination of many descriptors.

The y-axis:

```text
PC2
```

is another independent combination.

For example:

```text
PC1

=

0.4 Density

+

0.3 Atomic Radius

+

0.5 Electronegativity

```

Therefore,

the axes represent directions of maximum variation in the descriptor space.

---

# 11A.10.4 Adding Material Labels

Often we want to identify individual materials.

Example:

```python
plt.figure(
    figsize=(8,6)
)


plt.scatter(
    X_pca[:,0],
    X_pca[:,1]
)


for i, name in enumerate(df["formula"]):

    plt.text(
        X_pca[i,0],
        X_pca[i,1],
        name
    )


plt.show()
```

This displays material formulas near their locations.

However,

for thousands of materials this becomes unreadable.

Therefore, labels are usually only used for selected materials.

---

# 11A.10.5 Coloring by a Materials Property

A PCA plot becomes much more informative when points are colored according to a physical property.

Examples:

- band gap,
- formation energy,
- density,
- magnetic moment,
- elastic modulus.

Example:

```python
plt.figure(
    figsize=(8,6)
)


scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["band_gap"]
)


plt.colorbar(
    scatter,
    label="Band Gap (eV)"
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.title(
    "PCA Space Colored by Band Gap"
)


plt.show()
```

Now the spatial distribution contains scientific information.

---

# 11A.10.6 Interpreting Property Gradients

A useful pattern is a smooth change in color.

Example:

```text
Low Band Gap

● ● ●

  ● ● ●

    ● ● ●

          ●

High Band Gap
```

This suggests:

```text
Position in PCA Space

↓

Related to Band Gap Variation
```

The descriptors used for PCA contain information related to the electronic property.

---

# 11A.10.7 Finding Materials Families

Suppose a PCA plot shows:

```text
                PC2

                 ↑


        ● ● ●

      ● ● ● ●


                         ● ● ●

                       ● ● ● ●


------------------------------------→ PC1
```

Two groups appear.

Possible interpretations:

- different chemical families,
- different crystal structures,
- different bonding environments.

However,

PCA alone does not define the groups.

Clustering algorithms will be introduced later for automatic identification.

---

# 11A.10.8 Three-Dimensional PCA Visualization

Sometimes two components are insufficient.

A third component can provide additional information.

Perform PCA:

```python
pca = PCA(
    n_components=3
)


X_pca = pca.fit_transform(
    X_scaled
)
```

Now:

```text
PC1

PC2

PC3
```

are available.

---

# 11A.10.9 Creating a 3D PCA Plot

Use matplotlib 3D plotting.

```python
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(
    figsize=(8,6)
)


ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    X_pca[:,2]
)


ax.set_xlabel(
    "PC1"
)


ax.set_ylabel(
    "PC2"
)


ax.set_zlabel(
    "PC3"
)


plt.show()
```

This allows exploration of a three-dimensional materials space.

---

# 11A.10.10 Explained Variance Visualization

A PCA plot is useful only when we understand how much information each component contains.

The explained variance ratio:

```python
pca.explained_variance_ratio_
```

can be visualized.

Example:

```python
variance = (
    pca.explained_variance_ratio_
)


plt.figure(
    figsize=(8,5)
)


plt.bar(
    range(
        1,
        len(variance)+1
    ),
    variance
)


plt.xlabel(
    "Principal Component"
)


plt.ylabel(
    "Explained Variance"
)


plt.title(
    "PCA Explained Variance"
)


plt.show()
```

---

# 11A.10.11 Cumulative Explained Variance Plot

A more common analysis is cumulative variance.

```python
import numpy as np


cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)


plt.figure(
    figsize=(8,5)
)


plt.plot(
    cumulative_variance
)


plt.xlabel(
    "Number of Components"
)


plt.ylabel(
    "Cumulative Variance"
)


plt.title(
    "PCA Variance Retention"
)


plt.grid()


plt.show()
```

This helps determine:

```text
How many components are needed?
```

---

# 11A.10.12 PCA Visualization With Materials Categories

If categorical information is available,

we can color materials by category.

Examples:

- metal,
- semiconductor,
- insulator.

Example:

```python
colors = df["category"]


plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=colors
)
```

This allows comparison between known material classes.

---

# 11A.10.13 Example Materials Workflow

A complete visualization workflow:

```python
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


# Load data

df = pd.read_csv(
    "materials_descriptors.csv"
)


# Select descriptors

X = df.drop(
    [
        "formula",
        "band_gap"
    ],
    axis=1
)


# Scale

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# PCA

pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X_scaled
)


# Plot

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["band_gap"]
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.colorbar(
    label="Band Gap"
)


plt.show()
```

---

# 11A.10.14 Scientific Interpretation of PCA Maps

A PCA map can reveal:

## Similar Materials

Close points:

```text
Similar Descriptor Fingerprint
```

---

## Different Materials

Far points:

```text
Different Descriptor Fingerprint
```

---

## Hidden Trends

Gradients may indicate relationships between:

- composition,
- structure,
- electronic properties.

---

## Outliers

Isolated points may represent:

- unusual chemistry,
- rare structures,
- data errors.

---

# 11A.10.15 PCA Visualization in Research Papers

Many materials informatics papers use PCA figures to demonstrate:

- dataset diversity,
- similarity between materials,
- chemical space coverage,
- model applicability.

A typical research workflow:

```text
Materials Database

↓

Descriptor Generation

↓

Scaling

↓

PCA

↓

Chemical Space Map

↓

Clustering / Discovery
```

---

# 11A.10.16 Limitations of PCA Visualization

Although powerful,

PCA visualization has limitations.

## Limitation 1

Only a small fraction of information may be shown.

Example:

```text
200 dimensions

↓

2 dimensions
```

Some information is lost.

---

## Limitation 2

Distance in PCA space is not always physically meaningful.

Two materials close together may not always have identical properties.

---

## Limitation 3

Interpretation requires scientific knowledge.

A pattern in a PCA plot requires understanding of:

- chemistry,
- physics,
- crystallography.

---

# 11A.10.17 Summary

PCA visualization converts complex materials descriptor spaces into interpretable maps.

The complete workflow is:

```text
Descriptors

↓

Scaling

↓

PCA

↓

2D / 3D Coordinates

↓

Visualization

↓

Scientific Analysis
```

Using PCA visualization, materials scientists can explore:

- chemical similarity,
- material families,
- property trends,
- unusual structures.

However, PCA only reveals patterns visually.

To automatically identify groups of similar materials, we need clustering algorithms.

In the next section, we will implement:

**K-Means clustering for discovering material families.**

# 11A.11 K-Means Clustering Implementation for Materials Discovery

In the previous sections, we used PCA to reduce the dimensionality of materials descriptor spaces and visualize hidden patterns.

However, PCA itself does not automatically identify groups.

It only transforms the data.

A natural next question is:

> Can we automatically divide materials into groups based on their similarity?

This is the purpose of clustering.

Clustering is an **unsupervised learning technique** that discovers natural structures inside data without requiring predefined labels.

In materials informatics, clustering can help identify:

- material families,
- chemical similarity groups,
- structural classes,
- regions of similar properties,
- unusual materials.

The workflow is:

```text
Materials Descriptors

↓

Feature Scaling

↓

PCA (optional)

↓

Clustering Algorithm

↓

Material Groups
```

In this section, we will implement one of the most widely used clustering algorithms:

**K-Means clustering.**

---

# 11A.11.1 What Is K-Means Clustering?

K-Means attempts to divide a dataset into a predefined number of groups called:

```text
Clusters
```

Each cluster is represented by a central point called:

```text
Centroid
```

The algorithm attempts to assign materials to the closest centroid.

Conceptually:

```text
Materials

↓

Find Cluster Centers

↓

Assign Materials

↓

Update Centers

↓

Repeat
```

---

# 11A.11.2 Why K-Means for Materials Informatics?

Materials databases often contain thousands or millions of structures.

Manually identifying groups is impossible.

For example:

```text
50000 Materials

↓

Unknown Relationships

↓

K-Means

↓

Material Families
```

Possible applications:

- grouping similar compositions,
- identifying structural families,
- exploring chemical space,
- finding candidates similar to known materials.

---

# 11A.11.3 Mathematical Idea Behind K-Means

K-Means tries to minimize the distance between materials and their cluster centers.

The objective function is:

```text
Total Distance

=

Sum of distances between

materials

and

their assigned centroid
```

The algorithm searches for centroid positions that minimize this quantity.

---

# 11A.11.4 The K-Means Algorithm Steps

K-Means follows four main steps.

---

## Step 1: Choose Number of Clusters

The user specifies:

```python
n_clusters
```

Example:

```text
4 clusters
```

---

## Step 2: Initialize Centroids

The algorithm selects initial cluster centers.

Example:

```text
Centroid 1

Centroid 2

Centroid 3

Centroid 4
```

---

## Step 3: Assign Materials

Each material is assigned to the nearest centroid.

Conceptually:

```text
Material

↓

Calculate Distance

↓

Closest Centroid

↓

Cluster Assignment
```

---

## Step 4: Update Centroids

The centroid position is recalculated.

The process repeats until the clusters become stable.

---

# 11A.11.5 Importing K-Means in Python

K-Means is available in scikit-learn.

```python
from sklearn.cluster import KMeans
```

---

# 11A.11.6 Preparing Data for K-Means

K-Means uses distances.

Therefore,

feature scaling is essential.

The correct workflow is:

```text
Raw Descriptors

↓

StandardScaler

↓

Scaled Features

↓

K-Means
```

Example:

```python
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)
```

---

# 11A.11.7 Creating a K-Means Model

Create a clustering model:

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)
```

Parameters:

---

## n_clusters

Defines the number of groups.

Example:

```python
n_clusters=4
```

means:

```text
Create four material groups
```

---

## random_state

Controls initialization randomness.

Using:

```python
random_state=42
```

ensures reproducible results.

---

# 11A.11.8 Training the Model

Apply K-Means:

```python
kmeans.fit(
    X_scaled
)
```

During this step:

```text
Input Data

↓

Initialize Centroids

↓

Assign Clusters

↓

Optimize Centroids

↓

Final Model
```

---

# 11A.11.9 Getting Cluster Labels

After fitting,

each material receives a cluster number.

```python
labels = kmeans.labels_
```

Example:

```text
Material 1 → Cluster 0

Material 2 → Cluster 2

Material 3 → Cluster 1

Material 4 → Cluster 0
```

These labels represent the discovered material families.

---

# 11A.11.10 Using fit_predict()

A shorter approach:

```python
labels = kmeans.fit_predict(
    X_scaled
)
```

This performs:

```text
fit()

+

labels_
```

in one command.

---

# 11A.11.11 Adding Cluster Labels to Dataset

Usually we store cluster information with the original data.

```python
df["cluster"] = labels
```

Now the dataset contains:

```text
Material

Descriptors

Band Gap

Cluster ID
```

Example:

```text
Material      Band Gap     Cluster

SiO2          8.9          1

TiO2          3.2          2

Fe2O3         2.1          3
```

---

# 11A.11.12 Visualizing K-Means Clusters Using PCA

High-dimensional clusters cannot be directly visualized.

Therefore, PCA is often used before plotting.

Workflow:

```text
Descriptors

↓

Scaling

↓

K-Means

↓

PCA

↓

Visualization
```

or:

```text
Descriptors

↓

Scaling

↓

PCA

↓

K-Means

↓

Visualization
```

Both approaches are common.

---

# 11A.11.13 PCA + K-Means Visualization

Example:

```python
import matplotlib.pyplot as plt


plt.figure(
    figsize=(8,6)
)


plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.title(
    "K-Means Clustering in Materials Space"
)


plt.colorbar(
    label="Cluster"
)


plt.show()
```

The plot shows:

- each material,
- its location in PCA space,
- its assigned cluster.

---

# 11A.11.14 Understanding Cluster Centers

K-Means stores centroid positions.

Access them:

```python
kmeans.cluster_centers_
```

Example:

```text
[
 [0.52, -0.31, 1.2],

 [-0.45, 0.82, -0.5]
]
```

Each row represents one cluster center.

---

# 11A.11.15 Interpreting Cluster Centers in Materials Science

Cluster centers represent average materials characteristics.

For example:

Cluster 1 may have:

```text
High density

Large atomic mass

Low electronegativity
```

Cluster 2 may have:

```text
Low density

Small atomic radius

High electronegativity
```

This can reveal chemical trends.

---

# 11A.11.16 Complete K-Means Example

```python
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans


# Load dataset

df = pd.read_csv(
    "materials_descriptors.csv"
)


# Select descriptors

X = df[
    [
        "density",
        "volume",
        "atomic_mass",
        "electronegativity"
    ]
]


# Scale

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# K-Means

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)


labels = kmeans.fit_predict(
    X_scaled
)


# Add clusters

df["cluster"] = labels


print(
    df.head()
)
```

---

# 11A.11.17 Choosing the Number of Clusters

One of the biggest challenges in K-Means is selecting:

```python
n_clusters
```

The algorithm requires this value beforehand.

But real materials datasets usually do not tell us:

```text
How many material families exist?
```

Therefore, we need evaluation methods.

The two most common methods are:

- Elbow method
- Silhouette score

---

# 11A.11.18 Elbow Method

The elbow method uses:

```python
inertia_
```

Inertia measures:

```text
Total distance between materials

and

their cluster centers
```

Example:

```python
kmeans.inertia_
```

Lower values indicate tighter clusters.

---

# 11A.11.19 Elbow Plot

Example:

```python
inertias = []


for k in range(1,10):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(
        X_scaled
    )

    inertias.append(
        model.inertia_
    )


plt.plot(
    range(1,10),
    inertias
)


plt.xlabel(
    "Number of Clusters"
)


plt.ylabel(
    "Inertia"
)


plt.title(
    "Elbow Method"
)


plt.show()
```

The optimal number is often near the point where improvement slows.

---

# 11A.11.20 Silhouette Score

The silhouette score measures how well-separated clusters are.

Range:

```text
-1 to 1
```

Higher values indicate better clustering.

Import:

```python
from sklearn.metrics import silhouette_score
```

Example:

```python
score = silhouette_score(
    X_scaled,
    labels
)


print(score)
```

---

# 11A.11.21 Testing Different Cluster Numbers

A complete evaluation:

```python
from sklearn.metrics import silhouette_score


for k in range(2,10):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )


    labels = model.fit_predict(
        X_scaled
    )


    score = silhouette_score(
        X_scaled,
        labels
    )


    print(
        k,
        score
    )
```

The best cluster number usually produces the highest silhouette score.

---

# 11A.11.22 Materials Informatics Application

A realistic workflow:

```text
Materials Project Database

↓

Crystal Structures

↓

pymatgen

↓

matminer Descriptors

↓

Standardization

↓

K-Means

↓

Material Families
```

Possible discoveries:

- similar compounds,
- unexplored chemical regions,
- candidate materials,
- unusual structures.

---

# 11A.11.23 Limitations of K-Means

Although powerful, K-Means has limitations.

---

## Requires Number of Clusters

The user must specify:

```python
n_clusters
```

---

## Assumes Spherical Clusters

K-Means works best when clusters are compact and separated.

---

## Sensitive to Outliers

Extreme materials can affect centroid positions.

---

## Depends on Initialization

Different initial centroids may produce slightly different results.

---

# 11A.11.24 Summary

K-Means is one of the most widely used clustering algorithms in materials informatics.

The complete workflow is:

```text
Materials Descriptors

↓

Scaling

↓

K-Means

↓

Cluster Labels

↓

Materials Families
```

Using Python and scikit-learn, we learned how to:

- train K-Means models,
- assign materials to clusters,
- visualize clusters,
- analyze centroids,
- select cluster numbers.

However, K-Means is not suitable for every type of materials dataset.

Some materials spaces contain irregular shapes, noise, or hierarchical relationships.

In the next sections, we will explore alternative clustering approaches:

- Hierarchical clustering,
- DBSCAN.

# 11A.12 Hierarchical Clustering for Materials Space Exploration

K-Means clustering is powerful and widely used, but it has an important limitation:

The number of clusters must be specified before training.

In many materials discovery problems, we do not know beforehand:

```text
How many material families exist?
```

The chemical space may naturally contain:

- large groups,
- small subgroups,
- nested relationships.

For example:

```text
All Materials

↓

Ceramics

↓

Oxides

↓

Transition Metal Oxides

↓

Titanium Oxides
```

This type of relationship is hierarchical.

Hierarchical clustering is designed to discover these structures.

---

# 11A.12.1 What Is Hierarchical Clustering?

Hierarchical clustering creates a tree-like organization of materials.

The output is called a:

```text
Dendrogram
```

A dendrogram represents how materials are grouped based on similarity.

Conceptually:

```text
Individual Materials

↓

Small Groups

↓

Larger Groups

↓

Complete Materials Family Tree
```

Unlike K-Means,

hierarchical clustering does not require selecting the number of clusters at the beginning.

---

# 11A.12.2 Why Hierarchical Clustering Is Useful in Materials Science

Materials often have natural hierarchical relationships.

For example:

```text
Materials

├── Metals

│
├── Transition Metals

│
└── Rare Earth Metals


├── Semiconductors

│
├── Oxides

│
└── Chalcogenides
```

A hierarchical algorithm can reveal these relationships.

Applications include:

- chemical space exploration,
- grouping crystal structures,
- discovering material families,
- identifying similarity between compounds.

---

# 11A.12.3 Basic Idea of Hierarchical Clustering

The algorithm starts with every material as an individual cluster.

Example:

```text
Material A

Material B

Material C

Material D
```

Then the closest materials are merged.

Step 1:

```text
A + B

C + D
```

Step 2:

```text
(A,B) + (C,D)
```

Eventually:

```text
All Materials

↓

One Large Cluster
```

The scientist chooses where to cut the hierarchy.

---

# 11A.12.4 Two Approaches to Hierarchical Clustering

There are two major strategies.

---

## Agglomerative Clustering

This is the most common approach.

It starts with:

```text
Each material = individual cluster
```

Then repeatedly merges clusters.

Workflow:

```text
N Materials

↓

N Clusters

↓

N-1 Clusters

↓

N-2 Clusters

↓

...

↓

1 Cluster
```

---

## Divisive Clustering

This starts with:

```text
One large cluster
```

and divides it into smaller groups.

Workflow:

```text
All Materials

↓

Large Groups

↓

Smaller Groups

↓

Individual Materials
```

Agglomerative clustering is more commonly used in machine learning libraries.

---

# 11A.12.5 Distance Between Materials

Hierarchical clustering requires a definition of similarity.

Usually this is based on distance.

For example:

```text
Material A

        ●


Material B

        ●
```

If two materials have similar descriptors:

```text
Small Distance

↓

High Similarity
```

If descriptors are different:

```text
Large Distance

↓

Low Similarity
```

Common distance metrics:

- Euclidean distance,
- Manhattan distance,
- cosine distance.

---

# 11A.12.6 Importance of Scaling

Like K-Means,

hierarchical clustering uses distances.

Therefore,

feature scaling is required.

Correct workflow:

```text
Materials Descriptors

↓

StandardScaler

↓

Hierarchical Clustering
```

Example:

```python
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)
```

---

# 11A.12.7 Importing Agglomerative Clustering

Scikit-learn provides an implementation.

```python
from sklearn.cluster import AgglomerativeClustering
```

---

# 11A.12.8 Creating the Model

Example:

```python
model = AgglomerativeClustering(
    n_clusters=4
)
```

Parameters:

---

## n_clusters

The final number of groups.

Example:

```python
n_clusters=4
```

means:

```text
Divide materials into four groups
```

Unlike K-Means,

the algorithm builds the complete hierarchy first.

---

# 11A.12.9 Training the Model

Apply clustering:

```python
labels = model.fit_predict(
    X_scaled
)
```

The algorithm:

```text
Calculate Distances

↓

Merge Similar Materials

↓

Create Hierarchy

↓

Assign Cluster Labels
```

---

# 11A.12.10 Adding Labels to Materials Dataset

Store the cluster assignments:

```python
df["cluster"] = labels
```

Example:

```text
Formula       Cluster

SiO2             0

TiO2             1

Fe2O3            2

Al2O3            0
```

---

# 11A.12.11 Complete Hierarchical Clustering Example

```python
import pandas as pd


from sklearn.preprocessing import StandardScaler


from sklearn.cluster import AgglomerativeClustering



# Load dataset

df = pd.read_csv(
    "materials_descriptors.csv"
)



# Select descriptors

X = df[
    [
        "density",
        "volume",
        "atomic_mass",
        "electronegativity"
    ]
]



# Scale features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)



# Hierarchical clustering

cluster_model = AgglomerativeClustering(
    n_clusters=4
)



labels = cluster_model.fit_predict(
    X_scaled
)



# Store clusters

df["cluster"] = labels



print(
    df.head()
)
```

---

# 11A.12.12 Creating a Dendrogram

The major advantage of hierarchical clustering is visualization of the hierarchy.

The dendrogram shows:

```text
Which materials merge first

↓

Which groups are similar

↓

Large-scale relationships
```

Scikit-learn does not directly generate dendrograms.

We use scipy.

---

# 11A.12.13 Importing Dendrogram Tools

```python
from scipy.cluster.hierarchy import dendrogram, linkage
```

---

# 11A.12.14 Calculating Linkage

```python
linked = linkage(
    X_scaled,
    method="ward"
)
```

The linkage matrix stores:

- merged clusters,
- distances,
- hierarchy information.

---

# 11A.12.15 Plotting the Dendrogram

```python
import matplotlib.pyplot as plt


plt.figure(
    figsize=(10,6)
)


dendrogram(
    linked
)


plt.xlabel(
    "Materials"
)


plt.ylabel(
    "Distance"
)


plt.title(
    "Materials Hierarchical Clustering"
)


plt.show()
```

---

# 11A.12.16 Understanding a Dendrogram

Example:

```text
Distance

 |

 |             _______

 |            |

 |      ______|

 |     |

 |_____|_____|_____|_____

     A     B     C     D
```

The vertical axis represents:

```text
Distance between clusters
```

Low merging distance:

```text
High Similarity
```

High merging distance:

```text
Low Similarity
```

---

# 11A.12.17 Choosing Number of Clusters From Dendrogram

Unlike K-Means,

we can inspect the dendrogram.

Example:

```text
Large vertical gap

↓

Possible cluster boundary
```

A horizontal cut produces clusters.

Example:

```text
          _________

         |

---------|----------------

    |        |       |

 Cluster1 Cluster2 Cluster3
```

---

# 11A.12.18 Linkage Methods

The linkage method determines how cluster distances are calculated.

Common choices:

---

## Ward Linkage

```python
method="ward"
```

Attempts to minimize variance inside clusters.

Often a good default.

---

## Complete Linkage

Uses the maximum distance between points.

Creates compact clusters.

---

## Average Linkage

Uses average distance.

Provides balanced grouping.

---

Example:

```python
linked = linkage(
    X_scaled,
    method="average"
)
```

---

# 11A.12.19 PCA Visualization of Hierarchical Clusters

As before,

high-dimensional clusters can be visualized using PCA.

Workflow:

```text
Descriptors

↓

Scaling

↓

Hierarchical Clustering

↓

PCA

↓

Visualization
```

Example:

```python
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.title(
    "Hierarchical Clustering in PCA Space"
)


plt.show()
```

---

# 11A.12.20 Comparing K-Means and Hierarchical Clustering

| Feature | K-Means | Hierarchical |
|---|---|---|
| Need cluster number | Yes | Not initially |
| Output | Cluster labels | Hierarchy |
| Uses centroids | Yes | No |
| Handles hierarchy | No | Yes |
| Speed | Faster | Slower |
| Large datasets | Better | More expensive |

---

# 11A.12.21 Materials Informatics Example

Imagine a database containing:

```text
10000 Crystal Structures
```

with descriptors:

```text
Composition

Atomic Properties

Structural Features

Bond Information
```

Hierarchical clustering may reveal:

```text
All Materials

↓

Oxides

↓

Transition Metal Oxides

↓

Iron Oxides
```

This provides a chemically meaningful organization.

---

# 11A.12.22 Limitations of Hierarchical Clustering

## Computational Cost

The method becomes expensive for very large datasets.

For millions of materials,

K-Means or density methods may be preferred.

---

## Sensitive to Distance Metric

Different distance choices can produce different structures.

---

## Early Decisions Cannot Be Easily Changed

Once two clusters merge,

the algorithm does not separate them again.

---

# 11A.12.23 Summary

Hierarchical clustering provides a powerful method for exploring relationships between materials.

The workflow is:

```text
Materials Descriptors

↓

Feature Scaling

↓

Distance Calculation

↓

Hierarchical Merging

↓

Dendrogram

↓

Material Families
```

Using Python, we learned how to:

- perform agglomerative clustering,
- generate cluster labels,
- create dendrograms,
- analyze material similarity.

Hierarchical clustering is especially useful when materials naturally form nested chemical families.

However, many real materials datasets contain noise and irregular cluster shapes.

For these cases, density-based methods are more suitable.

The next section introduces:

**DBSCAN clustering for discovering irregular materials structures and outliers.**

# 11A.13 DBSCAN Clustering for Materials Discovery and Outlier Detection

In the previous sections, we explored two major clustering approaches:

```text
K-Means

↓

Centroid-based clustering
```

and

```text
Hierarchical Clustering

↓

Tree-based grouping
```

Both methods are powerful, but they have limitations.

Many materials datasets do not contain perfectly separated groups.

Real chemical spaces often contain:

- irregular clusters,
- overlapping families,
- rare materials,
- isolated compounds,
- noisy data.

For these situations, a density-based approach is more appropriate.

This leads us to:

**DBSCAN**

which stands for:

```text
Density-Based Spatial Clustering of Applications with Noise
```

DBSCAN identifies clusters based on the density of data points rather than predefined centers.

---

# 11A.13.1 Why DBSCAN Is Important in Materials Informatics

Materials discovery often involves searching large chemical spaces.

A database may contain:

```text
Millions of Possible Structures

↓

A Small Number of Known Materials

↓

Rare Interesting Candidates
```

Traditional clustering methods may force every material into a group.

However, in materials science,

some materials are naturally unusual.

Examples:

- rare compositions,
- extreme electronic properties,
- unusual crystal structures,
- metastable phases.

DBSCAN can identify these automatically.

---

# 11A.13.2 Main Idea of DBSCAN

DBSCAN is based on density.

The idea:

```text
High Density Region

↓

Cluster
```

```text
Low Density Region

↓

Noise / Outlier
```

Instead of asking:

```text
Where is the nearest centroid?
```

DBSCAN asks:

```text
Are there enough nearby points to form a group?
```

---

# 11A.13.3 Difference Between K-Means and DBSCAN

K-Means:

```text
Choose Number of Clusters

↓

Find Centroids

↓

Assign Points
```

DBSCAN:

```text
Find Dense Regions

↓

Create Clusters

↓

Identify Noise
```

Important difference:

DBSCAN does not require:

```python
n_clusters
```

before training.

---

# 11A.13.4 DBSCAN Parameters

DBSCAN mainly depends on two parameters:

## 1. epsilon (eps)

Defines the neighborhood radius.

Conceptually:

```text
eps

↓

Maximum distance for neighbors
```

If two materials are closer than eps:

```text
Potentially Related
```

---

## 2. min_samples

Defines the minimum number of nearby points required to create a dense region.

Example:

```python
min_samples=5
```

means:

A point needs at least five nearby points to become part of a cluster.

---

# 11A.13.5 DBSCAN Point Classification

DBSCAN divides points into three categories.

---

## Core Point

A point with enough neighbors.

Example:

```text
Point A

Nearby points:

B C D E F
```

Since the density requirement is satisfied:

```text
Core Point
```

---

## Border Point

A point near a cluster but not dense enough itself.

Example:

```text
Cluster

● ● ● ●

    ●

Border point
```

---

## Noise Point

An isolated point.

Example:

```text
●



        ● ● ●
```

The isolated material is classified as:

```text
Noise
```

---

# 11A.13.6 Why Noise Detection Matters for Materials

In materials science,

outliers are often scientifically interesting.

A noise point may represent:

- unusual chemistry,
- rare structures,
- incorrect data,
- promising discovery candidates.

Therefore,

DBSCAN provides two outputs:

```text
Material Families

+

Potentially Interesting Outliers
```

---

# 11A.13.7 Preparing Data for DBSCAN

DBSCAN uses distance calculations.

Therefore:

```text
Feature Scaling

↓

Required
```

Workflow:

```text
pymatgen descriptors

↓

matminer features

↓

StandardScaler

↓

DBSCAN
```

Example:

```python
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)
```

---

# 11A.13.8 Importing DBSCAN

Scikit-learn provides DBSCAN.

```python
from sklearn.cluster import DBSCAN
```

---

# 11A.13.9 Creating a DBSCAN Model

Example:

```python
dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)
```

Parameters:

---

## eps

Controls neighborhood size.

Small value:

```text
Small Dense Clusters

More Noise
```

Large value:

```text
Large Clusters

Less Noise
```

---

## min_samples

Controls density requirement.

Large value:

```text
Only Very Dense Groups
```

Small value:

```text
More Clusters
```

---

# 11A.13.10 Training DBSCAN

Apply:

```python
labels = dbscan.fit_predict(
    X_scaled
)
```

The algorithm:

```text
Calculate Distances

↓

Find Dense Regions

↓

Assign Clusters

↓

Mark Noise
```

---

# 11A.13.11 Understanding DBSCAN Labels

DBSCAN labels look like:

```python
labels
```

Example:

```text
[
0,

0,

1,

1,

-1,

2
]
```

Meaning:

```text
0 → Cluster 0

1 → Cluster 1

2 → Cluster 2

-1 → Noise
```

The value:

```text
-1
```

is reserved for outliers.

---

# 11A.13.12 Complete DBSCAN Example

```python
import pandas as pd


from sklearn.preprocessing import StandardScaler


from sklearn.cluster import DBSCAN



# Load dataset

df = pd.read_csv(
    "materials_descriptors.csv"
)



# Select descriptors

X = df[
    [
        "density",
        "volume",
        "atomic_mass",
        "electronegativity"
    ]
]



# Scale features

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)



# DBSCAN

dbscan = DBSCAN(
    eps=0.6,
    min_samples=10
)



labels = dbscan.fit_predict(
    X_scaled
)



# Store labels

df["cluster"] = labels



print(
    df.head()
)
```

---

# 11A.13.13 Counting Discovered Clusters

To count clusters:

```python
import numpy as np


clusters = np.unique(
    labels
)


print(clusters)
```

Example:

```text
[-1 0 1 2 3]
```

Interpretation:

```text
Four clusters

+

Noise points
```

---

# 11A.13.14 Counting Noise Materials

Noise points are:

```python
labels == -1
```

Example:

```python
noise = np.sum(
    labels == -1
)


print(noise)
```

Output:

```text
125
```

Meaning:

```text
125 materials were classified as outliers.
```

---

# 11A.13.15 Visualizing DBSCAN Clusters With PCA

As with previous clustering methods,

we often visualize DBSCAN results using PCA.

Workflow:

```text
Descriptors

↓

Scaling

↓

DBSCAN

↓

PCA

↓

Plot
```

Example:

```python
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.title(
    "DBSCAN Materials Clusters"
)


plt.colorbar(
    label="Cluster"
)


plt.show()
```

---

# 11A.13.16 Highlighting Outliers

Because DBSCAN identifies noise,

we can visualize unusual materials.

Example:

```python
noise_points = labels == -1


plt.scatter(
    X_pca[noise_points,0],
    X_pca[noise_points,1]
)
```

These points may represent:

- unusual compounds,
- rare structures,
- possible new candidates.

---

# 11A.13.17 Choosing DBSCAN Parameters

Unlike K-Means,

DBSCAN does not ask:

```text
How many clusters?
```

Instead, we choose:

```text
eps

and

min_samples
```

These strongly affect the result.

---

# 11A.13.18 Effect of eps

Small eps:

```text
Strict Density Requirement

↓

Many Noise Points
```

Large eps:

```text
Loose Requirement

↓

Large Combined Clusters
```

---

# 11A.13.19 Effect of min_samples

Small value:

```text
Small Groups Accepted
```

Large value:

```text
Only Strong Patterns Accepted
```

---

# 11A.13.20 Selecting eps Using Nearest Neighbor Analysis

A common approach is examining nearest-neighbor distances.

Import:

```python
from sklearn.neighbors import NearestNeighbors
```

Example:

```python
neighbors = NearestNeighbors(
    n_neighbors=5
)


neighbors.fit(
    X_scaled
)


distances, indices = neighbors.kneighbors(
    X_scaled
)
```

The distance distribution helps estimate a suitable eps value.

---

# 11A.13.21 Materials Informatics Workflow With DBSCAN

A realistic research workflow:

```text
Materials Database

↓

Crystal Structures

↓

pymatgen

↓

matminer Descriptors

↓

Scaling

↓

DBSCAN

↓

Material Families

+

Rare Materials
```

---

# 11A.13.22 Comparing Three Clustering Methods

| Method | Main Idea | Requires Number of Clusters | Finds Noise |
|---|---|---|---|
| K-Means | Centroids | Yes | No |
| Hierarchical | Tree structure | Optional | No |
| DBSCAN | Density | No | Yes |

---

# 11A.13.23 When Should We Use DBSCAN?

DBSCAN is useful when:

## The number of groups is unknown

Example:

```text
Large Materials Database

↓

Unknown Families
```

---

## Outliers are important

Example:

```text
Rare Materials Discovery
```

---

## Clusters are irregular

Example:

```text
Non-spherical chemical spaces
```

---

# 11A.13.24 Limitations of DBSCAN

## Sensitive to eps

A poor choice can produce:

- too many clusters,
- too much noise.

---

## Struggles With Different Density Regions

If one cluster is very dense and another is sparse,

DBSCAN may not separate them correctly.

---

## High-Dimensional Difficulty

In very high dimensions,

distance becomes less informative.

This is why PCA preprocessing is often helpful.

---

# 11A.13.25 DBSCAN After PCA

A common materials workflow is:

```text
200 Descriptors

↓

StandardScaler

↓

PCA

↓

10 Principal Components

↓

DBSCAN

↓

Clusters
```

Why?

PCA removes:

- noise,
- redundant features,
- highly correlated dimensions.

This often improves clustering performance.

---

# 11A.13.26 Complete PCA + DBSCAN Workflow

```python
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import DBSCAN



# Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)



# PCA reduction

pca = PCA(
    n_components=10
)


X_reduced = pca.fit_transform(
    X_scaled
)



# DBSCAN

dbscan = DBSCAN(
    eps=0.5,
    min_samples=8
)


labels = dbscan.fit_predict(
    X_reduced
)
```

---

# 11A.13.27 Summary

DBSCAN provides a powerful method for discovering hidden structures in materials datasets.

The workflow is:

```text
Materials Descriptors

↓

Scaling

↓

Dimensionality Reduction

↓

Density Analysis

↓

Clusters + Outliers
```

Unlike K-Means and hierarchical clustering,

DBSCAN can identify:

- unknown numbers of material families,
- irregular clusters,
- unusual materials.

For materials discovery,

the ability to identify rare points is especially valuable because unusual regions of chemical space often contain the most interesting candidates.

In the next section, we will combine PCA and clustering methods into a complete **materials chemical space exploration workflow**.

# 11A.14 Complete Materials Chemical Space Exploration Workflow

In the previous sections, we studied the major unsupervised learning techniques used in materials informatics:

- Principal Component Analysis (PCA)
- K-Means clustering
- Hierarchical clustering
- DBSCAN clustering

Each method provides a different view of materials space.

However, in real research projects, these methods are rarely used independently.

A typical materials informatics workflow combines:

```text
Materials Data

↓

Descriptor Generation

↓

Feature Processing

↓

Dimensionality Reduction

↓

Clustering

↓

Scientific Interpretation
```

This section combines everything into a complete practical workflow.

---

# 11A.14.1 What Is Chemical Space?

Chemical space refers to the complete set of possible materials represented by their chemical and structural characteristics.

A material is not represented only by its formula.

For example:

```text
TiO2
```

contains information about:

- elements,
- crystal structure,
- atomic arrangement,
- bonding,
- electronic environment.

In machine learning, we represent a material using descriptors.

Example:

```text
TiO2

↓

[
density,

volume,

atomic radius,

electronegativity,

coordination number,

bond length,

...
]
```

This descriptor vector defines its position in chemical space.

---

# 11A.14.2 Materials Space Representation

The complete transformation is:

```text
Crystal Structure

↓

pymatgen

↓

Matminer

↓

Numerical Descriptors

↓

High-dimensional Materials Space
```

Example:

```text
10000 Materials

×

200 Descriptors
```

This means:

```text
Every material is a point

inside a 200-dimensional space.
```

---

# 11A.14.3 Why Explore Chemical Space?

Chemical space exploration helps answer important research questions.

Examples:

---

## Finding Similar Materials

Question:

```text
Which materials are chemically similar to this known compound?
```

Workflow:

```text
Known Material

↓

Descriptor Space

↓

Nearest Similar Materials
```

---

## Finding Material Families

Question:

```text
Do unknown materials belong to known chemical groups?
```

Workflow:

```text
Descriptors

↓

Clustering

↓

Material Families
```

---

## Finding Unusual Candidates

Question:

```text
Which materials are different from everything known?
```

Workflow:

```text
Descriptor Space

↓

Outlier Detection

↓

Candidate Discovery
```

---

# 11A.14.4 Complete Workflow Overview

A complete unsupervised materials workflow:

```text
Crystal Structures

        ↓

pymatgen

        ↓

matminer Descriptors

        ↓

Feature Matrix

        ↓

StandardScaler

        ↓

PCA

        ↓

Chemical Space Visualization

        ↓

Clustering

        ↓

Material Families
```

---

# 11A.14.5 Step 1 — Obtaining Crystal Structures

The first step is collecting structural information.

Sources may include:

- Materials databases,
- experimental repositories,
- DFT calculations.

Examples:

```text
CIF files

POSCAR files

Computed structures
```

The structure contains:

- atomic species,
- coordinates,
- lattice parameters,
- symmetry information.

---

# 11A.14.6 Step 2 — Reading Structures Using pymatgen

Example:

```python
from pymatgen.core import Structure


structure = Structure.from_file(
    "POSCAR"
)


print(
    structure
)
```

The structure object now contains complete crystallographic information.

---

# 11A.14.7 Step 3 — Generating Descriptors

Machine learning cannot directly use crystal structures.

They must be converted into numerical features.

Using matminer:

Example:

```python
from matminer.featurizers.composition import ElementProperty
```

Create a featurizer:

```python
featurizer = ElementProperty.from_preset(
    "magpie"
)
```

Apply:

```python
features = featurizer.featurize(
    composition
)
```

Output:

```text
[
mean atomic number,

average radius,

electronegativity,

...
]
```

---

# 11A.14.8 Step 4 — Creating the Materials Dataset

After feature generation:

```text
Material ID

Formula

Descriptor 1

Descriptor 2

Descriptor 3

...

Descriptor 200
```

Example:

```python
import pandas as pd


df = pd.DataFrame(
    feature_data
)
```

This dataframe becomes the machine learning input.

---

# 11A.14.9 Step 5 — Cleaning the Dataset

Before analysis:

Check:

```python
df.info()
```

Missing values:

```python
df.isnull().sum()
```

Duplicates:

```python
df.duplicated().sum()
```

Remove problematic entries:

```python
df = df.dropna()
```

---

# 11A.14.10 Step 6 — Feature Scaling

Because descriptors have different units:

```text
Density

↓

g/cm³


Atomic Mass

↓

amu


Volume

↓

Å³
```

we standardize them.

```python
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)
```

Now:

```text
Mean = 0

Standard deviation = 1
```

---

# 11A.14.11 Step 7 — PCA Dimensionality Reduction

The original feature space may contain hundreds of dimensions.

Example:

```text
200 descriptors
```

Reduce:

```python
from sklearn.decomposition import PCA


pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X_scaled
)
```

Now:

```text
200 dimensions

↓

2 dimensions
```

---

# 11A.14.12 Visualizing Materials Space

Example:

```python
import matplotlib.pyplot as plt


plt.scatter(
    X_pca[:,0],
    X_pca[:,1]
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.title(
    "Materials Chemical Space"
)


plt.show()
```

The result is a map of materials similarity.

---

# 11A.14.13 Step 8 — Applying Clustering

After PCA,

we can identify groups.

Example using K-Means:

```python
from sklearn.cluster import KMeans


model = KMeans(
    n_clusters=5,
    random_state=42
)


clusters = model.fit_predict(
    X_pca
)
```

Add labels:

```python
df["cluster"] = clusters
```

---

# 11A.14.14 Complete PCA + K-Means Visualization

```python
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)


plt.xlabel(
    "PC1"
)


plt.ylabel(
    "PC2"
)


plt.title(
    "Clustered Materials Space"
)


plt.colorbar(
    label="Cluster"
)


plt.show()
```

---

# 11A.14.15 Scientific Interpretation

The final map may show:

```text
Cluster 1

↓

Oxide Materials


Cluster 2

↓

Metallic Materials


Cluster 3

↓

Semiconductors
```

The algorithm does not know chemistry.

It only sees descriptor similarity.

The scientist must interpret the clusters.

---

# 11A.14.16 Combining Clustering With Known Properties

Suppose we have:

```text
Band Gap

Formation Energy

Elastic Modulus
```

We can analyze cluster behavior.

Example:

```python
df.groupby(
    "cluster"
)["band_gap"].mean()
```

Output:

```text
Cluster 0    0.5 eV

Cluster 1    2.8 eV

Cluster 2    5.1 eV
```

This reveals property trends inside material families.

---

# 11A.14.17 Searching Similar Materials

PCA space can also be used for similarity search.

Example:

A known material:

```text
Target Material
```

Find nearby points:

```text
Nearest Materials

↓

Potential Candidates
```

This is useful for:

- replacing expensive elements,
- finding analog compounds,
- discovering related structures.

---

# 11A.14.18 Outlier Detection Workflow

Using DBSCAN:

```python
from sklearn.cluster import DBSCAN


dbscan = DBSCAN(
    eps=0.5,
    min_samples=10
)


labels = dbscan.fit_predict(
    X_pca
)
```

Materials labeled:

```python
-1
```

are unusual.

These may represent:

- rare chemistry,
- unexpected structures,
- possible discovery targets.

---

# 11A.14.19 Example Research Pipeline

A realistic research project:

```text
Materials Project Database

↓

100000 Crystal Structures

↓

pymatgen Processing

↓

matminer Descriptors

↓

500 Features

↓

Scaling

↓

PCA

↓

Chemical Space Map

↓

DBSCAN / K-Means

↓

Material Families

↓

Candidate Selection
```

---

# 11A.14.20 Combining With DFT Calculations

Unsupervised learning can guide expensive calculations.

Example:

```text
Chemical Space Exploration

↓

Find Interesting Region

↓

Select Candidates

↓

DFT Calculation

↓

Property Evaluation
```

Instead of calculating thousands of random materials,

machine learning helps prioritize promising regions.

---

# 11A.14.21 Example: Battery Material Discovery

Workflow:

```text
Known Battery Materials

↓

Descriptor Generation

↓

Chemical Space Mapping

↓

Cluster Analysis

↓

Identify Similar Unknown Materials

↓

DFT Calculation

↓

Experimental Testing
```

Possible targets:

- high voltage materials,
- stable electrodes,
- fast ion conductors.

---

# 11A.14.22 Example: Semiconductor Discovery

For semiconductor materials:

Descriptors:

```text
Atomic properties

Crystal structure

Composition

Bonding information
```

Workflow:

```text
Descriptor Space

↓

PCA

↓

Clusters

↓

Band Gap Analysis

↓

Candidate Screening
```

---

# 11A.14.23 Advantages of Combining PCA and Clustering

Using both methods provides:

## Visualization

PCA:

```text
Where are materials located?
```

---

## Organization

Clustering:

```text
Which materials belong together?
```

---

## Discovery

Outliers:

```text
Which materials are unusual?
```

Together:

```text
Understanding

+

Exploration

+

Discovery
```

---

# 11A.14.24 Limitations of Chemical Space Exploration

Unsupervised learning has limitations.

---

## Similarity Depends on Descriptors

If descriptors are poor:

```text
Poor Representation

↓

Poor Clustering
```

---

## Clusters Are Not Guaranteed Physical Families

A mathematical cluster may not always correspond to a real chemical class.

---

## Interpretation Requires Domain Knowledge

Machine learning finds patterns.

Scientists explain them.

---

# 11A.14.25 Complete Python Workflow Summary

```python
# 1. Load descriptors

X = df[features]


# 2. Scale

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# 3. PCA

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


# 4. Clustering

kmeans = KMeans(
    n_clusters=5
)

labels = kmeans.fit_predict(
    X_pca
)


# 5. Visualization

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels
)

plt.show()
```

---

# 11A.14.26 Summary

A complete materials chemical space exploration workflow combines:

```text
Physics

+

Crystal Structures

+

pymatgen

+

matminer

+

Machine Learning
```

The computational pipeline is:

```text
Crystal Structure

↓

Descriptors

↓

Scaling

↓

PCA

↓

Visualization

↓

Clustering

↓

Materials Discovery
```

PCA allows scientists to see complex materials landscapes.

Clustering organizes materials into families.

Together, they provide powerful tools for exploring chemical space before expensive calculations or experiments.

This completes the practical implementation part of Chapter 11.

The next major chapter will move from unsupervised learning to a new paradigm:

# Chapter 12 — Neural Networks for Materials Science

where we will learn how deep learning models learn complex representations directly from data.