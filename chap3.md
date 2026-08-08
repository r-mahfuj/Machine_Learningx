
# Chapter 3 — Decision Trees

# 3.1 Introduction

In the previous chapter, we studied Linear Regression in detail.

We learned that Linear Regression attempts to describe the relationship between input features and a target variable using a mathematical equation.

For example,

if we want to predict the hardness of an alloy,

Linear Regression assumes that hardness can be represented approximately as

$$
Hardness =
b_0
+
b_1(Carbon)
+
b_2(Grain\ Size)
+
b_3(Density)
$$

The model learns the coefficients

$$
b_0,b_1,b_2,b_3
$$

from the training data.

This approach is powerful when relationships are approximately linear.

However, real materials science problems are often much more complicated.

---

# 3.2 Why Linear Models Are Sometimes Insufficient

Consider predicting the strength of an alloy.

The strength depends on many interacting factors:

- chemical composition,
- crystal structure,
- grain size,
- phase distribution,
- heat treatment,
- defects,
- processing conditions.

The relationship between these variables is rarely a simple straight line.

For example,

a small increase in carbon concentration may have different effects depending on chromium content.

A grain refinement process may improve strength up to a certain point, after which additional refinement produces little improvement.

These relationships are nonlinear.

A Linear Regression model may struggle because it tries to force every relationship into a straight-line approximation.

This is where Decision Trees become useful.

---

# 3.3 The Basic Idea of a Decision Tree

A Decision Tree makes predictions by asking a sequence of questions.

The structure is similar to how humans make decisions.

For example, imagine deciding whether a material is suitable for a high-temperature application.

A human engineer might think:

```
Is melting temperature above 1500°C?

        |
        Yes

Is oxidation resistance high?

        |
        Yes

Is mechanical strength sufficient?

        |
        Yes

Suitable material
```

A Decision Tree works in a similar way.

It divides the dataset into smaller groups by asking questions about the features.

---

# 3.4 Tree Structure

A Decision Tree consists of several important components.

## Root Node

The first question in the tree.

It represents the complete dataset before any splitting occurs.

Example:

```
All Materials
```

---

## Internal Nodes

These are intermediate decision points.

They divide the data according to feature conditions.

Example:

```
Carbon > 0.5%
```

---

## Branches

The outcomes of each decision.

Example:

```
Yes

No
```

---

## Leaf Nodes

The final prediction.

For regression,

a leaf contains a numerical value.

For example:

```
Predicted Hardness = 280 HV
```

---

A simple tree may look like:

```
                 Carbon > 0.3%

                 /          \

              Yes            No

             /                 \

     Grain Size < 20        Prediction

          |

       Prediction
```

The model moves from the root node through branches until it reaches a final prediction.

---

# 3.5 Decision Trees as a Series of Questions

A Decision Tree does not directly learn an equation.

Instead,

it learns a set of rules.

For example:

```
IF carbon content > 0.4%

AND grain size < 15 μm

THEN predict hardness = 320 HV
```

Another region of the dataset may follow:

```
IF carbon content < 0.2%

THEN predict hardness = 180 HV
```

The complete collection of these rules forms the decision tree.

---

# 3.6 Decision Trees for Classification and Regression

Decision Trees can solve two major types of machine learning problems.

## Classification

The output is a category.

Examples:

- metal or ceramic,
- magnetic or non-magnetic,
- stable phase or unstable phase.

Example:

```
Input:

Composition + Structure

Output:

Phase = Austenite
```

---

## Regression

The output is a continuous numerical value.

Examples:

- hardness,
- band gap,
- formation energy,
- elastic modulus.

Example:

```
Input:

Crystal Structure + Composition

Output:

Band Gap = 2.4 eV
```

This chapter focuses mainly on **Decision Tree Regression**, because predicting material properties is usually a regression problem.

---

# 3.7 How Does a Decision Tree Learn?

The central question is:

> How does the algorithm decide where to split the data?

Suppose we have a dataset of alloys.

The model must decide:

Should it split according to

- carbon content?
- density?
- grain size?
- chromium concentration?

It tries different possibilities and chooses the split that creates the most useful separation.

The exact definition of "useful" depends on the problem.

For classification,

we use measures such as:

- Gini impurity,
- entropy,
- information gain.

For regression,

we use:

- variance reduction,
- mean squared error reduction.

---

# 3.8 Example of a Regression Split

Suppose we have the following hardness data.

| Carbon (%) | Hardness (HV) |
|------------|--------------:|
|0.10|160|
|0.20|190|
|0.30|230|
|0.40|280|
|0.50|320|

The tree considers possible questions.

Example:

```
Is Carbon > 0.25?
```

This creates two groups.

Group 1:

```
Carbon ≤ 0.25
```

contains

```
160,190
```

Group 2:

```
Carbon > 0.25
```

contains

```
230,280,320
```

The algorithm checks whether this split reduces prediction error.

If it does,

the split is accepted.

---

# 3.9 The Philosophy Behind Decision Trees

A Decision Tree follows a simple principle:

> Divide the data into increasingly similar groups until each group can be predicted accurately.

At the beginning,

the dataset may contain very different materials.

After multiple splits,

each region contains materials with similar behavior.

The final prediction is usually the average target value of samples inside each leaf.

---

# 3.10 Decision Trees in Materials Informatics

Decision Trees are particularly useful in materials science because they naturally handle complex relationships.

Examples include predicting:

- phase stability,
- alloy strength,
- battery performance,
- catalytic activity,
- thermal conductivity.

Unlike Linear Regression,

Decision Trees do not require the user to manually define mathematical relationships.

They can automatically discover nonlinear patterns.

For example,

a tree may discover a rule such as:

```
IF

Valence Electron Concentration > 7.2

AND

Atomic Radius Difference < 6%

THEN

Predict stable phase
```

Such rules can sometimes provide scientific insight into the factors controlling material behavior.

---

# 3.11 Advantages of Decision Trees

Decision Trees have several important advantages.

## 1. They Capture Nonlinear Relationships

A tree can model complex interactions between variables.

---

## 2. They Require Little Data Preparation

Unlike many algorithms,

Decision Trees generally do not require:

- feature scaling,
- normalization.

---

## 3. They Are Easy to Interpret

The final model can be converted into human-readable rules.

This is valuable in scientific research.

---

## 4. They Handle Different Feature Types

Trees can work with:

- numerical features,
- categorical features.

---

# 3.12 Limitations of Decision Trees

Despite their advantages,

Decision Trees also have weaknesses.

## 1. They Can Overfit Easily

A tree can continue splitting until it memorizes the training data.

This produces:

```
Very low training error

+

Poor test performance
```

---

## 2. They Can Be Unstable

Small changes in the dataset may create a completely different tree.

---

## 3. They May Not Generalize Well

A single tree often performs worse than ensemble methods.

This limitation leads to algorithms such as:

- Random Forest,
- Gradient Boosting,
- XGBoost.

---

# 3.13 Why Decision Trees Are Important Before XGBoost

Since your final goal is materials machine learning with models such as XGBoost,

understanding Decision Trees is essential.

XGBoost is not a completely different idea.

It is built from many Decision Trees combined intelligently.

The progression is:

```
Linear Regression

↓

Decision Trees

↓

Random Forest

↓

Gradient Boosting

↓

XGBoost
```

Each step improves the limitations of the previous model.

Therefore,

a strong understanding of Decision Trees will make XGBoost much easier to understand.

---

# 3.14 Connection with Pymatgen and Quantum ESPRESSO

In materials informatics,

Decision Trees can directly use descriptors generated from computational materials workflows.

For example:

Quantum ESPRESSO calculates:

- total energy,
- band structure,
- optimized lattice parameters.

Pymatgen extracts:

- composition features,
- structural descriptors,
- coordination information.

These become inputs:

```
Crystal Structure

↓

Pymatgen Descriptors

↓

Decision Tree Model

↓

Predicted Material Property
```

The algorithm does not know that the features came from quantum calculations.

It simply learns patterns between descriptors and target properties.

---

# 3.15 Chapter Roadmap

In this chapter, we will study Decision Trees from both theory and implementation.

We will cover:

- how trees split data,
- regression trees,
- impurity measures,
- Gini impurity,
- entropy and information gain,
- variance reduction,
- tree construction,
- controlling overfitting,
- Python implementation using Scikit-learn,
- visualization of trees,
- applying Decision Trees to materials datasets.

By the end of this chapter, you will be able to build, train, interpret, and modify Decision Tree models confidently.



# 3.16 How Decision Trees Split Data

The most important concept in understanding Decision Trees is the **splitting process**.

A Decision Tree does not learn a mathematical equation like Linear Regression.

Instead, it repeatedly divides the dataset into smaller and more organized groups.

Each division is called a **split**.

The quality of these splits determines how accurately the final model can predict new data.

---

# 3.17 The Basic Idea of Splitting

Imagine we have a dataset containing steel samples.

Each sample has:

- carbon content,
- grain size,
- density,

and the target variable:

- hardness.

A Decision Tree starts with the complete dataset.

At this point, the model knows only the average hardness of all samples.

For example:

```
All Steel Samples

Average Hardness = 250 HV
```

However, the samples may have very different properties.

Some may have hardness around

```
150 HV
```

while others may have hardness around

```
350 HV.
```

The model wants to divide these samples into groups where the hardness values become more similar.

---

# 3.18 Choosing a Feature for Splitting

The tree asks:

> Which feature should I use to divide the data?

Possible choices:

```
Carbon Content

Density

Grain Size

Chromium Concentration

```

The algorithm tests many possible splits.

For example:

```
Carbon > 0.30%
```

or

```
Grain Size < 20 μm
```

or

```
Density > 7.8 g/cm³
```

Each possible split creates different groups.

The algorithm chooses the split that produces the best improvement.

---

# 3.19 Example of a Split

Consider the following simplified dataset.

| Carbon (%) | Hardness (HV) |
|------------|--------------:|
|0.10|160|
|0.20|180|
|0.30|220|
|0.40|280|
|0.50|330|

Initially:

```
All Samples

Hardness:

160,180,220,280,330

Average = 234 HV
```

If the model predicts every sample as

```
234 HV
```

the error is large.

The tree tries a split:

```
Carbon ≤ 0.30%
```

The dataset becomes:

## Left Group

```
Carbon ≤ 0.30%

Hardness:

160
180
220
```

Average:

```
186.7 HV
```

---

## Right Group

```
Carbon > 0.30%

Hardness:

280
330
```

Average:

```
305 HV
```

Now the predictions are much closer to the actual values.

The split reduced the prediction error.

Therefore,

the tree considers this a useful split.

---

# 3.20 Recursive Splitting

A Decision Tree does not stop after one split.

After creating new groups,

it repeats the same process.

The structure becomes:

```
                 All Materials

                       |

              Carbon > 0.30%

                 /          \

              Yes            No

              |              |

      Further Splitting   Prediction

```

The left and right groups may again be divided.

For example:

```
Carbon > 0.30%

        |

Grain Size < 15 μm

        |

High Hardness Region
```

This repeated process is called **recursive partitioning**.

---

# 3.21 What Is the Tree Trying to Achieve?

The purpose of every split is the same:

> Make the samples inside each group as similar as possible.

For regression problems,

similar means:

the target values should have low variation.

For example,

this group is difficult to predict:

```
Hardness:

150

250

350

```

The values are widely spread.

The model does not know what value to choose.

However,

this group is easier:

```
Hardness:

245

250

255

```

The average value is a very good prediction.

Therefore,

a good split creates groups with small internal variation.

---

# 3.22 Splitting in Regression Trees

For regression problems,

Decision Trees usually measure split quality using **variance reduction** or **Mean Squared Error (MSE)**.

The idea is simple:

Before splitting:

```
Large error
```

After splitting:

```
Smaller errors in each group
```

The algorithm prefers the split that creates the largest reduction in error.

---

# 3.23 Mean Squared Error as a Split Criterion

For a group of samples,

the Mean Squared Error is:

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\bar{y})^2
$$

where:

- $(y_i)$ is the actual target value,
- $(\bar{y})$ is the average target value,
- $(n)$ is the number of samples.

The tree calculates this value before and after a possible split.

A good split produces a lower total MSE.

---

# 3.24 Understanding MSE Intuitively

Suppose a leaf contains:

```
Hardness:

200

210

220
```

The average is:

```
210
```

Prediction errors:

```
200 - 210 = -10

210 - 210 = 0

220 - 210 = +10
```

The errors are small.

Therefore,

MSE is low.

Now consider:

```
Hardness:

100

250

400
```

Average:

```
250
```

Errors:

```
-150

0

+150
```

The errors are much larger.

Therefore,

MSE is high.

The tree tries to create groups like the first example.

---

# 3.25 Finding the Best Split

The actual algorithm follows this procedure:

```
Step 1:

Select a feature.


Step 2:

Try possible split points.


Step 3:

Calculate the error after each split.


Step 4:

Choose the split with the lowest error.


Step 5:

Repeat the process on the new groups.

```

For a dataset with many features,

the tree may test thousands of possible splits.

Modern implementations perform this efficiently using optimized algorithms.

---

# 3.26 Example with Materials Features

Imagine a dataset containing:

| Feature | Meaning |
|-|-|
|Carbon|Composition|
|Atomic Radius|Element property|
|Electronegativity|Chemical behavior|
|Density|Physical property|

Target:

```
Formation Energy
```

The tree may discover a relationship such as:

```
Is Electronegativity < 1.8?

        |

       Yes

        |

Is Atomic Radius Difference < 5%?

        |

Formation Energy = -2.4 eV
```

The tree automatically discovers these decision rules from the data.

The researcher does not manually program them.

---

# 3.27 Why Decision Trees Are Called Non-Parametric Models

Linear Regression is called a **parametric model** because it assumes a fixed mathematical form:

$$
y=mx+b
$$

The model learns a limited number of parameters.

Decision Trees are different.

They do not assume that the relationship follows any specific equation.

They learn the structure directly from the data.

Therefore,

Decision Trees are called **non-parametric models**.

This flexibility allows them to capture highly complex relationships.

---

# 3.28 The Problem of Excessive Splitting

Although creating more splits can reduce training error,

it can create a serious problem.

Consider a tree that continues splitting until every sample has its own leaf.

The model may learn:

```
Sample 1 → Prediction A

Sample 2 → Prediction B

Sample 3 → Prediction C

```

The training error becomes almost zero.

However,

the tree has memorized the training data.

When a new material is introduced,

it cannot generalize.

This problem is called **overfitting**.

---

# 3.29 Controlling Tree Growth

To prevent overfitting,

we control how large the tree becomes.

Important parameters include:

## Maximum Depth

Controls the maximum number of levels.

Example:

```python
max_depth=5
```

means the tree cannot grow deeper than five levels.

---

## Minimum Samples Split

Controls how many samples are required before creating another split.

Example:

```python
min_samples_split=10
```

means a node needs at least ten samples before splitting.

---

## Minimum Samples Leaf

Controls the minimum number of samples allowed in a final leaf.

Example:

```python
min_samples_leaf=5
```

means every prediction region must contain at least five samples.

---

# 3.30 Decision Tree Thinking

A useful way to remember Decision Trees is:

Linear Regression asks:

> What equation best describes the relationship?

Decision Tree asks:

> What sequence of questions divides the data into meaningful groups?

Both methods attempt to learn relationships between inputs and outputs.

They simply represent those relationships differently.

---

# 3.31 Connection to Future Models

Understanding splitting is essential because every major tree-based algorithm builds upon this idea.

The progression is:

```
Decision Tree

↓

Many Trees Combined

↓

Random Forest

↓

Trees Built Sequentially

↓

Gradient Boosting

↓

Optimized Gradient Boosting

↓

XGBoost
```

Therefore,

learning how one tree makes decisions is the foundation for understanding the most powerful algorithms used in modern materials machine learning.


# 3.32 Mathematical Foundation of Regression Trees

A Decision Tree for regression may appear simple from the outside.

It looks like a collection of questions:

```
Is Carbon > 0.3?

Is Grain Size < 20?

Predict Hardness = 280 HV
```

However, behind these simple rules is a mathematical optimization process.

The tree is constantly searching for the set of splits that produces the most accurate predictions.

To understand Decision Trees deeply, we must understand the mathematics behind this optimization.

---

# 3.33 The Goal of a Regression Tree

The objective of a regression tree is:

> Divide the feature space into regions where the target values inside each region are as similar as possible.

Each final region is called a **leaf**.

For every leaf,

the model predicts the average value of all training samples that fall into that leaf.

Mathematically,

if a leaf contains samples:

$$
y_1,y_2,y_3,...,y_n
$$

the prediction of that leaf is:

$$
\hat{y}
=
\frac{1}{n}
\sum_{i=1}^{n}y_i
$$

This means the tree does not predict a complicated equation.

It predicts the average response of similar samples.

---

# 3.34 Example of a Leaf Prediction

Suppose a leaf contains three steel samples.

Their measured hardness values are:

$$
220,240,260
$$

The prediction becomes:

$$
\hat{y}
=
\frac{220+240+260}{3}
$$

$$
\hat{y}=240
$$

Therefore,

every new material entering this leaf receives:

```
Predicted hardness = 240 HV
```

---

# 3.35 Measuring Prediction Error Inside a Leaf

The question becomes:

How good is this prediction?

The tree compares the predicted value with the actual values.

The difference is called the residual.

$$
Residual =
Actual\ Value - Predicted\ Value
$$

For example:

Prediction:

$$
240
$$

Actual values:

$$
220,240,260
$$

Residuals:

$$
-20,0,+20
$$

The smaller these errors are,

the better the leaf represents the data.

---

# 3.36 Mean Squared Error (MSE)

The most common criterion used by regression trees is Mean Squared Error.

The formula is:

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y})^2
$$

where:

- $(y_i)$= actual target value,
- $(\hat{y})$ = predicted value,
- $(n)$ = number of samples.

---

# 3.37 Why Square the Errors?

You may ask:

Why not simply calculate the average error?

Because positive and negative errors cancel each other.

Example:

Errors:

$$
-20,+20
$$

Average error:

$$
\frac{-20+20}{2}=0
$$

This incorrectly suggests perfect prediction.

By squaring the errors:

$$
(-20)^2=400
$$

$$
(20)^2=400
$$

both errors contribute positively.

Therefore,

MSE correctly measures the magnitude of prediction error.

---

# 3.38 Example Calculation

Suppose a leaf contains hardness values:

$$
200,220,240
$$

Prediction:

$$
\hat{y}=220
$$

The errors are:

$$
200-220=-20
$$

$$
220-220=0
$$

$$
240-220=20
$$

Square the errors:

$$
400,0,400
$$

Average:

$$
MSE=
\frac{400+0+400}{3}
$$

$$
MSE=266.67
$$

This value represents the impurity of the leaf.

---

# 3.39 Choosing the Best Split

Suppose the tree considers splitting the dataset into two groups.

Before splitting:

```
All Materials

MSE = 500
```

After splitting:

```
Left Leaf:

MSE = 100


Right Leaf:

MSE = 150
```

The tree calculates the weighted error:

$$
MSE_{split}
=
\frac{n_L}{n}
MSE_L
+
\frac{n_R}{n}
MSE_R
$$

where:

- $(n_L)$ = number of samples in left child,
- $(n_R)$ = number of samples in right child,
- $(n)$ = total samples.

The tree chooses the split that minimizes this value.

---

# 3.40 Variance Reduction

Another way to describe the same idea is variance reduction.

Variance measures how spread out the values are.

A good split reduces variance.

The reduction is:

$$
Reduction
=
Variance_{parent}
-
Weighted\ Variance_{children}
$$

The tree chooses the split with the largest reduction.

---

# 3.41 Feature Space Partitioning

A powerful way to understand Decision Trees is to imagine the feature space.

Suppose we have two features:

- Carbon content,
- Grain size.

The tree divides this two-dimensional space.

Initially:

```
+----------------------+

|                      |

|     All Materials    |

|                      |

+----------------------+
```

After a split:

```
+------------+---------+

|            |         |

| Low Carbon | High C  |

|            |         |

+------------+---------+
```

After more splits:

```
+------+------+

|      |      |

|  A   |  B   |

+------+------+

|  C   |  D   |

+------+------+
```

Each rectangle represents a leaf.

Every region has its own prediction.

---

# 3.42 Why Decision Trees Can Model Nonlinear Relationships

A single straight line cannot easily represent complicated patterns.

However,

a Decision Tree can create many small regions.

Each region can have a different prediction.

Together,

these regions approximate complex functions.

For example:

A material property may behave like:

```
Low carbon:

slow increase


Medium carbon:

rapid increase


High carbon:

saturation
```

A tree can represent this behavior naturally through different branches.

---

# 3.43 The CART Algorithm

Most modern Decision Trees use an algorithm called:

**CART**

which stands for:

**Classification and Regression Trees**

CART builds binary trees.

This means every split creates exactly two branches.

Example:

```
Carbon > 0.3

        |

   Yes       No
```

CART searches for the best:

- feature,
- threshold,
- split.

For regression,

it usually minimizes:

$$
MSE
$$

or equivalently,

reduces variance.

---

# 3.44 Finding the Threshold

Suppose carbon values are:

```
0.1

0.2

0.3

0.4

0.5
```

The algorithm tests possible thresholds:

```
Carbon < 0.15

Carbon < 0.25

Carbon < 0.35

Carbon < 0.45
```

For each threshold,

it calculates the resulting error.

The best threshold is selected.

This process is repeated recursively.

---

# 3.45 Computational Complexity

For a large dataset,

testing every possible split can become expensive.

A materials database may contain:

- thousands of compounds,
- hundreds of descriptors,
- millions of possible splits.

Therefore,

efficient implementations use optimized searching techniques.

Libraries such as Scikit-learn use highly optimized algorithms written in low-level languages internally.

As a user,

you only interact with Python.

---

# 3.46 Implementing a Regression Tree in Python

Now we move from theory to implementation.

First,

import the model.

```python
from sklearn.tree import DecisionTreeRegressor
```

Create the model:

```python
tree_model = DecisionTreeRegressor()
```

Train it:

```python
tree_model.fit(
    X_train,
    y_train
)
```

Predict:

```python
predictions = tree_model.predict(
    X_test
)
```

---

# 3.47 Understanding the Code

## Importing the Model

```python
from sklearn.tree import DecisionTreeRegressor
```

Imports the regression version of the Decision Tree algorithm.

---

## Creating the Model

```python
tree_model = DecisionTreeRegressor()
```

Creates an empty decision tree.

At this stage:

- no splits exist,
- no rules have been learned.

---

## Training

```python
tree_model.fit(X_train,y_train)
```

The algorithm searches for:

- best features,
- best thresholds,
- best splits.

It continues until stopping conditions are reached.

---

## Prediction

```python
tree_model.predict(X_test)
```

For every new sample:

1. Start at the root.
2. Follow the decision rules.
3. Reach a leaf.
4. Return the leaf prediction.

---

# 3.48 Connection with Materials Data

Suppose your features are generated using Pymatgen:

```
Atomic Radius

Electronegativity

Coordination Number

Volume

Density
```

and your target is:

```
Formation Energy
```

The Decision Tree may discover rules such as:

```
IF

Volume < 80 Å³

AND

Electronegativity > 2

THEN

Formation Energy = -3.1 eV
```

These rules are not manually created.

They are learned from the dataset.

---

# 3.49 Looking Ahead

A single Decision Tree is powerful because it can capture nonlinear relationships.

However,

it has one major weakness:

it is unstable.

A small change in the training data can produce a completely different tree.

The solution is to combine many trees together.

The next section introduces **ensemble learning**, where multiple Decision Trees cooperate to create much stronger models.

This idea leads directly to:

- Random Forests,
- Gradient Boosting,
- XGBoost,

which are among the most successful algorithms in modern materials machine learning.


# 3.50 Overfitting in Decision Trees and How to Control Tree Growth

One of the most important concepts in Decision Trees is **overfitting**.

Understanding overfitting is essential because almost every advanced tree-based algorithm, including Random Forest and XGBoost, is designed partly to solve this problem.

A Decision Tree is a very flexible model.

This flexibility is both its greatest advantage and its greatest weakness.

It can learn extremely complicated relationships from data.

However,

if the tree becomes too complicated,

it may stop learning the true pattern and simply memorize the training dataset.

---

# 3.51 What Is Overfitting?

Overfitting occurs when a machine learning model learns the details and noise of the training data instead of learning the general relationship.

In simple words:

The model performs extremely well on data it has already seen,

but performs poorly on new unseen data.

A model that overfits has:

```
Low training error

+

High testing error
```

---

# 3.52 Example of Overfitting

Imagine we have experimental data for alloy hardness.

Training samples:

| Carbon (%) | Grain Size | Hardness |
|-|-|-|
|0.1|50|150|
|0.2|40|190|
|0.3|30|230|
|0.4|20|280|
|0.5|10|340|

A simple tree may learn:

```
Carbon > 0.3

        |

Higher hardness
```

This captures the general trend.

However,

a very deep tree may learn:

```
If carbon = 0.3

AND grain size = 30

AND sample number = 3

then hardness = 230
```

This is not learning material behavior.

It is memorizing specific samples.

---

# 3.53 Training Error vs Testing Error

To understand overfitting,

we compare two errors.

## Training Error

The error calculated using the same data used to train the model.

A complex tree usually performs extremely well here.

---

## Testing Error

The error calculated using new data.

This tells us whether the model can generalize.

A useful model should have:

```
Low training error

+

Low testing error
```

An overfitted model has:

```
Almost zero training error

+

Large testing error
```

---

# 3.54 Why Decision Trees Overfit Easily

Decision Trees are naturally prone to overfitting because they can keep creating more branches.

Imagine a tree with unlimited growth.

It can eventually create:

```
One sample

↓

One leaf

↓

Perfect prediction
```

The training error becomes zero.

However,

the tree has learned the dataset rather than the underlying physical relationship.

---

# 3.55 Tree Depth

The most important parameter controlling tree complexity is:

## Maximum Depth

Maximum depth controls how many levels the tree can have.

Example:

```python
DecisionTreeRegressor(
    max_depth=3
)
```

This limits the tree to three levels.

A shallow tree:

```
       Root

     /      \

   Node    Node

```

is simple and may underfit.

A very deep tree:

```
Root

 |

Node

 |

Node

 |

Node

 |

Leaf
```

may overfit.

The goal is to find the balance.

---

# 3.56 Underfitting vs Overfitting

There are three possible situations.

## 1. Underfitting

The model is too simple.

Characteristics:

- poor training performance,
- poor testing performance.

Example:

A tree with depth = 1 for a very complex problem.

---

## 2. Good Fit

The model captures the important patterns.

Characteristics:

- good training performance,
- good testing performance.

---

## 3. Overfitting

The model is too complex.

Characteristics:

- excellent training performance,
- poor testing performance.

---

# 3.57 Controlling Depth in Python

Example:

```python
tree_model = DecisionTreeRegressor(
    max_depth=5
)
```

---

## Explanation

### `max_depth`

Defines the maximum number of levels allowed.

A smaller value creates a simpler model.

A larger value creates a more complex model.

---

# 3.58 Minimum Samples Split

Another important parameter is:

```python
min_samples_split
```

It controls when a node is allowed to split.

Example:

```python
tree_model = DecisionTreeRegressor(
    min_samples_split=10
)
```

Meaning:

A node must contain at least 10 samples before it can be divided.

---

## Why Is This Useful?

Without this restriction,

the tree may create branches based on only one or two unusual samples.

These branches usually represent noise.

Increasing the minimum split size forces the model to learn broader patterns.

---

# 3.59 Minimum Samples Leaf

Another useful parameter is:

```python
min_samples_leaf
```

Example:

```python
tree_model = DecisionTreeRegressor(
    min_samples_leaf=5
)
```

This means every final leaf must contain at least five samples.

A prediction cannot be based on only one observation.

This improves generalization.

---

# 3.60 Maximum Features

Decision Trees can also limit how many features are considered during splitting.

Example:

```python
max_features=3
```

If a dataset contains 100 descriptors,

the tree only considers three at each split.

This can reduce model complexity.

This parameter becomes especially important in Random Forests.

---

# 3.61 Pre-Pruning

Controlling tree growth before the tree becomes too large is called:

## Pre-pruning

Examples:

- limiting maximum depth,
- increasing minimum samples split,
- increasing minimum samples leaf.

The tree stops growing early.

---

# 3.62 Post-Pruning

Another strategy is:

Allow the tree to grow,

then remove unnecessary branches afterward.

This is called:

## Post-pruning

The idea is:

1. Build a large tree.
2. Evaluate branches.
3. Remove branches that do not improve performance.

Post-pruning is more flexible but computationally more expensive.

---

# 3.63 Choosing Tree Parameters Experimentally

There is no universal value for:

```python
max_depth
```

or

```python
min_samples_leaf
```

The correct values depend on:

- dataset size,
- number of features,
- noise level,
- scientific problem.

For example:

A DFT dataset with thousands of calculated structures may support a deeper tree.

A small experimental dataset with only 100 alloys may require a much simpler tree.

---

# 3.64 Hyperparameter Tuning

Finding the best tree parameters is called:

## Hyperparameter tuning

Instead of manually guessing values,

we test many combinations.

Example:

```python
max_depth = [
3,
5,
10,
20
]
```

The model is trained multiple times.

The best performing configuration is selected.

Later chapters will discuss advanced tuning methods such as:

- Grid Search,
- Random Search,
- Bayesian Optimization.

---

# 3.65 Cross-Validation for Tree Selection

A single train-test split may give a misleading result.

Therefore,

we often use cross-validation.

Example:

```
Dataset

↓

Fold 1

Train / Validate


Fold 2

Train / Validate


Fold 3

Train / Validate

```

The average performance provides a more reliable estimate.

---

# 3.66 Overfitting in Materials Machine Learning

Materials datasets have special challenges.

Many datasets contain:

- limited experimental samples,
- measurement noise,
- correlated descriptors,
- computational errors.

Therefore,

overfitting is a major concern.

For example,

a model trained on 50 alloy samples may memorize specific compositions rather than learning general alloy behavior.

A model trained on thousands of DFT calculations may tolerate greater complexity.

The amount and quality of data determine how complex the model should be.

---

# 3.67 Practical Example

A researcher trains a Decision Tree to predict band gap.

Model A:

```
max_depth = 30
```

Results:

```
Training R² = 0.99

Testing R² = 0.40
```

The model memorized the training data.

---

Model B:

```
max_depth = 5
```

Results:

```
Training R² = 0.85

Testing R² = 0.82
```

Although Model B has a slightly worse training score,

it performs much better on new materials.

Therefore,

Model B is scientifically more useful.

---

# 3.68 Important Principle

The goal of machine learning is not:

> Create the model that remembers the dataset best.

The goal is:

> Create the model that predicts unseen data accurately.

This difference separates a scientific machine learning workflow from simple curve fitting.

---

# 3.69 Looking Ahead

Now that we understand how a single Decision Tree learns and how to control its complexity,

the next step is to understand how multiple trees can work together.

A single tree can be unstable.

But combining hundreds or thousands of trees can create a much more powerful and reliable model.

The next section introduces **Random Forests**, where many Decision Trees cooperate to reduce variance and improve prediction accuracy.

This concept is one of the most important foundations before moving toward XGBoost and modern materials machine learning workflows.


# 3.70 Introduction to Random Forest

A single Decision Tree is a powerful machine learning model.

It can capture nonlinear relationships, handle complex interactions between variables, and produce interpretable rules.

However, a single tree has a serious weakness:

**it is unstable.**

Small changes in the training data can produce a completely different tree.

This instability makes individual Decision Trees less reliable when the dataset contains noise.

The solution is to combine many Decision Trees together.

This idea leads to one of the most important machine learning algorithms:

# Random Forest

---

# 3.71 The Main Idea Behind Random Forest

The basic idea of Random Forest is simple:

> Instead of trusting one Decision Tree, create many different Decision Trees and combine their predictions.

A single tree may make mistakes.

However,

many trees making independent decisions can produce a much more reliable prediction.

The process is similar to human decision-making.

Suppose you want to choose the best material for a spacecraft component.

Instead of asking only one engineer,

you ask:

- a metallurgist,
- a mechanical engineer,
- a computational materials scientist,
- a corrosion expert.

Each person provides an opinion.

The final decision is made by combining all opinions.

Random Forest works in a similar way.

---

# 3.72 From One Tree to Many Trees

A single Decision Tree:

```
Training Data

       |

       |

 Decision Tree

       |

 Prediction
```

Random Forest:

```
              Training Data

                    |

        -----------------------

        |          |          |

     Tree 1     Tree 2     Tree 3

        |          |          |

        -----------------------

                    |

          Combined Prediction

```

Each tree learns slightly different patterns.

Their predictions are combined to create the final answer.

---

# 3.73 Why Is It Called "Forest"?

A collection of trees is called a forest.

Therefore:

```
Many Decision Trees

        =

Random Forest
```

The word "Random" comes from the randomness introduced during training.

Randomness is not a weakness.

It is the mechanism that allows different trees to learn different perspectives of the data.

---

# 3.74 The Two Sources of Randomness

Random Forest introduces randomness in two main ways.

## 1. Random Sampling of Data

Each tree receives a slightly different training dataset.

This process is called:

## Bootstrap Sampling

---

## 2. Random Selection of Features

Each tree considers only a random subset of features during splitting.

This prevents every tree from becoming identical.

---

# 3.75 Bootstrap Sampling

Suppose our dataset contains 1000 materials.

A Random Forest does not give all 1000 samples to every tree.

Instead,

each tree receives a random sample.

Example:

```
Tree 1:

Samples:

1,5,7,20,45...


Tree 2:

Samples:

2,6,18,50...


Tree 3:

Samples:

4,9,25,80...

```

Because each tree sees different data,

each tree learns a slightly different model.

---

# 3.76 Sampling With Replacement

Bootstrap sampling is performed:

**with replacement.**

This means after selecting a sample,

it is returned to the dataset and can be selected again.

Example:

Original dataset:

```
A B C D E
```

Possible bootstrap sample:

```
A C C D E
```

Notice that:

- C appears twice,
- B is missing.

This creates diversity between trees.

---

# 3.77 Why Random Sampling Helps

Consider three Decision Trees.

Without randomness:

```
Tree 1 = Tree 2 = Tree 3

```

All trees make the same mistakes.

Combining them provides little benefit.

With randomness:

```
Tree 1 learns pattern A

Tree 2 learns pattern B

Tree 3 learns pattern C

```

The errors become less correlated.

When combined,

the overall prediction becomes more accurate.

---

# 3.78 Feature Randomness

The second source of randomness occurs during splitting.

Suppose a materials dataset contains:

```
Density

Atomic Radius

Electronegativity

Band Gap

Formation Energy

Coordination Number

```

A normal Decision Tree considers every feature.

A Random Forest tree may only consider:

```
Density

Atomic Radius

Band Gap

```

at a particular split.

Another tree may consider:

```
Electronegativity

Formation Energy

Coordination Number

```

This creates diverse trees.

---

# 3.79 Prediction Process in Random Forest Regression

For regression problems,

each tree produces a numerical prediction.

Example:

A new alloy is given to the model.

Tree predictions:

```
Tree 1:

280 HV


Tree 2:

290 HV


Tree 3:

275 HV


Tree 4:

285 HV

```

The Random Forest averages all predictions.

$$
Prediction =
\frac{
280+290+275+285
}{4}
$$

$$
Prediction=282.5
$$

Final prediction:

```
282.5 HV
```

---

# 3.80 Prediction Process in Classification

For classification,

the trees vote.

Example:

Predicting whether a material is magnetic.

```
Tree 1:

Magnetic


Tree 2:

Non-magnetic


Tree 3:

Magnetic


Tree 4:

Magnetic

```

The majority wins:

```
Magnetic
```

---

# 3.81 Why Random Forest Reduces Overfitting

A single Decision Tree has high variance.

Meaning:

small changes in data create large changes in predictions.

Random Forest reduces this variance by averaging many trees.

This concept is called:

## Bagging

which stands for:

## Bootstrap Aggregating

The formula for averaging predictions is:

$$
\hat{y}
=
\frac{1}{N}
\sum_{i=1}^{N}
T_i(x)
$$

where:

- $(N)$ = number of trees,
- $(T_i(x))$ = prediction from tree $(i)$.

---

# 3.82 Bias and Variance

To understand why Random Forest works,

we need to understand two sources of model error.

## Bias

Bias is the error caused by overly simple assumptions.

Example:

A straight line used for a highly nonlinear problem.

High bias:

```
Underfitting
```

---

## Variance

Variance is sensitivity to the training data.

Example:

A very deep Decision Tree.

High variance:

```
Overfitting
```

---

# 3.83 Decision Tree vs Random Forest

| Decision Tree | Random Forest |
|-|-|
| One tree | Many trees |
| High variance | Lower variance |
| Easily overfits | More resistant to overfitting |
| Easy interpretation | Less interpretable |
| Fast training | More computationally expensive |
| Simple | More powerful |

---

# 3.84 Random Forest in Materials Science

Random Forest has become extremely popular in materials informatics.

Applications include:

- predicting formation energies,
- estimating band gaps,
- discovering stable compounds,
- predicting mechanical properties,
- analyzing battery materials.

A typical workflow:

```
Crystal Structure

↓

Pymatgen Descriptors

↓

Feature Matrix

↓

Random Forest

↓

Predicted Property

```

---

# 3.85 Example with Pymatgen

Suppose Pymatgen extracts:

```text
Atomic Volume

Average Electronegativity

Coordination Number

Density

Number of Elements
```

These become features:

```python
X
```

The target:

```python
y
```

could be:

```
Formation Energy
```

Random Forest learns relationships such as:

```
High electronegativity difference

+

Specific coordination environment

↓

Lower formation energy
```

The researcher does not manually define these rules.

The algorithm discovers them.

---

# 3.86 Random Forest Python Implementation Preview

Scikit-learn provides:

```python
RandomForestRegressor
```

Import:

```python
from sklearn.ensemble import RandomForestRegressor
```

Create model:

```python
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Train:

```python
model.fit(
    X_train,
    y_train
)
```

Predict:

```python
prediction = model.predict(
    X_test
)
```

We will study every parameter and every line of code in detail later.

---

# 3.87 Why Learn Random Forest Before XGBoost?

Random Forest and XGBoost are both ensemble methods.

However, they work differently.

Random Forest:

```
Many trees independently

↓

Average predictions

```

XGBoost:

```
Trees built sequentially

↓

Each tree corrects previous errors

```

Understanding Random Forest makes the transition to boosting algorithms much easier.

---

# 3.88 Looking Ahead

In the next sections, we will study Random Forest in depth.

We will learn:

- bootstrap sampling mathematically,
- feature randomness,
- ensemble learning theory,
- important hyperparameters,
- feature importance,
- Python implementation line by line,
- application to Pymatgen-generated descriptors,
- comparison between Decision Tree and Random Forest.

After mastering Random Forest, we will move toward **Gradient Boosting and XGBoost**, which are among the most powerful algorithms currently used in materials machine learning.


# 3.89 Mathematical Foundation of Random Forest

Random Forest may look like a simple idea:

"Train many Decision Trees and average their predictions."

However, the reason this method works so well is based on important mathematical concepts from statistics and machine learning.

To understand Random Forest deeply, we need to understand:

- ensemble learning,
- bootstrap sampling,
- variance reduction,
- correlation between models,
- bias-variance tradeoff.

These concepts are not only important for Random Forest.

They are also the foundation of modern algorithms such as:

- Gradient Boosting,
- XGBoost,
- LightGBM,
- CatBoost.

---

# 3.90 Ensemble Learning

The word **ensemble** means a group working together.

In machine learning,

ensemble learning means:

> Combining multiple models to create a stronger model.

The basic assumption is:

A group of models can often make better predictions than a single model.

This idea is similar to scientific research.

If one experiment gives an uncertain result,

researchers often repeat the experiment multiple times and analyze the combined evidence.

Machine learning ensembles follow a similar philosophy.

---

# 3.91 Why Combining Models Works

Suppose we have one Decision Tree.

Its prediction error comes from:

- incorrect splits,
- noisy data,
- random variations.

Another tree trained differently may make different mistakes.

If the errors are not identical,

combining the predictions can reduce the total error.

The important idea is:

> Errors that are different can cancel each other.

---

# 3.92 Mathematical View of Averaging

Suppose we have $(N)$ trees.

Each tree produces a prediction:

$$
T_1(x),T_2(x),...,T_N(x)
$$

The Random Forest prediction is:

$$
\hat{y}
=
\frac{1}{N}
\sum_{i=1}^{N}T_i(x)
$$

This is simply the average prediction.

However,

the power comes from reducing variance.

---

# 3.93 Understanding Variance

Variance describes how much a model changes when the training data changes.

Consider a Decision Tree trained on two slightly different datasets.

Dataset A:

```
Tree prediction:

250 HV
```

Dataset B:

```
Tree prediction:

320 HV
```

The model changes significantly.

This means the tree has high variance.

---

# 3.94 Why Single Decision Trees Have High Variance

Decision Trees make decisions using hierarchical splits.

Example:

```
Carbon > 0.35

        |

Grain Size < 15

        |

Prediction
```

A small change in the data may change the first split.

Once the first split changes,

all later branches may also change.

Therefore,

small data changes create large model changes.

---

# 3.95 Variance Reduction Through Averaging

Now imagine many trees.

Tree predictions:

$$
250,260,255,270,245
$$

Average:

$$
256
$$

The average prediction is usually more stable than any individual tree.

This is the mathematical reason Random Forest performs better.

---

# 3.96 The Variance Formula

For independent models,

the variance of their average is:

$$
Var(\bar{X})
=
\frac{\sigma^2}{N}
$$

where:

- $(\sigma^2)$ is the variance of individual models,
- $(N)$ is the number of models.

This means:

More trees generally reduce variance.

---

# 3.97 But Trees Are Not Completely Independent

In reality,

Random Forest trees are not completely independent.

All trees are trained from the same original dataset.

Therefore,

their errors are often correlated.

The variance becomes:

$$
Variance
=
\rho\sigma^2
+
\frac{(1-\rho)\sigma^2}{N}
$$

where:

- $(\rho)$ = correlation between trees,
- $(\sigma^2)$ = variance of individual trees,
- $(N)$ = number of trees.

---

# 3.98 Meaning of the Correlation Term

This equation explains why Random Forest needs randomness.

If all trees are identical:

$$
\rho=1
$$

Then:

$$
Variance=\sigma^2
$$

No improvement occurs.

The forest behaves like one tree.

---

If trees are completely independent:

$$
\rho=0
$$

Variance becomes:

$$
\frac{\sigma^2}{N}
$$

The improvement is maximum.

Therefore,

Random Forest tries to create trees that are:

- accurate,
- but different from each other.

---

# 3.99 How Random Forest Creates Diversity

Random Forest uses two mechanisms.

## Bootstrap Sampling

Different trees receive different training samples.

This changes the learned patterns.

---

## Random Feature Selection

Different trees consider different features.

This prevents all trees from choosing the same splits.

Together,

these methods reduce correlation between trees.

---

# 3.100 Bootstrap Sampling Mathematics

Suppose a dataset contains:

$$
n
$$

samples.

A bootstrap sample also contains:

$$
n
$$

samples.

However,

because sampling is performed with replacement,

some samples appear multiple times.

The probability that a particular sample is not selected is:

$$
\left(1-\frac{1}{n}\right)^n
$$

As $(n)$ becomes large:

$$
\left(1-\frac{1}{n}\right)^n
\approx e^{-1}
$$

which is approximately:

$$
0.368
$$

Therefore,

about 36.8% of samples are not included in each bootstrap sample.

These unused samples are called:

# Out-of-Bag Samples

---

# 3.101 Out-of-Bag Evaluation

An important advantage of Random Forest is:

it can evaluate itself using out-of-bag samples.

For each tree:

- some samples were used for training,
- some samples were left out.

The left-out samples can test the tree.

This provides an internal estimate of model performance.

In Scikit-learn:

```python
RandomForestRegressor(
    oob_score=True
)
```

enables this feature.

---

# 3.102 Bias-Variance Tradeoff

A perfect machine learning model must balance two errors:

- bias,
- variance.

The total prediction error can be thought of as:

$$
Error
=
Bias^2
+
Variance
+
Noise
$$

---

# 3.103 Bias in Random Forest

Random Forest usually has lower bias than simple models.

Because many trees can represent complex relationships.

However,

extremely restricted trees can increase bias.

Example:

```python
max_depth=2
```

The trees may become too simple.

---

# 3.104 Variance in Random Forest

Random Forest is mainly designed to reduce variance.

Compared with one Decision Tree:

- less sensitive to noise,
- more stable,
- better generalization.

This is why Random Forest is often a strong baseline model.

---

# 3.105 Number of Trees

The number of trees is controlled by:

```python
n_estimators
```

Example:

```python
RandomForestRegressor(
    n_estimators=500
)
```

More trees generally improve stability.

However,

after a certain point,

the improvement becomes very small.

---

# 3.106 Does More Trees Cause Overfitting?

An important question:

If we add thousands of trees,

will Random Forest overfit?

Usually,

No.

Increasing the number of trees mainly reduces variance.

Unlike increasing the depth of a single tree,

adding more trees usually does not make the model memorize the data.

The main cost is:

- increased memory,
- longer training time.

---

# 3.107 Random Forest and Materials Informatics

In materials science,

Random Forest is powerful because datasets often contain:

- many descriptors,
- nonlinear relationships,
- limited experimental samples.

Example:

Features:

```
Atomic radius

Electronegativity

Density

Crystal volume

Coordination number

Formation energy from DFT
```

Target:

```
Mechanical strength
```

A Random Forest can discover complex relationships between these descriptors.

---

# 3.108 Connection with Pymatgen Workflow

A complete materials machine learning workflow may look like:

```
Crystal Structure

        |

      Pymatgen

        |

Structural Descriptors

        |

Random Forest Model

        |

Predicted Property

```

For example:

Pymatgen extracts:

```
Average atomic radius

Volume

Density

Element fractions

```

Random Forest predicts:

```
Formation Energy
```

---

# 3.109 Connection with Quantum ESPRESSO

Quantum ESPRESSO can provide high-quality labels:

Examples:

- formation energy,
- band gap,
- magnetic moment,
- elastic constants.

The workflow:

```
Quantum ESPRESSO

↓

Generate Accurate Data

↓

Pymatgen Extracts Features

↓

Random Forest Learns Relationship

↓

Predict New Materials

```

This approach allows researchers to screen thousands of possible materials without performing expensive calculations for every candidate.

---

# 3.110 Looking Ahead

We now understand why Random Forest works mathematically.

The next step is to study the practical implementation.

We will learn:

- every important Random Forest parameter,
- Python code line by line,
- feature importance analysis,
- interpreting Random Forest predictions,
- applying Random Forest to real materials datasets,
- comparing Decision Tree and Random Forest performance.

After mastering Random Forest,

we will move to the next major breakthrough in machine learning:

**Gradient Boosting**, where trees are built sequentially to correct previous mistakes.


# 3.111 Random Forest Python Implementation

So far, we have studied the theory behind Random Forest.

We understand:

- why multiple trees are combined,
- how bootstrap sampling works,
- why randomness improves performance,
- how averaging reduces variance.

Now we move from theory to implementation.

The goal is not simply to copy Python commands.

The goal is to understand:

- what every line does,
- why that line is required,
- how the model works internally,
- how to modify the code for materials science problems.

---

# 3.112 Importing Required Libraries

For a Random Forest regression project, we need several Python libraries.

```python
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

# 3.113 Understanding Each Import

## Import Pandas

```python
import pandas as pd
```

Pandas is used for handling tabular datasets.

Materials datasets are usually stored as:

- CSV files,
- Excel files,
- database tables.

Pandas converts these files into DataFrames.

Example:

```
Material ID | Density | Band Gap | Formation Energy
----------------------------------------------------
1           | 5.2     | 2.1      | -3.5
2           | 6.1     | 1.8      | -2.9
```

A DataFrame allows us to manipulate this information easily.

---

## Import NumPy

```python
import numpy as np
```

NumPy provides numerical operations.

We will use it for mathematical calculations such as:

- square roots,
- arrays,
- numerical transformations.

---

## Import train_test_split

```python
from sklearn.model_selection import train_test_split
```

Machine learning requires separating data into:

- training data,
- testing data.

This function performs that division.

---

## Import Random Forest Model

```python
from sklearn.ensemble import RandomForestRegressor
```

This imports the Random Forest algorithm designed for regression problems.

Remember:

Regression predicts numerical values.

Examples:

- band gap,
- hardness,
- formation energy,
- thermal conductivity.

---

## Import Evaluation Metrics

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

These functions measure model performance.

They tell us:

- how large the prediction errors are,
- how well the model explains the data.

---

# 3.114 Loading the Dataset

Suppose we have a materials dataset:

```
materials.csv
```

containing:

| Feature | Description |
|-|-|
|Density|Material density|
|AtomicRadius|Average atomic radius|
|Electronegativity|Chemical descriptor|
|BandGap|Target property|

Our goal:

Predict band gap.

---

Python code:

```python
df = pd.read_csv(
    "materials.csv"
)
```

---

# 3.115 Understanding This Code

## `pd.read_csv()`

This function reads a CSV file.

CSV means:

Comma-Separated Values.

Example file:

```
Density,AtomicRadius,Electronegativity,BandGap

5.2,1.4,2.1,3.2

6.0,1.6,1.8,2.4

```

Pandas converts this into a DataFrame.

---

## Variable `df`

The dataset is stored in:

```python
df
```

The name means:

DataFrame.

You can choose another name, but `df` is a common convention.

---

# 3.116 Inspecting the Dataset

Before training any model,

always inspect the data.

```python
print(df.head())
```

---

## Explanation

`head()` displays the first five rows.

Example output:

```
Density  AtomicRadius  Electronegativity  BandGap

5.2      1.4           2.1                3.2

6.0      1.6           1.8                2.4
```

This helps confirm that the data were loaded correctly.

---

# 3.117 Checking Dataset Information

```python
df.info()
```

This displays:

- number of rows,
- number of columns,
- data types,
- missing values.

Example:

```
Column                 Non-Null Count

Density                500

AtomicRadius           500

BandGap                500
```

---

# 3.118 Statistical Summary

```python
df.describe()
```

Provides:

- mean,
- standard deviation,
- minimum,
- maximum.

Example:

```
Density

Mean = 5.8

Std = 1.2

Maximum = 9.5
```

This helps detect unusual values.

---

# 3.119 Separating Features and Target

Now we separate:

Input variables:

$$
X
$$

and target:

$$
y
$$

Python:

```python
X = df[
    [
        "Density",
        "AtomicRadius",
        "Electronegativity"
    ]
]

y = df["BandGap"]
```

---

# 3.120 Understanding Feature Matrix

The variable:

```python
X
```

contains the information used by the model.

Example:

```
Density   Radius   Electronegativity

5.2       1.4      2.1

6.0       1.6      1.8
```

Each row represents one material.

Each column represents one descriptor.

---

# 3.121 Understanding Target Variable

The variable:

```python
y
```

contains the value we want to predict.

Example:

```
BandGap

3.2

2.4

```

The Random Forest learns:

$$
X \rightarrow y
$$

Meaning:

Descriptors → Material Property

---

# 3.122 Splitting Training and Testing Data

Code:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 3.123 Explanation of train_test_split()

This function divides the dataset.

Here:

```python
test_size=0.2
```

means:

20% of data goes to testing.

80% remains for training.

Example:

Dataset:

```
1000 materials
```

Training:

```
800 materials
```

Testing:

```
200 materials
```

---

# 3.124 Why Use random_state?

```python
random_state=42
```

controls randomness.

Without it,

every execution may create a different split.

Example:

First run:

```
Training samples:

1,2,3,4...
```

Second run:

```
Training samples:

5,8,10...
```

Using a fixed value ensures reproducibility.

---

# 3.125 Creating the Random Forest Model

Now we create the model.

```python
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

---

# 3.126 Understanding Parameters

## n_estimators

```python
n_estimators=100
```

means:

Create 100 Decision Trees.

The forest contains:

```
Tree 1

Tree 2

Tree 3

...

Tree 100
```

Their predictions are averaged.

---

## random_state

```python
random_state=42
```

ensures reproducible randomness.

The bootstrap samples and feature selection will be generated consistently.

---

# 3.127 Training the Model

Code:

```python
model.fit(
    X_train,
    y_train
)
```

This begins the learning process.

Internally:

The algorithm:

1. Creates bootstrap datasets.
2. Builds many Decision Trees.
3. Searches for optimal splits.
4. Stores every learned tree.

After this step,

the model has learned the relationship between descriptors and the material property.

---

# 3.128 Making Predictions

Code:

```python
predictions = model.predict(
    X_test
)
```

For each test material:

1. Send the sample through every tree.
2. Collect all tree predictions.
3. Average the predictions.

Example:

```
Tree 1:

2.4 eV


Tree 2:

2.6 eV


Tree 3:

2.5 eV


Final:

2.5 eV
```

---

# 3.129 Evaluating the Model

Calculate errors.

```python
mae = mean_absolute_error(
    y_test,
    predictions
)
```

MAE tells us the average absolute difference between:

- actual values,
- predicted values.

---

MSE:

```python
mse = mean_squared_error(
    y_test,
    predictions
)
```

MSE penalizes large errors more strongly.

---

RMSE:

```python
rmse = np.sqrt(mse)
```

RMSE returns the error in the original unit.

Example:

If predicting band gap:

```
RMSE = 0.15 eV
```

---

R²:

```python
r2 = r2_score(
    y_test,
    predictions
)
```

Measures how much variation in the target is explained by the model.

---

# 3.130 Displaying Results

```python
print("MAE:", mae)

print("RMSE:", rmse)

print("R2:", r2)
```

Example:

```
MAE:

0.12


RMSE:

0.18


R2:

0.91
```

---

# 3.131 Complete Random Forest Workflow

The complete workflow is:

```
Load Data

↓

Inspect Dataset

↓

Clean Data

↓

Select Features

↓

Split Dataset

↓

Create Random Forest

↓

Train Model

↓

Predict

↓

Evaluate

↓

Interpret

```

This workflow is identical to professional materials machine learning projects.

---

# 3.132 Connection with Pymatgen Features

In a real materials informatics project,

the feature matrix will often not come directly from a CSV.

Instead:

```
CIF Structure

↓

Pymatgen

↓

Descriptors

↓

X Matrix

↓

Random Forest

```

Example descriptors:

```text
Atomic Fraction

Average Atomic Radius

Density

Volume

Coordination Number

```

The Random Forest does not care where the features came from.

They can come from:

- experiments,
- Pymatgen,
- Quantum ESPRESSO,
- Materials Project databases.

The learning process remains the same.

---

# 3.133 Looking Ahead

In the next section,

we will study the most important part of interpreting Random Forest models:

**Feature Importance.**

A major advantage of tree-based models is that they can tell us which descriptors influence predictions the most.

For materials science,

this is extremely valuable because we do not only want accurate predictions.

We also want scientific understanding.

We will learn how Random Forest identifies important physical descriptors and how this information can guide materials discovery.


# 3.134 Random Forest Feature Importance

One of the biggest advantages of tree-based machine learning models is that they can provide information about **which features are most important for prediction**.

In materials science, this ability is extremely valuable.

A machine learning model that only predicts a property is useful.

However, a model that also helps us understand **why a material behaves in a certain way** is much more scientifically powerful.

For example:

A model predicting battery performance is useful.

But discovering that:

- ionic radius,
- crystal volume,
- oxidation state,

are the most influential descriptors can provide new scientific insight.

This is where feature importance becomes important.

---

# 3.135 What Is Feature Importance?

Feature importance measures how much each input feature contributes to the decisions made by the model.

Suppose we train a Random Forest to predict formation energy.

Our features are:

```
Atomic Radius

Electronegativity

Density

Coordination Number

Volume

```

After training, the model may report:

| Feature | Importance |
|-|-:|
|Electronegativity|0.35|
|Atomic Radius|0.25|
|Coordination Number|0.20|
|Density|0.15|
|Volume|0.05|

This means:

Electronegativity contributed the most to the model's predictions.

---

# 3.136 Why Feature Importance Matters in Materials Science

Machine learning in materials science is not only about prediction.

Researchers also want:

- understanding,
- interpretation,
- physical insight.

Suppose a Random Forest predicts alloy strength.

The model finds:

```
Grain Size Importance = 0.45

Carbon Content Importance = 0.30

Density Importance = 0.10
```

This suggests grain size strongly influences the prediction.

A researcher may then investigate grain refinement mechanisms experimentally.

Therefore,

machine learning becomes a tool for scientific discovery.

---

# 3.137 How Random Forest Calculates Importance

Random Forest feature importance is usually calculated using:

## Mean Decrease in Impurity

(MDI)

The idea is:

A feature is important if using it creates large improvements in the tree.

---

Remember:

A Decision Tree chooses splits that reduce impurity.

For regression:

impurity is usually measured using:

$$
MSE
$$

If splitting on a feature greatly reduces MSE,

that feature receives high importance.

---

# 3.138 Mathematical Idea Behind MDI

For a feature $(j)$,

importance is calculated from the reduction in impurity:

$$
Importance_j
=
\sum
Weighted\ impurity\ reduction
$$

The reduction is calculated every time that feature is used in a split.

The contributions are added across:

- all trees,
- all branches.

Finally,

the values are normalized.

---

# 3.139 Example of Importance Calculation

Imagine a single tree.

Root split:

```
Electronegativity

MSE reduction:

50
```

Second split:

```
Density

MSE reduction:

10
```

Third split:

```
Atomic Radius

MSE reduction:

20
```

The contributions are:

```
Electronegativity = 50

Density = 10

Atomic Radius = 20
```

After normalization:

```
Electronegativity = Highest importance
```

The Random Forest averages this information from all trees.

---

# 3.140 Extracting Feature Importance in Python

After training a Random Forest:

```python
importance = model.feature_importances_
```

This returns an array.

Example:

```text
[0.35,0.25,0.20,0.15,0.05]
```

Each value corresponds to one feature.

---

# 3.141 Connecting Importance Values with Feature Names

The numbers alone are not useful.

We need to combine them with feature names.

Example:

```python
importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

print(importance_df)
```

---

# 3.142 Understanding This Code

## Creating DataFrame

```python
pd.DataFrame()
```

creates a table.

---

Inside:

```python
"Feature": X.columns
```

extracts the names of input variables.

Example:

```
Density

Atomic Radius

Electronegativity

```

---

```python
"Importance": model.feature_importances_
```

extracts the importance values from the trained model.

---

The result:

|Feature|Importance|
|-|-:|
|Density|0.15|
|Atomic Radius|0.25|
|Electronegativity|0.35|

---

# 3.143 Sorting Features by Importance

Usually,

we want the most important features first.

Code:

```python
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
```

---

Explanation:

```python
by="Importance"
```

means sort using the importance column.

---

```python
ascending=False
```

means highest values appear first.

Result:

|Feature|Importance|
|-|-:|
|Electronegativity|0.35|
|Atomic Radius|0.25|
|Density|0.15|

---

# 3.144 Visualizing Feature Importance

A bar chart is often used.

Example:

```python
import matplotlib.pyplot as plt


plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xticks(
    rotation=90
)

plt.show()
```

---

# 3.145 Explanation of Visualization Code

## Import Matplotlib

```python
import matplotlib.pyplot as plt
```

Used for creating graphs.

---

## Create Bar Plot

```python
plt.bar()
```

Creates bars representing importance values.

---

## Rotate Labels

```python
plt.xticks(rotation=90)
```

Rotates feature names so they remain readable.

---

## Display Graph

```python
plt.show()
```

Shows the final figure.

---

# 3.146 Example in Materials Discovery

Suppose we train a Random Forest to predict band gap.

Features:

```
Average Atomic Number

Atomic Radius

Electronegativity

Volume

Density

```

The model gives:

```
Electronegativity

0.42


Atomic Radius

0.25


Volume

0.15


Density

0.10


Atomic Number

0.08

```

Interpretation:

Chemical bonding information appears to dominate the prediction.

This agrees with physical intuition because electronic structure strongly influences band gap.

---

# 3.147 Limitations of Feature Importance

Although feature importance is useful,

it must be interpreted carefully.

It does not always mean:

"Feature causes the property."

It only means:

"The model uses this feature frequently to make accurate predictions."

Correlation does not automatically imply causation.

---

# 3.148 Bias Toward Certain Features

Mean Decrease in Impurity can sometimes favor:

- continuous variables,
- features with many possible split points.

For example:

A numerical descriptor with hundreds of possible values may appear more important simply because the tree has more opportunities to split on it.

Therefore,

feature importance should not be interpreted blindly.

---

# 3.149 Correlated Features Problem

Materials datasets often contain correlated descriptors.

Example:

```
Atomic Volume

Lattice Constant

Density

```

These may contain overlapping information.

The model may assign:

```
Feature A:

0.40 importance


Feature B:

0.10 importance
```

even though both describe the same physical phenomenon.

The lower value does not necessarily mean the feature is unimportant.

---

# 3.150 Permutation Importance

A more reliable method is:

## Permutation Importance

The idea:

1. Measure normal model performance.
2. Randomly shuffle one feature.
3. Measure performance again.
4. Calculate performance decrease.

If shuffling a feature destroys performance,

that feature is important.

---

# 3.151 Example

Original model:

```
R² = 0.90
```

Shuffle electronegativity:

```
R² = 0.55
```

Performance decreases greatly.

Therefore:

```
Electronegativity is important.
```

---

Shuffle density:

```
R² = 0.88
```

Small change.

Therefore:

```
Density contributes less.
```

---

# 3.152 Permutation Importance in Python

Import:

```python
from sklearn.inspection import permutation_importance
```

Calculate:

```python
result = permutation_importance(
    model,
    X_test,
    y_test
)
```

The output contains importance values based on prediction performance.

---

# 3.153 Random Forest and Explainable AI

Feature importance is the beginning of a larger field:

## Explainable Artificial Intelligence (XAI)

Modern materials machine learning increasingly focuses on understanding models.

Important methods include:

- permutation importance,
- SHAP values,
- partial dependence plots.

These methods help researchers answer:

"Why did the model make this prediction?"

---

# 3.154 Connection with Pymatgen Descriptors

When using Pymatgen,

you may generate hundreds of descriptors.

Examples:

```
Average electronegativity

Atomic volume

Ionic radius

Coordination number

Element fractions

```

A Random Forest can identify which descriptors matter most.

This helps with:

- removing unnecessary features,
- improving model efficiency,
- understanding physical mechanisms.

---

# 3.155 Connection with Quantum ESPRESSO

Quantum ESPRESSO can generate accurate targets:

```
Formation Energy

Band Gap

Elastic Modulus

Magnetic Moment
```

After training:

Random Forest feature importance may reveal:

```
Which structural descriptors control the DFT property.
```

This creates a bridge:

```
Quantum Physics

↓

Machine Learning

↓

Scientific Understanding

```

---

# 3.156 Looking Ahead

Feature importance gives us insight into what the model learns.

However, Random Forest still treats each tree independently.

The next major improvement is to build trees sequentially, where each new tree focuses on correcting the mistakes of previous trees.

This idea is called:

**Boosting.**

The next chapter will introduce Gradient Boosting, which leads directly to powerful algorithms such as:

- XGBoost,
- LightGBM,
- CatBoost.

These algorithms dominate many modern materials machine learning competitions and research applications.
