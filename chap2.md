# Chapter 2
# Linear Regression

> *"The simplest machine learning algorithm often becomes the first model a researcher trusts when exploring a new materials dataset."*

In Chapter 1, we learned the complete machine learning workflow and built our first predictive model.

Now we begin studying the first machine learning algorithm in depth:

**Linear Regression.**

Although Linear Regression is mathematically simple, it remains one of the most widely used algorithms in computational materials science and materials informatics.

Researchers routinely use Linear Regression to

- estimate mechanical properties,
- predict elastic constants,
- model thermal conductivity,
- approximate formation energy,
- correlate descriptors with material properties,
- establish baseline models before using more advanced algorithms.

Even when sophisticated algorithms such as Random Forests, XGBoost, or Graph Neural Networks are ultimately used, researchers almost always begin with a simple baseline model like Linear Regression.

For this reason, mastering Linear Regression is an essential first step.

---

# 2.1 Linear Regression in the Materials Informatics Pipeline

Throughout the remainder of this book, we will use the following workflow repeatedly.

```text
Crystal Structure
        │
        ▼
Quantum ESPRESSO
(DFT Calculation)
        │
        ▼
Electronic & Structural Properties
        │
        ▼
Pymatgen
(Parse Calculation Results)
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning
        │
        ▼
Prediction of New Materials
```

Notice something important.

Machine learning is **not** the first step.

It is one component of a much larger scientific workflow.

The quality of the machine learning model depends heavily on the quality of the computational and experimental data that come before it.

---

# 2.2 Where Does Linear Regression Fit?

Suppose you perform DFT calculations on several steel alloys using Quantum ESPRESSO.

For each alloy you calculate

- lattice constant,
- unit cell volume,
- density,
- bulk modulus,
- total energy.

Using Pymatgen, you extract these quantities into a structured dataset.

The dataset may look like

| Density | Volume | Bulk Modulus | Hardness |
|---------:|-------:|-------------:|---------:|
| 7.82 | 11.3 | 162 | 241 |
| 7.91 | 11.1 | 169 | 258 |
| 7.65 | 11.7 | 151 | 226 |

Now machine learning begins.

Instead of running another expensive DFT calculation,

Linear Regression learns the relationship between

```text
Density

Volume

Bulk Modulus

↓

Hardness
```

Once trained,

the model can estimate the hardness of new materials in milliseconds.

---

# 2.3 Why This Matters

A single DFT calculation may require

- several minutes,
- several hours,
- or even days,

depending on

- system size,
- k-point mesh,
- exchange-correlation functional,
- computational hardware.

A trained Linear Regression model,

however,

typically makes predictions in less than a millisecond.

Machine learning does **not** replace density functional theory.

Instead,

it complements DFT by dramatically reducing the number of expensive calculations required during materials discovery.

This combination of

- first-principles calculations,
- descriptor engineering,
- machine learning,

forms the foundation of modern materials informatics.

---

# 2.4 Chapter Roadmap

Unlike most introductory machine learning books, this chapter does not stop after explaining the mathematics.

By the end of this chapter, you will be able to

✓ Understand the mathematical foundations of Linear Regression.

✓ Derive the regression equation.

✓ Interpret the slope and intercept physically.

✓ Understand the cost function.

✓ Learn how the algorithm minimizes prediction error.

✓ Build Linear Regression models in Python.

✓ Explain every line of Python code.

✓ Parse real computational materials data using Pymatgen.

✓ Build datasets suitable for machine learning.

✓ Train a Linear Regression model using descriptors extracted from DFT calculations.

✓ Evaluate model performance.

✓ Understand the strengths and limitations of Linear Regression for materials science.

---

# 2.5 A New Type of Example

From this chapter onward,

every major algorithm in this book will include three progressively more realistic examples.

### Example 1 — Learning Example

A very small dataset.

Purpose:

Understand the mathematics and the Python code.

---

### Example 2 — Materials Science Example

A manually prepared materials dataset.

Purpose:

Understand how machine learning is applied to engineering problems.

---

### Example 3 — Computational Materials Workflow

A complete workflow beginning with

- Quantum ESPRESSO calculations,
- Pymatgen feature extraction,
- machine learning model training,
- prediction of new materials.

Purpose:

Learn how machine learning is actually used in computational materials science research.

This three-level approach ensures that you understand not only the algorithm itself but also how to apply it to real scientific problems.


# 2.6 What Is Linear Regression?

Now that we understand where Linear Regression fits into the materials informatics workflow, we are ready to study the algorithm itself.

Linear Regression is one of the simplest supervised machine learning algorithms.

Its goal is to learn the relationship between one or more **input variables (features)** and a **continuous numerical output (target)**.

Once this relationship has been learned,

the model can predict the target for materials it has never seen before.

Unlike Decision Trees or Neural Networks,

Linear Regression assumes that the relationship between the inputs and the output is **approximately linear**.

This assumption makes the algorithm

- simple,
- fast,
- interpretable,
- and surprisingly effective for many engineering problems.

---

# 2.7 A Materials Science Problem

Suppose you have performed Quantum ESPRESSO calculations for several steel alloys.

After each calculation, you extract the density and hardness.

Your dataset looks like this.

| Alloy | Density (g/cm³) | Hardness (HV) |
|-------|----------------:|--------------:|
| A | 7.50 | 182 |
| B | 7.60 | 205 |
| C | 7.70 | 233 |
| D | 7.80 | 255 |
| E | 7.90 | 281 |
| F | 8.00 | 304 |

Suppose another researcher gives you a newly designed alloy.

Its density is

```text
7.75 g/cm³
```

However,

its hardness has not yet been measured experimentally.

Instead of performing expensive experiments,

we would like our machine learning model to estimate the hardness.

This is exactly what Linear Regression is designed to do.

---

# 2.8 Looking at the Data

Imagine plotting every alloy on a graph.

```text
Hardness (HV)

310 |                           ●
290 |
270 |                      ●
250 |
230 |                 ●
210 |            ●
190 |       ●
170 |
    +-----------------------------------------
      7.5  7.6  7.7  7.8  7.9  8.0

             Density (g/cm³)
```

Notice the overall trend.

As density increases,

hardness also tends to increase.

The points approximately follow a straight line.

Linear Regression tries to discover that line automatically.

---

# 2.9 Why Not Simply Connect Every Point?

A beginner might ask,

> "Why don't we simply connect all the experimental points?"

There are two reasons.

First,

real experimental data always contain

- measurement error,
- experimental uncertainty,
- instrument noise,
- natural material variability.

The points rarely lie perfectly on one line.

Second,

our goal is **prediction**.

We want to estimate properties for materials that are **not** in the dataset.

Drawing a smooth line through every point may fit the existing data,

but it may perform poorly on unseen materials.

Instead,

Linear Regression searches for the **single straight line that best represents the overall trend**.

---

# 2.10 The Regression Line

The line learned by the algorithm is called the **regression line** or the **line of best fit**.

It summarizes the overall relationship between the variables.

```text
Hardness

310 |                          ●
300 |                       /
290 |                    /
280 |                 ●
270 |               /
260 |            /
250 |         ●
240 |       /
230 |     /
220 |   ●
210 | /
200 |●
    +--------------------------------
      Density
```

Notice something important.

The line does **not** pass through every point.

Instead,

it balances the errors across all observations.

This idea is one of the central principles of Linear Regression.

Later,

we will learn exactly how the algorithm determines this line mathematically.

---

# 2.11 The Mathematical Equation

Every straight line can be written using a simple equation.

$$
y = mx + b
$$

This equation may already be familiar from school mathematics.

In Linear Regression,

the same equation appears,

but each symbol has a specific meaning.

| Symbol | Meaning |
|---------|---------|
| \(y\) | Predicted target |
| \(x\) | Feature |
| \(m\) | Slope |
| \(b\) | Intercept |

The machine learning algorithm automatically learns the values of

- the slope,
- the intercept,

from the training data.

We do **not** choose them ourselves.

---

# 2.12 Interpreting the Equation Physically

Suppose the trained model produces

$$
\text{Hardness}
=
450 \times \text{Density}
-
3190
$$

This equation has physical meaning.

The coefficient

```text
450
```

indicates how rapidly hardness changes as density changes.

The intercept

```text
-3190
```

is the value predicted when density equals zero.

Although a density of zero is physically impossible,

the intercept is still mathematically necessary because it determines where the regression line crosses the vertical axis.

Later in this chapter,

we will study the physical interpretation of coefficients in much greater detail.

---

# 2.13 How Does the Algorithm Learn This Equation?

Suppose the algorithm initially guesses

$$
y = 100x + 5
$$

The predictions are poor.

It then tries another equation.

$$
y = 250x - 900
$$

The predictions improve,

but they are still not optimal.

The algorithm continues adjusting the slope and intercept,

making thousands of mathematical comparisons,

until it finds the equation that minimizes prediction error.

This process is called **training**.

In later sections,

we will learn exactly how this optimization works.

---

# 2.14 Materials Informatics Workflow

Now let us connect Linear Regression with computational materials science.

Imagine you have completed several Quantum ESPRESSO calculations.

Your project directory contains

```text
Project/

├── Alloy_01/

├── Alloy_02/

├── Alloy_03/

├── Alloy_04/

└── Alloy_05/
```

Each directory contains

- input files,
- output files,
- structural information.

Machine learning cannot directly use these files.

Instead,

we first extract numerical information.

```text
Quantum ESPRESSO

↓

Output Files

↓

Pymatgen

↓

Numerical Descriptors

↓

Machine Learning Dataset

↓

Linear Regression
```

This conversion from scientific calculations to numerical features is one of the most important skills in materials informatics.

---

# 2.15 Preview: Parsing Data with Pymatgen

Later in this chapter,

we will build a complete dataset automatically.

For example,

Pymatgen allows us to load a crystal structure.

```python
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
```

From this structure,

we can extract useful descriptors such as

```python
structure.volume

structure.density

structure.composition

structure.lattice.a
```

These numerical quantities eventually become the input features for Linear Regression.

Do not worry if this code is not yet completely clear.

When we reach the research workflow section of this chapter,

every line will be explained in detail.

---

# 2.16 What Makes Linear Regression Useful?

Linear Regression has remained popular for decades because it possesses several desirable properties.

### It is fast.

Training usually takes only fractions of a second for small and medium-sized datasets.

---

### It is interpretable.

Unlike many complex machine learning models,

we can understand exactly how each feature influences the prediction.

This is extremely valuable in scientific research.

---

### It requires relatively little data.

Many engineering projects contain only a few hundred experimental measurements.

Linear Regression often performs surprisingly well under these conditions.

---

### It provides a baseline.

Before trying sophisticated models like

- Random Forest,
- XGBoost,
- LightGBM,
- Graph Neural Networks,

researchers usually train a Linear Regression model first.

If a simple model already achieves high accuracy,

there may be little benefit in using a more complicated algorithm.

---

# 2.17 Research Workflow Preview

Every algorithm chapter in this book will conclude with a complete research workflow.

For Linear Regression,

the workflow will be

```text
Crystal Structure

↓

Quantum ESPRESSO

↓

Pymatgen

↓

Descriptor Generation

↓

Pandas DataFrame

↓

Feature Selection

↓

Linear Regression

↓

Model Evaluation

↓

Interpretation

↓

Comparison with DFT

↓

Scientific Conclusions
```

By the end of this chapter,

you will be able to perform every step in this workflow yourself.

Rather than merely learning another machine learning algorithm,

you will learn how Linear Regression is actually used in computational materials science research.


# 2.18 Simple Linear Regression

In the previous sections, we introduced the idea behind Linear Regression.

We learned that the algorithm tries to find a straight-line relationship between an input variable and an output variable.

Now we will study the first type of Linear Regression:

**Simple Linear Regression.**

The word **simple** means that the model uses only **one feature** to predict the target.

Mathematically,

there is only one input variable:

\[
x
\]

and one output variable:

\[
y
\]

The relationship is represented as:

\[
y = mx+b
\]

where

- \(m\) = slope of the line,
- \(b\) = intercept,
- \(x\) = input feature,
- \(y\) = predicted output.

---

# 2.19 Understanding the Components

Let us examine each part of the equation.

## The Feature (x)

The feature is the information provided to the model.

In materials science examples, the feature could be:

- density,
- atomic radius,
- lattice parameter,
- grain size,
- carbon concentration.

For example:

\[
x = \text{Carbon content}
\]

---

## The Target (y)

The target is the property we want to predict.

Examples:

- hardness,
- yield strength,
- bandgap,
- formation energy,
- thermal conductivity.

For example:

\[
y = \text{Hardness}
\]

---

## The Slope (m)

The slope represents how much the output changes when the input changes by one unit.

For example:

\[
m=50
\]

means:

for every increase of 1 unit in \(x\),

the predicted \(y\) increases by 50 units.

In materials science,

the slope can sometimes have physical interpretation.

For example,

if

\[
m=120
\]

for a model predicting hardness from carbon content,

it means increasing carbon content by 1% increases predicted hardness by approximately 120 HV.

---

## The Intercept (b)

The intercept is the predicted value of \(y\) when \(x=0\).

Mathematically:

\[
y=b
\]

when

\[
x=0
\]

In many physical systems,

the intercept may not have direct physical meaning.

For example,

a material with zero carbon content may not exist in the studied system.

However,

the intercept is necessary for positioning the regression line correctly.

---

# 2.20 Example: Predicting Hardness from Carbon Content

Suppose we have the following steel dataset.

| Carbon (%) | Hardness (HV) |
|-----------:|---------------:|
| 0.10 | 150 |
| 0.20 | 190 |
| 0.30 | 240 |
| 0.40 | 280 |
| 0.50 | 330 |

Here:

Feature:

\[
x=\text{Carbon percentage}
\]

Target:

\[
y=\text{Hardness}
\]

The machine learning model searches for the best equation:

\[
\text{Hardness}=m(\text{Carbon})+b
\]

---

# 2.21 Finding the Best Line

Many possible lines can be drawn through the data.

For example:

\[
y=200x+100
\]

or

\[
y=400x+120
\]

or

\[
y=350x+115
\]

Some lines will make better predictions than others.

The question becomes:

> How do we decide which line is the best?

The answer is:

**We calculate the prediction error.**

---

# 2.22 Residuals

The difference between the actual value and the predicted value is called the **residual**.

Mathematically:

\[
\text{Residual}=y_i-\hat{y_i}
\]

where:

- \(y_i\) = actual value,
- \(\hat{y_i}\) = predicted value.

Example:

Actual hardness:

\[
250
\]

Predicted hardness:

\[
240
\]

Residual:

\[
250-240=10
\]

The model underestimated the hardness by 10 HV.

---

# 2.23 Visual Meaning of Residuals

Imagine the regression line:

```text
Hardness

350 |              ●
    |
300 |          ●
    |        |
250 |     ●  |
    |     |  |
200 | ●---|---------------- Line
    |
150 |
    +----------------------------
          Carbon
```

The vertical distance between each data point and the regression line is the residual.

A perfect model would have all points exactly on the line.

However,

real scientific data always contain noise.

Therefore,

the goal is not zero error.

The goal is to minimize the total error.

---

# 2.24 Why Not Add All Residuals?

At first glance,

we might think:

"Why not simply add all residuals and choose the line with the smallest value?"

The problem is that positive and negative errors cancel each other.

Example:

Prediction errors:

\[
+10,-10,+5,-5
\]

Sum:

\[
10-10+5-5=0
\]

The total error appears to be zero,

but the model is clearly making mistakes.

Therefore,

we need a better method.

---

# 2.25 Least Squares Method

Linear Regression solves this problem using the **least squares method**.

Instead of adding errors directly,

we square them:

\[
(y_i-\hat{y_i})^2
\]

Then we add all squared errors:

\[
\sum_{i=1}^{n}(y_i-\hat{y_i})^2
\]

This quantity is called the:

**Sum of Squared Errors (SSE)**

The best regression line is the one that produces the smallest SSE.

---

# 2.26 Why Square the Errors?

Squaring has two important effects.

## 1. Negative values become positive

Example:

\[
(-5)^2=25
\]

\[
5^2=25
\]

Both contribute equally.

---

## 2. Large mistakes are punished more

Compare:

Error = 2

\[
2^2=4
\]

Error = 10

\[
10^2=100
\]

A large prediction mistake becomes much more important.

This encourages the model to avoid large errors.

---

# 2.27 The Cost Function

The average squared error is called the:

**Mean Squared Error (MSE)**

The equation is:

\[
MSE=
\frac{1}{n}
\sum_{i=1}^{n}(y_i-\hat{y_i})^2
\]

The goal of Linear Regression training is:

\[
\text{Minimize MSE}
\]

The algorithm searches for the values of

\[
m
\]

and

\[
b
\]

that give the lowest possible error.

---

# 2.28 Connection to Machine Learning Training

When we write:

```python
model.fit(X_train,y_train)
```

we are asking the computer:

> Find the values of slope and intercept that minimize the prediction error.

The computer performs the mathematical optimization automatically.

The result is a trained model.

---

# 2.29 Python Implementation: Simple Linear Regression

Now we will implement this using Python.

First,

we create a small materials science dataset.

```python
import pandas as pd

data = {
    "Carbon": [0.10,0.20,0.30,0.40,0.50],
    "Hardness": [150,190,240,280,330]
}

df = pd.DataFrame(data)

print(df)
```

Output:

```
   Carbon  Hardness
0    0.10       150
1    0.20       190
2    0.30       240
3    0.40       280
4    0.50       330
```

---

# 2.30 Code Explanation

```python
import pandas as pd
```

Imports the Pandas library.

Pandas allows us to store and manipulate tabular data.

---

```python
data = {
```

Creates a Python dictionary.

The dictionary contains our dataset.

---

```python
"Carbon": [...]
```

Creates the feature column.

---

```python
"Hardness": [...]
```

Creates the target column.

---

```python
df = pd.DataFrame(data)
```

Converts the dictionary into a DataFrame.

The DataFrame is the format commonly used for machine learning.

---

The next section will continue from here and train the first Linear Regression model, extract the learned slope and intercept, visualize the fitted line, and then move into the complete Pymatgen + Quantum ESPRESSO materials workflow.


# 2.31 Training Your First Linear Regression Model

In the previous section, we created a small materials science dataset containing

- Carbon percentage
- Hardness

Now we will train our first **Linear Regression model**.

Unlike Chapter 1, where we focused mainly on the machine learning workflow, this chapter focuses on understanding **exactly what the Linear Regression algorithm learns**.

By the end of this section, you will understand

- how to train the model,
- what happens internally,
- what the learned equation looks like,
- how to interpret the slope and intercept physically.

---

# 2.32 Importing the Linear Regression Class

First, we import the Linear Regression algorithm.

```python
from sklearn.linear_model import LinearRegression
```

---

## Understanding the Code

### `from`

The keyword `from` tells Python that we want to import something from a library.

---

### `sklearn`

This is the **Scikit-learn** library.

It contains dozens of machine learning algorithms.

Examples include

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- K-Nearest Neighbors

Throughout this book, Scikit-learn will be our primary library for classical machine learning.

---

### `linear_model`

Scikit-learn organizes algorithms into different modules.

The `linear_model` module contains algorithms based on linear equations.

Examples include

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Logistic Regression

---

### `LinearRegression`

This is the class that implements the Linear Regression algorithm.

Think of it as the blueprint for creating a regression model.

---

# 2.33 Preparing the Data

Next, we separate the dataset into features and targets.

```python
X = df[["Carbon"]]

y = df["Hardness"]
```

Notice something important.

`X` contains **double square brackets**.

```python
[["Carbon"]]
```

while `y` contains only one pair.

```python
["Hardness"]
```

This difference is intentional.

---

# 2.34 Why Double Square Brackets?

A machine learning model expects the features to be stored as a **matrix**.

Even if there is only one feature,

Scikit-learn still expects a two-dimensional structure.

Suppose our DataFrame contains

| Carbon |
|---------|
| 0.10 |
| 0.20 |
| 0.30 |

Using

```python
df["Carbon"]
```

produces

```text
0.10
0.20
0.30
```

This is called a **Series**.

Using

```python
df[["Carbon"]]
```

produces

| Carbon |
|---------|
| 0.10 |
| 0.20 |
| 0.30 |

This is a **DataFrame**, which is two-dimensional.

Scikit-learn expects the second form.

---

# 2.35 Creating the Model

Now we create the Linear Regression model.

```python
model = LinearRegression()
```

At this moment,

the model has not learned anything.

It does not know

- the slope,
- the intercept,
- or the relationship between carbon content and hardness.

It is simply an empty model waiting to be trained.

---

# 2.36 Training the Model

Training is performed using

```python
model.fit(X, y)
```

This is one of the most important commands in machine learning.

The method

```python
fit()
```

means

> Learn from the data.

During training,

the algorithm

1. examines every sample,
2. calculates many possible regression lines,
3. computes the prediction error,
4. finds the line with the smallest error.

After training,

the model has learned the mathematical relationship.

---

# 2.37 What Happens Internally?

When we execute

```python
model.fit(X, y)
```

the computer performs something similar to

```text
Try Line 1

↓

Calculate Error

↓

Try Line 2

↓

Calculate Error

↓

Try Line 3

↓

Calculate Error

↓

...

↓

Find Line with Smallest Error
```

The algorithm is not guessing randomly.

It uses mathematics to determine the best slope and intercept.

For simple Linear Regression,

this solution can even be computed directly using linear algebra.

Later in the chapter, we will derive this mathematically.

---

# 2.38 The Learned Equation

After training,

the model has discovered an equation of the form

\[
\text{Hardness}
=
m(\text{Carbon})+b
\]

Unlike before,

the values of

- \(m\),
- \(b\),

are no longer unknown.

The model has learned them from the data.

---

# 2.39 Viewing the Slope

Scikit-learn stores the learned slope inside the model.

```python
print(model.coef_)
```

Example output

```text
[450.]
```

The exact value depends on the training data.

---

## Understanding `coef_`

The attribute

```python
coef_
```

contains the coefficient of each feature.

Since our model has only one feature,

there is only one coefficient.

If we later build a model using

- Density
- Carbon
- Grain Size

then `coef_` will contain three numbers.

Each coefficient measures how strongly that feature influences the prediction.

---

# 2.40 Viewing the Intercept

The intercept is stored separately.

```python
print(model.intercept_)
```

Example output

```text
105
```

Again,

the exact value depends on the dataset.

---

# 2.41 Writing the Learned Equation

Suppose the model learned

```text
Slope = 450

Intercept = 105
```

The regression equation becomes

\[
\text{Hardness}
=
450\times
\text{Carbon}
+
105
\]

Now we can make predictions without using Python.

For example,

if

Carbon = 0.40%

then

\[
\text{Hardness}
=
450(0.40)+105
\]

\[
=
180+105
\]

\[
=
285
\]

The model predicts

```text
Hardness ≈ 285 HV
```

Notice that this prediction comes directly from the learned equation.

---

# 2.42 Making Predictions in Python

Instead of calculating manually,

we let Python perform the computation.

```python
new_material = [[0.45]]

predicted_hardness = model.predict(new_material)

print(predicted_hardness)
```

Example output

```text
[307.5]
```

The model predicts that a steel alloy containing **0.45% carbon** has an estimated hardness of approximately **308 HV**.

---

# 2.43 Understanding the Prediction Code

### Line 1

```python
new_material = [[0.45]]
```

Creates a new sample.

Notice the double square brackets.

The outer brackets represent the dataset.

The inner brackets represent one material.

---

### Line 3

```python
model.predict(new_material)
```

Uses the learned regression equation.

No additional learning occurs.

The model simply substitutes the new carbon value into the equation.

---

### Line 5

```python
print(predicted_hardness)
```

Displays the predicted hardness.

The result is returned as a NumPy array because Scikit-learn always assumes that predictions may contain multiple samples.

---

# 2.44 Complete Program

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Carbon": [0.10, 0.20, 0.30, 0.40, 0.50],
    "Hardness": [150, 190, 240, 280, 330]
}

df = pd.DataFrame(data)

X = df[["Carbon"]]
y = df["Hardness"]

model = LinearRegression()

model.fit(X, y)

print("Slope:", model.coef_)

print("Intercept:", model.intercept_)

new_material = [[0.45]]

prediction = model.predict(new_material)

print("Predicted Hardness:", prediction)
```

Although this program is short, it performs an entire machine learning task:

1. Creates a dataset.
2. Separates features and targets.
3. Builds a Linear Regression model.
4. Trains the model.
5. Learns the regression equation.
6. Predicts the hardness of a new material.

In the next section, we will visualize the regression line on a graph and learn how to interpret the quality of the fit before moving toward more realistic materials datasets generated using **Pymatgen** and **Quantum ESPRESSO**.

# 2.45 Visualizing the Regression Line

So far, we have trained our first Linear Regression model and used it to predict the hardness of a new material.

However, machine learning is not only about obtaining numerical predictions.

A good machine learning engineer should also **visualize** the model.

Visualization helps us answer important questions such as

- Does the model fit the data well?
- Are there any unusual data points?
- Does the relationship really appear linear?
- Are the predictions reasonable?

For Linear Regression, visualization is particularly important because the model itself is a straight line.

By plotting both the experimental data and the learned regression line, we can immediately understand how well the model describes the dataset.

---

# 2.46 What Should the Graph Show?

A Linear Regression graph contains two components.

### 1. Experimental Data

Each experiment is represented by a point.

For our steel dataset,

each point corresponds to one alloy.

---

### 2. Regression Line

The model predicts one hardness value for every carbon percentage.

Connecting these predictions produces a straight line.

The graph therefore combines

```text
Experimental Data

+

Regression Line
```

into a single figure.

---

# 2.47 Why Is Visualization Important?

Suppose your model produces an excellent prediction error.

That sounds encouraging.

However,

after plotting the data,

you notice that most points follow a curved pattern rather than a straight line.

Immediately,

you realize that Linear Regression may not be the most appropriate algorithm.

Without visualization,

this important observation could easily be missed.

This is why experienced machine learning practitioners almost always visualize their models before drawing scientific conclusions.

---

# 2.48 Importing Matplotlib

To create graphs,

we will use **Matplotlib**.

It is the most widely used plotting library in Python.

Import it as follows.

```python
import matplotlib.pyplot as plt
```

---

## Understanding the Code

### `matplotlib`

The complete plotting library.

---

### `pyplot`

A module that contains functions for creating figures, graphs, and charts.

---

### `as plt`

Creates the nickname

```python
plt
```

instead of repeatedly typing

```python
matplotlib.pyplot
```

Almost every Python machine learning project follows this convention.

---

# 2.49 Plotting the Experimental Data

We first plot the experimental measurements.

```python
plt.scatter(
    X,
    y
)
```

This creates a **scatter plot**.

Each point represents one material in the dataset.

---

## Understanding the Code

### `plt.scatter()`

Creates a scatter plot.

Unlike a line plot,

each observation appears as an individual point.

---

### First Argument

```python
X
```

Provides the horizontal coordinates.

These represent carbon content.

---

### Second Argument

```python
y
```

Provides the vertical coordinates.

These represent hardness.

---

The resulting graph resembles

```text
Hardness

330 |                 ●

300 |

270 |            ●

240 |        ●

210 |    ●

180 | ●

    +----------------------------

      Carbon Content
```

Only the experimental observations are shown.

No regression line has been drawn yet.

---

# 2.50 Plotting the Regression Line

Now we ask the model to predict the hardness of every training sample.

```python
predicted = model.predict(X)
```

These predicted values lie on the regression line.

Next,

we plot them.

```python
plt.plot(
    X,
    predicted
)
```

Now the graph contains both

- experimental points,
- regression line.

Conceptually,

it looks like

```text
Hardness

330 |                 ●

320 |              /

300 |           /

280 |        ●

260 |      /

240 |    ●

220 |  /

200 |●

    +----------------------------

      Carbon Content
```

The straight line represents the mathematical relationship learned during training.

---

# 2.51 Why Use Predictions Instead of Drawing a Line Manually?

A beginner might wonder,

> "Why don't we simply draw a straight line ourselves?"

The answer is simple.

The regression line depends on

- the slope,
- the intercept,

which were learned automatically from the data.

Using

```python
model.predict(X)
```

ensures that the plotted line exactly matches the model.

If the model changes,

the graph changes automatically.

---

# 2.52 Adding Axis Labels

A graph without labels is difficult to interpret.

We therefore label both axes.

```python
plt.xlabel("Carbon (%)")

plt.ylabel("Hardness (HV)")
```

---

## Understanding the Code

### `plt.xlabel()`

Adds a label to the horizontal axis.

---

### `plt.ylabel()`

Adds a label to the vertical axis.

Always include units whenever possible.

This is especially important for scientific publications.

---

# 2.53 Adding a Title

Every scientific graph should have a descriptive title.

```python
plt.title(
    "Linear Regression: Carbon vs Hardness"
)
```

The title explains exactly what the graph represents.

Good titles improve readability in reports, presentations, and research papers.

---

# 2.54 Displaying the Figure

Finally,

display the graph.

```python
plt.show()
```

Without this command,

many Python environments will not display the figure.

---

# 2.55 Complete Visualization Program

```python
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = {
    "Carbon": [0.10, 0.20, 0.30, 0.40, 0.50],
    "Hardness": [150, 190, 240, 280, 330]
}

df = pd.DataFrame(data)

X = df[["Carbon"]]

y = df["Hardness"]

model = LinearRegression()

model.fit(X, y)

predicted = model.predict(X)

plt.scatter(X, y)

plt.plot(X, predicted)

plt.xlabel("Carbon (%)")

plt.ylabel("Hardness (HV)")

plt.title("Linear Regression: Carbon vs Hardness")

plt.show()
```

---

# 2.56 Understanding the Entire Program

The program follows the same workflow we have been building throughout this chapter.

```text
Create Dataset

↓

Train Linear Regression Model

↓

Predict Hardness

↓

Plot Experimental Data

↓

Plot Regression Line

↓

Display Graph
```

Notice that visualization occurs **after** training.

The graph allows us to inspect what the model has learned.

---

# 2.57 Interpreting the Graph

When examining a regression graph, ask yourself the following questions.

### Are the points close to the line?

If yes,

the model is likely capturing the relationship well.

---

### Are the points widely scattered?

If yes,

the relationship may be weak,

or another algorithm may be more appropriate.

---

### Is the relationship curved?

If the data clearly form a curve,

Linear Regression may not be suitable.

Later in this book,

we will study algorithms capable of learning complex nonlinear relationships.

---

### Are there unusual points?

Sometimes,

one or two points lie far away from the regression line.

These are called **outliers**.

Outliers can significantly influence Linear Regression.

We will learn how to detect and handle them later in this chapter.

---

# 2.58 Looking Ahead

So far,

our dataset has contained only one feature.

Real materials science problems are rarely this simple.

A material's properties usually depend on many variables simultaneously,

such as

- density,
- composition,
- lattice parameters,
- grain size,
- formation energy,
- atomic radius,
- electronegativity.

The next section introduces **Multiple Linear Regression**, where several features are combined into a single predictive model. This is the form of Linear Regression most commonly used in modern materials informatics and will prepare us for building models from descriptors extracted with **Pymatgen** and **Quantum ESPRESSO**.

# 2.59 Multiple Linear Regression

So far, every example in this chapter has used **only one feature**.

For example,

we predicted hardness using only

- Carbon content

Although this was useful for understanding the basic ideas of Linear Regression, it is rarely sufficient for solving real scientific problems.

In materials science, a material property is almost never controlled by a single variable.

Instead, multiple factors work together to determine the final behavior of a material.

For example, the hardness of steel may depend on

- carbon content,
- chromium content,
- manganese content,
- grain size,
- density,
- heat treatment temperature,
- cooling rate.

If we use only carbon content, we ignore a large amount of valuable information.

This motivates the use of **Multiple Linear Regression**.

---

# 2.60 From One Feature to Many Features

In Simple Linear Regression, we used

\[
y = mx + b
\]

There was only one feature.

Now suppose we have three features.

- Density
- Carbon Content
- Grain Size

The model becomes

\[
y=b+m_1x_1+m_2x_2+m_3x_3
\]

Instead of learning one slope,

the algorithm now learns **three slopes**.

Each slope describes the influence of one feature.

The intercept remains only one number.

---

# 2.61 General Equation

For a dataset containing many features,

the Linear Regression equation becomes

\[
y=b+m_1x_1+m_2x_2+\cdots+m_nx_n
\]

where

- \(x_1,x_2,\ldots,x_n\) are the features,
- \(m_1,m_2,\ldots,m_n\) are the coefficients,
- \(b\) is the intercept.

Although the equation looks more complicated,

the underlying idea is exactly the same.

The algorithm still tries to find the coefficients that minimize the prediction error.

---

# 2.62 A Materials Science Example

Suppose we wish to predict the hardness of steel.

Instead of using only carbon percentage,

we include additional information.

| Density | Carbon | Grain Size | Hardness |
|---------:|--------:|-----------:|---------:|
|7.82|0.20|24|218|
|7.76|0.35|20|254|
|7.90|0.42|18|287|
|7.71|0.28|25|231|
|7.95|0.50|16|322|

Notice something important.

Every row still represents **one material**.

However,

each material now contains **three input variables** instead of one.

This is the most common type of dataset encountered in engineering research.

---

# 2.63 Why Multiple Features Improve Predictions

Imagine trying to predict the weight of a person.

Would you use only height?

Probably not.

Weight also depends on

- age,
- sex,
- muscle mass,
- body composition.

Similarly,

the hardness of a material rarely depends on only one descriptor.

By combining several meaningful descriptors,

the model usually becomes more accurate.

However,

adding features blindly is **not** always beneficial.

Later in this book,

we will learn how to choose useful features and remove unnecessary ones.

This process is called **feature selection**.

---

# 2.64 Preparing Multiple Features in Python

The only major difference in the code is how we select the feature matrix.

Previously,

we wrote

```python
X = df[["Carbon"]]
```

Now,

we include several columns.

```python
X = df[
    [
        "Density",
        "Carbon",
        "GrainSize"
    ]
]
```

The target remains the same.

```python
y = df["Hardness"]
```

Notice that nothing else changes.

Scikit-learn automatically recognizes that the dataset now contains three features.

---

# 2.65 Complete Example

Let us build our first Multiple Linear Regression model.

```python
import pandas as pd

from sklearn.linear_model import LinearRegression

data = {
    "Density": [7.82,7.76,7.90,7.71,7.95],
    "Carbon": [0.20,0.35,0.42,0.28,0.50],
    "GrainSize": [24,20,18,25,16],
    "Hardness": [218,254,287,231,322]
}

df = pd.DataFrame(data)

X = df[
    [
        "Density",
        "Carbon",
        "GrainSize"
    ]
]

y = df["Hardness"]

model = LinearRegression()

model.fit(X,y)
```

Although this program looks almost identical to the previous one,

it now learns the relationship between **three features** and hardness.

---

# 2.66 Explaining the Code

### Creating the Dataset

```python
data = {...}
```

Stores four columns.

Three are features.

One is the target.

---

### Selecting Features

```python
X = df[
    [
        "Density",
        "Carbon",
        "GrainSize"
    ]
]
```

This creates a feature matrix.

Each row corresponds to one alloy.

Each column corresponds to one descriptor.

Conceptually,

the matrix looks like

| Density | Carbon | Grain Size |
|---------:|--------:|-----------:|
|7.82|0.20|24|
|7.76|0.35|20|
|7.90|0.42|18|
|7.71|0.28|25|
|7.95|0.50|16|

The machine learning algorithm treats each row as one training example.

---

### Selecting the Target

```python
y = df["Hardness"]
```

This stores the property we want to predict.

---

### Creating the Model

```python
model = LinearRegression()
```

Creates an empty Linear Regression model.

---

### Training

```python
model.fit(X,y)
```

The algorithm analyzes all three descriptors simultaneously.

It determines how each descriptor contributes to the predicted hardness.

---

# 2.67 Viewing the Learned Coefficients

After training,

we can examine the learned coefficients.

```python
print(model.coef_)
```

Example output

```text
[38.4  251.7  -4.8]
```

Your numbers will depend on the dataset.

Each value corresponds to one feature.

```text
Density

↓

38.4

Carbon

↓

251.7

Grain Size

↓

-4.8
```

The order is exactly the same as the order of the columns in `X`.

---

# 2.68 Interpreting the Coefficients

Suppose the model learned

```text
Density      = 38.4

Carbon       = 251.7

Grain Size   = -4.8
```

These numbers describe how the predicted hardness changes when one feature increases while all other features remain constant.

For example,

the positive coefficient for carbon indicates that increasing carbon content tends to increase hardness.

The negative coefficient for grain size suggests that larger grains are associated with lower hardness.

This observation is consistent with the Hall–Petch relationship, which states that decreasing grain size generally increases the strength and hardness of many polycrystalline materials.

This illustrates an important advantage of Linear Regression.

The coefficients are often physically interpretable.

---

# 2.69 Viewing the Intercept

The intercept is obtained exactly as before.

```python
print(model.intercept_)
```

Example

```text
-176.3
```

Together,

the coefficients and intercept define the complete regression equation.

---

# 2.70 Making Predictions

Suppose a new alloy has

- Density = 7.85
- Carbon = 0.38
- Grain Size = 19

We predict its hardness as follows.

```python
new_alloy = [
    [
        7.85,
        0.38,
        19
    ]
]

prediction = model.predict(new_alloy)

print(prediction)
```

Example output

```text
[272.6]
```

The model estimates that the hardness of this alloy is approximately **273 HV**.

---

# 2.71 Understanding the Input Format

Notice the structure of

```python
new_alloy
```

```python
[
    [
        7.85,
        0.38,
        19
    ]
]
```

The outer list represents the dataset.

The inner list represents one material.

The feature order must **exactly match** the order used during training.

If the model was trained using

```text
Density

Carbon

Grain Size
```

then every future prediction must follow the same order.

Changing the order can produce completely incorrect predictions.

---

# 2.72 Why Multiple Linear Regression Matters in Materials Informatics

Most real materials informatics datasets contain **dozens**, **hundreds**, or even **thousands** of descriptors.

For example,

using **Pymatgen** and **Matminer**, we can automatically generate descriptors such as

- density,
- unit cell volume,
- average atomic weight,
- mean electronegativity,
- average covalent radius,
- packing fraction,
- oxidation-state statistics,
- elemental property statistics,
- structural fingerprints.

Each descriptor becomes one feature in the machine learning model.

The concepts you have learned here remain exactly the same.

The only difference is that the number of features increases dramatically.

In the next section, we will begin connecting Multiple Linear Regression with **Pymatgen**, showing how to transform crystal structures into numerical descriptors that can be used directly for machine learning.


# 2.73 From Crystal Structures to Machine Learning Features

Up to this point, we have manually created our datasets.

For example,

```python
data = {
    "Density": [7.82, 7.76, 7.90],
    "Carbon": [0.20, 0.35, 0.42],
    "Hardness": [218, 254, 287]
}
```

This approach is useful for learning machine learning concepts.

However, **real materials informatics research almost never begins with manually typed datasets.**

Instead, researchers start with

- crystal structures,
- DFT calculations,
- experimental measurements,
- or materials databases.

The first task is to convert this scientific information into numerical features that a machine learning algorithm can understand.

This process is called **feature extraction**.

Feature extraction is one of the most important steps in the entire machine learning pipeline.

In many research projects, choosing good features has a greater impact on model performance than choosing the machine learning algorithm itself.

---

# 2.74 Why Can't We Train Directly on a Crystal Structure?

Suppose you have a POSCAR file from Quantum ESPRESSO or VASP.

A portion of the file may look like

```text
Fe
1.0
2.87 0.00 0.00
0.00 2.87 0.00
0.00 0.00 2.87
Fe
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
```

A human can recognize that this describes a crystal structure.

A machine learning model cannot.

Algorithms such as Linear Regression expect numerical input arranged in rows and columns.

They cannot interpret lattice vectors, atomic positions, or crystallographic information directly.

Therefore, we must transform the crystal structure into meaningful numerical descriptors.

---

# 2.75 The Role of Pymatgen

This is where **Pymatgen** becomes indispensable.

Pymatgen is capable of reading crystal structures from many common file formats, including

- POSCAR,
- CIF,
- CONTCAR,
- CHGCAR,
- vasprun.xml,
- Quantum ESPRESSO outputs (through appropriate parsers and converters),
- and many others.

After reading the structure, Pymatgen provides access to hundreds of structural and compositional properties through Python objects.

These properties become candidate features for machine learning.

---

# 2.76 Loading a Crystal Structure

Suppose you have a POSCAR file in your working directory.

We first import the required class.

```python
from pymatgen.core import Structure
```

Next, we load the structure.

```python
structure = Structure.from_file("POSCAR")
```

At this point,

the entire crystal structure is stored inside the variable

```python
structure
```

This object contains

- lattice parameters,
- atomic coordinates,
- element types,
- symmetry information,
- composition,
- and many additional properties.

---

# 2.77 Understanding Every Line of Code

### Import Statement

```python
from pymatgen.core import Structure
```

The keyword

```python
from
```

tells Python where to find the class.

The module

```python
pymatgen.core
```

contains the fundamental objects used to describe materials.

The class

```python
Structure
```

represents a periodic crystal structure.

---

### Loading the File

```python
Structure.from_file("POSCAR")
```

The method

```python
from_file()
```

reads the structure file,

parses its contents,

creates a `Structure` object,

and returns it.

We assign the returned object to

```python
structure
```

so that it can be used later.

---

# 2.78 Extracting the Unit Cell Volume

Once the structure has been loaded,

obtaining the unit cell volume is straightforward.

```python
volume = structure.volume

print(volume)
```

---

## What Does `structure.volume` Do?

The attribute

```python
volume
```

returns the volume of the unit cell.

The unit is typically

\[
\text{Å}^3
\]

The calculation is performed automatically by Pymatgen using the lattice vectors.

There is no need to derive or implement the crystallographic formula yourself.

---

# 2.79 Extracting the Density

The density is equally easy to obtain.

```python
density = structure.density

print(density)
```

The returned value is usually expressed in

\[
\text{g/cm}^3
\]

Pymatgen calculates the density from

- the atomic masses,
- the number of atoms,
- and the unit cell volume.

Again,

all calculations are handled internally.

---

# 2.80 Extracting the Chemical Composition

Next, we obtain the composition.

```python
composition = structure.composition

print(composition)
```

Example output

```text
Fe2
```

For a more complex material,

the output might be

```text
LiFePO4
```

or

```text
BaTiO3
```

The composition object contains much more information than a simple chemical formula.

Later, we will use it to generate many compositional descriptors.

---

# 2.81 Extracting Lattice Parameters

The lattice constants are often useful machine learning features.

For example,

```python
a = structure.lattice.a

b = structure.lattice.b

c = structure.lattice.c

print(a)

print(b)

print(c)
```

For cubic crystals,

the values are identical.

For lower-symmetry crystals,

they may differ.

These lattice parameters frequently appear in materials informatics datasets.

---

# 2.82 Combining Multiple Features

Instead of extracting only one property,

we usually collect several descriptors.

```python
volume = structure.volume

density = structure.density

a = structure.lattice.a

b = structure.lattice.b

c = structure.lattice.c
```

These quantities now become candidate features.

Conceptually,

our machine learning dataset begins to take shape.

| Volume | Density | a | b | c |
|---------:|---------:|--:|--:|--:|
|11.78|7.86|2.87|2.87|2.87|

Notice that we have transformed a crystal structure into numerical data.

This is exactly what machine learning algorithms require.

---

# 2.83 Building a Dataset from Many Structures

Real research rarely involves a single material.

Suppose you have calculated 500 crystal structures.

Your project directory might look like

```text
Materials/

├── Alloy_001/
│   └── POSCAR
│
├── Alloy_002/
│   └── POSCAR
│
├── Alloy_003/
│   └── POSCAR
│
...
│
└── Alloy_500/
    └── POSCAR
```

Rather than extracting features manually,

we automate the process.

For each structure,

we

1. load the POSCAR,
2. extract descriptors,
3. store the results.

The final output becomes a machine learning dataset containing hundreds of rows.

---

# 2.84 The Complete Workflow So Far

We have now expanded our understanding of the machine learning pipeline.

Instead of beginning with a manually created DataFrame,

our workflow is now

```text
Crystal Structure

↓

Pymatgen

↓

Feature Extraction

↓

Numerical Descriptors

↓

Pandas DataFrame

↓

Linear Regression
```

This is much closer to how modern materials informatics research is performed.

---

# 2.85 Looking Ahead

The descriptors we extracted in this section were obtained one at a time.

In practice,

we often need dozens or even hundreds of descriptors for each material.

Writing code to compute every descriptor manually would be tedious and error-prone.

Fortunately, the Python ecosystem provides tools that automate descriptor generation.

In the next section, we will introduce **Matminer**, a library built on top of Pymatgen that can generate rich compositional and structural descriptors automatically. These descriptors will become the input features for increasingly powerful machine learning models throughout the rest of this book.


# 2.73 From Crystal Structures to Machine Learning Features

Up to this point, we have manually created our datasets.

For example,

```python
data = {
    "Density": [7.82, 7.76, 7.90],
    "Carbon": [0.20, 0.35, 0.42],
    "Hardness": [218, 254, 287]
}
```

This approach is useful for learning machine learning concepts.

However, **real materials informatics research almost never begins with manually typed datasets.**

Instead, researchers start with

- crystal structures,
- DFT calculations,
- experimental measurements,
- or materials databases.

The first task is to convert this scientific information into numerical features that a machine learning algorithm can understand.

This process is called **feature extraction**.

Feature extraction is one of the most important steps in the entire machine learning pipeline.

In many research projects, choosing good features has a greater impact on model performance than choosing the machine learning algorithm itself.

---

# 2.74 Why Can't We Train Directly on a Crystal Structure?

Suppose you have a POSCAR file from Quantum ESPRESSO or VASP.

A portion of the file may look like

```text
Fe
1.0
2.87 0.00 0.00
0.00 2.87 0.00
0.00 0.00 2.87
Fe
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
```

A human can recognize that this describes a crystal structure.

A machine learning model cannot.

Algorithms such as Linear Regression expect numerical input arranged in rows and columns.

They cannot interpret lattice vectors, atomic positions, or crystallographic information directly.

Therefore, we must transform the crystal structure into meaningful numerical descriptors.

---

# 2.75 The Role of Pymatgen

This is where **Pymatgen** becomes indispensable.

Pymatgen is capable of reading crystal structures from many common file formats, including

- POSCAR,
- CIF,
- CONTCAR,
- CHGCAR,
- vasprun.xml,
- Quantum ESPRESSO outputs (through appropriate parsers and converters),
- and many others.

After reading the structure, Pymatgen provides access to hundreds of structural and compositional properties through Python objects.

These properties become candidate features for machine learning.

---

# 2.76 Loading a Crystal Structure

Suppose you have a POSCAR file in your working directory.

We first import the required class.

```python
from pymatgen.core import Structure
```

Next, we load the structure.

```python
structure = Structure.from_file("POSCAR")
```

At this point,

the entire crystal structure is stored inside the variable

```python
structure
```

This object contains

- lattice parameters,
- atomic coordinates,
- element types,
- symmetry information,
- composition,
- and many additional properties.

---

# 2.77 Understanding Every Line of Code

### Import Statement

```python
from pymatgen.core import Structure
```

The keyword

```python
from
```

tells Python where to find the class.

The module

```python
pymatgen.core
```

contains the fundamental objects used to describe materials.

The class

```python
Structure
```

represents a periodic crystal structure.

---

### Loading the File

```python
Structure.from_file("POSCAR")
```

The method

```python
from_file()
```

reads the structure file,

parses its contents,

creates a `Structure` object,

and returns it.

We assign the returned object to

```python
structure
```

so that it can be used later.

---

# 2.78 Extracting the Unit Cell Volume

Once the structure has been loaded,

obtaining the unit cell volume is straightforward.

```python
volume = structure.volume

print(volume)
```

---

## What Does `structure.volume` Do?

The attribute

```python
volume
```

returns the volume of the unit cell.

The unit is typically

\[
\text{Å}^3
\]

The calculation is performed automatically by Pymatgen using the lattice vectors.

There is no need to derive or implement the crystallographic formula yourself.

---

# 2.79 Extracting the Density

The density is equally easy to obtain.

```python
density = structure.density

print(density)
```

The returned value is usually expressed in

\[
\text{g/cm}^3
\]

Pymatgen calculates the density from

- the atomic masses,
- the number of atoms,
- and the unit cell volume.

Again,

all calculations are handled internally.

---

# 2.80 Extracting the Chemical Composition

Next, we obtain the composition.

```python
composition = structure.composition

print(composition)
```

Example output

```text
Fe2
```

For a more complex material,

the output might be

```text
LiFePO4
```

or

```text
BaTiO3
```

The composition object contains much more information than a simple chemical formula.

Later, we will use it to generate many compositional descriptors.

---

# 2.81 Extracting Lattice Parameters

The lattice constants are often useful machine learning features.

For example,

```python
a = structure.lattice.a

b = structure.lattice.b

c = structure.lattice.c

print(a)

print(b)

print(c)
```

For cubic crystals,

the values are identical.

For lower-symmetry crystals,

they may differ.

These lattice parameters frequently appear in materials informatics datasets.

---

# 2.82 Combining Multiple Features

Instead of extracting only one property,

we usually collect several descriptors.

```python
volume = structure.volume

density = structure.density

a = structure.lattice.a

b = structure.lattice.b

c = structure.lattice.c
```

These quantities now become candidate features.

Conceptually,

our machine learning dataset begins to take shape.

| Volume | Density | a | b | c |
|---------:|---------:|--:|--:|--:|
|11.78|7.86|2.87|2.87|2.87|

Notice that we have transformed a crystal structure into numerical data.

This is exactly what machine learning algorithms require.

---

# 2.83 Building a Dataset from Many Structures

Real research rarely involves a single material.

Suppose you have calculated 500 crystal structures.

Your project directory might look like

```text
Materials/

├── Alloy_001/
│   └── POSCAR
│
├── Alloy_002/
│   └── POSCAR
│
├── Alloy_003/
│   └── POSCAR
│
...
│
└── Alloy_500/
    └── POSCAR
```

Rather than extracting features manually,

we automate the process.

For each structure,

we

1. load the POSCAR,
2. extract descriptors,
3. store the results.

The final output becomes a machine learning dataset containing hundreds of rows.

---

# 2.84 The Complete Workflow So Far

We have now expanded our understanding of the machine learning pipeline.

Instead of beginning with a manually created DataFrame,

our workflow is now

```text
Crystal Structure

↓

Pymatgen

↓

Feature Extraction

↓

Numerical Descriptors

↓

Pandas DataFrame

↓

Linear Regression
```

This is much closer to how modern materials informatics research is performed.

---

# 2.85 Looking Ahead

The descriptors we extracted in this section were obtained one at a time.

In practice,

we often need dozens or even hundreds of descriptors for each material.

Writing code to compute every descriptor manually would be tedious and error-prone.

Fortunately, the Python ecosystem provides tools that automate descriptor generation.

In the next section, we will introduce **Matminer**, a library built on top of Pymatgen that can generate rich compositional and structural descriptors automatically. These descriptors will become the input features for increasingly powerful machine learning models throughout the rest of this book.


# 2.101 The Mathematics of Linear Regression

So far, we have successfully trained Linear Regression models using Scikit-learn.

Whenever we wanted to train a model, we simply wrote

```python
model.fit(X, y)
```

and the computer automatically learned the regression equation.

This is convenient.

However, if you want to become a machine learning researcher—or understand why a model succeeds or fails—you must know **what happens mathematically inside `fit()`**.

From this point onward, we will temporarily set Python aside and focus on the mathematics behind Linear Regression.

Do not worry if some equations seem unfamiliar at first.

We will derive every important result step by step and explain the intuition behind each equation before moving to the next.

By the end of this section, you will understand not only **how to use** Linear Regression but also **why it works**.

---

# 2.102 The Goal of the Algorithm

Imagine that we have collected data for several materials.

| Carbon (%) | Hardness (HV) |
|------------|---------------|
| 0.10 | 150 |
| 0.20 | 190 |
| 0.30 | 240 |
| 0.40 | 280 |
| 0.50 | 330 |

The algorithm receives only these experimental observations.

It does **not** know the relationship between carbon content and hardness.

Its objective is to discover a mathematical equation that best describes the data.

That equation will later be used to predict the hardness of new materials.

---

# 2.103 The Regression Equation

For Simple Linear Regression, we assume that the relationship between the feature and the target is linear.

The model is written as

\[
\hat{y}=mx+b
\]

Notice that we use

\[
\hat{y}
\]

instead of

\[
y.
\]

The symbol

\[
\hat{y}
\]

(pronounced *"y hat"*) represents the **predicted value** produced by the model.

The symbol

\[
y
\]

represents the **actual value** measured experimentally.

This distinction is extremely important.

Throughout machine learning,

we will always use

- \(y\) for the true value,
- \(\hat{y}\) for the predicted value.

---

# 2.104 What Are We Trying to Learn?

At first,

the equation

\[
\hat{y}=mx+b
\]

contains two unknown quantities.

- The slope, \(m\)
- The intercept, \(b\)

Training simply means finding the values of \(m\) and \(b\) that produce the best predictions.

Everything else in Linear Regression is designed to accomplish this single objective.

---

# 2.105 Making Predictions

Suppose we guess

\[
m=450
\]

and

\[
b=105.
\]

Our model becomes

\[
\hat{y}=450x+105.
\]

If the carbon content is

\[
x=0.40,
\]

then the predicted hardness is

\[
\hat{y}=450(0.40)+105
\]

\[
=180+105
\]

\[
=285.
\]

The model predicts

```text
Hardness = 285 HV
```

Suppose the experimentally measured hardness is

```text
280 HV
```

The prediction is close,

but it is not perfect.

This difference is called the **prediction error**.

---

# 2.106 Defining the Error

The prediction error for a single observation is called the **residual**.

It is calculated as

\[
e=y-\hat{y}
\]

where

- \(e\) is the residual,
- \(y\) is the actual value,
- \(\hat{y}\) is the predicted value.

For the previous example,

\[
e=280-285=-5.
\]

The negative sign indicates that the model predicted a value slightly larger than the experimental measurement.

---

# 2.107 Why Residuals Matter

Imagine that we have five experimental observations.

Each prediction has its own residual.

| Material | Actual | Predicted | Residual |
|-----------|--------|-----------|----------|
| A |150|152|-2|
| B |190|187|3|
| C |240|242|-2|
| D |280|276|4|
| E |330|328|2|

Some residuals are positive.

Others are negative.

Our goal is to make **all** of them as small as possible.

However, simply adding the residuals creates a problem.

---

# 2.108 Why We Cannot Simply Sum the Errors

Suppose the residuals are

\[
-5,\quad +5,\quad -10,\quad +10.
\]

Their sum is

\[
-5+5-10+10=0.
\]

The total error appears to be zero,

even though every prediction is imperfect.

Positive and negative errors cancel each other.

Clearly,

this is not a useful measure of model quality.

We therefore need a better way to quantify prediction error.

---

# 2.109 Squaring the Errors

To prevent positive and negative residuals from canceling,

we square each residual.

Instead of

\[
e,
\]

we calculate

\[
e^2.
\]

For example,

| Residual | Squared Residual |
|-----------|-----------------|
| -5 | 25 |
| 3 | 9 |
| -2 | 4 |
| 4 | 16 |

Notice that every squared error is positive.

Large mistakes also become much larger after squaring.

This property encourages the algorithm to avoid large prediction errors.

---

# 2.110 Sum of Squared Errors (SSE)

For a dataset containing many observations,

we add all squared residuals together.

This quantity is called the **Sum of Squared Errors (SSE)**.

\[
SSE=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

where

- \(n\) is the number of observations,
- \(y_i\) is the actual value,
- \(\hat{y}_i\) is the predicted value.

The symbol

\[
\sum
\]

(Greek capital sigma) means

**add all the terms together**.

The smaller the SSE,

the better the regression line fits the data.

---

# 2.111 The Central Idea of Linear Regression

Everything we have learned so far can now be summarized in one sentence:

> **Linear Regression chooses the values of the slope and intercept that produce the smallest possible Sum of Squared Errors.**

This idea is known as the **Least Squares Principle**.

It is the mathematical foundation of Linear Regression.

Every implementation—from Scikit-learn to advanced scientific software—ultimately follows this principle.

---

# 2.112 A Visual Interpretation

Imagine three possible regression lines passing through the same experimental data.

```text
Line A
\
 \
  \

Line B
---------
---------

Line C
     /
    /
   /
```

One line may lie far from most of the data points.

Another may pass close to nearly every point.

The algorithm computes the SSE for **every possible line** (using mathematics rather than brute force) and selects the one with the smallest total squared error.

That line becomes the trained model.

---

# 2.113 Connecting Back to Python

When we write

```python
model.fit(X, y)
```

Scikit-learn is effectively solving the optimization problem

\[
\min_{m,b}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2.
\]

The library performs all of the mathematics internally,

but the objective is exactly what we have just derived.

Understanding this equation allows you to see beyond the Python code and appreciate what the algorithm is truly doing.

---

# 2.114 Looking Ahead

We now understand **what** Linear Regression tries to minimize.

The next question is even more interesting:

**How does the algorithm actually find the values of the slope and intercept that minimize the Sum of Squared Errors?**

In the next section, we will derive the least squares solution step by step using calculus and linear algebra, building the mathematical machinery that powers Linear Regression.


# 2.115 Deriving the Least Squares Solution

In the previous section, we discovered the central idea behind Linear Regression.

The algorithm searches for the values of the slope and intercept that minimize the **Sum of Squared Errors (SSE)**.

Now we will answer the next question:

> **How are those values actually calculated?**

This is where mathematics enters the picture.

Do not be intimidated by the equations.

We will derive them one small step at a time, explaining the reasoning behind every step.

---

# 2.116 Starting with the Regression Equation

For Simple Linear Regression, the model predicts

\[
\hat{y}=mx+b
\]

where

- \(m\) is the slope,
- \(b\) is the intercept,
- \(x\) is the feature,
- \(\hat{y}\) is the predicted value.

Suppose we have \(n\) experimental observations.

For the first material,

the prediction is

\[
\hat{y}_1=mx_1+b
\]

For the second material,

\[
\hat{y}_2=mx_2+b
\]

For the third material,

\[
\hat{y}_3=mx_3+b
\]

The same equation is applied to every material in the dataset.

---

# 2.117 Writing the Residual

The residual for each observation is

\[
e_i=y_i-\hat{y}_i
\]

Substituting the regression equation,

we obtain

\[
e_i
=
y_i-(mx_i+b)
\]

This expression tells us exactly how far the prediction is from the experimental measurement.

---

# 2.118 Constructing the Cost Function

Earlier, we learned that simply adding the residuals is not useful.

Instead, we square them.

The Sum of Squared Errors becomes

\[
SSE
=
\sum_{i=1}^{n}
\left(
y_i-(mx_i+b)
\right)^2
\]

This equation is called the **cost function** or **objective function**.

It measures how well the regression line fits the data.

Large values indicate poor predictions.

Small values indicate a good fit.

Our goal is therefore

\[
\boxed{\text{Minimize }SSE}
\]

Everything in Linear Regression revolves around minimizing this equation.

---

# 2.119 Why Is It Called a Cost Function?

Imagine that every prediction error costs money.

A small prediction error might cost

```text
$1
```

A large prediction error might cost

```text
$100
```

The cost function tells us the total "price" of making inaccurate predictions.

The machine learning algorithm tries to reduce this cost as much as possible.

Although no actual money is involved, the mathematical idea is identical.

---

# 2.120 Finding the Minimum

Think about a simple parabola.

\[
y=x^2
\]

If we draw its graph,

it forms a bowl-shaped curve.

The lowest point on this curve is called the **minimum**.

Our SSE function behaves in a similar way.

If we plot the SSE for different values of the slope and intercept,

the graph forms a curved surface.

The very bottom of this surface corresponds to the smallest possible prediction error.

That is exactly where we want our algorithm to be.

---

# 2.121 Using Calculus

How do mathematicians locate the lowest point of a curve?

They use **calculus**.

A fundamental result from calculus states:

> At the minimum of a differentiable function, its derivative is zero.

Therefore, to minimize the SSE, we differentiate it.

Since the SSE depends on two unknown quantities,

- the slope \(m\),
- the intercept \(b\),

we calculate two derivatives.

\[
\frac{\partial SSE}{\partial m}=0
\]

and

\[
\frac{\partial SSE}{\partial b}=0
\]

Notice that we use the symbol

\[
\partial
\]

instead of

\[
d.
\]

This is because the SSE depends on **more than one variable**.

These are called **partial derivatives**.

---

# 2.122 Why Two Equations?

Suppose you are standing on a hill.

To determine whether you have reached the lowest point,

you must check movement in every direction.

Similarly,

the SSE depends on both

- slope,
- intercept.

Changing either one changes the prediction error.

Therefore, we must optimize both simultaneously.

This leads to two equations.

Solving these equations gives the optimal values of \(m\) and \(b\).

---

# 2.123 The Final Least Squares Solution

After carrying out the differentiation and simplifying the algebra, we obtain a remarkably elegant result.

The optimal slope is

\[
m
=
\frac{
\sum
(x_i-\bar{x})
(y_i-\bar{y})
}
{
\sum
(x_i-\bar{x})^2
}
\]

where

- \(\bar{x}\) is the average of all feature values,
- \(\bar{y}\) is the average of all target values.

Once the slope has been calculated, the intercept is

\[
b
=
\bar{y}
-
m\bar{x}
\]

These two equations provide the exact regression line that minimizes the Sum of Squared Errors.

---

# 2.124 Understanding the Slope Formula

Although the equation may appear complicated, its meaning is intuitive.

The numerator

\[
\sum
(x_i-\bar{x})
(y_i-\bar{y})
\]

measures how the feature and target vary together.

If larger values of \(x\) tend to occur with larger values of \(y\),

the numerator becomes positive.

If larger values of \(x\) tend to occur with smaller values of \(y\),

the numerator becomes negative.

The denominator

\[
\sum
(x_i-\bar{x})^2
\]

measures how much the feature itself varies.

The slope is therefore a ratio describing how changes in the feature are associated with changes in the target.

---

# 2.125 Why This Is Important

Most machine learning libraries compute these values automatically.

For example,

when we write

```python
model.fit(X, y)
```

Scikit-learn internally computes the regression coefficients using efficient numerical algorithms based on these mathematical principles.

Understanding these equations allows you to

- interpret model behavior,
- recognize numerical issues,
- understand scientific papers,
- appreciate the assumptions behind Linear Regression.

This knowledge separates someone who merely **uses** machine learning from someone who truly **understands** it.

---

# 2.126 A Practical Perspective

In modern machine learning,

datasets often contain

- hundreds of features,
- thousands of features,
- or even millions of observations.

Writing explicit formulas like the ones above quickly becomes impractical.

Instead, we represent the entire problem using **matrices**.

Matrix notation is more compact, computationally efficient, and naturally extends to Multiple Linear Regression.

Nearly every modern machine learning algorithm relies heavily on linear algebra.

For this reason, the next section introduces the matrix formulation of Linear Regression.

Although it may seem more abstract initially, it will greatly simplify our understanding of more advanced algorithms later in the book, including Ridge Regression, Principal Component Analysis, Support Vector Machines, and even deep learning.


# 2.127 Matrix Formulation of Linear Regression

In the previous section, we derived the equations for the slope and intercept of **Simple Linear Regression**.

Those equations work perfectly when there is only **one feature**.

However, modern machine learning problems rarely involve just one feature.

A materials informatics dataset may contain

- density,
- atomic radius,
- electronegativity,
- lattice parameters,
- formation energy,
- band gap,
- elastic modulus,

and hundreds of other descriptors.

Writing a separate equation for every feature quickly becomes impossible.

To solve this problem, machine learning uses **matrix notation**.

Although matrix notation may look unfamiliar at first, it actually makes the mathematics **simpler**, not more complicated.

---

# 2.128 Why Do We Need Matrices?

Suppose we want to predict hardness using three features.

- Density
- Carbon Content
- Grain Size

The regression equation is

\[
\hat{y}
=
b
+
m_1x_1
+
m_2x_2
+
m_3x_3
\]

Now imagine we increase the number of features.

Instead of three,

suppose we have

- 25 descriptors,
- 100 descriptors,
- 500 descriptors.

Writing the regression equation becomes impractical.

Instead,

we represent all features together using matrices.

---

# 2.129 Representing the Dataset

Suppose we have four materials.

| Density | Carbon | Grain Size |
|---------:|--------:|-----------:|
|7.82|0.20|24|
|7.76|0.35|20|
|7.90|0.42|18|
|7.71|0.28|25|

Instead of writing the table,

we represent it mathematically as

\[
X=
\begin{bmatrix}
7.82 & 0.20 & 24\\
7.76 & 0.35 & 20\\
7.90 & 0.42 & 18\\
7.71 & 0.28 & 25
\end{bmatrix}
\]

This matrix is called the **feature matrix** or **design matrix**.

Notice its structure.

- Every **row** represents one material.
- Every **column** represents one feature.

This convention is used throughout machine learning.

---

# 2.130 Representing the Target

The target values are stored in a column vector.

Suppose the hardness values are

| Hardness |
|----------:|
|218|
|254|
|287|
|231|

Mathematically,

\[
y=
\begin{bmatrix}
218\\
254\\
287\\
231
\end{bmatrix}
\]

Again,

each row corresponds to one material.

---

# 2.131 Representing the Coefficients

The regression coefficients are also stored as a vector.

\[
\beta=
\begin{bmatrix}
b\\
m_1\\
m_2\\
m_3
\end{bmatrix}
\]

Notice something interesting.

The intercept is included in the coefficient vector.

This makes the mathematics much cleaner.

---

# 2.132 Where Does the Intercept Go?

You might wonder,

> "If the intercept is inside the coefficient vector, where does its corresponding feature come from?"

We create an additional feature whose value is always

\[
1
\]

The feature matrix becomes

\[
X=
\begin{bmatrix}
1&7.82&0.20&24\\
1&7.76&0.35&20\\
1&7.90&0.42&18\\
1&7.71&0.28&25
\end{bmatrix}
\]

The first column consists entirely of ones.

When multiplied by the intercept,

it contributes

\[
1\times b=b
\]

for every material.

This elegant trick allows us to treat the intercept exactly like every other coefficient.

---

# 2.133 The Matrix Equation

Instead of writing

\[
\hat{y}
=
b
+
m_1x_1
+
m_2x_2
+
m_3x_3
\]

we simply write

\[
\boxed{\hat{y}=X\beta}
\]

This compact equation represents every prediction for every material simultaneously.

Although it occupies only one line,

it replaces hundreds or even thousands of individual equations.

This is why machine learning relies so heavily on linear algebra.

---

# 2.134 Understanding the Matrix Equation

Let us examine each symbol carefully.

### \(\hat{y}\)

The vector of predicted values.

Each entry corresponds to one material.

---

### \(X\)

The feature matrix.

Contains all descriptors for all materials.

---

### \(\beta\)

The coefficient vector.

Contains

- intercept,
- coefficient for feature 1,
- coefficient for feature 2,
- coefficient for feature 3,
- and so on.

---

When we multiply

\[
X\beta,
\]

every row of the feature matrix is multiplied by the coefficient vector,

producing one prediction for each material.

---

# 2.135 Writing the Cost Function in Matrix Form

Earlier,

we wrote the Sum of Squared Errors as

\[
SSE
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2.
\]

Using matrices,

the same idea becomes

\[
SSE
=
(y-X\beta)^T
(y-X\beta)
\]

This expression may appear intimidating,

but it is simply another way of writing the sum of squared residuals.

The superscript

\[
T
\]

means **transpose**.

It converts a column vector into a row vector,

allowing the multiplication to be performed correctly.

The meaning of the equation remains exactly the same:

measure the prediction errors,

square them,

and add them together.

---

# 2.136 Solving for the Coefficients

By differentiating the matrix cost function with respect to the coefficient vector,

we obtain the celebrated **Normal Equation**.

\[
\boxed{
\beta
=
(X^TX)^{-1}
X^Ty
}
\]

This equation directly computes the optimal regression coefficients.

Every term has a specific meaning.

- \(X^T\) is the transpose of the feature matrix.
- \(X^TX\) summarizes relationships among the features.
- \((X^TX)^{-1}\) is the matrix inverse.
- \(X^Ty\) connects the features with the target values.

Together,

they produce the coefficient vector that minimizes the Sum of Squared Errors.

---

# 2.137 Why the Normal Equation Matters

The Normal Equation has several important advantages.

First,

it provides the **exact** least-squares solution.

There is no trial and error.

No iterative optimization is required.

Second,

it reveals that Linear Regression is fundamentally a problem in **linear algebra** rather than simple arithmetic.

Finally,

many advanced machine learning algorithms are built upon similar matrix formulations.

Understanding this equation will make later topics such as Ridge Regression and Principal Component Analysis much easier.

---

# 2.138 Does Scikit-learn Always Use the Normal Equation?

An interesting question is whether Scikit-learn literally computes

\[
(X^TX)^{-1}X^Ty.
\]

The answer is:

**not always**.

For small datasets,

the Normal Equation is practical.

However,

for very large datasets,

explicitly computing a matrix inverse can be computationally expensive and numerically unstable.

Modern libraries often use more robust numerical techniques such as

- Singular Value Decomposition (SVD),
- QR decomposition,
- or iterative optimization methods.

These methods produce the same least-squares solution while improving numerical stability and computational efficiency.

The mathematics you have learned remains valid;

only the computational strategy changes.

---

# 2.139 Connecting Mathematics to Python

Recall the familiar command

```python
model.fit(X, y)
```

Behind this single line,

Scikit-learn performs operations equivalent to

1. constructing the feature matrix,
2. representing the coefficients as a vector,
3. minimizing the least-squares cost function,
4. computing the optimal coefficient vector.

Although the mathematics is hidden from the user,

it is always there.

Understanding it transforms `model.fit()` from a mysterious command into a well-defined mathematical procedure.

---

# 2.140 Looking Ahead

We now understand the mathematics of Linear Regression from both the scalar and matrix perspectives.

The next step is to evaluate how well a trained model performs.

A regression model is only useful if we can measure its quality objectively.

In the next section, we will study the most important evaluation metrics for regression models—**Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the coefficient of determination (\(R^2\))**—and learn not only how to compute them in Python but also what they mean physically in the context of materials science.


# 2.141 Evaluating Linear Regression Models

Congratulations.

At this point, you know how to

- prepare a dataset,
- train a Linear Regression model,
- make predictions,
- understand the mathematics behind the algorithm.

However, one important question remains.

> **How do we know whether our model is actually good?**

Suppose you build two Linear Regression models.

Model A predicts the hardness of steel.

Model B also predicts the hardness of steel.

Which model should you use?

Without a quantitative way of measuring performance, we have no objective answer.

This is why **evaluation metrics** are essential.

They allow us to measure how well a machine learning model predicts unseen data.

In materials science, evaluation metrics are just as important as the model itself.

A beautiful model with poor predictive performance has little scientific value.

---

# 2.142 Comparing Actual and Predicted Values

Suppose we measured the hardness of five steel samples.

Our model also predicts their hardness.

| Material | Actual (HV) | Predicted (HV) |
|-----------|------------:|---------------:|
| A |150|148|
| B |190|194|
| C |240|236|
| D |280|285|
| E |330|326|

The predictions are not perfect.

Some are slightly higher.

Some are slightly lower.

To judge the model,

we must measure these prediction errors systematically.

---

# 2.143 Prediction Error

For every material,

the prediction error is

\[
Error = Actual - Predicted
\]

or mathematically,

\[
e_i=y_i-\hat{y_i}
\]

Example:

Actual hardness

```text
280 HV
```

Predicted hardness

```text
285 HV
```

Error

```text
280 − 285 = -5 HV
```

The negative sign tells us that the model predicted a value slightly larger than the experimental measurement.

---

# 2.144 Why One Error Is Not Enough

Looking at a single prediction tells us almost nothing.

A model may perform well for one material but poorly for hundreds of others.

Therefore,

we compute evaluation metrics using **every sample** in the dataset.

Each metric summarizes the overall prediction quality in a different way.

The four most important regression metrics are

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (\(R^2\))

These four metrics appear in almost every machine learning paper involving regression.

---

# 2.145 Mean Absolute Error (MAE)

The simplest metric is the **Mean Absolute Error**.

Instead of using positive and negative errors,

we take the absolute value.

\[
|e|
=
|y-\hat{y}|
\]

The MAE is

\[
MAE
=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y_i}|
\]

---

## Why Use Absolute Values?

Suppose the errors are

```text
-5

+5

-10

+10
```

If we simply add them,

the total error becomes

```text
0
```

This is misleading.

Taking the absolute value removes the sign.

The errors become

```text
5

5

10

10
```

Now every mistake contributes positively to the final result.

---

# 2.146 Interpreting MAE

Suppose

```text
MAE = 8 HV
```

This means

> On average, the model's predictions differ from the experimental hardness by **8 HV**.

Notice how easy this is to interpret.

The units remain the same as the target property.

If the target is

- hardness,

the MAE is measured in HV.

If the target is

- formation energy,

the MAE is measured in eV.

If the target is

- thermal conductivity,

the MAE has units of W·m⁻¹·K⁻¹.

This makes MAE particularly intuitive.

---

# 2.147 Mean Squared Error (MSE)

The second metric is the **Mean Squared Error**.

Instead of taking absolute values,

we square every error.

\[
MSE
=
\frac{1}{n}
\sum
(y_i-\hat{y_i})^2
\]

This is the same quantity minimized by Linear Regression during training.

Large errors contribute much more strongly than small errors.

For example,

| Error | Squared Error |
|-------:|--------------:|
|2|4|
|5|25|
|10|100|

Notice that an error of 10 is not merely twice as bad as an error of 5.

It contributes **four times** as much to the MSE.

---

# 2.148 Advantages and Disadvantages of MSE

### Advantages

- Penalizes large prediction errors.
- Smooth mathematical properties make optimization easier.
- Used internally by many machine learning algorithms.

### Disadvantages

The units become squared.

For example,

if hardness is measured in HV,

the MSE is measured in

\[
HV^2
\]

This makes interpretation less intuitive.

---

# 2.149 Root Mean Squared Error (RMSE)

To solve the unit problem,

we take the square root of the MSE.

\[
RMSE
=
\sqrt{MSE}
\]

Now the units return to the original target units.

For hardness prediction,

RMSE is again measured in

```text
HV
```

This makes RMSE much easier to interpret than MSE.

---

# 2.150 Why RMSE Is Popular

RMSE combines the strengths of both previous metrics.

Like MSE,

it strongly penalizes large prediction errors.

Like MAE,

it is expressed in the same units as the target property.

For this reason,

RMSE is one of the most frequently reported regression metrics in scientific publications.

---

# 2.151 Coefficient of Determination (\(R^2\))

The final metric is the

**Coefficient of Determination**

or simply

\[
R^2
\]

Unlike MAE or RMSE,

\(R^2\) does **not** measure prediction error directly.

Instead,

it measures how much of the variation in the data is explained by the model.

Mathematically,

\[
R^2
=
1-
\frac
{\sum(y_i-\hat{y_i})^2}
{\sum(y_i-\bar{y})^2}
\]

where

\[
\bar{y}
\]

is the average of all target values.

---

# 2.152 Understanding \(R^2\)

Suppose

```text
R² = 1.00
```

The model predicts every point perfectly.

---

Suppose

```text
R² = 0.95
```

The model explains

95% of the variation in the experimental data.

This is generally considered an excellent result.

---

Suppose

```text
R² = 0.50
```

Only half of the variation is explained.

The model is moderately useful but may require improvement.

---

Suppose

```text
R² = 0
```

The model performs no better than simply predicting the average value for every material.

---

Suppose

```text
R² < 0
```

The model performs **worse** than predicting the average.

This indicates a very poor model.

---

# 2.153 Computing Metrics in Python

Scikit-learn provides functions for calculating these metrics.

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

These functions compare

- actual values,
- predicted values,

and compute the evaluation metrics automatically.

---

# 2.154 Calculating MAE

```python
mae = mean_absolute_error(
    y,
    predicted
)

print(mae)
```

### Explanation

The first argument

```python
y
```

contains the experimental values.

The second argument

```python
predicted
```

contains the model predictions.

The function computes the average absolute difference.

---

# 2.155 Calculating MSE

```python
mse = mean_squared_error(
    y,
    predicted
)

print(mse)
```

This calculates the average squared prediction error.

---

# 2.156 Calculating RMSE

Scikit-learn returns MSE directly.

We obtain RMSE by taking its square root.

```python
import numpy as np

rmse = np.sqrt(mse)

print(rmse)
```

---

### Understanding the Code

```python
np.sqrt()
```

computes the square root of the MSE,

returning the RMSE in the original units.

---

# 2.157 Calculating \(R^2\)

```python
r2 = r2_score(
    y,
    predicted
)

print(r2)
```

The returned value lies close to

```text
1
```

for good models.

---

# 2.158 Complete Evaluation Program

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

mae = mean_absolute_error(y, predicted)

mse = mean_squared_error(y, predicted)

rmse = np.sqrt(mse)

r2 = r2_score(y, predicted)

print("MAE :", mae)

print("MSE :", mse)

print("RMSE:", rmse)

print("R²  :", r2)
```

This small block of code provides a comprehensive evaluation of the regression model.

---

# 2.159 Which Metric Should We Report?

In materials informatics papers,

it is common to report **multiple metrics** rather than relying on just one.

A good practice is to include

- **MAE**, because it is easy to interpret,
- **RMSE**, because it penalizes large errors,
- **R²**, because it measures how well the model explains the data.

Reporting all three gives readers a much more complete understanding of model performance.

---

# 2.160 Looking Ahead

So far, we have evaluated the model using the **same data that was used for training**.

This is dangerous.

A model may perform extremely well on the training data but fail completely when predicting new materials.

This phenomenon is called **overfitting**, and understanding it is one of the most important lessons in machine learning.

In the next section, we will study **training sets, test sets, overfitting, underfitting, and data splitting**, learning how to evaluate models properly so that their reported performance reflects their ability to generalize to unseen materials rather than simply memorizing the training data.


# 2.161 Why Training Accuracy Is Not Enough

In the previous section, we learned how to calculate

- MAE,
- MSE,
- RMSE,
- \(R^2\).

Suppose your Linear Regression model reports

```text
MAE = 0

RMSE = 0

R² = 1.00
```

At first glance, this seems perfect.

You might think you have built the best possible machine learning model.

However, there is a serious problem.

Those metrics were calculated using **the same data that the model learned from**.

This is similar to giving students the exam questions before the exam and then claiming they are excellent because they answered every question correctly.

A fair evaluation requires testing on data the model has **never seen before**.

---

# 2.162 An Everyday Analogy

Imagine a student preparing for a physics exam.

There are two possible approaches.

### Student A

Memorizes every solved example in the textbook.

When the teacher asks those exact questions,

the student scores 100%.

However,

when a new question appears,

the student cannot solve it.

---

### Student B

Understands the concepts.

The questions are different from the textbook,

yet the student still performs well because the underlying principles have been learned.

Machine learning models behave in exactly the same way.

A good model should **learn patterns**, not memorize data.

---

# 2.163 Training Set and Test Set

To evaluate a model fairly,

we divide the dataset into two parts.

```text
Entire Dataset

↓

Training Set

+

Test Set
```

The **training set** is used to teach the model.

The **test set** is used only for evaluation.

The model must never see the test data during training.

This ensures that the evaluation reflects the model's ability to predict unseen materials.

---

# 2.164 A Simple Example

Suppose we have ten materials.

```text
Material 1

Material 2

Material 3

Material 4

Material 5

Material 6

Material 7

Material 8

Material 9

Material 10
```

We might split them as follows.

Training set

```text
1

2

3

4

5

6

7

8
```

Test set

```text
9

10
```

The model learns only from Materials 1–8.

After training,

it predicts the properties of Materials 9 and 10.

These predictions provide an unbiased estimate of model performance.

---

# 2.165 Why Not Train on Everything?

A common beginner's question is

> "If more data is better, why not train on the entire dataset?"

Because we would have **nothing left to test**.

Without unseen data,

we cannot determine whether the model has truly learned or merely memorized the training examples.

The purpose of the test set is to simulate future materials that the model has never encountered.

---

# 2.166 Splitting Data in Python

Scikit-learn makes data splitting extremely simple.

```python
from sklearn.model_selection import train_test_split
```

This function automatically divides the dataset.

---

# 2.167 Performing the Split

Suppose our feature matrix is

```python
X
```

and our target vector is

```python
y
```

We split the data as follows.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This single command prepares the data for machine learning.

---

# 2.168 Understanding Every Argument

### `X`

The feature matrix.

Contains all input descriptors.

---

### `y`

The target values.

These are the properties we wish to predict.

---

### `test_size=0.2`

Specifies that

20%

of the dataset will become the test set.

The remaining

80%

will become the training set.

For example,

if the dataset contains

1000 materials,

then

```text
Training

↓

800 materials

Test

↓

200 materials
```

---

### `random_state=42`

Before splitting,

Scikit-learn randomly shuffles the dataset.

Setting

```python
random_state=42
```

ensures that the same random split is produced every time the program is executed.

This makes experiments reproducible.

The number

```text
42
```

has no mathematical significance.

It has simply become a traditional example in programming and machine learning.

You could replace it with

```python
1

10

100

2024
```

or any other integer.

---

# 2.169 Training Only on the Training Set

After splitting,

we train the model.

```python
model = LinearRegression()

model.fit(
    X_train,
    y_train
)
```

Notice something very important.

We do **not** use

```python
X_test

y_test
```

during training.

The test data remain hidden from the model.

---

# 2.170 Predicting the Test Set

Once training is complete,

we make predictions only on the test data.

```python
predicted = model.predict(X_test)
```

These predictions represent how well the model generalizes to unseen materials.

---

# 2.171 Evaluating the Test Set

Now we compute the evaluation metrics.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(
    y_test,
    predicted
)

print(mae)
```

Notice the difference.

Earlier,

we compared

```python
y

predicted
```

Now,

we compare

```python
y_test

predicted
```

Only the unseen test data are used for evaluation.

This provides a much more realistic estimate of performance.

---

# 2.172 What Is Overfitting?

Overfitting occurs when a model learns the **training data too well**.

Instead of discovering general scientific relationships,

it memorizes the individual training samples.

As a result,

training accuracy becomes extremely high,

but test accuracy becomes poor.

Conceptually,

it looks like this.

```text
Training Error

↓

Very Small

Test Error

↓

Large
```

The model appears impressive during training,

but performs poorly on new materials.

---

# 2.173 What Is Underfitting?

Underfitting is the opposite problem.

The model is too simple to capture the underlying relationship.

Both the training and test errors remain large.

Conceptually,

```text
Training Error

↓

Large

Test Error

↓

Large
```

The model has failed to learn useful patterns.

---

# 2.174 The Ideal Situation

The best model strikes a balance.

It learns the important trends in the training data without memorizing noise.

In this case,

both the training and test errors are small.

```text
Training Error

↓

Small

Test Error

↓

Small
```

This indicates good **generalization**.

Generalization is one of the primary goals of machine learning.

---

# 2.175 A Materials Science Perspective

Imagine you train a model using

300 experimentally measured alloys.

If the model performs well only on those 300 alloys,

it has little practical value.

A useful materials informatics model should accurately predict the properties of

- newly synthesized alloys,
- hypothetical compounds,
- structures generated by DFT,
- materials retrieved from databases such as the Materials Project.

The purpose of machine learning is not to reproduce known data.

It is to make reliable predictions for **new materials**.

---

# 2.176 Complete Example

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predicted = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predicted
)

print("Test MAE:", mae)
```

This program demonstrates the standard workflow used in nearly every supervised machine learning project.

---

# 2.177 Best Practices

Whenever you build a regression model,

follow this sequence.

```text
Prepare Dataset

↓

Split Data

↓

Train on Training Set

↓

Predict Test Set

↓

Evaluate Test Predictions

↓

Interpret Results
```

Avoid evaluating the model on the same data used for training unless you are explicitly investigating training performance.

---

# 2.178 Looking Ahead

A single train-test split provides one estimate of model performance.

However, the result depends on **which samples happen to fall into the training set and which fall into the test set**.

For small materials science datasets, this randomness can significantly influence the reported accuracy.

In the next section, we will study **cross-validation**, a more robust evaluation strategy that repeatedly trains and tests the model on different subsets of the data. Cross-validation is widely used in materials informatics research because it provides a more reliable estimate of how well a model will perform on unseen materials.


# 2.179 Cross-Validation

In the previous section, we learned how to divide a dataset into

- a training set,
- a test set.

This approach is simple and widely used.

However, it has one important limitation.

The performance of the model depends on **which samples happen to be selected for training and which are selected for testing**.

Imagine a dataset containing only 100 materials.

If the test set accidentally contains several unusual materials,

the reported accuracy may appear much worse than the model's true performance.

On the other hand,

if the test set contains only easy-to-predict materials,

the reported accuracy may appear unrealistically good.

To overcome this problem, machine learning uses **Cross-Validation**.

---

# 2.180 The Main Idea

Instead of testing the model only once,

Cross-Validation tests it **multiple times**.

Each time,

a different part of the dataset becomes the test set.

The remaining data become the training set.

The final performance is obtained by averaging the results.

Instead of trusting one experiment,

we trust the average of many experiments.

This provides a much more reliable estimate of model performance.

---

# 2.181 K-Fold Cross-Validation

The most common method is called **K-Fold Cross-Validation**.

The dataset is divided into **K equal parts**, called **folds**.

Suppose

\[
K = 5
\]

The dataset is divided like this.

```text
Fold 1

Fold 2

Fold 3

Fold 4

Fold 5
```

Each fold contains approximately the same number of samples.

---

# 2.182 How It Works

The model is trained and tested five times.

### First Iteration

```text
Test

↓

Fold 1

Training

↓

Fold 2 + Fold 3 + Fold 4 + Fold 5
```

---

### Second Iteration

```text
Test

↓

Fold 2

Training

↓

Fold 1 + Fold 3 + Fold 4 + Fold 5
```

---

### Third Iteration

```text
Test

↓

Fold 3

Training

↓

Fold 1 + Fold 2 + Fold 4 + Fold 5
```

---

The process continues until every fold has served as the test set exactly once.

---

# 2.183 Why Is This Better?

Suppose one material is particularly unusual.

In a simple train-test split,

that material might strongly influence the reported accuracy.

In Cross-Validation,

every material appears

- in the training set several times,
- in the test set exactly once.

No single random split dominates the evaluation.

The reported performance becomes much more stable.

---

# 2.184 Visualizing Five-Fold Cross-Validation

Conceptually,

the process looks like this.

```text
Iteration 1

[Test][Train][Train][Train][Train]

Iteration 2

[Train][Test][Train][Train][Train]

Iteration 3

[Train][Train][Test][Train][Train]

Iteration 4

[Train][Train][Train][Test][Train]

Iteration 5

[Train][Train][Train][Train][Test]
```

After all five iterations,

the evaluation metrics are averaged.

---

# 2.185 Choosing the Value of K

The value of

\[
K
\]

is chosen by the researcher.

The most common choices are

- 5
- 10

Five-fold and ten-fold Cross-Validation are widely accepted in machine learning research.

Larger values generally produce more stable estimates,

but they also require more computation because the model must be trained many more times.

---

# 2.186 Leave-One-Out Cross-Validation

An extreme version of Cross-Validation is called

**Leave-One-Out Cross-Validation (LOOCV).**

Suppose the dataset contains

100 materials.

The model is trained

100 times.

Each time,

only **one material** is used for testing,

while the remaining 99 materials are used for training.

LOOCV uses almost all available data for training,

making it attractive for very small datasets.

However,

it can become computationally expensive for larger datasets.

---

# 2.187 Cross-Validation in Python

Scikit-learn provides an easy way to perform Cross-Validation.

First,

import the required function.

```python
from sklearn.model_selection import cross_val_score
```

---

# 2.188 Training with Cross-Validation

Suppose we already have

```python
model

X

y
```

We perform five-fold Cross-Validation.

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)
```

This command automatically

- divides the dataset,
- trains the model five times,
- evaluates each model,
- returns five performance scores.

---

# 2.189 Understanding Every Argument

### `model`

The machine learning algorithm.

For example,

```python
LinearRegression()
```

---

### `X`

The feature matrix.

---

### `y`

The target values.

---

### `cv=5`

Requests five-fold Cross-Validation.

The dataset is automatically divided into five folds.

---

# 2.190 Examining the Results

Suppose we print

```python
print(scores)
```

Example output

```text
[0.93 0.95 0.91 0.94 0.96]
```

These values represent the

\[
R^2
\]

score obtained during each fold.

Notice that every training-test split produces a slightly different result.

This is expected because the training data change during every iteration.

---

# 2.191 Computing the Average Score

Usually,

we report the average performance.

```python
import numpy as np

print(np.mean(scores))
```

Example output

```text
0.938
```

This average provides a more reliable estimate than any single train-test split.

---

# 2.192 Reporting the Standard Deviation

The average alone is not enough.

We should also know how much the scores vary.

This is measured using the standard deviation.

```python
print(np.std(scores))
```

Example output

```text
0.018
```

A small standard deviation indicates that the model performs consistently across different data splits.

A large standard deviation suggests that the model is sensitive to the particular choice of training and test data.

---

# 2.193 Using Different Evaluation Metrics

By default,

Cross-Validation returns the

\[
R^2
\]

score.

However,

we can evaluate other metrics as well.

For example,

to compute the Mean Absolute Error,

we write

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)
```

You may notice the word

```text
neg
```

This is because Scikit-learn assumes that **higher scores are always better**.

Since smaller MAE values are better,

the library returns the negative MAE.

To obtain the actual MAE,

simply multiply by

```python
-1
```

---

# 2.194 Complete Example

```python
import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

model = LinearRegression()

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("Fold Scores")

print(scores)

print()

print("Average R²")

print(np.mean(scores))

print()

print("Standard Deviation")

print(np.std(scores))
```

This short program performs an evaluation that would otherwise require repeatedly writing train-test split code.

---

# 2.195 Why Materials Scientists Prefer Cross-Validation

Many materials science datasets are relatively small.

For example,

a dataset may contain only

- 150 alloys,
- 300 crystal structures,
- 500 DFT calculations.

Losing 20% of the data to a single test set may significantly reduce the amount of information available for training.

Cross-Validation allows every sample to contribute to both

- training,
- testing,

while still ensuring that each prediction is made on data not used during training.

For this reason,

Cross-Validation is widely used in materials informatics and computational materials science.

---

# 2.196 Cross-Validation Is Not a Replacement for a Final Test Set

An important point often misunderstood by beginners is that Cross-Validation is **not** always the final evaluation step.

A common workflow is

```text
Entire Dataset

↓

Hold-Out Test Set

↓

Remaining Data

↓

Cross-Validation

↓

Model Selection

↓

Final Evaluation on the Hold-Out Test Set
```

The hold-out test set is used only once,

after all model development and tuning have been completed.

This provides an unbiased estimate of real-world performance.

---

# 2.197 Looking Ahead

We now know how to evaluate regression models fairly and reliably.

The next question is equally important:

**What assumptions does Linear Regression make about the data?**

Linear Regression performs well only when certain mathematical assumptions are approximately satisfied.

Violating these assumptions can produce misleading predictions and incorrect scientific conclusions.

In the next section, we will study the fundamental assumptions of Linear Regression, including **linearity, independence of observations, homoscedasticity, normality of residuals, and multicollinearity**, along with practical methods for checking each assumption using both Python and visual diagnostics.


# 2.198 The Assumptions of Linear Regression

Up to this point, we have treated Linear Regression as if it were a universal solution.

We collected data,

trained a model,

evaluated its performance,

and obtained predictions.

However, there is something extremely important that every machine learning practitioner must understand.

**Linear Regression is built upon several mathematical assumptions.**

If these assumptions are approximately satisfied,

Linear Regression performs remarkably well.

If they are severely violated,

the model may produce

- inaccurate predictions,
- misleading coefficients,
- unreliable scientific conclusions.

Understanding these assumptions is therefore just as important as knowing how to write the Python code.

---

# 2.199 Why Do Assumptions Matter?

Imagine using a thermometer to measure pressure.

The instrument works perfectly,

but it is designed for the wrong task.

The problem is not the thermometer.

The problem is using it under inappropriate conditions.

Machine learning algorithms are similar.

Every algorithm is designed for particular types of data.

Linear Regression assumes that the relationship between the features and the target follows certain patterns.

If those patterns are absent,

another algorithm may be more appropriate.

---

# 2.200 The Five Main Assumptions

Classical Linear Regression is based on five fundamental assumptions.

1. Linearity

2. Independence of observations

3. Homoscedasticity

4. Normality of residuals

5. Absence of multicollinearity

We will study each assumption individually.

---

# 2.201 Assumption 1 — Linearity

The first assumption is the most obvious.

Linear Regression assumes that the relationship between the features and the target is approximately linear.

Mathematically,

\[
\hat{y}
=
b
+
m_1x_1
+
m_2x_2
+\cdots
+
m_nx_n
\]

The contribution of each feature is assumed to be linear.

---

## Example

Suppose increasing carbon content produces

```text
Carbon (%)

↓

0.10

0.20

0.30

0.40

0.50
```

Hardness increases approximately along a straight line.

Linear Regression is likely to perform well.

---

Now imagine the relationship looks like

```text
Hardness

↑

        ●

     ●

  ●

●

      ●
```

or follows a curved trend.

In this case,

Linear Regression may systematically underpredict or overpredict certain regions of the data.

---

# 2.202 Checking Linearity

The easiest method is visualization.

Plot

- the experimental data,
- the regression line.

If the data roughly follow a straight trend,

the assumption is reasonable.

If a strong curve appears,

consider a nonlinear model.

---

## Python Example

```python
import matplotlib.pyplot as plt

plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.show()
```

This is one reason why plotting the regression line, which we learned earlier, is so important.

---

# 2.203 Assumption 2 — Independence of Observations

The second assumption states that each observation should be independent of the others.

In other words,

one sample should not determine or influence another sample.

---

## Materials Science Example

Suppose we measure the hardness of

100 completely different steel samples.

These measurements are generally independent.

Now imagine we measure the same steel specimen

100 times without changing anything.

Those measurements are highly correlated.

Treating them as independent observations would artificially inflate the dataset without providing new information.

---

Independence is particularly important in time-series data,

where consecutive measurements often influence one another.

---

# 2.204 Assumption 3 — Homoscedasticity

The word

**homoscedasticity**

sounds intimidating,

but the concept is straightforward.

It means that the prediction errors should have approximately the same spread across all predicted values.

---

Suppose we plot

Predicted Value

versus

Residual.

A good model may produce a pattern resembling

```text
Residual

↑

  •  •   •

 • • • •

  • • •

--------------------→

Predicted
```

The spread is fairly constant.

---

Now consider

```text
Residual

↑

•

 •

  •

    ••

      ••••

          •••••••

--------------------→

Predicted
```

The spread increases dramatically.

This phenomenon is called

**heteroscedasticity**.

It violates one of the assumptions of Linear Regression.

---

# 2.205 Why Homoscedasticity Matters

When the variance of the errors changes across the prediction range,

the regression coefficients may still be unbiased,

but statistical interpretations,

such as confidence intervals and hypothesis tests,

become less reliable.

In practical machine learning,

heteroscedasticity often indicates that another model or a transformation of the data may produce better results.

---

# 2.206 Plotting Residuals

Residual plots are commonly used to diagnose this assumption.

First,

compute the residuals.

```python
residuals = y - model.predict(X)
```

Then plot them.

```python
plt.scatter(
    model.predict(X),
    residuals
)

plt.axhline(
    y=0,
    color="red"
)

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.show()
```

---

## Understanding the Code

### Residual Calculation

```python
residuals = y - model.predict(X)
```

Subtracts the predicted values from the experimental values.

Every residual measures one prediction error.

---

### Scatter Plot

```python
plt.scatter(...)
```

Each point corresponds to one material.

---

### Horizontal Line

```python
plt.axhline(y=0)
```

Draws a horizontal reference line where the residual equals zero.

Ideally,

the residuals should be randomly distributed around this line.

---

# 2.207 Assumption 4 — Normality of Residuals

Linear Regression also assumes that the residuals are approximately normally distributed.

Notice that **the target values themselves do not need to be normally distributed.**

It is the **residuals** that matter.

---

If we draw a histogram of the residuals,

it should resemble the familiar bell-shaped curve.

```text
          •

       •••••

     •••••••••

   ••••••••••••

     ••••••••

       ••••

          •
```

Small deviations are acceptable.

Large departures from normality may indicate

- missing variables,
- outliers,
- nonlinear relationships,
- incorrect model assumptions.

---

# 2.208 Checking Normality

A simple histogram provides a quick visual assessment.

```python
plt.hist(
    residuals,
    bins=20
)

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.show()
```

If the histogram roughly resembles a bell-shaped distribution,

the assumption is reasonably satisfied.

For more formal statistical testing,

researchers often use

- the Shapiro–Wilk test,
- the Anderson–Darling test,
- or Q–Q plots.

We will encounter these techniques later in the book.

---

# 2.209 Assumption 5 — Absence of Multicollinearity

The final assumption applies only when we have multiple features.

Multicollinearity occurs when two or more features contain nearly the same information.

---

## Example

Suppose our dataset contains

- Density
- Mass
- Volume

Since

\[
Density
=
\frac{Mass}{Volume},
\]

these variables are mathematically related.

Including all of them may confuse the regression model because it becomes difficult to determine which variable is actually responsible for the prediction.

---

Another example is

- lattice constant,
- unit cell volume.

For many crystal systems,

these quantities are strongly correlated.

Including both may introduce multicollinearity.

---

# 2.210 Why Multicollinearity Is a Problem

When multicollinearity is severe,

the regression coefficients become unstable.

A small change in the dataset may produce a large change in the learned coefficients.

Interestingly,

the predictions themselves may still appear accurate,

but interpreting the coefficients becomes unreliable.

This is particularly important in scientific research,

where we often wish to understand **which descriptors influence the material property**.

---

# 2.211 Measuring Multicollinearity

One commonly used metric is the

**Variance Inflation Factor (VIF).**

A large VIF indicates that a feature is highly correlated with other features.

As a rule of thumb,

- VIF ≈ 1 indicates little correlation,
- VIF between 1 and 5 is usually acceptable,
- VIF greater than 10 often suggests serious multicollinearity.

Later in the book,

we will compute VIF using Python and learn strategies for reducing multicollinearity.

---

# 2.212 Summary of the Assumptions

The five assumptions can be summarized as follows.

| Assumption | What It Means |
|------------|---------------|
| Linearity | The relationship between features and target is approximately linear. |
| Independence | Observations do not influence one another. |
| Homoscedasticity | Residuals have roughly constant variance. |
| Normality | Residuals are approximately normally distributed. |
| No Multicollinearity | Features are not excessively correlated with one another. |

These assumptions form the mathematical foundation of Linear Regression.

Fortunately,

many real materials science datasets satisfy them reasonably well,

or can be transformed so that they do.

---

# 2.213 Looking Ahead

You have now learned how Linear Regression works,

how to implement it,

how to evaluate it,

and the assumptions that govern its behavior.

However,

real experimental and computational datasets are rarely perfect.

They often contain

- missing values,
- incorrect measurements,
- duplicate entries,
- extreme outliers,
- inconsistent units.

Before training any machine learning model,

these issues must be addressed.

In the next section, we will study **data preprocessing and cleaning for Linear Regression**, learning professional techniques for preparing materials science datasets before model training. This stage is one of the most critical parts of every real-world machine learning project.

# 2.214 Data Preprocessing for Linear Regression

A machine learning model is only as good as the data used to train it.

This statement is repeated so often in machine learning that it has become almost a proverb:

> **Garbage in, garbage out.**

If poor-quality data are provided to even the most advanced machine learning algorithm, the resulting predictions will also be poor. Conversely, a relatively simple algorithm such as Linear Regression can often achieve excellent performance when trained on a clean, carefully prepared dataset.

For this reason, experienced data scientists typically spend far more time preparing data than training machine learning models. In many real-world projects, data preprocessing accounts for **60–80% of the total project time**, while model training occupies only a small fraction.

This chapter focuses on the preprocessing steps that should be performed before fitting a Linear Regression model.

---

# 2.215 Why Data Preprocessing Is Necessary

Real datasets are almost never perfect.

Whether the data originate from laboratory experiments, Density Functional Theory (DFT) calculations, the Materials Project database, AFLOW, OQMD, or industrial measurements, they usually contain imperfections.

Common problems include

- missing values,
- duplicated samples,
- inconsistent units,
- typing errors,
- impossible values,
- measurement noise,
- outliers,
- highly correlated variables.

Machine learning algorithms cannot determine whether a value is physically meaningful.

For example, suppose a dataset contains

| Density (g/cm³) |
|----------------:|
|7.85|
|7.82|
|−4.10|
|7.90|

A human immediately recognizes that a negative density is physically impossible.

Linear Regression, however, simply treats it as another number.

Unless such errors are detected and corrected, they can significantly reduce model accuracy.

---

# 2.216 The Complete Data Preprocessing Pipeline

Before training any machine learning model, it is useful to follow a systematic workflow.

```text
Raw Dataset

↓

Inspect the Data

↓

Handle Missing Values

↓

Remove Duplicate Samples

↓

Detect Outliers

↓

Correct Data Types

↓

Scale or Normalize Features (if needed)

↓

Feature Engineering

↓

Train the Model
```

Every step contributes to improving the reliability of the final model.

Skipping preprocessing is one of the most common mistakes made by beginners.

---

# 2.217 Loading a Dataset

Most machine learning projects begin by loading a dataset stored as a CSV file.

```python
import pandas as pd

df = pd.read_csv("steel_properties.csv")
```

---

## Understanding the Code

### Importing Pandas

```python
import pandas as pd
```

Imports the Pandas library using the conventional alias `pd`.

---

### Reading the CSV File

```python
pd.read_csv("steel_properties.csv")
```

Reads the CSV file into memory and converts it into a Pandas DataFrame.

---

### Assigning the Result

```python
df =
```

Stores the DataFrame inside the variable `df`.

Throughout this book, we will consistently use the name `df` for DataFrames.

---

# 2.218 Inspecting the Dataset

Before performing any preprocessing, always inspect the dataset.

The first few rows can be displayed using

```python
print(df.head())
```

Example output

```text
   Density  Carbon  GrainSize  Hardness
0     7.82     0.20         24       218
1     7.76     0.35         20       254
2     7.90     0.42         18       287
3     7.71     0.28         25       231
4     7.95     0.50         16       322
```

The method

```python
head()
```

returns the first five rows by default.

This provides a quick overview of the dataset.

---

# 2.219 Viewing Dataset Information

To obtain information about every column,

use

```python
df.info()
```

Example output

```text
<class 'pandas.core.frame.DataFrame'>

RangeIndex: 500 entries

Data columns (total 8 columns)

Density        float64

Carbon         float64

GrainSize      int64

Hardness       float64
```

This output provides valuable information including

- number of rows,
- number of columns,
- column names,
- data types,
- missing values.

Experienced data scientists almost always examine this information before beginning any analysis.

---

# 2.220 Obtaining Summary Statistics

A numerical summary of the dataset can be obtained using

```python
df.describe()
```

Example output

```text
           Density    Carbon   Hardness

count      500.00     500.00     500.00

mean         7.84       0.31     248.60

std          0.08       0.09      41.50

min          7.62       0.10     160.00

max          8.05       0.60     340.00
```

This table reports

- number of observations,
- mean,
- standard deviation,
- minimum,
- maximum,
- quartiles.

These statistics often reveal unusual values immediately.

For example, if the minimum density were

```text
−12.4
```

we would immediately suspect an error in the dataset.

---

# 2.221 Understanding the Statistics

Each quantity reported by `describe()` has a specific meaning.

**Count**

The number of non-missing observations.

**Mean**

The arithmetic average.

**Standard Deviation**

Measures how much the data vary around the mean.

**Minimum**

The smallest observed value.

**Maximum**

The largest observed value.

**25%, 50%, and 75%**

These values divide the data into four equal parts and are known as quartiles.

Quartiles are particularly useful for detecting outliers, a topic we will study shortly.

---

# 2.222 Developing Good Habits

Before writing even a single line of machine learning code, ask yourself the following questions.

- How many samples are available?
- How many features are present?
- Are there missing values?
- Are the units consistent?
- Do any values appear physically impossible?
- Are the numerical ranges reasonable?

Answering these questions at the beginning of a project often prevents hours of debugging later.

Professional machine learning practitioners rarely train a model immediately after loading a dataset. They first develop a thorough understanding of the data itself.


# 2.223 Handling Missing Values

One of the most common problems encountered in real-world datasets is the presence of **missing values**.

Missing values occur whenever information is unavailable for one or more observations.

This may happen because

- an experiment failed,
- an instrument malfunctioned,
- a measurement was forgotten,
- a simulation did not converge,
- or the value simply does not exist.

Missing values are unavoidable in scientific research.

The important question is not **whether** they exist, but **how to handle them correctly**.

---

# 2.224 Why Missing Values Are a Problem

Suppose we have the following dataset.

| Density | Carbon | Grain Size | Hardness |
|---------:|--------:|-----------:|---------:|
|7.82|0.20|24|218|
|7.76|—|20|254|
|7.90|0.42|18|287|
|—|0.28|25|231|
|7.95|0.50|16|322|

Two values are missing.

A human can immediately recognize the gaps.

Most machine learning algorithms cannot.

Linear Regression expects every feature to contain a valid numerical value.

If missing values are left untreated,

the model will usually produce an error instead of training successfully.

---

# 2.225 How Missing Values Are Represented

In a CSV file,

missing values may appear in several different forms.

```text
NaN
```

```text
NULL
```

```text
None
```

```text
(blank cell)
```

When Pandas reads a dataset,

most of these are automatically converted into the special value

```python
NaN
```

which stands for

**Not a Number**.

---

# 2.226 Detecting Missing Values

The first step is to determine whether the dataset contains missing values.

Pandas provides a simple function for this purpose.

```python
print(df.isnull())
```

---

## Understanding the Code

### `isnull()`

This function examines every cell in the DataFrame.

If a value is missing,

it returns

```python
True
```

Otherwise,

it returns

```python
False
```

Example output

```text
   Density  Carbon  GrainSize  Hardness

0    False   False      False      False

1    False    True      False      False

2    False   False      False      False

3     True   False      False      False

4    False   False      False      False
```

Here,

the missing values appear as

```text
True
```

---

# 2.227 Counting Missing Values

Displaying every missing value is useful for small datasets.

For larger datasets,

we usually count how many missing values exist in each column.

```python
print(df.isnull().sum())
```

Example output

```text
Density      1

Carbon       1

GrainSize    0

Hardness     0
```

---

## Understanding the Code

First,

```python
df.isnull()
```

creates a table of

```python
True

False
```

values.

Next,

```python
.sum()
```

treats

```python
True = 1

False = 0
```

and adds the values in each column.

The result is the total number of missing values.

---

# 2.228 Removing Rows with Missing Values

The simplest solution is to remove incomplete rows.

Pandas provides the function

```python
dropna()
```

```python
df = df.dropna()
```

---

## What Does This Do?

Every row containing at least one missing value is removed.

Suppose the original dataset contains

```text
500 rows
```

and

20 rows contain missing values.

After applying

```python
dropna()
```

the dataset contains

```text
480 rows
```

---

# 2.229 When Is Removing Rows Appropriate?

Removing rows is appropriate when

- only a small number of observations are missing,
- the dataset is sufficiently large,
- the missing values occur randomly.

For example,

removing

5 samples

from a dataset containing

10,000 samples

has almost no effect.

However,

removing

5 samples

from a dataset containing

30 samples

would discard a substantial fraction of the available information.

---

# 2.230 Filling Missing Values

Instead of deleting data,

we can replace missing values with estimated values.

This process is called

**imputation**.

One of the simplest approaches is replacing missing values with the column mean.

```python
df["Density"] = df["Density"].fillna(
    df["Density"].mean()
)
```

---

## Understanding Every Line

### Selecting the Column

```python
df["Density"]
```

Chooses the Density column.

---

### Computing the Mean

```python
.mean()
```

Calculates the average density.

Suppose the average is

```text
7.84
```

---

### Filling Missing Values

```python
fillna(...)
```

Replaces every missing value in the column with

```text
7.84
```

The remaining values remain unchanged.

---

# 2.231 Other Imputation Strategies

The mean is not always the best choice.

Other common strategies include

### Median Imputation

```python
df["Density"] = df["Density"].fillna(
    df["Density"].median()
)
```

The median is more robust when outliers are present.

---

### Mode Imputation

```python
df["Phase"] = df["Phase"].fillna(
    df["Phase"].mode()[0]
)
```

The mode is commonly used for categorical variables.

---

### Constant Value

Sometimes,

researchers intentionally replace missing values with a fixed number.

```python
df["Carbon"] = df["Carbon"].fillna(0)
```

This approach should only be used when the chosen value has a clear physical meaning.

---

# 2.232 Which Method Should You Choose?

There is no universal answer.

The appropriate method depends on

- the amount of missing data,
- the reason the values are missing,
- the scientific meaning of the variable.

As a general guideline,

| Situation | Recommended Approach |
|-----------|----------------------|
| Very few missing values | Remove the affected rows |
| Numerical feature | Mean or median imputation |
| Categorical feature | Mode imputation |
| Large proportion of missing values | Investigate whether the feature should be removed entirely |

Blindly filling every missing value with the mean is rarely the best scientific choice.

Always consider the physical meaning of the data.

---

# 2.233 Missing Values in Materials Informatics

Missing values frequently appear in computational materials science.

For example,

suppose a database contains

- density,
- bulk modulus,
- band gap,
- magnetic moment.

A DFT calculation may successfully compute the density but fail to converge for the band gap.

The resulting dataset naturally contains missing entries.

Similarly,

experimental datasets often contain incomplete measurements because not every property is measured for every sample.

Handling these situations correctly is an essential part of building reliable machine learning models.

---

# 2.234 Best Practice

Before training any model,

always perform the following checks.

1. Detect missing values.

2. Count the number of missing values in each feature.

3. Decide whether to remove or impute them.

4. Verify that no missing values remain.

A simple verification step is

```python
print(df.isnull().sum())
```

If every column reports

```text
0
```

then the dataset is ready for the next stage of preprocessing.


# 2.235 Detecting and Removing Duplicate Data

Another common problem in real-world datasets is the presence of **duplicate observations**.

A duplicate occurs when the same sample appears more than once in the dataset.

This may happen for several reasons.

- The dataset was merged from multiple sources.
- The same experiment was entered twice.
- A data entry mistake occurred.
- Database synchronization created repeated records.

Unlike missing values, duplicate entries are often difficult to notice by simply looking at the dataset.

Nevertheless, they can significantly influence machine learning models.

---

# 2.236 Why Are Duplicates a Problem?

Suppose we have the following dataset.

| Density | Carbon | Grain Size | Hardness |
|---------:|--------:|-----------:|---------:|
|7.82|0.20|24|218|
|7.76|0.35|20|254|
|7.90|0.42|18|287|
|7.90|0.42|18|287|
|7.71|0.28|25|231|

Notice that the third and fourth rows are identical.

If this duplicate remains in the dataset, the Linear Regression model effectively "sees" that material twice.

As a result, that single material receives **twice as much influence** during training compared to the other materials.

This can bias the learned coefficients, especially when the dataset is small.

---

# 2.237 Detecting Duplicate Rows

Pandas provides a convenient function for identifying duplicate observations.

```python
print(df.duplicated())
```

Example output

```text
0    False

1    False

2    False

3     True

4    False
```

Every row marked

```text
True
```

is considered a duplicate of an earlier row.

---

# 2.238 Understanding the Code

### `duplicated()`

This function compares every row with all previous rows.

If a row has already appeared earlier in the DataFrame,

the function returns

```python
True
```

Otherwise,

it returns

```python
False
```

Only the later occurrence is marked as a duplicate.

The first occurrence is treated as the original observation.

---

# 2.239 Counting Duplicate Rows

Instead of displaying every duplicate,

we often want to know how many duplicates exist.

```python
print(df.duplicated().sum())
```

Example output

```text
12
```

This indicates that the dataset contains twelve duplicated rows.

Counting duplicates should be one of the first quality checks performed after loading any dataset.

---

# 2.240 Removing Duplicate Rows

Once duplicates have been identified,

they can be removed easily.

```python
df = df.drop_duplicates()
```

---

## Understanding the Code

### `drop_duplicates()`

This function removes repeated rows while preserving the first occurrence.

Suppose the dataset originally contained

```text
500 rows
```

including

```text
15 duplicate rows.
```

After executing

```python
drop_duplicates()
```

the DataFrame contains

```text
485 unique rows.
```

---

# 2.241 Should Every Duplicate Be Removed?

Not always.

Imagine performing ten independent hardness measurements on the **same alloy**.

Those measurements may legitimately produce identical values.

They are repeated experiments rather than accidental duplicates.

In such cases,

removing them would discard valuable experimental information.

Therefore,

before deleting duplicates,

always ask

> **Does this duplicate represent a data-entry mistake or a genuine repeated measurement?**

Scientific understanding should always guide preprocessing decisions.

---

# 2.242 Inconsistent Data

Another common problem is **inconsistent data**.

Unlike duplicate data,

inconsistent data are not repeated.

Instead,

different observations follow different conventions.

For example,

suppose density is recorded as

| Sample | Density |
|---------|---------:|
|A|7.85|
|B|7850|
|C|7.90|
|D|7900|

At first glance,

all values appear numerical.

However,

two samples are measured in

```text
g/cm³
```

while the other two are measured in

```text
kg/m³.
```

Although both units are scientifically correct,

mixing them in the same dataset is a serious error.

Machine learning algorithms have no understanding of physical units.

They simply interpret the numbers.

---

# 2.243 Another Example of Inconsistent Data

Suppose temperature is recorded as

| Sample | Temperature |
|---------|------------:|
|A|25|
|B|30|
|C|298|
|D|310|

Again,

the problem is not obvious.

The first two values are recorded in

```text
°C
```

while the remaining values are recorded in

```text
K.
```

Without converting them to a common unit,

the dataset becomes physically inconsistent.

---

# 2.244 Why Unit Consistency Matters

Machine learning assumes that every feature has the same meaning throughout the dataset.

If one density value is measured in

```text
g/cm³
```

and another in

```text
kg/m³,
```

the algorithm interprets them as very different physical quantities,

even though they describe the same property.

Such inconsistencies can severely degrade model performance.

For this reason,

always verify that

- units,
- measurement scales,
- and conventions

are consistent before training any model.

---

# 2.245 Correcting Units

Suppose density is stored in

```text
kg/m³
```

but we want

```text
g/cm³.
```

The conversion is

```text
1 g/cm³ = 1000 kg/m³
```

In Python,

we divide by

```python
1000
```

```python
df["Density"] = df["Density"] / 1000
```

Now every observation uses the same unit.

---

# 2.246 Correcting Data Types

Sometimes,

numerical values are accidentally stored as text.

For example,

the Carbon column might contain

```text
"0.25"

"0.30"

"0.40"
```

instead of numerical values.

Pandas allows us to convert the data type.

```python
df["Carbon"] = df["Carbon"].astype(float)
```

---

## Understanding the Code

### `astype(float)`

Converts every value in the selected column into a floating-point number.

After conversion,

mathematical operations such as

- addition,
- multiplication,
- averaging,

can be performed correctly.

---

# 2.247 Checking Data Types

To verify the data types,

use

```python
print(df.dtypes)
```

Example output

```text
Density      float64

Carbon       float64

GrainSize      int64

Hardness     float64
```

This confirms that every numerical column has an appropriate numerical type.

---

# 2.248 Best Practices

Before training any machine learning model,

always perform the following checks.

- Search for duplicate observations.
- Remove accidental duplicates.
- Verify that repeated measurements are genuine before deleting them.
- Ensure that all units are consistent.
- Confirm that numerical columns use numerical data types.
- Check for impossible physical values.

These simple checks often eliminate subtle problems that would otherwise reduce model performance or lead to incorrect scientific conclusions.

Good preprocessing is not merely about writing Python code.

It is about ensuring that the dataset accurately represents the underlying physical system before any machine learning algorithm is allowed to learn from it.


# 2.249 Detecting and Handling Outliers

After removing missing values and duplicate observations, the next important step is identifying **outliers**.

Outliers are observations that differ significantly from the rest of the dataset.

They are not necessarily incorrect.

Sometimes they represent

- measurement errors,
- experimental mistakes,
- data-entry errors,

but sometimes they correspond to genuinely unusual materials with exceptional properties.

The challenge is to distinguish between these possibilities.

A careless removal of outliers can eliminate valuable scientific discoveries, while ignoring erroneous outliers can seriously reduce model performance.

---

# 2.250 What Is an Outlier?

Consider the following hardness measurements.

```text
210

218

225

231

240

228

222

217

950
```

Almost every value lies between

210 HV

and

240 HV.

The value

```text
950 HV
```

is dramatically different from the others.

This observation is an outlier.

Before deciding what to do with it, we must investigate **why** it is different.

---

# 2.251 Where Do Outliers Come From?

Outliers can arise for many reasons.

### Measurement Errors

An instrument may malfunction during an experiment.

For example,

a hardness tester may have been improperly calibrated.

---

### Data Entry Errors

Suppose the actual hardness was

```text
295 HV
```

but someone accidentally typed

```text
2950 HV.
```

This creates an artificial outlier.

---

### Unit Conversion Errors

A density measured in

```text
kg/m³
```

may accidentally be mixed with values in

```text
g/cm³.
```

This creates extremely large numerical differences.

---

### Genuine Scientific Discoveries

Sometimes an outlier is not an error at all.

A newly developed alloy may truly possess exceptional hardness.

Removing such a sample would eliminate precisely the kind of material we hope machine learning will help us discover.

For this reason,

outliers should never be removed automatically.

---

# 2.252 Visualizing Outliers

One of the simplest ways to identify potential outliers is by plotting the data.

A commonly used visualization is the **box plot**.

```python
import matplotlib.pyplot as plt

plt.boxplot(df["Hardness"])

plt.ylabel("Hardness (HV)")

plt.show()
```

The box plot summarizes the distribution of the data and highlights unusually large or small observations.

---

# 2.253 Understanding the Code

### `plt.boxplot()`

Creates a box-and-whisker plot for the selected feature.

The function automatically identifies observations that lie far from the majority of the data.

---

### `plt.ylabel()`

Adds a label to the vertical axis.

Always include informative axis labels so that anyone reading your figures immediately understands what is being plotted.

---

### `plt.show()`

Displays the completed figure.

---

# 2.254 Understanding a Box Plot

A box plot consists of several components.

```text
        •

        |

   -----------

   |         |

   |   Box   |

   |         |

   -----------

      |   |

```

The **box** contains the middle 50% of the data.

The horizontal line inside the box represents the median.

The whiskers extend toward the smallest and largest observations that are not considered outliers.

Points beyond the whiskers are plotted individually and are treated as potential outliers.

Notice the word **potential**.

The plot only identifies unusual observations.

It does not determine whether they are errors.

---

# 2.255 The Interquartile Range (IQR)

Box plots identify outliers using the **Interquartile Range**, abbreviated as **IQR**.

Recall that

- the first quartile is denoted by \(Q_1\),
- the third quartile is denoted by \(Q_3\).

The IQR is defined as

\[
IQR = Q_3 - Q_1
\]

The IQR measures the spread of the middle 50% of the data.

Unlike the standard deviation,

it is relatively insensitive to extreme observations.

---

# 2.256 The Outlier Rule

Using the IQR,

we define two limits.

The lower limit is

\[
Q_1 - 1.5 \times IQR
\]

The upper limit is

\[
Q_3 + 1.5 \times IQR
\]

Any observation outside these limits is considered a potential outlier.

This rule is widely used in statistics and machine learning.

However,

it should always be interpreted together with scientific knowledge.

---

# 2.257 Detecting Outliers in Python

First,

calculate the quartiles.

```python
Q1 = df["Hardness"].quantile(0.25)

Q3 = df["Hardness"].quantile(0.75)
```

Next,

compute the interquartile range.

```python
IQR = Q3 - Q1
```

Now calculate the limits.

```python
lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR
```

Finally,

identify the outliers.

```python
outliers = df[
    (df["Hardness"] < lower_limit) |
    (df["Hardness"] > upper_limit)
]
```

---

# 2.258 Explaining Every Line of Code

### `quantile(0.25)`

Returns the first quartile.

Twenty-five percent of the observations lie below this value.

---

### `quantile(0.75)`

Returns the third quartile.

Seventy-five percent of the observations lie below this value.

---

### `IQR = Q3 - Q1`

Computes the spread of the middle half of the dataset.

---

### Logical Conditions

```python
df["Hardness"] < lower_limit
```

Selects values that are unusually small.

```python
df["Hardness"] > upper_limit
```

Selects values that are unusually large.

The vertical bar

```python
|
```

means **OR**.

Therefore,

the code selects observations that satisfy either condition.

---

# 2.259 Should Outliers Always Be Removed?

No.

This is one of the most important principles in scientific machine learning.

Before removing an outlier,

ask the following questions.

- Is the value physically possible?
- Was there a measurement error?
- Was the data entered correctly?
- Is the unit correct?
- Does the value represent an unusual but genuine material?

Only after answering these questions should you decide whether the observation should remain in the dataset.

Blindly removing outliers simply because they look unusual is poor scientific practice.

---

# 2.260 Removing Confirmed Erroneous Outliers

If you have confirmed that an outlier is caused by an error,

it can be removed.

```python
df = df[
    (df["Hardness"] >= lower_limit) &
    (df["Hardness"] <= upper_limit)
]
```

The ampersand

```python
&
```

means **AND**.

Only observations satisfying both conditions are retained.

---

# 2.261 Outliers in Materials Informatics

Outliers are particularly common in materials science.

Imagine a database containing thousands of materials.

Most metals have thermal conductivities between

20 and 250 W·m⁻¹·K⁻¹.

Diamond,

however,

has an exceptionally high thermal conductivity.

Compared with ordinary metals,

diamond appears to be an extreme outlier.

Yet it is not an error.

It is one of the most scientifically interesting materials in the dataset.

Removing such observations would prevent the machine learning model from learning about exceptional materials.

This illustrates why scientific understanding must always accompany statistical methods.

---

# 2.262 Practical Guidelines

When working with outliers,

follow these principles.

1. Detect potential outliers using visualization and statistical methods.

2. Investigate each unusual observation carefully.

3. Remove observations only when there is clear evidence of an error.

4. Preserve scientifically meaningful extreme values.

5. Document every preprocessing decision so that your analysis remains reproducible and transparent.

A machine learning model should learn from reality—not from an artificially cleaned dataset that no longer reflects the true behavior of materials.


# 2.263 Feature Scaling and Normalization

At this stage, our dataset has become much cleaner.

We have

- handled missing values,
- removed duplicate observations,
- corrected inconsistent data,
- investigated outliers.

The next preprocessing step concerns the **numerical scale of the features**.

Different features often have very different ranges.

For example, consider a dataset containing the following descriptors.

| Feature | Typical Values |
|---------|---------------:|
| Density (g/cm³) | 7.8 |
| Carbon Content (%) | 0.25 |
| Grain Size (µm) | 20 |
| Young's Modulus (GPa) | 210 |
| Electrical Conductivity (S/m) | 5,800,000 |

Notice how dramatically the magnitudes differ.

Electrical conductivity is measured in millions,

while carbon content is less than one.

Although all of these values are scientifically correct,

their large differences in scale can create problems for certain machine learning algorithms.

The process of bringing features onto comparable numerical scales is called **feature scaling**.

---

# 2.264 Does Linear Regression Require Feature Scaling?

This is an important question.

The answer is

**not necessarily.**

Ordinary Linear Regression computes its solution analytically using the least-squares method.

As a result,

it is generally much less sensitive to feature scaling than algorithms based on distances or gradient optimization.

However,

there are several important reasons why scaling is still useful.

- It improves numerical stability.
- It makes regression coefficients easier to compare.
- It becomes essential for algorithms such as Support Vector Machines, Neural Networks, K-Nearest Neighbors, Principal Component Analysis, and regularized regression methods like Ridge and Lasso.

Because later chapters of this book will cover all of these algorithms,

it is good practice to understand feature scaling now.

---

# 2.265 Two Common Scaling Methods

The two most widely used scaling techniques are

1. Standardization

2. Normalization

Although these terms are often used interchangeably in casual conversation,

they represent different mathematical operations.

We will study both.

---

# 2.266 Standardization

Standardization transforms a feature so that

- its mean becomes zero,
- its standard deviation becomes one.

The transformed value is called the **standard score** or **Z-score**.

The formula is

\[
z =
\frac{x-\mu}{\sigma}
\]

where

- \(x\) is the original value,
- \(\mu\) is the mean of the feature,
- \(\sigma\) is the standard deviation.

After standardization,

most values lie roughly between

-3

and

+3.

---

# 2.267 Understanding the Formula

Suppose the average density of a dataset is

```text
7.80 g/cm³
```

and the standard deviation is

```text
0.10 g/cm³.
```

Now consider a material with a density of

```text
7.95 g/cm³.
```

The standardized value becomes

\[
z=
\frac{7.95-7.80}{0.10}
=
1.5
\]

This means the material's density is

**1.5 standard deviations above the mean**.

Notice that the units disappear.

Standardized features are dimensionless,

making variables with different units directly comparable.

---

# 2.268 Standardization in Python

Scikit-learn provides the `StandardScaler` class.

First,

import it.

```python
from sklearn.preprocessing import StandardScaler
```

Create a scaler object.

```python
scaler = StandardScaler()
```

Now transform the feature matrix.

```python
X_scaled = scaler.fit_transform(X)
```

---

# 2.269 Explaining Every Line of Code

### Import Statement

```python
from sklearn.preprocessing import StandardScaler
```

Imports the scaling class from Scikit-learn.

---

### Creating the Scaler

```python
scaler = StandardScaler()
```

Creates an object that knows how to perform standardization.

At this stage,

no calculations have been performed.

---

### `fit_transform()`

```python
X_scaled = scaler.fit_transform(X)
```

This command performs two operations.

First,

`fit()`

computes the mean and standard deviation of every feature.

Second,

`transform()`

applies the standardization formula to every value.

The transformed data are stored in

```python
X_scaled
```

while the original data remain unchanged.

---

# 2.270 Why Use `fit_transform()` Instead of Calling Two Functions?

You could write

```python
scaler.fit(X)

X_scaled = scaler.transform(X)
```

This produces exactly the same result.

However,

because these two operations are commonly performed together,

Scikit-learn provides the convenient method

```python
fit_transform()
```

which combines both steps into a single command.

---

# 2.271 Normalization

Normalization is different from standardization.

Instead of making the mean equal to zero,

normalization rescales every value into a fixed range.

The most common range is

\[
0
\]

to

\[
1.
\]

The formula is

\[
x_{new}
=
\frac{x-x_{min}}
{x_{max}-x_{min}}
\]

where

- \(x_{min}\) is the smallest value,
- \(x_{max}\) is the largest value.

After normalization,

the smallest observation becomes

0,

the largest becomes

1,

and every other value lies between them.

---

# 2.272 Example of Normalization

Suppose grain size varies between

```text
10 µm
```

and

```text
30 µm.
```

A grain size of

```text
20 µm
```

becomes

\[
\frac{20-10}{30-10}
=
0.5
\]

The original value

20 µm

is therefore mapped to

0.5.

Again,

the units disappear.

---

# 2.273 Normalization in Python

Normalization is performed using

`MinMaxScaler`.

```python
from sklearn.preprocessing import MinMaxScaler
```

Create the scaler.

```python
scaler = MinMaxScaler()
```

Transform the data.

```python
X_scaled = scaler.fit_transform(X)
```

Notice that the workflow is almost identical to `StandardScaler`.

Only the mathematical transformation changes.

---

# 2.274 Standardization vs. Normalization

The two methods have different objectives.

| Standardization | Normalization |
|-----------------|---------------|
| Mean becomes 0 | Minimum becomes 0 |
| Standard deviation becomes 1 | Maximum becomes 1 |
| Values may be negative | Values remain between 0 and 1 |
| Good for Gaussian-like data | Good for bounded ranges |

Neither method is universally superior.

The appropriate choice depends on the machine learning algorithm and the characteristics of the dataset.

---

# 2.275 Should the Test Set Be Scaled Separately?

This is one of the most common beginner mistakes.

The answer is

**No.**

The scaler must learn only from the **training data**.

First,

fit the scaler using the training set.

```python
scaler.fit(X_train)
```

Then,

transform the training set.

```python
X_train = scaler.transform(X_train)
```

Finally,

use the **same scaler** to transform the test set.

```python
X_test = scaler.transform(X_test)
```

Notice that we do **not** call

```python
fit()
```

on the test data.

Doing so would leak information from the test set into the training process,

leading to overly optimistic evaluation results.

This problem is known as **data leakage**, and avoiding it is an essential principle of machine learning.

---

# 2.276 Scaling in Materials Informatics

Materials science datasets often contain descriptors with vastly different magnitudes.

For example,

- density may be around 5–20,
- lattice parameters around 2–10,
- formation energy between −10 and 5,
- electrical conductivity may exceed one million,
- elastic constants may be several hundred.

Without scaling,

algorithms that depend on distances or optimization may become dominated by the largest numerical features rather than the most physically informative ones.

Proper feature scaling ensures that every descriptor contributes appropriately during model training.

Although ordinary Linear Regression can often perform well without scaling,

developing the habit of applying scaling correctly will prepare you for the more advanced machine learning methods that follow later in this book.


# 2.277 Feature Engineering and Feature Selection

So far, we have focused on cleaning the dataset.

We have

- handled missing values,
- removed duplicates,
- corrected inconsistencies,
- investigated outliers,
- learned when and how to scale features.

At this point, many beginners believe the dataset is ready for machine learning.

In reality, one of the most important stages still remains.

This stage is called **feature engineering**.

Experienced machine learning practitioners often say

> **Better features are usually more valuable than more complicated algorithms.**

A carefully engineered feature can improve model performance far more than replacing Linear Regression with a more sophisticated algorithm.

---

# 2.278 What Is a Feature?

Before discussing feature engineering, let us clearly define the word **feature**.

A feature is any measurable property used by the machine learning model as an input.

For example,

when predicting the hardness of steel,

possible features include

- carbon content,
- density,
- grain size,
- chromium concentration,
- nickel concentration,
- heat-treatment temperature.

The target variable,

such as hardness,

is **not** a feature.

It is the quantity the model attempts to predict.

---

# 2.279 What Is Feature Engineering?

Feature engineering is the process of creating new features from existing information.

Instead of collecting new experimental data,

we derive additional descriptors that may better capture the underlying physical relationships.

The goal is to provide the machine learning model with more informative inputs.

---

# 2.280 A Simple Example

Suppose our dataset contains

| Length (mm) | Width (mm) |
|-------------|-----------:|
|20|10|
|25|12|
|18|8|

These are useful features,

but we might also create

```text
Area = Length × Width
```

The new feature may describe the object better than either length or width individually.

The machine learning algorithm can then decide whether this engineered feature improves prediction accuracy.

---

# 2.281 Feature Engineering in Materials Science

Feature engineering is particularly important in materials informatics.

Suppose our dataset contains

- lattice constant,
- atomic mass,
- density.

Instead of using only these quantities,

we may derive additional physically meaningful descriptors,

such as

- atomic volume,
- packing density,
- valence electron concentration,
- average electronegativity,
- atomic size mismatch,
- configurational entropy.

These engineered descriptors often capture the physics of the material more effectively than the original measurements.

This is one reason why domain knowledge is so valuable in scientific machine learning.

---

# 2.282 Using Pymatgen for Feature Engineering

Because you already know **Pymatgen**, it becomes one of the most powerful tools for feature engineering.

Instead of manually calculating descriptors,

Pymatgen can automatically extract structural and compositional information from crystal structures.

For example,

suppose we have a CIF file.

```python
from pymatgen.core import Structure

structure = Structure.from_file(
    "steel.cif"
)
```

This loads the crystal structure into Python.

From this structure,

we can compute many descriptors that later become machine learning features.

---

# 2.283 Extracting Simple Structural Information

Suppose we want the crystal density.

```python
density = structure.density
```

---

## Understanding the Code

### `structure`

Represents the crystal loaded from the CIF file.

---

### `.density`

Calculates the theoretical density directly from

- lattice parameters,
- atomic positions,
- atomic masses.

Instead of manually computing the density,

Pymatgen performs the calculation automatically.

The resulting value can be added to the machine learning dataset as a feature.

---

# 2.284 Another Example

Suppose we wish to know the unit-cell volume.

```python
volume = structure.volume
```

Again,

this numerical value can become another feature.

By repeating this process,

we gradually transform a crystal structure into a numerical feature vector suitable for machine learning.

This conversion from physical structure to numerical descriptors is one of the central ideas of materials informatics.

---

# 2.285 From DFT to Machine Learning

Quantum ESPRESSO and other DFT packages generate a large amount of scientifically valuable information.

Examples include

- total energy,
- lattice parameters,
- magnetic moments,
- electronic band gap,
- elastic constants,
- charge density,
- density of states.

These quantities should not simply remain in output files.

They can be extracted and incorporated into machine learning datasets.

For example,

a workflow may proceed as follows.

```text
Quantum ESPRESSO Calculation

↓

Optimized Crystal Structure

↓

Pymatgen Reads the Output

↓

Physical Descriptors Are Calculated

↓

Descriptors Become Machine Learning Features

↓

Train Regression Model
```

This workflow represents one of the most common pipelines used in modern computational materials science.

---

# 2.286 What Is Feature Selection?

As datasets become larger,

they often contain many features.

Some may contribute useful information.

Others may be irrelevant.

Feature selection is the process of identifying the most informative features while discarding those that contribute little or no predictive value.

Removing unnecessary features often

- improves model performance,
- reduces training time,
- simplifies interpretation,
- decreases the risk of overfitting.

---

# 2.287 Why More Features Are Not Always Better

A common misconception is

> "If ten features are good, then one hundred features must be even better."

This is not necessarily true.

Imagine predicting hardness using

- grain size,
- carbon content,
- density,

along with

- the student's identification number,
- laboratory room number,
- sample storage shelf.

The last three variables have no physical relationship to hardness.

Including them only introduces noise into the model.

A smaller set of informative features often performs better than a much larger set containing irrelevant information.

---

# 2.288 Correlation-Based Feature Selection

One simple approach is to examine the correlation between each feature and the target variable.

Highly correlated features are often more useful for prediction.

Pandas provides an easy way to compute correlations.

```python
correlation = df.corr()
```

This produces a correlation matrix showing how strongly every pair of variables is related.

Later in this chapter,

we will learn how to interpret this matrix in detail.

---

# 2.289 Removing Highly Correlated Features

Sometimes,

two features contain almost identical information.

For example,

suppose a dataset contains

- unit-cell volume,
- lattice constant,

for a cubic crystal.

These variables are mathematically related.

Keeping both may introduce multicollinearity.

In such situations,

removing one of the features often simplifies the model without reducing predictive performance.

---

# 2.290 Domain Knowledge Is Essential

Unlike many general machine learning applications,

materials informatics depends heavily on scientific understanding.

Two descriptors may appear statistically similar,

yet one may have far greater physical significance.

For example,

average electronegativity and valence electron concentration may both correlate with hardness,

but they describe different physical mechanisms.

Feature selection should therefore combine

- statistical analysis,
- machine learning,
- and materials science expertise.

The best models emerge when all three perspectives are considered together.

---

# 2.291 A Typical Materials Informatics Workflow

A modern materials informatics project often follows this sequence.

```text
Experimental Data
        +
DFT Calculations
        +
Materials Databases

↓

Pymatgen Extracts Structural Descriptors

↓

Additional Features Are Engineered

↓

Irrelevant Features Are Removed

↓

Machine Learning Model Is Trained

↓

Material Properties Are Predicted
```

This workflow forms the foundation of many published studies in computational materials science and materials discovery.

By learning feature engineering early, you are building one of the most valuable skills required for applying machine learning to real materials research.


# 2.277 Feature Engineering and Feature Selection

So far, we have focused on cleaning the dataset.

We have

- handled missing values,
- removed duplicates,
- corrected inconsistencies,
- investigated outliers,
- learned when and how to scale features.

At this point, many beginners believe the dataset is ready for machine learning.

In reality, one of the most important stages still remains.

This stage is called **feature engineering**.

Experienced machine learning practitioners often say

> **Better features are usually more valuable than more complicated algorithms.**

A carefully engineered feature can improve model performance far more than replacing Linear Regression with a more sophisticated algorithm.

---

# 2.278 What Is a Feature?

Before discussing feature engineering, let us clearly define the word **feature**.

A feature is any measurable property used by the machine learning model as an input.

For example,

when predicting the hardness of steel,

possible features include

- carbon content,
- density,
- grain size,
- chromium concentration,
- nickel concentration,
- heat-treatment temperature.

The target variable,

such as hardness,

is **not** a feature.

It is the quantity the model attempts to predict.

---

# 2.279 What Is Feature Engineering?

Feature engineering is the process of creating new features from existing information.

Instead of collecting new experimental data,

we derive additional descriptors that may better capture the underlying physical relationships.

The goal is to provide the machine learning model with more informative inputs.

---

# 2.280 A Simple Example

Suppose our dataset contains

| Length (mm) | Width (mm) |
|-------------|-----------:|
|20|10|
|25|12|
|18|8|

These are useful features,

but we might also create

```text
Area = Length × Width
```

The new feature may describe the object better than either length or width individually.

The machine learning algorithm can then decide whether this engineered feature improves prediction accuracy.

---

# 2.281 Feature Engineering in Materials Science

Feature engineering is particularly important in materials informatics.

Suppose our dataset contains

- lattice constant,
- atomic mass,
- density.

Instead of using only these quantities,

we may derive additional physically meaningful descriptors,

such as

- atomic volume,
- packing density,
- valence electron concentration,
- average electronegativity,
- atomic size mismatch,
- configurational entropy.

These engineered descriptors often capture the physics of the material more effectively than the original measurements.

This is one reason why domain knowledge is so valuable in scientific machine learning.

---

# 2.282 Using Pymatgen for Feature Engineering

Because you already know **Pymatgen**, it becomes one of the most powerful tools for feature engineering.

Instead of manually calculating descriptors,

Pymatgen can automatically extract structural and compositional information from crystal structures.

For example,

suppose we have a CIF file.

```python
from pymatgen.core import Structure

structure = Structure.from_file(
    "steel.cif"
)
```

This loads the crystal structure into Python.

From this structure,

we can compute many descriptors that later become machine learning features.

---

# 2.283 Extracting Simple Structural Information

Suppose we want the crystal density.

```python
density = structure.density
```

---

## Understanding the Code

### `structure`

Represents the crystal loaded from the CIF file.

---

### `.density`

Calculates the theoretical density directly from

- lattice parameters,
- atomic positions,
- atomic masses.

Instead of manually computing the density,

Pymatgen performs the calculation automatically.

The resulting value can be added to the machine learning dataset as a feature.

---

# 2.284 Another Example

Suppose we wish to know the unit-cell volume.

```python
volume = structure.volume
```

Again,

this numerical value can become another feature.

By repeating this process,

we gradually transform a crystal structure into a numerical feature vector suitable for machine learning.

This conversion from physical structure to numerical descriptors is one of the central ideas of materials informatics.

---

# 2.285 From DFT to Machine Learning

Quantum ESPRESSO and other DFT packages generate a large amount of scientifically valuable information.

Examples include

- total energy,
- lattice parameters,
- magnetic moments,
- electronic band gap,
- elastic constants,
- charge density,
- density of states.

These quantities should not simply remain in output files.

They can be extracted and incorporated into machine learning datasets.

For example,

a workflow may proceed as follows.

```text
Quantum ESPRESSO Calculation

↓

Optimized Crystal Structure

↓

Pymatgen Reads the Output

↓

Physical Descriptors Are Calculated

↓

Descriptors Become Machine Learning Features

↓

Train Regression Model
```

This workflow represents one of the most common pipelines used in modern computational materials science.

---

# 2.286 What Is Feature Selection?

As datasets become larger,

they often contain many features.

Some may contribute useful information.

Others may be irrelevant.

Feature selection is the process of identifying the most informative features while discarding those that contribute little or no predictive value.

Removing unnecessary features often

- improves model performance,
- reduces training time,
- simplifies interpretation,
- decreases the risk of overfitting.

---

# 2.287 Why More Features Are Not Always Better

A common misconception is

> "If ten features are good, then one hundred features must be even better."

This is not necessarily true.

Imagine predicting hardness using

- grain size,
- carbon content,
- density,

along with

- the student's identification number,
- laboratory room number,
- sample storage shelf.

The last three variables have no physical relationship to hardness.

Including them only introduces noise into the model.

A smaller set of informative features often performs better than a much larger set containing irrelevant information.

---

# 2.288 Correlation-Based Feature Selection

One simple approach is to examine the correlation between each feature and the target variable.

Highly correlated features are often more useful for prediction.

Pandas provides an easy way to compute correlations.

```python
correlation = df.corr()
```

This produces a correlation matrix showing how strongly every pair of variables is related.

Later in this chapter,

we will learn how to interpret this matrix in detail.

---

# 2.289 Removing Highly Correlated Features

Sometimes,

two features contain almost identical information.

For example,

suppose a dataset contains

- unit-cell volume,
- lattice constant,

for a cubic crystal.

These variables are mathematically related.

Keeping both may introduce multicollinearity.

In such situations,

removing one of the features often simplifies the model without reducing predictive performance.

---

# 2.290 Domain Knowledge Is Essential

Unlike many general machine learning applications,

materials informatics depends heavily on scientific understanding.

Two descriptors may appear statistically similar,

yet one may have far greater physical significance.

For example,

average electronegativity and valence electron concentration may both correlate with hardness,

but they describe different physical mechanisms.

Feature selection should therefore combine

- statistical analysis,
- machine learning,
- and materials science expertise.

The best models emerge when all three perspectives are considered together.

---

# 2.291 A Typical Materials Informatics Workflow

A modern materials informatics project often follows this sequence.

```text
Experimental Data
        +
DFT Calculations
        +
Materials Databases

↓

Pymatgen Extracts Structural Descriptors

↓

Additional Features Are Engineered

↓

Irrelevant Features Are Removed

↓

Machine Learning Model Is Trained

↓

Material Properties Are Predicted
```

This workflow forms the foundation of many published studies in computational materials science and materials discovery.

By learning feature engineering early, you are building one of the most valuable skills required for applying machine learning to real materials research.


# 2.310 Common Beginner Mistakes in Linear Regression

By now, you understand

- the mathematical foundations of Linear Regression,
- how the algorithm learns,
- how to implement it in Python,
- how to evaluate its performance,
- and how to prepare data before training.

At this stage, it is worth discussing some of the mistakes that almost every beginner makes.

These mistakes are not limited to students.

Many research papers and industrial projects have reported misleading results because one or more of these principles were ignored.

Learning to avoid these pitfalls is an important step toward becoming a skilled machine learning practitioner.

---

# 2.311 Mistake 1 — Evaluating on the Training Data

Perhaps the most common mistake is evaluating the model using the same data that were used for training.

For example,

```python
model.fit(X, y)

predictions = model.predict(X)
```

Although this code executes successfully,

it does not tell us how well the model performs on unseen data.

Instead,

the model is being tested on the very examples from which it learned.

This often produces unrealistically optimistic performance metrics.

The correct approach is

```python
model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)
```

Always evaluate using data that the model has never encountered during training.

---

# 2.312 Mistake 2 — Ignoring Missing Values

Many beginners attempt to train a model immediately after loading a dataset.

For example,

```python
df = pd.read_csv(
    "data.csv"
)

model.fit(X, y)
```

If the dataset contains missing values,

the training process may fail or produce unreliable results.

Always inspect the dataset first.

```python
print(df.isnull().sum())
```

Only after handling missing values should model training begin.

---

# 2.313 Mistake 3 — Ignoring Units

Machine learning algorithms process numbers,

not physical units.

Suppose one density value is recorded as

```text
7.85 g/cm³
```

while another is stored as

```text
7850 kg/m³.
```

Scientifically,

these values represent the same density.

Numerically,

they differ by a factor of one thousand.

If such inconsistencies remain in the dataset,

the learned model may become unreliable.

Always verify that every feature uses a consistent unit system.

---

# 2.314 Mistake 4 — Forgetting Feature Scaling When It Is Needed

Although ordinary Linear Regression often performs well without scaling,

many later machine learning algorithms do not.

Some beginners become accustomed to never scaling their data.

Later,

when using Support Vector Machines or Neural Networks,

their models perform poorly because the features remain on incompatible numerical scales.

Developing good preprocessing habits early makes future algorithms much easier to use correctly.

---

# 2.315 Mistake 5 — Believing a High \(R^2\) Guarantees a Good Model

Suppose a model reports

```text
R² = 0.99
```

Many beginners immediately conclude that the model is excellent.

Not necessarily.

A very high \(R^2\) may result from

- overfitting,
- data leakage,
- duplicated observations,
- evaluating on the training data.

Always examine multiple evaluation metrics,

inspect residual plots,

and verify that the evaluation procedure is scientifically valid.

---

# 2.316 Mistake 6 — Ignoring the Physical Meaning of Features

Machine learning is not magic.

If a feature has no physical relationship with the target,

including it rarely improves the model.

Imagine predicting hardness using

- grain size,
- carbon content,
- density,
- laboratory building number,
- student's registration number.

The final two variables have no physical connection to hardness.

Adding irrelevant features usually increases noise rather than predictive power.

Always ask

> **Does this feature make scientific sense?**

---

# 2.317 Mistake 7 — Assuming Correlation Means Causation

Suppose two variables have a very strong correlation.

This does **not** automatically mean that one causes the other.

For example,

ice cream sales and drowning accidents often increase during summer.

The two quantities are correlated.

However,

buying ice cream does not cause drowning.

The common cause is higher temperature.

The same principle applies to materials science.

Two descriptors may appear strongly correlated,

yet the underlying physical mechanism may involve a completely different variable.

Statistical relationships should always be interpreted together with scientific knowledge.

---

# 2.318 Mistake 8 — Using Too Few Data Samples

Linear Regression can be trained on small datasets,

but extremely small datasets often produce unstable models.

Suppose we attempt to predict hardness using

three features

but have only

ten experimental samples.

The model has very little information from which to learn.

Whenever possible,

increase the number of high-quality observations rather than adding increasingly complicated algorithms.

In machine learning,

data quality is usually more valuable than algorithmic complexity.

---

# 2.319 Mistake 9 — Ignoring Outliers Completely

Some beginners remove every statistical outlier automatically.

Others never examine outliers at all.

Both approaches are problematic.

Every unusual observation should be investigated individually.

Ask

- Is it physically possible?
- Was there an experimental error?
- Was the unit entered correctly?
- Could this represent a genuinely exceptional material?

Only after answering these questions should an outlier be removed.

---

# 2.320 Mistake 10 — Memorizing Code Without Understanding It

One of the most dangerous habits is copying machine learning code from the internet without understanding what each line does.

For example,

many beginners can write

```python
model.fit(
    X_train,
    y_train
)
```

but cannot explain

- what the model is learning,
- how the coefficients are computed,
- why train-test splitting is necessary,
- what evaluation metrics measure.

Throughout this book,

our objective is different.

Every line of code is accompanied by a detailed explanation so that you understand **both the programming and the underlying mathematics**.

This approach will allow you to modify existing code confidently and build your own machine learning workflows rather than simply copying examples.

---

# 2.321 Best Practices for Every Linear Regression Project

Before considering a Linear Regression project complete,

verify the following checklist.

- The dataset has been inspected.
- Missing values have been handled appropriately.
- Duplicate observations have been removed if necessary.
- Units are consistent.
- Outliers have been investigated.
- Features have been selected thoughtfully.
- The dataset has been divided into training and testing subsets.
- The model has been evaluated using unseen data.
- Multiple evaluation metrics have been reported.
- The results have been interpreted in the context of materials science rather than statistics alone.

Following these practices greatly increases the reliability and scientific value of your machine learning models.

---

# 2.322 Chapter Summary

In this chapter, we developed a complete understanding of **Linear Regression**, beginning with the basic mathematical concepts and progressing toward a professional machine learning workflow.

We learned how Linear Regression models the relationship between input features and a target variable, how the regression coefficients are estimated using the least-squares principle, and how these ideas are implemented in Python using Scikit-learn.

Beyond the core algorithm, we explored the essential stages of a real machine learning project, including data preprocessing, missing-value handling, duplicate removal, outlier detection, feature scaling, feature engineering, train-test splitting, model evaluation, and cross-validation. We also discussed how Linear Regression connects naturally with **Pymatgen** and **Quantum ESPRESSO**, demonstrating how descriptors generated from computational materials science can become machine learning features.

By the end of this chapter, you should be able to:

- explain the mathematics behind Linear Regression,
- implement the algorithm entirely in Python without copying code,
- interpret every important regression metric,
- prepare experimental or computational materials datasets for machine learning,
- build an end-to-end Linear Regression workflow,
- recognize and avoid the most common mistakes made by beginners.

This foundation is essential because many advanced machine learning algorithms build upon the same concepts introduced here. Understanding Linear Regression thoroughly will make the remaining chapters significantly easier to learn.

---

## Looking Ahead

Linear Regression is one of the simplest supervised learning algorithms, but it also has important limitations. It assumes a linear relationship between features and the target, making it unsuitable for many complex real-world problems.

The next chapter introduces **Decision Trees**, the first nonlinear machine learning algorithm in this book. Unlike Linear Regression, Decision Trees learn by recursively splitting the feature space into regions, allowing them to capture highly nonlinear relationships without requiring explicit mathematical equations. They also form the foundation of powerful ensemble methods such as **Random Forests** and **XGBoost**, which you will study in the following chapters.

By understanding Decision Trees, you will take your first step beyond linear models into the modern world of machine learning.

