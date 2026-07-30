# Chapter 8 — Model Evaluation and Validation

# 8.1 Why Model Evaluation Matters

Imagine that you have spent several days training an XGBoost model or a Graph Neural Network (GNN) to predict the band gap of crystalline materials. The training completes successfully, the loss decreases steadily, and the model appears to perform exceptionally well.

Excited by the results, you evaluate the model on your dataset and obtain an impressive number:

```
Accuracy = 99%
```

or

```
R² = 0.99
```

At first glance, this seems like a remarkable achievement. It is tempting to conclude that the model has successfully learned the relationship between crystal structure and electronic properties.

However, experienced machine learning researchers immediately ask a different question:

> **"99% accuracy on what data?"**

This simple question lies at the heart of model evaluation.

A machine learning model is not valuable because it performs well on data it has already seen. Its true value lies in its ability to make reliable predictions for **new, unseen materials**.

For example, suppose we train a model using 50,000 crystal structures from the Materials Project database. The real objective is not to reproduce those known values. Instead, we want the model to accurately predict the properties of a completely new material that has never been synthesized or simulated before.

That capability is called **generalization**.

Generalization is one of the primary goals of machine learning.

---

# 8.2 Memorization Versus Learning

To understand why evaluation is necessary, consider two students preparing for an examination.

### Student A

Student A memorizes every solved example in the textbook.

Whenever an identical problem appears, the student immediately writes the correct answer.

However, if the instructor changes the numbers slightly or asks a new type of question, Student A becomes confused.

---

### Student B

Student B studies the underlying concepts.

Instead of memorizing answers, the student learns the physical principles, derives equations independently, and understands why each solution works.

When confronted with a completely new problem, Student B successfully solves it.

---

Which student actually learned the subject?

Obviously,

```
Student B
```

Machine learning models behave in exactly the same way.

A model can either

```
Memorize

or

Learn
```

Good evaluation distinguishes between these two possibilities.

---

# 8.3 What Does "Generalization" Mean?

Generalization refers to the ability of a model to perform well on data that were **not used during training**.

Conceptually,

```
Training Data

↓

Learn Patterns

↓

Unseen Material

↓

Correct Prediction
```

Notice that the new material has never been shown to the model before.

If the prediction remains accurate,

the model has generalized successfully.

If performance collapses,

the model has memorized the training data.

---

# 8.4 Why Evaluation Is Especially Important in Materials Science

In many areas of artificial intelligence,

millions of labeled examples are available.

For example,

image recognition datasets may contain millions of photographs.

Natural language processing models are trained on billions of words.

Materials science is very different.

Generating one labeled data point often requires:

- expensive Density Functional Theory (DFT) calculations,
- high-performance computing resources,
- lengthy laboratory experiments,
- complex characterization techniques.

Consequently,

materials datasets are usually much smaller.

Examples include

| Dataset | Approximate Size |
|----------|-----------------:|
| Experimental alloy datasets | Hundreds to thousands |
| Mechanical property datasets | Thousands |
| High-quality DFT datasets | Tens of thousands |
| Materials Project | Hundreds of thousands (property dependent) |
| OQMD | Hundreds of thousands |
| JARVIS | Tens to hundreds of thousands |

Although modern databases appear large, they are still tiny compared with datasets commonly used in computer vision or language models.

This limited data availability makes proper evaluation even more important.

---

# 8.5 Why Accuracy Alone Can Be Dangerous

Suppose we build a classification model to determine whether a material is metallic or insulating.

Our dataset contains

```
1000 materials
```

Among them,

```
980

↓

Metal
```

```
20

↓

Insulator
```

Now imagine an extremely simple model.

It predicts

```
Metal

↓

Every time
```

It never predicts "Insulator."

How accurate is this model?

Correct predictions:

```
980
```

Total samples:

```
1000
```

Accuracy:

```
980 / 1000

=

98%
```

A naïve observer might conclude that the model is excellent because it achieves

```
98% accuracy.
```

However,

the model completely fails to identify insulators.

From a scientific perspective,

it is practically useless.

This example illustrates one of the most important lessons in machine learning:

> **A high accuracy does not necessarily imply a good model.**

---

# 8.6 The Cost of Incorrect Predictions

Different prediction errors have different consequences.

Suppose we use machine learning to identify promising battery materials.

If the model incorrectly rejects a revolutionary new material,

years of scientific progress could be delayed.

Conversely,

if it incorrectly recommends a poor material,

researchers may waste months performing unnecessary DFT calculations or expensive experiments.

Therefore,

evaluation metrics should reflect the actual scientific objective rather than maximizing a single number.

---

# 8.7 The Machine Learning Evaluation Pipeline

A complete machine learning workflow consists of several stages.

```
Raw Dataset

↓

Preprocessing

↓

Feature Engineering

↓

Training

↓

Validation

↓

Testing

↓

Performance Analysis

↓

Model Deployment
```

Notice that

**evaluation is not a single step.**

Instead,

it occurs repeatedly throughout model development.

---

# 8.8 The Three Fundamental Questions

Whenever evaluating a machine learning model, researchers attempt to answer three questions.

### Question 1

Can the model fit the training data?

---

### Question 2

Can the model generalize to unseen data?

---

### Question 3

Can the model be trusted for scientific decision-making?

A complete evaluation strategy should answer all three.

---

# 8.9 Categories of Evaluation Metrics

Different machine learning problems require different evaluation metrics.

Broadly,

evaluation falls into two categories.

### Regression

Regression predicts continuous numerical quantities.

Examples include

- band gap,
- formation energy,
- density,
- Young's modulus,
- thermal conductivity,
- Curie temperature.

Common regression metrics include

- Mean Absolute Error (MAE),
- Mean Squared Error (MSE),
- Root Mean Squared Error (RMSE),
- Coefficient of Determination (R²).

---

### Classification

Classification predicts discrete categories.

Examples include

- metal vs insulator,
- stable vs unstable,
- magnetic vs non-magnetic,
- brittle vs ductile.

Common classification metrics include

- Accuracy,
- Precision,
- Recall,
- F1 Score,
- ROC Curve,
- ROC-AUC.

These metrics measure different aspects of predictive performance and often complement one another.

---

# 8.10 Roadmap of This Chapter

In this chapter, we will progressively build a complete understanding of model evaluation.

We will begin with regression metrics because most materials informatics problems involve predicting continuous physical properties.

After mastering regression evaluation, we will study classification metrics and learn why accuracy alone is often misleading.

Next, we will explore advanced validation techniques such as cross-validation, K-fold validation, leave-one-out validation, and nested cross-validation. These methods become essential when working with limited scientific datasets.

Finally, we will discuss evaluation from a materials science perspective, including challenges unique to DFT databases such as Materials Project, OQMD, JARVIS, NOMAD, and AFLOW. We will examine issues like data leakage, composition bias, structural similarity, and distribution shift, and we will establish best practices for evaluating machine learning models intended for real materials discovery.

By the end of this chapter, you will not only know **how** to compute evaluation metrics but also **when**, **why**, and **which** metrics should be used for different scientific problems. This understanding is essential before trusting any machine learning model in research or industrial applications.

# 8.11 Regression Problems in Materials Science

In the previous section, we learned that evaluating a machine learning model requires much more than simply reporting a single number such as accuracy. Before choosing an evaluation metric, however, we must first understand **what kind of problem we are trying to solve**.

Machine learning problems are generally divided into two broad categories:

```
Regression

and

Classification
```

Since this book focuses primarily on **materials informatics**, we will begin with regression because it represents the majority of property prediction problems encountered in computational materials science.

---

# 8.12 What Is Regression?

Regression is a supervised machine learning task in which the goal is to predict a **continuous numerical value**.

Unlike classification, where outputs belong to distinct categories, regression outputs can take any value within a continuous range.

For example,

```
Band Gap

↓

2.37 eV
```

or

```
Formation Energy

↓

−3.84 eV/atom
```

These values are not categories.

Instead, they are real numbers.

A regression model attempts to learn the mathematical relationship between the input features and these numerical targets.

Conceptually,

```
Material Features

↓

Machine Learning Model

↓

Numerical Prediction
```

---

# 8.13 Regression Problems in Materials Informatics

Regression is one of the most common machine learning tasks in modern materials science.

Researchers frequently use regression models to predict physical and chemical properties that are expensive to compute using Density Functional Theory (DFT) or difficult to measure experimentally.

Examples include

| Property | Typical Unit |
|-----------|--------------|
| Band Gap | eV |
| Formation Energy | eV/atom |
| Bulk Modulus | GPa |
| Shear Modulus | GPa |
| Young's Modulus | GPa |
| Density | g/cm³ |
| Thermal Conductivity | W·m⁻¹·K⁻¹ |
| Seebeck Coefficient | μV/K |
| Melting Temperature | K |
| Debye Temperature | K |

Notice that every target is a **continuous quantity**.

Therefore,

these are all regression problems.

---

# 8.14 A Simple Example

Suppose we want to predict the band gap of crystalline materials.

Our dataset may look like

| Material | Band Gap (eV) |
|-----------|--------------:|
| Silicon | 1.12 |
| Diamond | 5.47 |
| Germanium | 0.67 |
| Gallium Arsenide | 1.42 |
| Zinc Oxide | 3.37 |

The machine learning model receives structural or compositional information and attempts to predict the corresponding band gap.

Example

```
Input

↓

Crystal Structure

↓

Prediction

↓

2.41 eV
```

Since the prediction is a numerical value,

this is a regression task.

---

# 8.15 Why Perfect Predictions Are Almost Impossible

Suppose the experimentally measured band gap of a material is

```
2.50 eV
```

Our model predicts

```
2.48 eV
```

Is the model wrong?

Technically,

yes.

The prediction differs from the experimental value.

However,

the error is extremely small.

In regression,

we rarely expect predictions to be **exactly** equal to the true value.

Instead,

we ask

> **How close is the prediction to the correct answer?**

This question naturally leads to regression evaluation metrics.

---

# 8.16 Measuring Prediction Error

The simplest way to evaluate a prediction is to calculate its error.

Conceptually,

```
Prediction

−

True Value

↓

Prediction Error
```

Suppose

```
Predicted Band Gap

↓

2.30 eV
```

Actual value

```
2.60 eV
```

Prediction error

```
2.30 − 2.60

=

−0.30 eV
```

The negative sign indicates that the model underestimated the true value.

Similarly,

if

```
Prediction

↓

2.90 eV
```

True value

```
2.60 eV
```

then

```
Error

=

+0.30 eV
```

Now the model has overestimated the property.

---

# 8.17 Why Raw Error Is Not Sufficient

Imagine two predictions.

Prediction A

```
Error

=

−0.50 eV
```

Prediction B

```
Error

=

+0.50 eV
```

If we simply average these errors,

we obtain

```
(−0.50 + 0.50)

/

2

=

0
```

The average error becomes zero.

Does this mean the model is perfect?

Of course not.

Both predictions are wrong by the same magnitude.

The positive and negative errors have simply canceled each other.

This problem motivates the development of more meaningful evaluation metrics.

---

# 8.18 Characteristics of a Good Regression Metric

A useful regression metric should satisfy several requirements.

It should

- measure how close predictions are to the true values,
- treat overestimation and underestimation fairly,
- summarize performance using a single number,
- allow meaningful comparison between different models,
- have a clear physical interpretation whenever possible.

Different metrics satisfy these requirements in different ways.

For this reason,

machine learning researchers usually report **multiple regression metrics** rather than relying on only one.

---

# 8.19 The Four Most Important Regression Metrics

Throughout the scientific literature,

four regression metrics appear far more frequently than any others.

```
Prediction

↓

Compute Error

↓

Choose Metric
```

The most common choices are

1. **Mean Absolute Error (MAE)**
2. **Mean Squared Error (MSE)**
3. **Root Mean Squared Error (RMSE)**
4. **Coefficient of Determination (R²)**

Each metric emphasizes a different aspect of model performance.

Understanding their strengths and weaknesses is essential for interpreting published research papers.

---

# 8.20 Roadmap for the Next Sections

In the following sections, we will study each regression metric individually.

For every metric, we will examine

- the mathematical definition,
- the intuition behind the formula,
- step-by-step manual calculations,
- implementation in NumPy,
- implementation in Scikit-learn,
- implementation in PyTorch,
- interpretation of the results,
- advantages,
- limitations,
- and practical examples drawn from materials science.

By the end of this part of the chapter, you will be able to read evaluation tables in research papers with confidence and choose the most appropriate metric for your own machine learning models.

The first metric we will study is **Mean Absolute Error (MAE)**, one of the simplest and most interpretable measures of regression performance.

# 8.21 Mean Absolute Error (MAE)

Among all regression evaluation metrics, **Mean Absolute Error (MAE)** is probably the easiest to understand. It is often the first metric introduced in machine learning courses because its mathematical definition closely matches our intuitive understanding of prediction error.

The central idea behind MAE is remarkably simple:

> **Measure how far every prediction is from the true value, ignore whether the prediction is too high or too low, and compute the average error.**

Because of its simplicity and physical interpretability, MAE is one of the most frequently reported evaluation metrics in materials informatics research.

---

# 8.22 The Intuition Behind MAE

Suppose we predict the band gaps of several materials.

The results are

| Material | Actual (eV) | Predicted (eV) |
|-----------|------------:|---------------:|
| Silicon | 1.10 | 1.25 |
| Diamond | 5.50 | 5.20 |
| GaAs | 1.42 | 1.60 |
| ZnO | 3.30 | 3.10 |

If we compare each prediction with its true value, we obtain the prediction errors.

| Material | Error (eV) |
|-----------|-----------:|
| Silicon | +0.15 |
| Diamond | −0.30 |
| GaAs | +0.18 |
| ZnO | −0.20 |

Immediately, we encounter a problem.

Some errors are positive.

Some errors are negative.

If we simply average them,

positive and negative values begin canceling one another.

Therefore,

we first convert every error into its **absolute value**.

---

# 8.23 What Is an Absolute Value?

The absolute value of a number represents its distance from zero.

The direction is ignored.

Examples

```
|5|

=

5
```

```
|-5|

=

5
```

Similarly,

```
|0.25|

=

0.25
```

```
|-0.25|

=

0.25
```

Notice that

```
+0.25

and

−0.25
```

are treated equally because both represent an error of

```
0.25
```

units.

---

# 8.24 Computing Absolute Errors

Returning to our previous example,

the absolute errors become

| Material | Absolute Error (eV) |
|-----------|--------------------:|
| Silicon | 0.15 |
| Diamond | 0.30 |
| GaAs | 0.18 |
| ZnO | 0.20 |

Every value is now positive.

These numbers tell us **how wrong** the predictions are,

without worrying about whether the model overestimated or underestimated the property.

---

# 8.25 Calculating the Mean

The word

```
Mean
```

simply refers to the arithmetic average.

Suppose our absolute errors are

```
0.15

0.30

0.18

0.20
```

Their average is

```
(0.15 + 0.30 + 0.18 + 0.20)

/

4

=

0.2075
```

Therefore,

the Mean Absolute Error becomes

```
0.2075 eV
```

---

# 8.26 The Mathematical Definition of MAE

The complete mathematical expression is

```
MAE = (1 / N) × Σ |yi − ŷi|
```

where

- **N** is the total number of samples,
- **yi** is the true value,
- **ŷi** is the predicted value,
- **| |** denotes the absolute value.

Rather than memorizing the equation,

remember its meaning:

```
Prediction

↓

Difference

↓

Absolute Value

↓

Average
```

That sequence completely describes MAE.

---

# 8.27 Step-by-Step Manual Example

Suppose a model predicts the formation energies of five materials.

| Material | True (eV/atom) | Predicted (eV/atom) |
|-----------|---------------:|--------------------:|
| A | −3.80 | −3.70 |
| B | −2.15 | −2.40 |
| C | −4.30 | −4.10 |
| D | −1.95 | −2.05 |
| E | −5.10 | −4.95 |

---

### Step 1

Compute the prediction errors.

| Material | Error |
|-----------|------:|
| A | +0.10 |
| B | −0.25 |
| C | +0.20 |
| D | −0.10 |
| E | +0.15 |

---

### Step 2

Take absolute values.

| Material | Absolute Error |
|-----------|---------------:|
| A | 0.10 |
| B | 0.25 |
| C | 0.20 |
| D | 0.10 |
| E | 0.15 |

---

### Step 3

Average them.

```
(0.10 + 0.25 + 0.20 + 0.10 + 0.15)

/

5

=

0.16
```

Therefore,

```
MAE

=

0.16 eV/atom
```

This means that,

on average,

our predictions differ from the true formation energies by

```
0.16 eV/atom.
```

---

# 8.28 Interpreting MAE

One of MAE's greatest strengths is that it uses the **same units as the target property**.

Suppose we predict

```
Band Gap
```

Then

```
MAE

↓

eV
```

Suppose we predict

```
Young's Modulus
```

Then

```
MAE

↓

GPa
```

Suppose we predict

```
Density
```

Then

```
MAE

↓

g/cm³
```

Because the units remain unchanged,

the results are immediately meaningful to scientists.

---

# 8.29 Why Materials Scientists Like MAE

Imagine reading a research paper that reports

```
MAE = 0.08 eV
```

You immediately know

> On average, the predicted band gaps differ from the true values by only **0.08 electron volts**.

No further interpretation is necessary.

This direct physical meaning is one reason why MAE is frequently reported in materials informatics publications.

---

# 8.30 Computing MAE Using NumPy

Suppose we already have two arrays.

```python
import numpy as np

actual = np.array(

    [1.10, 5.50, 1.42, 3.30]

)

predicted = np.array(

    [1.25, 5.20, 1.60, 3.10]

)
```

The MAE can be computed as

```python
mae = np.mean(

    np.abs(

        actual - predicted

    )

)

print(mae)
```

This code follows exactly the three steps we performed manually.

1. Compute the differences.
2. Take the absolute values.
3. Compute the average.

---

# 8.31 Computing MAE with Scikit-learn

Scikit-learn provides a dedicated function.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(

    actual,

    predicted

)

print(mae)
```

This is the implementation most commonly used in machine learning projects.

---

# 8.32 Computing MAE in PyTorch

When training neural networks,

we often work directly with tensors.

```python
import torch

actual = torch.tensor(

    [1.10, 5.50, 1.42, 3.30]

)

predicted = torch.tensor(

    [1.25, 5.20, 1.60, 3.10]

)

mae = torch.mean(

    torch.abs(

        actual - predicted

    )

)

print(mae)
```

Notice that the logic remains identical.

Only the data structure changes.

---

# 8.33 Advantages of MAE

MAE possesses several desirable properties.

### Easy to Understand

Scientists immediately understand what the reported error means.

---

### Same Units as the Target

Errors remain physically interpretable.

---

### Less Sensitive to Outliers

Compared with metrics that square the error,

MAE does not excessively punish a few unusually poor predictions.

This makes it relatively robust.

---

### Widely Used

Many materials informatics papers report MAE as the primary evaluation metric.

---

# 8.34 Limitations of MAE

Despite its strengths,

MAE is not perfect.

Every prediction error contributes linearly.

For example,

errors of

```
0.2

and

2.0
```

are weighted exactly according to their size.

Sometimes,

we want very large errors to receive much stronger penalties.

Imagine two models.

Model A makes many small mistakes.

Model B performs perfectly most of the time but occasionally makes enormous prediction errors.

MAE may not distinguish these situations strongly enough.

For this reason,

another metric,

**Mean Squared Error (MSE),**

is often used during model training.

---

# 8.35 Summary

Mean Absolute Error is one of the simplest and most interpretable regression metrics.

It measures the average magnitude of prediction errors while ignoring whether predictions are above or below the true values.

Its major advantages include

- simple interpretation,
- same physical units as the target,
- robustness to extreme values,
- widespread use in materials science.

However,

because MAE treats all errors linearly,

it does not strongly penalize very large mistakes.

To address this limitation,

the next section introduces **Mean Squared Error (MSE)**, where prediction errors are squared before averaging, giving larger mistakes much greater influence on the final evaluation score.

# 8.36 Mean Squared Error (MSE)

In the previous section, we learned that Mean Absolute Error (MAE) measures the average magnitude of prediction errors by taking the absolute value of every error before averaging.

MAE is intuitive and easy to interpret.

However, suppose we compare the following two machine learning models.

### Model A

Prediction errors

```
0.10

0.12

0.15

0.20

0.18
```

### Model B

Prediction errors

```
0.01

0.02

0.03

0.04

1.80
```

Which model is better?

Model B predicts almost every sample extremely well.

Unfortunately,

it makes one catastrophic mistake.

In many scientific applications,

that single large error may be unacceptable.

For example,

if a model predicts the formation energy of a material incorrectly by

```
2 eV/atom
```

it could completely change conclusions regarding stability.

Therefore,

we sometimes want **large prediction errors to receive much greater punishment** than small ones.

This idea leads to **Mean Squared Error (MSE).**

---

# 8.37 The Core Idea Behind MSE

Instead of taking the absolute value of each error,

MSE squares every prediction error before averaging.

Conceptually,

```
Prediction

↓

Compute Error

↓

Square Error

↓

Average
```

Squaring has two important effects.

First,

all negative values become positive.

Second,

large errors become dramatically larger.

For example,

```
Error

↓

0.20
```

Squared error

```
0.20²

=

0.04
```

Now consider

```
Error

↓

2.00
```

Squared error

```
2.00²

=

4.00
```

Notice what happened.

The error increased by a factor of

```
10
```

but

the squared error increased by a factor of

```
100
```

Large mistakes now dominate the metric.

---

# 8.38 Why Squaring Works

Consider three prediction errors.

```
0.10

0.20

1.50
```

Their squared values become

```
0.01

0.04

2.25
```

Observe the relative contributions.

The smallest error contributes almost nothing.

The largest error dominates the total.

This behavior is intentional.

MSE is designed to discourage models from making extremely poor predictions.

---

# 8.39 A Physical Interpretation

Suppose we predict the band gap of several semiconductors.

Most predictions differ from the true values by only

```
0.05 eV
```

One prediction,

however,

is wrong by

```
1.80 eV
```

Should this single failure matter?

In materials science,

the answer is often

```
Yes.
```

A prediction error of

```
1.80 eV
```

may completely misclassify a semiconductor as an insulator or even as a metallic material.

MSE strongly penalizes such failures.

---

# 8.40 The Mathematical Definition of MSE

The mathematical formula is

```
MSE = (1 / N) × Σ (yi − ŷi)²
```

where

- **N** is the number of samples,
- **yi** is the true value,
- **ŷi** is the predicted value.

Notice the only difference from MAE.

Instead of

```
Absolute Value
```

we now compute

```
Square
```

Everything else remains the same.

---

# 8.41 Step-by-Step Manual Calculation

Suppose we predict the band gaps of four materials.

| Material | True (eV) | Predicted (eV) |
|-----------|----------:|---------------:|
| A | 1.20 | 1.10 |
| B | 2.80 | 3.00 |
| C | 4.50 | 4.20 |
| D | 0.90 | 1.20 |

---

### Step 1

Compute prediction errors.

| Material | Error |
|-----------|------:|
| A | −0.10 |
| B | +0.20 |
| C | −0.30 |
| D | +0.30 |

---

### Step 2

Square every error.

| Material | Squared Error |
|-----------|--------------:|
| A | 0.01 |
| B | 0.04 |
| C | 0.09 |
| D | 0.09 |

---

### Step 3

Average the squared errors.

```
(0.01 + 0.04 + 0.09 + 0.09)

/

4

=

0.0575
```

Therefore,

```
MSE

=

0.0575
```

---

# 8.42 Comparing MAE and MSE

Consider two prediction errors.

```
0.20

and

2.00
```

Now compare how each metric treats them.

| Error | Absolute Error | Squared Error |
|-------:|---------------:|--------------:|
| 0.20 | 0.20 | 0.04 |
| 2.00 | 2.00 | 4.00 |

Notice that

MAE increases

```
10×

```

while

MSE increases

```
100×

```

Therefore,

MSE is much more sensitive to large prediction errors.

---

# 8.43 Units of MSE

One important characteristic of MSE is that its units are **squared**.

Suppose we predict

```
Band Gap

↓

eV
```

Then

```
MSE

↓

eV²
```

Suppose we predict

```
Young's Modulus

↓

GPa
```

Then

```
MSE

↓

GPa²
```

This creates a problem.

Scientists usually think in terms of

```
eV

or

GPa
```

—not

```
eV²

or

GPa².
```

Consequently,

MSE is mathematically convenient,

but its numerical value is often less intuitive than MAE.

Later,

we will solve this issue using **Root Mean Squared Error (RMSE).**

---

# 8.44 Computing MSE Using NumPy

The implementation closely resembles the MAE calculation.

```python
import numpy as np

actual = np.array(

    [1.20, 2.80, 4.50, 0.90]

)

predicted = np.array(

    [1.10, 3.00, 4.20, 1.20]

)

mse = np.mean(

    (

        actual - predicted

    ) ** 2

)

print(mse)
```

The only difference is

```
**2
```

which squares every prediction error.

---

# 8.45 Computing MSE with Scikit-learn

Scikit-learn provides a dedicated implementation.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(

    actual,

    predicted

)

print(mse)
```

This function is widely used in both research and industry.

---

# 8.46 Computing MSE in PyTorch

Deep learning models commonly compute MSE directly during training.

```python
import torch

actual = torch.tensor(

    [1.20, 2.80, 4.50, 0.90]

)

predicted = torch.tensor(

    [1.10, 3.00, 4.20, 1.20]

)

mse = torch.mean(

    (

        actual - predicted

    ) ** 2

)

print(mse)
```

PyTorch also includes a built-in loss function.

```python
loss_function = torch.nn.MSELoss()

loss = loss_function(

    predicted,

    actual

)
```

During neural network training,

`MSELoss` is one of the most frequently used regression loss functions.

---

# 8.47 Advantages of MSE

MSE has several important strengths.

### Strongly Penalizes Large Errors

Large prediction mistakes receive much greater weight than small ones.

---

### Smooth Mathematical Properties

The squared function is continuous and differentiable.

This makes optimization using gradient descent straightforward.

---

### Standard Loss Function

Many regression algorithms,

including neural networks,

use MSE as their default training objective.

---

### Encourages Stable Predictions

Models trained using MSE generally avoid producing occasional catastrophic prediction errors.

---

# 8.48 Limitations of MSE

Despite its popularity,

MSE has several limitations.

### Sensitive to Outliers

One unusually poor prediction can dominate the metric.

---

### Difficult to Interpret

The units are squared.

Scientists rarely interpret

```
eV²

or

GPa²
```

directly.

---

### May Overemphasize Rare Mistakes

If the dataset contains measurement errors or noisy labels,

MSE may focus excessively on those unusual samples.

---

# 8.49 When Should You Use MSE?

MSE is especially useful when

- large prediction errors are unacceptable,
- neural networks are trained using gradient descent,
- optimization stability is important.

However,

when communicating results to scientists,

MAE is often easier to interpret because it remains in the original physical units.

Consequently,

many research papers report **both MAE and MSE**.

---

# 8.50 Summary

Mean Squared Error evaluates regression models by squaring every prediction error before averaging.

Compared with MAE,

it places much greater emphasis on large prediction mistakes,

making it an excellent objective function for training machine learning models.

However,

its squared units make interpretation less intuitive.

To overcome this limitation,

the next section introduces **Root Mean Squared Error (RMSE)**.

RMSE simply takes the square root of MSE,

returning the evaluation metric to the original physical units while retaining much of MSE's sensitivity to large errors.

# 8.51 Root Mean Squared Error (RMSE)

In the previous section, we introduced **Mean Squared Error (MSE)** and learned that it strongly penalizes large prediction errors by squaring them before averaging.

This property makes MSE an excellent loss function for training machine learning models.

However, MSE has one important drawback.

Its units are squared.

Suppose we are predicting

```
Band Gap

↓

eV
```

The corresponding MSE has units of

```
eV²
```

Scientists rarely think in terms of

```
electron volt squared.
```

Likewise,

if we predict Young's modulus,

the MSE is expressed in

```
GPa²
```

rather than

```
GPa.
```

This makes MSE difficult to interpret physically.

To solve this problem, we introduce **Root Mean Squared Error (RMSE).**

---

# 8.52 The Core Idea Behind RMSE

RMSE is extremely simple.

After computing the Mean Squared Error,

we take its square root.

Conceptually,

```
Prediction

↓

Compute Error

↓

Square Error

↓

Average

↓

Square Root

↓

RMSE
```

Taking the square root restores the metric to the original units of the target variable.

---

# 8.53 Why Take the Square Root?

Suppose the Mean Squared Error of a model is

```
0.09 eV²
```

Although mathematically correct,

it is difficult to understand what

```
0.09 eV²
```

means physically.

Now take the square root.

```
√0.09

=

0.30
```

The RMSE becomes

```
0.30 eV
```

This value immediately tells us that

> On average, prediction errors are approximately **0.30 electron volts**.

Scientists can interpret this number much more naturally.

---

# 8.54 The Mathematical Definition of RMSE

RMSE is defined as

```
RMSE = √MSE
```

Expanding the expression gives

```
RMSE = √[(1 / N) × Σ (yi − ŷi)²]
```

where

- **N** is the number of samples,
- **yi** is the true value,
- **ŷi** is the predicted value.

Notice that RMSE is simply MSE with one additional mathematical operation:

```
Square Root
```

---

# 8.55 Step-by-Step Manual Example

Suppose we predict the formation energies of four materials.

| Material | True (eV/atom) | Predicted (eV/atom) |
|-----------|---------------:|--------------------:|
| A | −3.50 | −3.30 |
| B | −2.20 | −2.10 |
| C | −4.80 | −5.10 |
| D | −1.70 | −1.60 |

---

### Step 1

Compute prediction errors.

| Material | Error |
|-----------|------:|
| A | +0.20 |
| B | +0.10 |
| C | −0.30 |
| D | +0.10 |

---

### Step 2

Square every error.

| Material | Squared Error |
|-----------|--------------:|
| A | 0.04 |
| B | 0.01 |
| C | 0.09 |
| D | 0.01 |

---

### Step 3

Compute the Mean Squared Error.

```
(0.04 + 0.01 + 0.09 + 0.01)

/

4

=

0.0375
```

Therefore,

```
MSE

=

0.0375 eV²/atom²
```

---

### Step 4

Take the square root.

```
RMSE

=

√0.0375

≈

0.194
```

The final answer is

```
RMSE

≈

0.194 eV/atom
```

Notice that the units have returned to

```
eV/atom
```

making the result much easier to interpret.

---

# 8.56 Comparing MAE, MSE, and RMSE

Suppose a regression model produces the following evaluation results.

```
MAE

=

0.18 eV
```

```
MSE

=

0.07 eV²
```

```
RMSE

=

0.26 eV
```

Observe the differences.

### MAE

Represents the average prediction error.

Easy to interpret.

---

### MSE

Strongly penalizes large mistakes.

Useful for optimization.

Units are squared.

---

### RMSE

Retains MSE's emphasis on larger errors,

while restoring the original units.

This combination makes RMSE extremely popular in scientific research.

---

# 8.57 Why RMSE Is Usually Larger Than MAE

Consider two models.

### Model A

Errors

```
0.20

0.22

0.18

0.21
```

All predictions are consistently good.

---

### Model B

Errors

```
0.01

0.02

0.03

0.90
```

Three predictions are excellent,

but one prediction is extremely poor.

MAE increases slightly because of the large error.

RMSE increases much more dramatically because the

```
0.90
```

error is squared.

Therefore,

RMSE generally becomes **larger than MAE** whenever large prediction errors are present.

This makes RMSE particularly useful when catastrophic mistakes must be avoided.

---

# 8.58 Computing RMSE Using NumPy

NumPy allows us to compute RMSE directly.

```python
import numpy as np

actual = np.array(

    [1.20, 2.80, 4.50, 0.90]

)

predicted = np.array(

    [1.10, 3.00, 4.20, 1.20]

)

rmse = np.sqrt(

    np.mean(

        (

            actual - predicted

        ) ** 2

    )

)

print(rmse)
```

Notice that we simply wrap the MSE calculation inside

```python
np.sqrt()
```

---

# 8.59 Computing RMSE with Scikit-learn

One approach is to compute the MSE first and then take its square root.

```python
from sklearn.metrics import mean_squared_error

import numpy as np

mse = mean_squared_error(

    actual,

    predicted

)

rmse = np.sqrt(

    mse

)

print(rmse)
```

In recent versions of Scikit-learn, some APIs also support directly requesting RMSE, but understanding the relationship

```
RMSE = √MSE
```

is more important than memorizing a specific function.

---

# 8.60 Computing RMSE in PyTorch

The implementation is straightforward.

```python
import torch

actual = torch.tensor(

    [1.20, 2.80, 4.50, 0.90]

)

predicted = torch.tensor(

    [1.10, 3.00, 4.20, 1.20]

)

rmse = torch.sqrt(

    torch.mean(

        (

            actual - predicted

        ) ** 2

    )

)

print(rmse)
```

Although neural networks are usually trained using MSE,

RMSE is often reported when presenting final results.

---

# 8.61 RMSE in Materials Science Research

Suppose two machine learning models are developed to predict band gaps.

| Model | RMSE (eV) |
|-------|----------:|
| Model A | 0.18 |
| Model B | 0.31 |

Model A has the smaller RMSE.

This indicates that,

on average,

its predictions remain closer to the true band gaps.

Because RMSE uses the same physical units as the target,

researchers can immediately judge whether the prediction accuracy is acceptable for a particular application.

---

# 8.62 Advantages of RMSE

RMSE offers several important advantages.

### Same Units as the Target

Results remain physically meaningful.

---

### Sensitive to Large Errors

Large prediction mistakes receive greater emphasis.

---

### Easy to Compare Between Models

Lower RMSE indicates better predictive performance.

---

### Widely Reported

Many materials informatics studies report both

```
MAE

and

RMSE.
```

---

# 8.63 Limitations of RMSE

RMSE is not perfect.

### Sensitive to Outliers

A few unusually poor predictions can significantly increase RMSE.

---

### Can Be Influenced by Noisy Labels

If experimental measurements contain large uncertainties,

RMSE may become unnecessarily large.

---

### Less Robust Than MAE

When datasets contain many outliers,

MAE often provides a more stable estimate of prediction quality.

---

# 8.64 When Should You Report RMSE?

RMSE is an excellent choice when

- large prediction errors are particularly undesirable,
- comparison with previous literature is important,
- results should remain in physically meaningful units.

For this reason,

many materials informatics papers report

- MAE,
- RMSE,

and sometimes MSE,

allowing readers to understand both the average prediction error and the model's sensitivity to larger mistakes.

---

# 8.65 Summary

Root Mean Squared Error combines the strengths of Mean Squared Error with the interpretability of Mean Absolute Error.

By taking the square root of MSE,

RMSE returns the evaluation metric to the original physical units while preserving its strong emphasis on large prediction errors.

As a result,

RMSE has become one of the most widely reported regression metrics in materials science and machine learning research.

Although MAE, MSE, and RMSE each provide valuable information,

none of them tells us **how much of the variation in the data is actually explained by the model**.

To answer that question,

the next section introduces the **Coefficient of Determination (R²)**, one of the most important metrics for evaluating regression models.

# 8.66 Coefficient of Determination (R² Score)

In the previous sections, we studied three important regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

All three metrics answer essentially the same question:

> **"How large are the prediction errors?"**

However, they do **not** answer another equally important question:

> **"How well does the model explain the variation in the data?"**

Imagine two machine learning models.

Model A has an RMSE of

```
0.35 eV
```

Model B has an RMSE of

```
0.28 eV
```

Clearly,

Model B performs better.

But **how much better?**

Does Model B explain

- half of the variation?
- 80% of the variation?
- almost all of the variation?

Error-based metrics cannot answer these questions.

To measure how well a model captures the underlying relationship in the data, we use the **Coefficient of Determination**, commonly called the **R² Score**.

---

# 8.67 The Intuition Behind R²

Suppose you want to predict the band gaps of crystalline materials.

Imagine two different prediction strategies.

### Strategy 1

Always predict the **average band gap** of the training dataset.

For every material,

the prediction remains identical.

```
Material A

↓

2.15 eV
```

```
Material B

↓

2.15 eV
```

```
Material C

↓

2.15 eV
```

Clearly,

this approach ignores all structural and chemical information.

---

### Strategy 2

Use a machine learning model that considers

- composition,
- crystal structure,
- atomic properties,
- bonding information.

Predictions now vary from one material to another.

Ideally,

they become much closer to the true values.

The natural question becomes

> **How much improvement did the machine learning model achieve compared with simply predicting the average?**

R² measures exactly this improvement.

---

# 8.68 Understanding Variation

Before defining R²,

we must understand the idea of **variation**.

Consider the following band gaps.

```
1.10

1.35

2.20

3.10

5.40
```

These values are clearly different.

They vary considerably.

Now consider

```
2.40

2.42

2.39

2.41

2.40
```

These values exhibit very little variation.

Regression models attempt to explain this variation using the available input features.

A better model explains more of it.

---

# 8.69 A Simple Analogy

Imagine several students taking an examination.

The class average is

```
70
```

Suppose you know nothing about an individual student.

The safest prediction is simply

```
70
```

for everyone.

Now imagine that you know

- study hours,
- attendance,
- assignment scores,
- previous grades.

Using this information,

you can make much better predictions.

The more accurately your predictions reflect the actual scores,

the more variation your model explains.

R² measures how successful this explanation is.

---

# 8.70 The Range of R²

One reason R² is so popular is that it has an intuitive interpretation.

Its values typically fall within the following range.

| R² Value | Interpretation |
|----------:|----------------|
| 1.00 | Perfect prediction |
| 0.90 | Excellent model |
| 0.75 | Good model |
| 0.50 | Moderate model |
| 0.00 | No improvement over predicting the mean |
| Less than 0 | Worse than predicting the mean |

Unlike MAE or RMSE,

R² has **no physical units**.

Instead,

it measures the fraction of variation explained by the model.

---

# 8.71 What Does R² = 1 Mean?

Suppose

```
Actual Band Gap

↓

2.40 eV
```

Prediction

```
↓

2.40 eV
```

Every prediction is exactly correct.

Prediction errors are zero.

The regression line passes through every data point.

The model explains **100% of the variation**.

Therefore,

```
R²

=

1
```

This is the best possible value.

---

# 8.72 What Does R² = 0 Mean?

Now imagine another model.

Instead of learning from the data,

it simply predicts the average target value.

For every material,

```
Prediction

↓

Average
```

The model has learned nothing.

It performs no better than a naïve baseline.

Therefore,

```
R²

=

0
```

Notice that

```
R² = 0
```

does **not** necessarily mean the predictions are completely wrong.

It simply means the model is no better than predicting the average.

---

# 8.73 Can R² Be Negative?

Surprisingly,

yes.

Many beginners assume that R² must lie between

```
0

and

1.
```

This is incorrect.

Suppose a model makes extremely poor predictions.

Its errors become even larger than those obtained by always predicting the average.

In that situation,

```
R²

<

0
```

A negative R² indicates

> **The machine learning model performs worse than an extremely simple baseline.**

Such a result usually suggests

- severe underfitting,
- incorrect preprocessing,
- unsuitable features,
- or implementation errors.

---

# 8.74 The Mathematical Definition

The R² score is defined as

```
R² = 1 − (SSres / SStot)
```

where

```
SSres
```

is the **Residual Sum of Squares**, representing the squared prediction errors.

```
SStot
```

is the **Total Sum of Squares**, representing the total variation present in the true data.

Conceptually,

```
Variation Explained

↓

Large

↓

High R²
```

```
Prediction Error

↓

Large

↓

Low R²
```

Although the equation may initially appear intimidating,

its interpretation is remarkably straightforward.

It simply compares

```
Your Model

vs

Predicting the Mean
```

---

# 8.75 Visualizing Different R² Values

Imagine plotting predicted values against true values.

### Excellent Model

```
•

  •

    •

      •

        •
```

The points lie almost perfectly along a straight line.

R² approaches

```
1
```

---

### Average Model

```
•

•

    •

         •

      •
```

The points scatter noticeably.

R² decreases.

---

### Poor Model

```
•

         •

•

      •

            •
```

The points show little relationship.

R² approaches

```
0
```

or even becomes negative.

---

# 8.76 Computing R² Using Scikit-learn

Scikit-learn provides a built-in implementation.

```python
from sklearn.metrics import r2_score

actual = [

    1.20,

    2.80,

    4.50,

    0.90

]

predicted = [

    1.10,

    3.00,

    4.20,

    1.20

]

r2 = r2_score(

    actual,

    predicted

)

print(r2)
```

The output might be

```text
0.94
```

This means the model explains approximately

```
94%
```

of the variation in the target values.

---

# 8.77 Computing R² with NumPy

Although Scikit-learn is recommended,

it is useful to understand the calculation.

```python
import numpy as np

actual = np.array(

    [1.20, 2.80, 4.50, 0.90]

)

predicted = np.array(

    [1.10, 3.00, 4.20, 1.20]

)

ss_res = np.sum(

    (

        actual - predicted

    ) ** 2

)

ss_tot = np.sum(

    (

        actual - np.mean(

            actual

        )

    ) ** 2

)

r2 = 1 - (

    ss_res / ss_tot

)

print(r2)
```

Notice that the implementation follows the mathematical definition exactly.

---

# 8.78 Advantages of R²

R² possesses several important strengths.

### Dimensionless

It has no physical units,

making comparisons across different regression problems easier.

---

### Easy Interpretation

Higher values indicate that the model explains more variation.

---

### Useful for Comparing Models

Suppose two models produce

| Model | R² |
|-------|---:|
| Linear Regression | 0.81 |
| XGBoost | 0.94 |

The XGBoost model explains substantially more variation in the data.

---

### Widely Reported

R² appears in nearly every regression paper in materials informatics.

---

# 8.79 Limitations of R²

Despite its popularity,

R² should never be used alone.

### It Does Not Measure Error Magnitude

Two models may have similar R² values but very different MAE or RMSE values.

---

### Sensitive to Dataset Characteristics

Different datasets naturally produce different achievable R² scores.

Comparisons should therefore be made using the same dataset.

---

### High R² Does Not Guarantee Good Predictions

A model can achieve a high R² while still making prediction errors that are scientifically unacceptable.

For example,

an RMSE of

```
0.60 eV
```

may be too large for accurate band gap prediction,

even if R² appears high.

---

# 8.80 Which Regression Metrics Should You Report?

In professional research,

one metric is rarely sufficient.

Instead,

multiple metrics are reported together.

A common evaluation table looks like

| Metric | Purpose |
|---------|----------|
| MAE | Average prediction error |
| RMSE | Penalizes large errors while remaining interpretable |
| R² | Measures explained variation |

MSE is often used during model optimization,

while MAE,

RMSE,

and R² are commonly reported in published results.

Using multiple metrics provides a more complete understanding of model performance.

---

# 8.81 Summary

The Coefficient of Determination (R²) measures how well a regression model explains the variation present in the target data.

Unlike MAE, MSE, and RMSE,

which focus on the magnitude of prediction errors,

R² evaluates the **overall explanatory power** of the model.

A good regression study in materials science should therefore report multiple complementary metrics,

typically

- MAE,
- RMSE,
- and R².

Together,

these metrics provide a balanced picture of prediction accuracy, robustness, and explanatory capability.

---

# 8.82 Transition to Classification Evaluation

So far, every evaluation metric we have studied has been designed for **regression**, where the target is a continuous numerical value such as band gap, formation energy, or elastic modulus.

However,

many materials informatics problems are **classification** tasks rather than regression tasks.

For example,

we may wish to determine whether a material is

- metallic or insulating,
- stable or unstable,
- magnetic or non-magnetic,
- superconducting or non-superconducting.

These problems require an entirely different set of evaluation metrics.

In the next part of this chapter, we will begin by introducing the **confusion matrix**, the foundation upon which nearly all classification metrics—including accuracy, precision, recall, F1 score, and ROC-AUC—are built.

# 8.83 Classification Problems in Materials Science

In the previous sections, we focused on regression problems, where the objective was to predict continuous numerical values.

Examples included:

- band gap prediction,
- formation energy prediction,
- elastic modulus prediction,
- thermal conductivity prediction.

The model output in these cases was a number.

For example:

```
Band Gap

↓

2.45 eV
```

However, many important materials science problems do not require predicting a continuous value.

Instead, we may only need to determine which category a material belongs to.

These problems are called **classification problems**.

---

# 8.84 What Is Classification?

Classification is a supervised machine learning task where the goal is to assign an input into one or more predefined categories.

The output is not a continuous number.

Instead, it is a class label.

Conceptually:

```
Material Information

↓

Machine Learning Model

↓

Category
```

Examples:

```
Crystal Structure

↓

Stable
```

or

```
Crystal Structure

↓

Unstable
```

---

# 8.85 Examples of Classification in Materials Informatics

Classification appears frequently in materials discovery workflows.

Some examples include:

| Problem | Classes |
|---------|---------|
| Electronic classification | Metal / Semiconductor / Insulator |
| Stability prediction | Stable / Unstable |
| Magnetic prediction | Magnetic / Non-magnetic |
| Battery material screening | Suitable / Unsuitable |
| Catalyst screening | Active / Inactive |
| Phase identification | Phase A / Phase B |
| Superconductor prediction | Superconductor / Non-superconductor |

Unlike regression,

the output is a category rather than a numerical property.

---

# 8.86 Binary Classification

The simplest classification problem contains only two classes.

This is called **binary classification**.

Examples:

```
Metal

or

Insulator
```

```
Stable

or

Unstable
```

```
Magnetic

or

Non-magnetic
```

The model attempts to learn a boundary separating the two groups.

Conceptually:

```
Input Features

↓

Decision Boundary

↓

Class 0 or Class 1
```

---

# 8.87 Multiclass Classification

Some problems contain more than two categories.

For example:

A material may belong to:

```
Class 0

↓

Metal
```

```
Class 1

↓

Semiconductor
```

```
Class 2

↓

Insulator
```

This is called multiclass classification.

The model must learn how to distinguish between several possible outcomes.

---

# 8.88 The Need for Classification Metrics

At first glance,

classification evaluation appears simple.

A prediction is either

```
Correct

or

Wrong
```

Therefore,

we might think that measuring the percentage of correct predictions is enough.

This leads to the metric:

```
Accuracy
```

However,

as we discussed earlier,

accuracy can be extremely misleading.

To understand why,

we need to introduce the foundation of classification evaluation:

the **confusion matrix**.

---

# 8.89 The Confusion Matrix

A confusion matrix is a table that compares:

```
Actual Class

vs

Predicted Class
```

It shows exactly how a classifier succeeds and fails.

For binary classification,

the confusion matrix contains four possible outcomes.

---

# 8.90 The Four Outcomes

Suppose we build a model to predict whether a material is stable.

The two classes are:

```
Stable

and

Unstable
```

The model makes predictions.

Every prediction falls into one of four categories.

---

## True Positive (TP)

The model predicts positive,

and the material is actually positive.

Example:

```
Actual:

Stable

Prediction:

Stable
```

The model is correct.

---

## True Negative (TN)

The model predicts negative,

and the material is actually negative.

Example:

```
Actual:

Unstable

Prediction:

Unstable
```

Again,

the model is correct.

---

## False Positive (FP)

The model predicts positive,

but the material is actually negative.

Example:

```
Actual:

Unstable

Prediction:

Stable
```

The model incorrectly identifies an unstable material as stable.

This is also called a

```
False Alarm
```

---

## False Negative (FN)

The model predicts negative,

but the material is actually positive.

Example:

```
Actual:

Stable

Prediction:

Unstable
```

The model fails to identify a useful material.

This type of error can be especially costly in materials discovery.

---

# 8.91 Confusion Matrix Representation

The four outcomes can be arranged as:

| | Predicted Positive | Predicted Negative |
|-|-|-|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

This simple table forms the foundation of almost every classification metric.

---

# 8.92 Why Confusion Matrix Matters in Materials Science

Consider a machine learning model designed to discover new battery materials.

The model predicts:

```
Promising Battery Material

or

Not Promising
```

Now consider two possible mistakes.

---

## False Positive

The model predicts:

```
Promising
```

but the material performs poorly.

The consequence:

Researchers waste time testing a bad candidate.

---

## False Negative

The model predicts:

```
Not Promising
```

but the material is actually excellent.

The consequence:

A potentially revolutionary material may never be investigated.

---

These two errors have completely different scientific consequences.

A single accuracy value cannot show this difference.

---

# 8.93 Class Imbalance Problem

One of the biggest challenges in materials classification is **class imbalance**.

Suppose we want to identify superconducting materials.

Our dataset contains:

```
10,000 materials
```

Among them:

```
9,900

↓

Non-superconductors
```

and

```
100

↓

Superconductors
```

The dataset is highly imbalanced.

---

# 8.94 Why Accuracy Fails Under Imbalance

Imagine a useless model that always predicts:

```
Non-superconductor
```

It never identifies superconductors.

Its accuracy is:

```
9900 / 10000

=

99%
```

The model appears excellent.

But scientifically,

it has completely failed.

It cannot discover a single superconducting material.

This is why researchers require more informative metrics.

---

# 8.95 Classification Metrics Overview

After understanding the confusion matrix,

we can define meaningful classification metrics.

The most important ones are:

## Accuracy

Measures overall correctness.

---

## Precision

Measures how reliable positive predictions are.

---

## Recall

Measures how many actual positives were found.

---

## F1 Score

Balances precision and recall.

---

## ROC-AUC

Measures classification ability across different decision thresholds.

---

# 8.96 Choosing the Right Metric

Different scientific problems require different priorities.

For example:

### Drug discovery

Missing a useful compound may be unacceptable.

Therefore:

```
Recall

is important.
```

---

### Expensive experimental validation

Testing false candidates is costly.

Therefore:

```
Precision

is important.
```

---

### Balanced problems

Both types of mistakes matter.

Therefore:

```
F1 Score

is useful.
```

---

### Overall comparison

ROC-AUC provides a broader measurement of classifier quality.

---

# 8.97 Classification Evaluation Workflow

A complete classification evaluation process follows:

```
Predictions

↓

Confusion Matrix

↓

TP, TN, FP, FN

↓

Calculate Metrics

↓

Interpret Scientific Meaning
```

The final step is extremely important.

A metric is not valuable unless it connects to the actual scientific objective.

---

# 8.98 Summary

Classification problems are fundamentally different from regression problems.

Instead of predicting continuous numerical values,

classification models assign materials into categories.

Because different classification mistakes have different consequences,

accuracy alone is often insufficient.

The confusion matrix provides the foundation for understanding classification performance by separating predictions into:

- True Positives,
- True Negatives,
- False Positives,
- False Negatives.

Using these four quantities,

we can calculate more meaningful metrics such as precision, recall, F1 score, and ROC-AUC.

In the next section,

we will examine the first and most commonly used classification metric:

**Accuracy**

and understand both its usefulness and its limitations.

# 8.99 Accuracy: The Simplest Classification Metric

In the previous section, we introduced the confusion matrix and learned that every binary classification prediction falls into one of four categories:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

These four quantities form the foundation of classification evaluation.

The first metric derived from the confusion matrix is **accuracy**.

Accuracy is the most commonly reported classification metric because it answers a simple question:

> **"What fraction of all predictions did the model get correct?"**

Although accuracy is easy to understand, it is also one of the most frequently misunderstood metrics in machine learning.

---

# 8.100 Understanding Accuracy

Suppose we build a machine learning model that predicts whether a material is thermodynamically stable.

The model examines:

- chemical composition,
- crystal structure,
- atomic features,

and predicts:

```
Stable

or

Unstable
```

After testing the model on 1000 materials,

we compare its predictions with the true labels.

The results are:

```
850 predictions were correct

150 predictions were incorrect
```

The natural question is:

> What percentage of predictions were correct?

This percentage is the accuracy.

---

# 8.101 Mathematical Definition of Accuracy

Accuracy is defined as:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

where:

- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

The numerator represents all correct predictions.

The denominator represents all predictions.

Conceptually:

```
Correct Predictions

↓

Divide by

↓

Total Predictions
```

---

# 8.102 Step-by-Step Accuracy Calculation

Consider a binary classifier for identifying magnetic materials.

After testing 200 materials, the confusion matrix is:

| | Predicted Magnetic | Predicted Non-Magnetic |
|-|-|-|
| Actual Magnetic | TP = 60 | FN = 10 |
| Actual Non-Magnetic | FP = 20 | TN = 110 |

Now calculate accuracy.

---

## Step 1: Count Correct Predictions

Correct predictions are:

```
True Positive

+

True Negative
```

Therefore:

```
60 + 110

=

170
```

---

## Step 2: Count Total Predictions

Total samples:

```
60 + 10 + 20 + 110

=

200
```

---

## Step 3: Calculate Accuracy

```
Accuracy

=

170 / 200

=

0.85
```

Therefore:

```
Accuracy = 85%
```

This means the model correctly classified 85% of all materials.

---

# 8.103 Interpreting Accuracy

Accuracy is intuitive.

For example:

```
Accuracy = 95%
```

means:

> The model made correct predictions for 95 out of every 100 samples.

This simplicity explains why accuracy is often the first metric reported when evaluating classification models.

However,

accuracy hides important information.

It does not tell us:

- what type of mistakes were made,
- whether rare classes were detected,
- whether false positives or false negatives dominate.

---

# 8.104 Why Accuracy Can Be Misleading

Consider a dataset for discovering superconducting materials.

Suppose we have:

```
10,000 materials
```

Among them:

```
9,900

↓

Non-superconductors
```

and

```
100

↓

Superconductors
```

The dataset is highly imbalanced.

---

Now imagine a useless model:

It predicts every material as:

```
Non-superconductor
```

The confusion matrix becomes:

| | Predicted Superconductor | Predicted Non-superconductor |
|-|-|-|
| Actual Superconductor | TP = 0 | FN = 100 |
| Actual Non-superconductor | FP = 0 | TN = 9900 |

Accuracy:

```
(0 + 9900) / 10000

=

99%
```

The model appears excellent.

But what did it actually learn?

Nothing.

It failed to identify even a single superconducting material.

---

# 8.105 Accuracy and Materials Discovery

This problem appears frequently in materials informatics.

Many interesting materials belong to rare classes.

Examples:

- superconductors,
- highly efficient catalysts,
- topological materials,
- stable high-energy-density battery compounds.

The scientifically important materials are often the minority class.

Therefore,

a model that achieves high accuracy by ignoring rare materials may be useless.

---

# 8.106 Accuracy Versus Scientific Importance

Consider two errors.

### Error 1

A model predicts:

```
Unstable

instead of

Stable
```

The researcher may ignore a potentially useful material.

---

### Error 2

A model predicts:

```
Stable

instead of

Unstable
```

The researcher may waste resources testing a poor candidate.

---

Both are incorrect predictions.

However,

their scientific consequences differ.

Accuracy treats them equally.

This is one of its major limitations.

---

# 8.107 When Accuracy Is Appropriate

Accuracy is useful when:

### Classes Are Balanced

Example:

```
500 stable materials

500 unstable materials
```

Both classes are represented equally.

---

### Errors Have Similar Consequences

If false positives and false negatives are equally undesirable,

accuracy provides a reasonable summary.

---

### Quick Model Comparison

During early experiments,

accuracy can provide a fast indication of whether the model is learning.

---

# 8.108 When Accuracy Should Not Be Used Alone

Accuracy becomes unreliable when:

- classes are highly imbalanced,
- one class is scientifically more important,
- false positives and false negatives have different costs.

In these situations,

we should examine additional metrics:

- precision,
- recall,
- F1 score,
- ROC-AUC.

---

# 8.109 Computing Accuracy with Scikit-learn

Scikit-learn provides a simple function.

```python
from sklearn.metrics import accuracy_score

actual = [

    1,

    0,

    1,

    1,

    0

]

predicted = [

    1,

    0,

    1,

    0,

    0

]

accuracy = accuracy_score(

    actual,

    predicted

)

print(accuracy)
```

Output:

```
0.8
```

This means:

```
80%

of predictions are correct.
```

---

# 8.110 Computing Accuracy Manually

Accuracy can also be calculated directly.

```python
correct = 0

for i in range(len(actual)):

    if actual[i] == predicted[i]:

        correct += 1


accuracy = correct / len(actual)

print(accuracy)
```

The logic is:

```
Count Correct Predictions

↓

Divide by Total Samples
```

---

# 8.111 Accuracy in Neural Networks

In deep learning,

models usually output probabilities.

Example:

```
Stable probability = 0.92
```

The prediction is converted into a class label using a threshold.

Commonly:

```
Probability > 0.5

↓

Positive Class
```

```
Probability < 0.5

↓

Negative Class
```

Accuracy is then calculated from the resulting labels.

---

# 8.112 Accuracy During Model Training

Accuracy is often monitored during training.

Example:

```
Epoch 1

Accuracy = 72%

Epoch 20

Accuracy = 89%

Epoch 50

Accuracy = 94%
```

Increasing accuracy suggests that the model is learning.

However,

training accuracy alone is not enough.

A model may memorize training data and achieve very high accuracy while failing on unseen materials.

Therefore,

accuracy should always be evaluated on:

- validation data,
- test data.

---

# 8.113 Advantages of Accuracy

Accuracy has several strengths.

### Simple Interpretation

Anyone can understand what it means.

---

### Easy Calculation

Only correct and incorrect predictions are required.

---

### Useful for Balanced Datasets

When classes are evenly distributed,

accuracy provides a reliable summary.

---

### Common Benchmark

Many classification studies report accuracy for comparison.

---

# 8.114 Limitations of Accuracy

Despite its simplicity,

accuracy has serious limitations.

### Ignores Class Distribution

A model can achieve high accuracy by predicting only the majority class.

---

### Ignores Error Type

False positives and false negatives are treated identically.

---

### Poor for Rare Materials Discovery

Rare but important materials may be completely missed.

---

# 8.115 Summary

Accuracy measures the fraction of predictions that are correctly classified.

Its formula is based on the confusion matrix:

```
Correct Predictions

↓

(TP + TN)

divided by

Total Predictions
```

Accuracy is useful when datasets are balanced and all mistakes have similar importance.

However,

in materials science applications,

important classes are often rare.

A model predicting superconductors, catalysts, or stable materials may achieve very high accuracy while failing at the actual scientific objective.

Therefore,

accuracy should rarely be reported alone.

The next metric,

**Precision**,

will answer a different and often more important question:

> "When the model predicts a material belongs to an important class, how often is it actually correct?"

# 8.116 Precision: Measuring the Reliability of Positive Predictions

In the previous section, we studied **accuracy**, the simplest classification metric.

Accuracy answers the question:

> **"Out of all predictions made by the model, how many were correct?"**

Although useful,

accuracy does not tell us whether the model's positive predictions are trustworthy.

This becomes extremely important in materials science.

Imagine a machine learning model designed to discover new battery materials.

The model predicts:

```
Promising Battery Material
```

for several candidates.

Before spending months on experimental testing,

researchers need to know:

> **"When the model says a material is promising, how often is it actually promising?"**

This question is answered by **precision**.

---

# 8.117 The Intuition Behind Precision

Precision measures the reliability of positive predictions.

In simple words:

> **Precision tells us how many predicted positive samples are truly positive.**

Consider a model searching for stable materials.

The model predicts:

```
100 materials

↓

Stable
```

After experimental verification,

only

```
80 materials

↓

Actually Stable
```

The remaining

```
20 materials

↓

Actually Unstable
```

were false alarms.

Precision asks:

```
Among all materials predicted as stable,

how many were truly stable?
```

---

# 8.118 Mathematical Definition of Precision

Precision is defined as:

```
Precision = TP / (TP + FP)
```

where:

- TP = True Positives
- FP = False Positives

The numerator represents:

```
Correct Positive Predictions
```

The denominator represents:

```
All Positive Predictions Made by the Model
```

Conceptually:

```
True Positives

↓

Divide by

↓

Everything Predicted Positive
```

---

# 8.119 Understanding False Positives

Precision is mainly concerned with one type of mistake:

```
False Positive (FP)
```

A false positive occurs when:

```
Model Prediction

↓

Positive

```

but the real answer is:

```
Negative
```

Example:

A machine learning model predicts:

```
Excellent Catalyst
```

but experimental testing shows:

```
Poor Catalyst
```

This is a false positive.

---

# 8.120 Why False Positives Matter in Materials Science

False positives can be extremely expensive.

Consider catalyst discovery.

A machine learning model screens

```
50,000 compounds
```

and identifies

```
200 candidates
```

for experimental testing.

If most of those candidates are actually poor catalysts,

researchers waste:

- laboratory chemicals,
- experimental time,
- financial resources,
- human effort.

A high-precision model reduces this problem.

---

# 8.121 Step-by-Step Precision Calculation

Suppose a model predicts whether materials are suitable for hydrogen storage.

The confusion matrix is:

| | Predicted Suitable | Predicted Unsuitable |
|-|-|-|
| Actual Suitable | TP = 90 | FN = 10 |
| Actual Unsuitable | FP = 30 | TN = 70 |

---

## Step 1: Identify True Positives

The model correctly identifies:

```
TP = 90
```

suitable materials.

---

## Step 2: Identify All Positive Predictions

The model predicted:

```
Suitable
```

for:

```
TP + FP
```

materials.

Therefore:

```
90 + 30

=

120
```

---

## Step 3: Calculate Precision

```
Precision

=

90 / 120

=

0.75
```

Therefore:

```
Precision = 75%
```

---

# 8.122 Interpretation of Precision

A precision of

```
75%
```

means:

> Among all materials predicted as suitable, 75% were actually suitable.

The remaining

```
25%
```

were false discoveries.

---

# 8.123 Precision Versus Accuracy

Precision and accuracy may appear similar,

but they answer completely different questions.

Accuracy:

```
How many total predictions were correct?
```

Precision:

```
How trustworthy are positive predictions?
```

Consider a rare materials discovery problem.

Dataset:

```
10,000 materials
```

Only:

```
100

↓

Useful materials
```

A model predicts:

```
200 materials

↓

Useful
```

Among them:

```
80 are actually useful
```

Precision:

```
80 / 200

=

40%
```

The model's positive predictions are not very reliable.

---

# 8.124 High Precision Example

Imagine a model searching for superconductors.

The model predicts:

```
20 materials

↓

Superconducting
```

Experimental verification shows:

```
19 are superconductors
```

Only one prediction was wrong.

Precision:

```
19 / 20

=

95%
```

This model produces highly reliable candidates.

Researchers can confidently prioritize these materials for further study.

---

# 8.125 Precision and Conservative Models

A model can achieve high precision by becoming very selective.

For example:

The model may only predict:

```
Extremely confident candidates
```

as positive.

This reduces false positives.

However,

it may also miss many actual positive samples.

This introduces a trade-off between:

```
Precision

and

Recall
```

---

# 8.126 The Precision-Recall Trade-Off

Consider a classifier predicting whether a material is promising.

The decision threshold determines how strict the model is.

Suppose the model requires:

```
Probability > 0.95
```

before predicting positive.

The model becomes very selective.

Result:

```
High Precision

Low Recall
```

---

Now reduce the threshold:

```
Probability > 0.50
```

The model identifies more candidates.

Result:

```
Higher Recall

Lower Precision
```

Therefore,

precision cannot be interpreted without considering recall.

---

# 8.127 Precision in Materials Discovery Workflow

Precision is especially important when:

- experimental validation is expensive,
- false discoveries waste resources,
- only the highest-confidence candidates should be tested.

Examples:

## Catalyst Discovery

High precision means:

```
Predicted catalysts

↓

Mostly active catalysts
```

---

## Battery Materials

High precision means:

```
Predicted candidates

↓

Mostly high-performance materials
```

---

## Superconductor Search

High precision means:

```
Predicted superconductors

↓

Mostly true superconductors
```

---

# 8.128 Computing Precision with Scikit-learn

Scikit-learn provides a built-in function.

```python
from sklearn.metrics import precision_score

actual = [

    1,

    0,

    1,

    1,

    0

]

predicted = [

    1,

    1,

    1,

    0,

    0

]

precision = precision_score(

    actual,

    predicted

)

print(precision)
```

The output represents the fraction of positive predictions that were correct.

---

# 8.129 Precision Calculation from Confusion Matrix

Precision can also be calculated manually.

```python
true_positive = 80

false_positive = 20


precision = true_positive / (

    true_positive + false_positive

)


print(precision)
```

The calculation follows:

```
Correct Positive Predictions

↓

Divide by

↓

All Positive Predictions
```

---

# 8.130 Precision in Machine Learning Libraries

For neural networks,

precision is usually calculated after obtaining predicted labels.

The workflow is:

```
Model Output

↓

Probability

↓

Classification Threshold

↓

Predicted Class

↓

Precision Calculation
```

During training,

precision is usually monitored alongside:

- loss,
- accuracy,
- recall,
- F1 score.

---

# 8.131 Advantages of Precision

Precision has several important strengths.

### Measures Trustworthiness

It tells researchers whether positive predictions can be trusted.

---

### Important for Expensive Experiments

It reduces wasted experimental effort.

---

### Useful for Rare Positive Classes

It prevents models from producing large numbers of false discoveries.

---

# 8.132 Limitations of Precision

Precision also has limitations.

### Ignores False Negatives

A model can have high precision while missing many useful materials.

---

### Can Be Increased by Predicting Very Few Positives

A model predicting only one extremely confident candidate may achieve high precision but have limited practical value.

---

### Must Be Used with Recall

Precision alone does not measure the ability to discover all important materials.

---

# 8.133 Precision Versus Recall: The Scientific Question

The difference between precision and recall can be summarized as:

Precision asks:

> "When the model finds something interesting, is it usually correct?"

Recall asks:

> "Did the model find most of the interesting things?"

These are different scientific objectives.

For materials discovery,

the correct choice depends on the research goal.

---

# 8.134 Summary

Precision measures the reliability of positive predictions.

Its formula:

```
Precision = TP / (TP + FP)
```

focuses on reducing false positives.

A high-precision model produces fewer false discoveries,

which is extremely valuable when experimental validation is expensive.

However,

precision alone does not tell us whether the model misses important materials.

A model can be highly precise but fail to discover many useful candidates.

Therefore,

the next metric,

**Recall**,

will examine the opposite question:

> "Among all truly positive materials, how many did the model successfully identify?"

# 8.135 Recall: Measuring the Ability to Find Important Materials

In the previous section, we introduced **precision**.

Precision answered the question:

> **"When the model predicts a material as positive, how often is that prediction correct?"**

In other words,

precision measures the reliability of positive predictions.

However,

there is another equally important question:

> **"Among all materials that are actually positive, how many did the model successfully discover?"**

This question is answered by **recall**.

---

# 8.136 The Intuition Behind Recall

Recall measures the ability of a model to identify all relevant positive samples.

In simple words:

> **Recall tells us how many of the truly important materials were found by the model.**

Consider a machine learning model designed to discover high-performance battery materials.

Suppose there are:

```
100 truly excellent battery materials
```

in a large database.

The model identifies:

```
85 materials
```

correctly.

However,

it misses:

```
15 materials
```

that could have been valuable.

Recall measures this discovery ability.

---

# 8.137 Mathematical Definition of Recall

Recall is defined as:

```
Recall = TP / (TP + FN)
```

where:

- TP = True Positives
- FN = False Negatives

The numerator represents:

```
Positive Materials Correctly Found
```

The denominator represents:

```
All Actually Positive Materials
```

Conceptually:

```
Correct Positive Discoveries

↓

Divide by

↓

Everything That Should Have Been Found
```

---

# 8.138 Understanding False Negatives

Recall focuses mainly on:

```
False Negatives (FN)
```

A false negative occurs when:

```
Actual Material

↓

Positive
```

but the model predicts:

```
Negative
```

Example:

A material is actually:

```
Excellent Hydrogen Storage Material
```

but the model predicts:

```
Not Useful
```

The model fails to discover an important candidate.

---

# 8.139 Why False Negatives Matter in Materials Science

False negatives can be scientifically costly.

Imagine a database containing:

```
1,000,000 hypothetical materials
```

Among them:

```
500

↓

Potentially revolutionary battery materials
```

A machine learning model identifies only:

```
50
```

The model has missed:

```
450
```

potential candidates.

Although the model may appear accurate,

it has failed at the actual goal:

**materials discovery.**

---

# 8.140 Step-by-Step Recall Calculation

Consider a classifier predicting whether materials are suitable for photovoltaic applications.

The confusion matrix is:

| | Predicted Suitable | Predicted Unsuitable |
|-|-|-|
| Actual Suitable | TP = 80 | FN = 20 |
| Actual Unsuitable | FP = 10 | TN = 90 |

---

## Step 1: Identify True Positives

The model correctly discovers:

```
TP = 80
```

suitable materials.

---

## Step 2: Identify All Actual Positive Samples

The total number of suitable materials is:

```
TP + FN
```

Therefore:

```
80 + 20

=

100
```

---

## Step 3: Calculate Recall

```
Recall

=

80 / 100

=

0.80
```

Therefore:

```
Recall = 80%
```

---

# 8.141 Interpretation of Recall

A recall of:

```
80%
```

means:

> The model successfully identified 80% of all truly suitable materials.

The remaining:

```
20%
```

were missed.

---

# 8.142 Precision Versus Recall

Precision and recall are often confused because both involve positive predictions.

However,

they measure completely different things.

---

## Precision

Question:

> "Are the positive predictions trustworthy?"

Focus:

```
False Positives
```

Formula:

```
TP / (TP + FP)
```

---

## Recall

Question:

> "Did we find most of the positive samples?"

Focus:

```
False Negatives
```

Formula:

```
TP / (TP + FN)
```

---

# 8.143 Example: Catalyst Discovery

Suppose a machine learning model searches for catalysts.

There are:

```
100 active catalysts
```

in a database.

The model predicts:

```
80 candidates
```

and all are actually active.

The result:

```
TP = 80

FP = 0

FN = 20
```

---

Precision:

```
80 / (80 + 0)

=

100%
```

Excellent precision.

Every selected candidate is useful.

---

Recall:

```
80 / (80 + 20)

=

80%
```

The model still missed 20 important catalysts.

---

This example shows:

A model can have perfect precision but incomplete discovery ability.

---

# 8.144 High Recall Models

A high-recall model attempts to find almost every important material.

It accepts that some false positives may occur.

For example:

```
Prediction threshold = 0.40
```

The model becomes less strict.

More materials are classified as positive.

Result:

```
Higher Recall

Lower Precision
```

---

# 8.145 Low Recall Problems

A low-recall model may appear accurate but fail scientifically.

Example:

A model predicts only:

```
10 superconductors
```

from a database containing:

```
1000 superconductors
```

It may produce very reliable predictions,

but it misses most discoveries.

For discovery problems,

this is usually unacceptable.

---

# 8.146 Recall in Materials Discovery

Recall becomes especially important when:

- missing a candidate is expensive,
- the positive class is rare,
- discovering all possible materials is the main objective.

Examples:

---

## Superconductor Discovery

Goal:

Find as many superconductors as possible.

High recall is valuable.

---

## Battery Material Screening

Goal:

Identify every promising candidate for further analysis.

High recall prevents missed opportunities.

---

## Drug-Like Material Discovery

Goal:

Avoid ignoring potentially useful compounds.

High recall is prioritized.

---

# 8.147 Precision-Recall Trade-Off

A fundamental relationship exists between precision and recall.

Improving one often reduces the other.

Consider changing the classification threshold.

---

### Strict Threshold

Example:

```
Probability > 0.95
```

Only highly confident predictions are accepted.

Result:

```
High Precision

Low Recall
```

---

### Relaxed Threshold

Example:

```
Probability > 0.50
```

More candidates are accepted.

Result:

```
Higher Recall

Lower Precision
```

---

# 8.148 Why There Is No Perfect Metric

Imagine a model for discovering new materials.

A researcher wants:

- every useful material to be discovered,
- every predicted material to be truly useful.

Unfortunately,

these goals compete.

Finding everything usually means accepting more false candidates.

Being extremely selective usually means missing some valuable materials.

Therefore,

machine learning evaluation requires understanding the scientific objective.

---

# 8.149 Computing Recall with Scikit-learn

Scikit-learn provides a built-in function.

```python
from sklearn.metrics import recall_score

actual = [

    1,

    0,

    1,

    1,

    0

]

predicted = [

    1,

    1,

    1,

    0,

    0

]

recall = recall_score(

    actual,

    predicted

)

print(recall)
```

The output represents the fraction of actual positive samples successfully identified.

---

# 8.150 Computing Recall Manually

Recall can be calculated directly from the confusion matrix.

```python
true_positive = 80

false_negative = 20


recall = true_positive / (

    true_positive + false_negative

)


print(recall)
```

The calculation follows:

```
Correct Positive Discoveries

↓

Divide by

↓

All Actual Positive Samples
```

---

# 8.151 Recall in Machine Learning Pipelines

A typical classification workflow is:

```
Material Features

↓

Machine Learning Model

↓

Prediction Probability

↓

Classification Threshold

↓

Predicted Labels

↓

Recall Calculation
```

The threshold choice strongly influences recall.

Therefore,

researchers should select thresholds based on scientific requirements.

---

# 8.152 Advantages of Recall

Recall has several important advantages.

### Measures Discovery Ability

It tells us whether important materials are being found.

---

### Useful for Rare Classes

It prevents models from ignoring minority classes.

---

### Important When Missing Candidates Is Costly

It is valuable in exploratory research.

---

# 8.153 Limitations of Recall

Recall also has limitations.

### Ignores False Positives

A model can achieve high recall by predicting almost everything as positive.

---

### Can Produce Many False Discoveries

Researchers may waste resources testing poor candidates.

---

### Must Be Considered With Precision

Recall alone does not indicate prediction reliability.

---

# 8.154 Summary

Recall measures how effectively a classification model identifies all truly positive samples.

Its formula:

```
Recall = TP / (TP + FN)
```

focuses on reducing false negatives.

In materials science,

high recall is especially valuable when missing a potentially important material would be costly.

However,

high recall alone is not enough.

A model that identifies every possible candidate may also produce many false discoveries.

Therefore,

precision and recall should usually be evaluated together.

The next section introduces the **F1 Score**,

a metric that combines precision and recall into a single balanced measurement.

# 8.155 F1 Score: Balancing Precision and Recall

In the previous two sections, we studied two of the most important classification metrics:

- Precision
- Recall

Precision and recall measure different aspects of a model's performance.

Precision asks:

> **"When the model predicts a material as important, how often is it correct?"**

Recall asks:

> **"Among all important materials, how many did the model successfully find?"**

Both metrics are valuable.

However, they often move in opposite directions.

Improving precision may reduce recall.

Improving recall may reduce precision.

This creates a challenge:

> **How can we evaluate a model when both avoiding false discoveries and finding all important materials matter?**

The answer is the:

# F1 Score

---

# 8.156 The Need for a Combined Metric

Consider a machine learning model designed to discover new superconducting materials.

Two models are tested.

---

## Model A

```
Precision = 95%

Recall = 40%
```

This model is extremely reliable.

Almost every predicted superconductor is truly superconducting.

However,

it discovers less than half of all superconductors.

---

## Model B

```
Precision = 50%

Recall = 95%
```

This model discovers almost every superconductor.

However,

half of its predictions are incorrect.

---

Which model is better?

The answer depends on the scientific objective.

But if we need a single number representing a balance between these two properties,

we use the F1 score.

---

# 8.157 The Intuition Behind F1 Score

The F1 score combines:

```
Precision

and

Recall
```

into one metric.

However,

it does not simply calculate the average.

Instead,

it uses the **harmonic mean**.

The harmonic mean gives stronger importance to low values.

This means:

A model cannot achieve a high F1 score by having excellent precision but extremely poor recall.

Both must be reasonably good.

---

# 8.158 Why Not Use the Simple Average?

Suppose a model has:

```
Precision = 100%

Recall = 10%
```

The simple average would be:

```
(100 + 10) / 2

=

55%
```

This looks acceptable.

But scientifically,

this model is poor.

It finds very few actual positive materials.

The harmonic mean used by F1 prevents this misleading situation.

---

# 8.159 Mathematical Definition of F1 Score

The F1 score is defined as:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Since:

```
Precision = TP / (TP + FP)
```

and

```
Recall = TP / (TP + FN)
```

the F1 score depends on all four confusion matrix values:

- TP
- TN
- FP
- FN

However,

notice that true negatives do not appear directly.

This is because F1 focuses specifically on the positive class.

---

# 8.160 Step-by-Step F1 Calculation

Suppose a material classification model gives:

```
Precision = 0.80

Recall = 0.60
```

The F1 score is:

```
F1 = 2 × (0.80 × 0.60)

/

(0.80 + 0.60)
```

First:

```
0.80 × 0.60

=

0.48
```

Then:

```
2 × 0.48

=

0.96
```

Denominator:

```
0.80 + 0.60

=

1.40
```

Therefore:

```
F1

=

0.96 / 1.40

=

0.686
```

The final result:

```
F1 = 0.686
```

---

# 8.161 Interpreting F1 Score

The F1 score ranges between:

```
0

and

1
```

Higher values indicate better balance.

| F1 Score | Interpretation |
|-|-|
| 1.0 | Perfect precision and recall |
| 0.9+ | Excellent classifier |
| 0.7–0.9 | Good performance |
| 0.5–0.7 | Moderate performance |
| Below 0.5 | Poor balance |

---

# 8.162 Perfect F1 Score

The best possible case:

```
Precision = 1

Recall = 1
```

Therefore:

```
F1

=

1
```

This means:

- every predicted positive sample is correct,
- every actual positive sample is discovered.

The classifier is perfect.

---

# 8.163 F1 Score and Materials Discovery

F1 score is especially useful in materials problems where both types of errors matter.

Examples:

---

## New Material Discovery

A model should:

- find promising materials,
- avoid overwhelming researchers with false candidates.

Both precision and recall matter.

---

## Phase Classification

Suppose a model predicts:

```
Crystal Phase A

or

Crystal Phase B
```

Missing many samples and making many false predictions are both undesirable.

---

## Defect Detection

A model identifying defects in materials images should:

- detect real defects,
- avoid falsely labeling perfect materials as defective.

---

# 8.164 F1 Score Under Class Imbalance

F1 score is particularly useful when classes are imbalanced.

Recall the superconducting example:

```
9900 non-superconductors

100 superconductors
```

Accuracy may reach:

```
99%
```

even with a useless model.

However,

F1 score focuses on the minority positive class.

A model that never identifies superconductors has:

```
Precision = 0

Recall = 0
```

Therefore:

```
F1 = 0
```

The metric correctly identifies the failure.

---

# 8.165 F1 Score Versus Accuracy

Consider the following comparison.

| Metric | Question Answered |
|-|-|
| Accuracy | How many predictions are correct overall? |
| Precision | How reliable are positive predictions? |
| Recall | How many positives were discovered? |
| F1 Score | How well are precision and recall balanced? |

No single metric is universally best.

The correct choice depends on the scientific objective.

---

# 8.166 Computing F1 Score with Scikit-learn

Scikit-learn provides a built-in function.

```python
from sklearn.metrics import f1_score

actual = [

    1,

    0,

    1,

    1,

    0

]

predicted = [

    1,

    1,

    1,

    0,

    0

]

f1 = f1_score(

    actual,

    predicted

)

print(f1)
```

The function automatically calculates:

```
Precision

+

Recall

↓

F1 Score
```

---

# 8.167 Computing F1 Score Manually

The calculation can also be performed directly.

```python
precision = 0.8

recall = 0.6


f1 = 2 * (

    precision * recall

) / (

    precision + recall

)


print(f1)
```

This follows the mathematical definition exactly.

---

# 8.168 F1 Score in Multi-Class Problems

Many materials classification problems contain more than two classes.

Example:

```
Metal

Semiconductor

Insulator
```

In such cases,

we calculate F1 scores for each class.

Then we combine them using averaging strategies.

Common approaches include:

---

## Macro F1 Score

Every class receives equal importance.

Useful when rare classes are scientifically important.

---

## Weighted F1 Score

Classes are weighted according to their frequency.

Useful when class sizes are very different.

---

## Micro F1 Score

Calculates metrics globally across all samples.

Useful when overall performance is the main objective.

---

# 8.169 Advantages of F1 Score

F1 score has several advantages.

### Balances Precision and Recall

Neither metric can be ignored.

---

### Useful for Imbalanced Datasets

It focuses on the positive class.

---

### Better Than Accuracy for Rare Events

Important in materials discovery problems.

---

### Single Summary Number

Allows quick comparison between models.

---

# 8.170 Limitations of F1 Score

Despite its usefulness,

F1 score has limitations.

### Ignores True Negatives

It focuses only on positive-class performance.

---

### Does Not Show Error Type

Two models may have identical F1 scores but different scientific behaviors.

---

### Depends on Threshold Choice

Changing the classification threshold changes precision, recall, and therefore F1.

---

# 8.171 Precision, Recall, and F1 Together

A proper materials classification study often reports:

```
Accuracy

Precision

Recall

F1 Score
```

Together they provide a complete picture.

For example:

| Metric | Value |
|-|-:|
| Accuracy | 96% |
| Precision | 85% |
| Recall | 78% |
| F1 Score | 81% |

Interpretation:

The model performs well overall,

but some important materials are still missed.

---

# 8.172 Summary

The F1 score combines precision and recall into a single balanced metric.

Its formula:

```
F1 = 2 × (Precision × Recall)

/

(Precision + Recall)
```

makes it especially valuable when dealing with:

- rare materials classes,
- expensive experiments,
- discovery problems,
- imbalanced datasets.

Unlike accuracy,

F1 does not allow a model to appear successful simply by predicting the majority class.

However,

F1 is still dependent on the chosen classification threshold.

To evaluate a classifier across different thresholds,

we need a more comprehensive approach.

The next section introduces the **ROC Curve and ROC-AUC**, which measure how well a classifier separates classes across all possible decision thresholds.

# 8.173 ROC Curve and ROC-AUC: Evaluating Classification Across Thresholds

In the previous section, we studied the F1 score, which provides a balance between precision and recall.

The F1 score is extremely useful when:

- classes are imbalanced,
- false positives and false negatives are both important,
- we need a single performance measure.

However, F1 score has one important limitation.

It depends on a specific classification threshold.

To understand this limitation,

we must first understand how machine learning models actually produce predictions.

---

# 8.174 From Probability to Class Prediction

Most classification models do not directly output:

```
Positive

or

Negative
```

Instead,

they first produce a probability.

For example,

a model predicting whether a material is superconducting may output:

```
Material A

Probability of superconductivity

=

0.92
```

```
Material B

Probability of superconductivity

=

0.64
```

```
Material C

Probability of superconductivity

=

0.35
```

The final class depends on a chosen threshold.

---

# 8.175 Classification Threshold

A threshold determines how probability values are converted into class labels.

The most common threshold is:

```
0.50
```

Meaning:

```
Probability ≥ 0.50

↓

Positive Class
```

and

```
Probability < 0.50

↓

Negative Class
```

---

Consider:

```
Material A

Probability = 0.92
```

Prediction:

```
Positive
```

---

```
Material B

Probability = 0.35
```

Prediction:

```
Negative
```

---

The threshold controls how strict the model is.

---

# 8.176 Changing the Threshold

The threshold does not have to be 0.50.

Suppose we reduce it.

From:

```
0.50

↓

0.30
```

Now more materials will be classified as positive.

The model becomes more sensitive.

Consequences:

```
Recall increases
```

because fewer positive materials are missed.

However:

```
False positives increase
```

so:

```
Precision decreases
```

---

Now increase the threshold:

```
0.50

↓

0.90
```

The model becomes more selective.

Only extremely confident predictions are accepted.

Consequences:

```
Precision increases
```

but:

```
Recall decreases
```

---

# 8.177 The Need for Threshold-Independent Evaluation

Because changing the threshold changes performance,

a single value such as:

```
F1 = 0.82
```

does not describe the complete behavior of the classifier.

A researcher may ask:

> How well does the model separate positive and negative materials regardless of the chosen threshold?

This is the purpose of:

- ROC Curve
- ROC-AUC

---

# 8.178 What Is the ROC Curve?

ROC stands for:

```
Receiver Operating Characteristic
```

The ROC curve shows the relationship between:

```
True Positive Rate

and

False Positive Rate
```

at different classification thresholds.

Instead of evaluating one threshold,

we evaluate the model across all possible thresholds.

---

# 8.179 True Positive Rate (TPR)

True Positive Rate is another name for recall.

It measures:

> How many actual positive samples were correctly identified?

Formula:

```
TPR = TP / (TP + FN)
```

A higher TPR means the model discovers more positive materials.

---

# 8.180 False Positive Rate (FPR)

False Positive Rate measures:

> How many negative samples were incorrectly classified as positive?

Formula:

```
FPR = FP / (FP + TN)
```

A lower FPR is desirable.

---

# 8.181 Building the ROC Curve

To create an ROC curve,

we repeatedly change the classification threshold.

For each threshold:

1. Calculate TPR.
2. Calculate FPR.
3. Plot the point.

Example:

| Threshold | TPR | FPR |
|-|-:|-:|
| 0.90 | 0.40 | 0.02 |
| 0.70 | 0.65 | 0.08 |
| 0.50 | 0.80 | 0.20 |
| 0.30 | 0.95 | 0.45 |

These points form the ROC curve.

---

# 8.182 Understanding the ROC Plot

The ROC graph contains:

Horizontal axis:

```
False Positive Rate
```

Vertical axis:

```
True Positive Rate
```

A perfect classifier reaches:

```
TPR = 1

FPR = 0
```

meaning:

- all positive materials are discovered,
- no negative materials are incorrectly selected.

---

# 8.183 Ideal ROC Curve

A perfect model follows a curve near the upper-left corner.

Conceptually:

```
TPR

1 |        ______
  |       /
  |      /
  |_____/
0 |
  ----------------
        FPR
```

The model achieves:

- high true positive rate,
- low false positive rate.

---

# 8.184 Random Classifier

A random classifier has no ability to distinguish between classes.

Its ROC curve follows approximately:

```
Diagonal line
```

Meaning:

```
TPR ≈ FPR
```

The model performs no better than guessing.

---

# 8.185 What Is ROC-AUC?

AUC means:

```
Area Under the Curve
```

ROC-AUC summarizes the entire ROC curve using one number.

Instead of comparing curves visually,

we calculate the area beneath them.

---

# 8.186 Interpretation of ROC-AUC Values

ROC-AUC usually ranges from:

```
0

to

1
```

Interpretation:

| ROC-AUC | Meaning |
|-|-|
| 1.0 | Perfect classifier |
| 0.9–1.0 | Excellent |
| 0.8–0.9 | Good |
| 0.7–0.8 | Acceptable |
| 0.5 | Random guessing |
| <0.5 | Worse than random |

---

# 8.187 Scientific Meaning of ROC-AUC

A ROC-AUC value can be interpreted as:

> The probability that the model assigns a higher score to a randomly chosen positive sample than to a randomly chosen negative sample.

For materials discovery,

this means:

If ROC-AUC is high,

the model successfully ranks promising materials above poor candidates.

---

# 8.188 Example: Battery Material Discovery

Suppose a model predicts lithium battery electrode materials.

Testing:

```
1000 materials
```

The goal is to identify:

```
High-performance electrodes
```

The model produces:

```
ROC-AUC = 0.94
```

Interpretation:

The model has excellent ability to distinguish high-performance and low-performance materials.

Even if the classification threshold changes,

the model maintains strong separation ability.

---

# 8.189 ROC-AUC Versus Accuracy

Accuracy:

```
Evaluates one threshold
```

ROC-AUC:

```
Evaluates all thresholds
```

Therefore,

ROC-AUC is often more informative when:

- class imbalance exists,
- threshold selection is uncertain,
- ranking candidates is important.

---

# 8.190 ROC-AUC Versus F1 Score

Both metrics are useful but answer different questions.

| Metric | Question |
|-|-|
| F1 Score | How well does the model balance precision and recall at one threshold? |
| ROC-AUC | How well does the model separate classes across all thresholds? |

---

# 8.191 Computing ROC-AUC with Scikit-learn

Scikit-learn provides a direct implementation.

```python
from sklearn.metrics import roc_auc_score

actual = [

    1,

    0,

    1,

    1,

    0

]


probabilities = [

    0.90,

    0.30,

    0.80,

    0.40,

    0.20

]


auc = roc_auc_score(

    actual,

    probabilities

)

print(auc)
```

Important:

ROC-AUC requires probabilities,

not only final class labels.

---

# 8.192 Why Probabilities Are Required

Suppose a model outputs:

```
Positive

Negative

Positive

Negative
```

The threshold information is lost.

ROC analysis requires knowing:

```
How confident was the model?
```

Therefore,

we use:

```
Probability scores
```

instead of:

```
Final labels
```

---

# 8.193 Advantages of ROC-AUC

ROC-AUC has several strengths.

### Threshold Independent

It evaluates the entire classification behavior.

---

### Useful for Ranking Problems

Important in materials screening.

---

### Handles Different Decision Thresholds

Researchers can choose thresholds later depending on experimental resources.

---

### Good for Model Comparison

Two classifiers can be compared using one number.

---

# 8.194 Limitations of ROC-AUC

Despite its usefulness,

ROC-AUC has limitations.

### Can Be Misleading With Extreme Imbalance

A model may have excellent ROC-AUC but poor performance on the rare positive class.

---

### Does Not Show Actual Prediction Quality

A high ROC-AUC does not guarantee high precision.

---

### Less Direct Scientific Interpretation

Unlike MAE or RMSE,

it does not directly represent physical prediction error.

---

# 8.195 ROC-AUC in Materials Informatics

ROC-AUC is especially valuable for:

- screening large databases,
- ranking candidate materials,
- rare materials discovery,
- binary property prediction.

Examples:

```
Stable

vs

Unstable
```

```
Magnetic

vs

Non-magnetic
```

```
Catalytic

vs

Non-catalytic
```

---

# 8.196 Complete Classification Evaluation Strategy

A strong materials classification study should usually report:

| Metric | Purpose |
|-|-|
| Accuracy | Overall correctness |
| Precision | Reliability of positive predictions |
| Recall | Ability to discover positives |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Overall class separation ability |

Each metric provides different information.

---

# 8.197 Summary

ROC-AUC evaluates how well a classifier separates positive and negative classes across all possible decision thresholds.

The ROC curve plots:

```
True Positive Rate

against

False Positive Rate
```

and the area under this curve gives ROC-AUC.

Unlike accuracy and F1 score,

ROC-AUC does not depend on a single threshold.

This makes it particularly useful in materials discovery problems where researchers may adjust screening criteria depending on experimental limitations.

However,

ROC-AUC should not replace precision and recall.

A complete evaluation requires multiple metrics that reflect both mathematical performance and scientific objectives.

---

# 8.198 Transition to Advanced Validation

So far,

we have discussed evaluation metrics.

However,

a model can achieve excellent metrics on a particular test set and still fail when applied to new materials.

This problem is especially severe in materials informatics because:

- DFT datasets are often small,
- chemical spaces are enormous,
- similar structures can appear in both training and testing data.

Therefore,

after understanding evaluation metrics,

the next major topic is:

**Model Validation**

where we will study:

- cross-validation,
- K-fold validation,
- leave-one-out validation,
- nested validation,

and why careful validation is essential for reliable materials machine learning.

# 8.199 Model Validation: Why Evaluation Metrics Are Not Enough

In the previous sections, we studied classification and regression evaluation metrics.

For regression, we discussed:

- MAE
- MSE
- RMSE
- R²

For classification, we discussed:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

These metrics tell us how well a model performs.

However, an important question remains:

> **Are these performance measurements reliable?**

A machine learning model can show excellent evaluation scores and still fail when applied to new materials.

The reason is:

**The model may have learned the dataset instead of learning the underlying physical relationship.**

This problem is called:

```
Overfitting
```

To prevent this problem,

we need proper model validation.

---

# 8.200 What Is Model Validation?

Model validation is the process of testing whether a machine learning model can generalize beyond the data it has already seen.

The fundamental goal is:

```
Training Data

↓

Learn Patterns

↓

Validation

↓

Test on Unseen Data

↓

Estimate Real Performance
```

A reliable model should not only perform well on known examples.

It should also make accurate predictions for new materials.

---

# 8.201 Training Error vs Real Performance

Consider a machine learning model predicting material stability.

During training:

```
Training Accuracy

=

99.8%
```

This looks impressive.

However,

when tested on new materials:

```
Test Accuracy

=

65%
```

A huge performance drop occurs.

Why?

The model memorized the training examples instead of learning general chemical principles.

---

# 8.202 The Problem of Overfitting

Overfitting occurs when a model becomes too specialized to the training dataset.

The model learns:

- noise,
- measurement errors,
- accidental patterns,
- dataset-specific features.

Instead of learning:

- chemical trends,
- structural relationships,
- physical mechanisms.

Conceptually:

```
Training Data

↓

Model Memorizes

↓

Poor Generalization
```

---

# 8.203 Example: Materials Informatics Overfitting

Imagine training an XGBoost model to predict formation energy.

The dataset contains:

```
10,000 crystal structures
```

The model achieves:

```
Training RMSE

=

0.01 eV/atom
```

This appears excellent.

However,

on new structures:

```
Test RMSE

=

0.35 eV/atom
```

The model has learned the training database rather than the physical relationship between structure and energy.

---

# 8.204 Why Materials Science Has a Serious Validation Problem

Materials informatics has unique challenges compared with many other machine learning fields.

---

## Small Dataset Size

Many materials datasets contain only:

```
Hundreds

or

Thousands
```

of samples.

Compared with fields such as image recognition,

this is extremely small.

---

## Expensive Labels

Obtaining new data may require:

- DFT calculations,
- experiments,
- synthesis,
- characterization.

Therefore,

we cannot simply collect millions of examples.

---

## High-Dimensional Chemical Space

The possible number of materials is enormous.

For example:

Different combinations of:

- elements,
- crystal structures,
- defects,
- compositions,

create an almost unlimited design space.

---

## Similar Materials

Many materials datasets contain highly related structures.

For example:

```
FeO

Fe2O3

Fe3O4
```

These materials share chemical similarities.

Randomly splitting data can accidentally place very similar compounds into both training and testing sets.

This creates overly optimistic performance.

---

# 8.205 Data Splitting

The simplest validation approach is dividing the dataset into different subsets.

Usually:

```
Training Set

+

Validation Set

+

Test Set
```

---

# 8.206 Training Set

The training set is used to teach the model.

The algorithm learns relationships between:

Input features

and

Target values.

Example:

Input:

```
Composition

Crystal Structure

Atomic Features
```

Output:

```
Formation Energy
```

The model adjusts its parameters using this data.

---

# 8.207 Validation Set

The validation set is used during model development.

It helps answer:

- Is the model improving?
- Is it overfitting?
- Which hyperparameters work best?

Examples of hyperparameters:

For XGBoost:

```
Maximum tree depth

Learning rate

Number of estimators

Subsample ratio
```

The validation set guides these choices.

---

# 8.208 Test Set

The test set is the final unbiased evaluation.

It should only be used after:

- model selection,
- hyperparameter tuning,
- feature selection.

The test set represents completely unseen materials.

---

# 8.209 Why We Cannot Test on Training Data

Suppose we train an XGBoost model.

Then evaluate it on the same data.

The result:

```
Training R²

=

0.99
```

Does this mean the model is excellent?

No.

The model has already seen those examples.

This is similar to giving a student the exact questions before an exam.

The score does not represent true understanding.

---

# 8.210 The Need for Cross-Validation

When datasets are small,

a single train-test split may produce unreliable results.

Example:

Dataset:

```
500 materials
```

Suppose we randomly divide:

```
80%

Training

20%

Testing
```

The test set contains only:

```
100 materials
```

A few unusual materials can strongly influence the final score.

A different random split may give a very different result.

To solve this problem,

we use:

```
Cross-validation
```

---

# 8.211 What Is Cross-Validation?

Cross-validation repeatedly divides the dataset into training and validation portions.

The model is trained and evaluated multiple times.

Instead of asking:

> "How well did the model perform on one split?"

we ask:

> "How consistently does the model perform across many different splits?"

Conceptually:

```
Dataset

↓

Multiple Splits

↓

Train + Validate

↓

Average Performance
```

---

# 8.212 Advantages of Cross-Validation

Cross-validation provides:

### More Reliable Estimates

Performance is averaged over multiple experiments.

---

### Better Use of Small Datasets

Every sample contributes to both training and validation.

---

### Reduced Dependence on Random Splitting

One lucky or unlucky split has less influence.

---

# 8.213 Cross-Validation Workflow

A typical workflow is:

```
Original Dataset

↓

Split into folds

↓

Train Model

↓

Validate Model

↓

Repeat

↓

Average Results
```

The number of splits is called:

```
k
```

This leads to:

```
K-Fold Cross-Validation
```

---

# 8.214 Materials Perspective

Cross-validation is extremely important in materials machine learning.

Suppose we train a model predicting:

```
Formation Energy
```

using:

- elemental properties,
- atomic radius,
- electronegativity,
- crystal descriptors.

A single split may accidentally place similar compounds in training and testing.

The model appears extremely accurate.

Cross-validation exposes whether the model truly generalizes.

---

# 8.215 Example: Formation Energy Prediction

Suppose a model gives:

Single train-test split:

```
RMSE = 0.05 eV/atom
```

This looks excellent.

However,

5-fold cross-validation gives:

```
Fold 1:

0.08 eV/atom


Fold 2:

0.12 eV/atom


Fold 3:

0.15 eV/atom


Fold 4:

0.10 eV/atom


Fold 5:

0.13 eV/atom
```

Average:

```
≈0.116 eV/atom
```

The true performance estimate is closer to:

```
0.116 eV/atom
```

not:

```
0.05 eV/atom
```

---

# 8.216 Summary

Evaluation metrics tell us how well a model performs.

However,

without proper validation,

these metrics may be misleading.

A model can achieve excellent results by memorizing training data.

This problem is especially severe in materials informatics because:

- datasets are small,
- chemical space is enormous,
- materials are highly related,
- experimental validation is expensive.

Model validation ensures that machine learning models learn general physical relationships rather than dataset-specific patterns.

The next section introduces the most widely used validation method:

**K-Fold Cross-Validation**

and explains how it is implemented in practical materials machine learning workflows.

# 8.217 K-Fold Cross-Validation: Reliable Performance Estimation

In the previous section, we introduced the concept of model validation and explained why a single train-test split can sometimes produce unreliable conclusions.

This problem becomes especially serious in materials informatics because datasets are often small.

For example:

```
DFT Dataset

↓

500 materials
```

If we randomly separate:

```
80%

Training

20%

Testing
```

the test set contains only:

```
100 materials
```

A few unusual materials can significantly affect the evaluation score.

A model may appear excellent simply because the random split was favorable.

To obtain a more reliable estimate of model performance,

we use:

# K-Fold Cross-Validation

---

# 8.218 The Basic Idea of K-Fold Cross-Validation

K-fold cross-validation divides the dataset into:

```
K

equal-sized groups
```

called:

```
folds
```

The model is trained and evaluated multiple times.

During each iteration:

- one fold is used for validation,
- the remaining folds are used for training.

After all iterations,

the performance values are averaged.

Conceptually:

```
Dataset

↓

Split into K folds

↓

Train + Validate K times

↓

Average Performance

↓

Final Score
```

---

# 8.219 Why Is It Called K-Fold?

The letter:

```
K
```

represents the number of groups.

Common choices are:

```
K = 5
```

or

```
K = 10
```

For example:

```
10-fold cross-validation
```

means the dataset is divided into ten parts.

---

# 8.220 Example of 5-Fold Cross-Validation

Suppose we have:

```
500 materials
```

and choose:

```
K = 5
```

The dataset is divided into:

```
Fold 1

100 materials
```

```
Fold 2

100 materials
```

```
Fold 3

100 materials
```

```
Fold 4

100 materials
```

```
Fold 5

100 materials
```

---

The model is trained five separate times.

---

## Iteration 1

Training:

```
Fold 2 + Fold 3 + Fold 4 + Fold 5
```

Validation:

```
Fold 1
```

---

## Iteration 2

Training:

```
Fold 1 + Fold 3 + Fold 4 + Fold 5
```

Validation:

```
Fold 2
```

---

## Iteration 3

Training:

```
Fold 1 + Fold 2 + Fold 4 + Fold 5
```

Validation:

```
Fold 3
```

---

## Iteration 4

Training:

```
Fold 1 + Fold 2 + Fold 3 + Fold 5
```

Validation:

```
Fold 4
```

---

## Iteration 5

Training:

```
Fold 1 + Fold 2 + Fold 3 + Fold 4
```

Validation:

```
Fold 5
```

---

Finally,

all five validation scores are averaged.

---

# 8.221 Mathematical Representation

Suppose the model produces validation scores:

```
Score₁

Score₂

Score₃

...

Scoreₖ
```

The final cross-validation score is:

```
Average Score

=

(Score₁ + Score₂ + ... + Scoreₖ)

/

K
```

For example:

Five RMSE values:

```
0.12

0.15

0.10

0.14

0.13
```

Average:

```
(0.12 + 0.15 + 0.10 + 0.14 + 0.13)

/

5

=

0.128
```

The estimated model error becomes:

```
RMSE = 0.128 eV/atom
```

---

# 8.222 Why K-Fold Is Better Than One Split

Consider a dataset containing rare materials.

A random split may accidentally produce:

Training set:

```
Mostly common materials
```

Testing set:

```
Several unusual materials
```

The model may appear worse than it actually is.

Another split may produce:

Training set:

```
Contains diverse materials
```

Testing set:

```
Easy examples
```

The model appears better than it really is.

K-fold validation reduces this randomness by testing the model multiple times.

---

# 8.223 K-Fold Cross-Validation in Materials Science

Materials datasets often contain strong chemical relationships.

Example:

A dataset contains:

```
Titanium oxides
```

including:

```
TiO

Ti2O3

TiO2
```

These materials are chemically related.

A random split may place:

Training:

```
TiO

Ti2O3
```

Testing:

```
TiO2
```

The model benefits from seeing very similar compounds.

The resulting accuracy may be overly optimistic.

---

# 8.224 Random K-Fold vs Chemical Generalization

Standard K-fold assumes that samples are independent.

However,

materials are not always independent.

Many compounds share:

- elements,
- structures,
- crystal prototypes,
- chemical environments.

Therefore,

ordinary random K-fold may not fully represent real-world prediction.

Researchers often use more advanced strategies such as:

- composition-based splitting,
- structure-based splitting,
- time-based splitting.

These approaches test whether the model can predict genuinely new materials.

---

# 8.225 Choosing the Value of K

The value of K affects the bias and variance of the estimate.

Common choices:

```
K = 5
```

and

```
K = 10
```

---

# 8.226 Five-Fold Cross-Validation

Advantages:

- computationally cheaper,
- commonly used,
- suitable for medium-sized datasets.

Example:

```
10,000 materials
```

Training five models is usually manageable.

---

# 8.227 Ten-Fold Cross-Validation

Advantages:

- generally provides a more reliable estimate,
- common in scientific machine learning studies.

Disadvantage:

Requires more computation.

---

# 8.228 Large K Values

If:

```
K = 100
```

the model is trained many times.

This increases computational cost.

The improvement in reliability may not justify the extra time.

---

# 8.229 K-Fold Cross-Validation with XGBoost

For materials datasets,

XGBoost combined with K-fold validation is extremely common.

Example workflow:

```
Pymatgen

↓

Generate Material Features

↓

XGBoost Model

↓

K-Fold Validation

↓

MAE / RMSE / R²
```

This provides a more reliable estimate of predictive performance.

---

# 8.230 Implementing K-Fold Cross-Validation in Scikit-learn

Example:

```python
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor


model = RandomForestRegressor()


kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


scores = cross_val_score(

    model,

    X,

    y,

    cv=kf,

    scoring="neg_mean_absolute_error"

)


print(scores)
```

The model is trained and evaluated five times automatically.

---

# 8.231 Understanding Negative Scores in Scikit-learn

Scikit-learn represents loss metrics as negative values.

For example:

Output:

```
[-0.12 -0.15 -0.10 -0.14 -0.13]
```

These correspond to:

```
MAE:

0.12

0.15

0.10

0.14

0.13
```

The negative sign exists because Scikit-learn assumes:

```
Higher score = better
```

while errors should be minimized.

---

# 8.232 Calculating Average Performance

After cross-validation:

```python
import numpy as np


mean_mae = -np.mean(

    scores

)


print(mean_mae)
```

Output:

```
0.128
```

This represents the average validation error.

---

# 8.233 K-Fold Validation for Classification

K-fold is not limited to regression.

It can also evaluate classification models.

Example:

Predict:

```
Stable

vs

Unstable
```

The folds maintain different training and validation groups.

Metrics can include:

- Accuracy,
- Precision,
- Recall,
- F1,
- ROC-AUC.

---

# 8.234 Stratified K-Fold

For classification,

ordinary K-fold can create problems when classes are imbalanced.

Example:

Dataset:

```
9900 unstable materials

100 stable materials
```

A random fold may contain very few stable materials.

To solve this,

we use:

```
Stratified K-Fold
```

---

# 8.235 What Is Stratified K-Fold?

Stratified K-fold preserves the class distribution in every fold.

Example:

Original dataset:

```
99% Negative

1% Positive
```

Each fold maintains approximately:

```
99% Negative

1% Positive
```

This produces more reliable classification evaluation.

---

# 8.236 Materials Example: Stable Material Classification

Suppose:

```
20,000 compounds
```

are classified as:

```
Stable

or

Unstable
```

Only:

```
500

↓

Stable
```

A normal K-fold split may accidentally create folds with very few stable materials.

Stratified K-fold ensures every fold contains stable examples.

---

# 8.237 Advantages of K-Fold Cross-Validation

K-fold provides:

### Better Use of Limited Data

Every sample contributes to validation.

---

### More Stable Performance Estimates

Results are averaged over multiple splits.

---

### Reduced Randomness

One unusual split cannot dominate the result.

---

### Suitable for Small Scientific Datasets

Very useful for materials informatics.

---

# 8.238 Limitations of K-Fold Cross-Validation

Despite its advantages,

K-fold has limitations.

---

## Increased Computational Cost

The model is trained:

```
K times
```

instead of once.

For expensive models,

this can be significant.

---

## Does Not Solve Dataset Bias

If the dataset itself is biased,

cross-validation cannot fix it.

---

## Random Splits May Be Physically Unrealistic

Chemically similar materials may still appear in both training and validation sets.

---

# 8.239 Summary

K-fold cross-validation improves the reliability of machine learning evaluation by repeatedly training and validating a model on different subsets of the data.

Instead of depending on one random split,

the model performance is averaged across multiple folds.

This is especially important in materials informatics because:

- datasets are often small,
- materials are chemically related,
- accurate generalization is difficult.

However,

standard K-fold validation may still overestimate performance when chemically similar materials appear in both training and validation sets.

For extremely small datasets,

an even more intensive approach is sometimes used:

**Leave-One-Out Cross-Validation (LOOCV)**.

The next section will discuss LOOCV and why it is important for small DFT datasets.

# 8.240 Leave-One-Out Cross-Validation (LOOCV): Maximum Data Utilization for Small Datasets

In the previous section, we discussed **K-fold cross-validation**.

K-fold validation divides the dataset into several groups and repeatedly trains and evaluates the model.

For example:

```
5-fold cross-validation
```

means:

```
Dataset

↓

5 separate validation experiments

↓

Average performance
```

K-fold validation is widely used because it provides a good balance between:

- reliability,
- computational cost.

However,

many materials science datasets are extremely small.

Examples include:

- experimental datasets,
- high-level quantum calculations,
- expensive DFT calculations,
- rare material classes.

When the dataset contains only a few hundred or even a few dozen samples,

we may want to use every possible data point for training.

This motivates a more extreme validation strategy:

# Leave-One-Out Cross-Validation (LOOCV)

---

# 8.241 The Basic Idea of LOOCV

Leave-One-Out Cross-Validation is a special case of K-fold cross-validation where:

```
K = Number of Samples
```

Instead of dividing the dataset into large groups,

we leave exactly one sample out for validation.

The model is trained on:

```
All samples except one
```

and tested on:

```
The single remaining sample
```

This process is repeated until every sample has been used once as the validation point.

---

# 8.242 Example of LOOCV

Suppose we have a very small materials dataset:

```
5 materials
```

The dataset contains:

```
Material A

Material B

Material C

Material D

Material E
```

LOOCV creates five experiments.

---

## Experiment 1

Training:

```
B

C

D

E
```

Validation:

```
A
```

---

## Experiment 2

Training:

```
A

C

D

E
```

Validation:

```
B
```

---

## Experiment 3

Training:

```
A

B

D

E
```

Validation:

```
C
```

---

## Experiment 4

Training:

```
A

B

C

E
```

Validation:

```
D
```

---

## Experiment 5

Training:

```
A

B

C

D
```

Validation:

```
E
```

---

Finally,

the five prediction errors are averaged.

---

# 8.243 Mathematical Representation

Suppose the dataset contains:

```
N

samples
```

LOOCV performs:

```
N

training experiments
```

The final performance is:

```
Average Error

=

(Error₁ + Error₂ + ... + Errorₙ)

/

N
```

Every sample contributes exactly once to the validation process.

---

# 8.244 Why LOOCV Is Useful for Materials Science

Materials informatics often faces a fundamental limitation:

```
High-quality data is expensive.
```

Unlike image datasets containing millions of examples,

materials datasets may contain:

```
50

100

500
```

samples.

Examples:

- experimentally measured thermal conductivity,
- rare magnetic compounds,
- superconducting materials,
- complex alloy systems.

When data is limited,

removing a large validation set wastes valuable information.

LOOCV maximizes training data usage.

---

# 8.245 Example: DFT Formation Energy Dataset

Imagine a dataset containing:

```
80 materials
```

with accurate DFT formation energies.

A normal train-test split:

```
80%

Training

20%

Testing
```

creates:

```
64 training samples

16 testing samples
```

The model learns from only:

```
64 materials
```

However,

using LOOCV:

Each model trains on:

```
79 materials
```

and validates on:

```
1 material
```

The model receives much more training information.

---

# 8.246 LOOCV and Small Dataset Advantage

The major advantage of LOOCV is:

```
Maximum Training Data
```

For every experiment:

Training samples:

```
N - 1
```

Validation samples:

```
1
```

Therefore,

the model always learns from almost the complete dataset.

---

# 8.247 LOOCV Compared With K-Fold

Consider a dataset containing:

```
100 materials
```

---

## 5-Fold Cross-Validation

Each model trains on:

```
80 materials
```

and validates on:

```
20 materials
```

Number of models:

```
5
```

---

## LOOCV

Each model trains on:

```
99 materials
```

and validates on:

```
1 material
```

Number of models:

```
100
```

---

Comparison:

| Method | Training Samples | Number of Models |
|-|-|-|
| 5-Fold | 80 | 5 |
| 10-Fold | 90 | 10 |
| LOOCV | 99 | 100 |

LOOCV uses more training data but requires much more computation.

---

# 8.248 Bias and Variance in LOOCV

Validation methods involve a trade-off between:

- bias,
- variance.

---

## Bias

A biased estimate systematically overestimates or underestimates performance.

---

## Variance

Variance describes how much the estimate changes with different data splits.

---

LOOCV usually has:

```
Low Bias
```

because almost all data is used for training.

However,

it can have:

```
High Variance
```

because each validation set contains only one sample.

---

# 8.249 Why Single-Sample Validation Can Be Unstable

Imagine predicting formation energy.

One validation material happens to be:

```
An unusual crystal structure
```

The model makes a large error.

Because the validation set contains only one sample,

this error strongly affects that iteration.

Therefore,

LOOCV results may fluctuate more than K-fold validation.

---

# 8.250 Computational Cost of LOOCV

The biggest disadvantage of LOOCV is computational expense.

Suppose:

```
Dataset size = 10,000 materials
```

LOOCV requires:

```
10,000 model trainings
```

For simple models:

```
Linear Regression

↓

Possible
```

For expensive models:

```
Large XGBoost

↓

Very expensive
```

---

# 8.251 LOOCV with XGBoost

Using LOOCV with XGBoost can become computationally demanding.

Example:

Dataset:

```
500 materials
```

LOOCV requires:

```
500 XGBoost models
```

Each model requires:

- tree construction,
- feature evaluation,
- optimization.

For this reason,

materials researchers often choose:

```
5-fold

or

10-fold cross-validation
```

instead.

---

# 8.252 LOOCV Implementation in Scikit-learn

Example:

```python
from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor


model = RandomForestRegressor()


loo = LeaveOneOut()


scores = cross_val_score(

    model,

    X,

    y,

    cv=loo,

    scoring="neg_mean_absolute_error"

)


print(scores)
```

The model is trained once for every sample.

---

# 8.253 Calculating the Final Error

After LOOCV:

```python
import numpy as np


mae = -np.mean(

    scores

)


print(mae)
```

This gives the average prediction error across all left-out samples.

---

# 8.254 LOOCV for Classification Problems

LOOCV can also be used for classification.

Example:

Predict:

```
Stable

or

Unstable
```

For each iteration:

Training:

```
All materials except one
```

Validation:

```
One material
```

The final metrics may include:

- accuracy,
- precision,
- recall,
- F1 score.

---

# 8.255 When Should LOOCV Be Used?

LOOCV is useful when:

### Dataset Is Extremely Small

Example:

```
< 200 samples
```

---

### Every Sample Is Valuable

Example:

Experimental measurements requiring months of work.

---

### Model Training Is Fast

Example:

Linear regression.

---

# 8.256 When Should LOOCV Be Avoided?

LOOCV may not be suitable when:

### Dataset Is Large

Thousands of samples make it unnecessarily expensive.

---

### Model Training Is Expensive

Deep neural networks and large ensemble models become impractical.

---

### Data Contains Strong Chemical Similarity

LOOCV does not automatically solve dataset leakage problems.

---

# 8.257 LOOCV and Materials Data Leakage

A major misconception is:

> "Using LOOCV guarantees reliable performance."

This is not always true.

Suppose a dataset contains:

```
TiO2 polymorph A

TiO2 polymorph B

TiO2 polymorph C
```

During LOOCV:

Training:

```
TiO2 A

TiO2 B
```

Validation:

```
TiO2 C
```

The model has already seen very similar chemistry.

Performance may still be overly optimistic.

Therefore,

validation strategy must consider chemical relationships.

---

# 8.258 Improving LOOCV for Materials Problems

Better strategies include:

### Composition-Based Splitting

Remove all compounds containing certain elements from training.

Example:

Training:

```
Non-titanium compounds
```

Testing:

```
Titanium compounds
```

---

### Structure-Based Splitting

Separate materials according to crystal prototype.

---

### Time-Based Splitting

Train using older discoveries.

Test on newer materials.

---

These methods better represent real materials discovery scenarios.

---

# 8.259 Advantages of LOOCV

LOOCV provides:

### Maximum Data Usage

Almost all samples are used for training.

---

### Low Bias Estimate

Training uses nearly the full dataset.

---

### Suitable for Tiny Scientific Datasets

Very useful for expensive measurements.

---

# 8.260 Limitations of LOOCV

LOOCV has important disadvantages:

### Computationally Expensive

Requires training many models.

---

### High Variance

Single validation samples can strongly affect results.

---

### Does Not Remove Dataset Bias

Poor data quality remains a problem.

---

# 8.261 K-Fold vs LOOCV in Materials Informatics

A practical comparison:

| Feature | K-Fold | LOOCV |
|-|-|-|
| Data requirement | Small to medium | Extremely small |
| Computation | Moderate | High |
| Training usage | High | Maximum |
| Stability | Usually better | Can fluctuate |
| Common usage | Very common | Less common |

---

# 8.262 Practical Recommendation

For most materials machine learning studies:

Use:

```
5-fold

or

10-fold cross-validation
```

with careful splitting strategies.

Use LOOCV when:

- the dataset is extremely small,
- every sample is scientifically valuable,
- computational cost is acceptable.

---

# 8.263 Summary

Leave-One-Out Cross-Validation is an extreme form of cross-validation where each sample is used once as the validation set.

Its main advantage is:

```
Maximum training data utilization.
```

This makes it attractive for small materials datasets where collecting more data is difficult.

However,

LOOCV requires many model trainings and can produce high-variance estimates.

In materials informatics,

the choice between K-fold and LOOCV should depend on:

- dataset size,
- computational resources,
- chemical diversity,
- research objective.

After understanding K-fold and LOOCV,

the next challenge is selecting and tuning models without accidentally using validation information.

This leads to the next important concept:

**Nested Cross-Validation**.

# 8.264 Nested Cross-Validation: Preventing Optimistic Model Selection

In the previous sections, we discussed:

- K-fold cross-validation,
- Leave-One-Out Cross-Validation (LOOCV).

Both methods provide better estimates of model performance compared with a single train-test split.

However,

there is another hidden problem that can make machine learning results overly optimistic.

This problem appears when we use the same validation data for:

- choosing the best model,
- tuning hyperparameters,
- estimating final performance.

This creates a form of information leakage.

The solution is:

# Nested Cross-Validation

---

# 8.265 The Problem of Model Selection Bias

In practical machine learning,

we rarely train only one model.

Instead,

we compare many possibilities.

For example:

Different algorithms:

```
Linear Regression

Random Forest

XGBoost

Neural Network
```

Different hyperparameters:

```
Learning rate

Tree depth

Number of estimators

Regularization strength
```

The researcher chooses the model that performs best.

The question is:

> How do we know that the selected model will perform equally well on completely new data?

---

# 8.266 A Simple Example of Selection Bias

Imagine testing 100 different XGBoost models.

Each model is evaluated on the same validation set.

The results:

```
Model 1

R² = 0.82
```

```
Model 2

R² = 0.85
```

```
Model 3

R² = 0.91
```

Naturally,

we select Model 3.

However,

Model 3 may not actually be the best model.

It may simply have matched accidental patterns in that validation dataset.

The validation data has influenced model selection.

Therefore,

the performance estimate becomes too optimistic.

---

# 8.267 The Idea Behind Nested Cross-Validation

Nested cross-validation separates two tasks:

## Model Optimization

Finding the best model.

## Model Evaluation

Estimating how well that model performs on unseen data.

These two tasks must use different data.

Nested cross-validation creates:

```
Outer Loop

+

Inner Loop
```

---

# 8.268 Structure of Nested Cross-Validation

Nested cross-validation contains two levels.

```
Complete Dataset

↓

Outer Cross-Validation

↓

Training Portion + Test Portion

↓

Inner Cross-Validation

↓

Hyperparameter Optimization

↓

Final Evaluation
```

The inner loop chooses the model.

The outer loop evaluates the chosen model.

---

# 8.269 The Outer Loop

The outer loop estimates the final performance.

Suppose we perform:

```
5-fold outer cross-validation
```

The dataset is divided into:

```
Fold 1

Fold 2

Fold 3

Fold 4

Fold 5
```

Each fold becomes a final test set once.

---

Example:

Iteration 1:

Training:

```
Fold 2

Fold 3

Fold 4

Fold 5
```

Testing:

```
Fold 1
```

---

Iteration 2:

Training:

```
Fold 1

Fold 3

Fold 4

Fold 5
```

Testing:

```
Fold 2
```

---

The outer loop measures true generalization.

---

# 8.270 The Inner Loop

Inside each outer training set,

we perform another cross-validation.

The purpose:

```
Find Best Model Configuration
```

For example,

XGBoost hyperparameters:

```
Maximum depth:

3, 5, 7


Learning rate:

0.01, 0.05, 0.1


Number of trees:

100, 300, 500
```

The inner loop tests these combinations.

The best configuration is selected.

---

# 8.271 Complete Nested Cross-Validation Workflow

Consider predicting:

```
Formation Energy
```

using XGBoost.

The workflow becomes:

---

## Step 1

Split the dataset using outer 5-fold cross-validation.

```
Dataset

↓

5 outer folds
```

---

## Step 2

Take one outer training set.

Example:

```
80% of materials
```

---

## Step 3

Perform inner cross-validation.

Example:

```
5-fold inner validation
```

to optimize:

- learning rate,
- tree depth,
- regularization.

---

## Step 4

Train the optimized model.

---

## Step 5

Evaluate on the outer test fold.

This test fold has never influenced model selection.

---

## Step 6

Repeat for all outer folds.

---

## Step 7

Average the outer test scores.

The result is a realistic estimate of model performance.

---

# 8.272 Why Nested Validation Is More Reliable

The key advantage:

The final evaluation data remains completely untouched.

The model has never seen:

- the data,
- the features,
- the outcomes,

during optimization.

Therefore,

the performance estimate is much closer to real-world behavior.

---

# 8.273 Nested Cross-Validation in Materials Science

Nested validation is especially valuable when:

- datasets are small,
- many models are compared,
- hyperparameter tuning is extensive.

This is common in materials informatics.

Example:

Predicting:

```
Elastic modulus
```

from:

- elemental properties,
- crystal descriptors,
- pymatgen features.

A researcher may test:

```
Random Forest

XGBoost

Gradient Boosting

Neural Network
```

with hundreds of parameter combinations.

Without nested validation,

the final reported score may be artificially high.

---

# 8.274 Example: XGBoost Materials Model

Suppose we have:

```
2000 crystal structures
```

Target:

```
Band Gap
```

Features:

```
Composition descriptors

Crystal structure descriptors

Pymatgen features
```

The researcher performs:

Random search:

```
500 XGBoost parameter combinations
```

The best model gives:

```
RMSE = 0.05 eV
```

on the validation set.

But because 500 models were tested,

some may perform unusually well by chance.

---

Nested validation may reveal:

```
Outer CV RMSE

=

0.14 eV
```

The second value is more realistic.

---

# 8.275 Nested Cross-Validation and Hyperparameter Tuning

Hyperparameter tuning is one of the main reasons nested validation is needed.

Consider:

```
Learning Rate
```

A researcher tries:

```
0.01

0.05

0.10

0.20
```

The best value is selected using validation performance.

However,

that validation set has now influenced the model.

It is no longer an unbiased evaluator.

Nested validation separates:

```
Choosing Parameters

from

Testing Performance
```

---

# 8.276 Nested Cross-Validation with Classification

The same idea applies to classification.

Example:

Predict:

```
Stable

vs

Unstable Materials
```

Metrics:

- F1 score,
- ROC-AUC,
- precision,
- recall.

The inner loop optimizes the classifier.

The outer loop estimates real performance.

---

# 8.277 Implementation Concept in Scikit-learn

Scikit-learn provides tools for nested validation.

Common components:

```
GridSearchCV

RandomizedSearchCV

cross_val_score
```

Conceptually:

```python
Inner Loop:

GridSearchCV

↓

Find Best Parameters


Outer Loop:

cross_val_score

↓

Evaluate Model
```

---

# 8.278 Example Workflow

```python
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score


model = XGBRegressor()


parameter_grid = {

    "max_depth":

    [3, 5, 7],

    "learning_rate":

    [0.01, 0.05, 0.1]

}


search = GridSearchCV(

    model,

    parameter_grid,

    cv=5

)


scores = cross_val_score(

    search,

    X,

    y,

    cv=5

)
```

Here:

Inner:

```
GridSearchCV
```

selects parameters.

Outer:

```
cross_val_score
```

evaluates performance.

---

# 8.279 Computational Cost of Nested Validation

The major disadvantage is computation.

Suppose:

Outer loop:

```
5 folds
```

Inner loop:

```
5 folds
```

Parameter combinations:

```
100
```

Total model trainings:

```
5 × 5 × 100

=

2500 models
```

For simple models:

```
Acceptable
```

For complex models:

```
Expensive
```

---

# 8.280 When Should Nested Cross-Validation Be Used?

Nested validation is recommended when:

### Comparing Multiple Models

Example:

```
XGBoost

Random Forest

Neural Network
```

---

### Performing Extensive Hyperparameter Search

Example:

Hundreds of parameter combinations.

---

### Publishing Scientific Results

Especially when reporting state-of-the-art performance.

---

### Dataset Is Small

Where overestimating performance is dangerous.

---

# 8.281 When Is Nested Validation Unnecessary?

For simple exploratory work:

```
Single model

Few parameters

Large dataset
```

standard validation may be sufficient.

---

# 8.282 Nested Validation and Research Publication

In materials machine learning papers,

reviewers often question:

> "Was the reported performance obtained after unbiased evaluation?"

A model with:

```
Single train-test split
```

may be considered unreliable.

A model with:

```
Nested cross-validation
```

provides stronger evidence.

---

# 8.283 Summary of Validation Methods

| Method | Purpose | Suitable For |
|-|-|-|
| Train-Test Split | Quick evaluation | Large datasets |
| K-Fold CV | Reliable estimate | Most ML studies |
| LOOCV | Maximum data usage | Very small datasets |
| Nested CV | Unbiased model selection | Research-grade studies |

---

# 8.284 Validation Strategy for Materials Informatics

A practical workflow:

```
Raw Materials Dataset

↓

Feature Engineering

(Pymatgen descriptors)

↓

Train/Validation Strategy

↓

Hyperparameter Optimization

↓

Nested Cross-Validation

↓

Final Test Evaluation

↓

Report Metrics
```

The validation strategy should match:

- dataset size,
- computational resources,
- scientific objective.

---

# 8.285 Final Perspective

Machine learning performance is not only about obtaining a high score.

A model showing:

```
R² = 0.95

or

ROC-AUC = 0.98
```

does not automatically mean it has discovered useful physical relationships.

Reliable machine learning requires:

- correct evaluation metrics,
- careful validation,
- prevention of data leakage,
- realistic testing conditions.

This is especially important in materials science,

where datasets are small and chemical relationships are complex.

In the next section,

we will discuss one of the most important issues in materials machine learning:

# Why Small DFT Datasets Require Careful Validation

# 8.286 Why Small DFT Datasets Require Careful Validation

In the previous sections, we discussed several validation strategies:

- Train-test split.
- K-fold cross-validation.
- Leave-One-Out Cross-Validation.
- Nested cross-validation.

These methods are general machine learning techniques.

However,

materials informatics has a unique challenge:

> **Most high-quality materials datasets are small compared with datasets used in other machine learning fields.**

This makes validation much more difficult.

A model that appears highly accurate may simply be exploiting hidden biases in the dataset.

Understanding this problem is essential for building reliable materials machine learning models.

---

# 8.287 The Nature of DFT-Based Materials Datasets

Density Functional Theory (DFT) is one of the most important sources of computational materials data.

DFT can calculate:

- formation energy,
- band gap,
- elastic properties,
- magnetic moments,
- phonon properties,
- adsorption energies.

A typical workflow is:

```
Crystal Structure

↓

DFT Calculation

↓

Material Property

↓

Machine Learning Dataset
```

The output of DFT becomes the target value for machine learning.

---

# 8.288 Why DFT Data Is Expensive

Unlike collecting images or text data,

generating materials data requires significant computational effort.

A single DFT calculation may require:

- electronic structure calculations,
- geometry optimization,
- convergence testing,
- high-performance computing resources.

For complex systems:

```
One calculation

↓

Hours

or

Days
```

may be required.

Therefore,

creating millions of high-quality labeled materials is difficult.

---

# 8.289 Typical Dataset Sizes in Materials Machine Learning

Compared with other machine learning fields:

| Field | Typical Dataset Size |
|-|-|
| Image recognition | Millions of images |
| Natural language processing | Billions of words |
| Materials informatics | Hundreds to hundreds of thousands |

Many materials studies operate with:

```
100

500

5000
```

samples.

This creates a difficult learning environment.

---

# 8.290 The Small Data Problem

Machine learning models require examples to learn patterns.

A small dataset provides limited information.

Imagine predicting:

```
Formation Energy
```

using only:

```
100 materials
```

The model must learn relationships between:

- elemental composition,
- atomic arrangement,
- bonding,
- crystal structure,
- electronic effects.

This is a huge chemical space.

The available examples represent only a tiny fraction of possible materials.

---

# 8.291 The Chemical Space Problem

The possible number of materials is enormous.

Even simple combinations of elements create thousands of possibilities.

For example:

```
A

B

C

D
```

elements can combine into many:

- compositions,
- structures,
- phases.

The known materials database represents only a small sample of the possible design space.

Therefore:

```
Training Data

≠

Entire Materials Universe
```

---

# 8.292 Why Random Splitting Can Be Dangerous

A common machine learning workflow is:

```
Randomly divide data

↓

Training set

+

Test set
```

This assumes every sample is independent.

However,

materials are often strongly related.

Example:

Training set:

```
TiO2 rutile

TiO2 anatase
```

Test set:

```
TiO2 brookite
```

The model has already seen very similar chemistry.

The test performance may look excellent.

But the model has not truly learned how to predict unknown materials.

---

# 8.293 Data Leakage in Materials Informatics

Data leakage occurs when information from the test set accidentally influences training.

This can happen in several ways.

---

## Structural Similarity Leakage

Very similar crystal structures appear in both:

```
Training

and

Testing
```

---

## Composition Leakage

Materials containing the same elements appear in both groups.

Example:

Training:

```
LiFePO4
```

Testing:

```
Li2FeP2O7
```

The model benefits from seeing related chemistry.

---

## Duplicate Data

The same material may appear multiple times:

- different database entries,
- different calculation conditions,
- different naming conventions.

---

# 8.294 Example: Overestimated Band Gap Prediction

Suppose a machine learning model predicts:

```
Band Gap
```

Dataset:

```
10,000 materials
```

Random split result:

```
RMSE = 0.08 eV
```

This looks excellent.

However,

after using a more realistic composition-based split:

```
RMSE = 0.35 eV
```

Why did performance decrease?

Because the original split allowed the model to use chemical similarity rather than learning general principles.

---

# 8.295 Distribution Shift Problem

Another major challenge is:

```
Training Data Distribution

≠

Real Application Distribution
```

This is called:

```
Distribution Shift
```

---

Example:

Training dataset:

Mostly:

```
Common inorganic oxides
```

Application:

Predict:

```
Novel battery compounds
```

The model has never seen similar chemistry.

Performance may collapse.

---

# 8.296 Types of Distribution Shift in Materials

Materials problems often involve:

---

## Composition Shift

Training:

```
Fe-based materials
```

Testing:

```
Rare-earth materials
```

---

## Structure Shift

Training:

```
Cubic structures
```

Testing:

```
Layered structures
```

---

## Property Range Shift

Training:

```
Band gaps: 0–3 eV
```

Testing:

```
Band gaps: 5–8 eV
```

---

A reliable validation strategy should test whether the model can handle these shifts.

---

# 8.297 Validation Strategies for Materials Data

Standard random validation is often insufficient.

Researchers use more realistic approaches.

---

# 8.298 Composition-Based Splitting

In composition-based splitting,

materials containing certain elements are completely removed from training.

Example:

Training:

```
Materials without lithium
```

Testing:

```
Lithium-containing materials
```

This tests whether the model can generalize to new chemical systems.

---

# 8.299 Structure-Based Splitting

Crystal structures are separated.

Example:

Training:

```
Cubic structures
```

Testing:

```
Hexagonal structures
```

This evaluates structural generalization.

---

# 8.300 Time-Based Splitting

Materials are separated according to discovery date.

Example:

Training:

```
Materials discovered before 2020
```

Testing:

```
Materials discovered after 2020
```

This simulates real scientific discovery.

---

# 8.301 Why Validation Strategy Changes Model Ranking

Different validation methods can produce different winners.

Example:

Random split:

```
Model A

RMSE = 0.10
```

```
Model B

RMSE = 0.15
```

Model A appears better.

---

Composition split:

```
Model A

RMSE = 0.40
```

```
Model B

RMSE = 0.25
```

Now Model B is better.

The original conclusion was caused by the validation method.

---

# 8.302 Small DFT Data and Model Complexity

Small datasets create another problem:

complex models can easily overfit.

Examples:

- deep neural networks,
- very large ensemble models,
- high-dimensional feature sets.

A model may memorize:

```
Training Examples
```

instead of learning:

```
Physical Relationships
```

---

# 8.303 Feature Selection and Validation

Materials models often use many descriptors:

Examples:

- atomic radius,
- electronegativity,
- atomic mass,
- valence electrons,
- crystal volume,
- density.

With limited data,

too many features can create overfitting.

Example:

Dataset:

```
200 materials
```

Features:

```
500 descriptors
```

The model may find accidental correlations.

Validation helps identify whether these features truly improve prediction.

---

# 8.304 The Role of Physical Knowledge

One advantage of materials informatics is that physical knowledge can guide machine learning.

Instead of using arbitrary features,

we can include meaningful descriptors:

Examples:

- bond length,
- coordination number,
- electronegativity difference,
- ionic radius mismatch.

Better features reduce the amount of data required.

---

# 8.305 Validation Example: Formation Energy Prediction

Suppose we build an XGBoost model.

Features:

Generated using:

```
pymatgen

+

chemical descriptors
```

Dataset:

```
3000 compounds
```

Evaluation approaches:

---

Random split:

```
RMSE = 0.06 eV/atom
```

---

5-fold cross-validation:

```
RMSE = 0.10 eV/atom
```

---

Composition-based split:

```
RMSE = 0.22 eV/atom
```

The final value is usually the most realistic for discovering new materials.

---

# 8.306 Recommended Validation Workflow for Materials ML

A robust workflow:

```
Materials Database

↓

Remove duplicates

↓

Analyze chemical distribution

↓

Generate descriptors

↓

Choose realistic split strategy

↓

Cross-validation

↓

Hyperparameter optimization

↓

Final unseen test

↓

Report multiple metrics
```

---

# 8.307 Reporting Validation in Research Papers

A strong materials machine learning paper should report:

## Dataset Information

- Number of materials.
- Source of data.
- Calculation method.

---

## Validation Method

Example:

```
10-fold cross-validation
```

or

```
Nested cross-validation
```

---

## Performance Metrics

Regression:

- MAE.
- RMSE.
- R².

Classification:

- Accuracy.
- Precision.
- Recall.
- F1.
- ROC-AUC.

---

## Limitations

Researchers should discuss:

- dataset size,
- chemical coverage,
- possible bias,
- generalization limits.

---

# 8.308 Final Perspective

Small DFT datasets create one of the biggest challenges in materials machine learning.

A model can achieve excellent performance numbers while failing to generalize to truly new materials.

Reliable prediction requires:

- careful validation,
- realistic data splitting,
- prevention of leakage,
- understanding chemical similarity.

In materials science,

validation is not simply a statistical procedure.

It determines whether a machine learning model has learned meaningful physical relationships or merely memorized known compounds.

---

# 8.309 Chapter 8 Summary: Model Evaluation and Validation

In this chapter, we learned:

## Regression Evaluation

- MAE measures average prediction error.
- MSE penalizes large errors.
- RMSE provides error in original units.
- R² measures explained variance.

---

## Classification Evaluation

- Accuracy measures overall correctness.
- Precision measures prediction reliability.
- Recall measures discovery ability.
- F1 balances precision and recall.
- ROC-AUC evaluates class separation across thresholds.

---

## Validation Methods

- K-fold cross-validation improves reliability.
- LOOCV maximizes training usage for tiny datasets.
- Nested validation prevents model selection bias.

---

## Materials Perspective

Small DFT datasets require special attention because:

- data is expensive,
- chemical space is enormous,
- materials are correlated,
- random splits can be misleading.

A successful materials machine learning model is not the one with the highest score.

It is the one that maintains reliable performance when predicting materials it has never seen before.

---

**End of Chapter 8**