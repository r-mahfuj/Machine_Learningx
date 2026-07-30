# Chapter 6
# Gradient Boosting

## 6.1 Why Was Gradient Boosting Developed?

At the end of the previous chapter,

we learned that

Random Forest

greatly improves the predictive performance of a single decision tree.

Instead of relying on one complex tree,

Random Forest trains

many independent trees

and combines their predictions.

This simple idea dramatically reduces variance and improves generalization.

For many years,

Random Forest became one of the most successful machine learning algorithms used in scientific research,

including materials informatics.

However,

despite its impressive performance,

researchers observed an important limitation.

Every tree inside a Random Forest is trained

**independently.**

Each tree receives a different bootstrap sample,

makes its own predictions,

and never attempts to learn from the mistakes made by other trees.

Consider an analogy.

Suppose ten students independently solve the same mathematics problem.

Each student works alone.

Afterward,

their answers are averaged.

This resembles the philosophy of Random Forest.

Now imagine a different scenario.

The first student solves the problem.

The second student carefully studies the mistakes made by the first student and attempts to correct them.

The third student studies the remaining mistakes and improves the solution even further.

The fourth student continues the correction process.

Instead of working independently,

every student builds upon the previous solution.

This second strategy is fundamentally different.

Rather than averaging many independent models,

it creates a sequence of progressively improving models.

This idea became the foundation of

# Boosting.

---

## 6.2 The Main Philosophy of Boosting

The central philosophy of boosting can be summarized by one sentence.

> **Instead of building many independent strong models, build many weak models that gradually correct each other's mistakes.**

The phrase

**weak model**

does not mean

"bad model."

Instead,

it refers to a model that performs only slightly better than random guessing.

For example,

suppose a binary classification problem has

50%

random guessing accuracy.

A weak learner might achieve

60%

accuracy.

Individually,

this model is not impressive.

However,

boosting demonstrates something remarkable.

If many weak learners are combined intelligently,

the final model can become extremely powerful.

Rather than expecting one perfect model,

boosting allows many simple models to cooperate.

Each model contributes only a small improvement,

but together,

these improvements accumulate into a highly accurate predictor.

---

## 6.3 Weak Learners Versus Strong Learners

Before studying the boosting algorithm,

it is important to understand the distinction between

weak learners

and

strong learners.

Suppose we are predicting whether a crystalline material is mechanically stable.

A very small decision tree,

containing only one split,

might classify materials as follows.

```text
Density > 5.5 g/cm³ ?

        /           \

      Stable     Unstable
```

This model is obviously too simple.

Many materials violate this rule.

Nevertheless,

it may still perform slightly better than random guessing.

Such a model is called

a

**weak learner.**

A strong learner,

on the other hand,

captures much more of the underlying relationship.

Random Forest,

XGBoost,

and deep neural networks

are examples of strong learners.

Interestingly,

boosting does not begin with a strong learner.

It intentionally begins with a weak learner

and repeatedly improves it.

---

## 6.4 Why Not Simply Build One Very Large Tree?

At first,

boosting may appear unnecessary.

Why not simply construct

one extremely deep decision tree?

Unfortunately,

deep trees introduce serious problems.

As tree depth increases,

the model becomes increasingly flexible.

Eventually,

it memorizes the training data.

The result is

overfitting.

Random Forest addressed this issue

by averaging many independent trees.

Boosting approaches the problem differently.

Instead of constructing one enormous tree,

it builds

many very small trees,

each responsible for correcting only a portion of the remaining error.

Consequently,

the final model becomes highly expressive

without relying on a single over-complex decision tree.

---

## 6.5 The Sequential Learning Process

The defining characteristic of boosting is

**sequential learning.**

Unlike Random Forest,

where every tree is trained independently,

boosting trains one tree at a time.

The process follows a simple pattern.

```text
Train Tree 1

↓

Measure Errors

↓

Train Tree 2

↓

Correct Previous Errors

↓

Measure Remaining Errors

↓

Train Tree 3

↓

Correct Remaining Errors

↓

Continue Until Desired Performance
```

Every new tree depends upon

all previous trees.

This sequential dependency is the defining characteristic of boosting.

Because each tree attempts to improve the existing model,

boosting gradually reduces prediction error with every iteration.

---

## 6.6 A Materials Informatics Analogy

Suppose we wish to predict

formation energy

using descriptors extracted from Pymatgen.

Examples include

- density,
- average electronegativity,
- average atomic radius,
- unit-cell volume,
- packing fraction.

Imagine that the first decision tree predicts formation energy for

2,000

materials.

Most predictions are reasonably accurate,

but several materials exhibit unusually large prediction errors.

Instead of ignoring these difficult samples,

boosting asks

> **Which materials were predicted poorly?**

The next tree concentrates primarily on those difficult cases.

After incorporating the second tree,

prediction improves.

Some difficult materials remain.

A third tree now focuses on those remaining errors.

The process continues.

Each tree contributes only a modest improvement,

but together,

they gradually construct an increasingly accurate predictive model.

---

## 6.7 The Difference Between Bagging and Boosting

Random Forest is based upon

Bagging

(Bootstrap Aggregating).

Gradient Boosting belongs to a completely different family called

Boosting.

Although both methods combine many decision trees,

their learning philosophies differ fundamentally.

| Bagging (Random Forest) | Boosting (Gradient Boosting) |
|--------------------------|------------------------------|
| Trees are trained independently. | Trees are trained sequentially. |
| Every tree receives a bootstrap sample. | Every tree learns from previous errors. |
| Predictions are averaged. | Predictions are added together. |
| Main objective is variance reduction. | Main objective is bias reduction through iterative improvement. |
| Trees can be trained in parallel. | Trees must be trained one after another. |

These differences may appear small,

but they lead to dramatically different algorithms.

Gradient Boosting is not merely

"a better Random Forest."

It is based upon an entirely different learning strategy.

---

## 6.8 Why Gradient Boosting Became So Successful

Gradient Boosting quickly became one of the most influential machine learning algorithms because it combines several desirable properties.

It can

- model highly nonlinear relationships,
- capture complex feature interactions,
- achieve excellent predictive accuracy,
- work with relatively small datasets,
- naturally handle mixed numerical features,
- serve as the theoretical foundation for modern algorithms such as XGBoost, LightGBM, and CatBoost.

In materials informatics,

Gradient Boosting has been applied successfully to predict

- formation energy,
- band gap,
- elastic modulus,
- thermal conductivity,
- battery capacity,
- catalytic activity,
- superconducting transition temperature,

and many other materials properties.

Its ability to learn complex nonlinear relationships makes it especially valuable when descriptors extracted from Pymatgen exhibit intricate interactions that cannot be captured by linear models.

---

## 6.9 The Road Ahead

Although the basic philosophy of boosting appears simple,

the mathematical ideas underlying Gradient Boosting are considerably more sophisticated.

Unlike Random Forest,

Gradient Boosting introduces several entirely new concepts,

including

- additive models,
- residual learning,
- loss functions,
- optimization,
- gradients,
- functional gradient descent.

These ideas ultimately lead to

XGBoost,

one of the most powerful machine learning algorithms used in modern scientific research.

Before studying gradients and optimization,

however,

we must first understand the simplest and most intuitive idea underlying Gradient Boosting.

That idea is

**learning from residuals.**

Residual learning is the bridge connecting ordinary decision trees to Gradient Boosting,

and it forms the foundation upon which the remainder of this chapter will be built.

## 6.10 From Independent Trees to Residual Learning

In the previous section,

we learned that Gradient Boosting trains decision trees **sequentially** rather than independently.

This naturally raises an important question.

If every new tree is supposed to improve the previous one,

**what exactly does the new tree learn?**

Does it learn the original target values again?

Does it build an entirely new prediction model?

Or does it focus only on the mistakes made by previous trees?

The answer is the third option.

A Gradient Boosting model does **not** repeatedly learn the original target variable.

Instead,

every new tree learns the **errors** made by the existing model.

These errors are called

# Residuals.

Residual learning is the fundamental idea that distinguishes Gradient Boosting from almost every other tree-based algorithm.

Before introducing gradients,

loss functions,

or optimization,

we must first understand residuals thoroughly.

Everything else in Gradient Boosting builds upon this concept.

---

## 6.11 What Is a Residual?

Suppose we wish to predict

formation energy

for crystalline materials.

For one material,

the experimentally measured formation energy is

```text
True Formation Energy

-3.20 eV/atom
```

Suppose the first decision tree predicts

```text
Predicted Formation Energy

-2.90 eV/atom
```

The prediction is close,

but not perfect.

The difference between the true value and the predicted value is

```text
Residual

=

True Value

−

Predicted Value
```

Therefore,

```text
Residual

=

-3.20

−

(-2.90)

=

-0.30 eV/atom
```

The model underestimated the magnitude of the formation energy.

That remaining

−0.30 eV/atom

is exactly what the next tree will attempt to learn.

---

## 6.12 Residuals Represent Unexplained Information

Residuals can be interpreted as

the information that the current model has **not yet learned**.

Suppose we predict the formation energies of five materials.

| Material | True Value | Prediction | Residual |
|----------|-----------:|-----------:|----------:|
| A | -3.20 | -2.90 | -0.30 |
| B | -2.70 | -2.85 | 0.15 |
| C | -1.80 | -1.70 | -0.10 |
| D | -4.10 | -3.95 | -0.15 |
| E | -2.30 | -2.32 | 0.02 |

Notice something interesting.

The predictions themselves are already reasonably accurate.

The remaining residuals are relatively small.

Instead of rebuilding another complete model,

Gradient Boosting asks

> **Can we build a small decision tree that predicts only these remaining errors?**

If we succeed,

we can simply add those corrections to the existing predictions.

---

## 6.13 Correcting Predictions Instead of Replacing Them

This is perhaps the most important conceptual shift in Gradient Boosting.

Random Forest produces many complete predictions and averages them.

Gradient Boosting behaves differently.

Suppose the first tree predicts

```text
-2.90 eV/atom
```

The second tree predicts only the residual

```text
-0.25 eV/atom
```

The updated prediction becomes

```text
-2.90

+

(-0.25)

=

-3.15 eV/atom
```

Notice what happened.

The second tree did **not** replace the first prediction.

It merely corrected it.

Suppose another small residual still remains.

A third tree predicts

```text
-0.04 eV/atom
```

The prediction becomes

```text
-3.15

+

(-0.04)

=

-3.19 eV/atom
```

A fourth tree predicts

```text
-0.01 eV/atom
```

The final prediction becomes

```text
-3.20 eV/atom
```

Every tree contributes only a small correction.

Together,

they gradually converge toward the true value.

---

## 6.14 An Everyday Analogy

Imagine drawing a portrait.

Your first sketch captures the general face shape,

but many details are missing.

You do not erase the drawing and begin again.

Instead,

you improve it step by step.

First,

you adjust the eyes.

Next,

you refine the nose.

Then,

you improve the hair.

Finally,

you add shadows and fine details.

Each revision corrects the mistakes of the previous version.

By the end,

the portrait closely resembles the original subject.

Gradient Boosting works in exactly the same way.

The first decision tree produces only a rough approximation.

Every subsequent tree refines that approximation by correcting the remaining errors.

---

## 6.15 Why Small Corrections Work Better Than Large Changes

One might wonder why Gradient Boosting prefers many small corrections instead of building a completely new prediction each time.

The reason lies in optimization.

Suppose a model already predicts formation energy reasonably well.

Most predictions differ from the true values by only a small amount.

Learning these small residuals is often much easier than learning the entire physical relationship from scratch.

Each correction tree therefore solves a much simpler problem.

Instead of discovering the complete mapping

```text
Descriptors

↓

Formation Energy
```

the correction tree learns

```text
Descriptors

↓

Remaining Error
```

Because the remaining error is usually much smaller and less complicated,

small decision trees can model it effectively.

---

## 6.16 The Additive Nature of Gradient Boosting

The prediction produced by Gradient Boosting is not generated by a single tree.

Instead,

it is the sum of many successive trees.

Mathematically,

after the first tree,

the prediction is

\[
\hat{y}=F_1(x).
\]

After the second tree,

the prediction becomes

\[
\hat{y}=F_1(x)+F_2(x).
\]

After three trees,

\[
\hat{y}=F_1(x)+F_2(x)+F_3(x).
\]

More generally,

after

\(M\)

trees,

the prediction is

\[
\hat{y}=\sum_{m=1}^{M}F_m(x).
\]

This is called an

# Additive Model

because every new tree is **added** to all previous trees.

Notice the contrast with Random Forest.

Random Forest averages predictions.

Gradient Boosting adds predictions.

Although this appears to be a minor mathematical difference,

it completely changes how the algorithm learns.

---

## 6.17 Visualizing Residual Learning

The learning process can be visualized as follows.

```text
Training Data

↓

Tree 1

↓

Prediction

↓

Residuals

↓

Tree 2 Learns Residuals

↓

Updated Prediction

↓

New Residuals

↓

Tree 3 Learns Remaining Residuals

↓

Updated Prediction

↓

Continue Until Errors Become Very Small
```

Each iteration reduces the unexplained information remaining in the dataset.

Eventually,

the residuals become sufficiently small that additional trees provide little improvement.

At that point,

training stops.

---

## 6.18 Residual Learning in Materials Informatics

Suppose we train a Gradient Boosting model to predict

bulk modulus

using descriptors extracted with Pymatgen,

including

- density,
- average atomic radius,
- average electronegativity,
- packing fraction,
- unit-cell volume.

The first tree may successfully capture the broad relationship between density and bulk modulus.

However,

it may consistently underestimate materials possessing unusually strong covalent bonding.

Those systematic prediction errors appear in the residuals.

The second tree may discover that

average electronegativity

explains much of these remaining errors.

After correcting those,

small residuals still remain.

A third tree may recognize that

packing fraction

accounts for part of the remaining discrepancy.

Tree after tree,

the model gradually incorporates increasingly subtle physical relationships until prediction accuracy reaches a satisfactory level.

This ability to sequentially learn increasingly complex corrections is one of the primary reasons Gradient Boosting frequently achieves higher predictive accuracy than Random Forest on structured scientific datasets.

In the next section,

we will investigate an important question.

If each new tree learns the residuals,

**how does the algorithm know what residuals should be learned for every iteration?**

Answering that question requires introducing the concept of the **loss function**—the mathematical quantity that Gradient Boosting attempts to minimize throughout training.

## 6.19 Why Do We Need a Loss Function?

In the previous section,

we learned that every new tree in Gradient Boosting attempts to learn the residuals left behind by the previous trees.

This naturally leads to another question.

How does the algorithm determine

whether its predictions are becoming better or worse?

Suppose two different models predict the formation energy of the same material.

Model A predicts

```text
-3.18 eV/atom
```

Model B predicts

```text
-2.60 eV/atom
```

The true formation energy is

```text
-3.20 eV/atom
```

Clearly,

Model A is much better.

But how can a computer determine this automatically?

Computers cannot simply "look" at two predictions and decide which is superior.

Instead,

they require a numerical quantity that measures prediction quality.

This numerical quantity is called the

# Loss Function.

The entire objective of Gradient Boosting is

not merely to reduce residuals,

but more fundamentally,

to minimize a carefully chosen loss function.

---

## 6.20 What Is a Loss Function?

A loss function is a mathematical equation that measures

how different the model's predictions are from the true target values.

In simple terms,

it answers one question.

> **How bad are the current predictions?**

If predictions are very accurate,

the loss is small.

If predictions are poor,

the loss is large.

Therefore,

machine learning can be viewed as an optimization problem.

Instead of asking

> "Can we build a model?"

we ask

> **"Can we build a model that minimizes the loss function?"**

This idea appears throughout modern machine learning,

including

- Linear Regression,
- Logistic Regression,
- Neural Networks,
- Gradient Boosting,
- XGBoost,
- Deep Learning.

Understanding loss functions therefore provides the foundation for many advanced algorithms.

---

## 6.21 Revisiting Linear Regression

Earlier in this book,

we studied Linear Regression.

Its objective was

not

to maximize prediction accuracy directly.

Instead,

Linear Regression minimized

the

Sum of Squared Errors (SSE).

Recall that

\[
SSE=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2.
\]

Here,

- \(y_i\) represents the true value,
- \(\hat{y}_i\) represents the predicted value.

Notice something important.

The quantity

\[
y_i-\hat{y}_i
\]

is simply

the residual.

Therefore,

Linear Regression already minimized a loss function.

Gradient Boosting extends this same philosophy.

The difference is that

instead of solving one optimization problem analytically,

Gradient Boosting reduces the loss gradually,

one decision tree at a time.

---

## 6.22 Why Are Errors Squared?

One may wonder why the errors are squared rather than simply added together.

Suppose two prediction errors are

```text
+5

and

-5
```

If we simply add them,

```text
5

+

(-5)

=

0
```

The total error appears to be zero,

even though both predictions are incorrect.

Squaring solves this problem.

\[
(+5)^2=25
\]

and

\[
(-5)^2=25.
\]

Now,

both contribute positively to the total loss.

Furthermore,

larger errors become much more heavily penalized.

For example,

an error of

10

produces

\[
10^2=100,
\]

whereas an error of

2

produces only

\[
2^2=4.
\]

Thus,

the loss function encourages the model to eliminate large mistakes first.

---

## 6.23 Example: Calculating Loss

Suppose we predict the formation energy of four materials.

| Material | True Value | Prediction |
|----------|-----------:|-----------:|
| A | -3.20 | -3.00 |
| B | -2.70 | -2.60 |
| C | -1.80 | -2.00 |
| D | -4.10 | -3.80 |

The residuals are

| Material | Residual |
|----------|---------:|
| A | -0.20 |
| B | -0.10 |
| C | 0.20 |
| D | -0.30 |

Squaring each residual gives

| Material | Squared Error |
|----------|--------------:|
| A | 0.040 |
| B | 0.010 |
| C | 0.040 |
| D | 0.090 |

Therefore,

the total squared error becomes

\[
SSE
=
0.04
+
0.01
+
0.04
+
0.09
=
0.18.
\]

The value

0.18

summarizes the overall prediction quality of the model.

A better model would produce

a smaller loss.

---

## 6.24 Loss as a Landscape

An intuitive way to understand optimization is to imagine the loss function as a landscape.

Suppose every possible machine learning model corresponds to one location on a mountainous terrain.

High mountains represent

large prediction errors.

Deep valleys represent

small prediction errors.

```text
Large Loss

        /\

       /  \

      /    \

_____/      \______

               \

                \

                 \______

                 Minimum Loss
```

Training a machine learning model is equivalent to moving downhill until the lowest possible point is reached.

The lowest point represents

the model with the smallest prediction error.

Gradient Boosting is essentially a sophisticated method for descending this mathematical landscape.

---

## 6.25 Different Problems Require Different Loss Functions

Not every machine learning problem uses the same loss function.

The choice depends upon the prediction task.

For regression,

common loss functions include

- Mean Squared Error,
- Mean Absolute Error,
- Huber Loss.

For binary classification,

common choices include

- Binary Cross-Entropy,
- Logistic Loss.

For multi-class classification,

algorithms often use

- Categorical Cross-Entropy.

This flexibility makes Gradient Boosting extremely powerful.

Unlike ordinary least-squares regression,

it is not restricted to one particular objective.

Instead,

the algorithm can optimize many different loss functions depending upon the scientific problem.

---

## 6.26 Loss Functions in Materials Informatics

Suppose we are predicting

formation energy.

Formation energy is a continuous numerical quantity.

Therefore,

Mean Squared Error is a natural loss function.

Now consider another problem.

Suppose we wish to classify materials as

```text
Metal

or

Semiconductor.
```

This is no longer a regression problem.

Instead,

it is a binary classification problem.

The loss function must now measure

classification error

rather than numerical prediction error.

Although the machine learning model changes,

the underlying optimization philosophy remains identical.

Every iteration attempts to reduce the chosen loss function.

This unified perspective explains why Gradient Boosting can solve

regression,

binary classification,

and multiclass classification

using essentially the same algorithmic framework.

---

## 6.27 Residuals Depend on the Loss Function

Earlier,

we stated that every new tree learns the residuals.

While this statement is true for ordinary regression,

it is actually a simplified description.

More generally,

the quantity learned by Gradient Boosting depends upon

the selected loss function.

For Mean Squared Error,

the residuals happen to equal

\[
y-\hat{y}.
\]

For other loss functions,

the correction term is different.

This observation leads to one of the most important discoveries in modern machine learning.

Instead of directly learning ordinary residuals,

Gradient Boosting actually learns

the **direction in which the loss decreases most rapidly.**

Mathematically,

that direction is determined by

the

# Gradient.

The introduction of gradients transforms boosting from a simple residual-correction algorithm into a general optimization framework capable of minimizing almost any differentiable loss function.

Understanding gradients is therefore the next major milestone in our study of Gradient Boosting.

## 6.28 From Residuals to Gradients

In the previous sections,

we repeatedly stated that every new decision tree learns the residuals left by the previous model.

This description is perfectly accurate for the simplest regression problems,

where the loss function is

Mean Squared Error (MSE).

However,

Gradient Boosting was designed to solve a much broader class of problems.

It can optimize

- regression,
- binary classification,
- multi-class classification,
- ranking problems,
- and many other objective functions.

This immediately creates a challenge.

For some loss functions,

the residual

\[
y-\hat{y}
\]

is no longer the correct quantity to learn.

If ordinary residuals cannot always be used,

what should the next tree learn instead?

The answer is one of the most elegant ideas in machine learning.

Instead of learning the residual itself,

Gradient Boosting learns the **gradient** of the loss function.

In fact,

this is the reason the algorithm is called

**Gradient Boosting.**

The word

"Gradient"

is not merely a mathematical decoration.

It describes the exact mechanism that allows the algorithm to improve its predictions.

---

## 6.29 What Is a Gradient?

Before discussing machine learning,

let us recall an idea from elementary calculus.

Suppose we have a function

\[
f(x).
\]

The derivative

\[
\frac{df}{dx}
\]

tells us

how rapidly the function changes when

\(x\)

changes.

If the derivative is positive,

the function increases.

If the derivative is negative,

the function decreases.

If the derivative is zero,

the function has reached a stationary point,

which may correspond to a minimum or maximum.

The derivative therefore provides

both

the direction

and

the rate of change.

When a function depends on many variables,

the collection of all partial derivatives is called the

# Gradient.

For example,

suppose

\[
f(x,y,z).
\]

Its gradient is

\[
\nabla f=
\left(
\frac{\partial f}{\partial x},
\frac{\partial f}{\partial y},
\frac{\partial f}{\partial z}
\right).
\]

This vector points in the direction of the steepest increase of the function.

---

## 6.30 A Mountain Analogy

Imagine standing somewhere on a mountain.

Your goal is to reach the lowest point in the valley.

However,

you are surrounded by fog.

You cannot see the entire mountain.

The only information available is

the slope beneath your feet.

If the ground slopes upward toward the east,

you should move west.

If the ground slopes upward toward the north,

you should move south.

The local slope tells you

which direction decreases your elevation.

```text
          Mountain Peak

               /\

              /  \

             /    \

            /

     You ●

          ↓

      Move Downhill

______________________________

         Valley
```

Notice that

you never need to know the entire mountain.

You only need to know

the local slope.

Optimization algorithms work in exactly the same way.

They repeatedly compute the local slope

and move toward lower values.

---

## 6.31 The Gradient of the Loss Function

In machine learning,

the "mountain"

is not a physical landscape.

Instead,

it is the

loss function.

The height of every point represents

prediction error.

Large loss corresponds to high elevations.

Small loss corresponds to valleys.

Our objective is to find the lowest possible point.

Instead of asking

> "How should the prediction change?"

Gradient Boosting asks

> **"In which direction should the prediction move to decrease the loss most rapidly?"**

The answer is given by

the gradient of the loss function.

---

## 6.32 Why Do We Move Opposite to the Gradient?

An important detail often confuses beginners.

The gradient points toward

the direction of **maximum increase**.

However,

our goal is

to decrease the loss.

Therefore,

we move in the opposite direction.

Mathematically,

if

\[
\nabla L
\]

represents the gradient,

then optimization follows

\[
-\nabla L.
\]

This quantity is called

the

# Negative Gradient.

The negative gradient always points toward

the direction of greatest reduction in the loss.

Gradient Boosting trains every new tree to approximate

this negative gradient.

---

## 6.33 Connecting Residuals and Gradients

At this point,

you may wonder

whether residuals and gradients are two completely different concepts.

Surprisingly,

they are closely related.

Consider the Mean Squared Error loss.

\[
L
=
\frac12
(y-\hat{y})^2.
\]

The factor

\(\frac12\)

is introduced only to simplify differentiation.

It does not change the location of the minimum.

Taking the derivative with respect to the prediction,

\[
\hat{y},
\]

gives

\[
\frac{\partial L}{\partial \hat{y}}
=
\hat{y}
-
y.
\]

Therefore,

the negative gradient becomes

\[
-\frac{\partial L}{\partial \hat{y}}
=
y
-
\hat{y}.
\]

But

\[
y-\hat{y}
\]

is exactly

the residual.

This is a remarkable result.

For ordinary least-squares regression,

**the negative gradient is identical to the residual.**

This explains why introductory descriptions of Gradient Boosting often say

"the next tree learns the residuals."

Strictly speaking,

the next tree learns

the negative gradient.

Residuals are simply a special case.

---

## 6.34 Why This Generalization Matters

Suppose we replace Mean Squared Error with another loss function.

For example,

classification problems often use

Log Loss.

If we differentiate Log Loss,

the resulting negative gradient is

**not**

equal to

\[
y-\hat{y}.
\]

Consequently,

learning ordinary residuals would no longer minimize the loss correctly.

By learning

negative gradients instead,

Gradient Boosting automatically adapts to

any differentiable loss function.

This single mathematical idea explains why the same algorithm can solve many different machine learning problems.

---

## 6.35 An Intuitive View

Rather than memorizing equations,

it is useful to remember the overall philosophy.

The current model makes predictions.

↓

The loss function measures

how bad those predictions are.

↓

The gradient tells us

how the predictions should change to reduce the loss.

↓

A new decision tree learns those required corrections.

↓

The corrections are added to the model.

↓

The loss decreases.

↓

The process repeats.

This cycle continues until additional trees produce only negligible improvements.

---

## 6.36 Why Gradients Are One of the Most Important Ideas in Machine Learning

The concept of gradients extends far beyond Gradient Boosting.

Neural Networks,

Deep Learning,

Logistic Regression,

Support Vector Machines,

and many other algorithms all rely on gradients during optimization.

Once you understand

that a gradient simply indicates

**the direction of greatest increase of a function,**

and that optimization moves in

the opposite direction,

many seemingly unrelated machine learning algorithms begin to follow the same underlying logic.

Gradient Boosting is therefore much more than a tree algorithm.

It is one of the earliest successful applications of gradient-based optimization outside classical numerical analysis.

In the next section,

we will combine everything learned so far and derive the complete Gradient Boosting algorithm step by step,

showing exactly how trees,

loss functions,

negative gradients,

and additive models work together during training.

## 6.37 The Complete Gradient Boosting Algorithm

We have now introduced all of the major ideas required to understand Gradient Boosting.

So far, we have learned that

- a decision tree makes predictions,
- a loss function measures prediction quality,
- gradients indicate how predictions should change,
- every new tree learns the negative gradient,
- and the predictions from successive trees are added together.

Each of these ideas is important individually.

However,

the true power of Gradient Boosting becomes apparent only when we combine them into one complete learning algorithm.

Instead of viewing Gradient Boosting as a mysterious "black box,"

we will now build it step by step from first principles.

---

## 6.38 Step 1 — Start with an Initial Prediction

Every machine learning algorithm requires a starting point.

Gradient Boosting is no exception.

Before training the first decision tree,

the algorithm constructs a very simple initial model.

This initial model ignores every input feature.

It predicts exactly the same value for every training sample.

For regression problems using Mean Squared Error,

the optimal constant prediction is simply the mean of the target values.

Suppose we have measured the formation energies of five materials.

| Material | Formation Energy (eV/atom) |
|----------|---------------------------:|
| A | -3.20 |
| B | -2.80 |
| C | -2.40 |
| D | -3.60 |
| E | -3.00 |

The average becomes

\[
\frac{-3.20-2.80-2.40-3.60-3.00}{5}
=
-3.00.
\]

Therefore,

before any trees are built,

the model predicts

```text
Material A → -3.00

Material B → -3.00

Material C → -3.00

Material D → -3.00

Material E → -3.00
```

Clearly,

these predictions are not perfect.

Nevertheless,

they provide a reasonable starting point.

---

## 6.39 Why Start with the Mean?

At first,

predicting the same value for every material may seem like a poor strategy.

However,

there is an important mathematical reason.

Suppose we choose

\[
c
\]

as the constant prediction.

The loss function becomes

\[
L(c)
=
\sum_{i=1}^{n}(y_i-c)^2.
\]

Differentiating with respect to

\(c\),

\[
\frac{dL}{dc}
=
-2
\sum_{i=1}^{n}
(y_i-c).
\]

Setting the derivative equal to zero,

\[
\sum_{i=1}^{n}(y_i-c)=0.
\]

Rearranging,

\[
nc
=
\sum_{i=1}^{n}y_i.
\]

Therefore,

\[
c
=
\frac1n
\sum_{i=1}^{n}y_i.
\]

This is simply

the arithmetic mean.

Thus,

for squared-error regression,

the average target value is mathematically the best constant prediction possible.

---

## 6.40 Step 2 — Compute the Residuals

After making the initial predictions,

the algorithm calculates

the residual for every training sample.

Recall that

\[
Residual
=
True
-
Prediction.
\]

Using the previous example,

| Material | True | Initial Prediction | Residual |
|----------|-----:|-------------------:|---------:|
| A | -3.20 | -3.00 | -0.20 |
| B | -2.80 | -3.00 | 0.20 |
| C | -2.40 | -3.00 | 0.60 |
| D | -3.60 | -3.00 | -0.60 |
| E | -3.00 | -3.00 | 0.00 |

These residuals represent

the remaining information

that the model has not yet explained.

The first decision tree will be trained

not on the original formation energies,

but on these residuals.

---

## 6.41 Step 3 — Train the First Decision Tree

The first tree receives

- the original feature matrix,

and

- the residuals as its target values.

Suppose our descriptors include

- density,
- average electronegativity,
- atomic radius,
- unit-cell volume.

The tree attempts to discover

which combinations of these descriptors explain the residuals.

Notice something important.

The tree is **not**

predicting formation energy.

Instead,

it predicts

how much the current model should be corrected.

Its output may look like

```text
Material A

Predicted Correction

-0.18
```

or

```text
Material D

Predicted Correction

-0.55
```

These are not final predictions.

They are correction terms.

---

## 6.42 Step 4 — Update the Predictions

Once the tree predicts the corrections,

the model updates its previous predictions.

Mathematically,

\[
F_1(x)
=
F_0(x)
+
T_1(x),
\]

where

- \(F_0(x)\) is the initial model,
- \(T_1(x)\) is the first decision tree.

Suppose

the initial prediction was

```text
-3.00
```

and

the tree predicts

```text
-0.18
```

The updated prediction becomes

```text
-3.18
```

Notice that

the prediction is gradually moving closer to the true value.

The original prediction is not discarded.

It is refined.

---

## 6.43 Step 5 — Compute New Residuals

After updating the predictions,

the residuals change.

Suppose the true value is

```text
-3.20
```

The updated prediction becomes

```text
-3.18
```

The new residual is

```text
-0.02
```

The remaining error is now much smaller.

The next decision tree therefore solves a much easier problem.

Instead of correcting

0.20,

it needs to correct only

0.02.

---

## 6.44 Step 6 — Train Another Tree

The second tree again receives

the original descriptors,

but now it learns

the new residuals.

Perhaps it predicts

```text
-0.015
```

The prediction becomes

```text
-3.18

+

(-0.015)

=

-3.195
```

The remaining residual shrinks further.

The process continues repeatedly.

Each tree contributes

smaller

and

smaller

corrections.

---

## 6.45 The Complete Learning Cycle

The complete Gradient Boosting training procedure can now be summarized.

```text
Training Dataset

↓

Initial Prediction

↓

Compute Residuals

↓

Train Decision Tree

↓

Predict Corrections

↓

Update Predictions

↓

Compute New Residuals

↓

Train Next Tree

↓

Repeat Until Loss Stops Improving
```

Notice that

every iteration follows exactly the same sequence.

Nothing fundamentally changes.

Only the residuals become progressively smaller.

---

## 6.46 Why Does This Process Converge?

At each iteration,

the model attempts to reduce the loss function.

As long as

every new tree decreases the loss,

the predictions become increasingly accurate.

Eventually,

the residuals become extremely small.

At that stage,

even an additional decision tree cannot significantly reduce the loss.

Training naturally approaches a point where

the model has extracted nearly all predictable information from the available data.

This gradual improvement is one of the reasons Gradient Boosting often produces remarkably accurate models.

Rather than making one large adjustment,

it performs

hundreds

or even

thousands

of small, carefully chosen corrections.

---

## 6.47 A Materials Informatics Perspective

Suppose we wish to predict

the bulk modulus

of thousands of crystalline compounds.

The initial prediction might simply be

the average bulk modulus of the entire dataset.

The first tree may discover that

materials with high density generally possess larger bulk moduli.

The second tree may recognize that

covalent compounds deviate systematically from this trend.

The third tree may account for

crystal packing effects.

The fourth tree may capture

the influence of average atomic radius.

The fifth tree may identify

specific interactions between density and electronegativity.

Rather than attempting to learn all of these relationships simultaneously,

Gradient Boosting discovers them gradually,

one correction at a time.

Each tree captures a portion of the remaining unexplained physics.

Together,

these small corrections combine to produce an extremely accurate predictive model.

In the next section,

we will introduce another critical concept that makes Gradient Boosting both accurate and stable:

the **Learning Rate (Shrinkage)**.

Although adding every correction directly seems reasonable,

doing so often causes the model to overfit.

The learning rate controls how much each tree is allowed to influence the final prediction and is one of the most important hyperparameters in Gradient Boosting.

## 6.48 Why Can't We Simply Add Every Correction?

The Gradient Boosting algorithm developed in the previous section appears almost perfect.

At every iteration,

a new decision tree learns the remaining prediction errors.

Those corrections are then added to the existing model.

The loss decreases.

The predictions improve.

This naturally suggests another question.

If adding corrections improves the model,

why not add the **entire** correction produced by every tree?

Wouldn't that reduce the error as quickly as possible?

Surprisingly,

the answer is often

**no.**

Adding every correction completely can cause the model to learn too aggressively.

Instead of gradually discovering the underlying physical relationships,

the model may begin memorizing random fluctuations in the training dataset.

To understand why,

we first need to examine how a prediction changes after every iteration.

---

## 6.49 Updating Predictions Without a Learning Rate

Suppose the current prediction for a material is

```text
Predicted Formation Energy

-3.00 eV/atom
```

The next decision tree predicts a correction of

```text
-0.40 eV/atom
```

If we add the entire correction,

the updated prediction becomes

```text
-3.00

+

(-0.40)

=

-3.40 eV/atom
```

Mathematically,

the update is

\[
F_{m}(x)
=
F_{m-1}(x)
+
T_m(x),
\]

where

- \(F_{m-1}(x)\) is the previous model,
- \(T_m(x)\) is the prediction of the new tree.

Every tree contributes its full prediction.

At first glance,

this appears reasonable.

However,

there is an important problem.

---

## 6.50 The Danger of Large Corrections

Imagine learning to throw darts.

Your first throw lands

20 cm

to the left of the target.

Someone tells you

"Move your aim 20 cm to the right."

You follow the advice exactly.

Now,

the next dart lands

20 cm

to the right of the target.

You have corrected too aggressively.

Instead of approaching the center gradually,

you overshot it.

A more sensible strategy would be

to adjust your aim only part of the way,

observe the result,

and continue making small improvements.

Machine learning behaves in exactly the same way.

Large corrections can overshoot the optimal solution.

Small corrections usually produce more stable learning.

---

## 6.51 Introducing the Learning Rate

To prevent overly aggressive updates,

Gradient Boosting introduces one of its most important hyperparameters.

# Learning Rate

The learning rate determines

how much influence each new decision tree has on the final prediction.

Instead of adding the entire correction,

the algorithm multiplies it by a small constant,

usually denoted by

\[
\eta
\]

(the Greek letter eta).

The prediction update becomes

\[
F_m(x)
=
F_{m-1}(x)
+
\eta T_m(x).
\]

The value

\[
\eta
\]

lies between

0

and

1.

Typical values include

```text
0.3

0.1

0.05

0.01
```

The smaller the learning rate,

the smaller each correction becomes.

---

## 6.52 Example of Shrinkage

Suppose the current prediction is

```text
-3.00
```

The new tree predicts

```text
-0.40
```

If the learning rate is

```text
1.0
```

the updated prediction becomes

```text
-3.40
```

Now suppose

\[
\eta=0.1.
\]

The correction becomes

```text
0.1

×

(-0.40)

=

-0.04
```

The updated prediction is now

```text
-3.00

+

(-0.04)

=

-3.04
```

Instead of making one large jump,

the model moves only a small distance toward the optimum.

---

## 6.53 Why Is It Called Shrinkage?

The learning rate is often called

# Shrinkage

because it shrinks

every tree's prediction before it is added to the ensemble.

The tree itself may predict

```text
-0.40
```

but after shrinkage,

only

```text
-0.04
```

is actually used.

The tree has not changed.

Only its contribution has been reduced.

This seemingly simple idea dramatically improves the generalization ability of Gradient Boosting.

---

## 6.54 Small Learning Rates Usually Produce Better Models

An interesting observation repeatedly appears in practical machine learning.

Models trained with

smaller learning rates

often achieve

higher predictive accuracy.

Why?

Because the model learns more cautiously.

Instead of allowing a single tree to dominate,

many trees cooperate to build the final prediction.

This gradual learning process produces smoother approximations of complex functions.

It also reduces the chance of memorizing random noise.

However,

there is a trade-off.

Smaller learning rates require

more decision trees.

---

## 6.55 The Trade-Off Between Learning Rate and Number of Trees

Suppose we compare two Gradient Boosting models.

Model A

```text
Learning Rate

0.5

Trees

40
```

Model B

```text
Learning Rate

0.05

Trees

400
```

Although Model B uses many more trees,

each tree contributes only a small correction.

Consequently,

the final model often generalizes better to unseen data.

This relationship is one of the most important practical principles in Gradient Boosting.

Large learning rate

↓

Fewer trees

↓

Faster training

↓

Higher risk of overfitting.

Small learning rate

↓

More trees

↓

Slower training

↓

Better generalization.

Neither approach is universally superior.

The optimal balance depends upon

the dataset,

the computational resources,

and the prediction task.

---

## 6.56 Visualizing the Effect of the Learning Rate

Imagine trying to descend a staircase.

A large learning rate resembles taking very large jumps.

```text
Top

↓

████████

↓

████████

↓

Bottom
```

You may reach the bottom quickly,

but you also risk missing a step and falling.

A small learning rate resembles walking carefully,

one step at a time.

```text
Top

↓

█

↓

█

↓

█

↓

█

↓

█

↓

Bottom
```

The journey requires more steps,

but it is considerably more stable.

Gradient Boosting follows the same philosophy.

Many small improvements are usually safer than a few large ones.

---

## 6.57 Learning Rate in Materials Informatics

Consider a dataset containing

20,000 crystalline materials

from the Materials Project.

The descriptors include

- density,
- average electronegativity,
- atomic radius,
- packing fraction,
- oxidation-state statistics,
- lattice parameters,
- and compositional descriptors generated using Pymatgen.

These descriptors interact in highly nonlinear ways.

If each decision tree were allowed to make very large corrections,

the model might quickly memorize subtle numerical fluctuations produced by

- finite DFT convergence tolerances,
- small numerical differences,
- or limited sampling of chemical space.

A smaller learning rate forces the model to learn these complex relationships gradually.

Instead of allowing one tree to dominate,

hundreds of trees collaborate,

each capturing a small portion of the remaining unexplained physics.

This cooperative learning process is one of the primary reasons Gradient Boosting often performs exceptionally well on complex materials informatics datasets.

---

## 6.58 Choosing the Learning Rate in Practice

There is no universally optimal learning rate.

Nevertheless,

years of practical experience have produced several useful guidelines.

| Learning Rate | Typical Behavior |
|---------------|------------------|
| 1.0 | Very aggressive updates, often unstable |
| 0.3 | Fast learning, moderate overfitting risk |
| 0.1 | Common default for many applications |
| 0.05 | Slower but often more accurate |
| 0.01 | Very stable, usually requires many trees |

Rather than selecting the learning rate arbitrarily,

machine learning practitioners usually tune it together with

the number of trees,

tree depth,

and other hyperparameters.

This coordinated tuning process will become even more important when we study

**XGBoost**,

where the learning rate works together with powerful regularization techniques to produce state-of-the-art predictive performance.

In the next section,

we will examine another fundamental hyperparameter of Gradient Boosting:

the **Number of Trees (Estimators)**,

and understand why simply adding more trees does not always improve prediction accuracy.## 6.59 The Number of Trees (Estimators)

After introducing the learning rate,

another important question naturally follows.

If each decision tree contributes only a small correction,

how many trees should the model contain?

Should we build

10 trees?

100 trees?

1,000 trees?

10,000 trees?

At first,

it may seem that adding more trees should always improve prediction accuracy.

After all,

every new tree attempts to reduce the remaining prediction error.

However,

machine learning rarely works this way.

Beyond a certain point,

adding additional trees may provide almost no improvement,

and in some situations,

it may even reduce the model's ability to generalize to unseen data.

Understanding the role of the number of trees is therefore essential for building effective Gradient Boosting models.

---

## 6.60 What Is an Estimator?

In Scikit-learn,

the number of trees is controlled using the parameter

```python
n_estimators
```

The term

**estimator**

simply refers to

one individual decision tree

within the boosting ensemble.

If

```python
n_estimators = 100
```

the final model consists of

100 sequentially trained decision trees.

Unlike Random Forest,

these trees are **not** independent.

Every estimator depends on the predictions made by all previous estimators.

The learning process therefore looks like

```text
Initial Model

↓

Tree 1

↓

Tree 2

↓

Tree 3

↓

...

↓

Tree 100

↓

Final Model
```

Each tree builds upon the work of its predecessors.

---

## 6.61 Why More Trees Usually Improve the Model

Suppose the initial prediction for a material's formation energy is

```text
-3.00 eV/atom
```

Tree 1 improves it to

```text
-3.08
```

Tree 2 improves it further

```text
-3.13
```

Tree 3 refines it again

```text
-3.17
```

Tree 4 produces

```text
-3.19
```

Eventually,

the prediction approaches the true value

```text
-3.20
```

Notice what happens.

Each tree contributes only a small improvement.

No single tree solves the entire problem.

Instead,

accuracy emerges from the accumulation of many small corrections.

This gradual refinement is the defining characteristic of Gradient Boosting.

---

## 6.62 Diminishing Returns

Although every new tree attempts to reduce the loss,

its contribution generally becomes smaller over time.

The first few trees usually correct large systematic errors.

Later trees address increasingly subtle patterns.

Eventually,

very little error remains.

At this stage,

new trees may reduce the loss by only a tiny amount.

Conceptually,

the improvement often resembles the following.

```text
Prediction Error

|

|\
| \
|  \
|   \
|    \
|     \
|      \__
|          \__
|             \____

+------------------------------------>

Number of Trees
```

The largest improvements occur during the early stages of training.

Later,

the curve begins to flatten.

---

## 6.63 When Too Many Trees Become Harmful

Suppose our training dataset contains

small measurement errors,

experimental uncertainty,

or numerical noise from Density Functional Theory calculations.

Initially,

Gradient Boosting learns genuine physical relationships.

For example,

it may discover that

- higher density generally increases bulk modulus,
- stronger bonding lowers formation energy,
- larger band gaps correlate with ionic compounds.

Once these important patterns have been learned,

additional trees begin searching for any remaining error.

Unfortunately,

the remaining error may no longer represent meaningful physics.

Instead,

it may consist of

- measurement noise,
- rounding errors,
- convergence fluctuations,
- accidental statistical variations.

The later trees may begin fitting these random fluctuations.

This process eventually leads to

**overfitting.**

---

## 6.64 Training Error Versus Testing Error

The effect of increasing the number of trees can be understood by comparing

training error

and

testing error.

```text
Prediction Error

^

|

|\

| \

|  \

|   \

|    \_________

|              \____

|                   \______

+---------------------------------------->

Number of Trees

Training Error

Testing Error
```

The training error almost always decreases as more trees are added.

However,

the testing error behaves differently.

Initially,

it also decreases.

After reaching an optimal point,

it begins increasing because the model starts memorizing the training dataset.

The goal is therefore

not to minimize the training error,

but to minimize the testing error.

---

## 6.65 Choosing the Optimal Number of Trees

How do we know when enough trees have been trained?

One possibility is

trial and error.

Train several models with different values of

```python
n_estimators
```

and compare their validation performance.

For example,

| Trees | Validation RMSE |
|------:|----------------:|
| 50 | 0.48 |
| 100 | 0.39 |
| 200 | 0.32 |
| 300 | 0.30 |
| 500 | 0.31 |
| 800 | 0.34 |

Here,

the best model uses

approximately

300 trees.

Adding more trees no longer improves prediction accuracy.

Instead,

the validation error begins increasing.

---

## 6.66 Interaction Between Learning Rate and Number of Trees

One of the most important relationships in Gradient Boosting is

the interaction between

the learning rate

and

the number of estimators.

Suppose we compare two models.

Model A

```text
Learning Rate

0.3

Trees

100
```

Model B

```text
Learning Rate

0.03

Trees

1000
```

Although both models may achieve similar prediction accuracy,

they learn in completely different ways.

Model A

takes

large corrective steps

using relatively few trees.

Model B

takes

very small corrective steps

using many trees.

In practice,

smaller learning rates combined with more trees frequently produce smoother,

more robust models.

This relationship is so important that these two hyperparameters are almost always tuned together.

Changing one usually requires adjusting the other.

---

## 6.67 A Materials Informatics Example

Imagine predicting

the formation energies of

50,000 inorganic compounds.

The feature matrix contains descriptors generated from crystal structures,

including

- density,
- packing efficiency,
- average atomic mass,
- electronegativity statistics,
- lattice constants,
- oxidation-state descriptors,
- coordination statistics.

If we use

only

20 trees,

the model may learn only the most obvious trends.

Many complex nonlinear relationships remain unexplained.

If we increase the number of trees to

300,

the model begins capturing subtle interactions between composition,

structure,

and bonding.

However,

if we continue increasing the number of trees to

5,000

without appropriate regularization,

the model may begin learning numerical artifacts originating from

- DFT convergence thresholds,
- incomplete structural relaxation,
- finite precision calculations,

rather than genuine materials physics.

Consequently,

prediction accuracy on unseen compounds may decrease.

---

## 6.68 Computational Cost

Adding more trees not only affects prediction accuracy,

it also increases computational cost.

Every additional tree requires

- gradient computation,
- tree construction,
- prediction updates,
- memory allocation.

Training time therefore grows approximately in proportion to the number of estimators.

Similarly,

prediction also becomes slower because every tree contributes to the final output.

A model containing

1,000 trees

must evaluate

1,000 decision trees

before producing a prediction.

Consequently,

choosing an unnecessarily large value of

```python
n_estimators
```

may increase computational expense without providing meaningful gains in accuracy.

---

## 6.69 Practical Guidelines

Although every dataset is different,

several practical guidelines have emerged from years of experience.

- Use relatively few trees only when the learning rate is large.
- Use more trees when the learning rate is small.
- Monitor validation performance rather than training performance.
- Avoid selecting the number of trees solely by intuition.
- Tune the learning rate and number of estimators together.

These principles form the foundation of efficient Gradient Boosting models.

However,

they still do not solve the entire overfitting problem.

Even with an appropriate number of trees,

each individual decision tree may itself become too complex.

Fortunately,

Gradient Boosting provides another powerful mechanism for controlling model complexity:

limiting the **depth of each decision tree**.

In the next section,

we will study why Gradient Boosting intentionally uses **small, shallow decision trees** instead of allowing every tree to grow as deeply as possible.

## 6.70 Why Doesn't Gradient Boosting Use Large Decision Trees?

At this point,

we have learned that Gradient Boosting builds an ensemble of many decision trees.

A natural question now arises.

If decision trees become more accurate as they grow deeper,

why doesn't Gradient Boosting simply build

very large,

fully grown trees?

Wouldn't larger trees produce better predictions?

Surprisingly,

the answer is usually

**no.**

Modern Gradient Boosting algorithms intentionally use

**small, shallow decision trees.**

These trees are often called

**weak learners**.

Although each tree is individually simple,

their sequential combination produces an extremely powerful predictive model.

Understanding why this strategy works is one of the keys to understanding Gradient Boosting.

---

## 6.71 What Is a Weak Learner?

A weak learner is a model that performs

only slightly better

than random guessing.

This definition may sound strange.

Why would we intentionally choose

a weak model

instead of a strong one?

The answer lies in cooperation.

Imagine asking

one scientist

to solve an extremely difficult research problem.

The scientist may overlook important details.

Now imagine assembling

one hundred specialists,

each contributing a small piece of knowledge.

Together,

their combined understanding becomes far more powerful than any individual expert.

Gradient Boosting follows exactly this philosophy.

Instead of building

one highly complex decision tree,

it builds

many simple trees,

each correcting the mistakes of the previous ones.

---

## 6.72 Decision Stumps

The simplest possible decision tree is called a

# Decision Stump

A decision stump contains

only

one split.

For example,

```text
Density > 5.2 ?

        /       \

     Yes        No
```

That is the entire tree.

Only one decision is made.

Although such a tree cannot accurately solve complicated problems,

it usually captures

one important trend

within the data.

Gradient Boosting combines

many such simple trees

to approximate highly complex nonlinear relationships.

---

## 6.73 Why Small Trees Generalize Better

Suppose we allow every tree to grow

until every leaf becomes perfectly pure.

The first tree alone might already memorize a large portion of the training dataset.

The second tree would then have very little meaningful information left to learn.

Instead,

it would begin fitting random fluctuations.

The third tree would continue fitting increasingly insignificant patterns.

The ensemble would rapidly overfit.

Now consider shallow trees.

Each tree captures

only the largest remaining error.

Because every individual tree has limited complexity,

it cannot memorize every detail of the dataset.

Instead,

the ensemble gradually accumulates knowledge across many iterations.

This collaborative learning process generally produces

better generalization.

---

## 6.74 The Bias-Variance Perspective

Earlier in this book,

we studied the

Bias–Variance Trade-off.

Gradient Boosting provides an interesting example of this principle.

A shallow decision tree possesses

relatively high bias.

Because it is simple,

it cannot perfectly describe complicated nonlinear relationships.

However,

its variance is relatively low.

It is less sensitive to random fluctuations within the training dataset.

Conversely,

a very deep decision tree possesses

low bias

but

very high variance.

It easily memorizes individual training samples.

Gradient Boosting deliberately starts with

high-bias,

low-variance trees.

The boosting process gradually reduces the overall bias

without allowing the variance to become excessively large.

This balance is one of the primary reasons for its excellent predictive performance.

---

## 6.75 The Role of Tree Depth

The complexity of an individual decision tree is controlled using the parameter

```python
max_depth
```

For example,

```python
max_depth = 1
```

produces

decision stumps.

```python
max_depth = 2
```

allows

two consecutive splits.

```text
            Root

           /    \

       Node     Node

      /   \     /   \

    Leaf Leaf Leaf Leaf
```

Increasing the depth produces increasingly complex trees.

In practical Gradient Boosting,

typical values are surprisingly small.

Many successful models use

```text
3

4

5

6
```

rather than

20

or

30.

---

## 6.76 Why More Depth Is Not Always Better

Suppose we compare two Gradient Boosting models.

Model A

```text
Depth = 3

Trees = 500
```

Model B

```text
Depth = 20

Trees = 500
```

Although both models contain the same number of trees,

their behavior differs dramatically.

Model A

uses

many simple corrections.

Each tree captures broad,

general patterns.

Model B

allows every tree to learn highly specific decision rules.

Consequently,

the ensemble may memorize individual training samples

instead of learning robust physical relationships.

As the tree depth increases,

the risk of overfitting also increases.

---

## 6.77 An Intuitive Analogy

Imagine sculpting a statue.

One approach is to remove

a huge amount of material

with every cut.

You may reach the final shape quickly,

but one incorrect cut can permanently damage the sculpture.

Another approach is to remove

small amounts of material

during each pass.

Although the process requires more time,

the final sculpture is usually far more precise.

Gradient Boosting behaves similarly.

Each shallow decision tree performs

only a modest refinement.

The final model emerges through

hundreds of careful improvements,

rather than a few aggressive corrections.

---

## 6.78 Tree Depth in Materials Informatics

Suppose we are predicting

the band gap

of semiconductor materials.

The descriptors include

- average electronegativity,
- atomic radius statistics,
- density,
- crystal symmetry,
- packing efficiency,
- oxidation-state features,
- coordination environment.

A very deep decision tree may discover rules such as

```text
IF

Density > 5.2841

AND

Average Radius < 1.4213

AND

Packing Fraction > 0.6837

AND

Coordination Number = 5

AND

...
```

Eventually,

the decision rule becomes so specific

that it describes

only one or two training materials.

Such rules rarely represent universal physical principles.

Instead,

they reflect accidental details of the dataset.

A shallow tree,

however,

might learn

```text
IF

Average Electronegativity > 2.3

AND

Density > 5.0

↓

Higher Band Gap
```

This rule is much broader.

It is more likely to generalize to newly discovered compounds.

---

## 6.79 Computational Advantages

Shallow trees offer another important advantage.

They are computationally efficient.

Constructing a decision tree requires evaluating many possible splits.

A deep tree contains

many nodes,

many candidate splits,

and many recursive operations.

Training time therefore increases rapidly.

Shallow trees,

by contrast,

contain relatively few nodes.

They are faster to train,

require less memory,

and produce predictions more efficiently.

These computational benefits become especially important when working with

large materials databases containing

hundreds of thousands

or even

millions

of crystal structures.

---

## 6.80 Weak Learners Become Strong Together

One weak learner

cannot accurately model a complicated nonlinear function.

Two weak learners

perform slightly better.

Ten weak learners

capture increasingly complex relationships.

Hundreds of weak learners

working sequentially

can approximate remarkably complicated functions with very high accuracy.

This remarkable phenomenon explains the success of Gradient Boosting.

The algorithm does not rely on

one powerful decision tree.

Instead,

its strength emerges from

the cooperation of

many carefully controlled,

weak decision trees.

The idea is both elegant

and surprisingly effective.

In the next section,

we will bring together

the learning rate,

the number of trees,

and the tree depth,

and examine how these hyperparameters interact to determine the overall behavior of a Gradient Boosting model.

## 6.81 The Three Most Important Hyperparameters

At this stage,

we have encountered three hyperparameters that largely determine the behavior of a Gradient Boosting model.

1. Learning Rate

2. Number of Trees

3. Tree Depth

Individually,

each parameter influences how the model learns.

Collectively,

they determine

- prediction accuracy,
- computational cost,
- model complexity,
- and the tendency to overfit or underfit.

Understanding these parameters independently is useful.

Understanding how they interact is essential.

Professional machine learning practitioners rarely tune one parameter while ignoring the others.

Instead,

they consider the model as an interconnected system.

Changing one hyperparameter often requires adjusting the others.

---

## 6.82 Interaction Between Learning Rate and Number of Trees

Suppose we reduce the learning rate.

Every decision tree now contributes a smaller correction.

As a result,

the model learns more slowly.

Since each tree accomplishes less,

we need

more trees

to reach the same prediction accuracy.

Conversely,

if we increase the learning rate,

each tree contributes a larger correction.

The model learns faster,

so fewer trees are required.

This relationship can be summarized as

```text
Smaller Learning Rate

↓

Smaller Corrections

↓

Need More Trees
```

and

```text
Larger Learning Rate

↓

Larger Corrections

↓

Need Fewer Trees
```

Notice that neither parameter can be chosen independently.

---

## 6.83 Interaction Between Tree Depth and Number of Trees

Tree depth introduces another important interaction.

A shallow tree

captures only simple relationships.

Therefore,

many trees are required to approximate a complicated function.

A deeper tree,

however,

captures more information during each iteration.

Consequently,

fewer trees may be sufficient.

Conceptually,

the relationship is

```text
Shallow Trees

↓

Simple Corrections

↓

Need More Trees
```

whereas

```text
Deep Trees

↓

Complex Corrections

↓

Need Fewer Trees
```

Unfortunately,

deeper trees also increase the risk of overfitting.

The goal is therefore

not simply to minimize the number of trees,

but to achieve the best balance between complexity and generalization.

---

## 6.84 Interaction Between Learning Rate and Tree Depth

Now consider

learning rate

and

tree depth

together.

Suppose we use

very deep trees

combined with

a very large learning rate.

Each tree now performs

a large,

complex correction.

The model may rapidly memorize the training data.

Overfitting becomes highly likely.

On the other hand,

suppose we combine

very shallow trees

with

a very small learning rate.

Each iteration contributes only a tiny improvement.

Although this configuration often produces excellent generalization,

training may require thousands of trees.

The training process becomes computationally expensive.

The most effective Gradient Boosting models therefore balance

all three hyperparameters simultaneously.

---

## 6.85 Visualizing the Balance

Imagine adjusting three control knobs.

```text
             Gradient Boosting

        ┌───────────────────────┐

        │ Learning Rate         │

        │ Number of Trees       │

        │ Tree Depth            │

        └───────────────────────┘
```

Turning only one knob rarely produces the best model.

Instead,

all three must be adjusted together.

A balanced configuration produces

- low prediction error,
- strong generalization,
- reasonable training time,
- and robust performance on unseen data.

---

## 6.86 Typical Practical Configurations

Although every dataset is different,

certain parameter combinations appear frequently in practice.

| Learning Rate | Trees | Maximum Depth | Typical Behavior |
|---------------|------:|--------------:|------------------|
| 0.3 | 100 | 3 | Fast training, moderate complexity |
| 0.1 | 200 | 3 | Common default, good balance |
| 0.05 | 500 | 4 | Higher accuracy, slower training |
| 0.01 | 1000+ | 5 | Very cautious learning, computationally expensive |

These values are

starting points,

not universal rules.

The optimal configuration depends on

- dataset size,
- feature quality,
- noise level,
- and prediction objective.

---

## 6.87 Hyperparameter Tuning

Since the best hyperparameter values cannot usually be determined analytically,

they are found experimentally.

This process is called

# Hyperparameter Tuning

Unlike model parameters,

which are learned automatically during training,

hyperparameters are selected by the practitioner.

Examples include

```text
Learning Rate

Maximum Depth

Number of Trees

Minimum Samples per Leaf

Subsample Ratio
```

The objective is to discover the combination that minimizes prediction error on unseen data.

Several systematic search methods exist,

including

- Grid Search,
- Random Search,
- Bayesian Optimization,
- and evolutionary optimization techniques.

These methods will be studied in detail later in this book.

---

## 6.88 An Example From Materials Informatics

Suppose we are predicting

the bulk modulus

of crystalline materials.

The feature matrix contains

- compositional statistics,
- density,
- lattice constants,
- coordination descriptors,
- electronic descriptors,
- structural features extracted using Pymatgen.

We first train a Gradient Boosting model using

```text
Learning Rate

0.3

Trees

100

Depth

3
```

The model trains quickly,

but the validation error remains relatively high.

We reduce the learning rate to

```text
0.05
```

Since learning is now slower,

we increase the number of trees to

```text
500
```

The validation error decreases.

Next,

we experiment with

maximum depth.

Increasing the depth from

3

to

4

allows each tree to model slightly more complex interactions.

The validation error decreases further.

Increasing the depth again,

however,

causes the validation error to increase.

The model has begun to overfit.

The optimal configuration therefore becomes

```text
Learning Rate

0.05

Trees

500

Maximum Depth

4
```

Notice that

the improvement did not come from changing a single parameter.

Instead,

it resulted from balancing several interacting hyperparameters.

---

## 6.89 Why Hyperparameter Tuning Matters

Many beginners assume that machine learning performance depends primarily on the choice of algorithm.

In reality,

the same algorithm can produce dramatically different results depending on its hyperparameters.

A poorly tuned Gradient Boosting model may perform worse than

a well-tuned Random Forest.

Conversely,

a carefully optimized Gradient Boosting model may substantially outperform many alternative algorithms.

Hyperparameter tuning therefore represents

an essential component of professional machine learning,

not an optional refinement.

---

## 6.90 Preparing for Modern Gradient Boosting Algorithms

Everything we have studied so far describes

the fundamental principles

behind Gradient Boosting.

However,

classical Gradient Boosting still suffers from several practical limitations.

For example,

training can become

- computationally slow,
- memory intensive,
- and difficult to scale to very large datasets.

Furthermore,

the original algorithm lacks many of the optimization techniques required for modern large-scale machine learning.

To overcome these limitations,

new algorithms were developed.

Among them,

one algorithm became extraordinarily successful in both industry and scientific research:

**Extreme Gradient Boosting**, more commonly known as **XGBoost**.

Rather than replacing Gradient Boosting,

XGBoost refines and extends its ideas through

regularization,

efficient optimization,

parallel computation,

advanced tree construction,

and sophisticated handling of missing data.

In the next chapter,

we will begin a comprehensive study of XGBoost,

starting with the historical motivation behind its development and the limitations of classical Gradient Boosting that it was designed to solve.
