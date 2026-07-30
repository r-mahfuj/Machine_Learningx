# Chapter 9 — Hyperparameter Optimization

## 9.1 Introduction

In the previous chapters, we learned how machine learning models are built and how their performance is evaluated.

We studied:

- Linear Regression
- Decision Trees
- Random Forests
- Gradient Boosting
- XGBoost

We also learned how to evaluate these models using metrics such as:

- MAE
- RMSE
- R²
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Finally, we discussed model validation techniques, including:

- Train-Test Split
- K-Fold Cross-Validation
- Leave-One-Out Cross-Validation
- Nested Cross-Validation

At this point, you might think that once we choose an algorithm, the machine learning task is complete.

Unfortunately, that is not true.

Choosing the algorithm is only the beginning.

The real performance of a machine learning model often depends more on **how it is configured** than on **which algorithm is selected**.

Those configurations are called:

```
Hyperparameters
```

Learning how to choose them properly is one of the most important skills in practical machine learning.

---

# 9.2 Why Hyperparameter Optimization Matters

Imagine two researchers working on exactly the same materials dataset.

Both researchers use:

```
XGBoost
```

Both researchers use the same:

- descriptors,
- training data,
- validation strategy,
- evaluation metric.

Yet, one researcher reports:

```
RMSE = 0.08 eV/atom
```

while the other reports:

```
RMSE = 0.23 eV/atom
```

How is this possible?

The answer is often not the algorithm itself.

Instead, the difference lies in the hyperparameters.

Poorly chosen hyperparameters can make an excellent algorithm perform badly.

Carefully selected hyperparameters can dramatically improve prediction accuracy.

---

# 9.3 An Analogy: Building a Furnace

Consider a heat-treatment furnace used in materials engineering.

Suppose the furnace itself is the machine learning algorithm.

The operator must still decide:

- heating temperature,
- heating rate,
- holding time,
- cooling rate,
- protective atmosphere.

These settings determine whether the final microstructure is excellent or poor.

The furnace is the same.

Only the settings change.

Machine learning behaves in exactly the same way.

The algorithm remains unchanged,

but different settings can produce very different models.

These settings are called hyperparameters.

---

# 9.4 What Will You Learn in This Chapter?

By the end of this chapter, you will understand:

- what hyperparameters are,
- why they affect model performance,
- how to optimize them,
- when different optimization methods should be used,
- how hyperparameter tuning is performed in modern materials informatics.

You will also learn how hyperparameter optimization is applied to:

- Decision Trees,
- Random Forests,
- Gradient Boosting,
- XGBoost.

Finally, you will implement practical optimization workflows using Scikit-learn and modern optimization libraries.

---

# 9.5 Learning Objectives

After completing this chapter, you should be able to:

✔ Distinguish between model parameters and hyperparameters.

✔ Understand why hyperparameters control model complexity.

✔ Explain the advantages and disadvantages of Grid Search.

✔ Explain why Random Search often outperforms Grid Search.

✔ Understand the intuition behind Bayesian Optimization.

✔ Tune Decision Trees, Random Forests, Gradient Boosting, and XGBoost.

✔ Apply hyperparameter optimization to real materials datasets.

✔ Avoid common mistakes that lead to overfitting.

---

# 9.6 Why Materials Informatics Requires Careful Hyperparameter Tuning

Hyperparameter optimization is useful in every field of machine learning.

However,

it becomes especially important in materials informatics because of several unique challenges.

---

## Challenge 1 — Expensive DFT Calculations

Many machine learning datasets are generated using Density Functional Theory (DFT).

Each data point may require:

- geometry optimization,
- electronic structure calculations,
- convergence testing,
- high-performance computing resources.

Generating one additional data point may require hours or even days of computation.

Because data is expensive,

we must obtain the highest possible performance from every sample.

---

## Challenge 2 — Small Datasets

Many published materials datasets contain:

- a few hundred samples,
- a few thousand samples,
- rarely more than several tens of thousands.

Small datasets make overfitting much easier.

Hyperparameters therefore become critical for controlling model complexity.

---

## Challenge 3 — High-Dimensional Descriptors

Modern feature generation tools such as **Pymatgen** can produce hundreds or even thousands of descriptors.

Examples include:

- elemental properties,
- oxidation-state statistics,
- electronic descriptors,
- structural descriptors,
- compositional statistics.

Not every descriptor is equally informative.

Complex models can easily overfit these high-dimensional feature spaces.

Proper hyperparameter tuning helps prevent this problem.

---

## Challenge 4 — Large Chemical Space

Machine learning models are expected to predict properties for materials they have never seen before.

Examples include:

- new battery materials,
- novel catalysts,
- unexplored alloys,
- hypothetical crystal structures.

Hyperparameters strongly influence whether a model learns:

```
General chemical relationships
```

or simply memorizes the training database.

---

# 9.7 Hyperparameter Optimization Workflow

In practical materials machine learning,

hyperparameter optimization is not performed in isolation.

Instead,

it is integrated into the complete machine learning pipeline.

A typical workflow is:

```
Raw Materials Dataset

↓

Data Cleaning

↓

Feature Engineering
(Pymatgen)

↓

Feature Selection

↓

Choose Machine Learning Algorithm

↓

Hyperparameter Optimization

↓

Cross Validation

↓

Select Best Model

↓

Final Evaluation

↓

Predict New Materials
```

Every stage influences the next.

Hyperparameter optimization acts as the bridge between model construction and model evaluation.

---

# 9.8 The Cost of Ignoring Hyperparameters

Suppose we train an XGBoost model using only the default settings.

The model may produce:

```
RMSE = 0.24 eV/atom
```

After systematic hyperparameter optimization,

the same algorithm on the same dataset might achieve:

```
RMSE = 0.13 eV/atom
```

No additional data were collected.

No new descriptors were created.

The improvement came entirely from selecting better hyperparameters.

This illustrates an important lesson:

> In machine learning, choosing good hyperparameters can be as important as choosing the algorithm itself.

---

# 9.9 Roadmap of This Chapter

The remainder of this chapter is organized as follows.

First,

we will distinguish between **parameters** and **hyperparameters**, a distinction that every machine learning practitioner must understand.

Next,

we will examine the hyperparameters of the most important tree-based algorithms:

- Decision Trees,
- Random Forests,
- Gradient Boosting,
- XGBoost.

After understanding what each hyperparameter controls,

we will study the major optimization techniques:

- Manual Search,
- Grid Search,
- Random Search,
- Bayesian Optimization.

Finally,

we will apply these methods to realistic materials informatics problems involving:

- expensive DFT datasets,
- high-dimensional descriptors generated with Pymatgen,
- limited computational resources,
- XGBoost model optimization.

By the end of the chapter, you will understand not only **how** to tune hyperparameters, but **why** certain strategies are preferred in modern materials machine learning research.

# 9.10 Parameters vs Hyperparameters

Before learning how to optimize hyperparameters, we must first answer a fundamental question:

> **What exactly is a hyperparameter?**

Many beginners confuse **parameters** and **hyperparameters** because both influence the final machine learning model.

Although their names are similar, they represent two completely different concepts.

Understanding this distinction is essential because machine learning algorithms automatically learn **parameters**, while the user must choose **hyperparameters**.

---

# 9.11 What Are Parameters?

Parameters are the internal quantities that a machine learning algorithm learns automatically from the training data.

They represent the knowledge acquired by the model during training.

As the algorithm processes more data, these values change continuously until the learning process is complete.

In other words,

**parameters are discovered by the algorithm itself.**

The user does not directly choose their values.

---

# 9.12 Example: Linear Regression Parameters

Consider the familiar linear regression equation:

```
y = mx + b
```

where

```
m
```

is the slope,

and

```
b
```

is the intercept.

Suppose we want to predict the formation energy of a material from one descriptor.

Initially, the algorithm might start with:

```
Slope

=

0
```

```
Intercept

=

0
```

After observing many training samples, these values gradually change.

Eventually, the model may learn:

```
Slope

=

2.83
```

```
Intercept

=

−1.45
```

These values were **not chosen by the researcher**.

They were automatically learned from the data.

Therefore,

the slope and intercept are **parameters**.

---

# 9.13 Example: Decision Tree Parameters

A Decision Tree also learns parameters automatically.

During training, the algorithm determines:

- which feature to split,
- where to split,
- the structure of the tree,
- the prediction stored in each leaf node.

For example,

the algorithm may learn:

```
Root Split

Average Atomic Radius

<

1.15 Å
```

Later,

another split might be:

```
Electronegativity Difference

>

1.2
```

The researcher never specifies these split values.

The Decision Tree discovers them during training.

Therefore,

the split locations and leaf predictions are model parameters.

---

# 9.14 Example: XGBoost Parameters

XGBoost learns thousands of internal parameters.

These include:

- split locations,
- leaf weights,
- tree structures,
- prediction scores.

For a model containing hundreds of trees,

millions of numerical values may be learned automatically.

Again,

the researcher does not specify these numbers.

The optimization algorithm computes them.

---

# 9.15 What Are Hyperparameters?

Hyperparameters are different.

A hyperparameter is a value that controls **how the learning process occurs**.

Unlike parameters,

hyperparameters are **not learned from the data**.

Instead,

they are chosen before training begins.

Examples include:

- tree depth,
- learning rate,
- number of trees,
- regularization strength.

These settings determine:

- how flexible the model is,
- how quickly it learns,
- how much it can memorize,
- how well it generalizes.

---

# 9.16 A Simple Analogy

Imagine teaching a student for an examination.

The student's knowledge after studying represents the **parameters**.

The study plan represents the **hyperparameters**.

The teacher decides:

- how many hours to study,
- which books to use,
- how difficult the exercises should be,
- how often revision occurs.

These choices affect learning,

but they are decided **before** studying begins.

Similarly,

hyperparameters determine **how a machine learning algorithm learns**.

---

# 9.17 Comparing Parameters and Hyperparameters

The differences become clearer when viewed side by side.

| Parameters | Hyperparameters |
|------------|-----------------|
| Learned automatically | Chosen before training |
| Estimated from data | Selected by the user or optimization algorithm |
| Change during training | Usually fixed during training |
| Represent learned knowledge | Control the learning process |
| Different for every trained model | Can be reused across different datasets |

---

# 9.18 Why Hyperparameters Exist

One common question is:

> **If machine learning can learn parameters automatically, why can't it also learn every hyperparameter?**

The answer is that hyperparameters define the learning strategy itself.

For example,

suppose a Decision Tree has unlimited depth.

The algorithm can continue splitting until every training sample is isolated.

This produces extremely low training error,

but the model memorizes the dataset.

On the other hand,

if the tree is forced to remain very shallow,

the model cannot capture important relationships.

Neither choice is universally correct.

The appropriate value depends on:

- dataset size,
- feature complexity,
- noise level,
- prediction objective.

Therefore,

the researcher must guide the learning process.

---

# 9.19 Parameters and Hyperparameters in Different Algorithms

Every machine learning algorithm contains both parameters and hyperparameters.

Consider several examples.

### Linear Regression

Parameters:

- slope,
- intercept.

Hyperparameters:

Most basic linear regression has very few,

although regularized versions introduce hyperparameters such as regularization strength.

---

### Decision Tree

Parameters:

- learned split positions,
- leaf predictions.

Hyperparameters:

- maximum depth,
- minimum samples per split,
- splitting criterion.

---

### Random Forest

Parameters:

- every tree structure,
- every split,
- every leaf prediction.

Hyperparameters:

- number of trees,
- maximum depth,
- number of features considered at each split,
- bootstrap sampling.

---

### Gradient Boosting

Parameters:

- all learned trees,
- leaf values.

Hyperparameters:

- learning rate,
- number of estimators,
- maximum depth,
- subsampling ratio.

---

### XGBoost

Parameters:

- tree structures,
- leaf weights,
- split values.

Hyperparameters:

- learning rate,
- maximum depth,
- gamma,
- minimum child weight,
- subsample,
- column sampling,
- regularization parameters,
- number of estimators.

---

# 9.20 Why Hyperparameters Affect Performance

Imagine training two XGBoost models using exactly the same materials dataset.

Model A:

```
Maximum Depth

=

2
```

Model B:

```
Maximum Depth

=

12
```

Both models use identical:

- training data,
- features,
- evaluation metric.

Yet,

their predictions can differ dramatically.

The shallow model may fail to capture complex nonlinear relationships.

The deep model may memorize the training data.

Neither depth is automatically correct.

Choosing an appropriate depth is therefore a hyperparameter optimization problem.

---

# 9.21 Hyperparameters Define Model Complexity

Hyperparameters determine how complex a model is.

A model with too little complexity suffers from:

```
Underfitting
```

A model with excessive complexity suffers from:

```
Overfitting
```

The objective of hyperparameter optimization is to find a balance between these two extremes.

This balance is closely related to the bias-variance tradeoff discussed in earlier chapters.

---

# 9.22 Materials Science Example

Suppose we are predicting the elastic modulus of crystalline materials.

The dataset contains:

```
500 compounds
```

generated using DFT.

We decide to use XGBoost.

Without changing the dataset,

we can train hundreds of different models simply by changing the hyperparameters.

Each model may produce a different:

- RMSE,
- MAE,
- R² score.

The underlying algorithm remains identical.

Only the hyperparameters change.

This demonstrates that hyperparameter selection can be just as important as choosing the algorithm itself.

---

# 9.23 Can Hyperparameters Be Learned Automatically?

Although hyperparameters are not learned during ordinary model training,

they can be optimized automatically using specialized algorithms.

Instead of manually trying different values,

we can use:

- Grid Search,
- Random Search,
- Bayesian Optimization.

These optimization methods systematically explore possible hyperparameter values and identify combinations that produce the best validation performance.

We will study each of these techniques later in this chapter.

---

# 9.24 Summary

Machine learning models contain two distinct types of numerical quantities.

**Parameters** are learned automatically from the training data and represent the model's acquired knowledge.

Examples include:

- regression coefficients,
- Decision Tree split values,
- XGBoost leaf weights.

**Hyperparameters** are selected before training begins and control how the learning process proceeds.

Examples include:

- maximum tree depth,
- learning rate,
- number of estimators,
- regularization strength.

Understanding this distinction is the foundation of hyperparameter optimization.

In the next section, we will revisit the **bias-variance tradeoff** and see why nearly every hyperparameter ultimately influences one of these two competing sources of prediction error.

# 9.25 The Bias-Variance Tradeoff Revisited

In Chapter 3, we introduced one of the most fundamental ideas in machine learning:

```
Bias

vs

Variance
```

At that time, we discussed the concept mainly from the perspective of model complexity.

Now, we return to this topic because **almost every hyperparameter directly affects either bias, variance, or both**.

Understanding this relationship is essential.

If you understand how hyperparameters influence bias and variance,

you can often predict whether changing a hyperparameter will improve or worsen your model before running any code.

---

# 9.26 Why the Bias-Variance Tradeoff Matters

Every machine learning model tries to answer one question:

> **How can we accurately predict unseen data?**

Unfortunately,

there are two major sources of prediction error.

The first is:

```
Bias
```

The second is:

```
Variance
```

These two errors compete with one another.

Reducing one often increases the other.

The purpose of hyperparameter optimization is to find the balance where both errors are reasonably small.

---

# 9.27 What Is Bias?

Bias measures how much a model oversimplifies reality.

A high-bias model assumes that the relationship between inputs and outputs is much simpler than it actually is.

As a result,

the model cannot capture important patterns in the data.

This phenomenon is called:

```
Underfitting
```

---

# 9.28 Materials Science Example of High Bias

Suppose we want to predict:

```
Formation Energy
```

using descriptors generated by Pymatgen.

The true relationship depends on:

- atomic size,
- electronegativity,
- crystal symmetry,
- oxidation states,
- bonding environment,
- electronic interactions.

Now imagine using an extremely simple model that assumes:

```
Formation Energy

=

a × Atomic Radius + b
```

This model ignores almost every important physical factor.

Its predictions will be poor.

Even if we collect more training data,

the model still cannot represent the underlying relationship.

This is high bias.

---

# 9.29 Characteristics of High Bias

High-bias models usually have:

- simple decision boundaries,
- low model complexity,
- poor training accuracy,
- poor testing accuracy.

Typical symptoms include:

```
Training Error

High
```

```
Validation Error

High
```

Adding more data usually does not solve the problem.

The model itself is too simple.

---

# 9.30 What Is Variance?

Variance measures how sensitive a model is to changes in the training data.

A high-variance model learns the training data extremely well.

Unfortunately,

it also learns:

- random noise,
- measurement errors,
- accidental correlations.

As a result,

the model performs poorly on unseen data.

This phenomenon is called:

```
Overfitting
```

---

# 9.31 Materials Science Example of High Variance

Imagine training an XGBoost model on:

```
300 DFT calculations
```

The model contains:

```
500 trees

Maximum depth = 15
```

It learns almost every detail of the training dataset.

Training error becomes:

```
Nearly Zero
```

However,

when predicting newly synthesized materials,

the error increases dramatically.

The model memorized the training database instead of learning general chemical principles.

This is high variance.

---

# 9.32 Characteristics of High Variance

High-variance models usually show:

```
Training Error

Very Low
```

```
Validation Error

High
```

The model performs exceptionally well on familiar materials,

but poorly on new ones.

---

# 9.33 Visualizing the Bias-Variance Tradeoff

Imagine model complexity increasing from left to right.

```
Simple Model

──────────────►

Complex Model
```

As complexity increases:

```
Bias

↓

Decreases
```

because the model becomes more flexible.

However,

```
Variance

↑

Increases
```

because the model becomes more sensitive to the training data.

The goal is not to minimize bias alone,

or variance alone.

The goal is to minimize the **total prediction error**.

---

# 9.34 The Sweet Spot

The ideal model lies between the two extremes.

```
Too Simple

↓

Underfitting

↓

Balanced Complexity

↓

Good Generalization

↓

Too Complex

↓

Overfitting
```

Hyperparameter optimization searches for this balanced region.

---

# 9.35 Hyperparameters Control the Tradeoff

One of the most important ideas in this chapter is:

> **Hyperparameters do not directly improve accuracy.**

Instead,

they change the balance between:

- bias,
- variance.

Every hyperparameter shifts the model somewhere along this spectrum.

---

# 9.36 Example: Decision Tree Depth

Consider the Decision Tree hyperparameter:

```
max_depth
```

Suppose we choose:

```
Maximum Depth = 2
```

The tree becomes very shallow.

Advantages:

- simple,
- stable,
- unlikely to overfit.

Disadvantages:

- cannot model complex relationships.

Result:

```
High Bias

Low Variance
```

---

Now suppose:

```
Maximum Depth = 30
```

The tree becomes extremely complex.

Advantages:

- captures intricate patterns.

Disadvantages:

- memorizes training data,
- sensitive to noise.

Result:

```
Low Bias

High Variance
```

---

# 9.37 Example: Number of Trees in Random Forest

Consider:

```
n_estimators
```

A forest with:

```
10 trees
```

may not capture enough information.

Increasing the number of trees:

```
10

↓

100

↓

300

↓

500
```

usually reduces variance because predictions are averaged across many trees.

However,

after a certain point,

adding more trees produces only small improvements while increasing computational cost.

---

# 9.38 Example: Learning Rate in Gradient Boosting

Gradient Boosting and XGBoost introduce another important hyperparameter:

```
learning_rate
```

Suppose:

```
learning_rate = 0.5
```

Each new tree makes a very large correction.

Training proceeds quickly,

but the model may overfit.

Now suppose:

```
learning_rate = 0.01
```

Each tree contributes only a small improvement.

Learning becomes slower,

but often generalizes better.

This demonstrates another bias-variance tradeoff.

---

# 9.39 Example: Regularization

Regularization hyperparameters deliberately reduce model complexity.

Examples include:

```
L1 Regularization

(reg_alpha)
```

and

```
L2 Regularization

(reg_lambda)
```

Increasing regularization:

- reduces variance,
- increases bias.

Too much regularization,

however,

may cause underfitting.

---

# 9.40 Materials Science Perspective

Bias and variance have important physical interpretations.

A high-bias model may ignore meaningful scientific relationships.

For example,

it may fail to recognize:

- crystal symmetry,
- bonding effects,
- coordination environments.

A high-variance model,

on the other hand,

may incorrectly interpret random fluctuations in DFT calculations as real physical trends.

Neither situation is desirable.

The best model captures genuine physical relationships while ignoring random noise.

---

# 9.41 Bias-Variance in Small DFT Datasets

The bias-variance tradeoff becomes even more important when datasets are small.

Suppose we have:

```
200 materials
```

generated using expensive DFT calculations.

A highly flexible model may perfectly memorize these 200 compounds.

However,

its predictions for the next 200 unknown compounds may be poor.

Hyperparameter optimization helps control this tendency.

---

# 9.42 Bias-Variance and Cross-Validation

Cross-validation provides a practical way to observe the bias-variance tradeoff.

For example,

during Grid Search,

we may test several values of:

```
Maximum Depth
```

Each value produces a different validation score.

Very small depths often produce:

```
Underfitting
```

Very large depths often produce:

```
Overfitting
```

The optimal hyperparameter lies somewhere in between.

---

# 9.43 Why There Is No Universally Best Hyperparameter

A common beginner's question is:

> **What is the best value for maximum tree depth?**

There is no universal answer.

The best value depends on:

- dataset size,
- feature quality,
- noise level,
- prediction task,
- computational resources.

A value that performs well for:

```
Band Gap Prediction
```

may perform poorly for:

```
Elastic Modulus Prediction
```

Hyperparameter optimization exists precisely because these values are problem-dependent.

---

# 9.44 Summary

Every machine learning model balances two competing sources of error:

```
Bias

↓

Underfitting
```

and

```
Variance

↓

Overfitting
```

Hyperparameters control where a model lies on this spectrum.

Increasing model flexibility generally reduces bias but increases variance.

Increasing regularization or limiting model complexity generally reduces variance but increases bias.

The objective of hyperparameter optimization is to locate the point where this tradeoff produces the best generalization performance.

In the next section, we will begin studying the specific hyperparameters of tree-based algorithms, starting with the **Decision Tree**, and learn how each hyperparameter influences model complexity, bias, and variance.

# 9.45 Decision Tree Hyperparameters

In Chapter 2, we learned how a Decision Tree constructs a predictive model by repeatedly splitting the dataset into smaller and smaller groups.

The algorithm automatically decides:

- which feature to split,
- where to split,
- how many leaf nodes to create.

These learned quantities are the **parameters** of the Decision Tree.

However,

before the algorithm begins learning,

we must specify several settings that control **how the tree is allowed to grow**.

These settings are the **hyperparameters**.

The quality of a Decision Tree depends heavily on these hyperparameters.

A poorly chosen configuration may produce:

- severe underfitting,
- severe overfitting,
- unstable predictions,
- poor generalization.

Understanding each hyperparameter is therefore essential.

---

# 9.46 The Most Important Decision Tree Hyperparameters

The most commonly tuned Decision Tree hyperparameters are:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_leaf_nodes`
- `criterion`
- `splitter`

Each controls a different aspect of tree construction.

Together,

they determine the complexity of the final model.

---

# 9.47 The Role of Tree Growth

A Decision Tree begins with the complete dataset.

```
Entire Dataset

↓

Split

↓

Smaller Groups

↓

Split Again

↓

Leaf Nodes
```

Without restrictions,

the algorithm continues splitting until every leaf contains very few—or even a single—training sample.

Such a tree usually memorizes the training data.

Hyperparameters prevent uncontrolled growth.

---

# 9.48 Hyperparameter 1: max_depth

The most important Decision Tree hyperparameter is:

```
max_depth
```

It specifies the maximum number of levels that the tree is allowed to grow.

Example:

```
max_depth = 3
```

means that the longest path from the root node to any leaf can contain only three decision levels.

---

# 9.49 Understanding Tree Depth

Consider two trees.

Tree A:

```
Depth = 2
```

```
Root

├── Left

└── Right
```

Tree B:

```
Depth = 8
```

```
Root

↓

Several Levels

↓

Many Leaf Nodes
```

Tree B is much more flexible.

It can capture much more complicated relationships.

---

# 9.50 Effect of Small max_depth

Suppose:

```
max_depth = 2
```

The tree becomes shallow.

Advantages:

- simple interpretation,
- fast prediction,
- low computational cost,
- reduced overfitting.

Disadvantages:

- limited flexibility,
- high bias,
- may ignore important nonlinear relationships.

Typical result:

```
Training Error

High
```

```
Validation Error

High
```

This is underfitting.

---

# 9.51 Effect of Large max_depth

Now suppose:

```
max_depth = 20
```

The tree becomes much larger.

Advantages:

- captures complicated relationships,
- fits training data very well.

Disadvantages:

- memorizes noise,
- unstable predictions,
- high variance,
- poor generalization.

Typical result:

```
Training Error

Very Low
```

```
Validation Error

High
```

This is overfitting.

---

# 9.52 Materials Science Example

Suppose we are predicting:

```
Formation Energy
```

using descriptors such as:

- atomic radius,
- electronegativity,
- valence electrons,
- density,
- crystal volume.

Dataset:

```
400 materials
```

Using:

```
max_depth = 2
```

the tree cannot model complex chemical interactions.

Prediction accuracy remains poor.

Using:

```
max_depth = 40
```

the tree memorizes the individual compounds.

Neither extreme is desirable.

Cross-validation is used to determine an appropriate depth.

---

# 9.53 Hyperparameter 2: min_samples_split

The second important hyperparameter is:

```
min_samples_split
```

This determines the minimum number of samples required before a node is allowed to split.

Suppose:

```
min_samples_split = 2
```

Any node containing at least two samples may be divided.

This allows the tree to continue growing aggressively.

---

# 9.54 Increasing min_samples_split

Suppose instead:

```
min_samples_split = 20
```

Now,

a node must contain at least twenty samples before another split is considered.

Small groups remain unchanged.

Advantages:

- simpler trees,
- reduced overfitting,
- better stability.

Disadvantages:

- may stop splitting too early,
- increased bias.

---

# 9.55 Materials Example

Imagine one branch contains:

```
12 materials
```

If:

```
min_samples_split = 2
```

the branch may continue splitting.

If:

```
min_samples_split = 20
```

the branch immediately becomes a leaf.

The resulting tree is much smaller.

---

# 9.56 Hyperparameter 3: min_samples_leaf

Another important hyperparameter is:

```
min_samples_leaf
```

This controls the minimum number of samples allowed inside every leaf node.

Example:

```
min_samples_leaf = 1
```

Each leaf may contain a single training example.

This allows perfect memorization.

---

# 9.57 Increasing min_samples_leaf

Suppose:

```
min_samples_leaf = 10
```

Now,

every leaf must contain at least ten samples.

The tree cannot create extremely small regions.

Advantages:

- smoother predictions,
- improved robustness,
- reduced sensitivity to noise.

Disadvantages:

- less flexible,
- increased bias.

---

# 9.58 Materials Perspective

Experimental materials data often contain:

- measurement uncertainty,
- synthesis variability,
- experimental noise.

Allowing leaves with only one sample may cause the tree to memorize these imperfections.

Increasing:

```
min_samples_leaf
```

helps the model learn broader chemical trends instead of individual observations.

---

# 9.59 Hyperparameter 4: max_leaf_nodes

This hyperparameter limits the total number of terminal nodes.

Example:

```
max_leaf_nodes = 25
```

The tree can contain at most twenty-five leaf nodes.

Even if further splitting would reduce training error,

the algorithm stops once the limit is reached.

---

# 9.60 Why Limit the Number of Leaves?

A tree with thousands of leaves becomes extremely specialized.

Restricting the number of leaf nodes:

- reduces complexity,
- improves interpretability,
- often increases generalization performance.

This hyperparameter provides another way to control overfitting.

---

# 9.61 Hyperparameter 5: criterion

The hyperparameter:

```
criterion
```

determines how the algorithm evaluates possible splits.

For regression trees,

common options include:

```
"squared_error"
```

and

```
"absolute_error"
```

For classification trees,

common choices include:

```
"gini"
```

and

```
"entropy"
```

Different criteria measure node quality in different ways.

The algorithm selects the split that produces the greatest improvement according to the chosen criterion.

---

# 9.62 Regression Criteria

For regression,

the default criterion is usually:

```
Squared Error
```

This attempts to minimize the variance within each leaf.

Another option:

```
Absolute Error
```

is generally more robust to extreme outliers but can require more computation.

The choice depends on the nature of the dataset.

---

# 9.63 Classification Criteria

For classification,

the two most common criteria are:

```
Gini Impurity
```

and

```
Entropy
```

Both aim to create purer child nodes.

In practice,

their predictive performance is often similar,

although training speed may differ slightly.

---

# 9.64 Hyperparameter 6: splitter

The final major Decision Tree hyperparameter is:

```
splitter
```

This determines how candidate splits are selected.

Common options are:

```
"best"
```

and

```
"random"
```

---

# 9.65 Best Split

When:

```
splitter = "best"
```

the algorithm evaluates many possible split locations and chooses the one producing the greatest improvement.

Advantages:

- usually highest predictive accuracy,
- deterministic when the random seed is fixed.

Disadvantages:

- slightly slower.

---

# 9.66 Random Split

When:

```
splitter = "random"
```

the algorithm considers random candidate splits instead of exhaustively searching for the optimal one.

Advantages:

- faster,
- introduces randomness,
- useful in certain ensemble methods.

Disadvantages:

- individual trees may be less accurate.

---

# 9.67 Interaction Between Hyperparameters

Decision Tree hyperparameters should never be viewed independently.

For example,

consider the following configuration:

```
max_depth = 30

min_samples_split = 2

min_samples_leaf = 1
```

This combination allows unrestricted growth and almost guarantees overfitting on a small dataset.

Now consider:

```
max_depth = 6

min_samples_split = 15

min_samples_leaf = 8
```

This tree grows much more conservatively and is likely to generalize better.

The overall behavior of the model depends on the combined effect of all hyperparameters.

---

# 9.68 Choosing Initial Values

Although there is no universal best configuration,

the following values often provide a reasonable starting point for medium-sized regression datasets:

| Hyperparameter | Example Starting Value |
|----------------|-----------------------:|
| `max_depth` | 5–10 |
| `min_samples_split` | 2–10 |
| `min_samples_leaf` | 1–5 |
| `max_leaf_nodes` | None or 20–100 |
| `criterion` | `squared_error` |
| `splitter` | `best` |

These are only initial guesses.

The optimal values should always be determined using cross-validation.

---

# 9.69 Summary

Decision Trees are controlled by several important hyperparameters that determine how the tree grows.

Among them,

`max_depth` has the greatest influence on model complexity,

while `min_samples_split`, `min_samples_leaf`, and `max_leaf_nodes` help prevent excessive growth.

The `criterion` determines how split quality is measured,

and the `splitter` controls how candidate splits are selected.

Together,

these hyperparameters define the balance between:

- flexibility,
- interpretability,
- computational efficiency,
- generalization.

In the next section,

we will study **Random Forest hyperparameters** and see how they extend the ideas introduced for a single Decision Tree into an ensemble of hundreds of trees.

# 9.70 Random Forest Hyperparameters

In the previous section, we studied the hyperparameters of a single Decision Tree.

A Random Forest is fundamentally different.

Instead of constructing one large tree,

it builds **many Decision Trees** and combines their predictions.

This ensemble approach greatly improves:

- prediction accuracy,
- robustness,
- resistance to overfitting,
- generalization performance.

However,

because a Random Forest consists of many trees rather than one,

it introduces additional hyperparameters that control both:

- the individual trees,
- the forest as a whole.

Understanding these hyperparameters is essential for building high-performance Random Forest models.

---

# 9.71 How a Random Forest Is Constructed

A Random Forest follows three major steps.

First,

multiple bootstrap datasets are generated.

Second,

a separate Decision Tree is trained on each bootstrap dataset.

Third,

the predictions from all trees are combined.

For regression:

```
Average Prediction
```

For classification:

```
Majority Vote
```

Conceptually:

```
Training Dataset

↓

Bootstrap Sampling

↓

Tree 1

Tree 2

Tree 3

...

Tree N

↓

Combine Predictions

↓

Final Prediction
```

Every important hyperparameter influences one or more parts of this process.

---

# 9.72 Categories of Random Forest Hyperparameters

Random Forest hyperparameters can be divided into two groups.

### Forest-Level Hyperparameters

These control the entire forest.

Examples:

- number of trees,
- bootstrap sampling,
- feature sampling.

---

### Tree-Level Hyperparameters

These control the individual Decision Trees.

Examples:

- maximum depth,
- minimum samples per split,
- minimum samples per leaf.

The tree-level hyperparameters are inherited directly from the Decision Tree algorithm.

---

# 9.73 Hyperparameter 1: n_estimators

The most important Random Forest hyperparameter is:

```
n_estimators
```

This specifies the number of Decision Trees inside the forest.

Example:

```
n_estimators = 100
```

means the Random Forest contains one hundred Decision Trees.

---

# 9.74 Why Multiple Trees Improve Performance

Individual Decision Trees are unstable.

A small change in the training data can produce a completely different tree.

Random Forest reduces this instability by averaging many independent trees.

Imagine asking:

```
One Expert
```

versus

```
One Hundred Experts
```

The average opinion of many experts is generally more reliable than relying on a single individual.

Random Forest applies exactly the same principle.

---

# 9.75 Effect of Small n_estimators

Suppose:

```
n_estimators = 5
```

Advantages:

- fast training,
- low memory usage.

Disadvantages:

- predictions remain unstable,
- higher variance,
- less accurate ensemble.

The forest has not yet benefited fully from averaging.

---

# 9.76 Effect of Large n_estimators

Suppose:

```
n_estimators = 500
```

Advantages:

- lower variance,
- more stable predictions,
- smoother decision boundaries.

Disadvantages:

- increased computation,
- greater memory consumption,
- diminishing performance improvements after many trees.

Unlike Decision Trees,

adding more trees usually **does not increase overfitting**.

Instead,

it mainly increases computational cost.

---

# 9.77 Choosing n_estimators

For modern computers,

common values include:

```
100

200

300

500
```

Increasing beyond several hundred trees often produces only small improvements.

Therefore,

the optimal value depends on:

- dataset size,
- available memory,
- training time requirements.

---

# 9.78 Materials Science Example

Suppose we wish to predict:

```
Bulk Modulus
```

using:

```
2000 materials
```

generated from DFT.

Results might appear as:

| Trees | Validation RMSE |
|--------|----------------:|
| 10 | 0.34 |
| 50 | 0.25 |
| 100 | 0.21 |
| 300 | 0.19 |
| 500 | 0.19 |

Notice that after several hundred trees,

performance stabilizes.

Adding additional trees increases computational cost without significant gains.

---

# 9.79 Hyperparameter 2: max_features

One of the defining characteristics of Random Forest is:

```
Random Feature Selection
```

This behavior is controlled by:

```
max_features
```

At every split,

the algorithm does **not** examine every feature.

Instead,

it randomly selects a subset of features.

The best split is chosen only from that subset.

---

# 9.80 Why Random Feature Selection?

Suppose one feature is extremely informative.

If every tree always considers every feature,

most trees become nearly identical.

Identical trees provide little benefit when averaged.

Random feature selection forces different trees to explore different parts of the feature space.

This increases diversity,

which improves ensemble performance.

---

# 9.81 Example of max_features

Suppose we have:

```
100 descriptors
```

generated using Pymatgen.

If:

```
max_features = 10
```

each split considers only ten randomly selected descriptors.

Another tree may consider a completely different subset.

Consequently,

every tree develops a unique structure.

---

# 9.82 Common Choices for max_features

Typical settings include:

```
sqrt(number_of_features)
```

```
log2(number_of_features)
```

or

```
all features
```

The optimal choice depends on:

- feature dimensionality,
- feature correlation,
- dataset size.

---

# 9.83 Materials Perspective

Materials datasets often contain highly correlated descriptors.

For example:

- atomic radius,
- covalent radius,
- ionic radius.

These descriptors carry similar information.

Random feature selection prevents every tree from repeatedly using the same correlated variables.

This increases model robustness.

---

# 9.84 Hyperparameter 3: bootstrap

Random Forest introduces another important hyperparameter:

```
bootstrap
```

This determines whether bootstrap sampling is used.

When:

```
bootstrap = True
```

each tree is trained on a randomly sampled dataset generated **with replacement**.

Some samples appear multiple times.

Others are omitted.

---

# 9.85 What Does "With Replacement" Mean?

Suppose the original dataset contains:

```
Material A

Material B

Material C

Material D
```

A bootstrap sample might become:

```
Material A

Material B

Material B

Material D
```

Notice that:

- Material B appears twice.
- Material C is absent.

Every tree therefore receives a slightly different training dataset.

---

# 9.86 Why Bootstrap Sampling Works

Different datasets produce different trees.

Different trees produce different prediction errors.

Averaging these diverse predictions reduces variance.

This is one of the major reasons Random Forest performs so well.

---

# 9.87 Hyperparameter 4: max_depth

Random Forest trees also possess:

```
max_depth
```

just like individual Decision Trees.

However,

because predictions are averaged across many trees,

Random Forest can safely use somewhat deeper trees than a single Decision Tree.

Nevertheless,

very deep trees can still increase computational cost and occasionally contribute to overfitting.

---

# 9.88 Hyperparameter 5: min_samples_split

This hyperparameter has the same meaning as in Decision Trees.

It specifies the minimum number of samples required before another split may occur.

Larger values:

- simplify trees,
- reduce variance,
- increase bias.

Smaller values:

- produce more complex trees,
- reduce bias,
- increase variance.

---

# 9.89 Hyperparameter 6: min_samples_leaf

Again,

this hyperparameter behaves exactly as it does in Decision Trees.

Increasing:

```
min_samples_leaf
```

prevents tiny leaf nodes,

leading to smoother predictions.

This is especially useful for noisy experimental materials datasets.

---

# 9.90 Hyperparameter 7: random_state

Random Forest contains many random operations.

Examples include:

- bootstrap sampling,
- feature selection,
- tie breaking.

Without fixing the random seed,

running the same program twice may produce slightly different models.

Example:

```python
random_state = 42
```

ensures reproducible results.

Scientific research should always report the random seed whenever possible.

---

# 9.91 Hyperparameter 8: n_jobs

Random Forest is highly parallelizable.

The hyperparameter:

```
n_jobs
```

controls how many CPU cores are used during training.

Examples:

```
n_jobs = 1
```

Use one CPU core.

```
n_jobs = -1
```

Use all available CPU cores.

This can significantly reduce training time for large forests.

---

# 9.92 Hyperparameter Interactions

Random Forest hyperparameters influence one another.

For example,

consider:

```
n_estimators = 500

max_depth = 3
```

The forest contains many trees,

but each tree is extremely simple.

Now consider:

```
n_estimators = 20

max_depth = 30
```

Each tree is highly complex,

but there are very few of them.

Neither configuration is universally optimal.

Good performance requires balancing multiple hyperparameters simultaneously.

---

# 9.93 Typical Starting Values

A reasonable starting configuration for many regression problems is:

| Hyperparameter | Example Value |
|----------------|--------------:|
| `n_estimators` | 200 |
| `max_depth` | None or 10–20 |
| `max_features` | `"sqrt"` |
| `bootstrap` | `True` |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `random_state` | 42 |
| `n_jobs` | -1 |

These values should be refined through cross-validation.

---

# 9.94 Materials Informatics Example

Suppose we are predicting:

```
Band Gap
```

using:

- composition descriptors,
- structural descriptors,
- electronic descriptors generated by Pymatgen.

The dataset contains:

```
5000 materials
```

A Random Forest with carefully tuned hyperparameters may:

- outperform a single Decision Tree,
- resist overfitting,
- provide stable predictions,
- rank feature importance effectively.

This makes Random Forest an excellent baseline algorithm for many materials informatics studies.

---

# 9.95 Summary

Random Forest extends the Decision Tree by combining many independently trained trees into a single ensemble.

Its most important hyperparameters include:

- `n_estimators`,
- `max_features`,
- `bootstrap`,
- `max_depth`,
- `min_samples_split`,
- `min_samples_leaf`,
- `random_state`,
- `n_jobs`.

Together,

these hyperparameters determine:

- model complexity,
- prediction stability,
- computational cost,
- generalization performance.

Although Random Forest is generally less sensitive to hyperparameter choices than Gradient Boosting or XGBoost,

careful tuning can still produce significant improvements.

In the next section,

we will study the hyperparameters of **Gradient Boosting**, where model performance becomes much more sensitive to hyperparameter selection because trees are trained **sequentially rather than independently**.

# 9.96 Gradient Boosting Hyperparameters

In the previous section, we studied the hyperparameters of Random Forest.

Although Random Forest and Gradient Boosting are both ensemble methods built from Decision Trees, they learn in fundamentally different ways.

A Random Forest trains many trees **independently**.

Each tree learns without considering the mistakes made by the others.

Gradient Boosting follows a completely different philosophy.

Instead of building independent trees, it builds trees **sequentially**.

Each new tree is trained specifically to correct the mistakes made by the previous trees.

Because of this sequential learning process, Gradient Boosting is generally **more sensitive to hyperparameter selection** than Random Forest.

A poorly chosen hyperparameter can significantly reduce prediction accuracy or lead to severe overfitting.

---

# 9.97 How Gradient Boosting Learns

Recall the basic Gradient Boosting workflow.

```
Training Data

↓

Tree 1

↓

Calculate Errors

↓

Tree 2 Learns Errors

↓

Calculate Remaining Errors

↓

Tree 3 Learns Remaining Errors

↓

...

↓

Final Prediction
```

Unlike Random Forest,

every new tree depends on the trees built before it.

This dependency is the reason why Gradient Boosting requires careful tuning.

---

# 9.98 The Most Important Gradient Boosting Hyperparameters

The most commonly optimized hyperparameters include:

- `learning_rate`
- `n_estimators`
- `max_depth`
- `subsample`
- `min_samples_split`
- `min_samples_leaf`

Together,

these determine:

- learning speed,
- model complexity,
- resistance to overfitting,
- computational cost.

---

# 9.99 Hyperparameter 1: learning_rate

The most important Gradient Boosting hyperparameter is:

```
learning_rate
```

It controls how much each newly added tree contributes to the final prediction.

Every new tree attempts to correct previous errors.

However,

instead of applying the full correction,

Gradient Boosting scales the correction by the learning rate.

A smaller learning rate means:

```
Small Corrections

↓

Slow Learning
```

A larger learning rate means:

```
Large Corrections

↓

Fast Learning
```

---

# 9.100 Why Is Learning Rate Important?

Imagine climbing down a mountain toward the lowest point.

Large steps allow rapid movement.

Unfortunately,

very large steps may cause you to overshoot the minimum.

Small steps require more time,

but they provide much finer control.

Gradient Boosting behaves in exactly the same way.

The learning rate determines the size of each correction step.

---

# 9.101 Small Learning Rate

Suppose:

```
learning_rate = 0.01
```

Advantages:

- stable learning,
- reduced overfitting,
- smoother optimization,
- often better generalization.

Disadvantages:

- requires many trees,
- longer training time.

The model learns slowly,

but usually produces high-quality predictions.

---

# 9.102 Large Learning Rate

Suppose:

```
learning_rate = 0.5
```

Advantages:

- faster convergence,
- fewer trees required.

Disadvantages:

- unstable learning,
- higher risk of overfitting,
- poorer generalization.

Large learning rates can cause the model to fit noise rather than meaningful relationships.

---

# 9.103 Typical Learning Rate Values

Common values include:

```
0.3

0.1

0.05

0.01
```

Many practical applications begin with:

```
0.1
```

and adjust based on cross-validation performance.

---

# 9.104 Relationship Between learning_rate and n_estimators

One of the most important concepts in Gradient Boosting is that these two hyperparameters work together.

Suppose:

```
learning_rate = 0.1

n_estimators = 100
```

Now reduce the learning rate:

```
learning_rate = 0.01
```

To achieve similar learning capacity,

the number of trees often needs to increase:

```
n_estimators = 1000
```

This tradeoff is fundamental.

A smaller learning rate generally requires more trees.

---

# 9.105 Hyperparameter 2: n_estimators

This hyperparameter specifies the total number of boosting stages.

Each stage adds one new Decision Tree.

Example:

```
n_estimators = 300
```

means that three hundred trees are trained sequentially.

Unlike Random Forest,

these trees are **not independent**.

Every tree attempts to improve the predictions of the previous ensemble.

---

# 9.106 Too Few Trees

Suppose:

```
n_estimators = 20
```

The model may stop learning before important relationships are captured.

Typical result:

```
High Bias

↓

Underfitting
```

---

# 9.107 Too Many Trees

Suppose:

```
n_estimators = 3000
```

If the learning rate is not sufficiently small,

the model may eventually memorize the training data.

Typical result:

```
High Variance

↓

Overfitting
```

Choosing the appropriate number of trees is therefore essential.

---

# 9.108 Hyperparameter 3: max_depth

Each boosting stage contains a Decision Tree.

These trees also have:

```
max_depth
```

Unlike ordinary Decision Trees,

Gradient Boosting usually performs best with **shallow trees**.

Common values include:

```
2

3

4

5
```

The idea is simple.

Instead of one extremely powerful tree,

Gradient Boosting combines many weak trees.

Together,

they form a strong predictive model.

---

# 9.109 Why Use Shallow Trees?

Suppose each tree has:

```
Depth = 20
```

The individual trees become extremely complex.

Each tree may already overfit the data.

Boosting these complex trees often worsens overfitting.

Using shallow trees encourages gradual learning and better generalization.

---

# 9.110 Hyperparameter 4: subsample

Gradient Boosting also introduces:

```
subsample
```

This determines the fraction of the training data used for each boosting stage.

Example:

```
subsample = 1.0
```

Every tree uses the entire training dataset.

Now consider:

```
subsample = 0.7
```

Each tree is trained using a random 70% of the training data.

---

# 9.111 Why Use Subsampling?

Using different subsets introduces randomness.

This reduces correlation between trees and helps prevent overfitting.

Advantages:

- improved robustness,
- reduced variance,
- better generalization.

Disadvantages:

- if the value is too small,
  the model may not learn enough information.

---

# 9.112 Hyperparameter 5: min_samples_split

This hyperparameter behaves exactly as it does in Decision Trees.

It specifies the minimum number of samples required before a node may split.

Larger values:

- produce simpler trees,
- reduce variance,
- increase bias.

Smaller values:

- produce more complex trees,
- reduce bias,
- increase variance.

---

# 9.113 Hyperparameter 6: min_samples_leaf

Again,

this hyperparameter controls the smallest number of samples allowed in any leaf node.

Increasing:

```
min_samples_leaf
```

prevents tiny leaves,

which usually improves stability,

especially for noisy materials datasets.

---

# 9.114 Hyperparameter Interactions

Gradient Boosting hyperparameters should never be optimized independently.

Consider two configurations.

Configuration A:

```
learning_rate = 0.5

n_estimators = 50
```

Training is fast,

but overfitting risk is high.

Configuration B:

```
learning_rate = 0.05

n_estimators = 500
```

Training is slower,

but predictions are often more accurate.

These combinations illustrate why hyperparameter optimization must consider multiple variables simultaneously.

---

# 9.115 Materials Science Example

Suppose we wish to predict:

```
Elastic Modulus
```

using:

- composition descriptors,
- structural descriptors,
- electronic descriptors.

Dataset:

```
1500 DFT calculations
```

If:

```
learning_rate = 0.5
```

the model may memorize individual compounds.

If:

```
learning_rate = 0.05

n_estimators = 500
```

the model learns more gradually,

capturing broader physical trends rather than isolated observations.

---

# 9.116 Typical Starting Values

For many regression problems,

a reasonable starting configuration is:

| Hyperparameter | Example Value |
|----------------|--------------:|
| `learning_rate` | 0.1 |
| `n_estimators` | 100–300 |
| `max_depth` | 3 |
| `subsample` | 0.8–1.0 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1–5 |

These values serve only as initial estimates.

Cross-validation should always determine the final configuration.

---

# 9.117 Why Gradient Boosting Requires More Care Than Random Forest

Random Forest benefits from averaging many independent trees.

This naturally reduces variance.

Gradient Boosting,

however,

builds trees sequentially.

Every mistake made early in training influences later trees.

As a result,

Gradient Boosting is generally more sensitive to hyperparameter choices.

Small changes in:

- learning rate,
- tree depth,
- number of estimators,

can produce significant differences in model performance.

---

# 9.118 Summary

Gradient Boosting constructs an ensemble of Decision Trees sequentially,

with each new tree correcting the errors made by previous trees.

Its most important hyperparameters include:

- `learning_rate`,
- `n_estimators`,
- `max_depth`,
- `subsample`,
- `min_samples_split`,
- `min_samples_leaf`.

Among these,

the interaction between **learning rate** and **number of estimators** is especially important.

A smaller learning rate generally improves generalization but requires more trees,

whereas a larger learning rate speeds up training but increases the risk of overfitting.

These concepts form the foundation for understanding **XGBoost**, which extends Gradient Boosting with additional optimization techniques, regularization methods, and advanced hyperparameters.

In the next section, we will study the extensive set of hyperparameters that make **XGBoost** one of the most powerful machine learning algorithms in modern materials informatics.

# 9.119 XGBoost Hyperparameters

In the previous section, we studied the hyperparameters of Gradient Boosting.

If Gradient Boosting were simply made faster, it would already be a useful algorithm.

However,

XGBoost goes much further.

It introduces:

- regularization,
- efficient tree construction,
- advanced sampling techniques,
- parallel computation,
- sophisticated optimization algorithms.

These improvements make XGBoost one of the most powerful machine learning algorithms for structured data.

For this reason,

XGBoost has become one of the most widely used algorithms in:

- materials informatics,
- computational chemistry,
- finance,
- medicine,
- engineering,
- Kaggle competitions.

Its impressive performance comes at a cost.

XGBoost has **many hyperparameters**.

Learning what each one controls is essential for building accurate and reliable models.

---

# 9.120 Categories of XGBoost Hyperparameters

Instead of memorizing a long list of hyperparameters, it is better to organize them into logical groups.

Most XGBoost hyperparameters fall into five categories.

## Tree Complexity

These determine how complicated each Decision Tree is.

Examples:

- `max_depth`
- `min_child_weight`
- `gamma`

---

## Learning Control

These determine how boosting progresses.

Examples:

- `learning_rate`
- `n_estimators`

---

## Sampling

These determine how much data and how many features are used during training.

Examples:

- `subsample`
- `colsample_bytree`

---

## Regularization

These reduce overfitting.

Examples:

- `reg_alpha`
- `reg_lambda`

---

## Objective and Evaluation

These specify:

- prediction task,
- optimization objective,
- evaluation metric.

Examples:

- `objective`
- `eval_metric`

---

# 9.121 Hyperparameter 1: learning_rate (eta)

The most influential XGBoost hyperparameter is:

```
learning_rate
```

also called:

```
eta
```

This determines how strongly each new tree influences the existing model.

Suppose a newly trained tree predicts a correction of:

```
+0.8
```

If:

```
learning_rate = 1
```

the full correction is applied.

If:

```
learning_rate = 0.1
```

only:

```
0.08
```

is added.

Therefore,

smaller learning rates make the model learn more gradually.

---

# 9.122 Choosing the Learning Rate

Typical values include:

```
0.3

0.1

0.05

0.01
```

General guidelines:

| Learning Rate | Characteristics |
|---------------|----------------|
| Large | Faster training, higher overfitting risk |
| Small | Slower learning, better generalization |

A common starting point is:

```
0.1
```

---

# 9.123 Hyperparameter 2: n_estimators

This specifies the total number of boosting rounds.

Each boosting round adds one new Decision Tree.

Example:

```
n_estimators = 500
```

means:

```
500 sequential trees
```

will be trained.

This hyperparameter works closely with the learning rate.

---

# 9.124 Interaction Between learning_rate and n_estimators

Suppose:

```
learning_rate = 0.3

n_estimators = 100
```

Training proceeds quickly.

Now reduce the learning rate:

```
learning_rate = 0.03
```

To achieve similar learning capacity,

we usually increase:

```
n_estimators

↓

1000
```

This relationship is one of the most important principles in XGBoost.

---

# 9.125 Hyperparameter 3: max_depth

Every boosting stage contains a Decision Tree.

The hyperparameter:

```
max_depth
```

limits the depth of each tree.

Small values produce:

- simpler trees,
- higher bias,
- lower variance.

Large values produce:

- more flexible trees,
- lower bias,
- higher variance.

Typical values:

```
3

4

5

6

8
```

Very deep trees are rarely necessary for most structured datasets.

---

# 9.126 Materials Science Example

Suppose we predict:

```
Band Gap
```

using:

- atomic descriptors,
- structural descriptors,
- electronic descriptors.

Dataset:

```
2500 materials
```

If:

```
max_depth = 2
```

important nonlinear chemical interactions may be missed.

If:

```
max_depth = 20
```

the model may memorize individual compounds rather than learning general chemistry.

Cross-validation determines the appropriate balance.

---

# 9.127 Hyperparameter 4: min_child_weight

One hyperparameter unique to XGBoost is:

```
min_child_weight
```

Despite its name,

it does **not** refer to physical weight.

Instead,

it specifies the minimum amount of information required before another split is allowed.

Large values:

- discourage unnecessary splitting,
- reduce overfitting.

Small values:

- permit complex trees,
- increase flexibility.

This hyperparameter is particularly useful for noisy datasets.

---

# 9.128 Hyperparameter 5: gamma

Another uniquely important XGBoost hyperparameter is:

```
gamma
```

Gamma specifies the **minimum improvement in the loss function** required before creating another split.

Imagine a possible split improves prediction only slightly.

If:

```
gamma = 0
```

the split is accepted.

If:

```
gamma = 5
```

the same split may be rejected because the improvement is too small.

Increasing gamma therefore produces:

- smaller trees,
- simpler models,
- better resistance to overfitting.

---

# 9.129 Hyperparameter 6: subsample

This hyperparameter specifies the fraction of training samples used for each boosting stage.

Example:

```
subsample = 0.8
```

means:

```
80%

of the training data
```

is randomly selected for each tree.

Advantages:

- reduced variance,
- improved generalization,
- faster training.

Typical values:

```
0.6

0.7

0.8

1.0
```

---

# 9.130 Hyperparameter 7: colsample_bytree

XGBoost also allows random feature sampling.

The hyperparameter:

```
colsample_bytree
```

specifies the fraction of input features available when constructing each tree.

Suppose we generated:

```
200 descriptors
```

using Pymatgen.

If:

```
colsample_bytree = 0.5
```

each tree receives only:

```
100 randomly selected descriptors
```

This increases diversity and reduces feature dominance.

---

# 9.131 Why Feature Sampling Helps

Many materials descriptors are strongly correlated.

Examples include:

- atomic radius,
- covalent radius,
- metallic radius.

Allowing every tree to use every descriptor often produces highly similar trees.

Random feature sampling forces different trees to discover different predictive relationships.

This improves ensemble diversity.

---

# 9.132 Hyperparameter 8: reg_alpha (L1 Regularization)

XGBoost introduces explicit regularization.

The first regularization hyperparameter is:

```
reg_alpha
```

This implements:

```
L1 Regularization
```

L1 regularization encourages simpler models by reducing the influence of less important tree weights.

Large values increase sparsity,

causing unnecessary contributions to shrink toward zero.

This can improve interpretability and reduce overfitting.

---

# 9.133 Hyperparameter 9: reg_lambda (L2 Regularization)

The second regularization hyperparameter is:

```
reg_lambda
```

This implements:

```
L2 Regularization
```

Unlike L1,

L2 does not force values exactly to zero.

Instead,

it gradually reduces excessively large weights.

Advantages include:

- smoother predictions,
- reduced overfitting,
- improved stability.

In practice,

L2 regularization is almost always active in XGBoost.

---

# 9.134 Hyperparameter 10: objective

The hyperparameter:

```
objective
```

defines the prediction task.

Examples include:

Regression:

```
reg:squarederror
```

Binary Classification:

```
binary:logistic
```

Multi-class Classification:

```
multi:softprob
```

Choosing the correct objective is essential because it determines the loss function optimized during training.

---

# 9.135 Hyperparameter 11: eval_metric

During training,

XGBoost monitors model performance using:

```
eval_metric
```

Regression examples:

- RMSE
- MAE

Classification examples:

- Log Loss
- Error Rate
- AUC

The evaluation metric does not change how the model learns,

but it determines how performance is reported and monitored during optimization.

---

# 9.136 Hyperparameter 12: tree_method

XGBoost supports several algorithms for building Decision Trees.

Examples include:

```
exact
```

and

```
hist
```

The histogram method:

```
tree_method = "hist"
```

is widely used because it provides:

- much faster training,
- lower memory usage,
- nearly identical prediction accuracy for most practical datasets.

It is especially useful for medium and large materials datasets.

---

# 9.137 Hyperparameter 13: early_stopping_rounds

Sometimes,

training continues long after the validation performance has stopped improving.

This wastes computation and may increase overfitting.

The hyperparameter:

```
early_stopping_rounds
```

solves this problem.

Example:

```
early_stopping_rounds = 20
```

If validation performance fails to improve for twenty consecutive boosting rounds,

training automatically stops.

This often produces a better final model while reducing computational cost.

---

# 9.138 Interaction Between XGBoost Hyperparameters

XGBoost hyperparameters interact strongly with one another.

Consider two configurations.

Configuration A:

```
learning_rate = 0.3

max_depth = 10

gamma = 0

subsample = 1
```

This model learns aggressively and may overfit.

---

Configuration B:

```
learning_rate = 0.05

max_depth = 4

gamma = 2

subsample = 0.8
```

This model learns more conservatively and often generalizes better.

Hyperparameter optimization therefore searches for the **best combination**, not the best individual value.

---

# 9.139 Typical Starting Values

A practical starting configuration for many regression problems is:

| Hyperparameter | Example Value |
|----------------|--------------:|
| `learning_rate` | 0.1 |
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `min_child_weight` | 1 |
| `gamma` | 0 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |
| `objective` | `reg:squarederror` |
| `tree_method` | `hist` |

These values are only initial guesses and should always be optimized using validation data.

---

# 9.140 Summary

XGBoost contains a rich collection of hyperparameters that control:

- learning speed,
- tree complexity,
- sampling strategy,
- regularization,
- optimization objective.

Among them,

the most influential are:

- `learning_rate`,
- `n_estimators`,
- `max_depth`,
- `gamma`,
- `min_child_weight`,
- `subsample`,
- `colsample_bytree`.

Unlike simpler algorithms,

XGBoost derives much of its power from the careful interaction of these hyperparameters.

For this reason,

manual tuning is rarely sufficient.

In the next section, we will begin exploring **hyperparameter optimization strategies**, starting with the simplest approach: **Manual Search**, before progressing to **Grid Search**, **Random Search**, and **Bayesian Optimization**.

# 9.141 Manual Hyperparameter Search

Now that we understand the most important hyperparameters of Decision Trees, Random Forests, Gradient Boosting, and XGBoost, the next question naturally arises:

> **How do we choose the best values?**

One possible approach is the simplest one imaginable.

We manually try different values, observe the results, and keep the combination that performs best.

This approach is called:

```
Manual Hyperparameter Search
```

Although modern machine learning provides sophisticated optimization algorithms, every practitioner should first understand manual search because it illustrates the fundamental idea behind hyperparameter optimization.

---

# 9.142 The Basic Idea

Manual search follows a simple cycle.

```
Choose Hyperparameters

↓

Train Model

↓

Evaluate Performance

↓

Modify Hyperparameters

↓

Train Again

↓

Compare Results

↓

Repeat
```

The researcher uses experience and intuition to decide which hyperparameter should be changed next.

---

# 9.143 A Materials Science Example

Suppose we wish to predict:

```
Formation Energy
```

using an XGBoost model.

We begin with a simple configuration.

| Hyperparameter | Value |
|---------------|------:|
| learning_rate | 0.1 |
| max_depth | 6 |
| n_estimators | 100 |

After five-fold cross-validation, we obtain:

```
RMSE

=

0.21 eV/atom
```

The result is acceptable,

but perhaps it can be improved.

---

# 9.144 First Manual Adjustment

Suppose we suspect that the model is underfitting.

One possible solution is to increase the tree depth.

We modify:

```
max_depth

6

↓

8
```

Everything else remains unchanged.

After training again,

the validation result becomes:

```
RMSE

=

0.18 eV/atom
```

The prediction improved.

---

# 9.145 Second Manual Adjustment

Now we wonder whether adding more trees will help.

We increase:

```
n_estimators

100

↓

300
```

The new validation score becomes:

```
RMSE

=

0.16 eV/atom
```

Again,

the model improves.

---

# 9.146 Third Manual Adjustment

Encouraged by these improvements,

we decide to increase the depth even further.

```
max_depth

8

↓

15
```

After training:

```
Training RMSE

=

0.01
```

```
Validation RMSE

=

0.27
```

The model now performs much worse on unseen data.

It has begun to overfit.

---

# 9.147 What Did We Learn?

Through only a few experiments,

we discovered three important ideas.

- Increasing tree depth initially improved performance.
- Adding more trees also helped.
- Excessively deep trees caused overfitting.

Manual experimentation allowed us to understand how the model behaves.

---

# 9.148 Why Beginners Should Learn Manual Search

Although manual search is not the most efficient optimization technique,

it teaches valuable intuition.

After experimenting with several models,

you begin to recognize patterns.

For example,

you may observe:

```
Large max_depth

↓

Overfitting
```

or

```
Small learning_rate

↓

Needs More Trees
```

This intuition becomes extremely useful when using automated optimization algorithms later.

---

# 9.149 Advantages of Manual Search

Manual search offers several benefits.

## Easy to Understand

No additional algorithms are required.

The researcher simply changes one or more hyperparameters and retrains the model.

---

## Good for Learning

Because every change is made intentionally,

it becomes easier to understand how individual hyperparameters influence prediction performance.

---

## Useful for Small Problems

If only one or two hyperparameters require adjustment,

manual tuning may be perfectly adequate.

---

## No Extra Software

Only the machine learning library itself is needed.

No optimization framework is required.

---

# 9.150 Disadvantages of Manual Search

Despite its simplicity,

manual search has significant limitations.

---

## Time Consuming

Suppose we wish to optimize:

```
learning_rate
```

Possible values:

```
0.3

0.1

0.05

0.01
```

Now consider:

```
max_depth
```

Possible values:

```
3

4

5

6

8

10
```

Already,

we have:

```
4 × 6 = 24
```

possible combinations.

Adding only one more hyperparameter causes the number of experiments to increase rapidly.

---

## Depends on Human Experience

A beginner may not know:

- which hyperparameter to modify,
- which direction to move,
- when to stop searching.

Experienced researchers often perform better manual searches simply because they possess greater intuition.

---

## Easily Misses Good Solutions

Suppose the true optimum is:

```
learning_rate = 0.08
```

If we only test:

```
0.3

0.1

0.05
```

we never discover the better value.

Manual search explores only the combinations selected by the researcher.

---

## Difficult for High-Dimensional Optimization

Modern XGBoost models often contain more than ten important hyperparameters.

Exploring every combination manually quickly becomes impossible.

---

# 9.151 The Curse of Dimensionality

Imagine optimizing only six hyperparameters.

Each hyperparameter has five possible values.

The total number of combinations is:

```
5 × 5 × 5 × 5 × 5 × 5

=

15,625
```

Testing more than fifteen thousand models manually is clearly impractical.

This rapid growth in search space is known as the **curse of dimensionality**.

As the number of hyperparameters increases, manual tuning becomes increasingly inefficient.

---

# 9.152 Manual Search in Materials Informatics

Materials informatics often presents additional challenges.

Suppose we are working with:

- expensive DFT data,
- five-fold cross-validation,
- XGBoost.

If one model requires:

```
5 minutes
```

to train,

then testing:

```
100 configurations
```

requires:

```
500 minutes

≈

8 hours
```

Testing thousands of configurations manually may require several days.

Clearly,

more efficient optimization strategies are needed.

---

# 9.153 When Manual Search Is Still Useful

Despite its limitations,

manual tuning remains valuable in several situations.

### Exploratory Research

Researchers often begin with manual experiments to understand the behavior of a new dataset.

---

### Educational Purposes

Students develop intuition by observing how individual hyperparameters affect model performance.

---

### Small Models

For algorithms with very few hyperparameters,

manual optimization may be entirely sufficient.

---

### Fine-Tuning

After automated optimization,

researchers sometimes make small manual adjustments based on domain knowledge.

---

# 9.154 Manual Search vs Trial-and-Error

Some beginners assume manual search means randomly changing numbers.

Good manual search is much more systematic.

A typical workflow is:

```
Choose Baseline Model

↓

Modify One Hyperparameter

↓

Evaluate

↓

Record Results

↓

Compare

↓

Repeat
```

Only one or two hyperparameters should be changed at a time.

Otherwise,

it becomes difficult to determine which change caused the improvement.

---

# 9.155 Recording Experiments

A common practice in machine learning research is to maintain a tuning table.

Example:

| Experiment | learning_rate | max_depth | n_estimators | RMSE |
|------------|--------------:|----------:|-------------:|-----:|
| 1 | 0.10 | 6 | 100 | 0.21 |
| 2 | 0.10 | 8 | 100 | 0.18 |
| 3 | 0.10 | 8 | 300 | 0.16 |
| 4 | 0.10 | 15 | 300 | 0.27 |

Such records help identify useful trends and prevent repeating unsuccessful experiments.

---

# 9.156 Why Manual Search Inspired Automated Methods

Manual search demonstrates the fundamental objective of hyperparameter optimization:

```
Try

↓

Evaluate

↓

Compare

↓

Improve
```

Modern optimization algorithms follow exactly the same philosophy.

The difference is that they automate the search process.

Instead of relying on human intuition,

they use mathematical strategies to explore the hyperparameter space more efficiently.

---

# 9.157 Summary

Manual hyperparameter search is the simplest optimization strategy.

It involves selecting hyperparameter values based on experience, training the model, evaluating performance, and repeating the process until satisfactory results are obtained.

Its greatest strengths are:

- simplicity,
- educational value,
- intuitive understanding of model behavior.

Its greatest weaknesses are:

- slow exploration,
- dependence on human expertise,
- inability to efficiently search large hyperparameter spaces.

For modern machine learning models such as XGBoost,

manual tuning is usually only the starting point.

More systematic approaches are required.

The first of these approaches is **Grid Search**, where the computer automatically evaluates every specified hyperparameter combination.

In the next section, we will study Grid Search in detail and learn why it became one of the earliest standardized methods for hyperparameter optimization.

# 9.158 Grid Search

Manual hyperparameter tuning helps us understand how machine learning models behave.

However, as the number of hyperparameters increases, manual experimentation quickly becomes impractical.

Suppose we wish to optimize only three hyperparameters of an XGBoost model:

- `learning_rate`
- `max_depth`
- `n_estimators`

If each hyperparameter has five possible values, then the total number of combinations becomes

```
5 × 5 × 5 = 125
```

Testing 125 different models manually would be tedious and error-prone.

Fortunately, we can automate this process.

The simplest automated hyperparameter optimization method is called

```
Grid Search
```

Grid Search systematically evaluates every possible combination of predefined hyperparameter values and selects the one that produces the best validation performance.

---

# 9.159 The Fundamental Idea

Imagine that we want to tune only two hyperparameters.

```
max_depth

=

{3, 5, 7}
```

```
learning_rate

=

{0.01, 0.05, 0.1}
```

Instead of choosing combinations manually,

Grid Search tests **every possible pair**.

```
max_depth = 3
    learning_rate = 0.01
    learning_rate = 0.05
    learning_rate = 0.10

max_depth = 5
    learning_rate = 0.01
    learning_rate = 0.05
    learning_rate = 0.10

max_depth = 7
    learning_rate = 0.01
    learning_rate = 0.05
    learning_rate = 0.10
```

Total models trained:

```
3 × 3 = 9
```

After evaluating all nine models,

Grid Search simply selects the one with the highest validation performance.

---

# 9.160 Why Is It Called a Grid?

Suppose we arrange the hyperparameter values in a table.

| max_depth ↓ / learning_rate → | 0.01 | 0.05 | 0.10 |
|------------------------------|------|------|------|
| 3 | ✓ | ✓ | ✓ |
| 5 | ✓ | ✓ | ✓ |
| 7 | ✓ | ✓ | ✓ |

Every cell in the table corresponds to one experiment.

The table forms a grid,

which explains the name:

```
Grid Search
```

---

# 9.161 Step-by-Step Workflow

Grid Search follows a straightforward algorithm.

```
Choose Hyperparameter Values

↓

Generate Every Combination

↓

Train Model

↓

Evaluate Using Cross Validation

↓

Store Validation Score

↓

Repeat

↓

Choose Best Combination
```

Unlike manual tuning,

the researcher does not need to decide which combination to test next.

The computer performs the search automatically.

---

# 9.162 Mathematical View

Suppose we have

```
n
```

hyperparameters.

Each hyperparameter has

```
k
```

candidate values.

The total number of models trained becomes

```
kⁿ
```

This exponential growth is one of the major limitations of Grid Search.

For example,

| Hyperparameters | Values Each | Total Models |
|-----------------|------------:|-------------:|
| 2 | 5 | 25 |
| 3 | 5 | 125 |
| 4 | 5 | 625 |
| 5 | 5 | 3,125 |
| 6 | 5 | 15,625 |

Even a modest increase in the number of hyperparameters causes the computational cost to grow rapidly.

---

# 9.163 Materials Informatics Example

Suppose we wish to optimize an XGBoost model for predicting

```
Formation Energy
```

Our search space is

| Hyperparameter | Candidate Values |
|---------------|------------------|
| learning_rate | 0.01, 0.05, 0.10 |
| max_depth | 4, 6, 8 |
| n_estimators | 100, 300, 500 |

The total number of combinations is

```
3 × 3 × 3 = 27
```

If we use five-fold cross-validation,

each combination requires training five models.

Therefore,

the total number of model fits becomes

```
27 × 5 = 135
```

Even this relatively small search already requires significant computation.

---

# 9.164 Why Cross-Validation Is Used

Grid Search almost always works together with cross-validation.

For each hyperparameter combination,

the model is trained multiple times using different training-validation splits.

For example,

using five-fold cross-validation,

the workflow becomes

```
Hyperparameter Combination

↓

Fold 1

↓

Fold 2

↓

Fold 3

↓

Fold 4

↓

Fold 5

↓

Average Validation Score
```

The average score provides a much more reliable estimate of model performance than a single train-test split.

---

# 9.165 Example Using Scikit-learn

Scikit-learn provides the `GridSearchCV` class, which automates both Grid Search and cross-validation.

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "n_estimators": [100, 300, 500]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)
```

In this example,

Scikit-learn automatically

- generates every hyperparameter combination,
- performs five-fold cross-validation,
- evaluates each model,
- identifies the best configuration.

---

# 9.166 Obtaining the Best Hyperparameters

After training,

the optimal hyperparameters can be retrieved easily.

```python
print(grid.best_params_)
```

Example output:

```python
{
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 300
}
```

Similarly,

the best validation score is available through

```python
print(grid.best_score_)
```

---

# 9.167 Advantages of Grid Search

Grid Search has several important strengths.

## Simple

The algorithm is easy to understand and easy to implement.

Every candidate combination is evaluated systematically.

---

## Reproducible

Given the same search space,

Grid Search always evaluates the same combinations.

This makes scientific experiments highly reproducible.

---

## Exhaustive

If the optimal hyperparameter values exist within the predefined grid,

Grid Search is guaranteed to find them.

This exhaustive nature is one of its greatest strengths.

---

## Easy to Parallelize

Each hyperparameter combination is independent.

Therefore,

Grid Search can distribute different combinations across multiple CPU cores,

greatly reducing execution time.

---

# 9.168 Limitations of Grid Search

Despite its popularity,

Grid Search suffers from several important weaknesses.

---

## Computationally Expensive

The largest disadvantage is computational cost.

Suppose we tune

- six hyperparameters,
- each with ten candidate values.

The total number of combinations becomes

```
10⁶

=

1,000,000
```

Training one million models is unrealistic for most research projects.

---

## Many Experiments Are Wasted

Imagine the true optimum is

```
learning_rate = 0.08
```

Our grid contains

```
0.05

0.10
```

Grid Search never evaluates

```
0.08
```

Instead,

it spends equal effort evaluating many poor combinations.

---

## Uniform Search Is Inefficient

Grid Search assumes that every hyperparameter is equally important.

In reality,

some hyperparameters strongly influence performance,

while others have only a minor effect.

Grid Search does not distinguish between them.

---

## Poor Scalability

As the number of hyperparameters grows,

the search space expands exponentially.

This phenomenon is known as the **curse of dimensionality**.

---

# 9.169 Materials Science Perspective

Materials datasets often contain

- expensive DFT calculations,
- limited experimental measurements,
- high-dimensional descriptors generated by Pymatgen.

Training a single XGBoost model may already require considerable time.

If Grid Search evaluates hundreds or thousands of hyperparameter combinations,

the total computational cost becomes substantial.

Consequently,

many modern materials informatics studies use Grid Search only for relatively small search spaces or as an initial optimization step.

---

# 9.170 When Should Grid Search Be Used?

Grid Search is particularly useful when

- only a few hyperparameters need optimization,
- the search space is relatively small,
- computational resources are sufficient,
- reproducibility is important.

For larger problems,

more efficient optimization methods are often preferred.

---

# 9.171 Grid Search vs Manual Search

| Manual Search | Grid Search |
|--------------|-------------|
| User chooses combinations | Computer generates combinations |
| May overlook good values | Tests every predefined combination |
| Depends on experience | Systematic and reproducible |
| Suitable for very small problems | Suitable for small to medium search spaces |

Grid Search removes much of the human effort while retaining a straightforward and interpretable optimization process.

---

# 9.172 Summary

Grid Search is the first widely used automated hyperparameter optimization technique.

It evaluates every possible combination within a predefined search space and selects the configuration that achieves the best validation performance.

Its major strengths are

- simplicity,
- reproducibility,
- exhaustive exploration of the specified grid.

However,

its computational cost grows exponentially with the number of hyperparameters, making it inefficient for large search spaces.

This limitation motivates the next optimization strategy.

Instead of testing **every** combination,

Random Search samples only a subset of combinations intelligently.

Surprisingly,

despite evaluating far fewer models,

Random Search often produces results that are as good as—or even better than—Grid Search.

In the next section, we will examine why this happens and how Random Search has become one of the most widely used hyperparameter optimization methods in practical machine learning.

# 9.173 Random Search

In the previous section, we learned that Grid Search evaluates **every possible hyperparameter combination**.

This guarantees that the best combination within the predefined grid will be found.

However,

Grid Search also suffers from a serious limitation.

As the number of hyperparameters increases,

the computational cost grows exponentially.

For many modern machine learning problems,

especially XGBoost,

evaluating every possible combination is simply impossible.

To overcome this limitation,

researchers developed a much more efficient approach:

```
Random Search
```

At first glance,

Random Search seems surprising.

Instead of evaluating every possible hyperparameter combination,

it simply samples a limited number of combinations **at random**.

Despite appearing simplistic,

Random Search often performs as well as—or even better than—Grid Search while requiring far fewer model evaluations.

---

# 9.174 The Basic Idea

Suppose we wish to optimize three hyperparameters.

```
learning_rate

=

{0.01, 0.05, 0.10}
```

```
max_depth

=

{4, 6, 8}
```

```
n_estimators

=

{100, 300, 500}
```

Grid Search evaluates

```
3 × 3 × 3 = 27
```

combinations.

Random Search,

however,

may decide to evaluate only

```
10
```

randomly selected combinations.

For example,

it might test

```
(0.05, 6, 300)

↓

(0.10, 8, 500)

↓

(0.01, 4, 100)

↓

...
```

until the desired number of experiments has been completed.

---

# 9.175 Visualizing the Difference

Imagine the same hyperparameter grid.

Grid Search examines every cell.

```
✓ ✓ ✓

✓ ✓ ✓

✓ ✓ ✓
```

Random Search visits only a few randomly chosen cells.

```
✓ ○ ○

○ ✓ ○

○ ○ ✓
```

The unexplored combinations are ignored.

Surprisingly,

this often has very little impact on the final model quality.

---

# 9.176 Why Does Random Search Work?

At first,

Random Search appears inefficient.

After all,

how can testing fewer combinations produce comparable results?

The answer lies in an important observation made by researchers.

**Not all hyperparameters are equally important.**

For many machine learning algorithms,

only a few hyperparameters strongly influence performance.

Others have relatively minor effects.

Grid Search wastes enormous computational effort exploring unimportant hyperparameters.

Random Search spends more effort exploring different values of the important ones.

---

# 9.177 An Intuitive Analogy

Imagine searching for buried treasure on a large beach.

Grid Search walks systematically through every square meter.

```
□□□□□□□□

□□□□□□□□

□□□□□□□□
```

Random Search instead visits randomly scattered locations.

```
□■□□□□□

□□□□■□□

□□□■□□□
```

If the treasure occupies a reasonably large region,

Random Search often finds it much sooner.

Similarly,

if good hyperparameter values occupy a reasonably large region of the search space,

Random Search can discover them with far fewer experiments.

---

# 9.178 Theoretical Insight

In 2012,

James Bergstra and Yoshua Bengio published a highly influential paper showing that Random Search often outperforms Grid Search for high-dimensional optimization problems.

The key conclusion was that

> when only a small number of hyperparameters strongly affect performance,

Random Search explores those important dimensions much more efficiently.

This finding fundamentally changed the way researchers perform hyperparameter optimization.

Today,

Random Search is considered a standard baseline optimization technique.

---

# 9.179 Example

Suppose we want to optimize four hyperparameters.

| Hyperparameter | Candidate Values |
|---------------|------------------|
| learning_rate | 5 values |
| max_depth | 5 values |
| gamma | 5 values |
| subsample | 5 values |

Grid Search requires

```
5⁴

=

625
```

combinations.

Random Search may evaluate only

```
50
```

random combinations.

This reduces computational cost by

```
625

↓

50
```

while often achieving nearly identical predictive performance.

---

# 9.180 Continuous Hyperparameter Spaces

One major advantage of Random Search is that it does not require predefined grids.

Instead,

hyperparameters may be sampled from probability distributions.

For example,

rather than choosing

```
learning_rate

=

0.01

0.05

0.10
```

we may sample

```
0.034

0.071

0.118

0.052

...
```

This allows Random Search to explore values that Grid Search would never consider.

---

# 9.181 Materials Science Example

Suppose we are predicting

```
Band Gap
```

using XGBoost.

Instead of defining only three learning rates,

we specify

```
learning_rate

Uniformly Sampled

Between

0.01

and

0.30
```

Every iteration selects a different value.

The algorithm therefore explores a much richer portion of the hyperparameter space.

---

# 9.182 Example Using Scikit-learn

Scikit-learn provides the `RandomizedSearchCV` class.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
from xgboost import XGBRegressor

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

param_dist = {
    "learning_rate": uniform(0.01, 0.29),
    "max_depth": randint(3, 10),
    "n_estimators": randint(100, 600)
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
```

Unlike Grid Search,

we specify

```
n_iter = 50
```

meaning that only fifty random combinations will be evaluated.

---

# 9.183 Choosing the Number of Iterations

The parameter

```
n_iter
```

controls how many random hyperparameter combinations are evaluated.

Small values

```
10

20

30
```

provide rapid exploration.

Larger values

```
100

200

500
```

provide a more thorough search.

Increasing

```
n_iter
```

improves the probability of finding excellent hyperparameters,

although computational cost also increases.

---

# 9.184 Advantages of Random Search

Random Search offers several important advantages.

## Computational Efficiency

Only a limited number of combinations are evaluated.

This dramatically reduces training time.

---

## Better Exploration

Random Search investigates many different regions of the hyperparameter space.

Grid Search,

in contrast,

may repeatedly evaluate similar combinations.

---

## Continuous Hyperparameters

Hyperparameters can be sampled from mathematical distributions,

allowing much finer exploration than fixed grids.

---

## Scales Well

As the number of hyperparameters increases,

Random Search remains practical,

whereas Grid Search rapidly becomes computationally infeasible.

---

# 9.185 Limitations of Random Search

Despite its strengths,

Random Search also has limitations.

---

## No Guarantee of Finding the Best Solution

Because only a subset of combinations is evaluated,

the true optimum may never be sampled.

---

## Results Depend on Random Sampling

Different random seeds produce different search paths.

Two independent Random Search experiments may identify different hyperparameter combinations.

---

## Inefficient After Many Evaluations

Random Search does not learn from previous experiments.

Even after evaluating hundreds of models,

future hyperparameter choices remain completely random.

Later optimization algorithms overcome this limitation.

---

# 9.186 Grid Search vs Random Search

| Grid Search | Random Search |
|-------------|---------------|
| Evaluates every combination | Evaluates random combinations |
| Computationally expensive | Computationally efficient |
| Uses predefined grids | Can sample continuous distributions |
| Exhaustive within the grid | Non-exhaustive |
| Suitable for small search spaces | Suitable for medium and large search spaces |

For modern machine learning,

Random Search is often preferred when the search space contains many hyperparameters.

---

# 9.187 Materials Informatics Perspective

Materials informatics frequently involves

- expensive DFT calculations,
- high-dimensional descriptor sets,
- limited computational resources.

Suppose each model requires

```
15 minutes
```

to train.

Grid Search evaluating

```
1000
```

combinations would require

```
15,000 minutes

≈

250 hours
```

Random Search evaluating only

```
100
```

carefully sampled combinations requires

```
25 hours
```

while often achieving nearly the same prediction quality.

This dramatic reduction in computational cost makes Random Search highly attractive for materials informatics.

---

# 9.188 When Should Random Search Be Used?

Random Search is particularly appropriate when

- many hyperparameters require optimization,
- computational resources are limited,
- hyperparameters are continuous,
- approximate optimal solutions are acceptable.

For these reasons,

Random Search has become one of the most widely used hyperparameter optimization techniques in practical machine learning.

---

# 9.189 Summary

Random Search improves upon Grid Search by evaluating only a randomly selected subset of hyperparameter combinations.

Although it explores far fewer models,

it often discovers equally good—or even better—solutions because it spends more effort exploring the most important hyperparameters.

Its major strengths include

- computational efficiency,
- scalability,
- support for continuous hyperparameter distributions.

However,

Random Search still treats every experiment independently.

It does not learn from previous evaluations.

This limitation motivates the next and most sophisticated optimization strategy.

Instead of selecting hyperparameters randomly,

**Bayesian Optimization** uses the results of previous experiments to intelligently choose the next hyperparameter combination, dramatically reducing the number of expensive model evaluations required to locate high-performing solutions.

In the next section, we will study Bayesian Optimization in detail and understand why it has become one of the preferred optimization methods for computationally expensive machine learning problems such as XGBoost applied to materials informatics.

# 9.190 Bayesian Optimization

Grid Search systematically evaluates every predefined hyperparameter combination.

Random Search evaluates randomly selected combinations.

Both methods have one important limitation:

> **Neither method learns from previous experiments.**

Suppose Random Search has already evaluated one hundred XGBoost models.

When selecting the one hundred and first hyperparameter combination,

it completely ignores everything learned from the previous one hundred experiments.

This seems inefficient.

As humans,

we naturally use previous experience to guide future decisions.

If a certain region of the hyperparameter space consistently produces good models,

we tend to explore nearby regions.

Bayesian Optimization follows exactly this principle.

Instead of searching blindly,

it **learns from previous experiments** and intelligently predicts where the next promising hyperparameter combination is likely to be found.

For expensive machine learning models,

this strategy can dramatically reduce the number of required evaluations.

---

# 9.191 The Central Idea

Imagine searching for the highest mountain in an unknown landscape.

Grid Search walks across every square meter.

Random Search jumps to random locations.

Bayesian Optimization behaves differently.

It first climbs a few hills,

observes the terrain,

and then predicts where the tallest mountain is most likely located.

Each new observation improves its understanding of the landscape.

Hyperparameter optimization works in exactly the same way.

---

# 9.192 Learning From Experience

Suppose we have already evaluated four hyperparameter combinations.

| Experiment | learning_rate | max_depth | RMSE |
|------------|--------------:|----------:|-----:|
| 1 | 0.30 | 3 | 0.24 |
| 2 | 0.10 | 6 | 0.18 |
| 3 | 0.05 | 8 | 0.17 |
| 4 | 0.01 | 10 | 0.23 |

Instead of choosing the fifth experiment randomly,

Bayesian Optimization analyzes these results.

It may conclude:

- small learning rates appear promising,
- medium tree depths perform well,
- extremely deep trees provide little improvement.

The next experiment is chosen based on these observations.

---

# 9.193 Two Main Components

Bayesian Optimization consists of two major parts.

```
Surrogate Model

↓

Predicts Performance
```

and

```
Acquisition Function

↓

Chooses Next Experiment
```

These two components work together throughout the optimization process.

---

# 9.194 What Is a Surrogate Model?

Training an XGBoost model may require

- several seconds,
- several minutes,
- or even several hours,

depending on the dataset.

Instead of repeatedly training expensive models,

Bayesian Optimization constructs a much simpler mathematical model called a

```
Surrogate Model
```

The surrogate approximates the relationship between

```
Hyperparameters

↓

Prediction Performance
```

Because the surrogate model is inexpensive to evaluate,

it can quickly estimate which hyperparameter combinations are likely to perform well.

---

# 9.195 A Simple Analogy

Suppose you wish to estimate the elevation of an entire mountain range.

Measuring every point directly is expensive.

Instead,

you measure only a limited number of locations,

then fit a smooth surface through those observations.

This smooth surface becomes your approximation of the landscape.

Similarly,

Bayesian Optimization approximates the relationship between

```
Hyperparameters

↓

Validation Score
```

without evaluating every possible model.

---

# 9.196 Common Surrogate Models

Several mathematical models can serve as surrogates.

The most common include

- Gaussian Processes,
- Tree-structured Parzen Estimators (TPE),
- Random Forest regressors.

Among these,

Gaussian Processes are traditionally associated with Bayesian Optimization because they provide both

- predicted performance,
- prediction uncertainty.

This uncertainty plays an important role during optimization.

---

# 9.197 Why Prediction Uncertainty Matters

Suppose the surrogate predicts that one region of the search space has an RMSE of

```
0.18
```

with high confidence.

Another region is predicted to have an RMSE of

```
0.17
```

but the uncertainty is much larger.

Should we explore the uncertain region?

Perhaps.

The prediction may actually be much better than expected.

Bayesian Optimization explicitly considers both

- predicted quality,
- uncertainty.

This balance is one of its greatest strengths.

---

# 9.198 Exploration vs Exploitation

Bayesian Optimization constantly balances two competing objectives.

## Exploitation

Search near regions that already perform well.

```
Known Good Region

↓

Search Nearby
```

---

## Exploration

Investigate regions that remain highly uncertain.

```
Unknown Region

↓

Gather Information
```

Neither strategy alone is sufficient.

The optimizer must intelligently alternate between them.

---

# 9.199 Acquisition Function

The component responsible for balancing exploration and exploitation is called the

```
Acquisition Function
```

The acquisition function examines the surrogate model and decides

```
Which Hyperparameter Combination

↓

Should Be Evaluated Next?
```

Importantly,

the acquisition function is inexpensive to compute.

Only after selecting a promising candidate is the expensive machine learning model trained.

---

# 9.200 Optimization Workflow

The complete Bayesian Optimization workflow can be summarized as

```
Choose Initial Hyperparameters

↓

Train Machine Learning Model

↓

Measure Validation Score

↓

Update Surrogate Model

↓

Use Acquisition Function

↓

Select Next Hyperparameters

↓

Train Again

↓

Repeat
```

Each iteration improves the optimizer's understanding of the hyperparameter landscape.

---

# 9.201 Materials Science Example

Suppose we wish to optimize an XGBoost model predicting

```
Formation Energy
```

using

- composition descriptors,
- structural descriptors,
- electronic descriptors.

Training one model requires

```
20 minutes
```

Grid Search evaluates

```
500

models

↓

166 hours
```

Random Search evaluates

```
100

models

↓

33 hours
```

Bayesian Optimization may discover an equally good solution after evaluating only

```
30

models

↓

10 hours
```

For expensive DFT-based machine learning,

this reduction is extremely valuable.

---

# 9.202 Example Using Optuna

One of the most popular Bayesian optimization libraries is

```
Optuna
```

A simple optimization objective might appear as

```python
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def objective(trial):

    model = XGBRegressor(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        objective="reg:squarederror",
        random_state=42
    )

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    return -score.mean()
```

The optimizer repeatedly calls this objective function,

learning from every completed trial.

---

# 9.203 Running the Optimization

Once the objective function is defined,

optimization is straightforward.

```python
study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=50)
```

After fifty intelligently selected trials,

the best hyperparameters can be obtained using

```python
print(study.best_params)
```

Unlike Random Search,

each new trial depends on everything learned from previous trials.

---

# 9.204 Advantages of Bayesian Optimization

Bayesian Optimization offers several important advantages.

## Efficient

It usually requires far fewer model evaluations than Grid Search or Random Search.

---

## Learns From Previous Experiments

Every completed experiment improves future hyperparameter selection.

---

## Ideal for Expensive Models

When training requires substantial computation,

Bayesian Optimization often provides enormous time savings.

---

## Handles Continuous Hyperparameters Naturally

Continuous search spaces are explored efficiently without predefined grids.

---

# 9.205 Limitations of Bayesian Optimization

Despite its strengths,

Bayesian Optimization also has limitations.

---

## Greater Complexity

The underlying mathematics is considerably more sophisticated than Grid Search or Random Search.

---

## Additional Overhead

Constructing and updating the surrogate model introduces computational overhead.

For inexpensive machine learning models,

this overhead may outweigh the benefits.

---

## Scalability Challenges

Traditional Gaussian Process Bayesian Optimization becomes computationally demanding for very high-dimensional optimization problems.

Modern libraries address many of these issues,

but they remain important considerations.

---

# 9.206 Comparison of Optimization Methods

| Method | Learns From Previous Trials | Computational Cost | Search Strategy |
|---------|----------------------------|-------------------|-----------------|
| Manual Search | No | Low to Moderate | Human intuition |
| Grid Search | No | Very High | Exhaustive |
| Random Search | No | Moderate | Random sampling |
| Bayesian Optimization | Yes | Low to Moderate | Intelligent sequential search |

The key distinction is that Bayesian Optimization continuously improves its search strategy using accumulated knowledge.

---

# 9.207 Materials Informatics Perspective

Modern materials informatics often combines

- expensive DFT simulations,
- thousands of descriptors,
- complex machine learning models.

Every additional model evaluation may require considerable computational resources.

Bayesian Optimization minimizes unnecessary experiments,

making it particularly attractive for

- alloy design,
- battery materials,
- catalyst discovery,
- semiconductor prediction,
- high-entropy alloy optimization.

As computational cost increases,

the value of intelligent hyperparameter optimization increases as well.

---

# 9.208 When Should Bayesian Optimization Be Used?

Bayesian Optimization is particularly appropriate when

- model training is computationally expensive,
- only a limited number of evaluations is affordable,
- high predictive accuracy is desired,
- continuous hyperparameter optimization is required.

For many modern XGBoost applications,

Bayesian Optimization has become one of the preferred optimization strategies.

---

# 9.209 Summary

Bayesian Optimization represents one of the most sophisticated hyperparameter optimization techniques available.

Unlike Grid Search and Random Search,

it learns from previous experiments by constructing a surrogate model of the hyperparameter space and using an acquisition function to intelligently select future experiments.

Its greatest strength lies in dramatically reducing the number of expensive model evaluations required to locate high-performing hyperparameter combinations.

For computationally demanding problems—particularly those encountered in materials informatics—Bayesian Optimization often provides the best balance between computational efficiency and predictive performance.

With this discussion, we have completed the four major hyperparameter optimization strategies:

- Manual Search,
- Grid Search,
- Random Search,
- Bayesian Optimization.

In the next section, we will shift from optimization algorithms to **practical hyperparameter tuning workflows**, learning how experienced machine learning researchers systematically tune XGBoost models on real-world materials datasets instead of adjusting hyperparameters randomly.

# 9.210 A Practical Hyperparameter Tuning Workflow

So far in this chapter, we have learned

- what hyperparameters are,
- why they matter,
- the most important hyperparameters for tree-based algorithms,
- four major hyperparameter optimization techniques.

However,

a practical question still remains.

> **When faced with a completely new dataset, where should we begin?**

Many beginners open the XGBoost documentation,

see dozens of hyperparameters,

and immediately become overwhelmed.

Should they tune

- `learning_rate`,
- `gamma`,
- `max_depth`,
- `min_child_weight`,
- `subsample`,
- `colsample_bytree`,
- `reg_alpha`,
- `reg_lambda`

all at the same time?

The answer is

```
No.
```

Experienced machine learning practitioners follow a systematic workflow.

Instead of optimizing everything simultaneously,

they tune the most influential hyperparameters first,

then gradually refine the remaining ones.

This structured approach is both more efficient and easier to interpret.

---

# 9.211 Why a Workflow Is Necessary

Imagine trying to adjust every control on a microscope simultaneously.

If the image becomes clearer,

which adjustment actually caused the improvement?

It would be impossible to know.

Hyperparameter tuning presents exactly the same problem.

Changing too many hyperparameters at once makes it difficult to understand which modification improved—or degraded—the model.

A structured workflow avoids this confusion.

---

# 9.212 Step 1: Build a Baseline Model

Every optimization process should begin with a simple baseline model.

The purpose of the baseline is **not** to achieve the highest possible accuracy.

Instead,

it provides a reference point against which all future improvements can be measured.

For example,

an initial XGBoost configuration might be

```python
model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)
```

All remaining hyperparameters are left at their default values.

---

# 9.213 Evaluate the Baseline

The baseline model should always be evaluated using cross-validation.

Suppose five-fold cross-validation produces

```
RMSE

=

0.21 eV/atom
```

This value becomes our benchmark.

Every future modification must be compared against this baseline.

Without a baseline,

it is impossible to determine whether the model is actually improving.

---

# 9.214 Step 2: Tune the Most Important Hyperparameters First

Not every hyperparameter contributes equally to model performance.

Some have a dramatic influence,

while others produce only minor changes.

Experienced practitioners therefore begin with the hyperparameters that control the overall learning process.

For XGBoost,

these typically include

- `learning_rate`,
- `n_estimators`,
- `max_depth`.

These three hyperparameters usually produce the largest performance improvements.

---

# 9.215 Step 3: Optimize Tree Complexity

Once the basic learning behavior has been established,

attention shifts to controlling tree complexity.

Important hyperparameters include

- `max_depth`,
- `min_child_weight`,
- `gamma`.

These determine

- how flexible each tree becomes,
- how easily new splits are created,
- how resistant the model is to overfitting.

At this stage,

cross-validation should be performed after every optimization cycle.

---

# 9.216 Step 4: Optimize Sampling

After determining an appropriate tree structure,

sampling hyperparameters can be optimized.

Examples include

- `subsample`,
- `colsample_bytree`.

These control

- the fraction of training samples,
- the fraction of input features

used during each boosting iteration.

Optimizing these parameters often improves model robustness without substantially increasing complexity.

---

# 9.217 Step 5: Tune Regularization

Regularization should usually be adjusted after the primary learning hyperparameters have been optimized.

Important regularization hyperparameters include

- `reg_alpha`,
- `reg_lambda`.

If cross-validation indicates persistent overfitting,

regularization strength can be increased gradually.

The objective is to reduce variance without introducing excessive bias.

---

# 9.218 Step 6: Fine-Tune the Learning Rate

One of the final optimization stages often involves reducing

```
learning_rate
```

while simultaneously increasing

```
n_estimators
```

For example,

the optimization process may proceed as

```
learning_rate

0.10

↓

0.05

↓

0.03
```

while increasing

```
n_estimators

200

↓

400

↓

800
```

This slower learning process frequently improves generalization.

---

# 9.219 Monitor Training and Validation Performance

Throughout optimization,

both training and validation performance should be monitored.

Three common situations may occur.

### Situation 1

```
Training Error

High

Validation Error

High
```

The model is underfitting.

Possible solutions include

- increasing tree depth,
- increasing the number of trees,
- reducing regularization.

---

### Situation 2

```
Training Error

Low

Validation Error

High
```

The model is overfitting.

Possible solutions include

- reducing tree depth,
- increasing regularization,
- decreasing learning rate,
- increasing subsampling.

---

### Situation 3

```
Training Error

Low

Validation Error

Low
```

The model is generalizing well.

Further optimization may provide only marginal improvements.

---

# 9.220 Record Every Experiment

Professional machine learning research always includes careful record keeping.

Every experiment should document

- hyperparameter values,
- validation score,
- training time,
- observations.

An example experiment log is shown below.

| Experiment | learning_rate | max_depth | gamma | RMSE |
|------------|--------------:|----------:|------:|-----:|
| 1 | 0.10 | 6 | 0 | 0.210 |
| 2 | 0.10 | 8 | 0 | 0.184 |
| 3 | 0.05 | 8 | 0 | 0.171 |
| 4 | 0.05 | 8 | 2 | 0.168 |
| 5 | 0.05 | 6 | 2 | 0.166 |

Keeping detailed records prevents confusion and makes the optimization process reproducible.

---

# 9.221 Materials Science Case Study

Suppose we wish to predict

```
Formation Energy
```

using approximately

```
3,000 DFT calculations
```

generated from the Materials Project database.

A typical tuning workflow might proceed as follows.

### Baseline

```
RMSE

=

0.210
```

---

### Tune Tree Depth

```
RMSE

↓

0.184
```

---

### Tune Learning Rate

```
RMSE

↓

0.171
```

---

### Tune Gamma

```
RMSE

↓

0.168
```

---

### Tune Sampling

```
RMSE

↓

0.166
```

Notice that each stage provides a gradual improvement.

Optimization is usually incremental rather than dramatic.

---

# 9.222 When Should Optimization Stop?

A common beginner question is

> **How do I know when to stop tuning?**

Optimization should stop when

- validation performance no longer improves significantly,
- additional experiments require excessive computational cost,
- improvements become smaller than the expected experimental uncertainty.

Continuing to tune indefinitely often produces diminishing returns.

---

# 9.223 Common Beginner Mistakes

Several mistakes are frequently encountered during hyperparameter tuning.

### Changing Too Many Hyperparameters Simultaneously

This makes it impossible to determine which modification caused the observed change.

---

### Ignoring Cross-Validation

Evaluating performance using only one train-test split may produce misleading conclusions.

---

### Optimizing Only Training Accuracy

The objective is not to minimize training error.

The objective is to maximize performance on unseen data.

---

### Forgetting Reproducibility

Always fix

```python
random_state
```

when comparing hyperparameter configurations.

Otherwise,

performance differences may simply result from random variation.

---

# 9.224 Best Practices

Experienced practitioners generally follow these recommendations.

- Begin with a simple baseline.
- Tune only a few hyperparameters at a time.
- Use cross-validation throughout optimization.
- Record every experiment.
- Optimize validation performance rather than training performance.
- Increase model complexity gradually.
- Prefer simpler models whenever predictive performance is comparable.

These principles apply not only to XGBoost,

but to almost every machine learning algorithm.

---

# 9.225 Summary

Effective hyperparameter optimization is not a random process.

It follows a structured workflow beginning with a baseline model and progressing through increasingly refined stages of optimization.

The most influential hyperparameters are tuned first,

followed by tree complexity,

sampling,

and regularization.

Cross-validation serves as the primary tool for evaluating each modification,

while careful experiment logging ensures reproducibility.

By following this systematic approach,

researchers can build highly accurate and robust machine learning models while avoiding unnecessary experimentation.

With this section,

we have completed the practical aspects of hyperparameter optimization.

In the next and final section of this chapter,

we will summarize the key concepts learned throughout Chapter 9 and present a set of practical recommendations for applying hyperparameter optimization to real-world materials informatics problems using XGBoost.

# 9.226 Chapter Summary

Throughout this chapter, we explored one of the most important aspects of machine learning:

```
Hyperparameter Optimization
```

Even the most sophisticated machine learning algorithm cannot achieve its full potential if its hyperparameters are poorly chosen.

Conversely,

a carefully tuned model can often outperform a much more complex algorithm that has not been optimized.

For this reason,

hyperparameter optimization has become a standard component of nearly every modern machine learning workflow.

In materials informatics,

where datasets are often small and expensive to generate,

careful hyperparameter tuning is particularly important.

---

# 9.227 What We Learned

We began by distinguishing two fundamental concepts.

### Parameters

Parameters are learned automatically during training.

Examples include

- Decision Tree split locations,
- leaf values,
- boosting coefficients.

The user does not specify these values.

The learning algorithm determines them.

---

### Hyperparameters

Hyperparameters are chosen **before training begins**.

They determine how the learning algorithm behaves.

Examples include

- tree depth,
- learning rate,
- number of estimators,
- regularization strength.

Unlike parameters,

hyperparameters must be selected by the practitioner or an optimization algorithm.

---

# 9.228 Hyperparameters Across Tree-Based Algorithms

We then examined the most important hyperparameters for several machine learning algorithms.

For Decision Trees,

we studied

- `max_depth`,
- `min_samples_split`,
- `min_samples_leaf`,
- `max_leaf_nodes`,
- `criterion`.

These hyperparameters determine the complexity of a single Decision Tree.

---

For Random Forest,

we introduced additional hyperparameters such as

- `n_estimators`,
- `max_features`,
- `bootstrap`.

These control how multiple Decision Trees are combined into an ensemble.

---

For Gradient Boosting,

we learned that

- `learning_rate`,
- `n_estimators`,
- `subsample`

become especially important because trees are trained sequentially rather than independently.

---

Finally,

we explored the extensive collection of XGBoost hyperparameters,

including

- `gamma`,
- `min_child_weight`,
- `colsample_bytree`,
- `reg_alpha`,
- `reg_lambda`.

These additional controls allow XGBoost to achieve remarkable predictive performance while maintaining strong resistance to overfitting.

---

# 9.229 Hyperparameter Optimization Methods

After understanding individual hyperparameters,

we examined four major optimization strategies.

### Manual Search

The simplest method.

Researchers manually modify hyperparameters,

train the model,

evaluate performance,

and repeat the process.

Its primary advantage is educational value,

while its major limitation is inefficiency.

---

### Grid Search

Grid Search evaluates every possible combination within a predefined search space.

Advantages:

- systematic,
- reproducible,
- exhaustive.

Disadvantages:

- computationally expensive,
- poor scalability.

---

### Random Search

Random Search evaluates only randomly selected hyperparameter combinations.

Although it explores far fewer models,

it often achieves performance comparable to Grid Search because it allocates more effort to exploring different regions of the search space.

---

### Bayesian Optimization

Bayesian Optimization represents the most sophisticated method studied in this chapter.

Instead of selecting hyperparameters randomly,

it learns from previous experiments using a surrogate model and an acquisition function.

This intelligent search strategy dramatically reduces the number of expensive model evaluations required.

---

# 9.230 Practical Workflow

We also developed a practical hyperparameter tuning workflow suitable for real-world machine learning projects.

Rather than optimizing every hyperparameter simultaneously,

experienced practitioners

1. build a baseline model,
2. tune the most influential hyperparameters,
3. adjust tree complexity,
4. optimize sampling,
5. apply regularization,
6. fine-tune the learning rate.

This systematic approach reduces unnecessary experimentation and improves reproducibility.

---

# 9.231 Key Principles

Several important principles emerged repeatedly throughout this chapter.

### There Is No Universal Best Hyperparameter

The optimal configuration always depends on

- dataset size,
- feature representation,
- prediction task,
- noise level.

A hyperparameter setting that performs exceptionally well for one dataset may perform poorly for another.

---

### Cross-Validation Is Essential

Hyperparameters should never be selected using only the training set.

Cross-validation provides a far more reliable estimate of generalization performance,

particularly for small materials datasets.

---

### Simpler Models Are Often Better

Increasing model complexity does not always improve predictive performance.

A slightly simpler model that generalizes well is usually preferable to a highly complex model that memorizes the training data.

---

### Record Every Experiment

Scientific machine learning requires reproducibility.

Every hyperparameter configuration,

validation score,

random seed,

and dataset version should be documented carefully.

---

# 9.232 Hyperparameter Optimization in Materials Informatics

Materials informatics presents several unique challenges.

Datasets are often

- relatively small,
- computationally expensive,
- high dimensional.

For example,

obtaining one additional DFT calculation may require several hours of computation.

Consequently,

efficient hyperparameter optimization becomes more valuable than in many other machine learning applications.

Bayesian Optimization,

Random Search,

and carefully designed cross-validation procedures are therefore particularly important in materials science research.

---

# 9.233 Looking Ahead

By completing this chapter,

you have developed a solid understanding of

- hyperparameters,
- model complexity,
- regularization,
- automated optimization,
- practical tuning strategies.

These concepts form an essential foundation for building high-performance machine learning models.

However,

successful machine learning involves much more than selecting algorithms and tuning hyperparameters.

Equally important is the quality of the data itself.

As the well-known saying states:

> **"Garbage in, garbage out."**

Even the most carefully optimized XGBoost model cannot compensate for poorly prepared data.

Feature quality,

data cleaning,

descriptor engineering,

and domain knowledge often influence predictive performance more than the choice of algorithm.

---

# 9.234 Chapter Takeaways

After completing this chapter, you should be able to

- distinguish between parameters and hyperparameters,
- explain why hyperparameter optimization is necessary,
- identify the most important hyperparameters of Decision Trees, Random Forests, Gradient Boosting, and XGBoost,
- compare Manual Search, Grid Search, Random Search, and Bayesian Optimization,
- design a systematic hyperparameter tuning workflow,
- recognize the signs of underfitting and overfitting during optimization,
- apply cross-validation correctly during model selection,
- understand why efficient optimization is especially important in materials informatics.

These skills are essential for developing reliable, accurate, and reproducible machine learning models.

---

# 9.235 Exercises

The following exercises are designed to reinforce the concepts introduced in this chapter.

### Conceptual Questions

1. Explain the difference between parameters and hyperparameters.

2. Why can increasing `max_depth` improve performance initially but eventually lead to overfitting?

3. Why is `learning_rate` closely related to `n_estimators` in Gradient Boosting and XGBoost?

4. Compare Grid Search and Random Search.

Under what circumstances would you choose one over the other?

5. Why does Bayesian Optimization generally require fewer model evaluations than Random Search?

---

### Practical Exercises

1. Train an XGBoost regression model using default hyperparameters.

Record its five-fold cross-validation RMSE.

2. Perform Grid Search on

- `learning_rate`,
- `max_depth`,
- `n_estimators`.

Compare the optimized model with the baseline.

3. Repeat the optimization using Random Search.

How many model evaluations were required?

4. Use Optuna to perform Bayesian Optimization.

Compare

- optimization time,
- validation performance,
- best hyperparameters

with the previous methods.

5. Investigate how changing

```
gamma
```

affects model complexity for a materials dataset obtained from the Materials Project.

---

### Materials Informatics Project

Download a materials dataset containing

- composition,
- crystal structure,
- target property.

Using descriptors generated by Pymatgen,

develop an XGBoost model.

Optimize its hyperparameters using Bayesian Optimization,

evaluate the final model using nested cross-validation,

and prepare a short report discussing

- optimization strategy,
- validation procedure,
- predictive performance,
- feature importance,
- limitations of the study.

---

# 9.236 Final Remarks

Hyperparameter optimization bridges the gap between a machine learning algorithm and a high-performing predictive model.

It transforms theoretical algorithms into practical scientific tools capable of solving real-world engineering and materials science problems.

The techniques presented in this chapter—from simple manual tuning to advanced Bayesian Optimization—represent the standard workflow used by researchers and industry practitioners worldwide.

Mastering these concepts will enable you to build more accurate, robust, and scientifically reliable machine learning models, providing a strong foundation for the advanced materials informatics techniques that follow in the subsequent chapters.