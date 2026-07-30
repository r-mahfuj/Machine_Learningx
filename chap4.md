# Chapter 4 — Tree-Based Machine Learning Algorithms

## 4.1 Why Tree-Based Algorithms Were Developed

Machine learning has a simple objective:

> Learn the relationship between input variables and an output variable so that the model can make accurate predictions for unseen data.

Suppose a materials scientist wants to predict the **formation energy** of crystalline compounds. A dataset is collected in which every material is represented by a collection of descriptors generated using **Pymatgen**.

| Material | Average Electronegativity | Density (g/cm³) | Volume (Å³) | Formation Energy (eV/atom) |
|----------|--------------------------:|----------------:|------------:|---------------------------:|
| Material A | 2.10 | 5.42 | 152 | -3.24 |
| Material B | 1.87 | 6.15 | 168 | -2.81 |
| Material C | 2.45 | 4.98 | 141 | -4.02 |

Our goal is to discover a function

$$f(\mathbf{x}) = y$$

where

- $\mathbf{x}$ is the vector of material descriptors, and
- $y$ is the property we want to predict.

At first glance this appears straightforward. We simply collect enough data and fit a mathematical function.

Unfortunately, real materials rarely behave this simply.

---

## 4.2 The Complexity of Materials Data

Consider one of the simplest descriptors:

**Average electronegativity.**

If electronegativity alone determined formation energy, a graph might look like this:

```text
Formation Energy

^

|

|          *

|       *

|    *

| *

+---------------------------->

Electronegativity
```

A straight line could describe this relationship.

However, actual materials datasets almost never resemble this idealized picture.

Formation energy depends simultaneously on dozens—or even hundreds—of interacting variables:

- atomic size,
- oxidation states,
- crystal symmetry,
- coordination environment,
- bond lengths,
- bond angles,
- electronic configuration,
- lattice distortion,
- magnetic ordering,
- pressure,
- temperature.

These variables do not influence the target independently. They interact with one another in highly nonlinear ways.

For example, increasing the atomic radius may decrease the formation energy for one crystal structure but increase it for another. The effect of electronegativity may depend entirely on the coordination number or the oxidation state of neighboring atoms.

In other words,

$$Formation\ Energy \neq f(Electronegativity)$$

Instead,

$$Formation\ Energy = f(x_1,x_2,x_3,\ldots,x_n)$$

where $n$ may easily exceed one hundred.

Learning such a function is considerably more difficult than fitting a straight line.

This complexity is one of the primary reasons why machine learning has become so important in modern computational materials science. Traditional empirical equations often fail to capture these intricate nonlinear relationships, whereas machine learning algorithms can approximate them directly from data.

---

## 4.3 Why Linear Models Become Insufficient

Suppose we attempt to model formation energy using linear regression.

The model assumes

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n.$$

This equation assumes that every descriptor contributes independently and additively to the final prediction.

That assumption is often unrealistic for crystalline materials.

Imagine two compounds:

- **NaCl**
- **MgO**

Both crystallize in the rock-salt structure.

Their geometric arrangements are nearly identical, yet replacing Na with Mg changes the electronic structure, bonding character, lattice energy, elastic response, dielectric behavior, and defect chemistry dramatically.

The relationship between composition and properties is therefore not simply additive. Instead, it involves complex interactions between atomic species and crystal geometry.

Linear regression struggles because it cannot naturally model such interactions unless they are explicitly engineered.

Another important limitation is that linear models assume a single mathematical relationship applies across the entire dataset. Materials datasets, however, often contain multiple families of compounds whose governing physics differ significantly.

For example, the descriptors controlling the band gap of oxide semiconductors may differ from those controlling the band gap of chalcogenides or halide perovskites. Attempting to force a single linear equation to describe all of these systems usually results in poor predictive performance.

---

## 4.4 A Different Way of Thinking

Instead of asking,

> "Can I write one complicated equation describing every material?"

suppose we ask a different question.

> "Can I divide the materials into smaller, simpler groups?"

Imagine predicting whether a material has a high or low band gap.

Rather than fitting one global equation, we might reason like this:

1. Is the average electronegativity greater than 2.3?
2. If yes, examine the average bond length.
3. If no, examine the crystal density.
4. Continue asking increasingly specific questions until the remaining materials are very similar.

This resembles how an experienced materials scientist reasons.

When researchers manually screen candidate materials, they rarely evaluate every possible descriptor simultaneously. Instead, they gradually narrow the search space by applying successive physical criteria.

For example, a researcher searching for solid-state electrolyte materials may first eliminate unstable compounds, then select materials with large electrochemical windows, followed by high ionic conductivity, and finally compare activation energies.

Each decision reduces the complexity of the problem.

Decision trees mimic this reasoning process computationally.

Instead of solving one enormous equation, the algorithm gradually narrows the possibilities through a sequence of decisions.

This observation led to one of the most influential ideas in machine learning:

**Decision Trees.**

---

## 4.5 Learning by Asking Questions

A Decision Tree learns by repeatedly asking questions that divide the dataset into increasingly homogeneous groups.

For example, consider a simplified dataset for band-gap prediction.

```text
                     All Materials
                           │
          Is electronegativity > 2.3?
                 /                     \
              Yes                       No
               │                         │
     Is density > 5.5?         Is volume > 180 Å³?
         /         \              /             \
 High Band Gap  Medium      Low Band Gap   Medium
```

Every question partitions the dataset.

Each partition becomes easier to model than the original dataset.

Instead of fitting one complex nonlinear equation over the entire feature space, the algorithm decomposes the problem into a hierarchy of simpler regions.

An important consequence of this strategy is that each decision focuses only on a subset of the data. As the tree grows deeper, the algorithm gradually specializes. Early decisions capture broad trends, while later decisions capture subtle local relationships that would be extremely difficult to describe using a single analytical equation.

This ability to combine global reasoning with local specialization is one of the major strengths of tree-based algorithms.

It also explains why decision trees can successfully model highly nonlinear materials datasets without requiring complicated feature transformations or polynomial expansions.

This seemingly simple idea forms the foundation not only of Decision Trees but also of Random Forests, Gradient Boosting, XGBoost, and many of the most successful algorithms used in modern materials informatics.

In the next section, we will study how a computer determines the **best possible question** to ask at each node of the tree. Rather than selecting questions randomly, decision trees use mathematical criteria such as entropy, information gain, and Gini impurity to identify the split that produces the greatest improvement in predictive power.

## 4.6 How Does a Decision Tree Decide Which Question to Ask?

At first glance, a decision tree appears almost intelligent.

It repeatedly asks questions such as:

- Is the electronegativity greater than 2.3?
- Is the density less than 5.5 g/cm³?
- Is the average atomic radius greater than 1.4 Å?

But an important question arises.

**How does the algorithm know which question is the best one?**

Does it randomly choose a feature?

Does it test every possible feature?

Does it use chemistry or physics knowledge?

The answer is both simple and elegant.

A decision tree systematically evaluates many possible questions and selects the one that best separates the data according to a mathematical criterion.

To understand this process, we first need to understand the concept of **impurity**.

---

## 4.7 Understanding Purity and Impurity

Imagine you have a basket containing colored balls.

### Basket A

```text
🔴 🔴 🔴 🔴 🔴 🔴
```

Every ball is red.

This basket is perfectly pure.

Now consider another basket.

### Basket B

```text
🔴 🔵 🔴 🔵 🔴 🔵
```

Half the balls are red and half are blue.

This basket is highly mixed.

We describe it as impure.

Decision trees think in exactly the same way.

Instead of colored balls, they examine class labels or target values.

For a classification problem, a node containing only one class is considered pure.

For example,

| Material | Stable? |
|----------|----------|
| A | Yes |
| B | Yes |
| C | Yes |
| D | Yes |

Every sample belongs to the same class.

The node is perfectly pure.

Now consider another node.

| Material | Stable? |
|----------|----------|
| A | Yes |
| B | No |
| C | Yes |
| D | No |

This node contains a mixture of classes.

It is impure.

The objective of a decision tree is therefore remarkably simple:

> At every step, divide the data into groups that are as pure as possible.

---

## 4.8 Why Purity Matters

Suppose we want to classify materials as either

- Stable
- Unstable

Imagine that after applying one decision rule,

```
Density > 5.0 g/cm³
```

the dataset becomes

### Left Node

| Material | Stable? |
|----------|----------|
| A | Yes |
| B | Yes |
| C | Yes |

### Right Node

| Material | Stable? |
|----------|----------|
| D | No |
| E | No |
| F | No |

Each node now contains only one class.

Making predictions becomes trivial.

Every material entering the left node is predicted as stable.

Every material entering the right node is predicted as unstable.

This is exactly what a decision tree wants to achieve.

---

## 4.9 A Poor Split

Now imagine a different question.

```
Average Atomic Radius > 1.25 Å
```

Suppose this produces

### Left Node

| Material | Stable? |
|----------|----------|
| A | Yes |
| B | No |
| C | Yes |

### Right Node

| Material | Stable? |
|----------|----------|
| D | No |
| E | Yes |
| F | No |

Both nodes still contain mixtures of classes.

Although the data have been divided, very little has actually been learned.

The algorithm therefore considers this split much less useful.

---

## 4.10 The Goal of Every Split

At every decision node, the algorithm asks one question:

> "Which split produces the purest child nodes?"

Notice that it does **not** ask

> "Which feature is most important?"

or

> "Which descriptor has the highest correlation?"

Instead, it evaluates one possible split after another and measures how much impurity decreases.

The split producing the largest reduction in impurity is selected.

This process repeats recursively until the tree is complete.

---

## 4.11 Measuring Impurity

Humans can easily recognize whether a group is mixed.

A computer cannot.

It requires a numerical measure.

Several impurity measures have been developed over the years.

The three most important are

1. Entropy
2. Information Gain
3. Gini Impurity

Different decision tree algorithms use different measures.

For example,

- ID3 uses Entropy and Information Gain.
- C4.5 also uses Entropy.
- CART primarily uses Gini Impurity for classification.

Despite using different mathematical formulas, they all pursue exactly the same objective:

**Create the purest possible child nodes.**

---

## 4.12 An Intuitive Example

Imagine a node containing ten materials.

| Class | Number |
|-------|--------:|
| Stable | 5 |
| Unstable | 5 |

This node is highly uncertain.

If you randomly select one material,

there is a 50% chance that it belongs to either class.

Now suppose we split the data.

### Left Child

| Class | Number |
|-------|--------:|
| Stable | 5 |
| Unstable | 0 |

### Right Child

| Class | Number |
|-------|--------:|
| Stable | 0 |
| Unstable | 5 |

The uncertainty disappears completely.

Each child node contains only one class.

This split is therefore considered excellent.

Now compare it with another split.

### Left Child

| Class | Number |
|-------|--------:|
| Stable | 3 |
| Unstable | 2 |

### Right Child

| Class | Number |
|-------|--------:|
| Stable | 2 |
| Unstable | 3 |

Although the data have been divided,

both nodes remain uncertain.

The algorithm naturally prefers the first split.

---

## 4.13 A Materials Science Perspective

Consider a database of oxide materials.

Suppose our objective is to predict whether a compound is thermodynamically stable.

Initially,

```
1000 materials

↓

600 Stable

400 Unstable
```

A candidate split using electronegativity produces

```
Left Node

520 Stable

40 Unstable


Right Node

80 Stable

360 Unstable
```

Both child nodes are now much more homogeneous than the original dataset.

This indicates that average electronegativity is an informative descriptor for this particular problem.

Now suppose another candidate split based on lattice volume produces

```
Left Node

300 Stable

220 Unstable


Right Node

300 Stable

180 Unstable
```

Very little separation has occurred.

Although lattice volume may still contain useful information, it is clearly less effective than electronegativity for this dataset.

A decision tree therefore chooses the first split because it creates a much larger reduction in impurity.

---

## 4.14 From Intuition to Mathematics

We now understand the central objective of a decision tree.

The algorithm repeatedly searches for the split that maximizes node purity.

However, we still have one unanswered question.

**How can purity be measured mathematically?**

To answer this, we need a quantitative measure of uncertainty.

The first and historically most influential measure is **Entropy**, a concept originally developed in information theory by Claude Shannon.

In the next section, we will derive the entropy equation from first principles, understand its physical intuition, compute entropy by hand using real examples, and see why it became one of the foundations of modern decision tree learning.

## 4.15 Entropy: Measuring Uncertainty in Decision Trees

To build an effective decision tree, we need a mathematical way to answer one question:

> **How mixed is a node?**

Earlier, we described a node as either **pure** or **impure**.

Humans can easily recognize this.

For example,

```text
Node A

Stable
Stable
Stable
Stable
```

is obviously pure.

Similarly,

```text
Node B

Stable
Stable
Unstable
Stable
Unstable
```

is clearly mixed.

A computer, however, cannot rely on intuition.

It requires a numerical quantity that measures the amount of uncertainty inside a node.

This quantity is called **Entropy**.

Entropy is one of the most important concepts in decision tree learning and has applications far beyond machine learning, including physics, chemistry, communication systems, cryptography, and information theory.

---

## 4.16 Historical Background of Entropy

The word **entropy** originally comes from thermodynamics.

In the nineteenth century, physicists such as Rudolf Clausius introduced entropy to describe the degree of disorder within a physical system.

Later, in 1948, the American mathematician and engineer **Claude Shannon** introduced a new interpretation of entropy in his groundbreaking work *A Mathematical Theory of Communication*.

Instead of measuring physical disorder, Shannon used entropy to measure **information uncertainty**.

Decision trees adopt Shannon's idea almost directly.

A node containing many different classes has high uncertainty.

A node containing only one class has almost no uncertainty.

Thus,

Higher uncertainty

↓

Higher entropy

Lower uncertainty

↓

Lower entropy

---

## 4.17 Understanding Entropy Through Everyday Examples

Imagine you have a bag containing colored balls.

### Bag 1

```text
🔴 🔴 🔴 🔴 🔴 🔴
```

Every ball is red.

Suppose someone asks,

> "What color will the next ball be?"

The answer is obvious.

There is no uncertainty.

The entropy is therefore very low.

Now consider another bag.

```text
🔴 🔵 🔴 🔵 🔴 🔵
```

Half the balls are red.

Half are blue.

Before drawing a ball, you genuinely do not know which color you will obtain.

The uncertainty is much greater.

Consequently,

the entropy is much higher.

Decision trees treat class labels exactly like these colored balls.

---

## 4.18 Entropy and Machine Learning

Consider a classification problem.

Suppose a node contains

| Class | Number |
|-------|--------:|
| Stable | 10 |
| Unstable | 0 |

Every material belongs to the same class.

Prediction is trivial.

Entropy is zero.

Now consider another node.

| Class | Number |
|-------|--------:|
| Stable | 5 |
| Unstable | 5 |

The classes are perfectly balanced.

Prediction becomes much more difficult.

Entropy reaches its maximum.

This illustrates an important principle.

> The more evenly classes are distributed, the greater the entropy.

---

## 4.19 The Entropy Equation

Shannon showed that the uncertainty of a probability distribution can be measured by

$$H(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

where

- $H(S)$ is the entropy of the dataset,
- $p_i$ is the probability of class $i$,
- $n$ is the number of classes.

The logarithm is taken with base two because information is commonly measured in **bits**.

Although the equation may appear intimidating at first, every term has a simple interpretation.

The probability tells us how likely each class is.

The logarithm measures how much information is gained when that class is observed.

Multiplying them together gives the contribution of each class to the total uncertainty.

Finally, the negative sign ensures that entropy is always a positive quantity.

---

## 4.20 Why Does Entropy Become Zero?

Suppose every material is stable.

Then,

$p(\text{Stable})=1$

and

$p(\text{Unstable})=0$

Substituting into the entropy equation,

$H = -(1)\log_2(1) - 0\log_2(0)$

Since

$$\log_2(1)=0,$$

the first term becomes zero.

The second term is defined mathematically to approach zero because

$$\lim_{p\rightarrow0} p\log_2(p) = 0.$$

Therefore,

$$H=0.$$

A perfectly pure node contains no uncertainty.

---

## 4.21 Maximum Entropy

Now suppose the dataset contains

50% stable materials

and

50% unstable materials.

Then,

$$p_1=0.5,$$

$$p_2=0.5.$$

The entropy becomes

$$H = -0.5\log_2(0.5) - 0.5\log_2(0.5).$$

Since

$$\log_2(0.5) = -1,$$

we obtain

$$H = -0.5(-1) - 0.5(-1) = 1.$$

For a binary classification problem,

the maximum possible entropy is

$$H=1.$$

This represents complete uncertainty.

---

## 4.22 Visualizing Entropy

The relationship between class balance and entropy can be summarized intuitively.

| Stable (%) | Unstable (%) | Entropy |
|------------:|-------------:|---------:|
|100|0|0.000|
|90|10|0.469|
|80|20|0.722|
|70|30|0.881|
|60|40|0.971|
|50|50|1.000|

Notice that entropy increases as the classes become more evenly balanced.

When one class dominates completely,

entropy decreases again.

---

## 4.23 Worked Example

Suppose a node contains twelve materials.

| Property | Number |
|----------|--------:|
| Stable | 9 |
| Unstable | 3 |

The class probabilities are

$$p(\text{Stable}) = \frac{9}{12} = 0.75,$$

$$p(\text{Unstable}) = \frac{3}{12} = 0.25.$$ 

Applying Shannon's equation,

$$H = -0.75\log_2(0.75) - 0.25\log_2(0.25).$$

Using

$$\log_2(0.75) = -0.415,$$

and

$$\log_2(0.25) = -2,$$

we obtain

$$H = -0.75(-0.415) - 0.25(-2).$$

Therefore,

$$H = 0.311 + 0.500 = 0.811.$$ 

The node is neither perfectly pure nor completely uncertain.

Its entropy lies between the two extremes.

---

## 4.24 Interpreting the Result

An entropy value of

$$0.811$$

does **not** mean that the classifier is 81.1% accurate.

This is a common misconception.

Entropy measures only the amount of uncertainty within the node.

Higher entropy means the node contains a greater mixture of classes.

Lower entropy means the node is more homogeneous.

Decision trees seek splits that reduce entropy as much as possible.

---

## 4.25 Entropy in Materials Science

Consider a database containing oxide materials.

Initially,

| Class | Number |
|-------|--------:|
| Stable | 500 |
| Unstable | 500 |

Entropy is close to its maximum.

Now suppose we split the data using average electronegativity.

### Left Child

| Class | Number |
|-------|--------:|
| Stable | 470 |
| Unstable | 40 |

### Right Child

| Class | Number |
|-------|--------:|
| Stable | 30 |
| Unstable | 460 |

Both child nodes are much more homogeneous than the original dataset.

Their entropy is substantially lower.

This indicates that average electronegativity is an informative descriptor for predicting stability.

A decision tree therefore considers this split highly desirable.

---

## 4.26 Strengths and Limitations of Entropy

Entropy has several advantages.

- It has a strong mathematical foundation.
- It measures uncertainty naturally.
- It works well for multiclass problems.
- It forms the basis of several classical decision tree algorithms.

However, entropy also has limitations.

Computing logarithms repeatedly is computationally more expensive than simpler impurity measures such as the Gini index.

For modern computers this difference is usually small, but historically it influenced the development of alternative decision tree algorithms.

---

## 4.27 Preparing for Information Gain

Entropy tells us **how impure a node is**.

But a decision tree is interested in something even more important.

It wants to know

> **How much does a proposed split reduce the entropy?**

A split that barely changes the entropy is not useful.

A split that dramatically decreases entropy is highly valuable.

This reduction in entropy is called **Information Gain**.

Information Gain is the mathematical criterion that allows a decision tree to compare hundreds or even thousands of possible questions and automatically choose the best one.

In the next section, we will derive the Information Gain equation, calculate it step by step using numerical examples, and understand why maximizing Information Gain leads to better decision trees.

## 4.28 Information Gain: How a Decision Tree Chooses the Best Split

Entropy tells us how much uncertainty exists inside a node.

However, knowing the entropy alone is not enough.

Imagine a node with an entropy of

$$H=0.95.$$

A decision tree now has hundreds—or even thousands—of possible questions it could ask.

For example,

- Is the average electronegativity greater than 2.1?
- Is the density less than 5.4 g/cm³?
- Is the average atomic radius greater than 1.25 Å?
- Is the coordination number greater than 6?
- Is the lattice volume greater than 180 Å³?

Which one should it choose?

The answer is simple.

The tree evaluates every candidate question and asks:

> **"Which split removes the largest amount of uncertainty?"**

The mathematical quantity that measures this reduction in uncertainty is called **Information Gain**.

---

## 4.29 What Does "Information" Mean?

The word **information** is often misunderstood.

In machine learning,

information does **not** mean knowledge stored in a database.

Instead,

information measures how much uncertainty disappears after making an observation.

Consider a simple example.

Imagine flipping a fair coin.

Before the coin lands,

you do not know the outcome.

Your uncertainty is high.

Once you observe

```
Heads
```

all uncertainty disappears.

You have gained information.

Decision trees behave similarly.

Before making a split,

the algorithm is uncertain about the class labels.

After making a good split,

the uncertainty decreases.

The amount by which uncertainty decreases is called **Information Gain**.

---

## 4.30 An Everyday Analogy

Imagine you are a doctor diagnosing patients.

Initially,

a patient could have one of several diseases.

Your uncertainty is high.

You ask the patient,

> "Do you have a fever?"

If the answer is **yes**,

you eliminate many possibilities.

Your uncertainty decreases.

Next you ask,

> "Do you have a cough?"

Again,

your uncertainty decreases.

Every useful question increases your knowledge.

Decision trees follow exactly the same strategy.

Each split is simply another carefully chosen question.

---

## 4.31 Mathematical Definition of Information Gain

Information Gain is defined as

$$IG = H(\text{Parent}) - H(\text{Children})$$

More precisely,

$$IG = H(S) - \sum_{i=1}^{n} \frac{|S_i|}{|S|} H(S_i)$$

where

- $H(S)$ is the entropy of the parent node,
- $S_i$ represents each child node,
- $|S_i|$ is the number of samples in child $i$,
- $|S|$ is the number of samples in the parent node.

The second term is called the **weighted entropy** of the children.

The weighting is necessary because larger child nodes should influence the calculation more than very small ones.

---

## 4.32 Why Do We Use Weighted Entropy?

Suppose a parent node contains

100 materials.

A split produces

Left child:

```
95 materials
```

Right child:

```
5 materials
```

Clearly,

the larger child should contribute much more to the final impurity than the smaller child.

If both children were treated equally,

a tiny node containing only a few samples could disproportionately influence the result.

Weighting prevents this problem.

Each child's contribution is proportional to its size.

---

## 4.33 Step-by-Step Numerical Example

Suppose a parent node contains

20 materials.

| Class | Number |
|-------|--------:|
| Stable | 10 |
| Unstable | 10 |

Since both classes are equally likely,

the parent entropy is

$$H(\text{Parent})=1.$$

Now consider a possible split.

### Left Child

| Class | Number |
|-------|--------:|
| Stable | 8 |
| Unstable | 2 |

### Right Child

| Class | Number |
|-------|--------:|
| Stable | 2 |
| Unstable | 8 |

Each child contains

10 materials.

---

## 4.34 Entropy of the Left Child

The probabilities are

$$p(\text{Stable})
=
0.8,$$

$$p(\text{Unstable})
=
0.2.$$

Using Shannon's equation,

$$H
=
-
0.8\log_2(0.8)
-
0.2\log_2(0.2).$$

Substituting the logarithms,

$$H
=
-
0.8(-0.322)
-
0.2(-2.322).$$

Therefore,

$$H
=
0.258
+
0.464
=
0.722.$$

The right child has the same class proportions,

so its entropy is also

$$0.722.$$

---

## 4.35 Calculating the Weighted Entropy

Each child contains

10 of the original 20 materials.

Therefore,

the weighted entropy becomes

$$\frac{10}{20}(0.722)
+
\frac{10}{20}(0.722).$$

This simplifies to

$$0.722.$$

---

## 4.36 Calculating Information Gain

Now substitute everything into the Information Gain equation.

Parent entropy:

$$1.000.$$

Weighted child entropy:

$$0.722.$$

Therefore,

$$IG
=
1.000
-
0.722
=
0.278.$$

The split removes

0.278

units of uncertainty.

---

## 4.37 Comparing Two Different Splits

Suppose the tree evaluates another candidate split.

Its weighted entropy is

$$0.45.$$

The Information Gain becomes

$$IG
=
1.00
-
0.45
=
0.55.$$

Now compare both candidates.

| Split | Information Gain |
|-------|-----------------:|
| Split A | 0.278 |
| Split B | 0.550 |

Split B removes much more uncertainty.

Therefore,

the algorithm selects Split B.

This illustrates the fundamental learning strategy of a decision tree.

It does not choose the feature with the highest numerical value.

It chooses the split that maximizes Information Gain.

---

## 4.38 Why Information Gain Works

Information Gain rewards splits that create homogeneous child nodes.

Imagine two possible outcomes.

### Good Split

```text
Parent

Stable Stable Stable Stable

Unstable Unstable Unstable Unstable

↓

Split

Left

Stable Stable Stable Stable

Right

Unstable Unstable Unstable Unstable
```

Almost all uncertainty disappears.

Information Gain is large.

---

### Poor Split

```text
Parent

Stable Stable Stable Stable

Unstable Unstable Unstable Unstable

↓

Split

Left

Stable Stable Unstable Unstable

Right

Stable Stable Unstable Unstable
```

Each child still contains a mixture of classes.

Very little uncertainty is removed.

Information Gain is small.

---

## 4.39 Information Gain in Materials Science

Consider a database of lithium-ion battery materials.

Our objective is to classify compounds as

- High ionic conductivity
- Low ionic conductivity

The parent node contains

| Class | Number |
|-------|--------:|
| High | 400 |
| Low | 400 |

The entropy is high because both classes are equally represented.

Now suppose we test two descriptors.

### Candidate Split 1

Feature:

```
Average Electronegativity
```

After splitting,

most highly conductive materials fall into one child,

while low-conductivity materials fall into the other.

Entropy decreases substantially.

Information Gain is high.

---

### Candidate Split 2

Feature:

```
Lattice Volume
```

After splitting,

both child nodes still contain similar mixtures of high and low conductivity materials.

Entropy changes very little.

Information Gain is low.

The decision tree therefore selects **Average Electronegativity** as the first decision rule.

Notice that the algorithm reached this conclusion without any knowledge of electrochemistry.

It relied entirely on the statistical structure of the training data.

---

## 4.40 Information Gain Is a Greedy Strategy

One important characteristic of decision trees is that they use a **greedy algorithm**.

A greedy algorithm always makes the best decision **at the current step** without considering future decisions.

At each node,

the tree selects the split with the largest Information Gain.

It does not examine every possible complete tree.

This strategy makes training computationally efficient,

but it also means that the resulting tree is not always the globally optimal tree.

Despite this limitation,

greedy learning has proven remarkably successful in practice.

---

## 4.41 Advantages of Information Gain

Information Gain has several important strengths.

- It has a clear theoretical foundation in information theory.
- It naturally favors purer child nodes.
- It is easy to compute.
- It performs well for many classification problems.
- It forms the basis of classical algorithms such as ID3 and C4.5.

Because of these advantages,

Information Gain became one of the earliest and most influential criteria used for constructing decision trees.

---

## 4.42 A Hidden Weakness

Although Information Gain is powerful,

it has an important weakness that researchers soon discovered.

It tends to prefer features with many unique values.

For example,

consider a dataset containing

```
Material ID
```

Every material has a unique identification number.

If the tree splits using this feature,

each child node may contain only one sample.

Entropy becomes zero.

Information Gain becomes extremely large.

However,

the split is completely useless because material identification numbers contain no physical information.

The tree has simply memorized the training data.

This phenomenon is a form of overfitting.

To address this weakness,

later algorithms introduced improved splitting criteria,

such as the **Gain Ratio** and **Gini Impurity**.

---

## 4.43 Transition to the Next Topic

We have now answered one of the most important questions in decision tree learning:

> **How does a decision tree choose the best question to ask?**

The answer is:

It evaluates every candidate split,

computes the Information Gain,

and selects the split that removes the greatest amount of uncertainty.

However,

modern decision tree implementations—especially the CART algorithm used in **scikit-learn** and **Random Forests**—usually do **not** use Information Gain.

Instead,

they rely on another impurity measure called the **Gini Impurity**.

Although Gini Impurity pursues the same objective,

its mathematical formulation is different and offers several computational advantages.

In the next section,

we will derive the Gini Impurity equation from first principles, compare it directly with entropy, solve numerical examples by hand, and understand why it became the default choice for many modern tree-based algorithms.

## 4.44 Gini Impurity: The Mathematics Behind CART and Random Forest

When studying decision trees, many beginners assume that **Entropy** is the standard impurity measure used by every algorithm.

This is not true.

Although entropy was introduced first and has a beautiful mathematical foundation rooted in information theory, many modern decision tree implementations use a different impurity measure:

# Gini Impurity

In fact, if you train a `DecisionTreeClassifier` in **scikit-learn** without specifying the criterion,

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
```

the algorithm automatically uses

```
criterion = "gini"
```

Similarly,

Random Forest classifiers also use Gini impurity by default.

This immediately raises an important question.

> **If entropy already measures impurity, why was another measure developed?**

To answer this question, we must first understand what Gini impurity actually measures.

---

## 4.45 The Intuition Behind Gini Impurity

Imagine that you have a bag containing different colored balls.

Suppose the bag contains

```text
🔴 🔴 🔴 🔴 🔴
```

Every ball is red.

You close your eyes and randomly pick one ball.

Since every ball has the same color,

your prediction will always be correct.

Now imagine another bag.

```text
🔴 🔴 🔵 🔵 🔵
```

Without looking,

you randomly choose one ball.

Now there is a reasonable chance that your guess will be wrong.

Finally,

consider a third bag.

```text
🔴 🔵 🟢 🟡 ⚫
```

Every ball has a different color.

Your chance of making a wrong prediction becomes much larger.

This simple thought experiment captures the essence of Gini impurity.

Rather than measuring uncertainty directly, Gini impurity measures

> **the probability of incorrectly classifying a randomly selected sample if it were labeled according to the class distribution of the node.**

In other words,

higher impurity means a higher probability of making mistakes.

---

## 4.46 A Different Perspective

Entropy asks

> "How uncertain is this node?"

Gini asks

> "If I randomly assign labels according to the class probabilities, how often will I be wrong?"

Although these questions sound different,

they both reward exactly the same type of split:

one that produces purer child nodes.

This is why entropy and Gini often generate very similar decision trees.

---

## 4.47 The Gini Impurity Equation

For a node containing $n$ classes,

the Gini impurity is defined as

$$G
=
1-
\sum_{i=1}^{n}
p_i^2$$

where

- $G$ is the Gini impurity,
- $p_i$ is the probability of class $i$.

Unlike entropy,

this equation contains

- no logarithms,
- no exponentials,
- no complicated mathematical functions.

It requires only

- probabilities,
- squaring,
- addition,
- subtraction.

This simplicity makes Gini impurity computationally efficient.

---

## 4.48 Why Does the Formula Use Squared Probabilities?

At first glance,

the equation

$$1-\sum p_i^2$$

may seem mysterious.

Why do we square the probabilities?

To understand this,

consider a node with two classes.

Suppose

$$p(\text{Stable})=0.8$$

and

$$p(\text{Unstable})=0.2.$$

If we randomly assign labels,

the probability of correctly choosing

"Stable"

is

$$0.8\times0.8
=
0.64.$$

Similarly,

the probability of correctly choosing

"Unstable"

is

$$0.2\times0.2
=
0.04.$$

Therefore,

the total probability of making a correct prediction is

$$0.64+0.04
=
0.68.$$

Consequently,

the probability of making a mistake is

$$1-0.68
=
0.32.$$

Notice that this is exactly the Gini impurity.

Thus,

Gini impurity has a clear probabilistic interpretation.

---

## 4.49 Example 1: Perfectly Pure Node

Suppose every material is stable.

| Class | Probability |
|-------|------------:|
| Stable |1.0|
| Unstable|0.0|

The Gini impurity becomes

$$G
=
1-
(1)^2-
(0)^2.$$

Therefore,

$$G=0.$$

A perfectly pure node always has

$$G=0.$$

This is exactly what we expect.

No mistakes are possible because every sample belongs to the same class.

---

## 4.50 Example 2: Equal Class Distribution

Now suppose

| Class | Probability |
|-------|------------:|
| Stable |0.5|
| Unstable|0.5|

The Gini impurity becomes

$$G
=
1-
(0.5)^2-
(0.5)^2.$$

Calculating,

$$G
=
1-
0.25-
0.25
=
0.5.$$

Notice something interesting.

For binary classification,

the maximum possible Gini impurity is

$$0.5,$$

whereas the maximum entropy is

$$1.$$

The numerical scales differ,

but both measures identify this node as the most impure possible.

---

## 4.51 Example 3: Unequal Class Distribution

Suppose a node contains

| Class | Number |
|-------|--------:|
| Stable |8|
| Unstable|2|

The probabilities are

$$0.8$$

and

$$0.2.$$

The Gini impurity becomes

$$G
=
1-
(0.8)^2-
(0.2)^2.$$

Substituting,

$$G
=
1-
0.64-
0.04.$$

Therefore,

$$G=0.32.$$

Notice that

the node is less impure than the previous

50–50

example,

which had

$$G=0.5.$$

As one class begins to dominate,

Gini impurity decreases.

---

## 4.52 Comparing Entropy and Gini

Let us compare both impurity measures for several class distributions.

| Stable (%) | Unstable (%) | Entropy | Gini |
|------------:|-------------:|--------:|------:|
|100|0|0.000|0.000|
|90|10|0.469|0.180|
|80|20|0.722|0.320|
|70|30|0.881|0.420|
|60|40|0.971|0.480|
|50|50|1.000|0.500|

Several observations become immediately apparent.

Both measures

- reach their minimum value for perfectly pure nodes,
- increase as class distributions become more balanced,
- reach their maximum when classes are equally represented.

Although the numerical values differ,

their behavior is remarkably similar.

---

## 4.53 Visual Comparison

A simplified comparison is shown below.

```text
Impurity

^

|                Entropy

|              /

|            /

|          /

|        /

|      /

|    /  Gini

|  /

+------------------------------------>

Class Balance

Pure                     Mixed
```

Entropy increases slightly faster near the middle,

making it marginally more sensitive to changes in class distribution.

However,

for most practical machine learning problems,

the difference in predictive performance is very small.

---

## 4.54 Why CART Uses Gini Instead of Entropy

The **Classification and Regression Tree (CART)** algorithm,

developed by Breiman, Friedman, Olshen, and Stone,

adopted Gini impurity as its default splitting criterion.

There were several reasons for this decision.

### 1. Faster Computation

Entropy requires repeated logarithm calculations.

During tree construction,

the algorithm may evaluate

millions of candidate splits.

Avoiding logarithms reduces computational cost.

Although modern processors perform logarithms quickly,

this advantage was much more significant when CART was originally developed.

---

### 2. Simpler Mathematics

The Gini equation is computationally straightforward.

It requires only

- multiplication,
- addition,
- subtraction.

This simplicity also makes implementations easier.

---

### 3. Similar Predictive Performance

Extensive empirical studies have shown that

decision trees built using entropy and Gini usually produce very similar predictions.

Because Gini is computationally cheaper,

many implementations adopt it as the default.

---

## 4.55 Materials Science Example

Suppose we are developing a classifier that predicts whether a crystal is suitable for solid-state battery applications.

Initially,

the training node contains

| Class | Number |
|-------|--------:|
| Suitable |300|
| Unsuitable|300|

The Gini impurity is

$$0.5.$$

Now we test a split using

```
Average Oxidation State
```

After splitting,

Left Child

| Suitable |260|
|----------|---:|
| Unsuitable|40|

Right Child

| Suitable |40|
|----------|---:|
| Unsuitable|260|

Both child nodes have much lower impurity.

The CART algorithm therefore considers this descriptor highly informative.

Notice that the algorithm reached this conclusion

without any knowledge of electrochemistry.

It relied entirely on statistical evidence extracted from the training data.

---

## 4.56 Does It Matter Which Criterion We Choose?

One of the most common questions asked by beginners is

> "Should I use entropy or Gini?"

The practical answer is

**usually not.**

For most datasets,

both produce nearly identical trees.

However,

there are situations where entropy may produce slightly more balanced trees,

while Gini may isolate the dominant class slightly earlier.

The differences are generally small,

and hyperparameter selection often has a much larger impact on model performance than the choice between these impurity measures.

---

## 4.57 From Measuring Impurity to Building a Complete Tree

We now understand

- what impurity is,
- how entropy measures uncertainty,
- how Information Gain selects useful splits,
- how Gini impurity estimates classification error.

But one important question remains unanswered.

Even after selecting the best split,

**when should the algorithm stop growing the tree?**

Should it continue splitting forever?

Should every leaf contain exactly one sample?

Or should the tree stop earlier?

The answer to these questions determines whether a model generalizes well—or severely overfits the training data.

In the next section,

we will study the complete process of constructing a decision tree, understand recursive partitioning, examine stopping criteria, and learn why controlling tree growth is one of the most important aspects of building accurate machine learning models.

## 4.58 Building a Decision Tree: From Root to Leaf

So far, we have studied the mathematical tools that help a decision tree evaluate possible splits.

We now understand

- entropy,
- information gain,
- Gini impurity.

However, these concepts answer only **one** question:

> **"Which split is best?"**

They do **not** explain how an entire decision tree is constructed.

A complete decision tree may contain

- hundreds of nodes,
- dozens of levels,
- thousands of decision rules.

How are all of these generated?

Surprisingly,

the answer is based on one of the simplest ideas in computer science:

**Recursion.**

---

## 4.59 What Is Recursion?

Recursion means

> **solving a large problem by repeatedly solving smaller versions of the same problem.**

Many natural phenomena exhibit recursive behavior.

Consider the branching pattern of a tree.

```text
               Trunk
                 │
          ┌──────┴──────┐
          │             │
      Branch A      Branch B
       │    │         │    │
      A1   A2       B1    B2
```

Every branch grows in the same way.

A large branch produces smaller branches.

Each smaller branch again produces even smaller branches.

Decision trees follow exactly the same principle.

The algorithm repeatedly applies the same procedure:

1. Find the best split.
2. Divide the data.
3. Repeat the process for each child node.

The same algorithm is applied again and again until a stopping condition is reached.

---

## 4.60 The Recursive Tree-Building Algorithm

The entire decision tree learning process can be summarized as follows.

```text
Start with all training samples

            │

Find the best split

            │

Divide the dataset

      ┌──────────────┐
      │              │

Left Child      Right Child

      │              │

Repeat         Repeat

      │              │

Continue until stopping criteria are met
```

Notice that the algorithm never "plans" the complete tree in advance.

Instead,

it builds the tree one node at a time.

---

## 4.61 The Root Node

Every decision tree begins with a single node called the **root node**.

The root node contains

**every sample in the training dataset.**

For example,

suppose we have

1,000 crystalline materials.

Initially,

```text
Root Node

1000 Materials

Stable : 600

Unstable : 400
```

The algorithm evaluates every possible split and selects the one that produces the largest reduction in impurity.

Suppose the best question is

```
Average Electronegativity > 2.2 ?
```

The root node now splits into two child nodes.

---

## 4.62 Creating Child Nodes

After the first split,

the dataset becomes

```text
                 Root

Average Electronegativity > 2.2?

             /              \

          Yes                No

      520 Samples       480 Samples
```

Notice that

each child now contains a **smaller learning problem**.

Instead of analyzing

1,000 materials,

the algorithm now analyzes

520 materials

and

480 materials

independently.

---

## 4.63 The Same Process Repeats

The algorithm now completely ignores the parent node.

Instead,

it focuses only on one child.

Suppose we examine

```text
520 Materials
```

Again,

the algorithm asks

> "What is the best possible split for these 520 samples?"

Perhaps the answer is

```
Density > 5.6 g/cm³
```

The node divides again.

```text
520 Samples

      │

Density > 5.6?

     /        \

320         200
```

The same process continues.

Notice that

the algorithm does **not** change.

Only the dataset becomes smaller.

---

## 4.64 Recursive Partitioning

This repeated splitting process is called

# Recursive Partitioning

The term "partitioning" simply means

dividing a dataset into smaller subsets.

The term "recursive"

means

the same procedure is applied repeatedly.

Mathematically,

if the original dataset is

$$S,$$

then after one split,

$$S
=
S_1
\cup
S_2.$$

Each subset

is then partitioned again.

Eventually,

the dataset becomes

$$S_{11},
S_{12},
S_{21},
S_{22},
\ldots$$

until no further useful splits remain.

---

## 4.65 Why Does Recursive Partitioning Work?

Suppose we attempt to model an extremely complicated function

$$f(x).$$

Rather than approximating

the entire function at once,

a decision tree approximates

many small regions separately.

For example,

instead of

```text
One difficult problem

↓

Solve everything simultaneously
```

the tree performs

```text
One difficult problem

↓

Two easier problems

↓

Four even easier problems

↓

Eight simple problems
```

Eventually,

each region becomes so simple

that prediction is straightforward.

This divide-and-conquer strategy is one of the reasons decision trees perform well on highly nonlinear datasets.

---

## 4.66 Decision Regions

Every split divides the feature space.

Suppose we have only two descriptors.

- Density
- Electronegativity

Initially,

the entire feature space belongs to one region.

```text
+------------------------+

|                        |

|       All Samples      |

|                        |

+------------------------+
```

After one split,

```text
+------------+-----------+

|            |           |

| Region A   | Region B  |

|            |           |

+------------+-----------+
```

After several splits,

```text
+-----+--------+---------+

| A1  |   A2   |   B1    |

+-----+--------+---------+

| A3  |   B2   |   B3    |

+-----+--------+---------+
```

Each small rectangle corresponds to

one leaf node of the decision tree.

Instead of learning one complicated equation,

the algorithm learns many simple local decision rules.

---

## 4.67 Leaf Nodes

Eventually,

the recursive splitting process stops.

The final nodes are called

# Leaf Nodes

A leaf node contains

the final prediction.

For classification,

the leaf predicts

the majority class.

Example

```text
Leaf Node

Stable : 48

Unstable : 2

↓

Prediction

Stable
```

For regression,

the leaf predicts

the average target value.

Example

```text
Leaf Node

Formation Energy

-3.10

-3.05

-3.22

↓

Prediction

-3.12 eV/atom
```

---

## 4.68 Every Path Represents a Rule

One remarkable feature of decision trees is that

every path from the root to a leaf

forms an interpretable decision rule.

For example,

```text
Electronegativity > 2.2

↓

Density > 5.4

↓

Band Gap > 2 eV
```

This becomes

> **IF**
>
> Electronegativity > 2.2
>
> AND Density > 5.4
>
> THEN Predict High Band Gap.

Unlike many machine learning models,

decision trees naturally produce human-readable rules.

This interpretability is one of their greatest strengths.

---

## 4.69 A Materials Science Example

Suppose we are predicting whether a crystal is mechanically stable.

The trained decision tree may produce rules such as

```text
IF

Average Atomic Radius < 1.35 Å

AND

Average Electronegativity > 2.5

AND

Density > 6.0 g/cm³

↓

Predict

Mechanically Stable
```

Another branch might produce

```text
IF

Average Atomic Radius > 1.60 Å

AND

Density < 4.5 g/cm³

↓

Predict

Mechanically Unstable
```

Notice that

the algorithm automatically discovered these rules from data.

No human explicitly programmed them.

---

## 4.70 Does the Tree Ever Stop Growing?

At this point,

an obvious question arises.

If recursive partitioning keeps dividing the data,

what prevents the algorithm from continuing forever?

Could it keep splitting until

every leaf contains exactly one training sample?

The answer is

**yes.**

If unrestricted,

a decision tree will continue growing until every leaf becomes perfectly pure.

Although this sounds desirable,

it creates one of the most serious problems in machine learning:

**Overfitting.**

A perfectly pure training tree often performs poorly on completely new materials because it memorizes the training data instead of learning the underlying physical relationships.

Understanding when to stop growing a tree is therefore just as important as understanding how to grow it.

In the next section,

we will study stopping criteria, tree depth, pruning, underfitting, overfitting, and how modern decision tree algorithms balance model complexity with predictive accuracy.


## 4.71 Pruning: Controlling Tree Complexity

In the previous section, we learned that a decision tree can continue growing until every leaf becomes perfectly pure. From the perspective of the training dataset, this may appear ideal because every training sample is classified or predicted correctly. However, machine learning models are not built to memorize the training data. Their true objective is to learn the underlying relationship between input features and target values so that they can make reliable predictions for completely unseen data.

This distinction introduces one of the most important concepts in decision tree learning: **pruning**.

Pruning is the process of reducing the complexity of a decision tree by removing branches that contribute little or nothing to predictive performance on unseen data. Instead of preserving every possible split discovered during training, pruning evaluates whether a branch captures meaningful patterns or merely memorizes accidental variations present in the training dataset.

The objective of pruning is therefore not simply to build a smaller tree. The true objective is to maximize **generalization**, which is the ability of a machine learning model to perform well on new data that it has never encountered before.

To understand why pruning is necessary, consider a materials informatics dataset containing several hundred crystalline compounds. Suppose the descriptors were generated using Pymatgen from crystal structures and include properties such as average atomic radius, average electronegativity, density, lattice volume, packing fraction, and coordination number. A decision tree may continue splitting until each leaf contains only a single material.

Such a tree may correctly predict every training sample, achieving nearly perfect training accuracy. At first glance, this appears to indicate an excellent model. In reality, however, the tree has probably memorized individual materials rather than discovering the physical relationships that govern the property being predicted.

Imagine introducing a newly synthesized material whose descriptor values differ only slightly from those observed during training. Because the tree contains many highly specific decision rules, the new material may follow an entirely different path through the tree and receive an inaccurate prediction. The model has become extremely specialized for the training data and has lost its ability to generalize.

This phenomenon is known as **overfitting**.

Pruning addresses this problem by removing branches that primarily model random fluctuations instead of genuine scientific trends.

An intuitive analogy can help clarify this idea. Imagine two students preparing for an examination. The first student memorizes every solved example in a textbook without understanding the underlying concepts. The second student studies the physical principles, derives the equations, and understands why each method works.

If the examination questions are identical to the textbook examples, the first student performs exceptionally well. However, if the questions are modified slightly, that student's performance declines because the knowledge was based on memorization rather than understanding.

The second student may not remember every numerical example, but they can solve unfamiliar problems because they understand the underlying concepts.

An overgrown decision tree behaves like the first student.

A properly pruned decision tree behaves like the second.

Machine learning seeks understanding rather than memorization.

The relationship between tree complexity and prediction error can be illustrated conceptually.

```text
Prediction Error

^

|\
| \
|  \
|   \
|    \         Testing Error
|     \_______/\

|              \

|               \____________ Training Error

+----------------------------------------------------> Tree Complexity

## 4.72 Pre-Pruning and Post-Pruning

Pruning can be performed at two different stages of the tree-building process.

The first strategy attempts to prevent unnecessary branches from ever being created.

The second strategy allows the complete tree to grow first and then removes branches that are found to be unnecessary.

These two approaches are known as

- **Pre-Pruning**
- **Post-Pruning**

Although both methods have the same objective—reducing overfitting—they achieve this objective in fundamentally different ways.

Understanding the distinction between them is essential because modern decision tree algorithms often rely on one or both of these strategies to produce models that generalize well.

---

### Pre-Pruning

Pre-pruning, sometimes called **early stopping**, prevents the tree from becoming excessively large during the training process.

Instead of allowing recursive partitioning to continue indefinitely, the algorithm continuously checks whether a proposed split satisfies predefined conditions.

If these conditions are not satisfied, the node immediately becomes a leaf.

The algorithm therefore decides

> **"This node is already good enough. Further splitting is unlikely to improve predictions."**

Rather than growing a large tree and simplifying it later,

pre-pruning simply avoids creating unnecessary branches in the first place.

The recursive tree-building process therefore changes slightly.

Instead of

```text
Node

↓

Find Best Split

↓

Always Split

↓

Continue Growing
```

the algorithm becomes

```text
Node

↓

Find Best Split

↓

Check Stopping Rules

↓

Stop
or
Split
```

Every potential split must pass several conditions before it is accepted.

If any stopping criterion is violated,

the recursive process immediately terminates for that branch.

---

### Common Pre-Pruning Criteria

Modern decision tree implementations usually provide several hyperparameters that control pre-pruning.

Some of the most important are

- maximum tree depth,
- minimum samples required for splitting,
- minimum samples allowed in a leaf,
- minimum impurity decrease,
- maximum number of leaf nodes.

These hyperparameters determine how aggressively the algorithm limits tree growth.

Instead of relying on a single stopping rule,

multiple criteria are usually evaluated simultaneously.

For example,

a node may still contain many samples,

but the impurity reduction produced by another split may be so small that continuing the recursion is no longer worthwhile.

In that situation,

the algorithm stops even though additional splits are technically possible.

---

### Example of Pre-Pruning

Suppose a decision tree is predicting the formation energy of crystalline materials.

Initially,

the root node contains

```text
1000 Materials
```

After several recursive splits,

one branch eventually contains

```text
12 Materials
```

The algorithm discovers another possible split.

However,

assume that the hyperparameter

```python
min_samples_split = 20
```

has already been specified.

Since the node contains only

12 samples,

the stopping condition is activated.

The branch immediately becomes a leaf.

No additional children are created.

Even if another split could slightly reduce the impurity,

the algorithm refuses to continue because doing so would likely increase the risk of overfitting.

---

### Advantages of Pre-Pruning

Pre-pruning offers several practical advantages.

First,

training becomes significantly faster because the algorithm does not waste computational effort constructing branches that will probably be removed later.

Second,

the resulting trees are usually much smaller,

making them easier to visualize and interpret.

Finally,

smaller trees often require less memory,

which becomes increasingly important for large datasets containing hundreds of thousands of samples.

These advantages explain why pre-pruning is widely used in industrial machine learning systems where computational efficiency is important.

---

### Limitations of Pre-Pruning

Although pre-pruning reduces overfitting,

it introduces another possible problem.

The algorithm may stop **too early**.

Imagine that a node currently appears unimportant,

but one additional split would reveal an important physical relationship hidden inside the data.

Because pre-pruning terminates the recursion prematurely,

that relationship is never discovered.

The resulting tree may therefore become

**too simple**.

This produces the opposite problem of overfitting.

Instead of memorizing the training data,

the model now fails to learn enough from it.

This phenomenon is known as

**underfitting**.

A heavily pre-pruned tree may ignore important nonlinear relationships between descriptors such as density, atomic radius, electronegativity, and formation energy simply because the stopping criteria prevented further exploration.

Finding appropriate pre-pruning parameters therefore requires balancing computational efficiency against predictive accuracy.

Stopping too late produces overfitting.

Stopping too early produces underfitting.

The ideal solution lies somewhere between these two extremes.

In the next section,

we will study **post-pruning**, an alternative strategy that first allows the decision tree to grow completely before identifying and removing unnecessary branches.


## 4.73 Post-Pruning

Pre-pruning attempts to prevent unnecessary branches from being created.

Post-pruning follows the opposite philosophy.

Instead of restricting the tree during training,

the algorithm first allows the decision tree to grow almost completely.

Only after the entire tree has been constructed does it begin asking an important question:

> **Which branches actually improve prediction on unseen data, and which branches merely memorize the training set?**

Branches that contribute little to generalization are removed one by one.

This process is called **post-pruning** because pruning occurs **after** the complete tree has been generated.

The workflow therefore becomes

```text
Training Dataset

↓

Grow Complete Tree

↓

Evaluate Every Branch

↓

Remove Weak Branches

↓

Simplified Tree
```

Unlike pre-pruning,

post-pruning allows the algorithm to explore every possible relationship before deciding which parts of the tree are genuinely useful.

Because the complete structure is available,

the algorithm has considerably more information when deciding whether a branch should remain.

---

### Why Grow the Entire Tree First?

At first glance,

post-pruning may appear inefficient.

Why spend computational resources creating branches that will later be removed?

The answer lies in the difficulty of making local decisions during recursive tree construction.

Suppose the algorithm reaches a node containing only fifty materials.

At that moment,

the split appears to reduce impurity only slightly.

A pre-pruning algorithm might immediately stop.

However,

if the tree were allowed to continue growing,

those fifty materials might later separate into several physically meaningful groups based on combinations of descriptors that were not obvious earlier.

Stopping too early would permanently hide these relationships.

Post-pruning avoids this problem.

Instead of making decisions using only local information,

it evaluates the completed tree using global performance.

Consequently,

the algorithm can determine whether an entire branch genuinely improves prediction instead of judging individual splits in isolation.

---

### Conceptual Example

Suppose a decision tree predicting formation energy grows into the following structure.

```text
                     Root

                       │

          Average Electronegativity

             /                  \

          Node A              Node B

         /      \            /      \

      Node C   Node D     Node E   Node F

      /   \      |          |         |

    Leaf Leaf   Leaf      Leaf      Leaf
```

After evaluating the model on unseen validation data,

suppose the algorithm discovers that the entire branch beginning at **Node D** contributes almost nothing to predictive accuracy.

Instead of keeping this unnecessary complexity,

the branch is removed.

The simplified tree becomes

```text
                     Root

                       │

          Average Electronegativity

             /                  \

          Node A              Node B

         /                     /      \

      Node C                Node E   Node F

      /   \                   |         |

    Leaf Leaf               Leaf      Leaf
```

Notice that the pruning process does not randomly delete nodes.

Only branches that fail to improve predictive performance are removed.

---

### Cost Complexity Pruning

Modern implementations of decision trees often rely on a technique called **Cost Complexity Pruning**, sometimes referred to as **Weakest Link Pruning**.

Instead of considering only prediction accuracy,

the algorithm simultaneously considers

- prediction error,
- tree complexity.

The objective is no longer simply

> "Find the most accurate tree."

Instead,

the objective becomes

> "Find the simplest tree that still predicts accurately."

Mathematically,

this idea can be expressed using the following objective function.

$$R_\alpha(T)=R(T)+\alpha|T|$$

where

- $R(T)$ represents the prediction error of the tree,
- $|T|$ represents the number of terminal leaf nodes,
- $\alpha$ is the complexity penalty.

The first term rewards accurate prediction.

The second term penalizes unnecessarily large trees.

The parameter $\alpha$ controls the balance between these competing objectives.

---

### Understanding the Complexity Parameter

Consider three different values of the complexity parameter.

#### Small $\alpha$

When

$$\alpha \approx 0,$$

almost no penalty is applied.

The algorithm therefore focuses almost entirely on minimizing prediction error.

Large trees are likely to remain.

The resulting model may become highly accurate on the training data but also more susceptible to overfitting.

---

#### Moderate $\alpha$

As the penalty increases,

small improvements in training accuracy are no longer considered worthwhile if they require many additional branches.

The algorithm begins removing branches whose contribution is relatively small.

This usually produces the best balance between model complexity and predictive performance.

---

#### Large $\alpha$

If the penalty becomes excessively large,

the algorithm aggressively removes branches.

Eventually,

the tree may become so simple that it resembles a single decision rule.

Although overfitting is greatly reduced,

the model may now suffer from underfitting because it cannot represent sufficiently complex relationships.

The choice of $\alpha$ therefore determines where the model lies on the spectrum between underfitting and overfitting.

---

### Comparing Pre-Pruning and Post-Pruning

Although both approaches attempt to improve generalization,

their philosophies differ considerably.

| Feature | Pre-Pruning | Post-Pruning |
|---------|-------------|--------------|
| When pruning occurs | During tree construction | After the complete tree is built |
| Computational cost | Lower | Higher |
| Risk | Underfitting | Lower risk of premature stopping |
| Information available | Local node information | Entire tree structure |
| Typical accuracy | Good | Often slightly better |

Pre-pruning is computationally efficient because unnecessary branches are never created.

Post-pruning is computationally more expensive,

but its decisions are generally better informed because they consider the performance of the complete tree.

For relatively small datasets,

post-pruning often produces superior models.

---

### Which Strategy Is Used in Practice?

Different machine learning libraries implement different pruning strategies.

For example,

Scikit-learn primarily controls tree complexity through hyperparameters such as

- `max_depth`,
- `min_samples_split`,
- `min_samples_leaf`,
- `max_leaf_nodes`,
- `ccp_alpha`.

The first four perform **pre-pruning**.

The final parameter,

`ccp_alpha`,

implements **Cost Complexity Post-Pruning**.

This allows researchers to choose whether complexity should be controlled before the tree grows,

after the tree grows,

or by combining both approaches.

In practical materials informatics workflows,

both strategies are commonly used together.

A moderate maximum depth prevents excessively large trees,

while cost-complexity pruning removes any remaining unnecessary branches after training.

This combination often produces highly interpretable models with strong predictive performance.

---

## 4.74 Hyperparameters That Control Decision Tree Growth

Pruning strategies rely on several quantities that are chosen **before** the model begins learning.

These quantities are called **hyperparameters**.

Unlike the decision rules inside the tree,

hyperparameters are **not learned automatically** from the data.

Instead,

they are selected by the researcher based on experience, experimentation, or systematic optimization.

Changing these values can dramatically alter the structure of the final decision tree.

Two researchers using the same dataset but different hyperparameter values may produce completely different trees.

Understanding these hyperparameters is therefore just as important as understanding entropy, information gain, or recursive partitioning.

In the following sections, we will study every major decision tree hyperparameter individually, understand its mathematical role, examine how it changes tree growth, and learn how to choose appropriate values for real materials informatics problems.





## 4.75 Pre-Pruning: Preventing Overly Complex Trees Before They Grow

In the previous sections, we learned that an unrestricted decision tree will continue splitting until it perfectly fits the training data. Although this produces extremely low training error, it often leads to severe overfitting because the tree begins to memorize individual training samples rather than discovering the underlying physical relationships that govern the dataset.

One strategy for avoiding this problem is to stop unnecessary growth **before** the tree becomes excessively complex. This approach is known as **pre-pruning**, sometimes called **early stopping**.

Rather than allowing the tree to grow completely and simplifying it later, pre-pruning introduces a set of rules that determine whether another split is worthwhile. If a proposed split fails to satisfy these rules, the algorithm immediately stops expanding that branch and converts the current node into a leaf node.

Conceptually, the procedure becomes

```text
Current Node

      │

Evaluate Best Split

      │

Is Split Worthwhile?

      │

 ┌────┴────┐

 Yes      No

 │         │

Split    Stop Growing
```

The algorithm therefore asks an additional question before creating every new branch:

> **"Will this split likely improve the model's ability to predict unseen data?"**

If the answer is no, the branch is never created.

---

### Why Is Pre-Pruning Necessary?

Suppose we are predicting the formation energy of crystalline materials using descriptors extracted from Pymatgen.

Our dataset contains

- density,
- average electronegativity,
- atomic volume,
- packing fraction,
- average atomic radius,
- valence electron concentration.

Initially, the decision tree discovers meaningful physical relationships.

For example,

```text
Density > 5.8

↓

Average Electronegativity > 2.1

↓

Formation Energy Prediction
```

These rules describe broad trends shared by many materials.

However, as the tree grows deeper, it may begin creating rules such as

```text
Density > 5.8123

AND

Average Radius < 1.427 Å

AND

Volume > 157.91 Å³

AND

Packing Fraction < 0.6312
```

Such highly specific conditions often separate only one or two training samples.

Instead of learning general materials behavior, the tree begins memorizing numerical noise.

Pre-pruning prevents the algorithm from creating these unnecessary branches.

---

### Advantages of Pre-Pruning

A properly pre-pruned tree offers several benefits.

First,

training becomes significantly faster because fewer nodes are evaluated.

Second,

the resulting model consumes less memory.

Third,

the tree becomes easier to interpret since it contains fewer decision rules.

Most importantly,

pre-pruning usually improves the model's ability to generalize to unseen materials by reducing overfitting.

---

### Potential Disadvantage

Although pre-pruning helps prevent overfitting, it introduces another possible problem.

If growth stops too early, the tree may fail to learn important relationships hidden within the data.

Consider a node containing 300 materials.

Perhaps the immediate reduction in impurity produced by the next split appears small.

A pre-pruning rule might reject the split.

However, that rejected split could have enabled several extremely informative splits later in the tree.

Because the algorithm never explores those future possibilities, valuable information is lost.

This phenomenon is known as **underfitting**.

The model becomes too simple to represent the complexity of the underlying physical system.

Consequently, selecting appropriate pre-pruning parameters requires balancing two competing objectives:

- preventing overfitting,
- avoiding excessive simplification.

---

## 4.76 Common Pre-Pruning Parameters

Modern decision tree implementations provide several hyperparameters that control tree growth.

Rather than allowing unlimited expansion, these parameters impose practical limits on the learning process.

The most commonly used parameters include

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_leaf_nodes`
- `min_impurity_decrease`
- `max_features`

Each parameter influences a different aspect of tree construction.

Understanding their behavior is essential for building reliable machine learning models.

---

## 4.77 Maximum Depth (`max_depth`)

The simplest and most widely used pre-pruning parameter is

```python
max_depth
```

This parameter specifies the maximum number of levels that a decision tree is allowed to grow.

For example,

```python
DecisionTreeRegressor(max_depth=3)
```

restricts the tree to only three levels below the root node.

Conceptually,

```text
Depth 0

Root

↓

Depth 1

↓

Depth 2

↓

Depth 3

↓

Stop
```

Even if additional splits could further reduce impurity, the algorithm refuses to grow beyond the specified depth.

---

### Effect of Tree Depth

Tree depth has a direct influence on model complexity.

A shallow tree

```text
Small Depth

↓

Simple Rules

↓

Lower Variance

↓

Possible Underfitting
```

A deep tree

```text
Large Depth

↓

Complex Rules

↓

Higher Variance

↓

Possible Overfitting
```

Neither extreme is universally optimal.

Instead, the best depth depends on

- dataset size,
- feature quality,
- measurement noise,
- complexity of the underlying physical relationships.

---

### Choosing Tree Depth for Materials Datasets

Materials datasets often contain only a few hundred or a few thousand samples.

Examples include

- experimentally measured elastic constants,
- DFT formation energies,
- band gaps,
- thermal conductivities.

Because these datasets are relatively small, extremely deep trees frequently memorize individual compounds instead of learning transferable physical relationships.

For many materials informatics problems, moderate tree depths such as

```text
3

5

8

10
```

often provide a good balance between model flexibility and generalization.

However, the optimal value should always be determined using cross-validation rather than fixed assumptions.

---

## 4.78 Minimum Samples Required to Split (`min_samples_split`)

Another important parameter is

```python
min_samples_split
```

This parameter controls the minimum number of samples that must exist inside a node before the algorithm is allowed to create another split.

Suppose

```python
min_samples_split=20
```

Now imagine the current node contains

```text
18 Samples
```

Because

```text
18 < 20
```

the algorithm immediately stops growing that branch.

The node becomes a leaf even if impurity remains relatively high.

On the other hand,

if the node contains

```text
35 Samples
```

another split is permitted because

```text
35 ≥ 20
```

---

### Why Is This Useful?

Very small nodes usually represent highly specific subsets of the training data.

Further splitting these nodes often creates rules that apply to only one or two samples.

Such rules rarely generalize well.

By requiring a minimum number of observations before splitting, the algorithm focuses on statistically meaningful partitions rather than random fluctuations.

As the value of `min_samples_split` increases,

the resulting trees become progressively simpler.

Smaller values produce more detailed trees but also increase the risk of overfitting.

---

## 4.79 Minimum Samples per Leaf (`min_samples_leaf`)

While `min_samples_split` controls whether a node can be divided,

another parameter controls the size of the final prediction regions.

This parameter is

```python
min_samples_leaf
```

It specifies the minimum number of training samples that every leaf node must contain.

For example,

```python
min_samples_leaf=5
```

means that every terminal node must contain at least five observations.

Consider the following split.

```text
Current Node

12 Samples

↓

Possible Split

↓

Left Leaf : 10 Samples

Right Leaf : 2 Samples
```

Because the second leaf contains only two samples,

the split is rejected.

The algorithm keeps the original node unchanged.

This restriction prevents the creation of extremely small prediction regions that are dominated by individual observations.

In regression problems, larger leaf sizes usually produce smoother predictions because every prediction represents the average of multiple training samples rather than a single observation.

For noisy materials datasets, increasing `min_samples_leaf` is often an effective way to improve generalization without drastically reducing model flexibility.

## 4.80 Maximum Number of Leaf Nodes (`max_leaf_nodes`)

Until now, the complexity of a decision tree has been controlled primarily by restricting its depth or by limiting how many samples are allowed in each node. There is another intuitive way to control complexity that focuses directly on the final prediction regions rather than the intermediate branches.

This hyperparameter is

```python
max_leaf_nodes
```

Instead of asking

> **"How deep can the tree become?"**

it asks

> **"How many final prediction regions is the tree allowed to create?"**

Recall that every leaf node represents one final prediction.

For a classification tree,

each leaf predicts one class.

For a regression tree,

each leaf predicts one numerical value.

Consequently,

every additional leaf increases the flexibility of the model.

---

### Understanding Leaf Nodes

Consider the following simple tree.

```text
                 Root

               /      \

          Node        Node

         /   \       /   \

      Leaf  Leaf   Leaf  Leaf
```

This tree contains

```text
4 Leaf Nodes
```

Suppose we specify

```python
max_leaf_nodes=4
```

The tree above satisfies the restriction.

However, if the algorithm attempts another split,

```text
                 Root

               /      \

          Node        Node

         /   \       /   \

      Node  Leaf   Leaf  Leaf

     /   \

 Leaf   Leaf
```

the number of terminal nodes becomes

```text
5 Leaf Nodes
```

Since

```text
5 > 4
```

the additional split is rejected.

The tree immediately stops growing.

---

### Relationship Between Leaves and Decision Regions

Every leaf corresponds to one region in feature space.

Suppose our model uses only two descriptors:

- Density
- Average Electronegativity

Initially,

the entire feature space belongs to a single prediction region.

```text
+---------------------------+

|                           |

|      One Prediction       |

|                           |

+---------------------------+
```

After several splits,

multiple regions appear.

```text
+---------+---------+

|    A    |    B    |

+----+----+---------+

| C  | D  |    E    |

+----+----+---------+
```

Each region corresponds to one leaf node.

Increasing the number of leaves allows increasingly complicated decision boundaries.

Reducing the number of leaves forces the algorithm to combine nearby regions into broader, smoother prediction areas.

---

### Why Restrict the Number of Leaves?

Suppose we are predicting the elastic modulus of crystalline materials.

If the tree contains

```text
3 Leaves
```

each prediction represents a relatively broad category of materials.

The resulting model is simple and easy to interpret.

Now imagine allowing

```text
300 Leaves
```

Many leaves may contain only one or two materials.

Instead of identifying general relationships such as

```text
Higher Density

↓

Higher Elastic Modulus
```

the tree begins creating highly specialized rules that apply to only a handful of compounds.

Although training accuracy improves,

generalization often becomes worse.

Restricting the maximum number of leaves prevents this unnecessary fragmentation of the feature space.

---

### Comparing `max_depth` and `max_leaf_nodes`

Although both parameters reduce model complexity,

they operate differently.

`max_depth`

controls

```text
Vertical Growth
```

The tree cannot continue growing beyond a specified level.

`max_leaf_nodes`

controls

```text
Horizontal Complexity
```

The total number of prediction regions is limited regardless of the tree's shape.

Consider two different trees.

Tree A

```text
Depth = 6

Leaves = 12
```

Tree B

```text
Depth = 10

Leaves = 12
```

Although Tree B is deeper,

both trees possess the same number of prediction regions.

Therefore,

tree depth alone does not completely describe model complexity.

The number of leaves is equally important.

---

### Choosing an Appropriate Number of Leaves

There is no universal rule for selecting

```python
max_leaf_nodes
```

Very small values

```text
5

10

20
```

produce highly interpretable models but may underfit complicated datasets.

Very large values

```text
500

1000

Unlimited
```

produce extremely flexible models that may memorize the training data.

For many materials informatics applications,

moderate values combined with cross-validation often provide the best balance between interpretability and predictive accuracy.

---

## 4.81 Minimum Impurity Decrease (`min_impurity_decrease`)

Until now,

every split has been accepted whenever it reduced impurity.

However,

not every reduction is meaningful.

Suppose a parent node has a Gini impurity of

```text
0.4200
```

After splitting,

the weighted child impurity becomes

```text
0.4198
```

The impurity reduction is

```text
0.0002
```

Technically,

this split improves the model.

Practically,

the improvement is almost negligible.

Creating two additional branches merely to reduce impurity by

```text
0.0002
```

rarely improves predictive performance.

Instead,

it usually increases complexity without producing significant scientific benefit.

To avoid this problem,

decision tree algorithms provide the parameter

```python
min_impurity_decrease
```

---

### How Does It Work?

Suppose we specify

```python
min_impurity_decrease=0.01
```

Now consider two possible splits.

Split A

```text
Impurity Before

0.48

↓

Impurity After

0.44

↓

Reduction

0.04
```

Since

```text
0.04 > 0.01
```

the split is accepted.

Now consider another candidate.

```text
Impurity Before

0.48

↓

Impurity After

0.475

↓

Reduction

0.005
```

Since

```text
0.005 < 0.01
```

the split is rejected.

The node immediately becomes a leaf.

In this way,

the algorithm ignores insignificant improvements and concentrates only on branches that meaningfully increase predictive quality.

---

### Why Tiny Improvements Can Be Misleading

As the tree grows deeper,

the number of samples inside each node becomes progressively smaller.

When only a few observations remain,

even random fluctuations may produce slight decreases in impurity.

Without any restriction,

the algorithm interprets these tiny improvements as meaningful and continues growing.

Eventually,

the tree memorizes the training data.

By requiring a minimum impurity reduction,

the algorithm distinguishes between

- genuine structural patterns,
- random statistical noise.

This produces simpler and more reliable models.

---

### Materials Informatics Example

Suppose we are predicting the formation energy of oxide materials.

One branch currently contains

```text
42 Materials
```

The best candidate split separates them into

```text
21 Materials

21 Materials
```

However,

the reduction in impurity is only

```text
0.0007
```

Such a small improvement suggests that both child groups possess nearly identical statistical behavior.

Creating an entirely new branch therefore contributes very little scientific information.

A minimum impurity threshold prevents this unnecessary split.

Instead,

the algorithm preserves a simpler decision rule that represents the broader physical trend shared by all forty-two materials.

---

## 4.82 Combining Multiple Pre-Pruning Criteria

In practical machine learning,

decision trees almost never rely on a single stopping rule.

Instead,

several pre-pruning criteria operate simultaneously.

During training,

the algorithm continuously checks questions such as

- Has the maximum depth been reached?
- Does the node contain enough samples to split?
- Will every child contain enough samples?
- Does the proposed split reduce impurity sufficiently?
- Has the maximum number of leaf nodes already been created?

Only if **all** conditions are satisfied does recursive partitioning continue.

Otherwise,

the current node immediately becomes a leaf.

This combination of stopping rules allows researchers to carefully balance

- model complexity,
- computational efficiency,
- predictive accuracy,
- generalization to unseen materials.

For this reason,

effective decision tree construction is not simply about finding the best split.

It is equally about deciding **when further splitting is no longer scientifically justified**.

## 4.83 Post-Pruning: Simplifying a Fully Grown Decision Tree

In the previous sections, we studied **pre-pruning**, where the algorithm decides *before* creating a branch whether further growth is worthwhile.

There is another strategy that approaches the problem from the opposite direction.

Instead of stopping the tree early,

we intentionally allow the tree to grow almost completely.

Only after the entire tree has been constructed do we begin removing unnecessary branches.

This strategy is known as

# Post-Pruning

or

# Cost-Complexity Pruning.

Conceptually, the workflow becomes

```text
Training Dataset

        │

Grow Complete Tree

        │

Large Complex Tree

        │

Remove Unnecessary Branches

        │

Simplified Tree

        │

Final Model
```

Unlike pre-pruning,

which attempts to prevent overfitting,

post-pruning first allows the model to capture every possible relationship and then carefully determines which branches genuinely improve prediction and which merely memorize the training data.

---

## 4.84 Why Grow the Entire Tree First?

At first glance,

allowing the tree to overfit may seem like a poor strategy.

However,

there is a good reason for doing so.

During pre-pruning,

the algorithm makes decisions using only local information.

Suppose the current split appears to reduce impurity only slightly.

The algorithm may reject it immediately.

Unfortunately,

that seemingly unimportant split might eventually lead to several highly informative splits deeper in the tree.

Because pre-pruning never explores those possibilities,

valuable information can be lost.

Post-pruning avoids this problem.

Instead of making early decisions,

it allows every branch to develop fully.

Only after seeing the complete tree does it decide which branches are actually useful.

In other words,

pre-pruning asks

> **"Should I create this branch?"**

Post-pruning asks

> **"Now that the branch exists, was it actually worth creating?"**

This generally leads to more informed pruning decisions.

---

## 4.85 Understanding Cost-Complexity Pruning

The most widely used post-pruning algorithm is called

**Cost-Complexity Pruning**.

Its central idea is surprisingly intuitive.

A good decision tree should satisfy two objectives simultaneously.

First,

it should predict accurately.

Second,

it should remain reasonably simple.

These objectives often conflict.

A very large tree usually achieves excellent training accuracy but poor generalization.

A very small tree generalizes well but may fail to capture important relationships.

Cost-complexity pruning balances these competing goals by introducing a penalty for unnecessary complexity.

Instead of evaluating only prediction error,

the algorithm evaluates

```text
Prediction Error

+

Complexity Penalty
```

The preferred tree is the one that minimizes the combined quantity rather than prediction error alone.

---

## 4.86 Mathematical Formulation of Cost-Complexity Pruning

The objective function used during pruning can be written as

$$R_\alpha(T)
=
R(T)
+
\alpha |T|$$

where

- $R(T)$ represents the prediction error of the tree,
- $|T|$ represents the number of leaf nodes,
- $\alpha$ is the complexity penalty.

Each term has an important interpretation.

The first term rewards predictive accuracy.

The second term penalizes large trees.

As the number of leaf nodes increases,

the penalty also increases.

Consequently,

adding another branch is worthwhile only if the resulting improvement in prediction error is larger than the additional complexity penalty.

This simple equation formalizes one of the central ideas of machine learning:

> **The best model is not necessarily the most accurate on the training data. The best model achieves the best balance between accuracy and simplicity.**

---

## 4.87 Understanding the Complexity Parameter $\alpha$

The parameter

$$\alpha$$

controls how strongly model complexity is penalized.

Suppose

$$\alpha = 0.$$

Then

$$R_\alpha(T)
=
R(T)$$

Only prediction error matters.

The algorithm therefore keeps every branch that improves training accuracy,

regardless of how small the improvement may be.

This usually produces a very large tree.

Now suppose

$$\alpha$$

becomes much larger.

Every additional leaf increases the penalty significantly.

The algorithm now removes many branches because the increase in complexity outweighs the slight improvement in training accuracy.

Therefore,

larger values of

$$\alpha$$

produce

- smaller trees,
- lower variance,
- simpler decision rules.

Smaller values produce

- larger trees,
- higher flexibility,
- greater risk of overfitting.

Choosing the optimal value of

$$\alpha$$

is therefore one of the most important steps in post-pruning.

---

## 4.88 How Branches Are Removed

Imagine the following portion of a trained decision tree.

```text
                 Density > 5.6

               /              \

      Volume < 140         Band Gap > 2

      /        \            /         \

   Leaf      Leaf       Leaf       Leaf
```

After evaluating the complete tree,

the pruning algorithm determines that splitting on

```text
Band Gap > 2
```

provides almost no improvement in prediction accuracy.

Instead of keeping the unnecessary branch,

the algorithm removes it.

The simplified tree becomes

```text
                 Density > 5.6

               /              \

      Volume < 140           Leaf

      /        \

   Leaf      Leaf
```

Notice that

the prediction regions become slightly larger,

but the overall model becomes simpler and usually generalizes better.

The algorithm continues removing branches one at a time until further pruning would begin to decrease predictive performance.

---

## 4.89 Selecting the Optimal Pruned Tree

One important question remains.

How do we know how much pruning should be performed?

If we remove too few branches,

overfitting remains.

If we remove too many,

underfitting occurs.

Modern machine learning libraries solve this problem using **cross-validation**.

The algorithm typically generates many candidate trees,

each corresponding to a different value of

$$\alpha.$$

For each candidate,

cross-validation estimates predictive performance on unseen data.

The tree producing the highest validation performance is selected as the final model.

Conceptually,

the process becomes

```text
Large Tree

      │

Generate Many Pruned Trees

      │

Cross-Validation

      │

Compare Performance

      │

Choose Best Tree
```

Instead of selecting the largest tree or the smallest tree,

the algorithm selects the tree that demonstrates the strongest ability to generalize.

For materials informatics,

where datasets are often expensive to generate through experiments or Quantum ESPRESSO calculations,

this balance between predictive accuracy and model complexity is especially important because every training sample carries significant scientific value.

## 4.90 Pre-Pruning vs. Post-Pruning

Both pre-pruning and post-pruning aim to solve the same problem:

> **Prevent decision trees from becoming unnecessarily complex and overfitting the training data.**

However, they achieve this objective in fundamentally different ways.

Pre-pruning attempts to stop complexity **before** it appears.

Post-pruning allows complexity to develop first and then removes unnecessary parts afterward.

Their workflows can be summarized as

```text
Pre-Pruning

Grow Tree

↓

Check Stopping Rules

↓

Stop Early

↓

Final Tree
```

whereas

```text
Post-Pruning

Grow Complete Tree

↓

Evaluate Entire Structure

↓

Remove Weak Branches

↓

Final Tree
```

Although both approaches can produce excellent models, each has its own advantages and limitations.

---

### Comparing the Two Strategies

The major differences can be summarized as follows.

| Property | Pre-Pruning | Post-Pruning |
|----------|-------------|--------------|
| When pruning occurs | During tree construction | After the complete tree is built |
| Training time | Faster | Slower |
| Memory usage | Lower | Higher |
| Risk of underfitting | Higher | Lower |
| Risk of overfitting | Lower | Controlled after pruning |
| Ability to discover deeper relationships | Sometimes limited | Better |

Notice that neither approach is universally superior.

The best choice depends on

- dataset size,
- computational resources,
- prediction task,
- desired model complexity.

---

### Which Approach Is Used More Often?

Modern machine learning libraries commonly implement both strategies.

In practice,

many researchers combine them.

For example,

a tree may first be restricted using

```python
max_depth
```

and

```python
min_samples_leaf
```

while later applying

cost-complexity pruning

to simplify the remaining branches even further.

Combining both approaches often produces models that are computationally efficient while maintaining strong predictive performance.

---

## 4.91 Decision Trees in Materials Informatics

Throughout this chapter,

we have discussed decision trees using general examples.

It is now useful to examine how these models are actually employed in modern materials informatics research.

Decision trees rarely predict only one property.

Instead,

they can model a wide variety of materials problems depending on the available descriptors and target variables.

Typical regression tasks include predicting

- formation energy,
- bulk modulus,
- shear modulus,
- thermal conductivity,
- elastic modulus,
- density,
- melting temperature,
- Seebeck coefficient.

Typical classification tasks include predicting

- metallic versus semiconductor behavior,
- mechanically stable versus unstable materials,
- crystal family,
- magnetic versus non-magnetic materials,
- successful versus failed DFT calculations.

In every case,

the workflow remains exactly the same.

Only the target variable changes.

---

## 4.92 Example: Predicting Formation Energy

Suppose we wish to estimate the formation energy of oxide materials without performing new Density Functional Theory calculations.

Using Quantum ESPRESSO,

we first calculate formation energies for several hundred compounds.

Pymatgen is then used to extract structural information from the crystal structures.

For example,

we may generate descriptors such as

- density,
- average atomic mass,
- average electronegativity,
- unit cell volume,
- packing fraction,
- lattice parameters,
- coordination number.

These descriptors become the feature matrix

$$X.$$

The calculated formation energies become the target vector

$$y.$$

The complete workflow becomes

```text
Crystal Structures

↓

Quantum ESPRESSO

↓

Formation Energy

↓

Pymatgen

↓

Descriptors

↓

Decision Tree

↓

Formation Energy Prediction
```

After training,

the decision tree automatically discovers which descriptors are most useful for separating materials with different formation energies.

No explicit physical equations need to be programmed.

---

## 4.93 Example: Predicting Mechanical Stability

Decision trees are equally useful for classification problems.

Suppose we have a database containing thousands of crystal structures.

Each material has already been labeled as either

```text
Stable
```

or

```text
Unstable
```

using phonon calculations or elastic stability criteria.

Again,

Pymatgen extracts descriptors describing each structure.

The decision tree now learns rules such as

```text
IF

Density > 5.8

AND

Average Electronegativity > 2.3

↓

Predict

Stable
```

or

```text
IF

Packing Fraction < 0.45

AND

Average Atomic Radius > 1.7 Å

↓

Predict

Unstable
```

The remarkable feature is that these rules emerge automatically from the data.

The researcher never explicitly writes them.

---

## 4.94 Strengths of Decision Trees

Decision trees remain popular despite the existence of more sophisticated algorithms because they possess several important advantages.

### 1. Easy Interpretation

Every prediction can be traced through a sequence of human-readable decisions.

Researchers can understand

why

a particular prediction was made.

This transparency is particularly valuable in scientific research.

---

### 2. Nonlinear Modeling

Unlike Linear Regression,

decision trees naturally model nonlinear relationships.

Many materials properties exhibit highly nonlinear dependence on composition and crystal structure.

Decision trees capture these behaviors without requiring explicit nonlinear equations.

---

### 3. No Feature Scaling Required

Algorithms such as Linear Regression often benefit from feature scaling.

Decision trees,

however,

split data according to ordering rather than numerical magnitude.

Consequently,

standardization and normalization are generally unnecessary.

This simplifies preprocessing.

---

### 4. Mixed Data Types

Decision trees can work with

- continuous variables,
- integer variables,
- categorical variables.

This flexibility allows researchers to combine structural descriptors,

chemical information,

and categorical material classes within the same model.

---

### 5. Automatic Feature Selection

During training,

the algorithm automatically determines which descriptors provide the most informative splits.

Unimportant features naturally receive little or no influence on the final model.

---

## 4.95 Limitations of Decision Trees

Although decision trees are powerful,

they also possess important weaknesses.

Understanding these limitations is essential for selecting the appropriate algorithm.

### High Variance

Small changes in the training data may produce a completely different tree.

Consequently,

individual decision trees are often unstable.

---

### Overfitting

Without proper pruning,

decision trees readily memorize training data.

This is the primary motivation behind the pruning techniques discussed throughout this chapter.

---

### Axis-Aligned Splits

Decision trees split one feature at a time.

Each decision boundary is therefore parallel to one coordinate axis.

Very complicated relationships may require many successive splits before they can be approximated accurately.

---

### Lower Predictive Accuracy Than Ensembles

Although a single decision tree is interpretable,

its predictive performance is often inferior to ensemble methods such as

- Random Forest,
- Gradient Boosting,
- XGBoost.

These algorithms combine many trees to produce more accurate and stable predictions.

---

## 4.96 Looking Ahead

This chapter introduced the mathematical foundations that make decision trees one of the most important algorithms in machine learning.

We studied

- recursive partitioning,
- decision regions,
- leaf nodes,
- interpretable decision rules,
- overfitting,
- pre-pruning,
- post-pruning,
- cost-complexity pruning,
- practical applications in materials informatics.

A single decision tree is capable of solving surprisingly complex prediction problems.

However,

it also suffers from one significant weakness:

its predictions can change dramatically with small variations in the training dataset.

Researchers therefore asked an important question:

> **Instead of relying on one decision tree, what if we trained many different trees and combined their predictions?**

This simple idea leads to one of the most successful ensemble learning algorithms ever developed:

# Random Forest.

In the next chapter,

we will learn how combining hundreds of independently trained decision trees dramatically improves predictive accuracy, reduces variance, increases robustness, and forms the foundation for many modern machine learning models used in materials informatics.

# Chapter Summary

## 4.97 Key Concepts Learned in This Chapter

This chapter developed the theoretical foundation of decision trees from first principles. Rather than treating decision trees as black-box algorithms, we examined how they construct predictive models through recursive partitioning, measure node impurity mathematically, and generate interpretable decision rules.

The major concepts introduced include

- Tree-based learning
- Node purity and impurity
- Shannon Entropy
- Information Gain
- Gini Impurity
- Recursive Partitioning
- Decision Regions
- Leaf Nodes
- IF–THEN Decision Rules
- Overfitting
- Pre-Pruning
- Post-Pruning
- Cost-Complexity Pruning

Together, these concepts explain not only **how** a decision tree makes predictions, but also **why** different trees can exhibit dramatically different predictive performance.

---

## 4.98 Mathematical Concepts Covered

Throughout this chapter, several important mathematical formulations were introduced.

### Shannon Entropy

$$H(S)
=
-\sum_{i=1}^{n}
p_i
\log_2(p_i)$$

used to quantify uncertainty within a node.

---

### Information Gain

$$IG
=
H(\text{Parent})
-
\sum_i
\frac{|S_i|}{|S|}
H(S_i)$$

used to measure the reduction in uncertainty after splitting.

---

### Gini Impurity

$$G
=
1
-
\sum_i
p_i^2$$

used by the CART algorithm to estimate the probability of incorrect classification.

---

### Cost-Complexity Pruning

$$R_\alpha(T)
=
R(T)
+
\alpha|T|$$

which balances prediction accuracy with model complexity.

Although these equations appear different,

they all pursue the same objective:

> **Construct the simplest possible decision tree that produces accurate predictions on unseen data.**

---

## 4.99 Practical Skills Acquired

By completing this chapter, you should now be able to

- explain why decision trees can model nonlinear relationships,
- distinguish between classification trees and regression trees,
- calculate entropy, information gain, and Gini impurity,
- understand how recursive partitioning constructs an entire tree,
- interpret every path from the root node to a leaf node,
- explain why unrestricted trees overfit,
- describe the purpose of pre-pruning and post-pruning,
- understand the major hyperparameters controlling tree growth,
- explain why pruning improves generalization,
- identify situations where decision trees are appropriate for materials informatics research.

These abilities form the conceptual foundation for every tree-based ensemble algorithm studied later in this book.

---

## 4.100 Decision Trees in the Machine Learning Family

It is useful to place decision trees within the broader context of machine learning algorithms discussed throughout this book.

```text
Linear Regression

↓

Models Linear Relationships

↓

Decision Tree

↓

Models Nonlinear Relationships

↓

Random Forest

↓

Many Independent Trees

↓

Gradient Boosting

↓

Sequential Error Correction

↓

XGBoost

↓

Optimized Gradient Boosting
```

Notice that every algorithm appearing after the decision tree is built upon the same fundamental concept:

**learning from decision trees.**

The difference lies in

- how many trees are trained,
- how those trees interact,
- how predictions are combined,
- how optimization is performed.

Therefore,

a solid understanding of decision trees is essential before studying modern ensemble learning algorithms.

---

## 4.101 Bridge to Ensemble Learning

Imagine asking a difficult scientific question to only one researcher.

That researcher may provide an excellent answer,

or may make an unfortunate mistake.

Now imagine asking the same question to

100 independent experts

and averaging their conclusions.

Individual mistakes begin to cancel one another,

while consistent patterns become much clearer.

This simple principle is known as

# Ensemble Learning.

Instead of trusting one model,

ensemble methods combine multiple models to obtain a prediction that is generally more accurate, more stable, and more reliable.

Decision trees are particularly well suited for ensemble learning because

- they train quickly,
- they naturally capture nonlinear relationships,
- different trees often make different errors.

These properties make them ideal building blocks for larger predictive systems.

---

## 4.102 Looking Forward

The next chapter begins our study of **ensemble learning**, starting with one of the most influential algorithms in modern machine learning:

# Random Forest.

Rather than relying on a single decision tree,

Random Forest trains hundreds of different trees using random subsets of both the training samples and the available features.

We will examine

- why bootstrap sampling improves robustness,
- how random feature selection reduces correlation between trees,
- why averaging predictions dramatically lowers variance,
- how Out-of-Bag (OOB) evaluation works,
- why Random Forest often outperforms an individual decision tree,
- how feature importance can be extracted from an entire forest,
- and how Random Forest is used to predict complex materials properties such as formation energy, elastic modulus, thermal conductivity, and electronic properties.

By the end of the next chapter, you will understand why Random Forest has become one of the most widely used algorithms in both machine learning and materials informatics, serving as a bridge between simple interpretable models and more advanced ensemble methods such as Gradient Boosting and XGBoost.
