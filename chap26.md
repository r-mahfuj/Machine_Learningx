# Chapter 26 — Explainable AI for Materials

Modern Materials Informatics has moved from relatively simple descriptor-based machine-learning models toward increasingly complex models capable of learning directly from chemical compositions, crystal structures, graphs, images, spectra, and other scientific representations.

These models can achieve excellent predictive performance.

However, a prediction by itself is often not enough for scientific research.

Consider a machine-learning model trained to predict the band gap of inorganic crystalline materials. The model receives information about a crystal structure and produces:

```text
Predicted band gap = 2.84 eV
```

From an engineering perspective, this prediction may already be useful.

From a scientific perspective, however, several additional questions immediately arise:

```text
Why did the model predict 2.84 eV?

Which chemical features influenced the prediction?

Which atoms were important?

Which local structural environments mattered?

Which atomic interactions contributed to the result?

Did the model learn a physically meaningful relationship?

Could the explanation suggest a new materials-science hypothesis?
```

These questions motivate **Explainable Artificial Intelligence (XAI)**.

In Materials Informatics, explainability is particularly important because the ultimate objective is usually not simply to produce accurate numerical predictions. The objective is to use computational models to understand, design, optimize, and discover materials.

Therefore, the desired workflow is not merely:

```text
Materials
   ↓
Machine Learning
   ↓
Prediction
```

but rather:

```text
Materials
   ↓
Representation
   ↓
Machine-Learning Model
   ↓
Prediction
   ↓
Explanation
   ↓
Scientific Interpretation
   ↓
Materials Knowledge
```

The final stage is what makes explainable machine learning especially valuable in scientific research.

An explanation can help connect a mathematical model back to concepts that materials scientists already understand:

```text
Composition
      ↓
Electronic structure
      ↓
Bonding
      ↓
Crystal structure
      ↓
Local coordination
      ↓
Material property
```

The challenge is to determine whether the machine-learning model has learned relationships that are consistent with this scientific picture.

This chapter develops a systematic framework for doing exactly that.

The discussion begins with conventional feature attribution for descriptor-based models and progressively moves toward deep-learning explanations. The chapter then focuses heavily on crystal graph neural networks, where explanations can be mapped directly onto atoms, edges, and local structural environments.

The major methods developed throughout the chapter include:

* feature attribution
* permutation importance
* SHAP-style explanations
* attention visualization
* saliency maps
* Integrated Gradients
* node attribution
* edge attribution
* GNNExplainer
* crystal-graph visualization
* scientific interpretation and validation

The central objective is not simply to learn how to generate an explanation.

It is to learn how to **interpret that explanation scientifically**.

---

# 26.1 Introduction to Explainable AI in Materials Informatics

Explainable AI refers broadly to methods that help humans understand the behavior of machine-learning systems.

A useful distinction is between:

```text
Prediction
```

and:

```text
Explanation
```

Prediction asks:

```text
What will the model predict?
```

Explanation asks:

```text
Why did the model produce that prediction?
```

For example, suppose a model predicts the formation energy of a crystal:

```text
Input:
Crystal structure + chemical descriptors

Output:
Formation energy = -2.31 eV/atom
```

The prediction tells us the estimated property.

An explanation may additionally reveal:

```text
Strong positive contribution:
mean electronegativity

Strong negative contribution:
atomic-size mismatch

Moderate contribution:
density

Weak contribution:
cell volume
```

The exact meaning of these contributions depends on the explanation method, but the general purpose is to decompose or characterize the model's decision.

For a descriptor-based model, this may produce a list of important features.

For a crystal graph neural network, the explanation can become much more structural:

```text
Crystal
   ↓
Atoms
   ↓
Edges / neighborhoods
   ↓
GNN
   ↓
Prediction
   ↓
Node attribution
   ↓
Edge attribution
   ↓
Important structural motifs
```

This is a major reason why XAI becomes increasingly important as Materials Informatics moves toward graph-based deep learning.

---

## 26.1.1 The Black-Box Problem

Many modern machine-learning models are difficult to interpret directly.

Consider a simple linear model:

```text
y = β₀ + β₁x₁ + β₂x₂ + β₃x₃
```

The coefficients provide an immediate indication of how the input variables influence the prediction, assuming the features and scaling are interpreted appropriately.

A decision tree is also relatively transparent:

```text
if density < threshold:
    go left
else:
    go right
```

The decision path can often be inspected directly.

Now consider a deep neural network:

```text
Input
  ↓
Linear transformation
  ↓
Nonlinear activation
  ↓
Linear transformation
  ↓
Nonlinear activation
  ↓
...
  ↓
Prediction
```

or a crystal GNN:

```text
Crystal Graph
      ↓
Message Passing
      ↓
Node Embeddings
      ↓
Graph Representation
      ↓
Prediction
```

The internal computation may contain millions of learned parameters.

The model can therefore be highly accurate while remaining difficult to inspect manually.

This is commonly referred to as the **black-box problem**.

The purpose of XAI is not necessarily to make the entire model mathematically transparent.

Instead, XAI attempts to construct useful explanations of its behavior.

---

## 26.1.2 Accuracy Alone Is Not Sufficient

Consider two hypothetical materials models.

```text
Model A
RMSE = 0.12 eV

Model B
RMSE = 0.14 eV
```

Based only on RMSE, Model A appears superior.

However, suppose Model A makes predictions using complicated interactions that cannot be scientifically interpreted, while Model B consistently identifies chemically meaningful descriptors.

The choice becomes less obvious.

For scientific research, we may care about:

```text
Predictive accuracy
+
Generalization
+
Robustness
+
Interpretability
+
Physical consistency
```

This does not mean that an interpretable model should automatically be preferred over a more accurate model.

Rather, predictive performance and interpretability answer different questions.

Accuracy asks:

```text
How well does the model predict?
```

Interpretability asks:

```text
What information does the model use?
```

A strong research workflow evaluates both.

---

## 26.1.3 Explainability as a Scientific Tool

Explainability becomes especially valuable when it reveals relationships between the model and established materials-science concepts.

Suppose a model predicts thermal conductivity.

The model may identify strong contributions from descriptors related to:

```text
atomic mass
bond stiffness
density
coordination
```

These relationships may be consistent with known physical mechanisms.

This creates an opportunity to compare:

```text
Machine-Learning Explanation
            ↓
Materials-Science Knowledge
```

If the two agree, confidence in the interpretation may increase.

If they disagree, the disagreement itself can be scientifically interesting.

For example:

```text
Model:
Feature X is highly important.

Literature:
Feature X is not normally considered dominant.
```

This situation should not immediately be interpreted as a new discovery.

Several possibilities must first be investigated:

```text
correlated descriptors
dataset bias
data leakage
measurement artifacts
representation limitations
model artifacts
```

Only after careful validation should a surprising explanation be treated as a scientific hypothesis.

---

## 26.1.4 Global and Local Explanations

A central distinction in XAI is between **global** and **local** explanations.

### Global explanation

A global explanation attempts to characterize the model's behavior across many materials.

For example:

```text
Feature                         Importance

Mean electronegativity             0.29
Mean atomic radius                 0.21
Density                            0.16
Valence electron count             0.14
Average atomic number               0.10
Other features                      0.10
```

This suggests that the model frequently relies on certain descriptors.

The scientific question becomes:

```text
What general relationships has the model learned?
```

### Local explanation

A local explanation focuses on one material.

Suppose:

```text
Material:
SrTiO₃

Prediction:
Band gap = 3.15 eV
```

A local explanation asks:

```text
Why did the model predict 3.15 eV for this particular material?
```

The explanation may identify:

```text
Ti-related descriptors       → strong contribution
O-related descriptors        → strong contribution
local coordination           → moderate contribution
cell dimensions              → weak contribution
```

Local explanations are especially useful when analyzing individual candidates during materials discovery.

---

## 26.1.5 Global and Local Explanations Are Complementary

A research workflow should often use both.

Consider a dataset containing 50,000 materials.

A global explanation might show:

```text
electronegativity
atomic radius
valence electrons
```

as dominant features.

The researcher may then select one particular material and perform a local explanation.

The local explanation may reveal that:

```text
electronegativity → dominant
atomic radius → moderate
density → unexpectedly strong
```

This can reveal behavior that is hidden by a dataset-level average.

Therefore:

```text
Global XAI
    +
Local XAI
```

provides a more complete understanding than either alone.

---

## 26.1.6 Post-Hoc Explainability

Most of the methods developed in this chapter are **post-hoc explanation methods**.

The model is trained first.

```text
Training Dataset
      ↓
Model Training
      ↓
Trained Model
```

The trained model is then analyzed.

```text
Trained Model
      ↓
XAI Method
      ↓
Explanation
```

This has an important practical advantage.

The same trained model can potentially be analyzed using several explanation techniques.

For example:

```text
Trained GNN
    ├── Saliency
    ├── Integrated Gradients
    ├── Attention
    └── GNNExplainer
```

The resulting explanations can then be compared.

This is useful because no single XAI method should automatically be treated as the definitive representation of model reasoning.

---

## 26.1.7 Intrinsically Interpretable Versus Post-Hoc Models

Some models are naturally easier to interpret.

Examples include:

```text
Linear regression
Small decision tree
Simple rule-based model
```

Their internal structure itself provides information.

Other models require an additional explanation layer:

```text
Random Forest
Gradient Boosting
Deep Neural Network
Crystal GNN
```

The distinction can therefore be represented as:

```text
Intrinsic Interpretability

Model
 ↓
Direct interpretation
```

versus:

```text
Post-Hoc Interpretability

Model
 ↓
Prediction
 ↓
Explanation algorithm
 ↓
Interpretation
```

Both approaches are useful.

For Materials Informatics, however, post-hoc methods become particularly important because highly expressive models can capture complex nonlinear relationships that simpler models cannot.

---

## 26.1.8 Materials-Specific Challenges

Explainability in Materials Informatics has several complications that do not appear as strongly in ordinary tabular machine learning.

### Chemical feature correlation

Many materials descriptors are correlated.

For example:

```text
atomic number
atomic mass
atomic radius
electron count
```

may exhibit strong relationships across the periodic table.

If several correlated features contain similar information, attribution can be distributed among them.

Therefore:

```text
Feature A importance = 0.30
```

does not necessarily mean that Feature A alone contains the underlying scientific mechanism.

---

### Structural hierarchy

A crystal contains information at several levels:

```text
Atom
 ↓
Local environment
 ↓
Bond / interaction
 ↓
Coordination polyhedron
 ↓
Unit cell
 ↓
Crystal structure
```

An explanation method should ideally respect this hierarchy.

---

### Periodicity

Crystal structures are periodic.

An atom near one unit-cell boundary may interact with an atom represented on the opposite boundary.

Therefore, explanations based on crystal graphs must account correctly for periodic neighbor relationships.

---

### Symmetry

Equivalent atoms may have equivalent structural roles.

A physically meaningful explanation should not incorrectly distinguish symmetry-equivalent sites merely because of an arbitrary indexing choice.

---

### Multiple length scales

Some properties are dominated by local environments, while others depend on longer-range interactions.

Therefore:

```text
Local explanation
```

may not always capture the complete physical mechanism.

This becomes an important consideration when interpreting GNNs.

---

# 26.1.9 The Explainable Materials ML Workflow

A general XAI workflow can now be defined.

```text
Materials Dataset
       ↓
Data Cleaning
       ↓
Materials Representation
       ↓
Model Training
       ↓
Model Validation
       ↓
Prediction
       ↓
Explanation Method
       ↓
Attribution
       ↓
Visualization
       ↓
Scientific Interpretation
       ↓
Independent Validation
```

The last step is particularly important.

An explanation is not automatically a scientific truth.

It is evidence about how the trained model behaves.

Therefore:

```text
XAI Explanation
      ≠
Physical Law
```

Instead:

```text
XAI Explanation
      ↓
Potential Scientific Hypothesis
      ↓
Independent Validation
      ↓
Scientific Conclusion
```

This principle will remain important throughout the chapter.

---

# 26.1.10 A Simple Materials Example

Consider a model predicting band gap from five descriptors:

```text
x₁ = mean_atomic_number
x₂ = mean_electronegativity
x₃ = mean_atomic_radius
x₄ = density
x₅ = volume
```

The model predicts:

```text
ŷ = 2.75 eV
```

An explanation algorithm may produce:

```text
Baseline prediction       1.90 eV

mean_atomic_number       +0.31 eV
mean_electronegativity   +0.42 eV
mean_atomic_radius       +0.18 eV
density                  -0.04 eV
volume                   -0.02 eV
```

The contributions approximately reconstruct the final prediction:

```text
1.90
+ 0.31
+ 0.42
+ 0.18
- 0.04
- 0.02
≈ 2.75 eV
```

The researcher can now investigate the result.

For example:

```text
Why is electronegativity strongly associated
with the prediction?

Is this relationship physically reasonable?

Is electronegativity correlated with another
descriptor?

Does the relationship remain in an independent dataset?

Does DFT support the inferred trend?
```

This is the mindset required for scientific XAI.

---

# 26.1.11 From Explanation to Hypothesis

The most valuable use of explainability is not simply producing a colorful plot.

The ultimate objective is to transform model behavior into a testable scientific idea.

The workflow is:

```text
Model Prediction
       ↓
Attribution
       ↓
Pattern
       ↓
Scientific Interpretation
       ↓
Hypothesis
       ↓
DFT / Experiment / Literature
       ↓
Validation
```

For example:

```text
GNN
 ↓
Strong attribution to Ti–O environments
 ↓
Local Ti–O coordination appears important
 ↓
Hypothesis:
local coordination influences the target property
 ↓
Test using controlled structural perturbations
 ↓
DFT validation
```

This is where explainable AI becomes genuinely useful to Materials Informatics research.

---

# 26.1.12 The Progression of This Chapter

The remainder of the chapter will progressively move from simple feature explanations toward structural explanations.

The progression is:

```text
Classical ML
     ↓
Feature Attribution
     ↓
Permutation Importance
     ↓
SHAP
     ↓
Deep Neural Networks
     ↓
Saliency Maps
     ↓
Integrated Gradients
     ↓
Attention Visualization
     ↓
Crystal Graph Neural Networks
     ↓
Node Importance
     ↓
Edge Importance
     ↓
GNNExplainer
     ↓
Crystal Visualization
     ↓
Scientific Interpretation
     ↓
Validation
     ↓
Materials Discovery
```

This progression is deliberate.

A researcher who understands feature attribution for a descriptor-based model will have a much stronger conceptual foundation for understanding attribution on a crystal graph.

The fundamental question remains the same:

```text
Which parts of the input contributed to the prediction?
```

Only the representation changes.

For tabular materials ML:

```text
Input = descriptors
```

For a crystal GNN:

```text
Input = nodes + edges + features + geometry
```

Therefore, the explanation must evolve accordingly.

---

# 26.1.13 Chapter Objective

By the end of this chapter, the reader should be able to move from:

```text
"This GNN predicts the property accurately."
```

to a much more scientifically useful statement:

```text
"This GNN predicts the property accurately,
and its explanation indicates that specific
atomic environments and interatomic interactions
are strongly associated with the prediction.
These patterns can be visualized on the crystal
structure, tested for stability, compared with
known materials-science principles, and used to
generate experimentally or computationally
testable hypotheses."
```

That is the central purpose of Explainable AI for Materials Informatics.

The goal is therefore not merely to **open the black box**.

The goal is to turn the model's behavior into **scientifically interpretable evidence**.

# 26.2 Why Interpretability Matters in Materials Science

The importance of interpretability becomes clearer when machine learning is treated not merely as a prediction engine but as a component of the scientific research process.

In conventional supervised learning, the primary objective is often expressed as minimizing a prediction error:

```text
Training data
      ↓
Model
      ↓
Prediction
      ↓
Loss
      ↓
Optimization
```

After training, the model is evaluated on previously unseen materials.

If the model achieves sufficiently low error, it may be considered successful from a predictive perspective.

Materials science, however, usually has a deeper objective.

A researcher may want to know:

```text
Which chemical factors control the property?

Which structural motifs are favorable?

Why does one material outperform another?

Which atoms are responsible for the observed behavior?

Which structural changes should be tested next?

Can the model reveal a previously unknown structure–property relationship?
```

Explainable AI provides computational tools for investigating these questions.

It therefore adds another dimension to Materials Informatics:

```text
Prediction
    +
Interpretation
    =
Scientific Understanding
```

This does not mean that an explanation automatically provides a physical theory. Rather, it provides evidence about what information the trained model is using and how that information contributes to its predictions.

---

## 26.2.1 Trust in Machine-Learning Predictions

One of the first reasons interpretability matters is **trust**.

Suppose a machine-learning model predicts:

```text
Material A:
Predicted formation energy = -3.42 eV/atom
```

The researcher may want to use this prediction to decide whether Material A deserves a computational investigation.

Before doing so, several questions arise:

```text
Has the model encountered chemically similar materials?

Is the material inside the training distribution?

Which features drive the prediction?

Is the prediction consistent with chemically related materials?

Does the model rely on physically meaningful information?
```

A prediction without any explanation can make these questions difficult to answer.

An explanation does not eliminate uncertainty, but it provides additional information for evaluating the prediction.

For example:

```text
Prediction:
-3.42 eV/atom

Important features:
chemical composition
coordination environment
density
local structural descriptors
```

This is more informative than the numerical prediction alone.

---

## 26.2.2 Trust Does Not Mean Blind Acceptance

Explainability should not be confused with proof of correctness.

A model can produce a convincing-looking explanation and still be wrong.

For example, suppose a model consistently assigns high importance to density when predicting a materials property.

This may appear physically reasonable.

However, density could simply be correlated with another variable that actually carries the predictive information.

Therefore:

```text
Explanation
    ≠
Proof
```

Instead:

```text
Explanation
    ↓
Evidence about model behavior
    ↓
Scientific investigation
    ↓
Independent validation
```

This distinction is fundamental to responsible use of XAI in scientific research.

---

## 26.2.3 Detecting Spurious Correlations

One of the most important applications of interpretability is identifying **spurious correlations**.

A model may achieve excellent validation performance by learning relationships that are statistically useful but scientifically irrelevant.

Consider a dataset containing:

```text
Material composition
Crystal structure
Target property
Database source
Calculation method
```

Suppose almost all materials from one database source belong to a particular chemical family.

A machine-learning model might accidentally learn:

```text
Database source
      ↓
Chemical family
      ↓
Target property
```

rather than learning the underlying physical relationship.

If the database identifier is accidentally included as a feature, feature attribution may reveal that the identifier has unusually high importance.

That is a warning sign.

---

## 26.2.4 A Simple Example of a Spurious Feature

Suppose a dataset contains:

```python
features = [
    "mean_atomic_number",
    "mean_electronegativity",
    "density",
    "database_id"
]
```

A model is trained normally:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)
```

Feature importance can then be inspected:

```python
import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance)
```

Suppose the result is:

```text
database_id              0.41
mean_atomic_number       0.25
density                   0.18
mean_electronegativity    0.16
```

This should immediately trigger investigation.

A database identifier should normally have no intrinsic physical meaning.

The model may therefore be exploiting a dataset artifact.

Without interpretability, the model could appear successful while hiding this problem.

---

## 26.2.5 Identifying Dataset Bias

Materials datasets are rarely perfectly uniform.

A dataset may contain:

```text
more oxides than sulfides
more stable compounds than metastable compounds
more experimentally studied materials than unexplored materials
more materials from certain chemical families
more structures calculated using one computational protocol
```

Such biases can influence the learned model.

Suppose a model is trained primarily on oxide materials.

It may perform extremely well on:

```text
new oxide
```

but poorly on:

```text
new nitride
```

or:

```text
new chalcogenide
```

An explanation can help reveal whether the model is relying on broad physical descriptors or on patterns strongly associated with the dominant chemical family.

This becomes particularly important when applying Materials Informatics models to unexplored chemical spaces.

---

## 26.2.6 Explainability for Model Validation

Model validation traditionally focuses on numerical metrics:

```text
MAE
RMSE
R²
```

These remain essential.

However, a research-grade validation process can be expanded:

```text
Predictive validation
        +
Data-distribution analysis
        +
Explanation analysis
        +
Physical consistency
```

For example:

```text
Model performance:
RMSE = 0.18 eV

Explanation:
electronic descriptors dominate

Physical interpretation:
consistent with expected electronic behavior

Independent test:
relationship remains stable
```

This provides stronger evidence than the error metric alone.

---

## 26.2.7 Detecting Dataset Leakage

Interpretability can also assist in identifying **data leakage**.

Data leakage occurs when information that should not be available to the model during prediction is indirectly included in the input.

Consider a target property:

```text
formation energy
```

and a feature:

```text
formation_energy_related_descriptor
```

If the feature was generated using information that already contains the target, the model may appear extremely accurate.

Feature attribution could reveal that this feature dominates the prediction.

The workflow should then be:

```text
Unexpectedly important feature
            ↓
Trace feature generation
            ↓
Check target dependence
            ↓
Check data pipeline
            ↓
Remove leakage if present
            ↓
Retrain model
```

Thus XAI can become part of the data-quality audit.

---

## 26.2.8 Scientific Hypothesis Generation

Perhaps the most interesting role of explainability is **hypothesis generation**.

Suppose a model is trained to predict ionic conductivity.

After explanation, the researcher observes:

```text
High model attribution
        ↓
Large local structural distortion
        ↓
Particular coordination environment
        ↓
Possible connection to ion migration
```

This does not establish a mechanism.

But it suggests a hypothesis:

> The local structural environment may influence ion migration and therefore ionic conductivity.

The researcher can then test the hypothesis using:

```text
DFT
molecular dynamics
experimental characterization
literature analysis
```

This creates a scientific feedback loop:

```text
ML
 ↓
Explanation
 ↓
Hypothesis
 ↓
Scientific test
 ↓
Validation / rejection
 ↓
New knowledge
```

This is one of the strongest reasons to integrate XAI into Materials Informatics.

---

## 26.2.9 From Feature Importance to Structure–Property Relationships

Materials science is fundamentally concerned with relationships between structure, composition, processing, and properties.

A simplified representation is:

```text
Composition
     ↓
Atomic interactions
     ↓
Crystal structure
     ↓
Electronic / vibrational behavior
     ↓
Macroscopic property
```

Machine learning may learn a complicated approximation to this relationship.

Explainability attempts to identify which parts of the input representation are most relevant to that approximation.

For descriptor-based models:

```text
Descriptor
    ↓
Attribution
    ↓
Chemical interpretation
```

For crystal GNNs:

```text
Atom
 ↓
Node attribution

Bond / interaction
 ↓
Edge attribution

Local environment
 ↓
Subgraph attribution
```

The interpretation therefore becomes increasingly structural.

---

## 26.2.10 Example: Predicting Formation Energy

Consider a model trained using composition and structural descriptors.

Suppose a local explanation for Material A gives:

```text
Feature                         Contribution

Mean electronegativity              -0.41
Atomic-size mismatch                +0.17
Density                             -0.13
Coordination descriptor             -0.08
Cell volume                         +0.03
```

The researcher should not simply conclude:

```text
Electronegativity causes formation energy.
```

A more scientifically appropriate interpretation is:

```text
The trained model relies strongly on
mean electronegativity when predicting
formation energy for this material.
```

The next question is:

```text
Why?
```

Possible explanations include:

```text
bonding characteristics
chemical stability trends
correlation with composition
correlation with atomic-size mismatch
dataset-specific patterns
```

Additional analysis is therefore required.

---

## 26.2.11 Cross-Checking an XAI Observation

Suppose XAI identifies a descriptor as highly important.

A robust scientific workflow can investigate the observation using several independent approaches.

### Step 1 — Attribution

```text
Feature X
   ↓
High importance
```

### Step 2 — Statistical analysis

Examine:

```text
correlation
conditional dependence
distribution
chemical-family dependence
```

### Step 3 — Model comparison

Train different models:

```text
Random Forest
Gradient Boosting
Neural Network
```

and determine whether Feature X remains important.

### Step 4 — Dataset perturbation

Remove Feature X and retrain:

```python
X_without_feature = X.drop(
    columns=["feature_X"]
)

model.fit(
    X_without_feature,
    y
)
```

Then compare performance.

### Step 5 — Scientific validation

Investigate whether the observed relationship agrees with:

```text
known theory
literature
DFT calculations
experiments
```

This transforms an explanation from a visualization into a research procedure.

---

## 26.2.12 Explanation Stability

An explanation should ideally be reasonably stable.

Suppose the same material is explained five times:

```text
Run 1 → Feature A dominant
Run 2 → Feature A dominant
Run 3 → Feature A dominant
Run 4 → Feature B dominant
Run 5 → Feature C dominant
```

This instability may indicate that the explanation is sensitive to:

```text
random initialization
model parameters
training subset
feature correlations
noise
explanation method
```

A useful research question is therefore:

```text
Would a small perturbation of the model
produce the same scientific explanation?
```

Explanation stability will become important later when evaluating XAI methods systematically.

---

## 26.2.13 Repeated Explanations Across Models

Suppose three independently trained models produce:

```text
Model 1 → Feature A, B
Model 2 → Feature A, B
Model 3 → Feature A, C
```

Feature A appears consistently.

This provides stronger evidence that Feature A is genuinely important to the predictive task than if only one model identifies it.

A practical workflow is:

```text
Train multiple models
        ↓
Generate explanations
        ↓
Rank features
        ↓
Measure agreement
        ↓
Identify stable features
```

For example:

```python
rankings = []

for model in models:

    importance = get_importance(
        model,
        X_test
    )

    ranking = importance.sort_values(
        ascending=False
    )

    rankings.append(ranking)
```

The rankings can then be compared.

This is particularly useful when the final scientific conclusion depends on the interpretation.

---

## 26.2.14 Interpretability and Materials Discovery

Interpretability becomes even more valuable during materials discovery.

Suppose an active-learning or generative workflow produces:

```text
Candidate A
Candidate B
Candidate C
Candidate D
```

A model predicts:

```text
Candidate A → high performance
Candidate B → high performance
Candidate C → moderate performance
Candidate D → high performance
```

Prediction alone identifies promising candidates.

XAI can provide another layer:

```text
Candidate A
    ↓
Important chemical features

Candidate B
    ↓
Important structural features

Candidate D
    ↓
Important local atomic environment
```

The researcher can then compare not only:

```text
Which candidate is best?
```

but also:

```text
Why are these candidates predicted to be good?
```

This can reveal whether multiple candidates exploit the same scientific mechanism or whether they represent different routes to the desired property.

---

## 26.2.15 Explainability in Human-in-the-Loop Research

A Materials Informatics workflow often involves a human researcher between computational steps.

For example:

```text
ML Model
   ↓
Candidate Ranking
   ↓
Researcher
   ↓
DFT / Experiment
```

An explanation can become part of the information presented to the researcher:

```text
Candidate
Prediction
Uncertainty
Important features
Important atoms
Important bonds
```

The researcher can then combine:

```text
model prediction
+
uncertainty
+
scientific explanation
+
domain knowledge
```

to make a decision.

This is especially important when the model operates in chemical spaces where purely numerical ranking may not be sufficient.

---

## 26.2.16 Interpretability for DFT-Guided Research

DFT can be used not only to generate training labels but also to test hypotheses suggested by XAI.

Suppose a GNN explanation identifies a local coordination environment as important for a target property.

The researcher can construct related structures:

```text
Original structure
       ↓
Controlled structural modification
       ↓
DFT calculation
       ↓
Property comparison
```

For example:

```text
Coordination = 4
        vs
Coordination = 5
        vs
Coordination = 6
```

If the property changes consistently with the XAI prediction, confidence in the proposed relationship increases.

Thus:

```text
GNN
 ↓
XAI
 ↓
Structural hypothesis
 ↓
DFT
 ↓
Validation
```

creates a powerful computational discovery workflow.

---

## 26.2.17 Interpretability for Experimental Research

The same principle applies to experimental materials science.

Suppose XAI indicates that a particular structural feature is associated with high catalytic activity.

The researcher may design experiments that deliberately modify that feature.

For example:

```text
Change composition
       ↓
Modify local coordination
       ↓
Synthesize material
       ↓
Characterize structure
       ↓
Measure catalytic activity
```

The experiment then tests a hypothesis suggested by the model.

This is fundamentally different from using ML only as a screening tool.

Instead of:

```text
ML → prediction
```

the workflow becomes:

```text
ML → explanation → hypothesis → experiment
```

---

## 26.2.18 What an Explanation Should Ultimately Provide

A useful materials-science explanation should ideally answer several questions.

### Question 1

```text
What information influenced the prediction?
```

### Question 2

```text
How strongly did each relevant input contribute?
```

### Question 3

```text
Is the explanation stable?
```

### Question 4

```text
Is the explanation consistent with known science?
```

### Question 5

```text
Can the explanation be independently tested?
```

### Question 6

```text
Does it suggest a useful scientific hypothesis?
```

This creates a hierarchy:

```text
Attribution
    ↓
Interpretation
    ↓
Physical reasoning
    ↓
Hypothesis
    ↓
Validation
```

The final stages are the most scientifically valuable.

---

## 26.2.19 The Role of XAI in a Modern Materials Informatics Pipeline

The ideas developed so far can be integrated into the broader Materials Informatics workflow.

```text
                    Materials Space
                          ↓
                  Candidate Materials
                          ↓
              Pymatgen / Matminer / Graph
                          ↓
                    ML / GNN Model
                          ↓
             ┌────────────┴────────────┐
             ↓                         ↓
        Prediction                 Uncertainty
             ↓                         ↓
             └────────────┬────────────┘
                          ↓
                     XAI Analysis
                          ↓
              Feature / Node / Edge
                    Attribution
                          ↓
                Scientific Interpretation
                          ↓
                  DFT / Experiment
                          ↓
                 Scientific Validation
```

This architecture shows that XAI does not replace prediction.

It sits alongside prediction and uncertainty to provide an additional scientific perspective.

---

## 26.2.20 Transition to Feature Attribution

The first practical family of explainability techniques considered in this chapter is **feature attribution**.

This is the natural starting point because many Materials Informatics workflows begin with descriptor vectors.

For example:

```text
Composition
    ↓
Pymatgen
    ↓
Matminer
    ↓
Descriptor Vector
    ↓
ML Model
```

A descriptor vector might contain:

```text
mean atomic number
mean atomic mass
mean electronegativity
mean atomic radius
density
volume
fractional composition
coordination descriptors
```

The model converts these features into a prediction.

Feature attribution attempts to determine how those features influence the prediction.

The progression will therefore be:

```text
Materials Descriptors
        ↓
Feature Attribution
        ↓
Global Importance
        ↓
Local Importance
        ↓
Permutation Importance
        ↓
SHAP
        ↓
Scientific Interpretation
```

After establishing this foundation, the chapter will move toward neural-network explanations and ultimately to the more difficult problem of interpreting **atoms and interatomic interactions in crystal graph neural networks**.

# 26.3 Feature Attribution for Materials ML

Feature attribution is one of the most fundamental approaches to explainable machine learning.

The central idea is straightforward:

> If a model receives several input features and produces a prediction, how much did each feature contribute to that prediction?

For Materials Informatics, this question is particularly useful because many traditional machine-learning workflows represent materials using numerical descriptors.

A material may be represented as:

```text
Composition
    ↓
Chemical descriptors
    ↓
Structural descriptors
    ↓
Electronic descriptors
    ↓
Numerical feature vector
```

For example, a single material might be represented by:

```text
x =
[
    mean_atomic_number,
    mean_atomic_mass,
    mean_electronegativity,
    mean_atomic_radius,
    density,
    volume,
    ...
]
```

A trained model then computes:

```text
x → f(x) → predicted property
```

Feature attribution attempts to open this mapping.

Instead of reporting only:

```text
Prediction = 2.73 eV
```

we want something closer to:

```text
Prediction = 2.73 eV

Important contributions:
    electronegativity → strong contribution
    atomic radius     → moderate contribution
    density            → moderate contribution
    volume             → weak contribution
```

The precise mathematical interpretation depends on the attribution method.

This distinction is important because different XAI techniques answer slightly different questions.

---

## 26.3.1 The Basic Attribution Problem

Let a trained machine-learning model be represented by:

```text
ŷ = f(x)
```

where:

* `x` is the input feature vector,
* `f` is the trained model,
* `ŷ` is the prediction.

Suppose the feature vector contains `n` features:

```text
x = [x₁, x₂, ..., xₙ]
```

Feature attribution attempts to determine a set of contributions:

```text
φ₁, φ₂, ..., φₙ
```

such that the prediction can be interpreted in terms of those contributions.

A generic decomposition can be written as:

```text
Prediction
=
Reference prediction
+
Feature contributions
```

or:

```text
f(x)
≈
f(reference)
+
Σ φᵢ
```

The exact equality, approximation, and interpretation depend on the particular XAI framework.

The reference prediction is important.

For example, if the average band gap in a dataset is:

```text
2.10 eV
```

and the model predicts:

```text
3.00 eV
```

an attribution method may explain how the input features move the prediction away from the reference value.

Conceptually:

```text
Reference
   ↓
2.10 eV
   ↓
Feature contributions
   ↓
3.00 eV
```

This provides a much more informative interpretation than simply knowing the final number.

---

# 26.3.2 Positive and Negative Contributions

Feature attribution often assigns signed contributions.

For example:

```text
Feature                  Contribution

Electronegativity           +0.43
Atomic radius               +0.17
Density                     -0.12
Volume                      -0.04
Valence electrons           +0.09
```

The signs indicate the direction of the contribution relative to the reference under the particular explanation convention.

A positive contribution may push the prediction upward.

A negative contribution may push the prediction downward.

For example:

```text
Reference prediction = 2.10 eV

Electronegativity   +0.43
Atomic radius       +0.17
Density             -0.12
Volume              -0.04
Valence electrons   +0.09
```

The resulting prediction is influenced by all of these contributions.

However, a researcher should not immediately interpret:

```text
positive attribution
```

as:

```text
physically increases the property
```

without considering the model, baseline, feature scaling, correlations, and explanation method.

Attribution is first a statement about **model behavior**.

Scientific interpretation comes afterward.

---

# 26.3.3 Global Feature Attribution

A global explanation attempts to describe the behavior of the model across a population of materials.

Suppose a model uses:

```text
20 materials descriptors
```

A global analysis may produce:

```text
Feature                       Mean importance

Mean electronegativity            0.24
Mean atomic radius                0.19
Density                           0.15
Mean atomic number                0.13
Volume                            0.09
Valence electrons                 0.08
Other features                    0.12
```

This suggests that the model frequently relies on the first few descriptors.

The important word is **frequently**.

It does not necessarily mean that every material is explained by exactly the same features.

One material might be dominated by electronegativity.

Another might be dominated by structural descriptors.

Therefore, global attribution describes the model's general behavior rather than any single prediction.

---

# 26.3.4 Local Feature Attribution

Local attribution focuses on a single material or a small subset of materials.

Suppose:

```text
Material = MgAl₂O₄
```

and the model predicts:

```text
Predicted formation energy = -1.87 eV/atom
```

A local explanation might show:

```text
Mean electronegativity      -0.32
Atomic-radius mismatch      +0.19
Density                     -0.11
Mean atomic number          -0.08
Volume                      +0.04
```

Another material may produce a completely different attribution pattern.

This is extremely useful in materials discovery because researchers frequently care about individual candidates rather than only the average behavior of the entire dataset.

For example:

```text
Candidate A
   ↓
Why is it predicted to have high stability?

Candidate B
   ↓
Why is it predicted to have high conductivity?

Candidate C
   ↓
Why does the model consider it unusual?
```

Local explanations can address these questions.

---

# 26.3.5 Global Versus Local Attribution

The distinction can be summarized as:

```text
Global Attribution

Dataset
   ↓
Model
   ↓
Average / aggregated attribution
   ↓
General model behavior
```

while:

```text
Local Attribution

One material
   ↓
Model
   ↓
Individual attribution
   ↓
Material-specific explanation
```

Both should be used when possible.

A useful research workflow is:

```text
Global analysis
      ↓
Identify important variables
      ↓
Select scientifically interesting materials
      ↓
Local analysis
      ↓
Understand individual predictions
```

This creates a bridge between population-level statistical analysis and material-specific investigation.

---

# 26.3.6 Feature Attribution for a Linear Model

The simplest case is linear regression.

Suppose:

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

The coefficient `βᵢ` determines the change in prediction associated with a change in feature `xᵢ`, assuming the other variables remain fixed.

For example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(
    X_train,
    y_train
)
```

The coefficients can be inspected:

```python
import pandas as pd

coefficients = pd.Series(
    model.coef_,
    index=feature_names
)

print(coefficients)
```

A possible output is:

```text
mean_electronegativity     0.84
mean_atomic_radius        -0.41
density                    0.27
volume                    -0.12
```

For a linear model, this is relatively easy to interpret.

However, coefficient magnitude depends on feature scale.

For example:

```text
Feature A:
range = 0–1

Feature B:
range = 0–1000
```

A coefficient of:

```text
0.5
```

for one feature cannot necessarily be compared directly with:

```text
0.1
```

for another feature unless their scales are appropriately considered.

This is one reason standardized features can be useful when interpreting linear models.

---

# 26.3.7 Standardized Linear Models

A standardization workflow can be implemented using scikit-learn:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(
    X_train,
    y_train
)
```

The coefficients now correspond to standardized feature changes.

They can be extracted as:

```python
regressor = model.named_steps["regressor"]

coefficients = pd.Series(
    regressor.coef_,
    index=feature_names
)

print(
    coefficients.sort_values(
        ascending=False
    )
)
```

This makes coefficient magnitude more useful for comparison.

Nevertheless, even standardized coefficients should not be interpreted as causal effects.

The model still represents a statistical relationship learned from the dataset.

---

# 26.3.8 Feature Attribution for Decision Trees

Decision trees provide another relatively interpretable model.

A tree repeatedly divides the feature space.

Conceptually:

```text
                Density < threshold?
                    /        \
                  Yes         No
                  /            \
        Electronegativity    Radius
             /     \          /    \
           ...     ...      ...    ...
```

A prediction can therefore be associated with a sequence of decisions.

For example:

```text
Density < 4.5
       ↓
Electronegativity > 2.1
       ↓
Atomic radius < 1.3
       ↓
Prediction = 2.8 eV
```

This provides a local explanation in the form of a decision path.

Using scikit-learn:

```python
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree.fit(
    X_train,
    y_train
)
```

A particular sample can be traced through the tree:

```python
decision_path = tree.decision_path(
    X_test.iloc[[0]]
)

print(decision_path)
```

The tree structure can also be visualized:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 10))

plot_tree(
    tree,
    feature_names=feature_names,
    filled=True
)

plt.show()
```

Small trees can therefore provide an intuitive explanation.

However, very deep trees become difficult to interpret.

---

# 26.3.9 Feature Importance in Random Forests

Random Forests consist of many decision trees.

Their ensemble structure improves predictive performance but makes direct interpretation more difficult.

A Random Forest exposes impurity-based feature importance:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)
```

Feature importance can then be calculated:

```python
importance = pd.Series(
    rf.feature_importances_,
    index=feature_names
)

importance = importance.sort_values(
    ascending=False
)

print(importance)
```

A simple visualization is:

```python
import matplotlib.pyplot as plt

top_features = importance.head(15)

top_features.sort_values().plot(
    kind="barh",
    figsize=(8, 6)
)

plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
```

This provides a global view of the model.

But Random Forest feature importance has important limitations.

---

# 26.3.10 The Problem with Impurity-Based Importance

Impurity-based importance measures how much a feature contributes to reducing impurity across the ensemble's decision nodes.

This can be useful, but it is not a perfect measure of scientific importance.

Problems can arise when:

```text
features are correlated
```

or when:

```text
features have different numbers of possible split points
```

For example:

```text
Feature A
Feature B
```

may contain nearly identical information.

The model might use Feature A frequently in one set of trees and Feature B in another.

The resulting importances may be:

```text
Feature A = 0.18
Feature B = 0.16
```

even though the underlying information represented by the two features is closely related.

Alternatively, one feature may receive most of the importance while its correlated partner receives very little.

Therefore:

```text
Low feature importance
```

does not necessarily mean:

```text
Feature contains no useful scientific information.
```

The information may simply be represented elsewhere.

---

# 26.3.11 Permutation Feature Importance

Permutation importance provides a model-agnostic alternative.

The method begins with the original test set.

Suppose the model has a baseline performance:

```text
MAE = 0.18 eV
```

Now one feature is randomly shuffled.

For example:

```text
Original:

Density:
3.2
4.1
5.3
6.2
...

Permuted:

Density:
5.3
3.2
6.2
4.1
...
```

The model is then evaluated again.

If performance deteriorates substantially:

```text
Original MAE = 0.18
Permuted MAE = 0.42
```

then the feature was important to the model.

If performance barely changes:

```text
Original MAE = 0.18
Permuted MAE = 0.19
```

the model was not strongly dependent on that feature under the tested conditions.

---

# 26.3.12 Implementing Permutation Importance

Scikit-learn provides a direct implementation:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="neg_mean_absolute_error"
)
```

The mean importance can be extracted:

```python
perm_importance = pd.Series(
    result.importances_mean,
    index=feature_names
)

perm_importance = perm_importance.sort_values(
    ascending=False
)

print(perm_importance)
```

The standard deviation can also be inspected:

```python
perm_std = pd.Series(
    result.importances_std,
    index=feature_names
)

print(perm_std)
```

A useful table is therefore:

```python
importance_table = pd.DataFrame({
    "mean_importance": result.importances_mean,
    "std_importance": result.importances_std
}, index=feature_names)

importance_table = importance_table.sort_values(
    "mean_importance",
    ascending=False
)

print(importance_table)
```

This is much more informative than reporting only a single importance number.

---

# 26.3.13 Interpreting Permutation Importance Correctly

Suppose we obtain:

```text
Feature                   Importance     Std

electronegativity           0.31         0.03
atomic radius               0.22         0.04
density                     0.11         0.02
volume                      0.02         0.01
```

A reasonable interpretation is:

> Randomizing electronegativity substantially reduces model performance, suggesting that the trained model relies strongly on this feature.

The statement should not be:

> Electronegativity is physically responsible for the property.

The first statement concerns model behavior.

The second statement claims a physical mechanism.

The distinction is critical.

---

# 26.3.14 Negative Permutation Importance

Permutation importance can sometimes produce negative values.

For example:

```text
Feature A = +0.21
Feature B = +0.09
Feature C = -0.03
```

A negative value means that shuffling the feature improved the measured performance in that particular evaluation.

This may happen because of:

```text
sampling variability
feature redundancy
noise
finite test-set size
model instability
```

It does not automatically mean that the feature has a physically negative effect.

Therefore, negative permutation importance should be investigated rather than directly interpreted as a scientific conclusion.

---

# 26.3.15 Repeated Permutation Experiments

A single permutation is not sufficient because randomization itself introduces variability.

That is why the parameter:

```python
n_repeats=20
```

or a larger value is useful.

A more detailed experiment can use:

```python
result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=50,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
```

The resulting distribution can be analyzed rather than only its mean.

For a feature `j`:

```text
Importance distribution:
      |----|----|----|----|----|
      low                 high
```

A narrow distribution indicates relatively stable importance.

A broad distribution indicates greater sensitivity to the permutation process.

---

# 26.3.16 Feature Attribution with Correlated Materials Descriptors

Correlated features represent one of the most important problems in Materials Informatics.

Consider descriptors:

```text
atomic number
atomic mass
periodic-table position
valence electron count
```

These are not independent quantities.

Similarly:

```text
unit-cell volume
density
lattice parameters
```

may be strongly related.

If a model relies on several correlated features, attribution can become ambiguous.

Suppose:

```text
Feature A = 0.30
Feature B = 0.05
```

It would be incorrect to conclude automatically that:

```text
Feature A is six times more scientifically important.
```

Feature B may contain information that overlaps strongly with Feature A.

A better analysis investigates:

```text
feature correlation
conditional relationships
feature groups
model stability
```

---

# 26.3.17 Grouped Feature Interpretation

One practical approach is to interpret correlated features as a group.

For example:

```text
Atomic-property group:

atomic number
atomic mass
atomic radius
electronegativity
```

and:

```text
Structural group:

density
volume
lattice parameters
coordination descriptors
```

Instead of asking only:

```text
Which single descriptor is most important?
```

the researcher can ask:

```text
Which family of descriptors contains the
information most strongly used by the model?
```

This can provide a more scientifically meaningful interpretation.

---

# 26.3.18 Local Perturbation as an Attribution Tool

Another intuitive approach is to perturb one feature for one material.

Suppose a material has:

```text
density = 5.2
```

The researcher can evaluate the model at:

```text
density = 4.8
density = 5.0
density = 5.2
density = 5.4
density = 5.6
```

and observe the prediction.

A simple implementation is:

```python
import numpy as np

sample = X_test.iloc[0].copy()

values = np.linspace(
    4.8,
    5.6,
    20
)

predictions = []

for value in values:

    modified = sample.copy()

    modified["density"] = value

    pred = rf.predict(
        modified.to_frame().T
    )[0]

    predictions.append(pred)
```

The relationship can then be visualized:

```python
import matplotlib.pyplot as plt

plt.plot(
    values,
    predictions,
    marker="o"
)

plt.xlabel("Density")
plt.ylabel("Predicted property")
plt.tight_layout()
plt.show()
```

This provides a local sensitivity analysis.

However, it must be used carefully.

Changing one descriptor independently may produce combinations that are chemically impossible.

For example, changing density while keeping every structural descriptor fixed may violate the physical relationships present in real materials.

Therefore, perturbation-based interpretation should respect the constraints of the materials representation.

---

# 26.3.19 Why Physically Invalid Perturbations Matter

Suppose a model receives:

```text
density
volume
mass
```

These quantities are physically related:

```text
density = mass / volume
```

If we perturb density without changing the corresponding mass or volume, we may create an impossible feature combination.

The model can still produce a prediction.

But that prediction may have no physical meaning.

Therefore, a materials-aware perturbation should ideally preserve relevant constraints.

This principle becomes even more important for crystal graphs, where changing one structural variable may require coordinated changes to:

```text
atomic positions
cell parameters
neighbor relationships
periodic images
```

Explainability methods that ignore the underlying structure can therefore produce misleading interpretations.

---

# 26.3.20 Feature Attribution and Feature Engineering

Feature attribution also provides feedback about feature engineering.

Suppose a model uses 500 materials descriptors.

After attribution analysis, the researcher finds that only a small group consistently contributes strongly.

This may suggest:

```text
500 descriptors
     ↓
Attribution analysis
     ↓
Important descriptor groups
     ↓
Feature refinement
```

The researcher can then investigate whether:

```text
redundant descriptors
```

can be removed or whether:

```text
new physically meaningful descriptors
```

should be introduced.

However, feature selection should not be based solely on attribution.

Removing a feature because it has low importance in one model may reduce performance or remove information that becomes useful in another model.

Therefore, feature attribution should guide feature engineering rather than automatically determine it.

---

# 26.3.21 A Complete Descriptor-Based Attribution Workflow

A practical Materials Informatics workflow can now be assembled.

```text
Crystal Structures
       ↓
Pymatgen
       ↓
Materials Descriptors
       ↓
Feature Cleaning
       ↓
Train/Test Split
       ↓
ML Model
       ↓
Prediction
       ↓
Global Attribution
       ↓
Local Attribution
       ↓
Feature Correlation Analysis
       ↓
Physical Interpretation
       ↓
Independent Validation
```

A simplified Python workflow is:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("materials_dataset.csv")

target = "band_gap"

feature_names = [
    "mean_atomic_number",
    "mean_atomic_mass",
    "mean_electronegativity",
    "mean_atomic_radius",
    "density",
    "volume"
]

X = df[feature_names]
y = df[target]

# --------------------------------------------------
# 2. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 3. Train model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# 5. Global impurity-based importance
# --------------------------------------------------

global_importance = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(
    ascending=False
)

print("Global feature importance:")
print(global_importance)

# --------------------------------------------------
# 6. Permutation importance
# --------------------------------------------------

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

permutation_table = pd.DataFrame({
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}, index=feature_names)

permutation_table = permutation_table.sort_values(
    "importance_mean",
    ascending=False
)

print("\nPermutation importance:")
print(permutation_table)
```

This workflow provides two different views of importance:

```text
Random Forest internal importance
```

and:

```text
Permutation importance
```

Comparing them can reveal whether the model's interpretation is robust to the choice of importance method.

---

# 26.3.22 Comparing Two Attribution Methods

Suppose the results are:

```text
Feature                   RF importance   Permutation

electronegativity              0.28          0.31
atomic radius                  0.19          0.22
density                        0.16          0.11
volume                         0.12          0.03
atomic number                  0.11          0.18
```

The broad agreement is encouraging.

However, the difference for volume is notable:

```text
RF importance = 0.12
Permutation   = 0.03
```

This suggests that the model may use information associated with volume internally, but destroying volume alone does not strongly reduce performance.

Possible explanations include:

```text
volume is redundant with other descriptors
volume correlates with density
other features contain similar information
```

This is a scientifically useful observation.

Instead of simply ranking features, the researcher begins investigating the structure of the learned representation.

---

# 26.3.23 Why Multiple XAI Methods Are Valuable

No single explanation method should automatically be considered the final truth.

Different methods may emphasize different aspects of the model.

For example:

```text
Random Forest importance
        ↓
Model-specific split information

Permutation importance
        ↓
Performance dependence

SHAP
        ↓
Prediction-level contribution

Partial dependence
        ↓
Feature-response relationship
```

Using multiple approaches allows the researcher to ask:

```text
Do these methods tell a consistent story?
```

If they do, confidence increases.

If they disagree, the disagreement should be investigated.

This principle will become even more important when the chapter moves from conventional descriptors to deep-learning and crystal-graph explanations.

---

# 26.3.24 From Feature Importance to Feature Attribution

It is useful to distinguish the terms **feature importance** and **feature attribution**.

Feature importance often refers to a global ranking:

```text
Feature A → 0.31
Feature B → 0.22
Feature C → 0.12
```

Feature attribution is often more detailed and can describe the contribution of individual features to an individual prediction:

```text
Material X:

Feature A → +0.31
Feature B → -0.08
Feature C → +0.04
```

Therefore:

```text
Feature importance
=
How important is this feature generally?
```

while:

```text
Feature attribution
=
How did this feature contribute to this prediction?
```

The distinction will become central when using SHAP, Integrated Gradients, and GNN explanation methods.

---

# 26.3.25 Materials-Specific Interpretation of Attribution

A final step is to translate numerical attribution into materials-science language.

Suppose an explanation reports:

```text
mean electronegativity → strong contribution
```

The researcher should investigate:

```text
Which elements dominate this descriptor?

Are the relevant atoms concentrated around particular sites?

Is electronegativity correlated with bonding type?

Does the same pattern appear across related compounds?

Is the trend consistent with known electronic structure?
```

Similarly, if:

```text
atomic radius → strong contribution
```

the researcher may investigate:

```text
size mismatch
packing
coordination
strain
local distortion
```

The numerical explanation is therefore only the beginning.

The actual scientific value appears when the attribution is connected to chemical and structural reasoning.

---

# 26.3.26 Transition from Descriptor Attribution to Deep Learning

Feature attribution provides the conceptual foundation for the next stage of explainability.

For descriptor-based models:

```text
Material
   ↓
Feature vector
   ↓
ML model
   ↓
Feature attribution
```

For a deep neural network:

```text
Material
   ↓
Feature vector / representation
   ↓
Neural network
   ↓
Prediction
   ↓
Gradient-based explanation
```

The basic question remains unchanged:

```text
Which parts of the input influence the prediction?
```

But the explanation becomes more challenging because the relationship between input and output is nonlinear and distributed across many learned parameters.

For crystal graph neural networks, the problem becomes more complex still:

```text
Crystal
   ↓
Graph
   ↓
Atoms + edges
   ↓
Message passing
   ↓
Graph representation
   ↓
Prediction
```

Now the researcher wants to know not only:

```text
Which descriptor matters?
```

but also:

```text
Which atom matters?

Which atomic feature matters?

Which edge matters?

Which local environment matters?

Which structural interaction contributes to the prediction?
```

The next sections therefore move from conventional feature attribution toward **deep-learning explanations**, beginning with the interpretation of neural-network behavior and then developing **saliency maps** and **Integrated Gradients** before applying these ideas to crystal graph neural networks.

# 26.4 Explainability for Neural Networks

Feature attribution is relatively straightforward when the model has an explicit structure, such as a linear equation or a shallow decision tree.

Neural networks are substantially more complicated.

A neural network does not normally make a prediction through a small number of directly interpretable rules. Instead, information passes through multiple nonlinear transformations:

```text
Input Features
      ↓
Linear Transformation
      ↓
Activation
      ↓
Hidden Representation
      ↓
Linear Transformation
      ↓
Activation
      ↓
Hidden Representation
      ↓
...
      ↓
Output
```

Consequently, a prediction may depend on many input features simultaneously.

For Materials Informatics, this creates an important challenge.

A neural network may predict:

```text
Band gap = 2.84 eV
```

with excellent accuracy, while providing no obvious explanation for why the prediction is 2.84 eV.

The objective of neural-network explainability is therefore to investigate the internal relationship:

```text
Material representation
        ↓
Neural network
        ↓
Prediction
```

and determine which parts of the representation most strongly influence that prediction.

---

## 26.4.1 Why Neural Networks Are Difficult to Interpret

Consider a neural network with six input descriptors:

```text
x₁ = mean atomic number
x₂ = mean electronegativity
x₃ = mean atomic radius
x₄ = density
x₅ = volume
x₆ = coordination descriptor
```

A simple neural network might be written conceptually as:

```text
h = σ(Wx + b)

ŷ = W₂h + b₂
```

where:

* `x` is the input vector,
* `W` represents learned weights,
* `b` represents biases,
* `σ` is an activation function,
* `h` is the hidden representation,
* `ŷ` is the prediction.

The difficulty is that the contribution of an input feature is not necessarily represented by a single weight.

For example:

```text
mean electronegativity
        ↓
hidden neuron 1
        ↓
hidden neuron 4
        ↓
hidden neuron 8
        ↓
output
```

while another feature may follow a different pathway.

Therefore:

```text
Input feature
      ↓
Many nonlinear interactions
      ↓
Prediction
```

A simple inspection of network weights is usually insufficient to explain the prediction.

---

# 26.4.2 Why Neural-Network Weights Are Not Direct Explanations

A common beginner mistake is to inspect the neural-network weights and interpret large weights as important features.

For example:

```python
weights = model.fc1.weight.detach().cpu()
print(weights)
```

One might observe:

```text
Feature A → large weight
Feature B → small weight
```

and conclude:

```text
Feature A is more important.
```

This conclusion is generally unsafe.

The reason is that neural networks contain:

```text
multiple layers
+
nonlinear activation functions
+
bias terms
+
interactions
+
feature scaling
```

A feature with a relatively small weight at one layer may still have a substantial effect on the final prediction through subsequent layers.

Conversely, a large weight may not produce a large final contribution.

Therefore:

```text
Neural-network weight
        ≠
Feature attribution
```

A dedicated attribution method is required.

---

# 26.4.3 Gradient-Based Explainability

One of the most natural approaches for neural networks is to examine gradients.

Suppose the model is:

```text
ŷ = f(x)
```

The gradient with respect to the input is:

```text
∇ₓf(x)
```

This represents how sensitive the model output is to small changes in the input.

For a feature `xᵢ`:

```text
∂ŷ / ∂xᵢ
```

indicates the local sensitivity of the prediction to that feature.

Conceptually:

```text
Large gradient
     ↓
Prediction is locally sensitive
to this input
```

while:

```text
Small gradient
     ↓
Prediction is locally less sensitive
to this input
```

This is the foundation of several saliency-based methods.

---

# 26.4.4 A Simple PyTorch Gradient Explanation

Consider a neural network trained for band-gap prediction.

```python
import torch
import torch.nn as nn

class BandGapModel(nn.Module):

    def __init__(self, n_features):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)
```

After training, select one material:

```python
sample = X_test[0].clone()
sample.requires_grad_(True)
```

Generate the prediction:

```python
prediction = model(sample)
```

Now calculate the gradient:

```python
model.zero_grad()

prediction.backward()

gradient = sample.grad
```

The gradient can be inspected:

```python
print(gradient)
```

For example:

```text
tensor([
    0.41,
   -0.08,
    0.17,
    0.02,
   -0.13,
    0.05
])
```

The corresponding features might be:

```text
mean_atomic_number       0.41
mean_electronegativity  -0.08
mean_atomic_radius       0.17
density                   0.02
volume                   -0.13
coordination              0.05
```

This provides a local sensitivity profile.

---

# 26.4.5 Absolute Gradient Magnitude

Often the absolute gradient is more convenient for ranking sensitivity:

```python
importance = gradient.abs()
```

The result can be displayed together with feature names:

```python
import pandas as pd

importance_table = pd.DataFrame({
    "feature": feature_names,
    "gradient": gradient.detach().cpu().numpy(),
    "absolute_gradient": importance.detach().cpu().numpy()
})

importance_table = importance_table.sort_values(
    "absolute_gradient",
    ascending=False
)

print(importance_table)
```

A possible result is:

```text
feature                  gradient    absolute_gradient

mean_atomic_number          0.41          0.41
mean_atomic_radius          0.17          0.17
volume                     -0.13          0.13
mean_electronegativity     -0.08          0.08
coordination                0.05          0.05
density                     0.02          0.02
```

The largest absolute values identify features to which the model is most locally sensitive.

However, gradient magnitude is not necessarily the same as total contribution.

This distinction becomes important.

---

# 26.4.6 Gradient Versus Contribution

Suppose:

```text
Feature A:
x = 100
gradient = 0.01
```

and:

```text
Feature B:
x = 1
gradient = 0.5
```

The gradient suggests that the model is more locally sensitive to Feature B.

But Feature A may still have a large influence depending on the reference point and the scale of the feature.

This motivates attribution approaches that consider both:

```text
input value
+
model sensitivity
```

One simple form is:

```text
Attribution ≈ xᵢ × ∂ŷ/∂xᵢ
```

This idea is closely related to gradient × input methods.

---

# 26.4.7 Gradient × Input

A basic gradient × input attribution can be implemented as:

```python
attribution = sample * gradient
```

For example:

```python
attribution = (
    sample.detach()
    * gradient.detach()
)
```

The values can then be organized:

```python
attribution_table = pd.DataFrame({
    "feature": feature_names,
    "input": sample.detach().cpu().numpy(),
    "gradient": gradient.detach().cpu().numpy(),
    "attribution": attribution.cpu().numpy()
})

print(attribution_table)
```

The conceptual interpretation is:

```text
Input magnitude
       ×
Local sensitivity
       ↓
Approximate contribution
```

This can be more informative than gradients alone.

Nevertheless, gradient × input has its own limitations, especially when the input contains nonlinear relationships or when zero is not a meaningful baseline.

This leads to a more principled method:

**Integrated Gradients.**

---

# 26.4.8 Saliency Maps

A **saliency map** represents the sensitivity of a model output to its inputs.

In image-based machine learning, saliency maps often highlight pixels.

For Materials Informatics, the same idea can be applied to different representations.

For a descriptor vector:

```text
Feature 1   ███████
Feature 2   ██
Feature 3   █████
Feature 4   █
```

For a crystal representation, the saliency may instead correspond to:

```text
Atom 1      ███████
Atom 2      ██
Atom 3      █████
Atom 4      █
```

or:

```text
Bond 1      ██████
Bond 2      █
Bond 3      █████
```

The underlying idea is the same:

```text
Model output
      ↓
Sensitivity / attribution
      ↓
Input locations with important influence
```

The representation determines what the "location" means.

---

# 26.4.9 Saliency for Materials Descriptors

For a descriptor-based neural network, a saliency map can simply be a feature-importance vector.

Suppose:

```text
features =
[
    atomic_number,
    electronegativity,
    radius,
    density,
    volume
]
```

The model produces:

```text
attribution =
[
    0.42,
    -0.18,
    0.07,
    0.11,
    -0.03
]
```

A bar chart can visualize this.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.barh(
    feature_names,
    attribution.cpu().numpy()
)

plt.xlabel("Attribution")
plt.ylabel("Feature")
plt.tight_layout()

plt.show()
```

The resulting visualization provides a local explanation of one material.

---

# 26.4.10 Saliency for Crystal Graphs

The concept becomes more interesting when the input is a crystal graph.

Suppose a crystal is represented as:

```text
Graph G = (V, E)
```

where:

* `V` contains atomic nodes,
* `E` contains interactions or neighbor connections.

A GNN predicts:

```text
G → ŷ
```

The explanation problem becomes:

```text
Which nodes influenced ŷ?

Which edges influenced ŷ?

Which local environments influenced ŷ?
```

This is substantially more useful for Materials Informatics than a simple descriptor-level explanation because the result can potentially be mapped back to the crystal structure.

For example:

```text
Crystal
   ↓
Graph representation
   ↓
GNN
   ↓
Prediction
   ↓
Node attribution
   +
Edge attribution
```

This provides a direct connection between machine learning and atomic structure.

---

# 26.4.11 A Conceptual Crystal-Graph Example

Consider a simplified crystal graph:

```text
          O₁
         /  \
       M₁----O₂
        |
       O₃
```

Suppose the GNN predicts:

```text
Formation energy = -2.73 eV/atom
```

An explanation method might identify:

```text
Node importance:

M₁ → 0.51
O₁ → 0.12
O₂ → 0.19
O₃ → 0.08
```

and:

```text
Edge importance:

M₁-O₁ → 0.21
M₁-O₂ → 0.38
M₁-O₃ → 0.14
M₁-O₂ → particularly important
```

The researcher can then ask:

```text
What is special about the M₁-O₂ interaction?
```

This is much closer to a scientific question than:

```text
Which neuron has the largest activation?
```

---

# 26.4.12 Gradient-Based Node Attribution

Suppose the node features are stored in a tensor:

```python
node_features.requires_grad_(True)
```

After obtaining the graph prediction:

```python
prediction = model(graph)
```

calculate the gradient:

```python
model.zero_grad()

prediction.backward()

node_gradients = node_features.grad
```

A node-level score can be constructed using the gradient magnitude:

```python
node_importance = node_gradients.abs().sum(
    dim=1
)
```

If there are `N` atoms and each atom has `F` features:

```text
node_gradients.shape = (N, F)
```

then:

```text
node_importance.shape = (N,)
```

Each value represents the aggregate gradient sensitivity of one atomic node.

The workflow is therefore:

```text
Node features
      ↓
Gradient
      ↓
Aggregate feature dimensions
      ↓
Node importance
```

---

# 26.4.13 Mapping Node Importance Back to the Crystal

The numerical result becomes scientifically useful only when it is mapped back to the structure.

Suppose:

```python
node_importance = node_gradients.abs().sum(
    dim=1
)
```

and atomic species are available:

```python
species = [
    "Mg",
    "Al",
    "O",
    "O",
    "O",
    "O"
]
```

We can create a table:

```python
node_table = pd.DataFrame({
    "node": range(len(species)),
    "element": species,
    "importance": node_importance.detach().cpu().numpy()
})

node_table = node_table.sort_values(
    "importance",
    ascending=False
)

print(node_table)
```

A possible result:

```text
node    element    importance

1       Al           0.82
0       Mg           0.61
3       O            0.44
2       O            0.37
5       O            0.29
4       O            0.21
```

The researcher can now inspect the corresponding atomic environments.

For example:

```text
Al site
 ↓
coordination environment
 ↓
bond lengths
 ↓
local distortion
 ↓
possible physical interpretation
```

This is an important bridge from numerical attribution to structural science.

---

# 26.4.14 Edge-Level Attribution

Node importance alone may not be sufficient.

Two atoms can individually have moderate importance while their interaction is highly important.

For a crystal graph:

```text
Nodes
=
atoms

Edges
=
neighbor interactions
```

An edge attribution method attempts to estimate:

```text
How strongly does this atomic interaction influence the prediction?
```

Conceptually:

```text
Atom A -------- Atom B
       ↑
   important edge
```

The result can be used to identify:

```text
important bonds
important neighbor interactions
important local environments
```

This becomes particularly valuable for properties related to:

```text
bonding
magnetism
ionic transport
electronic structure
phonons
mechanical behavior
```

However, the exact physical interpretation depends on what the graph edge represents.

An edge in a GNN is often a computational representation of neighborhood information rather than a literal chemical bond.

This distinction must always be preserved.

---

# 26.4.15 What an Edge Means in a Materials GNN

Suppose a graph is constructed by connecting atoms within a cutoff radius:

```text
if distance(i,j) < r_cut:
       create edge(i,j)
```

Then an edge means:

```text
Atom i and Atom j
are included in the model's local neighborhood.
```

It does not necessarily mean:

```text
A chemical bond exists between i and j.
```

For example, a GNN may connect atoms because they are spatial neighbors even if there is no conventional chemical bond.

Therefore, an explanation such as:

```text
Edge A-B has high attribution
```

should initially be interpreted as:

> The model's prediction is strongly influenced by the represented interaction between these two neighboring nodes.

Only additional chemical analysis can determine whether this corresponds to a physically meaningful bonding interaction.

---

# 26.4.16 Saliency Maps and Their Limitations

Saliency methods are attractive because they are relatively simple.

They can be implemented directly using automatic differentiation.

However, several problems must be considered.

### Saturation

A neural network can enter a region where the output changes very little with respect to the input.

Then:

```text
gradient ≈ 0
```

even though the feature may have played an important role in producing the final prediction.

### Noise

Gradients may vary rapidly between nearby inputs.

### Feature scaling

Gradients depend on the scale of the input representation.

### Correlated features

Multiple features may encode similar information.

### Baseline dependence

Some attribution methods require choosing a reference input.

These issues motivate more robust approaches.

---

# 26.4.17 Why Integrated Gradients Is Needed

Suppose we have:

```text
baseline x'
```

and:

```text
actual input x
```

A simple gradient evaluates the model only at:

```text
x
```

Integrated Gradients instead considers the path from the baseline to the actual input.

Conceptually:

```text
Baseline
   ↓
Intermediate input
   ↓
Intermediate input
   ↓
...
   ↓
Actual material
```

The gradient is evaluated along this path.

This provides a more complete picture of how the model output changes as the input moves from the reference state to the actual material.

The basic idea is:

```text
Integrated attribution
=
Accumulated gradients
along a path
```

This will be developed mathematically in the next section.

---

# 26.4.18 Choosing a Baseline in Materials Informatics

The baseline is particularly important for materials applications.

In image classification, a black image may be a reasonable baseline.

In Materials Informatics, the appropriate baseline depends on the representation.

For a descriptor vector, one possible baseline is:

```text
feature-wise mean
```

For example:

```python
baseline = X_train.mean(axis=0)
```

For a standardized representation, the zero vector may correspond to the dataset mean:

```text
z = 0
```

which can make it a useful reference.

However, the baseline must be scientifically meaningful.

For crystal graphs, a zero-feature graph may not represent a physically meaningful material at all.

This creates a major challenge for graph-based explainability.

A useful baseline should ideally represent a meaningful reference state while remaining compatible with the model architecture.

---

# 26.4.19 Materials-Aware Interpretation of Neural Attributions

Suppose Integrated Gradients identifies:

```text
mean_atomic_radius → high attribution
```

The interpretation should proceed carefully.

A researcher might investigate:

```text
Atomic radius
      ↓
Size mismatch
      ↓
Local structural distortion
      ↓
Coordination changes
      ↓
Property modification
```

Similarly:

```text
Electronegativity
      ↓
Bond polarity
      ↓
Electronic distribution
      ↓
Target property
```

The attribution itself does not establish this causal chain.

It identifies where the model is using information.

The scientific interpretation requires additional analysis.

---

# 26.4.20 From Neural Attributions to Scientific Hypotheses

A powerful workflow is:

```text
Neural Network
      ↓
Attribution
      ↓
Important input
      ↓
Materials interpretation
      ↓
Scientific hypothesis
      ↓
DFT / experiment
```

For example:

```text
GNN
 ↓
High attribution around a distorted octahedral site
 ↓
Hypothesis:
local octahedral distortion affects target property
 ↓
Construct structures with controlled distortion
 ↓
DFT calculations
 ↓
Compare predicted property
```

This turns explainability into a tool for scientific discovery.

---

# 26.4.21 Practical PyTorch Workflow

A minimal neural-network explanation pipeline can now be assembled.

```python
import torch
import torch.nn as nn
import pandas as pd

# --------------------------------------------------
# 1. Neural network
# --------------------------------------------------

class MaterialsMLP(nn.Module):

    def __init__(self, n_features):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        return self.network(x)


# --------------------------------------------------
# 2. Create model
# --------------------------------------------------

model = MaterialsMLP(
    n_features=len(feature_names)
)

# --------------------------------------------------
# 3. Select one material
# --------------------------------------------------

sample = X_test_tensor[0].clone()

sample = sample.unsqueeze(0)

sample.requires_grad_(True)

# --------------------------------------------------
# 4. Prediction
# --------------------------------------------------

prediction = model(sample)

# --------------------------------------------------
# 5. Gradient
# --------------------------------------------------

model.zero_grad()

prediction.backward()

gradient = sample.grad.squeeze(0)

# --------------------------------------------------
# 6. Gradient magnitude
# --------------------------------------------------

gradient_importance = gradient.abs()

# --------------------------------------------------
# 7. Gradient × input
# --------------------------------------------------

gradient_input = (
    sample.detach().squeeze(0)
    * gradient.detach()
)

# --------------------------------------------------
# 8. Explanation table
# --------------------------------------------------

explanation = pd.DataFrame({

    "feature": feature_names,

    "gradient":
        gradient.detach().cpu().numpy(),

    "abs_gradient":
        gradient_importance.detach().cpu().numpy(),

    "gradient_x_input":
        gradient_input.detach().cpu().numpy()
})

print(explanation)
```

This is not yet a complete XAI framework.

It is a transparent starting point that demonstrates the central computational idea:

```text
Input
 ↓
Prediction
 ↓
Backward pass
 ↓
Gradient
 ↓
Attribution
```

The next step is to make this attribution more principled through **Integrated Gradients**.

---

# 26.4.22 Interpreting the Explanation Table

Suppose the resulting table is:

```text
feature                  gradient    abs_gradient    gradient_x_input

atomic_number              0.12          0.12              0.18
electronegativity         -0.31          0.31             -0.24
atomic_radius              0.08          0.08              0.11
density                    0.02          0.02              0.04
volume                    -0.17          0.17             -0.21
coordination               0.09          0.09              0.13
```

Three different quantities should be distinguished.

### Gradient

```text
∂ŷ / ∂xᵢ
```

Measures local sensitivity.

### Absolute gradient

```text
|∂ŷ / ∂xᵢ|
```

Measures magnitude of local sensitivity without direction.

### Gradient × input

```text
xᵢ × ∂ŷ / ∂xᵢ
```

Provides an input-weighted local attribution.

These are related but not identical.

A research report should clearly state which one is being used.

---

# 26.4.23 Reproducibility of Neural-Network Explanations

Neural-network explanations can vary because neural networks themselves can vary.

Two models trained with different random seeds may produce:

```text
Model A:
electronegativity → dominant

Model B:
atomic radius → dominant
```

even if their predictive performance is similar.

Therefore, a reproducible XAI experiment should record:

```text
random seed
model architecture
training data
feature preprocessing
trained weights
baseline
attribution method
number of integration steps
software versions
```

For example:

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

For CUDA environments, additional deterministic settings may be required depending on the experiment.

The goal is to make it possible to reproduce both:

```text
the prediction
```

and:

```text
the explanation
```

---

# 26.4.24 Explanation Is Part of the Research Record

For Materials Informatics research, the explanation itself can be saved.

For example:

```python
explanation.to_csv(
    "material_001_explanation.csv",
    index=False
)
```

The research record can contain:

```text
material ID
model version
prediction
uncertainty
feature attribution
timestamp
training dataset version
```

For a larger experiment:

```text
results/
│
├── predictions.csv
├── uncertainties.csv
├── global_importance.csv
├── local_explanations/
│   ├── material_001.csv
│   ├── material_002.csv
│   └── material_003.csv
│
└── model/
    ├── weights.pt
    └── configuration.json
```

This is particularly valuable when explanations are later used to formulate scientific hypotheses.

---

# 26.4.25 From Feature Attribution to Integrated Gradients

At this point, the basic progression is:

```text
Classical ML
    ↓
Feature importance
    ↓
Feature attribution
```

and:

```text
Neural Network
    ↓
Gradient
    ↓
Saliency
    ↓
Gradient × Input
```

However, these approaches can be sensitive to local behavior and baseline assumptions.

Integrated Gradients provides a more principled framework by accumulating gradients between a reference input and the actual input.

The next section will therefore develop Integrated Gradients mathematically and implement it in PyTorch, followed by its application to materials descriptors and, later, crystal graph neural networks.

# 26.5 Integrated Gradients

Gradient-based saliency provides a useful first approximation of how a neural network responds to its inputs.

However, a local gradient has an important limitation.

It describes the model's behavior **at one point**.

For a materials model:

```text
Material representation x
        ↓
Neural network
        ↓
Prediction
```

the ordinary gradient evaluates:

```text
∂ŷ / ∂x
```

at the specific material `x`.

This can be problematic when the neural network is highly nonlinear.

A feature may have a small gradient at the final input even though changing that feature from a reference state to the actual material was important for producing the prediction.

Integrated Gradients addresses this problem by considering the entire path between a **baseline** and the actual input.

The central idea is:

```text
Baseline material representation
             ↓
       Intermediate states
             ↓
       Intermediate states
             ↓
       Intermediate states
             ↓
Actual material representation
```

Instead of asking only:

> How sensitive is the model at this material?

Integrated Gradients asks:

> How did the model respond as the input moved from a reference state to the actual material?

This makes it particularly useful for nonlinear neural networks.

---

## 26.5.1 The Integrated Gradients Concept

Let the trained model be:

```text
ŷ = F(x)
```

where:

* `x` is the input,
* `F` is the trained neural network,
* `ŷ` is the model output.

Let:

```text
x'
```

be a baseline input.

The baseline may represent a reference state such as:

```text
average material
```

or:

```text
zero-centered feature vector
```

depending on the representation.

The actual material is:

```text
x
```

Integrated Gradients considers the straight-line path:

```text
x'
 ↓
x' + α(x - x')
 ↓
x' + 2α(x - x')
 ↓
...
 ↓
x
```

where:

```text
0 ≤ α ≤ 1
```

At:

```text
α = 0
```

we have:

```text
x'
```

and at:

```text
α = 1
```

we have:

```text
x
```

Thus the complete path is:

```text
x(α) = x' + α(x - x')
```

---

## 26.5.2 Mathematical Definition

The Integrated Gradients attribution for feature `i` is defined as:

```text
IGᵢ(x)
=
(xᵢ - xᵢ')
∫₀¹
∂F(x' + α(x - x')) / ∂xᵢ
dα
```

The expression contains two important components.

First:

```text
xᵢ - xᵢ'
```

represents how far the feature moves from the baseline to the actual material.

Second:

```text
∫₀¹
∂F / ∂xᵢ
dα
```

represents the accumulated sensitivity of the model along that path.

Therefore:

```text
Integrated Gradient
=
Input difference
×
Accumulated model sensitivity
```

This is more informative than evaluating the gradient only at the final point.

---

## 26.5.3 Intuitive Interpretation

Consider a material descriptor:

```text
mean electronegativity
```

Suppose the baseline is:

```text
1.8
```

and the actual material has:

```text
2.6
```

The path is:

```text
1.8
 ↓
1.88
 ↓
1.96
 ↓
2.04
 ↓
...
 ↓
2.60
```

At each point, the neural network's gradient is evaluated.

Conceptually:

```text
Electronegativity
      ↓
Model sensitivity
      ↓
Model sensitivity
      ↓
Model sensitivity
      ↓
Accumulated effect
```

The final attribution therefore accounts for how the model responds across the entire transition.

---

## 26.5.4 Why the Baseline Matters

Integrated Gradients depends on the baseline.

This is one of the most important practical considerations.

Suppose:

```text
x' = average material
```

Then the attribution answers approximately:

> How did the features of this material move the prediction away from the model's prediction for the reference material?

If instead:

```text
x' = zero vector
```

the interpretation becomes:

> How did the input differ from the zero representation in producing the prediction?

These are not necessarily the same scientific question.

Therefore, the baseline must be selected deliberately.

---

## 26.5.5 Baseline Selection for Descriptor Models

For a numerical materials descriptor matrix, a common baseline is the training-set mean.

```python
baseline = X_train.mean(axis=0)
```

If the features have already been standardized:

```text
mean ≈ 0
```

so the zero vector may become a natural baseline.

For example:

```python
import numpy as np

baseline = np.zeros(
    X_train.shape[1],
    dtype=np.float32
)
```

This is convenient, but convenience does not automatically imply physical meaning.

A researcher should ask:

```text
Does zero correspond to a meaningful reference?
```

If not, a different baseline may be more appropriate.

---

## 26.5.6 Baselines in Materials Science

A materials-specific baseline might be constructed from:

```text
dataset mean
```

or:

```text
reference composition
```

or:

```text
reference structure
```

or:

```text
chemically meaningful neutral representation
```

depending on the model.

For descriptor-based models:

```text
Mean descriptor vector
```

is often straightforward.

For crystal graphs, however, the problem is more complicated because a graph cannot always be meaningfully represented by simply setting every feature to zero.

This distinction becomes important later when Integrated Gradients is applied to GNNs.

---

## 26.5.7 Numerical Approximation

The integral in Integrated Gradients usually cannot be solved analytically for a neural network.

Instead, it is approximated numerically.

Choose `m` interpolation steps.

For example:

```text
m = 50
```

The values of `α` become:

```text
0
0.02
0.04
0.06
...
1.00
```

For each value:

```text
xα = x' + α(x - x')
```

the gradient is evaluated.

The integral is then approximated using a numerical sum.

Conceptually:

```text
Integral
   ↓
Gradient at many points
   ↓
Numerical average
   ↓
Integrated attribution
```

---

## 26.5.8 Riemann Approximation

A simple approximation is:

```text
IGᵢ
≈
(xᵢ - xᵢ')
×
(1/m)
Σₖ
∂F(xₖ)/∂xᵢ
```

where:

```text
xₖ = x' + αₖ(x - x')
```

and:

```text
αₖ = k/m
```

for:

```text
k = 1, ..., m
```

Increasing `m` generally provides a more accurate approximation, although it also increases computational cost.

For example:

```text
m = 10
```

is relatively cheap.

```text
m = 100
```

is more accurate.

```text
m = 1000
```

may be unnecessarily expensive for some models.

The appropriate number of steps should therefore be validated rather than chosen arbitrarily.

---

## 26.5.9 Implementing Integrated Gradients Manually in PyTorch

A manual implementation is useful because it exposes the underlying algorithm.

Suppose:

```python
import torch
```

and a trained model:

```python
model
```

are already available.

We first select a material:

```python
sample = X_test_tensor[0:1]
```

and define a baseline:

```python
baseline = torch.zeros_like(sample)
```

Now choose the number of integration steps:

```python
steps = 100
```

Create interpolation values:

```python
alphas = torch.linspace(
    0,
    1,
    steps + 1
)
```

The interpolation tensor can then be generated:

```python
scaled_inputs = torch.cat([
    baseline + alpha * (sample - baseline)
    for alpha in alphas
])
```

The shape becomes approximately:

```text
(number_of_steps + 1, number_of_features)
```

Each row corresponds to one point along the baseline-to-input path.

---

## 26.5.10 Computing Gradients Along the Path

We now enable gradient tracking:

```python
scaled_inputs.requires_grad_(True)
```

Generate predictions:

```python
outputs = model(scaled_inputs)
```

For a scalar output per sample, we can compute gradients with:

```python
gradients = torch.autograd.grad(
    outputs=outputs,
    inputs=scaled_inputs,
    grad_outputs=torch.ones_like(outputs)
)[0]
```

The resulting tensor contains:

```text
gradient at α₀
gradient at α₁
gradient at α₂
...
gradient at αₘ
```

This is the central computational step of Integrated Gradients.

---

## 26.5.11 Averaging the Gradients

The gradients can be averaged:

```python
avg_gradients = gradients.mean(
    dim=0,
    keepdim=True
)
```

Then multiply by the difference between the actual input and baseline:

```python
integrated_gradients = (
    sample - baseline
) * avg_gradients
```

The result has the same dimensionality as the input.

Thus:

```text
Input features
      ↓
Integrated Gradients
      ↓
One attribution value per feature
```

---

## 26.5.12 Complete Manual Implementation

The entire calculation can be packaged into a function.

```python
def integrated_gradients(
    model,
    input_tensor,
    baseline=None,
    steps=100
):

    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(
            input_tensor
        )

    alphas = torch.linspace(
        0.0,
        1.0,
        steps + 1,
        device=input_tensor.device
    )

    scaled_inputs = torch.cat([
        baseline
        + alpha * (
            input_tensor - baseline
        )
        for alpha in alphas
    ])

    scaled_inputs.requires_grad_(True)

    outputs = model(
        scaled_inputs
    )

    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=scaled_inputs,
        grad_outputs=torch.ones_like(outputs)
    )[0]

    avg_gradients = gradients.mean(
        dim=0,
        keepdim=True
    )

    attributions = (
        input_tensor - baseline
    ) * avg_gradients

    return attributions
```

The method can then be called:

```python
attributions = integrated_gradients(
    model,
    sample,
    baseline=baseline,
    steps=100
)
```

The result can be inspected:

```python
print(attributions)
```

---

## 26.5.13 Converting Attributions into a Materials Explanation

Suppose the model has:

```python
feature_names = [
    "mean_atomic_number",
    "mean_electronegativity",
    "mean_atomic_radius",
    "density",
    "volume",
    "coordination"
]
```

We can construct an explanation table:

```python
explanation = pd.DataFrame({
    "feature": feature_names,
    "attribution": (
        attributions
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
})

explanation["absolute_attribution"] = (
    explanation["attribution"]
    .abs()
)

explanation = explanation.sort_values(
    "absolute_attribution",
    ascending=False
)

print(explanation)
```

A possible result is:

```text
feature                  attribution

mean_electronegativity       +0.47
mean_atomic_radius           -0.21
density                      +0.14
mean_atomic_number           +0.10
volume                       -0.07
coordination                 +0.03
```

This provides a local explanation for one material.

---

## 26.5.14 Signed Attribution

The sign of Integrated Gradients is useful.

Suppose:

```text
electronegativity = +0.47
```

and:

```text
atomic radius = -0.21
```

Under the chosen baseline and model:

```text
electronegativity
    ↓
pushes prediction upward

atomic radius
    ↓
pushes prediction downward
```

The exact interpretation depends on the target and output convention.

For regression:

```text
positive attribution
```

generally means movement toward a higher prediction relative to the baseline output.

For classification, attribution may refer to a particular class score or logit.

Therefore, the researcher must explicitly state what model output is being explained.

---

## 26.5.15 Attribution Conservation

One of the attractive properties of Integrated Gradients is its connection to the difference between the model output at the actual input and baseline.

Ideally:

```text
Σ IGᵢ
≈
F(x) - F(x')
```

This relationship is known as the **completeness** property.

In practical numerical calculations:

```text
Σ IGᵢ
```

may not exactly equal:

```text
F(x) - F(x')
```

because of numerical approximation error.

This provides an important diagnostic.

---

## 26.5.16 The Completeness Check

We can explicitly check the approximation.

First calculate the model outputs:

```python
baseline_output = model(
    baseline
)

input_output = model(
    sample
)
```

Then calculate:

```python
attribution_sum = attributions.sum()
```

Compare:

```python
difference = (
    input_output
    - baseline_output
)
```

and:

```python
error = (
    attribution_sum
    - difference
)
```

For example:

```python
print(
    "Attribution sum:",
    attribution_sum.item()
)

print(
    "Output difference:",
    difference.item()
)

print(
    "Approximation error:",
    error.item()
)
```

A small error indicates that the numerical integration is reasonably accurate.

---

## 26.5.17 Increasing the Number of Integration Steps

Suppose:

```text
steps = 10
```

produces:

```text
completeness error = 0.08
```

Increasing the number of steps:

```text
steps = 100
```

may produce:

```text
error = 0.009
```

and:

```text
steps = 500
```

might produce:

```text
error = 0.002
```

The exact values depend on the model.

The researcher can therefore evaluate convergence:

```python
for steps in [10, 25, 50, 100, 200]:

    attr = integrated_gradients(
        model,
        sample,
        baseline,
        steps=steps
    )

    print(
        steps,
        attr.sum().item()
    )
```

This is preferable to selecting an arbitrary integration resolution without checking numerical behavior.

---

## 26.5.18 Computational Cost

Integrated Gradients requires multiple forward and backward operations.

Therefore:

```text
ordinary prediction
≈ one forward pass
```

while:

```text
Integrated Gradients
≈ many forward/backward evaluations
```

If:

```text
steps = 100
```

the computational workload can be substantially larger than one prediction.

For a dataset containing thousands of materials, calculating explanations for every sample can become expensive.

A practical strategy is therefore:

```text
Large dataset
      ↓
Train model
      ↓
Identify scientifically interesting samples
      ↓
Run detailed explanations
```

rather than automatically explaining every material.

---

## 26.5.19 Batch Processing of Attributions

When explanations are required for many materials, the interpolation steps can be processed in batches.

Conceptually:

```text
Material 1
Material 2
Material 3
...
Material N
```

can be divided into manageable groups.

For example:

```python
batch_size = 32
```

This reduces memory pressure on the GPU.

The exact implementation depends on:

```text
model size
number of features
number of integration steps
GPU memory
number of materials
```

The important computational principle is:

> Explanation workloads should be treated as separate inference workloads and engineered accordingly.

---

## 26.5.20 Integrated Gradients for a Band-Gap Model

Consider a model trained to predict band gap:

```text
Input:
composition + structure descriptors

Output:
band gap
```

A typical workflow is:

```text
Materials Dataset
       ↓
Descriptor Generation
       ↓
Normalization
       ↓
Neural Network
       ↓
Band-gap prediction
       ↓
Integrated Gradients
       ↓
Feature attribution
```

Python implementation:

```python
sample = X_test_tensor[0:1]

baseline = torch.zeros_like(
    sample
)

attributions = integrated_gradients(
    model,
    sample,
    baseline=baseline,
    steps=100
)
```

Then:

```python
explanation = pd.DataFrame({
    "feature": feature_names,
    "attribution": (
        attributions
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
})

explanation = explanation.sort_values(
    "attribution",
    key=lambda x: x.abs(),
    ascending=False
)

print(explanation)
```

This produces a material-specific explanation.

---

## 26.5.21 Visualizing Integrated Gradients

A horizontal bar plot is often convenient.

```python
import matplotlib.pyplot as plt

plot_data = explanation.copy()

plot_data = plot_data.sort_values(
    "attribution"
)

plt.figure(figsize=(8, 6))

plt.barh(
    plot_data["feature"],
    plot_data["attribution"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    "Integrated Gradient Attribution"
)

plt.ylabel(
    "Materials Descriptor"
)

plt.tight_layout()

plt.show()
```

The result communicates:

```text
positive contribution
        ←
baseline
        →
negative contribution
```

and allows the researcher to identify the dominant descriptors for the particular material.

---

## 26.5.22 Comparing Multiple Materials

Explaining one material is useful, but scientific interpretation often requires comparison.

Suppose three materials are selected:

```text
Material A
Material B
Material C
```

Their explanations may be:

```text
Material A:
electronegativity → dominant

Material B:
atomic radius → dominant

Material C:
density → dominant
```

This can reveal that the same trained model uses different information in different regions of materials space.

A useful workflow is:

```text
Materials
   ↓
Individual predictions
   ↓
Individual attributions
   ↓
Compare attribution patterns
   ↓
Identify material families
```

---

## 26.5.23 Attribution Similarity

Suppose each material has an attribution vector:

```text
Material A:
[0.4, 0.1, -0.2, 0.05]

Material B:
[0.38, 0.12, -0.18, 0.07]

Material C:
[-0.1, 0.5, 0.04, 0.31]
```

Materials A and B have similar attribution patterns.

Material C is different.

This suggests a possible research direction:

```text
Attribution vectors
      ↓
Similarity analysis
      ↓
Materials grouped by
model reasoning
```

This does not necessarily mean that the materials are chemically similar.

Instead, they may be similar in terms of **how the trained model makes predictions**.

This distinction can be scientifically interesting.

---

## 26.5.24 Attribution Across a Materials Dataset

For a dataset of `N` materials and `F` features, the attribution matrix can be represented as:

```text
A ∈ R^(N × F)
```

where:

```text
Aᵢⱼ
```

is the attribution of feature `j` for material `i`.

Conceptually:

```text
             Features
          F₁   F₂   F₃   F₄
       ┌────────────────────
M₁     │ .4   .1  -.2   .0
M₂     │ .3   .2  -.1   .1
M₃     │-.1   .5   .0   .3
M₄     │ .2   .1  -.3   .1
...
```

This matrix can itself become an object of analysis.

For example:

```text
Attribution matrix
       ↓
PCA
       ↓
Clustering
       ↓
Model-reasoning space
```

This creates an interesting connection between explainability and unsupervised analysis.

---

## 26.5.25 Attribution Is Not Causality

This principle deserves explicit emphasis.

Suppose Integrated Gradients identifies:

```text
electronegativity
```

as an important feature.

The correct statement is:

> The trained model strongly uses electronegativity in producing this prediction relative to the selected baseline.

The incorrect statement is:

> Electronegativity causes the predicted property.

Machine-learning attribution does not automatically establish causality.

A causal claim requires additional evidence, such as:

```text
controlled computational experiments
DFT analysis
physical theory
experimental validation
causal modeling
```

This distinction is especially important in scientific publications.

---

## 26.5.26 Attribution Does Not Guarantee Physical Correctness

A model can learn a spurious relationship.

For example:

```text
Dataset
   ↓
Certain material family
   ↓
Specific descriptor
   ↓
Target property
```

The model may use the descriptor because it is correlated with the target within the dataset.

But outside the dataset, the relationship may disappear.

Therefore:

```text
Good prediction
+
Good attribution
```

does not automatically imply:

```text
Correct physical mechanism
```

Explainability should therefore be combined with independent scientific validation.

---

## 26.5.27 Integrated Gradients and Dataset Bias

Suppose the training dataset contains mostly:

```text
oxides
```

and only a few:

```text
sulfides
```

The model may learn patterns that work extremely well for oxides.

An attribution method may then identify:

```text
electronegativity
```

as highly important.

However, the attribution may reflect the dataset distribution rather than a universally valid physical principle.

Therefore, explanation should always be interpreted together with:

```text
dataset composition
chemical coverage
structural diversity
training distribution
test distribution
```

This is particularly important when the model is used for materials discovery outside its training domain.

---

## 26.5.28 Integrated Gradients for Scientific Investigation

A strong scientific workflow is:

```text
Train neural network
        ↓
Validate predictive performance
        ↓
Select interesting materials
        ↓
Calculate Integrated Gradients
        ↓
Identify important features
        ↓
Investigate physical meaning
        ↓
Formulate hypothesis
        ↓
Test using DFT or experiment
```

For example:

```text
GNN predicts unusual stability
        ↓
Attribution highlights local environment
        ↓
Researcher identifies unusual coordination
        ↓
Construct controlled structural variants
        ↓
DFT comparison
        ↓
Validate or reject hypothesis
```

This is where explainable AI becomes more than a visualization technique.

It becomes part of the scientific workflow.

---

# 26.5.29 Integrated Gradients in Crystal Graph Neural Networks

For a crystal graph neural network, the same conceptual framework can be applied.

The graph contains:

```text
G = (V, E)
```

with node features:

```text
hᵢ
```

and edge features:

```text
eᵢⱼ
```

The GNN produces:

```text
ŷ = F(G)
```

The explanation problem becomes:

```text
Which atomic and structural inputs
contributed to ŷ?
```

Potential attribution targets include:

```text
Node features
Edge features
Node representations
Atomic environments
Neighbor interactions
```

For example:

```text
Crystal
   ↓
Graph
   ↓
Node features
   +
Edge features
   ↓
GNN
   ↓
Prediction
   ↓
Integrated Gradients
   ↓
Atomic / structural attribution
```

This is the next major step in the chapter.

---

# 26.5.30 Challenges of Integrated Gradients for Crystal Graphs

Applying Integrated Gradients to crystal graphs introduces several additional issues.

### Variable graph size

Different crystals contain different numbers of atoms.

### Discrete atomic identities

Atomic species are categorical rather than continuous.

### Edge construction

Changing coordinates can alter neighbor relationships.

### Periodic boundary conditions

A crystal graph includes periodic structure that must be represented consistently.

### Baseline construction

A physically meaningful reference graph is difficult to define.

### Structural validity

Interpolating between two structures may produce physically meaningless intermediate states.

These issues mean that Integrated Gradients cannot simply be applied to a crystal graph without considering how the graph representation is constructed.

---

# 26.5.31 Continuous Node Representations

One practical approach is to apply attribution to continuous representations associated with the atoms.

For example, an atomic feature vector might contain:

```text
atomic number
period
group
electronegativity
atomic radius
valence information
```

These values can be embedded into a continuous vector:

```text
Atomic properties
      ↓
Embedding
      ↓
Continuous node representation
```

Integrated Gradients can then operate on the continuous representation.

However, the resulting explanation must still be mapped back to chemically meaningful quantities.

---

# 26.5.32 Attribution at Multiple Representation Levels

A GNN explanation can potentially be performed at several levels:

```text
Crystal level
      ↓
Node level
      ↓
Edge level
      ↓
Feature level
```

For example:

```text
Prediction
   ↓
important atom
   ↓
important local environment
   ↓
important atomic feature
```

This hierarchical explanation is particularly attractive for Materials Informatics.

It can connect:

```text
Model behavior
```

with:

```text
Atomic-scale scientific interpretation
```

The next sections will develop this idea further through dedicated saliency and graph-explanation methods.

# 26.6 Saliency Maps for Materials Models

Saliency methods provide one of the most direct approaches for understanding how a neural network responds to its input.

The central idea is simple:

> Identify the input components to which the model output is most sensitive.

For a neural network:

```text
Input
  ↓
Model
  ↓
Prediction
```

saliency analysis adds:

```text
Input
  ↓
Model
  ↓
Prediction
  ↑
Gradient with respect to input
```

The resulting gradient can be converted into a saliency score.

For Materials Informatics, the meaning of a saliency score depends strongly on the representation used by the model.

For a descriptor-based model:

```text
Feature
 ↓
Saliency
```

may identify important chemical or structural descriptors.

For a crystal graph neural network:

```text
Atom
 ↓
Node saliency
```

or:

```text
Atomic interaction
 ↓
Edge saliency
```

may identify structurally important regions.

Therefore, saliency should not be treated as one universal visualization technique. It is a family of gradient-based explanation approaches whose interpretation depends on the representation.

---

## 26.6.1 Basic Saliency Definition

Let a trained model be:

```text
ŷ = F(x)
```

For an input feature `xᵢ`, the local saliency can be defined using:

```text
∂F / ∂xᵢ
```

A commonly used magnitude-based score is:

```text
Sᵢ = |∂F / ∂xᵢ|
```

The absolute value removes the direction and retains only sensitivity magnitude.

Thus:

```text
Large Sᵢ
    ↓
Strong local sensitivity
```

while:

```text
Small Sᵢ
    ↓
Weak local sensitivity
```

The sign can also be retained when directional interpretation is important.

---

## 26.6.2 Saliency Versus Integrated Gradients

Saliency and Integrated Gradients are closely related but answer slightly different questions.

### Saliency

Evaluates the model locally:

```text
x
 ↓
gradient
```

It asks:

> How sensitive is the prediction to a small perturbation around this material?

### Integrated Gradients

Evaluates a path:

```text
baseline
   ↓
   ↓
   ↓
actual material
```

It asks:

> How did the model accumulate sensitivity as the input moved from the baseline to the actual material?

The conceptual distinction is:

```text
Saliency
=
local explanation
```

whereas:

```text
Integrated Gradients
=
path-based explanation
```

Both can be useful in Materials Informatics.

---

## 26.6.3 Simple Saliency Implementation

Consider a trained PyTorch model:

```python
import torch

model.eval()
```

Select a material:

```python
sample = X_test_tensor[0:1].clone()

sample.requires_grad_(True)
```

Calculate the prediction:

```python
prediction = model(sample)
```

Calculate the gradient:

```python
model.zero_grad()

prediction.backward()
```

Retrieve the gradient:

```python
gradient = sample.grad
```

The saliency score can then be calculated as:

```python
saliency = gradient.abs()
```

This produces one saliency value for every input feature.

---

## 26.6.4 Complete Saliency Function

The procedure can be encapsulated:

```python
def compute_saliency(model, sample):

    model.eval()

    sample = sample.clone().detach()
    sample.requires_grad_(True)

    prediction = model(sample)

    model.zero_grad()

    prediction.sum().backward()

    saliency = sample.grad.abs()

    return saliency
```

Use it as:

```python
saliency = compute_saliency(
    model,
    X_test_tensor[0:1]
)

print(saliency)
```

For a descriptor-based materials model, this produces:

```text
feature 1 → saliency
feature 2 → saliency
feature 3 → saliency
...
```

---

## 26.6.5 Preserving the Sign

Absolute saliency is useful for ranking sensitivity, but it removes direction.

If directional information is required:

```python
signed_gradient = sample.grad
```

can be retained.

For example:

```text
Feature A → +0.42
Feature B → -0.18
Feature C → +0.05
```

can be interpreted as:

```text
Feature A:
positive local influence

Feature B:
negative local influence

Feature C:
weak positive local influence
```

The exact interpretation depends on the target.

For regression:

```text
positive
→ locally increases prediction

negative
→ locally decreases prediction
```

For classification, the explanation must specify which output or class is being differentiated.

---

## 26.6.6 Feature Scaling and Saliency

Feature scaling is particularly important for gradient-based explanations.

Suppose one feature has values:

```text
density = 5–10
```

while another has:

```text
atomic number = 1–100
```

Their raw gradients cannot automatically be compared as though they were on identical scales.

For this reason, materials neural networks commonly use standardized inputs:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
```

The neural network is then trained on the standardized representation.

However, the explanation must be interpreted in the same coordinate system in which the gradient was calculated.

A careful research workflow therefore records:

```text
Original feature
↓
Scaling transformation
↓
Model input
↓
Saliency calculation
```

---

## 26.6.7 Returning to Physical Units

Suppose:

```text
x_scaled = (x - μ) / σ
```

The gradient with respect to the scaled feature is:

```text
∂ŷ / ∂x_scaled
```

If the researcher wants sensitivity with respect to the original physical variable, the chain rule gives:

```text
∂ŷ / ∂x
=
∂ŷ / ∂x_scaled
×
1/σ
```

This distinction matters when reporting scientific sensitivity.

For example, a model may be highly sensitive to:

```text
atomic radius
```

but the numerical gradient depends on whether radius is represented in:

```text
Å
```

or as:

```text
standardized radius
```

Therefore, scientific reporting should clearly state the representation used for attribution.

---

## 26.6.8 Saliency for Multiple Materials

A single saliency vector provides only a local explanation.

To understand the model globally, saliency can be calculated for many materials.

Suppose:

```text
N materials
F features
```

Then we can construct:

```text
S ∈ R^(N × F)
```

where:

```text
Sᵢⱼ
```

is the saliency of feature `j` for material `i`.

A simple implementation is:

```python
all_saliency = []

for i in range(len(X_test_tensor)):

    sample = X_test_tensor[i:i+1]

    saliency = compute_saliency(
        model,
        sample
    )

    all_saliency.append(
        saliency.detach().cpu()
    )

all_saliency = torch.cat(
    all_saliency,
    dim=0
)
```

Now:

```text
all_saliency.shape
```

will approximately be:

```text
(N, F)
```

This matrix can be analyzed statistically.

---

## 26.6.9 Global Saliency from Local Explanations

A simple global summary is the mean absolute saliency:

```python
global_saliency = (
    all_saliency.abs()
    .mean(dim=0)
)
```

Then create a table:

```python
global_table = pd.DataFrame({
    "feature": feature_names,
    "mean_saliency": (
        global_saliency
        .numpy()
    )
})

global_table = global_table.sort_values(
    "mean_saliency",
    ascending=False
)

print(global_table)
```

This produces a global ranking such as:

```text
feature                  mean_saliency

electronegativity            0.31
atomic_radius                0.24
density                      0.18
atomic_number                0.15
volume                       0.11
coordination                 0.08
```

This is useful for summarizing which features the neural network generally responds to across the dataset.

However, it should not replace local explanations.

---

## 26.6.10 Local Versus Global Explainability

The distinction is fundamental.

### Global explanation

Answers:

> What information does the model generally use?

Example:

```text
electronegativity
atomic radius
density
```

### Local explanation

Answers:

> What information did the model use for this particular material?

Example:

```text
Material A:
electronegativity → dominant

Material B:
density → dominant

Material C:
atomic radius → dominant
```

A scientifically useful Materials Informatics study should often consider both.

The workflow becomes:

```text
Dataset
   ↓
Global explanation
   ↓
Important general patterns
   ↓
Local explanations
   ↓
Material-specific mechanisms
```

---

# 26.6.11 Saliency Maps for Crystal Structures

For crystal graph models, the concept of a saliency map changes.

Instead of:

```text
Feature 1
Feature 2
Feature 3
```

the input may contain:

```text
Atom 1
Atom 2
Atom 3
...
```

and:

```text
Edge 1
Edge 2
Edge 3
...
```

The explanation therefore needs to be mapped onto the crystal graph.

Conceptually:

```text
Crystal structure
        ↓
Graph representation
        ↓
GNN
        ↓
Prediction
        ↓
Gradients
        ↓
Node / edge saliency
        ↓
Crystal visualization
```

This is one of the most important transitions from conventional tabular XAI to structural XAI.

---

## 26.6.12 Node Saliency

Let the node feature matrix be:

```text
H ∈ R^(N × F)
```

where:

* `N` = number of atoms,
* `F` = number of node features.

The gradient is:

```text
∂ŷ / ∂H
```

with the same shape:

```text
N × F
```

A node-level saliency score can be calculated by aggregating over feature dimensions.

For example:

```python
node_saliency = (
    node_gradients.abs()
    .sum(dim=1)
)
```

This produces:

```text
N values
```

one for each atom.

---

## 26.6.13 Normalizing Node Saliency

Different crystals may have different numbers of atoms and different absolute gradient magnitudes.

For visualization, it can be useful to normalize the scores:

```python
node_saliency = node_saliency / (
    node_saliency.max() + 1e-12
)
```

Now the scores lie approximately between:

```text
0 and 1
```

This allows visualization such as:

```text
0.00 → low importance
0.25 → moderate
0.50 → significant
0.75 → high
1.00 → highest
```

Normalization is primarily a visualization choice.

It does not change the underlying attribution ranking.

---

## 26.6.14 Mapping Saliency to Atomic Sites

Suppose a crystal contains:

```text
0 → Mg
1 → Al
2 → O
3 → O
4 → O
5 → O
```

and the saliency values are:

```text
0 → 0.42
1 → 1.00
2 → 0.33
3 → 0.21
4 → 0.19
5 → 0.27
```

The explanation becomes:

```text
Al site → highest saliency
Mg site → second highest
O sites → lower saliency
```

The researcher can then inspect:

```text
Al coordination
Al–O distances
local symmetry
polyhedral distortion
neighboring atoms
```

This is the beginning of structural scientific interpretation.

---

## 26.6.15 Saliency Does Not Automatically Identify a Mechanism

Suppose the Al atom has the highest node saliency.

The correct interpretation is:

> The model output is highly sensitive to the representation of this atomic site.

It is not automatically valid to conclude:

> The Al atom is physically responsible for the property.

Several possibilities remain.

The model may be responding to:

```text
Al itself
```

or:

```text
Al coordination
```

or:

```text
Al–O distances
```

or:

```text
a broader local environment
```

or even:

```text
a dataset-specific correlation
```

Therefore, node saliency should trigger structural investigation rather than end the investigation.

---

# 26.6.16 Edge Saliency

For edge features `E`, the corresponding gradient is:

```text
∂ŷ / ∂E
```

The magnitude can be used as an edge saliency score:

```python
edge_saliency = (
    edge_gradients.abs()
    .sum(dim=1)
)
```

The resulting values can be associated with graph connections:

```text
edge 0 → atom 1–atom 7 → 0.83
edge 1 → atom 1–atom 8 → 0.21
edge 2 → atom 7–atom 9 → 0.64
```

This can identify interactions that the model considers particularly influential.

---

## 26.6.17 Distance-Based Edge Features

Many crystal GNNs represent edges using interatomic distances.

For example:

```text
dᵢⱼ
```

may be expanded using Gaussian basis functions:

```text
φ_k(dᵢⱼ)
```

or another continuous radial encoding.

The model therefore does not necessarily receive:

```text
bond length = 2.13 Å
```

directly.

Instead, it may receive a vector such as:

```text
[
φ₁(d),
φ₂(d),
φ₃(d),
...
]
```

An edge attribution may therefore initially operate on this encoded representation.

To interpret it physically, the explanation should be mapped back to:

```text
atom i
atom j
interatomic distance
neighbor environment
```

rather than simply reporting the importance of an abstract neural-network channel.

---

# 26.6.18 A Crystal-Level Explanation Pipeline

A practical graph explanation pipeline can be organized as:

```text
Crystal Structure
       ↓
Pymatgen
       ↓
Crystal Graph
       ↓
Node Features + Edge Features
       ↓
GNN
       ↓
Prediction
       ↓
Gradient / Attribution
       ↓
Node Scores
+
Edge Scores
       ↓
Map Back to Structure
       ↓
Scientific Interpretation
```

The critical step is:

```text
Attribution
      ↓
Physical representation
```

Without this mapping, the explanation remains a numerical artifact rather than a useful materials-science result.

---

# 26.6.19 Example: Explaining Formation Energy

Consider a GNN predicting formation energy:

```text
Crystal
   ↓
GNN
   ↓
Formation energy
```

Suppose the model predicts:

```text
-1.87 eV/atom
```

The explanation identifies:

```text
Node 12 → high importance
Node 19 → high importance
Edge 12–19 → very high importance
```

The researcher can inspect the corresponding structural region.

Suppose these atoms belong to a distorted coordination polyhedron.

The resulting hypothesis might be:

```text
Local coordination environment
        ↓
Strong model attribution
        ↓
Potential relationship with stability
```

The hypothesis should then be tested independently.

For example:

```text
Original structure
        ↓
Modify local coordination
        ↓
DFT calculation
        ↓
Compare formation energies
```

Thus:

```text
XAI
 ↓
Hypothesis
 ↓
DFT
```

creates a scientifically testable workflow.

---

# 26.6.20 Saliency and Crystal Symmetry

Crystal symmetry adds another consideration.

Suppose a crystal contains symmetry-equivalent atoms:

```text
Atom A
Atom B
Atom C
```

that belong to equivalent crystallographic sites.

If the model assigns substantially different saliency values:

```text
A → 0.91
B → 0.42
C → 0.37
```

the researcher should investigate whether:

```text
the graph representation
```

or:

```text
the model
```

breaks a symmetry that should be preserved.

This can reveal potential issues in:

```text
graph construction
feature encoding
periodic neighbor handling
model architecture
numerical precision
```

Therefore, explainability can also serve as a diagnostic tool for materials ML pipelines.

---

# 26.6.21 Saliency as a Model-Debugging Tool

Suppose a model performs well on a test set but its saliency maps repeatedly highlight:

```text
a descriptor unrelated to the target
```

This may indicate:

```text
dataset leakage
spurious correlations
representation problems
```

For example, suppose a dataset contains:

```text
material ID
```

encoded numerically.

If the model assigns high saliency to the ID:

```text
material_ID → very high importance
```

the model may be exploiting an accidental relationship rather than learning materials science.

This demonstrates an important principle:

> Explainability is not only for explaining successful predictions; it can also reveal why a model may be learning the wrong thing.

---

# 26.6.22 Saliency and Data Leakage

Consider a dataset containing:

```text
composition
structure
target property
database identifier
```

If the identifier is accidentally included as a numerical feature:

```python
features = [
    "composition",
    "structure",
    "target",
    "database_id"
]
```

the neural network may discover artificial correlations.

A saliency analysis could expose this:

```text
database_id → unusually high attribution
```

The correct response is not to interpret the database ID scientifically.

Instead:

```text
Remove leakage
       ↓
Retrain model
       ↓
Recalculate explanation
```

This makes XAI part of model validation.

---

# 26.6.23 Saliency Stability

A scientifically meaningful explanation should ideally be reasonably stable.

Suppose we train five models:

```text
Seed 1
Seed 2
Seed 3
Seed 4
Seed 5
```

and calculate saliency for the same material.

If the results are:

```text
Feature A → high
Feature A → high
Feature A → high
Feature A → high
Feature A → high
```

the explanation is relatively stable.

But if:

```text
Seed 1 → Feature A
Seed 2 → Feature C
Seed 3 → Feature B
Seed 4 → Feature A
Seed 5 → Feature D
```

then the interpretation should be treated cautiously.

A useful research practice is therefore:

```text
Train multiple models
       ↓
Generate explanations
       ↓
Compare attribution rankings
       ↓
Measure stability
```

---

# 26.6.24 Measuring Explanation Stability

One simple approach is to compare normalized attribution vectors.

For two models:

```text
a = attribution vector from Model A
b = attribution vector from Model B
```

one can calculate similarity using:

```text
cosine similarity
```

or rank-based measures such as:

```text
Spearman correlation
```

For example:

```python
from scipy.stats import spearmanr

correlation, p_value = spearmanr(
    attribution_a,
    attribution_b
)

print(
    "Spearman correlation:",
    correlation
)
```

A high correlation indicates that the two models produce similar feature rankings.

This does not prove that the explanation is correct.

It only provides evidence that the explanation is stable across model variations.

---

# 26.6.25 Saliency Maps in a Research Workflow

A complete descriptor-based workflow can now be written as:

```text
Materials Dataset
       ↓
Feature Engineering
       ↓
Train Neural Network
       ↓
Validate Model
       ↓
Select Materials
       ↓
Calculate Saliency
       ↓
Rank Important Features
       ↓
Interpret Materials Meaning
       ↓
Check Stability
       ↓
Generate Scientific Hypothesis
       ↓
DFT / Experiment
```

For crystal GNNs:

```text
Crystal Structures
       ↓
Graph Construction
       ↓
Train GNN
       ↓
Prediction
       ↓
Node / Edge Saliency
       ↓
Map to Crystal
       ↓
Inspect Local Environment
       ↓
Scientific Hypothesis
       ↓
Validation
```

This establishes the computational foundation for more specialized graph explanation methods.

---

# 26.6.26 Limitations of Simple Saliency

Although saliency maps are useful, they should not be treated as a definitive explanation.

Important limitations include:

### 1. Gradient saturation

The gradient can become very small even when the feature is important.

### 2. Noise

Small changes in the input can produce unstable gradients.

### 3. Feature scaling

Raw gradient magnitudes depend on representation scale.

### 4. Correlated descriptors

Several features may contain overlapping information.

### 5. Baseline absence

Simple saliency does not explicitly define a reference state.

### 6. Model dependence

Different trained models can produce different saliency maps.

### 7. No automatic causality

Attribution does not establish a physical causal mechanism.

These limitations motivate methods specifically designed for graph-structured data.

---

# 26.6.27 From Saliency to Graph Explainability

For a crystal graph:

```text
G = (V, E)
```

a useful explanation should ideally answer three questions:

```text
1. Which atoms matter?

2. Which interactions matter?

3. Which atomic features matter?
```

Simple gradient saliency can provide initial answers.

However, modern graph neural networks contain message-passing operations:

```text
Node
 ↓
Neighbors
 ↓
Message
 ↓
Updated node representation
 ↓
More message passing
 ↓
Prediction
```

Therefore, the model may rely on complex subgraphs rather than individual atoms or edges.

A more advanced explanation method should therefore identify **important graph structures**.

This leads to dedicated graph-explanation approaches such as **GNNExplainer**.

---

# 26.6.28 Transition to GNNExplainer

The progression developed so far is:

```text
Neural Network
      ↓
Gradient
      ↓
Saliency
      ↓
Feature attribution
      ↓
Integrated Gradients
```

For crystal graph neural networks, we now extend this:

```text
Crystal Graph
      ↓
GNN
      ↓
Prediction
      ↓
Important nodes
      +
Important edges
      +
Important features
      ↓
Important subgraph
```

This is the central motivation for **GNNExplainer**.

Rather than simply asking:

> Which individual input values have large gradients?

GNNExplainer asks a more structural question:

> Which parts of the graph and which node features are most important for the model's prediction?

This distinction is particularly valuable for Materials Informatics because a crystal is naturally represented as a graph.

The next section develops GNNExplainer in detail, including its optimization objective, node and edge masks, PyTorch Geometric implementation, interpretation of important atomic environments, and the limitations that must be considered when applying graph explanations to crystal materials.

# 26.7 GNNExplainer for Crystal Graphs

Saliency methods provide a useful first level of graph interpretation.

They can indicate which node or edge representations are locally influential, but they do not necessarily identify a coherent structural explanation.

A crystal graph, however, is not merely a collection of independent atoms.

Its prediction emerges from relationships among:

```text
Atoms
 ↓
Local environments
 ↓
Neighbor interactions
 ↓
Message passing
 ↓
Higher-order structural patterns
 ↓
Crystal-level representation
 ↓
Prediction
```

Consequently, an explanation method for crystal GNNs should ideally identify not only important features, but also the **subgraph that is most relevant to the prediction**.

GNNExplainer was developed around this idea.

Instead of directly asking which input gradient is largest, GNNExplainer searches for a compact combination of:

```text
Important nodes
+
Important edges
+
Important node features
```

that preserves the model's prediction.

For Materials Informatics, this creates a natural interpretation:

```text
Crystal structure
      ↓
Crystal graph
      ↓
GNN prediction
      ↓
GNNExplainer
      ↓
Important structural region
      ↓
Atomic-scale interpretation
```

The important distinction is that GNNExplainer attempts to explain the **model's decision using a subgraph and feature mask**, rather than simply reporting raw gradients.

---

## 26.7.1 The Core Question

Suppose a crystal contains 40 atoms.

Its graph may contain hundreds of edges after periodic neighbor construction.

The GNN predicts:

```text
Formation energy = -1.82 eV/atom
```

A natural scientific question is:

> Which part of the crystal caused the model to make this prediction?

A naive approach might return:

```text
Atom 1 → important
Atom 17 → important
Feature 4 → important
```

But this does not tell us whether the important atoms form a meaningful structural environment.

GNNExplainer instead attempts to identify a subgraph such as:

```text
        O
        |
    O — Ti — O
        |
        O
```

and determine whether this local environment is especially important for the prediction.

This is much closer to the language used by materials scientists:

```text
coordination environment
local bonding environment
polyhedral distortion
neighbor interactions
structural motif
```

---

## 26.7.2 Crystal Graph Representation

Let a crystal be represented as:

```text
G = (V, E)
```

where:

* `V` represents atoms,
* `E` represents graph connections between atoms.

The node-feature matrix is:

```text
X ∈ R^(N × F)
```

where:

* `N` = number of atoms,
* `F` = number of node features.

The edge structure is represented by:

```text
A
```

or, in practical graph-learning frameworks, by an `edge_index` tensor.

A GNN computes:

```text
ŷ = f(G, X)
```

where `ŷ` is the predicted material property.

The explanation problem is therefore:

```text
Given:

G
X
f
ŷ

find:

important subgraph
+
important features
```

---

## 26.7.3 Why a Subgraph Matters

Consider two situations.

### Situation A

Two atoms are important individually:

```text
Atom 5 → high importance
Atom 21 → high importance
```

but they are far apart and have no meaningful structural relationship.

### Situation B

Three neighboring atoms form a local coordination environment:

```text
Atom 5
 /   \
Atom 21 — Atom 8
```

and the entire structure receives high importance.

Situation B is generally more useful scientifically because it provides a structural hypothesis.

The researcher can ask:

```text
What local coordination exists here?

What are the interatomic distances?

What are the bond angles?

Is the environment distorted?

Is the motif chemically unusual?

Does DFT support its importance?
```

Thus, graph explanations can connect machine learning with structural materials science.

---

# 26.7.4 Node Masks

One component of GNNExplainer is a node-feature mask.

Conceptually:

```text
M_X
```

has the same shape as the node-feature matrix:

```text
M_X ∈ R^(N × F)
```

Each value indicates how important a particular feature of a particular node is.

For example:

```text
             Z     radius    electronegativity
Atom 1      0.2      0.1          0.3
Atom 2      0.9      0.8          0.7
Atom 3      0.1      0.2          0.1
```

This suggests that the model is especially sensitive to the representation of Atom 2.

The mask can be aggregated over features to obtain node-level importance.

For example:

```text
Atom 1 → 0.20
Atom 2 → 0.80
Atom 3 → 0.13
```

---

## 26.7.5 Edge Masks

A second important component is the edge mask.

Let:

```text
M_E
```

represent edge importance.

For a graph with `|E|` edges:

```text
M_E ∈ R^|E|
```

A possible result is:

```text
edge 0 → 0.08
edge 1 → 0.91
edge 2 → 0.15
edge 3 → 0.72
```

The highly weighted edges can then be mapped back to the crystal structure.

For example:

```text
edge 1
↓
Ti — O
↓
distance = 1.94 Å
```

This is much easier to interpret scientifically than an abstract neural-network channel.

---

# 26.7.6 From Masks to a Structural Explanation

The explanation workflow becomes:

```text
GNN
 ↓
Prediction
 ↓
Node mask
+
Edge mask
 ↓
Threshold important values
 ↓
Extract subgraph
 ↓
Map atoms back to crystal
 ↓
Analyze local structure
```

For example:

```text
Important atoms:
Ti(3), O(7), O(12), O(18)

Important edges:
Ti(3)-O(7)
Ti(3)-O(12)
Ti(3)-O(18)
```

This may reveal a local Ti–O coordination environment.

The researcher can then calculate:

```text
Ti–O distances
O–Ti–O angles
coordination number
polyhedral distortion
local symmetry
```

The GNN explanation has therefore produced a structural hypothesis.

---

# 26.7.7 The Optimization Perspective

GNNExplainer does not simply read an existing importance value.

It searches for a mask that preserves the model's behavior.

Let:

```text
G_S
```

be an explanatory subgraph.

The goal is approximately:

```text
f(G_S) ≈ f(G)
```

while keeping the explanation sufficiently small and interpretable.

Conceptually:

```text
Full crystal graph
       ↓
Remove unnecessary information
       ↓
Keep important structure
       ↓
Prediction remains similar
```

This creates an optimization problem.

The explanation should be:

```text
small
+
predictive
+
informative
```

rather than simply containing every atom and every edge.

---

# 26.7.8 Why Sparsity Matters

Suppose the original crystal graph contains:

```text
N = 50 atoms
```

and:

```text
E = 500 edges
```

If the explanation contains all 50 atoms and all 500 edges, it provides almost no simplification.

A useful explanation might instead identify:

```text
8 important atoms
```

and:

```text
12 important edges
```

This is much easier to inspect.

The principle is:

```text
Full graph
   ↓
Compact explanatory graph
```

The smaller explanatory structure is often easier to connect with materials-science concepts.

---

# 26.7.9 Prediction Preservation

Suppose the full graph produces:

```text
ŷ = 2.43
```

The selected explanatory subgraph might produce:

```text
ŷ_explanation = 2.39
```

The predictions are relatively close.

If removing most of the graph changes the prediction dramatically:

```text
2.43 → 0.71
```

then the selected subgraph is not sufficient to explain the original decision.

This illustrates the central criterion:

> A good explanation should preserve the behavior of the original model while simplifying the input.

---

# 26.7.10 GNNExplainer in PyTorch Geometric

PyTorch Geometric provides graph-explanation infrastructure that can be used with GNN models.

A typical environment begins with:

```python
import torch
import torch.nn as nn

from torch_geometric.explain import (
    Explainer,
    GNNExplainer
)
```

The exact API can vary across PyTorch Geometric versions, so a reproducible research implementation should record:

```python
print(torch.__version__)
```

and:

```python
import torch_geometric

print(torch_geometric.__version__)
```

Version recording is important because explanation APIs have evolved across releases.

---

# 26.7.11 Example Crystal GNN

Consider a simplified graph neural network:

```python
import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_mean_pool
)


class CrystalGNN(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.fc = torch.nn.Linear(
            hidden_channels,
            out_channels
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

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        return self.fc(x)
```

This model follows:

```text
Node features
      ↓
Graph convolution
      ↓
Graph convolution
      ↓
Global pooling
      ↓
Crystal representation
      ↓
Property prediction
```

The same conceptual architecture appears in many crystal GNN systems, although real Materials Informatics models may use substantially more sophisticated message-passing and geometric representations.

---

# 26.7.12 Preparing an Explanation Configuration

A typical PyTorch Geometric explanation configuration specifies:

```python
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(
        epochs=200
    ),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="regression",
        task_level="graph",
        return_type="raw"
    )
)
```

The important components are:

```text
model
 ↓
GNNExplainer
 ↓
node mask
 ↓
edge mask
 ↓
model configuration
```

For a materials property regression problem:

```text
mode = regression
```

is appropriate.

For a crystal-level prediction:

```text
task_level = graph
```

is appropriate.

The exact configuration should match the model's actual output structure.

---

# 26.7.13 Explaining One Crystal

Suppose a PyTorch Geometric `Data` object is:

```python
data
```

with:

```python
data.x
data.edge_index
data.batch
```

The explanation can be generated with:

```python
explanation = explainer(
    data.x,
    data.edge_index,
    batch=data.batch
)
```

The resulting explanation can contain:

```python
explanation.node_mask
```

and:

```python
explanation.edge_mask
```

depending on the configured mask types and PyTorch Geometric version.

The researcher can inspect their shapes:

```python
print(
    explanation.node_mask.shape
)

print(
    explanation.edge_mask.shape
)
```

This is the bridge between the abstract explanation algorithm and the actual crystal graph.

---

# 26.7.14 Interpreting the Node Mask

Suppose:

```python
node_mask = explanation.node_mask
```

contains one score per atom-feature pair.

A node-level importance score can be obtained by aggregation:

```python
node_importance = (
    node_mask.abs()
    .sum(dim=-1)
)
```

Normalize it:

```python
node_importance = (
    node_importance
    /
    (node_importance.max() + 1e-12)
)
```

Then:

```python
print(node_importance)
```

might produce:

```text
tensor([
    0.05,
    0.12,
    0.91,
    0.84,
    0.18,
    ...
])
```

Atoms 2 and 3 would then be candidates for detailed structural inspection.

---

# 26.7.15 Interpreting the Edge Mask

Similarly:

```python
edge_importance = (
    explanation.edge_mask
)
```

can be normalized:

```python
edge_importance = (
    edge_importance
    /
    (edge_importance.max() + 1e-12)
)
```

Now an edge score close to:

```text
1
```

indicates that the edge received a high explanatory weight relative to other edges in that explanation.

It should not automatically be interpreted as:

```text
physical bond strength
```

This distinction is critical.

A graph edge may represent a neighbor relationship created by a cutoff radius rather than an actual chemical bond.

Therefore:

```text
edge importance
≠
bond strength
```

unless the representation explicitly encodes a physically meaningful bond and additional evidence supports that interpretation.

---

# 26.7.16 Crystal Neighbor Graphs Are Not Always Bond Graphs

This issue is particularly important in crystal machine learning.

Suppose the graph is constructed using:

```text
cutoff = 5 Å
```

Then an edge may mean:

```text
two atoms are within 5 Å
```

It does not necessarily mean:

```text
chemical bond exists
```

Therefore, if GNNExplainer identifies:

```text
Atom A — Atom B
```

as important, the researcher should first determine:

```text
Why does the graph contain this edge?
```

Possible meanings include:

```text
nearest-neighbor relationship
periodic neighbor
distance-based interaction
bond
coordination relationship
```

The graph-construction definition must always be considered before assigning physical meaning to an edge explanation.

---

# 26.7.17 Selecting Important Edges

A simple threshold can be used for visualization.

For example:

```python
threshold = 0.7

important_edges = (
    edge_importance
    > threshold
)
```

The selected edge indices are:

```python
selected_edge_indices = torch.where(
    important_edges
)[0]
```

These indices can then be mapped back to:

```python
edge_index
```

For example:

```python
important_connections = (
    data.edge_index[
        :,
        selected_edge_indices
    ]
)
```

The resulting tensor identifies the atom pairs belonging to the explanatory subgraph.

---

# 26.7.18 Extracting an Explanatory Subgraph

Suppose the selected edges are:

```text
Atom 4 — Atom 8
Atom 4 — Atom 11
Atom 8 — Atom 13
```

The explanatory graph becomes:

```text
        Atom 11
           |
           |
Atom 8 — Atom 4
   |
   |
Atom 13
```

This can then be mapped back to the original crystal.

The researcher may discover:

```text
a coordination polyhedron
```

or:

```text
a defect environment
```

or:

```text
a chemically unusual local motif
```

This is precisely the type of structural information that graph explainability can provide.

---

# 26.7.19 Mapping Graph Nodes to Chemical Species

Suppose the graph stores atomic numbers:

```python
atomic_numbers = data.atomic_number
```

and important node indices are:

```python
important_nodes = torch.tensor([
    4,
    8,
    11
])
```

The corresponding atomic numbers can be obtained:

```python
important_Z = atomic_numbers[
    important_nodes
]
```

If the values are:

```text
22
8
8
```

the researcher can interpret them as:

```text
Ti
O
O
```

assuming the standard atomic-number mapping.

This produces a chemically meaningful explanation:

```text
Important subgraph:

Ti
├── O
└── O
```

rather than simply:

```text
Node 4
Node 8
Node 11
```

---

# 26.7.20 Mapping Nodes to Pymatgen Structures

If the original structure is stored as a Pymatgen `Structure`:

```python
from pymatgen.core import Structure

structure = Structure.from_file(
    "material.cif"
)
```

the graph node index can be mapped to:

```python
structure[i]
```

For example:

```python
site = structure[4]

print(
    site.species
)

print(
    site.frac_coords
)

print(
    site.coords
)
```

This provides:

```text
chemical species
fractional coordinates
Cartesian coordinates
```

for the important site.

Now the explanation can be connected directly to the crystallographic structure.

---

# 26.7.21 Calculating Local Coordination

Once an important atom has been identified, its local environment can be inspected with Pymatgen.

For example:

```python
site_index = 4

neighbors = structure.get_neighbors(
    structure[site_index],
    r=3.0
)

for neighbor in neighbors:

    print(
        neighbor.species,
        neighbor.distance
    )
```

This produces information such as:

```text
O    1.91 Å
O    1.94 Å
O    2.03 Å
O    2.11 Å
```

The researcher can now ask whether the model's important node corresponds to:

```text
short bonds
long bonds
coordination changes
distorted geometry
```

This is where graph explainability becomes genuinely materials-specific.

---

# 26.7.22 Example Scientific Interpretation

Suppose a GNN predicts:

```text
band gap = 2.31 eV
```

GNNExplainer identifies a local region containing:

```text
transition-metal atom
+
four oxygen neighbors
```

with high node and edge importance.

Pymatgen analysis shows:

```text
four nearest-neighbor distances
=
1.88 Å
1.91 Å
2.12 Å
2.18 Å
```

This suggests a distorted local coordination environment.

The correct scientific workflow is not:

```text
GNNExplainer
↓
Therefore distortion causes band gap
```

Instead:

```text
GNNExplainer
↓
Model highlights distorted environment
↓
Hypothesis:
local distortion may influence band gap
↓
Construct controlled structural variants
↓
DFT calculations
↓
Compare electronic structure
↓
Evaluate hypothesis
```

The explanation therefore becomes a **hypothesis-generation tool**.

---

# 26.7.23 GNNExplainer and Message Passing

The importance of GNNExplainer becomes clearer when considering message passing.

A typical GNN layer can be represented conceptually as:

```text
hᵢ^(l+1)
=
UPDATE(
    hᵢ^(l),
    AGGREGATE(
        hⱼ^(l),
        eᵢⱼ
    )
)
```

The representation of an atom therefore depends on neighboring atoms.

After several layers:

```text
Layer 1
↓
nearest neighbors

Layer 2
↓
two-hop environment

Layer 3
↓
larger structural environment
```

Consequently, a crystal-level prediction may depend on a distributed structural pattern.

GNNExplainer attempts to identify which graph components contribute most strongly to this prediction.

---

# 26.7.24 Receptive Field and Explanation Size

Suppose the GNN contains:

```text
3 message-passing layers
```

An atom can potentially receive information from approximately three-hop neighborhoods, depending on the architecture.

Therefore, an explanation containing several neighboring atoms may reflect the model's receptive field.

Conceptually:

```text
Central atom
     ↓
1-hop neighbors
     ↓
2-hop neighbors
     ↓
3-hop neighbors
```

The researcher should therefore avoid assuming that the explanation must correspond to a single chemical bond or a single coordination shell.

The model may be using a larger structural environment.

---

# 26.7.25 Comparing GNNExplainer with Saliency

The methods can now be compared.

| Method               | Main signal              | Typical materials interpretation          |
| -------------------- | ------------------------ | ----------------------------------------- |
| Saliency             | Local gradient           | Feature sensitivity                       |
| Integrated Gradients | Path-integrated gradient | Feature contribution relative to baseline |
| GNNExplainer         | Optimized masks          | Important nodes, edges, and subgraphs     |

The progression is:

```text
Saliency
   ↓
Which input is locally sensitive?

Integrated Gradients
   ↓
How did the input contribute relative to a baseline?

GNNExplainer
   ↓
Which graph structure best preserves the prediction?
```

This hierarchy is useful because no single explanation method answers every scientific question.

---

# 26.7.26 Explanation Agreement

For a high-confidence scientific interpretation, multiple explanation methods can be compared.

Suppose:

```text
Integrated Gradients
        ↓
transition-metal features important

Saliency
        ↓
transition-metal features important

GNNExplainer
        ↓
transition-metal-centered subgraph important
```

The agreement is encouraging.

The three methods are identifying related information from different perspectives.

However, disagreement is also informative.

For example:

```text
Integrated Gradients
        ↓
Feature A

GNNExplainer
        ↓
Feature B / structural motif
```

may indicate:

```text
representation dependence
feature correlation
model instability
explanation-method sensitivity
```

Such disagreements should be investigated rather than ignored.

---

# 26.7.27 Explanation Stability in GNNs

As with descriptor models, graph explanations can vary across:

```text
random seeds
model checkpoints
training splits
hyperparameters
graph construction settings
```

A robust study should therefore avoid presenting a single explanation as universally valid.

A stronger workflow is:

```text
Train multiple GNNs
       ↓
Explain the same crystal
       ↓
Compare node masks
       ↓
Compare edge masks
       ↓
Identify stable structural regions
```

For example:

```text
Model 1 → Ti-centered region
Model 2 → Ti-centered region
Model 3 → Ti-centered region
Model 4 → Ti-centered region
```

provides stronger evidence than one isolated explanation.

---

# 26.7.28 Reproducible GNN Explanation

A research implementation should record:

```text
Dataset version
Crystal preprocessing
Graph cutoff
Neighbor-generation method
Node features
Edge features
GNN architecture
Model weights
Random seed
PyTorch version
PyTorch Geometric version
GNNExplainer configuration
Optimization epochs
Mask thresholds
```

For example:

```python
import torch
import torch_geometric

print("PyTorch:", torch.__version__)
print(
    "PyG:",
    torch_geometric.__version__
)
```

The graph-construction parameters should also be stored.

For example:

```python
graph_config = {
    "cutoff": 5.0,
    "max_neighbors": 12,
    "periodic": True
}
```

This matters because changing the graph construction changes the explanation space itself.

---

# 26.7.29 Important Limitation: Explanation Is Model-Specific

Suppose:

```text
GNN A
```

and:

```text
GNN B
```

both predict band gap.

GNN A may use:

```text
local coordination
```

while GNN B uses:

```text
composition-derived information
```

even though both achieve similar test error.

Therefore:

```text
prediction agreement
≠
reasoning agreement
```

This is a crucial lesson in scientific machine learning.

Two models can make similar predictions while relying on different representations of the data.

XAI allows the researcher to investigate this difference.

---

# 26.7.30 Important Limitation: Mask Scores Are Not Physical Quantities

An edge mask such as:

```text
0.92
```

does not mean:

```text
92% bond strength
```

Similarly, a node mask:

```text
0.75
```

does not mean:

```text
atom contributes 75% of the physical property
```

These values are explanation-model quantities.

They should be interpreted comparatively:

```text
higher mask
→ more important within this explanation
```

rather than as direct physical measurements.

---

# 26.7.31 From Explanation to Scientific Hypothesis

A useful Materials Informatics explanation workflow is:

```text
Model prediction
      ↓
Explanation
      ↓
Important atoms / edges
      ↓
Chemical interpretation
      ↓
Structural analysis
      ↓
Physical hypothesis
      ↓
Independent validation
```

The final stage is essential.

Without independent validation, an explanation remains a model-based interpretation.

With validation, it can become part of a scientific argument.

For example:

```text
GNN
 ↓
high importance of distorted octahedron
 ↓
hypothesis:
distortion affects stability
 ↓
DFT calculations
 ↓
energy comparison
 ↓
validated structural trend
```

This is a much stronger research workflow than simply plotting an explanation figure.

---

## 26.7.32 Practical Research Checklist

Before publishing a GNN explanation, verify:

```text
[ ] The model has been independently validated.

[ ] The crystal graph construction is documented.

[ ] The meaning of graph edges is clearly defined.

[ ] Node and edge masks are interpreted comparatively.

[ ] The target output being explained is specified.

[ ] Explanation stability has been examined.

[ ] Important atoms are mapped back to actual species.

[ ] Important edges are mapped to structural relationships.

[ ] Local coordination environments are analyzed.

[ ] Physical interpretation is separated from model attribution.

[ ] Causal claims are not made from attribution alone.

[ ] Important hypotheses are independently tested.
```

This checklist prevents one of the most common mistakes in scientific XAI:

```text
visual explanation
        ↓
unverified physical conclusion
```

---

## 26.7.33 Transition to Feature-Level Attribution in Deep Models

GNNExplainer provides a structural explanation by identifying important graph components.

However, it does not completely answer another important question:

> Which specific input features inside those atoms or interactions are responsible?

Suppose GNNExplainer identifies:

```text
Ti-centered local environment
```

as important.

We may still want to know whether the model is responding primarily to:

```text
atomic number
electronegativity
valence information
atomic radius
coordination
distance
angular information
```

This requires returning from the graph level to the feature level.

The hierarchy of explanations can therefore be written as:

```text
Crystal
   ↓
Important subgraph
   ↓
Important atoms
   ↓
Important edges
   ↓
Important features
   ↓
Scientific interpretation
```

The combination of these levels is considerably more informative than using any one explanation technique in isolation.

The next section therefore develops **attention visualization in materials GNNs**, where learned attention weights can be examined to understand how information is selectively propagated through atomic neighborhoods.

# 26.7 GNNExplainer for Crystal Graphs

Saliency methods provide a useful first level of graph interpretation.

They can indicate which node or edge representations are locally influential, but they do not necessarily identify a coherent structural explanation.

A crystal graph, however, is not merely a collection of independent atoms.

Its prediction emerges from relationships among:

```text
Atoms
 ↓
Local environments
 ↓
Neighbor interactions
 ↓
Message passing
 ↓
Higher-order structural patterns
 ↓
Crystal-level representation
 ↓
Prediction
```

Consequently, an explanation method for crystal GNNs should ideally identify not only important features, but also the **subgraph that is most relevant to the prediction**.

GNNExplainer was developed around this idea.

Instead of directly asking which input gradient is largest, GNNExplainer searches for a compact combination of:

```text
Important nodes
+
Important edges
+
Important node features
```

that preserves the model's prediction.

For Materials Informatics, this creates a natural interpretation:

```text
Crystal structure
      ↓
Crystal graph
      ↓
GNN prediction
      ↓
GNNExplainer
      ↓
Important structural region
      ↓
Atomic-scale interpretation
```

The important distinction is that GNNExplainer attempts to explain the **model's decision using a subgraph and feature mask**, rather than simply reporting raw gradients.

---

## 26.7.1 The Core Question

Suppose a crystal contains 40 atoms.

Its graph may contain hundreds of edges after periodic neighbor construction.

The GNN predicts:

```text
Formation energy = -1.82 eV/atom
```

A natural scientific question is:

> Which part of the crystal caused the model to make this prediction?

A naive approach might return:

```text
Atom 1 → important
Atom 17 → important
Feature 4 → important
```

But this does not tell us whether the important atoms form a meaningful structural environment.

GNNExplainer instead attempts to identify a subgraph such as:

```text
        O
        |
    O — Ti — O
        |
        O
```

and determine whether this local environment is especially important for the prediction.

This is much closer to the language used by materials scientists:

```text
coordination environment
local bonding environment
polyhedral distortion
neighbor interactions
structural motif
```

---

## 26.7.2 Crystal Graph Representation

Let a crystal be represented as:

```text
G = (V, E)
```

where:

* `V` represents atoms,
* `E` represents graph connections between atoms.

The node-feature matrix is:

```text
X ∈ R^(N × F)
```

where:

* `N` = number of atoms,
* `F` = number of node features.

The edge structure is represented by:

```text
A
```

or, in practical graph-learning frameworks, by an `edge_index` tensor.

A GNN computes:

```text
ŷ = f(G, X)
```

where `ŷ` is the predicted material property.

The explanation problem is therefore:

```text
Given:

G
X
f
ŷ

find:

important subgraph
+
important features
```

---

## 26.7.3 Why a Subgraph Matters

Consider two situations.

### Situation A

Two atoms are important individually:

```text
Atom 5 → high importance
Atom 21 → high importance
```

but they are far apart and have no meaningful structural relationship.

### Situation B

Three neighboring atoms form a local coordination environment:

```text
Atom 5
 /   \
Atom 21 — Atom 8
```

and the entire structure receives high importance.

Situation B is generally more useful scientifically because it provides a structural hypothesis.

The researcher can ask:

```text
What local coordination exists here?

What are the interatomic distances?

What are the bond angles?

Is the environment distorted?

Is the motif chemically unusual?

Does DFT support its importance?
```

Thus, graph explanations can connect machine learning with structural materials science.

---

# 26.7.4 Node Masks

One component of GNNExplainer is a node-feature mask.

Conceptually:

```text
M_X
```

has the same shape as the node-feature matrix:

```text
M_X ∈ R^(N × F)
```

Each value indicates how important a particular feature of a particular node is.

For example:

```text
             Z     radius    electronegativity
Atom 1      0.2      0.1          0.3
Atom 2      0.9      0.8          0.7
Atom 3      0.1      0.2          0.1
```

This suggests that the model is especially sensitive to the representation of Atom 2.

The mask can be aggregated over features to obtain node-level importance.

For example:

```text
Atom 1 → 0.20
Atom 2 → 0.80
Atom 3 → 0.13
```

---

## 26.7.5 Edge Masks

A second important component is the edge mask.

Let:

```text
M_E
```

represent edge importance.

For a graph with `|E|` edges:

```text
M_E ∈ R^|E|
```

A possible result is:

```text
edge 0 → 0.08
edge 1 → 0.91
edge 2 → 0.15
edge 3 → 0.72
```

The highly weighted edges can then be mapped back to the crystal structure.

For example:

```text
edge 1
↓
Ti — O
↓
distance = 1.94 Å
```

This is much easier to interpret scientifically than an abstract neural-network channel.

---

# 26.7.6 From Masks to a Structural Explanation

The explanation workflow becomes:

```text
GNN
 ↓
Prediction
 ↓
Node mask
+
Edge mask
 ↓
Threshold important values
 ↓
Extract subgraph
 ↓
Map atoms back to crystal
 ↓
Analyze local structure
```

For example:

```text
Important atoms:
Ti(3), O(7), O(12), O(18)

Important edges:
Ti(3)-O(7)
Ti(3)-O(12)
Ti(3)-O(18)
```

This may reveal a local Ti–O coordination environment.

The researcher can then calculate:

```text
Ti–O distances
O–Ti–O angles
coordination number
polyhedral distortion
local symmetry
```

The GNN explanation has therefore produced a structural hypothesis.

---

# 26.7.7 The Optimization Perspective

GNNExplainer does not simply read an existing importance value.

It searches for a mask that preserves the model's behavior.

Let:

```text
G_S
```

be an explanatory subgraph.

The goal is approximately:

```text
f(G_S) ≈ f(G)
```

while keeping the explanation sufficiently small and interpretable.

Conceptually:

```text
Full crystal graph
       ↓
Remove unnecessary information
       ↓
Keep important structure
       ↓
Prediction remains similar
```

This creates an optimization problem.

The explanation should be:

```text
small
+
predictive
+
informative
```

rather than simply containing every atom and every edge.

---

# 26.7.8 Why Sparsity Matters

Suppose the original crystal graph contains:

```text
N = 50 atoms
```

and:

```text
E = 500 edges
```

If the explanation contains all 50 atoms and all 500 edges, it provides almost no simplification.

A useful explanation might instead identify:

```text
8 important atoms
```

and:

```text
12 important edges
```

This is much easier to inspect.

The principle is:

```text
Full graph
   ↓
Compact explanatory graph
```

The smaller explanatory structure is often easier to connect with materials-science concepts.

---

# 26.7.9 Prediction Preservation

Suppose the full graph produces:

```text
ŷ = 2.43
```

The selected explanatory subgraph might produce:

```text
ŷ_explanation = 2.39
```

The predictions are relatively close.

If removing most of the graph changes the prediction dramatically:

```text
2.43 → 0.71
```

then the selected subgraph is not sufficient to explain the original decision.

This illustrates the central criterion:

> A good explanation should preserve the behavior of the original model while simplifying the input.

---

# 26.7.10 GNNExplainer in PyTorch Geometric

PyTorch Geometric provides graph-explanation infrastructure that can be used with GNN models.

A typical environment begins with:

```python
import torch
import torch.nn as nn

from torch_geometric.explain import (
    Explainer,
    GNNExplainer
)
```

The exact API can vary across PyTorch Geometric versions, so a reproducible research implementation should record:

```python
print(torch.__version__)
```

and:

```python
import torch_geometric

print(torch_geometric.__version__)
```

Version recording is important because explanation APIs have evolved across releases.

---

# 26.7.11 Example Crystal GNN

Consider a simplified graph neural network:

```python
import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_mean_pool
)


class CrystalGNN(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.fc = torch.nn.Linear(
            hidden_channels,
            out_channels
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

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        return self.fc(x)
```

This model follows:

```text
Node features
      ↓
Graph convolution
      ↓
Graph convolution
      ↓
Global pooling
      ↓
Crystal representation
      ↓
Property prediction
```

The same conceptual architecture appears in many crystal GNN systems, although real Materials Informatics models may use substantially more sophisticated message-passing and geometric representations.

---

# 26.7.12 Preparing an Explanation Configuration

A typical PyTorch Geometric explanation configuration specifies:

```python
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(
        epochs=200
    ),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="regression",
        task_level="graph",
        return_type="raw"
    )
)
```

The important components are:

```text
model
 ↓
GNNExplainer
 ↓
node mask
 ↓
edge mask
 ↓
model configuration
```

For a materials property regression problem:

```text
mode = regression
```

is appropriate.

For a crystal-level prediction:

```text
task_level = graph
```

is appropriate.

The exact configuration should match the model's actual output structure.

---

# 26.7.13 Explaining One Crystal

Suppose a PyTorch Geometric `Data` object is:

```python
data
```

with:

```python
data.x
data.edge_index
data.batch
```

The explanation can be generated with:

```python
explanation = explainer(
    data.x,
    data.edge_index,
    batch=data.batch
)
```

The resulting explanation can contain:

```python
explanation.node_mask
```

and:

```python
explanation.edge_mask
```

depending on the configured mask types and PyTorch Geometric version.

The researcher can inspect their shapes:

```python
print(
    explanation.node_mask.shape
)

print(
    explanation.edge_mask.shape
)
```

This is the bridge between the abstract explanation algorithm and the actual crystal graph.

---

# 26.7.14 Interpreting the Node Mask

Suppose:

```python
node_mask = explanation.node_mask
```

contains one score per atom-feature pair.

A node-level importance score can be obtained by aggregation:

```python
node_importance = (
    node_mask.abs()
    .sum(dim=-1)
)
```

Normalize it:

```python
node_importance = (
    node_importance
    /
    (node_importance.max() + 1e-12)
)
```

Then:

```python
print(node_importance)
```

might produce:

```text
tensor([
    0.05,
    0.12,
    0.91,
    0.84,
    0.18,
    ...
])
```

Atoms 2 and 3 would then be candidates for detailed structural inspection.

---

# 26.7.15 Interpreting the Edge Mask

Similarly:

```python
edge_importance = (
    explanation.edge_mask
)
```

can be normalized:

```python
edge_importance = (
    edge_importance
    /
    (edge_importance.max() + 1e-12)
)
```

Now an edge score close to:

```text
1
```

indicates that the edge received a high explanatory weight relative to other edges in that explanation.

It should not automatically be interpreted as:

```text
physical bond strength
```

This distinction is critical.

A graph edge may represent a neighbor relationship created by a cutoff radius rather than an actual chemical bond.

Therefore:

```text
edge importance
≠
bond strength
```

unless the representation explicitly encodes a physically meaningful bond and additional evidence supports that interpretation.

---

# 26.7.16 Crystal Neighbor Graphs Are Not Always Bond Graphs

This issue is particularly important in crystal machine learning.

Suppose the graph is constructed using:

```text
cutoff = 5 Å
```

Then an edge may mean:

```text
two atoms are within 5 Å
```

It does not necessarily mean:

```text
chemical bond exists
```

Therefore, if GNNExplainer identifies:

```text
Atom A — Atom B
```

as important, the researcher should first determine:

```text
Why does the graph contain this edge?
```

Possible meanings include:

```text
nearest-neighbor relationship
periodic neighbor
distance-based interaction
bond
coordination relationship
```

The graph-construction definition must always be considered before assigning physical meaning to an edge explanation.

---

# 26.7.17 Selecting Important Edges

A simple threshold can be used for visualization.

For example:

```python
threshold = 0.7

important_edges = (
    edge_importance
    > threshold
)
```

The selected edge indices are:

```python
selected_edge_indices = torch.where(
    important_edges
)[0]
```

These indices can then be mapped back to:

```python
edge_index
```

For example:

```python
important_connections = (
    data.edge_index[
        :,
        selected_edge_indices
    ]
)
```

The resulting tensor identifies the atom pairs belonging to the explanatory subgraph.

---

# 26.7.18 Extracting an Explanatory Subgraph

Suppose the selected edges are:

```text
Atom 4 — Atom 8
Atom 4 — Atom 11
Atom 8 — Atom 13
```

The explanatory graph becomes:

```text
        Atom 11
           |
           |
Atom 8 — Atom 4
   |
   |
Atom 13
```

This can then be mapped back to the original crystal.

The researcher may discover:

```text
a coordination polyhedron
```

or:

```text
a defect environment
```

or:

```text
a chemically unusual local motif
```

This is precisely the type of structural information that graph explainability can provide.

---

# 26.7.19 Mapping Graph Nodes to Chemical Species

Suppose the graph stores atomic numbers:

```python
atomic_numbers = data.atomic_number
```

and important node indices are:

```python
important_nodes = torch.tensor([
    4,
    8,
    11
])
```

The corresponding atomic numbers can be obtained:

```python
important_Z = atomic_numbers[
    important_nodes
]
```

If the values are:

```text
22
8
8
```

the researcher can interpret them as:

```text
Ti
O
O
```

assuming the standard atomic-number mapping.

This produces a chemically meaningful explanation:

```text
Important subgraph:

Ti
├── O
└── O
```

rather than simply:

```text
Node 4
Node 8
Node 11
```

---

# 26.7.20 Mapping Nodes to Pymatgen Structures

If the original structure is stored as a Pymatgen `Structure`:

```python
from pymatgen.core import Structure

structure = Structure.from_file(
    "material.cif"
)
```

the graph node index can be mapped to:

```python
structure[i]
```

For example:

```python
site = structure[4]

print(
    site.species
)

print(
    site.frac_coords
)

print(
    site.coords
)
```

This provides:

```text
chemical species
fractional coordinates
Cartesian coordinates
```

for the important site.

Now the explanation can be connected directly to the crystallographic structure.

---

# 26.7.21 Calculating Local Coordination

Once an important atom has been identified, its local environment can be inspected with Pymatgen.

For example:

```python
site_index = 4

neighbors = structure.get_neighbors(
    structure[site_index],
    r=3.0
)

for neighbor in neighbors:

    print(
        neighbor.species,
        neighbor.distance
    )
```

This produces information such as:

```text
O    1.91 Å
O    1.94 Å
O    2.03 Å
O    2.11 Å
```

The researcher can now ask whether the model's important node corresponds to:

```text
short bonds
long bonds
coordination changes
distorted geometry
```

This is where graph explainability becomes genuinely materials-specific.

---

# 26.7.22 Example Scientific Interpretation

Suppose a GNN predicts:

```text
band gap = 2.31 eV
```

GNNExplainer identifies a local region containing:

```text
transition-metal atom
+
four oxygen neighbors
```

with high node and edge importance.

Pymatgen analysis shows:

```text
four nearest-neighbor distances
=
1.88 Å
1.91 Å
2.12 Å
2.18 Å
```

This suggests a distorted local coordination environment.

The correct scientific workflow is not:

```text
GNNExplainer
↓
Therefore distortion causes band gap
```

Instead:

```text
GNNExplainer
↓
Model highlights distorted environment
↓
Hypothesis:
local distortion may influence band gap
↓
Construct controlled structural variants
↓
DFT calculations
↓
Compare electronic structure
↓
Evaluate hypothesis
```

The explanation therefore becomes a **hypothesis-generation tool**.

---

# 26.7.23 GNNExplainer and Message Passing

The importance of GNNExplainer becomes clearer when considering message passing.

A typical GNN layer can be represented conceptually as:

```text
hᵢ^(l+1)
=
UPDATE(
    hᵢ^(l),
    AGGREGATE(
        hⱼ^(l),
        eᵢⱼ
    )
)
```

The representation of an atom therefore depends on neighboring atoms.

After several layers:

```text
Layer 1
↓
nearest neighbors

Layer 2
↓
two-hop environment

Layer 3
↓
larger structural environment
```

Consequently, a crystal-level prediction may depend on a distributed structural pattern.

GNNExplainer attempts to identify which graph components contribute most strongly to this prediction.

---

# 26.7.24 Receptive Field and Explanation Size

Suppose the GNN contains:

```text
3 message-passing layers
```

An atom can potentially receive information from approximately three-hop neighborhoods, depending on the architecture.

Therefore, an explanation containing several neighboring atoms may reflect the model's receptive field.

Conceptually:

```text
Central atom
     ↓
1-hop neighbors
     ↓
2-hop neighbors
     ↓
3-hop neighbors
```

The researcher should therefore avoid assuming that the explanation must correspond to a single chemical bond or a single coordination shell.

The model may be using a larger structural environment.

---

# 26.7.25 Comparing GNNExplainer with Saliency

The methods can now be compared.

| Method               | Main signal              | Typical materials interpretation          |
| -------------------- | ------------------------ | ----------------------------------------- |
| Saliency             | Local gradient           | Feature sensitivity                       |
| Integrated Gradients | Path-integrated gradient | Feature contribution relative to baseline |
| GNNExplainer         | Optimized masks          | Important nodes, edges, and subgraphs     |

The progression is:

```text
Saliency
   ↓
Which input is locally sensitive?

Integrated Gradients
   ↓
How did the input contribute relative to a baseline?

GNNExplainer
   ↓
Which graph structure best preserves the prediction?
```

This hierarchy is useful because no single explanation method answers every scientific question.

---

# 26.7.26 Explanation Agreement

For a high-confidence scientific interpretation, multiple explanation methods can be compared.

Suppose:

```text
Integrated Gradients
        ↓
transition-metal features important

Saliency
        ↓
transition-metal features important

GNNExplainer
        ↓
transition-metal-centered subgraph important
```

The agreement is encouraging.

The three methods are identifying related information from different perspectives.

However, disagreement is also informative.

For example:

```text
Integrated Gradients
        ↓
Feature A

GNNExplainer
        ↓
Feature B / structural motif
```

may indicate:

```text
representation dependence
feature correlation
model instability
explanation-method sensitivity
```

Such disagreements should be investigated rather than ignored.

---

# 26.7.27 Explanation Stability in GNNs

As with descriptor models, graph explanations can vary across:

```text
random seeds
model checkpoints
training splits
hyperparameters
graph construction settings
```

A robust study should therefore avoid presenting a single explanation as universally valid.

A stronger workflow is:

```text
Train multiple GNNs
       ↓
Explain the same crystal
       ↓
Compare node masks
       ↓
Compare edge masks
       ↓
Identify stable structural regions
```

For example:

```text
Model 1 → Ti-centered region
Model 2 → Ti-centered region
Model 3 → Ti-centered region
Model 4 → Ti-centered region
```

provides stronger evidence than one isolated explanation.

---

# 26.7.28 Reproducible GNN Explanation

A research implementation should record:

```text
Dataset version
Crystal preprocessing
Graph cutoff
Neighbor-generation method
Node features
Edge features
GNN architecture
Model weights
Random seed
PyTorch version
PyTorch Geometric version
GNNExplainer configuration
Optimization epochs
Mask thresholds
```

For example:

```python
import torch
import torch_geometric

print("PyTorch:", torch.__version__)
print(
    "PyG:",
    torch_geometric.__version__
)
```

The graph-construction parameters should also be stored.

For example:

```python
graph_config = {
    "cutoff": 5.0,
    "max_neighbors": 12,
    "periodic": True
}
```

This matters because changing the graph construction changes the explanation space itself.

---

# 26.7.29 Important Limitation: Explanation Is Model-Specific

Suppose:

```text
GNN A
```

and:

```text
GNN B
```

both predict band gap.

GNN A may use:

```text
local coordination
```

while GNN B uses:

```text
composition-derived information
```

even though both achieve similar test error.

Therefore:

```text
prediction agreement
≠
reasoning agreement
```

This is a crucial lesson in scientific machine learning.

Two models can make similar predictions while relying on different representations of the data.

XAI allows the researcher to investigate this difference.

---

# 26.7.30 Important Limitation: Mask Scores Are Not Physical Quantities

An edge mask such as:

```text
0.92
```

does not mean:

```text
92% bond strength
```

Similarly, a node mask:

```text
0.75
```

does not mean:

```text
atom contributes 75% of the physical property
```

These values are explanation-model quantities.

They should be interpreted comparatively:

```text
higher mask
→ more important within this explanation
```

rather than as direct physical measurements.

---

# 26.7.31 From Explanation to Scientific Hypothesis

A useful Materials Informatics explanation workflow is:

```text
Model prediction
      ↓
Explanation
      ↓
Important atoms / edges
      ↓
Chemical interpretation
      ↓
Structural analysis
      ↓
Physical hypothesis
      ↓
Independent validation
```

The final stage is essential.

Without independent validation, an explanation remains a model-based interpretation.

With validation, it can become part of a scientific argument.

For example:

```text
GNN
 ↓
high importance of distorted octahedron
 ↓
hypothesis:
distortion affects stability
 ↓
DFT calculations
 ↓
energy comparison
 ↓
validated structural trend
```

This is a much stronger research workflow than simply plotting an explanation figure.

---

## 26.7.32 Practical Research Checklist

Before publishing a GNN explanation, verify:

```text
[ ] The model has been independently validated.

[ ] The crystal graph construction is documented.

[ ] The meaning of graph edges is clearly defined.

[ ] Node and edge masks are interpreted comparatively.

[ ] The target output being explained is specified.

[ ] Explanation stability has been examined.

[ ] Important atoms are mapped back to actual species.

[ ] Important edges are mapped to structural relationships.

[ ] Local coordination environments are analyzed.

[ ] Physical interpretation is separated from model attribution.

[ ] Causal claims are not made from attribution alone.

[ ] Important hypotheses are independently tested.
```

This checklist prevents one of the most common mistakes in scientific XAI:

```text
visual explanation
        ↓
unverified physical conclusion
```

---

## 26.7.33 Transition to Feature-Level Attribution in Deep Models

GNNExplainer provides a structural explanation by identifying important graph components.

However, it does not completely answer another important question:

> Which specific input features inside those atoms or interactions are responsible?

Suppose GNNExplainer identifies:

```text
Ti-centered local environment
```

as important.

We may still want to know whether the model is responding primarily to:

```text
atomic number
electronegativity
valence information
atomic radius
coordination
distance
angular information
```

This requires returning from the graph level to the feature level.

The hierarchy of explanations can therefore be written as:

```text
Crystal
   ↓
Important subgraph
   ↓
Important atoms
   ↓
Important edges
   ↓
Important features
   ↓
Scientific interpretation
```

The combination of these levels is considerably more informative than using any one explanation technique in isolation.

The next section therefore develops **attention visualization in materials GNNs**, where learned attention weights can be examined to understand how information is selectively propagated through atomic neighborhoods.

# 26.9 Node Importance in Crystal Graphs

Attention visualization examines how a graph neural network distributes message-passing weights across neighboring atoms.

However, materials-science interpretation often requires a more direct question:

> Which atoms in the crystal are most important for the model's prediction?

This is the **node-importance** problem.

For a crystal graph,

```text
Atoms → Nodes
Interactions → Edges
Crystal → Graph
```

node importance attempts to quantify the contribution or sensitivity associated with individual atomic sites.

For example, consider a crystal containing:

```text
A₁  A₂  A₃  A₄  A₅
```

A model may produce a property prediction:

```text
ŷ = 2.41 eV
```

An explanation method may indicate:

```text
A₁ → low importance
A₂ → low importance
A₃ → high importance
A₄ → medium importance
A₅ → low importance
```

The important site may correspond to:

```text
transition-metal atom
defect site
dopant
under-coordinated atom
surface atom
chemically unusual environment
```

This makes node-level interpretation particularly valuable in Materials Informatics.

---

## 26.9.1 What Does Node Importance Mean?

Node importance is not a single universal quantity.

Depending on the explanation method, it may represent:

```text
gradient sensitivity
feature attribution
prediction change after perturbation
mask importance
attention-related importance
```

Therefore, the first question in any analysis should be:

> What mathematical definition of importance is being used?

For example, a gradient-based node importance can be defined as:

```text
Iᵢ = || ∂ŷ / ∂xᵢ ||
```

where:

```text
xᵢ
```

is the feature vector associated with atom `i`.

The interpretation is:

> How sensitive is the model output to small changes in the representation of node `i`?

This is a local sensitivity measure.

It is not automatically a statement about the physical importance of atom `i`.

---

## 26.9.2 Node Features in a Materials GNN

Before calculating node importance, we must understand what a node contains.

A crystal graph node may contain features such as:

```text
atomic number
periodic-table group
periodic-table period
electronegativity
atomic radius
formal charge
oxidation state
element embedding
```

For a node `i`, the feature vector may be:

```text
xᵢ =
[
Z,
χ,
r,
q,
...
]
```

where:

```text
Z → atomic number
χ → electronegativity
r → atomic radius
q → charge-related descriptor
```

Alternatively, the model may use a learned embedding:

```text
element
   ↓
embedding vector
   ↓
node representation
```

In a deeper GNN, the node representation becomes:

```text
hᵢ⁰
 ↓
GNN layer 1
 ↓
hᵢ¹
 ↓
GNN layer 2
 ↓
hᵢ²
 ↓
hᵢ³
```

Thus node importance can be calculated at different representation levels.

---

## 26.9.3 Input-Level Node Importance

The simplest case is importance with respect to the original node features.

Suppose:

```python
x.requires_grad_(True)
```

and the model produces:

```python
prediction = model(
    x,
    edge_index,
    batch
)
```

We can calculate gradients:

```python
prediction.backward()
```

and inspect:

```python
x.grad
```

A simple node-level score is the L2 norm:

```python
node_importance = (
    x.grad
    .norm(
        p=2,
        dim=1
    )
)
```

Complete example:

```python
import torch

x = x.clone().detach()
x.requires_grad_(True)

prediction = model(
    x,
    edge_index,
    batch
)

prediction.sum().backward()

node_importance = (
    x.grad
    .norm(
        p=2,
        dim=1
    )
)

print(node_importance)
```

The resulting vector contains one importance score per node.

---

## 26.9.4 Why the Gradient Norm Is Useful

Suppose:

```text
node_importance =
[
0.02,
0.04,
0.91,
0.13,
0.06
]
```

Then node 3 has substantially larger local sensitivity.

A normalized version is easier to visualize:

```python
node_importance = (
    node_importance /
    node_importance.max()
)
```

Now:

```text
node 1 → 0.02
node 2 → 0.04
node 3 → 1.00
node 4 → 0.14
node 5 → 0.07
```

This can be mapped onto the crystal structure.

But the interpretation must remain:

> The prediction is locally more sensitive to the representation of node 3.

It should not automatically become:

> Atom 3 physically determines the material property.

---

## 26.9.5 Signed Versus Absolute Importance

A gradient can be positive or negative.

For a feature `xᵢₖ`:

```text
∂ŷ / ∂xᵢₖ > 0
```

means increasing that feature locally tends to increase the prediction.

Conversely:

```text
∂ŷ / ∂xᵢₖ < 0
```

means increasing that feature locally tends to decrease the prediction.

If we calculate:

```python
importance = x.grad.norm(
    p=2,
    dim=1
)
```

the sign disappears.

This gives:

```text
magnitude of sensitivity
```

rather than direction.

For materials interpretation, both can be useful.

A researcher may want:

```text
How important is this atom?
```

or:

```text
Does changing its representation tend to increase or decrease the prediction?
```

These are different questions.

---

## 26.9.6 Feature-Wise Node Importance

Instead of combining all features into one score, we can inspect each feature separately.

Suppose:

```text
x.shape =
[N, F]
```

where:

```text
N → number of atoms
F → number of node features
```

Then:

```python
gradient = x.grad
```

has the same shape:

```text
[N, F]
```

We can calculate:

```python
feature_importance = gradient.abs()
```

For example:

```text
            Z      χ      radius      charge
Atom 1     0.03   0.01    0.02        0.01
Atom 2     0.04   0.08    0.03        0.02
Atom 3     0.72   0.31    0.12        0.04
Atom 4     0.05   0.02    0.07        0.01
```

This allows a much more detailed interpretation.

The researcher can ask:

```text
Which atom matters?
```

and:

```text
Which feature of that atom matters?
```

---

## 26.9.7 Aggregating Feature Importance into Node Importance

If the node contains multiple features, several aggregation strategies are possible.

### L1 aggregation

```python
node_importance_l1 = (
    gradient.abs()
    .sum(dim=1)
)
```

### L2 aggregation

```python
node_importance_l2 = (
    gradient.pow(2)
    .sum(dim=1)
    .sqrt()
)
```

### Maximum feature importance

```python
node_importance_max = (
    gradient.abs()
    .max(dim=1)
    .values
)
```

These methods answer slightly different questions.

L1 emphasizes total contribution across features.

L2 emphasizes overall vector magnitude.

Maximum emphasizes the strongest individual feature sensitivity.

A research workflow should state explicitly which aggregation was used.

---

## 26.9.8 Ranking Atomic Sites

Once node scores are calculated, the atoms can be ranked.

```python
ranking = torch.argsort(
    node_importance,
    descending=True
)

for rank, idx in enumerate(ranking):

    print(
        rank + 1,
        idx.item(),
        node_importance[idx].item()
    )
```

Example:

```text
Rank   Site   Importance
1      17     0.982
2      4      0.761
3      19     0.543
4      2      0.421
5      11     0.388
```

The indices can then be mapped back to Pymatgen sites.

```python
for idx in ranking[:5]:

    site = structure[idx.item()]

    print(
        idx.item(),
        site.species_string,
        site.frac_coords
    )
```

This converts an abstract neural-network result into a physical atomic location.

---

## 26.9.9 Connecting Node Importance to Chemical Identity

The next step is to ask whether important atoms share chemical characteristics.

For example:

```python
important_sites = []

for idx in ranking[:10]:

    site = structure[idx.item()]

    important_sites.append({
        "index": idx.item(),
        "element":
            site.species_string,
        "importance":
            node_importance[
                idx
            ].item()
    })
```

The resulting information may show:

```text
Element     Importance
Ti          0.98
Ti          0.91
O           0.74
O           0.71
Ba          0.31
```

This could suggest that the model strongly relies on the transition-metal sublattice.

But again, the result should be compared across many structures.

A single crystal is insufficient to establish a general materials-science relationship.

---

## 26.9.10 Node Importance Across a Dataset

Suppose we have:

```text
1000 crystal structures
```

For every crystal, we calculate node importance.

We can then aggregate by element.

For example:

```python
from collections import defaultdict
import numpy as np

element_scores = defaultdict(list)

for structure, scores in results:

    for i, site in enumerate(structure):

        element = site.species_string

        element_scores[element].append(
            scores[i]
        )
```

Calculate mean importance:

```python
mean_importance = {
    element: np.mean(scores)
    for element, scores
    in element_scores.items()
}
```

This allows analysis such as:

```text
Element    Mean node importance
Ti         0.72
Fe         0.68
O          0.41
S          0.36
Ba         0.19
```

Such results can reveal whether the model systematically relies on particular elemental environments.

However, the analysis must account for dataset composition.

If titanium appears much more frequently than another element, its aggregate score may be statistically easier to estimate.

---

## 26.9.11 Importance Is Not Frequency

A common mistake is to confuse:

```text
frequently occurring atom
```

with:

```text
important atom
```

Suppose oxygen appears in:

```text
90% of structures
```

while a dopant appears in:

```text
5% of structures
```

If the dopant consistently receives high importance, that may be more interesting than oxygen simply having many observations.

Therefore, report both:

```text
frequency
```

and:

```text
importance
```

when making dataset-level claims.

---

## 26.9.12 Node Importance and Defects

Node-level explanations become especially useful when defects are present.

Consider a vacancy:

```text
A — B — C
|   |   |
D —   — F
|       |
G — H — I
```

Suppose node `E` is missing.

The neighboring atoms may experience:

```text
coordination change
bond-length relaxation
symmetry breaking
charge redistribution
```

A node-importance map can show whether the model concentrates its sensitivity around the defect.

The workflow is:

```text
Defective crystal
      ↓
Crystal graph
      ↓
Node importance
      ↓
Map important atoms
      ↓
Compare with defect neighborhood
```

If the highest-importance atoms surround the vacancy, this provides an interesting model-behavior observation.

---

## 26.9.13 Defect-Centered Importance Analysis

We can explicitly calculate distance from a defect site.

Suppose:

```python
defect_index = 12
```

For each atom:

```python
distance = structure.get_distance(
    defect_index,
    i
)
```

Then construct:

```python
records.append({
    "site": i,
    "distance_from_defect":
        distance,
    "importance":
        node_importance[i].item()
})
```

Now we can test:

```text
importance
vs.
distance from defect
```

using:

```python
import pandas as pd

df = pd.DataFrame(records)

print(
    df[
        [
            "distance_from_defect",
            "importance"
        ]
    ].corr()
)
```

A systematic relationship can then be investigated across many defective structures.

---

## 26.9.14 Node Importance and Dopants

Dopants are another important application.

Consider:

```text
Host crystal
+
Dopant atom
```

The dopant may occupy a substitutional site:

```text
Host → Host → Host
             ↓
           Dopant
```

A node-importance map can determine whether the trained model assigns unusual importance to the dopant environment.

A useful comparison is:

```text
dopant importance
vs.
host-site importance
```

For example:

```python
dopant_score = node_importance[
    dopant_index
]

neighbor_scores = node_importance[
    neighboring_indices
]

print("Dopant:", dopant_score.item())
print(
    "Neighbor mean:",
    neighbor_scores.mean().item()
)
```

This provides a local contrast.

---

## 26.9.15 Importance Relative to the Local Environment

Absolute node importance can sometimes be misleading.

Instead, compare an atom with its local neighbors.

Define:

```text
relative importance =
node importance
/
mean neighbor importance
```

Python:

```python
neighbor_mean = (
    neighbor_scores.mean()
)

relative_score = (
    node_score /
    (neighbor_mean + 1e-12)
)
```

If:

```text
relative_score >> 1
```

the node is much more important than its local neighbors according to the chosen explanation method.

This is often more informative than comparing a raw score across unrelated crystals.

---

## 26.9.16 Node Importance and Coordination Environment

Two atoms of the same element may have completely different roles.

For example:

```text
O₁ → tetrahedral coordination
O₂ → octahedral coordination
O₃ → under-coordinated surface site
```

A node-level explanation can distinguish these sites.

The workflow becomes:

```text
Element
   +
Coordination
   +
Local geometry
   ↓
Node importance
```

This is particularly useful because Materials Informatics models often learn structure–property relationships that are not captured by elemental identity alone.

---

## 26.9.17 Using Pymatgen to Characterize Important Sites

Once important sites have been identified, Pymatgen can be used to inspect their local environments.

For example:

```python
from pymatgen.analysis.local_env import (
    CrystalNN
)

cnn = CrystalNN()

for idx in ranking[:10]:

    i = idx.item()

    neighbors = cnn.get_nn_info(
        structure,
        i
    )

    print(
        "Site:",
        i,
        structure[i].species_string
    )

    print(
        "Coordination:",
        len(neighbors)
    )
```

This converts:

```text
neural-network importance
```

into:

```text
local structural description
```

The researcher can then examine:

```text
coordination number
neighbor species
bond distances
local geometry
```

---

## 26.9.18 A Materials-Specific Explanation Table

For publication-quality analysis, construct a table such as:

```text
| Site | Element | Importance | Coordination | Mean Bond Length | Local Environment |
|------|---------|------------|--------------|------------------|-------------------|
| 17   | Ti      | 0.98       | 6            | 1.94 Å           | TiO₆              |
| 4    | O       | 0.76       | 3            | 1.91 Å           | distorted          |
| 19   | O       | 0.71       | 2            | 1.96 Å           | under-coordinated |
```

This is much more scientifically useful than reporting only:

```text
Node 17 has importance 0.98.
```

The goal is to connect:

```text
ML explanation
```

with:

```text
materials structure
```

---

## 26.9.19 Node Importance at Different GNN Depths

Node importance can change with message-passing depth.

At layer 1:

```text
hᵢ¹
```

primarily contains information from immediate neighbors.

At layer 2:

```text
hᵢ²
```

contains information from a larger graph neighborhood.

At layer 3:

```text
hᵢ³
```

the receptive field becomes larger again.

Therefore:

```text
Layer 1
→ local chemistry

Layer 2
→ local structural motif

Layer 3
→ larger structural environment
```

This creates a useful interpretability experiment.

Calculate node importance with respect to intermediate representations and compare the spatial patterns.

---

## 26.9.20 Retaining Node Gradients During Intermediate Analysis

Suppose the model contains:

```python
self.conv1
self.conv2
self.conv3
```

We can retain intermediate representations:

```python
x1 = self.conv1(
    x,
    edge_index
)

x1.retain_grad()

x2 = self.conv2(
    x1,
    edge_index
)

x2.retain_grad()

x3 = self.conv3(
    x2,
    edge_index
)
```

After calculating the prediction and calling:

```python
prediction.backward()
```

we can inspect:

```python
x1.grad
x2.grad
```

This provides sensitivity information at different levels of learned representation.

---

## 26.9.21 Node Importance for Classification

The same framework applies to classification.

Suppose a model predicts:

```text
stable
unstable
```

with:

```python
probabilities = torch.softmax(
    logits,
    dim=-1
)
```

If we want to explain the probability of class 1:

```python
target = probabilities[:, 1]

target.sum().backward()
```

Then:

```python
node_importance = (
    x.grad.abs()
    .sum(dim=1)
)
```

Now the explanation specifically answers:

> Which atoms are most associated with the model's prediction of class 1?

This is different from explaining the overall output.

---

## 26.9.22 Node Importance for Regression

For regression:

```text
band gap
formation energy
elastic modulus
thermal conductivity
```

we can directly explain the scalar output.

For example:

```python
prediction = model(
    x,
    edge_index,
    batch
)

target = prediction[0]

target.backward()
```

Then:

```python
importance = (
    x.grad.abs()
    .sum(dim=1)
)
```

This provides a node-level sensitivity map for that specific prediction.

---

## 26.9.23 Signed Node Contributions

Magnitude alone can hide direction.

A simple signed summary can be constructed by summing gradients:

```python
signed_importance = (
    x.grad.sum(dim=1)
)
```

For example:

```text
Atom 1 → +0.03
Atom 2 → -0.11
Atom 3 → +0.72
Atom 4 → -0.08
```

The interpretation becomes:

```text
positive
→ local feature changes tend to increase prediction

negative
→ local feature changes tend to decrease prediction
```

This interpretation is local and depends on the feature parameterization.

It should therefore be used carefully.

---

## 26.9.24 Normalizing Node Importance Across a Crystal

For visualization, normalization is often necessary.

A min-max normalization is:

```python
scores = node_importance

normalized = (
    scores - scores.min()
) / (
    scores.max()
    - scores.min()
    + 1e-12
)
```

The resulting range is:

```text
0 → least important
1 → most important
```

A z-score normalization can also be used:

```python
normalized = (
    scores - scores.mean()
) / (
    scores.std() + 1e-12
)
```

The choice should be reported because the visual appearance of the explanation depends on normalization.

---

## 26.9.25 Top-k Node Selection

For large structures, displaying every atom may be visually overwhelming.

A top-k selection can be used:

```python
k = 10

top_values, top_indices = torch.topk(
    node_importance,
    k=k
)
```

Then:

```python
for value, idx in zip(
    top_values,
    top_indices
):

    print(
        idx.item(),
        value.item()
    )
```

This provides the ten highest-ranked atomic sites.

However, top-k visualization should not replace the full importance distribution when performing quantitative analysis.

It is primarily a visualization strategy.

---

## 26.9.26 Threshold-Based Node Selection

An alternative is to select atoms above a threshold.

```python
threshold = 0.5

important_nodes = torch.where(
    normalized > threshold
)[0]
```

This may be useful when:

```text
different crystals have different numbers of important atoms.
```

Unlike top-k selection, thresholding can produce different numbers of highlighted sites.

---

## 26.9.27 Comparing Node Importance Between Materials

Suppose two polymorphs have:

```text
Structure A
Structure B
```

We can calculate:

```text
node-importance distribution A
node-importance distribution B
```

Then compare:

```python
mean_A = scores_A.mean()
mean_B = scores_B.mean()

max_A = scores_A.max()
max_B = scores_B.max()
```

More informative comparisons may include:

```text
entropy
top-k concentration
element-specific importance
coordination-specific importance
```

This can reveal whether the model uses similar or different structural reasoning for related materials.

---

## 26.9.28 Node Importance and Symmetry

Crystal symmetry provides another important validation opportunity.

Suppose two atoms are symmetry-equivalent.

If the model is expected to respect the symmetry of the representation, their importance scores should often be similar.

For example:

```text
Atom A
Atom B
```

may be related by a symmetry operation.

If:

```text
importance(A) ≈ importance(B)
```

the explanation is consistent with the symmetry.

If:

```text
importance(A) ≫ importance(B)
```

this deserves investigation.

Possible explanations include:

```text
graph-construction asymmetry
numerical noise
feature asymmetry
model behavior
incorrect periodic handling
```

This makes symmetry a powerful diagnostic tool for crystal GNN explanations.

---

## 26.9.29 Symmetry-Aware Validation

A practical workflow is:

```text
Crystal
 ↓
Identify symmetry-equivalent sites
 ↓
Calculate node importance
 ↓
Group equivalent sites
 ↓
Compare importance distributions
```

Pymatgen can provide symmetry information through its symmetry-analysis tools.

The exact implementation depends on the desired symmetry tolerance and structure-processing workflow.

The important methodological principle is:

> Explainability should respect known invariances and symmetries whenever the underlying model is designed to respect them.

---

## 26.9.30 Node Importance and Periodic Boundary Conditions

Crystal graphs differ from ordinary molecular graphs because periodicity creates interactions across unit-cell boundaries.

An important atom may have neighbors represented through periodic images.

Therefore, node-level explanations must distinguish:

```text
atom identity
```

from:

```text
periodic image of neighbor
```

For example:

```text
Atom i
 ↓
Neighbor j
translation vector T
```

The graph edge may correspond to:

```text
j + T
```

rather than simply the atom `j` in the same unit cell.

Ignoring this can produce incorrect structural interpretations.

---

## 26.9.31 Mapping Periodic Edges Correctly

A crystal graph should ideally retain periodic information such as:

```text
source site
target site
lattice translation
distance
```

For an edge record:

```python
edge_record = {
    "source": i,
    "target": j,
    "image": image_vector,
    "distance": distance,
    "importance": score
}
```

This allows the explanation to reconstruct the physical relationship correctly.

For materials research, this is especially important when the important interaction crosses the unit-cell boundary.

---

## 26.9.32 Node Importance and Surface Materials

For surfaces and slabs, node importance can identify surface-specific environments.

A slab may contain:

```text
surface atoms
subsurface atoms
bulk-like atoms
```

The researcher can calculate the distance from each atom to the surface.

Then compare:

```text
node importance
vs.
surface depth
```

For example:

```python
records.append({
    "site": i,
    "depth": depth[i],
    "importance":
        node_importance[i].item()
})
```

A systematic increase in importance near the surface may indicate that the model relies heavily on surface environments for a surface-sensitive target property.

---

## 26.9.33 Node Importance for Adsorption Problems

For adsorption-energy prediction:

```text
surface
+
adsorbate
```

node importance can be used to investigate whether the model focuses on:

```text
adsorbate atoms
surface active sites
neighboring surface atoms
```

A useful comparison is:

```text
adsorbate importance
vs.
surface-site importance
```

This can be especially valuable for catalytic materials.

For example:

```text
Prediction
   ↓
Node importance
   ↓
High importance around surface metal
   ↓
Inspect local coordination
   ↓
Compare with known catalytic active-site hypothesis
```

The model explanation can therefore generate a hypothesis for subsequent computational or experimental investigation.

---

## 26.9.34 Node Importance and Electronic Properties

For electronic properties such as:

```text
band gap
formation energy
magnetic moment
```

important atoms may correspond to chemically distinctive environments.

However, a node-level GNN explanation alone cannot establish orbital-level mechanisms.

For example:

```text
high importance on transition-metal atoms
```

does not prove:

```text
d-orbitals control the band gap.
```

To make such a claim, additional information may be needed:

```text
DFT density of states
projected DOS
charge density
orbital analysis
bonding analysis
```

This is a critical distinction between:

```text
model interpretation
```

and:

```text
physical mechanism.
```

---

## 26.9.35 From Node Importance to Scientific Hypothesis

A robust workflow therefore looks like:

```text
GNN prediction
      ↓
Node importance
      ↓
Important atomic sites
      ↓
Local structural analysis
      ↓
Chemical interpretation
      ↓
Independent computational analysis
      ↓
Scientific hypothesis
```

For example:

```text
Node importance
      ↓
Transition-metal site
      ↓
Distorted octahedral environment
      ↓
DFT projected DOS
      ↓
Electronic-state change
      ↓
Testable hypothesis
```

This is where explainable Materials Informatics becomes scientifically useful.

---

## 26.9.36 Comparing Multiple Node-Importance Methods

Node importance should ideally not depend on one explanation method.

For the same crystal, calculate:

```text
Gradient importance
Integrated Gradients
GNNExplainer node mask
Attention-derived importance
```

Then compare rankings.

For example:

```text
             Site 17   Site 4   Site 19
Gradient       1         3         2
IG             1         2         3
GNNExplainer   1         2         3
Attention      2         1         3
```

If several methods identify the same sites, confidence in the model-behavior interpretation increases.

---

## 26.9.37 Rank Correlation Between Explanations

A quantitative comparison can use Spearman rank correlation.

```python
from scipy.stats import spearmanr

rho, p = spearmanr(
    gradient_scores,
    integrated_gradient_scores
)

print(
    "Rank correlation:",
    rho
)

print(
    "p-value:",
    p
)
```

A high correlation suggests similar ranking behavior.

A low correlation indicates that the explanation methods may be emphasizing different aspects of the model.

This disagreement is not necessarily a failure.

It may indicate that the methods answer different questions.

---

## 26.9.38 Node Importance Stability Across Random Seeds

As with attention, explanation stability should be tested.

Train:

```text
5 independent models
```

Then calculate node importance for the same material.

The results can be compared using:

```text
Spearman correlation
top-k overlap
Jaccard similarity
rank consistency
```

For top-k sets:

```python
def jaccard(set_a, set_b):

    intersection = len(
        set_a & set_b
    )

    union = len(
        set_a | set_b
    )

    return intersection / union
```

For example:

```python
score = jaccard(
    top_nodes_seed1,
    top_nodes_seed2
)

print(
    "Top-k Jaccard:",
    score
)
```

High overlap across seeds makes the explanation more credible.

---

## 26.9.39 A Complete Node-Importance Function

A reusable implementation is:

```python
def compute_node_importance(
    model,
    x,
    edge_index,
    batch,
    target_index=None
):

    x = x.clone().detach()
    x.requires_grad_(True)

    model.zero_grad()

    output = model(
        x,
        edge_index,
        batch
    )

    if target_index is None:

        target = output.sum()

    else:

        target = output[
            target_index
        ]

    target.backward()

    gradients = x.grad

    importance = gradients.norm(
        p=2,
        dim=1
    )

    return (
        importance.detach(),
        gradients.detach()
    )
```

Usage:

```python
importance, gradients = (
    compute_node_importance(
        model,
        x,
        edge_index,
        batch
    )
)
```

This returns both:

```text
node importance
```

and:

```text
raw gradients
```

so that magnitude-based and signed analyses can both be performed.

---

## 26.9.40 Building a Crystal Explanation Dataset

For large-scale research, explanations themselves can become a dataset.

Each record may contain:

```text
material ID
site index
element
fractional coordinates
cartesian coordinates
importance
gradient vector
coordination number
neighbor species
mean bond length
```

For example:

```python
records = []

for i, site in enumerate(structure):

    records.append({
        "material_id": material_id,
        "site_index": i,
        "element":
            site.species_string,
        "importance":
            importance[i].item(),
        "x":
            site.coords[0],
        "y":
            site.coords[1],
        "z":
            site.coords[2]
    })
```

Then:

```python
import pandas as pd

explanation_df = pd.DataFrame(
    records
)
```

This allows downstream statistical analysis.

---

## 26.9.41 Dataset-Level Scientific Questions

Once explanations are collected across many materials, much deeper questions become possible:

```text
Which elements are consistently important?

Which coordination environments are important?

Are defect sites systematically emphasized?

Does importance correlate with bond length?

Does importance correlate with oxidation state?

Do important sites differ between material classes?

Are important sites conserved across polymorphs?

Does the model focus on chemically reasonable environments?
```

These questions transform explainability from a visualization technique into a quantitative research methodology.

---

## 26.9.42 Node Importance Does Not Replace Model Validation

A model can produce scientifically attractive explanations while making poor predictions.

Therefore, the correct order is:

```text
Predictive validation
        ↓
Explanation
        ↓
Scientific interpretation
```

not:

```text
Interesting explanation
        ↓
Assume model is correct
```

The model should first demonstrate acceptable performance using appropriate validation procedures.

Only then should the explanations be interpreted as descriptions of model behavior.

---

## 26.9.43 Practical Research Checklist

Before reporting node-importance results, verify:

```text
[ ] Model performance was independently validated

[ ] Node representation is clearly documented

[ ] Explanation method is mathematically defined

[ ] Gradient or attribution target is specified

[ ] Importance normalization is documented

[ ] Crystal periodicity is handled correctly

[ ] Symmetry has been considered

[ ] Multiple random seeds were tested

[ ] Important sites were mapped to physical atoms

[ ] Local coordination environments were analyzed

[ ] Results were compared with another XAI method

[ ] Physical claims were independently validated
```

This checklist prevents a common problem in explainable Materials Informatics:

> converting a model-dependent visualization directly into a physical conclusion.

---

## 26.9.44 Summary of Node Importance

Node importance provides a bridge between a graph neural network and the atomic structure on which it operates.

The basic workflow is:

```text
Crystal
   ↓
Crystal graph
   ↓
GNN
   ↓
Prediction
   ↓
Node-level attribution
   ↓
Important atoms
   ↓
Local structural analysis
   ↓
Scientific interpretation
```

The most important principles are:

1. **Node importance is method-dependent.**

2. **Gradient magnitude measures local sensitivity, not causality.**

3. **Feature-wise attribution can reveal which atomic descriptors matter.**

4. **Important nodes should be mapped back to physical crystal sites.**

5. **Pymatgen can be used to characterize their local environments.**

6. **Defects, dopants, surfaces, and adsorption sites are particularly useful applications.**

7. **Periodic boundary conditions must be handled correctly.**

8. **Crystal symmetry provides an important validation constraint.**

9. **Importance should be evaluated across multiple structures rather than one example.**

10. **Stability across random seeds should be investigated.**

11. **Agreement between multiple XAI methods provides stronger evidence than a single explanation.**

12. **Node importance should generate scientific hypotheses rather than automatically establish physical mechanisms.**

The next level of interpretation is to move from **which atoms matter** to **which interactions between atoms matter**.

A crystal property is rarely determined by isolated atoms.

Chemical bonding, coordination, local geometry, strain transmission, electrostatic interactions, and structural connectivity are inherently relational.

Therefore, the next section focuses on **edge importance**, where the explanation shifts from individual atomic sites to the interactions connecting them.

# 26.10 Edge Importance

Node importance asks:

> Which atoms are most important to the model's prediction?

For crystal graph neural networks, this is only half of the interpretability problem.

A crystal property is not determined only by the identity of individual atoms. It also depends on the **relationships between atoms**:

```text
Atomic identity
      +
Bonding
      +
Interatomic distance
      +
Coordination
      +
Local geometry
      +
Periodic connectivity
      ↓
Materials property
```

A graph neural network explicitly represents these relationships through **edges**.

Therefore, an important interpretability question is:

> Which atomic interactions are most important for the model's prediction?

This is the purpose of **edge importance**.

For a crystal graph:

```text
G = (V, E)
```

where:

```text
V → atoms
E → atomic interactions
```

node explanation investigates:

```text
importance(vᵢ)
```

while edge explanation investigates:

```text
importance(eᵢⱼ)
```

An edge may represent:

```text
neighbor relationship
bond-like interaction
periodic interaction
distance-based connection
message-passing relationship
```

depending on how the crystal graph was constructed.

This distinction is particularly important in Materials Informatics because many scientifically meaningful mechanisms are relational rather than purely atomic.

---

## 26.10.1 Why Edge Importance Matters

Consider two crystals containing exactly the same elements:

```text
Crystal A
A — B — A
```

and:

```text
Crystal B
A   B   A
 \     /
  -------
```

Their chemical compositions may be identical, but their structural connectivity can differ.

A model predicting:

```text
band gap
formation energy
elastic modulus
magnetic property
thermal property
```

may therefore rely strongly on particular interactions.

For example:

```text
A — B
```

may be highly important while:

```text
A — A
```

may contribute relatively little.

An edge-importance map could reveal:

```text
A₁ ── B₁     high
│
A₂ ── B₂     low
```

This provides information about **which local interactions the model considers predictive**.

---

## 26.10.2 Edge Representation in a Crystal GNN

A crystal graph commonly contains an edge feature vector:

```text
eᵢⱼ
```

which may include:

```text
interatomic distance
radial basis expansion
periodic image information
bond direction
geometric descriptors
```

For example:

```python
edge_attr = torch.tensor([
    distance,
    radial_feature_1,
    radial_feature_2,
    radial_feature_3
])
```

A simplified edge representation might therefore be:

```text
eᵢⱼ =
[
dᵢⱼ,
g₁,
g₂,
g₃,
...
]
```

where:

```text
dᵢⱼ → interatomic distance
gₖ → geometric representation
```

The GNN uses this information during message passing.

---

## 26.10.3 Message Passing and Edge Importance

A generic message-passing layer can be written as:

```text
mᵢⱼ = φ(hᵢ, hⱼ, eᵢⱼ)
```

and node updates as:

```text
hᵢ' = ψ(hᵢ, Σⱼ mᵢⱼ)
```

The edge therefore determines how information travels from node `j` to node `i`.

Conceptually:

```text
Atom j
   ↓
edge eᵢⱼ
   ↓
message
   ↓
Atom i
```

If the model's prediction is highly sensitive to a particular edge representation, that edge may receive high attribution.

Thus edge importance provides a way to inspect the **information pathways** used by the GNN.

---

## 26.10.4 Gradient-Based Edge Importance

The simplest edge-level explanation is gradient sensitivity.

Suppose:

```python
edge_attr.requires_grad_(True)
```

and:

```python
prediction = model(
    x,
    edge_index,
    edge_attr,
    batch
)
```

Then:

```python
prediction.sum().backward()
```

produces:

```python
edge_attr.grad
```

A scalar edge score can be calculated as:

```python
edge_importance = (
    edge_attr.grad
    .norm(
        p=2,
        dim=1
    )
)
```

Complete example:

```python
import torch

edge_attr = (
    edge_attr
    .clone()
    .detach()
)

edge_attr.requires_grad_(True)

model.zero_grad()

prediction = model(
    x,
    edge_index,
    edge_attr,
    batch
)

prediction.sum().backward()

edge_importance = (
    edge_attr.grad
    .norm(
        p=2,
        dim=1
    )
)

print(edge_importance)
```

The result contains one score for every graph edge.

---

## 26.10.5 Interpreting Gradient-Based Edge Importance

Suppose:

```text
edge_importance =
[
0.02,
0.11,
0.83,
0.07,
0.31
]
```

The third edge has the largest local sensitivity.

This means:

> The model output is locally more sensitive to the representation of the third edge.

It does **not** automatically mean:

> The third edge is the physically strongest bond in the material.

These are fundamentally different statements.

The first is about:

```text
model sensitivity
```

while the second is about:

```text
physical bonding strength
```

The distinction must remain explicit throughout XAI analysis.

---

## 26.10.6 Edge Features Versus Edge Identity

There are two different questions we may want to answer.

### Question 1

Which edge is important?

```text
e₁₂
e₂₃
e₃₄
...
```

### Question 2

Which feature of the edge is important?

For example:

```text
distance
radial basis feature
directional descriptor
```

These should not be confused.

Suppose:

```text
edge_attr.shape = [E, F]
```

Then:

```python
edge_grad = edge_attr.grad
```

has shape:

```text
[E, F]
```

We can calculate feature-level importance:

```python
feature_importance = (
    edge_grad.abs()
)
```

This gives:

```text
             distance   RBF1   RBF2
Edge 1          0.02    0.03   0.01
Edge 2          0.11    0.08   0.04
Edge 3          0.74    0.31   0.12
```

The researcher can then investigate whether the model's sensitivity is primarily associated with distance or another edge descriptor.

---

## 26.10.7 Aggregating Edge Features

As with node importance, edge-level features can be aggregated.

### L1 score

```python
edge_importance_l1 = (
    edge_grad.abs()
    .sum(dim=1)
)
```

### L2 score

```python
edge_importance_l2 = (
    edge_grad.pow(2)
    .sum(dim=1)
    .sqrt()
)
```

### Maximum feature sensitivity

```python
edge_importance_max = (
    edge_grad.abs()
    .max(dim=1)
    .values
)
```

A research paper should clearly state which aggregation method was used.

---

## 26.10.8 Mapping Edges Back to Atomic Sites

A raw edge index such as:

```text
edge 37
```

has little scientific meaning.

We need to convert it into:

```text
source atom
target atom
elements
distance
periodic image
```

Suppose:

```python
src = edge_index[0]
dst = edge_index[1]
```

Then:

```python
for edge_id in range(
    edge_index.shape[1]
):

    i = src[edge_id].item()
    j = dst[edge_id].item()

    print(
        edge_id,
        i,
        j,
        structure[i].species_string,
        structure[j].species_string
    )
```

This converts the graph representation into a chemical interpretation.

---

## 26.10.9 Including Interatomic Distance

For crystal interpretation, distance is especially important.

Pymatgen can calculate distances between sites:

```python
distance = structure.get_distance(
    i,
    j
)
```

A useful edge record is therefore:

```python
edge_record = {
    "edge_id": edge_id,
    "source": i,
    "target": j,
    "source_element":
        structure[i].species_string,
    "target_element":
        structure[j].species_string,
    "distance": distance,
    "importance":
        edge_importance[
            edge_id
        ].item()
}
```

This produces a physically meaningful explanation table.

---

## 26.10.10 Edge Explanation Table

A publication-quality table might look like:

```text
| Edge | Site i | Site j | Pair | Distance | Importance |
|------|--------|--------|------|----------|------------|
| 17   | 4      | 11     | Ti–O  | 1.92 Å  | 0.91       |
| 23   | 7      | 12     | Fe–O  | 1.97 Å  | 0.73       |
| 31   | 2      | 9      | O–O   | 2.81 Å  | 0.22       |
```

Now the explanation can be interpreted chemically.

For example:

```text
The model strongly relies on Ti–O interactions
near 1.9 Å.
```

That is substantially more informative than:

```text
Edge 17 has importance 0.91.
```

---

## 26.10.11 Edge Importance and Bond Length

One natural scientific analysis is:

```text
edge importance
vs.
interatomic distance
```

Create a dataset:

```python
records = []

for edge_id in range(
    edge_index.shape[1]
):

    i = edge_index[
        0, edge_id
    ].item()

    j = edge_index[
        1, edge_id
    ].item()

    distance = structure.get_distance(
        i,
        j
    )

    records.append({
        "distance": distance,
        "importance":
            edge_importance[
                edge_id
            ].item()
    })
```

Convert to a DataFrame:

```python
import pandas as pd

edge_df = pd.DataFrame(
    records
)

print(edge_df.head())
```

We can then examine the relationship statistically.

```python
correlation = edge_df[
    [
        "distance",
        "importance"
    ]
].corr()

print(correlation)
```

A strong correlation may be interesting, but it should not automatically be interpreted as a physical law.

The model may be using distance because distance correlates with other structural information.

---

## 26.10.12 Edge Importance and Element Pair

A more chemically interpretable analysis groups edges by element pair.

For example:

```text
Ti–O
Fe–O
Ba–O
O–O
```

Create a canonical pair:

```python
def element_pair(
    element_a,
    element_b
):

    return "-".join(
        sorted(
            [
                element_a,
                element_b
            ]
        )
    )
```

Then:

```python
for record in records:

    pair = element_pair(
        record["source_element"],
        record["target_element"]
    )

    record["pair"] = pair
```

Now group:

```python
pair_importance = (
    edge_df
    .groupby("pair")[
        "importance"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)
```

The output may be:

```text
Ti–O    0.72
Fe–O    0.63
Ba–O    0.38
O–O     0.21
```

This can reveal which interaction classes the model relies on most strongly.

---

## 26.10.13 Important Edge Versus Strong Physical Bond

This distinction deserves special emphasis.

Suppose the model identifies:

```text
Ti–O
```

as highly important.

That does not necessarily mean Ti–O is the strongest chemical bond.

The model may rely on Ti–O because:

```text
Ti–O distance
```

correlates strongly with:

```text
octahedral distortion
electronic structure
coordination
phase stability
```

Therefore:

```text
XAI result
```

should be interpreted as:

> The Ti–O interaction is highly informative for the model.

Physical bonding strength requires independent evidence.

Possible validation tools include:

```text
bond-order analysis
charge density
COHP/ICOHP
projected DOS
electron localization function
DFT calculations
```

This is a critical boundary between explainability and physical interpretation.

---

## 26.10.14 Directed Versus Undirected Crystal Edges

Many GNN implementations represent an undirected physical interaction using two directed graph edges:

```text
i → j
j → i
```

Therefore:

```text
edge 1 = i → j
edge 2 = j → i
```

may correspond to the same physical neighbor relationship.

If we simply rank graph edges, we may accidentally count one physical interaction twice.

This is particularly important in crystal GNNs.

---

## 26.10.15 Combining Reverse Edges

A physical edge score can be obtained by combining both directions.

For example:

```python
physical_score = (
    score_ij + score_ji
) / 2
```

Alternatively:

```python
physical_score = max(
    score_ij,
    score_ji
)
```

or:

```python
physical_score = (
    abs(score_ij)
    +
    abs(score_ji)
)
```

The choice depends on the explanation objective.

Averaging is appropriate when both directions should represent the same interaction.

Maximum emphasizes the strongest directional contribution.

The methodology should always be documented.

---

## 26.10.16 Finding Reverse Edges

A simple implementation can create an edge dictionary.

```python
edge_lookup = {}

for edge_id in range(
    edge_index.shape[1]
):

    i = edge_index[
        0, edge_id
    ].item()

    j = edge_index[
        1, edge_id
    ].item()

    edge_lookup[
        (i, j)
    ] = edge_id
```

Then:

```python
physical_edges = []

visited = set()

for (i, j), edge_id in edge_lookup.items():

    if (i, j) in visited:
        continue

    reverse_id = edge_lookup.get(
        (j, i)
    )

    if reverse_id is not None:

        score = (
            edge_importance[edge_id]
            +
            edge_importance[reverse_id]
        ) / 2

        physical_edges.append({
            "i": i,
            "j": j,
            "importance":
                score.item()
        })

        visited.add((i, j))
        visited.add((j, i))
```

This prevents duplicate reporting of the same interaction.

---

## 26.10.17 Edge Importance and Periodic Images

Crystal graphs introduce an additional complication.

Two graph edges can connect the same pair of crystallographic sites but through different periodic images.

For example:

```text
i → j + T₁
i → j + T₂
```

where:

```text
T₁ ≠ T₂
```

These may represent different spatial interactions.

Therefore, a correct edge record should preserve:

```text
source site
target site
lattice translation
distance
edge importance
```

For example:

```python
edge_record = {
    "source": i,
    "target": j,
    "translation": image_vector.tolist(),
    "distance": distance,
    "importance": score
}
```

This distinction becomes essential for:

```text
extended solids
layered materials
framework materials
low-symmetry structures
long-range neighbor graphs
```

---

## 26.10.18 Edge Importance and Coordination

Edge importance can reveal which neighbor interactions are particularly informative within a coordination environment.

Consider:

```text
        O₁
         |
O₂ — Ti — O₃
         |
        O₄
```

If all Ti–O edges have similar importance:

```text
Ti–O₁ → 0.72
Ti–O₂ → 0.69
Ti–O₃ → 0.74
Ti–O₄ → 0.71
```

the model may be using the overall TiO₄ environment.

If one edge is much more important:

```text
Ti–O₁ → 0.91
Ti–O₂ → 0.42
Ti–O₃ → 0.39
Ti–O₄ → 0.44
```

the model may be sensitive to a local distortion or asymmetric environment.

This should then be investigated using the actual geometry.

---

## 26.10.19 Edge Importance and Distorted Coordination

Suppose:

```text
Ti–O₁ = 1.82 Å
Ti–O₂ = 1.93 Å
Ti–O₃ = 1.94 Å
Ti–O₄ = 2.08 Å
```

and the explanation gives:

```text
importance:
0.91
0.44
0.39
0.78
```

The model appears sensitive to the unusually short and long bonds.

This could suggest that:

```text
local distortion
```

is important to the prediction.

The correct next step is not to declare a mechanism.

Instead:

```text
XAI
 ↓
Identify unusual edge
 ↓
Inspect bond geometry
 ↓
Compare with other structures
 ↓
Perform independent physical analysis
```

This transforms explanation into hypothesis generation.

---

## 26.10.20 Edge Importance in Defective Materials

Defects often change interactions more directly than they change elemental identities.

Consider a vacancy:

```text
A — B — C
|       |
D       E
```

Removing one atom changes several edges:

```text
A — X
B — X
C — X
D — X
E — X
```

The edges disappear.

Other neighboring edges may become structurally modified after relaxation.

An edge-importance analysis can therefore reveal which interactions surrounding the defect are most relevant to the model.

Workflow:

```text
Defect
 ↓
Relaxed structure
 ↓
Crystal graph
 ↓
Edge importance
 ↓
Defect-neighbor interactions
 ↓
Physical interpretation
```

This is particularly useful for:

```text
vacancy formation energy
defect stability
migration barriers
electronic defect properties
```

---

## 26.10.21 Edge Importance for Dopant Effects

A dopant can affect not only itself but also its surrounding interactions.

Consider:

```text
Host — Host
  \     /
   Dopant
  /     \
Host — Host
```

The model may assign high importance to:

```text
Dopant–Host
```

edges rather than to the dopant node alone.

This distinction is scientifically useful.

It suggests that the predictive information may be associated with:

```text
dopant
+
local chemical environment
```

rather than simply:

```text
presence of dopant
```

---

## 26.10.22 Edge Importance in Adsorption Models

For catalytic systems, edge explanations can be particularly informative.

Consider:

```text
Surface metal
      |
Adsorbate atom
```

The relevant model behavior may depend on the interaction between:

```text
adsorbate
```

and:

```text
surface active site
```

rather than either node independently.

An edge-importance analysis can therefore identify:

```text
M–O
M–C
M–N
M–H
```

interactions that the model considers predictive.

This can be compared against known catalytic descriptors and independent electronic-structure calculations.

---

## 26.10.23 Edge Importance for Magnetic Materials

For magnetic materials, interactions between neighboring magnetic atoms can be particularly important.

A model may predict:

```text
magnetic moment
exchange interaction
magnetic ordering
```

from local graph structure.

If edge explanations identify:

```text
Fe–O–Fe
Mn–O–Mn
Co–O
```

environments, these interactions may provide useful hypotheses about the learned structural patterns.

However, an edge explanation still does not directly provide an exchange coupling constant.

Independent calculations are required.

---

## 26.10.24 Edge Importance and Long-Range Interactions

A crystal graph often uses a cutoff radius:

```text
r < r_cut
```

Only neighbors within this radius become edges.

For example:

```python
cutoff = 5.0  # Å
```

Changing the cutoff changes:

```text
number of edges
graph connectivity
receptive field
possible explanations
```

Therefore, edge-importance analysis should document the graph construction parameters.

A model using:

```text
r_cut = 4 Å
```

and a model using:

```text
r_cut = 8 Å
```

may produce very different edge explanations.

---

## 26.10.25 Edge Importance and Cutoff Sensitivity

A useful robustness experiment is:

```text
Train model A
cutoff = 4 Å

Train model B
cutoff = 5 Å

Train model C
cutoff = 6 Å
```

Then compare the important interaction classes.

For example:

```text
             4 Å    5 Å    6 Å
Ti–O         high   high   high
O–O          low    low    medium
Ti–Ti        low    medium medium
```

If important interactions remain stable, the explanation becomes more convincing.

If the explanation changes dramatically, the graph representation itself may be influencing the interpretation.

---

## 26.10.26 Edge Importance and Message-Passing Layers

As with node importance, edge relevance can change with GNN depth.

At an early layer:

```text
edge
 ↓
local message
```

At deeper layers:

```text
edge
 ↓
local message
 ↓
updated node representation
 ↓
larger structural context
```

Thus an edge can be important because of:

```text
direct local interaction
```

or because it contributes to:

```text
larger structural information propagation.
```

This is one reason why explanation methods should be interpreted within the architecture being analyzed.

---

## 26.10.27 Edge Masks

Many graph explanation algorithms directly learn an edge mask.

A mask may take values:

```text
0 → irrelevant
1 → highly important
```

or continuous values:

```text
0.00
0.12
0.43
0.87
1.00
```

The model can then be evaluated using the masked graph:

```text
G
 ↓
Edge mask
 ↓
Important subgraph
 ↓
Prediction
```

The objective is often to identify a compact subgraph that preserves the prediction.

This idea becomes central to **GNNExplainer**, which will be discussed in detail later in the chapter.

---

## 26.10.28 Gradient Edge Importance Versus Learned Edge Masks

These are conceptually different.

### Gradient-based method

Asks:

> How sensitive is the prediction to this edge representation?

### Learned edge mask

Asks:

> Which subset of edges is sufficient or especially useful for reproducing the prediction?

Therefore, they should not be treated as identical.

A useful research strategy is to compare them.

```text
Gradient
   ↓
Edge ranking

GNNExplainer
   ↓
Edge mask

       ↓

Compare important interactions
```

---

## 26.10.29 Edge Importance Visualization

A crystal graph can be visualized with:

```text
low importance
      ↓
thin/faint edge

high importance
      ↓
thick/prominent edge
```

Conceptually:

```text
       O
       ║
       ║  high
O ─────Ti──── O
       |
       |
       O
```

The thickness or opacity can represent the importance score.

A proper scientific visualization should include a legend explaining:

```text
edge width
→ importance score
```

and should not imply that visualization thickness represents physical bond strength unless it actually does.

---

## 26.10.30 Building an Edge Explanation DataFrame

A reusable function can collect the required information.

```python
def build_edge_explanation_table(
    structure,
    edge_index,
    edge_importance
):

    records = []

    for edge_id in range(
        edge_index.shape[1]
    ):

        i = edge_index[
            0, edge_id
        ].item()

        j = edge_index[
            1, edge_id
        ].item()

        distance = structure.get_distance(
            i,
            j
        )

        records.append({

            "edge_id":
                edge_id,

            "source":
                i,

            "target":
                j,

            "source_element":
                structure[
                    i
                ].species_string,

            "target_element":
                structure[
                    j
                ].species_string,

            "distance":
                distance,

            "importance":
                edge_importance[
                    edge_id
                ].item()
        })

    return pd.DataFrame(
        records
    )
```

Usage:

```python
edge_df = (
    build_edge_explanation_table(
        structure,
        edge_index,
        edge_importance
    )
)

print(edge_df.head())
```

This provides a reusable foundation for downstream analysis.

---

## 26.10.31 Ranking the Most Important Interactions

The most important edges can be extracted using:

```python
top_edges = (
    edge_df
    .sort_values(
        "importance",
        ascending=False
    )
    .head(20)
)

print(top_edges)
```

This produces a ranked interaction list.

For example:

```text
Edge    Pair     Distance    Importance
17      Ti–O      1.91 Å      0.94
23      Ti–O      2.03 Å      0.88
31      Fe–O      1.87 Å      0.79
42      O–H       1.02 Å      0.74
```

This is a practical way to identify the local interactions that deserve further scientific investigation.

---

## 26.10.32 Aggregating by Interaction Type

For dataset-level interpretation:

```python
edge_df["pair"] = (
    edge_df.apply(
        lambda row:
        element_pair(
            row[
                "source_element"
            ],
            row[
                "target_element"
            ]
        ),
        axis=1
    )
)
```

Then:

```python
pair_summary = (
    edge_df
    .groupby("pair")
    .agg(
        mean_importance=(
            "importance",
            "mean"
        ),
        max_importance=(
            "importance",
            "max"
        ),
        count=(
            "importance",
            "count"
        )
    )
    .sort_values(
        "mean_importance",
        ascending=False
    )
)
```

Now the analysis considers:

```text
average importance
maximum importance
number of observations
```

rather than relying on one example.

---

## 26.10.33 Avoiding a Statistical Trap

Suppose:

```text
Ti–O → 5000 edges
Cu–O  → 50 edges
```

and:

```text
mean importance:
Ti–O → 0.51
Cu–O → 0.68
```

The Cu–O result appears more important, but it is based on far fewer observations.

Therefore, report:

```text
mean
median
standard deviation
sample count
```

where appropriate.

For example:

```python
summary = (
    edge_df
    .groupby("pair")[
        "importance"
    ]
    .agg([
        "mean",
        "median",
        "std",
        "count"
    ])
    .sort_values(
        "mean",
        ascending=False
    )
)
```

This produces a much more defensible dataset-level interpretation.

---

## 26.10.34 Edge Importance and Scientific Validation

The strongest workflow is:

```text
GNN
 ↓
Edge explanation
 ↓
Important interaction
 ↓
Structural characterization
 ↓
Independent physical calculation
 ↓
Scientific interpretation
```

For example:

```text
GNN
 ↓
High Ti–O edge importance
 ↓
Ti–O distortion identified
 ↓
DFT electronic analysis
 ↓
DOS changes near Fermi level
 ↓
Hypothesis about learned structure–property relationship
```

The XAI method identifies where to look.

The independent scientific calculation tests whether the proposed interpretation is physically meaningful.

---

## 26.10.35 A Complete Gradient-Based Edge Workflow

A compact end-to-end implementation is:

```python
import torch
import pandas as pd


def explain_edges(
    model,
    structure,
    x,
    edge_index,
    edge_attr,
    batch
):

    x = (
        x.clone()
        .detach()
    )

    edge_attr = (
        edge_attr
        .clone()
        .detach()
    )

    edge_attr.requires_grad_(True)

    model.zero_grad()

    prediction = model(
        x,
        edge_index,
        edge_attr,
        batch
    )

    prediction.sum().backward()

    gradients = (
        edge_attr.grad
        .detach()
    )

    importance = (
        gradients
        .norm(
            p=2,
            dim=1
        )
    )

    records = []

    for edge_id in range(
        edge_index.shape[1]
    ):

        i = edge_index[
            0, edge_id
        ].item()

        j = edge_index[
            1, edge_id
        ].item()

        distance = (
            structure.get_distance(
                i,
                j
            )
        )

        records.append({

            "edge_id":
                edge_id,

            "source":
                i,

            "target":
                j,

            "source_element":
                structure[
                    i
                ].species_string,

            "target_element":
                structure[
                    j
                ].species_string,

            "distance":
                distance,

            "importance":
                importance[
                    edge_id
                ].item()
        })

    return (
        pd.DataFrame(
            records
        ),
        prediction.detach()
    )
```

Usage:

```python
edge_df, prediction = (
    explain_edges(
        model=model,
        structure=structure,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch
    )
)
```

Then:

```python
print(
    edge_df
    .sort_values(
        "importance",
        ascending=False
    )
    .head(10)
)
```

This gives a complete path from:

```text
model prediction
```

to:

```text
ranked physical interactions.
```

---

## 26.10.36 Important Limitation: Edge Definition Is Model-Dependent

One of the most important limitations is that an edge is not necessarily a physical chemical bond.

In a crystal GNN, edges are often generated using a geometric cutoff:

```text
distance < cutoff
```

Therefore, an edge may simply mean:

> These two atoms are close enough to exchange messages in the graph.

It may not mean:

> These two atoms form a chemically defined bond.

This distinction is essential.

A model using:

```text
r_cut = 6 Å
```

may contain edges that are not normally described as conventional chemical bonds.

Therefore, edge importance should often be described as:

```text
important graph interaction
```

rather than automatically:

```text
important chemical bond.
```

---

## 26.10.37 From Edge Importance to Interaction-Level Interpretation

The main conceptual progression is:

```text
Node importance

Which atoms matter?
```

followed by:

```text
Edge importance

Which atomic interactions matter?
```

Together they produce a richer explanation:

```text
                 Prediction
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
     Node importance       Edge importance
          ↓                     ↓
    Important atoms       Important interactions
          ↓                     ↓
    Local environment     Bond/connectivity pattern
          └──────────┬──────────┘
                     ↓
             Scientific analysis
```

This combined representation is substantially more informative than either node or edge explanations alone.

---

## 26.10.38 Key Takeaways

The central ideas of edge importance are:

1. **Edges represent information pathways between atoms in a crystal graph.**

2. **Edge importance asks which interactions are most relevant to the model's prediction.**

3. **Gradient-based methods provide a simple local sensitivity measure.**

4. **Edge features can be analyzed individually or aggregated into a scalar edge score.**

5. **Important graph edges should be mapped back to atomic pairs and structural distances.**

6. **Reverse directed edges may represent the same physical interaction and should be handled carefully.**

7. **Periodic image information must be preserved when interpreting crystal edges.**

8. **Edge importance can be analyzed by element pair, bond distance, coordination environment, and structural motif.**

9. **Defects, dopants, surfaces, adsorption systems, and magnetic materials are particularly useful applications.**

10. **An important graph edge is not automatically a chemically strong bond.**

11. **Edge explanations should be validated across structures, model seeds, and explanation methods.**

12. **Independent DFT or experimental analysis is required before converting an XAI observation into a physical mechanism.**

The combined node-and-edge view now allows the researcher to move from:

```text
Which atoms does the model use?
```

to:

```text
Which interactions does the model use?
```

The next challenge is to determine whether these important nodes and edges form a coherent **substructure or subgraph** responsible for a prediction. This leads naturally to **GNNExplainer**, which attempts to identify a compact graph explanation rather than evaluating nodes and edges independently.

# 26.11 GNNExplainer

The previous section examined **node importance** and **edge importance** independently.

That analysis is useful, but it has an important limitation.

A crystal property is rarely determined by one atom or one atomic interaction in isolation.

Instead, the prediction may depend on a **combination of atoms and interactions forming a local structural motif**.

For example:

```text
        O
        |
   O — Ti — O
        |
        O
```

The model may not simply depend on:

```text
Ti
```

or:

```text
Ti–O
```

independently.

It may depend on the complete local environment:

```text
Ti
+
four O neighbors
+
their distances
+
their connectivity
```

Therefore, a more powerful question is:

> Which subgraph of the crystal is sufficient to explain the model's prediction?

This is the central idea behind **GNNExplainer**.

GNNExplainer was introduced as a model-agnostic explanation framework for graph neural networks. Instead of assigning an importance score independently to every node or edge, it attempts to identify a compact combination of graph components that preserves the model's prediction.

For Materials Informatics, this idea is particularly valuable because the resulting explanation can potentially be interpreted as a **structural motif**.

---

## 26.11.1 From Individual Importance to Subgraph Explanation

The progression developed so far is:

```text
Feature importance
        ↓
Which input features matter?
```

then:

```text
Node importance
        ↓
Which atoms matter?
```

then:

```text
Edge importance
        ↓
Which interactions matter?
```

GNNExplainer extends this to:

```text
Subgraph importance
        ↓
Which combination of atoms and interactions explains the prediction?
```

The complete conceptual progression is:

```text
                    Model Prediction
                           ↓
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
          Features       Nodes         Edges
             ↓             ↓             ↓
             └─────────────┼─────────────┘
                           ↓
                       Subgraph
                           ↓
                  Scientific Structure
                           ↓
                  Materials Interpretation
```

This is especially important for crystal materials because many scientifically meaningful properties arise from **local coordination environments, polyhedra, motifs, defects, and connected structural regions**.

---

## 26.11.2 What Is GNNExplainer Trying to Find?

Consider a crystal graph:

```text
G = (V, E)
```

where:

```text
V → atoms
E → graph interactions
```

Suppose the graph contains:

```text
50 atoms
200 edges
```

but the prediction may actually depend strongly on a much smaller region:

```text
6 atoms
8 important edges
```

GNNExplainer attempts to identify this smaller explanatory structure.

Conceptually:

```text
Full crystal graph

O — Ti — O — O
|    |    |
Ti — O — Ti
|         |
O — O — O — H
      |
     Na
```

might be reduced to:

```text
O — Ti — O
|    |
O — Ti
```

if that region is sufficient to explain the prediction.

The explanation therefore becomes:

> The model's prediction is primarily associated with this local structural subgraph.

This is much closer to a scientific interpretation than a simple list of numerical feature scores.

---

## 26.11.3 Mathematical Representation

GNNExplainer introduces learnable masks over graph components.

Let:

```text
M_E
```

represent an edge mask.

Each edge receives a continuous value:

```text
0 ≤ M_E(i,j) ≤ 1
```

Similarly, a feature mask can be defined as:

```text
M_X
```

The masked graph can then be represented conceptually as:

```text
G' = (V, M_E ⊙ E)
```

where:

```text
⊙
```

denotes element-wise multiplication.

The objective is to find masks that preserve the model's prediction while keeping the explanation sufficiently compact.

Conceptually:

```text
Full graph
    ↓
Apply mask
    ↓
Important subgraph
    ↓
Model prediction
    ↓
Compare with original prediction
```

If removing an edge causes the prediction to change substantially, that edge is potentially important.

If removing an edge has little effect, the edge may be unnecessary for the explanation.

---

## 26.11.4 Why a Mask Is Useful

Suppose a crystal graph contains:

```text
E = 100
```

edges.

A learned mask might produce:

```text
0.02
0.01
0.94
0.87
0.03
...
0.91
```

After applying a threshold:

```python
important_edges = (
    edge_mask > 0.5
)
```

we might obtain:

```text
100 total edges
        ↓
12 important edges
```

The explanation has therefore compressed the graph.

Instead of asking the researcher to inspect the entire crystal, we can focus attention on:

```text
12 edges
+
their connected atoms
```

This is the main practical advantage.

---

## 26.11.5 GNNExplainer in Materials Informatics

For materials problems, a useful workflow is:

```text
Crystal structure
        ↓
Pymatgen
        ↓
Crystal graph
        ↓
GNN
        ↓
Property prediction
        ↓
GNNExplainer
        ↓
Important subgraph
        ↓
Chemical interpretation
```

The important subgraph can then be investigated using:

```text
coordination number
bond distances
bond angles
polyhedral distortion
element identity
local symmetry
defect environment
```

This transforms XAI from a purely computational exercise into a materials-analysis workflow.

---

## 26.11.6 Example: Band-Gap Prediction

Suppose a GNN predicts:

```text
Band gap = 2.14 eV
```

The crystal contains:

```text
A
B
C
D
E
F
G
H
...
```

GNNExplainer may identify:

```text
A — C
 \  |
  D
 /  \
B    E
```

as the important subgraph.

The researcher can then ask:

```text
What elements are A, B, C, D, E?

What are their distances?

What coordination environment do they form?

Does this motif occur repeatedly?

Is it associated with known electronic behavior?
```

The model explanation therefore becomes a hypothesis-generation mechanism.

---

## 26.11.7 GNNExplainer in PyTorch Geometric

PyTorch Geometric provides explainability functionality for graph neural networks.

A typical workflow begins by importing the required components:

```python
import torch

from torch_geometric.explain import (
    Explainer,
    GNNExplainer
)
```

The exact API can vary between PyTorch Geometric versions, so researchers should always check the documentation corresponding to the installed version.

The general structure is:

```python
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(...),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=...
)
```

The important conceptual parameters are:

```text
model
algorithm
explanation type
node mask
edge mask
model configuration
```

---

## 26.11.8 Model Configuration

Before creating an explainer, the model's output behavior must be specified.

For example, a crystal-property model may perform graph-level regression:

```text
Crystal graph
      ↓
GNN
      ↓
one scalar
      ↓
formation energy
```

This is different from node classification:

```text
Graph
 ↓
GNN
 ↓
prediction for every node
```

Materials property prediction usually corresponds to **graph-level prediction**.

Therefore, the explanation configuration must reflect the actual task.

A simplified configuration concept is:

```python
model_config = dict(
    mode="regression",
    task_level="graph",
    return_type="raw"
)
```

The exact configuration should match the installed PyTorch Geometric API and the model's output.

---

## 26.11.9 Basic GNNExplainer Structure

A conceptual implementation is:

```python
explainer = Explainer(
    model=model,

    algorithm=GNNExplainer(
        epochs=200,
        lr=0.01
    ),

    explanation_type="model",

    node_mask_type="attributes",

    edge_mask_type="object",

    model_config=model_config
)
```

The parameters control the optimization of the explanation masks.

For example:

```text
epochs
```

determines how long the explanation optimization is performed.

A larger number of epochs may provide a more optimized mask but increases computation.

---

## 26.11.10 Generating an Explanation

For a graph:

```python
explanation = explainer(
    x=x,
    edge_index=edge_index
)
```

If the model requires edge attributes:

```python
explanation = explainer(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)
```

The resulting explanation can contain:

```text
node mask
edge mask
feature mask
```

depending on configuration.

For example:

```python
print(
    explanation.edge_mask
)
```

may return:

```text
tensor([
    0.04,
    0.02,
    0.91,
    0.83,
    0.07,
    ...
])
```

These values represent the learned edge-level explanation mask.

---

## 26.11.11 Extracting Important Edges

A simple threshold can be applied:

```python
edge_mask = (
    explanation.edge_mask
)

important_edges = (
    edge_mask > 0.5
)
```

Then:

```python
selected_edge_index = (
    edge_index[
        :,
        important_edges
    ]
)
```

The resulting graph contains only the selected edges.

However, thresholding should not be treated as a universal rule.

A threshold of `0.5` is convenient, but:

```text
0.5
```

does not have a universal physical meaning.

The researcher should inspect the mask distribution and perform sensitivity analysis.

---

## 26.11.12 Ranking Instead of Thresholding

An alternative is to rank edges.

```python
ranking = torch.argsort(
    edge_mask,
    descending=True
)
```

The top `k` edges can then be selected:

```python
k = 10

top_edge_ids = (
    ranking[:k]
)
```

This is often more useful for visualization.

For example:

```text
Top 10 edges
```

can be compared across several crystals.

---

## 26.11.13 Mapping the GNNExplainer Mask to Chemical Interactions

The mask itself is not yet chemically meaningful.

We must map each selected edge to:

```text
source atom
target atom
element pair
distance
periodic information
```

For example:

```python
records = []

for edge_id in top_edge_ids:

    edge_id = edge_id.item()

    i = edge_index[
        0, edge_id
    ].item()

    j = edge_index[
        1, edge_id
    ].item()

    records.append({
        "edge_id": edge_id,
        "source": i,
        "target": j,
        "source_element":
            structure[
                i
            ].species_string,
        "target_element":
            structure[
                j
            ].species_string,
        "distance":
            structure.get_distance(
                i,
                j
            ),
        "mask":
            edge_mask[
                edge_id
            ].item()
    })
```

Then:

```python
import pandas as pd

explanation_df = pd.DataFrame(
    records
)

print(explanation_df)
```

The output might resemble:

```text
| edge | source | target | pair | distance | mask |
|------|--------|--------|------|----------|------|
| 17   | 4      | 11     | Ti-O | 1.91 Å   | 0.96 |
| 23   | 7      | 12     | Ti-O | 2.03 Å   | 0.91 |
| 31   | 2      | 9      | Fe-O | 1.88 Å   | 0.87 |
```

Now the explanation can be interpreted scientifically.

---

## 26.11.14 The Difference Between Edge Importance and GNNExplainer

It is important not to treat the previous section and this section as duplicates.

Gradient-based edge importance asks:

> How sensitive is the prediction to each edge?

GNNExplainer asks:

> Which subset of edges can provide a compact explanation of the prediction?

Therefore:

```text
Gradient method
     ↓
Sensitivity ranking
```

whereas:

```text
GNNExplainer
     ↓
Optimized explanatory subgraph
```

The two methods can produce different results.

That difference itself can be scientifically informative.

---

## 26.11.15 Explanation as a Subgraph

Suppose the original crystal graph is:

```text
A — B — C — D
|   |   |   |
E — F — G — H
|       |
I — J — K
```

GNNExplainer may identify:

```text
B — C
|   |
F — G
```

as the important region.

This can be interpreted as:

```text
Important local structural motif
```

rather than merely:

```text
B important
C important
F important
G important
```

The connectivity is part of the explanation.

This is one of the major advantages of graph-specific XAI.

---

## 26.11.16 Why Subgraph Connectivity Matters in Crystals

Materials properties frequently depend on connected structural motifs.

Examples include:

```text
MO₆ octahedra
MO₄ tetrahedra
perovskite frameworks
layered coordination networks
metal–oxygen networks
polyhedral chains
corner-sharing units
edge-sharing units
```

A node-only explanation might identify several atoms.

An edge-only explanation might identify several interactions.

A subgraph explanation can reveal:

```text
which atoms
+
which connections
+
which local arrangement
```

together.

That is much closer to the language used by materials scientists.

---

## 26.11.17 Example: Octahedral Environment

Consider:

```text
        O
        |
    O — Ti — O
        |
        O
```

The model may predict formation energy.

GNNExplainer could identify the Ti atom and all four Ti–O edges.

This suggests that the local TiO₄ environment is important.

But suppose another explanation identifies:

```text
Ti
|
O — Ti
```

instead.

The researcher should not immediately conclude that the model has discovered a new mechanism.

The explanation must be compared against:

```text
coordination number
bond lengths
bond-angle distributions
polyhedral distortion
dataset statistics
```

and, where appropriate:

```text
DFT electronic or energetic analysis.
```

---

## 26.11.18 Explanation Stability

An important research question is:

> Will GNNExplainer identify the same subgraph if the model is retrained?

Suppose five models are trained with different random seeds:

```text
Model 1 → motif A
Model 2 → motif A
Model 3 → motif B
Model 4 → motif A
Model 5 → motif A
```

The repeated identification of motif A is more convincing than a motif identified by only one model.

Therefore, explanation stability should be evaluated.

---

## 26.11.19 Repeated Explanation Experiment

A simple framework is:

```python
all_explanations = []

for seed in range(5):

    torch.manual_seed(seed)

    model = train_model(
        seed=seed
    )

    explanation = explainer(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    all_explanations.append(
        explanation.edge_mask.detach()
    )
```

Now the masks can be compared.

For example:

```python
mask_matrix = torch.stack(
    all_explanations
)

mean_mask = (
    mask_matrix.mean(
        dim=0
    )
)

std_mask = (
    mask_matrix.std(
        dim=0
    )
)
```

This gives:

```text
mean importance
+
explanation variability
```

An edge with:

```text
mean = 0.90
std  = 0.03
```

is much more stable than:

```text
mean = 0.90
std  = 0.40
```

---

## 26.11.20 Explanation Stability as a Scientific Criterion

A useful hierarchy is:

```text
Single explanation
        ↓
Interesting observation
```

while:

```text
Repeated explanations
        ↓
Stable pattern
```

and:

```text
Stable pattern
+
Independent physical validation
        ↓
Scientific hypothesis
```

This distinction is critical when using XAI in research.

Explainability should not be used merely to generate attractive pictures.

It should help produce **reproducible scientific insight**.

---

## 26.11.21 Comparing GNNExplainer Across Materials

Suppose the dataset contains:

```text
1000 crystals
```

and GNNExplainer is applied to:

```text
200 test structures
```

The resulting subgraphs can be analyzed collectively.

For example:

```text
Most common important interaction:

Ti–O     42%
Fe–O     27%
O–O      11%
Ti–Ti     8%
Other    12%
```

This can reveal recurring structural patterns.

However, the frequencies must be interpreted relative to how frequently those interactions occur in the original dataset.

Otherwise, common interactions may appear important simply because they are common.

---

## 26.11.22 Normalizing Interaction Frequency

Suppose:

```text
Ti–O appears in 900/1000 structures
Fe–O appears in 100/1000 structures
```

and:

```text
Ti–O explanation frequency = 450
Fe–O explanation frequency = 70
```

Raw counts favor Ti–O.

A normalized measure can be:

```text
explanation frequency
---------------------
occurrence frequency
```

For example:

```python
normalized_score = (
    explanation_count
    /
    occurrence_count
)
```

This asks:

> Given that an interaction exists, how often does it appear in the explanation?

That can be much more informative for dataset-level analysis.

---

## 26.11.23 GNNExplainer for Defect Structures

Defect systems are an especially interesting application.

Suppose a vacancy is introduced:

```text
Perfect crystal
      ↓
Remove atom
      ↓
Relax structure
      ↓
Build graph
      ↓
GNN prediction
      ↓
GNNExplainer
```

The explanation may identify edges surrounding the vacancy.

For example:

```text
O — M — O
 \     /
  vacancy
```

If the important subgraph consistently surrounds the defect, the model may be relying on the local defect environment.

This can be compared with:

```text
local relaxation
charge redistribution
bond-length changes
electronic density
```

from independent calculations.

---

## 26.11.24 GNNExplainer for Dopant Environments

For doped materials:

```text
Host lattice
     ↓
Dopant
     ↓
Local structural perturbation
```

GNNExplainer can potentially identify:

```text
dopant node
+
nearest neighbors
+
important connecting edges
```

This gives a local explanation such as:

```text
Dopant
  |
Host₁
  |
Host₂
```

rather than simply reporting:

```text
Dopant feature = important
```

This distinction is useful when investigating how substitution modifies the surrounding crystal environment.

---

## 26.11.25 GNNExplainer for Formation Energy

Formation energy is a natural graph-level regression target.

A simplified workflow is:

```python
prediction = model(
    x,
    edge_index,
    edge_attr,
    batch
)
```

Then:

```python
explanation = explainer(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    batch=batch
)
```

The resulting subgraph can be investigated for:

```text
chemical environment
coordination
bond lengths
polyhedral connectivity
```

The researcher can then compare the explanation with known energetic descriptors.

---

## 26.11.26 GNNExplainer for Band Gap

For band-gap prediction, explanations may highlight local environments associated with electronic structure.

Possible patterns include:

```text
transition-metal coordination
oxygen environments
layered structures
local distortions
specific element pairs
```

However, band gap is fundamentally an electronic property.

Therefore, an XAI explanation should ideally be validated using electronic-structure quantities such as:

```text
DOS
PDOS
band structure
orbital character
charge density
```

The workflow becomes:

```text
GNN explanation
       ↓
Important structural motif
       ↓
Electronic-structure analysis
       ↓
Physical interpretation
```

---

## 26.11.27 GNNExplainer for Mechanical Properties

For elastic or mechanical-property prediction, the important subgraph may involve:

```text
strongly connected polyhedra
framework connectivity
layer structures
coordination networks
```

The explanation may therefore reveal structural motifs associated with rigidity.

Again, the XAI result should be treated as a hypothesis.

Independent analysis might include:

```text
elastic tensor
bond stiffness
phonon calculations
deformation analysis
```

depending on the research question.

---

## 26.11.28 A Complete GNNExplainer Workflow

A research-grade workflow can be summarized as:

```text
Crystal Dataset
       ↓
Graph Construction
       ↓
GNN Training
       ↓
Model Evaluation
       ↓
Select Structure
       ↓
Generate GNNExplainer Mask
       ↓
Extract Important Edges
       ↓
Identify Important Nodes
       ↓
Construct Explanatory Subgraph
       ↓
Map to Crystal Structure
       ↓
Analyze Chemical Environment
       ↓
Compare Across Materials
       ↓
Validate Physically
```

The important point is that the explanation is not the final result.

It is an intermediate step between:

```text
machine-learning prediction
```

and:

```text
scientific interpretation.
```

---

## 26.11.29 Practical Code Pipeline

A simplified research implementation can be organized as:

```python
# 1. Train model
model = train_model(
    train_loader,
    val_loader
)

# 2. Select a crystal
data = test_dataset[
    sample_id
]

# 3. Generate explanation
explanation = explainer(
    x=data.x,
    edge_index=data.edge_index,
    edge_attr=data.edge_attr
)

# 4. Extract masks
edge_mask = (
    explanation.edge_mask
)

# 5. Rank edges
ranking = torch.argsort(
    edge_mask,
    descending=True
)

# 6. Select top interactions
top_edges = ranking[:10]

# 7. Convert to chemical information
records = []

for edge_id in top_edges:

    edge_id = edge_id.item()

    i = data.edge_index[
        0, edge_id
    ].item()

    j = data.edge_index[
        1, edge_id
    ].item()

    records.append({
        "edge_id": edge_id,
        "source": i,
        "target": j,
        "mask":
            edge_mask[
                edge_id
            ].item()
    })

# 8. Inspect explanation
explanation_df = pd.DataFrame(
    records
)

print(explanation_df)
```

This pipeline can then be extended with:

```text
Pymatgen
matminer
visualization
DFT validation
statistical analysis
```

---

## 26.11.30 Visualizing the Explanatory Subgraph

The final goal is often to display the important region of the crystal.

Conceptually:

```text
Full crystal
     ↓
GNNExplainer mask
     ↓
Important subgraph
     ↓
3D crystal visualization
```

For a research workflow, the visualization should show:

```text
atom identity
edge identity
edge importance
distance
coordination
```

rather than simply displaying a generic graph.

The important edges should be clearly distinguished from the remaining structure.

For example:

```text
Full crystal:
gray/background structure

Important subgraph:
highlighted structure
```

The visualization should include a legend.

---

## 26.11.31 What GNNExplainer Does Not Prove

This is one of the most important principles in this chapter.

GNNExplainer does **not** prove that:

```text
important edge
=
causal physical mechanism
```

It identifies a graph region that is useful for explaining the model's prediction.

Therefore:

```text
GNNExplainer
```

answers approximately:

> Which graph components are sufficient or highly useful for reproducing this prediction?

It does not directly answer:

> What fundamental physical law causes this material property?

The second question requires scientific investigation.

---

## 26.11.32 Common Failure Modes

Several problems can occur.

### 1. Unstable masks

Different runs may produce different subgraphs.

### 2. Too many edges

The explanation may remain almost as large as the original graph.

### 3. Too few edges

The mask may remove important contextual information.

### 4. Model dependence

Different GNN architectures may produce different explanations.

### 5. Dataset bias

The explanation may reflect correlations in the training dataset.

### 6. Graph-construction bias

The explanation is limited to the edges that exist in the graph.

### 7. Chemical misinterpretation

An important graph edge may not represent a conventional chemical bond.

These limitations must be reported in serious Materials Informatics research.

---

## 26.11.33 Improving Explanation Reliability

A robust study should combine several approaches:

```text
GNNExplainer
     +
Gradient attribution
     +
Integrated gradients
     +
Node importance
     +
Edge importance
```

If several methods identify the same region:

```text
Method A → motif X
Method B → motif X
Method C → motif X
```

confidence in the interpretation increases.

This is known as **explanation agreement**.

It does not prove causality, but it provides stronger evidence that the model consistently relies on the identified structural pattern.

---

## 26.11.34 Cross-Model Explanation

A similar experiment can compare different architectures.

For example:

```text
CGCNN
MEGNet
ALIGNN
```

can all be trained for the same task.

Then:

```text
CGCNN explanation → motif A
MEGNet explanation → motif A
ALIGNN explanation → motif A
```

would provide stronger evidence than:

```text
CGCNN explanation → motif A
```

alone.

This is particularly valuable when the goal is to make a scientific claim about a learned structure–property relationship.

---

## 26.11.35 From GNNExplainer to Scientific Discovery

The ultimate purpose of GNN explainability is not visualization.

The purpose is to identify patterns that can generate new scientific hypotheses.

The complete loop is:

```text
Materials Dataset
       ↓
GNN
       ↓
Prediction
       ↓
GNNExplainer
       ↓
Important Subgraph
       ↓
Structural Interpretation
       ↓
Scientific Hypothesis
       ↓
DFT / Experiment
       ↓
Validation
       ↓
New Understanding
```

This transforms explainable AI from:

```text
"Why did the model predict this?"
```

into a much more useful scientific question:

```text
"What structural patterns has the model learned,
and can those patterns teach us something about materials?"
```

---

## 26.11.36 Key Takeaways

The central ideas of GNNExplainer are:

1. **GNNExplainer identifies graph components that help explain a model prediction.**

2. **It moves beyond isolated node or edge importance toward subgraph-level explanations.**

3. **Learned masks can identify important edges and features.**

4. **A crystal explanation can potentially correspond to a meaningful local structural motif.**

5. **Important subgraphs should be mapped back to real crystal structures using Pymatgen or equivalent tools.**

6. **Chemical interpretation should consider element identity, distance, coordination, geometry, and periodicity.**

7. **Explanation stability should be tested across random seeds and model retraining.**

8. **Dataset-level analysis can reveal recurring structural motifs associated with predictions.**

9. **GNNExplainer does not establish physical causality.**

10. **Independent DFT or experimental analysis is required to validate scientific hypotheses derived from explanations.**

11. **Combining GNNExplainer with gradient, node, edge, and other attribution methods provides stronger evidence than relying on one explanation technique.**

12. **For Materials Informatics, the most valuable outcome is often the identification of interpretable structural motifs rather than a numerical explanation score alone.**

The conceptual transition is now:

```text
Feature Importance
        ↓
Node Importance
        ↓
Edge Importance
        ↓
GNNExplainer
        ↓
Important Subgraph
        ↓
Structural Motif
        ↓
Scientific Interpretation
```

GNNExplainer therefore provides the bridge between **individual graph components** and **coherent crystal structures**.

The next stage is to examine a different class of attribution methods that does not rely on optimizing a graph mask: **Integrated Gradients**, which provides a principled path-based attribution framework for understanding how individual input features contribute to a prediction.

# 26.12 Integrated Gradients

GNNExplainer provides a powerful way to identify an explanatory subgraph, but it relies on optimizing an explanation mask.

A different approach is to ask a more fundamental question:

> How much did each input feature contribute to the change from a reference input to the actual material?

This is the central idea behind **Integrated Gradients (IG)**.

Integrated Gradients is a gradient-based attribution method originally developed for neural networks. Instead of examining the gradient at only one point, it integrates gradients along a continuous path from a **baseline input** to the actual input.

For Materials Informatics, this provides an important way to investigate:

```text
Crystal representation
        ↓
Input features
        ↓
Integrated Gradients
        ↓
Feature attribution
        ↓
Scientific interpretation
```

For a crystal GNN, the inputs may include:

```text
atomic features
element identity
atomic number
electronegativity
valence information
local descriptors
edge features
interatomic distance
radial basis features
```

Integrated Gradients can therefore help answer:

> Which parts of the numerical representation contributed most strongly to the model's prediction?

This makes Integrated Gradients complementary to GNNExplainer.

GNNExplainer primarily asks:

```text
Which graph components explain the prediction?
```

Integrated Gradients asks:

```text
How did the input features contribute to the prediction?
```

---

## 26.12.1 Why Integrated Gradients Is Different

Consider a neural network:

```text
Input
 ↓
Neural Network
 ↓
Prediction
```

Suppose the input contains:

```text
x₁
x₂
x₃
...
xₙ
```

A simple gradient method calculates:

```text
∂F/∂xᵢ
```

at the actual input.

This tells us the local sensitivity of the prediction to the feature.

However, the gradient at one point can be misleading.

A feature may have:

```text
small gradient at the final input
```

while having made a large contribution along the path from the baseline to the actual input.

Integrated Gradients addresses this problem by accumulating gradients along the entire path.

---

## 26.12.2 Mathematical Definition

Let:

```text
F(x)
```

be the model output.

Let:

```text
x'
```

be a baseline input.

Let:

```text
x
```

be the actual input.

The Integrated Gradient attribution for feature `i` is:

```text
IGᵢ(x)
=
(xᵢ - xᵢ')
∫₀¹
∂F(x' + α(x - x'))
-------------------
∂xᵢ
dα
```

The path is:

```text
x(α) = x' + α(x - x')
```

where:

```text
α = 0
```

corresponds to the baseline and:

```text
α = 1
```

corresponds to the actual input.

Thus:

```text
Baseline
   ↓
α = 0
   ↓
intermediate representations
   ↓
α = 0.5
   ↓
Actual material
   ↓
α = 1
```

The method integrates the gradient along this path.

---

## 26.12.3 Intuitive Interpretation

Imagine gradually transforming a reference material representation into the representation of the actual material.

```text
Baseline
   ↓
10%
   ↓
20%
   ↓
30%
   ↓
...
   ↓
100%
   ↓
Actual material
```

At every point, we ask:

```text
How sensitive is the prediction to this feature?
```

The total contribution is accumulated across the entire path.

Therefore:

```text
Integrated Gradients
=
accumulated sensitivity
```

rather than:

```text
single-point sensitivity
```

This is the key conceptual difference.

---

## 26.12.4 Why a Baseline Is Necessary

Integrated Gradients requires a reference input.

The baseline represents a hypothetical starting point from which the actual material representation is constructed.

This creates an important materials-specific question:

> What should the baseline crystal representation be?

There is no universally correct answer.

Possible choices include:

```text
zero feature vector
mean feature vector
reference composition
reference crystal
average material
chemically neutral representation
```

The choice depends on the scientific question.

---

## 26.12.5 Zero Baseline

The simplest baseline is:

```python id="c7t9na"
baseline = torch.zeros_like(
    x
)
```

This means the explanation measures the contribution relative to a zero representation.

For normalized numerical features, this may sometimes correspond approximately to an average or neutral feature value.

However, this depends entirely on the preprocessing.

For example, if a feature was standardized:

```text
z = (x - μ) / σ
```

then:

```text
x = 0
```

corresponds to the dataset mean.

Therefore, the same numerical zero can have very different meanings depending on preprocessing.

---

## 26.12.6 Mean Feature Baseline

Suppose:

```python id="j75y6v"
baseline = x.mean(
    dim=0,
    keepdim=True
)
```

This produces the average feature vector.

The baseline can then be expanded to match the graph:

```python id="fdx2m9"
baseline = baseline.expand_as(
    x
)
```

This can be useful when the scientific question is:

> How does this material differ from the average material representation?

But the average feature vector may not correspond to any physically realizable material.

This is an important limitation.

---

## 26.12.7 Reference Crystal Baseline

A more materials-aware strategy is to use a real reference material.

For example:

```text
target:
BaTiO₃

baseline:
SrTiO₃
```

The explanation then asks:

> Which input features distinguish the target prediction from the reference material?

This can be particularly useful for comparative materials studies.

For example:

```text
Parent material
      ↓
Chemical substitution
      ↓
Target material
```

Integrated Gradients can then help identify which representation changes are associated with the prediction difference.

---

## 26.12.8 Integrated Gradients and Feature Scaling

Integrated Gradients is sensitive to the numerical representation.

Suppose one feature has:

```text
range = 0–1
```

while another has:

```text
range = 0–1000
```

Their raw attribution magnitudes may not be directly comparable.

Therefore, materials ML workflows should normally perform appropriate preprocessing before attribution.

For example:

```python id="r4t3ga"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)
```

The model is then trained using:

```python id="oj8yq4"
X_scaled
```

and explanations are generated in the same representation space.

---

## 26.12.9 Completeness Property

One of the important theoretical properties of Integrated Gradients is the **completeness property**.

The sum of feature attributions approximately equals the difference between the model output for the actual input and the baseline.

Conceptually:

```text
Σᵢ IGᵢ(x)
≈
F(x) - F(x')
```

Therefore:

```text
prediction difference
=
sum of feature contributions
```

up to numerical integration error.

This is useful because it provides a consistency check for the explanation.

---

## 26.12.10 Numerical Approximation

The integral is usually approximated numerically.

Choose:

```text
m
```

interpolation points.

Then:

```text
IGᵢ(x)
≈
(xᵢ - xᵢ')
×
(1/m)
Σₖ
∂F(x' + k/m(x-x'))
-------------------
∂xᵢ
```

The simplest implementation uses a Riemann sum.

A larger number of steps generally gives a better approximation but increases computational cost.

---

## 26.12.11 Implementing Integrated Gradients Manually

A basic PyTorch implementation can be written directly.

```python id="qum3gn"
import torch


def integrated_gradients(
    model,
    x,
    baseline,
    forward_fn,
    steps=100
):

    scaled_inputs = []

    for alpha in torch.linspace(
        0.0,
        1.0,
        steps
    ):

        scaled = (
            baseline
            +
            alpha * (
                x - baseline
            )
        )

        scaled_inputs.append(
            scaled
        )

    gradients = []

    for scaled in scaled_inputs:

        scaled = (
            scaled
            .clone()
            .detach()
            .requires_grad_(True)
        )

        output = forward_fn(
            scaled
        )

        model.zero_grad()

        output.backward(
            retain_graph=False
        )

        gradients.append(
            scaled.grad.detach()
        )

    gradients = torch.stack(
        gradients
    )

    avg_gradient = (
        gradients.mean(
            dim=0
        )
    )

    attribution = (
        (x - baseline)
        *
        avg_gradient
    )

    return attribution
```

This is the basic numerical idea.

In real GNN applications, the `forward_fn` must include the graph structure and any additional inputs required by the model.

---

## 26.12.12 Applying the Method to a Crystal GNN

Suppose a model accepts:

```python id="o9w8l1"
x
edge_index
edge_attr
batch
```

and predicts a graph-level property:

```python id="xv1a6z"
prediction = model(
    x,
    edge_index,
    edge_attr,
    batch
)
```

We can define a function that varies only the node features:

```python id="14w2tg"
def forward_node_features(
    x_input
):

    return model(
        x_input,
        edge_index,
        edge_attr,
        batch
    ).squeeze()
```

Then:

```python id="4j7d5h"
baseline = torch.zeros_like(
    x
)
```

and:

```python id="v1k8tb"
attributions = (
    integrated_gradients(
        model=model,
        x=x,
        baseline=baseline,
        forward_fn=
            forward_node_features,
        steps=100
    )
)
```

The result has the same shape as:

```text
x
```

For example:

```text
x.shape
=
[number_of_atoms, number_of_features]
```

therefore:

```text
attributions.shape
=
[number_of_atoms, number_of_features]
```

This is extremely useful for crystal interpretation.

---

## 26.12.13 Feature-Level Attribution

Suppose each atom has:

```text
10 features
```

Then:

```python id="a2jmb4"
print(
    attributions.shape
)
```

may produce:

```text
torch.Size([32, 10])
```

for a 32-atom crystal.

Each row corresponds to an atom.

Each column corresponds to an atomic feature.

For example:

```text
             Z     radius   EN    valence
Atom 1      0.42   -0.13   0.08    0.04
Atom 2      0.71    0.02  -0.22    0.11
Atom 3     -0.04    0.03   0.01    0.02
```

This allows the researcher to ask two separate questions:

```text
Which atoms matter?
```

and:

```text
Which atomic features matter?
```

---

## 26.12.14 Converting Feature Attributions to Atom Importance

A scalar score can be obtained by aggregating across features.

For example:

```python id="0s8vqi"
atom_importance = (
    attributions
    .abs()
    .sum(dim=1)
)
```

Alternatively:

```python id="shxy93"
atom_importance = (
    attributions
    .norm(
        p=2,
        dim=1
    )
)
```

The resulting vector has:

```text
one score per atom
```

For example:

```text
Atom 0 → 0.81
Atom 1 → 0.12
Atom 2 → 0.74
Atom 3 → 0.03
```

This can then be compared with node importance obtained using other methods.

---

## 26.12.15 Signed Versus Absolute Attribution

Integrated Gradients produces signed values.

For example:

```text
+0.82
-0.61
+0.12
-0.04
```

The sign contains information.

A positive attribution means the feature contributes in the direction of increasing the selected model output.

A negative attribution means the feature contributes in the opposite direction.

Therefore, blindly taking absolute values removes information.

For example:

```text
+0.8
```

and:

```text
-0.8
```

have the same magnitude but opposite effects.

For scientific interpretation, both should often be reported.

---

## 26.12.16 Positive and Negative Contributions

A useful separation is:

```python id="3z3t6r"
positive = torch.clamp(
    attributions,
    min=0
)

negative = torch.clamp(
    attributions,
    max=0
)
```

The total positive contribution is:

```python id="3z6y8n"
positive_sum = (
    positive.sum()
)
```

and:

```python id="6jz1jr"
negative_sum = (
    negative.sum()
)
```

The researcher can therefore distinguish:

```text
features supporting the prediction
```

from:

```text
features suppressing the prediction.
```

---

## 26.12.17 Example: Formation Energy

Suppose the target is formation energy.

The model predicts:

```text
-2.45 eV/atom
```

and the baseline prediction is:

```text
-1.20 eV/atom
```

Then:

```text
prediction difference
=
-1.25 eV/atom
```

Integrated Gradients distributes this difference across the input features.

For example:

```text
Ti atomic identity        -0.41
O electronegativity       -0.28
Ti atomic radius          -0.19
O valence                 -0.12
distance feature          -0.17
other features            -0.08
```

approximately:

```text
sum = -1.25
```

This provides a decomposition of the model's prediction relative to the selected baseline.

---

## 26.12.18 Attribution Is Relative to the Baseline

This is one of the most important points in interpreting Integrated Gradients.

Suppose:

```text
baseline A
```

produces:

```text
F(A) = 1.0
```

while:

```text
target
```

produces:

```text
F(x) = 2.0
```

The explanation distributes:

```text
2.0 - 1.0
=
1.0
```

across the input features.

If a different baseline produces:

```text
F(B) = 1.5
```

then the attribution distributes:

```text
2.0 - 1.5
=
0.5
```

instead.

Therefore:

> Integrated Gradients explanations are always relative to the chosen baseline.

This means baseline selection is a scientific modeling decision, not merely a technical implementation detail.

---

## 26.12.19 Baseline Sensitivity Analysis

A robust materials XAI study should test multiple baselines.

For example:

```python id="1c2r3x"
baselines = {

    "zero":
        torch.zeros_like(x),

    "mean":
        x.mean(
            dim=0,
            keepdim=True
        ).expand_as(x),

}
```

Then:

```python id="z1y6ru"
results = {}

for name, baseline in baselines.items():

    results[name] = (
        integrated_gradients(
            model=model,
            x=x,
            baseline=baseline,
            forward_fn=
                forward_node_features,
            steps=100
        )
    )
```

The explanations can then be compared.

If the same atoms and features remain important:

```text
zero baseline → Ti–O environment
mean baseline → Ti–O environment
reference baseline → Ti–O environment
```

the interpretation becomes more robust.

---

## 26.12.20 Integrated Gradients for Edge Features

Integrated Gradients is not restricted to node features.

Suppose the GNN uses:

```text
edge_attr
```

containing:

```text
distance
radial basis functions
directional features
```

We can define:

```python id="yq4w3h"
def forward_edge_features(
    edge_input
):

    return model(
        x,
        edge_index,
        edge_input,
        batch
    ).squeeze()
```

Then:

```python id="v1w7i2"
edge_baseline = (
    torch.zeros_like(
        edge_attr
    )
)
```

and:

```python id="7ipk8v"
edge_attribution = (
    integrated_gradients(
        model=model,
        x=edge_attr,
        baseline=edge_baseline,
        forward_fn=
            forward_edge_features,
        steps=100
    )
)
```

The output has shape:

```text
[number_of_edges, number_of_edge_features]
```

This allows direct attribution to interaction descriptors.

---

## 26.12.21 Distance Attribution

Suppose the first edge feature represents:

```text
interatomic distance
```

Then:

```python id="s9g3j7"
distance_attribution = (
    edge_attribution[:, 0]
)
```

This allows analysis of how the distance representation contributes to the prediction.

For example:

```text
Edge       Distance       Attribution
Ti–O₁      1.89 Å          -0.42
Ti–O₂      1.94 Å          -0.31
Ti–O₃      2.02 Å          -0.17
O–O        2.80 Å          +0.03
```

The researcher can then investigate whether the model is particularly sensitive to certain interaction distances.

Again, this is a model attribution, not automatically a physical bonding law.

---

## 26.12.22 Integrated Gradients for Crystal Graph Features

A typical crystal GNN may have a representation such as:

```text
Node features:
    atomic number
    atomic mass
    electronegativity
    covalent radius
    oxidation state
    valence features

Edge features:
    distance
    radial basis functions
    directional information
```

Integrated Gradients can therefore generate:

```text
             Prediction
                 ↓
       ┌─────────┴─────────┐
       ↓                   ↓
Node attribution     Edge attribution
       ↓                   ↓
Atomic features      Interaction features
       └─────────┬─────────┘
                 ↓
        Materials interpretation
```

This provides a more detailed explanation than a single scalar prediction.

---

## 26.12.23 Feature Names Must Be Preserved

A common implementation mistake is to generate attribution arrays without retaining feature names.

Instead, define:

```python id="ph7p1h"
node_feature_names = [
    "atomic_number",
    "atomic_mass",
    "electronegativity",
    "covalent_radius",
    "valence"
]
```

Then:

```python id="5d8o7y"
feature_importance = (
    attributions
    .abs()
    .mean(dim=0)
)
```

Create a table:

```python id="d3zuxo"
feature_df = pd.DataFrame({
    "feature":
        node_feature_names,

    "importance":
        feature_importance
        .cpu()
        .numpy()
})
```

Sort it:

```python id="1c8q1g"
feature_df = (
    feature_df
    .sort_values(
        "importance",
        ascending=False
    )
)
```

Now the result is scientifically interpretable.

---

## 26.12.24 Dataset-Level Integrated Gradients

A single explanation can be misleading.

A stronger analysis computes Integrated Gradients across many materials.

For example:

```python id="wd4n1f"
all_attributions = []

for data in test_dataset:

    attribution = (
        explain_material(
            model,
            data
        )
    )

    all_attributions.append(
        attribution
    )
```

The explanations can then be aggregated.

For example:

```python id="0l84m0"
mean_importance = torch.stack(
    all_attributions
).abs().mean(
    dim=0
)
```

This provides a dataset-level estimate of which features the model relies on most strongly.

---

## 26.12.25 Distribution of Feature Attributions

Mean attribution alone may hide important variation.

Suppose:

```text
electronegativity
mean attribution = 0.01
```

This may appear unimportant.

But the distribution could be:

```text
-0.90
-0.70
-0.30
+0.80
+0.95
```

with positive and negative contributions cancelling.

Therefore, researchers should often examine:

```text
mean
absolute mean
median
standard deviation
percentiles
```

rather than only one statistic.

---

## 26.12.26 Attribution by Chemical Element

A materials-specific analysis can aggregate attribution by element.

Suppose:

```python id="5f9pjx"
records = []

for atom_id in range(
    len(structure)
):

    element = (
        structure[
            atom_id
        ].species_string
    )

    score = (
        attributions[
            atom_id
        ].abs().sum().item()
    )

    records.append({
        "element": element,
        "importance": score
    })
```

Then:

```python id="6h2tup"
element_df = pd.DataFrame(
    records
)
```

Group:

```python id="ptw5zr"
element_summary = (
    element_df
    .groupby("element")
    ["importance"]
    .agg([
        "mean",
        "median",
        "count"
    ])
)
```

This can reveal whether the model frequently relies on certain elemental environments.

However, again, frequent occurrence must be distinguished from genuine importance.

---

## 26.12.27 Attribution by Local Environment

Element-level analysis may still be too coarse.

For example:

```text
O
```

may occur in:

```text
Ti–O
Fe–O
Al–O
Si–O
```

environments.

A more meaningful analysis combines:

```text
element
+
neighbor identity
+
distance
+
coordination
```

For example:

```text
Ti-centered O environment
Fe-centered O environment
```

This begins to approximate the local chemical environment used by the GNN.

---

## 26.12.28 Integrated Gradients and Substitutions

Consider a materials series:

```text
A₁BO₃
A₂BO₃
A₃BO₃
```

where the A-site element changes.

Integrated Gradients can be used to investigate how the representation changes across the series.

Workflow:

```text
Material 1
   ↓
Material 2
   ↓
Material 3
```

For each material:

```python id="6xyd0g"
attr = integrated_gradients(
    ...
)
```

Then compare:

```text
A-site attribution
B-site attribution
O-site attribution
distance attribution
```

This can help determine whether the model's prediction changes primarily because of:

```text
chemical identity
local geometry
interaction distances
```

or some combination.

---

## 26.12.29 Integrated Gradients and Counterfactual Thinking

Integrated Gradients itself is not a counterfactual method, but it supports counterfactual reasoning.

Suppose:

```text
Material A
```

has:

```text
band gap = 2.0 eV
```

and:

```text
Material B
```

has:

```text
band gap = 3.0 eV
```

The researcher can compare their attribution patterns.

For example:

```text
             A       B
A-site       +0.2    +0.7
B-site       -0.1    +0.3
O features   +0.4    +0.5
distance     -0.2    +0.6
```

This suggests that the difference in predictions may be associated with changes in:

```text
A-site representation
distance features
```

Such observations can motivate controlled computational experiments.

---

## 26.12.30 Integrated Gradients Versus Raw Gradients

Raw gradient:

```text
∂F/∂x
```

measures local sensitivity.

Integrated Gradients:

```text
∫ gradient along path
```

captures accumulated sensitivity.

Conceptually:

```text
Raw Gradient

Baseline ───────────────► Material
                         ↑
                  evaluate here
```

whereas:

```text
Integrated Gradients

Baseline
   ↓
10%
   ↓
20%
   ↓
30%
   ↓
...
   ↓
100%
   ↓
Material

Gradient evaluated throughout
the path
```

Therefore, Integrated Gradients can provide a more complete attribution than a single gradient evaluation.

---

## 26.12.31 Numerical Convergence

The number of integration steps matters.

For example:

```python id="j0nq5b"
for steps in [
    20,
    50,
    100,
    200,
    500
]:
```

compute the attribution and compare:

```text
steps = 20
steps = 50
steps = 100
steps = 200
steps = 500
```

If the explanation stabilizes:

```text
100 ≈ 200 ≈ 500
```

then the numerical approximation is likely sufficient.

If it changes substantially:

```text
20 ≠ 50 ≠ 100 ≠ 200
```

the integration resolution should be increased.

---

## 26.12.32 Convergence Check in Code

A simple comparison is:

```python id="b5g0fr"
attributions = {}

for steps in [
    20,
    50,
    100,
    200
]:

    attributions[steps] = (
        integrated_gradients(
            model,
            x,
            baseline,
            forward_node_features,
            steps=steps
        )
    )
```

Then calculate differences:

```python id="i6ddkq"
difference = (
    attributions[200]
    -
    attributions[100]
)

error = (
    difference.abs()
    .mean()
)

print(
    "Convergence error:",
    error.item()
)
```

This is much better than selecting `100` steps arbitrarily.

---

## 26.12.33 Completeness Check in Code

The model outputs can be compared:

```python id="2v3f5s"
with torch.no_grad():

    baseline_output = (
        forward_node_features(
            baseline
        )
    )

    target_output = (
        forward_node_features(
            x
        )
    )
```

Then:

```python id="w3cth6"
attribution_sum = (
    attributions
    .sum()
)
```

The expected relationship is approximately:

```python id="r0qzue"
difference = (
    target_output
    -
    baseline_output
)

print(
    "Attribution sum:",
    attribution_sum.item()
)

print(
    "Output difference:",
    difference.item()
)
```

The two values should be reasonably close, subject to numerical approximation and implementation details.

This is an important sanity check.

---

## 26.12.34 Integrated Gradients and Crystal Symmetry

Crystal symmetry introduces an additional consideration.

Equivalent atomic sites should often have equivalent structural roles.

If two symmetry-equivalent atoms receive dramatically different attribution values, several possibilities exist:

```text
model behavior
graph representation
numerical noise
symmetry breaking
different node indexing
```

Therefore, attribution should sometimes be aggregated over symmetry-equivalent sites.

Pymatgen can be used to analyze symmetry, while the attribution values are mapped onto the corresponding sites.

Conceptually:

```text
Symmetry-equivalent sites
        ↓
Group attribution
        ↓
Compare average contribution
```

This can produce a more physically meaningful explanation.

---

## 26.12.35 Attribution on Periodic Crystal Graphs

Periodic graphs introduce another complication.

An atom may interact with an image of another atom:

```text
i → j + T
```

where:

```text
T
```

is a lattice translation.

If edge attributions are analyzed, the periodic image must be retained.

Otherwise, two physically distinct interactions may appear to be the same.

Therefore an edge attribution record should ideally contain:

```python id="b4q6o5"
{
    "source": i,
    "target": j,
    "translation": T,
    "distance": distance,
    "attribution": score
}
```

This preserves the crystallographic meaning of the attribution.

---

## 26.12.36 A Complete Crystal Feature Attribution Function

A reusable function can simplify experiments.

```python id="p2b3c7"
def explain_node_features(
    model,
    data,
    baseline,
    steps=100
):

    x = (
        data.x
        .clone()
        .detach()
    )

    baseline = (
        baseline
        .clone()
        .detach()
    )

    def forward_fn(
        x_input
    ):

        return model(
            x_input,
            data.edge_index,
            data.edge_attr,
            data.batch
        ).squeeze()

    attributions = (
        integrated_gradients(
            model=model,
            x=x,
            baseline=baseline,
            forward_fn=forward_fn,
            steps=steps
        )
    )

    return attributions
```

Usage:

```python id="s8kqgi"
baseline = torch.zeros_like(
    data.x
)

attributions = (
    explain_node_features(
        model,
        data,
        baseline
    )
)
```

Then:

```python id="4o1l4b"
atom_importance = (
    attributions
    .abs()
    .sum(dim=1)
)
```

and:

```python id="yd8fap"
feature_importance = (
    attributions
    .abs()
    .mean(dim=0)
)
```

This gives both:

```text
atom-level importance
```

and:

```text
feature-level importance.
```

---

## 26.12.37 Building a Materials XAI Table

For practical research, explanations should be converted into structured records.

```python id="9gdysp"
records = []

for atom_id in range(
    len(structure)
):

    atom_attr = (
        attributions[
            atom_id
        ]
    )

    records.append({

        "atom_id":
            atom_id,

        "element":
            structure[
                atom_id
            ].species_string,

        "importance":
            atom_attr
            .abs()
            .sum()
            .item(),

        "signed_sum":
            atom_attr
            .sum()
            .item()
    })
```

Then:

```python id="qv31k2"
atom_df = pd.DataFrame(
    records
)

atom_df = (
    atom_df
    .sort_values(
        "importance",
        ascending=False
    )
)
```

This provides a direct bridge between numerical attribution and crystal structure.

---

## 26.12.38 Combining Integrated Gradients with GNNExplainer

The two methods answer complementary questions.

```text
GNNExplainer
      ↓
Which subgraph matters?

Integrated Gradients
      ↓
Which input features contribute?
```

A combined workflow is:

```text
Crystal
   ↓
GNN prediction
   ↓
GNNExplainer
   ↓
Important subgraph
   ↓
Integrated Gradients
   ↓
Important features inside subgraph
   ↓
Materials interpretation
```

For example:

```text
GNNExplainer:
Ti–O environment important

Integrated Gradients:
electronegativity and distance features important
```

Together, these findings are much more informative than either one alone.

---

## 26.12.39 Example of a Combined Interpretation

Suppose a model predicts formation energy.

GNNExplainer identifies:

```text
TiO₆ local environment
```

Integrated Gradients identifies:

```text
O electronegativity
Ti atomic representation
Ti–O distance
```

The resulting hypothesis might be:

> The model appears to rely strongly on Ti-centered oxygen coordination and on the chemical and geometric descriptors describing the Ti–O environment.

This is a defensible ML interpretation.

The next scientific step could be:

```text
DFT
 ↓
Ti–O bond analysis
 ↓
charge density
 ↓
electronic structure
```

Only after this validation should a stronger physical claim be made.

---

## 26.12.40 Common Failure Modes

Integrated Gradients has several practical failure modes.

### 1. Poor baseline selection

A meaningless baseline can produce a difficult-to-interpret explanation.

### 2. Insufficient integration steps

The approximation may be inaccurate.

### 3. Feature scaling problems

Features with incompatible scales can complicate attribution interpretation.

### 4. Ignoring attribution signs

Positive and negative effects may be incorrectly treated as equivalent.

### 5. Single-structure interpretation

One crystal may not represent the behavior of the entire dataset.

### 6. Ignoring symmetry

Equivalent sites may receive apparently different explanations because of representation or indexing.

### 7. Confusing attribution with causality

A high attribution does not prove a physical mechanism.

### 8. Ignoring graph construction

The explanation can only operate on the representation provided to the model.

---

## 26.12.41 Best Practices for Materials Research

A robust Integrated Gradients study should therefore include:

```text
1. Clearly defined baseline
2. Baseline sensitivity analysis
3. Adequate integration steps
4. Completeness check
5. Signed attribution
6. Absolute attribution where appropriate
7. Multiple structures
8. Multiple model seeds
9. Symmetry awareness
10. Chemical mapping
11. Independent scientific validation
```

A strong workflow is:

```text
Baseline selection
        ↓
Integrated Gradients
        ↓
Convergence check
        ↓
Completeness check
        ↓
Atom/feature mapping
        ↓
Dataset-level aggregation
        ↓
Cross-model validation
        ↓
Scientific interpretation
```

---

## 26.12.42 Key Takeaways

The central ideas introduced in this section are:

1. **Integrated Gradients attributes a model prediction to input features by integrating gradients along a path from a baseline to the actual input.**

2. **It differs from raw gradients because it considers the entire interpolation path rather than only the final input.**

3. **Baseline selection is one of the most important decisions in an Integrated Gradients analysis.**

4. **Zero, mean, and reference-material baselines can answer different scientific questions.**

5. **The completeness property provides an important consistency check.**

6. **Integrated Gradients can be applied to both node features and edge features in crystal GNNs.**

7. **Node-level attribution can reveal which atomic descriptors influence predictions.**

8. **Edge-level attribution can reveal the contribution of distance and other interaction descriptors.**

9. **Signed attribution distinguishes features that increase a prediction from features that suppress it.**

10. **Attributions should be aggregated carefully across atoms, elements, interaction types, and materials.**

11. **Crystal symmetry and periodicity must be considered when mapping numerical attributions back to structures.**

12. **Convergence and baseline-sensitivity tests improve the reliability of explanations.**

13. **Integrated Gradients and GNNExplainer are complementary rather than competing methods.**

14. **The strongest Materials Informatics workflow combines subgraph explanation with feature attribution and independent scientific validation.**

The progression is now:

```text
Node Importance
       ↓
Edge Importance
       ↓
GNNExplainer
       ↓
Important Subgraph
       ↓
Integrated Gradients
       ↓
Important Input Features
       ↓
Chemical + Structural Interpretation
```

The next step is to move from numerical feature attribution to a more visual form of neural-network interpretation: **saliency maps**, where attribution values are transformed into spatial or structural representations that can be directly inspected on a crystal.

# 26.13 Saliency Maps

Integrated Gradients provides a numerical attribution for input features. However, numerical attribution alone is not always the easiest way to understand a materials machine-learning model.

A researcher may want to see **where the model is focusing**.

For an image model, this can mean:

```text
Which pixels influence the prediction?
```

For a molecular model, it may mean:

```text
Which atoms or bonds influence the prediction?
```

For a crystal graph neural network, the corresponding question becomes:

> Which atoms, interactions, or structural regions of the crystal are most influential for the model prediction?

A **saliency map** converts model sensitivity or attribution into a representation that can be associated with the material's structure.

The conceptual workflow is:

```text
Crystal structure
       ↓
Crystal graph
       ↓
GNN
       ↓
Prediction
       ↓
Gradient / attribution
       ↓
Saliency values
       ↓
Map values back to crystal
       ↓
Visual interpretation
```

This makes saliency maps particularly useful for connecting a numerical neural-network model with a physical crystal structure.

---

## 26.13.1 What Is a Saliency Map?

A saliency map is a representation showing the relative importance or sensitivity of different input components to a model output.

For an input vector:

```text
x = [x₁, x₂, x₃, ..., xₙ]
```

a simple gradient-based saliency score is:

```text
Sᵢ = |∂F(x) / ∂xᵢ|
```

where:

```text
F(x)
```

is the model output.

The absolute value is often used because the basic saliency map is intended to represent **magnitude of sensitivity**.

However, this removes the direction of the effect.

Therefore, in scientific analysis it is often useful to preserve both:

```text
signed saliency
```

and:

```text
absolute saliency
```

---

## 26.13.2 Saliency Versus Integrated Gradients

Saliency and Integrated Gradients are closely related but not identical.

A basic saliency method asks:

```text
How sensitive is the prediction to the input at this point?
```

Integrated Gradients asks:

```text
How much accumulated sensitivity occurs along
the path from a baseline to the actual input?
```

Therefore:

```text
Saliency
    ↓
Local sensitivity

Integrated Gradients
    ↓
Path-integrated attribution
```

The distinction is important.

A saliency map can be very fast because it may require only one backward pass.

Integrated Gradients generally requires many backward passes.

---

## 26.13.3 Why Saliency Maps Matter for Crystal GNNs

Consider a crystal:

```text
A₁B₁O₃
```

with several atomic sites.

The GNN produces:

```text
Predicted band gap = 2.31 eV
```

A conventional prediction tells us only:

```text
material → 2.31 eV
```

A saliency analysis can produce:

```text
A-site     low importance
B-site     high importance
O-sites    moderate importance
B–O edges  very high importance
```

This changes the scientific interpretation.

Instead of only asking:

> What does the model predict?

we can ask:

> Which structural components appear to influence the prediction?

---

## 26.13.4 Node Saliency

Suppose the crystal graph contains:

```text
N
```

nodes.

Each node has feature vector:

```text
xᵢ
```

The model output is:

```text
F(G)
```

where `G` is the crystal graph.

A node-level saliency score can be obtained by measuring the gradient with respect to the node features.

For node `i`:

```text
Sᵢ = ||∂F / ∂xᵢ||
```

A common implementation is the L2 norm:

```python
node_saliency = (
    x.grad
    .norm(
        p=2,
        dim=1
    )
)
```

This produces one scalar per atom.

---

## 26.13.5 Basic PyTorch Saliency Calculation

Suppose:

```python
x = data.x.clone().detach()
```

We enable gradient tracking:

```python
x.requires_grad_(True)
```

Then perform a forward pass:

```python
output = model(
    x,
    data.edge_index,
    data.edge_attr,
    data.batch
)
```

Select the prediction:

```python
target = output.squeeze()
```

Then calculate the gradient:

```python
model.zero_grad()

target.backward()
```

The gradient is now available:

```python
gradients = x.grad
```

The shape is:

```text
[number_of_atoms, number_of_features]
```

We can aggregate across features:

```python
node_saliency = (
    gradients
    .abs()
    .sum(dim=1)
)
```

or:

```python
node_saliency = (
    gradients
    .norm(
        p=2,
        dim=1
    )
)
```

---

## 26.13.6 Complete Node-Saliency Function

A reusable implementation is:

```python
import torch


def node_saliency(
    model,
    data
):

    model.eval()

    x = (
        data.x
        .clone()
        .detach()
        .requires_grad_(True)
    )

    model.zero_grad()

    output = model(
        x,
        data.edge_index,
        data.edge_attr,
        data.batch
    )

    output = output.squeeze()

    output.backward()

    gradients = (
        x.grad.detach()
    )

    saliency = (
        gradients
        .norm(
            p=2,
            dim=1
        )
    )

    return saliency
```

Usage:

```python
saliency = node_saliency(
    model,
    data
)

print(saliency)
```

For a crystal containing 20 atoms, the output may look like:

```text
tensor([
    0.031,
    0.182,
    0.054,
    0.713,
    0.641,
    ...
])
```

Each value corresponds to an atomic node.

---

## 26.13.7 Mapping Saliency to Atomic Sites

The numerical values become scientifically meaningful only when mapped back to the crystal.

For example:

```python
for i, site in enumerate(
    structure
):

    print(
        i,
        site.species_string,
        saliency[i].item()
    )
```

Output:

```text
0 Ti 0.713
1 O  0.641
2 O  0.598
3 O  0.102
4 O  0.091
```

This tells us that the model is particularly sensitive to some atomic sites.

However, the researcher should not immediately conclude:

> Ti is physically responsible for the property.

The correct interpretation is:

> The model prediction is particularly sensitive to the input representation associated with these sites.

That distinction is fundamental to responsible XAI.

---

## 26.13.8 Normalizing Saliency

Raw saliency values can be difficult to compare.

A simple normalization is:

```python
saliency_norm = (
    saliency
    /
    saliency.max()
)
```

This produces values between approximately:

```text
0 → low importance
1 → highest importance
```

A safer implementation is:

```python
def normalize_scores(scores):

    minimum = scores.min()
    maximum = scores.max()

    return (
        (scores - minimum)
        /
        (maximum - minimum + 1e-12)
    )
```

Then:

```python
normalized = normalize_scores(
    saliency
)
```

---

## 26.13.9 Why Normalization Should Be Reported

Suppose two crystals produce:

```text
Crystal A:
max saliency = 0.8

Crystal B:
max saliency = 5.7
```

After independent normalization, both will have:

```text
maximum = 1
```

This is useful for visualizing patterns within each crystal.

But it destroys information about absolute magnitude.

Therefore:

```text
normalized saliency
```

is useful for visualization, while:

```text
raw saliency
```

may be useful for quantitative comparison.

Both should be retained when possible.

---

## 26.13.10 Feature-Level Saliency

Node-level saliency can hide which feature caused the sensitivity.

Suppose:

```text
x.shape
=
[atoms, features]
```

The gradient is:

```text
gradients.shape
=
[atoms, features]
```

We can inspect individual features:

```python
feature_saliency = (
    gradients.abs()
)
```

For atom `i`:

```python
atom_i = feature_saliency[i]
```

This might produce:

```text
atomic_number       0.72
electronegativity   0.41
atomic_radius       0.09
valence             0.33
```

Now the explanation becomes more detailed.

---

## 26.13.11 Signed Saliency

Instead of:

```python
gradients.abs()
```

we can preserve the gradient:

```python
signed = gradients
```

For example:

```text
atomic number       +0.72
electronegativity   -0.41
atomic radius       +0.09
valence             -0.33
```

This provides directional information.

For a regression target such as:

```text
band gap
```

a positive gradient indicates that locally increasing the corresponding input feature would tend to increase the model output, while a negative gradient indicates the opposite local tendency.

This interpretation must be made carefully because neural networks are nonlinear.

---

## 26.13.12 Saliency for Classification

Saliency is also applicable to classification.

Suppose a model predicts:

```text
stable
unstable
```

and the output is a class logit:

```python
logit = output[
    0,
    target_class
]
```

We can calculate:

```python
model.zero_grad()

logit.backward()
```

Then:

```python
saliency = (
    data.x.grad
    .abs()
    .norm(
        p=2,
        dim=1
    )
)
```

The resulting map represents sensitivity toward that specific class.

This is important because the same material may have different explanations for different target classes.

---

## 26.13.13 Choosing the Target Output

For multi-output materials models:

```text
band gap
formation energy
bulk modulus
thermal conductivity
```

the saliency calculation must specify which output is being explained.

For example:

```python
band_gap = output[:, 0]

model.zero_grad()

band_gap.sum().backward()
```

For formation energy:

```python
formation_energy = output[:, 1]

model.zero_grad()

formation_energy.sum().backward()
```

The saliency maps can therefore be different:

```text
Crystal
   ↓
GNN
   ├── Band gap
   │      ↓
   │   Saliency A
   │
   └── Formation energy
          ↓
       Saliency B
```

This allows property-specific interpretation.

---

## 26.13.14 Property-Specific Structural Importance

Suppose the same crystal is analyzed for:

```text
band gap
```

and:

```text
formation energy
```

The model may produce:

```text
Band gap:
B-site → high
B–O distance → high

Formation energy:
O coordination → high
A-site → moderate
```

This demonstrates that:

> Different material properties may be predicted using different structural information.

This is scientifically useful because it can suggest that different properties are associated with different regions of the learned representation.

---

## 26.13.15 Edge Saliency

Crystal GNNs do not only contain nodes.

They also contain edges representing interactions.

An edge may encode:

```text
atom pair
distance
radial basis
direction
periodic translation
```

We can therefore calculate sensitivity with respect to edge features.

Suppose:

```python
edge_attr = (
    data.edge_attr
    .clone()
    .detach()
    .requires_grad_(True)
)
```

Then:

```python
output = model(
    data.x,
    data.edge_index,
    edge_attr,
    data.batch
)

target = output.squeeze()

model.zero_grad()

target.backward()
```

Now:

```python
edge_grad = (
    edge_attr.grad
)
```

and:

```python
edge_saliency = (
    edge_grad
    .norm(
        p=2,
        dim=1
    )
)
```

Each edge receives a saliency score.

---

## 26.13.16 Edge Saliency and Interatomic Distance

Suppose the edge representation is generated from:

```text
dᵢⱼ
```

the interatomic distance.

A strong edge saliency may indicate that the model prediction is sensitive to the representation of that interaction.

For example:

```text
Edge       Distance     Saliency
Ti–O₁       1.87 Å       0.82
Ti–O₂       1.91 Å       0.77
Ti–O₃       2.03 Å       0.31
O–O         2.78 Å       0.04
```

This may suggest that short Ti–O interactions are important to the model.

But the correct scientific workflow is:

```text
Model observation
       ↓
Hypothesis
       ↓
Structural analysis
       ↓
DFT / literature / experiment
       ↓
Physical validation
```

The saliency map itself is not proof of the mechanism.

---

## 26.13.17 Visualizing a Crystal Saliency Map

A useful crystal saliency map should preserve the spatial arrangement of the material.

Conceptually:

```text
Low saliency
     ○
     
Medium saliency
     ◐

High saliency
     ●
```

The crystal can be represented as:

```text
atoms → node size / intensity
bonds → edge width / intensity
```

For example:

```text
                O
                ●
                │
        O ───── Ti ───── O
        ○               ●
                │
                O
                ○
```

where the most influential sites are emphasized.

The important point is that the visualization should preserve crystallographic context.

---

## 26.13.18 Generating a Structural Saliency Table

Before creating a visualization, it is useful to generate a structured table.

```python
import pandas as pd


def crystal_saliency_table(
    structure,
    saliency
):

    records = []

    for i, site in enumerate(
        structure
    ):

        records.append({

            "site_index":
                i,

            "element":
                site.species_string,

            "x":
                site.coords[0],

            "y":
                site.coords[1],

            "z":
                site.coords[2],

            "saliency":
                saliency[i].item()
        })

    return pd.DataFrame(
        records
    )
```

Usage:

```python
saliency_df = (
    crystal_saliency_table(
        structure,
        saliency
    )
)

print(
    saliency_df
)
```

This produces a table such as:

```text
site_index  element      x       y       z    saliency
0           Ti         0.00    0.00    0.00   0.91
1           O          1.94    0.00    0.00   0.73
2           O          0.00    1.94    0.00   0.68
3           O          0.00    0.00    1.95   0.19
```

This table is the foundation for reproducible visualization.

---

## 26.13.19 Mapping Saliency to a Pymatgen Structure

Because the user is assumed to already know Pymatgen, it can be used here as the bridge between model output and crystal geometry.

```python
from pymatgen.core import Structure


structure = Structure.from_file(
    "POSCAR"
)
```

Suppose:

```python
saliency = node_saliency(
    model,
    data
)
```

The values can be attached to the corresponding sites:

```python
site_information = []

for i, site in enumerate(
    structure
):

    site_information.append({
        "index": i,
        "species":
            site.species_string,
        "frac_coords":
            site.frac_coords.tolist(),
        "cart_coords":
            site.coords.tolist(),
        "saliency":
            float(
                saliency[i]
            )
    })
```

Now every atomic site has both:

```text
crystallographic information
```

and:

```text
model-derived importance
```

---

## 26.13.20 Saliency Distribution Across a Crystal

Before visualization, inspect the distribution.

```python
import numpy as np

values = (
    saliency
    .detach()
    .cpu()
    .numpy()
)

print(
    "Minimum:",
    values.min()
)

print(
    "Maximum:",
    values.max()
)

print(
    "Mean:",
    values.mean()
)

print(
    "Median:",
    np.median(values)
)
```

This helps determine whether the model is:

```text
focused on a few atoms
```

or:

```text
distributed across the entire crystal.
```

For example:

```text
Case A:

● ● ● ○ ○ ○ ○

localized explanation
```

versus:

```text
Case B:

● ● ● ● ● ● ●

distributed explanation
```

These patterns can lead to very different scientific interpretations.

---

## 26.13.21 Sparse Versus Distributed Saliency

A highly concentrated saliency pattern may indicate:

```text
localized structural control
```

while a broadly distributed pattern may indicate:

```text
global structural dependence
```

For example, a defect-related property may produce:

```text
defect site
   ↓
high saliency
```

whereas a bulk thermodynamic property may involve:

```text
many atoms
   ↓
distributed saliency
```

However, this interpretation must be tested against the model architecture and dataset.

---

## 26.13.22 Saliency Thresholding

For visualization, one may identify the most important atoms.

For example:

```python
threshold = torch.quantile(
    saliency,
    0.90
)
```

Then:

```python
important_atoms = (
    saliency >= threshold
)
```

This selects approximately the top 10% of atoms.

Alternatively:

```python
top_k = 5

values, indices = torch.topk(
    saliency,
    k=top_k
)
```

Then:

```python
for score, index in zip(
    values,
    indices
):

    print(
        index.item(),
        score.item()
    )
```

This gives the most salient atomic sites.

---

## 26.13.23 Why Thresholding Can Be Dangerous

A threshold can create a visually attractive explanation while hiding important information.

For example:

```text
original:
0.91 0.88 0.85 0.83 0.81 0.79 ...
```

A top-10% threshold may display only:

```text
● ● ●
```

This can falsely suggest that only three atoms matter.

The remaining atoms may still have substantial contributions.

Therefore, a continuous saliency scale is generally preferable for quantitative analysis.

Thresholded maps should be treated as a visualization aid rather than the complete explanation.

---

## 26.13.24 Saliency Maps and Crystal Symmetry

As with Integrated Gradients, symmetry must be considered.

Suppose a crystal contains six symmetry-equivalent oxygen sites.

A saliency calculation produces:

```text
O₁ → 0.71
O₂ → 0.69
O₃ → 0.72
O₄ → 0.14
O₅ → 0.17
O₆ → 0.16
```

If all six sites are expected to be symmetry-equivalent, this requires investigation.

Possible explanations include:

```text
graph representation
periodic-neighbor construction
node ordering
numerical optimization
model architecture
```

A useful analysis therefore groups sites according to structural equivalence where appropriate.

---

## 26.13.25 Saliency and Periodic Boundary Conditions

A crystal graph may contain edges crossing the unit-cell boundary.

For example:

```text
Atom i
  │
  └──── Atom j + T
```

where `T` is a lattice translation.

A visualization that ignores the periodic translation may draw the interaction incorrectly.

Therefore, a structural saliency implementation should retain:

```text
source atom
target atom
periodic translation
edge saliency
distance
```

rather than storing only:

```text
source
target
```

This is particularly important when interpreting edge-level saliency.

---

## 26.13.26 Saliency Maps for Defect Structures

Saliency maps become especially interesting for defect materials.

Consider:

```text
perfect crystal
        ↓
vacancy
        ↓
defect crystal
```

Suppose a vacancy is introduced.

The model may identify:

```text
vacancy-neighbor atoms
        ↓
high saliency
```

A researcher can therefore investigate whether the model is relying strongly on the local defect environment.

The workflow becomes:

```text
Defect structure
      ↓
GNN prediction
      ↓
Saliency map
      ↓
High-importance neighborhood
      ↓
Local structural analysis
      ↓
Electronic/energetic validation
```

This is a particularly useful application of XAI in materials research.

---

## 26.13.27 Saliency Maps for Grain or Interface Models

The same idea extends beyond isolated periodic crystals.

For an interface model:

```text
Material A
────────────
Interface
────────────
Material B
```

saliency may concentrate around:

```text
interface atoms
```

rather than bulk atoms.

This can provide a useful hypothesis:

> The model prediction may depend strongly on the local structural environment near the interface.

The same principle can be applied to:

```text
grain boundaries
surfaces
heterostructures
defects
dopants
dislocations
```

provided the graph representation contains the relevant structural information.

---

## 26.13.28 Saliency Maps for Dopants

Consider:

```text
Host crystal
+
dopant
```

For example:

```text
TiO₂
 ↓
Fe-doped TiO₂
```

The researcher can compare:

```text
saliency(host)
```

with:

```text
saliency(doped)
```

A difference map can be constructed.

Conceptually:

```text
ΔSᵢ
=
Sᵢ(doped)
-
Sᵢ(host)
```

This can highlight regions where the model's sensitivity changes following chemical substitution.

However, comparing structures with different atom counts requires careful alignment of sites or local environments.

---

## 26.13.29 Saliency Maps Across a Materials Series

A single crystal provides one explanation.

A materials series provides a trend.

Suppose:

```text
La₁₋ₓSrₓMnO₃
```

is studied across several compositions.

For each material:

```python
saliency = node_saliency(
    model,
    data
)
```

Store the results:

```python
results.append({

    "composition": composition,
    "saliency": saliency
})
```

The researcher can then investigate:

```text
composition
      ↓
prediction
      ↓
saliency distribution
```

This can reveal whether the model's focus changes as composition changes.

---

## 26.13.30 Saliency Stability Across Model Seeds

Neural networks can produce different learned representations depending on initialization.

Therefore:

```text
Seed 1 → saliency A
Seed 2 → saliency B
Seed 3 → saliency C
```

should not automatically be interpreted as three independent scientific conclusions.

A stronger approach trains several models.

```python
seed_results = []

for seed in [0, 1, 2, 3, 4]:

    set_seed(seed)

    model = train_model(
        train_dataset
    )

    saliency = node_saliency(
        model,
        data
    )

    seed_results.append(
        saliency
    )
```

Then calculate:

```python
stacked = torch.stack(
    seed_results
)

mean_saliency = (
    stacked.mean(dim=0)
)

std_saliency = (
    stacked.std(dim=0)
)
```

This provides:

```text
mean importance
+
importance variability
```

which is much more informative than a single model explanation.

---

## 26.13.31 Saliency Stability as an XAI Metric

Suppose:

```text
Atom A:
mean saliency = 0.81
std = 0.03
```

and:

```text
Atom B:
mean saliency = 0.77
std = 0.41
```

Both may have high mean saliency.

But Atom A is much more stable across model runs.

Therefore, the explanation should report both:

```text
importance
```

and:

```text
stability
```

when possible.

This is especially important for research publications.

---

## 26.13.32 Comparing Saliency with Integrated Gradients

A useful validation strategy is:

```text
Raw Saliency
      +
Integrated Gradients
      +
GNNExplainer
```

Suppose all three methods identify:

```text
B-site
B–O interactions
local coordination
```

as important.

The explanation becomes more convincing.

If they disagree completely:

```text
Saliency → A-site
IG       → B-site
GNNExp   → O-network
```

the disagreement should be investigated rather than ignored.

Possible causes include:

```text
different attribution definitions
baseline dependence
nonlinear model behavior
graph masking
feature scaling
```

---

## 26.13.33 Quantifying Agreement Between Explanations

Suppose we have two importance vectors:

```python
saliency_scores
ig_scores
```

They can be compared using rank correlation.

```python
from scipy.stats import spearmanr

rho, p_value = spearmanr(
    saliency_scores
    .detach()
    .cpu()
    .numpy(),

    ig_scores
    .detach()
    .cpu()
    .numpy()
)

print(
    "Spearman correlation:",
    rho
)

print(
    "p-value:",
    p_value
)
```

A high positive correlation indicates that both methods tend to rank components similarly.

This does not prove that either method is physically correct, but it provides evidence of explanation consistency.

---

## 26.13.34 A Complete Saliency Analysis Pipeline

A practical workflow can therefore be organized as:

```text
Crystal dataset
      ↓
Train GNN
      ↓
Evaluate predictive performance
      ↓
Select material
      ↓
Select target property
      ↓
Compute gradients
      ↓
Calculate node saliency
      ↓
Calculate edge saliency
      ↓
Normalize for visualization
      ↓
Map scores to crystal structure
      ↓
Analyze important atoms
      ↓
Analyze important interactions
      ↓
Compare with Integrated Gradients
      ↓
Compare across model seeds
      ↓
Form scientific hypothesis
      ↓
Validate independently
```

This makes saliency analysis part of the research workflow rather than an isolated visualization.

---

## 26.13.35 Reusable Saliency Analysis Class

For repeated experiments, it is useful to create a small XAI utility.

```python
class CrystalSaliency:

    def __init__(
        self,
        model
    ):

        self.model = model

        self.model.eval()


    def explain_nodes(
        self,
        data
    ):

        x = (
            data.x
            .clone()
            .detach()
            .requires_grad_(True)
        )

        self.model.zero_grad()

        output = self.model(
            x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        output = output.squeeze()

        output.backward()

        gradients = (
            x.grad.detach()
        )

        scores = (
            gradients
            .norm(
                p=2,
                dim=1
            )
        )

        return scores


    def explain_edges(
        self,
        data
    ):

        edge_attr = (
            data.edge_attr
            .clone()
            .detach()
            .requires_grad_(True)
        )

        self.model.zero_grad()

        output = self.model(
            data.x,
            data.edge_index,
            edge_attr,
            data.batch
        )

        output = output.squeeze()

        output.backward()

        gradients = (
            edge_attr
            .grad
            .detach()
        )

        scores = (
            gradients
            .norm(
                p=2,
                dim=1
            )
        )

        return scores
```

Usage:

```python
explainer = CrystalSaliency(
    model
)

node_scores = (
    explainer.explain_nodes(
        data
    )
)

edge_scores = (
    explainer.explain_edges(
        data
    )
)
```

This provides a simple reusable interface for experiments.

---

## 26.13.36 Important Implementation Consideration: Graph Batches

In practical PyTorch Geometric workflows, several crystals may be processed simultaneously.

For example:

```text
Batch
 ├── Crystal 1
 ├── Crystal 2
 ├── Crystal 3
 └── Crystal 4
```

The node matrix contains nodes from all crystals.

Therefore, node saliency must be associated with the correct graph.

The `batch` vector provides this mapping.

For example:

```python
graph_id = data.batch
```

A researcher can separate explanations:

```python
for graph_index in range(
    graph_id.max().item() + 1
):

    mask = (
        graph_id == graph_index
    )

    graph_saliency = (
        node_scores[mask]
    )
```

This prevents attribution from one crystal being incorrectly assigned to another.

---

## 26.13.37 Batch-Level Saliency Aggregation

Suppose multiple materials belong to the same chemical family.

Their saliency maps can be aggregated.

For example:

```python
family_scores = []

for data in family_dataset:

    scores = (
        explainer.explain_nodes(
            data
        )
    )

    family_scores.append(
        scores.mean().item()
    )
```

The resulting distribution can be compared across families.

This allows XAI to move from:

```text
single-material explanation
```

toward:

```text
materials-family explanation
```

---

## 26.13.38 Saliency Is Not a Physical Observable

This distinction must remain explicit.

A saliency value such as:

```text
0.91
```

does not correspond automatically to:

```text
0.91 eV
0.91 eV/atom
0.91 Å
0.91 electrons
```

It is a property of the model's sensitivity to its representation.

Therefore:

```text
Saliency
≠
physical quantity
```

Instead:

```text
Saliency
=
model-dependent explanatory signal
```

This distinction prevents a common misuse of explainable AI.

---

## 26.13.39 From Saliency to Scientific Hypothesis

The correct interpretation pipeline is:

```text
Saliency
   ↓
Model behavior
   ↓
Structural pattern
   ↓
Scientific hypothesis
   ↓
Independent validation
```

For example:

```text
Saliency:
short B–O edges highly important
```

should lead to:

```text
Hypothesis:
the model relies strongly on local B–O coordination
```

rather than immediately claiming:

```text
B–O bonding causes the property.
```

The latter requires independent evidence.

Possible validation methods include:

```text
DFT electronic structure
charge density
bond-order analysis
phonon calculations
formation-energy calculations
experimental characterization
literature comparison
```

---

## 26.13.40 Key Takeaways

The central ideas introduced in this section are:

1. **Saliency maps transform model sensitivity into a form that can be associated with structural components of a material.**

2. **Basic saliency can be calculated from gradients of the model output with respect to the input representation.**

3. **Node saliency provides atom-level importance scores in crystal GNNs.**

4. **Feature-level saliency reveals which atomic descriptors contribute to local sensitivity.**

5. **Edge saliency can identify important interactions and distance-related features.**

6. **Saliency can be calculated for different target properties independently.**

7. **Mapping saliency back to Pymatgen structures makes the explanation structurally interpretable.**

8. **Periodic boundary conditions and symmetry must be preserved when interpreting crystal-level explanations.**

9. **Saliency distributions can reveal localized or distributed model dependence.**

10. **Defects, dopants, surfaces, interfaces, and grain boundaries provide particularly useful applications for structural saliency analysis.**

11. **Saliency should be evaluated across multiple model seeds when possible.**

12. **Agreement between saliency, Integrated Gradients, and GNNExplainer can strengthen confidence in an interpretation.**

13. **Saliency is model-dependent and should never automatically be interpreted as a physical observable.**

14. **The final goal of saliency analysis is not merely visualization; it is to generate scientifically testable hypotheses.**

The progression of Chapter 26 is now:

```text
Feature Attribution
        ↓
Attention Visualization
        ↓
GNNExplainer
        ↓
Integrated Gradients
        ↓
Saliency Maps
        ↓
Node Importance
        ↓
Edge Importance
        ↓
Scientific Interpretation
        ↓
Interpreting Crystal Graphs
```

Saliency therefore provides the visual and structural bridge between numerical attribution and the actual crystal.

The next section develops **node importance** more explicitly, moving from general saliency maps to systematic identification, ranking, aggregation, and comparison of the atoms that a crystal GNN relies upon most strongly.

# 26.14 Node Importance in Crystal Graph Neural Networks

Saliency maps provide a general mechanism for estimating which parts of an input influence a model prediction.

For crystal graph neural networks, one of the most natural forms of explanation is **node importance**.

In a crystal graph:

```text
Atoms
  ↓
Nodes
```

Therefore, node importance asks:

> Which atomic sites contribute most strongly to the prediction made by the crystal GNN?

This question is particularly useful in Materials Informatics because many material properties depend strongly on local atomic environments.

For example:

```text
Crystal
   ↓
Atomic species
   ↓
Local coordination
   ↓
Bonding environment
   ↓
Electronic structure
   ↓
Material property
```

A GNN does not explicitly perform this physical reasoning in the same form as a materials scientist.

Instead, it learns statistical relationships from the training data.

Node-importance analysis provides a way to inspect which atomic representations the learned model is using.

---

## 26.14.1 From Saliency to Node Importance

A saliency map may produce a value for every atom:

```text
Atom 0 → 0.12
Atom 1 → 0.83
Atom 2 → 0.17
Atom 3 → 0.91
Atom 4 → 0.74
```

We can interpret these as relative model sensitivity:

```text
low importance
      ↓
Atom 0

high importance
      ↓
Atom 3
```

However, raw saliency values should not immediately be interpreted as physical importance.

A more precise terminology is:

```text
model-derived node importance
```

rather than:

```text
physical importance of the atom
```

The distinction matters because the model may rely on:

```text
atomic features
+
neighbor relationships
+
graph connectivity
+
training-data correlations
```

rather than directly representing a fundamental physical mechanism.

---

## 26.14.2 Formal Definition

Let a crystal graph be:

```text
G = (V, E)
```

where:

```text
V = {v₁, v₂, ..., vₙ}
```

represents atoms and:

```text
E
```

represents interactions between atoms.

Each node has feature vector:

```text
xᵢ
```

The GNN predicts:

```text
ŷ = f(G)
```

A gradient-based node-importance score can be defined as:

```text
Iᵢ = || ∂ŷ / ∂xᵢ ||
```

where `Iᵢ` measures the sensitivity of the output to the features of node `i`.

This gives:

```text
Crystal
   ↓
Node features
   ↓
GNN
   ↓
Prediction
   ↓
∂Prediction / ∂Node Features
   ↓
Node importance
```

---

## 26.14.3 Node Importance Is Feature-Dependent

A critical point is that an atom itself is not necessarily the complete unit being interpreted.

Suppose a node contains:

```text
xᵢ =
[
atomic number,
electronegativity,
atomic radius,
valence,
...
]
```

Then:

```text
∂ŷ / ∂xᵢ
```

contains one derivative for every feature.

Therefore, node importance is normally obtained by aggregating feature-level attribution.

For example:

```python
node_importance = (
    gradients
    .abs()
    .norm(
        p=2,
        dim=1
    )
)
```

This transforms:

```text
[atoms, features]
```

into:

```text
[atoms]
```

The resulting vector provides one importance score per atomic node.

---

## 26.14.4 Different Aggregation Strategies

There is no single mandatory way to aggregate feature-level attribution.

Common choices include:

### L1 aggregation

```python
importance = (
    gradients
    .abs()
    .sum(dim=1)
)
```

This measures the total absolute sensitivity.

### L2 aggregation

```python
importance = (
    gradients
    .norm(
        p=2,
        dim=1
    )
)
```

This emphasizes larger feature-level contributions.

### Maximum-feature aggregation

```python
importance = (
    gradients
    .abs()
    .max(dim=1)
    .values
)
```

This asks whether at least one feature is highly influential.

These approaches can produce different rankings.

Therefore, the aggregation method should be reported in a research implementation.

---

## 26.14.5 Signed Node Importance

Absolute attribution removes direction.

For example:

```text
+0.8
-0.8
```

both become:

```text
0.8
```

when absolute values are used.

However, the sign can contain useful information.

A signed aggregation can be defined as:

```python
signed_importance = (
    gradients
    .sum(dim=1)
)
```

For example:

```text
Ti₁ → +0.72
O₁  → -0.41
O₂  → +0.13
```

This indicates different local effects on the model output.

However, signed aggregation should be used carefully because positive and negative feature contributions can cancel.

For example:

```text
[+0.8, -0.7]
```

would produce:

```text
+0.1
```

even though both features individually have substantial sensitivity.

Therefore, signed and magnitude-based importance should often be retained separately.

---

## 26.14.6 Ranking Atomic Sites

Once node importance has been calculated, the atomic sites can be ranked.

```python
ranking = torch.argsort(
    node_importance,
    descending=True
)
```

Then:

```python
for index in ranking:

    print(
        index.item(),
        node_importance[
            index
        ].item()
    )
```

A result might look like:

```text
Site 17 → 0.941
Site 4  → 0.903
Site 11 → 0.872
Site 2  → 0.841
Site 8  → 0.799
```

This produces a ranked list of the most influential sites.

The ranking can then be connected to crystallographic information.

---

## 26.14.7 Combining Importance with Atomic Identity

A ranking based only on indices is not scientifically useful.

Instead:

```python
for index in ranking:

    i = index.item()

    print(
        i,
        structure[i].species_string,
        node_importance[i].item()
    )
```

Possible output:

```text
17  Fe  0.941
4   O   0.903
11  O   0.872
2   Fe  0.841
8   O   0.799
```

Now the researcher can ask:

```text
Are Fe sites systematically important?

Are oxygen sites systematically important?

Are only particular sites important?

Are symmetry-equivalent sites treated similarly?
```

These are more meaningful scientific questions.

---

## 26.14.8 Importance by Element

A simple first-level analysis is to aggregate node importance by chemical species.

Suppose:

```text
Fe → [0.94, 0.84, 0.31]
O  → [0.90, 0.87, 0.80, 0.22, 0.19]
```

The mean importance can be calculated:

```python
from collections import defaultdict


element_scores = defaultdict(list)

for i, site in enumerate(structure):

    element = (
        site.species_string
    )

    element_scores[
        element
    ].append(
        node_importance[i].item()
    )
```

Then:

```python
for element, scores in (
    element_scores.items()
):

    mean_score = (
        sum(scores)
        /
        len(scores)
    )

    print(
        element,
        mean_score
    )
```

This gives a coarse chemical interpretation.

---

## 26.14.9 Why Element-Level Averaging Can Mislead

Suppose:

```text
Fe₁ → 0.95
Fe₂ → 0.12
Fe₃ → 0.10
```

The average is:

```text
0.39
```

This may make Fe appear moderately important.

But the actual pattern is:

```text
one Fe site → extremely important
two Fe sites → relatively unimportant
```

Therefore, element-level averaging can hide site-specific structural information.

A better analysis often reports both:

```text
element-level statistics
```

and:

```text
site-level statistics.
```

---

## 26.14.10 Site Importance Versus Element Importance

These concepts should be distinguished.

### Element importance

Asks:

```text
How important is this chemical species on average?
```

### Site importance

Asks:

```text
How important is this particular atomic position?
```

### Environment importance

Asks:

```text
How important is the local atomic environment around this site?
```

These are increasingly structural interpretations.

For crystal GNNs, site and environment importance are often more informative than simply ranking elements.

---

## 26.14.11 Local Environment Importance

A crystal GNN does not generally make predictions from isolated atoms.

Message passing allows information to move between neighboring atoms.

A node representation after one layer may be:

```text
hᵢ^(1)
=
UPDATE(
    hᵢ^(0),
    AGGREGATE(
        hⱼ^(0)
    )
)
```

After several layers:

```text
hᵢ^(L)
```

may contain information from a larger local neighborhood.

Therefore, high node importance can reflect not only:

```text
atom i
```

but also:

```text
local environment around atom i.
```

This is one reason why node importance should not automatically be interpreted as the atom acting independently.

---

## 26.14.12 Neighborhood-Based Importance

A useful extension is to define the importance of a local neighborhood.

For atom `i`, define:

```text
N(i)
```

as its neighbors.

A simple neighborhood score is:

```text
I_N(i)
=
Iᵢ
+
Σ Iⱼ
```

for:

```text
j ∈ N(i)
```

In Python:

```python
def neighborhood_importance(
    node_importance,
    edge_index
):

    scores = torch.zeros_like(
        node_importance
    )

    src = edge_index[0]
    dst = edge_index[1]

    for s, d in zip(src, dst):

        scores[s] += (
            node_importance[d]
        )

    scores += node_importance

    return scores
```

This provides a rough measure of how much importance is concentrated around each site.

---

## 26.14.13 Avoiding Double Counting in Periodic Graphs

Crystal graphs often contain multiple edges representing periodic images.

For example:

```text
A → B
A → B + T₁
A → B + T₂
```

These may correspond to physically equivalent interactions under periodicity.

Therefore, simply summing every graph edge can overcount some environments.

A research-grade implementation should retain periodic image information and decide whether importance is being measured for:

```text
graph edges
```

or:

```text
unique physical interactions.
```

The choice should be explicitly documented.

---

## 26.14.14 Symmetry-Aware Node Importance

Crystal symmetry provides another opportunity for systematic analysis.

Suppose:

```text
A₁
A₂
A₃
A₄
```

are symmetry-equivalent sites.

If the model is physically expected to be invariant under those symmetry operations, their importance should often exhibit corresponding behavior, subject to the details of the representation and numerical implementation.

A symmetry-aware analysis can therefore group equivalent sites:

```python
symmetry_groups = {
    0: [0, 4, 8],
    1: [1, 5, 9],
}
```

Then calculate:

```python
for group, indices in (
    symmetry_groups.items()
):

    values = (
        node_importance[
            indices
        ]
    )

    print(
        group,
        values.mean().item(),
        values.std().item()
    )
```

This allows the researcher to quantify whether importance is consistent within structural equivalence classes.

---

## 26.14.15 Importance Variance Within Symmetry Classes

Consider:

```text
Symmetry class A:
0.81
0.80
0.82
0.79
```

The model is highly consistent.

Now consider:

```text
Symmetry class B:
0.91
0.42
0.13
0.87
```

This is much less consistent.

A useful statistic is:

```text
importance variance
```

within each symmetry class.

High variance may indicate:

```text
representation asymmetry
```

or:

```text
model sensitivity to graph construction
```

or:

```text
numerical instability.
```

This should be investigated rather than immediately interpreted as physical asymmetry.

---

## 26.14.16 Node Importance Across Multiple Models

A single model provides only one learned explanation.

For robust research, train several models:

```python
models = []

for seed in range(5):

    set_seed(seed)

    model = train_model(
        dataset
    )

    models.append(
        model
    )
```

Calculate importance for each:

```python
all_scores = []

for model in models:

    explainer = CrystalSaliency(
        model
    )

    scores = (
        explainer.explain_nodes(
            data
        )
    )

    all_scores.append(
        scores
    )
```

Then:

```python
scores = torch.stack(
    all_scores
)

mean_scores = (
    scores.mean(dim=0)
)

std_scores = (
    scores.std(dim=0)
)
```

This creates a more robust importance estimate.

---

## 26.14.17 Consensus Node Importance

The mean importance can be used as a consensus estimate:

```text
Īᵢ
=
1/M
Σ Iᵢ^(m)
```

where `M` is the number of trained models.

In code:

```python
mean_importance = (
    scores.mean(
        dim=0
    )
)
```

The corresponding uncertainty is:

```python
importance_uncertainty = (
    scores.std(
        dim=0
    )
)
```

The result is:

```text
Atom
 ↓
Mean importance
+
Importance uncertainty
```

This is substantially more informative than a single importance number.

---

## 26.14.18 Ranking with Importance Uncertainty

Suppose:

```text
Atom A:
mean = 0.90
std  = 0.03

Atom B:
mean = 0.94
std  = 0.31
```

Atom B has a higher mean but much greater variability.

A researcher may therefore prefer to report:

```text
Atom A:
stable high importance

Atom B:
potentially high importance,
but unstable across models
```

This distinction is important when converting XAI results into scientific claims.

---

## 26.14.19 Top-k Node Analysis

A common research analysis is to inspect the top `k` nodes.

```python
k = 10

top_values, top_indices = (
    torch.topk(
        mean_importance,
        k=k
    )
)
```

Then:

```python
for score, index in zip(
    top_values,
    top_indices
):

    i = index.item()

    print(
        f"Site {i}: "
        f"{structure[i].species_string} "
        f"{score.item():.4f}"
    )
```

The resulting list can be used for structural investigation.

However, `k` should not be chosen solely because it produces an attractive visualization.

It should be motivated by the analysis.

---

## 26.14.20 Cumulative Importance

Instead of selecting an arbitrary top `k`, one can study cumulative importance.

Sort:

```python
sorted_scores, indices = (
    torch.sort(
        mean_importance,
        descending=True
    )
)
```

Calculate:

```python
cumulative = (
    torch.cumsum(
        sorted_scores,
        dim=0
    )
)
```

Normalize:

```python
cumulative_fraction = (
    cumulative
    /
    sorted_scores.sum()
)
```

Now the researcher can ask:

```text
How many atoms account for 50%
of total attribution?

How many account for 80%?

How many account for 90%?
```

This provides a more principled description of explanation concentration.

---

## 26.14.21 Explanation Concentration

Define:

```text
K₅₀
```

as the minimum number of nodes required to account for 50% of total attribution.

Similarly:

```text
K₉₀
```

represents the number required for 90%.

A small `K₅₀` indicates a highly concentrated explanation.

A large `K₅₀` indicates a more distributed explanation.

This can be useful when comparing different materials or model architectures.

---

## 26.14.22 Comparing Materials Through Node Importance

Suppose two materials have:

```text
Material A:
K₅₀ = 3

Material B:
K₅₀ = 18
```

The model explanation for Material A is much more localized.

This may indicate:

```text
localized structural feature
```

while Material B may rely on:

```text
distributed crystal information.
```

Again, this is a model interpretation rather than direct physical proof.

---

## 26.14.23 Element-Conditioned Importance Distributions

Rather than calculating only means, preserve the full distribution.

```python
import pandas as pd


records = []

for i, site in enumerate(structure):

    records.append({

        "site": i,

        "element":
            site.species_string,

        "importance":
            mean_importance[
                i
            ].item()
    })


importance_df = pd.DataFrame(
    records
)
```

This DataFrame can then be used to compare distributions by element.

For example:

```python
summary = (
    importance_df
    .groupby("element")
    ["importance"]
    .agg([
        "count",
        "mean",
        "std",
        "median",
        "min",
        "max"
    ])
)

print(summary)
```

This provides a more complete statistical description.

---

## 26.14.24 Importance and Chemical Substitution

Node importance can also be studied under substitution.

Consider:

```text
ABO₃
```

and:

```text
A'BO₃
```

where:

```text
A → A'
```

The researcher can compare:

```text
I_A
```

and:

```text
I_A'
```

as well as the importance of neighboring sites.

A useful workflow is:

```text
Original crystal
       ↓
Train / predict
       ↓
Node importance
       ↓
Substituted crystal
       ↓
Predict
       ↓
Node importance
       ↓
Compare
```

This can reveal whether the model changes its focus after chemical substitution.

---

## 26.14.25 Importance Difference Maps

For aligned structures, define:

```text
ΔIᵢ
=
Iᵢ^(B)
-
Iᵢ^(A)
```

In code:

```python
importance_difference = (
    importance_B
    -
    importance_A
)
```

Then:

```text
ΔI > 0
```

means increased model importance.

While:

```text
ΔI < 0
```

means decreased model importance.

This provides a useful way to investigate how the learned explanation changes under a controlled structural modification.

---

## 26.14.26 Node Importance and Defect Neighborhoods

For defect systems, it is often useful to compare:

```text
defect site
```

with:

```text
first-neighbor shell
```

and:

```text
second-neighbor shell.
```

Suppose the defect is at site `d`.

Define:

```text
N₁(d)
```

as first neighbors.

Then:

```text
N₂(d)
```

as second neighbors.

The researcher can calculate:

```python
def average_importance(
    scores,
    indices
):

    values = scores[
        indices
    ]

    return values.mean().item()
```

Then compare:

```text
Defect site       → 0.91
1st shell         → 0.74
2nd shell         → 0.41
Bulk              → 0.18
```

This provides a spatial picture of the model's dependence on the defect environment.

---

## 26.14.27 Importance Profiles Along Structural Distance

For a defect or interface, importance can be plotted against distance from the special structural feature.

Conceptually:

```text
Importance
   │
1.0│ ●
   │   ●
0.8│     ●
   │
0.6│       ●
   │
0.4│           ●
   │
0.2│               ● ●
   └────────────────────
      Distance from defect
```

A decreasing trend may indicate that the model's sensitivity is concentrated near the defect.

This is a useful bridge between:

```text
machine-learning explanation
```

and:

```text
local structural analysis.
```

---

## 26.14.28 Node Importance in Attention-Based GNNs

Some GNN architectures already contain attention coefficients.

For example:

```text
αᵢⱼ
```

may describe the learned weighting of message `j → i`.

This creates an important question:

> Is an attention coefficient the same thing as node importance?

The answer is generally:

**No.**

Attention indicates how the architecture weights information during message passing.

Node attribution asks how the final prediction depends on the input.

These are related but distinct quantities.

Therefore:

```text
Attention
≠
Attribution
```

This distinction should remain clear throughout XAI analysis.

---

## 26.14.29 Combining Attention and Node Attribution

Although they are not equivalent, they can be compared.

For example:

```text
Attention map
      +
Node saliency
      ↓
Agreement analysis
```

Suppose both identify the same local environment.

This may provide useful evidence that the model architecture is consistently focusing on that region.

But disagreement can also be informative.

For example:

```text
high attention
+
low attribution
```

may indicate that an interaction receives strong intermediate representation weight but contributes little to the final prediction.

---

## 26.14.30 Node Importance and Model Architecture

Different GNN architectures may produce different node-importance patterns.

For example:

```text
CGCNN
MEGNet
ALIGNN
M3GNet
```

may use different representations and message-passing mechanisms.

Therefore, comparing explanations across models can reveal whether the prediction is supported by:

```text
architecture-independent patterns
```

or:

```text
architecture-specific behavior.
```

A robust materials interpretation should ideally identify patterns that persist across reasonable model choices.

---

## 26.14.31 Cross-Architecture Explanation Comparison

A practical experiment can train:

```python
models = {
    "CGCNN": cgcnn_model,
    "MEGNet": megnet_model,
    "ALIGNN": alignn_model
}
```

Then calculate:

```python
explanations = {}

for name, model in models.items():

    explanations[name] = (
        explain_nodes(
            model,
            data
        )
    )
```

The resulting importance vectors can be compared.

For example:

```python
from scipy.stats import spearmanr


rho_cgcnn_alignn, _ = (
    spearmanr(
        explanations["CGCNN"]
        .detach()
        .cpu()
        .numpy(),

        explanations["ALIGNN"]
        .detach()
        .cpu()
        .numpy()
    )
)
```

A high rank correlation suggests similar site rankings.

---

## 26.14.32 What Makes a Good Node Explanation?

A useful node-importance explanation should satisfy several properties.

### Predictive relevance

The selected nodes should actually influence the prediction.

### Stability

The explanation should not change dramatically with small perturbations or random seeds.

### Structural consistency

The explanation should respect relevant crystal structure.

### Reproducibility

The procedure should be repeatable.

### Scientific usefulness

The result should produce a testable hypothesis.

Therefore:

```text
Good XAI
=
Attribution
+
Stability
+
Structural context
+
Scientific validation
```

---

## 26.14.33 Node Importance Is a Starting Point

The final goal is not:

```text
"Atom 17 is important."
```

The more useful scientific question is:

```text
"Why does the model depend on the environment
around Atom 17?"
```

This leads to deeper analysis:

```text
Atom 17
   ↓
Neighbors
   ↓
Bond lengths
   ↓
Coordination geometry
   ↓
Local symmetry
   ↓
Electronic environment
   ↓
Predicted property
```

This is where explainable AI becomes genuinely useful for Materials Informatics.

---

## 26.14.34 Research Workflow for Node Importance

A research-grade workflow can therefore be structured as:

```text
Trained Crystal GNN
        ↓
Select target prediction
        ↓
Compute node attribution
        ↓
Aggregate feature-level scores
        ↓
Normalize / preserve raw values
        ↓
Map to atomic sites
        ↓
Rank nodes
        ↓
Analyze chemical species
        ↓
Analyze local environments
        ↓
Analyze symmetry
        ↓
Analyze defects / substitutions
        ↓
Repeat across model seeds
        ↓
Compare explanation methods
        ↓
Validate structural hypothesis
```

This workflow prevents node importance from becoming merely a colorful visualization.

---

## 26.14.35 Complete Node-Importance Implementation

The following implementation combines the main ideas introduced in this section.

```python
import torch
import pandas as pd


def compute_node_importance(
    model,
    data
):

    model.eval()

    x = (
        data.x
        .clone()
        .detach()
        .requires_grad_(True)
    )

    model.zero_grad()

    output = model(
        x,
        data.edge_index,
        data.edge_attr,
        data.batch
    )

    target = output.squeeze()

    target.backward()

    gradients = (
        x.grad.detach()
    )

    importance = (
        gradients
        .norm(
            p=2,
            dim=1
        )
    )

    return importance
```

The structural mapping can then be implemented as:

```python
def build_node_importance_table(
    structure,
    importance
):

    rows = []

    for i, site in enumerate(
        structure
    ):

        rows.append({

            "site":
                i,

            "element":
                site.species_string,

            "x":
                site.coords[0],

            "y":
                site.coords[1],

            "z":
                site.coords[2],

            "importance":
                importance[i].item()
        })

    df = pd.DataFrame(
        rows
    )

    df = df.sort_values(
        "importance",
        ascending=False
    )

    return df
```

Usage:

```python
importance = (
    compute_node_importance(
        model,
        data
    )
)

importance_table = (
    build_node_importance_table(
        structure,
        importance
    )
)

print(
    importance_table.head(10)
)
```

This produces a reproducible ranked list of important atomic sites.

---

## 26.14.36 Recommended Research Reporting

When publishing node-importance results, report at least:

```text
Model architecture
Training dataset
Target property
Feature representation
Attribution method
Aggregation method
Normalization method
Random seed(s)
Number of models
Crystal structure
Periodic representation
Symmetry treatment
Node ranking procedure
```

For example:

```text
Model:
ALIGNN

Target:
Formation energy

Attribution:
Gradient-based node saliency

Aggregation:
L2 norm across node features

Normalization:
Per-crystal min-max normalization

Seeds:
5 independent training runs

Reported:
Mean ± standard deviation
```

This level of reporting makes the explanation reproducible.

---

## 26.14.37 Key Takeaways

The central ideas introduced in this section are:

1. **Node importance identifies atomic sites associated with model sensitivity.**

2. **Node importance can be derived by aggregating feature-level attribution.**

3. **L1, L2, maximum, and signed aggregation provide different interpretations.**

4. **Site-level importance should be distinguished from element-level importance.**

5. **Crystal GNN node importance often reflects local environments rather than isolated atoms.**

6. **Periodic graph construction must be considered when interpreting atomic neighborhoods.**

7. **Symmetry can provide an important consistency test for node explanations.**

8. **Node importance can be aggregated across model seeds to estimate explanation stability.**

9. **Cumulative attribution provides a way to quantify how concentrated an explanation is.**

10. **Defects, dopants, interfaces, and substitutions provide useful applications for node-level interpretation.**

11. **Attention coefficients and node attribution are related but should not be treated as identical.**

12. **Cross-model explanation agreement can increase confidence in a learned structural pattern.**

13. **Node importance is not itself a physical observable.**

14. **The scientific value of node importance comes from converting model attribution into testable structural hypotheses.**

The progression is now:

```text
Saliency Maps
      ↓
Node Importance
      ↓
Atomic Site Ranking
      ↓
Local Environment Analysis
      ↓
Symmetry / Defect Analysis
      ↓
Cross-Model Stability
      ↓
Scientific Hypothesis
```

Node importance therefore provides the atomic-scale interpretation layer of a crystal GNN.

The next step is to move one level deeper into the graph representation and examine **edge importance**: which atomic interactions, neighbor relationships, and interatomic connections contribute most strongly to the model prediction.

## 26.15 Edge Importance in Crystal Graph Neural Networks

Node importance identifies which atomic sites are associated with a model prediction.

However, crystal properties are rarely determined by isolated atoms.

They also depend strongly on **interatomic interactions**.

Examples include:

```text
Bonding
Coordination
Interatomic distance
Local geometry
Chemical interactions
Periodic interactions
Neighbor connectivity
```

A crystal GNN explicitly represents these relationships through graph edges.

Therefore, a natural question is:

> Which atomic interactions contribute most strongly to the model prediction?

This is the central idea of **edge importance**.

The conceptual transition is:

```text
Node Importance

Which atoms matter?
```

to:

```text
Edge Importance

Which atomic interactions matter?
```

For a crystal graph:

```text
Atoms → Nodes
Interactions → Edges
```

and therefore:

```text
Crystal
   ↓
Crystal Graph
   ↓
Nodes + Edges
   ↓
GNN
   ↓
Prediction
   ↓
Edge Attribution
   ↓
Important Interactions
```

This can provide substantially richer structural information than node attribution alone.

---

## 26.15.1 Why Edge Importance Matters

Consider two structures containing the same elements:

```text
Crystal A

A — B — C
```

and:

```text
Crystal B

A     B
 \   /
   C
```

The chemical composition is identical.

The atoms are identical.

But the connectivity and geometry are different.

A GNN can distinguish these structures because the graph representation contains information about:

```text
Who interacts with whom
```

and often:

```text
How strongly or at what distance
```

those interactions occur.

Therefore, edge-level explanation can reveal the structural relationships that the model uses.

---

## 26.15.2 Mathematical Representation of an Edge

Let the crystal graph be:

```text
G = (V, E)
```

where each edge is:

```text
eᵢⱼ
```

connecting nodes:

```text
vᵢ
```

and:

```text
vⱼ
```

An edge can contain features such as:

```text
Distance
Radial basis representation
Bond type
Periodic image information
Relative position
Angle-related information
```

For example:

```python
edge_attr = torch.tensor([
    [2.31, ...],
    [1.87, ...],
    [3.02, ...],
])
```

where the first value could represent an interatomic distance.

The exact representation depends on the GNN architecture.

---

## 26.15.3 Edge-Level Prediction Dependence

Suppose the model predicts:

```text
ŷ = f(G)
```

and an edge has feature vector:

```text
eᵢⱼ
```

A gradient-based edge sensitivity can be written conceptually as:

```text
Iᵢⱼ
=
|| ∂ŷ / ∂eᵢⱼ ||
```

where:

```text
Iᵢⱼ
```

is the edge importance.

This gives the basic interpretation:

```text
Large edge attribution
        ↓
Model prediction is sensitive
to this edge representation
```

while:

```text
Small edge attribution
        ↓
The prediction is less sensitive
to this edge representation
```

Again, this is **model attribution**, not a direct measurement of physical bond strength.

---

## 26.15.4 Edge Importance Is Not Bond Strength

This distinction is extremely important in Materials Informatics.

Suppose:

```text
Fe — O
```

has high model attribution.

It does **not** automatically mean:

```text
Fe—O bond is physically the strongest bond
```

The model may consider that interaction important because it helps predict:

```text
Formation energy
Band gap
Magnetic moment
Elastic property
Thermal property
```

The attribution depends on the target.

Therefore:

```text
Edge importance
≠
Bond energy
```

and:

```text
Edge importance
≠
Experimental bond strength
```

Instead:

```text
Edge importance
=
importance of that graph interaction
for the model's prediction
```

---

## 26.15.5 Edge Importance and Message Passing

The reason edge importance is particularly meaningful for GNNs is that edges control message passing.

A simplified message-passing layer can be written as:

```text
mᵢⱼ
=
φ(hᵢ, hⱼ, eᵢⱼ)
```

where:

```text
hᵢ
```

and:

```text
hⱼ
```

are node representations and:

```text
eᵢⱼ
```

is the edge representation.

The messages are aggregated:

```text
mᵢ
=
Σ mᵢⱼ
```

and used to update the node:

```text
hᵢ'
=
ψ(hᵢ, mᵢ)
```

Therefore:

```text
Edge
 ↓
Message
 ↓
Node representation
 ↓
Crystal representation
 ↓
Prediction
```

An edge can therefore influence the prediction indirectly by controlling information flow through the graph.

---

## 26.15.6 Direct Edge Attribution

If edge features are differentiable model inputs, the simplest approach is to enable gradients on them.

For example:

```python
edge_attr = (
    data.edge_attr
    .clone()
    .detach()
    .requires_grad_(True)
)
```

Then:

```python
model.zero_grad()

output = model(
    data.x,
    data.edge_index,
    edge_attr,
    data.batch
)

target = output.squeeze()

target.backward()
```

The gradients are then:

```python
edge_gradients = (
    edge_attr.grad.detach()
)
```

A magnitude-based score can be calculated:

```python
edge_importance = (
    edge_gradients
    .norm(
        p=2,
        dim=1
    )
)
```

This produces:

```text
One importance score
per edge
```

---

## 26.15.7 Complete Gradient-Based Edge Attribution

A reusable implementation is:

```python
import torch


def compute_edge_importance(
    model,
    data
):

    model.eval()

    edge_attr = (
        data.edge_attr
        .clone()
        .detach()
        .requires_grad_(True)
    )

    model.zero_grad()

    output = model(
        data.x,
        data.edge_index,
        edge_attr,
        data.batch
    )

    target = output.squeeze()

    target.backward()

    gradients = (
        edge_attr.grad.detach()
    )

    importance = (
        gradients
        .norm(
            p=2,
            dim=1
        )
    )

    return importance
```

Usage:

```python
edge_importance = (
    compute_edge_importance(
        model,
        data
    )
)

print(
    edge_importance.shape
)
```

If the graph contains:

```text
E = 128
```

edges, the output might be:

```text
torch.Size([128])
```

---

## 26.15.8 Edge Attribution Requires Careful Model Interfaces

Not every GNN exposes edge features in the same way.

For example, a model may use:

```python
model(
    x,
    edge_index,
    edge_attr
)
```

while another model may use:

```python
model(
    graph
)
```

where the graph object contains all information internally.

Some architectures may also transform edge features before message passing.

Therefore, the attribution procedure must be adapted to the model architecture.

The general principle remains:

```text
Find the representation
that carries edge information
```

and compute attribution with respect to that representation.

---

## 26.15.9 Edge Masks

A more general explanation approach is to introduce a learnable or differentiable edge mask.

Let:

```text
mᵢⱼ ∈ [0,1]
```

be an edge mask.

The modified edge representation becomes:

```text
e'ᵢⱼ
=
mᵢⱼ eᵢⱼ
```

Therefore:

```text
Original graph
      ↓
Edge mask
      ↓
Masked graph
      ↓
GNN
      ↓
Prediction
```

If:

```text
mᵢⱼ ≈ 1
```

the edge is retained.

If:

```text
mᵢⱼ ≈ 0
```

the edge is effectively removed.

This provides a flexible framework for identifying important graph relationships.

---

## 26.15.10 Learnable Edge Masks

Suppose:

```python
num_edges = data.edge_index.shape[1]
```

We initialize:

```python
edge_mask = torch.nn.Parameter(
    torch.zeros(num_edges)
)
```

Convert the mask to the range `[0, 1]`:

```python
mask = torch.sigmoid(
    edge_mask
)
```

Then apply it:

```python
masked_edge_attr = (
    data.edge_attr
    * mask.unsqueeze(-1)
)
```

The model receives:

```python
output = model(
    data.x,
    data.edge_index,
    masked_edge_attr,
    data.batch
)
```

The mask can then be optimized to preserve the prediction while using as few edges as possible.

This idea is closely related to graph explanation methods such as **GNNExplainer**.

---

## 26.15.11 Edge Masks and Sparse Explanations

Suppose a crystal graph contains:

```text
200 edges
```

but only:

```text
20 edges
```

are sufficient to preserve most of the prediction.

An explanation can then focus on those 20 interactions.

Conceptually:

```text
Full Crystal Graph

200 edges
   ↓
Explanation Algorithm
   ↓
20 important edges
```

This creates a compact structural explanation.

The resulting graph can be much easier for a scientist to inspect.

---

## 26.15.12 Edge Importance Through GNNExplainer

GNNExplainer attempts to identify a compact subgraph and feature subset that are sufficient for explaining a prediction.

Conceptually:

```text
Original Graph
      ↓
Candidate Edge Mask
      ↓
Candidate Feature Mask
      ↓
Prediction
      ↓
Optimize Masks
      ↓
Important Subgraph
```

For a crystal graph, this can produce:

```text
Important atoms
+
Important interactions
+
Important features
```

which is more informative than edge ranking alone.

---

## 26.15.13 Optimization Objective

A simplified explanation objective can be written conceptually as:

```text
Explanation Loss
=
Prediction Loss
+
Sparsity Penalty
+
Regularization
```

The prediction term encourages the masked graph to preserve the original prediction.

The sparsity term encourages fewer edges.

The regularization term encourages a stable or meaningful mask.

Therefore:

```text
Preserve prediction
        +
Remove unnecessary information
        ↓
Compact explanation
```

This is the fundamental intuition behind subgraph-based explanation.

---

## 26.15.14 A Simplified Edge-Mask Optimization

A conceptual PyTorch implementation can be written as:

```python
import torch


def optimize_edge_mask(
    model,
    data,
    steps=300,
    lr=0.01
):

    model.eval()

    num_edges = (
        data.edge_index.shape[1]
    )

    mask_logits = torch.nn.Parameter(
        torch.zeros(
            num_edges,
            device=data.x.device
        )
    )

    optimizer = torch.optim.Adam(
        [mask_logits],
        lr=lr
    )

    with torch.no_grad():

        original_output = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

    original_output = (
        original_output.detach()
    )

    for step in range(steps):

        optimizer.zero_grad()

        mask = torch.sigmoid(
            mask_logits
        )

        masked_edge_attr = (
            data.edge_attr
            *
            mask.unsqueeze(-1)
        )

        output = model(
            data.x,
            data.edge_index,
            masked_edge_attr,
            data.batch
        )

        prediction_loss = (
            output - original_output
        ).pow(2).mean()

        sparsity_loss = (
            mask.mean()
        )

        loss = (
            prediction_loss
            +
            0.01 * sparsity_loss
        )

        loss.backward()

        optimizer.step()

    return torch.sigmoid(
        mask_logits
    ).detach()
```

This is a simplified educational implementation.

A production implementation should use a carefully designed explanation objective, temperature relaxation where appropriate, numerical stabilization, and architecture-specific handling.

---

## 26.15.15 Why the Simplified Implementation Is Useful

Even though the implementation is simplified, it demonstrates the core idea:

```text
Initialize mask
      ↓
Apply mask
      ↓
Predict
      ↓
Compare with original prediction
      ↓
Penalize unnecessary edges
      ↓
Optimize mask
```

The resulting mask can then be interpreted as:

```text
edge relevance
```

rather than merely:

```text
gradient magnitude.
```

This distinction is useful because gradient-based and mask-based explanations answer slightly different questions.

---

## 26.15.16 Gradient Attribution Versus Edge Masks

### Gradient attribution

Asks:

```text
How sensitive is the prediction
to this edge representation?
```

### Edge-mask explanation

Asks:

```text
Can this edge be removed while
preserving the prediction?
```

These are not identical.

A graph may contain an edge with:

```text
high gradient
```

but that edge may not be necessary when other correlated information is available.

Conversely, an edge with moderate local gradient may be essential when removed.

Therefore, comparing both methods can be valuable.

---

## 26.15.17 Edge Importance and Interatomic Distance

In many crystal GNNs, edge features contain information about interatomic distance.

For example:

```text
eᵢⱼ
=
RBF(
dᵢⱼ
)
```

where:

```text
dᵢⱼ
```

is the interatomic distance.

Therefore, edge importance can sometimes be analyzed against distance.

A useful analysis is:

```text
Edge importance
vs.
Interatomic distance
```

The researcher may discover that the model relies predominantly on:

```text
short-range interactions
```

or:

```text
longer-range interactions.
```

However, the interpretation depends strongly on the graph construction cutoff.

---

## 26.15.18 Distance-Binned Edge Importance

Suppose edge distances are available:

```python
distances = data.edge_attr[:, 0]
```

and edge importance is:

```python
importance = edge_importance
```

Create bins:

```python
bins = torch.tensor([
    0.0,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0
])
```

Assign edges:

```python
bin_ids = torch.bucketize(
    distances,
    bins
)
```

Then calculate mean importance:

```python
for b in torch.unique(
    bin_ids
):

    values = importance[
        bin_ids == b
    ]

    print(
        b.item(),
        values.mean().item()
    )
```

This produces an approximate:

```text
distance → mean edge importance
```

relationship.

---

## 26.15.19 Important Warning About Distance Correlation

If the graph was constructed using:

```text
cutoff = 5 Å
```

then all edges beyond 5 Å are absent.

Therefore, observing that:

```text
short edges
```

are important does not necessarily prove that short-range interactions are intrinsically more important.

The graph construction itself imposes a prior.

This is a critical XAI principle:

> Explanations can only reveal information available to the model.

If long-range interactions are excluded from the graph, the model cannot use them.

---

## 26.15.20 Edge Importance by Element Pair

Another useful analysis is to group edges by chemical pair.

For example:

```text
Fe—O
Fe—Fe
O—O
```

Suppose the graph contains:

```python
edge_pairs = [
    ("Fe", "O"),
    ("Fe", "Fe"),
    ("O", "O")
]
```

We can aggregate:

```python
from collections import defaultdict


pair_scores = defaultdict(list)

for pair, score in zip(
    edge_pairs,
    edge_importance
):

    canonical_pair = tuple(
        sorted(pair)
    )

    pair_scores[
        canonical_pair
    ].append(
        score.item()
    )
```

Then:

```python
for pair, scores in (
    pair_scores.items()
):

    print(
        pair,
        sum(scores) / len(scores)
    )
```

Possible result:

```text
('Fe', 'O') → 0.72
('Fe', 'Fe') → 0.31
('O', 'O')   → 0.18
```

This may indicate that Fe–O interactions are especially relevant to the learned prediction.

Again:

```text
model importance
```

should not automatically be interpreted as:

```text
physical bond strength.
```

---

## 26.15.21 Directed Versus Undirected Crystal Edges

Many crystal graphs contain both:

```text
i → j
```

and:

```text
j → i
```

because message passing is directional.

Therefore, an apparently duplicated pair may represent two directed messages.

For example:

```text
Fe₁ → O₁
O₁ → Fe₁
```

These should not necessarily be treated as two independent physical bonds.

A physical interaction may instead be represented by the pair:

```text
{Fe₁, O₁}
```

Therefore, edge importance analysis should specify whether scores are reported for:

```text
directed graph edges
```

or:

```text
undirected physical interactions.
```

---

## 26.15.22 Combining Reciprocal Directed Edges

For a bidirectional graph, one simple aggregation is:

```text
I(Fe,O)
=
I(Fe→O)
+
I(O→Fe)
```

or:

```text
I(Fe,O)
=
mean(
    I(Fe→O),
    I(O→Fe)
)
```

In code:

```python
def combine_bidirectional_edges(
    edge_index,
    importance
):

    pair_scores = {}

    for k in range(
        edge_index.shape[1]
    ):

        i = (
            edge_index[0, k]
            .item()
        )

        j = (
            edge_index[1, k]
            .item()
        )

        pair = tuple(
            sorted((i, j))
        )

        pair_scores.setdefault(
            pair,
            []
        ).append(
            importance[k].item()
        )

    combined = {}

    for pair, values in (
        pair_scores.items()
    ):

        combined[pair] = (
            sum(values)
            /
            len(values)
        )

    return combined
```

This creates one importance value per physical node pair.

---

## 26.15.23 Periodic Edge Interpretation

Crystal graphs introduce another complication.

An edge may connect:

```text
Atom i
```

to:

```text
Periodic image of Atom j
```

For example:

```text
O₁
 |
 | periodic translation
 |
Fe₂ + T
```

where:

```text
T
```

is a lattice translation vector.

This information is important because two edges connecting the same elemental species may correspond to different periodic images.

Therefore, edge explanations should ideally preserve:

```text
source atom
target atom
periodic image
distance
```

rather than only:

```text
element pair.
```

---

## 26.15.24 Mapping Edge Importance Back to the Crystal

Once edge importance has been computed, it must be mapped back to structural coordinates.

A useful table contains:

```text
Source
Target
Source element
Target element
Distance
Periodic image
Importance
```

For example:

```python
records = []

for k in range(
    data.edge_index.shape[1]
):

    i = (
        data.edge_index[0, k]
        .item()
    )

    j = (
        data.edge_index[1, k]
        .item()
    )

    records.append({

        "source": i,

        "target": j,

        "source_element":
            structure[
                i
            ].species_string,

        "target_element":
            structure[
                j
            ].species_string,

        "importance":
            edge_importance[
                k
            ].item()
    })
```

Then:

```python
edge_df = pd.DataFrame(
    records
)

edge_df = edge_df.sort_values(
    "importance",
    ascending=False
)
```

Now the researcher can inspect the most important interactions directly.

---

## 26.15.25 Top Edge Analysis

The top edges can be extracted using:

```python
top_edges = (
    edge_df
    .head(20)
)

print(
    top_edges
)
```

A result might look like:

```text
source target source_element target_element importance

17     32     Fe             O              0.93
4      17     O              Fe             0.89
11     17     O              Fe             0.87
...
```

This provides a direct list of interactions associated with the model prediction.

---

## 26.15.26 Visualizing Important Edges

A useful visualization is to display the crystal structure and encode edge importance using:

```text
line thickness
```

or:

```text
opacity
```

Conceptually:

```text
Low importance
────────

Medium importance
════════

High importance
━━━━━━━━
```

The visualization should be accompanied by a clear legend.

For example:

```text
Edge thickness
∝
model attribution
```

The researcher should not label the edges simply as:

```text
"strong bonds"
```

unless independent physical evidence supports that interpretation.

---

## 26.15.27 Important Subgraph Extraction

The highest-importance edges can be used to construct a subgraph.

Suppose:

```text
top 10%
```

of edges are retained.

```python
threshold = torch.quantile(
    edge_importance,
    0.90
)

important_edges = (
    edge_importance
    >= threshold
)
```

Then:

```python
selected_edge_index = (
    data.edge_index[
        :,
        important_edges
    ]
)
```

The resulting graph contains only the most highly attributed interactions.

This can reveal structural motifs.

---

## 26.15.28 Structural Motif Discovery

Suppose an explanation repeatedly identifies:

```text
Metal
   |
O — Metal — O
   |
   O
```

as an important subgraph.

This may suggest that the model is strongly using a particular coordination motif.

The researcher can then investigate:

```text
Does this motif occur across high-performing materials?

Does it correlate with the target property?

Is it chemically plausible?

Does it appear in independent datasets?

Does DFT support the hypothesis?
```

This transforms XAI from:

```text
post-hoc visualization
```

into:

```text
scientific hypothesis generation.
```

---

## 26.15.29 Edge Importance Across Multiple Targets

The same crystal may have multiple predicted properties.

For example:

```text
Formation energy
Band gap
Bulk modulus
Magnetic moment
```

The important edges may differ for each target.

Therefore:

```text
Crystal
  ↓
      ┌─────────────┐
      │             │
Formation Energy  Band Gap
      │             │
Edge Map A       Edge Map B
```

A practical implementation can compute separate explanations:

```python
importance_energy = (
    explain_edges(
        model_energy,
        data
    )
)

importance_bandgap = (
    explain_edges(
        model_bandgap,
        data
    )
)
```

The maps can then be compared.

---

## 26.15.30 Target-Dependent Structural Reasoning

Suppose:

```text
Fe—O edges
```

are highly important for formation-energy prediction.

But:

```text
Fe—Fe edges
```

are highly important for magnetic-moment prediction.

This would suggest that the model has learned target-dependent structural relationships.

Such a result can be scientifically interesting.

However, it should be validated against domain knowledge and independent calculations.

---

## 26.15.31 Edge Importance Stability Across Random Seeds

As with node attribution, explanations should be tested across multiple models.

```python
all_edge_scores = []

for seed in range(5):

    set_seed(seed)

    model = train_model(
        dataset
    )

    scores = (
        compute_edge_importance(
            model,
            data
        )
    )

    all_edge_scores.append(
        scores
    )
```

Then:

```python
edge_scores = torch.stack(
    all_edge_scores
)

mean_edge_scores = (
    edge_scores.mean(
        dim=0
    )
)

std_edge_scores = (
    edge_scores.std(
        dim=0
    )
)
```

Now every edge has:

```text
mean importance
+
importance uncertainty
```

This is preferable to relying on one trained model.

---

## 26.15.32 Edge Explanation Agreement

Two explanation methods may be compared.

For example:

```text
Gradient
```

versus:

```text
Mask-based explanation
```

Suppose the top 20 edges from each method are:

```python
top_gradient = set(
    torch.topk(
        gradient_scores,
        20
    ).indices.tolist()
)

top_mask = set(
    torch.topk(
        mask_scores,
        20
    ).indices.tolist()
)
```

The overlap can be measured:

```python
intersection = (
    top_gradient
    &
    top_mask
)

union = (
    top_gradient
    |
    top_mask
)

jaccard = (
    len(intersection)
    /
    len(union)
)
```

The Jaccard score is:

```text
0 → no overlap

1 → complete overlap
```

This provides a quantitative measure of explanation agreement.

---

## 26.15.33 Edge Importance and Counterfactual Removal

Another useful experiment is to remove important edges and observe the prediction change.

Suppose:

```text
Original prediction:
ŷ = 1.24
```

Remove top 10% important edges:

```text
Prediction:
ŷ_removed = 0.71
```

Then:

```text
Δŷ
=
ŷ
-
ŷ_removed
```

can be calculated.

```python
delta_prediction = (
    original_prediction
    -
    masked_prediction
)
```

If removing high-attribution edges strongly changes the output, the explanation has meaningful predictive relevance.

---

## 26.15.34 Comparing Important and Random Edges

A stronger test compares removal of:

```text
important edges
```

against:

```text
random edges.
```

For example:

```python
important_prediction = (
    evaluate_after_removal(
        model,
        data,
        important_edges
    )
)

random_prediction = (
    evaluate_after_removal(
        model,
        data,
        random_edges
    )
)
```

If:

```text
important-edge removal
```

causes a substantially larger prediction change than:

```text
random-edge removal,
```

this provides evidence that the explanation identifies genuinely influential graph structure.

---

## 26.15.35 Perturbation-Based Validation

A general explanation-validation procedure is:

```text
Original Graph
      ↓
Explanation
      ↓
Select Important Edges
      ↓
Perturb / Remove
      ↓
Prediction Change
```

The explanation is considered more useful if the selected edges cause larger prediction changes than appropriately chosen controls.

This is called a form of:

```text
faithfulness evaluation.
```

---

## 26.15.36 Faithfulness Versus Plausibility

An explanation can be:

```text
Plausible
```

without being:

```text
Faithful
```

For example, a materials scientist may find:

```text
Fe—O
```

chemically intuitive.

But if removing that edge has almost no effect on the model prediction, the explanation may not be faithful.

Conversely, a model may rely on a graph interaction that appears chemically surprising.

That does not automatically make the explanation wrong.

Therefore, a good XAI analysis should consider both:

```text
Faithfulness
```

and:

```text
Scientific plausibility.
```

---

## 26.15.37 Research-Grade Edge Explanation Workflow

The complete workflow can now be summarized as:

```text
Trained Crystal GNN
        ↓
Target prediction
        ↓
Select edge attribution method
        ↓
Compute edge importance
        ↓
Aggregate edge features
        ↓
Map edges to atomic pairs
        ↓
Restore periodic information
        ↓
Rank edges
        ↓
Analyze distance
        ↓
Analyze element pairs
        ↓
Analyze coordination motifs
        ↓
Analyze symmetry
        ↓
Test across random seeds
        ↓
Compare explanation methods
        ↓
Perform perturbation tests
        ↓
Generate scientific hypothesis
```

This makes edge importance a complete analysis pipeline rather than a single visualization.

---

## 26.15.38 Integrated Node-and-Edge Explanation

The strongest crystal explanation often combines node and edge attribution.

Instead of:

```text
Important atoms
```

or:

```text
Important interactions
```

we can construct:

```text
Important atomic environment
```

For example:

```text
        O
        |
        |
O — Fe — O
        |
        O
```

with:

```text
Fe site       → high node importance

Fe—O edges    → high edge importance

outer O sites → moderate importance
```

This creates a much more complete explanation.

---

## 26.15.39 Hierarchical Crystal Explanation

A useful hierarchy is:

```text
Crystal
   ↓
Region
   ↓
Atomic site
   ↓
Local environment
   ↓
Interaction
   ↓
Feature
```

Different XAI techniques operate at different levels.

For example:

```text
Node attribution
        ↓
Atomic site

Edge attribution
        ↓
Interaction

Feature attribution
        ↓
Atomic descriptor

Subgraph explanation
        ↓
Local structural motif
```

A research-grade XAI system should ideally support several levels simultaneously.

---

## 26.15.40 Key Takeaways

The central ideas introduced in this section are:

1. **Edge importance identifies graph interactions associated with model predictions.**

2. **Gradient-based attribution can estimate sensitivity with respect to edge features.**

3. **Edge masks provide an alternative explanation based on preserving predictions while removing unnecessary graph information.**

4. **GNNExplainer provides a general framework for identifying important subgraphs and features.**

5. **Edge attribution should not automatically be interpreted as physical bond strength.**

6. **Periodic crystal graphs require careful treatment of periodic-image edges.**

7. **Directed graph edges may need to be combined into physical interactions.**

8. **Edge importance can be analyzed by interatomic distance and chemical pair.**

9. **Important-edge subgraphs can reveal structural motifs used by the model.**

10. **Edge importance can differ strongly between target properties.**

11. **Explanation stability should be evaluated across random seeds and, where possible, across explanation methods.**

12. **Perturbation tests provide an important way to evaluate explanation faithfulness.**

13. **Node and edge explanations are most informative when interpreted together.**

14. **The ultimate goal is to transform graph attribution into a scientifically testable structural hypothesis.**

The progression is therefore:

```text
Node Importance
      ↓
Edge Importance
      ↓
Important Interactions
      ↓
Important Subgraphs
      ↓
Structural Motifs
      ↓
Perturbation Validation
      ↓
Scientific Interpretation
```

Edge attribution adds the interaction-level perspective that node attribution alone cannot provide.

The next section will move from individual node and edge scores toward **integrated crystal-graph explanations**, combining feature attribution, node importance, edge importance, and structural context into a unified scientific interpretation framework.

## 26.15.41 Integrated Crystal-Graph Explanations

Node importance and edge importance provide two complementary perspectives.

Node attribution asks:

```text
Which atoms matter?
```

Edge attribution asks:

```text
Which interactions matter?
```

Feature attribution asks:

```text
Which input characteristics matter?
```

A complete explanation should ideally combine all three.

The resulting interpretation becomes:

```text
Crystal
   ↓
Crystal Graph
   ↓
┌─────────────────────────────┐
│                             │
│   Node Importance           │
│   Edge Importance           │
│   Feature Importance        │
│                             │
└─────────────────────────────┘
   ↓
Important Local Environment
   ↓
Structural Interpretation
   ↓
Scientific Hypothesis
```

This integrated perspective is particularly important for Materials Informatics because a crystal property is rarely controlled by a single atom, a single bond, or a single numerical feature.

Instead, the prediction may emerge from an interaction between:

```text
Chemical Identity
+
Local Coordination
+
Interatomic Distance
+
Connectivity
+
Crystal Symmetry
+
Longer-Range Structure
```

Therefore, explainability should attempt to reconstruct the **structural context** behind the prediction.

---

## 26.15.42 Why Individual Attribution Scores Are Not Enough

Suppose an explanation produces:

```text
Atom 17 → importance = 0.91
```

This tells us that atom 17 is associated with the prediction.

But it does not tell us:

```text
Why is atom 17 important?
```

We may then examine its edges:

```text
Atom 17
 ├── O₄
 ├── O₇
 ├── O₁₁
 └── Fe₃
```

Suppose the important edges are:

```text
17—O₄  → 0.92
17—O₇  → 0.87
17—O₁₁ → 0.84
```

Now the explanation becomes more informative.

We can begin to describe the model's reasoning as:

```text
Important atom
      ↓
Important neighbors
      ↓
Important local environment
```

This is much closer to a scientific interpretation.

---

## 26.15.43 Local Environment Reconstruction

For a selected important atom, the first step is to reconstruct its local environment.

Suppose:

```python
important_atom = 17
```

We can identify neighboring atoms using the graph connectivity.

```python
neighbors = []

for k in range(
    data.edge_index.shape[1]
):

    source = (
        data.edge_index[0, k]
        .item()
    )

    target = (
        data.edge_index[1, k]
        .item()
    )

    if source == important_atom:

        neighbors.append(
            target
        )
```

The neighboring atoms can then be inspected:

```python
for atom_index in neighbors:

    print(
        atom_index,
        structure[
            atom_index
        ].species_string
    )
```

This gives a basic local chemical environment.

---

## 26.15.44 Combining Node and Edge Scores

Suppose we have:

```text
node_importance[i]
```

for every atom and:

```text
edge_importance[k]
```

for every graph edge.

A local environment score can be constructed conceptually as:

```text
Environment Importance
=
Node Importance
+
Neighbor Edge Importance
```

For atom `i`:

```text
Eᵢ
=
Nᵢ
+
Σ Iᵢⱼ
```

where:

```text
Nᵢ
```

is the node attribution and:

```text
Iᵢⱼ
```

represents the importance of interactions involving atom `i`.

This should not be treated as a universal mathematically correct attribution rule.

It is an analysis tool for combining complementary signals.

---

## 26.15.45 A Simple Implementation

For a graph with edge importance scores:

```python
def combined_node_edge_scores(
    edge_index,
    node_importance,
    edge_importance
):

    combined = (
        node_importance
        .clone()
    )

    for k in range(
        edge_index.shape[1]
    ):

        i = (
            edge_index[0, k]
            .item()
        )

        j = (
            edge_index[1, k]
            .item()
        )

        score = (
            edge_importance[k]
        )

        combined[i] += score
        combined[j] += score

    return combined
```

The resulting score provides an approximate measure of how strongly each atomic site participates in the attributed graph structure.

Because the scale of node and edge attribution can differ, normalization should normally be performed before combining them.

---

## 26.15.46 Attribution Normalization

Suppose:

```text
Node scores:
0.1 → 0.9

Edge scores:
0.001 → 0.02
```

Directly adding them would make the edge contribution appear negligible simply because the numerical scales differ.

Therefore, normalization is important.

A simple normalization is:

```python
def normalize_scores(
    scores
):

    minimum = scores.min()

    maximum = scores.max()

    return (
        scores - minimum
    ) / (
        maximum - minimum
        + 1e-12
    )
```

Then:

```python
node_scores = normalize_scores(
    node_importance
)

edge_scores = normalize_scores(
    edge_importance
)
```

The normalized scores can then be compared more meaningfully.

---

## 26.15.47 Signed Versus Absolute Importance

Not all attribution methods produce only positive values.

A feature, node, or edge may contribute:

```text
positively
```

or:

```text
negatively
```

to the prediction.

For example:

```text
+0.8
```

may indicate that the feature pushes the prediction upward.

While:

```text
-0.6
```

may indicate that it pushes the prediction downward.

Therefore, taking only the absolute value:

```python
importance = gradient.abs()
```

can hide directional information.

The appropriate representation depends on the scientific question.

For example:

```text
Magnitude:
How strongly does this feature matter?

Sign:
Does it increase or decrease the prediction?
```

These are different questions.

---

## 26.15.48 Positive and Negative Structural Contributions

Suppose a model predicts band gap.

An attributed structural interaction may have:

```text
positive contribution
```

toward a larger predicted band gap.

Another interaction may have:

```text
negative contribution
```

toward a smaller predicted band gap.

A useful explanation therefore distinguishes:

```text
Interactions supporting high band gap
```

from:

```text
Interactions suppressing high band gap
```

This can be much more scientifically informative than simply ranking interactions by absolute magnitude.

However, signed attribution should only be interpreted relative to the specific baseline and explanation method used.

---

## 26.15.49 Baselines in Crystal Explanations

Several attribution methods require a baseline.

For example, Integrated Gradients computes attribution by comparing:

```text
Baseline
```

with:

```text
Actual input
```

For a crystal graph, selecting the baseline is not trivial.

Possible choices include:

```text
Zero features
Average feature values
Reference crystal
Masked graph
Simplified local environment
```

Each choice answers a slightly different question.

For example:

```text
Zero-feature baseline
```

asks approximately:

> How does the actual crystal differ from an artificial zero-information representation?

A reference-crystal baseline asks:

> Which structural differences from this reference are associated with the prediction?

Therefore, baseline selection is part of the scientific interpretation.

---

## 26.15.50 Integrated Gradients for Crystal Features

Integrated Gradients can be expressed conceptually as:

```text
IGᵢ
=
(xᵢ - xᵢ')
×
∫₀¹
∂F(
x' + α(x-x')
)
/
∂xᵢ
dα
```

where:

```text
x
```

is the actual input and:

```text
x'
```

is the baseline.

For crystal GNNs, `x` may contain:

```text
Atomic features
Edge features
```

depending on the implementation.

The method evaluates gradients along a path between the baseline and actual structure.

Conceptually:

```text
Baseline Crystal Representation
          ↓
Intermediate Representation
          ↓
Intermediate Representation
          ↓
          ...
          ↓
Actual Crystal Representation
          ↓
Integrated Attribution
```

This can produce smoother explanations than a single gradient evaluation.

---

## 26.15.51 Why Integrated Gradients Can Be Useful

A single gradient is local.

It tells us how the prediction responds to a very small perturbation around the current input.

Integrated Gradients instead evaluates the model along an entire path.

Therefore:

```text
Single Gradient
      ↓
Local sensitivity
```

whereas:

```text
Integrated Gradients
      ↓
Accumulated sensitivity
along a baseline-to-input path
```

For nonlinear crystal GNNs, this distinction can be important.

---

## 26.15.52 Applying Integrated Gradients to Atomic Features

Suppose node features are:

```python
data.x
```

An attribution library can be used to construct an explanation around the model.

A simplified conceptual workflow is:

```python
from captum.attr import IntegratedGradients
```

The model wrapper may be:

```python
def forward(
    x
):

    return model(
        x,
        data.edge_index,
        data.edge_attr,
        data.batch
    )
```

Then:

```python
ig = IntegratedGradients(
    forward
)
```

Attributions can be computed using:

```python
attribution = ig.attribute(
    data.x,
    baselines=baseline
)
```

The resulting tensor has the same structural organization as the node features.

Therefore:

```text
Attribution
    ↓
Atom
    ↓
Feature
```

can be analyzed directly.

---

## 26.15.53 From Feature Attribution to Atomic Interpretation

Suppose each atom contains:

```text
Atomic number
Period
Group
Electronegativity
Atomic radius
Valence information
```

and the attribution identifies:

```text
Atom 17
Electronegativity
→ high importance
```

The explanation becomes:

```text
Atom 17
   ↓
Electronegativity feature
   ↓
Strong contribution
   ↓
Prediction
```

This is much more informative than simply saying:

```text
Atom 17 is important.
```

The researcher can now investigate whether the importance is consistent with known chemical behavior.

---

## 26.15.54 Feature, Node, and Edge Attribution Together

A complete explanation can therefore be represented as:

```text
                   Prediction
                       ↑
          ┌────────────┼────────────┐
          │            │            │
       Features      Nodes        Edges
          │            │            │
          ↓            ↓            ↓
   Electronegativity  Fe site     Fe—O
   Atomic radius      O site      Fe—Fe
   Valence            etc.        O—O
```

This provides three levels of interpretation:

```text
Feature level
"What property of the atom matters?"

Node level
"Which atom matters?"

Edge level
"Which interaction matters?"
```

Together they create a much stronger explanation.

---

## 26.15.55 Attention Visualization

Some GNN architectures use attention mechanisms.

Instead of treating all neighboring messages equally, an attention-based layer may assign different weights.

Conceptually:

```text
        O₁
        ↑
        │ 0.82
        │
O₂ ── Fe ── O₃
     ↑ 0.11   ↑ 0.67
```

The attention coefficients can then be visualized.

A simplified attention equation is:

```text
αᵢⱼ
=
softmax(
score(hᵢ,hⱼ,eᵢⱼ)
)
```

where:

```text
αᵢⱼ
```

represents the attention weight assigned to the message from neighbor `j` to node `i`.

Large attention means the model gives that message greater weight during that layer.

---

## 26.15.56 Attention Is Not Automatically Explanation

This distinction is critical.

It is tempting to interpret:

```text
high attention
```

as:

```text
high importance.
```

But attention weights describe how the model distributes information within an attention mechanism.

They do not necessarily provide a complete causal explanation of the final prediction.

Therefore:

```text
Attention visualization
```

should be considered:

```text
an interpretability signal
```

rather than automatically:

```text
ground-truth feature importance.
```

Attention should ideally be compared with other explanation techniques.

---

## 26.15.57 Extracting Attention Weights

If a custom GNN layer exposes attention weights:

```python
output, attention = layer(
    x,
    edge_index,
    edge_attr,
    return_attention_weights=True
)
```

the returned attention can be stored:

```python
attention_weights = (
    attention.detach()
)
```

Then:

```python
print(
    attention_weights.shape
)
```

may produce something similar to:

```text
[number_of_edges, number_of_heads]
```

For multi-head attention, the researcher may average:

```python
mean_attention = (
    attention_weights.mean(
        dim=1
    )
)
```

or inspect each head separately.

---

## 26.15.58 Multi-Head Attention in Crystal Graphs

Suppose a model has:

```text
4 attention heads
```

Each head may learn a different structural pattern.

Conceptually:

```text
Head 1 → short-range interactions
Head 2 → chemical environment
Head 3 → coordination
Head 4 → longer-range pattern
```

This is only an interpretation hypothesis.

It must be tested rather than assumed.

A useful analysis is therefore:

```text
Attention Head
      ↓
Important edges
      ↓
Element pairs
      ↓
Distance distribution
      ↓
Structural motif
```

This can reveal whether different attention heads specialize in different structural patterns.

---

## 26.15.59 Comparing Attention With Attribution

Suppose an edge has:

```text
Attention = 0.91
Gradient importance = 0.22
```

while another has:

```text
Attention = 0.42
Gradient importance = 0.88
```

The two measures disagree.

This is not necessarily a problem.

They measure different aspects of the model.

Attention asks approximately:

```text
How strongly is this message weighted?
```

Gradient attribution asks:

```text
How sensitive is the output to this representation?
```

Therefore, disagreement can itself be informative.

---

## 26.15.60 Saliency Maps for Crystal Graphs

Saliency methods can be used to visualize which input components strongly influence the output.

For image models, saliency is often shown as:

```text
bright pixels
```

For crystal graphs, the equivalent representation is:

```text
important atoms
+
important edges
```

The visualization may therefore look conceptually like:

```text
        O
        ║
        ║
O ━━━ Fe ━━━ O
        ║
        O
```

where thicker or brighter graph elements indicate larger attribution.

The crystal structure becomes a **structural saliency map**.

---

## 26.15.61 Crystal Saliency Versus Image Saliency

The underlying principle is similar:

```text
Image:
pixels → importance

Crystal:
atoms/edges → importance
```

But the crystal case is more structured.

A crystal has:

```text
3D geometry
+
periodicity
+
chemical identity
+
graph connectivity
```

Therefore, the saliency representation must preserve these properties.

A flat heatmap can be misleading if it removes important three-dimensional or periodic information.

---

## 26.15.62 Periodic Crystal Visualization

A robust visualization should distinguish:

```text
Atom
```

from:

```text
Periodic image of atom
```

For example:

```text
       Cell A          Cell B

       O₁              O₁'
        \               /
         \             /
          Fe₁ ───── Fe₂
```

An important interaction may cross the unit-cell boundary.

If the visualization only displays atoms inside the primitive cell, the interaction can appear disconnected.

Therefore, periodic image information should be retained during visualization.

---

## 26.15.63 Three-Dimensional Explanation

For realistic crystal analysis, a 3D visualization is often preferable.

The visual representation can encode:

```text
Atom size
→ atomic identity

Atom opacity
→ node importance

Edge thickness
→ edge importance

Edge color
→ positive/negative attribution

Edge style
→ periodic interaction
```

This allows multiple explanatory dimensions to be shown simultaneously.

However, too many visual encodings can make the figure difficult to interpret.

Therefore, scientific visualization should prioritize clarity.

---

## 26.15.64 Explanation Tables

Visualizations should be accompanied by structured tables.

For example:

```text
Atom  Element  Node Importance  Key Feature
------------------------------------------------
17    Fe       0.94             Valence
32    O        0.81             Electronegativity
11    O        0.76             Atomic radius
```

and:

```text
Edge    Pair    Distance    Importance
---------------------------------------
17-32   Fe-O    1.92 Å      0.93
17-11   Fe-O    2.01 Å      0.87
17-44   Fe-Fe   2.74 Å      0.64
```

These tables make explanations reproducible and easier to compare across materials.

---

## 26.15.65 From Explanation to Scientific Hypothesis

The ultimate purpose of XAI is not merely to generate attractive figures.

The explanation should help formulate a scientific question.

For example:

```text
Observation:

High-band-gap materials
frequently show strong attribution
around a particular coordination motif.
```

This leads to:

```text
Hypothesis:

The local coordination environment
may contribute significantly to
the predicted band gap.
```

The hypothesis can then be tested independently.

For example:

```text
Dataset analysis
      ↓
DFT calculations
      ↓
Controlled structural perturbation
      ↓
Experimental investigation
```

This is where explainable AI becomes scientifically valuable.

---

## 26.15.66 Counterfactual Structural Testing

Suppose XAI identifies:

```text
Metal–O coordination
```

as highly important.

A counterfactual test can modify that environment.

For example:

```text
Original:
Metal — O

Counterfactual:
Metal — S
```

The model prediction can then be compared.

Conceptually:

```text
Original Structure
       ↓
Prediction
       ↓
Modify Important Motif
       ↓
New Structure
       ↓
New Prediction
       ↓
Δ Prediction
```

A large change supports the hypothesis that the identified structural motif is relevant to the model.

However, the counterfactual structure must also be chemically meaningful.

---

## 26.15.67 Scientific Validation Loop

The complete interpretation process becomes:

```text
Train GNN
   ↓
Generate Prediction
   ↓
Explain Prediction
   ↓
Identify Important Atoms
   ↓
Identify Important Edges
   ↓
Identify Important Features
   ↓
Form Structural Hypothesis
   ↓
Construct Counterfactual
   ↓
Recalculate Prediction
   ↓
Validate With DFT / Experiment
```

This is significantly stronger than:

```text
Train GNN
   ↓
Plot Attention
```

because the explanation is connected to a testable scientific workflow.

---

## 26.15.68 Avoiding Explanation Overinterpretation

Several mistakes are common in Materials XAI.

### Mistake 1: Treating attribution as causality

A model attribution does not prove a physical causal relationship.

```text
Attribution
≠
Causality
```

### Mistake 2: Treating attention as importance

Attention is an internal weighting mechanism, not automatically a faithful explanation.

### Mistake 3: Ignoring the training distribution

The model may rely on correlations specific to the dataset.

### Mistake 4: Ignoring graph construction

A model cannot use structural information that was excluded from the graph.

### Mistake 5: Ignoring correlated features

Two chemically related features may receive unstable attribution scores.

### Mistake 6: Interpreting one example as a universal law

An explanation for one crystal does not establish a general materials-science principle.

Therefore:

```text
Model Explanation
       ↓
Hypothesis
       ↓
Independent Validation
```

should always be the preferred workflow.

---

## 26.15.69 Explanation Across a Dataset

A single-crystal explanation is useful for case studies.

However, scientific conclusions generally require population-level analysis.

Suppose we explain:

```text
10,000 crystals
```

and repeatedly observe:

```text
Metal–O interactions
```

receiving high attribution.

We can then calculate:

```text
Frequency of important motif
```

across the dataset.

For example:

```python
motif_counts = {}

for explanation in explanations:

    motifs = extract_motifs(
        explanation
    )

    for motif in motifs:

        motif_counts[motif] = (
            motif_counts.get(
                motif,
                0
            )
            + 1
        )
```

This allows the researcher to identify recurring explanatory patterns.

---

## 26.15.70 Global Explanation From Local Explanations

This leads to an important distinction:

```text
Local explanation
```

answers:

> Why did the model make this prediction for this crystal?

while:

```text
Global explanation
```

asks:

> What structural patterns does the model generally use across the dataset?

A practical workflow is:

```text
Many local explanations
        ↓
Aggregate
        ↓
Cluster explanation patterns
        ↓
Identify recurring motifs
        ↓
Global scientific interpretation
```

This is particularly powerful for large crystal datasets.

---

## 26.15.71 Explanation Clustering

Suppose every crystal produces a vector:

```text
E =
[
edge_1,
edge_2,
...,
edge_n
]
```

or a structural explanation descriptor such as:

```text
Fe-O importance
Fe-Fe importance
coordination importance
distance importance
```

These explanation vectors can be clustered.

For example:

```python
from sklearn.cluster import KMeans

clusters = KMeans(
    n_clusters=4,
    random_state=42
)

labels = clusters.fit_predict(
    explanation_matrix
)
```

The resulting groups may correspond to different model reasoning patterns.

For example:

```text
Cluster 1 → metal–oxygen dominated
Cluster 2 → metal–metal dominated
Cluster 3 → mixed coordination
Cluster 4 → long-range interactions
```

These clusters can then be compared with known materials families.

---

## 26.15.72 Explanation-Driven Materials Taxonomy

A particularly interesting possibility is to use explanations to construct a new materials taxonomy.

Traditional classification may use:

```text
Composition
Crystal system
Space group
```

An explanation-based taxonomy may additionally consider:

```text
Model-relevant structural interactions
```

The workflow becomes:

```text
Materials Dataset
       ↓
Train GNN
       ↓
Generate Explanations
       ↓
Extract Important Motifs
       ↓
Cluster Explanation Patterns
       ↓
Materials Families
```

This does not replace conventional materials classification.

Instead, it provides another perspective:

```text
How does the model perceive
structural similarity?
```

---

## 26.15.73 Explanation Consistency Across Similar Crystals

If two crystals are structurally very similar, their explanations should often show some degree of similarity.

Suppose:

```text
Crystal A
```

and:

```text
Crystal B
```

have nearly identical local environments.

If their predictions are similar but explanations are completely different, this may indicate:

```text
Model instability
```

or:

```text
Explanation instability.
```

Therefore, explanation consistency can be evaluated alongside prediction similarity.

This creates another useful diagnostic:

```text
Structural Similarity
        +
Prediction Similarity
        +
Explanation Similarity
```

---

## 26.15.74 Explanation Stability as a Model Diagnostic

A model may have excellent predictive metrics:

```text
MAE = low
R² = high
```

but unstable explanations.

This matters because a scientist may want to use the model for mechanistic interpretation.

Therefore:

```text
Predictive Performance
```

and:

```text
Interpretability Stability
```

should be treated as separate evaluation dimensions.

A research-grade report may therefore include:

```text
Prediction Metrics
+
Explanation Metrics
```

such as:

```text
MAE
RMSE
R²
Attribution stability
Top-edge overlap
Top-node overlap
Explanation sparsity
Perturbation faithfulness
```

---

## 26.15.75 A Research-Grade Integrated XAI Pipeline

The complete workflow developed in this chapter can now be represented as:

```text
                         Materials Dataset
                                ↓
                         Crystal Representation
                                ↓
                         Crystal Graph
                                ↓
                           GNN Training
                                ↓
                         Model Prediction
                                ↓
                 ┌──────────────┼──────────────┐
                 ↓              ↓              ↓
          Feature XAI       Node XAI       Edge XAI
                 ↓              ↓              ↓
          Important        Important      Important
           Features          Atoms        Interactions
                 └──────────────┼──────────────┘
                                ↓
                     Local Environment
                                ↓
                       Structural Motifs
                                ↓
                     Scientific Hypothesis
                                ↓
                     Counterfactual Tests
                                ↓
                     DFT / Experimental Test
                                ↓
                       Scientific Validation
```

This represents the intended role of explainable AI in Materials Informatics.

The model is not treated as an unquestionable scientific authority.

Instead, it becomes a tool for identifying potentially meaningful relationships that can then be investigated using established scientific methods.

---

## 26.15.76 Chapter-Level Perspective

The major concepts developed throughout this chapter can now be connected.

The first question was:

```text
Why should materials ML models be interpretable?
```

This motivated:

```text
Feature Attribution
```

which identifies influential input characteristics.

For neural and graph models, the analysis was extended to:

```text
Attention
```

and:

```text
Saliency
```

For crystal graphs, the explanation was further specialized into:

```text
Node Importance
```

and:

```text
Edge Importance
```

These can then be combined into:

```text
Integrated Crystal-Graph Explanation
```

Finally, explanations can be connected to:

```text
Structural Motifs
+
Counterfactual Testing
+
DFT Validation
+
Experimental Validation
```

The resulting progression is:

```text
Prediction
   ↓
Attribution
   ↓
Structural Explanation
   ↓
Scientific Hypothesis
   ↓
Independent Validation
```

This is the central philosophy of Explainable AI for Materials.

---

## 26.15.77 Final Research Checklist

Before claiming that an XAI result provides scientific insight, the researcher should ask:

```text
[ ] Is the model prediction reliable?

[ ] Is the explanation method appropriate for the architecture?

[ ] Is the baseline scientifically meaningful?

[ ] Are feature scales normalized appropriately?

[ ] Are node and edge attributions clearly defined?

[ ] Are periodic interactions handled correctly?

[ ] Are directed and physical edges distinguished?

[ ] Has explanation stability been tested?

[ ] Have multiple explanation methods been compared?

[ ] Has the explanation been tested through perturbation?

[ ] Are important structural motifs identified?

[ ] Has the interpretation been checked against materials knowledge?

[ ] Has the result been tested across multiple structures?

[ ] Has the observation been distinguished from causality?

[ ] Can the hypothesis be independently tested?

[ ] Has DFT or experimental evidence been considered where appropriate?
```

If these questions can be answered carefully, XAI becomes substantially more than a visualization technique.

It becomes part of a rigorous Materials Informatics research methodology.

---

## 26.15.78 Closing Perspective

Machine learning can discover correlations that are difficult to recognize manually.

But correlation alone is not enough for scientific discovery.

A materials researcher ultimately wants to understand:

```text
Why does the model believe this material
has this property?
```

For conventional tabular models, the answer may involve:

```text
Feature importance
```

For neural networks, it may involve:

```text
Integrated gradients
+
Saliency
```

For attention-based models:

```text
Attention patterns
```

For crystal GNNs:

```text
Important atoms
+
Important interactions
+
Important local environments
```

The final goal is to translate these computational signals into scientifically meaningful hypotheses.

The complete conceptual chain is therefore:

```text
Materials
    ↓
Crystal Representation
    ↓
Machine Learning Model
    ↓
Prediction
    ↓
Explainability
    ↓
Important Features
    ↓
Important Atoms
    ↓
Important Interactions
    ↓
Structural Motifs
    ↓
Scientific Hypothesis
    ↓
DFT / Experiment
    ↓
Validated Scientific Knowledge
```

The most important principle is:

> **Explainable AI should not replace materials science reasoning; it should provide evidence that helps materials scientists formulate and test better hypotheses.**

A trustworthy Materials Informatics workflow therefore maintains a clear distinction between:

```text
What the model predicts
```

and:

```text
What science has independently established.
```

This distinction is essential when machine-learning models are used for research-grade materials discovery.

The chapter has now established the complete explainability framework required to interpret materials machine-learning models, including conventional feature attribution and the specialized interpretation of crystal graph neural networks.

