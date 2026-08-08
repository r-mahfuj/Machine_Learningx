# Chapter 27 — Uncertainty Quantification for Materials Machine Learning

## 27.1 Introduction to Uncertainty Quantification

Machine-learning models used in Materials Informatics are often presented as prediction systems.

A model receives a material representation:

```text
Crystal Structure
       ↓
Features / Graph / Representation
       ↓
Machine-Learning Model
       ↓
Predicted Property
```

For example, a model may predict:

```text
Band gap = 2.14 eV
```

or:

```text
Formation energy = -2.31 eV/atom
```

or:

```text
Bulk modulus = 178 GPa
```

These numerical predictions are useful, but a prediction by itself is incomplete.

A much more scientifically meaningful output is:

```text
Predicted property
        +
Uncertainty associated with prediction
```

For example:

```text
Band gap = 2.14 ± 0.18 eV
```

The second quantity tells us something about how much confidence we should place in the prediction.

This distinction is extremely important in Materials Informatics because machine-learning models are frequently used to predict properties for materials that have not been experimentally measured or explicitly calculated using high-fidelity methods.

The model may therefore encounter structures that are:

```text
Well represented in the training data
```

or:

```text
Only weakly represented in the training data
```

or even:

```text
Very different from the training data
```

A single point prediction does not clearly communicate this difference.

Uncertainty quantification attempts to address this problem.

---

## 27.1.1 Why Point Predictions Are Insufficient

Consider two hypothetical crystals.

A model predicts:

```text
Crystal A:
Band gap = 1.80 eV
```

and:

```text
Crystal B:
Band gap = 1.82 eV
```

If only the predictions are reported, the two results appear similarly reliable.

However, suppose the model's uncertainty estimates are:

```text
Crystal A:
1.80 ± 0.05 eV

Crystal B:
1.82 ± 0.90 eV
```

Now the scientific interpretation is very different.

For Crystal A, the model has relatively low uncertainty.

For Crystal B, the prediction is much less certain.

Therefore:

```text
Prediction alone
        ↓
"What does the model predict?"

Prediction + uncertainty
        ↓
"What does the model predict,
and how uncertain is that prediction?"
```

This distinction becomes especially important when predictions are used to make scientific decisions.

---

## 27.1.2 Prediction Versus Confidence

A machine-learning prediction is an estimate of an unknown quantity.

For a regression problem:

```text
ŷ = model(x)
```

where:

```text
x
```

represents the material input and:

```text
ŷ
```

is the predicted property.

The prediction does not automatically provide a measure of confidence.

For example:

```text
ŷ = 5.2
```

does not tell us whether the model expects the true value to be close to:

```text
5.2 ± 0.1
```

or:

```text
5.2 ± 2.0
```

These represent very different levels of uncertainty.

Therefore, a useful uncertainty-aware model attempts to provide information about the predictive distribution rather than only its central estimate.

Conceptually:

```text
Input Material
      ↓
Model
      ↓
┌──────────────────────┐
│ Prediction           │
│ Uncertainty          │
└──────────────────────┘
      ↓
Scientific Interpretation
```

It is important, however, not to interpret every uncertainty number as a formal probability of correctness.

For example:

```text
uncertainty = 0.2 eV
```

does not automatically mean:

```text
"there is a 95% probability that
the true value lies within ±0.2 eV."
```

That interpretation requires an appropriately defined and calibrated uncertainty model.

This issue will be examined in detail later in the chapter.

---

## 27.1.3 Why Uncertainty Matters in Materials Informatics

Materials Informatics often operates at the intersection of:

```text
Large chemical spaces
+
Limited high-quality data
+
Expensive calculations
+
Expensive experiments
```

A typical materials dataset may contain thousands or millions of structures, but only a fraction may have reliable property labels.

For example:

```text
Possible crystal structures
          ↓
      Very large
          ↓
Known structures
          ↓
      Smaller
          ↓
DFT-calculated structures
          ↓
      Smaller
          ↓
Experimental measurements
          ↓
       Smaller
```

This means that machine-learning models may have to make predictions beyond regions where abundant reliable training data exist.

Uncertainty can help identify predictions that deserve greater caution.

For example:

```text
Candidate A
Prediction = 3.1 eV
Uncertainty = low

Candidate B
Prediction = 3.2 eV
Uncertainty = high
```

Even though Candidate B has a similar predicted value, it may be a less trustworthy prediction.

---

## 27.1.4 Uncertainty-Aware Scientific Decision Making

The purpose of uncertainty quantification is not simply to attach error bars to plots.

Its deeper purpose is to improve scientific decisions.

Suppose a researcher wants to identify materials with:

```text
Band gap > 2 eV
```

A model may produce:

```text
Material A:
Prediction = 2.8 eV
Uncertainty = 0.1 eV

Material B:
Prediction = 2.3 eV
Uncertainty = 0.8 eV
```

Material A is comfortably above the target according to the model.

Material B is much less certain.

An uncertainty-aware workflow can therefore distinguish:

```text
High-confidence prediction
```

from:

```text
High-uncertainty prediction
```

This does not mean that the high-confidence prediction is guaranteed to be correct.

It means that the model has provided a more informative assessment of its predictive uncertainty.

---

## 27.1.5 Examples from DFT and Experimental Materials Data

Uncertainty can enter Materials Informatics workflows through many different sources.

Consider a dataset of DFT-calculated formation energies.

Even if the numerical calculation is deterministic, the result depends on modeling choices such as:

```text
Exchange-correlation functional
Pseudopotential
Basis-set parameters
k-point sampling
Convergence criteria
Structural relaxation
```

Different computational choices can produce different calculated values.

Similarly, experimental measurements can vary because of:

```text
Instrument precision
Sample preparation
Temperature
Composition variation
Measurement protocol
Operator effects
```

Therefore, a materials dataset should not always be treated as if every target value were an exact representation of an absolute physical truth.

A machine-learning model may learn from labels that contain uncertainty or systematic limitations.

This creates an important distinction:

```text
Observed label
        ≠
Perfect physical truth
```

The uncertainty associated with the data and the uncertainty associated with the machine-learning model are related but different concepts.

---

## 27.1.6 A Simple Materials Example

Suppose a dataset contains calculated band gaps:

```text
Material       Band Gap
-----------------------
A              1.10 eV
B              1.45 eV
C              2.20 eV
D              3.10 eV
E              4.05 eV
```

A regression model is trained on these data.

For a new material:

```text
Predicted band gap = 2.60 eV
```

A conventional machine-learning workflow may stop here.

An uncertainty-aware workflow instead produces something such as:

```text
Predicted band gap
=
2.60 eV

Predictive uncertainty
=
0.15 eV
```

The result can then be communicated as:

```text
2.60 ± 0.15 eV
```

provided that the uncertainty measure has a clearly defined interpretation.

The researcher can now ask two separate questions:

```text
1. What value does the model predict?

2. How uncertain is that prediction?
```

These are distinct scientific questions.

---

## 27.1.7 Uncertainty Is Not the Same as Error

This distinction must be established at the beginning of any uncertainty analysis.

Suppose the model predicts:

```text
ŷ = 2.5 eV
```

and the actual reference value is:

```text
y = 2.2 eV
```

The prediction error is:

```text
error = y - ŷ
```

so:

```text
error = 2.2 - 2.5
      = -0.3 eV
```

The absolute error is:

```text
|error| = 0.3 eV
```

This is the realized prediction error.

Uncertainty is different.

An uncertainty estimate might have been:

```text
σ = 0.2 eV
```

before the true value was known.

Therefore:

```text
Error
→ Difference between prediction and observed/reference value

Uncertainty
→ Model's estimated uncertainty about its prediction
```

After the true value becomes available, the error can be calculated.

Before that happens, uncertainty may be used to characterize the prediction.

---

## 27.1.8 Why Error and Uncertainty Must Be Compared

A useful uncertainty estimate should have a meaningful relationship with actual errors.

Suppose a model produces:

```text
Prediction 1:
uncertainty = low
actual error = low

Prediction 2:
uncertainty = high
actual error = high
```

This is desirable behavior.

However, consider:

```text
Prediction 1:
uncertainty = low
actual error = very high
```

This is dangerous.

The model is confidently wrong.

Conversely:

```text
Prediction 2:
uncertainty = very high
actual error = very low
```

The model may be overly cautious.

Therefore, uncertainty estimates must be evaluated rather than blindly trusted.

Later sections will introduce:

```text
Calibration
Coverage
Interval width
Negative log-likelihood
Sharpness
Proper scoring rules
```

to evaluate whether uncertainty estimates are meaningful.

---

## 27.1.9 Uncertainty as a Distribution

A powerful way to think about uncertainty is to move from a single prediction to a predictive distribution.

Instead of:

```text
ŷ = 2.6 eV
```

we may describe the prediction as:

```text
p(y | x)
```

where:

```text
x
```

is the material representation and:

```text
y
```

is the unknown target property.

The predictive distribution describes possible values of the target given the input.

Conceptually:

```text
                 Probability
                     ↑
                     │
                 ┌───┐
               ┌─┘   └─┐
             ┌─┘       └─┐
─────────────┴─────────────┴────────→
                    y
                   2.6
```

The center of the distribution may correspond to the predicted mean.

Its spread represents predictive uncertainty.

A narrow distribution indicates that the model assigns probability to a relatively small range of values.

A broad distribution indicates greater predictive uncertainty.

---

## 27.1.10 Predictive Mean and Variance

A common probabilistic representation uses a mean and variance:

```text
μ(x)
```

and:

```text
σ²(x)
```

The mean represents the central prediction.

The variance represents the spread of the predictive distribution.

For example:

```text
μ = 2.60 eV
σ = 0.15 eV
```

The corresponding variance is:

```text
σ² = 0.0225 eV²
```

The model output can therefore contain both:

```text
Mean
+
Variance
```

rather than only:

```text
Mean
```

This idea will become particularly important when discussing probabilistic regression and heteroscedastic uncertainty.

---

## 27.1.11 Different Sources of Uncertainty

Materials ML uncertainty does not arise from one single mechanism.

A useful conceptual structure is:

```text
Materials Prediction Uncertainty
            │
     ┌──────┴──────┐
     ↓             ↓
Data-related    Model-related
     │             │
     ↓             ↓
Aleatoric      Epistemic
```

However, real materials datasets can contain additional complications such as:

```text
Distribution shift
Representation limitations
Dataset incompleteness
Computational approximation
Experimental variability
```

These will be examined systematically in the next sections.

---

## 27.1.12 From Deterministic ML to Uncertainty-Aware ML

A conventional regression model can be represented as:

```text
x
↓
fθ(x)
↓
ŷ
```

where:

```text
θ
```

represents the model parameters.

An uncertainty-aware model instead aims to characterize:

```text
p(y | x, D)
```

where:

```text
D
```

represents the available training data.

The result is no longer simply:

```text
x → ŷ
```

but:

```text
x
 ↓
Predictive model
 ↓
┌───────────────────┐
│ Mean prediction   │
│ Predictive spread │
└───────────────────┘
```

This framework allows the model to communicate more information about its prediction.

---

## 27.1.13 Materials Informatics Perspective

For a materials researcher, uncertainty quantification should answer practical questions such as:

```text
How reliable is this predicted property?

Is this material well represented by the training data?

Is the model extrapolating into an unfamiliar chemical region?

Is the uncertainty small enough for the intended application?

Does the uncertainty estimate correspond to actual prediction errors?

Should the prediction be treated as preliminary or high-confidence?
```

These questions are especially important when machine-learning predictions are used to guide expensive downstream calculations or experiments.

The goal is therefore not:

```text
"Make the model uncertain."
```

The goal is:

```text
"Estimate uncertainty in a scientifically meaningful,
measurable, and calibrated way."
```

---

## 27.1.14 The Core Principle of the Chapter

The central principle of uncertainty-aware Materials Machine Learning is:

```text
Prediction
+
Uncertainty
+
Calibration
+
Validation
```

A model that produces uncertainty values without evaluating whether those values are meaningful is incomplete.

Similarly, a model with excellent predictive accuracy may still be unsuitable for scientific decision making if it cannot identify when its predictions are unreliable.

Therefore, throughout this chapter, uncertainty will be treated as a complete workflow:

```text
Materials Data
      ↓
Representation
      ↓
Model
      ↓
Prediction
      ↓
Uncertainty Estimation
      ↓
Calibration
      ↓
Evaluation
      ↓
Scientific Interpretation
```

This framework will be developed progressively.

The next section examines the **sources of uncertainty in Materials Data**, including experimental uncertainty, measurement noise, DFT approximation, dataset incompleteness, model uncertainty, distribution shift, and structural and chemical diversity.

## 27.2 Sources of Uncertainty in Materials Data

Uncertainty in Materials Machine Learning does not originate only from the machine-learning algorithm.

It can enter the workflow at almost every stage:

```text
Materials System
      ↓
Measurement / Calculation
      ↓
Dataset
      ↓
Representation
      ↓
Machine-Learning Model
      ↓
Prediction
```

Each stage can introduce a different type of uncertainty.

For example, a material property may be uncertain because:

```text
The experiment is noisy
```

or:

```text
The DFT calculation is an approximation
```

or:

```text
The training dataset is incomplete
```

or:

```text
The model has insufficient knowledge
```

or:

```text
The new material is very different
from the training materials
```

Therefore, uncertainty quantification should begin by understanding **where uncertainty comes from**.

A useful conceptual representation is:

```text
                    Materials Prediction
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Data Uncertainty   Model Uncertainty   Distribution Shift
        │                  │                  │
        ↓                  ↓                  ↓
 Experimentation       Limited model       Unfamiliar
 DFT approximation     knowledge            materials
 Measurement noise    Sparse training      Chemical shift
 Label variation      data coverage         Structural shift
```

These sources are related, but they should not automatically be treated as the same kind of uncertainty.

---

## 27.2.1 Experimental Uncertainty

Experimental measurements are not perfectly exact.

Suppose the band gap of a material is experimentally measured as:

```text
Eg = 2.10 eV
```

A measurement report may also contain an uncertainty such as:

```text
Eg = 2.10 ± 0.05 eV
```

The uncertainty may arise from:

```text
Instrument precision
Sample preparation
Temperature
Measurement method
Calibration
Surface condition
Composition variation
Operator procedure
```

Therefore, the value stored in a machine-learning dataset may itself represent an uncertain observation.

Conceptually:

```text
True physical property
        ↓
Experimental process
        ↓
Measured value
        ↓
Dataset label
```

The measured value is an observation of the physical system, not necessarily an exact representation of an unknowable absolute value.

---

## 27.2.2 Measurement Noise

Measurement noise is one of the simplest sources of uncertainty.

Suppose repeated measurements of a material property produce:

```text
Measurement 1 → 10.2
Measurement 2 → 10.5
Measurement 3 → 10.1
Measurement 4 → 10.4
Measurement 5 → 10.3
```

The values are not identical.

A simple summary might be:

```text
Mean ≈ 10.3
```

with some observed spread around the mean.

This variability can arise even when the underlying material is unchanged.

Therefore:

```text
Observed variation
≠
necessarily a change in the material itself
```

Some variation may instead be caused by the measurement process.

---

## 27.2.3 Noise in Materials Datasets

Machine-learning datasets often combine measurements obtained from different sources.

For example:

```text
Dataset
├── Laboratory A
├── Laboratory B
├── Laboratory C
└── Literature source D
```

Different sources may use different:

```text
Measurement instruments
Experimental conditions
Sample preparation procedures
Data-processing methods
```

As a result, two measurements of nominally the same material may not be perfectly identical.

This creates an important challenge for Materials Informatics:

```text
Same material
      ↓
Different measurement conditions
      ↓
Different reported values
```

A machine-learning model trained on such data may learn both:

```text
Physical relationships
```

and:

```text
Measurement variability
```

---

## 27.2.4 Repeated Measurements

Repeated measurements provide a useful way to characterize experimental uncertainty.

Suppose the same material is measured several times:

```python
measurements = [
    2.10,
    2.14,
    2.08,
    2.12,
    2.11
]
```

The sample mean can be calculated as:

```python
import numpy as np

mean = np.mean(measurements)
std = np.std(
    measurements,
    ddof=1
)

print(mean)
print(std)
```

The resulting standard deviation describes the observed variation among measurements.

However, this should not automatically be interpreted as the complete uncertainty of the physical property.

The experimental protocol, systematic errors, calibration errors, and sampling effects may contribute additional uncertainty.

Therefore:

```text
Observed repeatability
```

is not necessarily equivalent to:

```text
Total measurement uncertainty
```

---

## 27.2.5 Random and Systematic Experimental Effects

Experimental uncertainty can be broadly associated with random and systematic effects.

### Random effects

These produce variation around the underlying value.

Examples include:

```text
Instrument fluctuations
Small environmental changes
Measurement noise
```

Repeated measurements may partially characterize these effects statistically.

### Systematic effects

These introduce a consistent bias.

For example:

```text
Instrument calibration error
```

might shift every measurement upward.

If the true value is:

```text
10.0
```

but the instrument consistently reports:

```text
10.3
```

then repeated measurements may have very small spread while still being systematically biased.

This is important for machine-learning datasets because:

```text
Small variance
```

does not necessarily mean:

```text
High accuracy.
```

---

## 27.2.6 DFT Approximation and Numerical Uncertainty

Materials datasets are frequently generated using Density Functional Theory.

DFT is extremely valuable, but a DFT result is not an exact representation of the real material.

The calculated property depends on the computational methodology.

For example:

```text
DFT calculation
      ↓
Exchange-correlation functional
      ↓
Pseudopotential
      ↓
Basis-set / plane-wave cutoff
      ↓
k-point sampling
      ↓
Convergence criteria
      ↓
Structural relaxation
      ↓
Calculated property
```

Different choices can produce different numerical results.

Therefore, computational materials data also contain uncertainty.

---

## 27.2.7 Exchange-Correlation Functional

One major source of variation is the exchange-correlation functional.

For example, different functionals may produce different predictions for:

```text
Lattice constants
Formation energies
Band gaps
Magnetic properties
Structural energies
```

Consider a hypothetical material:

```text
Functional A → band gap = 1.2 eV
Functional B → band gap = 1.7 eV
Functional C → band gap = 2.0 eV
```

The difference does not necessarily mean that the calculations are numerically unstable.

It reflects the approximation used to describe the electronic structure.

Therefore:

```text
DFT label
=
result produced by a particular computational model
```

rather than:

```text
DFT label
=
exact experimental truth
```

---

## 27.2.8 Numerical Convergence

Even when the theoretical method is fixed, numerical parameters influence the calculated result.

For example:

```text
Plane-wave cutoff
```

may be increased:

```text
400 eV
500 eV
600 eV
700 eV
```

until the property becomes sufficiently stable.

Similarly, the k-point mesh may change:

```text
2 × 2 × 2
4 × 4 × 4
6 × 6 × 6
8 × 8 × 8
```

If the result changes significantly as these parameters are modified, the calculation has not yet achieved the desired numerical convergence.

Therefore, uncertainty can arise from incomplete numerical convergence.

---

## 27.2.9 Structural Relaxation

The predicted property can also depend on whether the structure has been fully relaxed.

Consider:

```text
Initial structure
      ↓
DFT relaxation
      ↓
Optimized structure
      ↓
Property calculation
```

If the initial structure is not at equilibrium, its predicted property may differ from the relaxed structure.

For example:

```text
Initial band gap → 1.8 eV
Relaxed band gap → 2.1 eV
```

A machine-learning dataset that mixes relaxed and unrelaxed structures may therefore contain additional variation.

This is especially important for Materials Informatics because the model may treat:

```text
Structural variation
```

as if it were:

```text
Property variation
```

when the difference actually originates from computational protocol.

---

## 27.2.10 Dataset Incompleteness

Materials datasets rarely cover the entire space of possible materials.

Consider the enormous chemical space:

```text
Elements
   ↓
Compositions
   ↓
Crystal structures
   ↓
Defect configurations
   ↓
Processing conditions
   ↓
Microstructures
   ↓
Properties
```

Only a small portion of this space may be represented in a particular dataset.

For example:

```text
Possible materials
████████████████████████████████████
████████████████████████████████████
████████████████████████████████████

Training data
████████
██████
███
```

The model therefore learns from an incomplete sample of materials space.

---

## 27.2.11 Sparse Regions of Materials Space

Suppose a dataset contains many oxides:

```text
Oxide A
Oxide B
Oxide C
...
Oxide N
```

but only a few sulfides:

```text
Sulfide A
Sulfide B
```

A model may have considerably more information about the oxide region than the sulfide region.

Therefore:

```text
Dense training region
→ more observed examples

Sparse training region
→ fewer observed examples
```

The uncertainty of predictions can differ between these regions.

This idea becomes especially important when discussing epistemic uncertainty.

---

## 27.2.12 Missing Chemical Systems

A particularly severe form of incompleteness occurs when an entire chemical system is absent.

Suppose the training dataset contains:

```text
Li-Fe-O
Na-Fe-O
Li-Co-O
```

but no:

```text
Na-Co-O
```

If the model is then asked to predict properties of Na-Co-O compounds, it may be extrapolating into a less familiar region.

The model may still produce a numerical prediction.

However:

```text
Numerical prediction
≠
Evidence that the model understands the new chemical system.
```

This distinction becomes central to uncertainty-aware materials screening.

---

## 27.2.13 Dataset Bias

A dataset can be incomplete in a non-random way.

For example, experimentally measured materials may disproportionately include:

```text
Well-known compounds
Stable compounds
Easy-to-synthesize materials
Historically important materials
Materials with interesting properties
```

As a result, the dataset may not represent the full materials space.

This creates:

```text
Sampling bias
```

which can influence both predictions and uncertainty estimates.

A model trained on a biased dataset may appear highly confident within the dataset while performing poorly on underrepresented materials.

---

## 27.2.14 Model Uncertainty

The machine-learning model itself can be uncertain.

Suppose two models are trained on the same dataset:

```text
Model A → 2.10 eV
Model B → 2.15 eV
Model C → 2.07 eV
```

Their predictions are close.

Now consider another material:

```text
Model A → 1.2 eV
Model B → 2.1 eV
Model C → 3.0 eV
```

The models disagree strongly.

This disagreement can indicate that the available training data do not sufficiently constrain the model's prediction in that region.

This type of uncertainty is closely related to what is commonly called:

```text
Epistemic uncertainty
```

which will be examined in detail later.

---

## 27.2.15 Model Architecture and Optimization

Different machine-learning models can learn different functions from the same dataset.

For example:

```text
Linear Regression
Random Forest
Gradient Boosting
Neural Network
Graph Neural Network
```

may produce different predictions.

Even within the same architecture, different:

```text
Initializations
Training procedures
Hyperparameters
```

can lead to different learned models.

Therefore, model uncertainty can arise from limited information about the mapping between inputs and outputs.

This is different from measurement noise.

Measurement noise concerns the data-generating or observation process.

Model uncertainty concerns the model's knowledge of the underlying relationship.

---

## 27.2.16 Distribution Shift

A machine-learning model assumes, explicitly or implicitly, that future inputs are sufficiently related to the data used for training.

Let:

```text
Training distribution = P_train(x)
```

and:

```text
Test distribution = P_test(x)
```

If:

```text
P_train(x) ≈ P_test(x)
```

the model operates under a relatively familiar distribution.

If:

```text
P_train(x) ≠ P_test(x)
```

the model experiences distribution shift.

Conceptually:

```text
Training Materials
████████████████
████████████████

Test Material
                 ███████
                 ███████
```

The test material may occupy a region that was poorly represented during training.

---

## 27.2.17 Chemical Distribution Shift

In Materials Informatics, distribution shift can occur chemically.

For example, a training dataset may contain:

```text
Fe
Co
Ni
Mn
```

while the test set contains a high fraction of:

```text
Ru
Rh
Ir
```

The new elements may have different:

```text
Atomic properties
Electronic structures
Bonding behavior
Coordination preferences
```

The model may therefore be asked to extrapolate beyond the chemical region it learned from.

This is an important source of predictive uncertainty.

---

## 27.2.18 Structural Distribution Shift

Distribution shift can also occur in crystal structures.

Suppose the training data contain mostly:

```text
Cubic
Tetragonal
Orthorhombic
```

structures.

The test data may contain many:

```text
Highly distorted
Low-symmetry
Complex
Large-unit-cell
```

structures.

Even if the chemical elements are familiar, the structural representation may be unfamiliar.

Therefore:

```text
Chemical familiarity
```

does not guarantee:

```text
Structural familiarity.
```

A model may have low uncertainty for known chemistry but high uncertainty for unusual crystal structures.

---

## 27.2.19 Property-Space Extrapolation

Distribution shift can also be considered in property space.

Suppose a training dataset contains formation energies primarily between:

```text
-5 eV/atom
and
0 eV/atom
```

A new material may correspond to a region that is poorly represented by the training examples.

Similarly, the relationship between descriptors and properties may change outside the observed range.

This is particularly important when predicting extreme properties.

For example:

```text
Training:
band gap = 0.5–4 eV

Prediction:
band gap = 8 eV
```

The model may be extrapolating strongly.

The numerical output may still look perfectly reasonable.

Therefore, the magnitude of the prediction alone cannot establish whether the model is operating within its learned domain.

---

## 27.2.20 Structural and Chemical Diversity

Materials are inherently diverse.

A dataset can vary in:

```text
Composition
Crystal system
Space group
Unit-cell size
Coordination environment
Bonding
Defects
Magnetic state
Electronic structure
```

This diversity creates a fundamental challenge for uncertainty estimation.

Two materials with similar compositions may have very different structures.

Conversely, two structurally similar materials may have different chemical compositions.

Therefore, uncertainty should ideally reflect the relevant dimensions of materials similarity.

A simple distance in composition space may not be sufficient.

---

## 27.2.21 Composition Similarity Does Not Guarantee Structural Similarity

Consider:

```text
Material A:
Fe₂O₃
```

and:

```text
Material B:
Fe₂O₃
```

They have identical composition but could represent different structural arrangements or phases.

Their properties can therefore differ.

This illustrates:

```text
Same composition
≠
same crystal structure
```

For crystal-property prediction, the model must therefore account for structural information.

Uncertainty analysis should similarly consider whether the new structure is represented by similar structures in the training data.

---

## 27.2.22 Structural Similarity Does Not Guarantee Identical Properties

The reverse situation can also occur.

Two materials may have similar structural motifs but differ chemically.

For example:

```text
Structure A
→ Fe–O environment

Structure B
→ Co–O environment
```

The local geometry may be similar, but electronic and magnetic properties may differ significantly.

Therefore:

```text
Structural similarity
+
Chemical similarity
```

are both relevant when considering the familiarity of a new material.

---

## 27.2.23 Representation-Dependent Uncertainty

The model does not directly see the physical material.

It sees a representation.

For example:

```text
Crystal
   ↓
Pymatgen
   ↓
Composition / Structure
   ↓
Descriptor Vector
```

or:

```text
Crystal
   ↓
Crystal Graph
   ↓
GNN
```

Different representations preserve different information.

A representation may omit:

```text
Defects
Magnetic ordering
Disorder
Long-range interactions
Processing history
```

If the omitted information is important for the target property, the model may have uncertainty that is not obvious from the numerical feature vector.

Therefore, uncertainty is partly connected to how the material is represented.

---

## 27.2.24 Missing Structural Information

Consider a model trained using ideal crystal structures.

A real material may contain:

```text
Vacancies
Substitutions
Disorder
Defects
Strain
Grain boundaries
```

If these are absent from the representation, the model is predicting from an incomplete description of the physical system.

For example:

```text
Ideal crystal
     ↓
ML model
     ↓
Predicted property
```

may ignore:

```text
Real-world defects
```

that strongly influence the experimentally measured property.

This creates a distinction between:

```text
Model uncertainty
```

and:

```text
Uncertainty caused by incomplete physical representation.
```

Both can matter in Materials Informatics.

---

## 27.2.25 DFT Data Versus Experimental Data

A Materials Informatics dataset may combine:

```text
DFT values
+
Experimental values
```

This requires care.

Suppose:

```text
DFT band gap = 1.8 eV
Experimental band gap = 2.4 eV
```

The difference does not necessarily mean that one of the values is simply incorrect.

The methods may be measuring or predicting related but not identical quantities under different assumptions.

Therefore, combining data sources without considering their provenance can introduce additional uncertainty.

A research dataset should ideally retain metadata such as:

```text
Material ID
Composition
Structure
Property
Measurement / calculation method
Source
Computational settings
Experimental conditions
```

This information can later help identify uncertainty sources.

---

## 27.2.26 Label Noise in Supervised Learning

Suppose a model is trained on:

```text
(x₁, y₁)
(x₂, y₂)
...
(xₙ, yₙ)
```

The labels:

```text
yᵢ
```

may contain noise.

For example:

```text
True underlying property
        +
measurement error
        ↓
Observed label
```

A model cannot necessarily distinguish the physical signal from the noise unless the dataset and model formulation provide enough information.

Therefore, uncertainty can be connected directly to the target variable.

This becomes particularly important when the uncertainty varies between samples.

For example:

```text
Material A:
measurement uncertainty = 0.02 eV

Material B:
measurement uncertainty = 0.30 eV
```

A model may need to account for this difference.

This leads naturally to the concept of **aleatoric uncertainty**, which is discussed in Section 27.3.

---

## 27.2.27 Heterogeneous Data Quality

Not all materials in a dataset necessarily have equally reliable labels.

For example:

```text
Material A
High-quality experimental measurement

Material B
Approximate literature value

Material C
Low-convergence DFT calculation

Material D
High-quality converged DFT calculation
```

If all are treated identically, the model may not know that some targets are more uncertain than others.

Therefore, dataset construction should consider data quality whenever possible.

A useful conceptual representation is:

```text
Material
   ↓
Property value
   +
Data-quality information
```

This does not automatically solve the uncertainty problem, but it makes the source of uncertainty more explicit.

---

## 27.2.28 Uncertainty from Structural and Chemical Diversity

Consider two regions of materials space:

```text
Region A

████████████████
████████████████
████████████████
```

and:

```text
Region B

█
        █
             █
   █
```

Region A contains many closely related materials.

Region B contains sparse and diverse examples.

A model trained on Region A may have stronger empirical support for predictions in that region.

In Region B, the model may have less information.

Therefore, structural and chemical diversity affect how well the training data constrain predictions.

---

## 27.2.29 Why More Data Does Not Always Remove All Uncertainty

It is tempting to assume:

```text
More data
   ↓
Zero uncertainty
```

This is not generally correct.

Some uncertainty is irreducible.

For example, if a measurement process contains unavoidable noise:

```text
More training data
```

may help the model learn the underlying trend, but it cannot necessarily eliminate the variability inherent in the measurement process.

This distinction leads to the fundamental separation between:

```text
Aleatoric uncertainty
```

and:

```text
Epistemic uncertainty.
```

The former is associated with variability that may remain even with abundant data.

The latter is associated with limited knowledge and can often decrease as relevant information becomes available.

---

## 27.2.30 A Unified View of Materials Uncertainty

The major sources discussed so far can be organized as:

```text
                         Materials ML
                           Prediction
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
   Data Sources          Model Knowledge      Distribution
        │                     │                  Shift
        ↓                     ↓                     ↓
Experimental          Sparse training       Chemical shift
noise                  regions              Structural shift
Measurement            Rare materials       Property extrapolation
uncertainty             Model limitations
DFT approximation
Dataset incompleteness
```

Another useful classification is:

```text
Uncertainty
│
├── Data-related
│   ├── Measurement noise
│   ├── Experimental variability
│   ├── DFT approximation
│   └── Label uncertainty
│
├── Knowledge-related
│   ├── Sparse training regions
│   ├── Rare chemical systems
│   └── Unseen structures
│
└── Distribution-related
    ├── Chemical shift
    ├── Structural shift
    └── Property-space extrapolation
```

This classification provides the foundation for the next sections.

---

## 27.2.31 Practical Example: Band-Gap Dataset

Consider a dataset used to predict band gaps.

The dataset contains:

```text
Crystal Structure
Composition
Calculated Band Gap
```

Several uncertainty sources may exist simultaneously.

### Source 1: DFT approximation

The calculated band gap depends on the chosen computational method.

### Source 2: Numerical convergence

Insufficient convergence can introduce numerical variation.

### Source 3: Dataset incompleteness

Some chemical systems may be poorly represented.

### Source 4: Structural diversity

The test material may have a structure rarely encountered during training.

### Source 5: Model uncertainty

Different trained models may produce substantially different predictions.

### Source 6: Distribution shift

The new material may occupy a chemical region absent from the training set.

Therefore, a single uncertainty number may actually summarize multiple underlying phenomena.

This is why the next sections distinguish different forms of uncertainty rather than treating all uncertainty as one quantity.

---

## 27.2.32 Practical Example: Experimental Elastic Modulus

Suppose a model predicts Young's modulus from crystal descriptors.

The experimental labels may vary because of:

```text
Sample porosity
Grain structure
Measurement direction
Temperature
Testing method
```

A crystal-structure-only model may not contain all of this information.

Therefore:

```text
Crystal representation
        ↓
Model
        ↓
Prediction
```

cannot completely capture every factor affecting the experimental measurement.

The resulting uncertainty may therefore arise from both:

```text
Measurement variability
```

and:

```text
Incomplete representation of the physical system.
```

This example illustrates why uncertainty analysis must consider the entire materials-data pipeline.

---

## 27.2.33 Data Provenance and Uncertainty

For research-grade Materials Informatics, uncertainty analysis should be connected to data provenance.

A useful record may include:

```text
Structure ID
Composition
Property value
Property unit
Experimental / DFT
Calculation method
Functional
Pseudopotential
Convergence settings
Experimental temperature
Experimental method
Source
```

The exact metadata depend on the dataset.

The important principle is:

```text
Data provenance
        ↓
Understanding uncertainty sources
```

Without provenance, it can be difficult to determine why apparently similar materials have different target values.

---

## 27.2.34 Why Uncertainty Must Be Treated Explicitly

If uncertainty is ignored, a Materials ML workflow may become:

```text
Dataset
   ↓
Model
   ↓
Prediction
   ↓
Candidate
```

An uncertainty-aware workflow becomes:

```text
Dataset
   ↓
Model
   ↓
Prediction
   +
Uncertainty
   ↓
Calibration / Validation
   ↓
Scientific Interpretation
```

The second workflow provides more information for evaluating whether a prediction should be trusted.

However, uncertainty itself must later be validated.

A model that simply outputs large uncertainty for every material is not useful.

Likewise, a model that outputs extremely small uncertainty for every material is not necessarily reliable.

The uncertainty must correspond meaningfully to observed predictive behavior.

---

## 27.2.35 Transition to Aleatoric Uncertainty

The sources discussed in this section can now be divided into an important conceptual distinction.

Some uncertainty arises from variability that is inherent to the data-generating process.

Examples include:

```text
Measurement noise
Experimental variability
Sample-dependent noise
```

This is commonly associated with:

```text
Aleatoric uncertainty
```

Other uncertainty arises because the model does not have sufficient knowledge.

Examples include:

```text
Sparse training regions
Rare chemical systems
Unseen crystal structures
Limited training coverage
```

This is commonly associated with:

```text
Epistemic uncertainty
```

The distinction is:

```text
Materials uncertainty
        │
        ├───────────────┐
        ↓               ↓
   Aleatoric        Epistemic
   variability      model knowledge
```

The next section, **27.3 Aleatoric Uncertainty**, will examine this first category in detail, including measurement noise, irreducible uncertainty, homoscedastic uncertainty, heteroscedastic uncertainty, materials examples, and models that predict both mean and variance.

## 27.3 Aleatoric Uncertainty

Aleatoric uncertainty describes uncertainty associated with the inherent variability of the data-generating or measurement process.

The central idea is:

```text
Same input
+
Unavoidable variability
        ↓
Different possible observations
```

In Materials Informatics, this can occur when the same nominal material does not always produce exactly the same measured property.

For example, suppose the Young's modulus of a material is measured repeatedly:

```text
Measurement 1 → 201 GPa
Measurement 2 → 198 GPa
Measurement 3 → 204 GPa
Measurement 4 → 200 GPa
Measurement 5 → 202 GPa
```

The variation may arise from experimental conditions, sample variability, measurement noise, or other factors that cannot be completely eliminated.

A machine-learning model trained on such data should not necessarily be expected to predict one perfectly deterministic value.

Instead, the model can attempt to learn both:

```text
Expected property value
+
Magnitude of inherent variability
```

This is the fundamental idea behind aleatoric uncertainty.

---

## 27.3.1 Definition of Aleatoric Uncertainty

Consider a material represented by:

```text
x
```

and its measured property:

```text
y
```

A deterministic model assumes approximately:

```text
y = f(x)
```

This implies that a particular input corresponds to one fixed output.

An uncertainty-aware formulation instead recognizes that:

```text
y
```

may vary even when:

```text
x
```

is unchanged.

Therefore, a more appropriate description is:

```text
y ~ p(y | x)
```

The model describes a conditional distribution of possible target values.

For example:

```text
Material
   ↓
Representation x
   ↓
┌──────────────────────┐
│ Predictive           │
│ distribution         │
└──────────────────────┘
   ↓
Mean + Variability
```

Aleatoric uncertainty is therefore associated with the spread of possible observations conditioned on the same input.

---

## 27.3.2 Irreducible Uncertainty

Aleatoric uncertainty is often described as **irreducible uncertainty**.

This does not mean that every source of experimental variation is fundamentally impossible to reduce.

Rather, it means that once the available information is fixed, some variability remains in the target.

For example, suppose a model uses only:

```text
Crystal structure
Composition
```

to predict an experimentally measured property.

The actual property may also depend on:

```text
Temperature
Defect concentration
Microstructure
Processing history
Measurement conditions
```

If these variables are not available to the model, the same input representation may correspond to multiple observed outcomes.

Conceptually:

```text
Same representation
        │
        ├──→ Property value 1
        ├──→ Property value 2
        ├──→ Property value 3
        └──→ Property value 4
```

The model cannot perfectly predict which outcome will occur without additional information.

Thus, the uncertainty associated with the unobserved variability becomes part of the predictive uncertainty.

---

## 27.3.3 Materials Example: Experimental Measurements

Consider a hypothetical experiment measuring the thermal conductivity of a material.

Five measurements produce:

```text
Material X

Measurement 1 → 12.1 W/mK
Measurement 2 → 11.8 W/mK
Measurement 3 → 12.4 W/mK
Measurement 4 → 12.0 W/mK
Measurement 5 → 12.2 W/mK
```

The mean is approximately:

```text
12.1 W/mK
```

but the material does not always produce exactly:

```text
12.1 W/mK
```

in every measurement.

A probabilistic model could therefore represent the property as:

```text
Thermal conductivity
≈
12.1 W/mK
with some associated spread
```

The spread represents uncertainty in possible observations.

---

## 27.3.4 Measurement Noise

Measurement noise is one common contributor to aleatoric uncertainty.

Suppose a measurement process can be represented as:

```text
Observed value
=
Underlying value
+
Measurement noise
```

Mathematically:

```text
y = f(x) + ε
```

where:

```text
f(x)
```

represents the underlying relationship and:

```text
ε
```

represents random noise.

A common simple assumption is:

```text
ε ~ N(0, σ²)
```

where:

```text
σ²
```

represents the noise variance.

Under this assumption:

```text
y | x ~ N(f(x), σ²)
```

The important point is that the uncertainty belongs to the target-generating process, not necessarily to uncertainty about the model parameters.

---

## 27.3.5 Homoscedastic Uncertainty

The simplest case is **homoscedastic uncertainty**.

Here, the noise level is assumed to be approximately constant across the dataset.

Mathematically:

```text
σ²(x) = σ²
```

The variance does not depend on the material.

For example:

```text
Material A → σ = 0.10
Material B → σ = 0.10
Material C → σ = 0.10
Material D → σ = 0.10
```

The model assumes approximately the same observation noise for every input.

Conceptually:

```text
Input space
──────────────────────────────→

Uncertainty
    │
    │    ───────────────────
    │
    └────────────────────────→
```

This assumption can be useful when the measurement process has approximately uniform noise.

However, materials datasets often violate this assumption.

---

## 27.3.6 Example of Homoscedastic Materials Data

Suppose a property is measured using a highly standardized experimental procedure.

For a simplified example:

```text
Material      Property
----------------------
A             10.2
B             11.1
C             12.4
D             13.0
E             14.2
```

Suppose the measurement process has approximately:

```text
σ = 0.2
```

for all materials.

Then a probabilistic model might represent:

```text
y | x ~ N(μ(x), 0.2²)
```

The model predicts the mean:

```text
μ(x)
```

while the measurement uncertainty remains approximately fixed.

This is a useful baseline, but it may be too restrictive for many real materials datasets.

---

## 27.3.7 Heteroscedastic Uncertainty

In many scientific datasets, the uncertainty changes from one sample to another.

This is called **heteroscedastic uncertainty**.

Instead of:

```text
σ²(x) = constant
```

we have:

```text
σ²(x) = function of x
```

or:

```text
σ² = σ²(x)
```

The model therefore predicts both:

```text
Mean
+
Input-dependent variance
```

Conceptually:

```text
Input A
   ↓
Small uncertainty

Input B
   ↓
Large uncertainty

Input C
   ↓
Medium uncertainty
```

This is particularly relevant to Materials Informatics because different materials may have very different levels of measurement variability.

---

## 27.3.8 Why Materials Data Can Be Heteroscedastic

Consider experimental measurements of a material property.

Some materials may have:

```text
Highly reproducible measurements
```

while others may be strongly affected by:

```text
Defects
Porosity
Phase composition
Microstructure
Sample preparation
Temperature
Measurement direction
```

Therefore:

```text
Material A → small variability
Material B → large variability
```

is entirely plausible.

A single global noise value would fail to represent this difference.

---

## 27.3.9 Heteroscedasticity in Experimental Materials Data

Consider two groups of materials.

### Group A

```text
Material A1 → 100.1
Material A2 → 100.2
Material A3 → 100.0
Material A4 → 100.1
```

The observations are tightly clustered.

### Group B

```text
Material B1 → 98
Material B2 → 104
Material B3 → 101
Material B4 → 107
```

The observations are much more variable.

A model that assumes:

```text
σ_A = σ_B
```

would ignore this difference.

A heteroscedastic model instead attempts to learn:

```text
σ_A < σ_B
```

when the data support such a relationship.

---

## 27.3.10 Modeling the Conditional Distribution

For a Gaussian predictive model, the target can be represented as:

```text
y | x ~ N(μ(x), σ²(x))
```

The model therefore needs to learn two functions:

```text
μ(x)
```

and:

```text
σ²(x)
```

The first describes the expected property.

The second describes the variability around that expectation.

Conceptually:

```text
                 Input x
                    ↓
             Neural Network
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
       μ(x)                σ²(x)
          │                   │
          └─────────┬─────────┘
                    ↓
          Predictive Distribution
```

This is one of the most important practical formulations of aleatoric uncertainty.

---

## 27.3.11 Why Variance Must Remain Positive

A variance cannot be negative.

Therefore, a neural network should not usually predict variance directly without a suitable transformation.

Suppose the network produces:

```python
raw_variance
```

This value could be positive or negative.

Instead, a positive transformation can be applied.

A common approach is:

```python
import torch

variance = torch.nn.functional.softplus(
    raw_variance
)
```

The softplus transformation produces positive values.

Another common approach is to predict:

```text
log variance
```

and recover the variance through:

```python
variance = torch.exp(
    log_variance
)
```

This can be numerically convenient, although appropriate stabilization should be used in practical implementations.

---

## 27.3.12 Predicting Mean and Log Variance with PyTorch

A simple neural network can output two quantities:

```text
Output 1 → predicted mean
Output 2 → predicted log variance
```

For example:

```python
import torch
import torch.nn as nn

class ProbabilisticRegressor(nn.Module):

    def __init__(
        self,
        input_dim
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                64
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        output = self.network(x)

        mean = output[:, 0]

        log_variance = output[:, 1]

        return mean, log_variance
```

The two outputs have different meanings:

```text
mean
→ predicted property

log_variance
→ predicted aleatoric uncertainty
```

The variance can be recovered as:

```python
variance = torch.exp(
    log_variance
)
```

and the standard deviation as:

```python
std = torch.sqrt(
    variance
)
```

---

## 27.3.13 Gaussian Negative Log-Likelihood

If the model predicts both mean and variance, ordinary mean-squared error is no longer sufficient as the complete probabilistic objective.

A common loss is the Gaussian negative log-likelihood.

For a single observation:

```text
y
```

with predicted:

```text
μ
```

and:

```text
σ²
```

the negative log-likelihood is:

```text
L =
1/2 [
log(σ²)
+
(y - μ)² / σ²
]
```

up to constants that do not depend on the model parameters.

The loss contains two important terms.

The first is:

```text
log(σ²)
```

which discourages the model from simply making the uncertainty arbitrarily large.

The second is:

```text
(y - μ)² / σ²
```

which penalizes prediction errors relative to the predicted uncertainty.

Thus, the model must balance:

```text
Prediction accuracy
```

and:

```text
Uncertainty magnitude
```

---

## 27.3.14 PyTorch Implementation of Gaussian NLL

A simple implementation is:

```python
def gaussian_nll(
    mean,
    log_variance,
    target
):

    variance = torch.exp(
        log_variance
    )

    loss = 0.5 * (
        log_variance
        +
        (target - mean) ** 2
        / variance
    )

    return loss.mean()
```

Training can then be performed as:

```python
model = ProbabilisticRegressor(
    input_dim=20
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(1000):

    optimizer.zero_grad()

    mean, log_variance = model(
        X_train
    )

    loss = gaussian_nll(
        mean,
        log_variance,
        y_train
    )

    loss.backward()

    optimizer.step()
```

After training:

```python
mean, log_variance = model(
    X_test
)

variance = torch.exp(
    log_variance
)

std = torch.sqrt(
    variance
)
```

The model now provides:

```text
Predicted mean
+
Predicted standard deviation
```

for each material.

---

## 27.3.15 Interpreting the Output

Suppose the model produces:

```text
Material A
Mean = 2.40 eV
Std  = 0.08 eV

Material B
Mean = 2.50 eV
Std  = 0.45 eV
```

The predictions are:

```text
A → 2.40 eV
B → 2.50 eV
```

but their predicted variability is very different.

Material A has relatively small aleatoric uncertainty.

Material B has substantially larger predicted variability.

The important interpretation is not:

```text
Material B is a worse prediction.
```

Rather:

```text
The target associated with Material B
is predicted to have greater inherent variability
under the learned conditional model.
```

This distinction is important.

Aleatoric uncertainty does not necessarily indicate that the model is ignorant.

It may indicate that the target itself is variable or noisy given the available input information.

---

## 27.3.16 Aleatoric Uncertainty and Missing Variables

Suppose the true property depends on:

```text
y = f(
composition,
structure,
temperature,
defects,
processing
)
```

but the model receives only:

```text
x =
composition,
structure
```

Then several different physical conditions may correspond to the same model input.

Conceptually:

```text
Composition + Structure
          │
     ┌────┼────┐
     ↓    ↓    ↓
  Temp  Defect Processing
     │    │      │
     └────┼──────┘
          ↓
       Property
```

If these additional variables are not available, their effects may appear as conditional variability in the target.

Thus, heteroscedastic aleatoric uncertainty can sometimes indicate that the available representation does not contain all variables controlling the observed property.

This is an important scientific interpretation.

---

## 27.3.17 Aleatoric Uncertainty in DFT-Derived Data

Aleatoric uncertainty can also arise in computational datasets, although its interpretation differs from experimental noise.

Suppose a dataset contains formation energies calculated under different computational conditions.

If those conditions introduce systematic differences, then the observed target values may have additional variability.

For example:

```text
Structure
   ↓
DFT calculation
   ↓
Different computational settings
   ↓
Different numerical results
```

If the dataset combines these results without recording the computational settings, the model may observe unexplained target variability.

Therefore, data provenance is particularly important when modeling computational materials data.

---

## 27.3.18 Aleatoric Uncertainty Does Not Mean Random Guessing

A probabilistic model should not be interpreted as saying:

```text
"The material property is completely random."
```

Instead, the model assumes that:

```text
Given the available information,
multiple outcomes remain plausible.
```

For example:

```text
Crystal representation
        ↓
Expected band gap = 2.1 eV
        +
Some unavoidable variation
```

The model still learns a structured relationship.

The uncertainty describes the spread around that relationship.

Thus:

```text
Structured prediction
+
Conditional variability
```

is a better interpretation than:

```text
Random prediction.
```

---

## 27.3.19 Homoscedastic Versus Heteroscedastic Models

The difference can be summarized as follows.

### Homoscedastic model

```text
y | x ~ N(
    μ(x),
    σ²
)
```

The variance is constant.

### Heteroscedastic model

```text
y | x ~ N(
    μ(x),
    σ²(x)
)
```

The variance depends on the input.

Conceptually:

```text
Homoscedastic

Prediction
   │
   │   ───────
   │  /       \
   │ /         \
───┴──────────────→ x
```

versus:

```text
Heteroscedastic

Prediction
   │
   │  ──      ───────
   │ /  \    /       \
   │/    \__/         \
───┴────────────────────→ x
```

The second model can represent situations where some materials are inherently easier to predict precisely than others.

---

## 27.3.20 When Heteroscedastic Modeling Is Useful

Heteroscedastic uncertainty is particularly useful when uncertainty changes systematically with:

```text
Composition
Structure
Property magnitude
Measurement conditions
Material class
```

For example, suppose experimental error becomes larger for very high-temperature measurements.

Then:

```text
Low temperature
→ small uncertainty

High temperature
→ larger uncertainty
```

A model with a single global variance cannot represent this behavior.

A heteroscedastic model can potentially learn the relationship between input conditions and expected observation variability.

---

## 27.3.21 Important Limitation

A predicted variance is not automatically a trustworthy uncertainty estimate.

A neural network can learn to output a variance, but that does not guarantee that:

```text
Predicted uncertainty
```

matches:

```text
Observed prediction errors.
```

For example, the model may systematically underestimate uncertainty:

```text
Predicted σ = 0.05 eV
Actual errors ≈ 0.30 eV
```

or overestimate it:

```text
Predicted σ = 0.80 eV
Actual errors ≈ 0.10 eV
```

Therefore:

```text
Predicting variance
```

is only the first step.

The resulting uncertainty must later be evaluated and calibrated.

This is why uncertainty calibration is an essential part of the broader Chapter 27 workflow.

---

## 27.3.22 Aleatoric Uncertainty in Materials Screening

Suppose a researcher wants materials with:

```text
Band gap > 2.0 eV
```

Two candidates produce:

```text
Candidate A:
μ = 2.40 eV
σ = 0.05 eV

Candidate B:
μ = 2.40 eV
σ = 0.50 eV
```

Both have the same predicted mean.

However, their predictive distributions are very different.

Candidate A is tightly concentrated around the target prediction.

Candidate B has a much wider distribution.

Therefore, an uncertainty-aware screening system can distinguish between:

```text
Strongly concentrated prediction
```

and:

```text
Broad prediction distribution.
```

The researcher can then decide how much confidence is appropriate for the intended scientific application.

This is uncertainty-aware screening, not merely point prediction.

---

## 27.3.23 Relationship to Prediction Intervals

Suppose the model assumes:

```text
y | x ~ N(μ, σ²)
```

Then an interval around the mean can be constructed using the predicted standard deviation.

For example, a commonly used approximate interval is:

```text
μ ± 1.96σ
```

under the appropriate Gaussian assumptions.

If:

```text
μ = 2.40 eV
```

and:

```text
σ = 0.10 eV
```

then:

```text
2.40 ± 0.196 eV
```

gives approximately:

```text
[2.204, 2.596] eV
```

However, the interpretation of such an interval depends critically on whether the predictive distribution is correctly specified and calibrated.

It should therefore not automatically be called a guaranteed 95% prediction interval.

Calibration will be discussed later in the chapter.

---

## 27.3.24 Scientific Interpretation of Aleatoric Uncertainty

A materials researcher should ask:

```text
Why is this material associated with larger variability?
```

Possible explanations include:

```text
Defect sensitivity
Microstructural variability
Measurement instability
Phase mixtures
Temperature dependence
Composition variation
Incomplete representation
```

Therefore, aleatoric uncertainty can sometimes provide scientific information.

For example, if a model consistently predicts larger uncertainty for materials with mixed phases, this may suggest that the property is strongly sensitive to structural or compositional variability.

However, such interpretations require independent scientific validation.

The uncertainty value itself does not prove the underlying mechanism.

---

## 27.3.25 Key Concepts of Aleatoric Uncertainty

The main ideas can be summarized as:

```text
Aleatoric uncertainty
        ↓
Variability in possible observations
        ↓
Often associated with noise or
inherent conditional variability
        ↓
May remain even with more training data
        ↓
Can be modeled using predictive distributions
```

Two important forms are:

```text
Homoscedastic
→ approximately constant variance

Heteroscedastic
→ input-dependent variance
```

A probabilistic regression model can therefore produce:

```text
Mean prediction
+
Predicted variance
```

rather than only a point prediction.

---

## 27.3.26 Transition to Epistemic Uncertainty

Aleatoric uncertainty describes variability associated with the data-generating process.

The next question is different:

```text
What happens when the model itself
does not have enough knowledge?
```

Consider a crystal belonging to a chemical system that is almost completely absent from the training dataset.

The target may not necessarily be inherently noisy.

Instead, the model may simply lack sufficient information to make a reliable prediction.

This leads to:

```text
Epistemic uncertainty
```

which is commonly associated with:

```text
Limited training data
Sparse materials space
Unseen chemical systems
Unfamiliar crystal structures
Model uncertainty
```

The distinction can therefore be summarized as:

```text
Aleatoric
→ "The outcome itself has variability."

Epistemic
→ "The model has limited knowledge."
```

The next section, **27.4 Epistemic Uncertainty**, will examine model ignorance, sparse training regions, rare chemical systems, unseen crystal structures, and the relationship between training-data coverage and epistemic uncertainty.

## 27.4 Epistemic Uncertainty

Epistemic uncertainty describes uncertainty caused by **limited knowledge of the model**.

The central idea is:

```text
Limited training information
          ↓
Limited model knowledge
          ↓
Greater uncertainty
```

Unlike aleatoric uncertainty, which describes variability associated with the data-generating process, epistemic uncertainty is associated with what the model has not learned sufficiently well.

For Materials Informatics, this distinction is particularly important because materials datasets are usually sparse compared with the enormous space of possible materials.

A model may have thousands of training examples and still encounter a new crystal that lies far outside the region represented by those examples.

Conceptually:

```text
Training Materials
        ↓
Machine-Learning Model
        ↓
Learned Materials Space
        ↓
New Material
        ↓
Is it familiar?
        │
     ┌──┴──┐
    Yes    No
     ↓      ↓
Lower     Higher
epistemic epistemic
uncertainty uncertainty
```

The model may still produce a numerical prediction for the unfamiliar material.

However, the prediction can be less reliable because the available training information does not strongly constrain the model in that region.

---

## 27.4.1 Definition of Epistemic Uncertainty

Suppose a machine-learning model learns a relationship:

```text
x → y
```

where:

```text
x = materials representation
y = target property
```

The model learns this relationship from a finite training dataset.

If the training dataset is sufficiently informative around a particular material, the model may have relatively strong knowledge of the corresponding prediction.

If the training dataset contains little or no relevant information, many different models may be consistent with the available training observations.

This can be represented conceptually as:

```text
Training data
     ↓
┌────┴────┐
Model A   Model B
Model C   Model D
└────┬────┘
     ↓
Prediction
```

If the models produce similar predictions:

```text
Model A → 2.10
Model B → 2.12
Model C → 2.08
Model D → 2.11
```

the model family is relatively consistent.

If they produce:

```text
Model A → 1.20
Model B → 2.10
Model C → 2.80
Model D → 3.40
```

there is substantial disagreement.

Such disagreement can indicate epistemic uncertainty.

---

## 27.4.2 Model Ignorance

Epistemic uncertainty can be understood as **model ignorance**.

This does not mean that the model is incapable of producing an output.

A trained model will normally produce a prediction for almost any numerical input that can be passed through it.

The important question is:

```text
Does the training data provide sufficient evidence
to support that prediction?
```

For example:

```text
Known region

Training data
████████████████
████████████████
████████████████

New material
      █
```

versus:

```text
Sparse region

Training data
█
       █
             █

New material
                 █
```

The second case is more concerning because the model has little direct training experience in that region.

Therefore:

```text
Prediction exists
```

does not imply:

```text
Prediction is well supported.
```

---

## 27.4.3 Why Epistemic Uncertainty Matters in Materials Informatics

Materials space is extremely large.

A dataset may contain:

```text
Thousands
```

or:

```text
Millions
```

of materials, yet still represent only a small portion of the possible chemical and structural space.

A model trained on such data may encounter:

```text
New composition
New crystal structure
New chemical combination
New coordination environment
New structural motif
```

during prediction.

If these materials are poorly represented in training data, epistemic uncertainty can become important.

This is especially relevant when the model is used for:

```text
Materials screening
Property prediction
Crystal generation
Candidate ranking
```

because the most interesting candidates may be precisely those that lie in poorly explored regions.

---

## 27.4.4 Sparse Training Regions

Consider a simplified one-dimensional materials descriptor.

Suppose training data exist mainly in this region:

```text
x

0    1    2    3    4    5    6    7    8

     ● ● ● ● ● ● ● ●
```

The model has many examples between approximately:

```text
x = 1
```

and:

```text
x = 5
```

Now consider a new material at:

```text
x = 8
```

The model is being asked to predict outside the dense training region.

This is an extrapolation problem.

Conceptually:

```text
Training region
████████████

Sparse / unfamiliar region
            ────────────→
```

Epistemic uncertainty may increase because the model has fewer observations constraining its behavior there.

---

## 27.4.5 Materials Example: Chemical Space

Suppose a model is trained mostly on:

```text
Li–Fe–O
Na–Fe–O
Li–Co–O
Na–Co–O
```

and is then asked to predict:

```text
Li–Ru–O
```

The new system contains an element that may be poorly represented or absent in the training data.

The model may still generate:

```text
Band gap = 2.7 eV
```

but this number should not automatically be treated as equally reliable to a prediction for a well-represented Fe–O material.

The important distinction is:

```text
Well-represented chemistry
→ stronger empirical support

Poorly represented chemistry
→ weaker empirical support
```

The latter can produce greater epistemic uncertainty.

---

## 27.4.6 Rare Chemical Systems

Some chemical systems naturally occur less frequently in a dataset.

For example:

```text
Common systems
████████████████████

Rare systems
██
```

If a model has only a handful of examples for a particular element combination, its learned relationship for that region is less constrained.

Consider:

```text
Chemical system      Training examples

Fe–O                       2,500
Co–O                       1,800
Ni–O                       1,500
Rare system                  12
```

The model has substantially more information about the common systems.

The rare system may therefore exhibit greater epistemic uncertainty.

However, the number of examples alone is not sufficient.

The examples must also be relevant and representative of the new material.

---

## 27.4.7 Training Density

A useful conceptual idea is **training-data density**.

Suppose a material is represented by a feature vector:

```text
x = [x₁, x₂, ..., x_d]
```

The model has many nearby training examples:

```text
x₁ ●
   ●
   ●  New material
   ●
x₂ ●
```

In this situation, the model is operating in a relatively familiar region.

Now consider:

```text
●

             New material

                         ●
```

where nearby training examples are sparse.

The second situation provides less empirical support.

Therefore, training-data density can be related to epistemic uncertainty.

This relationship is especially useful in Materials Informatics because chemical and structural spaces can be highly nonuniform.

---

## 27.4.8 Epistemic Uncertainty and Model Disagreement

One practical way to estimate epistemic uncertainty is to train multiple models.

Suppose:

```text
Model 1 → y₁
Model 2 → y₂
Model 3 → y₃
...
Model M → yM
```

The mean prediction is:

```text
μ = average(y₁, y₂, ..., yM)
```

and the variation among predictions can be used as an uncertainty estimate.

For example:

```text
Material A

Model 1 → 2.10
Model 2 → 2.13
Model 3 → 2.08
Model 4 → 2.11
```

The predictions are tightly clustered.

Now:

```text
Material B

Model 1 → 1.20
Model 2 → 2.10
Model 3 → 2.80
Model 4 → 3.00
```

The models strongly disagree.

The second material therefore has a much larger model-disagreement signal.

This idea forms the basis of several practical uncertainty-estimation methods discussed later in the chapter.

---

## 27.4.9 Why Ensembles Can Estimate Epistemic Uncertainty

Suppose several models are trained independently:

```text
Dataset
   │
   ├──→ Model 1
   ├──→ Model 2
   ├──→ Model 3
   └──→ Model 4
```

Each model may learn a slightly different function.

If the training data strongly constrain the relationship, the models tend to make similar predictions.

If the training data leave substantial ambiguity, the learned models can disagree.

Therefore:

```text
Model disagreement
        ↓
Information about
epistemic uncertainty
```

This is one reason ensemble-based uncertainty estimation is widely useful.

The ensemble does not directly reveal a mathematically exact epistemic uncertainty in every situation, but it provides a practical approximation of uncertainty arising from model variability.

---

## 27.4.10 Sparse Crystal-Structure Regions

Epistemic uncertainty is not limited to chemical composition.

It can also arise from unusual crystal structures.

Suppose the training dataset contains mostly:

```text
Simple structures
Small unit cells
High-symmetry crystals
```

but a new material has:

```text
Large unit cell
Low symmetry
Complex coordination
Unusual structural motif
```

The chemistry may be familiar, but the structure may not be.

Conceptually:

```text
Chemical space
       +
Structural space
       ↓
Materials representation
```

A material can therefore be familiar in one dimension and unfamiliar in another.

For crystal-property prediction, both aspects matter.

---

## 27.4.11 Unseen Crystal Structures

Consider a dataset containing:

```text
Cubic
Tetragonal
Orthorhombic
```

structures.

Suppose the model is then asked to predict a material with a highly unusual low-symmetry structure.

The model has no guarantee that its learned relationships remain accurate in this new structural region.

This is particularly important for graph neural networks.

A GNN may learn patterns involving:

```text
Atomic species
Bond distances
Neighbor relationships
Local coordination
```

but an unfamiliar crystal graph can contain combinations of these features that were rarely or never observed during training.

Thus:

```text
Unseen graph patterns
        ↓
Potentially higher epistemic uncertainty
```

---

## 27.4.12 Rare Coordination Environments

Consider a model trained primarily on common coordination environments:

```text
4-coordinate
6-coordinate
8-coordinate
```

A new crystal may contain an unusual local environment:

```text
5-coordinate
```

or a coordination geometry that is rare in the training dataset.

The model may have limited evidence for how such a configuration relates to the target property.

This can produce epistemic uncertainty even if the individual atomic species are familiar.

Therefore, crystal-level uncertainty can arise from local structural novelty.

---

## 27.4.13 Training Data Coverage

Training-data coverage refers to how well the available dataset represents the region in which predictions will be made.

A useful conceptual representation is:

```text
Training coverage
        ↓
┌──────────────────────────────┐
│ Chemical coverage            │
│ Structural coverage          │
│ Composition coverage         │
│ Property coverage            │
│ Local-environment coverage   │
└──────────────────────────────┘
```

High coverage generally provides stronger empirical support.

Low coverage creates a greater possibility of epistemic uncertainty.

However, coverage should not be interpreted simply as the number of samples.

A dataset containing many nearly identical materials may still provide poor coverage of the broader materials space.

---

## 27.4.14 Number of Samples Versus Information Content

Suppose a dataset contains:

```text
10,000 structures
```

but:

```text
9,500
```

belong to nearly identical structural families.

The dataset may appear large while still having limited diversity.

Another dataset might contain:

```text
3,000 structures
```

covering a much wider range of:

```text
Elements
Compositions
Crystal systems
Coordination environments
```

The second dataset may provide better coverage for some prediction tasks.

Therefore:

```text
Dataset size
≠
Dataset information coverage
```

This distinction is essential when interpreting epistemic uncertainty.

---

## 27.4.15 Epistemic Uncertainty and Extrapolation

A model is usually strongest when interpolating between familiar examples.

Consider:

```text
Training data

●     ●     ●     ●
```

and a new point:

```text
●  New  ●
```

The model is predicting between observed examples.

This is approximately interpolation.

Now consider:

```text
Training data

●     ●     ●     ●

                         New
```

The new point lies outside the observed region.

This is extrapolation.

Extrapolation is generally more uncertain because the model must infer behavior beyond the region directly constrained by the training data.

Therefore:

```text
Interpolation
→ often lower epistemic uncertainty

Extrapolation
→ potentially higher epistemic uncertainty
```

The exact behavior depends on the model and representation, so this relationship should be evaluated rather than assumed.

---

## 27.4.16 Why Neural Networks Can Be Overconfident

A particularly important problem is that ordinary neural networks can produce confident-looking predictions even when they are operating outside the training distribution.

For example:

```text
Training region
████████████████

New material
                         X
```

The network still performs a forward pass:

```text
X
↓
Neural network
↓
2.73 eV
```

The output may appear precise.

But the numerical precision of the output does not indicate scientific confidence.

This produces the important distinction:

```text
Numerical certainty
≠
Scientific certainty
```

A model can output:

```text
2.731846 eV
```

without having strong evidence that the true value is close to that number.

Therefore, epistemic uncertainty estimation is particularly important for scientific machine learning.

---

## 27.4.17 Epistemic Uncertainty and Model Parameters

In a Bayesian formulation, epistemic uncertainty can be understood through uncertainty in the model parameters.

Let:

```text
θ
```

represent model parameters.

A conventional neural network learns one parameter set:

```text
θ*
```

A Bayesian model instead considers a distribution over possible parameter values:

```text
p(θ | D)
```

where:

```text
D
```

is the training dataset.

If the data strongly constrain the parameters:

```text
p(θ | D)
```

is relatively concentrated.

If the data provide weak constraints:

```text
p(θ | D)
```

may be broader.

Conceptually:

```text
Strong training evidence
        ↓
Narrow parameter uncertainty
        ↓
Lower epistemic uncertainty
```

versus:

```text
Weak training evidence
        ↓
Broad parameter uncertainty
        ↓
Higher epistemic uncertainty
```

This Bayesian interpretation provides a theoretical foundation for epistemic uncertainty.

---

## 27.4.18 Predictive Uncertainty from Parameter Uncertainty

Consider a Bayesian model:

```text
p(y | x, θ)
```

The model parameters themselves are uncertain:

```text
p(θ | D)
```

The predictive distribution is obtained by considering possible parameter values:

```text
p(y | x, D)
=
∫ p(y | x, θ)
   p(θ | D)
   dθ
```

The important concept is that uncertainty in the model parameters propagates into uncertainty in the prediction.

Therefore:

```text
Training data
      ↓
Parameter uncertainty
      ↓
Predictive uncertainty
```

This provides a formal interpretation of epistemic uncertainty.

In practical machine-learning systems, exact Bayesian inference may be computationally difficult, so approximate methods are often used.

---

## 27.4.19 More Data and Epistemic Uncertainty

One of the defining characteristics of epistemic uncertainty is that it can often decrease as relevant data become available.

Conceptually:

```text
Few relevant examples
        ↓
High epistemic uncertainty
```

then:

```text
More relevant examples
        ↓
Better model knowledge
        ↓
Lower epistemic uncertainty
```

For example:

```text
Training set

Before:
Fe–O → many examples
Ru–O → 2 examples

After:
Fe–O → many examples
Ru–O → 500 examples
```

The model may become better constrained in the Ru–O region.

However, simply adding more samples does not guarantee reduced uncertainty.

The new samples must be:

```text
Relevant
Reliable
Representative
```

of the prediction region.

---

## 27.4.20 More Data Versus More Relevant Data

Consider two ways of increasing a dataset.

### Strategy A

Add:

```text
10,000 additional materials
```

that are very similar to existing training examples.

### Strategy B

Add:

```text
500 materials
```

from an underrepresented chemical and structural region.

For epistemic uncertainty in that underrepresented region, Strategy B may be much more informative.

Therefore:

```text
Data quantity
```

and:

```text
Data relevance
```

are different concepts.

Epistemic uncertainty is reduced most effectively when new information constrains previously uncertain regions of the model.

---

## 27.4.21 Epistemic Uncertainty in Crystal Graph Models

For a crystal graph:

```text
G = (V, E)
```

where:

```text
V = atoms
E = interactions
```

the model learns relationships among:

```text
Atomic species
Bond distances
Neighbor environments
Graph connectivity
```

Suppose the training dataset contains many examples of:

```text
Metal–O
Metal–S
Metal–N
```

but very few examples of:

```text
Metal–Se
```

A new crystal containing unusual Metal–Se environments may produce greater model uncertainty.

Similarly, a graph with a rare combination of:

```text
Coordination number
Bond lengths
Neighbor species
Structural topology
```

may be outside the familiar graph distribution.

Thus, epistemic uncertainty can be interpreted at the crystal-graph level.

---

## 27.4.22 Node-Level Sources of Epistemic Uncertainty

A crystal graph contains nodes representing atoms.

Some atomic environments may be common:

```text
O surrounded by common metal neighbors
```

while others may be rare:

```text
Unusual element
+
Rare coordination
+
Unusual local geometry
```

A GNN may have stronger learned representations for the first environment than the second.

Therefore, some parts of a crystal graph may be more epistemically uncertain than others.

Conceptually:

```text
       O
      / \
     /   \
    Fe---O
     \
      X*

X* = unusual local environment
```

The unusual environment may contribute disproportionately to uncertainty in the graph-level prediction.

This provides a bridge between epistemic uncertainty and uncertainty analysis of crystal representations.

---

## 27.4.23 Graph-Level Epistemic Uncertainty

At the graph level, the entire crystal can be considered unfamiliar.

For example:

```text
Training graphs
G₁
G₂
G₃
...
Gₙ
```

A new crystal:

```text
G*
```

may contain a combination of local environments that rarely appeared together during training.

Even if each individual atom type is familiar, the complete graph may be novel.

Therefore:

```text
Familiar atoms
+
Familiar local environments
≠
necessarily familiar crystal
```

This is important for materials GNNs because the model learns relationships between local and global structural patterns.

---

## 27.4.24 Estimating Epistemic Uncertainty with Multiple Models

A practical approach is to train multiple independently initialized models.

For example:

```python
models = []

for seed in range(5):

    torch.manual_seed(seed)

    model = build_model()

    train(
        model,
        X_train,
        y_train
    )

    models.append(model)
```

Predictions can then be collected:

```python
predictions = []

for model in models:

    with torch.no_grad():

        pred = model(
            X_test
        )

    predictions.append(
        pred
    )
```

They can be combined:

```python
predictions = torch.stack(
    predictions
)

mean_prediction = (
    predictions.mean(dim=0)
)

epistemic_variance = (
    predictions.var(
        dim=0,
        unbiased=False
    )
)
```

The variance among the independently trained models provides a practical model-disagreement measure.

It can therefore serve as an approximate epistemic uncertainty signal.

---

## 27.4.25 Interpreting Ensemble Disagreement

Suppose five models predict:

```text
Material A

2.01
2.03
2.02
2.04
2.01
```

The ensemble disagreement is small.

For another material:

```text
Material B

1.2
1.9
2.7
3.1
2.4
```

the disagreement is large.

The second prediction should therefore be treated with greater caution.

The key interpretation is:

```text
Models agree
→ learned relationship is relatively consistent

Models disagree
→ learned relationship is less constrained
```

This is one practical interpretation of epistemic uncertainty.

---

## 27.4.26 Limitations of Ensemble Disagreement

Ensemble disagreement is useful, but it is not a perfect uncertainty measure.

Models may agree for the wrong reason.

For example:

```text
Model 1 → 3.0
Model 2 → 3.0
Model 3 → 3.0
Model 4 → 3.0
```

does not guarantee that:

```text
True property ≈ 3.0
```

The models may share the same systematic bias.

Therefore:

```text
Low disagreement
```

does not automatically imply:

```text
Low prediction error.
```

Similarly, high disagreement may sometimes arise from optimization instability rather than genuine uncertainty about the physical relationship.

This is why uncertainty estimates must eventually be evaluated against observed errors and calibrated.

---

## 27.4.27 Epistemic Uncertainty Versus Aleatoric Uncertainty

The two concepts can now be compared.

### Aleatoric

```text
Source:
Data-generating variability

Example:
Measurement noise

Question:
How variable can the outcome be
even for a given input?
```

### Epistemic

```text
Source:
Limited model knowledge

Example:
Sparse training data

Question:
How uncertain is the model because
it lacks sufficient information?
```

A useful conceptual diagram is:

```text
                    Total Predictive Uncertainty
                              │
                 ┌────────────┴────────────┐
                 ↓                         ↓
            Aleatoric                  Epistemic
                 │                         │
          Data variability          Model knowledge
                 │                         │
          Measurement noise         Sparse data
          Intrinsic variation       Unseen chemistry
                                    Unseen structures
```

This distinction is fundamental to uncertainty quantification.

---

## 27.4.28 Why the Distinction Matters

Suppose a material has high uncertainty.

The researcher should ask:

```text
Why is the uncertainty high?
```

If it is aleatoric:

```text
The target itself may have substantial variability.
```

If it is epistemic:

```text
The model may lack sufficient knowledge.
```

These situations have different scientific interpretations.

For epistemic uncertainty, additional relevant training data may potentially improve the model.

For aleatoric uncertainty, additional measurements may characterize the variability more accurately, but the underlying variability may remain.

Therefore:

```text
High uncertainty
```

is not by itself enough information.

The type of uncertainty matters.

---

## 27.4.29 A Materials Example Combining Both Types

Consider predicting an experimentally measured mechanical property.

A new material belongs to a chemical system that appears only twice in the training dataset.

At the same time, its experimental property is known to vary substantially with microstructure.

The prediction may therefore contain both:

```text
Epistemic uncertainty
+
Aleatoric uncertainty
```

Conceptually:

```text
New material
     │
     ├───────────────┐
     ↓               ↓
Limited training   Intrinsic /
knowledge          measurement
     ↓               variability
Epistemic          Aleatoric
uncertainty        uncertainty
     └───────┬───────┘
             ↓
      Predictive uncertainty
```

This is often the realistic situation in Materials Informatics.

The two sources should not be assumed to be identical.

---

## 27.4.30 Total Predictive Uncertainty

When both forms are present, predictive uncertainty can be conceptually decomposed as:

```text
Total uncertainty
=
Aleatoric uncertainty
+
Epistemic uncertainty
```

In probabilistic modeling, the exact decomposition depends on the model formulation.

For an ensemble of probabilistic models, a common conceptual decomposition is:

```text
Total predictive variance
=
Average predicted data variance
+
Variance of model means
```

The first term corresponds approximately to:

```text
Aleatoric uncertainty
```

and the second to:

```text
Epistemic uncertainty
```

This decomposition is particularly useful for understanding why an uncertainty estimate can contain multiple contributions.

---

## 27.4.31 Conceptual Ensemble Decomposition

Suppose an ensemble contains:

```text
Model 1:
μ₁ = 2.1
σ₁² = 0.04

Model 2:
μ₂ = 2.3
σ₂² = 0.09

Model 3:
μ₃ = 2.0
σ₃² = 0.05
```

The model means differ:

```text
2.1
2.3
2.0
```

This disagreement contributes to epistemic uncertainty.

Each model also predicts an individual variance:

```text
0.04
0.09
0.05
```

which represents aleatoric uncertainty under the probabilistic model.

Thus:

```text
Within-model variance
→ aleatoric component

Between-model variance
→ epistemic component
```

This provides an intuitive way to separate the two sources in ensemble probabilistic models.

---

## 27.4.32 Epistemic Uncertainty and Training-Data Coverage

A useful practical principle is:

```text
Better coverage
        ↓
More relevant information
        ↓
Better constrained model
        ↓
Potentially lower epistemic uncertainty
```

Conversely:

```text
Poor coverage
        ↓
Limited information
        ↓
Weakly constrained model
        ↓
Potentially higher epistemic uncertainty
```

For Materials Informatics, coverage should be examined across:

```text
Chemical space
Structural space
Composition space
Local coordination space
Descriptor space
```

A model may be well covered in one space but poorly covered in another.

Therefore, uncertainty analysis should consider the representation relevant to the prediction task.

---

## 27.4.33 Scientific Interpretation of Epistemic Uncertainty

Epistemic uncertainty can provide a warning signal:

```text
"This material is not well supported
by the available training knowledge."
```

This does not mean that the material is bad.

It means:

```text
The prediction requires greater caution.
```

For example:

```text
Candidate A
Prediction = 2.4 eV
Low epistemic uncertainty

Candidate B
Prediction = 2.6 eV
High epistemic uncertainty
```

Candidate B may still have a true band gap close to 2.6 eV.

The uncertainty simply indicates that the model has weaker evidence for that prediction.

Therefore, epistemic uncertainty should be interpreted as a statement about **model knowledge**, not as a direct statement about material quality.

---

## 27.4.34 Practical Checklist for Epistemic Uncertainty

When a prediction appears highly uncertain, examine:

```text
[ ] Is the chemical system represented in training data?

[ ] Are similar compositions present?

[ ] Are similar crystal structures present?

[ ] Are similar local environments present?

[ ] Is the material outside the descriptor range?

[ ] Do independent models disagree?

[ ] Is the prediction an extrapolation?

[ ] Is the training dataset sufficiently diverse?

[ ] Could the representation itself be unfamiliar?
```

These questions help distinguish model-knowledge limitations from other sources of uncertainty.

---

## 27.4.35 Key Concepts of Epistemic Uncertainty

The main principles are:

```text
Epistemic uncertainty
        ↓
Limited model knowledge
        ↓
Often associated with incomplete
or sparse training information
```

Important sources include:

```text
Sparse training regions
Rare chemical systems
Unseen crystal structures
Rare coordination environments
Distribution shift
Extrapolation
Limited representation coverage
```

Practical indicators can include:

```text
Model disagreement
Ensemble variance
Bayesian parameter uncertainty
Low training-data density
```

However:

```text
Low model disagreement
≠
guaranteed correctness
```

and:

```text
High model disagreement
≠
necessarily large physical noise.
```

Uncertainty must therefore be interpreted and evaluated carefully.

---

## 27.4.36 Transition to Aleatoric and Epistemic Decomposition

The previous two sections establish the two fundamental sources of uncertainty:

```text
Aleatoric
→ variability associated with the data

Epistemic
→ uncertainty associated with limited model knowledge
```

A realistic Materials ML prediction may contain both.

The next section, **27.5 Aleatoric vs Epistemic Uncertainty**, will compare these two forms directly, develop the conceptual and mathematical decomposition of predictive uncertainty, examine which uncertainty can be reduced with additional information, and illustrate the distinction using materials examples.

## 27.5 Aleatoric vs Epistemic Uncertainty

The previous two sections introduced the two fundamental sources of uncertainty in machine-learning predictions:

```text
Aleatoric uncertainty
→ variability associated with the data-generating process

Epistemic uncertainty
→ uncertainty associated with limited model knowledge
```

Both can contribute to the uncertainty of a materials-property prediction.

A useful way to think about the distinction is:

```text
                    Prediction
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
        Data variability      Model uncertainty
             │                     │
             ↓                     ↓
         Aleatoric             Epistemic
```

The distinction is not merely theoretical.

For a materials researcher, knowing the source of uncertainty helps determine how a prediction should be interpreted and what type of additional information could improve confidence.

---

## 27.5.1 The Fundamental Difference

Consider a model predicting the band gap of a crystal.

Suppose the model predicts:

```text
Band gap = 2.5 eV
```

There are two fundamentally different reasons why the prediction might be uncertain.

### Case 1: Intrinsic or measurement variability

The material-property measurement itself may vary:

```text
2.3 eV
2.5 eV
2.4 eV
2.6 eV
```

Even with perfect knowledge of the underlying relationship, the observed target has some variability.

This corresponds to:

```text
Aleatoric uncertainty
```

### Case 2: Limited model knowledge

The crystal may belong to a chemical system that was poorly represented in the training dataset.

The model may therefore be unsure whether its learned relationship applies to this crystal.

This corresponds to:

```text
Epistemic uncertainty
```

Thus:

```text
Same prediction
+
Different source of uncertainty
=
Different scientific interpretation
```

---

## 27.5.2 A Simple Analogy

Consider throwing darts at a target.

### Aleatoric uncertainty

Suppose the dart launcher is inherently noisy.

Even if the launcher is perfectly calibrated, the darts land in different locations:

```text
        •
    •       •

       ●

    •       •
```

The variability is associated with the process itself.

This resembles aleatoric uncertainty.

### Epistemic uncertainty

Now suppose the launcher is perfectly precise, but the shooter does not know where the target is.

The uncertainty comes from incomplete knowledge.

This resembles epistemic uncertainty.

In machine learning:

```text
Aleatoric
→ noisy outcome

Epistemic
→ incomplete knowledge
```

The analogy is simplified, but it helps establish the conceptual distinction.

---

## 27.5.3 Which Uncertainty Can Be Reduced?

One of the most important differences concerns how uncertainty responds to additional information.

Aleatoric uncertainty is generally associated with variability that remains for a given information state.

Epistemic uncertainty can often decrease when the model receives more relevant information.

Conceptually:

```text
                More relevant data
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Aleatoric             Epistemic
       variability             uncertainty
             │                   │
             │             often decreases
             ↓                   ↓
       may remain          better knowledge
```

For example, suppose a material has intrinsically variable experimental behavior.

Adding more training examples may allow the model to estimate that variability more accurately, but it does not necessarily eliminate the variability itself.

In contrast, if the model has never seen similar materials, adding representative training examples can improve its knowledge of that region.

---

## 27.5.4 Additional Measurements Versus Additional Training Knowledge

These two situations should be distinguished carefully.

Suppose a material has:

```text
Observed property:

10.1
10.4
9.8
10.2
10.5
```

Additional measurements can help estimate the distribution of the property more accurately.

However, the underlying experimental variation may remain.

This is different from:

```text
Training data:

Material A
Material B
Material C

New material:
Material X
```

If Material X is poorly represented by the training data, adding relevant examples around that region may improve the model's knowledge.

Therefore:

```text
More measurements
→ better characterization of variability

More relevant training examples
→ potentially better model knowledge
```

These are not identical goals.

---

## 27.5.5 Mathematical View

Let:

```text
x
```

represent the materials input and:

```text
y
```

represent the target property.

The predictive distribution can be written as:

```text
p(y | x, D)
```

where:

```text
D
```

is the training dataset.

A useful conceptual decomposition is:

```text
Predictive uncertainty
=
Aleatoric contribution
+
Epistemic contribution
```

For probabilistic models, this can be expressed through the law of total variance:

```text
Var(y | x, D)
=
Eθ[
    Var(y | x, θ)
]
+
Varθ[
    E(y | x, θ)
]
```

The first term represents the average uncertainty remaining in the outcome for a given model.

The second term represents variation in the predicted mean across possible models or parameter values.

Conceptually:

```text
Eθ[ Var(y | x, θ) ]
        ↓
Aleatoric component

Varθ[ E(y | x, θ) ]
        ↓
Epistemic component
```

The exact interpretation depends on the probabilistic model, but this decomposition provides a useful theoretical framework.

---

## 27.5.6 Interpretation of the Variance Decomposition

The first term:

```text
Eθ[
    Var(y | x, θ)
]
```

asks:

```text
"If the model were fixed,
how much variability remains in the outcome?"
```

This corresponds to the data-level uncertainty represented by the model.

The second term:

```text
Varθ[
    E(y | x, θ)
]
```

asks:

```text
"How much do different plausible models
disagree about the prediction?"
```

This corresponds to model-level uncertainty.

Therefore:

```text
Outcome variability
        +
Model disagreement
        ↓
Total predictive uncertainty
```

This is one of the most useful conceptual decompositions in uncertainty-aware machine learning.

---

## 27.5.7 Ensemble Interpretation

Suppose an ensemble contains several probabilistic models.

For a particular crystal:

```text
Model 1:
μ₁ = 2.10 eV
σ₁ = 0.10 eV

Model 2:
μ₂ = 2.20 eV
σ₂ = 0.12 eV

Model 3:
μ₃ = 2.05 eV
σ₃ = 0.11 eV
```

There are two types of variation.

### Within-model variation

Each model predicts a standard deviation:

```text
0.10
0.12
0.11 eV
```

This represents the uncertainty assigned to the target distribution by each model.

### Between-model variation

The predicted means are:

```text
2.10
2.20
2.05 eV
```

The models disagree.

This disagreement provides an epistemic uncertainty signal.

Thus:

```text
Within-model spread
→ approximately aleatoric

Between-model spread
→ approximately epistemic
```

This decomposition is particularly useful when using probabilistic deep ensembles.

---

## 27.5.8 Example: Two Materials with Different Uncertainty Sources

Consider two crystals.

### Material A

```text
Model means:

2.40
2.41
2.39
2.40

Predicted standard deviations:

0.30
0.31
0.29
0.30
```

The models strongly agree.

However, each model predicts substantial target variability.

This suggests:

```text
Low epistemic uncertainty
+
Higher aleatoric uncertainty
```

### Material B

```text
Model means:

1.70
2.20
2.80
3.10

Predicted standard deviations:

0.08
0.07
0.09
0.08
```

The models strongly disagree.

But each model predicts relatively little intrinsic variability.

This suggests:

```text
Higher epistemic uncertainty
+
Lower aleatoric uncertainty
```

The two materials may have similar total uncertainty but for completely different reasons.

---

## 27.5.9 Why This Distinction Matters Scientifically

Suppose a researcher sees:

```text
High uncertainty
```

and stops there.

That provides incomplete information.

Instead, the researcher should ask:

```text
What is causing the uncertainty?
```

If the dominant component is aleatoric:

```text
The property may genuinely be variable
under the available conditions.
```

If the dominant component is epistemic:

```text
The model may lack sufficient knowledge
about this material.
```

This distinction changes how the prediction should be interpreted.

For example:

```text
Candidate A
High aleatoric uncertainty
```

may indicate a property that is inherently variable.

Whereas:

```text
Candidate B
High epistemic uncertainty
```

may indicate that the model has little experience with that material.

These situations should not be treated as equivalent.

---

## 27.5.10 Materials Example: Experimental Property Prediction

Suppose a model predicts yield strength.

For Material A:

```text
Prediction = 500 MPa
Aleatoric uncertainty = 50 MPa
Epistemic uncertainty = 10 MPa
```

For Material B:

```text
Prediction = 500 MPa
Aleatoric uncertainty = 10 MPa
Epistemic uncertainty = 70 MPa
```

Both have the same predicted mean:

```text
500 MPa
```

but their uncertainty sources differ.

Material A:

```text
500 MPa
+
substantial expected variability
```

Material B:

```text
500 MPa
+
limited model knowledge
```

A researcher should therefore not interpret the two predictions in the same way.

---

## 27.5.11 Materials Example: DFT Dataset

Consider a model trained on DFT formation energies.

Suppose the computational dataset is internally consistent:

```text
Same exchange-correlation functional
Same general computational protocol
Similar convergence criteria
```

Then the target values may have relatively limited methodological variability.

However, the model may encounter a new chemical system that is almost absent from the training data.

The dominant uncertainty may therefore be epistemic.

Conceptually:

```text
Consistent target generation
        ↓
Lower aleatoric contribution

Poor chemical coverage
        ↓
Higher epistemic contribution
```

This illustrates why uncertainty analysis must consider both the target-generation process and the model's training coverage.

---

## 27.5.12 Materials Example: Experimental Dataset

Now consider an experimental dataset.

Different laboratories may report measurements for nominally similar materials.

The observed values may differ because of:

```text
Sample preparation
Processing conditions
Microstructure
Instrument variation
Temperature
Measurement protocol
```

This can introduce substantial variability into the target.

Even if the training dataset is chemically and structurally diverse, the target itself may remain noisy.

Thus:

```text
Good training coverage
+
Variable measurements
```

can still produce substantial aleatoric uncertainty.

The model should not necessarily be considered poorly trained simply because the target distribution has a nonzero spread.

---

## 27.5.13 High Epistemic Uncertainty Is Not Necessarily Bad

A high epistemic uncertainty estimate does not mean that a material is undesirable.

It means:

```text
The model has limited knowledge
about the prediction.
```

Consider a hypothetical new crystal:

```text
Novel chemical system
Novel structure
High predicted property
High epistemic uncertainty
```

The material may be scientifically interesting precisely because it occupies a poorly represented region.

The uncertainty indicates that the prediction requires stronger validation.

Therefore:

```text
High epistemic uncertainty
≠
Bad material
```

Instead:

```text
High epistemic uncertainty
→
Prediction requires greater caution.
```

This distinction is essential for scientific interpretation.

---

## 27.5.14 High Aleatoric Uncertainty Is Also Not Necessarily Bad

Similarly, high aleatoric uncertainty does not automatically mean that the material is poor.

A property may naturally vary strongly with:

```text
Temperature
Microstructure
Defect concentration
Processing
Composition
```

A model may correctly identify that the conditional target distribution is broad.

For example:

```text
Material X
Mean thermal conductivity = 10 W/mK
Aleatoric uncertainty = 2 W/mK
```

This does not mean the model has failed.

It may mean that:

```text
The available representation
does not uniquely determine
the experimentally observed property.
```

The scientific response should therefore focus on understanding the source of the variability.

---

## 27.5.15 Can Aleatoric Uncertainty Be Reduced?

The phrase "irreducible uncertainty" requires careful interpretation.

If the uncertainty is caused by truly random variability in the target process, it cannot simply be removed by collecting more training examples.

However, uncertainty that appears aleatoric may sometimes actually arise from missing explanatory variables.

For example, suppose:

```text
Model inputs:
Composition
Crystal structure
```

but the property strongly depends on:

```text
Temperature
```

which is not included.

The resulting variability may appear as aleatoric uncertainty.

If temperature is subsequently included:

```text
Composition
+
Crystal structure
+
Temperature
```

some of that apparent uncertainty may decrease.

Therefore:

```text
Observed variability
```

and:

```text
Fundamentally irreducible variability
```

should not always be treated as identical.

This is an important modeling consideration.

---

## 27.5.16 Conditional Variability and Missing Information

Consider the true relationship:

```text
y = f(x, z)
```

where:

```text
x
```

is included in the model, but:

```text
z
```

is missing.

The model effectively tries to learn:

```text
y | x
```

even though the true relationship depends on:

```text
z
```

Different values of `z` can therefore produce different values of `y` for the same `x`.

Conceptually:

```text
              x
              │
        ┌─────┴─────┐
        ↓           ↓
      z = 1       z = 2
        ↓           ↓
       y₁          y₂
```

The variation between `y₁` and `y₂` can appear as conditional uncertainty.

This is one reason feature selection and representation quality influence uncertainty estimates.

---

## 27.5.17 Which Uncertainty Should Be Reduced?

The answer depends on the scientific objective.

If the problem is:

```text
Limited model knowledge
```

then the focus should be on improving knowledge of the relevant materials space.

If the problem is:

```text
Large measurement variability
```

then the focus should be on understanding or controlling the data-generating process.

Conceptually:

```text
High epistemic
      ↓
Model knowledge problem

High aleatoric
      ↓
Data variability problem
```

The uncertainty decomposition therefore helps diagnose the type of limitation affecting the prediction.

---

## 27.5.18 Relationship to Model Complexity

Increasing model complexity does not automatically reduce epistemic uncertainty.

A more complex model may actually become more uncertain or unstable when training data are insufficient.

For example:

```text
Small dataset
+
Very complex neural network
```

can produce poorly constrained model behavior.

Therefore:

```text
Model complexity
```

must be considered together with:

```text
Training-data quantity
Training-data diversity
Representation quality
```

The goal is not simply to use the most complex model.

The goal is to obtain reliable predictions with appropriately characterized uncertainty.

---

## 27.5.19 Relationship to Training Distribution

Suppose the training distribution is:

```text
p_train(x)
```

and the prediction distribution is:

```text
p_test(x)
```

If the two distributions differ substantially:

```text
p_train(x) ≠ p_test(x)
```

the model may encounter unfamiliar inputs.

This can increase epistemic uncertainty.

Conceptually:

```text
Training distribution
████████████████

Prediction region
              ███████████
```

The farther the prediction region moves away from the training distribution, the greater the concern about model knowledge.

This idea will become important later when uncertainty is used to assess distribution shift and out-of-distribution materials.

---

## 27.5.20 Uncertainty in Crystal Graphs

For crystal graph models, the two uncertainty sources can also be considered in terms of the graph.

Suppose:

```text
G = (V, E)
```

represents a crystal.

Aleatoric uncertainty may arise from variability in the target property associated with that crystal under the measurement or calculation conditions.

Epistemic uncertainty may arise because:

```text
The graph contains unfamiliar
atomic or structural patterns.
```

For example:

```text
Training graphs
        ↓
Common local environments
        ↓
Model learns graph-property relationships
```

while:

```text
New graph
        ↓
Rare local environments
        ↓
Weakly constrained prediction
```

This provides a graph-level interpretation of model uncertainty.

---

## 27.5.21 Local Versus Global Uncertainty

A crystal graph can contain both familiar and unfamiliar regions.

For example:

```text
        O
       / \
      Fe  O
       \ /
        X
```

Suppose the Fe–O environments are common in the training dataset, but the environment surrounding `X` is rare.

The graph-level prediction may therefore contain uncertainty associated with the unusual local environment.

This suggests a useful conceptual distinction:

```text
Local uncertainty
→ unfamiliar local structural environment

Global uncertainty
→ unfamiliar complete crystal graph
```

Both can contribute to the final property prediction.

---

## 27.5.22 A Unified View

The two uncertainty types can be placed into one framework:

```text
                       Materials Input
                              │
                              ↓
                     Machine-Learning Model
                              │
                 ┌────────────┴────────────┐
                 ↓                         ↓
        Data-generating variability    Model knowledge
                 │                         │
                 ↓                         ↓
             Aleatoric                 Epistemic
                 │                         │
                 └────────────┬────────────┘
                              ↓
                    Predictive uncertainty
```

This is the central conceptual structure of uncertainty decomposition.

---

## 27.5.23 Practical Interpretation Table

| Property                              | Aleatoric Uncertainty     | Epistemic Uncertainty     |
| ------------------------------------- | ------------------------- | ------------------------- |
| Main source                           | Data variability          | Limited model knowledge   |
| Typical example                       | Measurement noise         | Sparse training data      |
| Depends on input                      | Can be                    | Often can be              |
| Reduced by relevant new training data | Usually not fundamentally | Often                     |
| Associated with model ignorance       | No                        | Yes                       |
| Can occur in familiar materials       | Yes                       | Usually lower             |
| Can increase for unseen materials     | Not necessarily           | Often                     |
| Ensemble disagreement                 | Not its primary signal    | Useful signal             |
| Scientific meaning                    | Outcome variability       | Lack of empirical support |

The table should not be interpreted as an absolute rule.

Real datasets and models can contain interacting sources of uncertainty.

The purpose of the distinction is to provide a useful framework for analyzing those sources.

---

## 27.5.24 A Practical Example

Suppose a materials ML model predicts the formation energy of a new crystal.

The model produces:

```text
Mean prediction = -1.80 eV
Total uncertainty = high
```

An ensemble analysis gives:

```text
Between-model variance = high
Within-model variance = low
```

This suggests:

```text
Epistemic uncertainty
→ dominant
```

A reasonable interpretation is:

```text
The model does not have strong knowledge
about this crystal's region of materials space.
```

Now consider another material:

```text
Mean prediction = -1.80 eV
Total uncertainty = high

Between-model variance = low
Within-model variance = high
```

This suggests:

```text
Aleatoric uncertainty
→ dominant
```

The interpretation becomes:

```text
The models agree on the expected value,
but the target distribution is broad.
```

The same total uncertainty can therefore represent very different scientific situations.

---

## 27.5.25 Why Total Uncertainty Alone Can Be Misleading

Suppose two materials have:

```text
Material A:
Total uncertainty = 0.50

Material B:
Total uncertainty = 0.50
```

If only the total uncertainty is reported, they appear equivalent.

But suppose:

```text
Material A:
Aleatoric = 0.45
Epistemic = 0.05

Material B:
Aleatoric = 0.05
Epistemic = 0.45
```

Their scientific interpretations are completely different.

Material A:

```text
Large target variability
+
Strong model knowledge
```

Material B:

```text
Stable-looking target
+
Weak model knowledge
```

Therefore, uncertainty decomposition can provide information that a single uncertainty number cannot.

---

## 27.5.26 Important Caution About Decomposition

The decomposition between aleatoric and epistemic uncertainty is model-dependent.

Different uncertainty-estimation methods may provide different approximations.

For example:

```text
Deep ensemble
→ model disagreement

MC Dropout
→ stochastic prediction variation

Bayesian model
→ parameter posterior uncertainty

Gaussian Process
→ posterior predictive uncertainty
```

These methods do not necessarily produce identical numerical decompositions.

Therefore, researchers should clearly document:

```text
Model
Uncertainty method
Assumptions
Calibration procedure
Evaluation procedure
```

when reporting uncertainty.

A number called "epistemic uncertainty" is meaningful only when its definition and estimation procedure are clear.

---

## 27.5.27 Research Interpretation Framework

For a new materials prediction, a useful reasoning sequence is:

```text
Step 1
Obtain prediction

        ↓

Step 2
Estimate total uncertainty

        ↓

Step 3
Separate or approximate
aleatoric and epistemic components

        ↓

Step 4
Ask whether the material
is well represented in training data

        ↓

Step 5
Check whether the target itself
is intrinsically variable

        ↓

Step 6
Interpret the prediction
with the uncertainty information
```

This prevents the common mistake of treating all uncertainty as the same phenomenon.

---

## 27.5.28 Key Takeaways

The distinction between aleatoric and epistemic uncertainty can be summarized as:

```text
Aleatoric uncertainty
=
variability associated with the outcome
```

```text
Epistemic uncertainty
=
uncertainty associated with limited model knowledge
```

A more complete view is:

```text
Total Predictive Uncertainty
        =
Aleatoric
        +
Epistemic
```

In Materials Informatics:

```text
Aleatoric
→ measurement variability
→ experimental conditions
→ intrinsic target variability
→ missing explanatory variables

Epistemic
→ sparse training regions
→ rare chemical systems
→ unseen crystal structures
→ limited model knowledge
→ extrapolation
```

The most important scientific principle is:

```text
High uncertainty is not an explanation by itself.
```

The researcher must determine **what kind of uncertainty is present and why**.

---

## 27.5.29 Transition to Probabilistic Regression

So far, uncertainty has been discussed conceptually.

The next step is to construct models that explicitly represent uncertainty in their predictions.

A conventional regression model produces:

```text
Input
  ↓
Model
  ↓
Single value
```

A probabilistic regression model instead produces:

```text
Input
  ↓
Model
  ↓
Predictive distribution
  ↓
Mean + uncertainty
```

The next section, **27.6 Probabilistic Regression**, will introduce point prediction versus predictive distributions, Gaussian predictive distributions, predictive mean and variance, predictive intervals, and the limitations of Gaussian assumptions.

## 27.6 Probabilistic Regression

Traditional regression models usually produce a single numerical prediction for each input.

For a materials-property prediction problem, the workflow may look like:

```text
Crystal Structure
       ↓
Materials Representation
       ↓
Regression Model
       ↓
Predicted Property
```

For example:

```text
Crystal
   ↓
Machine-Learning Model
   ↓
Band gap = 2.43 eV
```

This is called a **point prediction**.

A point prediction is useful, but it does not describe how uncertain the prediction is.

A probabilistic regression model goes one step further.

Instead of predicting only:

```text
2.43 eV
```

the model predicts a distribution such as:

```text
Mean = 2.43 eV
Uncertainty = 0.18 eV
```

The conceptual difference is:

```text
Point Regression

x
↓
Model
↓
ŷ


Probabilistic Regression

x
↓
Model
↓
Predictive Distribution
↓
Mean + Variance
```

This makes probabilistic regression an important foundation for uncertainty-aware Materials Informatics.

---

## 27.6.1 Point Prediction Versus Predictive Distribution

Suppose a model predicts the formation energy of a crystal.

A conventional regression model might produce:

```text
Formation energy = -2.15 eV/atom
```

The model gives no direct information about how strongly this value is supported.

A probabilistic model may instead produce:

```text
Mean = -2.15 eV/atom
Standard deviation = 0.20 eV/atom
```

The prediction can then be represented as:

```text
-2.15 ± 0.20 eV/atom
```

The important point is that the model is now describing a **distribution of plausible outcomes** rather than a single number.

Conceptually:

```text
Probability
   │
   │              /\
   │            /    \
   │          /        \
   │________/____________\_______
             -2.15
             Formation Energy
```

The center of the distribution represents the expected prediction, while its spread represents uncertainty under the model.

---

## 27.6.2 Why Predictive Distributions Are Useful

Consider two crystals.

### Crystal A

```text
Prediction = 2.50 eV
Uncertainty = 0.05 eV
```

### Crystal B

```text
Prediction = 2.50 eV
Uncertainty = 0.60 eV
```

The predicted means are identical.

However, the two predictions do not have the same level of confidence.

Conceptually:

```text
Crystal A

        /\
       /  \
______/____\______
      narrow


Crystal B

      /      \
_____/________\_____
       broad
```

The first prediction is sharply concentrated.

The second is much more uncertain.

Therefore, probabilistic regression allows the model to distinguish between:

```text
Precise prediction
```

and:

```text
Uncertain prediction
```

even when their predicted means are identical.

---

## 27.6.3 Predictive Mean

The most basic quantity in probabilistic regression is the predictive mean.

Let the predictive distribution be:

```text
p(y | x)
```

The expected prediction is:

```text
μ(x) = E[y | x]
```

In practical terms:

```text
μ
```

represents the model's central prediction for the target property.

For example:

```text
Input:
Crystal structure

Output:

μ = 3.2 eV
```

The mean can therefore still be used as the primary point estimate.

The difference is that the probabilistic model also estimates the spread around that mean.

---

## 27.6.4 Predictive Variance

The second important quantity is predictive variance.

It describes how broadly the predictive distribution is spread.

Conceptually:

```text
Small variance
→ narrow distribution
→ lower uncertainty

Large variance
→ broad distribution
→ higher uncertainty
```

The variance is often written as:

```text
σ²(x)
```

and the corresponding standard deviation is:

```text
σ(x)
```

For example:

```text
Crystal A:

μ = 2.50 eV
σ = 0.10 eV
```

versus:

```text
Crystal B:

μ = 2.50 eV
σ = 0.50 eV
```

The second prediction has a substantially broader predictive distribution.

---

## 27.6.5 Gaussian Predictive Distribution

One of the simplest probabilistic regression assumptions is that the target follows a Gaussian distribution conditioned on the input.

The model assumes:

```text
y | x ~ Normal(μ(x), σ²(x))
```

where:

```text
μ(x)
```

is the predicted mean and:

```text
σ²(x)
```

is the predicted variance.

The probability density is:

```text
p(y | x)
=
1
---------
σ√(2π)

exp(
    -(y - μ)²
    ------------
      2σ²
)
```

The important idea is not the equation itself but what the model produces:

```text
Input
  ↓
Model
  ↓
μ(x), σ²(x)
```

Therefore:

```text
Traditional regression:
x → μ

Probabilistic regression:
x → μ, σ²
```

---

## 27.6.6 Interpreting the Gaussian Model

Suppose the model predicts:

```text
μ = 1.80 eV
σ = 0.15 eV
```

The Gaussian predictive distribution is centered around:

```text
1.80 eV
```

with a characteristic spread determined by:

```text
0.15 eV
```

A useful conceptual interpretation is:

```text
            Probability

                 /\
               /    \
             /        \
___________/____________\___________
          1.50  1.80  2.10
                 ↑
                μ
```

The exact probability assigned to any interval depends on the Gaussian distribution.

The important point is that the model now describes both:

```text
Expected value
+
Expected variability
```

---

## 27.6.7 Predicting Mean and Variance with a Neural Network

A neural network can be designed to output two quantities.

Instead of:

```python
prediction = model(x)
```

where `prediction` is a single value, the model can produce:

```python
mean, variance = model(x)
```

A simple PyTorch architecture is:

```python
import torch
import torch.nn as nn

class ProbabilisticRegressor(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=64
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(
            hidden_dim,
            1
        )

        self.log_variance_head = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x):

        h = self.network(x)

        mean = self.mean_head(h)

        log_variance = (
            self.log_variance_head(h)
        )

        return (
            mean,
            log_variance
        )
```

The network therefore contains two output heads:

```text
Shared representation
        │
        ├──────────────→ Mean
        │
        └──────────────→ Log variance
```

This structure allows the model to learn both the expected target and its uncertainty.

---

## 27.6.8 Why Predict Log Variance?

Variance must be positive:

```text
σ² > 0
```

A neural network output, however, can be any real number.

If the network directly predicts:

```text
variance
```

it could produce:

```text
-0.5
```

which is physically invalid as a variance.

A common solution is to predict:

```text
log(σ²)
```

instead.

The variance can then be recovered using:

```python
variance = torch.exp(
    log_variance
)
```

Since:

```text
exp(z) > 0
```

for all real `z`, the resulting variance is automatically positive.

Conceptually:

```text
Neural network output
        ↓
log(σ²)
        ↓
exp(...)
        ↓
σ² > 0
```

This is a standard technique for probabilistic neural-network regression.

---

## 27.6.9 Predictive Standard Deviation

Once the model predicts:

```python
log_variance
```

the standard deviation can be calculated as:

```python
variance = torch.exp(
    log_variance
)

std = torch.sqrt(
    variance
)
```

Equivalently:

```python
std = torch.exp(
    0.5 * log_variance
)
```

Therefore:

```text
Model output
      ↓
log variance
      ↓
variance
      ↓
standard deviation
```

For a materials-property prediction, the final result might be:

```text
Band gap:

Mean = 2.35 eV
Std   = 0.18 eV
```

---

## 27.6.10 Heteroscedastic Probabilistic Regression

A particularly important case occurs when the uncertainty changes with the input.

Suppose some materials have very predictable properties while others exhibit much larger variability.

Then:

```text
σ = constant
```

is not appropriate.

Instead:

```text
σ = σ(x)
```

should depend on the input.

This is called **heteroscedastic uncertainty**.

Conceptually:

```text
Material A
→ small σ

Material B
→ medium σ

Material C
→ large σ
```

The predictive distribution therefore changes from one material to another.

A conceptual graph is:

```text
Target
  │
  │            broad uncertainty
  │              ╱──────╲
  │        ╱────╲
  │      ╱        ╲
  │  ╱──╲
  │ ╱    ╲
  └────────────────────────
             Input
```

The width of the predictive distribution varies across the materials space.

---

## 27.6.11 Homoscedastic Versus Heteroscedastic Uncertainty

The difference can be summarized as:

### Homoscedastic

```text
σ(x) = constant
```

Every input receives approximately the same noise level.

Conceptually:

```text
Material 1 → σ = 0.2
Material 2 → σ = 0.2
Material 3 → σ = 0.2
```

### Heteroscedastic

```text
σ(x) varies with x
```

Different materials receive different uncertainty estimates.

For example:

```text
Material 1 → σ = 0.05
Material 2 → σ = 0.20
Material 3 → σ = 0.60
```

For Materials Informatics, heteroscedastic models can be useful when measurement or target variability changes across chemical or structural regions.

---

## 27.6.12 Example: Materials Property With Variable Noise

Suppose a model predicts a property across several materials families.

The observations might look conceptually like:

```text
Family A

● ● ● ● ●
     ●


Family B

●      ●
   ●       ●
      ●


Family C

●
        ●
   ●
             ●
```

Family A has relatively small target variability.

Family C has much larger variability.

A probabilistic regression model can learn:

```text
Family A → smaller σ
Family B → intermediate σ
Family C → larger σ
```

This allows the uncertainty estimate to depend on the material itself.

---

## 27.6.13 Gaussian Negative Log-Likelihood

To train a probabilistic regression model, the loss function must consider both the predicted mean and variance.

For a Gaussian likelihood, a commonly used loss is the negative log-likelihood:

```text
NLL
=
1/2
[
log(σ²)
+
(y - μ)² / σ²
]
```

Ignoring constants that do not depend on the model parameters, this captures two competing objectives.

First:

```text
log(σ²)
```

discourages the model from simply predicting an extremely large uncertainty.

Second:

```text
(y - μ)² / σ²
```

penalizes prediction errors relative to the predicted variance.

Therefore, the model must learn a balance between:

```text
Accurate mean prediction
+
Reasonable uncertainty estimate
```

---

## 27.6.14 Why the Variance Cannot Simply Become Very Large

Suppose the model makes a large prediction error:

```text
y - μ
```

One possible strategy would be to increase:

```text
σ²
```

so that:

```text
(y - μ)² / σ²
```

becomes smaller.

However, the loss also contains:

```text
log(σ²)
```

which increases when the variance becomes large.

Therefore, the model cannot minimize the loss simply by predicting enormous uncertainty for every material.

The two terms work together:

```text
Prediction error
        ↓
Requires appropriate uncertainty

Excessively large uncertainty
        ↓
Penalized by log variance
```

This is what allows the model to learn meaningful predictive variance.

---

## 27.6.15 PyTorch Gaussian NLL Implementation

PyTorch provides a convenient loss for Gaussian negative log-likelihood.

For example:

```python
import torch
import torch.nn.functional as F

mean, log_variance = model(x)

variance = torch.exp(
    log_variance
)

loss = F.gaussian_nll_loss(
    mean,
    y,
    variance
)
```

The important sequence is:

```text
Input
 ↓
Model
 ↓
Mean
+
Log variance
 ↓
Variance
 ↓
Gaussian NLL
 ↓
Optimization
```

This allows the model to learn a probabilistic regression output.

---

## 27.6.16 Training Example

A simplified training loop is:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    mean, log_variance = model(
        X_train
    )

    variance = torch.exp(
        log_variance
    )

    loss = F.gaussian_nll_loss(
        mean,
        y_train,
        variance
    )

    loss.backward()

    optimizer.step()
```

The model learns parameters that maximize the likelihood of the observed training targets under the predicted distributions.

---

## 27.6.17 Numerical Stability

Directly exponentiating very large positive or negative values can create numerical problems.

For example:

```python
variance = torch.exp(
    log_variance
)
```

can become extremely large if:

```text
log_variance
```

is very large.

Similarly, very negative values can produce extremely small variances.

In practical implementations, numerical stability should therefore be considered.

One approach is to constrain the predicted log variance within a reasonable range:

```python
log_variance = torch.clamp(
    log_variance,
    min=-10,
    max=10
)
```

The appropriate range depends on the scale of the target variable.

Another approach is to use a numerically stable positive transformation for the standard deviation or variance.

The important principle is:

```text
Variance must remain positive
+
Training must remain numerically stable
```

---

## 27.6.18 Scaling the Target Variable

Probabilistic regression is sensitive to the scale of the target.

Suppose the target is:

```text
Formation energy
```

with values around:

```text
-10 to 5 eV
```

while another property may have values around:

```text
10⁵ MPa
```

Using raw values without considering scale can make optimization difficult.

A common approach is to standardize the target:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

y_train_scaled = scaler.fit_transform(
    y_train.reshape(-1, 1)
)
```

The probabilistic model is then trained on the scaled target.

After prediction, the mean and standard deviation must be transformed back consistently.

This is particularly important because both:

```text
mean
```

and:

```text
uncertainty
```

have units associated with the target property.

---

## 27.6.19 Transforming Uncertainty Back to Physical Units

Suppose the target was standardized:

```text
y_scaled = (y - μ_y) / σ_y
```

and the model predicts:

```text
mean_scaled
std_scaled
```

The mean can be transformed back to the original units.

The standard deviation must also be transformed.

If:

```text
σ_y
```

is the standard deviation used for target scaling, then:

```text
std_original
=
std_scaled × σ_y
```

For example:

```text
Target scaling factor = 2.0 eV

Predicted scaled standard deviation = 0.10

Original uncertainty = 0.20 eV
```

The uncertainty must therefore always be reported in physically meaningful units when results are presented to materials researchers.

---

## 27.6.20 Prediction Intervals

A predictive distribution can be converted into an interval.

For a Gaussian distribution:

```text
μ ± zσ
```

defines an interval corresponding to a chosen standard-normal multiplier `z`.

A common approximate interval is:

```text
μ ± 1.96σ
```

which corresponds to approximately 95% probability under the Gaussian assumption.

For example:

```text
Mean = 2.50 eV
σ = 0.20 eV
```

gives:

```text
Lower = 2.50 - 1.96(0.20)
Upper = 2.50 + 1.96(0.20)
```

approximately:

```text
[2.11, 2.89] eV
```

The interval is meaningful only if the Gaussian assumption and uncertainty calibration are appropriate.

---

## 27.6.21 Prediction Interval Versus Confidence Interval

These terms should not be confused.

A **prediction interval** describes uncertainty about a future target observation.

A **confidence interval** generally describes uncertainty associated with an estimated statistical quantity.

In uncertainty-aware materials prediction, the quantity of interest is usually the future material property.

Therefore, predictive intervals are often the more relevant concept.

Conceptually:

```text
Model uncertainty about parameter
        ↓
Confidence interval

Uncertainty about future observation
        ↓
Prediction interval
```

The distinction becomes especially important when reporting uncertainty in scientific predictions.

---

## 27.6.22 Prediction Intervals for Materials Properties

Suppose a model predicts:

```text
Elastic modulus:

μ = 180 GPa
σ = 12 GPa
```

Under a Gaussian assumption, an approximate 95% prediction interval is:

```text
180 ± 1.96(12)
```

giving approximately:

```text
[156.5, 203.5] GPa
```

The result can be reported as:

```text
Predicted elastic modulus:
180 GPa

Approximate 95% prediction interval:
156.5–203.5 GPa
```

This communicates considerably more information than:

```text
180 GPa
```

alone.

---

## 27.6.23 Prediction Intervals Must Be Validated

A model cannot simply report:

```text
95% prediction interval
```

and automatically claim that 95% of future observations will fall inside the interval.

The interval must be evaluated.

Suppose 1,000 test materials are evaluated.

If the model claims:

```text
95% prediction intervals
```

then approximately:

```text
950
```

observations should fall within their corresponding intervals if the intervals are well calibrated.

If only:

```text
700
```

fall inside, the intervals are too narrow or otherwise miscalibrated.

If:

```text
999
```

fall inside, the intervals may be excessively wide.

Therefore:

```text
Nominal coverage
```

must be compared with:

```text
Observed coverage
```

---

## 27.6.24 Coverage Probability

Suppose each test material receives a prediction interval:

```text
[Lᵢ, Uᵢ]
```

where:

```text
Lᵢ = lower bound
Uᵢ = upper bound
```

The observation is covered if:

```text
Lᵢ ≤ yᵢ ≤ Uᵢ
```

The empirical coverage is:

```text
Coverage
=
Number of covered observations
/
Total number of observations
```

For example:

```text
900 covered
100 total? 
```

would not be meaningful if the denominator is inconsistent; the correct calculation must use the complete test set.

For 1,000 observations:

```text
950 covered
1000 total
```

gives:

```text
Coverage = 95%
```

This is one of the most important ways to evaluate predictive intervals.

---

## 27.6.25 Sharpness and Calibration

A good uncertainty model should ideally produce intervals that are:

```text
Well calibrated
```

and:

```text
As narrow as justified
```

These properties are related but different.

### Calibration

Does the observed coverage match the nominal coverage?

### Sharpness

How concentrated are the predictive distributions?

Conceptually:

```text
Good uncertainty model

Correct coverage
        +
Reasonably narrow intervals
```

A model that always predicts extremely wide intervals may achieve high coverage but provide little useful information.

Therefore:

```text
Calibration
+
Sharpness
```

should both be considered.

---

## 27.6.26 Limitation of the Gaussian Assumption

Gaussian predictive distributions are convenient, but real materials-property distributions may not always be Gaussian.

A Gaussian distribution assumes:

```text
Symmetric
Bell-shaped
Continuous
Unimodal
```

behavior.

Real materials data can instead exhibit:

```text
Skewness
Multiple modes
Heavy tails
Boundaries
Outliers
```

For example, a property may have a physical lower bound:

```text
y ≥ 0
```

while a Gaussian distribution extends mathematically toward:

```text
y < 0
```

This can be inappropriate.

Therefore, Gaussian probabilistic regression should be treated as a modeling assumption rather than a universal description of materials-property uncertainty.

---

## 27.6.27 Skewed Materials Properties

Suppose a property has a distribution like:

```text
Probability
   │
   │       /\
   │      /  \
   │     /    \
   │____/      \____________
   └────────────────────────
```

The distribution is strongly right-skewed.

A symmetric Gaussian approximation may poorly represent the actual uncertainty.

In such situations, alternative probabilistic models may be more appropriate.

The important principle is:

```text
Choose a predictive distribution
that is compatible with the target.
```

---

## 27.6.28 Bounded Properties

Some materials properties have natural physical boundaries.

For example:

```text
Probability
```

cannot be negative.

Similarly, classification probabilities must satisfy:

```text
0 ≤ p ≤ 1
```

A Gaussian distribution is not naturally bounded.

Therefore, when the target has strict bounds, the predictive distribution should account for those constraints where appropriate.

This is another reason why probabilistic regression is not simply equivalent to:

```text
Prediction ± standard deviation
```

The probability model itself matters.

---

## 27.6.29 Multimodal Predictive Distributions

Some materials problems may contain multiple physically meaningful regimes.

For example:

```text
Structure family A
→ property around 2

Structure family B
→ property around 5
```

The conditional distribution may therefore contain two modes:

```text
Probability

     /\          /\
    /  \        /  \
___/____\______/____\____
       2          5
```

A single Gaussian distribution would place substantial probability between the two modes.

That may not accurately represent the underlying physical possibilities.

This demonstrates an important limitation of simple Gaussian probabilistic regression.

---

## 27.6.30 Probabilistic Regression for Materials Screening

Probabilistic predictions become particularly useful when screening many candidate materials.

Suppose five candidates are predicted:

| Candidate | Mean band gap | Standard deviation |
| --------- | ------------: | -----------------: |
| A         |        2.1 eV |            0.05 eV |
| B         |        2.3 eV |            0.10 eV |
| C         |        2.5 eV |            0.40 eV |
| D         |        2.7 eV |            0.15 eV |
| E         |        2.9 eV |            0.60 eV |

A point-prediction-only model would provide only the middle column.

A probabilistic model provides both:

```text
Expected property
+
Prediction uncertainty
```

This allows the researcher to distinguish:

```text
High predicted value
```

from:

```text
High predicted value with strong uncertainty.
```

The uncertainty does not by itself determine whether a candidate should be selected, but it provides important information for interpreting the prediction.

---

## 27.6.31 Uncertainty-Aware Ranking

Suppose the desired property is high.

Candidate A:

```text
Mean = 3.0
σ = 0.05
```

Candidate B:

```text
Mean = 3.2
σ = 0.60
```

Candidate B has the higher predicted mean.

However, the prediction is much less precise.

A researcher may therefore want to report:

```text
Candidate A:
3.0 ± 0.05

Candidate B:
3.2 ± 0.60
```

rather than simply:

```text
B > A
```

This is an example of why uncertainty should accompany predictions in scientific workflows.

Detailed candidate-selection strategies are separate decision-making questions; here the focus is only on representing and interpreting the predictive uncertainty.

---

## 27.6.32 Complete Probabilistic Regression Workflow

A basic workflow is:

```text
Materials Dataset
        ↓
Feature / Crystal Representation
        ↓
Train Probabilistic Regression Model
        ↓
Predict Mean
        +
Predict Variance
        ↓
Construct Predictive Distribution
        ↓
Construct Prediction Interval
        ↓
Evaluate Coverage
        ↓
Evaluate Calibration
```

The essential transformation is:

```text
Materials Input
      ↓
Probabilistic Model
      ↓
μ(x), σ²(x)
      ↓
Predictive Distribution
```

---

## 27.6.33 Practical Python Example

A simplified example using a materials dataset might look like:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticModel(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(
                input_dim,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                64
            ),
            nn.ReLU()
        )

        self.mean = nn.Linear(
            64,
            1
        )

        self.log_var = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        h = self.shared(x)

        mean = self.mean(h)

        log_var = self.log_var(h)

        return mean, log_var
```

Training:

```python
model = ProbabilisticModel(
    input_dim=X_train.shape[1]
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(200):

    optimizer.zero_grad()

    mean, log_var = model(
        X_train
    )

    variance = torch.exp(
        log_var
    )

    loss = F.gaussian_nll_loss(
        mean,
        y_train,
        variance
    )

    loss.backward()

    optimizer.step()
```

Prediction:

```python
model.eval()

with torch.no_grad():

    mean, log_var = model(
        X_test
    )

    variance = torch.exp(
        log_var
    )

    std = torch.sqrt(
        variance
    )
```

The resulting arrays contain:

```text
mean
std
```

for every test material.

---

## 27.6.34 Converting Predictions into Intervals

For an approximate 95% Gaussian prediction interval:

```python
lower = (
    mean
    - 1.96 * std
)

upper = (
    mean
    + 1.96 * std
)
```

The output can then be organized as:

```python
results = {
    "mean": mean,
    "std": std,
    "lower_95": lower,
    "upper_95": upper
}
```

For a materials dataset, the result might conceptually look like:

```text
Material     Mean     Std      Lower     Upper
------------------------------------------------
M1           2.40     0.10     2.20      2.60
M2           3.10     0.20     2.71      3.49
M3           1.80     0.50     0.82      2.78
```

The model therefore provides considerably richer information than a single prediction.

---

## 27.6.35 Evaluating the Predictive Intervals

Suppose the test dataset contains:

```python
y_test
```

and the model produces:

```python
lower
upper
```

Coverage can be calculated using:

```python
covered = (
    (y_test >= lower)
    &
    (y_test <= upper)
)

coverage = covered.mean()

print(
    "Coverage:",
    coverage
)
```

If the nominal interval is 95%, a well-calibrated model should produce empirical coverage reasonably close to:

```text
0.95
```

on a sufficiently large and representative evaluation set.

This is an evaluation of the uncertainty estimate, not merely of the prediction mean.

---

## 27.6.36 What a Good Probabilistic Regression Model Should Provide

A useful probabilistic regression model should ideally provide:

```text
1. Accurate mean predictions

2. Meaningful predictive variance

3. Appropriate uncertainty variation
   across materials

4. Well-calibrated prediction intervals

5. Reasonable sharpness

6. Stable numerical training

7. Scientifically interpretable units
```

Therefore, a probabilistic model should not be evaluated solely using:

```text
RMSE
```

or:

```text
MAE
```

The quality of its uncertainty estimates must also be assessed.

---

## 27.6.37 Limitations of Simple Probabilistic Regression

Gaussian probabilistic regression has several limitations.

### 1. Distributional assumption

It assumes a Gaussian target distribution.

### 2. Uncertainty interpretation

The predicted variance may combine different sources of uncertainty depending on the model.

### 3. Calibration

A theoretically defined 95% interval may not achieve 95% empirical coverage.

### 4. Model misspecification

The true relationship may be nonlinear, multimodal, or otherwise poorly represented.

### 5. Extrapolation

A probabilistic model can still make poorly supported predictions outside the training distribution.

Therefore:

```text
Probabilistic output
≠
Automatically trustworthy uncertainty
```

The uncertainty estimate itself must be validated.

---

## 27.6.38 Probabilistic Regression and Epistemic Uncertainty

A single probabilistic regression model can estimate predictive variance, but the interpretation of that variance requires care.

For example:

```text
Single model
    ↓
Mean + variance
```

does not necessarily provide a clean separation between:

```text
Aleatoric
```

and:

```text
Epistemic
```

A model may primarily represent conditional target variability while failing to fully capture uncertainty caused by limited model knowledge.

Methods such as:

```text
Deep ensembles
Monte Carlo dropout
Bayesian neural networks
Gaussian processes
```

can provide additional mechanisms for representing model uncertainty.

These methods will be developed later in the chapter.

---

## 27.6.39 Probabilistic Regression in Crystal Property Prediction

For crystal machine learning, the input may be represented as:

```text
Composition
+
Lattice
+
Atomic coordinates
```

or through a learned crystal representation:

```text
Crystal
   ↓
Crystal graph
   ↓
GNN
```

A probabilistic GNN can then produce:

```text
Mean property
+
Predictive uncertainty
```

Conceptually:

```text
Crystal Structure
       ↓
Crystal Representation
       ↓
GNN
       ↓
┌───────────────┐
│ Mean          │
│ Variance      │
└───────────────┘
       ↓
Predictive Distribution
```

This allows uncertainty estimation to be integrated directly into crystal-property prediction.

---

## 27.6.40 Key Takeaways

Probabilistic regression extends ordinary regression by predicting a distribution rather than only a point estimate.

The central workflow is:

```text
Input
 ↓
Probabilistic Model
 ↓
Mean + Variance
 ↓
Predictive Distribution
 ↓
Prediction Interval
```

The most important concepts are:

```text
Point prediction
Predictive mean
Predictive variance
Standard deviation
Gaussian predictive distribution
Homoscedastic uncertainty
Heteroscedastic uncertainty
Negative log-likelihood
Prediction intervals
Coverage
Calibration
Sharpness
```

For Materials Informatics, this provides a foundation for reporting predictions together with uncertainty.

However, a probabilistic model does not automatically guarantee that its uncertainty estimates are reliable.

The uncertainty must be evaluated against observed outcomes and calibrated appropriately.

---

## 27.6.41 Transition to Ensemble-Based Uncertainty

A single probabilistic regression model provides a useful predictive distribution, but it does not necessarily capture all uncertainty associated with limited model knowledge.

A complementary strategy is to train multiple models and examine their predictions.

The basic idea is:

```text
Materials Dataset
       │
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
Model  Model Model Model
 1      2     3     4
 └─────┬─────┴─────┘
       ↓
Multiple Predictions
       ↓
Mean + Prediction Spread
       ↓
Uncertainty Estimate
```

The next section, **27.7 Ensemble-Based Uncertainty**, will introduce ensemble prediction, model disagreement, prediction variance, and the use of multiple independently trained models to estimate uncertainty in Materials Machine Learning.

## 27.7 Ensemble-Based Uncertainty

A probabilistic regression model can estimate a predictive distribution for a material property. However, uncertainty can also arise because the model itself has limited knowledge.

A model trained on a finite materials dataset does not know every possible relationship between composition, structure, and properties.

One way to estimate this form of uncertainty is to train **multiple models** and compare their predictions.

This is the basic idea behind **ensemble-based uncertainty estimation**.

Instead of relying on:

```text
One dataset
     ↓
One model
     ↓
One prediction
```

an ensemble uses:

```text
Materials Dataset
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
Model  Model  Model  Model
 1      2      3      4
 └─────┬─────┴─────┘
       ↓
Multiple Predictions
       ↓
Mean + Prediction Spread
       ↓
Uncertainty Estimate
```

If the models make similar predictions, the model disagreement is small.

If the models produce substantially different predictions, the disagreement is large.

Therefore:

```text
Small model disagreement
        ↓
More consistent prediction

Large model disagreement
        ↓
Greater model uncertainty
```

This makes ensembles particularly useful for estimating **epistemic uncertainty**.

---

## 27.7.1 What Is an Ensemble?

An ensemble is a collection of models that are trained to solve the same prediction problem.

For example, suppose the task is to predict the band gap of a crystal.

Instead of training one model:

```text
Crystal
  ↓
Model
  ↓
Band gap
```

we train several:

```text
Crystal
   │
   ├──→ Model 1 ──→ 2.10 eV
   │
   ├──→ Model 2 ──→ 2.18 eV
   │
   ├──→ Model 3 ──→ 2.06 eV
   │
   └──→ Model 4 ──→ 2.14 eV
```

The predictions can then be combined.

For example:

```text
Predictions:

2.10
2.18
2.06
2.14
```

The ensemble mean becomes the final prediction.

The variation among these predictions provides information about model disagreement.

---

## 27.7.2 Why Multiple Models Can Disagree

Two models trained on the same materials dataset do not necessarily learn exactly the same function.

Differences can arise from:

* Different random initialization
* Different bootstrap samples
* Different subsets of training data
* Different optimization trajectories
* Different model architectures
* Different learned parameters

Conceptually:

```text
Same training problem
        ↓
Different models
        ↓
Different learned functions
        ↓
Different predictions
```

If the available training data strongly constrain the relationship, the models may converge toward similar predictions.

If the training data provide weak information about a region of materials space, the models may disagree more strongly.

---

## 27.7.3 Ensemble Mean

Suppose an ensemble contains `M` models.

For a material `x`, let the prediction from model `m` be:

```text
ŷₘ(x)
```

The ensemble mean is:

```text
ŷ_ensemble(x)
=
1/M
Σ ŷₘ(x)
```

In simple terms:

```text
Ensemble prediction
=
Average of individual predictions
```

For example:

```text
Model 1 → 2.10 eV
Model 2 → 2.18 eV
Model 3 → 2.06 eV
Model 4 → 2.14 eV
```

The ensemble prediction is:

```text
(2.10 + 2.18 + 2.06 + 2.14) / 4
```

which gives:

```text
2.12 eV
```

The ensemble therefore produces a central prediction by combining the models.

---

## 27.7.4 Prediction Variance

The predictions can also be used to estimate their spread.

The ensemble prediction variance can be written as:

```text
σ²_ensemble
=
1/M
Σ
(ŷₘ - ŷ_ensemble)²
```

The corresponding standard deviation is:

```text
σ_ensemble
=
√σ²_ensemble
```

A small value means that the models agree closely.

A large value means that they disagree.

For example:

```text
Case A:

2.10
2.11
2.09
2.10
```

has very small disagreement.

Whereas:

```text
Case B:

1.40
2.10
2.90
3.50
```

has much larger disagreement.

The second case should therefore receive a substantially larger ensemble uncertainty estimate.

---

## 27.7.5 Model Disagreement as an Uncertainty Signal

The central idea is:

```text
Model disagreement
        ↓
Information about uncertainty
```

Suppose the model ensemble predicts the band gap of two materials.

### Material A

```text
Model 1 = 2.51 eV
Model 2 = 2.49 eV
Model 3 = 2.52 eV
Model 4 = 2.50 eV
```

The models agree strongly.

### Material B

```text
Model 1 = 1.80 eV
Model 2 = 2.40 eV
Model 3 = 3.10 eV
Model 4 = 2.70 eV
```

The models disagree substantially.

Even if both materials have similar ensemble means, the uncertainty estimates should be different.

This is one of the most useful properties of ensemble-based uncertainty estimation.

---

## 27.7.6 Ensemble Uncertainty and Epistemic Uncertainty

Epistemic uncertainty is associated with limitations in the model's knowledge.

For example, suppose the training dataset contains many materials similar to:

```text
Li-Fe-O
```

but very few materials from:

```text
Li-Co-S
```

A new Li-Co-S crystal may be located in a poorly represented region of the training space.

Different models may respond differently:

```text
New Li-Co-S Crystal
        ↓
 ┌──────┼──────┬──────┐
 ↓      ↓      ↓      ↓
2.1    2.8    3.4    1.9 eV
```

The disagreement may indicate that the model has limited knowledge about that region.

This is why ensemble disagreement is often used as an approximate measure of epistemic uncertainty.

It should not, however, be interpreted as a perfect or universal decomposition of epistemic uncertainty.

---

## 27.7.7 Ensemble Uncertainty Does Not Automatically Equal Total Uncertainty

An important distinction must be maintained.

Suppose an ensemble gives:

```text
Mean = 2.50 eV
Ensemble standard deviation = 0.20 eV
```

The `0.20 eV` value represents disagreement among the ensemble members.

It does not necessarily represent every possible source of uncertainty.

For example, the target may also contain measurement uncertainty:

```text
Experimental uncertainty
```

or intrinsic variability:

```text
Aleatoric uncertainty
```

Therefore:

```text
Total predictive uncertainty
```

may involve multiple components.

Conceptually:

```text
Total uncertainty
        │
        ├── Aleatoric component
        │
        └── Epistemic component
                 ↑
          Ensemble disagreement
```

The exact decomposition depends on the ensemble construction and the probabilistic assumptions of the individual models.

---

## 27.7.8 Simple Ensemble Construction

The simplest ensemble consists of independently trained copies of the same model.

For example:

```text
Model 1
Model 2
Model 3
Model 4
Model 5
```

Each model receives the same training dataset but begins from a different random initialization.

The training process is:

```text
Dataset
   ↓
 ┌─────────────────────┐
 │ Different random    │
 │ initializations     │
 └─────────────────────┘
   ↓
Model 1
Model 2
Model 3
...
Model M
```

Because neural-network optimization is non-convex, different initializations can lead to different learned parameter configurations.

This creates prediction diversity within the ensemble.

---

## 27.7.9 Independent Initialization

Consider a neural network with parameters:

```text
θ
```

The first model begins from:

```text
θ₁⁰
```

while another begins from:

```text
θ₂⁰
```

and so on.

The models are optimized independently:

```text
θ₁⁰ → θ₁*
θ₂⁰ → θ₂*
θ₃⁰ → θ₃*
```

The resulting models may therefore produce slightly different predictions.

The important requirement is that the models should have enough diversity for disagreement to contain useful information.

---

## 27.7.10 Deep Ensemble Architecture

A simple deep ensemble can be represented as:

```text
                    Crystal
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Network 1    Network 2    Network 3
          ↓            ↓            ↓
       Prediction   Prediction   Prediction
          │            │            │
          └────────────┼────────────┘
                       ↓
                Mean + Spread
```

For a crystal graph neural network:

```text
Crystal Structure
       ↓
Crystal Graph
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
GNN1  GNN2  GNN3  GNN4
 ↓     ↓     ↓     ↓
ŷ1    ŷ2    ŷ3    ŷ4
 └─────┬─────┴─────┘
       ↓
Ensemble Mean
+
Ensemble Spread
```

This allows uncertainty estimation to be incorporated into crystal-property prediction.

---

## 27.7.11 PyTorch Deep Ensemble

A basic implementation can create several independent models.

```python
import torch
import torch.nn as nn

models = []

for seed in range(5):

    torch.manual_seed(seed)

    model = nn.Sequential(
        nn.Linear(
            X_train.shape[1],
            64
        ),
        nn.ReLU(),

        nn.Linear(
            64,
            64
        ),
        nn.ReLU(),

        nn.Linear(
            64,
            1
        )
    )

    models.append(model)
```

Each model should then be trained independently.

A simple training function is:

```python
def train_model(
    model,
    X,
    y,
    epochs=200
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction = model(X)

        loss = (
            prediction - y
        ).pow(2).mean()

        loss.backward()

        optimizer.step()

    return model
```

The ensemble can then be trained:

```python
trained_models = []

for model in models:

    model = train_model(
        model,
        X_train,
        y_train
    )

    trained_models.append(
        model
    )
```

Each trained model now represents one member of the ensemble.

---

## 27.7.12 Generating Ensemble Predictions

The models can be evaluated independently.

```python
predictions = []

for model in trained_models:

    model.eval()

    with torch.no_grad():

        prediction = model(
            X_test
        )

    predictions.append(
        prediction
    )
```

The list contains predictions from every ensemble member.

It can be converted into a tensor:

```python
predictions = torch.stack(
    predictions
)
```

The resulting shape is conceptually:

```text
Number of models
×
Number of test materials
```

For example:

```text
5 models × 1000 materials
```

produces:

```text
[5, 1000]
```

---

## 27.7.13 Ensemble Mean in PyTorch

The mean prediction can be calculated using:

```python
ensemble_mean = (
    predictions.mean(
        dim=0
    )
)
```

Here:

```text
dim=0
```

means that predictions from different models are averaged.

The result contains one prediction for each material.

Conceptually:

```text
Model dimension
      ↓
Average
      ↓
One prediction per material
```

---

## 27.7.14 Ensemble Standard Deviation

The prediction spread can be calculated as:

```python
ensemble_std = (
    predictions.std(
        dim=0
    )
)
```

Now each material has:

```text
ensemble_mean
```

and:

```text
ensemble_std
```

For example:

```text
Material A:
Mean = 2.20 eV
Std  = 0.04 eV

Material B:
Mean = 2.20 eV
Std  = 0.35 eV
```

The two materials have the same predicted mean but very different ensemble disagreement.

---

## 27.7.15 Complete Ensemble Prediction Function

A reusable function can be written as:

```python
def ensemble_predict(
    models,
    X
):

    predictions = []

    for model in models:

        model.eval()

        with torch.no_grad():

            prediction = model(X)

        predictions.append(
            prediction
        )

    predictions = torch.stack(
        predictions
    )

    mean = predictions.mean(
        dim=0
    )

    std = predictions.std(
        dim=0
    )

    return (
        mean,
        std,
        predictions
    )
```

The function returns:

```text
Mean prediction
+
Standard deviation
+
Individual predictions
```

Keeping the individual predictions is useful because they allow more detailed analysis of model disagreement.

---

## 27.7.16 Why Keep Individual Predictions?

It may be tempting to calculate only:

```python
mean
std
```

and discard the individual model outputs.

However, retaining the individual predictions is useful.

For example:

```text
Model 1 = 2.10
Model 2 = 2.12
Model 3 = 2.11
Model 4 = 2.09
Model 5 = 2.13
```

shows a tight cluster.

Another material might produce:

```text
Model 1 = 1.20
Model 2 = 2.00
Model 3 = 3.40
Model 4 = 2.80
Model 5 = 1.70
```

The second case has a much more complicated prediction pattern.

Therefore, individual predictions can reveal whether uncertainty comes from:

```text
Small random variation
```

or:

```text
Strong model disagreement
```

---

## 27.7.17 Ensemble Size

The number of models affects the quality and stability of the uncertainty estimate.

A very small ensemble may contain insufficient information.

For example:

```text
2 models
```

provides only a limited estimate of prediction variability.

Larger ensembles provide more samples from the distribution of model predictions:

```text
5 models
10 models
20 models
50 models
```

However, computational cost increases with ensemble size.

The trade-off is:

```text
More models
     ↓
Potentially more stable uncertainty estimate
     ↓
Higher computational cost
```

Therefore, ensemble size should be selected based on:

* Dataset size
* Model complexity
* Computational resources
* Required uncertainty precision

---

## 27.7.18 Ensemble Diversity

An ensemble is useful only if its members provide meaningful diversity.

If all models are effectively identical:

```text
Model 1 = Model 2 = Model 3
```

then:

```text
Prediction spread ≈ 0
```

even if the model itself is wrong.

Therefore, low disagreement does not automatically guarantee low uncertainty.

This is a critical limitation.

The ensemble must have mechanisms that allow its members to explore different plausible models.

Diversity can be introduced through:

```text
Different initialization
Different bootstrap samples
Different training subsets
Different architectures
Different hyperparameters
```

The specific choice depends on the ensemble method.

---

## 27.7.19 Bootstrap Ensembles

One important way to create model diversity is **bootstrap resampling**.

Instead of training every model on exactly the same dataset, different models receive different resampled training datasets.

Suppose the original dataset contains:

```text
D = {x₁, x₂, ..., xₙ}
```

A bootstrap sample is generated by sampling observations with replacement.

For example:

```text
Original:

A B C D E

Bootstrap sample:

A C C E B
```

Another model may receive:

```text
B B D A E
```

and another:

```text
C D D A B
```

The models therefore learn from slightly different datasets.

Conceptually:

```text
Original Dataset
       │
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
Sample Sample Sample Sample
  1      2      3      4
 ↓      ↓      ↓      ↓
Model  Model  Model  Model
```

This is the basis of bootstrap-based ensemble uncertainty.

---

## 27.7.20 Why Bootstrap Sampling Produces Diversity

Consider a small materials dataset.

Some materials may appear repeatedly in one bootstrap sample but not in another.

Therefore, the effective training emphasis differs between models.

This produces:

```text
Different training experience
        ↓
Different fitted functions
        ↓
Different predictions
        ↓
Prediction disagreement
```

The resulting disagreement can be used as an uncertainty signal.

Bootstrap ensembles are particularly useful when the dataset is not extremely large.

---

## 27.7.21 Bootstrap Sampling in Python

Bootstrap samples can be generated using NumPy.

```python
import numpy as np

def bootstrap_sample(
    X,
    y
):

    n = len(X)

    indices = np.random.choice(
        n,
        size=n,
        replace=True
    )

    return (
        X[indices],
        y[indices]
    )
```

A complete bootstrap ensemble can then be constructed:

```python
models = []

for i in range(10):

    X_boot, y_boot = (
        bootstrap_sample(
            X_train,
            y_train
        )
    )

    model = train_model(
        X_boot,
        y_boot
    )

    models.append(
        model
    )
```

The resulting models have been trained on different bootstrap realizations of the training data.

---

## 27.7.22 Bootstrap Ensemble for Materials Data

Suppose the dataset contains:

```text
Crystal features
+
Band gap
```

The workflow becomes:

```text
Materials Dataset
       ↓
Bootstrap Sample 1 → Model 1
Bootstrap Sample 2 → Model 2
Bootstrap Sample 3 → Model 3
...
Bootstrap Sample M → Model M
       ↓
Predictions
       ↓
Mean + Standard Deviation
```

The final result for each material can be represented as:

```text
Material
Mean prediction
Ensemble standard deviation
```

This gives a simple uncertainty-aware prediction workflow.

---

## 27.7.23 Random Forest as an Ensemble

Random Forest provides another important example of ensemble learning.

A Random Forest contains many decision trees.

Conceptually:

```text
Dataset
   ↓
 ┌──┼──┬──┬──┐
 ↓  ↓  ↓  ↓  ↓
Tree Tree Tree Tree Tree
 1    2    3    4    5
 └──┬──┴──┬──┘
    ↓
Average prediction
```

Each tree is trained using randomized sampling and randomized feature selection.

Therefore, the trees can produce different predictions.

For regression:

```text
Tree 1 → 2.4
Tree 2 → 2.5
Tree 3 → 2.3
Tree 4 → 2.7
Tree 5 → 2.4
```

The forest prediction is approximately the average:

```text
2.46
```

The variation among the tree predictions can also provide an approximate uncertainty signal.

---

## 27.7.24 Random Forest Tree-Level Predictions

Scikit-learn exposes the individual trees inside a Random Forest.

For example:

```python
from sklearn.ensemble import (
    RandomForestRegressor
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)
```

The individual estimators can be accessed using:

```python
tree_predictions = np.array([
    tree.predict(X_test)
    for tree in rf.estimators_
])
```

The resulting array has approximately:

```text
100 trees
×
number of test materials
```

The mean prediction is:

```python
rf_mean = (
    tree_predictions.mean(
        axis=0
    )
)
```

and the tree-to-tree standard deviation is:

```python
rf_std = (
    tree_predictions.std(
        axis=0
    )
)
```

This provides a straightforward estimate of prediction dispersion inside the forest.

---

## 27.7.25 Important Limitation of Random Forest Variance

The standard deviation of Random Forest tree predictions should not automatically be interpreted as a formally calibrated predictive standard deviation.

The trees are not independent posterior samples from a Bayesian model.

Therefore:

```text
Tree prediction spread
```

is better understood as an **ensemble disagreement measure**.

It can be useful for identifying difficult or unfamiliar predictions, but it requires validation before being interpreted as a quantitative uncertainty interval.

This distinction is essential in research-grade Materials Informatics.

---

## 27.7.26 Ensemble Uncertainty Across Materials Space

Ensemble disagreement can also be analyzed across the materials dataset.

Suppose materials are represented by feature vectors:

```text
x₁, x₂, ..., xₙ
```

For each material, calculate:

```text
ensemble mean
ensemble standard deviation
```

The resulting dataset can be visualized conceptually as:

```text
Materials Space

Low uncertainty:
● ● ● ● ●

Medium uncertainty:
○ ○ ○

High uncertainty:
× ×
```

This can reveal whether uncertainty is concentrated in particular regions of materials space.

For example:

```text
Known chemical family
→ low disagreement

Rare chemical family
→ high disagreement
```

This provides a connection between model uncertainty and the coverage of the training dataset.

---

## 27.7.27 Uncertainty Versus Training Density

A useful analysis is to compare ensemble uncertainty with the density of training examples.

Conceptually:

```text
Training density ↑
        ↓
Often better representation
        ↓
Often lower epistemic uncertainty
```

while:

```text
Training density ↓
        ↓
Poorer representation
        ↓
Potentially greater model disagreement
```

This relationship is not guaranteed.

A dense region can still be difficult to model, and a sparse region can sometimes be easy to predict.

Therefore, training density should be treated as an explanatory variable rather than a direct uncertainty measure.

---

## 27.7.28 Ensemble Predictions for Crystal Graph Models

The same concept applies to graph neural networks.

Suppose a crystal graph is:

```text
G = (V, E)
```

where:

```text
V = atoms
E = interactions
```

A GNN predicts a material property:

```text
G
↓
GNN
↓
ŷ
```

An ensemble uses multiple GNNs:

```text
G
 │
 ├──→ GNN 1 → ŷ₁
 │
 ├──→ GNN 2 → ŷ₂
 │
 ├──→ GNN 3 → ŷ₃
 │
 └──→ GNN 4 → ŷ₄
 │
 ↓
Mean + Spread
```

The crystal graph is therefore passed through several independently trained models.

The resulting prediction disagreement provides an uncertainty signal at the graph level.

---

## 27.7.29 Deep Ensemble for a Crystal GNN

A research workflow might be:

```text
Crystal Structure
       ↓
Pymatgen
       ↓
Crystal Graph
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
GNN1  GNN2  GNN3  GNN4
 ↓     ↓     ↓     ↓
ŷ1    ŷ2    ŷ3    ŷ4
 └─────┬─────┴─────┘
       ↓
Mean Prediction
       +
Ensemble Spread
```

For example:

```text
Band gap predictions:

GNN 1 = 1.92 eV
GNN 2 = 2.01 eV
GNN 3 = 1.88 eV
GNN 4 = 1.97 eV
```

The ensemble mean provides the final prediction.

The disagreement provides an uncertainty indicator.

---

## 27.7.30 Ensemble Uncertainty and Crystal Diversity

Crystal datasets can contain very different structural families.

For example:

```text
Cubic
Tetragonal
Orthorhombic
Monoclinic
Triclinic
```

An ensemble may show different levels of disagreement across these regions.

Suppose the training dataset contains many cubic materials but relatively few triclinic structures.

A new triclinic structure may produce:

```text
GNN 1 → 2.1
GNN 2 → 2.4
GNN 3 → 3.0
GNN 4 → 2.2
```

while a well-represented cubic structure may produce:

```text
GNN 1 → 2.41
GNN 2 → 2.43
GNN 3 → 2.40
GNN 4 → 2.42
```

The second prediction has much stronger agreement.

This illustrates how ensemble disagreement can provide information about regions where the learned model is less certain.

---

## 27.7.31 Combining Mean and Ensemble Spread

For every material, an ensemble can therefore produce:

```text
Mean prediction
+
Prediction spread
```

A simple results table might be:

| Material | Mean prediction | Ensemble std |
| -------- | --------------: | -----------: |
| M1       |         2.41 eV |      0.04 eV |
| M2       |         2.72 eV |      0.09 eV |
| M3       |         1.85 eV |      0.31 eV |
| M4       |         3.10 eV |      0.52 eV |

The table should not be interpreted as saying that the final uncertainty intervals are automatically calibrated.

Instead, it provides a first uncertainty signal that must be evaluated.

---

## 27.7.32 Evaluating Ensemble Uncertainty

A useful evaluation compares:

```text
Prediction error
```

with:

```text
Ensemble uncertainty
```

For each test material:

```text
Absolute error
=
|y - ŷ_ensemble|
```

and:

```text
Ensemble uncertainty
=
σ_ensemble
```

can be calculated.

The desired qualitative behavior is:

```text
Low uncertainty
        ↔
Usually smaller errors

High uncertainty
        ↔
Potentially larger errors
```

The relationship should be tested quantitatively rather than assumed.

---

## 27.7.33 Error Versus Ensemble Uncertainty

A useful diagnostic is a scatter plot:

```text
Absolute Error
     │
     │             ×
     │        ×
     │
     │   ×
     │ ×
     └────────────────────
       Ensemble Uncertainty
```

If uncertainty is informative, larger uncertainty should generally be associated with larger prediction errors.

However, the relationship does not need to be perfect.

A model can occasionally be:

```text
Confident and wrong
```

or:

```text
Uncertain but accurate
```

Therefore, ensemble uncertainty should be treated as a probabilistic signal rather than a guarantee.

---

## 27.7.34 Ranking Materials by Uncertainty

After calculating ensemble standard deviation, materials can be ranked.

```python
uncertainty = (
    ensemble_std.numpy()
)

ranking = np.argsort(
    -uncertainty
)
```

This produces materials ordered from highest uncertainty to lowest uncertainty.

The researcher can then inspect the most uncertain structures.

For example:

```text
Rank  Material   Uncertainty
1     M184       0.82 eV
2     M093       0.71 eV
3     M421       0.65 eV
4     M117       0.60 eV
```

This does not itself provide a scientific explanation for the uncertainty.

It simply identifies predictions where the ensemble members disagree most strongly.

---

## 27.7.35 Sources of High Ensemble Disagreement

High ensemble disagreement can occur for several reasons.

### Sparse training data

```text
Few similar materials
```

### Distribution shift

```text
New material differs strongly
from training materials
```

### Difficult structure-property relationship

```text
Similar structures
→ strongly varying properties
```

### Model instability

```text
Different models learn
substantially different functions
```

### Noisy labels

```text
Training targets contain
substantial variability
```

Therefore:

```text
High uncertainty
```

is a signal that requires interpretation.

It should not automatically be assigned to a single cause.

---

## 27.7.36 Ensemble Uncertainty Does Not Replace Validation

A common mistake is:

```text
Train ensemble
↓
Calculate standard deviation
↓
Call it reliable uncertainty
```

This is insufficient.

The ensemble uncertainty must be evaluated using held-out data.

The correct workflow is:

```text
Training Dataset
       ↓
Train Ensemble
       ↓
Validation / Test Predictions
       ↓
Calculate Error
       +
Calculate Ensemble Spread
       ↓
Compare Error and Uncertainty
       ↓
Evaluate Calibration
```

This establishes whether the uncertainty estimate actually contains useful information.

---

## 27.7.37 Ensemble-Based Prediction Intervals

An ensemble standard deviation can sometimes be used to construct an approximate interval:

```text
ŷ ± zσ_ensemble
```

For example:

```text
Mean = 2.40 eV
Ensemble std = 0.15 eV
```

gives the approximate interval:

```text
2.40 ± 1.96(0.15)
```

or approximately:

```text
[2.11, 2.69] eV
```

However, this interval should **not automatically be called a calibrated 95% prediction interval**.

Its validity depends on how the ensemble uncertainty behaves and whether it has been calibrated and evaluated on appropriate held-out data.

This distinction is essential.

---

## 27.7.38 Advantages of Ensemble-Based Uncertainty

Ensembles have several practical advantages.

### 1. Conceptually simple

Multiple models are trained and their predictions are compared.

### 2. Compatible with many model types

Ensembles can be constructed from:

```text
Linear models
Decision trees
Random forests
Neural networks
GNNs
```

### 3. Useful for deep learning

Deep ensembles work naturally with PyTorch-based materials models.

### 4. Can capture model disagreement

Different models can provide information about epistemic uncertainty.

### 5. Flexible

The same principle can be applied to:

```text
Composition models
Descriptor models
Crystal GNNs
```

---

## 27.7.39 Limitations of Ensemble-Based Uncertainty

Ensembles also have important limitations.

### Computational cost

Training several models requires more computation than training one.

```text
1 model
↓
1 training cost

10 models
↓
approximately 10 training runs
```

### Storage

Each trained model must usually be stored.

### Ensemble dependence

Models may not be sufficiently diverse.

### Interpretation

Prediction disagreement does not provide a perfect decomposition of uncertainty.

### Calibration

Ensemble spread does not automatically produce calibrated prediction intervals.

Therefore:

```text
Ensemble uncertainty
```

is useful but must be validated.

---

## 27.7.40 Ensemble Workflow for Materials Informatics

A practical workflow is:

```text
Materials Dataset
       ↓
Train / Validation / Test Split
       ↓
Create Multiple Models
       ↓
Train Independently
       ↓
Generate Predictions
       ↓
Calculate Ensemble Mean
       ↓
Calculate Ensemble Spread
       ↓
Compare Against Test Errors
       ↓
Evaluate Calibration
       ↓
Report Prediction + Uncertainty
```

For crystal materials:

```text
Crystal Structure
       ↓
Pymatgen / Crystal Representation
       ↓
Crystal Graph
       ↓
Multiple GNNs
       ↓
Predictions
       ↓
Mean + Spread
```

---

## 27.7.41 Example Research Interpretation

Suppose an ensemble predicts the band gap of three candidate crystals.

```text
Candidate A:
2.20 ± 0.04 eV

Candidate B:
2.35 ± 0.11 eV

Candidate C:
2.40 ± 0.55 eV
```

The predictions indicate:

```text
A:
High model agreement

B:
Moderate model agreement

C:
Strong model disagreement
```

Candidate C therefore deserves greater scrutiny of the model's knowledge about that particular material.

The correct scientific interpretation is not:

```text
Candidate C is wrong.
```

Instead:

```text
Candidate C has substantially greater
ensemble disagreement and therefore a
less certain model prediction.
```

This distinction is fundamental to responsible uncertainty reporting.

---

## 27.7.42 Key Takeaways

Ensemble-based uncertainty estimation uses multiple predictive models to measure disagreement.

The basic workflow is:

```text
Multiple Models
      ↓
Multiple Predictions
      ↓
Mean
+
Prediction Spread
      ↓
Uncertainty Signal
```

The most important concepts are:

```text
Ensemble
Ensemble mean
Prediction variance
Prediction standard deviation
Model disagreement
Bootstrap ensemble
Random Forest ensemble
Deep ensemble
Ensemble diversity
Epistemic uncertainty
Calibration
Held-out evaluation
```

The central principle is:

```text
Agreement between models
        ↓
More consistent prediction

Disagreement between models
        ↓
Potentially greater epistemic uncertainty
```

However:

```text
Ensemble spread
≠
Automatically calibrated uncertainty
```

The uncertainty estimate must be evaluated against independent observations.

---

## 27.7.43 Transition to Bootstrap Ensembles

A simple ensemble can be created by changing model initialization, but a more explicit way to generate model diversity is to expose different models to different resampled versions of the training dataset.

This leads to **bootstrap ensembles**.

The next section will therefore focus specifically on:

```text
Bootstrap Resampling
        ↓
Different Training Sets
        ↓
Multiple Models
        ↓
Multiple Predictions
        ↓
Prediction Distribution
        ↓
Uncertainty Estimate
```

The discussion will remain focused on **bootstrap-based uncertainty estimation for Materials Machine Learning**.

## 27.8 Bootstrap Ensembles

Bootstrap ensembles provide a simple and powerful way to estimate model uncertainty from finite datasets.

The central idea is to train several models on different **bootstrap samples** of the same training dataset.

Instead of asking one model to represent the entire uncertainty associated with the available training data, we construct multiple slightly different training datasets and observe how the resulting models behave.

The workflow is:

```text
Original Training Dataset
          ↓
   Bootstrap Resampling
          ↓
 ┌────────┼────────┬────────┐
 ↓        ↓        ↓        ↓
Sample 1 Sample 2 Sample 3 Sample M
 ↓        ↓        ↓        ↓
Model 1  Model 2  Model 3  Model M
 └────────┼────────┴────────┘
          ↓
   Multiple Predictions
          ↓
   Mean + Prediction Spread
          ↓
   Uncertainty Estimate
```

This method is particularly useful when the available materials dataset is limited.

---

## 27.8.1 What Is Bootstrap Resampling?

Suppose a training dataset contains `N` observations:

```text
D = {x₁, x₂, x₃, ..., xₙ}
```

A bootstrap sample is created by randomly selecting `N` observations **with replacement**.

The important word is **replacement**.

After an observation is selected, it is returned to the population before the next observation is selected.

Therefore, the same observation can appear multiple times in one bootstrap sample.

For example, consider a small dataset:

```text
Original:

A
B
C
D
E
```

A possible bootstrap sample is:

```text
A
C
C
E
B
```

Another could be:

```text
B
B
D
A
E
```

And another:

```text
C
D
D
C
A
```

Each sample has the same number of observations as the original dataset, but the composition is different.

---

## 27.8.2 Why Sampling With Replacement Matters

If sampling were performed without replacement, every sample containing `N` observations would simply reproduce the original dataset.

There would be no useful diversity.

Bootstrap sampling instead allows:

```text
Repeated observations
+
Omitted observations
```

within each training sample.

Therefore, every bootstrap model receives a slightly different view of the available data.

Conceptually:

```text
Original Dataset
       ↓
Random sampling with replacement
       ↓
Different datasets
       ↓
Different fitted models
       ↓
Different predictions
```

This variation is what makes bootstrap ensembles useful for uncertainty estimation.

---

## 27.8.3 Bootstrap Samples and Materials Datasets

Consider a materials dataset containing:

```text
Crystal Structure
Composition
Descriptors
Target Property
```

For example:

```text
Material     Band Gap
M1           1.20 eV
M2           2.10 eV
M3           0.80 eV
M4           3.40 eV
M5           1.70 eV
...
```

A bootstrap sample is generated from the training data.

For example:

```text
Bootstrap Dataset 1

M1
M4
M4
M7
M8
M2
...
```

A second model receives:

```text
Bootstrap Dataset 2

M2
M3
M5
M5
M9
M1
...
```

The models therefore experience different empirical versions of the same materials dataset.

---

## 27.8.4 Bootstrap Models

Suppose we create `M` bootstrap samples.

For each sample, we train one model:

```text
D₁ → Model 1
D₂ → Model 2
D₃ → Model 3
...
Dₘ → Model M
```

For a new material `x`, the models produce:

```text
ŷ₁(x)
ŷ₂(x)
ŷ₃(x)
...
ŷₘ(x)
```

The ensemble mean is:

```text
ŷ(x)
=
1/M
Σ ŷₘ(x)
```

The prediction spread can then be estimated from the variation among these predictions.

Thus:

```text
Bootstrap sampling
        ↓
Model diversity
        ↓
Prediction diversity
        ↓
Uncertainty estimate
```

---

## 27.8.5 Bootstrap Distribution of Predictions

One of the most useful ways to understand bootstrap uncertainty is to consider the collection of predictions as a distribution.

Suppose ten bootstrap models predict the band gap of a new crystal:

```text
1.82
1.91
1.87
1.95
1.84
1.89
1.92
1.86
1.90
1.88
```

The predictions form a relatively narrow distribution.

The mean is approximately:

```text
1.88 eV
```

and the spread is small.

Now consider another crystal:

```text
1.20
1.65
2.10
2.80
1.40
2.30
2.70
1.80
2.45
1.10
```

The predictions are much more dispersed.

The second material therefore exhibits greater bootstrap model uncertainty.

---

## 27.8.6 Bootstrap Mean and Standard Deviation

Suppose the bootstrap predictions are stored in an array:

```python
predictions = np.array([
    1.82,
    1.91,
    1.87,
    1.95,
    1.84,
    1.89,
    1.92,
    1.86,
    1.90,
    1.88
])
```

The mean can be calculated using:

```python
mean_prediction = (
    predictions.mean()
)
```

The standard deviation can be calculated using:

```python
uncertainty = (
    predictions.std()
)
```

The resulting representation is:

```text
Prediction
=
mean_prediction

Uncertainty
=
bootstrap standard deviation
```

This provides a simple uncertainty-aware prediction.

---

## 27.8.7 Bootstrap Ensemble Implementation

A general bootstrap ensemble can be implemented using scikit-learn.

Consider a regression problem:

```python
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor
)
```

A bootstrap sample can be created with:

```python
def make_bootstrap_sample(
    X,
    y
):

    n_samples = len(X)

    indices = np.random.choice(
        n_samples,
        size=n_samples,
        replace=True
    )

    return (
        X[indices],
        y[indices]
    )
```

The function returns a dataset containing the same number of observations as the original training dataset.

---

## 27.8.8 Training a Bootstrap Ensemble

We can now train several models.

```python
def train_bootstrap_ensemble(
    X_train,
    y_train,
    n_models=20
):

    models = []

    for i in range(n_models):

        X_boot, y_boot = (
            make_bootstrap_sample(
                X_train,
                y_train
            )
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=i
        )

        model.fit(
            X_boot,
            y_boot
        )

        models.append(
            model
        )

    return models
```

The workflow is:

```text
Original training data
        ↓
Bootstrap sample
        ↓
Train model
        ↓
Store model
        ↓
Repeat
```

The resulting list contains multiple trained models.

---

## 27.8.9 Generating Predictions

The trained models can be applied to a test dataset.

```python
def bootstrap_predict(
    models,
    X_test
):

    predictions = []

    for model in models:

        prediction = (
            model.predict(
                X_test
            )
        )

        predictions.append(
            prediction
        )

    return np.array(
        predictions
    )
```

If there are:

```text
20 models
```

and:

```text
500 test materials
```

the prediction array has the conceptual shape:

```text
20 × 500
```

where each row corresponds to one bootstrap model.

---

## 27.8.10 Calculating Bootstrap Mean

The mean prediction for every test material is:

```python
bootstrap_mean = (
    predictions.mean(
        axis=0
    )
)
```

Here:

```text
axis=0
```

means that predictions from different bootstrap models are averaged.

The output contains one prediction for each material.

For example:

```text
Material 1 → 2.10 eV
Material 2 → 1.84 eV
Material 3 → 3.02 eV
```

---

## 27.8.11 Calculating Bootstrap Uncertainty

The prediction standard deviation is:

```python
bootstrap_std = (
    predictions.std(
        axis=0
    )
)
```

This produces one uncertainty estimate for every test material.

For example:

| Material | Mean prediction | Bootstrap std |
| -------- | --------------: | ------------: |
| M1       |         2.10 eV |       0.04 eV |
| M2       |         1.84 eV |       0.12 eV |
| M3       |         3.02 eV |       0.48 eV |

The third material produces substantially more disagreement among the bootstrap models.

---

## 27.8.12 Complete Bootstrap Prediction Workflow

The complete workflow can therefore be written as:

```text
Materials Dataset
       ↓
Training / Test Split
       ↓
Bootstrap Resampling
       ↓
Multiple Training Sets
       ↓
Multiple Models
       ↓
Predictions
       ↓
Mean Prediction
       +
Prediction Spread
       ↓
Uncertainty Estimate
```

This is one of the simplest practical approaches for adding uncertainty estimation to an existing Materials Machine Learning workflow.

---

## 27.8.13 Bootstrap Confidence Intervals

Bootstrap predictions can also be used to construct empirical intervals.

Suppose the bootstrap predictions for one material are:

```text
1.82
1.91
1.87
1.95
1.84
1.89
1.92
1.86
1.90
1.88
```

Instead of assuming a Gaussian distribution, the empirical prediction distribution can be examined directly.

For example, percentile-based limits can be calculated:

```python
lower = np.percentile(
    predictions,
    2.5
)

upper = np.percentile(
    predictions,
    97.5
)
```

This produces an empirical interval:

```text
[lower, upper]
```

The advantage is that the interval is based directly on the observed bootstrap prediction distribution.

However, the interpretation of this interval requires care.

It is not automatically a calibrated 95% predictive interval for future experimental observations.

It describes the distribution generated by the bootstrap modeling procedure.

---

## 27.8.14 Percentile-Based Bootstrap Intervals

Suppose the bootstrap predictions are:

```text
1.70
1.75
1.80
1.82
1.85
1.87
1.90
1.94
2.00
2.05
```

The empirical distribution can be summarized using percentiles.

For example:

```text
2.5th percentile
50th percentile
97.5th percentile
```

The resulting structure is:

```text
Lower percentile
        ↓
Bootstrap prediction distribution
        ↓
Upper percentile
```

This approach is useful when the bootstrap predictions are not approximately symmetric.

For example, the distribution might be:

```text
        **
      ****
    ******
  ********
**********
```

rather than a symmetric bell-shaped distribution.

In such cases, percentile intervals can provide a more faithful summary of the bootstrap distribution than:

```text
mean ± 1.96 × standard deviation
```

---

## 27.8.15 Bootstrap Uncertainty and Small Materials Datasets

Materials datasets are often smaller than datasets encountered in general machine learning.

This makes uncertainty estimation particularly important.

Suppose a dataset contains only:

```text
500 crystals
```

rather than:

```text
5,000,000 samples
```

A model trained on only 500 examples may have limited information about some chemical and structural regions.

Bootstrap resampling allows us to examine how sensitive the model is to changes in the available training observations.

Conceptually:

```text
Limited Dataset
      ↓
Resample Dataset
      ↓
Train Multiple Models
      ↓
Observe Prediction Stability
```

If predictions remain stable across bootstrap samples, the model may have learned a relatively robust relationship.

If predictions change substantially, the model may be sensitive to the particular composition of the training data.

---

## 27.8.16 Bootstrap Uncertainty as Data-Sensitivity Analysis

This provides an important interpretation of bootstrap uncertainty.

Bootstrap uncertainty can answer a question such as:

> How sensitive is the model prediction to the particular finite sample used for training?

Suppose a new crystal receives:

```text
Model 1 → 2.10 eV
Model 2 → 2.11 eV
Model 3 → 2.09 eV
Model 4 → 2.12 eV
```

The prediction is stable.

Now suppose another crystal receives:

```text
Model 1 → 1.20 eV
Model 2 → 2.00 eV
Model 3 → 2.80 eV
Model 4 → 3.10 eV
```

The prediction is highly sensitive to the training sample.

Therefore:

```text
Bootstrap disagreement
        ↓
Training-data sensitivity
```

is an important way to understand bootstrap-based uncertainty.

---

## 27.8.17 Out-of-Bag Observations

Because bootstrap sampling uses replacement, some observations from the original training dataset are not selected in a particular bootstrap sample.

These observations are called **out-of-bag observations** for that model.

For example:

```text
Original:

A B C D E F G H
```

A bootstrap sample might be:

```text
A C C D F G G H
```

The observations:

```text
B
E
```

were not selected.

Therefore:

```text
B and E
```

are out-of-bag observations for that particular bootstrap model.

This provides a useful mechanism for evaluating model predictions without using exactly the same observations used to fit that model.

---

## 27.8.18 Why Out-of-Bag Data Matter

For every bootstrap model:

```text
Bootstrap sample
        ↓
Training observations

Out-of-bag observations
        ↓
Unused observations
```

The out-of-bag observations can be used to evaluate that particular model.

This is especially useful in ensemble algorithms such as Random Forest.

The concept is:

```text
Training subset
+
Unused observations
```

rather than requiring a completely separate validation set for every bootstrap model.

However, a properly designed independent test set remains important for final research evaluation.

---

## 27.8.19 Out-of-Bag Prediction in Materials ML

Suppose a materials dataset contains:

```text
M1
M2
M3
...
M1000
```

For one bootstrap model:

```text
Selected:
M1, M2, M4, M4, M6, ...
```

while:

```text
Out-of-bag:
M3, M5, M8, ...
```

The model can predict the properties of the out-of-bag materials.

Repeating this process across many bootstrap models allows each material to receive predictions from models that did not train directly on that observation.

This provides a useful internal evaluation mechanism.

---

## 27.8.20 Bootstrap Ensembles for Descriptor-Based Materials Models

Bootstrap ensembles are not limited to neural networks.

Suppose materials are represented using descriptors:

```text
Composition features
+
Structural features
+
Electronic descriptors
```

A conventional regression model can be used:

```text
Descriptors
    ↓
Bootstrap Models
    ↓
Property Predictions
    ↓
Mean + Uncertainty
```

For example:

```python
from sklearn.ensemble import (
    RandomForestRegressor
)

models = []

for seed in range(20):

    X_boot, y_boot = (
        make_bootstrap_sample(
            X_train,
            y_train
        )
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=seed
    )

    model.fit(
        X_boot,
        y_boot
    )

    models.append(
        model
    )
```

This approach can be applied to properties such as:

```text
Band gap
Formation energy
Bulk modulus
Elastic modulus
Density
Thermal conductivity
```

provided that the corresponding dataset and representation are appropriate.

---

## 27.8.21 Bootstrap Ensembles for Crystal Graph Models

The same concept can be applied to GNNs.

Suppose the dataset contains crystal graphs:

```text
G₁
G₂
G₃
...
Gₙ
```

Each bootstrap sample contains a resampled collection of these graphs.

For example:

```text
Bootstrap 1:
G1 G4 G4 G7 G9 ...

Bootstrap 2:
G2 G3 G5 G5 G8 ...

Bootstrap 3:
G1 G2 G6 G6 G10 ...
```

A GNN is trained independently on each sample.

The workflow becomes:

```text
Crystal Graph Dataset
        ↓
Bootstrap Resampling
        ↓
 ┌──────┼──────┬──────┐
 ↓      ↓      ↓      ↓
GNN 1  GNN 2  GNN 3  GNN M
 ↓      ↓      ↓      ↓
ŷ₁     ŷ₂     ŷ₃     ŷₘ
 └──────┼──────┴──────┘
        ↓
Mean + Spread
```

This provides a direct route to uncertainty-aware crystal GNN predictions.

---

## 27.8.22 Bootstrap Ensembles and Crystal Structure Diversity

Crystal datasets can contain highly diverse structures.

For example:

```text
Different compositions
Different crystal systems
Different coordination environments
Different lattice parameters
Different local bonding environments
```

If a particular structural family is represented by only a small number of examples, bootstrap resampling can make the model sensitive to whether those examples appear in a particular bootstrap sample.

This can lead to larger prediction variation for materials related to poorly represented structural families.

Thus:

```text
Structural representation
        ↓
Bootstrap sampling
        ↓
Model variation
        ↓
Prediction variation
```

can reveal sensitivity to the available structural training data.

---

## 27.8.23 Bootstrap Uncertainty Is Not a Guarantee of Correctness

A model can produce low bootstrap uncertainty and still be wrong.

For example:

```text
Model 1 → 2.10 eV
Model 2 → 2.11 eV
Model 3 → 2.09 eV
Model 4 → 2.10 eV
```

The models agree strongly.

But suppose the true value is:

```text
3.00 eV
```

Then:

```text
Prediction error = 0.90 eV
```

despite very small bootstrap disagreement.

This can happen when all models share the same systematic error.

Therefore:

```text
Low bootstrap uncertainty
```

does not imply:

```text
Guaranteed accurate prediction
```

This is one of the most important limitations of ensemble-based uncertainty estimation.

---

## 27.8.24 Bootstrap Uncertainty and Systematic Error

Suppose every model uses the same descriptor representation.

If that representation systematically fails to capture an important physical feature, every bootstrap model may make a similar error.

For example:

```text
Important physical effect
        ↓
Not represented in features
        ↓
All models miss the effect
        ↓
Similar predictions
        ↓
Low ensemble disagreement
```

The models may therefore be confidently wrong.

This demonstrates why uncertainty estimation must always be considered together with:

```text
Model validation
+
Representation quality
+
Test-set performance
```

---

## 27.8.25 Practical Interpretation of Bootstrap Uncertainty

A useful interpretation is:

```text
Bootstrap uncertainty asks:

"How much does the prediction change
when the available training dataset
is perturbed through resampling?"
```

It does not automatically answer:

```text
"What is the exact probability that
the prediction is correct?"
```

These are different questions.

Therefore, bootstrap uncertainty should be reported carefully.

A research paper might report:

```text
Predicted band gap:
2.31 eV

Bootstrap ensemble standard deviation:
0.14 eV
```

and clearly define how that value was obtained.

---

## 27.8.26 Recommended Bootstrap Workflow

For a research-grade Materials Informatics study, a practical workflow is:

```text
Materials Dataset
        ↓
Data Cleaning
        ↓
Train / Validation / Test Split
        ↓
Bootstrap Resampling
        ↓
Train Multiple Models
        ↓
Generate Test Predictions
        ↓
Calculate Mean
        ↓
Calculate Prediction Spread
        ↓
Compare Spread with Test Error
        ↓
Evaluate Calibration
        ↓
Report Prediction + Uncertainty
```

The independent test set should remain untouched during model development.

This prevents the uncertainty analysis from becoming overly optimistic.

---

## 27.8.27 Example of a Research Result

Suppose a bootstrap ensemble contains 30 models.

For a new material, the predictions are summarized as:

```text
Mean prediction:
2.46 eV

Bootstrap standard deviation:
0.18 eV
```

The result can be reported as:

```text
Band gap prediction = 2.46 eV
Bootstrap ensemble spread = 0.18 eV
```

If the uncertainty estimate has been properly calibrated and validated, an appropriate prediction interval may also be reported.

Without such validation, the standard deviation should be described explicitly as an **ensemble spread estimate**, rather than automatically presenting it as a statistically guaranteed confidence interval.

---

## 27.8.28 Key Takeaways

Bootstrap ensembles create multiple training datasets by sampling the original dataset with replacement.

The main workflow is:

```text
Original Dataset
       ↓
Bootstrap Samples
       ↓
Multiple Models
       ↓
Multiple Predictions
       ↓
Prediction Distribution
       ↓
Mean + Spread
```

The most important concepts are:

```text
Bootstrap sampling
Sampling with replacement
Bootstrap model
Bootstrap prediction distribution
Out-of-bag observations
Prediction mean
Prediction standard deviation
Percentile intervals
Training-data sensitivity
Ensemble disagreement
```

Bootstrap ensembles are useful because they reveal how predictions change when the finite training dataset is perturbed.

However:

```text
Bootstrap spread
≠
Guaranteed prediction interval
```

and:

```text
Low bootstrap spread
≠
Guaranteed correctness
```

Therefore, bootstrap uncertainty must be evaluated against independent test observations and interpreted together with model performance and data coverage.

---

## 27.8.29 Transition to Random Forest Uncertainty

Bootstrap resampling provides the foundation for understanding uncertainty from multiple fitted models.

Random Forest extends this idea naturally by combining many randomized decision trees, where bootstrap samples and randomized feature selection create an ensemble of predictors.

The next section therefore focuses specifically on:

```text
Random Forest
      ↓
Individual Tree Predictions
      ↓
Tree-to-Tree Variation
      ↓
Prediction Spread
      ↓
Random Forest Uncertainty
```

The discussion will remain focused on **uncertainty estimation from Random Forest models**.

## 27.9 Random Forest Uncertainty

Random Forest models provide a natural starting point for uncertainty estimation because a Random Forest is already an ensemble of many decision trees.

Instead of producing a prediction from one decision tree, a Random Forest combines predictions from many trees:

```text
Training Dataset
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
Tree 1 Tree 2 Tree 3 ... Tree M
 ↓     ↓     ↓
ŷ₁    ŷ₂    ŷ₃    ... ŷₘ
 └─────┼─────┴─────┘
       ↓
   Mean Prediction
       +
   Tree Variation
```

The individual trees do not necessarily produce identical predictions.

That disagreement can provide useful information about the stability of the model prediction.

For Materials Informatics, this is particularly useful because a Random Forest can provide both:

```text
Predicted Materials Property
```

and an estimate of:

```text
Prediction Variation
```

from the individual trees.

---

## 27.9.1 Why Random Forests Can Provide Uncertainty Information

A standard regression model produces one prediction:

```text
ŷ = f(x)
```

A Random Forest instead contains many decision trees:

```text
f₁(x)
f₂(x)
f₃(x)
...
fₘ(x)
```

The final Random Forest prediction is usually the average:

```text
ŷ = 1/M Σ fₘ(x)
```

where:

```text
M = number of trees
```

For a new material, the individual trees may predict:

```text
Tree 1 → 2.10 eV
Tree 2 → 2.15 eV
Tree 3 → 2.08 eV
Tree 4 → 2.12 eV
Tree 5 → 2.11 eV
```

The predictions are close together.

The forest therefore exhibits low tree-to-tree variation.

For another material:

```text
Tree 1 → 1.20 eV
Tree 2 → 2.00 eV
Tree 3 → 2.80 eV
Tree 4 → 1.50 eV
Tree 5 → 2.70 eV
```

the trees disagree substantially.

This disagreement can be used as an uncertainty indicator.

---

## 27.9.2 Individual Tree Predictions

Consider a Random Forest regression model:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)
```

The trained model contains:

```python
model.estimators_
```

This is a collection of the individual decision trees.

For example:

```python
len(
    model.estimators_
)
```

returns:

```text
100
```

when the forest contains 100 trees.

Each tree can therefore be evaluated independently.

---

## 27.9.3 Obtaining Predictions From Individual Trees

Suppose:

```python
X_test
```

contains the descriptors of the test materials.

A prediction from one tree can be obtained using:

```python
tree_prediction = (
    model.estimators_[0].predict(
        X_test
    )
)
```

The second tree can be evaluated using:

```python
tree_prediction = (
    model.estimators_[1].predict(
        X_test
    )
)
```

The same procedure can be repeated for every tree.

A complete collection of tree predictions can be constructed using:

```python
import numpy as np

tree_predictions = np.array([
    tree.predict(X_test)
    for tree in model.estimators_
])
```

If the forest contains:

```text
100 trees
```

and the test dataset contains:

```text
500 materials
```

then:

```text
tree_predictions.shape
```

will conceptually be:

```text
(100, 500)
```

Each row corresponds to one tree.

Each column corresponds to one material.

---

## 27.9.4 Random Forest Mean Prediction

The mean prediction across the individual trees is:

```python
mean_prediction = (
    tree_predictions.mean(
        axis=0
    )
)
```

This should correspond closely to the prediction returned by:

```python
model.predict(
    X_test
)
```

The structure is therefore:

```text
Tree predictions
       ↓
Average
       ↓
Random Forest prediction
```

For example:

| Material | Tree predictions      |    Mean |
| -------- | --------------------- | ------: |
| M1       | 2.10, 2.15, 2.08, ... | 2.11 eV |
| M2       | 1.80, 1.72, 1.90, ... | 1.81 eV |
| M3       | 3.20, 2.70, 3.50, ... | 3.14 eV |

The mean provides the standard Random Forest prediction.

---

## 27.9.5 Tree-to-Tree Standard Deviation

The simplest uncertainty estimate is the standard deviation of the individual tree predictions:

```python
tree_std = (
    tree_predictions.std(
        axis=0
    )
)
```

For each material:

```text
Mean prediction
+
Tree prediction standard deviation
```

can then be reported.

For example:

| Material | Prediction | Tree standard deviation |
| -------- | ---------: | ----------------------: |
| M1       |    2.11 eV |                 0.04 eV |
| M2       |    1.81 eV |                 0.12 eV |
| M3       |    3.14 eV |                 0.47 eV |

Material M3 produces substantially larger disagreement among the trees.

This makes M3 a higher-uncertainty prediction according to this ensemble-spread measure.

---

## 27.9.6 A Simple Random Forest Uncertainty Function

The calculation can be packaged into a reusable function:

```python
def rf_predict_with_uncertainty(
    model,
    X
):

    tree_predictions = np.array([
        tree.predict(X)
        for tree in model.estimators_
    ])

    mean_prediction = (
        tree_predictions.mean(
            axis=0
        )
    )

    uncertainty = (
        tree_predictions.std(
            axis=0
        )
    )

    return (
        mean_prediction,
        uncertainty
    )
```

The function can then be used as:

```python
mean_prediction, uncertainty = (
    rf_predict_with_uncertainty(
        model,
        X_test
    )
)
```

The outputs are:

```text
mean_prediction
uncertainty
```

with one value for each material.

---

## 27.9.7 Interpreting Tree Disagreement

Tree disagreement should be interpreted as a measure of **ensemble variability**.

Suppose:

```text
Material A

Tree predictions:
2.01
2.03
2.02
2.04
2.02
```

The trees agree strongly.

Now consider:

```text
Material B

Tree predictions:
1.10
1.80
2.40
2.90
1.50
```

The trees disagree substantially.

Therefore:

```text
Small tree variation
        ↓
Stable ensemble prediction

Large tree variation
        ↓
Unstable ensemble prediction
```

This is useful because the forest provides an internal view of prediction stability without requiring a completely separate model family.

---

## 27.9.8 Why Tree Disagreement Can Increase

Random Forest trees are deliberately different.

Their differences arise from mechanisms such as:

```text
Bootstrap sampling
+
Random feature selection
+
Different tree structures
```

Therefore, each tree learns a slightly different approximation to the relationship between the materials descriptors and the target property.

When the available training information is strong and consistent, the trees often produce similar predictions.

When the model encounters a poorly represented region of feature space, predictions may become more variable.

Conceptually:

```text
Well-represented region
        ↓
Similar tree predictions
        ↓
Low ensemble spread
```

while:

```text
Poorly represented region
        ↓
Different tree predictions
        ↓
High ensemble spread
```

This interpretation is useful, but it should not be treated as a universal rule.

---

## 27.9.9 Random Forest Uncertainty for Materials Properties

Suppose a Random Forest is trained to predict:

```text
Formation Energy
```

from materials descriptors.

For a new material:

```text
Prediction = -2.84 eV/atom
Tree spread = 0.05 eV/atom
```

Another material might produce:

```text
Prediction = -1.72 eV/atom
Tree spread = 0.42 eV/atom
```

The second prediction is substantially less stable across the ensemble.

The uncertainty estimate therefore provides an additional piece of information beyond the predicted formation energy.

The screening result can be represented as:

```text
Material
   ↓
Predicted Property
   +
Prediction Uncertainty
```

rather than:

```text
Material
   ↓
Predicted Property only
```

---

## 27.9.10 Random Forest Uncertainty for Band-Gap Prediction

Consider a band-gap dataset:

```text
Crystal descriptors
        ↓
Random Forest
        ↓
Band-gap prediction
```

Suppose the individual trees produce:

```text
1.92
1.88
1.95
1.90
1.91
1.94
1.89
```

The average is approximately:

```text
1.91 eV
```

and the standard deviation is relatively small.

The model is therefore internally consistent for this prediction.

For another crystal:

```text
0.80
1.40
2.10
1.10
2.30
1.70
0.90
```

the mean may still produce a reasonable-looking number, but the large spread indicates that the forest is much less consistent.

This distinction is valuable in materials screening.

---

## 27.9.11 Uncertainty-Aware Prediction Table

A materials prediction dataset can be augmented with an uncertainty column.

For example:

| Material | Predicted band gap | RF uncertainty |
| -------- | -----------------: | -------------: |
| M1       |            1.91 eV |        0.06 eV |
| M2       |            2.43 eV |        0.11 eV |
| M3       |            0.84 eV |        0.09 eV |
| M4       |            3.12 eV |        0.51 eV |
| M5       |            1.72 eV |        0.07 eV |

This makes it possible to distinguish:

```text
High-confidence-looking prediction
```

from:

```text
High-disagreement prediction
```

according to the Random Forest ensemble.

However, the uncertainty column must be clearly defined as the chosen tree-spread measure.

---

## 27.9.12 Complete Implementation Example

A complete workflow can be written as:

```python
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor
)
from sklearn.model_selection import (
    train_test_split
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# Split data
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# Predictions from individual trees
tree_predictions = np.array([
    tree.predict(X_test)
    for tree in model.estimators_
])

# Mean prediction
mean_prediction = (
    tree_predictions.mean(
        axis=0
    )
)

# Ensemble spread
uncertainty = (
    tree_predictions.std(
        axis=0
    )
)

# Prediction error
mae = mean_absolute_error(
    y_test,
    mean_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        mean_prediction
    )
)

print(
    "MAE:",
    mae
)

print(
    "RMSE:",
    rmse
)
```

The important point is that prediction accuracy and uncertainty are evaluated separately.

```text
MAE / RMSE
    ↓
Prediction accuracy

Tree standard deviation
    ↓
Ensemble variability
```

Both should be reported.

---

## 27.9.13 Comparing Prediction Error With Uncertainty

A useful research analysis is to compare uncertainty estimates with actual prediction errors.

For each test material:

```python
absolute_error = np.abs(
    y_test
    - mean_prediction
)
```

We now have two quantities:

```text
Absolute prediction error
```

and:

```text
Random Forest uncertainty
```

These can be compared.

Ideally, materials with larger uncertainty should tend to have larger prediction errors.

For example:

| Material | RF uncertainty | Absolute error |
| -------- | -------------: | -------------: |
| M1       |           0.03 |           0.04 |
| M2       |           0.07 |           0.08 |
| M3       |           0.12 |           0.14 |
| M4       |           0.31 |           0.39 |
| M5       |           0.48 |           0.61 |

This would suggest that the uncertainty estimate contains useful information.

However, the relationship does not need to be perfect.

A useful uncertainty estimate should be assessed statistically across many test samples rather than judged from a few examples.

---

## 27.9.14 Uncertainty–Error Relationship

The relationship can be visualized conceptually as:

```text
Prediction Error
      ↑
      │                *
      │            *
      │         *
      │      *
      │   *
      │ *
      └──────────────────→
          RF Uncertainty
```

A positive relationship is desirable.

It means that larger ensemble uncertainty tends to correspond to larger prediction errors.

However, this relationship should be measured quantitatively.

Possible analyses include:

```text
Correlation
Rank correlation
Error by uncertainty quantile
Calibration analysis
Prediction interval coverage
```

The important principle is:

```text
Uncertainty should be evaluated
against observed prediction behavior.
```

---

## 27.9.15 Uncertainty Quantiles

One useful analysis is to divide predictions into uncertainty groups.

For example:

```text
Lowest 20% uncertainty
20–40%
40–60%
60–80%
Highest 20%
```

The mean absolute error can then be calculated within each group.

Suppose the results are:

| Uncertainty group |  MAE |
| ----------------- | ---: |
| Lowest 20%        | 0.05 |
| 20–40%            | 0.07 |
| 40–60%            | 0.10 |
| 60–80%            | 0.16 |
| Highest 20%       | 0.31 |

This indicates that uncertainty is informative because prediction error increases as uncertainty increases.

Such an analysis is often more scientifically meaningful than simply reporting the average uncertainty.

---

## 27.9.16 Important Limitation: Tree Spread Is Not Automatically a Probability Distribution

A common mistake is to interpret the individual tree predictions as if they automatically form a fully calibrated predictive probability distribution.

For example:

```text
Mean ± 2 × tree_std
```

does not automatically represent a valid 95% prediction interval.

The reason is that the individual trees are not independent draws from the true predictive distribution.

They are correlated models trained using related data.

Therefore:

```text
Tree standard deviation
```

should initially be interpreted as:

```text
Ensemble prediction spread
```

rather than automatically as:

```text
95% statistical confidence interval
```

Calibration and coverage testing are required before making stronger probabilistic claims.

---

## 27.9.17 Random Forest Uncertainty and Bootstrap Sampling

The connection to bootstrap ensembles is direct.

Random Forest already uses bootstrap sampling in its standard construction.

Conceptually:

```text
Training Dataset
       ↓
Bootstrap Sample 1 → Tree 1
Bootstrap Sample 2 → Tree 2
Bootstrap Sample 3 → Tree 3
...
Bootstrap Sample M → Tree M
```

Therefore, tree-to-tree prediction variation naturally contains information related to sensitivity to the sampled training data.

Random feature selection introduces an additional source of tree diversity.

Thus:

```text
Bootstrap variation
+
Feature-selection variation
+
Tree-structure variation
        ↓
Random Forest ensemble variation
```

This is why Random Forests can be useful for practical uncertainty estimation.

---

## 27.9.18 Random Forest Uncertainty vs Bootstrap Ensemble Uncertainty

The two approaches are related but should not be considered identical.

### Bootstrap ensemble

One can explicitly train separate models:

```text
Bootstrap sample
      ↓
Independent model
```

and calculate variation among those models.

### Random Forest

A Random Forest contains many decision trees:

```text
Bootstrap samples
+
Random feature selection
+
Tree construction
```

and the variation among trees can be analyzed.

Therefore:

```text
Bootstrap ensemble
```

is a general uncertainty framework, while:

```text
Random Forest tree spread
```

is an uncertainty estimate obtained from the internal ensemble structure of a specific model family.

---

## 27.9.19 Random Forest Uncertainty for Small Materials Datasets

Random Forests are often useful for materials datasets containing hundreds or thousands of observations.

For example:

```text
500 crystals
```

may contain descriptors such as:

```text
Atomic number statistics
Element fractions
Density
Lattice parameters
Structural descriptors
Electronic descriptors
```

A Random Forest can learn relationships between these features and a target property.

The ensemble structure then provides a practical way to estimate prediction variability.

However, if the dataset is extremely small, the uncertainty estimate itself may be unstable.

Therefore, the number of trees and the stability of the uncertainty estimates should be investigated.

---

## 27.9.20 Number of Trees and Uncertainty Stability

Suppose a Random Forest is trained with:

```text
10 trees
```

The uncertainty estimate may fluctuate because only a small number of trees contribute to the prediction distribution.

Increasing the number of trees:

```text
10
↓
50
↓
100
↓
300
↓
500
```

generally produces a more stable estimate of the ensemble distribution.

A practical experiment is to monitor the uncertainty estimate as the number of trees increases.

For example:

```text
Number of trees     Mean uncertainty
10                  0.21
50                  0.19
100                 0.18
300                 0.18
500                 0.18
```

Once the estimate becomes stable, additional trees may provide diminishing benefits.

The exact behavior depends on the dataset and model.

---

## 27.9.21 Random Forest Uncertainty and Feature Space

Suppose a materials dataset contains two dominant descriptors:

```text
Atomic radius
Electronegativity
```

The training data may occupy only a particular region of this feature space.

A new material far from that region may produce greater disagreement among the trees.

Conceptually:

```text
Training materials

      • • •
    • • • • •
      • • •
```

while a new material appears at:

```text
                         X
```

The model is being asked to predict in a region with less direct training information.

Tree disagreement may therefore increase.

This makes Random Forest uncertainty potentially useful as an indicator of unfamiliarity.

However, this interpretation should be verified rather than assumed.

---

## 27.9.22 Example: Materials Screening

Suppose 1,000 candidate materials are screened for a target property.

The Random Forest produces:

```text
Predicted property
+
Uncertainty
```

for every candidate.

The resulting dataset might contain:

| Candidate | Prediction | Uncertainty |
| --------- | ---------: | ----------: |
| C1        |       4.21 |        0.05 |
| C2        |       4.18 |        0.07 |
| C3        |       4.25 |        0.09 |
| C4        |       4.31 |        0.42 |
| C5        |       4.27 |        0.06 |

Candidate C4 has a strong predicted value but substantially larger uncertainty.

Therefore, the researcher should distinguish between:

```text
High predicted performance
```

and:

```text
High predicted performance with stable model behavior
```

This is the practical value of adding uncertainty to materials screening.

The uncertainty estimate does not decide whether the material is correct.

It tells the researcher how much caution should accompany the model prediction.

---

## 27.9.23 Reporting Random Forest Uncertainty

A research report should clearly state:

```text
Model:
Random Forest Regressor

Number of trees:
300

Prediction:
Mean of individual tree predictions

Uncertainty:
Standard deviation of individual tree predictions
```

For example:

```text
The predicted band gap was obtained as the
mean prediction across 300 Random Forest trees.
The reported uncertainty represents the standard
deviation of the individual tree predictions.
```

This is much clearer than simply writing:

```text
Band gap = 2.31 ± 0.14 eV
```

without explaining the origin of the uncertainty.

---

## 27.9.24 What Random Forest Uncertainty Can and Cannot Tell Us

Random Forest ensemble spread can provide information about:

```text
Prediction stability
Tree disagreement
Sensitivity to ensemble construction
Possible unfamiliarity
Relative uncertainty between candidates
```

It does not automatically provide:

```text
Guaranteed probability of correctness
Exact experimental uncertainty
Exact DFT uncertainty
Guaranteed 95% prediction interval
Complete epistemic uncertainty
```

Therefore, Random Forest uncertainty should be treated as one component of a broader uncertainty-analysis framework.

---

## 27.9.25 Research-Grade Random Forest Workflow

A robust workflow is:

```text
Materials Dataset
        ↓
Data Cleaning
        ↓
Train / Validation / Test Split
        ↓
Train Random Forest
        ↓
Collect Individual Tree Predictions
        ↓
Calculate Mean Prediction
        ↓
Calculate Tree-to-Tree Spread
        ↓
Compare Uncertainty With Test Error
        ↓
Evaluate Calibration
        ↓
Report Prediction + Uncertainty
```

This workflow keeps uncertainty estimation connected to measurable model performance.

The key principle is:

```text
Prediction
+
Uncertainty Estimate
+
Independent Validation
```

rather than treating tree variance as automatically trustworthy.

---

## 27.9.26 Key Takeaways

A Random Forest contains many decision trees, making it possible to examine the distribution of individual tree predictions.

The basic procedure is:

```text
Random Forest
      ↓
Individual Tree Predictions
      ↓
Mean
      +
Standard Deviation
      ↓
Prediction + Ensemble Spread
```

The most important concepts are:

```text
Individual tree predictions
Tree-to-tree variation
Ensemble mean
Ensemble standard deviation
Bootstrap sampling
Out-of-bag observations
Uncertainty–error relationship
Uncertainty quantiles
Calibration
```

The most important caution is:

```text
Tree spread
≠
Automatically calibrated probability
```

A useful Random Forest uncertainty estimate should therefore be evaluated against independent test-set errors and, when used as an interval or probability statement, subjected to calibration analysis.

Random Forest uncertainty is especially useful as a practical, low-cost uncertainty estimate for descriptor-based Materials Informatics models.

The next section will move to **27.10 Deep Ensembles**, where the same ensemble principle is applied to neural networks and uncertainty is estimated from independently trained deep-learning models.

## 27.10 Deep Ensembles

Deep ensembles extend the ensemble-based uncertainty principle to neural networks.

A single neural network produces one prediction:

```text
Input Crystal
      ↓
Neural Network
      ↓
Prediction
```

A deep ensemble instead trains several neural networks independently:

```text
                  Crystal
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     Network 1   Network 2   Network 3   ... Network M
        ↓           ↓           ↓
       ŷ₁          ŷ₂          ŷ₃          ... ŷₘ
        └───────────┼───────────┘
                    ↓
             Mean Prediction
                    +
             Prediction Spread
```

The individual networks may learn slightly different functions even when they use the same architecture and training dataset.

This variation provides information about the stability of the model prediction.

For Materials Informatics, deep ensembles are particularly useful for neural-network models that predict properties from:

```text
Composition
Crystal descriptors
Crystal graphs
Atomistic representations
Learned material representations
```

The central idea is:

```text
Multiple independently trained neural networks
                    ↓
            Multiple predictions
                    ↓
             Prediction distribution
                    ↓
          Mean + uncertainty estimate
```

---

## 27.10.1 Why Use Deep Ensembles?

Deep neural networks contain many trainable parameters.

During training, the optimization process can converge to different parameter configurations depending on factors such as:

```text
Random initialization
Mini-batch ordering
Data shuffling
Optimization trajectory
Regularization
```

Consequently, two networks with the same architecture can produce slightly different predictions.

For example:

```text
Network 1 → 2.31 eV
Network 2 → 2.35 eV
Network 3 → 2.28 eV
Network 4 → 2.33 eV
```

The predictions are similar.

For another material:

```text
Network 1 → 1.20 eV
Network 2 → 2.00 eV
Network 3 → 2.75 eV
Network 4 → 1.55 eV
```

The networks disagree strongly.

This disagreement can be used as an uncertainty estimate.

---

## 27.10.2 Deep Ensemble Principle

Suppose a neural network model is represented by:

```text
f(x; θ)
```

where:

```text
x = input features
θ = learned model parameters
```

Instead of training one parameter set:

```text
θ
```

we train several:

```text
θ₁
θ₂
θ₃
...
θₘ
```

The corresponding predictions are:

```text
ŷ₁ = f(x; θ₁)

ŷ₂ = f(x; θ₂)

...

ŷₘ = f(x; θₘ)
```

The ensemble prediction can be calculated as:

```text
ŷ = 1/M Σ ŷₘ
```

and the prediction spread can be estimated from:

```text
σ² = 1/M Σ (ŷₘ - ŷ)²
```

Thus, the ensemble provides both:

```text
Mean prediction
```

and:

```text
Prediction variability
```

---

## 27.10.3 Independent Training

The networks in a deep ensemble should be trained independently.

For example:

```text
Network 1
Random seed = 1

Network 2
Random seed = 2

Network 3
Random seed = 3

Network 4
Random seed = 4
```

The architecture can remain identical.

For example:

```text
Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Linear Layer
  ↓
ReLU
  ↓
Output
```

The networks differ primarily through their independent training processes.

This allows the ensemble to capture differences between independently learned solutions.

---

## 27.10.4 Deep Ensembles vs Bootstrap Ensembles

Bootstrap ensembles and deep ensembles use the same broad principle:

```text
Multiple Models
       ↓
Multiple Predictions
       ↓
Prediction Distribution
```

However, the source of model diversity differs.

In a bootstrap ensemble:

```text
Different training samples
        ↓
Different models
```

In a deep ensemble:

```text
Independent neural-network training
        ↓
Different learned parameter configurations
        ↓
Different models
```

A deep ensemble can therefore be constructed using the same training dataset for every network.

For example:

```text
Same dataset
    │
    ├── Network 1
    ├── Network 2
    ├── Network 3
    └── Network 4
```

The independent optimization trajectories generate model diversity.

---

## 27.10.5 Deep Ensembles for Materials Property Prediction

Consider a dataset containing:

```text
Crystal Structure
       ↓
Representation
       ↓
Neural Network
       ↓
Band Gap
```

A deep ensemble changes this to:

```text
Crystal Structure
       ↓
Representation
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
NN 1  NN 2  NN 3  NN M
 ↓     ↓     ↓     ↓
ŷ₁    ŷ₂    ŷ₃    ŷₘ
 └─────┼─────┴─────┘
       ↓
Mean + Spread
```

This can be applied to many materials properties, including:

```text
Band gap
Formation energy
Elastic properties
Thermal properties
Magnetic properties
Electronic properties
```

provided that the corresponding neural-network representation and dataset are appropriate.

---

## 27.10.6 A Simple PyTorch Regression Model

A basic neural network can be implemented using PyTorch:

```python
import torch
import torch.nn as nn


class MaterialsMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)
```

The model receives a materials representation and returns a property prediction.

For example:

```text
Descriptors
    ↓
MaterialsMLP
    ↓
Predicted property
```

---

## 27.10.7 Training One Network

A simple training function can be written as:

```python
def train_model(
    model,
    X_train,
    y_train,
    epochs=200,
    learning_rate=1e-3
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_function = (
        nn.MSELoss()
    )

    model.train()

    for epoch in range(
        epochs
    ):

        optimizer.zero_grad()

        prediction = model(
            X_train
        ).squeeze(-1)

        loss = loss_function(
            prediction,
            y_train
        )

        loss.backward()

        optimizer.step()

    return model
```

This function trains one neural network.

The deep ensemble simply repeats the process with independently initialized networks.

---

## 27.10.8 Training a Deep Ensemble

A basic ensemble can be created using:

```python
def train_deep_ensemble(
    X_train,
    y_train,
    input_dim,
    n_models=5
):

    models = []

    for seed in range(
        n_models
    ):

        torch.manual_seed(
            seed
        )

        model = MaterialsMLP(
            input_dim=input_dim
        )

        model = train_model(
            model,
            X_train,
            y_train
        )

        models.append(
            model
        )

    return models
```

The important step is:

```python
torch.manual_seed(seed)
```

which produces a different initialization for each network.

Thus:

```text
Seed 0 → Network 1
Seed 1 → Network 2
Seed 2 → Network 3
...
```

---

## 27.10.9 Generating Ensemble Predictions

After training, every network can predict the test set.

```python
def ensemble_predict(
    models,
    X
):

    predictions = []

    for model in models:

        model.eval()

        with torch.no_grad():

            prediction = (
                model(X)
                .squeeze(-1)
            )

        predictions.append(
            prediction
        )

    return torch.stack(
        predictions
    )
```

If there are:

```text
5 networks
```

and:

```text
100 test materials
```

the resulting tensor has the conceptual shape:

```text
5 × 100
```

Each row represents one network.

Each column represents one material.

---

## 27.10.10 Ensemble Mean

The ensemble mean is calculated as:

```python
predictions = (
    ensemble_predict(
        models,
        X_test
    )
)

mean_prediction = (
    predictions.mean(
        dim=0
    )
)
```

The result contains one prediction for every material.

For example:

```text
Material 1 → 2.31 eV
Material 2 → 1.84 eV
Material 3 → 3.12 eV
```

The ensemble mean is the central prediction.

---

## 27.10.11 Ensemble Standard Deviation

The uncertainty estimate can be calculated as:

```python
uncertainty = (
    predictions.std(
        dim=0
    )
)
```

For example:

| Material | Ensemble mean | Ensemble std |
| -------- | ------------: | -----------: |
| M1       |       2.31 eV |      0.04 eV |
| M2       |       1.84 eV |      0.11 eV |
| M3       |       3.12 eV |      0.47 eV |

Material M3 produces much greater disagreement among the networks.

Therefore, its prediction is less stable according to the deep-ensemble uncertainty measure.

---

## 27.10.12 Complete Prediction Function

The prediction process can be combined:

```python
def predict_with_uncertainty(
    models,
    X
):

    predictions = (
        ensemble_predict(
            models,
            X
        )
    )

    mean_prediction = (
        predictions.mean(
            dim=0
        )
    )

    uncertainty = (
        predictions.std(
            dim=0
        )
    )

    return (
        mean_prediction,
        uncertainty
    )
```

The output is:

```text
Mean prediction
+
Ensemble uncertainty
```

for every input material.

---

## 27.10.13 Example Interpretation

Suppose five neural networks predict the band gap of a crystal:

```text
2.28
2.34
2.31
2.30
2.35
```

The predictions are tightly clustered.

The ensemble therefore indicates relatively low model disagreement.

Now consider:

```text
1.20
1.75
2.60
2.10
3.00
```

The networks disagree strongly.

The second prediction therefore has a larger ensemble uncertainty.

The interpretation is:

```text
Small disagreement
      ↓
Stable model prediction

Large disagreement
      ↓
Unstable model prediction
```

As with Random Forests, this is an uncertainty indicator rather than an automatic guarantee of correctness.

---

## 27.10.14 Why Different Neural Networks Disagree

Even when the same dataset and architecture are used, neural networks can converge to different parameter configurations.

This happens because neural-network optimization is non-convex.

The loss landscape can contain many different regions corresponding to different parameter configurations.

Conceptually:

```text
Loss
 ↑
 │       \       /
 │        \_____/ 
 │   ___
 │__/   \____
 └────────────────→
       Parameters
```

Different initializations can lead optimization toward different solutions.

Therefore:

```text
Different initialization
        ↓
Different optimization trajectory
        ↓
Different learned parameters
        ↓
Different predictions
```

This diversity is useful for deep ensembles.

---

## 27.10.15 Deep Ensemble Uncertainty and Epistemic Uncertainty

Deep ensemble disagreement is commonly used as an approximate measure of **epistemic uncertainty**.

Epistemic uncertainty represents uncertainty associated with limited model knowledge.

For example, if the training dataset contains many similar crystals:

```text
Training data
████████████████
```

the networks may learn similar relationships.

If a new material lies in a poorly represented region:

```text
Training data              New material
████████████               X
```

the networks may disagree more strongly.

Thus:

```text
Limited information
       ↓
Different learned functions
       ↓
Prediction disagreement
       ↓
Approximate epistemic uncertainty
```

This interpretation is useful but should not be treated as an exact decomposition of all epistemic uncertainty.

---

## 27.10.16 Deep Ensemble Uncertainty Is Not Aleatoric Uncertainty by Itself

A standard deep ensemble predicts a collection of model outputs.

The spread among these outputs mainly represents model disagreement.

It does not automatically represent measurement noise.

Suppose experimental measurements have intrinsic noise:

```text
True property
      ↓
Measurement process
      ↓
Observed values
```

A standard deep ensemble does not automatically know the magnitude of that measurement uncertainty.

Therefore:

```text
Deep ensemble spread
```

should not automatically be interpreted as:

```text
Total predictive uncertainty
```

when the target contains substantial aleatoric uncertainty.

If both types of uncertainty are important, the neural network must be designed to model them explicitly.

---

## 27.10.17 Predicting Mean and Variance

A neural network can be designed to predict both:

```text
Mean
```

and:

```text
Variance
```

for the target distribution.

Instead of:

```text
Input
 ↓
Network
 ↓
ŷ
```

the network produces:

```text
Input
 ↓
Network
 ↓
┌───────────────┐
│ Mean          │
│ Variance      │
└───────────────┘
```

For example:

```python
class ProbabilisticMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=128
    ):

        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU()
        )

        self.mean_head = (
            nn.Linear(
                hidden_dim,
                1
            )
        )

        self.log_variance_head = (
            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        x
    ):

        h = self.hidden(x)

        mean = (
            self.mean_head(h)
        )

        log_variance = (
            self.log_variance_head(h)
        )

        return (
            mean,
            log_variance
        )
```

The variance is often represented through its logarithm to avoid directly predicting negative variance values.

---

## 27.10.18 Why Predict Log Variance?

Variance must satisfy:

```text
variance > 0
```

A neural network output, however, can be any real number.

Therefore, directly predicting:

```text
variance
```

can produce invalid negative values.

Instead, the model predicts:

```text
log_variance
```

and the variance can be obtained through:

```python
variance = torch.exp(
    log_variance
)
```

This guarantees:

```text
variance > 0
```

The transformation is therefore useful for probabilistic regression.

---

## 27.10.19 Combining Deep Ensemble and Aleatoric Uncertainty

A more complete uncertainty-aware neural network can combine:

```text
Deep ensemble
```

with:

```text
Predicted observation variance
```

Each network produces:

```text
Mean
+
Aleatoric variance
```

while differences between networks provide model-level variation.

Conceptually:

```text
                Deep Ensemble
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    Network 1     Network 2     Network M
       ↓             ↓             ↓
   Mean + Var    Mean + Var    Mean + Var
       └─────────────┼─────────────┘
                     ↓
          Ensemble statistics
                     ↓
       ┌─────────────┴─────────────┐
       ↓                           ↓
Aleatoric component       Model disagreement
```

This provides a richer representation of uncertainty than simple ensemble standard deviation.

However, the two components must be defined and evaluated carefully.

---

## 27.10.20 Total Predictive Variance

Suppose each network predicts:

```text
μ₁, μ₂, ..., μₘ
```

and corresponding observation variances:

```text
σ₁², σ₂², ..., σₘ²
```

The total predictive variance can be decomposed conceptually into:

```text
Total variance
=
Average predicted variance
+
Variance of model means
```

That is:

```text
Total uncertainty
≈
Aleatoric uncertainty
+
Epistemic uncertainty
```

This decomposition is useful because it separates:

```text
Noise in the target
```

from:

```text
Disagreement between models
```

The exact interpretation depends on the probabilistic model and assumptions.

---

## 27.10.21 Deep Ensembles for Crystal GNNs

Deep ensembles become especially useful for crystal graph neural networks.

Suppose a crystal is represented as:

```text
Crystal
   ↓
Periodic Crystal Graph
   ↓
GNN
   ↓
Property
```

A deep ensemble produces:

```text
Crystal
   ↓
Crystal Graph
   ↓
 ┌──────┼──────┬──────┐
 ↓      ↓      ↓      ↓
GNN 1  GNN 2  GNN 3  GNN M
 ↓      ↓      ↓      ↓
ŷ₁     ŷ₂     ŷ₃     ŷₘ
 └──────┼──────┴──────┘
        ↓
Mean + Uncertainty
```

This can be applied to crystal properties such as:

```text
Formation energy
Band gap
Elastic modulus
Bulk modulus
Magnetic properties
```

The key requirement is that every network receives the same appropriately constructed crystal representation while being trained independently.

---

## 27.10.22 Deep Ensemble Workflow for Crystal GNNs

A practical workflow is:

```text
Crystal Dataset
       ↓
Pymatgen
       ↓
Periodic Crystal Graphs
       ↓
Train / Validation / Test Split
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
GNN 1 GNN 2 GNN 3 ... GNN M
 ↓     ↓     ↓
Predictions
       ↓
Mean
       +
Ensemble Spread
       ↓
Uncertainty-Aware Prediction
```

The crystal graph construction should remain fixed across the ensemble.

Otherwise, variation caused by inconsistent preprocessing may be confused with model uncertainty.

---

## 27.10.23 Deep Ensemble Uncertainty and Crystal Diversity

Consider two crystals.

Crystal A belongs to a structural family that is well represented in the training dataset:

```text
Training:
Many related structures
```

Crystal B belongs to a poorly represented region:

```text
Training:
Few related structures
```

The ensemble may produce:

```text
Crystal A:

2.10
2.11
2.09
2.12
2.10
```

and:

```text
Crystal B:

1.20
1.80
2.40
2.00
2.90
```

The second crystal produces greater disagreement.

This can be a useful signal that the neural-network ensemble is less stable for that structure.

However, the relationship between structural novelty and uncertainty must be experimentally evaluated.

---

## 27.10.24 Deep Ensembles and Training Data Coverage

The uncertainty estimate becomes more informative when analyzed against training-data coverage.

Suppose materials are embedded in a learned representation space.

A well-covered region might look like:

```text
      • • •
    • • • • •
   • • • • • •
    • • • •
```

A new material inside the region may receive relatively consistent predictions.

A material far away may receive greater disagreement:

```text
      • • •
    • • • • •


                         X
```

This suggests a useful research analysis:

```text
Representation-space location
             ↓
Deep ensemble uncertainty
```

The purpose is not to assume that distance automatically determines uncertainty, but to investigate whether uncertainty correlates with the regions of materials space that are weakly represented.

---

## 27.10.25 Deep Ensemble Size

The number of networks affects both computational cost and uncertainty stability.

For example:

```text
5 networks
10 networks
20 networks
50 networks
```

may produce progressively more stable estimates.

However, every additional network requires another complete training run.

Thus:

```text
More models
     ↓
Higher computational cost
     ↓
Potentially more stable ensemble statistics
```

A practical experiment can evaluate uncertainty stability as the ensemble size increases.

For example:

| Ensemble size | Mean uncertainty |
| ------------: | ---------------: |
|             5 |             0.21 |
|            10 |             0.19 |
|            20 |             0.18 |
|            50 |             0.18 |

If the estimate changes very little after 20 models, training substantially larger ensembles may provide limited additional benefit.

The appropriate ensemble size depends on the dataset, architecture, and computational resources.

---

## 27.10.26 Computational Cost

The main disadvantage of deep ensembles is computational cost.

A single model requires:

```text
One training run
```

A deep ensemble with `M` networks requires approximately:

```text
M training runs
```

Therefore:

```text
Training cost
≈
M × single-model training cost
```

For a small descriptor-based neural network, this may be inexpensive.

For a large crystal GNN trained on thousands or millions of structures, it can become substantial.

The uncertainty benefit must therefore be considered together with computational requirements.

---

## 27.10.27 Deep Ensembles vs MC Dropout

Both deep ensembles and MC Dropout generate multiple predictions.

However, their mechanisms are different.

### Deep Ensemble

```text
Network 1 → Prediction 1
Network 2 → Prediction 2
Network 3 → Prediction 3
```

The networks are independently trained.

### MC Dropout

```text
One trained network
       ↓
Different dropout masks
       ↓
Multiple stochastic predictions
```

Therefore:

```text
Deep ensemble:
Multiple trained models

MC Dropout:
One trained model + stochastic inference
```

Deep ensembles generally require more training computation but can provide strong uncertainty estimates.

MC Dropout can reduce training cost because only one network is trained.

---

## 27.10.28 Deep Ensembles and Prediction Error

As with Random Forests, ensemble uncertainty should be compared with actual test-set errors.

For each test material:

```python
absolute_error = torch.abs(
    y_test
    - mean_prediction
)
```

and:

```python
ensemble_uncertainty = (
    predictions.std(
        dim=0
    )
)
```

can be compared.

A useful uncertainty estimator should ideally show a relationship such as:

```text
Higher ensemble uncertainty
        ↓
Larger average prediction error
```

The relationship should be evaluated across the entire test set.

---

## 27.10.29 Uncertainty Ranking

One practical application is ranking materials by uncertainty.

For example:

```python
ranking = torch.argsort(
    uncertainty,
    descending=True
)
```

This identifies materials with the largest ensemble disagreement.

A researcher can then inspect those predictions more carefully.

For example:

```text
Highest uncertainty

C17 → 0.61
C83 → 0.55
C41 → 0.49
C09 → 0.44
...
```

This does not mean that these materials are necessarily wrong.

It means that the trained neural-network ensemble disagrees more strongly about them.

---

## 27.10.30 High-Confidence and High-Uncertainty Predictions

A materials prediction table can therefore contain:

| Material | Prediction | Ensemble uncertainty |
| -------- | ---------: | -------------------: |
| C1       |    2.42 eV |              0.04 eV |
| C2       |    2.31 eV |              0.06 eV |
| C3       |    2.51 eV |              0.09 eV |
| C4       |    2.60 eV |              0.42 eV |

The researcher can distinguish:

```text
C1
Stable prediction

C4
Large model disagreement
```

This makes the prediction output more informative than a single property value.

---

## 27.10.31 A Critical Failure Mode: Shared Model Bias

Deep ensemble disagreement can be small even when all models are wrong.

Suppose:

```text
Network 1 → 2.10
Network 2 → 2.11
Network 3 → 2.09
Network 4 → 2.10
```

but the true property is:

```text
3.00
```

The networks agree but are systematically biased.

This can happen when:

```text
Training data
+
Representation
+
Model architecture
```

share the same limitation.

Therefore:

```text
Low ensemble disagreement
```

does not guarantee:

```text
Low prediction error
```

This is one of the most important limitations of deep-ensemble uncertainty.

---

## 27.10.32 Distribution Shift

Deep ensembles may also fail to provide sufficiently large uncertainty for some out-of-distribution inputs.

For example, suppose the training dataset contains:

```text
Oxides
Sulfides
Nitrides
```

but a new material belongs to:

```text
Halides
```

The networks may all extrapolate similarly.

They may therefore produce:

```text
Similar predictions
```

even though the material lies outside the dominant training distribution.

This demonstrates why uncertainty estimation must be evaluated together with distribution-shift analysis.

The important principle is:

```text
Model disagreement
+
Data-distribution analysis
```

provides a stronger basis for interpreting uncertainty than ensemble spread alone.

---

## 27.10.33 Calibration of Deep Ensemble Uncertainty

The standard deviation of ensemble predictions is not automatically calibrated.

Suppose the model reports:

```text
Prediction = 2.50 eV
Uncertainty = 0.10 eV
```

This does not automatically mean that:

```text
95% probability
```

lies inside:

```text
2.30–2.70 eV
```

The interval must be evaluated using independent validation data.

For example, one can examine whether prediction intervals achieve their intended empirical coverage.

This creates the workflow:

```text
Deep Ensemble
      ↓
Prediction + Uncertainty
      ↓
Calibration
      ↓
Coverage Evaluation
```

Calibration will therefore determine whether the uncertainty estimate is quantitatively reliable.

---

## 27.10.34 Deep Ensemble Uncertainty for Scientific Interpretation

The uncertainty estimate can be used to describe model behavior scientifically.

For example:

```text
Material A:
Predicted band gap = 2.41 eV
Ensemble spread = 0.05 eV

Material B:
Predicted band gap = 2.44 eV
Ensemble spread = 0.39 eV
```

Although the predicted values are similar, the model confidence differs substantially.

This distinction is important.

A researcher should not treat:

```text
2.41 eV
```

and:

```text
2.44 eV
```

as equally reliable simply because the predictions are numerically close.

The uncertainty provides additional information about the stability of the model prediction.

---

## 27.10.35 Research-Grade Deep Ensemble Workflow

A complete workflow is:

```text
Materials Dataset
        ↓
Materials Representation
        ↓
Train / Validation / Test Split
        ↓
Independent Neural Networks
        ↓
Independent Training
        ↓
Test Predictions
        ↓
Ensemble Mean
        +
Ensemble Spread
        ↓
Compare Uncertainty With Error
        ↓
Calibration
        ↓
Uncertainty-Aware Materials Prediction
```

For crystal GNNs:

```text
Crystal Structures
        ↓
Pymatgen
        ↓
Crystal Graphs
        ↓
Independent GNNs
        ↓
Ensemble Predictions
        ↓
Mean + Uncertainty
```

---

## 27.10.36 Key Takeaways

Deep ensembles estimate uncertainty by training multiple neural networks independently and examining their prediction disagreement.

The fundamental workflow is:

```text
Independent Networks
        ↓
Multiple Predictions
        ↓
Prediction Distribution
        ↓
Mean Prediction
        +
Ensemble Spread
```

The main concepts are:

```text
Independent initialization
Independent training
Model disagreement
Ensemble mean
Ensemble standard deviation
Epistemic uncertainty
Aleatoric uncertainty
Probabilistic prediction
Crystal GNN ensembles
Uncertainty calibration
```

The most important limitations are:

```text
Deep ensemble spread
≠
Guaranteed prediction interval
```

and:

```text
Low ensemble disagreement
≠
Guaranteed correctness
```

Deep ensembles are computationally expensive because multiple neural networks must be trained, but they provide a practical and powerful approach to uncertainty estimation for neural-network-based Materials Informatics models.

The next section will examine **27.11 Monte Carlo Dropout**, where multiple stochastic predictions are generated from a single trained neural network.

## 27.11 Monte Carlo Dropout

Monte Carlo Dropout is a practical method for estimating uncertainty in neural-network predictions.

It uses an important property of dropout:

```text
Dropout
```

is normally used during training as a form of regularization.

During training, some neurons are randomly deactivated.

In ordinary inference, dropout is usually disabled.

Monte Carlo Dropout changes this procedure.

Instead of disabling dropout during prediction, the model keeps dropout active and performs multiple stochastic forward passes.

Conceptually:

```text
                    Input Material
                         ↓
                  Trained Neural Network
                         ↓
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     Dropout Mask 1  Dropout Mask 2  Dropout Mask 3 ... Mask M
          ↓              ↓              ↓
         ŷ₁             ŷ₂             ŷ₃             ŷₘ
          └──────────────┼──────────────┘
                         ↓
                  Mean Prediction
                         +
                  Prediction Spread
```

The different dropout masks produce different predictions.

The variation among those predictions can then be used as an uncertainty estimate.

For Materials Informatics, this provides a relatively inexpensive alternative to training a complete deep ensemble.

---

## 27.11.1 What Is Dropout?

Dropout is a neural-network regularization technique.

During training, a fraction of neurons is randomly removed from each selected layer during every forward pass.

For example, consider:

```text
Input
  ↓
● ● ● ● ●
  ↓
● ● ● ● ●
  ↓
Output
```

With dropout, some neurons may be temporarily deactivated:

```text
Input
  ↓
● ○ ● ● ○
  ↓
● ● ○ ● ○
  ↓
Output
```

where:

```text
● = active neuron
○ = dropped neuron
```

The exact dropout pattern changes from one training step to another.

This forces the network to avoid relying excessively on individual neurons.

---

## 27.11.2 Dropout Probability

A dropout layer is commonly defined using a probability `p`.

For example:

```python
import torch.nn as nn

dropout = nn.Dropout(
    p=0.2
)
```

Here:

```text
p = 0.2
```

means that approximately 20% of the units are dropped during a dropout operation.

A typical neural network may therefore contain:

```python
self.network = nn.Sequential(
    nn.Linear(
        input_dim,
        128
    ),

    nn.ReLU(),

    nn.Dropout(
        p=0.2
    ),

    nn.Linear(
        128,
        128
    ),

    nn.ReLU(),

    nn.Dropout(
        p=0.2
    ),

    nn.Linear(
        128,
        1
    )
)
```

The dropout probability is a hyperparameter and should be chosen carefully.

---

## 27.11.3 Dropout During Ordinary Inference

Normally, dropout behaves differently during training and evaluation.

During training:

```python
model.train()
```

activates dropout.

During ordinary evaluation:

```python
model.eval()
```

disables dropout.

Therefore, if we write:

```python
model.eval()

prediction = model(
    X_test
)
```

the same input normally produces the same prediction every time.

For example:

```text
Run 1 → 2.31 eV
Run 2 → 2.31 eV
Run 3 → 2.31 eV
```

There is no stochastic variation from dropout.

---

## 27.11.4 Monte Carlo Dropout Changes Inference

Monte Carlo Dropout deliberately keeps dropout active during prediction.

The model therefore performs:

```text
Forward pass 1 → Dropout mask 1 → Prediction 1

Forward pass 2 → Dropout mask 2 → Prediction 2

Forward pass 3 → Dropout mask 3 → Prediction 3

...

Forward pass M → Dropout mask M → Prediction M
```

For example:

```text
2.28
2.34
2.31
2.25
2.38
```

The variation among these predictions provides an uncertainty estimate.

The central idea is:

```text
One trained network
        ↓
Multiple stochastic predictions
        ↓
Prediction distribution
        ↓
Mean + spread
```

---

## 27.11.5 Why Monte Carlo Dropout Can Represent Uncertainty

A trained neural network contains learned parameters.

Dropout introduces stochasticity into the effective network during each forward pass.

Therefore, different dropout masks correspond to different subnetworks of the original model.

Conceptually:

```text
Full trained network
        ↓
 ┌──────┼──────┬──────┐
 ↓      ↓      ↓      ↓
Mask 1 Mask 2 Mask 3 ... Mask M
 ↓      ↓      ↓
ŷ₁     ŷ₂     ŷ₃
```

The variation among these subnetworks can provide information about model uncertainty.

Monte Carlo Dropout is therefore commonly interpreted as an approximate method for estimating epistemic uncertainty.

---

## 27.11.6 Monte Carlo Sampling

The word **Monte Carlo** refers to repeated random sampling.

Suppose a material is passed through the network `M` times.

The resulting predictions are:

```text
ŷ₁, ŷ₂, ŷ₃, ..., ŷₘ
```

The mean prediction is:

```text
ŷ = 1/M Σ ŷₘ
```

The prediction variance can be estimated as:

```text
σ² = 1/M Σ (ŷₘ - ŷ)²
```

and the standard deviation is:

```text
σ = √σ²
```

Therefore, Monte Carlo Dropout follows the same general uncertainty-estimation structure used by other ensemble methods:

```text
Multiple predictions
        ↓
Mean
        +
Variation
```

The important difference is how the multiple predictions are generated.

---

## 27.11.7 Building a Materials Neural Network With Dropout

Consider a neural network for predicting a materials property from descriptors:

```python
import torch
import torch.nn as nn


class MaterialsDropoutMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        dropout_rate=0.2
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)
```

The important components for Monte Carlo Dropout are:

```text
nn.Dropout
```

layers inside the trained model.

---

## 27.11.8 Training the Network

The network can be trained normally.

For example:

```python
model = MaterialsDropoutMLP(
    input_dim=X_train.shape[1],
    hidden_dim=128,
    dropout_rate=0.2
)
```

An optimizer can then be defined:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

A mean-squared-error loss can be used for regression:

```python
loss_function = nn.MSELoss()
```

A simplified training loop is:

```python
for epoch in range(
    200
):

    model.train()

    optimizer.zero_grad()

    prediction = (
        model(X_train)
        .squeeze(-1)
    )

    loss = loss_function(
        prediction,
        y_train
    )

    loss.backward()

    optimizer.step()
```

During training:

```python
model.train()
```

ensures that dropout remains active.

---

## 27.11.9 Enabling Dropout During Prediction

The central implementation challenge is that Monte Carlo Dropout requires:

```text
Dropout active
```

while avoiding unnecessary changes to other evaluation behavior.

A simple approach is:

```python
model.train()
```

during stochastic prediction.

For a model containing only standard layers and dropout, this is often sufficient.

However, for more complicated networks containing layers such as Batch Normalization, setting the entire model to training mode can alter other layers as well.

Therefore, research implementations should explicitly manage the model's stochastic layers when necessary.

---

## 27.11.10 Basic Monte Carlo Prediction

A simple implementation is:

```python
def mc_dropout_predict(
    model,
    X,
    n_samples=100
):

    model.train()

    predictions = []

    with torch.no_grad():

        for _ in range(
            n_samples
        ):

            prediction = (
                model(X)
                .squeeze(-1)
            )

            predictions.append(
                prediction
            )

    predictions = torch.stack(
        predictions
    )

    mean_prediction = (
        predictions.mean(
            dim=0
        )
    )

    uncertainty = (
        predictions.std(
            dim=0
        )
    )

    return (
        mean_prediction,
        uncertainty
    )
```

The important steps are:

```text
Enable stochastic dropout
        ↓
Perform many forward passes
        ↓
Collect predictions
        ↓
Calculate mean
        ↓
Calculate standard deviation
```

---

## 27.11.11 Number of Monte Carlo Samples

The number of stochastic forward passes affects the stability of the uncertainty estimate.

For example:

```text
10 samples
50 samples
100 samples
200 samples
500 samples
```

A small number of samples may produce a noisy estimate.

A larger number provides a more stable approximation but increases computational cost.

For example:

| MC samples | Estimated uncertainty |
| ---------: | --------------------: |
|         10 |                  0.18 |
|         50 |                  0.20 |
|        100 |                  0.19 |
|        200 |                  0.19 |
|        500 |                  0.19 |

If the estimate stabilizes around 100–200 samples, using substantially more samples may provide little additional benefit.

The appropriate value depends on the model and dataset.

---

## 27.11.12 Monte Carlo Dropout for Band-Gap Prediction

Suppose a neural network predicts the band gap of a crystal.

One crystal is evaluated 100 times.

The predictions may be:

```text
2.30
2.34
2.28
2.31
2.35
...
```

The mean might be:

```text
2.31 eV
```

and the standard deviation:

```text
0.04 eV
```

Another crystal may produce:

```text
1.10
1.50
2.20
1.80
2.70
...
```

with:

```text
Mean = 1.86 eV

Standard deviation = 0.46 eV
```

The second crystal therefore produces substantially larger stochastic variation.

The model is less stable for that input.

---

## 27.11.13 Monte Carlo Dropout for Crystal GNNs

Monte Carlo Dropout is not limited to multilayer perceptrons.

It can also be applied to crystal graph neural networks.

The workflow becomes:

```text
Crystal Structure
       ↓
Pymatgen
       ↓
Crystal Graph
       ↓
GNN With Dropout
       ↓
Multiple Stochastic Forward Passes
       ↓
Multiple Property Predictions
       ↓
Mean + Uncertainty
```

For example:

```text
Crystal Graph
      ↓
Message Passing
      ↓
Dropout
      ↓
Graph Representation
      ↓
Property Prediction
```

During Monte Carlo inference, different dropout masks create different effective neural networks.

The resulting prediction variation can then be used as an uncertainty indicator.

---

## 27.11.14 Where Dropout Can Be Applied in a GNN

A crystal GNN may contain:

```text
Node embedding
       ↓
Message passing
       ↓
Graph convolution
       ↓
Pooling
       ↓
Prediction head
```

Dropout can potentially be introduced at several locations.

For example:

```text
Node Representation
       ↓
Graph Convolution
       ↓
Dropout
       ↓
Graph Convolution
       ↓
Dropout
       ↓
Pooling
       ↓
Prediction
```

The exact placement depends on the architecture.

The important requirement is that the stochastic mechanism should be applied consistently during Monte Carlo inference.

---

## 27.11.15 Monte Carlo Dropout and Epistemic Uncertainty

Monte Carlo Dropout is primarily useful for estimating model-related uncertainty.

For example:

```text
Limited training information
        ↓
Different stochastic subnetworks
        ↓
Different predictions
        ↓
Larger prediction spread
```

This is commonly interpreted as approximate epistemic uncertainty.

However, the estimate is model-dependent.

It should therefore be validated empirically against prediction errors and other uncertainty measures.

---

## 27.11.16 Monte Carlo Dropout Does Not Automatically Capture Aleatoric Uncertainty

Suppose experimental measurements contain substantial noise.

The target might be represented as:

```text
Observed property
=
Underlying property
+
Measurement noise
```

Standard MC Dropout does not explicitly model this observation noise.

Therefore:

```text
MC Dropout spread
```

should not automatically be interpreted as:

```text
Total predictive uncertainty
```

If aleatoric uncertainty is important, the model can be extended to predict a conditional variance.

The distinction remains:

```text
MC Dropout
→ primarily model uncertainty

Probabilistic output
→ can model observation uncertainty
```

These components can also be combined.

---

## 27.11.17 Combining MC Dropout With Probabilistic Outputs

A neural network can produce both:

```text
Mean
```

and:

```text
Variance
```

while MC Dropout produces multiple stochastic versions of those predictions.

Conceptually:

```text
                    Input
                      ↓
                Neural Network
                      ↓
             ┌────────┴────────┐
             ↓                 ↓
           Mean             Variance
             ↓                 ↓
       MC Dropout Samples
             ↓
      Multiple Predictions
             ↓
      Mean + Model Spread
             +
      Predicted Observation Variance
```

This allows a richer uncertainty representation.

The model can distinguish between:

```text
Variation caused by model uncertainty
```

and:

```text
Variation expected in the target itself
```

when the probabilistic assumptions are appropriate.

---

## 27.11.18 MC Dropout vs Deep Ensembles

Both methods generate multiple predictions, but their computational structures differ.

| Property                     | Deep Ensemble      | MC Dropout                    |
| ---------------------------- | ------------------ | ----------------------------- |
| Number of trained networks   | Multiple           | One                           |
| Stochastic inference         | Not required       | Required                      |
| Training cost                | High               | Lower                         |
| Inference passes             | Multiple           | Multiple                      |
| Model diversity              | Independent models | Dropout masks                 |
| Typical interpretation       | Model disagreement | Approximate model uncertainty |
| Suitable for neural networks | Yes                | Yes                           |
| Suitable for GNNs            | Yes                | Yes                           |

The central distinction is:

```text
Deep Ensemble
=
Many independently trained models
```

whereas:

```text
MC Dropout
=
One trained model
+
Many stochastic forward passes
```

---

## 27.11.19 Computational Advantage

One major advantage of MC Dropout is that only one model needs to be trained.

For a deep ensemble containing 10 models:

```text
Training:
10 complete neural-network training runs
```

For MC Dropout:

```text
Training:
1 neural-network training run
```

Inference still requires multiple forward passes:

```text
1 model
×
100 stochastic passes
```

Therefore, MC Dropout can be attractive when training large materials models is expensive.

However, inference becomes more computationally expensive than ordinary single-pass prediction.

---

## 27.11.20 MC Dropout and Prediction Stability

The stochastic predictions can be inspected directly.

For a stable prediction:

```text
2.30
2.31
2.29
2.30
2.32
2.31
```

The distribution is narrow.

For an unstable prediction:

```text
1.20
1.70
2.30
1.40
2.60
1.90
```

the distribution is much wider.

The uncertainty therefore provides a measure of prediction stability under stochastic model perturbation.

---

## 27.11.21 Visualizing Monte Carlo Predictions

For one material, the predictions can be visualized as a histogram.

Conceptually:

```text
Frequency
  ↑
  │       ███
  │     ███████
  │   ███████████
  │     ███████
  │       ███
  └────────────────→
       Prediction
```

A narrow histogram indicates:

```text
Small stochastic variation
```

while a broad histogram indicates:

```text
Large stochastic variation
```

For Materials Informatics, the distributions can be examined for:

```text
Band-gap predictions
Formation-energy predictions
Elastic-property predictions
Other predicted materials properties
```

---

## 27.11.22 Comparing MC Dropout Uncertainty With Error

As with Random Forests and deep ensembles, MC Dropout uncertainty should be compared with actual prediction error.

For each test material:

```python
absolute_error = torch.abs(
    y_test
    - mean_prediction
)
```

and:

```python
mc_uncertainty = uncertainty
```

can be compared.

A useful uncertainty estimate should ideally satisfy:

```text
Higher uncertainty
        ↓
Higher average prediction error
```

across groups of predictions.

For example:

| MC uncertainty group |  MAE |
| -------------------- | ---: |
| Lowest 20%           | 0.05 |
| 20–40%               | 0.07 |
| 40–60%               | 0.11 |
| 60–80%               | 0.17 |
| Highest 20%          | 0.29 |

Such a trend would suggest that the uncertainty estimate is informative.

---

## 27.11.23 A Critical Limitation: Dropout Uncertainty Is Approximate

Monte Carlo Dropout is not an exact calculation of the true posterior distribution over neural-network parameters.

It is an approximation based on stochastic dropout masks.

Therefore:

```text
MC Dropout uncertainty
```

should be described as an approximate uncertainty estimate.

It should not automatically be interpreted as an exact Bayesian posterior uncertainty.

This distinction is especially important in research publications.

A precise statement would be:

```text
Monte Carlo Dropout was used to obtain an
approximate model-uncertainty estimate from
multiple stochastic forward passes.
```

rather than:

```text
The exact Bayesian uncertainty was calculated.
```

---

## 27.11.24 Dropout Rate and Uncertainty

The dropout probability affects the stochastic behavior of the model.

For example:

```text
p = 0.05
p = 0.10
p = 0.20
p = 0.30
p = 0.50
```

Different values may produce different uncertainty estimates.

A very small dropout rate may produce little variation.

A very large dropout rate may produce excessively noisy predictions.

Therefore, dropout probability should be treated as a model hyperparameter.

It should be selected using appropriate validation procedures rather than chosen solely because it produces a desirable uncertainty estimate.

---

## 27.11.25 Number of MC Samples and Computational Cost

Suppose one forward pass takes:

```text
1 ms
```

Then:

```text
100 MC samples
≈
100 ms
```

per batch, ignoring other overheads.

For a large crystal dataset, the additional inference cost can become significant.

Therefore:

```text
More MC samples
        ↓
More stable uncertainty
        +
Higher inference cost
```

A practical implementation should test uncertainty convergence as the number of samples increases.

---

## 27.11.26 MC Dropout for Materials Screening

Suppose a trained neural network screens 10,000 candidate materials.

Ordinary inference gives:

```text
Candidate
    ↓
Predicted property
```

MC Dropout gives:

```text
Candidate
    ↓
100 stochastic predictions
    ↓
Mean prediction
+
Uncertainty
```

The resulting screening table might be:

| Candidate | Predicted property | MC uncertainty |
| --------- | -----------------: | -------------: |
| C1        |               4.21 |           0.04 |
| C2        |               4.18 |           0.06 |
| C3        |               4.24 |           0.08 |
| C4        |               4.32 |           0.41 |

Candidate C4 has a high predicted value but also substantially higher model uncertainty.

This allows the researcher to distinguish the prediction itself from the stability of the model output.

---

## 27.11.27 MC Dropout and Distribution Shift

MC Dropout uncertainty can sometimes increase when the model encounters unfamiliar inputs.

For example:

```text
Training materials
       ↓
Common chemical region
       ↓
Low stochastic variation
```

while:

```text
New chemical region
       ↓
Different internal representations
       ↓
Potentially larger stochastic variation
```

This can make MC Dropout useful for identifying potentially difficult predictions.

However, as with deep ensembles, uncertainty may remain low even when all predictions are systematically wrong.

Therefore:

```text
MC Dropout
+
Independent test evaluation
+
Distribution analysis
```

provides a stronger assessment than MC Dropout alone.

---

## 27.11.28 MC Dropout for Crystal Graph Models

For a crystal GNN, a practical implementation may follow:

```text
Crystal Structure
       ↓
Pymatgen
       ↓
Periodic Graph
       ↓
GNN
       ↓
Dropout
       ↓
Property Prediction
```

During inference:

```text
Graph
 ↓
GNN + Dropout Mask 1 → ŷ₁
GNN + Dropout Mask 2 → ŷ₂
GNN + Dropout Mask 3 → ŷ₃
...
GNN + Dropout Mask M → ŷₘ
```

Then:

```text
Mean(ŷ₁,...,ŷₘ)
```

provides the central prediction.

The standard deviation provides the basic stochastic uncertainty estimate.

This makes MC Dropout particularly attractive when a full deep ensemble of crystal GNNs would be computationally expensive.

---

## 27.11.29 Research-Grade MC Dropout Workflow

A complete workflow is:

```text
Crystal Dataset
       ↓
Crystal Representation
       ↓
GNN / Neural Network With Dropout
       ↓
Model Training
       ↓
Activate Stochastic Dropout
       ↓
Multiple Forward Passes
       ↓
Prediction Distribution
       ↓
Mean Prediction
       +
Uncertainty
       ↓
Compare With Test Error
       ↓
Calibration
       ↓
Uncertainty-Aware Prediction
```

The central principle is:

```text
One trained model
+
Multiple stochastic predictions
=
Approximate uncertainty information
```

---

## 27.11.30 Deep Ensembles vs MC Dropout: Practical Choice

For a Materials Informatics project, the choice may depend on computational resources.

If computational resources allow several complete training runs:

```text
Deep Ensemble
```

provides independently trained models.

If training is expensive but stochastic inference is affordable:

```text
MC Dropout
```

can provide an approximate uncertainty estimate using one trained model.

A useful research strategy is to compare both methods on the same test dataset:

```text
Same Dataset
     ↓
 ┌───┴────────────┐
 ↓                ↓
Deep Ensemble   MC Dropout
 ↓                ↓
Uncertainty     Uncertainty
 └───────┬────────┘
         ↓
Compare Calibration
and Error Relationship
```

This allows the uncertainty methods themselves to be evaluated rather than assuming one is automatically superior.

---

## 27.11.31 Key Takeaways

Monte Carlo Dropout estimates uncertainty by keeping dropout active during inference and performing multiple stochastic forward passes.

The basic workflow is:

```text
Trained Neural Network
        ↓
Dropout Active
        ↓
Multiple Forward Passes
        ↓
Multiple Predictions
        ↓
Mean Prediction
        +
Prediction Spread
```

The major concepts are:

```text
Dropout
Monte Carlo sampling
Stochastic forward passes
Prediction mean
Prediction variance
Epistemic uncertainty
Probabilistic outputs
Crystal GNN uncertainty
Uncertainty–error relationship
Calibration
```

The most important limitations are:

```text
MC Dropout uncertainty
=
Approximate model uncertainty
```

not:

```text
Exact Bayesian uncertainty
```

and:

```text
Low MC Dropout uncertainty
≠
Guaranteed correct prediction
```

Monte Carlo Dropout is therefore a practical uncertainty-estimation method for neural-network-based Materials Informatics, particularly when training many independent networks is computationally expensive.

The next section will examine **27.12 Bayesian Neural Networks**, focusing on probabilistic model parameters, posterior distributions, and predictive uncertainty.

## 27.12 Bayesian Neural Networks

A standard neural network learns a single set of parameters that produces predictions for new materials.

Conceptually:

```text
Input Material
      ↓
Neural Network
      ↓
Learned Parameters
      ↓
Prediction
```

For example, a neural network predicting band gap may learn parameters:

```text
θ₁, θ₂, θ₃, ..., θₙ
```

and then calculate:

```text
ŷ = f(x; θ)
```

The parameters are treated as fixed after training.

A **Bayesian Neural Network (BNN)** takes a different approach.

Instead of treating the neural-network parameters as a single fixed set of values, a Bayesian neural network represents them as probability distributions.

Conceptually:

```text
Standard Neural Network

Training
   ↓
One parameter set θ
   ↓
Prediction
```

whereas:

```text
Bayesian Neural Network

Training
   ↓
Distribution over parameters
   ↓
Multiple plausible parameter configurations
   ↓
Predictive distribution
```

This provides a direct probabilistic framework for representing uncertainty associated with the learned model.

For Materials Informatics, this is particularly relevant when the model must report not only:

```text
Predicted property
```

but also:

```text
How uncertain the prediction is
```

---

## 27.12.1 Deterministic Neural Networks

Before discussing Bayesian neural networks, it is useful to understand the standard deterministic formulation.

Suppose:

```text
x
```

represents the material input.

This could contain:

```text
Composition
Crystal descriptors
Structural descriptors
Learned crystal representation
```

A neural network applies a function:

```text
f(x; θ)
```

where:

```text
θ = all trainable neural-network parameters
```

Training attempts to find a parameter set that minimizes a loss function.

For example:

```python
loss = loss_function(
    prediction,
    target
)
```

After training, the model has one parameter configuration.

Therefore:

```text
θ = θ*
```

where `θ*` represents the learned parameters.

The prediction is then:

```text
ŷ = f(x; θ*)
```

This formulation produces a point prediction.

---

## 27.12.2 Bayesian Interpretation

A Bayesian neural network does not assume that the parameters are exactly known.

Instead, the parameters are treated as random variables.

The model therefore considers:

```text
p(θ)
```

rather than a single fixed value of:

```text
θ
```

The probability distribution:

```text
p(θ)
```

is called the **prior distribution**.

After observing training data, the model updates this distribution.

The resulting distribution is:

```text
p(θ | D)
```

where:

```text
D = training dataset
```

This is the **posterior distribution** over neural-network parameters.

The central Bayesian workflow is therefore:

```text
Prior
p(θ)
  ↓
Training Data
D
  ↓
Posterior
p(θ | D)
```

The posterior represents parameter configurations that remain plausible after considering the observed data.

---

## 27.12.3 Bayes' Theorem

The Bayesian update is based on Bayes' theorem:

```text
p(θ | D)
=
[p(D | θ) p(θ)]
/
p(D)
```

where:

```text
p(θ | D)
```

is the posterior distribution,

```text
p(D | θ)
```

is the likelihood,

```text
p(θ)
```

is the prior,

and:

```text
p(D)
```

is the evidence.

The interpretation is:

```text
Posterior
∝
Likelihood × Prior
```

This relationship is fundamental to Bayesian neural networks.

---

## 27.12.4 Prior Distribution

The prior describes what parameter values are considered plausible before observing the training data.

For example, a simple Gaussian prior may be written as:

```text
θ ~ N(0, σ²)
```

This expresses the assumption that parameter values are centered around zero with some specified spread.

In practice, a neural network can contain millions of parameters.

Therefore, the prior applies to a high-dimensional parameter space.

Conceptually:

```text
Prior
      ↓
Possible neural-network parameters
      ↓
Training data
```

The prior can influence the posterior, particularly when the dataset is small.

---

## 27.12.5 Likelihood

The likelihood describes how probable the observed training data is under a particular parameter configuration.

For a regression problem, a common assumption is:

```text
yᵢ ~ N(f(xᵢ; θ), σ²)
```

This means that the observed target is modeled as a distribution centered around the neural-network prediction.

The likelihood for the complete dataset can then be written conceptually as:

```text
p(D | θ)
```

A parameter configuration that produces predictions consistent with the observed data receives a higher likelihood.

A parameter configuration that produces poor predictions receives a lower likelihood.

---

## 27.12.6 Posterior Distribution

The posterior combines the prior information with evidence from the training dataset.

```text
Prior
  +
Observed Data
  ↓
Posterior
```

Mathematically:

```text
p(θ | D)
```

The posterior is therefore not a single parameter vector.

It is a probability distribution over possible parameter values.

For a single parameter, it can be visualized conceptually as:

```text
Probability
    ↑
    │          ███
    │        ███████
    │      ███████████
    │    ███████████████
    └────────────────────→ θ
```

For a neural network, the actual posterior exists in a very high-dimensional parameter space.

This makes exact Bayesian inference extremely difficult.

---

## 27.12.7 Predictive Distribution

The main scientific advantage of the Bayesian formulation is that uncertainty in the parameters can be propagated into uncertainty in predictions.

For a new material `x*`, the predictive distribution is:

```text
p(y* | x*, D)
=
∫ p(y* | x*, θ)
   p(θ | D)
   dθ
```

The equation contains two important components:

```text
p(y* | x*, θ)
```

describes prediction uncertainty for a particular parameter configuration.

```text
p(θ | D)
```

describes uncertainty about the model parameters.

The integration averages over plausible parameter configurations.

Conceptually:

```text
Posterior Parameter Distribution
              ↓
     Multiple plausible models
              ↓
     Multiple predictions
              ↓
       Predictive distribution
```

This is the central mechanism by which Bayesian neural networks represent model uncertainty.

---

## 27.12.8 Bayesian Neural Networks for Materials

Consider a neural network trained to predict formation energy.

The input may contain:

```text
Crystal representation
```

and the target is:

```text
Formation energy
```

A deterministic model produces:

```text
ŷ = -2.41 eV
```

A Bayesian model instead produces a predictive distribution:

```text
Formation energy
        ↓
Mean = -2.41 eV
Uncertainty = 0.18 eV
```

The uncertainty arises from the posterior distribution over model parameters and, depending on the likelihood model, from observation noise.

This is useful because the model can distinguish between:

```text
Prediction
```

and:

```text
Uncertainty associated with that prediction
```

---

## 27.12.9 Epistemic Uncertainty in Bayesian Neural Networks

Bayesian neural networks provide a natural framework for representing epistemic uncertainty.

Suppose a region of materials space is well represented in the training data:

```text
Training materials

• • • •
 • • • •
• • • • •
```

The training data constrains the model strongly in that region.

The posterior over parameters becomes more concentrated with respect to predictions in that region.

Now consider a poorly represented region:

```text
Training materials                 New material

• • • •
 • • • •
• • • • •                         X
```

The available data provides weaker constraints on the model behavior there.

The posterior predictive distribution may therefore become broader.

Conceptually:

```text
Limited training information
        ↓
Greater parameter uncertainty
        ↓
Greater predictive uncertainty
```

This is one reason Bayesian neural networks are attractive for uncertainty-aware Materials Informatics.

---

## 27.12.10 Aleatoric Uncertainty in Bayesian Neural Networks

Bayesian neural networks can also model aleatoric uncertainty if the output distribution explicitly includes observation noise.

For example:

```text
y ~ N(μ(x), σ²(x))
```

where:

```text
μ(x)
```

is the predicted mean and:

```text
σ²(x)
```

is the predicted observation variance.

The neural network can therefore output:

```text
Mean
+
Variance
```

for each material.

This allows the model to represent uncertainty that is associated with noisy observations rather than only uncertainty about the neural-network parameters.

Thus, a sufficiently general Bayesian predictive model can contain both:

```text
Epistemic uncertainty
```

and:

```text
Aleatoric uncertainty
```

---

## 27.12.11 Bayesian Neural Network Architecture

The basic architecture can look similar to an ordinary neural network:

```text
Input
  ↓
Bayesian Layer
  ↓
Bayesian Layer
  ↓
Bayesian Layer
  ↓
Output
```

The important difference is that the parameters of the layers are probabilistic.

For example, instead of:

```python
nn.Linear(
    input_dim,
    hidden_dim
)
```

with fixed learned weights, a Bayesian layer conceptually contains:

```text
Weight distribution
+
Bias distribution
```

For a weight:

```text
w
```

we may model:

```text
w ~ q(w)
```

where `q(w)` represents an approximate posterior distribution.

---

## 27.12.12 Distribution Over a Neural-Network Weight

Suppose one network weight has posterior:

```text
w ~ N(0.4, 0.05²)
```

The model does not treat the weight as exactly:

```text
w = 0.4
```

Instead, it considers values around 0.4 according to the posterior distribution.

For example:

```text
0.36
0.39
0.42
0.44
0.38
```

may all be plausible values.

When many weights are uncertain, different sampled parameter configurations produce different predictions.

This provides the Bayesian mechanism for generating predictive uncertainty.

---

## 27.12.13 Sampling From the Posterior

If the posterior distribution were available exactly, one could sample parameter configurations:

```text
θ₁ ~ p(θ | D)
θ₂ ~ p(θ | D)
θ₃ ~ p(θ | D)
...
θₘ ~ p(θ | D)
```

Then each parameter sample produces a prediction:

```text
ŷ₁ = f(x; θ₁)

ŷ₂ = f(x; θ₂)

...

ŷₘ = f(x; θₘ)
```

The resulting prediction distribution can be summarized using:

```text
Mean
Variance
Quantiles
Prediction intervals
```

This resembles the ensemble procedures discussed earlier, but the source of model diversity is different.

Deep ensemble:

```text
Different independently trained models
```

Bayesian neural network:

```text
Different parameter samples from the posterior
```

---

## 27.12.14 Why Exact Bayesian Inference Is Difficult

A neural network may contain millions of parameters.

The posterior:

```text
p(θ | D)
```

therefore exists in an extremely high-dimensional space.

Computing the exact posterior generally requires evaluating a difficult high-dimensional integral.

For example:

```text
p(θ | D)
```

may contain:

```text
Millions of dimensions
```

and the posterior distribution may be highly non-Gaussian and complicated.

Therefore, exact Bayesian inference is generally computationally impractical for modern large neural networks.

This motivates **approximate Bayesian inference**.

---

## 27.12.15 Variational Inference

One common approach is **variational inference**.

Instead of calculating the exact posterior:

```text
p(θ | D)
```

we introduce a simpler approximate distribution:

```text
q(θ)
```

and attempt to make:

```text
q(θ)
```

as close as possible to:

```text
p(θ | D)
```

Conceptually:

```text
Exact Posterior
      ↓
   difficult
      ↓
Approximate Distribution
q(θ)
```

The approximate distribution is chosen so that its parameters can be optimized efficiently.

---

## 27.12.16 Variational Distribution

A simple approximation may assume that each neural-network weight has an independent Gaussian distribution.

For example:

```text
q(w)
=
N(μ, σ²)
```

Instead of learning a single:

```text
w
```

the model learns:

```text
μ
```

and:

```text
σ
```

for the weight distribution.

The same idea is applied to many parameters.

Conceptually:

```text
Weight
  ↓
Mean parameter μ
Variance parameter σ²
```

The model therefore learns a distribution over weights rather than one deterministic value.

---

## 27.12.17 Reparameterization

Sampling from a learned distribution must be performed in a way that allows the distribution parameters to be optimized through gradient-based learning.

A common technique is the reparameterization trick.

For a Gaussian weight:

```text
w ~ N(μ, σ²)
```

we can write:

```text
w = μ + σ ε
```

where:

```text
ε ~ N(0, 1)
```

The randomness is moved into:

```text
ε
```

while:

```text
μ
```

and:

```text
σ
```

remain differentiable model parameters.

Conceptually:

```text
Random sample ε
       ↓
μ + σ ε
       ↓
Sampled weight
```

This allows gradient-based optimization of the variational parameters.

---

## 27.12.18 Variational Objective

The variational approximation can be learned by optimizing an objective related to the evidence lower bound, commonly called the ELBO.

Conceptually:

```text
ELBO
=
Expected log likelihood
-
KL divergence
```

The first term encourages the model to explain the observed data.

The KL-divergence term controls how far the approximate posterior moves from the prior.

Thus:

```text
Data fit
+
Regularization toward prior
```

jointly determine the approximate posterior.

---

## 27.12.19 Interpretation of the KL Term

The KL divergence can be written conceptually as:

```text
KL(
q(θ)
||
p(θ)
)
```

It measures how different the approximate posterior is from the prior.

If the posterior remains close to the prior:

```text
q(θ) ≈ p(θ)
```

the data have provided relatively little information about those parameters.

If the posterior differs strongly from the prior:

```text
q(θ) ≠ p(θ)
```

the observed data have caused a substantial Bayesian update.

This provides a probabilistic interpretation of learning.

---

## 27.12.20 A Simple Bayesian Linear Layer

A conceptual Bayesian linear layer can assign distributions to its weights and biases.

For example:

```python
import torch
import torch.nn as nn


class BayesianLinear(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        output_dim
    ):

        super().__init__()

        self.weight_mu = nn.Parameter(
            torch.zeros(
                output_dim,
                input_dim
            )
        )

        self.weight_log_sigma = (
            nn.Parameter(
                torch.full(
                    (
                        output_dim,
                        input_dim
                    ),
                    -3.0
                )
            )
        )

        self.bias_mu = nn.Parameter(
            torch.zeros(
                output_dim
            )
        )

        self.bias_log_sigma = (
            nn.Parameter(
                torch.full(
                    (
                        output_dim
                    ),
                    -3.0
                )
            )
        )

    def forward(
        self,
        x
    ):

        weight_sigma = torch.exp(
            self.weight_log_sigma
        )

        bias_sigma = torch.exp(
            self.bias_log_sigma
        )

        weight_epsilon = (
            torch.randn_like(
                weight_sigma
            )
        )

        bias_epsilon = (
            torch.randn_like(
                bias_sigma
            )
        )

        weight = (
            self.weight_mu
            +
            weight_sigma
            * weight_epsilon
        )

        bias = (
            self.bias_mu
            +
            bias_sigma
            * bias_epsilon
        )

        return torch.nn.functional.linear(
            x,
            weight,
            bias
        )
```

This is a simplified educational implementation.

A complete Bayesian neural-network implementation also requires an appropriate Bayesian objective and prior/posterior treatment.

---

## 27.12.21 Building a Simple Bayesian MLP

The Bayesian layer can be used to construct a neural network:

```python
class BayesianMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=64
    ):

        super().__init__()

        self.layer1 = BayesianLinear(
            input_dim,
            hidden_dim
        )

        self.layer2 = BayesianLinear(
            hidden_dim,
            hidden_dim
        )

        self.output = BayesianLinear(
            hidden_dim,
            1
        )

    def forward(
        self,
        x
    ):

        x = torch.relu(
            self.layer1(x)
        )

        x = torch.relu(
            self.layer2(x)
        )

        return self.output(x)
```

Every forward pass samples new weights.

Therefore:

```text
Forward pass 1
→ sampled θ₁
→ prediction ŷ₁

Forward pass 2
→ sampled θ₂
→ prediction ŷ₂

Forward pass 3
→ sampled θ₃
→ prediction ŷ₃
```

This produces a predictive distribution.

---

## 27.12.22 Predictive Sampling

After training the approximate Bayesian network, multiple predictions can be generated.

```python
def bayesian_predict(
    model,
    X,
    n_samples=100
):

    predictions = []

    model.eval()

    with torch.no_grad():

        for _ in range(
            n_samples
        ):

            prediction = (
                model(X)
                .squeeze(-1)
            )

            predictions.append(
                prediction
            )

    predictions = torch.stack(
        predictions
    )

    mean_prediction = (
        predictions.mean(
            dim=0
        )
    )

    uncertainty = (
        predictions.std(
            dim=0
        )
    )

    return (
        mean_prediction,
        uncertainty
    )
```

The important difference from ordinary MC Dropout is that the stochasticity now comes from sampling the Bayesian parameters.

---

## 27.12.23 Bayesian Neural Networks vs MC Dropout

Both methods generate multiple predictions, but their underlying interpretations differ.

| Property                  | MC Dropout            | Bayesian Neural Network        |
| ------------------------- | --------------------- | ------------------------------ |
| Main stochastic mechanism | Dropout masks         | Parameter distributions        |
| Parameter representation  | Usually deterministic | Probabilistic                  |
| Posterior approximation   | Implicit/approximate  | Explicit approximate posterior |
| Multiple predictions      | Yes                   | Yes                            |
| Epistemic uncertainty     | Approximate           | Direct Bayesian formulation    |
| Implementation complexity | Lower                 | Higher                         |
| Computational cost        | Moderate              | Higher                         |
| Materials applications    | Neural networks, GNNs | Neural networks, GNNs          |

The central conceptual distinction is:

```text
MC Dropout
=
Stochastic subnetworks
```

whereas:

```text
Bayesian Neural Network
=
Distribution over model parameters
```

---

## 27.12.24 Bayesian Neural Networks vs Deep Ensembles

The two approaches also produce multiple models in different ways.

Deep ensembles:

```text
Dataset
  ↓
Independent training
  ↓
θ₁, θ₂, θ₃, ..., θₘ
```

Bayesian neural networks:

```text
Dataset
  ↓
Posterior approximation
  ↓
p(θ | D)
  ↓
Samples θ₁, θ₂, θ₃, ..., θₘ
```

Therefore:

```text
Deep Ensemble
→ multiple independently trained parameter sets

Bayesian NN
→ samples from a learned parameter distribution
```

Both can provide useful epistemic uncertainty estimates, but they rely on different assumptions and computational procedures.

---

## 27.12.25 Bayesian Neural Networks for Crystal Property Prediction

A Bayesian neural network can be applied to crystal-property prediction.

The workflow is:

```text
Crystal Structure
       ↓
Crystal Representation
       ↓
Bayesian Neural Network
       ↓
Posterior Over Parameters
       ↓
Posterior Predictive Distribution
       ↓
Property Mean + Uncertainty
```

For example:

```text
Crystal
  ↓
Graph Representation
  ↓
Bayesian GNN
  ↓
Formation Energy
  ↓
Mean = -2.35 eV
Uncertainty = 0.21 eV
```

The same framework can be applied to other continuous materials properties.

---

## 27.12.26 Bayesian GNN Concept

For a crystal graph:

```text
G = (V, E)
```

where:

```text
V = atoms
E = periodic interactions
```

a Bayesian GNN can treat some or all neural-network parameters as probability distributions.

The workflow becomes:

```text
Crystal Graph
      ↓
Bayesian Message Passing
      ↓
Bayesian Graph Representation
      ↓
Bayesian Prediction Layer
      ↓
Predictive Distribution
```

The resulting prediction is not simply:

```text
ŷ
```

but:

```text
p(y | G, D)
```

which represents the predictive distribution conditioned on the crystal graph and training data.

---

## 27.12.27 Uncertainty Across Chemical Space

Bayesian models can be particularly informative when the training dataset covers materials space unevenly.

Suppose the dataset contains many:

```text
Oxides
```

but very few:

```text
Halides
```

The posterior uncertainty may differ between those regions.

Conceptually:

```text
Well represented region
       ↓
Strong data constraint
       ↓
Lower epistemic uncertainty
```

and:

```text
Poorly represented region
       ↓
Weak data constraint
       ↓
Potentially higher epistemic uncertainty
```

This provides a useful framework for investigating whether uncertainty reflects limitations in training-data coverage.

---

## 27.12.28 Small Materials Datasets

Bayesian neural networks can be attractive when materials datasets are relatively small.

In ordinary deep learning, a highly flexible neural network may fit a small dataset poorly or become strongly dependent on the training sample.

A Bayesian formulation introduces prior information and a distribution over parameters.

Conceptually:

```text
Small Dataset
     +
Prior
     ↓
Posterior
     ↓
Predictive Distribution
```

However, Bayesian neural networks are not automatically superior for small datasets.

The choice depends on:

```text
Dataset size
Representation
Model architecture
Prior specification
Inference method
Computational resources
```

Therefore, empirical comparison remains necessary.

---

## 27.12.29 Limitations of Bayesian Neural Networks

Bayesian neural networks provide a principled framework, but they also introduce substantial challenges.

### Computational complexity

Large neural networks may contain millions of parameters.

Representing distributions over all parameters can therefore be expensive.

### Approximate inference

Exact posterior inference is usually impractical.

Approximation methods may produce inaccurate posterior distributions.

### Prior sensitivity

The choice of prior can influence the resulting posterior, particularly when training data are limited.

### Optimization difficulty

Variational objectives can be difficult to optimize.

### Interpretation

A mathematically Bayesian uncertainty estimate does not automatically guarantee that the resulting predictive intervals are empirically calibrated.

Therefore:

```text
Bayesian formulation
≠
Automatically perfect uncertainty
```

The uncertainty still needs to be evaluated.

---

## 27.12.30 Bayesian Uncertainty Must Be Validated

Suppose a Bayesian neural network reports:

```text
Prediction = 2.50 eV
Uncertainty = 0.10 eV
```

The reported uncertainty should be tested against independent data.

A useful evaluation asks:

```text
Do predictions with larger uncertainty
actually have larger errors?
```

and:

```text
Do predictive intervals achieve
their expected coverage?
```

Therefore, the workflow should remain:

```text
Bayesian Model
      ↓
Predictive Distribution
      ↓
Uncertainty Estimate
      ↓
Calibration
      ↓
Independent Evaluation
```

The Bayesian derivation provides a principled basis for uncertainty, but empirical validation remains essential for scientific use.

---

## 27.12.31 Bayesian Neural Networks and Total Predictive Uncertainty

A Bayesian neural network can be combined with a probabilistic likelihood.

For example:

```text
y | x, θ ~ N(
    μ(x, θ),
    σ²(x, θ)
)
```

The uncertainty in the final prediction can then contain contributions from:

```text
Parameter uncertainty
+
Observation uncertainty
```

Conceptually:

```text
Posterior Parameter Distribution
             ↓
      Model uncertainty
             +
      Observation model
             ↓
     Predictive distribution
```

This provides a unified framework for uncertainty quantification.

The exact decomposition depends on the probabilistic model and inference method.

---

## 27.12.32 Practical Bayesian Workflow for Materials ML

A research-oriented workflow can be organized as:

```text
Materials Dataset
       ↓
Materials Representation
       ↓
Train / Validation / Test Split
       ↓
Define Prior
       ↓
Bayesian Neural Network
       ↓
Approximate Posterior Inference
       ↓
Posterior Predictive Sampling
       ↓
Mean Prediction
       +
Predictive Uncertainty
       ↓
Calibration
       ↓
Independent Test Evaluation
```

For crystal materials:

```text
Crystal Structure
       ↓
Pymatgen / Crystal Representation
       ↓
Bayesian GNN
       ↓
Posterior Parameter Distribution
       ↓
Posterior Predictive Distribution
       ↓
Crystal Property + Uncertainty
```

---

## 27.12.33 Comparison of Bayesian Approaches

The major uncertainty approaches discussed so far can be summarized as:

| Method             | Source of multiple predictions   | Main uncertainty interpretation |
| ------------------ | -------------------------------- | ------------------------------- |
| Random Forest      | Individual trees                 | Tree disagreement               |
| Bootstrap ensemble | Resampled training datasets      | Model/data variation            |
| Deep ensemble      | Independently trained networks   | Model disagreement              |
| MC Dropout         | Stochastic dropout masks         | Approximate model uncertainty   |
| Bayesian NN        | Samples from parameter posterior | Bayesian parameter uncertainty  |

These methods should not be treated as mathematically identical.

They differ in:

```text
Assumptions
Training procedure
Computational cost
Uncertainty interpretation
Calibration behavior
```

Therefore, a research study should define exactly which uncertainty method is being used and how its output is interpreted.

---

## 27.12.34 Research Example

Consider a dataset of crystal structures and formation energies.

The model workflow is:

```text
Crystal Structures
        ↓
Crystal Representation
        ↓
Bayesian Neural Network
        ↓
Approximate Posterior
        ↓
Posterior Samples
        ↓
Formation-Energy Predictions
        ↓
Mean + Uncertainty
```

Suppose the model predicts:

| Crystal | Predicted formation energy | Uncertainty |
| ------- | -------------------------: | ----------: |
| A       |                   -2.41 eV |     0.05 eV |
| B       |                   -2.36 eV |     0.08 eV |
| C       |                   -2.52 eV |     0.31 eV |
| D       |                   -2.44 eV |     0.07 eV |

Crystal C has a relatively large uncertainty.

The correct scientific interpretation is not:

```text
Crystal C is unstable.
```

Instead:

```text
The model has greater predictive uncertainty
for Crystal C.
```

The underlying physical stability must be evaluated independently.

This distinction prevents uncertainty estimates from being confused with physical properties.

---

## 27.12.35 Key Takeaways

A Bayesian neural network treats neural-network parameters probabilistically rather than as a single fixed parameter configuration.

The fundamental Bayesian workflow is:

```text
Prior
  ↓
Training Data
  ↓
Posterior
  ↓
Posterior Samples
  ↓
Predictive Distribution
  ↓
Prediction + Uncertainty
```

The major concepts are:

```text
Bayes' theorem
Prior
Likelihood
Posterior
Predictive distribution
Epistemic uncertainty
Aleatoric uncertainty
Variational inference
Approximate posterior
Posterior sampling
Bayesian neural networks
Bayesian crystal-property prediction
Bayesian GNNs
```

The most important conceptual distinction is:

```text
Standard neural network
→ one learned parameter set

Bayesian neural network
→ probability distribution over parameters
```

This provides a principled framework for representing model uncertainty.

However, Bayesian neural networks introduce substantial computational and methodological challenges because exact posterior inference is generally impractical for large neural networks.

Therefore, approximate inference methods are commonly required, and the resulting uncertainty estimates must still be calibrated and evaluated on independent data.

The next section will examine **27.13 Gaussian Processes**, focusing on probabilistic regression, predictive mean and variance, covariance functions, and why Gaussian Processes are useful for uncertainty quantification on relatively small materials datasets.

## 27.13 Gaussian Processes

Gaussian Processes (GPs) provide a probabilistic framework for regression in which predictions are represented by distributions rather than only single values.

This makes Gaussian Processes particularly useful for uncertainty quantification.

A conventional regression model may produce:

```text
Predicted band gap = 2.14 eV
```

A Gaussian Process instead produces a predictive distribution:

```text
Predicted band gap
=
2.14 eV
±
predictive uncertainty
```

The central idea is that a Gaussian Process does not directly assume a particular finite-dimensional parameter vector.

Instead, it defines a probability distribution over possible functions.

Conceptually:

```text
Training Data
      ↓
Gaussian Process
      ↓
Distribution over Functions
      ↓
Predictive Distribution
      ↓
Mean + Variance
```

This makes the Gaussian Process naturally suited to uncertainty-aware materials property prediction.

---

## 27.13.1 From Regression Functions to Distributions Over Functions

Consider an ordinary regression model:

```text
y = f(x)
```

where:

```text
x = material features
y = target property
```

A deterministic model learns one function:

```text
f(x)
```

For example:

```text
Input descriptors
       ↓
Regression model
       ↓
One predicted value
```

A Gaussian Process instead considers many possible functions:

```text
f₁(x)
f₂(x)
f₃(x)
...
fₙ(x)
```

and assigns probabilities to them.

Conceptually:

```text
             Possible functions

f₁(x)  ─────────────────────
f₂(x)  ───────────────────
f₃(x)  ───────────────────────
f₄(x)  ─────────────────
              ↓
       Gaussian Process
              ↓
     Probability distribution
```

The observed training data constrains which functions remain plausible.

---

## 27.13.2 What Is a Gaussian Process?

A Gaussian Process is a collection of random variables such that any finite collection of those variables follows a joint Gaussian distribution.

It is commonly written as:

```text
f(x) ~ GP(m(x), k(x,x'))
```

where:

```text
m(x)
```

is the mean function and:

```text
k(x,x')
```

is the covariance function, also called the kernel.

Therefore, a Gaussian Process is defined primarily by:

```text
Mean function
+
Covariance function
```

For many regression problems, the mean function is chosen to be zero:

```text
m(x) = 0
```

and the kernel controls how strongly predictions at different inputs are related.

---

## 27.13.3 Mean Function

The mean function is:

```text
m(x) = E[f(x)]
```

It represents the expected value of the latent function before considering the observed data.

A simple choice is:

```text
m(x) = 0
```

This is often used because the covariance structure and observed data provide most of the useful information.

However, a nonzero mean function can also be used when prior knowledge suggests a meaningful baseline.

For example:

```text
m(x) = baseline_prediction(x)
```

could encode a known approximate relationship.

---

## 27.13.4 Covariance Function

The covariance function is central to a Gaussian Process.

It is written as:

```text
k(x, x')
```

It describes how function values at two input locations are related.

For example:

```text
k(x₁, x₂)
```

measures the covariance between:

```text
f(x₁)
```

and:

```text
f(x₂)
```

If two materials have similar feature representations, the kernel may assign them a high covariance.

Conceptually:

```text
Similar materials
      ↓
High covariance
      ↓
Related predictions
```

while:

```text
Very different materials
      ↓
Lower covariance
      ↓
Less related predictions
```

This relationship is particularly important in Materials Informatics because similarity between materials can be expressed through composition, structural descriptors, or learned representations.

---

## 27.13.5 Kernel Functions

The covariance function is often called a kernel.

Several kernels are commonly used.

Examples include:

```text
RBF kernel
Matérn kernel
Linear kernel
Polynomial kernel
Rational quadratic kernel
```

The choice of kernel determines assumptions about how smoothly the target property changes through feature space.

For materials data, this choice should be guided by the representation and expected behavior of the property.

---

## 27.13.6 Radial Basis Function Kernel

One of the most common kernels is the Radial Basis Function (RBF) kernel.

It can be written as:

```text
k(x,x')
=
σ_f²
exp(
    -||x-x'||²
    /
    2l²
)
```

where:

```text
σ_f²
```

controls the overall function variance and:

```text
l
```

is the length scale.

The Euclidean distance:

```text
||x-x'||
```

measures the distance between two input points.

The kernel therefore decreases as the two inputs become more distant.

Conceptually:

```text
Small feature distance
        ↓
High covariance
```

and:

```text
Large feature distance
        ↓
Low covariance
```

---

## 27.13.7 Length Scale

The length scale controls how quickly the covariance decreases with distance.

A small length scale means that even relatively small changes in the input can produce substantially different function values.

Conceptually:

```text
Small length scale

x₁ ─── x₂
 ↓     ↓
weak relationship
```

A large length scale produces smoother functions:

```text
Large length scale

x₁ ───────── x₂
 ↓           ↓
strong relationship
```

For materials data, the length scale therefore affects how rapidly the model assumes a property can change across materials space.

---

## 27.13.8 Matérn Kernel

The Matérn kernel is another important choice.

It provides more flexible assumptions about function smoothness than the RBF kernel.

A Matérn kernel contains a parameter controlling smoothness.

Different settings correspond to functions with different degrees of differentiability.

This can be useful because materials properties are not always perfectly smooth functions of descriptor space.

For example, changes in:

```text
Composition
Crystal structure
Coordination
Bonding environment
```

can sometimes produce relatively abrupt changes in predicted properties.

Therefore, a Matérn kernel may be preferable when a highly smooth RBF assumption is too restrictive.

---

## 27.13.9 Linear Kernel

A linear kernel assumes a linear relationship in feature space.

Conceptually:

```text
k(x,x')
=
xᵀx'
```

This can be useful when the representation already captures a meaningful approximately linear relationship with the target.

For example:

```text
Descriptors
   ↓
Linear relationship
   ↓
Target property
```

However, many materials-property relationships are nonlinear.

Therefore, nonlinear kernels are often more appropriate when the underlying property relationship is complex.

---

## 27.13.10 Kernel Choice in Materials Informatics

Kernel selection should not be treated as an arbitrary modeling choice.

It encodes assumptions about the structure of the target function.

For example:

| Kernel             | Main assumption                   |
| ------------------ | --------------------------------- |
| Linear             | Approximately linear relationship |
| RBF                | Smooth nonlinear relationship     |
| Matérn             | Flexible smoothness               |
| Polynomial         | Polynomial-like relationship      |
| Rational quadratic | Mixture of length scales          |

The appropriate kernel depends on:

```text
Materials representation
+
Target property
+
Dataset size
+
Expected smoothness
```

A kernel should therefore be evaluated using validation data.

---

## 27.13.11 Training Data Representation

Suppose a materials dataset contains:

```text
X
```

as a feature matrix:

```text
X =
[
x₁
x₂
...
xₙ
]
```

where each row represents one material.

For example:

```text
Composition descriptors
Crystal descriptors
Electronic descriptors
Structural descriptors
```

The corresponding target values are:

```text
y =
[
y₁
y₂
...
yₙ
]
```

The Gaussian Process evaluates the covariance between all training samples.

This produces a covariance matrix:

```text
K
```

where:

```text
Kᵢⱼ = k(xᵢ,xⱼ)
```

This matrix contains the pairwise covariance relationships among the training materials.

---

## 27.13.12 The Covariance Matrix

Suppose there are three training materials.

The covariance matrix is:

```text
K =
| k(x₁,x₁)  k(x₁,x₂)  k(x₁,x₃) |
| k(x₂,x₁)  k(x₂,x₂)  k(x₂,x₃) |
| k(x₃,x₁)  k(x₃,x₂)  k(x₃,x₃) |
```

The diagonal elements:

```text
k(xᵢ,xᵢ)
```

represent the covariance of each point with itself.

The off-diagonal elements represent relationships between different materials.

For example:

```text
k(x₁,x₂)
```

may be large if materials 1 and 2 are similar in descriptor space.

---

## 27.13.13 Adding Observation Noise

Real materials datasets often contain noise.

For example, experimental measurements may differ because of:

```text
Measurement uncertainty
Sample preparation
Instrument limitations
Experimental conditions
```

A Gaussian Process can model observation noise by adding a noise term to the covariance matrix.

The effective covariance matrix becomes:

```text
K + σ_n²I
```

where:

```text
σ_n²
```

is the observation-noise variance and:

```text
I
```

is the identity matrix.

This separates the smooth latent function from measurement noise.

---

## 27.13.14 Gaussian Process Prediction

Suppose a new material has feature vector:

```text
x*
```

The Gaussian Process calculates its covariance with the training data:

```text
k*
```

where:

```text
k*
=
[
k(x₁,x*)
k(x₂,x*)
...
k(xₙ,x*)
]
```

The model then combines:

```text
Training covariance
+
Training targets
+
New-point covariance
```

to obtain a predictive distribution.

The predictive distribution is Gaussian:

```text
y*
~
N(
    μ*,
    σ*²
)
```

where:

```text
μ*
```

is the predictive mean and:

```text
σ*²
```

is the predictive variance.

This is the main reason Gaussian Processes are useful for uncertainty quantification.

---

## 27.13.15 Predictive Mean

For a Gaussian Process with Gaussian observation noise, the predictive mean can be written as:

```text
μ*
=
k*ᵀ
(K + σ_n²I)⁻¹
y
```

This expression shows that the prediction depends on the relationship between the new material and the training materials.

If the new material is strongly related to several training samples, its prediction is strongly informed by the observed data.

---

## 27.13.16 Predictive Variance

The predictive variance of the latent function can be written as:

```text
σ*²
=
k(x*,x*)
-
k*ᵀ
(K + σ_n²I)⁻¹
k*
```

The exact expression depends on whether the uncertainty refers to the latent function or the noisy observed target.

The important conceptual relationship is:

```text
New material
      ↓
Similarity to training data
      ↓
Predictive variance
```

A material located close to well-characterized training data generally has lower uncertainty.

A material located far from the training data can have higher uncertainty.

---

## 27.13.17 Why Gaussian Processes Naturally Provide Uncertainty

Unlike ordinary deterministic regression, a Gaussian Process directly defines a probability distribution over functions.

Therefore, uncertainty is part of the model rather than an additional quantity added after prediction.

Conceptually:

```text
Gaussian Process
       ↓
Predictive distribution
       ↓
 ┌───────────────┐
 │ Mean          │
 │ Variance      │
 │ Confidence   │
 │ intervals     │
 └───────────────┘
```

For example:

```text
Band gap
=
2.10 eV
±
0.08 eV
```

The uncertainty is derived from the covariance structure and the relationship between the new input and the training dataset.

---

## 27.13.18 Uncertainty and Training-Data Density

Consider two new materials.

Material A is surrounded by many similar training materials:

```text
Training space

• • • •
 • A •
• • • •
```

Material B lies far from the observed data:

```text
Training space

• • • •


             B
```

The Gaussian Process generally produces:

```text
Material A
→ lower predictive variance
```

and:

```text
Material B
→ higher predictive variance
```

This behavior is particularly useful in Materials Informatics.

The uncertainty can therefore provide information about whether a prediction is being made in a well-supported region of materials space.

---

## 27.13.19 Gaussian Processes and Extrapolation

Gaussian Processes are generally most reliable when predicting near regions supported by training data.

Suppose the training materials occupy:

```text
Region A
```

and a new material lies within that region:

```text
Training
████████
████ X ██
████████
```

The model has substantial information about that region.

But if the material lies far outside:

```text
Training
████████


                    X
```

the predictive uncertainty may increase.

This makes the uncertainty estimate useful for identifying potential extrapolation.

However, high uncertainty should not be interpreted as proof that the material is physically unusual.

It indicates that the model has less information from the training dataset in that region.

---

## 27.13.20 Gaussian Processes for Small Materials Datasets

One of the most important advantages of Gaussian Processes is their usefulness for relatively small datasets.

Suppose a DFT dataset contains:

```text
200 crystals
```

A large deep neural network may have many more trainable parameters than the number of training examples.

A Gaussian Process can instead operate directly on the available feature representation while providing probabilistic predictions.

Conceptually:

```text
Small DFT Dataset
       ↓
Materials Descriptors
       ↓
Gaussian Process
       ↓
Prediction + Uncertainty
```

This makes GPs attractive for expensive materials calculations where generating thousands or millions of labeled examples is difficult.

---

## 27.13.21 Why Dataset Size Matters

The standard Gaussian Process implementation requires operations involving the covariance matrix:

```text
K
```

where `K` has dimensions:

```text
N × N
```

for `N` training samples.

Therefore, computational and memory costs grow rapidly as the dataset becomes large.

The commonly encountered computational complexity of exact GP training is approximately:

```text
O(N³)
```

with memory requirements approximately:

```text
O(N²)
```

because of covariance-matrix operations.

For:

```text
N = 1,000
```

this may still be manageable depending on the implementation and hardware.

For:

```text
N = 1,000,000
```

an exact Gaussian Process becomes impractical.

Therefore, Gaussian Processes are particularly attractive for small-to-medium datasets or when approximate GP methods are used.

---

## 27.13.22 Gaussian Process Workflow in Materials Informatics

A typical workflow is:

```text
Crystal Structures
       ↓
Materials Representation
       ↓
Feature Matrix
       ↓
Training Dataset
       ↓
Gaussian Process
       ↓
Kernel Evaluation
       ↓
Predictive Mean
       +
Predictive Variance
```

The representation could come from:

```text
Composition descriptors
Structural descriptors
Matminer features
Pymatgen-derived descriptors
Other numerical materials representations
```

The GP itself does not require a specific materials representation.

The representation must simply produce numerical inputs suitable for the chosen kernel.

---

## 27.13.23 Gaussian Process With Materials Descriptors

Suppose a dataset contains descriptors such as:

```text
mean atomic radius
mean electronegativity
density
volume
number of elements
coordination-related descriptors
```

The feature matrix can be written as:

```python
X = [
    [x11, x12, x13, ...],
    [x21, x22, x23, ...],
    ...
]
```

and the target property:

```python
y = [
    y1,
    y2,
    ...
]
```

can be used to train a Gaussian Process.

Feature scaling is often important because kernel distances depend directly on the numerical scale of the inputs.

---

## 27.13.24 Feature Scaling

Suppose one descriptor ranges between:

```text
0–1
```

while another ranges between:

```text
0–100000
```

The second feature can dominate distance calculations.

For example:

```text
||x - x'||
```

may be determined mostly by the large-scale feature.

Therefore, standardization is commonly useful.

For example:

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

The scaler must be fitted only on the training data.

The same transformation is then applied to validation and test data.

---

## 27.13.25 Gaussian Process Regression in scikit-learn

A basic Gaussian Process can be implemented using scikit-learn.

```python
from sklearn.gaussian_process import (
    GaussianProcessRegressor
)

from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel
)
```

A kernel can be defined as:

```python
kernel = (
    ConstantKernel(
        1.0
    )
    *
    RBF(
        length_scale=1.0
    )
)
```

The model is then:

```python
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.1,
    normalize_y=True
)
```

and trained using:

```python
gp.fit(
    X_train_scaled,
    y_train
)
```

---

## 27.13.26 Predicting With Uncertainty

One of the most useful features of `GaussianProcessRegressor` is that it can return both the mean prediction and standard deviation.

```python
mean_prediction, std_prediction = (
    gp.predict(
        X_test_scaled,
        return_std=True
    )
)
```

Therefore:

```text
mean_prediction
```

contains the predicted property while:

```text
std_prediction
```

contains the predictive standard deviation.

For example:

```python
for mean, std in zip(
    mean_prediction,
    std_prediction
):

    print(
        f"{mean:.3f} ± {std:.3f}"
    )
```

A result might look like:

```text
2.143 ± 0.081
2.315 ± 0.094
1.782 ± 0.221
```

The third material has substantially higher predictive uncertainty.

---

## 27.13.27 Materials Property Example

Suppose the target is formation energy.

The prediction output might be:

```text
Material A
Formation energy = -2.31 eV
Uncertainty = 0.04 eV

Material B
Formation energy = -2.28 eV
Uncertainty = 0.07 eV

Material C
Formation energy = -1.92 eV
Uncertainty = 0.31 eV
```

The correct interpretation is:

```text
Material C
→ less certain model prediction
```

not:

```text
Material C
→ physically unstable
```

The uncertainty describes the model's predictive uncertainty.

Physical stability must be determined using appropriate physical calculations or experimental evidence.

---

## 27.13.28 Predictive Intervals

The predictive mean and standard deviation can be used to construct approximate prediction intervals under Gaussian assumptions.

For example, an approximate interval can be expressed as:

```text
μ ± 1.96σ
```

for a nominal 95% Gaussian interval.

For example:

```text
μ = 2.10 eV
σ = 0.10 eV
```

gives approximately:

```text
2.10 ± 0.196 eV
```

or:

```text
[1.904 eV, 2.296 eV]
```

However, this interval is meaningful as a 95% interval only when the underlying assumptions and calibration behavior support that interpretation.

Therefore, predictive intervals must be evaluated empirically.

---

## 27.13.29 Gaussian Process Uncertainty and Calibration

A GP may produce mathematically well-defined predictive variances.

That does not automatically mean the intervals are perfectly calibrated.

Suppose a model reports:

```text
95% prediction intervals
```

for 100 test materials.

Ideally, approximately:

```text
95 out of 100
```

observations should fall within the corresponding intervals, subject to sampling variability and the assumptions of the calibration setting.

If only 70 observations fall inside:

```text
Observed coverage = 70%
```

then the uncertainty intervals are overconfident.

If nearly all observations fall inside very wide intervals, the model may be underconfident or poorly sharp.

Thus:

```text
Predictive variance
        ↓
Prediction interval
        ↓
Coverage evaluation
```

is essential.

---

## 27.13.30 Gaussian Processes and Noisy Measurements

Consider an experimental materials dataset.

The true property may be:

```text
y_true
```

but the measured value is:

```text
y_observed
=
y_true
+
ε
```

where:

```text
ε
```

represents measurement noise.

A Gaussian Process can include a noise parameter:

```python
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=measurement_noise
)
```

or learn an appropriate noise contribution depending on the model configuration.

This allows the GP to distinguish the smooth underlying function from noisy observations.

---

## 27.13.31 Gaussian Processes and DFT Data

DFT-generated datasets have different uncertainty characteristics from experimental measurements.

DFT values are often deterministic for a specified computational setup, but they still depend on approximations such as:

```text
Exchange-correlation functional
Pseudopotentials
Basis-set choices
Convergence criteria
Structural relaxation
Magnetic configuration
```

A Gaussian Process trained on DFT data primarily quantifies uncertainty associated with the learned regression relationship and the observed dataset.

It does not automatically quantify every source of physical or computational uncertainty in the DFT calculation itself.

This distinction is important:

```text
GP predictive uncertainty
≠
Total DFT uncertainty
```

A separate model would be required to explicitly represent uncertainty arising from different DFT methodologies.

---

## 27.13.32 Gaussian Processes and Chemical Similarity

Kernel functions provide a natural way to encode similarity.

Suppose two materials have similar descriptor vectors:

```text
x₁ ≈ x₂
```

Then an RBF kernel may produce:

```text
k(x₁,x₂) ≈ large
```

This means the model considers their latent property values strongly related.

For very different materials:

```text
x₁ far from x₂
```

the covariance may become small.

Therefore:

```text
Materials similarity
        ↓
Kernel similarity
        ↓
Prediction correlation
```

This provides a useful bridge between materials descriptors and probabilistic prediction.

---

## 27.13.33 Gaussian Process Prediction Across Materials Space

Suppose materials are represented in a two-dimensional descriptor space.

A GP can produce a predicted property surface:

```text
Descriptor 2
     ↑
     │   ─────────
     │  /        \
     │ /          \
     │/            \
     └────────────────→ Descriptor 1
```

It can also produce an uncertainty surface.

Conceptually:

```text
Descriptor 2
     ↑
     │   low uncertainty
     │  ███████████
     │  ███
     │       high uncertainty
     │          █████
     └────────────────→ Descriptor 1
```

This allows researchers to identify regions where predictions are well supported and regions where the model is less certain.

---

## 27.13.34 Gaussian Process for a Small DFT Dataset

A practical research workflow might contain:

```text
500 DFT structures
        ↓
Pymatgen descriptors
        ↓
Feature scaling
        ↓
Train / validation / test split
        ↓
Gaussian Process
        ↓
Kernel optimization
        ↓
Prediction
        ↓
Predictive uncertainty
        ↓
Calibration evaluation
```

For a small dataset, this can be computationally attractive while still providing probabilistic predictions.

---

## 27.13.35 Hyperparameters of a Gaussian Process

Important GP hyperparameters may include:

```text
Kernel amplitude
Length scale
Noise level
Kernel-specific parameters
```

For an RBF kernel:

```text
σ_f
```

controls the scale of function variation.

The length scale:

```text
l
```

controls how rapidly the function changes.

The noise:

```text
σ_n
```

controls observation noise.

These parameters strongly influence predictions and uncertainty.

They should therefore be optimized using appropriate training procedures and validated using held-out data.

---

## 27.13.36 Automatic Kernel Parameter Optimization

In practical implementations, kernel parameters can be optimized from the training data.

For example:

```python
kernel = (
    ConstantKernel(
        1.0,
        (
            1e-3,
            1e3
        )
    )
    *
    RBF(
        length_scale=1.0,
        length_scale_bounds=(
            1e-2,
            1e2
        )
    )
)
```

The Gaussian Process implementation can optimize these hyperparameters during fitting.

After training, the learned kernel can be inspected:

```python
print(
    gp.kernel_
)
```

This helps determine what length scale and amplitude the model selected.

---

## 27.13.37 Kernel Optimization Must Be Interpreted Carefully

A fitted kernel parameter is not automatically a physical law.

For example, if the model learns a particular length scale, it does not necessarily mean:

```text
Physical property changes on exactly this
materials-space length scale.
```

The learned parameter describes the behavior of the chosen statistical model under the chosen representation.

Therefore:

```text
Kernel parameter
```

should not automatically be interpreted as:

```text
Physical constant
```

without independent scientific justification.

---

## 27.13.38 Gaussian Processes and High-Dimensional Materials Descriptors

Materials datasets can contain hundreds or thousands of descriptors.

A standard exact GP can become difficult in high-dimensional feature spaces.

Possible issues include:

```text
Large feature dimension
Distance concentration
Irrelevant descriptors
Redundant descriptors
Computational cost
```

Therefore, descriptor preprocessing may be important.

For example:

```text
Materials descriptors
        ↓
Feature selection / dimensionality control
        ↓
Scaled representation
        ↓
Gaussian Process
```

The purpose is to provide a representation in which the kernel can meaningfully measure similarity.

---

## 27.13.39 Gaussian Processes With Learned Representations

A Gaussian Process does not necessarily have to operate directly on hand-crafted descriptors.

A learned representation can first be generated:

```text
Crystal
   ↓
Neural representation
   ↓
Latent vector
   ↓
Gaussian Process
   ↓
Property distribution
```

This creates a hybrid architecture.

For example:

```text
Crystal GNN
     ↓
Graph embedding
     ↓
Gaussian Process
     ↓
Property prediction + uncertainty
```

Such a system combines representation learning with probabilistic regression.

However, the uncertainty behavior of the complete system must be evaluated carefully because uncertainty can arise from both the learned representation and the GP component.

---

## 27.13.40 Gaussian Processes for Crystal Graph Representations

A crystal graph can be converted into a vector representation:

```text
Crystal Graph
      ↓
Graph Encoder
      ↓
Crystal Embedding
      ↓
Gaussian Process
      ↓
Property + Uncertainty
```

The GP then operates on the learned embedding rather than directly on atomic coordinates.

This can be useful when the crystal representation contains complex structural information that would be difficult to encode manually.

The GP still provides its probabilistic prediction at the regression stage.

---

## 27.13.41 Limitations of Gaussian Processes

Gaussian Processes have important limitations.

### Computational scaling

Exact GP methods scale poorly with dataset size.

The covariance matrix requires substantial memory and computational effort.

### Kernel dependence

The model depends strongly on the chosen kernel and representation.

### High-dimensional inputs

Poor feature representations can make kernel similarity less meaningful.

### Large datasets

Very large materials datasets may require sparse or approximate GP methods.

### Extrapolation

A GP may become highly uncertain far from the training distribution, but uncertainty is still model-dependent and should not be treated as a guaranteed detector of every failure mode.

Therefore:

```text
Gaussian Process
```

is not automatically the best uncertainty method for every materials dataset.

---

## 27.13.42 When Gaussian Processes Are Particularly Useful

Gaussian Processes are especially attractive when:

```text
Dataset is relatively small
        +
Labels are expensive
        +
Predictive uncertainty is important
        +
A meaningful similarity representation exists
```

For example:

```text
Expensive DFT calculations
        ↓
Few hundred labeled structures
        ↓
Materials descriptors
        ↓
Gaussian Process
        ↓
Prediction + uncertainty
```

This is a natural use case for GP regression.

---

## 27.13.43 Gaussian Process vs Neural-Network Uncertainty

A Gaussian Process and a neural network can both produce uncertainty-aware predictions, but they approach the problem differently.

| Property                  | Gaussian Process                       | Neural Network                         |
| ------------------------- | -------------------------------------- | -------------------------------------- |
| Basic representation      | Distribution over functions            | Parametric function                    |
| Uncertainty               | Native predictive distribution         | Requires probabilistic/ensemble method |
| Small datasets            | Often strong                           | Can be difficult                       |
| Very large datasets       | Computationally difficult for exact GP | Often more scalable                    |
| Kernel required           | Yes                                    | No                                     |
| Representation learning   | Limited in basic GP                    | Strong                                 |
| Crystal GNN compatibility | Possible through embeddings            | Native                                 |
| Predictive variance       | Direct                                 | Method-dependent                       |

The choice should depend on the scientific problem, dataset size, representation, and computational resources.

---

## 27.13.44 Research-Grade Gaussian Process Workflow

A complete Materials Informatics GP workflow is:

```text
Crystal Dataset
       ↓
Structure / Composition Representation
       ↓
Feature Engineering
       ↓
Feature Scaling
       ↓
Train / Validation / Test Split
       ↓
Kernel Selection
       ↓
GP Training
       ↓
Kernel Parameter Optimization
       ↓
Predictive Mean
       +
Predictive Variance
       ↓
Prediction Intervals
       ↓
Calibration Evaluation
       ↓
Independent Test Evaluation
```

The uncertainty estimate should therefore be treated as part of the complete evaluation pipeline rather than simply an additional output column.

---

## 27.13.45 Practical Interpretation of GP Predictions

Suppose three materials receive:

```text
Material A:
Prediction = 2.30 eV
Uncertainty = 0.05 eV

Material B:
Prediction = 2.35 eV
Uncertainty = 0.08 eV

Material C:
Prediction = 2.41 eV
Uncertainty = 0.45 eV
```

Material C has the highest predicted value.

However, it also has substantially larger uncertainty.

A scientifically careful interpretation is:

```text
Material C has the highest predicted property,
but the model is substantially less certain
about that prediction.
```

This is more informative than ranking candidates using predicted values alone.

---

## 27.13.46 Key Takeaways

Gaussian Processes provide a probabilistic approach to regression by defining a distribution over functions.

The central formulation is:

```text
f(x) ~ GP(m(x), k(x,x'))
```

where:

```text
m(x)
```

is the mean function and:

```text
k(x,x')
```

is the covariance function or kernel.

For a new material, the GP provides:

```text
Predictive mean
+
Predictive variance
```

The major concepts are:

```text
Gaussian Process
Mean function
Covariance function
Kernel
RBF kernel
Matérn kernel
Length scale
Covariance matrix
Observation noise
Predictive mean
Predictive variance
Prediction interval
Materials-space similarity
Small-data regression
```

Gaussian Processes are particularly useful for materials datasets where:

```text
Labels are expensive
Dataset size is relatively small
Uncertainty is important
A meaningful materials representation is available
```

Their main limitation is computational scaling with the number of training samples.

Most importantly:

```text
GP prediction
≠
GP certainty
```

The predictive variance must still be evaluated for calibration, coverage, and practical usefulness.

The next section will examine **27.14 Uncertainty in Classification**, focusing on classification probabilities, confidence, predictive entropy, and uncertainty in materials classification tasks.

## 27.14 Uncertainty in Classification

Uncertainty quantification is not limited to regression problems.

Materials Machine Learning frequently involves classification tasks such as:

```text
Stable / Unstable
Metal / Semiconductor / Insulator
Magnetic / Non-magnetic
Synthesizable / Difficult to synthesize
High-performance / Low-performance
```

A classification model does not usually predict a continuous numerical property.

Instead, it assigns a material to one or more discrete classes.

For example:

```text
Crystal
   ↓
Classification Model
   ↓
Stable
```

However, a scientifically useful classification system should provide more than the final class label.

Consider two predictions:

```text
Material A → Stable
Material B → Stable
```

The model might assign:

```text
Material A → P(stable) = 0.99
Material B → P(stable) = 0.54
```

Both materials receive the same class label, but the model's confidence in those predictions is very different.

Therefore, uncertainty quantification in classification focuses on understanding the reliability of predicted class probabilities.

---

## 27.14.1 Classification Labels vs Probabilities

Suppose a binary classifier predicts whether a material belongs to class 1 or class 0.

The model may produce a probability:

```text
P(y = 1 | x)
```

where:

```text
x
```

is the material representation.

For example:

```text
P(stable | crystal) = 0.87
```

A classification decision can then be made using a threshold.

For a threshold of 0.5:

```text
P(stable) ≥ 0.5
        ↓
Stable
```

and:

```text
P(stable) < 0.5
        ↓
Unstable
```

The probability therefore contains more information than the final class label.

For example:

```text
0.51 → Stable
0.95 → Stable
```

produce the same class but represent very different levels of model confidence.

---

## 27.14.2 Why Confidence Matters in Materials Classification

Consider a model used to identify potentially stable crystal structures.

Suppose it predicts:

```text
Crystal A → Stable, probability = 0.99

Crystal B → Stable, probability = 0.52
```

If only the class labels are reported:

```text
A → Stable
B → Stable
```

the difference between the predictions disappears.

The probability allows the researcher to distinguish:

```text
High-confidence prediction
```

from:

```text
Low-confidence prediction
```

This becomes especially important when classification results are used to select materials for further investigation.

---

## 27.14.3 Binary Classification Uncertainty

For binary classification, a model commonly outputs:

```text
p = P(y = 1 | x)
```

The probability of the other class is:

```text
P(y = 0 | x) = 1 - p
```

For example:

```text
p = 0.90
```

means:

```text
Class 1 → 90%
Class 0 → 10%
```

while:

```text
p = 0.51
```

means:

```text
Class 1 → 51%
Class 0 → 49%
```

The second prediction is much more ambiguous.

Therefore, probability values near the decision boundary often indicate greater classification ambiguity.

---

## 27.14.4 Decision Boundary

Consider a binary classification problem in materials feature space.

Conceptually:

```text
              Feature 2
                 ↑
                 │
        Class 1  │  ● ● ●
                 │ ● ● ●
─────────────────┼────────────────→ Feature 1
                 │
          ○ ○    │
        ○ ○ ○    │
        Class 0
```

The classifier learns a decision boundary separating the two classes.

A material far from the boundary may receive a strong classification probability.

A material close to the boundary may receive a probability near:

```text
0.5
```

For example:

```text
Distance from boundary
        ↓
Classification ambiguity
```

However, probability near 0.5 should not automatically be interpreted as a formally calibrated 50% chance of physical truth.

That interpretation requires calibration.

---

## 27.14.5 Confidence and Probability Are Not Always the Same

A classifier may output:

```text
P(class 1) = 0.95
```

but this does not automatically mean that the prediction is correct 95% of the time.

For example, suppose among 100 predictions assigned probability 0.95, only 80 actually belong to class 1.

Then the classifier is overconfident.

A well-calibrated classifier should approximately satisfy:

```text
Predicted probability
≈
Observed frequency
```

Thus:

```text
0.90 probability
```

should correspond approximately to:

```text
90% empirical correctness
```

over a sufficiently large collection of comparable predictions.

---

## 27.14.6 Classification Calibration

Calibration is therefore essential for uncertainty-aware classification.

Suppose predictions are grouped into probability bins:

```text
0.5–0.6
0.6–0.7
0.7–0.8
0.8–0.9
0.9–1.0
```

For each group, the actual fraction of correct predictions can be calculated.

For example:

```text
Predicted probability    Actual accuracy

0.55                     0.54
0.65                     0.63
0.75                     0.74
0.85                     0.72
0.95                     0.76
```

The final groups demonstrate strong overconfidence.

The model predicts very high probabilities while achieving considerably lower empirical accuracy.

Therefore, classification probability should be evaluated rather than blindly interpreted.

---

## 27.14.7 Reliability Diagrams

A reliability diagram compares:

```text
Predicted probability
```

against:

```text
Observed accuracy
```

The ideal calibration line is:

```text
Observed accuracy = Predicted probability
```

Conceptually:

```text
Observed
accuracy
   ↑
1.0│             /
   │           /
   │         /
   │       /
   │     /
   │   /
0.0└────────────────→
      Predicted
      probability
```

A perfectly calibrated classifier would lie close to the diagonal.

If predictions lie below the diagonal:

```text
Observed accuracy
<
Predicted probability
```

the model is overconfident.

If predictions lie above the diagonal:

```text
Observed accuracy
>
Predicted probability
```

the model is underconfident.

---

## 27.14.8 Classification Uncertainty Near the Decision Boundary

Suppose the classifier predicts:

```text
P(stable) = 0.50
```

The model has little preference between the two classes.

If instead:

```text
P(stable) = 0.99
```

the classifier strongly favors the stable class.

Therefore, a simple uncertainty indicator for binary classification is the distance from the probability extremes.

Conceptually:

```text
P = 0.50
   ↓
High ambiguity

P = 0.75
   ↓
Moderate ambiguity

P = 0.99
   ↓
Low classification ambiguity
```

This is useful, but it is only a model-based confidence measure.

It does not by itself account for distribution shift or model misspecification.

---

## 27.14.9 Predictive Entropy

A more formal measure of classification uncertainty is entropy.

For a binary classifier:

```text
H(p)
=
-p log(p)
-
(1-p)log(1-p)
```

where:

```text
p = P(y=1|x)
```

Entropy is highest when the model is maximally uncertain:

```text
p = 0.5
```

and lowest when the model strongly favors one class:

```text
p → 0
```

or:

```text
p → 1
```

Conceptually:

```text
Probability
0.0 ───── 0.5 ───── 1.0
 │         │         │
Low       High      Low
uncertainty uncertainty uncertainty
```

Thus entropy provides a general mathematical measure of uncertainty in the predicted class distribution.

---

## 27.14.10 Multiclass Classification

Many materials problems involve more than two classes.

For example:

```text
Metal
Semiconductor
Insulator
```

A classifier may produce:

```text
Metal          = 0.10
Semiconductor  = 0.75
Insulator      = 0.15
```

The predicted class is:

```text
Semiconductor
```

because it has the highest probability.

However, the complete probability distribution is more informative than the class label alone.

Another material might receive:

```text
Metal          = 0.34
Semiconductor  = 0.33
Insulator      = 0.33
```

The predicted class may still be:

```text
Metal
```

but the model is highly uncertain.

---

## 27.14.11 Multiclass Predictive Entropy

For a multiclass problem with class probabilities:

```text
p₁, p₂, ..., pₖ
```

predictive entropy is:

```text
H(p)
=
-Σ pᵢ log(pᵢ)
```

A sharply concentrated distribution:

```text
[0.98, 0.01, 0.01]
```

has low entropy.

A nearly uniform distribution:

```text
[0.34, 0.33, 0.33]
```

has high entropy.

Therefore:

```text
Concentrated probability distribution
        ↓
Lower uncertainty

Uniform probability distribution
        ↓
Higher uncertainty
```

This is useful for multiclass materials classification.

---

## 27.14.12 Example: Materials Stability Classification

Suppose a model classifies crystals into:

```text
Stable
Metastable
Unstable
```

For Crystal A:

```text
Stable     = 0.92
Metastable = 0.06
Unstable   = 0.02
```

For Crystal B:

```text
Stable     = 0.40
Metastable = 0.35
Unstable   = 0.25
```

Both predictions may be converted into a single class:

```text
Crystal A → Stable
Crystal B → Stable
```

But their probability distributions are very different.

Crystal A has a concentrated distribution.

Crystal B has a diffuse distribution.

Therefore:

```text
Crystal A
→ lower classification ambiguity

Crystal B
→ higher classification ambiguity
```

This distinction can be valuable when evaluating candidate materials.

---

## 27.14.13 Classification Confidence vs Scientific Certainty

A critical distinction must be maintained:

```text
Model confidence
≠
Scientific certainty
```

Suppose a classifier reports:

```text
P(stable) = 0.98
```

This means the model strongly favors the stable class under its learned probability model.

It does not prove that the material is physically stable.

The prediction may still fail because of:

```text
Training-data limitations
Representation limitations
Model misspecification
Distribution shift
Incorrect labels
Unmodeled physical effects
```

Therefore:

```text
High model confidence
```

should be interpreted as:

```text
Strong model preference
```

rather than:

```text
Physical proof
```

---

## 27.14.14 Classification Uncertainty From Ensembles

Multiple classification models can be trained to estimate model disagreement.

Suppose five models produce:

```text
Model 1 → 0.95
Model 2 → 0.93
Model 3 → 0.96
Model 4 → 0.94
Model 5 → 0.95
```

The predictions are highly consistent.

The ensemble therefore shows low disagreement.

For another material:

```text
Model 1 → 0.95
Model 2 → 0.71
Model 3 → 0.48
Model 4 → 0.82
Model 5 → 0.37
```

The models disagree substantially.

This disagreement can provide an estimate of epistemic uncertainty.

Conceptually:

```text
Multiple models
      ↓
Multiple probabilities
      ↓
Model disagreement
      ↓
Epistemic uncertainty
```

This connects classification uncertainty with the epistemic-uncertainty framework introduced earlier in the chapter.

---

## 27.14.15 Probability Distribution Across an Ensemble

Suppose an ensemble contains `M` classifiers.

For a material `x`, model `j` produces:

```text
pⱼ(x)
```

The ensemble mean probability can be estimated as:

```text
p̄(x)
=
1/M Σ pⱼ(x)
```

The variation among the individual probabilities provides information about model disagreement.

A low variation indicates:

```text
Models agree
```

while high variation indicates:

```text
Models disagree
```

This does not automatically provide a perfectly calibrated uncertainty measure, but it is a useful practical indicator of epistemic uncertainty.

---

## 27.14.16 Classification With Random Forests

Random Forest classifiers naturally contain many individual decision trees.

Each tree produces a class prediction or probability.

For example:

```text
Tree 1 → Stable
Tree 2 → Stable
Tree 3 → Stable
Tree 4 → Unstable
Tree 5 → Stable
```

The fraction of trees predicting a class can be used to estimate an ensemble probability.

For example:

```text
4 / 5 trees → Stable
```

gives an approximate probability of:

```text
0.80
```

With many trees:

```text
Stable predictions
------------------
Total trees
```

provides the Random Forest class probability.

---

## 27.14.17 Random Forest Classification Example

A scikit-learn implementation is straightforward.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)
```

Class probabilities can be obtained with:

```python
probabilities = model.predict_proba(
    X_test
)
```

For a binary problem:

```python
p_stable = probabilities[:, 1]
```

The predicted classes can be obtained using:

```python
predictions = model.predict(
    X_test
)
```

The researcher should retain both:

```text
Predicted class
+
Predicted probability
```

rather than discarding the probability information.

---

## 27.14.18 Neural-Network Classification

A neural network classifier often produces logits.

For a binary classification problem, a sigmoid transformation can convert a logit into a probability:

```text
p = sigmoid(z)
```

For multiclass classification, the softmax function is commonly used:

```text
pᵢ
=
exp(zᵢ)
/
Σⱼ exp(zⱼ)
```

The resulting probabilities satisfy:

```text
Σ pᵢ = 1
```

These probabilities can then be used to quantify classification ambiguity.

However, neural-network probabilities can be poorly calibrated.

Therefore, probability output should not automatically be treated as a reliable uncertainty estimate.

---

## 27.14.19 Predictive Entropy in PyTorch

For a multiclass neural network, predictive entropy can be calculated from the predicted probabilities.

```python
import torch

probs = torch.softmax(
    logits,
    dim=-1
)

entropy = -torch.sum(
    probs * torch.log(
        probs + 1e-12
    ),
    dim=-1
)
```

The small value:

```text
1e-12
```

prevents numerical problems when probabilities are extremely close to zero.

A larger entropy indicates a more uncertain class distribution.

---

## 27.14.20 Materials Example: Electronic Classification

Consider a classifier predicting:

```text
Metal
Semiconductor
Insulator
```

For a crystal:

```text
Metal          0.02
Semiconductor  0.96
Insulator      0.02
```

The entropy is low.

For another crystal:

```text
Metal          0.35
Semiconductor  0.38
Insulator      0.27
```

the entropy is much higher.

Therefore:

```text
Crystal 1
→ confident class distribution

Crystal 2
→ uncertain class distribution
```

This can be particularly useful when the classification boundary is difficult to learn.

---

## 27.14.21 Classification Uncertainty and Imbalanced Data

Materials datasets can contain highly imbalanced classes.

For example:

```text
Stable      = 90%
Unstable    = 10%
```

A model that predicts:

```text
Stable
```

for almost every material may appear highly accurate.

However, its uncertainty estimates and probability outputs may still be poorly informative.

Therefore, uncertainty analysis should be performed alongside appropriate classification metrics.

Important considerations include:

```text
Class distribution
Precision
Recall
F1 score
Probability calibration
Confusion matrix
```

The purpose is not simply to maximize classification accuracy.

The probability estimates must also be scientifically meaningful.

---

## 27.14.22 Classification Uncertainty and Distribution Shift

A classifier may encounter a material unlike anything in the training dataset.

For example, the training set may contain:

```text
Oxides
Sulfides
Nitrides
```

while the test material belongs to a chemical family rarely represented in training.

The classifier may still output:

```text
P(stable) = 0.97
```

even though the model has little relevant experience with that chemical region.

This demonstrates an important limitation:

```text
High predicted probability
```

does not necessarily imply:

```text
High reliability
```

under distribution shift.

Therefore, classification uncertainty should be considered together with the applicability of the training distribution.

---

## 27.14.23 Classification Uncertainty and Applicability

A useful interpretation combines:

```text
Predicted probability
+
Training-data support
```

For example:

```text
Material A
Probability = 0.96
Well represented in training data

→ Stronger evidence
```

versus:

```text
Material B
Probability = 0.96
Far from training distribution

→ Probability should be treated cautiously
```

This is one reason why uncertainty quantification must not be reduced to a single probability number.

---

## 27.14.24 Probability Calibration With Temperature Scaling

Neural-network classification probabilities can sometimes be calibrated using temperature scaling.

Suppose the model produces logits:

```text
z
```

Temperature scaling modifies them as:

```text
z' = z / T
```

where:

```text
T > 0
```

is a learned temperature parameter.

The probabilities are then computed from:

```text
softmax(z / T)
```

A validation dataset is used to determine an appropriate value of `T`.

If the model is overconfident, temperature scaling can soften the predicted probabilities.

For example:

```text
Before:
[0.99, 0.01]

After:
[0.88, 0.12]
```

The exact values depend on the fitted calibration parameter.

---

## 27.14.25 Classification Uncertainty Metrics

Several quantities can be used to evaluate uncertainty-aware classification.

### Predictive entropy

Measures uncertainty in the predicted probability distribution.

### Brier score

Measures the difference between predicted probabilities and actual outcomes.

For binary classification:

```text
Brier Score
=
1/N Σ(pᵢ-yᵢ)²
```

Lower values indicate better probabilistic predictions.

### Log loss

Measures the quality of predicted probabilities and strongly penalizes confident incorrect predictions.

### Expected Calibration Error

Compares predicted confidence with empirical accuracy across probability bins.

These metrics complement ordinary classification metrics.

---

## 27.14.26 Brier Score for Materials Classification

Suppose the classifier predicts:

```text
Material A → 0.90
Material B → 0.80
Material C → 0.20
```

and the true labels are:

```text
A → 1
B → 1
C → 0
```

The Brier score measures how close the predicted probabilities are to these observed outcomes.

A prediction such as:

```text
0.90
```

for a positive sample is rewarded.

A prediction such as:

```text
0.99
```

for a negative sample is strongly penalized.

This makes probabilistic scoring especially useful for evaluating uncertainty-aware classifiers.

---

## 27.14.27 Why Classification Uncertainty Requires Validation

A classification model can produce:

```text
Class label
+
Probability
+
Entropy
```

but none of these automatically guarantees scientific reliability.

A research-grade workflow should therefore evaluate:

```text
Classification performance
        ↓
Probability calibration
        ↓
Uncertainty behavior
        ↓
Distribution-shift behavior
```

This prevents a common mistake:

```text
Model probability
      ↓
Assumed to be truth
```

Instead:

```text
Model probability
      ↓
Validated probabilistic prediction
```

should be the goal.

---

## 27.14.28 Research Workflow for Classification Uncertainty

A practical materials classification workflow is:

```text
Materials Dataset
       ↓
Crystal Representation
       ↓
Train Classifier
       ↓
Predicted Class Probabilities
       ↓
Confidence / Entropy
       ↓
Calibration Evaluation
       ↓
Distribution-Support Analysis
       ↓
Uncertainty-Aware Interpretation
```

For ensemble models:

```text
Materials Dataset
       ↓
Multiple Classifiers
       ↓
Probability Predictions
       ↓
Mean Probability
       +
Model Disagreement
       ↓
Uncertainty Analysis
```

The important principle is that classification uncertainty should be evaluated as part of the complete predictive system.

---

## 27.14.29 Key Takeaways

Classification uncertainty concerns the reliability and ambiguity of predicted class probabilities.

Important concepts include:

```text
Class label
Class probability
Decision boundary
Confidence
Predictive entropy
Calibration
Reliability diagrams
Model disagreement
Ensemble probability
Distribution shift
Brier score
Log loss
```

A class prediction such as:

```text
Stable
```

contains less information than:

```text
Stable
P(stable) = 0.94
```

but even the probability must be interpreted carefully.

The central principles are:

```text
Probability ≠ guaranteed correctness

Confidence ≠ scientific certainty

High entropy → greater class ambiguity

Model disagreement → evidence of epistemic uncertainty

Calibration is necessary

Distribution shift can invalidate naive confidence
```

For Materials Informatics, this is especially important because classification outputs may influence which materials are selected for more expensive computational or experimental investigation.

The next section will examine **27.15 Distribution Shift and Out-of-Distribution Materials**, focusing specifically on how unfamiliar chemical systems, crystal structures, and materials-space regions can affect uncertainty and prediction reliability.

## 27.15 Distribution Shift and Out-of-Distribution Materials

A machine learning model is trained using a particular collection of materials.

That training dataset defines the region of materials space from which the model has learned relationships between inputs and target properties.

For example, a training dataset may contain:

```text
Composition
Structure
Descriptors
Properties
```

and the model learns a relationship of the form:

```text
Materials Representation
        ↓
Machine Learning Model
        ↓
Predicted Property
```

The model performs best when new materials are reasonably similar to the materials represented during training.

However, a new material may differ substantially from the training data.

For example:

```text
Training Data
      ↓
Oxides
Sulfides
Nitrides
      ↓
New Material
      ↓
Unusual Halide Structure
```

The model may still produce a numerical prediction.

The important question is whether that prediction is reliable.

This leads to the concept of **distribution shift**.

---

## 27.15.1 What Is Distribution Shift?

Distribution shift occurs when the statistical distribution of new input data differs from the distribution represented in the training dataset.

Let the training data be sampled from:

```text
P_train(x)
```

and the new data from:

```text
P_test(x)
```

If:

```text
P_train(x) ≠ P_test(x)
```

then the model is operating under a distribution shift.

Conceptually:

```text
Training Distribution
        ↓
   Model learns
        ↓
New Distribution
        ↓
Prediction
```

The model may perform well when the two distributions overlap strongly.

It may perform poorly when the new data lies outside the region represented by the training dataset.

---

## 27.15.2 Why Distribution Shift Matters in Materials Informatics

Materials datasets are often highly heterogeneous.

They may contain variation in:

```text
Elements
Composition
Crystal systems
Space groups
Lattice parameters
Coordination environments
Defect structures
Electronic properties
Magnetic states
```

A training dataset may cover only part of this space.

For example:

```text
Training Dataset

Li
Na
K
Fe
Co
Ni
O
S
```

A model trained on these materials may later encounter:

```text
Cs
W
Te
I
```

The new chemical system may be poorly represented in training.

The model can still produce a prediction because machine learning models generally accept numerical input regardless of whether the underlying material is familiar.

This creates a major scientific risk:

```text
Prediction exists
        ≠
Prediction is reliable
```

---

## 27.15.3 In-Distribution vs Out-of-Distribution Materials

A new material can be approximately classified into two situations.

### In-distribution

The material lies within a region sufficiently represented by the training data.

```text
Training materials
● ● ● ● ●
 ● ● ● ●
  ● ● ●
      ▲
   New material
```

The new material lies near known examples.

### Out-of-distribution

The material lies far outside the region represented by the training data.

```text
Training materials
● ● ● ● ●
 ● ● ● ●
  ● ● ●

                    ▲
             New material
```

The second situation is more difficult for reliable prediction.

However, OOD detection is rarely a simple binary decision.

A material may be:

```text
Very familiar
      ↓
Moderately familiar
      ↓
Poorly represented
      ↓
Strongly out-of-distribution
```

Therefore, applicability should often be treated as a continuous concept.

---

## 27.15.4 Chemical Distribution Shift

One important form of distribution shift occurs in chemical composition.

Suppose a model is trained primarily on:

```text
A-B-O
```

systems.

It may learn strong relationships involving:

```text
A
B
O
```

but then encounter:

```text
A-C-F
```

where both the chemical species and bonding environment differ.

The model may have little empirical evidence for the new system.

This is a chemical distribution shift.

Conceptually:

```text
Training Chemical Space
          ↓
    Model Learning
          ↓
New Chemical Space
          ↓
Potential Distribution Shift
```

The severity depends on how different the new chemical system is from the training data.

---

## 27.15.5 Elemental Coverage

A simple first diagnostic is elemental coverage.

Suppose the training dataset contains:

```text
Li
Na
K
Fe
Co
Ni
O
S
```

A new material containing:

```text
Li
Fe
O
```

uses elements already observed during training.

This does not guarantee that the material is in-distribution, but the model has at least encountered the individual chemical species.

Now consider:

```text
Li
Fe
Te
```

If Te was completely absent from training, the model must extrapolate to a chemical element for which it has no direct training examples.

This represents a much stronger distribution-shift warning.

Therefore, elemental coverage can provide a useful first-level diagnostic.

---

## 27.15.6 Composition Distribution Shift

Even when all elements are known, the composition may be unfamiliar.

For example, suppose the training dataset contains:

```text
LiFeO₂
Li₂FeO₃
LiFe₂O₄
```

and the new material is:

```text
Li₇FeO₁₁
```

The model has seen:

```text
Li
Fe
O
```

individually.

However, the elemental proportions may be substantially different from those represented in training.

Therefore:

```text
Elemental familiarity
```

does not imply:

```text
Compositional familiarity
```

Both should be considered when evaluating distribution shift.

---

## 27.15.7 Structural Distribution Shift

Distribution shift can also occur in crystal structure.

Two materials may contain exactly the same elements but have very different structures.

For example:

```text
Composition:
ABO₃
```

may appear in different structural forms.

The materials can differ in:

```text
Crystal system
Space group
Coordination
Bond lengths
Bond angles
Octahedral distortions
Atomic connectivity
```

A model trained mainly on one structural family may therefore encounter difficulty when presented with a substantially different structure.

Conceptually:

```text
Same Composition
       ↓
Different Structures
       ↓
Different Structural Representation
       ↓
Potential Distribution Shift
```

This is particularly important for structure-based machine learning models.

---

## 27.15.8 Property-Space Distribution Shift

Distribution shift can also occur in the target property.

Suppose the training dataset contains band gaps approximately between:

```text
0 eV and 5 eV
```

and a new material is predicted to have:

```text
8 eV
```

This may indicate that the model is extrapolating beyond the property range represented during training.

Similarly, a training dataset might contain formation energies only within a particular range.

A prediction far outside that range should be examined carefully.

However, property-space extrapolation is not by itself proof of an invalid prediction.

It is a warning that additional validation may be required.

---

## 27.15.9 Representation-Space Distribution Shift

A particularly useful approach is to examine the representation learned or constructed by the model.

Suppose each material is represented by:

```text
x ∈ Rᵈ
```

where `d` is the number of features.

For example:

```text
x =
[
atomic radius,
electronegativity,
density,
coordination,
...
]
```

The training materials occupy a region in this feature space.

A new material can then be compared against this region.

Conceptually:

```text
Feature Space

Training:
● ● ●
 ● ● ● ●
  ● ● ●

New:
              ▲
```

If the new point lies far from the training distribution, it may be considered less familiar.

---

## 27.15.10 Distance-Based Familiarity

A simple approach is to measure the distance between a new material and training materials.

For example, using Euclidean distance:

```text
d(x, xᵢ)
=
√Σⱼ(xⱼ - xᵢⱼ)²
```

The nearest-neighbor distance can be defined as:

```text
d_min(x)
=
minᵢ d(x, xᵢ)
```

A small value means that at least one training material is close in representation space.

A large value means that the new material is far from all training examples.

Conceptually:

```text
Small nearest-neighbor distance
        ↓
More familiar representation

Large nearest-neighbor distance
        ↓
Less familiar representation
```

This is a simple but useful diagnostic.

---

## 27.15.11 Limitations of Raw Distance

Raw Euclidean distance must be interpreted carefully.

Suppose one feature ranges from:

```text
0 to 1
```

while another ranges from:

```text
0 to 10,000
```

The second feature can dominate the distance calculation.

Therefore, feature scaling is important.

For example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_train
)
```

The same transformation must then be applied to new materials:

```python
X_new_scaled = scaler.transform(
    X_new
)
```

Distance calculations should generally be performed in the appropriately transformed representation space.

---

## 27.15.12 Nearest-Neighbor Analysis

Nearest-neighbor analysis can provide more information than a single nearest point.

Suppose the `k` nearest training materials are identified.

```text
New Material
      ↓
Nearest neighbors
      ↓
Material A
Material B
Material C
Material D
Material E
```

The researcher can examine:

```text
Chemical similarity
Structural similarity
Property similarity
Descriptor similarity
```

This helps determine whether the model is making a prediction in a familiar region.

For Materials Informatics, nearest-neighbor analysis can therefore serve as a practical applicability-domain diagnostic.

---

## 27.15.13 Chemical-Space Visualization

Dimensionality-reduction methods can be used to visualize materials distributions.

For example:

```text
High-dimensional descriptors
          ↓
       PCA
          ↓
Two-dimensional representation
          ↓
Materials-space visualization
```

The training and new materials can then be plotted together.

Conceptually:

```text
PCA Component 2
      ↑
      │ ● ● ●
      │ ● ● ●
      │  ● ●
      │
      │                  ▲
      │
      └──────────────────────→ PCA Component 1
```

If the new materials appear within the dense region occupied by training materials, they may be relatively familiar.

If they appear far away, distribution shift should be investigated.

Visualization does not prove whether a prediction is correct, but it can reveal obvious differences between training and prediction populations.

---

## 27.15.14 Example With PCA

Suppose materials are represented using many descriptors.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_train
)

pca = PCA(
    n_components=2
)

X_train_pca = pca.fit_transform(
    X_scaled
)
```

New materials must use the same transformation:

```python
X_new_scaled = scaler.transform(
    X_new
)

X_new_pca = pca.transform(
    X_new_scaled
)
```

The resulting coordinates can then be visualized.

The key rule is:

```text
Fit transformation on training data
                ↓
Apply same transformation to new data
```

The transformation must not be independently fitted to the new data.

---

## 27.15.15 Structural Distribution in Crystal Graphs

For crystal graph models, distribution shift can occur at several levels.

A crystal graph contains:

```text
Nodes
Edges
Node features
Edge features
Graph topology
Periodic connections
```

A new crystal may therefore differ from training structures in:

```text
Number of atoms
Coordination environments
Neighbor distances
Element combinations
Graph connectivity
Local motifs
```

A GNN can still process the graph.

However, the graph may contain local environments that were rarely or never represented during training.

This creates a structural OOD problem.

---

## 27.15.16 Local Environment Distribution Shift

Consider a crystal containing a metal atom coordinated by oxygen.

If the training dataset contains thousands of similar environments:

```text
Metal
 ├── O
 ├── O
 ├── O
 └── O
```

the model has substantial experience with that local environment.

Now suppose a new material contains an unusual environment:

```text
Metal
 ├── Te
 ├── Te
 ├── O
 ├── O
 └── F
```

If such environments were almost absent during training, the corresponding message-passing patterns may be poorly supported by data.

Thus, even when the overall crystal graph looks valid, local structural novelty can create epistemic uncertainty.

---

## 27.15.17 Model Confidence Under Distribution Shift

One of the most dangerous situations is:

```text
Out-of-distribution input
+
High model confidence
```

For example:

```text
New crystal
     ↓
Model
     ↓
Stable = 0.98
```

The probability appears highly confident.

But if the crystal is far outside the training distribution, the probability may not reflect actual reliability.

This is sometimes described as a **confidently wrong prediction**.

The underlying reason is that standard machine learning models optimize prediction performance on the training distribution.

They are not automatically designed to recognize every unfamiliar scientific situation.

---

## 27.15.18 Why Neural Networks Can Be Overconfident

Neural networks can produce strongly peaked probability distributions even for unfamiliar inputs.

For example:

```text
Training examples
● ● ● ● ●

OOD example
                    ▲
                    ↓
              Neural Network
                    ↓
             Class A = 0.99
```

The model may assign a high probability because its learned function maps the new representation to a region associated with Class A.

That does not mean that the model has evidence comparable to the evidence available for training-like materials.

Therefore, uncertainty estimation and OOD analysis should be considered together.

---

## 27.15.19 Ensemble Disagreement as an OOD Signal

Suppose several independently trained models receive the same material.

For an in-distribution crystal:

```text
Model 1 → 0.91
Model 2 → 0.93
Model 3 → 0.90
Model 4 → 0.92
```

The models agree.

For an unfamiliar crystal:

```text
Model 1 → 0.95
Model 2 → 0.62
Model 3 → 0.31
Model 4 → 0.78
```

The predictions differ strongly.

This disagreement can be used as a warning signal.

Conceptually:

```text
Model agreement
      ↓
Lower epistemic warning

Model disagreement
      ↓
Higher epistemic warning
```

However, disagreement is not a universal OOD detector.

Some models may agree for the wrong reason.

Therefore, it should be combined with other diagnostics.

---

## 27.15.20 Uncertainty and Training Density

Another useful diagnostic is the relationship between:

```text
Training-data density
```

and:

```text
Prediction uncertainty
```

A common expectation is:

```text
Dense training region
        ↓
Lower epistemic uncertainty
```

while:

```text
Sparse training region
        ↓
Higher epistemic uncertainty
```

For example:

```text
Materials Space

Dense region:
● ● ● ● ● ●
 ● ● ● ●
   ● ●
   ↓
Low uncertainty

Sparse region:
                 ▲
                 ↓
High uncertainty
```

This relationship should be measured rather than assumed.

A model can sometimes remain uncertain even in dense regions, while some sparse regions may still produce consistent predictions.

---

## 27.15.21 OOD Detection Is Not the Same as Error Prediction

An important distinction is:

```text
OOD detection
```

versus:

```text
Error prediction
```

OOD detection asks:

```text
"Is this material unfamiliar relative to the training distribution?"
```

Error prediction asks:

```text
"How likely is the model's prediction to be wrong?"
```

These questions are related but not identical.

A material can be:

```text
In-distribution
but difficult to predict
```

or:

```text
Out-of-distribution
but accidentally predicted accurately
```

Therefore, OOD status should be interpreted as a warning about model applicability rather than a direct measurement of prediction error.

---

## 27.15.22 Multiple Levels of Distribution Shift

Materials can experience distribution shift at multiple levels simultaneously.

### Element level

New chemical species.

### Composition level

Unusual elemental ratios.

### Structure level

Unfamiliar crystal structures.

### Local-environment level

Unusual coordination or bonding patterns.

### Descriptor level

Feature combinations outside the training region.

### Property level

Targets outside the observed training range.

Conceptually:

```text
Element
   ↓
Composition
   ↓
Structure
   ↓
Local Environment
   ↓
Representation
   ↓
Property
```

Distribution shift should therefore be investigated at the level appropriate to the model and scientific problem.

---

## 27.15.23 Practical OOD Screening Workflow

A practical screening workflow can combine several diagnostics.

```text
New Crystal
     ↓
Element Coverage
     ↓
Composition Similarity
     ↓
Structural Similarity
     ↓
Representation Distance
     ↓
Ensemble Agreement
     ↓
Uncertainty Estimate
     ↓
Applicability Assessment
```

The result should not necessarily be a simple:

```text
OOD / Not OOD
```

Instead, a graded assessment can be more informative:

```text
High familiarity
Moderate familiarity
Low familiarity
Strong OOD warning
```

This provides a more useful basis for interpreting uncertainty.

---

## 27.15.24 Example: Screening Generated Crystal Candidates

Suppose a model predicts properties for 1,000 candidate crystals.

Each candidate receives:

```text
Prediction
+
Uncertainty
+
Representation distance
```

The results might be summarized as:

```text
Candidate    Prediction    Uncertainty    OOD distance

A            2.1 eV        Low             Low
B            2.4 eV        Low             Moderate
C            3.8 eV        High            High
D            2.7 eV        Low             Low
E            4.2 eV        High            High
```

Candidate A is relatively familiar to the model.

Candidates C and E deserve more caution because they combine:

```text
High uncertainty
+
Large distance from training data
```

The important point is that the model should not simply rank candidates by predicted property.

Prediction reliability must also be considered.

---

## 27.15.25 OOD Analysis for Crystal GNNs

For a crystal GNN, a practical workflow can examine:

```text
Crystal
   ↓
Graph Representation
   ↓
Node Embeddings
   ↓
Graph Embedding
   ↓
Training-Distribution Comparison
```

A graph-level embedding can be extracted from the trained model.

For example:

```python
graph_embedding = model.encode(
    crystal_graph
)
```

The new graph embedding can then be compared with embeddings from the training dataset.

A large distance from the training embedding distribution may indicate structural unfamiliarity.

The exact implementation depends on the GNN architecture.

The scientific principle remains:

```text
Learned representation
        ↓
Compare with training representation
        ↓
Assess familiarity
```

---

## 27.15.26 Distribution Shift and Uncertainty Must Be Combined

A useful uncertainty-aware materials workflow should consider both:

```text
Prediction uncertainty
```

and:

```text
Distribution familiarity
```

For example:

```text
                 Low OOD        High OOD
               ┌───────────┬─────────────┐
Low uncertainty│ Stronger  │ Caution     │
               │ confidence│ required    │
               ├───────────┼─────────────┤
High uncertainty│ Moderate │ Strong      │
               │ caution   │ warning     │
               └───────────┴─────────────┘
```

The most concerning case is:

```text
High uncertainty
+
High OOD indication
```

A low uncertainty estimate together with strong OOD evidence should also be treated carefully because the model may simply be overconfident.

---

## 27.15.27 Important Scientific Interpretation

The correct interpretation of OOD analysis is not:

```text
OOD
↓
Prediction is wrong
```

Instead:

```text
OOD
↓
Training evidence is limited
↓
Prediction reliability is less established
↓
Additional validation may be appropriate
```

Similarly:

```text
In-distribution
```

does not mean:

```text
Prediction is guaranteed correct
```

It only means that the new material is relatively well represented by the training distribution.

This distinction is essential for research-grade uncertainty quantification.

---

## 27.15.28 Key Takeaways

Distribution shift occurs when new materials differ from the distribution represented in the training data.

Important forms include:

```text
Chemical shift
Composition shift
Structural shift
Local-environment shift
Representation shift
Property-space shift
```

Out-of-distribution analysis is important because machine learning models can produce predictions for unfamiliar materials without automatically recognizing that they are extrapolating.

The main principles are:

```text
Training distribution matters

Element coverage is not sufficient

Composition similarity matters

Structural similarity matters

Representation-space distance can reveal unfamiliar regions

Ensemble disagreement can provide an epistemic warning

OOD detection is not identical to error prediction

High confidence does not guarantee reliability

Uncertainty and distribution familiarity should be evaluated together
```

For Materials Informatics, the central question is therefore not only:

```text
"What does the model predict?"
```

but also:

```text
"How familiar is this material to the model,
and how reliable is that prediction likely to be?"
```

This provides the foundation for the next stage of uncertainty quantification: **27.16 Out-of-Distribution Detection**, where practical methods for identifying unfamiliar materials and quantifying their distance from the training distribution will be examined.

## 27.16 Out-of-Distribution Detection

Distribution shift describes a difference between the training distribution and the distribution encountered during prediction.

Out-of-distribution detection focuses on the more specific practical question:

```text
Is this new material sufficiently different
from the materials used to train the model?
```

This question is important because a machine learning model can produce a prediction for almost any numerical input, even when that input is very different from the training data.

For example:

```text
Training Materials
       ↓
Machine Learning Model
       ↓
New Crystal
       ↓
Prediction
```

The model may return:

```text
Band gap = 2.4 eV
```

even when the new crystal belongs to a chemical and structural region that was barely represented during training.

Therefore:

```text
Prediction available
        ≠
Training support available
```

Out-of-distribution detection attempts to identify such situations.

---

## 27.16.1 Why OOD Detection Is Necessary

Suppose a model is trained using:

```text
10,000 crystal structures
```

and a new candidate is presented.

The model produces:

```text
Predicted band gap = 3.1 eV
```

Without additional information, this number does not reveal whether the model is interpolating or extrapolating.

The candidate could be:

```text
Very similar to training materials
```

or:

```text
Chemically and structurally unfamiliar
```

Both situations can produce a numerical prediction.

OOD detection provides an additional diagnostic:

```text
Prediction
+
Familiarity assessment
```

This allows the researcher to distinguish between predictions supported by the training distribution and predictions made in poorly represented regions.

---

## 27.16.2 In-Distribution and OOD Detection

Let:

```text
X_train
```

represent the training materials.

For a new material:

```text
x_new
```

the objective is to determine whether:

```text
x_new ∈ familiar training region
```

or:

```text
x_new ∉ familiar training region
```

The output does not necessarily have to be a strict binary label.

A practical system may instead produce an OOD score:

```text
OOD score = low
```

for a familiar material and:

```text
OOD score = high
```

for an unfamiliar material.

Conceptually:

```text
Low OOD score
      ↓
More familiar

High OOD score
      ↓
Less familiar
```

The threshold used to classify a material as OOD depends on the dataset and application.

---

## 27.16.3 OOD Detection Is Not the Same as Classification

A classification model predicts:

```text
Stable
Unstable
```

An OOD detector asks:

```text
Is this material represented adequately
by the training distribution?
```

These are different tasks.

For example:

```text
Material A
    ↓
Stable = 0.95
    ↓
OOD score = low
```

This is a high-confidence prediction for a familiar material.

Another material may produce:

```text
Material B
    ↓
Stable = 0.95
    ↓
OOD score = high
```

The classifier is confident, but the material is unfamiliar.

This illustrates why OOD detection is an important complement to predictive confidence.

---

## 27.16.4 Distance-Based OOD Detection

One of the simplest approaches is to measure the distance between a new material and the training dataset.

Suppose each material is represented by a feature vector:

```text
x = [x₁, x₂, ..., x_d]
```

A distance function can be used:

```text
d(x, x_i)
```

between the new material and each training material.

The nearest training distance is:

```text
d_min(x)
=
min_i d(x, x_i)
```

A small value indicates that the new material is close to at least one training example.

A large value indicates that it lies far from all training examples.

Therefore:

```text
Small d_min
    ↓
More familiar

Large d_min
    ↓
Potential OOD
```

This is one of the most intuitive OOD approaches for Materials Informatics.

---

## 27.16.5 Euclidean Distance

The standard Euclidean distance between two feature vectors is:

```text
d(x, x_i)
=
√Σⱼ(xⱼ - xᵢⱼ)²
```

For a new material, the nearest-neighbor distance is:

```text
d_min
=
min_i d(x, x_i)
```

For example:

```text
Training material 1 → distance = 1.4
Training material 2 → distance = 0.8
Training material 3 → distance = 2.1
Training material 4 → distance = 3.5
```

Then:

```text
d_min = 0.8
```

The new material is closest to training material 2.

However, distance values are meaningful only when the feature representation and scaling are appropriate.

---

## 27.16.6 Feature Scaling

Consider two features:

```text
Atomic radius:
0–3 Å

Formation energy:
-10–1000 eV
```

The formation-energy feature could dominate an ordinary Euclidean distance.

Therefore, features are often standardized before distance-based OOD detection.

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

The same fitted scaler must be applied to new materials.

```python
X_new_scaled = scaler.transform(
    X_new
)
```

The OOD calculation can then be performed in the standardized feature space.

---

## 27.16.7 Nearest-Neighbor OOD Detection

A practical implementation can use nearest-neighbor search.

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(
    n_neighbors=5
)

nn.fit(
    X_train_scaled
)

distances, indices = nn.kneighbors(
    X_new_scaled
)
```

For each new material, the returned distances describe its nearest training examples.

The nearest distance can be extracted using:

```python
nearest_distance = distances[:, 0]
```

A researcher can then inspect the distribution of nearest-neighbor distances for validation data.

For example:

```text
Validation materials
        ↓
Nearest-neighbor distances
        ↓
Reference distribution
```

A new material with an unusually large distance may receive an OOD warning.

---

## 27.16.8 Choosing an OOD Threshold

An OOD threshold should not be selected arbitrarily.

Suppose the nearest-neighbor distances of validation materials are:

```text
0.4
0.6
0.7
0.8
0.9
1.0
1.1
1.2
```

A new material with:

```text
distance = 0.8
```

is relatively typical.

A new material with:

```text
distance = 4.5
```

is much farther away.

One possible threshold can be selected using a high percentile of validation distances.

For example:

```python
import numpy as np

threshold = np.percentile(
    validation_distances,
    95
)
```

Then:

```python
is_ood = (
    nearest_distance > threshold
)
```

The exact percentile should be chosen according to the desired false-positive and false-negative tradeoff.

---

## 27.16.9 Why Validation Data Should Be Used

The training dataset itself should not be the only reference for selecting an OOD threshold.

Training points are, by definition, extremely familiar to the model.

Therefore, their distances to themselves may be close to zero.

A better procedure is:

```text
Training Dataset
      ↓
Train Model
      ↓
Validation Dataset
      ↓
Calculate OOD Scores
      ↓
Determine Reference Distribution
```

The validation set provides examples that were not used to fit the model.

This gives a more realistic estimate of how unfamiliar ordinary in-distribution materials appear.

---

## 27.16.10 Mahalanobis Distance

Euclidean distance treats every feature direction similarly after scaling.

A more flexible approach accounts for correlations between features using the covariance matrix.

This leads to Mahalanobis distance:

```text
D_M(x)
=
√[
(x - μ)ᵀ
Σ⁻¹
(x - μ)
]
```

where:

```text
μ
```

is the mean feature vector and:

```text
Σ
```

is the covariance matrix.

The covariance matrix captures relationships between features.

This can be useful when materials descriptors are strongly correlated.

For example:

```text
Atomic volume
      ↕
Atomic radius
```

may contain related information.

Mahalanobis distance can account for such correlations.

---

## 27.16.11 Interpreting Mahalanobis Distance

A small Mahalanobis distance indicates that the material lies relatively close to the central region of the training distribution after accounting for feature covariance.

A large distance indicates that the material occupies an unusual region.

Conceptually:

```text
Training distribution

        ● ●
     ● ● ● ●
    ● ● ● ● ●
     ● ● ●
        ●

           ▲
       New material
```

If the new point lies far from the distribution, its Mahalanobis distance becomes large.

This provides a distribution-aware OOD score.

---

## 27.16.12 Practical Mahalanobis Implementation

A simple implementation is:

```python
import numpy as np

mu = X_train_scaled.mean(
    axis=0
)

cov = np.cov(
    X_train_scaled,
    rowvar=False
)

cov_inv = np.linalg.pinv(
    cov
)
```

For a new sample:

```python
x = X_new_scaled[0]

diff = x - mu

distance = np.sqrt(
    diff.T
    @ cov_inv
    @ diff
)
```

The pseudoinverse:

```python
np.linalg.pinv()
```

can be useful when the covariance matrix is singular or poorly conditioned.

For high-dimensional materials descriptors, numerical conditioning must be examined carefully.

---

## 27.16.13 Density-Based OOD Detection

Another approach is to estimate how densely the training data occupies different regions of feature space.

Conceptually:

```text
High training density
        ↓
More familiar

Low training density
        ↓
Potential OOD
```

A new material located in a very sparse region can receive a larger OOD score.

Density estimation can be performed using methods such as:

```text
Kernel density estimation
Gaussian mixture models
Nearest-neighbor density estimates
```

The appropriate method depends on the dimensionality and structure of the materials representation.

---

## 27.16.14 Kernel Density Estimation

Kernel density estimation attempts to estimate the probability density of the training distribution.

Conceptually:

```text
Training materials
      ↓
Density estimation
      ↓
p(x)
```

A new material with:

```text
p(x) high
```

lies in a relatively dense region.

A new material with:

```text
p(x) low
```

may be considered less familiar.

An OOD score can therefore be defined using negative log density:

```text
OOD score
=
-log p(x)
```

A low density produces a large score.

However, density estimation becomes increasingly difficult as the dimensionality of the feature space grows.

---

## 27.16.15 The Curse of Dimensionality

Materials descriptors can contain hundreds or thousands of features.

Direct density estimation in such a high-dimensional space can become unreliable.

For example:

```text
1000-dimensional descriptor
```

may contain many directions in which the training dataset is sparse.

Therefore, OOD detection may benefit from a lower-dimensional representation.

A common workflow is:

```text
High-dimensional descriptors
          ↓
Representation learning / dimensionality reduction
          ↓
Lower-dimensional space
          ↓
Distance or density analysis
```

The reduced representation must still preserve scientifically relevant information.

---

## 27.16.16 PCA-Based OOD Detection

Principal Component Analysis can provide a practical representation for OOD analysis.

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=20
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_new_pca = pca.transform(
    X_new_scaled
)
```

The new material can then be compared with the training materials in PCA space.

For example:

```text
Training PCA space
        ↓
Nearest-neighbor distance
        ↓
OOD score
```

The number of principal components should be selected carefully.

Too few components may remove relevant chemical information.

Too many components may retain unnecessary noise.

---

## 27.16.17 Representation-Based OOD Detection

For neural networks, handcrafted descriptors are not always the best representation for OOD detection.

A trained neural network often learns an internal representation.

For example:

```text
Crystal
   ↓
Input representation
   ↓
Neural network layers
   ↓
Latent representation
   ↓
Prediction
```

The latent representation can be extracted before the final prediction layer.

```python
embedding = model.encoder(
    x
)
```

Training embeddings can then be used to define a reference distribution.

A new material can be compared with this learned representation.

This approach asks:

```text
Does the model consider this material
similar to the materials it learned from?
```

---

## 27.16.18 OOD Detection in Crystal GNNs

For a crystal GNN, the learned representation may capture:

```text
Composition
Local environments
Connectivity
Periodic interactions
Structural patterns
```

A simplified workflow is:

```text
Crystal Structure
       ↓
Crystal Graph
       ↓
GNN
       ↓
Graph Embedding
       ↓
OOD Detector
```

For example:

```python
embedding = gnn_model(
    graph,
    return_embedding=True
)
```

The exact implementation depends on the architecture.

The important concept is that OOD detection can be performed using the same representation that the predictive model learned.

---

## 27.16.19 Energy-Based OOD Scores

Neural-network classifiers can also be analyzed using scores derived from their outputs.

One example is an energy-style score based on logits:

```text
E(x)
=
-T log Σᵢ exp(fᵢ(x)/T)
```

where:

```text
fᵢ(x)
```

represents the model logit for class `i`.

The exact interpretation depends on the model and implementation.

The important idea is that the output distribution of a classifier can contain information about whether an input resembles known training examples.

However, energy-based scores should be validated empirically before being used as scientific OOD indicators.

---

## 27.16.20 Ensemble-Based OOD Detection

Ensembles provide another practical approach.

Suppose several independently trained models produce predictions:

```text
Model 1 → 0.92
Model 2 → 0.91
Model 3 → 0.93
Model 4 → 0.92
```

The predictions are consistent.

Now consider:

```text
Model 1 → 0.98
Model 2 → 0.72
Model 3 → 0.41
Model 4 → 0.86
```

The large disagreement indicates that the models respond differently to the new material.

A disagreement score can therefore be used as an OOD-related diagnostic.

For example:

```python
predictions = np.array([
    0.98,
    0.72,
    0.41,
    0.86
])

disagreement = np.var(
    predictions
)
```

A larger variance indicates greater model disagreement.

This is especially useful as an approximate indicator of epistemic uncertainty.

---

## 27.16.21 OOD Detection Using Predictive Entropy

Predictive entropy can also be examined.

For a classifier:

```text
p = [p₁, p₂, ..., pₖ]
```

the entropy is:

```text
H(p)
=
-Σ pᵢ log(pᵢ)
```

An unfamiliar material may sometimes produce a more uniform probability distribution:

```text
[0.34, 0.33, 0.33]
```

leading to high entropy.

However, this is not guaranteed.

A neural network can produce:

```text
[0.99, 0.01]
```

for an OOD material.

Therefore:

```text
High entropy
```

can be an OOD warning, but:

```text
Low entropy
```

does not prove that an input is in-distribution.

---

## 27.16.22 Combining OOD Scores

No single OOD method is universally reliable.

A stronger system can combine multiple diagnostics:

```text
Representation Distance
        +
Nearest-Neighbor Distance
        +
Ensemble Disagreement
        +
Predictive Entropy
        ↓
OOD Assessment
```

For example:

```text
Material A
Distance       → Low
Disagreement   → Low
Entropy        → Low

Material B
Distance       → High
Disagreement   → High
Entropy        → High
```

Material B has multiple independent warning signals.

This provides stronger evidence of unfamiliarity than any single metric.

---

## 27.16.23 OOD Detection Score Table

A practical screening table might contain:

```text
Material   Distance   Entropy   Ensemble Var.   OOD Risk

A          0.8        0.12      0.002           Low
B          1.4        0.25      0.010           Moderate
C          4.8        0.81      0.093           High
```

The values are illustrative.

The purpose is to combine several diagnostics into an interpretable uncertainty report.

The final OOD risk should be based on validation rather than arbitrary numerical rules.

---

## 27.16.24 OOD Detection for Regression

OOD detection is not restricted to classification.

Suppose a model predicts band gap:

```text
Band gap = 2.8 eV
```

The model can also receive an OOD score.

For example:

```text
Material A
Prediction = 2.8 eV
OOD score = low
```

versus:

```text
Material B
Prediction = 2.8 eV
OOD score = high
```

The same predicted property can therefore have different levels of applicability.

This demonstrates an important principle:

```text
Prediction value
```

and:

```text
Prediction applicability
```

are separate pieces of information.

---

## 27.16.25 OOD Detection for Crystal Generation

Generated crystal structures provide a particularly important use case.

Suppose a generative model produces:

```text
10,000 candidate crystals
```

A property-prediction model evaluates them.

If many candidates are far outside the training distribution, their predicted properties may be less reliable.

Therefore:

```text
Generated Crystals
       ↓
Crystal Representation
       ↓
OOD Detection
       ↓
Applicability Assessment
       ↓
Property Prediction
```

can provide a safer prediction workflow.

The purpose here is not to optimize or select candidates.

The purpose is to determine whether the property model has adequate training support for the generated structures.

---

## 27.16.26 OOD Detection and Crystal Diversity

Generated materials may intentionally explore unusual chemical or structural regions.

Therefore, OOD detection should not automatically be treated as a mechanism for rejecting unusual materials.

An OOD candidate can be scientifically interesting.

For example:

```text
High OOD score
        ↓
Unfamiliar material
        ↓
Potentially unreliable ML prediction
```

The correct response is not necessarily:

```text
Discard immediately
```

Instead:

```text
Flag
   ↓
Recognize limited model support
   ↓
Interpret prediction cautiously
```

This distinction is important in materials research.

Unfamiliarity is a statement about model knowledge, not necessarily about the scientific value of the material.

---

## 27.16.27 Validation of an OOD Detector

An OOD detector itself must be evaluated.

A useful experiment is to construct:

```text
Known in-distribution materials
+
Artificially shifted materials
```

The OOD detector should assign:

```text
Lower OOD scores
```

to the first group and:

```text
Higher OOD scores
```

to the second group.

For example:

```text
Training-like test set
        ↓
Expected low OOD scores

Chemically held-out test set
        ↓
Expected higher OOD scores
```

This provides an empirical test of whether the detector actually recognizes unfamiliar materials.

---

## 27.16.28 Chemical Holdout for OOD Evaluation

A materials-specific evaluation can intentionally hold out a chemical system.

For example:

```text
Training:
Li-Fe-O
Na-Fe-O
K-Fe-O

OOD evaluation:
Li-Co-O
Na-Co-O
K-Co-O
```

The held-out chemical family becomes a controlled distribution shift.

The OOD detector can then be evaluated on whether it identifies the held-out materials as less familiar.

This provides a more meaningful test than simply using random test samples.

---

## 27.16.29 Structural Holdout for OOD Evaluation

Similarly, structural families can be held out.

For example:

```text
Training:
Perovskite-like structures

OOD evaluation:
Spinel-like structures
```

or:

```text
Training:
Mostly cubic structures

OOD evaluation:
Mostly monoclinic structures
```

The OOD detector can then be evaluated using structural differences.

This is particularly useful for crystal graph models.

---

## 27.16.30 OOD Detection and Uncertainty Calibration

OOD detection and uncertainty calibration answer related but different questions.

Calibration asks:

```text
When the model reports a particular
uncertainty or probability, is it accurate?
```

OOD detection asks:

```text
Is the material represented adequately
by the training distribution?
```

A research-grade system can therefore combine:

```text
Prediction
+
Uncertainty
+
Calibration
+
OOD score
```

For example:

```text
Prediction:
2.7 eV

Uncertainty:
±0.2 eV

Calibration:
Good

OOD score:
Low
```

This provides much stronger information than:

```text
Prediction:
2.7 eV
```

alone.

---

## 27.16.31 Limitations of OOD Detection

OOD detection has several important limitations.

### No universal OOD definition

A material can be unusual in one representation but ordinary in another.

### Representation dependence

An OOD detector is only as meaningful as the representation on which it operates.

### Threshold dependence

A threshold selected too aggressively may label many valid materials as OOD.

### False positives

A scientifically valid but novel material may be incorrectly flagged.

### False negatives

A model may fail to recognize an unfamiliar material.

### Model dependence

Different models may produce different notions of familiarity.

Therefore, OOD scores should be treated as diagnostics rather than absolute scientific truth.

---

## 27.16.32 OOD Detection Should Be Reported Transparently

A research paper using OOD detection should clearly state:

```text
Representation used
Distance or score used
Reference dataset
Threshold selection
Validation procedure
OOD evaluation strategy
Known limitations
```

For example:

```text
Representation:
Standardized materials descriptors

OOD score:
Nearest-neighbor distance

Reference:
Validation-set distance distribution

Threshold:
95th percentile

Evaluation:
Chemical-system holdout
```

This makes the uncertainty analysis reproducible.

---

## 27.16.33 Research-Grade OOD Workflow

A complete workflow is:

```text
Materials Dataset
       ↓
Train / Validation / Test Split
       ↓
Model Training
       ↓
Material Representation
       ↓
OOD Score Calculation
       ↓
Threshold Calibration
       ↓
OOD Validation
       ↓
Prediction + Uncertainty
       ↓
Applicability Assessment
```

For crystal GNNs:

```text
Crystal Structure
       ↓
Crystal Graph
       ↓
GNN Representation
       ↓
Graph Embedding
       ↓
OOD Score
       ↓
Property Prediction
       +
Uncertainty
```

The final result should provide both the prediction and an assessment of whether the model has adequate support for that prediction.

---

## 27.16.34 Key Takeaways

Out-of-distribution detection is an important component of uncertainty-aware Materials Machine Learning.

The central concepts are:

```text
Training distribution
OOD material
Representation space
Nearest-neighbor distance
Mahalanobis distance
Density estimation
PCA-based analysis
Latent representations
Ensemble disagreement
Predictive entropy
OOD thresholds
Chemical holdout
Structural holdout
```

The most important principles are:

```text
A model can predict unfamiliar materials.

Prediction does not prove familiarity.

High confidence does not guarantee in-distribution behavior.

OOD detection is representation-dependent.

OOD thresholds must be validated.

Chemical and structural holdouts provide useful evaluations.

OOD detection is not the same as prediction-error estimation.

An OOD material is not necessarily scientifically uninteresting.

OOD detection should be combined with uncertainty estimation.
```

For Materials Informatics, the final objective is not simply to label materials as familiar or unfamiliar.

The objective is to understand **how much evidence the training data provides for trusting a model prediction on a particular material**.

This naturally leads to the next topic: **27.17 Calibration of Uncertainty Estimates**, where the chapter will examine how to determine whether reported uncertainty actually corresponds to observed prediction errors and coverage.

## 27.17 Calibration of Uncertainty Estimates

An uncertainty estimate is useful only if it corresponds reasonably well to the actual behavior of the model.

A model may report:

```text
Prediction = 2.5 eV
Uncertainty = ±0.2 eV
```

but this statement has no scientific value by itself.

The researcher must ask:

```text
Does the reported uncertainty actually
describe the prediction errors?
```

This is the purpose of **uncertainty calibration**.

Calibration evaluates whether predicted confidence, probability, intervals, or uncertainty values correspond to observed outcomes.

The central idea is:

```text
Predicted uncertainty
        ↓
Compare with
        ↓
Observed prediction errors
        ↓
Evaluate calibration
```

A well-calibrated uncertainty model should produce uncertainty estimates that have a meaningful statistical relationship with actual prediction errors.

---

## 27.17.1 Why Calibration Matters

Consider two materials predictions:

```text
Material A:
Prediction = 3.0 eV
Uncertainty = ±0.1 eV

Material B:
Prediction = 3.0 eV
Uncertainty = ±0.8 eV
```

The model is communicating that Material A is predicted more precisely than Material B.

But this interpretation is justified only if the uncertainty estimates are reliable.

Suppose the actual errors over many predictions are:

```text
Material A group:
Typical error ≈ 0.5 eV

Material B group:
Typical error ≈ 0.3 eV
```

Then the uncertainty estimates are misleading.

The model is:

```text
Overconfident for A
Underconfident for B
```

Therefore, uncertainty should be evaluated against observed errors.

---

## 27.17.2 Calibration vs Accuracy

Calibration and predictive accuracy are different properties.

Accuracy asks:

```text
How close are predictions to the true values?
```

Calibration asks:

```text
Does the uncertainty correctly describe
the reliability of those predictions?
```

A model can therefore be:

```text
Accurate + well calibrated
Accurate + poorly calibrated
Inaccurate + well calibrated
Inaccurate + poorly calibrated
```

For example, suppose a model consistently predicts:

```text
True value ≈ Prediction ± 1.0 eV
```

The model may have relatively large errors but still provide useful uncertainty estimates if the reported intervals correctly reflect those errors.

Thus:

```text
High accuracy
≠
Good uncertainty calibration
```

and:

```text
Good calibration
≠
High predictive accuracy
```

Both should be evaluated separately.

---

## 27.17.3 Calibration in Regression

Regression models predict continuous quantities such as:

```text
Band gap
Formation energy
Bulk modulus
Formation enthalpy
Magnetic moment
Elastic properties
```

Suppose a model predicts a mean:

```text
μ(x)
```

and an uncertainty:

```text
σ(x)
```

The predictive distribution can be represented as:

```text
y | x
~
N(μ(x), σ²(x))
```

A common interpretation is that an interval around the prediction should contain the true value with a particular frequency.

For example, an approximate 68% interval is:

```text
μ ± σ
```

while an approximate 95% interval for a Gaussian predictive distribution is:

```text
μ ± 1.96σ
```

However, these coverage statements are valid only when the predictive uncertainty is properly calibrated.

---

## 27.17.4 Prediction Intervals

A prediction interval represents a range of plausible values.

For example:

```text
Predicted band gap:

2.8 eV ± 0.3 eV
```

produces:

```text
[2.5 eV, 3.1 eV]
```

The interval can be written generally as:

```text
[lower_bound, upper_bound]
```

The important question is:

```text
How often does the true value
actually fall inside this interval?
```

This is called **coverage**.

---

## 27.17.5 Coverage Probability

Suppose a model produces 100 prediction intervals intended to provide 90% coverage.

Ideally:

```text
Approximately 90
of the 100 true values
```

should fall inside their corresponding intervals.

The empirical coverage is:

```text
Coverage
=
Number of true values inside intervals
/
Total number of predictions
```

For example:

```text
90 intervals contain the true value
100 total intervals
```

gives:

```text
Coverage = 0.90
```

This is consistent with the intended 90% coverage.

If only 60 of the 100 values are inside the intervals:

```text
Coverage = 0.60
```

then the uncertainty intervals are too narrow for the claimed 90% coverage.

---

## 27.17.6 Undercoverage and Overcoverage

Two important calibration failures are **undercoverage** and **overcoverage**.

### Undercoverage

Suppose a model claims:

```text
90% prediction interval
```

but only:

```text
65%
```

of true values fall inside the interval.

The intervals are too narrow.

The model is underestimating uncertainty.

This is a form of overconfidence.

### Overcoverage

Suppose a model claims:

```text
90% prediction interval
```

but:

```text
99%
```

of observations fall inside.

The intervals are much wider than necessary.

The model is conservative or underconfident.

Therefore:

```text
Undercoverage
→ uncertainty too small

Overcoverage
→ uncertainty too large
```

A useful uncertainty model should balance coverage with interval sharpness.

---

## 27.17.7 Coverage Is Not Enough

A model could produce extremely wide intervals:

```text
Prediction interval:
-100 eV to +100 eV
```

Almost every material would fall inside.

This could result in:

```text
Coverage ≈ 100%
```

but the uncertainty estimate would not be useful.

Therefore, calibration must consider both:

```text
Coverage
+
Interval width
```

This leads to the concept of **sharpness**.

A useful uncertainty model should provide:

```text
High appropriate coverage
+
Narrow useful intervals
```

rather than simply maximizing coverage.

---

## 27.17.8 Sharpness

Sharpness describes how concentrated the predictive distributions or intervals are.

For prediction intervals:

```text
Narrow interval
→ higher sharpness

Wide interval
→ lower sharpness
```

Suppose two models both achieve approximately 90% coverage.

Model A:

```text
Average interval width = 0.4 eV
```

Model B:

```text
Average interval width = 1.5 eV
```

If both are correctly calibrated, Model A provides more informative predictions.

Therefore, uncertainty evaluation should consider:

```text
Calibration
+
Sharpness
```

together.

---

## 27.17.9 Calibration Curve for Regression

A simple way to examine calibration is to compare nominal coverage with empirical coverage.

Suppose prediction intervals are constructed at several nominal levels:

```text
50%
60%
70%
80%
90%
95%
```

For each level, calculate the fraction of true values that actually fall inside the intervals.

For example:

```text
Nominal       Empirical

50%           48%
60%           57%
70%           68%
80%           76%
90%           84%
95%           90%
```

A perfectly calibrated model would approximately follow:

```text
Empirical coverage
        =
Nominal coverage
```

Conceptually:

```text
Empirical
coverage
   ↑
1.0│             /
   │           /
   │         /
   │       /
   │     /
   │   /
0.0└────────────────→
      Nominal coverage
```

Deviation from the diagonal indicates miscalibration.

---

## 27.17.10 Calibration Curve Implementation

Suppose a model provides:

```python
mean_predictions
std_predictions
```

and the true values are:

```python
y_true
```

Prediction intervals can be evaluated at different confidence levels.

```python
import numpy as np

levels = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95
]
```

For a Gaussian predictive distribution, the corresponding interval multiplier can be obtained from a standard normal distribution.

```python
from scipy.stats import norm

empirical_coverage = []

for level in levels:

    alpha = 1.0 - level

    z = norm.ppf(
        1.0 - alpha / 2.0
    )

    lower = (
        mean_predictions
        - z * std_predictions
    )

    upper = (
        mean_predictions
        + z * std_predictions
    )

    covered = (
        (y_true >= lower)
        &
        (y_true <= upper)
    )

    coverage = covered.mean()

    empirical_coverage.append(
        coverage
    )
```

The resulting values can be compared with:

```python
levels
```

to construct a calibration curve.

---

## 27.17.11 Example: Band-Gap Prediction Calibration

Suppose a materials model predicts band gaps.

For a set of test crystals, it produces:

```text
Prediction
+
Standard deviation
```

The model is intended to provide approximately:

```text
90% prediction intervals
```

Suppose:

```text
Number of test crystals = 500
```

and:

```text
True band gap inside interval = 425
```

Then:

```text
Empirical coverage
=
425 / 500
=
0.85
```

The model therefore provides:

```text
85% coverage
```

instead of the intended:

```text
90%
```

This indicates undercoverage.

The model's uncertainty is likely too small.

---

## 27.17.12 Calibration of Heteroscedastic Models

Some models predict different uncertainty values for different materials.

For example:

```text
Crystal A:
σ = 0.05 eV

Crystal B:
σ = 0.20 eV

Crystal C:
σ = 0.80 eV
```

This is useful because some materials may intrinsically be easier to predict than others.

However, the predicted uncertainty should still correlate with actual error.

A useful diagnostic is to compare:

```text
Predicted σ
```

with:

```text
Absolute prediction error
```

For each material:

```text
errorᵢ
=
|yᵢ - ŷᵢ|
```

Then the researcher can examine whether larger predicted uncertainties generally correspond to larger errors.

Conceptually:

```text
Predicted uncertainty
        ↑
        │        ●
        │      ●
        │   ●
        │ ●
        └────────────→
          Actual error
```

A positive relationship is desirable, although exact equality is not required.

---

## 27.17.13 Reliability of Uncertainty Ranking

Sometimes the researcher is more interested in ranking materials by uncertainty than in obtaining perfectly calibrated numerical intervals.

For example:

```text
Material A → low uncertainty
Material B → medium uncertainty
Material C → high uncertainty
```

A useful uncertainty model should ideally identify materials that actually have larger prediction errors.

Therefore, one can compare:

```text
Predicted uncertainty ranking
```

against:

```text
Observed error ranking
```

If materials with larger predicted uncertainty consistently have larger errors, the uncertainty estimate provides useful information even if it is not perfectly calibrated.

---

## 27.17.14 Calibration of Classification Probabilities

Calibration is also essential for classification.

Suppose a classifier predicts:

```text
P(stable) = 0.90
```

For a well-calibrated model, among many materials assigned approximately 0.90 probability, roughly 90% should actually be stable.

Thus:

```text
Predicted probability
        ↓
Observed frequency
```

should agree.

This is the classification analogue of regression interval coverage.

---

## 27.17.15 Reliability Diagram

A reliability diagram groups predictions by confidence.

For example:

```text
0.5–0.6
0.6–0.7
0.7–0.8
0.8–0.9
0.9–1.0
```

For each group, calculate:

```text
Average predicted confidence
```

and:

```text
Observed accuracy
```

Suppose:

```text
Confidence     Accuracy

0.55           0.57
0.65           0.64
0.75           0.73
0.85           0.82
0.95           0.77
```

The final group shows overconfidence.

The model claims approximately 95% confidence but achieves only 77% accuracy.

This is a strong calibration problem.

---

## 27.17.16 Reliability Diagram in Materials Classification

Consider a classifier predicting whether a crystal is stable.

Suppose 1,000 predictions are grouped into probability bins.

The researcher calculates:

```text
Bin          Mean confidence     Accuracy

0.5–0.6      0.55                0.54
0.6–0.7      0.65                0.67
0.7–0.8      0.75                0.72
0.8–0.9      0.85                0.83
0.9–1.0      0.95                0.78
```

The model is reasonably calibrated in the middle ranges but strongly overconfident at high probabilities.

This is scientifically important because high-confidence predictions may be used to prioritize candidate materials.

---

## 27.17.17 Expected Calibration Error

A commonly used classification calibration metric is Expected Calibration Error, or ECE.

The predictions are divided into probability bins.

For each bin:

```text
confidence = average predicted probability
```

and:

```text
accuracy = observed fraction correct
```

The calibration gap is:

```text
|accuracy - confidence|
```

ECE is then computed as a weighted average of these gaps:

```text
ECE
=
Σₘ
(nₘ / N)
|
accuracyₘ - confidenceₘ
|
```

where:

```text
nₘ
```

is the number of predictions in bin `m`, and:

```text
N
```

is the total number of predictions.

Lower ECE indicates better agreement between predicted confidence and observed accuracy.

---

## 27.17.18 Example ECE Calculation

Suppose there are three confidence bins:

```text
Bin 1:
n = 100
confidence = 0.60
accuracy = 0.58

Bin 2:
n = 300
confidence = 0.75
accuracy = 0.72

Bin 3:
n = 600
confidence = 0.95
accuracy = 0.85
```

The calibration gaps are:

```text
Bin 1:
|0.58 - 0.60| = 0.02

Bin 2:
|0.72 - 0.75| = 0.03

Bin 3:
|0.85 - 0.95| = 0.10
```

The final bin contributes strongly to the total ECE because it contains many predictions and has a large calibration error.

This illustrates why a model can appear well calibrated overall while still being substantially overconfident for high-confidence predictions.

---

## 27.17.19 Calibration and Class Imbalance

Calibration becomes more complicated when classes are highly imbalanced.

Suppose:

```text
Stable      = 90%
Unstable    = 10%
```

A model that predicts high probability for the majority class may appear accurate.

Therefore, calibration should be evaluated using appropriate test distributions and should not be interpreted only through overall accuracy.

The researcher should examine:

```text
Class-specific behavior
Probability calibration
Confusion matrix
Prediction distribution
```

This is particularly important when the minority class is scientifically important.

---

## 27.17.20 Calibration of Ensemble Uncertainty

Ensemble models often provide uncertainty from variation among predictions.

Suppose five models predict:

```text
2.1
2.2
2.0
2.1
2.3 eV
```

The ensemble mean is:

```text
2.14 eV
```

and the ensemble spread can be used as an uncertainty estimate.

However, raw ensemble spread is not automatically calibrated.

Suppose the ensemble reports:

```text
σ = 0.1 eV
```

but actual test errors are commonly:

```text
0.4 eV
```

Then the ensemble is overconfident.

Therefore:

```text
Ensemble variance
```

must also be evaluated against observed prediction errors and coverage.

---

## 27.17.21 Calibration After Ensemble Prediction

A practical workflow is:

```text
Training Data
      ↓
Train Ensemble
      ↓
Validation Predictions
      ↓
Estimate Uncertainty
      ↓
Calibrate
      ↓
Test Evaluation
```

The calibration step should use data that were not used to fit the original predictive model.

This prevents the calibration procedure from simply learning the training errors.

---

## 27.17.22 Calibration Data and Test Data

A clean experimental setup separates:

```text
Training
Validation / Calibration
Test
```

The training set is used to fit the model.

The calibration set is used to adjust uncertainty or probability estimates.

The test set is reserved for final evaluation.

Conceptually:

```text
Training
   ↓
Model fitting

Calibration
   ↓
Uncertainty calibration

Test
   ↓
Final evaluation
```

Using the test set repeatedly to tune calibration can lead to optimistic estimates of performance.

---

## 27.17.23 Calibration for Materials Data

A materials dataset should ideally preserve meaningful scientific separation.

For example:

```text
Training:
Known chemical systems

Calibration:
Independent materials

Test:
Held-out materials
```

Depending on the research question, the test set may use:

```text
Random split
Chemical-system split
Structural-family split
```

The calibration procedure should be evaluated using the same type of distribution expected during actual deployment.

This is important because a calibration method that works on random test data may not remain calibrated under a strong chemical distribution shift.

---

## 27.17.24 Calibration Under Distribution Shift

Suppose a model is well calibrated on ordinary test data:

```text
90% interval
→ 90% empirical coverage
```

Now consider a chemically shifted dataset.

The same model may produce:

```text
90% interval
→ 60% empirical coverage
```

The model has become substantially undercalibrated under the shifted distribution.

This demonstrates:

```text
Calibration depends on the prediction environment.
```

Therefore, calibration should be tested under the distribution relevant to the scientific application.

---

## 27.17.25 Calibration and OOD Materials

Calibration and OOD detection should be interpreted together.

Suppose:

```text
Material A:
Uncertainty = low
OOD score = low
Calibration = good
```

This is relatively strong evidence that the model is operating in a familiar region.

Now consider:

```text
Material B:
Uncertainty = low
OOD score = high
Calibration = unknown
```

The prediction should be treated cautiously.

Finally:

```text
Material C:
Uncertainty = high
OOD score = high
Calibration = poor
```

This is a strong warning that the model has limited support for the prediction.

Thus:

```text
Uncertainty
+
Calibration
+
OOD assessment
```

provide complementary information.

---

## 27.17.26 Practical Regression Calibration Workflow

A research-grade regression calibration workflow can be structured as:

```text
Materials Dataset
       ↓
Train Model
       ↓
Validation Predictions
       ↓
Predictive Mean
       +
Predictive Uncertainty
       ↓
Construct Prediction Intervals
       ↓
Calculate Empirical Coverage
       ↓
Compare With Nominal Coverage
       ↓
Evaluate Sharpness
       ↓
Calibrated Uncertainty
       ↓
Test Evaluation
```

The key quantities are:

```text
Nominal coverage
Empirical coverage
Interval width
Prediction error
```

---

## 27.17.27 Practical Classification Calibration Workflow

For classification:

```text
Materials Dataset
       ↓
Train Classifier
       ↓
Validation Probabilities
       ↓
Reliability Diagram
       ↓
ECE / Brier Score / Log Loss
       ↓
Probability Calibration
       ↓
Test Evaluation
```

The researcher should retain:

```text
Predicted class
+
Predicted probability
```

throughout the workflow.

Converting probabilities immediately into class labels destroys useful information about confidence.

---

## 27.17.28 Simple Classification Calibration Code

A scikit-learn classifier can be calibrated using a separate validation strategy.

For example:

```python id="n8q3tb"
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = (
    CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=5
    )
)

calibrated_model.fit(
    X_train,
    y_train
)
```

Predicted probabilities can then be obtained with:

```python id="72c0cg"
probabilities = (
    calibrated_model
    .predict_proba(X_test)
)
```

The important point is not the specific calibration method.

The important point is that the resulting probabilities should be evaluated against observed outcomes.

---

## 27.17.29 Calibration Does Not Improve Every Aspect of the Model

Calibration modifies the interpretation of uncertainty or probability.

It does not necessarily improve the underlying representation of materials.

For example, calibration cannot automatically fix:

```text
Poor crystal representation
Insufficient training data
Incorrect labels
Systematic physical bias
Severe distribution shift
```

Therefore:

```text
Calibration
```

should not be treated as a replacement for:

```text
Good data
+
Good representation
+
Good predictive model
```

Calibration is an additional layer that makes uncertainty estimates more meaningful.

---

## 27.17.30 Calibration and Scientific Interpretation

Suppose a materials model predicts:

```text
Formation energy = -2.4 eV
90% prediction interval = [-2.8, -2.0] eV
```

If the uncertainty is calibrated, the researcher can interpret the interval according to its validated coverage behavior.

For example, if the method demonstrates approximately 90% coverage on a comparable evaluation set, then the interval has a meaningful statistical interpretation.

Without calibration, the same interval:

```text
[-2.8, -2.0] eV
```

may simply be an arbitrary numerical range.

Thus:

```text
Numerical interval
        ↓
Calibration
        ↓
Statistically interpretable uncertainty
```

---

## 27.17.31 Calibration as a Research Requirement

For research-grade Materials Machine Learning, uncertainty should not be reported as:

```text
Prediction ± uncertainty
```

without explaining how the uncertainty was evaluated.

A stronger report includes:

```text
Prediction
Uncertainty method
Calibration procedure
Coverage
Interval width
Test-set performance
Distribution-shift evaluation
```

For example:

```text
Band gap:
2.7 eV

90% prediction interval:
[2.3, 3.1] eV

Empirical 90% coverage:
89%

Median interval width:
0.82 eV
```

This gives the reader information about both the prediction and the reliability of the uncertainty estimate.

---

## 27.17.32 Common Calibration Failure Modes

Several problems occur frequently.

### 1. Treating predicted standard deviation as automatically valid

A neural network or ensemble may output a standard deviation, but this does not guarantee correct coverage.

### 2. Calibrating on the test set

This can produce optimistic evaluation.

### 3. Ignoring distribution shift

Calibration on random test data may not apply to chemically shifted materials.

### 4. Maximizing coverage without considering interval width

Very wide intervals can achieve high coverage while providing little useful information.

### 5. Reporting uncertainty without its definition

`±0.2 eV` is ambiguous unless the researcher explains what the quantity represents.

### 6. Confusing confidence with probability of physical truth

A model probability is not automatically a physical probability.

---

## 27.17.33 Recommended Calibration Report

For a Materials Informatics study, a useful uncertainty report can contain:

```text
Model:
Deep Ensemble

Prediction:
2.6 eV

Uncertainty:
0.25 eV

Interval:
[2.1, 3.1] eV

Nominal coverage:
90%

Observed coverage:
88%

Mean interval width:
1.0 eV

OOD assessment:
Low

Calibration:
Acceptable
```

This provides a much more complete description than reporting the point prediction alone.

---

## 27.17.34 Key Takeaways

Calibration determines whether uncertainty estimates and class probabilities correspond to observed model behavior.

The most important concepts are:

```text
Calibration
Prediction interval
Coverage
Undercoverage
Overcoverage
Sharpness
Reliability diagram
Expected Calibration Error
Brier score
Log loss
Calibration dataset
Distribution-shift calibration
```

For regression:

```text
Nominal coverage
        ↓
Compare with
        ↓
Empirical coverage
```

For classification:

```text
Predicted confidence
        ↓
Compare with
        ↓
Observed accuracy
```

The central principles are:

```text
Accuracy and calibration are different.

Uncertainty must be empirically evaluated.

Coverage is important, but coverage alone is insufficient.

Sharpness matters.

Calibration should use independent validation data.

Calibration should be evaluated under relevant distribution shifts.

OOD detection and calibration provide complementary information.

Reported uncertainty should have a clearly defined interpretation.
```

The goal is therefore not simply to produce an uncertainty value.

The goal is to produce an uncertainty estimate whose behavior has been **validated against observed outcomes**.

The next section will examine **27.18 Calibration Methods**, focusing specifically on practical techniques for improving poorly calibrated classification probabilities and uncertainty estimates.

## 27.18 Calibration Methods

Calibration analysis determines whether predicted probabilities or uncertainty intervals are reliable.

However, identifying miscalibration is only the first step.

If a model is systematically overconfident or underconfident, the researcher may need to **calibrate the predictions**.

Calibration methods transform the original model outputs so that their statistical interpretation better matches observed outcomes.

The general workflow is:

```text
Trained Model
      ↓
Raw Predictions
      ↓
Calibration Dataset
      ↓
Calibration Method
      ↓
Calibrated Predictions
      ↓
Independent Evaluation
```

The original predictive model does not necessarily need to be retrained.

Instead, a separate calibration procedure can learn how the model's outputs should be adjusted.

For Materials Machine Learning, this is useful because uncertainty and probability estimates are often used to determine how much trust should be placed in a prediction.

---

## 27.18.1 Why Calibration Methods Are Needed

A classification model may produce:

```text
P(stable) = 0.95
```

but the actual frequency of stable materials among predictions near 0.95 may be only:

```text
80%
```

The model is therefore overconfident.

Similarly, a regression model may report:

```text
90% prediction interval
```

while only 70% of the true values fall inside those intervals.

In both cases, the model's uncertainty information is miscalibrated.

Calibration methods attempt to correct this mismatch.

Conceptually:

```text
Raw prediction
      ↓
Calibration mapping
      ↓
Better-calibrated prediction
```

---

## 27.18.2 Calibration Should Use Separate Data

The calibration procedure should normally use data that were not used to fit the original model.

A clean experimental structure is:

```text
Training Set
    ↓
Fit predictive model

Calibration Set
    ↓
Fit calibration mapping

Test Set
    ↓
Evaluate final performance
```

This separation is important.

If the same data are used to train the model and calibrate its uncertainty, the calibration procedure may learn characteristics specific to the training data.

The resulting uncertainty can then appear better calibrated than it actually is on unseen materials.

---

## 27.18.3 Calibration as a Mapping

Suppose a classifier produces a raw probability:

```text
p_raw
```

A calibration method learns a mapping:

```text
p_calibrated
=
g(p_raw)
```

where:

```text
g()
```

is learned using calibration data.

For example:

```text
Raw probability → Calibrated probability

0.50 → 0.51
0.60 → 0.58
0.70 → 0.66
0.80 → 0.74
0.90 → 0.84
0.95 → 0.88
```

This type of mapping can correct systematic overconfidence.

The important point is that the calibration mapping is learned from observed outcomes.

---

# 27.18.4 Temperature Scaling

Temperature scaling is one of the simplest calibration methods for classification models.

It is particularly useful for neural-network classifiers.

A classifier produces logits:

```text
z₁, z₂, ..., zₖ
```

The standard softmax probability is:

```text
pᵢ
=
exp(zᵢ)
/
Σⱼ exp(zⱼ)
```

Temperature scaling introduces a positive scalar:

```text
T
```

and modifies the logits:

```text
pᵢ
=
exp(zᵢ / T)
/
Σⱼ exp(zⱼ / T)
```

The temperature is learned using a separate calibration dataset.

---

## 27.18.5 Interpreting the Temperature

The value of `T` controls the sharpness of the probability distribution.

If:

```text
T = 1
```

the original softmax probabilities are unchanged.

If:

```text
T > 1
```

the logits are softened.

The probabilities become less extreme.

For example:

```text
Before:

[0.98, 0.02]
```

may become:

```text
After temperature scaling:

[0.88, 0.12]
```

This can correct an overconfident model.

If:

```text
T < 1
```

the probability distribution becomes sharper.

Therefore, temperature scaling can correct systematic overconfidence or underconfidence.

---

## 27.18.6 Why Temperature Scaling Is Attractive

Temperature scaling has several advantages:

```text
Simple
Low computational cost
Does not modify the model architecture
Uses only one learned parameter
Works well for many neural classifiers
```

The original neural network remains unchanged.

Only the interpretation of its logits is modified.

The workflow is:

```text
Neural Network
      ↓
Logits
      ↓
Temperature T
      ↓
Calibrated probabilities
```

This makes temperature scaling particularly convenient when a Materials Informatics project already has a trained PyTorch classifier.

---

## 27.18.7 Temperature Scaling Example

Suppose a crystal stability classifier produces:

```python
logits = model(x)
```

A temperature parameter can be applied:

```python
temperature = 2.0

scaled_logits = (
    logits / temperature
)
```

Then:

```python
import torch

probabilities = torch.softmax(
    scaled_logits,
    dim=-1
)
```

In actual calibration, the temperature should not simply be chosen as `2.0`.

It should be optimized using a calibration dataset.

---

## 27.18.8 Learning the Temperature

Suppose the validation dataset contains:

```text
logits
true_labels
```

The temperature can be optimized by minimizing a calibration loss such as negative log-likelihood.

Conceptually:

```text
Initialize T
      ↓
Calculate calibrated probabilities
      ↓
Calculate validation loss
      ↓
Update T
      ↓
Repeat
```

The objective is:

```text
T*
=
argmin_T
NLL(T)
```

where `T*` is the temperature that provides the best fit to the calibration data according to the selected loss.

---

## 27.18.9 Temperature Scaling in Materials Classification

Consider a model that predicts whether a material is:

```text
Stable
Unstable
```

The raw classifier might produce:

```text
Material A → 0.97 stable
Material B → 0.91 stable
Material C → 0.84 stable
```

After calibration:

```text
Material A → 0.88 stable
Material B → 0.83 stable
Material C → 0.79 stable
```

The model has not necessarily changed its ranking.

Instead, the probabilities have become less extreme.

The calibrated probability can then be compared against observed frequencies.

---

# 27.18.10 Platt-Style Calibration

Another common method is logistic calibration, often associated with Platt scaling.

The raw model score is transformed using a logistic function:

```text
p
=
1
/
(
1 + exp(Af + B)
)
```

where:

```text
f
```

is the original model score and:

```text
A, B
```

are learned using calibration data.

This method is especially useful for binary classification.

The workflow becomes:

```text
Raw classifier score
        ↓
Logistic calibration
        ↓
Calibrated probability
```

---

## 27.18.11 Materials Example of Logistic Calibration

Suppose a binary model predicts whether a material satisfies a target criterion.

The raw model score may be:

```text
Material A → 3.2
Material B → 1.8
Material C → -0.4
Material D → -2.1
```

These scores are not necessarily probabilities.

A logistic calibration model can transform them into:

```text
Material A → 0.96
Material B → 0.82
Material C → 0.40
Material D → 0.09
```

The mapping is learned from calibration examples for which the actual material outcomes are known.

---

# 27.18.12 Isotonic Regression

Isotonic regression provides a non-parametric calibration method.

Instead of assuming a particular mathematical form such as a sigmoid, isotonic regression learns a monotonic mapping:

```text
p_calibrated
=
g(p_raw)
```

subject to:

```text
p₁ ≤ p₂
⇒
g(p₁) ≤ g(p₂)
```

The mapping is therefore monotonic.

Conceptually:

```text
Raw probability

0.1 ──┐
0.2 ──┤
0.3 ──┤
0.4 ──┤
0.5 ──┤
0.6 ──┤
0.7 ──┤
0.8 ──┤
0.9 ──┘

       ↓

Learned monotonic mapping
```

The mapping can be more flexible than temperature scaling or logistic calibration.

---

## 27.18.13 Advantages of Isotonic Regression

Isotonic regression can model complicated calibration relationships.

For example, the raw model might have:

```text
Moderate confidence
→ reasonably calibrated

Very high confidence
→ strongly overconfident
```

A simple temperature parameter may not fully capture this behavior.

Isotonic regression can learn a more flexible transformation.

However, flexibility creates another problem:

```text
Too much flexibility
        ↓
Potential overfitting
```

Therefore, isotonic regression generally requires sufficient calibration data.

---

## 27.18.14 Isotonic Regression with scikit-learn

A simple example is:

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(
    out_of_bounds="clip"
)

calibrator.fit(
    raw_probabilities,
    y_calibration
)
```

Calibrated probabilities are obtained with:

```python
calibrated_probabilities = (
    calibrator.predict(
        raw_probabilities
    )
)
```

The calibration model should be fitted only using the calibration dataset.

---

# 27.18.15 Temperature Scaling vs Isotonic Regression

The two methods make different assumptions.

| Property                  | Temperature Scaling | Isotonic Regression |
| ------------------------- | ------------------- | ------------------- |
| Mapping                   | Parametric          | Non-parametric      |
| Parameters                | One temperature     | Flexible mapping    |
| Complexity                | Low                 | Higher              |
| Data requirement          | Relatively small    | Larger              |
| Overfitting risk          | Low                 | Higher              |
| Neural networks           | Very suitable       | Suitable            |
| Binary classification     | Suitable            | Suitable            |
| Multiclass classification | Very suitable       | More complicated    |

Temperature scaling is often a good first choice for neural-network classifiers.

Isotonic regression can be useful when calibration is clearly nonlinear and sufficient calibration data are available.

---

# 27.18.16 Calibration of Regression Uncertainty

Classification calibration methods such as temperature scaling operate on probabilities.

Regression uncertainty requires a somewhat different approach.

Suppose a model predicts:

```text
μ(x)
```

and:

```text
σ(x)
```

A simple Gaussian predictive interval is:

```text
μ(x) ± zσ(x)
```

If the intervals are systematically too narrow, the uncertainty can be rescaled.

For example:

```text
σ_calibrated
=
cσ_predicted
```

where:

```text
c > 1
```

can increase uncertainty when the model is overconfident.

---

## 27.18.17 Variance Scaling

Suppose a model produces uncertainties:

```text
0.1
0.2
0.3
0.4 eV
```

but validation experiments indicate that actual errors are systematically larger.

A scaling factor can be learned:

```text
c = 1.5
```

giving:

```text
0.15
0.30
0.45
0.60 eV
```

The goal is not simply to increase uncertainty arbitrarily.

The scaling factor should be estimated from calibration data so that the resulting prediction intervals achieve the desired empirical coverage.

---

## 27.18.18 Regression Calibration by Coverage

Suppose the desired coverage is:

```text
90%
```

The researcher evaluates the current intervals on calibration data.

If the observed coverage is:

```text
65%
```

the intervals are too narrow.

A scaling factor can then be optimized so that:

```text
Empirical coverage
≈
90%
```

The calibration workflow becomes:

```text
Raw uncertainty
      ↓
Construct intervals
      ↓
Measure coverage
      ↓
Adjust uncertainty scale
      ↓
Measure coverage again
```

This procedure can be repeated until the calibration criterion is satisfied.

---

# 27.18.19 Calibrating Heteroscedastic Uncertainty

A single global scaling factor may not always be sufficient.

Suppose the model predicts:

```text
Crystal A:
σ = 0.05

Crystal B:
σ = 0.20

Crystal C:
σ = 0.80
```

The model may correctly identify that Crystal C is more uncertain, but the absolute magnitudes may still be wrong.

For example:

```text
Predicted σ:
0.05 → actual typical error 0.10
0.20 → actual typical error 0.35
0.80 → actual typical error 0.90
```

A calibration procedure should ideally preserve the relative ordering while correcting the scale.

This is one reason calibration should be evaluated across uncertainty ranges rather than only through a single average metric.

---

# 27.18.20 Calibration by Standardized Residuals

For regression, a useful diagnostic is the standardized residual:

```text
rᵢ
=
(yᵢ - μᵢ)
/
σᵢ
```

where:

```text
yᵢ
```

is the observed value,

```text
μᵢ
```

is the predicted mean, and:

```text
σᵢ
```

is the predicted standard deviation.

If the predictive Gaussian model is well calibrated, the standardized residuals should approximately follow a standard normal distribution:

```text
r
~
N(0,1)
```

This provides a useful diagnostic.

---

## 27.18.21 Interpreting Standardized Residuals

Suppose the standardized residuals are:

```text
-2.1
-1.4
-0.8
-0.2
0.1
0.6
1.0
1.7
2.3
```

If the values are systematically much larger than expected, the model may be underestimating uncertainty.

For example:

```text
Predicted σ too small
        ↓
Standardized residuals too large
```

Conversely:

```text
Predicted σ too large
        ↓
Standardized residuals too concentrated
```

This provides another way to diagnose regression uncertainty calibration.

---

# 27.18.22 Gaussian NLL as a Calibration Diagnostic

For a Gaussian predictive model, the negative log-likelihood is:

```text
NLL
=
1/2
[
log(2πσ²)
+
(y-μ)²/σ²
]
```

The first term penalizes excessively large uncertainty.

The second term penalizes uncertainty that is too small relative to the observed error.

Therefore, NLL balances:

```text
Accuracy
+
Uncertainty magnitude
```

A model that predicts enormous uncertainty for every material does not automatically achieve a good NLL.

This makes NLL a useful metric for probabilistic regression.

---

# 27.18.23 Calibrating Deep Ensembles

Suppose a deep ensemble contains:

```text
M = 5
```

models.

Each model predicts:

```text
μ₁, μ₂, ..., μ₅
```

The ensemble mean is:

```text
μ_ensemble
=
1/M Σₘ μₘ
```

The spread among models provides an estimate of epistemic uncertainty.

However, the raw ensemble variance may not be calibrated.

A calibration factor can therefore be learned using a validation dataset.

The workflow is:

```text
Deep Ensemble
      ↓
Validation Predictions
      ↓
Ensemble Mean + Variance
      ↓
Calibration
      ↓
Calibrated Predictive Distribution
```

The test set should then be used only for final evaluation.

---

# 27.18.24 Calibrating Monte Carlo Dropout

Monte Carlo dropout generates multiple stochastic predictions:

```text
ŷ₁
ŷ₂
...
ŷₘ
```

The mean prediction is:

```text
μ
=
1/M Σₘ ŷₘ
```

and the sample variance can provide an uncertainty estimate.

Again, the raw variance should not automatically be interpreted as a calibrated standard deviation.

A calibration dataset can be used to evaluate:

```text
Predicted uncertainty
vs
Observed prediction errors
```

and determine whether the MC Dropout uncertainty requires rescaling.

---

# 27.18.25 Calibration of Gaussian Processes

Gaussian Processes naturally produce predictive distributions.

For example:

```text
Prediction:
μ(x)

Variance:
σ²(x)
```

This is one of the major advantages of Gaussian Processes.

However, even a probabilistic model with a mathematically defined predictive variance can be poorly calibrated if:

```text
Kernel is inappropriate
Noise model is incorrect
Training distribution differs from deployment
```

Therefore, GP uncertainty should also be evaluated empirically.

The workflow remains:

```text
GP predictive distribution
        ↓
Calibration dataset
        ↓
Coverage evaluation
        ↓
Test evaluation
```

---

# 27.18.26 Conformal Prediction

Conformal prediction provides a different approach to uncertainty calibration.

Instead of assuming a particular predictive distribution, conformal methods use calibration residuals to construct prediction intervals with a specified coverage under their statistical assumptions.

For regression, a simple residual-based procedure is:

```text
Train model
    ↓
Predict calibration set
    ↓
Calculate absolute residuals
    ↓
Determine residual quantile
    ↓
Construct prediction interval
```

For example:

```text
Residual:
rᵢ = |yᵢ - ŷᵢ|
```

A high quantile of the calibration residuals can define the interval width.

This approach will be examined in more detail in the next dedicated section on conformal prediction.

---

# 27.18.27 Calibration Method Selection

The appropriate calibration method depends on the model and dataset.

For neural-network classification:

```text
Temperature scaling
```

is often a strong baseline.

For flexible binary calibration:

```text
Isotonic regression
```

can be considered when enough calibration data are available.

For regression uncertainty:

```text
Variance scaling
Residual-based calibration
Conformal prediction
```

may be appropriate.

For ensemble models:

```text
Ensemble variance
+
Calibration
```

can provide a practical uncertainty pipeline.

The important principle is:

```text
Calibration method
must match
prediction type
and
uncertainty representation.
```

---

# 27.18.28 Calibration Method Comparison

| Method               | Primary Use                 | Flexibility | Data Requirement | Main Advantage                | Main Limitation                           |
| -------------------- | --------------------------- | ----------- | ---------------- | ----------------------------- | ----------------------------------------- |
| Temperature Scaling  | Classification              | Low         | Low–Moderate     | Simple and stable             | Limited mapping flexibility               |
| Platt-Style Scaling  | Binary Classification       | Moderate    | Low–Moderate     | Simple probabilistic mapping  | Mainly binary                             |
| Isotonic Regression  | Classification              | High        | Moderate–High    | Flexible monotonic mapping    | Can overfit                               |
| Variance Scaling     | Regression                  | Low         | Low–Moderate     | Simple uncertainty correction | May not correct complex miscalibration    |
| Residual Calibration | Regression                  | Moderate    | Moderate         | Directly uses observed errors | Depends on calibration distribution       |
| Conformal Prediction | Regression / Classification | High        | Moderate         | Coverage-oriented framework   | Coverage depends on assumptions and setup |

These methods should not be treated as interchangeable.

They solve related but distinct calibration problems.

---

# 27.18.29 A Practical Materials ML Calibration Pipeline

Consider a crystal-property prediction model.

The complete workflow can be:

```text
Crystal Dataset
       ↓
Train Model
       ↓
Calibration Dataset
       ↓
Raw Predictions
       ↓
Raw Uncertainty
       ↓
Calibration Method
       ↓
Calibrated Prediction
       ↓
Coverage / Reliability Evaluation
       ↓
Independent Test Set
```

For a classification problem:

```text
Crystal
   ↓
GNN
   ↓
Raw logits
   ↓
Temperature scaling
   ↓
Calibrated stability probability
```

For a regression problem:

```text
Crystal
   ↓
GNN
   ↓
Mean + uncertainty
   ↓
Variance / residual calibration
   ↓
Prediction interval
```

---

# 27.18.30 Practical Example: Calibrating a Crystal Stability Classifier

Suppose a trained model predicts:

```python
logits = model(
    X_calibration
)
```

The raw probabilities are:

```python
import torch

raw_probabilities = torch.softmax(
    logits,
    dim=1
)
```

Suppose the model is systematically overconfident.

Temperature scaling can be applied:

```python
T = torch.tensor(
    2.0
)

calibrated_probabilities = torch.softmax(
    logits / T,
    dim=1
)
```

The correct value of `T` should be learned from the calibration data rather than manually fixed.

After calibration, the researcher should evaluate:

```text
Reliability diagram
ECE
Brier score
Log loss
```

on an independent test set.

---

# 27.18.31 Practical Example: Regression Uncertainty Scaling

Suppose a model predicts:

```python
mean_prediction
predicted_std
```

A simple scaling factor can be introduced:

```python
scale = 1.25

calibrated_std = (
    scale
    * predicted_std
)
```

A 90% Gaussian interval can then be constructed:

```python
from scipy.stats import norm

z = norm.ppf(
    0.95
)

lower = (
    mean_prediction
    - z * calibrated_std
)

upper = (
    mean_prediction
    + z * calibrated_std
)
```

The resulting interval should then be evaluated on independent data.

The scaling factor is useful only if it was estimated from a calibration dataset.

---

# 27.18.32 Calibration Should Preserve Scientific Ranking When Possible

In many Materials Informatics applications, the researcher cares about ranking candidate materials.

For example:

```text
Candidate A → highest predicted property
Candidate B → second
Candidate C → third
```

A calibration method should ideally improve uncertainty interpretation without unnecessarily destroying useful ranking information.

For probability calibration, this often means preserving the ordering:

```text
p_A > p_B
```

while changing the numerical probabilities:

```text
0.99 → 0.90
0.92 → 0.84
```

The calibrated probabilities become more reliable while the model's ordering remains similar.

---

# 27.18.33 Calibration Does Not Create New Information

Calibration cannot create knowledge that the predictive model does not contain.

If a model has never seen a particular chemical system, calibration cannot automatically make its prediction scientifically reliable.

For example:

```text
Poor representation
+
No training examples
+
Severe distribution shift
```

cannot be solved simply by applying temperature scaling.

Calibration is therefore a correction of uncertainty interpretation, not a replacement for model knowledge.

This distinction is especially important for materials discovery.

---

# 27.18.34 Calibration Under Chemical Distribution Shift

Suppose a model is calibrated on:

```text
Oxide materials
```

but is then applied to:

```text
Sulfide materials
```

The calibration relationship may change.

For example:

```text
Oxide validation:
90% interval → 91% coverage

Sulfide evaluation:
90% interval → 63% coverage
```

This indicates that the original calibration does not transfer to the new chemical domain.

Therefore, calibration should be evaluated under the intended deployment distribution.

---

# 27.18.35 Calibration Under Structural Distribution Shift

The same problem occurs for crystal structures.

A model calibrated on:

```text
Cubic crystals
```

may behave differently on:

```text
Monoclinic crystals
```

or other structural families.

Therefore, materials-specific calibration studies should consider the type of shift that the model is expected to encounter.

Possible evaluation settings include:

```text
Random test split
Chemical-system holdout
Structural-family holdout
```

Each tests a different form of generalization.

---

# 27.18.36 Research-Grade Calibration Procedure

A strong uncertainty-calibration experiment should follow:

```text
1. Define predictive task
        ↓
2. Train model
        ↓
3. Reserve calibration dataset
        ↓
4. Generate raw predictions
        ↓
5. Measure raw calibration
        ↓
6. Select calibration method
        ↓
7. Fit calibration mapping
        ↓
8. Evaluate calibrated predictions
        ↓
9. Test on independent data
        ↓
10. Evaluate under relevant distribution shifts
```

The researcher should report both before and after calibration.

For example:

```text
Before calibration:
90% coverage = 68%

After calibration:
90% coverage = 89%
```

This demonstrates whether calibration actually improved the uncertainty estimate.

---

# 27.18.37 What Should Be Reported?

A Materials Informatics paper should report enough information for another researcher to reproduce the calibration procedure.

At minimum:

```text
Predictive model
Calibration dataset
Calibration method
Calibration objective
Number of calibration samples
Calibration metric
Test-set metric
Distribution-shift evaluation
```

For regression:

```text
Nominal coverage
Empirical coverage
Interval width
NLL
```

For classification:

```text
ECE
Brier score
Log loss
Reliability diagram
```

This makes the uncertainty analysis transparent.

---

# 27.18.38 Common Mistakes in Calibration

### Mistake 1: Calibrating on training data

This can produce misleadingly good calibration.

### Mistake 2: Calibrating on the final test set

Repeated test-set calibration compromises the independence of the final evaluation.

### Mistake 3: Using isotonic regression with very little data

The flexible mapping can overfit.

### Mistake 4: Assuming temperature scaling fixes all uncertainty problems

It mainly addresses systematic probability miscalibration in classification.

### Mistake 5: Ignoring distribution shift

A calibration mapping learned in one chemical domain may fail in another.

### Mistake 6: Evaluating only average calibration

A model may be well calibrated overall but badly calibrated in scientifically important regions.

### Mistake 7: Reporting calibrated probabilities without reporting the method

A probability such as `0.92` has little meaning without knowing how it was calibrated.

---

# 27.18.39 Key Takeaways

Calibration methods transform raw model outputs into uncertainty estimates or probabilities with improved statistical reliability.

The central methods discussed in this section are:

```text
Temperature Scaling
Platt-Style Calibration
Isotonic Regression
Variance Scaling
Residual Calibration
Conformal Prediction
```

The major principles are:

```text
Calibration requires independent calibration data.

Temperature scaling is simple and effective
for many neural classification models.

Isotonic regression provides a more flexible
monotonic calibration mapping.

Regression uncertainty can be calibrated using
residuals or uncertainty scaling.

Calibration cannot replace missing scientific knowledge.

Calibration should be evaluated under relevant
chemical and structural distributions.

The final test set should remain independent.

Calibration should be reported quantitatively.
```

The essential distinction is:

```text
Raw prediction
      ↓
Uncertainty estimate
      ↓
Calibration
      ↓
Validated uncertainty
```

A calibrated uncertainty estimate is therefore much more scientifically useful than an arbitrary confidence value.

The next section will focus specifically on **27.19 Conformal Prediction for Materials ML**, including prediction intervals, coverage guarantees, calibration sets, regression conformal prediction, classification conformal prediction, and practical materials-property examples.

## 27.19 Conformal Prediction for Materials Machine Learning

Conformal prediction provides a framework for constructing prediction sets or prediction intervals with a specified level of statistical coverage.

The central idea is different from simply asking a model to output a standard deviation.

A conventional predictive model produces:

```text
Input Crystal
      ↓
Prediction
      ↓
ŷ
```

Conformal prediction adds a calibration procedure:

```text
Training Data
      ↓
Predictive Model
      ↓
Calibration Data
      ↓
Prediction Errors
      ↓
Calibration Quantile
      ↓
Prediction Interval
```

The resulting interval is designed to achieve a specified empirical coverage under the assumptions of the conformal procedure.

For Materials Informatics, this is useful because a model can produce not only a predicted materials property but also an uncertainty interval around that prediction.

For example:

```text
Predicted band gap = 2.7 eV

90% conformal prediction interval:
[2.1 eV, 3.3 eV]
```

The interval is obtained from the model's behavior on calibration data rather than relying entirely on a Gaussian uncertainty assumption.

---

## 27.19.1 Why Conformal Prediction Is Useful

Many Materials Machine Learning models produce point predictions.

For example:

```text
Crystal A → 2.7 eV
Crystal B → 1.4 eV
Crystal C → 3.2 eV
```

These predictions do not indicate how reliable they are.

A researcher may instead want:

```text
Crystal A → 2.7 ± uncertainty
Crystal B → 1.4 ± uncertainty
Crystal C → 3.2 ± uncertainty
```

Conformal prediction provides a systematic way to construct such intervals.

The key advantage is that the method can be applied to many different predictive models.

For example:

```text
Random Forest
Gradient Boosting
Neural Network
GNN
CGCNN
MEGNet
Other regression models
```

The conformal procedure is therefore separated from the underlying prediction model.

Conceptually:

```text
Any suitable predictor
        ↓
Calibration residuals
        ↓
Conformal calibration
        ↓
Prediction interval
```

---

## 27.19.2 Point Prediction vs Prediction Interval

Suppose a model predicts formation energy:

```text
ŷ = -2.4 eV
```

A point prediction provides only:

```text
-2.4 eV
```

A conformal predictor may instead produce:

```text
[-2.9 eV, -1.9 eV]
```

The interval communicates uncertainty.

The difference is important.

A point prediction answers:

```text
What value does the model predict?
```

A prediction interval answers:

```text
What range of values should contain
the unknown true value at the chosen
coverage level?
```

The second statement is much more useful for uncertainty-aware scientific analysis.

---

## 27.19.3 The Basic Conformal Prediction Idea

The simplest regression conformal method uses a calibration dataset.

Suppose a model has already been trained.

For every calibration material:

```text
True value:
yᵢ

Prediction:
ŷᵢ
```

Calculate the absolute residual:

```text
rᵢ
=
|yᵢ - ŷᵢ|
```

For example:

```text
Material 1:
True = 2.8
Predicted = 2.6
Residual = 0.2

Material 2:
True = 1.5
Predicted = 1.8
Residual = 0.3

Material 3:
True = 3.1
Predicted = 2.7
Residual = 0.4
```

The collection of calibration residuals provides empirical information about model error.

---

## 27.19.4 Calibration Residuals

Suppose the calibration set produces:

```text
Residuals:

0.05
0.10
0.12
0.15
0.18
0.21
0.25
0.31
0.37
0.44
```

These values describe how far the model's predictions were from the actual values on the calibration materials.

A conformal method uses an appropriate upper quantile of these residuals.

For example, if a 90% prediction interval is desired, a high quantile of the calibration residual distribution is selected.

Call this value:

```text
q
```

The prediction interval for a new material can then be written as:

```text
[ŷ - q, ŷ + q]
```

---

## 27.19.5 Simple Split Conformal Prediction

The simplest implementation is **split conformal prediction**.

The dataset is divided into:

```text
Training Set
+
Calibration Set
+
Test Set
```

The training set is used to fit the predictive model.

The calibration set is used to estimate the distribution of prediction errors.

The test set is used to evaluate the final prediction intervals.

The workflow is:

```text
Complete Dataset
       ↓
┌───────────────┐
│ Training Set  │
└───────┬───────┘
        ↓
   Train Model
        │
        ↓
┌──────────────────┐
│ Calibration Set  │
└────────┬─────────┘
         ↓
 Calculate Residuals
         ↓
 Determine Quantile
         ↓
 New Material
         ↓
 Prediction ± q
         ↓
 Prediction Interval
```

This separation makes the method straightforward to implement.

---

## 27.19.6 Dataset Splitting

Suppose there are:

```text
10,000 crystals
```

A possible split is:

```text
Training:
7,000

Calibration:
1,500

Test:
1,500
```

The exact proportions depend on the dataset size and research objective.

The important principle is:

```text
Training data
≠
Calibration data
≠
Final test data
```

The calibration set must provide observations that are not used to fit the predictive model.

---

## 27.19.7 Training the Base Model

Suppose the target property is band gap.

A conventional regression model can be trained:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)
```

The model itself does not need to know anything about conformal prediction.

It simply learns:

```text
Materials Features
        ↓
Band Gap Prediction
```

After training, it is applied to the calibration set.

```python
calibration_predictions = model.predict(
    X_calibration
)
```

---

## 27.19.8 Calculating Calibration Residuals

The residuals are:

```python
import numpy as np

calibration_errors = np.abs(
    y_calibration
    - calibration_predictions
)
```

The resulting array contains:

```text
|y₁ - ŷ₁|
|y₂ - ŷ₂|
|y₃ - ŷ₃|
...
```

These errors form the calibration score distribution.

For a simple absolute-residual conformal method, larger residuals correspond to larger uncertainty.

---

## 27.19.9 Choosing the Conformal Quantile

Suppose the desired coverage is:

```text
90%
```

Then:

```text
α = 1 - 0.90
```

so:

```text
α = 0.10
```

The conformal procedure selects an appropriately adjusted upper quantile of the calibration scores.

In a simplified large-sample description, this corresponds approximately to the:

```text
90th percentile
```

of the calibration residuals.

The finite-sample conformal quantile should use the appropriate order-statistic adjustment rather than blindly calling a generic percentile routine.

The distinction matters when the calibration set is small.

---

## 27.19.10 Constructing the Prediction Interval

Once the conformal error threshold `q` has been obtained, a new material with prediction:

```text
ŷ_new
```

receives:

```text
Lower bound
=
ŷ_new - q
```

and:

```text
Upper bound
=
ŷ_new + q
```

Therefore:

```text
Conformal interval
=
[ŷ_new - q,
 ŷ_new + q]
```

For example, suppose:

```text
Prediction = 2.8 eV
q = 0.5 eV
```

Then:

```text
90% conformal interval:

[2.3 eV, 3.3 eV]
```

The width of the interval is determined by the calibration error distribution.

---

## 27.19.11 Complete Split-Conformal Example

A simple implementation is:

```python
import numpy as np

# Train the model
model.fit(
    X_train,
    y_train
)

# Predict calibration data
calibration_predictions = model.predict(
    X_calibration
)

# Calculate nonconformity scores
scores = np.abs(
    y_calibration
    - calibration_predictions
)

# Desired miscoverage
alpha = 0.10

# Finite-sample conformal quantile
n = len(scores)

q_level = np.ceil(
    (n + 1) * (1 - alpha)
) / n

q_level = min(
    q_level,
    1.0
)

q = np.quantile(
    scores,
    q_level,
    method="higher"
)

# Predict new materials
predictions = model.predict(
    X_test
)

# Construct intervals
lower = predictions - q
upper = predictions + q
```

The result is:

```text
Prediction
+
Lower bound
+
Upper bound
```

for every test material.

---

## 27.19.12 Evaluating Conformal Coverage

After constructing prediction intervals, compare them with the true test values.

```python
covered = (
    (y_test >= lower)
    &
    (y_test <= upper)
)

coverage = covered.mean()

print(
    "Coverage:",
    coverage
)
```

Suppose:

```text
Desired coverage = 90%
```

and the test set contains:

```text
1,000 materials
```

If:

```text
905
```

true values fall inside the intervals:

```text
Empirical coverage = 90.5%
```

This is close to the intended coverage.

---

## 27.19.13 Coverage Is a Central Evaluation

Conformal prediction is particularly concerned with coverage.

The researcher should report:

```text
Nominal coverage
Empirical coverage
```

For example:

```text
Nominal coverage:
90%

Empirical coverage:
89.7%
```

This is more informative than reporting only the mean absolute error of the underlying model.

The two quantities measure different properties:

```text
MAE
→ prediction accuracy

Coverage
→ uncertainty reliability
```

Both should be reported.

---

## 27.19.14 Interval Width

Coverage should not be considered alone.

Suppose two methods provide:

```text
Method A:
90% coverage
Average interval width = 0.8 eV

Method B:
94% coverage
Average interval width = 3.0 eV
```

Method B has higher coverage, but its intervals are much less informative.

Therefore, interval width should also be reported.

For an interval:

```text
[Lᵢ, Uᵢ]
```

the width is:

```text
Wᵢ
=
Uᵢ - Lᵢ
```

The average width is:

```text
Mean Width
=
1/N Σᵢ Wᵢ
```

A useful conformal model aims for:

```text
Desired coverage
+
Reasonably narrow intervals
```

---

## 27.19.15 Conformal Prediction and Model Accuracy

Conformal prediction does not automatically make the underlying model more accurate.

Suppose:

```text
Random Forest MAE = 0.30 eV
```

Conformal prediction does not necessarily reduce that MAE.

Instead, it adds:

```text
Prediction interval
```

around the model's prediction.

Therefore:

```text
Base model:
Prediction accuracy

Conformal layer:
Uncertainty quantification
```

The quality of the prediction interval still depends strongly on the quality and representativeness of the underlying model and calibration data.

---

## 27.19.16 The Nonconformity Score

The absolute residual is only one possible nonconformity score.

In conformal prediction, a **nonconformity score** measures how unusual a calibration example is relative to the predictive model.

For regression, a simple score is:

```text
sᵢ
=
|yᵢ - ŷᵢ|
```

Other scores can be used.

For example, if the model predicts its own uncertainty:

```text
μᵢ
σᵢ
```

a normalized score can be defined as:

```text
sᵢ
=
|yᵢ - μᵢ|
/
σᵢ
```

This allows the conformal method to account for input-dependent uncertainty.

---

## 27.19.17 Why Normalized Scores Can Be Useful

Suppose two crystals receive:

```text
Crystal A:
Predicted uncertainty = 0.1 eV

Crystal B:
Predicted uncertainty = 0.8 eV
```

An absolute error of:

```text
0.4 eV
```

means very different things for the two crystals.

For Crystal A:

```text
0.4 / 0.1 = 4
```

For Crystal B:

```text
0.4 / 0.8 = 0.5
```

Thus the normalized nonconformity score captures whether the error is large relative to the model's expected uncertainty.

This is useful for heteroscedastic materials prediction.

---

## 27.19.18 Adaptive Conformal Prediction

A fixed interval:

```text
ŷ ± q
```

gives approximately the same absolute width to every material.

This may be inefficient when prediction difficulty varies substantially across materials.

For example:

```text
Simple materials
→ small uncertainty

Complex materials
→ large uncertainty
```

An adaptive conformal method can use a model's predicted uncertainty to produce intervals whose widths depend on the input.

Conceptually:

```text
Crystal
   ↓
Model prediction
   +
Predicted uncertainty
   ↓
Conformal calibration
   ↓
Material-specific interval
```

This can produce narrower intervals for easy materials and wider intervals for difficult materials while maintaining the desired coverage under the relevant assumptions.

---

## 27.19.19 Example of Adaptive Intervals

Suppose a model predicts:

```text
Material A:
Prediction = 2.0 eV
Predicted σ = 0.1 eV

Material B:
Prediction = 2.0 eV
Predicted σ = 0.5 eV
```

A fixed conformal interval might produce:

```text
A:
[1.4, 2.6]

B:
[1.4, 2.6]
```

An adaptive procedure could instead produce:

```text
A:
[1.8, 2.2]

B:
[1.2, 2.8]
```

The second representation is more informative if the uncertainty model correctly captures differences in prediction difficulty.

---

## 27.19.20 Conformal Prediction for Crystal Graph Models

Conformal prediction can also be applied to graph neural networks.

Consider:

```text
Crystal Structure
       ↓
Pymatgen
       ↓
Crystal Graph
       ↓
GNN
       ↓
Property Prediction
       ↓
Conformal Calibration
       ↓
Prediction Interval
```

The GNN may predict:

```text
Band gap = 2.4 eV
```

The calibration set is then used to estimate the appropriate nonconformity scores.

A conformal interval might become:

```text
90% interval:
[1.9, 2.9] eV
```

The conformal procedure does not require the GNN itself to be probabilistic.

It can be applied to an ordinary point-prediction GNN.

---

## 27.19.21 Example with a PyTorch Model

Suppose a trained GNN provides:

```python
model.eval()

with torch.no_grad():

    predictions = model(
        calibration_graphs
    )
```

The predictions can be converted to NumPy:

```python
calibration_predictions = (
    predictions
    .cpu()
    .numpy()
)
```

The calibration scores can then be calculated:

```python
scores = np.abs(
    y_calibration
    - calibration_predictions
)
```

The rest of the conformal procedure is independent of the neural-network architecture.

This separation is useful because the same uncertainty framework can be applied to:

```text
MLP
CGCNN
MEGNet
ALIGNN
Other GNNs
```

provided that they produce suitable predictions.

---

## 27.19.22 Conformal Classification

Conformal prediction is not limited to regression.

For classification, the output is a **prediction set** rather than a numerical interval.

Suppose a model predicts probabilities for:

```text
Stable
Metastable
Unstable
```

For one crystal:

```text
Stable      = 0.80
Metastable  = 0.15
Unstable    = 0.05
```

A conventional classifier might return:

```text
Stable
```

A conformal classifier may instead return a set such as:

```text
{Stable}
```

for a high-confidence case, or:

```text
{Stable, Metastable}
```

for a more uncertain material.

The prediction set communicates uncertainty by allowing multiple plausible classes.

---

## 27.19.23 Classification Prediction Sets

For classification, the conformal output can be:

```text
Material A:
{Stable}

Material B:
{Stable, Metastable}

Material C:
{Stable, Metastable, Unstable}
```

The size of the prediction set provides useful information.

```text
Small set
→ stronger classification confidence

Large set
→ greater classification uncertainty
```

However, the set should be interpreted through the specified coverage guarantee rather than simply treating set size as a probability.

---

## 27.19.24 Classification Nonconformity Scores

One simple classification nonconformity score is:

```text
s(x,y)
=
1 - p_y
```

where:

```text
p_y
```

is the predicted probability assigned to the true class.

If the model assigns:

```text
p_true = 0.95
```

then:

```text
s = 0.05
```

The observation is highly conforming.

If:

```text
p_true = 0.20
```

then:

```text
s = 0.80
```

The observation is highly nonconforming.

The calibration set is used to determine the appropriate threshold for constructing prediction sets.

---

## 27.19.25 Conformal Classification Example

Suppose a crystal classifier produces:

```text
Stable      0.75
Metastable  0.20
Unstable    0.05
```

If the conformal threshold is sufficiently restrictive, the prediction set may be:

```text
{Stable}
```

For another crystal:

```text
Stable      0.52
Metastable  0.43
Unstable    0.05
```

the prediction set may become:

```text
{Stable, Metastable}
```

This reflects greater classification uncertainty.

---

## 27.19.26 Conformal Prediction and Materials Screening

Conformal prediction can be useful when screening large numbers of generated or database materials.

Suppose a model predicts formation energy for:

```text
100,000 candidate crystals
```

A point prediction alone produces:

```text
Candidate
+
Predicted energy
```

A conformal approach provides:

```text
Candidate
+
Predicted energy
+
Prediction interval
```

For example:

```text
Candidate A:
-2.8 eV
[-3.1, -2.5] eV

Candidate B:
-2.6 eV
[-4.0, -1.2] eV
```

Candidate A has a much narrower uncertainty interval.

This difference is scientifically relevant when deciding how strongly to trust the predicted values.

---

## 27.19.27 Conformal Prediction Does Not Guarantee Physical Stability

A crucial distinction must be maintained.

Suppose a model predicts:

```text
Formation energy = -3.0 eV
```

with a narrow conformal interval:

```text
[-3.2, -2.8] eV
```

This does not prove that the material is physically stable.

Conformal prediction addresses the statistical reliability of the predictive interval under its assumptions.

It does not replace:

```text
DFT calculations
Structural relaxation
Thermodynamic analysis
Experimental validation
```

Therefore:

```text
Conformal coverage
≠
Physical stability guarantee
```

The conformal interval should be interpreted as a statistical uncertainty statement about the model prediction.

---

## 27.19.28 Exchangeability Assumption

A central assumption in standard conformal prediction is **exchangeability**.

Informally, the calibration and future test examples should be statistically comparable in the sense required by the conformal construction.

For independently and identically distributed data, this condition is naturally satisfied.

In Materials Informatics, however, this assumption requires careful consideration.

For example:

```text
Training:
Common oxide crystals

Calibration:
Similar oxide crystals

Test:
Similar oxide crystals
```

is closer to the standard setting than:

```text
Training:
Oxides

Calibration:
Oxides

Test:
Previously unseen sulfides
```

The second situation introduces chemical distribution shift.

---

## 27.19.29 Why Exchangeability Matters

Suppose calibration errors are:

```text
Typical error = 0.2 eV
```

but the model is applied to a new chemical family where:

```text
Typical error = 0.8 eV
```

A conformal interval calibrated using the first distribution may be too narrow for the second.

Therefore:

```text
Calibration distribution
        ↓
should represent
        ↓
future prediction distribution
```

when the standard conformal coverage interpretation is intended.

This is one of the most important limitations to communicate in materials research.

---

## 27.19.30 Conformal Prediction Under Distribution Shift

Standard conformal prediction is not automatically immune to distribution shift.

If the test distribution differs substantially from the calibration distribution, nominal coverage may not hold.

For example:

```text
Calibration:
Common chemical systems

Test:
Rare chemical systems
```

can lead to:

```text
Nominal coverage = 90%

Observed coverage = 62%
```

This does not mean that conformal prediction itself has failed mathematically.

It means that the conditions required for the intended coverage interpretation are not satisfied.

Therefore, Materials Informatics studies should explicitly evaluate whether the calibration and deployment populations are scientifically comparable.

---

## 27.19.31 Grouped Materials Calibration

Materials datasets often contain groups:

```text
Chemical systems
Crystal families
Space groups
Composition families
Structure prototypes
```

If these groups are strongly correlated, random splitting may place very similar materials in both calibration and test sets.

This can make calibration appear stronger than it really is for genuinely novel materials.

A more challenging evaluation may hold out groups.

For example:

```text
Training:
Oxide families A, B, C

Calibration:
Oxide family D

Test:
Oxide family E
```

or:

```text
Training:
Structural families A–D

Calibration:
Family E

Test:
Family F
```

The appropriate strategy depends on the scientific question.

---

## 27.19.32 Coverage by Materials Subgroup

Overall coverage can hide subgroup failures.

Suppose the total test coverage is:

```text
90%
```

but subgroup coverage is:

```text
Cubic:
96%

Tetragonal:
91%

Monoclinic:
72%
```

The overall value may look acceptable while the model is poorly calibrated for monoclinic materials.

Therefore, researchers should consider subgroup-specific coverage where scientifically meaningful.

Possible groups include:

```text
Elemental system
Crystal system
Space group
Composition family
Property range
Structural complexity
```

This is particularly important when uncertainty is intended to support materials screening.

---

## 27.19.33 Conformal Prediction with Heterogeneous Materials

Materials datasets can contain very different levels of prediction difficulty.

For example:

```text
Simple binary compounds
Complex multicomponent compounds
Highly distorted structures
Rare compositions
```

A single fixed conformal interval may not represent this variation efficiently.

Adaptive conformal approaches can use model-derived difficulty estimates to construct more informative intervals.

The conceptual workflow is:

```text
Crystal
   ↓
Representation
   ↓
Predictive Model
   ↓
Prediction + Difficulty Estimate
   ↓
Conformal Calibration
   ↓
Adaptive Prediction Interval
```

The central goal remains:

```text
Reliable coverage
+
Useful interval width
```

---

## 27.19.34 Comparing Standard and Adaptive Conformal Intervals

Consider two materials:

```text
Material A:
Easy prediction

Material B:
Difficult prediction
```

A standard conformal method might produce:

```text
A:
[1.5, 2.5]

B:
[2.0, 3.0]
```

with the same interval width.

An adaptive method may produce:

```text
A:
[1.8, 2.2]

B:
[1.5, 3.5]
```

This can provide more useful uncertainty information if the difficulty estimate is informative and the resulting method remains appropriately calibrated.

---

## 27.19.35 Practical Evaluation Table

A conformal Materials ML study can report:

| Quantity              |  Result |
| --------------------- | ------: |
| Model                 |     GNN |
| Calibration samples   |   1,500 |
| Nominal coverage      |     90% |
| Empirical coverage    |   89.4% |
| Mean interval width   | 0.82 eV |
| Median interval width | 0.71 eV |
| Test MAE              | 0.31 eV |
| Test RMSE             | 0.48 eV |

The exact values above are illustrative.

The important structure is that both predictive accuracy and uncertainty quality are reported.

---

## 27.19.36 Conformal Prediction Workflow for Materials Research

A complete research workflow can be summarized as:

```text
Materials Dataset
       ↓
Train / Calibration / Test Split
       ↓
Train Predictive Model
       ↓
Generate Calibration Predictions
       ↓
Calculate Nonconformity Scores
       ↓
Estimate Conformal Quantile
       ↓
Predict New Materials
       ↓
Construct Prediction Intervals
       ↓
Evaluate Coverage
       ↓
Evaluate Interval Width
       ↓
Evaluate Subgroups
       ↓
Evaluate Distribution Shift
```

For a crystal GNN:

```text
Crystal Structure
       ↓
Pymatgen
       ↓
Crystal Graph
       ↓
GNN
       ↓
Property Prediction
       ↓
Conformal Calibration
       ↓
Prediction Interval
```

---

## 27.19.37 Advantages of Conformal Prediction

Conformal prediction has several attractive properties for Materials Machine Learning.

### Model independence

It can be applied to many predictive models.

### Simple calibration

A separate calibration dataset can be used.

### Distribution-free formulation

The basic framework does not require the predictive errors themselves to follow a Gaussian distribution.

### Explicit coverage target

The researcher can specify a desired coverage level.

### Flexible integration

It can be combined with classical ML and deep learning.

These properties make conformal prediction a useful component of uncertainty-aware Materials Informatics workflows.

---

## 27.19.38 Limitations of Conformal Prediction

Conformal prediction also has important limitations.

### 1. Dependence on calibration data

Poor calibration data can produce poor intervals.

### 2. Distribution shift

Standard coverage guarantees rely on assumptions that may fail under chemical or structural distribution shift.

### 3. Interval width

High coverage can require wide intervals.

### 4. Model dependence in practice

Although the framework is model-agnostic, poor base predictions can still lead to uninformative intervals.

### 5. Computational considerations

Very large materials datasets may require careful implementation when calibration scores or repeated predictions become expensive.

### 6. Physical interpretation

Statistical coverage should not be confused with physical certainty.

---

## 27.19.39 Conformal Prediction vs Gaussian Uncertainty

Consider two uncertainty approaches.

### Gaussian predictive uncertainty

```text
Model
 ↓
Mean + standard deviation
 ↓
Gaussian interval
```

This relies on the interpretation of the predicted distribution.

### Conformal prediction

```text
Model
 ↓
Calibration residuals
 ↓
Conformal quantile
 ↓
Prediction interval
```

The conformal approach does not require the residual distribution itself to be Gaussian.

Therefore, the two methods answer uncertainty questions differently.

They can also be combined.

For example:

```text
Neural Network
      ↓
Mean + predicted σ
      ↓
Normalized residual scores
      ↓
Conformal calibration
      ↓
Calibrated adaptive interval
```

This can combine model-based uncertainty information with empirical calibration.

---

## 27.19.40 Conformal Prediction and Scientific Trust

A prediction such as:

```text
Band gap = 2.8 eV
```

is incomplete from an uncertainty perspective.

A stronger result is:

```text
Band gap = 2.8 eV

90% conformal prediction interval:
[2.3, 3.3] eV

Empirical coverage on comparable
held-out materials:
approximately 90%
```

This does not make the prediction physically certain.

It provides a quantitatively evaluated measure of predictive uncertainty.

The distinction is important:

```text
Statistical coverage
        ≠
Physical truth
```

---

## 27.19.41 Key Takeaways

Conformal prediction provides a systematic framework for converting model predictions into statistically calibrated prediction sets or intervals.

The central workflow is:

```text
Train Model
    ↓
Calibration Data
    ↓
Nonconformity Scores
    ↓
Conformal Quantile
    ↓
Prediction Interval / Set
    ↓
Coverage Evaluation
```

For regression:

```text
Prediction
+
Conformal interval
```

For classification:

```text
Prediction
+
Conformal prediction set
```

The most important concepts are:

```text
Calibration set
Nonconformity score
Split conformal prediction
Coverage
Interval width
Adaptive conformal prediction
Classification prediction sets
Exchangeability
Distribution shift
Subgroup coverage
```

For Materials Informatics, conformal prediction is particularly useful when uncertainty estimates must be evaluated quantitatively rather than simply reported as arbitrary `±` values.

The central principle is:

```text
Prediction
+
Calibration Data
+
Conformal Procedure
=
Statistically Evaluated Uncertainty
```

However, conformal coverage should never be interpreted as a guarantee of physical stability, synthesizability, or experimental success.

A conformal interval describes the statistical behavior of the predictive system under the conditions and assumptions of the conformal procedure.

The next section will examine **27.20 Uncertainty Evaluation Metrics**, focusing on quantitative measures such as negative log-likelihood, calibration error, coverage, interval width, sharpness, and proper scoring rules.

## 27.20 Uncertainty Evaluation Metrics

Estimating uncertainty is only the first step.

A model can produce:

```text
Prediction
+
Uncertainty estimate
```

but this does not automatically mean that the uncertainty estimate is useful.

An uncertainty model must itself be evaluated.

For example, suppose a Materials ML model predicts:

```text
Band gap = 2.5 eV
Uncertainty = ±0.2 eV
```

The important question is not simply whether the interval looks reasonable.

The researcher must ask:

```text
Are the actual errors consistent with
the reported uncertainty?
```

If the model repeatedly makes errors of approximately:

```text
0.8 eV
```

while reporting:

```text
±0.2 eV
```

then the uncertainty estimate is clearly unreliable.

Therefore, uncertainty evaluation should be treated as a separate evaluation problem:

```text
Predictive Model
       ↓
Prediction
       +
Uncertainty
       ↓
Uncertainty Evaluation
       ↓
Calibration
Reliability
Sharpness
Coverage
```

A research-grade Materials Informatics study should therefore report both:

```text
Prediction quality
```

and

```text
Uncertainty quality
```

---

## 27.20.1 Prediction Error vs Uncertainty

Prediction error and predictive uncertainty are related but fundamentally different.

For a material with true property:

```text
y
```

and prediction:

```text
ŷ
```

the prediction error is:

```text
e = y - ŷ
```

The model's uncertainty estimate may be represented by:

```text
σ
```

These quantities answer different questions.

Prediction error answers:

```text
How wrong was the model?
```

Uncertainty answers:

```text
How uncertain did the model believe
the prediction was?
```

For example:

```text
Material A

True band gap = 2.8 eV
Prediction = 2.7 eV
Uncertainty = 0.1 eV
```

The absolute error is:

```text
|2.8 - 2.7|
=
0.1 eV
```

The uncertainty estimate is also:

```text
0.1 eV
```

This is a reasonably consistent prediction.

Now consider:

```text
Material B

True band gap = 3.4 eV
Prediction = 2.6 eV
Uncertainty = 0.1 eV
```

The error is:

```text
0.8 eV
```

but the model reported only:

```text
0.1 eV
```

This is an example of **overconfident uncertainty estimation**.

---

## 27.20.2 Why Uncertainty Metrics Are Necessary

Suppose two models have identical prediction accuracy:

```text
Model A:
MAE = 0.30 eV

Model B:
MAE = 0.30 eV
```

Their uncertainty estimates may nevertheless be very different.

For example:

```text
Model A:
Accurate predictions
+
Reliable uncertainty

Model B:
Accurate predictions
+
Poor uncertainty
```

If the model is being used for materials screening, Model A is generally more useful.

Therefore, uncertainty-aware evaluation should not stop at:

```text
MAE
RMSE
R²
```

Instead, the study should also evaluate:

```text
Calibration
Coverage
Interval width
Sharpness
Likelihood
Uncertainty-error relationship
```

---

## 27.20.3 Negative Log-Likelihood

Negative log-likelihood, commonly abbreviated as **NLL**, evaluates the probability that a predictive distribution assigns to the observed target.

Suppose a model predicts a Gaussian distribution:

```text
y ~ N(μ, σ²)
```

where:

```text
μ
```

is the predicted mean and:

```text
σ
```

is the predicted standard deviation.

The Gaussian negative log-likelihood for one observation is:

```text
NLL
=
1/2 log(2πσ²)
+
(y - μ)² / (2σ²)
```

The first term depends on the predicted uncertainty.

The second term measures how far the observation lies from the predicted mean relative to the predicted variance.

Therefore, NLL evaluates both:

```text
Prediction accuracy
```

and

```text
Uncertainty magnitude
```

simultaneously.

---

## 27.20.4 Interpreting Negative Log-Likelihood

Consider two predictions for the same material.

### Model A

```text
μ = 2.5 eV
σ = 0.1 eV
```

True value:

```text
y = 2.6 eV
```

The error is:

```text
0.1 eV
```

which is approximately one predicted standard deviation.

This is reasonably consistent.

Now consider:

### Model B

```text
μ = 2.5 eV
σ = 0.01 eV
```

with the same:

```text
y = 2.6 eV
```

Now the error is:

```text
10 × σ
```

The model was extremely confident but substantially wrong.

The NLL therefore penalizes this combination strongly.

This is an important advantage over ordinary MAE.

MAE evaluates:

```text
|y - μ|
```

whereas NLL evaluates the error relative to the predicted probability distribution.

---

## 27.20.5 NLL and Overconfidence

An uncertainty model can fail in two opposite directions.

### Overconfident model

```text
Small σ
+
Large prediction error
```

This is dangerous because the model claims high confidence when it is wrong.

### Underconfident model

```text
Very large σ
+
Small prediction error
```

This model may avoid severe overconfidence, but its predictions become less informative.

NLL balances these effects.

A model cannot obtain a good probabilistic score simply by making its uncertainty arbitrarily large.

Increasing:

```text
σ
```

reduces the penalty associated with large residuals, but eventually the logarithmic uncertainty term becomes unfavorable.

Therefore, NLL rewards uncertainty estimates that are both:

```text
Accurate
+
Appropriately scaled
```

---

## 27.20.6 Computing NLL in Python

For a Gaussian predictive model:

```python
import numpy as np

def gaussian_nll(
    y_true,
    mean,
    std
):

    variance = std ** 2

    nll = (
        0.5 * np.log(
            2 * np.pi * variance
        )
        +
        (y_true - mean) ** 2
        /
        (2 * variance)
    )

    return np.mean(nll)
```

Example:

```python
y_true = np.array([
    2.8,
    1.5,
    3.1
])

mean = np.array([
    2.7,
    1.7,
    3.0
])

std = np.array([
    0.2,
    0.3,
    0.2
])

score = gaussian_nll(
    y_true,
    mean,
    std
)

print(
    "NLL:",
    score
)
```

Lower NLL is generally preferable when comparing probabilistic regression models evaluated on the same dataset and target definition.

---

## 27.20.7 Calibration Error

Calibration asks whether predicted uncertainty corresponds to observed errors.

Suppose a model reports:

```text
90% prediction intervals
```

If the model is well calibrated, approximately:

```text
90%
```

of comparable future observations should fall inside those intervals under the relevant assumptions.

Therefore:

```text
Predicted confidence
        ↓
Observed frequency
```

should agree.

A mismatch indicates poor calibration.

---

## 27.20.8 Coverage Probability

For prediction intervals, one of the most direct uncertainty metrics is **coverage**.

Suppose each test material has:

```text
Lower bound = Lᵢ
Upper bound = Uᵢ
```

and the true target is:

```text
yᵢ
```

Define:

```text
Iᵢ =
1
```

if:

```text
Lᵢ ≤ yᵢ ≤ Uᵢ
```

and:

```text
Iᵢ =
0
```

otherwise.

Empirical coverage is:

```text
Coverage
=
1/N Σᵢ Iᵢ
```

For example, with:

```text
N = 1000
```

and:

```text
900
```

observations inside their intervals:

```text
Coverage = 900 / 1000
         = 0.90
```

or:

```text
90%
```

---

## 27.20.9 Nominal vs Empirical Coverage

Two quantities should always be distinguished.

### Nominal coverage

The target specified by the uncertainty method.

For example:

```text
90%
```

### Empirical coverage

The fraction actually observed on held-out data.

For example:

```text
87.4%
```

The comparison is:

```text
Nominal:
90%

Observed:
87.4%
```

The difference indicates a calibration problem.

A useful uncertainty evaluation should therefore report both.

---

## 27.20.10 Coverage Error

A simple calibration measure is:

```text
Coverage Error
=
|Empirical Coverage
-
Nominal Coverage|
```

Suppose:

```text
Nominal coverage = 90%
Empirical coverage = 87%
```

Then:

```text
Coverage error
=
|87 - 90|
=
3 percentage points
```

This provides a simple summary of the deviation from the target coverage.

However, one coverage level alone does not fully describe an uncertainty model.

A model should ideally be evaluated at several confidence levels.

---

## 27.20.11 Coverage Across Multiple Confidence Levels

Suppose a model produces intervals at:

```text
50%
80%
90%
95%
99%
```

The researcher can calculate empirical coverage at every level.

For example:

| Nominal Coverage | Empirical Coverage |
| ---------------: | -----------------: |
|              50% |                48% |
|              80% |                79% |
|              90% |                88% |
|              95% |                94% |
|              99% |                97% |

These values can be compared directly.

A well-calibrated uncertainty model should produce empirical coverage close to the corresponding nominal coverage across the range.

---

## 27.20.12 Coverage Curve

The relationship can be visualized as:

```text
Empirical
Coverage
   |
1.0|                         /
   |                       /
   |                    /
   |                 /
   |              /
   |           /
   |        /
   |     /
   |  /
0.0+------------------------------ Nominal
   0.0                          1.0
```

The ideal calibration curve follows:

```text
y = x
```

where:

```text
x = nominal coverage
y = empirical coverage
```

If the observed curve lies below the diagonal, the intervals tend to contain fewer observations than expected.

If it lies above the diagonal, the intervals tend to contain more observations than expected.

---

## 27.20.13 Interval Width

Coverage alone is insufficient.

Consider two uncertainty models.

```text
Model A:
90% coverage
Average width = 0.5 eV

Model B:
99% coverage
Average width = 5.0 eV
```

Model B may appear more reliable because it covers more observations.

However, extremely wide intervals are not useful for precise scientific screening.

Therefore, interval width should be evaluated together with coverage.

For each prediction interval:

```text
Wᵢ = Uᵢ - Lᵢ
```

The mean interval width is:

```text
Mean Width
=
1/N Σᵢ(Uᵢ - Lᵢ)
```

---

## 27.20.14 Sharpness

**Sharpness** describes how concentrated the predictive distributions or intervals are.

A sharper uncertainty model generally produces narrower distributions while maintaining appropriate calibration.

For example:

```text
Model A:
[2.3, 2.7]

Model B:
[1.5, 3.5]
```

Both might achieve the same coverage in a particular dataset.

Model A is sharper because its interval is narrower.

However:

```text
Sharpness alone
≠
Good calibration
```

A model could produce extremely narrow intervals and be badly overconfident.

Therefore, sharpness should always be interpreted together with coverage or another calibration metric.

---

## 27.20.15 The Calibration–Sharpness Trade-Off

There is an important relationship between:

```text
Coverage
```

and:

```text
Interval width
```

If the model makes intervals wider:

```text
↓
Coverage generally increases
```

but:

```text
↓
Scientific informativeness decreases
```

If the model makes intervals narrower:

```text
↓
Scientific informativeness increases
```

but:

```text
↓
Coverage may decrease
```

A useful uncertainty model seeks a good balance:

```text
Reliable Coverage
        +
Narrow Intervals
```

This is why reporting coverage without interval width can be misleading.

---

## 27.20.16 Mean Absolute Error vs Uncertainty Quality

Consider two models:

| Model |     MAE | 90% Coverage | Mean Width |
| ----- | ------: | -----------: | ---------: |
| A     | 0.30 eV |          89% |    0.80 eV |
| B     | 0.30 eV |          89% |    1.80 eV |

Both models have identical:

```text
MAE
```

and:

```text
Coverage
```

but Model A produces narrower intervals.

Therefore, Model A provides more informative uncertainty estimates.

This illustrates why uncertainty evaluation requires multiple metrics.

---

## 27.20.17 Prediction Interval Coverage Probability

A commonly used metric is **Prediction Interval Coverage Probability**, or PICP.

For intervals:

```text
[Lᵢ, Uᵢ]
```

PICP is:

```text
PICP
=
1/N Σᵢ
I(Lᵢ ≤ yᵢ ≤ Uᵢ)
```

This is equivalent to empirical coverage.

For a nominal 95% interval:

```text
PICP ≈ 0.95
```

would indicate appropriate empirical coverage on a representative evaluation dataset.

---

## 27.20.18 Mean Prediction Interval Width

A corresponding metric is **Mean Prediction Interval Width**, or MPIW.

It is:

```text
MPIW
=
1/N Σᵢ(Uᵢ - Lᵢ)
```

The two metrics can therefore be considered together:

```text
PICP
+
MPIW
```

For example:

```text
Model A:
PICP = 0.91
MPIW = 0.70 eV

Model B:
PICP = 0.91
MPIW = 1.60 eV
```

Both models have similar coverage, but Model A has sharper intervals.

---

## 27.20.19 Uncertainty–Error Correlation

A useful diagnostic is to examine whether uncertainty increases when the model makes larger errors.

For each test material, calculate:

```text
Absolute Error
```

and:

```text
Predicted Uncertainty
```

Then compare them.

For example:

```text
Material | Error | Uncertainty
--------------------------------
A        | 0.10  | 0.08
B        | 0.20  | 0.15
C        | 0.50  | 0.40
D        | 0.80  | 0.70
```

A useful uncertainty model should generally assign greater uncertainty to difficult predictions.

This relationship can be examined using correlation measures.

For example:

```python
from scipy.stats import spearmanr

correlation, p_value = spearmanr(
    absolute_errors,
    predicted_uncertainties
)

print(
    "Spearman correlation:",
    correlation
)
```

A positive relationship indicates that higher predicted uncertainty tends to correspond to larger observed errors.

---

## 27.20.20 Why Correlation Alone Is Insufficient

A strong correlation between:

```text
uncertainty
```

and:

```text
error
```

does not guarantee calibration.

For example:

```text
Actual error:
0.1, 0.2, 0.3, 0.4

Predicted uncertainty:
1.0, 2.0, 3.0, 4.0
```

The relationship is perfectly monotonic.

However, the uncertainty estimates are much larger than the actual errors.

Therefore:

```text
Correlation
```

measures ranking or association, while:

```text
Calibration
```

measures whether the uncertainty magnitude is quantitatively appropriate.

Both concepts should be distinguished.

---

## 27.20.21 Reliability Diagrams for Classification

For classification models, uncertainty is often represented by predicted probabilities.

Suppose a model predicts that several crystals are stable with probabilities:

```text
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95
```

These predictions can be grouped into probability bins.

For each bin, calculate:

```text
Average predicted probability
```

and:

```text
Observed frequency of correct predictions
```

For example:

| Predicted Confidence | Observed Accuracy |
| -------------------: | ----------------: |
|                 0.55 |              0.58 |
|                 0.65 |              0.61 |
|                 0.75 |              0.72 |
|                 0.85 |              0.82 |
|                 0.95 |              0.91 |

The closer these values are to one another, the better calibrated the probabilities are.

---

## 27.20.22 Expected Calibration Error

A common classification calibration metric is **Expected Calibration Error**, or ECE.

Suppose predictions are divided into `M` confidence bins.

For bin `m`:

```text
accuracy_m
```

is the observed accuracy, and:

```text
confidence_m
```

is the average predicted confidence.

Then:

```text
ECE
=
Σₘ
(nₘ / N)
|
accuracyₘ - confidenceₘ
|
```

where:

```text
nₘ
```

is the number of predictions in bin `m`.

Lower ECE indicates better calibration.

For example:

```text
ECE = 0.02
```

is generally better calibrated than:

```text
ECE = 0.15
```

when calculated under the same binning and evaluation procedure.

---

## 27.20.23 Calibration Error in Materials Classification

Consider a model classifying crystals into:

```text
Stable
Unstable
```

Suppose the model frequently outputs:

```text
90% probability of stability
```

but only:

```text
70%
```

of those crystals are actually stable.

The model is overconfident.

Its:

```text
Predicted confidence = 90%
Observed frequency = 70%
```

difference indicates poor calibration.

This matters because a researcher might otherwise interpret:

```text
90%
```

as strong scientific confidence.

Probability calibration is therefore an important component of uncertainty evaluation.

---

## 27.20.24 Brier Score

For probabilistic classification, another useful metric is the **Brier score**.

For binary classification:

```text
Brier Score
=
1/N Σᵢ(pᵢ - yᵢ)²
```

where:

```text
pᵢ
```

is the predicted probability of the positive class and:

```text
yᵢ
```

is the actual binary outcome.

Lower Brier score indicates better probabilistic predictions.

Unlike simple accuracy, the Brier score evaluates the quality of the predicted probabilities themselves.

---

## 27.20.25 Proper Scoring Rules

A **proper scoring rule** rewards a probabilistic model for assigning appropriate probabilities to outcomes.

Examples include:

```text
Negative Log-Likelihood
Brier Score
```

The important principle is that a model should not improve its score simply by becoming artificially uncertain or artificially confident.

Proper scoring rules therefore provide useful tools for evaluating probabilistic predictions.

For Materials Informatics, this is important whenever the model outputs:

```text
Probability distributions
Prediction probabilities
Predictive variance
```

rather than only point predictions.

---

## 27.20.26 Calibration of Regression Distributions

For probabilistic regression, calibration can also be evaluated through standardized residuals.

Suppose the model predicts:

```text
μᵢ
```

and:

```text
σᵢ
```

for each material.

The standardized residual is:

```text
zᵢ
=
(yᵢ - μᵢ)
/
σᵢ
```

If the Gaussian uncertainty model is appropriate and calibrated, the standardized residuals should approximately follow:

```text
N(0,1)
```

This provides a diagnostic.

For example:

```text
Mean(z) ≈ 0
Variance(z) ≈ 1
```

would be broadly consistent with a well-scaled Gaussian predictive uncertainty model.

---

## 27.20.27 Standardized Residual Histogram

The standardized residuals can be visualized as a histogram.

Conceptually:

```text
Frequency
   |
   |            ███
   |         █████████
   |       █████████████
   |     █████████████████
   |       █████████████
   |         █████████
   |            ███
   +--------------------------> z
             0
```

Strong deviations from the expected distribution can reveal problems.

For example:

```text
Very wide distribution
```

may indicate underestimated uncertainty.

A distribution concentrated too tightly around zero may indicate overestimated uncertainty.

Strong asymmetry may indicate that the Gaussian assumption is inappropriate.

---

## 27.20.28 Empirical Coverage of Standard-Deviation Intervals

Another simple diagnostic is to examine how many observations fall within:

```text
μ ± σ
```

```text
μ ± 2σ
```

```text
μ ± 3σ
```

For a correctly calibrated Gaussian predictive distribution, the expected coverage values are approximately:

```text
±1σ
→ 68.3%

±2σ
→ 95.4%

±3σ
→ 99.7%
```

These values are consequences of the Gaussian assumption.

If the observed coverage differs strongly, the uncertainty model may be miscalibrated or the predictive distribution may not be approximately Gaussian.

---

## 27.20.29 Coverage–Width Evaluation

For interval-based uncertainty, a practical evaluation should report at least:

```text
Nominal Coverage
Empirical Coverage
Coverage Error
Mean Interval Width
Median Interval Width
```

For example:

```text
90% interval

Nominal coverage:
90%

Observed coverage:
89.2%

Coverage error:
0.8 percentage points

Mean width:
0.76 eV

Median width:
0.68 eV
```

This provides a much clearer picture than simply reporting:

```text
Uncertainty = ±0.38 eV
```

---

## 27.20.30 Uncertainty Evaluation Across Materials Space

An uncertainty model can appear well calibrated globally while failing in particular regions of materials space.

For example:

```text
Chemical Space

Region A:
Low uncertainty
Good coverage

Region B:
Moderate uncertainty
Good coverage

Region C:
High uncertainty
Poor coverage
```

Therefore, uncertainty should sometimes be evaluated as a function of:

```text
Composition
Crystal structure
Chemical family
Property range
Structural complexity
```

This can reveal regions where the model is unreliable.

---

## 27.20.31 Uncertainty vs Training Density

A useful diagnostic is to compare uncertainty against the density of similar training examples.

Conceptually:

```text
Many similar training materials
        ↓
Lower epistemic uncertainty

Few similar training materials
        ↓
Higher epistemic uncertainty
```

A researcher can therefore examine:

```text
Training-data density
        vs
Predicted uncertainty
```

If uncertainty systematically increases in sparsely populated regions, this provides evidence that the uncertainty estimate captures some aspect of model unfamiliarity.

However, this relationship is diagnostic rather than a universal law.

---

## 27.20.32 Uncertainty Evaluation for Crystal GNNs

For crystal graph neural networks, uncertainty can be evaluated at the graph level.

Suppose:

```text
Crystal Graph
      ↓
GNN Ensemble
      ↓
Predictions:
2.4
2.6
2.5
2.7
2.3 eV
```

The ensemble mean is:

```text
2.5 eV
```

and the spread provides an uncertainty measure.

The researcher should then evaluate whether:

```text
High ensemble disagreement
```

corresponds to:

```text
Large prediction errors
```

and whether the resulting intervals have appropriate coverage.

The same evaluation principles used for classical ML apply to GNNs.

---

## 27.20.33 Research-Level Uncertainty Evaluation

A research-grade uncertainty study should ideally contain several levels of evaluation.

### Level 1 — Prediction accuracy

```text
MAE
RMSE
R²
```

### Level 2 — Uncertainty calibration

```text
Coverage
Calibration error
Reliability
```

### Level 3 — Uncertainty sharpness

```text
Interval width
Predictive variance
```

### Level 4 — Probabilistic quality

```text
NLL
Brier score
Other proper scoring rules
```

### Level 5 — Scientific subgroup analysis

```text
Chemical families
Crystal systems
Structural families
Property ranges
```

This provides a much stronger assessment than a single uncertainty metric.

---

## 27.20.34 Example Evaluation Report

A Materials ML uncertainty experiment might report:

```text
Model:
Crystal GNN

Target:
Band gap

Test MAE:
0.29 eV

Test RMSE:
0.47 eV

90% interval coverage:
89.6%

Coverage error:
0.4 percentage points

Mean interval width:
0.81 eV

Median interval width:
0.73 eV

NLL:
0.42

Uncertainty-error Spearman correlation:
0.61
```

These values are illustrative.

The important point is that the report separates:

```text
Accuracy
Calibration
Sharpness
Probabilistic quality
Uncertainty-error relationship
```

---

## 27.20.35 Comparing Two Uncertainty Methods

Suppose two methods are evaluated:

| Metric                        | Deep Ensemble | MC Dropout |
| ----------------------------- | ------------: | ---------: |
| MAE                           |       0.29 eV |    0.31 eV |
| 90% Coverage                  |         90.2% |      86.7% |
| Mean Width                    |       0.78 eV |    0.61 eV |
| NLL                           |          0.41 |       0.53 |
| Error–Uncertainty Correlation |          0.64 |       0.48 |

The interpretation should not simply be:

```text
Deep Ensemble is better.
```

Instead:

```text
Deep Ensemble:
Better calibration
Better probabilistic score
Stronger uncertainty-error relationship

MC Dropout:
Narrower intervals
But lower coverage
```

The narrower intervals of MC Dropout do not automatically make it superior.

They may indicate underestimation of uncertainty.

---

## 27.20.36 Why Multiple Metrics Should Be Reported

No single uncertainty metric captures every desirable property.

For example:

```text
NLL
```

evaluates probabilistic predictions.

```text
Coverage
```

evaluates interval reliability.

```text
Interval width
```

evaluates sharpness.

```text
ECE
```

evaluates classification calibration.

```text
Brier score
```

evaluates probabilistic classification.

Therefore, a robust uncertainty evaluation should use a combination of complementary metrics.

A useful summary is:

```text
Prediction Accuracy
        +
Calibration
        +
Sharpness
        +
Probabilistic Score
        +
Subgroup Analysis
```

---

## 27.20.37 Recommended Uncertainty Evaluation Workflow

For a Materials Informatics research project, the evaluation workflow can be:

```text
Test Dataset
     ↓
Generate Predictions
     ↓
Generate Uncertainty Estimates
     ↓
Calculate Prediction Errors
     ↓
Evaluate Coverage
     ↓
Evaluate Interval Width
     ↓
Evaluate Calibration
     ↓
Calculate NLL / Proper Scores
     ↓
Analyze Error–Uncertainty Relationship
     ↓
Analyze Materials Subgroups
     ↓
Evaluate Distribution Shift
```

This workflow prevents uncertainty from being treated as merely an additional column in a prediction table.

---

## 27.20.38 Practical Python Evaluation Function

A simple regression evaluation function can combine several metrics:

```python
import numpy as np

def evaluate_intervals(
    y_true,
    lower,
    upper
):

    covered = (
        (y_true >= lower)
        &
        (y_true <= upper)
    )

    coverage = covered.mean()

    widths = upper - lower

    mean_width = widths.mean()

    median_width = np.median(
        widths
    )

    return {
        "coverage": coverage,
        "mean_width": mean_width,
        "median_width": median_width
    }
```

It can be used as:

```python
results = evaluate_intervals(
    y_test,
    lower,
    upper
)

print(results)
```

The output might be:

```text
{
    'coverage': 0.896,
    'mean_width': 0.81,
    'median_width': 0.73
}
```

This can be extended with:

```text
MAE
RMSE
NLL
Coverage error
Subgroup coverage
```

for a complete uncertainty evaluation.

---

## 27.20.39 What a Good Uncertainty Model Looks Like

A useful uncertainty model should generally satisfy several properties.

### 1. Large errors should often receive larger uncertainty

```text
Error ↑
Uncertainty ↑
```

### 2. Prediction intervals should achieve appropriate coverage

```text
Nominal coverage
≈
Empirical coverage
```

### 3. Intervals should not be unnecessarily wide

```text
High coverage
+
Reasonably narrow intervals
```

### 4. Probabilistic predictions should score well

```text
Low NLL
```

or an appropriate alternative proper scoring rule.

### 5. Calibration should remain meaningful across relevant materials subgroups

For example:

```text
Oxides
Sulfides
Intermetallics
Different crystal systems
```

should be examined when these groups are scientifically relevant.

---

## 27.20.40 Common Misinterpretations

Several mistakes occur frequently when reporting uncertainty.

### Mistake 1

```text
Small uncertainty
=
Correct prediction
```

This is not necessarily true.

A model can be confidently wrong.

### Mistake 2

```text
Large uncertainty
=
Bad model
```

Not necessarily.

Large uncertainty may correctly indicate that a material lies in a difficult or poorly represented region.

### Mistake 3

```text
High coverage
=
Good uncertainty model
```

Not by itself.

Extremely wide intervals can achieve high coverage.

### Mistake 4

```text
Strong error-uncertainty correlation
=
Calibration
```

Correlation and calibration are different properties.

### Mistake 5

```text
Low MAE
=
Good uncertainty
```

Prediction accuracy and uncertainty quality must be evaluated separately.

---

## 27.20.41 The Central Principle of Uncertainty Evaluation

The central question is:

```text
Does the uncertainty estimate
correctly describe the reliability
of the prediction?
```

This requires comparing:

```text
Predicted uncertainty
```

against:

```text
Observed prediction behavior
```

Therefore:

```text
Uncertainty Estimate
        ↓
Evaluation
        ↓
Observed Errors
        ↓
Calibration + Sharpness
        ↓
Scientific Reliability
```

For Materials Informatics, this distinction is essential.

A model should not be considered uncertainty-aware merely because it outputs a number called:

```text
σ
```

or:

```text
confidence
```

The uncertainty must be quantitatively evaluated.

---

## 27.20.42 Key Takeaways

Uncertainty evaluation determines whether an uncertainty estimate is actually useful.

The major metrics include:

```text
Negative Log-Likelihood
Coverage
Coverage Error
Prediction Interval Width
Sharpness
PICP
MPIW
Expected Calibration Error
Brier Score
Proper Scoring Rules
Error–Uncertainty Correlation
```

The most important distinctions are:

```text
Prediction Error
≠
Prediction Uncertainty

Coverage
≠
Sharpness

Correlation
≠
Calibration

High Confidence
≠
Scientific Certainty
```

For regression models, prediction intervals should be evaluated using:

```text
Coverage
+
Interval Width
+
Probabilistic Scores
```

For classification models, probability calibration can be evaluated using:

```text
Reliability Analysis
ECE
Brier Score
NLL
```

For Materials ML, evaluation should also consider:

```text
Chemical Subgroups
Crystal Structures
Training-Data Density
Distribution Shift
```

The complete principle is:

```text
Prediction
      +
Uncertainty
      ↓
Evaluate
      ↓
Calibration
      +
Sharpness
      +
Probabilistic Quality
      +
Materials-Specific Analysis
```

An uncertainty estimate becomes scientifically useful only after demonstrating that it behaves reliably on appropriate held-out materials.

The next section will focus on **27.21 Uncertainty Visualization**, where these uncertainty estimates are represented through prediction intervals, error bars, confidence bands, uncertainty-versus-error plots, and uncertainty maps across materials space.

## 27.21 Uncertainty Visualization

Uncertainty estimates are most useful when they can be inspected visually.

A numerical value such as:

```text
Prediction = 2.8 eV
Uncertainty = 0.3 eV
```

contains useful information, but it does not immediately show how uncertainty changes across a dataset.

Visualization allows the researcher to examine relationships between:

```text
Prediction
Error
Uncertainty
Confidence
Materials composition
Crystal structure
Training-data density
```

Therefore, uncertainty visualization should be treated as an important part of uncertainty analysis.

A typical workflow is:

```text
Model
   ↓
Prediction
   +
Uncertainty
   ↓
Evaluation
   ↓
Visualization
   ↓
Scientific Interpretation
```

The purpose of visualization is not simply to make figures attractive.

The purpose is to answer scientific questions such as:

```text
Where is the model confident?

Where is the model uncertain?

Does uncertainty increase when prediction error increases?

Are certain materials systematically more uncertain?

Are there regions of materials space where the model should not be trusted?
```

---

## 27.21.1 Prediction Intervals

One of the simplest ways to visualize uncertainty is through a prediction interval.

Suppose a model predicts:

```text
ŷ = 2.8 eV
```

with an estimated uncertainty:

```text
σ = 0.2 eV
```

A simple interval may be written as:

```text
2.8 ± 0.2 eV
```

or:

```text
[2.6, 3.0] eV
```

If a model provides a 90% prediction interval, it can be represented as:

```text
[L, U]
```

where:

```text
L = lower prediction bound
U = upper prediction bound
```

The interval provides a visual representation of the range in which the model expects the target to lie.

---

## 27.21.2 Prediction Interval Plot

Suppose several materials have measured and predicted band gaps.

The predictions might be:

```text
Material A → 1.8 eV
Material B → 2.4 eV
Material C → 3.1 eV
Material D → 2.7 eV
```

with corresponding uncertainty intervals.

A visualization can be constructed as:

```text
Band Gap
  |
3.5|                         ●
   |                       ──┼──
3.0|             ●        ──┼──
   |           ──┼──
2.5|      ●
   |    ──┼──
2.0| ●
   |──┼────────────────────────
1.5|
   +---------------------------->
       A    B    C    D
```

The central point represents the prediction.

The vertical line represents the uncertainty interval.

This allows the researcher to immediately identify:

```text
Small interval → high model confidence

Large interval → high model uncertainty
```

---

## 27.21.3 Error Bars

A common implementation uses error bars.

For symmetric uncertainty:

```text
ŷ ± σ
```

the plotting code can be written as:

```python
import matplotlib.pyplot as plt

plt.errorbar(
    y_pred,
    y_true,
    xerr=uncertainty,
    fmt="o",
    alpha=0.7
)

plt.xlabel(
    "Predicted property"
)

plt.ylabel(
    "True property"
)

plt.show()
```

This type of plot is particularly useful when examining individual predictions.

However, the exact interpretation of the error bars must be stated.

For example:

```text
±1 standard deviation
```

is not automatically equivalent to:

```text
95% prediction interval
```

The figure caption should therefore explicitly define what the error bars represent.

---

## 27.21.4 Prediction vs True Value with Uncertainty

A standard diagnostic for regression models is:

```text
Predicted value
vs
True value
```

The ideal prediction line is:

```text
y = x
```

Uncertainty can be added around each prediction.

Conceptually:

```text
True
Value
  |
4 |                         ●
  |                       ╱
3 |                  ●───╱
  |              ●───╱
2 |         ●────╱
  |     ●──╱
1 | ●───╱
  +--------------------------> Predicted
    1    2    3    4
```

When uncertainty is included, each point can additionally display its prediction interval.

This allows three quantities to be examined simultaneously:

```text
Prediction
+
True value
+
Uncertainty
```

---

## 27.21.5 Residual vs Uncertainty Plot

A particularly useful uncertainty diagnostic is:

```text
Absolute Error
vs
Predicted Uncertainty
```

For each material:

```text
errorᵢ = |yᵢ - ŷᵢ|
```

and:

```text
uncertaintyᵢ = σᵢ
```

The resulting scatter plot can be interpreted as:

```text
Absolute
Error
  |
1.0|                       ●
   |                  ●
0.8|              ●
   |          ●
0.6|       ●
   |    ●
0.4| ●
   |____________________________
0.2|____________________________
   +---------------------------->
       Predicted Uncertainty
```

If uncertainty estimates are informative, larger uncertainties should generally be associated with larger errors.

However, the relationship does not need to be perfectly linear.

The important question is whether uncertainty provides useful information about prediction reliability.

---

## 27.21.6 Python: Error vs Uncertainty

A simple implementation is:

```python
import numpy as np
import matplotlib.pyplot as plt

absolute_error = np.abs(
    y_true - y_pred
)

plt.scatter(
    uncertainty,
    absolute_error,
    alpha=0.6
)

plt.xlabel(
    "Predicted uncertainty"
)

plt.ylabel(
    "Absolute prediction error"
)

plt.show()
```

This plot is often one of the first visualizations that should be generated when evaluating an uncertainty-aware model.

---

## 27.21.7 Interpreting the Error–Uncertainty Plot

Several patterns may appear.

### Pattern 1 — Useful uncertainty

```text
Low uncertainty
        ↓
Mostly small errors

High uncertainty
        ↓
Mostly larger errors
```

This indicates that the uncertainty estimate contains useful information.

### Pattern 2 — No relationship

```text
Low uncertainty
        ↓
Small and large errors mixed

High uncertainty
        ↓
Small and large errors mixed
```

The uncertainty estimate may not be informative.

### Pattern 3 — Systematic overconfidence

```text
Very low uncertainty
        +
Large errors
```

This is particularly concerning.

### Pattern 4 — Excessive uncertainty

```text
Very large uncertainty
        +
Small errors
```

The model may be overly conservative.

---

## 27.21.8 Confidence Bands

For ordered data, uncertainty can be visualized using a confidence or prediction band.

Suppose a model predicts a property as a function of temperature:

```text
T → predicted property
```

The model may produce:

```text
Mean prediction
Lower bound
Upper bound
```

The visualization can be:

```text
Property
  |
  |             ─────────
  |          ───   Mean
  |       ───
  |    ╱╲
  |  ╱    ╲
  | ╱      ╲
  +------------------------> Temperature
```

The shaded region around the prediction represents uncertainty.

For example:

```python
plt.plot(
    temperature,
    mean_prediction
)

plt.fill_between(
    temperature,
    lower_bound,
    upper_bound,
    alpha=0.2
)

plt.xlabel(
    "Temperature"
)

plt.ylabel(
    "Predicted property"
)

plt.show()
```

This is useful for materials properties that vary continuously with variables such as:

```text
Temperature
Pressure
Composition
Strain
Time
```

---

## 27.21.9 Confidence Band vs Prediction Band

These terms should not be used interchangeably.

A confidence band generally describes uncertainty in an estimated underlying relationship.

A prediction band generally describes uncertainty associated with future individual observations.

For Materials ML, the researcher should explicitly state which quantity is being shown.

For example:

```text
90% prediction interval
```

is much clearer than simply writing:

```text
90% confidence band
```

when the figure actually represents uncertainty around individual material-property predictions.

---

## 27.21.10 Uncertainty Distribution

Another useful visualization is the distribution of predicted uncertainties.

For example:

```python
plt.hist(
    uncertainty,
    bins=30,
    alpha=0.7
)

plt.xlabel(
    "Predicted uncertainty"
)

plt.ylabel(
    "Number of materials"
)

plt.show()
```

This answers:

```text
How uncertain is the model
across the dataset?
```

The distribution might look like:

```text
Number
of
Materials
  |
  |       ███
  |      █████
  |    ████████
  |  ███████████
  | █████████████
  +------------------------>
       Uncertainty
```

A long right tail may indicate a small number of particularly difficult materials.

---

## 27.21.11 Comparing Uncertainty Distributions

Uncertainty distributions can be compared between material groups.

For example:

```text
Oxides
Sulfides
Intermetallics
Polymers
```

A box plot can show the difference.

```python
plt.boxplot(
    [
        oxide_uncertainty,
        sulfide_uncertainty,
        intermetallic_uncertainty
    ],
    labels=[
        "Oxides",
        "Sulfides",
        "Intermetallics"
    ]
)

plt.ylabel(
    "Predicted uncertainty"
)

plt.show()
```

This can reveal whether certain chemical classes systematically produce more uncertain predictions.

---

## 27.21.12 Uncertainty by Crystal System

Crystal structures can also be grouped according to crystal system.

For example:

```text
Cubic
Tetragonal
Orthorhombic
Hexagonal
Monoclinic
Triclinic
```

The uncertainty distribution can then be compared between groups.

A result such as:

```text
Cubic:
Low median uncertainty

Triclinic:
Higher median uncertainty
```

could indicate that the model finds certain structural classes more difficult.

However, such a conclusion should not be made from visualization alone.

Other factors may explain the difference, including:

```text
Dataset size
Chemical diversity
Property range
Structural complexity
Training-data density
```

Therefore, visualization should lead to scientific investigation rather than automatically establish causality.

---

## 27.21.13 Uncertainty vs Property Value

Another useful visualization is:

```text
Predicted Property
vs
Predicted Uncertainty
```

For example:

```python
plt.scatter(
    y_pred,
    uncertainty,
    alpha=0.6
)

plt.xlabel(
    "Predicted band gap (eV)"
)

plt.ylabel(
    "Predicted uncertainty (eV)"
)

plt.show()
```

This can reveal whether uncertainty depends on the property range.

For example, the model may have:

```text
Low uncertainty:
Band gaps between 1–3 eV

High uncertainty:
Band gaps above 5 eV
```

Such a pattern may indicate limited representation of high-band-gap materials in the training dataset.

---

## 27.21.14 Uncertainty vs Training-Data Density

A particularly important visualization for Materials Informatics is:

```text
Training-data density
vs
Prediction uncertainty
```

Suppose each test material has a similarity score to the nearest training examples.

Then the researcher can plot:

```text
Similarity to training data
vs
Uncertainty
```

Conceptually:

```text
Uncertainty
  |
  | ●
  |   ●
  |     ●
  |       ●
  |          ●
  |______________●________>
       Training similarity
```

A decreasing trend may indicate:

```text
Greater similarity to known materials
        ↓
Lower uncertainty
```

while unfamiliar materials receive higher uncertainty.

This is particularly relevant when evaluating whether uncertainty reflects model familiarity.

---

## 27.21.15 Uncertainty Maps in Materials Space

Materials datasets often contain many descriptors.

For example:

```text
Composition
Atomic properties
Crystal structure
Electronic descriptors
Thermodynamic descriptors
```

A dimensionality-reduction method may represent the materials dataset in two dimensions.

For example:

```text
Materials
   ↓
Feature Representation
   ↓
2D Materials Space
```

Uncertainty can then be mapped onto this space.

Conceptually:

```text
                    Materials Space

       ● ● ● ●
      ● ● ● ● ●
     ● ● ● ● ●
                ○
                 ○
                  ○

● = low uncertainty
○ = high uncertainty
```

This allows researchers to identify regions where predictions are less reliable.

---

## 27.21.16 Uncertainty Heat Maps

If materials are arranged according to two scientifically meaningful variables, uncertainty can be displayed as a heat map.

For example:

```text
             Composition A
          low       high
       ┌──────────────────┐
low    │ low  low  medium │
       │ low  med  high   │
high   │ med  high high   │
       └──────────────────┘
```

The axes might represent:

```text
Composition
Structural descriptor
Temperature
Pressure
```

while the color or intensity represents:

```text
Prediction uncertainty
```

This creates an uncertainty landscape across the investigated materials space.

---

## 27.21.17 Uncertainty Visualization for Crystal Graphs

Crystal graph neural networks provide additional possibilities for visualization.

A crystal graph contains:

```text
Atoms → Nodes
Interactions → Edges
```

An uncertainty-aware GNN can therefore be analyzed at multiple levels.

```text
Crystal
   ↓
Graph
   ↓
Nodes + Edges
   ↓
Prediction
   +
Uncertainty
```

At the graph level:

```text
Crystal uncertainty
```

can be visualized.

At finer levels, one can inspect:

```text
Node-related contributions
Edge-related contributions
```

when the particular uncertainty or attribution method supports such analysis.

The important point is that uncertainty visualization should respect the representation used by the model.

---

## 27.21.18 Graph-Level Uncertainty

For a crystal property prediction model, the simplest uncertainty visualization is graph-level.

For example:

```text
Crystal A → 2.1 ± 0.1 eV
Crystal B → 2.8 ± 0.2 eV
Crystal C → 4.7 ± 1.0 eV
```

The visualization immediately shows that:

```text
Crystal A:
high confidence

Crystal B:
moderate confidence

Crystal C:
high uncertainty
```

The high uncertainty of Crystal C can then be investigated using:

```text
Composition
Structure
Training similarity
Model ensemble disagreement
```

---

## 27.21.19 Ensemble Prediction Visualization

Suppose five independently trained models produce:

```text
Model 1 → 2.4 eV
Model 2 → 2.5 eV
Model 3 → 2.6 eV
Model 4 → 2.5 eV
Model 5 → 2.4 eV
```

The predictions cluster tightly.

This indicates relatively low ensemble disagreement.

Now consider:

```text
Model 1 → 2.0 eV
Model 2 → 2.8 eV
Model 3 → 3.1 eV
Model 4 → 2.2 eV
Model 5 → 3.0 eV
```

The predictions are widely separated.

The ensemble therefore indicates greater epistemic uncertainty.

This can be visualized directly:

```text
Prediction
   |
3.2|       ●
3.0|             ●
2.8|     ●
2.6|
2.4| ●
2.2|         ●
2.0|
   +-------------------->
       Ensemble members
```

The spread itself becomes an interpretable uncertainty signal.

---

## 27.21.20 Uncertainty Across Multiple Models

A useful visualization compares uncertainty estimates produced by different methods.

For example:

```text
Material
   ↓
Random Forest
Deep Ensemble
MC Dropout
Gaussian Process
   ↓
Uncertainty estimates
```

The researcher may find:

```text
Material A:
RF = 0.15
DE = 0.18
MCD = 0.16

Material B:
RF = 0.20
DE = 0.65
MCD = 0.58
```

Material B is consistently more uncertain across methods.

This provides stronger evidence that the material is difficult for the models.

However, disagreement between uncertainty methods should itself be investigated rather than automatically interpreted as proof of high scientific uncertainty.

---

## 27.21.21 Uncertainty vs Absolute Error by Bins

A useful summary visualization is to group predictions by uncertainty.

For example:

```text
Uncertainty bins:

0.0–0.1
0.1–0.2
0.2–0.3
0.3–0.5
>0.5
```

For each bin, calculate mean absolute error.

Example:

| Uncertainty Range | Mean Absolute Error |
| ----------------- | ------------------: |
| 0.0–0.1           |                0.08 |
| 0.1–0.2           |                0.14 |
| 0.2–0.3           |                0.23 |
| 0.3–0.5           |                0.39 |
| >0.5              |                0.71 |

A pattern like this indicates that the uncertainty ranking is informative.

The model is effectively saying:

```text
More uncertain
        ↓
More likely to be wrong
```

---

## 27.21.22 Binned Uncertainty Plot in Python

This can be implemented using NumPy.

```python
import numpy as np
import matplotlib.pyplot as plt

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.5,
    np.inf
]

bin_indices = np.digitize(
    uncertainty,
    bins
)

mean_errors = []

for i in range(
    1,
    len(bins)
):

    mask = (
        bin_indices == i
    )

    if mask.any():

        mean_errors.append(
            np.mean(
                absolute_error[mask]
            )
        )

plt.plot(
    range(
        len(mean_errors)
    ),
    mean_errors,
    marker="o"
)

plt.xlabel(
    "Uncertainty bin"
)

plt.ylabel(
    "Mean absolute error"
)

plt.show()
```

This visualization provides a compact diagnostic of whether uncertainty ranking corresponds to actual prediction difficulty.

---

## 27.21.23 Reliability Visualization for Regression

For regression uncertainty, a reliability-style visualization can compare:

```text
Nominal coverage
```

with:

```text
Empirical coverage
```

For example:

```text
Empirical
Coverage
   |
1.0|                         ●
   |                    ●
0.8|                 ●
   |             ●
0.6|          ●
   |       ●
0.4|    ●
   | ●
0.0+------------------------------> Nominal
   0.0                          1.0
```

The ideal line is:

```text
y = x
```

The closer the empirical points are to the diagonal, the better the calibration.

This plot is particularly valuable because it evaluates uncertainty directly rather than relying only on average error.

---

## 27.21.24 Visualizing Overconfidence

Overconfidence often appears clearly in calibration plots.

Suppose:

```text
Nominal:
50%, 70%, 90%, 95%
```

but empirical coverage is:

```text
40%, 55%, 75%, 83%
```

The resulting curve lies below the diagonal.

This means:

```text
Intervals are too narrow
```

relative to the stated confidence levels.

The model is therefore overconfident.

---

## 27.21.25 Visualizing Underconfidence

Underconfidence produces the opposite pattern.

Suppose:

```text
Nominal:
50%, 70%, 90%, 95%
```

and:

```text
Empirical:
60%, 80%, 96%, 99%
```

The curve lies above the diagonal.

The intervals contain more observations than expected.

This may indicate that the model is producing overly wide intervals.

Therefore:

```text
Below diagonal
→ overconfidence

Above diagonal
→ underconfidence
```

---

## 27.21.26 Uncertainty Visualization for Material Screening

Suppose a model evaluates 10,000 candidate crystals.

The prediction table may contain:

```text
Material ID
Predicted property
Uncertainty
```

A simple screening plot can display:

```text
Predicted Property
   |
   |       ●
   |   ●       ●
   | ●
   |          ●
   |      ●
   +-------------------->
          Uncertainty
```

The researcher can identify:

```text
High property
+
Low uncertainty
```

as potentially high-confidence predictions.

Separately:

```text
High property
+
High uncertainty
```

indicates predictions that should be interpreted more cautiously.

This visualization does not decide which material is scientifically best.

It simply makes the prediction–uncertainty relationship visible.

---

## 27.21.27 Uncertainty as a Second Dimension of Prediction

A point prediction represents one quantity:

```text
ŷ
```

An uncertainty-aware prediction contains at least two:

```text
ŷ
σ
```

Therefore, a materials screening plot should often consider both.

Conceptually:

```text
                 High prediction
                       ↑
                       |
       High uncertainty|  ●
                       |
                       |
       Low uncertainty |  ●
                       |
                       +----------------→
                         Prediction
```

The researcher can therefore distinguish:

```text
High prediction
+
Low uncertainty
```

from:

```text
High prediction
+
High uncertainty
```

These are scientifically different situations.

---

## 27.21.28 Visualization Does Not Replace Quantitative Evaluation

A visually attractive uncertainty map is not evidence that the uncertainty model is calibrated.

For example:

```text
Beautiful uncertainty map
        ≠
Reliable uncertainty
```

Visualization should therefore be combined with:

```text
Coverage
NLL
Calibration metrics
Interval width
Error analysis
```

The correct workflow is:

```text
Quantitative Evaluation
        ↓
Visualization
        ↓
Scientific Interpretation
```

rather than:

```text
Visualization
        ↓
Assume Reliability
```

---

## 27.21.29 Recommended Visualization Set

For a research-grade Materials ML study, a useful minimum visualization set is:

### 1. Prediction vs true value

```text
ŷ vs y
```

### 2. Error vs uncertainty

```text
|y - ŷ| vs σ
```

### 3. Uncertainty distribution

```text
σ histogram
```

### 4. Coverage calibration curve

```text
Empirical coverage
vs
Nominal coverage
```

### 5. Interval-width distribution

```text
Prediction interval width
```

### 6. Materials-space uncertainty map

```text
Materials representation
vs
Uncertainty
```

### 7. Group-wise uncertainty

```text
Chemical family
vs
Uncertainty
```

Together these figures provide a much more complete picture of uncertainty behavior.

---

## 27.21.30 Example Research Figure Layout

A complete uncertainty analysis could contain:

```text
┌─────────────────────┬─────────────────────┐
│ Prediction vs True  │ Error vs Uncertainty│
│                     │                     │
├─────────────────────┼─────────────────────┤
│ Uncertainty         │ Coverage Calibration│
│ Distribution        │                     │
├─────────────────────┼─────────────────────┤
│ Materials-Space     │ Group-wise          │
│ Uncertainty Map     │ Uncertainty         │
└─────────────────────┴─────────────────────┘
```

Each figure answers a different question.

```text
Prediction vs True
→ How accurate is the model?

Error vs Uncertainty
→ Does uncertainty identify difficult predictions?

Uncertainty Distribution
→ How uncertain is the model overall?

Coverage Curve
→ Is uncertainty calibrated?

Materials-Space Map
→ Where is the model uncertain?

Group-wise Plot
→ Which material classes are difficult?
```

---

## 27.21.31 Reproducible Visualization

Visualization should be generated from stored prediction results rather than manually edited figures.

A useful prediction table might contain:

```text
material_id
y_true
y_pred
uncertainty
lower_bound
upper_bound
chemical_system
crystal_system
```

For example:

```python
import pandas as pd

results = pd.DataFrame({
    "material_id": material_ids,
    "y_true": y_true,
    "y_pred": y_pred,
    "uncertainty": uncertainty,
    "lower_bound": lower_bound,
    "upper_bound": upper_bound
})
```

The same table can then generate:

```text
Error plots
Coverage plots
Histograms
Group comparisons
Materials-space maps
```

This makes the analysis reproducible.

---

## 27.21.32 Visualization and Scientific Interpretation

The final purpose of uncertainty visualization is scientific interpretation.

Suppose a visualization shows:

```text
High uncertainty
```

for a particular group of crystals.

The researcher should ask:

```text
Why?
```

Possible explanations include:

```text
Few training examples
Large structural diversity
Unusual compositions
Property range outside training data
Difficult structural representations
Model limitations
```

The visualization identifies the pattern.

Scientific analysis must determine the reason.

Therefore:

```text
Visualization
        ↓
Pattern
        ↓
Hypothesis
        ↓
Scientific Investigation
```

This prevents visual patterns from being mistaken for explanations.

---

## 27.21.33 Key Takeaways

Uncertainty visualization converts numerical uncertainty estimates into interpretable scientific information.

Important visualizations include:

```text
Prediction intervals
Error bars
Confidence / prediction bands
Uncertainty distributions
Error vs uncertainty plots
Coverage calibration curves
Uncertainty vs property plots
Uncertainty vs training density
Materials-space uncertainty maps
Group-wise uncertainty plots
Crystal graph uncertainty visualization
```

The most important principles are:

```text
Uncertainty should be visible.

Uncertainty should be compared with error.

Calibration should be visualized.

Sharpness should be visualized.

Different materials regions should be compared.

Visualization should support,
not replace, quantitative evaluation.
```

For Materials Informatics, the most useful conceptual workflow is:

```text
Prediction
     +
Uncertainty
     ↓
Quantitative Evaluation
     ↓
Visualization
     ↓
Identify Uncertain Regions
     ↓
Scientific Interpretation
```

The objective is not simply to show that a model is uncertain.

The objective is to understand **where, when, and under what materials conditions the model becomes uncertain**.

This provides a direct connection between uncertainty estimation and interpretation of crystal-property predictions.

The next section will examine **27.22 Uncertainty in Crystal Graph Neural Networks**, focusing specifically on uncertainty estimation for graph-based crystal property prediction.

## 27.22 Uncertainty in Crystal Graph Neural Networks

Crystal Graph Neural Networks (GNNs) represent a crystal structure as a graph and use this representation to predict material properties.

A simplified representation is:

```text
Crystal Structure
       ↓
Crystal Graph
       ↓
GNN
       ↓
Property Prediction
```

For a conventional GNN, the output may be a single value:

```text
Band gap = 2.41 eV
```

An uncertainty-aware GNN instead produces:

```text
Band gap = 2.41 ± 0.18 eV
```

The second prediction contains substantially more information.

The first tells us what the model predicts.

The second tells us both:

```text
What the model predicts
+
How uncertain the model is
```

This distinction is particularly important for crystal-property prediction because crystal datasets often contain chemically diverse compositions, different structural environments, and regions of materials space that are poorly represented in the training data.

Therefore, uncertainty estimation should be integrated into the evaluation of crystal GNNs rather than treated as an unrelated post-processing step.

---

## 27.22.1 Why Uncertainty Matters for Crystal GNNs

Crystal GNNs can achieve strong predictive performance while still producing unreliable predictions for particular structures.

Consider two crystals:

```text
Crystal A

Very similar to many structures
in the training dataset
```

and:

```text
Crystal B

Chemically unusual
Structurally unusual
Poorly represented in training data
```

A GNN may produce:

```text
Crystal A → 2.50 eV ± 0.08 eV

Crystal B → 4.10 eV ± 0.75 eV
```

The uncertainty estimate indicates that the model has much greater uncertainty for Crystal B.

This provides information that cannot be obtained from the point prediction alone.

Without uncertainty:

```text
Crystal A → 2.50 eV
Crystal B → 4.10 eV
```

the two predictions appear equally reliable.

With uncertainty:

```text
Crystal A → 2.50 ± 0.08 eV
Crystal B → 4.10 ± 0.75 eV
```

the distinction becomes explicit.

---

## 27.22.2 Sources of Uncertainty in Crystal GNNs

Several sources can contribute to uncertainty in a crystal GNN.

A useful conceptual decomposition is:

```text
Crystal GNN Uncertainty
        │
        ├── Data uncertainty
        │
        ├── Model uncertainty
        │
        ├── Representation uncertainty
        │
        └── Distribution uncertainty
```

### Data uncertainty

This can arise from:

```text
Experimental noise
DFT approximation
Inconsistent labels
Measurement uncertainty
```

### Model uncertainty

This can arise when:

```text
Training data are sparse
Model parameters are poorly constrained
Crystal is outside familiar training regions
```

### Representation-related effects

These can arise from:

```text
Neighbor cutoff
Graph construction
Missing structural information
Choice of node features
Choice of edge features
```

### Distribution uncertainty

This becomes important when the target crystal differs substantially from the training distribution.

For example:

```text
Training:
Oxides + sulfides

Prediction:
Unusual halide structure
```

The model may have limited knowledge of this region of crystal space.

---

## 27.22.3 Graph-Level Uncertainty

The most direct form of uncertainty estimation for a crystal GNN is graph-level uncertainty.

A crystal is represented by a graph:

```text
G = (V, E)
```

where:

```text
V = atoms
E = atomic interactions
```

The GNN maps the graph to a property:

```text
ŷ = f(G)
```

An uncertainty-aware model can instead produce:

```text
ŷ, σ
```

where:

```text
ŷ = predicted property

σ = estimated uncertainty
```

For example:

```text
Crystal graph
      ↓
GNN
      ↓
2.73 eV
      +
0.21 eV uncertainty
```

This is the simplest uncertainty representation for a crystal GNN.

---

## 27.22.4 Ensemble GNNs

One of the most practical methods for estimating epistemic uncertainty in crystal GNNs is the deep ensemble.

Instead of training one GNN:

```text
Crystal
   ↓
GNN
   ↓
Prediction
```

we train multiple independently initialized GNNs:

```text
                 ┌── GNN 1 ──→ ŷ₁
                 │
Crystal ─────────┼── GNN 2 ──→ ŷ₂
                 │
                 ├── GNN 3 ──→ ŷ₃
                 │
                 ├── GNN 4 ──→ ŷ₄
                 │
                 └── GNN 5 ──→ ŷ₅
```

The predictions can then be combined.

The ensemble mean is:

```text
ŷ = (1/M) Σ ŷₘ
```

where `M` is the number of models.

The spread among the predictions provides an estimate of model disagreement.

A commonly used estimate is:

```text
σ² = (1/M) Σ (ŷₘ - ŷ)²
```

or its square root:

```text
σ = sqrt[
        (1/M) Σ (ŷₘ - ŷ)²
    ]
```

The exact variance convention should be stated because sample and population variance use different denominators.

---

## 27.22.5 Example of Ensemble GNN Predictions

Suppose five crystal GNNs predict the band gap of the same material:

```text
GNN 1 → 2.40 eV
GNN 2 → 2.45 eV
GNN 3 → 2.38 eV
GNN 4 → 2.42 eV
GNN 5 → 2.41 eV
```

The predictions are tightly clustered.

Therefore:

```text
Low model disagreement
        ↓
Low epistemic uncertainty
```

Now consider another crystal:

```text
GNN 1 → 1.8 eV
GNN 2 → 2.7 eV
GNN 3 → 3.2 eV
GNN 4 → 2.1 eV
GNN 5 → 3.0 eV
```

The predictions differ substantially.

Therefore:

```text
High model disagreement
        ↓
High epistemic uncertainty
```

This is one of the main reasons ensemble methods are useful for crystal GNNs.

---

## 27.22.6 PyTorch Implementation of an Ensemble

Suppose a trained crystal GNN is represented by:

```python
model
```

Several independently trained models can be stored:

```python
models = [
    model_1,
    model_2,
    model_3,
    model_4,
    model_5
]
```

Predictions can then be collected:

```python
import torch

predictions = []

with torch.no_grad():

    for model in models:

        prediction = model(
            crystal_graph
        )

        predictions.append(
            prediction
        )
```

The predictions can be converted into a tensor:

```python
predictions = torch.stack(
    predictions
)
```

Then calculate the ensemble mean:

```python
mean_prediction = (
    predictions.mean(
        dim=0
    )
)
```

and ensemble standard deviation:

```python
uncertainty = (
    predictions.std(
        dim=0
    )
)
```

The result is:

```text
mean_prediction
+
uncertainty
```

for each crystal.

---

## 27.22.7 Important Limitation of Ensemble Spread

Ensemble disagreement should not automatically be interpreted as the complete predictive uncertainty.

The ensemble spread primarily captures disagreement between model predictions.

Therefore, it is especially useful as an estimate of epistemic uncertainty.

It does not automatically capture all sources of aleatoric uncertainty.

For example:

```text
Experimental measurement noise
```

may remain even if all ensemble models produce nearly identical predictions.

Therefore:

```text
Ensemble spread
≠
Complete uncertainty in every situation
```

A research study should clearly define what its uncertainty estimate represents.

---

## 27.22.8 Combining Aleatoric and Epistemic Uncertainty

A crystal GNN can be designed to predict both:

```text
Mean
+
Aleatoric variance
```

while an ensemble provides:

```text
Epistemic variance
```

The total predictive variance can then be conceptually decomposed as:

```text
Total Variance
=
Aleatoric Variance
+
Epistemic Variance
```

For an ensemble with model-specific predicted variances:

```text
σ_total²
=
(1/M) Σ σₘ²
+
(1/M) Σ (μₘ - μ)²
```

where:

```text
μₘ = prediction of model m

σₘ² = predicted aleatoric variance of model m

μ = ensemble mean
```

The first term represents average predicted data uncertainty.

The second term represents disagreement among models.

This decomposition provides a more informative uncertainty estimate than ensemble disagreement alone.

---

## 27.22.9 Heteroscedastic Crystal GNNs

Crystal datasets may contain materials with different levels of intrinsic uncertainty.

For example:

```text
Simple material
→ relatively consistent property

Complex material
→ greater uncertainty in target value
```

A GNN can therefore be designed to predict both:

```text
μ(G)
```

and:

```text
σ²(G)
```

as functions of the crystal graph.

The model becomes:

```text
Crystal Graph
      ↓
GNN
      ↓
┌───────────────┐
│ Mean μ        │
│ Variance σ²   │
└───────────────┘
```

This allows the model to learn input-dependent uncertainty.

---

## 27.22.10 Gaussian Negative Log-Likelihood

If a GNN predicts a Gaussian distribution:

```text
y ~ N(μ, σ²)
```

a common training objective is the Gaussian negative log-likelihood.

Ignoring constants, it can be written as:

```text
L =
1/2 [
    log(σ²)
    +
    (y - μ)² / σ²
]
```

The important behavior is that the model learns two things simultaneously:

```text
Prediction error
+
Appropriate uncertainty
```

If the model predicts a large error but very small uncertainty, the loss becomes large.

Therefore, the model is encouraged not only to predict accurately but also to produce uncertainty estimates that are consistent with observed errors.

---

## 27.22.11 Why Predicting Variance Is Not Enough

A GNN can technically produce a variance estimate for every crystal.

However, the variance is only scientifically useful if it is calibrated.

For example:

```text
Prediction:
2.5 eV

Estimated uncertainty:
0.1 eV
```

does not guarantee that the true value lies within:

```text
2.4–2.6 eV
```

with any particular probability.

The relationship must be evaluated empirically.

Therefore:

```text
Predicted uncertainty
        ↓
Calibration
        ↓
Validated uncertainty
```

is necessary.

---

## 27.22.12 Monte Carlo Dropout for Crystal GNNs

Monte Carlo Dropout can also be applied to crystal GNNs.

During ordinary inference, dropout is usually disabled.

For Monte Carlo Dropout, dropout remains active and the same crystal graph is evaluated multiple times.

The workflow becomes:

```text
Crystal Graph
      ↓
GNN + Dropout
      ↓
Prediction 1
Prediction 2
Prediction 3
...
Prediction N
      ↓
Mean + Spread
```

For example:

```python
predictions = []

model.train()

with torch.no_grad():

    for _ in range(100):

        prediction = model(
            crystal_graph
        )

        predictions.append(
            prediction
        )

predictions = torch.stack(
    predictions
)

mean_prediction = (
    predictions.mean(dim=0)
)

uncertainty = (
    predictions.std(dim=0)
)
```

The stochastic predictions provide an approximate uncertainty estimate.

---

## 27.22.13 Important MC Dropout Consideration

The model should not simply be switched to training mode without considering what other layers do.

For example, if the GNN contains Batch Normalization, using:

```python
model.train()
```

may change more than dropout behavior.

A more careful implementation may selectively activate dropout layers while keeping other inference behavior fixed.

Conceptually:

```text
Dropout → stochastic

Other inference behavior → fixed
```

This avoids introducing unintended changes into the prediction process.

---

## 27.22.14 Comparing Deep Ensembles and MC Dropout

Both approaches can estimate model uncertainty.

### Deep Ensemble

```text
Multiple independently trained GNNs
```

Advantages:

```text
Simple concept
Strong empirical performance
Different learned parameter configurations
```

Disadvantages:

```text
Higher computational cost
Multiple models must be stored
Multiple training runs required
```

### MC Dropout

```text
One GNN
+
Multiple stochastic forward passes
```

Advantages:

```text
Lower training cost
One trained model
Easy integration with existing dropout networks
```

Disadvantages:

```text
Approximate uncertainty
Sensitive to dropout design
Requires many forward passes
```

Neither method should automatically be assumed to produce perfectly calibrated uncertainty.

Calibration must still be evaluated.

---

## 27.22.15 Uncertainty in Message-Passing GNNs

Crystal GNNs often use message passing.

A simplified message-passing layer can be written conceptually as:

```text
Node Features
      ↓
Neighbor Messages
      ↓
Aggregation
      ↓
Updated Node Features
```

For a crystal:

```text
Atom i
  ↑
Neighbors
  ↑
Periodic interactions
```

The final graph representation is constructed from repeated message-passing operations.

Uncertainty can therefore emerge from uncertainty in how the model represents these interactions.

For example:

```text
Common local environment
        ↓
Models agree

Unusual local environment
        ↓
Models disagree
```

This can contribute to higher epistemic uncertainty.

---

## 27.22.16 Crystal Structure Complexity and Uncertainty

Some crystal structures may be more difficult for a GNN to model.

Possible factors include:

```text
Large number of atoms
Complex coordination environments
Multiple chemical species
Unusual local bonding
Large unit cells
Rare structural motifs
```

However, structural complexity should not automatically be equated with high uncertainty.

A complex structure can still be well represented if many similar examples exist in the training data.

Conversely, a relatively simple structure can have high uncertainty if it belongs to an unfamiliar chemical region.

Therefore:

```text
Structural complexity
```

and:

```text
Model uncertainty
```

should be measured separately.

---

## 27.22.17 Uncertainty and Neighbor Cutoffs

Crystal GNNs commonly construct edges using a distance cutoff.

For example:

```text
r < r_cut
```

defines which atoms interact in the graph.

The chosen cutoff influences the graph representation.

Consider:

```text
r_cut = 4 Å
```

versus:

```text
r_cut = 6 Å
```

The resulting graphs may contain different numbers of edges.

Therefore, the model's predictions and uncertainty estimates can change with graph construction.

This means that uncertainty analysis should be performed using a clearly defined graph-construction procedure.

---

## 27.22.18 Periodic Neighbors and Uncertainty

Crystal graphs must account for periodicity.

An atom near one boundary of a unit cell may interact with an atom in a neighboring periodic image.

Therefore:

```text
Unit Cell
   +
Periodic Images
   ↓
Complete Neighbor Graph
```

If periodic neighbors are incorrectly constructed, the GNN may receive an incomplete representation.

This can lead to:

```text
Incorrect prediction
```

and potentially:

```text
Misleading uncertainty
```

Therefore, uncertainty analysis cannot compensate for an incorrect crystal graph.

The representation itself must first be physically meaningful.

---

## 27.22.19 Uncertainty and Chemical Novelty

Suppose a crystal GNN is trained mainly on:

```text
Li–Fe–O
Na–Fe–O
Li–Co–O
```

and then receives:

```text
Rare multi-element composition
```

The model may produce high ensemble disagreement.

For example:

```text
GNN 1 → 3.1 eV
GNN 2 → 3.8 eV
GNN 3 → 2.9 eV
GNN 4 → 4.2 eV
```

This suggests that the model has limited agreement in this chemical region.

However, the researcher should not immediately conclude:

```text
The material is scientifically uncertain.
```

The more precise statement is:

```text
The model is uncertain about its prediction for this material.
```

This distinction is essential.

Model uncertainty is not automatically the same as physical uncertainty.

---

## 27.22.20 Uncertainty and Structural Novelty

The same principle applies to structural novelty.

Suppose the training dataset contains many common crystal structures but very few examples of a particular coordination environment.

A new crystal may contain:

```text
Rare coordination
```

or:

```text
Unusual local geometry
```

The GNN ensemble may disagree more strongly.

This can produce:

```text
High epistemic uncertainty
```

because the model has less evidence from which to infer the correct property.

Therefore, uncertainty can sometimes act as an indicator of structural unfamiliarity.

---

## 27.22.21 Uncertainty vs Crystal Similarity

A useful analysis is to calculate a similarity measure between a test crystal and training crystals.

Conceptually:

```text
Crystal
   ↓
Graph / Feature Representation
   ↓
Similarity to Training Set
   ↓
Compare with GNN Uncertainty
```

If uncertainty tends to increase as similarity decreases, the model may be behaving sensibly.

The relationship can be visualized as:

```text
Uncertainty
   |
   | ●
   |  ●
   |    ●
   |      ●
   |         ●
   +-------------------->
      Crystal Similarity
```

The precise relationship depends on the representation and similarity metric.

It should therefore be evaluated empirically rather than assumed.

---

## 27.22.22 Group-Level Uncertainty Analysis

Crystal GNN uncertainty can also be grouped by:

```text
Chemical system
Crystal system
Space group
Number of elements
Number of atoms
Coordination environment
Property range
```

For example:

```text
Chemical System       Median Uncertainty
-----------------------------------------
Li-Fe-O               0.10 eV
Na-Fe-O               0.13 eV
Li-Co-O               0.15 eV
Rare systems          0.42 eV
```

This may reveal that rare chemical systems are more uncertain.

However, group sizes must be reported.

A group containing five crystals should not be interpreted with the same confidence as a group containing thousands.

---

## 27.22.23 Node-Level Uncertainty

A graph model contains individual nodes corresponding to atoms.

This raises a more detailed question:

```text
Which atoms are associated with uncertainty?
```

Node-level uncertainty is more difficult to define than graph-level uncertainty because the final prediction is usually made for the entire crystal.

Nevertheless, node-related analysis can be useful when a method provides uncertainty or attribution information at the node level.

For example:

```text
Crystal
 ┌─────────────┐
 │ O   O   O   │
 │   Fe        │
 │ O   O   O   │
 └─────────────┘
```

One may identify that unusual atomic environments contribute strongly to model behavior.

However, node importance should not automatically be interpreted as a direct physical uncertainty.

The distinction between:

```text
Attribution
```

and:

```text
Uncertainty
```

must remain clear.

---

## 27.22.24 Edge-Level Uncertainty

Edges represent interactions between atoms.

For example:

```text
Fe ─ O
```

may represent a local interaction within the crystal graph.

An uncertainty-aware analysis may examine whether unusual or poorly represented interactions correspond to increased model disagreement.

Again, this requires careful interpretation.

An important distinction is:

```text
Edge importance
```

versus:

```text
Edge uncertainty
```

An important edge is not necessarily an uncertain edge.

Similarly:

```text
High uncertainty
```

does not automatically identify a specific edge as physically responsible.

---

## 27.22.25 Graph-Level vs Node-Level Analysis

The levels should therefore be separated:

```text
Graph level
→ Is the crystal prediction uncertain?

Node level
→ Which atomic representations are associated with the model behavior?

Edge level
→ Which interactions are associated with the model behavior?
```

The first is directly an uncertainty question.

The latter two require specialized methods and should be interpreted carefully.

---

## 27.22.26 Practical Uncertainty Workflow for Crystal GNNs

A practical workflow can be organized as:

```text
Crystal Structures
       ↓
Pymatgen
       ↓
Crystal Graph Construction
       ↓
Train Multiple GNNs
       ↓
Predict Each Crystal
       ↓
Calculate Ensemble Mean
       ↓
Calculate Ensemble Spread
       ↓
Calibrate Uncertainty
       ↓
Evaluate Error vs Uncertainty
       ↓
Analyze Chemical / Structural Groups
```

The key point is that uncertainty estimation occurs after obtaining multiple model predictions but before scientific interpretation.

---

## 27.22.27 Example End-to-End Ensemble Workflow

Suppose a dataset contains:

```text
Crystal graph
+
Target property
```

The researcher trains five models:

```python
models = []

for seed in range(5):

    model = build_gnn()

    train_model(
        model,
        train_loader,
        seed=seed
    )

    models.append(
        model
    )
```

Predictions are then generated:

```python
all_predictions = []

for model in models:

    model.eval()

    predictions = []

    with torch.no_grad():

        for batch in test_loader:

            pred = model(
                batch
            )

            predictions.append(
                pred
            )

    predictions = torch.cat(
        predictions
    )

    all_predictions.append(
        predictions
    )
```

The model predictions are stacked:

```python
all_predictions = torch.stack(
    all_predictions
)
```

Then:

```python
mean_prediction = (
    all_predictions.mean(
        dim=0
    )
)

uncertainty = (
    all_predictions.std(
        dim=0
    )
)
```

The resulting arrays contain:

```text
mean_prediction[i]
uncertainty[i]
```

for each crystal `i`.

---

## 27.22.28 Creating an Uncertainty-Aware Results Table

The predictions can be stored together with crystal information.

```python
results = pd.DataFrame({

    "material_id": material_ids,

    "y_true": y_true,

    "prediction":
        mean_prediction.cpu().numpy(),

    "uncertainty":
        uncertainty.cpu().numpy(),

    "chemical_system":
        chemical_systems,

    "crystal_system":
        crystal_systems

})
```

The table can then be used to investigate:

```text
Most uncertain crystals
Least uncertain crystals
Largest prediction errors
Chemical groups with high uncertainty
Structural groups with high uncertainty
```

For example:

```python
most_uncertain = (
    results
    .sort_values(
        "uncertainty",
        ascending=False
    )
    .head(20)
)
```

This produces a list of the crystals for which the ensemble disagrees most strongly.

---

## 27.22.29 Inspecting Highly Uncertain Crystals

The most uncertain predictions should not simply be discarded.

They should be inspected.

A useful procedure is:

```text
Highly uncertain crystal
        ↓
Check composition
        ↓
Check structure
        ↓
Check training-set similarity
        ↓
Check graph construction
        ↓
Check model predictions
        ↓
Determine possible reason
```

For example, a highly uncertain material may turn out to have:

```text
Rare composition
+
Rare coordination environment
+
Few similar training examples
```

This gives scientific meaning to the uncertainty estimate.

---

## 27.22.30 Uncertainty Does Not Mean Bad Material

A crucial distinction is:

```text
High model uncertainty
```

does not mean:

```text
Bad material
```

and:

```text
Low model uncertainty
```

does not mean:

```text
Good material
```

Uncertainty describes the model's confidence in its prediction.

For example:

```text
Crystal A:
Predicted band gap = 4.2 eV
Uncertainty = 0.8 eV
```

does not mean the material is undesirable.

It means the prediction should be interpreted cautiously.

Similarly:

```text
Crystal B:
Predicted band gap = 2.1 eV
Uncertainty = 0.05 eV
```

means the model is relatively confident, not that the material is scientifically superior.

---

## 27.22.31 Uncertainty and Prediction Error

After testing the GNN, the researcher can compare:

```text
Absolute Error
```

with:

```text
Uncertainty
```

For each crystal:

```text
errorᵢ = |yᵢ - ŷᵢ|
```

and:

```text
uncertaintyᵢ = σᵢ
```

Then calculate the correlation between them.

For example:

```python
correlation = np.corrcoef(
    absolute_error,
    uncertainty
)[0, 1]

print(
    correlation
)
```

A positive relationship indicates that larger uncertainty tends to be associated with larger errors.

However, correlation alone is not sufficient to establish calibration.

Coverage analysis is still required.

---

## 27.22.32 Uncertainty Evaluation for Crystal GNNs

A research-grade evaluation should therefore contain several components:

```text
Prediction Accuracy
        +
Uncertainty Quality
        +
Calibration
        +
Distribution Analysis
```

For example:

```text
MAE
RMSE
R²
```

describe predictive accuracy.

While:

```text
Coverage
NLL
Interval width
Error–uncertainty relationship
```

describe uncertainty quality.

A strong crystal GNN study should report both.

---

## 27.22.33 Recommended Crystal GNN Uncertainty Report

A useful research table could contain:

| Metric                        |  Result |
| ----------------------------- | ------: |
| MAE                           | 0.18 eV |
| RMSE                          | 0.31 eV |
| Mean uncertainty              | 0.22 eV |
| Median uncertainty            | 0.17 eV |
| 90% interval coverage         |     88% |
| Mean interval width           | 0.71 eV |
| Error–uncertainty correlation |    0.64 |

The exact numbers are illustrative.

The important point is that model accuracy and uncertainty quality should be reported separately.

---

## 27.22.34 Research Interpretation

Suppose a crystal GNN produces:

```text
MAE = 0.15 eV
```

and:

```text
90% interval coverage = 62%
```

The model may have good average predictive accuracy but poorly calibrated uncertainty.

Conversely:

```text
MAE = 0.30 eV
```

with excellent calibration does not mean the model is accurate enough for every scientific application.

Therefore:

```text
Accuracy
```

and:

```text
Uncertainty reliability
```

answer different questions.

The correct interpretation is:

```text
How accurately does the model predict?

+
How reliably does it know when its prediction may be wrong?
```

---

## 27.22.35 Final Crystal GNN Uncertainty Framework

The complete framework can be summarized as:

```text
                 CRYSTAL
                    │
                    ▼
            Crystal Graph
                    │
                    ▼
             GNN Ensemble
          ┌────┬────┬────┐
          ▼    ▼    ▼    ▼
         GNN1 GNN2 GNN3 ... GNNM
          │    │    │      │
          └────┴────┴──────┘
                    │
                    ▼
             Mean Prediction
                    +
             Model Uncertainty
                    │
                    ▼
               Calibration
                    │
                    ▼
           Error–Uncertainty
                Analysis
                    │
                    ▼
        Chemical / Structural
             Analysis
                    │
                    ▼
          Scientific Interpretation
```

The central idea is:

```text
Crystal Graph
      ↓
Prediction
      +
Uncertainty
      ↓
Calibration
      ↓
Evaluation
      ↓
Scientific Interpretation
```

Uncertainty-aware crystal GNNs therefore provide more information than ordinary point-prediction models.

However, uncertainty should always be interpreted carefully.

A high uncertainty value means:

```text
The model has limited confidence
in its prediction.
```

It does not automatically mean:

```text
The material is unstable.
The material is poor.
The material is physically uncertain.
The prediction is definitely wrong.
```

Likewise, a low uncertainty value means that the model is relatively confident, but confidence alone does not establish scientific correctness.

The appropriate research principle is therefore:

```text
Prediction
+
Uncertainty
+
Calibration
+
Validation
```

rather than prediction alone.

This makes uncertainty estimation an important component of reliable crystal-property prediction with Graph Neural Networks.
