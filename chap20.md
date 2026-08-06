# Chapter 20 — Equivariant Graph Neural Networks

# 20.1 Introduction

The development of machine learning models for materials science has progressed remarkably over the past decade.

Initially, most machine learning approaches relied on **handcrafted descriptors**, where researchers manually designed numerical features representing composition, crystal structure, or chemical properties. These descriptors were then used with algorithms such as Random Forests, Support Vector Machines, or Gradient Boosting to predict material properties.

Although these methods achieved considerable success, they depended heavily on human expertise. A poorly designed descriptor could significantly reduce model performance, while designing effective descriptors often required extensive domain knowledge in chemistry, crystallography, and materials science.

The emergence of **Graph Neural Networks (GNNs)** fundamentally changed this paradigm.

Instead of manually engineering descriptors, GNNs learn directly from the crystal structure itself. Atoms are represented as graph nodes, chemical bonds or neighboring interactions as graph edges, and neural message passing automatically learns meaningful representations of the local atomic environment.

This breakthrough led to highly successful models such as

* Crystal Graph Convolutional Neural Networks (CGCNN),
* Materials Graph Networks (MEGNet),
* Atomistic Line Graph Neural Networks (ALIGNN),
* Crystal Hamiltonian Graph Neural Networks (CHGNet).

These models demonstrated that machine learning could accurately predict

* formation energies,
* band gaps,
* elastic properties,
* magnetic moments,
* atomic forces,
* stress tensors,

while requiring only the crystal structure as input.

However, despite their success, researchers soon recognized an important limitation.

Traditional graph neural networks often fail to fully exploit one of the most fundamental principles of physics:

> **Physical laws are governed by symmetry.**

Understanding this principle marks the transition from conventional graph neural networks to **Equivariant Graph Neural Networks (EGNNs)**.

---

# 20.2 The Next Evolution of Graph Neural Networks

Imagine holding a crystal of silicon in your hand.

If you rotate the crystal by 90°,

has the material changed?

The answer is clearly **no**.

The crystal still possesses

* the same atomic composition,
* the same bonding,
* the same lattice,
* the same electronic structure,
* the same total energy.

Only the coordinate system from which we observe the crystal has changed.

Similarly,

if we move the entire crystal by several centimeters,

its physical properties remain exactly the same.

Nature does not care about our choice of coordinate system.

Unfortunately, many conventional neural networks do.

---

# 20.3 Why Conventional Neural Networks Struggle

A traditional neural network treats coordinates simply as numbers.

For example,

```text
Atom A

(2.1, 3.4, 1.8)
```

is interpreted merely as three floating-point values.

If we rotate the crystal,

the coordinates become

```text
Atom A

(-3.4, 2.1, 1.8)
```

Although both coordinate sets describe exactly the same physical atom,

the neural network may interpret them as completely different inputs.

Consequently,

the model must learn from data that

* rotated crystals have identical energies,
* translated crystals represent identical structures,
* forces rotate together with the crystal.

Learning these relationships purely from data is inefficient.

---

# 20.4 Learning Symmetry from Data Is Difficult

Suppose we wish to teach a conventional neural network that rotating a crystal does not change its energy.

One approach is data augmentation.

We repeatedly rotate every crystal.

```text
Original Crystal

↓

Rotate 10°

↓

Rotate 20°

↓

Rotate 45°

↓

Rotate 90°

↓

Rotate 180°

↓

Rotate 270°
```

Every rotated structure is added to the training set.

Although this works,

it introduces several problems.

First,

the dataset becomes much larger.

Second,

training becomes slower.

Third,

the network still has no guarantee that it has perfectly learned rotational symmetry.

Instead,

it merely approximates it statistically.

---

# 20.5 Physics Already Knows the Answer

Physics tells us something that machine learning should not need to rediscover.

For many physical quantities,

the behavior under geometric transformations is already known.

For example,

the total energy of a crystal is unchanged under rotation.

The force acting on an atom rotates together with the crystal.

The stress tensor transforms according to well-defined mathematical rules.

Rather than asking a neural network to discover these relationships from millions of examples,

we can build them directly into the network architecture.

This idea is the foundation of **equivariant neural networks**.

---

# 20.6 From Data-Driven Learning to Physics-Informed Learning

The evolution of machine learning for materials science can be viewed as a gradual incorporation of physical knowledge.

Initially,

models relied almost entirely on statistical learning.

```text
Data

↓

Machine Learning

↓

Prediction
```

Modern approaches integrate both data and physical principles.

```text
Data

+

Physics

↓

Machine Learning

↓

Prediction
```

By embedding physical symmetries directly into the neural network,

the model becomes both more accurate and more efficient.

---

# 20.7 The Importance of Geometry

Many material properties depend strongly on the three-dimensional arrangement of atoms.

Consider two silicon atoms bonded to oxygen.

```text
O

 \

  Si

 /

O
```

If the bond angle changes,

the

* orbital overlap,
* bond strength,
* local strain,
* electronic structure

also change.

A graph neural network must therefore understand not only which atoms are neighbors, but also **how they are arranged in space**.

Equivariant neural networks accomplish this by treating geometry as a fundamental part of the learning process.

---

# 20.8 Why Coordinates Alone Are Not Enough

Coordinates describe the positions of atoms,

but physical interactions depend on geometric relationships such as

* distances,
* angles,
* orientations,
* relative positions.

Two crystals may contain identical coordinates expressed in different coordinate systems.

A physically meaningful model should recognize these structures as equivalent.

Equivariant architectures achieve this automatically.

---

# 20.9 The Role of Symmetry in Materials Science

Symmetry plays a central role throughout materials science.

Examples include

* crystal structures,
* lattice vibrations,
* phonons,
* electronic wavefunctions,
* molecular orbitals,
* diffraction,
* elasticity,
* magnetism.

Many of the governing equations in physics remain unchanged under symmetry transformations.

Consequently,

machine learning models that respect these symmetries naturally produce more physically consistent predictions.

---

# 20.10 Equivariant Graph Neural Networks

Equivariant Graph Neural Networks (EGNNs) are graph neural networks whose internal operations obey the symmetry transformations of three-dimensional space.

Instead of learning rotational behavior from data,

they are mathematically constructed so that rotations, translations, and other geometric transformations are handled correctly by design.

This means that

* scalar quantities remain unchanged,
* vector quantities rotate correctly,
* tensor quantities transform according to physical laws.

These properties dramatically improve predictive performance for atomistic systems.

---

# 20.11 Advantages Over Conventional Graph Neural Networks

Compared with earlier graph neural networks,

equivariant architectures provide several important benefits.

They generally require

* fewer training samples,
* fewer training epochs,
* less data augmentation.

They also produce

* more accurate energies,
* more accurate atomic forces,
* more stable molecular dynamics,
* better generalization to unseen structures.

Because the network already understands geometric symmetry,

it can devote more of its learning capacity to modeling chemistry rather than rediscovering physics.

---

# 20.12 Modern Equivariant Models

Several state-of-the-art machine learning force fields are built upon equivariant neural networks.

Among the most influential are

* **e3nn**,
* **NequIP**,
* **Allegro**,
* **MACE**,
* **PaiNN**,
* **TorchMD-NET**.

These models have achieved near-DFT accuracy for a wide variety of materials while enabling simulations that are several orders of magnitude faster than first-principles calculations.

Many of today's most advanced atomistic machine learning methods are based on these architectures.

---

# 20.13 Applications in Materials Science

Equivariant graph neural networks are now being applied to numerous research areas, including

* crystal structure prediction,
* interatomic potentials,
* molecular dynamics,
* battery materials,
* catalytic surfaces,
* defect migration,
* grain boundaries,
* amorphous materials,
* phase transitions,
* phonon calculations,
* diffusion,
* adsorption,
* nanomaterials.

Their ability to accurately model three-dimensional atomic interactions has made them one of the fastest-growing areas of materials informatics.

---

# 20.14 Roadmap of This Chapter

This chapter develops equivariant graph neural networks from first principles.

We begin by examining why rotational symmetry is fundamental in physics and materials science. We then introduce the concepts of **invariance** and **equivariance**, followed by the mathematical foundations of Euclidean symmetry, group theory, the symmetry groups **SO(3)**, **O(3)**, and **E(3)**. Building on these ideas, we study tensor features, spherical harmonics, and irreducible representations, which provide the mathematical language for modern equivariant neural networks.

With these foundations established, we explore **equivariant message passing**, the **e3nn framework**, and the architectures of **NequIP**, **Allegro**, and **MACE**. Finally, we examine their applications in materials science and compare the strengths and limitations of today's leading equivariant graph neural networks.

By the end of this chapter, you will understand both the mathematical principles and practical implementations of the models that represent the current state of the art in machine learning for atomistic simulations.

---

## Transition to Section 20.2

Before introducing concepts such as **SO(3)** or **tensor representations**, we must first understand **why rotational symmetry matters**. The next section examines how physical quantities respond to rotations, why conventional neural networks struggle with these transformations, and how respecting symmetry fundamentally changes the way machine learning models represent materials.


# 20.15 Motivation

Machine learning has transformed the way researchers discover and understand materials. Instead of performing expensive quantum mechanical calculations for every candidate material, modern machine learning models can predict important properties in a fraction of the time.

However, speed alone is not sufficient.

A machine learning model used in physics must also obey the fundamental laws of nature. If the model violates basic physical principles, its predictions may be numerically accurate for the training data but unreliable when applied to new materials.

This realization motivates the development of **Equivariant Graph Neural Networks (EGNNs)**.

---

# 20.16 Machine Learning Should Respect Physics

Traditional machine learning focuses primarily on finding statistical patterns.

Suppose we have a dataset of crystal structures and their corresponding formation energies.

A conventional neural network attempts to learn the mapping

```text id="mot1"
Crystal Structure

↓

Neural Network

↓

Formation Energy
```

The network has no prior knowledge of

* Newton's laws,
* rotational symmetry,
* crystal symmetry,
* conservation laws.

Everything must be learned from the data.

While this approach is powerful, it is also inefficient because many physical relationships are already known.

---

# 20.17 The Role of Prior Knowledge

Humans rarely learn from raw observations alone.

A physics student does not rediscover Newton's laws by repeatedly dropping objects.

Instead, they begin with established physical principles and then apply them to new problems.

Machine learning models can benefit from the same philosophy.

Instead of asking a neural network to rediscover symmetry from millions of training examples, we can incorporate symmetry directly into the model architecture.

This approach is known as **physics-informed machine learning**.

---

# 20.18 A Simple Thought Experiment

Imagine two researchers studying the same crystal.

The first researcher views the crystal in its original orientation.

```text id="mot2"
      z

      ↑

      ●

     /|

    / |

x ←───→ y
```

The second researcher rotates the crystal by 90°.

```text id="mot3"
      y

      ↑

      ●

     /|

    / |

z ←───→ x
```

Although their coordinate systems differ,

they are studying exactly the same material.

A physically correct machine learning model should therefore produce identical predictions for quantities such as

* total energy,
* formation energy,
* band gap,
* density.

---

# 20.19 Why This Is Difficult for Ordinary Neural Networks

A conventional neural network receives numerical coordinates.

For example,

```text id="mot4"
Atom

(2.1, 1.8, 0.5)
```

After rotating the crystal,

the coordinates become

```text id="mot5"
Atom

(-1.8, 2.1, 0.5)
```

Although these coordinates represent the same physical atom,

they appear completely different numerically.

The neural network cannot automatically recognize that both coordinate sets describe the same material.

---

# 20.20 Learning Every Possible Rotation

One possible solution is to include many rotated copies of every crystal in the training dataset.

For example,

```text id="mot6"
Original

↓

Rotate 15°

↓

Rotate 30°

↓

Rotate 45°

↓

Rotate 90°

↓

Rotate 180°
```

This process is known as **data augmentation**.

Although helpful,

it has significant disadvantages.

* The dataset becomes much larger.
* Training takes longer.
* Memory requirements increase.
* The model still has no mathematical guarantee of rotational consistency.

---

# 20.21 A Better Solution

Instead of teaching the neural network every possible rotation,

we can build rotational symmetry directly into the model.

Conceptually,

```text id="mot7"
Rotation

↓

Network Knows Rotation

↓

Correct Prediction
```

The network no longer needs thousands of rotated examples.

It already understands how physical quantities should transform.

This dramatically improves learning efficiency.

---

# 20.22 Symmetry Reduces Learning Complexity

Suppose a conventional neural network must learn

```text id="mot8"
Original Crystal

↓

Prediction

Rotated Crystal

↓

Prediction

Translated Crystal

↓

Prediction

Reflected Crystal

↓

Prediction
```

Every transformation must be learned independently.

An equivariant network instead learns

```text id="mot9"
Crystal

↓

Symmetry-Aware Network

↓

All Transformations Handled Automatically
```

The learning problem becomes much simpler.

---

# 20.23 Physical Laws Are Coordinate Independent

One of the central ideas in theoretical physics is that physical laws should not depend on the observer's coordinate system.

For example,

Newton's laws,

Maxwell's equations,

and the Schrödinger equation

remain valid regardless of how we orient our coordinate axes.

Similarly,

a machine learning model should produce predictions that are independent of arbitrary coordinate choices whenever the underlying physics requires it.

---

# 20.24 Example: Energy

Suppose the energy of a crystal is

```text id="mot10"
-8.42 eV
```

After rotating the crystal,

the energy should still be

```text id="mot11"
-8.42 eV
```

because energy is a **scalar quantity**.

Its value does not depend on orientation.

---

# 20.25 Example: Force

Now consider the force acting on one atom.

Initially,

```text id="mot12"
F = (2, 0, 0)
```

After rotating the crystal by 90°,

the force should become

```text id="mot13"
F = (0, 2, 0)
```

The force changes direction,

but not magnitude.

Therefore,

force should **rotate together with the crystal**.

---

# 20.26 Example: Stress

The stress tensor behaves differently.

Instead of remaining unchanged or simply rotating like a vector,

the stress transforms according to tensor transformation rules.

Equivariant neural networks are specifically designed to handle these different transformation behaviors correctly.

---

# 20.27 Data Efficiency

One of the greatest practical advantages of equivariant neural networks is improved data efficiency.

Suppose two models are trained on the same dataset.

```text id="mot14"
Conventional GNN

↓

100,000 Structures

↓

Good Accuracy
```

An equivariant model may achieve similar or better accuracy using

```text id="mot15"
Equivariant GNN

↓

20,000 Structures

↓

Good Accuracy
```

Because the network already understands symmetry,

it requires fewer examples to learn the underlying physics.

---

# 20.28 Better Generalization

Generalization refers to the ability of a machine learning model to make accurate predictions on previously unseen data.

Equivariant networks often generalize better because they learn physical relationships rather than memorizing coordinate patterns.

As a result,

they are more robust when applied to new crystal structures.

---

# 20.29 Improved Molecular Dynamics

Accurate molecular dynamics requires precise force predictions.

Even small force errors accumulate over thousands of simulation steps.

Because equivariant neural networks naturally preserve rotational behavior,

they typically produce

* smoother forces,
* more stable trajectories,
* improved long-term simulations.

This makes them particularly attractive for machine-learning interatomic potentials.

---

# 20.30 Why Modern Force Fields Are Equivariant

Nearly every state-of-the-art machine-learning force field developed in recent years uses equivariant neural networks.

Examples include

* NequIP,
* Allegro,
* MACE,
* PaiNN,
* TorchMD-NET.

The widespread adoption of equivariance reflects its ability to combine

* physical consistency,
* computational efficiency,
* high predictive accuracy.

---

# 20.31 Motivation Summary

The motivation for equivariant graph neural networks can be summarized as

```text id="mot16"
Physics

↓

Symmetry

↓

Equivariance

↓

Better Learning

↓

Higher Accuracy

↓

Improved Generalization
```

Rather than forcing a neural network to rediscover the geometric laws of nature from data, equivariant architectures encode these laws directly into the learning process. This leads to models that are more efficient, more physically meaningful, and better suited for modern materials informatics.

---

## Transition to Section 20.3

The central idea behind equivariant neural networks is that **rotations matter**. Before defining concepts such as invariance or equivariance mathematically, we must first understand **why rotation symmetry is one of the most fundamental principles in physics**. In the next section, we will examine how rotating a crystal affects different physical quantities and why respecting rotational symmetry dramatically improves machine learning models for atomistic systems.

# 20.32 Why Rotation Symmetry Matters

Among all geometric transformations encountered in physics,

**rotation** is perhaps the most fundamental.

Every day, objects around us undergo rotations.

A crystal may be rotated under a microscope.

A molecule may rotate in space.

A protein may adopt a different orientation.

A battery material may be viewed from another crystallographic direction.

Although the orientation changes,

the material itself remains exactly the same.

This simple observation has profound consequences for machine learning.

---

# 20.33 Rotating an Object Does Not Change the Object

Consider a sodium chloride crystal.

Initially,

```text id="rot1"
Na   Cl

Cl   Na
```

Now rotate the crystal by

90°

```text id="rot2"
Cl   Na

Na   Cl
```

The orientation has changed.

However,

nothing about the crystal itself has changed.

It still has

* the same composition,
* the same crystal structure,
* the same lattice,
* the same chemical bonding,
* the same energy.

The crystal has merely been viewed from a different direction.

---

# 20.34 A Coordinate System Is Human-Made

Coordinates are not physical objects.

They are simply a way for humans to describe positions.

For example,

an atom located at

```text id="rot3"
(2, 1, 0)
```

may become

```text id="rot4"
(-1, 2, 0)
```

after rotating the coordinate system.

The atom has not moved.

Only the description has changed.

A physically meaningful machine learning model should recognize that these two coordinate sets describe exactly the same physical atom.

---

# 20.35 Physical Properties Should Behave Correctly

Different physical quantities respond differently to rotation.

Some remain unchanged.

Others rotate together with the crystal.

Still others follow tensor transformation rules.

Understanding these behaviors is the foundation of equivariant learning.

---

# 20.36 Scalar Quantities

A **scalar** has magnitude but no direction.

Examples include

* total energy,
* formation energy,
* temperature,
* mass,
* density.

If a crystal is rotated,

these quantities remain identical.

For example,

```text id="rot5"
Original Energy

↓

-7.84 eV
```

After rotation,

```text id="rot6"
Rotated Energy

↓

-7.84 eV
```

Nothing changes.

---

# 20.37 Why Energy Cannot Depend on Orientation

Suppose a neural network predicts

```text id="rot7"
Original

↓

Energy = -7.84 eV
```

but after rotating the crystal,

predicts

```text id="rot8"
Rotated

↓

Energy = -7.21 eV
```

This result is physically impossible.

The crystal has not changed.

Only the observer's viewpoint has changed.

Such a model would violate a fundamental principle of physics.

---

# 20.38 Vector Quantities

Vectors possess both

* magnitude,
* direction.

Examples include

* force,
* velocity,
* acceleration,
* electric field.

Unlike energy,

vectors **must rotate together with the crystal**.

---

# 20.39 Example: Atomic Force

Suppose an atom experiences the force

```text id="rot9"
F

→
```

If the crystal is rotated by

90°,

the force should become

```text id="rot10"
F

↑
```

The force direction changes,

but the physical interaction remains identical.

---

# 20.40 Mathematical Example

Assume the force acting on an atom is

$$
\mathbf{F}
==========

(2,0,0).
$$

Rotating the crystal by

90°

around the z-axis produces

$$
\mathbf{F}'
===========

(0,2,0).
$$

Notice that

* the magnitude remains unchanged,
* only the direction changes.

This is the correct physical behavior.

---

# 20.41 Tensor Quantities

Some physical quantities are neither scalars nor vectors.

Instead,

they are **tensors**.

Examples include

* stress,
* strain,
* dielectric tensor,
* elastic tensor.

Tensors transform according to more complex mathematical rules under rotation.

Modern equivariant neural networks are specifically designed to handle these transformations correctly.

---

# 20.42 Crystal Orientation in Materials Science

In crystallography,

the same crystal can be viewed along different directions.

For example,

```text id="rot11"
[100]

↓

[110]

↓

[111]
```

These orientations are different descriptions of the same crystal.

The material properties remain unchanged,

although the observed geometry appears different.

---

# 20.43 Rotation in Molecular Dynamics

During molecular dynamics,

atoms continuously move,

and molecules often rotate.

For example,

```text id="rot12"
Time

↓

Rotate

↓

Rotate

↓

Rotate
```

The molecular orientation changes continuously.

A machine learning force field must therefore produce physically consistent predictions regardless of orientation.

---

# 20.44 Why Conventional Networks Fail

Suppose we train a conventional neural network only on crystals aligned along one orientation.

When presented with a rotated crystal,

the network may incorrectly interpret it as a completely new structure.

Consequently,

its predictions may become less accurate.

This lack of rotational awareness limits generalization.

---

# 20.45 Rotational Data Augmentation

One possible solution is to augment the dataset.

```text id="rot13"
Crystal

↓

Rotate

↓

Rotate

↓

Rotate

↓

Training
```

Although this improves robustness,

it increases

* dataset size,
* computational cost,
* training time.

Moreover,

perfect rotational consistency is still not guaranteed.

---

# 20.46 Equivariant Networks Solve the Problem

Equivariant neural networks do not require the network to learn rotational behavior from examples.

Instead,

rotation symmetry is built directly into the architecture.

Conceptually,

```text id="rot14"
Crystal

↓

Rotate

↓

Equivariant Layer

↓

Correct Prediction
```

The network automatically transforms its internal features according to the rotation.

---

# 20.47 Learning Physics Instead of Coordinates

Traditional neural networks often learn

```text id="rot15"
Coordinates

↓

Patterns
```

Equivariant networks instead learn

```text id="rot16"
Geometry

↓

Physics
```

This distinction is one of the major reasons why equivariant architectures outperform conventional graph neural networks in atomistic simulations.

---

# 20.48 Rotation Symmetry Improves Generalization

Suppose the training data contains crystals rotated only between

0°

and

45°.

An ordinary neural network may struggle when encountering a crystal rotated by

120°.

An equivariant network,

however,

already understands rotational transformations mathematically.

Consequently,

it can generalize naturally to arbitrary orientations.

---

# 20.49 Why Rotation Symmetry Improves Data Efficiency

Because rotational behavior is built into the architecture,

the neural network does not waste capacity learning identical physical situations expressed in different coordinate systems.

Instead,

it can devote its parameters to learning

* bonding,
* chemical interactions,
* electronic effects,
* many-body physics.

This improves learning efficiency while reducing the amount of required training data.

---

# 20.50 Real-World Importance

Rotation symmetry is essential for

* crystal structure prediction,
* interatomic potentials,
* molecular dynamics,
* catalyst simulations,
* protein modeling,
* molecular chemistry,
* battery materials,
* nanomaterials.

Virtually every modern atomistic machine learning model benefits from respecting rotational symmetry.

---

# 20.51 Key Insight

The most important idea of this section can be summarized as

```text id="rot17"
Rotate Crystal

↓

Material Does Not Change

↓

Prediction Must Behave Correctly
```

The neural network should not memorize individual coordinate systems.

Instead,

it should understand the underlying physical object independent of how it is oriented.

This insight motivates the concepts of **invariance** and **equivariance**, which provide the mathematical framework for describing how different physical quantities transform under symmetry operations.

---

## Transition to Section 20.4

We have established that different physical quantities respond differently to rotations. **Energy remains unchanged**, **forces rotate**, and **stress transforms as a tensor**. To formalize these behaviors, we now introduce two fundamental concepts in geometric deep learning: **invariance** and **equivariance**. These concepts define exactly how a neural network's outputs should respond when its inputs undergo symmetry transformations, and they form the mathematical foundation of every modern equivariant graph neural network.

# 20.52 Invariance vs. Equivariance

The concepts of **invariance** and **equivariance** are the mathematical foundation of modern geometric deep learning.

Nearly every modern machine learning potential—including **NequIP**, **Allegro**, **MACE**, **PaiNN**, and the **e3nn** framework—is built upon these ideas.

Although the two terms sound similar, they describe fundamentally different behaviors under geometric transformations.

Understanding this distinction is essential before studying equivariant message passing or tensor features.

---

# 20.53 Why We Need New Mathematical Concepts

Suppose we rotate an entire crystal.

Different physical quantities respond differently.

For example,

| Quantity         | After Rotation |
| ---------------- | -------------- |
| Energy           | Unchanged      |
| Formation Energy | Unchanged      |
| Atomic Position  | Rotates        |
| Force            | Rotates        |
| Velocity         | Rotates        |
| Stress           | Transforms     |

Clearly,

not every physical quantity behaves in the same way.

Therefore,

a machine learning model should not treat every prediction identically.

This observation leads naturally to the concepts of **invariance** and **equivariance**.

---

# 20.54 What Is Invariance?

A function is called **invariant** if its output does **not change** after applying a transformation to the input.

Conceptually,

```text id="inv1"
Input

↓

Transformation

↓

Neural Network

↓

Same Output
```

No matter how the input is rotated,

translated,

or reflected,

the prediction remains identical.

---

# 20.55 Mathematical Definition of Invariance

Suppose

* (x) is the input,
* (T) is a transformation,
* (f) is a neural network.

The function is invariant if

[
f(Tx)=f(x).
]

This simple equation states

> Applying the transformation before the neural network gives exactly the same prediction as using the original input.

---

# 20.56 Physical Interpretation

Imagine rotating a crystal.

```text id="inv2"
Crystal

↓

Rotate

↓

Same Material
```

The total energy should remain

```text id="inv3"
Original

↓

−8.35 eV
```

After rotation,

```text id="inv4"
Rotated

↓

−8.35 eV
```

The prediction is identical.

Energy is therefore an **invariant quantity**.

---

# 20.57 Examples of Invariant Quantities

Many important material properties are invariant.

Examples include

* total energy,
* formation energy,
* band gap,
* density,
* mass,
* temperature,
* composition,
* crystal formula.

These quantities do not depend on the orientation of the coordinate system.

---

# 20.58 Example: Formation Energy

Suppose the formation energy of a crystal is

```text id="inv5"
−2.41 eV/atom
```

Rotate the crystal.

The formation energy is still

```text id="inv6"
−2.41 eV/atom
```

Nothing changes because formation energy is a scalar.

---

# 20.59 What Is Equivariance?

Equivariance is different.

Instead of remaining unchanged,

the output transforms **in exactly the same way** as the input.

Conceptually,

```text id="eq1"
Input

↓

Rotate

↓

Neural Network

↓

Output Rotates
```

The prediction changes,

but it changes correctly.

---

# 20.60 Mathematical Definition of Equivariance

A function is equivariant if

[
f(Tx)=Tf(x).
]

Notice the difference.

For invariance,

the transformation disappears.

For equivariance,

the same transformation appears on both sides.

This equation means

> Rotating the input is equivalent to rotating the output.

---

# 20.61 Physical Interpretation

Suppose an atom experiences a force pointing toward the right.

```text id="eq2"
Force

→
```

Rotate the crystal by

90°.

The force should now point upward.

```text id="eq3"
Force

↑
```

The force changes,

but it changes exactly as dictated by the rotation.

This is equivariance.

---

# 20.62 Example Using Coordinates

Suppose an atom is located at

[
(2,1,0).
]

After a 90° rotation around the z-axis,

the coordinates become

[
(-1,2,0).
]

A coordinate prediction should rotate exactly the same way.

Coordinates are therefore **equivariant**.

---

# 20.63 Example Using Force

Suppose

[
\mathbf{F}
==========

(3,0,0).
]

After rotation,

[
\mathbf{F}'
===========

(0,3,0).
]

Notice that

* the magnitude remains unchanged,
* the direction rotates.

The neural network should produce this transformed vector automatically.

---

# 20.64 Example Using Velocity

Velocity behaves exactly like force.

Original velocity

```text id="eq4"
→
```

Rotated crystal

```text id="eq5"
↑
```

The velocity rotates together with the coordinate system.

Velocity is therefore another equivariant quantity.

---

# 20.65 Example Using Stress

Stress is more complicated.

Stress is not a scalar.

It is also not a simple vector.

Instead,

stress is a **second-order tensor**.

When the crystal rotates,

the stress tensor transforms according to tensor transformation rules.

Equivariant neural networks are capable of learning these tensor transformations naturally.

---

# 20.66 Comparing Invariance and Equivariance

The difference between the two concepts can be summarized visually.

```text id="compare1"
INVARIANCE

Rotate Input

↓

Output Unchanged
```

```text id="compare2"
EQUIVARIANCE

Rotate Input

↓

Output Rotates
```

The input undergoes the same transformation in both cases,

but the outputs behave differently.

---

# 20.67 Everyday Analogy

Imagine taking a photograph of a coffee mug.

Rotate the mug.

The **weight** of the mug remains identical.

Weight is analogous to an invariant quantity.

Now consider the **handle direction**.

After rotating the mug,

the handle points in a different direction.

The handle orientation is analogous to an equivariant quantity.

The object itself has not changed,

only its orientation.

---

# 20.68 Why Machine Learning Needs Both

Different prediction tasks require different symmetry behavior.

If we predict

* total energy,

the model should be invariant.

If we predict

* atomic forces,

the model should be equivariant.

Therefore,

modern atomistic neural networks often contain both invariant and equivariant features simultaneously.

---

# 20.69 Internal Feature Transformations

Interestingly,

many equivariant neural networks produce invariant outputs by first learning equivariant internal representations.

Conceptually,

```text id="eq6"
Coordinates

↓

Equivariant Features

↓

Invariant Energy
```

This strategy allows the model to fully exploit geometric information while producing physically correct predictions.

---

# 20.70 Why Equivariance Improves Learning

Without equivariance,

the neural network must independently learn every possible orientation.

With equivariance,

all orientations are automatically related.

Consequently,

the model requires

* fewer parameters,
* less data,
* fewer training epochs.

This is one reason why modern equivariant architectures outperform earlier graph neural networks.

---

# 20.71 Examples in Materials Science

The concepts of invariance and equivariance appear throughout computational materials science.

| Quantity           | Symmetry Behavior  |
| ------------------ | ------------------ |
| Total Energy       | Invariant          |
| Formation Energy   | Invariant          |
| Band Gap           | Invariant          |
| Force              | Equivariant        |
| Velocity           | Equivariant        |
| Atomic Coordinates | Equivariant        |
| Stress Tensor      | Tensor Equivariant |
| Dipole Moment      | Equivariant        |

Understanding these transformation rules is essential when designing physically meaningful neural networks.

---

# 20.72 Why This Matters for Modern GNNs

Architectures such as

* NequIP,
* Allegro,
* MACE,
* PaiNN,
* e3nn

are all designed around equivariant operations.

Instead of merely predicting correct numerical values,

they also ensure that every prediction transforms according to the underlying symmetry of three-dimensional space.

This capability is one of the primary reasons they have achieved state-of-the-art performance in atomistic machine learning.

---

# 20.73 Key Takeaways

The distinction between invariance and equivariance can be summarized as

```text id="summary_eq"

INVARIANCE

Rotate Input

↓

Prediction Does Not Change

(Energy)


EQUIVARIANCE

Rotate Input

↓

Prediction Rotates

(Force)
```

Both concepts are indispensable.

Invariance guarantees that scalar physical quantities remain independent of the observer's coordinate system, while equivariance ensures that vectors and tensors transform exactly as required by the laws of physics.

Together, they form the mathematical language that allows modern graph neural networks to respect the geometric structure of the physical world.

---

## Transition to Section 20.5

So far, we have discussed symmetry conceptually using rotations of crystals and vectors. The next step is to formalize these ideas within the broader framework of **Euclidean symmetry**, which encompasses **translations, rotations, and reflections**. Understanding Euclidean symmetry provides the bridge from intuitive geometric transformations to the mathematical groups **SO(3)**, **O(3)**, and **E(3)** that underpin modern equivariant graph neural networks.

# 20.74 Euclidean Symmetry

The concepts of invariance and equivariance describe **how** physical quantities should respond to transformations.

The next question is

> **What transformations are physically meaningful?**

In materials science, molecules and crystals exist in ordinary three-dimensional space.

The geometry of this space is called **Euclidean geometry**, and the transformations that preserve this geometry are known as **Euclidean symmetries**.

Nearly every modern equivariant graph neural network is designed around these symmetries.

---

# 20.75 What Is Euclidean Space?

Euclidean space is the familiar three-dimensional space in which we live.

Every atom in a crystal can be represented by coordinates

[
(x,y,z).
]

For example,

```text id="euclid1"
Atom A

(1.2, 0.8, 2.5)
```

and

```text id="euclid2"
Atom B

(2.7, 1.1, 4.0)
```

Together, these coordinates describe the geometry of the crystal.

---

# 20.76 Geometry Is More Than Coordinates

Coordinates themselves are not physically important.

What matters are the **geometric relationships** between atoms.

Examples include

* distances,
* bond lengths,
* bond angles,
* relative orientations,
* neighbor relationships.

A physically meaningful transformation should preserve these relationships.

---

# 20.77 Distance Preservation

Consider two atoms.

```text id="euclid3"
A ●────────● B

Distance = d
```

Rotate the entire crystal.

```text id="euclid4"
      ● A

     /

    /

● B
```

Although the coordinates have changed,

the distance

[
d
]

remains exactly the same.

Transformations that preserve distances are called **isometries**.

---

# 20.78 Euclidean Transformations

There are three fundamental Euclidean transformations.

They are

* translation,
* rotation,
* reflection.

Each transformation changes the coordinate description while preserving the physical object.

---

# 20.79 Translation

A translation shifts every atom by the same displacement.

Suppose every atom moves

```text id="euclid5"
+2 Å

along x
```

Initially,

```text id="euclid6"
●────●────●
```

After translation,

```text id="euclid7"
      ●────●────●
```

Nothing about the crystal has changed.

Only its position has changed.

---

# 20.80 Mathematical Representation of Translation

Suppose an atom has coordinates

[
\mathbf{r}.
]

After translation by vector

[
\mathbf{t},
]

the new position becomes

[
\mathbf{r}'
===========

\mathbf{r}
+
\mathbf{t}.
]

Every atom receives exactly the same displacement.

---

# 20.81 Why Translation Should Not Affect Predictions

Suppose the total energy of a crystal is

```text id="euclid8"
−9.15 eV
```

Move the entire crystal by

100 Å.

The energy remains

```text id="euclid9"
−9.15 eV
```

Energy is therefore **translation invariant**.

---

# 20.82 Rotation

A rotation changes the orientation of an object while preserving

* distances,
* angles,
* connectivity.

For example,

```text id="euclid10"
Before

→
```

After rotating by

90°

```text id="euclid11"
↑
```

The object is identical,

only its orientation changes.

---

# 20.83 Reflection

Reflection creates a mirror image.

```text id="euclid12"
Original

A B C
```

Mirror reflection

```text id="euclid13"
C B A
```

Many physical systems remain valid after reflection,

although some quantities—such as molecular chirality—can distinguish between an object and its mirror image.

---

# 20.84 Euclidean Symmetry Preserves Geometry

All Euclidean transformations preserve

* distances,
* bond lengths,
* bond angles,
* neighbor topology.

Therefore,

the physical identity of the crystal remains unchanged.

Only the coordinate description changes.

---

# 20.85 What Does Not Preserve Euclidean Geometry?

Suppose we stretch a crystal.

```text id="euclid14"
Before

●──●
```

Stretch

```text id="euclid15"
●──────●
```

The bond length has changed.

This is **not** a Euclidean symmetry.

Similarly,

compressing,

shearing,

or distorting a crystal changes its physical structure.

These are genuine physical changes,

not merely changes of viewpoint.

---

# 20.86 Why Machine Learning Should Respect Euclidean Symmetry

Imagine two identical crystals.

One is located at

```text id="euclid16"
Origin
```

The other is translated by

```text id="euclid17"
100 Å
```

A conventional neural network may see two different coordinate sets.

An equivariant network recognizes that the two crystals are physically identical.

This greatly improves learning efficiency.

---

# 20.87 Euclidean Symmetry in Crystal Structures

Crystal structures are naturally described in Euclidean space.

Each atom possesses

* Cartesian coordinates,
* neighboring atoms,
* bond directions.

All geometric relationships are measured using Euclidean geometry.

Therefore,

equivariant graph neural networks operate directly within Euclidean space rather than treating coordinates as arbitrary numerical values.

---

# 20.88 Euclidean Symmetry in Molecular Dynamics

During molecular dynamics,

atoms continuously translate and rotate.

For example,

```text id="euclid18"
Time

↓

Move

↓

Rotate

↓

Move

↓

Rotate
```

Although coordinates change every time step,

the governing physical laws remain unchanged.

Equivariant neural networks naturally respect these transformations throughout the simulation.

---

# 20.89 Local Atomic Environments

Most interatomic interactions depend on the **relative positions** of neighboring atoms rather than their absolute coordinates.

For example,

consider a central atom.

```text id="euclid19"
      O

      |

O — Si — O

      |

      O
```

If the entire local environment is translated,

the bonding remains identical.

If it is rotated,

the local chemistry also remains identical.

This observation motivates the use of **relative coordinates** in modern atomistic neural networks.

---

# 20.90 Euclidean Symmetry and Graph Neural Networks

Modern graph neural networks represent atoms as graph nodes.

However,

ordinary graph connectivity alone does not encode three-dimensional geometry.

Equivariant graph neural networks enrich each graph with

* coordinates,
* relative vectors,
* distances,
* angular information,

while ensuring that every operation respects Euclidean symmetry.

---

# 20.91 From Euclidean Symmetry to Symmetry Groups

Mathematicians describe Euclidean transformations using **groups**.

The most important symmetry groups for atomistic machine learning are

* **SO(3)** — rotations,
* **O(3)** — rotations and reflections,
* **E(3)** — translations, rotations, and reflections.

These groups provide the mathematical framework underlying equivariant neural networks.

The next sections of this chapter examine each of these groups in detail.

---

# 20.92 Why Euclidean Symmetry Matters

The importance of Euclidean symmetry can be summarized as

```text id="euclid20"
Physical System

↓

Translation

↓

Rotation

↓

Reflection

↓

Same Physical Laws
```

A neural network that respects these transformations is inherently more consistent with the geometry of the physical world.

Instead of memorizing coordinate systems, it learns the underlying physical relationships between atoms.

---

# 20.93 Key Takeaways

Euclidean symmetry is the geometric foundation of modern atomistic machine learning.

It tells us that physically meaningful models should preserve the essential geometric relationships between atoms under translations, rotations, and reflections.

This principle forms the bridge between intuitive geometric reasoning and the mathematical symmetry groups that power state-of-the-art equivariant graph neural networks.

---

## Transition to Section 20.6

We have seen that Euclidean symmetry consists of translations, rotations, and reflections. To describe these transformations rigorously, mathematicians use **group theory**. Before studying the specific symmetry groups **SO(3)**, **O(3)**, and **E(3)**, we must first understand what a **group** is, why symmetry operations form groups, and how group theory provides the mathematical language for equivariant neural networks.

# 20.94 Group Theory Basics

Group theory is one of the most important mathematical tools in modern physics, chemistry, crystallography, and machine learning.

Although the term **group theory** may initially sound abstract, its central idea is remarkably simple:

> **A group is a collection of transformations that can be combined while preserving certain mathematical rules.**

In equivariant graph neural networks, rotations, reflections, and translations are not treated as isolated operations. Instead, they are viewed as members of mathematical groups whose properties guarantee physically consistent transformations.

Understanding group theory allows us to describe symmetry precisely rather than relying on intuition alone.

---

# 20.95 Why Do We Need Group Theory?

Suppose we rotate a crystal.

Then rotate it again.

The result is simply another rotation.

For example,

```text id="group1"
Rotate 30°

↓

Rotate 60°

↓

Rotate 90°
```

Instead of considering these as separate actions,

mathematics treats them as members of a single structured system.

That system is called a **group**.

---

# 20.96 What Is a Mathematical Group?

A **group** is a set of elements together with an operation that satisfies four fundamental properties.

These properties are

1. Closure
2. Identity
3. Inverse
4. Associativity

Every symmetry group used in physics satisfies these four requirements.

---

# 20.97 Property 1: Closure

Closure means that combining any two elements of the group always produces another element belonging to the same group.

For rotations,

```text id="group2"
Rotate 20°

+

Rotate 40°

↓

Rotate 60°
```

The result is still a rotation.

Therefore,

rotations satisfy closure.

---

# 20.98 Why Closure Matters

Imagine a neural network processing multiple geometric transformations.

If applying two valid transformations produced an invalid operation,

the mathematical framework would break down.

Closure guarantees consistency regardless of how many transformations are combined.

---

# 20.99 Property 2: Identity

Every group contains an **identity element**.

The identity transformation changes nothing.

For rotations,

the identity is

```text id="group3"
Rotate 0°
```

Applying it leaves the crystal unchanged.

```text id="group4"
Crystal

↓

Rotate 0°

↓

Same Crystal
```

---

# 20.100 Identity in Neural Networks

Suppose no transformation is applied.

The neural network should produce exactly the same prediction as before.

The existence of an identity transformation ensures that the original configuration is always part of the symmetry group.

---

# 20.101 Property 3: Inverse

Every transformation must have another transformation that reverses its effect.

For example,

```text id="group5"
Rotate 45°

↓

Rotate −45°

↓

Original Orientation
```

The second rotation completely cancels the first.

The two operations are inverses of one another.

---

# 20.102 Physical Meaning of Inverses

Suppose we rotate a crystal.

If we later rotate it back,

the crystal should return exactly to its original orientation.

This simple idea is guaranteed by the inverse property.

---

# 20.103 Property 4: Associativity

Associativity means that the order of grouping operations does not affect the final result.

Mathematically,

[
(A \circ B)\circ C
==================

A\circ(B\circ C).
]

For example,

```text id="group6"
(20° + 30°)

+

40°
```

produces the same final rotation as

```text id="group7"
20°

+

(30° + 40°)
```

The grouping changes,

but the overall transformation does not.

---

# 20.104 Summary of the Four Properties

A mathematical group must satisfy all four properties simultaneously.

| Property      | Meaning                                                  |
| ------------- | -------------------------------------------------------- |
| Closure       | Combining two group elements gives another group element |
| Identity      | A transformation exists that changes nothing             |
| Inverse       | Every transformation can be undone                       |
| Associativity | Grouping operations does not affect the result           |

Without these properties,

the set of transformations cannot be considered a group.

---

# 20.105 Symmetry Operations Form Groups

Many symmetry operations naturally satisfy these properties.

Examples include

* rotations,
* translations,
* reflections,
* permutations.

Consequently,

they can all be studied using group theory.

---

# 20.106 Example: Rotations Around a Circle

Imagine rotating an object around a fixed axis.

```text id="group8"
0°

↓

90°

↓

180°

↓

270°

↓

360°
```

Notice that

360°

returns the object to its original orientation.

Therefore,

the identity transformation naturally appears within the rotation group.

---

# 20.107 Continuous and Discrete Groups

Groups can be divided into two major categories.

### Discrete Groups

These contain a finite or countable number of transformations.

Examples include

* crystal point groups,
* space groups,
* mirror symmetries.

---

### Continuous Groups

These contain infinitely many transformations.

Examples include

* arbitrary rotations,
* arbitrary translations.

Continuous groups play the central role in equivariant neural networks.

---

# 20.108 Why Continuous Groups Matter

Suppose a crystal is rotated by

1°

instead of

90°.

Or

17.6°.

Or

143.2°.

There are infinitely many possible rotation angles.

A neural network cannot memorize every possibility individually.

Instead,

it learns the mathematical rules governing the continuous rotation group.

---

# 20.109 Groups in Physics

Group theory appears throughout modern physics.

Examples include

* rotational symmetry,
* translational symmetry,
* quantum mechanics,
* particle physics,
* crystallography,
* relativity,
* electromagnetism.

Many of the most fundamental equations in physics are written using group-theoretic ideas.

---

# 20.110 Groups in Crystallography

Crystallography relies heavily on symmetry groups.

Examples include

* point groups,
* space groups,
* lattice symmetries,
* screw axes,
* glide planes.

These groups classify crystal structures according to their symmetry properties.

Machine learning models for crystals inherit many of these ideas.

---

# 20.111 Groups in Machine Learning

Modern geometric deep learning extends these concepts into neural networks.

Instead of requiring the model to learn every transformation independently,

group theory provides the mathematical rules that guarantee consistent behavior.

Conceptually,

```text id="group9"
Symmetry

↓

Group Theory

↓

Neural Network

↓

Physically Consistent Prediction
```

---

# 20.112 Why Group Theory Improves Learning

Without symmetry,

a neural network must separately learn

```text id="group10"
Rotate 5°

Rotate 10°

Rotate 20°

Rotate 30°

...
```

With group theory,

the network learns the transformation rule itself.

This dramatically improves

* data efficiency,
* generalization,
* physical consistency.

---

# 20.113 From Groups to Rotation Groups

Not every group is equally important for atomistic machine learning.

The most important are

* **SO(3)**,
* **O(3)**,
* **E(3)**.

Each describes a different collection of symmetry transformations in three-dimensional space.

Understanding these groups is essential for modern equivariant neural networks.

---

# 20.114 Key Insight

The central idea of this section is

```text id="group11"
Transformations

↓

Mathematical Structure

↓

Group

↓

Symmetry

↓

Equivariant Neural Network
```

Group theory provides the rigorous mathematical language needed to describe symmetry transformations. Rather than treating rotations or reflections as isolated operations, it organizes them into well-defined structures with predictable behavior.

This mathematical framework allows equivariant graph neural networks to encode physical symmetries directly into their architecture instead of learning them solely from data.

---

## Transition to Section 20.7

Now that we understand what a mathematical group is, we can study one of the most important symmetry groups in physics: **SO(3)**, the **Special Orthogonal Group in three dimensions**. SO(3) describes every possible rotation in three-dimensional space and forms the mathematical foundation of nearly all modern equivariant neural networks used in atomistic machine learning.

# 20.115 The Special Orthogonal Group SO(3)

Among all symmetry groups encountered in physics, **SO(3)** is arguably the most important for atomistic machine learning.

Every crystal,

every molecule,

and every atom exists in three-dimensional space.

Whenever these systems rotate,

their transformations are described mathematically by the **Special Orthogonal Group**, abbreviated as **SO(3)**.

Nearly every modern equivariant neural network—including **e3nn**, **NequIP**, **Allegro**, and **MACE**—is fundamentally built upon the mathematics of SO(3).

Understanding SO(3) is therefore essential for understanding how these models work.

---

# 20.116 What Does SO(3) Mean?

The notation **SO(3)** has a precise mathematical meaning.

* **S** stands for **Special**.
* **O** stands for **Orthogonal**.
* **3** indicates that the transformations occur in **three-dimensional space**.

Thus,

SO(3) represents the set of all proper rotations in three-dimensional Euclidean space.

---

# 20.117 Why "Orthogonal"?

Consider a rotation matrix

[
R.
]

A rotation should preserve

* distances,
* angles,
* lengths.

For this to happen,

the matrix must satisfy

[
R^{T}R=I,
]

where

* (R^T) is the transpose of the matrix,
* (I) is the identity matrix.

Matrices satisfying this condition are called **orthogonal matrices**.

This property guarantees that rotating a crystal does not stretch or compress it.

---

# 20.118 Why "Special"?

Not every orthogonal matrix represents a proper rotation.

Some orthogonal matrices also include reflections.

To distinguish pure rotations,

SO(3) requires an additional condition:

[
\det(R)=1.
]

The determinant of the rotation matrix must equal **+1**.

This excludes reflections, whose determinant is **−1**.

---

# 20.119 Defining SO(3)

Combining both conditions,

SO(3) is defined as

[
SO(3)=
\left{
R\in\mathbb{R}^{3\times3}
;|;
R^{T}R=I,;
\det(R)=1
\right}.
]

Although this definition appears abstract,

its physical interpretation is simple:

> **SO(3) contains every possible rotation of a rigid object in three-dimensional space.**

---

# 20.120 Examples of SO(3) Rotations

Every rotation belongs to SO(3).

Examples include

* rotating a crystal by 5°,
* rotating by 45°,
* rotating by 90°,
* rotating by 180°,
* rotating by 270°.

Likewise,

rotations around

* the x-axis,
* the y-axis,
* the z-axis,

are all members of SO(3).

---

# 20.121 Infinite Number of Rotations

Unlike crystal point groups,

SO(3) is a **continuous group**.

There are infinitely many possible rotations.

For example,

```text id="so31"
0°

1°

2°

3°

...

89.3°

...

179.6°

...

359.999°
```

Every one of these rotations belongs to SO(3).

---

# 20.122 Rotation Around the x-Axis

A rotation about the x-axis leaves the x-coordinate unchanged while rotating the y–z plane.

```text id="so32"
        z

        ↑

        |

        |

--------●--------→ x

       /

      y
```

This is one of the three fundamental rotation axes.

---

# 20.123 Rotation Around the y-Axis

Similarly,

rotation around the y-axis changes the x and z coordinates while leaving y unchanged.

```text id="so33"
        z

        ↑

       /

      ●

     /

y ↑

     x
```

---

# 20.124 Rotation Around the z-Axis

Rotation around the z-axis changes the x–y plane.

```text id="so34"
        z

        ↑

        ●

      ↺

x ←────→ y
```

This is the most commonly illustrated rotation in introductory examples.

---

# 20.125 Rotation Matrices

Each rotation in SO(3) can be represented by a **3×3 rotation matrix**.

For example,

rotation by an angle (\theta) about the z-axis is represented by

[
R_z(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta & 0\
\sin\theta & \cos\theta & 0\
0 & 0 & 1
\end{bmatrix}.
]

Applying this matrix rotates every point in the crystal.

---

# 20.126 Rotating Coordinates

Suppose an atom has position

[
\mathbf{r}.
]

After rotation,

its new position is

[
\mathbf{r}'
===========

R\mathbf{r}.
]

Here,

(R)

is the appropriate SO(3) rotation matrix.

The crystal itself has not changed.

Only its coordinate description has changed.

---

# 20.127 Rotating Vectors

Vectors transform exactly like coordinates.

If

[
\mathbf{F}
]

is an atomic force,

then after rotation,

[
\mathbf{F}'
===========

R\mathbf{F}.
]

Thus,

forces rotate together with the crystal.

This is precisely the behavior that equivariant neural networks must reproduce.

---

# 20.128 Rotating Entire Crystals

Imagine rotating an entire crystal.

```text id="so35"
Crystal

↓

Rotate

↓

Crystal
```

Although every coordinate changes,

the crystal remains physically identical.

Its

* energy,
* composition,
* bonding,
* electronic structure

remain unchanged.

---

# 20.129 Why SO(3) Matters for Machine Learning

Suppose a neural network predicts the energy of a crystal.

If rotating the crystal changes the predicted energy,

the model violates SO(3) symmetry.

A physically meaningful model should instead satisfy

```text id="so36"
Rotate Crystal

↓

Predict Energy

↓

Same Energy
```

Likewise,

force predictions should rotate exactly with the crystal.

---

# 20.130 SO(3) and Neural Network Features

Modern equivariant neural networks do not treat hidden features as arbitrary numbers.

Instead,

each hidden feature is assigned a specific transformation behavior under SO(3).

Some features behave like

* scalars,
* vectors,
* higher-order tensors.

This allows every intermediate layer to remain mathematically consistent with rotational symmetry.

---

# 20.131 Why Learning SO(3) Is Difficult

Without built-in symmetry,

a conventional neural network must independently learn predictions for every possible orientation.

Because SO(3) contains infinitely many rotations,

this task is essentially impossible using finite training data alone.

Equivariant neural networks overcome this challenge by embedding SO(3) symmetry directly into their architecture.

---

# 20.132 SO(3) in Modern Architectures

Several state-of-the-art models explicitly enforce SO(3) symmetry.

These include

* e3nn,
* NequIP,
* Allegro,
* MACE,
* PaiNN.

Rather than memorizing rotated examples,

these models mathematically guarantee correct rotational behavior.

---

# 20.133 Relationship Between SO(3) and Quantum Mechanics

SO(3) is also fundamental in quantum mechanics.

Angular momentum,

atomic orbitals,

and spherical harmonics are all described using representations of SO(3).

This is one reason why spherical harmonics appear naturally in equivariant graph neural networks.

The mathematical tools developed for quantum mechanics have become essential components of modern geometric deep learning.

---

# 20.134 Key Takeaways

The key properties of SO(3) are summarized below.

| Property             | Description                |
| -------------------- | -------------------------- |
| Space                | Three-dimensional          |
| Transformation       | Proper rotations           |
| Matrix condition     | (R^{T}R=I)                 |
| Determinant          | (+1)                       |
| Preserves            | Distances, angles, lengths |
| Continuous group     | Yes                        |
| Includes reflections | No                         |

SO(3) provides the mathematical description of every proper rotation in three-dimensional space and serves as the rotational symmetry group underlying modern equivariant neural networks.

---

## Transition to Section 20.8

SO(3) describes **proper rotations**, but not every symmetry operation is a proper rotation. Mirror reflections and inversion operations also play important roles in crystallography and materials science. To incorporate these transformations, we must extend SO(3) to the **Orthogonal Group O(3)**, which includes both rotations and reflections. Understanding the distinction between these two groups is essential before introducing the full Euclidean symmetry group **E(3)**.

# 20.135 The Orthogonal Group O(3)

In the previous section, we introduced **SO(3)**, the group of all proper rotations in three-dimensional space.

However, rotations alone do not describe every symmetry encountered in physics and crystallography.

Many materials also exhibit

* mirror symmetry,
* inversion symmetry,
* improper rotations,

which cannot be represented by SO(3).

To include these additional transformations, mathematicians define a larger symmetry group called the **Orthogonal Group**, denoted by **O(3)**.

O(3) contains **every distance-preserving linear transformation in three-dimensional space**, including both rotations and reflections.

---

# 20.136 What Does O(3) Mean?

The notation **O(3)** stands for

* **O** — Orthogonal
* **3** — Three-dimensional space

Unlike SO(3),

the letter **S (Special)** is absent.

This small difference has an important consequence.

O(3) includes

* proper rotations,
* reflections,
* inversion,
* improper rotations.

Thus,

SO(3) is a subset of O(3).

---

# 20.137 Mathematical Definition

An orthogonal matrix satisfies

[
R^{T}R=I.
]

Unlike SO(3),

there is **no restriction** that the determinant must be positive.

Instead,

[
\det(R)=\pm1.
]

Therefore,

O(3) consists of all orthogonal matrices,

whether they represent rotations or reflections.

---

# 20.138 Comparing SO(3) and O(3)

The relationship between the two groups is summarized below.

| Group | Determinant | Contains                  |
| ----- | ----------- | ------------------------- |
| SO(3) | +1          | Proper rotations only     |
| O(3)  | ±1          | Rotations and reflections |

Every element of SO(3) belongs to O(3),

but not every element of O(3) belongs to SO(3).

---

# 20.139 Proper Rotations

A **proper rotation** preserves the orientation of an object.

For example,

```text id="o31"
Rotate 90°

↓

Same Handedness
```

If a right-handed coordinate system is rotated,

it remains right-handed.

Proper rotations therefore belong to SO(3).

---

# 20.140 Reflections

A reflection produces a mirror image.

Imagine reflecting a crystal across a plane.

```text id="o32"
Original

A B C
```

Mirror reflection

```text id="o33"
C B A
```

The geometry is preserved,

but the orientation changes.

Reflections are members of O(3),

not SO(3).

---

# 20.141 Mirror Symmetry

Consider a mirror placed beside a crystal.

```text id="o34"
Crystal

|

Mirror

↓

Mirror Image
```

The reflected crystal has identical bond lengths and bond angles.

Only its handedness has changed.

Because distances are preserved,

reflection belongs to O(3).

---

# 20.142 Inversion Symmetry

Another important transformation is **inversion**.

Under inversion,

every coordinate changes sign.

If

[
(x,y,z)
]

is the original position,

the inverted position becomes

[
(-x,-y,-z).
]

Geometrically,

every point is reflected through the origin.

---

# 20.143 Improper Rotations

An **improper rotation** combines

1. a rotation,
2. a reflection.

Conceptually,

```text id="o35"
Rotate

↓

Reflect

↓

Final Configuration
```

Improper rotations appear frequently in crystallography and molecular symmetry.

---

# 20.144 Orientation and Handedness

One of the key differences between rotations and reflections is **handedness**.

Consider your hands.

Your left hand and right hand are mirror images.

No rotation can transform one into the other.

Only a reflection can.

This change in handedness distinguishes O(3) from SO(3).

---

# 20.145 Chirality

Many molecules are **chiral**.

A chiral molecule cannot be superimposed onto its mirror image.

Famous examples include

* amino acids,
* sugars,
* many pharmaceuticals.

For these systems,

reflection produces a physically distinct structure.

Consequently,

handling O(3) symmetry correctly becomes important.

---

# 20.146 Why Reflections Matter in Materials Science

Reflection symmetry appears throughout materials science.

Examples include

* crystal mirror planes,
* surface symmetries,
* grain boundaries,
* defects,
* lattice inversion centers,
* molecular chirality.

Many crystalline materials possess reflection or inversion symmetries that strongly influence their physical properties.

---

# 20.147 O(3) and Neural Networks

Not every machine-learning model needs full O(3) symmetry.

The required symmetry depends on the prediction task.

For example,

predicting

* total energy,
* atomic forces,
* stress,

often requires rotational symmetry,

while certain problems involving

* chirality,
* magnetic ordering,
* mirror symmetry,

may require the broader O(3) framework.

---

# 20.148 Determinant as a Symmetry Test

A convenient way to distinguish the two groups is by examining the determinant.

If

[
\det(R)=1,
]

the transformation is a proper rotation and belongs to SO(3).

If

[
\det(R)=-1,
]

the transformation includes a reflection and belongs only to O(3).

Thus,

the determinant immediately reveals whether orientation has been preserved.

---

# 20.149 Visualizing the Difference

The distinction can be summarized conceptually.

```text id="o36"
SO(3)

Rotate

↓

Orientation Preserved
```

```text id="o37"
O(3)

Rotate

or

Reflect

↓

Orientation May Change
```

SO(3) is therefore the orientation-preserving part of O(3).

---

# 20.150 O(3) in Equivariant Deep Learning

Modern equivariant neural networks often describe features according to how they transform under O(3).

For example,

some features remain unchanged under reflection,

while others change sign.

This classification allows the network to distinguish between

* scalar quantities,
* vectors,
* pseudovectors,
* higher-order tensors.

Such distinctions are essential for accurately modeling physical systems.

---

# 20.151 O(3) and Spherical Harmonics

The spherical harmonics introduced later in this chapter naturally transform according to representations of O(3).

Likewise,

tensor products,

irreducible representations,

and equivariant convolutions all rely on O(3) symmetry.

Consequently,

understanding O(3) provides the mathematical foundation for the architectures used in **e3nn**, **NequIP**, **Allegro**, and **MACE**.

---

# 20.152 Key Takeaways

The essential differences between SO(3) and O(3) are summarized below.

| Property              | SO(3) | O(3) |
| --------------------- | ----- | ---- |
| Proper rotations      | ✓     | ✓    |
| Reflections           | ✗     | ✓    |
| Inversion             | ✗     | ✓    |
| Determinant           | +1    | ±1   |
| Preserves distances   | ✓     | ✓    |
| Preserves orientation | ✓     | ✗    |

SO(3) describes only orientation-preserving rotations, whereas O(3) extends this framework to include reflections and inversion. This broader symmetry group is particularly important in crystallography, molecular symmetry, and modern equivariant neural networks that must correctly model systems exhibiting mirror or inversion symmetry.

---

## Transition to Section 20.9

We have now examined the two most important rotational symmetry groups: **SO(3)** and **O(3)**. However, real materials are not only rotated—they can also be **translated** through space without changing their physical identity. To describe rotations, reflections, and translations within a single mathematical framework, we introduce the **Euclidean Group E(3)**. This group provides the complete symmetry description used by many state-of-the-art equivariant graph neural networks for atomistic modeling.

# 20.153 The Euclidean Group E(3)

In the previous sections, we studied two important symmetry groups:

* **SO(3)** — proper rotations,
* **O(3)** — rotations and reflections.

However, real materials undergo another equally important transformation:

**translation**.

A crystal can be moved from one location to another without changing its physical properties.

Likewise,

a molecule floating in space can drift several nanometers while remaining exactly the same molecule.

To describe rotations, reflections, and translations within a single mathematical framework, mathematicians define the **Euclidean Group**, denoted by **E(3)**.

E(3) is the complete symmetry group of ordinary three-dimensional space and forms the foundation of modern geometric deep learning for atomistic systems.

---

# 20.154 Why Translation Matters

Imagine a silicon crystal placed at the origin.

```text id="e31"
Origin

↓

Crystal
```

Now move the entire crystal

100 Å

to the right.

```text id="e32"
100 Å

↓

Crystal
```

Has the material changed?

No.

The crystal still has

* identical atoms,
* identical bonding,
* identical lattice,
* identical electronic structure,
* identical energy.

Only its position in space has changed.

---

# 20.155 Translation Is Not a Physical Change

Translation changes the coordinates,

but not the material.

For example,

suppose an atom is located at

[
(2,1,4).
]

Translate the entire crystal by

[
(5,0,-2).
]

The new coordinates become

[
(7,1,2).
]

Although every coordinate has changed,

the relative positions between atoms remain identical.

---

# 20.156 Relative Position Is What Matters

Suppose two atoms have coordinates

[
(1,2,3)
]

and

[
(4,2,3).
]

Their separation is

3 Å.

Translate both atoms by

[
(10,5,-8).
]

Their new positions become

[
(11,7,-5)
]

and

[
(14,7,-5).
]

Their separation is still

3 Å.

Physical interactions depend on these **relative positions**, not on absolute coordinates.

---

# 20.157 Definition of E(3)

The Euclidean Group consists of all transformations that preserve Euclidean geometry.

These include

* translations,
* rotations,
* reflections.

Conceptually,

```text id="e33"
Translation

+

Rotation

+

Reflection

↓

E(3)
```

Every transformation belonging to E(3) preserves

* distances,
* angles,
* geometric relationships.

---

# 20.158 Mathematical Representation

An E(3) transformation consists of two components:

1. a rotation or reflection,

2. a translation.

If

[
R
]

represents an orthogonal matrix,

and

[
\mathbf{t}
]

represents a translation vector,

then an E(3) transformation is written as

[
\mathbf{x}'
===========

R\mathbf{x}
+
\mathbf{t}.
]

This single equation describes every Euclidean transformation.

---

# 20.159 Components of E(3)

The Euclidean group combines several simpler symmetry operations.

```text id="e34"
Rotation

↓

Reflection

↓

Translation

↓

E(3)
```

Thus,

E(3) contains every transformation that leaves the physical geometry unchanged.

---

# 20.160 Relationship Between SO(3), O(3), and E(3)

The three symmetry groups are closely related.

```text id="e35"
SO(3)

↓

O(3)

↓

E(3)
```

More precisely,

* SO(3) contains only rotations,
* O(3) contains rotations and reflections,
* E(3) contains rotations, reflections, and translations.

Each group extends the previous one.

---

# 20.161 Why E(3) Matters for Materials

Atoms inside a crystal do not possess an absolute position in the universe.

Only their positions relative to neighboring atoms determine

* bonding,
* electronic structure,
* atomic forces,
* total energy.

Therefore,

machine learning models should ignore arbitrary translations while preserving relative geometry.

---

# 20.162 Translation Invariance

Suppose the total energy of a crystal is

```text id="e36"
−12.51 eV
```

Translate the crystal by

1 meter.

The energy remains

```text id="e37"
−12.51 eV
```

This property is called **translation invariance**.

---

# 20.163 Translation Equivariance

Coordinates themselves behave differently.

Suppose an atom is translated by

[
(2,0,0).
]

The predicted coordinates should also shift by

[
(2,0,0).
]

Coordinates are therefore **translation equivariant**.

---

# 20.164 Physical Quantities Under E(3)

Different quantities transform differently under the Euclidean group.

| Quantity    | Translation              | Rotation              | Reflection            |
| ----------- | ------------------------ | --------------------- | --------------------- |
| Energy      | Invariant                | Invariant             | Usually invariant     |
| Force       | Invariant to translation | Equivariant           | Depends on parity     |
| Coordinates | Equivariant              | Equivariant           | Equivariant           |
| Stress      | Invariant to translation | Tensor transformation | Tensor transformation |

These transformation rules determine how neural networks should process different outputs.

---

# 20.165 Why Conventional Neural Networks Fail

Suppose a neural network receives the coordinates

```text id="e38"
(1,2,3)
```

and later receives

```text id="e39"
(101,102,103)
```

Although these structures differ only by translation,

the network may incorrectly interpret them as unrelated inputs.

Consequently,

it wastes capacity learning translation invariance from data.

---

# 20.166 Relative Coordinates Solve the Problem

Modern atomistic neural networks often replace absolute coordinates with

* relative vectors,

* interatomic distances,

* neighbor displacements.

For two atoms,

the relative vector is

[
\mathbf{r}_{ij}
===============

## \mathbf{r}_j

\mathbf{r}_i.
]

Notice that translating both atoms by the same amount leaves

[
\mathbf{r}_{ij}
]

unchanged.

This naturally provides translation invariance.

---

# 20.167 E(3)-Equivariant Neural Networks

A neural network is called **E(3)-equivariant** if its outputs transform correctly under every Euclidean transformation.

Conceptually,

```text id="e40"
Translate

↓

Rotate

↓

Reflect

↓

Neural Network

↓

Correctly Transformed Output
```

Such networks automatically respect the geometry of three-dimensional space.

---

# 20.168 Why Modern Architectures Use E(3)

State-of-the-art atomistic neural networks generally target E(3) symmetry because it captures the complete set of physically meaningful rigid-body transformations.

Examples include

* NequIP,
* Allegro,
* MACE,
* e3nn-based architectures.

These models guarantee that predictions remain physically consistent regardless of how the system is positioned or oriented in space.

---

# 20.169 E(3) in Molecular Dynamics

During molecular dynamics,

atoms continuously

* translate,
* rotate,

while the underlying physical laws remain unchanged.

An E(3)-equivariant model naturally handles these transformations,

making it ideal for predicting

* energies,
* forces,
* stresses,

throughout long simulations.

---

# 20.170 E(3) and Materials Informatics

Many tasks in materials informatics require respecting Euclidean symmetry.

Examples include

* crystal relaxation,
* force prediction,
* defect migration,
* phonon calculations,
* diffusion,
* interatomic potential development,
* catalyst simulations,
* battery materials.

Ignoring E(3) symmetry often leads to reduced accuracy and poorer generalization.

---

# 20.171 Summary of Symmetry Groups

The hierarchy of symmetry groups can be summarized as follows.

| Group | Rotations | Reflections | Translations |
| ----- | --------- | ----------- | ------------ |
| SO(3) | ✓         | ✗           | ✗            |
| O(3)  | ✓         | ✓           | ✗            |
| E(3)  | ✓         | ✓           | ✓            |

Each successive group incorporates additional physically meaningful transformations.

---

# 20.172 Key Takeaways

The Euclidean Group **E(3)** represents the full symmetry of three-dimensional Euclidean space. By combining translations, rotations, and reflections into a single mathematical framework, E(3) provides the natural symmetry language for atomistic systems.

Modern equivariant graph neural networks are designed to respect E(3) because real materials can move and rotate freely without changing their intrinsic physical properties. Embedding this symmetry into neural network architectures greatly improves physical consistency, data efficiency, and predictive performance.

---

## Transition to Section 20.10

Having established the symmetry groups **SO(3)**, **O(3)**, and **E(3)**, we are now ready to examine the objects that transform under these groups. The next section introduces **tensor features**, explaining why scalars and vectors are insufficient for describing complex physical interactions and how higher-order tensors provide the expressive mathematical language required by modern equivariant graph neural networks such as **NequIP**, **MACE**, and **e3nn**.


# 20.173 Tensor Features

In the previous sections, we introduced the symmetry groups **SO(3)**, **O(3)**, and **E(3)**. These groups describe how geometric transformations such as rotations, reflections, and translations act on physical systems.

The next question is equally important:

> **What mathematical objects transform under these symmetry groups?**

The answer is **tensors**.

Tensors provide the mathematical language used to describe almost every physical quantity encountered in materials science, including

* energy,
* forces,
* stress,
* strain,
* polarization,
* magnetic moments,
* elastic constants.

Modern equivariant graph neural networks do not simply process numbers; they process **tensor features** that transform predictably under rotations and other symmetry operations.

---

# 20.174 Why Scalars Are Not Enough

Traditional machine learning models often represent each feature as a single scalar value.

For example,

```text id="tensor1"
Atomic Number

14

↓

Scalar
```

or

```text id="tensor2"
Electronegativity

1.90

↓

Scalar
```

These scalar features are useful, but they cannot describe directional information.

Suppose we wish to predict the force acting on an atom.

Force has both magnitude and direction.

Representing it as a single number would discard essential physical information.

---

# 20.175 Direction Matters

Consider an atom experiencing a force.

```text id="tensor3"
      ↑

      F
```

Another atom may experience a force

```text id="tensor4"
→
```

Both forces may have identical magnitudes,

yet they represent completely different physical situations.

A neural network must therefore preserve directional information.

This motivates the introduction of vectors.

---

# 20.176 Scalars

A scalar is the simplest type of tensor.

It consists of a single numerical value.

Examples include

* total energy,
* temperature,
* atomic mass,
* density,
* formation energy,
* band gap.

A scalar remains unchanged under rotation.

For this reason,

scalars are called **rank-0 tensors**.

---

# 20.177 Vectors

A vector contains both

* magnitude,
* direction.

Examples include

* force,
* velocity,
* displacement,
* electric field.

Vectors rotate together with the coordinate system.

Therefore,

vectors are **rank-1 tensors**.

---

# 20.178 Higher-Order Tensors

Some physical quantities require more than one direction.

Examples include

* stress,
* strain,
* dielectric tensors,
* elastic stiffness.

These quantities cannot be represented by vectors alone.

Instead,

they require **higher-order tensors**.

---

# 20.179 What Is a Tensor?

A tensor is a mathematical object that transforms according to specific rules under coordinate transformations.

Unlike ordinary arrays,

the defining characteristic of a tensor is **not its shape**, but **how it transforms**.

This transformation behavior is what makes tensors fundamental in physics and equivariant deep learning.

---

# 20.180 Tensor Rank

The **rank** (or order) of a tensor indicates the number of indices required to describe it.

| Rank | Example             | Physical Quantity        |
| ---- | ------------------- | ------------------------ |
| 0    | Scalar              | Energy                   |
| 1    | Vector              | Force                    |
| 2    | Matrix              | Stress                   |
| 3    | Third-order tensor  | Piezoelectric tensor     |
| 4    | Fourth-order tensor | Elastic stiffness tensor |

As the rank increases,

the tensor can describe increasingly complex directional relationships.

---

# 20.181 Rank-0 Tensor

A scalar requires no direction.

For example,

```text id="tensor5"
Energy

−12.5 eV
```

After rotating the crystal,

the value remains

```text id="tensor6"
Energy

−12.5 eV
```

No transformation occurs.

---

# 20.182 Rank-1 Tensor

A vector changes direction under rotation.

Suppose

```text id="tensor7"
Force

→
```

Rotate the crystal by

90°.

The force becomes

```text id="tensor8"
Force

↑
```

The magnitude is unchanged,

but the direction rotates.

---

# 20.183 Rank-2 Tensor

Stress is a classic example of a rank-2 tensor.

Stress acts simultaneously along multiple directions.

It is commonly written as

[
\boldsymbol{\sigma}
===================

\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} & \sigma_{xz}\
\sigma_{yx} & \sigma_{yy} & \sigma_{yz}\
\sigma_{zx} & \sigma_{zy} & \sigma_{zz}
\end{bmatrix}.
]

When the coordinate system rotates,

all nine components transform together according to tensor transformation rules.

---

# 20.184 Why Higher-Order Tensors Matter

Many materials properties involve interactions between multiple spatial directions.

Examples include

* elastic deformation,
* anisotropic conductivity,
* dielectric response,
* magnetic susceptibility.

Such phenomena cannot be described using only scalars or vectors.

Higher-order tensors provide the necessary mathematical framework.

---

# 20.185 Tensor Features in Neural Networks

In ordinary neural networks,

hidden features are simply vectors of numbers.

For example,

```text id="tensor9"
[0.42, 1.71, -0.35]
```

These numbers possess no geometric meaning.

In equivariant neural networks,

hidden features are instead organized into tensor representations.

Each feature knows how it should transform under rotation.

---

# 20.186 Example of Tensor Features

Suppose a hidden layer contains

* scalar features,
* vector features,
* tensor features.

Conceptually,

```text id="tensor10"
Hidden Layer

↓

Scalar

↓

Vector

↓

Tensor
```

Each type of feature transforms differently,

allowing the network to model increasingly rich geometric information.

---

# 20.187 Why Tensor Features Improve Learning

Imagine predicting atomic forces.

If hidden features were only scalars,

the network would struggle to represent directional interactions.

Tensor features allow the model to capture

* bond orientations,
* angular relationships,
* local symmetry,
* anisotropic interactions.

This dramatically increases expressive power.

---

# 20.188 Tensor Features and Atomic Environments

Consider a central atom surrounded by neighbors.

```text id="tensor11"
      O

      |

O — Si — O

      |

      O
```

The local environment contains

* distances,
* directions,
* angles.

Tensor features encode this geometric information in a form that transforms correctly under rotations.

---

# 20.189 Tensor Features During Rotation

Suppose the entire crystal rotates.

```text id="tensor12"
Crystal

↓

Rotate
```

Scalar features remain unchanged.

Vector features rotate.

Higher-order tensor features transform according to tensor algebra.

Because every feature transforms correctly,

the network remains physically consistent throughout all layers.

---

# 20.190 Internal Representations

One remarkable aspect of modern equivariant neural networks is that they maintain tensor-valued hidden representations throughout the network.

Instead of repeatedly converting everything into scalar numbers,

the model continuously propagates geometric information using tensors.

This enables the network to preserve rotational information from the input all the way to the final prediction.

---

# 20.191 Tensor Products

Tensor features can be combined using **tensor products**.

Tensor products allow the network to construct more complex geometric representations from simpler ones.

For example,

combining two vectors can produce

* a scalar,
* another vector,
* a rank-2 tensor,

depending on how the tensor product is decomposed.

This operation forms the computational backbone of many equivariant neural networks.

---

# 20.192 Tensor Features in Modern Models

State-of-the-art architectures such as

* e3nn,
* NequIP,
* Allegro,
* MACE

represent nearly every hidden feature as tensors associated with specific symmetry properties.

These tensor features enable the models to achieve remarkable accuracy while respecting the physical laws of three-dimensional space.

---

# 20.193 Key Takeaways

Tensor features extend the concept of neural network representations beyond ordinary numerical vectors.

By organizing hidden information into scalars, vectors, and higher-order tensors that transform correctly under symmetry operations, equivariant graph neural networks can naturally model the rich geometric structure of atomistic systems.

These tensor representations provide the bridge between geometric symmetry and deep learning, enabling modern architectures to predict complex physical properties with unprecedented accuracy.

---

## Transition to Section 20.11

Although tensor features provide the appropriate mathematical objects for equivariant learning, we still need a systematic way to describe **how these tensors rotate**. The mathematical tool that accomplishes this is **representation theory**, specifically **irreducible representations (irreps)**. In the next section, we will introduce irreducible representations, explain why they are fundamental to SO(3) symmetry, and show how they form the building blocks of modern equivariant neural networks such as **e3nn**, **NequIP**, and **MACE**.

Excellent. This is the most important mathematical section of the book. From here onward, the chapter moves into the theory that powers **e3nn, NequIP, Allegro, and MACE**.

---

# 20.11 Irreducible Representations (Irreps)

## 20.11.1 Introduction

In the previous sections, we learned that materials exist in **three-dimensional Euclidean space** and that physically meaningful machine learning models should respect the symmetry groups

* SO(3)
* O(3)
* E(3)

We also introduced **tensor features**, which transform according to these symmetry groups.

However, an important question remains unanswered.

> **How do we mathematically describe the action of a symmetry operation on a tensor?**

For example,

suppose we rotate a crystal by **45°**.

How should

* atomic coordinates,
* forces,
* stress,
* hidden neural-network features,

change under this rotation?

To answer this question, we need one of the most powerful ideas in modern mathematics:

> **Representation Theory.**

Representation theory provides a bridge between **abstract symmetry groups** and **concrete matrices** that computers can manipulate.

Without representation theory,

modern equivariant neural networks would not exist.

---

# 20.11.2 Why Representation Theory?

Recall that a symmetry group is simply a collection of transformations.

For example,

SO(3) contains infinitely many rotations.

A neural network, however, cannot manipulate an abstract mathematical object like

> "Rotate by 63.8° around an arbitrary axis."

Instead,

it performs operations using

* vectors,
* matrices,
* tensors,

which are numerical objects stored in computer memory.

Representation theory tells us

> **How to convert an abstract symmetry transformation into a concrete matrix operation.**

This is the key idea.

---

# 20.11.3 From Abstract Mathematics to Linear Algebra

Suppose we have an element

[
g \in SO(3)
]

representing a rotation.

Representation theory assigns a matrix

[
D(g)
]

to this rotation.

Instead of saying

```text
Rotate the crystal.
```

we can now write

[
x'
==

D(g)x.
]

The rotation has become an ordinary matrix multiplication.

Since neural networks already perform millions of matrix multiplications,

this formulation is perfectly suited for deep learning.

---

# 20.11.4 A Simple Analogy

Imagine a translator.

One person speaks English.

Another speaks Japanese.

The translator converts one language into another.

Representation theory performs a similar role.

```text
Symmetry Group

↓

Representation

↓

Matrices

↓

Neural Network
```

It translates the language of symmetry into the language of linear algebra.

---

# 20.11.5 What Is a Representation?

A **representation** of a group is a mapping that assigns every group element to a matrix while preserving the group structure.

Mathematically,

if

[
g_1,g_2\in G,
]

then

[
D(g_1g_2)
=========

D(g_1)D(g_2).
]

This equation is extremely important.

It means

> Performing two symmetry operations and then representing them is exactly the same as representing each operation individually and multiplying the corresponding matrices.

In other words,

representation theory preserves the algebraic structure of the symmetry group.

---

# 20.11.6 Why This Property Matters

Suppose we rotate a crystal twice.

First,

rotate by

30°.

Then,

rotate by

60°.

Physically,

this is identical to one rotation of

90°.

Representation theory guarantees

[
D(90^\circ)
===========

D(60^\circ)
D(30^\circ).
]

The neural network therefore behaves exactly like the underlying physics.

---

# 20.11.7 Example: Scalar Representation

Consider a scalar quantity,

such as total energy.

Rotating a crystal does not change its energy.

Therefore,

every rotation is represented simply by

[
D(g)=1.
]

No matter which rotation is applied,

the scalar remains unchanged.

This is called the **trivial representation**.

---

# 20.11.8 Example: Vector Representation

Now consider a force vector.

Suppose

[
\mathbf F
=========

\begin{bmatrix}
1\
0\
0
\end{bmatrix}.
]

Rotating the crystal by

90°

about the z-axis gives

[
D(R)
====

\begin{bmatrix}
0&-1&0\
1&0&0\
0&0&1
\end{bmatrix}.
]

The rotated force becomes

[
\mathbf F'
==========

D(R)\mathbf F.
]

Unlike the scalar,

the vector changes according to the rotation matrix.

---

# 20.11.9 Every Physical Quantity Has a Representation

Different physical quantities transform differently.

For example,

| Physical Quantity | Representation |
| ----------------- | -------------- |
| Energy            | Scalar         |
| Temperature       | Scalar         |
| Force             | Vector         |
| Velocity          | Vector         |
| Stress            | Rank-2 tensor  |
| Electric field    | Vector         |
| Polarization      | Vector         |

Representation theory provides the transformation rule for every one of these quantities.

---

# 20.11.10 Hidden Features Also Need Representations

A remarkable idea behind equivariant neural networks is that

**hidden features are treated exactly like physical quantities.**

Instead of storing arbitrary numbers,

the network stores objects that transform according to representations of SO(3).

For example,

a hidden feature may behave like

* a scalar,

or

* a vector,

or

* a higher-order tensor.

This ensures that every hidden layer remains physically consistent.

---

# 20.11.11 Many Possible Representations

A group can possess many different representations.

For SO(3),

there are infinitely many.

Some representations are

* one-dimensional,
* three-dimensional,
* five-dimensional,
* seven-dimensional,

and so on.

Each describes a different way in which an object may transform under rotation.

---

# 20.11.12 Why Not Use One Giant Matrix?

Suppose we describe a complicated tensor using a huge transformation matrix.

Although mathematically correct,

this matrix may contain enormous redundancy.

Many components are correlated.

Some represent independent physical information,

while others are mixtures of simpler transformations.

To make computation efficient,

we would like to break this large representation into its smallest independent pieces.

This leads to the concept of **irreducible representations**, or **irreps**.

---

# 20.11.13 Reducible vs. Irreducible Representations

Imagine a machine made from several detachable modules.

```text
Entire Machine

↓

Motor

↓

Battery

↓

Controller
```

Although the complete machine performs a complex task,

it is built from simpler independent components.

Similarly,

a large representation can often be decomposed into smaller representations.

If this decomposition is possible,

the representation is called **reducible**.

If no further decomposition is possible,

it is called **irreducible**.

---

# 20.11.14 Why Irreps Matter

Irreducible representations are analogous to atoms in chemistry.

Just as every molecule can be built from atoms,

every representation of SO(3) can be constructed from irreducible representations.

This is why irreps are considered the **fundamental building blocks** of representation theory.

Modern equivariant neural networks operate almost entirely in terms of irreps rather than arbitrary tensors.

---

# 20.11.15 Roadmap

In the next subsections, we will develop irreducible representations step by step. We will first understand **reducible representations**, then learn how they can be decomposed into simpler components, and finally introduce the irreducible representations of **SO(3)** labeled by the angular momentum quantum number (l). These irreps will become the mathematical foundation for **spherical harmonics**, **tensor products**, and the equivariant message-passing layers used by **e3nn**, **NequIP**, **Allegro**, and **MACE**.

---

This is only the **introduction** to Section 20.11. The full treatment of irreducible representations will span multiple subsections (roughly 25–35 textbook pages), including reducible representations, direct sums, basis changes, angular momentum, (l=0,1,2,\ldots), parity, and their implementation in modern equivariant GNNs.

# 20.11.16 Reducible Representations

In the previous section, we introduced the idea of a **representation**—a way of expressing abstract symmetry operations as concrete matrices.

However, not every representation is equally simple.

Some representations contain hidden internal structure.

These larger representations can often be decomposed into smaller, independent pieces.

Such representations are called **reducible representations**.

Understanding reducibility is essential because modern equivariant neural networks are built by decomposing complex tensor features into their simplest symmetry components.

---

# 20.11.17 An Everyday Analogy

Imagine a computer.

```text id="red1"
Computer

↓

CPU

↓

RAM

↓

GPU

↓

Storage
```

Although we often refer to it as one machine,

it is actually composed of several independent modules.

Each module performs its own function.

Similarly,

a large representation may appear to be one object,

while actually being composed of several smaller representations.

---

# 20.11.18 A Matrix That Contains Two Independent Parts

Consider the matrix

[
D(g)=
\begin{bmatrix}
A(g) & 0\
0 & B(g)
\end{bmatrix},
]

where

* (A(g)) is one representation,
* (B(g)) is another representation.

Notice that the upper-left block never interacts with the lower-right block.

The two parts evolve independently.

Therefore,

the representation is **reducible**.

---

# 20.11.19 Block-Diagonal Structure

A reducible representation often has a **block-diagonal** form.

Conceptually,

```text id="red2"
┌──────────────┐

│  Block A   0 │

│              │

│   0     Block B │

└──────────────┘
```

Each block represents an independent symmetry transformation.

Because the blocks do not interact,

they can be studied separately.

---

# 20.11.20 Independent Subspaces

Suppose our feature vector is

[
x=
\begin{bmatrix}
x_A\
x_B
\end{bmatrix}.
]

After applying the representation,

we obtain

[
x'=
\begin{bmatrix}
A(g)x_A\
B(g)x_B
\end{bmatrix}.
]

Notice that

* Block A affects only (x_A),
* Block B affects only (x_B).

Neither influences the other.

The representation naturally separates into two independent subspaces.

---

# 20.11.21 Physical Interpretation

Imagine two completely independent physical systems.

For example,

```text id="red3"
Crystal A

↓

Rotation
```

and

```text id="red4"
Crystal B

↓

Rotation
```

If the rotation of Crystal A never affects Crystal B,

the combined system is reducible.

Each subsystem can be analyzed independently.

---

# 20.11.22 Example: Scalars and Vectors

Suppose a neural network stores

* one scalar feature,
* one vector feature.

Conceptually,

```text id="red5"
Hidden Features

↓

Scalar

↓

Vector
```

The scalar transforms trivially.

The vector rotates.

Since these transformation rules are independent,

the combined representation is reducible.

---

# 20.11.23 Why Reducible Representations Are Inefficient

Working directly with a large reducible representation is often wasteful.

Many computations are repeated.

Independent components are unnecessarily mixed into a single mathematical object.

Instead,

it is more efficient to split the representation into its smallest independent parts.

This simplification improves

* computation,
* interpretation,
* numerical stability.

---

# 20.11.24 Decomposition

The process of splitting a reducible representation into independent pieces is called **decomposition**.

Conceptually,

```text id="red6"
Large Representation

↓

Decompose

↓

Small Representation

+

Small Representation

+

Small Representation
```

Each smaller piece transforms independently.

---

# 20.11.25 Why Decomposition Matters

Imagine solving a very large system of equations.

If the equations naturally separate into several smaller systems,

we solve each one independently.

This is much faster than solving the entire system simultaneously.

Representation theory applies the same principle.

---

# 20.11.26 Example from Linear Algebra

Suppose a matrix has the form

[
\begin{bmatrix}
2 & 0 & 0\
0 & 5 & 0\
0 & 0 & 8
\end{bmatrix}.
]

Each diagonal element acts independently.

There is no coupling between components.

Similarly,

block-diagonal representations contain independent symmetry sectors.

---

# 20.11.27 Representation Decomposition

Mathematically,

a reducible representation can often be written as

[
D
=

D_1
\oplus
D_2
\oplus
D_3,
]

where

(\oplus)

denotes the **direct sum**.

Each

(D_i)

is a smaller representation.

Later,

we will study the direct sum operation in detail.

---

# 20.11.28 Analogy with Chemistry

Chemists rarely study an entire protein atom by atom without first identifying

* amino acids,
* functional groups,
* secondary structures.

Likewise,

representation theory simplifies large symmetry transformations by identifying their fundamental components.

The reducible representation is analogous to the complete protein,

while the irreducible representations are analogous to its basic building blocks.

---

# 20.11.29 Hidden Features in Equivariant Networks

Modern equivariant neural networks continuously decompose hidden features into irreducible components.

Instead of storing

```text id="red7"
One Large Tensor
```

the network stores

```text id="red8"
Scalar

+

Vector

+

Higher-Order Features
```

Each feature transforms independently according to SO(3).

This greatly simplifies learning.

---

# 20.11.30 Why Neural Networks Prefer Small Pieces

Smaller representations provide several advantages.

They are

* easier to compute,
* easier to interpret,
* mathematically cleaner,
* physically meaningful.

Most importantly,

they correspond directly to fundamental symmetry objects.

---

# 20.11.31 From Reducible to Irreducible

Not every representation can be decomposed forever.

Eventually,

the decomposition reaches pieces that cannot be split any further.

These smallest symmetry components are called **irreducible representations**, or **irreps**.

They play the same role in representation theory that prime numbers play in arithmetic.

Every larger representation can be constructed from these elementary building blocks.

---

# 20.11.32 Key Takeaways

A **reducible representation** is one that can be separated into smaller independent representations.

Its transformation matrix can be rearranged into block-diagonal form, with each block acting independently on a different subspace.

Modern equivariant neural networks exploit this property by decomposing complex tensor features into smaller symmetry components, allowing computations to be performed more efficiently and in a way that naturally respects the underlying geometry of three-dimensional space.

---

## Transition to Section 20.11.17 — Irreducible Representations

Having seen that many representations can be decomposed into independent pieces, we now arrive at the fundamental building blocks of representation theory: **irreducible representations**. These irreps cannot be decomposed any further and form the mathematical foundation of **SO(3)** symmetry, **spherical harmonics**, **e3nn**, **NequIP**, **Allegro**, and **MACE**. Understanding irreps is one of the most important steps toward mastering modern equivariant graph neural networks.

# 20.11.17 Irreducible Representations (Irreps)

After studying reducible representations, we now arrive at one of the most important concepts in modern mathematics, theoretical physics, and geometric deep learning:

> **Irreducible Representations**, usually abbreviated as **irreps**.

Almost every modern equivariant graph neural network—including **e3nn**, **NequIP**, **Allegro**, **MACE**, and **Tensor Field Networks**—is built using irreducible representations of the rotation group **SO(3)**.

Understanding irreps is therefore essential for understanding how these models achieve rotational equivariance.

---

# 20.11.18 What Is an Irreducible Representation?

An **irreducible representation** is a representation that **cannot be decomposed into smaller independent representations**.

Unlike reducible representations,

there is no change of basis that transforms the representation into block-diagonal form.

It is already as simple as possible.

Mathematically,

an irrep is the smallest non-divisible building block of representation theory.

---

# 20.11.19 A Chemistry Analogy

An excellent analogy comes from chemistry.

Consider a water molecule.

```text id="irrep1"
H

 \

  O

 /

H
```

A water molecule can be separated into

* one oxygen atom,
* two hydrogen atoms.

Therefore,

the molecule is **composite**.

However,

consider a hydrogen atom.

```text id="irrep2"
H
```

It cannot be chemically divided into smaller atoms.

It is already fundamental.

Irreducible representations play exactly the same role.

A reducible representation is like a molecule,

while an irrep is like an atom.

---

# 20.11.20 Prime Number Analogy

Another useful analogy comes from number theory.

Consider the number

[
30.
]

It can be factorized as

[
30
==

2
\times
3
\times
5.
]

The numbers

2,

3,

and

5

cannot be factorized further.

They are **prime numbers**.

Similarly,

every reducible representation can be decomposed into irreducible representations.

Irreps are therefore the **prime factors of symmetry**.

---

# 20.11.21 Why Irreps Matter

Suppose we wish to understand a complicated crystal.

Instead of studying the entire system at once,

we first identify its simplest components.

Representation theory follows exactly the same philosophy.

Rather than studying one enormous transformation matrix,

we decompose it into irreducible pieces.

Each irrep represents one elementary mode of rotational behavior.

---

# 20.11.22 Building Blocks of SO(3)

Every finite-dimensional representation of SO(3) can be expressed as a combination of irreducible representations.

Conceptually,

```text id="irrep3"
Large Representation

↓

Irrep

+

Irrep

+

Irrep
```

Nothing smaller exists.

These irreps are the elementary objects used by modern equivariant neural networks.

---

# 20.11.23 Why Neural Networks Love Irreps

Suppose a hidden feature transforms under rotation.

Instead of storing one complicated tensor,

an equivariant neural network stores

```text id="irrep4"
Scalar

+

Vector

+

Higher-Order Components
```

Each component corresponds to one irrep.

Because every component transforms independently,

the network naturally preserves rotational symmetry.

---

# 20.11.24 Irreps of SO(3)

One remarkable property of SO(3) is that its irreducible representations are completely classified.

Each irrep is labeled by a non-negative integer

[
l
=

0,
1,
2,
3,
\ldots
]

This integer is known in physics as the **angular momentum quantum number**.

Each value of

(l)

defines a different representation.

---

# 20.11.25 The Meaning of (l)

The value of

[
l
]

determines

* the dimensionality,
* the transformation behavior,
* the geometric complexity,

of the representation.

As

(l)

increases,

the representation can describe increasingly complex angular patterns.

---

# 20.11.26 The First Few Irreps

The first few irreducible representations are

| (l) | Dimension | Interpretation      |
| --- | --------- | ------------------- |
| 0   | 1         | Scalar              |
| 1   | 3         | Vector              |
| 2   | 5         | Quadrupole          |
| 3   | 7         | Octupole            |
| 4   | 9         | Higher-order tensor |

Notice an important pattern.

The dimension always equals

[
2l+1.
]

This simple formula appears throughout quantum mechanics,

spherical harmonics,

and equivariant deep learning.

---

# 20.11.27 The Dimension Formula

For every irreducible representation of SO(3),

the dimension is

[
2l+1.
]

Examples include

| (l) | Dimension |
| --- | --------- |
| 0   | 1         |
| 1   | 3         |
| 2   | 5         |
| 3   | 7         |
| 4   | 9         |

As the angular momentum increases,

the representation contains more independent components.

---

# 20.11.28 Why the Dimension Grows

Higher-order geometric patterns require more information.

A scalar requires only one number.

A vector requires three numbers.

More complicated angular structures require

five,

seven,

nine,

or more independent components.

The increasing dimensionality reflects the increasing complexity of rotational behavior.

---

# 20.11.29 Visual Intuition

The hierarchy of irreps can be visualized conceptually.

```text id="irrep5"
l = 0

↓

Scalar

↓

1 Component
```

```text id="irrep6"
l = 1

↓

Vector

↓

3 Components
```

```text id="irrep7"
l = 2

↓

Quadrupole

↓

5 Components
```

Each successive irrep captures richer angular information.

---

# 20.11.30 Hidden Features as Irreps

Modern equivariant neural networks do not represent hidden features as arbitrary vectors.

Instead,

they organize hidden channels according to irreps.

For example,

a hidden layer might contain

```text id="irrep8"
32 × l=0

16 × l=1

8 × l=2
```

This notation means

* thirty-two scalar channels,
* sixteen vector channels,
* eight quadrupole channels.

Each channel follows the correct SO(3) transformation rules.

---

# 20.11.31 Why This Is Powerful

Suppose the crystal rotates.

Every hidden feature automatically transforms according to its assigned irrep.

The neural network never has to "learn"

how vectors rotate.

It already knows.

This dramatically improves

* physical consistency,
* learning efficiency,
* generalization.

---

# 20.11.32 Irreps in Quantum Mechanics

Irreducible representations are not unique to machine learning.

They have been central to quantum mechanics for nearly a century.

Examples include

* atomic orbitals,
* electron angular momentum,
* nuclear spin,
* spherical harmonics,
* spectroscopy.

Modern equivariant deep learning borrows these mathematical tools directly from quantum physics.

---

# 20.11.33 Irreps in Materials Science

Many physical phenomena naturally decompose into irreducible representations.

Examples include

* atomic orbitals,
* crystal fields,
* phonon modes,
* lattice vibrations,
* multipole moments,
* magnetic ordering.

Consequently,

using irreps inside neural networks closely mirrors the mathematical structure of real physical systems.

---

# 20.11.34 Why Irreps Are the Language of e3nn

The **e3nn** framework represents nearly every hidden feature using irreps.

Instead of ordinary tensors,

every feature is labeled by

* its degree (l),
* its parity (introduced later),
* its multiplicity.

This structured representation allows every neural-network layer to remain exactly equivariant under SO(3) and O(3).

---

# 20.11.35 Key Takeaways

Irreducible representations are the **elementary building blocks of rotational symmetry**.

Every complex representation can be decomposed into irreps, just as every integer can be factorized into prime numbers or every molecule can be decomposed into atoms.

Modern equivariant graph neural networks operate almost entirely in terms of these irreps, enabling them to encode rotational symmetry directly into their architecture rather than learning it from data.

---

## Transition to Section 20.11.18 — Angular Momentum Interpretation

The labels (l = 0, 1, 2, 3, \ldots) introduced in this section are not arbitrary. They originate from **angular momentum theory** in quantum mechanics and determine the dimensionality and transformation behavior of each irrep. In the next section, we will explore the physical meaning of these quantum numbers and explain why angular momentum provides the natural language for describing rotations in both quantum physics and equivariant graph neural networks.

# 20.11.18 Angular Momentum Interpretation of Irreducible Representations

One of the most fascinating aspects of representation theory is that the irreducible representations of **SO(3)** are **exactly the same mathematical objects** that describe **angular momentum in quantum mechanics**.

This is not a coincidence.

Nature itself is rotationally symmetric.

Because of this symmetry,

every rotating physical system—from electrons inside atoms to entire galaxies—obeys the mathematics of SO(3).

Modern equivariant graph neural networks inherit this same mathematical framework.

---

# 20.11.19 Why Angular Momentum Appears

Imagine rotating an isolated atom.

Nothing about the laws of physics changes.

The atom may point in a different direction,

but its physical behavior remains governed by the same equations.

This rotational symmetry implies that the mathematical description of the atom must respect SO(3).

Quantum mechanics therefore classifies atomic states according to the irreducible representations of SO(3).

Exactly the same classification is used in equivariant neural networks.

---

# 20.11.20 What Is Angular Momentum?

In classical mechanics,

angular momentum measures rotational motion.

For a particle,

it is defined as

[
\mathbf{L}
==========

\mathbf{r}
\times
\mathbf{p},
]

where

* (\mathbf r) is the position vector,
* (\mathbf p) is the linear momentum.

Angular momentum tells us

* how fast an object rotates,
* around which axis it rotates,
* in which direction it rotates.

---

# 20.11.21 Angular Momentum in Quantum Mechanics

Quantum mechanics introduces an important difference.

Angular momentum cannot take arbitrary values.

Instead,

it is **quantized**.

The total angular momentum is characterized by a quantum number

[
l
=

0,1,2,3,\ldots
]

These are exactly the same integers that label the irreducible representations of SO(3).

---

# 20.11.22 The Quantum Number (l)

The integer

[
l
]

determines

* the angular complexity,
* the rotational behavior,
* the dimensionality of the representation.

Each value corresponds to a different symmetry type.

For example,

| (l) | Physical Meaning      |
| --- | --------------------- |
| 0   | Spherically symmetric |
| 1   | Dipole-like           |
| 2   | Quadrupole-like       |
| 3   | Octupole-like         |

Higher values correspond to increasingly complex angular patterns.

---

# 20.11.23 The (2l+1) Rule

One of the most famous results in quantum mechanics is that each angular momentum state contains

[
2l+1
]

independent components.

For example,

| (l) | Number of Components |
| --- | -------------------- |
| 0   | 1                    |
| 1   | 3                    |
| 2   | 5                    |
| 3   | 7                    |
| 4   | 9                    |

This is precisely the dimension of the corresponding SO(3) irrep.

---

# 20.11.24 Physical Meaning of (l=0)

The simplest irrep is

[
l=0.
]

Its dimension is

1.

It corresponds to quantities that look identical from every direction.

Examples include

* total energy,
* temperature,
* atomic mass,
* density.

Graphically,

```text id="ang1"
      ●
```

A perfect sphere appears the same regardless of rotation.

This is the geometric intuition behind the scalar representation.

---

# 20.11.25 Physical Meaning of (l=1)

The next irrep is

[
l=1.
]

Its dimension is

3.

This corresponds to vectors.

Examples include

* force,
* velocity,
* displacement,
* electric field.

Graphically,

```text id="ang2"
→
```

Rotating the coordinate system rotates the vector.

---

# 20.11.26 Physical Meaning of (l=2)

The next representation has

[
l=2.
]

Its dimension is

5.

This representation describes quadrupole-like patterns.

Examples include

* stress,
* strain,
* electric quadrupole moments,
* some crystal-field interactions.

Unlike vectors,

quadrupole patterns possess more complicated angular structure.

---

# 20.11.27 Physical Meaning of Higher (l)

As

(l)

increases,

the angular complexity grows.

Examples include

| (l) | Interpretation   |
| --- | ---------------- |
| 3   | Octupole         |
| 4   | Hexadecapole     |
| 5   | Higher multipole |

These higher-order representations become increasingly important when describing subtle anisotropic interactions in atoms and materials.

---

# 20.11.28 Atomic Orbitals

One of the most familiar examples comes from atomic orbitals.

Chemistry students already encounter these objects.

| Orbital | (l) |
| ------- | --- |
| s       | 0   |
| p       | 1   |
| d       | 2   |
| f       | 3   |

Remarkably,

these orbital labels are identical to the SO(3) irreducible representations.

This is not accidental.

Atomic orbitals are solutions of the Schrödinger equation in a rotationally symmetric potential.

---

# 20.11.29 Orbital Shapes

The increasing value of

(l)

produces increasingly complex angular shapes.

Conceptually,

```text id="ang3"
s

↓

Sphere
```

```text id="ang4"
p

↓

Dumbbell
```

```text id="ang5"
d

↓

Four Lobes
```

```text id="ang6"
f

↓

Complex Lobes
```

These familiar orbital shapes arise directly from SO(3) representation theory.

---

# 20.11.30 Why This Matters for Machine Learning

Modern equivariant neural networks borrow this exact mathematical structure.

Instead of representing hidden features as arbitrary vectors,

they organize them according to

* (l=0),
* (l=1),
* (l=2),
* ...

Each hidden feature therefore behaves like a generalized orbital under rotation.

---

# 20.11.31 Hidden Channels in e3nn

For example,

an e3nn layer may contain

```text id="ang7"
64 × l=0

32 × l=1

16 × l=2
```

This means

* sixty-four scalar channels,
* thirty-two vector channels,
* sixteen quadrupole channels.

Every channel transforms according to its assigned angular momentum.

---

# 20.11.32 Why Higher (l) Helps

Suppose two neighboring atoms interact.

A scalar feature can describe

"how strong"

the interaction is.

A vector feature can describe

"which direction"

the interaction points.

A higher-order feature can describe

the full angular distribution of neighboring atoms.

Consequently,

higher values of

(l)

allow the network to represent increasingly sophisticated geometric information.

---

# 20.11.33 Angular Momentum and Local Geometry

Consider a central atom surrounded by neighbors.

```text id="ang8"
      O

      |

O — Si — O

      |

      O
```

The angular arrangement of these neighbors determines many material properties.

Representing this geometry requires more than simple vectors.

Higher-order irreps naturally encode these angular relationships.

---

# 20.11.34 Why Physics and Deep Learning Agree

The appearance of angular momentum in equivariant neural networks is not merely a convenient mathematical trick.

Both quantum mechanics and geometric deep learning describe objects that must transform consistently under rotations.

Since SO(3) governs rotational symmetry,

both fields inevitably arrive at the same mathematical framework.

This deep connection explains why concepts developed for quantum physics now play a central role in modern machine learning.

---

# 20.11.35 Key Takeaways

The irreducible representations of SO(3) are naturally labeled by the angular momentum quantum number (l). Each value of (l) describes a distinct type of rotational behavior, with dimensionality given by (2l+1).

These representations already appear throughout quantum mechanics in the description of atomic orbitals and angular momentum. Modern equivariant graph neural networks adopt the same mathematical language, allowing hidden features to transform exactly as physical quantities do under rotations.

Understanding this connection provides the conceptual bridge between quantum mechanics and geometric deep learning.

---

## Transition to Section 20.11.19 — Direct Sums of Irreducible Representations

Real physical systems rarely consist of a single irrep. Instead, they combine multiple irreducible representations—for example, scalars, vectors, and higher-order tensors—into a single feature space. The mathematical operation that combines independent irreps is called the **direct sum**. In the next section, we will study direct sums and show how modern equivariant neural networks construct rich hidden representations by combining multiple irreducible components while preserving rotational symmetry.

# 20.11.19 Direct Sums of Irreducible Representations

In the previous sections, we learned that **irreducible representations (irreps)** are the fundamental building blocks of rotational symmetry.

However, real physical systems are rarely described by only a single irrep.

For example,

a material simultaneously possesses

* scalar quantities,
* vector quantities,
* higher-order tensor quantities.

Similarly,

an equivariant neural network rarely stores only scalar features or only vector features.

Instead,

it combines many different irreps into a single feature representation.

The mathematical operation that performs this combination is called the **direct sum**.

---

# 20.11.20 Why Do We Need Direct Sums?

Suppose we wish to describe an atom.

One scalar is insufficient because we may need

* atomic energy,
* atomic charge,
* atomic number.

Likewise,

a single vector is insufficient because we may also need

* force,
* displacement,
* magnetic moment.

Instead,

we combine several independent symmetry objects into one larger representation.

This combination is called a **direct sum**.

---

# 20.11.21 An Everyday Analogy

Imagine a university student.

A student possesses

* a student ID,
* a GPA,
* an address,
* a photograph,
* course records.

Each item stores different information.

Together,

they describe the student.

Conceptually,

```text id="ds1"
Student

↓

ID

+

GPA

+

Photo

+

Courses
```

The student profile is not one indivisible object.

It is a combination of several independent pieces.

A direct sum works in exactly the same way.

---

# 20.11.22 Mathematical Definition

Suppose we have two irreducible representations,

[
D_1
]

and

[
D_2.
]

Their direct sum is written as

[
D_1
\oplus
D_2.
]

The symbol

[
\oplus
]

is pronounced

> **direct sum**.

It means

> Place the two representations side by side while keeping them independent.

---

# 20.11.23 Matrix Form of a Direct Sum

If

[
D_1
===

A,
]

and

[
D_2
===

B,
]

their direct sum becomes

[
A
\oplus
B
=

\begin{bmatrix}
A & 0\
0 & B
\end{bmatrix}.
]

Notice the block-diagonal structure.

Each block transforms independently.

There is no interaction between them.

---

# 20.11.24 Visual Interpretation

Conceptually,

a direct sum looks like

```text id="ds2"
Large Representation

↓

Scalar

+

Vector

+

Tensor
```

Each component retains its own transformation rule.

Together,

they form a richer feature representation.

---

# 20.11.25 Example: Scalar + Vector

Suppose we combine

* one scalar,

* one vector.

The scalar has dimension

1.

The vector has dimension

3.

Their direct sum therefore has dimension

4.

Conceptually,

```text id="ds3"
1

+

3

=

4
```

The combined feature contains

* one scalar component,
* three vector components.

---

# 20.11.26 Example: Multiple Irreps

Suppose a neural network stores

* two scalar channels,

* one vector channel,

* one quadrupole channel.

The total representation becomes

[
2\times(l=0)
\oplus
1\times(l=1)
\oplus
1\times(l=2).
]

This notation is commonly used in e3nn.

---

# 20.11.27 Dimension of the Combined Representation

Recall

[
l=0
]

has dimension

1,

[
l=1
]

has dimension

3,

and

[
l=2
]

has dimension

5.

The total dimension is therefore

[
2\times1
+
1\times3
+
1\times5
========

10.

]

Although the representation contains ten numerical values,

they belong to different irreps.

---

# 20.11.28 Independent Transformations

Suppose the crystal rotates.

The scalar remains unchanged.

The vector rotates.

The quadrupole transforms according to its own SO(3) rule.

Conceptually,

```text id="ds4"
Scalar

↓

Same
```

```text id="ds5"
Vector

↓

Rotate
```

```text id="ds6"
Quadrupole

↓

Higher-Order Rotation
```

Each part behaves independently.

---

# 20.11.29 Why Independence Is Important

Imagine storing every feature inside one enormous tensor.

A small rotation would affect every element simultaneously,

making interpretation difficult.

By separating features into irreps,

each component follows its own mathematically well-defined transformation.

This makes the network

* simpler,
* more interpretable,
* more physically consistent.

---

# 20.11.30 Direct Sums in e3nn

The e3nn library represents feature spaces almost entirely using direct sums.

For example,

```text id="ds7"
32x0e

16x1o

8x2e
```

This notation means

* 32 scalar irreps,
* 16 vector irreps,
* 8 quadrupole irreps.

(The parity labels **e** and **o** will be discussed later.)

Internally,

e3nn stores these as one large direct-sum representation.

---

# 20.11.31 Hidden Features in Modern Networks

A hidden layer may therefore look like

```text id="ds8"
Hidden Layer

↓

Scalars

↓

Vectors

↓

Higher Tensors
```

Every feature belongs to one specific irrep.

During message passing,

each transforms correctly under SO(3).

---

# 20.11.32 Why Direct Sums Improve Learning

Different physical quantities require different symmetry behaviors.

For example,

energy behaves as

(l=0),

while force behaves as

(l=1).

Instead of forcing both quantities into a single representation,

the network simply combines their irreps through a direct sum.

This preserves the correct physics automatically.

---

# 20.11.33 Example from Materials Science

Consider predicting

* formation energy,
* atomic force,
* local crystal anisotropy.

These correspond approximately to

```text id="ds9"
Energy

↓

Scalar
```

```text id="ds10"
Force

↓

Vector
```

```text id="ds11"
Angular Environment

↓

Higher Irrep
```

A direct sum allows all three types of information to coexist within the same neural-network feature space.

---

# 20.11.34 Why Direct Sums Are Everywhere

Direct sums appear throughout

* quantum mechanics,
* crystallography,
* particle physics,
* representation theory,
* equivariant deep learning.

Whenever several independent symmetry objects must be described simultaneously,

the direct sum provides the natural mathematical framework.

---

# 20.11.35 Key Takeaways

A **direct sum** combines multiple irreducible representations into a larger representation while preserving the independence of each component. Rather than mixing scalars, vectors, and higher-order tensors into a single undifferentiated feature, modern equivariant neural networks organize them as a direct sum of irreps.

This structure allows every feature to transform according to its own symmetry rule, ensuring mathematical correctness, computational efficiency, and physical interpretability.

---

## Transition to Section 20.11.20 — Basis Transformations

Although direct sums tell us **which irreps** are present in a representation, the numerical values of these representations still depend on the **choice of basis**. Different coordinate systems or feature bases can describe the same physical object. In the next section, we will study **basis transformations**, showing why changing the basis does not alter the underlying physics and why equivariant neural networks can freely change feature bases while preserving symmetry.

# 20.11.20 Basis Transformations

In the previous section, we learned that complex representations can be constructed by combining irreducible representations using **direct sums**.

However, there is another important idea that often causes confusion.

A mathematical object can be described using **different coordinate systems**, or more generally, **different bases**.

Although the numerical values may change,

the underlying physical object remains exactly the same.

This idea is called a **basis transformation**.

Understanding basis transformations is essential because modern equivariant neural networks frequently change the basis of their hidden features while preserving the same physical information.

---

# 20.11.21 What Is a Basis?

A **basis** is a set of vectors that allows us to describe every vector in a space.

For ordinary three-dimensional space, the standard basis is

[
\mathbf e_x=
\begin{bmatrix}
1\
0\
0
\end{bmatrix},
\qquad
\mathbf e_y=
\begin{bmatrix}
0\
1\
0
\end{bmatrix},
\qquad
\mathbf e_z=
\begin{bmatrix}
0\
0\
1
\end{bmatrix}.
]

These three vectors define the familiar Cartesian coordinate system.

---

# 20.11.22 Describing a Vector

Suppose we have a vector

[
\mathbf v=
\begin{bmatrix}
2\
1\
3
\end{bmatrix}.
]

This simply means

[
\mathbf v
=========

2\mathbf e_x
+
1\mathbf e_y
+
3\mathbf e_z.
]

The numbers

2,

1,

and

3

depend entirely on the chosen basis.

---

# 20.11.23 A Simple Analogy

Imagine describing the location of a city.

One person uses

* latitude and longitude.

Another uses

* kilometers east and north.

Another uses

* GPS coordinates.

Although the numbers differ,

they all describe the same physical location.

Changing the basis changes the description,

not the object.

---

# 20.11.24 Rotating the Coordinate Axes

Suppose we rotate our coordinate axes by

90°.

```text id="basis1"
Original Axes

↓

Rotate Axes

↓

New Axes
```

The coordinates of every vector change.

However,

the physical vector itself has not moved.

Only the coordinate system has changed.

---

# 20.11.25 Physical Objects vs. Coordinates

This distinction is fundamental.

A force acting on an atom is a physical quantity.

Its coordinates are merely one way of describing it.

Changing coordinates does **not** change the force itself.

Modern equivariant neural networks exploit this fact by learning representations that are independent of any particular coordinate system.

---

# 20.11.26 Mathematical Basis Transformation

Suppose

[
P
]

is a change-of-basis matrix.

If

[
\mathbf v
]

is represented in one basis,

its coordinates in the new basis become

[
\mathbf v'
==========

P^{-1}\mathbf v.
]

The vector itself remains unchanged.

Only its numerical representation changes.

---

# 20.11.27 Transforming Representation Matrices

Changing the basis also changes the representation matrix.

If

[
D(g)
]

is the representation in the original basis,

then in the new basis,

[
D'(g)
=====

P^{-1}
D(g)
P.
]

This equation is called a **similarity transformation**.

It is one of the central equations of representation theory.

---

# 20.11.28 Similarity Transformations Preserve Physics

Although

[
D(g)
]

and

[
D'(g)
]

look different,

they describe exactly the same symmetry transformation.

They differ only because they are expressed in different coordinate systems.

This is analogous to measuring temperature in Celsius or Kelvin.

The numerical values differ,

but the physical temperature is identical.

---

# 20.11.29 Why Basis Choice Matters

Some bases make mathematical calculations much easier than others.

For example,

a complicated representation matrix may become block diagonal after an appropriate basis transformation.

Conceptually,

```text id="basis2"
Complicated Matrix

↓

Change Basis

↓

Block-Diagonal Matrix
```

This simplification allows us to identify irreducible representations.

---

# 20.11.30 Diagonalization Analogy

You may recall from linear algebra that matrices are often diagonalized.

Diagonal matrices are much easier to understand than arbitrary matrices.

Representation theory follows a similar philosophy.

Instead of diagonalizing whenever possible,

it searches for bases that expose the irreducible structure of the representation.

---

# 20.11.31 Basis Transformations in Quantum Mechanics

Quantum mechanics routinely changes bases.

For example,

a quantum state may be expressed in

* the position basis,
* the momentum basis,
* the energy basis,
* the angular momentum basis.

Each basis emphasizes different physical information,

yet they all describe the same underlying state.

---

# 20.11.32 Basis Transformations in e3nn

The **e3nn** framework also performs basis transformations internally.

Instead of using arbitrary tensor coordinates,

it often converts features into bases built from irreducible representations.

This makes tensor products and equivariant operations much simpler.

The user rarely sees these transformations,

but they occur continuously inside the network.

---

# 20.11.33 Why Neural Networks Change Basis

Changing basis can

* simplify computations,
* separate independent symmetry components,
* reduce computational cost,
* improve numerical stability.

The network is therefore free to choose whichever basis is most convenient,

provided the transformation rules remain mathematically consistent.

---

# 20.11.34 Physical Interpretation

Imagine looking at a crystal from different directions.

```text id="basis3"
Front View

↓

Side View

↓

Top View
```

Each view appears different,

yet the crystal itself never changes.

A basis transformation is the mathematical equivalent of changing the viewing direction.

---

# 20.11.35 Basis Independence

A physically meaningful neural network should never depend on an arbitrary choice of basis.

If two scientists describe the same crystal using different coordinate systems,

the network should ultimately predict the same physical quantities.

Equivariant neural networks achieve this by ensuring that every layer transforms consistently under basis changes.

---

# 20.11.36 From Basis Changes to Spherical Harmonics

One particularly useful basis for describing rotations is the **spherical harmonic basis**.

In this basis,

the irreducible representations of SO(3) take an especially elegant form.

This is why spherical harmonics appear throughout

* quantum mechanics,
* computer graphics,
* crystallography,
* geometric deep learning.

They provide one of the most natural coordinate systems for rotational symmetry.

---

# 20.11.37 Key Takeaways

A **basis transformation** changes the numerical description of a mathematical object without changing the object itself. Representation matrices expressed in different bases are related by similarity transformations, ensuring that the underlying symmetry remains unchanged.

Modern equivariant neural networks take advantage of basis transformations to simplify computations, expose irreducible structure, and implement efficient equivariant operations. The choice of basis is therefore a matter of mathematical convenience rather than physical significance.

---

## Transition to Section 20.11.21 — From Irreps to Spherical Harmonics

We have now introduced representations, irreducible representations, direct sums, and basis transformations. The next step is to identify the concrete mathematical functions that realize the irreducible representations of **SO(3)**. These functions are known as **spherical harmonics**. They provide the angular basis used throughout quantum mechanics and form the mathematical foundation of message passing in modern equivariant graph neural networks such as **e3nn**, **NequIP**, **Allegro**, and **MACE**.

# 20.11.21 Irreducible Representations in Modern Equivariant Neural Networks

Up to this point, we have studied irreducible representations from a mathematical perspective.

We have learned that

* irreps are the smallest building blocks of rotational symmetry,
* every representation can be decomposed into irreps,
* irreps are labeled by the angular momentum quantum number (l),
* direct sums combine multiple irreps,
* basis transformations change coordinates without changing the underlying symmetry.

The natural question now is:

> **How are irreducible representations actually used inside a neural network?**

The answer is that **irreps are the fundamental data type of modern equivariant graph neural networks**.

Instead of treating hidden features as ordinary vectors of numbers, these networks organize every feature according to its transformation behavior under the rotation group SO(3).

---

# 20.11.22 Traditional Neural Networks

Consider an ordinary multilayer perceptron.

A hidden layer might contain

```text id="nn1"
Hidden Layer

↓

0.24

↓

−1.35

↓

2.18

↓

0.71
```

These numbers have no physical meaning.

The network does not know whether a feature represents

* energy,
* force,
* angle,
* distance.

They are simply numerical values optimized during training.

---

# 20.11.23 Hidden Features Have No Geometry

Suppose we rotate a crystal by

90°.

The hidden features of a conventional neural network do not possess any predefined transformation rule.

The network must discover rotational behavior entirely from the training data.

This creates two major problems.

First,

the network requires many rotated examples.

Second,

it may still fail to generalize to unseen orientations.

---

# 20.11.24 The Equivariant Approach

Modern equivariant neural networks take a completely different approach.

Instead of storing arbitrary numbers,

they explicitly assign each hidden feature to an irreducible representation.

Conceptually,

```text id="nn2"
Hidden Layer

↓

Scalar

↓

Vector

↓

Quadrupole

↓

Higher Irrep
```

Each feature already "knows"

how it should rotate.

---

# 20.11.25 Features Become Physical Objects

Inside an equivariant neural network,

hidden features behave like physical quantities.

Some features behave like

* energies,

others behave like

* forces,

while others behave like

* higher-order angular tensors.

This makes the hidden representation resemble a physical description of the atomic environment rather than an arbitrary collection of numbers.

---

# 20.11.26 The Language of e3nn

The **e3nn** library describes feature spaces using a concise notation.

For example,

```text id="nn3"
64x0e
```

means

64 scalar irreps.

Likewise,

```text id="nn4"
32x1o
```

means

32 vector irreps.

Similarly,

```text id="nn5"
16x2e
```

means

16 quadrupole irreps.

Each feature belongs to a specific symmetry class.

---

# 20.11.27 Meaning of the Notation

The notation

```text id="nn6"
16x2e
```

contains three pieces of information.

* **16** → multiplicity (number of channels)
* **2** → angular momentum (l=2)
* **e** → even parity

Later,

we will study parity in detail.

For now,

the important point is that every hidden feature is explicitly labeled by its symmetry properties.

---

# 20.11.28 Hidden Layer Example

Suppose an equivariant hidden layer is written as

```text id="nn7"
64x0e

32x1o

16x2e
```

The network therefore contains

* 64 scalar channels,
* 32 vector channels,
* 16 rank-2 channels.

Each group transforms independently under rotations.

---

# 20.11.29 Rotation of Hidden Features

Imagine rotating the crystal.

The hidden features respond differently depending on their irrep.

```text id="nn8"
l=0

↓

Unchanged
```

```text id="nn9"
l=1

↓

Rotate Like Vector
```

```text id="nn10"
l=2

↓

Transform as Quadrupole
```

Every transformation follows the mathematical rules of SO(3).

---

# 20.11.30 Why This Is Better

Consider predicting atomic forces.

A conventional neural network must learn

> "When the crystal rotates, the force should rotate."

An equivariant neural network already knows this.

The transformation rule is built directly into the architecture.

As a result,

the network requires

* fewer training examples,
* less computational effort,
* better generalization.

---

# 20.11.31 Information Flow During Message Passing

During message passing,

neighboring atoms exchange tensor features rather than ordinary vectors.

Conceptually,

```text id="nn11"
Atom A

↓

Tensor Feature

↓

Atom B
```

These tensor features retain their symmetry properties throughout every layer.

---

# 20.11.32 Why Irreps Remain Separate

One important design principle is that different irreps are never mixed arbitrarily.

For example,

a scalar channel does not suddenly become a vector channel.

Instead,

special tensor-product operations combine irreps in mathematically valid ways.

These tensor products are governed by the rules of angular momentum addition.

We will study these operations later in this chapter.

---

# 20.11.33 Example from Materials Science

Suppose we wish to predict

* total energy,
* atomic forces,
* stress.

These outputs possess different symmetry.

```text id="nn12"
Energy

↓

Scalar
```

```text id="nn13"
Force

↓

Vector
```

```text id="nn14"
Stress

↓

Rank-2 Tensor
```

Using irreps allows the network to represent all three quantities naturally within the same framework.

---

# 20.11.34 Why Hidden Features Become More Expressive

As information propagates through the network,

higher-order irreps capture increasingly rich geometric information.

For example,

neighbor arrangements,

bond angles,

local anisotropy,

and crystal symmetry

can all be represented using higher values of

(l).

Consequently,

the hidden representation becomes progressively more informative.

---

# 20.11.35 Connection to Quantum Mechanics

The use of irreps inside neural networks is strongly inspired by quantum mechanics.

In quantum physics,

electronic wavefunctions are expanded in

* atomic orbitals,
* spherical harmonics,
* angular momentum eigenstates.

Modern equivariant neural networks perform a remarkably similar operation.

Instead of expanding wavefunctions,

they expand hidden features into irreducible representations of SO(3).

---

# 20.11.36 Modern Architectures

Nearly every state-of-the-art equivariant architecture follows this philosophy.

Examples include

* **e3nn**
* **NequIP**
* **Allegro**
* **MACE**
* **Tensor Field Networks**

Although these models differ in message-passing strategy,

they all use irreducible representations as their fundamental feature type.

---

# 20.11.37 Key Takeaways

Modern equivariant neural networks replace ordinary hidden vectors with **irreducible representations of SO(3)**. Each hidden channel is assigned a specific symmetry behavior—such as scalar, vector, or higher-order tensor—and transforms according to well-defined mathematical rules under rotation.

By embedding these symmetry properties directly into the architecture, equivariant models achieve greater physical consistency, improved data efficiency, and superior generalization compared with conventional neural networks.

---

## Transition to Chapter 20.12 — Spherical Harmonics

We have now established the mathematical framework of irreducible representations and seen how they are used within modern equivariant neural networks. However, one crucial question remains:

**How are these irreducible representations computed in practice?**

The answer lies in **spherical harmonics**—a family of special mathematical functions that form the angular basis of SO(3) representations. Spherical harmonics are the language through which geometric information is encoded in **e3nn**, **NequIP**, **Allegro**, and **MACE**. In the next major section, we will develop spherical harmonics from first principles and show why they are indispensable for modern geometric deep learning.

# Chapter 20.12 Spherical Harmonics

## 20.12.1 Introduction

In the previous chapter, we developed the mathematical foundation of **equivariant graph neural networks**.

We introduced

* SO(3),
* O(3),
* E(3),
* tensor features,
* representations,
* irreducible representations (irreps),
* direct sums,
* basis transformations.

However, one fundamental question still remains unanswered.

> **How are irreducible representations actually represented numerically inside a neural network?**

The answer is through **spherical harmonics**.

Spherical harmonics provide the mathematical functions that describe how quantities vary with **direction**.

They form the angular basis of the irreducible representations of SO(3) and are one of the most important mathematical tools in

* quantum mechanics,
* crystallography,
* electromagnetism,
* computer graphics,
* signal processing,
* geometric deep learning.

Every modern equivariant graph neural network—including **e3nn**, **NequIP**, **Allegro**, and **MACE**—uses spherical harmonics to encode angular information.

---

# 20.12.2 Why Do We Need Spherical Harmonics?

Consider two neighboring atoms.

```text id="sh1"
Atom A

↓

Atom B
```

The distance between the atoms is important.

However,

distance alone is not sufficient.

Suppose another atom lies at exactly the same distance but in a different direction.

```text id="sh2"
      O

      |

O — Si
```

The local chemistry is different.

The neural network must therefore distinguish

* **how far**
* **in which direction**

neighbors are located.

Distances describe the radial information.

Directions require angular information.

Spherical harmonics provide this angular description.

---

# 20.12.3 Radial vs Angular Information

Every neighbor position can be separated into two independent parts.

* Radial information
* Angular information

Mathematically,

a position vector

[
\mathbf r
]

can be written as

[
(r,\theta,\phi),
]

where

* (r) is the distance,
* (\theta) is the polar angle,
* (\phi) is the azimuthal angle.

The radial part tells us **how far** the neighbor is.

The angular part tells us **where** it is.

---

# 20.12.4 An Everyday Analogy

Imagine describing the location of a city.

You could say

> 150 km away.

This gives only the distance.

You also need the direction.

For example,

> 150 km northeast.

Distance alone cannot uniquely specify the location.

Likewise,

atomic environments require both radial and angular information.

---

# 20.12.5 Cartesian Coordinates

Most machine learning datasets use Cartesian coordinates.

An atom is represented as

[
(x,y,z).
]

For example,

```text id="sh3"
(2.1, 1.5, −0.8)
```

These coordinates are convenient for computation,

but they do not naturally separate distance from direction.

---

# 20.12.6 Spherical Coordinates

Instead,

we can describe the same point using spherical coordinates.

```text id="sh4"
        z

        ↑

       •

     /|

   /  |

 θ    |

↓

x ------ y
```

The coordinates become

[
(r,\theta,\phi).
]

This representation naturally separates

* radial distance,
* angular orientation.

---

# 20.12.7 Why Spherical Coordinates Are Better

Rotations primarily affect

* (\theta),
* (\phi),

while leaving

[
r
]

unchanged.

Therefore,

rotational symmetry becomes much easier to describe in spherical coordinates.

This is one reason why spherical harmonics are built upon spherical coordinates rather than Cartesian coordinates.

---

# 20.12.8 Separating Geometry

Many physical equations become dramatically simpler after separating radial and angular components.

Conceptually,

```text id="sh5"
Position

↓

Distance

+

Direction
```

The radial information can be handled by one neural-network component.

The angular information can be handled using spherical harmonics.

---

# 20.12.9 What Are Spherical Harmonics?

Spherical harmonics are a family of mathematical functions defined on the surface of a sphere.

They depend only on

[
\theta
]

and

[
\phi.
]

Unlike ordinary functions,

they describe **angular patterns** rather than one-dimensional curves.

Each spherical harmonic corresponds to one irreducible representation of SO(3).

---

# 20.12.10 Mathematical Notation

Spherical harmonics are written as

[
Y_l^m(\theta,\phi).
]

The notation contains two integers.

* (l) — degree
* (m) — order

These indices determine the angular pattern of the function.

---

# 20.12.11 Meaning of (l)

The integer

[
l
=

0,1,2,3,\ldots
]

is exactly the same angular momentum quantum number introduced in the previous chapter.

It determines

* angular complexity,
* rotational behavior,
* dimension of the corresponding irrep.

---

# 20.12.12 Meaning of (m)

For each value of

(l),

the integer

[
m
]

takes values

[
-l,
-l+1,
\ldots,
l-1,
l.
]

Therefore,

there are

[
2l+1
]

different spherical harmonics for every value of

(l).

This matches the dimension of the SO(3) irreducible representation.

---

# 20.12.13 The First Few Harmonics

The first few families are

| Degree (l) | Number of Functions |
| ---------- | ------------------- |
| 0          | 1                   |
| 1          | 3                   |
| 2          | 5                   |
| 3          | 7                   |
| 4          | 9                   |

Again,

the familiar

[
2l+1
]

rule appears.

---

# 20.12.14 Why This Is Important

Notice the remarkable correspondence.

| SO(3) Irrep | Spherical Harmonics |
| ----------- | ------------------- |
| (l=0)       | 1 harmonic          |
| (l=1)       | 3 harmonics         |
| (l=2)       | 5 harmonics         |
| (l=3)       | 7 harmonics         |

Spherical harmonics provide the concrete mathematical realization of the irreducible representations introduced earlier.

---

# 20.12.15 Physical Interpretation

Each spherical harmonic describes a different angular pattern on the surface of a sphere.

For example,

* (l=0) describes a completely uniform sphere.
* (l=1) describes dipole-like patterns.
* (l=2) describes quadrupole-like patterns.
* Higher values describe increasingly complex angular distributions.

These patterns are exactly the angular structures needed to describe atomic environments.

---

# 20.12.16 Connection to Atomic Orbitals

Chemistry students have already encountered spherical harmonics without realizing it.

Atomic orbitals are constructed from spherical harmonics.

For example,

| Orbital | (l) |
| ------- | --- |
| s       | 0   |
| p       | 1   |
| d       | 2   |
| f       | 3   |

The familiar orbital shapes arise because the angular part of the Schrödinger equation is solved using spherical harmonics.

---

# 20.12.17 Why Machine Learning Uses Them

Modern equivariant neural networks also need to describe angular geometry.

Instead of electron probability,

they describe

* neighboring atoms,
* bond directions,
* local crystal environments.

The mathematical problem is identical.

Consequently,

the same spherical harmonics used in quantum mechanics naturally appear in geometric deep learning.

---

# 20.12.18 Key Takeaways

Spherical harmonics are mathematical functions that describe **angular information** on the surface of a sphere. They provide the concrete realization of the irreducible representations of **SO(3)** and naturally separate directional information from radial distance.

This makes them indispensable for modeling atomic environments, where both distance and orientation determine material properties. As a result, spherical harmonics have become one of the core mathematical tools underlying modern equivariant graph neural networks such as **e3nn**, **NequIP**, **Allegro**, and **MACE**.

---

## Transition to Section 20.12.2 — Mathematical Definition of Spherical Harmonics

In this introductory section, we developed an intuitive understanding of why spherical harmonics are needed. In the next section, we will derive their mathematical definition, explain the roles of the degree (l) and order (m), introduce the associated Legendre polynomials, and show how the complete family of spherical harmonics is constructed. This mathematical foundation will be essential before using spherical harmonics in equivariant message passing and neural network implementations later in the chapter.

# 20.12.2 Mathematical Definition of Spherical Harmonics

In the previous section, we developed an intuitive understanding of spherical harmonics.

We learned that they

* describe angular information,
* form the irreducible representations of SO(3),
* are used extensively in quantum mechanics and equivariant graph neural networks.

We now study their mathematical definition.

Although the equations may initially appear complicated, every component has a clear geometric meaning. By the end of this section, you will understand how spherical harmonics are constructed and why they naturally describe rotational symmetry.

---

# 20.12.3 Coordinates on a Sphere

Spherical harmonics are defined on the surface of a sphere.

Instead of Cartesian coordinates

[
(x,y,z),
]

we use spherical coordinates

[
(r,\theta,\phi).
]

These coordinates have the following meanings:

* (r) — distance from the origin,
* (\theta) — polar angle measured from the positive (z)-axis,
* (\phi) — azimuthal angle measured around the (z)-axis.

Graphically,

```text id="shm1"
           z

           ↑

          •

         /|

        / |

     θ /  |

      /   |

x ---------- y

     φ
```

Notice that spherical harmonics depend only on

[
\theta
]

and

[
\phi,
]

not on

[
r.
]

This is because they describe **direction**, not distance.

---

# 20.12.4 Separation of Variables

Many physical equations become easier to solve after separating variables.

For a point in space,

we separate

```text id="shm2"
Position

↓

Radial Part

+

Angular Part
```

Mathematically,

the dependence becomes

[
f(r,\theta,\phi)
================

R(r),
Y(\theta,\phi),
]

where

* (R(r)) is the radial function,
* (Y(\theta,\phi)) is the angular function.

The angular function is precisely the spherical harmonic.

---

# 20.12.5 General Definition

The complex spherical harmonic is written as

[
Y_l^{,m}(\theta,\phi).
]

Its mathematical form is

[
Y_l^{,m}(\theta,\phi)
=====================

N_l^{,m}
P_l^{,m}(\cos\theta)
e^{im\phi},
]

where

* (N_l^{,m}) is a normalization constant,
* (P_l^{,m}) is the associated Legendre polynomial,
* (e^{im\phi}) describes the azimuthal variation.

Although this expression may look intimidating, each factor plays a distinct role.

---

# 20.12.6 Understanding Each Term

The equation consists of three pieces.

### 1. Normalization Constant

The normalization constant

[
N_l^{,m}
]

ensures that every spherical harmonic has unit magnitude over the sphere.

This allows different harmonics to be compared consistently.

---

### 2. Associated Legendre Polynomial

The function

[
P_l^{,m}(\cos\theta)
]

controls how the harmonic changes with the polar angle.

Different values of

(l)

and

(m)

produce different angular patterns.

---

### 3. Complex Exponential

The factor

[
e^{im\phi}
]

controls how the function varies around the azimuthal direction.

It introduces the rotational behavior about the (z)-axis.

---

# 20.12.7 Degree (l)

The degree

[
l
=

0,1,2,\ldots
]

determines the overall complexity of the angular pattern.

As

(l)

increases,

the number of oscillations on the sphere also increases.

Higher values correspond to increasingly detailed angular structures.

---

# 20.12.8 Order (m)

For each

(l),

the order

[
m
]

can take the values

[
-l,-l+1,\ldots,l.
]

Thus,

each degree contains

[
2l+1
]

different spherical harmonics.

For example,

| (l) | Allowed (m)            |
| --- | ---------------------- |
| 0   | 0                      |
| 1   | -1, 0, 1               |
| 2   | -2, -1, 0, 1, 2        |
| 3   | -3, -2, -1, 0, 1, 2, 3 |

This again matches the dimension of the SO(3) irrep.

---

# 20.12.9 Example: (l=0)

For

[
l=0,
]

there is only one possible value,

[
m=0.
]

The corresponding spherical harmonic is constant over the entire sphere.

Graphically,

```text id="shm3"
Uniform Sphere

████████████
████████████
████████████
```

Every direction has exactly the same value.

This corresponds to a scalar.

---

# 20.12.10 Example: (l=1)

For

[
l=1,
]

there are three harmonics.

These produce dipole-like angular patterns.

Conceptually,

```text id="shm4"
Positive

████

────

Negative
```

The sphere is divided into positive and negative regions.

These harmonics correspond to vector-like behavior.

---

# 20.12.11 Example: (l=2)

For

[
l=2,
]

five independent harmonics exist.

These describe quadrupole patterns with alternating positive and negative lobes.

Compared with

(l=1),

the angular structure becomes more intricate.

---

# 20.12.12 Increasing Angular Complexity

As

(l)

increases,

the surface pattern becomes progressively more detailed.

Conceptually,

```text id="shm5"
l = 0

↓

Smooth
```

```text id="shm6"
l = 1

↓

Two Lobes
```

```text id="shm7"
l = 2

↓

Four Lobes
```

```text id="shm8"
l = 3

↓

Multiple Lobes
```

Each higher degree captures finer directional information.

---

# 20.12.13 Orthogonality

One of the most important properties of spherical harmonics is **orthogonality**.

Different harmonics are mutually independent.

Mathematically,

[
\int
Y_l^{,m}
Y_{l'}^{,m'*}
,d\Omega
========

\delta_{ll'}
\delta_{mm'}.
]

This equation means that different spherical harmonics do not overlap.

Each harmonic carries unique angular information.

---

# 20.12.14 Why Orthogonality Matters

Orthogonality allows complicated angular functions to be expanded uniquely as sums of spherical harmonics.

This is analogous to Fourier series,

where complicated one-dimensional signals are expanded using sine and cosine functions.

Spherical harmonics perform the same role for functions defined on the sphere.

---

# 20.12.15 Spherical Harmonics as a Basis

Because of orthogonality,

the complete family of spherical harmonics forms a basis for angular functions.

Conceptually,

```text id="shm9"
Complex Angular Pattern

↓

Expand

↓

Y₀

+

Y₁

+

Y₂

+

Y₃

+ ...
```

Any sufficiently smooth angular function can be represented by an appropriate combination of spherical harmonics.

---

# 20.12.16 Why Neural Networks Use This Basis

Atomic environments are fundamentally angular.

Neighboring atoms occupy different directions around a central atom.

Instead of describing these directions directly in Cartesian coordinates,

equivariant neural networks project them onto spherical harmonic basis functions.

This converts geometric information into symmetry-adapted features.

---

# 20.12.17 Connection to Irreducible Representations

Recall that

* (l=0) corresponds to a scalar irrep,
* (l=1) corresponds to a vector irrep,
* (l=2) corresponds to a rank-2 irrep.

The spherical harmonics

[
Y_l^{,m}
]

provide the explicit mathematical realization of these irreducible representations.

This is why spherical harmonics and irreps always appear together.

---

# 20.12.18 Key Takeaways

Spherical harmonics are mathematical functions defined on the surface of a sphere. Each harmonic is indexed by a degree (l) and an order (m), with (2l+1) functions for every value of (l). Their orthogonality allows them to serve as a complete basis for angular functions, making them the natural language for describing rotational symmetry.

In equivariant graph neural networks, spherical harmonics transform atomic directions into symmetry-adapted features that obey the irreducible representations of SO(3). They therefore provide the mathematical foundation for encoding angular information in models such as **e3nn**, **NequIP**, **Allegro**, and **MACE**.

---

## Transition to Section 20.12.3 — Visualizing Spherical Harmonics

Although the mathematical definition explains how spherical harmonics are constructed, their true significance becomes much clearer when viewed geometrically. In the next section, we will visualize the first few spherical harmonics ((l=0,1,2,3)), interpret their positive and negative lobes, connect them to familiar atomic orbitals, and develop an intuitive understanding of how increasing (l) captures increasingly complex angular structure.

# 20.12.3 Visualizing Spherical Harmonics

The mathematical definition of spherical harmonics provides a rigorous description of angular functions.

However, equations alone rarely provide intuition.

To truly understand spherical harmonics, it is helpful to **visualize them as patterns on the surface of a sphere**.

In this section, we will build an intuitive understanding of these functions by examining how they look, how they change with the angular momentum quantum number (l), and why they are so effective for representing atomic environments.

---

# 20.12.4 A Sphere as an Angular Canvas

Imagine a perfectly smooth sphere centered at the origin.

Every point on its surface corresponds to one direction in three-dimensional space.

```text id="vis1"
          North Pole

              ●

         •         •

      •               •

West ●                 ● East

      •               •

         •         •

              ●

         South Pole
```

Instead of assigning a color to represent temperature or elevation, imagine assigning a numerical value given by a spherical harmonic.

The sphere becomes a map of angular information.

---

# 20.12.5 Positive and Negative Regions

A spherical harmonic assigns either positive or negative values to different directions.

Conceptually,

```text id="vis2"
++++++

++++++

------

------
```

The boundary separating positive and negative regions is called a **nodal surface**.

Across a nodal surface, the value of the spherical harmonic changes sign.

---

# 20.12.6 The Simplest Harmonic: (l = 0)

The first spherical harmonic is

[
Y_0^0.
]

It is constant over the entire sphere.

Every direction has exactly the same value.

Graphically,

```text id="vis3"
████████████

████████████

████████████
```

There are

* no positive regions,
* no negative regions,
* no nodal lines.

This represents perfect rotational symmetry.

---

# 20.12.7 Physical Meaning of (l = 0)

Since every direction is identical,

(l=0)

describes scalar quantities such as

* total energy,
* atomic charge,
* density,
* temperature.

Rotating the sphere does not change anything because every direction already looks the same.

---

# 20.12.8 The First Directional Pattern: (l = 1)

The next family corresponds to

[
l=1.
]

There are three independent functions,

corresponding to

[
m=-1,0,1.
]

Each function resembles a dipole.

Conceptually,

```text id="vis4"
++++++

++++++

------

------
```

Half the sphere is positive.

Half is negative.

---

# 20.12.9 Why It Looks Like a Vector

Suppose we rotate this pattern.

The positive and negative regions rotate together.

The pattern behaves exactly like a vector.

This is why

[
l=1
]

corresponds to vector quantities.

---

# 20.12.10 Connection to p Orbitals

Chemistry students have already seen this shape.

The

* (p_x),
* (p_y),
* (p_z)

orbitals all arise from the

[
l=1
]

spherical harmonics.

Graphically,

```text id="vis5"
   (+)

    ●

    |

    |

    ●

   (−)
```

The familiar dumbbell shape comes directly from the angular variation of the spherical harmonics.

---

# 20.12.11 The Quadrupole Pattern: (l = 2)

The next family has

[
l=2.
]

Now,

five independent spherical harmonics exist.

Instead of two lobes,

the sphere develops multiple alternating positive and negative regions.

Conceptually,

```text id="vis6"
 +   -

 -   +

```

These more intricate patterns describe quadrupole symmetry.

---

# 20.12.12 Connection to d Orbitals

The familiar

* (d_{xy}),
* (d_{xz}),
* (d_{yz}),
* (d_{x^2-y^2}),
* (d_{z^2})

orbitals all originate from

[
l=2
]

spherical harmonics.

Their characteristic four-lobed and donut-shaped geometries are direct visualizations of these angular functions.

---

# 20.12.13 Higher Degrees

As the degree

[
l
]

increases,

the number of positive and negative regions increases.

Conceptually,

```text id="vis7"
l = 0

↓

Smooth Sphere
```

```text id="vis8"
l = 1

↓

Two Lobes
```

```text id="vis9"
l = 2

↓

Four Lobes
```

```text id="vis10"
l = 3

↓

Six or More Lobes
```

Each increase in

(l)

adds angular complexity.

---

# 20.12.14 Increasing Number of Nodes

The complexity of a spherical harmonic is closely related to its nodal structure.

Higher-degree harmonics contain more nodal lines and nodal planes.

These nodes divide the sphere into increasingly smaller positive and negative regions.

Consequently,

higher values of

(l)

can describe increasingly fine angular details.

---

# 20.12.15 Frequency Analogy

An excellent analogy comes from sound.

A pure musical tone has one simple frequency.

A complicated sound contains many frequencies.

Similarly,

* low values of (l) describe simple angular patterns,
* high values of (l) describe rapidly varying angular patterns.

In this sense,

(l)

plays a role analogous to frequency.

---

# 20.12.16 Fourier Series Analogy

Most students are familiar with Fourier series.

A complicated one-dimensional signal can be written as

```text id="vis11"
Constant

+

Low Frequency

+

High Frequency

+

Higher Frequency
```

Spherical harmonics perform the same role on the sphere.

Instead of sine and cosine waves,

they use angular basis functions.

---

# 20.12.17 Expanding Angular Functions

Suppose we have a complicated angular distribution around an atom.

Conceptually,

```text id="vis12"
Complex Shape

↓

Y₀

+

Y₁

+

Y₂

+

Y₃

+ ...
```

Any sufficiently smooth angular function can be represented by combining spherical harmonics.

This is one of the most powerful properties of the spherical harmonic basis.

---

# 20.12.18 Why Materials Need Higher (l)

Simple crystal environments can often be described using

[
l=0
]

or

[
l=1.
]

However,

complex coordination environments require higher-order harmonics.

Examples include

* octahedral coordination,
* tetrahedral coordination,
* distorted polyhedra,
* grain boundaries,
* defect structures.

Higher-degree harmonics capture these subtle angular differences.

---

# 20.12.19 Why Equivariant GNNs Use Multiple Degrees

Modern architectures rarely use only one value of

(l).

Instead,

they combine many degrees simultaneously.

For example,

an equivariant layer might use

```text id="vis13"
l = 0

+

l = 1

+

l = 2

+

l = 3
```

Lower degrees capture coarse geometric information,

while higher degrees capture increasingly detailed angular structure.

---

# 20.12.20 Visual Summary

The progression of spherical harmonics can be summarized as follows.

| Degree (l) | Visual Complexity      | Physical Interpretation |
| ---------- | ---------------------- | ----------------------- |
| 0          | Uniform sphere         | Scalar                  |
| 1          | Two lobes              | Vector / Dipole         |
| 2          | Four lobes             | Quadrupole              |
| 3          | Multiple lobes         | Octupole                |
| Higher     | Increasingly intricate | Higher-order tensors    |

This hierarchy mirrors the increasing complexity of atomic environments.

---

# 20.12.21 Why This Matters for Deep Learning

An equivariant neural network processes local atomic neighborhoods.

Instead of describing neighbor directions directly in Cartesian coordinates,

it expands them into spherical harmonics.

This converts raw geometric information into a hierarchy of angular features that naturally respect rotational symmetry.

As a result,

the network can distinguish between different local environments while remaining exactly equivariant under rotations.

---

# 20.12.22 Key Takeaways

Visualizing spherical harmonics reveals that they are not abstract mathematical objects but structured angular patterns on the surface of a sphere. As the degree (l) increases, these patterns become progressively more intricate, allowing increasingly detailed directional information to be represented.

This hierarchy is analogous to the role of sine and cosine functions in Fourier analysis and provides the foundation for describing atomic environments in equivariant graph neural networks. The same spherical harmonics that define atomic orbitals in quantum mechanics now serve as the angular basis functions for models such as **e3nn**, **NequIP**, **Allegro**, and **MACE**.

---

## Transition to Section 20.12.4 — Real vs. Complex Spherical Harmonics

The mathematical definition introduced earlier uses **complex-valued spherical harmonics**, which are natural in quantum mechanics because wavefunctions are generally complex. However, most machine learning models operate with **real-valued tensors**. In the next section, we will explain the difference between **complex** and **real spherical harmonics**, show how real spherical harmonics are constructed, and discuss why nearly all modern equivariant graph neural networks—including **e3nn**, **NequIP**, **Allegro**, and **MACE**—use the real form in practice.

# 20.12.4 Real vs. Complex Spherical Harmonics

In the previous sections, we introduced spherical harmonics using their standard mathematical definition

[
Y_l^m(\theta,\phi).
]

These functions are traditionally **complex-valued**.

This form naturally arises in quantum mechanics because quantum wavefunctions are generally complex.

However, modern machine learning models almost always work with **real-valued features**.

This raises an important question:

> **Why do equivariant graph neural networks use real spherical harmonics instead of complex ones?**

In this section, we answer this question by comparing complex and real spherical harmonics and explaining why the real form is preferred in geometric deep learning.

---

# 20.12.5 Why Complex Numbers Appear

Recall that the mathematical definition of a spherical harmonic contains

[
e^{im\phi}.
]

Using Euler's formula,

[
e^{ix}
======

\cos x
+
i\sin x,
]

we can write

[
e^{im\phi}
==========

\cos(m\phi)
+
i\sin(m\phi).
]

Since the expression contains the imaginary unit

[
i=\sqrt{-1},
]

the spherical harmonic is generally complex.

---

# 20.12.6 Complex-Valued Functions

A complex spherical harmonic has both

* a real part,
* an imaginary part.

Conceptually,

```text id="real1"
Y

↓

Real Part

+

Imaginary Part
```

Both parts vary smoothly over the sphere.

Together,

they completely describe the angular function.

---

# 20.12.7 Why Physics Uses Complex Functions

Quantum mechanics naturally uses complex numbers.

The Schrödinger equation,

wavefunctions,

probability amplitudes,

and phase information are all complex-valued.

Therefore,

complex spherical harmonics fit naturally into quantum theory.

For example,

the angular part of the hydrogen atom wavefunction is expressed using complex spherical harmonics.

---

# 20.12.8 Machine Learning Is Different

Neural networks usually operate on

* floating-point numbers,
* matrices,
* tensors.

Almost all of these quantities are real.

Although complex neural networks do exist,

they are considerably more expensive to train and are unnecessary for most materials applications.

Consequently,

modern equivariant GNNs prefer real-valued representations.

---

# 20.12.9 Constructing Real Spherical Harmonics

Fortunately,

the complex spherical harmonics are not the only possible basis.

By combining pairs of complex harmonics,

we can construct an equivalent **real-valued basis**.

Conceptually,

```text id="real2"
Complex Harmonics

↓

Linear Combination

↓

Real Harmonics
```

Nothing is lost.

Only the basis changes.

---

# 20.12.10 Same Information, Different Basis

This transformation is another example of a **basis transformation**, which we studied earlier.

The underlying representation of SO(3) remains exactly the same.

Only the coordinates used to describe it change.

Therefore,

complex and real spherical harmonics contain identical information.

---

# 20.12.11 Complex Basis

The complex basis consists of

[
Y_l^{-l},
\ldots,
Y_l^{0},
\ldots,
Y_l^{l}.
]

These functions are orthogonal,

complete,

and mathematically elegant.

However,

they contain imaginary numbers.

---

# 20.12.12 Real Basis

The real basis is obtained by combining

positive and negative

values of

[
m.
]

The resulting functions depend only on

* sine,
* cosine,
* associated Legendre polynomials.

They are completely real-valued.

Consequently,

they are much easier to integrate into standard deep-learning frameworks.

---

# 20.12.13 Visual Appearance

Interestingly,

real and complex spherical harmonics produce the same kinds of angular patterns.

For example,

the familiar

* p orbitals,
* d orbitals,
* f orbitals,

shown in chemistry textbooks,

are almost always drawn using **real spherical harmonics** rather than the complex ones.

---

# 20.12.14 Example: p Orbitals

The three familiar p orbitals

```text id="real3"
px

py

pz
```

are real combinations of the complex

[
Y_1^{-1},
\quad
Y_1^{0},
\quad
Y_1^{1}.
]

Chemists almost never use the complex versions when visualizing orbitals.

---

# 20.12.15 Example: d Orbitals

Similarly,

the five familiar d orbitals

```text id="real4"
dxy

dxz

dyz

dx²−y²

dz²
```

are also constructed from real spherical harmonics.

These shapes are much easier to interpret than their complex counterparts.

---

# 20.12.16 Why Machine Learning Prefers the Real Basis

There are several practical advantages.

Real spherical harmonics

* eliminate complex arithmetic,
* reduce computational cost,
* simplify implementation,
* integrate naturally with PyTorch,
* improve numerical efficiency.

For these reasons,

they have become the standard choice in geometric deep learning.

---

# 20.12.17 What e3nn Uses

The **e3nn** library uses **real spherical harmonics** by default.

Every directional feature is expanded using this real basis.

Consequently,

all hidden tensor features remain real-valued throughout the network.

This greatly simplifies both training and inference.

---

# 20.12.18 What NequIP Uses

**NequIP** is built directly upon **e3nn**.

Therefore,

it also employs real spherical harmonics.

Neighbor directions are projected into the real harmonic basis before message passing begins.

---

# 20.12.19 What Allegro Uses

**Allegro** likewise uses real spherical harmonics.

Although its message-passing strategy differs from NequIP,

its angular encoding follows exactly the same mathematical foundation.

---

# 20.12.20 What MACE Uses

**MACE** also relies on real spherical harmonics.

Combined with higher-order tensor products,

these harmonics enable MACE to construct highly expressive equivariant representations of local atomic environments.

---

# 20.12.21 Do We Lose Information?

An important question is whether converting to the real basis removes information.

The answer is

**No.**

Real and complex spherical harmonics span exactly the same vector space.

The transformation between them is simply a change of basis.

Therefore,

the expressive power of the representation remains unchanged.

---

# 20.12.22 Comparison

The differences can be summarized as follows.

| Complex Spherical Harmonics   | Real Spherical Harmonics          |
| ----------------------------- | --------------------------------- |
| Complex-valued                | Real-valued                       |
| Natural in quantum mechanics  | Natural in machine learning       |
| Uses complex exponentials     | Uses sine and cosine combinations |
| Elegant mathematically        | Efficient computationally         |
| Common in theoretical physics | Common in equivariant GNNs        |

---

# 20.12.23 Why Understanding Both Is Important

Researchers working in

* quantum chemistry,
* condensed matter physics,
* computational materials science,

often encounter the complex form in textbooks and research papers.

On the other hand,

those implementing equivariant neural networks will primarily use the real form.

Understanding that these are simply two different bases for the same mathematical object prevents unnecessary confusion.

---

# 20.12.24 Key Takeaways

Complex and real spherical harmonics describe the same angular information but in different bases. Complex harmonics arise naturally in quantum mechanics because wavefunctions are complex-valued, whereas real harmonics are obtained through a basis transformation that removes imaginary components.

Because modern deep-learning frameworks operate primarily on real-valued tensors, nearly all equivariant graph neural networks—including **e3nn**, **NequIP**, **Allegro**, and **MACE**—use **real spherical harmonics**. This choice preserves all the mathematical properties of the SO(3) irreducible representations while greatly simplifying computation.

---

## Transition to Section 20.12.5 — Spherical Harmonics as the Basis of SO(3) Irreducible Representations

We have now seen how spherical harmonics are defined and why their real-valued form is preferred in modern machine learning. The next step is to understand their deeper mathematical role. In the next section, we will show that **spherical harmonics are not merely useful angular functions—they are the explicit basis functions of the irreducible representations of the rotation group SO(3)**. This connection explains why every modern equivariant graph neural network relies on spherical harmonics to achieve exact rotational equivariance.

# 20.12.5 Spherical Harmonics as the Basis of SO(3) Irreducible Representations

By now, we have studied two important concepts independently:

* **Irreducible representations (irreps)** of the rotation group **SO(3)**.
* **Spherical harmonics** as functions defined on the surface of a sphere.

At first glance, these topics may appear unrelated.

However, they are in fact **two different views of exactly the same mathematical object**.

This is one of the most profound ideas in representation theory and forms the mathematical foundation of modern equivariant graph neural networks.

---

# 20.12.6 The Missing Connection

Recall that every irreducible representation of SO(3) is labeled by an integer

[
l=0,1,2,\ldots
]

For each value of

[
l,
]

the representation has dimension

[
2l+1.
]

Now recall the spherical harmonics.

For every degree

[
l,
]

there are exactly

[
2l+1
]

independent functions

[
Y_l^{-l},
Y_l^{-l+1},
\ldots,
Y_l^{,l}.
]

This is not a coincidence.

These functions **are the basis vectors of the SO(3) irreducible representation**.

---

# 20.12.7 Functions as Vectors

In linear algebra,

we usually think of vectors as columns of numbers.

For example,

[
\begin{bmatrix}
1\
2\
3
\end{bmatrix}.
]

Representation theory extends this idea.

A vector does not have to be a list of numbers.

A vector can also be

* a polynomial,
* a signal,
* a wavefunction,
* or a function on a sphere.

Therefore,

the spherical harmonics themselves form a vector space.

---

# 20.12.8 Basis Functions

Consider the case

[
l=1.
]

There are three functions,

[
Y_1^{-1},
\quad
Y_1^{0},
\quad
Y_1^{1}.
]

These three functions span a three-dimensional vector space.

Conceptually,

```text id="basisfunc1"
Y₁⁻¹

+

Y₁⁰

+

Y₁¹
```

This space is precisely the

three-dimensional irreducible representation of SO(3).

---

# 20.12.9 General Case

For any value of

[
l,
]

the basis consists of

[
Y_l^{-l},
Y_l^{-l+1},
\ldots,
Y_l^{l}.
]

Since there are

[
2l+1
]

functions,

they span a

[
2l+1
]

dimensional vector space.

This vector space is the irrep labeled by

[
l.
]

---

# 20.12.10 Rotation Acts on the Basis

Suppose we rotate the sphere.

Every spherical harmonic changes.

However,

the rotated function is still a combination of the same basis functions.

Conceptually,

```text id="basisfunc2"
Rotate Sphere

↓

Y₁⁰

↓

Combination of

Y₁⁻¹

Y₁⁰

Y₁¹
```

Rotation mixes the basis functions,

but never leaves the vector space.

This is exactly how an irreducible representation behaves.

---

# 20.12.11 Matrix Representation of Rotation

Suppose

[
\mathbf Y_l
]

contains all spherical harmonics of degree

[
l.
]

After a rotation

[
R,
]

the basis transforms as

[
\mathbf Y_l'
============

D^{(l)}(R)
\mathbf Y_l,
]

where

[
D^{(l)}(R)
]

is the irreducible representation matrix of SO(3).

This equation establishes the connection between spherical harmonics and representation theory.

---

# 20.12.12 Why This Is Remarkable

Notice what has happened.

A geometric rotation of the sphere has become an ordinary matrix multiplication.

Instead of manipulating complicated functions,

we only need to multiply vectors by matrices.

This greatly simplifies both mathematics and computation.

---

# 20.12.13 Example: (l = 0)

For

[
l=0,
]

the vector space contains only one basis function,

[
Y_0^0.
]

Rotating the sphere leaves this function unchanged.

Therefore,

the representation matrix is simply

[
[1].
]

This is the scalar representation.

---

# 20.12.14 Example: (l = 1)

For

[
l=1,
]

the basis contains three functions.

A rotation mixes these three functions together.

Mathematically,

the representation matrix has size

[
3\times3.
]

This is exactly the familiar vector representation.

---

# 20.12.15 Example: (l = 2)

For

[
l=2,
]

there are five basis functions.

Rotations mix only these five functions.

The corresponding representation matrix therefore has size

[
5\times5.
]

No components from other values of

(l)

appear.

This is the defining property of an irreducible representation.

---

# 20.12.16 Why Different Degrees Never Mix

A crucial consequence of irreducibility is that rotations never mix different values of

[
l.
]

For example,

a rotation cannot transform

[
Y_1^m
]

into

[
Y_2^m.
]

Instead,

rotations only mix functions having the same degree.

Conceptually,

```text id="basisfunc3"
l = 1

↓

Only mixes with

l = 1
```

```text id="basisfunc4"
l = 2

↓

Only mixes with

l = 2
```

Each degree defines an independent symmetry space.

---

# 20.12.17 Why This Matters for Neural Networks

Suppose a hidden feature belongs to

[
l=2.
]

After rotating the crystal,

that feature must remain inside the

[
l=2
]

representation.

It should never become

* a scalar,
* a vector,
* or another unrelated tensor.

This constraint guarantees exact rotational equivariance.

---

# 20.12.18 The Role of e3nn

The **e3nn** library stores tensor features according to their irreducible representations.

Internally,

each feature is expanded in the basis of real spherical harmonics.

During every rotation,

the corresponding representation matrix

[
D^{(l)}(R)
]

updates the coefficients.

The basis functions themselves remain unchanged.

---

# 20.12.19 A Signal Processing Analogy

Imagine representing a musical sound using Fourier series.

The basis functions

* sine,
* cosine,

remain fixed.

Only the coefficients change.

Similarly,

spherical harmonics remain fixed basis functions,

while the coefficients describing the atomic environment change from one atom to another.

---

# 20.12.20 Atomic Environment Expansion

Suppose a central atom has many neighbors.

Its angular environment can be written conceptually as

```text id="basisfunc5"
Environment

↓

Y₀

+

Y₁

+

Y₂

+

Y₃

+ ...
```

Each coefficient measures how strongly that spherical harmonic contributes to the local geometry.

This expansion forms the starting point of many equivariant message-passing algorithms.

---

# 20.12.21 Why This Is the Perfect Basis

The spherical harmonic basis possesses several remarkable properties.

It is

* complete,
* orthogonal,
* symmetry-adapted,
* mathematically elegant,
* physically meaningful.

Most importantly,

it transforms exactly according to the irreducible representations of SO(3).

No other basis provides these advantages simultaneously.

---

# 20.12.22 From Mathematics to Deep Learning

The workflow inside an equivariant neural network now becomes clearer.

```text id="basisfunc6"
Neighbor Coordinates

↓

Directions

↓

Spherical Harmonics

↓

Irreducible Features

↓

Message Passing

↓

Prediction
```

Every stage preserves rotational symmetry.

---

# 20.12.23 Connection to Modern Architectures

All major equivariant architectures follow this principle.

* **e3nn** uses real spherical harmonics as irrep bases.
* **NequIP** projects neighbor directions into these bases before message passing.
* **Allegro** constructs local equivariant interactions using the same representation.
* **MACE** builds higher-order tensor products from spherical harmonic expansions.

Although the architectures differ,

their mathematical foundation is identical.

---

# 20.12.24 Key Takeaways

Spherical harmonics are not merely convenient angular functions—they are the **basis functions of the irreducible representations of SO(3)**. For each degree (l), the (2l+1) spherical harmonics span the corresponding irreducible representation, and rotations act by mixing only the basis functions within that degree through the representation matrix (D^{(l)}(R)).

This relationship provides the mathematical bridge between representation theory and geometric deep learning. By expressing atomic directions in the spherical harmonic basis, equivariant graph neural networks obtain features that transform exactly according to the symmetry of three-dimensional rotations.

---

## Transition to Section 20.12.6 — Computing Spherical Harmonics from Atomic Coordinates

Understanding the mathematical role of spherical harmonics is only the first step. In practical machine-learning applications, we begin with Cartesian atomic coordinates rather than angles or basis functions. The next section will show **how a pair of atomic coordinates is converted into spherical harmonic features**, following the complete pipeline used inside **e3nn**, **NequIP**, **Allegro**, and **MACE**, from neighbor vectors to real spherical harmonic values that serve as inputs to equivariant message passing.

# 20.12.6 Computing Spherical Harmonics from Atomic Coordinates

So far, we have treated spherical harmonics as mathematical functions defined on the surface of a sphere.

However, an equivariant graph neural network never starts with spherical coordinates.

Instead, the input is simply a list of atomic positions.

For example,

```text
Atom 1 : (1.25, 0.80, 2.10)

Atom 2 : (2.40, 1.10, 3.05)

Atom 3 : (0.70, 2.30, 1.45)
```

These are ordinary Cartesian coordinates.

The neural network must somehow transform these coordinates into spherical harmonic features.

This conversion is one of the most important steps in every modern equivariant GNN.

---

# 20.12.7 The Overall Pipeline

Nearly all equivariant architectures follow the same sequence of operations.

```text id="pipeline1"
Atomic Coordinates

↓

Neighbor Vector

↓

Distance + Direction

↓

Spherical Coordinates

↓

Real Spherical Harmonics

↓

Equivariant Message Passing
```

Although different models implement this pipeline differently,

the underlying mathematics is identical.

---

# 20.12.8 Step 1 — Atomic Coordinates

Suppose two neighboring atoms have positions

[
\mathbf r_i
===========

(x_i,y_i,z_i)
]

and

[
\mathbf r_j
===========

(x_j,y_j,z_j).
]

For example,

```text id="pipeline2"
Atom i

↓

(1.2, 0.8, 2.0)

Atom j

↓

(2.4, 1.5, 3.1)
```

These coordinates contain all geometric information.

---

# 20.12.9 Step 2 — Neighbor Vector

The first operation is computing the relative position between the two atoms.

The neighbor vector is

[
\mathbf r_{ij}
==============

## \mathbf r_j

\mathbf r_i.
]

This vector points from atom

(i)

toward atom

(j).

Graphically,

```text id="pipeline3"
Atom i

•

────────▶

•

Atom j
```

The direction of this vector is what ultimately determines the spherical harmonic values.

---

# 20.12.10 Why Relative Coordinates?

Notice that absolute coordinates are never used.

Only the relative vector matters.

If the entire crystal is translated,

the neighbor vector remains unchanged.

This immediately guarantees **translational invariance**.

---

# 20.12.11 Step 3 — Compute the Distance

The length of the neighbor vector is

[
r
=

|\mathbf r_{ij}|.
]

This quantity represents the interatomic distance.

Distance is used by the radial part of the neural network.

It is **not** used directly by the spherical harmonics.

---

# 20.12.12 Step 4 — Normalize the Direction

The direction is obtained by dividing the vector by its length.

[
\hat{\mathbf r}
===============

\frac{\mathbf r}{|\mathbf r|}.
]

This produces a unit vector.

Every unit vector lies on the surface of the unit sphere.

Consequently,

every bond direction corresponds to one point on a sphere.

---

# 20.12.13 Why Normalization Is Necessary

Consider two bonds.

```text id="pipeline4"
Short Bond

────▶
```

```text id="pipeline5"
Long Bond

────────────▶
```

Although their lengths differ,

their directions are identical.

Spherical harmonics should produce the same angular values for both bonds.

Normalization removes the influence of distance.

---

# 20.12.14 Step 5 — Convert to Spherical Coordinates

The normalized direction is converted into

[
(\theta,\phi).
]

Conceptually,

```text id="pipeline6"
Direction Vector

↓

Polar Angle θ

↓

Azimuth Angle φ
```

These two angles uniquely specify the direction of the bond.

---

# 20.12.15 Step 6 — Evaluate the Spherical Harmonics

Once

[
\theta
]

and

[
\phi
]

are known,

the network evaluates

[
Y_l^m(\theta,\phi).
]

For each degree

[
l,
]

all

[
2l+1
]

harmonics are computed simultaneously.

---

# 20.12.16 Example for (l=1)

Suppose the model uses

[
l=1.
]

The output consists of

three values,

corresponding to

```text id="pipeline7"
Y₁⁻¹

Y₁⁰

Y₁¹
```

Together,

these three numbers encode the bond direction.

---

# 20.12.17 Example for (l=2)

If

[
l=2,
]

five values are produced.

```text id="pipeline8"
Y₂⁻²

Y₂⁻¹

Y₂⁰

Y₂¹

Y₂²
```

These five values describe more detailed angular information.

---

# 20.12.18 Using Multiple Degrees

Modern equivariant GNNs rarely stop at one degree.

Instead,

they evaluate several values of

[
l.
]

For example,

```text id="pipeline9"
l = 0

↓

1 value
```

```text id="pipeline10"
l = 1

↓

3 values
```

```text id="pipeline11"
l = 2

↓

5 values
```

The complete angular feature is formed by concatenating all of these outputs.

---

# 20.12.19 Final Angular Feature

Suppose we use

[
l=0,
1,
2.
]

The total number of spherical harmonic values becomes

[
1
+
3
+
5
=

9.

]

Thus,

every neighbor direction is converted into a nine-dimensional angular feature vector.

---

# 20.12.20 Combining with Radial Features

Angular information alone is not sufficient.

Distance also matters.

Therefore,

equivariant GNNs combine

```text id="pipeline12"
Distance

+

Spherical Harmonics

↓

Edge Feature
```

The radial basis describes **how far** the neighbor is.

The spherical harmonics describe **where** it is.

Together,

they completely characterize the local geometry.

---

# 20.12.21 Input to Message Passing

The resulting edge feature is then supplied to the message-passing layer.

Conceptually,

```text id="pipeline13"
Atom i

↓

Edge Feature

↓

Atom j
```

Every interaction between neighboring atoms is conditioned on these symmetry-aware features.

---

# 20.12.22 Why This Guarantees Equivariance

Suppose the entire crystal rotates.

The neighbor vectors rotate.

The spherical harmonic values change according to the irreducible representation matrices of SO(3).

Since every subsequent operation in the network respects these transformation rules,

the entire model remains exactly equivariant.

This property is built into the architecture rather than learned from data.

---

# 20.12.23 Computational Perspective

From the programmer's point of view,

the process is surprisingly simple.

1. Read atomic coordinates.
2. Construct neighbor list.
3. Compute relative vectors.
4. Normalize directions.
5. Evaluate real spherical harmonics.
6. Feed the resulting features into the equivariant message-passing layer.

Although the underlying mathematics is sophisticated,

the computational pipeline is remarkably systematic.

---

# 20.12.24 Example in e3nn

The **e3nn** library provides optimized routines that compute real spherical harmonics directly from Cartesian vectors.

Internally,

the library

* converts vectors into angular representations,
* evaluates the appropriate basis functions,
* returns real-valued equivariant features.

Users therefore do not need to implement the mathematical formulas manually.

---

# 20.12.25 How Other Models Use the Same Pipeline

The same basic workflow appears in every major equivariant architecture.

* **NequIP** computes spherical harmonics for every neighbor edge before message passing.
* **Allegro** uses them to construct local equivariant interactions while avoiding repeated message passing.
* **MACE** combines spherical harmonics with higher-order tensor products to represent many-body geometric correlations.

Although their architectures differ, the initial geometric encoding follows the same mathematical principles.

---

# 20.12.26 Key Takeaways

Modern equivariant graph neural networks begin with ordinary Cartesian atomic coordinates but transform them into symmetry-aware features through a well-defined sequence of operations. Neighbor vectors are computed, normalized to isolate direction, converted into spherical coordinates, and expanded using **real spherical harmonics**.

These spherical harmonic coefficients provide a compact and mathematically rigorous description of bond directions that transform exactly according to the irreducible representations of **SO(3)**. Combined with radial basis functions, they form the edge features used throughout **e3nn**, **NequIP**, **Allegro**, and **MACE**, enabling exact rotational equivariance.

---

## Transition to Section 20.12.7 — Spherical Harmonics in Equivariant Message Passing

Computing spherical harmonics is only the first stage of the process. The next question is how these angular features actually influence communication between atoms. In the following section, we will examine **how spherical harmonic features are incorporated into equivariant message passing**, how they interact with learned atomic features through tensor products, and why this mechanism allows geometric information to propagate throughout the crystal while preserving rotational symmetry.


# 20.12.7 Spherical Harmonics in Equivariant Message Passing

In the previous section, we learned how spherical harmonics are computed from atomic coordinates.

For every neighboring pair of atoms, the network constructs an angular feature vector that describes the direction of the bond.

However, computing spherical harmonics is only the beginning.

The next question is far more important:

> **How do these spherical harmonic features actually influence the information exchanged between atoms?**

The answer lies in **equivariant message passing**.

Unlike conventional graph neural networks, equivariant models use spherical harmonics to control how messages are transmitted while preserving rotational symmetry.

---

# 20.12.8 Review of Ordinary Message Passing

Recall the general message-passing framework introduced earlier.

For every edge

[
i \rightarrow j,
]

the message is computed as

[
m_{ij}
======

\phi(h_i,h_j,e_{ij}),
]

where

* (h_i) is the feature of atom (i),
* (h_j) is the feature of atom (j),
* (e_{ij}) is the edge feature.

Graphically,

```text id="msg1"
Atom i

↓

Message

↓

Atom j
```

The edge feature determines how strongly neighboring atoms interact.

---

# 20.12.9 Edge Features in Ordinary GNNs

Traditional graph neural networks usually construct edge features from

* bond length,
* bond type,
* atomic number.

For example,

```text id="msg2"
Distance

+

Bond Type

↓

Edge Feature
```

Notice that direction is often ignored.

This makes ordinary GNNs insensitive to three-dimensional geometry.

---

# 20.12.10 Edge Features in Equivariant GNNs

Equivariant models add directional information.

The edge feature becomes

```text id="msg3"
Distance

+

Spherical Harmonics

↓

Edge Feature
```

Now every message depends not only on **how far** atoms are separated but also on **where** they are located.

---

# 20.12.11 Why Direction Matters

Consider carbon.

The four nearest neighbors in diamond all have similar bond lengths.

Yet their directions differ.

Likewise,

tetrahedral,

octahedral,

and square-planar environments may contain similar bond distances but very different angular arrangements.

Without directional information,

these environments can appear deceptively similar.

Spherical harmonics allow the network to distinguish them naturally.

---

# 20.12.12 Neighbor Geometry

Suppose atom

(i)

has three neighbors.

```text id="msg4"
      A

      •

      |

B •---i---• C
```

Each bond has

* its own distance,
* its own direction,
* its own spherical harmonic expansion.

Consequently,

every edge carries unique geometric information.

---

# 20.12.13 Angular Encoding

For each neighbor,

the network computes

```text id="msg5"
Direction

↓

Y₀

Y₁

Y₂

↓

Angular Feature
```

These features become part of the message passed between atoms.

---

# 20.12.14 Hidden Atomic Features

Every atom also possesses hidden features.

For example,

```text id="msg6"
Atom Feature

↓

Scalar

↓

Vector

↓

Tensor
```

These hidden features already transform according to irreducible representations.

---

# 20.12.15 Combining Features with Geometry

The key operation is combining

* atomic features,
* spherical harmonic features.

Conceptually,

```text id="msg7"
Atomic Feature

×

Spherical Harmonics

↓

Geometric Message
```

This multiplication is **not** ordinary multiplication.

Instead,

it is performed using **tensor products**.

---

# 20.12.16 Why Ordinary Multiplication Fails

Suppose we simply multiplied two vectors component by component.

After rotating the crystal,

the result would generally fail to transform correctly.

Rotational symmetry would be broken.

Therefore,

ordinary multiplication is mathematically invalid for equivariant features.

---

# 20.12.17 Tensor Products Preserve Symmetry

Instead,

equivariant GNNs use tensor products.

Tensor products obey the symmetry rules of SO(3).

They ensure that

* scalar outputs remain scalars,
* vector outputs remain vectors,
* tensor outputs remain tensors.

This preserves exact equivariance.

---

# 20.12.18 Conceptual Workflow

For every neighboring pair,

the computation looks like

```text id="msg8"
Atom Feature

↓

Tensor Product

↑

Spherical Harmonics

↓

Equivariant Message
```

The spherical harmonics determine **how geometry influences the interaction**.

---

# 20.12.19 Message Aggregation

Messages from all neighboring atoms are then summed.

```text id="msg9"
Neighbor 1

↓

Neighbor 2

↓

Neighbor 3

↓

Sum

↓

Updated Atom
```

Because every message transforms correctly,

their sum also transforms correctly.

Equivariance is preserved throughout the aggregation process.

---

# 20.12.20 Multiple Interaction Layers

This process repeats layer after layer.

```text id="msg10"
Coordinates

↓

Spherical Harmonics

↓

Layer 1

↓

Layer 2

↓

Layer 3

↓

Prediction
```

Each successive layer captures increasingly long-range geometric information.

---

# 20.12.21 Learning Angular Correlations

The first layer primarily learns local bond directions.

Later layers learn

* bond angles,
* coordination geometry,
* many-body interactions,
* crystal symmetry.

The network gradually builds an increasingly sophisticated geometric representation.

---

# 20.12.22 Example: Tetrahedral Environment

Consider silicon in diamond.

```text id="msg11"
       •

      /|\

     • Si •

      \|/

       •
```

Each neighbor contributes a different spherical harmonic expansion.

Together,

these expansions uniquely characterize the tetrahedral coordination.

The network can therefore recognize this environment regardless of how the crystal is rotated.

---

# 20.12.23 Example: Octahedral Environment

Now consider an octahedral environment.

```text id="msg12"
      •

      |

• — M — •

      |

      •
```

Although the bond lengths may resemble other structures,

the angular distribution is entirely different.

The spherical harmonic features immediately capture this difference.

---

# 20.12.24 Why This Improves Learning

Since rotational symmetry is built directly into the architecture,

the network no longer needs to learn it from data.

Instead,

training focuses entirely on discovering meaningful chemical relationships.

This greatly improves

* data efficiency,
* accuracy,
* generalization,
* physical consistency.

---

# 20.12.25 Connection to Modern Architectures

Although the implementation details differ,

all modern equivariant GNNs follow this general strategy.

* **e3nn** uses spherical harmonics together with tensor products to construct equivariant messages.
* **NequIP** combines neighbor features with spherical harmonics before updating atomic representations.
* **Allegro** performs similar operations while emphasizing efficient local interactions.
* **MACE** extends this idea further by incorporating higher-order many-body tensor products.

In every case,

spherical harmonics provide the angular information that guides message passing.

---

# 20.12.26 Why Spherical Harmonics Are Indispensable

Without spherical harmonics,

an equivariant message-passing network would know only

* distances,
* atom types.

It would lose nearly all information about

* bond orientations,
* angular geometry,
* local symmetry.

Consequently,

its ability to distinguish different crystal environments would be severely limited.

Spherical harmonics supply precisely the missing directional information.

---

# 20.12.27 Key Takeaways

Spherical harmonics play a central role in equivariant message passing by encoding the **direction** of every interatomic bond. Combined with radial information and atomic features through symmetry-preserving tensor products, they enable messages to carry rich geometric information while remaining exactly equivariant under rotations.

This mechanism allows modern equivariant graph neural networks to learn local coordination, bond angles, and many-body crystal geometry in a mathematically rigorous manner. Whether in **e3nn**, **NequIP**, **Allegro**, or **MACE**, spherical harmonics are the bridge between raw atomic coordinates and symmetry-aware information flow.

---

## Transition to Section 20.12.8 — Implementing Spherical Harmonics with e3nn

We now understand the mathematical role of spherical harmonics and how they participate in equivariant message passing. The next step is practical implementation. In the following section, we will learn how **e3nn** computes real spherical harmonics directly from Cartesian neighbor vectors, explore the relevant API, explain the meaning of its parameters, and demonstrate the computation with Python code that forms the foundation of real-world equivariant graph neural network implementations.

# 20.12.8 Implementing Spherical Harmonics with e3nn

Up to this point, we have focused on the mathematical theory of spherical harmonics.

We now shift our attention to practical implementation.

Fortunately, modern equivariant deep learning libraries allow us to compute spherical harmonics with only a few lines of code.

The most widely used library is **e3nn**.

Rather than implementing complicated mathematical formulas manually, e3nn provides highly optimized routines that

* compute real spherical harmonics,
* maintain rotational equivariance,
* integrate seamlessly with PyTorch,
* support GPU acceleration,
* serve as the foundation of many modern equivariant GNNs.

This section introduces the practical workflow for computing spherical harmonics using e3nn.

---

# 20.12.9 Installing e3nn

The library can be installed using pip.

```bash
pip install e3nn
```

Most research projects also install

```bash
pip install torch
```

since e3nn is built directly on top of PyTorch.

---

# 20.12.10 Importing Required Libraries

A typical program begins with

```python
import torch
from e3nn import o3
```

The module

```python
o3
```

contains

* rotations,
* irreducible representations,
* spherical harmonics,
* tensor products,

and many other SO(3)-related operations.

---

# 20.12.11 Defining Neighbor Vectors

Suppose we have three neighbor directions.

```python
import torch

vectors = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
])
```

Each row represents one neighbor vector.

Notice that these vectors are still ordinary Cartesian coordinates.

---

# 20.12.12 Why Cartesian Coordinates Are Enough

One remarkable feature of e3nn is that it **does not require the user to compute**

* polar angles,
* azimuthal angles,
* Legendre polynomials,

manually.

Internally,

the library performs all necessary coordinate transformations automatically.

The user simply provides Cartesian vectors.

---

# 20.12.13 Computing Spherical Harmonics

The simplest computation is

```python
Y = o3.spherical_harmonics(
    l=2,
    x=vectors,
    normalize=True,
    normalization="component"
)
```

This single function evaluates all real spherical harmonics of degree

[
l=2.
]

---

# 20.12.14 Understanding the Arguments

The first argument is

```python
l=2
```

which specifies the harmonic degree.

Possible choices include

```python
l=0
l=1
l=2
l=3
...
```

Each value corresponds to one SO(3) irreducible representation.

---

# 20.12.15 The Input Vectors

The argument

```python
x=vectors
```

contains the neighbor directions.

Each row represents

```text
x

y

z
```

coordinates.

The function automatically interprets these as vectors in three-dimensional space.

---

# 20.12.16 Normalization

The option

```python
normalize=True
```

normalizes every vector before computing spherical harmonics.

This removes the effect of bond length.

Only the direction remains.

This is precisely what spherical harmonics require.

---

# 20.12.17 Component Normalization

The argument

```python
normalization="component"
```

specifies how the spherical harmonics are scaled.

Other normalization conventions exist,

but

```python
component
```

is widely used in equivariant neural networks because it interacts naturally with tensor products.

---

# 20.12.18 Output Shape

Suppose

```python
l=2
```

and we provide

three neighbor vectors.

The output has shape

```python
torch.Size([3,5])
```

Why five?

Because

[
2l+1
====

5.

]

Each neighbor therefore produces five spherical harmonic coefficients.

---

# 20.12.19 Example Output

A simplified output might look like

```text
Neighbor 1

[0.41
-0.21
 0.52
 0.18
-0.36]

Neighbor 2

[...]

Neighbor 3

[...]
```

Each row represents one neighbor.

Each column corresponds to one basis function.

---

# 20.12.20 Computing Multiple Degrees

Modern equivariant GNNs usually use several degrees simultaneously.

This is straightforward.

```python
Y = o3.spherical_harmonics(
    l=[0,1,2],
    x=vectors,
    normalize=True,
    normalization="component"
)
```

Now the output contains

* one scalar feature,
* three vector features,
* five rank-2 features.

The total feature dimension becomes

[
1+3+5=9.
]

---

# 20.12.21 Irreducible Representation Notation

Instead of writing

```python
l=[0,1,2]
```

many e3nn programs use irreducible representation strings.

For example,

```python
irreps = o3.Irreps("1x0e + 1x1o + 1x2e")
```

This notation specifies

* one scalar even representation,
* one vector odd representation,
* one rank-2 even representation.

We will study irreps strings in detail later in this chapter.

---

# 20.12.22 Visualizing the Output

Conceptually,

the computation performed by e3nn is

```text
Neighbor Vector

↓

Normalize

↓

Real Spherical Harmonics

↓

Angular Features
```

Every neighbor direction becomes a structured set of symmetry-aware features.

---

# 20.12.23 Batch Computation

One major advantage of e3nn is that it processes many vectors simultaneously.

Instead of computing one bond at a time,

thousands of neighbor vectors can be evaluated in parallel.

For example,

```python
vectors = torch.randn(10000,3)

Y = o3.spherical_harmonics(
    l=2,
    x=vectors,
    normalize=True,
    normalization="component"
)
```

The computation remains fully vectorized and GPU-compatible.

---

# 20.12.24 Using the Results in a Neural Network

The spherical harmonic coefficients are rarely the final output.

Instead,

they become part of the edge features.

Conceptually,

```text
Neighbor Vector

↓

Spherical Harmonics

↓

Tensor Product

↓

Message Passing

↓

Updated Atomic Features
```

Thus,

the output of

```python
o3.spherical_harmonics()
```

is immediately consumed by subsequent equivariant layers.

---

# 20.12.25 Why Researchers Use e3nn

Although the mathematics of spherical harmonics is highly sophisticated,

e3nn hides nearly all implementation complexity.

Researchers can therefore focus on

* designing architectures,
* selecting datasets,
* optimizing training,

rather than manually implementing representation theory.

This is one reason why e3nn has become the standard software framework for equivariant geometric deep learning.

---

# 20.12.26 Best Practices

When using spherical harmonics in practice, several guidelines are recommended:

* Always normalize neighbor vectors before computing spherical harmonics.
* Choose the maximum degree (l_{\text{max}}) based on the complexity of the material system. Higher values capture richer angular information but increase computational cost.
* Use batched tensor operations rather than Python loops to maximize GPU efficiency.
* Keep the same normalization convention throughout the model to ensure compatibility with tensor products and other equivariant operations.

---

# 20.12.27 Complete Example

The following program demonstrates the complete workflow.

```python
import torch
from e3nn import o3

# Three neighbor vectors
vectors = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
])

# Compute real spherical harmonics up to l=2
Y = o3.spherical_harmonics(
    l=[0, 1, 2],
    x=vectors,
    normalize=True,
    normalization="component"
)

print("Shape:", Y.shape)
print(Y)
```

This code illustrates the core operation performed inside many equivariant graph neural networks. In larger models, these spherical harmonic features are passed directly into equivariant tensor-product layers and message-passing modules.

---

# 20.12.28 Key Takeaways

The **e3nn** library provides an efficient and user-friendly implementation of real spherical harmonics, allowing researchers to compute symmetry-adapted angular features directly from Cartesian neighbor vectors. By handling coordinate transformations, normalization, and basis evaluation internally, e3nn removes the need to implement the underlying mathematics manually.

These computed spherical harmonic coefficients form the angular component of edge features in modern equivariant graph neural networks and serve as the starting point for tensor-product operations and equivariant message passing.

---

## Transition to Section 20.12.9 — Choosing the Maximum Degree (l_{\text{max}})

A practical design question remains: **How many spherical harmonics should a model use?** Using only low degrees may miss important angular information, while very high degrees increase computational cost and memory usage. In the next section, we will discuss the role of the maximum degree (l_{\text{max}}), its effect on model expressiveness, computational complexity, and accuracy, and examine how architectures such as **NequIP**, **Allegro**, and **MACE** choose appropriate values in practice.

# 20.12.9 Choosing the Maximum Degree (l_{\text{max}})

One of the first design decisions when building an equivariant graph neural network is choosing the **maximum spherical harmonic degree**, commonly written as

[
l_{\text{max}}.
]

This parameter determines

* how much angular information the model can represent,
* how expressive the learned features become,
* how expensive the computation will be.

Choosing an appropriate value of (l_{\text{max}}) is therefore a balance between **accuracy** and **efficiency**.

---

# 20.12.10 What is (l_{\text{max}})?

Recall that spherical harmonics exist for

[
l=0,1,2,\ldots
]

Instead of using infinitely many harmonics,

we truncate the expansion at some maximum degree.

If

[
l_{\text{max}}=2,
]

then the network uses

* (l=0),
* (l=1),
* (l=2),

but ignores

* (l=3),
* (l=4),
* higher orders.

---

# 20.12.11 Why Can We Truncate?

In theory,

any angular function can be represented using infinitely many spherical harmonics.

However,

real atomic environments are relatively smooth.

Most useful geometric information is captured by the first few degrees.

Therefore,

using very large values of

[
l
]

usually provides diminishing returns.

---

# 20.12.12 Analogy with Image Resolution

Imagine looking at a photograph.

```text id="lmax1"
Low Resolution

↓

Blurry
```

```text id="lmax2"
High Resolution

↓

Sharp Details
```

The same idea applies here.

Small values of

[
l_{\text{max}}
]

capture only coarse angular information,

while larger values capture increasingly fine geometric details.

---

# 20.12.13 Information Captured by Different Degrees

Each value of

[
l
]

captures a different level of angular complexity.

| Degree  | Information Captured               |
| ------- | ---------------------------------- |
| (l=0)   | Scalar (no direction)              |
| (l=1)   | Basic directions                   |
| (l=2)   | Bond-angle geometry                |
| (l=3)   | More complex angular patterns      |
| (l\ge4) | Fine many-body directional details |

Higher degrees describe increasingly intricate atomic environments.

---

# 20.12.14 Feature Dimension Growth

Each degree contributes

[
2l+1
]

features.

Therefore,

the total number of spherical harmonic features is

[
\sum_{l=0}^{l_{\text{max}}}(2l+1).
]

This simplifies to

[
(l_{\text{max}}+1)^2.
]

For example,

| (l_{\text{max}}) | Number of Features |
| ---------------- | ------------------ |
| 0                | 1                  |
| 1                | 4                  |
| 2                | 9                  |
| 3                | 16                 |
| 4                | 25                 |
| 5                | 36                 |

Notice that the number of angular features grows quadratically.

---

# 20.12.15 Computational Cost

More spherical harmonic features mean

* more memory,
* more tensor products,
* larger intermediate tensors,
* longer training time.

Conceptually,

```text id="lmax3"
Higher lmax

↓

More Features

↓

More Computation

↓

Longer Training
```

Thus,

increasing

[
l_{\text{max}}
]

always increases computational cost.

---

# 20.12.16 Accuracy vs. Efficiency

Choosing

[
l_{\text{max}}
]

requires balancing two competing goals.

```text id="lmax4"
Small lmax

↓

Fast

↓

Less Expressive
```

```text id="lmax5"
Large lmax

↓

More Accurate

↓

More Expensive
```

The best choice depends on the complexity of the material system.

---

# 20.12.17 Example: Simple Materials

For relatively simple materials,

such as

* elemental metals,
* simple semiconductors,
* highly symmetric crystals,

values like

[
l_{\text{max}}=1
]

or

[
2
]

are often sufficient.

These systems do not require extremely detailed angular descriptions.

---

# 20.12.18 Example: Complex Materials

More challenging systems include

* amorphous materials,
* surfaces,
* grain boundaries,
* defects,
* catalysts,
* molecular crystals.

These environments exhibit much richer local geometry.

Higher values such as

[
l_{\text{max}}=3
]

or

[
4
]

may provide measurable improvements.

---

# 20.12.19 Diminishing Returns

Increasing

[
l_{\text{max}}
]

does not guarantee better performance.

Beyond a certain point,

the additional angular information becomes redundant.

Training becomes slower,

while prediction accuracy changes very little.

This phenomenon is known as **diminishing returns**.

---

# 20.12.20 Influence on Tensor Products

Recall that tensor products combine irreducible representations.

Higher values of

[
l
]

produce larger irreducible representations,

which lead to

* more tensor-product channels,
* larger coupling matrices,
* increased computational complexity.

Thus,

the cost grows faster than the number of spherical harmonic features alone.

---

# 20.12.21 Typical Values in Research

Most published equivariant GNNs use surprisingly small values.

Typical choices include

| Model               | Typical (l_{\text{max}}) |
| ------------------- | ------------------------ |
| NequIP              | 1–2                      |
| Allegro             | 1–2                      |
| MACE                | 2–3                      |
| Research prototypes | 3–4                      |

Very few models exceed

[
l_{\text{max}}=4.
]

The computational cost increases rapidly,

while the improvement often becomes marginal.

---

# 20.12.22 Why Small Values Work

At first,

it may seem surprising that

[
l_{\text{max}}=2
]

is often enough.

The reason is that

deep neural networks contain many layers.

Each layer combines information from multiple neighbors.

Although individual layers use only low-order harmonics,

the network gradually constructs highly complex geometric representations through repeated message passing.

---

# 20.12.23 Interaction with Network Depth

There are two ways to increase model capacity.

One can

* increase

[
l_{\text{max}},
]

or

* increase the number of layers.

Conceptually,

```text id="lmax6"
Higher lmax

↓

Richer Local Geometry
```

```text id="lmax7"
More Layers

↓

Longer-Range Geometry
```

Modern architectures often prefer deeper networks with modest

[
l_{\text{max}}
]

because this is computationally more efficient.

---

# 20.12.24 Choosing (l_{\text{max}}) in Practice

When designing a new model,

a sensible strategy is

1. Begin with

[
l_{\text{max}}=2.
]

2. Evaluate validation accuracy.

3. Increase to

[
l_{\text{max}}=3
]

only if the task requires richer angular information.

4. Avoid unnecessarily large values unless supported by careful benchmarking.

This approach balances computational efficiency with predictive performance.

---

# 20.12.25 Practical Example

Suppose we are predicting

* formation energy,
* band gap,
* elastic constants,

for crystalline solids.

A model with

[
l_{\text{max}}=2
]

often achieves excellent performance.

However,

if the goal is to model

* highly distorted catalysts,
* defect migration,
* amorphous structures,

then increasing

[
l_{\text{max}}
]

may improve the description of local geometry.

---

# 20.12.26 Summary of Trade-Offs

The influence of (l_{\text{max}}) can be summarized as follows.

| Small (l_{\text{max}})   | Large (l_{\text{max}}) |
| ------------------------ | ---------------------- |
| Faster training          | Slower training        |
| Lower memory usage       | Higher memory usage    |
| Simpler tensor products  | More tensor products   |
| Captures coarse geometry | Captures fine geometry |
| Better efficiency        | Greater expressiveness |

Choosing the optimal value depends on the scientific problem, available computational resources, and desired prediction accuracy.

---

# 20.12.27 Key Takeaways

The maximum spherical harmonic degree (l_{\text{max}}) determines the richness of angular information available to an equivariant graph neural network. Increasing (l_{\text{max}}) improves the model's ability to represent complex local geometries but also increases the number of spherical harmonic features, tensor-product operations, memory usage, and computational cost.

In practice, values between **1 and 3** are sufficient for most materials science applications, and deeper networks often provide greater benefits than extremely large values of (l_{\text{max}}). Selecting an appropriate value therefore requires balancing expressiveness with efficiency.

---

## Transition to Section 20.12.10 — Limitations of Spherical Harmonics

Spherical harmonics provide an elegant and mathematically rigorous description of angular geometry, but they are not a complete solution on their own. They describe **direction** but not **distance**, and higher-order angular information quickly becomes computationally expensive. In the next section, we will examine the limitations of spherical harmonic representations, understand why radial basis functions and tensor products are also essential, and see how modern architectures such as **NequIP**, **Allegro**, and **MACE** overcome these challenges.

# 20.12.10 Limitations of Spherical Harmonics

Throughout this chapter, spherical harmonics have appeared as one of the central mathematical tools behind equivariant graph neural networks.

They provide a complete basis for representing angular functions on a sphere, transform exactly according to the irreducible representations of **SO(3)**, and enable neural networks to preserve rotational symmetry.

Given these remarkable properties, one might naturally wonder:

> **Why do modern equivariant GNNs need anything else?**

Could we simply compute spherical harmonics and use them directly to predict material properties?

The answer is **no**.

Although spherical harmonics are indispensable, they describe only one aspect of atomic geometry. Real materials require additional information that spherical harmonics alone cannot provide.

This section examines the major limitations of spherical harmonics and explains why modern architectures combine them with radial basis functions, tensor products, nonlinear neural networks, and message passing.

---

# 20.12.11 Limitation 1 — Only Angular Information

The most important limitation is that spherical harmonics describe **direction**, not **distance**.

Consider two neighboring atoms.

```text id="limit1"
Atom A -------- Atom B
```

Suppose the bond length doubles.

```text id="limit2"
Atom A ---------------------- Atom B
```

The direction remains identical.

Therefore,

the spherical harmonic values remain exactly the same.

---

# 20.12.12 Why Distance Matters

Chemically,

bond length strongly influences

* bond strength,
* electron overlap,
* orbital hybridization,
* total energy,
* elastic constants.

Two atoms separated by

1.5 Å

and

4.5 Å

may have nearly identical directions,

but their physical interactions are completely different.

Spherical harmonics cannot distinguish between them.

---

# 20.12.13 Radial Basis Functions Solve This Problem

Modern equivariant GNNs therefore separate geometry into two independent parts.

```text id="limit3"
Distance

↓

Radial Basis
```

```text id="limit4"
Direction

↓

Spherical Harmonics
```

The final edge representation combines both.

---

# 20.12.14 Complete Edge Representation

Conceptually,

```text id="limit5"
Neighbor Vector

↓

Distance

+

Direction

↓

Radial Basis

+

Spherical Harmonics

↓

Edge Feature
```

This decomposition allows the network to model both radial and angular information simultaneously.

---

# 20.12.15 Limitation 2 — Local Information Only

A spherical harmonic describes one bond direction.

It does **not** automatically describe the relationships among many neighbors.

For example,

```text id="limit6"
      •

     / \

    •   •

      M
```

The individual bond directions are known,

but the complete coordination environment is not.

---

# 20.12.16 Many-Body Geometry

Materials properties often depend on

* bond angles,
* dihedral angles,
* collective atomic arrangements,
* coordination polyhedra.

These are **many-body** geometric relationships.

Spherical harmonics alone describe only one direction at a time.

Message passing is required to combine information from multiple neighbors.

---

# 20.12.17 Limitation 3 — High Degrees Become Expensive

Recall that the number of basis functions grows as

[
2l+1.
]

The total number of angular features up to

[
l_{\text{max}}
]

is

[
(l_{\text{max}}+1)^2.
]

As

[
l_{\text{max}}
]

increases,

both memory usage and computational cost grow rapidly.

---

# 20.12.18 Tensor Product Complexity

Higher-order spherical harmonics also increase the complexity of tensor products.

More irreducible representations imply

* more coupling coefficients,
* larger intermediate tensors,
* more floating-point operations.

Consequently,

very large values of

[
l_{\text{max}}
]

are rarely practical.

---

# 20.12.19 Limitation 4 — No Learning

Spherical harmonics are **fixed mathematical functions**.

They are not trainable.

Regardless of the dataset,

the functions themselves never change.

Conceptually,

```text id="limit7"
Input Direction

↓

Fixed Basis

↓

Output
```

Unlike neural network weights,

the spherical harmonics contain no learnable parameters.

---

# 20.12.20 Learning Occurs Elsewhere

The neural network learns through

* linear layers,
* tensor products,
* nonlinear activations,
* attention mechanisms,
* message-passing weights.

Spherical harmonics simply provide a symmetry-preserving coordinate system.

---

# 20.12.21 Limitation 5 — No Chemical Information

Suppose two bonds have identical geometry.

```text id="limit8"
C —— O
```

and

```text id="limit9"
Si —— Si
```

Their spherical harmonic values may be identical because their directions are identical.

However,

their chemistry is completely different.

Therefore,

atomic embeddings are also necessary.

---

# 20.12.22 Atomic Features

Modern GNNs combine

```text id="limit10"
Atomic Type

+

Distance

+

Direction
```

rather than relying on direction alone.

The chemical identity of neighboring atoms contributes just as much as geometry.

---

# 20.12.23 Limitation 6 — Sensitivity to Noise

High-degree spherical harmonics vary rapidly with angle.

A very small change in direction can produce a relatively large change in the higher-order coefficients.

For noisy datasets,

this sensitivity may reduce robustness.

Consequently,

many architectures intentionally limit

[
l_{\text{max}}
]

to relatively small values.

---

# 20.12.24 Limitation 7 — No Long-Range Information

Spherical harmonics describe only the immediate neighborhood.

They cannot directly represent

* long-range electrostatic interactions,
* long-range elastic fields,
* extended crystal defects.

These effects emerge only after multiple rounds of message passing or through additional physical modeling.

---

# 20.12.25 Why Message Passing Is Essential

Consider the following sequence.

```text id="limit11"
Layer 1

↓

Nearest Neighbors
```

```text id="limit12"
Layer 2

↓

Neighbors of Neighbors
```

```text id="limit13"
Layer 3

↓

Larger Environment
```

Although spherical harmonics encode local geometry,

repeated message passing gradually builds a representation of the entire crystal.

---

# 20.12.26 Modern Solution

Current equivariant architectures combine several complementary ideas.

```text id="limit14"
Atomic Embeddings

+

Radial Basis

+

Spherical Harmonics

+

Tensor Products

+

Message Passing

↓

Prediction
```

Each component addresses a limitation of the others.

---

# 20.12.27 How Different Models Address These Limitations

Different architectures emphasize different solutions.

* **NequIP** combines spherical harmonics with learned radial networks and equivariant message passing.
* **Allegro** reduces computational cost by using efficient local interactions while retaining equivariance.
* **MACE** captures richer many-body correlations using higher-order tensor products instead of relying solely on larger values of (l_{\text{max}}).

Although their implementations differ, they all recognize that spherical harmonics alone are insufficient.

---

# 20.12.28 Should We Increase (l_{\text{max}}) Instead?

A common misconception is that increasing

[
l_{\text{max}}
]

solves every problem.

In reality,

larger angular bases cannot replace

* better message passing,
* richer atomic embeddings,
* improved radial functions,
* deeper architectures.

Most state-of-the-art models achieve their performance through a balanced combination of all these components rather than relying on extremely high-order spherical harmonics.

---

# 20.12.29 Key Takeaways

Spherical harmonics provide an elegant and symmetry-preserving description of **bond directions**, but they represent only one component of the information needed for accurate materials modeling. They do not encode bond lengths, atomic identities, many-body interactions, or long-range effects, and higher-order harmonics quickly become computationally expensive.

Modern equivariant graph neural networks therefore combine spherical harmonics with **radial basis functions**, **atomic embeddings**, **tensor products**, and **message passing** to construct rich, physically meaningful representations of crystal structures. The power of models such as **NequIP**, **Allegro**, and **MACE** comes not from spherical harmonics alone, but from the careful integration of all these components into a unified equivariant architecture.

---

## Transition to Section 20.12.11 — From Mathematics to Complete Equivariant GNN Architectures

We have now developed the mathematical foundation of spherical harmonics and understood both their strengths and limitations. The next step is to integrate everything we have learned into a complete equivariant neural network. In the next section, we will assemble the full pipeline—from atomic coordinates to spherical harmonics, radial basis functions, tensor products, equivariant message passing, and final property prediction—providing a unified view of how modern architectures such as **NequIP**, **Allegro**, and **MACE** operate end to end.

# 20.12.11 From Mathematics to Complete Equivariant GNN Architectures

At this point in the chapter, we have studied every major mathematical ingredient used in modern equivariant graph neural networks.

Individually, we have learned about

* Euclidean symmetry,
* SO(3),
* O(3),
* irreducible representations,
* tensor features,
* spherical harmonics,
* tensor products,
* equivariant message passing.

However, understanding each concept separately is only the first step.

The real power of equivariant neural networks comes from combining all of these ideas into a single computational pipeline.

In this section, we assemble the complete architecture used by modern models such as **NequIP**, **Allegro**, and **MACE**, showing how raw atomic coordinates are transformed into accurate material property predictions while preserving rotational symmetry throughout the computation.

---

# 20.12.12 The Big Picture

Every equivariant graph neural network follows the same high-level workflow.

```text id="full1"
Crystal Structure

↓

Atomic Coordinates

↓

Graph Construction

↓

Radial Features

+

Angular Features

↓

Equivariant Message Passing

↓

Updated Atomic Features

↓

Pooling

↓

Material Property
```

Although different architectures modify individual components, this overall pipeline remains remarkably consistent.

---

# 20.12.13 Step 1 — Crystal Structure

The input is a crystal.

For example,

```text id="full2"
Si

↓

Diamond Structure
```

or

```text id="full3"
LiFePO₄

↓

Crystal Structure
```

or

```text id="full4"
MoS₂

↓

Layered Crystal
```

The network begins only with atomic positions and atomic species.

---

# 20.12.14 Step 2 — Graph Construction

The crystal is converted into a graph.

```text id="full5"
Atoms

↓

Nodes
```

```text id="full6"
Neighbor Bonds

↓

Edges
```

Edges are usually constructed using a cutoff radius.

Every atom connects only to nearby neighbors.

---

# 20.12.15 Step 3 — Atomic Embeddings

Each atom is assigned an initial learnable feature vector.

For example,

```text id="full7"
Si

↓

Embedding Vector
```

```text id="full8"
O

↓

Embedding Vector
```

```text id="full9"
Fe

↓

Embedding Vector
```

These embeddings contain chemical information rather than geometric information.

---

# 20.12.16 Step 4 — Neighbor Geometry

For every edge,

the network computes

```text id="full10"
Neighbor Vector

↓

Distance

+

Direction
```

This separates radial and angular information.

---

# 20.12.17 Step 5 — Radial Basis Functions

Distances are expanded using radial basis functions.

```text id="full11"
Distance

↓

Gaussian Basis

↓

Radial Features
```

These features describe

how far apart

two atoms are.

---

# 20.12.18 Step 6 — Spherical Harmonics

Directions are expanded using spherical harmonics.

```text id="full12"
Direction

↓

Real Spherical Harmonics

↓

Angular Features
```

These features describe

where neighboring atoms are located.

---

# 20.12.19 Step 7 — Edge Features

The radial and angular information are combined.

Conceptually,

```text id="full13"
Radial Features

+

Angular Features

↓

Edge Features
```

These edge features encode the complete local geometry of every bond.

---

# 20.12.20 Step 8 — Equivariant Tensor Products

The edge features interact with atomic features through tensor products.

```text id="full14"
Atomic Features

×

Edge Features

↓

Equivariant Features
```

Tensor products ensure that every output transforms correctly under rotation.

---

# 20.12.21 Step 9 — Message Passing

Neighboring atoms exchange information.

```text id="full15"
Neighbor Messages

↓

Aggregation

↓

Updated Atom
```

Every message respects rotational symmetry because all intermediate features are equivariant.

---

# 20.12.22 Step 10 — Multiple Layers

The message-passing process is repeated.

```text id="full16"
Layer 1

↓

Layer 2

↓

Layer 3

↓

Layer 4
```

Each layer expands the receptive field.

Atoms gradually acquire information from increasingly distant regions of the crystal.

---

# 20.12.23 Step 11 — Global Pooling

After the final message-passing layer,

all atomic features are combined.

Typical pooling operations include

* sum,
* average,
* attention pooling.

Conceptually,

```text id="full17"
Atom 1

↓

Atom 2

↓

Atom 3

↓

Pooling

↓

Crystal Feature
```

The result is a single representation of the entire material.

---

# 20.12.24 Step 12 — Property Prediction

The pooled feature is passed through a prediction head.

```text id="full18"
Crystal Feature

↓

Linear Layer

↓

Prediction
```

Possible predictions include

* formation energy,
* band gap,
* elastic constants,
* magnetic moment,
* forces,
* stress tensor.

---

# 20.12.25 Where Equivariance Appears

It is useful to identify where symmetry enters the architecture.

```text id="full19"
Coordinates

↓

Neighbor Directions

↓

Spherical Harmonics

↓

Tensor Products

↓

Message Passing
```

Every one of these operations is carefully designed to satisfy the transformation laws of SO(3).

As a result,

the entire network is exactly equivariant.

---

# 20.12.26 Data Flow Through the Network

The complete information flow can be summarized as

```text id="full20"
Atomic Coordinates

↓

Neighbor Graph

↓

Distance

↓

Radial Basis

↓

Spherical Harmonics

↓

Tensor Products

↓

Equivariant Messages

↓

Updated Features

↓

Pooling

↓

Prediction
```

This is the computational backbone of nearly every modern equivariant graph neural network.

---

# 20.12.27 Comparison with Ordinary GNNs

The difference between conventional and equivariant GNNs becomes clear.

| Ordinary GNN                         | Equivariant GNN                                  |
| ------------------------------------ | ------------------------------------------------ |
| Uses node features                   | Uses irreducible tensor features                 |
| Often ignores direction              | Encodes direction using spherical harmonics      |
| Standard linear layers               | Equivariant tensor products                      |
| Learns rotational behavior from data | Rotation symmetry built into architecture        |
| Usually scalar messages              | Scalar, vector, and higher-order tensor messages |

This built-in symmetry is the defining characteristic of equivariant models.

---

# 20.12.28 Example: Predicting Formation Energy

Suppose we wish to predict the formation energy of silicon.

The pipeline proceeds as follows.

```text id="full21"
Silicon Crystal

↓

Neighbor Graph

↓

Radial Features

+

Angular Features

↓

Equivariant Layers

↓

Crystal Embedding

↓

Formation Energy
```

Every stage respects rotational symmetry.

If the crystal is rotated,

the predicted energy remains unchanged.

---

# 20.12.29 Example: Predicting Atomic Forces

Now consider force prediction.

Unlike energy,

forces are vectors.

The pipeline becomes

```text id="full22"
Crystal

↓

Equivariant Features

↓

Vector Output

↓

Atomic Forces
```

Because the network is equivariant,

rotating the crystal automatically rotates the predicted forces by the same amount.

No additional training is required.

---

# 20.12.30 Why This Architecture Works So Well

The remarkable success of modern equivariant GNNs comes from combining

* physical symmetry,
* geometric representation theory,
* graph neural networks,
* deep learning.

Instead of forcing the network to learn rotational symmetry from examples,

the architecture guarantees it mathematically.

This greatly improves

* sample efficiency,
* prediction accuracy,
* physical consistency,
* generalization to unseen crystal orientations.

---

# 20.12.31 Relationship Between Major Architectures

Although implementations differ,

the overall pipeline remains nearly identical.

| Component                   | NequIP  | Allegro                        | MACE   |
| --------------------------- | ------- | ------------------------------ | ------ |
| Atomic embeddings           | ✓       | ✓                              | ✓      |
| Radial basis                | ✓       | ✓                              | ✓      |
| Real spherical harmonics    | ✓       | ✓                              | ✓      |
| Tensor products             | ✓       | ✓                              | ✓      |
| Equivariant message passing | ✓       | Local equivariant interactions | ✓      |
| Higher-order interactions   | Limited | Moderate                       | Strong |

The differences lie primarily in how efficiently these components are implemented and how many-body interactions are modeled.

---

# 20.12.32 Key Takeaways

A modern equivariant graph neural network is much more than a collection of mathematical concepts. It is a carefully designed computational pipeline that transforms atomic coordinates into symmetry-aware representations through graph construction, radial basis functions, spherical harmonics, tensor products, and equivariant message passing before producing material property predictions.

This unified architecture enables models such as **NequIP**, **Allegro**, and **MACE** to learn highly accurate representations of crystalline materials while rigorously respecting the rotational symmetries of three-dimensional space.

---

## Chapter Summary

This chapter introduced the mathematical foundations of **Equivariant Graph Neural Networks**, beginning with the importance of rotational symmetry in materials science and progressing through the concepts of **SO(3)**, **O(3)**, **irreducible representations**, **tensor features**, **spherical harmonics**, and **equivariant message passing**. We examined how neighbor directions are converted into spherical harmonic features, how these features participate in tensor-product interactions, and how practical libraries such as **e3nn** implement these operations efficiently.

Most importantly, we assembled these ideas into the complete computational workflow used by state-of-the-art models including **NequIP**, **Allegro**, and **MACE**. This mathematical framework now prepares us to study individual architectures in depth.

---

# Next Chapter

## **Chapter 20.13 — The e3nn Library: Building Equivariant Neural Networks in Practice**

In the next chapter, we move from theory to implementation. We will explore the **e3nn** library in detail, learn its representation system, understand tensor products and equivariant layers through code, build progressively more complex equivariant models in PyTorch, and finally implement a complete equivariant neural network from scratch before moving on to **NequIP**, **Allegro**, and **MACE**. This chapter will bridge the gap between mathematical understanding and practical research implementation.

Based on the structure you've developed, the next major section should begin the **first real architecture** after the mathematical foundations.

---

# Chapter 20.13 — e3nn: The Foundation of Equivariant Deep Learning

---

# 20.13.1 Introduction

Throughout this chapter, we have repeatedly encountered the name **e3nn**.

Whenever we discussed

* irreducible representations,
* spherical harmonics,
* tensor products,
* equivariant message passing,

we mentioned that these mathematical operations are implemented by the **e3nn** library.

At this point, you may naturally ask:

> **What exactly is e3nn?**

Is it

* a neural network?
* a graph neural network?
* an architecture like NequIP?
* or simply a software library?

The answer is subtle.

**e3nn is not a single neural network architecture.**

Instead,

it is a **general-purpose deep learning framework for building E(3)-equivariant neural networks.**

Just as **PyTorch** provides building blocks for ordinary neural networks,

**e3nn provides the building blocks for equivariant neural networks.**

Understanding e3nn is therefore essential before studying modern architectures such as

* NequIP,
* Allegro,
* MACE,
* PaiNN,
* GemNet,
* TensorNet,

because all of these models either directly use e3nn or implement the same underlying mathematical principles.

---

# 20.13.2 Why Was e3nn Created?

Before e3nn,

implementing an equivariant neural network was extremely difficult.

A researcher had to manually implement

* rotation matrices,
* irreducible representations,
* Clebsch–Gordan coefficients,
* spherical harmonics,
* tensor products,
* equivariant linear layers,
* equivariant nonlinearities.

Each of these topics comes from advanced mathematics.

Even a small implementation error would destroy rotational equivariance.

As a result,

building an equivariant neural network from scratch required months of work.

---

# 20.13.3 The Goal of e3nn

The creators of e3nn asked a simple question.

> **Can we make equivariant neural networks as easy to build as ordinary neural networks?**

Their solution was to create a library that automatically handles

* representation theory,
* group theory,
* tensor algebra,
* symmetry constraints,

allowing researchers to focus on

* model design,
* datasets,
* scientific problems,

instead of complicated mathematics.

---

# 20.13.4 PyTorch Analogy

Most readers are already familiar with PyTorch.

For example,

ordinary neural networks use

```python
torch.nn.Linear
```

instead of manually implementing

[
Wx+b.
]

Similarly,

e3nn provides

```python
o3.Linear
```

instead of manually deriving equivariant linear maps.

This dramatically simplifies implementation.

---

# 20.13.5 Position of e3nn in the Software Stack

The relationship between PyTorch and e3nn can be visualized as

```text
Application

↓

NequIP
Allegro
MACE

↓

e3nn

↓

PyTorch

↓

CUDA / CPU
```

PyTorch provides

* tensors,
* automatic differentiation,
* optimization,
* GPU acceleration.

e3nn builds on PyTorch and adds

* rotational symmetry,
* irreducible representations,
* tensor products,
* spherical harmonics.

Finally,

NequIP,

Allegro,

and MACE build complete architectures using e3nn.

---

# 20.13.6 What Does e3nn Provide?

The library contains many specialized components.

Among the most important are

* SO(3) rotations
* O(3) symmetry operations
* E(3)-equivariant operations
* Irreducible representations
* Real spherical harmonics
* Tensor products
* Clebsch–Gordan coefficients
* Equivariant linear layers
* Equivariant nonlinear activations
* Radial functions
* Gating operations

Together,

these components make it possible to construct highly expressive equivariant neural networks.

---

# 20.13.7 Core Philosophy

Unlike ordinary deep learning,

equivariant deep learning does not treat symmetry as something to be learned.

Instead,

symmetry is built directly into every layer.

Conceptually,

```text
Ordinary Deep Learning

↓

Learn Symmetry
```

```text
Equivariant Deep Learning

↓

Symmetry Built Into Architecture
```

This difference explains why equivariant networks often require

* less training data,
* fewer parameters,
* better generalization.

---

# 20.13.8 Relationship with This Book

Everything learned in the previous chapter now becomes practical.

Previously we studied

* SO(3)
* spherical harmonics
* tensor products
* irreducible representations

as mathematical concepts.

Now,

we will see how each one appears directly inside e3nn.

For example,

| Mathematics                 | e3nn Object             |
| --------------------------- | ----------------------- |
| SO(3)                       | `o3`                    |
| Irreducible representations | `Irreps`                |
| Spherical harmonics         | `spherical_harmonics()` |
| Tensor products             | `TensorProduct`         |
| Equivariant linear maps     | `Linear`                |
| Rotation matrices           | `D_from_matrix()`       |

The correspondence is almost one-to-one.

---

# 20.13.9 Learning Objectives

By the end of this chapter, you will be able to

* install e3nn,
* understand its design philosophy,
* use irreducible representations,
* compute rotations,
* compute spherical harmonics,
* construct tensor products,
* build equivariant layers,
* assemble complete equivariant neural networks,
* understand how NequIP, Allegro, and MACE are implemented internally.

Rather than treating e3nn as a "black box," you will understand the mathematical purpose of every major component.

---

# 20.13.10 Chapter Roadmap

This chapter is organized as follows:

1. Introduction to the e3nn library
2. Installation and project setup
3. Understanding `Irreps`
4. Working with rotations
5. Computing spherical harmonics
6. Tensor products
7. Equivariant linear layers
8. Equivariant nonlinearities
9. Building an equivariant message-passing layer
10. Constructing a complete equivariant neural network
11. Best practices and debugging
12. Preparing for NequIP

Each section builds directly on the mathematical concepts introduced in the previous chapter.

---

## Transition to Section 20.13.2 — Installing e3nn

Before we can build equivariant neural networks, we first need to install the required software environment. In the next section, we will install **PyTorch**, **e3nn**, and the supporting scientific Python packages, verify that the installation works correctly, and prepare a development environment suitable for research in materials informatics.

# 20.13.2 Installing e3nn

Before building an equivariant neural network, we must first prepare a suitable software environment.

Unlike ordinary deep learning, equivariant neural networks require specialized mathematical operations such as

* spherical harmonics,
* tensor products,
* irreducible representations,
* rotation matrices.

Fortunately, the **e3nn** library provides efficient implementations of all these operations.

Since e3nn is built directly on top of **PyTorch**, the installation process is straightforward.

In this section, we will install the complete software stack required for developing equivariant graph neural networks for materials science.

---

# 20.13.3 Software Stack

The complete software environment consists of several layers.

```text id="stack1"
Your Code

↓

e3nn

↓

PyTorch

↓

CUDA / CPU

↓

Operating System
```

Each layer depends on the one below it.

---

# 20.13.4 Required Packages

For most research projects, the following Python packages are sufficient.

| Package                      | Purpose                            |
| ---------------------------- | ---------------------------------- |
| PyTorch                      | Deep learning framework            |
| e3nn                         | Equivariant neural network library |
| NumPy                        | Numerical computing                |
| SciPy                        | Scientific algorithms              |
| Matplotlib                   | Visualization                      |
| ASE                          | Atomic structures                  |
| pymatgen                     | Materials data processing          |
| PyTorch Geometric (optional) | Graph neural networks              |

---

# 20.13.5 Python Version

e3nn supports modern versions of Python.

The recommended versions are

* Python 3.10
* Python 3.11

Older versions may not support the latest releases of PyTorch or e3nn.

---

# 20.13.6 Creating a Virtual Environment

It is strongly recommended to use a virtual environment.

Using **conda**

```bash id="e3nn1"
conda create -n e3nn_env python=3.11
```

Activate it

```bash id="e3nn2"
conda activate e3nn_env
```

Alternatively,

using Python's built-in virtual environment

```bash id="e3nn3"
python -m venv e3nn_env
```

Activate

**Windows**

```bash id="e3nn4"
e3nn_env\Scripts\activate
```

**Linux / macOS**

```bash id="e3nn5"
source e3nn_env/bin/activate
```

Using isolated environments prevents dependency conflicts between projects.

---

# 20.13.7 Installing PyTorch

The first dependency is PyTorch.

CPU version

```bash id="e3nn6"
pip install torch torchvision torchaudio
```

If you have an NVIDIA GPU,

install the CUDA version recommended on the official PyTorch website.

For example,

```bash id="e3nn7"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

The exact CUDA version depends on your graphics driver.

---

# 20.13.8 Verifying PyTorch

Open Python.

```python id="e3nn8"
import torch

print(torch.__version__)
```

Example output

```text id="verify1"
2.6.0
```

Next,

verify GPU support.

```python id="e3nn9"
import torch

print(torch.cuda.is_available())
```

Typical output

```text id="verify2"
True
```

If the output is

```text id="verify3"
False
```

PyTorch will simply use the CPU.

The code will still work,

although training will be slower.

---

# 20.13.9 Installing e3nn

Installing e3nn is simple.

```bash id="e3nn10"
pip install e3nn
```

This automatically installs all required dependencies.

---

# 20.13.10 Verifying e3nn

Open Python.

```python id="e3nn11"
import e3nn

print(e3nn.__version__)
```

If no errors occur,

the installation is successful.

---

# 20.13.11 Importing the Main Module

Most programs begin with

```python id="e3nn12"
from e3nn import o3
```

The module

```python id="e3nn13"
o3
```

contains nearly all rotation-related functionality.

It includes

* rotations,
* irreducible representations,
* spherical harmonics,
* tensor products,
* Wigner matrices.

---

# 20.13.12 Installing Scientific Libraries

Materials science projects usually require several additional libraries.

```bash id="e3nn14"
pip install numpy scipy matplotlib
```

These libraries are used for

* numerical calculations,
* plotting,
* linear algebra,
* optimization.

---

# 20.13.13 Installing ASE

The Atomic Simulation Environment (ASE) is widely used for handling atomic structures.

```bash id="e3nn15"
pip install ase
```

ASE supports

* reading crystal structures,
* writing crystal structures,
* geometry optimization,
* interfacing with DFT codes.

---

# 20.13.14 Installing pymatgen

Materials informatics almost always requires pymatgen.

```bash id="e3nn16"
pip install pymatgen
```

This library allows us to

* read CIF files,
* analyze crystal symmetry,
* compute neighbor lists,
* manipulate crystal structures.

---

# 20.13.15 Optional: Installing PyTorch Geometric

Some equivariant architectures use PyTorch Geometric.

Install it with

```bash id="e3nn17"
pip install torch_geometric
```

Depending on the PyTorch version,

additional packages may also be required.

Consult the official PyTorch Geometric documentation if installation errors occur.

---

# 20.13.16 Recommended Development Environment

Although any Python IDE can be used,

the following combination is common in research.

```text id="ide1"
VS Code

+

Python Extension

+

Jupyter

+

Git
```

Other popular choices include

* PyCharm
* JupyterLab
* Google Colab
* Visual Studio

---

# 20.13.17 Checking the Complete Installation

Run the following script.

```python id="e3nn18"
import torch
import numpy as np
from e3nn import o3

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

x = torch.randn(5, 3)

Y = o3.spherical_harmonics(
    l=2,
    x=x,
    normalize=True,
    normalization="component"
)

print("Output shape:", Y.shape)
```

Expected output

```text id="verify4"
PyTorch: 2.x.x

CUDA: True

Output shape: torch.Size([5, 5])
```

This confirms that

* PyTorch works,
* e3nn works,
* spherical harmonics can be computed successfully.

---

# 20.13.18 Common Installation Problems

Several issues commonly arise.

### Problem 1

```text id="error1"
ModuleNotFoundError:
No module named 'e3nn'
```

Solution

```bash id="e3nn19"
pip install e3nn
```

inside the active virtual environment.

---

### Problem 2

CUDA unavailable

```text id="error2"
torch.cuda.is_available()

↓

False
```

Possible causes

* incorrect PyTorch installation,
* outdated GPU driver,
* unsupported CUDA version.

The code will still function using the CPU.

---

### Problem 3

Version mismatch

Sometimes,

PyTorch and e3nn versions are incompatible.

The safest solution is

```bash id="e3nn20"
pip install --upgrade torch e3nn
```

---

# 20.13.19 Best Practices

For research projects,

it is good practice to

* use virtual environments,
* record package versions,
* keep PyTorch and e3nn updated,
* test the installation before beginning model development.

Many researchers also maintain a

```text id="best1"
requirements.txt
```

or

```text id="best2"
environment.yml
```

file so that experiments can be reproduced easily.

---

# 20.13.20 Software Stack for This Book

Throughout the remainder of this book,

we will primarily use

| Package    | Purpose                |
| ---------- | ---------------------- |
| PyTorch    | Deep learning          |
| e3nn       | Equivariant operations |
| ASE        | Atomic structures      |
| pymatgen   | Materials processing   |
| NumPy      | Numerical computing    |
| Matplotlib | Visualization          |

Additional libraries such as

* PyTorch Geometric,
* Lightning,
* wandb,

will be introduced only when needed.

---

# 20.13.21 Key Takeaways

Setting up the software environment for equivariant graph neural networks is relatively straightforward because **e3nn** is built directly on top of **PyTorch**. After installing PyTorch, e3nn, and the supporting scientific Python libraries, researchers gain access to efficient implementations of rotations, irreducible representations, spherical harmonics, tensor products, and other symmetry-aware operations.

Establishing a clean, reproducible development environment is an essential first step before implementing more advanced equivariant architectures.

---

## Transition to Section 20.13.3 — Understanding the `Irreps` Class

With the software successfully installed, we are ready to explore the most fundamental abstraction in **e3nn**: the **Irreps** class. Almost every object in e3nn is described in terms of irreducible representations, making `Irreps` the language through which equivariant neural networks express symmetry. In the next section, we will learn how irreducible representations are encoded, interpret notation such as `0e`, `1o`, and `2e`, construct custom representation strings, and understand why these objects are central to every equivariant layer in e3nn.

# 20.13.3 Understanding the `Irreps` Class

If there is one concept that every user of **e3nn** must understand, it is the **Irreps** class.

Almost every object in e3nn is described using irreducible representations.

Whether we are defining

* input features,
* hidden layers,
* tensor products,
* output layers,
* nonlinear activations,

we must specify their symmetry properties using **Irreps**.

For this reason, the `Irreps` class is often described as **the language of e3nn**.

Understanding it thoroughly is essential before building any equivariant neural network.

---

# 20.13.4 What Does "Irreps" Mean?

The word **Irreps** is simply shorthand for

> **Irreducible Representations**

Recall from the previous chapter that every irreducible representation of the rotation group SO(3) is labeled by

[
l = 0,1,2,\ldots
]

Each value of (l) corresponds to a different type of geometric object.

| (l) | Physical Meaning     |
| --- | -------------------- |
| 0   | Scalar               |
| 1   | Vector               |
| 2   | Rank-2 tensor        |
| 3   | Rank-3 tensor        |
| ... | Higher-order tensors |

The `Irreps` class stores this information in a compact, machine-readable format.

---

# 20.13.5 Why Does e3nn Need Irreps?

In ordinary neural networks,

a layer only needs to know

```text id="irrep1"
Input Dimension

↓

Output Dimension
```

For example,

```python id="irrep2"
Linear(128,256)
```

means

128 input features

↓

256 output features.

No information about symmetry is required.

---

# 20.13.6 Equivariant Layers Need More Information

Equivariant layers must know not only

* **how many features**,

but also

* **how each feature transforms under rotation**.

For example,

consider two features.

```text id="irrep3"
Temperature
```

and

```text id="irrep4"
Force
```

Temperature is a scalar.

Force is a vector.

Although both are numerical quantities,

they behave differently under rotation.

Therefore,

they cannot be treated identically inside an equivariant network.

---

# 20.13.7 Irreps Describe Transformation Rules

The purpose of `Irreps` is to describe exactly how every feature transforms.

Conceptually,

```text id="irrep5"
Feature

↓

Transformation Rule

↓

Irrep
```

Instead of merely storing feature dimensions,

e3nn stores the corresponding symmetry behavior.

---

# 20.13.8 Creating an Irreps Object

The simplest example is

```python id="irrep6"
from e3nn import o3

irreps = o3.Irreps("1x0e")
```

This creates one scalar feature.

---

# 20.13.9 Breaking Down the Notation

The string

```text id="irrep7"
1x0e
```

contains three parts.

```text id="irrep8"
1

×

0

e
```

Each part has a precise meaning.

---

# 20.13.10 Multiplicity

The first number is the multiplicity.

```text id="irrep9"
1
```

means

one copy

of the representation.

If we write

```text id="irrep10"
5x0e
```

we have

five independent scalar features.

---

# 20.13.11 Angular Degree

The second number is

[
l.
]

For example,

```text id="irrep11"
0
```

means

scalar.

```text id="irrep12"
1
```

means

vector.

```text id="irrep13"
2
```

means

rank-2 tensor.

---

# 20.13.12 Parity

The final letter indicates parity.

There are two possibilities.

```text id="irrep14"
e
```

means

**even parity**.

```text id="irrep15"
o
```

means

**odd parity**.

Parity determines how a quantity behaves under inversion.

---

# 20.13.13 Even Parity

Suppose we invert coordinates.

[
(x,y,z)

\rightarrow

(-x,-y,-z).
]

An even-parity quantity remains unchanged.

Examples include

* energy,
* mass,
* temperature,
* charge density.

---

# 20.13.14 Odd Parity

An odd-parity quantity changes sign.

Examples include

* position,
* velocity,
* force,
* electric field.

These vectors reverse direction after inversion.

---

# 20.13.15 Examples

The following notation appears frequently.

```text id="irrep16"
1x0e
```

One scalar.

---

```text id="irrep17"
1x1o
```

One vector.

---

```text id="irrep18"
1x2e
```

One rank-2 tensor.

---

# 20.13.16 Multiple Representations

Different irreducible representations can be combined.

For example,

```python id="irrep19"
o3.Irreps("1x0e + 1x1o")
```

This means

```text id="irrep20"
Scalar

+

Vector
```

The resulting feature set contains both types simultaneously.

---

# 20.13.17 Larger Example

Consider

```python id="irrep21"
o3.Irreps("4x0e + 8x1o + 4x2e")
```

This means

* four scalar features,
* eight vector features,
* four rank-2 tensor features.

These become the hidden features of a neural network layer.

---

# 20.13.18 Feature Dimensions

Recall that an irrep of degree

[
l
]

contains

[
2l+1
]

components.

Therefore,

```text id="irrep22"
0e

↓

1 component
```

---

```text id="irrep23"
1o

↓

3 components
```

---

```text id="irrep24"
2e

↓

5 components
```

---

# 20.13.19 Computing Total Dimension

Suppose we have

```python id="irrep25"
o3.Irreps("2x0e + 3x1o")
```

Dimension calculation:

Scalars

[
2\times1=2
]

Vectors

[
3\times3=9
]

Total dimension

[
2+9=11.
]

---

# 20.13.20 e3nn Can Compute This Automatically

Fortunately,

manual calculation is unnecessary.

```python id="irrep26"
irreps = o3.Irreps("2x0e + 3x1o")

print(irreps.dim)
```

Output

```text id="irrep27"
11
```

The library automatically computes the total feature dimension.

---

# 20.13.21 Printing an Irreps Object

```python id="irrep28"
irreps = o3.Irreps("2x0e + 1x1o")

print(irreps)
```

Output

```text id="irrep29"
2x0e+1x1o
```

This representation appears throughout e3nn programs.

---

# 20.13.22 Inspecting Individual Irreps

Each component can also be accessed separately.

```python id="irrep30"
for mul, ir in irreps:
    print(mul, ir)
```

Example output

```text id="irrep31"
2 0e

1 1o
```

This is useful when constructing custom layers.

---

# 20.13.23 Why Neural Networks Need Mixed Irreps

Real materials contain

* scalar quantities,
* vector quantities,
* tensor quantities.

A hidden layer therefore rarely contains only one representation.

Instead,

it contains a mixture.

Conceptually,

```text id="irrep32"
Scalars

Vectors

Rank-2 Tensors

↓

Hidden Layer
```

This rich feature space enables the network to model complex physical interactions.

---

# 20.13.24 Example from NequIP

A typical hidden representation in NequIP might look like

```python id="irrep33"
"64x0e + 64x1o + 32x2e"
```

This means

* 64 scalar channels,
* 64 vector channels,
* 32 tensor channels.

Such mixed representations allow the network to simultaneously learn

* chemical information,
* directional information,
* higher-order geometric correlations.

---

# 20.13.25 Why Irreps Are Better Than Raw Tensors

Without irreducible representations,

the network would need to manually keep track of

* tensor ranks,
* rotation rules,
* parity,
* dimensionality.

The `Irreps` class encapsulates all of this information in a single object.

This greatly simplifies both implementation and debugging.

---

# 20.13.26 Key Takeaways

The **Irreps** class is the fundamental data structure of **e3nn**. Rather than storing only feature dimensions, it specifies **how every feature transforms under rotations and inversions**. Each representation is described by its multiplicity, angular degree (l), and parity, allowing equivariant layers to manipulate scalar, vector, and tensor features while rigorously preserving symmetry.

Understanding `Irreps` is essential because every equivariant layer, tensor product, activation function, and message-passing operation in e3nn is defined in terms of these representations.

---

## Transition to Section 20.13.4 — Understanding `0e`, `1o`, `2e`, and Higher-Order Irreps

Although we can now read an `Irreps` string, it is important to build deeper intuition for what each individual representation actually means. In the next section, we will examine the physical interpretation of **`0e`**, **`1o`**, **`2e`**, and higher-order irreducible representations, connect them to familiar quantities in physics and materials science, and understand why these representations naturally arise in equivariant neural networks.

# 20.13.4 Understanding `0e`, `1o`, `2e`, and Higher-Order Irreducible Representations

Although an **Irreps** string appears simple,

```text
0e
1o
2e
```

each symbol represents a well-defined mathematical object with a precise physical interpretation.

These representations are not arbitrary labels invented by e3nn.

They originate directly from the representation theory of the rotation group **SO(3)** and describe how different physical quantities behave under rotations and inversions.

Understanding these representations is essential because they form the building blocks of every equivariant neural network.

---

# 20.13.4.1 The Three Parts of an Irrep

Every irreducible representation has the general form

```text
l parity
```

or inside an `Irreps` string,

```text
Multiplicity × l Parity
```

For example,

```text
64x1o
```

means

* 64 copies,
* angular degree (l=1),
* odd parity.

The angular degree determines **how the object transforms under rotation**, while the parity determines **how it transforms under spatial inversion**.

---

# 20.13.4.2 The Simplest Representation: `0e`

The representation

```text
0e
```

is the simplest possible irrep.

It corresponds to

* scalar quantities,
* rotation-invariant values,
* inversion-even quantities.

Its dimension is

[
2(0)+1=1.
]

Therefore,

a `0e` representation contains only one number.

---

# 20.13.4.3 Physical Examples of `0e`

Many familiar material properties are scalars.

Examples include

* total energy,
* formation energy,
* temperature,
* atomic mass,
* pressure,
* band gap,
* density.

Rotating the crystal does not change any of these quantities.

For example,

```text
Crystal

↓

Rotate 90°

↓

Energy unchanged
```

Therefore,

they belong to the `0e` representation.

---

# 20.13.4.4 Why "e"?

The letter

```text
e
```

stands for

**even parity**.

Under inversion,

[
(x,y,z)

\rightarrow

(-x,-y,-z),
]

a scalar remains unchanged.

Mathematically,

[
f(-\mathbf{x})=f(\mathbf{x}).
]

Hence,

energy,

temperature,

and density are all even-parity quantities.

---

# 20.13.4.5 The `1o` Representation

The next representation is

```text
1o
```

This corresponds to ordinary vectors.

Its dimension is

[
2(1)+1=3.
]

Therefore,

a `1o` feature consists of three numbers.

Conceptually,

```text
x

y

z
```

These three components transform together as a vector.

---

# 20.13.4.6 Physical Examples of `1o`

Many quantities in materials science are vectors.

Examples include

* atomic position,
* velocity,
* force,
* momentum,
* electric field,
* displacement.

Each has three Cartesian components.

---

# 20.13.4.7 Rotation of a `1o` Feature

Suppose a force vector points along

```text
+x
```

After rotating the crystal,

the force rotates with it.

The vector changes orientation,

but its physical meaning remains identical.

This behavior is exactly what the `1o` representation describes.

---

# 20.13.4.8 Why "o"?

The letter

```text
o
```

means

**odd parity**.

Under inversion,

a vector changes sign.

For example,

[
\mathbf{F}

\rightarrow

-\mathbf{F}.
]

Thus,

forces,

positions,

and velocities are odd-parity quantities.

---

# 20.13.4.9 The `2e` Representation

The next commonly encountered representation is

```text
2e
```

Its dimension is

[
2(2)+1=5.
]

Therefore,

it contains

five independent components.

Unlike vectors,

these components cannot be interpreted as

```text
x

y

z
```

Instead,

they represent the independent degrees of freedom of a rank-2 tensor expressed in an irreducible basis.

---

# 20.13.4.10 Physical Examples of `2e`

Rank-2 tensors appear frequently in materials science.

Examples include

* quadrupole moments,
* anisotropic electron density,
* deviatoric stress,
* certain crystal-field quantities,
* second-order angular information.

These quantities carry directional information that is richer than a simple vector.

---

# 20.13.4.11 Why Five Components?

A full (3\times3) matrix contains

nine elements.

However,

many physically important tensors are

* symmetric,
* traceless.

After removing redundant information,

only five independent degrees of freedom remain.

These five components correspond exactly to the `2e` representation.

---

# 20.13.4.12 Higher-Order Representations

The pattern continues naturally.

| Irrep | Dimension |
| ----- | --------: |
| `0e`  |         1 |
| `1o`  |         3 |
| `2e`  |         5 |
| `3o`  |         7 |
| `4e`  |         9 |

The general formula is

[
2l+1.
]

Each increase in (l) introduces more complex angular behavior.

---

# 20.13.4.13 Physical Meaning of Higher Orders

Higher-order irreducible representations describe increasingly complex directional structures.

For example,

```text
l = 0

↓

No Direction
```

```text
l = 1

↓

One Direction
```

```text
l = 2

↓

Angular Shape
```

```text
l = 3

↓

More Complex Geometry
```

As (l) increases,

the representation captures finer details of the local atomic environment.

---

# 20.13.4.14 Relationship with Spherical Harmonics

Recall that spherical harmonics satisfy

[
Y_l^m.
]

For every value of

[
l,
]

there are

[
2l+1
]

basis functions.

Therefore,

every spherical harmonic degree corresponds directly to one irreducible representation.

| Spherical Harmonic | Irrep |
| ------------------ | ----- |
| (Y_0^0)            | `0e`  |
| (Y_1^m)            | `1o`  |
| (Y_2^m)            | `2e`  |
| (Y_3^m)            | `3o`  |

This connection explains why spherical harmonics naturally produce irreducible tensor features.

---

# 20.13.4.15 Why Neural Networks Mix Different Irreps

A realistic material contains many types of information.

For example,

```text
Chemical Identity

↓

Scalar
```

```text
Bond Direction

↓

Vector
```

```text
Local Geometry

↓

Higher-Order Tensor
```

An equivariant neural network therefore combines several irreducible representations in the same hidden layer.

---

# 20.13.4.16 Example Hidden Representation

A hidden layer may use

```python
"32x0e + 32x1o + 16x2e"
```

This means the network simultaneously stores

* scalar information,
* vector information,
* tensor information.

Each type evolves according to its own transformation rules.

---

# 20.13.4.17 Why Not Convert Everything to Scalars?

One might ask why we do not simply convert every feature into a scalar.

Doing so would discard valuable directional information.

For example,

consider two bonds

```text
Bond A

↓

+x
```

and

```text
Bond B

↓

+y
```

If both are converted into the same scalar,

their orientations become indistinguishable.

The network would lose information about crystal geometry.

Vector and tensor representations preserve this information.

---

# 20.13.4.18 Summary Table

The most common irreducible representations encountered in materials science are summarized below.

| Irrep | Dimension | Physical Interpretation | Example                       |
| ----- | --------: | ----------------------- | ----------------------------- |
| `0e`  |         1 | Scalar                  | Energy, band gap              |
| `1o`  |         3 | Vector                  | Force, position               |
| `2e`  |         5 | Rank-2 tensor           | Quadrupole, anisotropy        |
| `3o`  |         7 | Rank-3 tensor           | Higher-order angular features |
| `4e`  |         9 | Rank-4 tensor           | Complex many-body geometry    |

---

# 20.13.4.19 Key Takeaways

The symbols **`0e`**, **`1o`**, **`2e`**, and higher-order irreducible representations are compact descriptions of how different physical quantities transform under rotations and inversions. Scalars belong to `0e`, vectors to `1o`, and increasingly complex tensorial quantities correspond to higher values of (l). The dimension of each representation is always (2l+1), matching the number of spherical harmonics of the same degree.

These irreducible representations provide the symmetry-aware feature types used throughout **e3nn**, allowing equivariant neural networks to manipulate scalar, vector, and tensor information without violating the geometric symmetries of three-dimensional space.

---

## Transition to Section 20.13.5 — Constructing and Manipulating `Irreps` Objects in e3nn

Now that we understand the physical meaning of the most common irreducible representations, we are ready to use them programmatically. In the next section, we will learn how to create, inspect, combine, simplify, and manipulate `Irreps` objects in **e3nn**, preparing us to define the input, hidden, and output representations of complete equivariant neural networks.

