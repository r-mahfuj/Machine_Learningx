# Materials Informatics & Machine Learning: The XGBoost Masterclass

# Chapter 0 — Python and Machine Learning Foundations for Materials Scientists

> *"Before building intelligent machine learning models, we must first learn the language they speak. That language is Python."*

---

# 0.1 Why Learn Python?

Machine learning is ultimately mathematics implemented through programming.

A machine learning model cannot be trained using equations alone. We must tell the computer how to represent data, perform calculations, and make predictions. Python has become the standard language for scientific computing because it is simple to learn, highly readable, and supported by an enormous ecosystem of scientific libraries.

In Materials Informatics, almost every workflow follows the same general pattern.

```text
Materials Data

↓

Python

↓

Machine Learning Library

↓

Prediction

↓

Scientific Interpretation
```

Throughout this book you will gradually learn everything required to build this workflow yourself.

**Important:** This book assumes **no prior experience with Python programming.** Every concept required for machine learning will be introduced before it is used.

---

# 0.2 Installing Python

The easiest way to begin scientific programming is to install **Anaconda**, which includes Python together with many scientific libraries.

After installation, open either

- Jupyter Notebook
- JupyterLab
- VS Code
- Spyder

Any of these environments is suitable for following this book.

---

# 0.3 Your First Python Program

Every programming language begins with creating variables.

A variable stores information inside the computer's memory.

Consider the following program.

```python
material = "Fe2O3"
density = 5.24
bandgap = 2.10
```

## Line-by-Line Explanation

### Line 1

```python
material = "Fe2O3"
```

This creates a variable named `material`.

The value stored is the chemical formula **Fe₂O₃**, which represents iron(III) oxide.

Variables act as containers that store information for later use.

After executing this line, Python remembers that whenever we write `material`, we mean the text `"Fe2O3"`.

---

### Line 2

```python
density = 5.24
```

This creates another variable named `density`.

The value `5.24` represents the density of the material.

Unlike the previous variable, this value is numerical rather than textual.

Numerical variables allow Python to perform mathematical calculations.

---

### Line 3

```python
bandgap = 2.10
```

This creates a variable named `bandgap`.

In Materials Informatics, this may represent the target property we wish to predict.

At this stage we are only storing information.

No machine learning has occurred yet.

---

# 0.4 Variables

Variables are fundamental to every Python program.

Think of a variable as a labeled storage box.

```text
Variable Name

↓

Stored Value

↓

Used Later
```

Examples:

```python
temperature = 850
pressure = 1.0
crystal_structure = "FCC"
hardness = 310
```

Each variable stores one piece of information.

Machine learning models eventually combine thousands of such variables.

---

# 0.5 Python Data Types

Python automatically recognizes different kinds of data.

The most important types for machine learning are shown below.

| Data Type | Example | Purpose |
|-----------|----------|---------|
| Integer | 15 | Whole numbers |
| Float | 3.75 | Decimal numbers |
| String | "Steel" | Text |
| Boolean | True | Logical values |

Examples:

```python
atoms = 12
density = 7.85
material = "Steel"
stable = True
```

Understanding data types is important because different operations are allowed on different types of data.

---

# 0.6 Printing Information

Programs become useful when they can display information.

Python uses the `print()` function.

```python
material = "Al2O3"
print(material)
```

## Line-by-Line Explanation

### Line 1

```python
material = "Al2O3"
```

Stores the material name.

---

### Line 2

```python
print(material)
```

The `print()` function tells Python to display the contents of the variable.

The output becomes

```text
Al2O3
```

Notice that `print()` displays the value stored inside the variable rather than the variable name itself.

---

# 0.7 Numbers and Arithmetic

Python behaves like a scientific calculator.

```python
density = 7.85
volume = 15
mass = density * volume
print(mass)
```

## Line-by-Line Explanation

### Line 1

```python
density = 7.85
```

Stores the density.

---

### Line 2

```python
volume = 15
```

Stores the volume.

---

### Line 3

```python
mass = density * volume
```

Python multiplies the two variables.

The result is stored in a new variable named `mass`.

Notice that Python first performs the calculation and only then stores the answer.

---

### Line 4

```python
print(mass)
```

Displays the calculated mass.

This demonstrates that variables can participate in mathematical expressions.

---

# 0.8 Why Programming Matters in Machine Learning

Machine learning algorithms do not directly understand materials.

Instead, they manipulate numbers.

Suppose we have experimental measurements.

| Density | Atomic Radius | Bandgap |
|---------:|--------------:|---------:|
| 5.1 | 1.20 | 2.3 |
| 4.8 | 1.42 | 1.8 |
| 6.0 | 1.05 | 3.1 |

Python stores these numbers.

Machine learning algorithms discover relationships among them.

Without programming, machine learning cannot exist.

---

# 0.9 Your First Scientific Goal

By the end of this chapter you should understand

- what variables are,
- how Python stores information,
- how to perform calculations,
- how to display results,
- why programming is essential for machine learning.

These concepts may appear simple, but they are the building blocks for every machine learning model you will create later in this book.

---

# Coming Next

In the next section, we will learn **Python collections**:

- Lists
- Tuples
- Dictionaries
- Sets

These data structures are essential because machine learning datasets contain thousands or even millions of values rather than a single number.

# 0.10 Python Lists — Storing Multiple Values

So far, every variable has stored only one value.

For example,

```python
density = 7.85
```

stores only a single number.

However, Materials Informatics rarely works with one material.

A typical dataset may contain

- hundreds,
- thousands,
- or even millions

of materials.

Instead of creating thousands of variables, Python provides **lists**.

A list stores multiple values inside a single variable.

---

## Creating Your First List

```python
densities = [2.70, 7.85, 8.96, 19.30]
```

### Line-by-Line Explanation

### Line 1

```python
densities = [2.70, 7.85, 8.96, 19.30]
```

A variable named `densities` is created.

Instead of storing one number, it stores four numbers.

Each number could represent the density of a different material.

The square brackets `[]` tell Python that this is a list.

After execution, the computer stores

| Position | Value |
|---------:|------:|
| 0 | 2.70 |
| 1 | 7.85 |
| 2 | 8.96 |
| 3 | 19.30 |

---

# 0.11 Accessing Individual Elements

Each value inside a list has a position called an **index**.

Python starts counting from **0**, not 1.

```python
densities = [2.70, 7.85, 8.96, 19.30]

print(densities[0])

print(densities[2])
```

## Line-by-Line Explanation

### Line 1

```python
densities = [2.70, 7.85, 8.96, 19.30]
```

Creates the list.

---

### Line 3

```python
print(densities[0])
```

`densities[0]` means

> "Retrieve the first value in the list."

Python prints

```text
2.7
```

---

### Line 5

```python
print(densities[2])
```

Python retrieves the third value.

Remember,

index 2 means

```text
First → 0

Second → 1

Third → 2
```

The output becomes

```text
8.96
```

---

# 0.12 Why Does Python Start Counting from Zero?

Many beginners find this confusing.

Suppose we have

```python
materials = ["Steel", "Copper", "Titanium"]
```

Python stores them internally as

| Index | Material |
|------:|----------|
| 0 | Steel |
| 1 | Copper |
| 2 | Titanium |

Although unusual at first, zero-based indexing simplifies many computational operations and is used throughout Python's scientific ecosystem.

---

# 0.13 Finding the Length of a List

Often we need to know how many materials are stored.

Python provides the `len()` function.

```python
materials = ["Steel", "Copper", "Titanium"]

count = len(materials)

print(count)
```

## Line-by-Line Explanation

### Line 1

```python
materials = ["Steel", "Copper", "Titanium"]
```

Creates a list containing three material names.

---

### Line 3

```python
count = len(materials)
```

The `len()` function counts how many elements are stored inside the list.

The answer is assigned to the variable `count`.

---

### Line 5

```python
print(count)
```

Displays

```text
3
```

because there are three elements.

---

# 0.14 Adding New Elements

Experimental datasets continually grow.

Python allows us to add new values using the `append()` method.

```python
materials = ["Steel", "Copper", "Titanium"]

materials.append("Aluminum")

print(materials)
```

## Line-by-Line Explanation

### Line 1

```python
materials = ["Steel", "Copper", "Titanium"]
```

Creates the initial list.

---

### Line 3

```python
materials.append("Aluminum")
```

The `append()` method adds one new element to the end of the list.

After execution, the list contains four materials.

---

### Line 5

```python
print(materials)
```

Displays

```text
['Steel', 'Copper', 'Titanium', 'Aluminum']
```

Notice that the original list has changed.

---

# 0.15 Removing Elements

Sometimes experimental data must be removed.

Python provides the `remove()` method.

```python
materials = ["Steel", "Copper", "Titanium"]

materials.remove("Copper")

print(materials)
```

## Line-by-Line Explanation

### Line 1

```python
materials = ["Steel", "Copper", "Titanium"]
```

Creates the list.

---

### Line 3

```python
materials.remove("Copper")
```

Python searches for the value `"Copper"`.

When it finds the value, it removes it from the list.

---

### Line 5

```python
print(materials)
```

The output becomes

```text
['Steel', 'Titanium']
```

---

# 0.16 Lists Can Store Different Data Types

A list does not have to contain only numbers.

For example,

```python
sample = ["Fe2O3", 5.24, 2.10]
```

This list contains

| Value | Meaning |
|-------|---------|
| "Fe2O3" | Material name |
| 5.24 | Density |
| 2.10 | Bandgap |

Although Python allows mixed data types, machine learning algorithms usually require numerical data.

Later, we will learn how Pandas organizes this information more effectively.

---

# 0.17 Nested Lists

Lists may contain other lists.

```python
dataset = [
    [5.24, 2.10],
    [4.95, 1.85],
    [6.10, 3.05]
]
```

This resembles a table.

Each inner list represents one material.

| Density | Bandgap |
|---------:|---------:|
| 5.24 | 2.10 |
| 4.95 | 1.85 |
| 6.10 | 3.05 |

This idea is the foundation of machine learning datasets.

---

# 0.18 Limitations of Lists

Lists are flexible and easy to use.

However, they are not optimized for scientific numerical computation.

Suppose we have one million density values.

Performing mathematical operations using ordinary Python lists becomes slow.

For high-performance numerical calculations, scientists use **NumPy arrays**, which are specifically designed for mathematical computing.

NumPy arrays form the backbone of almost every machine learning library, including Scikit-Learn, XGBoost, TensorFlow, and PyTorch.

---


In the next section, we will learn **tuples, dictionaries, and sets**, followed by **NumPy**, where Python begins to resemble a true scientific computing environment suitable for Materials Informatics.
# 0.19 Python Tuples — Data That Should Not Change

Lists are extremely useful because they allow us to modify their contents.

However, some information should remain constant throughout a program.

For example,

- the lattice type of a material,
- RGB color definitions for plots,
- fixed crystal directions,
- Cartesian coordinates that should not be modified accidentally.

Python provides **tuples** for such situations.

A tuple is an ordered collection of values that cannot be modified after it has been created.

This property is called **immutability**.

---

## Creating Your First Tuple

```python
crystal_systems = ("FCC", "BCC", "HCP")
```

### Line-by-Line Explanation

### Line 1

```python
crystal_systems = ("FCC", "BCC", "HCP")
```

A variable named `crystal_systems` is created.

Unlike lists, tuples use **parentheses `()`** instead of square brackets.

The tuple stores three crystal structures.

Once created, these values cannot be changed accidentally.

---

# 0.20 Accessing Tuple Elements

Accessing values inside a tuple is identical to accessing values inside a list.

```python
crystal_systems = ("FCC", "BCC", "HCP")

print(crystal_systems[1])
```

## Line-by-Line Explanation

### Line 1

```python
crystal_systems = ("FCC", "BCC", "HCP")
```

Creates the tuple.

---

### Line 3

```python
print(crystal_systems[1])
```

Python retrieves the element at index **1**.

Remember that indexing starts from zero.

The output becomes

```text
BCC
```

---

# 0.21 When Should You Use a Tuple?

Use a tuple when the information should remain fixed.

Examples include

- crystal systems,
- Cartesian coordinates,
- RGB color definitions,
- fixed experimental settings.

If you expect to add or remove values later, use a list instead.

---

# 0.22 Python Dictionaries — Storing Information with Labels

Lists store values using positions.

Sometimes positions are inconvenient.

Suppose we have

```text
Density = 7.85

Bandgap = 2.1

Melting Point = 1538
```

Instead of remembering index numbers, it is often easier to use descriptive names.

Python provides **dictionaries** for this purpose.

A dictionary stores information as

```text
Key

↓

Value
```

---

## Creating Your First Dictionary

```python
steel = {
    "density": 7.85,
    "hardness": 250,
    "melting_point": 1538
}
```

## Line-by-Line Explanation

### Line 1

```python
steel = {
```

Creates a dictionary named `steel`.

The opening curly brace `{` tells Python that a dictionary is being created.

---

### Line 2

```python
"density": 7.85,
```

The key is `"density"`.

The corresponding value is `7.85`.

Whenever we ask for `"density"`, Python returns `7.85`.

---

### Line 3

```python
"hardness": 250,
```

Stores another property.

The key is `"hardness"`.

The value is `250`.

---

### Line 4

```python
"melting_point": 1538
```

Stores the melting point.

---

### Line 5

```python
}
```

Closes the dictionary.

---

# 0.23 Accessing Dictionary Values

Dictionary values are retrieved using their keys.

```python
steel = {
    "density": 7.85,
    "hardness": 250
}

print(steel["density"])
```

## Line-by-Line Explanation

### Line 1–4

Creates the dictionary.

---

### Line 6

```python
print(steel["density"])
```

Python searches for the key `"density"`.

It finds the value `7.85`.

The output becomes

```text
7.85
```

Unlike lists, dictionaries do not use index positions.

They use meaningful names.

---

# 0.24 Adding New Entries to a Dictionary

Dictionaries can grow as new information becomes available.

```python
steel = {
    "density": 7.85
}

steel["bandgap"] = 0

print(steel)
```

## Line-by-Line Explanation

### Line 1–3

Creates a dictionary containing one property.

---

### Line 5

```python
steel["bandgap"] = 0
```

Adds a new key called `"bandgap"`.

The associated value is `0`.

Python automatically expands the dictionary.

---

### Line 7

```python
print(steel)
```

Displays

```text
{'density': 7.85, 'bandgap': 0}
```

---

# 0.25 Why Dictionaries Are Useful in Materials Informatics

Imagine downloading information for one material from a database.

Instead of receiving

```text
7.85

250

1538
```

you receive

```text
Density → 7.85

Hardness → 250

Melting Point → 1538
```

The property names remain attached to their values.

This makes programs much easier to understand.

Many scientific libraries return data as dictionaries.

---

# 0.26 Python Sets

A set stores **unique** values.

Duplicate entries are removed automatically.

```python
elements = {"Fe", "Fe", "Cr", "Ni", "Cr"}

print(elements)
```

## Line-by-Line Explanation

### Line 1

```python
elements = {"Fe", "Fe", "Cr", "Ni", "Cr"}
```

Creates a set.

Although `"Fe"` and `"Cr"` appear twice, a set stores only one copy of each unique value.

---

### Line 3

```python
print(elements)
```

Displays something similar to

```text
{'Fe', 'Cr', 'Ni'}
```

The order may vary because sets are unordered collections.

---

# 0.27 When Should You Use a Set?

Sets are useful when duplicate values should be removed.

Example applications include

- finding unique chemical elements,
- removing repeated material IDs,
- identifying unique crystal structures,
- determining distinct processing temperatures.

---

# 0.28 Choosing the Correct Data Structure

Python provides several ways to organize data.

Each has a different purpose.

| Data Structure | Ordered | Can Change? | Typical Use |
|---------------|---------|-------------|-------------|
| List | Yes | Yes | Collections that grow or shrink |
| Tuple | Yes | No | Fixed information |
| Dictionary | Yes | Yes | Named properties |
| Set | No | Yes | Unique values |

Choosing the correct data structure makes programs easier to write, understand, and maintain.

---

# 0.29 Which Data Structure Is Used Most in Machine Learning?

Although all four data structures are important, they are not equally common in machine learning.

A typical workflow looks like this:

```text
Lists

↓

NumPy Arrays

↓

Pandas DataFrames

↓

Machine Learning Models
```

Lists are often used while collecting data.

NumPy arrays perform efficient numerical computation.

Pandas DataFrames organize datasets into rows and columns.

Machine learning libraries then use these numerical datasets for training.

For this reason, **NumPy** is the next major topic you will learn.

It forms the computational foundation of almost every machine learning library used throughout this book.
# 0.30 NumPy — The Foundation of Scientific Computing

In the previous sections, we learned how to store information using Python variables, lists, tuples, dictionaries, and sets.

These data structures are excellent for general programming.

However, scientific computing has different requirements.

Materials Informatics datasets often contain

- tens of thousands of materials,
- hundreds of descriptors,
- millions of numerical values.

Ordinary Python lists become inefficient for this type of computation.

To solve this problem, scientists use **NumPy**.

NumPy stands for **Numerical Python**.

It is one of the most important scientific libraries ever developed for Python and serves as the computational foundation for many other libraries, including

- Pandas
- Scikit-Learn
- SciPy
- TensorFlow
- PyTorch
- XGBoost

If you become comfortable with NumPy, learning the rest of the scientific Python ecosystem becomes much easier.

---

# 0.31 Why Do We Need NumPy?

Suppose we want to calculate the density of one million materials.

Using ordinary Python lists would be relatively slow because Python performs calculations element by element.

NumPy performs the same operations using highly optimized code written in low-level languages.

As a result,

- calculations are much faster,
- memory usage is lower,
- mathematical operations become much easier.

For this reason, almost every machine learning model expects its numerical data to be stored as NumPy arrays.

---

# 0.32 Importing NumPy

Before using NumPy, we must import it.

```python
import numpy as np
```

## Line-by-Line Explanation

### Line 1

```python
import numpy as np
```

This statement imports the NumPy library into our Python program.

The keyword `import` tells Python to load an external library.

`numpy` is the library's official name.

`as np` creates a short alias.

Instead of writing

```python
numpy.array()
```

we can simply write

```python
np.array()
```

Almost every scientific Python program uses this convention.

Whenever you see `np`, you should immediately recognize that it refers to NumPy.

---

# 0.33 Creating Your First NumPy Array

A NumPy array stores numerical values.

```python
import numpy as np

densities = np.array([2.70, 7.85, 8.96, 19.30])
```

## Line-by-Line Explanation

### Line 1

```python
import numpy as np
```

Imports NumPy.

If this line is omitted, Python will not recognize `np`.

---

### Line 3

```python
densities = np.array([2.70, 7.85, 8.96, 19.30])
```

Let's examine this carefully.

`densities`

This is the variable that will store the NumPy array.

---

`np`

Tells Python to use the NumPy library.

---

`array()`

This is a NumPy function that creates an array.

Everything inside the parentheses becomes part of the array.

---

`[2.70, 7.85, 8.96, 19.30]`

These are the numerical values that will be stored.

Each value could represent the density of a different material.

After execution, the variable `densities` contains four numerical values stored in a NumPy array.

---

# 0.34 Printing a NumPy Array

Once an array has been created, we can display it.

```python
import numpy as np

densities = np.array([2.70, 7.85, 8.96, 19.30])

print(densities)
```

## Line-by-Line Explanation

### Line 1

Imports NumPy.

---

### Line 3

Creates the array.

---

### Line 5

```python
print(densities)
```

Displays every value stored inside the array.

The output is

```text
[ 2.7   7.85  8.96 19.3 ]
```

Notice that NumPy prints the array using its own formatting style.

---

# 0.35 Accessing Individual Elements

Just like Python lists, NumPy arrays use indexing.

```python
import numpy as np

densities = np.array([2.70, 7.85, 8.96, 19.30])

print(densities[1])
```

## Line-by-Line Explanation

### Line 1

Imports NumPy.

---

### Line 3

Creates the array.

---

### Line 5

```python
print(densities[1])
```

`densities[1]`

means

> Retrieve the element stored at index 1.

Remember,

Python starts counting from zero.

Therefore,

| Index | Value |
|------:|------:|
| 0 | 2.70 |
| 1 | 7.85 |
| 2 | 8.96 |
| 3 | 19.30 |

The output becomes

```text
7.85
```

---

# 0.36 NumPy Arrays versus Python Lists

Although they appear similar, they are fundamentally different.

Python List

```python
densities = [2.70, 7.85, 8.96]
```

NumPy Array

```python
densities = np.array([2.70, 7.85, 8.96])
```

The first creates a Python list.

The second creates a NumPy array.

A NumPy array

- stores only one primary data type,
- performs calculations much faster,
- is optimized for mathematics,
- is the preferred format for machine learning.

---

# 0.37 Performing Arithmetic with NumPy

One of NumPy's greatest strengths is that mathematical operations can be applied to an entire array at once.

Consider the following example.

```python
import numpy as np

densities = np.array([2.70, 7.85, 8.96])

new_densities = densities + 1

print(new_densities)
```

## Line-by-Line Explanation

### Line 1

Imports NumPy.

---

### Line 3

Creates the array.

---

### Line 5

```python
new_densities = densities + 1
```

This statement adds the number **1** to **every element** in the array.

Python performs

```text
2.70 + 1

7.85 + 1

8.96 + 1
```

automatically.

You do not need to write a loop.

This feature is called **vectorized computation**, and it is one of the main reasons NumPy is so powerful.

---

### Line 7

```python
print(new_densities)
```

Displays

```text
[3.70 8.85 9.96]
```

---

# 0.38 Why Vectorized Computation Matters

Suppose a dataset contains

500,000 materials.

Without NumPy, you would need to update each value individually.

With NumPy, one mathematical statement can update the entire dataset.

This makes scientific programs shorter, faster, and easier to read.

---


In the next section, we will explore **two-dimensional NumPy arrays**, where our data begins to resemble real machine learning datasets containing many materials and many features.

# 0.39 Two-Dimensional NumPy Arrays

So far, every NumPy array we have created has been **one-dimensional**.

For example,

```python
densities = np.array([2.70, 7.85, 8.96])
```

This array stores only one property.

However, machine learning rarely works with a single property.

Suppose we want to predict the hardness of several materials.

For each material, we may have

- Density
- Atomic Radius
- Electronegativity
- Melting Point

Instead of storing each property separately, we store everything in a **two-dimensional array**.

A two-dimensional array resembles a spreadsheet.

For example,

| Density | Atomic Radius | Electronegativity |
|---------:|--------------:|------------------:|
| 2.70 | 1.43 | 1.61 |
| 7.85 | 1.26 | 1.83 |
| 8.96 | 1.28 | 1.90 |

Notice that

- each **row** represents one material,
- each **column** represents one feature.

This is exactly how machine learning libraries expect data to be organized.

---

# 0.40 Creating a Two-Dimensional Array

```python
import numpy as np

materials = np.array([
    [2.70, 1.43, 1.61],
    [7.85, 1.26, 1.83],
    [8.96, 1.28, 1.90]
])

print(materials)
```

## Line-by-Line Explanation

### Line 1

```python
import numpy as np
```

Imports the NumPy library.

Every NumPy program begins with this statement.

---

### Line 3

```python
materials = np.array([
```

Creates a NumPy array named `materials`.

Notice that the opening square bracket is followed by another square bracket.

This indicates that we are creating a list of rows.

---

### Line 4

```python
[2.70, 1.43, 1.61],
```

This is the **first row**.

It represents one material.

Its features are

- Density = 2.70
- Atomic Radius = 1.43
- Electronegativity = 1.61

---

### Line 5

```python
[7.85, 1.26, 1.83],
```

This is the second material.

---

### Line 6

```python
[8.96, 1.28, 1.90]
```

This is the third material.

---

### Line 7

```python
])
```

Closes the NumPy array.

---

### Line 9

```python
print(materials)
```

Displays the complete matrix.

The output is

```text
[[2.70 1.43 1.61]
 [7.85 1.26 1.83]
 [8.96 1.28 1.90]]
```

---

# 0.41 Rows and Columns

Understanding rows and columns is one of the most important concepts in machine learning.

Suppose we have

```text
          Density   Radius   Electronegativity

Material A   2.70      1.43          1.61

Material B   7.85      1.26          1.83

Material C   8.96      1.28          1.90
```

Rows correspond to **samples**.

Columns correspond to **features**.

Machine learning models always assume

```text
Rows

↓

Samples (Materials)

Columns

↓

Features (Descriptors)
```

This convention is used throughout Scikit-Learn, XGBoost, TensorFlow, and PyTorch.

---

# 0.42 Checking the Shape of an Array

Before training a machine learning model, we must know the size of our dataset.

NumPy provides the `shape` attribute.

```python
import numpy as np

materials = np.array([
    [2.70, 1.43, 1.61],
    [7.85, 1.26, 1.83],
    [8.96, 1.28, 1.90]
])

print(materials.shape)
```

## Line-by-Line Explanation

### Line 1

Imports NumPy.

---

### Lines 3–7

Create the two-dimensional array.

---

### Line 9

```python
print(materials.shape)
```

`shape` tells us the dimensions of the array.

Unlike `print()`, **`shape` is not a function**.

It is an **attribute** of the NumPy array.

Therefore, we write

```python
materials.shape
```

and **not**

```python
materials.shape()
```

The output is

```text
(3, 3)
```

This means

- 3 rows,
- 3 columns.

---

# 0.43 What Is an Attribute?

This is an excellent place to introduce an important programming concept.

Python objects have two kinds of things associated with them:

- **Attributes**
- **Methods**

### Attributes

An attribute stores information **about** an object.

Example:

```python
materials.shape
```

asks,

> "What is the shape of this array?"

No calculation is performed.

Python simply returns stored information.

---

### Methods

A method performs an action.

Example:

```python
materials.reshape()
```

This asks Python to perform an operation.

A useful way to remember the difference is

| Type | Purpose |
|------|---------|
| Attribute | Stores information |
| Method | Performs an action |

We will encounter many attributes and methods throughout this book.

---

# 0.44 Finding the Number of Rows

Sometimes we need only the number of materials.

```python
import numpy as np

materials = np.array([
    [2.70, 1.43, 1.61],
    [7.85, 1.26, 1.83],
    [8.96, 1.28, 1.90]
])

print(materials.shape[0])
```

## Line-by-Line Explanation

### Lines 1–7

Import NumPy and create the array.

---

### Line 9

```python
print(materials.shape[0])
```

Let's break this down carefully.

`materials.shape`

returns

```text
(3, 3)
```

The first value represents the number of rows.

The second value represents the number of columns.

`[0]`

means

> "Retrieve the first value."

Therefore,

```python
materials.shape[0]
```

returns

```text
3
```

which is the number of materials.

---

# 0.45 Finding the Number of Columns

Similarly,

```python
print(materials.shape[1])
```

returns

```text
3
```

This time,

`[1]`

retrieves the second value of the shape tuple.

This corresponds to the number of features.

---

# 0.46 Why Shape Is Important in Machine Learning

Suppose your dataset contains

- 5,000 materials,
- 150 descriptors.

The shape would be

```text
(5000, 150)
```

This immediately tells you

- you have 5,000 samples,
- each sample contains 150 features.

Whenever you load a new dataset, one of the first things professional data scientists do is check its shape.

It quickly reveals whether the data has been loaded correctly and whether the expected number of samples and features are present.

---


In the next section, we will learn **array indexing and slicing**, one of the most frequently used skills in scientific programming and machine learning.

# 0.47 Array Indexing and Slicing

In the previous section, we learned how to create a two-dimensional NumPy array.

Now we must learn how to retrieve information from it.

This process is called **indexing**.

Professional machine learning engineers use indexing constantly because they frequently need to

- select specific materials,
- select particular features,
- create training datasets,
- prepare data for machine learning models.

If you master indexing, working with datasets becomes much easier.

---

# 0.48 Our Example Dataset

Throughout this section, we will use the following dataset.

```python
import numpy as np

materials = np.array([
    [2.70, 1.43, 1.61],
    [7.85, 1.26, 1.83],
    [8.96, 1.28, 1.90],
    [4.51, 1.47, 1.54]
])
```

This dataset represents four materials.

| Material | Density | Atomic Radius | Electronegativity |
|---------:|---------:|--------------:|------------------:|
| 1 | 2.70 | 1.43 | 1.61 |
| 2 | 7.85 | 1.26 | 1.83 |
| 3 | 8.96 | 1.28 | 1.90 |
| 4 | 4.51 | 1.47 | 1.54 |

Remember:

- Rows represent materials.
- Columns represent features.

---

# 0.49 Accessing One Element

Suppose we want the density of the second material.

```python
import numpy as np

materials = np.array([
    [2.70, 1.43, 1.61],
    [7.85, 1.26, 1.83],
    [8.96, 1.28, 1.90],
    [4.51, 1.47, 1.54]
])

print(materials[1, 0])
```

## Line-by-Line Explanation

### Line 1

```python
import numpy as np
```

Imports the NumPy library.

---

### Lines 3–7

```python
materials = np.array([
    [2.70, 1.43, 1.61],
    [7.85, 1.26, 1.83],
    [8.96, 1.28, 1.90],
    [4.51, 1.47, 1.54]
])
```

Creates a two-dimensional NumPy array.

---

### Line 9

```python
print(materials[1, 0])
```

Let's examine this carefully.

`materials`

refers to the NumPy array.

`[1, 0]`

contains two indices.

The first index selects the **row**.

The second index selects the **column**.

Therefore,

- Row 1 → second material
- Column 0 → density

Python retrieves

```text
7.85
```

Notice the order:

```text
[row, column]
```

Always remember this convention.

---

# 0.50 Selecting an Entire Row

Sometimes we need all the features belonging to one material.

```python
print(materials[2])
```

## Line-by-Line Explanation

### Line 1

```python
print(materials[2])
```

Here only one index is provided.

Python understands that we want the entire third row.

The output becomes

```text
[8.96 1.28 1.90]
```

This row represents one complete material.

In machine learning,

one row corresponds to one **sample**.

---

# 0.51 Selecting an Entire Column

Often we need every value of a single feature.

Suppose we want all density values.

```python
print(materials[:, 0])
```

## Line-by-Line Explanation

### Line 1

```python
print(materials[:, 0])
```

This statement introduces a new symbol:

```python
:
```

The colon means

> "Select everything."

Therefore,

`:` selects every row.

`0` selects the first column.

Python returns

```text
[2.70 7.85 8.96 4.51]
```

These are all density values.

This operation is extremely common in machine learning because models often process one feature at a time.

---

# 0.52 Understanding the Colon (:)

The colon is called the **slice operator**.

It allows us to select ranges of data.

Think of it as saying

> "Give me everything in this direction."

Examples:

```python
materials[:, 0]
```

All rows,

first column.

---

```python
materials[1, :]
```

Second row,

all columns.

---

```python
materials[:, :]
```

All rows,

all columns.

This simply returns the entire dataset.

---

# 0.53 Selecting Multiple Rows

Suppose we want only the first three materials.

```python
print(materials[0:3])
```

## Line-by-Line Explanation

### Line 1

```python
print(materials[0:3])
```

The notation

```python
0:3
```

means

Start at row **0**.

Stop **before** row **3**.

Therefore,

Python returns rows

- 0
- 1
- 2

The fourth row is **not included**.

This is an important rule:

> The ending index is excluded.

---

# 0.54 Selecting Multiple Columns

Suppose we want only

- Density
- Atomic Radius

and we do not need Electronegativity.

```python
print(materials[:, 0:2])
```

## Line-by-Line Explanation

### Line 1

```python
print(materials[:, 0:2])
```

The colon selects every material.

`0:2`

means

Start at column 0.

Stop before column 2.

Therefore,

Python returns

- Density
- Atomic Radius

but not Electronegativity.

The result is

```text
[[2.70 1.43]
 [7.85 1.26]
 [8.96 1.28]
 [4.51 1.47]]
```

---

# 0.55 Why Slicing Is Important

Suppose a machine learning dataset contains

150 features.

You may decide that only the first 20 features are useful.

Instead of creating a new dataset manually,

you can simply write

```python
X = data[:, 0:20]
```

This creates a new feature matrix containing only the selected columns.

Professional data scientists use slicing constantly because it is efficient and keeps code clean.

---

# 0.56 Visualizing Indexing

Consider the following matrix.

```text
          Column
        0     1     2

Row 0  2.70  1.43  1.61

Row 1  7.85  1.26  1.83

Row 2  8.96  1.28  1.90

Row 3  4.51  1.47  1.54
```

Examples:

```python
materials[2,1]
```

returns

```text
1.28
```

---

```python
materials[:,2]
```

returns the entire Electronegativity column.

---

```python
materials[1,:]
```

returns every feature of the second material.

---


These operations will be used repeatedly throughout the rest of the book when preparing datasets, selecting features, splitting data, and training machine learning models.

The next section introduces **NumPy mathematical functions**, where we will learn how to calculate statistics such as the mean, maximum, minimum, and standard deviation of material properties using only a few lines of Python.

# 0.57 NumPy Mathematical Functions

Now that we know how to create NumPy arrays and extract information from them, we can begin performing scientific calculations.

In Materials Informatics, raw data alone has little value.

Researchers constantly ask questions such as:

- What is the average density of these materials?
- Which material has the highest hardness?
- Which sample has the lowest thermal conductivity?
- How much variation exists in the experimental measurements?

NumPy provides built-in mathematical functions that answer these questions efficiently.

Instead of writing long programs, we can often obtain the answer using a single function.

---

# 0.58 Example Dataset

Throughout this section we will use the following dataset.

```python
import numpy as np

density = np.array([2.70, 7.85, 8.96, 4.51, 19.30])
```

Each value represents the density (g/cm³) of a material.

---

# 0.59 Calculating the Average (Mean)

One of the most common statistical quantities is the **mean**.

The mathematical definition of the mean is

$$
\bar{x}=\frac{x_1+x_2+x_3+\cdots+x_n}{n}
$$

where

- $x_i$ represents each observation.
- $n$ is the total number of observations.

Instead of calculating this manually, NumPy provides the `mean()` function.

```python
import numpy as np

density = np.array([2.70, 7.85, 8.96, 4.51, 19.30])

average_density = np.mean(density)

print(average_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
import numpy as np
```

Imports the NumPy library.

Without this line, Python would not recognize `np.mean()`.

---

### Line 3

```python
density = np.array([2.70, 7.85, 8.96, 4.51, 19.30])
```

Creates a NumPy array containing five density values.

These values will be used for all subsequent calculations.

---

### Line 5

```python
average_density = np.mean(density)
```

Let's examine this carefully.

`np`

refers to the NumPy library.

`mean()`

is a NumPy function that calculates the arithmetic mean.

`density`

is passed into the function.

NumPy automatically

1. adds all five numbers,
2. divides the sum by five,
3. returns the average.

The result is stored in the variable `average_density`.

---

### Line 7

```python
print(average_density)
```

Displays the calculated average.

---

# 0.60 Finding the Maximum Value

Researchers often need to identify the material with the highest property value.

NumPy provides the `max()` function.

```python
maximum_density = np.max(density)

print(maximum_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
maximum_density = np.max(density)
```

`max()` examines every value in the array.

It identifies the largest numerical value.

The answer is stored in the variable `maximum_density`.

---

### Line 3

```python
print(maximum_density)
```

Displays

```text
19.3
```

which is the highest density in the dataset.

---

# 0.61 Finding the Minimum Value

Similarly, NumPy provides the `min()` function.

```python
minimum_density = np.min(density)

print(minimum_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
minimum_density = np.min(density)
```

NumPy searches every value.

It returns the smallest value.

---

### Line 3

```python
print(minimum_density)
```

Displays

```text
2.7
```

---

# 0.62 Calculating the Sum

Sometimes researchers require the total rather than the average.

NumPy provides the `sum()` function.

```python
total_density = np.sum(density)

print(total_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
total_density = np.sum(density)
```

NumPy adds every value stored inside the array.

The result is stored in `total_density`.

---

### Line 3

```python
print(total_density)
```

Displays the total of all density values.

---

# 0.63 Standard Deviation

In experimental science,

measurements rarely have identical values.

Some variation always exists.

The **standard deviation** measures how widely the data are spread around the mean.

A small standard deviation indicates that most measurements are close to the average.

A large standard deviation indicates significant variation.

Mathematically,

$$
\sigma=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}(x_i-\bar{x})^2
}
$$

Fortunately,

NumPy performs this calculation automatically.

```python
standard_deviation = np.std(density)

print(standard_deviation)
```

---

## Line-by-Line Explanation

### Line 1

```python
standard_deviation = np.std(density)
```

`std()` calculates the standard deviation of every value in the array.

Internally,

NumPy

1. calculates the mean,
2. determines the difference between each value and the mean,
3. squares each difference,
4. averages those squared differences,
5. takes the square root.

All of these calculations occur automatically.

---

### Line 3

```python
print(standard_deviation)
```

Displays the calculated standard deviation.

---

# 0.64 Why Are These Functions Important?

These functions may appear simple, but they are used constantly in machine learning.

Before training any model, a data scientist usually explores the dataset by asking questions such as

- What is the average value?
- What is the smallest value?
- What is the largest value?
- Is the data highly variable?

This process is called **Exploratory Data Analysis (EDA)**.

EDA helps identify errors, unusual values, and important patterns before building a machine learning model.

---

# 0.65 Summary Statistics in One Table

| Function | Purpose |
|-----------|---------|
| `np.mean()` | Calculates the average |
| `np.max()` | Finds the largest value |
| `np.min()` | Finds the smallest value |
| `np.sum()` | Adds all values |
| `np.std()` | Calculates the standard deviation |

You will use these functions repeatedly throughout the remainder of this book.

---

# 0.66 Practice Exercise

Consider the following dataset.

```python
hardness = np.array([120, 145, 132, 158, 149])
```

Without looking at the answers, try to write Python code that calculates

1. the average hardness,
2. the maximum hardness,
3. the minimum hardness,
4. the total hardness,
5. the standard deviation of hardness.

Attempting these exercises yourself is one of the fastest ways to become comfortable with NumPy.

---

In the next section, we will learn **NumPy array operations and broadcasting**, a powerful feature that allows mathematical operations to be applied efficiently to entire datasets with a single line of code.

# 0.67 NumPy Array Operations and Broadcasting

One of the biggest advantages of NumPy is that it allows us to perform mathematical operations on **entire arrays** rather than one value at a time.

This capability is one of the main reasons NumPy is used in almost every scientific and machine learning application.

Imagine that you have measured the density of 50,000 materials.

If every density value must be converted from **g/cm³** to **kg/m³**, would you really want to calculate each one individually?

Of course not.

NumPy allows us to perform the conversion with a single statement.

This feature is called **vectorized computation**.

---

# 0.68 Mathematical Operations on Arrays

Suppose we have the following density values.

```python
import numpy as np

density = np.array([2.70, 7.85, 8.96])
```

Now suppose we want to increase every value by 1.

```python
new_density = density + 1

print(new_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
new_density = density + 1
```

Python examines every value stored inside `density`.

Internally it performs

```text
2.70 + 1

7.85 + 1

8.96 + 1
```

The results become

```text
3.70

8.85

9.96
```

These new values are stored in the variable `new_density`.

Notice that the original array remains unchanged.

---

### Line 3

```python
print(new_density)
```

Displays

```text
[3.70 8.85 9.96]
```

---

# 0.69 Multiplying an Entire Array

Suppose we want to double every measurement.

```python
double_density = density * 2

print(double_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
double_density = density * 2
```

NumPy multiplies every element by 2.

Internally,

```text
2.70 × 2

7.85 × 2

8.96 × 2
```

The resulting array becomes

```text
[5.40 15.70 17.92]
```

---

### Line 3

```python
print(double_density)
```

Displays the new array.

---

# 0.70 Dividing an Array

Division works in exactly the same way.

```python
half_density = density / 2

print(half_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
half_density = density / 2
```

Each value is divided by 2.

The operation is automatically applied to every element.

---

### Line 3

```python
print(half_density)
```

Displays the divided values.

---

# 0.71 Performing Operations Between Two Arrays

NumPy can also perform calculations between two arrays.

Suppose we measured the densities of two groups of materials.

```python
density_A = np.array([2.70, 7.85, 8.96])

density_B = np.array([0.30, 0.15, 0.04])

total_density = density_A + density_B

print(total_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
density_A = np.array([2.70, 7.85, 8.96])
```

Creates the first array.

---

### Line 3

```python
density_B = np.array([0.30, 0.15, 0.04])
```

Creates the second array.

Notice that both arrays contain the same number of elements.

---

### Line 5

```python
total_density = density_A + density_B
```

NumPy adds corresponding elements.

Internally,

```text
2.70 + 0.30

7.85 + 0.15

8.96 + 0.04
```

The resulting array becomes

```text
[3.00 8.00 9.00]
```

---

### Line 7

```python
print(total_density)
```

Displays the calculated array.

---

# 0.72 What Is Broadcasting?

So far,

the arrays we added had the same size.

Broadcasting allows NumPy to perform calculations even when the sizes are different, provided they are compatible.

Suppose we have

```python
density = np.array([2.70, 7.85, 8.96])
```

and we write

```python
density + 1
```

The number `1` is not an array.

It is a single value.

NumPy automatically treats it as if it were

```text
[1 1 1]
```

before performing the calculation.

This automatic expansion is called **broadcasting**.

The programmer does not have to create the larger array manually.

---

# 0.73 Why Is Broadcasting Useful?

Imagine a dataset containing

100,000 materials.

Suppose every temperature measurement must be increased by 273.15 to convert from Celsius to Kelvin.

Instead of writing a loop,

you simply write

```python
temperature_kelvin = temperature_celsius + 273.15
```

NumPy automatically adds **273.15** to every value.

This makes programs

- shorter,
- faster,
- easier to read,
- less prone to programming errors.

---

# 0.74 Broadcasting Rules (Beginner Level)

At this stage, you only need to remember one simple rule.

If one value can reasonably be applied to every element in an array, NumPy usually performs the operation automatically.

Examples include

```python
array + 5
```

```python
array - 10
```

```python
array * 2
```

```python
array / 100
```

Later in the book, when we work with multi-dimensional datasets, we will study the complete broadcasting rules in more detail.

---

# 0.75 Why Vectorized Operations Matter in Machine Learning

Machine learning algorithms repeatedly perform millions of mathematical operations.

If every calculation required a Python loop, training would become much slower.

Vectorized NumPy operations are implemented in highly optimized compiled code.

This is one of the reasons Python remains competitive for scientific computing despite being an interpreted language.

---

# 0.76 Practice Exercise

Create the following array.

```python
hardness = np.array([120, 140, 160, 180])
```

Write Python code that

1. adds 10 to every value,
2. multiplies every value by 2,
3. divides every value by 5,
4. subtracts 20 from every value.

Before running your code, try to predict the resulting arrays.

Developing this habit will greatly improve your programming skills.

---


In the next section, we will begin working with **Pandas**, the library used to organize, inspect, clean, and manipulate datasets before they are fed into machine learning models. Nearly every real-world machine learning project starts with Pandas.

# 0.77 Introduction to Pandas

So far, we have learned how to use NumPy arrays for numerical computation.

NumPy is extremely powerful for mathematics.

However, real-world machine learning datasets are rarely stored as anonymous arrays.

Instead, they usually look like spreadsheets.

For example,

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |
| Aluminum | 2.70 | 0.00 | 167 |
| Silicon | 2.33 | 1.12 | 1150 |
| Copper | 8.96 | 0.00 | 369 |

Notice something important.

Each column has a meaningful name.

Instead of referring to

```text
Column 0

Column 1

Column 2
```

we can refer to

```text
Density

Bandgap

Hardness
```

This makes datasets much easier to understand.

Python provides the **Pandas** library specifically for working with tabular data.

If NumPy is the language of numerical computation,

then Pandas is the language of datasets.

Almost every machine learning project begins with Pandas.

---

# 0.78 Why Do We Need Pandas?

Suppose you download experimental data from a research laboratory.

The dataset may contain

- material names,
- crystal structures,
- densities,
- elastic moduli,
- thermal conductivity,
- melting temperatures,
- processing conditions.

Some columns contain numbers.

Some contain text.

Some contain missing values.

Managing all of this information with NumPy alone becomes inconvenient.

Pandas was designed to solve exactly this problem.

It provides powerful tools for

- organizing data,
- filtering data,
- cleaning data,
- selecting columns,
- handling missing values,
- preparing datasets for machine learning.

---

# 0.79 Importing Pandas

Before using Pandas, we must import it.

```python
import pandas as pd
```

## Line-by-Line Explanation

### Line 1

```python
import pandas as pd
```

The keyword `import` tells Python to load an external library.

`pandas`

is the official name of the library.

`as pd`

creates a short alias.

Instead of writing

```python
pandas.DataFrame()
```

we simply write

```python
pd.DataFrame()
```

This is the standard convention used by almost every Python programmer.

Whenever you see `pd`, you should immediately recognize that it refers to Pandas.

---

# 0.80 What Is a DataFrame?

The most important object in Pandas is the **DataFrame**.

A DataFrame is a table consisting of rows and columns.

You can think of it as a spreadsheet inside Python.

For example,

| Material | Density | Bandgap |
|-----------|---------:|---------:|
| Steel | 7.85 | 0.00 |
| Aluminum | 2.70 | 0.00 |
| Silicon | 2.33 | 1.12 |

Each row represents one observation.

Each column represents one variable.

This is exactly the format expected by most machine learning algorithms.

---

# 0.81 Creating Your First DataFrame

```python
import pandas as pd

data = {
    "Material": ["Steel", "Aluminum", "Silicon"],
    "Density": [7.85, 2.70, 2.33],
    "Bandgap": [0.00, 0.00, 1.12]
}

df = pd.DataFrame(data)

print(df)
```

---

## Line-by-Line Explanation

### Line 1

```python
import pandas as pd
```

Imports the Pandas library.

---

### Line 3

```python
data = {
```

Creates a Python dictionary.

Each key will become a column name.

---

### Line 4

```python
"Material": ["Steel", "Aluminum", "Silicon"],
```

Creates the first column.

The column name is `"Material"`.

The values are stored as a list.

---

### Line 5

```python
"Density": [7.85, 2.70, 2.33],
```

Creates the second column.

Each density corresponds to the material in the same row.

---

### Line 6

```python
"Bandgap": [0.00, 0.00, 1.12]
```

Creates the third column.

Again, each value corresponds to the material in the same row.

---

### Line 7

```python
}
```

Closes the dictionary.

---

### Line 9

```python
df = pd.DataFrame(data)
```

This is the most important line.

`pd`

refers to the Pandas library.

`DataFrame()`

is a constructor that creates a DataFrame.

The dictionary `data` is converted into a neatly organized table.

The resulting DataFrame is stored in the variable `df`.

Many programmers use the name `df` because it is short for **DataFrame**.

---

### Line 11

```python
print(df)
```

Displays the complete table.

The output resembles

```text
    Material  Density  Bandgap
0      Steel     7.85      0.00
1  Aluminum     2.70      0.00
2   Silicon     2.33      1.12
```

Notice that Pandas automatically assigns row numbers called the **index**.

---

# 0.82 Understanding Rows and Columns

Let's examine the DataFrame carefully.

| Index | Material | Density | Bandgap |
|------:|-----------|---------:|---------:|
| 0 | Steel | 7.85 | 0.00 |
| 1 | Aluminum | 2.70 | 0.00 |
| 2 | Silicon | 2.33 | 1.12 |

The leftmost numbers are called the **index**.

The index uniquely identifies each row.

The remaining columns contain the actual data.

In machine learning,

- rows represent samples,
- columns represent features or target variables.

---

# 0.83 Why DataFrames Are Better Than NumPy Arrays

Suppose someone gives you the following NumPy array.

```text
[[7.85 0.00]
 [2.70 0.00]
 [2.33 1.12]]
```

Can you immediately identify

- which column is density?
- which column is bandgap?

Probably not.

Now consider the equivalent DataFrame.

| Material | Density | Bandgap |
|-----------|---------:|---------:|
| Steel | 7.85 | 0.00 |
| Aluminum | 2.70 | 0.00 |
| Silicon | 2.33 | 1.12 |

The meaning of each column is immediately clear.

This is why Pandas is widely used for data analysis and machine learning.

---

# 0.84 DataFrames in the Machine Learning Workflow

A typical machine learning workflow looks like this:

```text
CSV File

↓

Pandas DataFrame

↓

Data Cleaning

↓

Feature Selection

↓

NumPy Array

↓

Machine Learning Model

↓

Predictions
```

Notice that both Pandas and NumPy play important roles.

Pandas is primarily used for organizing and cleaning data.

NumPy is primarily used for numerical computation.

Machine learning libraries use both extensively.

---


In the next section, we will learn how to **read datasets from CSV files**, inspect their contents, and explore them using the most commonly used Pandas functions in data science.

# 0.85 Reading Datasets with Pandas

Creating a DataFrame manually is useful for learning.

However, in real research projects, datasets usually already exist.

They may come from

- laboratory experiments,
- computational simulations,
- public materials databases,
- Excel spreadsheets,
- CSV (Comma-Separated Values) files.

Among these, **CSV files** are the most common format used in machine learning.

Learning how to read CSV files is one of the first practical skills every data scientist must master.

---

# 0.86 What Is a CSV File?

A CSV file is a plain text file in which values are separated by commas.

For example, a file named `materials.csv` might contain

```text
Material,Density,Bandgap,Hardness
Steel,7.85,0.00,250
Aluminum,2.70,0.00,167
Silicon,2.33,1.12,1150
Copper,8.96,0.00,369
```

Each line represents one material.

Each comma separates one column from the next.

Although the file is plain text, Pandas can automatically organize it into a DataFrame.

---

# 0.87 Reading a CSV File

Suppose the file `materials.csv` is stored in the same folder as your Python program.

The following code loads the dataset.

```python
import pandas as pd

df = pd.read_csv("materials.csv")

print(df)
```

---

## Line-by-Line Explanation

### Line 1

```python
import pandas as pd
```

Imports the Pandas library.

---

### Line 3

```python
df = pd.read_csv("materials.csv")
```

Let's examine this carefully.

`pd`

refers to the Pandas library.

`read_csv()`

is a Pandas function that reads CSV files.

`"materials.csv"`

is the filename.

Pandas opens the file, reads every row and column, and converts the contents into a DataFrame.

The DataFrame is stored in the variable `df`.

---

### Line 5

```python
print(df)
```

Displays the dataset.

If the file contains hundreds or thousands of rows, they will be displayed in tabular form.

---

# 0.88 Viewing the First Few Rows

Large datasets may contain thousands of materials.

Displaying the entire dataset is often unnecessary.

Pandas provides the `head()` method.

```python
print(df.head())
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.head())
```

`head()` returns the first five rows of the DataFrame by default.

This allows you to quickly verify that the dataset has been loaded correctly.

For example,

```text
    Material  Density  Bandgap  Hardness
0      Steel     7.85      0.00       250
1  Aluminum     2.70      0.00       167
2   Silicon     2.33      1.12      1150
3     Copper     8.96      0.00       369
4     Nickel     8.90      0.00       638
```

---

# 0.89 Viewing the Last Few Rows

Sometimes we want to inspect the end of the dataset.

Pandas provides the `tail()` method.

```python
print(df.tail())
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.tail())
```

`tail()` displays the last five rows of the DataFrame.

This is useful for checking whether the dataset has been loaded completely.

---

# 0.90 Viewing a Specific Number of Rows

Both `head()` and `tail()` accept a numerical argument.

For example,

```python
print(df.head(3))
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.head(3))
```

The number `3` tells Pandas to display only the first three rows instead of the default five.

Similarly,

```python
print(df.tail(2))
```

displays the last two rows.

---

# 0.91 Determining the Size of the Dataset

Before building a machine learning model, we should know how large our dataset is.

Pandas provides the `shape` attribute.

```python
print(df.shape)
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.shape)
```

`shape` returns a tuple containing

```text
(number of rows, number of columns)
```

For example,

```text
(5000, 18)
```

means

- 5,000 materials,
- 18 columns.

Remember that `shape` is an **attribute**, not a method.

Therefore,

✔ Correct

```python
df.shape
```

✘ Incorrect

```python
df.shape()
```

---

# 0.92 Displaying Column Names

Machine learning begins by understanding the available features.

Pandas allows us to display all column names.

```python
print(df.columns)
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.columns)
```

`columns` is an attribute of the DataFrame.

It returns the names of every column.

The output might look like

```text
Index(['Material',
       'Density',
       'Bandgap',
       'Hardness'],
      dtype='object')
```

Knowing the column names is essential because later we will select specific features for model training.

---

# 0.93 Displaying Basic Information

Another useful method is `info()`.

```python
df.info()
```

---

## Line-by-Line Explanation

### Line 1

```python
df.info()
```

This method provides a summary of the DataFrame.

It reports

- the number of rows,
- the number of columns,
- the name of each column,
- the data type of each column,
- the number of non-missing values,
- the memory usage.

For example,

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 18 columns):
...
```

This is one of the first commands that professional data scientists execute after loading a dataset.

---

# 0.94 Why Inspecting Data Is Important

Imagine training a machine learning model without first checking the dataset.

Potential problems include

- missing columns,
- incorrect data types,
- empty rows,
- incomplete data,
- incorrectly loaded files.

Inspecting the dataset before analysis helps detect these issues early and prevents errors later in the workflow.

---

# 0.95 A Typical Workflow

Most data science projects begin with the following sequence of commands:

```python
import pandas as pd

df = pd.read_csv("materials.csv")

print(df.head())

print(df.shape)

print(df.columns)

df.info()
```

This small block of code provides a quick overview of almost any dataset.

As you gain experience, these commands will become second nature.

---


In the next section, we will learn how to **select rows and columns**, filter data, and manipulate DataFrames using Pandas. These skills are essential for preparing datasets before training machine learning models.

# 0.96 Selecting Data from a Pandas DataFrame

Loading a dataset is only the first step.

The next step is learning how to retrieve exactly the information you need.

Real-world datasets often contain

- hundreds of columns,
- thousands of rows,
- multiple target variables.

Machine learning rarely uses every column.

Instead, we carefully select the information relevant to the prediction task.

Pandas provides powerful tools for selecting rows, columns, and subsets of data.

Learning these tools is essential because you will use them in almost every machine learning project.

---

# 0.97 Selecting a Single Column

Consider the following DataFrame.

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |
| Aluminum | 2.70 | 0.00 | 167 |
| Silicon | 2.33 | 1.12 | 1150 |
| Copper | 8.96 | 0.00 | 369 |

Suppose we want only the **Density** column.

```python
density = df["Density"]

print(density)
```

---

## Line-by-Line Explanation

### Line 1

```python
density = df["Density"]
```

Let's examine every part.

`df`

refers to the DataFrame.

`["Density"]`

specifies the column named **Density**.

Pandas searches for that column and returns all of its values.

The result is stored in the variable `density`.

Notice that the column name is written as a string because it is a label, not a variable.

---

### Line 3

```python
print(density)
```

Displays

```text
0    7.85
1    2.70
2    2.33
3    8.96
Name: Density, dtype: float64
```

The returned object is called a **Series**.

A Series is a one-dimensional data structure in Pandas.

You can think of it as a single column extracted from a DataFrame.

---

# 0.98 Selecting Multiple Columns

Machine learning models usually require more than one feature.

Suppose we want both

- Density
- Hardness

```python
features = df[["Density", "Hardness"]]

print(features)
```

---

## Line-by-Line Explanation

### Line 1

```python
features = df[["Density", "Hardness"]]
```

Notice something important.

We now use **two pairs of square brackets**.

The outer brackets tell Pandas that we are selecting columns from the DataFrame.

The inner brackets create a Python list containing the names of the columns.

Pandas returns a new DataFrame containing only these columns.

---

### Line 3

```python
print(features)
```

Displays

```text
   Density  Hardness
0     7.85       250
1     2.70       167
2     2.33      1150
3     8.96       369
```

Unlike selecting a single column, selecting multiple columns returns another **DataFrame** rather than a Series.

---

# 0.99 Selecting Rows with `iloc`

Pandas provides two primary ways to select rows.

The first is `iloc`, which selects rows by **integer position**.

Suppose we want the first row.

```python
print(df.iloc[0])
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.iloc[0])
```

`iloc`

stands for **integer location**.

The number `0` refers to the first row.

The output contains all values from that row.

```text
Material      Steel
Density        7.85
Bandgap        0.00
Hardness        250
```

---

# 0.100 Selecting Multiple Rows with `iloc`

Suppose we want the first three rows.

```python
print(df.iloc[0:3])
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.iloc[0:3])
```

The slice `0:3` means

- start at row 0,
- stop before row 3.

The result contains rows

- 0
- 1
- 2

This slicing behavior is identical to NumPy.

---

# 0.101 Selecting Specific Rows and Columns

`iloc` can also select both rows and columns simultaneously.

```python
print(df.iloc[0:3, 1:3])
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.iloc[0:3, 1:3])
```

The notation follows the pattern

```python
[row_selection, column_selection]
```

`0:3`

selects the first three rows.

`1:3`

selects columns with integer positions

- 1
- 2

If the columns are

| Position | Column |
|---------:|---------|
| 0 | Material |
| 1 | Density |
| 2 | Bandgap |
| 3 | Hardness |

then the result contains

- Density
- Bandgap

for the first three materials.

---

# 0.102 Selecting Rows with `loc`

Unlike `iloc`, which uses integer positions, `loc` selects data using **labels**.

Suppose the DataFrame has an index containing material IDs.

```python
df = df.set_index("Material")
```

The DataFrame now looks like

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |
| Aluminum | 2.70 | 0.00 | 167 |
| Silicon | 2.33 | 1.12 | 1150 |
| Copper | 8.96 | 0.00 | 369 |

Now we can retrieve the row for Silicon.

```python
print(df.loc["Silicon"])
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.loc["Silicon"])
```

`loc`

stands for **label location**.

Instead of using an integer,

we use the row label.

Pandas searches for the label `"Silicon"` and returns the corresponding row.

---

# 0.103 Selecting a Column with Dot Notation

Pandas also allows columns to be accessed using dot notation.

```python
print(df.Density)
```

This is shorter than

```python
print(df["Density"])
```

However, many professional programmers prefer the bracket notation because it works for all column names, including those containing spaces or special characters.

For this reason, this book will primarily use

```python
df["Density"]
```

---

# 0.104 Why Data Selection Matters

Suppose our dataset contains

- 200 descriptors,
- one target property.

A machine learning model should not use every column.

Instead, we might select only

- Density,
- Atomic Radius,
- Electronegativity,
- Melting Point.

Data selection allows us to isolate the relevant features before model training.

This is a fundamental step in every machine learning workflow.

---

# 0.105 Practice Exercise

Using the DataFrame below,

```python
import pandas as pd

data = {
    "Material": ["Steel", "Aluminum", "Silicon", "Copper"],
    "Density": [7.85, 2.70, 2.33, 8.96],
    "Bandgap": [0.00, 0.00, 1.12, 0.00],
    "Hardness": [250, 167, 1150, 369]
}

df = pd.DataFrame(data)
```

Try to write Python code that

1. selects only the `Density` column,
2. selects both `Density` and `Bandgap`,
3. displays the first two rows,
4. displays the last two rows using `iloc`,
5. displays rows 1 to 3 and columns 1 to 2.

Attempt these exercises before checking the solutions.

Writing code yourself is the fastest way to become proficient.

---

In the next section, we will learn how to **filter data using conditions**, allowing us to extract subsets of materials that satisfy specific scientific criteria, such as density greater than a threshold or bandgap within a desired range.

# 0.106 Filtering Data Using Conditions

Selecting rows by their position is useful, but in scientific research we usually select data based on **conditions** rather than row numbers.

For example, we may ask questions such as

- Which materials have a density greater than **7 g/cm³**?
- Which materials are semiconductors with a bandgap larger than **1 eV**?
- Which materials have hardness above **500 HV**?
- Which materials satisfy multiple conditions simultaneously?

Answering these questions is called **filtering**.

Filtering is one of the most frequently used operations in data science and machine learning.

---

# 0.107 Our Example Dataset

Throughout this section, we will use the following DataFrame.

```python
import pandas as pd

data = {
    "Material": ["Steel", "Aluminum", "Silicon", "Copper", "Titanium"],
    "Density": [7.85, 2.70, 2.33, 8.96, 4.51],
    "Bandgap": [0.00, 0.00, 1.12, 0.00, 0.00],
    "Hardness": [250, 167, 1150, 369, 349]
}

df = pd.DataFrame(data)
```

This DataFrame contains five materials and three measured properties.

---

# 0.108 Comparison Operators

Before filtering data, we must understand **comparison operators**.

Comparison operators compare two values and return either

- `True`
- `False`

The most common comparison operators are

| Operator | Meaning |
|----------|---------|
| `==` | Equal to |
| `!=` | Not equal to |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal to |
| `<=` | Less than or equal to |

For example,

```python
7 > 5
```

returns

```text
True
```

while

```python
2 > 8
```

returns

```text
False
```

Filtering is built upon these logical comparisons.

---

# 0.109 Filtering a Single Condition

Suppose we want materials with density greater than **7 g/cm³**.

```python
high_density = df[df["Density"] > 7]

print(high_density)
```

---

## Line-by-Line Explanation

### Line 1

```python
high_density = df[df["Density"] > 7]
```

Let's examine this carefully.

`df["Density"]`

selects the Density column.

This column contains

```text
7.85
2.70
2.33
8.96
4.51
```

The comparison

```python
df["Density"] > 7
```

checks each value individually.

Internally, Pandas evaluates

```text
7.85 > 7   → True
2.70 > 7   → False
2.33 > 7   → False
8.96 > 7   → True
4.51 > 7   → False
```

This produces a **Boolean Series**

```text
True
False
False
True
False
```

Finally,

```python
df[ ... ]
```

keeps only the rows where the result is `True`.

The filtered DataFrame is stored in `high_density`.

---

### Line 3

```python
print(high_density)
```

Displays

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |
| Copper | 8.96 | 0.00 | 369 |

Only materials satisfying the condition remain.

---

# 0.110 Filtering with Equality

Suppose we want only Silicon.

```python
silicon = df[df["Material"] == "Silicon"]

print(silicon)
```

---

## Line-by-Line Explanation

### Line 1

```python
silicon = df[df["Material"] == "Silicon"]
```

`==`

means **equal to**.

Pandas compares every material name with `"Silicon"`.

Only one row satisfies the condition.

That row becomes the new DataFrame.

---

### Line 3

```python
print(silicon)
```

Displays only the Silicon row.

---

# 0.111 Filtering with Multiple Conditions

Scientific research often requires multiple conditions.

Suppose we want materials that

- have density greater than 5,
- and hardness greater than 300.

```python
selected = df[
    (df["Density"] > 5) &
    (df["Hardness"] > 300)
]

print(selected)
```

---

## Line-by-Line Explanation

### Line 1

```python
selected = df[
```

Begins a filtering operation.

---

### Line 2

```python
(df["Density"] > 5)
```

Creates the first condition.

---

### Line 3

```python
&
```

The symbol `&` means **logical AND**.

A row is selected **only if both conditions are true**.

---

### Line 4

```python
(df["Hardness"] > 300)
```

Creates the second condition.

---

### Line 5

```python
]
```

Closes the filtering expression.

---

### Line 7

```python
print(selected)
```

Displays only rows satisfying **both** conditions.

---

# 0.112 The Difference Between AND and OR

Python provides two important logical operators for combining conditions.

## AND (`&`)

Both conditions must be true.

```python
(df["Density"] > 5) &
(df["Hardness"] > 300)
```

Only rows satisfying **both** conditions are selected.

---

## OR (`|`)

At least one condition must be true.

```python
(df["Density"] > 7) |
(df["Bandgap"] > 1)
```

A row is selected if either condition is true.

For example,

Silicon has a large bandgap.

Copper has a high density.

Both would be selected because each satisfies one of the two conditions.

---

# 0.113 Filtering Semiconductors

Suppose we define semiconductors as materials with a bandgap greater than **1 eV**.

```python
semiconductors = df[df["Bandgap"] > 1]

print(semiconductors)
```

The resulting DataFrame contains only materials meeting this criterion.

This type of filtering is common in materials discovery.

Researchers often search databases for materials that satisfy property constraints before performing detailed simulations or experiments.

---

# 0.114 Why Filtering Is Important in Machine Learning

Filtering is commonly used before training a model.

Examples include

- removing materials with missing target values,
- selecting only metallic materials,
- selecting only semiconductors,
- removing measurements outside a valid experimental range,
- creating specialized datasets for different prediction tasks.

Proper filtering improves data quality and often leads to better machine learning performance.

---

# 0.115 Practice Exercise

Using the example DataFrame, write Python code that

1. selects materials with density less than **5**,
2. selects materials with hardness greater than **300**,
3. selects materials whose bandgap is greater than **0**,
4. selects materials with density greater than **5** **and** hardness greater than **300**,
5. selects materials with density greater than **7** **or** bandgap greater than **1**.

Try to solve these problems without looking back at the examples.

Learning comes from writing code yourself.

---


In the next section, we will learn how to **handle missing data (NaN values)**, one of the most important preprocessing steps before training any machine learning model. Most real-world datasets contain missing values, and learning how to detect and manage them is a critical data science skill.


# 0.116 Handling Missing Data

One of the biggest challenges in real-world machine learning is **missing data**.

In textbooks, datasets are usually complete.

Every material has

- density,
- hardness,
- bandgap,
- melting point,
- elastic modulus,

and every value is present.

Real scientific datasets are very different.

For example,

- an experiment may fail,
- an instrument may malfunction,
- a researcher may forget to record a value,
- some properties may simply never have been measured.

As a result, datasets often contain **missing values**.

Machine learning algorithms generally cannot handle missing values automatically.

Therefore, identifying and handling missing data is one of the first tasks performed by every data scientist.

---

# 0.117 What Is NaN?

Pandas represents missing numerical data using

```text
NaN
```

NaN stands for

> **Not a Number**

Although the name may seem strange, NaN simply indicates that the value is missing or undefined.

For example,

| Material | Density | Bandgap |
|-----------|---------:|---------:|
| Steel | 7.85 | 0.00 |
| Aluminum | NaN | 0.00 |
| Silicon | 2.33 | 1.12 |
| Copper | 8.96 | NaN |

Notice that

- Aluminum has no recorded density.
- Copper has no recorded bandgap.

---

# 0.118 Creating a DataFrame with Missing Values

```python
import pandas as pd
import numpy as np

data = {
    "Material": ["Steel", "Aluminum", "Silicon", "Copper"],
    "Density": [7.85, np.nan, 2.33, 8.96],
    "Bandgap": [0.00, 0.00, 1.12, np.nan]
}

df = pd.DataFrame(data)

print(df)
```

---

## Line-by-Line Explanation

### Line 1

```python
import pandas as pd
```

Imports Pandas.

---

### Line 2

```python
import numpy as np
```

Imports NumPy.

NumPy provides the special value

```python
np.nan
```

which represents missing numerical data.

---

### Lines 4–8

```python
data = {
    ...
}
```

Creates a dictionary.

Notice the entries

```python
np.nan
```

These indicate missing values.

---

### Line 10

```python
df = pd.DataFrame(data)
```

Converts the dictionary into a DataFrame.

---

### Line 12

```python
print(df)
```

Displays

```text
    Material  Density  Bandgap
0      Steel     7.85      0.00
1  Aluminum      NaN      0.00
2   Silicon     2.33      1.12
3     Copper     8.96       NaN
```

---

# 0.119 Detecting Missing Values

Before fixing missing data, we must first locate it.

Pandas provides the `isna()` method.

```python
print(df.isna())
```

---

## Line-by-Line Explanation

### Line 1

```python
print(df.isna())
```

`isna()` examines every value in the DataFrame.

If a value is missing,

it returns

```text
True
```

Otherwise,

it returns

```text
False
```

For our example,

```text
   Material Density Bandgap
0    False   False    False
1    False    True    False
2    False   False    False
3    False   False     True
```

This makes it easy to identify missing entries.

---

# 0.120 Counting Missing Values

Instead of looking through an entire table, we often want to know how many missing values exist in each column.

```python
print(df.isna().sum())
```

---

## Line-by-Line Explanation

### Line 1

```python
df.isna()
```

Creates a table of

`True`

and

`False`

values.

---

### Line 1 (continued)

```python
.sum()
```

In Python,

`True`

is treated as

```text
1
```

and

`False`

is treated as

```text
0
```

Therefore,

adding them counts the number of missing values.

The output becomes

```text
Material    0
Density     1
Bandgap     1
dtype: int64
```

This tells us

- Material has no missing values.
- Density has one missing value.
- Bandgap has one missing value.

---

# 0.121 Removing Missing Values

One simple solution is to remove incomplete rows.

Pandas provides the `dropna()` method.

```python
clean_df = df.dropna()

print(clean_df)
```

---

## Line-by-Line Explanation

### Line 1

```python
clean_df = df.dropna()
```

`dropna()`

removes every row containing at least one missing value.

The cleaned DataFrame is stored in

```python
clean_df
```

The original DataFrame

```python
df
```

remains unchanged.

---

### Line 3

```python
print(clean_df)
```

Displays

```text
   Material  Density  Bandgap
0     Steel     7.85      0.00
2  Silicon     2.33      1.12
```

Rows containing missing values have been removed.

---

# 0.122 Filling Missing Values

Removing rows is not always a good idea.

If the dataset is small,

deleting data may significantly reduce the amount of information available.

Instead,

we often replace missing values.

One simple approach is to use the column average.

```python
df["Density"] = df["Density"].fillna(df["Density"].mean())
```

---

## Line-by-Line Explanation

### Line 1

```python
df["Density"]
```

Selects the Density column.

---

### Line 1 (continued)

```python
.mean()
```

Calculates the average of all available density values.

Missing values are automatically ignored.

---

### Line 1 (continued)

```python
fillna(...)
```

Replaces every missing value with the calculated average.

---

### Line 1 (continued)

```python
df["Density"] =
```

Stores the updated column back into the DataFrame.

Now every row has a density value.

---

# 0.123 Common Strategies for Handling Missing Data

There is no single correct method.

The best strategy depends on the problem.

Some common approaches include

| Method | When to Use |
|---------|-------------|
| Remove rows | Few missing values |
| Replace with mean | Continuous numerical features |
| Replace with median | Data containing outliers |
| Replace with mode | Categorical variables |
| Predict missing values | Large, complex datasets |

As you progress through this book, you will learn more advanced imputation techniques.

---

# 0.124 Why Missing Data Matters

Imagine training a machine learning model using incomplete data.

If the algorithm encounters missing values unexpectedly,

it may

- stop with an error,
- produce unreliable predictions,
- learn incorrect relationships.

Therefore,

handling missing data is considered an essential preprocessing step before model training.

Professional data scientists always inspect datasets for missing values before fitting a machine learning model.

---

# 0.125 Practice Exercise

Using the DataFrame from this section,

write Python code to

1. display the locations of missing values,
2. count missing values in each column,
3. remove incomplete rows,
4. replace missing density values with the average density,
5. verify that no missing density values remain.

Attempt these exercises without referring to the solutions.

Practical coding experience is the best way to build confidence.

---

In the next section, we will learn **data visualization using Matplotlib**, where we will create scientific plots to explore material properties before training machine learning models.


# 0.126 Introduction to Data Visualization with Matplotlib

Machine learning begins with **understanding the data**.

Although tables of numbers are useful, they often hide important patterns.

Consider the following density values.

```text
2.70
7.85
8.96
4.51
19.30
```

Looking at these numbers alone, it is difficult to answer questions such as

- Which values occur most frequently?
- Are there any unusually large values?
- Is the data evenly distributed?
- Are there groups of similar materials?

A graph can answer these questions almost instantly.

For this reason, **data visualization** is an essential part of data science.

The most widely used visualization library in Python is **Matplotlib**.

Almost every scientific Python project uses it.

---

# 0.127 Why Visualization Is Important

Suppose two researchers receive the same dataset.

The first researcher immediately trains a machine learning model.

The second researcher first creates several graphs.

The second researcher is much more likely to discover

- incorrect measurements,
- missing values,
- unusual outliers,
- unexpected relationships,
- errors during data collection.

Visualization helps us understand the data before building a model.

Professional data scientists almost always visualize a dataset before performing machine learning.

---

# 0.128 Installing and Importing Matplotlib

Most Python environments already include Matplotlib.

To use it, we import the library.

```python
import matplotlib.pyplot as plt
```

---

## Line-by-Line Explanation

### Line 1

```python
import matplotlib.pyplot as plt
```

Let's examine this carefully.

`matplotlib`

is the name of the library.

`pyplot`

is a module that contains plotting functions.

`as plt`

creates a short alias.

Instead of writing

```python
matplotlib.pyplot.plot(...)
```

we simply write

```python
plt.plot(...)
```

Nearly every Python programmer uses the alias

```python
plt
```

so you should become familiar with it.

---

# 0.129 Our Example Dataset

Throughout this section, we will use the following data.

```python
import numpy as np

density = np.array([2.70, 7.85, 8.96, 4.51, 19.30])
```

These values represent the densities of several materials.

---

# 0.130 Creating Your First Line Plot

A line plot connects data points with straight lines.

Although line plots are mainly used for data that change continuously (such as temperature over time), they provide an excellent introduction to Matplotlib.

```python
import matplotlib.pyplot as plt
import numpy as np

density = np.array([2.70, 7.85, 8.96, 4.51, 19.30])

plt.plot(density)

plt.show()
```

---

## Line-by-Line Explanation

### Line 1

```python
import matplotlib.pyplot as plt
```

Imports the plotting module.

---

### Line 2

```python
import numpy as np
```

Imports NumPy.

---

### Line 4

```python
density = np.array([2.70, 7.85, 8.96, 4.51, 19.30])
```

Creates the dataset.

---

### Line 6

```python
plt.plot(density)
```

This is the most important plotting command.

`plot()`

creates a line graph.

Because only one array is supplied,

Matplotlib automatically

- uses the array indices as the x-axis,
- uses the array values as the y-axis.

Therefore,

the x-axis becomes

```text
0 1 2 3 4
```

and the y-axis becomes

```text
2.70
7.85
8.96
4.51
19.30
```

---

### Line 8

```python
plt.show()
```

This command displays the figure.

Without `show()`, many Python environments will not display the graph.

Think of `plt.show()` as telling Python,

> "I have finished creating the graph. Display it now."

---

# 0.131 Adding a Title

Every scientific figure should have a clear title.

```python
plt.plot(density)

plt.title("Density of Materials")

plt.show()
```

---

## Line-by-Line Explanation

### Line 2

```python
plt.title("Density of Materials")
```

Adds a title above the graph.

The title should briefly describe what the figure represents.

Good titles make scientific reports much easier to read.

---

# 0.132 Labeling the Axes

Scientific figures should also identify the meaning of each axis.

```python
plt.plot(density)

plt.xlabel("Material Number")

plt.ylabel("Density (g/cm³)")

plt.show()
```

---

## Line-by-Line Explanation

### Line 2

```python
plt.xlabel("Material Number")
```

Labels the horizontal (x) axis.

---

### Line 4

```python
plt.ylabel("Density (g/cm³)")
```

Labels the vertical (y) axis.

Including units is considered good scientific practice whenever possible.

---

# 0.133 Combining Everything

A professional-looking figure typically includes

- the graph,
- a title,
- x-axis label,
- y-axis label.

```python
import matplotlib.pyplot as plt
import numpy as np

density = np.array([2.70, 7.85, 8.96, 4.51, 19.30])

plt.plot(density)

plt.title("Density of Materials")

plt.xlabel("Material Number")

plt.ylabel("Density (g/cm³)")

plt.show()
```

This is the basic structure of many scientific plots.

---

# 0.134 Why Visualization Matters in Machine Learning

Suppose one density value is accidentally recorded as

```text
1930
```

instead of

```text
19.30
```

A quick graph would immediately reveal one point far above all the others.

Without visualization,

this mistake might go unnoticed and negatively affect the machine learning model.

Visualization helps detect

- data entry mistakes,
- measurement errors,
- unusual observations,
- unexpected trends.

---

# 0.135 Practice Exercise

Create a NumPy array containing

```python
hardness = np.array([120, 145, 132, 158, 149])
```

Write Python code that

1. creates a line plot,
2. adds the title **"Material Hardness"**,
3. labels the x-axis as **Sample Number**,
4. labels the y-axis as **Hardness (HV)**,
5. displays the figure.

Try writing the program without referring to the previous examples.

---

In the next section, we will learn how to create **scatter plots, histograms, and bar charts**, the three most commonly used visualization techniques in data science and materials informatics.


# 0.136 Scatter Plots, Histograms, and Bar Charts

The line plot introduced in the previous section is useful for showing changes in data over an ordered sequence.

However, in machine learning and materials science, three other types of graphs are used much more frequently:

1. **Scatter plots** – to study relationships between two variables.
2. **Histograms** – to understand how data are distributed.
3. **Bar charts** – to compare values across different categories.

Mastering these three plots will allow you to perform basic exploratory data analysis (EDA) on almost any dataset.

---

# 0.137 Scatter Plot

A **scatter plot** displays individual data points on a graph.

Unlike a line plot, the points are **not connected by lines**.

Scatter plots are extremely important because machine learning often tries to discover relationships between variables.

For example, we may ask

- Does hardness increase as density increases?
- Is bandgap related to atomic radius?
- Does elastic modulus depend on crystal density?

A scatter plot helps answer these questions visually.

---

# 0.138 Creating Your First Scatter Plot

Suppose we have measured the density and hardness of five materials.

```python
import matplotlib.pyplot as plt
import numpy as np

density = np.array([2.70, 4.51, 7.85, 8.96, 19.30])

hardness = np.array([167, 349, 250, 369, 400])

plt.scatter(density, hardness)

plt.show()
```

---

## Line-by-Line Explanation

### Line 1

```python
import matplotlib.pyplot as plt
```

Imports the plotting library.

---

### Line 2

```python
import numpy as np
```

Imports NumPy.

---

### Lines 4–5

```python
density = np.array([...])

hardness = np.array([...])
```

Create two arrays.

The first array contains x-axis values.

The second array contains y-axis values.

Both arrays must contain the same number of elements.

Each pair of values represents one material.

---

### Line 7

```python
plt.scatter(density, hardness)
```

Creates a scatter plot.

The first argument specifies the horizontal axis.

The second argument specifies the vertical axis.

Matplotlib places one point for every pair of values.

For example,

```text
Density = 2.70

Hardness = 167
```

becomes one point.

---

### Line 9

```python
plt.show()
```

Displays the figure.

---

# 0.139 Interpreting a Scatter Plot

Suppose the graph shows that

- as density increases,
- hardness also increases.

This suggests a **positive relationship**.

If hardness decreases while density increases,

the graph indicates a **negative relationship**.

If the points appear randomly scattered,

there may be little or no relationship between the variables.

One of the first tasks in machine learning is to determine whether useful relationships exist among features.

Scatter plots provide an intuitive way to begin this investigation.

---

# 0.140 Histogram

A histogram answers a different question.

Instead of comparing two variables,

it asks

> **How are the values distributed?**

For example,

- Are most materials low-density?
- Are there only a few high-density materials?
- Is the distribution symmetric?
- Are there unusual extreme values?

Histograms help answer these questions.

---

# 0.141 Creating a Histogram

```python
import matplotlib.pyplot as plt
import numpy as np

density = np.array([2.70, 4.51, 7.85, 8.96, 19.30])

plt.hist(density)

plt.show()
```

---

## Line-by-Line Explanation

### Line 7

```python
plt.hist(density)
```

`hist()`

creates a histogram.

Instead of plotting each value individually,

Matplotlib groups nearby values into intervals called **bins**.

The height of each bar represents the number of observations within that interval.

---

### Line 9

```python
plt.show()
```

Displays the histogram.

---

# 0.142 What Are Bins?

Suppose our density values are

```text
2.70

4.51

7.85

8.96

19.30
```

A histogram might divide them into intervals such as

```text
0–5

5–10

10–15

15–20
```

The graph counts how many values fall into each interval.

This makes it easy to understand the overall distribution of the data.

---

# 0.143 Choosing the Number of Bins

The number of bins can be controlled using the `bins` parameter.

```python
plt.hist(density, bins=4)

plt.show()
```

---

## Line-by-Line Explanation

### Line 1

```python
plt.hist(density, bins=4)
```

The argument

```python
bins=4
```

tells Matplotlib to divide the data into four intervals.

Changing the number of bins changes the appearance of the histogram.

Choosing an appropriate number of bins is important for interpreting the data correctly.

---

# 0.144 Bar Chart

A bar chart compares values belonging to different categories.

For example,

we may wish to compare

- hardness of different materials,
- electrical conductivity of several alloys,
- average grain size of different processing methods.

Unlike a histogram,

the bars represent **categories**, not numerical intervals.

---

# 0.145 Creating a Bar Chart

```python
import matplotlib.pyplot as plt

materials = ["Steel", "Aluminum", "Copper", "Titanium"]

hardness = [250, 167, 369, 349]

plt.bar(materials, hardness)

plt.show()
```

---

## Line-by-Line Explanation

### Line 1

```python
import matplotlib.pyplot as plt
```

Imports Matplotlib.

---

### Lines 3–4

```python
materials = [...]

hardness = [...]
```

Create two Python lists.

The first list contains category names.

The second list contains numerical values.

Each material corresponds to one hardness value.

---

### Line 6

```python
plt.bar(materials, hardness)
```

Creates a vertical bar chart.

Each material appears on the x-axis.

The height of each bar represents its hardness.

---

### Line 8

```python
plt.show()
```

Displays the graph.

---

# 0.146 Adding Titles and Axis Labels

Professional scientific figures should always include descriptive labels.

```python
plt.bar(materials, hardness)

plt.title("Hardness of Different Materials")

plt.xlabel("Material")

plt.ylabel("Hardness (HV)")

plt.show()
```

This makes the figure easy to understand even without reading the surrounding text.

---

# 0.147 Which Plot Should You Use?

Different graphs answer different questions.

| Plot | Best Used For |
|-------|---------------|
| Line Plot | Data changing with sequence or time |
| Scatter Plot | Relationship between two numerical variables |
| Histogram | Distribution of a single variable |
| Bar Chart | Comparing categories |

Choosing the correct visualization is an important part of exploratory data analysis.

---

# 0.148 Why Visualization Comes Before Machine Learning

Experienced data scientists rarely train a model immediately after loading data.

Instead, they first create several visualizations.

These graphs help them identify

- unusual values (outliers),
- skewed distributions,
- unexpected trends,
- possible relationships between variables,
- errors in data collection.

A few minutes spent visualizing data can prevent hours of debugging and improve the quality of the final machine learning model.

---

# 0.149 Practice Exercise

Create the following dataset.

```python
materials = ["Steel", "Copper", "Aluminum", "Titanium"]

density = [7.85, 8.96, 2.70, 4.51]

hardness = [250, 369, 167, 349]
```

Write Python code to

1. create a scatter plot of density versus hardness,
2. create a histogram of density values using five bins,
3. create a bar chart showing the hardness of each material,
4. add appropriate titles and axis labels to each figure.

Before running your code, predict what each graph will look like.

Developing this habit strengthens both your programming skills and your ability to interpret data.

---


The next section concludes **Chapter 0** by introducing a **complete mini data analysis workflow**, combining NumPy, Pandas, and Matplotlib to inspect, clean, summarize, visualize, and prepare a dataset for machine learning. This will serve as your first end-to-end Python data analysis project.


# 0.150 Mini Project: Your First Complete Data Analysis Workflow

Congratulations!

By reaching this point, you have learned the fundamental Python tools used by data scientists:

- Python programming
- NumPy
- Pandas
- Matplotlib

However, learning these tools separately is not enough.

In real machine learning projects, they are used **together**.

This mini project demonstrates how a professional data scientist approaches a new dataset.

The goal is **not** to build a machine learning model yet.

Instead, you will learn how to

1. load a dataset,
2. inspect it,
3. clean it,
4. calculate useful statistics,
5. visualize it,
6. prepare it for machine learning.

This workflow is followed in almost every machine learning project.

---

# 0.151 The Dataset

Assume we have a CSV file named

```text
materials.csv
```

containing

| Material | Density | Bandgap | Hardness |
|-----------|---------:|---------:|---------:|
| Steel | 7.85 | 0.00 | 250 |
| Aluminum | 2.70 | 0.00 | 167 |
| Silicon | 2.33 | 1.12 | 1150 |
| Copper | 8.96 | 0.00 | 369 |
| Titanium | 4.51 | 0.00 | 349 |

This dataset is intentionally small so that every step is easy to understand.

Later chapters will use much larger datasets.

---

# 0.152 Step 1 — Import the Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

## Line-by-Line Explanation

### Line 1

```python
import pandas as pd
```

Imports Pandas for handling tabular data.

---

### Line 2

```python
import numpy as np
```

Imports NumPy for numerical computations.

---

### Line 3

```python
import matplotlib.pyplot as plt
```

Imports Matplotlib for data visualization.

These three libraries form the foundation of almost every Python machine learning project.

---

# 0.153 Step 2 — Load the Dataset

```python
df = pd.read_csv("materials.csv")
```

This command reads the CSV file and stores it as a Pandas DataFrame.

From this point onward, all operations are performed on `df`.

---

# 0.154 Step 3 — Inspect the Dataset

```python
print(df.head())

print(df.shape)

print(df.columns)

df.info()
```

---

## Why These Commands?

Each command answers a different question.

`head()`

> What do the first few rows look like?

`shape`

> How many rows and columns are present?

`columns`

> What features are available?

`info()`

> Are there missing values? What data types are used?

Professional data scientists almost always execute these commands immediately after loading a new dataset.

---

# 0.155 Step 4 — Calculate Summary Statistics

Suppose we want to analyze the Density column.

```python
print(df["Density"].mean())

print(df["Density"].max())

print(df["Density"].min())

print(df["Density"].std())
```

---

## Line-by-Line Explanation

### Line 1

```python
df["Density"].mean()
```

Calculates the average density.

---

### Line 3

```python
df["Density"].max()
```

Finds the largest density.

---

### Line 5

```python
df["Density"].min()
```

Finds the smallest density.

---

### Line 7

```python
df["Density"].std()
```

Calculates the standard deviation, indicating how much the density values vary.

These statistics provide a quick numerical summary of the dataset.

---

# 0.156 Step 5 — Check for Missing Values

```python
print(df.isna().sum())
```

This command counts the number of missing values in each column.

If missing values are present, we must decide how to handle them before training a machine learning model.

Ignoring missing data can lead to errors or unreliable predictions.

---

# 0.157 Step 6 — Visualize the Data

A histogram helps us understand the distribution of density values.

```python
plt.hist(df["Density"])

plt.title("Distribution of Density")

plt.xlabel("Density (g/cm³)")

plt.ylabel("Frequency")

plt.show()
```

This figure allows us to identify

- clusters,
- skewness,
- possible outliers.

Visualization is often the fastest way to discover unexpected patterns.

---

# 0.158 Step 7 — Study Relationships Between Features

Suppose we want to know whether harder materials also tend to have higher density.

```python
plt.scatter(df["Density"], df["Hardness"])

plt.title("Density vs Hardness")

plt.xlabel("Density (g/cm³)")

plt.ylabel("Hardness (HV)")

plt.show()
```

A scatter plot makes potential relationships visible.

If the points show a clear trend,

a machine learning model may be able to learn that relationship.

---

# 0.159 Step 8 — Select Features for Machine Learning

Machine learning models require two types of data:

- **Features (X)** – the information used to make predictions.
- **Target (y)** – the value we want to predict.

Suppose we want to predict hardness using density and bandgap.

```python
X = df[["Density", "Bandgap"]]

y = df["Hardness"]
```

---

## Line-by-Line Explanation

### Line 1

```python
X = df[["Density", "Bandgap"]]
```

Selects two feature columns.

The variable name `X` is the standard convention in machine learning for the feature matrix.

---

### Line 3

```python
y = df["Hardness"]
```

Selects the target variable.

The variable name `y` is the standard convention for the values that the model will learn to predict.

Throughout the rest of this book, nearly every machine learning model will use variables named `X` and `y`.

---

# 0.160 The Complete Workflow

The entire process can be summarized as follows.

```text
CSV Dataset
      │
      ▼
Read with Pandas
      │
      ▼
Inspect Data
      │
      ▼
Handle Missing Values
      │
      ▼
Calculate Statistics
      │
      ▼
Visualize Data
      │
      ▼
Select Features (X)
Select Target (y)
      │
      ▼
Ready for Machine Learning
```

Notice that **machine learning is only the final step**.

Most of the work involves preparing and understanding the data.

This is why data preprocessing is often said to consume the majority of a data scientist's time.

---

# 0.161 Common Beginner Mistakes

When starting out, many learners make the following mistakes:

1. **Skipping data inspection**
   - Always check `head()`, `shape`, and `info()` before analysis.

2. **Ignoring missing values**
   - Models generally cannot work with incomplete data.

3. **Using the wrong target column**
   - Carefully identify the variable you want to predict.

4. **Not visualizing the data**
   - Simple graphs often reveal problems that are difficult to detect from tables.

5. **Training a model without understanding the dataset**
   - Always explore the data before fitting a model.

Avoiding these mistakes will save considerable time during machine learning projects.

---

# 0.162 Chapter 0 Summary

You have now completed the prerequisite programming chapter.

You learned how to

- write Python programs,
- create variables,
- work with numbers and strings,
- use lists, tuples, dictionaries, and sets,
- write conditional statements,
- use loops,
- define functions,
- import Python modules,
- perform numerical computations with NumPy,
- manipulate tabular data with Pandas,
- create scientific plots with Matplotlib,
- inspect and clean datasets,
- select features and targets,
- prepare data for machine learning.

These skills form the foundation for every chapter that follows.

---

# 0.163 What's Next?

Beginning with **Chapter 1**, we leave general Python programming behind and enter the world of **Machine Learning for Materials Informatics**.

You will learn

- what machine learning really is,
- how computers learn from data,
- the difference between supervised and unsupervised learning,
- the complete machine learning workflow,
- the terminology used by researchers,
- and how machine learning is transforming materials discovery.

From Chapter 1 onward, every concept will include

- detailed theoretical explanations,
- mathematical foundations,
- complete Python implementations,
- line-by-line code explanations,
- materials science examples,
- professional best practices.

By the end of this book, you will not only understand the theory behind modern machine learning algorithms but also be able to implement them confidently in Python and apply them to real materials informatics problems.

# 0.164 Setting Up Your Machine Learning Environment

Before we begin building machine learning models, we must prepare our programming environment.

Writing Python code is only one part of machine learning.

A data scientist also needs to

- install libraries,
- organize projects,
- work with notebooks,
- manage datasets,
- understand where code is stored.

This section introduces the tools that we will use throughout the remainder of this book.

---

# 0.165 Why Do We Need Additional Libraries?

Python itself contains only basic functionality.

Machine learning requires specialized libraries.

Some of the most important libraries are

| Library | Purpose |
|----------|---------|
| NumPy | Numerical computation |
| Pandas | Data analysis |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning algorithms |
| XGBoost | Gradient boosting |
| LightGBM | Fast gradient boosting |
| CatBoost | Gradient boosting for categorical data |
| PyTorch | Deep learning |
| TensorFlow | Deep learning |

Throughout this book, we will gradually learn these libraries.

---

# 0.166 What Is Scikit-learn?

Scikit-learn is the world's most popular machine learning library.

Instead of implementing algorithms from scratch,

it provides highly optimized implementations of

- Linear Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- K-Nearest Neighbors
- Clustering algorithms
- Dimensionality reduction
- Model evaluation tools

Almost every beginner learns machine learning using Scikit-learn.

---

# 0.167 Installing Packages with pip

Python packages are installed using a program called **pip**.

For example,

```bash
pip install scikit-learn
```

downloads and installs Scikit-learn.

Similarly,

```bash
pip install xgboost
```

installs XGBoost.

If a package is already installed,

pip simply reports that it is already available.

---

# 0.168 What Is Jupyter Notebook?

Throughout this book, most examples will run inside **Jupyter Notebook**.

A Jupyter Notebook is an interactive programming environment.

Unlike a normal Python file,

a notebook is divided into **cells**.

Each cell can contain

- Python code,
- equations,
- explanations,
- figures,
- tables.

This makes Jupyter ideal for scientific computing and machine learning research.

---

# 0.169 Code Cells and Markdown Cells

A notebook mainly contains two kinds of cells.

## Code Cell

A code cell executes Python.

Example

```python
x = 10

print(x)
```

Running the cell immediately displays

```text
10
```

---

## Markdown Cell

A Markdown cell contains formatted text.

It is used for

- explanations,
- headings,
- equations,
- notes.

Most research notebooks contain a mixture of code cells and Markdown cells.

---

# 0.170 Organizing Machine Learning Projects

A professional project is usually organized like this.

```text
Materials_Project/

│

├── data/

│   └── materials.csv

│

├── notebooks/

│   └── analysis.ipynb

│

├── models/

│

├── figures/

│

└── README.md
```

Keeping files organized makes projects easier to understand and maintain.

---

# 0.171 The Workflow We Will Follow

Every chapter from this point onward will follow the same structure.

```text
Import Libraries

↓

Load Dataset

↓

Explore Data

↓

Preprocess Data

↓

Train Model

↓

Evaluate Model

↓

Interpret Results

↓

Improve Model
```

By following the same workflow repeatedly, you will gradually develop professional machine learning habits.

---

You are now fully prepared to begin **Chapter 1: Introduction to Machine Learning**.

# 0.172 Introduction to Scikit-learn

So far, we have learned how to

- write Python programs,
- manipulate arrays using NumPy,
- organize datasets with Pandas,
- visualize data using Matplotlib.

Now we are ready to learn the library that powers most classical machine learning projects:

**Scikit-learn**.

Throughout this book, almost every machine learning model will be built using Scikit-learn.

Understanding its basic structure now will make the remaining chapters much easier.

---

# 0.173 What Is Scikit-learn?

Scikit-learn is an open-source Python library for machine learning.

Instead of writing complex algorithms from scratch, Scikit-learn provides efficient and well-tested implementations.

For example, a single line of code can create

- a Linear Regression model,
- a Decision Tree,
- a Random Forest,
- an XGBoost-like Gradient Boosting model,
- a Support Vector Machine,
- a K-Nearest Neighbors classifier.

Scikit-learn is one of the most widely used machine learning libraries in academia and industry.

---

# 0.174 The General Structure of Every Scikit-learn Program

Although different machine learning algorithms work differently, almost every Scikit-learn program follows the same basic structure.

```python
from sklearn.some_module import SomeModel

model = SomeModel()

model.fit(X, y)

predictions = model.predict(X)
```

At this stage, do not worry about understanding every detail.

Our goal is to become familiar with the overall workflow.

---

# 0.175 Understanding the Import Statement

```python
from sklearn.linear_model import LinearRegression
```

## Line-by-Line Explanation

### Line 1

```python
from sklearn.linear_model import LinearRegression
```

Let's examine every part.

`from`

tells Python that we want to import something from a library.

`sklearn`

is the package name for Scikit-learn.

`linear_model`

is the module containing linear models.

`LinearRegression`

is the specific class that implements linear regression.

Notice that we are importing only one class instead of the entire library.

This makes the code cleaner and slightly more efficient.

---

# 0.176 Creating a Machine Learning Model

```python
model = LinearRegression()
```

## Line-by-Line Explanation

### Line 1

```python
model = LinearRegression()
```

This line creates a **machine learning model object**.

Think of it as an empty model.

At this stage,

the model has not learned anything.

It contains no knowledge about your dataset.

It is simply ready to be trained.

---

# 0.177 Training a Model

```python
model.fit(X, y)
```

This is one of the most important commands in machine learning.

`fit()`

means

> "Learn from the data."

Here,

`X`

contains the input features.

For example,

- Density,
- Bandgap,
- Atomic Radius,
- Melting Point.

`y`

contains the target variable.

For example,

- Hardness,
- Yield Strength,
- Elastic Modulus.

During training,

the model examines the relationship between `X` and `y`.

After `fit()` finishes,

the model has learned patterns from the training data.

---

# 0.178 Making Predictions

Once the model has learned,

we can ask it to make predictions.

```python
predictions = model.predict(X)
```

## Line-by-Line Explanation

### Line 1

```python
predictions = model.predict(X)
```

`predict()`

asks the trained model to estimate the target values corresponding to the input features.

The predicted values are stored in

```python
predictions
```

Later chapters will explain exactly how different algorithms generate these predictions.

---

# 0.179 The Universal Machine Learning Pattern

Almost every Scikit-learn algorithm follows the same sequence.

```text
Import Algorithm

↓

Create Model

↓

Train Model

↓

Make Predictions

↓

Evaluate Performance
```

Whether you use

- Linear Regression,
- Decision Trees,
- Random Forests,
- Support Vector Machines,
- Gradient Boosting,

the workflow remains remarkably similar.

This consistency is one of Scikit-learn's greatest strengths.

---

# 0.180 Why Learn This Now?

From Chapter 1 onward, you will repeatedly encounter code such as

```python
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Because you have already learned the meaning of

- `fit()`,
- `predict()`,
- `X`,
- `y`,
- `model`,

you will be able to focus on understanding the machine learning algorithms themselves rather than the syntax.

---

**Chapter 0 is now complete.**

You now possess all the Python, NumPy, Pandas, Matplotlib, and Scikit-learn fundamentals required for the rest of the book.

The next chapter begins your formal study of **Machine Learning**, where you will learn what it means for a computer to "learn" from data before implementing your first real machine learning model.

