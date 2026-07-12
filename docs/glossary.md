# Glossary (Plain English)

This glossary explains common `autotokamak` and fusion-modeling terms in beginner-friendly language.

## A

### Agent
In this project, an "agent" is an AI assistant that plans tasks and runs tools or code steps to complete them.

### AutoML
Automatic model search. Instead of manually trying many model settings, AutoML tests options and picks what works best.

## B

### Boundary
The edge of the plasma shape in a cross-section plot. It is the outline the solver uses to know where the plasma region ends.

## D

### Dataset
A collection of examples used to train or test machine-learning models. Here, each example includes plasma inputs and the resulting flux field.

### Discretization
Turning a smooth physical problem into many small numerical pieces so a computer can solve it.

## E

### EQDSK
A common file format for tokamak equilibrium data. Think of it as a saved "snapshot" of equilibrium inputs/results.

### Equilibrium (MHD equilibrium)
A stable balance of forces in the plasma, where pressure and magnetic forces are consistent with each other.

## F

### Fixed-boundary
A solve mode where the plasma boundary is provided ahead of time and kept fixed during the solve.

## G

### Grad–Shafranov equation (GS equation)
The main physics equation used to compute axisymmetric tokamak equilibrium in 2D cross-section.

## H

### HDF5 (`.h5`)
A file format for storing large scientific arrays and metadata in one file.

### Hyperparameters
Settings you choose before training a model (for example learning rate, number of layers, or regularization strength).

## I

### Interpolation
Estimating values between known points. In this repo, it is used to place solver outputs onto a common grid.

### `Ip` (plasma current)
The total electric current flowing through the plasma column.

## L

### LCFS (Last Closed Flux Surface)
The outermost magnetic surface where field lines still close on themselves. In plain terms: it is the "main plasma boundary" used in many equilibrium setups.

## M

### Mesh
A network of many small cells (often triangles) that covers the plasma region so the solver can do numerical calculations.

### Model zoo
A small list of candidate ML models to compare (for example GP, kernel ridge, polynomial ridge, and MLP).

## O

### OFT (OpenFUSIONToolkit)
An open-source toolkit with fusion solvers. This repo uses OFT's TokaMaker solver for ground-truth equilibria.

### Optimization
Trying different parameter choices to improve a target score (for example lower prediction error).

## P

### Planner / Planning agent
The component that decides the sequence of steps to solve a task.

### PoC (Proof of Concept)
A first working version built to show an idea is feasible, not a final production system.

### Profiles (pressure/current profiles)
Functions that describe how quantities like pressure or current change across flux surfaces.

## S

### Solver
Software that numerically solves equations. Here, it usually means TokaMaker solving the Grad–Shafranov problem.

### Surrogate model
A fast ML model trained to imitate a slower physics solver.

## T

### TokaMaker
The equilibrium solver inside OFT used in this repo to produce reference (ground-truth) solutions.

## U

### URSA
An agent framework used here to run plan-and-execute workflows.

## W

### Workspace
A folder where an agent run writes generated scripts, configs, logs, and outputs.
