# LTI-Systems-for-Sympy

A package for linear, time invariant control systems for symbolic python. This is very new and has only basic functionality as transforming StateSpaceModels and TransferFunctionModels into one another and evaluating the systems symbolicaly and numericaly. Furthermore the `utils` module provides some basic tools the `models`module uses but which can be useful for working linear systems in general. (such as Lyapunov Equation solver, numerical inverse Laplace Transform, matrix vectorization and tools for matrix valued Polynomials).

## Installation

There is a python installer for this package. You can download the *.zip* file in the `dist` folder or clone the whole repository using
```
git clone https://github.com/m3zz0m1x/LTI-Systems-for-Sympy.git
```
After extracting, cd in the direcory and run
```
python setup.py install
```

## Usage

Include the modules in your program using:
```python
from lti_systems import *
mod = models
utl = utils
```

### Creating models
You can create a 'StateSpaceModel' object using four matrices (state space representation) or a or 'TransferFunctionModel' using a transfer matrix:
```python
from sympy import *     # needed for matrix class

var('a:d')
A, B, C, D = Matrix([a]), Matrix([b]), Matrix([c]), Matrix([d])

var('s, num, denom')
T = Matrix([num /denom], s)

ssm = mod.StateSpaceModel([A, B, C, D])
tfm = mod.TransferFunctionModel(T)
```
You can also create the models from one another:
```python
ssm2 = mod.StateSpaceModel(tfm)
tfm2 = mod.TransferFunctionModel(ssm, s)
```

### Evaluating models
Models are always evaluated using the `evaluate` method. It is done numercialy or symbolicaly, depending on the parameters of the call:
```python
# symbolic evaluation:
var('t')
u = Matrix([Heaviside(t)])
x0 = zeros(1, 1)
t0 = 0
y = ssm.evaluate(u, x0, t, t0)

# numerical evaluation
import numpy as np
y_list = ssm.evaluate(u, x0, (t, np.arange(0, 10, 0.1)), t0)
```
Note that the only difference between the calls is that in the second one, t is substituted by `(t, np.arange(0, 10, 0.1))`; a tuple of the symbol t and a list of times to evaluate for. 
The first call will return a matrix valued expression in terms of t.
The second call will return a list of matrix valued outputs, matching the time input.

For details on syntax and additional methods, see the docstings of each class and its methods.

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D
