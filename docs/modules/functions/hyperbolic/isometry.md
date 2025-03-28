# Hyperbolic Isometry Function

The hyperbolic isometry function applies isometries (transformations that
preserve distances) in the hyperbolic plane.

## Function Definition

```python
from typing import Tuple
import numpy as np

def hyperbolic_isometry(point: Tuple[float, float], matrix: np.ndarray) -> np.ndarray:
    """Apply an isometry transformation to a point in the hyperbolic plane.

    Args:
        point (Tuple[float, float]): The point (x, y) to be transformed.
        matrix (np.ndarray): The 2x2 isometry matrix.

    Returns:
        np.ndarray: The transformed point (x', y').

    Examples:
        >>> hyperbolic_isometry((1, 1), np.array([[1, 1], [1, 1]]))
        array([2., 2.])
    """
    return np.dot(matrix, point)
```

## Examples and Usage

Here are some examples of how to use the hyperbolic isometry function:

```python
>>> from umf.functions.hyperbolic import hyperbolic_isometry
>>> hyperbolic_isometry((1, 1), np.array([[1, 1], [1, 1]]))
array([2., 2.])

>>> hyperbolic_isometry((2, 3), np.array([[0, 1], [1, 0]]))
array([3., 2.])

>>> hyperbolic_isometry((1, 2), np.array([[2, 0], [0, 2]]))
array([2., 4.])
```

## Visualization

To visualize the hyperbolic isometry transformation, you can use the following
code:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_hyperbolic_isometry(point, matrix):
    transformed_point = hyperbolic_isometry(point, matrix)

    fig, ax = plt.subplots()
    ax.plot(point[0], point[1], 'ro', label='Original Point')
    ax.plot(transformed_point[0], transformed_point[1], 'bo', label='Transformed Point')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hyperbolic Isometry')
    plt.grid(True)
    plt.show()

plot_hyperbolic_isometry((1, 1), np.array([[1, 1], [1, 1]]))
```

This code will generate a plot showing the original point and the transformed
point after applying the isometry transformation.
