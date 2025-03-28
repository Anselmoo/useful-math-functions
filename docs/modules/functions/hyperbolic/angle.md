# Hyperbolic Angle Function

The hyperbolic angle function computes the angle between two vectors in the
hyperbolic plane.

## Function Definition

```python
from typing import Tuple
import numpy as np

def hyperbolic_angle(vector1: Tuple[float, float], vector2: Tuple[float, float]) -> float:
    """Compute the angle between two vectors in the hyperbolic plane.

    Args:
        vector1 (Tuple[float, float]): The first vector (x1, y1).
        vector2 (Tuple[float, float]): The second vector (x2, y2).

    Returns:
        float: The angle between the two vectors in radians.

    Examples:
        >>> hyperbolic_angle((1, 0), (0, 1))
        1.5707963267948966
    """
    x1, y1 = vector1
    x2, y2 = vector2
    dot_product = x1 * x2 + y1 * y2
    norm1 = np.sqrt(x1**2 + y1**2)
    norm2 = np.sqrt(x2**2 + y2**2)
    return np.arccos(dot_product / (norm1 * norm2))
```

## Examples and Usage

Here are some examples of how to use the hyperbolic angle function:

```python
>>> from umf.functions.hyperbolic import hyperbolic_angle
>>> hyperbolic_angle((1, 0), (0, 1))
1.5707963267948966

>>> hyperbolic_angle((1, 1), (1, 1))
0.0

>>> hyperbolic_angle((1, 0), (1, 1))
0.7853981633974483
```

## Visualization

To visualize the hyperbolic angle between two vectors, you can use the following
code:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_hyperbolic_angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    angle = hyperbolic_angle(vector1, vector2)

    fig, ax = plt.subplots()
    ax.plot([0, x1], [0, y1], 'ro-')
    ax.plot([0, x2], [0, y2], 'bo-')
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f'{angle:.2f}', fontsize=12, ha='center')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hyperbolic Angle')
    plt.grid(True)
    plt.show()

plot_hyperbolic_angle((1, 0), (0, 1))
```

This code will generate a plot showing the hyperbolic angle between the two
vectors (1, 0) and (0, 1).
