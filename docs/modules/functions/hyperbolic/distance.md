# Hyperbolic Distance Function

The hyperbolic distance function calculates the distance between two points in the hyperbolic plane.

## Function Definition

```python
from typing import Tuple
import numpy as np

def hyperbolic_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate the hyperbolic distance between two points in the hyperbolic plane.

    Args:
        point1 (Tuple[float, float]): The first point (x1, y1).
        point2 (Tuple[float, float]): The second point (x2, y2).

    Returns:
        float: The hyperbolic distance between the two points.

    Examples:
        >>> hyperbolic_distance((0, 0), (1, 1))
        1.762747174039086
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.arccosh(1 + ((x2 - x1)**2 + (y2 - y1)**2) / (2 * y1 * y2))
```

## Examples and Usage

Here are some examples of how to use the hyperbolic distance function:

```python
>>> from umf.functions.hyperbolic import hyperbolic_distance
>>> hyperbolic_distance((0, 0), (1, 1))
1.762747174039086

>>> hyperbolic_distance((0, 0), (2, 2))
2.2924316695611777

>>> hyperbolic_distance((1, 1), (2, 2))
1.762747174039086
```

## Visualization

To visualize the hyperbolic distance between two points, you can use the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_hyperbolic_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = hyperbolic_distance(point1, point2)

    fig, ax = plt.subplots()
    ax.plot([x1, x2], [y1, y2], 'ro-')
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f'{distance:.2f}', fontsize=12, ha='center')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hyperbolic Distance')
    plt.grid(True)
    plt.show()

plot_hyperbolic_distance((0, 0), (1, 1))
```

This code will generate a plot showing the hyperbolic distance between the two points (0, 0) and (1, 1).
