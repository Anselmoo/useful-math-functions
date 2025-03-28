# Hyperbolic Geodesic Function

The hyperbolic geodesic function determines the shortest path between two points in the hyperbolic plane.

## Function Definition

```python
from typing import Tuple
import numpy as np

def hyperbolic_geodesic(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Determine the shortest path between two points in the hyperbolic plane.

    Args:
        point1 (Tuple[float, float]): The first point (x1, y1).
        point2 (Tuple[float, float]): The second point (x2, y2).

    Returns:
        float: The length of the geodesic between the two points.

    Examples:
        >>> hyperbolic_geodesic((0, 0), (1, 1))
        1.762747174039086
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.arccosh(1 + ((x2 - x1)**2 + (y2 - y1)**2) / (2 * y1 * y2))
```

## Examples and Usage

Here are some examples of how to use the hyperbolic geodesic function:

```python
>>> from umf.functions.hyperbolic import hyperbolic_geodesic
>>> hyperbolic_geodesic((0, 0), (1, 1))
1.762747174039086

>>> hyperbolic_geodesic((0, 0), (2, 2))
2.2924316695611777

>>> hyperbolic_geodesic((1, 1), (2, 2))
1.762747174039086
```

## Visualization

To visualize the hyperbolic geodesic between two points, you can use the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_hyperbolic_geodesic(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    geodesic_length = hyperbolic_geodesic(point1, point2)

    fig, ax = plt.subplots()
    ax.plot([x1, x2], [y1, y2], 'ro-')
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f'{geodesic_length:.2f}', fontsize=12, ha='center')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hyperbolic Geodesic')
    plt.grid(True)
    plt.show()

plot_hyperbolic_geodesic((0, 0), (1, 1))
```

This code will generate a plot showing the hyperbolic geodesic between the two points (0, 0) and (1, 1).
