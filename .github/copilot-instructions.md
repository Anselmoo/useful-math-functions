# Copilot Instructions

This repository contains mathematical functions implemented in Python. When contributing or modifying code, please adhere to the following guidelines:

## Code Structure

- Follow the initial structure with `api.py` for interface definitions and `functions.py` for implementations
- Each function should be implemented in `functions.py` and exposed through `api.py`

## Coding Standards

1. Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
2. Use type annotations (Python 3.10+):

   ```python
   def function_name(x: float, y: int) -> float:
        """Function description.

       Note:
            - Description of the function
            - Mathematical notation using LaTeX where applicable
            - Example: $f(x) = x^2 + y^2$ or

               $$
               f(x) = \sum_{i=0}^{n} a_i x^i
               $$


       Args:
            x (float): Description of x
            y (int): Description of y

       Returns:
            float: Description of return value
        """
        pass
   ```

## Documentation

1. Include docstrings with:
   - Brief description
   - Parameters with types
   - Return value with type
   - Mathematical notation using LaTeX where applicable

Example:

```python
def calculate_polynomial(x: float, coefficients: list[float]) -> float:
     """Evaluates a polynomial with given coefficients.

     Args:
          x: Value to evaluate at
          coefficients: List of coefficients [a₀, a₁, ..., aₙ]

     Returns:
          float: Result of polynomial: ∑(aᵢ * x^i)

     Note:
          LaTeX: f(x) = \sum_{i=0}^{n} a_i x^i
     """
     pass
```
