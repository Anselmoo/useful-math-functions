"""Exceptions for the umf package."""

from __future__ import annotations


class MissingXError(Exception):
    """Exception for missing input data.

    This exception is raised when input data '*x' has not been specified.

    Attributes:
        None
    """

    def __init__(self) -> None:
        """Initialize the exception."""
        super().__init__("Input data '*x' has to be specified.")


class OutOfDimensionError(Exception):
    """Exception for out of dimension input data.

    This exception is raised when input data '*x' has a dimension that is not
    supported by the function.
    """

    def __init__(self, *, function_name: str, dimension: int) -> None:
        """Initialize the exception.

        Args:
            function_name (str): Name of the function.
            dimension (int): Dimension of the function.
        """
        super().__init__(
            f"The class of '{function_name}' only accepts '{dimension}d' inputs.",
        )


class TooHighDimensionError(Exception):
    """Exception for too high dimension input data.

    This exception is raised when input data '*x' has a dimension that is higher
    than the function supports.

    Attributes:
        max_dimension (int): Maximum dimension of the function.
        current_dimension (int): Current dimension of the input data.
    """

    def __init__(self, *, max_dimension: int, current_dimension: int) -> None:
        """Initialize the exception.

        Args:
            max_dimension (int): Maximum dimension of the function.
            current_dimension (int): Current dimension of the input data.
        """
        super().__init__(
            f"The maximum dimension of the function is '{max_dimension}', but the "
            f"input data has a dimension of '{current_dimension}'.",
        )


class ExcessiveExponentError(Exception):
    """Exception raised when the exponent in an operation is excessively large."""

    def __init__(self, *, max_exponent: float, current_exponent: float) -> None:
        """Initialize the exception."""
        self.message = (
            f"(n={current_exponent}) is too large."
            f" The maximum exponent is {max_exponent}."
        )
        super().__init__(self.message)


class TooLowDimensionError(Exception):
    """Exception for too low dimension input data.

    This exception is raised when input data '*x' has a dimension that is lower
    than the function supports.

    Attributes:
        min_dimension (int): Minimum dimension of the function.
        current_dimension (int): Current dimension of the input data.
    """

    def __init__(self, *, min_dimension: int, current_dimension: int) -> None:
        """Initialize the exception.

        Args:
            min_dimension (int): Minimum dimension of the function.
            current_dimension (int): Current dimension of the input data.
        """
        super().__init__(
            f"The minimum dimension of the function is '{min_dimension}', but the "
            f"input data has a dimension of '{current_dimension}'.",
        )


class OutOfRangeError(Exception):
    """Exception raised when input values are out of range for a given function.

    Attributes:
        function_name (str): Name of the function.
        start_range (float): Start range of the function.
        end_range (float): End range of the function.
    """

    def __init__(
        self,
        *,
        function_name: str,
        start_range: float,
        end_range: float,
    ) -> None:
        """Initialize the exception.

        Args:
            function_name (str): Name of the function.
            start_range (float): Start range of the function.
            end_range (float): End range of the function.
        """
        super().__init__(
            f"The input values for '{function_name}' should be within the range of "
            f"{start_range} and {end_range}.",
        )


class TooSmallDimensionError(Exception):
    """Exception raised when a function is called with too few inputs.

    Attributes:
        function_name (str): Name of the function.
        dimension (int): Dimension of the function.
        len_x (int): Length of the input data.
    """

    def __init__(self, function_name: str, dimension: int, len_x: int) -> None:
        """Initialize the exception.

        Args:
            function_name (str): Name of the function.
            dimension (int): Dimension of the function.
            len_x (int): Length of the input data.
        """
        super().__init__(
            f"The '{function_name}' requires at least '{dimension}' inputs, "
            f"but only '{len_x}' were given.",
        )


class PlotAttributeError(Exception):
    """Exception for when an invalid plot attribute at is chosen.

    !!! note "Available plot attributes"

        This exception is raised when an invalid plot attribute is chosen.
        The following plot attributes error can be arrived at:

        - `type`: The chosen plot type is not available.
        - `style`: The chosen plot style is not available.

    Attributes:
        choose (str): The chosen plot type.
        modes (set[str]): The available plot types.
        error_type (str): The type of error.
    """

    def __init__(
        self,
        *,
        choose: str,
        modes: set[str],
        error_type: str = "type",
    ) -> None:
        """Initialize the exception.

        Args:
            choose (str): The chosen plot type.
            modes (set[str]): The available plot types.
            error_type (str, optional): The type of error. Defaults to "type".
        """
        modes = sorted(modes)
        _modes = "".join(f"'{mode}', " for mode in modes[:-1])
        _modes += f"or '{modes[-1]}'"

        error_msg: dict[str, str] = {
            "type": f"The chosen plot type '{choose}' is not available. "
            f"Please choose one of the following plot types: {_modes}.",
            "style": f"The chosen plot style '{choose}' is not available. "
            f"Please choose one of the following plot styles: {_modes}.",
        }
        super().__init__(error_msg[error_type])


class MatchLengthError(Exception):
    """Exception raised when the length of two objects do not match.

    Attributes:
        _object (str): The object that has a different length than the target.
        _target (str): The target object that has a different length than the object.
    """

    def __init__(self, _object: str, _target: str) -> None:
        """Initialize the exception.

        Args:
            _object (str): The object.
            _target (str): The target.
        """
        super().__init__(
            f"The length of the '{_object}' does not match the length of the "
            f"'{_target}'.",
        )


class NoCumulativeError(Exception):
    """Exception raised when the cumulative distribution function is not defined."""

    def __init__(self) -> None:
        """Initialize the exception."""
        super().__init__(
            "The cumulative distribution function is not defined for this "
            "distribution model.",
        )


class NotAPositiveNumberError(Exception):
    """Exception raised when a number is not positive."""

    def __init__(self, var_number: str, number: float) -> None:
        """Initialize the exception.

        Args:
            var_number (str): The variable name of the number.
            number (float): The number that is not positive.
        """
        super().__init__(f"The number '{number}' of '{var_number}' is not positive.")


class NotLargerThanZeroError(Exception):
    """Exception raised when a number is not larger than zero."""

    def __init__(self, var_number: str, number: float) -> None:
        """Initialize the exception.

        Args:
            var_number (str): The variable name of the number.
            number (float): The number that is not larger than zero.
        """
        super().__init__(
            f"The number '{number}' of variable "
            f"'{var_number}' is not larger than zero.",
        )


class NotLargerThanAnyError(Exception):
    """Exception raised when a number is not larger than zero."""

    def __init__(self, var_number: str, number: float, minimum: float) -> None:
        """Initialize the exception.

        Args:
            var_number (str): The variable name of the number.
            number (float): The number that is not larger than zero.
            minimum (float): The minimum value.
        """
        super().__init__(
            f"The number '{number}' of variable '{var_number}' is not larger than "
            f"the required minimum '{minimum}'.",
        )


class NotSmallerThanAnyError(Exception):
    """Exception raised when a number is not smaller than zero."""

    def __init__(self, var_number: str, number: float, maximum: float) -> None:
        """Initialize the exception.

        Args:
            var_number (str): The variable name of the number.
            number (float): The number that is not smaller than zero.
            maximum (float): The maximum value.
        """
        super().__init__(
            f"The number '{number}' of variable '{var_number}' is not smaller than "
            f"the required maximum '{maximum}'.",
        )


class NotInRangesError(Exception):
    """Exception raised when a number is not in the required ranges."""

    def __init__(
        self,
        var_number: str,
        number: float,
        ranges: tuple[float, float],
    ) -> None:
        """Initialize the exception.

        Args:
            var_number (str): The variable name of the number.
            number (float): The number that is not in the required ranges.
            ranges (tuple[float, float]): The required ranges.
        """
        super().__init__(
            f"The number '{number}' of variable '{var_number}' is not in the required "
            f"ranges '{ranges}'.",
        )


class TimeFormatError(Exception):
    """Exception raised when the time format is not recognized."""

    def __init__(
        self,
        time_format: str,
        valid_formats: list[str] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            time_format (str): The time format that is not recognized.
            valid_formats (list[str], optional): A list of valid time formats. Defaults
                to None.
        """
        self.time_format = time_format
        self.valid_formats = valid_formats or ["seconds", "minutes", "hours", "days"]
        super().__init__(self._generate_message())

    def _generate_message(self) -> str:
        """Generate the exception message."""
        valid_formats_str = ", ".join(self.valid_formats)
        return (
            f"The time format '{self.time_format}' is not recognized. "
            f"Valid formats are: {valid_formats_str}."
        )

    def __str__(self) -> str:
        """Return the string representation of the exception."""
        return self._generate_message()
