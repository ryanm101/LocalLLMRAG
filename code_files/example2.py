class SimpleCalculator:
    """
    A simple calculator class that performs basic arithmetic operations.
    """

    def __init__(self):
        pass

    def multiply(self, x, y):
        """
        Multiplies two numbers.

        Args:
            x: The first number.
            y: The second number.

        Returns:
            The product of x and y.
        """
        return x * y

    def divide(self, x, y):
        """
        Divides two numbers.

        Args:
            x: The numerator.
            y: The denominator.

        Returns:
            The result of x divided by y.
        Raises:
            ValueError: If y is zero.
        """
        if y == 0:
            raise ValueError("Cannot divide by zero.")
        return x / y

if __name__ == "__main__":
    calculator = SimpleCalculator()
    product = calculator.multiply(8, 4)
    print(f"8 * 4 = {product}")

    try:
        quotient = calculator.divide(15, 3)
        print(f"15 / 3 = {quotient}")
        calculator.divide(10, 0) # This will raise a ValueError
    except ValueError as e:
        print(f"Error: {e}")