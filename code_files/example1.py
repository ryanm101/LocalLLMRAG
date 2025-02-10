def calculate_sum(a, b):
    """
    This function calculates the sum of two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    """
    return a + b

def greet(name):
    """
    This function greets a person by name.

    Args:
        name: The name of the person to greet.

    Returns:
        A greeting message string.
    """
    return f"Hello, {name}!"

# Example usage
if __name__ == "__main__":
    num1 = 10
    num2 = 5
    sum_result = calculate_sum(num1, num2)
    print(f"The sum of {num1} and {num2} is: {sum_result}")

    person_name = "Alice"
    greeting_message = greet(person_name)
    print(greeting_message)