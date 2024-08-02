import random
import math
import pickle


class NeuralNetNode:
    def __init__(
        self, bias: float, weights: list[float], activation: callable = lambda x: x
    ):
        """
        Initializes a new instance of the Node class with the specified bias, weights, and activation function.

        Args:
            bias (float): The bias value for the node.
            weights (list[float]): The weights for the inputs to the node.
            activation (callable, optional): The activation function to apply to the weighted input. Defaults to lambda x: x.
        """
        self.bias = bias
        self.weights = weights
        self.activation = activation

    def calculate(self, inputs: list[float]) -> float:
        """
        Calculate the output of the node based on the given inputs.

        Args:
            inputs (list[float]): A list of input values.

        Returns:
            float: The output of the node.

        Raises:
            TypeError: If the inputs are not of type list.
            ValueError: If the inputs list is empty.
        """
        if not isinstance(inputs, list):
            raise TypeError("Inputs must be of type list.")
        if len(inputs) == 0:
            raise ValueError("Inputs list cannot be empty.")

        self.inputs = inputs  # Store inputs for backpropagation
        weighted_sum = sum(
            weight * input_value for weight, input_value in zip(self.weights, inputs)
        )
        self.output = self.activation(
            weighted_sum + self.bias
        )  # Store output for backpropagation
        return self.output

    def update_weights(self, delta: float, learning_rate: float):
        """
        Update the weights and bias of the node based on the delta value.

        Args:
            delta (float): The delta value calculated during backpropagation.
            learning_rate (float): The learning rate for weight updates.
        """
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * delta * self.inputs[i]
        self.bias += learning_rate * delta


class NeuralNetLayer:
    def __init__(self, nodes: list[NeuralNetNode]):
        """
        Initializes a new instance of the NeuralNetLayer class with the specified nodes.

        Args:
            nodes (list[NeuralNetNode]): A list of NeuralNetNode objects representing the nodes in the layer.
        """
        self.nodes = nodes

    def calculate(self, inputs: list[float]) -> list[float]:
        """
        Calculates the output of each node in the layer using the given inputs.

        Args:
            inputs (list[float]): A list of input values.

        Returns:
            list[float]: A list of output values corresponding to each node in the layer.
        """
        return [node.calculate(inputs) for node in self.nodes]

    def backpropagate(self, errors: list[float], learning_rate: float):
        """
        Perform backpropagation on the layer.

        Args:
            errors (list[float]): The errors for each node in the layer.
            learning_rate (float): The learning rate for weight updates.
        """
        for node, error in zip(self.nodes, errors):
            delta = error * sigmoid_derivative(node.output)
            node.update_weights(delta, learning_rate)


class NeuralNetModel:
    def __init__(self, input_size: int, layers: list[NeuralNetLayer]):
        """
        Initializes a new instance of the NeuralNetModel class with the specified input size and layers.

        Args:
            input_size (int): The size of the input.
            layers (list[NeuralNetLayer]): A list of NeuralNetLayer objects representing the layers in the model.
        """
        self.input_size = input_size
        self.layers = layers

    def calculate(self, inputs: list[float]) -> list[float]:
        """
        Calculates the output of the neural network model using the given inputs.

        Args:
            inputs (list[float]): A list of input values.

        Returns:
            list[float]: A list of output values corresponding to each layer in the model.
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer.calculate(outputs)
        return outputs

    def backpropagate(
        self,
        expected_output: list[float],
        actual_output: list[float],
        learning_rate: float,
    ):
        """
        Perform backpropagation through the network.

        Args:
            expected_output (list[float]): The expected output values.
            actual_output (list[float]): The actual output values.
            learning_rate (float): The learning rate for weight updates.
        """
        # Calculate the error for the output layer
        output_errors = [
            expected - actual
            for expected, actual in zip(expected_output, actual_output)
        ]

        # Backpropagate errors through the layers
        errors = output_errors
        for layer in reversed(self.layers):
            layer.backpropagate(errors, learning_rate)
            errors = [
                sum(
                    node.weights[j] * sigmoid_derivative(node.output) * error
                    for node, error in zip(layer.nodes, output_errors)
                )
                for j in range(len(layer.nodes[0].weights))
            ]

    def train(
        self,
        learning_rate: float,
        epochs: int,
        inputs: list[list[float]],
        outputs: list[list[float]],
    ):
        """
        Trains the neural network model using gradient descent.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): The number of epochs to train for.
            inputs (list[list[float]]): A list of input values.
            outputs (list[list[float]]): A list of output values.
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0")
        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0")
        if not inputs or not outputs:
            raise ValueError("Inputs and outputs lists cannot be empty")
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs lists must have the same length")

        for epoch in range(epochs):
            for i in range(len(inputs)):
                input = inputs[i]
                expected_output = outputs[i]

                if not input or not expected_output:
                    raise ValueError(
                        "Inputs and outputs lists cannot contain empty lists"
                    )
                if len(input) != self.input_size:
                    raise ValueError("Input size must match the model input size")
                if len(expected_output) != len(self.layers[-1].nodes):
                    raise ValueError(
                        "Output size must match the number of nodes in the last layer"
                    )

                actual_output = self.calculate(input)
                error = calculate_error(expected_output, actual_output)

                print(
                    f"Epoch: {epoch + 1}, Input: {input}, Expected: {expected_output}, Actual: {actual_output}, Error: {error}"
                )

                self.backpropagate(expected_output, actual_output, learning_rate)
                
    def save(self, path: str):
        """
        Saves the model to a file.

        Args:
            path (str): The path to save the model to.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function.

    Args:
        x (float): The input value.

    Returns:
        float: The sigmoid of the input value.
    """
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    """
    Derivative of the sigmoid function.

    Args:
        x (float): The input value.

    Returns:
        float: The derivative of the sigmoid function.
    """
    return x * (1 - x)


def calculate_error(expected_output: list[float], actual_output: list[float]) -> float:
    """
    Calculates the mean squared error between the expected output and the actual output.

    Args:
        expected_output (list[float]): The expected output.
        actual_output (list[float]): The actual output.

    Returns:
        float: The mean squared error between the expected output and the actual output.
    """
    if len(expected_output) != len(actual_output):
        raise ValueError("Expected output and actual output must have the same length")

    return sum(
        (expected_output[i] - actual_output[i]) ** 2
        for i in range(len(expected_output))
    ) / len(expected_output)


class Architect:
    @staticmethod
    def create(input_size: int, output_size: int, hidden_sizes: list[int]):
        """
        Creates a neural network model with the specified input size, output size, and hidden layer sizes.

        Parameters:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            hidden_sizes (list[int]): A list of integers representing the sizes of the hidden layers.

        Returns:
            NeuralNetModel: The created neural network model.
        """
        layers = []

        prev_size = input_size
        for size in hidden_sizes:
            layers.append(
                NeuralNetLayer(
                    [
                        NeuralNetNode(
                            random.uniform(-0.5, 0.5),
                            [random.uniform(-0.5, 0.5) for _ in range(prev_size)],
                            sigmoid,
                        )
                        for _ in range(size)
                    ]
                )
            )
            prev_size = size

        layers.append(
            NeuralNetLayer(
                [
                    NeuralNetNode(
                        random.uniform(-0.5, 0.5),
                        [random.uniform(-0.5, 0.5) for _ in range(prev_size)],
                        sigmoid,
                    )
                    for _ in range(output_size)
                ]
            )
        )

        return NeuralNetModel(input_size, layers)

    @staticmethod
    def load(path: str):
        """
        Loads the model from a file.

        Args:
            path (str): The path to load the model from.

        Returns:
            NeuralNetModel: The loaded model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


def train(model: NeuralNetModel, inputs: list[list[float]], outputs: list[list[float]]):
    """
    Trains the neural network model using gradient descent.

    Args:
        model (NeuralNetModel): The neural network model to train.
        inputs (list[list[float]]): A list of input values.
        outputs (list[list[float]]): A list of output values.
    """
    model.train(model, 0.1, 100, inputs, outputs)
