import random
import math
import pickle

import numpy as np


normal = 0
liquid = 1


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
        is_liquid: bool = False,
    ):
        """
        Perform backpropagation through the network.

        Args:
            expected_output (list[float]): The expected output values.
            actual_output (list[float]): The actual output values.
            learning_rate (float): The learning rate for weight updates.
            is_liquid (bool, optional): Whether the model is a liquid model. Defaults to False.
        """
        # Calculate the error for the output layer
        output_errors = [
            expected - actual
            for expected, actual in zip(expected_output, actual_output)
        ]

        if is_liquid:
            # Backpropagate errors through the output layer
            errors = output_errors
            self.layers[-1].backpropagate(errors, learning_rate)
            errors = [
                sum(
                    node.weights[j] * sigmoid_derivative(node.output) * error
                    for node, error in zip(self.layers[-1].nodes, output_errors)
                )
                for j in range(len(self.layers[-1].nodes[0].weights))
            ]
        else:
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
        is_liquid: bool = False,
    ):
        """
        Trains the neural network model using gradient descent.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): The number of epochs to train for.
            inputs (list[list[float]]): A list of input values.
            outputs (list[list[float]]): A list of output values.
            is_liquid (bool, optional): Whether the model is a liquid model. Defaults to False.
        """
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be greater than 0 (got {learning_rate})")
        if epochs <= 0:
            raise ValueError(f"Epochs must be greater than 0 (got {epochs})")
        if not inputs or not outputs:
            raise ValueError("Inputs and outputs lists cannot be empty")
        if len(inputs) != len(outputs):
            raise ValueError(f"Inputs ({len(inputs)}) and outputs ({len(outputs)}) lists must have the same length")

        for epoch in range(epochs):
            for i in range(len(inputs)):
                input = inputs[i]
                expected_output = outputs[i]

                if not input or not expected_output:
                    raise ValueError(
                        "Inputs and outputs lists cannot contain empty lists"
                    )
                if len(input) != self.input_size:
                    raise ValueError(f"Input size ({len(input)}) must match the model input size ({self.input_size})")
                if len(expected_output) != len(self.layers[-1].nodes):
                    raise ValueError(
                        f"Output size ({len(expected_output)}) must match the number of nodes in the last layer ({len(self.layers[-1].nodes)})"
                    )

                actual_output = self.calculate(input)
                error = calculate_error(expected_output, actual_output)

                print(
                    f"Epoch: {epoch + 1}, Input: {input}, Expected: {expected_output}, Actual: {actual_output}, Error: {error}"
                )

                self.backpropagate(expected_output, actual_output, learning_rate)

    def train_with_feedback(
        self,
        learning_rate: float,
        epochs: int,
        feedback_loops: int,
        initial_inputs: list[list[float]],
        initial_outputs: list[list[float]],
        test_inputs: list[list[float]],
        test_outputs: list[list[float]],
        is_liquid: bool = False,
    ):
        # Initial training
        self.train(learning_rate, epochs, initial_inputs, initial_outputs, is_liquid)

        # Feedback loop for self-adjustment
        for loop in range(feedback_loops):
            print(f"Feedback Loop: {loop + 1}")
            for x, expected in zip(test_inputs, test_outputs):
                actual_output = self.calculate(x)
                self.train(learning_rate, 1, [actual_output], [expected], is_liquid)
                print(f"Test Input: {x}, Expected: {expected}, Actual: {actual_output}")

    def save(self, path: str):
        """
        Saves the model to a file.

        Args:
            path (str): The path to save the model to.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def test(self, inputs: list[list[float]], outputs: list[list[float]]):
        """
        Tests the neural network model on the given inputs and outputs.

        Args:
            inputs (list[list[float]]): A list of input values.
            outputs (list[list[float]]): A list of output values.
        """
        if not inputs or not outputs:
            raise ValueError("Inputs and outputs lists cannot be empty")
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs lists must have the same length")

        for input, expected_output in zip(inputs, outputs):
            actual_output = self.calculate(input)
            error = calculate_error(expected_output, actual_output)
            print(
                f"Input: {input}, Expected: {expected_output}, Actual: {actual_output}, Error: {error}"
            )


class Tokenizer:
    def __init__(self, vocab: list):
        """
        Initializes a tokenizer with a given vocabulary.

        Args:
            vocab (list): A list of tokens.
        """
        self.vocab = vocab

    def tokenize(self, text: str) -> list[int]:
        """
        Converts a text string into a list of token indices.

        Args:
            text (str): The input text string.

        Returns:
            list[int]: A list of token indices.
        """
        return [self.vocab.index(token) if token in self.vocab else self.vocab.index("<UNK>") for token in text.split()]

    def detokenize(self, tokens: list[int]) -> str:
        """
        Converts a list of token indices back into a text string.

        Args:
            tokens (list[int]): A list of token indices.

        Returns:
            str: The resulting text string.
        """
        return " ".join([self.vocab[token] if token < len(self.vocab) else "<UNK>" for token in tokens])


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 5000):
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.encoding = pe

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (np.ndarray): The input tensor of shape (seq_len, d_model).

        Returns:
            np.ndarray: The tensor with positional encoding added.
        """
        seq_len, d_model = x.shape
        return x + self.encoding[:seq_len, :]


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        """
        Implements the multi-head attention mechanism.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
        """
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = np.random.rand(d_model, d_model)
        self.k_linear = np.random.rand(d_model, d_model)
        self.v_linear = np.random.rand(d_model, d_model)
        self.out_linear = np.random.rand(d_model, d_model)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _scaled_dot_product_attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ):
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return np.matmul(weights, v)

    def __call__(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> np.ndarray:
        q = np.dot(query, self.q_linear)
        k = np.dot(key, self.k_linear)
        v = np.dot(value, self.v_linear)

        q, k, v = map(self._split_heads, (q, k, v))

        attention = self._scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(0, 2, 1, 3).reshape(
            query.shape[0], -1, self.num_heads * self.d_k
        )
        return np.dot(attention, self.out_linear)


class TransformerBlock:
    def __init__(self, d_model: int, num_heads: int, ff_dim: int):
        """
        Implements a single transformer block with multi-head attention and feedforward layers.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimension of the feedforward network.
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = [np.random.rand(d_model, ff_dim), np.random.rand(ff_dim, d_model)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        attn_output = self.attention(x, x, x)
        ff_output = np.maximum(0, np.dot(attn_output, self.ffn[0]))  # ReLU activation
        ff_output = np.dot(ff_output, self.ffn[1])
        return ff_output


class GPTModel(NeuralNetModel):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        vocab_size: int,
        vocab: list[str],
    ):
        super().__init__(input_size, [])
        self.tokenizer = Tokenizer(vocab)
        self.embedding = np.random.rand(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, ff_dim) for _ in range(num_layers)
        ]
        self.final_linear = np.random.rand(d_model, vocab_size)
        self.vocab_size = vocab_size

    def calculate(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.tokenize(text)
        embeddings = self.embedding[tokens]  # Shape: (seq_len, d_model)
        embeddings = self.positional_encoding(embeddings)  # Shape: (seq_len, d_model)

        for block in self.transformer_blocks:
            embeddings = block(embeddings)

        logits = np.dot(embeddings, self.final_linear)  # Shape: (seq_len, vocab_size)
        return logits

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        generated_text = prompt
        for _ in range(max_length):
            logits = self.calculate(generated_text)
            probabilities = np.exp(logits[-1]) / np.sum(np.exp(logits[-1]), axis=-1)
            next_token = np.argmax(probabilities)
            next_word = self.tokenizer.detokenize([next_token])
            generated_text += " " + next_word
            if next_word == "<EOS>":
                break
        return generated_text

    def train(self, dataset: list[str], epochs: int = 10, learning_rate: float = 0.001):
        for epoch in range(epochs):
            total_loss = 0
            for text in dataset:
                tokens = self.tokenizer.tokenize(text)
                target = tokens[1:] + [
                    self.tokenizer.tokenize("<EOS>")[0]
                ]  # Shift tokens for target

                # Forward pass
                logits = self.calculate(text[:-1])  # Exclude last token for input
                loss = self.cross_entropy_loss(logits, target)
                total_loss += loss

                # Backward pass
                gradients = self.backpropagate(logits, target)
                self.update_weights(gradients, learning_rate)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset)}")

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        max_logits = np.max(logits, axis=-1, keepdims=True)
        stable_logits = logits - max_logits
        exp_logits = np.exp(stable_logits)
        sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
        probabilities = exp_logits / sum_exp_logits
        return probabilities

    def cross_entropy_loss(self, logits: np.ndarray, target: list[int]) -> float:
        """
        Computes the cross-entropy loss between logits and the target tokens.

        Args:
            logits (np.ndarray): The output logits from the model.
            target (list[int]): The target token indices.

        Returns:
            float: The computed cross-entropy loss.
        """
        assert logits is not None, "Logits should not be None"
        assert target is not None, "Target should not be None"
        assert len(target) > 0, "Target should not be empty"

        print("Logits shape:", logits.shape)
        logits = np.squeeze(logits, axis=1)
        print("Logits shape after squeeze:", logits.shape)
        probabilities = self.softmax(logits)
        print("Probabilities shape:", probabilities.shape)
        print("Probabilities:", probabilities)
        print("Target shape:", len(target))
        print("Target indices:", target)

        # make sure no index is out of range
        for index in target:
            if index >= probabilities.shape[1]:
                target[target.index(index)] = probabilities.shape[1] - 1

        if len(target) > logits.shape[0]:
            raise ValueError(
                f"Target length {len(target)} is greater than logits length {logits.shape[0]}"
            )

        correct_log_probs = -np.log(probabilities[range(len(target)), target])
        return np.sum(correct_log_probs) / len(target)

    def backpropagate(self, logits: np.ndarray, target: list[int]) -> list[np.ndarray]:
        """
        Performs backpropagation to compute the gradients of the loss with respect to model weights.

        Args:
            logits (np.ndarray): The output logits from the model.
            target (list[int]): The target token indices.

        Returns:
            list[np.ndarray]: The gradients for each weight matrix in the model.
        """
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        probabilities = np.squeeze(probabilities, axis=1)
        print("Probabilities:", probabilities)
        print("Probabilities shape:", probabilities.shape)
        probabilities[
            range(len(target)), target
        ] -= 1  # Subtract 1 from the correct class
        gradients = []

        print("Embedding shape (T):", self.embedding.T.shape)

        # Compute gradient for the final linear layer
        grad_final_linear = np.dot(probabilities, self.embedding.T)
        gradients.append(grad_final_linear)

        # Backpropagate through the transformer blocks
        grad_embeddings = np.dot(probabilities, self.final_linear.T)
        for block in reversed(self.transformer_blocks):
            grad_embeddings = block.backward(grad_embeddings)
        gradients.append(grad_embeddings)

        return gradients

    def update_weights(self, gradients: list[np.ndarray], learning_rate: float):
        """
        Updates the model's weights using the computed gradients and the learning rate.

        Args:
            gradients (list[np.ndarray]): The computed gradients for the weights.
            learning_rate (float): The learning rate for the update.
        """
        # Update weights for the final linear layer
        self.final_linear -= learning_rate * gradients[0]

        # Update weights for the transformer blocks
        for block, grad in zip(self.transformer_blocks, gradients[1:]):
            block.update_weights(grad, learning_rate)


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
    def create(
        model_type: int, input_size: int, output_size: int, hidden_sizes: list[int]
    ):
        """
        Creates a neural network model with the specified type, input size, output size, and hidden layer sizes.

        Parameters:
            model_type (int): The type of the neural network model.
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            hidden_sizes (list[int]): A list of integers representing the sizes of the hidden layers.

        Returns:
            NeuralNetModel: The created neural network model.
            or LiquidNeuralNetModel: The created liquid neural network (LNN) model.

        Raises:
            ValueError: If the model type is invalid.
        """
        if model_type == 0:
            return Architect._create_neural_net_model(
                input_size, output_size, hidden_sizes
            )
        elif model_type == 1:
            return Architect._create_liquid_neural_net_model(
                input_size, output_size, hidden_sizes
            )
        else:
            raise ValueError("Invalid model type")

    @staticmethod
    def _create_neural_net_model(
        input_size: int, output_size: int, hidden_sizes: list[int]
    ):
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
    def _create_liquid_neural_net_model(
        input_size: int, output_size: int, hidden_sizes: list[int]
    ):
        """
        Creates a liquid neural network (LNN) model with the specified input size, output size, and hidden layer sizes.

        Parameters:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            hidden_sizes (list[int]): A list of integers representing the sizes of the hidden layers.

        Returns:
            NeuralNetModel: The created liquid neural network (LNN) model.
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
    def load(path: str) -> NeuralNetModel:
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


def train_test_split(X: list[float], y: list[float], test_size: float):
    """
    Splits the input data into training and test sets. Randomizes the order.
    Splits the input data into training and test sets.

    Args:
        X (list[float]): The input data.
        y (list[float]): The target data.
        test_size (float): The proportion of the data to use for the test set.

    Returns:
        tuple[list[float], list[float], list[float], list[float]]: The training and test sets.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("Test size must be between 0 and 1")

    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    test_indices = indices[: int(n * test_size)]
    train_indices = indices[int(n * test_size) :]
    X_train, X_test = [X[i] for i in train_indices], [X[i] for i in test_indices]
    y_train, y_test = [y[i] for i in train_indices], [y[i] for i in test_indices]
    train_size = 1 - test_size
    X_train, X_test = X[: int(train_size * len(X))], X[int(train_size * len(X)) :]
    y_train, y_test = y[: int(train_size * len(y))], y[int(train_size * len(y)) :]
    return X_train, X_test, y_train, y_test
