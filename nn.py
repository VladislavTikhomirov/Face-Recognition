import numpy as np
from PIL import Image
import glob

# Load and preprocess images
def load_images(image_folder, image_size=(64, 64)):
    images = []
    for filepath in glob.glob(f"{image_folder}/*.jpg"): 
        img = Image.open(filepath).convert("L")
        img = img.resize(image_size)
        img_array = np.array(img).flatten() / 255.0 
        images.append(img_array)
    return np.array(images)

# Define the neural network
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, output, learning_rate):
        # Compute gradients
        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
        
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
        
        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")



import fnmatch, os
# Example Usage
if __name__ == "__main__":
    # Load images
    image_folder = "/face_data"
    for path,dirs,files in os.walk('.'):
        for file in files:
            if fnmatch.fnmatch(file,'*.txt'):
                fullname = os.path.join(path,file)
                print(fullname)
    images = load_images(image_folder)
    
    # Generate dummy labels (e.g., binary labels for classification)
    labels = np.random.randint(0, 2, size=(images.shape[0], 1))

    # Initialize the neural network
    input_size = images.shape[1]
    hidden_size = 64  # Arbitrary choice
    output_size = 1   # Binary classification
    nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    # Train the network
    nn.train(images, labels, epochs=1000, learning_rate=0.01)