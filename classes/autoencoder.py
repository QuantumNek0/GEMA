from config import *

# Define the hyperparameters
input_size = 784   # number of input neurons (28x28 pixels)
hidden_size = 256  # number of neurons in the hidden layer
output_size = 128  # number of neurons in the output layer
learning_rate = 0.01
num_epochs = 10


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Pass the input through the encoder
        x = self.encoder(x)

        # Pass the encoded input through the decoder
        x = self.decoder(x)

        return x


def main():
    # Instantiate the autoencoder
    autoencoder = Autoencoder()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Train the autoencoder
    for epoch in range(num_epochs):
        pass


if __name__ == '__main__':
    main()
