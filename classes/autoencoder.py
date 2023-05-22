from config import *
from utility.data_augmentation import AugmentedMidi
from utility.music import relative_min_key, relative_maj_key

device = 'cuda' if torch.cuda.is_available() else 'cpu'

short_key_encoding = { # Even = Major, Odd = Minor // Relative key is +-1
    "Abm": 10,
    "B": 11,
    "Ebm": 12,
    "F#": 13,
    "Bbm": 14,
    "Db": 15,
    "Fm": 16,
    "Ab": 17,
    "Cm": 18,
    "Eb": 19,
    "Gm": 20,
    "Bb": 21,
    "Dm": 22,
    "F": 23,
    "Am": 24,
    "C": 1,
    "Em": 2,
    "G": 3,
    "Bm": 4,
    "D": 5,
    "F#m": 6,
    "A": 7,
    "C#m": 8,
    "E": 9,
}

maj_key_encoding = {
    "B": 6,
    "F#": 7,
    "Db": 8,
    "Ab": 9,
    "Eb": 10,
    "Bb": 11,
    "F": 12,
    "C": 1,
    "G": 2,
    "D": 3,
    "A": 4,
    "E": 5,
}

min_key_encoding = {
    "Abm": 6,
    "Ebm": 7,
    "Bbm": 8,
    "Fm": 9,
    "Cm": 10,
    "Gm": 11,
    "Dm": 12,
    "Am": 1,
    "Em": 2,
    "Bm": 3,
    "F#m": 4,
    "C#m": 5,
}


class MidiDataset(torch_data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.files = []
        for i, cls in enumerate(self.classes):
            if cls == ".DS_Store":
                continue
            cls_dir = os.path.join(data_dir, cls)
            cls_files = [(os.path.join(cls_dir, f), i) for f in os.listdir(cls_dir) if f.endswith(".mid")]
            self.files.extend(cls_files)

    def __getitem__(self, index):
        file_path, class_idx = self.files[index]

        m = AugmentedMidi(single_notes=True, playable=False)
        m.read_midi(file_path)
        m = m.normalize()

        while sum(m.melody[1]["durations"]) < 16:
            m.add_padding(1, track=1)

        notes = [note / N_MIDI_VALUES for note in m.melody[1]["notes"]]
        notes = torch.tensor(notes, dtype=torch.float32)
        # notes = np.expand_dims(notes, axis=0)

        # encoded_original_key = short_key_encoding[m.key]
        #
        # if encoded_original_key % 2 == 0:
        #     encoded_maj_key = encoded_original_key
        #     encoded_min_key = (encoded_original_key - 1)
        #
        #     if encoded_min_key > 24:
        #         encoded_min_key = 1
        #
        #     elif encoded_min_key < 1:
        #         encoded_min_key = 24
        # else:
        #     encoded_min_key = encoded_original_key
        #     encoded_maj_key = (encoded_original_key + 1) % 25
        #
        #     if encoded_maj_key > 24:
        #         encoded_maj_key = 1
        #
        #     elif encoded_min_key < 1:
        #         encoded_maj_key = 24

        if m.key.find('m') == -1:
            encoded_maj_key = maj_key_encoding[m.key]
            encoded_min_key = min_key_encoding[relative_min_key[m.key]]
        else:
            encoded_min_key = min_key_encoding[m.key]
            encoded_maj_key = maj_key_encoding[relative_maj_key[m.key]]

        return notes, (encoded_maj_key, encoded_min_key)

    def __len__(self):
        return len(self.files)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        self.activation = DEFAULT_ACTIVATION

        self.linear1 = nn.Linear(DEFAULT_INPUT_SIZE, DEFAULT_HIDDEN1_SIZE, bias=False)
        self.linear2 = nn.Linear(DEFAULT_HIDDEN1_SIZE, DEFAULT_HIDDEN2_SIZE, bias=False)
        self.linear3 = nn.Linear(DEFAULT_HIDDEN2_SIZE, DEFAULT_HIDDEN3_SIZE, bias=False)
        self.linear4 = nn.Linear(DEFAULT_HIDDEN3_SIZE, latent_dims)
        self.linear5 = nn.Linear(DEFAULT_HIDDEN3_SIZE, latent_dims)

        self.batch1 = nn.BatchNorm1d(DEFAULT_HIDDEN1_SIZE)
        self.batch2 = nn.BatchNorm1d(DEFAULT_HIDDEN2_SIZE)
        self.batch3 = nn.BatchNorm1d(DEFAULT_HIDDEN3_SIZE)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        # self.apply(self._init_weights)

        if device == "cuda":
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.batch1(self.linear1(x)))
        x = F.relu(self.batch2(self.linear2(x)))
        x = F.relu(self.batch3(self.linear3(x)))

        mu = self.linear4(x)
        sigma = torch.exp(self.linear5(x))

        # reparameterization
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        return z

    # @staticmethod
    # def _init_weights(module):
    #     class_name = module.__class__.__name__
    #
    #     if class_name.find('Linear') != -1:
    #         n = module.in_features
    #         y = 1.0 / np.sqrt(n)
    #
    #         module.weight.data.uniform_(-y, y)
    #         module.bias.data.fill_(0)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()

        self.activation = DEFAULT_ACTIVATION
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, DEFAULT_HIDDEN3_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN3_SIZE, DEFAULT_HIDDEN2_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN2_SIZE, DEFAULT_HIDDEN1_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN1_SIZE, DEFAULT_INPUT_SIZE),
            nn.Sigmoid()
        )
        # self.apply(self._init_weights)

    def forward(self, x):
        decoded_x = self.decoder(x)

        return decoded_x

    # @staticmethod
    # def _init_weights(module):
    #     class_name = module.__class__.__name__
    #
    #     if class_name.find('Linear') != -1:
    #         n = module.in_features
    #         y = 1.0 / np.sqrt(n)
    #
    #         module.weight.data.uniform_(-y, y)
    #         module.bias.data.fill_(0)


class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        encoded_x = self.encoder(x)

        return self.decoder(encoded_x)

    # @staticmethod
    # def reparameterize(mu, log_var):
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #
    #     return mu + eps * std


class MajorClassifier(nn.Module):
    def __init__(self, latent_dims):
        super(MajorClassifier, self).__init__()

        self.activation = DEFAULT_ACTIVATION
        self.classifier = nn.Sequential(
            nn.Linear(latent_dims, DEFAULT_HIDDEN3_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN3_SIZE, DEFAULT_HIDDEN2_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN2_SIZE, DEFAULT_HIDDEN1_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN1_SIZE, DEFAULT_INPUT_SIZE),
        )
        # self.apply(self._init_weights)

    def forward(self, x):
        decoded_x = self.classifier(x)

        return decoded_x


class MinorClassifier(nn.Module):
    def __init__(self, latent_dims):
        super(MinorClassifier, self).__init__()

        self.activation = DEFAULT_ACTIVATION
        self.classifier = nn.Sequential(
            nn.Linear(latent_dims, DEFAULT_HIDDEN3_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN3_SIZE, DEFAULT_HIDDEN2_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN2_SIZE, DEFAULT_HIDDEN1_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN1_SIZE, DEFAULT_INPUT_SIZE),
        )
        # self.apply(self._init_weights)

    def forward(self, x):
        decoded_x = self.classifier(x)

        return decoded_x


def train(vae: VariationalAutoencoder, maj_classifier: MajorClassifier, min_classifier: MinorClassifier, dataset, epochs: int):
    opt = torch.optim.Adam(vae.parameters(), lr=DEFAULT_LEARNING_RATE)

    # Define the proportions for each subset
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Calculate the lengths of each subset
    train_len = int(len(dataset) * train_ratio)
    val_len = int(len(dataset) * val_ratio)
    test_len = len(dataset) - train_len - val_len

    train_subset, val_subset, test_subset = random_split(dataset, [train_len, val_len, test_len])

    training_loader = torch_data.DataLoader(train_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    validation_loader = torch_data.DataLoader(val_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    for epoch in range(epochs):

        epoch_loss = train_epoch(vae, maj_classifier, min_classifier, training_loader, opt)
        val_loss = test_epoch(vae, maj_classifier, min_classifier, validation_loader)

        print(f"epoch: {epoch+1}/{epochs}, epoch loss = {epoch_loss}\tval loss = {val_loss}\n")
        time.sleep(0.1)

    return vae


def train_epoch(vae: VariationalAutoencoder, maj_classifier: MajorClassifier, min_classifier: MinorClassifier, data, opt):
    # Set train mode for both the encoder and the decoder
    vae.train()
    maj_classifier.train()
    min_classifier.train()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    epoch_loss = 0.0

    for x, y in tqdm(data, desc="training epoch", unit="batch"):

        x = x.to(device)
        # y = y.to(device)

        opt.zero_grad()

        encoded_x = vae.encoder(x)

        y_hat_1 = maj_classifier(encoded_x)
        y_hat_2 = min_classifier(encoded_x)

        maj_classifier_loss = classifier_criterion(y_hat_1, y[0])
        min_classifier_loss = classifier_criterion(y_hat_2, y[1])

        # vae_loss = ((x - x_hat) ** 2).sum()
        # loss = vae_loss + classifier_loss + vae.encoder.kl

        loss = maj_classifier_loss + (0)*min_classifier_loss + (0)*vae.encoder.kl

        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data.dataset)


def test_epoch(vae: VariationalAutoencoder, maj_classifier: MajorClassifier, min_classifier: MinorClassifier, data):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    maj_classifier.eval()
    min_classifier.eval()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    val_loss = 0.0

    with torch.no_grad(): # No need to track the gradients
        for x, y in tqdm(data, desc="validating epoch", unit="batch"):

            x = x.to(device)
            # y = y.to(device)

            encoded_x = vae.encoder(x)

            y_hat_1 = maj_classifier(encoded_x)
            y_hat_2 = min_classifier(encoded_x)

            maj_classifier_loss = classifier_criterion(y_hat_1, y[0])
            min_classifier_loss = classifier_criterion(y_hat_2, y[1])

            # vae_loss = ((x - x_hat)**2).sum()
            # loss = vae_loss + classifier_loss + vae.encoder.kl

            loss = maj_classifier_loss + (0)*min_classifier_loss + (0)*vae.encoder.kl

            val_loss += loss.item()

    return val_loss / len(data.dataset)


def main():
    # Hyperparameters
    latent_dims = DEFAULT_LATENT_SIZE
    n_epochs = DEFAULT_N_EPOCHS

    midi_dataset = MidiDataset("data/augmented_data")

    vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
    maj_classifier = MajorClassifier(latent_dims).to(device)
    min_classifier = MinorClassifier(latent_dims).to(device)

    vae = train(vae, maj_classifier, min_classifier, midi_dataset, n_epochs)
    torch.save(vae.state_dict(), 'music_vae.pt')
    # torch.save(classifier.state_dict(), 'music_classifier.pt')


if __name__ == '__main__':
    main()
