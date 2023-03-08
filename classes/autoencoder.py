from config import *
from utility.data_augmentation import AugmentedMidi

device = 'cuda' if torch.cuda.is_available() else 'cpu'

short_key_encoding = {
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
        # notes = np.expand_dims(notes, axis=0)

        notes = torch.tensor(notes, dtype=torch.float32)
        return notes, short_key_encoding[m.key]

    def __len__(self):
        return len(self.files)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        self.activation = DEFAULT_ACTIVATION

        self.linear1 = nn.Linear(DEFAULT_INPUT_SIZE, DEFAULT_HIDDEN1_SIZE)
        self.linear2 = nn.Linear(DEFAULT_HIDDEN1_SIZE, DEFAULT_HIDDEN2_SIZE)
        self.linear3 = nn.Linear(DEFAULT_HIDDEN2_SIZE, DEFAULT_HIDDEN3_SIZE)
        self.linear4 = nn.Linear(DEFAULT_HIDDEN3_SIZE, latent_dims)
        self.linear5 = nn.Linear(DEFAULT_HIDDEN3_SIZE, latent_dims)

        self.batch1 = nn.BatchNorm1d(DEFAULT_HIDDEN1_SIZE)
        self.batch2 = nn.BatchNorm1d(DEFAULT_HIDDEN2_SIZE)
        self.batch3 = nn.BatchNorm1d(DEFAULT_HIDDEN3_SIZE)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        self.apply(self._init_weights)

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

    @staticmethod
    def _init_weights(module):
        class_name = module.__class__.__name__

        if class_name.find('Linear') != -1:
            n = module.in_features
            y = 1.0 / np.sqrt(n)

            module.weight.data.uniform_(-y, y)
            module.bias.data.fill_(0)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()

        self.activation = DEFAULT_ACTIVATION
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, DEFAULT_HIDDEN3_SIZE),
            nn.BatchNorm1d(DEFAULT_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN3_SIZE, DEFAULT_HIDDEN2_SIZE),
            nn.BatchNorm1d(DEFAULT_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN2_SIZE, DEFAULT_HIDDEN1_SIZE),
            nn.BatchNorm1d(DEFAULT_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_HIDDEN1_SIZE, DEFAULT_INPUT_SIZE),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def forward(self, x):
        decoded_x = self.decoder(x)

        return decoded_x

    @staticmethod
    def _init_weights(module):
        class_name = module.__class__.__name__

        if class_name.find('Linear') != -1:
            n = module.in_features
            y = 1.0 / np.sqrt(n)

            module.weight.data.uniform_(-y, y)
            module.bias.data.fill_(0)


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


def train(vae: VariationalAutoencoder, dataset, epochs: int):
    opt = torch.optim.Adam(vae.parameters(), lr=DEFAULT_LEARNING_RATE)

    training_loader = torch_data.DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    validation_loader = torch_data.DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    for epoch in range(epochs):

        epoch_loss = train_epoch(vae, training_loader, opt)
        val_loss = test_epoch(vae, validation_loader)

        print(f"epoch: {epoch+1}/{epochs}, epoch loss = {epoch_loss}\tval loss = {val_loss}\n")
        time.sleep(0.1)

    return vae


def train_epoch(vae: VariationalAutoencoder, data, opt):
    # Set train mode for both the encoder and the decoder
    vae.train()
    epoch_loss = 0.0

    for x, _ in tqdm(data, desc="training epoch", unit="batch"):

        x = x.to(device)
        opt.zero_grad()

        x_hat = vae(x)
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl

        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data.dataset)


def test_epoch(vae: VariationalAutoencoder, data):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0

    with torch.no_grad(): # No need to track the gradients
        for x, _ in tqdm(data, desc="testing epoch", unit="batch"):

            x = x.to(device)
            encoded_x = vae.encoder(x)
            x_hat = vae(x)

            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(data.dataset)


def main():
    # Hyperparameters
    latent_dims = DEFAULT_LATENT_SIZE
    n_epochs = DEFAULT_N_EPOCHS

    midi_dataset = MidiDataset("data/augmented_data")

    vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
    vae = train(vae, midi_dataset, epochs=n_epochs)
    torch.save(vae.state_dict(), 'music_vae.pt')


if __name__ == '__main__':
    main()
