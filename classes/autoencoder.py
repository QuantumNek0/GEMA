from config import *
from utility.data_augmentation import AugmentedMidi

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

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


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(DEFAULT_INPUT_SIZE, DEFAULT_HIDDEN_SIZE)
        self.linear2 = nn.Linear(DEFAULT_HIDDEN_SIZE, latent_dims)
        self.linear3 = nn.Linear(DEFAULT_HIDDEN_SIZE, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        if device == "cuda":
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, DEFAULT_HIDDEN_SIZE)
        self.linear2 = nn.Linear(DEFAULT_HIDDEN_SIZE, DEFAULT_INPUT_SIZE)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder: VariationalAutoencoder, data, epochs: int):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        epoch_loss = 0.0

        if epoch != 0:
          print("\n")
        print(f"epoch: {epoch + 1}/{epochs}")

        i = 0
        len_data = len(data)
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)

            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

            i = i + 1
            print(f"\r{'{:.2f}'.format(i*100 / len_data)}%", end='')

        epoch_loss = epoch_loss / len_data
        print(f", loss = {'{:.6f}'.format(epoch_loss)}", end='')
    return autoencoder


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

        # midi_data = pretty_midi.PrettyMIDI(file_path)
        # piano_roll = midi_data.get_piano_roll(fs=20)
        # piano_roll = np.transpose(piano_roll, (1, 0))
        # piano_roll = np.expand_dims(piano_roll, axis=0)
        # piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
        # return piano_roll, class_idx

        m = AugmentedMidi(single_notes=True, playable=False)
        m.read_midi(file_path)
        m = m.normalize()

        while sum(m.melody[1]["durations"]) < 16:
            m.add_padding(1, track=1)

        notes = [note / 127 for note in m.melody[1]["notes"]]
        # notes = np.expand_dims(notes, axis=0)
        notes = torch.tensor(notes, dtype=torch.float32)

        return notes, short_key_encoding[m.key]

    def __len__(self):
        return len(self.files)


def main():
    # Hyperparameters
    latent_dims = DEFAULT_LATENT_SIZE
    no_epochs = DEFAULT_NO_EPOCHS

    vae = VariationalAutoencoder(latent_dims).to(device)  # GPU

    midi_dataset = MidiDataset("data/augmented_data")
    data_loader = torch_data.DataLoader(midi_dataset, batch_size=32, shuffle=True)

    vae = train(vae, data_loader, epochs=no_epochs)
    vae.eval()
    torch.save(vae.state_dict(), 'music_vae.pt')


if __name__ == '__main__':
    main()
