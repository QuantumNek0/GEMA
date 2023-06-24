from config import *
from classes.midi import MIDI
from utility.data_augmentation import AugmentedMidi
from utility.music import relative_min_key, relative_maj_key

device = 'cuda' if torch.cuda.is_available() else 'cpu'

short_key_encoding = { # Even = Major, Odd = Minor // Relative key is +-1
    "Abm": 9,
    "B": 10,
    "Ebm": 11,
    "F#": 12,
    "Bbm": 13,
    "Db": 14,
    "Fm": 15,
    "Ab": 16,
    "Cm": 17,
    "Eb": 18,
    "Gm": 19,
    "Bb": 20,
    "Dm": 21,
    "F": 22,
    "Am": 23,
    "C": 0,
    "Em": 1,
    "G": 2,
    "Bm": 3,
    "D": 4,
    "F#m": 5,
    "A": 6,
    "C#m": 7,
    "E": 8,
}

maj_key_encoding = {
    "B": 5,
    "F#": 6,
    "Db": 7,
    "Ab": 8,
    "Eb": 9,
    "Bb": 10,
    "F": 11,
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
}

min_key_encoding = {
    "Abm": 5,
    "Ebm": 6,
    "Bbm": 7,
    "Fm": 8,
    "Cm": 9,
    "Gm": 10,
    "Dm": 11,
    "Am": 0,
    "Em": 1,
    "Bm": 2,
    "F#m": 3,
    "C#m": 4,
}


class MidiDataset(torch_data.Dataset):
    def __init__(self, data_dir, are_tensors: bool = True):
        self.data_dir = data_dir
        self.are_tensors = are_tensors
        self.classes = sorted(os.listdir(data_dir))
        self.midi = []
        for i, cls in enumerate(self.classes):
            if cls == ".DS_Store":
                continue
            cls_dir = os.path.join(data_dir, cls)

            for f in tqdm(os.listdir(cls_dir), desc="loading midis", unit="midi"):
                if f.endswith(".mid"):
                    midi_path = os.path.join(cls_dir, f)

                    m = MIDI(single_notes=True, playable=False)
                    m.read_midi(midi_path)

                    notes = [note for note in m.melody[1]["notes"]]
                    note_lengths = [note_len for note_len in m.melody[1]["durations"]]

                    if self.are_tensors:
                        notes = torch.tensor(notes, dtype=torch.float32)
                        note_lengths = torch.tensor(note_lengths, dtype=torch.float32)

                    self.midi.append((notes, note_lengths, m.key, m.bpm))

    def __getitem__(self, index):
        return self.midi[index]

    def __setitem__(self, key, value):
        self.midi[key] = value

    def __len__(self):
        return len(self.midi)


def preprocess_midi(midi_data: MidiDataset):

    i = 0
    for notes, note_lengths, key, bpm in tqdm(midi_data, desc="preprocessing midis", unit="midi"):

        m = AugmentedMidi(single_notes=True, playable=False)
        m.add_key(key)
        m.add_tempo(bpm)

        for note, note_length in zip(notes, note_lengths):

            m.add_note(int(note.numpy()), note_length.numpy())

        m.normalize()
        while sum(m.melody[0]["durations"]) < 16: # Duration of 4 bars
            m.add_padding(1)

        normalized_notes = [raw_note / N_MIDI_VALUES for raw_note in m.melody[0]["notes"]]
        if midi_data.are_tensors:
            normalized_notes = torch.tensor(normalized_notes, dtype=torch.float32)

        if key.find('m') == -1:
            encoded_maj_key = maj_key_encoding[key]
            # encoded_min_key = min_key_encoding[relative_min_key[m.key]]
        else:
            # encoded_min_key = min_key_encoding[m.key]
            encoded_maj_key = maj_key_encoding[relative_maj_key[key]]

        midi_data[i] = (normalized_notes, encoded_maj_key)
        i += 1

    return midi_data


def data_report(data: torch_data.Dataset, columns: [], title: str = "Data Report"):
    from dataprep.eda import create_report

    df = pd.DataFrame(columns=columns)

    for i, sample_point in enumerate(data):
        df.loc[i] = [*sample_point]

    report = create_report(df, title=title)
    report.save()


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        self.activation = DEFAULT_ACTIVATION

        self.linear1 = nn.Linear(DEFAULT_VAE_INPUT_SIZE, DEFAULT_VAE_HIDDEN1_SIZE, bias=False)
        self.linear2 = nn.Linear(DEFAULT_VAE_HIDDEN1_SIZE, DEFAULT_VAE_HIDDEN2_SIZE, bias=False)
        self.linear3 = nn.Linear(DEFAULT_VAE_HIDDEN2_SIZE, DEFAULT_VAE_HIDDEN3_SIZE, bias=False)
        self.linear4 = nn.Linear(DEFAULT_VAE_HIDDEN3_SIZE, latent_dims, bias=False)
        self.linear5 = nn.Linear(DEFAULT_VAE_HIDDEN3_SIZE, latent_dims)

        self.batch1 = nn.BatchNorm1d(DEFAULT_VAE_HIDDEN1_SIZE)
        self.batch2 = nn.BatchNorm1d(DEFAULT_VAE_HIDDEN2_SIZE)
        self.batch3 = nn.BatchNorm1d(DEFAULT_VAE_HIDDEN3_SIZE)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        # self.apply(self._init_weights)

        if device == "cuda":
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = x.to(device)
        x = self.activation(self.batch1(self.linear1(x)))
        x = self.activation(self.batch2(self.linear2(x)))
        x = self.activation(self.batch3(self.linear3(x)))

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
            nn.Linear(latent_dims, DEFAULT_VAE_HIDDEN3_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_VAE_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_VAE_HIDDEN3_SIZE, DEFAULT_VAE_HIDDEN2_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_VAE_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_VAE_HIDDEN2_SIZE, DEFAULT_VAE_HIDDEN1_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_VAE_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_VAE_HIDDEN1_SIZE, DEFAULT_VAE_INPUT_SIZE, bias=False),
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


class Classifier(nn.Module):
    def __init__(self, latent_dims):
        super(Classifier, self).__init__()

        self.activation = DEFAULT_ACTIVATION
        self.classifier = nn.Sequential(
            nn.Linear(latent_dims, DEFAULT_CLASSIFIER_HIDDEN1_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_CLASSIFIER_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_CLASSIFIER_HIDDEN1_SIZE, DEFAULT_CLASSIFIER_HIDDEN2_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_CLASSIFIER_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_CLASSIFIER_HIDDEN2_SIZE, DEFAULT_CLASSIFIER_HIDDEN3_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_CLASSIFIER_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_CLASSIFIER_HIDDEN3_SIZE, DEFAULT_CLASSIFIER_OUTPUT_SIZE, bias=False),
        )
        # self.apply(self._init_weights)

    def forward(self, x):
        decoded_x = self.classifier(x)

        return decoded_x


def train(vae: VariationalAutoencoder, classifier: Classifier, dataset, epochs: int):
    params = list(vae.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=DEFAULT_LEARNING_RATE)

    # Define the proportions for each subset
    train_ratio = 0.8
    val_ratio = 0.1

    # Calculate the lengths of each subset
    train_len = int(len(dataset) * train_ratio)
    val_len = int(len(dataset) * val_ratio)
    test_len = int(len(dataset) - (train_len + val_len))

    train_subset, val_subset, test_subset = random_split(dataset, [train_len, val_len, test_len])

    training_loader = torch_data.DataLoader(train_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    validation_loader = torch_data.DataLoader(val_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    test_loader = torch_data.DataLoader(test_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    epoch_losses = []
    val_losses = []
    epochs_entropy = []
    epochs_precision = []
    epochs_accuracy = []

    for epoch in range(epochs):

        epoch_loss, entropy_train, precision_train, accuracy_train = train_epoch(vae, classifier, training_loader, opt)
        val_loss, entropy_val, precision_val, accuracy_val = val_epoch(vae, classifier, validation_loader)

        entropy = (entropy_train + entropy_val) / 2
        precision = (precision_train + precision_val) / 2
        accuracy = (accuracy_train + accuracy_val) / 2

        epoch_losses.append(epoch_loss)
        val_losses.append(val_loss)
        epochs_entropy.append(entropy)
        epochs_precision.append(precision)
        epochs_accuracy.append(accuracy)

        print(f"epoch: {epoch+1}/{epochs}, epoch loss = {epoch_loss}\tval loss = {val_loss}\tentropy = {entropy}"
              f"\tprecision = {precision}\taccuracy = {accuracy}\n")
        time.sleep(0.1)

    test_loss, entropy, precision, accuracy = test_epoch(vae, classifier, test_loader)
    print(f"test loss = {test_loss}\tentropy = {entropy}\tprecision = {precision}\taccuracy = {accuracy}\n")

    import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 150

    plt.plot(range(epochs), epoch_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.plot(range(epochs), epochs_entropy, label='Entropy')
    plt.plot(range(epochs), epochs_precision, label='Precision')
    plt.plot(range(epochs), epochs_accuracy, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss at Different Epochs')
    plt.legend(loc='upper center')
    plt.show()

    return vae


def train_epoch(vae: VariationalAutoencoder, classifier: Classifier, data, opt):
    # Set train mode for both the encoder and the decoder
    vae.train()
    classifier.train()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    epoch_loss = 0.0
    entropy = 0.0
    precision = 0.0
    accuracy = 0.0

    for x, y in tqdm(data, desc="training epoch", unit="batch"):

        x = x.to(device)
        # y = y.to(device)

        opt.zero_grad()

        encoded_x = vae.encoder(x)
        y_hat = classifier(encoded_x)
        x_hat = vae(x)

        classifier_loss = classifier_criterion(y_hat, y)
        vae_loss = ((x - x_hat) ** 2).sum()
        # loss = vae_loss + classifier_loss + vae.encoder.kl

        loss = classifier_loss + vae.encoder.kl

        _, y_pred = torch.max(y_hat, 1)
        p = precision_score(y.numpy(), y_pred.numpy(), average='micro')
        acc = accuracy_score(y.numpy(), y_pred.numpy())

        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        entropy += vae_loss.item()
        precision += p.item()
        accuracy += acc

    len_data = len(data.dataset)

    epoch_loss /= len_data
    entropy /= len_data
    precision /= len_data
    accuracy /= len_data

    return epoch_loss, entropy, precision, accuracy


def val_epoch(vae: VariationalAutoencoder, classifier: Classifier, data):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    classifier.eval()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    val_loss = 0.0
    entropy = 0.0
    precision = 0.0
    accuracy = 0.0

    with torch.no_grad(): # No need to track the gradients
        for x, y in tqdm(data, desc="validating epoch", unit="batch"):

            x = x.to(device)
            # y = y.to(device)

            encoded_x = vae.encoder(x)
            y_hat = classifier(encoded_x)
            x_hat = vae(x)

            classifier_loss = classifier_criterion(y_hat, y)
            vae_loss = ((x - x_hat)**2).sum()
            # loss = vae_loss + classifier_loss + vae.encoder.kl

            loss = classifier_loss + vae.encoder.kl

            _, y_pred = torch.max(y_hat, 1)
            p = precision_score(y.numpy(), y_pred.numpy(), average='micro')
            acc = accuracy_score(y.numpy(), y_pred.numpy())

            val_loss += loss.item()
            entropy += vae_loss.item()
            precision += p.item()
            accuracy += acc

    len_data = len(data.dataset)

    val_loss /= len_data
    entropy /= len_data
    precision /= len_data
    accuracy /= len_data

    return val_loss, entropy, precision, accuracy


def test_epoch(vae: VariationalAutoencoder, classifier: Classifier, data):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    classifier.eval()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    test_loss = 0.0
    entropy = 0.0
    precision = 0.0
    accuracy = 0.0

    with torch.no_grad(): # No need to track the gradients
        for x, y in tqdm(data, desc="testing epochs", unit="batch"):

            x = x.to(device)
            # y = y.to(device)

            encoded_x = vae.encoder(x)
            y_hat = classifier(encoded_x)
            x_hat = vae(x)

            classifier_loss = classifier_criterion(y_hat, y)

            vae_loss = ((x - x_hat)**2).sum()
            # loss = vae_loss + classifier_loss + vae.encoder.kl

            loss = classifier_loss + vae.encoder.kl

            _, y_pred = torch.max(y_hat, 1)
            p = precision_score(y.numpy(), y_pred.numpy(), average='micro')
            acc = accuracy_score(y.numpy(), y_pred.numpy())

            print(f"\ny = {y}")
            print(f"y_pred = {y_pred}")

            test_loss += loss.item()
            entropy = vae_loss.item()
            precision += p.item()
            accuracy += acc

    len_data = len(data.dataset)

    test_loss /= len_data
    entropy /= len_data
    precision /= len_data
    accuracy /= len_data

    return test_loss, entropy, precision, accuracy


def main():
    # Hyperparameters
    latent_dims = DEFAULT_LATENT_SIZE
    n_epochs = DEFAULT_N_EPOCHS

    midi_dataset = MidiDataset("data/augmented_data")

    # data_report(midi_dataset, columns=['notes', 'note_lengths', 'key', 'bpm'], title="Raw Data")
    midi_dataset = preprocess_midi(midi_dataset)
    # data_report(midi_dataset, columns=['notes', 'label'], title="Preprocessed Data")

    vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
    classifier = Classifier(latent_dims).to(device)

    vae = train(vae, classifier, midi_dataset, n_epochs)
    torch.save(vae.state_dict(), 'music_vae.pt')
    torch.save(classifier.state_dict(), 'music_classifier.pt')


if __name__ == '__main__':
    main()
