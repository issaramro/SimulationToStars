import torch
import torch.nn as nn
import numpy as np

def prepare_data(state_data, N, known_idx):
    """
    Prepares input–output pairs for the PINN from full N-body state data.

    For each time steps, the function:
    - Uses positions and velocities of the known bodies (excluding Earth) as inputs X.
    - Uses the full state vector (all positions and velocities, including Earth) as outputs Y.

    Arguments:
    state_data : Array of shape (n_samples, 6*N) containing positions (3*N) and velocities (3*N)
    N : Total number of bodies.
    known_idx : Indices of the known bodies (Earth excluded).

    Returns:
    X : Input array of known bodies' positions and velocities.
    Y : Output array of full system states.
    """
    X = [] # input: positions/velocities of known bodies (all except Earth)
    Y = [] # output: full positions/velocities including Earth
    for row in state_data:
        pos = row[:3*N].reshape((N,3))
        vel = row[3*N:].reshape((N,3))
        known_pos_vel = np.hstack([pos[known_idx], vel[known_idx]]).flatten()
        X.append(known_pos_vel)
        Y.append(row)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)




class PINN_NBody(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for learning N-body dynamics.

    The network maps the observed states of known bodies (planets and sun) to the full system
    state (including Earth). Training is guided by both data loss and a physics loss enforcing
    Newtonian N-body equations.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_hidden=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def nbody_acc(self, state, masses, G, N):
        """
        Computes time derivatives of the state using Newtonian N-body dynamics.

        Given positions and velocities, this function computes gravitational
        accelerations for all bodies and returns the full time derivative
        (velocities and accelerations).

        Arguments:
        state : Predicted state tensor of shape (batch, 6N).
        masses : Masses of the N bodies.
        G : the Gravitational constant.
        N : Number of bodies (planets + sun)

        Returns:
        dydt : Time-derivative of the state (velocities and accelerations).
        """
        # state: [batch, 6N]
        # positions: [batch, N,3]
        # velocities: [batch,N,3]
        batch = state.shape[0]
        pos = state[:, :3*N].reshape(batch, N, 3)
        vel = state[:, 3*N:].reshape(batch, N, 3)
        acc = torch.zeros_like(pos)
        for i in range(N):
            ri = pos[:, i:i+1, :]  # [batch,1,3]
            rj = pos  # [batch,N,3]
            rij = rj - ri  # [batch,N,3]
            dist3 = torch.norm(rij, dim=-1)**3 + 1e-10  # to avoid division by zero
            dist3[:, i] = 1.0  # ignore self interaction
            ai = torch.sum(G * torch.tensor(masses, dtype=torch.float32).reshape(1,N,1) * rij / dist3.unsqueeze(-1), dim=1)
            acc[:, i, :] = ai
        dydt = torch.cat([vel.reshape(batch, -1), acc.reshape(batch, -1)], dim=1)
        return dydt

    def loss_fn(self, y_pred, y_true, earth_idx, N, masses, G, λ_phys=1.0, data_loss=True):
        """
        Computes the total loss as a combination of data loss and physics loss.

        - Data loss enforces agreement between predicted and true Earth
          position and velocity.
        - Physics loss enforces consistency with the N-body equations of motion
          using finite-difference time derivatives.

        Arguments:
        y_pred : Predicted full system states.
        y_true : Ground-truth full system states.
        earth_idx : Index of Earth in the N-body system.
        N : Number of bodies.
        masses : array of the Masses of all bodies
        G : teh gravitational constant.
        λ_phys : Weight of the physics loss.
        data_loss : Whether to include data loss

        Returns:
        total_loss : Combined data and physics loss.
        loss_data :h data loss.
        loss_phys : Physics (ODEs residual) loss
        """
        # Data loss 
        loss_data = torch.tensor(0.0)
        if data_loss:
            earth_pos_vel_idx = np.r_[earth_idx*3:(earth_idx+1)*3, 3*N + earth_idx*3:3*N + (earth_idx+1)*3]
            y_true_earth = y_true[:, earth_pos_vel_idx]
            y_pred_earth = y_pred[:, earth_pos_vel_idx]
            loss_data = nn.MSELoss()(y_pred_earth, y_true_earth)

        # Physics loss
        # approximate time derivative with finite differences from y_pred itself
        # uniform dt = 1 day
        dydt_pred = self.nbody_acc(y_pred, masses, G, N )
        dt = 1.0
        dydt_true = (y_pred[1:] - y_pred[:-1]) / dt
        dydt_pred_cut = dydt_pred[:-1]
        loss_phys = nn.MSELoss()(dydt_pred_cut, dydt_true)

        return loss_data + λ_phys*loss_phys, loss_data, loss_phys




class PINN_NBody_unk(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for N-body dynamics with unknown
    physical parameters.

    This model simultaneously learns:
    - The full system state from partial observations.
    - The gravitational constant G and all body masses as learnable parameters.

    G and masses are optimized in log-space to enforce physical positivity,
    and are initialized far from their true values for inverse discovery.
    """
    def __init__(self, input_dim, output_dim, G_initial, masses_initial, hidden_dim=128, n_hidden=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

        # Learnable physical parameters 
        # log-parameterization to enforce positivity
        self.log_G = nn.Parameter(torch.tensor(np.log(G_initial), dtype=torch.float32))
        self.log_masses = nn.Parameter(torch.log(torch.tensor(masses_initial, dtype=torch.float32)))

    def forward(self, x):
        return self.model(x)

    def nbody_acc(self, state, N):
        """
        Computes time derivatives using learned N-body gravitational dynamics.

        Uses the current learned values of G and the masses (via exponential
        of log-parameters) to compute accelerations for each body and returns
        the full state derivative.

        Arguments:
        state : Predicted state tensor of shape (batch, 6N)
        N : Number of bodies.

        Returns:
        dydt :Time derivative of the state (velocities and accelerations).
        """
        batch = state.shape[0]
        device = state.device

        # Learned physical parameters
        G_eff = torch.exp(self.log_G)                          
        masses_eff = torch.exp(self.log_masses).to(device)     
        masses_broadcast = masses_eff.view(1, N, 1)            

        pos = state[:, :3*N].reshape(batch, N, 3)      # [batch, N, 3]
        vel = state[:, 3*N:].reshape(batch, N, 3)    # [batch, N, 3]
        acc = torch.zeros_like(pos)      # [batch, N, 3]

        for i in range(N):
            ri = pos[:, i:i+1, :]       
            rj = pos                  
            rij = rj - ri             
            dist3 = torch.norm(rij, dim=-1)**3 + 1e-10  # [batch, N]
            # ignore self-interaction
            dist3[:, i] = 1.0

            # Sum over j != i: G m_j (r_j - r_i) / |r_j - r_i|^3
            ai = torch.sum(
                G_eff * masses_broadcast * rij / dist3.unsqueeze(-1),
                dim=1
            )  # [batch, 3]

            acc[:, i, :] = ai

        dydt = torch.cat([vel.reshape(batch, -1), acc.reshape(batch, -1)], dim=1)
        return dydt

    def loss_fn(self, y_pred, y_true, N, earth_idx, λ_phys=1.0):
        """
        Computes the total PINN loss with unknown physical parameters.

        - Data loss constrains the Earth position and velocity using ground truth.
        - Physics loss enforces the learned G and masses to satisfy the N-body ODEs
          through finite-difference time derivatives.

        Arguments:
        y_pred : Predicted system states/data
        y_true : Ground-truth system states.
        N : Number of bodies.
        earth_idx : Index of Earth in the system.
        λ_phys : Weight of the physics loss.

        Returns:
        loss_total : Total combined loss.
        loss_data : Earth data loss.
        loss_phys : Physics consistency loss
        """
       # data loss
        earth_pos_vel_idx = np.r_[
            earth_idx*3:(earth_idx+1)*3,          # position indices (x,y,z)
            3*N + earth_idx*3:3*N + (earth_idx+1)*3  # velocity indices (vx,vy,vz)
        ]
        earth_pos_vel_idx = torch.tensor(earth_pos_vel_idx, dtype=torch.long, device=y_true.device)

        y_true_earth = y_true[:, earth_pos_vel_idx]
        y_pred_earth = y_pred[:, earth_pos_vel_idx]
        loss_data = nn.MSELoss()(y_pred_earth, y_true_earth)

        # physics loss 
        dydt_pred = self.nbody_acc(y_pred, N)        
        # approximate time derivative with finite differences from y_pred itself
        # uniform dt = 1 day
        dt = 1.0
        dydt_true = (y_pred[1:] - y_pred[:-1]) / dt     # [T-1, 6N]
        dydt_pred_cut = dydt_pred[:-1]                  # [T-1, 6N]
        loss_phys = nn.MSELoss()(dydt_pred_cut, dydt_true)

        loss_total = loss_data + λ_phys * loss_phys
        return loss_total, loss_data, loss_phys

    def get_physical_params(self):
        """
        Return learned G and masses as numpy arrays (in physical units)
        """
        with torch.no_grad():
            G_eff = torch.exp(self.log_G).cpu().item()
            masses_eff = torch.exp(self.log_masses).cpu().numpy()
        return G_eff, masses_eff



# Model: encoder LSTM + decoder LSTMCell (autoregressive)
class Seq2SeqLSTM(nn.Module):
    """
    sequence-to-sequence RNN for multivariate time-series
    forecasting consisting of an LSTM encoder and an autoregressive LSTMCell decoder

  Architecture
    1. Encoder (LSTM):
        - Processes the input time-series sequence `src_seq` of shape (B, T, F),
          where:
              B = batch size
              T = input sequence length
              F = number of features per timestep
        - The encoder encodes the full input sequence into its hidden
          and cell states (h_T, c_T) that represents the learned latent
          representation of the dynamics

    2. Decoder (LSTMCell):
        - Generate predictions autoregressively over `pred_horizon` future time steps
        - Use a single LSTMCell instead of a full LSTM to allow timestep by timestep and optional teacher forcing.
        - The decoder is initialized using the encoder’s final states.
        - At each prediction time step it receives a feature vector as input: either the ground-truth
        next step (teacher forcing) or its own previous prediction + it the prroduces a new hidden state,
        then a predicted feature vector.

    3. Output Layer:
        - A fully connected layer that maps the decoder hidden state to F output
          features that represent the predicted next timestep.

    This autoregressive decoder proved that it allows the model to learn long-term temporal
    dependencies while optional teacher forcing stabilizes the training and
    control error accumulation early in the training.
    """

    def __init__(self, n_features, hidden_size=128, n_layers=2, pred_horizon=10):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.pred_horizon = pred_horizon
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=n_features, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, src_seq, target_seq=None, teacher_forcing_prob=0.0):
        """
        Arguments:
        src_seq :The input sequence to encode. The encoder processes this entire
            sequence and compresses it into its final hidden and cell states.

        target_seq : Ground-truth future sequence used for teacher forcing during training.
            If provided and teacher_forcing_prob > 0, the decoder may replace its
            own predicted output with the corresponding entry from target_seq.

        teacher_forcing_prob : Probability of using teacher forcing at each prediction step.

        Returns:
        predictions: the predicted future sequence over the forecasting horiozn
        """
        _, (h_n, c_n) = self.encoder(src_seq)
        h = h_n[-1]  
        c = c_n[-1]
        outputs = []
        decoder_input = src_seq[:, -1, :]
        for t in range(self.pred_horizon):
            h, c = self.decoder_cell(decoder_input, (h, c))
            out = self.fc(h)
            outputs.append(out.unsqueeze(1))
            if (target_seq is not None) and (np.random.rand() < teacher_forcing_prob):
                decoder_input = target_seq[:, t, :]
            else:
                decoder_input = out.detach()
        return torch.cat(outputs, dim=1)