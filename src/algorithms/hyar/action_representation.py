# Adapted from HyAR (TJU-DRL-LAB/self-supervised-rl), Apache-2.0; see LICENSE and NOTICE.md here.
# TODO: s discrete continue ->s"
import numpy as np
import torch
from torch import float32
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from src.algorithms.hyar.nn_utils import NeuralNet, pairwise_distances
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as functional


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, action_embedding_dim, parameter_action_dim, latent_dim, max_action,
                 hidden_size=256, delta_scale=1.0):
        super(VAE, self).__init__()
        # delta_scale: the delta-state head outputs delta_scale * tanh(.), so it can represent
        # state-deltas larger than 1 (the raw tanh cap). Default 1.0 == original HyAR behavior.
        # On unstable systems (e.g. LQR A=1.1 I) normalized deltas routinely exceed 1, which
        # would otherwise floor recon_s and degrade LSC/RSC. Set from warmup-data delta stats.
        self.delta_scale = float(delta_scale)

        # embedding table
        init_tensor = torch.rand(action_dim,
                                 action_embedding_dim) * 2 - 1  # Don't initialize near the extremes.
        self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)
        # (porting) removed noisy debug print of the full embedding tensor
        self.e0_0 = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.e0_1 = nn.Linear(parameter_action_dim, hidden_size)

        self.e1 = nn.Linear(hidden_size, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d0_0 = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.d0_1 = nn.Linear(latent_dim, hidden_size)
        self.d1 = nn.Linear(hidden_size, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)

        self.parameter_action_output = nn.Linear(hidden_size, parameter_action_dim)

        self.d3 = nn.Linear(hidden_size, hidden_size)

        self.delta_state_output = nn.Linear(hidden_size, state_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action, action_parameter):

        z_0 = F.relu(self.e0_0(torch.cat([state, action], 1)))
        z_1 = F.relu(self.e0_1(action_parameter))
        z = z_0 * z_1

        z = F.relu(self.e1(z))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)

        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u, s = self.decode(state, z, action)

        return u, s, mean, std

    def decode(self, state, z=None, action=None, clip=None, raw=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device)
            if clip is not None:
                z = z.clamp(-clip, clip)
        v_0 = F.relu(self.d0_0(torch.cat([state, action], 1)))
        v_1 = F.relu(self.d0_1(z))
        v = v_0 * v_1
        v = F.relu(self.d1(v))
        v = F.relu(self.d2(v))

        parameter_action = self.parameter_action_output(v)

        v = F.relu(self.d3(v))
        s = self.delta_state_output(v)

        if raw: return parameter_action, s
        return self.max_action * torch.tanh(parameter_action), self.delta_scale * torch.tanh(s)


class Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 parameter_action_dim,
                 reduced_action_dim=2,
                 reduce_parameter_action_dim=2,
                 embed_lr=1e-4,
                 delta_scale=1.0,
                 vae_hidden_size=256,
                 kl_weight=0.5,
                 persistent_optimizer=False,
                 ):
        super(Action_representation, self).__init__()
        # Upstream builds a fresh Adam optimizer at every VAE step, which makes each step a
        # fixed-size sign update; persistent_optimizer keeps one Adam across steps instead.
        self.kl_weight = float(kl_weight)
        self.persistent_optimizer = persistent_optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.parameter_action_dim = parameter_action_dim
        self.reduced_action_dim = reduced_action_dim
        self.reduce_parameter_action_dim = reduce_parameter_action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Action embeddings to project the predicted action into original dimensions
        # latent_dim=action_dim*2+parameter_action_dim*2
        self.latent_dim = self.reduce_parameter_action_dim
        self.embed_lr = embed_lr
        self.vae = VAE(state_dim=self.state_dim, action_dim=self.action_dim,
                       action_embedding_dim=self.reduced_action_dim, parameter_action_dim=self.parameter_action_dim,
                       latent_dim=self.latent_dim, max_action=1.0,
                       hidden_size=vae_hidden_size, delta_scale=delta_scale).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

    def discrete_embedding(self, ):
        emb = self.vae.embeddings

        return emb

    def unsupervised_loss(self, s1, a1, a2, s2, sup_batch_size, embed_lr):

        a1 = self.get_embedding(a1).to(self.device)

        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)

        vae_loss, recon_loss_d, recon_loss_c, KL_loss = self.train_step(s1, a1, a2, s2, sup_batch_size, embed_lr)
        return vae_loss, recon_loss_d, recon_loss_c, KL_loss

    def loss(self, state, action_d, action_c, next_state, sup_batch_size):

        recon_c, recon_s, mean, std = self.vae(state, action_d, action_c)

        recon_loss_s = F.mse_loss(recon_s, next_state, reduction='mean')
        recon_loss_c = F.mse_loss(recon_c, action_c, reduction='mean')

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # vae_loss = 0.25 * recon_loss_s + recon_loss_c + 0.5 * KL_loss
        # vae_loss = 0.25 * recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss  #best
        vae_loss = recon_loss_s + 2.0 * recon_loss_c + self.kl_weight * KL_loss
        # print("vae loss",vae_loss)
        # return vae_loss, 0.25 * recon_loss_s, recon_loss_c, 0.5 * KL_loss
        # return vae_loss, 0.25 * recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss #best
        return vae_loss, recon_loss_s, 2.0 * recon_loss_c, self.kl_weight * KL_loss

    def train_step(self, s1, a1, a2, s2, sup_batch_size, embed_lr=1e-4):
        state = s1
        action_d = a1
        action_c = a2
        next_state = s2
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = self.loss(state, action_d, action_c, next_state,
                                                                  sup_batch_size)

        # 更新VAE
        if not self.persistent_optimizer:
            self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=embed_lr)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return vae_loss.cpu().data.numpy(), recon_loss_s.cpu().data.numpy(), recon_loss_c.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def select_parameter_action(self, state, z, action):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
            action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
            action_c, state = self.vae.decode(state, z, action)
        return action_c.cpu().data.numpy().flatten()

    # def select_delta_state(self, state, z, action):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #         z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
    #         action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
    #         action_c, state = self.vae.decode(state, z, action)
    #     return state.cpu().data.numpy().flatten()
    def select_delta_state(self, state, z, action):
        with torch.no_grad():
            action_c, state = self.vae.decode(state, z, action)
        return state.cpu().data.numpy()

    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.vae.embeddings[action]
        action_emb = torch.tanh(action_emb)
        return action_emb

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm
        embeddings = self.vae.embeddings
        embeddings = torch.tanh(embeddings)
        action = action.to(self.device)
        # compute similarity probability based on L2 norm
        similarity = - pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score
        return similarity

        # 获得最优动作，输出于embedding最相近的action 作为最优动作.

    def select_discrete_action(self, action):
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        # print("pos",pos,len(pos))
        if len(pos) == 1:
            return pos.cpu().item()  # data.numpy()[0]
        else:
            # print("pos.cpu().item()", pos.cpu().numpy())
            return pos.cpu().numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
        # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
        # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)

    def get_c_rate(self, s1, a1, a2, s2, batch_size=100, range_rate=5):
        a1 = self.get_embedding(a1).to(self.device)
        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)
        recon_c, recon_s, mean, std = self.vae(s1, a1, a2)
        # print("recon_s",recon_s.shape)
        z = mean + std * torch.randn_like(std)
        z = z.cpu().data.numpy()
        c_rate = self.z_range(z, batch_size, range_rate)
        # print("s2",s2.shape)

        recon_s_loss = F.mse_loss(recon_s, s2, reduction='mean')

        # recon_s = abs(np.mean(recon_s.cpu().data.numpy()))
        return c_rate, recon_s_loss.detach().cpu().numpy()

    def z_range(self, z, batch_size=100, range_rate=5):
        """Per latent dimension, the values `border` places in from each end of the sorted batch.

        Same result as upstream's hand-written version (which only handled latent sizes
        3, 4, 6, 8, 10, 12 and 16), for any latent size: a list of [up, down] per dimension.
        """
        border = int(range_rate * (batch_size / 100))
        zs = np.sort(np.asarray(z), axis=0)
        return tuple([zs[-border - 1, i], zs[border, i]] for i in range(zs.shape[1]))
