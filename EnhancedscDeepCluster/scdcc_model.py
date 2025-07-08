import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model_layers import ZINBLoss, MeanAct, DispAct, SupConLoss
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from sklearn.mixture import GaussianMixture


def attentionNetwork(d_model, type, n_head=1, num_layers=1, dim_ff=64):
    attention_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_ff)
    attention = TransformerEncoder(attention_layer, num_layers=num_layers)
    return attention

def neuralNetwork(layers, type, use_layernorm=False, use_batchnorm=False, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if use_layernorm:
            net.append(nn.LayerNorm(layers[i]))
        elif use_batchnorm:
            net.append(nn.BatchNorm1d(layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

def euclidean_dist(x, y):
    return torch.sum(torch.square(x - y), dim=1)

class scDeepCluster(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], use_attention=False, use_layernorm=False, use_batchnorm=False, dropout=0.8,
                 activation="relu", sigma=1., alpha=1., gamma=1., device="cuda"):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.use_attention = use_attention 
        self.device = device

        self.dropoutLayer = nn.Dropout(p=dropout)
        self.encoder = neuralNetwork([input_dim]+encodeLayer, type="encode", activation=activation, use_layernorm=use_layernorm, use_batchnorm=use_batchnorm)
        self.attention = attentionNetwork(encodeLayer[-1], "attention")
        self.decoder = neuralNetwork([z_dim]+decodeLayer, type="decode", activation=activation, use_layernorm=use_layernorm, use_batchnorm=use_batchnorm)

        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.activation == "relu":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif self.activation == "sigmoid":
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            m.weight.data = m.weight.data.double()
            if m.bias is not None:
                m.bias.data = m.bias.data.double()

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

            m.weight.data = m.weight.data.double()
            m.bias.data = m.bias.data.double()


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        if x.dtype != torch.float64:
            x = x.double()
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        if self.use_attention:
            h = h.unsqueeze(0)
            h = self.attention(h)
            h = h.squeeze(0)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        z0 = self._enc_mu(self.encoder(x))
        return z0, _mean, _disp, _pi
    
    def forwardCL(self, x):
        h = self.dropoutLayer(x)
        h = self.encoder(h)
        if self.use_attention:
            h = h.unsqueeze(0)
            h = self.attention(h)
            h = h.squeeze(0)
        z = F.normalize(self._enc_mu(h), dim = 1)
        return z

    def forward(self, x):
        z0, _mean, _disp, _pi = self.forwardAE(x)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        num_batch = math.ceil(X.shape[0] / batch_size)
        for i in range(num_batch):
            x_batch = X[i*batch_size:(i+1)*batch_size]
            inputs = Variable(x_batch).to(self.device)
            z, _, _, _ = self.forwardAE(inputs)
            encoded.append(z.detach())
        return torch.cat(encoded, dim=0).to(self.device)
    
    def encodeBatchCL(self, X, batch_size=256):
        self.eval()
        encoded = []
        num_batch = math.ceil(X.shape[0] / batch_size)
        for i in range(num_batch):
            x_batch = X[i*batch_size:(i+1)*batch_size]
            inputs = Variable(x_batch).to(self.device)
            z = self.forwardCL(inputs)
            encoded.append(z.detach())
        return torch.cat(encoded, dim=0).to(self.device)

    def cluster_loss(self, p, q):
        kld = torch.mean(torch.sum(p * torch.log(p / (q + 1e-6)), dim=-1))
        return self.gamma * kld

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=1e-3, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        self.train()
        dataset = TensorDataset(torch.tensor(X), torch.tensor(X_raw), torch.tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)

        for epoch in range(epochs):
            loss_val = 0
            for x, x_raw, sf in dataloader:
                x = x.to(self.device)
                x_raw = x_raw.to(self.device)
                sf = sf.to(self.device)
                _, mean, disp, pi = self.forwardAE(x)
                loss = self.zinb_loss(x_raw, mean, disp, pi, sf)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item() * len(x)
            print(f"Pretrain epoch {epoch+1:3d}, ZINB loss: {loss_val/X.shape[0]:.8f}")

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def pretrain_CL(self, X, noise_std=0.1, temperature=0.07, batch_size=256, lr=1e-3, epochs=400, cl_save=True, cl_weights='CL_weights.pth.tar'):
        self.train()
        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(self.dropoutLayer.parameters()) + list(self.encoder.parameters()) + list(self._enc_mu.parameters())), lr=lr, amsgrad=True)
        criterion = SupConLoss(temperature=temperature).to(self.device)
        
        for epoch in range(epochs):
            total_loss = 0.0   
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(self.device).double()
                x1 = x_batch + torch.randn_like(x_batch) * noise_std
                x2 = x_batch + torch.randn_like(x_batch) * noise_std
                z1 = self.forwardCL(x1)
                z2 = self.forwardCL(x2)
                features = torch.stack([z1, z2], dim=1)
                loss = criterion(features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)

            print(f"Pretrain epoch {epoch+1:3d}, Contrastive Loss: {total_loss/len(dataset):.8f}")

        if cl_save:
            torch.save({'cl_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, cl_weights)

    def save_checkpoint(self, state, index, filename):
        torch.save(state, os.path.join(filename, f'FTcheckpoint_{index}.pth.tar'))

    def fit(self, X, X_raw, size_factor, n_clusters, init_centroid=None, y=None, y_pred_init=None,
            lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        self.train()
        print("Clustering stage")

        X = torch.tensor(X, dtype=torch.float64)
        X_raw = torch.tensor(X_raw, dtype=torch.float64)
        size_factor = torch.tensor(size_factor, dtype=torch.float64)
        self.mu = Parameter(torch.empty(n_clusters, self.z_dim, device=self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.cpu().numpy())
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float64))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float64))
            self.y_pred = y_pred_init

        self.y_pred_last = self.y_pred

        if y is not None:
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print(f'Initializing k-means: NMI= {nmi:.4f}, ARI= {ari:.4f}')

        num_batch = math.ceil(X.shape[0] / batch_size)
        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).detach()
                self.y_pred = torch.argmax(q, dim=1).cpu().numpy()

                if y is not None:
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print(f'Clustering {epoch+1:3d}: NMI= {nmi:.4f}, ARI= {ari:.4f}')

                delta_label = np.mean(self.y_pred != self.y_pred_last)
                if (epoch > 0 and delta_label < tol) or (epoch % 10 == 0):
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'state_dict': self.state_dict(),
                        'mu': self.mu,
                        'y_pred': self.y_pred,
                        'y_pred_last': self.y_pred_last,
                        'y': y
                    }, epoch+1, save_dir)

                self.y_pred_last = self.y_pred
                if epoch > 0 and delta_label < tol:
                    print(f'delta_label {delta_label:.6f} < tol {tol:.6f}\nStopping training.')
                    break

            train_loss, cluster_loss_val, recon_loss_val = 0.0, 0.0, 0.0
            for i in range(num_batch):
                x = X[i*batch_size:(i+1)*batch_size].to(self.device)
                x_raw = X_raw[i*batch_size:(i+1)*batch_size].to(self.device)
                sf = size_factor[i*batch_size:(i+1)*batch_size].to(self.device)
                p_batch = p[i*batch_size:(i+1)*batch_size].to(self.device)

                optimizer.zero_grad()
                z, q_batch, mean, disp, pi = self.forward(x)
                cluster_loss = self.cluster_loss(p_batch, q_batch)
                recon_loss = self.zinb_loss(x_raw, mean, disp, pi, sf)

                loss = cluster_loss * self.gamma + recon_loss
                loss.backward()
                optimizer.step()

                cluster_loss_val += cluster_loss.item() * len(x)
                recon_loss_val += recon_loss.item() * len(x)
                train_loss += loss.item() * len(x)

            print(f"Epoch {epoch+1:3d}: Total: {train_loss/X.shape[0]:.8f} Clustering Loss: {cluster_loss_val/X.shape[0]:.8f} ZINB Loss: {recon_loss_val/X.shape[0]:.8f}")

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
    
    def fitCL(self, X, X_raw, size_factor, n_clusters, noise_std=0.1, temperature=0.07, init_centroid=None, y=None, y_pred_init=None,
            lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        self.train()
        print("Clustering stage")

        X = torch.tensor(X, dtype=torch.float64)
        X_raw = torch.tensor(X_raw, dtype=torch.float64)
        size_factor = torch.tensor(size_factor, dtype=torch.float64)
        self.mu = Parameter(torch.empty(n_clusters, self.z_dim, device=self.device))
        # optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(self.dropoutLayer.parameters()) + list(self.encoder.parameters()) + list(self._enc_mu.parameters())), lr=lr, amsgrad=True)
        criterion = SupConLoss(temperature=temperature).to(self.device)

        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            data = self.encodeBatchCL(X)
            self.y_pred = kmeans.fit_predict(data.cpu().numpy())
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float64))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float64))
            self.y_pred = y_pred_init

        self.y_pred_last = self.y_pred

        if y is not None:
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print(f'Initializing k-means: NMI= {nmi:.4f}, ARI= {ari:.4f}')

        num_batch = math.ceil(X.shape[0] / batch_size)
        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                latent = self.encodeBatchCL(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).detach()
                self.y_pred = torch.argmax(q, dim=1).cpu().numpy()

                if y is not None:
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print(f'Clustering {epoch+1:3d}: NMI= {nmi:.4f}, ARI= {ari:.4f}')

                delta_label = np.mean(self.y_pred != self.y_pred_last)
                if (epoch > 0 and delta_label < tol) or (epoch % 10 == 0):
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'state_dict': self.state_dict(),
                        'mu': self.mu,
                        'y_pred': self.y_pred,
                        'y_pred_last': self.y_pred_last,
                        'y': y
                    }, epoch+1, save_dir)

                self.y_pred_last = self.y_pred
                if epoch > 0 and delta_label < tol:
                    print(f'delta_label {delta_label:.6f} < tol {tol:.6f}\nStopping training.')
                    break
                

            train_loss, cluster_loss_val, contrast_loss_val = 0.0, 0.0, 0.0
            for i in range(num_batch):
                x_batch = X[i*batch_size:(i+1)*batch_size].to(self.device)
                p_batch = p[i*batch_size:(i+1)*batch_size].to(self.device)

                x1 = x_batch + torch.randn_like(x_batch) * noise_std
                x2 = x_batch + torch.randn_like(x_batch) * noise_std
                z1 = self.forwardCL(x1)
                z2 = self.forwardCL(x2)
                features = torch.stack([z1, z2], dim=1)
                
                optimizer.zero_grad()
                z = self.encoder(x_batch)
                if self.use_attention:
                    z = z.unsqueeze(0)
                    z = self.attention(z)
                    z = z.squeeze(0)
                z = self._enc_mu(z)
                q_batch = self.soft_assign(z)

                cluster_loss = self.cluster_loss(p_batch, q_batch)
                contrast_loss = criterion(features)

                loss = cluster_loss * self.gamma + contrast_loss
                loss.backward()
                optimizer.step()

                cluster_loss_val += cluster_loss.item() * len(x_batch)
                contrast_loss_val += contrast_loss.item() * len(x_batch)
                train_loss += loss.item() * len(x_batch)

            print(f"Epoch {epoch+1:3d}: Total: {train_loss/X.shape[0]:.8f} Clustering Loss: {cluster_loss_val/X.shape[0]:.8f} Contrastive Loss: {contrast_loss_val/X.shape[0]:.8f}")

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch