from river.drift import ADWIN, KSWIN
import torch
import torch.nn as nn
import torch.nn.functional as F

class DriftDetector:
    def __init__(self, name,n_features=10, n_classes=3):
        self.name = name
        self.n_features = n_features 
        self.n_classes= n_classes
        self.detector = self._create_detector()
        self.drift_detected = False 

    def _create_detector(self):
        if self.name == "KSWIN":  
            return KSWIN(alpha=0.0001, window_size=300, stat_size=30)
        elif self.name=="DHAE":
            return DHAE(n_features= self.n_features,n_classes=self.n_classes, lambda_p=5) 
        else:
            return ADWIN(delta=0.001) 
        
    def update(self, smooth_error, probas, x):
        self.drift_detected = False 
        if self.name=="KSWIN":
            self.drift_detector= self.detector.update(x[0])
            self.drift_detected = self.detector.drift_detected
        elif self.name == "DHAE":
            self.drift_detected = self.detector.update(x, probas)
        else: 
            self.detector.update(smooth_error)
            self.drift_detected = self.detector.drift_detected

    def detected(self):
        return self.drift_detected

    def pretrain(self, train_X, epochs=15):
        if self.name=="DHAE":
            if not hasattr(self, "trained"):
                self.detector.pretrain(train_X,epochs)
                self.trained = True

class Autoencoder(nn.Module):
    def __init__(self, x_dim, p_dim, hidden_dim=16, latent_dim=8):
        super().__init__()

        self.x_dim = x_dim
        self.p_dim = p_dim
        self.input_dim = x_dim + p_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Feature decoder
        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

        # Probability decoder
        self.decoder_p = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p_dim),
            nn.Softmax(dim=1) 
        )

    def forward(self, x, p):
        z = torch.cat([x, p], dim=1)
        h = self.encoder(z)

        x_hat = self.decoder_x(h)
        p_hat = self.decoder_p(h)

        return x_hat, p_hat
    
class DHAE: # Dual Head AutoEncoder
    def __init__(
        self,
        n_features,
        n_classes,
        hidden_dim=16,
        latent_dim=8,
        lr=1e-3,
        lambda_x=0.5,
        lambda_p=0.5, 
        adwin_delta=0.002
    ):
        self.model = Autoencoder(n_features, n_classes, hidden_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.lambda_x = lambda_x
        self.lambda_p = lambda_p

        self.adwin = ADWIN(delta=adwin_delta)

        self.cooldown = 200
        self.cooldown_counter = 0

        self.n_classes=n_classes

    def _compute_loss(self, x, x_hat, p, p_hat):
        loss_x = F.mse_loss(x_hat, x)
        loss_p = F.mse_loss(p_hat, p)

        total_loss = self.lambda_x * loss_x + self.lambda_p * loss_p
        return total_loss 

    def update(self, x, probas):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False 
        
        if not probas:
            probas = {c: 1.0 / self.n_classes for c in range(self.n_classes)}
        p_list = [probas.get(c, 1e-3) for c in range(self.n_classes)]
        if sum(p_list) == 0:
            p_list = [1.0 / self.n_classes] * self.n_classes

        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        p_tensor = torch.FloatTensor(p_list).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            x_hat, p_hat = self.model(x_tensor, p_tensor)

            total_loss= self._compute_loss(
                x_tensor, x_hat, p_tensor, p_hat
            )

        score = total_loss.item()

        # Drift detection 
        self.adwin.update(score)

        if self.adwin.drift_detected:
            self.cooldown_counter = self.cooldown
            return True 

        # Training only when stable
        if score < 0.09:
            self._train_step(x_tensor, p_tensor)  

        return False 

    def _train_step(self, x, p):
        self.model.train()
        self.optimizer.zero_grad()

        x_hat, p_hat = self.model(x, p)

        total_loss = self._compute_loss(x, x_hat, p, p_hat)

        total_loss.backward()
        self.optimizer.step()

    def pretrain(self, X, P, epochs=5):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for x_raw, p_raw in zip(X, P):
                x = torch.FloatTensor(x_raw).unsqueeze(0)
                p = torch.FloatTensor(p_raw).unsqueeze(0)

                self.optimizer.zero_grad()

                x_hat, p_hat = self.model(x, p)
                loss = self._compute_loss(x, x_hat, p, p_hat)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: {total_loss / len(X):.6f}")

