from river.drift import ADWIN, KSWIN, PageHinkley
from river.drift.binary import DDM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold


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
        elif self.name=="ADWIN":
            return ADWIN(delta=0.01) 
        elif self.name=="PH":
            return PageHinkley(delta=0.005, threshold=50, min_instances=30) 
        elif self.name=="DDM":
            return DDM()
        elif self.name=="AEDD":
            return AEDD(n_features=self.n_features, delta=0.005)
        elif self.name=="D3":
            return D3(w=300,threshold=0.75)
        
    def update(self, smooth_error, x):
        self.drift_detected = False 
        if self.name =="KSWIN":
            self.drift_detector= self.detector.update(x[0])
            self.drift_detected = self.detector.drift_detected
        elif self.name in ["D3","AEDD"]:
            self.drift_detector= self.detector.update(x)
            self.drift_detected = self.detector.drift_detected
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


class D3:
    """Implementation of the Discriminative Drift Detector (D3) based on Gözüaçik et al."""
    def __init__(self, w=1000, rho=0.1, threshold=0.75):
        self.w = w                 
        self.rho = rho              
        self.threshold = threshold  
        
        self.size = int(w * (1 + rho)) 
        self.dim = None
        self.buffer = []
        self.step = 0

        self.drift_detected=False

    def update(self, x, **kwargs):
        self.step += 1
        features = np.fromiter(x.values(), dtype=float) if isinstance(x, dict) else np.array(x)
        
        if self.dim is None:
            self.dim = len(features)
            
        self.buffer.append(features)

        if len(self.buffer) < self.size:
            self.drift_detected=False
            return self.drift_detected

        if self._drift_check():
            new_data_start = self.w
            self.buffer = self.buffer[new_data_start:]
            self.drift_detected=True
            return self.drift_detected
        else:
            shift = int(self.w * self.rho)
            self.buffer = self.buffer[shift:]
            self.drift_detected=False
            return self.drift_detected

    def _drift_check(self):
        data = np.array(self.buffer)
        
        S = data[:self.w]
        T = data[self.w:]
        
        labels = np.zeros(len(data))
        labels[:self.w] = 1 
        
        clf = LogisticRegression(solver='liblinear', max_iter=1000)
        predictions = np.zeros(labels.shape)
        
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        try:
            for train_idx, test_idx in skf.split(data, labels):
                X_train, X_test = data[train_idx], data[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                
                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_test)[:, 1]
                predictions[test_idx] = probs
                
            auc_score = AUC(labels, predictions)
            return auc_score > self.threshold
        except Exception as e:
            print(e)
            self.drift_detected=False
            return self.drift_detected

class AEDD: 
    def __init__(self, n_features, hidden_dim=16, latent_dim=8, lr=1e-3, delta=0.002):
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features)
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.adwin = ADWIN(delta=delta)
        
        self.step = 0
        self.warmup_steps = 300
        
        self.is_adapting = False
        self.adaptation_counter = 0
        self.min_adaptation_steps = 100

        self.drift_detected=False

    def _train_step(self, x_tensor):
        self.model.train()
        self.optimizer.zero_grad()
        x_hat = self.model(x_tensor)
        loss = F.mse_loss(x_hat, x_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update(self, x, **kwargs):
        self.step += 1
        
        x_vals = list(x.values()) if isinstance(x, dict) else x
        x_tensor = torch.FloatTensor(x_vals).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            x_hat = self.model(x_tensor)
            current_loss = current_loss = torch.norm(x_tensor - x_hat,p=2,dim=1).item()

        # WARMUP 
        if self.step <= self.warmup_steps:
            self._train_step(x_tensor)
            self.drift_detected=False
            return self.drift_detected

        # ADAPTATION 
        if self.is_adapting:
            self._train_step(x_tensor)
            self.adaptation_counter += 1
            if self.adaptation_counter >= self.min_adaptation_steps:
                self.is_adapting = False
            self.drift_detected=False
            return self.drift_detected
        
        # MONITORING
        self.adwin.update(current_loss)
        
        if self.adwin.drift_detected:
            self.is_adapting = True
            self.adaptation_counter = 0
            self.drift_detected=True
            return self.drift_detected 
        self.drift_detected=False
        return self.drift_detected
