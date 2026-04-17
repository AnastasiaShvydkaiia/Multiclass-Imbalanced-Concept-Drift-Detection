import random
import numpy as np
from river.datasets import synth

def create_stream(drift_type='no', 
                  drift_speed='sudden', 
                  drift_intensity="global",
                  imbalance_ratio={}, 
                  injection_points=[], 
                  n_samples=5000,
                  n_features=20):
    if imbalance_ratio == "10:1:1":
        probs = {0: 0.8, 1: 0.1, 2: 0.1}
    elif imbalance_ratio == "3:2:1":
        probs = {0: 0.5, 1: 0.3, 2: 0.2}
    elif imbalance_ratio == "5:2:1":
        probs = {0: 0.6, 1: 0.25, 2: 0.15}
    else:
        probs = {0: 1.0, 1: 1.0, 2: 1.0}
    stream = base_rbf(n_samples=n_samples,n_classes=3, n_features=n_features, class_probs=probs)
    gradual = drift_speed == "gradual"
    if drift_type == "real":
        final_stream = real_drift_label_swap(stream, gradual=gradual, injection_points=injection_points)
    elif drift_type =="virtual":
        final_stream = virtual_drift(stream, gradual=gradual,injection_points=injection_points,drift_intensity=drift_intensity)
    else:
        final_stream=stream
    for i, (x, y) in enumerate(final_stream):
        yield x, y, i

def base_rbf(n_samples, n_classes, n_features, class_probs, seed=42):
    
    dataset = synth.RandomRBF(
        seed_model=seed,
        seed_sample=seed,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=12
    )
    rng = np.random.RandomState(seed)
    count = 0
    for x_dict, y in dataset:
        if count >= n_samples:
            break

        if rng.rand() > class_probs.get(y, 1.0):
            continue
        
        x_array = np.array(list(x_dict.values()))
        yield x_array, y
        
        count += 1

def real_drift_label_swap(stream, gradual=False, width=200, injection_points=[]):
    for i, (x, y) in enumerate(stream):
    
        # Determine current drift
        current_drift_idx = sum(1 for dp in injection_points if i >= dp)
  
        if current_drift_idx > 0:
            last_dp = injection_points[current_drift_idx-1] 
            
            if not gradual:
                alpha = 1.0
            else:
                alpha = min(1.0, (i - last_dp) / width)

            # Label swap
            if random.random() < alpha:
                y = (y + current_drift_idx) % 3
        yield x, y

def virtual_drift(stream, gradual=False, width=200, injection_points=[], drift_intensity="global"):
    for i, (x, y) in enumerate(stream):
        active_points = [dp for dp in injection_points if i >= dp]
        
        if active_points:
            total_multiplier = 1.0
            total_offset = 0.0
            
            for dp in active_points:
                if not gradual:
                    alpha = 1.0 
                else:
                    alpha = min(1.0, (i - dp) / width) 

                total_multiplier += alpha 
                total_offset += alpha 
            
            # local drift and global drift
            if drift_intensity=="global":
                x = x * total_multiplier + total_offset
            else:
                if y==2:
                    x = x * total_multiplier + total_offset
        
        yield x, y
