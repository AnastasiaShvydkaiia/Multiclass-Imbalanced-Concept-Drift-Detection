from river import metrics

class MetricsTracker:
    def __init__(self):
        self.en_accuracy = metrics.Accuracy()
        self.g_mean = metrics.GeometricMean()
        
        self.current_step = 0

        self.stable_steps = 0
        self.false_alarms = 0 

        self.injected_drifts = [] 
        self.detected_drift_ids = set() 
        self.all_delays = [] 

    def update(self, y_true, y_pred, drift_detected=False, manual_drifts=None):
        if y_pred is not None:
            self.en_accuracy.update(y_true, y_pred)
            self.g_mean.update(y_true, y_pred)

        if manual_drifts is not None:
            self.injected_drifts = manual_drifts

        if drift_detected:
            # find closest true drift not yet matched
            closest_idx = None
            closest_dist = float("inf")
            for i, drift_step in enumerate(self.injected_drifts):
                if i in self.detected_drift_ids:
                    continue
                dist = self.current_step - drift_step
                # only forward matching (after drift happens)
                if dist >= 0 and dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            # valid detection
            if closest_idx is not None: 
                self.detected_drift_ids.add(closest_idx)
                delay = closest_dist
                self.all_delays.append(delay)
            # false alarms
            else:
                self.false_alarms += 1

        self.current_step += 1

    def calculate_hdd(self):
        total_inj = len(self.injected_drifts)
        ad = len(self.detected_drift_ids) / total_inj if total_inj > 0 else 1.0
        
        real_detects = len(self.detected_drift_ids)
        total_signals = real_detects + self.false_alarms
        a0 = real_detects / total_signals if total_signals > 0 else 1.0

        if a0 > 0 and ad > 0:
            return 2 / ((1 / a0) + (1 / ad))
        return 0.0

    def get_metrics(self):
        avg_delay = sum(self.all_delays) / len(self.all_delays) if self.all_delays else 0
        return {
            "en_accuracy": self.en_accuracy.get(),
            "g_mean": self.g_mean.get(),
            "avg_detection_delay": avg_delay,
            "h_dd": self.calculate_hdd(),
            "drifts_detected": len(self.detected_drift_ids),
            "total_drifts": len(self.injected_drifts),
            "false_alarms": self.false_alarms
        }


