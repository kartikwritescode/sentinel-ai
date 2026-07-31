# GRU model

import torch
import torch.nn as nn
import numpy as np
import config

class SuspiciousActivityGRU(nn.Module):
    """
    A small Bidirectional GRU that classifies a sequence of feature vectors
    as 'suspicious' or 'normal'.
    
    Input shape:  (batch_size, sequence_length, input_size)
    Output shape: (batch_size, 1) — probability of being suspicious
    
    'Bidirectional' means the GRU reads the sequence forwards AND backwards,
    then combines both passes. For behavior recognition, this means the model
    can learn "this arm movement at frame 5 makes more sense given what 
    happens at frame 15" it has full context.
    """
    def __init__(
            self,
            input_size = config.TIER2_INPUT_SIZE,
            hidden_size = config.TIER2_HIDDEN_SIZE,
            num_layers = config.TIER2_NUM_LAYERS,
            dropout=0.3
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # input shape: (batch, seq, features)
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # After bidirectional GRU, hidden size is doubled (forward + backward)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,1),
            nn.Sigmoid()
        ) 

    def forward(self,x):
        # x shape: (batch, seq_len, input_size)
        gru_out, _ = self.gru(x)

        # gru_out shape: (batch , seq_len , hidden_size*2)

        # take only the last timestep's output 
        last_out = self.gru_out[:,-1,:] #shape: (batch , hidden_size*2)

        return self.classifier(last_out)

    # Training Function

    def train_tier2(X_train,y_train,X_val,y_val,epochs=30):
        """
        X_train: np.ndarray of shape (num_clips, sequence_length, input_size)
        y_train: np.ndarray of shape (num_clips,) with 0/1 labels
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {device}")
        
        model = SuspiciousActivityGRU().to(device)

        criterion = nn.BCELoss() #Binary Cross Entropy Loss

        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4) #adam optimizer with weight decay(L2 Regularization) prevents overfitting

        # converting numpy -> pytorch tensors
        X_tr = torch.FloatTensor(X_train).to(device)
        y_tr = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_v = torch.FloatTensor(X_val).to(device)
        y_v = torch.FloatTensor(y_val).unsqueeze(1).to(device)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train() #enables dropout
            optimizer.zero_grad() # clear gradients from last step
            preds = model(X_tr)
            loss = criterion(preds,y_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_v)
                val_loss = criterion(val_preds, y_v)
                if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.TIER2_MODEL_PATH)
    
        print(f"Best val loss: {best_val_loss:.4f} | Saved to {config.TIER2_MODEL_PATH}")
        return model


# Inference 

class Tier2Inferencer:
    """
    Loads the trained GRU model and runs inference on a window of features.
    
    This is separate from training so your live pipeline only imports
    what it needs (no training code in production).
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = SuspiciousActivityGRU().to(self.device)
        self.model.load_state_dict(
            torch.load(config.TIER2_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()   # always set to eval mode for inference
        
        # Rolling buffer of feature vectors
        self._feature_buffer = []
        self._window_size    = config.FEATURE_WINDOW_FRAMES
    
    def push_features(self, feature_vector):
        """
        Add one frame's feature vector. Returns a prediction when the
        buffer is full, otherwise returns None.
        """
        self._feature_buffer.append(feature_vector)
        
        if len(self._feature_buffer) >= self._window_size:
            # Keep only the last window_size frames
            window = self._feature_buffer[-self._window_size:]
            return self._predict(np.array(window))
        
        return None
    
    def _predict(self, feature_window):
        """
        Args:
            feature_window: np.ndarray of shape (window_size, input_size)
        
        Returns:
            float between 0.0 and 1.0 — probability of suspicious activity
        """
        x = torch.FloatTensor(feature_window).unsqueeze(0).to(self.device)
        # unsqueeze(0) adds the batch dimension: (window, features) → (1, window, features)
        
        with torch.no_grad():
            prob = self.model(x).item()
        
        return prob





