import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import CFG
from src.model import SiameseResidualMLP
from src.dataset import TripletFSC147DatasetV2

if __name__ == '__main__':
    os.makedirs(CFG.WEIGHTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = TripletFSC147DatasetV2(CFG.JSON_PATH, CFG.SPLIT_PATH, CFG.IMAGE_DIR, CFG.FEATURES_DIR, split_name='train')
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = TripletFSC147DatasetV2(CFG.JSON_PATH, CFG.SPLIT_PATH, CFG.IMAGE_DIR, CFG.FEATURES_DIR, split_name='val')
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SiameseResidualMLP(input_dim=CFG.FEATURE_DIM, latent_dim=CFG.LATENT_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.TripletMarginLoss(margin=CFG.MARGIN, p=2)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(CFG.EPOCHS):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [TRAIN]")
        for anchors, positives, negatives in train_bar:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            optimizer.zero_grad()
            loss = criterion(model(anchors), model(positives), model(negatives))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.MAX_GRAD_NORM)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchors, positives, negatives in val_loader:
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                val_loss += criterion(model(anchors), model(positives), model(negatives)).item()
                
        avg_val = val_loss / len(val_loader)
        print(f"Val Loss = {avg_val:.4f}")
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CFG.BEST_MODEL_PATH)