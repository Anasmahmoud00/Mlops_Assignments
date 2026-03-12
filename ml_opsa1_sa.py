

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import mlflow

# ── MLflow Setup ────────────────────────────────────────
mlflow.set_experiment("Assignment3_Anas")

# ── Parameters ──────────────────────────────────────────
batch_size = 64
lr = 0.0002
epochs = 50

# ── Load Kaggle CSV (Using 10k rows to save memory) ───────
df = pd.read_csv("./digit-recognizer/train.csv", nrows=10000)
images = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
images = (images / 127.5) - 1.0
loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=True)

# ── Simple Generator & Discriminator ────────────────────
G = nn.Sequential(
    nn.Linear(64, 256), nn.ReLU(),
    nn.Linear(256, 784), nn.Tanh()
)
D = nn.Sequential(
    nn.Linear(784, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 1),   nn.Sigmoid()
)
# ── Phase 2: Setup Loss & Optimization ────────────────────
loss_fn = nn.BCELoss()

# ── Phase 3: Hyperparameter Search ────────────────────────
learning_rates = [0.01, 0.001, 0.0002, 0.00005]

for lr in learning_rates:
    # Reset model and optimizer for each experiment
    G = nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 784), nn.Tanh()
    )
    D = nn.Sequential(
        nn.Linear(784, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 1),   nn.Sigmoid()
    )
    opt_G = torch.optim.Adam(G.parameters(), lr=lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr)
    
    g_loss_history, d_loss_history = [], []

    with mlflow.start_run(run_name=f"LR_{lr}"):
        mlflow.set_tag("student_id", "202201304")
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr
        })
        
        for epoch in range(epochs):
            for (real,) in loader:
                b = real.size(0)
                z = torch.randn(b, 64)
                fake = G(z).detach()
                loss_d = loss_fn(D(real), torch.ones(b, 1)) + \
                        loss_fn(D(fake), torch.zeros(b, 1))
                opt_D.zero_grad(); loss_d.backward(); opt_D.step()

                z = torch.randn(b, 64)
                loss_g = loss_fn(D(G(z)), torch.ones(b, 1))
                opt_G.zero_grad(); loss_g.backward(); opt_G.step()

            # ── End of Epoch Logging ──
            g_loss_history.append(loss_g.item())
            d_loss_history.append(loss_d.item())

            with torch.no_grad():
                real_preds = (D(real) > 0.5).float()
                fake_imgs_for_acc = G(torch.randn(real.size(0), 64))
                fake_preds = (D(fake_imgs_for_acc) < 0.5).float()
                d_accuracy = (torch.cat([real_preds, fake_preds]).mean()).item()

            mlflow.log_metric("d_loss", loss_d.item(), step=epoch)
            mlflow.log_metric("g_loss", loss_g.item(), step=epoch)
            mlflow.log_metric("d_accuracy", d_accuracy, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"LR: {lr} | Epoch {epoch+1}/{epochs} | D Loss: {loss_d.item():.3f} | G Loss: {loss_g.item():.3f}")

        # ── Save Models & Visualizations ──
        mlflow.pytorch.log_model(G, "generator_model")
        mlflow.pytorch.log_model(D, "discriminator_model")
        
        # Plot generated digits
        G.eval()
        with torch.no_grad():
            samples = G(torch.randn(16, 64)).view(16, 28, 28).numpy()
            samples = (samples + 1) / 2.0

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i], cmap="gray"); ax.axis("off")
        plt.suptitle(f"Generated Digits (LR={lr})", fontsize=11)
        img_name = f"digits_lr_{lr}.png"
        plt.savefig(img_name)
        mlflow.log_artifact(img_name)
        plt.close(fig)

        # Plot loss curves
        plt.figure(figsize=(8, 4))
        plt.plot(g_loss_history, label="G Loss")
        plt.plot(d_loss_history, label="D Loss")
        plt.title(f"Loss Curves (LR={lr})")
        plt.legend()
        loss_name = f"loss_lr_{lr}.png"
        plt.savefig(loss_name)
        mlflow.log_artifact(loss_name)
        plt.close()

print("\nDone! All 4 search runs completed. Check MLflow UI.")