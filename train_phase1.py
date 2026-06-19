# train_phase1.py
"""Phase 1 DMC training script. Run: python train_phase1.py"""
import torch
from model.net import PolicyValueNet
from train.dmc import run_dmc

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    model = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    run_dmc(
        deck=DECK,
        model=model,
        optimizer=optimizer,
        device=device,
        save_path='model.pt',
        n_iterations=500,
        games_per_iter=10,
        batch_size=64,
        buffer_capacity=50_000,
        epsilon_start=0.5,
        epsilon_end=0.05,
        search_count=10,
        eval_every=20,
        eval_games=50,
        gate_winrate=0.80,
    )


if __name__ == '__main__':
    main()
