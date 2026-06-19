# train_phase2.py
"""Phase 2 PPO+UPGO League training. Run: python -u train_phase2.py > logs/train_phase2.log 2>&1"""
import copy
import os
import torch
from model.net import PolicyValueNet
from train.league import run_league

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DMC_CKPT = 'model.pt'
SAVE_PATH = 'model_phase2.pt'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    main_agent = PolicyValueNet().to(device)
    if os.path.exists(DMC_CKPT):
        try:
            ckpt = torch.load(DMC_CKPT, map_location=device, weights_only=False)
            # Filter out keys with shape mismatches (value_head changed due to oracle in Phase 2)
            current_shapes = {k: v.shape for k, v in main_agent.state_dict().items()}
            filtered = {k: v for k, v in ckpt['model'].items()
                        if k in current_shapes and v.shape == current_shapes[k]}
            skipped = [k for k in ckpt['model'] if k not in filtered]
            result = main_agent.load_state_dict(filtered, strict=False)
            missing = result.missing_keys
            unexpected = result.unexpected_keys
            print(f'Loaded DMC checkpoint (missing={len(missing)}, unexpected={len(unexpected)}, skipped_shape_mismatch={len(skipped)})')
        except Exception as e:
            print(f'Could not load DMC checkpoint ({e}), starting from scratch')

    teacher = copy.deepcopy(main_agent)
    exploiter = copy.deepcopy(main_agent)

    optimizer = torch.optim.AdamW(main_agent.parameters(), lr=1e-4, weight_decay=1e-4)

    run_league(
        deck=DECK,
        main_agent=main_agent,
        exploiter_agent=exploiter,
        teacher=teacher,
        optimizer=optimizer,
        device=device,
        save_path=SAVE_PATH,
        n_iterations=2000,
        games_per_iter=5,
        batch_size=64,
        buffer_capacity=50_000,
        epsilon_start=0.3,
        epsilon_end=0.05,
        search_count=5,
        eval_every=50,
        eval_games=30,
        exploiter_update_every=200,
        kl_weight_init=0.01,
        kl_anneal_iters=1500,
    )


if __name__ == '__main__':
    main()
