import torch
import torch.nn as nn
import torch.nn.functional as F

CARD_COUNT = 1268       # max cardId + 1
OPTION_TYPE_COUNT = 17  # OptionType enum 0–16
SLOT_FEATURES = 40
NUM_SLOTS = 12
D_EMBED = 64
D_SETS = 128
RESNET_CHANNELS = 128
NUM_RESBLOCKS = 6
TRUNK_DIM = 256
SCALAR_DIM = 8
# Combined: RESNET_CHANNELS + 3*D_SETS + SCALAR_DIM = 128+384+8 = 520
COMBINED_DIM = RESNET_CHANNELS + 3 * D_SETS + SCALAR_DIM
D_VALUE_HIDDEN = 64      # value head hidden size
D_OPT_TYPE_EMBED = 16   # option type embedding dim
D_OPT_PROJ = 64         # option projection output dim


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        r = x
        x = x.transpose(1, 2)  # [B, L, C]
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [B, C, L]
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        return x + r


class PolicyValueNet(nn.Module):
    """ResNet over board slots + EmbeddingBag sets → value + per-option scores."""

    def __init__(self):
        super().__init__()
        # Board branch: project slot features → ResNet → global avg pool
        self.board_proj = nn.Conv1d(SLOT_FEATURES, RESNET_CHANNELS, kernel_size=1)
        self.resblocks = nn.ModuleList([_ResBlock(RESNET_CHANNELS) for _ in range(NUM_RESBLOCKS)])

        # Set branches: EmbeddingBag with mean pooling. Card ID 0 is reserved as padding/unknown.
        self.hand_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)
        self.discard_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)
        self.deck_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)

        # Oracle branch: opponent's hand cards (training only; zeroed at inference).
        # padding_idx=0 ensures an all-zero input tensor produces a zero embedding,
        # so passing None (→ dummy zeros) is identical to passing an all-zero oracle.
        # NOTE: Phase 1 checkpoints are NOT compatible with this value head.
        self.oracle_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)

        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(COMBINED_DIM, TRUNK_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_DIM, TRUNK_DIM),
            nn.ReLU(),
        )

        # Value head (takes trunk + oracle embedding as input)
        self.value_head = nn.Sequential(
            nn.Linear(TRUNK_DIM + D_SETS, D_VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(D_VALUE_HIDDEN, 1),
            nn.Tanh(),
        )

        # Option embedding (type + card)
        self.opt_type_embed = nn.Embedding(OPTION_TYPE_COUNT, D_OPT_TYPE_EMBED)
        self.opt_card_embed = nn.Embedding(CARD_COUNT, D_EMBED, padding_idx=0)
        self.opt_proj = nn.Linear(D_OPT_TYPE_EMBED + D_EMBED, D_OPT_PROJ)

        # Action scorer
        self.action_scorer = nn.Linear(TRUNK_DIM + D_OPT_PROJ, 1)

    def _encode_state(
        self,
        board: torch.Tensor,         # [B, 12, 40]
        hand_ids: torch.Tensor,      # [B, max_hand]
        discard_ids: torch.Tensor,   # [B, max_discard]
        deck_ids: torch.Tensor,      # [B, max_deck]
        scalars: torch.Tensor,       # [B, 8]
    ) -> torch.Tensor:               # [B, TRUNK_DIM]
        # Board: [B, 40, 12] for Conv1d
        x = board.transpose(1, 2)
        x = F.relu(self.board_proj(x))   # [B, 128, 12]
        for block in self.resblocks:
            x = block(x)
        board_emb = x.mean(dim=2)        # [B, 128]

        hand_emb = self.hand_embed(hand_ids)        # [B, 128]
        discard_emb = self.discard_embed(discard_ids)
        deck_emb = self.deck_embed(deck_ids)

        combined = torch.cat([board_emb, hand_emb, discard_emb, deck_emb, scalars], dim=-1)
        return self.trunk(combined)  # [B, TRUNK_DIM]

    def forward(
        self,
        board: torch.Tensor,         # [B, 12, 40]
        hand_ids: torch.Tensor,      # [B, max_hand]
        discard_ids: torch.Tensor,   # [B, max_discard]
        deck_ids: torch.Tensor,      # [B, max_deck]
        scalars: torch.Tensor,       # [B, 8]
        opt_types: torch.Tensor,     # [N] option type IDs
        opt_cards: torch.Tensor,     # [N] option card IDs
        opp_hand_ids: torch.Tensor | None = None,  # [B, max_opp_hand] or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (value [B,1], scores [N]).

        opp_hand_ids: opponent hand card IDs for oracle training signal.
            Pass None (or all-zeros) at inference time to zero out the oracle.
            padding_idx=0 guarantees None → zeros → zero embedding.
        """
        assert board.size(0) == 1, "PolicyValueNet.forward requires batch size 1"
        trunk = self._encode_state(board, hand_ids, discard_ids, deck_ids, scalars)

        # Oracle embedding: zero when None (inference), non-zero during training
        if opp_hand_ids is None:
            oracle_emb = self.oracle_embed(torch.zeros(1, 1, dtype=torch.long, device=board.device))
        else:
            oracle_emb = self.oracle_embed(opp_hand_ids)  # [B, D_SETS]

        value = self.value_head(torch.cat([trunk, oracle_emb], dim=-1))  # [B, 1]

        # Option embeddings
        type_emb = self.opt_type_embed(opt_types)   # [N, 16]
        card_emb = self.opt_card_embed(opt_cards)   # [N, D_EMBED]
        opt_emb = F.relu(self.opt_proj(torch.cat([type_emb, card_emb], dim=-1)))  # [N, 64]

        # Score each option against state (B=1 assumed for inference)
        trunk_exp = trunk.expand(opt_emb.size(0), -1)  # [N, TRUNK_DIM]
        scores = self.action_scorer(torch.cat([trunk_exp, opt_emb], dim=-1)).squeeze(-1)  # [N]
        return value, scores
