import torch
import torch.nn as nn
import torch.nn.functional as F


class OneHotEncoder(nn.Module):
    def __init__(self, num_idx, cat_info, transform_net=None):
        """
        Args:
            num_idx      (Sequence[int]):
                List of numeric‐column indices (order matters).
            cat_info     (Dict[int, int]):
                Mapping {cat_col_idx: num_levels}.
                Insertion order determines one‐hot block order.
            transform_net (nn.Module or None):
                If not None, should be a network mapping
                sum(cat_dims) → new_cat_dim. If None, the raw one‐hot
                block is used unchanged.
        """
        super().__init__()
        self.num_idx = list(num_idx)
        self.cat_idx = list(cat_info.keys())
        self.cat_dims = list(cat_info.values())
        self.transform_net = transform_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape
               - (N, D)
               - (B, N, D)
               - (B1, B2, ..., N, D)
          (last dimension = D features)

        Returns:
            Tensor of shape
               - (N, len(num_idx) + out_cat_dim)
               - (B, N, len(num_idx) + out_cat_dim)
               - (B1, B2, ..., N, len(num_idx) + out_cat_dim)
          where
             out_cat_dim = transform_net’s output size if transform_net is not None,
                           otherwise = sum(cat_dims).
        """
        device = x.device

        # 1) Slice numeric columns from last axis
        idx_tensor = torch.tensor(self.num_idx, dtype=torch.long, device=device)
        # Shape: (..., N, len(num_idx))
        num_feats = x.index_select(dim=-1, index=idx_tensor).float()

        # 2) One‐hot each categorical column
        oh_list = []
        for col_idx, dim in zip(self.cat_idx, self.cat_dims):
            codes = x[..., col_idx].long()  # shape (..., N)
            oh_block = F.one_hot(codes, num_classes=dim)  # shape (..., N, dim)
            oh_list.append(oh_block)

        # 3) If there is at least one categorical column, build cat_feats
        if oh_list:
            # cat_feats: shape (..., N, sum(cat_dims))
            cat_feats = torch.cat(oh_list, dim=-1).float()

            # 3a) If transform_net is provided, apply it; else leave as-is
            if self.transform_net is not None:
                # transform_net accepts (..., N, in_features)
                # and returns (..., N, new_cat_dim)
                transformed = self.transform_net(cat_feats)
            else:
                transformed = cat_feats

        else:
            # No categorical columns → keep an empty‐width tensor of shape (..., N, 0)
            shape_prefix = x.shape[:-1]  # i.e. (..., N)
            transformed = torch.empty(*shape_prefix, 0, device=device)

        # 4) Concatenate numeric + (possibly transformed) categorical on last axis
        # num_feats:   (..., N, len(num_idx))
        # transformed: (..., N, out_cat_dim)
        return torch.cat([num_feats, transformed], dim=-1)
