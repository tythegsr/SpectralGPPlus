import torch
import torch.nn.functional as F


def one_hot_encoding(X, qual_dict, return_all=False):
    """
    Apply one-hot encoding to qualitative variables in the input tensor.

    Args:
        X (torch.Tensor): Input tensor of shape [n_samples, n_features]
        qual_dict (dict): Dictionary specifying qualitative variables and their levels.
            Keys are column indices, values are number of levels.
            If empty, returns an empty tensor.

    Returns:
        torch.Tensor: Concatenated tensor containing one-hot encoded categorical
        variables, with optional inclusion of original numerical variables.
        Returns empty tensor if no categorical variables are present.
    """
    # Return empty tensor if no categorical variables
    if not qual_dict:
        return torch.empty((X.shape[0], 0), device=X.device)

    encoded = []
    for col_idx, n_levels in qual_dict.items():
        # Get the unique values for this column
        unique_values = torch.unique(X[:, col_idx])
        # Create a mapping from value to index
        value_to_idx = {val.item(): idx for idx, val in enumerate(unique_values)}

        # Convert values to indices using the mapping
        indices = torch.tensor([value_to_idx[val.item()] for val in X[:, col_idx]], device=X.device)

        # One-hot encode
        one_hot = F.one_hot(indices, num_classes=n_levels)
        encoded.append(one_hot)

    if return_all:
        remove_idx = list(qual_dict.keys())[0]  # 10
        X_cont = torch.cat([X[:, :remove_idx], X[:, remove_idx + 1 :]], dim=1)
        return torch.cat([encoded[0], X_cont], dim=1)

    else:
        return torch.cat(encoded, dim=1)


# Made by kian, this version doesnt require qual dict
def one_hot_encode_integer_columns(data: torch.Tensor, int_tol: float = 1e-6):
    """
    One-hot encode integer columns in a tensor by detecting columns that are approximately integer.

    Args:
        data: (N, D) torch tensor (float or int) - the input data to encode
        int_tol: float, default=1e-6 - tolerance for determining if a column contains integer values.
                 A column is considered integer if |value - round(value)| < int_tol for all values.
                 Use smaller values (e.g., 1e-8) for strict integer detection, larger values (e.g., 1e-3)
                 for more lenient detection of near-integer columns.

    Note:
        This function automatically detects and encodes ALL columns that meet the integer tolerance criteria.
        If you want to encode only specific columns, you should pre-filter the data tensor to include only
        those columns before calling this function, or use a different encoding function that accepts
        column indices as parameters.

    Returns:
        encoded_data: torch.Tensor with integer columns one-hot encoded
        column_info: list of dicts describing how each original column was treated
    """
    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float32)

    num_rows, num_cols = data.shape
    encoded_columns = []
    column_info = []

    for col_idx in range(num_cols):
        col = data[:, col_idx]

        is_integer_like = torch.all(torch.abs(col - torch.round(col)) < int_tol)
        if is_integer_like:
            col_long = torch.round(col).to(torch.long)
            min_val = int(col_long.min().item())
            max_val = int(col_long.max().item())
            num_classes = max_val - min_val + 1

            # Shift so the smallest value maps to 0, then one-hot
            class_indices = col_long - min_val
            ohe = F.one_hot(class_indices, num_classes=num_classes).to(data.dtype)
            encoded_columns.append(ohe)

            column_info.append(
                {
                    "column": col_idx,
                    "type": "one_hot",
                    "min_value": min_val,
                    "max_value": max_val,
                    "num_classes": num_classes,
                }
            )
        else:
            encoded_columns.append(col.unsqueeze(1))
            column_info.append({"column": col_idx, "type": "continuous"})

    encoded_data = torch.cat(encoded_columns, dim=1)
    return encoded_data, column_info
