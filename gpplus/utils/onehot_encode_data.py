import torch
import torch.nn.functional as F


def encode_qual_data(data: torch.Tensor, qual_dict: dict, source_col: int = None, grouped: bool = False):
    """
    Encode data using a single specification dict for all non-continuous variables.

    Args:
        data: (N, D) torch tensor
        qual_dict: dict mapping column_index -> num_classes for ALL non-continuous columns
        source_col: optional int; if provided, identifies which column in qual_dict is the source

    Returns:
        encoded_data: torch.Tensor ordered as [continuous | categorical | source]
        cont_cols: list of ints for continuous columns (empty if none)
        cat_cols: list of lists of ints for categorical one-hot groups (empty if none)
        source_cols: list of lists of ints for source one-hot groups (empty if none)
    """
    if qual_dict is None or len(qual_dict) == 0:
        import warnings

        warnings.warn("qual_dict is empty or None. No categorical encoding to perform.", UserWarning)
        # Return all columns as continuous
        cont_cols = list(range(data.shape[1]))
        return data, cont_cols, [], []

    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float64)

    num_rows, num_cols = data.shape
    qual_dict = dict(qual_dict)

    # Handle -1 as special case for last column in qual_dict
    if source_col == -1:
        if len(qual_dict) == 0:
            raise ValueError("qual_dict is empty, cannot use -1 for source_col")
        source_col = max(qual_dict.keys())

    if source_col is not None and source_col not in qual_dict:
        raise ValueError("source_col must be a key in qual_dict if provided.")

    qual_cols = sorted(qual_dict.keys())
    continuous_cols = [c for c in range(num_cols) if c not in qual_dict]
    categorical_cols = [c for c in qual_cols if c != source_col]

    parts = []
    continuous_tensors = [data[:, c].unsqueeze(1) for c in continuous_cols]
    if len(continuous_tensors) > 0:
        parts.append(torch.cat(continuous_tensors, dim=1))

    def _ohe(col_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
        values = torch.round(col_tensor).to(torch.long)
        unique_vals = torch.unique(values)
        if len(unique_vals) != num_classes:
            raise ValueError(
                f"Expected {num_classes} unique values but found {len(unique_vals)}: {unique_vals.tolist()}"
            )

        # Create mapping from actual values to 0-based indices
        val_to_idx = {val.item(): idx for idx, val in enumerate(unique_vals)}
        indices = torch.tensor([val_to_idx[val.item()] for val in values], dtype=torch.long)
        return F.one_hot(indices, num_classes=num_classes).to(data.dtype)

    categorical_tensors = [_ohe(data[:, c], int(qual_dict[c])) for c in categorical_cols]
    if len(categorical_tensors) > 0:
        parts.append(torch.cat(categorical_tensors, dim=1))

    source_tensors = []
    if source_col is not None:
        source_tensors.append(_ohe(data[:, source_col], int(qual_dict[source_col])))
    if len(source_tensors) > 0:
        parts.append(torch.cat(source_tensors, dim=1))

    encoded_data = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    # Build blocks
    offset = 0
    if len(continuous_tensors) > 0:
        offset += len(continuous_cols)

    cat_cols_blocks = []
    for c in categorical_cols:
        width = int(qual_dict[c])
        cat_cols_blocks.append(list(range(offset, offset + width)))
        offset += width

    source_cols_blocks = []
    if source_col is not None:
        width = int(qual_dict[source_col])
        source_cols_blocks = list(range(offset, offset + width))
        offset += width

    if grouped and len(cat_cols_blocks) > 0:
        start_idx = cat_cols_blocks[0][0]
        end_idx = cat_cols_blocks[-1][-1]
        cat_cols_blocks = [list(range(start_idx, end_idx + 1))]

    return encoded_data, continuous_cols, cat_cols_blocks, source_cols_blocks


def learn_encodings(data: torch.Tensor, int_tol: float = 1e-6, cont_cols=None):
    """
    Infer a qualitative encoding specification (qual_dict) from the provided data.

    For each column that appears integer-like (within tolerance), compute the
    minimum and maximum rounded values and derive the number of classes as
    (max - min + 1). Returns a dict mapping column_index -> num_classes.

    Args:
        data: (N, D) torch tensor
        int_tol: tolerance for determining integer-like columns
        cont_cols: optional int, list of ints, or None. If provided, specifies columns
                  that should be treated as continuous (even if they appear integer-like).
                  For example, age might be stored as integers but should be treated as
                  continuous. Columns in cont_cols will be excluded from qual_dict.

    Returns:
        qual_dict: dict[int, int] mapping column index to inferred num_classes
    """
    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float64)

    _, num_cols = data.shape
    qual_dict = {}

    # Normalize cont_cols to a set for easy checking
    if cont_cols is None:
        cont_cols_set = set()
    elif isinstance(cont_cols, int):
        cont_cols_set = {cont_cols}
    elif isinstance(cont_cols, (list, tuple)):
        cont_cols_set = set(cont_cols)
    else:
        raise TypeError(f"cont_cols must be int, list of ints, or None, got {type(cont_cols)}")

    for col_idx in range(num_cols):
        # Skip columns explicitly marked as continuous
        if col_idx in cont_cols_set:
            continue
            
        col = data[:, col_idx]
        is_integer_like = torch.all(torch.abs(col - torch.round(col)) < int_tol)
        if not is_integer_like:
            continue
        col_long = torch.round(col).to(torch.long)
        unique_vals = torch.unique(col_long)
        num_classes = len(unique_vals)
        if num_classes <= 0:
            continue
        qual_dict[col_idx] = num_classes

    return qual_dict


# def one_hot_encode_integer_columns(data: torch.Tensor, int_tol: float = 1e-6):
#     """
#     data: (N, D) torch tensor (float or int)
#     Returns:
#       encoded_data: torch.Tensor with integer columns one-hot encoded
#       column_info: list of dicts describing how each original column was treated
#     """
#     if data.dtype not in (torch.float32, torch.float64):
#         data = data.to(torch.float64)

#     num_rows, num_cols = data.shape
#     encoded_columns = []
#     column_info = []

#     for col_idx in range(num_cols):
#         col = data[:, col_idx]

#         is_integer_like = torch.all(torch.abs(col - torch.round(col)) < int_tol)
#         if is_integer_like:
#             col_long = torch.round(col).to(torch.long)
#             min_val = int(col_long.min().item())
#             max_val = int(col_long.max().item())
#             num_classes = max_val - min_val + 1

#             # Shift so the smallest value maps to 0, then one-hot
#             class_indices = col_long - min_val
#             ohe = F.one_hot(class_indices, num_classes=num_classes).to(data.dtype)
#             encoded_columns.append(ohe)

#             column_info.append({
#                 "column": col_idx,
#                 "type": "one_hot",
#                 "min_value": min_val,
#                 "max_value": max_val,
#                 "num_classes": num_classes
#             })
#         else:
#             encoded_columns.append(col.unsqueeze(1))
#             column_info.append({
#                 "column": col_idx,
#                 "type": "continuous"
#             })

#     encoded_data = torch.cat(encoded_columns, dim=1)
#     return encoded_data, column_info
