def get_categorical_dims(qual_dict, discrete_cols=None, original_columns=None):
    """Returns categorical dimensions in proper order"""
    if discrete_cols is not None and original_columns is not None:
        original_indices = [original_columns[col] for col in discrete_cols]
        return [qual_dict[col] for col in original_indices]
    # Default to sorted by column index
    return [qual_dict[col] for col in sorted(qual_dict.keys())]
