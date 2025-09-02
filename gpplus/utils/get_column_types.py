def get_column_types(qual_dict, num_features=10):
    """
    Identify continuous and discrete columns.

    Args:
        qual_dict: Dictionary of {column_index: num_levels} for discrete variables
        num_features: Total number of features in original data

    Returns:
        (continuous_cols, discrete_cols): Lists of column indices
    """
    all_cols = set(range(num_features))
    discrete_cols = list(qual_dict.keys())
    continuous_cols = list(all_cols - set(discrete_cols))
    return sorted(continuous_cols), sorted(discrete_cols)
