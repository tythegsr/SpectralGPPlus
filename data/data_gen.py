import torch


def wing_mixed_variables(X, source="s0"):
    """
    Compute wing weight given input variables.

    Args:
        X (np.ndarray): Input array of shape [n_samples, 10] with columns:
            0: Sw (wing area, sq ft)
            1: Wfw (weight of fuel in the wing, lb)
            2: A (aspect ratio)
            3: Gama (quarter-chord sweep angle, degrees)
            4: q (dynamic pressure at cruise, lb/sq ft)
            5: lamb (taper ratio)
            6: tc (airfoil thickness to chord ratio)
            7: Nz (ultimate load factor)
            8: Wdg (flight design gross weight, lb)
            9: Wp (paint weight, lb/sq ft)
        source (str): Source of the data
    Returns:
        np.ndarray: Wing weight values for each input sample
    """
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (torch.pi / 180.0)  # Convert to radians
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    cos_Gama = torch.cos(Gama)
    # Wing weight calculation
    if source == "s0":
        result = (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + Sw * Wp
        )
    elif source == "s1":
        result = (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s2":
        result = (
            0.036
            * Sw**0.8
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s3":
        result = (
            0.036
            * Sw**0.9
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 0 * Wp
        )

    return result