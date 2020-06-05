import numpy as np

def compute_stable_bce_cost(Y, Z):
    """
    This function computes the "Stable" Binary Cross-Entropy(stable_bce) Cost and returns the Cost and its
    derivative w.r.t Z_last(the last linear node) .
    The Stable Binary Cross-Entropy Cost is defined as:
    => (1/m) * np.sum(max(Z,0) - ZY + log(1+exp(-|Z|)))
    Args:
        Y: labels of data
        Z: Values from the last linear node
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    m = len(Y)
    cost = (1/m) * np.sum(np.maximum(Z, 0) - Z*Y + np.log(1+ np.exp(- np.abs(Z))))
    return cost

def numpy_bce(y_true, y_pred, from_logits=False):
    """
    This function computes the Binary Cross-Entropy(stable_bce) Cost function the way Keras
    implements it. Accepting either probabilities(P_hat) from the sigmoid neuron or values direct
    from the linear node(Z)
    Args:
        y_true: labels of data
        y_pred: Probabilities from sigmoid function
        from_logits: flag to check if logits are being provided or not(Default: False)
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    if from_logits:
        # assume that P_hat contains logits and not probabilities
        return compute_stable_bce_cost(y_true, Z=y_pred)

    else:
        # Assume P_hat contains probabilities. So make logits out of them

        # First clip probabilities to stable range
        EPSILON = 1e-07
        P_MAX = 1- EPSILON  # 0.9999999

        y_pred = np.clip(y_pred, a_min=EPSILON, a_max=P_MAX)

        # Second, Convert stable probabilities to logits(Z)
        Z = np.log(y_pred / (1 - y_pred))

        # now call compute_stable_bce_cost
        return compute_stable_bce_cost(y_true, Z)

numpy_mse = lambda y, p: ((y - p) ** 2).mean()