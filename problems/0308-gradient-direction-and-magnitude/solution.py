import numpy as np
import math

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    # Your code here
    magnitude = np.linalg.norm(gradient)
    if magnitude == 0:
        direction = [0.0] * len(gradient)
    else:
        direction = [g / magnitude for g in gradient]
    descent_direction= [-x for x in direction]

    return {'magnitude':magnitude, 'direction': direction, 'descent_direction': descent_direction}
    