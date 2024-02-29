import numpy as np

def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        half length, width and height

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])

    return bbx