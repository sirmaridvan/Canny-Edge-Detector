Canny Edge Detector Algorithm
1. Convert to Grayscale
2. Smoothing: Blurring the image to remove noise.
3. Finding gradients: The edges should be marked where the gradients of the image has large
magnitudes.
4. Non-maximum suppression: Only local maxima should be marked as edges.
5. Double thresholding: Potential edges are determined by thresholding.
6. Edge tracking by hysteresis: Final edges are determined by suppressing all edges that are
not connected to a very strong edge.
