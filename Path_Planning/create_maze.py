from PIL import Image
import numpy as np

def image_to_grid(image_path, threshold=128):
    # Open the image
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    
    # Threshold the image
    thresholded_image = grayscale_image.point(lambda p: 255 if p < threshold else 0)
    
    # Convert pixel values to 0s and 1s
    binary_grid = np.array(thresholded_image) // 255
    
    # Show the binary grid image
    thresholded_image.show()

    return binary_grid

binary_grid = image_to_grid('..\images\maze.png')
np.save('maze.npy', binary_grid)
