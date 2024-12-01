import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_lbp(image, width):
    """
    Compute the Local Binary Pattern (LBP) of an input grayscale image.

    Parameters:
        image (numpy.ndarray): The input grayscale image.
        width (int): Width of the square image.

    Returns:
        numpy.ndarray: The LBP image.

    """
    
    # Initialize an array to store the LBP image
    lbp_image = np.zeros(shape=(width, width))
    
    # Define the number of neighboring pixels to consider
    num_neighbors = 1
    
    # Iterate over each pixel in the image
    for i in range(num_neighbors, image.shape[0] - num_neighbors):
        for j in range(num_neighbors, image.shape[1] - num_neighbors):
            
            # Get the pixel at [i,j] as the center pixel
            center_pixel = image[i, j]
            
            # Initialize a binary string
            binary_string = ""
            
            # Iterate over the neighboring pixels of the center pixel
            
            ''' 
            Local Binary Patterns (LBP) is a texture descriptor that 
            encodes the local structure of an image by comparing each 
            pixel with its surrounding neighbors and representing the 
            result as a binary pattern. Comparing each pixel with its 
            surrounding neighbors in local binary patterns (LBP) involves 
            examining the intensity values of neighboring pixels relative 
            to the central pixel. By thresholding these intensity 
            differences, binary patterns are generated, encoding local 
            variations in texture and structure. This process captures 
            information about the local texture patterns within the image, 
            allowing for robust analysis and classification of textures 
            and objects. The significance of the binary patterns lies 
            in their ability to represent local texture variations 
            efficiently and effectively. By encoding the relationships 
            between a central pixel and its neighboring pixels as binary 
            values (0 or 1), complex texture information is simplified 
            into a compact form. This binary representation is robust 
            to changes in illumination and noise, making it highly 
            suitable for tasks such as texture classification, facial 
            recognition, and object detection in computer vision 
            applications. Additionally, the binary nature of LBP 
            enables simple and fast computation, facilitating 
            real-time processing in various systems and scenarios.
            '''
            
            for m in range(i - num_neighbors, i + num_neighbors + 1):
                for n in range(j - num_neighbors, j + num_neighbors + 1):
                    
                    # Skip the center pixel
                    if [i, j] == [m, n]:
                        pass
                    
                    else:
                        neighbor_pixel = image[m, n]                                           
                        # Compare the intensity of the center pixel with its neighbors
                        if center_pixel >= neighbor_pixel:
                            binary_string += '0'
                        else:
                            binary_string += '1'
            
            # Convert the binary string to decimal and assign it to the corresponding pixel in the LBP image
            lbp_image[i, j] = int(binary_string, 2)
            
    return lbp_image


def get_features(image, size):
    """
    Compute and concatenate histograms of image blocks.

    Parameters:
        image (numpy.ndarray): The input image.
        size (int): Size of the blocks for histogram computation.

    Returns:
        list: Concatenated histograms of the image blocks.

    """
    # Initialize a list to store histograms
    histograms = []
    
    # Iterate over image blocks
    for i in range(0, image.shape[0], size):
        for j in range(0, image.shape[1], size):
            # Extract a block from the image
            block = image[i:i+size, j:j+size]
            
            # Compute histogram of the block and append it to the list of histograms
            
        
            if i == 40:
                image_box = np.zeros(shape=(100, 100))
                image_box[i:i+size, j:j+size] = image[i:i+size, j:j+size]
                plt.figure()
                plt.imshow(image_box, cmap="gray")
                
                plt.figure()
                plt.imshow(block, cmap="gray")
                
                plt.figure()
                plt.hist(block)
                plt.show()
            
               
            histograms.extend(np.histogram(block, bins=[0, 51, 102, 153, 204, 255])[0])
    
    return histograms



######################################################################
######################################################################
######################################################################

# Load the image from the specified directory
img_file = "sample-3.png"

#img_file = "lbp.drawio.png"
img = cv2.imread(img_file)

# Resize the image to a fixed size (100x100 pixels)
img = cv2.resize(img, (100, 100))
plt.figure()
plt.imshow(img, cmap="gray")

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute Local Binary Pattern (LBP) features of the grayscale image
lbp_img = get_lbp(img_gray, 100)

# Display the LBP image
plt.figure()
plt.imshow(lbp_img, cmap="gray")

# Extract features from the LBP image
features = get_features(lbp_img, 20)
plt.figure()
plt.plot(features)


'''
# define the alpha and beta
alpha = .5 # Contrast control
beta = 70 # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
plt.figure()
plt.imshow(adjusted, cmap="gray")

# Convert the image to grayscale
img_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

# Compute Local Binary Pattern (LBP) features of the grayscale image
lbp_img = get_lbp(img_gray, 100)

# Display the LBP image
plt.figure()
plt.imshow(lbp_img, cmap="gray")

# Extract features from the LBP image
features = get_features(lbp_img, 20)
plt.figure()
plt.plot(features)
'''




