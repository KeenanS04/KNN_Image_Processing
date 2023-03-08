import numpy as np
from PIL import Image

NUM_CHANNELS = 3


class RGBImage:
    """
    A template for image objects in RGB color spaces.
    """

    def __init__(self, pixels):
        """
        Constructor that takes a 3 dimensional list of pixels and stores it
        as an image object.

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # Check if pixels is a list and has at least one element
        if not isinstance(pixels, list) or (len(pixels) < 1):
            raise TypeError()
        # Check if every row in pixels is a list and has at least one element
        if not all([(isinstance(row, list) and (len(pixels) >= 1)) 
                    for row in pixels]):
            raise TypeError()
        # Make sure all rows are the same length
        if not all(*[map(lambda x: x == len(pixels[0]), 
                         [len(row) for row in pixels])]):
            raise TypeError()
        # Make sure all pixels are lists of length 3
        if not all([((isinstance(pixel, list)) and (len(pixel) == 3)) 
                    for row in pixels for pixel in row]):
            raise TypeError()
        # Checks if all RGB values are in the correct range
        if any([((value < 0) or (value > 255)) for row in pixels 
                for pixel in row for value in pixel]):
            raise ValueError()
        
        # Initialize pixels matrix and dimensions
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns a tuple that represents the dimensions of the image in the
        format, (rows, columns).

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a deep copy of the pixels matrix.

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[value for value in val] for val in row] 
                for row in self.pixels]

    def copy(self):
        """
        Returns a copy of the original RGBImage instance.

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the pixel at the given row and column.

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        # Check if row and column numbers are ints
        if (not isinstance(row, int)) or (not isinstance(col, int)):
            raise TypeError()
        # Check if rows are columns are valid numbers given image size
        if ((row < 0) or (row >= self.num_rows) 
            or (col < 0) or (col >= self.num_cols)):
            raise ValueError()
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the color of the pixel at the given row and column to new_color.
        If any values are negative, skip over them but update the others.

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the resulting pixel list
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
         # Check if row and column numbers are ints
        if (not isinstance(row, int)) or (not isinstance(col, int)):
            raise TypeError()
        # Check if rows are columns are valid numbers given image size
        if ((row < 0) or (row >= self.num_rows) 
            or (col < 0) or (col >= self.num_cols)):
            raise ValueError()
        # Check if new_color is a tuple of length 3 and every element in it
        # is an int.
        if (not isinstance(new_color, tuple)) or (not len(new_color) == 3):
            if not all([isinstance(new_color, int)]):
                raise TypeError()
        # Make sure that all intensity values are valid
        if any([value > 255 for value in new_color]):
            raise ValueError()
        for i in range(len(new_color)):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]
            
        


class ImageProcessingTemplate:
    """
    Template for image processing, parent class for both the premium and
    standard version of the app. Image processing methods return copies of the
    images and does not modify the original.
    """

    def __init__(self):
        """
        Initializes the ImageProcessingTemplate object and sets the cost to 0

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns cost associated with ImageProcessingTemplate object, shoud
        always be 0 unless inherited by a subclass.

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns the negative of the given image.

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        negative = image.copy()
        # List comprehension to invert each pixel (255-val)
        [negative.set_pixel(row, col, tuple([255-negative.get_pixel(row, col)[i] 
                                             for i in range(3)])) 
         for row in range(negative.size()[0]) 
         for col in range(negative.size()[1])]
        return negative

    def grayscale(self, image):
        """
        Returns a grayscale version of the image.

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        grayscale = RGBImage(image.get_pixels())
        # List comprehension to grayscale each pixel ((R+G+B)/3)
        [grayscale.set_pixel(row, col, tuple([sum(grayscale.get_pixel(row, col))//3 
                                    for i in range(3)]))  
         for row in range(grayscale.size()[0]) 
         for col in range(grayscale.size()[1])]
        return grayscale

    def rotate_180(self, image):
        """
        Rotates the given image 180 degrees

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        rotated = image.copy()
        # Start at bottom right corner of image
        image_rows = image.size()[0]-1
        image_cols = image.size()[1]-1
        # List comprehension that works its way backwards and moves each pixel
        # to the opposite end of the image.
        [rotated.set_pixel(image_rows-row, image_cols-col, image.get_pixel(row, col)) 
         for row in range(rotated.size()[0]) 
         for col in range(rotated.size()[1])]
        return rotated



class StandardImageProcessing(ImageProcessingTemplate):
    """
    Monetized version of the template image processing class. Image processing 
    methods return copies of the images and does not modify the original.
    """

    def __init__(self):
        """
        Initializes the image processing object with a default cost of 0.

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0
        self.num_coupons = 0
        self.num_rotations = 0

    def negate(self, image):
        """
        Returns the negative of the given image. Cost of this process is 6.

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # Check for coupons
        if self.num_coupons > 0:
            self.num_coupons -= 1
            return super().negate(image)
        self.cost += 5
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale version of the image. Cost of this process is 6.

        """
        if self.num_coupons > 0:
            self.num_coupons -= 1
            return super().grayscale(image)
        self.cost += 6
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Rotates the given image 180 degrees. Since rotating twice will undo
        the rotation, the cost of 10 will be refunded every 2 rotations.

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # Check for coupons
        if self.num_coupons > 0:
            self.num_coupons -= 1
            return super().rotate_180(image)
        # Check number of rotations to see if refund needed
        if self.num_rotations%2 == 1:
            self.cost -= 10
        else:
            self.cost += 10
        self.num_rotations += 1
        return super().rotate_180(image)

    def redeem_coupon(self, amount):
        """
        Makes the next number of processes free, multiple calls will add to the
        total amount of free images.

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """
        self.num_coupons += amount



class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Premium version of the template image processing class. Image processing 
    methods return copies of the images and does not modify the original.
    """

    def __init__(self):
        """
        Initializes the premium image processing object with a set cost of 50.

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Replaces any pixels in the chroma image of the given color with the 
        corresponding pixels in the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        # Make sure background_image and chroma_image are RGBImage
        if (not isinstance(chroma_image, RGBImage) or 
            not isinstance(background_image, RGBImage)):
            raise TypeError()
        # Make sure images are same size
        if chroma_image.size() != background_image.size():
            raise ValueError()
        chroma_keyed = RGBImage(chroma_image.get_pixels())
        # Search for pixels of required color and replace them with the pixel
        # at the same location in background_image
        for row in range(chroma_image.size()[0]):
            for col in range(chroma_image.size()[1]):
                if chroma_image.get_pixel(row, col) == color:
                    chroma_keyed.set_pixel(row, col, background_image.get_pixel(row, col))
        return chroma_keyed

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Creates a new image where the top left corner of sticker_image is
        placed at the given position in the background image.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        # Make sure background_image and chroma_image are RGBImage
        if (not isinstance(sticker_image, RGBImage) or
            not isinstance(background_image, RGBImage)):
            raise TypeError()
        # Make sure sticker image is smaller than background image
        if (sticker_image.size()[0] >= background_image.size()[0] or
            sticker_image.size()[1] >= background_image.size()[1]):
            raise ValueError()
        # Check types of coordinate
        if (not isinstance(x_pos, int) or 
            not isinstance(y_pos, int)):
            raise TypeError()
        # Make sure sticker at given position will be within the bounds of
        # the image
        if (sticker_image.size()[0] + y_pos > background_image.size()[0] or
            sticker_image.size()[1] + x_pos > background_image.size()[1]):
            raise ValueError()
        
        stickered_image = RGBImage(background_image.get_pixels())
        # Start at given position and iterate through pixels in sticker and
        # replace pixel at position in background image with corresponding
        # sticker pixel.
        for row in range(sticker_image.size()[0]):
            for col in range(sticker_image.size()[1]):
                stickered_image.set_pixel(y_pos + row, x_pos + col, sticker_image.get_pixel(row, col))
        return stickered_image
                


class ImageKNNClassifier:
    """
    Class to use a KNN classifier to predict similar images using a K-nearest
    neighbors strategy.
    """

    def __init__(self, n_neighbors):
        """
        Initializes the object and n-nearest neighbors for the algorithm.
        """
        self.n_neighbors = n_neighbors
        self.data = list()

    def fit(self, data):
        """
        Fits classifier by storing training data in the instance.
        """
        # Make sure training data is big enough to fit within n neighbors
        if len(data) <= self.n_neighbors:
            raise ValueError()
        # Cannot overwrite training data
        if len(self.data) > 0:
            raise ValueError()
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        Calculates the Euclidian distance between image1 and image2
        """
        # Make sure both images are of type RGBImage
        if (not isinstance(image1, RGBImage) or
            not isinstance(image2, RGBImage)):
            raise TypeError()
        # Make sure both images are of same size
        if image1.size() != image2.size():
            raise ValueError()
        
        # Calculate Euclidian distance by summing square differences between
        # pixels and finding the square root.
        return sum([sum([sum([(image1.get_pixel(row, col)[i]-image2.get_pixel(row, col)[i])**2 
                              for i in range(3)]) 
                         for col in range(image1.size()[1])]) 
                    for row in range(image1.size()[0])])**.5

    @staticmethod
    def vote(candidates):
        """
        Find the most popular label from a list of candidates
        """
        label = candidates[0]
        count = 0
        for candidate in candidates:
            frequency = candidates.count(candidate)
            if frequency > count:
                count = frequency
                label = candidate
        return label
            

    def predict(self, image):
        """
        Predicts the label of the given image using KNN classification.
        """
        # Make sure there is training data for the predictor
        if len(self.data) == 0:
            raise ValueError()
        
        # Create list of distances between image and training data
        distances = []
        for trained_image in self.data:
            distances.append(
                (ImageKNNClassifier.distance(image, trained_image[0]), 
                 trained_image[1])
                )
        # Sort distances and find best label for image among n neighbors
        distances_sorted = sorted(distances)
        return ImageKNNClassifier.vote([image[1] 
                                        for image in distances_sorted][:self.n_neighbors])
        


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)
