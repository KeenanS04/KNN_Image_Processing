import numpy as np
from PIL import Image
from processor import (
    RGBImage,
    ImageProcessingTemplate,
    StandardImageProcessing,
    PremiumImageProcessing,
    ImageKNNClassifier,
)


def run_tests():
    # Create image processor
    img_proc = PremiumImageProcessing()

    # Read an image from file
    dsc20_img = img_read_helper("img/dsc20.png")

    # Negate and save
    negative_dsc20_img = img_proc.negate(dsc20_img)
    img_save_helper("img/out/dsc20_negate.png", negative_dsc20_img)

    # Chroma key with a background image and save
    bg_img = img_read_helper("img/blue_gradient.png")
    swap_color = (255, 255, 255)
    chroma__img = img_proc.chroma_key(dsc20_img, bg_img, swap_color)
    img_save_helper("img/out/dsc20_chroma_white.png", chroma__img)

    # Uncomment this to run the K-Nearest-Neighbors tests
    knn_example_tests()


def knn_example_tests():
    # make random training data (type: List[Tuple[RGBImage, str]])
    train = []
    # create training images with low intensity values
    train.extend(
        (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
        for _ in range(20)
    )
    # create training images with high intensity values
    train.extend(
        (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
        for _ in range(20)
    )

    # initialize and fit the classifier
    knn = ImageKNNClassifier(5)
    knn.fit(train)

    # should be "low"
    print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    # can be either "low" or "high" randomly
    print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    # should be "high"
    print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))


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
    img_array = np.array(image.get_pixels())
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(path)


def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    """
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()


def pixels_example():
    """
    An example of the 3-dimensional pixels matrix (5 x 4 x 3).
    """
    [[[206, 214, 233], [138, 190, 188], [253, 173, 214], [211, 141, 175]],
     [[204, 209, 163], [208, 136, 169], [220, 131, 131], [214, 187, 209]],
     [[113, 239, 137], [196, 236, 135], [133, 177, 181], [235, 243, 146]],
     [[152, 168, 230], [156, 119, 233], [143, 120, 206], [114, 182, 227]],
     [[231, 251, 117], [193, 222, 188], [123, 205, 127], [154, 102, 166]]]


# Run the test when this file is run
if __name__ == '__main__':
    run_tests()
