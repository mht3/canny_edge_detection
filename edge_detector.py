import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2
import argparse
from skimage import transform

class Sobel():
    def __init__(self, filepath, size, norm=True):
        self.filepath = filepath
        self.norm = norm
        self.size = size

    def _read_image(self, filepath):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # Nearest neighbor resampling to specified size
        return transform.resize(image, self.size, order=0)

    def _norm(self, image):
        # 
        # Min-max normalization
        mx = image.max()
        denom = mx if mx > 0 else 1.
        return 255.*(image) / denom

    def _convolve(self, image):
        # gradient in the y direction -> horizontal edges
        kernel_y = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        # gradient in the x direction -> vertical edges
        kernel_x = kernel_y.T
        # horizontal edges from y gradient approximation
        Q = ndimage.convolve(image, kernel_y, mode='nearest')
        # vertical edges from x gradient approximation
        P = ndimage.convolve(image, kernel_x, mode='nearest')
        # 2 Norm of X and Y coordinates (combines vertical and horizontal esges)
        edge_map = np.sqrt(P**2 + Q**2)
        # calculate angles (dy/dx)
        edge_angles = np.arctan2(Q, P)

        if self.norm:
            edge_map = self._norm(edge_map)

        return edge_map, edge_angles

    def __call__(self):
        # Read image as black and white
        image = self._read_image(self.filepath)
        # Perform sobel operation on image with 3x3 fixed kernels
        # Do not need edge angles if Canny is not used.
        edge_map, _ = self._convolve(image)
        return edge_map

class Canny(Sobel):

    def __init__(self, filepath, size, sigma=0., threshold=50):
        # initialize fixed sobel kernels and black and white image
        super().__init__(filepath, size)

        # gaussian smoothing standard deviation parameter
        self.sigma = sigma
        # Threshold for hysteresis thresholding
        self.threshold = threshold

    def _gaussian_smoothing(self, image, sigma):
        return ndimage.gaussian_filter(image, sigma=sigma, mode='nearest')

    def _get_orientation(self, angle):
        '''
        Helper function to check if angles fall into 1 of 4 orientations.
        Used for nonmaxima suppression.

        returns a single orientation in radians
        '''
        options = [0., np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        # Putting all if statements in cases for simplicity
        case1 = angle > np.deg2rad(-22.5) and angle <= np.deg2rad(22.5)
        case2 = angle > np.deg2rad(22.5) and angle <= np.deg2rad(67.5)
        case3 = (angle > np.deg2rad(67.5) and angle <= np.deg2rad(90.)) or (angle > np.deg2rad(-90.) and angle <= np.deg2rad(-67.5))
        case4 = angle > np.deg2rad(-67.5) and angle <= np.deg2rad(-22.5)
        if case1:
            idx = 0
        elif case2:
            idx = 1
        elif case3:
            idx = 2
        elif case4:
            idx = 3
        else:
            raise ValueError(f'Angle `{angle}` is must be from -pi/2 to pi/2.')
        
        # orientation if needed
        orientation = options[idx]
        return idx

    def _nonmaxima_suppression(self, image, angles):
        '''
        Edge thinning technique
        Makes sure only magnitudes stemming from greatest local changes remain.

        Steps
        1. Round orientation of each pixel to nearest 45 degree (0, 45, 90, and 135. Covers all 8 directions)
        atan2 outputs angles -pi/2 to pi/2
            0: -22.5 < theta <= 22.5
            45: 22.5 < theta <= 67.5
            90: 67.5 < theta <= 90
                -90 < theta <= -67.5
            135: -67.5 < thete <= -22.5
        2. Compare edge strength along orientation line. Check if pixel (i,j) is max with neighbors:
            0: (i, j+1) and (i, j-1)
            45: (i-1, j+1) and (i+1, j-1)
            90: (i-1, j) and (i+1, j)
            135: (i-1, j-1) and (i+1, j+1)

            original value if maximum and 0 otherwise
        '''

        output = np.zeros_like(image)
        # pad with zeros around all sides
        padded_img = np.pad(image, pad_width=1)

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                angle = angles[i, j]
                # gives integer values of 0-3 representing 0, 45, 90, and 135 degree lines
                orientation = self._get_orientation(angle)

                # Offset indices for padded image
                k = i + 1
                m = j + 1
                p = padded_img[k, m]
                if orientation == 0:
                    # check 0 degree line neighboring points
                    q = padded_img[k, m - 1]
                    r = padded_img[k, m + 1]

                elif orientation == 1:
                    # check 45 degree line neighboring points
                    q = padded_img[k - 1, m + 1]
                    r = padded_img[k + 1, m - 1]
                elif orientation == 2:
                    # check 90 degree line neighboring points
                    q = padded_img[k - 1, m]
                    r = padded_img[k + 1, m]
                else:
                    # check 135 degree line
                    q = padded_img[k - 1, m - 1]
                    r = padded_img[k + 1, m + 1]
                if p > q and p > r:
                    # p is a local maximum -> fill output with original magnitude
                    output[i, j] = p
                # otherwise keep at 0

        return output

    def _hysteresis(self, image, threshold):
        '''
        Double thresholding technique.
        '''
        MAX_THRESH = 255.
        t_lower = threshold
        t_upper = 2. * threshold

        if t_upper > MAX_THRESH:
            t_upper = MAX_THRESH
        

        strong = (image >= t_upper)
        output = np.zeros_like(image)
        for i in range(1, output.shape[0] - 1):
            for j in range(1, output.shape[1] - 1):
                # current pixel
                p = image[i, j]
                isStrong = strong[i, j]
                isWeak = p < t_upper and p >= t_lower
                if isWeak:
                    # 8 neighboring pixels
                    neighbors = [strong[i-1, j-1], strong[i-1, j], strong[i-1, j+1], strong[i, j-1],
                                strong[i, j+1], strong[i+1, j-1], strong[i+1, j], strong[i+1, j+1]]
                    # check if one of neighboring pixels is strong, and if so make this value 255
                    for n in neighbors:
                        if n == True:
                            output[i, j] = 255.
                            break
                elif isStrong:
                    output[i, j] = 255.

                

        return output
        

    def __call__(self):
        '''
        Overrides Sobel's call method with the canny implementation.
        '''
        # Read image as black and white
        image = self._read_image(self.filepath)

        # Gaussian Smoothing to remove noise
        gaussian_img = self._gaussian_smoothing(image, self.sigma)

        # Perform sobel operation on image with 3x3 fixed kernels
        edge_map, edge_angles = self._convolve(gaussian_img)

        # edge thinning
        suppressed_img = self._nonmaxima_suppression(edge_map, edge_angles)

        # Hysteresis Thresholding
        output = self._hysteresis(suppressed_img, self.threshold)
        return output

def plot(image, output):

    fig = plt.figure(figsize=(6, 8))
    # Left and right subplots
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # Original image
    ax1.imshow(image)
    # Model output for comparison
    ax2.imshow(output, cmap='gray', vmin=0, vmax=255)
    ax1.set_axis_off()
    ax2.set_axis_off()

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Arguments for epochs and batch size. 
    parser.add_argument("-t", "--threshold", type=float, help="Enter lower bound threshold.", default=100)
    parser.add_argument("-s", "--sigma", type=float, help="Standard deviation for gaussian blur.", default=0.)
    parser.add_argument("-f", "--filename", default='data/perry.png')
    parser.add_argument("--height", type=int, help="Image height.", default=None)
    parser.add_argument("--width", type=int, help="Image width.", default=None)

    args = parser.parse_args()
    # Read in image as black and white
    image = cv2.cvtColor(cv2.imread(args.filename), cv2.COLOR_BGR2RGB)

    if args.height is None:
        height = image.shape[0]
    else:
        height = args.height

    if args.width is None:
        width = image.shape[1]
    else:
        width = args.width

    # image = transform.resize(image, (height, width), order=0)
    # print(height, width)
    # sobel = Sobel(filepath=args.filename, size=(height, width))
    # sobel_output = sobel()
    
    canny = Canny(filepath=args.filename, sigma=args.sigma, threshold=args.threshold, size=(height, width))
    canny_output = canny()

    # plot(image, sobel_output)
    plot(image, canny_output)