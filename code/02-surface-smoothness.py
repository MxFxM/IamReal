import cv2
import matplotlib.pyplot as plt
import numpy as np

SHOW_CV2 = False
SHOW_CV2_SLICE = False
SHOW_MATPLOTLIB_SLICE = False

# read image as grayscale
input_image = "images/cut.png"
img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

# image parameters
img_height = img.shape[0]
img_width = img.shape[1]

if img_width%100 != 0:
    img = img[0:img_height, 0:img_width-(img_width%100-1)]
    img_width = img.shape[1]

# this will be the output image
out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# show the input image
if SHOW_CV2:
    cv2.imshow("input image", img)

# log the spreads
spreads = []

# step through the image, 100 pixel per step
for step in range(100, img_width, 100):

    # extract a slice of the image
    part = img[0:img_height, step-100:step]
    part_flat = part.ravel()

    q1 = np.quantile(part_flat, 0.25)
    q3 = np.quantile(part_flat, 0.75)
    spread = q3-q1

    if spread < 25:
        pass
    else:
        out[0:img_height, step-100:step, 0] = 200

    spreads.append(spread)

    # show the slice
    if SHOW_CV2_SLICE:
        cv2.imshow(f"Slice {step}", part)

    if SHOW_MATPLOTLIB_SLICE:
        # make a matplotlib figure
        plt.figure()

        # first show the current slice
        plt.subplot(221)
        plt.imshow(part)

        # show the histogram
        plt.subplot(223)
        plt.hist(part_flat, 256, [0,256])

        # show a boxplot
        plt.subplot(224)
        plt.boxplot(part_flat)

        # show
        plt.show()

plt.figure()
plt.subplot(311)
plt.imshow(img, cmap='gray')
plt.subplot(312)
plt.plot(spreads)
plt.subplot(313)
plt.imshow(out)
plt.show()

if SHOW_CV2 or SHOW_CV2_SLICE:
    # wait for a button
    cv2.waitKey(0)

    # clear screens
    cv2.destroyAllWindows()
