from skimage import io, measure, filters, color
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Image epicycle fit')
parser.add_argument('input_file', metavar='file', help='input file')
args = parser.parse_args()

image = io.imread(args.input_file)
image = color.rgb2gray(image)
thresh = filters.threshold_mean(image)
binary = image > thresh

fig, ax = plt.subplots()
contours = measure.find_contours(binary, 0.8)
ax.imshow(binary, cmap=plt.cm.gray)
for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
print(len(contours[1]))
plt.show()
