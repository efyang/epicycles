from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import argparse
parser = argparse.ArgumentParser(description='Image epicycle fit')
parser.add_argument('input_file', metavar='input_file', help='input file')
parser.add_argument('output_file', metavar='output_file', help='output file')
args = parser.parse_args()

import procimg
imageorig = io.imread(args.input_file)

fig_width_in = 10
image = color.rgb2gray(imageorig)
height,width = np.shape(image)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width_in, fig_width_in*(height/width + 0.1)))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[0, 0].imshow(imageorig)
ax = axes[1, 1]
ax.set_xlim([0, width])
ax.set_ylim([0, height])
axes[1, 0].set_xlim([0, width])
axes[1, 0].set_ylim([0, height])
ax.set_title("Epicycle Approximation of Image")
axes[0, 0].set_title("Original Image")
axes[0, 1].set_title("Binarized, Thresholded Version of Image")
axes[1, 0].set_title("Connected Contours of Image")
plt.suptitle("KAM Project: Approximating Images with Epicycles using the Fourier Transform")

def circle(x, y, r):
    ang = np.arange(0, 2*np.pi, 0.01)
    xp = r * np.cos(ang)
    yp = r * np.sin(ang)
    return ax.plot(x+xp, y+yp, 'orange', linewidth=0.5)


def set_circle(h, x, y, r):
    ang = np.arange(0, 2*np.pi, 0.01)
    xp = r * np.cos(ang)
    yp = r * np.sin(ang)
    h.set_data(x+xp, y+yp)


def line(x1, y1, x2, y2):
    return ax.plot(np.array([x1, x2]), np.array([y1, y2]), 'orange', linewidth=0.5)


def set_line(h, x1, y1, x2, y2):
    h.set_data(np.array([x1, x2]), np.array([y1, y2]))


# x = np.arange(0, 2*np.pi, 0.01)
# theta = np.arange(-2*np.pi, 2*np.pi, 0.01)
# x = theta + 1j * np.sin(theta)
# x = 2*np.array([-2+2*1j, -1+2*1j, 2*1j, 1+2*1j, 2+2*1j, 2+1j, 2, 2-1j, 2-2*1j, 1-2*1j, -2*1j, -1-2*1j, -2-2*1j, -2-1j, -2, -2+1j, -2+2*1j]);
# xm = np.array([-15+4j, -13+4j, -12.5+1j, -11.5+3j, -9.5+3j, -8.5+1j, -8+4j, -6+4j,
              # -4+4j, -3+2j, -2+4j, 0+4j,
              # 2+4j, 4+1j, 6+4j, 8+4j, 14+4j, 14+2j, 12+2j, 12-2j, 10-2j, 10+2j, 8+2j,
              # 8-2j, 6-2j, 6+1j, 5-1j, 3-1j, 2+1j, 2-2j, 0-2j, 0+4j,
              # -2+1j, -2-2j, -4-2j, -4+1j, -6+4j,
              # -7-2j, -9-2j, -10.5+1j, -12-2j, -14-2j, -15+4j]) + 10j
# interpolate to make it smoother
# xk = np.arange(0, len(xm))
# xq = np.arange(0, len(xm), 0.5)
# x = np.interp(xq, xk, xm)

x = procimg.process(image, axes[0, 1], axes[1, 0])
# we want to reduce to around 1000 sample points so that we can fft them in
# reasonable time
x = x[::(len(x)//1000)]
print("using sample of", len(x), "points")
# x = np.arange(-5, 5, 0.5) + 1j*np.arange(-5,5, 0.5)
ax.plot(x.real, x.imag, 'g', linewidth=0.4)
y = np.fft.fftshift(np.fft.fft(x))
n = len(y)
radii = np.abs(y) / n
phase_angles = np.arctan2(y.imag, y.real)
points = np.array([])
P = 1000

# delete the 0 frequency circle (the offset)
import math
zerofreqind = math.ceil(n/2) + (n+1)%2 - 1
center = y[zerofreqind] / n
# radii = np.delete(radii, zerofreqind)
# y = np.delete(y, zerofreqind)
# phase_angles = np.delete(phase_angles, zerofreqind)
fraw = np.arange(0, n)
fraw = np.delete(fraw, zerofreqind)
radii_del = np.delete(radii, zerofreqind)
radii_sort_order = radii_del.argsort()[::-1]
f = fraw[radii_sort_order]

# only keep the most important frequencies
modes = 200
print("Approximating with", modes, "modes")
f = f[:modes]

circles = []
lines = []
for i in np.arange(0, n):
    c, = circle(0, 0, 0)
    m, = line(0, 0, 0, 0)
    circles.append(c)
    lines.append(m)


tracing, = ax.plot(points.real, points.imag, 'r', linewidth=2)
finished_period = False
finished_updating = False

def animate(t):
    print('\r' + str(100 * t/P) + "%   ", end='')
    global points, finished_period, finished_updating
    curpt = center
    artists = []
    for k in f:
        r = radii[k]
        phase_angle = phase_angles[k]
        set_circle(circles[k], curpt.real, curpt.imag, r)
        offset = y[k]/n * np.exp(2*np.pi*1j * (k - zerofreqind) * (t/P))
        next_center = curpt + offset
        set_line(lines[k], curpt.real, curpt.imag, next_center.real, next_center.imag)
        curpt = next_center

    artists.extend(lines)
    artists.extend(circles)
    if not finished_updating:
        points = np.append(points, [curpt])
    if t == P-1:
        finished_period = True
    if finished_period and t == 0:
        finished_updating = True
    tracing.set_data(points.real, points.imag)
    artists.append(tracing)
    return artists



# Init only required for blitting to give a clean slate.
def init():
    tracing.set_ydata(np.ma.array(points.real, mask=True))
    return tracing,


# demo show
# ani = animation.FuncAnimation(fig, animate, np.arange(0, P), init_func=init,
                              # interval=25, blit=True)
# for saved animation
ani = animation.FuncAnimation(fig, animate, np.arange(0, P), init_func=init,
                              interval=25, blit=True, repeat=False)
ani.save(args.output_file, writer='ffmpeg', fps=30)


plt.show()
