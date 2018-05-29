from skimage import measure, filters
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
# def draw():
    # plt.draw()
    # plt.pause(0.001)


# image should be in grayscale already
def process(image, bin_ax, connect_ax):
    # fig, ax = plt.subplots()
    thresh = filters.threshold_mean(image)
    binary = image > thresh
    height = np.size(image, 0)
    bin_ax.imshow(binary, cmap=plt.cm.gray)
    # draw()

    print("find contours...")
    contours = measure.find_contours(binary, 0.8)
    # for n, contour in enumerate(contours):
        # ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax.plot(contours[0][:, 1], contours[0][:, 0], 'r', linewidth=2)
    # draw()

    print("done.")

    print("calculate contour distances...")
    # combine all the contour points into one contiguous block
    combined_points = np.concatenate(contours)
    print(len(combined_points), "points")
    total_pts = len(combined_points)

    # turn the points into complex numbers
    combined_points = np.array([complex(c[1], height - c[0]) for c in combined_points])
    m, n = np.meshgrid(combined_points, combined_points)

    # calculate the distances
    distance_matrix = abs(m - n)

    # get the contour segment lengths
    contour_segment_indices = dict()
    contour_index_offset = 0
    for n, contour in enumerate(contours):
        # create the contour segments
        next_segment_index = len(contour) + contour_index_offset
        contour_segment_indices[n] = (contour_index_offset, next_segment_index)
        contour_index_offset = next_segment_index

    print("done.")
    print("connect contours...")
    # we do this suboptimally with a greedy algorithm to connect contours together
    # with the shortest distance between each, which is good enough. We can reuse
    # much of the original contour reordering, and thus only require single line
    # connections between contours
    num_contours_to_process = len(contours)
    # set our starting contour
    current_contour = 0
    contours_to_connect = set(range(1, num_contours_to_process))
    # [((c1, c2), (p1, p2)), ((c2, c3), (p3, p4))] connects c1 to c2 through the
    # points at indices p1 and p2 and so on
    contour_connections_soln = []
    connected_back_to_origin = False
    # find the index of the smallest distance between a point on the current contour
    # and any other contour
    while len(contours_to_connect) > 0 or not connected_back_to_origin:
        if len(contours_to_connect) != 0:
            contours_to_try = contours_to_connect.copy()
        else:
            contours_to_try = set([0])
        min_connection_point = ()
        min_connection_distance = np.inf
        min_connection_contour = -1
        contour_indices = contour_segment_indices[current_contour]

        # for each of the potential next contours
        while len(contours_to_try) > 0:
            next_contour = contours_to_try.pop()
            next_contour_indices = contour_segment_indices[next_contour]
            # get the portion of the distance matrix that corresponds to the
            # distances between the current contour and the next potential contour
            submat = distance_matrix[contour_indices[0]:contour_indices[1],
                                    next_contour_indices[0]:next_contour_indices[1]]
            # find the minimum distance along with its index
            r,c = np.unravel_index(np.argmin(submat), submat.shape)
            this_contour_min_value = submat[r,c]
            r += contour_indices[0]
            c += next_contour_indices[0]
            # set the new minimum connections
            if this_contour_min_value < min_connection_distance:
                min_connection_point = (r, c)
                min_connection_distance = this_contour_min_value
                min_connection_contour = next_contour

        if len(contours_to_connect) != 0:
            contours_to_connect.remove(min_connection_contour)
        else:
            connected_back_to_origin = True
        # add this next solution
        contour_connections_soln.append(((current_contour, min_connection_contour),
                                        min_connection_point))

        current_contour = min_connection_contour
    num_connections = len(contour_connections_soln)
    # generate the final ordering of points
    final_ordering = np.array([])
    for n in range(num_connections):
        # go through each solution pair - the two points on each contour
        a = contour_connections_soln[(n-1)%num_connections]
        b = contour_connections_soln[n]
        contour_num = a[0][1]
        contour_min, contour_max = contour_segment_indices[contour_num]
        p1, p2 = a[1]
        p3, p4 = b[1]
        # we add the portion of the segment we need to go from p2 to p3
        if p3 - p2 > 0:
            # if we are going in a normal order
            final_ordering = np.concatenate((final_ordering,
                                            combined_points[p2:contour_max],
                                            combined_points[contour_min:p3]))
        else:
            # if we are reversing
            # POTENTIAL BUG WITH INDICES HERE!!!

            # for an open contour
            # final_ordering = np.concatenate((final_ordering,
                                            # np.flipud(combined_points[(contour_min+1):(p2+1)]),
                                            # combined_points[contour_min:p3]))
            # for a closed contour
            final_ordering = np.concatenate((final_ordering,
                                            np.flipud(combined_points[(contour_min+1):(p2+1)]),
                                            np.flipud(combined_points[p3:contour_max])))
    # plt.cla()
    connect_ax.plot(final_ordering.real, final_ordering.imag)
    print("done.")

    # plt.draw()
    # input("Press [enter] to finish.")
    return final_ordering


# import argparse
# parser = argparse.ArgumentParser(description='Image epicycle fit')
# parser.add_argument('input_file', metavar='file', help='input file')
# args = parser.parse_args()

# print(process(args.input_file))


