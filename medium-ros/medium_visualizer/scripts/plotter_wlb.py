import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import math

class WlbPlotter:

    def __init__(self,
                 phi_range=np.arange(-15,20,5)*np.pi/180,
                 theta_range=np.arange(-60, 65, 5)*np.pi/180,
                 r_range=np.arange(40, 504, 4)):

        self.num_theta = theta_range.shape[0]
        self.num_phi = phi_range.shape[0]
        self.num_r = r_range.shape[0]

        # Compute transform from each arena cube to cartesian coordinates
        self.transform = np.zeros((3, self.num_theta, self.num_phi, self.num_r))

        for i, theta in enumerate(theta_range):
            for j, phi in enumerate(phi_range):
                for k, r in enumerate(r_range):
                    self.transform[:, i, j, k] = polar2cart(theta, phi, r)

    def plot_marching_cubes(self, raw_image_list, bins_x=40, bins_y=40, power_threshold=0):
        """
        Convert raw_images into cartesian voxel map, and compute and visualize surfaces of voxel bodies.
        :param raw_image_list:
        :param bins_x:
        :param bins_y:
        :param power_threshold:
        :return:
        """
        voxels = np.zeros((self.num_theta, bins_x, bins_y))
        xx = np.linspace(0, 500, bins_x) - 250
        yy = np.linspace(0, 500, bins_y) - 250

        # Compute voxel powers
        for raw_image in raw_image_list:
            for i, height_slice in enumerate(raw_image):
                for j in range(self.num_phi):
                    for k in range(self.num_r):
                        # if raw_image[i,j,k] > power_threshold:
                        x_idx = np.argmin(np.abs(self.transform[0, i, j, k] - xx))
                        y_idx = np.argmin(np.abs(self.transform[1, i, j, k] - yy))

                        voxels[i, x_idx, y_idx] += raw_image[i, j, k]
        try:
            if voxels.sum() > 0:
                # Use marching cubes to obtain the surface mesh of these ellipsoids
                verts, faces, normals, values = measure.marching_cubes_lewiner(voxels, 0)
                # Fancy indexing: `verts[faces]` to generate a collection of triangles
                mesh = Poly3DCollection(verts[faces])
                mesh.set_edgecolor('k')

                # Display resulting triangular mesh using Matplotlib. This can also be done
                # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
                fig = Figure(figsize=(20, 20), dpi=100)
                fig.patch.set_facecolor('#000000')
                canvas = FigureCanvasAgg(fig)

                ax = fig.add_subplot(111, projection='3d')

                ax.add_collection3d(mesh)

                ax.set_xlabel("x-axis: a = 6 per ellipsoid")
                ax.set_ylabel("y-axis: b = 10")
                ax.set_zlabel("z-axis: c = 16")

                ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
                ax.set_ylim(0, 20)  # b = 10
                ax.set_zlim(0, 32)  # c = 16

                ax.set_facecolor('#000000')

                # Plot to numpy array for publishing
                canvas.draw()
                s, (width, height) = canvas.print_to_buffer()

                return np.fromstring(s, np.uint8).reshape((height, width, 4))

        except Exception as e:
            print("Error in plot_robot_man: " + e.message)

        return np.zeros_like(raw_image_list[0])

    def plot_raw_measurements(self, raw_image):
        """
        Convert the raw image into a RGB image.
        :param raw_image: 
        :return: 
        """

        image = np.sum(raw_image, axis=0)
        cmap = plt.cm.jet
        image = image.astype(np.float)/255.0

        return cmap(image)*255.0

    def plot_raw_image_slice(self, raw_image_slice_np):
        """
        Convert raw_image_slice into RGB image.
        :param raw_image_slice_np: 
        :return: 
        """

        fig = Figure(figsize=(20, 20), dpi=100)
        fig.patch.set_facecolor('#000000')
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.imshow(raw_image_slice_np, aspect ='auto')

        ax.set_facecolor('#000000')
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        return np.fromstring(s, np.uint8).reshape((height, width, 4))


def polar2cart(theta, phi, r):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

if __name__ == "__main__":
    import json

    input = np.load("/home/kingkolibri/10_catkin_ws/src/p2g-ros/medium_localizer/scripts/sample_walabot.npy", allow_pickle=True)

    #keypoints = extract_keypoints_max(input[5])

    plt_wlb = WlbPlotter()

    fig = plt.figure()

    img = plt_wlb.plot_marching_cubes(input)
    plt.imshow(img.astype(np.uint8))
    plt.show()
