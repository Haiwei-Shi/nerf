import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import configparser


def generate_upper_hemisphere_path_with_orientation(radius=1.0, num_points=5):
    """
    Generate 6D points (3D position + 3D orientation) along the upper hemisphere of a sphere.

    Parameters:
        radius (float): Radius of the sphere.
        num_points (int): Number of points along the path.

    Returns:
        numpy.ndarray: Array of 6D points [x, y, z, theta, phi].
    """


    # Generate elevation angles (0 at the top, π/2 at the equator)
    theta = np.linspace(0, np.pi / 2, np.int32(num_points/2)+1)  
    # Generate azimuthal angles (full circle around Z-axis)
    phi = np.linspace(0, 2 * np.pi, num_points+1)
    phi = phi[:-1]

    # Create a meshgrid for all combinations of theta and phi
    theta, phi = np.meshgrid(theta, phi)
    
    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Flatten all arrays into a list of points
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten(), theta.flatten(), phi.flatten()))

    return points


def visualize_sphere_with_path(radius=1.0, path_points=None):
    """
    Visualize the sphere, path, and points along the upper hemisphere.

    Parameters:
        radius (float): Radius of the sphere.
        path_points (np.ndarray): Array of 6D points along the path.
    """
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # Create the sphere
    # phi = np.linspace(0, 2 * np.pi, 30)
    # theta = np.linspace(0, np.pi, 30)
    # phi, theta = np.meshgrid(phi, theta)
    # x = radius * np.sin(theta) * np.cos(phi)
    # y = radius * np.sin(theta) * np.sin(phi)
    # z = radius * np.cos(theta)

    # ax.plot_surface(x, y, z, color='white', alpha=0.3, edgecolor='gray')
    # ax.scatter(0, 0, 0, color='black', s=30, label="Center (Object)")
    
    axis = []

    if path_points is not None:
        
        path_x, path_y, path_z, theta_v, phi_v = path_points.T
        # ax.plot(path_x, path_y, path_z, color='red', label="Path Trajectory")

        # Initialize rolling frame
        prev_x_axis = np.array([0, 1, 0])  # Initial reference x-axis
        
        # Plot and label each point with its sequential number
        for index, (x, y, z, theta, phi) in enumerate(path_points):
            # ax.scatter(x, y, z, color='red', s=20, label="Path Points" if index == 0 else "")
            # ax.text(x, y, z, str(index + 1), color='black', fontsize=12)  # Label with sequential number

            # Compute local axes
            point = np.array([x, y, z])
            # Normalize the position vector (local z-axis points toward center)
            local_z_axis = -point / np.linalg.norm(point)
            
            # Local x-axis (tangential to azimuthal direction)
            tangent_azimuth = np.array([-np.sin(phi), np.cos(phi), 0]) 
            local_x_axis = tangent_azimuth / np.linalg.norm(tangent_azimuth)
            
            if np.dot(local_x_axis, prev_x_axis) < 0:  # Check for a flip
                local_x_axis = -local_x_axis
            # Update rolling reference
            prev_x_axis = local_x_axis
            
            # Local y-axis (orthogonal to x-axis and z-axis)
            local_y_axis = np.cross(local_z_axis, local_x_axis)
            local_y_axis /= np.linalg.norm(local_y_axis)
            
            # Store local axes for convert to colmap poses
            axis.append(np.column_stack((local_y_axis, local_x_axis, local_z_axis)))  # Stack as rows

            # Plot local axes at this point
            # ax.quiver(
            #     point[0], point[1], point[2],
            #     local_x_axis[0], local_x_axis[1], local_x_axis[2],
            #     color='blue', label="Local X-axis" if index == 0 else ""
            #     ,length=0.3, linewidth=1
            # )
            # ax.quiver(
            #     point[0], point[1], point[2],
            #     local_y_axis[0], local_y_axis[1], local_y_axis[2],
            #     color='green', label="Local Y-axis" if index == 0 else ""
            #     ,length=0.3, linewidth=1
            # )
            # ax.quiver(
            #     point[0], point[1], point[2],
            #     local_z_axis[0], local_z_axis[1], local_z_axis[2],
            #     color='purple', label="Local Z-axis" if index == 0 else ""
            #     ,length=0.3, linewidth=1
            # )


    # Global axes
    # ax.quiver(0, 0, 0, 1, 0, 0, color='black', linewidth=1, label="Global X-axis")
    # ax.quiver(0, 0, 0, 0, 1, 0, color='black', linewidth=1, linestyle='dashed', label="Global Y-axis")
    # ax.quiver(0, 0, 0, 0, 0, 1, color='black', linewidth=1, linestyle='dotted', label="Global Z-axis")

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title("Upper Hemisphere Path with Local Axes and Sequential Numbering")
    # ax.legend()
    # plt.show()
    axis = np.array(axis)
    return axis


# Generate path points with position and orientation
radius = 1.0
num_points = 6
path_points = generate_upper_hemisphere_path_with_orientation(radius, num_points)

# Visualize the sphere and path
visualize_sphere_with_path(radius, path_points)