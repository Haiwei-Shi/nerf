import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv


def generate_upper_hemisphere_path_with_orientation(radius=1.0, num_points=10):
    """
    Generate 6D points (3D position + 3D orientation) along the upper hemisphere of a sphere.

    Parameters:
        radius (float): Radius of the sphere.
        num_points (int): Number of points along the path.

    Returns:
        np.ndarray: Array of 6D points [x, y, z, roll, pitch, yaw].
    """

    # Generate points evenly distributed along the upper hemisphere
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # Elevation angles (Rotation about the Y-axis)
    # phi = np.linspace(0, 2 * np.pi, num_points)           # Azimuthal angles
    phi = 0                                                 # Azimuthal angles (Rotation about the Z-axis)

    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Calculate global orientation (roll, pitch, yaw)
    # roll = np.zeros(num_points)                  # Rotation about the X-axis => This assumes there is no rotation about the local X-axis
    # pitch = np.arctan2(z, np.sqrt(x**2 + y**2))  # Elevation angle 
    yaw = np.arctan2(y, x)                       # Azimuthal angle 

    points = np.column_stack((x, y, z, yaw))
    return points


def visualize_sphere_with_path(radius=1.0, path_points=None):
    """
    Visualize the sphere, path, and points along the upper hemisphere.

    Parameters:
        radius (float): Radius of the sphere.
        path_points (np.ndarray): Array of 5D points along the path.
    """


    # Create the sphere
    phi = np.linspace(0, 2 * np.pi, 30)
    theta = np.linspace(0, np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    

    # Plot the sphere and the center
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='white', alpha=0.3, edgecolor='gray')
    ax.scatter(0, 0, 0, color='black', s=20, label="Center (Object)")


    # Plot the path
    if path_points is not None:
        path_x, path_y, path_z, yaw = path_points.T
        ax.scatter(path_x, path_y, path_z, color='red', s=20, label="Path Points")
        ax.plot(path_x, path_y, path_z, color='red', label="Path Trajectory")

        # Initialize rolling frame
        prev_x_axis = np.array([0, 1, 0])  # Initial reference x-axis

        # Visualize axes at each point
        for i in range(len(path_points)):
            point = np.array([path_x[i], path_y[i], path_z[i]])
            norm = np.linalg.norm(point)

            # Normalize the position vector (local z-axis points toward center)
            local_z_axis = -point / norm

            # Local x-axis (tangential to azimuthal direction)
            tangent_azimuth = np.array([-np.sin(yaw[i]), np.cos(yaw[i]), 0])
            local_x_axis = tangent_azimuth / np.linalg.norm(tangent_azimuth)

            if np.dot(local_x_axis, prev_x_axis) < 0:  # Check for a flip
                local_x_axis = -local_x_axis

            # Update rolling reference
            prev_x_axis = local_x_axis

            # Local y-axis (orthogonal to x-axis and z-axis)
            local_y_axis = np.cross(local_z_axis, local_x_axis)
            local_y_axis /= np.linalg.norm(local_y_axis)


            print(f"point{i}: ")
            print("global position: ", point)
            print("local orientation (x): ", local_x_axis)
            print("local orientation (y): ", local_y_axis)
            print("local orientation (z): ", local_z_axis)
            print()


            # Plot local axes at this point
            ax.quiver(
                point[0], point[1], point[2],
                local_x_axis[0], local_x_axis[1], local_x_axis[2],
                color='blue', label="Local X-axis" if i == 0 else ""
            )
            ax.quiver(
                point[0], point[1], point[2],
                local_y_axis[0], local_y_axis[1], local_y_axis[2],
                color='green', label="Local Y-axis" if i == 0 else ""
            )
            ax.quiver(
                point[0], point[1], point[2],
                local_z_axis[0], local_z_axis[1], local_z_axis[2],
                color='purple', label="Local Z-axis" if i == 0 else ""
            )


    # Plot global center axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='black', label="Global X-axis")
    ax.quiver(0, 0, 0, 0, 1, 0, color='black', linestyle='dashed', label="Global Y-axis")
    ax.quiver(0, 0, 0, 0, 0, 1, color='black', linestyle='dotted', label="Global Z-axis")

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Upper Hemisphere Path")
    ax.legend()
    plt.show()


# Generate path points with position and orientation
radius = 2.0
num_points = 13
path_points = generate_upper_hemisphere_path_with_orientation(radius, num_points)

# Visualize the sphere and path
visualize_sphere_with_path(radius, path_points)