import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse


def get_points(radius=1.0, num_points=5):
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


def get_axis(radius=1.0, path_points=None, length=0.0, H=800, W=800, focal=1.0):
    """
    Visualize the sphere, path, and points along the upper hemisphere.

    Parameters:
        radius (float): Radius of the sphere.
        path_points (np.ndarray): Array of 6D points along the path.
    """
    
    poses = []

    if path_points is not None:
        
        path_x, path_y, path_z, theta_v, phi_v = path_points.T
        # ax.plot(path_x, path_y, path_z, color='red', label="Path Trajectory")

        # Initialize rolling frame
        prev_x_axis = np.array([0, 1, 0])  # Initial reference x-axis
        
        # Plot and label each point with its sequential number
        for index, (x, y, z, theta, phi) in enumerate(path_points):
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
            local_y_axis = np.cross(local_z_axis, -local_x_axis)
            local_y_axis /= np.linalg.norm(local_y_axis)
            
            translation = np.array([x, y, z]) + length * local_z_axis  # Move along local z-axis
            
            hwf = np.array([H, W, focal])
            
            distance = np.linalg.norm(translation)
            near_bound = distance * 0.9
            far_bound = distance * 1.1
            
            # Store local axes for convert to colmap poses
            p = np.column_stack((local_y_axis, local_x_axis, local_z_axis, translation, hwf)).flatten()
            p = np.concatenate((p, [near_bound, far_bound]))
            poses.append(p)  # Stack as rows

    poses = np.array(poses)
    return poses

def convert_store(poses):
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
        
    datadir = os.path.join(args.outputdir, 'poses_bounds.npy')
    np.save(datadir, poses)
    print(f"Saved poses to {datadir}")



parser = argparse.ArgumentParser()
parser.add_argument('--radius', type=float, default=1.0, help='radius of the sphere')
parser.add_argument('--num_points', type=int, default=6, help='number of points along the circle')
parser.add_argument("--length", type=float, default=0.0, help='length from the arm to camera')
parser.add_argument("--outputdir", type=str, default='./data', help='output poses_bounds.npy directory')
parser.add_argument("--H", type=int, default=800, help='height of the image')
parser.add_argument("--W", type=int, default=800, help='width of the image')
parser.add_argument("--focal", type=float, default=1.0, help='focal length of the camera')
args = parser.parse_args()

if __name__ == '__main__':
    points = get_points(radius=args.radius, num_points=args.num_points)
    poses = get_axis(radius=args.radius, path_points=points, length=args.length, H=args.H, W=args.W, focal=args.focal)
    convert_store(poses)