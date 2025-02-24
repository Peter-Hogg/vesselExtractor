import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from skimage.draw import disk
from scipy.ndimage import center_of_mass
from skimage.measure import label
from skimage.draw import circle_perimeter

def wedge_mask(shape, center, radius, start_angle=0, arc_length=360):
    """
    Create a binary mask of a wedge from a circle.
    
    Parameters:
        shape (tuple): Shape of the output mask (height, width).
        center (tuple): Center of the circle (y, x).
        radius (float): Radius of the circle.
        start_angle (float): Starting angle of the wedge in degrees.
        arc_length (float): Arc length of the wedge in degrees.
        
    Returns:
        np.ndarray: Binary mask with the wedge set to True.
    """
    # Create meshgrid
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    
    # Calculate angles
    angles = np.degrees(np.arctan2(Y - center[0], X - center[1])) % 360
    
    # Define wedge conditions
    within_radius = dist_from_center <= radius
    within_angle = (angles >= start_angle) & (angles <= start_angle + arc_length)
    
    # Combine conditions
    wedge = within_radius & within_angle
    
    return wedge

def estimate_wedge(mask, start_angle=0, arc_length=360):
    """
    Estimate the wedge mask based on the binary mask of an object.
    
    Parameters:
        mask (ndarray): Binary mask of the object.
        start_angle (float): Starting angle of the wedge in degrees.
        arc_length (float): Arc length of the wedge in degrees.
        
    Returns:
        np.ndarray: Binary mask with the wedge.
    """
    # Get center of mass
    center = center_of_mass(mask)
    
    # Calculate radius using area
    area = np.sum(mask)
    radius = np.sqrt(area / np.pi)
    
    # Generate wedge mask
    wedge = wedge_mask(mask.shape, center, radius, start_angle, arc_length)
    
    return wedge


def plot_wedges(img, pupil_mask, sclera_mask, pupil_arc=(0, 360), sclera_arc=(0, 360)):
    """
    Plot pupil and sclera wedges on the image.
    
    Parameters:
        img (ndarray): Grayscale image to display as background.
        pupil_mask (ndarray): Binary mask of the pupil.
        sclera_mask (ndarray): Binary mask of the sclera.
        pupil_arc (tuple): (start_angle, arc_length) for the pupil wedge.
        sclera_arc (tuple): (start_angle, arc_length) for the sclera wedge.
    """
    plt.imshow(img, cmap='gray')
    
    # Pupil wedge
    pupil_wedge = estimate_wedge(pupil_mask, start_angle=pupil_arc[0], arc_length=pupil_arc[1])
    plt.contour(pupil_wedge, colors='red', linewidths=2)
    
    # Sclera wedge
    sclera_wedge = estimate_wedge(sclera_mask, start_angle=sclera_arc[0], arc_length=sclera_arc[1])
    plt.contour(sclera_wedge, colors='blue', linewidths=2)
    
    plt.tight_layout()
    plt.show()

    return pupil_wedge, sclera_wedge

def estimate_circle(mask):
    """Estimate the center and radius of a binary mask."""
    center = center_of_mass(mask)
    area = np.sum(mask)
    radius = np.sqrt(area / np.pi)
    return center, radius


def generate_concentric_circles(pupil_center, sclera_center, pupil_radius, sclera_radius, num_circles=10):
    """Generate concentric circles with linearly spaced centers and radii."""
    radii = np.linspace(pupil_radius, sclera_radius, num_circles)
    centers_x = np.linspace(pupil_center[0], sclera_center[0], num_circles)
    centers_y = np.linspace(pupil_center[1], sclera_center[1], num_circles)

    circles = []
    for x, y, r in zip(centers_x, centers_y, radii):
        circle_coords = circle_perimeter(int(x), int(y), int(r))
        circles.append((circle_coords, (x, y, r)))

    return circles, radii


def restrict_to_arc(circle_coords, center, start_angle=0, arc_length=360):
    """Restrict circle coordinates to a specific arc based on the starting angle and arc length."""
    rr, cc = circle_coords
    angles = np.degrees(np.arctan2(rr - center[0], cc - center[1])) % 360
    angle_range = (angles >= start_angle) & (angles <= start_angle + arc_length)
    return rr[angle_range], cc[angle_range]


def count_intersections(vessel_mask, circles, start_angle=0, arc_length=360):
    """Count intersections of blood vessels with each circle in a specific arc."""
    intersections = []
    height, width = vessel_mask.shape

    for circle_coords, center_data in circles:
        rr, cc = restrict_to_arc(circle_coords, center_data[:2], start_angle, arc_length)
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        intersections.append(np.sum(vessel_mask[rr[valid_indices], cc[valid_indices]]))

    return intersections


def plot_sholl_analysis(pupil_mask, sclera_mask, vessel_mask, num_circles=10, start_angle=0, arc_length=360):
    """Perform Sholl analysis and plot results within a specified arc."""
    # Estimate circles
    pupil_center, pupil_radius = estimate_circle(pupil_mask)
    sclera_center, sclera_radius = estimate_circle(sclera_mask)

    # Generate concentric circles
    circles, radii = generate_concentric_circles(pupil_center, sclera_center, pupil_radius, sclera_radius, num_circles)

    # Count intersections
    intersections = count_intersections(vessel_mask, circles, start_angle, arc_length)

    # Plot vessel mask and arcs
    plt.imshow(vessel_mask, cmap='gray', vmax=0.5)
    plt.plot(pupil_center[1], pupil_center[0], 'o', color='red', label='Pupil center')
    for circle_coords, center_data in circles:
        rr, cc = restrict_to_arc(circle_coords, center_data[:2], start_angle, arc_length)
        plt.plot(cc, rr, '.', color='yellow', markersize=0.5, alpha=0.5)
    plt.title(f'Blood Vessel Intersections in Arc ({start_angle}° to {start_angle + arc_length}°)')
    plt.legend()
    plt.show()

    # Plot Sholl analysis results
    plt.plot(radii, intersections, marker='o', linestyle='-', color='blue')
    plt.xlabel('Radius from Pupil Center (pixels)')
    plt.ylabel('Number of Intersections')
    plt.title('Sholl Analysis')
    plt.grid(True)
    plt.show()

    return radii, intersections
