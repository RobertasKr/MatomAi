import matplotlib.pyplot as plt
import numpy as np
import laspy

class OctreeNode:
    def __init__(self, boundary, min_size):
        self.boundary = boundary  # A 3D cube represented as [x, y, z, size]
        self.children = []  # List of child nodes
        self.min_size = min_size  # Minimum size for subdivision
        self.points_inside = []  # Points that are inside the node's sphere

    def inside_sphere(self, point, center, radius):
        """Check if the point is inside a given sphere."""
        px, py, pz = point
        cx, cy, cz = center
        distance = np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
        return distance <= radius

    def cube_inside_sphere(self, center, radius):
        """Check if the center of the cube is inside the root's sphere."""
        x, y, z, size = self.boundary
        cx, cy, cz = x + size / 2, y + size / 2, z + size / 2
        return self.inside_sphere((cx, cy, cz), center, radius)

    def subdivide(self, points, root_center, root_radius):
        """Subdivide the cube if it is inside the root's sphere and contains points within its own sphere."""
        x, y, z, size = self.boundary
        half_size = size / 2

        cx, cy, cz = x + size / 2, y + size / 2, z + size / 2
        points_inside_sphere = [p for p in points if self.inside_sphere(p, (cx, cy, cz), half_size)]

        if not points_inside_sphere:
            return

        # Store points for export later
        self.points_inside.extend(points_inside_sphere)

        # Create 8 smaller cubes
        children_boundaries = [
            [x, y, z, half_size],
            [x + half_size, y, z, half_size],
            [x, y + half_size, z, half_size],
            [x + half_size, y + half_size, z, half_size],
            [x, y, z + half_size, half_size],
            [x + half_size, y, z + half_size, half_size],
            [x, y + half_size, z + half_size, half_size],
            [x + half_size, y + half_size, z + half_size, half_size]
        ]

        for boundary in children_boundaries:
            child = OctreeNode(boundary, self.min_size)
            cx, cy, cz = boundary[0] + half_size / 2, boundary[1] + half_size / 2, boundary[2] + half_size / 2
            if child.cube_inside_sphere(root_center, root_radius):
                child_points = [p for p in points_inside_sphere if child.inside_sphere(p, (cx, cy, cz), half_size / 2)]
                if len(child_points) > 0 and half_size > self.min_size:
                    child.subdivide(child_points, root_center, root_radius)  # Recursively subdivide
                self.children.append(child)

    def gather_points(self):
        """Recursively gather points from all the nodes."""
        all_points = self.points_inside[:]
        for child in self.children:
            all_points.extend(child.gather_points())
        return all_points

    def draw_cube(self, ax, boundary):
        """Draw a cube with given boundary."""
        x, y, z, size = boundary
        r = [[x, y, z],
             [x + size, y, z],
             [x + size, y + size, z],
             [x, y + size, z],
             [x, y, z + size],
             [x + size, y, z + size],
             [x + size, y + size, z + size],
             [x, y + size, z + size]]

        # List of vertices that form the 12 edges of a cube
        edges = [
            [r[0], r[1]], [r[1], r[2]], [r[2], r[3]], [r[3], r[0]],  # Bottom square
            [r[4], r[5]], [r[5], r[6]], [r[6], r[7]], [r[7], r[4]],  # Top square
            [r[0], r[4]], [r[1], r[5]], [r[2], r[6]], [r[3], r[7]]  # Vertical edges
        ]

        for edge in edges:
            ax.plot3D(*zip(*edge), color="black", lw=0.5)

    def draw_sphere(self, ax, alpha):
        """Draw the inscribed sphere for this node with decreasing opacity."""
        x, y, z, size = self.boundary
        cx, cy, cz = x + size / 2, y + size / 2, z + size / 2
        radius = size / 2

        # Create a sphere using parametric equations
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        xs = cx + radius * np.cos(u) * np.sin(v)
        ys = cy + radius * np.sin(u) * np.sin(v)
        zs = cz + radius * np.cos(v)

        ax.plot_wireframe(xs, ys, zs, color="blue", alpha=alpha)

    def draw(self, ax, alpha):
        """Recursively draw the cube, its children, and the inscribed sphere with decreasing opacity."""
        self.draw_cube(ax, self.boundary)
        self.draw_sphere(ax, alpha)
        for child in self.children:
            child.draw(ax, alpha * 0.33)  # Decrease opacity by 33% for each level of subdivision


def plot_octree(root):
    """Visualize the octree in a 3D plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw the octree's boundaries and spheres starting with full opacity (alpha = 1)
    root.draw(ax, alpha=1)

    # Set the plot limits
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])

    plt.show()


def export_to_las(points, filename="output.las"):
    """Export the given points to a LAS file."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.write(filename)
    print(f"Exported {len(points)} points to {filename}")

# Define a 3D space with origin (0, 0, 0) and size 100
root_boundary = [0, 0, 0, 100]  # x, y, z, size
min_size = 2  # Minimum cube size before stopping subdivision

# Generate random points within the bounding box [0, 100] for each axis
num_points = 1000  # Use a large number of points to ensure some are inside spheres
points = np.random.uniform(0, 100, (num_points, 3))

# Create the root of the Octree
root = OctreeNode(root_boundary, min_size)

# Subdivide the root node based on the points inside the root sphere
root_center = (50, 50, 50)  # Center of the root's sphere
root_radius = 50  # Radius of the root's sphere
root.subdivide(points, root_center, root_radius)

# Plot the Octree
plot_octree(root)

# Gather all points from the Octree
collected_points = np.array(root.gather_points())

# Export the points to a LAS file
export_to_las(collected_points, "octree_spheres_test.las")
