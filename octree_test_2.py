import matplotlib.pyplot as plt
import numpy as np


class OctreeNode:
    def __init__(self, boundary, min_size):
        self.boundary = boundary  # A 3D cube represented as [x, y, z, size]
        self.children = []  # List of child nodes
        self.min_size = min_size  # Minimum size for subdivision

    def contains_point(self, point):
        """Check if a point is within the cube boundary."""
        x, y, z, size = self.boundary
        px, py, pz = point
        return (x <= px <= x + size and
                y <= py <= y + size and
                z <= pz <= z + size)

    def inside_sphere(self, point):
        """Check if the point is inside the inscribed sphere of the cube."""
        x, y, z, size = self.boundary
        cx, cy, cz = x + size / 2, y + size / 2, z + size / 2
        radius = size / 2

        px, py, pz = point
        distance = np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)

        return distance <= radius

    def subdivide(self, points):
        """Subdivide the cube if it contains any points, and remove empty cubes."""
        x, y, z, size = self.boundary
        half_size = size / 2

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
            # Keep only points that are inside the child cube and inside the sphere
            child_points = [p for p in points if child.contains_point(p) and child.inside_sphere(p)]

            # Only keep the child if it contains points and is larger than min_size
            if len(child_points) > 0:
                if half_size > self.min_size:
                    child.subdivide(child_points)
                self.children.append(child)

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

    def draw_sphere(self, ax):
        """Draw the inscribed sphere for this node."""
        x, y, z, size = self.boundary
        cx, cy, cz = x + size / 2, y + size / 2, z + size / 2
        radius = size / 2

        # Create a sphere using parametric equations
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        xs = cx + radius * np.cos(u) * np.sin(v)
        ys = cy + radius * np.sin(u) * np.sin(v)
        zs = cz + radius * np.cos(v)

        ax.plot_wireframe(xs, ys, zs, color="blue", alpha=0.3)

    def draw(self, ax):
        """Recursively draw the cube, its children, and the inscribed sphere."""
        self.draw_cube(ax, self.boundary)
        self.draw_sphere(ax)
        for child in self.children:
            child.draw(ax)


def plot_octree(root, points):
    """Visualize the octree and the points in a 3D plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(points) > 0:
        px, py, pz = zip(*points)
        ax.scatter(px, py, pz, color='red', s=50, marker='o')

    root.draw(ax)

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])

    plt.show()

# Define a 3D space with origin (0, 0, 0) and size 100
root_boundary = [0, 0, 0, 100]  # x, y, z, size
min_size = 2  # Minimum cube size before stopping subdivision

# Generate random points within the bounding box [0, 100] for each axis
num_points = 6
points = np.random.uniform(0, 100, (num_points, 3))

# Create the root of the Octree
root = OctreeNode(root_boundary, min_size)

# Subdivide the root node based on the points
root.subdivide(points)

# Plot the Octree and the points
plot_octree(root, points)
