from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


class CombineMode(Enum):
    """Boolean operations for combining ripples."""

    OR = "or"  # Ripples add together (white only on collision)
    XOR = "xor"  # Exclusive or - overlapping cancels out (black where collision)
    AND = (
        "and"  # Intersection - only where ripples overlap (white only where collision)
    )


@dataclass
class Droplet:
    """Represents a single droplet/ripple source."""

    x: float
    y: float
    start_frame: int
    intensity: int
    speed: float = 1.0
    max_radius: float = None


class BooleanRipple:
    def __init__(self, size: int, combine_mode: CombineMode = CombineMode.XOR):
        """
        Initialize the simulation.

        Args:
            size: Size of the square grid (size x size)
            combine_mode: How to combine multiple ripples (OR, XOR, AND)
        """
        self.size = size
        self.combine_mode = combine_mode
        self.grid = np.zeros((size, size), dtype=bool)
        self.droplets: list[Droplet] = []
        self.current_frame = 0

        # Pre-compute coordinate grids for distance calculations
        self.y_coords, self.x_coords = np.mgrid[0:size, 0:size]

    def add_droplet(
        self,
        x: float,
        y: float,
        intensity: int = 2,
        speed: float = 1.0,
        max_radius: float = None,
    ):
        """
        Add a new droplet at position (x, y).

        Args:
            x: x position (column index)
            y: y position (row index)
            intensity: Width of the ripple ring in cells (integer for boolean)
            speed: Expansion speed in cells per frame
            max_radius: Optional maximum radius before the ripple stops
        """
        if max_radius is None:
            max_radius = np.sqrt(2) * self.size

        droplet = Droplet(
            x=x,
            y=y,
            start_frame=self.current_frame,
            intensity=int(intensity),
            speed=speed,
            max_radius=max_radius,
        )
        self.droplets.append(droplet)

    def _compute_ripple(self, droplet: Droplet) -> np.ndarray:
        """Compute the boolean ripple pattern for a single droplet."""
        distances = np.sqrt(
            (self.x_coords - droplet.x) ** 2 + (self.y_coords - droplet.y) ** 2
        )

        # Current ripple radius based on time since droplet was added
        elapsed_frames = self.current_frame - droplet.start_frame
        ripple_radius = elapsed_frames * droplet.speed

        # Boolean ring: True if distance is within [radius - intensity/2, radius + intensity/2]
        half_intensity = droplet.intensity / 2
        inner_radius = max(0, ripple_radius - half_intensity)
        outer_radius = ripple_radius + half_intensity

        ripple = (distances >= inner_radius) & (distances <= outer_radius)
        if ripple_radius > droplet.max_radius:
            ripple = np.zeros_like(ripple, dtype=bool)

        return ripple

    def update(self) -> np.ndarray:
        """
        Update the simulation by one frame.

        Returns:
            The updated grid as a 2D boolean numpy array
        """
        # Start with empty grid
        if self.combine_mode == CombineMode.AND:
            combined = np.ones((self.size, self.size), dtype=bool)
        else:
            combined = np.zeros((self.size, self.size), dtype=bool)

        droplets_to_remove = []
        for i, droplet in enumerate(self.droplets):
            elapsed = self.current_frame - droplet.start_frame
            current_radius = elapsed * droplet.speed

            if current_radius > droplet.max_radius + droplet.intensity:
                droplets_to_remove.append(i)
                continue

            ripple = self._compute_ripple(droplet)

            if self.combine_mode == CombineMode.OR:
                combined = np.logical_or(combined, ripple)
            elif self.combine_mode == CombineMode.XOR:
                combined = np.logical_xor(combined, ripple)
            elif self.combine_mode == CombineMode.AND:
                combined = np.logical_and(combined, ripple)

        for i in reversed(droplets_to_remove):
            self.droplets.pop(i)

        self.grid = combined
        self.current_frame += 1
        return self.grid


def create_boolean_animation(
    size: int = 100,
    combine_mode: CombineMode = CombineMode.XOR,
    droplet_schedule: list[tuple] = None,
    frames: int = 200,
    interval: int = 50,
    save_path: str = None,
):
    """
    Create and display/save the boolean ripple animation.

    Args:
        size: Grid size (size x size)
        combine_mode: Boolean operation for combining ripples
        droplet_schedule: List of (frame, x, y, intensity, speed) tuples
        frames: Total number of animation frames
        interval: Milliseconds between frames
        save_path: If provided, save animation to this path

    Returns:
        The animation object
    """
    sim = BooleanRipple(size, combine_mode=combine_mode)

    # Some arbitrary droplets
    if droplet_schedule is None:
        droplet_schedule = [
            (0, size * 0.3, size * 0.3, 3, 1.5),
            (20, size * 0.7, size * 0.5, 2, 1.0),
            (40, size * 0.4, size * 0.7, 4, 1.2),
            (70, size * 0.8, size * 0.2, 3, 0.8),
            (100, size * 0.2, size * 0.8, 3, 1.3),
            (130, size * 0.6, size * 0.4, 2, 1.0),
            (160, size * 0.5, size * 0.5, 5, 1.5),
        ]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")

    binary_cmap = ListedColormap(["black", "white"])

    # Initial image
    im = ax.imshow(
        sim.grid.astype(int),
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        extent=[0, size, 0, size],
        origin="lower",
        interpolation="nearest",
    )

    def init():
        im.set_array(sim.grid.astype(int))
        return [im]

    def animate(frame):
        for scheduled in droplet_schedule:
            if scheduled[0] == frame:
                _, x, y, intensity, speed = scheduled
                sim.add_droplet(x, y, intensity=intensity, speed=speed)

        grid = sim.update()
        im.set_array(grid.astype(int))

        return [im]

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=interval, blit=True
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)

    return anim, fig


if __name__ == "__main__":
    anim_xor, _ = create_boolean_animation(
        size=100,
        combine_mode=CombineMode.XOR,
        frames=220,
        interval=40,
        save_path="anim.gif",
    )
