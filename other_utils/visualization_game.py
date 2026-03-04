# The idea behind this small visualization game is to show the the relationship between the measured values and how they appear on the image
# We attempt to get as close as the camera conditions as possible

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import (
    glEnable,
    glBegin,
    glEnd,
    glVertex3f,
    glColor3f,
    glLoadIdentity,
    glTranslatef,
    glRotatef,
    glViewport,
    glClearColor,
    glMatrixMode,
    glClear,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_QUADS,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
)

from OpenGL.GLU import gluPerspective
import numpy as np
import tkinter as tk

# Initialize Pygame and OpenGL
pygame.init()
display = (1920, 1200)


class InfoWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Camera Parameters")
        self.root.geometry("600x900")  # Increased height for new matrix

        # Create text widgets
        self.intrinsics_label = tk.Label(
            self.root, text="Intrinsics Matrix:", font=("Courier", 12)
        )
        self.intrinsics_label.pack(anchor="w", padx=10, pady=5)

        self.intrinsics_text = tk.Text(self.root, height=4, width=50, font=("Courier", 12))
        self.intrinsics_text.pack(padx=10, pady=5)

        self.extrinsics_label = tk.Label(
            self.root, text="Camera Extrinsics Matrix (World to Camera):", font=("Courier", 12)
        )
        self.extrinsics_label.pack(anchor="w", padx=10, pady=5)

        self.extrinsics_text = tk.Text(self.root, height=5, width=50, font=("Courier", 12))
        self.extrinsics_text.pack(padx=10, pady=5)

        self.view_matrix_label = tk.Label(
            self.root, text="OpenGL View Matrix (What you see):", font=("Courier", 12)
        )
        self.view_matrix_label.pack(anchor="w", padx=10, pady=5)

        self.view_matrix_text = tk.Text(self.root, height=5, width=50, font=("Courier", 12))
        self.view_matrix_text.pack(padx=10, pady=5)

        self.viewport_label = tk.Label(self.root, text="Viewport Info:", font=("Courier", 12))
        self.viewport_label.pack(anchor="w", padx=10, pady=5)

        self.viewport_text = tk.Text(self.root, height=2, width=50, font=("Courier", 12))
        self.viewport_text.pack(padx=10, pady=5)

        # Controls info
        controls_text = """
Controls:
WASD: Move camera X/Z
QE: Move camera Y
Arrow keys: Rotate camera
ZX: Change FOV
1-2: Adjust viewport X
3-4: Adjust viewport Y
5-6: Adjust viewport width
7-8: Adjust viewport height
        """
        self.controls_label = tk.Label(self.root, text="Controls:", font=("Courier", 12))
        self.controls_label.pack(anchor="w", padx=10, pady=5)

        self.controls_text = tk.Text(self.root, height=10, width=50, font=("Courier", 12))
        self.controls_text.pack(padx=10, pady=5)
        self.controls_text.insert("1.0", controls_text)
        self.controls_text.config(state="disabled")

    def update_matrices(self, intrinsics, extrinsics, view_matrix, viewport_info):
        # Update intrinsics
        self.intrinsics_text.delete("1.0", tk.END)
        intrinsics_str = "\n".join([" ".join([f"{x:7.2f}" for x in row]) for row in intrinsics])
        self.intrinsics_text.insert("1.0", intrinsics_str)

        # Update extrinsics
        self.extrinsics_text.delete("1.0", tk.END)
        extrinsics_str = "\n".join([" ".join([f"{x:7.2f}" for x in row]) for row in extrinsics])
        self.extrinsics_text.insert("1.0", extrinsics_str)

        # Update view matrix
        self.view_matrix_text.delete("1.0", tk.END)
        view_matrix_str = "\n".join([" ".join([f"{x:7.2f}" for x in row]) for row in view_matrix])
        self.view_matrix_text.insert("1.0", view_matrix_str)

        # Update viewport
        self.viewport_text.delete("1.0", tk.END)
        self.viewport_text.insert("1.0", viewport_info)

        # Update the window
        self.root.update()


class CameraIntrinsics:
    def __init__(self):
        self.cx = display[0] / 2
        self.cy = display[1] / 2
        # Keep near and far for OpenGL (they don't affect the intrinsic matrix)
        self.near = 0.1
        self.far = 2000.0
        self.aspect = display[0] / display[1]

        # Your given parameters
        # Focal length[m]
        self.fwx = 0.0176
        self.fwy = 0.0176
        # Size of the pixels [m / pixel]
        self.ppx = 5.86e-6
        self.ppy = self.ppx

        # Focal length[pixels]
        self.fx = self.fwx / self.ppx
        self.fy = self.fwy / self.ppy

        # Calculate FOV from focal length and sensor size
        sensor_width = display[0] * self.ppx  # sensor width in meters
        sensor_height = display[1] * self.ppy  # sensor height in meters

        # Calculate FOV in degrees
        self.fov_horizontal = np.degrees(2 * np.arctan((sensor_width / 2) / self.fwx))
        self.fov_vertical = np.degrees(2 * np.arctan((sensor_height / 2) / self.fwy))

        # Use vertical FOV for OpenGL (conventional)
        self.fov = self.fov_vertical

    def get_matrix(self):
        return np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])


class CameraExtrinsics:
    def __init__(self):
        self.position = [0, 0, -5]
        self.rotation = [0, 0, 0]

    def get_matrix(self):
        # Camera extrinsics matrix (camera in world coordinates)
        rx, ry, rz = np.radians(self.rotation)

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        R = Rz @ Ry @ Rx
        t = np.array(self.position).reshape(3, 1)

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t.flatten()

        return transform

    def get_view_matrix(self):
        # OpenGL view matrix (world in camera coordinates)
        rx, ry, rz = np.radians(-np.array(self.rotation))  # Note the negative angles

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        R = Rx @ Ry @ Rz  # Note the reverse order
        t = -R @ np.array(self.position).reshape(3, 1)

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t.flatten()

        return transform


def draw_cube():
    LENGTH = 0.5
    glEnable(GL_DEPTH_TEST)
    glBegin(GL_QUADS)

    # Each face is drawn with vertices in a consistent order
    # Front face (red)
    glColor3f(0.8, 0.2, 0.2)
    glVertex3f(-LENGTH, -LENGTH, -LENGTH)
    glVertex3f(LENGTH, -LENGTH, -LENGTH)
    glVertex3f(LENGTH, LENGTH, -LENGTH)
    glVertex3f(-LENGTH, LENGTH, -LENGTH)

    # Right face (green)
    glColor3f(0.2, 0.8, 0.2)
    glVertex3f(LENGTH, -LENGTH, -LENGTH)
    glVertex3f(LENGTH, -LENGTH, LENGTH)
    glVertex3f(LENGTH, LENGTH, LENGTH)
    glVertex3f(LENGTH, LENGTH, -LENGTH)

    # Back face (blue)
    glColor3f(0.2, 0.2, 0.8)
    glVertex3f(LENGTH, -LENGTH, LENGTH)
    glVertex3f(-LENGTH, -LENGTH, LENGTH)
    glVertex3f(-LENGTH, LENGTH, LENGTH)
    glVertex3f(LENGTH, LENGTH, LENGTH)

    # Left face (yellow)
    glColor3f(0.8, 0.8, 0.2)
    glVertex3f(-LENGTH, -LENGTH, LENGTH)
    glVertex3f(-LENGTH, -LENGTH, -LENGTH)
    glVertex3f(-LENGTH, LENGTH, -LENGTH)
    glVertex3f(-LENGTH, LENGTH, LENGTH)

    # Top face (purple)
    glColor3f(0.8, 0.2, 0.8)
    glVertex3f(-LENGTH, LENGTH, -LENGTH)
    glVertex3f(LENGTH, LENGTH, -LENGTH)
    glVertex3f(LENGTH, LENGTH, LENGTH)
    glVertex3f(-LENGTH, LENGTH, LENGTH)

    # Bottom face (cyan)
    glColor3f(0.2, 0.8, 0.8)
    glVertex3f(-LENGTH, -LENGTH, LENGTH)
    glVertex3f(LENGTH, -LENGTH, LENGTH)
    glVertex3f(LENGTH, -LENGTH, -LENGTH)
    glVertex3f(-LENGTH, -LENGTH, -LENGTH)

    glEnd()


def main():
    # Create info window
    info_window = InfoWindow()

    # Initialize OpenGL window
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    camera_intrinsics = CameraIntrinsics()
    camera_extrinsics = CameraExtrinsics()

    viewport_x = 0
    viewport_y = 0
    viewport_w = display[0]
    viewport_h = display[1]

    # Set white background
    glClearColor(1.0, 1.0, 1.0, 1.0)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(
        camera_intrinsics.fov,
        camera_intrinsics.aspect,
        camera_intrinsics.near,
        camera_intrinsics.far,
    )

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                info_window.root.destroy()
                return

        # Get the currently pressed keys
        keys = pygame.key.get_pressed()

        # Camera position controls
        if keys[pygame.K_w]:
            camera_extrinsics.position[2] += 0.1
        if keys[pygame.K_s]:
            camera_extrinsics.position[2] -= 0.1
        if keys[pygame.K_a]:
            camera_extrinsics.position[0] += 0.1
        if keys[pygame.K_d]:
            camera_extrinsics.position[0] -= 0.1
        if keys[pygame.K_q]:
            camera_extrinsics.position[1] += 0.1
        if keys[pygame.K_e]:
            camera_extrinsics.position[1] -= 0.1

        # Camera rotation controls
        if keys[pygame.K_LEFT]:
            camera_extrinsics.rotation[1] -= 2
        if keys[pygame.K_RIGHT]:
            camera_extrinsics.rotation[1] += 2
        if keys[pygame.K_UP]:
            camera_extrinsics.rotation[0] -= 2
        if keys[pygame.K_DOWN]:
            camera_extrinsics.rotation[0] += 2

        # FOV controls
        if keys[pygame.K_z]:
            camera_intrinsics.fov = max(1, camera_intrinsics.fov - 1)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(
                camera_intrinsics.fov,
                camera_intrinsics.aspect,
                camera_intrinsics.near,
                camera_intrinsics.far,
            )
        if keys[pygame.K_x]:
            camera_intrinsics.fov += 1
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(
                camera_intrinsics.fov,
                camera_intrinsics.aspect,
                camera_intrinsics.near,
                camera_intrinsics.far,
            )

        # Viewport controls
        if keys[pygame.K_1]:
            viewport_x = max(0, viewport_x - 2)
        if keys[pygame.K_2]:
            viewport_x = min(display[0] - 100, viewport_x + 2)
        if keys[pygame.K_3]:
            viewport_y = max(0, viewport_y - 2)
        if keys[pygame.K_4]:
            viewport_y = min(display[1] - 100, viewport_y + 2)
        if keys[pygame.K_5]:
            viewport_w = max(100, viewport_w - 2)
        if keys[pygame.K_6]:
            viewport_w = min(display[0] - viewport_x, viewport_w + 2)
        if keys[pygame.K_7]:
            viewport_h = max(100, viewport_h - 2)
        if keys[pygame.K_8]:
            viewport_h = min(display[1] - viewport_y, viewport_h + 2)

        # Clear and setup viewport
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(viewport_x, viewport_y, viewport_w, viewport_h)

        # Draw 3D scene
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(*camera_extrinsics.position)
        glRotatef(camera_extrinsics.rotation[0], 1, 0, 0)
        glRotatef(camera_extrinsics.rotation[1], 0, 1, 0)
        glRotatef(camera_extrinsics.rotation[2], 0, 0, 1)
        draw_cube()

        # Update info window
        viewport_info = f"x={viewport_x}, y={viewport_y}, w={viewport_w}, h={viewport_h}"
        info_window.update_matrices(
            camera_intrinsics.get_matrix(),
            camera_extrinsics.get_matrix(),
            camera_extrinsics.get_view_matrix(),
            viewport_info,
        )

        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS


if __name__ == "__main__":
    main()
