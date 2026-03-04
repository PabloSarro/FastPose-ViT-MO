# The idea behind this script is to show if the new rotation formula works as expected. You can test between three views: when a cube is centered, when it is translated away or when it is centered and its rotation is corrected

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
    glPushMatrix,
    glPopMatrix,
    glMatrixMode,
    glClear,
    glClearColor,
    glMultMatrixf,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_QUADS,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
)
from OpenGL.GLU import gluPerspective
import numpy as np
import torch
from src.utils import get_apparent_orientation, quaternion_to_rotation_matrix


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

        # Use vertical FOV for OpenGL
        self.fov = self.fov_vertical

    def get_matrix(self):
        return np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])


# Initialize Pygame and OpenGL
pygame.init()
display = (1920, 1200)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)


def draw_cube():
    LENGTH = 0.5
    glEnable(GL_DEPTH_TEST)
    glBegin(GL_QUADS)

    # Each face is drawn with vertices in a consistent order
    # Front face (black)
    glColor3f(0.0, 0.0, 0.0)
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

    # Left face (red)
    glColor3f(0.8, 0.2, 0.2)
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


def create_window(width, height, title):
    pygame.display.set_caption(title)
    return pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)


def main():
    pygame.init()
    window_width, window_height = 1920, 1200

    pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Translated Cube View (Press TAB to cycle)")

    camera_intrinsics = CameraIntrinsics()
    camera_pos = [0, 0, 0]
    camera_rot = [0, 0, 0]

    TX = 1.5
    TY = 0.0
    DEPTH = 8.0

    # # Other examples
    # TX = -2.0
    # TX = 0.0
    # TX = 1.0
    # TY = -1.0
    # TY = 3.0
    # DEPTH = 5.0
    # DEPTH = 7.0

    translation = torch.tensor([TX, TY, DEPTH])
    initial_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # # Other examples
    # initial_quat = torch.tensor([0.456,0.881,-0.056,-0.108])
    # initial_quat = torch.tensor([0.843,0.232,-0.289,0.402])

    apparent_rotation = get_apparent_orientation(
        translation.unsqueeze(0), initial_quat.unsqueeze(0)
    ).squeeze(0)
    apparent_rotation_matrix = quaternion_to_rotation_matrix(
        apparent_rotation.unsqueeze(0)
    ).squeeze(0)

    # Create 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = apparent_rotation_matrix.numpy()

    # Create rotation matrix from initial quaternion
    initial_rotation_matrix = (
        quaternion_to_rotation_matrix(initial_quat.unsqueeze(0)).squeeze(0).numpy()
    )
    initial_transform_matrix = np.eye(4)
    initial_transform_matrix[:3, :3] = initial_rotation_matrix

    clock = pygame.time.Clock()
    active_window = "centered"  # Can be "centered", "translated", "centered_rotated" or "translated_rotated_backwards"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    # Cycle through the four windows
                    if active_window == "translated":
                        active_window = "centered_rotated"
                    elif active_window == "centered_rotated":
                        active_window = "translated_rotated_backwards"
                    elif active_window == "translated_rotated_backwards":
                        active_window = "centered"
                    else:
                        active_window = "translated"

                    pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
                    if active_window == "translated":
                        pygame.display.set_caption("Translated Cube View (Press TAB to cycle)")
                    elif active_window == "centered_rotated":
                        pygame.display.set_caption(
                            "Rotated Centered Cube View (Press TAB to cycle)"
                        )
                    elif active_window == "translated_rotated_backwards":
                        pygame.display.set_caption(
                            "Translated Rotated Backwards Cube View (Press TAB to cycle)"
                        )
                    else:
                        pygame.display.set_caption("Centered Cube View (Press TAB to cycle)")

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            camera_intrinsics.fov,
            camera_intrinsics.aspect,
            camera_intrinsics.near,
            camera_intrinsics.far,
        )

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(*camera_pos)
        glRotatef(camera_rot[0], 1, 0, 0)
        glRotatef(camera_rot[1], 0, 1, 0)
        glRotatef(camera_rot[2], 0, 0, 1)
        glRotatef(180, 0, 1, 0)  # Look in the positive z-direction

        if active_window == "translated":
            # Draw translated cube
            glPushMatrix()
            glTranslatef(TX, TY, DEPTH)
            glMultMatrixf(initial_transform_matrix.T)
            draw_cube()
            glPopMatrix()
        elif active_window == "centered_rotated":
            # Draw centered_rotated cube with rotation
            glPushMatrix()
            glTranslatef(0.0, 0.0, DEPTH)
            glMultMatrixf(transform_matrix.T)
            draw_cube()
            glPopMatrix()
        elif active_window == "translated_rotated_backwards":
            # Draw translated cube with backwards apparent rotation
            glPushMatrix()
            glTranslatef(TX, -TY, DEPTH)
            glMultMatrixf(transform_matrix)
            draw_cube()
            glPopMatrix()
        else:  # centered
            # Draw centered cube without rotation
            glPushMatrix()
            glTranslatef(0.0, 0.0, DEPTH)
            glMultMatrixf(initial_transform_matrix.T)
            draw_cube()
            glPopMatrix()

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
