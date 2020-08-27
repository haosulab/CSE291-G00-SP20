# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from final_env import FinalEnv, SolutionBase
import numpy as np


class Solution(SolutionBase):
    """
    Implement the init function and act functions to control the robot
    Your task is to transport all cubes into the blue bin
    You may only use the following functions

    FinalEnv class:
    - get_agents
    - get_metadata

    Robot class:
    - get_observation
    - configure_controllers
    - set_action
    - get_metadata
    - get_compute_functions

    Camera class:
    - get_metadata
    - get_observation

    All other functions will be marked private at test time, calling those
    functions will result in a runtime error.

    How your solution is tested:
    1. new testing environment is initialized
    2. the init function gets called
    3. the timestep  is set to 0
    4. every 5 time steps, the act function is called
    5. when act function returns False or 200 seconds have passed, go to 1
    """
    def init(self, env: FinalEnv):
        """called before the first step, this function should also reset the state of
        your solution class to prepare for the next run

        """
        pass

    def act(self, env: FinalEnv, current_timestep: int):
        """called at each (actionable) time step to set robot actions. return False to
        indicate that the agent decides to move on to the next environment.
        Returning False early could mean a lower success rate (correctly placed
        boxes / total boxes), but it can also save you some time, so given a
        fixed total time budget, you may be able to place more boxes.

        """
        robot_left, robot_right, camera_front, camera_left, camera_right, camera_top = env.get_agents()

    def get_global_position_from_camera(self, camera, depth, x, y):
        """
        This function is provided only to show how to convert camera observation to world space coordinates.
        It can be removed if not needed.

        camera: an camera agent
        depth: the depth obsrevation
        x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
        """
        cm = camera.get_metadata()
        proj, model = cm['projection_matrix'], cm['model_matrix']
        w, h = cm['width'], cm['height']

        # get 0 to 1 coordinate for (x, y) coordinates
        xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

        # get 0 to 1 depth value at (x,y)
        zf = depth[int(y), int(x)]

        # get the -1 to 1 (x,y,z) coordinate
        ndc = np.array([xf, yf, zf, 1]) * 2 - 1

        # transform from image space to view space
        v = np.linalg.inv(proj) @ ndc
        v /= v[3]

        # transform from view space to world space
        v = model @ v

        return v
