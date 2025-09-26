# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
import sys
import traceback

from isaaclab.app import AppLauncher
import omni.kit.app


def get_isaac_sim_version() -> str:
    """Get the version of Isaac Sim."""
    return omni.kit.app.get_app().get_app_version()


class SimulationAppContext:
    """Context manager for launching and closing a simulation app."""

    def __init__(self, headless: bool, enable_cameras: bool):
        """
        Args:
            headless (bool): Whether to run the app in headless mode.
            enable_cameras (bool): Whether to enable cameras.
        """
        self.app_launcher = None
        self.headless = headless
        self.enable_cameras = enable_cameras

    def is_running(self) -> bool:
        return self.app_launcher.app.is_running()

    def is_exiting(self) -> bool:
        return self.app_launcher.app.is_exiting()

    def __enter__(self):
        # NOTE(alexmillane, 2025.05.13): Import pinocchio before launching the app to avoid version conflicts.
        # This is a work-around for a conflict in the assimp versions used by pinocchio and Isaac Sim/Lab.
        # See thread here: https://nvidia.slack.com/archives/C06HLQ6CB41/p1741907494471379
        # From the slack thread, the issue appears to be fixed internally, but the fix is not yet released.
        # Remove this function once the issue is fixed in a released version of Isaac Sim.
        # We warn if IsaacSim has been upgraded.
        import pinocchio

        self.app_launcher = AppLauncher(headless=self.headless, enable_cameras=self.enable_cameras)
        if get_isaac_sim_version() != "4.5.0":
            print(f"WARNING: IsaacSim has been upgraded to {get_isaac_sim_version()}.")
            print(f"Please remove the pinocchio related hacks in: simulation_app.py")

        # Make sure that MINDMAP's custom tasks are registered
        import mindmap.tasks.task_definitions

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing simulation app")
        # app_launcher.close() will terminate the whole process with exit code 0, i.e. preventing errors from being seen by the caller. There are seemingly no ways around this.
        # As a workaround, we call os._exit(1) that terminates immediately. The downside is that any cleanup would be omitted
        if exc_type is None:
            self.app_launcher.app.close()
        else:
            print(f"Exception caught in SimulationAppContext: {exc_type.__name__}: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            print("Killing the process without cleaning up")
            os._exit(1)
