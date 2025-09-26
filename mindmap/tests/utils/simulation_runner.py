# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import multiprocessing
from typing import Any, Callable

from mindmap.isaaclab_utils.simulation_app import SimulationAppContext


def runner(q: multiprocessing.Queue, function: Callable[[Any], bool], *args):
    # Launch the simulator
    with SimulationAppContext(headless=True, enable_cameras=True) as simulation_app:
        # Run the function
        try:
            test_passed = function(simulation_app, *args)
        except Exception as e:
            print(f"Exception occurred while running the policy: {e}")
            test_passed = False
        finally:
            # NOTE(alexmillane, 2025.04.09): Closing the simulation app here causes pytest
            # sessions to prematurely exit. The best solution I've found so far is to run
            # the test in a separate process.
            # Close environment and simulation app
            print("Communicating test result to main process...")
            q.put_nowait(test_passed)


def run_simulation_app_in_separate_process(function: Callable[[Any], bool], *args) -> bool:
    """Run a simulation app in a separate process.

    This is sometimes required to prevent simulation app shutdown interrupting pytest.

    Args:
        function: The function to run in the simulation app.
            - The function should take a SimulationAppContext instance as its first argument,
            and then a variable number of additional arguments.
            - The function should return a boolean indicating whether the test passed.
        *args: The arguments to pass to the function (after the SimulationAppContext instance).

    Returns:
        The boolean result of the function.
    """
    # NOTE(alexmillane, 2025.04.10): I got CUDA issues without this.
    multiprocessing.set_start_method("spawn", force=True)
    # Queue to communicate the test result to the main process.
    q = multiprocessing.Queue()
    # Start the test
    # NOTE(alexmillane, 2025.04.10): We need to start the test in a separate process
    # because the simulation app cannot be closed in the main process, because it
    # kills the entire pytest process.
    p = multiprocessing.Process(target=runner, args=(q, function, *args))
    p.start()
    p.join()

    # NOTE(alexmillane, 2025.04.10): This is sort of a useless check, because the calls to
    # close the simulation app in the child process appear to eat exceptions, so the exitcode
    # is always 0.
    assert p.exitcode == 0, "The closed loop dummy policy failed to run."

    # Get the test result from the child process.
    test_result = q.get()
    # assert test_result, "The closed loop dummy policy failed to run."

    return test_result
