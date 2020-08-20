"""
RTPT class to rename your processes giving information on who is launching the
process, and the remaining time for it.
Created to be used with our AIML IRON table.
"""
from time import time
from setproctitle import setproctitle

from collections import deque


class RTPT:
    def __init__(
        self,
        name_initials: str,
        experiment_name: str,
        max_iterations: int,
        iteration_start=0,
        moving_avg_window_size=20,
        update_interval=1,
    ):
        """
        Initialize the Remaining-Time-To-Process (RTPT) object.

        Args:
            name_initials (str): Name initials (e.g. "QD" for "Quentin Delfosse").
            experiment_name (str): A unique name to identify the running experiment. Must be less than 30 characters.
                Spaces will be replaced with underscores.
            max_iterations (int): The maximum number of iterations.
            iteration_start (int): The iteration at which to start (optional, default: 0).
            moving_avg_window_size (int): The window size for the moving average for the ETA approximation (optional, default: 20).
            update_interval (int): After how many iterations the title should be updated (optional, default: 1).
        """
        # Some assertions upfront
        assert (
            len(experiment_name) < 30
        ), f"Experiment name has to be less than 30 chars but was {len(experiment_name)}."
        assert (
            max_iterations > 0
        ), f"Maximum number of iterations has to be greater than 0 but was {max_iterations}"
        assert (
            iteration_start >= 0
        ), f"Starting iteration count must be equal or greater than 0 but was {iteration_start}"

        # Store args
        self.name_initials = name_initials
        self.experiment_name = experiment_name.replace(" ", "_")
        self._current_iteration = iteration_start
        self.max_iterations = max_iterations
        self.update_interval = update_interval

        # Store time for each iterations in a deque
        self.deque = deque(maxlen=moving_avg_window_size)

        # Track end of iterations
        self._last_iteration_time_end = None

        # Perform an initial title update
        self._update_title()


    def start(self):
        """Start the internal iteration timer."""
        self._last_iteration_time_start = time()

    def step(self):
        """
        Perform an update step:
        - Measure new time for the last epoch
        - Update deque
        - Compute new ETA from deque
        - Set new process title with the latest ETA
        """
        # Add the time delta of the current iteration to the deque
        time_end = time()
        time_delta = time_end - self._last_iteration_time_start
        self.deque.append(time_delta)

        self._update_title()
        self._current_iteration += 1
        self._last_iteration_time_start = time_end

    def _moving_average_seconds_per_iteration(self):
        """Compute moving average of seconds per iteration."""
        return sum(list(self.deque)) / len(self.deque)

    def _get_eta_str(self):
        """Get the eta string in the format 'Dd:H:M:S'."""
        # TODO: This is currently expected and hardcoded in IRON for the first iteration
        if self._current_iteration == 0:
            return "first_epoch"

        # Get mean seconds per iteration
        avg = self._moving_average_seconds_per_iteration()

        # Compute the ETA based on the remaining number of iterations
        remaining_iterations = self.max_iterations - self._current_iteration
        c = remaining_iterations * avg

        # Compute days/hours/minutes/seconds
        days = round(c // 86400)
        hours = round(c // 3600 % 24)
        minutes = round(c // 60 % 60)
        seconds = round(c % 60)

        # Format
        eta_str = f"{days}d:{hours}:{minutes}:{seconds}"

        return eta_str

    def _get_title(self):
        """Get the full process title. Includes name initials, base name and ETA."""
        # Obtain the ETA
        eta_str = self._get_eta_str()

        # Construct the title
        title = f"@{self.name_initials}_{self.experiment_name}#{eta_str}"

        return title

    def _update_title(self):
        """Update the process title."""
        title = self._get_title()

        if self._current_iteration % self.update_interval == 0:
            setproctitle(title)
