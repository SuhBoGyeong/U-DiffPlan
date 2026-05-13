from time import sleep, time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import figaspect
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy import sin, cos, hstack, min as npmin, max as npmax, asarray, zeros
from numpy.linalg import norm
import numpy as np
from matplotlib.patches import Polygon


# Vehicle artist indices
IDX_EV_TRAJ = 0
IDX_EV_VEHICLE = 1
IDX_TV_TRAJ = 2
IDX_TV_VEHICLE = 3
IDX_TV_PRED_TRAJ_1 = 4
IDX_TV_PRED_VEHICLE_1 = 5
IDX_TV_PRED_TRAJ_2 = 6
IDX_TV_PRED_VEHICLE_2 = 7
IDX_EV_PRED_TRAJ_1 = 8
IDX_EV_PRED_VEHICLE_1 = 9
IDX_EV_PRED_TRAJ_2 = 10
IDX_EV_PRED_VEHICLE_2 = 11
IDX_TV_MU_PATH = 12
IDX_TV_ENVELOPE = 13


class LivePlot2Players:

    def __init__(self, track, vehicle_model, mode="rgb_array", vmin=0.0, vmax=4.5):
        self.mode = mode
        self.track = track
        self.model = vehicle_model
        self.vmin = vmin
        self.vmax = vmax

        self._init_track_borders(track, vehicle_model.safe_distance)
        self._init_figure()

    def _init_track_borders(self, track, safe_distance):
        """Initialize track border coordinates."""
        sin_psiref = sin(track.psiref)
        cos_psiref = cos(track.psiref)

        self.border_left_x = track.Xref - track.border_left * sin_psiref
        self.border_left_y = track.Yref + track.border_left * cos_psiref
        self.border_right_x = track.Xref + track.border_right * sin_psiref
        self.border_right_y = track.Yref - track.border_right * cos_psiref

        self.safe_border_left_x = track.Xref - (track.border_left - safe_distance) * sin_psiref
        self.safe_border_left_y = track.Yref + (track.border_left - safe_distance) * cos_psiref
        self.safe_border_right_x = track.Xref + (track.border_right - safe_distance) * sin_psiref
        self.safe_border_right_y = track.Yref - (track.border_right - safe_distance) * cos_psiref

    def _init_figure(self):
        """Initialize matplotlib figure and axes."""
        xmin = min(npmin(self.border_left_x), npmin(self.border_right_x))
        xmax = max(npmax(self.border_left_x), npmax(self.border_right_x))
        ymin = min(npmin(self.border_left_y), npmin(self.border_right_y))
        ymax = max(npmax(self.border_left_y), npmax(self.border_right_y))

        margin = max(xmax - xmin, ymax - ymin) * 0.01
        self.figsize = figaspect((ymax - ymin + 120 * margin) / (xmax - xmin + 150 * margin))

        self.fig = plt.figure(figsize=self.figsize)
        self.ax = plt.gca()
        self.ax.set_xlim(left=xmin - margin, right=xmax + margin)
        self.ax.set_ylim(bottom=ymin - margin, top=ymax + margin)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x[m]')
        self.ax.set_ylabel('y[m]')

        self.track_artists = self._create_track_artists()
        self.vehicle_artists = self._create_vehicle_artists()

        self.canvas_agg = FigureCanvasAgg(self.fig)
        self.canvas_agg.draw()
        for a in self.track_artists:
            self.ax.draw_artist(a)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    # -------------------------------------------------------------------------
    # Artist creation helpers
    # -------------------------------------------------------------------------
    def _create_scatter(self, cmap, marker='o', alpha=1.0):
        """Create a scatter artist with common settings."""
        return self.ax.scatter(
            [], [], c=[],
            s=16, vmin=self.vmin, vmax=self.vmax,
            cmap=cmap, edgecolor='none', marker=marker, alpha=alpha, animated=True
        )

    def _create_line(self, color, linewidth=1, linestyle='-', alpha=1.0):
        """Create a line artist with common settings."""
        return self.ax.plot(
            [], [], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, animated=True
        )[0]

    def _create_track_artists(self):
        """Create track visualization artists with data."""
        # Center line
        center_line = self._create_line('k', linewidth=1, linestyle='--')
        center_line.set_data(self.track.Xref, self.track.Yref)

        # Left border
        left_border = self._create_line('k', linewidth=1)
        left_border.set_data(self.border_left_x, self.border_left_y)

        # Right border
        right_border = self._create_line('k', linewidth=1)
        right_border.set_data(self.border_right_x, self.border_right_y)

        # Safe left border
        safe_left = self._create_line('r', linewidth=0.5)
        safe_left.set_data(self.safe_border_left_x, self.safe_border_left_y)

        # Safe right border
        safe_right = self._create_line('r', linewidth=0.5)
        safe_right.set_data(self.safe_border_right_x, self.safe_border_right_y)

        return [center_line, left_border, right_border, safe_left, safe_right]

    def _create_vehicle_artists(self):
        """Create vehicle visualization artists."""
        tv_envelope_poly = Polygon(
            [[0, 0]], closed=True,
            facecolor='tab:blue', alpha=0.2, edgecolor='none', animated=True
        )

        artists = [None] * 14

        # EV (Ego Vehicle)
        artists[IDX_EV_TRAJ] = self._create_scatter(cm.rainbow)
        artists[IDX_EV_VEHICLE] = self._create_line('k')

        # TV (Target Vehicle)
        artists[IDX_TV_TRAJ] = self._create_scatter('viridis')
        artists[IDX_TV_VEHICLE] = self._create_line('r')

        # TV predictions (x scatter removed)
        artists[IDX_TV_PRED_TRAJ_1] = self._create_scatter('gist_ncar', marker='x', alpha=0.0)  # Hidden
        artists[IDX_TV_PRED_VEHICLE_1] = self._create_line('r')
        artists[IDX_TV_PRED_TRAJ_2] = self._create_scatter('Wistia', alpha=0.3)
        artists[IDX_TV_PRED_VEHICLE_2] = self._create_line('r')

        # EV predictions (gray x scatter removed)
        artists[IDX_EV_PRED_TRAJ_1] = self._create_scatter('gray', marker='x', alpha=0.0)  # Hidden
        artists[IDX_EV_PRED_VEHICLE_1] = self._create_line('k')
        artists[IDX_EV_PRED_TRAJ_2] = self._create_scatter('spring', alpha=0.2)
        artists[IDX_EV_PRED_VEHICLE_2] = self._create_line('k')

        # TV mean path and envelope
        artists[IDX_TV_MU_PATH] = self._create_line('orange', linewidth=1.8, linestyle='--')
        artists[IDX_TV_ENVELOPE] = self.ax.add_patch(tv_envelope_poly)

        return artists

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def show(self):
        self.fig.show()

    def start_event_loop(self, timeout=0.001):
        self.fig.canvas.start_event_loop(timeout)

    def get_data(self):
        return asarray(self.canvas_agg.buffer_rgba()).copy()

    def get_track_artists(self, ax):
        """Get track artists (for backward compatibility)."""
        return self.track_artists

    def get_vehicle_artists(self, ax):
        """Get vehicle artists (for backward compatibility)."""
        return self.vehicle_artists

    def update(self, state, trajectory, opp_state, opp_trajectory,
               EV_dec_prob, TV_dec_prob, EV_decision, TV_decision,
               TV_pred_trajectory, EV_pred_trajectory, mu, mu_TV,
               TV_pred_trajectory_mean=None, TV_pred_trajectory_log_var=None):
        """Update plot with new data."""
        self.fig.canvas.restore_region(self.background)

        self._update_vehicle_artists(
            state, trajectory, opp_state, opp_trajectory,
            TV_pred_trajectory, EV_pred_trajectory,
            TV_pred_trajectory_mean, TV_pred_trajectory_log_var
        )

        for a in self.vehicle_artists:
            self.ax.draw_artist(a)


    # -------------------------------------------------------------------------
    # Artist update helpers
    # -------------------------------------------------------------------------
    def _update_scatter(self, artist, trajectory):
        """Update scatter artist with trajectory data."""
        if trajectory is not None:
            artist.set_offsets(trajectory[:, :2])
            artist.set_array(norm(trajectory[:, 3:5], axis=1))

    def _update_vehicle_outline(self, artist, state):
        """Update vehicle outline artist."""
        vx, vy = self.model.footprint(state[0], state[1], state[2])
        artist.set_data(hstack((vx, vx[:1])), hstack((vy, vy[:1])))

    def _update_vehicle_artists(self, state, trajectory, opp_state, opp_trajectory,
                                  TV_pred_trajectory, EV_pred_trajectory,
                                  TV_pred_trajectory_mean=None, TV_pred_trajectory_log_var=None):
        """Update all vehicle artists."""
        artists = self.vehicle_artists

        # Update EV trajectory and vehicle
        self._update_scatter(artists[IDX_EV_TRAJ], trajectory)
        self._update_vehicle_outline(artists[IDX_EV_VEHICLE], state)

        # Update TV trajectory and vehicle
        self._update_scatter(artists[IDX_TV_TRAJ], opp_trajectory)
        self._update_vehicle_outline(artists[IDX_TV_VEHICLE], opp_state)

        # Update TV predictions (x scatter skipped)
        if TV_pred_trajectory is not None:
            # Skip IDX_TV_PRED_TRAJ_1 (x scatter - hidden)
            self._update_vehicle_outline(artists[IDX_TV_PRED_VEHICLE_1], opp_state)
            self._update_vehicle_outline(artists[IDX_TV_PRED_VEHICLE_2], opp_state)

            # Update envelope if mean/log_var provided
            if TV_pred_trajectory_mean is not None and TV_pred_trajectory_log_var is not None:
                self._update_envelope(
                    artists[IDX_TV_ENVELOPE],
                    TV_pred_trajectory_mean.astype('float64'),
                    TV_pred_trajectory_log_var.astype('float64')
                )

        # Update EV predictions (gray x scatter skipped)
        if EV_pred_trajectory is not None:
            # Skip IDX_EV_PRED_TRAJ_1 (gray x scatter - hidden)
            self._update_vehicle_outline(artists[IDX_EV_PRED_VEHICLE_1], state)
            self._update_vehicle_outline(artists[IDX_EV_PRED_VEHICLE_2], state)

    def _update_envelope(self, artist, mean, log_var):
        """Update uncertainty envelope polygon."""
        dx = np.gradient(mean[:, 0])
        dy = np.gradient(mean[:, 1])
        norm_dxdy = np.hypot(dx, dy)
        nx, ny = -dy / norm_dxdy, dx / norm_dxdy

        r = np.sqrt(log_var[:, 0] ** 2 + log_var[:, 1] ** 2)
        left = np.column_stack([mean[:, 0] + r * nx, mean[:, 1] + r * ny])
        right = np.column_stack([mean[:, 0] - r * nx, mean[:, 1] - r * ny])
        envelope = np.vstack([left, right[::-1]])

        artist.set_xy(envelope)

    def update_vehicle_artists(self, vehicle_artists, state, trajectory, opp_state, opp_trajectory,
                                TV_pred_trajectory, EV_pred_trajectory,
                                TV_pred_trajectory_mean=None, TV_pred_trajectory_log_var=None):
        """Update vehicle artists (for backward compatibility)."""
        self._update_vehicle_artists(
            state, trajectory, opp_state, opp_trajectory,
            TV_pred_trajectory, EV_pred_trajectory,
            TV_pred_trajectory_mean, TV_pred_trajectory_log_var
        )
        return vehicle_artists

    def reset_vehicle_artists(self, vehicle_artists):
        """Reset vehicle artists to empty state."""
        vehicle_artists[IDX_EV_TRAJ].set_offsets(zeros((0, 2)))
        vehicle_artists[IDX_EV_TRAJ].set_array([])
        vehicle_artists[IDX_EV_VEHICLE].set_data([], [])
        return vehicle_artists


class Clock:
    """Simple clock for frame rate control."""

    def __init__(self):
        self.time_last_tick = None
        self.duration = None

    def tick(self, framerate):
        if self.time_last_tick is None:
            self.time_last_tick = time()
        else:
            time_current_tick = time()
            timeout = self.time_last_tick + 1.0 / framerate - time_current_tick
            if timeout > 0.0:
                sleep(timeout)
                time_current_tick = time()
            self.duration = time_current_tick - self.time_last_tick
            self.time_last_tick = time_current_tick

    def get_fps(self):
        if self.duration is None:
            return -1.0
        return 1.0 / self.duration
