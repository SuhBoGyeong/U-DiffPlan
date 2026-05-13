import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation
from time import asctime
import os

class Visualizer:
    def __init__(self, config, renderer, render_mode, dt):
        self.config = config
        self.renderer = renderer
        self.render_mode = render_mode
        self.dt = dt

    def export_video(self, screen, filename=None):
        """Logic for saving the animation to a video file."""

        render_mode = self.render_mode
        
        if render_mode == "rgb_array":
            fig = plt.figure()
            anim = ArtistAnimation(
                fig,
                [[plt.imshow(arr, animated=True)] for arr in self.renderer.render_list],
                interval=int(1000 * self.dt), blit=True
            )
            for ax in fig.axes:
                ax.axis('off')
            plt.tight_layout()
            anim.save(filename, writer="ffmpeg")
            plt.close(fig)

        elif render_mode == "raw_data":
            if screen is None: return
            fig = plt.figure(figsize=screen.figsize)
            ax = plt.gca()
            track_artists = screen.get_track_artists(ax)
            vehicle_artists = screen.get_vehicle_artists(ax)
            
            FuncAnimation(
                fig,
                lambda idx: screen.update_vehicle_artists(
                    vehicle_artists, *self.renderer.render_list[idx]
                ),
                frames=len(self.renderer.render_list),
                init_func=lambda: screen.reset_vehicle_artists(vehicle_artists),
                interval=int(1000 * self.dt), blit=True
            ).save(filename, writer="ffmpeg")
            plt.close(fig)