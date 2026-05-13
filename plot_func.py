import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def getTrack(filename):
    array=np.loadtxt(filename)
    
    sref=array[:,0]
    xref=array[:,1]
    yref=array[:,2]
    psiref=array[:,3]
    kapparef=array[:,4]
    sinref=array[:,5]
    cosref=array[:,6]
    return sref,xref,yref,psiref,kapparef, sinref, cosref

def get_boundary_future(map_file, traj):
    #! traj.shape = (51, 11)

    TRACK_WIDTH = 0.25
    [Sref,Xref,Yref,Psiref,_,_,_]=getTrack(map_file)
    ## find closest point

    curr = traj[0]
    future = traj[-1]

    start_distances = np.sqrt((Xref - curr[0])**2 + (Yref - curr[1])**2)
    start_idx = np.argmin(start_distances)
    future_distances = np.sqrt((Xref - future[0])**2 + (Yref - future[1])**2)
    future_idx = np.argmin(future_distances)

    return start_idx, future_idx


def plot_visualization(real_EV_past_x, real_EV_past_y, real_TV_past_x, real_TV_past_y,\
                            real_EV_future_x, real_EV_future_y, real_TV_future_x, real_TV_future_y,\
                                pred_TV_future_x, pred_TV_future_y, real_Y, pred_Y, config):

    TRACK_WIDTH = config.track_width
    map_file = config.map_file

    #Setup plot
    plt.figure(figsize=(10, 10))

    ylim_min = np.min([real_EV_future_y, real_TV_future_y])
    xlim_min = np.min([real_EV_future_x, real_TV_future_x])

    ylim_max = np.max([real_EV_future_y, real_TV_future_y])
    xlim_max = np.max([real_EV_future_x, real_TV_future_x])

    plt.ylim(bottom=(ylim_min+ylim_max)/2 - 1.5, top=(ylim_min+ylim_max)/2+1.5)
    plt.xlim(left=(xlim_min+xlim_max)/2-1.5, right=(xlim_min+xlim_max)/2+1.5)
    

    plt.ylabel(r'$Y$ (m)')
    plt.xlabel(r'$X$ (m)')

    # Plot center line
    [Sref,Xref,Yref,Psiref,_,_,_]=getTrack(map_file)
    plt.plot(Xref,Yref,'--',color='k')
    # Draw Trackboundaries
    Xboundleft=Xref-TRACK_WIDTH*np.sin(Psiref)
    Yboundleft=Yref+TRACK_WIDTH*np.cos(Psiref)
    Xboundright=Xref+TRACK_WIDTH*np.sin(Psiref)
    Yboundright=Yref-TRACK_WIDTH*np.cos(Psiref)
    plt.plot(Xboundleft,Yboundleft,color='k',linewidth=1)
    plt.plot(Xboundright,Yboundright,color='k',linewidth=1)

    heatmap = plt.scatter(real_EV_future_x, real_EV_future_y, s=70, cmap=cm.rainbow, color='gray' ,edgecolor='none', marker='o', label='EV future GT ', alpha=0.8)
    plt.plot(real_EV_future_x, real_EV_future_y, alpha=0.3, color='gray', linewidth=2)
    plt.scatter(real_TV_future_x, real_TV_future_y, s=70, cmap=cm.rainbow, color='red' ,edgecolor='none', marker='^', label=f'TV future GT ({real_Y})', alpha=0.8)
    plt.plot(real_TV_future_x, real_TV_future_y, alpha=0.3, color='red', linewidth=2)
    
    plt.scatter(pred_TV_future_x, pred_TV_future_y, s=70, cmap=cm.rainbow, color='blue' ,edgecolor='none', marker='^', label=f'TV future pred ({pred_Y})', alpha=0.8)
    plt.plot(pred_TV_future_x, pred_TV_future_y, alpha=0.3, color='blue', linewidth=2)

    cmap = plt.get_cmap('Greens')

    plt.scatter(real_EV_past_x[-1], real_EV_past_y[-1], marker='o', edgecolor='black', color=(1,1,1,1), linewidth=1.5)
    plt.scatter(real_TV_past_x[-1], real_TV_past_y[-1], marker='o', edgecolor='black', color=(1,1,1,1), linewidth=1.5)
    
    # cbar = plt.colorbar(heatmap, fraction=0.035)
    # cbar.set_label("velocity in [kph]")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.clim(0,170)    
    plt.tight_layout()                

    plt.legend()
   