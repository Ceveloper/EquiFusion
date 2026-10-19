import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from .numpy import get_fig_as_nparray, numpy_img_to_tensor

RADIUS = np.array([1,1,1])*0.025 #in m
RESOPTIONS = { # high res
              # lower res
              "figsize":  12,  "linewidth":  2,  
    
                }

def create_pose(ax, plots, vals, limbseq, left_right_limb, pred=True, num_multiple_preds=1, update=False, random_color=False, color=None, **kwargs):

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in limbseq])
    J = np.array([touple[1] for touple in limbseq])
    # Left / right indicator
    LR = np.array([left_right_limb[a] or left_right_limb[b] for a, b in limbseq])
    if pred:
        lcolor = "#D22B2B"
        if num_multiple_preds >1:
            lcolor = "#0000FF"
            if random_color:
                import random
                lcolor = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
        if color is not None:
            lcolor = color
        rcolor = lcolor #"#2ecc71"
    else:
        rcolor = "#383838" # dark grey
        lcolor = rcolor #"#8e8e8e"
        if random_color:
            import random
            lcolor = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
            rcolor = lcolor
        elif color is not None:
            lcolor = color
            rcolor = lcolor

    rang = len(I)*num_multiple_preds if pred else 0
    for i in np.arange(len(I)):
        x = np.array([vals[I[i], 0], vals[J[i], 0]])
        y = np.array([vals[I[i], 1], vals[J[i], 1]])
        z = np.array([vals[I[i], 2], vals[J[i], 2]])
        if not update:

            if i == 0:
                plots.append(ax.plot(x, y, z, c=rcolor if LR[i] else lcolor, lw=RESOPTIONS["linewidth"],
                                     label=['GT' if not pred else 'Pred'])) #lw=2, linestyle='--',
            else:
                plots.append(ax.plot(x, y, z,  lw=RESOPTIONS["linewidth"], c=lcolor if LR[i] else rcolor)) #lw=2, linestyle='--',

        elif update:
            plots[rang+i][0].set_xdata(x)
            plots[rang+i][0].set_ydata(y)
            plots[rang+i][0].set_3d_properties(z)
            plots[rang+i][0].set_color(lcolor if LR[i] else rcolor)
    return plots


def center_around_hip(vals, ax):
    global RADIUS
    rx, ry, rz = RADIUS[0], RADIUS[1], RADIUS[2]
    # remember that y and z are switched in the plot
    xroot, zroot, yroot = vals[0,0], vals[0,1], vals[0,2]
    #uncomment folowing lines to set a limit to the plot (relevant if grid is)
    ax.set_xlim3d([-rx+xroot*2, rx+xroot*2])
    ax.set_ylim3d([-ry+yroot*2, ry+yroot*2])
    ax.set_zlim3d([-rz+zroot*2, rz+zroot*2])

def update(num,data_gt,plots_gt,fig,ax, data_pred=None, center_pose=True, return_img=False, **kwargs):
    
    gt_vals=data_gt[num]
    ax.set_title(ax.get_title(loc='center').split("\n")[0] + f"\nframe {num+1}/{len(data_gt)}")
    
    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True, **kwargs)
    if data_pred is not None:
        if kwargs["multiple_data_pred"]:
            for i,prediction in enumerate(data_pred):
                gt_plots=create_pose(ax,plots_gt,prediction[num],num_multiple_preds=i+1, pred=True,update=True, **kwargs)
        else: 
            vals=data_pred[num]
            plots_gt=create_pose(ax,plots_gt,vals,pred=True,update=True, **kwargs)

    if center_pose:
        center_around_hip(gt_vals, ax)
    if return_img:
        img = get_fig_as_nparray(fig)
        return img
    return plots_gt


def get_np_frames_3d_projection(poses_reshaped3d, limbseq, left_right_limb, data_pred=None, multiple_data_pred=False, xyz_range=None, is_range_sym=False, center_pose=False, units="mm", 
                                as_tensor=False, if_as_overlapping_image=False, orientation_like=None, title=None, center_like=None, if_hide_grid=False, color=None, fig=None):
    assert len(poses_reshaped3d.shape) == 3, poses_reshaped3d.shape
    assert units in ["mm", "bins"]
    assert orientation_like is None or orientation_like in ["motron", "h36m", "somof", "3dpw", "freeman", "best"]
    vals = poses_reshaped3d.copy()
    if xyz_range is not None:
        xyz_range = xyz_range.clone()
    timesteps = vals.shape[0]
    if fig is None:
        fig = plt.figure(figsize=(RESOPTIONS["figsize"], RESOPTIONS["figsize"])) # change here for speed & lightweight, 20
    ax = plt.axes(projection='3d')
    if title is not None:
        ax.set_title(title)
    # vals[:,:,0] = poses_reshaped3d[:,:,2].copy()
    # vals[:,:,2] = poses_reshaped3d[:,:,0].copy()
    if units == "mm":
        vals /= 1000 # from mm to meters
        
    if data_pred is not None:
        data_pred = data_pred.copy()
        if units == "mm":
            data_pred/= 1000
    
    ax.set_xlabel("x")
    ax.set_ylabel("z") 
    ax.set_zlabel("y")

    global RADIUS 
    if xyz_range is not None:
        assert center_pose == False
        if units=="mm":
            xyz_range /= 1000
        if not is_range_sym:
            ax.set_xlim3d([0, xyz_range[0]])
            ax.set_ylim3d([0, xyz_range[2]])
            ax.set_zlim3d([0, xyz_range[1]])
        else: 
            ax.set_xlim3d([-xyz_range[0]/2, xyz_range[0]/2])
            ax.set_ylim3d([-xyz_range[2]/2, xyz_range[2]/2])
            ax.set_zlim3d([-xyz_range[1]/2, xyz_range[1]/2])
    elif center_like is not None:
        if units=="mm":
            RADIUS = (center_like[0].max(axis=0)/1000)//2 # symmetric radius distance from body center
            RADIUS[RADIUS==0] = 3
            center_around_hip(center_like[0]/1000, ax)
        else:
            RADIUS = center_like[0].max(axis=0)//2 # symmetric radius distance from body center
            RADIUS[RADIUS==0] = 1
            center_around_hip(center_like[0], ax)
    else:   
        RADIUS = vals[0].max(axis=0)//2 # symmetric radius distance from body center
        RADIUS[RADIUS==0] = 3
        center_around_hip(vals[0], ax)
    
    ax.set_box_aspect([1,1,1])
    if if_hide_grid:
        ax.grid(False)
        plt.axis('off')
        # following to remove ticks 
        # # following to remove axis labels

    if orientation_like is not None:
        if orientation_like =="h36m":
            ax.view_init(0, -70) # View angle from cameras in h36m, in altitude(angle xzplane with y axis), azimuth
        elif orientation_like =="motron":
            ax.view_init(12, 48) # motron h36m view angle
        elif orientation_like =="freeman" or orientation_like =="best":
            ax.invert_zaxis()
            ax.view_init(azim=-60, elev=30)
        elif orientation_like =="3dpw":
            ax.invert_zaxis()
            ax.view_init(azim=-90, elev=0) # somof view angle
        elif orientation_like =="somof":
            ax.view_init(azim=-90, elev=0) # somof view angle
    
    gt_plots=[]
    pred_plots=[]

    gt_plots=create_pose(ax,gt_plots,vals[0],pred=False,update=False, limbseq=limbseq, left_right_limb=left_right_limb, color=color)
    if data_pred is not None:
        if multiple_data_pred:
            import random
            colors = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]) for i in range(len(data_pred))] 
            for i,prediction in enumerate(data_pred):
                gt_plots=create_pose(ax,gt_plots,prediction[0],pred=True,num_multiple_preds=i+1, update=False, limbseq=limbseq, left_right_limb=left_right_limb, color=colors[i])
        else: 
            gt_plots=create_pose(ax,gt_plots,data_pred[0],pred=True,update=False, limbseq=limbseq, left_right_limb=left_right_limb, color=color)

    
    if if_as_overlapping_image:
        def add_timestep(t):
            if data_pred is not None and t < data_pred.shape[-3]:
                if multiple_data_pred:
                    for i,prediction in enumerate(data_pred):
                        __ = create_pose(ax,gt_plots,prediction[t],pred=True,num_multiple_preds=i+1, update=False, color=colors[i], limbseq=limbseq, left_right_limb=left_right_limb)
                else: 
                    __ =create_pose(ax,gt_plots,data_pred[t],pred=True,update=False, limbseq=limbseq, left_right_limb=left_right_limb, color=color)
            __ = create_pose(ax,gt_plots,vals[t],pred=False,update=False, limbseq=limbseq, left_right_limb=left_right_limb, color=color)
        for t in range(timesteps):
            add_timestep(t)
        frames = [get_fig_as_nparray(fig)]
    else:
        frames = [update(t,vals,gt_plots,fig,ax, data_pred=data_pred, center_pose=center_pose, return_img=True, limbseq=limbseq, left_right_limb=left_right_limb, color=color, multiple_data_pred=multiple_data_pred) for t in range(len(vals))]
    if as_tensor:
        frames = [numpy_img_to_tensor(f) for f in frames]
    plt.close() # necessary for memory if training)
    return frames#, (fig,ax)
