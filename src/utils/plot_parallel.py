import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from .plot import create_pose


RESOPTIONS = {}
RESOPTIONS["linewidth"] = 2
RESOPTIONS["figsize"] = 6


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])


def plot_skeleton(vals, limbseq, ax, update=False, plots=None, color="g"):
    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in limbseq])
    J = np.array([touple[1] for touple in limbseq])
    if plots is None:
        plots = []
    for i in np.arange(len(I)):
        x = np.array([vals[I[i], 0], vals[J[i], 0]])
        y = np.array([vals[I[i], 1], vals[J[i], 1]])
        z = np.array([vals[I[i], 2], vals[J[i], 2]])
        if not update:
            plots.append(
                ax.plot(x, y, z, lw=RESOPTIONS["linewidth"], color=color)[0]
            )  # , label=['GT' if not pred else 'Pred']) #lw=2, linestyle='--',
        elif update:
            plots[i].set_xdata(x)
            plots[i].set_ydata(y)
            plots[i].set_3d_properties(z)
            plots[i].set_color(color)
    return plots


def get_drawing_funct(
    kpts3d_obs, kpts3d_gt, kpts3d_all, plots, axes, skeleton, keep_gt=False
):
    kwargs = {"limbseq": skeleton.limbseq, "left_right_limb": skeleton.left_right_limb}
    n_limbs = len(kwargs["limbseq"])

    def iterate_over_2daxes(axes):
        for axarr in axes:
            for ax in axarr:
                yield ax

    def plot_multiple_pred(ax, ax_plot, ax_kpts_list, n, keep_gt=False):
        assert not keep_gt
        colors = ["g", "b"]
        # change color of first limb segments
        create_pose(
            ax,
            ax_plot,
            vals=ax_kpts_list[0][n],
            pred=False,
            num_multiple_preds=1,
            update=True,
            random_color=False,
            color="r",
            **kwargs,
        )
        # plot other predictions
        for i, pred in enumerate(ax_kpts_list[1:]):
            if len(ax_plot) < ax_kpts_list[0].shape[-2] * (i + 1):
                create_pose(
                    ax,
                    ax_plot,
                    vals=pred[n],
                    pred=True,
                    num_multiple_preds=i + 1,
                    update=False,
                    random_color=False,
                    color=colors[i],
                    **kwargs,
                )
            else:
                create_pose(
                    ax,
                    ax_plot,
                    vals=pred[n],
                    pred=True,
                    num_multiple_preds=i + 1,
                    update=True,
                    random_color=False,
                    color=colors[i],
                    **kwargs,
                )

    def plot_gt_and_pred(ax, ax_plots, ax_kpts_gt, ax_kpts, n):
        # first get target gt
        create_pose(
            ax,
            ax_plots,
            vals=ax_kpts_gt[n],
            pred=False,
            num_multiple_preds=1,
            update=True,
            random_color=False,
            **kwargs,
        )
        # and closestGT overlapped
        if len(ax_plots) < n_limbs + 2:
            create_pose(
                ax,
                ax_plots,
                ax_kpts[0],
                pred=True,
                update=False,
                num_multiple_preds=1,
                **kwargs,
            )
        else:
            create_pose(
                ax,
                ax_plots,
                vals=ax_kpts[n],
                pred=True,
                num_multiple_preds=1,
                update=True,
                random_color=False,
                **kwargs,
            )

    def draw_pred_ontopof_obs(ax, ax_plots, ax_kpts, n):
        create_pose(
            ax,
            ax_plots,
            vals=ax_kpts[n],
            pred=False,
            num_multiple_preds=1,
            update=True,
            random_color=False,
            color="r",
            **kwargs,
        )

    def drawframe(n):
        for ax in iterate_over_2daxes(axes):
            ax.set_title(
                ax.get_title(loc="center").split("\n")[0]
                + f"\nframe {n+1}/{len(kpts3d_obs) + len(kpts3d_gt)}"
            )
        if n < len(kpts3d_obs):
            # draw obervation
            for i, ax in enumerate(iterate_over_2daxes(axes)):
                ax_plots = plots[i]
                create_pose(
                    ax,
                    ax_plots,
                    vals=kpts3d_obs[n],
                    pred=False,
                    num_multiple_preds=1,
                    update=True,
                    random_color=False,
                    color=None,
                    **kwargs,
                )
        else:
            # draw prediction
            n -= len(kpts3d_obs)

            for i, (ax_plots, ax, vals) in enumerate(
                zip(plots, iterate_over_2daxes(axes), kpts3d_all)
            ):
                if i == 0 or keep_gt:
                    plot_gt_and_pred(ax, ax_plots, kpts3d_gt, vals, n)
                else:
                    draw_pred_ontopof_obs(ax, ax_plots, vals, n)

        return plots

    return drawframe


def create_plot_canvas(
    kpts3d_obs, plot_titles, figsize, skeleton, n_rows=None, n_cols=None
):
    kwargs = {"limbseq": skeleton.limbseq, "left_right_limb": skeleton.left_right_limb}
    fig = plt.figure(
        figsize=(figsize, figsize)
    )  # change here for speed & lightweight, 20
    label = "Multiple Predictions"

    nrows = (
        int(len(plot_titles) // 3) + int(len(plot_titles) % 3 != 0)
        if n_rows is None
        else n_rows
    )
    ncols = 3 if n_cols is None else n_cols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, subplot_kw=dict(projection="3d"), figsize=(16, 16)
    )
    title = f"Prediction  {label}"
    fig.suptitle(title, fontsize=16)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    for axarr in axes:
        for ax in axarr:
            ax.set_box_aspect([1, 1, 1])
            set_axes_equal(ax)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_zlabel("y")
        if nrows == 1:
            break

    plots = []
    for i, axarr in enumerate(axes):
        for j, ax in enumerate(axarr):
            plots.append(
                create_pose(
                    ax,
                    [],
                    vals=kpts3d_obs[0],
                    pred=False,
                    num_multiple_preds=1,
                    update=False,
                    **kwargs,
                )
            )
            idx = i * ncols + j
            ax.set_title(plot_titles[idx] if idx < len(plot_titles) else "")
            ax.view_init(elev=15.0, azim=70)
        if nrows == 1:
            break
    return fig, axes, plots
