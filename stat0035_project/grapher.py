import libraries as libs


def plot_graph(x, y_list, labels=None, colors=None, x_label=None, y_label=None, title=None, x_limits=None,
               y_limits=None, save_path=None):
    """
    Plots a graph using the given x values and multiple y datasets with optional customization.

    Parameters:
    - x: List or array-like, x-axis values
    - y_list: List of List or array-like, y-axis values for each dataset
    - labels: List of str, optional, labels for each dataset
    - colors: List of str, optional, colors for each dataset
    - x_label: str, optional, label for the x-axis
    - y_label: str, optional, label for the y-axis
    - title: str, optional, title of the graph
    - x_limits: tuple, optional, (min, max) limits for the x-axis
    - y_limits: tuple, optional, (min, max) limits for the y-axis
    - save_path: str, optional, directory to save the figure
    """
    libs.plt.figure(figsize=(8, 6))

    # Plot each dataset
    for i, y in enumerate(y_list):
        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else None
        libs.plt.scatter(x, y, label=label, color=color, marker='o')

    # Set labels and title if provided
    if x_label:
        libs.plt.xlabel(x_label)
    if y_label:
        libs.plt.ylabel(y_label)
    if title:
        libs.plt.title(title)

    # Set axis limits if provided
    if x_limits:
        libs.plt.xlim(x_limits)
    if y_limits:
        libs.plt.ylim(y_limits)

    if labels:
        libs.plt.legend(loc='upper left')  # Add legend in the top-left corner if labels are provided

    libs.plt.grid(True)  # Add a grid for better readability

    # Save the figure if a directory is provided
    if save_path:
        filename = title.replace(" ", "_") + ".png" if title else f"Figure_{id(libs.plt)}.png"
        # replace "_" with " " if a title is given, otherwise just do Figure_x.png
        full_path = libs.os.path.join(save_path, filename)
        libs.plt.savefig(full_path)
        print(f"Figure saved at: {full_path}")

    else:
        # only show if not saving
        libs.plt.show()
