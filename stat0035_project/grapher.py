import libraries as libs


def contains_illegal_chars(value, name):
    illegal_characters = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']
    if any(char in value for char in illegal_characters):
        raise ValueError(f"{name} contains illegal characters: {illegal_characters}")


def plot_graph(x, y_list, model_history_index, labels=None, colors=None, x_label=None, y_label=None, title=None,
               x_limits=None,
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

    illegal_characters = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']

    # Validate labels
    if x_label:
        contains_illegal_chars(x_label, "x_label")
    if y_label:
        contains_illegal_chars(y_label, "y_label")

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
        filename = ''

        if title:
            filename = title.replace(" ", "_")
        else:
            filename = f"{x_label} vs {y_label} - Modelling History Index {model_history_index} - {libs.datetime.now().strftime('%Y-%m-%d_%H-%M')}".replace(
                " ", "_")

        # replace "_" with " " if a title is given
        # otherwise just do "X vs. Y - Modelling History Index z - 1999-01-01_00-59.png"
        full_path = libs.os.path.join(save_path, filename + '.png')
        libs.plt.savefig(full_path)
        print(f"Figure saved at: {full_path}")

    else:
        # only show if not saving
        libs.plt.show()
