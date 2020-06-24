from matplotlib import pyplot as plt
#import os
from pathlib import Path


def plot_metrics_graph(num_examples, stats_dict, dst, amine=None, amine_index=0, show=False, models=[]):
    """Plot metrics graphs for all models in comparison

    The graph will have 4 subplots, which are for: accuracy, precision, recall, and bcr, from left to right,
        top to bottom

    Args:
        num_examples:       A list representing the number of examples we are working with at each point.
        stats_dict:         A dictionary with each model as key and a dictionary of model specific metrics as value.
                                Each metric dictionary has the same keys: 'accuracies', 'precisions', 'recalls', 'bcrs',
                                and their corresponding list of values for each model as dictionary values.
        dst:                A string representing the folder that the graph will be saved in.
        amine:              A string representing the amine that our model metrics are for. Default to be None.
        show:               A boolean representing whether we want to show the graph or not. Default to False to
                                seamlessly run the whole model,

    Returns:
        N/A
    """

    # Set up initial figure for plotting
    fig = plt.figure(figsize=(16, 12))

    # Setting up each sub-graph as axes
    # From left to right, top to bottom: Accuracy, Precision, Recall, BCR
    acc = plt.subplot(2, 2, 1)
    acc.set_ylabel('Accuracy')
    acc.set_title(f'Learning curve for {amine}') if amine else acc.set_title(
        f'Averaged learning curve')

    prec = plt.subplot(2, 2, 2)
    prec.set_ylabel('Precision')
    prec.set_title(f'Precision curve for {amine}') if amine else prec.set_title(
        f'Averaged precision curve')

    rec = plt.subplot(2, 2, 3)
    rec.set_ylabel('Recall')
    rec.set_title(f'Recall curve for {amine}') if amine else rec.set_title(
        f'Averaged recall curve')

    bcr = plt.subplot(2, 2, 4)
    bcr.set_ylabel('Balanced classification rate')
    bcr.set_title(f'BCR curve for {amine}') if amine else bcr.set_title(
        f'Averaged BCR curve')

    if not models:
        models = stats_dict.keys()

    # Plot each model's metrics
    for model in models:
        num_examples = [i for i in range(
            len(stats_dict[model]['accuracies'][amine_index]))]
        acc.plot(num_examples, stats_dict[model]
                 ['accuracies'][amine_index], 'o-', label=model)
        prec.plot(num_examples, stats_dict[model]
                  ['precisions'][amine_index], 'o-', label=model)
        rec.plot(num_examples, stats_dict[model]
                 ['recalls'][amine_index], 'o-', label=model)
        bcr.plot(num_examples, stats_dict[model]
                 ['bcrs'][amine_index], 'o-', label=model)

    # Display subplot legends
    acc.legend()
    prec.legend()
    rec.legend()
    bcr.legend()

    fig.text(0.5, 0.04, "Number of samples given", ha="center", va="center")

    # Set the metrics graph's name and designated folder
    graph_name = 'cv_metrics_{0:s}.png'.format(
        amine) if amine else 'average_metrics.png'

    graph_dst = Path(dst) / Path(graph_name)
    #graph_dst = '{0:s}/{1:s}'.format(dst, graph_name)

    # Remove duplicate graphs in case we can't directly overwrite the files
    # if os.path.isfile(graph_dst):
    #    os.remove(graph_dst)
    if graph_dst.exists():
        graph_dst.unlink()

    # Save graph in folder
    plt.savefig(graph_dst)
    print(f"Graph {graph_name} saved in folder {dst}")

    if show:
        plt.show()
