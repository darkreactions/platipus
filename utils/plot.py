from matplotlib import pyplot as plt
# import os
from pathlib import Path


def plot_metrics_graph(num_examples, stats_dict, dst, amine=None, amine_index=None, show=False, models=[]):
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
    fig = plt.figure(figsize=(24, 20))

    # Setting up each sub-graph as axes
    # From left to right, top to bottom: Accuracy, Precision, Recall, BCR
    acc = plt.subplot(2, 2, 1)
    acc.set_ylabel('Accuracy', fontsize=20)
    acc.set_title(f'Learning curve for {amine}', fontsize=20) if amine else acc.set_title(
        f'Averaged learning curve', fontsize=20)

    prec = plt.subplot(2, 2, 2)
    prec.set_ylabel('Precision', fontsize=20)
    prec.set_title(f'Precision curve for {amine}', fontsize=20) if amine else prec.set_title(
        f'Averaged precision curve', fontsize=20)

    rec = plt.subplot(2, 2, 3)
    rec.set_ylabel('Recall', fontsize=20)
    rec.set_title(f'Recall curve for {amine}', fontsize=20) if amine else rec.set_title(
        f'Averaged recall curve', fontsize=20)

    bcr = plt.subplot(2, 2, 4)
    bcr.set_ylabel('Balanced Classification Rate', fontsize=20)
    bcr.set_title(f'BCR curve for {amine}', fontsize=20) if amine else bcr.set_title(
        f'Averaged BCR curve', fontsize=20)

    # Exact all models available for plotting
    if not models:
        models = list(stats_dict.keys())

    """# Find the number of points on the x-axis to plot with for avg graph
    if not amine:
        num_examples = [i for i in range(num_examples)]"""

    # Plot each model's metrics
    for model in models:
        if amine:
            # Plotting amine-specific graphs
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
        else:
            # Plotting avg metrics graph
            num_examples = [i for i in range(len(stats_dict[model]['accuracies']))]
            acc.plot(num_examples, stats_dict[model]
            ['accuracies'], 'o-', label=model, alpha=0.6)
            prec.plot(num_examples, stats_dict[model]
            ['precisions'], 'o-', label=model, alpha=0.6)
            rec.plot(num_examples, stats_dict[model]
            ['recalls'], 'o-', label=model, alpha=0.6)
            bcr.plot(num_examples, stats_dict[model]
            ['bcrs'], 'o-', label=model, alpha=0.6)

    # Make the graph more readable
    """# PLATIPUS BASELINE TODO: TEMPORARY FOR AVG GRAPH
    acc.axhline(y=.88, linestyle='-.', linewidth=4, color='r')
    acc.axvline(x=32, linestyle='-.', linewidth=4, color='r')
    acc.annotate('PLATIPUS', (90, .86), fontsize='x-large')

    prec.axhline(y=.62, linestyle='-.', linewidth=4, color='r')
    prec.axvline(x=32, linestyle='-.', linewidth=4, color='r')
    prec.annotate('PLATIPUS', (90, .58), fontsize='x-large')

    rec.axhline(y=.91, linestyle='-.', linewidth=4, color='r')
    rec.axvline(x=32, linestyle='-.', linewidth=4, color='r')
    rec.annotate('PLATIPUS', (90, .87), fontsize='x-large')

    bcr.axhline(y=.87, linestyle='-.', linewidth=4, color='r')
    bcr.axvline(x=32, linestyle='-.', linewidth=4, color='r')
    bcr.annotate('PLATIPUS', (90, .85), fontsize='x-large')"""

    # Get rid of top and right spines for subplots
    # TODO: BULKY
    acc.spines['top'].set_visible(False)
    acc.spines['right'].set_visible(False)
    prec.spines['top'].set_visible(False)
    prec.spines['right'].set_visible(False)
    rec.spines['top'].set_visible(False)
    rec.spines['right'].set_visible(False)
    bcr.spines['top'].set_visible(False)
    bcr.spines['right'].set_visible(False)

    # Increase the font size of the x/y labels
    acc.tick_params(axis='both', labelsize=20)
    prec.tick_params(axis='both', labelsize=20)
    rec.tick_params(axis='both', labelsize=20)
    bcr.tick_params(axis='both', labelsize=20)

    # Display legends for all subplots
    handles, labels = acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(.15, .04), ncol=int(len(labels)/2), fontsize=18)

    # Put x-axis label at the bottom
    fig.text(0.5, 0.02, "Number of samples given", ha="center", va="center", fontsize=28)

    # Set the metrics graph's name and designated folder
    graph_name = 'cv_metrics_{0:s}.png'.format(
        amine) if amine else 'average_metrics.png'

    graph_dst = Path(dst) / Path(graph_name)
    # graph_dst = '{0:s}/{1:s}'.format(dst, graph_name)

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
