from matplotlib import pyplot as plt
# import os
from pathlib import Path


def plot_metrics_graph(num_examples, stats_dict, dst, amine=None, amine_index=None, models=[], show=False):
    """Plot metrics graphs for all models in comparison

    The graph will have 4 subplots, which are for: accuracy, precision, recall, and bcr, from left to right,
        top to bottom

    Args:
        num_examples:       A list representing the number of examples we are working with at each point.
        stats_dict:         A dictionary with each model as key and a dictionary of model specific metrics as value.
                                Each metric dictionary has the same keys: 'accuracies', 'precisions', 'recalls', 'bcrs',
                                and their corresponding list of values for each model as dictionary values.
        dst:                A string representing the relative path that the graph will be saved to.
                                Graph name and format should be included.
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

    # Temporary category identifier
    with_AL = True

    # Plot each model's metrics
    for model in models:
        if amine:
            # Plotting amine-specific graphs
            num_examples = [i for i in range(len(stats_dict[model]['accuracies'][amine_index]))]
            if len(num_examples) != 1:
                # Plot line graphs for models with active learning
                acc.plot(num_examples, stats_dict[model]
                ['accuracies'][amine_index], 'o-', label=model)
                prec.plot(num_examples, stats_dict[model]
                ['precisions'][amine_index], 'o-', label=model)
                rec.plot(num_examples, stats_dict[model]
                ['recalls'][amine_index], 'o-', label=model)
                bcr.plot(num_examples, stats_dict[model]
                ['bcrs'][amine_index], 'o-', label=model)
            else:
                # Plot bar graphs for models without active learning
                acc.bar(models.index(model)/2, stats_dict[model]['accuracies'][amine_index], 0.35, label=model)
                prec.bar(models.index(model)/2, stats_dict[model]['precisions'][amine_index], 0.35, label=model)
                rec.bar(models.index(model)/2, stats_dict[model]['recalls'][amine_index], 0.35, label=model)
                bcr.bar(models.index(model)/2, stats_dict[model]['bcrs'][amine_index], 0.35, label=model)
                with_AL = False
        else:
            # Plotting avg metrics graph
            num_examples = [i for i in range(len(stats_dict[model]['accuracies']))]
            if len(num_examples) != 1:
                # Plot line graphs for models with active learning
                acc.plot(num_examples, stats_dict[model]
                ['accuracies'], 'o-', label=model, alpha=0.6)
                prec.plot(num_examples, stats_dict[model]
                ['precisions'], 'o-', label=model, alpha=0.6)
                rec.plot(num_examples, stats_dict[model]
                ['recalls'], 'o-', label=model, alpha=0.6)
                bcr.plot(num_examples, stats_dict[model]
                ['bcrs'], 'o-', label=model, alpha=0.6)
            else:
                # Plot bar graphs for models without active learning
                acc.bar(models.index(model)/2, stats_dict[model]['accuracies'], 0.35, label=model)
                prec.bar(models.index(model)/2, stats_dict[model]['precisions'], 0.35, label=model)
                rec.bar(models.index(model)/2, stats_dict[model]['recalls'], 0.35, label=model)
                bcr.bar(models.index(model)/2, stats_dict[model]['bcrs'], 0.35, label=model)
                with_AL = False

    # Increase the font size of the x/y labels
    acc.tick_params(axis='both', labelsize=20)
    prec.tick_params(axis='both', labelsize=20)
    rec.tick_params(axis='both', labelsize=20)
    bcr.tick_params(axis='both', labelsize=20)

    # Adjust the x_ticks for better readability by category
    if with_AL:
        acc.set_xticks(num_examples)
        prec.set_xticks(num_examples)
        rec.set_xticks(num_examples)
        bcr.set_xticks(num_examples)
    else:
        acc.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        prec.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        rec.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        bcr.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Display legends for all subplots
    handles, labels = acc.get_legend_handles_labels()
    num_cols = int(len(labels)/2) if len(labels) > 7 else len(labels)
    fig.legend(handles, labels, loc="lower center", ncol=num_cols, fontsize=18, bbox_to_anchor=[0.5, 0.04])

    # Put x-axis label at the bottom
    fig.text(0.5, 0.02, "Number of samples given", ha="center", va="center", fontsize=28)

    # Set the metrics graph's name and designated folder
    graph_dst = Path(dst)

    # Remove duplicate graphs in case we can't directly overwrite the files
    if graph_dst.exists():
        graph_dst.unlink()

    # Save graph in folder
    plt.savefig(graph_dst)
    # TODO: change to logging.info
    print(f"Graph saved at {graph_dst}")

    if show:
        plt.show()


# TODO: finish this function
def plot_bcr_vs_success_rate():
    """TODO: DOCUMENTATION"""
    return