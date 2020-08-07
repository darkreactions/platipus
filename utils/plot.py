import os
from collections import defaultdict
#from matplotlib import pyplot as plt
from pathlib import Path

from utils import read_pickle, find_avg_metrics, find_success_rate, find_bcr, find_winning_models
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_categorical_graph(stats_dict, dst, style_combinations, amine=None, amine_index=None, models=None,
                           show=False):
    """Plot metrics graphs for all models under the same category in comparison

    The graph will have 4 subplots, which are for: accuracy, precision, recall, and bcr, from left to right,
        top to bottom

    Args:
        stats_dict:         A dictionary with each model as key and a dictionary of model specific metrics as value.
                                Each metric dictionary has the same keys: 'accuracies', 'precisions', 'recalls', 'bcrs',
                                and their corresponding list of values for each model as dictionary values.
        dst:                A string representing the relative path that the graph will be saved to.
                                Graph name and format should be included.
        style_combinations: A dictionary with each model as key and another dictionary as value, which has
                                color/linestyle as key and the corresponding color/marker/linestyle as value.
        amine:              A string representing the amine that our model metrics are for. Default to be None.
        amine_index:        An integer representing the index of the amine to plot.
        models:             A list representing the names all the models to plot.
        show:               A boolean representing whether we want to show the graph or not. Default to False to
                                seamlessly run the whole model,
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

    # Exact all models available for plotting if not given as input
    if not models:
        models = list(stats_dict.keys())

    # Temporary category identifier to determine line chart of bar chart
    with_AL = True

    # Plot each model's metrics
    for model in models:
        color = style_combinations[model]['color']
        marker = style_combinations[model]['marker']
        if amine:
            # Plotting amine-specific graphs
            num_examples = [i for i in range(len(stats_dict[model]['accuracies'][amine_index]))]
            if len(num_examples) != 1:
                # Plot line graphs for models with active learning
                acc.plot(num_examples, stats_dict[model]['accuracies'][amine_index], marker=marker, markersize=12,
                         color=color, label=model, alpha=0.6)
                prec.plot(num_examples, stats_dict[model]['precisions'][amine_index], marker=marker, markersize=12,
                         color=color, label=model, alpha=0.6)
                rec.plot(num_examples, stats_dict[model]['recalls'][amine_index], marker=marker, markersize=12,
                         color=color, label=model, alpha=0.6)
                bcr.plot(num_examples, stats_dict[model]['bcrs'][amine_index], marker=marker, markersize=12,
                         color=color, label=model, alpha=0.6)
            else:
                # Plot bar graphs for models without active learning
                acc.bar(models.index(model) / 2, stats_dict[model]['accuracies'][amine_index], 0.25, color=color,
                        label=model, alpha=0.6)
                prec.bar(models.index(model) / 2, stats_dict[model]['precisions'][amine_index], 0.25, color=color,
                        label=model, alpha=0.6)
                rec.bar(models.index(model) / 2, stats_dict[model]['recalls'][amine_index], 0.25, color=color,
                        label=model, alpha=0.6)
                bcr.bar(models.index(model) / 2, stats_dict[model]['bcrs'][amine_index], 0.25, color=color,
                        label=model, alpha=0.6)
                with_AL = False
        else:
            # Plotting avg metrics graph
            num_examples = [i for i in range(len(stats_dict[model]['accuracies']))]
            if len(num_examples) != 1:
                # Plot line graphs for models with active learning
                acc.plot(num_examples, stats_dict[model]['accuracies'], marker=marker, markersize=12,
                         color=color, label=model, alpha=0.6)
                prec.plot(num_examples, stats_dict[model]['precisions'], marker=marker, markersize=12,
                          color=color, label=model, alpha=0.6)
                rec.plot(num_examples, stats_dict[model]['recalls'], marker=marker, markersize=12,
                         color=color, label=model, alpha=0.6)
                bcr.plot(num_examples, stats_dict[model]['bcrs'], marker=marker, markersize=12,
                         color=color, label=model, alpha=0.6)
            else:
                # Plot bar graphs for models without active learning
                acc.bar(models.index(model) / 2, stats_dict[model]['accuracies'], 0.25, color=color,
                        label=model, alpha=0.6)
                prec.bar(models.index(model) / 2, stats_dict[model]['precisions'], 0.25, color=color,
                         label=model, alpha=0.6)
                rec.bar(models.index(model) / 2, stats_dict[model]['recalls'], 0.25, color=color,
                        label=model, alpha=0.6)
                bcr.bar(models.index(model) / 2, stats_dict[model]['bcrs'], 0.25, color=color,
                        label=model, alpha=0.6)
                with_AL = False

    # Making the graphs more readable
    # Increase the font size of the x/y labels
    acc.tick_params(axis='both', labelsize=20)
    prec.tick_params(axis='both', labelsize=20)
    rec.tick_params(axis='both', labelsize=20)
    bcr.tick_params(axis='both', labelsize=20)

    # Adjust the x_ticks for better readability by category
    # For line plots, have every point on the x-axis
    # For bar plots, remove the x-ticks
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
    num_cols = int(len(labels) / 2) if len(labels) > 7 else int(len(labels)/2)
    fig.legend(handles, labels, loc="lower center", ncol=num_cols, fontsize=16, bbox_to_anchor=[0.5, 0.04])

    # Move the subplots up
    plt.subplots_adjust(top=.95, bottom=.12)

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

    # Close fig in case more than 20 plt are open
    plt.close()


def plot_all_lines(stats_dict, dst, style_combinations, show=False):
    """Plot metrics graphs for all models under all categories in comparison

    The graph will have 4 subplots, which are for: accuracy, precision, recall, and bcr, from left to right,
        top to bottom

    Args:
        stats_dict:         A dictionary with each model as key and a dictionary of model specific metrics as value.
                                Each metric dictionary has the same keys: 'accuracies', 'precisions', 'recalls', 'bcrs',
                                and their corresponding list of values for each model as dictionary values.
        dst:                A string representing the relative path that the graph will be saved to.
                                Graph name and format should be included.
        style_combinations: A dictionary with each model as key and another dictionary as value, which has
                                color/linestyle as key and the corresponding color/marker/linestyle as value.
        show:               A boolean representing whether we want to show the graph or not. Default to False to
                                seamlessly run the whole model.
    """

    # Set up initial figure for plotting
    fig = plt.figure(figsize=(24, 20))

    # Setting up each sub-graph as axes
    # From left to right, top to bottom: Accuracy, Precision, Recall, BCR
    acc = plt.subplot(2, 2, 1)
    acc.set_ylabel('Accuracy', fontsize=20)
    acc.set_title(f'Averaged learning curve', fontsize=20)

    prec = plt.subplot(2, 2, 2)
    prec.set_ylabel('Precision', fontsize=20)
    prec.set_title(f'Averaged precision curve', fontsize=20)

    rec = plt.subplot(2, 2, 3)
    rec.set_ylabel('Recall', fontsize=20)
    rec.set_title(f'Averaged recall curve', fontsize=20)

    bcr = plt.subplot(2, 2, 4)
    bcr.set_ylabel('Balanced Classification Rate', fontsize=20)
    bcr.set_title(f'Averaged BCR curve', fontsize=20)

    # Exact all models available for plotting
    models = list(stats_dict.keys())

    # Plot each model's metrics
    for model in models:
        # Plotting avg metrics graph
        num_examples = [i for i in range(len(stats_dict[model]['accuracies']))]
        # Set up the color and line-style for model
        color = style_combinations[model]['color']
        linestyle = style_combinations[model]['linestyle']
        if len(num_examples) != 1:
            # Plot line graphs for models with active learning
            acc.plot(num_examples, stats_dict[model]['accuracies'], marker=linestyle, markersize=12, color=color,
                     linewidth=4, label=model)
            prec.plot(num_examples, stats_dict[model]['precisions'], marker=linestyle, markersize=12, color=color,
                      linewidth=4, label=model)
            rec.plot(num_examples, stats_dict[model]['recalls'], marker=linestyle, markersize=12, color=color,
                     linewidth=4, label=model)
            bcr.plot(num_examples, stats_dict[model]['bcrs'], marker=linestyle, markersize=12, color=color,
                     linewidth=4, label=model)
        else:
            # Plot horizontal lines for models without active learning
            acc.axhline(y=stats_dict[model]['accuracies'], linestyle=linestyle, linewidth=2, color=color, label=model)
            prec.axhline(y=stats_dict[model]['precisions'], linestyle=linestyle, linewidth=2, color=color, label=model)
            rec.axhline(y=stats_dict[model]['recalls'], linestyle=linestyle, linewidth=2, color=color, label=model)
            bcr.axhline(y=stats_dict[model]['bcrs'], linestyle=linestyle, linewidth=2, color=color, label=model)

    # Increase the font size of the x/y labels
    acc.tick_params(axis='both', labelsize=20)
    prec.tick_params(axis='both', labelsize=20)
    rec.tick_params(axis='both', labelsize=20)
    bcr.tick_params(axis='both', labelsize=20)

    # Adjust the x_ticks for better readability by category
    acc.set_xticks(num_examples)
    prec.set_xticks(num_examples)
    rec.set_xticks(num_examples)
    bcr.set_xticks(num_examples)

    # Display legends for all subplots
    handles, labels = acc.get_legend_handles_labels()
    num_cols = int(len(labels) / 5) if len(labels) > 7 else int(len(labels)/2)
    fig.legend(handles, labels, loc="lower center", ncol=num_cols, fontsize='x-large', bbox_to_anchor=[0.5, 0.03])

    # Move the subplots up
    plt.subplots_adjust(top=.95, bottom=.18)

    # Put x-axis label at the bottom
    fig.text(0.5, 0.02, "Number of samples given", ha="center", va="center", fontsize=20)

    # Set the metrics graph's name and designated folder
    graph_dst = Path(dst)

    # Remove duplicate graphs in case we can't directly overwrite the files
    if graph_dst.exists():
        graph_dst.unlink()

    # Save graph in folder
    plt.savefig(graph_dst)
    # TODO: logging instead
    print(f"Graph saved at {graph_dst}")

    if show:
        plt.show()

    plt.close()


def plot_bcr_vs_success_rate(models, cv_stats, dst, names, success_volume, success_percentage, category=None,
                             style_combinations=None, show=False):
    """TODO: DOCUMENTATION"""
    fig = plt.figure(figsize=(24, 12))
    vol = plt.subplot(1, 2, 1)
    per = plt.subplot(1, 2, 2)

    vol.set_title('BCR VS. Success Volume', fontsize=20)
    per.set_title('BCR VS. Success percentage', fontsize=20)

    vol.set_xlabel("Success volume", fontsize=20)
    vol.set_ylabel("BCR", fontsize=20)

    per.set_xlabel("Success percentage", fontsize=20)
    per.set_ylabel("BCR", fontsize=20)

    for model in models:
        if category:
            wanted_bcrs = find_bcr(model, cv_stats, names)
            vol.scatter(success_volume, wanted_bcrs, s=200, label=model, alpha=.6)
            per.scatter(success_percentage, wanted_bcrs, s=200, label=model, alpha=.6)
        else:
            color = style_combinations[model]['color']
            marker = style_combinations[model]['marker']
            wanted_bcrs = find_bcr(model, cv_stats, names)
            vol.scatter(success_volume, wanted_bcrs, s=200, label=model, alpha=.6, color=color, marker=marker)
            per.scatter(success_percentage, wanted_bcrs, s=200, label=model, alpha=.6, color=color, marker=marker)

    # Mark the 0.5 bcr line
    vol.axhline(y=.5, linewidth=2, color='tab:red')
    per.axhline(y=.5, linewidth=2, color='tab:red')

    # Increase the font size of the x/y labels
    vol.tick_params(axis='both', labelsize=20)
    per.tick_params(axis='both', labelsize=20)

    plt.subplots_adjust(top=.95, bottom=.25, wspace=0.2)

    handles, labels = vol.get_legend_handles_labels()
    num_cols = int(len(labels) / 5) if len(labels) > 10 else int(len(labels)/2)
    fig.legend(handles, labels, loc="lower center", ncol=num_cols, fontsize='x-large')

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

    plt.close()


def generate_style_combos(models):
    """TODO: DOCUMENTATION"""
    # Have a list of colors and line-styles/markers for different models and categories
    # I wish there's a better way to do this but it's matplotlib
    list_of_colors = ['violet', 'orangered', 'darkorange', 'seagreen', 'dodgerblue', 'darkviolet', 'r', 'violet']
    list_of_linestyles = ['dotted', 'solid', '-.', 'o', '*', 's', 'X']
    list_of_markers = ['d', 'p', '^', 'o', '*', 's', 'X']

    # Set same color for models and line-style for categories
    style_combinations = defaultdict(dict)
    for model in models:
        style_combinations[model] = defaultdict(dict)
        if 'KNN' in model:
            style_combinations[model]['color'] = list_of_colors[0]
        if 'SVM' in model:
            style_combinations[model]['color'] = list_of_colors[1]
        if 'Random' in model:
            style_combinations[model]['color'] = list_of_colors[2]
        if 'Logistic' in model:
            style_combinations[model]['color'] = list_of_colors[3]
        if 'Decision' in model:
            style_combinations[model]['color'] = list_of_colors[4]
        if 'Gradient' in model:
            style_combinations[model]['color'] = list_of_colors[5]
        if 'PLATIPUS' in model:
            style_combinations[model]['color'] = list_of_colors[6]
        if 'MAML' in model:
            style_combinations[model]['color'] = list_of_colors[7]
        # TODO: THE FOLLOWING MAY HAVE TO CHANGE ONCE MAML/PLATIPUS IS HERE
        if 'historical_only' in model:
            style_combinations[model]['linestyle'] = list_of_linestyles[0]
            style_combinations[model]['marker'] = list_of_markers[0]
        if 'historical_amine' in model:
            style_combinations[model]['linestyle'] = list_of_linestyles[1]
            style_combinations[model]['marker'] = list_of_markers[1]
        if 'amine_only' in model:
            style_combinations[model]['linestyle'] = list_of_linestyles[2]
            style_combinations[model]['marker'] = list_of_markers[2]
        if 'historical_amine_AL' in model:
            style_combinations[model]['linestyle'] = list_of_linestyles[3]
            style_combinations[model]['marker'] = list_of_markers[3]
        if 'amine_only_AL' in model:
            style_combinations[model]['linestyle'] = list_of_linestyles[4]
            style_combinations[model]['marker'] = list_of_markers[4]
        if '1' in model:
            style_combinations[model]['linestyle'] = list_of_linestyles[5]
            style_combinations[model]['marker'] = list_of_markers[5]
        if '2' in model:
            style_combinations[model]['linestyle'] = list_of_linestyles[6]
            style_combinations[model]['marker'] = list_of_markers[6]

    return style_combinations


def plot_all_graphs(cv_stats):
    """Aggregated plotting function to plot all the graphs needed

    Args:
        cv_stats:          A dictionary representing the performance statistics used of all desired models.
    """

    # Unload the models and amine to plot
    models_to_plot = list(cv_stats.keys())
    amines = cv_stats[models_to_plot[0]]['amine']

    # Parse the models into 3 main categories
    cat_3 = [model for model in models_to_plot if 'historical_only' in model]
    cat_4 = [model for model in models_to_plot if 'amine' in model and not ('AL' in model)]
    cat_5 = [model for model in models_to_plot if 'AL' in model or ('PLATIPUS' in model)]

    all_cats = {
        'category_3': cat_3,
        'category_4': cat_4,
        'category_5': cat_5,
    }

    # Parse the models into each model's names
    # TODO: Hard coded the model names
    knn = [model for model in models_to_plot if 'KNN' in model]
    rf = [model for model in models_to_plot if 'Random' in model]
    lr = [model for model in models_to_plot if 'Logistic' in model]
    dt = [model for model in models_to_plot if 'Decision' in model]
    svm = [model for model in models_to_plot if 'SVM' in model]
    gbc = [model for model in models_to_plot if 'Gradient' in model]
    platipus = [model for model in models_to_plot if 'PLATIPUS' in model]
    all_mod = {
        'KNN': knn,
        'Random_Forest': rf,
        'Logistic_Regression': lr,
        'Decision_Tree': dt,
        'SVM' : svm,
        'GBC' : gbc,
        'PLATIPUS': platipus
    }
    # Parse the models into specific categories
    hist_only = [model for model in models_to_plot if 'historical_only' in model]
    hist_amine = [model for model in models_to_plot if 'historical_amine' in model and not ('AL' in model)]
    amine_only = [model for model in models_to_plot if 'amine_only' in model and not ('AL' in model)]
    hist_amine_al = [model for model in models_to_plot if 'historical_amine_AL' in model or ('PLATIPUS' in model)]
    amine_only_al = [model for model in models_to_plot if 'amine_only_AL' in model]

    all_subcats = {
        'historical_only': hist_only,
        'historical_amine': hist_amine,
        'amine_only': amine_only,
        'historical_amine_AL': hist_amine_al,
        'amine_only_AL': amine_only_al,
    }
    # TODO: COMMENT
    avg_stats = find_avg_metrics(cv_stats, models_to_plot)
    rand_model = list(avg_stats.keys())[0]
    num_examples = len(avg_stats[rand_model]['accuracies'])
    style_combinations = generate_style_combos(models_to_plot)

    # Find the success rate of each amine, both volume wise and percentage wise
    names, success_volume, success_percentage = find_success_rate()

    # Find (and create) folder for bcr vs. success rate plots
    bcr_graph_folder = './results/success_rate'
    if not os.path.exists(bcr_graph_folder):
        os.makedirs(bcr_graph_folder)
        print(f'No folder for graphs of success rate vs bcr not found')
        print('Make folder to store results at')
    else:
        print('Found existing folder. Graphs will be stored at')
    print(bcr_graph_folder)

    # Plot graphs by category
    for cat in all_cats:
        # Identify category specific folder
        avg_graph_folder = './results/{}'.format(cat)

        # Check (and create) designated folder
        if not os.path.exists(avg_graph_folder):
            os.makedirs(avg_graph_folder)
            # TODO: Change to logging info
            print(f'No folder for graphs of {cat} models found')
            print('Make folder to store results at')
        else:
            # TODO: Change to logging info
            print('Found existing folder. Graphs will be stored at')
        print(avg_graph_folder)

        # Load all models to plot under the category
        models = all_cats[cat]

        # Plotting individual graphs for each task-specific model
        for i, amine in enumerate(amines):
            graph_dst = '{0:s}/cv_metrics_{1:s}.png'.format(avg_graph_folder, amine)
            plot_categorical_graph(cv_stats, graph_dst, style_combinations, amine=amine, amine_index=i, models=models)

        # Plotting avg graphs for all models in one category
        avg_graph_dst = '{0:s}/average_metrics_{1:s}.png'.format(avg_graph_folder, cat)
        plot_categorical_graph(avg_stats, avg_graph_dst, style_combinations, models=models)

        bcr_graph_dst = '{0:s}/bcr_against_success_{1:s}.png'.format(bcr_graph_folder, cat)
        plot_bcr_vs_success_rate(models, cv_stats, bcr_graph_dst, names, success_volume, success_percentage, category=cat)

    # save the graphs for each model
    mod_graph_folder = './results/by_model'
    if not os.path.exists(mod_graph_folder):
        os.makedirs(mod_graph_folder)
        print(f'No folder for graphs of models not found')
        print('Make folder to store results at')
    else:
        print('Found existing folder. Graphs will be stored at')
    print(mod_graph_folder)
    for mod in list(all_mod.keys()):
        dst_path = './results/by_model/{}.png'.format(mod)
        avg_stats_mod = find_avg_metrics(cv_stats, all_mod[mod])
        plot_all_lines(avg_stats_mod, dst_path, style_combinations)

    # save the graphs for each category
    subcat_graph_folder = './results/by_category'
    if not os.path.exists(subcat_graph_folder):
        os.makedirs(subcat_graph_folder)
        print(f'No folder for graphs by category not found')
        print('Make folder to store results at')
    else:
        print('Found existing folder. Graphs will be stored at')
    print(subcat_graph_folder)
    for subcat in list(all_subcats.keys()):
        dst_path = './results/by_category/{}.png'.format(subcat)
        avg_stats_cat = find_avg_metrics(cv_stats, all_subcats[subcat])
        plot_categorical_graph(avg_stats_cat, dst_path, style_combinations, models=all_subcats[subcat])

    # Plot BCR vs. success rate graphs for all models
    bcr_graph_dst = '{0:s}/bcr_against_all.png'.format(bcr_graph_folder)
    plot_bcr_vs_success_rate(models_to_plot, cv_stats, bcr_graph_dst, names, success_volume, success_percentage,
                             style_combinations=style_combinations)

    # Plot the winning models for each category
    winning_stats = find_winning_models(avg_stats, all_cats)
    plot_all_lines(winning_stats, './results/winning_models.png', style_combinations)

    # Plot graphs for all models
    plot_all_lines(avg_stats, './results/avg_metrics_all_models.png', style_combinations)
