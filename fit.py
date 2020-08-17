import json

import pyhf


def get_parameter_names(model):
    labels = []
    for parname in model.config.par_order:
        for i_par in range(model.config.param_set(parname).n_parameters):
            labels.append(
                "{}[bin_{}]".format(parname, i_par)
                if model.config.param_set(parname).n_parameters > 1
                else parname
            )
    return labels


def print_results(bestfit, uncertainty, labels):
    max_label_length = max([len(label) for label in labels])
    for i, label in enumerate(labels):
        l_with_spacer = label + " " * (max_label_length - len(label))
        print(f"{l_with_spacer}: {bestfit[i]:.6f} +/- {uncertainty[i]:.6f}")


def fit(spec):
    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))
    result = pyhf.infer.mle.fit(data, model, return_uncertainties=True)

    bestfit = result[:, 0]
    uncertainty = result[:, 1]
    labels = get_parameter_names(model)

    print_results(bestfit, uncertainty, labels)
    return bestfit, uncertainty, labels


if __name__ == "__main__":
    with open("higgs4l_pyhf_workspace.json") as f:
        ws = json.load(f)

    fit(ws)
