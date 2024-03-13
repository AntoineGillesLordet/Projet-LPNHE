import numpy as np

try:
    from tqdm import tqdm
except:
    tqdm = lambda x:x


def detection_av_ap(dset, detected):
    n_det = []
    dset.data["detected"] = (dset.data["flux"]/dset.data["fluxerr"]) > 5
    for i in tqdm(detected):
        target = dset.targets.data.loc[i]
        obs_data = dset.data[dset.data["detected"]].loc[i]
        n_det.append((np.sum(obs_data["time"].between(target["t0"] - 50, target["t0"])), np.sum(obs_data["time"].between(target["t0"], target["t0"]+200))))
    return detected[~np.any(np.array(n_det) < 3, axis=1)]
        

def fit_lc(dset, id_det):
    fixed = {"z": dset.targets.data.loc[id_det]["z"]}

    guess = {
        "t0": dset.targets.data.loc[id_det]["t0"],
        "c": dset.targets.data.loc[id_det]["c"],
        "x0": dset.targets.data.loc[id_det]["x0"],
        "x1": dset.targets.data.loc[id_det]["x1"],
    }
    bounds = {
        "t0": dset.targets.data.loc[id_det]["t0"].apply(lambda x: [x-5, x+5]),
        "c": dset.targets.data.loc[id_det]["c"].apply(lambda x: [-0.3, 1.0]),
        "x0": dset.targets.data.loc[id_det]["x0"].apply(lambda x: [-0.1, 0.1]),
        "x1": dset.targets.data.loc[id_det]["x1"].apply(lambda x: [-4, 4]),
    }

    results, meta = dset.fit_lightcurves(
        source=sncosmo.Model("salt2"),
        index=id_det,
        use_dask=False,
        fixedparams=fixed,
        guessparams=guess,
        bounds=bounds,
    )

    better_results = pandas.DataFrame(
        {
            **{col: np.array(results["value"].loc[map(lambda x: (x, col), id_det[:400])] - \
                          results["truth"].loc[map(lambda x: (x, col), id_det[:400])])
            for col in guess.keys()},
            **{"err_"+col: np.array(results["error"].loc[map(lambda x: (x, col), id_det[:400])]) for col in guess.keys()}
        }
    )
    return better_results