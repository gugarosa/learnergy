from learnergy.visual import convergence


def test_convergence_plot():
    new_model = {"mse": [1, 2, 3], "pl": [1.5, 2, 2.5], "time": [0.1, 0.2, 0.3]}

    try:
        convergence.plot(new_model["mse"], new_model["pl"], new_model["time"], labels=1)
    except:
        convergence.plot(
            new_model["mse"],
            new_model["pl"],
            new_model["time"],
            labels=["MSE", "log-PL", "time (s)"],
        )

    try:
        convergence.plot(
            new_model["mse"], new_model["pl"], new_model["time"], labels=["MSE"]
        )
    except:
        convergence.plot(new_model["mse"], new_model["pl"], new_model["time"])
