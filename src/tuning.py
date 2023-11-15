import neptuna

def objective(trial):
    pass


def optimize():
    study = neptuna.create_study(
        study_name="study",
        storage="sqlite:///example.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)