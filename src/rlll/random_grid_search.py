# Created by Pablo Campillo at 19/1/23
import subprocess
import random
import click


@click.command()
@click.argument('num_exps', type=click.INT)
@click.argument('seed', type=click.INT)
def queue_random_grid_search_experiments(num_exps, seed):
    # Automated random search experiments
    random.seed(seed)

    for _ in range(num_exps):
        params = {
            "rand_lr": round(random.uniform(0.01, 0.0001), 4),
            "rand_epsilon": round(random.uniform(0.5, 1), 1),
            "rand_epsilon_decay": round(random.uniform(0.95, 0.999), 3),
            "rand_gamma": round(random.uniform(0.95, 0.990), 3),
            "rand_batch_size": random.choice([8, 16, 32, 64]),
            "rand_dnn_update": random.choice([1, 2, 4, 6, 8]),
            "rand_dnn_sync": random.choice([1000, 2500, 5000, 10000]),
            "rand_stack_size": random.choice([1, 2]),
            "rand_hidden_size": random.choice([8, 16, 32, 64]),
            "rand_mem_size" : random.choice([10_000, 50_000, 100_000]),
            "rand_burn_in" : random.choice([1_000, 5_000, 10_000]),
        }
        subprocess.run(["dvc", "exp", "run", "--queue",
                        "--set-param", f"train.lr={params['rand_lr']}",                        
                        "--set-param", f"train.epsilon={params['rand_epsilon']}",
                        "--set-param", f"train.epsilon_decay={params['rand_epsilon_decay']}",
                        "--set-param", f"train.gamma={params['rand_gamma']}",
                        "--set-param", f"train.batch_size={params['rand_batch_size']}",
                        "--set-param", f"train.dnn_update={params['rand_dnn_update']}",
                        "--set-param", f"train.dnn_sync={params['rand_dnn_sync']}",
                        "--set-param", f"train.stack_size={params['rand_stack_size']}",
                        "--set-param", f"train.hidden_size={params['rand_hidden_size']}",
                        "--set-param", f"train.mem_size={params['rand_mem_size']}",
                        "--set-param", f"train.burn_in={params['rand_burn_in']}",
                        ])


if __name__ == '__main__':
    queue_random_grid_search_experiments()
