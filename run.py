from proj_spec import ttmanager
from env import crew_and_jobs
from env import crew_and_job_rollout_runner

num_mcts_episodes = 10000


def run_mcts():
    ttmanager.load_tt()
    env = crew_and_jobs.JobCrewsEnv()

    print("Done loading environment!")


