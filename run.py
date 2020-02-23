from proj_spec import ttmanager
from env import crew_and_jobs
from mtcs import execute_episode
from env import crew_and_job_rollout_runner

num_mtcs_episodes = 10000

def run():
    ttmanager.load_tt()
    env = crew_and_jobs.JobCrewsEnv()
    runner = crew_and_job_rollout_runner.RolloutRunner()
    print("Done loading environment!")

    for i in range(num_mtcs_episodes):
        execute_episode(runner, 32, env)


run()
