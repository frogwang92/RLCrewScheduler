from proj_spec import ttmanager
from env import crew_and_jobs
from env import crew_and_job_rollout_runner
import mcts
import mcts_node

num_mcts_episodes = 10000


def run_mcts():
    ttmanager.load_tt()
    env = crew_and_jobs.JobCrewsEnv()

    print("Done loading environment!")
    root_node = mcts_node.Node()
    mcts_runner = mcts.MCTS(root_node)
    mcts_runner.run(num_mcts_episodes)

def run():
    run_mcts()


run()
