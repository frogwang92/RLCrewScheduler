import gym
from gym import spaces
from gym.utils import seeding
from rlcrew_base import crew
from rlcrew_base import job
from rlcrew_base import job_event
import numpy as np
from proj_spec import ttmanager
from rlcrew_base.crew import Crew

ttmanager.load_tt()


class JobCrewsEnv(gym.Env):
    """
    get all jobs from rlcrew_base.job.Job
    create crew object 2 * (len(all_jobs))
    action: pick a crew, assign to a job
    """
    def render(self, mode='human'):
        pass

    def __init__(self, natural=False):
        self.all_jobs = job.Job.all_jobs
        self.all_job_events = sorted(job.Job.all_job_events, key=lambda x: x.time, reverse=False)
        self.job_num = len(self.all_jobs)
        self.crew_num = self.job_num
        self.crew_pool_num = 2  # it's a hard code
        # current job event index, every step + 1
        self.current_job_event_index = 0
        self.all_crews = []
        self._init_crew()

        self.action_space = spaces.Discrete(self.crew_num)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.job_num, self.crew_num), dtype=np.bool)

        self.natural = natural
        self._seed()
        self._reset()

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _init_crew(self):
        self.all_crews.clear()
        # array of crew start working time, use to observe the offwork
        self.np_crew_start_working_time = np.zeros(len(self.all_jobs), np.int)
        # array of crew pools, indicates the position of crew
        self.np_crew_pool = np.zeros((self.crew_pool_num, len(self.all_jobs)), np.bool)
        # array of resting crew pool, value = resting time, use to observe the resting status
        self.np_crew_resting_time = np.zeros(len(self.all_jobs), np.int)
        # crew cid starts from 0
        new_crew = crew.Crew(0, -1)
        self.all_crews.append(new_crew)

    def _update_crew_pool(self, _crew):
        if _crew is None:
            return
        self.np_crew_pool[:, _crew.cid] = 0
        if _crew.status == 2:  # resting
            self.np_crew_pool[_crew.position, _crew.cid] = 1

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.current_job_event_index = 0
        self.np_jobs = np.full(len(self.all_jobs), -1)
        self._init_crew()

        for j in self.all_jobs:
            j.reset()

        return self.np_jobs

    def get_init_state(self):
        return np.zeros(len(self.all_jobs))

    def _step(self, action):
        # assert self.action_space.contains(action)
        current_job = self.all_job_events[self.current_job_event_index].job
        assign_reward, current_crew, previous_crew \
            = self.all_crews[action].assign(current_job)
        self._update_crew_pool(current_crew)
        self._update_crew_pool(previous_crew)
        self.np_jobs[self.all_jobs.index(current_job)] = action
        self.np_crew_start_working_time[current_crew.cid] = current_crew.start_time
        if previous_crew is not None and previous_crew.status == 2:
            self.np_crew_resting_time[previous_crew.cid] = previous_crew.resting_time
        self.current_job_event_index += 1
        done = self._evaluate()
        return self.np_jobs, assign_reward, done, {}

    def _evaluate(self):
        return not self.np_jobs.min() == -1

    @staticmethod
    def plot():
        job.Job.plot_gantt()

    def observe_current_pool(self):
        return self.np_jobs, \
               self.np_crew_pool, \
               self.np_crew_resting_time, \
               self.np_crew_start_working_time, \
               self.all_job_events[self.current_job_event_index].job, \
               self.get_next_unassigned_crew()

    def get_next_unassigned_crew(self):
        if self.all_crews[-1].status != 0:
            new_crew = crew.Crew(self.all_crews[-1].cid + 1, -1)
            self.all_crews.append(new_crew)
        return self.all_crews[-1]

    def observe_actions(self):
        actions = []
        actions.append(self.get_next_unassigned_crew().cid)
        # if self.all_job_events[self.current_job_event_index].job.previous is not None:
        #     if self.all_job_events[self.current_job_event_index].job.previous.crew.status != 4:
        #         actions.append(self.all_job_events[self.current_job_event_index].job.previous.crew.cid)

        for i in range(0, len(self.np_crew_resting_time) - 1):
            _current_job = self.all_job_events[self.current_job_event_index].job
            if self.np_crew_pool[_current_job.start_pos, i] == 1:
                if _current_job.start_time - self.np_crew_resting_time[i] > Crew.min_resting_period \
                        and _current_job.end_time - self.np_crew_start_working_time[i] < Crew.max_day_working_load:
                    actions.append(i)
                    # print(actions)
        # the current crew
        # this actually given the keep-current-crew the highest priority
        if self.all_job_events[self.current_job_event_index].job.previous is not None:
            if self.all_job_events[self.current_job_event_index].job.previous.crew.status != 4:
                actions.append(self.all_job_events[self.current_job_event_index].job.previous.crew.cid)

        return actions