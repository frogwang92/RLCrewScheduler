import gym
import copy
from gym import spaces
from gym.utils import seeding
from rlcrew_base import crew
from rlcrew_base import job
from rlcrew_base import job_event
import numpy as np
from proj_spec import ttmanager
from rlcrew_base.crew import Crew

# ttmanager.load_tt()


class JobCrewsEnv(gym.Env):
    """
    get all jobs from rlcrew_base.job.Job
    create crew object 2 * (len(all_jobs))
    action: pick a crew, assign to a job
    """
    def render(self, mode='human'):
        pass

    def __init__(self, natural=False):
        self.all_jobs = sorted(job.Job.all_jobs, key=lambda x: x.start_time, reverse=False)
        # self.all_job_events = sorted(job.Job.all_job_events, key=lambda x: x.time, reverse=False)
        self.job_num = len(self.all_jobs)
        self.crew_num = self.job_num       # let the crew number equals the job numbers
        # np array which stores the job event assigned crew
        # job0 , job1,  job2...
        # [0   ,    5,    9...]
        # crew0, crew5, crew9
        self.np_all_job_assigned_crew = np.full(self.job_num, -1)
        self.crew_pool_num = 2  # it's a hard code, 2 pools

        # current job event index, every step + 1
        self.current_job_index = 0
        self._init_crew()

        self._seed()
        self._reset()
        self.action_space = spaces.Discrete(self.crew_num)

    def reset(self):
        return self._reset()

    def initial_state(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _init_crew(self):
        # array of crew start working time, use to observe the offwork
        self.np_crew_start_working_time = np.zeros(self.crew_num, np.int)
        # array of resting crew pool, value = resting time, use to observe the resting status
        self.np_crew_resting_time = np.zeros(self.crew_num, np.int)
        # array of crew pools, indicates the position of crew
        self.np_crew_pool = np.zeros((self.crew_pool_num, self.crew_num), np.bool)
        # array of crew status
        self.np_crew_status = np.full(self.crew_num, -1)
        # array of crew's last job
        self.np_crew_last_job = np.full(self.crew_num, -1)
        # crew cid starts from 0
        self.new_crew_index = 0

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
        self.current_job_index = 0
        self.np_jobs = np.full(self.job_num, -1)
        self._init_crew()

        for j in self.all_jobs:
            j.reset()

        return self.np_jobs

    def _step(self, action):
        # assert self.action_space.contains(action)
        assign_reward = self._assign(self.current_job_index)
        self.current_job_index += 1
        done = self._evaluate()
        return self.np_jobs, assign_reward, done, {}

    def _evaluate(self):
        return not self.np_jobs.min() == -1

    def _assign(self, crew_id, job_index):
        current_job = self.all_jobs[job_index]
        reward = 0
        # fill in the assign crew array
        self.np_all_job_assigned_crew[job_index] = crew_id
        # update current crew status

        # update current crew start working time

        # update current crew last job

        # update current crew resting_time

        # get the last crew

        # update last crew status

        # update last crew start working time

        # update last crew last job

        # update last crew resting_time

        # update crew pool

        # return assign reward
        return reward

    @staticmethod
    def plot():
        job.Job.plot_gantt()

    def observe_current_pool(self):
        return self.np_jobs, \
               self.np_crew_pool, \
               self.np_crew_resting_time, \
               self.np_crew_start_working_time, \
               self.all_jobs[self.current_job_index].job, \
               self.new_crew_index

    def observe_actions(self):
        actions = []

        # the current crew
        # this actually given the keep-current-crew the highest priority
        if self.all_jobs[self.current_job_index].job.previous is not None:
            previous_crew = self.all_jobs[self.current_job_index].job.previous.crew
            if previous_crew.status != 4 and previous_crew.get_continuous_working_time() < Crew.max_continuous_working_period:
                actions.append(self.all_jobs[self.current_job_index].job.previous.crew.cid)

        # the resting pool
        for i in range(0, len(self.np_crew_resting_time) - 1):
            _current_job = self.all_jobs[self.current_job_index].job
            if self.np_crew_pool[_current_job.start_pos, i] == 1:
                if _current_job.start_time - self.np_crew_resting_time[i] > Crew.min_resting_period \
                        and _current_job.end_time - self.np_crew_start_working_time[i] < Crew.max_day_working_load:
                    actions.append(i)
                    # print(actions)

        # the new crew
        actions.append(self.new_crew_index)

        return actions
