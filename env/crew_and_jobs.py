import gym
import copy
from gym import spaces
from gym.utils import seeding
from rlcrew_base import job
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
        self.all_jobs = sorted(job.Job.all_jobs, key=lambda x: x.start_time, reverse=False)
        self.job_num = len(self.all_jobs)
        self.crew_num = self.job_num  # let the crew number equals the job numbers
        # build the jid connections
        for i in range(self.job_num):
            self.all_jobs[i].jid = i
            if self.all_jobs[i].previous is not None:
                self.all_jobs[i].previous_jid = self.all_jobs.index(self.all_jobs[i].previous)
            if self.all_jobs[i].next is not None:
                self.all_jobs[i].next_jid = self.all_jobs.index(self.all_jobs[i].next)

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
        #
        self.np_crew_continuous_working_start_time = np.zeros(self.crew_num, np.int)
        # array of resting crew pool, value = resting time, use to observe the resting status
        self.np_crew_start_resting_time = np.zeros(self.crew_num, np.int)
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
        self.np_all_job_assigned_crew = np.full(self.job_num, -1)
        self._init_crew()

        for j in self.all_jobs:
            j.reset()

        return self.np_all_job_assigned_crew

    def _step(self, action):
        # assert self.action_space.contains(action)
        assign_reward = self._assign(action, self.current_job_index)
        self.current_job_index += 1
        done = self._evaluate()
        return self.np_all_job_assigned_crew, assign_reward, done, {}

    def _evaluate(self):
        return not self.np_all_job_assigned_crew.min() == -1

    def _assign(self, crew_id, job_index):
        if crew_id == self.new_crew_index:
            self.new_crew_index = self.new_crew_index + 1

        current_job = self.all_jobs[job_index]
        reward = 0
        # get the previous crew
        previous_job = current_job.previous_jid
        previous_crew = -1
        if previous_job != -1:
            previous_crew = self.np_all_job_assigned_crew[previous_job]
        # return assign reward
        if previous_crew == crew_id:
            reward = 1
        else:
            if self.np_crew_status[crew_id] == 2:  # assign a resting crew
                reward = 1

        # fill in the assign crew array
        self.np_all_job_assigned_crew[job_index] = crew_id
        # update continuous working time
        if self.np_crew_status[crew_id] == 2:  # resting -> working
            self.np_crew_continuous_working_start_time[crew_id] = current_job.start_time
        # update current crew status
        self.np_crew_status[crew_id] = 1  # working
        self.np_crew_pool[:, crew_id] = 0
        # update current crew start working time
        if self.np_crew_start_working_time[crew_id] == 0:
            self.np_crew_start_working_time[crew_id] = current_job.start_time
            self.np_crew_continuous_working_start_time[crew_id] = current_job.start_time
        # update current crew last job
        self.np_crew_last_job[crew_id] = job_index
        # update current crew resting_time
        self.np_crew_start_resting_time[crew_id] = 0

        # update previous crew status
        # update previous crew resting_time
        if previous_crew != -1 and previous_crew != crew_id:
            self.np_crew_status[previous_crew] = 2  # resting
            self.np_crew_start_resting_time[previous_crew] = self.all_jobs[previous_job].end_time
            self.np_crew_pool[self.all_jobs[previous_job].end_pos, previous_crew] = self.all_jobs[previous_job].end_time

        # assess if this is the last job of a run
        if current_job.next_jid == -1:
            self.np_crew_status[crew_id] = 2  # resting
            self.np_crew_start_resting_time[crew_id] = current_job.end_time
            self.np_crew_pool[current_job.end_pos, crew_id] = 1

        return reward

    def plot(self):
        # job.Job.plot_gantt()
        print(self.new_crew_index)

    def observe_actions(self):
        _current_job = self.all_jobs[self.current_job_index]
        avaiable_actions = \
            ((self.np_crew_start_resting_time - _current_job.start_time) < (-1) * Crew.min_resting_period) & \
            ((self.np_crew_start_working_time - _current_job.start_time) > (-1) * Crew.max_day_working_load) & \
            self.np_crew_pool[_current_job.start_pos]

        # the current crew
        # this actually given the keep-current-crew the highest priority
        previous_job_id = _current_job.previous_jid
        if previous_job_id != -1:
            previous_crew_id = self.np_all_job_assigned_crew[previous_job_id]
            if self.np_crew_status[previous_crew_id] == 1:
                # check the end time
                endtime = _current_job.end_time
                if endtime - self.np_crew_continuous_working_start_time[previous_crew_id] < Crew.max_continuous_working_period:
                    if endtime - self.np_crew_start_working_time[previous_crew_id] < Crew.max_day_working_load:
                        avaiable_actions[previous_crew_id] = True

        # the new crew
        avaiable_actions[self.new_crew_index] = True
        return avaiable_actions
