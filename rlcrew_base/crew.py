import logging


class Crew:
    """Crew
    status of crew:
    0 - waiting (not yet start working)
    1 - assigned (working)
    2 - resting (take a break every 3 hrs)
    3 - eating
    4 - offwork
    """
    max_day_working_load = 25200  # 3600 * 7
    max_continuous_working_period = 10800  # 3600 * 3
    min_resting_period = 900  # 60 * 15

    def __init__(self, cid, position):
        self.cid = cid
        self.current_job = None
        self.resting_time = 0
        self.start_time = 0
        self.status = 0
        self.jobs = []
        self.position = position

    # def assign(self, job):
    #     if self.position is not None and job.start_pos != self.position:
    #         return -1
    #     if self.status == 0:  # start working
    #         self._do_assign(job)
    #         self.status = 1
    #         return 0
    #     if self.status == 1:  # is working
    #         self.status = 5
    #         return -1
    #     if self.status == 2:  # is resting
    #         restingtime = job.start_time - self.jobs[-1].end_time
    #         if restingtime > 900:
    #             self._do_assign(job)
    #             self.status = 1
    #         else:
    #             return -1
    #     return 0

    def assign(self, job):
        reward = 1
        if self.status == 0:  # start working, assign a new crew
            reward = 0
            self.start_time = job.start_time
        self.status = 1
        previous_crew = self._do_assign(job)
        if previous_crew is not self and previous_crew is not None:
            reward = 0.9
        reward = reward + self.evaluate()
        return reward, self, previous_crew

    def rest(self):
        if self.status == 1:  # working
            self._set_position(self.current_job.end_pos)
        self.current_job = None
        self.resting_time = self.jobs[-1].end_time
        self.status = 2

    def reset(self):
        self.current_job = None
        self.status = 0
        self.jobs = []
        self.resting_time = 0

    def _do_assign(self, job):
        self.current_job = job
        job.crew = self
        self.jobs.append(job)
        # if the previous job is assigned with another crew, let the crew rest himself
        previous_crew = None
        if job.previous is not None and job.previous.crew is not None and job.previous.crew is not self:
            job.previous.crew.rest()
            job.previous.crew.evaluate()
            previous_crew = job.previous.crew
        # if at the end of a run, rest
        if job.next is None:
            self.rest()
            previous_crew = self
        return previous_crew

    def _set_position(self, pos):
        self.position = pos

    def get_rest_time(self, observe_time):
        return observe_time - self.jobs[-1].end_time

    def evaluate(self):
        if self.status == 1:
            if self.jobs[-1].end_time - self.start_time > Crew.max_day_working_load:   # over the max working time
                self.status = 4
                return 0
        return 0

    def get_continuous_working_time(self):
        if self.status == 1:
            return self.jobs[-1].end_time - self.jobs[0].start_time
        else:
            return 0
