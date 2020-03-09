import plotly.figure_factory as ff
from rlcrew_base.job_event import JobEvent
import random

class Job:
    all_jobs = []
    all_jobs_hash = {}
    all_job_events = []

    @classmethod
    def add_job(cls, run: object, task: object, start_time: object, end_time: object, start_pos: object, end_pos: object) -> object:
        j = Job(run, task, start_time, end_time, start_pos, end_pos)
        Job.all_jobs.append(j)
        if run not in Job.all_jobs_hash:
            Job.all_jobs_hash[run] = []
            # this is the first
            j.previous = None
        else:
            j.previous = Job.all_jobs_hash[run][-1]
            j.previous.next = j
        Job.all_jobs_hash[run].append(j)

    @classmethod
    def plot_gantt(cls):
        # print("jobs count = " + str(len(Job.all_jobs)))
        df = []
        df.append(dict(Task='Launch Time', Start=str(43200), Finish=str(46800), Crew='0'))
        df.append(dict(Task='Dinner Time', Start=str(61200), Finish=str(64800), Crew='0'))
        _colors = {}
        set_of_crews = set()
        for j in Job.all_jobs:
            crewid = 0
            if j.crew is not None:
                crewid = j.crew.cid
                set_of_crews.add(crewid)
            randr = random.randint(0, 255)
            randg = random.randint(0, 255)
            randb = random.randint(0, 255)
            _colors[str(crewid)] = 'rgb(' + str(randr) + ', ' + str(randg) + ', ' + str(randb) + ')'
            df.append(dict(Task=j.run, Start=str(j.start_time), Finish=str(j.end_time), Crew=str(crewid)))

        print("crews count = " + str(len(set_of_crews)) + "; jobs count = " + str(len(Job.all_jobs)))

        fig = ff.create_gantt(df, height=1000, index_col='Crew', title='Crew Schedule', show_colorbar=True, group_tasks=True,
                              showgrid_x=True, showgrid_y=True, colors=_colors)
        fig.show()

    def __init__(self, run, task, start_time, end_time, start_pos, end_pos):
        self.run = run
        self.task = task
        self.start_time = start_time
        self.end_time = end_time
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.previous = None
        self.next = None
        self.crew = None
        # numeric ids
        self.jid = -1
        self.previous_jid = -1
        self.next_jid = -1
        ev_start = JobEvent(self, 0, start_time)
        ev_end = JobEvent(self, 1, end_time)
        Job.all_job_events.append(ev_start)
        # Job.all_job_events.append(ev_end)

    def reset(self):
        self.crew = None

    def assign_crew(self, crew):
        self.crew = crew

    def __str__(self):
        return str(self.run) + "-" + str(self.start_time) + "-" + str(self.end_time)

