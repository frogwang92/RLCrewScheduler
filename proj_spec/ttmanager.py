import csv
import sys
if "../" not in sys.path:
    sys.path.append("../")
from rlcrew_base import job
from pytimeparse.timeparse import timeparse

all_runs = {}
all_jobs = {}

# ignore the depot start
run_ignore_point = ['横岗车辆段', '中心公园停车场']
run_split_point = ['塘坑', '华新']
run_ignore_runs = ['010', '043']
run_include_runs = ['050', '030', '021', '001', '031']


def add_dwellpoint(dp):
    if dp['Platform'] in run_ignore_point or dp['Run'] in run_ignore_runs:
        return
    # if dp['Run'] not in run_include_runs:
    #     return
    if all_runs.get(dp['Run']) is None:
        all_runs[dp['Run']] = []
        all_jobs[dp['Run']] = []
    all_runs[dp['Run']].append(dp)


def split(run):
    for dp in all_runs[run]:
        if dp['Platform'] in run_split_point:
            if all_jobs[run] and len(all_jobs[run][-1]) == 1:
                # print(all_jobs[run][-1])
                all_jobs[run][-1].append(dp)
                # print(all_jobs[run][-1])
                job.Job.add_job(run,
                                len(all_jobs[run]),
                                timeparse(all_jobs[run][-1][0]['Departure'], '%H:%M:%S'),
                                timeparse(all_jobs[run][-1][1]['Arrival'], '%H:%M:%S'),
                                run_split_point.index(all_jobs[run][-1][0]['Platform']),
                                run_split_point.index(all_jobs[run][-1][1]['Platform'])
                                )

            all_jobs[run].append([])
            all_jobs[run][-1].append(dp)


def load_tt():
    with open('l3.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            add_dwellpoint(row)
    for run in all_runs:
        split(run)
    job.Job.plot_gantt()
