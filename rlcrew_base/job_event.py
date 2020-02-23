
class JobEvent:
    """
    event type:
    0 - start
    1 - finish
    """
    def __init__(self, job, event_type, time):
        self.job = job
        self.event_type = event_type
        self.time = time

