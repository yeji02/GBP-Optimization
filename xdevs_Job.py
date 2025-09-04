class Job:
    def __init__(self, jid, creation_time, size=1.0):
        self.id = int(jid)
        self.creation_time = float(creation_time)
        self.size = float(size)
        self.processing_time = None
    def __repr__(self):
        return f"Job(id={self.id}, size={self.size:.3f}, t0={self.creation_time:.3f})"