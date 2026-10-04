class temporal:
    """
    args:
    score_threshold: is the set limit. What it do is after the image pass
    the the score threshold, it means there is a potential of fire
    time_limit: Is a set of duration 
    duration_s: is the duration between observations and restting time
    """
    def __init__(self, score_threshold, time_limit, duration_s):
        self.score_threshold = score_threshold
        self.time_limit = time_limit
        self.duration_s = duration_s
        self.reset()

    def reset(self):
        self.last_timestamp = None
        self.positive_time = None

    def update(self, visual_score, timestamp_s, valid=True):
        if(self.last_timestamp is not None and timestamp_s < self.last_timestamp):
            self.reset()
        elif self.last_timestamp is not None:
            gap = timestamp_s - self.last_timestamp
            if gap > self.duration_s:
                self.positive_time = None
        duration = 0.0 
        #If is fail to detect, pass the observation
        if not valid or visual_score is None:
            self.positive_time = None
            decision = "Unclear"
        #if the predition score is slower than what expectation score, no alert
        elif visual_score < self.score_threshold:
            self.positive_time = None
            decision = "No_Alert"
        
        else:
            #Reset the positive steak and check again
            if self.positive_time is None:
                self.positive_time = timestamp_s
            duration = timestamp_s - self.positive_time

            if duration >= self.duration_s:
                 decision = "Alert"
            else:
                decision = "No_Alert"
        self.last_timestamp = timestamp_s

        return{"decision": decision, "duration_s": duration}


#unit_test
if __name__ == "__main__":
    tracker = temporal(
        score_threshold=0.5,
        time_limit=1.0,
        duration_s=1.0
    )

    samples = [
        (0.0, 0.9),
        (0.5, 0.9),
        (1.0, 0.9),
        (1.5, 0.1)
    ]

    for timestamp, score in samples:
        result = tracker.update(score, timestamp)
        print(timestamp, result)