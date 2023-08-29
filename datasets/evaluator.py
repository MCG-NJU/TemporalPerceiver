import util.misc as utils

class general_evaluator(object):
    def __init__(self):
        self.predictions = []
    
    def update(self, vid, predictions):
        self.predictions+=[(vid, predictions)]
    
    def get_result(self):
        return self.predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions
    
    def summarize(self):
        results = {}
        for vid, p in self.predictions:
            try:
                results[vid].append(p)
            except KeyError:
                results[vid] = []
                results[vid].append(p)
        return results