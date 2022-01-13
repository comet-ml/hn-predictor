'''
For something like 
'''


class BaseMetrics:

    def __init__(self, metrics: List[callable]):
        pass

    def compute_metrics(model, X, y, metrics: dict={'accuracy': 'accuracy_score'}):
        '''Generic'''
        pass