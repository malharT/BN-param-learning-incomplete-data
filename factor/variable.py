from itertools import count
from functools import total_ordering, reduce

# does make it a bit slower, I'm told
@total_ordering
class variable:
    id_counter = count(0)

    def __init__(self, name):
        self.name = name
        self.id = next(self.id_counter)

    def __lt__(self, other):
        return self.id < other.id
    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __str__(self):
        return self.name

class discretevariable(variable):
    def __init__(self, name, nvals):
        super().__init__(name)
        self.nvals = nvals

    def __str__(self):
        return self.name
