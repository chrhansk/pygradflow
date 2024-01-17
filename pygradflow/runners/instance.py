class Instance:
    def __init__(self,
                 name,
                 num_vars,
                 num_cons):
        self.name = name
        self.num_vars = num_vars
        self.num_cons = num_cons

    @property
    def size(self):
        return self.num_vars + self.num_cons
