
class Node:
    def __init__(self, state):
        self.state = state
        self.wins = 0.0
        self.visits = 0.0
        self.ressq = 0.0
        self.parent = None
        self.children = []
        self.sputc = 0.0
        self.weight = 0.0

    def set_weight(self, weight):
        self.weight = weight

    def append_child(self, child):
        self.children.append(child)
        child.parent = self

    def is_equal(self, node):
        if self.state == node.state:
            return True
        else:
            return False