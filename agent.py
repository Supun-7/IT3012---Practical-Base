import random


class GreedyGridAgent:

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SimpleReflexAgent:

    def sense_and_act(self, percept):
        if percept['food_here']:
            return 'Suck'
        elif percept['wall_ahead']:
            return 'Left'
        else:
            return 'Forward'


class ModelBasedAgent:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.heading = 1
        self.visited_cells = {(0, 0): 1}
        self.known_walls = set()
        self.last_action = None
        self.last_view = None
        self.repeat_count = 0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def cell_in_direction(self, heading):
        dx, dy = self.directions[heading % 4]
        return (self.x + dx, self.y + dy)

    def update_state(self, percept):
        view = (percept['wall_ahead'], percept['food_here'])
        if view == self.last_view and self.last_action in ('Left', 'Right'):
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_view = view

        if self.last_action == 'Left':
            self.heading = (self.heading - 1) % 4
        elif self.last_action == 'Right':
            self.heading = (self.heading + 1) % 4
        elif self.last_action == 'Forward':
            self.x, self.y = self.cell_in_direction(self.heading)
            self.visited_cells[(self.x, self.y)] = self.visited_cells.get((self.x, self.y), 0) + 1

        if percept['wall_ahead']:
            self.known_walls.add(self.cell_in_direction(self.heading))

    def visit_cost(self, heading):
        cell = self.cell_in_direction(heading)
        if cell in self.known_walls:
            return float('inf')
        return self.visited_cells.get(cell, 0)

    def sense_and_act(self, percept):
        self.update_state(percept)

        if percept['food_here']:
            action = 'Suck'
        else:
            ahead = self.visit_cost(self.heading)
            left = self.visit_cost(self.heading - 1)
            right = self.visit_cost(self.heading + 1)

            if ahead != float('inf') and ahead <= min(left, right):
                action = 'Forward'
            elif self.repeat_count == 1:
                action = 'Left' if self.last_action == 'Right' else 'Right'
            elif left < right:
                action = 'Left'
            else:
                action = 'Right'

        self.last_action = action
        return action