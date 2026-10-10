import random
from collections import deque
import heapq


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


class SearchAgent:

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'
        self.nodes_expanded = 0
        self.moves = {'MoveUp': (0, 1), 'MoveDown': (0, -1), 'MoveLeft': (-1, 0), 'MoveRight': (1, 0)}

    def neighbours(self, pos, grid_size, walls):
        width, height = grid_size
        result = []
        for action, (dx, dy) in self.moves.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                result.append((action, (nx, ny)))
        return result

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        frontier = deque([(start, [])])
        reached = {start}
        while frontier:
            pos, path = frontier.popleft()
            self.nodes_expanded += 1
            if pos == goal:
                return path
            for action, nxt in self.neighbours(pos, grid_size, walls):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        frontier = [(start, [])]
        reached = set()
        while frontier:
            pos, path = frontier.pop()
            if pos in reached:
                continue
            reached.add(pos)
            self.nodes_expanded += 1
            if pos == goal:
                return path
            for action, nxt in self.neighbours(pos, grid_size, walls):
                if nxt not in reached:
                    frontier.append((nxt, path + [action]))
        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        counter = 0
        frontier = [(0, counter, start, [])]
        reached = {start: 0}
        while frontier:
            cost, _, pos, path = heapq.heappop(frontier)
            if cost > reached.get(pos, float('inf')):
                continue
            self.nodes_expanded += 1
            if pos == goal:
                return path
            for action, nxt in self.neighbours(pos, grid_size, walls):
                new_cost = cost + 1
                if new_cost < reached.get(nxt, float('inf')):
                    reached[nxt] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, nxt, path + [action]))
        return None

    def sense_and_act(self, percept):
        if not percept['all_food']:
            return 'Stay'
        if not self.plan:
            start = tuple(percept['agent_pos'])
            foods = [tuple(f) for f in percept['all_food']]
            walls = percept['walls']
            grid_size = percept['grid_size']
            foods.sort(key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))

            for goal in foods:
                if self.active_algo == 'BFS':
                    path = self.bfs_search(start, goal, walls, grid_size)
                elif self.active_algo == 'DFS':
                    path = self.dfs_search(start, goal, walls, grid_size)
                elif self.active_algo == 'UCS':
                    path = self.ucs_search(start, goal, walls, grid_size)
                else:
                    path = None
                if path is not None:
                    self.plan = path + ['Suck']
                    break

            if not self.plan:
                return 'Stay'
        return self.plan.pop(0)