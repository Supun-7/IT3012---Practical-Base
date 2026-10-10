import random
import tkinter as tk
from agent import SimpleReflexAgent, ModelBasedAgent, SearchAgent

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class VisualGridHuntGame:

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None, max_steps=400):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.facing = 1
        self.max_steps = max_steps

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        self.toxic_traps = set()
        num_traps = max(1, num_food // 3)
        while len(self.toxic_traps) < num_traps:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            trap_pos = (tx, ty)
            if (trap_pos != (0, 0)
                    and trap_pos not in self.walls
                    and trap_pos not in self.food_positions):
                self.toxic_traps.add(trap_pos)

        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        self.score = 0
        self.steps = 0
        self.collision = False

    def cell_ahead(self):
        dx, dy = DIRECTIONS[self.facing]
        return (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

    def get_percept(self) -> dict:
        ax, ay = self.cell_ahead()
        out_of_bounds = ax < 0 or ax >= self.width or ay < 0 or ay >= self.height
        return {
            'wall_ahead': out_of_bounds or (ax, ay) in self.walls,
            'food_here': tuple(self.agent_pos) in self.food_positions,
            'agent_pos': list(self.agent_pos),
            'grid_size': (self.width, self.height),
            'walls': list(self.walls),
            'all_food': list(self.food_positions)
        }

    def execute_action(self, action: str):
        self.steps += 1

        if action == 'Left':
            self.facing = (self.facing - 1) % 4
        elif action == 'Right':
            self.facing = (self.facing + 1) % 4
        elif action in ('Forward', 'MoveUp', 'MoveRight', 'MoveDown', 'MoveLeft'):
            if action != 'Forward':
                self.facing = {'MoveUp': 0, 'MoveRight': 1, 'MoveDown': 2, 'MoveLeft': 3}[action]
            if self.get_percept()['wall_ahead']:
                self.score -= 5
            else:
                self.agent_pos = list(self.cell_ahead())
                if tuple(self.agent_pos) in self.toxic_traps:
                    self.score -= 15
        elif action == 'Suck':
            tuple_pos = tuple(self.agent_pos)
            if tuple_pos in self.food_positions:
                self.food_positions.remove(tuple_pos)
                self.score += 20

        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= self.max_steps or self.collision


class GridGameGUI:

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None, seed=7):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")

        self.width = width
        self.height = height
        self.num_food = num_food
        self.num_opponents = num_opponents
        self.walls = walls
        self.seed = seed
        self.running = False

        self.new_environment()

        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        self.btn_reflex = tk.Button(button_frame, text="Run Simple Reflex Agent",
                                    command=lambda: self.run_loop(SimpleReflexAgent(), "Simple Reflex"),
                                    font=("Arial", 12), bg="#990000", fg="white")
        self.btn_reflex.pack(side="left", padx=5)

        self.btn_model = tk.Button(button_frame, text="Run Model-Based Agent",
                                   command=lambda: self.run_loop(ModelBasedAgent(), "Model-Based"),
                                   font=("Arial", 12), bg="#000066", fg="white")
        self.btn_model.pack(side="left", padx=5)

        search_frame = tk.Frame(root)
        search_frame.pack(pady=5)

        self.search_buttons = []
        for algo in ('BFS', 'DFS', 'UCS', 'AStar'):
            btn = tk.Button(search_frame, text="Run " + ("A*" if algo == 'AStar' else algo) + " Agent",
                            command=lambda a=algo: self.run_search(a),
                            font=("Arial", 12), bg="#047857", fg="white")
            btn.pack(side="left", padx=5)
            self.search_buttons.append(btn)

        self.draw_grid()

    def new_environment(self):
        random.seed(self.seed)
        self.env = VisualGridHuntGame(width=self.width, height=self.height, num_food=self.num_food,
                                      num_opponents=self.num_opponents, custom_walls=self.walls)

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        for tx, ty in self.env.toxic_traps:
            cx = tx * self.cell_size + self.cell_size / 2
            cy = (self.env.height - 1 - ty) * self.cell_size + self.cell_size / 2
            half = self.cell_size * 0.3
            self.canvas.create_polygon(
                cx, cy - half,
                cx + half, cy,
                cx, cy + half,
                cx - half, cy,
                fill="#7c3aed", outline="#5b21b6"
            )

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066",
                                outline="#1e3a8a")

        cx = ax * self.cell_size + self.cell_size / 2
        cy = (self.env.height - 1 - ay) * self.cell_size + self.cell_size / 2
        dx, dy = DIRECTIONS[self.env.facing]
        self.canvas.create_line(cx, cy, cx + dx * self.cell_size * 0.45, cy - dy * self.cell_size * 0.45,
                                fill="#facc15", width=3)

    def run_search(self, algo):
        agent = SearchAgent()
        agent.active_algo = algo
        self.run_loop(agent, algo)

    def set_buttons(self, state):
        self.btn_reflex.config(state=state)
        self.btn_model.config(state=state)
        for btn in self.search_buttons:
            btn.config(state=state)

    def run_loop(self, agent, agent_name):
        if self.running:
            return
        self.running = True
        self.set_buttons("disabled")

        self.new_environment()
        self.draw_grid()

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()
                action = agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"{agent_name} | Score: {self.env.score} | Steps: {self.env.steps} | "
                                       f"Food left: {len(self.env.food_positions)} | Action: {action}")
                self.root.after(60, step)
            else:
                if self.env.collision:
                    end_text = f"Collision! Game Over! Final Score: {self.env.score}"
                elif len(self.env.food_positions) == 0:
                    end_text = f"{agent_name} finished all food in {self.env.steps} steps! Final Score: {self.env.score}"
                else:
                    end_text = (f"{agent_name} ran out of steps with {len(self.env.food_positions)} food left. "
                                f"Final Score: {self.env.score}")
                self.label.config(text=end_text)
                self.running = False
                self.set_buttons("normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0)
    root.mainloop()