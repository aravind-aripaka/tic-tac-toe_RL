from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random
import pickle
import os
from collections import defaultdict

app = Flask(__name__)
CORS(app)

class TicTacToeRLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, q_table_file="q_table.pkl"):
        self.q_table = defaultdict(lambda: np.zeros(9)) 
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon 
        self.q_table_file = q_table_file
        self.load_q_table() 

    def get_q_values(self, state):
        return self.q_table[state]

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)  
        
        q_values = self.get_q_values(state)
        max_q = np.max(q_values[available_actions]) 
        best_actions = [a for a in available_actions if q_values[a] == max_q]
        return random.choice(best_actions)  

    def update_q_value(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        if done:
            q_values[action] += self.alpha * (reward - q_values[action])
        else:
            next_q_values = self.get_q_values(next_state)
            q_values[action] += self.alpha * (reward + self.gamma * np.max(next_q_values) - q_values[action])

    def save_q_table(self):
        with open(self.q_table_file, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, "rb") as f:
                self.q_table.update(pickle.load(f))
        else:
            print("No Q-table found. Starting fresh.")

agent = TicTacToeRLAgent()

@app.route('/get-move', methods=['POST'])
def get_move():
    data = request.get_json()
    if not data or 'board' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    board = data['board']
    state = tuple(board)

    available_actions = [i for i in range(9) if board[i] == ""]
    if not available_actions:
        return jsonify({"move": None})

    move = agent.choose_action(state, available_actions)
    return jsonify({"move": move})

@app.route('/update-q', methods=['POST'])
def update_q():
    data = request.get_json()
    required_keys = {'state', 'action', 'reward', 'next_state', 'done'}
    
    if not data or not required_keys.issubset(data):
        return jsonify({"error": "Invalid input"}), 400

    state = tuple(data['state'])
    action = data['action']
    reward = data['reward']
    next_state = tuple(data['next_state'])
    done = data['done']

    agent.update_q_value(state, action, reward, next_state, done)
    print(data)
    return jsonify({"status": "Q-table updated"})


@app.route('/save-q-table', methods=['POST'])
def save_q_table():
    agent.save_q_table()
    return jsonify({"status": "Q-table saved"})

@app.route('/send_table', methods=['GET'])
def send_q_table():
    q_table_data = [
        {
            "state": list(k), 
            "q_values": v.tolist()  
        }
        for k, v in agent.q_table.items()
    ]
    return jsonify(q_table_data)  

if __name__ == '__main__':
    app.run(debug=True)