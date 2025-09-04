"""
Module for optimizing the Far From Most Strings Problem (FFMSP) using a hybrid Ant Colony Optimization (ACO) approach.
"""
import numpy as np
import random
import sys
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import json

class FFMSP_Optimizer:
    alphabet = []
    sequences = []
    n = 0
    alphabet_size = 0
    sequence_length = 0
    drate = 0.8
    na = 10
    greedy_values = None
    t = 0
    tau_max = 0.999
    tau_min = 0.001
    ro = 0.1

    def __init__(self, file_path: str, t_scale = 0, config_path="configs/optimizer_strategies.json"):
        self.t_scale = t_scale
        self.file_path = file_path

        self.is_valid_file(file_path)
        # Load configuration from JSON
        self.load_config(config_path)

        self.read_input()
        self.validate_input()
        self.build_position_counts()
        self.initialize_greedy_values()

    def is_valid_file(self, file_path: str) -> bool:
        """
        Check if the input file is valid and can be opened.
        """
        try:
            with open(file_path, 'r') as file:
                # Just attempt to read the first line to ensure it's accessible
                file.readline()
            return True
        except FileNotFoundError:
            print(f"The file {file_path} was not found. Check if the datasets are downloaded/generated correctly.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while trying to read the file {file_path}.")
            sys.exit(1)

    def load_config(self, config_path: str):
        """
        Load configuration from a JSON file.
        """
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                self.path_relinking_strategy = config.get("path_relinking_strategy", "greedy")
                self.local_search_strategy = config.get("local_search_strategy", "best_improvement")
                self.apply_LS_after_PR = config.get("apply_LS_after_PR", False)

                print(json.dumps(config, indent=4))
        except FileNotFoundError:
            print(f"Configuration file {config_path} not found. Using default settings.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in {config_path}. Using default settings.")

    def read_input(self):
        with open(self.file_path, 'r') as file:
            self.sequences = [list(line.strip()) for line in file.readlines() if not line.startswith("#")]

        self.n = len(self.sequences)
        self.sequence_length = len(self.sequences[0])
        self.t = self.t_scale*self.sequence_length

        all_chars = set()
        for seq in self.sequences:
            all_chars.update(seq)

        self.alphabet = sorted(list(all_chars))
        self.alphabet_size = len(self.alphabet)

    def build_position_counts(self):
        self.position_counts = [defaultdict(int) for _ in range(self.sequence_length)]
        # Initialize all position counts to 0
        for i in range(self.sequence_length):
            for j in range(self.alphabet_size):
                self.position_counts[i][self.alphabet[j]] = 0
        
        # Count character occurrences at each position across all sequences
        for seq in self.sequences:
            for i, char in enumerate(seq):
                self.position_counts[i][char] += 1
                
        return self.position_counts

    def initialize_greedy_values(self):
        self.greedy_values = np.empty((self.sequence_length, self.alphabet_size), dtype=np.float64)
        for i in range(self.sequence_length):
            for j in range(self.alphabet_size):
                self.greedy_values[i][j] = (self.n - self.position_counts[i][self.alphabet[j]])/self.n

    def validate_input(self):
        sequence_length = len(self.sequences[0])
        for seq in self.sequences:
            # Ensure the sequences are of equal length
            if len(seq) != sequence_length:
                raise ValueError("All sequences must have the same length.")

    def occurances_of_letter_at_pos(self, j, i):
        number_of_occurances = 0
        for seq in self.sequences:
            if seq[i] == self.alphabet[j]:
                number_of_occurances += 1
        return number_of_occurances

    def Dh(self,s1,s2):
        distance = 0
        for i in range(self.sequence_length):
            if s1[i] != s2[i]:
                distance += 1

        return distance

    def initialize_pheromone_matrix(self) -> np.ndarray:
        """
        Initializes a pheromone matrix T setting all values to 0.5.
        Every row corresponds to one position in the sequence.
        Every column corresponds to one character from the alphabet.

        Returns:
        - np.ndarray: The initialized pheromone matrix.
        """
        return np.full((self.sequence_length, self.alphabet_size), 0.5, dtype=np.float64)


    def construct_solution(self, T: np.ndarray) -> list:
        """
        Constructs a string using a pheromone matrix with a probabilistic approach.

        Parameters:
        - T (np.ndarray): Pheromone matrix (sequence_length x alphabet_size).
        - alphabet (list): List of possible characters.

        Returns:
        - list: The constructed solution sequence.
        """
        solution = []

        for i in range(self.sequence_length):
            probabilities = []
            denom = sum([T[i][c] * self.greedy_values[i][c] for c in range(self.alphabet_size)])
            for j in range(self.alphabet_size):
                probabilities.append((T[i][j] * self.greedy_values[i][j]) / denom)

            z = np.random.uniform(0.5, 1.0)
            if z <= self.drate:
                max_value_idx = probabilities.index(max(probabilities))
                chosen_char = self.alphabet[max_value_idx]
            else:
                chosen_char = random.choices(self.alphabet, weights=probabilities, k=1)[0]

            solution.append(chosen_char)

        return solution

    objective_values_cache = {}
    def objective_function(self, s):
        """ Return the value of the objective function, which is 
        equal to the number of sequences with the distance (to s) at least t."""
        count = 0
        if tuple(s) in self.objective_values_cache:
            return self.objective_values_cache[tuple(s)]

        for seq in self.sequences:
            if self.Dh(seq, s) >= self.t:
                count += 1
        self.objective_values_cache[tuple(s)] = count

        return self.objective_values_cache[tuple(s)]

    qs_distance_cache = {}
    def Qs_distance(self, s):
        """ Calculates the sum of distances to s from all sequences whose distance to s
        is at least t, plus the largest distance to s from all distances lower than t.
        """
        if tuple(s) in self.qs_distance_cache:
            return self.qs_distance_cache[tuple(s)]

        distance = 0
        max_lower_than_t = 0
        for seq in self.sequences:
            current_dh = self.Dh(seq, s)
            if current_dh >= self.t:
                distance += current_dh
            else:
                if current_dh > max_lower_than_t:
                    max_lower_than_t = current_dh

        self.qs_distance_cache[tuple(s)] = distance + max_lower_than_t
        return self.qs_distance_cache[tuple(s)]

    def better(self, s1, s2):
        if s2 == []:
            return True

        f1 = self.objective_function(s1)
        f2 = self.objective_function(s2)

        if f1 > f2:
            return True

        if f1 < f2:
            return False

        if self.Qs_distance(s1) > self.Qs_distance(s2):
            return True

        return False
    
    def would_improve(self, s, i, c):
        """ Returns true if changing the character at position i in s to c would improve the solution."""
        if self.position_counts[i][c] < self.position_counts[i][s[i]]:
            return True

        return False

    def local_search(self, s):
        """For now we take into account all characters from alphabet when trying to improve current solution.
        RCL can be taken..."""
        has_improved = True

        while has_improved:
            has_improved = False
            for i in range(self.sequence_length):
                for c in self.alphabet:
                    # when self.better was used every time, it was really slower (5 seconds for a seq of length = 300)
                    if self.would_improve(s, i, c):
                        s[i] = c
                        has_improved = True
                        if self.local_search_strategy == "first_improvement":
                            return s

        return s

    def symmetric_difference(self, s1, s2):
        """ Returns a set of positions which 
        contain different elements in s1 and s2
        """
        diff = []
        for i in range(self.sequence_length):
            if s1[i] != s2[i]:
                diff.append(i)

        return diff

    def find_best_move(self, current, guiding_solution, moves):
        """
        Selects the best move from the list of possible moves based on the move selection strategy.
        """
        if self.path_relinking_strategy == "random":
            return random.choice(moves)

        best_move = moves[0]
        best_solution = current.copy()

        for move in moves:
            tmp = current[move]
            current[move] = guiding_solution[move]
            if self.better(current, best_solution):
                best_solution = current.copy()
                best_move = move
            current[move] = tmp

        return best_move

    def path_relinking(self, initial_solution, guiding_solution):
        """
        Implement path relinking from initial_solution
        to guiding_solution.
        """
        if guiding_solution == []:
            return initial_solution
        if initial_solution == []:
            return guiding_solution
        best = initial_solution
        current = initial_solution.copy()

        moves_left = self.symmetric_difference(initial_solution, guiding_solution)

        while moves_left != []:
            best_move = self.find_best_move(current, guiding_solution, moves_left)
            moves_left.remove(best_move)
            current[best_move] = guiding_solution[best_move]
            if self.better(current, best):
                best = current.copy()

        if self.apply_LS_after_PR:
            best = self.local_search(best)

        return best

    def get_ks(self, cf):
        """
        Returns the values of kib and krb based on the convergence factor cf.
        """
        kib = None
        krb = None

        if cf < 0.4:
            kib = 1
            krb = 0
        elif cf < 0.6:
            kib = 2/3
            krb = 1/3
        elif cf < 0.8:
            kib = 1/3
            krb = 2/3
        else:
            kib = 0
            krb = 1

        return kib, krb

    def calculate_eta_matrix(self, cf, sib, srb):
        eta = np.zeros((self.sequence_length, self.alphabet_size), dtype=np.float64)
        
        for i in range(self.sequence_length):
            for j in range(self.alphabet_size):
                kib, krb = self.get_ks(cf)

                if self.alphabet[j] == sib[i]:
                    eta[i][j] += kib
                if self.alphabet[j] == srb[i]:
                    eta[i][j] += krb

        return eta
    def apply_pheromone_update(self, cf, T, sib, srb):
        eta = self.calculate_eta_matrix(cf, sib, srb)

        for i in range(self.sequence_length):
            for j in range(self.alphabet_size):
                T[i][j] += self.ro * (eta[i][j] - T[i][j])

                if T[i][j] < self.tau_min:
                    T[i][j] = self.tau_min
                elif T[i][j] > self.tau_max:
                    T[i][j] = self.tau_max

        return

    def compute_convergence_factor(self, T):
        tau_sum = 0.0
        for i in range(self.sequence_length):
            for j in range(self.alphabet_size):
                tau_sum += max(abs(self.tau_max - T[i][j]), abs(T[i][j] - self.tau_min))

        return 2*(tau_sum / (self.sequence_length * self.alphabet_size * (self.tau_max - self.tau_min)) - 0.5)

    def construction_task(self, T):
        s = self.construct_solution(T)
        s = self.local_search(s)
        return s

    def RunACO(self, sbs):
        srb = [] # restart best solution (best solution found so far during the current application of RunACO())
        cf = 0
        T = self.initialize_pheromone_matrix()
        start =time.time()
        while cf < 0.99:
            #print(cf)
            sib = [] # iteration best solution
            for i in range(self.na):
                s = self.construct_solution(T)
                s = self.local_search(s)
                if self.better(s, sib):
                    sib = s

            sib = self.path_relinking(sib, sbs)
            
            if self.better(sib, srb):
                srb = sib.copy()
            if self.better(sib, sbs):
                sbs = sib.copy()
            self.apply_pheromone_update(cf, T, sib, srb)
            cf = self.compute_convergence_factor(T)
        end = time.time()
        print(end - start, "_", self.objective_function(sbs))

        return sbs

    def hybrid_ACO(self, tlim_aco, runlim_aco):
        sbs = []
        runs_aco = 0
        start = time.time()
        current = start

        while current - start < tlim_aco and runs_aco < runlim_aco:
            s = self.RunACO(sbs)
            if self.better(s, sbs):
                sbs = s
                runs_aco = 0
                
            else:
                runs_aco += 1
            
            current = time.time()

        #self.append_result(s)
        return sbs

    def solution_polishing(self, sbs):
        model = gp.Model("sentence")

        x = {}
        y = {}
        for i in range(1, self.sequence_length +1):
            for j in range(1, self.alphabet_size + 1):
                x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{self.alphabet[j]}")

        for i in range(1, self.n+1):
            y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

        for i in range(1, self.sequence_length+1):
            model.addConstr(gp.quicksum(x[i,j] for j in range(1, self.alphabet_size+1)) == 1, name=f"one_letter_at_pos_{i}")

        model.setObjective(gp.quicksum(y[i] for i in range(1, self.n+1)), GRB.MAXIMIZE)

        model.update()

    def append_result(self, s):
        with open("results.csv", 'a') as f:  # 'a' means append mode
            f.write(f"{self.file_path},{self.t_scale}, {self.ro}, {self.objective_function(s)}\n")
