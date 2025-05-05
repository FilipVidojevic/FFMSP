import json
import os
import random

class FFSMPInstanceGenerator:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.alphabet = self.config['alphabet']
        self.number_of_instances = self.config['number_of_instances']
        self.sequence_counts = self.config['number_of_sequences']
        self.sequence_lengths = self.config['sequence_length']
        self.dir_name = self.config['dir_name']

        os.makedirs(self.dir_name, exist_ok=True)

    def _load_config(self, path):
        with open(path, 'r') as file:
            return json.load(file)

    def _generate_sequence(self, length):
        return ''.join(random.choices(self.alphabet, k=length))

    def generate_instances(self):
        for n_seq in self.sequence_counts:
            for m_len in self.sequence_lengths:
                for instance_id in range(self.number_of_instances):
                    filename = f"n{n_seq}_m{m_len}_{instance_id}.txt"
                    filepath = os.path.join(self.dir_name, filename)

                    with open(filepath, 'w') as f:
                        for _ in range(n_seq):
                            seq = self._generate_sequence(m_len)
                            f.write(seq + '\n')

                    print(f"Generated: {filepath}")

# Example usage:
if __name__ == "__main__":
    generator = FFSMPInstanceGenerator("instance_generator_config.json")
    generator.generate_instances()
