from utils.instance_generator import FFSMPInstanceGenerator


generator = FFSMPInstanceGenerator("configs/instance_generator_config.json")
generator.generate_instances()
