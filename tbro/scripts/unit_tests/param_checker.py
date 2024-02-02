from scripts.utils.params import Parameters

if __name__ == '__main__':
	args = Parameters()
	param_dict = {
		'sequence_length': args.max_length,
		'batch_size': args.batch_size,
		'epochs': args.epochs,
		'radar_shape': args.radar_shape,
		'hidden_size': args.hidden_size,
		'learning_rate': args.learning_rate
	}
	print(param_dict)
