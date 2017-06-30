import load_input

data = load_input.InputData()
data.get_train(1)

data.next_batch(1)
data.print_img_at_idx(0)
