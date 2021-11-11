from DLIP.utils.loading.load_class import load_class

obj_name = "Adam"
params_obj = dict()
objective = load_class(f"torch.optim", obj_name)#(**params_obj)

print(objective)