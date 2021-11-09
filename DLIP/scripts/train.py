from DLIP.utils.loading.load_class import load_class

obj_name = "DiceLoss"
params_obj = dict()
objective = load_class(f"objectives", obj_name)(**params_obj)