def init():
    global global_var_map 
    global_var_map = {}


def set_global_var(key:str, var):
    global_var_map[key] = var


def get_global_var(key:str):
    try:
        return global_var_map[key]
    except:
        raise KeyError