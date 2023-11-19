import angr
import os
import stopit
import time

EMULATED = True
TIMEOUT = 10
USE_PCODE = False # Default engine is VEX

binaries = [f for f in os.listdir(os.getcwd()) if f.endswith('.exe')] # Needs to contain the filenames of all binaries in the dataset (later, the path of their directory is prepended to have the full path to each executable)
total_binaries = len(binaries)

def cfg_log_msg(msg, filename, i):
    print(msg + f" : {filename} : [{i+1}/{total_binaries}]")

results = []

for i, filename in enumerate(binaries):
    exe_path = os.path.join(os.getcwd(), filename)
    cfg_log_msg(f"Starting analysis", filename, i)
    
    try:
        if USE_PCODE:
            p = angr.Project(exe_path, load_options={'auto_load_libs': False}, engine=angr.engines.UberEnginePcode)
        else:
            p = angr.Project(exe_path, load_options={'auto_load_libs': False})
    except Exception as e:
        print(e)
        cfg_log_msg(f"Failed to load project", filename, i)
        results.append([filename, -1, 0, -1])
        continue
        
    cfg_manager = p.kb.cfgs
    cfg_model = cfg_manager.new_model(f"{filename}")

    timeout_reached = False
    complete_extraction_time = -1
    start_time = time.time()
    
    with stopit.ThreadingTimeout(TIMEOUT) as to_ctx_mgr:
        assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
        try:
            if EMULATED:
                p.analyses.CFGEmulated(fail_fast=False, resolve_indirect_jumps=True, normalize=True, model=cfg_model)
            else:
                p.analyses.CFGFast(fail_fast=False, resolve_indirect_jumps=True, normalize=True, model=cfg_model)
        except stopit.utils.TimeoutException:
            timeout_reached = True
            cfg_log_msg(f"Timeout reached when extracting CFG", filename, i)
        except Exception as e:
            print(e)
            cfg_log_msg(f"Failed to extract CFG", filename, i)
            found_node_at_entry = cfg_model.get_any_node(cfg_model.project.entry) is not None
            results.append([filename, complete_extraction_time, found_node_at_entry, -1])
            continue
        
        if not timeout_reached:
            end_time = time.time()
            complete_extraction_time = int(end_time - start_time)
            cfg_log_msg(f"Completed analysis in {complete_extraction_time} seconds", filename, i)
            
    found_node_at_entry = cfg_model.get_any_node(cfg_model.project.entry) is not None
    num_nodes_found = len(cfg_model.graph.nodes())
    
    results.append([filename, complete_extraction_time, found_node_at_entry, num_nodes_found])

longest_filename = max([len(x) for x in binaries])
print("{:<{}} | {:<15} | {:<16} | {:<13}".format('Filename', longest_filename, 'Extraction Time','Entry Node Found','# Nodes Found'))
for res in results:
    print("{:<{}} | {:<15} | {:<16} | {:<13}".format(res[0], longest_filename, res[1], res[2], res[3]))
