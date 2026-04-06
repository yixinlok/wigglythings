import os
import pandas as pd
import globals 

def save_results_to_csv(results, total_elapsed, precompute_time, bi, ix, csv_path="out/results.csv"):
    Uq_time = 0
    q_time = 0
    v_time = 0
    total_gpu_time = 0

    for r in results:
        if "wp_get_modal_displacement" in r.name:
            Uq_time += r.elapsed
        elif "wp_get_rot_transpose" in r.name or "wp_get_face_points" in r.name or "wp_compute_new_spikes" in r.name or "wp_update_v" in r.name:
            v_time += r.elapsed
        elif "wp_dyrt__locals" in r.name or "wp_update_q" in r.name:
            q_time += r.elapsed
        else:
            print(f"Unknown kernel {r.name}")
        total_gpu_time += r.elapsed
    print("---------------------------------------------------------")
    print(f"Uq elapsed time: {Uq_time:.4f} ms")  
    print(f"q elapsed time: {q_time:.4f} ms")
    print(f"v elapsed time: {v_time:.4f} ms")
    print(f"total elapsed time: {total_gpu_time:.4f} ms")
    print("---------------------------------------------------------")
    
    job_id = os.getenv("SLURM_JOB_ID", "nojobid")

    row = {
        "mesh": globals.BASE_MESH_NAME,
        "instance": globals.TET_NAME,
        "job_id": job_id,
        "# instances": ix.num_instances,
        "# vertices per instance": bi.v.shape[0],
        "# total vertices": bi.v.shape[0] * ix.num_instances,
        "# modes": globals.N_MODES,
        "precompute time (s)": precompute_time,
        "# frames": globals.NUM_FRAMES,
        "total elapsed (s)": total_elapsed,
        "total FPS": globals.NUM_FRAMES / total_elapsed,
        "total on gpu (ms)": total_gpu_time,
        "Uq (ms)": Uq_time,
        "q (ms)": q_time,
        "v (ms)": v_time,
    }

    # Append only the new row; do not re-read and re-append existing rows.
    row_df = pd.DataFrame([row])
    row_df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)