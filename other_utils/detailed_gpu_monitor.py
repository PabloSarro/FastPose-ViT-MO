# Detailed GPU monitoring for memory, context, and process tracking

import subprocess
import time
import json
import sys
import os
import psutil
from datetime import datetime


def get_cuda_processes():
    """Get all processes using CUDA"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            processes = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        processes.append(
                            {
                                "pid": int(parts[0]),
                                "name": parts[1],
                                "gpu_memory": int(parts[2]) if parts[2] != "N/A" else 0,
                            }
                        )
            return processes
    except Exception as e:
        print(f"Error getting CUDA processes: {e}")
    return []


def get_system_memory():
    """Get system memory info"""
    mem = psutil.virtual_memory()
    return {
        "total": mem.total // (1024**3),  # GB
        "available": mem.available // (1024**3),  # GB
        "percent": mem.percent,
    }


def monitor_detailed_gpu(output_file="detailed_gpu_monitoring.json", interval=0.2):
    """
    Detailed GPU monitoring with memory tracking and process analysis
    """

    print(f"Starting detailed GPU monitoring (interval: {interval}s)")
    print(f"Output file: {output_file}")
    print("Press Ctrl+C to stop")

    start_time = time.time()
    monitoring_data = []
    last_gpu_memory = 0
    memory_drop_threshold = 1000  # MB

    try:
        while True:
            current_time = time.time()

            # Get detailed GPU stats
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,memory.free,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    for line in lines:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 11:
                            gpu_id = int(parts[1])
                            gpu_memory_used = int(parts[4]) if parts[4] != "N/A" else 0

                            # Detect memory drops
                            memory_drop = False
                            if gpu_id == 0 and last_gpu_memory > 0:
                                if (last_gpu_memory - gpu_memory_used) > memory_drop_threshold:
                                    memory_drop = True
                                    print(
                                        f"\n*** MEMORY DROP DETECTED: {last_gpu_memory}MB -> {gpu_memory_used}MB (-{last_gpu_memory - gpu_memory_used}MB) ***"
                                    )

                            data_point = {
                                "timestamp": datetime.now().isoformat(),
                                "relative_time": current_time - start_time,
                                "gpu_id": gpu_id,
                                "gpu_util": float(parts[2]) if parts[2] != "N/A" else 0,
                                "mem_util": float(parts[3]) if parts[3] != "N/A" else 0,
                                "mem_used": gpu_memory_used,
                                "mem_total": int(parts[5]) if parts[5] != "N/A" else 0,
                                "mem_free": int(parts[6]) if parts[6] != "N/A" else 0,
                                "temperature": float(parts[7]) if parts[7] != "N/A" else 0,
                                "power_draw": float(parts[8]) if parts[8] != "N/A" else 0,
                                "gpu_clock": float(parts[9]) if parts[9] != "N/A" else 0,
                                "mem_clock": float(parts[10]) if parts[10] != "N/A" else 0,
                                "memory_drop": memory_drop,
                                "cuda_processes": get_cuda_processes(),
                                "system_memory": get_system_memory(),
                            }
                            monitoring_data.append(data_point)

                            # Track memory for next iteration
                            if gpu_id == 0:
                                last_gpu_memory = gpu_memory_used

                            # Print real-time stats for GPU 0
                            if gpu_id == 0:
                                mem_change = ""
                                if memory_drop:
                                    mem_change = " [MEMORY DROP!]"

                                print(
                                    f"\rGPU: {data_point['gpu_util']:5.1f}% | "
                                    f"Mem: {data_point['mem_util']:5.1f}% ({data_point['mem_used']:5d}MB) | "
                                    f"Temp: {data_point['temperature']:4.1f}°C | "
                                    f"Power: {data_point['power_draw']:6.1f}W | "
                                    f"Processes: {len(data_point['cuda_processes'])}{mem_change}",
                                    end="",
                                )

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
                print(f"\nError getting GPU stats: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Collected {len(monitoring_data)} data points")

    # Save data to JSON
    with open(output_file, "w") as f:
        json.dump(monitoring_data, f, indent=2)

    print(f"Data saved to {output_file}")

    # Generate analysis
    analyze_detailed_monitoring(monitoring_data)


def analyze_detailed_monitoring(data):
    """Analyze detailed monitoring data"""

    gpu0_data = [d for d in data if d["gpu_id"] == 0]
    if not gpu0_data:
        return

    print("\n=== DETAILED ANALYSIS ===")

    # Memory drop analysis
    memory_drops = [d for d in gpu0_data if d.get("memory_drop", False)]
    print(f"Memory drops detected: {len(memory_drops)}")

    if memory_drops:
        print("Memory drop times:")
        for i, drop in enumerate(memory_drops[:10]):
            print(f"  {i + 1:2d}. Time {drop['relative_time']:6.1f}s: {drop['mem_used']}MB used")

    # GPU utilization analysis
    utils = [d["gpu_util"] for d in gpu0_data]
    print("\nGPU Utilization:")
    print(f"  Average: {sum(utils) / len(utils):.1f}%")
    print(f"  Min: {min(utils):.1f}%")
    print(f"  Max: {max(utils):.1f}%")

    # Process analysis
    all_processes = []
    for d in gpu0_data:
        for proc in d["cuda_processes"]:
            if proc["name"] not in [p["name"] for p in all_processes]:
                all_processes.append(proc)

    print("\nCUDA Processes detected:")
    for proc in all_processes:
        print(f"  - {proc['name']} (PID: {proc['pid']})")

    # Clock analysis
    gpu_clocks = [d["gpu_clock"] for d in gpu0_data if d["gpu_clock"] > 0]
    if gpu_clocks:
        print(f"\nGPU Clock Range: {min(gpu_clocks):.0f} - {max(gpu_clocks):.0f} MHz")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        if os.path.exists("detailed_gpu_monitoring.json"):
            with open("detailed_gpu_monitoring.json", "r") as f:
                data = json.load(f)
            analyze_detailed_monitoring(data)
        else:
            print("No detailed_gpu_monitoring.json found")
    else:
        monitor_detailed_gpu()
