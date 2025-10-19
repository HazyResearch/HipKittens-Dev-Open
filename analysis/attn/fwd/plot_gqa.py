import json
import matplotlib.pyplot as plt
import numpy as np


# B = 16, H = 64, HK8, D = 128.
mi355x_baselines_causal = {
    "triton": {
        "1024": 456,
        "2048": 569,
        "4096": 614,
        "8192": 764,
        "16384": 848,
    },
    "ck": {
        "1024": 596.53,
        "2048": 695.71,
        "4096": 799.97,
        "8192": 861.79,
        "16384": 878.86,
    },
    "torch": {
        "1024": 14,
        "2048": 15,
        "4096": 15,
        "8192": "OOM",
        "16384": "OOM",
    }
}

# B = 16, H = 64, HK8, D = 128.
mi350x_baselines_causal = {
    "triton": {
        "1024": 398.220437,
        "2048": 462.552210,
        "4096": 488.052800,
        "8192": 618.134744,
        "16384": 691.502584,
    },
    "ck": {
        "1024": 461,
        "2048": 557,
        "4096": 627,
        "8192": 654,
        "16384": 677,
    },
    "torch": {
        "1024": 12.668953,
        "2048": 13.742200,
        "4096": 14.442294,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 567.01,
        "2048": 782.79,
        "4096": 844.96,
        "8192": 915.63,
        "16384": 968.90,
    }
}

mi355x_baselines_non_causal = {
    "triton": {
        "1024": 844,
        "2048": 945,
        "4096": 996,
        "8192": 1005,
        "16384":1011,
    },
    "ck": {
        "1024": 799,
        "2048": 847,
        "4096": 884,
        "8192": 904,
        "16384": 901,
    },
    "torch": {
        "1024": 29,
        "2048": 31,
        "4096": 34,
        "8192": "OOM",
        "16384": "OOM",
    }
}

mi350x_baselines_non_causal = {
    "triton": {
        "1024": 661.689624,
        "2048": 736.288587,
        "4096": 777.730750,
        "8192": 806.935961,
        "16384":811.201966,
    },
    "ck": {
        "1024": 611,
        "2048": 653,
        "4096": 694,
        "8192": 702,
        "16384": 691,
    },
    "torch": {
        "1024": 26.912733,
        "2048": 29.571504,
        "4096": 31.270997,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 818.35,
        "2048": 967.91,
        "4096": 968.28,
        "8192": 983.25,
        "16384": 1009.18,
    }
}


colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]


for device in ['mi300x', 'mi325x', 'mi350x', 'mi355x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    aiter_tflops = [data[str(size)]['tflops_ref'] for size in matrix_sizes]
    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars0 = ax.bar(x - width, aiter_tflops, width, label='AITER (AMD)', color=colors[1])
    bars1 = ax.bar(x, tk_tflops, width, label='ThunderKittens', color=colors[3])

    max_tflops = max(max(aiter_tflops), max(tk_tflops))

    # Add value labels on bars
    for bar, value in zip(bars0, aiter_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars1, tk_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Sequence Length (N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'Attention Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x, fontsize=16)
    ax.set_yticks(fontsize=16)
    ax.set_xticklabels(matrix_sizes, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_attn_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
    print(f"TK TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

