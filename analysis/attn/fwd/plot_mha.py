import json
import matplotlib.pyplot as plt
import numpy as np


# B = 16, H = 16, D = 128.
mi355x_baselines_causal = {
    "triton": {
        "1024": 371,
        "2048": 508,
        "4096": 573,
        "8192": 733,
        "16384": 845,
    },
    "ck": {
        "1024": 485,
        "2048": 601,
        "4096": 745,
        "8192": 834,
        "16384": 893,
    },
    "torch": {
        "1024": 13,
        "2048": 14,
        "4096": 15,
        "8192": 15,
        "16384": "OOM",
    }
}

# B = 16, H = 16, D = 128.
mi350x_baselines_causal = {
    "triton": {
        "1024": 314.211205,
        "2048": 424.128758,
        "4096": 473.762229,
        "8192": 593.775992,
        "16384": 654.668156,
    },
    "ck": {
        "1024": 412,
        "2048": 504,
        "4096": 600,
        "8192": 663,
        "16384": 700,
    },
    "torch": {
        "1024": 12.70,
        "2048": 13.45,
        "4096": 14.26,
        "8192": 13.84,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 392.17,
        "2048": 698.19,
        "4096": 953.31,
        "8192": 909.30,
        "16384": 952.64,
    }
}

# B = 16, H = 16, D = 128.
mi355x_baselines_non_causal = {
    "triton": {
        "1024": 694,
        "2048": 855,
        "4096": 944,
        "8192": 1001,
        "16384": 1011,
    },
    "ck": {
        "1024": 761,
        "2048": 733,
        "4096": 816,
        "8192": 896,
        "16384": 914,
    },
    "torch": {
        "1024": 29,
        "2048": 32,
        "4096": 34,
        "8192": 33,
        "16384": "OOM",
    }
}

# B = 16, H = 16, D = 128.
mi350x_baselines_non_causal = {
    "triton": {
        "1024": 554.682706,
        "2048": 667.633763,
        "4096": 736.837676,
        "8192": 775.493057,
        "16384": 802.799758,
    },
    "ck": {
        "1024": 591,
        "2048": 603,
        "4096": 665,
        "8192": 700,
        "16384": 715,
    },
    "torch": {
        "1024": 26.98,
        "2048": 29.24,
        "4096": 30.99,
        "8192": 30.72,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 570.79,
        "2048": 896.85,
        "4096": 976.44,
        "8192": 956.59,
        "16384": 974.80,
    }
}


colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]

for device in ['mi350x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract MHA_bkwd_asm_interleaved data
    mha_data = {}
    for key, value in data.items():
        if key.startswith('MHA_bkwd_asm_interleaved_'):
            n_value = value['N']
            mha_data[n_value] = {
                'tflops_tk': value['tflops_tk'],
                'tflops_ref': value['tflops_ref']
            }

    # Sort by N value
    n_values = sorted(mha_data.keys())
    tk_tflops = [mha_data[n]['tflops_tk'] for n in n_values]
    ref_tflops = [mha_data[n]['tflops_ref'] for n in n_values]

    # Create grouped bar chart
    x = np.arange(len(n_values))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, ref_tflops, width, label='AITER (AMD)', color=colors[1])
    bars2 = ax.bar(x, tk_tflops, width, label='HipKittens', color=colors[3])

    max_tflops = max(max(tk_tflops), max(ref_tflops))

    # Add value labels on bars
    for bar, value in zip(bars1, ref_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars2, tk_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Sequence Length (N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'MHA Backwards Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_mha_bkwd_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Sequence lengths tested: {n_values}")
    print(f"AITER (ASM) TFLOPS: {[f'{t:.2f}' for t in ref_tflops]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")
