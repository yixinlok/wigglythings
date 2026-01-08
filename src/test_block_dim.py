import warp as wp
import numpy as np

wp.init()

TILE_THREADS = 8  # Small for demonstration

@wp.kernel
def demonstrate_blocks(output: wp.array(dtype=float)):
    i = wp.tid()
    
    # Each thread computes a value
    x = float(i)
    
    # Tile this value across the block
    # This collects x from ALL threads in this block
    t = wp.tile(x)  # Shape: (1, TILE_THREADS)
    
    # Sum all values in the tile (values from all threads in block)
    # All threads in the block will get the same sum
    sum_val = wp.tile_sum(t)
    
    # Untile to get the scalar back (same for all threads in block)
    result = wp.untile(sum_val)
    
    # Store result
    output[i] = result

# Example 1: 8 threads, block_dim=8 (1 block total)
print("=" * 60)
print("Example 1: dim=8, block_dim=8 (1 block)")
print("=" * 60)
N = 8
output1 = wp.zeros(N, dtype=float)
wp.launch(demonstrate_blocks, dim=[N], inputs=[output1], block_dim=8)
print(f"Thread values: {np.arange(N)}")
print(f"Each thread's sum: {output1.numpy()}")
print(f"Expected: All threads see sum = 0+1+2+3+4+5+6+7 = {sum(range(N))}")
print()

# Example 2: 16 threads, block_dim=8 (2 blocks)
print("=" * 60)
print("Example 2: dim=16, block_dim=8 (2 blocks)")
print("=" * 60)
N = 16
output2 = wp.zeros(N, dtype=float)
wp.launch(demonstrate_blocks, dim=[N], inputs=[output2], block_dim=8)
print(f"Thread values: {np.arange(N)}")
print(f"Each thread's sum: {output2.numpy()}")
print()
print("Block 0 (threads 0-7):  sum = 0+1+2+3+4+5+6+7 = 28")
print("Block 1 (threads 8-15): sum = 8+9+10+11+12+13+14+15 = 92")
print()
print("Notice: Threads in same block see the SAME sum!")
print("        Threads in different blocks see DIFFERENT sums!")
