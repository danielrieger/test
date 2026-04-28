"""Quick diagnostic: what cluster sizes does eman2 produce on the full map?"""
from smlm_score.utility.data_handling import isolate_npcs_from_eman2_boxes, flexible_filter_smlm_data
from smlm_score.utility.input import read_experimental_data
from pathlib import Path

edir = Path(__file__).parent
df = read_experimental_data(edir / 'ShareLoc_Data/data.csv')
coords, vars_, _, _, cuts = flexible_filter_smlm_data(df, filter_type='none', fill_z_value=0.0, return_tree=True)
res = isolate_npcs_from_eman2_boxes(coords, edir / 'info/micrograph_info.json', edir / 'pixel_map.json')

all_info = res.get('all_cluster_info', [])
npc_info = res.get('npc_info', [])
print(f'Total clusters: {len(all_info)}')
print(f'NPC clusters (>=120 pts): {len(npc_info)}')

sizes = sorted([c['n_points'] for c in all_info], reverse=True)
print(f'All cluster sizes: {sizes}')

noise_10_120 = [c for c in all_info if 10 <= c['n_points'] < 120]
print(f'Noise (10-119): {len(noise_10_120)} -> {[c["n_points"] for c in noise_10_120]}')

noise_5_120 = [c for c in all_info if 5 <= c['n_points'] < 120]
print(f'Noise (5-119):  {len(noise_5_120)} -> {[c["n_points"] for c in noise_5_120]}')

noise_any = [c for c in all_info if c['n_points'] < 120]
print(f'Noise (<120):   {len(noise_any)} -> {[c["n_points"] for c in noise_any]}')
