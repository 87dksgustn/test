import pandas as pd
import numpy as np

old = pd.read_csv('initial_dataset_old.csv')
new = pd.read_csv('initial_dataset.csv')

print('INITIAL DATASET comparison')
print('OLD:', len(old), 'samples, TP=', (old['TP_NoTP']==1).sum(), 'NoTP=', (old['TP_NoTP']==0).sum())
print('NEW:', len(new), 'samples, TP=', (new['TP_NoTP']==1).sum(), 'NoTP=', (new['TP_NoTP']==0).sum())

print()
print('=== C_Barrier_Thx by TP label ===')
for name, df in [('OLD', old), ('NEW', new)]:
    tp_thx = df[df['TP_NoTP']==1]['C_Barrier_Thx']
    notp_thx = df[df['TP_NoTP']==0]['C_Barrier_Thx']
    print(name, 'TP:   mean=%.3f std=%.3f range=[%.2f, %.2f]' % (tp_thx.mean(), tp_thx.std(), tp_thx.min(), tp_thx.max()))
    print(name, 'NoTP: mean=%.3f std=%.3f range=[%.2f, %.2f]' % (notp_thx.mean(), notp_thx.std(), notp_thx.min(), notp_thx.max()))

print()
print('=== Boundary region (TP/NoTP mixed) by barrier_thx ===')
bin_edges = np.linspace(0.25, 2.5, 6)
bin_labels = ['0.25-0.70', '0.70-1.15', '1.15-1.60', '1.60-2.05', '2.05-2.50']

for name, df in [('OLD', old), ('NEW', new)]:
    print()
    print(name, 'dataset:')
    thx = df['C_Barrier_Thx'].values
    labels = df['TP_NoTP'].values
    for i in range(5):
        mask = (thx >= bin_edges[i]) & (thx < bin_edges[i+1])
        total = mask.sum()
        tp_count = ((labels == 1) & mask).sum()
        notp_count = ((labels == 0) & mask).sum()
        if total > 0:
            tp_ratio = tp_count / total
            is_boundary = 0.3 <= tp_ratio <= 0.7
            marker = ' <-- BOUNDARY' if is_boundary else ''
            print('  [%s]: %2d samples (TP:%d, NoTP:%d) ratio=%.2f%s' % (bin_labels[i], total, tp_count, notp_count, tp_ratio, marker))
